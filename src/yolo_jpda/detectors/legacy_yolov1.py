"""The original YOLOv1 backend, preserved.

This reproduces the ``YOLO_small`` graph the project was first built around.  It
depends on TensorFlow 1.x — the ``tf.placeholder`` / ``tf.Session`` API was
removed in TensorFlow 2 — and on a ``YOLO_small.ckpt`` checkpoint that is not
distributed with this repository.

Prefer :class:`~yolo_jpda.detectors.ultralytics_yolo.UltralyticsDetector`.  This
module exists so results from the original experiments remain reproducible; it
is not exercised by the test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..types import Detection

__all__ = ["LegacyYolov1Detector", "PASCAL_VOC_CLASSES"]

PASCAL_VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

# Geometry of the YOLOv1 output tensor.
_GRID = 7
_BOXES_PER_CELL = 2
_N_CLASSES = len(PASCAL_VOC_CLASSES)
_INPUT_SIZE = 448

# Layer widths of YOLO_small: (filters, kernel, stride) for convolutions,
# "pool" for max-pooling. Transcribed from the original build_networks().
_ARCHITECTURE = (
    (64, 7, 2), "pool",
    (192, 3, 1), "pool",
    (128, 1, 1), (256, 3, 1), (256, 1, 1), (512, 3, 1), "pool",
    (256, 1, 1), (512, 3, 1), (256, 1, 1), (512, 3, 1),
    (256, 1, 1), (512, 3, 1), (256, 1, 1), (512, 3, 1),
    (512, 1, 1), (1024, 3, 1), "pool",
    (512, 1, 1), (1024, 3, 1), (512, 1, 1), (1024, 3, 1),
    (1024, 3, 1), (1024, 3, 2), (1024, 3, 1), (1024, 3, 1),
)


class LegacyYolov1Detector:
    """YOLOv1 ``YOLO_small`` inference on TensorFlow 1.x.

    Args:
        weights: Path to ``YOLO_small.ckpt``.
        confidence: Minimum class-confidence score.
        iou: Non-maximum-suppression IoU threshold.
        classes: Class names to keep, or ``None`` for all.
        leaky_alpha: Negative slope of the leaky ReLU activations.
    """

    def __init__(
        self,
        weights: Path | str = "model/YOLO_small.ckpt",
        confidence: float = 0.3,
        iou: float = 0.5,
        classes: Optional[Sequence[str]] = ("car", "bus"),
        leaky_alpha: float = 0.1,
    ) -> None:
        self._tf = _import_tensorflow_v1()
        self.confidence = confidence
        self.iou = iou
        self.leaky_alpha = leaky_alpha
        self._wanted = set(classes) if classes is not None else None

        self._input, self._output = self._build_graph()
        self.session = self._tf.Session()
        self.session.run(self._tf.global_variables_initializer())
        self._tf.train.Saver().restore(self.session, str(weights))

    # ------------------------------------------------------------------ graph

    def _build_graph(self):
        tf = self._tf
        net = inputs = tf.placeholder("float32", [None, _INPUT_SIZE, _INPUT_SIZE, 3])

        for spec in _ARCHITECTURE:
            if spec == "pool":
                net = tf.nn.max_pool(
                    net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
                )
            else:
                net = self._conv(net, *spec)

        net = self._dense(net, 512, flatten=True)
        net = self._dense(net, 4096)
        # Dropout is skipped: inference only.
        outputs = self._dense(net, _GRID * _GRID * (_N_CLASSES + 5 * _BOXES_PER_CELL), linear=True)
        return inputs, outputs

    def _conv(self, inputs, filters: int, size: int, stride: int):
        tf = self._tf
        channels = int(inputs.get_shape()[3])
        weight = tf.Variable(tf.truncated_normal([size, size, channels, filters], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[filters]))

        # Explicit symmetric padding then VALID, so the output size does not
        # depend on TensorFlow's SAME-padding rounding.
        pad = size // 2
        padded = tf.pad(inputs, np.array([[0, 0], [pad, pad], [pad, pad], [0, 0]]))
        conv = tf.nn.conv2d(padded, weight, strides=[1, stride, stride, 1], padding="VALID")
        return self._leaky_relu(tf.add(conv, bias))

    def _dense(self, inputs, units: int, flatten: bool = False, linear: bool = False):
        tf = self._tf
        shape = inputs.get_shape().as_list()
        if flatten:
            dim = shape[1] * shape[2] * shape[3]
            # YOLOv1's first dense layer expects channel-major flattening.
            inputs = tf.reshape(tf.transpose(inputs, (0, 3, 1, 2)), [-1, dim])
        else:
            dim = shape[1]

        weight = tf.Variable(tf.truncated_normal([dim, units], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[units]))
        result = tf.add(tf.matmul(inputs, weight), bias)
        return result if linear else self._leaky_relu(result)

    def _leaky_relu(self, x):
        return self._tf.maximum(self.leaky_alpha * x, x)

    # -------------------------------------------------------------- inference

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in one BGR image."""
        height, width = frame.shape[:2]
        resized = _resize(frame, _INPUT_SIZE)
        # Normalise to [-1, 1], as the original weights expect.
        batch = ((resized / 255.0) * 2.0 - 1.0).astype("float32")[None, ...]

        raw = self.session.run(self._output, feed_dict={self._input: batch})[0]
        return self._interpret(raw, width, height)

    def _interpret(self, raw: np.ndarray, width: int, height: int) -> list[Detection]:
        """Decode the flat 1470-vector into detections in image coordinates."""
        split_a = _GRID * _GRID * _N_CLASSES
        split_b = split_a + _GRID * _GRID * _BOXES_PER_CELL

        class_probs = raw[:split_a].reshape(_GRID, _GRID, _N_CLASSES)
        confidences = raw[split_a:split_b].reshape(_GRID, _GRID, _BOXES_PER_CELL)
        boxes = raw[split_b:].reshape(_GRID, _GRID, _BOXES_PER_CELL, 4).copy()

        # Box centres are predicted relative to their grid cell.
        offset = np.tile(np.arange(_GRID)[None, :, None], (_GRID, 1, _BOXES_PER_CELL))
        boxes[..., 0] = (boxes[..., 0] + offset) / _GRID * width
        boxes[..., 1] = (boxes[..., 1] + offset.transpose(1, 0, 2)) / _GRID * height
        # Width and height are predicted as square roots.
        boxes[..., 2] = boxes[..., 2] ** 2 * width
        boxes[..., 3] = boxes[..., 3] ** 2 * height

        # score[i, j, b, c] = P(class c) * P(object) for box b of cell (i, j).
        scores = class_probs[:, :, None, :] * confidences[..., None]

        keep = scores >= self.confidence
        cells = np.nonzero(keep)
        if cells[0].size == 0:
            return []

        kept_boxes = boxes[cells[0], cells[1], cells[2]]
        kept_scores = scores[keep]
        kept_classes = cells[3]

        order = np.argsort(kept_scores)[::-1]
        kept_boxes, kept_scores, kept_classes = (
            kept_boxes[order],
            kept_scores[order],
            kept_classes[order],
        )
        kept_boxes, kept_scores, kept_classes = _suppress_overlaps(
            kept_boxes, kept_scores, kept_classes, self.iou
        )

        detections = []
        for box, score, class_id in zip(kept_boxes, kept_scores, kept_classes):
            label = PASCAL_VOC_CLASSES[int(class_id)]
            if self._wanted is not None and label not in self._wanted:
                continue
            detections.append(
                Detection(float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                          float(score), label)
            )
        return detections

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "LegacyYolov1Detector":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _suppress_overlaps(boxes, scores, classes, iou_threshold: float):
    """Greedy non-maximum suppression over score-sorted boxes."""
    keep = []
    for i in range(len(boxes)):
        if any(_iou(boxes[i], boxes[j]) > iou_threshold for j in keep):
            continue
        keep.append(i)
    return boxes[keep], scores[keep], classes[keep]


def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Intersection over union of two ``(cx, cy, w, h)`` boxes."""
    overlap_x = min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2) - max(
        box1[0] - box1[2] / 2, box2[0] - box2[2] / 2
    )
    overlap_y = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2) - max(
        box1[1] - box1[3] / 2, box2[1] - box2[3] / 2
    )
    if overlap_x <= 0 or overlap_y <= 0:
        return 0.0
    intersection = overlap_x * overlap_y
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
    return float(intersection / union) if union > 0 else 0.0


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(frame, (size, size))


def _import_tensorflow_v1():
    """Return a TF1-compatible API surface, or explain what is missing."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "LegacyYolov1Detector requires TensorFlow 1.x. This backend is kept "
            "only for reproducing the original results; prefer "
            "UltralyticsDetector for new work."
        ) from exc

    if tf.__version__.startswith("1."):
        return tf

    # TF2 still exposes the graph-mode API under compat.v1.
    compat = tf.compat.v1
    compat.disable_eager_execution()
    return compat
