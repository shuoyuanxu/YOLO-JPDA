"""Modern YOLO backend, built on the ``ultralytics`` package.

This is the recommended detector.  ``ultralytics`` is an optional dependency —
install it with ``pip install 'yolo-jpda[yolo]'`` — so the import is deferred
until the class is actually constructed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..types import Detection

__all__ = ["UltralyticsDetector"]

# Classes worth tracking in a road scene. ``None`` keeps everything.
VEHICLE_CLASSES = ("car", "bus", "truck", "motorcycle", "bicycle")


class UltralyticsDetector:
    """Wraps an ultralytics YOLO model behind the ``Detector`` protocol.

    Args:
        weights: Model name or path.  Plain names such as ``"yolov8n.pt"`` are
            downloaded on first use; pass a path under ``model/`` to pin a
            local checkpoint.
        confidence: Minimum detection confidence.
        iou: Non-maximum-suppression IoU threshold.
        classes: Class names to keep, or ``None`` for all.
        device: ``"cpu"``, ``"cuda"``, a device index, or ``None`` to let
            ultralytics choose.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        classes: Optional[Sequence[str]] = VEHICLE_CLASSES,
        device: Optional[str] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "UltralyticsDetector requires the 'ultralytics' package. "
                "Install it with: pip install 'yolo-jpda[yolo]'"
            ) from exc

        self.model = YOLO(weights)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self._wanted = set(classes) if classes is not None else None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in one BGR image."""
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        names = self.model.names
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for xyxy, score, class_id in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            ):
                label = names[int(class_id)]
                if self._wanted is not None and label not in self._wanted:
                    continue
                detections.append(
                    Detection.from_xyxy(*xyxy, score=float(score), label=label)
                )
        return detections
