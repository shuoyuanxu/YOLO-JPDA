"""Drawing tracks and detections onto frames."""

from __future__ import annotations

import colorsys
from typing import Iterable, Optional

import numpy as np

from .types import Detection, Track

__all__ = ["color_for_id", "draw_tracks", "draw_detections", "stack_side_by_side"]

_DETECTION_COLOR = (0, 0, 255)  # BGR red


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """A stable, well-separated BGR colour for a track id.

    Hues are spread by the golden-angle so consecutive ids look distinct, and
    the mapping is deterministic — the same track keeps its colour across runs,
    unlike the original's ``random.uniform`` palette which reshuffled every
    frame.
    """
    hue = (track_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(blue * 255), int(green * 255), int(red * 255))


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[Track],
    show_ids: bool = True,
    show_velocity: bool = False,
    thickness: int = 2,
) -> np.ndarray:
    """Draw track boxes onto a copy of ``frame`` and return it."""
    import cv2

    canvas = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(round(v)) for v in track.to_xyxy())
        color = color_for_id(track.track_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        if show_ids:
            label = str(track.track_id)
            # Keep the label on-screen when the box touches the top edge.
            baseline = max(y1 - 6, 14)
            cv2.putText(
                canvas, label, (x1 + 3, baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

        if show_velocity:
            cx, cy = track.position
            vx, vy = track.velocity
            cv2.arrowedLine(
                canvas,
                (int(cx), int(cy)),
                (int(cx + vx * 5), int(cy + vy * 5)),
                color, 2, cv2.LINE_AA, tipLength=0.3,
            )
    return canvas


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    show_scores: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Draw raw detection boxes onto a copy of ``frame`` and return it."""
    import cv2

    canvas = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection.to_xyxy())
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _DETECTION_COLOR, thickness)
        if show_scores:
            cv2.putText(
                canvas, f"{detection.score:.2f}", (x1 + 3, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _DETECTION_COLOR, 1, cv2.LINE_AA,
            )
    return canvas


def stack_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    left_label: Optional[str] = None,
    right_label: Optional[str] = None,
) -> np.ndarray:
    """Join two frames horizontally for before/after comparison videos.

    Frames of differing heights are padded rather than stretched, so neither
    side's aspect ratio is distorted.
    """
    import cv2

    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right

    for label, x_offset in ((left_label, 0), (right_label, left.shape[1])):
        if label:
            cv2.putText(
                canvas, label, (x_offset + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
            )
    return canvas
