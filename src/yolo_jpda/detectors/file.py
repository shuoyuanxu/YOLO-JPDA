"""Detectors that replay detections from disk.

Useful for two things: benchmarking the tracker against MOT-challenge
sequences, and running the whole pipeline with no deep-learning dependencies
installed at all.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..types import Detection

__all__ = ["FileDetector", "load_mot_detections"]


class FileDetector:
    """Replays precomputed detections, one frame at a time.

    Satisfies the :class:`~yolo_jpda.detectors.base.Detector` protocol, so it
    is a drop-in substitute for a real detector.  ``detect`` ignores the frame
    it is handed and returns the next stored set of detections.

    The replayed sequence spans the full frame-number range in the file, not
    just the frames that have entries.  A detection file omits frames where the
    detector found nothing, so walking only the present keys would pull later
    detections forward into earlier frames and desynchronise everything after
    the first empty frame — which real sequences always contain.
    """

    def __init__(self, detections_by_frame: dict[int, list[Detection]]) -> None:
        self._by_frame = detections_by_frame
        if detections_by_frame:
            first, last = min(detections_by_frame), max(detections_by_frame)
            self._frames = list(range(first, last + 1))
        else:
            self._frames = []
        self._cursor = 0

    @classmethod
    def from_mot(cls, path: Path | str, score_threshold: float = 0.0) -> "FileDetector":
        return cls(load_mot_detections(path, score_threshold))

    def detect(self, frame: Optional[np.ndarray] = None) -> Sequence[Detection]:
        """Return the next frame's detections, or an empty list once exhausted."""
        if self._cursor >= len(self._frames):
            return []
        detections = self._by_frame.get(self._frames[self._cursor], [])
        self._cursor += 1
        return detections

    def reset(self) -> None:
        self._cursor = 0

    def __len__(self) -> int:
        """Number of frames spanned, including those with no detections."""
        return len(self._frames)


def load_mot_detections(
    path: Path | str, score_threshold: float = 0.0
) -> dict[int, list[Detection]]:
    """Read a MOT-challenge ``det.txt``.

    Columns are ``frame, id, left, top, width, height, score, ...`` — the
    trailing 3-D world coordinates are ignored.  Frames with no detections are
    absent from the file and so absent from the result; callers iterating over a
    frame range should treat a missing key as an empty list.
    """
    rows = np.loadtxt(str(path), delimiter=",", ndmin=2)
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for row in rows:
        score = float(row[6]) if len(row) > 6 else 1.0
        if score < score_threshold:
            continue
        by_frame[int(row[0])].append(
            Detection.from_tlwh(float(row[2]), float(row[3]), float(row[4]), float(row[5]), score)
        )
    return dict(by_frame)
