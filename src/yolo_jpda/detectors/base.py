"""The detector interface the tracker programs against."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..types import Detection

__all__ = ["Detector"]


@runtime_checkable
class Detector(Protocol):
    """Anything that turns a frame into detections.

    Keeping this a ``Protocol`` rather than a base class means a detector need
    only provide ``detect`` — no inheritance, and third-party detectors can be
    adapted with a small wrapper.  The tracker never imports a concrete
    detector, so the heavy deep-learning dependencies stay optional.
    """

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        """Return the detections found in a single BGR image."""
        ...
