"""Core data types shared by detectors, the tracker and the I/O layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["Detection", "Track", "TrackStatus"]


@dataclass(frozen=True)
class Detection:
    """A single object detection in image coordinates.

    The tracker only consumes the centre ``(cx, cy)``; ``width``/``height`` are
    carried along so that the estimated box can be drawn at a plausible size.
    """

    cx: float
    cy: float
    width: float
    height: float
    score: float = 1.0
    label: Optional[str] = None

    @classmethod
    def from_tlwh(
        cls,
        left: float,
        top: float,
        width: float,
        height: float,
        score: float = 1.0,
        label: Optional[str] = None,
    ) -> "Detection":
        """Build from the MOT-challenge ``(left, top, width, height)`` convention."""
        return cls(left + width / 2.0, top + height / 2.0, width, height, score, label)

    @classmethod
    def from_xyxy(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        score: float = 1.0,
        label: Optional[str] = None,
    ) -> "Detection":
        """Build from corner coordinates, as returned by most modern detectors."""
        return cls((x1 + x2) / 2.0, (y1 + y2) / 2.0, abs(x2 - x1), abs(y2 - y1), score, label)

    @property
    def position(self) -> np.ndarray:
        """Measurement vector ``[cx, cy]`` consumed by the Kalman filter."""
        return np.array([self.cx, self.cy], dtype=float)

    def to_tlwh(self) -> tuple[float, float, float, float]:
        return (self.cx - self.width / 2.0, self.cy - self.height / 2.0, self.width, self.height)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.cx - self.width / 2.0,
            self.cy - self.height / 2.0,
            self.cx + self.width / 2.0,
            self.cy + self.height / 2.0,
        )


class TrackStatus:
    """Lifecycle states a track can be in."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    TERMINATED = "terminated"


@dataclass
class Track:
    """A tracked object.

    The state vector is ``[x, vx, y, vy]`` — position/velocity interleaved per
    axis, matching the block-diagonal constant-velocity model in
    :mod:`yolo_jpda.config`.
    """

    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    width: float
    height: float
    status: str = TrackStatus.TENTATIVE
    age: int = 0
    hits: int = 0
    consecutive_misses: int = 0
    last_association_prob: float = 0.0
    """Probability the track was associated with *some* detection this frame."""

    @property
    def position(self) -> np.ndarray:
        """Estimated centre ``[x, y]``."""
        return self.state[[0, 2]]

    @property
    def velocity(self) -> np.ndarray:
        """Estimated velocity ``[vx, vy]`` in pixels per time step."""
        return self.state[[1, 3]]

    def to_tlwh(self) -> tuple[float, float, float, float]:
        x, y = self.position
        return (x - self.width / 2.0, y - self.height / 2.0, self.width, self.height)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        x, y = self.position
        return (
            x - self.width / 2.0,
            y - self.height / 2.0,
            x + self.width / 2.0,
            y + self.height / 2.0,
        )

    def copy(self) -> "Track":
        """An independent snapshot, safe to keep after the tracker moves on.

        The state and covariance arrays are copied, so callers can accumulate
        per-frame history without every entry aliasing the live track.
        """
        return Track(
            track_id=self.track_id,
            state=self.state.copy(),
            covariance=self.covariance.copy(),
            width=self.width,
            height=self.height,
            status=self.status,
            age=self.age,
            hits=self.hits,
            consecutive_misses=self.consecutive_misses,
            last_association_prob=self.last_association_prob,
        )
