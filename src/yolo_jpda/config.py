"""Motion model and tracker configuration.

Both dataclasses are frozen: a configuration is a value, not something the
tracker mutates while running.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

__all__ = ["MotionModel", "TrackerConfig"]


@dataclass(frozen=True)
class MotionModel:
    """A 2-D constant-velocity (white-noise-acceleration) model.

    The state is ``[x, vx, y, vy]`` and the measurement is ``[x, y]``.  Both the
    transition and process-noise matrices are block-diagonal across the two
    axes, which are assumed independent.
    """

    dt: float = 1.0
    """Time between frames, in the same units as the velocity estimates."""

    process_noise: float = 0.1
    """Acceleration noise intensity ``q``.  Larger values track manoeuvres more
    aggressively at the cost of a noisier estimate."""

    measurement_noise_var: float = 7.0
    """Variance of the detector's centre-position noise, in pixels squared.

    Note: the original scripts called this a standard deviation but used it
    directly as a variance.  The name here reflects how it is actually used, so
    the numerics are unchanged from the tuned legacy values.
    """

    def transition_matrix(self) -> np.ndarray:
        """``F`` — advances position by ``velocity * dt``."""
        t = self.dt
        return np.array(
            [
                [1.0, t, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, t],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def process_noise_covariance(self) -> np.ndarray:
        """``Q`` — the discrete white-noise-acceleration covariance."""
        t, q = self.dt, self.process_noise
        block = np.array([[t**4 / 4.0, t**3 / 2.0], [t**3 / 2.0, t**2]]) * q
        cov = np.zeros((4, 4))
        cov[np.ix_([0, 1], [0, 1])] = block
        cov[np.ix_([2, 3], [2, 3])] = block
        return cov

    def measurement_matrix(self) -> np.ndarray:
        """``H`` — extracts ``[x, y]`` from the state."""
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    def measurement_covariance(self) -> np.ndarray:
        """``R`` — isotropic measurement noise."""
        return self.measurement_noise_var * np.eye(2)

    def initial_covariance(self, velocity_var: float = 1.0) -> np.ndarray:
        """``P0`` for a freshly initiated track.

        Position uncertainty starts at the measurement noise; velocity is
        unobserved on a single frame, so its variance is supplied separately.
        See :meth:`unknown_velocity_var`.
        """
        return np.diag(
            [self.measurement_noise_var, velocity_var, self.measurement_noise_var, velocity_var]
        )

    @staticmethod
    def unknown_velocity_var(max_speed: float) -> float:
        """Velocity variance for a track whose speed is entirely unknown.

        A track initiated from a single detection has no velocity information at
        all, only the prior that the target moves no faster than ``max_speed``.
        Modelling that as a uniform distribution over ``[-max_speed, max_speed]``
        gives variance ``max_speed**2 / 3``.

        Encoding it as a small value instead — pretending the target is roughly
        stationary — makes the first prediction confidently wrong, and the
        resulting gate is too narrow to contain a fast target's next detection.
        The track then dies and re-initiates every frame.
        """
        return max_speed**2 / 3.0


@dataclass(frozen=True)
class TrackerConfig:
    """Detection pruning, gating, association and track-lifecycle parameters."""

    score_threshold: float = 0.15
    """Detections scoring below this are discarded before tracking."""

    gate_threshold: float = 30.0**0.5
    """Validation-gate radius as a Mahalanobis *distance* (not squared).

    Use :meth:`gate_from_confidence` to derive this from a chi-squared quantile
    rather than picking it by hand.
    """

    detection_probability: float = 0.95
    """``P_D`` — probability that an existing target produces a detection."""

    clutter_density: float = 3.0 / (448 * 448)
    """``beta`` — spatial density of false detections, assumed Poisson.

    Use :meth:`clutter_from_image` to derive this from an expected false-alarm
    count and image size.
    """

    max_misses: int = 45
    """Consecutive frames a track may go unassociated before it is terminated."""

    min_hits: int = 3
    """Associations required before a tentative track is reported as confirmed."""

    max_speed: float = 7.0
    """Speed cap, in pixels per step, applied when seeding velocities from the
    first two frames.  Guards against absurd velocities from bad pairings."""

    max_exact_hypotheses: int = 100_000
    """Clusters whose joint-hypothesis count stays under this are solved by exact
    enumeration; larger ones fall back to belief propagation."""

    bp_iterations: int = 50
    """Maximum belief-propagation sweeps for clusters solved approximately."""

    bp_tolerance: float = 1e-6
    """Belief-propagation convergence threshold on the message update."""

    @staticmethod
    def gate_from_confidence(confidence: float = 0.99, dof: int = 2) -> float:
        """Gate radius admitting ``confidence`` of true measurements.

        >>> round(TrackerConfig.gate_from_confidence(0.99), 3)
        3.035
        """
        return float(np.sqrt(chi2.ppf(confidence, dof)))

    @staticmethod
    def clutter_from_image(
        expected_false_alarms: float, image_width: int, image_height: int
    ) -> float:
        """Clutter density from an expected false-alarm count per frame."""
        return expected_false_alarms / float(image_width * image_height)
