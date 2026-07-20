"""Kalman prediction, validation gating and the PDA measurement update.

These are the per-track pieces of the filter.  The *joint* part — deciding how
much each measurement belongs to each track — lives in
:mod:`yolo_jpda.tracking.association`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["Prediction", "GateResult", "predict", "gate", "pda_update"]


class Prediction(NamedTuple):
    """Output of the prediction step, reused by gating and the update."""

    state: np.ndarray  # (4,)   predicted state
    covariance: np.ndarray  # (4, 4) predicted state covariance
    innovation_cov: np.ndarray  # (2, 2) S = H P H' + R
    kalman_gain: np.ndarray  # (4, 2) K = P H' S^-1


class GateResult(NamedTuple):
    """Measurements accepted by the validation gate, plus their likelihoods."""

    indices: np.ndarray
    """Indices into the measurement array, ordered by increasing Mahalanobis
    distance.  Empty if nothing gated."""

    likelihoods: np.ndarray
    """Length ``1 + len(indices)``.  Element 0 is the *miss* likelihood
    ``(1 - P_D) * beta``; element ``j + 1`` is ``P_D * N(z_j; H x, S)``.

    These are unnormalised on purpose — normalising per track would collapse
    JPDA into independent PDA, which is exactly the bug this rewrite fixes.
    """


def predict(
    state: np.ndarray,
    covariance: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> Prediction:
    """Advance one step and precompute the innovation covariance and gain."""
    predicted_state = F @ state
    predicted_cov = F @ covariance @ F.T + Q

    innovation_cov = H @ predicted_cov @ H.T + R
    # Symmetrise: repeated propagation otherwise accumulates asymmetry that can
    # push the Cholesky factorisation below into a numerical failure.
    innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)

    kalman_gain = predicted_cov @ H.T @ np.linalg.inv(innovation_cov)
    return Prediction(predicted_state, predicted_cov, innovation_cov, kalman_gain)


def gate(
    prediction: Prediction,
    measurements: np.ndarray,
    H: np.ndarray,
    gate_threshold: float,
    detection_probability: float,
    clutter_density: float,
) -> GateResult:
    """Select measurements inside the validation gate and score them.

    ``measurements`` is ``(n, 2)``.  A measurement is accepted when its
    Mahalanobis distance from the prediction is below ``gate_threshold``.
    """
    miss_likelihood = (1.0 - detection_probability) * clutter_density

    if measurements.size == 0:
        return GateResult(np.empty(0, dtype=int), np.array([miss_likelihood]))

    residuals = measurements - (H @ prediction.state)
    # Solve rather than invert: cheaper and better conditioned.
    weighted = np.linalg.solve(prediction.innovation_cov, residuals.T).T
    squared_distances = np.einsum("ij,ij->i", residuals, weighted)
    # Tiny negatives can appear from round-off on near-singular S.
    squared_distances = np.maximum(squared_distances, 0.0)

    inside = np.flatnonzero(squared_distances < gate_threshold**2)
    if inside.size == 0:
        return GateResult(np.empty(0, dtype=int), np.array([miss_likelihood]))

    # Sort by distance so the nearest measurement is always index 0 — several
    # downstream heuristics rely on that ordering.
    inside = inside[np.argsort(squared_distances[inside])]

    dim = measurements.shape[1]
    normaliser = np.sqrt((2.0 * np.pi) ** dim * np.linalg.det(prediction.innovation_cov))
    densities = np.exp(-0.5 * squared_distances[inside]) / normaliser

    likelihoods = np.concatenate([[miss_likelihood], detection_probability * densities])
    return GateResult(inside, likelihoods)


def pda_update(
    prediction: Prediction,
    measurements: np.ndarray,
    gate_result: GateResult,
    association_probs: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse gated measurements using their marginal association probabilities.

    ``association_probs`` is the normalised counterpart of
    ``gate_result.likelihoods``: element 0 is the probability that the target
    was missed, the rest correspond to ``gate_result.indices``.

    Returns the updated ``(state, covariance)``.  This is the standard PDAF
    update, including the "spread of the innovations" term that widens the
    covariance when the association is ambiguous.
    """
    miss_prob = float(association_probs[0])

    if gate_result.indices.size == 0:
        # Nothing to fuse — coast on the prediction.
        return prediction.state.copy(), prediction.covariance.copy()

    hit_probs = association_probs[1:]
    residuals = measurements[gate_result.indices] - (H @ prediction.state)

    combined_residual = hit_probs @ residuals
    updated_state = prediction.state + prediction.kalman_gain @ combined_residual

    # Covariance of a correctly associated update.
    correct_cov = (
        prediction.covariance
        - prediction.kalman_gain @ prediction.innovation_cov @ prediction.kalman_gain.T
    )

    # Spread of the innovations: extra uncertainty from not knowing which
    # measurement was the right one.
    weighted_scatter = (residuals * hit_probs[:, None]).T @ residuals
    scatter = weighted_scatter - np.outer(combined_residual, combined_residual)
    spread = prediction.kalman_gain @ scatter @ prediction.kalman_gain.T

    updated_cov = miss_prob * prediction.covariance + (1.0 - miss_prob) * correct_cov + spread
    updated_cov = 0.5 * (updated_cov + updated_cov.T)
    return updated_state, updated_cov
