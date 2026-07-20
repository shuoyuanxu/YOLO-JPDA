"""Joint Probabilistic Data Association.

Given, for each track, the set of measurements that fell inside its validation
gate and the corresponding unnormalised likelihoods, this module computes the
*marginal* association probabilities under the JPDA feasibility constraint:

    a measurement originates from at most one target, and
    a target produces at most one measurement.

That coupling is what distinguishes JPDA from running independent PDA filters.
Normalising each track's likelihoods on its own — which the original code did —
silently discards the constraint and lets two tracks both claim the same
detection at full weight, the classic cause of track coalescence.

Three ideas keep the exact computation tractable:

1. **Clustering.**  Tracks and measurements form a bipartite graph.  Distinct
   connected components are independent problems, so each is solved separately.
   Most frames decompose into many tiny clusters.
2. **Exact enumeration** within a cluster, when the number of joint hypotheses
   is small enough to enumerate.
3. **Belief propagation** as the fallback for large clusters, following
   Williams & Lau (2014).  It converges to near-exact marginals on the sparse,
   locally tree-like graphs that gating produces, in time linear in the number
   of track/measurement pairs rather than exponential in cluster size.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

__all__ = ["joint_association_probabilities"]


def joint_association_probabilities(
    gated_indices: Sequence[np.ndarray],
    likelihoods: Sequence[np.ndarray],
    n_measurements: int,
    max_exact_hypotheses: int = 100_000,
    bp_iterations: int = 50,
    bp_tolerance: float = 1e-6,
) -> list[np.ndarray]:
    """Marginal association probabilities for every track.

    Args:
        gated_indices: For each track, the measurement indices inside its gate.
        likelihoods: For each track, an array of length ``1 + len(gated)``.
            Element 0 is the miss likelihood; the rest align with
            ``gated_indices``.  Must be unnormalised.
        n_measurements: Total measurements in the frame.
        max_exact_hypotheses: Enumerate exactly while a cluster's hypothesis
            count stays at or below this; otherwise use belief propagation.
        bp_iterations: Maximum belief-propagation sweeps.
        bp_tolerance: Belief-propagation convergence threshold.

    Returns:
        One probability vector per track, each summing to 1 and matching the
        layout of the corresponding ``likelihoods`` entry.
    """
    n_tracks = len(gated_indices)
    results: list[np.ndarray] = [np.array([1.0]) for _ in range(n_tracks)]
    if n_tracks == 0:
        return results

    for cluster in _find_clusters(gated_indices, n_tracks, n_measurements):
        cluster_gated = [gated_indices[i] for i in cluster]
        cluster_likelihoods = [likelihoods[i] for i in cluster]

        hypothesis_count = _hypothesis_count(cluster_likelihoods, max_exact_hypotheses)
        if hypothesis_count <= max_exact_hypotheses:
            marginals = _exact_marginals(cluster_gated, cluster_likelihoods)
        else:
            marginals = _belief_propagation_marginals(
                cluster_gated, cluster_likelihoods, bp_iterations, bp_tolerance
            )

        for local_index, track_index in enumerate(cluster):
            results[track_index] = marginals[local_index]

    return results


def _find_clusters(
    gated_indices: Sequence[np.ndarray], n_tracks: int, n_measurements: int
) -> list[list[int]]:
    """Group tracks into independent sub-problems.

    Two tracks share a cluster when they compete for a measurement, directly or
    through a chain of other tracks.  Tracks that gated nothing are returned as
    singletons — they are trivially independent.
    """
    rows: list[int] = []
    cols: list[int] = []
    for track_index, measurement_indices in enumerate(gated_indices):
        for measurement_index in measurement_indices:
            rows.append(track_index)
            cols.append(n_tracks + int(measurement_index))

    if not rows:
        return [[i] for i in range(n_tracks)]

    size = n_tracks + n_measurements
    adjacency = coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(size, size)
    )
    _, labels = connected_components(adjacency, directed=False)

    clusters: dict[int, list[int]] = {}
    for track_index in range(n_tracks):
        clusters.setdefault(labels[track_index], []).append(track_index)
    return list(clusters.values())


def _hypothesis_count(likelihoods: Sequence[np.ndarray], ceiling: int) -> int:
    """Size of the unconstrained hypothesis space, saturating at ``ceiling``.

    Saturating keeps the product from overflowing on pathological clusters; the
    caller only needs to know whether the ceiling was exceeded.
    """
    total = 1
    for likelihood in likelihoods:
        total *= len(likelihood)
        if total > ceiling:
            return ceiling + 1
    return total


def _exact_marginals(
    gated_indices: Sequence[np.ndarray], likelihoods: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Enumerate every feasible joint association event in a cluster.

    Depth-first over tracks, keeping a set of already-claimed measurements so
    infeasible branches are pruned as early as possible rather than generated
    and filtered afterwards.
    """
    n_tracks = len(gated_indices)
    marginals = [np.zeros(len(likelihood)) for likelihood in likelihoods]
    claimed: set[int] = set()
    choices = np.zeros(n_tracks, dtype=int)

    def recurse(track_index: int, weight: float) -> None:
        if track_index == n_tracks:
            for i, choice in enumerate(choices):
                marginals[i][choice] += weight
            return

        # Option 0: this target was missed. Always feasible.
        choices[track_index] = 0
        recurse(track_index + 1, weight * likelihoods[track_index][0])

        for slot, measurement_index in enumerate(gated_indices[track_index], start=1):
            measurement_index = int(measurement_index)
            if measurement_index in claimed:
                continue
            claimed.add(measurement_index)
            choices[track_index] = slot
            recurse(track_index + 1, weight * likelihoods[track_index][slot])
            claimed.discard(measurement_index)

    recurse(0, 1.0)
    return [_normalise(m) for m in marginals]


def _belief_propagation_marginals(
    gated_indices: Sequence[np.ndarray],
    likelihoods: Sequence[np.ndarray],
    max_iterations: int,
    tolerance: float,
) -> list[np.ndarray]:
    """Approximate marginals by loopy belief propagation.

    Messages are passed on the bipartite track/measurement graph until the
    measurement-to-track messages stop moving.  For a cluster with ``T`` tracks
    each gating ``m`` measurements this costs ``O(T * m)`` per sweep, against
    the ``O(m^T)`` of exact enumeration.
    """
    n_tracks = len(gated_indices)

    # Re-index the cluster's measurements to a dense local range.
    local_ids: dict[int, int] = {}
    for measurement_indices in gated_indices:
        for measurement_index in measurement_indices:
            local_ids.setdefault(int(measurement_index), len(local_ids))
    n_local = len(local_ids)

    if n_local == 0:
        return [_normalise(likelihood.copy()) for likelihood in likelihoods]

    # weights[i][slot] is the likelihood of track i taking its slot-th gated
    # measurement; targets[i][slot] is that measurement's local id.
    weights = [likelihood[1:] for likelihood in likelihoods]
    miss_weights = np.array([likelihood[0] for likelihood in likelihoods])
    targets = [
        np.array([local_ids[int(m)] for m in measurement_indices], dtype=int)
        for measurement_indices in gated_indices
    ]

    # nu[i][slot]: message from a measurement back to track i. Start neutral.
    nu = [np.ones(len(w)) for w in weights]

    for _ in range(max_iterations):
        # Track -> measurement messages.
        mu = []
        for i in range(n_tracks):
            weighted = weights[i] * nu[i]
            total = miss_weights[i] + weighted.sum()
            # Each slot's message excludes its own contribution to the sum.
            # The floor guards the degenerate P_D == 1 case, where a track with
            # a single gated measurement would otherwise divide by zero.
            denominator = np.maximum(total - weighted, np.finfo(float).tiny)
            mu.append(weights[i] / denominator)

        # Aggregate incoming messages per measurement, then subtract the
        # sender's own contribution to get the outgoing message.
        incoming = np.zeros(n_local)
        for i in range(n_tracks):
            np.add.at(incoming, targets[i], mu[i])

        max_delta = 0.0
        for i in range(n_tracks):
            others = incoming[targets[i]] - mu[i]
            updated = 1.0 / (1.0 + others)
            max_delta = max(max_delta, float(np.max(np.abs(updated - nu[i]), initial=0.0)))
            nu[i] = updated

        if max_delta < tolerance:
            break

    marginals = []
    for i in range(n_tracks):
        unnormalised = np.concatenate([[miss_weights[i]], weights[i] * nu[i]])
        marginals.append(_normalise(unnormalised))
    return marginals


def _normalise(values: np.ndarray) -> np.ndarray:
    """Scale to sum to 1, falling back to "certainly missed" if all mass is 0.

    An all-zero vector means every hypothesis underflowed, which happens when a
    track's gated measurements are all far out in the tails.  Treating that as a
    miss is the conservative reading and keeps the filter from producing NaNs.
    """
    total = values.sum()
    if not np.isfinite(total) or total <= 0.0:
        fallback = np.zeros_like(values)
        fallback[0] = 1.0
        return fallback
    return values / total
