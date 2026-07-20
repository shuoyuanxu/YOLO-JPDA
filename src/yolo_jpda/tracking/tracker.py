"""The multi-target JPDA tracker."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from ..config import MotionModel, TrackerConfig
from ..types import Detection, Track, TrackStatus
from .association import joint_association_probabilities
from .kalman import gate, pda_update, predict

__all__ = ["JPDATracker"]


class JPDATracker:
    """Tracks multiple targets by joint probabilistic data association.

    Each call to :meth:`update` consumes one frame of detections and advances
    every track by one time step:

    1. predict each track forward and gate the detections against it;
    2. solve the joint association problem across all tracks at once;
    3. fuse each track's gated detections weighted by their marginal
       probabilities;
    4. terminate stale tracks and initiate new ones from unclaimed detections.

    Example:
        >>> tracker = JPDATracker()
        >>> tracks = tracker.update([Detection(10.0, 10.0, 4.0, 4.0, 0.9)])
        >>> len(tracks)
        1
    """

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        motion: Optional[MotionModel] = None,
    ) -> None:
        self.config = config or TrackerConfig()
        self.motion = motion or MotionModel()

        # Cached because they are constant across frames.
        self._F = self.motion.transition_matrix()
        self._Q = self.motion.process_noise_covariance()
        self._H = self.motion.measurement_matrix()
        self._R = self.motion.measurement_covariance()

        self.tracks: list[Track] = []
        self.frame_index = 0
        self._next_track_id = 1

    # ------------------------------------------------------------------ API

    def update(self, detections: Iterable[Detection]) -> list[Track]:
        """Advance one frame and return the currently live tracks."""
        self.frame_index += 1

        kept = [d for d in detections if d.score >= self.config.score_threshold]
        measurements = (
            np.array([d.position for d in kept], dtype=float)
            if kept
            else np.zeros((0, 2), dtype=float)
        )

        predictions = []
        gate_results = []
        for track in self.tracks:
            prediction = predict(
                track.state, track.covariance, self._F, self._Q, self._H, self._R
            )
            predictions.append(prediction)
            gate_results.append(
                gate(
                    prediction,
                    measurements,
                    self._H,
                    self.config.gate_threshold,
                    self.config.detection_probability,
                    self.config.clutter_density,
                )
            )

        association_probs = joint_association_probabilities(
            [g.indices for g in gate_results],
            [g.likelihoods for g in gate_results],
            len(kept),
            self.config.max_exact_hypotheses,
            self.config.bp_iterations,
            self.config.bp_tolerance,
        )

        for track, prediction, gate_result, probs in zip(
            self.tracks, predictions, gate_results, association_probs
        ):
            self._update_track(track, prediction, gate_result, probs, measurements, kept)

        claimed = {
            int(index) for gate_result in gate_results for index in gate_result.indices
        }
        self._terminate_stale()
        self._initiate(kept, claimed)

        return self.live_tracks()

    def run(self, frames: Iterable[Sequence[Detection]]) -> list[list[Track]]:
        """Convenience wrapper: track a whole sequence, one frame at a time.

        Returns a snapshot of the live tracks per frame.  Snapshots are deep
        enough to survive later updates — the arrays are copied.
        """
        history = []
        for detections in frames:
            live = self.update(detections)
            history.append([track.copy() for track in live])
        return history

    def live_tracks(self) -> list[Track]:
        """Tracks that have not been terminated, confirmed ones first."""
        return [t for t in self.tracks if t.status != TrackStatus.TERMINATED]

    def reset(self) -> None:
        """Drop all state, as if the tracker had just been constructed."""
        self.tracks.clear()
        self.frame_index = 0
        self._next_track_id = 1

    # -------------------------------------------------------------- internals

    def _update_track(
        self,
        track: Track,
        prediction,
        gate_result,
        probs: np.ndarray,
        measurements: np.ndarray,
        detections: Sequence[Detection],
    ) -> None:
        track.state, track.covariance = pda_update(
            prediction, measurements, gate_result, probs, self._H
        )
        track.age += 1

        miss_prob = float(probs[0])
        track.last_association_prob = 1.0 - miss_prob

        # The miss hypothesis winning outright is the termination signal; this
        # is stricter than "no detection gated" and lets a track survive a frame
        # where a weak, distant detection was the only candidate.
        if gate_result.indices.size == 0 or np.argmax(probs) == 0:
            track.consecutive_misses += 1
            return

        track.consecutive_misses = 0
        track.hits += 1
        if track.hits >= self.config.min_hits:
            track.status = TrackStatus.CONFIRMED

        self._update_extent(track, gate_result, probs, detections)

    def _update_extent(
        self, track: Track, gate_result, probs: np.ndarray, detections: Sequence[Detection]
    ) -> None:
        """Blend the gated detections' box sizes by association probability.

        The box extent is not part of the filter state — it is a probability
        weighted average of whichever detections the track claimed, which keeps
        the drawn box the right size without inventing a dynamics model for it.
        """
        hit_probs = probs[1:]
        total = hit_probs.sum()
        if total <= 0.0:
            return
        weights = hit_probs / total
        widths = np.array([detections[int(i)].width for i in gate_result.indices])
        heights = np.array([detections[int(i)].height for i in gate_result.indices])
        track.width = float(weights @ widths)
        track.height = float(weights @ heights)

    def _terminate_stale(self) -> None:
        for track in self.tracks:
            if track.consecutive_misses > self.config.max_misses:
                track.status = TrackStatus.TERMINATED
        # Actually drop them, rather than carrying dead entries forever as the
        # original did — that grew the state arrays without bound.
        self.tracks = [t for t in self.tracks if t.status != TrackStatus.TERMINATED]

    def _initiate(self, detections: Sequence[Detection], claimed: set[int]) -> None:
        """Start a track for every detection no existing track gated.

        Velocity is unknown at this point, so the covariance says so — see
        :meth:`MotionModel.unknown_velocity_var`. The wide initial gate closes
        again within a few frames as the filter observes actual motion.
        """
        initial_cov = self.motion.initial_covariance(
            MotionModel.unknown_velocity_var(self.config.max_speed)
        )
        for index, detection in enumerate(detections):
            if index in claimed:
                continue
            state = np.array([detection.cx, 0.0, detection.cy, 0.0])
            self.tracks.append(
                Track(
                    track_id=self._next_track_id,
                    state=state,
                    covariance=initial_cov.copy(),
                    width=detection.width,
                    height=detection.height,
                    hits=1,
                    age=1,
                )
            )
            self._next_track_id += 1

    def seed_from_two_frames(
        self, first: Sequence[Detection], second: Sequence[Detection]
    ) -> None:
        """Initialise tracks with velocities estimated from two frames.

        Each first-frame detection is paired with its nearest second-frame
        neighbour and the implied velocity is kept only if it is below
        ``config.max_speed``; otherwise the track starts at rest.  This
        reproduces the original bootstrap, which measurably shortens the time
        the filter needs to lock on compared with starting every track at zero
        velocity.
        """
        self.reset()
        first = [d for d in first if d.score >= self.config.score_threshold]
        second = [d for d in second if d.score >= self.config.score_threshold]
        if not first:
            return

        # Unlike _initiate, velocity here is measured rather than unknown, so the
        # default (tight) velocity variance is the right prior.
        initial_cov = self.motion.initial_covariance()
        second_positions = (
            np.array([d.position for d in second]) if second else np.zeros((0, 2))
        )

        for detection in first:
            velocity = np.zeros(2)
            if second_positions.size:
                distances = np.linalg.norm(second_positions - detection.position, axis=1)
                nearest = second_positions[int(np.argmin(distances))]
                candidate = (nearest - detection.position) / self.motion.dt
                velocity = np.where(np.abs(candidate) < self.config.max_speed, candidate, 0.0)

            self.tracks.append(
                Track(
                    track_id=self._next_track_id,
                    state=np.array([detection.cx, velocity[0], detection.cy, velocity[1]]),
                    covariance=initial_cov.copy(),
                    width=detection.width,
                    height=detection.height,
                    hits=1,
                    age=1,
                )
            )
            self._next_track_id += 1
