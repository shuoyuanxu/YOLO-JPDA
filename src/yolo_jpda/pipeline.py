"""End-to-end detect-then-track pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .config import MotionModel, TrackerConfig
from .io.sequences import VideoWriter, write_mot_results
from .tracking.tracker import JPDATracker
from .types import Detection, Track
from .visualize import draw_detections, draw_tracks, stack_side_by_side

__all__ = ["PipelineResult", "run_pipeline"]


@dataclass
class PipelineResult:
    """What a pipeline run produced."""

    rows: list[tuple[int, Track]] = field(default_factory=list)
    """``(frame_number, track)`` for every live track in every frame."""

    frame_count: int = 0
    detection_count: int = 0

    def track_ids(self) -> set[int]:
        return {track.track_id for _, track in self.rows}

    def to_mot(self, path: Path | str, confirmed_only: bool = True) -> None:
        write_mot_results(path, self.rows, confirmed_only=confirmed_only)


def run_pipeline(
    frames: Iterable[np.ndarray],
    detector,
    config: Optional[TrackerConfig] = None,
    motion: Optional[MotionModel] = None,
    output_video: Optional[Path | str] = None,
    fps: float = 25.0,
    side_by_side: bool = False,
    progress: bool = False,
) -> PipelineResult:
    """Detect and track through a sequence of frames.

    Args:
        frames: Source frames, as BGR arrays.
        detector: Anything satisfying the ``Detector`` protocol.
        config: Tracker parameters; defaults are used when omitted.
        motion: Motion model; defaults are used when omitted.
        output_video: Where to write the annotated video, if anywhere.
        fps: Frame rate of the output video.
        side_by_side: Render raw detections next to tracks, which makes it
            obvious at a glance what the tracker added over the detector.
        progress: Print a one-line-per-frame progress report.

    Returns:
        A :class:`PipelineResult` holding every frame's live tracks.
    """
    tracker = JPDATracker(config=config, motion=motion)
    result = PipelineResult()

    writer = VideoWriter(output_video, fps=fps) if output_video else None
    try:
        for frame_number, frame in enumerate(frames, start=1):
            detections: Sequence[Detection] = detector.detect(frame)
            tracks = tracker.update(detections)

            result.frame_count += 1
            result.detection_count += len(detections)
            result.rows.extend((frame_number, track.copy()) for track in tracks)

            if writer is not None:
                writer.write(_render(frame, detections, tracks, side_by_side))

            if progress:
                print(
                    f"frame {frame_number:5d}  "
                    f"detections {len(detections):3d}  tracks {len(tracks):3d}"
                )
    finally:
        if writer is not None:
            writer.release()

    return result


def _render(
    frame: np.ndarray,
    detections: Sequence[Detection],
    tracks: Sequence[Track],
    side_by_side: bool,
) -> np.ndarray:
    tracked = draw_tracks(frame, tracks)
    if not side_by_side:
        return tracked
    return stack_side_by_side(
        tracked, draw_detections(frame, detections), "JPDA", "Detections"
    )
