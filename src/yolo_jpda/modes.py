"""The four run modes.

============  ====================================================
``detect``    Detector only. Writes detections, no tracking.
``track``     Detector + JPDA. The normal mode.
``replay``    JPDA over detections read from a file. No detector,
              so no deep-learning dependencies are needed.
``evaluate``  ``replay``, then CLEAR-MOT metrics against ground truth.
============  ====================================================
"""

from __future__ import annotations

import sys
from typing import Iterable

from .detectors.file import FileDetector
from .io.sequences import ImageSequence, VideoSource, write_mot_results
from .pipeline import run_pipeline
from .settings import Settings

__all__ = ["run"]


def run(settings: Settings) -> int:
    """Dispatch to the configured mode. Returns a process exit code."""
    return {
        "detect": _detect,
        "track": _track,
        "replay": _replay,
        "evaluate": _evaluate,
    }[settings.mode](settings)


# ------------------------------------------------------------------ modes


def _detect(settings: Settings) -> int:
    """Run the detector over the sequence and write MOT-format detections."""
    frames = _open_frames(settings)
    detector = _build_detector(settings)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.output_dir / "detections.txt"

    total = 0
    with destination.open("w") as handle:
        for frame_number, frame in enumerate(frames, start=1):
            detections = detector.detect(frame)
            total += len(detections)
            for detection in detections:
                left, top, width, height = detection.to_tlwh()
                handle.write(
                    f"{frame_number},-1,{left:.2f},{top:.2f},"
                    f"{width:.2f},{height:.2f},{detection.score:.4f},-1,-1,-1\n"
                )
            _report(settings, f"frame {frame_number:5d}  detections {len(detections):3d}")

    print(f"{total} detections -> {destination}")
    return 0


def _track(settings: Settings) -> int:
    """Detect and track, writing an annotated video and MOT results."""
    return _run_tracking(settings, _build_detector(settings))


def _replay(settings: Settings) -> int:
    """Track detections read from a file, with no detector involved."""
    detector = FileDetector.from_mot(settings.detections)
    _report(settings, f"replaying {len(detector)} frames of detections")

    # Frames and detections are paired positionally, so a length mismatch means
    # every result is misaligned. Silent misalignment looks like wild tracking
    # rather than a data problem, so say so.
    if settings.images:
        n_images = len(ImageSequence(settings.images, settings.pattern))
        if n_images != len(detector):
            print(
                f"warning: {n_images} images but {len(detector)} frames of "
                f"detections in {settings.detections}. They are paired in order, "
                f"so results past frame {min(n_images, len(detector))} will be "
                f"misaligned. Check input.pattern and detector.detections.",
                file=sys.stderr,
            )

    return _run_tracking(settings, detector)


def _evaluate(settings: Settings) -> int:
    """Track from a detection file, then score against ground truth."""
    settings.write_video = False
    settings.write_mot = True
    exit_code = _replay(settings)
    if exit_code:
        return exit_code

    try:
        import motmetrics as mm
    except ImportError:
        print(
            "\nTracking finished, but scoring needs motmetrics:\n"
            "    pip install motmetrics\n"
            f"Results are in {settings.output_dir / 'tracks.txt'}"
        )
        return 0

    ground_truth = mm.io.loadtxt(str(settings.ground_truth), fmt="mot15-2D")
    hypotheses = mm.io.loadtxt(str(settings.output_dir / "tracks.txt"), fmt="mot15-2D")

    accumulator = mm.utils.compare_to_groundtruth(ground_truth, hypotheses, "iou", distth=0.5)
    metrics = mm.metrics.create()
    summary = metrics.compute(accumulator, metrics=mm.metrics.motchallenge_metrics, name="jpda")

    print()
    print(mm.io.render_summary(
        summary,
        formatters=metrics.formatters,
        namemap=mm.io.motchallenge_metric_names,
    ))
    return 0


# ------------------------------------------------------------------ helpers


def _run_tracking(settings: Settings, detector) -> int:
    frames = _open_frames(settings)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    video_path = settings.output_dir / "tracked.mp4" if settings.write_video else None
    result = run_pipeline(
        frames,
        detector,
        config=settings.tracker,
        motion=settings.motion,
        output_video=video_path,
        fps=settings.fps,
        side_by_side=settings.side_by_side,
        progress=not settings.quiet,
    )

    if settings.write_mot:
        destination = settings.output_dir / "tracks.txt"
        write_mot_results(destination, result.rows)
        print(f"tracks -> {destination}")
    if video_path:
        print(f"video  -> {video_path}")

    print(
        f"{result.frame_count} frames, {result.detection_count} detections, "
        f"{len(result.track_ids())} tracks"
    )
    return 0


def _open_frames(settings: Settings) -> Iterable:
    if settings.images:
        return ImageSequence(settings.images, settings.pattern)
    return VideoSource(settings.video)


def _build_detector(settings: Settings):
    if settings.backend == "file":
        if not settings.detections:
            raise ValueError("detector.backend 'file' needs detector.detections")
        return FileDetector.from_mot(settings.detections)

    if settings.backend == "yolo":
        from .detectors.ultralytics_yolo import UltralyticsDetector

        return UltralyticsDetector(
            weights=settings.weights,
            confidence=settings.confidence,
            classes=settings.classes,
            device=settings.device,
        )

    if settings.backend == "legacy-yolov1":
        from .detectors.legacy_yolov1 import LegacyYolov1Detector

        return LegacyYolov1Detector(
            weights=settings.weights,
            confidence=settings.confidence,
            classes=settings.classes,
        )

    raise ValueError(
        f"unknown detector.backend {settings.backend!r}; "
        "expected 'yolo', 'file' or 'legacy-yolov1'"
    )


def _report(settings: Settings, message: str) -> None:
    if not settings.quiet:
        print(message)
