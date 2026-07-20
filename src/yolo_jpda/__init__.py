"""YOLO-JPDA: multi-object tracking by joint probabilistic data association.

Typical use::

    from yolo_jpda import Detection, JPDATracker

    tracker = JPDATracker()
    for frame_detections in sequence:
        tracks = tracker.update(frame_detections)

Detector backends live in :mod:`yolo_jpda.detectors` and are imported lazily,
so neither ``ultralytics`` nor TensorFlow is needed to use the tracker itself.
"""

from .config import MotionModel, TrackerConfig
from .pipeline import PipelineResult, run_pipeline
from .tracking.association import joint_association_probabilities
from .tracking.tracker import JPDATracker
from .types import Detection, Track, TrackStatus

__version__ = "1.0.0"

__all__ = [
    "Detection",
    "JPDATracker",
    "MotionModel",
    "PipelineResult",
    "Track",
    "TrackStatus",
    "TrackerConfig",
    "joint_association_probabilities",
    "run_pipeline",
]
