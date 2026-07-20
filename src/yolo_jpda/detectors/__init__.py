"""Detection backends.

Only :class:`~yolo_jpda.detectors.base.Detector` and the dependency-free
:class:`~yolo_jpda.detectors.file.FileDetector` are imported eagerly.  The
neural backends pull in heavy optional dependencies, so import them directly:

    from yolo_jpda.detectors.ultralytics_yolo import UltralyticsDetector
    from yolo_jpda.detectors.legacy_yolov1 import LegacyYolov1Detector
"""

from .base import Detector
from .file import FileDetector, load_mot_detections

__all__ = ["Detector", "FileDetector", "load_mot_detections"]
