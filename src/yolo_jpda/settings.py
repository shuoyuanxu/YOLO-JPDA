"""YAML-backed settings.

One config file drives every mode. Values map onto :class:`TrackerConfig` and
:class:`MotionModel`, with a few friendlier spellings — gate radius given as a
confidence, clutter given as an expected false-alarm count — that are converted
here rather than being left for the user to work out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import MotionModel, TrackerConfig

__all__ = ["Settings", "load_settings"]

MODES = ("detect", "track", "replay", "evaluate")


@dataclass
class Settings:
    """Everything a run needs, assembled from the config file."""

    mode: str = "track"

    # input
    images: Optional[Path] = None
    video: Optional[Path] = None
    pattern: str = "*.jpg"

    # detector
    backend: str = "yolo"
    weights: str = "yolov8n.pt"
    detections: Optional[Path] = None
    device: Optional[str] = None
    confidence: float = 0.25
    classes: Optional[list] = None

    # tracker / motion
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    motion: MotionModel = field(default_factory=MotionModel)

    # output
    output_dir: Path = Path("output")
    write_video: bool = True
    write_mot: bool = True
    side_by_side: bool = False
    fps: float = 25.0
    quiet: bool = False

    # evaluate
    ground_truth: Optional[Path] = None

    def validate(self) -> None:
        """Fail early, with a message that says what to change."""
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

        if bool(self.images) == bool(self.video):
            raise ValueError("set exactly one of input.images or input.video")

        if self.mode == "replay" and not self.detections:
            raise ValueError("mode 'replay' needs detector.detections")

        if self.mode == "evaluate" and not self.ground_truth:
            raise ValueError("mode 'evaluate' needs evaluate.ground_truth")

        for label, path in (("input.images", self.images), ("input.video", self.video),
                            ("detector.detections", self.detections),
                            ("evaluate.ground_truth", self.ground_truth)):
            if path is not None and not Path(path).exists():
                raise FileNotFoundError(f"{label}: no such path: {path}")


def load_settings(path: Path | str, **overrides: Any) -> Settings:
    """Read a YAML config, apply overrides, and validate.

    Overrides are flat keyword arguments (as produced by the command line) and
    win over the file. ``None`` values are ignored so an unset flag does not
    wipe a configured value.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("Reading a config file requires PyYAML: pip install pyyaml") from exc

    raw = yaml.safe_load(Path(path).read_text()) or {}

    section = lambda name: raw.get(name) or {}  # noqa: E731 - terse by design
    inp, det, trk, mot, out, ev = (
        section("input"), section("detector"), section("tracker"),
        section("motion"), section("output"), section("evaluate"),
    )

    motion = MotionModel(
        dt=mot.get("dt", MotionModel.dt),
        process_noise=mot.get("process_noise", MotionModel.process_noise),
        measurement_noise_var=mot.get("measurement_noise_var",
                                      MotionModel.measurement_noise_var),
    )

    tracker = TrackerConfig(
        score_threshold=trk.get("score_threshold", TrackerConfig.score_threshold),
        gate_threshold=_gate(trk),
        detection_probability=trk.get("detection_probability",
                                      TrackerConfig.detection_probability),
        clutter_density=_clutter(trk),
        max_misses=trk.get("max_misses", TrackerConfig.max_misses),
        min_hits=trk.get("min_hits", TrackerConfig.min_hits),
        max_speed=trk.get("max_speed", TrackerConfig.max_speed),
    )

    settings = Settings(
        mode=raw.get("mode", "track"),
        images=_path(inp.get("images")),
        video=_path(inp.get("video")),
        pattern=inp.get("pattern", "*.jpg"),
        backend=det.get("backend", "yolo"),
        weights=det.get("weights", "yolov8n.pt"),
        detections=_path(det.get("detections")),
        device=det.get("device"),
        confidence=det.get("confidence", 0.25),
        classes=det.get("classes"),
        tracker=tracker,
        motion=motion,
        output_dir=Path(out.get("dir", "output")),
        write_video=out.get("video", True),
        write_mot=out.get("mot", True),
        side_by_side=out.get("side_by_side", False),
        fps=out.get("fps", 25.0),
        quiet=out.get("quiet", False),
        ground_truth=_path(ev.get("ground_truth")),
    )

    for key, value in overrides.items():
        if value is not None and hasattr(settings, key):
            setattr(settings, key, value)

    # An overridden input source replaces the configured one rather than joining
    # it, or validation would see both images and video set and refuse.
    if overrides.get("images") is not None:
        settings.video = None
    elif overrides.get("video") is not None:
        settings.images = None

    settings.validate()
    return settings


def _gate(tracker: dict) -> float:
    """Prefer an explicit radius; otherwise derive one from a confidence."""
    if "gate_threshold" in tracker:
        return float(tracker["gate_threshold"])
    if "gate_confidence" in tracker:
        return TrackerConfig.gate_from_confidence(float(tracker["gate_confidence"]))
    return TrackerConfig.gate_threshold


def _clutter(tracker: dict) -> float:
    """Prefer an explicit density; otherwise derive from false alarms per frame."""
    if "clutter_density" in tracker:
        return float(tracker["clutter_density"])
    if "expected_false_alarms" in tracker and "image_size" in tracker:
        width, height = tracker["image_size"]
        return TrackerConfig.clutter_from_image(
            float(tracker["expected_false_alarms"]), int(width), int(height)
        )
    return TrackerConfig.clutter_density


def _path(value: Any) -> Optional[Path]:
    return Path(value) if value else None
