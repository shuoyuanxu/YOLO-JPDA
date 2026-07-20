"""Entry point. Configuration lives in YAML; flags only override it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .modes import run
from .settings import MODES, load_settings

__all__ = ["main", "build_parser"]

DEFAULT_CONFIG = Path("config/default.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolo-jpda",
        description="Multi-object tracking by YOLO detection and joint "
        "probabilistic data association.",
        epilog="Everything is set in the config file; the flags below are "
        "overrides for one-off runs.",
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"YAML config file (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument("-m", "--mode", choices=MODES, help="Override the configured mode.")
    parser.add_argument("--images", type=Path, help="Override input.images.")
    parser.add_argument("--video", type=Path, help="Override input.video.")
    parser.add_argument("--weights", help="Override detector.weights.")
    parser.add_argument("--detections", type=Path, help="Override detector.detections.")
    parser.add_argument("--device", help="Override detector.device, e.g. cuda:0.")
    parser.add_argument("--output-dir", type=Path, dest="output_dir",
                        help="Override output.dir.")
    parser.add_argument("--quiet", action="store_true", default=None,
                        help="Suppress per-frame progress.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.config.exists():
        print(
            f"error: no config at {args.config}\n"
            f"Copy config/default.yaml and edit it, or pass --config.",
            file=sys.stderr,
        )
        return 2

    overrides = vars(args).copy()
    overrides.pop("config")

    try:
        settings = load_settings(args.config, **overrides)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        return run(settings)
    except (ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
