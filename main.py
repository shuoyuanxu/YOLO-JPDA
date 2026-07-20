#!/usr/bin/env python3
"""YOLO-JPDA entry point.

    python main.py                              # config/default.yaml
    python main.py --mode replay                # override the mode
    python main.py -c config/mot17.yaml         # a different config

Four modes, selected by ``mode:`` in the config or ``--mode``:

    detect     detector only, writes detections.txt
    track      detector + JPDA tracking          (default)
    replay     JPDA over a detection file, no detector needed
    evaluate   replay, then CLEAR-MOT metrics vs. ground truth

Everything else is configured in the YAML file. See config/default.yaml.
"""

import sys
from pathlib import Path

# Support running from a checkout without installing the package first.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from yolo_jpda.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
