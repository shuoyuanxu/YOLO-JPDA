"""Reading image sequences and videos, and writing results back out."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from ..types import Track

__all__ = [
    "ImageSequence",
    "VideoSource",
    "VideoWriter",
    "write_mot_results",
]

_TRAILING_NUMBER = re.compile(r"(\d+)(?!.*\d)")


class ImageSequence:
    """An ordered folder of image files.

    Files are sorted by the trailing number in their name, so ``a2.jpg`` comes
    before ``a10.jpg`` — plain lexicographic sorting gets that backwards, which
    silently scrambles frame order and is a genuinely nasty bug to chase.
    """

    def __init__(self, directory: Path | str, pattern: str = "*.jpg") -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise NotADirectoryError(f"No such image directory: {self.directory}")

        self.paths = sorted(self.directory.glob(pattern), key=_sort_key)
        if not self.paths:
            raise FileNotFoundError(f"No files matching {pattern!r} in {self.directory}")

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[np.ndarray]:
        import cv2

        for path in self.paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise OSError(f"Could not decode image: {path}")
            yield frame

    def frame_size(self) -> tuple[int, int]:
        """``(width, height)`` of the first frame."""
        import cv2

        frame = cv2.imread(str(self.paths[0]))
        if frame is None:
            raise OSError(f"Could not decode image: {self.paths[0]}")
        return frame.shape[1], frame.shape[0]


def _sort_key(path: Path) -> tuple[int, str]:
    match = _TRAILING_NUMBER.search(path.stem)
    return (int(match.group(1)) if match else -1, path.stem)


class VideoSource:
    """Frames from a video file, as a context manager."""

    def __init__(self, path: Path | str) -> None:
        import cv2

        self.path = Path(path)
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise OSError(f"Could not open video: {path}")

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield frame

    def frame_size(self) -> tuple[int, int]:
        import cv2

        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def fps(self) -> float:
        import cv2

        return float(self._capture.get(cv2.CAP_PROP_FPS)) or 25.0

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


class VideoWriter:
    """Writes annotated frames to a video file.

    The frame size is taken from the first frame written, so callers do not
    have to declare it up front and cannot get it wrong.
    """

    def __init__(self, path: Path | str, fps: float = 25.0, codec: str = "mp4v") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.codec = codec
        self._writer = None
        self._size: Optional[tuple[int, int]] = None

    def write(self, frame: np.ndarray) -> None:
        import cv2

        height, width = frame.shape[:2]

        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self._writer.isOpened():
                raise OSError(f"Could not open video writer for {self.path} (codec {self.codec!r})")
            self._size = (width, height)

        elif (width, height) != self._size:
            # OpenCV would silently drop the frame after printing an opaque
            # ffmpeg warning, leaving a video that is quietly missing frames.
            raise ValueError(
                f"Frame size changed mid-video: expected {self._size}, got "
                f"({width}, {height}). Resize frames to a common size before "
                f"writing, or write each size to its own file."
            )

        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


def write_mot_results(
    path: Path | str,
    results: Sequence[tuple[int, Track]],
    confirmed_only: bool = True,
) -> None:
    """Write tracking output in MOT-challenge format.

    Each row is ``frame, id, left, top, width, height, conf, -1, -1, -1``,
    which is what the standard CLEAR-MOT evaluation tools expect.
    """
    from ..types import TrackStatus

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for frame_number, track in results:
            if confirmed_only and track.status != TrackStatus.CONFIRMED:
                continue
            left, top, width, height = track.to_tlwh()
            handle.write(
                f"{frame_number},{track.track_id},{left:.2f},{top:.2f},"
                f"{width:.2f},{height:.2f},{track.last_association_prob:.4f},-1,-1,-1\n"
            )
