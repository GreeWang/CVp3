from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_path(path_str: str | None, project_root: Path) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path


def normalize_frame_size(image: np.ndarray, max_long_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / float(long_side)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    max_long_side: int,
    frame_name_width: int,
) -> tuple[list[Path], float]:
    ensure_dir(output_dir)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_paths: list[Path] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = normalize_frame_size(frame, max_long_side)
        frame_path = output_dir / f"{index:0{frame_name_width}d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        index += 1
    capture.release()
    if not frame_paths:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frame_paths, fps


def list_frame_paths(frames_dir: Path, image_extensions: list[str]) -> list[Path]:
    extensions = {ext.lower() for ext in image_extensions}
    frame_paths = [path for path in sorted(frames_dir.iterdir()) if path.is_file() and path.suffix.lower() in extensions]
    if not frame_paths:
        raise RuntimeError(f"No frame images found in: {frames_dir}")
    return frame_paths


def write_video(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    if not frame_paths:
        raise RuntimeError("No frames provided for video export.")
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Unable to read frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {output_path}")
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Unable to read frame for video export: {frame_path}")
        if frame.shape[:2] != (height, width):
            raise RuntimeError(f"Frame size mismatch during video export: {frame_path}")
        writer.write(frame)
    writer.release()


def save_json(data: dict[str, Any], output_path: Path) -> None:
    def default_serializer(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=default_serializer)
