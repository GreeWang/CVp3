from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


class OfficialProPainterRunner:
    """Wrapper around the official ProPainter inference script."""

    def __init__(self, config: dict, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.repo_dir = self._resolve_path(config.get("propainter_repo_dir", "ProPainter"))
        self.script_path = self.repo_dir / "inference_propainter.py"
        if not self.script_path.exists():
            raise FileNotFoundError(f"Official ProPainter inference script not found: {self.script_path}")

    def run(
        self,
        frames_dir: Path,
        masks_dir: Path,
        output_root: Path,
    ) -> tuple[list[np.ndarray], Path]:
        command = [
            sys.executable,
            str(self.script_path),
            "--video",
            str(frames_dir),
            "--mask",
            str(masks_dir),
            "--output",
            str(output_root),
            "--save_frames",
            "--mask_dilation",
            str(int(self.config.get("mask_dilation", 4))),
            "--neighbor_length",
            str(int(self.config.get("neighbor_length", 10))),
            "--ref_stride",
            str(int(self.config.get("ref_stride", 10))),
            "--subvideo_length",
            str(int(self.config.get("subvideo_length", 80))),
            "--raft_iter",
            str(int(self.config.get("raft_iter", 20))),
        ]
        if bool(self.config.get("fp16", False)):
            command.append("--fp16")
        if int(self.config.get("height", -1)) > 0 and int(self.config.get("width", -1)) > 0:
            command.extend(["--height", str(int(self.config["height"])), "--width", str(int(self.config["width"]))])

        result = subprocess.run(
            command,
            cwd=str(self.repo_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Official ProPainter inference failed.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        video_name = frames_dir.name
        save_root = output_root / video_name
        saved_frames_dir = save_root / "frames"
        if not saved_frames_dir.exists():
            raise RuntimeError(
                f"ProPainter finished without saved frames under {saved_frames_dir}. "
                "Check whether the official repo downloaded weights correctly."
            )

        restored_frames = self._load_frames(saved_frames_dir)
        return restored_frames, save_root

    @staticmethod
    def _load_frames(frame_dir: Path) -> list[np.ndarray]:
        frame_paths = sorted(path for path in frame_dir.iterdir() if path.is_file())
        frames: list[np.ndarray] = []
        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path))
            if image is None:
                raise RuntimeError(f"Failed to read ProPainter output frame: {frame_path}")
            frames.append(image)
        return frames

    def _resolve_path(self, value: str | None) -> Path:
        if not value:
            raise ValueError("Missing required path value.")
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path
