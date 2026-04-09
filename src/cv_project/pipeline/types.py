from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DetectionRecord:
    instance_id: str
    class_name: str
    score: float
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    motion_score: float | None = None
    valid_points: int = 0
    is_dynamic: bool = False


@dataclass
class FrameRecord:
    frame_index: int
    image: np.ndarray
    source_path: str
    raw_detections: list[DetectionRecord] = field(default_factory=list)
    filtered_detections: list[DetectionRecord] = field(default_factory=list)
    raw_dynamic_mask: np.ndarray | None = None
    final_mask: np.ndarray | None = None
    temporal_fill: np.ndarray | None = None
    restored_image: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
