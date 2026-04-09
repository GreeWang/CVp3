from __future__ import annotations

import cv2
import numpy as np


def temporal_background_fill(
    frame_index: int,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    temporal_radius: int,
    use_temporal_median: bool,
) -> tuple[np.ndarray, np.ndarray]:
    filled = frames[frame_index].copy()
    current_mask = masks[frame_index] > 0
    unresolved = current_mask.copy()
    if not np.any(current_mask):
        return filled, unresolved.astype(np.uint8) * 255

    ys, xs = np.where(current_mask)
    for y, x in zip(ys, xs):
        candidates = []
        for offset in range(1, temporal_radius + 1):
            for neighbor in (frame_index - offset, frame_index + offset):
                if 0 <= neighbor < len(frames) and masks[neighbor][y, x] == 0:
                    candidates.append(frames[neighbor][y, x])
        if candidates:
            candidate_array = np.asarray(candidates, dtype=np.float32)
            filled[y, x] = (
                np.median(candidate_array, axis=0) if use_temporal_median else candidate_array[0]
            ).astype(np.uint8)
            unresolved[y, x] = False
    return filled, unresolved.astype(np.uint8) * 255


def spatial_inpaint(image: np.ndarray, residual_mask: np.ndarray, method: str, radius: int) -> np.ndarray:
    if not np.any(residual_mask > 0):
        return image.copy()
    cv_method = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(image, residual_mask.astype(np.uint8), radius, cv_method)
