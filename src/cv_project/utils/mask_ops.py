from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def merge_instance_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    merged = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        merged = np.maximum(merged, (mask > 0).astype(np.uint8) * 255)
    return merged


def fill_holes(mask: np.ndarray) -> np.ndarray:
    filled = ndimage.binary_fill_holes(mask > 0)
    return filled.astype(np.uint8) * 255


def remove_small_components(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = int(round(mask.shape[0] * mask.shape[1] * min_area_ratio))
    output = np.zeros_like(binary)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label] = 1
    return output.astype(np.uint8) * 255


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel, iterations=1)


def close_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def open_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def temporal_majority_vote(masks: list[np.ndarray], window: int, votes: int) -> list[np.ndarray]:
    if window <= 1:
        return [mask.copy() for mask in masks]
    half_window = window // 2
    binary_masks = [(mask > 0).astype(np.uint8) for mask in masks]
    smoothed: list[np.ndarray] = []
    for index in range(len(binary_masks)):
        start = max(0, index - half_window)
        end = min(len(binary_masks), index + half_window + 1)
        stacked = np.stack(binary_masks[start:end], axis=0)
        current = (stacked.sum(axis=0) >= votes).astype(np.uint8) * 255
        smoothed.append(current)
    return smoothed


def postprocess_masks(raw_masks: list[np.ndarray], config: dict) -> list[np.ndarray]:
    processed: list[np.ndarray] = []
    for mask in raw_masks:
        current = mask.copy()
        if config.get("fill_holes", True):
            current = fill_holes(current)
        current = remove_small_components(current, float(config["min_component_area_ratio"]))
        current = dilate_mask(current, int(config["dilation_kernel_size"]))
        processed.append(current)
    if config.get("temporal_smoothing", True):
        processed = temporal_majority_vote(
            processed,
            int(config["temporal_window"]),
            int(config["temporal_votes"]),
        )
    return processed


def warp_mask_with_flow(source_mask: np.ndarray, source_frame: np.ndarray, target_frame: np.ndarray) -> np.ndarray:
    source_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        target_gray,
        source_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    height, width = target_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped = cv2.remap(
        source_mask.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (warped > 127).astype(np.uint8) * 255


def flow_guided_temporal_consensus(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    current_weight: int,
    threshold: int,
) -> list[np.ndarray]:
    if len(frames) != len(masks):
        raise ValueError("frames and masks must have the same length.")
    if len(masks) <= 1:
        return [mask.copy() for mask in masks]

    refined: list[np.ndarray] = []
    for index, current_mask in enumerate(masks):
        votes = (current_mask > 0).astype(np.uint8) * current_weight
        if index > 0:
            votes += (warp_mask_with_flow(masks[index - 1], frames[index - 1], frames[index]) > 0).astype(np.uint8)
        if index + 1 < len(masks):
            votes += (warp_mask_with_flow(masks[index + 1], frames[index + 1], frames[index]) > 0).astype(np.uint8)
        refined.append((votes >= threshold).astype(np.uint8) * 255)
    return refined


def refine_mask_sequence(raw_masks: list[np.ndarray], frames: list[np.ndarray], config: dict) -> list[np.ndarray]:
    processed = postprocess_masks(raw_masks, config)
    close_kernel = int(config.get("close_kernel_size", 0))
    open_kernel = int(config.get("open_kernel_size", 0))
    final_dilate = int(config.get("final_dilation_kernel_size", 0))

    refined: list[np.ndarray] = []
    for mask in processed:
        current = mask.copy()
        if close_kernel > 1:
            current = close_mask(current, close_kernel)
        if open_kernel > 1:
            current = open_mask(current, open_kernel)
        if final_dilate > 1:
            current = dilate_mask(current, final_dilate)
        refined.append(current)

    if config.get("flow_guided_consistency", False):
        refined = flow_guided_temporal_consensus(
            frames=frames,
            masks=refined,
            current_weight=int(config.get("flow_consistency_current_weight", 2)),
            threshold=int(config.get("flow_consistency_threshold", 2)),
        )

    return refined
