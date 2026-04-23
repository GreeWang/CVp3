from __future__ import annotations

import cv2
import numpy as np

from cv_project.utils.mask_ops import close_mask, dilate_mask, open_mask, remove_small_components


def build_diffusion_target_masks(
    frames: list[np.ndarray],
    refined_masks: list[np.ndarray],
    restored_frames: list[np.ndarray],
    candidate_masks: list[np.ndarray],
    config: dict,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not (len(frames) == len(refined_masks) == len(restored_frames) == len(candidate_masks)):
        raise ValueError("frames, refined_masks, restored_frames, and candidate_masks must have the same length.")

    borrowable_masks: list[np.ndarray] = []
    diffusion_target_masks: list[np.ndarray] = []
    for frame_index in range(len(frames)):
        borrowable, support_fraction_map, mean_color_error_map = _build_borrowable_mask(
            frame_index=frame_index,
            frames=frames,
            masks=refined_masks,
            restored_frame=restored_frames[frame_index],
            candidate_mask=candidate_masks[frame_index],
            config=config,
        )
        diffusion_target = _build_diffusion_target_mask(
            candidate_mask=candidate_masks[frame_index],
            borrowable_mask=borrowable,
            support_fraction_map=support_fraction_map,
            mean_color_error_map=mean_color_error_map,
            config=config,
        )
        borrowable_masks.append(borrowable)
        diffusion_target_masks.append(diffusion_target)
    return borrowable_masks, diffusion_target_masks


def _build_borrowable_mask(
    frame_index: int,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    restored_frame: np.ndarray,
    candidate_mask: np.ndarray,
    config: dict,
) -> np.ndarray:
    candidate_binary = candidate_mask > 0
    if not np.any(candidate_binary):
        return np.zeros_like(candidate_mask, dtype=np.uint8)

    temporal_radius = int(config.get("borrow_temporal_radius", config.get("temporal_radius", 4)))
    min_neighbors = int(config.get("borrow_min_neighbors", config.get("min_temporal_neighbors", 2)))
    color_threshold = float(config.get("borrow_color_l1_threshold", 28.0))
    require_color = bool(config.get("borrow_require_mean_color", False))
    use_direct_coordinates = bool(config.get("borrow_use_direct_coordinates", True))

    support_count = np.zeros(candidate_mask.shape, dtype=np.uint8)
    color_error_sum = np.zeros(candidate_mask.shape, dtype=np.float32)
    restored_float = restored_frame.astype(np.float32)
    total_considered_neighbors = 0

    for offset in range(1, temporal_radius + 1):
        for neighbor_index in (frame_index - offset, frame_index + offset):
            if not 0 <= neighbor_index < len(frames):
                continue
            total_considered_neighbors += 1
            neighbor_good = np.zeros(candidate_mask.shape, dtype=bool)
            neighbor_error = np.zeros(candidate_mask.shape, dtype=np.float32)
            if use_direct_coordinates:
                direct_good, direct_error = _find_supported_pixels(
                    source_frame=frames[neighbor_index],
                    source_unmasked=masks[neighbor_index] == 0,
                    candidate_binary=candidate_binary,
                    restored_float=restored_float,
                    require_color=require_color,
                    color_threshold=color_threshold,
                )
                neighbor_good |= direct_good
                neighbor_error[direct_good] = direct_error[direct_good]

            warped_frame, warped_unmasked = _warp_neighbor_into_current(
                source_frame=frames[neighbor_index],
                source_mask=masks[neighbor_index],
                target_frame=frames[frame_index],
            )
            warped_good, warped_error = _find_supported_pixels(
                source_frame=warped_frame,
                source_unmasked=warped_unmasked,
                candidate_binary=candidate_binary,
                restored_float=restored_float,
                require_color=require_color,
                color_threshold=color_threshold,
            )
            new_warped = warped_good & ~neighbor_good
            overlap = warped_good & neighbor_good
            neighbor_error[new_warped] = warped_error[new_warped]
            if require_color and np.any(overlap):
                neighbor_error[overlap] = np.minimum(neighbor_error[overlap], warped_error[overlap])
            neighbor_good |= warped_good
            support_count[neighbor_good] += 1
            color_error_sum[neighbor_good] += neighbor_error[neighbor_good]

    borrowable = candidate_binary & (support_count >= min_neighbors)
    if require_color:
        mean_error = np.zeros_like(color_error_sum)
        np.divide(color_error_sum, support_count, out=mean_error, where=support_count > 0)
        borrowable &= mean_error <= color_threshold

    if total_considered_neighbors <= 0:
        support_fraction_map = np.zeros(candidate_mask.shape, dtype=np.float32)
    else:
        support_fraction_map = np.clip(
            support_count.astype(np.float32) / float(total_considered_neighbors),
            0.0,
            1.0,
        )

    mean_color_error_map = np.zeros_like(color_error_sum)
    np.divide(color_error_sum, support_count, out=mean_color_error_map, where=support_count > 0)

    borrowable_mask = borrowable.astype(np.uint8) * 255
    erode_kernel = int(config.get("borrowable_erode_kernel", 0))
    if erode_kernel > 1:
        borrowable_mask = cv2.erode(
            borrowable_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel, erode_kernel)),
            iterations=1,
        )
    return borrowable_mask, support_fraction_map, mean_color_error_map


def _find_supported_pixels(
    source_frame: np.ndarray,
    source_unmasked: np.ndarray,
    candidate_binary: np.ndarray,
    restored_float: np.ndarray,
    require_color: bool,
    color_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = candidate_binary & source_unmasked
    if not np.any(valid):
        return np.zeros(candidate_binary.shape, dtype=bool), np.zeros(candidate_binary.shape, dtype=np.float32)
    color_error = np.mean(np.abs(source_frame.astype(np.float32) - restored_float), axis=2)
    if require_color:
        good = valid & (color_error <= color_threshold)
    else:
        good = valid
    return good, color_error


def _build_diffusion_target_mask(
    candidate_mask: np.ndarray,
    borrowable_mask: np.ndarray,
    support_fraction_map: np.ndarray,
    mean_color_error_map: np.ndarray,
    config: dict,
) -> np.ndarray:
    target = ((candidate_mask > 0) & ~(borrowable_mask > 0)).astype(np.uint8) * 255

    if bool(config.get("diffusion_confidence_gating", True)):
        support_threshold = float(config.get("diffusion_gate_support_threshold", 0.5))
        color_threshold = float(config.get("diffusion_gate_color_l1_threshold", 22.0))
        has_support = support_fraction_map >= support_threshold
        color_reliable = mean_color_error_map <= color_threshold
        trusted = (candidate_mask > 0) & has_support & color_reliable
        target[trusted] = 0

    open_kernel = int(config.get("diffusion_target_open_kernel", 0))
    if open_kernel > 1:
        target = open_mask(target, open_kernel)

    close_kernel = int(config.get("diffusion_target_close_kernel", 5))
    if close_kernel > 1:
        target = close_mask(target, close_kernel)

    min_area_ratio = float(config.get("diffusion_target_min_area_ratio", config.get("hard_region_min_area_ratio", 0.0002)))
    target = remove_small_components(target, min_area_ratio)

    dilation_kernel = int(config.get("diffusion_target_dilation", 3))
    if bool(config.get("diffusion_target_adaptive_dilation", True)):
        target_ratio = float(np.count_nonzero(target)) / float(max(1, target.size))
        low_ratio = float(config.get("diffusion_target_small_ratio", 0.0015))
        high_ratio = float(config.get("diffusion_target_large_ratio", 0.01))
        if target_ratio <= low_ratio:
            dilation_kernel = max(1, dilation_kernel - 2)
        elif target_ratio >= high_ratio:
            dilation_kernel = dilation_kernel + 2

    if dilation_kernel > 1:
        target = dilate_mask(target, dilation_kernel)
        target = ((target > 0) & (candidate_mask > 0)).astype(np.uint8) * 255

    target = _cap_target_to_candidate_ratio(
        target=target,
        candidate_mask=candidate_mask,
        support_fraction_map=support_fraction_map,
        mean_color_error_map=mean_color_error_map,
        config=config,
    )

    return target


def _cap_target_to_candidate_ratio(
    *,
    target: np.ndarray,
    candidate_mask: np.ndarray,
    support_fraction_map: np.ndarray,
    mean_color_error_map: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Limit diffusion target to a configurable fraction of candidate pixels.

    When target coverage is too large, keep only the hardest pixels according to
    low temporal support and high color error.
    """
    max_ratio = float(config.get("diffusion_target_max_candidate_ratio", 1.0))
    max_ratio = float(np.clip(max_ratio, 0.0, 1.0))
    if max_ratio >= 0.999:
        return target

    candidate_binary = candidate_mask > 0
    target_binary = target > 0
    candidate_count = int(np.count_nonzero(candidate_binary))
    target_count = int(np.count_nonzero(target_binary))
    if candidate_count <= 0 or target_count <= 0:
        return target

    min_pixels = int(config.get("diffusion_target_min_pixels", 0))
    allowed = max(min_pixels, int(round(candidate_count * max_ratio)))
    allowed = max(0, min(allowed, candidate_count))
    if target_count <= allowed:
        return target
    if allowed == 0:
        return np.zeros_like(target, dtype=np.uint8)

    # Difficulty: low support is harder; high color error is harder.
    color_scale = float(config.get("diffusion_target_color_scale", 32.0))
    color_scale = max(1e-6, color_scale)
    color_weight = float(config.get("diffusion_target_color_weight", 0.35))
    color_weight = float(np.clip(color_weight, 0.0, 2.0))

    support_term = 1.0 - np.clip(support_fraction_map.astype(np.float32), 0.0, 1.0)
    color_term = np.clip(mean_color_error_map.astype(np.float32) / color_scale, 0.0, 2.0)
    difficulty = support_term + color_weight * color_term

    target_indices = np.flatnonzero(target_binary)
    values = difficulty.reshape(-1)[target_indices]
    if allowed >= len(target_indices):
        return target

    keep_order = np.argpartition(values, -allowed)[-allowed:]
    kept_indices = target_indices[keep_order]
    capped = np.zeros(target_binary.size, dtype=np.uint8)
    capped[kept_indices] = 255
    return capped.reshape(target.shape)


def _warp_neighbor_into_current(
    source_frame: np.ndarray,
    source_mask: np.ndarray,
    target_frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    source_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
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
    warped_frame = cv2.remap(
        source_frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    warped_mask = cv2.remap(
        (source_mask > 0).astype(np.uint8) * 255,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return warped_frame, warped_mask == 0
