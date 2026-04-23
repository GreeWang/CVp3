from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cv_project.inpainting.restoration import spatial_inpaint


@dataclass
class RestorationArtifacts:
    temporal_fill: np.ndarray
    residual_mask: np.ndarray
    hard_mask: np.ndarray
    hard_residual_mask: np.ndarray
    hard_weak_mask: np.ndarray
    support_map: np.ndarray
    restored_image: np.ndarray
    support_ratio: float
    candidate_pixels: int
    hard_threshold_used: float


class ProPainterLikeRestorer:
    """Flow-guided temporal filling with spatial fallback."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def restore_sequence(self, frames: list[np.ndarray], masks: list[np.ndarray]) -> list[RestorationArtifacts]:
        artifacts: list[RestorationArtifacts] = []
        for frame_index in range(len(frames)):
            temporal_fill, residual_mask, support_map, support_ratio, candidate_pixels = self._temporal_fill_frame(
                frame_index=frame_index,
                frames=frames,
                masks=masks,
            )
            restored = spatial_inpaint(
                temporal_fill,
                residual_mask,
                method=str(self.config.get("fallback_method", "telea")),
                radius=int(self.config.get("inpaint_radius", 3)),
            )
            hard_mask, hard_residual_mask, hard_weak_mask, hard_threshold_used = self._build_hard_mask(
                current_mask=masks[frame_index],
                residual_mask=residual_mask,
                support_map=support_map,
                current_frame=frames[frame_index],
                temporal_fill=temporal_fill,
            )
            artifacts.append(
                RestorationArtifacts(
                    temporal_fill=temporal_fill,
                    residual_mask=residual_mask,
                    hard_mask=hard_mask,
                    hard_residual_mask=hard_residual_mask,
                    hard_weak_mask=hard_weak_mask,
                    support_map=support_map,
                    restored_image=restored,
                    support_ratio=support_ratio,
                    candidate_pixels=candidate_pixels,
                    hard_threshold_used=hard_threshold_used,
                )
            )
        return artifacts

    def _temporal_fill_frame(
        self,
        frame_index: int,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        current_frame = frames[frame_index]
        current_mask = masks[frame_index] > 0
        if not np.any(current_mask):
            return current_frame.copy(), np.zeros_like(masks[frame_index]), np.zeros_like(masks[frame_index]), 1.0, 0

        temporal_radius = int(self.config.get("temporal_radius", 4))
        min_neighbors = int(self.config.get("min_temporal_neighbors", 2))
        weighted_sum = np.zeros_like(current_frame, dtype=np.float32)
        total_weight = np.zeros(current_mask.shape, dtype=np.float32)
        support_count = np.zeros(current_mask.shape, dtype=np.uint8)
        possible_support = 0

        for offset in range(1, temporal_radius + 1):
            for neighbor_index in (frame_index - offset, frame_index + offset):
                if not 0 <= neighbor_index < len(frames):
                    continue
                possible_support += 1
                warped_frame, warped_valid = self._warp_neighbor_into_current(
                    source_frame=frames[neighbor_index],
                    source_mask=masks[neighbor_index],
                    target_frame=current_frame,
                )
                valid = current_mask & warped_valid
                if not np.any(valid):
                    continue
                weight = 1.0 / float(offset)
                weighted_sum[valid] += warped_frame[valid] * weight
                total_weight[valid] += weight
                support_count[valid] += 1

        fillable = current_mask & (support_count >= min_neighbors) & (total_weight > 0)
        temporal_fill = current_frame.copy()
        temporal_fill[fillable] = np.clip(
            weighted_sum[fillable] / total_weight[fillable, None],
            0,
            255,
        ).astype(np.uint8)
        residual_mask = (current_mask & ~fillable).astype(np.uint8) * 255
        denominator = max(1, possible_support)
        support_fraction = support_count.astype(np.float32) / float(denominator)
        support_map = np.clip(support_fraction * 255.0, 0, 255).astype(np.uint8)
        support_ratio = float(np.count_nonzero(fillable)) / float(np.count_nonzero(current_mask))
        return temporal_fill, residual_mask, support_map, support_ratio, int(np.count_nonzero(fillable))

    def _build_hard_mask(
        self,
        current_mask: np.ndarray,
        residual_mask: np.ndarray,
        support_map: np.ndarray,
        current_frame: np.ndarray,
        temporal_fill: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        current_binary = (current_mask > 0).astype(np.uint8)
        residual_hard = (residual_mask > 0).astype(np.uint8) * 255
        hard_mask = residual_hard.copy()

        weak_support_threshold = float(self.config.get("weak_support_threshold", 0.25))
        if bool(self.config.get("adaptive_weak_support_threshold", True)):
            weak_support_threshold = self._adaptive_weak_support_threshold(
                base_threshold=weak_support_threshold,
                current_binary=current_binary,
                support_map=support_map,
                current_frame=current_frame,
                temporal_fill=temporal_fill,
            )

        weak_support = ((support_map.astype(np.float32) / 255.0) <= weak_support_threshold) & (current_binary > 0)
        weak_hard = weak_support.astype(np.uint8) * 255
        if bool(self.config.get("include_weak_support_regions", True)):
            hard_mask = np.maximum(hard_mask, weak_hard)

        min_area_ratio = float(self.config.get("hard_region_min_area_ratio", 0.0002))
        min_area = max(1, int(round(current_mask.shape[0] * current_mask.shape[1] * min_area_ratio)))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((hard_mask > 0).astype(np.uint8), connectivity=8)
        filtered = np.zeros_like(hard_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == label] = 255

        kernel_size = int(self.config.get("hard_region_dilation", 5))
        if kernel_size > 1:
            filtered = cv2.dilate(
                filtered,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
                iterations=1,
            )
        return (filtered > 0).astype(np.uint8) * 255, residual_hard, weak_hard, weak_support_threshold

    def _adaptive_weak_support_threshold(
        self,
        *,
        base_threshold: float,
        current_binary: np.ndarray,
        support_map: np.ndarray,
        current_frame: np.ndarray,
        temporal_fill: np.ndarray,
    ) -> float:
        active = current_binary > 0
        if not np.any(active):
            return float(np.clip(base_threshold, 0.0, 1.0))

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        texture_score = float(np.mean(grad_mag[active]) / 255.0)

        diff = np.mean(np.abs(current_frame.astype(np.float32) - temporal_fill.astype(np.float32)), axis=2)
        residual_motion = float(np.mean(diff[active]) / 255.0)

        support_fraction = support_map.astype(np.float32) / 255.0
        support_gap = float(np.mean(np.clip(1.0 - support_fraction[active], 0.0, 1.0)))

        texture_weight = float(self.config.get("weak_support_texture_weight", 0.1))
        motion_weight = float(self.config.get("weak_support_motion_weight", 0.1))
        support_weight = float(self.config.get("weak_support_gap_weight", 0.08))
        adaptive = base_threshold + texture_weight * texture_score + motion_weight * residual_motion + support_weight * support_gap

        min_threshold = float(self.config.get("weak_support_threshold_min", 0.12))
        max_threshold = float(self.config.get("weak_support_threshold_max", 0.6))
        return float(np.clip(adaptive, min_threshold, max_threshold))

    @staticmethod
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
