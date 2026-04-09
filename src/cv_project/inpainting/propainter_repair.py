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
    restored_image: np.ndarray
    support_ratio: float
    candidate_pixels: int


class ProPainterLikeRestorer:
    """Flow-guided temporal filling with spatial fallback."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def restore_sequence(self, frames: list[np.ndarray], masks: list[np.ndarray]) -> list[RestorationArtifacts]:
        artifacts: list[RestorationArtifacts] = []
        for frame_index in range(len(frames)):
            temporal_fill, residual_mask, support_ratio, candidate_pixels = self._temporal_fill_frame(
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
            hard_mask = self._build_hard_mask(
                current_mask=masks[frame_index],
                residual_mask=residual_mask,
                support_ratio=support_ratio,
            )
            artifacts.append(
                RestorationArtifacts(
                    temporal_fill=temporal_fill,
                    residual_mask=residual_mask,
                    hard_mask=hard_mask,
                    restored_image=restored,
                    support_ratio=support_ratio,
                    candidate_pixels=candidate_pixels,
                )
            )
        return artifacts

    def _temporal_fill_frame(
        self,
        frame_index: int,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        current_frame = frames[frame_index]
        current_mask = masks[frame_index] > 0
        if not np.any(current_mask):
            return current_frame.copy(), np.zeros_like(masks[frame_index]), 1.0, 0

        temporal_radius = int(self.config.get("temporal_radius", 4))
        min_neighbors = int(self.config.get("min_temporal_neighbors", 2))
        weighted_sum = np.zeros_like(current_frame, dtype=np.float32)
        total_weight = np.zeros(current_mask.shape, dtype=np.float32)
        support_count = np.zeros(current_mask.shape, dtype=np.uint8)

        for offset in range(1, temporal_radius + 1):
            for neighbor_index in (frame_index - offset, frame_index + offset):
                if not 0 <= neighbor_index < len(frames):
                    continue
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
        support_ratio = float(np.count_nonzero(fillable)) / float(np.count_nonzero(current_mask))
        return temporal_fill, residual_mask, support_ratio, int(np.count_nonzero(fillable))

    def _build_hard_mask(self, current_mask: np.ndarray, residual_mask: np.ndarray, support_ratio: float) -> np.ndarray:
        hard_mask = residual_mask.copy()
        if support_ratio < float(self.config.get("hard_region_support_ratio", 0.55)):
            kernel_size = int(self.config.get("hard_region_dilation", 11))
            expanded = cv2.dilate(
                current_mask.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
                iterations=1,
            )
            hard_mask = np.maximum(hard_mask, expanded)
        return (hard_mask > 0).astype(np.uint8) * 255

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
