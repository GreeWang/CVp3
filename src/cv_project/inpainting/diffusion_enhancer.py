from __future__ import annotations

import cv2
import numpy as np


class DiffusionLikeEnhancer:
    """Keyframe-only local enhancement with flow-based propagation."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def enhance_sequence(
        self,
        original_frames: list[np.ndarray],
        restored_frames: list[np.ndarray],
        hard_masks: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[int]]:
        enhanced_frames = [frame.copy() for frame in restored_frames]
        keyframes = self._select_keyframes(hard_masks)
        for keyframe_index in keyframes:
            enhanced_keyframe = self._enhance_keyframe(
                original_frame=original_frames[keyframe_index],
                restored_frame=restored_frames[keyframe_index],
                hard_mask=hard_masks[keyframe_index],
            )
            enhanced_frames[keyframe_index] = enhanced_keyframe
            self._propagate_from_keyframe(
                keyframe_index=keyframe_index,
                enhanced_frames=enhanced_frames,
                restored_frames=restored_frames,
                hard_masks=hard_masks,
            )
        return enhanced_frames, keyframes

    def _select_keyframes(self, hard_masks: list[np.ndarray]) -> list[int]:
        ratio_threshold = float(self.config.get("hard_ratio_threshold", 0.01))
        min_gap = int(self.config.get("min_keyframe_gap", 6))
        max_keyframes = int(self.config.get("max_keyframes", 12))
        selected: list[int] = []
        for index, hard_mask in enumerate(hard_masks):
            hard_ratio = float(np.count_nonzero(hard_mask)) / float(hard_mask.size)
            if hard_ratio < ratio_threshold:
                continue
            if selected and index - selected[-1] < min_gap:
                continue
            selected.append(index)
            if len(selected) >= max_keyframes:
                break
        return selected

    def _enhance_keyframe(self, original_frame: np.ndarray, restored_frame: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
        if not np.any(hard_mask > 0):
            return restored_frame.copy()

        kernel_size = int(self.config.get("mask_expand_kernel", 17))
        expanded_mask = cv2.dilate(
            hard_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=1,
        )
        candidate = cv2.inpaint(
            restored_frame,
            expanded_mask,
            int(self.config.get("inpaint_radius", 5)),
            cv2.INPAINT_TELEA,
        )
        candidate = cv2.edgePreservingFilter(
            candidate,
            flags=1,
            sigma_s=float(self.config.get("edge_sigma_s", 40)),
            sigma_r=float(self.config.get("edge_sigma_r", 0.25)),
        )
        candidate = cv2.detailEnhance(
            candidate,
            sigma_s=float(self.config.get("detail_sigma_s", 10)),
            sigma_r=float(self.config.get("detail_sigma_r", 0.15)),
        )

        original_hint = cv2.GaussianBlur(original_frame, (0, 0), sigmaX=1.0)
        candidate = cv2.addWeighted(candidate, 0.85, original_hint, 0.15, 0.0)
        feather = cv2.GaussianBlur(
            (expanded_mask > 0).astype(np.float32),
            (0, 0),
            sigmaX=float(self.config.get("feather_sigma", 5.0)),
        )
        feather = np.clip(feather[..., None], 0.0, 1.0)
        blended = restored_frame.astype(np.float32) * (1.0 - feather) + candidate.astype(np.float32) * feather
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _propagate_from_keyframe(
        self,
        keyframe_index: int,
        enhanced_frames: list[np.ndarray],
        restored_frames: list[np.ndarray],
        hard_masks: list[np.ndarray],
    ) -> None:
        propagation_radius = int(self.config.get("propagation_radius", 3))
        propagation_blend = float(self.config.get("propagation_blend", 0.45))
        keyframe_delta = enhanced_frames[keyframe_index].astype(np.float32) - restored_frames[keyframe_index].astype(np.float32)

        for offset in range(1, propagation_radius + 1):
            for target_index in (keyframe_index - offset, keyframe_index + offset):
                if not 0 <= target_index < len(enhanced_frames):
                    continue
                if not np.any(hard_masks[target_index] > 0):
                    continue
                warped_delta, warped_mask = self._warp_tensor_to_target(
                    source_frame=restored_frames[keyframe_index],
                    target_frame=restored_frames[target_index],
                    source_tensor=keyframe_delta,
                    source_mask=hard_masks[keyframe_index],
                )
                valid = (hard_masks[target_index] > 0) & (warped_mask > 0)
                if not np.any(valid):
                    continue
                current = enhanced_frames[target_index].astype(np.float32)
                candidate = np.clip(current + warped_delta, 0, 255)
                blend = propagation_blend / float(offset)
                current[valid] = current[valid] * (1.0 - blend) + candidate[valid] * blend
                enhanced_frames[target_index] = np.clip(current, 0, 255).astype(np.uint8)

    @staticmethod
    def _warp_tensor_to_target(
        source_frame: np.ndarray,
        target_frame: np.ndarray,
        source_tensor: np.ndarray,
        source_mask: np.ndarray,
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
        warped_tensor = cv2.remap(
            source_tensor.astype(np.float32),
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
            borderValue=0,
        )
        return warped_tensor, warped_mask
