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

        context_kernel = int(self.config.get("mask_expand_kernel", 17))
        context_mask = cv2.dilate(
            hard_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (context_kernel, context_kernel)),
            iterations=1,
        )
        replace_kernel = int(self.config.get("replace_expand_kernel", 5))
        replace_mask = cv2.dilate(
            hard_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (replace_kernel, replace_kernel)),
            iterations=1,
        )
        candidate = cv2.inpaint(
            restored_frame,
            context_mask,
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
            (replace_mask > 0).astype(np.float32),
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

    def _warp_tensor_to_target(
        self,
        source_frame: np.ndarray,
        target_frame: np.ndarray,
        source_tensor: np.ndarray,
        source_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        flow = self._compute_flow(source_frame, target_frame)
        height, width = flow.shape[:2]
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

    def _lazy_load_raft(self):
        if hasattr(self, "_raft_model"):
            return self._raft_model, getattr(self, "_raft_device", "cpu")

        import sys
        import os
        import torch

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        propainter_dir = self.config.get("propainter_repo_dir", "ProPainter")
        raft_dir = os.path.join(project_root, propainter_dir)

        if raft_dir not in sys.path:
            sys.path.insert(0, raft_dir)

        try:
            from model.modules.flow_comp_raft import initialize_RAFT
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_path = os.path.join(raft_dir, "weights", "raft-things.pth")

            raft_model = initialize_RAFT(model_path, device=device)
            raft_model.eval()
            self._raft_model = raft_model
            self._raft_device = device
            return raft_model, device
        except Exception as e:
            print(f"Failed to load RAFT: {e}")
            self._raft_model = None
            self._raft_device = None
            return None, None

    def _compute_flow(self, source_frame: np.ndarray, target_frame: np.ndarray) -> np.ndarray:
        raft_model, device = None, None
        if raft_model is not None:
            import torch
            import cv2
            
            t_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
            s_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)

            h, w = t_rgb.shape[:2]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8

            if pad_h > 0 or pad_w > 0:
                t_rgb = cv2.copyMakeBorder(t_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
                s_rgb = cv2.copyMakeBorder(s_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

            t_tensor = (torch.from_numpy(t_rgb).permute(2, 0, 1).float() / 255.0) * 2.0 - 1.0
            s_tensor = (torch.from_numpy(s_rgb).permute(2, 0, 1).float() / 255.0) * 2.0 - 1.0

            t_tensor = t_tensor.unsqueeze(0).to(device)
            s_tensor = s_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                # target->source backward logic: feed target as image1, source as image2
                _, flows_forward = raft_model(t_tensor, s_tensor, iters=20, test_mode=True)

            flow_np = flows_forward[0].permute(1, 2, 0).cpu().numpy()
            
            if pad_h > 0 or pad_w > 0:
                flow_np = flow_np[:h, :w, :]
                
            return flow_np
        else:
            # Fallback
            import cv2
            target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
            source_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
            return cv2.calcOpticalFlowFarneback(
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
