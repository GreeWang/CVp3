from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from cv_project.inpainting.diffusion_enhancer import DiffusionLikeEnhancer


class _NoOpTextImageReward:
    def process_text(self, prompt: object) -> object:
        return prompt

    def __call__(self, text_input: object, image: object, mask: object = None) -> object:
        # Keep graph connectivity while producing zero guidance.
        return image.mean() * 0.0


class _NoOpHarmonicReward:
    def __call__(self, image: object, mask: object) -> tuple[object, object]:
        # Keep graph connectivity while producing zero guidance.
        return image.mean() * 0.0, None


class FreeInpaintRefiner(DiffusionLikeEnhancer):
    """Keyframe refiner backed by FreeInpaint's SDXL inpaint pipeline."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_id = str(config.get("model_id", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"))
        self.device = str(config.get("device", "cuda:0"))
        self.dtype_name = str(config.get("dtype", "float16"))

        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "FreeInpaintRefiner requires torch and pillow. "
                "Install project requirements first."
            ) from exc

        freeinpaint_repo = Path(str(config.get("freeinpaint_repo_dir", "FreeInpaint-main")))
        if not freeinpaint_repo.is_absolute():
            freeinpaint_repo = Path.cwd() / freeinpaint_repo

        if not freeinpaint_repo.exists():
            raise FileNotFoundError(
                f"FreeInpaint repository not found at {freeinpaint_repo}. "
                "Set diffusion.freeinpaint_repo_dir to the correct path."
            )

        if str(freeinpaint_repo) not in sys.path:
            sys.path.insert(0, str(freeinpaint_repo))

        try:
            from examples.freeinpaint.pipe.pipeline_stable_diffusion_xl_inpaint_optno_guidance import (
                StableDiffusionXLInpaintOptNoGuidancePipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "Unable to import FreeInpaint pipeline class. "
                "Ensure FreeInpaint dependencies are installed (for example: pip install -e FreeInpaint-main)."
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"FreeInpaintRefiner is configured for {self.device}, but CUDA is not available.")

        self.torch = torch
        self.Image = Image
        self.dtype = self._resolve_torch_dtype(torch, self.dtype_name)

        load_kwargs: dict[str, object] = {"torch_dtype": self.dtype}
        variant = config.get("variant")
        if variant is None and self.dtype is torch.float16:
            variant = "fp16"
        if variant:
            load_kwargs["variant"] = str(variant)

        self.pipe = StableDiffusionXLInpaintOptNoGuidancePipeline.from_pretrained(
            self._resolve_model_id(self.model_id),
            **load_kwargs,
        )
        self.pipe = self.pipe.to(self.device)

        # FreeInpaint pipelines rely on these runtime attributes in __call__.
        self.pipe.max_round_initno = int(config.get("max_round_initno", 3))
        self.pipe.self_attn_loss_scale = float(config.get("self_attn_loss_scale", 0.0))
        self.pipe.opt_noise_steps = int(config.get("opt_noise_steps", 0))
        self.pipe.guide_per_steps = int(config.get("guide_per_steps", 1))
        self.pipe.attn_res_scale_factor = float(config.get("attn_res_scale_factor", 32.0))
        self.pipe.initno_lr = float(config.get("initno_lr", 0.01))

        self.pipe.overall_reward = _NoOpTextImageReward()
        self.pipe.prompt_reward = _NoOpTextImageReward()
        self.pipe.harmonic_reward = _NoOpHarmonicReward()

        self.pipe.reward_guidance_scale = float(config.get("reward_guidance_scale", 0.0))
        self.pipe.overall_reward_scale = float(config.get("overall_reward_scale", 0.0))
        self.pipe.prompt_reward_scale = float(config.get("prompt_reward_scale", 0.0))
        self.pipe.harmonic_reward_scale = float(config.get("harmonic_reward_scale", 0.0))

        if bool(config.get("enable_attention_slicing", True)) and hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if bool(config.get("enable_vae_slicing", True)) and hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()

    @staticmethod
    def _resolve_model_id(model_id: str) -> str:
        path = Path(model_id)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return str(path)
        return model_id

    @staticmethod
    def _resolve_torch_dtype(torch_module: object, dtype_name: str) -> object:
        normalized = dtype_name.lower()
        if normalized in {"float16", "fp16", "half"}:
            return torch_module.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch_module.bfloat16
        if normalized in {"float32", "fp32", "full"}:
            return torch_module.float32
        raise ValueError(f"Unsupported diffusion dtype: {dtype_name}")

    def _enhance_keyframe(self, original_frame: np.ndarray, restored_frame: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
        if not np.any(hard_mask > 0):
            return restored_frame.copy()

        hard_binary = (hard_mask > 0).astype(np.uint8) * 255
        min_area_ratio = float(self.config.get("min_hard_area_ratio", 0.0))
        if min_area_ratio > 0.0 and np.count_nonzero(hard_binary) / hard_binary.size < min_area_ratio:
            return restored_frame.copy()

        x1, y1, x2, y2 = self._select_roi(hard_binary, restored_frame.shape[:2])
        roi_bgr = restored_frame[y1:y2, x1:x2].copy()
        roi_mask = hard_binary[y1:y2, x1:x2].copy()
        if roi_bgr.size == 0 or not np.any(roi_mask > 0):
            return restored_frame.copy()

        roi_mask = self._expand_mask(roi_mask)
        original_roi_size = (roi_bgr.shape[1], roi_bgr.shape[0])
        model_bgr, model_mask = self._resize_for_model(roi_bgr, roi_mask)

        pil_image = self.Image.fromarray(cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
        pil_mask = self.Image.fromarray(model_mask).convert("L")

        result = self._run_inpaint(pil_image, pil_mask)
        generated_rgb = np.asarray(result.convert("RGB"))
        generated_bgr = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR)
        if (generated_bgr.shape[1], generated_bgr.shape[0]) != original_roi_size:
            generated_bgr = cv2.resize(generated_bgr, original_roi_size, interpolation=cv2.INTER_CUBIC)

        alpha = self._build_feather_alpha(roi_mask)
        blended_roi = roi_bgr.astype(np.float32) * (1.0 - alpha) + generated_bgr.astype(np.float32) * alpha
        blended_roi = self._apply_edge_color_match(
            blended_roi=blended_roi,
            roi_bgr=roi_bgr,
            roi_mask=roi_mask,
        )
        blended_roi = self._apply_edge_convolution_balance(
            blended_roi=blended_roi,
            roi_mask=roi_mask,
        )

        enhanced = restored_frame.copy()
        enhanced[y1:y2, x1:x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)
        return enhanced

    def _apply_edge_convolution_balance(self, *, blended_roi: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """Smooth only the seam band to reduce edge halos without blurring the full ROI."""
        if not bool(self.config.get("enable_edge_convolution_balance", False)):
            return blended_roi

        mask_binary = (roi_mask > 0).astype(np.uint8)
        if not np.any(mask_binary):
            return blended_roi

        edge_band_width = max(1, int(self.config.get("edge_band_width", 8)))
        edge_blend_alpha = float(np.clip(float(self.config.get("edge_blend_alpha", 0.6)), 0.0, 1.0))

        # Boundary ring: slightly expanded mask minus eroded core.
        dilated = cv2.dilate(
            mask_binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_band_width, edge_band_width)),
            iterations=1,
        )
        erode_size = max(1, edge_band_width // 2)
        eroded = cv2.erode(
            mask_binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size)),
            iterations=1,
        )
        seam_band = ((dilated > 0) & (eroded == 0)).astype(np.uint8)
        if not np.any(seam_band):
            return blended_roi

        smooth_kernel = int(self.config.get("edge_smooth_kernel", 5))
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        smooth_kernel = max(3, smooth_kernel)
        sigma_color = float(self.config.get("edge_sigma_color", 12.0))
        sigma_space = float(self.config.get("edge_sigma_space", 18.0))

        roi_u8 = np.clip(blended_roi, 0, 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(
            roi_u8,
            d=smooth_kernel,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
        ).astype(np.float32)

        out = blended_roi.copy()
        seam_idx = seam_band > 0
        out[seam_idx] = out[seam_idx] * (1.0 - edge_blend_alpha) + smoothed[seam_idx] * edge_blend_alpha
        return out

    def _apply_edge_color_match(self, *, blended_roi: np.ndarray, roi_bgr: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """Align seam-band color statistics to original ROI to suppress bright/dark halos."""
        if not bool(self.config.get("enable_edge_color_match", False)):
            return blended_roi

        mask_binary = (roi_mask > 0).astype(np.uint8)
        if not np.any(mask_binary):
            return blended_roi

        band_width = max(1, int(self.config.get("edge_color_band_width", 6)))
        blend_alpha = float(np.clip(float(self.config.get("edge_color_match_alpha", 0.7)), 0.0, 1.0))

        # Use a boundary ring near the seam, excluding the center region.
        dilated = cv2.dilate(
            mask_binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width, band_width)),
            iterations=1,
        )
        erode_size = max(1, band_width // 2)
        eroded = cv2.erode(
            mask_binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size)),
            iterations=1,
        )
        seam_band = ((dilated > 0) & (eroded == 0))
        if not np.any(seam_band):
            return blended_roi

        out = blended_roi.copy()
        source = out.reshape(-1, 3)
        target = roi_bgr.astype(np.float32).reshape(-1, 3)
        idx = seam_band.reshape(-1)

        src_vals = source[idx]
        tgt_vals = target[idx]
        if src_vals.shape[0] < 8:
            return out

        src_mean = src_vals.mean(axis=0)
        src_std = src_vals.std(axis=0)
        tgt_mean = tgt_vals.mean(axis=0)
        tgt_std = tgt_vals.std(axis=0)

        eps = 1e-6
        matched = (src_vals - src_mean) * (tgt_std + eps) / (src_std + eps) + tgt_mean
        matched = np.clip(matched, 0.0, 255.0)

        source[idx] = source[idx] * (1.0 - blend_alpha) + matched * blend_alpha
        return source.reshape(out.shape)

    def _select_roi(self, mask: np.ndarray, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        height, width = frame_shape
        if not bool(self.config.get("use_roi_crop", True)):
            return 0, 0, width, height

        ys, xs = np.where(mask > 0)
        pad = int(self.config.get("roi_padding", 32))
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(width, int(xs.max()) + pad + 1)
        y2 = min(height, int(ys.max()) + pad + 1)
        return x1, y1, x2, y2

    def _expand_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel_size = int(self.config.get("mask_expand_kernel", 17))
        if kernel_size <= 1:
            return (mask > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate((mask > 0).astype(np.uint8) * 255, kernel, iterations=1)

    def _resize_for_model(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        max_side = int(self.config.get("max_side", 1024))
        align_to = int(self.config.get("size_multiple", 8))
        align_to = max(1, align_to)

        scale = 1.0
        if max_side > 0 and max(width, height) > max_side:
            scale = max_side / float(max(width, height))

        target_width = max(align_to, int(np.floor((width * scale) / align_to) * align_to))
        target_height = max(align_to, int(np.floor((height * scale) / align_to) * align_to))
        if target_width == width and target_height == height:
            return image, mask

        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        resized_mask = (resized_mask > 127).astype(np.uint8) * 255
        return resized_image, resized_mask

    def _run_inpaint(self, pil_image: object, pil_mask: object) -> object:
        generator = None
        seed = self.config.get("seed")
        if seed is not None:
            generator = self.torch.Generator(device=self.device).manual_seed(int(seed))

        kwargs: dict[str, object] = {
            "prompt": str(self.config.get("prompt", "clean natural background, realistic texture, consistent with surrounding scene, no object")),
            "negative_prompt": str(
                self.config.get("negative_prompt", "object, person, artifact, blurry, distorted, duplicated, text, watermark")
            ),
            "image": pil_image,
            "mask_image": pil_mask,
            "num_inference_steps": int(self.config.get("num_inference_steps", 30)),
            "guidance_scale": float(self.config.get("guidance_scale", 7.0)),
        }
        if "strength" in self.config:
            kwargs["strength"] = float(self.config["strength"])
        if generator is not None:
            kwargs["generator"] = generator

        with self.torch.inference_mode():
            return self.pipe(**kwargs).images[0]

    def _build_feather_alpha(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        feather_radius = float(self.config.get("feather_radius", 11.0))
        
        if feather_radius <= 0.0:
            return binary.astype(np.float32)[..., None]
            
        # Use Distance Transform to create a linear gradient from the boundary inward.
        # This completely avoids the "hard cut" issue when preserve_core_kernel is 0,
        # ensuring 0.0 alpha at the boundary to eliminate VAE artifacts (white edges).
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        alpha = np.clip(dist / feather_radius, 0.0, 1.0)
        return alpha[..., None]
