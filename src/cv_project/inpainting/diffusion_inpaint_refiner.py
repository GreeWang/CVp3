from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cv_project.inpainting.diffusion_enhancer import DiffusionLikeEnhancer


class DiffusionInpaintRefiner(DiffusionLikeEnhancer):
    """Keyframe refiner backed by a real Diffusers inpainting pipeline."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_id = str(config.get("model_id", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"))
        self.device = str(config.get("device", "cuda:0"))
        self.dtype_name = str(config.get("dtype", "float16"))

        try:
            import torch
            from diffusers import AutoPipelineForInpainting
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "DiffusionInpaintRefiner requires diffusers, transformers, accelerate, safetensors, pillow, and torch. "
                "Install the updated requirements before using diffusion.method=sdxl_inpaint."
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"DiffusionInpaintRefiner is configured for {self.device}, but CUDA is not available.")

        self.torch = torch
        self.Image = Image
        self.dtype = self._resolve_torch_dtype(torch, self.dtype_name)

        load_kwargs: dict[str, object] = {"torch_dtype": self.dtype}
        variant = config.get("variant")
        if variant is None and self.dtype is torch.float16:
            variant = "fp16"
        if variant:
            load_kwargs["variant"] = str(variant)

        self.pipe = AutoPipelineForInpainting.from_pretrained(self._resolve_model_id(self.model_id), **load_kwargs)
        self.pipe = self.pipe.to(self.device)

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

        enhanced = restored_frame.copy()
        enhanced[y1:y2, x1:x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)
        return enhanced

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
        normalized = (mask > 0).astype(np.float32)
        if "feather_radius" in self.config:
            sigma = max(float(self.config.get("feather_radius", 11)) / 2.0, 0.1)
        else:
            sigma = float(self.config.get("feather_sigma", 5.0))
        if sigma > 0.0:
            normalized = cv2.GaussianBlur(normalized, (0, 0), sigmaX=sigma)
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized[..., None]
