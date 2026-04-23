from __future__ import annotations

import contextlib
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from cv_project.pipeline.types import DetectionRecord, FrameRecord
from cv_project.segmentation.yolo_segmenter import YoloSegmenter

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class Sam2VideoSegmenter:
    """Official SAM2 video propagation with YOLO box prompts on a seed frame."""

    def __init__(self, config: dict, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.repo_dir = self._resolve_path(config.get("sam2_repo_dir", "sam2"))
        self.model_cfg = str(config.get("sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml"))
        self.checkpoint_path = self._resolve_checkpoint(config.get("sam2_checkpoint"))
        self.device = self._resolve_runtime_device(str(config.get("sam2_device") or config.get("device", "cpu")))
        prompt_device = self._resolve_runtime_device(
            str(config.get("prompt_detector_device", config.get("device", "cpu")))
        )
        self.prompt_frame_idx = int(config.get("prompt_frame_idx", 0))
        self.max_initial_objects = int(config.get("max_initial_objects", 6))
        self.min_mask_area_ratio = float(config.get("min_mask_area_ratio", 0.00005))
        self.use_autocast = bool(config.get("sam2_use_autocast", True))
        self.vos_optimized = bool(config.get("sam2_vos_optimized", False))
        self.prompt_detector = YoloSegmenter(
            model_name=str(config["model_name"]),
            device=prompt_device,
            confidence_threshold=float(config["confidence_threshold"]),
            iou_threshold=float(config["iou_threshold"]),
            dynamic_classes=list(config["dynamic_classes"]),
        )

    @property
    def backend_name(self) -> str:
        return "sam2+box-prompts"

    def segment_video(self, frame_records: list[FrameRecord], sam2_frame_dir: Path) -> list[list[DetectionRecord]]:
        if not frame_records:
            return []

        prompt_frame_idx = min(max(0, self.prompt_frame_idx), len(frame_records) - 1)
        prompt_frame = frame_records[prompt_frame_idx].image
        prompt_detections = self.prompt_detector.predict(prompt_frame, prompt_frame_idx)
        if not prompt_detections:
            return [[] for _ in frame_records]

        prompt_detections = sorted(
            prompt_detections,
            key=lambda det: (det.score, np.count_nonzero(det.mask)),
            reverse=True,
        )[: self.max_initial_objects]

        build_sam2_video_predictor, torch = self._load_sam2_symbols()
        predictor = build_sam2_video_predictor(
            self.model_cfg,
            ckpt_path=str(self.checkpoint_path),
            device=self.device,
            vos_optimized=self.vos_optimized,
        )

        detections_per_frame: list[list[DetectionRecord]] = [[] for _ in frame_records]
        detections_per_frame[prompt_frame_idx] = [
            replace(
                detection,
                instance_id=f"sam2_{obj_id:03d}_{prompt_frame_idx:06d}",
            )
            for obj_id, detection in enumerate(prompt_detections, start=1)
        ]
        with torch.inference_mode(), self._autocast_context(torch):
            inference_state = predictor.init_state(video_path=str(sam2_frame_dir))
            prompt_meta: dict[int, DetectionRecord] = {}
            for obj_id, detection in enumerate(prompt_detections, start=1):
                x1, y1, x2, y2 = detection.bbox
                predictor.add_new_points_or_box(
                    inference_state,
                    frame_idx=prompt_frame_idx,
                    obj_id=obj_id,
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                )
                prompt_meta[obj_id] = detection

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                frame_idx = int(out_frame_idx)
                if not 0 <= frame_idx < len(frame_records):
                    continue
                frame_detections: list[DetectionRecord] = []
                for local_index, obj_id in enumerate(out_obj_ids):
                    prompt_detection = prompt_meta.get(int(obj_id))
                    if prompt_detection is None:
                        continue
                    mask = (out_mask_logits[local_index] > 0).detach().cpu().numpy().astype(np.uint8)
                    if mask.ndim == 3:
                        mask = mask.squeeze(0)
                    mask = mask.astype(np.uint8) * 255
                    if np.count_nonzero(mask) < mask.size * self.min_mask_area_ratio:
                        continue
                    bbox = self._mask_to_bbox(mask)
                    frame_detections.append(
                        replace(
                            prompt_detection,
                            instance_id=f"sam2_{int(obj_id):03d}_{frame_idx:06d}",
                            mask=mask,
                            bbox=bbox,
                        )
                    )
                detections_per_frame[frame_idx] = frame_detections

        return detections_per_frame

    def _load_sam2_symbols(self):
        sam2_parent = str(self.repo_dir)
        if sam2_parent not in sys.path:
            sys.path.insert(0, sam2_parent)
        try:
            from sam2.build_sam import build_sam2_video_predictor
            import torch
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "Failed to import local SAM2 repo. Install SAM2 dependencies in the current environment."
            ) from exc
        return build_sam2_video_predictor, torch

    def _autocast_context(self, torch_module):
        if self.device.startswith("cuda") and self.use_autocast:
            return torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
        return contextlib.nullcontext()

    def _resolve_path(self, value: str | None) -> Path:
        if not value:
            raise ValueError("Missing required path value.")
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _resolve_checkpoint(self, checkpoint_value: str | None) -> Path:
        if not checkpoint_value:
            raise FileNotFoundError(
                "SAM2 checkpoint is not configured. Set segmentation.sam2_checkpoint in the YAML config."
            )
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.repo_dir / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {checkpoint_path}. Download a checkpoint into sam2/checkpoints and point the config to it."
            )
        return checkpoint_path

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return (0, 0, 0, 0)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _resolve_runtime_device(device: str) -> str:
        normalized = str(device or "cpu")
        if normalized.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
            return "cpu"
        return normalized
