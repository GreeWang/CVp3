from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from cv_project.pipeline.types import DetectionRecord
from cv_project.segmentation.yolo_segmenter import YoloSegmenter


class PromptedMaskGenerator:
    """A runnable SAM2-style substitute built on detector prompts plus mask refinement."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.base_segmenter = YoloSegmenter(
            model_name=str(config["model_name"]),
            device=str(config["device"]),
            confidence_threshold=float(config["confidence_threshold"]),
            iou_threshold=float(config["iou_threshold"]),
            dynamic_classes=list(config["dynamic_classes"]),
        )
        self.refinement_backend = str(config.get("refinement_backend", "grabcut"))
        self.grabcut_iterations = int(config.get("grabcut_iterations", 2))
        self.bbox_padding = float(config.get("bbox_padding_ratio", 0.08))
        self.min_refined_area_ratio = float(config.get("min_refined_area_ratio", 0.4))

    @property
    def backend_name(self) -> str:
        return f"detector+{self.refinement_backend}"

    def predict(self, image: np.ndarray, frame_index: int) -> list[DetectionRecord]:
        detections = self.base_segmenter.predict(image, frame_index)
        if self.refinement_backend == "none":
            return detections

        refined: list[DetectionRecord] = []
        for detection in detections:
            refined_mask = self._refine_mask(image, detection.mask, detection.bbox)
            refined_bbox = self._mask_to_bbox(refined_mask, fallback=detection.bbox)
            refined.append(replace(detection, mask=refined_mask, bbox=refined_bbox))
        return refined

    def _refine_mask(self, image: np.ndarray, coarse_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        if self.refinement_backend != "grabcut":
            return coarse_mask.copy()

        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(round(box_w * self.bbox_padding))
        pad_y = int(round(box_h * self.bbox_padding))
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(width - 1, x2 + pad_x)
        ry2 = min(height - 1, y2 + pad_y)

        rect = (rx1, ry1, max(1, rx2 - rx1), max(1, ry2 - ry1))
        init_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
        init_mask[ry1 : ry2 + 1, rx1 : rx2 + 1] = cv2.GC_PR_BGD

        coarse_binary = (coarse_mask > 0).astype(np.uint8) * 255
        eroded = cv2.erode(
            coarse_binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        dilated = cv2.dilate(
            coarse_binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        init_mask[dilated > 0] = cv2.GC_PR_FGD
        init_mask[eroded > 0] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(
                image,
                init_mask,
                rect,
                bgd_model,
                fgd_model,
                self.grabcut_iterations,
                cv2.GC_INIT_WITH_MASK,
            )
            refined = np.where(
                (init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)
        except cv2.error:
            refined = coarse_binary

        coarse_area = int(np.count_nonzero(coarse_binary))
        refined_area = int(np.count_nonzero(refined))
        if coarse_area == 0 or refined_area < coarse_area * self.min_refined_area_ratio:
            return coarse_binary
        return refined

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return fallback
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
