from __future__ import annotations

import cv2
import numpy as np

from cv_project.pipeline.types import DetectionRecord

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


class YoloSegmenter:
    def __init__(self, model_name: str, device: str, confidence_threshold: float, iou_threshold: float, dynamic_classes: list[str]) -> None:
        if YOLO is None:
            raise ImportError("ultralytics is required to run segmentation. Install dependencies from requirements.txt.")
        self.model = YOLO(model_name)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.dynamic_classes = set(dynamic_classes)
        self.class_names = self.model.names

    def predict(self, image: np.ndarray, frame_index: int) -> list[DetectionRecord]:
        results = self.model.predict(
            source=image,
            device=self.device,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
            retina_masks=True,
        )
        if not results:
            return []
        result = results[0]
        if result.boxes is None or result.masks is None:
            return []

        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(int)
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.detach().cpu().numpy()
        mask_data = result.masks.data.detach().cpu().numpy()
        height, width = image.shape[:2]

        detections: list[DetectionRecord] = []
        for idx, (bbox, class_id, score, mask_logits) in enumerate(zip(boxes_xyxy, classes, scores, mask_data)):
            class_name = self.class_names[int(class_id)]
            if class_name not in self.dynamic_classes:
                continue
            resized_mask = cv2.resize(mask_logits.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255
            detections.append(
                DetectionRecord(
                    instance_id=f"{frame_index:06d}_{idx:03d}",
                    class_name=class_name,
                    score=float(score),
                    mask=binary_mask,
                    bbox=tuple(int(v) for v in bbox),
                )
            )
        return detections
