from __future__ import annotations

import cv2
import numpy as np

from cv_project.pipeline.types import DetectionRecord


class OpticalFlowDynamicFilter:
    def __init__(self, config: dict) -> None:
        self.config = config

    @staticmethod
    def _bbox_iou(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def score_detection(self, current_frame: np.ndarray, next_frame: np.ndarray, detection: DetectionRecord) -> tuple[float, int]:
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(
            current_gray,
            mask=detection.mask.astype(np.uint8),
            maxCorners=int(self.config["max_corners"]),
            qualityLevel=float(self.config["quality_level"]),
            minDistance=float(self.config["min_distance"]),
            blockSize=int(self.config["block_size"]),
        )
        if points is None or len(points) == 0:
            return 0.0, 0

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            next_gray,
            points,
            None,
            winSize=(int(self.config["lk_win_size"]), int(self.config["lk_win_size"])),
            maxLevel=int(self.config["lk_max_level"]),
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(self.config["lk_max_iterations"]),
                float(self.config["lk_epsilon"]),
            ),
        )
        if next_points is None or status is None:
            return 0.0, 0

        valid = status.reshape(-1) == 1
        if not np.any(valid):
            return 0.0, 0

        original = points.reshape(-1, 2)[valid]
        tracked = next_points.reshape(-1, 2)[valid]
        distances = np.linalg.norm(tracked - original, axis=1)
        return float(np.median(distances)), int(valid.sum())

    def apply(self, frames: list[np.ndarray], detections_per_frame: list[list[DetectionRecord]]) -> list[list[DetectionRecord]]:
        for index, detections in enumerate(detections_per_frame):
            if index >= len(frames) - 1:
                for detection in detections:
                    detection.motion_score = 0.0
                    detection.valid_points = 0
                    detection.is_dynamic = False
                continue

            for detection in detections:
                motion_score, valid_points = self.score_detection(frames[index], frames[index + 1], detection)
                detection.motion_score = motion_score
                detection.valid_points = valid_points
                detection.is_dynamic = (
                    valid_points >= int(self.config["min_valid_points"])
                    and motion_score >= float(self.config["displacement_threshold"])
                )

        filtered: list[list[DetectionRecord]] = []
        half_window = int(self.config["persistence_window"]) // 2
        required_votes = int(self.config["persistence_votes"])
        min_iou = float(self.config["persistence_iou_threshold"])
        for frame_index, detections in enumerate(detections_per_frame):
            frame_kept: list[DetectionRecord] = []
            for detection in detections:
                votes = 1 if detection.is_dynamic else 0
                for neighbor_index in range(max(0, frame_index - half_window), min(len(detections_per_frame), frame_index + half_window + 1)):
                    if neighbor_index == frame_index:
                        continue
                    for neighbor_detection in detections_per_frame[neighbor_index]:
                        if not neighbor_detection.is_dynamic:
                            continue
                        if neighbor_detection.class_name != detection.class_name:
                            continue
                        if self._bbox_iou(detection.bbox, neighbor_detection.bbox) >= min_iou:
                            votes += 1
                            break
                detection.is_dynamic = bool(detection.is_dynamic and votes >= required_votes)
                if detection.is_dynamic:
                    frame_kept.append(detection)
            filtered.append(frame_kept)
        return filtered
