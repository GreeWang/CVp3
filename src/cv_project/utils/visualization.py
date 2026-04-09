from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cv_project.pipeline.types import DetectionRecord


def overlay_detections(image: np.ndarray, detections: list[DetectionRecord], alpha: float = 0.45) -> np.ndarray:
    canvas = image.copy()
    tint = image.copy()
    for detection in detections:
        color = _color_for_class(detection.class_name)
        tint[detection.mask > 0] = color
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.score:.2f}"
        if detection.motion_score is not None:
            label += f" m={detection.motion_score:.2f}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(tint, alpha, canvas, 1 - alpha, 0.0)


def overlay_mask_contours(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    output = image.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, 2)
    return output


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def annotate(image: np.ndarray, title: str, font_scale: float, font_thickness: int) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(output, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return output


def create_comparison_panel(
    original: np.ndarray,
    raw_overlay: np.ndarray,
    dynamic_mask: np.ndarray,
    final_mask: np.ndarray,
    restored: np.ndarray,
    font_scale: float,
    font_thickness: int,
) -> np.ndarray:
    tiles = [
        annotate(original, "Original", font_scale, font_thickness),
        annotate(raw_overlay, "Raw Overlay", font_scale, font_thickness),
        annotate(mask_to_bgr(dynamic_mask), "Dynamic Mask", font_scale, font_thickness),
        annotate(mask_to_bgr(final_mask), "Final Mask", font_scale, font_thickness),
        annotate(restored, "Restored", font_scale, font_thickness),
    ]
    return np.concatenate(tiles, axis=1)


def save_report_frames(panel_paths: list[Path], output_dir: Path, requested_count: int, requested_indices: list[int]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not panel_paths:
        return []
    selected: list[Path] = []
    if requested_indices:
        selected = [panel_paths[index] for index in requested_indices if 0 <= index < len(panel_paths)]
    else:
        count = min(requested_count, len(panel_paths))
        indices = np.linspace(0, len(panel_paths) - 1, num=count, dtype=int)
        selected = [panel_paths[index] for index in indices]

    copied_paths: list[str] = []
    for panel_path in selected:
        image = cv2.imread(str(panel_path))
        if image is None:
            continue
        target = output_dir / panel_path.name
        cv2.imwrite(str(target), image)
        copied_paths.append(str(target))
    return copied_paths


def _color_for_class(class_name: str) -> tuple[int, int, int]:
    seed = abs(hash(class_name)) % (256 ** 3)
    return (seed % 255, (seed // 255) % 255, (seed // (255 * 255)) % 255)
