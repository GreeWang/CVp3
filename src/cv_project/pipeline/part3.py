from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from cv_project.data.io import (
    ensure_dir,
    extract_frames_from_video,
    list_frame_paths,
    normalize_frame_size,
    resolve_path,
    save_json,
    timestamp_now,
    write_video,
)
from cv_project.inpainting.diffusion_enhancer import DiffusionLikeEnhancer
from cv_project.inpainting.propainter_repair import ProPainterLikeRestorer
from cv_project.motion.dynamic_filter import OpticalFlowDynamicFilter
from cv_project.pipeline.types import FrameRecord
from cv_project.segmentation.prompted_mask_generator import PromptedMaskGenerator
from cv_project.utils.mask_ops import merge_instance_masks, refine_mask_sequence
from cv_project.utils.visualization import (
    annotate,
    mask_to_bgr,
    overlay_detections,
    overlay_mask_contours,
    save_report_frames,
)


def run_part3_pipeline(config: dict, project_root: Path) -> dict:
    started_at = time.time()
    dataset_name = config["output"]["dataset_name"]
    run_dir = ensure_dir(resolve_path(config["output"]["root_dir"], project_root) / dataset_name / timestamp_now())
    frames_dir = ensure_dir(run_dir / "frames")
    overlays_dir = ensure_dir(run_dir / "raw_overlays")
    raw_masks_dir = ensure_dir(run_dir / "raw_dynamic_masks")
    refined_masks_dir = ensure_dir(run_dir / "refined_masks")
    hard_masks_dir = ensure_dir(run_dir / "hard_masks")
    contour_dir = ensure_dir(run_dir / "mask_contours")
    temporal_fill_dir = ensure_dir(run_dir / "temporal_fill")
    restored_dir = ensure_dir(run_dir / "restored_frames")
    enhanced_dir = ensure_dir(run_dir / "enhanced_frames")
    panels_dir = ensure_dir(run_dir / "comparison_panels")
    figures_dir = ensure_dir(run_dir / "report_frames")
    keyframes_dir = ensure_dir(run_dir / "diffusion_keyframes")

    input_video_path = resolve_path(config["input"].get("video_path"), project_root)
    input_frames_dir = resolve_path(config["input"].get("frames_dir"), project_root)
    frame_width = int(config["output"]["frame_name_width"])
    max_long_side = int(config["input"]["max_long_side"])

    if input_frames_dir is not None:
        source_frame_paths = list_frame_paths(input_frames_dir, config["input"]["image_extensions"])
        fps = float(config["output"].get("save_fps") or 24.0)
    elif input_video_path is not None:
        source_frame_paths, fps = extract_frames_from_video(input_video_path, frames_dir, max_long_side, frame_width)
    else:
        raise ValueError("Provide either input.video_path or input.frames_dir in the config.")

    frame_records: list[FrameRecord] = []
    loaded_frames: list[np.ndarray] = []
    if input_frames_dir is not None:
        for index, source_path in enumerate(source_frame_paths):
            image = cv2.imread(str(source_path))
            if image is None:
                raise RuntimeError(f"Unable to read frame image: {source_path}")
            image = normalize_frame_size(image, max_long_side)
            normalized_path = frames_dir / f"{index:0{frame_width}d}.png"
            cv2.imwrite(str(normalized_path), image)
            frame_records.append(FrameRecord(frame_index=index, image=image, source_path=str(normalized_path)))
            loaded_frames.append(image)
    else:
        for index, source_path in enumerate(source_frame_paths):
            image = cv2.imread(str(source_path))
            if image is None:
                raise RuntimeError(f"Unable to read frame image: {source_path}")
            frame_records.append(FrameRecord(frame_index=index, image=image, source_path=str(source_path)))
            loaded_frames.append(image)

    segmenter = PromptedMaskGenerator(config["segmentation"])
    for record in tqdm(frame_records, desc="Prompted segmentation"):
        record.raw_detections = segmenter.predict(record.image, record.frame_index)
        raw_overlay = overlay_detections(record.image, record.raw_detections, alpha=float(config["visualization"]["mask_alpha"]))
        cv2.imwrite(str(overlays_dir / f"{record.frame_index:06d}.png"), raw_overlay)

    dynamic_filter = OpticalFlowDynamicFilter(config["motion"])
    filtered_detections = dynamic_filter.apply(loaded_frames, [record.raw_detections for record in frame_records])

    raw_dynamic_masks: list[np.ndarray] = []
    for record, filtered in zip(frame_records, filtered_detections):
        record.filtered_detections = filtered
        raw_mask = merge_instance_masks([det.mask for det in filtered], record.image.shape[:2])
        record.raw_dynamic_mask = raw_mask
        raw_dynamic_masks.append(raw_mask)
        cv2.imwrite(str(raw_masks_dir / f"{record.frame_index:06d}.png"), raw_mask)

    refined_masks = refine_mask_sequence(raw_dynamic_masks, loaded_frames, config["mask_postprocess"])
    for record, final_mask in zip(frame_records, refined_masks):
        record.final_mask = final_mask
        cv2.imwrite(str(refined_masks_dir / f"{record.frame_index:06d}.png"), final_mask)
        contour_overlay = overlay_mask_contours(record.image, final_mask)
        cv2.imwrite(str(contour_dir / f"{record.frame_index:06d}.png"), contour_overlay)

    propainter_like = ProPainterLikeRestorer(config["inpainting"])
    restoration_artifacts = propainter_like.restore_sequence(loaded_frames, refined_masks)

    restored_frames: list[np.ndarray] = []
    hard_masks: list[np.ndarray] = []
    total_temporal_filled_pixels = 0
    support_ratios: list[float] = []
    for record, artifacts in zip(frame_records, restoration_artifacts):
        record.temporal_fill = artifacts.temporal_fill
        record.restored_image = artifacts.restored_image
        record.metadata["support_ratio"] = artifacts.support_ratio
        support_ratios.append(artifacts.support_ratio)
        total_temporal_filled_pixels += artifacts.candidate_pixels
        restored_frames.append(artifacts.restored_image)
        hard_masks.append(artifacts.hard_mask)
        cv2.imwrite(str(temporal_fill_dir / f"{record.frame_index:06d}.png"), artifacts.temporal_fill)
        cv2.imwrite(str(restored_dir / f"{record.frame_index:06d}.png"), artifacts.restored_image)
        cv2.imwrite(str(hard_masks_dir / f"{record.frame_index:06d}.png"), artifacts.hard_mask)

    diffusion_like = DiffusionLikeEnhancer(config["diffusion"])
    enhanced_frames, keyframe_indices = diffusion_like.enhance_sequence(loaded_frames, restored_frames, hard_masks)

    panel_paths: list[Path] = []
    enhanced_frame_paths: list[Path] = []
    for record, enhanced_frame in zip(frame_records, enhanced_frames):
        enhanced_path = enhanced_dir / f"{record.frame_index:06d}.png"
        cv2.imwrite(str(enhanced_path), enhanced_frame)
        enhanced_frame_paths.append(enhanced_path)

        raw_overlay = cv2.imread(str(overlays_dir / f"{record.frame_index:06d}.png"))
        panel = _create_part3_panel(
            original=record.image,
            raw_overlay=raw_overlay if raw_overlay is not None else record.image,
            raw_mask=record.raw_dynamic_mask if record.raw_dynamic_mask is not None else np.zeros(record.image.shape[:2], dtype=np.uint8),
            refined_mask=record.final_mask if record.final_mask is not None else np.zeros(record.image.shape[:2], dtype=np.uint8),
            hard_mask=hard_masks[record.frame_index],
            restored=record.restored_image if record.restored_image is not None else record.image,
            enhanced=enhanced_frame,
            font_scale=float(config["visualization"]["panel_font_scale"]),
            font_thickness=int(config["visualization"]["panel_font_thickness"]),
        )
        panel_path = panels_dir / f"{record.frame_index:06d}.png"
        cv2.imwrite(str(panel_path), panel)
        panel_paths.append(panel_path)

    for keyframe_index in keyframe_indices:
        preview = _create_keyframe_preview(
            original=loaded_frames[keyframe_index],
            restored=restored_frames[keyframe_index],
            enhanced=enhanced_frames[keyframe_index],
            hard_mask=hard_masks[keyframe_index],
            font_scale=float(config["visualization"]["panel_font_scale"]),
            font_thickness=int(config["visualization"]["panel_font_thickness"]),
        )
        cv2.imwrite(str(keyframes_dir / f"{keyframe_index:06d}.png"), preview)

    output_fps = float(config["output"]["save_fps"] or fps or 24.0)
    restored_video_path = run_dir / "restored_propainter_like.mp4"
    final_video_path = run_dir / "restored_part3_enhanced.mp4"
    restored_frame_paths = [restored_dir / f"{index:06d}.png" for index in range(len(restored_frames))]
    write_video(restored_frame_paths, restored_video_path, output_fps)
    write_video(enhanced_frame_paths, final_video_path, output_fps)

    report_frames = save_report_frames(
        panel_paths=panel_paths,
        output_dir=figures_dir,
        requested_count=int(config["visualization"]["save_report_frames"]),
        requested_indices=list(config["visualization"].get("report_frame_indices", [])),
    )

    summary = {
        "dataset_name": dataset_name,
        "run_dir": str(run_dir),
        "config": config,
        "input": {
            "video_path": str(input_video_path) if input_video_path else None,
            "frames_dir": str(input_frames_dir) if input_frames_dir else None,
        },
        "artifacts": {
            "frames_dir": str(frames_dir),
            "raw_overlays_dir": str(overlays_dir),
            "raw_dynamic_masks_dir": str(raw_masks_dir),
            "refined_masks_dir": str(refined_masks_dir),
            "hard_masks_dir": str(hard_masks_dir),
            "mask_contours_dir": str(contour_dir),
            "temporal_fill_dir": str(temporal_fill_dir),
            "restored_frames_dir": str(restored_dir),
            "enhanced_frames_dir": str(enhanced_dir),
            "comparison_panels_dir": str(panels_dir),
            "report_frames_dir": str(figures_dir),
            "diffusion_keyframes_dir": str(keyframes_dir),
            "restored_video": str(restored_video_path),
            "final_video": str(final_video_path),
        },
        "stats": {
            "num_frames": len(frame_records),
            "fps": output_fps,
            "segmenter_backend": segmenter.backend_name,
            "detector_model": config["segmentation"]["model_name"],
            "motion_displacement_threshold": config["motion"]["displacement_threshold"],
            "temporal_filled_pixels": total_temporal_filled_pixels,
            "avg_support_ratio": round(float(np.mean(support_ratios)) if support_ratios else 0.0, 4),
            "diffusion_keyframes": keyframe_indices,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        "report_frames": report_frames,
    }
    save_json(summary, run_dir / "run_summary.json")
    return summary


def _create_part3_panel(
    original: np.ndarray,
    raw_overlay: np.ndarray,
    raw_mask: np.ndarray,
    refined_mask: np.ndarray,
    hard_mask: np.ndarray,
    restored: np.ndarray,
    enhanced: np.ndarray,
    font_scale: float,
    font_thickness: int,
) -> np.ndarray:
    tiles = [
        annotate(original, "Original", font_scale, font_thickness),
        annotate(raw_overlay, "Prompted Mask", font_scale, font_thickness),
        annotate(mask_to_bgr(raw_mask), "Raw Dynamic Mask", font_scale, font_thickness),
        annotate(mask_to_bgr(refined_mask), "Refined Mask", font_scale, font_thickness),
        annotate(mask_to_bgr(hard_mask), "Hard Region", font_scale, font_thickness),
        annotate(restored, "Propainter-Like", font_scale, font_thickness),
        annotate(enhanced, "Diffusion-Like", font_scale, font_thickness),
    ]
    return np.concatenate(tiles, axis=1)


def _create_keyframe_preview(
    original: np.ndarray,
    restored: np.ndarray,
    enhanced: np.ndarray,
    hard_mask: np.ndarray,
    font_scale: float,
    font_thickness: int,
) -> np.ndarray:
    tiles = [
        annotate(original, "Original", font_scale, font_thickness),
        annotate(mask_to_bgr(hard_mask), "Hard Mask", font_scale, font_thickness),
        annotate(restored, "Before Enhance", font_scale, font_thickness),
        annotate(enhanced, "After Enhance", font_scale, font_thickness),
    ]
    return np.concatenate(tiles, axis=1)
