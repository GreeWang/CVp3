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
from cv_project.inpainting.freeinpaint_refiner import FreeInpaintRefiner
from cv_project.inpainting.diffusion_inpaint_refiner import DiffusionInpaintRefiner
from cv_project.inpainting.diffusion_target import build_diffusion_target_masks
from cv_project.inpainting.propainter_official import OfficialProPainterRunner
from cv_project.inpainting.propainter_repair import ProPainterLikeRestorer
from cv_project.motion.dynamic_filter import OpticalFlowDynamicFilter
from cv_project.pipeline.types import FrameRecord
from cv_project.segmentation.sam2_video_segmenter import Sam2VideoSegmenter
from cv_project.segmentation.yolo_segmenter import YoloSegmenter
from cv_project.utils.mask_ops import merge_instance_masks, refine_mask_sequence
from cv_project.utils.visualization import (
    annotate,
    mask_to_bgr,
    overlay_detections,
    save_report_frames,
)


def run_part3_pipeline(config: dict, project_root: Path) -> dict:
    started_at = time.time()
    dataset_name = config["output"]["dataset_name"]
    run_dir = ensure_dir(resolve_path(config["output"]["root_dir"], project_root) / dataset_name / timestamp_now())
    inputs_dir = ensure_dir(run_dir / "inputs")
    masks_dir = ensure_dir(run_dir / "masks")
    outputs_dir = ensure_dir(run_dir / "outputs")
    previews_dir = ensure_dir(run_dir / "previews")
    work_dir = ensure_dir(run_dir / "work")
    frames_dir = ensure_dir(inputs_dir / "frames")
    sam2_frames_dir = ensure_dir(work_dir / "sam2_frames")
    refined_masks_dir = ensure_dir(masks_dir / "refined")
    hard_masks_dir = ensure_dir(masks_dir / "hard")
    borrowable_masks_dir = ensure_dir(masks_dir / "borrowable")
    diffusion_target_masks_dir = ensure_dir(masks_dir / "diffusion_targets")
    restored_dir = ensure_dir(outputs_dir / "restored_frames")
    enhanced_dir = ensure_dir(outputs_dir / "enhanced_frames")
    panels_dir = ensure_dir(previews_dir / "comparison")
    figures_dir = ensure_dir(previews_dir / "report")
    keyframes_dir = ensure_dir(previews_dir / "diffusion_keyframes")
    propainter_output_dir = ensure_dir(work_dir / "propainter")

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
            cv2.imwrite(str(sam2_frames_dir / f"{index:0{frame_width}d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frame_records.append(FrameRecord(frame_index=index, image=image, source_path=str(normalized_path)))
            loaded_frames.append(image)
    else:
        for index, source_path in enumerate(source_frame_paths):
            image = cv2.imread(str(source_path))
            if image is None:
                raise RuntimeError(f"Unable to read frame image: {source_path}")
            normalized_jpg_path = sam2_frames_dir / f"{index:0{frame_width}d}.jpg"
            cv2.imwrite(str(normalized_jpg_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frame_records.append(FrameRecord(frame_index=index, image=image, source_path=str(source_path)))
            loaded_frames.append(image)

    try:
        segmenter = Sam2VideoSegmenter(config["segmentation"], project_root)
        detections_per_frame = segmenter.segment_video(frame_records, sam2_frames_dir)
        segmenter_backend = segmenter.backend_name
    except FileNotFoundError as exc:
        print(f"Warning: {exc} Falling back to YOLO-only per-frame segmentation.")
        yolo_config = config["segmentation"]
        segmenter = YoloSegmenter(
            model_name=str(yolo_config["model_name"]),
            device=str(yolo_config.get("device", "cpu")),
            confidence_threshold=float(yolo_config["confidence_threshold"]),
            iou_threshold=float(yolo_config["iou_threshold"]),
            dynamic_classes=list(yolo_config["dynamic_classes"]),
        )
        detections_per_frame = [segmenter.predict(record.image, record.frame_index) for record in frame_records]
        segmenter_backend = "yolo-only"
    raw_overlays: list[np.ndarray] = []
    for record, detections in tqdm(list(zip(frame_records, detections_per_frame)), desc="SAM2 propagation"):
        record.raw_detections = detections
        raw_overlay = overlay_detections(record.image, record.raw_detections, alpha=float(config["visualization"]["mask_alpha"]))
        raw_overlays.append(raw_overlay)

    dynamic_filter = OpticalFlowDynamicFilter(config["motion"])
    filtered_detections = dynamic_filter.apply(loaded_frames, [record.raw_detections for record in frame_records])

    raw_dynamic_masks: list[np.ndarray] = []
    for record, filtered in zip(frame_records, filtered_detections):
        record.filtered_detections = filtered
        raw_mask = merge_instance_masks([det.mask for det in filtered], record.image.shape[:2])
        record.raw_dynamic_mask = raw_mask
        raw_dynamic_masks.append(raw_mask)

    refined_masks = refine_mask_sequence(raw_dynamic_masks, loaded_frames, config["mask_postprocess"])
    for record, final_mask in zip(frame_records, refined_masks):
        record.final_mask = final_mask
        cv2.imwrite(str(refined_masks_dir / f"{record.frame_index:06d}.png"), final_mask)

    official_save_root: Path | None = None
    official_propainter = OfficialProPainterRunner(config["inpainting"], project_root)
    try:
        restored_frames, official_save_root = official_propainter.run(
            frames_dir=frames_dir,
            masks_dir=refined_masks_dir,
            output_root=propainter_output_dir,
        )
        if len(restored_frames) != len(frame_records):
            raise RuntimeError(
                f"Official ProPainter returned {len(restored_frames)} frames, expected {len(frame_records)}."
            )
        restoration_artifacts = []
        propainter_like = ProPainterLikeRestorer(config["inpainting"])
        proxy_artifacts = propainter_like.restore_sequence(loaded_frames, refined_masks)
        for restored_frame, proxy in zip(restored_frames, proxy_artifacts):
            proxy.restored_image = restored_frame
            restoration_artifacts.append(proxy)
    except Exception as exc:
        print(f"Warning: Official ProPainter execution failed ({exc}). Falling back to local temporal inpainting.")
        propainter_like = ProPainterLikeRestorer(config["inpainting"])
        restoration_artifacts = propainter_like.restore_sequence(loaded_frames, refined_masks)

    restored_frames: list[np.ndarray] = []
    hard_masks: list[np.ndarray] = []
    hard_residual_masks: list[np.ndarray] = []
    hard_weak_masks: list[np.ndarray] = []
    total_temporal_filled_pixels = 0
    support_ratios: list[float] = []
    hard_thresholds: list[float] = []
    for record, artifacts in zip(frame_records, restoration_artifacts):
        record.temporal_fill = artifacts.temporal_fill
        record.restored_image = artifacts.restored_image
        record.metadata["support_ratio"] = artifacts.support_ratio
        record.metadata["support_map_mean"] = float(np.mean(artifacts.support_map)) / 255.0
        record.metadata["hard_threshold_used"] = float(artifacts.hard_threshold_used)
        support_ratios.append(artifacts.support_ratio)
        hard_thresholds.append(float(artifacts.hard_threshold_used))
        total_temporal_filled_pixels += artifacts.candidate_pixels
        restored_frames.append(artifacts.restored_image)
        hard_masks.append(artifacts.hard_mask)
        hard_residual_masks.append(artifacts.hard_residual_mask)
        hard_weak_masks.append(artifacts.hard_weak_mask)
        cv2.imwrite(str(restored_dir / f"{record.frame_index:06d}.png"), artifacts.restored_image)
        cv2.imwrite(str(hard_masks_dir / f"{record.frame_index:06d}.png"), artifacts.hard_mask)

    borrowable_masks, diffusion_target_masks = build_diffusion_target_masks(
        frames=loaded_frames,
        refined_masks=refined_masks,
        restored_frames=restored_frames,
        candidate_masks=hard_masks,
        config=config["inpainting"],
    )
    for record, borrowable_mask, diffusion_target_mask in zip(frame_records, borrowable_masks, diffusion_target_masks):
        cv2.imwrite(str(borrowable_masks_dir / f"{record.frame_index:06d}.png"), borrowable_mask)
        cv2.imwrite(str(diffusion_target_masks_dir / f"{record.frame_index:06d}.png"), diffusion_target_mask)

    diffusion_config = dict(config.get("diffusion", {}))
    diffusion_model_id = diffusion_config.get("model_id")
    if isinstance(diffusion_model_id, str):
        diffusion_model_path = Path(diffusion_model_id)
        if not diffusion_model_path.is_absolute() and diffusion_model_path.parts[:1] in {("checkpoints",), ("models",)}:
            diffusion_config["model_id"] = str(project_root / diffusion_model_path)
    diffusion_method = str(diffusion_config.get("method", "diffusion_like")).lower()
    if diffusion_method in {"diffusion_like", "opencv", "legacy"}:
        enhancer = DiffusionLikeEnhancer(diffusion_config)
        enhanced_label = "Diffusion-Like"
    elif diffusion_method in {"sdxl_inpaint", "stable_diffusion_inpaint", "diffusion_inpaint"}:
        resolved_model_id = diffusion_config.get("model_id")
        if isinstance(resolved_model_id, str) and Path(resolved_model_id).is_absolute() and not Path(resolved_model_id).exists():
            print(f"Warning: diffusion model not found at {resolved_model_id}. Falling back to diffusion-like enhancer.")
            enhancer = DiffusionLikeEnhancer(diffusion_config)
            enhanced_label = "Diffusion-Like"
            diffusion_method = "diffusion_like_fallback"
        else:
            enhancer = DiffusionInpaintRefiner(diffusion_config)
            enhanced_label = "Diffusion Inpaint"
    elif diffusion_method in {"freeinpaint", "freepaint"}:
        enhancer = FreeInpaintRefiner(diffusion_config)
        enhanced_label = "FreeInpaint"
    else:
        raise ValueError(f"Unsupported diffusion.method: {diffusion_method}")
    enhanced_frames, keyframe_indices = enhancer.enhance_sequence(loaded_frames, restored_frames, diffusion_target_masks)

    panel_paths: list[Path] = []
    enhanced_frame_paths: list[Path] = []
    for record, enhanced_frame in zip(frame_records, enhanced_frames):
        enhanced_path = enhanced_dir / f"{record.frame_index:06d}.png"
        cv2.imwrite(str(enhanced_path), enhanced_frame)
        enhanced_frame_paths.append(enhanced_path)

        panel = _create_part3_panel(
            original=record.image,
            raw_overlay=raw_overlays[record.frame_index] if record.frame_index < len(raw_overlays) else record.image,
            raw_mask=record.raw_dynamic_mask if record.raw_dynamic_mask is not None else np.zeros(record.image.shape[:2], dtype=np.uint8),
            refined_mask=record.final_mask if record.final_mask is not None else np.zeros(record.image.shape[:2], dtype=np.uint8),
            hard_mask=diffusion_target_masks[record.frame_index],
            restored=record.restored_image if record.restored_image is not None else record.image,
            enhanced=enhanced_frame,
            enhanced_label=enhanced_label,
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
            hard_mask=diffusion_target_masks[keyframe_index],
            font_scale=float(config["visualization"]["panel_font_scale"]),
            font_thickness=int(config["visualization"]["panel_font_thickness"]),
        )
        cv2.imwrite(str(keyframes_dir / f"{keyframe_index:06d}.png"), preview)

    output_fps = float(config["output"]["save_fps"] or fps or 24.0)
    restored_video_path = outputs_dir / "restored_propainter.mp4"
    final_video_path = outputs_dir / "restored_part3_enhanced.mp4"
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
            "layout": "compact",
            "inputs_dir": str(inputs_dir),
            "masks_dir": str(masks_dir),
            "outputs_dir": str(outputs_dir),
            "previews_dir": str(previews_dir),
            "work_dir": str(work_dir),
            "frames_dir": str(frames_dir),
            "sam2_frames_dir": str(sam2_frames_dir),
            "refined_masks_dir": str(refined_masks_dir),
            "hard_masks_dir": str(hard_masks_dir),
            "borrowable_masks_dir": str(borrowable_masks_dir),
            "diffusion_target_masks_dir": str(diffusion_target_masks_dir),
            "propainter_official_dir": str(official_save_root) if official_save_root is not None else None,
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
            "segmenter_backend": segmenter_backend,
            "detector_model": config["segmentation"]["model_name"],
            "motion_displacement_threshold": config["motion"]["displacement_threshold"],
            "temporal_filled_pixels": total_temporal_filled_pixels,
            "avg_support_ratio": round(float(np.mean(support_ratios)) if support_ratios else 0.0, 4),
            "avg_hard_region_ratio": round(
                float(np.mean([np.count_nonzero(mask) / mask.size for mask in hard_masks])) if hard_masks else 0.0,
                4,
            ),
            "avg_hard_residual_ratio": round(
                float(np.mean([np.count_nonzero(mask) / mask.size for mask in hard_residual_masks]))
                if hard_residual_masks
                else 0.0,
                4,
            ),
            "avg_hard_weak_ratio": round(
                float(np.mean([np.count_nonzero(mask) / mask.size for mask in hard_weak_masks])) if hard_weak_masks else 0.0,
                4,
            ),
            "avg_hard_threshold_used": round(float(np.mean(hard_thresholds)) if hard_thresholds else 0.0, 4),
            "avg_borrowable_ratio": round(
                float(np.mean([np.count_nonzero(mask) / mask.size for mask in borrowable_masks])) if borrowable_masks else 0.0,
                4,
            ),
            "avg_diffusion_target_ratio": round(
                float(np.mean([np.count_nonzero(mask) / mask.size for mask in diffusion_target_masks]))
                if diffusion_target_masks
                else 0.0,
                4,
            ),
            "avg_diffusion_target_of_hard_ratio": round(
                float(
                    np.mean(
                        [
                            np.count_nonzero(target) / max(1, np.count_nonzero(hard))
                            for target, hard in zip(diffusion_target_masks, hard_masks)
                        ]
                    )
                )
                if diffusion_target_masks
                else 0.0,
                4,
            ),
            "diffusion_method": diffusion_method,
            "diffusion_model": diffusion_config.get("model_id"),
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
    enhanced_label: str,
    font_scale: float,
    font_thickness: int,
) -> np.ndarray:
    tiles = [
        annotate(original, "Original", font_scale, font_thickness),
        annotate(raw_overlay, "Prompted Mask", font_scale, font_thickness),
        annotate(mask_to_bgr(raw_mask), "Raw Dynamic Mask", font_scale, font_thickness),
        annotate(mask_to_bgr(refined_mask), "Refined Mask", font_scale, font_thickness),
        annotate(mask_to_bgr(hard_mask), "Diffusion Target", font_scale, font_thickness),
        annotate(restored, "ProPainter", font_scale, font_thickness),
        annotate(enhanced, enhanced_label, font_scale, font_thickness),
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
        annotate(mask_to_bgr(hard_mask), "Diffusion Target", font_scale, font_thickness),
        annotate(restored, "Before Enhance", font_scale, font_thickness),
        annotate(enhanced, "After Enhance", font_scale, font_thickness),
    ]
    return np.concatenate(tiles, axis=1)
