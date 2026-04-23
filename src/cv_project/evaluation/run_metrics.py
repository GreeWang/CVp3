from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def evaluate_enhancement_run(
    frames_dir: Path,
    restored_dir: Path,
    enhanced_dir: Path,
    masks_dir: Path,
    *,
    keyframe_indices: set[int] | None = None,
    change_threshold: float = 6.0,
    seam_kernel_size: int = 9,
) -> dict:
    """Evaluate one enhancement run without ground-truth clean video.

    The function compares three aligned frame streams:
    - original frames (with target object)
    - restored frames (after ProPainter/object removal)
    - enhanced frames (after diffusion refinement)

    It reports:
    - per-frame spatial metrics in hard mask / seam / outside regions
    - temporal pair metrics on consecutive frames
    - aggregated summaries for all frames and keyframes
    - before/after comparisons between restored and enhanced outputs
    """
    common_files = _list_common_files(frames_dir, restored_dir, enhanced_dir, masks_dir)
    if not common_files:
        raise RuntimeError("No matching files found across frames/restored_frames/enhanced_frames/hard_masks.")

    per_frame: dict[str, dict[str, float | int | bool]] = {}
    frame_metrics: list[dict[str, float | int | bool]] = []

    prev_restored: np.ndarray | None = None
    prev_enhanced: np.ndarray | None = None
    prev_mask: np.ndarray | None = None
    temporal_pairs: list[dict[str, float | int]] = []
    keyframe_indices = keyframe_indices or set()

    for file_name in common_files:
        original = cv2.imread(str(frames_dir / file_name))
        restored = cv2.imread(str(restored_dir / file_name))
        enhanced = cv2.imread(str(enhanced_dir / file_name))
        mask = cv2.imread(str(masks_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if original is None or restored is None or enhanced is None or mask is None:
            continue

        if restored.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_AREA)
        if original.shape != restored.shape:
            original = cv2.resize(original, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_AREA)
        if mask.shape[:2] != restored.shape[:2]:
            mask = cv2.resize(mask, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_NEAREST)

        frame_index = _parse_frame_index(file_name)
        frame_result = _compute_frame_metrics(
            original=original,
            restored=restored,
            enhanced=enhanced,
            hard_mask=mask,
            change_threshold=change_threshold,
            seam_kernel_size=seam_kernel_size,
        )
        frame_result["frame_index"] = frame_index
        frame_result["is_keyframe"] = frame_index in keyframe_indices
        per_frame[file_name] = frame_result
        frame_metrics.append(frame_result)

        if prev_restored is not None and prev_enhanced is not None and prev_mask is not None:
            union_mask = np.maximum(prev_mask, mask)
            temporal_pairs.append(
                _compute_temporal_pair_metrics(
                    prev_restored=prev_restored,
                    current_restored=restored,
                    prev_enhanced=prev_enhanced,
                    current_enhanced=enhanced,
                    union_mask=union_mask,
                    frame_index=frame_index,
                )
            )

        prev_restored = restored
        prev_enhanced = enhanced
        prev_mask = mask

    all_summary = _aggregate_frame_metrics(frame_metrics)
    keyframe_metrics = [item for item in frame_metrics if bool(item["is_keyframe"])]
    keyframe_summary = _aggregate_frame_metrics(keyframe_metrics)
    temporal_summary = _aggregate_temporal_metrics(temporal_pairs)

    return {
        "num_frames": len(frame_metrics),
        "num_keyframes": len(keyframe_metrics),
        "settings": {
            "change_threshold": change_threshold,
            "seam_kernel_size": seam_kernel_size,
        },
        "all_frames": all_summary,
        "keyframes": keyframe_summary,
        "temporal": temporal_summary,
        "before_after_comparison": {
            "all_frames": _build_frame_comparison(all_summary),
            "keyframes": _build_frame_comparison(keyframe_summary),
            "temporal": _build_temporal_comparison(temporal_summary),
        },
        "per_frame": per_frame,
        "per_temporal_pair": temporal_pairs,
    }


def discover_keyframe_indices(run_dir: Path) -> set[int]:
    keyframes_dir = run_dir / "previews" / "diffusion_keyframes"
    if not keyframes_dir.exists():
        keyframes_dir = run_dir / "diffusion_keyframes"
    if not keyframes_dir.exists():
        return set()
    indices: set[int] = set()
    for path in sorted(keyframes_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            indices.add(int(path.stem))
        except ValueError:
            continue
    return indices


def _list_common_files(*directories: Path) -> list[str]:
    file_sets = []
    for directory in directories:
        if not directory.exists():
            raise RuntimeError(f"Directory not found: {directory}")
        file_sets.append({path.name for path in directory.iterdir() if path.is_file()})
    common = set.intersection(*file_sets) if file_sets else set()
    return sorted(common)


def _compute_frame_metrics(
    *,
    original: np.ndarray,
    restored: np.ndarray,
    enhanced: np.ndarray,
    hard_mask: np.ndarray,
    change_threshold: float,
    seam_kernel_size: int,
) -> dict[str, float]:
    """Compute region-aware metrics for a single frame.

    Metric groups:
    - change metrics: how much enhanced differs from restored
    - fidelity metrics: how restored/enhanced differ from original
    - artifact metrics: Laplacian energy as high-frequency artifact proxy
    - leakage metrics: how much change falls outside hard mask
    """
    hard_region = hard_mask > 0
    total_pixels = int(hard_region.size)
    hard_pixels = int(np.count_nonzero(hard_region))
    outside_region = ~hard_region
    seam_region = _build_seam_mask(hard_region, seam_kernel_size)

    delta_map = np.mean(np.abs(enhanced.astype(np.float32) - restored.astype(np.float32)), axis=2)
    restored_original_delta_map = np.mean(np.abs(restored.astype(np.float32) - original.astype(np.float32)), axis=2)
    enhanced_original_delta_map = np.mean(np.abs(enhanced.astype(np.float32) - original.astype(np.float32)), axis=2)
    delta_gray = cv2.cvtColor(
        np.clip(np.abs(enhanced.astype(np.int16) - restored.astype(np.int16)), 0, 255).astype(np.uint8),
        cv2.COLOR_BGR2GRAY,
    )
    restored_original_delta_gray = cv2.cvtColor(
        np.clip(np.abs(restored.astype(np.int16) - original.astype(np.int16)), 0, 255).astype(np.uint8),
        cv2.COLOR_BGR2GRAY,
    )
    enhanced_original_delta_gray = cv2.cvtColor(
        np.clip(np.abs(enhanced.astype(np.int16) - original.astype(np.int16)), 0, 255).astype(np.uint8),
        cv2.COLOR_BGR2GRAY,
    )
    delta_laplacian = np.abs(cv2.Laplacian(delta_gray.astype(np.float32), cv2.CV_32F))
    restored_original_laplacian = np.abs(cv2.Laplacian(restored_original_delta_gray.astype(np.float32), cv2.CV_32F))
    enhanced_original_laplacian = np.abs(cv2.Laplacian(enhanced_original_delta_gray.astype(np.float32), cv2.CV_32F))

    inside_delta_sum = float(np.sum(delta_map[hard_region])) if hard_pixels else 0.0
    outside_delta_sum = float(np.sum(delta_map[outside_region])) if np.any(outside_region) else 0.0
    total_delta_sum = inside_delta_sum + outside_delta_sum

    changed_pixels = delta_map > change_threshold
    inside_changed = int(np.count_nonzero(changed_pixels & hard_region))
    outside_changed = int(np.count_nonzero(changed_pixels & outside_region))
    total_changed = inside_changed + outside_changed

    return {
        "mask_ratio": hard_pixels / float(total_pixels) if total_pixels else 0.0,
        "inside_change_l1": _masked_mean(delta_map, hard_region),
        "outside_change_l1": _masked_mean(delta_map, outside_region),
        "seam_change_l1": _masked_mean(delta_map, seam_region),
        "restored_inside_original_l1": _masked_mean(restored_original_delta_map, hard_region),
        "restored_outside_original_l1": _masked_mean(restored_original_delta_map, outside_region),
        "enhanced_inside_original_l1": _masked_mean(enhanced_original_delta_map, hard_region),
        "enhanced_outside_original_l1": _masked_mean(enhanced_original_delta_map, outside_region),
        "inside_original_l1": _masked_mean(enhanced_original_delta_map, hard_region),
        "outside_original_l1": _masked_mean(enhanced_original_delta_map, outside_region),
        "change_concentration": inside_delta_sum / total_delta_sum if total_delta_sum > 1e-6 else 0.0,
        "inside_changed_ratio": inside_changed / float(hard_pixels) if hard_pixels else 0.0,
        "outside_changed_ratio": outside_changed / float(max(total_pixels - hard_pixels, 1)),
        "change_leakage_ratio": outside_changed / float(total_changed) if total_changed > 0 else 0.0,
        "restored_artifact_laplacian_hard": _masked_mean(restored_original_laplacian, hard_region),
        "restored_artifact_laplacian_seam": _masked_mean(restored_original_laplacian, seam_region),
        "enhanced_artifact_laplacian_hard": _masked_mean(enhanced_original_laplacian, hard_region),
        "enhanced_artifact_laplacian_seam": _masked_mean(enhanced_original_laplacian, seam_region),
        "artifact_laplacian_hard": _masked_mean(delta_laplacian, hard_region),
        "artifact_laplacian_seam": _masked_mean(delta_laplacian, seam_region),
    }


def _compute_temporal_pair_metrics(
    *,
    prev_restored: np.ndarray,
    current_restored: np.ndarray,
    prev_enhanced: np.ndarray,
    current_enhanced: np.ndarray,
    union_mask: np.ndarray,
    frame_index: int,
) -> dict[str, float | int]:
    """Compute temporal stability metrics on one consecutive frame pair.

    We compare frame-to-frame L1 variation inside union hard-mask region:
    - restored temporal L1 (reference)
    - enhanced temporal L1 (candidate)
    - instability ratio = enhanced / restored
    """
    active_region = union_mask > 0
    restored_diff = np.mean(np.abs(current_restored.astype(np.float32) - prev_restored.astype(np.float32)), axis=2)
    enhanced_diff = np.mean(np.abs(current_enhanced.astype(np.float32) - prev_enhanced.astype(np.float32)), axis=2)

    restored_l1 = _masked_mean(restored_diff, active_region)
    enhanced_l1 = _masked_mean(enhanced_diff, active_region)
    if restored_l1 <= 1e-6:
        instability_ratio = 1.0 if enhanced_l1 <= 1e-6 else float("inf")
    else:
        instability_ratio = enhanced_l1 / restored_l1

    return {
        "frame_index": frame_index,
        "temporal_restored_l1": restored_l1,
        "temporal_enhanced_l1": enhanced_l1,
        "temporal_instability_ratio": instability_ratio,
    }


def _aggregate_frame_metrics(metrics: list[dict[str, float | int | bool]]) -> dict:
    if not metrics:
        return {
            "count": 0,
            "mask_ratio_mean": 0.0,
            "inside_change_l1_mean": 0.0,
            "outside_change_l1_mean": 0.0,
            "seam_change_l1_mean": 0.0,
            "restored_inside_original_l1_mean": 0.0,
            "restored_outside_original_l1_mean": 0.0,
            "enhanced_inside_original_l1_mean": 0.0,
            "enhanced_outside_original_l1_mean": 0.0,
            "inside_original_l1_mean": 0.0,
            "outside_original_l1_mean": 0.0,
            "change_concentration_mean": 0.0,
            "inside_changed_ratio_mean": 0.0,
            "outside_changed_ratio_mean": 0.0,
            "change_leakage_ratio_mean": 0.0,
            "restored_artifact_laplacian_hard_mean": 0.0,
            "restored_artifact_laplacian_seam_mean": 0.0,
            "enhanced_artifact_laplacian_hard_mean": 0.0,
            "enhanced_artifact_laplacian_seam_mean": 0.0,
            "artifact_laplacian_hard_mean": 0.0,
            "artifact_laplacian_seam_mean": 0.0,
        }

    return {
        "count": len(metrics),
        "mask_ratio_mean": _mean_of(metrics, "mask_ratio"),
        "inside_change_l1_mean": _mean_of(metrics, "inside_change_l1"),
        "outside_change_l1_mean": _mean_of(metrics, "outside_change_l1"),
        "seam_change_l1_mean": _mean_of(metrics, "seam_change_l1"),
        "restored_inside_original_l1_mean": _mean_of(metrics, "restored_inside_original_l1"),
        "restored_outside_original_l1_mean": _mean_of(metrics, "restored_outside_original_l1"),
        "enhanced_inside_original_l1_mean": _mean_of(metrics, "enhanced_inside_original_l1"),
        "enhanced_outside_original_l1_mean": _mean_of(metrics, "enhanced_outside_original_l1"),
        "inside_original_l1_mean": _mean_of(metrics, "inside_original_l1"),
        "outside_original_l1_mean": _mean_of(metrics, "outside_original_l1"),
        "change_concentration_mean": _mean_of(metrics, "change_concentration"),
        "inside_changed_ratio_mean": _mean_of(metrics, "inside_changed_ratio"),
        "outside_changed_ratio_mean": _mean_of(metrics, "outside_changed_ratio"),
        "change_leakage_ratio_mean": _mean_of(metrics, "change_leakage_ratio"),
        "restored_artifact_laplacian_hard_mean": _mean_of(metrics, "restored_artifact_laplacian_hard"),
        "restored_artifact_laplacian_seam_mean": _mean_of(metrics, "restored_artifact_laplacian_seam"),
        "enhanced_artifact_laplacian_hard_mean": _mean_of(metrics, "enhanced_artifact_laplacian_hard"),
        "enhanced_artifact_laplacian_seam_mean": _mean_of(metrics, "enhanced_artifact_laplacian_seam"),
        "artifact_laplacian_hard_mean": _mean_of(metrics, "artifact_laplacian_hard"),
        "artifact_laplacian_seam_mean": _mean_of(metrics, "artifact_laplacian_seam"),
    }


def _aggregate_temporal_metrics(temporal_pairs: list[dict[str, float | int]]) -> dict:
    if not temporal_pairs:
        return {
            "count": 0,
            "temporal_restored_l1_mean": 0.0,
            "temporal_enhanced_l1_mean": 0.0,
            "temporal_instability_ratio_mean": 0.0,
        }

    finite_ratios = [float(item["temporal_instability_ratio"]) for item in temporal_pairs if np.isfinite(item["temporal_instability_ratio"])]
    return {
        "count": len(temporal_pairs),
        "temporal_restored_l1_mean": _mean_of(temporal_pairs, "temporal_restored_l1"),
        "temporal_enhanced_l1_mean": _mean_of(temporal_pairs, "temporal_enhanced_l1"),
        "temporal_instability_ratio_mean": float(np.mean(finite_ratios)) if finite_ratios else float("inf"),
    }


def _build_seam_mask(hard_region: np.ndarray, kernel_size: int) -> np.ndarray:
    """Build a seam band around hard region using dilate-erode ring."""
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = hard_region.astype(np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1) > 0
    eroded = cv2.erode(binary, kernel, iterations=1) > 0
    return dilated & (~eroded)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    count = int(np.count_nonzero(valid))
    if count == 0:
        return 0.0
    return float(np.mean(values[valid]))


def _mean_of(items: list[dict[str, float | int | bool]], key: str) -> float:
    values = [float(item[key]) for item in items]
    return float(np.mean(values)) if values else 0.0


def _build_frame_comparison(summary: dict) -> dict:
    if int(summary.get("count", 0)) == 0:
        return {"count": 0}
    return {
        "count": int(summary["count"]),
        "hard_region_fidelity_l1": _comparison_entry(
            before=float(summary["restored_inside_original_l1_mean"]),
            after=float(summary["enhanced_inside_original_l1_mean"]),
            lower_is_better=True,
        ),
        "background_preservation_l1": _comparison_entry(
            before=float(summary["restored_outside_original_l1_mean"]),
            after=float(summary["enhanced_outside_original_l1_mean"]),
            lower_is_better=True,
        ),
        "hard_region_artifact_laplacian": _comparison_entry(
            before=float(summary["restored_artifact_laplacian_hard_mean"]),
            after=float(summary["enhanced_artifact_laplacian_hard_mean"]),
            lower_is_better=True,
        ),
        "seam_artifact_laplacian": _comparison_entry(
            before=float(summary["restored_artifact_laplacian_seam_mean"]),
            after=float(summary["enhanced_artifact_laplacian_seam_mean"]),
            lower_is_better=True,
        ),
    }


def _build_temporal_comparison(summary: dict) -> dict:
    if int(summary.get("count", 0)) == 0:
        return {"count": 0}
    return {
        "count": int(summary["count"]),
        "masked_temporal_l1": _comparison_entry(
            before=float(summary["temporal_restored_l1_mean"]),
            after=float(summary["temporal_enhanced_l1_mean"]),
            lower_is_better=True,
        ),
        "temporal_instability_ratio_mean": float(summary["temporal_instability_ratio_mean"]),
    }


def _comparison_entry(*, before: float, after: float, lower_is_better: bool) -> dict[str, float | str]:
    """Create a normalized before/after comparison item with verdict."""
    delta = after - before
    if abs(before) <= 1e-6:
        relative_change = 0.0 if abs(delta) <= 1e-6 else float("inf")
    else:
        relative_change = delta / before

    if abs(delta) <= 1e-6:
        verdict = "unchanged"
    elif lower_is_better:
        verdict = "improved" if after < before else "worse"
    else:
        verdict = "improved" if after > before else "worse"

    return {
        "before": before,
        "after": after,
        "delta": delta,
        "relative_change": relative_change,
        "verdict": verdict,
    }


def _parse_frame_index(file_name: str) -> int:
    """Parse frame index from file name.

    Priority:
    1) full stem as integer, e.g. "000123"
    2) fallback to all digits in stem, e.g. "frame_123"
    3) return -1 when no numeric token exists
    """
    stem = Path(file_name).stem
    try:
        return int(stem)
    except ValueError:
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            return int(digits)
        return -1
