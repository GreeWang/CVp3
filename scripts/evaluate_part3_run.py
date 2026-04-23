from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_project.evaluation.run_metrics import discover_keyframe_indices, evaluate_enhancement_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independently evaluate a Part 3 result run without ground truth.")
    parser.add_argument("--run_dir", required=True, help="Run directory under results/part3/<dataset>/<timestamp>.")
    parser.add_argument("--output_json", default=None, help="Optional path for saving evaluation results.")
    parser.add_argument(
        "--change_threshold",
        type=float,
        default=6.0,
        help="Per-pixel mean absolute difference threshold used to count changed pixels.",
    )
    parser.add_argument(
        "--seam_kernel_size",
        type=int,
        default=9,
        help="Kernel size used to derive the seam band around the hard mask.",
    )
    parser.add_argument(
        "--top_k_failures",
        type=int,
        default=20,
        help="Number of worst frames to export for each diagnostic metric.",
    )
    parser.add_argument(
        "--baseline_eval_json",
        default="results/part3/bmx-trees-propainter-only-ab/20260416_174257/independent_eval.json",
        help="Optional independent_eval.json path used as baseline for score comparison.",
    )
    parser.add_argument("--score_alpha", type=float, default=2.0, help="Weight for hard-region gain term.")
    parser.add_argument("--score_beta", type=float, default=1.0, help="Weight for outside change penalty.")
    parser.add_argument("--score_gamma", type=float, default=1.0, help="Weight for seam change penalty.")
    parser.add_argument("--score_delta", type=float, default=1.0, help="Weight for temporal penalty.")
    parser.add_argument(
        "--tol_outside_change",
        type=float,
        default=0.003,
        help="Risk gate: max allowed outside_change_l1_mean.",
    )
    parser.add_argument(
        "--tol_seam_change",
        type=float,
        default=0.10,
        help="Risk gate: max allowed seam_change_l1_mean.",
    )
    parser.add_argument(
        "--tol_leakage",
        type=float,
        default=0.0,
        help="Risk gate: max allowed change_leakage_ratio_mean.",
    )
    parser.add_argument(
        "--tol_temporal_ratio",
        type=float,
        default=1.02,
        help="Risk gate: max allowed temporal_instability_ratio_mean.",
    )
    parser.add_argument(
        "--min_hard_gain",
        type=float,
        default=0.01,
        help="Minimum required hard artifact gain (before-after) to count as effective enhancement.",
    )
    parser.add_argument(
        "--min_score_margin_vs_baseline",
        type=float,
        default=0.0,
        help="Required score margin over baseline composite score.",
    )
    return parser


def main() -> int:
    """Run independent evaluation and write one JSON report for a run directory."""
    parser = build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    artifacts = _load_artifacts(run_dir)
    frames_dir = _artifact_path(run_dir, artifacts, "frames_dir", "frames")
    restored_dir = _artifact_path(run_dir, artifacts, "restored_frames_dir", "restored_frames")
    enhanced_dir = _artifact_path(run_dir, artifacts, "enhanced_frames_dir", "enhanced_frames")
    masks_dir = _artifact_path(run_dir, artifacts, "hard_masks_dir", "hard_masks")
    keyframe_indices = discover_keyframe_indices(run_dir)

    metrics = evaluate_enhancement_run(
        frames_dir=frames_dir,
        restored_dir=restored_dir,
        enhanced_dir=enhanced_dir,
        masks_dir=masks_dir,
        keyframe_indices=keyframe_indices,
        change_threshold=args.change_threshold,
        seam_kernel_size=args.seam_kernel_size,
    )

    results = {
        "run_dir": str(run_dir.resolve()),
        "keyframe_indices": sorted(keyframe_indices),
        "metrics": metrics,
        "failure_frames": _rank_failure_frames(
            per_frame=metrics["per_frame"],
            per_temporal_pair=metrics.get("per_temporal_pair", []),
            top_k=max(1, int(args.top_k_failures)),
        ),
    }

    baseline_eval = _load_baseline_eval(args.baseline_eval_json)
    strategy = _build_gain_risk_strategy(args)
    strategy_eval = _evaluate_gain_risk_strategy(
        metrics=metrics,
        strategy=strategy,
        baseline_eval=baseline_eval,
    )
    results["strategy"] = {
        "gain_risk": strategy_eval,
    }

    output_path = Path(args.output_json) if args.output_json else run_dir / "independent_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


def _load_artifacts(run_dir: Path) -> dict:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8")).get("artifacts", {})
    except json.JSONDecodeError:
        return {}


def _artifact_path(run_dir: Path, artifacts: dict, key: str, legacy_name: str) -> Path:
    value = artifacts.get(key)
    if value:
        return Path(value)
    compact_defaults = {
        "frames_dir": run_dir / "inputs" / "frames",
        "restored_frames_dir": run_dir / "outputs" / "restored_frames",
        "enhanced_frames_dir": run_dir / "outputs" / "enhanced_frames",
        "hard_masks_dir": run_dir / "masks" / "hard",
    }
    compact_path = compact_defaults.get(key)
    if compact_path is not None and compact_path.exists():
        return compact_path
    return run_dir / legacy_name


def _rank_failure_frames(
    *,
    per_frame: dict[str, dict[str, float | int | bool]],
    per_temporal_pair: list[dict[str, float | int]] | None,
    top_k: int,
) -> dict[str, list[dict[str, float | int | bool]]]:
    """Return top-k worst cases for diagnostics.

    Spatial failures are ranked from per-frame metrics.
    Temporal failures are ranked from true pair-wise instability ratio.
    """
    entries = []
    for file_name, metrics in per_frame.items():
        entry = dict(metrics)
        entry["file_name"] = file_name
        entries.append(entry)

    def _top(metric: str, descending: bool = True) -> list[dict[str, float | int | bool]]:
        return sorted(entries, key=lambda item: float(item.get(metric, 0.0)), reverse=descending)[:top_k]

    temporal_entries = per_temporal_pair or []
    temporal_top = sorted(
        temporal_entries,
        key=lambda item: float(item.get("temporal_instability_ratio", 0.0)),
        reverse=True,
    )[:top_k]

    return {
        "temporal_instability": temporal_top,
        "change_leakage": _top("change_leakage_ratio", descending=True),
        "seam_error": _top("seam_change_l1", descending=True),
        "hard_region_change": _top("inside_change_l1", descending=True),
    }


def _load_baseline_eval(path_value: str | None) -> dict | None:
    if not path_value:
        return None
    baseline_path = Path(path_value)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline eval file not found: {baseline_path}")
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def _build_gain_risk_strategy(args: argparse.Namespace) -> dict[str, float]:
    """Collect strategy hyper-parameters from CLI into one dict."""
    return {
        "score_alpha": float(args.score_alpha),
        "score_beta": float(args.score_beta),
        "score_gamma": float(args.score_gamma),
        "score_delta": float(args.score_delta),
        "tol_outside_change": float(args.tol_outside_change),
        "tol_seam_change": float(args.tol_seam_change),
        "tol_leakage": float(args.tol_leakage),
        "tol_temporal_ratio": float(args.tol_temporal_ratio),
        "min_hard_gain": float(args.min_hard_gain),
        "min_score_margin_vs_baseline": float(args.min_score_margin_vs_baseline),
    }


def _evaluate_gain_risk_strategy(*, metrics: dict, strategy: dict[str, float], baseline_eval: dict | None) -> dict:
    """Evaluate gain-risk score, hard gates, and optional baseline comparison."""
    current = _extract_gain_risk_features(metrics)
    current_score = _composite_score(current, strategy)
    gate_result = _apply_gates(current, strategy)

    result: dict[str, object] = {
        "strategy": strategy,
        "features": current,
        "composite_score": current_score,
        "gate_pass": gate_result["pass"],
        "gate_checks": gate_result["checks"],
    }

    if baseline_eval is None:
        result["decision"] = {
            "status": "no-baseline",
            "reason": "baseline_eval_json not provided",
        }
        return result

    baseline_metrics = baseline_eval.get("metrics", {})
    baseline_features = _extract_gain_risk_features(baseline_metrics)
    baseline_score = _composite_score(baseline_features, strategy)
    score_margin = float(current_score - baseline_score)
    required_margin = float(strategy["min_score_margin_vs_baseline"])
    beats_baseline = bool(gate_result["pass"] and score_margin > required_margin)

    result["baseline"] = {
        "run_dir": baseline_eval.get("run_dir"),
        "features": baseline_features,
        "composite_score": baseline_score,
    }
    result["decision"] = {
        "status": "beats-baseline" if beats_baseline else "not-beating-baseline",
        "beats_baseline": beats_baseline,
        "score_margin_vs_baseline": score_margin,
        "required_margin": required_margin,
        "gate_pass": gate_result["pass"],
    }
    return result


def _extract_gain_risk_features(metrics: dict) -> dict[str, float]:
    """Extract compact features used by composite score and gate checks."""
    all_frames = metrics.get("all_frames", {})
    temporal = metrics.get("temporal", {})
    cmp_all = metrics.get("before_after_comparison", {}).get("all_frames", {})

    outside_change = float(all_frames.get("outside_change_l1_mean", 0.0))
    seam_change = float(all_frames.get("seam_change_l1_mean", 0.0))
    leakage = float(all_frames.get("change_leakage_ratio_mean", 0.0))
    temporal_ratio = float(temporal.get("temporal_instability_ratio_mean", 0.0))
    hard_artifact_delta = float(cmp_all.get("hard_region_artifact_laplacian", {}).get("delta", 0.0))
    hard_gain = max(0.0, -hard_artifact_delta)
    temporal_penalty = max(0.0, temporal_ratio - 1.0)

    return {
        "outside_change_l1_mean": outside_change,
        "seam_change_l1_mean": seam_change,
        "change_leakage_ratio_mean": leakage,
        "temporal_instability_ratio_mean": temporal_ratio,
        "hard_artifact_gain": hard_gain,
        "temporal_penalty": temporal_penalty,
    }


def _composite_score(features: dict[str, float], strategy: dict[str, float]) -> float:
    """Linear gain-risk objective.

    score = alpha * hard_gain - beta * outside_penalty - gamma * seam_penalty - delta * temporal_penalty
    """
    return (
        float(strategy["score_alpha"]) * float(features["hard_artifact_gain"])
        - float(strategy["score_beta"]) * float(features["outside_change_l1_mean"])
        - float(strategy["score_gamma"]) * float(features["seam_change_l1_mean"])
        - float(strategy["score_delta"]) * float(features["temporal_penalty"])
    )


def _apply_gates(features: dict[str, float], strategy: dict[str, float]) -> dict[str, object]:
    """Apply hard constraints to reject risky enhancement runs.

    All checks must pass for gate_pass=True.
    """
    checks = {
        "outside_change": {
            "value": float(features["outside_change_l1_mean"]),
            "threshold": float(strategy["tol_outside_change"]),
            "pass": float(features["outside_change_l1_mean"]) <= float(strategy["tol_outside_change"]),
        },
        "seam_change": {
            "value": float(features["seam_change_l1_mean"]),
            "threshold": float(strategy["tol_seam_change"]),
            "pass": float(features["seam_change_l1_mean"]) <= float(strategy["tol_seam_change"]),
        },
        "leakage": {
            "value": float(features["change_leakage_ratio_mean"]),
            "threshold": float(strategy["tol_leakage"]),
            "pass": float(features["change_leakage_ratio_mean"]) <= float(strategy["tol_leakage"]),
        },
        "temporal_ratio": {
            "value": float(features["temporal_instability_ratio_mean"]),
            "threshold": float(strategy["tol_temporal_ratio"]),
            "pass": float(features["temporal_instability_ratio_mean"]) <= float(strategy["tol_temporal_ratio"]),
        },
        "hard_gain": {
            "value": float(features["hard_artifact_gain"]),
            "threshold": float(strategy["min_hard_gain"]),
            "pass": float(features["hard_artifact_gain"]) >= float(strategy["min_hard_gain"]),
        },
    }
    gate_pass = all(bool(item["pass"]) for item in checks.values())
    return {
        "pass": gate_pass,
        "checks": checks,
    }


if __name__ == "__main__":
    raise SystemExit(main())
