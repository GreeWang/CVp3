from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_project.evaluation.metrics import jaccard_mean, jaccard_recall, mask_iou


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate predicted masks against ground truth masks.")
    parser.add_argument("--pred_dir", required=True, help="Directory with predicted masks.")
    parser.add_argument("--gt_dir", required=True, help="Directory with ground truth masks.")
    parser.add_argument("--output_json", default=None, help="Optional path for saving evaluation results.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    common_files = sorted({path.name for path in pred_dir.iterdir() if path.is_file()} & {path.name for path in gt_dir.iterdir() if path.is_file()})
    if not common_files:
        raise RuntimeError("No matching mask files found between prediction and ground truth directories.")

    pred_masks = []
    gt_masks = []
    frame_scores: dict[str, float] = {}
    for file_name in common_files:
        pred_mask = cv2.imread(str(pred_dir / file_name), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(str(gt_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None or gt_mask is None:
            continue
        pred_masks.append(pred_mask)
        gt_masks.append(gt_mask)
        frame_scores[file_name] = round(mask_iou(pred_mask, gt_mask), 6)

    results = {
        "num_frames": len(pred_masks),
        "jaccard_mean": round(jaccard_mean(pred_masks, gt_masks), 6),
        "jaccard_recall": round(jaccard_recall(pred_masks, gt_masks), 6),
        "per_frame_iou": frame_scores,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
