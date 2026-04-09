from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv_project.evaluation.metrics import psnr, ssim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate restored frames against ground truth frames.")
    parser.add_argument("--pred_dir", required=True, help="Directory with restored frames.")
    parser.add_argument("--gt_dir", required=True, help="Directory with ground truth frames.")
    parser.add_argument("--output_json", default=None, help="Optional path for saving evaluation results.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    common_files = sorted({path.name for path in pred_dir.iterdir() if path.is_file()} & {path.name for path in gt_dir.iterdir() if path.is_file()})
    if not common_files:
        raise RuntimeError("No matching frame files found between prediction and ground truth directories.")

    psnr_scores: dict[str, float] = {}
    ssim_scores: dict[str, float] = {}
    for file_name in common_files:
        pred_image = cv2.imread(str(pred_dir / file_name))
        gt_image = cv2.imread(str(gt_dir / file_name))
        if pred_image is None or gt_image is None:
            continue
        if pred_image.shape != gt_image.shape:
            gt_image = cv2.resize(gt_image, (pred_image.shape[1], pred_image.shape[0]), interpolation=cv2.INTER_AREA)
        psnr_scores[file_name] = round(psnr(pred_image, gt_image), 6)
        ssim_scores[file_name] = round(ssim(pred_image, gt_image), 6)

    results = {
        "num_frames": len(psnr_scores),
        "psnr_mean": round(float(np.mean(list(psnr_scores.values()))) if psnr_scores else 0.0, 6),
        "ssim_mean": round(float(np.mean(list(ssim_scores.values()))) if ssim_scores else 0.0, 6),
        "per_frame_psnr": psnr_scores,
        "per_frame_ssim": ssim_scores,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
