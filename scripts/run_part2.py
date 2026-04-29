from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Part 2 runs the pipeline backbone with SAM2 and ProPainter.
from cv_project.pipeline.part3 import run_part3_pipeline
from cv_project.utils.config import load_config, parse_overrides

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Part 2 AI-driven pipeline (SAM2 + ProPainter).")
    parser.add_argument("--config", default="configs/part2_sam2_propainter.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values. Example: --set input.video_path=data.mp4",
    )
    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config))
    overrides = parse_overrides(args.set)
    for key, value in overrides.items():
        config.set_value(key, value)

    summary = run_part3_pipeline(config.to_dict(), project_root=PROJECT_ROOT)
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
