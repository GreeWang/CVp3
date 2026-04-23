from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "AI-ModelScope/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "sdxl-inpainting-0.1-modelscope"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download SD-XL Inpainting 0.1 from ModelScope.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="ModelScope model id.")
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Local directory used by diffusers AutoPipelineForInpainting.",
    )
    parser.add_argument(
        "--fp16-only",
        action="store_true",
        help="Download configs/tokenizers plus fp16 weight variants only.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of concurrent ModelScope download workers.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "modelscope is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    download_kwargs: dict[str, object] = {
        "local_dir": str(output_dir),
        "max_workers": args.max_workers,
    }
    if args.fp16_only:
        download_kwargs["allow_patterns"] = [
            "*.json",
            "*.txt",
            "*.model",
            "*.png",
            "*.md",
            "*.fp16.safetensors",
        ]

    model_path = snapshot_download(args.model_id, **download_kwargs)
    print(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
