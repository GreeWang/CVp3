# CV Part 3: Video Object Removal & Inpainting

This repository contains our Part 3 pipeline for automatic dynamic object removal in videos.
It combines:

- dynamic object segmentation (YOLO + SAM2 video propagation)
- temporal restoration (ProPainter-based branch)
- selective diffusion enhancement (SDXL/FreeInpaint branch)
- independent gain-risk evaluation for artifact and temporal analysis

## Repository Scope

To keep this repository lightweight and reproducible, large assets are not committed:

- raw datasets and generated outputs
- model checkpoints/weights
- locally cloned third-party repositories

See `.gitignore` for details.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required External Repositories

Clone these repos under the project root:

```bash
git clone https://github.com/sczhou/ProPainter.git
git clone https://github.com/facebookresearch/sam2.git
git clone https://github.com/fallenshock/FreeInpaint.git FreeInpaint-main
```

Note: the default configs in this repo expect folder names `ProPainter/`, `sam2/`, and `FreeInpaint-main/`.

## Weights

Prepare the following weights locally (not tracked by git):

- YOLO segmentation weight (for example `yolov8n-seg.pt`)
- SAM2 checkpoint (for example `checkpoints/sam2.1_hiera_large.pt`)
- SDXL inpainting model (download helper script available):

```bash
python scripts/download_sdxl_inpaint_modelscope.py
```

## Run Pipeline

```bash
python scripts/run_part3.py --config configs/part3_wildvideo_remove_person_camera_sparse_diffusion.yaml
```

You can override config values inline:

```bash
python scripts/run_part3.py \
  --config configs/part3_wildvideo_remove_person_camera_sparse_diffusion.yaml \
  --set output.dataset_name=wildvideo
```

## Evaluate a Run

```bash
python scripts/evaluate_part3_run.py \
  --run_dir results/part3/<dataset>/<timestamp>
```

This writes an `independent_eval.json` with spatial/temporal metrics and a gain-risk composite score.

## Main Code Paths

- `scripts/run_part3.py`: entrypoint for running Part 3 pipeline
- `scripts/evaluate_part3_run.py`: independent run evaluator
- `src/cv_project/pipeline/part3.py`: orchestration of full pipeline
- `src/cv_project/evaluation/run_metrics.py`: metric definitions and aggregation
- `configs/`: experiment configurations

## Reports

Report sources are included in the root directory (`*.tex`, `report_zh.md`, `p3.tex`).