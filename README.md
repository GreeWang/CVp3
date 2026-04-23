# CV Part 3: Video Object Removal and Inpainting

This repository contains our reproducible Part 3 implementation:

- dynamic object segmentation (YOLO + SAM2 video propagation)
- temporal inpainting/restoration (ProPainter branch)
- selective diffusion enhancement (SDXL + FreeInpaint branch)
- independent gain-risk evaluation (outside/seam/leakage/score)

Large assets are intentionally not tracked in git (datasets, checkpoints, generated results, third-party repos).

## 1. System Requirements

- Linux (tested on Ubuntu 22.04)
- Python 3.10+
- CUDA-capable GPU recommended

## 2. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Third-Party Repositories

Clone these repositories in the project root directory with the exact folder names below:

```bash
git clone https://github.com/sczhou/ProPainter.git
git clone https://github.com/facebookresearch/sam2.git
git clone https://github.com/fallenshock/FreeInpaint.git FreeInpaint-main
```

If the FreeInpaint URL is unavailable in your environment, use a compatible local copy and ensure the folder name is still `FreeInpaint-main`.

## 4. Checkpoints and Weights

### 4.1 YOLO Segmentation

- Default config uses `yolov8n-seg.pt`.
- Ultralytics can auto-download it, or you can place the file manually at repo root.

### 4.2 SAM2 Checkpoint

Download SAM2 checkpoints and keep this file available:

- `sam2/checkpoints/sam2.1_hiera_large.pt`

### 4.3 ProPainter Weights

Place required weights under `ProPainter/weights/` (see ProPainter README for official release files).

### 4.4 SDXL Inpainting Weights

Run:

```bash
python scripts/download_sdxl_inpaint_modelscope.py
```

This downloads to `checkpoints/sdxl-inpainting-0.1-modelscope/`.

## 5. Data Layout

Default wildvideo config expects:

- `data/raw/wild/wild_video.mp4`

You can also avoid fixed paths by overriding config values from CLI.

## 6. Reproducible Run Commands

### 6.1 Run with a Custom Video Path (Recommended)

```bash
python scripts/run_part3.py \
  --config configs/part3_wildvideo_remove_person_camera_sparse_diffusion.yaml \
  --set input.video_path=/absolute/path/to/your_video.mp4 \
  --set output.dataset_name=your_dataset_name
```

### 6.2 Run with Frame Directory Input

```bash
python scripts/run_part3.py \
  --config configs/part3_explore.yaml \
  --set input.video_path=null \
  --set input.frames_dir=/absolute/path/to/frames_dir \
  --set output.dataset_name=your_frames_dataset
```

Outputs are written to:

- `results/part3/<dataset_name>/<timestamp>/`

## 7. Evaluation

```bash
python scripts/evaluate_part3_run.py \
  --run_dir results/part3/<dataset_name>/<timestamp>
```

Main evaluation artifact:

- `independent_eval.json`

## 8. Main Entry Files

- `scripts/run_part3.py`: pipeline entrypoint
- `scripts/evaluate_part3_run.py`: independent run evaluation
- `src/cv_project/pipeline/part3.py`: end-to-end orchestration
- `src/cv_project/evaluation/run_metrics.py`: metric definitions and aggregation
- `configs/`: experiment presets