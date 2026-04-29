# Video Object Removal and Inpainting (Part 1, 2, 3)

This repository contains our reproducible implementation of the complete computer vision pipeline for video object removal and inpainting, split across three evolutionary parts.

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

## 3. Core Components / Third-Party Codebases

This repository utilizes three major codebases to advance from a baseline to a state-of-the-art pipeline. The source code for these is included to ensure the full project (`p1`, `p2`, `p3`) is reproducible:

1. **YOLOv8 & OpenCV** (Built-in via `ultralytics` and `cv2` for Part 1)
2. **SAM2** (`sam2/` folder, required for Part 2 and Part 3)
3. **ProPainter** (`ProPainter/` folder, required for Part 2 and Part 3)
4. **FreeInpaint** (`FreeInpaint-main/` folder, required for Part 3)

*Note: Heavy model weights and data are intentionally ignored via `.gitignore` to comply with storage best practices.*

## 4. Checkpoints and Weights Downloading

Before running the evaluation scripts, ensure you download the required checkpoints:

### 4.1 YOLO Segmentation (Part 1)
- The pipeline uses `yolov8n-seg.pt` by default. It will be downloaded automatically by the Ultralytics API on the first run.

### 4.2 SAM2 Checkpoint (Part 2 & 3)
- Download the SAM2 checkpoint into `sam2/checkpoints/sam2.1_hiera_large.pt`. 

### 4.3 ProPainter Weights (Part 2 & 3)
- Place required weights under `ProPainter/weights/` (e.g., `ProPainter.pth`, `raft-things.pth`, `recurrent_flow_completion.pth` from their official release).

### 4.4 SDXL + FreeInpaint Weights (Part 3)
Run our helper script to download the model from ModelScope into `checkpoints/sdxl-inpainting-0.1-modelscope/`:
```bash
python scripts/download_sdxl_inpaint_modelscope.py
```

## 5. Running the Pipeline

Our unified codebase architecture provides identical interfaces for running Part 1, Part 2, or Part 3 via their respective configurations. 

### Part 1: Baseline Pipeline (YOLOv8 + Optical Flow + Telea)
```bash
python scripts/run_part1.py \
  --set input.video_path=/absolute/path/to/your_video.mp4 \
  --set output.dataset_name=my_part1_results
```
*Note: This strictly disables SAM2 and ProPainter, falling back to bounding-box propagation and frame-by-frame Navier-Stokes/Telea inpainting.*

### Part 2: AI-Driven Pipeline (SAM2 + ProPainter)
```bash
python scripts/run_part2.py \
  --set input.video_path=/absolute/path/to/your_video.mp4 \
  --set output.dataset_name=my_part2_results
```
*Note: This utilizes state-of-the-art segmentation and temporal flow completion, without the final diffusion stage.*

### Part 3: Advanced Pipeline (SAM2 + ProPainter + Diffusion Enhancement)
```bash
python scripts/run_part3.py \
  --config configs/part3_wildvideo_remove_person_camera_sparse_diffusion.yaml \
  --set input.video_path=/absolute/path/to/your_video.mp4 \
  --set output.dataset_name=my_part3_results
```
*Note: Uses Stable Diffusion XL (SDXL) via FreeInpaint to re-texture failed large inpaint regions, guided by Distance Transform Alpha.*

## 6. Evaluation

After generating output frames, evaluate your run with:

```bash
python scripts/evaluate_part3_run.py \
  --run_dir results/part3/your_dataset_name/<timestamp>
```

Metrics (including Flow Warping Error, LPIPS, and SSIM) will be exported to `independent_eval.json` inside your run directory.

## 7. Main Entry Files

- `scripts/run_part1.py`: Baseline implementation.
- `scripts/run_part2.py`: AI-Driven implementation.
- `scripts/run_part3.py`: Final advanced implementation.
- `src/cv_project/pipeline/part3.py`: Unified end-to-end orchestration algorithm.
- `scripts/evaluate_part3_run.py`: Independent run evaluation.
