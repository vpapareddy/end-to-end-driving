## WOD Vision-based End-to-End Driving

## Overview
Implement and evaluate an end-to-end autonomous driving pipeline on Waymo’s long-tail dataset.
- **Backbone & Head:** Multi-view ResNet-50 + lightweight MLP to predict future 5s waypoints
- **Performance:** Reduced ADE by 68% (to 3.9 m) on held-out validation
- **Data Pipeline:** Cache-enabled TFRecord streaming of 4,021 segments from GCS, optimized for real-time edge-GPU inference

## Challenge Description
This project targets the Waymo Open Dataset (WOD) Vision-based End-to-End Driving Challenge, which focuses on rare “long-tail” scenarios (e.g., navigating construction zones, avoiding pedestrians falling from scooters, unexpected freeway obstacles).
1. **Objective:** Given 20 s of recorded data (8 cameras, past poses, routing), predict the bird’s-eye-view waypoints for the next 5 s.
2. **Evaluation:**
   - **Primary:** Rater feedback metric (human-rated waypoint quality)
   - **Tie-breaker:** Average Displacement Error (ADE)

## Dataset
- **Total Segments:** 4,021 (each 20 s long)
  - **Training:** 2,037 segments (full 20 s available)
  - **Validation:** 479 segments (full 20 s available)
  - **Testing:** Remaining segments; participants receive only first 12 s, then predict the final 8 s
- **Data Modalities:**
  - 8 synchronized camera streams (360° view)
  - Agent’s historical poses
  - Routing waypoints
- **Long-Tail Focus:**
  - Events <0.003% frequency in daily driving (e.g., construction, sudden obstacles)
  - Designed to stress-test robustness and generalization

## Key Features
1. **Model Architecture (PyTorch)**
   - **Backbone:** Multi-view ResNet-50
   - **Head:** Lightweight MLP projecting camera features to 5 s future waypoints (bird’s-eye-view)
   - **Outcome:** 68% ADE reduction → 3.9 m average error
2. **Data Pipeline (TensorFlow + TFRecord)**
   - Streams 4,021 segments directly from Google Cloud Storage
   - Cache-enabled reader minimizes I/O latency and memory footprint
   - Enables real-time inference on edge GPUs (e.g., Jetson, Drive AGX)
3. **End-to-End Driving Context**
   - Contrasts with modular pipelines (perception→prediction→planning)
   - Leverages recent LLM-inspired reasoning for direct raw-sensor→action mapping

## Dependencies
- **Python 3.8+**
- **PyTorch 1.x**
- **TensorFlow 2.x** (for TFRecord reading)
- **Waymo Open Dataset utilities** (https://github.com/waymo-research/waymo-open-dataset)
- **CUDA & cuDNN** (for GPU acceleration)
- Common libraries: `numpy`, `opencv-python`, `tqdm`, `matplotlib`

## Training
```bash
# Example: train on multi-view ResNet-50 + MLP head
python train.py   --config configs/config.yaml   --split train   --epochs 50   --batch_size 16   --lr 1e-4
```
- **Key config options** (in `configs/config.yaml`):
  - `model.backbone`: `resnet50_multiview`
  - `data.cache`: `true` (enable TFRecord caching)
  - `optimizer`: `AdamW`, `lr`, `weight_decay`

## Validation & Evaluation
```bash
# Run validation/inference to compute ADE and log rater feedback proxy
python evaluate.py   --config configs/config.yaml   --split val
```
- **Outputs:**
  - numeric ADE (3.9 m target)
  - simulated human-rating feedback scores

## Offline Inference / Edge Deployment
- **Generate TF-Lite**
  ```bash
  python export_model.py     --model checkpoints/best.pth
  ```

## Results
| Metric           | Baseline (ADE) | This Model       | Improvement |
| ---------------- | -------------- | ---------------- | ----------- |
| Val ADE (5 s)    | 12.2 m         | 3.9 m            | –68%        |
| Rater Feedback   | 5.15           | 6.47             | +25%        |

