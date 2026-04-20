# Assignment 3: UAV Drone Detection and Tracking

## Overview

A complete multi-object detection and tracking pipeline for drones in video. The system combines **SAHI** (Slicing Aided Hyper Inference) with a fine-tuned **YOLO11x** detector to locate drones frame-by-frame — including small/distant targets that standard inference misses — then a **SORT-style multi-object tracker** with Kalman filters and Hungarian assignment to maintain smooth trajectory tracking across frames, including through detection gaps.

## Dataset Choice

**Dataset:** [`pathikg/drone-detection-dataset`](https://huggingface.co/datasets/pathikg/drone-detection-dataset) on Hugging Face

- **Single-class (`drone`)** — not a broad multi-class aerial benchmark
- **54.1k total images** (51.4k train, 2.63k test) — large enough for robust fine-tuning
- **Schema:** `image`, `width`, `height`, `image_id`, `objects` with `[x, y, w, h]` bounding boxes
- Already in Parquet-backed HF format, matching the assignment's Parquet deliverable requirement

### Why this dataset over alternatives

The assignment requires detecting **the drone itself** — not analyzing imagery captured from a drone. `pathikg/drone-detection-dataset` is ground-view footage where drones are the labeled targets. VisDrone, while mentioned in the assignment, is primarily an aerial perspective benchmark not built around ground-view drone detection.

### Optional augmentation

For the strongest detector, the pipeline supports adding hard-negative images (sky, birds, empty backgrounds) via `prepare_dataset.py --negative-dir`, which creates empty YOLO labels for background images to reduce false positives.

## Detector: YOLO11x (Ultralytics) + SAHI Sliced Inference

The pipeline uses the [Ultralytics](https://docs.ultralytics.com/) deep learning framework for drone detection. The current checkpoint uses **YOLO11x** — the extra-large variant of YOLO11 (released September 2024) — from [`doguilmak/Drone-Detection-YOLOv11x`](https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x), which was fine-tuned specifically on drone imagery. YOLO11x achieves **precision 0.922, recall 0.831, mAP@50 0.905** on the validation set with ~8.9ms inference latency, representing the state of the art for single-class drone detection. The Ultralytics framework also supports the newer YOLO26 (January 2026), but no drone-specific YOLO26 checkpoint is publicly available yet.

### SAHI: Slicing Aided Hyper Inference

Standard object detectors struggle with small/distant drones in high-resolution footage. When a 1920x1080 frame is resized to the model's input size (1280px), a drone occupying 30x20 pixels shrinks below the detector's effective resolution.

[**SAHI**](https://github.com/obss/sahi) solves this by partitioning each frame into overlapping tiles (default: 640x640 with 20% overlap), running YOLO inference on each tile at full resolution, then merging detections across tiles via NMS. This dramatically improves recall for small objects without retraining the model.

Key SAHI settings:
- **Slice dimensions:** 640x640 (matches YOLO's native training resolution)
- **Overlap ratio:** 0.2 (20% overlap prevents missed detections at tile boundaries)
- **Enabled by default;** disable with `--no-sahi` to use direct YOLO inference

### Inference settings

- **Confidence threshold:** 0.10 (aggressive to catch small/distant drones)
- **Image size:** 1280px (high resolution for small object detection)
- **Frame sampling:** 10 FPS (high temporal resolution for smooth tracking)
- **Test-time augmentation (TTA):** Enabled — runs multi-scale inference which significantly improves detection of small drones at varying distances
- **Bounding box geometry filters:** Rejects false-positive detections where the box exceeds 3% of frame area, 15% of frame width, or 15% of frame height — drones are small objects, so any oversized detection is almost certainly a false positive (e.g., treeline or sky region)

The detector automatically resolves the "drone" class from the model's label map using fuzzy matching on `drone`, `uav`, and `quadcopter`. For single-class checkpoints, it falls back to using the only available class.

## Multi-Object Tracking: SORT with Kalman Filters

The tracker implements the **SORT (Simple Online and Realtime Tracking)** algorithm, combining per-object Kalman filters with frame-by-frame data association via the Hungarian algorithm.

### Why SORT

SORT is the standard approach for real-time multi-object tracking. It provides:
- **Automatic track management** — new tracks are created for unmatched detections, stale tracks are deleted
- **Optimal assignment** — the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) finds the global minimum-cost matching between detections and existing tracks using IoU
- **Scalability** — handles any number of simultaneous drones without code changes

### Kalman Filter Design (7-state SORT model)

Each tracked object maintains its own [`filterpy.kalman.KalmanFilter`](https://filterpy.readthedocs.io/) with a constant-velocity motion model that tracks both position and bounding box geometry.

| Component | Value | Description |
|-----------|-------|-------------|
| **State** `x` | `[cx, cy, s, r, vx, vy, vs]` | Center position, area, aspect ratio + velocities |
| **Measurement** `z` | `[cx, cy, s, r]` | Bounding box center, area, and aspect ratio from detector |
| **Transition** `F` | 7x7 constant-velocity with `dt = 1/fps` | Predicts next state from current state + velocities |
| **Observation** `H` | 4x7 matrix | Extracts `[cx, cy, s, r]` from full state |
| **Measurement noise** `R` | `diag([1, 1, 10, 10])` | Tight position trust, looser area/ratio trust |
| **Process noise** `Q` | Tuned per SORT defaults | Higher velocity uncertainty to handle maneuvering |

The 7-state model (vs. a simpler 4-state center-only model) tracks bounding box dimensions, which enables **IoU-based data association** — the key to correctly matching detections to tracks when multiple drones are present.

### Data association (per frame)

1. **Predict** all existing tracks forward using their Kalman filters
2. **Build IoU cost matrix** between predicted track bounding boxes and new detections
3. **Hungarian assignment** finds the optimal detection-to-track matching that maximizes total IoU
4. **Update** matched tracks with their assigned detections
5. **Create** new tracks for unmatched detections
6. **Delete** tracks that have gone unmatched for more than `max_age` frames (default: 50)

### Track lifecycle

- **Tentative:** New tracks require `min_hits` (default: 3) consecutive detections before being confirmed — this filters out spurious single-frame false positives
- **Confirmed:** Tracks with sufficient hits are rendered in the output video with a unique color and ID
- **Predicted:** When a confirmed track misses a detection, the Kalman filter continues predicting its position; the output video marks these frames with a "PREDICTED" badge
- **Deleted:** Tracks exceeding `max_age` (50 frames / 5 seconds) without a matching detection are removed

## Output Video Visualization

Each output frame includes:

- **Per-track colored bounding box** with corner accents, track ID, and confidence score
- **Per-track trajectory polyline** with a fade effect (older points dimmer, recent points brighter) in the track's assigned color
- **Center point:** solid circle (measured) or hollow circle (predicted), in track color
- **Status badge:** green "TRACKING N" showing active track count, orange "PREDICTED N" during gaps, red "NO TRACKS" when lost
- **Frame counter** showing current position in the video

Each track is assigned a visually distinct color via golden-ratio hue spacing, ensuring that simultaneously tracked objects are easy to distinguish.

## Failure Cases

- **Tiny drones at long range** — SAHI significantly improves recall for small objects, but extremely distant drones (< 10px) may still be missed
- **Bright haze / low contrast** — compression artifacts and atmospheric conditions reduce detector confidence
- **Birds and aircraft** — the main semantic confusers; hard-negative training images help but don't eliminate this
- **Very long dropouts** (>50 frames / 5 seconds) — force the tracker to terminate and reinitialize, creating a new track ID
- **Fast lateral maneuvers** — the constant-velocity model can lag behind sudden direction changes, though the process noise parameter helps compensate
- **SAHI speed tradeoff** — sliced inference is ~4-6x slower per frame than direct YOLO inference due to multiple tile passes; disable with `--no-sahi` when speed is critical
- **Track ID fragmentation** — if a drone is lost and re-detected, it receives a new track ID (standard SORT behavior); a re-identification model would be needed for cross-gap identity persistence

## Usage

All commands go through `main.py`:

```bash
# Task 1: save detection frames only (SAHI + multi-object tracking)
python main.py detect --weights best.pt

# Task 2: detection + tracking + output videos (default, SAHI enabled)
python main.py track --weights best.pt --overwrite-outputs

# Without SAHI (faster, lower recall on small drones)
python main.py track --weights best.pt --no-sahi --overwrite-outputs

# Custom SAHI tile size (larger tiles = faster but less small-object boost)
python main.py track --weights best.pt --sahi-slice-height 960 --sahi-slice-width 960

# Package detections as HF Parquet dataset
python main.py upload --detections-dir detections

# Push to Hugging Face
python main.py upload --detections-dir detections --repo-id YOUR_USER/drone-detections
```

### Training (optional — a pre-trained checkpoint is used)

```bash
# 1. Prepare dataset
python prepare_dataset.py --dataset-id pathikg/drone-detection-dataset --output-dir data/drone

# 2. Fine-tune
python train_detector.py --data data/drone/data.yaml --epochs 30 --device cuda
```

## Deliverables

- **Hugging Face dataset:** [HarshAgarwalNYU/Assignment3Drone](https://huggingface.co/datasets/HarshAgarwalNYU/Assignment3Drone)
- **Output video 1 (YouTube):** https://youtu.be/yjuZ6CPiang
- **Output video 2 (YouTube):** https://youtu.be/XuKAufi5Ngc
