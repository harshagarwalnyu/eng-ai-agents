"""Detection + Kalman tracking pipeline for drone videos.

Supports SAHI sliced inference for small-object detection and SORT-style
multi-object tracking with Hungarian assignment.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from ultralytics import YOLO

from tracker import (
    BBox,
    MultiObjectTracker,
    MultiTrackerState,
    TrackerState,
    iou,
)

VIDEO_EXTENSIONS = {".mp4"}
SUMMARY_PATH = Path("artifacts/pipeline_summary.json")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PipelineConfig:
    videos_dir: Path
    frames_dir: Path
    detections_dir: Path
    output_videos_dir: Path
    summary_path: Path
    weights: Path
    fps: float = 10.0
    conf: float = 0.10
    imgsz: int = 1280
    max_missing_frames: int = 50
    trajectory_points: int = 300
    overwrite_frames: bool = False
    overwrite_outputs: bool = False
    render_output_videos: bool = True
    device: str | None = None
    half: bool = False
    drone_class_name: str | None = "drone"
    drone_class_id: int | None = None
    max_frames: int | None = None
    augment: bool = True
    max_box_area_ratio: float | None = 0.03
    max_box_width_ratio: float | None = 0.15
    max_box_height_ratio: float | None = 0.15
    # SAHI sliced inference
    use_sahi: bool = True
    sahi_slice_height: int = 640
    sahi_slice_width: int = 640
    sahi_overlap_height_ratio: float = 0.2
    sahi_overlap_width_ratio: float = 0.2
    # SORT multi-object tracker
    sort_max_age: int = 50
    sort_min_hits: int = 3
    sort_iou_threshold: float = 0.3


@dataclass(slots=True)
class FrameDetection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(slots=True)
class VideoSummary:
    video_name: str
    input_video: str
    frame_dir: str
    detection_frames: int
    sampled_frames: int
    output_video: str | None
    detector_weights: str
    detector_class_ids: list[int]
    rendered_frames: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_writable(path: Path) -> bool:
    if path.exists():
        return path.is_dir() and os.access(path, os.W_OK | os.X_OK)
    return os.access(path.parent, os.W_OK | os.X_OK)


def writable_dir(requested: Path) -> Path:
    if is_writable(requested):
        return requested
    fallback = Path("artifacts") / requested.name
    ensure_dir(fallback)
    return fallback


def canonical_name(name: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return "".join(tokens[1:] if tokens[:1] == ["drone"] else tokens)


def find_frame_dir(video: Path, root: Path) -> Path:
    direct = root / video.stem
    if direct.exists():
        return direct

    if root.exists():
        matches = [
            d for d in root.iterdir()
            if d.is_dir() and canonical_name(d.name) == canonical_name(video.stem)
        ]
        if len(matches) == 1:
            return matches[0]

    base = root if is_writable(root) else writable_dir(root)
    return base / video.stem


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video: Path, dest: Path, fps: float, overwrite: bool) -> list[Path]:
    existing = sorted(dest.glob("*.jpg"))
    if existing and not overwrite:
        return existing

    ensure_dir(dest)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vf", f"fps={fps}",
        str(dest / "frame_%06d.jpg"),
    ], check=True)

    return sorted(dest.glob("*.jpg"))


def discover_videos(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)


# ---------------------------------------------------------------------------
# Class resolution
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    return name.strip().lower().replace("-", " ").replace("_", " ")


def resolve_drone_class(model: YOLO, name: str | None, cid: int | None) -> list[int]:
    names = model.names
    labels = (
        {i: v for i, v in enumerate(names)}
        if isinstance(names, list)
        else {int(k): v for k, v in names.items()}
    )

    if cid is not None:
        if cid not in labels:
            raise ValueError(f"class {cid} not in model: {labels}")
        return [cid]

    if name:
        ids = []
        for req in (s.strip() for s in name.split(",") if s.strip()):
            target = _norm(req)
            exact = [i for i, v in labels.items() if _norm(v) == target]
            if exact:
                ids.extend(exact)
            else:
                partial = [i for i, v in labels.items() if target in _norm(v)]
                if partial:
                    ids.extend(partial)
                else:
                    raise ValueError(f"'{req}' not found in model: {labels}")
        if ids:
            return sorted(set(ids))

    auto = [i for i, v in labels.items()
            if any(t in _norm(v) for t in ("drone", "uav", "quadcopter"))]
    if auto:
        return auto

    if len(labels) == 1:
        return [next(iter(labels))]

    raise ValueError(f"Cannot infer drone class from: {labels}")


# ---------------------------------------------------------------------------
# Detection selection with geometry filtering
# ---------------------------------------------------------------------------

def _box_ok(bbox: BBox, shape: tuple[int, int],
            max_area: float | None, max_w: float | None, max_h: float | None) -> bool:
    ih, iw = shape
    if iw <= 0 or ih <= 0:
        return False

    bw = max(0.0, bbox[2] - bbox[0])
    bh = max(0.0, bbox[3] - bbox[1])
    if bw <= 0 or bh <= 0:
        return False

    if max_area and (bw * bh) / (iw * ih) > max_area:
        return False
    if max_w and bw / iw > max_w:
        return False
    if max_h and bh / ih > max_h:
        return False
    return True


def pick_detection(result, class_ids: set[int],
                   max_area: float | None, max_w: float | None, max_h: float | None
                   ) -> FrameDetection | None:
    if not len(result.boxes):
        return None

    names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))
    best = None

    for box in result.boxes:
        cls = int(box.cls.item())
        if cls not in class_ids:
            continue

        conf = float(box.conf.item())
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        bbox = (x1, y1, x2, y2)

        if not _box_ok(bbox, result.orig_shape, max_area, max_w, max_h):
            continue

        candidate = FrameDetection(bbox, conf, cls, str(names[cls]))
        if best is None or conf > best.confidence:
            best = candidate

    return best


def pick_all_detections_yolo(
    result, class_ids: set[int],
    max_area: float | None, max_w: float | None, max_h: float | None,
) -> list[FrameDetection]:
    """Extract all valid detections from a YOLO result."""
    if not len(result.boxes):
        return []
    names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))
    dets: list[FrameDetection] = []
    for box in result.boxes:
        cls = int(box.cls.item())
        if cls not in class_ids:
            continue
        conf = float(box.conf.item())
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        bbox: BBox = (x1, y1, x2, y2)
        if not _box_ok(bbox, result.orig_shape, max_area, max_w, max_h):
            continue
        dets.append(FrameDetection(bbox, conf, cls, str(names[cls])))
    return dets


def pick_all_detections_sahi(
    sahi_result, class_ids: set[int], shape: tuple[int, int],
    max_area: float | None, max_w: float | None, max_h: float | None,
) -> list[FrameDetection]:
    """Extract all valid detections from a SAHI sliced prediction result."""
    dets: list[FrameDetection] = []
    for pred in sahi_result.object_prediction_list:
        cls = pred.category.id
        if cls not in class_ids:
            continue
        x1, y1, x2, y2 = pred.bbox.to_xyxy()
        bbox: BBox = (float(x1), float(y1), float(x2), float(y2))
        if not _box_ok(bbox, shape, max_area, max_w, max_h):
            continue
        dets.append(FrameDetection(
            bbox=bbox,
            confidence=float(pred.score.value),
            class_id=cls,
            class_name=str(pred.category.name) if pred.category.name else "drone",
        ))
    return dets


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_label(frame, text: str, x: int, y: int) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    y = max(y, th + bl + 4)
    cv2.rectangle(frame, (x, y - th - bl - 6), (x + tw + 8, y + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 4, y - bl - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_overlays(frame, det: FrameDetection | None, state: TrackerState,
                  trail_len: int, frame_idx: int, total: int) -> None:
    h, w = frame.shape[:2]

    # Bounding box or predicted crosshair
    if det:
        ix1, iy1 = int(det.bbox[0]), int(det.bbox[1])
        ix2, iy2 = int(det.bbox[2]), int(det.bbox[3])
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)

        corner = max(10, min(ix2 - ix1, iy2 - iy1) // 4)
        for cx, cy, dx, dy in [
            (ix1, iy1, 1, 1), (ix2, iy1, -1, 1),
            (ix1, iy2, 1, -1), (ix2, iy2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner, cy), (0, 255, 0), 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner), (0, 255, 0), 3)

        draw_label(frame, f"drone {det.confidence:.2f}", ix1, iy1)

    elif state.predicted and state.estimated_center:
        cx, cy = state.estimated_center
        for i in range(0, 30, 6):
            cv2.line(frame, (cx - 30 + i, cy), (cx - 27 + i, cy), (0, 165, 255), 1)
            cv2.line(frame, (cx + i, cy), (cx + i + 3, cy), (0, 165, 255), 1)
            cv2.line(frame, (cx, cy - 30 + i), (cx, cy - 27 + i), (0, 165, 255), 1)
            cv2.line(frame, (cx, cy + i), (cx, cy + i + 3), (0, 165, 255), 1)

    # Trajectory with gradient
    for seg in state.trajectory_segments:
        pts = seg[-trail_len:]
        n = len(pts)
        if n < 2:
            continue
        for i in range(1, n):
            t = i / n
            alpha = 0.2 + 0.8 * t
            color = (int(255 * alpha), int(140 * alpha), 0)
            cv2.line(frame, pts[i - 1], pts[i], color, max(1, int(1 + 2 * t)))

    # Center marker
    if state.estimated_center:
        if state.predicted:
            cv2.circle(frame, state.estimated_center, 8, (0, 165, 255), 2)
            cv2.circle(frame, state.estimated_center, 3, (0, 165, 255), -1)
        else:
            cv2.circle(frame, state.estimated_center, 6, (0, 0, 255), -1)
            cv2.circle(frame, state.estimated_center, 8, (255, 255, 255), 1)

    # HUD
    draw_label(frame, f"Frame {frame_idx}/{total}", w - 220, 30)

    if state.predicted:
        label, color = f"PREDICTED (miss {state.missing_frames})", (0, 165, 255)
    elif state.active:
        label, color = "TRACKING", (0, 255, 0)
    else:
        label, color = "NO TRACK", (0, 0, 255)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
    cv2.rectangle(frame, (10, 30 - th - bl - 6), (10 + tw + 12, 32), color, -1)
    cv2.putText(frame, label, (16, 30 - bl - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def track_color(track_id: int) -> tuple[int, int, int]:
    """Generate a stable, visually distinct BGR color from a track ID."""
    golden = 0.618033988749895
    hue = int(((track_id * golden) % 1.0) * 180)
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _find_detection_conf(dets: list[FrameDetection], track_bbox: BBox) -> float | None:
    """Find the detection that best overlaps a track's bbox; return its confidence."""
    best_iou, best_conf = 0.0, None
    for det in dets:
        overlap = iou(det.bbox, track_bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_conf = det.confidence
    return best_conf if best_iou > 0.1 else None


def draw_multi_overlays(
    frame, dets: list[FrameDetection], mstate: MultiTrackerState,
    trail_len: int, frame_idx: int, total: int,
) -> None:
    """Draw multi-object tracking overlays: per-track bbox, trajectory, center, HUD."""
    h, w = frame.shape[:2]

    for ts in mstate.tracks:
        if not ts.active:
            continue
        color = track_color(ts.track_id)

        if not ts.predicted:
            ix1, iy1 = int(ts.bbox[0]), int(ts.bbox[1])
            ix2, iy2 = int(ts.bbox[2]), int(ts.bbox[3])
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)

            corner = max(10, min(ix2 - ix1, iy2 - iy1) // 4)
            for cx, cy, dx, dy in [
                (ix1, iy1, 1, 1), (ix2, iy1, -1, 1),
                (ix1, iy2, 1, -1), (ix2, iy2, -1, -1),
            ]:
                cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, 3)
                cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, 3)

            conf = _find_detection_conf(dets, ts.bbox)
            label = f"drone #{ts.track_id}"
            if conf is not None:
                label += f" {conf:.2f}"
            draw_label(frame, label, ix1, iy1)

        elif ts.estimated_center:
            cx, cy = ts.estimated_center
            for i in range(0, 30, 6):
                cv2.line(frame, (cx - 30 + i, cy), (cx - 27 + i, cy), color, 1)
                cv2.line(frame, (cx + i, cy), (cx + i + 3, cy), color, 1)
                cv2.line(frame, (cx, cy - 30 + i), (cx, cy - 27 + i), color, 1)
                cv2.line(frame, (cx, cy + i), (cx, cy + i + 3), color, 1)

        # Trajectory with gradient in track color
        pts = ts.trajectory[-trail_len:]
        n = len(pts)
        if n >= 2:
            for i in range(1, n):
                t = i / n
                alpha = 0.2 + 0.8 * t
                c = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                cv2.line(frame, pts[i - 1], pts[i], c, max(1, int(1 + 2 * t)))

        # Center marker
        if ts.estimated_center:
            if ts.predicted:
                cv2.circle(frame, ts.estimated_center, 8, color, 2)
                cv2.circle(frame, ts.estimated_center, 3, color, -1)
            else:
                cv2.circle(frame, ts.estimated_center, 6, color, -1)
                cv2.circle(frame, ts.estimated_center, 8, (255, 255, 255), 1)

    # HUD — frame counter
    draw_label(frame, f"Frame {frame_idx}/{total}", w - 220, 30)

    # HUD — track count badge
    n_active = sum(1 for ts in mstate.tracks if ts.active and not ts.predicted)
    n_predicted = sum(1 for ts in mstate.tracks if ts.active and ts.predicted)
    if n_active > 0:
        badge_label = f"TRACKING {n_active}"
        if n_predicted:
            badge_label += f" (+{n_predicted} pred)"
        badge_color = (0, 255, 0)
    elif n_predicted > 0:
        badge_label = f"PREDICTED {n_predicted}"
        badge_color = (0, 165, 255)
    else:
        badge_label = "NO TRACKS"
        badge_color = (0, 0, 255)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), bl = cv2.getTextSize(badge_label, font, scale, thick)
    cv2.rectangle(frame, (10, 30 - th - bl - 6), (10 + tw + 12, 32), badge_color, -1)
    cv2.putText(frame, badge_label, (16, 30 - bl - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Video composition
# ---------------------------------------------------------------------------

def compose_video(frame_dir: Path, output: Path, fps: float) -> None:
    ensure_dir(output.parent)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output),
    ], check=True)


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(video: Path, model: YOLO, cfg: PipelineConfig,
                  class_ids: list[int], sahi_model=None) -> VideoSummary:
    name = video.stem
    frame_dir = find_frame_dir(video, cfg.frames_dir)
    frames = extract_frames(video, frame_dir, cfg.fps, cfg.overwrite_frames)

    if cfg.max_frames:
        frames = frames[:cfg.max_frames]
    if not frames:
        raise RuntimeError(f"No frames for {video}")

    tracker = MultiObjectTracker(
        fps=cfg.fps,
        max_age=cfg.sort_max_age,
        min_hits=cfg.sort_min_hits,
        iou_threshold=cfg.sort_iou_threshold,
    )
    det_count = 0
    rendered = 0
    out_path: Path | None = None

    predict_kwargs = dict(
        verbose=False, conf=cfg.conf, imgsz=cfg.imgsz,
        classes=class_ids, half=cfg.half, augment=cfg.augment,
    )
    if cfg.device:
        predict_kwargs["device"] = cfg.device

    id_set = set(class_ids)

    # Pre-import SAHI predict function if needed (avoid import inside hot loop)
    _get_sliced_prediction_fn = None
    if cfg.use_sahi and sahi_model is not None:
        from sahi.predict import get_sliced_prediction
        _get_sliced_prediction_fn = get_sliced_prediction

    with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp:
        render_dir = Path(tmp)

        for idx, fpath in enumerate(tqdm(frames, desc=name, unit="fr"), 1):
            frame_img = None

            # --- Detection (SAHI or direct YOLO) ---
            if _get_sliced_prediction_fn is not None:
                sahi_result = _get_sliced_prediction_fn(
                    str(fpath), sahi_model,
                    slice_height=cfg.sahi_slice_height,
                    slice_width=cfg.sahi_slice_width,
                    overlap_height_ratio=cfg.sahi_overlap_height_ratio,
                    overlap_width_ratio=cfg.sahi_overlap_width_ratio,
                    verbose=0,
                )
                frame_img = cv2.imread(str(fpath))
                if frame_img is None:
                    tqdm.write(f"WARNING: could not read {fpath}, skipping")
                    tracker.step([])
                    del sahi_result
                    continue
                shape = frame_img.shape[:2]
                dets = pick_all_detections_sahi(
                    sahi_result, id_set, shape,
                    cfg.max_box_area_ratio, cfg.max_box_width_ratio,
                    cfg.max_box_height_ratio,
                )
                del sahi_result
            else:
                result = model.predict(source=str(fpath), **predict_kwargs)[0]
                frame_img = result.orig_img.copy()
                dets = pick_all_detections_yolo(
                    result, id_set,
                    cfg.max_box_area_ratio, cfg.max_box_width_ratio,
                    cfg.max_box_height_ratio,
                )
                del result

            # --- Multi-object tracking ---
            det_bboxes = [d.bbox for d in dets]
            mstate = tracker.step(det_bboxes)

            any_active = any(ts.active for ts in mstate.tracks)
            has_dets = len(dets) > 0
            should_render = has_dets or any_active

            if should_render:
                draw_multi_overlays(
                    frame_img, dets, mstate, cfg.trajectory_points,
                    idx, len(frames),
                )

                if has_dets:
                    det_count += 1
                    cv2.imwrite(str(cfg.detections_dir / f"{name}_{fpath.name}"), frame_img)

                if cfg.render_output_videos:
                    rendered += 1
                    cv2.imwrite(str(render_dir / f"frame_{rendered:06d}.jpg"), frame_img)

            del frame_img

        if cfg.render_output_videos and rendered > 0:
            out_path = cfg.output_videos_dir / f"{name}.mp4"
            if out_path.exists() and not cfg.overwrite_outputs:
                raise FileExistsError(f"{out_path} exists; use --overwrite-outputs")
            compose_video(render_dir, out_path, cfg.fps)

    return VideoSummary(
        video_name=name,
        input_video=str(video),
        frame_dir=str(frame_dir),
        detection_frames=det_count,
        sampled_frames=len(frames),
        output_video=str(out_path) if out_path else None,
        detector_weights=str(cfg.weights),
        detector_class_ids=class_ids,
        rendered_frames=rendered if cfg.render_output_videos else None,
    )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> list[VideoSummary]:
    if not config.videos_dir.exists():
        raise FileNotFoundError(f"Videos dir missing: {config.videos_dir}")
    if not config.weights.exists():
        raise FileNotFoundError(f"Weights missing: {config.weights}")

    config.detections_dir = writable_dir(config.detections_dir)
    ensure_dir(config.detections_dir)

    if config.render_output_videos:
        config.output_videos_dir = writable_dir(config.output_videos_dir)
        ensure_dir(config.output_videos_dir)

    videos = discover_videos(config.videos_dir)
    if not videos:
        raise FileNotFoundError(f"No .mp4 files in {config.videos_dir}")

    model = YOLO(str(config.weights))
    class_ids = resolve_drone_class(model, config.drone_class_name, config.drone_class_id)

    sahi_model = None
    if config.use_sahi:
        from sahi import AutoDetectionModel

        sahi_device = config.device or "cpu"
        if sahi_device == "cuda":
            sahi_device = "cuda:0"
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(config.weights),
            confidence_threshold=config.conf,
            device=sahi_device,
        )

    summaries = []
    for v in videos:
        summaries.append(process_video(v, model, config, class_ids, sahi_model))
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    ensure_dir(config.summary_path.parent)
    config.summary_path.write_text(
        json.dumps([asdict(s) for s in summaries], indent=2), encoding="utf-8"
    )
    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--videos-dir", type=Path, default=Path("videos"))
    parser.add_argument("--frames-dir", type=Path, default=Path("frames"))
    parser.add_argument("--detections-dir", type=Path, default=Path("detections"))
    parser.add_argument("--output-videos-dir", type=Path, default=Path("output_videos"))
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-missing-frames", type=int, default=50)
    parser.add_argument("--trajectory-points", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--drone-class-name", default="drone")
    parser.add_argument("--drone-class-id", type=int, default=None)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.03)
    parser.add_argument("--max-box-width-ratio", type=float, default=0.15)
    parser.add_argument("--max-box-height-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--overwrite-outputs", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    # SAHI
    parser.add_argument("--no-sahi", dest="use_sahi", action="store_false", default=True)
    parser.add_argument("--sahi-slice-height", type=int, default=640)
    parser.add_argument("--sahi-slice-width", type=int, default=640)
    parser.add_argument("--sahi-overlap-height-ratio", type=float, default=0.2)
    parser.add_argument("--sahi-overlap-width-ratio", type=float, default=0.2)
    # SORT multi-object tracker
    parser.add_argument("--sort-max-age", type=int, default=50)
    parser.add_argument("--sort-min-hits", type=int, default=3)
    parser.add_argument("--sort-iou-threshold", type=float, default=0.3)


def config_from_args(args, render: bool) -> PipelineConfig:
    return PipelineConfig(
        videos_dir=args.videos_dir, frames_dir=args.frames_dir,
        detections_dir=args.detections_dir, output_videos_dir=args.output_videos_dir,
        summary_path=args.summary_path, weights=args.weights,
        fps=args.fps, conf=args.conf, imgsz=args.imgsz,
        max_missing_frames=args.max_missing_frames,
        trajectory_points=args.trajectory_points,
        overwrite_frames=args.overwrite_frames, overwrite_outputs=args.overwrite_outputs,
        render_output_videos=render, device=args.device, half=args.half,
        augment=args.augment, drone_class_name=args.drone_class_name,
        drone_class_id=args.drone_class_id, max_frames=args.max_frames,
        max_box_area_ratio=args.max_box_area_ratio,
        max_box_width_ratio=args.max_box_width_ratio,
        max_box_height_ratio=args.max_box_height_ratio,
        use_sahi=args.use_sahi,
        sahi_slice_height=args.sahi_slice_height,
        sahi_slice_width=args.sahi_slice_width,
        sahi_overlap_height_ratio=args.sahi_overlap_height_ratio,
        sahi_overlap_width_ratio=args.sahi_overlap_width_ratio,
        sort_max_age=args.sort_max_age,
        sort_min_hits=args.sort_min_hits,
        sort_iou_threshold=args.sort_iou_threshold,
    )


def detection_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 1: save drone detection frames.")
    add_args(parser)
    cfg = config_from_args(parser.parse_args(argv), render=False)
    for s in run_pipeline(cfg):
        print(f"{s.video_name}: {s.sampled_frames} sampled, {s.detection_frames} detections")
    return 0


def render_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 2: detection + tracking + video output.")
    add_args(parser)
    cfg = config_from_args(parser.parse_args(argv), render=True)
    for s in run_pipeline(cfg):
        print(f"{s.video_name}: {s.detection_frames} detections, "
              f"{s.rendered_frames} rendered, output={s.output_video}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return render_cli(argv)
