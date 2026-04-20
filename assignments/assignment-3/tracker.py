"""Kalman-filter-based drone tracker using filterpy.

Includes both a legacy single-object tracker (DroneTracker) and a SORT-style
multi-object tracker (MultiObjectTracker) with Hungarian assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

Point = tuple[int, int]
BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class TrackerState:
    estimated_center: Point | None
    active: bool
    missing_frames: int
    trajectory_segments: list[list[Point]]
    predicted: bool = False


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def make_kalman_filter(center: tuple[float, float], dt: float) -> KalmanFilter:
    """4-state constant-velocity Kalman filter: [x, y, vx, vy]."""
    kf = KalmanFilter(dim_x=4, dim_z=2)

    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=float)

    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    kf.R = np.eye(2) * 4.0
    kf.P = np.diag([100.0, 100.0, 25.0, 25.0])
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=15.0, block_size=2, order_by_dim=False)
    kf.x = np.array([[center[0]], [center[1]], [0], [0]], dtype=float)

    return kf


@dataclass(slots=True)
class DroneTracker:
    fps: float
    max_missing_frames: int = 50
    kf: KalmanFilter | None = None
    missing_frames: int = 0
    trajectory_segments: list[list[Point]] = field(default_factory=lambda: [[]])

    def _start_segment(self) -> None:
        if self.trajectory_segments and self.trajectory_segments[-1]:
            self.trajectory_segments.append([])

    def _record(self, x: float, y: float) -> None:
        if not self.trajectory_segments:
            self.trajectory_segments.append([])
        self.trajectory_segments[-1].append((round(x), round(y)))

    def _center(self) -> Point | None:
        if self.kf is None:
            return None
        return (round(float(self.kf.x[0, 0])), round(float(self.kf.x[1, 0])))

    def step(self, detection: BBox | None) -> TrackerState:
        measurement = bbox_center(detection) if detection else None

        if measurement is not None:
            if self.kf is None or self.missing_frames > self.max_missing_frames:
                self._start_segment()
                self.kf = make_kalman_filter(measurement, dt=1.0 / self.fps)
            else:
                self.kf.predict()
                self.kf.update(np.array([[measurement[0]], [measurement[1]]], dtype=float))
            self.missing_frames = 0
        elif self.kf is not None:
            self.kf.predict()
            self.missing_frames += 1

        center = self._center()
        active = self.kf is not None and self.missing_frames <= self.max_missing_frames

        if center and active:
            self._record(center[0], center[1])

        return TrackerState(
            estimated_center=center,
            active=active,
            missing_frames=self.missing_frames,
            trajectory_segments=self.trajectory_segments,
            predicted=measurement is None and active,
        )


# ---------------------------------------------------------------------------
# SORT-style multi-object tracker
# ---------------------------------------------------------------------------

def iou(bb1: BBox, bb2: BBox) -> float:
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, bb1[2] - bb1[0]) * max(0.0, bb1[3] - bb1[1])
    area2 = max(0.0, bb2[2] - bb2[0]) * max(0.0, bb2[3] - bb2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def bbox_to_z(bbox: BBox) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to measurement vector [cx, cy, area, aspect_ratio]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / h if h > 0 else 1.0
    return np.array([[cx], [cy], [s], [r]], dtype=float)


def x_to_bbox(x: np.ndarray) -> BBox:
    """Convert state vector [cx, cy, s, r, ...] back to (x1, y1, x2, y2)."""
    cx, cy = float(x[0, 0]), float(x[1, 0])
    s = max(float(x[2, 0]), 1.0)
    r = max(float(x[3, 0]), 1e-6)
    w = float(np.sqrt(max(s * r, 0.0)))
    h = s / w if w > 1e-6 else 1.0
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def make_sort_filter(bbox: BBox, dt: float) -> KalmanFilter:
    """7-state constant-velocity Kalman filter for SORT: [cx, cy, s, r, vx, vy, vs]."""
    kf = KalmanFilter(dim_x=7, dim_z=4)

    kf.F = np.eye(7)
    kf.F[0, 4] = dt
    kf.F[1, 5] = dt
    kf.F[2, 6] = dt

    kf.H = np.zeros((4, 7))
    kf.H[0, 0] = 1
    kf.H[1, 1] = 1
    kf.H[2, 2] = 1
    kf.H[3, 3] = 1

    kf.R = np.diag([1.0, 1.0, 10.0, 10.0])
    kf.P[4:, 4:] *= 1000.0
    kf.P *= 10.0
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01

    kf.x[:4] = bbox_to_z(bbox)
    return kf


@dataclass(slots=True)
class Track:
    id: int
    kf: KalmanFilter
    bbox: BBox
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    trajectory: list[Point] = field(default_factory=list)


@dataclass(slots=True)
class TrackState:
    track_id: int
    bbox: BBox
    estimated_center: Point
    active: bool
    predicted: bool
    missing_frames: int
    trajectory: list[Point]


@dataclass(slots=True)
class MultiTrackerState:
    tracks: list[TrackState]


class MultiObjectTracker:
    """SORT-style multi-object tracker with Hungarian assignment."""

    def __init__(self, fps: float, max_age: int = 50,
                 min_hits: int = 3, iou_threshold: float = 0.3):
        self.fps = fps
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._tracks: list[Track] = []

    def step(self, detections: list[BBox]) -> MultiTrackerState:
        """Process one frame of detections, return state of all confirmed tracks."""
        dt = 1.0 / self.fps

        # Predict all existing tracks forward
        for trk in self._tracks:
            trk.kf.predict()
            trk.age += 1
            trk.time_since_update += 1

        # Associate detections to tracks
        pred_bboxes = [x_to_bbox(t.kf.x) for t in self._tracks]
        matched, unmatched_dets, _unmatched_trks = self._associate(
            detections, pred_bboxes
        )

        # Update matched tracks
        for t_idx, d_idx in matched:
            trk = self._tracks[t_idx]
            trk.kf.update(bbox_to_z(detections[d_idx]))
            trk.bbox = detections[d_idx]
            trk.hits += 1
            trk.time_since_update = 0

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            trk = Track(
                id=self._next_id,
                kf=make_sort_filter(detections[d_idx], dt),
                bbox=detections[d_idx],
            )
            self._tracks.append(trk)
            self._next_id += 1

        # Record trajectory points and build output
        surviving: list[Track] = []
        states: list[TrackState] = []

        for trk in self._tracks:
            if trk.time_since_update > self.max_age:
                continue  # delete track
            surviving.append(trk)

            cur_bbox = x_to_bbox(trk.kf.x)
            cx = int(round(float(trk.kf.x[0, 0])))
            cy = int(round(float(trk.kf.x[1, 0])))

            if trk.time_since_update == 0 or trk.hits >= self.min_hits:
                trk.trajectory.append((cx, cy))

            confirmed = trk.hits >= self.min_hits or trk.age < self.min_hits
            if confirmed:
                states.append(TrackState(
                    track_id=trk.id,
                    bbox=cur_bbox,
                    estimated_center=(cx, cy),
                    active=trk.hits >= self.min_hits,
                    predicted=trk.time_since_update > 0,
                    missing_frames=trk.time_since_update,
                    trajectory=list(trk.trajectory),
                ))

        self._tracks = surviving
        return MultiTrackerState(tracks=states)

    def _associate(
        self, detections: list[BBox], pred_bboxes: list[BBox]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        n_trk = len(pred_bboxes)
        n_det = len(detections)
        if n_trk == 0:
            return [], list(range(n_det)), []
        if n_det == 0:
            return [], [], list(range(n_trk))

        cost = np.zeros((n_trk, n_det))
        for t in range(n_trk):
            for d in range(n_det):
                cost[t, d] = 1.0 - iou(pred_bboxes[t], detections[d])

        row_ind, col_ind = linear_sum_assignment(cost)

        matched = []
        unmatched_dets = set(range(n_det))
        unmatched_trks = set(range(n_trk))

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > 1.0 - self.iou_threshold:
                continue
            matched.append((r, c))
            unmatched_dets.discard(c)
            unmatched_trks.discard(r)

        return matched, sorted(unmatched_dets), sorted(unmatched_trks)
