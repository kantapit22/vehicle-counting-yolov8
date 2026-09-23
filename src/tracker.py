"""
tracker.py - Kalman Filter Multi-Object Tracking (SORT-based)
Maintains persistent track IDs across video frames using an 8-state
Kalman filter and Hungarian algorithm data association.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_iou(bb_test: List[float], bb_gt: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_test + area_gt - intersection
    if union <= 0:
        return 0.0
    return intersection / union


class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed as bounding box.
    State vector: [x_center, y_center, width, height, vx, vy, vw, vh]^T
    """
    count = 0

    def __init__(self, bbox: List[int], class_id: int, class_name: str, confidence: float):
        # State vector: [cx, cy, w, h, vx, vy, vw, vh]
        self.dim_x = 8
        self.dim_z = 4

        # Initialize state estimate x
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape((8, 1))

        # State transition matrix F (constant velocity model)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariance matrix P
        self.P = np.eye(8, dtype=np.float32) * 10.0
        for i in range(4, 8):
            self.P[i, i] *= 10.0  # high uncertainty for initial velocity

        # Measurement noise covariance R
        self.R = np.eye(4, dtype=np.float32) * 1.0

        # Process noise covariance Q
        self.Q = np.eye(8, dtype=np.float32) * 0.1
        for i in range(4, 8):
            self.Q[i, i] *= 0.01

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence

        # Centroid trajectory
        self.trajectory: List[Tuple[int, int]] = [(int(cx), int(cy))]

    def update(self, bbox: List[int], confidence: float, class_id: int, class_name: str):
        """
        Updates the state vector with observed bounding box.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        z = np.array([cx, cy, w, h], dtype=np.float32).reshape((4, 1))

        # Kalman update equations:
        # y = z - H * x
        # S = H * P * H^T + R
        # K = P * H^T * inv(S)
        # x = x + K * y
        # P = (I - K * H) * P
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        curr_cx = int(self.x[0, 0])
        curr_cy = int(self.x[1, 0])
        self.trajectory.append((curr_cx, curr_cy))
        if len(self.trajectory) > 60:
            self.trajectory.pop(0)

    def predict(self) -> List[int]:
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # x = F * x
        # P = F * P * F^T + Q
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        curr_bbox = self.get_state()
        self.history.append(curr_bbox)
        return curr_bbox

    def get_state(self) -> List[int]:
        """
        Returns the current bounding box estimate [x1, y1, x2, y2].
        """
        cx = self.x[0, 0]
        cy = self.x[1, 0]
        w = max(1.0, self.x[2, 0])
        h = max(1.0, self.x[3, 0])

        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return [x1, y1, x2, y2]

    def get_velocity(self) -> Tuple[float, float]:
        """
        Returns estimated velocity vector (vx, vy).
        """
        return float(self.x[4, 0]), float(self.x[5, 0])


class VehicleTracker:
    """
    Multi-Object Vehicle Tracker using Kalman Filter and Hungarian Association.
    """
    def __init__(self, max_age: int = 25, min_hits: int = 2, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates the tracker with detections in the current frame.

        Args:
            detections: List of detection dicts with 'bbox', 'confidence', 'class_id', 'class_name'.

        Returns:
            List of active tracks with track metadata.
        """
        self.frame_count += 1

        # 1. Get predicted locations from existing trackers
        predicted_boxes = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(i)
            else:
                predicted_boxes.append(pos)

        for i in reversed(to_del):
            self.trackers.pop(i)

        # 2. Associate detections with trackers
        det_boxes = [d["bbox"] for d in detections]
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            det_boxes, predicted_boxes, self.iou_threshold
        )

        # 3. Update matched trackers
        for m in matched:
            d_idx, t_idx = m[0], m[1]
            det = detections[d_idx]
            self.trackers[t_idx].update(
                bbox=det["bbox"],
                confidence=det["confidence"],
                class_id=det["class_id"],
                class_name=det["class_name"]
            )

        # 4. Create new trackers for unmatched detections
        for i in unmatched_dets:
            det = detections[i]
            trk = KalmanBoxTracker(
                bbox=det["bbox"],
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=det["confidence"]
            )
            self.trackers.append(trk)

        # 5. Extract output active tracks and prune dead ones
        active_tracks = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # Track is confirmed if hits >= min_hits or initial frames
            if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                vx, vy = trk.get_velocity()
                active_tracks.append({
                    "track_id": trk.id,
                    "bbox": d,
                    "class_id": trk.class_id,
                    "class_name": trk.class_name,
                    "confidence": trk.confidence,
                    "trajectory": list(trk.trajectory),
                    "velocity": (vx, vy),
                    "hits": trk.hits,
                    "age": trk.age
                })
            i -= 1
            # Remove trackers that have been missing for too long
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return active_tracks

    def _associate_detections_to_trackers(
        self,
        detections: List[List[int]],
        trackers: List[List[int]],
        iou_threshold: float
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Assigns detections to tracked object bounding boxes using Hungarian Algorithm.
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = calculate_iou(det, trk)

        # Convert IoU to cost matrix (1 - IoU)
        cost_matrix = 1.0 - iou_matrix

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_indices = []
        unmatched_detections = []
        unmatched_trackers = []

        for d in range(len(detections)):
            if d not in row_indices:
                unmatched_detections.append(d)

        for t in range(len(trackers)):
            if t not in col_indices:
                unmatched_trackers.append(t)

        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] < iou_threshold:
                unmatched_detections.append(r)
                unmatched_trackers.append(c)
            else:
                matched_indices.append([r, c])

        return np.array(matched_indices, dtype=int), unmatched_detections, unmatched_trackers
