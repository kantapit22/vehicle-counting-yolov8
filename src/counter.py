"""
counter.py - Double Zone Vehicle Counting Logic
Implements dual-zone directional counting (Inbound/Outbound) with state-machine
tracking to prevent false positives and duplicate counts.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import time
import numpy as np
import cv2

class DoubleZoneCounter:
    """
    Directional vehicle counter using two sequential spatial zones (Zone A & Zone B).
    """
    def __init__(
        self,
        zone_a: List[Tuple[int, int]],
        zone_b: List[Tuple[int, int]],
        direction_a_to_b_label: str = "Inbound (A->B)",
        direction_b_to_a_label: str = "Outbound (B->A)",
        state_timeout_seconds: float = 8.0,
    ):
        """
        Args:
            zone_a: List of 4 (x, y) vertices defining Zone A polygon.
            zone_b: List of 4 (x, y) vertices defining Zone B polygon.
            direction_a_to_b_label: Label for traffic moving from A to B.
            direction_b_to_a_label: Label for traffic moving from B to A.
        """
        self.zone_a = np.array(zone_a, dtype=np.int32)
        self.zone_b = np.array(zone_b, dtype=np.int32)
        self.dir_a_to_b_label = direction_a_to_b_label
        self.dir_b_to_a_label = direction_b_to_a_label
        self.state_timeout = state_timeout_seconds

        # Counters
        self.counts = {
            "total": 0,
            "inbound": 0,
            "outbound": 0,
            "by_class": {
                "car": 0,
                "motorcycle": 0,
                "bus": 0,
                "truck": 0,
            }
        }

        # Track state machine: track_id -> {'first_zone': 'A'|'B', 'entered_time': float, 'class_name': str}
        self.track_states: Dict[int, Dict[str, Any]] = {}
        # Set of track_ids that have already completed a count
        self.counted_ids: Set[int] = set()

        # Detailed event log for CSV export
        self.event_log: List[Dict[str, Any]] = []

        # Real-time zone visual triggers
        self.zone_a_active = False
        self.zone_b_active = False
        self.recently_counted_tracks: Dict[int, float] = {}

    def is_point_in_zone(self, point: Tuple[int, int], zone_pts: np.ndarray) -> bool:
        """
        Check if a 2D point (x, y) lies inside a convex/concave polygon zone.
        """
        dist = cv2.pointPolygonTest(zone_pts, (float(point[0]), float(point[1])), False)
        return dist >= 0

    def update(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process active tracks through the double zone state machine.

        Returns list of newly counted events in this frame:
        [
            {
                'track_id': 12,
                'class_name': 'car',
                'direction': 'Inbound (A->B)',
                'timestamp': 1695470000.0
            }, ...
        ]
        """
        current_time = time.time()
        new_events = []

        self.zone_a_active = False
        self.zone_b_active = False

        # Cleanup expired states
        expired_ids = [
            tid for tid, data in self.track_states.items()
            if (current_time - data["entered_time"]) > self.state_timeout
        ]
        for tid in expired_ids:
            del self.track_states[tid]

        # Cleanup flash timers
        self.recently_counted_tracks = {
            tid: t for tid, t in self.recently_counted_tracks.items()
            if (current_time - t) < 1.5
        }

        for track in tracks:
            track_id = track["track_id"]
            class_name = track["class_name"].lower()
            x1, y1, x2, y2 = track["bbox"]

            # Vehicle centroid & bottom-center (often best for ground plane)
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            bottom_center = (int((x1 + x2) / 2), int(y2 - 5))

            in_zone_a = self.is_point_in_zone(centroid, self.zone_a) or self.is_point_in_zone(bottom_center, self.zone_a)
            in_zone_b = self.is_point_in_zone(centroid, self.zone_b) or self.is_point_in_zone(bottom_center, self.zone_b)

            if in_zone_a:
                self.zone_a_active = True
            if in_zone_b:
                self.zone_b_active = True

            # If already counted, skip state transitions
            if track_id in self.counted_ids:
                continue

            # State transition evaluation
            if track_id not in self.track_states:
                # Vehicle enters a zone for the first time
                if in_zone_a and not in_zone_b:
                    self.track_states[track_id] = {
                        "first_zone": "A",
                        "entered_time": current_time,
                        "class_name": class_name,
                    }
                elif in_zone_b and not in_zone_a:
                    self.track_states[track_id] = {
                        "first_zone": "B",
                        "entered_time": current_time,
                        "class_name": class_name,
                    }
            else:
                first_zone = self.track_states[track_id]["first_zone"]

                # Case A -> B: Completed Inbound
                if first_zone == "A" and in_zone_b:
                    self._register_count(track_id, class_name, self.dir_a_to_b_label, current_time, new_events, "inbound")
                # Case B -> A: Completed Outbound
                elif first_zone == "B" and in_zone_a:
                    self._register_count(track_id, class_name, self.dir_b_to_a_label, current_time, new_events, "outbound")

        return new_events

    def _register_count(
        self,
        track_id: int,
        class_name: str,
        direction: str,
        current_time: float,
        event_accumulator: List[Dict[str, Any]],
        dir_key: str
    ):
        """
        Record a validated count event.
        """
        self.counted_ids.add(track_id)
        if track_id in self.track_states:
            del self.track_states[track_id]

        self.counts["total"] += 1
        self.counts[dir_key] += 1
        if class_name not in self.counts["by_class"]:
            self.counts["by_class"][class_name] = 0
        self.counts["by_class"][class_name] += 1

        self.recently_counted_tracks[track_id] = current_time

        event = {
            "track_id": track_id,
            "class_name": class_name,
            "direction": direction,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time)),
            "total_count": self.counts["total"],
        }
        self.event_log.append(event)
        event_accumulator.append(event)

    def draw_zones(self, frame: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        """
        Render semi-transparent filled polygon zones and boundary outlines on the frame.
        """
        overlay = frame.copy()

        # Zone A colors
        color_a = (0, 200, 255) if self.zone_a_active else (255, 180, 0)
        border_a = (0, 255, 255) if self.zone_a_active else (255, 220, 50)

        # Zone B colors
        color_b = (50, 255, 100) if self.zone_b_active else (200, 0, 255)
        border_b = (100, 255, 150) if self.zone_b_active else (230, 80, 255)

        # Fill transparent polygons
        cv2.fillPoly(overlay, [self.zone_a], color_a)
        cv2.fillPoly(overlay, [self.zone_b], color_b)

        # Blend
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # Draw crisp borders
        cv2.polylines(frame, [self.zone_a], True, border_a, 2, cv2.LINE_AA)
        cv2.polylines(frame, [self.zone_b], True, border_b, 2, cv2.LINE_AA)

        # Zone Labels
        mom_a = cv2.moments(self.zone_a)
        if mom_a["m00"] != 0:
            ax = int(mom_a["m10"] / mom_a["m00"])
            ay = int(mom_a["m01"] / mom_a["m00"])
            cv2.putText(frame, "ZONE A (Entry/Exit)", (ax - 70, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        mom_b = cv2.moments(self.zone_b)
        if mom_b["m00"] != 0:
            bx = int(mom_b["m10"] / mom_b["m00"])
            by = int(mom_b["m01"] / mom_b["m00"])
            cv2.putText(frame, "ZONE B (Entry/Exit)", (bx - 70, by), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
