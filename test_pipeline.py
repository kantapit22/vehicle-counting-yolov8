"""
test_pipeline.py - Verification & Unit Tests for the Vehicle Counting System
Tests all individual modules:
1. Detector (YOLOv8 loading & inference)
2. Shadow Remover (Canny Edge & Morphological Operations)
3. Occlusion Handler (Solidity Criterion & MinError Splitting)
4. Kalman Tracker (State estimation & Hungarian matching)
5. Double Zone Counter (Directional logic & duplicate prevention)
6. Full Pipeline Integration
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import VehicleDetector
from src.shadow_remover import ShadowRemover
from src.occlusion_handler import OcclusionHandler
from src.tracker import VehicleTracker, calculate_iou
from src.counter import DoubleZoneCounter
from src.main import run_pipeline

class TestVehicleCountingPipeline(unittest.TestCase):

    def test_01_detector_initialization(self):
        """Verify YOLOv8 detector loads weights and can infer on a dummy frame."""
        detector = VehicleDetector(model_path="models/yolov8n.pt", conf_thresh=0.25)
        self.assertIsNotNone(detector.model)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(dummy_frame)
        self.assertIsInstance(detections, list)
        print("\n[PASS] Test 1: Detector initialized and ran inference.")

    def test_02_shadow_remover(self):
        """Verify shadow remover processes an ROI and returns valid trimmed bbox and mask."""
        remover = ShadowRemover()
        # Create a mock ROI with a car-like shape and a dark bottom shadow
        roi = np.full((120, 80, 3), 180, dtype=np.uint8)
        # Vehicle body: add edges and textures
        cv2.rectangle(roi, (10, 10), (70, 80), (50, 50, 200), -1)
        cv2.line(roi, (15, 30), (65, 30), (0, 0, 0), 2)
        # Bottom cast shadow: dark intensity, no internal edges
        roi[85:120, 5:75] = (30, 30, 30)

        (dx1, dy1, dx2, dy2), mask = remover.remove_shadow_from_roi(roi)
        self.assertGreaterEqual(dx1, 0)
        self.assertLessEqual(dx2, 80)
        self.assertGreaterEqual(dy1, 0)
        self.assertLessEqual(dy2, 120)
        self.assertEqual(mask.shape, (120, 80))
        print("[PASS] Test 2: Shadow remover successfully processed mock vehicle & shadow.")

    def test_03_occlusion_handler_solidity_and_split(self):
        """Verify solidity calculation and splitting on a dumbbell / occluded shape."""
        handler = OcclusionHandler()

        # Single vehicle (convex rectangle) -> Solidity should be close to 1.0
        single_car = np.array([
            [[20, 20]], [[80, 20]], [[80, 100]], [[20, 100]]
        ], dtype=np.int32)
        solidity, _, _ = handler.calculate_solidity(single_car)
        self.assertGreater(solidity, 0.90)

        # Dumbbell / Merged 2 vehicles (concave notches) -> Solidity should be lower
        occluded_cars = np.array([
            [[20, 20]], [[80, 20]], [[80, 60]], [[35, 80]], [[80, 100]], [[80, 140]],
            [[20, 140]], [[20, 100]], [[65, 80]], [[20, 60]]
        ], dtype=np.int32)
        solidity_occ, _, defects = handler.calculate_solidity(occluded_cars)
        self.assertLess(solidity_occ, 0.80)

        # Test splitting on mock detection
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw two connected white blobs (merged cars in traffic)
        # Car 1 (top)
        cv2.rectangle(frame, (50, 30), (130, 80), (220, 220, 220), -1)
        # Car 2 (bottom)
        cv2.rectangle(frame, (50, 130), (130, 180), (220, 220, 220), -1)
        # Narrow bridge between them (contact / occlusion neck)
        cv2.rectangle(frame, (80, 80), (100, 130), (220, 220, 220), -1)

        det = {
            "bbox": [45, 25, 135, 185],
            "confidence": 0.85,
            "class_id": 2,
            "class_name": "car"
        }
        splits = handler.split_occluded_vehicle(frame, det)
        self.assertEqual(len(splits), 2, "Occlusion handler should split merged detection into 2 boxes")
        print("[PASS] Test 3: Occlusion handler correctly calculated solidity and split merged vehicle.")

    def test_04_kalman_tracker(self):
        """Verify Kalman tracker state estimation and continuous trajectory tracking."""
        tracker = VehicleTracker(max_age=10, min_hits=1)

        # Simulate a car moving downward across 5 frames
        track_ids = []
        for i in range(5):
            y_pos = 100 + i * 20
            detections = [{
                "bbox": [150, y_pos, 210, y_pos + 80],
                "confidence": 0.90,
                "class_id": 2,
                "class_name": "car"
            }]
            active = tracker.update(detections)
            self.assertEqual(len(active), 1)
            track_ids.append(active[0]["track_id"])

        # Track ID must remain constant across all frames
        self.assertEqual(len(set(track_ids)), 1, "Track ID must remain persistent across consecutive frames")
        print("[PASS] Test 4: Kalman tracker maintained persistent Track ID across frames.")

    def test_05_double_zone_counter(self):
        """Verify double zone counting state machine (Inbound & Outbound) and duplicate prevention."""
        # Zone A (top: y=50 to 90), Zone B (bottom: y=150 to 190)
        zone_a = [(50, 50), (250, 50), (250, 90), (50, 90)]
        zone_b = [(50, 150), (250, 150), (250, 190), (50, 190)]

        counter = DoubleZoneCounter(zone_a=zone_a, zone_b=zone_b)

        # Vehicle 1: Moves Downwards (A -> B) => Inbound
        # Frame 1: inside Zone A
        t1 = [{"track_id": 1, "class_name": "car", "bbox": [100, 60, 150, 80]}]
        events = counter.update(t1)
        self.assertEqual(len(events), 0)

        # Frame 2: between zones
        t2 = [{"track_id": 1, "class_name": "car", "bbox": [100, 110, 150, 130]}]
        events = counter.update(t2)
        self.assertEqual(len(events), 0)

        # Frame 3: inside Zone B => Triggers Inbound Count!
        t3 = [{"track_id": 1, "class_name": "car", "bbox": [100, 160, 150, 180]}]
        events = counter.update(t3)
        self.assertEqual(len(events), 1)
        self.assertEqual(counter.counts["total"], 1)
        self.assertEqual(counter.counts["inbound"], 1)

        # Frame 4: still in Zone B => Duplicate prevention check
        events = counter.update(t3)
        self.assertEqual(len(events), 0, "Must not double count vehicle remaining in Zone B")
        self.assertEqual(counter.counts["total"], 1)

        # Vehicle 2: Moves Upwards (B -> A) => Outbound
        # Frame 5: enters Zone B
        v2_f1 = [{"track_id": 2, "class_name": "bus", "bbox": [120, 160, 170, 180]}]
        counter.update(v2_f1)
        # Frame 6: enters Zone A => Triggers Outbound Count!
        v2_f2 = [{"track_id": 2, "class_name": "bus", "bbox": [120, 60, 170, 80]}]
        events_v2 = counter.update(v2_f2)
        self.assertEqual(len(events_v2), 1)
        self.assertEqual(counter.counts["total"], 2)
        self.assertEqual(counter.counts["outbound"], 1)
        self.assertEqual(counter.counts["by_class"]["bus"], 1)
        print("[PASS] Test 5: Double zone counter successfully verified Inbound, Outbound, and duplicate prevention.")

    def test_06_full_pipeline_sunny(self):
        """Run full pipeline on sunny_traffic.mp4 and verify output video and CSV."""
        out_video = "data/test_output_sunny.mp4"
        out_csv = "data/test_output_sunny.csv"
        counts = run_pipeline(
            source="data/sunny_traffic.mp4",
            weights="models/yolov8n.pt",
            scenario="sunny",
            enable_shadow_remover=True,
            show=False,
            save_output=out_video,
            save_csv=out_csv,
            max_frames=60
        )
        self.assertTrue(Path(out_video).exists())
        self.assertTrue(Path(out_csv).exists())
        print(f"[PASS] Test 6: Full pipeline executed on sunny_traffic.mp4. Counts: {counts}")

if __name__ == "__main__":
    unittest.main()
