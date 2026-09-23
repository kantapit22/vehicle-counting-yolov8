"""
main.py - Vehicle Counting and Tracking System Pipeline
Integrates YOLOv8 detection, Canny Edge & Morphological Shadow Removal,
Solidity Criterion & MinError Occlusion Splitting, Kalman Filter Tracking,
and Double Zone Counting into an end-to-end computer vision pipeline.
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import VehicleDetector
from src.shadow_remover import ShadowRemover
from src.occlusion_handler import OcclusionHandler
from src.tracker import VehicleTracker
from src.counter import DoubleZoneCounter
from models.download_model import ensure_model

CLASS_COLORS = {
    "car": (255, 128, 0),         # Azure / Orange
    "motorcycle": (0, 220, 255),  # Yellow
    "bus": (255, 50, 50),         # Blue
    "truck": (50, 200, 50),       # Green
}

def draw_hud(
    frame: np.ndarray,
    counter: DoubleZoneCounter,
    fps: float,
    active_tracks_count: int,
    shadow_remover_active: bool,
    occlusion_handler_active: bool,
    scenario: str
) -> np.ndarray:
    """
    Renders a modern, professional HUD overlay displaying real-time traffic statistics.
    """
    h, w = frame.shape[:2]

    # Panel dimensions
    panel_w = 340
    panel_h = 240
    panel_x = w - panel_w - 20
    panel_y = 20

    # Semi-transparent dark background card
    sub_img = frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
    dark_rect = np.full(sub_img.shape, 25, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.25, dark_rect, 0.75, 0)
    frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = res

    # Border glow
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 80, 80), 1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 36), (50, 50, 50), -1)

    # Title
    cv2.putText(frame, "TRAFFIC ANALYTICS HUB", (panel_x + 12, panel_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    # Metrics
    y_off = panel_y + 60
    cv2.putText(frame, f"TOTAL COUNT : {counter.counts['total']}", (panel_x + 15, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    y_off += 26
    cv2.putText(frame, f"INBOUND (A->B) : {counter.counts['inbound']}", (panel_x + 15, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 100), 1, cv2.LINE_AA)

    y_off += 24
    cv2.putText(frame, f"OUTBOUND (B->A) : {counter.counts['outbound']}", (panel_x + 15, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 150, 150), 1, cv2.LINE_AA)

    y_off += 26
    # Class breakdown
    c = counter.counts["by_class"]
    cls_str = f"Car: {c.get('car', 0)} | Moto: {c.get('motorcycle', 0)} | Bus: {c.get('bus', 0)} | Truck: {c.get('truck', 0)}"
    cv2.putText(frame, cls_str, (panel_x + 15, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    y_off += 30
    cv2.line(frame, (panel_x + 10, y_off - 10), (panel_x + panel_w - 10, y_off - 10), (60, 60, 60), 1)

    # Diagnostics
    fps_color = (0, 255, 0) if fps >= 20 else ((0, 255, 255) if fps >= 10 else (0, 0, 255))
    cv2.putText(frame, f"FPS: {fps:.1f} | Active Tracks: {active_tracks_count}", (panel_x + 15, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, fps_color, 1, cv2.LINE_AA)

    y_off += 22
    shd_status = "ON" if shadow_remover_active else "OFF"
    occ_status = "ON" if occlusion_handler_active else "OFF"
    status_text = f"Shadow: [{shd_status}] | Occlusion: [{occ_status}]"
    cv2.putText(frame, status_text, (panel_x + 15, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)

    # Top-left Scenario banner
    cv2.rectangle(frame, (20, 20), (240, 56), (30, 30, 30), -1)
    cv2.rectangle(frame, (20, 20), (240, 56), (100, 100, 100), 1)
    cv2.putText(frame, f"MODE: {scenario.upper()}", (30, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

def draw_tracks(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
    recently_counted: Dict[int, float]
) -> np.ndarray:
    """
    Renders vehicle bounding boxes, class labels, track IDs, and movement trails.
    """
    current_time = time.time()

    for trk in tracks:
        tid = trk["track_id"]
        x1, y1, x2, y2 = trk["bbox"]
        cname = trk["class_name"].lower()
        conf = trk.get("confidence", 0.0)

        # Highlight if just counted
        is_freshly_counted = (tid in recently_counted) and ((current_time - recently_counted[tid]) < 1.0)
        box_color = (0, 255, 0) if is_freshly_counted else CLASS_COLORS.get(cname, (0, 255, 255))
        thickness = 3 if is_freshly_counted else 2

        # Draw Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Draw trajectory trail
        traj = trk.get("trajectory", [])
        if len(traj) > 1:
            pts = np.array(traj, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, box_color, 2, cv2.LINE_AA)

        # Label tag
        label = f"ID:{tid} {cname.capitalize()} {conf:.2f}"
        if is_freshly_counted:
            label += " [COUNTED]"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y = max(th + 4, y1)
        cv2.rectangle(frame, (x1, tag_y - th - 4), (x1 + tw + 4, tag_y + baseline), box_color, -1)
        cv2.putText(frame, label, (x1 + 2, tag_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Center dot
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    return frame

def run_pipeline(
    source: str,
    weights: str = "models/yolov8n.pt",
    conf: float = 0.35,
    iou: float = 0.45,
    device: Optional[str] = None,
    scenario: str = "sunny",
    enable_shadow_remover: Optional[bool] = None,
    enable_occlusion_handler: Optional[bool] = None,
    show: bool = False,
    save_output: Optional[str] = None,
    save_csv: Optional[str] = None,
    max_frames: Optional[int] = None,
):
    """
    Main Execution Function for the Vehicle Counting & Tracking Pipeline.
    """
    # Auto-configure based on scenario if not explicitly specified
    if enable_shadow_remover is None:
        enable_shadow_remover = (scenario.lower() == "sunny")

    if enable_occlusion_handler is None:
        enable_occlusion_handler = (scenario.lower() == "heavy_traffic")

    print("=" * 65)
    print("   VEHICLE COUNTING & TRACKING SYSTEM (YOLOv8 + CV PIPELINE)   ")
    print("=" * 65)
    print(f"Source                  : {source}")
    print(f"Weights                 : {weights}")
    print(f"Scenario                : {scenario}")
    print(f"Shadow Remover Enabled  : {enable_shadow_remover}")
    print(f"Occlusion Split Enabled : {enable_occlusion_handler}")
    print(f"Confidence Threshold    : {conf}")
    print(f"Save Output Video       : {save_output}")
    print(f"Save CSV Statistics     : {save_csv}")
    print("=" * 65)

    # 1. Initialize Components
    ensure_model(Path(weights).name, Path(weights).parent)
    detector = VehicleDetector(
        model_path=weights,
        conf_thresh=conf,
        iou_thresh=iou,
        device=device
    )

    shadow_remover = ShadowRemover() if enable_shadow_remover else None
    occlusion_handler = OcclusionHandler() if enable_occlusion_handler else None
    tracker = VehicleTracker(max_age=30, min_hits=2, iou_threshold=0.3)

    # 2. Open Video Capture
    # Check if source is integer (webcam) or file
    is_webcam = source.isdigit()
    cap_src = int(source) if is_webcam else source
    cap = cv2.VideoCapture(cap_src)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    print(f"[Video Source] Resolution: {frame_w}x{frame_h} @ {fps_in:.1f} FPS (Total frames: {total_frames})")

    # 3. Define Double Counting Zones based on frame dimensions
    # Zone A (Upper Road Section) and Zone B (Lower Road Section)
    zone_a_pts = [
        (int(frame_w * 0.10), int(frame_h * 0.40)),
        (int(frame_w * 0.90), int(frame_h * 0.40)),
        (int(frame_w * 0.90), int(frame_h * 0.48)),
        (int(frame_w * 0.10), int(frame_h * 0.48)),
    ]

    zone_b_pts = [
        (int(frame_w * 0.10), int(frame_h * 0.60)),
        (int(frame_w * 0.90), int(frame_h * 0.60)),
        (int(frame_w * 0.90), int(frame_h * 0.68)),
        (int(frame_w * 0.10), int(frame_h * 0.68)),
    ]

    counter = DoubleZoneCounter(
        zone_a=zone_a_pts,
        zone_b=zone_b_pts,
        direction_a_to_b_label="Inbound",
        direction_b_to_a_label="Outbound"
    )

    # 4. Video Writer Setup
    writer = None
    if save_output:
        out_path = Path(save_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (frame_w, frame_h))
        print(f"[VideoWriter] Recording to {save_output}...")

    # 5. Processing Loop
    frame_idx = 0
    t_start = time.time()
    rolling_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            loop_start = time.time()

            # A. YOLOv8 Detection
            raw_detections = detector.detect(frame)

            # B. Shadow Removal (Canny Edge + Morphology)
            if shadow_remover:
                detections = shadow_remover.refine_detections(frame, raw_detections)
            else:
                detections = raw_detections

            # C. Occlusion Handling (Solidity Criterion & MinError Splitting)
            if occlusion_handler:
                detections = occlusion_handler.handle_occlusions(frame, detections)

            # D. Kalman Filter Multi-Object Tracking
            active_tracks = tracker.update(detections)

            # E. Double Zone Directional Counting
            new_counts = counter.update(active_tracks)
            if new_counts:
                for event in new_counts:
                    print(f"[{event['timestamp']}] Counted: {event['class_name'].upper()} | Dir: {event['direction']} | Total: {event['total_count']}")

            # F. Visualization
            annotated = frame.copy()
            counter.draw_zones(annotated)
            draw_tracks(annotated, active_tracks, counter.recently_counted_tracks)

            # Calculate FPS
            loop_time = max(0.001, time.time() - loop_start)
            current_fps = 1.0 / loop_time
            rolling_fps = 0.9 * rolling_fps + 0.1 * current_fps if rolling_fps > 0 else current_fps

            draw_hud(
                annotated,
                counter=counter,
                fps=rolling_fps,
                active_tracks_count=len(active_tracks),
                shadow_remover_active=enable_shadow_remover,
                occlusion_handler_active=enable_occlusion_handler,
                scenario=scenario
            )

            # G. Output Writing & Display
            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("Vehicle Counting & Tracking System - YOLOv8", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[Pipeline] Exiting on user request ('q')...")
                    break
                elif key == ord('p'):
                    # Pause playback
                    cv2.waitKey(-1)

            if frame_idx % 30 == 0:
                sys.stdout.write(f"\rProcessing frame {frame_idx}/{total_frames} | FPS: {rolling_fps:.1f} | Vehicles: {counter.counts['total']}")
                sys.stdout.flush()

    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    total_time = max(0.001, time.time() - t_start)
    avg_fps = frame_idx / total_time

    print(f"\n[Pipeline Completed] Processed {frame_idx} frames in {total_time:.2f}s ({avg_fps:.1f} FPS)")
    print(f"Final Counts: Total={counter.counts['total']}, Inbound={counter.counts['inbound']}, Outbound={counter.counts['outbound']}")
    print(f"Class Breakdown: {counter.counts['by_class']}")

    # 6. Export CSV if requested
    if save_csv:
        csv_path = Path(save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if counter.event_log:
            df = pd.DataFrame(counter.event_log)
        else:
            df = pd.DataFrame(columns=["track_id", "class_name", "direction", "timestamp", "total_count"])
        df.to_csv(csv_path, index=False)
        print(f"[Export] Saved count log CSV to {csv_path}")

    return counter.counts

def main():
    parser = argparse.ArgumentParser(description="Vehicle Counting and Tracking Pipeline with YOLOv8")
    parser.add_argument("--source", type=str, default="data/sunny_traffic.mp4", help="Video path or webcam index (0)")
    parser.add_argument("--weights", type=str, default="models/yolov8n.pt", help="YOLOv8 weights file")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', '0', etc.)")
    parser.add_argument("--scenario", type=str, default="sunny", choices=["sunny", "cloudy", "heavy_traffic"], help="Preset traffic scenario")
    parser.add_argument("--enable-shadow-remover", action="store_true", default=None, help="Explicitly enable shadow removal")
    parser.add_argument("--enable-occlusion-handler", action="store_true", default=None, help="Explicitly enable occlusion splitting")
    parser.add_argument("--show", action="store_true", help="Display OpenCV live window")
    parser.add_argument("--save-output", type=str, default=None, help="Output annotated video path (.mp4)")
    parser.add_argument("--save-csv", type=str, default=None, help="Output CSV path for traffic counts")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing N frames")

    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        scenario=args.scenario,
        enable_shadow_remover=args.enable_shadow_remover,
        enable_occlusion_handler=args.enable_occlusion_handler,
        show=args.show,
        save_output=args.save_output,
        save_csv=args.save_csv,
        max_frames=args.max_frames,
    )

if __name__ == "__main__":
    main()
