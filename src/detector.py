"""
detector.py - YOLOv8 Vehicle Detector
Performs vehicle detection on video frames using Ultralytics YOLOv8.
Filters for vehicle categories (car, motorcycle, bus, truck).
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from ultralytics import YOLO

# Standard COCO vehicle class mappings
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

class VehicleDetector:
    """
    YOLOv8-based vehicle detector specialized for traffic surveillance.
    """
    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        device: Optional[str] = None,
        target_classes: Optional[List[int]] = None,
    ):
        model_file = Path(model_path)
        if not model_file.exists():
            # Attempt to download if missing
            from models.download_model import ensure_model
            model_path = ensure_model(model_name=model_file.name)

        self.model_path = str(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.target_classes = target_classes or list(VEHICLE_CLASSES.keys())

        print(f"[VehicleDetector] Loading YOLOv8 model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        print(f"[VehicleDetector] Model loaded successfully. Target classes: {[VEHICLE_CLASSES.get(c, str(c)) for c in self.target_classes]}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run vehicle inference on a single frame.

        Args:
            frame: BGR numpy image array.

        Returns:
            List of detection dicts:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.89,
                    'class_id': 2,
                    'class_name': 'car'
                }, ...
            ]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.target_classes,
            device=self.device,
            verbose=False
        )

        detections = []
        if len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            x1 = max(0, int(round(xyxy[0])))
            y1 = max(0, int(round(xyxy[1])))
            x2 = min(w, int(round(xyxy[2])))
            y2 = min(h, int(round(xyxy[3])))

            # Basic sanity check
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            class_name = VEHICLE_CLASSES.get(cls_id, result.names.get(cls_id, f"vehicle_{cls_id}"))

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": class_name,
            })

        return detections

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color: tuple = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Helper method to draw bounding boxes and labels on a frame.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Label background
            (txt_w, txt_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - txt_h - 4)),
                (x1 + txt_w + 4, y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, max(txt_h, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        return annotated
