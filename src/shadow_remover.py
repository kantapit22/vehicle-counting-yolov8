"""
shadow_remover.py - Vehicle Cast Shadow Removal
Uses Canny Edge Detection, Color Space Analysis (HSV/Lab),
and Morphological Operations to separate cast shadows from vehicle bodies,
refining bounding boxes for accurate tracking and counting.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

class ShadowRemover:
    """
    Detects and trims cast shadows from vehicle detections using
    Canny Edge detection and Morphological operations.
    """
    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        morph_kernel_size: int = 5,
        min_edge_density: float = 0.015,
        shadow_v_ratio_thresh: float = 0.65,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        self.min_edge_density = min_edge_density
        self.shadow_v_ratio_thresh = shadow_v_ratio_thresh

    def remove_shadow_from_roi(
        self,
        roi: np.ndarray
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        """
        Analyze a vehicle ROI, detect cast shadows at boundaries,
        and return the refined relative bounding box (dx1, dy1, dx2, dy2)
        and the binary mask (255 = vehicle body, 0 = shadow/background).
        """
        h, w = roi.shape[:2]
        if h < 20 or w < 20:
            return (0, 0, w, h), np.ones((h, w), dtype=np.uint8) * 255

        # 1. Color space conversion
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # 2. Canny Edge Detection
        # Vehicles exhibit strong structural edges; cast shadows on asphalt have diffuse or no internal edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # 3. Morphological Operations
        # Connect edge components to bridge vehicle chassis and body panels
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
        dilated_edges = cv2.dilate(closed_edges, self.morph_kernel, iterations=2)

        # 4. Shadow candidate mask based on HSV luminance
        # Shadows have low V relative to mean, but retain non-zero S
        v_float = v_ch.astype(np.float32)
        mean_v = np.mean(v_float)
        shadow_candidate = (v_float < (mean_v * self.shadow_v_ratio_thresh)).astype(np.uint8) * 255

        # Regions that are shadow candidates AND have no significant edges are cast shadows
        cast_shadow = cv2.bitwise_and(shadow_candidate, cv2.bitwise_not(dilated_edges))
        cast_shadow = cv2.morphologyEx(cast_shadow, cv2.MORPH_OPEN, self.morph_kernel)

        # Vehicle body mask = ROI minus cast shadows
        vehicle_mask = cv2.bitwise_not(cast_shadow)

        # Clean vehicle body mask with morphological opening then closing
        vehicle_mask = cv2.morphologyEx(vehicle_mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)

        # 5. Vertical and Horizontal profile analysis to trim bounding box
        # Cast shadows typically project downwards (onto road below car) or to one side
        row_density = np.mean(vehicle_mask == 255, axis=1)
        col_density = np.mean(vehicle_mask == 255, axis=0)

        # Threshold to identify vehicle presence
        thresh = 0.25

        valid_rows = np.where(row_density > thresh)[0]
        valid_cols = np.where(col_density > thresh)[0]

        if len(valid_rows) >= 10 and len(valid_cols) >= 10:
            dy1 = int(valid_rows[0])
            dy2 = int(valid_rows[-1])
            dx1 = int(valid_cols[0])
            dx2 = int(valid_cols[-1])

            # Ensure we don't trim excessively (keep at least 60% of original dimension)
            if (dy2 - dy1) >= 0.6 * h and (dx2 - dx1) >= 0.6 * w:
                return (dx1, dy1, dx2, dy2), vehicle_mask

        return (0, 0, w, h), vehicle_mask

    def refine_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process a list of detections on the frame, trimming cast shadows.
        """
        refined_detections = []
        frame_h, frame_w = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0 or (x2 - x1) < 20 or (y2 - y1) < 20:
                refined_detections.append(det)
                continue

            (dx1, dy1, dx2, dy2), mask = self.remove_shadow_from_roi(roi)

            new_x1 = max(0, x1 + dx1)
            new_y1 = max(0, y1 + dy1)
            new_x2 = min(frame_w, x1 + dx2)
            new_y2 = min(frame_h, y1 + dy2)

            new_det = det.copy()
            new_det["original_bbox"] = [x1, y1, x2, y2]
            new_det["bbox"] = [new_x1, new_y1, new_x2, new_y2]
            new_det["shadow_trimmed"] = (new_x1 != x1 or new_y1 != y1 or new_x2 != x2 or new_y2 != y2)

            if debug:
                new_det["shadow_mask"] = mask

            refined_detections.append(new_det)

        return refined_detections
