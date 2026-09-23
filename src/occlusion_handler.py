"""
occlusion_handler.py - Occlusion Handling via Solidity Criterion & MinError Splitting
Detects merged/overlapping vehicle detections in dense traffic and splits them
into individual vehicle bounding boxes using contour concavity analysis.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

class OcclusionHandler:
    """
    Handles occluded/merged vehicle detections using the Solidity Criterion
    and Minimum Error / Concavity defect splitting.
    """
    def __init__(
        self,
        solidity_thresh: float = 0.82,
        min_aspect_ratio_split: float = 1.7,
        min_defect_depth: float = 8.0,
        min_area_for_split: int = 2000,
    ):
        self.solidity_thresh = solidity_thresh
        self.min_aspect_ratio_split = min_aspect_ratio_split
        self.min_defect_depth = min_defect_depth
        self.min_area_for_split = min_area_for_split

    def calculate_solidity(self, contour: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calculate contour solidity = contour_area / convex_hull_area.
        Also returns convex hull and convexity defects.
        """
        area = cv2.contourArea(contour)
        if area <= 0:
            return 1.0, None, None

        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            return 1.0, None, None

        hull_points = cv2.convexHull(contour, returnPoints=True)
        hull_area = cv2.contourArea(hull_points)

        if hull_area <= 0:
            return 1.0, None, None

        solidity = float(area) / float(hull_area)

        # Calculate convexity defects
        try:
            defects = cv2.convexityDefects(contour, hull)
        except Exception:
            defects = None

        return solidity, hull_points, defects

    def extract_vehicle_contour(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the primary foreground contour from a vehicle ROI.
        """
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu thresholding with adaptive fallback
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If background is bright and vehicle is dark, invert if needed
        # Check border pixels
        border_mean = (np.mean(thresh[0, :]) + np.mean(thresh[-1, :]) +
                       np.mean(thresh[:, 0]) + np.mean(thresh[:, -1])) / 4.0
        if border_mean > 127:
            thresh = cv2.bitwise_not(thresh)

        # Morphological closing to solidify vehicle silhouette
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Return largest contour
        return max(contours, key=cv2.contourArea)

    def find_split_point(
        self,
        roi: np.ndarray,
        contour: np.ndarray,
        defects: Optional[np.ndarray],
        orientation: str = "horizontal"
    ) -> int:
        """
        Determine the optimal partition cut coordinate (MinError split)
        either along the deepest concavity defects or the constriction neck.
        """
        h, w = roi.shape[:2]

        # Method A: Concavity Defects
        deep_defects = []
        if defects is not None and len(defects) > 0:
            defects_2d = defects.reshape(-1, 4)
            for row in defects_2d:
                f = int(row[2])
                depth = float(row[3]) / 256.0  # OpenCV returns depth scaled by 256
                if depth >= self.min_defect_depth and f < len(contour):
                    far_pt = tuple(contour[f][0])
                    deep_defects.append((depth, far_pt))

        if len(deep_defects) >= 2:
            # Sort defects by depth descending
            deep_defects.sort(key=lambda x: x[0], reverse=True)
            pt1 = deep_defects[0][1]
            pt2 = deep_defects[1][1]

            if orientation == "horizontal":
                # Splitting vehicles that are left/right of each other
                mid_x = int((pt1[0] + pt2[0]) / 2)
                if int(0.2 * w) < mid_x < int(0.8 * w):
                    return mid_x
            else:
                # Splitting vehicles that are front/back (top/bottom)
                mid_y = int((pt1[1] + pt2[1]) / 2)
                if int(0.2 * h) < mid_y < int(0.8 * h):
                    return mid_y

        # Method B: MinError Constriction Neck Analysis
        # Scan cross-sectional profile to find the minimum width/height neck
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        proj = np.sum(edges, axis=0 if orientation == "horizontal" else 1)

        # Look in the central 30% - 70% zone for the local minimum
        if orientation == "horizontal":
            start_idx = int(0.3 * w)
            end_idx = int(0.7 * w)
            if end_idx > start_idx:
                min_idx = start_idx + int(np.argmin(proj[start_idx:end_idx]))
                return min_idx
            return w // 2
        else:
            start_idx = int(0.3 * h)
            end_idx = int(0.7 * h)
            if end_idx > start_idx:
                min_idx = start_idx + int(np.argmin(proj[start_idx:end_idx]))
                return min_idx
            return h // 2

    def split_occluded_vehicle(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check if a single detection is an occluded pair; if so, split into two detections.
        """
        x1, y1, x2, y2 = detection["bbox"]
        w = x2 - x1
        h = y2 - y1
        area = w * h

        if area < self.min_area_for_split:
            return [detection]

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return [detection]

        contour = self.extract_vehicle_contour(roi)
        if contour is None or cv2.contourArea(contour) < (0.25 * area):
            return [detection]

        solidity, hull, defects = self.calculate_solidity(contour)
        aspect_ratio_w = float(w) / float(h)
        aspect_ratio_h = float(h) / float(w)

        # Check occlusion criterion:
        # 1. Low solidity (contour has concavities)
        # 2. Or excessively elongated bounding box with moderate concavity
        is_occluded = (solidity < self.solidity_thresh) or (
            (aspect_ratio_w > self.min_aspect_ratio_split or aspect_ratio_h > self.min_aspect_ratio_split)
            and solidity < 0.85
        )

        if not is_occluded:
            return [detection]

        # Determine split direction
        if aspect_ratio_w >= aspect_ratio_h:
            # Horizontally adjacent vehicles (side-by-side)
            cut_x = self.find_split_point(roi, contour, defects, orientation="horizontal")
            split_x = x1 + cut_x

            # Ensure sub-boxes are reasonable
            if (split_x - x1) < 20 or (x2 - split_x) < 20:
                return [detection]

            box1 = [x1, y1, split_x, y2]
            box2 = [split_x, y1, x2, y2]
        else:
            # Vertically adjacent vehicles (bumper-to-bumper)
            cut_y = self.find_split_point(roi, contour, defects, orientation="vertical")
            split_y = y1 + cut_y

            if (split_y - y1) < 20 or (y2 - split_y) < 20:
                return [detection]

            box1 = [x1, y1, x2, split_y]
            box2 = [x1, split_y, x2, y2]

        det1 = detection.copy()
        det1["bbox"] = box1
        det1["confidence"] = max(0.5, detection["confidence"] * 0.95)
        det1["is_split"] = True
        det1["solidity"] = round(solidity, 3)

        det2 = detection.copy()
        det2["bbox"] = box2
        det2["confidence"] = max(0.5, detection["confidence"] * 0.95)
        det2["is_split"] = True
        det2["solidity"] = round(solidity, 3)

        return [det1, det2]

    def handle_occlusions(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process detections and split any occluded vehicle clusters.
        """
        processed = []
        for det in detections:
            splits = self.split_occluded_vehicle(frame, det)
            processed.extend(splits)
        return processed
