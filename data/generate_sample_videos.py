"""
generate_sample_videos.py - Synthetic Traffic Video Generator
Generates realistic multi-scenario traffic test videos for:
1. Sunny Traffic: Strong illumination and prominent vehicle cast shadows (evaluates shadow_remover.py)
2. Cloudy Traffic: Diffuse overcast lighting and smooth traffic flow
3. Heavy Traffic: Dense traffic, overlapping vehicles, bumper-to-bumper queues (evaluates occlusion_handler.py)
"""

import os
import math
from pathlib import Path
import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent

def draw_road(width: int, height: int, condition: str = "sunny") -> np.ndarray:
    """
    Draw a multi-lane highway background with perspective perspective lines.
    """
    if condition == "sunny":
        sky_color = (220, 200, 160)   # Bright warm sky
        road_color = (65, 65, 70)      # Hot asphalt
        grass_color = (35, 120, 45)
    elif condition == "cloudy":
        sky_color = (180, 180, 185)   # Overcast gray sky
        road_color = (75, 75, 80)      # Dull asphalt
        grass_color = (40, 95, 45)
    else:  # heavy_traffic
        sky_color = (170, 175, 180)
        road_color = (60, 60, 65)
        grass_color = (30, 90, 40)

    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Road boundaries
    road_top_w = int(width * 0.45)
    road_bot_w = int(width * 0.85)
    horizon_y = int(height * 0.20)

    # Ground / grass
    img[horizon_y:, :] = grass_color
    # Sky
    img[:horizon_y, :] = sky_color

    # Road polygon
    road_pts = np.array([
        [(width - road_top_w) // 2, horizon_y],
        [(width + road_top_w) // 2, horizon_y],
        [(width + road_bot_w) // 2, height],
        [(width - road_bot_w) // 2, height]
    ], dtype=np.int32)
    cv2.fillPoly(img, [road_pts], road_color)

    # Shoulder lines
    cv2.line(img, tuple(road_pts[0]), tuple(road_pts[3]), (220, 220, 220), 3, cv2.LINE_AA)
    cv2.line(img, tuple(road_pts[1]), tuple(road_pts[2]), (220, 220, 220), 3, cv2.LINE_AA)

    return img

def draw_vehicle(
    img: np.ndarray,
    cx: int,
    cy: int,
    w: int,
    h: int,
    vtype: str = "car",
    color: tuple = (200, 50, 50),
    cast_shadow: bool = True,
    shadow_offset: tuple = (15, 18)
):
    """
    Renders a realistic vehicle with roof, windshield, headlights, tires, and ground cast shadow.
    """
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = cx + w // 2
    y2 = cy + h // 2

    # 1. Ground Cast Shadow (if sunny)
    if cast_shadow:
        sx_off, sy_off = shadow_offset
        shadow_pts = np.array([
            [x1 + sx_off, y2],
            [x2 + sx_off + 10, y2],
            [x2 + sx_off + 25, y2 + sy_off],
            [x1 + sx_off - 5, y2 + sy_off]
        ], dtype=np.int32)

        # Blend shadow onto roadway (darkening asphalt)
        overlay = img.copy()
        cv2.fillPoly(overlay, [shadow_pts], (20, 20, 25))
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # 2. Vehicle Wheels (Black rectangles)
    wheel_w = max(4, int(w * 0.12))
    wheel_h = max(8, int(h * 0.22))
    wheel_color = (25, 25, 25)
    # Left & right tires
    cv2.rectangle(img, (x1 - 2, y1 + int(h * 0.15)), (x1 + wheel_w, y1 + int(h * 0.15) + wheel_h), wheel_color, -1)
    cv2.rectangle(img, (x2 - wheel_w, y1 + int(h * 0.15)), (x2 + 2, y1 + int(h * 0.15) + wheel_h), wheel_color, -1)
    cv2.rectangle(img, (x1 - 2, y2 - int(h * 0.35)), (x1 + wheel_w, y2 - int(h * 0.35) + wheel_h), wheel_color, -1)
    cv2.rectangle(img, (x2 - wheel_w, y2 - int(h * 0.35)), (x2 + 2, y2 - int(h * 0.35) + wheel_h), wheel_color, -1)

    # 3. Vehicle Body
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), 2)

    # 4. Windshield & Cabin Roof
    if vtype == "car":
        roof_inset_x = int(w * 0.15)
        roof_y1 = y1 + int(h * 0.25)
        roof_y2 = y2 - int(h * 0.25)
        cv2.rectangle(img, (x1 + roof_inset_x, roof_y1), (x2 - roof_inset_x, roof_y2), (230, 230, 240), -1)
        # Front windshield (glass)
        cv2.rectangle(img, (x1 + roof_inset_x, roof_y1), (x2 - roof_inset_x, roof_y1 + int(h * 0.12)), (70, 70, 90), -1)
        # Rear windshield
        cv2.rectangle(img, (x1 + roof_inset_x, roof_y2 - int(h * 0.12)), (x2 - roof_inset_x, roof_y2), (70, 70, 90), -1)
        # Headlights / Taillights
        cv2.circle(img, (x1 + 6, y2 - 4), 4, (0, 0, 220), -1)
        cv2.circle(img, (x2 - 6, y2 - 4), 4, (0, 0, 220), -1)
        cv2.circle(img, (x1 + 6, y1 + 4), 4, (200, 255, 255), -1)
        cv2.circle(img, (x2 - 6, y1 + 4), 4, (200, 255, 255), -1)

    elif vtype == "truck":
        # Cargo container + Cab
        cab_h = int(h * 0.25)
        cv2.rectangle(img, (x1 + 4, y1), (x2 - 4, y1 + cab_h), (210, 210, 210), -1)
        cv2.rectangle(img, (x1 + 8, y1 + 4), (x2 - 8, y1 + cab_h - 4), (60, 60, 80), -1)
        # Cargo ridges
        for r_y in range(y1 + cab_h + 8, y2 - 8, 12):
            cv2.line(img, (x1 + 6, r_y), (x2 - 6, r_y), (max(0, color[0] - 40), max(0, color[1] - 40), max(0, color[2] - 40)), 2)

    elif vtype == "bus":
        # Bus roof windows
        roof_inset_x = int(w * 0.12)
        cv2.rectangle(img, (x1 + roof_inset_x, y1 + 10), (x2 - roof_inset_x, y2 - 10), (220, 220, 230), -1)
        for wy in range(y1 + 18, y2 - 18, 16):
            cv2.rectangle(img, (x1 + 4, wy), (x1 + roof_inset_x - 2, wy + 10), (50, 50, 70), -1)
            cv2.rectangle(img, (x2 - roof_inset_x + 2, wy), (x2 - 4, wy + 10), (50, 50, 70), -1)

def generate_sunny_video(filename: str, num_frames: int = 180):
    """
    Scenario 1: Sunny traffic with prominent vehicle cast shadows on asphalt.
    """
    width, height = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))

    # Cars in 2 inbound lanes and 1 outbound lane
    vehicles = [
        # Inbound Lane 1
        {"start_y": 140, "speed": 3.8, "lane_x": width * 0.38, "vtype": "car", "color": (40, 60, 200), "dir": 1},
        {"start_y": -40, "speed": 4.2, "lane_x": width * 0.38, "vtype": "car", "color": (190, 180, 50), "dir": 1},
        # Inbound Lane 2
        {"start_y": 120, "speed": 3.2, "lane_x": width * 0.52, "vtype": "truck", "color": (150, 100, 60), "dir": 1},
        {"start_y": -120, "speed": 3.6, "lane_x": width * 0.52, "vtype": "car", "color": (200, 200, 210), "dir": 1},
        # Outbound Lane
        {"start_y": height + 60, "speed": -3.5, "lane_x": width * 0.66, "vtype": "car", "color": (30, 140, 50), "dir": -1},
        {"start_y": height + 240, "speed": -4.0, "lane_x": width * 0.66, "vtype": "bus", "color": (40, 120, 220), "dir": -1},
    ]

    for frame_idx in range(num_frames):
        base_road = draw_road(width, height, condition="sunny")

        # Dashed lane divider
        dash_offset = (frame_idx * 4) % 40
        for dy in range(int(height * 0.20) + dash_offset, height, 40):
            cv2.line(base_road, (int(width * 0.45), dy), (int(width * 0.45), dy + 18), (255, 255, 255), 2)
            cv2.line(base_road, (int(width * 0.59), dy), (int(width * 0.59), dy + 18), (255, 255, 255), 2)

        # Draw Sun flare effect in top right
        cv2.circle(base_road, (width - 80, 50), 35, (180, 240, 255), -1)

        # Update & draw vehicles
        for v in vehicles:
            curr_y = int(v["start_y"] + frame_idx * v["speed"])
            # Perspective scale
            rel_depth = max(0.1, (curr_y - height * 0.15) / (height * 0.85))
            scale = 0.5 + 0.7 * rel_depth
            w = int(48 * scale)
            h = int(82 * scale) if v["vtype"] == "car" else (int(115 * scale) if v["vtype"] == "truck" else int(130 * scale))

            # Perspective horizontal position
            center_drift = (curr_y / height - 0.5) * (v["lane_x"] - width / 2) * 0.5
            x = int(v["lane_x"] + center_drift)

            if -h < curr_y < height + h:
                draw_vehicle(
                    base_road, x, curr_y, w, h,
                    vtype=v["vtype"],
                    color=v["color"],
                    cast_shadow=True,
                    shadow_offset=(int(18 * scale), int(22 * scale))
                )

        out.write(base_road)

    out.release()
    print(f"[SampleData] Generated: {filename}")

def generate_cloudy_video(filename: str, num_frames: int = 180):
    """
    Scenario 2: Cloudy overcast traffic with soft lighting and smooth flow.
    """
    width, height = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))

    vehicles = [
        {"start_y": 130, "speed": 3.6, "lane_x": width * 0.38, "vtype": "car", "color": (180, 40, 40), "dir": 1},
        {"start_y": -50, "speed": 4.0, "lane_x": width * 0.38, "vtype": "bus", "color": (50, 120, 200), "dir": 1},
        {"start_y": 80, "speed": 3.4, "lane_x": width * 0.52, "vtype": "truck", "color": (80, 80, 90), "dir": 1},
        {"start_y": -110, "speed": 3.7, "lane_x": width * 0.52, "vtype": "car", "color": (210, 210, 210), "dir": 1},
        {"start_y": height + 50, "speed": -3.8, "lane_x": width * 0.66, "vtype": "car", "color": (40, 160, 80), "dir": -1},
    ]

    for frame_idx in range(num_frames):
        base_road = draw_road(width, height, condition="cloudy")

        dash_offset = (frame_idx * 4) % 40
        for dy in range(int(height * 0.20) + dash_offset, height, 40):
            cv2.line(base_road, (int(width * 0.45), dy), (int(width * 0.45), dy + 18), (220, 220, 220), 2)
            cv2.line(base_road, (int(width * 0.59), dy), (int(width * 0.59), dy + 18), (220, 220, 220), 2)

        for v in vehicles:
            curr_y = int(v["start_y"] + frame_idx * v["speed"])
            rel_depth = max(0.1, (curr_y - height * 0.15) / (height * 0.85))
            scale = 0.5 + 0.7 * rel_depth
            w = int(48 * scale)
            h = int(82 * scale) if v["vtype"] == "car" else (int(115 * scale) if v["vtype"] == "truck" else int(130 * scale))
            center_drift = (curr_y / height - 0.5) * (v["lane_x"] - width / 2) * 0.5
            x = int(v["lane_x"] + center_drift)

            if -h < curr_y < height + h:
                # Soft shadow only
                draw_vehicle(
                    base_road, x, curr_y, w, h,
                    vtype=v["vtype"],
                    color=v["color"],
                    cast_shadow=True,
                    shadow_offset=(int(4 * scale), int(6 * scale))
                )

        out.write(base_road)

    out.release()
    print(f"[SampleData] Generated: {filename}")

def generate_heavy_traffic_video(filename: str, num_frames: int = 180):
    """
    Scenario 3: Heavy traffic congestion with overlapping vehicles and bumper-to-bumper occlusions.
    """
    width, height = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))

    # Dense clusters where vehicles travel very close and overlap in perspective
    vehicles = [
        # Cluster 1: Bumper-to-Bumper in Lane 1
        {"start_y": 140, "speed": 1.8, "lane_x": width * 0.38, "vtype": "car", "color": (190, 50, 50), "dir": 1},
        {"start_y": 80,  "speed": 1.8, "lane_x": width * 0.38, "vtype": "car", "color": (40, 100, 190), "dir": 1},
        {"start_y": 20,  "speed": 1.8, "lane_x": width * 0.38, "vtype": "truck", "color": (120, 120, 130), "dir": 1},

        # Cluster 2: Side-by-Side partial overlap in Lane 1 & 2
        {"start_y": 170, "speed": 1.7, "lane_x": width * 0.47, "vtype": "car", "color": (50, 160, 70), "dir": 1},
        {"start_y": 190, "speed": 1.7, "lane_x": width * 0.54, "vtype": "bus", "color": (210, 150, 40), "dir": 1},

        # Cluster 3: Upstream queue
        {"start_y": -40, "speed": 1.9, "lane_x": width * 0.48, "vtype": "car", "color": (200, 200, 200), "dir": 1},
        {"start_y": -95, "speed": 1.9, "lane_x": width * 0.48, "vtype": "car", "color": (160, 40, 160), "dir": 1},

        # Outbound queue
        {"start_y": height - 40, "speed": -1.5, "lane_x": width * 0.66, "vtype": "truck", "color": (100, 80, 70), "dir": -1},
        {"start_y": height + 20, "speed": -1.5, "lane_x": width * 0.66, "vtype": "car", "color": (30, 120, 180), "dir": -1},
    ]

    for frame_idx in range(num_frames):
        base_road = draw_road(width, height, condition="heavy_traffic")

        # Dashed dividers
        for dy in range(int(height * 0.20), height, 40):
            cv2.line(base_road, (int(width * 0.45), dy), (int(width * 0.45), dy + 18), (200, 200, 200), 2)
            cv2.line(base_road, (int(width * 0.59), dy), (int(width * 0.59), dy + 18), (200, 200, 200), 2)

        # Sort vehicles from back to front (top to bottom) for proper occlusion layering
        sorted_vehicles = sorted(vehicles, key=lambda v: v["start_y"] + frame_idx * v["speed"])

        for v in sorted_vehicles:
            curr_y = int(v["start_y"] + frame_idx * v["speed"])
            rel_depth = max(0.1, (curr_y - height * 0.15) / (height * 0.85))
            scale = 0.5 + 0.7 * rel_depth
            w = int(48 * scale)
            h = int(82 * scale) if v["vtype"] == "car" else (int(115 * scale) if v["vtype"] == "truck" else int(130 * scale))
            center_drift = (curr_y / height - 0.5) * (v["lane_x"] - width / 2) * 0.5
            x = int(v["lane_x"] + center_drift)

            if -h < curr_y < height + h:
                draw_vehicle(
                    base_road, x, curr_y, w, h,
                    vtype=v["vtype"],
                    color=v["color"],
                    cast_shadow=True,
                    shadow_offset=(int(8 * scale), int(10 * scale))
                )

        out.write(base_road)

    out.release()
    print(f"[SampleData] Generated: {filename}")

def generate_all_datasets():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sunny_file = str(DATA_DIR / "sunny_traffic.mp4")
    cloudy_file = str(DATA_DIR / "cloudy_traffic.mp4")
    heavy_file = str(DATA_DIR / "heavy_traffic.mp4")

    print("[SampleData] Generating test traffic datasets...")
    generate_sunny_video(sunny_file)
    generate_cloudy_video(cloudy_file)
    generate_heavy_traffic_video(heavy_file)
    print("[SampleData] All 3 dataset videos generated successfully!")

if __name__ == "__main__":
    generate_all_datasets()
