"""
Standalone Monocular Vehicle Distance Estimation Test Suite
===========================================================
Runs high-precision 3D distance estimation, lane offset calculation,
and renders a Top-Down Bird's-Eye-View (BEV) radar map.

Usage:
    python test_distance.py [image_path] [--height 6.0] [--pitch 18.0] [--out output_distance.jpg]
"""

import os
import sys
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

from traffic.distance_estimator import VehicleDistanceEstimator

CLASS_COLORS = {
    "Car": (0, 220, 0),
    "Bike": (255, 200, 0),
    "Auto-Rickshaw": (0, 230, 255),
    "Bus": (255, 50, 150),
    "Truck": (0, 120, 255),
}

COCO_MAP = {
    1: "Bike",
    2: "Car",
    3: "Bike",
    5: "Bus",
    7: "Truck"
}

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def compute_iomin(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    min_area = min((box1[2] - box1[0]) * (box1[3] - box1[1]),
                   (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return inter / min_area if min_area > 0 else 0.0

def find_sample_image():
    candidates = ["my_car.jpg", "bus.jpg", "testImage1.webp", "testImage2.webp", "testImage3.jpg"]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def run_distance_pipeline(
    image_path=None,
    camera_height=6.0,
    pitch_deg=18.0,
    vfov_deg=42.0,
    hfov_deg=65.0,
    conf_thresh=0.25,
    output_path="output_distance.jpg",
):
    if not image_path:
        image_path = find_sample_image()
        if not image_path:
            print("[ERROR] No input image found.")
            return

    if not os.path.exists(image_path):
        print(f"[ERROR] Image path does not exist: {image_path}")
        return

    print("=" * 80)
    print("MONOCULAR VEHICLE 3D DISTANCE & SPATIAL LOCALIZATION ENGINE")
    print(f"Target Image: {image_path}")
    print(f"Camera Configuration: Height = {camera_height:.1f}m | Downward Pitch = {pitch_deg:.1f}° | VFOV = {vfov_deg:.1f}°")
    print("=" * 80)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not decode image: {image_path}")
        return

    ih, iw = img.shape[:2]

    # 1. Detect vehicles using YOLO26
    weights_path = "yolo26x.pt" if os.path.exists("yolo26x.pt") else "yolo26s.pt"
    if not os.path.exists(weights_path):
        weights_path = "yolo11s.pt"

    print(f"[1/4] Running vehicle detector with weights: {weights_path}...")
    model = YOLO(weights_path)
    res = model(img, conf=conf_thresh, verbose=False)[0]

    raw_detections = []
    for b in res.boxes:
        cid = int(b.cls[0])
        if cid in COCO_MAP:
            label = COCO_MAP[cid]
            conf = float(b.conf[0])
            box = [float(c) for c in b.xyxy[0]]
            raw_detections.append({
                "label": label,
                "confidence": conf,
                "box": box
            })

    # Auto-Rickshaw detection using UVH weights with Yellow Taxi / Car guard
    uvh_weights = os.path.join("models_download", "weights", "YOLOv11-S", "UVH-26-MV-YOLOv11-S.pt")
    if os.path.exists(uvh_weights):
        m_uvh = YOLO(uvh_weights)
        res_uvh = m_uvh(img, conf=0.25, verbose=False)[0]
        for b in res_uvh.boxes:
            if int(b.cls[0]) == 6:  # Three-wheeler
                u_box = [float(c) for c in b.xyxy[0]]
                u_conf = float(b.conf[0])
                bw = max(1.0, u_box[2] - u_box[0])
                bh = max(1.0, u_box[3] - u_box[1])
                aspect = bh / bw

                # YELLOW TAXI GUARD:
                # 1) If it overlaps with an existing Car detection (IoU > 0.20 or containment > 0.35) -> It is a Taxi (Car)
                # 2) If it has car-like wide proportions (aspect < 1.05) -> Treat as Car, NOT Auto-Rickshaw
                overlaps_car = False
                for cd in raw_detections:
                    if cd["label"] == "Car" and (compute_iou(u_box, cd["box"]) > 0.20 or compute_iomin(u_box, cd["box"]) > 0.35):
                        overlaps_car = True
                        break

                if overlaps_car or aspect < 1.05:
                    # Treat yellow taxi strictly as Car; do not add duplicate auto-rickshaw
                    continue

                raw_detections.append({
                    "label": "Auto-Rickshaw",
                    "confidence": u_conf,
                    "box": u_box
                })

    # Class-Agnostic Non-Maximum Suppression (Eliminate duplicate/overlapping boxes)
    raw_detections.sort(key=lambda d: d["confidence"], reverse=True)
    detections = []
    for cand in raw_detections:
        suppress = False
        for kept in detections:
            if compute_iou(cand["box"], kept["box"]) > 0.30 or compute_iomin(cand["box"], kept["box"]) > 0.50:
                # If conflicting between Car and Auto-Rickshaw (e.g., yellow taxi), Car always wins
                if kept["label"] == "Car" and cand["label"] == "Auto-Rickshaw":
                    suppress = True
                    break
                elif cand["label"] == "Car" and kept["label"] == "Auto-Rickshaw":
                    kept["label"] = "Car"
                    kept["confidence"] = max(kept["confidence"], cand["confidence"])
                    kept["box"] = cand["box"]
                    suppress = True
                    break
                else:
                    suppress = True
                    break
        if not suppress:
            detections.append(cand)

    print(f"[2/4] Detected {len(detections)} candidate vehicles.")

    # 2. Initialize Distance Estimator
    estimator = VehicleDistanceEstimator(
        camera_height_m=camera_height,
        pitch_deg=pitch_deg,
        vfov_deg=vfov_deg,
        hfov_deg=hfov_deg,
    )

    # 3. Calculate 3D distances and spatial offsets
    print("[3/4] Estimating 3D Euclidean distances and lateral lane offsets...")
    dist_results = estimator.process_detections(detections, (ih, iw))

    # Sort from nearest to farthest
    dist_results.sort(key=lambda d: d["distance_m"])

    # Print comprehensive tabular metrics
    print("\n" + "=" * 92)
    print(f"{'ID':<4} {'Class':<14} {'Conf':<7} {'3D Distance':<13} {'Forward (Z)':<13} {'Lane Offset (X)':<16} {'Zone / Risk':<18}")
    print("=" * 92)

    for idx, d in enumerate(dist_results, 1):
        d["vehicle_id"] = idx
        lbl = d["label"]
        conf_str = f"{d['confidence']*100:.1f}%"
        dist_3d = f"{d['distance_m']:.2f} m"
        fwd_z = f"{d['longitudinal_z_m']:.2f} m"
        lane_str = f"{d['lateral_x_m']:+.2f} m ({d['lane_label']})"
        zone_str = f"{d['proximity_zone']} [{d['proximity_risk']}]"

        print(f"[{idx:>2}] {lbl:<14} {conf_str:<7} {dist_3d:<13} {fwd_z:<13} {lane_str:<16} {zone_str:<18}")

    print("=" * 92)

    # 4. Render HUD Visualization with BEV Top-Down Radar
    print(f"[4/4] Rendering HUD display and BEV radar to: {output_path}...")
    annotated = estimator.render_distance_visualization(img, dist_results, draw_radar=True, radar_size=260)

    # Add summary header banner to image
    header_h = 45
    cv2.rectangle(annotated, (0, 0), (iw, header_h), (12, 16, 20), -1)
    summary_txt = f"NETRAFLOW 3D SPATIAL RADAR | Vehicles: {len(dist_results)} | Cam Height: {camera_height:.1f}m | Tilt: {pitch_deg:.1f} deg"
    cv2.putText(annotated, summary_txt, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, annotated)
    print(f"\n[OK] High-Precision Distance Visualization saved successfully to: {os.path.abspath(output_path)}")
    return dist_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monocular Vehicle Distance Estimation Test Suite")
    parser.add_argument("image", nargs="?", default=None, help="Path to image file")
    parser.add_argument("--height", type=float, default=6.0, help="Camera height in meters (default: 6.0)")
    parser.add_argument("--pitch", type=float, default=18.0, help="Camera downward tilt pitch in degrees (default: 18.0)")
    parser.add_argument("--vfov", type=float, default=42.0, help="Vertical Field of View (default: 42.0)")
    parser.add_argument("--hfov", type=float, default=65.0, help="Horizontal Field of View (default: 65.0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--out", type=str, default="output_distance.jpg", help="Output file name")
    args = parser.parse_args()

    run_distance_pipeline(
        image_path=args.image,
        camera_height=args.height,
        pitch_deg=args.pitch,
        vfov_deg=args.vfov,
        hfov_deg=args.hfov,
        conf_thresh=args.conf,
        output_path=args.out
    )
