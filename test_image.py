import os
import sys
import argparse
import torch
import cv2
import numpy as np
from ultralytics import YOLO

COCO_WEIGHTS = "yolo26x.pt"
UVH_WEIGHTS = os.path.join("models_download", "weights", "YOLOv11-S", "UVH-26-MV-YOLOv11-S.pt")

CLASS_COLORS = {
    "Car": (0, 220, 0),                 # Vivid Lime Green
    "Bike": (255, 200, 0),              # Bright Cyan / Blue
    "Auto-Rickshaw": (0, 230, 255),      # Golden Yellow
    "Bus": (255, 50, 150),               # Magenta / Pink
    "Truck": (0, 120, 255),              # Amber / Orange
    "Bicycle": (200, 200, 200),          # Silver / Gray
}

COCO_MAP = {
    1: "Bicycle",
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
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def compute_iomin(box1, box2):
    """Intersection over minimum area: detects sub-crop containment."""
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

def find_default_image():
    candidates = ["my_car.jpg", "my_car.jpeg", "my_car.png", "my_car.webp", "bus.jpg"]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def detect_vehicles(image_path=None, conf_thresh=0.20, frame_pct=60.0, imgsz=None, output_path="output_detection.jpg"):
    if not image_path:
        image_path = find_default_image()
        if not image_path:
            print("[ERROR] No input image specified and no default sample found.")
            return []

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: '{image_path}'")
        return []

    device = 0 if torch.cuda.is_available() else "cpu"
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not decode image: '{image_path}'")
        return []

    ih, iw = img.shape[:2]

    # Calculate exactly 60% of the image frame (bottom 60% active roadway zone)
    # y_start = (1.0 - 0.60) * ih = 0.40 * ih to ih
    frame_pct = min(100.0, max(20.0, float(frame_pct)))
    if frame_pct < 100.0 and ih > 800:
        y_start_ratio = (100.0 - frame_pct) / 100.0
        y_offset = int(ih * y_start_ratio)
        roi_img = img[y_offset:, :]
        roi_h, roi_w = roi_img.shape[:2]
        mode_str = f"ACTIVE {frame_pct:.0f}% OF FRAME (y: {y_offset} to {ih}, height: {roi_h}px)"
    else:
        y_offset = 0
        roi_img = img
        roi_h, roi_w = ih, iw
        mode_str = "FULL FRAME (100% height)"

    if imgsz is None:
        imgsz = 1280 if max(roi_h, roi_w) >= 1200 else 640

    print("=" * 75)
    print("HIGH-ACCURACY ROADWAY VEHICLE DETECTOR")
    print(f"Image: {image_path} ({iw}x{ih}) | Device: {device}")
    print(f"Analysis Region: {mode_str}")
    print(f"Inference Resolution: {imgsz}px | Sensitivity Threshold: {conf_thresh:.2f}")
    print("=" * 75)

    # 1. Primary Road Detector on the 60% active frame
    m_coco = YOLO(COCO_WEIGHTS)
    res_coco = m_coco(roi_img, conf=max(0.12, conf_thresh - 0.05), imgsz=imgsz, device=device, verbose=False)[0]

    raw_candidates = []
    for b in res_coco.boxes:
        cid = int(b.cls[0])
        if cid in COCO_MAP:
            label = COCO_MAP[cid]
            conf = float(b.conf[0])
            rx1, ry1, rx2, ry2 = [float(c) for c in b.xyxy[0]]
            full_box = [rx1, ry1 + y_offset, rx2, ry2 + y_offset]
            raw_candidates.append({
                "label": label,
                "confidence": conf,
                "box": full_box
            })

    # 2. De-bias Car vs Truck Hallucinations
    # Passenger vehicles (hatchbacks, sedans, SUVs, taxis) take precedence as Car
    resolved_vehicles = []
    for cand in raw_candidates:
        box = cand["box"]
        lbl = cand["label"]
        conf = cand["confidence"]

        matched = False
        for rv in resolved_vehicles:
            if compute_iou(box, rv["box"]) > 0.40:
                matched = True
                if (rv["label"] == "Car" and lbl == "Truck") or (rv["label"] == "Truck" and lbl == "Car"):
                    rv["label"] = "Car"
                    rv["confidence"] = max(rv["confidence"], conf)
                elif (rv["label"] == "Bike" and lbl in ["Car", "Truck"]) or (rv["label"] in ["Car", "Truck"] and lbl == "Bike"):
                    rv["label"] = "Bike"
                    rv["confidence"] = max(rv["confidence"], conf)
                else:
                    if conf > rv["confidence"]:
                        rv["confidence"] = conf
                        rv["box"] = box
                break
        if not matched:
            resolved_vehicles.append(cand)

    # 3. Two-Wheeler / Scooter Geometry Refinement
    for rv in resolved_vehicles:
        box = rv["box"]
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        aspect = bh / bw

        if rv["label"] in ["Car", "Truck"] and (aspect >= 1.05 or rv["confidence"] < 0.50):
            pad = 10
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(iw, x2 + pad)
            cy2 = min(ih, y2 + pad)
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crop_res = m_coco(crop, conf=0.15, verbose=False)[0]
                crop_labels = [COCO_MAP.get(int(cb.cls[0]), None) for cb in crop_res.boxes]
                if "Bike" in crop_labels:
                    rv["label"] = "Bike"

    # 4. Integrate Indian Auto-Rickshaws with Anti-Hallucination Guard
    # Note: Mumbai Taxis (black & yellow cars) are CARS, not auto-rickshaws!
    if os.path.exists(UVH_WEIGHTS):
        m_uvh = YOLO(UVH_WEIGHTS)
        res_uvh = m_uvh(roi_img, conf=max(0.18, conf_thresh), imgsz=imgsz, device=device, verbose=False)[0]
        for b in res_uvh.boxes:
            cid = int(b.cls[0])
            if cid == 6: # Three-wheeler / Auto-rickshaw
                uvh_conf = float(b.conf[0])
                rx1, ry1, rx2, ry2 = [float(c) for c in b.xyxy[0]]
                uvh_box = [rx1, ry1 + y_offset, rx2, ry2 + y_offset]

                # Check if it overlaps with an existing detection
                replaced = False
                conflicting = False
                for fv in resolved_vehicles:
                    iou = compute_iou(uvh_box, fv["box"])
                    iomin = compute_iomin(uvh_box, fv["box"])
                    if iou > 0.25 or iomin > 0.45:
                        # COCO often mislabels Auto-Rickshaws as Truck or Bus or weak Car
                        if fv["label"] in ["Truck", "Bus"] or (fv["label"] == "Car" and fv["confidence"] < 0.70):
                            if uvh_conf >= fv["confidence"] or fv["label"] == "Truck":
                                fv["label"] = "Auto-Rickshaw"
                                fv["confidence"] = max(fv["confidence"], uvh_conf)
                                fv["box"] = uvh_box
                                fv["source"] = "uvh"
                                replaced = True
                                break
                        conflicting = True
                        break

                if not replaced and not conflicting and uvh_conf >= conf_thresh:
                    resolved_vehicles.append({
                        "label": "Auto-Rickshaw",
                        "confidence": uvh_conf,
                        "box": uvh_box,
                        "source": "uvh"
                    })

    # 5. Class-Agnostic Non-Maximum Suppression (Eliminates simultaneous conflicting labels)
    # Sort descending by confidence
    resolved_vehicles.sort(key=lambda d: d["confidence"], reverse=True)
    kept_vehicles = []
    
    for cand in resolved_vehicles:
        cand_box = cand["box"]
        suppress = False
        for kept in kept_vehicles:
            iou = compute_iou(cand_box, kept["box"])
            iomin = compute_iomin(cand_box, kept["box"])
            # If two detections share the same physical vehicle space (IoU > 0.30 or containment > 0.50)
            if iou > 0.30 or iomin > 0.50:
                # If one is high-confidence Car and other is Bus/Auto-Rickshaw hallucination, keep Car
                if kept["label"] == "Car" and cand["label"] in ["Bus", "Auto-Rickshaw", "Truck"]:
                    suppress = True
                    break
                elif cand["label"] == "Car" and kept["label"] in ["Bus", "Auto-Rickshaw", "Truck"] and cand["confidence"] > 0.60:
                    # Car takes precedence over ambiguous bus/truck
                    kept["label"] = "Car"
                    kept["confidence"] = max(kept["confidence"], cand["confidence"])
                    kept["box"] = cand["box"]
                    suppress = True
                    break
                else:
                    suppress = True
                    break
        if not suppress:
            kept_vehicles.append(cand)

    # Filter by confidence threshold
    final_detections = [d for d in kept_vehicles if d["confidence"] >= conf_thresh]
    final_detections.sort(key=lambda d: d["confidence"], reverse=True)

    # 6. Draw Clean Visualizations
    annotated = img.copy()
    counts = {}
    for d in final_detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    print(f"\n--- DETECTION SUMMARY ({frame_pct:.0f}% OF FRAME) ---")
    print(f"Total Vehicles Detected: {len(final_detections)}")
    for lbl, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {lbl:<15}: {count} vehicles")

    print("\n--- DETAILED VEHICLE LIST ---")
    font_scale = max(0.48, max(ih, iw) / 2100)
    thickness = max(2, int(round(max(ih, iw) / 800)))

    for i, det in enumerate(final_detections, 1):
        lbl = det["label"]
        conf = det["confidence"]
        box = det["box"]
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        color = CLASS_COLORS.get(lbl, (0, 255, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        txt = f"{lbl} {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), color, -1)
        cv2.putText(annotated, txt, (x1 + 3, max(th + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

        print(f"  [{i:>2}] {lbl:<12} {conf*100:>5.1f}% | Coordinates: [{x1}, {y1}, {x2}, {y2}]")

    # Save full annotated image
    cv2.imwrite(output_path, annotated)

    # Save cropped view of the analyzed 60% active frame
    if y_offset > 0:
        zoomed_roadway = annotated[y_offset:, :]
        cv2.imwrite("output_roadway_focused.jpg", zoomed_roadway)
        print(f"\nSaved 60% roadway view to: {os.path.abspath('output_roadway_focused.jpg')}")

    print("=" * 75)
    print(f"SUCCESS! Output visualization saved to: {os.path.abspath(output_path)}")
    print("=" * 75)
    return final_detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accurate vehicle detector focusing on active 60% of frame.")
    parser.add_argument("image", nargs="?", default=None, help="Path to image file")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (default: 0.20)")
    parser.add_argument("--pct", type=float, default=60.0, help="Percentage of image frame to analyze (default: 60)")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference resolution")
    parser.add_argument("--out", type=str, default="output_detection.jpg", help="Output file")
    args = parser.parse_args()

    detect_vehicles(args.image, conf_thresh=args.conf, frame_pct=args.pct, imgsz=args.imgsz, output_path=args.out)
