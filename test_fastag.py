"""
test_fastag.py
Robust FASTag 3x3 Windshield Grid Placement Detection Tool.

Usage:
  python test_fastag.py fastag.webp
  python test_fastag.py my_car.jpg

Pipeline:
1. Detects vehicle bounding box using YOLO26x (or accepts pre-cropped windshield).
2. Localizes the active Windshield region.
3. Overlays the standard 3x3 spatial grid on the windshield.
4. Detects the rectangular FASTag RFID tag (aspect ratio ~1.5 - 3.8).
5. Maps the tag centroid to the exact 3x3 grid cell [row, col].
6. Outputs annotated visualization to output_fastag.jpg.
"""
import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

CELL_NAMES = [
    ["[0,0] Top-Left", "[0,1] Top-Center", "[0,2] Top-Right"],
    ["[1,0] Mid-Left", "[1,1] Center",     "[1,2] Mid-Right"],
    ["[2,0] Bot-Left", "[2,1] Bot-Center", "[2,2] Bot-Right"]
]


def detect_fastag(image_path="fastag.webp", output_path="output_fastag.jpg"):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: '{image_path}'")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not decode image: '{image_path}'")
        return None

    ih, iw = img.shape[:2]
    annotated = img.copy()

    # Step 1: Detect vehicle to localize windshield accurately
    car_box = None
    if os.path.exists("yolo26x.pt"):
        try:
            m = YOLO("yolo26x.pt")
            res = m(img, conf=0.25, verbose=False)[0]
            best_car_area = 0
            for b in res.boxes:
                lbl = m.names[int(b.cls[0])]
                if lbl in ["car", "truck", "bus"]:
                    bx1, by1, bx2, by2 = [int(x) for x in b.xyxy[0]]
                    area = (bx2 - bx1) * (by2 - by1)
                    if area > best_car_area:
                        best_car_area = area
                        car_box = [bx1, by1, bx2, by2]
        except Exception:
            pass

    # Step 2: Determine Windshield Bounds
    if car_box is not None and (car_box[2] - car_box[0]) > 300 and (car_box[3] - car_box[1]) > 300:
        cx1, cy1, cx2, cy2 = car_box
        cw = cx2 - cx1
        ch = cy2 - cy1
        # Windshield typically occupies top 8%-50% vertically, inner 15%-85% horizontally
        wx1 = max(0, cx1 + int(cw * 0.12))
        wx2 = min(iw, cx1 + int(cw * 0.88))
        wy1 = max(0, cy1 + int(ch * 0.06))
        wy2 = min(ih, cy1 + int(ch * 0.52))
        is_isolated_windshield = False
    else:
        # Image is already a close-up of a windshield or cropped vehicle
        wx1, wy1, wx2, wy2 = 0, 0, iw, ih
        is_isolated_windshield = True

    ww = max(1, wx2 - wx1)
    wh = max(1, wy2 - wy1)
    windshield_crop = img[wy1:wy2, wx1:wx2]

    # Step 3: Multi-cue FASTag detection on the windshield
    gray = cv2.cvtColor(windshield_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Gradient & contrast map
    hsv = cv2.cvtColor(windshield_crop, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    val_diff = cv2.absdiff(val, cv2.GaussianBlur(val, (15, 15), 0))
    _, thresh_val = cv2.threshold(val_diff, 20, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(blurred, 30, 110)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    combined_mask = cv2.bitwise_or(edges_closed, thresh_val)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    ws_area = ww * wh

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        aspect = float(bw) / max(bh, 1)

        # FASTag aspect ratio: typically 1.3 to 3.8
        # Size inside windshield: between 0.3% and 15% of windshield area
        if (ws_area * 0.003) < area < (ws_area * 0.12) and 1.25 <= aspect <= 3.8:
            # Prefer upper half and center/sides of windshield (standard tag positions)
            y_norm = float(y) / wh
            pos_weight = 1.2 if y_norm < 0.60 else 0.8
            # Ratio score (standard FASTag aspect is ~2.0 - 2.8)
            aspect_score = 1.0 - abs(aspect - 2.2) * 0.25
            score = area * pos_weight * aspect_score
            candidates.append((x, y, bw, bh, aspect, score))

    best_tag = None
    if candidates:
        candidates.sort(key=lambda c: c[5], reverse=True)
        best_tag = candidates[0]

    # Step 4: Draw 3x3 Spatial Grid on the Windshield
    cell_w = ww / 3.0
    cell_h = wh / 3.0
    grid_color = (255, 200, 50) # Light Cyan / Blue

    # Windshield border
    cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), (0, 180, 255), 2)
    cv2.putText(annotated, "WINDSHIELD ZONE", (wx1 + 8, max(25, wy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.5, iw / 2400), (0, 180, 255), 2, cv2.LINE_AA)

    # 3x3 internal lines
    for i in range(1, 3):
        gx = int(round(wx1 + i * cell_w))
        gy = int(round(wy1 + i * cell_h))
        cv2.line(annotated, (gx, wy1), (gx, wy2), grid_color, 1, cv2.LINE_AA)
        cv2.line(annotated, (wx1, gy), (wx2, gy), grid_color, 1, cv2.LINE_AA)

    # Label grid cells
    for r in range(3):
        for c in range(3):
            lx = int(wx1 + c * cell_w + 10)
            ly = int(wy1 + r * cell_h + 24)
            cv2.putText(annotated, CELL_NAMES[r][c], (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.38, iw / 3000), (220, 220, 220), 1, cv2.LINE_AA)

    print("=" * 68)
    print("ROBUST FASTAG 3x3 WINDSHIELD GRID DETECTION")
    print(f"Image: {image_path} ({iw}x{ih})")
    print(f"Windshield Bounds: [{wx1}, {wy1}, {wx2}, {wy2}] ({ww}x{wh} px)")
    print("=" * 68)

    if best_tag is not None:
        tx_rel, ty_rel, tbw, tbh, taspect, _ = best_tag
        # Map to full image coordinates
        abs_x1 = wx1 + tx_rel
        abs_y1 = wy1 + ty_rel
        abs_x2 = abs_x1 + tbw
        abs_y2 = abs_y1 + tbh
        tcx_rel = tx_rel + tbw / 2.0
        tcy_rel = ty_rel + tbh / 2.0

        grid_col = min(int(tcx_rel / cell_w), 2)
        grid_row = min(int(tcy_rel / cell_h), 2)
        grid_name = CELL_NAMES[grid_row][grid_col]

        # Draw FASTag bounding box and centroid
        tag_color = (0, 255, 0) # Lime green
        cv2.rectangle(annotated, (abs_x1, abs_y1), (abs_x2, abs_y2), tag_color, 3)
        cv2.circle(annotated, (int(wx1 + tcx_rel), int(wy1 + tcy_rel)), 6, (0, 0, 255), -1)

        # Label tag
        tag_text = f"FASTag: {grid_name}"
        (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, max(0.55, iw / 2000), 2)
        cv2.rectangle(annotated, (abs_x1, max(0, abs_y1 - th - 8)), (abs_x1 + tw + 10, abs_y1), tag_color, -1)
        cv2.putText(annotated, tag_text, (abs_x1 + 5, max(18, abs_y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.55, iw / 2000), (0, 0, 0), 2, cv2.LINE_AA)

        print("\n--- DETECTION STATUS: FASTAG DETECTED ---")
        print(f"  • Windshield Coordinates : [{wx1}, {wy1}, {wx2}, {wy2}]")
        print(f"  • FASTag Bounding Box    : [{abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}]")
        print(f"  • Tag Dimensions         : {tbw}px x {tbh}px (Aspect: {taspect:.2f})")
        print(f"  • 3x3 Grid Placement     : Row {grid_row}, Column {grid_col} -> {grid_name}")
        print(f"  • Disambiguation Role    : Separates near-identical white/silver vehicles")
    else:
        print("\n--- DETECTION STATUS: NO FASTAG DETECTED ON WINDSHIELD ---")
        print("  • Windshield was scanned, but no qualifying FASTag sticker was found.")

    cv2.imwrite(output_path, annotated)
    print(f"\nVisualization saved to: {output_path}")
    print("=" * 68)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "fastag.webp"
    detect_fastag(target)
