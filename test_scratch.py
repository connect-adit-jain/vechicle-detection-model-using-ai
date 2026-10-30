"""
test_scratch.py
Vehicle Scratch, Dent & Surface Anomaly Testing Tool.

Usage:
  python test_scratch.py my_car.jpg
  python test_scratch.py fastag_crop.jpg

Uses Tier-2 YOLO27 Visual Fingerprint multi-scale bilateral LoG filter to isolate
surface paint scratches, dents, stickers, and road dust gradients.
Saves annotated visualization to output_scratch.jpg.
"""
import sys
import os
import cv2
import numpy as np
from shared.visual_fingerprint import VisualFingerprintEngine


def test_scratch_detection(image_path="my_car.jpg", output_path="output_scratch.jpg"):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: '{image_path}'")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not decode image: '{image_path}'")
        return

    engine = VisualFingerprintEngine()
    anomalies = engine.extract_surface_anomalies(img)

    # Generate visual scratch heatmap overlay
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)
    diff = cv2.absdiff(gray, smooth)
    _, scratch_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    annotated = img.copy()
    # Highlight detected scratch/dent edge pixels in bright red
    annotated[scratch_mask > 0] = [0, 0, 255]

    print("=" * 65)
    print("VEHICLE SCRATCH & SURFACE ANOMALY DETECTOR")
    print(f"Image: {image_path} ({img.shape[1]}x{img.shape[0]})")
    print("=" * 65)
    print("\n--- MEASURED ANOMALY METRICS ---")
    print(f"  • Scratch / Dent Energy   : {anomalies['scratch_dent_energy']:.3f}")
    print(f"  • Scratch / Dent Status   : {'DETECTED (Surface variations present)' if anomalies['scratch_dent_energy'] > 2.0 else 'CLEAN (Smooth factory paint)'}")
    print(f"  • Distinct Sticker Count  : {anomalies['sticker_hotspots']} hotspots")
    print(f"  • Dust Accumulation Grad  : {anomalies['dust_gradient']:.3f}")
    print(f"  • Anomaly Vector Norm     : {np.linalg.norm(anomalies['anomaly_vector']):.3f}")
    print(f"\nVisual heatmap saved to: {output_path}")
    print("=" * 65)

    cv2.imwrite(output_path, annotated)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "my_car.jpg"
    test_scratch_detection(target)

