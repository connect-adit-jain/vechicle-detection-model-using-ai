"""
tests/benchmark_detectors.py
Benchmarks candidate detectors:
1. yolo26s.pt (COCO)
2. yolo26n.pt (COCO low-power)
3. models/det (COCO / fine-tune wrapper)
4. UVH-26 YOLO11-S (reference benchmark)

Measures:
- Latency and FPS on GPU
- CPU-only FPS for yolo26n.pt at imgsz 640 and 416 (evaluates >= 25 FPS real-time threshold)
- Detections per class per 1000 frames (on UVH-26 validation images)
- Produces outputs/detector_report.md
"""
import os
import sys
import time
import json
import torch
import numpy as np
import cv2
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.det.det_model import load_model as load_det, detect as run_detect, UVH26_TO_TARGET, COCO_TO_TARGET


def get_sample_images(max_images=100):
    val_dir = os.path.join(ROOT, "data", "uvh26", "UVH-26-Val", "data", "000")
    images = []
    if os.path.exists(val_dir):
        files = [f for f in os.listdir(val_dir) if f.lower().endswith((".png", ".jpg"))]
        for f in sorted(files)[:max_images]:
            p = os.path.join(val_dir, f)
            img = cv2.imread(p)
            if img is not None:
                images.append(img)
    if not images:
        # Fallback to local images
        for f in ["my_car.jpg", "bus.jpg"]:
            p = os.path.join(ROOT, f)
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    images.append(img)
    return images


def benchmark_gpu_speed(model, sample_img, imgsz=640, warmup=15, runs=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Warmup
    for _ in range(warmup):
        _ = model.predict(sample_img, imgsz=imgsz, device=device, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(runs):
        _ = model.predict(sample_img, imgsz=imgsz, device=device, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - t0
    avg_latency_ms = (total_time / runs) * 1000.0
    fps = runs / total_time
    return avg_latency_ms, fps


def benchmark_cpu_speed(model, sample_img, imgsz=640, warmup=5, runs=20):
    # Warmup
    for _ in range(warmup):
        _ = model.predict(sample_img, imgsz=imgsz, device="cpu", verbose=False)
    
    t0 = time.time()
    for _ in range(runs):
        _ = model.predict(sample_img, imgsz=imgsz, device="cpu", verbose=False)
    total_time = time.time() - t0
    avg_latency_ms = (total_time / runs) * 1000.0
    fps = runs / total_time
    return avg_latency_ms, fps


def count_detections(model, images, class_map, imgsz=640, conf=0.3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    counts = {"car": 0, "two_wheeler": 0, "auto_rickshaw": 0, "bus": 0, "truck": 0}
    total_frames = len(images)
    if total_frames == 0:
        return counts
    
    for img in images:
        res = model.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)
        for r in res:
            if r.boxes is not None:
                for cls_id in r.boxes.cls:
                    cid = int(cls_id.item())
                    target_cls = class_map.get(cid, None)
                    if target_cls in counts:
                        counts[target_cls] += 1
    
    # Scale to per 1000 frames
    scale = 1000.0 / total_frames
    scaled_counts = {k: round(v * scale, 1) for k, v in counts.items()}
    return scaled_counts


def main():
    print("=" * 70)
    print("STEP 0.2: DETECTOR BENCHMARK")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    sample_images = get_sample_images(max_images=50)
    if not sample_images:
        print("[ERROR] No sample images available for detector benchmark.")
        return
    ref_img = sample_images[0]
    print(f"Loaded {len(sample_images)} test images for evaluation.")

    # Candidate models
    models_to_test = [
        ("yolo26s.pt", "yolo26s (COCO)", COCO_TO_TARGET),
        ("yolo26n.pt", "yolo26n (COCO low-power)", COCO_TO_TARGET),
        ("models_download/weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt", "UVH-26 YOLO11-S (Reference)", UVH26_TO_TARGET),
    ]

    results = {}

    for path, display_name, class_map in models_to_test:
        if not os.path.exists(path):
            print(f"[SKIP] Model file not found: {path}")
            continue
        print(f"\n--- Testing {display_name} ---")
        model = YOLO(path)
        
        # GPU Latency & FPS
        gpu_lat, gpu_fps = benchmark_gpu_speed(model, ref_img, imgsz=640)
        print(f"  GPU Latency: {gpu_lat:.2f} ms | GPU FPS: {gpu_fps:.1f}")

        # Class counts
        print("  Evaluating class detections on dataset images...")
        counts = count_detections(model, sample_images, class_map, imgsz=640, conf=0.3)
        print(f"  Detections per 1000 frames: {counts}")

        results[display_name] = {
            "path": path,
            "gpu_latency_ms": gpu_lat,
            "gpu_fps": gpu_fps,
            "counts_per_1000": counts,
        }

    # CPU-only evaluation for yolo26n.pt
    print("\n--- CPU-only Evaluation for yolo26n.pt ---")
    if os.path.exists("yolo26n.pt"):
        yolo_n = YOLO("yolo26n.pt")
        lat_640, fps_640 = benchmark_cpu_speed(yolo_n, ref_img, imgsz=640)
        print(f"  CPU imgsz=640: Latency = {lat_640:.2f} ms | FPS = {fps_640:.2f} | Real-time (>=25 FPS): {'YES' if fps_640 >= 25 else 'NO'}")
        
        lat_416, fps_416 = benchmark_cpu_speed(yolo_n, ref_img, imgsz=416)
        print(f"  CPU imgsz=416: Latency = {lat_416:.2f} ms | FPS = {fps_416:.2f} | Real-time (>=25 FPS): {'YES' if fps_416 >= 25 else 'NO'}")
        
        results["yolo26n_cpu"] = {
            "imgsz_640": {"latency_ms": lat_640, "fps": fps_640, "realtime": fps_640 >= 25},
            "imgsz_416": {"latency_ms": lat_416, "fps": fps_416, "realtime": fps_416 >= 25},
        }

    # Save output JSON
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out_json = os.path.join(ROOT, "outputs", "detector_comparison.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw detector benchmark metrics to {out_json}")

    # Generate Markdown Report
    report_md = os.path.join(ROOT, "outputs", "detector_report.md")
    with open(report_md, "w") as f:
        f.write("# Detector Benchmark Report (Step 0.2)\n\n")
        f.write(f"- **Evaluation Hardware**: {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n")
        f.write(f"- **Confidence Threshold**: 0.30\n")
        f.write(f"- **Input Image Size**: 640x640\n\n")
        f.write("## GPU Latency and Throughput\n\n")
        f.write("| Model | Path | GPU Latency (ms) | GPU FPS |\n")
        f.write("|---|---|---|---|\n")
        for k, v in results.items():
            if k == "yolo26n_cpu":
                continue
            f.write(f"| {k} | `{v['path']}` | {v['gpu_latency_ms']:.2f} ms | {v['gpu_fps']:.1f} FPS |\n")
        
        f.write("\n## CPU-only Real-Time Assessment (yolo26n.pt)\n\n")
        if "yolo26n_cpu" in results:
            c640 = results["yolo26n_cpu"]["imgsz_640"]
            c416 = results["yolo26n_cpu"]["imgsz_416"]
            f.write(f"- **imgsz 640**: {c640['latency_ms']:.2f} ms | {c640['fps']:.2f} FPS | Reaches >= 25 FPS: **{'YES' if c640['realtime'] else 'NO'}**\n")
            f.write(f"- **imgsz 416**: {c416['latency_ms']:.2f} ms | {c416['fps']:.2f} FPS | Reaches >= 25 FPS: **{'YES' if c416['realtime'] else 'NO'}**\n")
        
        f.write("\n## Detections Per 1000 Frames (Target 5 Classes)\n\n")
        f.write("| Model | Car | Two Wheeler | Auto Rickshaw | Bus | Truck |\n")
        f.write("|---|---|---|---|---|---|\n")
        for k, v in results.items():
            if k == "yolo26n_cpu":
                continue
            cnts = v["counts_per_1000"]
            f.write(f"| {k} | {cnts['car']} | {cnts['two_wheeler']} | {cnts['auto_rickshaw']} | {cnts['bus']} | {cnts['truck']} |\n")
        
        f.write("\n*Note: Plain COCO models cannot detect auto_rickshaws without Indian traffic fine-tuning; auto_rickshaw count is 0 as prohibited from hallucinating/inventing mapping.*\n")

    print(f"Generated detector report at {report_md}")


if __name__ == "__main__":
    main()
