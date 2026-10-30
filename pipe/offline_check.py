import os
import sys
import socket
from ultralytics import YOLO

def block_network():
    # Monkey-patch socket.socket to forbid any outbound network calls
    def guard(*args, **kwargs):
        raise RuntimeError("Network call blocked during OFFLINE test!")
    socket.socket = guard

def main():
    print("=" * 60)
    print("STEP 8: OFFLINE VERIFICATION CHECK")
    print("=" * 60)
    
    coco_weights = "yolo11s.pt"
    uvh_weights = os.path.join("models_download", "weights", "YOLOv11-S", "UVH-26-MV-YOLOv11-S.pt")
    sample_img = "bus.jpg"
    
    # 1. Check local file existence and sizes
    print("Checking local weight files:")
    for path, desc in [(coco_weights, "COCO YOLO11-S"), (uvh_weights, "UVH-26 Indian Traffic YOLO11-S"), (sample_img, "Sample Image")]:
        if not os.path.exists(path):
            print(f"[FAIL] Missing {desc} at {path}")
            sys.exit(1)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f" [OK] Found {desc}: {path} ({size_mb:.2f} MB)")
        
    # 2. Block network to guarantee offline execution
    print("\nEnforcing network block (offline guard)...")
    block_network()
    print("Network calls are now strictly blocked.")
    
    # 3. Load COCO yolo11s.pt completely offline
    print("\n--- 1/2: Loading COCO yolo11s.pt offline ---")
    model_coco = YOLO(coco_weights)
    print(f"Loaded successfully. Classes count: {len(model_coco.names)}")
    res_coco = model_coco(sample_img, device="cpu", verbose=False)
    detected_coco = [model_coco.names[int(c)] for c in res_coco[0].boxes.cls]
    print(f"COCO inference detected: {detected_coco}")
    
    # 4. Load UVH-26 YOLO11-S weights completely offline
    print("\n--- 2/2: Loading UVH-26 YOLO11-S offline ---")
    model_uvh = YOLO(uvh_weights)
    print(f"Loaded successfully. Classes count: {len(model_uvh.names)}")
    res_uvh = model_uvh(sample_img, device="cpu", verbose=False)
    detected_uvh = [model_uvh.names[int(c)] for c in res_uvh[0].boxes.cls]
    print(f"UVH-26 inference detected: {detected_uvh}")
    
    print("\n" + "=" * 60)
    print("OFFLINE CHECK PASSED: Both models loaded and ran with ZERO network calls!")
    print("=" * 60)

if __name__ == "__main__":
    main()

