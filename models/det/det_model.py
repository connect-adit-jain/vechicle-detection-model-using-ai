"""
Detector wrapper for YOLO26 (Ultralytics).
Interface:
  DET_MODEL_NAME, load_model(path, device, imgsz), detect(model, frame_bgr, conf) -> list of dicts
"""
import json
import os

DET_MODEL_NAME = "det_yolo26_coco"

# COCO class mapping to our 5 target classes
# COCO indices: 2=car, 3=motorcycle, 5=bus, 7=truck
# We do NOT invent auto_rickshaw for plain COCO.
COCO_TO_TARGET = {
    2: "car",
    3: "two_wheeler",
    5: "bus",
    7: "truck",
}

# UVH-26 14-class mapping to our 5 target classes
UVH26_TO_TARGET = {
    0: "car",       # Hatchback
    1: "car",       # Sedan
    2: "car",       # SUV
    3: "car",       # MUV
    4: "bus",       # Bus
    5: "truck",     # Truck
    6: "auto_rickshaw",  # Three-wheeler
    7: "two_wheeler",    # Two-wheeler
    8: "truck",     # LCV
    9: "bus",       # Mini-bus
    10: "bus",      # Tempo-traveller
    11: "two_wheeler",   # Bicycle
    12: "car",      # Van
    13: None,       # Others (skip)
}


def load_model(path="yolo26s.pt", device="cuda", imgsz=640):
    """Load a YOLO26 model.
    Args:
        path: path to .pt weights
        device: 'cuda' or 'cpu'
        imgsz: inference image size
    Returns:
        dict with model, device, name, class_map, imgsz
    """
    from ultralytics import YOLO
    model = YOLO(path)
    # Determine class mapping from model names
    names = model.names  # dict {int: str}
    # Check if this is a UVH-26 model (has 14 classes with Indian traffic names)
    is_uvh26 = len(names) == 14 and any("Hatchback" in str(v) or "Three-wheeler" in str(v) for v in names.values())
    if is_uvh26:
        class_map = UVH26_TO_TARGET
        det_name = "det_yolo26_uvh26"
    else:
        class_map = COCO_TO_TARGET
        det_name = DET_MODEL_NAME

    return {
        "model": model,
        "device": device,
        "name": det_name,
        "class_map": class_map,
        "imgsz": imgsz,
        "model_names": names,
    }


def detect(model_dict, frame_bgr, conf=0.3):
    """Run detection on a single frame.
    Args:
        model_dict: dict from load_model
        frame_bgr: HxWx3 uint8 BGR numpy array
        conf: confidence threshold
    Returns:
        list of {"box": [x1,y1,x2,y2], "conf": float, "class": str}
    """
    model = model_dict["model"]
    class_map = model_dict["class_map"]
    imgsz = model_dict["imgsz"]
    results = model.predict(frame_bgr, conf=conf, imgsz=imgsz, device=model_dict["device"], verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            target_class = class_map.get(cls_id, None)
            if target_class is None:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            c = float(boxes.conf[i].item())
            detections.append({
                "box": [x1, y1, x2, y2],
                "conf": c,
                "class": target_class,
            })
    return detections
