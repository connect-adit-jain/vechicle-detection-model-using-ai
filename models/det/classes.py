"""
Detector class mapping file.
Maps raw model class indices to the 5 pipeline target classes.
This is the SINGLE source of truth for class mapping, shown to the user before use.
"""

# === COCO 80-class mapping (for plain yolo26s.pt / yolo26n.pt) ===
# Only these COCO classes are relevant. We do NOT invent auto_rickshaw for COCO.
COCO_CLASS_MAP = {
    2: "car",           # COCO: car
    3: "two_wheeler",   # COCO: motorcycle
    5: "bus",           # COCO: bus
    7: "truck",         # COCO: truck
    # TODO: auto_rickshaw has no COCO equivalent.
    # Use a fine-tuned model (UVH-26 or custom) for auto_rickshaw detection.
}

# === UVH-26 14-class mapping (for UVH-26-MV-YOLOv11-S.pt and any YOLO26 fine-tune) ===
UVH26_CLASS_MAP = {
    0: "car",           # Hatchback
    1: "car",           # Sedan
    2: "car",           # SUV
    3: "car",           # MUV
    4: "bus",           # Bus
    5: "truck",         # Truck
    6: "auto_rickshaw", # Three-wheeler
    7: "two_wheeler",   # Two-wheeler
    8: "truck",         # LCV
    9: "bus",           # Mini-bus
    10: "bus",          # Tempo-traveller
    11: "two_wheeler",  # Bicycle
    12: "car",          # Van
    13: None,           # Others (skip)
}

# === Our 5 target classes ===
TARGET_CLASSES = ["car", "two_wheeler", "auto_rickshaw", "bus", "truck"]

