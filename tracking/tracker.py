"""
tracking/tracker.py
Single-Camera Multi-Object Tracking Engine.

Integrates:
- YOLO26x (High precision vehicle detection)
- UVH-26 (Indian traffic auto-rickshaw detection)
- ByteTrack (Multi-object tracking with Kalman filter association)
Assigns local track_id to each vehicle in a single camera feed.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

COCO_TO_TARGET = {2: "car", 3: "two_wheeler", 5: "bus", 7: "truck"}
UVH_WEIGHTS = os.path.join("models_download", "weights", "YOLOv11-S", "UVH-26-MV-YOLOv11-S.pt")


class CameraTracker:
    def __init__(self, camera_id="cam1", detector_weights="yolo26x.pt", conf_thresh=0.25, imgsz=640, device="cuda"):
        self.camera_id = camera_id
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.device = device
        
        # Load YOLO detector
        self.detector = YOLO(detector_weights)
        self.has_uvh = os.path.exists(UVH_WEIGHTS)
        if self.has_uvh:
            self.uvh_detector = YOLO(UVH_WEIGHTS)
        else:
            self.uvh_detector = None

    def track_frame(self, frame_bgr, frame_idx=0, timestamp=0.0):
        """
        Track vehicles in a single frame.
        Returns: list of dicts:
          {
            "track_id": int,
            "camera_id": str,
            "class": str,
            "box": [x1, y1, x2, y2],
            "conf": float,
            "frame_idx": int,
            "timestamp": float,
            "crop": np.ndarray (temporary BGR crop)
          }
        """
        h, w = frame_bgr.shape[:2]
        # ByteTrack via Ultralytics track
        results = self.detector.track(
            frame_bgr,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )[0]

        tracked_objects = []
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            cls_ids = results.boxes.cls.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for i in range(len(track_ids)):
                cid = cls_ids[i]
                target_cls = COCO_TO_TARGET.get(cid, None)
                if target_cls is None:
                    continue

                x1, y1, x2, y2 = [int(round(c)) for c in boxes[i]]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))

                crop = frame_bgr[y1:y2, x1:x2].copy()

                tracked_objects.append({
                    "track_id": int(track_ids[i]),
                    "camera_id": self.camera_id,
                    "class": target_cls,
                    "box": [x1, y1, x2, y2],
                    "conf": float(confs[i]),
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "crop": crop
                })

        # Integrate UVH-26 Indian auto-rickshaws if available
        if self.uvh_detector is not None:
            uvh_res = self.uvh_detector(frame_bgr, conf=0.30, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            if uvh_res.boxes is not None:
                for b in uvh_res.boxes:
                    if int(b.cls[0]) == 6: # Three-wheeler
                        uconf = float(b.conf[0])
                        ux1, uy1, ux2, uy2 = [int(round(c)) for c in b.xyxy[0]]
                        ubox = [ux1, uy1, ux2, uy2]
                        
                        # Check collision with existing tracked objects
                        collides = False
                        for t in tracked_objects:
                            # If overlapping a weak truck/car, replace with auto-rickshaw
                            x_inter = max(0, min(ux2, t["box"][2]) - max(ux1, t["box"][0]))
                            y_inter = max(0, min(uy2, t["box"][3]) - max(uy1, t["box"][1]))
                            inter = x_inter * y_inter
                            min_area = min((ux2 - ux1) * (uy2 - uy1), (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]))
                            if min_area > 0 and (inter / min_area) > 0.40:
                                if t["class"] in ["truck", "car"] and t["conf"] < 0.70:
                                    t["class"] = "auto_rickshaw"
                                    t["conf"] = max(t["conf"], uconf)
                                collides = True
                                break

        return tracked_objects

