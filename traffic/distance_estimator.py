"""
Monocular Vehicle Distance & 3D Spatial Localization Engine
===========================================================
Estimates 3D spatial metrics (Longitudinal Distance, Lateral Lane Offset, 
3D Euclidean Distance, Proximity Zone, and Top-Down Bird's-Eye-View Radar)
from a single CCTV traffic camera feed using calibrated camera geometry
fused with physical vehicle dimension priors.
"""

import math
import cv2
import numpy as np

# Real-world physical dimensions (Width, Height, Length in meters) for Indian traffic
VEHICLE_DIMENSION_PRIORS = {
    "Car": {"width": 1.78, "height": 1.52, "length": 4.35},
    "Bike": {"width": 0.78, "height": 1.25, "length": 1.95},
    "Auto-Rickshaw": {"width": 1.32, "height": 1.74, "length": 2.68},
    "Bus": {"width": 2.55, "height": 3.25, "length": 11.20},
    "Truck": {"width": 2.45, "height": 3.40, "length": 8.80},
    "Bicycle": {"width": 0.60, "height": 1.05, "length": 1.70},
}

# Distance proximity zone thresholds (meters)
ZONE_IMMEDIATE = 12.0  # <12m: Red
ZONE_NEAR = 28.0       # 12-28m: Yellow/Amber
ZONE_MID = 55.0        # 28-55m: Cyan/Green
                       # >55m: Muted Blue


class VehicleDistanceEstimator:
    """
    Monocular 3D spatial distance estimator for traffic surveillance cameras.
    Combines ground-plane contact ray intersection with physical size priors.
    """

    def __init__(
        self,
        camera_height_m: float = 6.0,
        pitch_deg: float = 18.0,
        vfov_deg: float = 42.0,
        hfov_deg: float = 65.0,
        lane_width_m: float = 3.5,
    ):
        """
        Parameters
        ----------
        camera_height_m : Mounting height of camera pole above roadway (meters)
        pitch_deg       : Downward tilt angle of camera from horizon (degrees)
        vfov_deg        : Vertical Field of View (degrees)
        hfov_deg        : Horizontal Field of View (degrees)
        lane_width_m    : Standard road lane width in meters
        """
        self.camera_height = float(camera_height_m)
        self.pitch_rad = math.radians(float(pitch_deg))
        self.vfov_rad = math.radians(float(vfov_deg))
        self.hfov_rad = math.radians(float(hfov_deg))
        self.lane_width = float(lane_width_m)

    def estimate_single_vehicle(
        self,
        box: list,
        label: str,
        image_shape: tuple,
        confidence: float = 1.0,
    ) -> dict:
        """
        Computes 3D distance and lane offset for a single vehicle bounding box.

        Parameters
        ----------
        box         : [x1, y1, x2, y2] pixel coordinates
        label       : Detected class label (Car, Bike, Bus, Truck, Auto-Rickshaw)
        image_shape : (height, width) of image
        confidence  : Detection confidence score (0.0 to 1.0)

        Returns
        -------
        Dictionary containing comprehensive distance & spatial localization metrics.
        """
        ih, iw = image_shape[:2]
        x1, y1, x2, y2 = [float(c) for c in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # 1. Camera Intrinsic Focal Lengths
        fy = (ih / 2.0) / math.tan(self.vfov_rad / 2.0)
        fx = (iw / 2.0) / math.tan(self.hfov_rad / 2.0)
        cx = iw / 2.0
        cy = ih / 2.0

        # Ground contact footprint: bottom-center of vehicle bounding box
        u_contact = (x1 + x2) / 2.0
        v_contact = y2

        # 2. Geometry-Based Distance (Ground Plane Ray Intersection)
        # Vertical ray angle relative to camera optical axis
        alpha_y = math.atan((v_contact - cy) / fy)
        total_pitch = self.pitch_rad + alpha_y

        if total_pitch > 0.03:  # Ray points downwards below horizon
            z_ground = self.camera_height / math.tan(total_pitch)
        else:
            z_ground = 120.0  # Cap at far distance for near-horizon detections

        # Lateral ground offset from optical center line (meters)
        # Negative = Left of camera centerline, Positive = Right of camera centerline
        alpha_x = math.atan((u_contact - cx) / fx)
        x_ground = z_ground * alpha_x

        # 3. Object Dimension Prior Distance (Pinhole Object Model)
        prior = VEHICLE_DIMENSION_PRIORS.get(label, VEHICLE_DIMENSION_PRIORS["Car"])
        real_width = prior["width"]
        real_height = prior["height"]

        z_from_width = (real_width * fx) / bw
        z_from_height = (real_height * fy) / bh

        # Fuse height and width priors based on aspect ratio
        aspect = bh / bw
        expected_aspect = real_height / real_width
        if abs(aspect - expected_aspect) < 0.35:
            z_prior = 0.65 * z_from_width + 0.35 * z_from_height
        else:
            z_prior = z_from_width

        # 4. Bayesian Dual-Constraint Adaptive Fusion
        # Ground plane is most reliable at mid-to-far ranges; priors stabilize close ranges
        if z_ground < 15.0:
            w_geom = 0.55
            w_prior = 0.45
        elif z_ground < 45.0:
            w_geom = 0.70
            w_prior = 0.30
        else:
            w_geom = 0.85
            w_prior = 0.15

        z_fused = max(2.5, min(140.0, (w_geom * z_ground) + (w_prior * z_prior)))
        x_fused = (z_fused / max(0.1, z_ground)) * x_ground

        # 3D Euclidean distance (straight line from camera lens to vehicle)
        euclidean_3d = math.sqrt(x_fused**2 + z_fused**2 + self.camera_height**2)

        # 5. Lane Positioning & Semantic Proximity Zone
        lane_idx = int(round(x_fused / self.lane_width))
        if lane_idx == 0:
            lane_desc = "Center Lane"
        elif lane_idx < 0:
            lane_desc = f"Left Lane {abs(lane_idx)}"
        else:
            lane_desc = f"Right Lane {lane_idx}"

        if euclidean_3d <= ZONE_IMMEDIATE:
            zone = "IMMEDIATE_ZONE"
            zone_color = (0, 0, 255)      # Red alert
            risk = "HIGH_PROXIMITY"
        elif euclidean_3d <= ZONE_NEAR:
            zone = "NEAR_ZONE"
            zone_color = (0, 165, 255)    # Amber
            risk = "MEDIUM_PROXIMITY"
        elif euclidean_3d <= ZONE_MID:
            zone = "MID_RANGE"
            zone_color = (0, 230, 230)    # Golden Yellow
            risk = "NORMAL"
        else:
            zone = "FAR_RANGE"
            zone_color = (255, 200, 50)   # Light Blue/Cyan
            risk = "FAR"

        return {
            "box": box,
            "label": label,
            "confidence": confidence,
            "contact_point_px": [int(round(u_contact)), int(round(v_contact))],
            "distance_m": round(euclidean_3d, 2),
            "longitudinal_z_m": round(z_fused, 2),
            "lateral_x_m": round(x_fused, 2),
            "lane_offset": f"{x_fused:+.2f}m",
            "lane_label": lane_desc,
            "proximity_zone": zone,
            "zone_color": zone_color,
            "proximity_risk": risk,
            "estimated_ground_dist_m": round(z_ground, 2),
            "estimated_prior_dist_m": round(z_prior, 2),
        }

    def process_detections(self, detections: list, image_shape: tuple) -> list:
        """
        Processes a list of vehicle detection dicts and adds distance estimations.
        """
        results = []
        for det in detections:
            box = det.get("box", [])
            label = det.get("label", "Car")
            conf = det.get("confidence", 1.0)
            dist_info = self.estimate_single_vehicle(box, label, image_shape, conf)
            merged = {**det, **dist_info}
            results.append(merged)
        return results

    def render_distance_visualization(
        self,
        image: np.ndarray,
        detections: list,
        draw_radar: bool = True,
        radar_size: int = 240,
    ) -> np.ndarray:
        """
        Renders futuristic HUD overlay with distance indicators, ground-plane footprint
        connectors, proximity alerts, and a top-down Bird's-Eye-View (BEV) radar map.
        """
        annotated = image.copy()
        ih, iw = image.shape[:2]

        # Font and line metrics
        scale = max(0.48, max(ih, iw) / 2000.0)
        thickness = max(2, int(round(max(ih, iw) / 800.0)))

        for det in detections:
            box = det.get("box")
            if not box:
                continue
            x1, y1, x2, y2 = [int(round(c)) for c in box]
            lbl = det.get("label", "Vehicle")
            conf = det.get("confidence", 0.0)
            dist_m = det.get("distance_m", 0.0)
            lane_str = det.get("lane_offset", "0.0m")
            zone_color = det.get("zone_color", (0, 255, 0))
            cx_px, cy_px = det.get("contact_point_px", [(x1 + x2) // 2, y2])

            # 1. Main Bounding Box with Rounded Aesthetic Corner Brackets
            cv2.rectangle(annotated, (x1, y1), (x2, y2), zone_color, thickness)
            corner_len = min(22, max(8, int((x2 - x1) * 0.2)))
            thick_corner = thickness + 2

            # Top-Left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), zone_color, thick_corner)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), zone_color, thick_corner)
            # Top-Right
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), zone_color, thick_corner)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), zone_color, thick_corner)
            # Bottom-Left
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), zone_color, thick_corner)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), zone_color, thick_corner)
            # Bottom-Right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), zone_color, thick_corner)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), zone_color, thick_corner)

            # 2. Road Ground Contact Footprint Marker
            cv2.circle(annotated, (cx_px, cy_px), 5, (0, 255, 255), -1)
            cv2.ellipse(annotated, (cx_px, cy_px), (int(round((x2 - x1) * 0.35)), 6), 0, 0, 360, zone_color, 2)

            # 3. High-Contrast HUD Distance Tag
            tag_text = f"{lbl} | {dist_m:.1f}m ({lane_str})"
            (tw, th), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)

            # Header badge above vehicle
            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, badge_y2), (20, 20, 20), -1)
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, badge_y2), zone_color, 2)
            cv2.putText(
                annotated,
                tag_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 4. Top-Down Bird's-Eye-View (BEV) Radar Display Overlay
        if draw_radar and len(detections) > 0:
            radar_img = self._generate_bev_radar(detections, radar_size)
            # Place in top-right corner with 15px margin
            pad = 18
            rh, rw = radar_img.shape[:2]
            if ih > rh + pad and iw > rw + pad:
                x_pos = iw - rw - pad
                y_pos = pad
                # Alpha blend radar into scene
                roi = annotated[y_pos:y_pos + rh, x_pos:x_pos + rw]
                cv2.addWeighted(radar_img, 0.90, roi, 0.10, 0, roi)
                # Outer glowing border for radar
                cv2.rectangle(annotated, (x_pos, y_pos), (x_pos + rw, y_pos + rh), (0, 255, 255), 2)
                cv2.putText(
                    annotated,
                    "BEV 3D RADAR (m)",
                    (x_pos + 10, y_pos + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return annotated

    def _generate_bev_radar(self, detections: list, size: int = 240) -> np.ndarray:
        """
        Creates a dark glassmorphic top-down Bird's-Eye-View (BEV) radar map
        plotting the camera position at the bottom and vehicle coordinates (X, Z).
        """
        radar = np.zeros((size, size, 3), dtype=np.uint8)
        radar[:] = (18, 22, 26)  # Deep dark slate

        cx = size // 2
        cy = size - 22  # Camera origin at bottom center

        max_range_m = 70.0
        scale = (size - 40) / max_range_m

        # Concentric distance rings (15m, 30m, 45m, 60m)
        ranges = [15.0, 30.0, 45.0, 60.0]
        for r in ranges:
            r_px = int(round(r * scale))
            cv2.circle(radar, (cx, cy), r_px, (40, 48, 55), 1)
            cv2.putText(
                radar,
                f"{int(r)}m",
                (cx + 4, cy - r_px + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (90, 105, 120),
                1,
                cv2.LINE_AA,
            )

        # Lateral road guidelines (+/- 3.5m lanes)
        lane_w_px = int(round(self.lane_width * scale))
        cv2.line(radar, (cx - lane_w_px, 20), (cx - lane_w_px, cy), (32, 40, 48), 1)
        cv2.line(radar, (cx + lane_w_px, 20), (cx + lane_w_px, cy), (32, 40, 48), 1)
        cv2.line(radar, (cx, 20), (cx, cy), (0, 100, 130), 1)

        # Plot camera origin icon
        cv2.circle(radar, (cx, cy), 6, (0, 255, 255), -1)
        cv2.putText(
            radar,
            "CAM",
            (cx - 11, cy + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Plot vehicle blips
        for det in detections:
            x_m = det.get("lateral_x_m", 0.0)
            z_m = det.get("longitudinal_z_m", 0.0)
            color = det.get("zone_color", (0, 255, 0))
            lbl = det.get("label", "C")[0]

            px = int(round(cx + (x_m * scale)))
            py = int(round(cy - (z_m * scale)))

            if 8 <= px <= size - 8 and 8 <= py <= size - 8:
                # Blip with glow
                cv2.circle(radar, (px, py), 6, color, -1)
                cv2.circle(radar, (px, py), 9, color, 1)
                # Class initial inside blip
                cv2.putText(
                    radar,
                    lbl,
                    (px - 3, py + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return radar
