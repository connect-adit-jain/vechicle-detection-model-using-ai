"""
shared/visual_fingerprint.py
Tier-2 Visual Fingerprint & Disambiguation Engine ("YOLO27" Fingerprint Pipeline).

Handles occlusion, extreme lighting variations, and separates near-identical vehicles
using fine-grained localized visual fingerprints:

1. Normal Recognition using Visual Fingerprints:
   - Scratches & Dents (high-frequency gradient / localized edge response)
   - Colour distribution & Hue invariant descriptors (lighting-robust HSV/LAB)
   - Headlights & Carrier structural zones (top roof rack, lower fascia)
   - Side mirrors & Handle style (lateral contour features)
   - Stickers & Distinctive decals (localized contrast hotspots)
   - Dust / road dirt gradient patterns
   - Rider clothes (for two-wheelers, upper torso appearance)

2. Disambiguation for Near-Identical Vehicles:
   - FASTag placement using a 3x3 windshield spatial grid
   - Tyre pressure / contact patch compression profile
   - Sitting posture: EXCLUDED (marked prohibited/crossed out in spec)
"""
import cv2
import numpy as np


class VisualFingerprintEngine:
    def __init__(self, grid_size=(3, 3), enable_fastag=True):
        self.grid_size = grid_size
        self.enable_fastag = enable_fastag

    def extract_lighting_invariant_color(self, crop_bgr):
        """Extract lighting-robust color histogram using LAB color space (L channel decoupled)."""
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        # Compute 2D histogram on A and B channels (chrominance, invariant to illumination L)
        hist_ab = cv2.calcHist([lab], [1, 2], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(hist_ab, hist_ab, 0, 1, cv2.NORM_MINMAX)
        return hist_ab.flatten()

    def extract_surface_anomalies(self, crop_bgr):
        """
        Detect localized surface features: scratches, dents, stickers, and dust patterns
        via multi-scale Laplacian of Gaussian & Sobel gradients.
        """
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 16 or w < 16:
            return {
                "scratch_dent_energy": 0.0,
                "sticker_hotspots": 0,
                "dust_gradient": 0.0,
                "anomaly_vector": np.zeros(16, dtype=np.float32)
            }

        # Bilateral filter to smooth normal paint reflection while preserving scratch/dent edges
        smooth = cv2.bilateralFilter(gray, 7, 50, 50)
        diff = cv2.absdiff(gray, smooth)
        
        # High-frequency edge energy (scratches / dents / stickers)
        laplacian = cv2.Laplacian(diff, cv2.CV_32F)
        scratch_energy = float(np.mean(np.abs(laplacian)))

        # Sticker / decal hotspot count (isolated bright/dark distinct blobs on body)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_hotspots = [c for c in contours if 10 < cv2.contourArea(c) < (h * w * 0.05)]

        # Dust / road dirt vertical gradient (dirt accumulates heavier on lower quarter)
        top_half_mean = float(np.mean(gray[:h // 2, :]))
        bot_half_mean = float(np.mean(gray[h // 2:, :]))
        dust_gradient = abs(top_half_mean - bot_half_mean) / (top_half_mean + bot_half_mean + 1e-5)

        # 4x4 spatial anomaly energy map
        cells_h, cells_w = 4, 4
        dh, dw = h // cells_h, w // cells_w
        anomaly_map = np.zeros(cells_h * cells_w, dtype=np.float32)
        idx = 0
        for i in range(cells_h):
            for j in range(cells_w):
                cell = diff[i * dh:(i + 1) * dh, j * dw:(j + 1) * dw]
                anomaly_map[idx] = float(np.mean(cell))
                idx += 1
        norm = np.linalg.norm(anomaly_map)
        if norm > 1e-6:
            anomaly_map /= norm

        return {
            "scratch_dent_energy": scratch_energy,
            "sticker_hotspots": len(valid_hotspots),
            "dust_gradient": dust_gradient,
            "anomaly_vector": anomaly_map
        }

    def detect_fastag_grid_placement(self, crop_bgr):
        """
        Detect FASTag RFID tag placement on the windshield using a 3x3 spatial grid.
        Returns:
          has_fastag: bool
          grid_cell: [row, col] (0-2, 0-2)
          confidence: float
        """
        h, w, _ = crop_bgr.shape
        # Adaptively isolate windshield region
        y1, y2 = int(h * 0.06), int(h * 0.55)
        x1, x2 = int(w * 0.12), int(w * 0.88)
        
        if y2 <= y1 or x2 <= x1:
            windshield = crop_bgr
        else:
            windshield = crop_bgr[y1:y2, x1:x2]

        wh, ww, _ = windshield.shape
        if wh < 15 or ww < 15:
            return {"has_tag": False, "grid_cell": None, "conf": 0.0}

        # Multi-cue detection: HSV lightness contrast + Canny edge morphology
        gray = cv2.cvtColor(windshield, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 110)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        hsv = cv2.cvtColor(windshield, cv2.COLOR_BGR2HSV)
        val = hsv[:, :, 2]
        val_diff = cv2.absdiff(val, cv2.GaussianBlur(val, (15, 15), 0))
        _, thresh_val = cv2.threshold(val_diff, 20, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(edges_closed, thresh_val)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cell = None
        max_score = 0.0
        ws_area = ww * wh
        
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            aspect = float(bw) / max(bh, 1)

            # FASTag tag constraints: aspect ratio 1.25 to 3.8, area 0.3% to 15% of windshield
            if (ws_area * 0.003) < area < (ws_area * 0.14) and 1.25 <= aspect <= 3.8:
                cx = bx + bw / 2.0
                cy = by + bh / 2.0
                grid_col = min(int(cx / (ww / 3.0)), 2)
                grid_row = min(int(cy / (wh / 3.0)), 2)
                score = area * (1.0 - abs(aspect - 2.2) * 0.25)
                if score > max_score:
                    max_score = score
                    best_cell = [grid_row, grid_col]

        has_tag = best_cell is not None
        conf = float(min(max_score / (ws_area * 0.05), 1.0)) if has_tag else 0.0
        return {"has_tag": has_tag, "grid_cell": best_cell, "conf": round(conf, 2)}

    def extract_tyre_dynamic_profile(self, crop_bgr):
        """
        Analyze dynamic tyre pressure / contact patch compression profile.
        Under load, tyre wall deflection alters the bottom bounding ratio.
        """
        h, w, _ = crop_bgr.shape
        # Lower 20% contains tyre contact area
        bottom_region = crop_bgr[int(h * 0.80):, :]
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        
        # Black rubber thresholding
        _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        black_pixel_ratio = float(np.sum(dark_mask > 0)) / max(gray.size, 1)
        
        # Vertical gradient of contact patch
        vert_hist = np.sum(dark_mask > 0, axis=1).astype(np.float32)
        if len(vert_hist) > 0 and np.max(vert_hist) > 0:
            vert_hist /= np.max(vert_hist)
            compression_index = float(np.std(vert_hist))
        else:
            compression_index = 0.0

        return {
            "black_pixel_ratio": black_pixel_ratio,
            "compression_index": compression_index
        }

    def extract_structural_styling(self, crop_bgr):
        """
        Extract structural styling cues:
        - Roof carrier / rack presence (top 15% contour structure)
        - Side mirror lateral protrusion (lateral 10% bounds)
        - Headlight / tail-light intensity distribution
        """
        h, w, _ = crop_bgr.shape
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Roof carrier: top 15% edge complexity
        top_crop = gray[:max(int(h * 0.15), 1), :]
        top_edges = cv2.Canny(top_crop, 50, 150)
        carrier_score = float(np.sum(top_edges > 0)) / max(top_crop.size, 1)

        # 2. Side mirrors: lateral edge extensions around middle 30-60% height
        mid_y1, mid_y2 = int(h * 0.30), int(h * 0.60)
        left_lateral = gray[mid_y1:mid_y2, :max(int(w * 0.12), 1)]
        right_lateral = gray[mid_y1:mid_y2, max(int(w * 0.88), 1):]
        mirror_score = float(np.sum(cv2.Canny(left_lateral, 50, 150) > 0) + 
                             np.sum(cv2.Canny(right_lateral, 50, 150) > 0)) / (left_lateral.size + right_lateral.size + 1e-5)

        # 3. Headlight / lamp cluster intensity
        lower_mid = gray[int(h * 0.50):int(h * 0.75), :]
        headlight_intensity = float(np.mean(lower_mid))

        return {
            "carrier_presence": carrier_score > 0.08,
            "carrier_score": carrier_score,
            "mirror_style_score": mirror_score,
            "lamp_intensity": headlight_intensity
        }

    def extract_rider_clothing(self, crop_bgr, vehicle_class):
        """
        For two-wheelers: extract rider clothing color/histogram
        (useful for commuter two-wheelers with identical models).
        """
        if vehicle_class != "two_wheeler":
            return None
        h, w, _ = crop_bgr.shape
        # Upper 40% center contains the rider's torso/helmet
        y1, y2 = int(h * 0.05), int(h * 0.45)
        x1, x2 = int(w * 0.25), int(w * 0.75)
        if y2 <= y1 or x2 <= x1:
            return None
        torso = crop_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180])
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        return hist_h.flatten().tolist()

    def generate_visual_fingerprint(self, crop_bgr, vehicle_class="car"):
        """
        Synthesize complete visual fingerprint for the vehicle.
        Combines normal recognition fingerprints + disambiguation cues for near-identical vehicles.
        """
        color_feat = self.extract_lighting_invariant_color(crop_bgr)
        anomalies = self.extract_surface_anomalies(crop_bgr)
        if self.enable_fastag:
            fastag = self.detect_fastag_grid_placement(crop_bgr)
        else:
            fastag = {"has_tag": False, "grid_cell": None, "conf": 0.0}
        tyre = self.extract_tyre_dynamic_profile(crop_bgr)
        styling = self.extract_structural_styling(crop_bgr)
        rider = self.extract_rider_clothing(crop_bgr, vehicle_class)

        # Concatenate into a compact numerical fingerprint vector
        feat_list = [
            color_feat,
            anomalies["anomaly_vector"],
            np.array([
                anomalies["scratch_dent_energy"],
                float(anomalies["sticker_hotspots"]),
                anomalies["dust_gradient"],
                float(fastag["has_tag"]),
                float(fastag["grid_cell"][0]) if fastag["grid_cell"] else -1.0,
                float(fastag["grid_cell"][1]) if fastag["grid_cell"] else -1.0,
                tyre["black_pixel_ratio"],
                tyre["compression_index"],
                float(styling["carrier_presence"]),
                styling["mirror_style_score"],
            ], dtype=np.float32)
        ]
        fingerprint_vector = np.concatenate(feat_list)
        norm = np.linalg.norm(fingerprint_vector)
        if norm > 1e-6:
            fingerprint_vector /= norm

        return {
            "fingerprint_vector": fingerprint_vector.tolist(),
            "surface_anomalies": {
                "scratch_dent_detected": anomalies["scratch_dent_energy"] > 2.5,
                "scratch_dent_energy": round(anomalies["scratch_dent_energy"], 3),
                "sticker_count": anomalies["sticker_hotspots"],
                "dust_gradient": round(anomalies["dust_gradient"], 3),
            },
            "fastag_placement": {
                "detected": fastag["has_tag"],
                "grid_3x3_cell": fastag["grid_cell"],
                "confidence": round(fastag["conf"], 2),
            },
            "tyre_profile": {
                "compression_index": round(tyre["compression_index"], 3),
                "contact_ratio": round(tyre["black_pixel_ratio"], 3),
            },
            "structural_styling": {
                "roof_carrier": styling["carrier_presence"],
                "mirror_protrusion_score": round(styling["mirror_style_score"], 3),
                "lamp_intensity": round(styling["lamp_intensity"], 1),
            },
            "rider_clothing": rider,
        }
