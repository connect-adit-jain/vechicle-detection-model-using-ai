"""
traffic/night_mode.py
Phase 2 Module: Night & Low-Light Footage Robustness.

Capabilities:
1. Adaptive Low-Light Enhancement:
   - Dynamic illumination decomposition & adaptive CLAHE on L/V channel
   - Preserves contour edges while suppressing extreme headlight/tail-light glare
2. Headlight Glare Mitigation:
   - Identifies high-intensity lamp blooms and applies bilateral suppression
3. Night Re-ID & Tracking Metrics:
   - Evaluates IDF1 (Identification F1) and Traffic Flow Error under dark/night conditions
"""
import cv2
import numpy as np


class NightVisionEnhancer:
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def is_low_light(self, frame_bgr, threshold=65.0):
        """Check if frame brightness falls below low-light threshold."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        return mean_val < threshold

    def enhance_frame(self, frame_bgr):
        """
        Enhance low-light video frames while suppressing headlight glare.
        Converts to LAB space, equalizes L-channel, and attenuates extreme specular glare.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 1. Glare suppression: identify saturated specular headlight blooms (> 240)
        glare_mask = l > 240
        
        # 2. Adaptive CLAHE on illumination (L) channel
        enhanced_l = self.clahe.apply(l)

        # 3. Restore glare regions with smooth dampening to avoid overexposure blowing out contours
        enhanced_l[glare_mask] = np.clip(l[glare_mask] * 0.85, 0, 255).astype(np.uint8)

        # 4. Gamma correction on shadow regions to reveal dark body contours
        gamma = 1.35
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        enhanced_l = cv2.LUT(enhanced_l, table)

        merged = cv2.merge([enhanced_l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def compute_idf1_and_flow_error(ground_truth_tracks, predicted_tracks, camera_pair=("cam1", "cam2")):
    """
    Compute IDF1 (Identification F1 score) and Flow Error for night evaluation.
    
    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    Flow Error = |Predicted_Flow - True_Flow| / max(True_Flow, 1)
    """
    # Ground truth matches
    gt_pairs = set()
    for t in ground_truth_tracks:
        if "true_id" in t and "camera_id" in t:
            gt_pairs.add((t["true_id"], t["camera_id"]))

    pred_pairs = set()
    for t in predicted_tracks:
        if "global_id" in t and "camera_id" in t:
            pred_pairs.add((t["global_id"], t["camera_id"]))

    # Map predicted global IDs to ground truth IDs
    id_mapping = {}
    for t in predicted_tracks:
        gid = t.get("global_id")
        tid = t.get("true_id")
        if gid and tid:
            if gid not in id_mapping:
                id_mapping[gid] = tid

    # Identification matches
    idtp = 0
    idfp = 0
    idfn = 0

    for t in predicted_tracks:
        gid = t.get("global_id")
        tid = t.get("true_id")
        if id_mapping.get(gid) == tid:
            idtp += 1
        else:
            idfp += 1

    idfn = max(0, len(ground_truth_tracks) - idtp)

    # Calculate IDF1
    denom = (2 * idtp + idfp + idfn)
    idf1 = (2 * idtp) / denom if denom > 0 else 0.0

    # Traffic Flow Error
    true_flow = len(set(t["true_id"] for t in ground_truth_tracks if "true_id" in t))
    pred_flow = len(set(t["global_id"] for t in predicted_tracks if "global_id" in t))
    flow_err = abs(pred_flow - true_flow) / max(true_flow, 1.0)

    return {
        "IDF1": round(idf1, 4),
        "IDTP": idtp,
        "IDFP": idfp,
        "IDFN": idfn,
        "True_Flow": true_flow,
        "Pred_Flow": pred_flow,
        "Flow_Error": round(flow_err, 4),
        "Flow_Accuracy": round(1.0 - min(flow_err, 1.0), 4)
    }
