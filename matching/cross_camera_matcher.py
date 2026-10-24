"""
matching/cross_camera_matcher.py
Cross-Camera Vehicle Re-Identification & Global Unique ID Assignment.

Associates vehicles across non-overlapping camera views using:
1. Global Visual Appearance Embedding (OSNet x1.0, 512-dim)
2. Tier-2 YOLO27 Visual Fingerprint (Scratches, Dents, FASTag 3x3 Grid, Tyre profile, Carrier, Mirrors)
3. Spatio-Temporal Travel Time & Roadway Topology Constraints
4. Global Persistent Vehicle ID Generation (VEH-0001, VEH-0002...)
"""
import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment

from shared.adapters import normalize_embedding, aggregate_embeddings as aggregate_track_embeddings
from shared.visual_fingerprint import VisualFingerprintEngine


class CrossCameraMatcher:
    def __init__(self, embedder_model_dict=None, sim_threshold=0.55, time_weight=0.25, fp_weight=0.35, device="cuda"):
        self.sim_threshold = sim_threshold
        self.time_weight = time_weight
        self.fp_weight = fp_weight
        self.device = device
        
        # Load OSNet embedder if provided, else lazy load
        if embedder_model_dict is None:
            from models.reid.reid_embed import load_model as load_reid, embed as embed_reid
            self.reid_model = load_reid(device=device)
            self.embed_fn = embed_reid
        else:
            self.reid_model = embedder_model_dict
            from models.reid.reid_embed import embed as embed_reid
            self.embed_fn = embed_reid

        # Visual fingerprint engine (Tier 2)
        self.fp_engine = VisualFingerprintEngine()
        self.global_id_counter = 1
        self.global_registry = {} # global_id -> aggregated record

    def extract_vehicle_features(self, crops_bgr, vehicle_class="car"):
        """
        Extract global Re-ID embedding and fine-grained visual fingerprints from vehicle crops.
        Privacy rule: raw crops are processed and discarded in-memory.
        """
        if not crops_bgr:
            return None, None

        # 1. OSNet 512-dim embedding
        raw_embs = self.embed_fn(self.reid_model, crops_bgr)
        global_emb = aggregate_track_embeddings(raw_embs)

        # 2. Tier-2 Visual Fingerprint (averaged across valid crops)
        best_crop = max(crops_bgr, key=lambda c: c.shape[0] * c.shape[1])
        visual_fp = self.fp_engine.generate_visual_fingerprint(best_crop, vehicle_class=vehicle_class)

        return global_emb, visual_fp

    def compute_fingerprint_similarity(self, fp1, fp2):
        """
        Compute similarity score [0, 1] between two Tier-2 Visual Fingerprints.
        Handles near-identical vehicles:
        - If FASTag 3x3 grid cells match -> strong positive boost
        - If FASTag 3x3 grid cells conflict (e.g. [0,1] vs [0,2]) -> strong penalty
        - Compares scratch/dent energy and structural styling (carrier, mirrors)
        """
        if not fp1 or not fp2:
            return 0.5

        # 1. Vector cosine similarity
        v1 = np.array(fp1["fingerprint_vector"], dtype=np.float32)
        v2 = np.array(fp2["fingerprint_vector"], dtype=np.float32)
        dot = float(np.dot(v1, v2))
        base_sim = (dot + 1.0) / 2.0 # map [-1, 1] to [0, 1]

        # 2. FASTag 3x3 Grid Disambiguation
        tag1 = fp1.get("fastag_placement", {})
        tag2 = fp2.get("fastag_placement", {})
        if tag1.get("detected") and tag2.get("detected"):
            cell1 = tag1.get("grid_3x3_cell")
            cell2 = tag2.get("grid_3x3_cell")
            if cell1 and cell2:
                if cell1 == cell2:
                    base_sim += 0.20 # Exact same placement cell
                else:
                    base_sim -= 0.35 # Different windshield location (proves DIFFERENT vehicle)
        elif tag1.get("detected") != tag2.get("detected"):
            base_sim -= 0.15 # One vehicle has tag, other does not

        # 3. Structural styling match (Roof carrier)
        carrier1 = fp1.get("structural_styling", {}).get("roof_carrier", False)
        carrier2 = fp2.get("structural_styling", {}).get("roof_carrier", False)
        if carrier1 != carrier2:
            base_sim -= 0.25 # One has roof carrier, other does not

        # 4. Scratch & Dent profile consistency
        anom1 = fp1.get("surface_anomalies", {}).get("scratch_dent_energy", 0)
        anom2 = fp2.get("surface_anomalies", {}).get("scratch_dent_energy", 0)
        scratch_diff = abs(anom1 - anom2)
        if scratch_diff < 1.0 and (anom1 > 2.0 or anom2 > 2.0):
            base_sim += 0.10 # Matching distinct surface anomaly

        return float(np.clip(base_sim, 0.0, 1.0))

    def evaluate_pair_match(self, track_a, track_b, travel_dist_m=None):
        """
        Evaluate match score between two vehicle tracks from different cameras.
        Returns: total_score (float), is_plausible (bool)
        """
        # Class compatibility constraint
        if track_a["class"] != track_b["class"]:
            return 0.0, False

        # Spatio-temporal causality: track_b must appear after track_a
        delta_t = track_b["t_start"] - track_a["t_end"]
        if delta_t < -2.0: # Track B started before Track A ended
            return 0.0, False

        # Physical speed feasibility (assuming typical road travel)
        if travel_dist_m is not None and delta_t > 0:
            speed_mps = travel_dist_m / max(delta_t, 1.0)
            # Allow speeds between 2 m/s (7 km/h) and 45 m/s (160 km/h)
            if speed_mps > 45.0 or speed_mps < 1.0:
                return 0.0, False
            time_score = 1.0 - abs(speed_mps - 13.8) / 30.0 # centered around ~50 km/h
            time_score = float(np.clip(time_score, 0.0, 1.0))
        else:
            time_score = 1.0 if delta_t >= 0 else 0.0

        # Global Re-ID embedding cosine similarity
        emb_a = np.array(track_a["embedding"])
        emb_b = np.array(track_b["embedding"])
        app_sim = float(np.dot(emb_a, emb_b)) # already L2-normalized

        # Tier-2 Visual Fingerprint similarity
        fp_sim = self.compute_fingerprint_similarity(track_a.get("visual_fingerprint"), track_b.get("visual_fingerprint"))

        # Fused Match Score
        total_score = (1.0 - self.time_weight - self.fp_weight) * app_sim + \
                      self.fp_weight * fp_sim + \
                      self.time_weight * time_score

        is_match = total_score >= self.sim_threshold
        return float(total_score), is_match

    def match_camera_tracks(self, tracks_cam_a, tracks_cam_b, distance_m=100.0):
        """
        Perform bipartite matching across Camera A and Camera B.
        Assigns persistent global unique IDs across cameras.
        """
        na = len(tracks_cam_a)
        nb = len(tracks_cam_b)
        if na == 0 or nb == 0:
            # Assign fresh global IDs to non-matching tracks
            for t in tracks_cam_a + tracks_cam_b:
                if "global_id" not in t:
                    t["global_id"] = f"VEH-{self.global_id_counter:04d}"
                    self.global_id_counter += 1
            return []

        cost_matrix = np.ones((na, nb), dtype=np.float32)

        for i, ta in enumerate(tracks_cam_a):
            for j, tb in enumerate(tracks_cam_b):
                score, is_plausible = self.evaluate_pair_match(ta, tb, travel_dist_m=distance_m)
                if is_plausible:
                    cost_matrix[i, j] = 1.0 - score
                else:
                    cost_matrix[i, j] = 10.0 # Prohibited match

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost < (1.0 - self.sim_threshold):
                score = 1.0 - cost
                ta = tracks_cam_a[r]
                tb = tracks_cam_b[c]

                # Propagate or create persistent global ID
                if "global_id" in ta:
                    gid = ta["global_id"]
                elif "global_id" in tb:
                    gid = tb["global_id"]
                else:
                    gid = f"VEH-{self.global_id_counter:04d}"
                    self.global_id_counter += 1

                ta["global_id"] = gid
                tb["global_id"] = gid

                matches.append({
                    "global_id": gid,
                    "cam_a": ta["camera_id"],
                    "track_a": ta["track_id"],
                    "cam_b": tb["camera_id"],
                    "track_b": tb["track_id"],
                    "class": ta["class"],
                    "match_score": round(score, 3),
                    "travel_time_s": round(tb["t_start"] - ta["t_end"], 2)
                })

        # Assign unique global IDs to all unmatched tracks
        for ta in tracks_cam_a:
            if "global_id" not in ta:
                ta["global_id"] = f"VEH-{self.global_id_counter:04d}"
                self.global_id_counter += 1

        for tb in tracks_cam_b:
            if "global_id" not in tb:
                tb["global_id"] = f"VEH-{self.global_id_counter:04d}"
                self.global_id_counter += 1

        return matches
