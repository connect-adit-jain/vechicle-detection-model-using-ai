"""
Step 0.1: Embedder Comparison on VeRi-776
Compares osnet_v1, osnet_grid_v1, and DINOv2-small fallback.
Metrics: VeRi mAP and Rank-1, plus SEMI-SIMULATED OD scenario.
"""
import os
import sys
import time
import random
import numpy as np
import torch
import cv2
from collections import defaultdict

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VERI_DIR = os.path.join(ROOT, "data", "veri")
QUERY_DIR = os.path.join(VERI_DIR, "image_query")
TEST_DIR = os.path.join(VERI_DIR, "image_test")

def parse_veri_filename(fname):
    """Parse VeRi filename: VVVV_cCCC_FFFFFFFF_T.jpg -> (vehicle_id, camera_id)"""
    parts = fname.split("_")
    vid = int(parts[0])
    cid = int(parts[1][1:])
    return vid, cid

def load_images_by_id(img_dir, max_per_id=6, max_ids=200):
    """Load images grouped by vehicle ID."""
    images = defaultdict(list)
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".jpg"):
            continue
        vid, cid = parse_veri_filename(fname)
        if len(images) >= max_ids and vid not in images:
            continue
        if len(images[vid]) < max_per_id:
            path = os.path.join(img_dir, fname)
            img = cv2.imread(path)
            if img is not None:
                images[vid].append({"img": img, "fname": fname, "cam": cid})
    return dict(images)

def compute_mAP_rank1(query_embeds, query_ids, query_cams, gallery_embeds, gallery_ids, gallery_cams):
    """Compute mAP and Rank-1 for VeRi evaluation."""
    # Cosine similarity matrix (embeddings are already L2-normalised)
    sim_matrix = query_embeds @ gallery_embeds.T  # (Q, G)
    
    aps = []
    rank1_hits = 0
    for qi in range(len(query_ids)):
        qid = query_ids[qi]
        qcam = query_cams[qi]
        sims = sim_matrix[qi]
        
        # Sort gallery by descending similarity
        sorted_idx = np.argsort(-sims)
        
        # Remove same-id same-camera (standard VeRi protocol)
        valid_mask = np.ones(len(gallery_ids), dtype=bool)
        for gi in range(len(gallery_ids)):
            if gallery_ids[gi] == qid and gallery_cams[gi] == qcam:
                valid_mask[gi] = False
        
        sorted_idx = [i for i in sorted_idx if valid_mask[i]]
        
        # Compute AP
        matches = [gallery_ids[i] == qid for i in sorted_idx]
        if sum(matches) == 0:
            continue
        
        # Rank-1
        if matches[0]:
            rank1_hits += 1
        
        # AP
        cum_tp = 0
        precision_sum = 0
        for rank, is_match in enumerate(matches, 1):
            if is_match:
                cum_tp += 1
                precision_sum += cum_tp / rank
        ap = precision_sum / sum(matches)
        aps.append(ap)
    
    mAP = np.mean(aps) if aps else 0.0
    rank1 = rank1_hits / len(query_ids) if len(query_ids) > 0 else 0.0
    return mAP, rank1


def run_semi_simulated_od(embedder_name, embed_fn, test_images):
    """
    SEMI-SIMULATED OD scenario:
    Real VeRi embeddings routed through synthetic 4-camera graph with travel times.
    Includes deliberately look-alike vehicles (same colour/type, nearest embeddings).
    """
    random.seed(42)
    np.random.seed(42)
    
    # Synthetic camera graph: 4 cameras, edges with travel time
    edges = [
        {"from": 1, "to": 2, "travel_mean_s": 30, "travel_std_s": 5},
        {"from": 2, "to": 3, "travel_mean_s": 45, "travel_std_s": 8},
        {"from": 1, "to": 3, "travel_mean_s": 60, "travel_std_s": 10},
        {"from": 3, "to": 4, "travel_mean_s": 20, "travel_std_s": 3},
    ]
    
    # Select vehicles and embed them
    vehicle_ids = sorted(test_images.keys())[:50]  # Use up to 50 vehicles
    vehicle_embeds = {}
    for vid in vehicle_ids:
        imgs = [item["img"] for item in test_images[vid]]
        if len(imgs) == 0:
            continue
        emb = embed_fn(imgs)
        vehicle_embeds[vid] = np.mean(emb, axis=0)
        vehicle_embeds[vid] /= np.linalg.norm(vehicle_embeds[vid]) + 1e-8
    
    vids_with_embeds = [v for v in vehicle_ids if v in vehicle_embeds]
    if len(vids_with_embeds) < 10:
        return None
    
    # Create synthetic tracks: each vehicle appears at 2-3 cameras
    tracks = []
    track_id = 0
    gt_matches = []
    
    for vid in vids_with_embeds:
        # Random route: appear at camera 1, then travel to camera 2 or 3
        cam1 = random.choice([1, 2, 3])
        t_start = random.uniform(0, 300)
        
        tracks.append({
            "track_id": track_id,
            "camera_id": cam1,
            "vehicle_id": vid,  # ground truth
            "t_start": t_start,
            "t_end": t_start + random.uniform(2, 8),
            "embedding": vehicle_embeds[vid],
            "embedder": embedder_name,
        })
        src_track = track_id
        track_id += 1
        
        # Find a valid edge from cam1
        valid_edges = [e for e in edges if e["from"] == cam1]
        if valid_edges:
            edge = random.choice(valid_edges)
            travel_time = np.random.normal(edge["travel_mean_s"], edge["travel_std_s"])
            travel_time = max(5, travel_time)
            t2_start = tracks[-1]["t_end"] + travel_time
            
            tracks.append({
                "track_id": track_id,
                "camera_id": edge["to"],
                "vehicle_id": vid,
                "t_start": t2_start,
                "t_end": t2_start + random.uniform(2, 8),
                "embedding": vehicle_embeds[vid],
                "embedder": embedder_name,
            })
            gt_matches.append({"src": src_track, "dst": track_id, "vehicle_id": vid})
            track_id += 1
    
    # Identify look-alikes: for each vehicle, find the nearest embedding from a different vehicle
    all_embeds = np.array([vehicle_embeds[v] for v in vids_with_embeds])
    sim_matrix = all_embeds @ all_embeds.T
    np.fill_diagonal(sim_matrix, -1)
    
    look_alike_pairs = []
    for i, vid in enumerate(vids_with_embeds):
        j = np.argmax(sim_matrix[i])
        look_alike_vid = vids_with_embeds[j]
        if sim_matrix[i, j] > 0.5:
            look_alike_pairs.append((vid, look_alike_vid, float(sim_matrix[i, j])))
    
    # Run matching: cosine similarity + travel time gating
    correct = 0
    total = 0
    look_alike_correct = 0
    look_alike_total = 0
    
    look_alike_vids = set()
    for v1, v2, _ in look_alike_pairs[:10]:
        look_alike_vids.add(v1)
        look_alike_vids.add(v2)
    
    for gt in gt_matches:
        src = tracks[gt["src"]]
        best_score = -1
        best_dst = None
        
        for t in tracks:
            if t["camera_id"] == src["camera_id"]:
                continue
            if t["track_id"] == src["track_id"]:
                continue
            
            # Time gating
            time_diff = t["t_start"] - src["t_end"]
            if time_diff < 0 or time_diff > 120:
                continue
            
            # Cosine similarity
            cos_sim = float(np.dot(src["embedding"], t["embedding"]))
            
            # Travel time score (gaussian)
            edge_matches = [e for e in edges if e["from"] == src["camera_id"] and e["to"] == t["camera_id"]]
            time_score = 0
            if edge_matches:
                e = edge_matches[0]
                time_score = np.exp(-0.5 * ((time_diff - e["travel_mean_s"]) / e["travel_std_s"]) ** 2)
            
            score = 0.6 * cos_sim + 0.3 * time_score + 0.1 * 1.0  # attr=neutral
            
            if score > best_score:
                best_score = score
                best_dst = t
        
        if best_dst is not None:
            predicted_vid = best_dst["vehicle_id"]
            actual_vid = gt["vehicle_id"]
            is_correct = (predicted_vid == actual_vid)
            correct += int(is_correct)
            total += 1
            
            if actual_vid in look_alike_vids:
                look_alike_correct += int(is_correct)
                look_alike_total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    look_alike_acc = look_alike_correct / look_alike_total if look_alike_total > 0 else 0.0
    
    # OD error: compare predicted OD matrix vs ground truth
    cameras = sorted(set(t["camera_id"] for t in tracks))
    n_cams = len(cameras)
    cam_idx = {c: i for i, c in enumerate(cameras)}
    
    gt_od = np.zeros((n_cams, n_cams))
    pred_od = np.zeros((n_cams, n_cams))
    
    for gt_m in gt_matches:
        src_cam = tracks[gt_m["src"]]["camera_id"]
        dst_cam = tracks[gt_m["dst"]]["camera_id"]
        gt_od[cam_idx[src_cam], cam_idx[dst_cam]] += 1
    
    # Predicted matches
    matched_srcs = set()
    for gt_m in gt_matches:
        src = tracks[gt_m["src"]]
        best_score = -1
        best_dst_cam = None
        
        for t in tracks:
            if t["camera_id"] == src["camera_id"] or t["track_id"] == src["track_id"]:
                continue
            time_diff = t["t_start"] - src["t_end"]
            if time_diff < 0 or time_diff > 120:
                continue
            cos_sim = float(np.dot(src["embedding"], t["embedding"]))
            edge_matches = [e for e in edges if e["from"] == src["camera_id"] and e["to"] == t["camera_id"]]
            time_score = 0
            if edge_matches:
                e = edge_matches[0]
                time_score = np.exp(-0.5 * ((time_diff - e["travel_mean_s"]) / e["travel_std_s"]) ** 2)
            score = 0.6 * cos_sim + 0.3 * time_score + 0.1
            if score > best_score:
                best_score = score
                best_dst_cam = t["camera_id"]
        
        if best_dst_cam is not None:
            pred_od[cam_idx[src["camera_id"]], cam_idx[best_dst_cam]] += 1
    
    od_error = float(np.mean(np.abs(gt_od - pred_od)))
    od_rel_error = float(np.mean(np.abs(gt_od - pred_od) / (gt_od + 1e-8)))
    
    return {
        "accuracy": accuracy,
        "look_alike_accuracy": look_alike_acc,
        "look_alike_pairs": len(look_alike_pairs),
        "od_mae": od_error,
        "od_relative_error": od_rel_error,
        "total_matches": total,
        "total_tracks": len(tracks),
    }


def main():
    print("=" * 70)
    print("STEP 0.1: EMBEDDER COMPARISON ON VeRi-776")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if not os.path.exists(QUERY_DIR) or not os.path.exists(TEST_DIR):
        print("ERROR: data/veri/image_query or image_test not found. Cannot proceed.")
        return
    
    # Load VeRi images
    print("\nLoading VeRi query images...")
    query_images = load_images_by_id(QUERY_DIR, max_per_id=1, max_ids=500)
    print(f"  Query: {len(query_images)} vehicle IDs")
    
    print("Loading VeRi test/gallery images...")
    gallery_images = load_images_by_id(TEST_DIR, max_per_id=6, max_ids=500)
    print(f"  Gallery: {len(gallery_images)} vehicle IDs")
    
    # Also load test images for SEMI-SIMULATED
    test_images_for_sim = load_images_by_id(TEST_DIR, max_per_id=4, max_ids=100)
    
    embedders = {}
    
    # --- osnet_v1 ---
    print("\n--- Loading osnet_v1 ---")
    try:
        from models.reid.reid_embed import load_model as load_osnet, embed as embed_osnet, EMBEDDER_NAME as osnet_name
        osnet_model = load_osnet(device=device)
        embedders["osnet_v1"] = (osnet_model, embed_osnet, osnet_name)
        print(f"  Loaded: {osnet_name}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # --- osnet_grid_v1 ---
    print("\n--- Loading osnet_grid_v1 ---")
    try:
        from models.reid_grid.reid_embed import load_model as load_grid, embed as embed_grid, EMBEDDER_NAME as grid_name
        grid_model = load_grid(device=device)
        embedders["osnet_grid_v1"] = (grid_model, embed_grid, grid_name)
        print(f"  Loaded: {grid_name}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # --- DINOv2-small fallback ---
    print("\n--- Loading DINOv2-small fallback ---")
    try:
        from shared.dinov2_fallback import load_model as load_dino, embed as embed_dino, EMBEDDER_NAME as dino_name
        dino_model = load_dino(device=device)
        embedders["dinov2_small"] = (dino_model, embed_dino, dino_name)
        print(f"  Loaded: {dino_name}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    results = {}
    
    for name, (model_dict, embed_fn, emb_name) in embedders.items():
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {name}")
        print(f"{'='*60}")
        
        # Embed query images
        print("  Embedding query images...")
        t0 = time.time()
        query_embeds = []
        query_ids = []
        query_cams = []
        for vid, items in query_images.items():
            for item in items:
                emb = embed_fn(model_dict, [item["img"]])
                query_embeds.append(emb[0])
                query_ids.append(vid)
                query_cams.append(item["cam"])
        query_embeds = np.array(query_embeds)
        q_time = time.time() - t0
        print(f"  Query: {len(query_embeds)} images embedded in {q_time:.1f}s")
        
        # Embed gallery images
        print("  Embedding gallery images...")
        t0 = time.time()
        gallery_embeds = []
        gallery_ids = []
        gallery_cams = []
        for vid, items in gallery_images.items():
            for item in items:
                emb = embed_fn(model_dict, [item["img"]])
                gallery_embeds.append(emb[0])
                gallery_ids.append(vid)
                gallery_cams.append(item["cam"])
        gallery_embeds = np.array(gallery_embeds)
        g_time = time.time() - t0
        print(f"  Gallery: {len(gallery_embeds)} images embedded in {g_time:.1f}s")
        
        # Compute mAP and Rank-1
        print("  Computing mAP and Rank-1...")
        mAP, rank1 = compute_mAP_rank1(
            query_embeds, query_ids, query_cams,
            gallery_embeds, gallery_ids, gallery_cams
        )
        print(f"  mAP:    {mAP:.4f}")
        print(f"  Rank-1: {rank1:.4f}")
        
        # SEMI-SIMULATED OD
        print("  Running SEMI-SIMULATED OD scenario...")
        od_result = run_semi_simulated_od(
            emb_name,
            lambda imgs, md=model_dict, fn=embed_fn: fn(md, imgs),
            test_images_for_sim,
        )
        
        results[name] = {
            "mAP": mAP,
            "rank1": rank1,
            "embed_dim": query_embeds.shape[1],
            "query_time_s": q_time,
            "gallery_time_s": g_time,
        }
        
        if od_result:
            print(f"  SEMI-SIMULATED OD Results:")
            print(f"    Match Accuracy:       {od_result['accuracy']:.4f}")
            print(f"    Look-alike Accuracy:  {od_result['look_alike_accuracy']:.4f}")
            print(f"    Look-alike Pairs:     {od_result['look_alike_pairs']}")
            print(f"    OD MAE:               {od_result['od_mae']:.3f}")
            print(f"    OD Relative Error:    {od_result['od_relative_error']:.3f}")
            results[name]["semi_sim"] = od_result
        else:
            print("  SEMI-SIMULATED: insufficient data")
    
    # Summary table
    print(f"\n{'='*70}")
    print("EMBEDDER COMPARISON SUMMARY")
    print(f"{'='*70}")
    header = f"{'Embedder':<20} {'Dim':>5} {'mAP':>8} {'Rank-1':>8} {'OD Acc':>8} {'LookAlike':>10} {'OD MAE':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        od_acc = r.get("semi_sim", {}).get("accuracy", 0)
        la_acc = r.get("semi_sim", {}).get("look_alike_accuracy", 0)
        od_mae = r.get("semi_sim", {}).get("od_mae", 0)
        print(f"{name:<20} {r['embed_dim']:>5} {r['mAP']:>8.4f} {r['rank1']:>8.4f} {od_acc:>8.4f} {la_acc:>10.4f} {od_mae:>8.3f}")
    
    print(f"\nAll SEMI-SIMULATED results use real VeRi embeddings with synthetic travel times.")
    print(f"Label: SEMI-SIMULATED")
    
    # Save results
    import json
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/embedder_comparison.json", "w") as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to outputs/embedder_comparison.json")


if __name__ == "__main__":
    main()

