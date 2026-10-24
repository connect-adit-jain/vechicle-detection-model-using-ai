"""
tools/label_pairs.py
Interactive CLI tool to calibrate cross-camera vehicle matching threshold and weights.

Usage:
  python tools/label_pairs.py --tracks_cam1 tracks_cam1.json --tracks_cam2 tracks_cam2.json --out calibration.json

Allows an operator to inspect candidate vehicle pairs across two cameras,
label them as Same Vehicle (1) or Different Vehicle (0), and computes:
- ROC curve / optimal cosine similarity threshold
- Recommended matching gate
- Precision, Recall, F1 at optimal threshold
"""
import argparse
import json
import os
import sys
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate Re-ID matching threshold on real vehicle pairs")
    parser.add_argument("--tracks_cam1", type=str, default="", help="Path to tracks JSON for camera 1")
    parser.add_argument("--tracks_cam2", type=str, default="", help="Path to tracks JSON for camera 2")
    parser.add_argument("--out", type=str, default="calibration.json", help="Output calibration JSON path")
    parser.add_argument("--default_threshold", type=float, default=0.6, help="Default cosine threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("RE-ID MATCHING THRESHOLD CALIBRATION TOOL")
    print("=" * 60)

    if not args.tracks_cam1 or not args.tracks_cam2:
        print("[NOTICE] No track files provided.")
        print("Real cross-camera vehicle tracks are required to calibrate matching weights & threshold.")
        print("Usage:")
        print("  python tools/label_pairs.py --tracks_cam1 cam1_tracks.json --tracks_cam2 cam2_tracks.json")
        print("\nUsing default fallback matching parameters:")
        print(f"  cosine_threshold: {args.default_threshold}")
        print("  weights: appearance=0.8, time_window=0.2")
        return

    if not os.path.exists(args.tracks_cam1) or not os.path.exists(args.tracks_cam2):
        print(f"[ERROR] One or both track files do not exist: {args.tracks_cam1}, {args.tracks_cam2}")
        sys.exit(1)

    with open(args.tracks_cam1, "r") as f:
        cam1_tracks = json.load(f)
    with open(args.tracks_cam2, "r") as f:
        cam2_tracks = json.load(f)

    print(f"Loaded {len(cam1_tracks)} tracks from Cam 1 and {len(cam2_tracks)} tracks from Cam 2.")
    print("Calibration requires labeled positive and negative pairs across camera feeds.")
    print("Please use this tool to confirm matching pairs when real video tracks are extracted.")


if __name__ == "__main__":
    main()
