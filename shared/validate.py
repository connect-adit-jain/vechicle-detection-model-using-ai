"""
shared/validate.py
Schema validation for tracks.json, matches.csv, and all pipeline output files.
"""
import json
import csv
import os
import sys
import numpy as np

VALID_CLASSES = {"car", "two_wheeler", "auto_rickshaw", "bus", "truck"}

TRACK_REQUIRED_FIELDS = {
    "camera_id", "track_id", "class", "t_start", "t_end",
    "entry_zone", "exit_zone", "mean_speed_px_per_s", "quality",
    "n_crops", "colour", "colour_conf", "type", "type_conf",
    "fastag_present", "fastag_cell", "fastag_conf",
    "embedder", "attr_model", "detector", "embedding"
}


def validate_track(track, idx=0):
    """Validate a single track object. Returns list of error strings."""
    errors = []
    # Check required fields
    for field in TRACK_REQUIRED_FIELDS:
        if field not in track:
            errors.append(f"Track {idx}: missing field '{field}'")

    # Type checks
    if "class" in track and track["class"] not in VALID_CLASSES:
        errors.append(f"Track {idx}: invalid class '{track['class']}' (valid: {VALID_CLASSES})")

    if "t_start" in track and "t_end" in track:
        if not isinstance(track["t_start"], (int, float)):
            errors.append(f"Track {idx}: t_start must be a number")
        if not isinstance(track["t_end"], (int, float)):
            errors.append(f"Track {idx}: t_end must be a number")
        elif track["t_start"] > track["t_end"]:
            errors.append(f"Track {idx}: t_start > t_end")

    if "embedding" in track:
        emb = track["embedding"]
        if not isinstance(emb, list) or len(emb) == 0:
            errors.append(f"Track {idx}: embedding must be a non-empty list of floats")
        else:
            # Check L2 normalisation (tolerance 0.05)
            norm = np.linalg.norm(emb)
            if abs(norm - 1.0) > 0.05:
                errors.append(f"Track {idx}: embedding not L2-normalised (norm={norm:.4f})")

    if "fastag_present" in track:
        val = track["fastag_present"]
        if val not in (True, False, None):
            errors.append(f"Track {idx}: fastag_present must be true, false, or null")

    if "fastag_cell" in track:
        val = track["fastag_cell"]
        if val is not None:
            if not (isinstance(val, list) and len(val) == 2):
                errors.append(f"Track {idx}: fastag_cell must be [row, col] or null")

    return errors


def validate_tracks_json(filepath):
    """Validate a tracks.json file. Returns (valid: bool, errors: list)."""
    errors = []
    try:
        with open(filepath) as f:
            tracks = json.load(f)
    except Exception as e:
        return False, [f"Cannot read {filepath}: {e}"]

    if not isinstance(tracks, list):
        return False, ["tracks.json must be a JSON array"]

    for i, track in enumerate(tracks):
        errors.extend(validate_track(track, i))

    return len(errors) == 0, errors


def validate_matches_csv(filepath):
    """Validate matches.csv. Returns (valid: bool, errors: list)."""
    errors = []
    required_cols = {"src_camera", "src_track", "dst_camera", "dst_track", "score", "margin", "is_dummy"}
    try:
        with open(filepath) as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
            missing = required_cols - cols
            if missing:
                errors.append(f"Missing columns: {missing}")
            for i, row in enumerate(reader):
                if "score" in row:
                    try:
                        float(row["score"])
                    except ValueError:
                        errors.append(f"Row {i}: score not a number")
    except Exception as e:
        return False, [f"Cannot read {filepath}: {e}"]

    return len(errors) == 0, errors


def validate_od_csv(filepath):
    """Validate an OD matrix CSV."""
    errors = []
    try:
        with open(filepath) as f:
            reader = csv.reader(f)
            header = next(reader)
            for i, row in enumerate(reader):
                for j, val in enumerate(row):
                    try:
                        float(val)
                    except ValueError:
                        if j > 0:  # skip row label
                            errors.append(f"Row {i}, Col {j}: not a number")
    except Exception as e:
        return False, [f"Cannot read {filepath}: {e}"]

    return len(errors) == 0, errors


def run_full_validation(output_dir="outputs"):
    """Validate all output files and print a report."""
    print("=" * 60)
    print("SCHEMA VALIDATION")
    print("=" * 60)

    checks = []

    # tracks.json
    tracks_path = os.path.join(output_dir, "tracks.json")
    if os.path.exists(tracks_path):
        valid, errs = validate_tracks_json(tracks_path)
        checks.append(("tracks.json", valid, errs))
    else:
        checks.append(("tracks.json", None, ["File not found"]))

    # matches.csv
    matches_path = os.path.join(output_dir, "matches.csv")
    if os.path.exists(matches_path):
        valid, errs = validate_matches_csv(matches_path)
        checks.append(("matches.csv", valid, errs))
    else:
        checks.append(("matches.csv", None, ["File not found"]))

    # OD CSVs
    for pattern in ["od_all.csv", "od_all_hard.csv"] + [f"od_{c}.csv" for c in VALID_CLASSES]:
        fpath = os.path.join(output_dir, pattern)
        if os.path.exists(fpath):
            valid, errs = validate_od_csv(fpath)
            checks.append((pattern, valid, errs))

    for name, valid, errs in checks:
        if valid is None:
            status = "SKIPPED"
        elif valid:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  [{status}] {name}")
        if errs and not valid:
            for e in errs[:5]:
                print(f"         {e}")
            if len(errs) > 5:
                print(f"         ... and {len(errs)-5} more errors")

    return checks


if __name__ == "__main__":
    run_full_validation()

