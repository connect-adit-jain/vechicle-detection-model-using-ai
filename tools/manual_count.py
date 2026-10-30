"""
tools/manual_count.py
Virtual line-crossing manual count tool for tracker validation.

Usage:
  python tools/manual_count.py --video data/videos/cam1.mp4 --tracks tracks.json

1. Opens the first frame of the video.
2. User clicks two points to draw a virtual counting line.
3. User enters the ground-truth count of vehicles per class that crossed the line.
4. Script reads tracks.json, counts unique track IDs that cross the line, 
   and reports the ratio (predicted / true) and the line-crossing count error per tracker.
"""
import argparse
import json
import os
import sys
import cv2
import numpy as np

line_points = []

def mouse_callback(event, x, y, flags, param):
    global line_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
            print(f"  Point {len(line_points)}: ({x}, {y})")

def line_crosses(p1, p2, a, b):
    """Check if segment p1-p2 crosses line segment a-b."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(a, b, p1)
    d2 = cross(a, b, p2)
    d3 = cross(p1, p2, a)
    d4 = cross(p1, p2, b)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def main():
    global line_points
    parser = argparse.ArgumentParser(description="Manual count validation for tracker")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--tracks", required=True, help="Path to tracks.json")
    parser.add_argument("--output", default="outputs/manual_count_result.json", help="Output file")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.tracks):
        print(f"ERROR: Tracks file not found: {args.tracks}")
        sys.exit(1)

    # Open video and show first frame
    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Cannot read video frame")
        sys.exit(1)

    print("\n=== MANUAL COUNT TOOL ===")
    print("Click TWO points on the frame to define a counting line.")
    print("Press 'q' to quit, 'r' to reset points.\n")

    cv2.namedWindow("Define Counting Line", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Define Counting Line", mouse_callback)

    while True:
        display = frame.copy()
        for pt in line_points:
            cv2.circle(display, pt, 5, (0, 255, 0), -1)
        if len(line_points) == 2:
            cv2.line(display, line_points[0], line_points[1], (0, 255, 0), 2)
            cv2.putText(display, "Line defined! Press ENTER to confirm, 'r' to reset",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Click point {len(line_points)+1}/2",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Define Counting Line", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord('r'):
            line_points = []
            print("  Reset points.")
        elif key == 13 and len(line_points) == 2:  # Enter
            break

    cv2.destroyAllWindows()

    line_a, line_b = line_points
    print(f"\nCounting line: {line_a} -> {line_b}")

    # Get ground truth counts from user
    classes = ["car", "two_wheeler", "auto_rickshaw", "bus", "truck"]
    gt_counts = {}
    print("\nEnter the ground-truth count for each class (vehicles that crossed this line):")
    for cls in classes:
        while True:
            try:
                count = int(input(f"  {cls}: "))
                gt_counts[cls] = count
                break
            except ValueError:
                print("  Please enter an integer.")

    total_gt = sum(gt_counts.values())
    print(f"\nTotal ground truth: {total_gt}")

    # Load tracks and count crossings
    with open(args.tracks) as f:
        tracks = json.load(f)

    crossing_ids = set()
    crossing_by_class = {}
    for cls in classes:
        crossing_by_class[cls] = set()

    # For each track, check if first/last box centers cross the line
    # (simplified: check consecutive frame positions)
    for track in tracks:
        tid = track.get("track_id")
        cls = track.get("class", "unknown")
        # If track has box trajectory, check line crossing
        # For now use entry/exit positions if available
        entry = track.get("entry_zone_pos")
        exit_pos = track.get("exit_zone_pos")
        if entry and exit_pos:
            if line_crosses(tuple(entry), tuple(exit_pos), line_a, line_b):
                crossing_ids.add(tid)
                if cls in crossing_by_class:
                    crossing_by_class[cls].add(tid)

    # Report
    print(f"\n{'='*50}")
    print("MANUAL COUNT CHECK RESULTS")
    print(f"{'='*50}")
    print(f"{'Class':<20} {'GT':>5} {'Pred':>5} {'Ratio':>8} {'Error':>8}")
    print("-" * 50)
    total_pred = 0
    results = {}
    for cls in classes:
        pred = len(crossing_by_class[cls])
        gt = gt_counts[cls]
        total_pred += pred
        ratio = pred / gt if gt > 0 else float('inf') if pred > 0 else 1.0
        error = pred - gt
        print(f"{cls:<20} {gt:>5} {pred:>5} {ratio:>8.3f} {error:>+8}")
        results[cls] = {"gt": gt, "pred": pred, "ratio": ratio, "error": error}

    total_ratio = total_pred / total_gt if total_gt > 0 else float('inf')
    print(f"{'TOTAL':<20} {total_gt:>5} {total_pred:>5} {total_ratio:>8.3f} {total_pred - total_gt:>+8}")
    results["total"] = {"gt": total_gt, "pred": total_pred, "ratio": total_ratio, "error": total_pred - total_gt}

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        "line": [list(line_a), list(line_b)],
        "ground_truth": gt_counts,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

