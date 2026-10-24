import json
import os
from collections import Counter

def inspect_json(filepath):
    print(f"\n==================================================")
    print(f"Inspecting: {filepath}")
    print(f"==================================================")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Keys in JSON: {list(data.keys())}")
    
    # Categories
    categories = data.get("categories", [])
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    print(f"\n--- Categories ({len(categories)}) ---")
    for c in categories:
        print(f"  ID {c['id']}: '{c['name']}' (supercategory: {c.get('supercategory', 'None')})")
        
    # Images
    images = data.get("images", [])
    print(f"\nTotal Images: {len(images)}")
    if images:
        print(f"Sample Image record: {images[0]}")
        
    # Annotations
    annotations = data.get("annotations", [])
    print(f"Total Annotations (Boxes): {len(annotations)}")
    if annotations:
        print(f"Sample Annotation record: {annotations[0]}")
        
    # Counts per category
    box_counts = Counter()
    image_per_cat = defaultdict_images = {}
    for c_id in cat_id_to_name:
        defaultdict_images[c_id] = set()
        
    for ann in annotations:
        cid = ann["category_id"]
        box_counts[cid] += 1
        defaultdict_images[cid].add(ann["image_id"])
        
    print("\n--- Class Statistics ---")
    print(f"{'Class ID':<10} | {'Class Name':<20} | {'Box Count':>12} | {'Image Count':>12}")
    print("-" * 62)
    for cid in sorted(cat_id_to_name.keys()):
        name = cat_id_to_name[cid]
        b_cnt = box_counts[cid]
        img_cnt = len(defaultdict_images[cid])
        print(f"{cid:<10} | {name:<20} | {b_cnt:>12} | {img_cnt:>12}")

def main():
    json_paths = [
        r"data\uvh26\UVH-26-Train\UVH-26-MV-Train.json",
        r"data\uvh26\UVH-26-Train\UVH-26-ST-Train.json",
        r"data\uvh26\UVH-26-Val\UVH-26-MV-Val.json",
        r"data\uvh26\UVH-26-Val\UVH-26-ST-Val.json"
    ]
    for p in json_paths:
        if os.path.exists(p):
            inspect_json(p)
        else:
            print(f"File not found: {p}")

if __name__ == "__main__":
    main()
