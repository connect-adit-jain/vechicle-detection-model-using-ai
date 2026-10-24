import os
import glob
import json
from collections import Counter, defaultdict
import cv2
import matplotlib.pyplot as plt

def analyze_coco_json(filepath):
    print(f"\n=======================================================")
    print(f"COCO Dataset Analysis: {os.path.basename(filepath)}")
    print(f"Path: {filepath}")
    print(f"=======================================================")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    
    box_counts = Counter()
    cat_to_images = defaultdict(set)
    
    for ann in annotations:
        cid = ann["category_id"]
        box_counts[cid] += 1
        cat_to_images[cid].add(ann["image_id"])
        
    print(f"Total Images: {len(images):,}")
    print(f"Total Bounding Boxes: {len(annotations):,}")
    print(f"Total Categories: {len(categories)}")
    print("\nReal Class Names & Statistics:")
    print(f"{'ID':<4} | {'Class Name':<20} | {'Box Count':>12} | {'Image Count':>12}")
    print("-" * 55)
    for cid in sorted(categories.keys()):
        name = categories[cid]
        print(f"{cid:<4} | {name:<20} | {box_counts[cid]:>12,} | {len(cat_to_images[cid]):>12,}")
        
    return data, categories

def draw_sample_images(json_path, output_dir="sample_visualizations", max_samples=6):
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    
    # Map image id to annotations
    img_to_anns = defaultdict(list)
    for ann in data.get("annotations", []):
        img_to_anns[ann["image_id"]].append(ann)
        
    # Search for local image files on disk
    print("\nScanning for downloaded image files...")
    local_images = {}
    for p in glob.glob(r"data\uvh26\**\*.png", recursive=True):
        local_images[os.path.basename(p)] = p
        
    print(f"Found {len(local_images)} local image files on disk.")
    
    # Match images that exist on disk and have annotations
    valid_images = []
    for img in data.get("images", []):
        fname = img["file_name"]
        if fname in local_images and len(img_to_anns[img["id"]]) > 0:
            valid_images.append((img, local_images[fname]))
            
    print(f"Found {len(valid_images)} annotated images available locally.")
    if not valid_images:
        print("[WARNING] No local images matched the JSON yet.")
        return []
        
    # Pick diverse sample images
    selected = valid_images[:max_samples]
    saved_images = []
    
    # Palette of colors for classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
        (0, 128, 128), (128, 128, 0), (0, 128, 0), (128, 0, 0),
        (75, 0, 130), (255, 192, 203)
    ]
    
    plt.figure(figsize=(18, 12))
    
    for idx, (img_info, path) in enumerate(selected):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w, _ = img.shape
        anns = img_to_anns[img_info["id"]]
        
        annotated = img.copy()
        for ann in anns:
            cid = ann["category_id"]
            cname = categories.get(cid, str(cid))
            bbox = ann["bbox"] # [x, y, width, height]
            x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            color = colors[(cid - 1) % len(colors)]
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 3)
            
            label = f"{cname}"
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, max(0, y - th - 6)), (x + tw + 4, y), color, -1)
            cv2.putText(annotated, label, (x + 2, max(th, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        out_path = os.path.join(output_dir, f"sample_{idx+1}_{img_info['file_name']}")
        cv2.imwrite(out_path, annotated)
        saved_images.append(out_path)
        print(f"Saved sample #{idx+1}: {out_path} ({len(anns)} boxes)")
        
        # Add to subplot collage
        plt.subplot(2, 3, idx + 1)
        rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.title(f"Sample {idx+1}: {img_info['file_name']} ({len(anns)} boxes)", fontsize=10)
        plt.axis("off")
        
    collage_path = os.path.join(output_dir, "collage_6_samples.png")
    plt.tight_layout()
    plt.savefig(collage_path, dpi=150)
    plt.close()
    print(f"\nSaved 6-sample collage to: {collage_path}")
    return saved_images

def main():
    json_files = [
        r"data\uvh26\UVH-26-Val\UVH-26-MV-Val.json",
        r"data\uvh26\UVH-26-Val\UVH-26-ST-Val.json",
        r"data\uvh26\UVH-26-Train\UVH-26-MV-Train.json",
        r"data\uvh26\UVH-26-Train\UVH-26-ST-Train.json",
    ]
    
    for jf in json_files:
        if os.path.exists(jf):
            analyze_coco_json(jf)
            
    # Draw samples using the Val split
    val_json = r"data\uvh26\UVH-26-Val\UVH-26-MV-Val.json"
    if os.path.exists(val_json):
        draw_sample_images(val_json, max_samples=6)

if __name__ == "__main__":
    main()
