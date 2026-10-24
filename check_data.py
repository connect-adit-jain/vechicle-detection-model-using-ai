import os
import sys
import glob
import hashlib
import yaml
import cv2
import numpy as np

def compute_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_base_name(filename):
    # Extract base name before Roboflow hash '.rf.'
    name = os.path.splitext(filename)[0]
    if ".rf." in name:
        return name.split(".rf.")[0]
    return name

def parse_label_line(line):
    parts = line.strip().split()
    if not parts:
        return None
    cls_id = int(parts[0])
    coords = [float(p) for p in parts[1:]]
    if len(coords) == 4:
        xc, yc, w, h = coords
        return cls_id, xc, yc, w, h
    elif len(coords) >= 6:
        # Polygon / segmentation format: [x1, y1, x2, y2, ...]
        xs = coords[0::2]
        ys = coords[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        w = xmax - xmin
        h = ymax - ymin
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        return cls_id, xc, yc, w, h
    return None

def main():
    base_dir = os.path.abspath(os.path.join("data", "fastag"))
    print("=" * 75)
    print("CHECK_DATA.PY: FASTAG DATASET VERIFICATION")
    print(f"Dataset Root Directory: {base_dir}")
    print("=" * 75)

    # -------------------------------------------------------------
    # a. Read README.roboflow and README.dataset
    # -------------------------------------------------------------
    print("\n[A] DATASET METADATA (README.roboflow & README.dataset)")
    readme_roboflow_path = os.path.join(base_dir, "README.roboflow")
    if not os.path.exists(readme_roboflow_path):
        readme_roboflow_path = os.path.join(base_dir, "README.roboflow.txt")
        
    readme_dataset_path = os.path.join(base_dir, "README.dataset")
    if not os.path.exists(readme_dataset_path):
        readme_dataset_path = os.path.join(base_dir, "README.dataset.txt")

    version = "fastag - v5 (2025-05-13)"
    image_counts_meta = "207 images"
    preprocessing = "None documented"
    augmentation = "No image augmentation techniques were applied"
    source = "https://universe.roboflow.com/akash002/fastag"
    license_type = "CC BY 4.0"

    if os.path.exists(readme_roboflow_path):
        with open(readme_roboflow_path, "r", encoding="utf-8") as f:
            for line in f:
                l = line.strip()
                if "fastag - v" in l:
                    version = l
                elif "The dataset includes" in l:
                    image_counts_meta = l
                elif "pre-processing was applied" in l:
                    preprocessing = l
                elif "augmentation techniques were applied" in l:
                    augmentation = l
                elif "https://universe.roboflow.com" in l:
                    source = l

    if os.path.exists(readme_dataset_path):
        with open(readme_dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                l = line.strip()
                if l.startswith("https://universe.roboflow.com"):
                    source = l
                elif "License:" in l:
                    license_type = l.split("License:")[-1].strip()

    print(f"  • Dataset Version: {version}")
    print(f"  • Image Counts:    {image_counts_meta}")
    print(f"  • Preprocessing:   {preprocessing}")
    print(f"  • Augmentation:    {augmentation}")
    print(f"  • Source URL:      {source}")
    print(f"  • Licence:         {license_type}")

    # -------------------------------------------------------------
    # b. Read data.yaml and write fixed copy data_fixed.yaml
    # -------------------------------------------------------------
    print("\n[B] GENERATING data_fixed.yaml WITH ABSOLUTE PATHS")
    data_yaml_path = os.path.join(base_dir, "data.yaml")
    with open(data_yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    class_names = data_cfg.get("names", [])
    print(f"  • Class Names in data.yaml: {class_names}")

    train_images_abs = os.path.abspath(os.path.join(base_dir, "train", "images"))
    valid_images_abs = os.path.abspath(os.path.join(base_dir, "valid", "images"))
    test_images_abs = os.path.abspath(os.path.join(base_dir, "test", "images"))

    fixed_cfg = {
        "path": base_dir,
        "train": train_images_abs,
        "val": valid_images_abs,
        "test": test_images_abs,
        "nc": len(class_names),
        "names": class_names
    }

    data_fixed_path = os.path.join(base_dir, "data_fixed.yaml")
    with open(data_fixed_path, "w") as f:
        yaml.dump(fixed_cfg, f, default_flow_style=False)
    
    with open("data_fixed.yaml", "w") as f:
        yaml.dump(fixed_cfg, f, default_flow_style=False)

    print(f"  • Written data_fixed.yaml to: {data_fixed_path}")
    print(f"    - train: {train_images_abs}")
    print(f"    - val:   {valid_images_abs}")
    print(f"    - test:  {test_images_abs}")

    # -------------------------------------------------------------
    # c. Class counts, box counts, and sticker width distribution
    # -------------------------------------------------------------
    print("\n[C] DATASET STATISTICS & BOX DISTRIBUTION")
    splits = ["train", "valid", "test"]
    split_images = {}
    split_boxes = {s: {cn: 0 for cn in class_names} for s in splits}
    sticker_widths = []
    windshield_widths = []

    for s in splits:
        img_dir = os.path.join(base_dir, s, "images")
        lbl_dir = os.path.join(base_dir, s, "labels")

        images = glob.glob(os.path.join(img_dir, "*.*"))
        split_images[s] = images

        for img_path in images:
            base_fname = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base_fname + ".txt")
            if not os.path.exists(lbl_path):
                continue

            with open(lbl_path, "r") as lf:
                for line in lf:
                    parsed = parse_label_line(line)
                    if parsed:
                        cls_id, xc, yc, w_val, h_val = parsed
                        cname = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                        split_boxes[s][cname] = split_boxes[s].get(cname, 0) + 1

                        if cname.lower() in ["fastag-sticker", "fastag"]:
                            sticker_widths.append(w_val)
                        elif cname.lower() in ["windshield"]:
                            windshield_widths.append(w_val)

    print(f"  • Split Image Counts:")
    for s in splits:
        print(f"    - {s:<6}: {len(split_images[s])} images")
    print(f"    - Total : {sum(len(split_images[s]) for s in splits)} images")

    print(f"\n  • Box Counts per Class and Split:")
    print(f"    {'Class':<18} {'Train':<8} {'Valid':<8} {'Test':<8} {'Total':<8}")
    print("    " + "-" * 50)
    for cn in class_names:
        tr = split_boxes["train"].get(cn, 0)
        va = split_boxes["valid"].get(cn, 0)
        te = split_boxes["test"].get(cn, 0)
        tot = tr + va + te
        print(f"    {cn:<18} {tr:<8} {va:<8} {te:<8} {tot:<8}")

    if sticker_widths:
        min_w = float(np.min(sticker_widths))
        med_w = float(np.median(sticker_widths))
        max_w = float(np.max(sticker_widths))
        print(f"\n  • FASTag Sticker Width Distribution (as fraction of image width):")
        print(f"    - Min:    {min_w:.4f} ({min_w*100:.2f}% of image width)")
        print(f"    - Median: {med_w:.4f} ({med_w*100:.2f}% of image width)")
        print(f"    - Max:    {max_w:.4f} ({max_w*100:.2f}% of image width)")
    else:
        print("  [!] No sticker boxes found!")

    if windshield_widths:
        med_ww = float(np.median(windshield_widths))
        min_ww = float(np.min(windshield_widths))
        max_ww = float(np.max(windshield_widths))
        print(f"\n  • Windshield Width Distribution (as fraction of image width):")
        print(f"    - Min:    {min_ww:.4f} ({min_ww*100:.2f}%)")
        print(f"    - Median: {med_ww:.4f} ({med_ww*100:.2f}%)")
        print(f"    - Max:    {max_ww:.4f} ({max_ww*100:.2f}%)")

    # -------------------------------------------------------------
    # d. LEAKAGE CHECK
    # -------------------------------------------------------------
    print("\n[D] DATASET LEAKAGE CHECK (Roboflow Near-Duplicate Check)")
    hashes = {s: {} for s in splits}
    basenames = {s: {} for s in splits}

    for s in splits:
        for p in split_images[s]:
            fn = os.path.basename(p)
            bn = get_base_name(fn)
            h = compute_file_hash(p)
            hashes[s][h] = p
            basenames[s].setdefault(bn, []).append(p)

    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    total_hash_overlaps = 0
    total_name_overlaps = 0
    overlapping_names = []

    for s1, s2 in pairs:
        hash_common = set(hashes[s1].keys()) & set(hashes[s2].keys())
        total_hash_overlaps += len(hash_common)

        name_common = set(basenames[s1].keys()) & set(basenames[s2].keys())
        total_name_overlaps += len(name_common)
        if name_common:
            for n in name_common:
                overlapping_names.append((s1, s2, n))
            print(f"  [!] LEAKAGE DETECTED between {s1} and {s2}: {len(name_common)} base-image overlaps!")
            for n in list(name_common)[:5]:
                print(f"      Example overlap: '{n}' ({s1} vs {s2})")

    print(f"\n  • Total Exact File Hash Overlaps across splits: {total_hash_overlaps}")
    print(f"  • Total Base-Name (Augmentation Variant) Overlaps: {total_name_overlaps}")

    if total_name_overlaps > 0:
        print("\n  [CRITICAL WARNING] Roboflow split leakage detected!")
        print("  Copies/augmented variants of the same photo exist in multiple splits.")
        print("  -> Proposed Solution: Rebuild validation/test splits by grouping base names.")
    else:
        print("  [OK] No leakage detected! Splits appear strictly partitioned by base image.")

    # -------------------------------------------------------------
    # e. Save 12 sample images with boxes drawn
    # -------------------------------------------------------------
    print("\n[E] SAVING 12 SAMPLE VISUALIZATIONS WITH BOXES")
    out_dir = os.path.join(base_dir, "samples_12_drawn")
    os.makedirs(out_dir, exist_ok=True)

    sample_imgs = split_images["train"][:12] if len(split_images["train"]) >= 12 else split_images["train"]
    colors = {
        "Windshield": (255, 180, 0),    # Cyan / Blue
        "fastag-sticker": (0, 220, 0)   # Vivid Green
    }

    for idx, img_path in enumerate(sample_imgs, 1):
        img = cv2.imread(img_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]

        base_fname = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(base_dir, "train", "labels", base_fname + ".txt")

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as lf:
                for line in lf:
                    parsed = parse_label_line(line)
                    if parsed:
                        cls_id, xc, yc, bw, bh = parsed
                        cname = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                        x1 = int(round((xc - bw / 2) * iw))
                        y1 = int(round((yc - bh / 2) * ih))
                        x2 = int(round((xc + bw / 2) * iw))
                        y2 = int(round((yc + bh / 2) * ih))

                        color = colors.get(cname, (0, 0, 255))
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label_txt = f"{cname}"
                        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, max(th + 4, y1)), color, -1)
                        cv2.putText(img, label_txt, (x1 + 2, max(th + 2, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        out_file = os.path.join(out_dir, f"sample_{idx:02d}_{os.path.basename(img_path)}")
        cv2.imwrite(out_file, img)

    print(f"  • Successfully saved {len(sample_imgs)} annotated sample images to:")
    print(f"    {os.path.abspath(out_dir)}")

    # -------------------------------------------------------------
    # f. Warnings check
    # -------------------------------------------------------------
    print("\n[F] DATASET HEALTH & QUALITY WARNINGS")
    warnings = []
    train_count = len(split_images["train"])
    if train_count < 300:
        warnings.append(f"Training split has ONLY {train_count} images (fewer than the recommended 300 images).")

    tr_windshield = split_boxes["train"].get("Windshield", 0)
    tr_sticker = split_boxes["train"].get("fastag-sticker", 0)
    if tr_windshield == 0 or tr_sticker == 0:
        warnings.append("One or more classes have ZERO boxes in the training split!")
    elif tr_windshield / max(1, tr_sticker) > 4.0 or tr_sticker / max(1, tr_windshield) > 4.0:
        warnings.append(f"Class imbalance detected (Windshield: {tr_windshield}, FASTag: {tr_sticker}). Ratio exceeds 4:1.")
    
    # Check images missing Windshield
    no_windshield_count = train_count - tr_windshield
    if no_windshield_count > 0:
        warnings.append(f"{no_windshield_count} out of {train_count} training images do NOT have any 'Windshield' annotation (only FASTag sticker annotated).")

    if windshield_widths and np.median(windshield_widths) > 0.75:
        warnings.append(f"Median windshield width is {np.median(windshield_widths)*100:.1f}% of image width, and median sticker width is {np.median(sticker_widths)*100:.1f}% of image width. Most images are extreme macro/close-up crops of the sticker/glass, NOT whole vehicle crops like those extracted from CCTV traffic cameras.")

    if warnings:
        for w in warnings:
            print(f"  [!] WARNING: {w}")
    else:
        print("  [OK] No critical warnings detected.")

    print("\n" + "=" * 75)
    print("STEP 2 VERIFICATION COMPLETE. WAITING FOR USER APPROVAL.")
    print("=" * 75)

if __name__ == "__main__":
    main()
