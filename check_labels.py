import os
import re
from collections import Counter
import xml.etree.ElementTree as ET

def load_mapping(filepath):
    mapping = {}
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
    return mapping

def check_labels(data_dir="data/veri"):
    print("=" * 65)
    print("CHECK LABELS: VeRi-776 Vehicle Attribute (Colour & Type) Verification")
    print("=" * 65)

    # 1. Load documentation mappings
    color_map_path = os.path.join(data_dir, "list_color.txt")
    type_map_path = os.path.join(data_dir, "list_type.txt")

    color_names = load_mapping(color_map_path)
    type_names = load_mapping(type_map_path)

    print("\n--- Attribute Mappings from Documentation ---")
    if color_names:
        print(f"  Colour IDs (from list_color.txt): {color_names}")
    else:
        print("  WARNING: No colour ID mapping found in dataset documentation.")

    if type_names:
        print(f"  Type IDs (from list_type.txt): {type_names}")
    else:
        print("  WARNING: No type ID mapping found in dataset documentation.")

    # 2. Inspect XML files
    xml_files = ["train_label.xml", "test_label.xml"]
    
    for xml_name in xml_files:
        xml_path = os.path.join(data_dir, xml_name)
        if not os.path.exists(xml_path):
            print(f"\nMissing {xml_name}")
            continue

        with open(xml_path, "rb") as f:
            head = f.read(200).decode("ascii", errors="ignore")
            enc_match = re.search(r'encoding=["\']([^"\']+)["\']', head)
            encoding = enc_match.group(1) if enc_match else "unknown"

        print(f"\n--- {xml_name} (Encoding: {encoding}, Size: {os.path.getsize(xml_path):,} bytes) ---")

        # Parse using declared encoding with lxml
        try:
            from lxml import etree
            parser = etree.XMLParser(encoding=encoding, recover=True)
            tree = etree.parse(xml_path, parser=parser)
            root = tree.getroot()
        except ImportError:
            with open(xml_path, "r", encoding=encoding, errors="ignore") as f:
                content = f.read().replace(f'encoding="{encoding}"', 'encoding="utf-8"')
            root = ET.fromstring(content.encode("utf-8"))

        items = root.find("Items")

        total = len(items) if items is not None else 0
        print(f"  Total records: {total:,}")

        # Print first 3 records (strictly omitting any personal / plate annotations)
        print("  First 3 records (allowed attributes only):")
        allowed_keys = ["imageName", "vehicleID", "cameraID", "colorID", "typeID"]
        for i, item in enumerate(items[:3] if items is not None else []):
            rec = {k: item.attrib.get(k) for k in allowed_keys if k in item.attrib}
            rec["colorName"] = color_names.get(rec.get("colorID"), "unknown")
            rec["typeName"] = type_names.get(rec.get("typeID"), "unknown")
            print(f"    Record {i+1}: {rec}")

        # Distribution per colour class and per type class
        color_counts = Counter()
        type_counts = Counter()

        if items is not None:
            for item in items:
                cid = item.attrib.get("colorID")
                tid = item.attrib.get("typeID")
                color_counts[cid] += 1
                type_counts[tid] += 1

        print(f"\n  Image Distribution by Colour ({xml_name}):")
        for cid in sorted(color_counts.keys(), key=lambda x: int(x) if x and x.isdigit() else 999):
            cname = color_names.get(cid, "unmapped")
            print(f"    ID {cid:>2} ({cname:<8}): {color_counts[cid]:>6} images")

        print(f"\n  Image Distribution by Vehicle Type ({xml_name}):")
        for tid in sorted(type_counts.keys(), key=lambda x: int(x) if x and x.isdigit() else 999):
            tname = type_names.get(tid, "unmapped")
            print(f"    ID {tid:>2} ({tname:<10}): {type_counts[tid]:>6} images")

if __name__ == "__main__":
    check_labels()
