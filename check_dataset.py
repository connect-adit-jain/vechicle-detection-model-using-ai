import os
import re
import xml.etree.ElementTree as ET

def check_dataset(data_dir="data/veri"):
    print("=" * 60)
    print("CHECK DATASET: VeRi-776 Vehicle Re-ID Split & ID Verification")
    print("=" * 60)

    splits = ["image_train", "image_query", "image_test"]
    split_counts = {}
    vehicle_ids = set()
    camera_ids = set()
    sample_files = []

    # Filename format: <vehicleID>_<cameraID>_<frameID>_<trackID>.jpg
    fname_pattern = re.compile(r"^(\d+)_c(\d+)_")

    for split in splits:
        split_path = os.path.join(data_dir, split)
        if os.path.isdir(split_path):
            files = [f for f in os.listdir(split_path) if f.lower().endswith(".jpg")]
            split_counts[split] = len(files)
            if not sample_files and files:
                sample_files = files[:5]
            for f in files:
                m = fname_pattern.match(f)
                if m:
                    vehicle_ids.add(m.group(1))
                    camera_ids.add(f"c{m.group(2)}")
        else:
            split_counts[split] = 0

    print("\n--- Image Counts per Split ---")
    for split, count in split_counts.items():
        print(f"  {split}: {count:,} images")

    print("\n--- Unique IDs Extracted from Filenames ---")
    print(f"  Unique Vehicle IDs across splits: {len(vehicle_ids)}")
    print(f"  Unique Camera IDs across splits: {len(camera_ids)} ({sorted(list(camera_ids))})")

    print("\n--- 5 Sample Filenames ---")
    for s in sample_files:
        print(f"  {s}")

    print("\n--- XML Label Files & Encodings ---")
    xml_files = [f for f in os.listdir(data_dir) if f.endswith(".xml")]
    for xml_name in xml_files:
        xml_path = os.path.join(data_dir, xml_name)
        with open(xml_path, "rb") as f:
            head = f.read(200).decode("ascii", errors="ignore")
            # Parse encoding from <?xml ... encoding="..." ?>
            enc_match = re.search(r'encoding=["\']([^"\']+)["\']', head)
            encoding = enc_match.group(1) if enc_match else "unknown"
            print(f"  {xml_name}: declared encoding = {encoding}, size = {os.path.getsize(xml_path):,} bytes")

    # Successfully parse train_label.xml using lxml / gb2312
    if "train_label.xml" in xml_files:
        sample_xml = os.path.join(data_dir, "train_label.xml")
        try:
            from lxml import etree
            parser = etree.XMLParser(encoding="gb2312", recover=True)
            tree = etree.parse(sample_xml, parser=parser)
            root = tree.getroot()
        except ImportError:
            with open(sample_xml, "r", encoding="gb2312", errors="ignore") as f:
                content = f.read().replace('encoding="gb2312"', 'encoding="utf-8"')
            root = ET.fromstring(content.encode("utf-8"))

        items = root.find("Items")
        total_items = len(items) if items is not None else 0
        print(f"\n--- Successfully Parsed {sample_xml} ---")
        print(f"  Root tag: <{root.tag}>, Total <Item> entries: {total_items:,}")
        if items is not None and len(items) > 0:
            first = dict(items[0].attrib)
            # Strictly exclude license-plate attributes
            filtered_attrib = {k: v for k, v in first.items() if "plate" not in k.lower()}
            print(f"  Sample Item (first record, plate fields omitted): {filtered_attrib}")

if __name__ == "__main__":
    check_dataset()

