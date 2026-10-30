from huggingface_hub import HfApi
from collections import defaultdict

def main():
    api = HfApi()
    info = api.dataset_info("iisc-aim/UVH-26", files_metadata=True)
    
    breakdown = defaultdict(lambda: {"count": 0, "size": 0})
    json_files = []
    
    for s in info.siblings:
        fname = s.rfilename
        size = getattr(s, "size", None)
        if size is None and hasattr(s, "lfs") and s.lfs:
            size = s.lfs.get("size", 0)
        size = size or 0
        
        parts = fname.split("/")
        top_folder = parts[0]
        breakdown[top_folder]["count"] += 1
        breakdown[top_folder]["size"] += size
        
        if fname.endswith(".json"):
            json_files.append((fname, size))
            
    print(f"{'Folder / Category':<30} | {'Files':>8} | {'Total Size (GB)':>15}")
    print("-" * 60)
    for folder, data in sorted(breakdown.items()):
        print(f"{folder:<30} | {data['count']:>8} | {data['size'] / (1024**3):>15.2f}")
        
    print("\n--- JSON Annotation Files ---")
    for jname, jsize in sorted(json_files):
        print(f" - {jname:<40} : {jsize / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()

