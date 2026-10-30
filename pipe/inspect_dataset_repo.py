import shutil
from huggingface_hub import HfApi

def main():
    api = HfApi()
    print("Fetching dataset info for iisc-aim/UVH-26...")
    try:
        info = api.dataset_info("iisc-aim/UVH-26", files_metadata=True)
    except Exception as e:
        print(f"Error fetching dataset info: {e}")
        return
        
    print(f"\nDataset ID: {info.id}")
    print(f"Author: {info.author}")
    print(f"Description/Card: {info.description if hasattr(info, 'description') else 'N/A'}")
    
    total_size = 0
    file_list = []
    
    if hasattr(info, "siblings") and info.siblings:
        for s in info.siblings:
            rfilename = s.rfilename
            size = getattr(s, "size", None)
            if size is None and hasattr(s, "lfs") and s.lfs:
                size = s.lfs.get("size", 0)
            if size is None:
                size = 0
            total_size += size
            file_list.append((rfilename, size))
            
    print(f"\n--- Total Files: {len(file_list)} ---")
    print(f"{'Filename':<55} | {'Size (MB)':>12}")
    print("-" * 70)
    for fname, size in sorted(file_list):
        print(f"{fname:<55} | {size / (1024*1024):>12.2f}")
        
    total_gb = total_size / (1024**3)
    print("-" * 70)
    print(f"Total Dataset Size: {total_gb:.2f} GB ({total_size} bytes)")
    
    # Check free disk space
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    threshold_gb = free_gb - 10
    print(f"\nDisk Space Check:")
    print(f" - Free disk: {free_gb:.2f} GB")
    print(f" - Safety threshold (Free - 10 GB): {threshold_gb:.2f} GB")
    print(f" - Dataset size: {total_gb:.2f} GB")
    
    if total_gb > threshold_gb:
        print("\n[WARNING] Dataset size EXCEEDS safety threshold!")
    else:
        print("\n[OK] Dataset size is well within disk capacity.")

if __name__ == "__main__":
    main()

