import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
import shutil

MAX_ALLOWED_GB = 9.5 # Strict ceiling under the 10GB limit

def get_folder_size_gb(folder_path):
    if not os.path.exists(folder_path):
        return 0.0
    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, d, fs in os.walk(folder_path)
        for f in fs
    )
    return total / (1024 ** 3)

def get_free_disk_gb(path="C:\\"):
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)

def download_batch(target_gb=2.0, split="Val", max_workers=12):
    print("=" * 65)
    print("INDIAN TRAFFIC VEHICLE DATASET DOWNLOADER (UVH-26)")
    print(f"Dataset: iisc-aim/UVH-26 | Target: {target_gb:.2f} GB (Strict Cap: {MAX_ALLOWED_GB} GB)")
    print("=" * 65)

    if target_gb > MAX_ALLOWED_GB:
        print(f"[ERROR] Requested {target_gb} GB exceeds strict ceiling of {MAX_ALLOWED_GB} GB.")
        target_gb = MAX_ALLOWED_GB

    free_disk = get_free_disk_gb()
    print(f"[*] Current Free Disk: {free_disk:.2f} GB")
    if free_disk < 10.0:
        print("[ERROR] Free disk is below 10 GB. Aborting download for safety.")
        return

    dest_dir = os.path.join("data", "uvh26")
    current_size_gb = get_folder_size_gb(dest_dir)
    print(f"[*] Current 'data/uvh26' size: {current_size_gb:.2f} GB")

    remaining_budget_gb = target_gb - current_size_gb
    if remaining_budget_gb <= 0:
        print(f"[!] Already at or above target {target_gb:.2f} GB. Current size: {current_size_gb:.2f} GB.")
        return

    print(f"[*] Budget remaining to download: {remaining_budget_gb:.2f} GB")
    print("[*] Fetching file index from Hugging Face hub...")

    api = HfApi()
    info = api.dataset_info("iisc-aim/UVH-26", files_metadata=True)

    prefix = f"UVH-26-{split}/data/"
    candidate_files = [
        s for s in info.siblings
        if s.rfilename.startswith(prefix) and s.rfilename.endswith(".png")
    ]
    print(f"[*] Found {len(candidate_files)} total available image files in split '{split}'.")

    # Filter out already downloaded files
    to_download = []
    accumulated_bytes = 0
    max_bytes = remaining_budget_gb * (1024 ** 3)

    for s in candidate_files:
        local_path = os.path.join(dest_dir, os.path.normpath(s.rfilename))
        if os.path.exists(local_path):
            continue
        to_download.append(s.rfilename)
        accumulated_bytes += (s.size or 3.3 * 1024 * 1024)
        if accumulated_bytes >= max_bytes:
            break

    total_files = len(to_download)
    est_gb = accumulated_bytes / (1024 ** 3)
    print(f"[*] Queued {total_files} new images (~{est_gb:.2f} GB) to download with {max_workers} worker threads.")
    print("=" * 65)

    downloaded = 0
    failed = 0
    start_time = time.time()

    def fetch_one(fname):
        try:
            hf_hub_download(
                repo_id="iisc-aim/UVH-26",
                filename=fname,
                repo_type="dataset",
                local_dir=dest_dir
            )
            return True, fname
        except Exception as e:
            return False, f"{fname}: {e}"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, fn): fn for fn in to_download}
        for future in as_completed(futures):
            success, msg = future.result()
            if success:
                downloaded += 1
            else:
                failed += 1
            if (downloaded + failed) % 25 == 0 or (downloaded + failed) == total_files:
                elapsed = time.time() - start_time
                rate = (downloaded * 3.29) / max(elapsed, 0.1) # approx MB/s
                pct = ((downloaded + failed) / total_files) * 100
                print(f"  Progress: {downloaded + failed}/{total_files} ({pct:.1f}%) | {rate:.2f} MB/s | Elapsed: {elapsed:.0f}s", flush=True)

    end_size_gb = get_folder_size_gb(dest_dir)
    final_free_disk = get_free_disk_gb()
    print("=" * 65)
    print("DOWNLOAD BATCH FINISHED")
    print(f"  Successfully downloaded: {downloaded} images (Failed: {failed})")
    print(f"  New data/uvh26 folder size: {end_size_gb:.2f} GB")
    print(f"  Remaining Free Disk: {final_free_disk:.2f} GB")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download UVH-26 vehicle dataset images up to max 10 GB limit.")
    parser.add_argument("--gb", type=float, default=2.0, help="Target total folder size in GB (max 9.5 GB)")
    parser.add_argument("--split", type=str, default="Val", choices=["Val", "Train"], help="Dataset split to download from")
    parser.add_argument("--workers", type=int, default=12, help="Number of concurrent download threads")
    args = parser.parse_args()

    download_batch(target_gb=args.gb, split=args.split, max_workers=args.workers)
