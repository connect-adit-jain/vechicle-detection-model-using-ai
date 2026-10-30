import os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download

def main():
    api = HfApi()
    print("Fetching repository file list...")
    info = api.dataset_info("iisc-aim/UVH-26", files_metadata=True)
    val_files = [
        s.rfilename for s in info.siblings
        if s.rfilename.startswith("UVH-26-Val/data/000/") and s.rfilename.endswith(".png")
    ][:30]

    print(f"Downloading {len(val_files)} sample validation images into data/uvh26/...")
    
    def download_file(fname):
        return hf_hub_download(
            repo_id="iisc-aim/UVH-26",
            filename=fname,
            repo_type="dataset",
            local_dir="data/uvh26"
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(download_file, val_files))

    print(f"Successfully downloaded {len(results)} sample images!")

if __name__ == "__main__":
    main()

