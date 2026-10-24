import os
import shutil
from huggingface_hub import HfApi, hf_hub_download
from ultralytics import YOLO

def main():
    repo_id = "iisc-aim/UVH-26"
    api = HfApi()
    
    print("=" * 60)
    print(f"Fetching Model Info for Hugging Face Model: {repo_id}")
    print("=" * 60)
    
    try:
        model_info = api.model_info(repo_id, files_metadata=True)
    except Exception as e:
        print(f"Error fetching model info: {e}")
        return
        
    print(f"Model ID: {model_info.id}")
    print(f"Author: {getattr(model_info, 'author', 'N/A')}")
    print(f"Tags: {getattr(model_info, 'tags', [])}")
    
    # Read files list and sizes
    print("\n--- Files in Model Repository ---")
    print(f"{'Filename':<45} | {'Size (MB)':>12}")
    print("-" * 60)
    files = []
    for s in model_info.siblings:
        size = getattr(s, "size", None)
        if size is None and hasattr(s, "lfs") and s.lfs:
            size = s.lfs.get("size", 0)
        size = size or 0
        files.append((s.rfilename, size))
        print(f"{s.rfilename:<45} | {size / (1024*1024):>12.2f}")
        
    # Read model card (README.md) and license
    os.makedirs("models_download", exist_ok=True)
    readme_path = None
    try:
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", local_dir="models_download")
        print("\n--- Model Card (README.md) Content ---")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            # print first 50 lines or key sections
            print(content[:3000])
    except Exception as e:
        print(f"Error downloading README.md: {e}")
        
    # Identify YOLO11-S weights
    yolo11s_file = None
    for fname, size in files:
        fname_lower = fname.lower()
        if "yolo11" in fname_lower and "s" in fname_lower and (fname.endswith(".pt") or fname.endswith(".engine") or fname.endswith(".onnx")):
            yolo11s_file = fname
            break
            
    if not yolo11s_file:
        # Check any .pt files
        pt_files = [f for f, _ in files if f.endswith(".pt")]
        print(f"\nExact YOLO11-S name pattern not matched immediately. Available .pt files: {pt_files}")
        for pf in pt_files:
            if "11s" in pf.lower() or ("11" in pf.lower() and "s" in pf.lower()):
                yolo11s_file = pf
                break
        if not yolo11s_file and pt_files:
            yolo11s_file = pt_files[0]
            
    print(f"\nTarget YOLO11-S weights file: {yolo11s_file}")
    
    if yolo11s_file:
        print(f"Downloading {yolo11s_file} into models_download/...")
        weights_local_path = hf_hub_download(
            repo_id=repo_id,
            filename=yolo11s_file,
            local_dir="models_download"
        )
        print(f"Downloaded weights to: {weights_local_path}")
        print(f"File size: {os.path.getsize(weights_local_path) / (1024*1024):.2f} MB")
        
        # Load with Ultralytics
        print("\n--- Loading Model with Ultralytics YOLO ---")
        model = YOLO(weights_local_path)
        print(f"Model class names ({len(model.names)} classes):")
        for k, v in model.names.items():
            print(f"  {k}: {v}")
            
        # Run on sample image
        sample_img = "bus.jpg"
        if not os.path.exists(sample_img):
            sample_img = "https://ultralytics.com/images/bus.jpg"
            
        print(f"\n--- Running inference on {sample_img} ---")
        results = model(sample_img, device="cpu")
        for r in results:
            print("\nDetected classes:")
            for c in r.boxes.cls:
                print(f" - {model.names[int(c)]}")

if __name__ == "__main__":
    main()
