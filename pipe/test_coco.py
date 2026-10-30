import torch
from ultralytics import YOLO

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    
    print("\n--- Downloading/Loading yolo11s.pt ---")
    model = YOLO("yolo11s.pt")
    
    sample_img = "https://ultralytics.com/images/bus.jpg"
    print(f"\n--- Running inference on sample image: {sample_img} ---")
    results = model(sample_img, device=device)
    
    for r in results:
        print(f"\nDevice used during inference: {r.boxes.xyxy.device if len(r.boxes) > 0 else device}")
        print("Detected classes:")
        classes_detected = set()
        for c in r.boxes.cls:
            class_name = model.names[int(c)]
            classes_detected.add(class_name)
        for cls in sorted(list(classes_detected)):
            count = sum(1 for c in r.boxes.cls if model.names[int(c)] == cls)
            print(f" - {cls}: {count}")

if __name__ == "__main__":
    main()

