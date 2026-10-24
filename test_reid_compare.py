import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
import torchreid

def compare_images(img1_path, img2_path):
    print("=" * 60)
    print("VEHICLE RE-ID COMPARISON (OSNet)")
    print("=" * 60)

    if not os.path.exists(img1_path):
        print(f"Error: {img1_path} not found.")
        return
    if not os.path.exists(img2_path):
        print(f"Error: {img2_path} not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading OSNet on {device}...")
    model = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True).to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    t1 = transform(Image.open(img1_path).convert("RGB")).unsqueeze(0).to(device)
    t2 = transform(Image.open(img2_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        feat1 = model(t1)
        feat2 = model(t2)

    # Compute Cosine Similarity and Euclidean Distance
    cos_sim = torch.nn.functional.cosine_similarity(feat1, feat2).item()
    eucl_dist = torch.dist(feat1, feat2, p=2).item()

    print(f"\nImage 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print("\n--- Metric Results ---")
    print(f"Cosine Similarity : {cos_sim:.4f}  (Range: -1.0 to +1.0; Higher = More Similar)")
    print(f"Euclidean Distance: {eucl_dist:.4f}  (Lower = More Similar)")

    if cos_sim > 0.70:
        print("\n=> Prediction: HIGH probability of being the SAME vehicle.")
    elif cos_sim > 0.50:
        print("\n=> Prediction: MODERATE similarity (visually related appearance).")
    else:
        print("\n=> Prediction: DIFFERENT vehicles.")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        compare_images(sys.argv[1], sys.argv[2])
    else:
        # Default test comparing two sample images of car 0001
        sample1 = "data/veri/image_train/0001_c001_00016450_0.jpg"
        sample2 = "data/veri/image_train/0001_c001_00016460_0.jpg"
        print("Usage: .venv\\Scripts\\python test_reid_compare.py <image1.jpg> <image2.jpg>\n")
        print("Running demo comparing two images of the same vehicle:")
        compare_images(sample1, sample2)
