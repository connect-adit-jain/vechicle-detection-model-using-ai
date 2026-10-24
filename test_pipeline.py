import os
import torch
import torchvision.transforms as T
from PIL import Image
import torchreid
from torchvision.models import resnet18, ResNet18_Weights

def test_pipeline():
    print("=" * 60)
    print("SMOKE TEST: GPU + Dataset + Models Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. Target Device: {device} ({torch.cuda.get_device_name(0)})")

    # Pick a sample image from the dataset
    train_dir = "data/veri/image_train"
    sample_img_name = os.listdir(train_dir)[0]
    sample_path = os.path.join(train_dir, sample_img_name)
    print(f"2. Testing on real image: {sample_img_name}")

    # Image preprocessing
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(sample_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device) # Shape: (1, 3, 224, 224)
    print(f"   Input tensor shape: {tensor.shape}")

    # Test OSNet (Re-ID)
    print("3. Testing OSNet x1.0 (Vehicle Re-ID Backbone)...")
    reid_model = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True).to(device)
    reid_model.eval()
    with torch.no_grad():
        features = reid_model(tensor)
    print(f"   OSNet Feature embedding shape: {features.shape} (512-dim vector)")

    # Test ResNet18 (Classifier)
    print("4. Testing ResNet18 (Vehicle Attribute Backbone)...")
    attr_model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    attr_model.eval()
    with torch.no_grad():
        logits = attr_model(tensor)
    print(f"   ResNet18 Logits shape: {logits.shape} (1000 classes)")

    # Check GPU memory
    allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
    print("5. GPU Memory Status:")
    print(f"   Allocated: {allocated_mb:.2f} MB | Reserved: {reserved_mb:.2f} MB")
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
