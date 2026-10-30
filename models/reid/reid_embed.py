"""
OSNet v1 Vehicle Re-ID Embedder
Wraps torchreid OSNet x1.0 with the standard interface:
  EMBEDDER_NAME, load_model(path, device), embed(model, crops_bgr) -> (N, D) float32 L2-normalised
"""
import numpy as np
import torch
import torchvision.transforms as T
import cv2

EMBEDDER_NAME = "osnet_v1"

_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(path=None, device="cuda"):
    """Load OSNet x1.0 for vehicle re-id embedding.
    Args:
        path: path to .pth weights (None = use pretrained ImageNet weights from cache)
        device: 'cuda' or 'cpu'
    Returns:
        dict with 'model', 'device', 'name'
    """
    import torchreid
    model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=1000, pretrained=(path is None)
    )
    if path is not None:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return {"model": model, "device": device, "name": EMBEDDER_NAME}


def embed(model_dict, crops_bgr):
    """Embed a list/array of BGR crops.
    Args:
        model_dict: dict returned by load_model
        crops_bgr: list of HxWx3 uint8 BGR numpy arrays
    Returns:
        (N, D) float32 numpy array, L2-normalised
    """
    if len(crops_bgr) == 0:
        return np.zeros((0, 512), dtype=np.float32)
    model = model_dict["model"]
    device = model_dict["device"]
    tensors = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(_TRANSFORM(rgb))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        features = model(batch)
    features = features.cpu().numpy().astype(np.float32)
    # L2 normalise
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    features = features / norms
    return features

