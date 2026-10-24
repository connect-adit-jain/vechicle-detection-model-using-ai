"""
OSNet Grid-Pooled v1 Vehicle Re-ID Embedder
Uses spatial 3x3 grid pooling on OSNet x1.0 feature maps.
Interface:
  EMBEDDER_NAME, load_model(path, device), embed(model, crops_bgr) -> (N, D) float32 L2-normalised
  cell_importance(model, crop_bgr) -> (3, 3) float32
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2

EMBEDDER_NAME = "osnet_grid_v1"

_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(path=None, device="cuda"):
    """Load OSNet x1.0 for grid-pooled embedding.
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


def _extract_feature_map(model_dict, crops_bgr):
    """Extract spatial feature maps before global pooling.
    Returns: (N, C, H, W) tensor on device
    """
    model = model_dict["model"]
    device = model_dict["device"]
    tensors = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(_TRANSFORM(rgb))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        fmaps = model.featuremaps(batch)  # (N, 512, H, W) typically (N, 512, 14, 14) for 256x128 input
    return fmaps


def embed(model_dict, crops_bgr):
    """Embed using 3x3 grid-pooled features.
    Each crop produces a 512*9=4608-dim vector, L2-normalised.
    Args:
        model_dict: dict from load_model
        crops_bgr: list of HxWx3 uint8 BGR numpy arrays
    Returns:
        (N, 4608) float32 numpy array, L2-normalised
    """
    if len(crops_bgr) == 0:
        return np.zeros((0, 4608), dtype=np.float32)
    fmaps = _extract_feature_map(model_dict, crops_bgr)  # (N, 512, H, W)
    N, C, H, W = fmaps.shape
    # Adaptive avg pool to 3x3 spatial grid
    grid_features = F.adaptive_avg_pool2d(fmaps, (3, 3))  # (N, 512, 3, 3)
    features = grid_features.reshape(N, -1).cpu().numpy().astype(np.float32)  # (N, 4608)
    # L2 normalise
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    features = features / norms
    return features


def cell_importance(model_dict, crop_bgr):
    """Compute relative importance of each cell in the 3x3 spatial grid.
    Args:
        model_dict: dict from load_model
        crop_bgr: HxWx3 uint8 BGR numpy array (single crop)
    Returns:
        (3, 3) float32 numpy array, values sum to 1.0
    """
    fmaps = _extract_feature_map(model_dict, [crop_bgr])  # (1, 512, H, W)
    grid_features = F.adaptive_avg_pool2d(fmaps, (3, 3))  # (1, 512, 3, 3)
    # Importance = L2 norm of the channel vector in each cell
    importance = torch.norm(grid_features[0], dim=0)  # (3, 3)
    importance = importance / importance.sum()
    return importance.cpu().numpy().astype(np.float32)
