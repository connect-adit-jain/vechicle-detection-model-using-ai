"""
DINOv2-small fallback embedder.
Used ONLY when no trained Re-ID model is available. Flagged as FALLBACK everywhere.
Interface:
  EMBEDDER_NAME, load_model(path, device), embed(model, crops_bgr) -> (N, D) float32 L2-normalised
"""
import numpy as np
import torch
import torchvision.transforms as T
import cv2

EMBEDDER_NAME = "dinov2_small_FALLBACK"

_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(path=None, device="cuda"):
    """Load DINOv2-small from torch hub (or local cache).
    This is a FALLBACK embedder.
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model = model.to(device).eval()
    return {"model": model, "device": device, "name": EMBEDDER_NAME}


def embed(model_dict, crops_bgr):
    """Embed crops using DINOv2-small (384-dim output).
    Returns: (N, 384) float32 L2-normalised. FALLBACK.
    """
    if len(crops_bgr) == 0:
        return np.zeros((0, 384), dtype=np.float32)
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
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    features = features / norms
    return features

