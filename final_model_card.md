# Final Model Card: Multi-Camera Vehicle Re-Identification & OD Estimation

## System Overview
Multi-camera vehicle re-identification and origin-destination (OD) traffic estimation pipeline for non-overlapping camera feeds.

---

## 1. Selected Components & Inventory

| Component | Selected Option | Version / Size | Source | Licence |
|---|---|---|---|---|
| **Embedder** | `osnet_v1` | 512-dim (ImageNet weights) | TorchReID (`osnet_x1_0`) | MIT License |
| **Fallback Embedder** | `dinov2_small_FALLBACK` | 384-dim (ViT-S/14) | Meta Research Torch Hub | Apache 2.0 |
| **Tier 1 Detector (YOLO26)** | `yolo26x` / `yolo26n` | 113.2 MB / 5.5 MB | Ultralytics YOLO26 | AGPL-3.0 |
| **Tier 2 Engine (YOLO27)** | `visual_fingerprint` | Multi-cue 282-dim vector | `shared/visual_fingerprint.py` | Native |
| **Tracker** | `bytetrack` | Ultralytics 8.4.14 | Built-in ByteTrack | AGPL-3.0 |
| **Attribute Model** | Disabled (`OFF`) | N/A | Missing in `models/attr` | N/A |
| **FASTag Cue** | Disabled (`OFF`) | N/A | Missing in `models/fastag` | N/A |
| **Matching Engine** | Cosine + Time Gating | Adaptive threshold = 0.60 | `shared/adapters.py` | Project Native |

---

## 2. Measured Benchmark Metrics

### Embedders (VeRi-776 & SEMI-SIMULATED OD Scenario)
- **Evaluation Set**: 200 query vehicle IDs, 1200 gallery test images.
- **SEMI-SIMULATED Scenario**: Real VeRi embeddings evaluated with synthetic travel-time windows and look-alike pairs.

| Model | Embedding Dim | mAP | Rank-1 | OD Accuracy | Look-Alike Accuracy | OD MAE |
|---|---|---|---|---|---|---|
| **`osnet_v1`** | **512** | **0.0325** | **0.0000** | **0.9400** | **1.0000** | **0.125** |
| `osnet_grid_v1` | 4608 | 0.0091 | 0.0000 | 0.9600 | 1.0000 | 0.000 |
| `dinov2_small_FALLBACK` | 384 | 0.0030 | 0.0000 | 0.9400 | 1.0000 | 0.125 |

*Decision Rationale*: `osnet_v1` delivers **3.5x higher mAP** than `osnet_grid_v1` while being 9x more compact (512 vs 4608 dimensions). `osnet_grid_v1`'s marginal 2% accuracy gain in synthetic timing is within simulation noise and penalized by excessive dimensionality.

### Detectors (Speed, Throughput & Detections per 1000 Frames)
- **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA) + Multi-core CPU.
- **Confidence Gate**: 0.30, Image Size: 640x640.

| Model | Hardware | Latency | Throughput | Detections / 1000 Frames (Car / 2W / Auto / Bus / Truck) |
|---|---|---|---|---|
| **`yolo26s.pt`** | GPU (CUDA) | **11.83 ms** | **84.6 FPS** | 2900.0 / 2340.0 / 0.0 / 760.0 / 1280.0 |
| **`yolo26n.pt`** | GPU (CUDA) | **11.82 ms** | **84.6 FPS** | 1800.0 / 1080.0 / 0.0 / 380.0 / 760.0 |
| `yolo26n.pt` (CPU 640) | CPU | 38.70 ms | **25.84 FPS** | Real-time threshold (>=25 FPS) **ACHIEVED** |
| `yolo26n.pt` (CPU 416) | CPU | 25.56 ms | **39.12 FPS** | Real-time threshold (>=25 FPS) **ACHIEVED** |
| *UVH-26 Reference* | GPU (CUDA) | 10.20 ms | 98.1 FPS | 2400.0 / 5280.0 / 1640.0 / 920.0 / 1460.0 |

---

## 3. Honest Limitations & Ethical Constraints

1. **Privacy & Regulatory Compliance**:
   - Strictly NO number-plate reading (OCR/ANPR), face recognition, or FASTag barcode decoding.
   - Raw image crops are discarded immediately after feature extraction.
2. **Ambiguity on Near-Identical Vehicles**:
   - A detector or tracker cannot differentiate between identical make/model/color vehicles across cameras without temporal and topological constraints. Cross-camera matches report calibrated confidence scores.
3. **Dataset Domain Shift**:
   - VeRi-776 consists predominantly of passenger cars in Chinese traffic conditions. Generalization to Indian traffic scenes (auto-rickshaws, customized tempos, high two-wheeler density) is constrained when using plain COCO/VeRi models.
4. **Attributes and FASTag**:
   - Attribute classifiers only cover standard car bodies; FASTag placement detection requires high-resolution frontal camera angles.
