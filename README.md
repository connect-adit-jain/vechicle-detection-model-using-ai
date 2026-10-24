# Multi-Camera Vehicle Detection & Re-Identification (NetraFlow)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision and traffic analytics system designed for **multi-camera vehicle detection, visual re-identification, and origin-destination (OD) traffic estimation** across non-overlapping surveillance camera feeds in dense, mixed urban traffic (cars, auto-rickshaws, two-wheelers, buses, and trucks).

---

## 🚀 Key Highlights & Capabilities

### 🔒 1. Privacy-Safe Visual Re-Identification
- **Zero License Plate Reading (ANPR-Free)**: Strictly operates without optical character recognition or personal identity tracking to ensure ethical and regulatory compliance.
- **Privacy-Preserving Aggregations**: Outputs traffic flows exclusively in aggregate groups (minimum cohort size $\ge 10$) with optional **Laplace Differential Privacy** noise injection.
- **Ephemeral Processing**: Vehicle crops are discarded immediately after embedding computation.

### 🧠 2. Two-Tier Detection & Visual Fingerprinting
- **Tier 1 — Real-Time Detection Engine (YOLO26)**:
  - High-throughput detection optimized across GPU (84+ FPS) and CPU (25.8 - 39.1 FPS real-time on Intel/AMD CPUs).
  - Handles 5 target vehicle categories: `car`, `two_wheeler`, `auto_rickshaw`, `bus`, `truck`.
- **Tier 2 — Disambiguation Engine (YOLO27)**:
  - Generates multi-cue 282-dimensional visual fingerprints to disambiguate look-alike vehicles of identical make, model, and color.
  - Features: LAB color-invariant decoupling, scratch/dent edge signatures, dust patterns, roof racks/carriers, and windshield 3×3 FASTag grid positioning.

### 🌐 3. Resilient Multi-Camera Tracking & Cross-Camera Re-ID
- **OSNet Re-ID Embedder**: 512-dimensional feature representations using omni-scale feature learning (`osnet_x1_0`).
- **ViT Fallback**: Integrated DINOv2 (`dinov2_small`) fallback mechanism when domain shift or lighting conditions degrade CNN features.
- **Bayesian Spatio-Temporal Gating**: Fuses visual cosine similarity with topological travel-time probability distributions between camera nodes.
- **Camera Outage Recovery**: Graph-based traffic conservation matrix estimation that interpolates vehicle flows when a camera goes offline.
- **Night-Mode Compensation**: Automatic CLAHE histogram equalization, gamma correction, and headlight glare suppression.
- **Unlabeled Camera Domain Adaptation**: Self-supervised clustering (DBSCAN + centroid alignment) for zero-shot deployment of newly introduced camera feeds.

---

## 📁 Repository Structure

```text
├── config.yaml               # Central pipeline & model configuration
├── requirements.txt          # Production environment dependencies
├── final_model_card.md       # Measured benchmark metrics and ethical limitations
├── setup_report.md           # Diagnostics, hardware validation, and baseline logs
│
├── matching/                 # Cross-camera Re-ID & spatio-temporal matching
│   └── cross_camera_matcher.py
├── tracking/                 # Single-camera trajectory tracker (ByteTrack)
│   └── tracker.py
├── traffic/                  # Traffic intelligence & phase 2 modules
│   ├── od_engine.py          # Origin-Destination flow aggregation & DP
│   ├── night_mode.py         # Night footage enhancement & glare mitigation
│   └── unlabeled_camera.py   # Domain adaptation for unannotated cameras
├── shared/                   # Common adapters, validation & feature engines
│   ├── adapters.py           # Unified interfaces & similarity gating
│   ├── dinov2_fallback.py    # DINOv2 feature extractor fallback
│   ├── validate.py           # Integrity assertions & sanity checks
│   └── visual_fingerprint.py # YOLO27 multi-cue visual fingerprinting
├── models/                   # Model wrappers & checksum verification
│   ├── CHECKSUMS.txt         # SHA256 hashes of tested model weights
│   ├── det/                  # Detection model wrappers
│   ├── reid/                 # OSNet 512-dim Re-ID wrapper
│   └── reid_grid/            # Grid-pooled 4608-dim Re-ID wrapper
├── deep-person-reid/         # TorchReID library with Windows Cython patch
├── tools/                    # Operational utilities
│   ├── label_pairs.py        # Pairwise labeling tool
│   └── manual_count.py       # Ground-truth count verification
├── tests/                    # Performance & benchmark suites
│   ├── benchmark_detectors.py# Latency & throughput benchmarking
│   └── benchmark_embedders.py# mAP, Rank-1, and look-alike evaluation
│
├── test_pipeline.py          # End-to-end integration test
├── run_all.py                # Pipeline execution across feeds
└── run_phase2.py             # Phase 2 advanced scenario runner
```

---

## 📊 Measured Benchmark Results

| Component | Architecture | Embedding Dim / Weight | Latency (GPU) | Throughput | Accuracy / mAP |
|---|---|---|---|---|---|
| **Embedder (Selected)** | `osnet_v1` | 512 | 14.2 ms | 70.4 img/s | **0.0325 mAP** (3.5x over baseline) |
| **Embedder (Grid)** | `osnet_grid_v1` | 4608 | 18.5 ms | 54.1 img/s | 0.0091 mAP |
| **Embedder (ViT Fallback)**| `dinov2_small` | 384 | 22.0 ms | 45.5 img/s | 0.0030 mAP |
| **Detector (Small)** | `yolo26s` | 20.4 MB | **11.8 ms** | **84.6 FPS** | High accuracy balanced |
| **Detector (Nano - Realtime)** | `yolo26n` | 5.5 MB | **25.5 ms (CPU)** | **39.1 FPS (CPU)** | Real-time on CPU achieved |

*Hardware: NVIDIA RTX GPU + Multi-Core CPU, evaluated on UVH-26 & VeRi-776 traffic benchmarks.*

---

## 🛠️ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/iamshresthraj/vehicle-detection-model.git
cd vehicle-detection-model
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# PyTorch with CUDA (or CPU version)
pip install torch torchvision torchaudio

# Install project requirements
pip install -r requirements.txt

# Install deep-person-reid in editable mode
cd deep-person-reid
python setup.py develop --no-deps
cd ..
```

### 4. Run Verification Tests
```bash
# Test the integrated pipeline
python test_pipeline.py

# Run Phase 2 full simulation (Night mode, Camera outage, Unlabeled camera, Privacy OD)
python run_phase2.py

# Run detector & embedder benchmarks
python tests/benchmark_detectors.py
python tests/benchmark_embedders.py
```

---

## ⚙️ Configuration

System behavior can be customized via [`config.yaml`](config.yaml):
- **Detector Weights**: Toggle between `yolo26x.pt`, `yolo26s.pt`, or `yolo26n.pt` for resource-constrained environments.
- **YOLO27 Features**: Toggle scratches, dust patterns, FASTag 3×3 grid, or tyre profile cues.
- **Matching Weights**: Adjust relative importance of appearance ($w_{\text{app}}$), attribute ($w_{\text{attr}}$), and travel time ($w_{\text{time}}$).
- **Privacy Controls**: Adjust aggregate group thresholds and differential privacy parameters.

---

## 📜 License & Acknowledgements

- **TorchReID**: Kaiyang Zhou ([MIT License](https://github.com/KaiyangZhou/deep-person-reid))
- **YOLO Implementations**: Ultralytics ([AGPL-3.0 License](https://github.com/ultralytics/ultralytics))
- **Dataset Benchmarks**: UVH-26 (Indian Urban Vision Hackathon, IISc AIM, CC-BY-4.0) & VeRi-776.
