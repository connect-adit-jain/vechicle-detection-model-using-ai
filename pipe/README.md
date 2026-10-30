# Indian Traffic Vehicle Detector & Re-ID Setup

This repository contains the environment setup, dataset verification, pretrained baseline inference, and vehicle Re-ID (OSNet) tools for Indian urban traffic scenarios.

---

## 🚦 Target Classes

1. **car** (Hatchback, Sedan, SUV, MUV, Van)
2. **two_wheeler** (Motorcycle, Scooter, Bicycle)
3. **auto_rickshaw** (Three-wheeler)
4. **bus** (Bus, Mini-bus, Tempo-traveller)
5. **truck** (Truck, LCV)

---

## 📁 Repository Structure

- `check_data.py`: Inspects UVH-26 dataset COCO annotations, prints class distributions, and generates sample visualizations with bounding boxes.
- `check_json_stats.py`: Detailed statistics reader for all UVH-26 JSON files (MV / ST variants).
- `inspect_dataset_repo.py`: Queries Hugging Face API for dataset files, sizes, and verifies disk capacity against safety thresholds.
- `summarize_dataset_structure.py`: Breaks down splits, files, and sizes for UVH-26.
- `inspect_models_repo.py`: Queries the Hugging Face model repository `iisc-aim/UVH-26`, downloads fine-tuned Indian traffic weights (`UVH-26-MV-YOLOv11-S.pt`), and runs inference.
- `test_coco.py`: Downloads baseline `yolo11s.pt` and runs verification inference on sample imagery.
- `offline_check.py`: Offline verification with outbound socket connections strictly blocked to confirm models load from local disk with zero network dependency.
- `deep-person-reid/`: TorchReID library (KaiyangZhou) configured for vehicle Re-ID with OSNet.
- `sample_visualizations/`: Annotated sample images and collage showing detection on real Indian traffic camera feeds.
- `requirements.txt`: Complete pinned dependencies.
- `setup_report.md`: System hardware diagnostics, environment specs, cache locations, and verification logs.

---

## 🛠️ Environment Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv311
venv311\Scripts\activate

# 2. Install PyTorch (CPU or matching CUDA build)
pip install torch torchvision torchaudio

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install TorchReID in develop mode
cd deep-person-reid
python setup.py develop --no-deps
cd ..
```

---

## 📊 Verification Commands

- **Run dataset analysis & sample visualization:**
  ```bash
  python check_data.py
  ```
- **Run baseline COCO inference:**
  ```bash
  python test_coco.py
  ```
- **Run Indian-traffic fine-tuned YOLO11-S inference:**
  ```bash
  python inspect_models_repo.py
  ```
- **Run offline verification test:**
  ```bash
  python offline_check.py
  ```

---

## 📜 Licences & Attribution

- **Dataset:** UVH-26 (`iisc-aim/UVH-26`) — CC-BY-4.0
- **Model:** UVH-26 (`iisc-aim/UVH-26`) — Apache-2.0
- **TorchReID:** `deep-person-reid` — MIT License
- **Technical Report:** *The Urban Vision Hackathon Dataset and Models: Towards Image Annotations and Accurate Vision Models for Indian Traffic* (arXiv:2511.02563)

