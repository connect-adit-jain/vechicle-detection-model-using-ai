# Indian-Traffic Vehicle Detector Setup Report

**Date:** 2026-10-24  
**Project:** Indian-Traffic Vehicle Detector (Classes: car, two_wheeler, auto_rickshaw, bus, truck)  
**Role:** Hackathon Laptop Environment Setup & Data Prep  
**Privacy & Compliance:** Strictly compliant — no number-plate reading, no face recognition, no personal identifier extraction.

---

## 1. Hardware & System Diagnostics

| Metric | Status / Value | Verification Command / Output |
|---|---|---|
| **OS** | Windows 11 Home Single Language (Build 26300) | `systeminfo` |
| **GPU (Discrete)** | **None** (Intel Iris Xe integrated graphics only) | `Get-CimInstance Win32_VideoController` |
| **NVIDIA SMI** | **Failed / Not Present** (`nvidia-smi` not found) | No NVIDIA hardware installed |
| **VRAM** | Shared system memory (~3.84 GB allocated by Intel Xe) | AdapterRAM = 4,122,664,960 bytes |
| **System RAM** | 7,863 MB (~8.0 GB total, ~700 MB free) | Total Physical Memory: 7,863 MB |
| **Disk Space** | C: Drive: 475.6 GB total, **229.8 GB free** | `Get-CimInstance Win32_LogicalDisk` |

> **Hardware Decision:** Because this laptop lacks an NVIDIA GPU, CUDA is unavailable. Per user approval, PyTorch was installed with the CPU build for local setup, dataset inspection, and inference demos. Training will be executed on Google Colab or cloud GPU.

---

## 2. Environment & Python

- **Python:** 3.11.9 (created at `.\venv311\`)
- **PyTorch Build:** `torch-2.14.1+cpu`, `torchvision-0.29.1+cpu`, `torchaudio-2.11.0+cpu`
- **CUDA Availability:** `torch.cuda.is_available() == False` (Device: `cpu`)

---

## 3. Installed Packages

All required packages installed into `venv311` and verified via clean import test:
- `ultralytics` == 8.4.186
- `huggingface_hub` == 2.3.0
- `opencv-python` == 5.0.0.93
- `pyyaml` == 6.0.3
- `pandas` == 3.1.0
- `matplotlib` == 3.11.2
- `tqdm` == 4.70.1
- `lxml` == 6.1.3
- `pycocotools` == 2.0.11

---

## 4. Cache & Storage Locations

- **Workspace Path:** `c:\Users\chtus\Downloads\Impact Forge`
- **Virtual Environment:** `c:\Users\chtus\Downloads\Impact Forge\venv311`
- **Dataset Local Directory:** `c:\Users\chtus\Downloads\Impact Forge\data\uvh26`
- **Models Local Directory:** `c:\Users\chtus\Downloads\Impact Forge\models_download`
- **Ultralytics Config/Cache:** `C:\Users\chtus\AppData\Roaming\Ultralytics`
- **Hugging Face Hub Cache:** `C:\Users\chtus\.cache\huggingface`

---

## 5. COCO Baseline Weights (`yolo11s.pt`)

- **Model:** `yolo11s.pt` (18.42 MB)
- **Classes:** 80 COCO classes
- **Inference Verification:** Executed on `bus.jpg` (CPU device, 1554 ms inference)
- **Detected Classes:** `bus: 1`, `person: 4`

---

## 6. Dataset Download: UVH-26 (`iisc-aim/UVH-26`)

- **Repository:** Hugging Face Dataset `iisc-aim/UVH-26`
- **License:** CC-BY-4.0 (Public, no login required)
- **Full Repository Size on Hub:** **83.81 GB** across 26,653 files
  - `UVH-26-Train`: 21,351 files (67.16 GB)
  - `UVH-26-Val`: 5,299 files (16.66 GB)
  - Annotation JSONs: 4 main files (~142 MB total)
- **Download Scope (User Approved):** `UVH-26-Val` split + all 4 annotation JSON files downloaded into `data/uvh26/`.
- **COCO JSON Files Inspected:**
  1. `UVH-26-MV-Train.json`: 21,349 images, 252,723 bounding boxes
  2. `UVH-26-ST-Train.json`: 17,387 images, 226,239 bounding boxes
  3. `UVH-26-MV-Val.json`: 5,297 images, 63,497 bounding boxes
  4. `UVH-26-ST-Val.json`: 4,339 images, 57,163 bounding boxes

### Real Class Names in UVH-26 & Mapping to Hackathon Goal:

| UVH-26 ID | Real Class Name | Hackathon Target Class | Box Count (MV-Train) | Box Count (MV-Val) |
|---|---|---|---|---|
| 1 | Hatchback | car | 24,196 | 6,094 |
| 2 | Sedan | car | 12,799 | 3,151 |
| 3 | SUV | car | 10,536 | 2,639 |
| 4 | MUV | car | 5,208 | 1,315 |
| 5 | Bus | bus | 7,456 | 1,830 |
| 6 | Truck | truck | 10,415 | 2,596 |
| 7 | Three-wheeler | auto_rickshaw | 41,961 | 10,467 |
| 8 | Two-wheeler | two_wheeler | 119,519 | 30,211 |
| 9 | LCV | truck | 13,854 | 3,491 |
| 10 | Mini-bus | bus | 699 | 174 |
| 11 | Tempo-traveller | bus | 1,345 | 335 |
| 12 | Bicycle | two_wheeler | 2,710 | 681 |
| 13 | Van | car | 1,743 | 443 |
| 14 | Others | — | 282 | 70 |

- **Sample Visualizations:** 6 sample images with drawn bounding boxes and labels saved to `sample_visualizations/` and combined in `sample_visualizations/collage_6_samples.png`.

---

## 7. Fine-Tuned Indian Traffic Model (`iisc-aim/UVH-26`)

- **Repository:** Hugging Face Model `iisc-aim/UVH-26`
- **License:** Apache-2.0
- **Model Card Citation:** *The Urban Vision Hackathon Dataset and Models: Towards Image Annotations and Accurate Vision Models for Indian Traffic* (arXiv:2511.02563)
- **Selected Checkpoint:** `weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt`
- **Local Download Path:** `models_download/weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt` (54.50 MB)
- **Classes Configured (14 classes):**
  `0: Hatchback, 1: Sedan, 2: SUV, 3: MUV, 4: Bus, 5: Truck, 6: Three-wheeler, 7: Two-wheeler, 8: LCV, 9: Mini-bus, 10: tempo-traveller, 11: bicycle, 12: Van, 13: Others`
- **Sample Inference Test:** Ran on `bus.jpg` (CPU device, 375 ms inference); successfully detected `Bus` instances.

---

## 8. Offline Verification Test

A standalone process (`offline_check.py`) with all outbound network sockets blocked verified that:
1. `yolo11s.pt` loaded from local disk and performed inference offline.
2. `UVH-26-MV-YOLOv11-S.pt` loaded from local disk and performed inference offline.
3. Zero network calls occurred. Result: **PASSED**.

