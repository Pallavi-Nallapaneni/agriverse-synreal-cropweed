# Synthetic-to-Real Agricultural Crop-Weed Detection

Train a YOLO object detector on procedurally generated synthetic agricultural
imagery (AgriVerse) to detect maize and weeds, evaluate performance on a
held-out synthetic test set, and investigate the synthetic-to-real domain gap
through real-world CornWeed evaluation while evaluating photometric
augmentation as a potential mitigation.

---

## Sample Synthetic Data

<p align="center">
<img src="assets/sample_images/train_sample_01.png" width="30%">
<img src="assets/sample_images/val_sample_01.png" width="30%">
<img src="assets/sample_images/test_sample_01.png" width="30%">
</p>

<p align="center">
<sub>
Sample AgriVerse synthetic renders from training, validation, and testing splits.
</sub>
</p>

---

# Motivation

Collecting and labeling real agricultural imagery for crop and weed detection is slow and expensive.

Procedurally generated synthetic data provides:

- Large-scale image generation
- Exact bounding-box annotations
- Controlled variation in plant morphology, lighting, and environment

However, models trained only on synthetic data often lose accuracy when deployed on real field images due to the **synthetic-to-real domain gap**. This gap comes from differences in:

- Plant textures
- Lighting conditions
- Camera sensor noise
- Background complexity
- Soil and environmental appearance

This project trains a detector using synthetic data only, evaluates it on both synthetic and real-world images, measures the domain gap, and evaluates photometric augmentation as a potential mitigation.

---

# Pipeline

The project follows a synthetic-to-real crop-weed detection workflow with photometric augmentation and deployment-oriented inference.

```text
┌─────────────────────────────────────┐
│ AgriVerse Synthetic Dataset          │
│                                       │
│ - Procedural crop/weed generation    │
│ - Automatic YOLO annotations         │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Baseline YOLOv8 Training              │
│                                       │
│ - Synthetic images only              │
│ - Maize + weed detection             │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Domain Gap Evaluation                 │
│                                       │
│ - Synthetic test set (in-domain)     │
│ - Real CornWeed set (out-of-domain)  │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Photometric Augmentation              │
│                                       │
│ - HSV/color variation                │
│ - Noise and blur                     │
│ - Gamma/brightness variation         │
│ - Shadows/compression artifacts      │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Augmented YOLOv8 Training              │
│                                       │
│ - Original + augmented training data │
│ - Validation/test kept unchanged     │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Evaluation & Error Analysis           │
│                                       │
│ - Precision / Recall / F1            │
│ - mAP50 / mAP50-95                   │
│ - Confidence threshold analysis      │
│ - Weed object-size analysis          │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ ROS 2 Detection Integration           │
│                                       │
│ - Real-time image inference          │
│ - Crop/weed detections               │
│ - Annotated image output             │
└─────────────────────────────────────┘
```

---

# Dataset

## AgriVerse Synthetic Dataset

The AgriVerse synthetic sample contains:

- **150 images**
- 120 training images
- 15 validation images
- 15 testing images

Images are:

- 1024×1024 RGB top-down agricultural renders
- Automatically annotated in YOLO format
- Two classes: `maize`, `weed`

Plant morphology, placement, soil appearance, and lighting are procedurally generated and randomized.

| Split | Images | Maize Boxes | Weed Boxes |
|---|---:|---:|---:|
| Train | 120 | 1,023 | 5,082 |
| Validation | 15 | 124 | 633 |
| Test | 15 | 127 | 610 |

The dataset contains approximately a **5:1 weed-to-maize imbalance**.

This repository contains a preview sample of a larger 6,500-image AgriVerse dataset.

See the original dataset source for applicable licensing and citation requirements.

Reference:

> Esfandiyar, I., Moroz, I., Plaskowski, D., Gawron, T. "Sim-to-Real Transferability of Deep Learning-Based Weed and Crop Detection Models Trained on Procedurally Generated Agricultural Simulation Data."

## Real Dataset: CornWeed (Weed-AI)

Used only for out-of-domain testing — the model never sees these images during training.

- 3,574 real-world RGB field images
- Bounding-box annotations, COCO and YOLO formats available
- Classes: `maize`, `weed`

| Dataset | Images | Maize Instances | Weed Instances |
|---|---:|---:|---:|
| CornWeed Test | 3,574 | 23,985 | 257,740 |

The real dataset has a stronger imbalance: synthetic ≈5:1 weed:maize vs. CornWeed ≈10.7:1 weed:maize.

Annotations are converted into YOLO format using `scripts/convert_cornweed.py`.

Reference:

> Iqbal, N., Manss, C., Scholz, C., König, D., Igelbrink, M., Ruckelshausen, A. "AI-Based Maize and Weeds Detection on the Edge with CornWeed Dataset." FedCSIS 2023.

---

# Experimental Results

## Domain Gap: Baseline Model, Synthetic vs. Real

The baseline YOLOv8n model, trained only on synthetic images, was evaluated on both a held-out synthetic test set and the real CornWeed dataset.

| Evaluation Set | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Synthetic Test | 0.811 | 0.551 | 0.656 | 0.584 | 0.416 |
| Real CornWeed | 0.485 | 0.446 | 0.464 | 0.372 | 0.124 |

| Metric | Synthetic | Real | Relative Drop |
|---|---:|---:|---:|
| Precision | 0.811 | 0.485 | 40.2% |
| Recall | 0.551 | 0.446 | 19.1% |
| F1 | 0.656 | 0.464 | 29.3% |
| mAP50 | 0.584 | 0.372 | 36.3% |
| mAP50-95 | 0.416 | 0.124 | 70.2% |

The largest degradation occurs in mAP50-95, indicating that **localization quality is affected far more than object recognition** — the detector still finds crops and weeds reasonably well on real images, but bounding-box precision decreases substantially. Likely causes: differing plant textures, real camera noise, lighting mismatch, more complex field backgrounds, and synthetic-to-real edge/texture differences.

## Baseline vs. Photometric Augmentation (Synthetic Test Set)

To target the localization-precision gap identified above, a second model was trained on the original synthetic training images plus photometrically augmented copies (see the Synthetic Photometric Augmentation section below). Validation and test splits were kept unchanged so the comparison stays fair.

| Metric | Baseline | Augmented | Improvement |
|---|---:|---:|---:|
| Precision | 0.4289 | **0.9027** | +0.4738 |
| Recall | 0.0458 | **0.8057** | +0.7599 |
| F1 | 0.0827 | **0.8514** | +0.7687 |
| mAP50 | 0.0439 | **0.8753** | +0.8314 |
| mAP50-95 | 0.0244 | **0.6776** | +0.6532 |
| AP50 Maize | 0.0452 | **0.9036** | +0.8584 |
| AP50 Weed | 0.0427 | **0.8470** | +0.8043 |

> **Note:** these baseline figures come from a separate training run than the domain-gap table above and are not directly comparable to it — this experiment isolates the effect of augmentation on the same evaluation protocol. The augmented model substantially outperformed its own baseline on the **held-out synthetic test set**. This confirms that augmentation improved robustness *under the evaluated test conditions*; it has not yet been confirmed to close the real-world CornWeed gap specifically (see Future Work below).

## Confidence Threshold Analysis

Performed with the augmented detector to examine the precision-recall trade-off for weed detection. IoU matching threshold fixed at 0.50.

| Confidence | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 506 | 277 | 103 | 0.6462 | 0.8309 | 0.7270 |
| 0.20 | 493 | 134 | 116 | 0.7863 | 0.8095 | 0.7977 |
| 0.30 | 478 | 87 | 131 | 0.8460 | 0.7849 | 0.8143 |
| 0.40 | 465 | 59 | 144 | 0.8874 | 0.7635 | 0.8208 |
| **0.50** | **454** | **39** | **155** | **0.9209** | **0.7455** | **0.8240** |
| 0.60 | 441 | 21 | 168 | 0.9545 | 0.7241 | 0.8235 |
| 0.70 | 408 | 11 | 201 | 0.9737 | 0.6700 | 0.7938 |

The highest F1 score (0.8240) occurs at a confidence threshold of **0.50**, providing the strongest balance between precision and recall — higher thresholds further reduce false positives at the cost of missing more weeds.

## Weed Object-Size Analysis

Examined the normalized bounding-box area of detected vs. missed weed instances.

| Statistic | Detected | Missed |
|---|---:|---:|
| Count | 495 | 114 |
| Minimum area | 0.000092 | 0.000001 |
| Q25 | 0.001153 | 0.000107 |
| Median | 0.002369 | 0.000541 |
| Q75 | 0.004475 | 0.001434 |
| Maximum | 0.034668 | 0.030624 |
| Mean | 0.003797 | 0.001829 |

The median bounding-box area of **missed** weeds (0.000541) is substantially smaller than that of **detected** weeds (0.002369), indicating small weed instances are considerably harder for the detector to identify — a practically relevant finding for early-stage, site-specific weed management.

---

# Synthetic Photometric Augmentation

The augmentation pipeline modifies image appearance only — it never changes object geometry or bounding-box annotations, so original YOLO labels can be copied unchanged onto augmented images.

Implemented transformations:

- Color/HSV variation
- Brightness and contrast variation
- Gamma variation
- Image noise
- Blur
- Shadow effects
- JPEG/compression artifacts

Script: `src/augment_synthetic.py`
Config: `configs/agriverse_synthetic_aug.yaml`

The augmented training configuration uses `images/train_combined` (original + augmented), while validation and test remain `images/val` and `images/test` respectively — keeping evaluation data completely separate from the augmentation process.

---

# ROS 2 Integration

A ROS 2 detector package is included for deployment-oriented, real-time inference.

```text
agriverse_detector/
├── agriverse_detector/
│   ├── __init__.py
│   └── detector_node.py
├── package.xml
├── setup.cfg
├── setup.py
└── resource/
```

The included detector node is designed to subscribe to a camera image topic,
load a trained YOLO checkpoint, run crop/weed detection filtered by
confidence, and publish annotated/structured detection outputs along with
inference timing and FPS. It builds and compiles successfully; live
camera/ROS-bag deployment has not yet been benchmarked (see Future Work
below).

A standalone (non-ROS) reference implementation is also provided in `src/detector_node.py`.

The package is intended to be adaptable to a real camera, a recorded ROS bag, or a robotic platform. For edge deployment, the trained YOLO model can be exported to an accelerated inference format such as TensorRT when supported by the target hardware (e.g., NVIDIA Jetson) — this path is untested so far.

---

# Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Development | VS Code, PowerShell, Git, GitHub |
| Computer Vision | OpenCV, Pillow |
| Deep Learning | YOLOv8, PyTorch |
| Data Processing | NumPy, Pandas, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Deployment | Docker, ROS 2 |

---

# Repository Structure

```text
├── assets/
│   └── sample_images/
│
├── configs/
│   ├── agriverse_synthetic.yaml
│   ├── agriverse_synthetic_aug.yaml
│   └── real_target.yaml
│
├── scripts/
│   ├── setup_env.ps1
│   └── convert_cornweed.py
│
├── src/
│   ├── data_prep.py
│   ├── visualize_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── augment_synthetic.py
│   └── detector_node.py
│
├── agriverse_detector/
│   ├── agriverse_detector/
│   │   ├── __init__.py
│   │   └── detector_node.py
│   ├── package.xml
│   ├── setup.cfg
│   ├── setup.py
│   └── resource/
│
├── error_analysis.py
├── confidence_threshold_experiment.py
├── weed_size_analysis.py
├── requirements.txt
└── README.md
```

Large datasets, training outputs, experiment results, Python cache files, and model weights are excluded via `.gitignore` — this README reports final numbers explicitly since `results/` and `runs/` are not available on GitHub.

---

# Setup

```powershell
git clone <repository-url>
cd agriverse-synreal-cropweed
.\scripts\setup_env.ps1
```

Alternative:

```powershell
python -m venv .venv
pip install -r requirements.txt
```

---

# Usage

**Dataset validation**

```bash
python src/data_prep.py --data configs/agriverse_synthetic.yaml
python src/visualize_dataset.py --data configs/agriverse_synthetic.yaml --split train --n 6
```

**Train the baseline model**

```bash
python src/train.py --data configs/agriverse_synthetic.yaml
```

**Generate photometrically augmented training data**

```bash
python src/augment_synthetic.py
```

**Train using the augmented dataset**

```bash
python src/train.py --data configs/agriverse_synthetic_aug.yaml
```

**Evaluate a trained model**

```bash
python src/evaluate.py \
    --weights runs/detect/<run>/weights/best.pt \
    --data configs/agriverse_synthetic_aug.yaml \
    --split test \
    --imgsz 1024 \
    --tag augmented_test
```

**Confidence-threshold analysis**

```bash
python confidence_threshold_experiment.py
```

**Weed-size analysis**

```bash
python weed_size_analysis.py
```

**ROS 2**

Build the `agriverse_detector` package from a ROS 2 workspace and launch the detector node with the trained model supplied through its `weights_path` parameter.

---

# Qualitative Prediction Example

<p align="center">
<img src="assets/sample_images/real_sample_01.png" width="45%">
<img src="assets/sample_images/real_prediction_01.png" width="45%">
</p>

<p align="center">
<sub>
Left: Original CornWeed image. Right: YOLOv8 prediction after synthetic-only training.
</sub>
</p>

---

# Future Work

- [x] Train YOLOv8 baseline on synthetic data
- [x] Evaluate synthetic-to-real transfer and quantify the domain gap
- [x] Implement and evaluate photometric augmentation (synthetic test set)
- [x] Confidence-threshold analysis
- [x] Weed object-size / error analysis
- [x] ROS 2 real-time deployment package

Still open:

- [ ] Evaluate the **augmented** model on the real CornWeed dataset, to confirm whether the synthetic-test improvement transfers to the real-world domain gap
- [ ] Benchmark the ROS 2 detector node against a live camera or recorded ROS bag (currently builds/compiles but is untested in a running deployment)
- [ ] Jetson/TensorRT edge-deployment benchmarking (FPS, latency)
- [ ] Class-balanced training
- [ ] Larger YOLO models
- [ ] Broader domain randomization / additional synthetic generators

---

# License

Code: MIT License

Datasets: see the original dataset sources (AgriVerse; CornWeed / Weed-AI) for applicable licensing and citation requirements — not independently confirmed here.

Please cite the original dataset publications when using the data.
