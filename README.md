<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Research%20Ready-brightgreen?style=for-the-badge"/>

# 🧠 Brain-Tumor-Detection-Using-MRI
### An Explainable AI Framework for Brain Tumor Detection

**Deep Learning MRI Classification | Explainable AI | Grad-CAM XAI**

This project focuses on detecting brain tumors from MRI scans using deep learning and explainable AI. It classifies MRI images into glioma, meningioma, pituitary tumor, and no tumor categories, while also generating Grad-CAM heatmaps to highlight the regions that influence each prediction. The system is designed to support medical image analysis workflows and research in computer-aided diagnosis.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Notebooks](#-notebooks)
- [Results](#-results)
- [Streamlit App](#-streamlit-app)
- [Future Work](#-future-work)
- [Citation](#-citation)

---

## 🎯 Overview

This project presents a **deep learning clinical decision support system** for automated
brain tumor classification from MRI images, with an integrated **Explainable AI (XAI)**
layer powered by **Grad-CAM**.

### Key Features

| Feature | Description |
|---|---|
| 🧠 **Custom CNN** | 4-block deep CNN with BN, GAP, Dropout |
| 🔬 **4-Class Classification** | Glioma, Meningioma, Pituitary, No Tumor |
| 🔍 **Grad-CAM XAI** | Visual explanation of model decisions |
| 📊 **Full Metrics Suite** | Accuracy, F1, Kappa, ROC-AUC |
| 🌐 **Streamlit App** | Interactive MRI upload + prediction UI |
| 🔄 **Modular Codebase** | PEP-8, pathlib, docstrings, reusable functions |
| 📓 **Research Notebooks** | 6 step-by-step Jupyter notebooks |
| 🚀 **IEEE-Ready** | Clean code for research paper integration |

---

## 🏗️ Architecture

```
Input MRI (224×224×3)
│
├── Block 1: Conv(32)×2 → BN → ReLU → MaxPool(2×2) → [112×112×32]
├── Block 2: Conv(64)×2 → BN → ReLU → MaxPool(2×2) → [56×56×64]
├── Block 3: Conv(128)×2 → BN → ReLU → MaxPool(2×2) → [28×28×128]
├── Block 4: Conv(256)×2 → BN → ReLU             → [28×28×256]  ← Grad-CAM Layer
│
├── Global Average Pooling                         → [256]
├── Dense(512) → BN → ReLU → Dropout(0.5)         → [512]
├── Dense(256) → BN → ReLU → Dropout(0.3)         → [256]
└── Dense(4, Softmax)                              → [4]  ← Class Probabilities
```

**Design Philosophy:**
- `BatchNormalization` after every Conv layer — stabilizes training
- `GlobalAveragePooling` instead of Flatten — fewer parameters, less overfitting
- No pooling in Block 4 — larger spatial maps = more detailed Grad-CAM heatmaps
- `He Normal` initialization — optimal for ReLU networks
- `Label Smoothing` (α=0.1) — improves calibration and generalization

---

## 📁 Dataset

**Brain Tumor MRI Dataset (4 Classes)** — [Available on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

```
data/raw/
├── Training/
│   ├── glioma/         ← Glioblastoma, astrocytoma MRIs
│   ├── meningioma/     ← Meningioma tumor MRIs
│   ├── pituitary/      ← Pituitary adenoma MRIs
│   └── notumor/        ← Normal brain MRIs
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

### Class Descriptions

| Class | Description | Malignancy |
|---|---|---|
| **Glioma** | Arises from glial cells; most common malignant brain tumor (~30% of all) | High |
| **Meningioma** | Arises from meninges; mostly benign (WHO Grade I) | Low–High |
| **Pituitary** | Pituitary gland adenoma; usually benign, hormonal effects | Low |
| **No Tumor** | Normal brain scan, no pathological findings | — |

---

## 📂 Project Structure

```
Brain-Tumor-Detection-Using-MRI/
│
├── 📄 main.py                    ← CLI pipeline entry point
├── 📄 requirements.txt           ← Pinned dependencies
├── 📄 README.md
│
├── 📁 app/                       ← Streamlit web application
│   ├── app.py                    ← Main Streamlit app (Home + Predict + About)
│   ├── pages/
│   └── assets/
│
├── 📁 data/
│   ├── raw/                      ← Raw dataset (Training/ + Testing/)
│   ├── processed/                ← Preprocessed data (optional cache)
│   └── metadata/                 ← Class distribution CSVs
│
├── 📁 docs/                      ← Project documentation
│   ├── architecture.md
│   ├── dataset.md
│   ├── gradcam_explanation.md
│   └── api_reference.md
│
├── 📁 notebooks/                 ← Step-by-step Jupyter notebooks
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Custom_CNN.ipynb
│   ├── 04_Model_Evaluation.ipynb
│   ├── 05_Explainable_AI_GradCAM.ipynb
│   └── 06_Streamlit_App.ipynb
│
├── 📁 outputs/                   ← All generated artifacts
│   ├── saved_models/             ← best_model.keras, final_model.keras
│   ├── plots/                    ← training_history.png, roc_curves.png
│   ├── confusion_matrix/         ← confusion_matrix_normalized.png
│   ├── gradcam/                  ← Grad-CAM per-class visualizations
│   ├── predictions/              ← Batch prediction results
│   └── metrics/                  ← classification_report.csv, scalar_metrics.json
│
└── 📁 src/                       ← Core Python package
    ├── __init__.py
    ├── config.py                 ← 🎛️ Central configuration
    ├── preprocessing/
    │   ├── data_loader.py        ← TF dataset pipeline
    │   ├── augmentation.py       ← Keras augmentation layers
    │   └── image_utils.py        ← Resize, normalize, decode
    ├── models/
    │   ├── custom_cnn.py         ← CNN architecture (Functional API)
    │   └── callbacks.py          ← Training callbacks factory
    ├── training/
    │   └── trainer.py            ← Training loop + model persistence
    ├── evaluation/
    │   ├── metrics.py            ← Accuracy, F1, Kappa, AUC
    │   └── visualizer.py         ← Plots: history, CM, ROC
    ├── explainability/
    │   └── gradcam.py            ← Grad-CAM class + batch visualization
    ├── deployment/
    │   └── inference.py          ← TumorClassifier + PredictionResult
    └── utils/
        ├── logger.py             ← Rotating file + console logger
        └── helpers.py            ← Seed, dirs, JSON I/O, formatting
```

---

## ⚡ Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/Ritika1014/Brain-Tumor-Detection-Using-MRI.git
cd Brain-Tumor-Detection-Using-MRI

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
and place it under:

```
data/raw/Training/glioma/
data/raw/Training/meningioma/
data/raw/Training/notumor/
data/raw/Training/pituitary/
data/raw/Testing/...
```

### 3. Run the Full Pipeline

```bash
# Full pipeline: preprocess → train → evaluate → Grad-CAM
python main.py

# Or run individual stages
python main.py --stage train
python main.py --stage evaluate
python main.py --stage gradcam

# Custom hyperparameters
python main.py --epochs 30 --lr 0.0005 --batch-size 16

# Show configuration
python main.py --show-config
```

### 4. Launch Streamlit App

```bash
streamlit run app/app.py
```

### 5. Open Jupyter Notebooks

```bash
jupyter lab notebooks/
```

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `01_Data_Understanding.ipynb` | EDA: class distribution, sample visualization, dataset statistics |
| `02_Preprocessing.ipynb` | Image resizing, normalization, augmentation pipeline |
| `03_Custom_CNN.ipynb` | Model architecture, training, history visualization |
| `04_Model_Evaluation.ipynb` | Confusion matrix, ROC curves, classification report |
| `05_Explainable_AI_GradCAM.ipynb` | Grad-CAM theory, implementation, per-class visualization |
| `06_Streamlit_App.ipynb` | Streamlit app walkthrough and deployment guide |

---

## 📊 Results

*(Run training to populate results. Representative expected results for this architecture:)*

| Metric | Expected Value |
|---|---|
| Test Accuracy | ~95–97% |
| Macro F1-Score | ~0.94–0.96 |
| Cohen's Kappa | ~0.93–0.96 |
| ROC-AUC (Macro) | ~0.98–0.99 |

### Sample Outputs

```
outputs/
├── plots/training_history.png        ← Accuracy & Loss curves
├── confusion_matrix/                 ← Normalized + count CM
├── plots/roc_curves.png              ← Multi-class ROC
├── metrics/classification_report.csv ← Per-class metrics
└── gradcam/                          ← Grad-CAM overlays per class
```

---

## 🌐 Streamlit App

The web application provides an interactive interface:

- **🏠 Home** — Project overview, architecture pipeline, class descriptions.
- **🔬 Predict MRI** — Upload MRI → AI prediction → Grad-CAM heatmap + confidence.
- **📊 About & Methods** — Model architecture table, training methodology, Grad-CAM theory, references.

```bash
streamlit run app/app.py
# Opens at http://localhost:8501
```

---

## 🔭 Future Work

This codebase is designed for extensibility. Planned upgrades:

| Phase | Feature |
|---|---|
| Phase 2 | **Transfer Learning** — ResNet50V2, EfficientNetB3, DenseNet121 |
| Phase 3 | **Attention Mechanisms** — Self-attention in CNN backbone |
| Phase 4 | **Gene Expression Integration** — Multi-modal fusion (MRI + genomics) |
| Phase 5 | **Generative AI** — Synthetic MRI generation with GANs/Diffusion |
| Phase 6 | **API Deployment** — FastAPI + Docker + Cloud Run |

---

## 📖 Key References

1. Selvaraju, R. R., et al. (2017). **Grad-CAM: Visual Explanations from Deep Networks.**
   *IEEE ICCV 2017.*

2. He, K., et al. (2016). **Deep Residual Learning for Image Recognition.** *CVPR 2016.*

3. Ioffe, S., & Szegedy, C. (2015). **Batch Normalization.** *ICML 2015.*

4. Simonyan, K., & Zisserman, A. (2015). **Very Deep Convolutional Networks (VGGNet).** *ICLR 2015.*

---

## 📄 Citation

If you use this code for research, please cite:

```bibtex
@misc{braintumordetection2024,
  title  = {Brain-Tumor-Detection-Using-MRI: An Explainable AI Framework for Brain Tumor Detection},
  author = {Ritika},
  year   = {2024},
  url    = {https://github.com/Ritika1014/Brain-Tumor-Detection-Using-MRI}
}
```

---

<div align="center">
Made with ❤️ for Brain Tumor Detection & Explainable AI
</div>
