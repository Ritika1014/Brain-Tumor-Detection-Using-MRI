"""
================================================================================
Notebook 05: Explainable AI — Grad-CAM Visualization
================================================================================

This notebook covers:
  1. Theory behind Grad-CAM (mathematical formulation)
  2. Implementing Grad-CAM using the GradCAM class
  3. Visualizing heatmaps for correctly classified samples
  4. Per-class Grad-CAM analysis (all 4 tumor classes)
  5. Visualizing Grad-CAM for misclassified samples
  6. Batch generation of Grad-CAM outputs for all classes
  7. Interpreting clinical significance of the heatmaps

================================================================================
"""

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.utils.helpers import ensure_dirs
from src.utils.logger import get_logger

logger = get_logger(__name__)
ensure_dirs(*Config.all_output_dirs())
print("✅ Environment ready.")

# ── Cell 2: Grad-CAM Theory ──────────────────────────────────────────────────
"""
GRAD-CAM THEORY (Selvaraju et al., 2017)
=========================================
Given a CNN and an input image I, Grad-CAM generates a heatmap L^c that
highlights regions important for classifying the image as class c.

Algorithm:
-----------
1. Forward pass → record conv activations A^k and class score y^c
2. Compute gradients: ∂y^c / ∂A^k_{ij}
3. Global average pool gradients: α^c_k = (1/Z) ΣΣ ∂y^c / ∂A^k_{ij}
4. Weighted sum + ReLU: L^c = ReLU(Σ_k α^c_k A^k)
5. Resize L^c to input image size, apply colormap, overlay

Key Insight:
  - α^c_k = importance weight for filter k in predicting class c
  - Positive α^c_k → filter k fires more → higher class score
  - ReLU discards negative activations (only keep relevant regions)
"""
print("📚 Grad-CAM Theory:")
print("   Reference: Selvaraju et al., IEEE ICCV 2017")
print("   Layer used:", Config.GRADCAM_LAYER_NAME)

# ── Cell 3: Load Model & Test Dataset ────────────────────────────────────────
from src.training.trainer import load_best_model
from src.preprocessing.data_loader import build_tf_dataset

model = load_best_model()
test_ds = build_tf_dataset(Config.TEST_DIR, split="test")
print(f"✅ Model loaded. Test dataset: {len(test_ds)} batches.")

# ── Cell 4: Initialize GradCAM ───────────────────────────────────────────────
from src.explainability.gradcam import GradCAM

cam = GradCAM(model, layer_name=Config.GRADCAM_LAYER_NAME)
print(f"✅ GradCAM initialized on layer: '{Config.GRADCAM_LAYER_NAME}'")

# ── Cell 5: Single Image Grad-CAM Example ────────────────────────────────────
"""
Test GradCAM on a single image from the test dataset.
We take the first batch and use the first image.
"""
for images, labels in test_ds.take(1):
    sample_img = images[0].numpy()         # (224, 224, 3) float [0,1]
    sample_label = np.argmax(labels[0])

# Compute heatmap
heatmap = cam.compute_heatmap(
    np.expand_dims(sample_img, 0),
    class_idx=int(sample_label),
)

# Create overlay
original_uint8 = (sample_img * 255).astype(np.uint8)
overlay = cam.overlay_on_image(original_uint8, heatmap)

# Plot three-panel visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
fig.suptitle(
    f"Grad-CAM Explanation — Class: {Config.CLASS_NAMES[sample_label]}",
    fontsize=13, fontweight="bold"
)
axes[0].imshow(original_uint8);         axes[0].set_title("Original MRI"); axes[0].axis("off")
im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
axes[1].set_title("Grad-CAM Heatmap");  axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
axes[2].imshow(overlay);               axes[2].set_title("Overlay"); axes[2].axis("off")
plt.show()
print(f"✅ Heatmap computed. Shape: {heatmap.shape}")

# ── Cell 6: Per-Class Grad-CAM Analysis ──────────────────────────────────────
"""
Collect one correctly classified sample per class and visualize.
This shows HOW the model differentiates each tumor type.
"""
from src.explainability.gradcam import batch_gradcam

print("\n🔍 Generating per-class Grad-CAM visualizations...")
figures = batch_gradcam(
    model=model,
    dataset=test_ds,
    class_names=Config.CLASS_NAMES,
    num_samples_per_class=2,
    layer_name=Config.GRADCAM_LAYER_NAME,
    save_dir=Config.GRADCAM_DIR,
)

for fig in figures:
    plt.show()

print(f"✅ Generated {len(figures)} Grad-CAM visualizations.")

# ── Cell 7: Analyze Misclassified Samples ────────────────────────────────────
"""
Applying Grad-CAM to misclassified samples helps understand WHY the model
made mistakes — a critical analysis for the IEEE paper discussion section.
"""
misclassified_imgs = []
for images, labels in test_ds:
    preds = model.predict(images.numpy(), verbose=0)
    true_labels = np.argmax(labels.numpy(), axis=1)
    pred_labels = np.argmax(preds, axis=1)

    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        if true != pred and len(misclassified_imgs) < 3:
            misclassified_imgs.append({
                "image": images[i].numpy(),
                "true": true,
                "pred": pred,
                "probs": preds[i],
            })

if misclassified_imgs:
    print(f"\n⚠️  Found {len(misclassified_imgs)} misclassified sample(s) — analyzing...")
    for sample in misclassified_imgs:
        img = sample["image"]
        true_cls = Config.CLASS_NAMES[sample["true"]]
        pred_cls = Config.CLASS_NAMES[sample["pred"]]
        conf = sample["probs"][sample["pred"]]

        heatmap = cam.compute_heatmap(
            np.expand_dims(img, 0), class_idx=sample["pred"]
        )
        overlay = cam.overlay_on_image((img * 255).astype(np.uint8), heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
        fig.suptitle(
            f"Misclassification Analysis\n"
            f"True: {true_cls.upper()} | Predicted: {pred_cls.upper()} ({conf:.1%})",
            fontsize=12, fontweight="bold", color="red"
        )
        axes[0].imshow((img * 255).astype(np.uint8)); axes[0].set_title("MRI"); axes[0].axis("off")
        im = axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Heatmap"); axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
        plt.show()
else:
    print("✅ No misclassified samples found in the sampled batches.")

# ── Cell 8: Grad-CAM Summary Analysis ────────────────────────────────────────
"""
Compute average Grad-CAM activation statistics per class.
Higher mean activation → model uses broader regions.
"""
print("\n📊 Grad-CAM Activation Statistics (per class):")
class_heatmaps = {cls: [] for cls in Config.CLASS_NAMES}

for images, labels in test_ds.take(5):     # Sample 5 batches
    preds = model.predict(images.numpy(), verbose=0)
    true_labels = np.argmax(labels.numpy(), axis=1)
    pred_labels = np.argmax(preds, axis=1)

    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        if true == pred:  # Only correctly classified
            hm = cam.compute_heatmap(
                np.expand_dims(images[i].numpy(), 0), class_idx=int(true)
            )
            class_heatmaps[Config.CLASS_NAMES[true]].append(hm)

print(f"\n{'Class':<15} {'Samples':>8} {'Mean Act':>12} {'Max Act':>10} {'Focus Area':>12}")
print("-" * 60)
for cls_name, heatmaps in class_heatmaps.items():
    if heatmaps:
        stacked = np.stack(heatmaps)
        mean_act = float(np.mean(stacked))
        max_act = float(np.max(stacked))
        # Focus area: fraction of pixels with activation > 0.5
        focus = float(np.mean(stacked > 0.5))
        print(f"{cls_name:<15} {len(heatmaps):>8} {mean_act:>12.4f} {max_act:>10.4f} {focus:>11.2%}")

print("""
================================================================================
  Notebook 05 Complete!

  Key Outputs:
  ✅ outputs/gradcam/              ← Per-class Grad-CAM overlays
  
  Clinical Interpretation:
  - Glioma:     Heatmap should focus on large, irregular mass regions
  - Meningioma: Heatmap should highlight boundary/edge of brain
  - Pituitary:  Heatmap should focus on central/inferior brain region
  - No Tumor:   Heatmap should show diffuse, low-activation patterns

  Next steps:
  → Open 06_Streamlit_App.ipynb for deployment guide.
================================================================================
""")
