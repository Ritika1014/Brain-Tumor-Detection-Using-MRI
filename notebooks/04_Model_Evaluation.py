"""
================================================================================
Notebook 04: Model Evaluation — Metrics, Confusion Matrix & ROC Curves
================================================================================

This notebook covers:
  1. Loading the best saved model from checkpoint
  2. Running inference on the held-out test set
  3. Computing: Accuracy, Precision, Recall, F1, Kappa, ROC-AUC
  4. Plotting the Confusion Matrix (normalized + counts)
  5. Plotting multi-class ROC curves (One-vs-Rest)
  6. Generating and saving the full classification report
  7. Interpreting results for IEEE paper reporting

Run after completing Notebook 03 (model must be trained).
================================================================================
"""

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.utils.helpers import ensure_dirs
from src.utils.logger import get_logger

logger = get_logger(__name__)
ensure_dirs(*Config.all_output_dirs())
print("✅ Environment ready.")

# ── Cell 2: Load Best Model ───────────────────────────────────────────────────
"""
Load the best model saved during training by ModelCheckpoint.
This is the model with the highest validation accuracy.
"""
from src.training.trainer import load_best_model

print(f"📦 Loading model from: {Config.BEST_MODEL_PATH}")
model = load_best_model()
print(f"✅ Model loaded.")
print(f"   Input shape  : {model.input_shape}")
print(f"   Output shape : {model.output_shape}")
print(f"   Parameters   : {model.count_params():,}")

# ── Cell 3: Build Test Dataset ────────────────────────────────────────────────
from src.preprocessing.data_loader import build_tf_dataset

test_ds = build_tf_dataset(Config.TEST_DIR, split="test")
print(f"✅ Test dataset ready: {len(test_ds)} batches.")

# ── Cell 4: Compute All Metrics ───────────────────────────────────────────────
"""
Compute all evaluation metrics in one call.
Returns: accuracy, precision, recall, F1, Kappa, AUC, y_true, y_pred, y_prob
"""
from src.evaluation.metrics import compute_metrics, classification_report_df

print("\n📊 Computing evaluation metrics...")
metrics = compute_metrics(model, test_ds, class_names=Config.CLASS_NAMES)

# Display scalar metrics in a clean table
scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
print("\n┌─────────────────────────────────────────────┐")
print("│           EVALUATION RESULTS                │")
print("├─────────────────────────────────────────────┤")
for name, value in scalar_metrics.items():
    print(f"│  {name:<30} {value:>10.4f} │")
print("└─────────────────────────────────────────────┘")

# ── Cell 5: Classification Report ─────────────────────────────────────────────
"""
Per-class breakdown of Precision, Recall, F1-Score, Support.
This is the primary table for IEEE paper Table II.
"""
print("\n📋 Classification Report:")
print(metrics["report_str"])

# Convert to DataFrame for display
report_df = classification_report_df(metrics, Config.CLASS_NAMES)
print("\nDataFrame version:")
print(report_df.to_string())

# ── Cell 6: Confusion Matrix ──────────────────────────────────────────────────
"""
Two versions:
  1. Normalized (fraction per row) — shows per-class error rates
  2. Raw counts — shows actual sample counts
"""
from src.evaluation.visualizer import plot_confusion_matrix

print("\n📊 Plotting Confusion Matrices...")

# Normalized
fig_cm_norm = plot_confusion_matrix(
    metrics,
    class_names=Config.CLASS_NAMES,
    normalize=True,
    save_path=Config.CONFUSION_MATRIX_DIR / "confusion_matrix_normalized.png",
    title="Normalized Confusion Matrix — Brain Tumor Classification",
)
plt.show()

# Raw counts
fig_cm_counts = plot_confusion_matrix(
    metrics,
    class_names=Config.CLASS_NAMES,
    normalize=False,
    save_path=Config.CONFUSION_MATRIX_DIR / "confusion_matrix_counts.png",
    title="Confusion Matrix — Sample Counts",
)
plt.show()
print("✅ Confusion matrices saved.")

# ── Cell 7: ROC Curves ────────────────────────────────────────────────────────
"""
Multi-class ROC curves using One-vs-Rest (OvR) binarization.
Each class has its own curve; macro-average is the summary metric.
AUC = 1.0 is perfect; AUC = 0.5 is random chance.
"""
from src.evaluation.visualizer import plot_roc_curve

fig_roc = plot_roc_curve(
    metrics,
    class_names=Config.CLASS_NAMES,
    save_path=Config.PLOTS_DIR / "roc_curves.png",
    title="Multi-Class ROC Curves (One-vs-Rest) — Brain Tumor Detection",
)
plt.show()
print("✅ ROC curve saved.")

# ── Cell 8: Save All Reports ──────────────────────────────────────────────────
from src.evaluation.visualizer import save_classification_report
from src.utils.helpers import save_json

# Save reports
save_classification_report(
    metrics,
    csv_path=Config.METRICS_DIR / "classification_report.csv",
    json_path=Config.METRICS_DIR / "classification_report.json",
    txt_path=Config.METRICS_DIR / "classification_report.txt",
)

# Save scalar metrics
save_json(scalar_metrics, Config.METRICS_DIR / "scalar_metrics.json")
print("✅ All reports saved.")

# ── Cell 9: Per-Class Analysis ────────────────────────────────────────────────
"""
Identify which classes the model finds hardest to classify.
High off-diagonal values in the confusion matrix indicate confusion between classes.
"""
y_true = metrics["y_true"]
y_pred = metrics["y_pred"]

print("\n📈 Per-Class Accuracy:")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
for i, cls_name in enumerate(Config.CLASS_NAMES):
    cls_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"   {cls_name:<15} : {cls_acc:.4f} ({cls_acc:.2%})")

print("""
================================================================================
  Notebook 04 Complete!

  Next steps:
  → Open 05_Explainable_AI_GradCAM.ipynb for visual explanations.

  Key outputs:
  ✅ outputs/confusion_matrix/    ← Normalized + count CMs
  ✅ outputs/plots/roc_curves.png ← Multi-class ROC
  ✅ outputs/metrics/             ← CSV, JSON, TXT reports
================================================================================
""")
