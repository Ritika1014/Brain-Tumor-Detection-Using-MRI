# =============================================================================
# src/evaluation/visualizer.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Results visualization module.

Generates publication-quality plots for:
- Training accuracy & loss curves
- Confusion matrix heatmap
- Multi-class ROC curves (One-vs-Rest)
- Classification report export

All plotting functions accept an optional ``save_path`` argument to
save figures to disk. Figures are returned as ``matplotlib.figure.Figure``
objects for display in notebooks and Streamlit.

Usage
-----
    from src.evaluation.visualizer import plot_training_history, plot_confusion_matrix

    fig = plot_training_history(history)
    fig = plot_confusion_matrix(metrics, class_names=Config.CLASS_NAMES)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras import callbacks as keras_callbacks

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Style Configuration ───────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": Config.FIGURE_DPI,
    }
)

_COLOR_TRAIN = "#4C72B0"     # Blue for training metrics
_COLOR_VAL = "#DD8452"       # Orange for validation metrics


# ── Training History ──────────────────────────────────────────────────────────


def plot_training_history(
    history: keras_callbacks.History | dict,
    save_path: Path | str | None = None,
    title: str = "Custom CNN — Training History",
) -> plt.Figure:
    """
    Plot training and validation accuracy & loss curves side by side.

    Parameters
    ----------
    history : History or dict
        Keras History object or a dict with keys like ``"accuracy"``,
        ``"val_accuracy"``, ``"loss"``, ``"val_loss"``.
    save_path : Path or str or None, optional
        If provided, saves the figure here. Defaults to None.
    title : str, optional
        Figure title. Defaults to ``"Custom CNN — Training History"``.

    Returns
    -------
    plt.Figure
        The Matplotlib figure with two subplots.
    """
    # Accept both History objects and plain dicts
    hist = history.history if hasattr(history, "history") else history

    epochs = range(1, len(hist["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=Config.FIGURE_SIZE_LARGE, constrained_layout=True
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── Accuracy ──────────────────────────────────────────────────────────
    ax1.plot(epochs, hist["accuracy"], color=_COLOR_TRAIN, linewidth=2, label="Train Accuracy")
    ax1.plot(epochs, hist["val_accuracy"], color=_COLOR_VAL, linewidth=2,
             linestyle="--", label="Val Accuracy")
    ax1.set_title("Model Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))

    # Mark best validation accuracy
    best_val_acc = max(hist["val_accuracy"])
    best_epoch = hist["val_accuracy"].index(best_val_acc) + 1
    ax1.annotate(
        f"Best: {best_val_acc:.3f}\n(Epoch {best_epoch})",
        xy=(best_epoch, best_val_acc),
        xytext=(best_epoch + 1, best_val_acc - 0.05),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    # ── Loss ───────────────────────────────────────────────────────────────
    ax2.plot(epochs, hist["loss"], color=_COLOR_TRAIN, linewidth=2, label="Train Loss")
    ax2.plot(epochs, hist["val_loss"], color=_COLOR_VAL, linewidth=2,
             linestyle="--", label="Val Loss")
    ax2.set_title("Model Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        logger.info("Training history plot saved: %s", save_path)

    return fig


# ── Confusion Matrix ──────────────────────────────────────────────────────────


def plot_confusion_matrix(
    metrics: dict,
    class_names: list[str] | None = None,
    normalize: bool = True,
    save_path: Path | str | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    if class_names is None:
        class_names = Config.CLASS_NAMES

    from sklearn.metrics import confusion_matrix

    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        cbar_label = "Fraction of true class"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_MEDIUM)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=Config.CONFUSION_MATRIX_CMAP,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    # Add raw counts in each cell below the fraction
    if normalize:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.75,
                    f"n={cm[i, j]}",
                    ha="center", va="center",
                    fontsize=7, color="gray",
                )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        logger.info("Confusion matrix saved: %s", save_path)

    return fig


# ── ROC Curves ────────────────────────────────────────────────────────────────


def plot_roc_curve(
    metrics: dict,
    class_names: list[str] | None = None,
    save_path: Path | str | None = None,
    title: str = "ROC Curves (One-vs-Rest)",
) -> plt.Figure:
    if class_names is None:
        class_names = Config.CLASS_NAMES

    y_true = metrics["y_true"]
    y_prob = metrics["y_prob"]
    n_classes = len(class_names)

    # Binarize true labels for OvR computation
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Per-class color palette dynamically generated
    from src.utils.helpers import get_class_color_map
    color_map = get_class_color_map(class_names)
    colors = [color_map[name] for name in class_names]

    fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE_MEDIUM)

    fpr_all = np.linspace(0, 1, 200)
    tpr_interp_all = np.zeros((n_classes, 200))

    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        tpr_interp_all[i] = np.interp(fpr_all, fpr, tpr)

        ax.plot(
            fpr, tpr,
            color=color,
            linewidth=2,
            label=f"{cls_name.capitalize()} (AUC = {roc_auc:.3f})",
        )


    # Macro-average ROC
    mean_tpr = np.mean(tpr_interp_all, axis=0)
    mean_tpr[0] = 0.0
    macro_auc = auc(fpr_all, mean_tpr)
    ax.plot(
        fpr_all, mean_tpr,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label=f"Macro Average (AUC = {macro_auc:.3f})",
    )

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k:", linewidth=1.0, label="Random Classifier")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        logger.info("ROC curve plot saved: %s", save_path)

    return fig


# ── Classification Report Export ──────────────────────────────────────────────


def save_classification_report(
    metrics: dict,
    class_names: list[str] | None = None,
    csv_path: Path | str | None = None,
    json_path: Path | str | None = None,
    txt_path: Path | str | None = None,
) -> None:
    """
    Save the classification report to CSV, JSON, and/or plain text.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from ``compute_metrics()``.
    class_names : list[str], optional
        Class names. Defaults to ``Config.CLASS_NAMES``.
    csv_path : Path or str or None, optional
        Path to save the CSV file. Defaults to None.
    json_path : Path or str or None, optional
        Path to save the JSON file. Defaults to None.
    txt_path : Path or str or None, optional
        Path to save the plain text report. Defaults to None.
    """
    from src.evaluation.metrics import classification_report_df
    from src.utils.helpers import save_json

    if csv_path:
        df = classification_report_df(metrics, class_names)
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path)
        logger.info("Classification report CSV saved: %s", csv_path)

    if json_path:
        save_json(metrics["report_dict"], json_path)
        logger.info("Classification report JSON saved: %s", json_path)

    if txt_path:
        txt_path = Path(txt_path)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(metrics["report_str"], encoding="utf-8")
        logger.info("Classification report TXT saved: %s", txt_path)
