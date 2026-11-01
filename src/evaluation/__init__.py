# =============================================================================
# src/evaluation/__init__.py
# =============================================================================
"""Evaluation subpackage: metrics computation and result visualization."""

from src.evaluation.metrics import compute_metrics, classification_report_df
from src.evaluation.visualizer import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    save_classification_report,
)

__all__ = [
    "compute_metrics",
    "classification_report_df",
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "save_classification_report",
]
