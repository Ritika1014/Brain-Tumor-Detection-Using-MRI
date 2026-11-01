# =============================================================================
# src/evaluation/metrics.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Model evaluation metrics module.

Provides functions to compute standard classification metrics from model
predictions, supporting IEEE paper reporting requirements.

Metrics computed
----------------
- Accuracy (overall)
- Per-class Precision, Recall, F1-Score
- Macro-averaged Precision, Recall, F1-Score
- Weighted-averaged Precision, Recall, F1-Score
- Cohen's Kappa (inter-rater reliability metric)
- ROC-AUC (macro, weighted, per-class)

Usage
-----
    from src.evaluation.metrics import compute_metrics

    metrics = compute_metrics(model, test_ds)
    print(metrics["accuracy"], metrics["macro_f1"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference on an entire dataset and collect results.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    dataset : tf.data.Dataset
        Batched, preprocessed dataset with ``(images, one_hot_labels)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - ``y_true`` : True class indices, shape ``(N,)``.
        - ``y_pred`` : Predicted class indices, shape ``(N,)``.
        - ``y_prob`` : Predicted probabilities (softmax), shape ``(N, C)``.
    """
    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        all_probs.append(probs)
        all_true.append(labels.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true_onehot = np.concatenate(all_true, axis=0)

    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_prob, axis=1)

    logger.debug("Predictions collected: %d samples.", len(y_true))
    return y_true, y_pred, y_prob


def compute_metrics(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str] | None = None,
) -> dict[str, float | str | np.ndarray]:
    if class_names is None:
        class_names = Config.CLASS_NAMES

    logger.info("Computing evaluation metrics for classes: %s", class_names)
    y_true, y_pred, y_prob = get_predictions(model, dataset)


    # ── Scalar Metrics ────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    # ── ROC-AUC (multi-class OvR) ─────────────────────────────────────────────
    try:
        auc_macro = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro"
        )
        auc_weighted = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="weighted"
        )
    except ValueError as exc:
        logger.warning("Could not compute ROC-AUC: %s", exc)
        auc_macro = auc_weighted = float("nan")

    # ── Full Classification Report ────────────────────────────────────────────
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "kappa": float(kappa),
        "auc_macro": float(auc_macro),
        "auc_weighted": float(auc_weighted),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report_str": report_str,
        "report_dict": report_dict,
    }

    # ── Log Summary ───────────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("─" * 50)
    logger.info("  Accuracy:          %.4f", accuracy)
    logger.info("  Macro Precision:   %.4f", macro_precision)
    logger.info("  Macro Recall:      %.4f", macro_recall)
    logger.info("  Macro F1-Score:    %.4f", macro_f1)
    logger.info("  Weighted F1-Score: %.4f", weighted_f1)
    logger.info("  Cohen's Kappa:     %.4f", kappa)
    logger.info("  AUC (Macro):       %.4f", auc_macro)
    logger.info("  AUC (Weighted):    %.4f", auc_weighted)
    logger.info("─" * 50)
    logger.info("\n%s", report_str)

    return metrics


def classification_report_df(
    metrics: dict,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    if class_names is None:
        class_names = Config.CLASS_NAMES

    """
    Convert the classification report dictionary to a formatted DataFrame.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary returned by ``compute_metrics()``.
    class_names : list[str], optional
        Ordered class names. Defaults to ``Config.CLASS_NAMES``.

    Returns
    -------
    pd.DataFrame
        DataFrame with per-class and summary metrics.
        Columns: ``precision``, ``recall``, ``f1-score``, ``support``.
        Rows: one per class, plus ``accuracy``, ``macro avg``, ``weighted avg``.
    """
    report_dict = metrics["report_dict"]
    df = pd.DataFrame(report_dict).T
    df = df.round(4)
    logger.debug("Classification report DataFrame created: shape %s.", df.shape)
    return df
