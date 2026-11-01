#!/usr/bin/env python3
# =============================================================================
# main.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Main CLI entry point for the full training and evaluation pipeline.

This script orchestrates the complete pipeline in a single command:
    1. Setup  — seed + directory creation
    2. Load   — build TF datasets from raw MRI data
    3. Train  — Custom CNN with callbacks
    4. Eval   — metrics, confusion matrix, ROC curves, classification report
    5. XAI    — Grad-CAM visualizations for all classes
    6. Export — Save all results to outputs/

Usage
-----
    # Run the full pipeline
    python main.py

    # Run specific stages
    python main.py --stage train
    python main.py --stage evaluate
    python main.py --stage gradcam

    # Use custom config overrides
    python main.py --epochs 30 --lr 0.0005 --batch-size 16

    # Show project configuration
    python main.py --show-config

Run Streamlit App separately:
    streamlit run app/app.py
"""

import sys
import time
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Ensure project root is in the path (for running from any directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.utils.helpers import ensure_dirs, save_json, set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file=Config.LOG_FILE)


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--stage",
    type=click.Choice(["all", "train", "evaluate", "gradcam"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Pipeline stage to run.",
)
@click.option("--epochs", type=int, default=Config.EPOCHS, show_default=True, help="Training epochs.")
@click.option("--lr", type=float, default=Config.LEARNING_RATE, show_default=True, help="Learning rate.")
@click.option("--batch-size", type=int, default=Config.BATCH_SIZE, show_default=True, help="Batch size.")
@click.option("--no-augment", is_flag=True, default=False, help="Disable data augmentation.")
@click.option("--no-class-weights", is_flag=True, default=False, help="Disable class weights.")
@click.option("--tensorboard", is_flag=True, default=False, help="Enable TensorBoard logging.")
@click.option("--show-config", is_flag=True, default=False, help="Print config and exit.")
def main(
    stage: str,
    epochs: int,
    lr: float,
    batch_size: int,
    no_augment: bool,
    no_class_weights: bool,
    tensorboard: bool,
    show_config: bool,
) -> None:
    """
    From Genes to Decisions — Brain Tumor Detection Pipeline.

    Run the full ML pipeline: preprocess → train → evaluate → explain.
    """
    if show_config:
        Config.display()
        return

    _print_banner()
    t_start = time.perf_counter()

    # ── 0. Setup ─────────────────────────────────────────────────────────────
    logger.info("Stage 0/4: Setup")
    set_seed(Config.RANDOM_SEED)
    ensure_dirs(*Config.all_output_dirs())
    logger.info("All output directories created.")

    # ── Check dataset exists ───────────────────────────────────────────────
    if not Config.TRAIN_DIR.exists():
        logger.error(
            "Training data not found at: %s\n"
            "Please download the Brain Tumor MRI dataset from Kaggle and place it at:\n"
            "  data/raw/Training/\n"
            "  data/raw/Testing/",
            Config.TRAIN_DIR,
        )
        sys.exit(1)

    # ── 1. Preprocessing ─────────────────────────────────────────────────────
    logger.info("Stage 1/4: Building TF Datasets")
    from src.preprocessing.data_loader import build_tf_dataset

    train_ds, val_ds = build_tf_dataset(
        Config.TRAIN_DIR,
        split="train",
        batch_size=batch_size,
        augment=not no_augment,
    )
    test_ds = build_tf_dataset(
        Config.TEST_DIR,
        split="test",
        batch_size=batch_size,
    )
    logger.info("Datasets ready: train=%d batches, val=%d batches, test=%d batches.",
                len(train_ds), len(val_ds), len(test_ds))

    # ── 2. Training ──────────────────────────────────────────────────────────
    if stage in ("all", "train"):
        logger.info("Stage 2/4: Training Custom CNN")
        from src.training.trainer import train_model

        history = train_model(
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=epochs,
            learning_rate=lr,
            use_class_weights=not no_class_weights,
            enable_tensorboard=tensorboard,
        )

        # Save training history
        save_json(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            Config.METRICS_DIR / "training_history.json",
        )

        # Plot training curves
        from src.evaluation.visualizer import plot_training_history
        fig = plot_training_history(
            history,
            save_path=Config.PLOTS_DIR / "training_history.png",
        )
        fig.savefig(Config.PLOTS_DIR / "training_history.pdf", bbox_inches="tight")
        logger.info("Training history plots saved.")

    # ── 3. Evaluation & Confusion Matrix ─────────────────────────────────────
    if stage in ("all", "train", "evaluate"):
        logger.info("Stage 3/4: Evaluating on Test Set & Generating Confusion Matrix")
        from src.evaluation.metrics import compute_metrics
        from src.evaluation.visualizer import (
            plot_confusion_matrix,
            plot_roc_curve,
            save_classification_report,
        )
        from src.training.trainer import load_best_model

        model = load_best_model()
        metrics = compute_metrics(model, test_ds)

        # Confusion Matrix
        plot_confusion_matrix(
            metrics,
            normalize=True,
            save_path=Config.CONFUSION_MATRIX_DIR / "confusion_matrix_normalized.png",
        )
        plot_confusion_matrix(
            metrics,
            normalize=False,
            save_path=Config.CONFUSION_MATRIX_DIR / "confusion_matrix_counts.png",
        )

        # ROC Curve
        plot_roc_curve(
            metrics,
            save_path=Config.PLOTS_DIR / "roc_curves.png",
        )

        # Classification Report
        save_classification_report(
            metrics,
            csv_path=Config.METRICS_DIR / "classification_report.csv",
            json_path=Config.METRICS_DIR / "classification_report.json",
            txt_path=Config.METRICS_DIR / "classification_report.txt",
        )

        # Scalar metrics summary
        scalar_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, float)
        }
        save_json(scalar_metrics, Config.METRICS_DIR / "scalar_metrics.json")
        logger.info("Evaluation complete. Results saved to: %s", Config.METRICS_DIR)

    # ── 4. Grad-CAM ──────────────────────────────────────────────────────────
    if stage in ("all", "gradcam"):
        logger.info("Stage 4/4: Generating Grad-CAM Explanations")
        from src.explainability.gradcam import batch_gradcam
        from src.training.trainer import load_best_model

        model = load_best_model()
        batch_gradcam(
            model=model,
            dataset=test_ds,
            num_samples_per_class=3,
            save_dir=Config.GRADCAM_DIR,
        )
        logger.info("Grad-CAM visualizations saved to: %s", Config.GRADCAM_DIR)

    # ── Done ─────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE in %.1f seconds (%.1f min).", elapsed, elapsed / 60)
    logger.info("  Results in: %s", Config.OUTPUTS_DIR)
    logger.info("  Launch app:  streamlit run app/app.py")
    logger.info("=" * 60)


def _print_banner() -> None:
    """Print ASCII art project banner to console."""
    banner = r"""
+------------------------------------------------------------------+
|     BRAIN TUMOR DETECTION SYSTEM                                 |
|     Deep Learning Based MRI Classification                       |
|     B.Tech Major Project                                         |
+------------------------------------------------------------------+
    """
    try:
        print(banner)
    except Exception:
        pass


if __name__ == "__main__":
    main()
