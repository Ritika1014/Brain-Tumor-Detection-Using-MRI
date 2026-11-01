# =============================================================================
# src/training/__init__.py
# =============================================================================
"""Training subpackage: model training loop and persistence."""

from src.training.trainer import train_model, load_best_model

__all__ = ["train_model", "load_best_model"]
