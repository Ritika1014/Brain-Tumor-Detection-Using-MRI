# =============================================================================
# src/deployment/__init__.py
# =============================================================================
"""Deployment subpackage: inference pipeline for production/Streamlit."""

from src.deployment.inference import TumorClassifier

__all__ = ["TumorClassifier"]
