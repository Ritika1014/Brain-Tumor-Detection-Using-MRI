# =============================================================================
# src/explainability/__init__.py
# =============================================================================
"""Explainability subpackage: Grad-CAM implementation for XAI."""

from src.explainability.gradcam import GradCAM, generate_gradcam_heatmap, overlay_heatmap, batch_gradcam

__all__ = ["GradCAM", "generate_gradcam_heatmap", "overlay_heatmap", "batch_gradcam"]
