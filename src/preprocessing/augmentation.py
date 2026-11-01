# =============================================================================
# src/preprocessing/augmentation.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Data augmentation module.

Medical imaging datasets are typically small and class-imbalanced. Data
augmentation artificially increases the effective training set size by applying
random geometric and photometric transformations to input images.

This module provides:
- A Keras augmentation layer for use inside the tf.data pipeline.
- A visualization function to compare original vs. augmented MRI images.

Usage
-----
    from src.preprocessing.augmentation import get_augmentation_layer

    augmentation = get_augmentation_layer()
    augmented_batch = augmentation(image_batch, training=True)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Augmentation Layer Factory ────────────────────────────────────────────────


def get_augmentation_layer(
    rotation_factor: float = 0.05,    # ≈ ±9° rotation (preserves brain orientation)
    zoom_factor: float = 0.05,        # ±5% random zoom (simulates scanner field-of-view variations)
    flip_mode: str = "horizontal",    # Horizontal flip (anatomically valid due to left-right brain symmetry)
    translation_factor: float = 0.05, # ±5% shift (simulates slight patient positioning offset)
) -> tf.keras.Sequential:
    """
    Build a Keras Sequential augmentation layer with medically appropriate transformations.

    Medical Rationale for Augmentation Parameters:
    ----------------------------------------------
    1. Horizontal Flip: Anatomically plausible because human brain hemispheres exhibit
       near axial symmetry. Vertical flip is excluded because up-down inversion creates
       anatomically impossible head orientations.
    2. Small Rotation (±9°): Accounts for slight tilt during MRI positioning without
       disrupting spatial anatomical landmarks.
    3. Small Zoom (±5%): Simulates slight scanner resolution/distance variations without
       distorting intracranial structures.
    4. Small Translation (±5%): Handles minor alignment offsets within the MRI frame.
    5. Value Clipping [0, 1]: Prevents out-of-bound pixel values following transformation.
    """
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode=flip_mode, name="aug_horizontal_flip"),
            tf.keras.layers.RandomRotation(factor=rotation_factor, fill_mode="nearest", name="aug_rotation"),
            tf.keras.layers.RandomZoom(height_factor=(-zoom_factor, zoom_factor), fill_mode="nearest", name="aug_zoom"),
            tf.keras.layers.RandomTranslation(
                height_factor=(-translation_factor, translation_factor),
                width_factor=(-translation_factor, translation_factor),
                fill_mode="nearest",
                name="aug_translation",
            ),
            # Clamp pixel values to valid normalized range [0.0, 1.0]
            tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(x, 0.0, 1.0),
                name="clip_to_valid_range",
            ),
        ],
        name="data_augmentation",
    )

    logger.info(
        "Medically conservative augmentation layer created: rotation=%.2f, zoom=%.2f, translation=%.2f, flip=%s.",
        rotation_factor,
        zoom_factor,
        translation_factor,
        flip_mode,
    )
    return augmentation


# ── Visualization ─────────────────────────────────────────────────────────────


def visualize_augmentation(
    dataset: tf.data.Dataset,
    augmentation_layer: tf.keras.Sequential,
    num_samples: int = 8,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Display a side-by-side comparison of original vs. augmented MRI images.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Batched dataset to sample images from.
    augmentation_layer : tf.keras.Sequential
        The augmentation layer returned by ``get_augmentation_layer()``.
    num_samples : int, optional
        Number of images to display. Defaults to 8.
    save_path : str or None, optional
        If provided, saves the figure to this path. Defaults to None.

    Returns
    -------
    plt.Figure
        Matplotlib figure with two rows: originals and augmented versions.

    Notes
    -----
    Images are assumed to be normalized to [0, 1]. They are clipped before
    display to prevent rendering artifacts.
    """
    # Extract one batch
    for images, _ in dataset.take(1):
        sample_images = images[:num_samples].numpy()
        break

    # Apply augmentation
    aug_images = augmentation_layer(
        tf.constant(sample_images), training=True
    ).numpy()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_samples,
        figsize=(num_samples * 2, 5),
        constrained_layout=True,
    )
    fig.suptitle(
        "Data Augmentation: Original (top) vs. Augmented (bottom)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for i in range(num_samples):
        # Original
        axes[0, i].imshow(np.clip(sample_images[i], 0, 1))
        axes[0, i].set_title("Original", fontsize=8)
        axes[0, i].axis("off")

        # Augmented
        axes[1, i].imshow(np.clip(aug_images[i], 0, 1))
        axes[1, i].set_title("Augmented", fontsize=8)
        axes[1, i].axis("off")

    if save_path:
        fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
        logger.info("Augmentation visualization saved to: %s", save_path)

    return fig
