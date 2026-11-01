# =============================================================================
# src/preprocessing/__init__.py
# =============================================================================
"""Preprocessing subpackage: data loading, augmentation, and image utilities."""

from src.preprocessing.data_loader import load_dataset_paths, build_tf_dataset, get_class_distribution
from src.preprocessing.augmentation import get_augmentation_layer, visualize_augmentation
from src.preprocessing.image_utils import (
    resize_and_normalize,
    load_and_preprocess_image,
    decode_image,
)

__all__ = [
    "load_dataset_paths",
    "build_tf_dataset",
    "get_class_distribution",
    "get_augmentation_layer",
    "visualize_augmentation",
    "resize_and_normalize",
    "load_and_preprocess_image",
    "decode_image",
]
