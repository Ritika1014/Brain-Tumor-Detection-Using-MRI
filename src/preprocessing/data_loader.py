# =============================================================================
# src/preprocessing/data_loader.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Dataset loading and TensorFlow data pipeline construction.

This module handles all dataset I/O operations:
- Collecting image paths and labels from the folder structure.
- Splitting the training folder into train / validation subsets.
- Building optimized ``tf.data.Dataset`` pipelines for training, validation,
  and testing with prefetching and caching.

The folder structure expected is:
    Training/
        glioma/          ← 0
        meningioma/      ← 1
        notumor/         ← 2
        pituitary/       ← 3
    Testing/
        glioma/
        ...

Usage
-----
    from src.preprocessing.data_loader import build_tf_dataset

    train_ds, val_ds = build_tf_dataset(Config.TRAIN_DIR, split="train")
    test_ds = build_tf_dataset(Config.TEST_DIR, split="test")
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported image file extensions
_SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Path Collection ───────────────────────────────────────────────────────────


def load_dataset_paths(
    data_dir: Path,
    class_names: list[str] | None = None,
    shuffle: bool = True,
    seed: int = Config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Scan a dataset directory and return a DataFrame of (filepath, label, class_name).

    Parameters
    ----------
    data_dir : Path
        Root directory containing one subdirectory per class.
        Example: ``data/raw/Training/``
    class_names : list[str] or None, optional
        Ordered list of class names. If None, uses ``Config.CLASS_NAMES``.
    shuffle : bool, optional
        Whether to shuffle the resulting DataFrame. Defaults to True.
    seed : int, optional
        Random seed for shuffling. Defaults to ``Config.RANDOM_SEED``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``filepath``, ``label`` (int), ``class_name`` (str).

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist.
    ValueError
        If no images are found in ``data_dir``.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    if class_names is None:
        from src.utils.helpers import detect_class_names, save_class_mapping
        class_names = detect_class_names(data_dir)
        if class_names:
            save_class_mapping(class_names)
        else:
            class_names = Config.CLASS_NAMES


    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    records: list[dict] = []

    for class_name in class_names:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            logger.warning("Class directory not found, skipping: %s", class_dir)
            continue

        for img_path in _iter_images(class_dir):
            records.append(
                {
                    "filepath": str(img_path),
                    "label": class_to_idx[class_name],
                    "class_name": class_name,
                }
            )

    if not records:
        raise ValueError(f"No images found in: {data_dir}")

    df = pd.DataFrame(records)

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(
        "Loaded %d images from '%s' across %d classes.",
        len(df),
        data_dir.name,
        df["label"].nunique(),
    )
    return df


def _iter_images(directory: Path) -> Generator[Path, None, None]:
    """
    Yield image file paths from a directory (non-recursive).

    Parameters
    ----------
    directory : Path
        Directory to scan.

    Yields
    ------
    Path
        Path to each image file.
    """
    for file_path in directory.iterdir():
        if file_path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            yield file_path


# ── Class Distribution ────────────────────────────────────────────────────────


def get_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-class image counts and percentage distribution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by ``load_dataset_paths()``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``class_name``, ``count``, ``percentage``.
    """
    dist = (
        df.groupby("class_name")
        .size()
        .reset_index(name="count")
        .sort_values("class_name")
    )
    dist["percentage"] = (dist["count"] / dist["count"].sum() * 100).round(2)
    return dist


# ── TF Dataset Builder ────────────────────────────────────────────────────────


def build_tf_dataset(
    data_dir: Path,
    split: str = "train",
    validation_split: float = Config.VALIDATION_SPLIT,
    batch_size: int = Config.BATCH_SIZE,
    image_size: tuple[int, int] = Config.IMAGE_SIZE,
    augment: bool = False,
    seed: int = Config.RANDOM_SEED,
) -> tf.data.Dataset | tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Build an optimized ``tf.data.Dataset`` from a class-structured directory.

    For ``split="train"``, returns a ``(train_dataset, val_dataset)`` tuple.
    For ``split="test"``, returns a single ``test_dataset``.

    Parameters
    ----------
    data_dir : Path
        Root data directory (``Training/`` or ``Testing/``).
    split : str, optional
        One of ``"train"`` or ``"test"``. Defaults to ``"train"``.
    validation_split : float, optional
        Fraction of training data to reserve for validation.
        Only used when ``split="train"``. Defaults to ``Config.VALIDATION_SPLIT``.
    batch_size : int, optional
        Number of images per batch. Defaults to ``Config.BATCH_SIZE``.
    image_size : tuple[int, int], optional
        Target (height, width) to resize all images. Defaults to ``Config.IMAGE_SIZE``.
    augment : bool, optional
        Whether to apply data augmentation (training only). Defaults to False.
    seed : int, optional
        Random seed for dataset shuffling. Defaults to ``Config.RANDOM_SEED``.

    Returns
    -------
    tf.data.Dataset or tuple[tf.data.Dataset, tf.data.Dataset]
        - If ``split="train"``: ``(train_ds, val_ds)``
        - If ``split="test"``: ``test_ds``

    Notes
    -----
    The pipeline applies:
    1. Image decoding and resizing
    2. Normalization to [0, 1] (``float32``)
    3. Optional augmentation (training only)
    4. Prefetching (``AUTOTUNE``) for performance

    Examples
    --------
    >>> train_ds, val_ds = build_tf_dataset(Config.TRAIN_DIR, split="train", augment=True)
    >>> test_ds = build_tf_dataset(Config.TEST_DIR, split="test")
    """
    data_dir = Path(data_dir)
    AUTOTUNE = tf.data.AUTOTUNE

    # Detect class names directly from dataset directory
    detected_classes = sorted(
        [folder.name for folder in data_dir.iterdir() if folder.is_dir()]
    ) if data_dir.exists() else []

    if detected_classes:
        from src.utils.helpers import save_class_mapping
        save_class_mapping(detected_classes)
        class_names = detected_classes
    else:
        class_names = Config.CLASS_NAMES

    logger.info("Building tf.data.Dataset from '%s' (split=%s, classes=%s).", data_dir.name, split, class_names)

    if split == "train":
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(data_dir),
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            class_names=class_names,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(data_dir),
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            class_names=class_names,
        )
        # Rescale pixel values to [0, 1]
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        train_ds = train_ds.map(
            lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE
        )
        val_ds = val_ds.map(
            lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE
        )

        # ── OPTIMIZATION: Cache BEFORE Augmentation & Shuffle ────────────────
        # Caching normalized images in RAM after decoding avoids re-reading
        # from disk and re-decoding JPEGs on every single epoch.
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

        # Shuffle cached in-memory tensors
        train_ds = train_ds.shuffle(buffer_size=1000, seed=seed)

        # Apply augmentation on cached in-memory batches dynamically
        if augment:
            from src.preprocessing.augmentation import get_augmentation_layer

            aug_layer = get_augmentation_layer()
            train_ds = train_ds.map(
                lambda x, y: (aug_layer(x, training=True), y),
                num_parallel_calls=AUTOTUNE,
            )
            logger.info("Fast in-memory data augmentation enabled on training set.")

        # Prefetch upcoming batches asynchronously
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

        logger.info(
            "Train dataset: %d batches | Val dataset: %d batches.",
            len(train_ds),
            len(val_ds),
        )
        return train_ds, val_ds

    else:  # split == "test"
        test_ds = tf.keras.utils.image_dataset_from_directory(
            str(data_dir),
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            class_names=class_names,
            shuffle=False,  # Keep order for consistent evaluation
        )
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        test_ds = test_ds.map(
            lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE
        )
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        logger.info("Test dataset: %d batches.", len(test_ds))
        return test_ds


# ── Label Extraction ──────────────────────────────────────────────────────────


def extract_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """
    Extract all true labels from a ``tf.data.Dataset`` into a NumPy array.

    Iterates through all batches to collect one-hot encoded labels, then
    converts them to integer class indices.

    Parameters
    ----------
    dataset : tf.data.Dataset
        A batched dataset with ``(images, one_hot_labels)`` structure.

    Returns
    -------
    np.ndarray
        1-D array of integer class indices, shape ``(num_samples,)``.
    """
    all_labels = []
    for _, labels in dataset:
        all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels, axis=0)

    # Convert one-hot to integer labels
    if all_labels.ndim == 2:
        all_labels = np.argmax(all_labels, axis=1)

    return all_labels
