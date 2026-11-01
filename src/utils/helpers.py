# =============================================================================
# src/utils/helpers.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
General-purpose helper utilities.

This module contains small, reusable functions that support the entire
pipeline — from reproducibility and directory management to JSON I/O and
metric formatting. Keeping these utilities here avoids code duplication
across preprocessing, training, and evaluation modules.

Usage
-----
    from src.utils.helpers import set_seed, ensure_dirs, save_json
"""

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Reproducibility ───────────────────────────────────────────────────────────


def set_seed(seed: int = 42) -> None:
    """
    Set all global random seeds for full reproducibility.

    Sets seeds for Python's ``random``, ``NumPy``, and TensorFlow.
    Also configures TensorFlow's determinism settings.

    Parameters
    ----------
    seed : int, optional
        The random seed to use. Defaults to 42.

    Notes
    -----
    GPU operations may still exhibit minor non-determinism due to
    floating-point precision. For strict reproducibility, consider
    disabling GPU or using ``tf.config.experimental.enable_op_determinism()``.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        logger.debug("TensorFlow random seed set to %d.", seed)
    except ImportError:
        logger.warning("TensorFlow not available; skipping TF seed.")

    logger.info("Global random seed set to %d.", seed)


# ── Directory Management ──────────────────────────────────────────────────────


def ensure_dirs(*paths: Path | str) -> None:
    """
    Create directories (and all parents) if they do not already exist.

    Parameters
    ----------
    *paths : Path or str
        One or more directory paths to create.

    Examples
    --------
    >>> ensure_dirs(Config.PLOTS_DIR, Config.METRICS_DIR)
    """
    for path in paths:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Directory ensured: %s", path)


# ── JSON I/O ─────────────────────────────────────────────────────────────────


def save_json(data: dict | list, filepath: Path | str, indent: int = 4) -> None:
    """
    Serialize a Python object to a JSON file.

    Parameters
    ----------
    data : dict or list
        The data to serialize.
    filepath : Path or str
        Destination file path. Parent directories are created automatically.
    indent : int, optional
        JSON indentation level. Defaults to 4.

    Raises
    ------
    TypeError
        If ``data`` contains non-serializable objects.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_serializer)

    logger.debug("JSON saved: %s", filepath)


def load_json(filepath: Path | str) -> Any:
    """
    Load and deserialize a JSON file.

    Parameters
    ----------
    filepath : Path or str
        Source file path.

    Returns
    -------
    Any
        Deserialized Python object (dict, list, etc.).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file content is not valid JSON.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.debug("JSON loaded: %s", filepath)
    return data


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for NumPy types.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    Any
        JSON-compatible Python type.

    Raises
    ------
    TypeError
        If the object type is not handled.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


# ── Metric Formatting ─────────────────────────────────────────────────────────


def format_metrics_table(metrics: dict[str, float]) -> str:
    """
    Format a metrics dictionary as a pretty-printed table string.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary mapping metric names to their float values.

    Returns
    -------
    str
        A formatted multi-line table string, suitable for console logging.

    Examples
    --------
    >>> metrics = {"accuracy": 0.9234, "f1_score": 0.9100}
    >>> print(format_metrics_table(metrics))
    ┌──────────────────────────┬──────────┐
    │ Metric                   │ Value    │
    ├──────────────────────────┼──────────┤
    │ accuracy                 │  0.9234  │
    │ f1_score                 │  0.9100  │
    └──────────────────────────┴──────────┘
    """
    header = f"{'Metric':<28} {'Value':>10}"
    separator = "-" * 40
    rows = [separator, header, separator]
    for name, value in metrics.items():
        rows.append(f"{name:<28} {value:>10.4f}")
    rows.append(separator)
    return "\n".join(rows)


# ── Class Utility & Dynamic Mapping ──────────────────────────────────────────


def format_class_label(class_name: str) -> str:
    """
    Format a raw dataset directory class name into a human-readable display label.

    Works dynamically for any dataset format (e.g. glioma -> Glioma Tumor,
    no_tumor -> No Tumor, notumor -> No Tumor).
    """
    name_lower = class_name.lower().replace("_", "")
    if "glioma" in name_lower:
        return "Glioma Tumor"
    elif "meningioma" in name_lower:
        return "Meningioma Tumor"
    elif "notumor" in name_lower or "normal" in name_lower:
        return "No Tumor"
    elif "pituitary" in name_lower:
        return "Pituitary Tumor"
    else:
        clean = class_name.replace("_", " ").title()
        return f"{clean} Tumor" if "tumor" not in clean.lower() else clean


def format_class_description(class_name: str) -> str:
    """
    Generate dynamic clinical description for a given class name.
    """
    name_lower = class_name.lower().replace("_", "")
    if "glioma" in name_lower:
        return (
            "Glioma is a common and aggressive brain tumor arising from glial cells. "
            "It accounts for ~30% of all brain tumors and ~80% of malignant tumors."
        )
    elif "meningioma" in name_lower:
        return (
            "Meningioma originates from the meninges — protective membranes surrounding "
            "the brain. Most are benign (WHO Grade I), but can be clinically significant."
        )
    elif "notumor" in name_lower or "normal" in name_lower:
        return (
            "No tumor detected. The MRI scan appears normal with no abnormal mass or "
            "lesion indicative of a brain tumor."
        )
    elif "pituitary" in name_lower:
        return (
            "Pituitary tumors arise in the pituitary gland at the base of the brain. "
            "Most are benign adenomas, but can cause hormonal and vision problems."
        )
    else:
        return f"Classification category: {format_class_label(class_name)}."


def detect_class_names(dataset_dir: Path | str | None = None) -> list[str]:
    """
    Automatically detect class names directly from subdirectories of dataset_dir.

    Classes are sorted alphabetically to match tf.keras conventions.
    If dataset_dir is missing or invalid, falls back to loading class_mapping.json.
    """
    from src.config import Config

    target_dir = Path(dataset_dir) if dataset_dir else Config.TRAIN_DIR
    if target_dir.exists():
        classes = sorted([folder.name for folder in target_dir.iterdir() if folder.is_dir()])
        if classes:
            return classes

    mapping_file = Config.CLASS_MAPPING_FILE
    if mapping_file.exists():
        try:
            mapping_data = load_json(mapping_file)
            if "class_names" in mapping_data and mapping_data["class_names"]:
                return mapping_data["class_names"]
        except Exception:
            pass

    return []


def save_class_mapping(class_names: list[str], mapping_file: Path | str | None = None) -> dict:
    """
    Save detected class names and generated label mappings to data/metadata/class_mapping.json.
    """
    from src.config import Config

    if mapping_file is None:
        mapping_file = Config.CLASS_MAPPING_FILE
    mapping_file = Path(mapping_file)

    mapping = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "idx_to_class": {str(idx): name for idx, name in enumerate(class_names)},
        "class_labels": {name: format_class_label(name) for name in class_names},
        "class_descriptions": {name: format_class_description(name) for name in class_names},
    }

    save_json(mapping, mapping_file, indent=4)
    logger.info("Class mapping saved to: %s (%d classes)", mapping_file, len(class_names))
    return mapping


def load_class_mapping(mapping_file: Path | str | None = None) -> dict:
    """
    Load saved class mapping from JSON file. Automatically detects from dataset if file missing.
    """
    from src.config import Config

    if mapping_file is None:
        mapping_file = Config.CLASS_MAPPING_FILE
    mapping_file = Path(mapping_file)

    if mapping_file.exists():
        try:
            return load_json(mapping_file)
        except Exception as exc:
            logger.warning("Failed to read class mapping from %s: %s", mapping_file, exc)

    detected = detect_class_names()
    if detected:
        return save_class_mapping(detected, mapping_file)

    return {
        "class_names": [],
        "num_classes": 0,
        "class_to_idx": {},
        "idx_to_class": {},
        "class_labels": {},
        "class_descriptions": {},
    }


def get_class_color_map(class_names: list[str]) -> dict[str, str]:
    """
    Assign a distinct hex color to each class for consistent visualization.
    """
    palette = [
        "#E63946",  # Red
        "#457B9D",  # Steel blue
        "#2DC653",  # Green
        "#F4A261",  # Orange
        "#9B59B6",  # Purple
        "#3498DB",  # Cyan
        "#E67E22",  # Amber
        "#1ABC9C",  # Teal
    ]
    color_map = {
        name: palette[i % len(palette)] for i, name in enumerate(class_names)
    }
    logger.debug("Color map generated for %d classes.", len(class_names))
    return color_map

