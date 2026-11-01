# =============================================================================
# src/preprocessing/image_utils.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Low-level image utility functions.

Provides functions for loading, decoding, resizing, and normalizing individual
MRI images. These are used primarily during inference (Streamlit app) and
Grad-CAM visualization, where images must be preprocessed one at a time
rather than in a tf.data pipeline batch.

Usage
-----
    from src.preprocessing.image_utils import load_and_preprocess_image

    image_tensor = load_and_preprocess_image("path/to/mri.jpg")
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── TF-Compatible Decoder ─────────────────────────────────────────────────────


def decode_image(
    image_path: str | Path,
    image_size: tuple[int, int] = Config.IMAGE_SIZE,
    normalize: bool = True,
) -> tf.Tensor:
    """
    Decode, resize, and optionally normalize an image file using TensorFlow ops.

    This function is designed to be used within ``tf.data.Dataset.map()`` for
    building data pipelines.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file (JPEG, PNG, BMP).
    image_size : tuple[int, int], optional
        Target ``(height, width)`` to resize to. Defaults to ``Config.IMAGE_SIZE``.
    normalize : bool, optional
        Divide pixel values by 255.0 to normalize to [0, 1]. Defaults to True.

    Returns
    -------
    tf.Tensor
        Float32 tensor of shape ``(height, width, 3)``.
    """
    raw = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)

    if normalize:
        image = image / 255.0

    return image


# ── NumPy / OpenCV Functions ──────────────────────────────────────────────────


def resize_and_normalize(
    image: np.ndarray,
    target_size: tuple[int, int] = Config.IMAGE_SIZE,
) -> np.ndarray:
    """
    Resize a NumPy image array and normalize pixel values to [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Input image array of shape ``(H, W, C)`` or ``(H, W)``.
        Pixel values can be uint8 [0, 255] or already float.
    target_size : tuple[int, int], optional
        Target ``(height, width)``. Defaults to ``Config.IMAGE_SIZE``.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(height, width, 3)`` with values in [0, 1].

    Notes
    -----
    Grayscale images are automatically converted to RGB by replicating the
    channel dimension. This matches the model's expected 3-channel input.
    """
    # Convert grayscale to RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Resize using INTER_AREA for downscaling, INTER_LINEAR for upscaling
    h, w = target_size
    interp = cv2.INTER_AREA if (image.shape[0] > h or image.shape[1] > w) else cv2.INTER_LINEAR
    image = cv2.resize(image, (w, h), interpolation=interp)

    # Normalize to [0, 1]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    return image


def load_and_preprocess_image(
    image_path: str | Path,
    target_size: tuple[int, int] = Config.IMAGE_SIZE,
    expand_dims: bool = True,
) -> np.ndarray:
    """
    Load an MRI image from disk and prepare it for model inference.

    This is the primary function used by the Streamlit app and the
    GradCAM explainer to preprocess a single MRI image.

    Parameters
    ----------
    image_path : str or Path
        Path to the MRI image file.
    target_size : tuple[int, int], optional
        Target ``(height, width)``. Defaults to ``Config.IMAGE_SIZE``.
    expand_dims : bool, optional
        If True, adds a batch dimension so the output has shape
        ``(1, height, width, 3)`` — required by ``model.predict()``.
        Defaults to True.

    Returns
    -------
    np.ndarray
        Preprocessed image. Shape is ``(1, H, W, 3)`` if ``expand_dims=True``,
        else ``(H, W, 3)``.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the image cannot be decoded by OpenCV.

    Examples
    --------
    >>> img = load_and_preprocess_image("mri.jpg")
    >>> predictions = model.predict(img)   # img shape: (1, 224, 224, 3)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # OpenCV reads as BGR — convert to RGB
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise ValueError(f"OpenCV could not decode the image: {image_path}")

    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    image = resize_and_normalize(image, target_size=target_size)

    if expand_dims:
        image = np.expand_dims(image, axis=0)  # (H, W, 3) → (1, H, W, 3)

    logger.debug("Image loaded and preprocessed: %s → shape %s", image_path.name, image.shape)
    return image


def load_image_for_display(image_path: str | Path) -> np.ndarray:
    """
    Load an image for visualization (RGB, uint8, original resolution).

    Use this function when you want to display the MRI without any model
    preprocessing — e.g., for showing the original image alongside GradCAM.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        RGB image array, dtype ``uint8``, shape ``(H, W, 3)``.
    """
    image_path = Path(image_path)
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    return image


def bytes_to_array(
    image_bytes: bytes,
    target_size: tuple[int, int] = Config.IMAGE_SIZE,
    expand_dims: bool = True,
) -> np.ndarray:
    """
    Decode image bytes (e.g., from Streamlit file uploader) to a NumPy array.

    Parameters
    ----------
    image_bytes : bytes
        Raw image bytes (e.g., from ``st.file_uploader().read()``).
    target_size : tuple[int, int], optional
        Target ``(height, width)``. Defaults to ``Config.IMAGE_SIZE``.
    expand_dims : bool, optional
        Add batch dimension. Defaults to True.

    Returns
    -------
    np.ndarray
        Preprocessed float32 array, shape ``(1, H, W, 3)`` or ``(H, W, 3)``.
    """
    import io
    from PIL import Image

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(pil_img)
    except Exception as exc:
        raise ValueError(f"Could not decode image bytes: {exc}") from exc

    image = resize_and_normalize(np_img, target_size=target_size)

    if expand_dims:
        image = np.expand_dims(image, axis=0)

    return image
