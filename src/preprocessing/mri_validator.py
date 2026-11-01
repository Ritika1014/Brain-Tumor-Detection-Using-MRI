# =============================================================================
# src/preprocessing/mri_validator.py
# Brain Tumor Detection System: Explainable AI Framework
# =============================================================================
"""
MRI Image Validation Module.

Before passing an uploaded image to the Brain Tumor Classification CNN,
this module validates whether the input image is genuinely a Brain MRI scan.

Filters out non-medical inputs such as:
- Human faces / portraits / body photos
- Animals & pets
- Mobile phones, cars, objects
- Documents & handwritten text
- Landscapes, scenery, cartoons, and arbitrary non-brain images

Validation Rules
----------------
1. **Chromaticity / Color Divergence**:
   Brain MRIs are single-channel grayscale scans (R=G=B). Natural color photos
   have significant divergence between red, green, and blue channels.
2. **Background Darkness Ratio**:
   Brain MRIs feature a dark background surrounding the skull mass.
   The image border (outer 8% margin) must consist predominantly of low-intensity pixels.
3. **Brain Tissue Contrast & Texture**:
   The central region must exhibit distinct tissue intensity variation (brain matter vs fluids/skull).
   Solid colors, documents, or pure black/white images are rejected.

Usage
-----
    from src.preprocessing.mri_validator import MRIScanValidator

    validator = MRIScanValidator()
    is_valid, message, details = validator.validate_image(image_bytes_or_array)
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MRIScanValidator:
    """
    Automated validator for Brain MRI scans.

    Analyzes spatial, spectral, and structural image characteristics
    to ensure uploaded files are valid brain MRI scans before executing CNN prediction.
    """

    def __init__(
        self,
        max_color_divergence: float = Config.MAX_COLOR_DIVERGENCE,
        min_border_darkness_ratio: float = Config.MIN_BACKGROUND_DARKNESS_RATIO,
        min_contrast: float = Config.MIN_BRAIN_CONTRAST,
    ) -> None:
        self.max_color_divergence = max_color_divergence
        self.min_border_darkness_ratio = min_border_darkness_ratio
        self.min_contrast = min_contrast

    def validate_image(self, image: np.ndarray) -> tuple[bool, str, dict[str, float]]:
        """
        Validate whether the given image array is a valid Brain MRI scan.

        Parameters
        ----------
        image : np.ndarray
            Input image array. Can be float32 [0, 1] or uint8 [0, 255], shape (H, W, 3) or (1, H, W, 3).

        Returns
        -------
        tuple[bool, str, dict[str, float]]
            - is_valid : True if the image is a valid Brain MRI, False otherwise.
            - message : User-friendly explanation if invalid.
            - metrics : Detailed diagnostic feature scores.
        """
        # ── Ensure uint8 (H, W, 3) format ─────────────────────────────────────
        img = np.squeeze(image)
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        h, w, _ = img.shape
        if h < 32 or w < 32:
            return False, "Image resolution is too low to be a valid Brain MRI scan.", {}

        # ── 1. Color Divergence (Chromaticity Check) ──────────────────────────
        r, g, b = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)
        diff_rg = np.std(r - g)
        diff_gb = np.std(g - b)
        diff_br = np.std(b - r)
        color_divergence = float(diff_rg + diff_gb + diff_br)

        # ── 2. Border Darkness Ratio ──────────────────────────────────────────
        # Extract outer 8% border pixels
        margin_h = max(2, int(h * 0.08))
        margin_w = max(2, int(w * 0.08))

        top_border = img[:margin_h, :, :]
        bottom_border = img[-margin_h:, :, :]
        left_border = img[:, :margin_w, :]
        right_border = img[:, -margin_w:, :]

        border_pixels = np.concatenate([
            top_border.reshape(-1, 3),
            bottom_border.reshape(-1, 3),
            left_border.reshape(-1, 3),
            right_border.reshape(-1, 3),
        ], axis=0)

        border_gray = cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).flatten()
        dark_border_ratio = float(np.mean(border_gray < 50))  # Intensity threshold < 50

        # ── 3. Central Contrast & Structure ──────────────────────────────────
        gray_full = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        center_crop = gray_full[margin_h:h-margin_h, margin_w:w-margin_w]
        contrast = float(np.std(center_crop))

        metrics = {
            "color_divergence": round(color_divergence, 2),
            "dark_border_ratio": round(dark_border_ratio, 3),
            "central_contrast": round(contrast, 2),
        }

        logger.debug("MRI Validation Metrics for upload: %s", metrics)

        # ── Evaluate Rules ────────────────────────────────────────────────────
        # Failure 1: Color Divergence too high (e.g. colorful photo/animal/landscape/object)
        if color_divergence > self.max_color_divergence:
            logger.warning("Rejected non-MRI upload: color divergence %.2f > max %.2f",
                           color_divergence, self.max_color_divergence)
            return False, "Invalid input. Please upload a valid Brain MRI scan in JPG, JPEG, or PNG format.", metrics

        # Failure 2: Border is not dark (e.g. photo, paper document, bright scene)
        if dark_border_ratio < self.min_border_darkness_ratio:
            logger.warning("Rejected non-MRI upload: border darkness ratio %.3f < min %.3f",
                           dark_border_ratio, self.min_border_darkness_ratio)
            return False, "Invalid input. Please upload a valid Brain MRI scan in JPG, JPEG, or PNG format.", metrics

        # Failure 3: Insufficient tissue contrast (e.g. solid color or blank image)
        if contrast < self.min_contrast:
            logger.warning("Rejected non-MRI upload: central contrast %.2f < min %.2f",
                           contrast, self.min_contrast)
            return False, "Invalid input. Please upload a valid Brain MRI scan in JPG, JPEG, or PNG format.", metrics

        return True, "Valid Brain MRI scan verified.", metrics


def validate_mri_image(image: np.ndarray) -> tuple[bool, str, dict[str, float]]:
    """Convenience functional interface for MRI image validation."""
    validator = MRIScanValidator()
    return validator.validate_image(image)
