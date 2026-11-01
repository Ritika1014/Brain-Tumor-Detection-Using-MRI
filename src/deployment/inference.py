# =============================================================================
# src/deployment/inference.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Production inference pipeline.

The ``TumorClassifier`` class encapsulates the complete end-to-end prediction
pipeline: image loading → preprocessing → model inference → Grad-CAM → result.

This class is used by:
- The Streamlit web application (``app/app.py``)
- Command-line prediction scripts
- API endpoints (future work)

Design decisions
----------------
- The model is loaded once at initialization and cached in memory.
- Preprocessing and Grad-CAM are called as methods on the same instance.
- Input can be a file path (str/Path) or raw bytes (Streamlit upload).
- All results are returned as a structured ``PredictionResult`` dataclass.

Usage
-----
    from src.deployment.inference import TumorClassifier

    classifier = TumorClassifier()
    result = classifier.predict_from_bytes(image_bytes)

    print(result.class_name)        # "glioma"
    print(result.confidence)        # 0.9734
    print(result.all_probabilities) # {"glioma": 0.97, ...}
    result.gradcam_overlay          # np.ndarray (RGB image)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import Config
from src.explainability.gradcam import GradCAM
from src.preprocessing.image_utils import bytes_to_array, load_and_preprocess_image
from src.preprocessing.mri_validator import MRIScanValidator
from src.training.trainer import load_best_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """
    Structured container for a single MRI prediction result.

    Attributes
    ----------
    class_name : str
        Predicted tumor class name (e.g., ``"glioma"``).
    class_label : str
        Human-readable class label (e.g., ``"Glioma Tumor"``).
    class_idx : int
        Integer index of the predicted class.
    confidence : float
        Softmax probability of the predicted class (0.0 – 1.0).
    all_probabilities : dict[str, float]
        Softmax probabilities for all classes, keyed by class name.
    gradcam_heatmap : np.ndarray
        Raw Grad-CAM heatmap, shape ``(H, W)``, values in [0, 1].
    gradcam_overlay : np.ndarray
        RGB overlay image (MRI + heatmap), dtype ``uint8``, shape ``(H, W, 3)``.
    original_image : np.ndarray
        Preprocessed image used for inference, shape ``(H, W, 3)``.
    description : str
        Clinical description of the predicted class.
    is_mri : bool
        Whether the image passed MRI validation.
    validation_message : str
        Detailed validation result or error message.
    is_confident : bool
        Whether prediction confidence exceeds Config.CONFIDENCE_THRESHOLD.
    confidence_message : str
        Warning message if confidence is below threshold.
    """

    class_name: str
    class_label: str
    class_idx: int
    confidence: float
    all_probabilities: dict[str, float]
    gradcam_heatmap: np.ndarray
    gradcam_overlay: np.ndarray
    original_image: np.ndarray
    description: str = ""
    is_mri: bool = True
    validation_message: str = ""
    is_confident: bool = True
    confidence_message: str = ""


class TumorClassifier:
    """
    End-to-end brain tumor MRI classifier with Grad-CAM explanations.

    This class wraps model loading, image preprocessing, prediction, and
    Grad-CAM visualization into a single, easy-to-use interface.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        layer_name: str = Config.GRADCAM_LAYER_NAME,
        class_names: list[str] | None = None,
    ) -> None:
        from src.utils.helpers import load_class_mapping
        mapping = load_class_mapping()

        if class_names is not None:
            self.class_names = class_names
        elif mapping.get("class_names"):
            self.class_names = mapping["class_names"]
        else:
            self.class_names = Config.CLASS_NAMES

        self.class_labels = mapping.get("class_labels") or Config.CLASS_LABELS
        self.class_descriptions = mapping.get("class_descriptions") or Config.CLASS_DESCRIPTIONS
        self.layer_name = layer_name
        self.validator = MRIScanValidator()
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD

        # ── Load Model ───────────────────────────────────────────────────────
        model_path = model_path or Config.BEST_MODEL_PATH
        logger.info("Loading model from: %s", model_path)
        self.model = load_best_model(model_path)

        # ── Initialize Grad-CAM ───────────────────────────────────────────────
        self.cam = GradCAM(self.model, layer_name=self.layer_name)
        logger.info("TumorClassifier ready. Classes: %s | Conf Threshold: %.2f",
                    self.class_names, self.confidence_threshold)

    def _run_inference(
        self,
        image_array: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> PredictionResult:
        """
        Core inference method with MRI Validation & Confidence Gating.
        """
        # Original uint8 image for display
        orig_img_uint8 = (image_array.squeeze() * 255.0).clip(0, 255).astype(np.uint8)

        # ── 1. MRI Validation Check ─────────────────────────────────────────
        is_mri, val_msg, metrics = self.validator.validate_image(orig_img_uint8)
        if not is_mri:
            logger.warning("MRI Validation failed: %s", val_msg)
            return PredictionResult(
                class_name="invalid_non_mri",
                class_label="Invalid Non-MRI Image",
                class_idx=-1,
                confidence=0.0,
                all_probabilities={},
                gradcam_heatmap=np.zeros((image_array.shape[1], image_array.shape[2])),
                gradcam_overlay=orig_img_uint8,
                original_image=orig_img_uint8,
                description="",
                is_mri=False,
                validation_message=val_msg,
                is_confident=False,
                confidence_message=val_msg,
            )

        # ── 2. Model Prediction ─────────────────────────────────────────────
        probs = self.model.predict(image_array, verbose=0)[0]  # Shape: (C,)
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        class_name = self.class_names[class_idx]
        class_label = self.class_labels.get(class_name, class_name)
        description = self.class_descriptions.get(class_name, "")

        all_probabilities = {
            name: float(probs[i]) for i, name in enumerate(self.class_names)
        }

        logger.info(
            "Prediction: %s (%.1f%% confidence).", class_label, confidence * 100
        )

        # ── 3. Confidence Threshold Check ──────────────────────────────────
        active_threshold = (
            confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        )
        is_confident = confidence >= active_threshold
        confidence_msg = ""
        if not is_confident:
            confidence_msg = (
                f"Model confidence ({confidence:.1%}) is below threshold ({active_threshold:.0%}). "
                "Clinical review recommended."
            )
            logger.info("Confidence advisory: %.1f%% vs %.1f%% threshold",
                        confidence * 100, active_threshold * 100)

        # ── 4. Grad-CAM Heatmap Computation (Always computed for inspection) ──
        try:
            heatmap = self.cam.compute_heatmap(image_array, class_idx=class_idx)
            overlay = self.cam.overlay_on_image(
                original_image=orig_img_uint8,
                heatmap=heatmap,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to compute Grad-CAM: %s", exc)
            heatmap = np.zeros((image_array.shape[1], image_array.shape[2]))
            overlay = orig_img_uint8

        return PredictionResult(
            class_name=class_name,
            class_label=class_label,
            class_idx=class_idx,
            confidence=confidence,
            all_probabilities=all_probabilities,
            gradcam_heatmap=heatmap,
            gradcam_overlay=overlay,
            original_image=orig_img_uint8,
            description=description,
            is_mri=True,
            validation_message="Valid Brain MRI scan verified.",
            is_confident=is_confident,
            confidence_message=confidence_msg,
        )

    def predict_from_path(
        self,
        image_path: str | Path,
        confidence_threshold: float | None = None,
    ) -> PredictionResult:
        """
        Predict tumor class from an image file path.

        Parameters
        ----------
        image_path : str or Path
            Path to the MRI image file (JPEG, PNG, BMP).
        confidence_threshold : float, optional
            Custom confidence threshold override.

        Returns
        -------
        PredictionResult
            Complete prediction result.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist.
        """
        image_array = load_and_preprocess_image(
            image_path, target_size=Config.IMAGE_SIZE, expand_dims=True
        )
        return self._run_inference(image_array, confidence_threshold=confidence_threshold)

    def predict_from_bytes(
        self,
        image_bytes: bytes,
        confidence_threshold: float | None = None,
    ) -> PredictionResult:
        """
        Predict tumor class from raw image bytes (e.g., Streamlit file uploader).

        Parameters
        ----------
        image_bytes : bytes
            Raw bytes of the image file.
        confidence_threshold : float, optional
            Custom confidence threshold override.

        Returns
        -------
        PredictionResult
            Complete prediction result.
        """
        image_array = bytes_to_array(
            image_bytes, target_size=Config.IMAGE_SIZE, expand_dims=True
        )
        return self._run_inference(image_array, confidence_threshold=confidence_threshold)

    def predict_from_array(
        self,
        image_array: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> PredictionResult:
        """
        Predict from a preprocessed NumPy array.

        Parameters
        ----------
        image_array : np.ndarray
            Float32 array in [0, 1], shape ``(H, W, 3)`` or ``(1, H, W, 3)``.
        confidence_threshold : float, optional
            Custom confidence threshold override.

        Returns
        -------
        PredictionResult
            Complete prediction result.
        """
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)
        return self._run_inference(image_array, confidence_threshold=confidence_threshold)
