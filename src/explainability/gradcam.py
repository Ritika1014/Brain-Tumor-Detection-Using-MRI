# =============================================================================
# src/explainability/gradcam.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.

Theory
------
Grad-CAM (Selvaraju et al., 2017) generates visual explanations for CNN
decisions by computing the gradient of the class score with respect to the
final convolutional feature maps.

Algorithm:
    1. Forward pass the input image through the CNN.
    2. Extract the activations from the last convolutional layer: A^k (H×W)
    3. Compute gradients of the predicted class score y^c with respect to A^k:
           ∂y^c / ∂A^k_{ij}
    4. Global Average Pool the gradients to get importance weights α^c_k:
           α^c_k = (1/Z) Σ_{i,j} ∂y^c / ∂A^k_{ij}
    5. Compute the weighted combination of activations and apply ReLU:
           L^c_GradCAM = ReLU(Σ_k α^c_k A^k)
    6. Resize L^c_GradCAM to input image size and overlay as heatmap.

Reference: Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations
from Deep Networks via Gradient-Based Localization." ICCV 2017.

Usage
-----
    from src.explainability.gradcam import GradCAM

    cam = GradCAM(model, layer_name="last_conv_layer")
    heatmap = cam.compute_heatmap(image_array, class_idx=0)
    overlay = cam.overlay_on_image(original_image, heatmap)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
    """
    Grad-CAM explainability engine for a Keras CNN model.

    This class wraps a Keras model and provides methods to compute and
    visualize gradient-weighted class activation maps for any image and
    target class.

    Parameters
    ----------
    model : tf.keras.Model
        The trained Keras model (must use Functional API or Sequential).
    layer_name : str, optional
        Name of the target convolutional layer. Defaults to
        ``Config.GRADCAM_LAYER_NAME``.

    Attributes
    ----------
    model : tf.keras.Model
        The original model.
    grad_model : tf.keras.Model
        Auxiliary model with two outputs: (layer activations, class predictions).
    layer_name : str
        Name of the Grad-CAM target layer.

    Examples
    --------
    >>> cam = GradCAM(model, layer_name="last_conv_layer")
    >>> heatmap = cam.compute_heatmap(img_array, class_idx=None)
    >>> overlay = cam.overlay_on_image(original_img, heatmap)
    """

    def __init__(
        self,
        model: tf.keras.Model,
        layer_name: str = Config.GRADCAM_LAYER_NAME,
    ) -> None:
        self.model = model
        self.layer_name = layer_name

        # Build auxiliary model for gradient tape
        try:
            conv_layer = model.get_layer(layer_name)
        except ValueError as exc:
            available = [l.name for l in model.layers]
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available layers: {available}"
            ) from exc

        # grad_model outputs both conv activations AND final predictions
        self.grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
            name="gradcam_model",
        )
        logger.debug("GradCAM initialized with layer: '%s'.", layer_name)

    def compute_heatmap(
        self,
        image_array: np.ndarray,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a given image.

        Parameters
        ----------
        image_array : np.ndarray
            Preprocessed image array of shape ``(1, H, W, 3)`` or ``(H, W, 3)``.
            Pixel values must be in [0, 1] (normalized).
        class_idx : int or None, optional
            Index of the target class. If None, uses the top predicted class
            (argmax). Defaults to None.

        Returns
        -------
        np.ndarray
            2-D heatmap array of shape ``(H, W)`` with values in [0, 1].
            Represents the spatial importance for the target class.
        """
        # Ensure batch dimension
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)

        image_tensor = tf.cast(image_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            conv_outputs, predictions = self.grad_model(image_tensor)

            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))

            # Score for the target class (before softmax normalization)
            target_score = predictions[:, class_idx]

        # Gradient of class score w.r.t. conv activations
        grads = tape.gradient(target_score, conv_outputs)

        # Global Average Pooling of gradients: shape (C,)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight conv outputs by pooled gradients
        conv_outputs = conv_outputs[0]  # Shape: (H, W, C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
        heatmap = tf.squeeze(heatmap)  # (H, W)

        # Apply ReLU — we only want positive activations
        heatmap = tf.maximum(heatmap, 0)

        # Normalize to [0, 1]
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy()

    def overlay_on_image(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = Config.GRADCAM_ALPHA,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay the Grad-CAM heatmap on the original MRI image.

        Parameters
        ----------
        original_image : np.ndarray
            Original RGB image, shape ``(H, W, 3)``. Can be uint8 or float [0,1].
        heatmap : np.ndarray
            2-D Grad-CAM heatmap, shape ``(H, W)``, values in [0, 1].
        alpha : float, optional
            Transparency of the heatmap overlay. Higher = more heatmap.
            Defaults to ``Config.GRADCAM_ALPHA`` = 0.4.
        colormap : int, optional
            OpenCV colormap constant. Defaults to ``cv2.COLORMAP_JET``.

        Returns
        -------
        np.ndarray
            RGB overlay image, dtype ``uint8``, shape ``(H, W, 3)``.
        """
        # Convert original image to uint8 RGB
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)

        h, w = original_image.shape[:2]

        # Resize heatmap to match original image
        heatmap_uint8 = np.uint8(heatmap * 255)
        heatmap_resized = cv2.resize(heatmap_uint8, (w, h))

        # Apply colormap (returns BGR)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)

        # Convert BGR to RGB for matplotlib
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend original and heatmap
        overlay = cv2.addWeighted(
            src1=original_image,
            alpha=1.0 - alpha,
            src2=heatmap_rgb,
            beta=alpha,
            gamma=0,
        )
        return overlay

    def visualize(
        self,
        image_array: np.ndarray,
        original_image: np.ndarray,
        class_names: list[str] | None = None,
        class_idx: int | None = None,
        save_path: Path | str | None = None,
    ) -> plt.Figure:
        if class_names is None:
            class_names = Config.CLASS_NAMES

        """
        Create a three-panel visualization: original MRI | heatmap | overlay.

        Parameters
        ----------
        image_array : np.ndarray
            Preprocessed image for model inference, shape ``(1, H, W, 3)``.
        original_image : np.ndarray
            Original RGB image for display, shape ``(H, W, 3)``.
        class_names : list[str], optional
            Class name list. Defaults to ``Config.CLASS_NAMES``.
        class_idx : int or None, optional
            Target class. None = predicted class. Defaults to None.
        save_path : Path or str or None, optional
            Save destination. Defaults to None.

        Returns
        -------
        plt.Figure
            Three-panel Grad-CAM visualization figure.
        """
        heatmap = self.compute_heatmap(image_array, class_idx=class_idx)
        overlay = self.overlay_on_image(original_image, heatmap)

        # Determine predicted class for title
        preds = self.model.predict(
            np.expand_dims(image_array.squeeze(), 0) if image_array.ndim == 3 else image_array,
            verbose=0,
        )
        pred_idx = int(np.argmax(preds[0]))
        pred_conf = float(preds[0][pred_idx])
        pred_class = class_names[pred_idx]

        target_idx = class_idx if class_idx is not None else pred_idx
        target_class = class_names[target_idx]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
        fig.suptitle(
            f"Grad-CAM Explanation — Target: {target_class.capitalize()}\n"
            f"Prediction: {pred_class.capitalize()} ({pred_conf:.1%} confidence)",
            fontsize=12,
            fontweight="bold",
        )

        # Panel 1: Original MRI
        axes[0].imshow(original_image)
        axes[0].set_title("Original MRI", fontsize=10)
        axes[0].axis("off")

        # Panel 2: Grad-CAM Heatmap (colormapped)
        im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Heatmap", fontsize=10)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Importance")

        # Panel 3: Overlay
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (MRI + GradCAM)", fontsize=10)
        axes[2].axis("off")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
            logger.info("Grad-CAM visualization saved: %s", save_path)

        return fig


# ── Module-Level Convenience Functions ───────────────────────────────────────


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    image_array: np.ndarray,
    layer_name: str = Config.GRADCAM_LAYER_NAME,
    class_idx: int | None = None,
) -> np.ndarray:
    """
    Convenience wrapper: compute Grad-CAM heatmap from model and image.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    image_array : np.ndarray
        Preprocessed image, shape ``(1, H, W, 3)`` or ``(H, W, 3)``.
    layer_name : str, optional
        Target convolutional layer name. Defaults to ``Config.GRADCAM_LAYER_NAME``.
    class_idx : int or None, optional
        Target class index. None = predicted class. Defaults to None.

    Returns
    -------
    np.ndarray
        Grad-CAM heatmap, shape ``(H, W)``, values in [0, 1].
    """
    cam = GradCAM(model, layer_name=layer_name)
    return cam.compute_heatmap(image_array, class_idx=class_idx)


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = Config.GRADCAM_ALPHA,
) -> np.ndarray:
    """
    Convenience wrapper: overlay heatmap on original image.

    Parameters
    ----------
    original_image : np.ndarray
        RGB image, shape ``(H, W, 3)``.
    heatmap : np.ndarray
        Grad-CAM heatmap, shape ``(H, W)``.
    alpha : float, optional
        Blend factor. Defaults to ``Config.GRADCAM_ALPHA``.

    Returns
    -------
    np.ndarray
        Blended RGB overlay image, dtype ``uint8``.
    """
    # We need a model-independent overlay — use a dummy GradCAM instance trick
    # by directly calling the static overlay logic
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)

    h, w = original_image.shape[:2]
    heatmap_uint8 = np.uint8(heatmap * 255)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        src1=original_image, alpha=1.0 - alpha,
        src2=heatmap_rgb, beta=alpha,
        gamma=0,
    )
    return overlay


def batch_gradcam(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str] | None = None,
    num_samples_per_class: int = 2,
    layer_name: str = Config.GRADCAM_LAYER_NAME,
    save_dir: Path | str | None = None,
) -> list[plt.Figure]:
    if class_names is None:
        class_names = Config.CLASS_NAMES

    """
    Generate Grad-CAM visualizations for a batch of images across all classes.

    Iterates over the dataset, collects ``num_samples_per_class`` correctly
    classified samples per class, and produces Grad-CAM visualizations.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    dataset : tf.data.Dataset
        Preprocessed dataset (no shuffle for reproducibility).
    class_names : list[str], optional
        Class names. Defaults to ``Config.CLASS_NAMES``.
    num_samples_per_class : int, optional
        How many visualizations per class. Defaults to 2.
    layer_name : str, optional
        Grad-CAM target layer. Defaults to ``Config.GRADCAM_LAYER_NAME``.
    save_dir : Path or str or None, optional
        If provided, saves figures to ``<save_dir>/<class_name>_<n>.png``.

    Returns
    -------
    list[plt.Figure]
        List of Matplotlib figures.
    """
    cam = GradCAM(model, layer_name=layer_name)
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Collect samples per class
    class_samples: dict[int, list[np.ndarray]] = {i: [] for i in range(len(class_names))}
    needed = {i: num_samples_per_class for i in range(len(class_names))}

    for images, labels in dataset:
        images_np = images.numpy()
        labels_np = np.argmax(labels.numpy(), axis=1)
        preds = model.predict(images_np, verbose=0)
        pred_labels = np.argmax(preds, axis=1)

        for i, (img, true, pred) in enumerate(zip(images_np, labels_np, pred_labels)):
            if true == pred and needed.get(true, 0) > 0:
                class_samples[true].append(img)
                needed[true] -= 1

        if all(v == 0 for v in needed.values()):
            break

    figures: list[plt.Figure] = []
    for cls_idx, samples in class_samples.items():
        cls_name = class_names[cls_idx]
        for n, img_arr in enumerate(samples):
            # img_arr is (H, W, 3) float in [0,1]
            original_rgb = (img_arr * 255).astype(np.uint8)
            save_path = (
                save_dir / f"{cls_name}_{n + 1}.png" if save_dir else None
            )
            fig = cam.visualize(
                image_array=np.expand_dims(img_arr, 0),
                original_image=original_rgb,
                class_names=class_names,
                class_idx=cls_idx,
                save_path=save_path,
            )
            figures.append(fig)
            logger.info(
                "Grad-CAM generated: class=%s, sample=%d.", cls_name, n + 1
            )

    logger.info("batch_gradcam complete: %d visualizations generated.", len(figures))
    return figures
