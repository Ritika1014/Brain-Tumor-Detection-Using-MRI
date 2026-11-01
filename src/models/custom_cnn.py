# =============================================================================
# src/models/custom_cnn.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Custom Convolutional Neural Network for Brain Tumor Classification.

Architecture Overview
---------------------
The model is a deep CNN inspired by VGGNet-style stacked convolutions but
augmented with Batch Normalization and Global Average Pooling (instead of
Flatten + large Dense layers) to reduce overfitting and parameter count.

Architecture:
    Input (224×224×3)
    │
    ├── Block 1: Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool(2×2)
    ├── Block 2: Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool(2×2)
    ├── Block 3: Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool(2×2)
    ├── Block 4: Conv(256) → BN → ReLU → Conv(256) → BN → ReLU [LAST CONV]
    │
    ├── GlobalAveragePooling2D
    ├── Dense(512) → BN → ReLU → Dropout(0.5)
    ├── Dense(256) → BN → ReLU → Dropout(0.3)
    └── Dense(4, softmax)   ← Output: 4-class probability vector

Design Decisions
----------------
- **Batch Normalization** after each Conv layer accelerates convergence and
  acts as a regularizer, reducing sensitivity to weight initialization.
- **Global Average Pooling** over Flatten reduces the parameter count from
  ~8M to ~2M for the same architecture depth, reducing overfitting.
- The last convolutional layer (``last_conv_layer``) is named explicitly
  for Grad-CAM visualization — a critical design requirement.
- **Dropout** (0.5, 0.3) in dense layers provides further regularization.

Future Extensibility
--------------------
This architecture serves as the baseline model. Future work will replace
the body with pretrained backbones (ResNet50V2, EfficientNetB3) while
keeping the same ``build_custom_cnn()`` interface.

Usage
-----
    from src.models.custom_cnn import build_custom_cnn

    model = build_custom_cnn()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _conv_bn_relu(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    padding: str = "same",
    name_prefix: str = "",
) -> tf.Tensor:
    """
    Apply Conv2D → BatchNormalization → ReLU activation.

    This helper keeps the architecture definition clean and ensures all
    intermediate layers follow the same Conv → BN → Activation pattern
    recommended by modern deep learning practice.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map.
    filters : int
        Number of convolutional filters.
    kernel_size : int, optional
        Convolution kernel size (square). Defaults to 3.
    padding : str, optional
        Padding mode. Defaults to ``"same"`` to preserve spatial dimensions.
    name_prefix : str, optional
        Prefix for layer names (improves readability in model.summary()).

    Returns
    -------
    tf.Tensor
        Output feature map after Conv → BN → ReLU.
    """
    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        padding=padding,
        use_bias=False,  # BN after Conv makes bias redundant
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    return x


def _conv_block(
    x: tf.Tensor,
    filters: int,
    block_name: str,
    pool: bool = True,
    last_conv_name: str | None = None,
) -> tf.Tensor:
    """
    Build a convolutional block with two Conv→BN→ReLU layers and optional MaxPool.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map.
    filters : int
        Number of filters for both Conv layers in this block.
    block_name : str
        Block identifier (e.g., ``"block1"``).
    pool : bool, optional
        If True, append MaxPooling2D(2×2) at the end. Defaults to True.
    last_conv_name : str or None, optional
        If provided, the second Conv layer is given this specific name.
        Used to name ``last_conv_layer`` for Grad-CAM. Defaults to None.

    Returns
    -------
    tf.Tensor
        Output feature map.
    """
    x = _conv_bn_relu(x, filters, name_prefix=f"{block_name}_1")

    # Second Conv — optionally named as last conv layer for Grad-CAM
    if last_conv_name is not None:
        x = layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=last_conv_name,  # Explicit name for Grad-CAM hook
        )(x)
        x = layers.BatchNormalization(name=f"{block_name}_2_bn")(x)
        x = layers.Activation("relu", name=f"{block_name}_2_relu")(x)
    else:
        x = _conv_bn_relu(x, filters, name_prefix=f"{block_name}_2")

    if pool:
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{block_name}_pool")(x)

    return x


def build_custom_cnn(
    input_shape: tuple[int, int, int] = Config.INPUT_SHAPE,
    num_classes: int = Config.NUM_CLASSES,
    dropout_rate: float = Config.DROPOUT_RATE,
    gradcam_layer_name: str = Config.GRADCAM_LAYER_NAME,
) -> keras.Model:
    """
    Build and return the Custom CNN model for brain tumor classification.

    The model uses the Keras Functional API to allow inspection of
    intermediate layers (required for Grad-CAM).

    Parameters
    ----------
    input_shape : tuple[int, int, int], optional
        Shape of input images ``(height, width, channels)``.
        Defaults to ``Config.INPUT_SHAPE`` = ``(224, 224, 3)``.
    num_classes : int, optional
        Number of output classes. Defaults to ``Config.NUM_CLASSES`` = 4.
    dropout_rate : float, optional
        Dropout probability in fully-connected head. Defaults to ``Config.DROPOUT_RATE``.
    gradcam_layer_name : str, optional
        Name assigned to the last convolutional layer (used by Grad-CAM).
        Defaults to ``Config.GRADCAM_LAYER_NAME``.

    Returns
    -------
    keras.Model
        Compiled-ready Keras Functional model.

    Notes
    -----
    The model is NOT compiled here — compilation is handled by the trainer
    to allow flexible optimizer/loss/metric configuration.

    Examples
    --------
    >>> model = build_custom_cnn()
    >>> model.compile(
    ...     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    ...     loss="categorical_crossentropy",
    ...     metrics=["accuracy"],
    ... )
    >>> model.summary()
    """
    inputs = keras.Input(shape=input_shape, name="input_mri")

    # ── Feature Extraction Backbone ──────────────────────────────────────────
    # Block 1: 224×224×3 → 112×112×32 (2× Conv32 + MaxPool)
    x = _conv_block(inputs, filters=32, block_name="block1", pool=True)

    # Block 2: 112×112×32 → 56×56×64 (2× Conv64 + MaxPool)
    x = _conv_block(x, filters=64, block_name="block2", pool=True)

    # Block 3: 56×56×64 → 28×28×128 (2× Conv128 + MaxPool)
    x = _conv_block(x, filters=128, block_name="block3", pool=True)

    # Block 4: 28×28×128 → 14×14×256 (2× Conv256 + MaxPool; contains last_conv_layer for Grad-CAM)
    x = _conv_block(x, filters=256, block_name="block4", pool=True, last_conv_name=gradcam_layer_name)

    # ── Classification Head ──────────────────────────────────────────────────
    # Global Average Pooling: 14×14×256 → 256
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dense block
    x = layers.Dense(256, use_bias=False, kernel_initializer="he_normal", name="dense_256")(x)
    x = layers.BatchNormalization(name="dense_256_bn")(x)
    x = layers.Activation("relu", name="dense_256_relu")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    # Output layer: softmax over num_classes
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="BrainTumor_CustomCNN")

    logger.info(
        "Custom CNN built: %d parameters, GradCAM layer = '%s'.",
        model.count_params(),
        gradcam_layer_name,
    )
    return model


def get_model_summary(model: keras.Model, save_path: Path | None = None) -> str:
    """
    Generate and optionally save the model architecture summary.

    Parameters
    ----------
    model : keras.Model
        The Keras model to summarize.
    save_path : Path or None, optional
        If provided, saves the summary as a text file at this path.
        Defaults to None.

    Returns
    -------
    str
        The model summary as a string.
    """
    # Capture summary as a string
    summary_lines: list[str] = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    summary_str = "\n".join(summary_lines)

    print(summary_str)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(summary_str, encoding="utf-8")
        logger.info("Model summary saved to: %s", save_path)

    return summary_str
