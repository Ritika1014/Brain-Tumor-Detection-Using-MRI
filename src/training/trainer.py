# =============================================================================
# src/training/trainer.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Model training module.

Provides the main training function ``train_model()`` that wires together the
model, datasets, optimizer, callbacks, and class weights into a single
reproducible training run. Also provides ``load_best_model()`` for inference.

The training loop:
  1. Builds the model.
  2. Computes class weights (handles class imbalance).
  3. Compiles the model (Adam + CategoricalCrossentropy + metrics).
  4. Fits with callbacks (checkpoint, early stopping, LR schedule, CSV log).
  5. Saves the final (last epoch) model separately from the best checkpoint.
  6. Returns the training History object for downstream analysis.

Usage
-----
    from src.training.trainer import train_model, load_best_model

    history = train_model(train_ds, val_ds)
    model = load_best_model()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.config import Config
from src.models.callbacks import get_callbacks
from src.models.custom_cnn import build_custom_cnn
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_class_weights() -> dict[int, float]:
    """
    Compute smoothed class weights directly from training folder metadata.
    Square-root smoothing prevents severe class weight imbalance (e.g., 1.8x skew),
    ensuring no single class dominates loss gradients during training.
    """
    from src.preprocessing.data_loader import load_dataset_paths
    df = load_dataset_paths(Config.TRAIN_DIR, shuffle=False)
    y_labels = df["label"].to_numpy()
    classes = np.unique(y_labels)
    raw_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_labels
    )
    # Apply square-root smoothing & normalize mean to 1.0
    smoothed_weights = np.sqrt(raw_weights)
    smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
    
    class_weight_dict = dict(zip(classes.tolist(), smoothed_weights.tolist()))
    logger.info("Smoothed class weights calculated: %s", class_weight_dict)
    return class_weight_dict


def train_model(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = Config.EPOCHS,
    learning_rate: float = Config.LEARNING_RATE,
    use_class_weights: bool = True,
    enable_tensorboard: bool = False,
) -> keras.callbacks.History:
    """
    Build, compile, and train the Custom CNN model.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Training dataset (batched, normalized, optionally augmented).
    val_ds : tf.data.Dataset
        Validation dataset (batched, normalized).
    epochs : int, optional
        Maximum number of training epochs. EarlyStopping may stop earlier.
        Defaults to ``Config.EPOCHS``.
    learning_rate : float, optional
        Initial Adam optimizer learning rate. Defaults to ``Config.LEARNING_RATE``.
    use_class_weights : bool, optional
        Whether to apply balanced class weights. Recommended for imbalanced
        datasets. Defaults to True.
    enable_tensorboard : bool, optional
        Whether to enable TensorBoard logging. Defaults to False.

    Returns
    -------
    keras.callbacks.History
        Keras History object containing per-epoch metrics. Access via
        ``history.history["accuracy"]``, ``history.history["val_loss"]``, etc.

    Notes
    -----
    - The best model (by ``val_accuracy``) is automatically saved to
      ``Config.BEST_MODEL_PATH`` by the ModelCheckpoint callback.
    - The final model (last epoch) is saved to ``Config.FINAL_MODEL_PATH``.
    - Training logs are saved to ``Config.TRAINING_CSV_LOG``.

    Examples
    --------
    >>> train_ds, val_ds = build_tf_dataset(Config.TRAIN_DIR, split="train", augment=True)
    >>> history = train_model(train_ds, val_ds, epochs=50)
    """
    logger.info("=" * 60)
    logger.info("  STARTING MODEL TRAINING")
    logger.info("  Epochs: %d | LR: %.4e | Augmented: True", epochs, learning_rate)
    logger.info("=" * 60)

    # ── Ensure output dirs exist BEFORE callbacks are created ─────────────────
    # ModelCheckpoint needs the parent directory to exist at callback creation.
    Config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Model save directory ready: %s", Config.SAVED_MODELS_DIR)

    from src.utils.helpers import detect_class_names, save_class_mapping
    class_names = Config.CLASS_NAMES
    if class_names:
        save_class_mapping(class_names)
    logger.info("Training model on %d detected classes: %s", len(class_names), class_names)

    # ── Build Model ──────────────────────────────────────────────────────────
    model = build_custom_cnn()


    # ── Compile ──────────────────────────────────────────────────────────────
    # clipnorm=1.0 prevents gradient explosions during training, especially
    # important for medical images with varying contrast levels.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,  # Gradient clipping for training stability
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=Config.LABEL_SMOOTHING),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )
    logger.info("Model compiled. Total parameters: %s.", f"{model.count_params():,}")
    model.summary(print_fn=logger.debug)

    # ── Class Weights ────────────────────────────────────────────────────────
    class_weight = None
    if use_class_weights:
        class_weight = _compute_class_weights()

    # ── Callbacks ────────────────────────────────────────────────────────────
    # ModelCheckpoint is configured with:
    #   filepath = Config.BEST_MODEL_PATH  (outputs/saved_models/best_model.keras)
    #   save_best_only = True
    #   monitor = "val_accuracy"
    tb_log_dir = Config.LOGS_DIR / "tensorboard" if enable_tensorboard else None
    callbacks = get_callbacks(tensorboard_log_dir=tb_log_dir)

    # ── Train ────────────────────────────────────────────────────────────────
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # ── Save Final Model & Guarantee Best Model Exists ────────────────────────
    # Save the final (last-epoch) model unconditionally.
    model.save(str(Config.FINAL_MODEL_PATH))
    logger.info("Final model saved to: %s", Config.FINAL_MODEL_PATH)

    # If ModelCheckpoint saved the best model → it already exists. If for any
    # reason it was skipped (e.g. only 1 epoch ran), fall back to saving the
    # current weights as the best model so the app always has a model to load.
    if not Config.BEST_MODEL_PATH.exists():
        model.save(str(Config.BEST_MODEL_PATH))
        logger.warning(
            "ModelCheckpoint did not save best_model.keras (possibly only 1 epoch ran). "
            "Saving current model as best: %s", Config.BEST_MODEL_PATH
        )
    else:
        logger.info("Best model already saved by ModelCheckpoint: %s", Config.BEST_MODEL_PATH)

    # ── Report Best Results ──────────────────────────────────────────────────
    # CategoricalAccuracy(name="accuracy") produces "val_accuracy" in history.
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_categorical_accuracy"
    val_acc_values = history.history.get(val_acc_key, [0.0])
    best_val_acc = max(val_acc_values)
    best_epoch = val_acc_values.index(best_val_acc) + 1
    logger.info(
        "Training complete. Best %s: %.4f at epoch %d.",
        val_acc_key, best_val_acc, best_epoch,
    )
    logger.info("Model saved at: %s", Config.BEST_MODEL_PATH)

    return history


def load_best_model(model_path: Path = Config.BEST_MODEL_PATH) -> keras.Model:
    """
    Load the best saved model from disk for inference or evaluation.

    Parameters
    ----------
    model_path : Path, optional
        Path to the ``.keras`` model file. Defaults to ``Config.BEST_MODEL_PATH``.

    Returns
    -------
    keras.Model
        Loaded Keras model, ready for inference.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at ``model_path``.

    Examples
    --------
    >>> model = load_best_model()
    >>> predictions = model.predict(test_ds)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Have you run training yet? Call train_model() first."
        )

    model = tf.keras.models.load_model(str(model_path))
    logger.info("Model loaded from: %s", model_path)
    logger.info("Model input shape: %s", model.input_shape)
    return model
