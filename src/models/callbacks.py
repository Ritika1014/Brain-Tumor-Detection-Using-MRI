# =============================================================================
# src/models/callbacks.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Keras training callbacks factory.

Callbacks control training behavior at the epoch and batch level. This module
centralizes all callback configuration so that the trainer can request a
standard set of callbacks without worrying about individual configuration.

Callbacks included
------------------
1. **ModelCheckpoint** — Save the best model weights (by val_accuracy).
2. **EarlyStopping** — Stop training when val_accuracy stops improving.
3. **ReduceLROnPlateau** — Halve LR when val_loss plateaus.
4. **CSVLogger** — Append epoch metrics to a CSV file for analysis.
5. **TensorBoard** — Write logs for TensorBoard visualization.

Usage
-----
    from src.models.callbacks import get_callbacks

    callbacks = get_callbacks()
    model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import callbacks as keras_callbacks

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_callbacks(
    model_checkpoint_path: Path = Config.BEST_MODEL_PATH,
    csv_log_path: Path = Config.TRAINING_CSV_LOG,
    tensorboard_log_dir: Path | None = None,
    monitor_metric: str = "val_accuracy",
    early_stopping_patience: int = Config.EARLY_STOPPING_PATIENCE,
    lr_reduce_patience: int = Config.LR_REDUCE_PATIENCE,
    lr_reduce_factor: float = Config.LR_REDUCE_FACTOR,
    lr_min: float = Config.LR_MIN,
) -> list[keras_callbacks.Callback]:
    """
    Create and return the standard list of training callbacks.

    Parameters
    ----------
    model_checkpoint_path : Path, optional
        Path to save the best model. Defaults to ``Config.BEST_MODEL_PATH``.
    csv_log_path : Path, optional
        Path for the CSV training history log. Defaults to ``Config.TRAINING_CSV_LOG``.
    tensorboard_log_dir : Path or None, optional
        Directory for TensorBoard logs. If None, TensorBoard is disabled.
        Defaults to None.
    monitor_metric : str, optional
        Metric to monitor for checkpoint and early stopping.
        Defaults to ``"val_accuracy"``.
    early_stopping_patience : int, optional
        Number of epochs to wait before stopping. Defaults to ``Config.EARLY_STOPPING_PATIENCE``.
    lr_reduce_patience : int, optional
        Epochs to wait before reducing LR. Defaults to ``Config.LR_REDUCE_PATIENCE``.
    lr_reduce_factor : float, optional
        Factor by which LR is reduced. Defaults to ``Config.LR_REDUCE_FACTOR``.
    lr_min : float, optional
        Lower bound on LR. Defaults to ``Config.LR_MIN``.

    Returns
    -------
    list[keras_callbacks.Callback]
        List of configured Keras callbacks.

    Notes
    -----
    - ModelCheckpoint uses ``save_best_only=True`` and ``save_weights_only=False``
      to save the full model (architecture + weights) in ``.keras`` format.
    - EarlyStopping restores best weights upon early stopping via
      ``restore_best_weights=True``.

    Examples
    --------
    >>> callbacks = get_callbacks()
    >>> history = model.fit(train_ds, validation_data=val_ds,
    ...                     epochs=50, callbacks=callbacks)
    """
    # Ensure output directories exist
    model_checkpoint_path = Path(model_checkpoint_path)
    model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    csv_log_path = Path(csv_log_path)
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)

    callback_list: list[keras_callbacks.Callback] = []

    # ── 1. Model Checkpoint ──────────────────────────────────────────────────
    # filepath must end in .keras so Keras uses the native format.
    # save_best_only=True + monitor="val_accuracy" ensures only the
    # epoch with highest validation accuracy is persisted to disk.
    checkpoint = keras_callbacks.ModelCheckpoint(
        filepath=str(model_checkpoint_path),
        monitor=monitor_metric,
        save_best_only=True,
        save_weights_only=False,  # Save full model (architecture + weights)
        mode="max",
        verbose=1,
    )
    callback_list.append(checkpoint)
    logger.info("Callback: ModelCheckpoint → %s (monitor=%s, save_best_only=True)",
                model_checkpoint_path, monitor_metric)

    # ── 2. Early Stopping ────────────────────────────────────────────────────
    early_stop = keras_callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=early_stopping_patience,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )
    callback_list.append(early_stop)
    logger.debug("Callback: EarlyStopping (patience=%d).", early_stopping_patience)

    # ── 3. Reduce Learning Rate on Plateau ───────────────────────────────────
    reduce_lr = keras_callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
        min_lr=lr_min,
        verbose=1,
    )
    callback_list.append(reduce_lr)
    logger.debug(
        "Callback: ReduceLROnPlateau (factor=%.2f, patience=%d).",
        lr_reduce_factor,
        lr_reduce_patience,
    )

    # ── 4. CSV Logger ────────────────────────────────────────────────────────
    csv_logger = keras_callbacks.CSVLogger(
        filename=str(csv_log_path),
        append=False,
    )
    callback_list.append(csv_logger)
    logger.debug("Callback: CSVLogger → %s", csv_log_path)

    # ── 5. TensorBoard (optional) ─────────────────────────────────────────────
    if tensorboard_log_dir is not None:
        tb_log_dir = Path(tensorboard_log_dir)
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tensorboard = keras_callbacks.TensorBoard(
            log_dir=str(tb_log_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        )
        callback_list.append(tensorboard)
        logger.debug("Callback: TensorBoard → %s", tb_log_dir)

    logger.info("Registered %d training callbacks.", len(callback_list))
    return callback_list
