# =============================================================================
# src/utils/logger.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Logging utility module.

Provides a consistent logger factory used across the entire project.
Each module calls `get_logger(__name__)` to get a named logger that writes
both to the console (INFO level) and to a rotating log file (DEBUG level).

Usage
-----
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Model training started.")
    logger.debug("Batch size: %d", batch_size)
    logger.error("Failed to load dataset: %s", str(error))
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(
    name: str,
    log_file: Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB per log file
    backup_count: int = 3,
) -> logging.Logger:
    """
    Create and return a named logger with console and optional file handlers.

    The logger is idempotent — calling this function multiple times with the
    same `name` returns the same logger without adding duplicate handlers.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    log_file : Path or None, optional
        Path to the log file. If None, file logging is disabled.
        Defaults to None.
    console_level : int, optional
        Logging level for the console handler. Defaults to ``logging.INFO``.
    file_level : int, optional
        Logging level for the file handler. Defaults to ``logging.DEBUG``.
    max_bytes : int, optional
        Maximum size per log file before rotation. Defaults to 5 MB.
    backup_count : int, optional
        Number of backup log files to keep. Defaults to 3.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Pipeline started.")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers filter

    # ── Formatter ────────────────────────────────────────────────────────────
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # ── Console Handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File Handler (Rotating) ──────────────────────────────────────────────
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
