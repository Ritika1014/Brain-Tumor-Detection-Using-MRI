# =============================================================================
# src/config.py
# From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection
# =============================================================================
"""
Central configuration module.

All project-wide constants, hyperparameters, and path definitions are
declared here. Import `Config` anywhere in the project to access them.
This design makes the project easy to modify without hunting for hardcoded
values scattered across multiple files.

Usage
-----
    from src.config import Config

    data_dir = Config.DATA_DIR
    img_size = Config.IMAGE_SIZE
"""

from pathlib import Path


class ConfigMeta(type):
    """Metaclass providing dynamic properties on Config for dataset attributes."""

    @property
    def CLASS_NAMES(cls) -> list[str]:
        return cls.get_class_names()

    @property
    def NUM_CLASSES(cls) -> int:
        return len(cls.CLASS_NAMES)

    @property
    def CLASS_LABELS(cls) -> dict[str, str]:
        return cls.get_class_labels()

    @property
    def CLASS_DESCRIPTIONS(cls) -> dict[str, str]:
        return cls.get_class_descriptions()


class Config(metaclass=ConfigMeta):
    """
    Master configuration class for the brain tumor detection project.

    All attributes are class-level constants. Do NOT instantiate this class;
    access attributes directly via `Config.<ATTRIBUTE>`.
    """


    # ── Project Root ─────────────────────────────────────────────────────────
    # Resolve the project root relative to this file's location so the
    # project works regardless of where it's launched from.
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # ── Dataset Paths ────────────────────────────────────────────────────────
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    METADATA_DIR: Path = DATA_DIR / "metadata"

    # MRI dataset directories (place your Kaggle dataset here)
    _train_p1: Path = RAW_DIR / "Training"
    _train_p2: Path = RAW_DIR / "Brain_Tumor_MRI_Dataset" / "Training"
    TRAIN_DIR: Path = _train_p2 if _train_p2.exists() else _train_p1

    _test_p1: Path = RAW_DIR / "Testing"
    _test_p2: Path = RAW_DIR / "Brain_Tumor_MRI_Dataset" / "Testing"
    TEST_DIR: Path = _test_p2 if _test_p2.exists() else _test_p1

    # ── Output Directories ───────────────────────────────────────────────────
    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
    SAVED_MODELS_DIR: Path = OUTPUTS_DIR / "saved_models"
    PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
    GRADCAM_DIR: Path = OUTPUTS_DIR / "gradcam"
    PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
    METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
    CONFUSION_MATRIX_DIR: Path = OUTPUTS_DIR / "confusion_matrix"

    # ── Logging ──────────────────────────────────────────────────────────────
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    LOG_FILE: Path = LOGS_DIR / "training.log"

    # ── Dataset Metadata & Dynamic Classes ──────────────────────────────────
    CLASS_MAPPING_FILE: Path = METADATA_DIR / "class_mapping.json"

    @classmethod
    def get_class_names(cls) -> list[str]:
        from src.utils.helpers import detect_class_names, load_class_mapping
        detected = detect_class_names(cls.TRAIN_DIR)
        if detected:
            return detected
        mapping = load_class_mapping(cls.CLASS_MAPPING_FILE)
        return mapping.get("class_names", [])

    @classmethod
    def get_class_labels(cls) -> dict[str, str]:
        from src.utils.helpers import load_class_mapping
        mapping = load_class_mapping(cls.CLASS_MAPPING_FILE)
        return mapping.get("class_labels", {})

    @classmethod
    def get_class_descriptions(cls) -> dict[str, str]:
        from src.utils.helpers import load_class_mapping
        mapping = load_class_mapping(cls.CLASS_MAPPING_FILE)
        return mapping.get("class_descriptions", {})


    # ── Image Parameters ────────────────────────────────────────────────────
    IMAGE_SIZE: tuple[int, int] = (224, 224)   # Model input resolution
    IMAGE_CHANNELS: int = 3                     # RGB channels
    INPUT_SHAPE: tuple[int, int, int] = (224, 224, 3)

    # ── Training Hyperparameters ─────────────────────────────────────────────
    BATCH_SIZE: int = 32
    EPOCHS: int = 60                    # Max epochs (more room with lower LR)
    LEARNING_RATE: float = 3e-4         # Adam LR: 3e-4 is more stable for medical imaging
    VALIDATION_SPLIT: float = 0.2       # 80/20 train-val split from Training/
    RANDOM_SEED: int = 42               # Global reproducibility seed
    DROPOUT_RATE: float = 0.4           # Reduced from 0.5: less aggressive with label smoothing

    # ── Data Augmentation ────────────────────────────────────────────────────
    AUGMENTATION_PARAMS: dict = {
        "rotation_range": 15,           # Degrees of random rotation
        "width_shift_range": 0.1,       # Fraction of width for horizontal shift
        "height_shift_range": 0.1,      # Fraction of height for vertical shift
        "zoom_range": 0.1,              # Random zoom factor
        "horizontal_flip": True,        # Randomly flip horizontally
        "brightness_range": (0.8, 1.2), # Random brightness adjustment
    }

    # ── Model Checkpoint ────────────────────────────────────────────────────
    BEST_MODEL_PATH: Path = SAVED_MODELS_DIR / "best_model.keras"
    FINAL_MODEL_PATH: Path = SAVED_MODELS_DIR / "final_model.keras"
    MODEL_WEIGHTS_PATH: Path = SAVED_MODELS_DIR / "best_weights.weights.h5"

    # ── Grad-CAM ────────────────────────────────────────────────────────────
    # Name of the last convolutional layer in the custom CNN
    GRADCAM_LAYER_NAME: str = "last_conv_layer"
    GRADCAM_ALPHA: float = 0.4          # Heatmap overlay transparency

    # ── Early Stopping & LR Scheduling ───────────────────────────────────
    EARLY_STOPPING_PATIENCE: int = 12   # Epochs with no val_accuracy improvement
    LR_REDUCE_PATIENCE: int = 5         # Epochs before reducing LR on plateau
    LR_REDUCE_FACTOR: float = 0.5       # Factor to reduce LR by
    LR_MIN: float = 1e-6                # Lower bound on learning rate
    LABEL_SMOOTHING: float = 0.05       # Categorical crossentropy label smoothing

    # ── Confidence Threshold & MRI Validation ────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.50   # Calibrated for 4-class classification (baseline random guess is 0.25)
    MAX_COLOR_DIVERGENCE: float = 12.0   # Max RGB channel std-dev (MRI is monochromatic)
    MIN_BACKGROUND_DARKNESS_RATIO: float = 0.20  # Minimum dark border pixel ratio
    MIN_BRAIN_CONTRAST: float = 15.0     # Minimum luminance standard deviation

    # ── Visualization ───────────────────────────────────────────────────────
    FIGURE_DPI: int = 150
    FIGURE_SIZE_LARGE: tuple[int, int] = (14, 6)
    FIGURE_SIZE_MEDIUM: tuple[int, int] = (10, 8)
    FIGURE_SIZE_SMALL: tuple[int, int] = (8, 6)
    COLORMAP_GRADCAM: str = "jet"       # Colormap for Grad-CAM heatmap overlay
    CONFUSION_MATRIX_CMAP: str = "Blues"

    # ── CSV Logger ──────────────────────────────────────────────────────────
    TRAINING_CSV_LOG: Path = METRICS_DIR / "training_history.csv"

    @classmethod
    def all_output_dirs(cls) -> list[Path]:
        """Return a list of all output directories that should be created."""
        return [
            cls.DATA_DIR,
            cls.RAW_DIR,
            cls.PROCESSED_DIR,
            cls.METADATA_DIR,
            cls.OUTPUTS_DIR,
            cls.SAVED_MODELS_DIR,
            cls.PLOTS_DIR,
            cls.GRADCAM_DIR,
            cls.PREDICTIONS_DIR,
            cls.METRICS_DIR,
            cls.CONFUSION_MATRIX_DIR,
            cls.LOGS_DIR,
        ]

    @classmethod
    def display(cls) -> None:
        """Print all configuration values to console for debugging."""
        print("=" * 60)
        print("  FROM GENES TO DECISIONS — Project Configuration")
        print("=" * 60)
        for attr, value in vars(cls).items():
            if not attr.startswith("_") and not callable(value):
                print(f"  {attr:<30} = {value}")
        print("=" * 60)
