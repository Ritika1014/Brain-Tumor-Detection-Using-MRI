# API Reference — From Genes to Decisions

## Module Index

| Module | Path | Description |
|---|---|---|
| `config` | `src/config.py` | Central configuration class |
| `data_loader` | `src/preprocessing/data_loader.py` | TF dataset pipeline |
| `augmentation` | `src/preprocessing/augmentation.py` | Keras augmentation layers |
| `image_utils` | `src/preprocessing/image_utils.py` | Image I/O and preprocessing |
| `custom_cnn` | `src/models/custom_cnn.py` | CNN architecture builder |
| `callbacks` | `src/models/callbacks.py` | Training callback factory |
| `trainer` | `src/training/trainer.py` | Training loop |
| `metrics` | `src/evaluation/metrics.py` | Evaluation metrics |
| `visualizer` | `src/evaluation/visualizer.py` | Plot generators |
| `gradcam` | `src/explainability/gradcam.py` | Grad-CAM XAI |
| `inference` | `src/deployment/inference.py` | Production classifier |
| `logger` | `src/utils/logger.py` | Logging factory |
| `helpers` | `src/utils/helpers.py` | General utilities |

---

## `src/config.py` — Config

```python
from src.config import Config

Config.PROJECT_ROOT        # Path: project root directory
Config.TRAIN_DIR           # Path: data/raw/Training/
Config.TEST_DIR            # Path: data/raw/Testing/
Config.BEST_MODEL_PATH     # Path: outputs/saved_models/best_model.keras
Config.CLASS_NAMES         # list: ["glioma", "meningioma", "notumor", "pituitary"]
Config.IMAGE_SIZE          # tuple: (224, 224)
Config.INPUT_SHAPE         # tuple: (224, 224, 3)
Config.NUM_CLASSES         # int: 4
Config.BATCH_SIZE          # int: 32
Config.EPOCHS              # int: 50
Config.LEARNING_RATE       # float: 0.001
Config.DROPOUT_RATE        # float: 0.5
Config.GRADCAM_LAYER_NAME  # str: "last_conv_layer"
Config.all_output_dirs()   # → list[Path]
Config.display()           # → None (prints all values)
```

---

## `src/preprocessing/data_loader.py`

### `load_dataset_paths(data_dir, class_names=None, shuffle=True) → pd.DataFrame`

Scan a directory and return a DataFrame of `(filepath, label, class_name)`.

```python
from src.preprocessing.data_loader import load_dataset_paths
df = load_dataset_paths(Config.TRAIN_DIR)
# df.columns: ["filepath", "label", "class_name"]
```

### `get_class_distribution(df) → pd.DataFrame`

```python
dist = get_class_distribution(df)
# dist.columns: ["class_name", "count", "percentage"]
```

### `build_tf_dataset(data_dir, split, ...) → Dataset | tuple[Dataset, Dataset]`

```python
train_ds, val_ds = build_tf_dataset(Config.TRAIN_DIR, split="train", augment=True)
test_ds = build_tf_dataset(Config.TEST_DIR, split="test")
```

### `extract_labels(dataset) → np.ndarray`

Extract integer class labels from a batched dataset.

---

## `src/models/custom_cnn.py`

### `build_custom_cnn(input_shape, num_classes, dropout_rate) → keras.Model`

```python
from src.models.custom_cnn import build_custom_cnn
model = build_custom_cnn()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
```

### `get_model_summary(model, save_path=None) → str`

---

## `src/training/trainer.py`

### `train_model(train_ds, val_ds, epochs, learning_rate, ...) → History`

```python
from src.training.trainer import train_model
history = train_model(train_ds, val_ds, epochs=50)
```

### `load_best_model(model_path=Config.BEST_MODEL_PATH) → keras.Model`

```python
model = load_best_model()
```

---

## `src/evaluation/metrics.py`

### `compute_metrics(model, dataset, class_names) → dict`

```python
from src.evaluation.metrics import compute_metrics
metrics = compute_metrics(model, test_ds)
# Keys: accuracy, macro_f1, kappa, auc_macro, y_true, y_pred, y_prob, report_str
```

### `classification_report_df(metrics, class_names) → pd.DataFrame`

---

## `src/evaluation/visualizer.py`

### `plot_training_history(history, save_path, title) → plt.Figure`
### `plot_confusion_matrix(metrics, class_names, normalize, save_path) → plt.Figure`
### `plot_roc_curve(metrics, class_names, save_path) → plt.Figure`
### `save_classification_report(metrics, csv_path, json_path, txt_path) → None`

---

## `src/explainability/gradcam.py`

### `class GradCAM`

```python
from src.explainability.gradcam import GradCAM

cam = GradCAM(model, layer_name="last_conv_layer")
heatmap = cam.compute_heatmap(image_array, class_idx=None)
overlay = cam.overlay_on_image(original_image, heatmap)
fig = cam.visualize(image_array, original_image)
```

### `generate_gradcam_heatmap(model, image_array, layer_name, class_idx) → np.ndarray`
### `overlay_heatmap(original_image, heatmap, alpha) → np.ndarray`
### `batch_gradcam(model, dataset, class_names, num_samples_per_class, save_dir) → list[Figure]`

---

## `src/deployment/inference.py`

### `class TumorClassifier`

```python
from src.deployment.inference import TumorClassifier

classifier = TumorClassifier()
result = classifier.predict_from_path("mri.jpg")
result = classifier.predict_from_bytes(image_bytes)
result = classifier.predict_from_array(image_array)

# result: PredictionResult
result.class_name          # str
result.class_label         # str
result.confidence          # float
result.all_probabilities   # dict[str, float]
result.gradcam_heatmap     # np.ndarray (H, W)
result.gradcam_overlay     # np.ndarray (H, W, 3) uint8
result.original_image      # np.ndarray (H, W, 3) uint8
result.description         # str
```

---

## `src/utils/logger.py`

### `get_logger(name, log_file=None, ...) → logging.Logger`

```python
from src.utils.logger import get_logger
logger = get_logger(__name__, log_file=Config.LOG_FILE)
logger.info("Training started.")
```

---

## `src/utils/helpers.py`

### `set_seed(seed=42) → None`
### `ensure_dirs(*paths) → None`
### `save_json(data, filepath) → None`
### `load_json(filepath) → Any`
### `format_metrics_table(metrics) → str`
### `get_class_color_map(class_names) → dict[str, str]`
