"""
================================================================================
Notebook 03: Custom CNN — Architecture, Training & History
================================================================================

This notebook covers:
  1. Building the Custom CNN using the Keras Functional API
  2. Understanding each layer's purpose and output shape
  3. Compiling with Adam, CategoricalCrossentropy + Label Smoothing
  4. Training with callbacks: ModelCheckpoint, EarlyStopping, ReduceLR
  5. Plotting Accuracy and Loss curves

Run each section sequentially after completing Notebook 02.
================================================================================
"""

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.utils.helpers import set_seed, ensure_dirs
from src.utils.logger import get_logger

logger = get_logger(__name__)
set_seed(Config.RANDOM_SEED)
ensure_dirs(*Config.all_output_dirs())
print("✅ Environment ready.")
print(f"   Image size  : {Config.IMAGE_SIZE}")
print(f"   Num classes : {Config.NUM_CLASSES}")
print(f"   Classes     : {Config.CLASS_NAMES}")

# ── Cell 2: Build Datasets ────────────────────────────────────────────────────
"""
We build the TF datasets with data augmentation enabled for training.
Augmentation is applied ON-DEVICE inside the tf.data pipeline,
so it doesn't slow down training.
"""
from src.preprocessing.data_loader import build_tf_dataset

print("\n📂 Building datasets...")
train_ds, val_ds = build_tf_dataset(
    Config.TRAIN_DIR,
    split="train",
    batch_size=Config.BATCH_SIZE,
    augment=True,             # Enable augmentation for training
)
test_ds = build_tf_dataset(Config.TEST_DIR, split="test")
print(f"   Train batches : {len(train_ds)}")
print(f"   Val batches   : {len(val_ds)}")
print(f"   Test batches  : {len(test_ds)}")

# ── Cell 3: Build Custom CNN ──────────────────────────────────────────────────
"""
The Custom CNN uses the Functional API for:
- Flexibility to inspect any intermediate layer (required for Grad-CAM)
- Clean layer naming for architecture documentation
"""
import tensorflow as tf
from src.models.custom_cnn import build_custom_cnn, get_model_summary

model = build_custom_cnn(
    input_shape=Config.INPUT_SHAPE,
    num_classes=Config.NUM_CLASSES,
    dropout_rate=Config.DROPOUT_RATE,
)
summary_str = get_model_summary(model, save_path=Config.METRICS_DIR / "model_summary.txt")

# Count trainable parameters
trainable_params = sum(
    np.prod(v.shape) for v in model.trainable_variables
)
print(f"\n📊 Model Statistics:")
print(f"   Trainable parameters : {trainable_params:,}")
print(f"   Total parameters     : {model.count_params():,}")

# ── Cell 4: Compile Model ─────────────────────────────────────────────────────
"""
Compile with:
  - Adam optimizer (LR=1e-3, decays via ReduceLROnPlateau)
  - CategoricalCrossentropy with Label Smoothing (α=0.1)
    → Smoothing prevents overconfident predictions, improves generalization
  - Metrics: Accuracy + AUC
"""
optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)
print("✅ Model compiled.")
print(f"   Optimizer    : Adam(lr={Config.LEARNING_RATE})")
print(f"   Loss         : CategoricalCrossentropy(label_smoothing=0.1)")

# ── Cell 5: Get Callbacks ─────────────────────────────────────────────────────
"""
Callbacks used:
1. ModelCheckpoint : Save model whenever val_accuracy improves.
2. EarlyStopping   : Stop after 10 epochs without improvement (restores best weights).
3. ReduceLROnPlateau: Halve LR after 5 epochs of no val_loss improvement.
4. CSVLogger       : Write metrics to CSV for offline analysis.
"""
from src.models.callbacks import get_callbacks

callbacks = get_callbacks()
print(f"✅ {len(callbacks)} callbacks registered:")
for cb in callbacks:
    print(f"   - {cb.__class__.__name__}")

# ── Cell 6: Compute Class Weights ─────────────────────────────────────────────
"""
Class weights balance the training loss across under-represented classes.
This is especially important when one class has fewer MRI samples.
"""
from sklearn.utils.class_weight import compute_class_weight

all_labels = []
for _, labels in train_ds:
    all_labels.append(np.argmax(labels.numpy(), axis=1))
all_labels = np.concatenate(all_labels)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(all_labels),
    y=all_labels,
)
class_weight_dict = dict(enumerate(class_weights))
print("⚖️  Class Weights:")
for idx, (name, weight) in enumerate(zip(Config.CLASS_NAMES, class_weights)):
    print(f"   {name:<15} → {weight:.4f}")

# ── Cell 7: Train Model ───────────────────────────────────────────────────────
"""
Training the model. With EarlyStopping and ReduceLR, the model will:
  - Stop before MAX_EPOCHS if validation accuracy plateaus
  - Automatically reduce learning rate when validation loss stagnates
  - Save the best checkpoint to: outputs/saved_models/best_model.keras
"""
print(f"\n🚀 Training for up to {Config.EPOCHS} epochs...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=Config.EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1,
)
print("\n✅ Training complete!")
best_val_acc = max(history.history["val_accuracy"])
print(f"   Best val_accuracy : {best_val_acc:.4f} ({best_val_acc:.2%})")
print(f"   Epochs trained    : {len(history.history['accuracy'])}")

# ── Cell 8: Plot Training History ─────────────────────────────────────────────
"""
Visualize training progress:
  - Accuracy curve: gap between train/val shows overfitting
  - Loss curve: should both decrease; divergence = overfitting
"""
from src.evaluation.visualizer import plot_training_history

fig = plot_training_history(
    history,
    save_path=Config.PLOTS_DIR / "training_history.png",
    title="Custom CNN — Training History",
)
plt.show()
print(f"✅ Plot saved: {Config.PLOTS_DIR / 'training_history.png'}")

# ── Cell 9: Save Training History to JSON ────────────────────────────────────
from src.utils.helpers import save_json

history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
save_json(history_data, Config.METRICS_DIR / "training_history.json")
print(f"✅ History saved: {Config.METRICS_DIR / 'training_history.json'}")

print("""
================================================================================
  Notebook 03 Complete!

  Next steps:
  → Open 04_Model_Evaluation.ipynb to evaluate on the test set.
  
  Key outputs:
  ✅ outputs/saved_models/best_model.keras      ← Best checkpoint
  ✅ outputs/metrics/model_summary.txt          ← Architecture summary
  ✅ outputs/plots/training_history.png         ← Accuracy/Loss curves
  ✅ outputs/metrics/training_history.json      ← Raw history data
================================================================================
""")
