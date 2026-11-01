import json
from pathlib import Path


def create_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def create_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }


def make_notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }


ROOT = Path(r"c:/Users/ritik/OneDrive/Desktop/From_Genes_to_Decisions")
NB_DIR = ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Notebook 03: Custom CNN
# ---------------------------------------------------------------------------
nb03_cells = [
    create_markdown_cell([
        "# Notebook 03: Custom CNN — Architecture, Training & History\n",
        "\n",
        "This notebook covers:\n",
        "1. Building the Custom CNN using the Keras Functional API\n",
        "2. Understanding each layer's purpose and output shape\n",
        "3. Compiling with Adam, CategoricalCrossentropy + Label Smoothing\n",
        "4. Training with callbacks: ModelCheckpoint, EarlyStopping, ReduceLR\n",
        "5. Plotting Accuracy and Loss curves\n",
    ]),
    create_code_cell([
        "import sys\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
        "sys.path.insert(0, str(PROJECT_ROOT))\n",
        "\n",
        "from src.config import Config\n",
        "from src.utils.helpers import set_seed, ensure_dirs\n",
        "from src.utils.logger import get_logger\n",
        "\n",
        "logger = get_logger(__name__)\n",
        "set_seed(Config.RANDOM_SEED)\n",
        "ensure_dirs(*Config.all_output_dirs())\n",
        "print('Environment ready.')\n",
        "print(f'Image size  : {Config.IMAGE_SIZE}')\n",
        "print(f'Num classes : {Config.NUM_CLASSES}')\n",
        "print(f'Classes     : {Config.CLASS_NAMES}')\n",
    ]),
    create_markdown_cell([
        "## Step 1: Build TF Datasets\n",
        "\n",
        "We build the TF datasets with data augmentation enabled for training. "
        "Augmentation is applied ON-DEVICE inside the tf.data pipeline.\n",
    ]),
    create_code_cell([
        "from src.preprocessing.data_loader import build_tf_dataset\n",
        "\n",
        "print('Building datasets...')\n",
        "train_ds, val_ds = build_tf_dataset(\n",
        "    Config.TRAIN_DIR,\n",
        "    split='train',\n",
        "    batch_size=Config.BATCH_SIZE,\n",
        "    augment=True,\n",
        ")\n",
        "test_ds = build_tf_dataset(Config.TEST_DIR, split='test')\n",
        "print(f'Train batches : {len(train_ds)}')\n",
        "print(f'Val batches   : {len(val_ds)}')\n",
        "print(f'Test batches  : {len(test_ds)}')\n",
    ]),
    create_markdown_cell([
        "## Step 2: Build Custom CNN\n",
        "\n",
        "The Custom CNN uses the Keras Functional API for flexibility to inspect "
        "intermediate layers (required for Grad-CAM).\n",
        "\n",
        "**Architecture:**\n",
        "- Block 1: Conv(32)×2 + BN + MaxPool → 112×112×32\n",
        "- Block 2: Conv(64)×2 + BN + MaxPool → 56×56×64\n",
        "- Block 3: Conv(128)×2 + BN + MaxPool → 28×28×128\n",
        "- Block 4: Conv(256)×2 + BN → 28×28×256 ← **GradCAM Layer**\n",
        "- GlobalAvgPool → Dense(512) → Dense(256) → Dense(4, softmax)\n",
    ]),
    create_code_cell([
        "import tensorflow as tf\n",
        "from src.models.custom_cnn import build_custom_cnn, get_model_summary\n",
        "\n",
        "model = build_custom_cnn(\n",
        "    input_shape=Config.INPUT_SHAPE,\n",
        "    num_classes=Config.NUM_CLASSES,\n",
        "    dropout_rate=Config.DROPOUT_RATE,\n",
        ")\n",
        "summary_str = get_model_summary(model, save_path=Config.METRICS_DIR / 'model_summary.txt')\n",
        "\n",
        "trainable_params = sum(np.prod(v.shape) for v in model.trainable_variables)\n",
        "print(f'\\nTrainable parameters : {trainable_params:,}')\n",
        "print(f'Total parameters     : {model.count_params():,}')\n",
    ]),
    create_markdown_cell([
        "## Step 3: Compile Model\n",
        "\n",
        "- **Optimizer**: Adam (LR=1e-3, decays via ReduceLROnPlateau)\n",
        "- **Loss**: CategoricalCrossentropy with Label Smoothing (α=0.1)\n",
        "  - Smoothing prevents overconfident predictions, improves generalization\n",
        "- **Metrics**: Accuracy + AUC\n",
    ]),
    create_code_cell([
        "optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)\n",
        "loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)\n",
        "\n",
        "model.compile(\n",
        "    optimizer=optimizer,\n",
        "    loss=loss,\n",
        "    metrics=[\n",
        "        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),\n",
        "        tf.keras.metrics.AUC(name='auc'),\n",
        "    ],\n",
        ")\n",
        "print('Model compiled.')\n",
        "print(f'Optimizer : Adam(lr={Config.LEARNING_RATE})')\n",
        "print(f'Loss      : CategoricalCrossentropy(label_smoothing=0.1)')\n",
    ]),
    create_markdown_cell([
        "## Step 4: Callbacks\n",
        "\n",
        "| Callback | Purpose |\n",
        "|---|---|\n",
        "| ModelCheckpoint | Save model whenever val_accuracy improves |\n",
        "| EarlyStopping | Stop after 10 epochs without improvement |\n",
        "| ReduceLROnPlateau | Halve LR after 5 epochs of no val_loss improvement |\n",
        "| CSVLogger | Write metrics to CSV for offline analysis |\n",
    ]),
    create_code_cell([
        "from src.models.callbacks import get_callbacks\n",
        "\n",
        "callbacks = get_callbacks()\n",
        "print(f'{len(callbacks)} callbacks registered:')\n",
        "for cb in callbacks:\n",
        "    print(f'  - {cb.__class__.__name__}')\n",
    ]),
    create_markdown_cell([
        "## Step 5: Class Weights\n",
        "\n",
        "Class weights balance the training loss across under-represented classes.\n"
        "This is critical for imbalanced MRI datasets.\n",
    ]),
    create_code_cell([
        "from sklearn.utils.class_weight import compute_class_weight\n",
        "\n",
        "all_labels = []\n",
        "for _, labels in train_ds:\n",
        "    all_labels.append(np.argmax(labels.numpy(), axis=1))\n",
        "all_labels = np.concatenate(all_labels)\n",
        "\n",
        "class_weights = compute_class_weight(\n",
        "    class_weight='balanced',\n",
        "    classes=np.unique(all_labels),\n",
        "    y=all_labels,\n",
        ")\n",
        "class_weight_dict = dict(enumerate(class_weights))\n",
        "print('Class Weights:')\n",
        "for name, weight in zip(Config.CLASS_NAMES, class_weights):\n",
        "    print(f'  {name:<15} -> {weight:.4f}')\n",
    ]),
    create_markdown_cell([
        "## Step 6: Train the Model\n",
        "\n",
        "Training runs for up to `Config.EPOCHS` epochs. EarlyStopping will stop "
        "training early if validation accuracy stops improving.\n",
        "\n",
        "> **Best model** is automatically saved to `outputs/saved_models/best_model.keras`\n",
    ]),
    create_code_cell([
        "print(f'Training for up to {Config.EPOCHS} epochs...')\n",
        "history = model.fit(\n",
        "    train_ds,\n",
        "    validation_data=val_ds,\n",
        "    epochs=Config.EPOCHS,\n",
        "    callbacks=callbacks,\n",
        "    class_weight=class_weight_dict,\n",
        "    verbose=1,\n",
        ")\n",
        "print('Training complete!')\n",
        "best_val_acc = max(history.history['val_accuracy'])\n",
        "print(f'Best val_accuracy : {best_val_acc:.4f} ({best_val_acc:.2%})')\n",
        "print(f'Epochs trained    : {len(history.history[\"accuracy\"])}')\n",
    ]),
    create_markdown_cell([
        "## Step 7: Plot Training History\n",
        "\n",
        "- **Accuracy curve**: gap between train/val shows overfitting\n",
        "- **Loss curve**: should both decrease; divergence = overfitting\n",
    ]),
    create_code_cell([
        "from src.evaluation.visualizer import plot_training_history\n",
        "\n",
        "fig = plot_training_history(\n",
        "    history,\n",
        "    save_path=Config.PLOTS_DIR / 'training_history.png',\n",
        "    title='Custom CNN - Training History',\n",
        ")\n",
        "plt.show()\n",
        "print(f'Plot saved: {Config.PLOTS_DIR}/training_history.png')\n",
    ]),
    create_code_cell([
        "from src.utils.helpers import save_json\n",
        "\n",
        "history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}\n",
        "save_json(history_data, Config.METRICS_DIR / 'training_history.json')\n",
        "print(f'History saved: {Config.METRICS_DIR}/training_history.json')\n",
        "print('\\nNotebook 03 Complete! -> Open 04_Model_Evaluation.ipynb')\n",
    ]),
]

nb03 = make_notebook(nb03_cells)
(NB_DIR / "03_Custom_CNN.ipynb").write_text(
    json.dumps(nb03, indent=1, ensure_ascii=False), encoding="utf-8"
)
print("Created: 03_Custom_CNN.ipynb")

# ---------------------------------------------------------------------------
# Notebook 04: Model Evaluation
# ---------------------------------------------------------------------------
nb04_cells = [
    create_markdown_cell([
        "# Notebook 04: Model Evaluation\n",
        "\n",
        "Covers: Metrics computation, Confusion Matrix, ROC Curves, and Classification Report.\n",
        "\n",
        "**Run after Notebook 03 (model must be trained).**\n",
    ]),
    create_code_cell([
        "import sys\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
        "sys.path.insert(0, str(PROJECT_ROOT))\n",
        "\n",
        "from src.config import Config\n",
        "from src.utils.helpers import ensure_dirs, save_json\n",
        "\n",
        "ensure_dirs(*Config.all_output_dirs())\n",
        "print('Environment ready.')\n",
    ]),
    create_markdown_cell(["## Step 1: Load Best Model\n"]),
    create_code_cell([
        "from src.training.trainer import load_best_model\n",
        "\n",
        "print(f'Loading model from: {Config.BEST_MODEL_PATH}')\n",
        "model = load_best_model()\n",
        "print(f'Model loaded. Parameters: {model.count_params():,}')\n",
    ]),
    create_markdown_cell(["## Step 2: Build Test Dataset\n"]),
    create_code_cell([
        "from src.preprocessing.data_loader import build_tf_dataset\n",
        "\n",
        "test_ds = build_tf_dataset(Config.TEST_DIR, split='test')\n",
        "print(f'Test dataset: {len(test_ds)} batches.')\n",
    ]),
    create_markdown_cell(["## Step 3: Compute All Metrics\n"]),
    create_code_cell([
        "from src.evaluation.metrics import compute_metrics, classification_report_df\n",
        "\n",
        "print('Computing evaluation metrics...')\n",
        "metrics = compute_metrics(model, test_ds, class_names=Config.CLASS_NAMES)\n",
        "\n",
        "scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}\n",
        "print('\\nEVALUATION RESULTS:')\n",
        "for name, value in scalar_metrics.items():\n",
        "    print(f'  {name:<30} {value:.4f}')\n",
    ]),
    create_markdown_cell(["## Step 4: Classification Report\n"]),
    create_code_cell([
        "print(metrics['report_str'])\n",
        "report_df = classification_report_df(metrics, Config.CLASS_NAMES)\n",
        "print('\\nDataFrame:')\n",
        "report_df\n",
    ]),
    create_markdown_cell([
        "## Step 5: Confusion Matrix\n",
        "\n",
        "- **Normalized**: fraction per true class (shows per-class error rates)\n",
        "- **Raw counts**: actual sample counts\n",
    ]),
    create_code_cell([
        "from src.evaluation.visualizer import plot_confusion_matrix\n",
        "\n",
        "fig_norm = plot_confusion_matrix(\n",
        "    metrics, normalize=True,\n",
        "    save_path=Config.CONFUSION_MATRIX_DIR / 'confusion_matrix_normalized.png',\n",
        "    title='Normalized Confusion Matrix',\n",
        ")\n",
        "plt.show()\n",
        "\n",
        "fig_counts = plot_confusion_matrix(\n",
        "    metrics, normalize=False,\n",
        "    save_path=Config.CONFUSION_MATRIX_DIR / 'confusion_matrix_counts.png',\n",
        "    title='Confusion Matrix - Sample Counts',\n",
        ")\n",
        "plt.show()\n",
    ]),
    create_markdown_cell([
        "## Step 6: ROC Curves\n",
        "\n",
        "Multi-class ROC using **One-vs-Rest (OvR)** binarization. "
        "Each class has its own AUC score; macro-average is the summary.\n",
    ]),
    create_code_cell([
        "from src.evaluation.visualizer import plot_roc_curve\n",
        "\n",
        "fig_roc = plot_roc_curve(\n",
        "    metrics,\n",
        "    save_path=Config.PLOTS_DIR / 'roc_curves.png',\n",
        "    title='Multi-Class ROC Curves (One-vs-Rest)',\n",
        ")\n",
        "plt.show()\n",
    ]),
    create_markdown_cell(["## Step 7: Save All Reports\n"]),
    create_code_cell([
        "from src.evaluation.visualizer import save_classification_report\n",
        "\n",
        "save_classification_report(\n",
        "    metrics,\n",
        "    csv_path=Config.METRICS_DIR / 'classification_report.csv',\n",
        "    json_path=Config.METRICS_DIR / 'classification_report.json',\n",
        "    txt_path=Config.METRICS_DIR / 'classification_report.txt',\n",
        ")\n",
        "save_json(scalar_metrics, Config.METRICS_DIR / 'scalar_metrics.json')\n",
        "print('All reports saved!')\n",
        "print('Next -> Open 05_Explainable_AI_GradCAM.ipynb')\n",
    ]),
]

nb04 = make_notebook(nb04_cells)
(NB_DIR / "04_Model_Evaluation.ipynb").write_text(
    json.dumps(nb04, indent=1, ensure_ascii=False), encoding="utf-8"
)
print("Created: 04_Model_Evaluation.ipynb")

print("All notebooks created successfully!")
