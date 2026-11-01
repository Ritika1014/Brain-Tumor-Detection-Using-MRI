# System Architecture — From Genes to Decisions

## Overview

The project follows a modular, layered pipeline architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  Streamlit App (app/app.py) | CLI (main.py) | Jupyter Notebooks │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      DEPLOYMENT LAYER                           │
│               src/deployment/inference.py                       │
│           TumorClassifier · PredictionResult                    │
└──────┬────────────────────────────────────────────┬────────────┘
       │                                            │
┌──────▼──────────┐                      ┌──────────▼──────────────┐
│  PREPROCESSING  │                      │    EXPLAINABILITY       │
│  data_loader.py │                      │    gradcam.py           │
│  augmentation.py│                      │    GradCAM class        │
│  image_utils.py │                      │    batch_gradcam()      │
└──────┬──────────┘                      └──────────┬──────────────┘
       │                                            │
┌──────▼──────────────────────────────────▼────────┐
│                   MODEL LAYER                     │
│   custom_cnn.py (build_custom_cnn)                │
│   callbacks.py (ModelCheckpoint, EarlyStopping)   │
│   trainer.py (train_model, load_best_model)       │
└──────────────────────────┬────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────┐
│                EVALUATION LAYER                    │
│   metrics.py (compute_metrics, get_predictions)   │
│   visualizer.py (training history, CM, ROC)       │
└──────────────────────────┬────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────┐
│                   UTILITIES                        │
│   config.py · logger.py · helpers.py              │
└────────────────────────────────────────────────────┘
```

## Data Flow

1. **Raw MRI Images** (JPEG/PNG) → `data/raw/Training/` and `data/raw/Testing/`
2. **Data Loader** → Builds `tf.data.Dataset` with normalization + optional augmentation
3. **Custom CNN** → Extracts features, outputs class probabilities
4. **Grad-CAM** → Hooks into `last_conv_layer`, computes gradient-weighted heatmap
5. **Outputs** → Saved to `outputs/{saved_models, plots, gradcam, metrics}`
6. **Streamlit** → Loads saved model, preprocesses uploaded image, shows results
