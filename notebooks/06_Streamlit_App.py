"""
================================================================================
Notebook 06: Streamlit App — Deployment Guide & Demo
================================================================================

This notebook covers:
  1. Overview of the Streamlit application architecture
  2. How the inference pipeline works end-to-end
  3. How to run the app locally
  4. Testing the app programmatically (without launching the UI)
  5. Deployment considerations for production
================================================================================
"""

# ── Cell 1: App Architecture Overview ────────────────────────────────────────
"""
STREAMLIT APP ARCHITECTURE
============================

The Streamlit app (app/app.py) is organized into three pages:

  🏠 Home Page
  ├── Hero banner with project description
  ├── Key statistics (4 classes, 224×224, CNN, XAI)
  ├── Tumor class descriptions with expandable details
  ├── How-to-use guide
  └── Architecture pipeline diagram

  🔬 Predict MRI Page
  ├── File uploader (JPG/PNG/BMP)
  ├── Image info display (dimensions, size, mode)
  ├── Run prediction button
  ├── Prediction result box (with class color coding)
  ├── Grad-CAM three-panel display (original | heatmap | overlay)
  ├── Class probability bar chart
  └── Medical disclaimer

  📊 About & Methods Page
  ├── Architecture table (layer shapes, parameter counts)
  ├── Training methodology details
  ├── Grad-CAM mathematical formulation
  └── Academic references

DEPLOYMENT PIPELINE:
  1. @st.cache_resource loads TumorClassifier once
  2. User uploads image → bytes are read
  3. bytes_to_array() preprocesses to (1, 224, 224, 3) float32
  4. model.predict() → softmax probabilities
  5. GradCAM.compute_heatmap() → 2D heatmap
  6. overlay_on_image() → blended visualization
  7. Results displayed in the Streamlit UI
"""
print("📋 App architecture overview above ↑")
print()

# ── Cell 2: Test the Inference Pipeline Programmatically ─────────────────────
"""
We can test the full inference pipeline without launching Streamlit.
This validates that all modules work correctly before deployment.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config

# Check if model exists
if not Config.BEST_MODEL_PATH.exists():
    print("⚠️  Model not found. Run notebook 03 first to train the model.")
    print(f"   Expected: {Config.BEST_MODEL_PATH}")
else:
    from src.deployment.inference import TumorClassifier

    print("📦 Loading TumorClassifier...")
    classifier = TumorClassifier()
    print("✅ Classifier ready.")

    # ── Test with a sample from the test set ──────────────────────────────────
    from src.preprocessing.data_loader import build_tf_dataset
    test_ds = build_tf_dataset(Config.TEST_DIR, split="test", batch_size=1)

    for images, labels in test_ds.take(1):
        sample_img = images[0].numpy()
        true_class = Config.CLASS_NAMES[np.argmax(labels[0])]

    # Predict
    result = classifier.predict_from_array(sample_img)

    print(f"\n🔬 Sample Prediction:")
    print(f"   True class       : {true_class}")
    print(f"   Predicted class  : {result.class_label}")
    print(f"   Confidence       : {result.confidence:.4f} ({result.confidence:.2%})")
    print(f"   Class index      : {result.class_idx}")
    print(f"   GradCAM shape    : {result.gradcam_heatmap.shape}")
    print(f"   Overlay shape    : {result.gradcam_overlay.shape}")

    print(f"\n   All probabilities:")
    for cls, prob in result.all_probabilities.items():
        bar = "█" * int(prob * 20)
        print(f"   {cls:<15} : {prob:.4f} {bar}")

# ── Cell 3: Launch Instructions ───────────────────────────────────────────────
print("""
================================================================================
  RUNNING THE STREAMLIT APP
================================================================================

  Prerequisites:
  ✅ Model trained (python main.py --stage train)
  ✅ Dependencies installed (pip install -r requirements.txt)

  Launch command:
  ┌──────────────────────────────────────────────────┐
  │  streamlit run app/app.py                        │
  └──────────────────────────────────────────────────┘

  The app opens at: http://localhost:8501

  Usage:
  1. Navigate to "🔬 Predict MRI" in the sidebar
  2. Upload a brain MRI image (JPG/PNG)
  3. Click "Analyze MRI"
  4. View predictions, Grad-CAM, and confidence scores

================================================================================
  DEPLOYMENT OPTIONS
================================================================================

  Local Development:
    streamlit run app/app.py

  Production (Streamlit Cloud):
    1. Push to GitHub
    2. Connect at share.streamlit.io
    3. Set main file: app/app.py

  Docker:
    FROM python:3.11-slim
    WORKDIR /app
    COPY . .
    RUN pip install -r requirements.txt
    EXPOSE 8501
    CMD ["streamlit", "run", "app/app.py", "--server.port=8501"]

  Cloud Run:
    gcloud run deploy brain-tumor-detector \\
      --source . \\
      --port 8501 \\
      --region us-central1

================================================================================
  PROJECT COMPLETE! 🎉
================================================================================

  All 6 notebooks completed:
  ✅ 01_Data_Understanding      → EDA & class distribution
  ✅ 02_Preprocessing           → Augmentation & TF pipeline
  ✅ 03_Custom_CNN              → Architecture & training
  ✅ 04_Model_Evaluation        → Metrics, CM, ROC
  ✅ 05_Explainable_AI_GradCAM → Grad-CAM visualizations
  ✅ 06_Streamlit_App           → Deployment guide

  Key artifacts:
  📂 outputs/saved_models/best_model.keras
  📂 outputs/plots/training_history.png
  📂 outputs/confusion_matrix/
  📂 outputs/plots/roc_curves.png
  📂 outputs/metrics/classification_report.csv
  📂 outputs/gradcam/

  Next Research Steps:
  🔬 Phase 2: Transfer Learning (ResNet50V2, EfficientNetB3)
  🔬 Phase 3: Attention Mechanisms
  🔬 Phase 4: Gene Expression Multi-modal Fusion
  🔬 Phase 5: Synthetic MRI with Diffusion Models
================================================================================
""")
