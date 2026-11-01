"""
scripts/pdf_chapters_13_to_18.py
Chapters 13 through 18 for the Master PDF report.
Includes PageBreaks between chapters and rich technical depth.
"""

from reportlab.platypus import Paragraph, Spacer, PageBreak, KeepTogether, HRFlowable
from reportlab.lib import colors

def add_chapter_13(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 13 — FILE-BY-FILE CODE EXPLANATION", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("13.1 Systematic Code Audit Protocol", styles["SectionHeading"]))
    story.append(Paragraph(
        "This chapter provides an exhaustive forensic code audit of every Python module, helper script, and Jupyter notebook "
        "in the repository. To maximize educational clarity for students and viva examiners, each function and class is explained "
        "in terms of its purpose, parameters, return values, algorithmic logic, and role in the pipeline.",
        styles["ReportBody"]
    ))

    # --- Module 1: src/config.py ---
    story.append(Paragraph("13.2 Module Audit: src/config.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Single source of truth for all global hyperparameters, paths, and metadata constants.<br/>"
        "• <code>ConfigMeta</code>: Metaclass enabling dynamic class-level properties. Ensures that calling <code>Config.CLASS_NAMES</code> or <code>Config.NUM_CLASSES</code> dynamically interrogates the dataset folder without hardcoding.<br/>"
        "• <code>Config</code>: Core class defining paths (<code>PROJECT_ROOT</code>, <code>DATA_DIR</code>, <code>OUTPUTS_DIR</code>), image parameters (<code>IMAGE_SIZE=(224, 224)</code>), training hyperparameters (<code>BATCH_SIZE=32</code>, <code>EPOCHS=60</code>, <code>LEARNING_RATE=3e-4</code>, <code>DROPOUT_RATE=0.4</code>), callbacks settings (<code>EARLY_STOPPING_PATIENCE=12</code>, <code>LR_REDUCE_PATIENCE=5</code>), and safety validation thresholds (<code>MAX_COLOR_DIVERGENCE=12.0</code>, <code>CONFIDENCE_THRESHOLD=0.50</code>).<br/>"
        "• <code>Config.all_output_dirs()</code>: Utility method returning all directories that must exist prior to pipeline execution.",
        styles["ReportBody"]
    ))

    # --- Module 2: src/preprocessing/data_loader.py ---
    story.append(Paragraph("13.3 Module Audit: src/preprocessing/data_loader.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Ingests raw images and constructs high-performance, asynchronous TensorFlow input pipelines.<br/>"
        "• <code>load_dataset_paths(data_dir, class_names, shuffle, seed)</code>: Scans class subdirectories, validates image file extensions (.jpg, .png, .bmp), and returns a shuffled Pandas DataFrame with columns [<code>filepath</code>, <code>label</code>, <code>class_name</code>].<br/>"
        "• <code>_iter_images(directory)</code>: Generator yielding file paths matching supported extensions, avoiding memory overhead when scanning massive directories.<br/>"
        "• <code>get_class_distribution(df)</code>: Calculates absolute counts and percentage frequencies per class to monitor class balance.<br/>"
        "• <code>build_tf_dataset(data_dir, split, validation_split, batch_size, image_size, augment, seed)</code>: The core pipeline builder. Uses <code>tf.keras.utils.image_dataset_from_directory</code> to parse images into batched categorical tensors. Applies pixel rescaling (1/255), RAM caching (<code>.cache()</code>), in-memory shuffling (buffer=1000), dynamic augmentation, and asynchronous GPU prefetching (<code>.prefetch(AUTOTUNE)</code>).<br/>"
        "• <code>extract_labels(dataset)</code>: Iterates through batches to extract ground-truth integer labels into a single NumPy array for evaluation.",
        styles["ReportBody"]
    ))

    # --- Module 3: src/preprocessing/augmentation.py ---
    story.append(Paragraph("13.4 Module Audit: src/preprocessing/augmentation.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Implements conservative, anatomically valid image transformations.<br/>"
        "• <code>get_augmentation_layer(rotation_factor, zoom_factor, flip_mode, translation_factor)</code>: Returns a <code>tf.keras.Sequential</code> container. Chains <code>RandomFlip('horizontal')</code>, <code>RandomRotation(0.05)</code>, <code>RandomZoom(0.05)</code>, <code>RandomTranslation(0.05)</code>, and a custom <code>Lambda(tf.clip_by_value(x, 0.0, 1.0))</code> to guarantee transformed pixels remain within valid bounds.<br/>"
        "• <code>visualize_augmentation(dataset, augmentation_layer, num_samples, save_path)</code>: Extracts a test batch and plots side-by-side comparisons of original vs. augmented MRIs for visual quality verification.",
        styles["ReportBody"]
    ))

    # --- Module 4: src/preprocessing/image_utils.py ---
    story.append(Paragraph("13.5 Module Audit: src/preprocessing/image_utils.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Low-level OpenCV and TensorFlow decoding for single-image inference.<br/>"
        "• <code>decode_image(image_path, image_size, normalize)</code>: Decodes image files using native TensorFlow operations (<code>tf.io.read_file</code> and <code>tf.image.decode_image</code>).<br/>"
        "• <code>resize_and_normalize(image, target_size)</code>: Resizes NumPy arrays via bilinear interpolation and scales pixel values to [0.0, 1.0]. Replicates single-channel grayscale arrays into 3-channel RGB.<br/>"
        "• <code>load_and_preprocess_image(image_path, target_size, expand_dims)</code>: Reads an image from disk using OpenCV, applies RGB conversion, normalization, and optionally adds a batch dimension (1, 224, 224, 3).<br/>"
        "• <code>bytes_to_array(image_bytes, target_size, expand_dims)</code>: Converts raw binary streams (from Streamlit file uploaders) into preprocessed float32 tensors without touching the local disk.",
        styles["ReportBody"]
    ))

    # --- Module 5: src/preprocessing/mri_validator.py ---
    story.append(Paragraph("13.6 Module Audit: src/preprocessing/mri_validator.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Pre-inference heuristic safety filter guarding against non-brain inputs.<br/>"
        "• <code>MRIScanValidator.__init__(max_color_divergence, min_border_darkness_ratio, min_contrast)</code>: Initializes acceptance thresholds.<br/>"
        "• <code>MRIScanValidator.validate_image(image)</code>: Executes three verification algorithms: (1) computes standard deviation across RGB channels to reject natural color images, (2) measures luminance in the outer 8% margin to ensure a dark air background, and (3) calculates standard deviation across central brain parenchyma to reject solid/blank images. Returns <code>(bool, error_message, metrics_dict)</code>.",
        styles["ReportBody"]
    ))

    # --- Module 6: src/models/custom_cnn.py ---
    story.append(Paragraph("13.7 Module Audit: src/models/custom_cnn.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Defines the 4-block deep convolutional neural network.<br/>"
        "• <code>_conv_bn_relu(x, filters, kernel_size, padding, name_prefix)</code>: Reusable building block executing Conv2D (He normal initialization, use_bias=False) → BatchNormalization → ReLU.<br/>"
        "• <code>_conv_block(x, filters, block_name, pool, last_conv_name)</code>: Stacks two Conv-BN-ReLU sequences followed by an optional MaxPooling2D(2x2). In Block 4, assigns <code>last_conv_name='last_conv_layer'</code> to hook Grad-CAM.<br/>"
        "• <code>build_custom_cnn(input_shape, num_classes, dropout_rate, gradcam_layer_name)</code>: Assembles the full Keras Functional model: Input (224x224x3) → Blocks 1 to 4 → GlobalAveragePooling2D → Dense(256) + BN + ReLU + Dropout(0.4) → Dense(4, Softmax).<br/>"
        "• <code>get_model_summary(model, save_path)</code>: Captures the architectural layer table as a string and saves it to a text file.",
        styles["ReportBody"]
    ))

    # --- Module 7: src/models/callbacks.py ---
    story.append(Paragraph("13.8 Module Audit: src/models/callbacks.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Centralized callback factory.<br/>"
        "• <code>get_callbacks(...)</code>: Instantiates and returns a list containing <code>ModelCheckpoint</code> (saving best_model.keras on max val_accuracy), <code>EarlyStopping</code> (patience=12 epochs with weight restoration), <code>ReduceLROnPlateau</code> (patience=5 epochs, factor=0.5), and <code>CSVLogger</code> (logging epoch history to training_history.csv).",
        styles["ReportBody"]
    ))

    # --- Module 8: src/training/trainer.py ---
    story.append(Paragraph("13.9 Module Audit: src/training/trainer.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Controls model compilation, weighting, training, and checkpoint restoration.<br/>"
        "• <code>_compute_class_weights()</code>: Computes Scikit-Learn balanced class weights from training labels, applies square-root dampening, and normalizes the mean to 1.0 to prevent gradient skew.<br/>"
        "• <code>train_model(train_ds, val_ds, epochs, learning_rate, use_class_weights, ...)</code>: Orchestrates model creation, compilation with Adam and label smoothing (0.05), fitting with callbacks, and saving the final model.<br/>"
        "• <code>load_best_model(model_path)</code>: Restores the trained best_model.keras checkpoint from disk, handling Keras deserialization.",
        styles["ReportBody"]
    ))

    # --- Module 9: src/evaluation/metrics.py ---
    story.append(Paragraph("13.10 Module Audit: src/evaluation/metrics.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Computes all quantitative evaluation metrics on the test dataset.<br/>"
        "• <code>get_predictions_and_labels(model, test_ds)</code>: Runs batched model inference on the test dataset, concatenating predicted probability vectors and true labels.<br/>"
        "• <code>compute_metrics(model, test_ds, class_names)</code>: Computes accuracy, macro/weighted precision, recall, F1-scores, Cohen's Kappa, One-vs-Rest ROC-AUC, confusion matrix array, and classification report text.<br/>"
        "• <code>classification_report_df(metrics_dict, class_names)</code>: Converts classification report dictionary into a clean Pandas DataFrame.",
        styles["ReportBody"]
    ))

    # --- Module 10: src/evaluation/visualizer.py ---
    story.append(Paragraph("13.11 Module Audit: src/evaluation/visualizer.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Produces publication-quality Matplotlib figures.<br/>"
        "• <code>plot_confusion_matrix(metrics_dict, normalize, save_path, title)</code>: Renders a stylized Seaborn heatmap of the confusion matrix.<br/>"
        "• <code>plot_training_history(history, save_path, title)</code>: Generates a twin-subplot figure plotting Loss and Accuracy curves across training and validation splits.<br/>"
        "• <code>plot_roc_curve(metrics_dict, save_path, title)</code>: Plots One-vs-Rest ROC curves for all four tumor classes alongside the macro-average curve and chance diagonal.<br/>"
        "• <code>save_classification_report(metrics_dict, csv_path, json_path, txt_path)</code>: Serializes performance reports to disk across CSV, JSON, and plain text formats.",
        styles["ReportBody"]
    ))

    # --- Module 11: src/explainability/gradcam.py ---
    story.append(Paragraph("13.12 Module Audit: src/explainability/gradcam.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Implements the Grad-CAM explainability engine.<br/>"
        "• <code>GradCAM.__init__(model, layer_name)</code>: Hooks into the target convolutional layer and constructs an auxiliary gradient model having two outputs: target layer feature activations and output class logits.<br/>"
        "• <code>GradCAM.compute_heatmap(image_array, class_idx)</code>: Uses <code>tf.GradientTape</code> to compute gradients of class logit y^c w.r.t. feature maps, pools gradients globally, performs linear combination with ReLU, and normalizes the heatmap to [0.0, 1.0].<br/>"
        "• <code>GradCAM.overlay_on_image(original_image, heatmap, alpha, colormap)</code>: Upsamples the 14x14 heatmap to 224x224, applies the OpenCV JET colormap, and blends it with the original MRI via alpha blending.<br/>"
        "• <code>GradCAM.visualize(...)</code>: Produces a publication-ready triple-panel figure (Original MRI, Isolated Heatmap, Overlay).",
        styles["ReportBody"]
    ))

    # --- Module 12: src/deployment/inference.py ---
    story.append(Paragraph("13.13 Module Audit: src/deployment/inference.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>Role in Pipeline:</b> Production inference engine encapsulating the full diagnostic workflow.<br/>"
        "• <code>PredictionResult</code>: Dataclass container storing predicted class name, confidence, probabilities, Grad-CAM arrays, validation status, and clinical description.<br/>"
        "• <code>TumorClassifier.__init__(model_path, class_names, confidence_threshold)</code>: Initializes the model, loads class mappings, and instantiates MRIScanValidator and GradCAM engines.<br/>"
        "• <code>TumorClassifier.predict_from_array(image_array, ...)</code>: Orchestrates validation, standardization, inference, threshold verification, and Grad-CAM generation.<br/>"
        "• <code>TumorClassifier.predict_from_bytes(image_bytes, ...)</code>: Decodes byte streams from web uploads and executes prediction.<br/>"
        "• <code>TumorClassifier.predict_from_path(image_path, ...)</code>: Loads images from disk paths and executes prediction.",
        styles["ReportBody"]
    ))

    # --- Module 13: main.py & debug_model.py ---
    story.append(Paragraph("13.14 Module Audit: main.py & debug_model.py", styles["SectionHeading"]))
    story.append(Paragraph(
        "• <code>main.py</code>: Click-based command-line interface orchestrating pipeline stages via <code>--stage [all|train|evaluate|gradcam]</code>. Provides CLI argument overrides for epochs, learning rate, batch size, and class weighting.<br/>"
        "• <code>debug_model.py</code>: Diagnostic script executing 10 automated verification checks: samples 5+ images per class from the test set, evaluates confidence distributions, isolates misclassified images into <code>debug/misclassified/</code>, and writes <code>debug/debug_report.txt</code>.",
        styles["ReportBody"]
    ))

    # --- Module 14: app/app.py & Notebooks ---
    story.append(Paragraph("13.15 Module Audit: app/app.py & Educational Notebooks", styles["SectionHeading"]))
    story.append(Paragraph(
        "• <code>app/app.py</code>: Complete 1,453-line Streamlit web dashboard. Houses the custom dark-mode CSS theme, cached model loading, reactive threshold sliders, real-time Grad-CAM computation, and 3 distinct page views.<br/>"
        "• <code>01_Data_Understanding.ipynb</code>: Explores dataset structure, pixel intensity distributions, and class frequencies.<br/>"
        "• <code>02_Preprocessing.ipynb</code>: Demonstrates image resizing, normalization, and on-the-fly data augmentation.<br/>"
        "• <code>03_Custom_CNN.ipynb / .py</code>: Step-by-step interactive construction, compilation, and training of the Custom CNN.<br/>"
        "• <code>04_Model_Evaluation.ipynb / .py</code>: Computes test metrics, plots confusion matrices, and generates ROC curves.<br/>"
        "• <code>05_Explainable_AI_GradCAM.py</code>: Dedicated tutorial demonstrating Grad-CAM theory, gradient extraction, and multi-class heatmaps.<br/>"
        "• <code>06_Streamlit_App.py</code>: Architectural specification and standalone runner for the Streamlit web dashboard.",
        styles["ReportBody"]
    ))
    story.append(PageBreak())


def add_chapter_14(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 14 — GENERATED OUTPUT FILES & ARTIFACTS", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("14.1 Complete Artifact Inventory", styles["SectionHeading"]))
    story.append(Paragraph(
        "Execution of the training, evaluation, explainability, and debugging pipelines populates the <code>outputs/</code> "
        "and <code>debug/</code> directories with reproducible artifacts. Table 14.1 documents every generated file.",
        styles["ReportBody"]
    ))

    art_headers = ["Artifact File Path", "File Format & Size", "Generating Script", "Clinical & Technical Purpose"]
    art_rows = [
        ["outputs/saved_models/best_model.keras", "Keras Zip Archive (15.0 MB)", "src/training/trainer.py", "Best checkpoint model saved by ModelCheckpoint based on maximum validation accuracy. Contains architecture, weights, and optimizer state."],
        ["outputs/saved_models/final_model.keras", "Keras Zip Archive (15.0 MB)", "src/training/trainer.py", "Model weights at the final epoch of training, used for comparative regression auditing."],
        ["outputs/metrics/scalar_metrics.json", "JSON (315 bytes)", "src/evaluation/metrics.py", "Stores scalar evaluation metrics: accuracy, macro precision, recall, F1, Cohen's Kappa, and ROC-AUC."],
        ["outputs/metrics/classification_report.json", "JSON (1.1 KB)", "src/evaluation/metrics.py", "Per-class precision, recall, F1-score, and support formatted as structured JSON."],
        ["outputs/metrics/classification_report.csv", "CSV (315 bytes)", "src/evaluation/metrics.py", "Tabular classification report for spreadsheet analysis and viva presentation."],
        ["outputs/metrics/training_history.csv", "CSV (4.4 KB)", "src/models/callbacks.py", "Epoch-by-epoch training and validation loss and accuracy logged by CSVLogger."],
        ["outputs/plots/training_history.png", "PNG Image (140.4 KB)", "src/evaluation/visualizer.py", "Dual-subplot plot showing training vs. validation loss and accuracy convergence."],
        ["outputs/plots/roc_curves.png", "PNG Image (110.9 KB)", "src/evaluation/visualizer.py", "Multi-class One-vs-Rest ROC curves displaying individual and macro-average AUC scores."],
        ["outputs/confusion_matrix/confusion_matrix_normalized.png", "PNG Image (85.4 KB)", "src/evaluation/visualizer.py", "Normalized confusion matrix heatmap illustrating per-class recall rates and misclassification percentages."],
        ["outputs/confusion_matrix/confusion_matrix_counts.png", "PNG Image (61.6 KB)", "src/evaluation/visualizer.py", "Raw integer count confusion matrix showing exact sample distributions."],
        ["outputs/gradcam/*.png (12 files)", "PNG Images (~500 KB each)", "src/explainability/gradcam.py", "Triple-panel Grad-CAM visualization figures (Original MRI, Heatmap, Overlay) for multiple samples across all 4 classes."],
        ["debug/debug_report.txt", "Text Log (3.0 KB)", "debug_model.py", "Automated debugging report summarizing sample accuracy, root causes, and confidence distributions."],
        ["debug/misclassified/*.jpg (16 files)", "JPEG Images (~20 KB each)", "debug_model.py", "Isolated image files of misclassified test samples with actual vs. predicted labels in filenames."]
    ]
    story.append(make_table(art_headers, art_rows, [125, 95, 105, 207], styles))
    story.append(PageBreak())


def add_chapter_15(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 15 — PROJECT ARCHITECTURE & DATA FLOW", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("15.1 Layered System Architecture", styles["SectionHeading"]))
    story.append(Paragraph(
        "The project is structured according to a <b>6-tier layered software architecture</b>. "
        "Each layer encapsulates specific operational boundaries with loose coupling, ensuring that modifications to the neural "
        "network architecture do not break the preprocessing pipeline or the user interface.",
        styles["ReportBody"]
    ))

    story.append(make_image_flowable(
        "outputs/doc_diagrams/system_architecture_diagram.png", width=490, height=220,
        caption="Figure 15.1: 6-Tier System Architecture Diagram illustrating data flow from storage up to client interfaces.",
        styles=styles
    ))

    story.append(Paragraph("15.2 Layer Responsibilities & Component Coupling", styles["SectionHeading"]))
    story.append(Paragraph("• <b>Layer 1 (Dataset & Storage Layer):</b> Manages persistent storage of raw Kaggle MRI files, JSON class mappings, training checkpoints (.keras), and metric exports.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Layer 2 (Preprocessing & Validation Layer):</b> Houses MRIScanValidator (filtering invalid inputs), data_loader.py (building tf.data streams), and augmentation.py (applying conservative transforms).", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Layer 3 (Model & Training Layer):</b> Contains the Custom CNN Functional definition, callback factories, and trainer.py (managing loss computation and class weights).", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Layer 4 (Explainable AI & Evaluation Layer):</b> Encapsulates GradCAM (computing gradient heatmaps) and visualizer.py (plotting ROC curves and confusion matrices).", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Layer 5 (Deployment & Inference Layer):</b> Houses TumorClassifier, which orchestrates validation, inference, thresholding, and packaging into PredictionResult dataclasses.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Layer 6 (User Interface Layer):</b> Provides user access through the interactive Streamlit web application (app/app.py), main.py CLI, and Jupyter notebooks.", styles["ReportBullet"]))
    story.append(PageBreak())


def add_chapter_16(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 16 — TECHNOLOGY STACK & JUSTIFICATIONS", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("16.1 Technology Stack Justification", styles["SectionHeading"]))
    story.append(Paragraph(
        "Every library incorporated into <code>requirements.txt</code> was selected based on strict technical criteria "
        "including stability, reproducibility, medical imaging compatibility, and runtime efficiency.",
        styles["ReportBody"]
    ))

    stack_headers = ["Library", "Version", "Technical Selection Justification"]
    stack_rows = [
        ["Python", "3.11.0", "Provides substantial runtime speedups (up to 25% faster opcode execution than 3.10), enhanced exception tracebacks, and robust type annotation support."],
        ["TensorFlow & Keras", "2.15.0", "Industry-standard deep learning engine. Offers robust Functional API for intermediate layer inspection (critical for Grad-CAM) and native tf.data asynchronous pipelines."],
        ["NumPy & SciPy", "1.26.4 / 1.13.0", "Fundamental vector array computing. SciPy provides scientific statistical functions used in Cohen's Kappa and curve fitting."],
        ["Pandas", "2.2.2", "High-performance tabular data structures used for parsing dataset paths, generating classification report tables, and exporting metric CSVs."],
        ["OpenCV (opencv-python)", "4.9.0.80", "Optimized C++ computer vision bindings for image decoding, bilinear interpolation, color space conversion, and JET colormap overlay rendering."],
        ["Pillow (PIL)", "10.3.0", "Python Imaging Library providing seamless integration with Streamlit file upload byte streams and metadata inspection."],
        ["Scikit-Learn", "1.4.2", "Standard machine learning toolkit providing robust metrics (classification_report, confusion_matrix, roc_auc_score) and balanced class weight calculation."],
        ["Matplotlib & Seaborn", "3.8.4 / 0.13.2", "Statistical visualization engines used to generate publication-grade training curves, ROC curves, and confusion matrix heatmaps."],
        ["Streamlit", "1.35.0", "Modern reactive Python web framework enabling rapid deployment of interactive machine learning dashboards without requiring complex frontend JavaScript."],
        ["ReportLab", "5.0.0", "Enterprise-grade PDF document generation engine enabling programmatically styled, vector-based technical reports with custom page canvasing."]
    ]
    story.append(make_table(stack_headers, stack_rows, [110, 65, 357], styles))
    story.append(PageBreak())


def add_chapter_17(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 17 — ENVIRONMENT SETUP & REPRODUCIBILITY", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("17.1 Environment Setup Guide", styles["SectionHeading"]))
    story.append(Paragraph(
        "To ensure 100% reproducibility across different developer workstations, the project uses isolated virtual environments. "
        "The complete setup procedure requires four terminal commands:",
        styles["ReportBody"]
    ))

    from scripts.generate_project_pdf import make_code_box
    setup_code = """
    # 1. Clone repository and navigate to root directory
    cd From_Genes_to_Decisions

    # 2. Create isolated Python 3.11 virtual environment
    python -m venv .venv

    # 3. Activate virtual environment (Windows PowerShell)
    .venv\\Scripts\\activate
    # (On Linux / macOS: source .venv/bin/activate)

    # 4. Install all locked dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    """
    story.append(make_code_box(setup_code.strip(), styles))
    story.append(Spacer(1, 8))

    story.append(Paragraph("17.2 Dataset Placement Protocol", styles["SectionHeading"]))
    story.append(Paragraph(
        "The project expects the Kaggle Brain Tumor MRI Dataset extracted into <code>data/raw/</code>. "
        "The directory structure must adhere strictly to the following layout:",
        styles["ReportBody"]
    ))

    data_layout = """
    data/raw/
    ├── Training/
    │   ├── glioma_tumor/       (826 images)
    │   ├── meningioma_tumor/   (822 images)
    │   ├── no_tumor/           (395 images)
    │   └── pituitary_tumor/    (827 images)
    └── Testing/
        ├── glioma_tumor/       (100 images)
        ├── meningioma_tumor/   (115 images)
        ├── no_tumor/           (105 images)
        └── pituitary_tumor/    (74 images)
    """
    story.append(make_code_box(data_layout.strip(), styles))
    story.append(PageBreak())


def add_chapter_18(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 18 — STEP-BY-STEP PROJECT EXECUTION GUIDE", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("18.1 Execution Lifecycle", styles["SectionHeading"]))
    story.append(Paragraph(
        "A student or researcher can execute the complete project lifecycle using either the modular command-line interface "
        "(<code>main.py</code>), the interactive Jupyter notebooks, or the Streamlit web dashboard. "
        "The recommended execution sequence is detailed below:",
        styles["ReportBody"]
    ))

    exec_headers = ["Step #", "Action / Command", "Expected Console Output", "Artifacts Generated"]
    exec_rows = [
        ["Step 1", "python main.py --show-config", "Prints full configuration table and resolved directories.", "None (Console verification)"],
        ["Step 2", "jupyter lab<br/>(Run 01_Data_Understanding.ipynb)", "Displays dataset statistics, pixel histograms, and class counts.", "data/metadata/dataset_summary.csv"],
        ["Step 3", "Run 02_Preprocessing.ipynb", "Displays original vs. augmented MRI sample pairs.", "Augmentation validation plots"],
        ["Step 4", "python main.py --stage train", "Executes model compilation, class weighting, and 60-epoch training loop with early stopping.", "outputs/saved_models/best_model.keras<br/>outputs/metrics/training_history.csv"],
        ["Step 5", "python main.py --stage evaluate", "Evaluates best model on 394 test images; prints accuracy, precision, recall, and F1.", "outputs/metrics/scalar_metrics.json<br/>outputs/plots/roc_curves.png<br/>outputs/confusion_matrix/*.png"],
        ["Step 6", "python main.py --stage gradcam", "Generates triple-panel Grad-CAM explanations for test samples across all classes.", "outputs/gradcam/*.png (12 files)"],
        ["Step 7", "python debug_model.py", "Executes automated 10-step diagnostic audit on random test samples.", "debug/debug_report.txt<br/>debug/misclassified/*.jpg"],
        ["Step 8", "streamlit run app/app.py", "Launches local web server at http://localhost:8501.", "Interactive Web Interface"],
        ["Step 9", "Upload MRI in Web Browser", "MRIScanValidator checks input; CNN predicts class; Grad-CAM renders heatmap.", "Real-time visual diagnostic report"]
    ]
    story.append(make_table(exec_headers, exec_rows, [50, 140, 160, 182], styles))
    story.append(PageBreak())

print("Chapters 13-18 module loaded with PageBreaks.")
