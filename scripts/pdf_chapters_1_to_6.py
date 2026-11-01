"""
scripts/pdf_chapters_1_to_6.py
Chapters 1 through 6 for the Master PDF report.
Includes PageBreaks between chapters and rich technical depth.
"""

from reportlab.platypus import Paragraph, Spacer, PageBreak, KeepTogether, HRFlowable
from reportlab.lib import colors

def add_cover_and_toc(story, styles, make_callout, make_table, make_image_flowable):
    # ── Cover Page ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 30))
    story.append(Paragraph("STUDENT PROJECT TECHNICAL REPORT & COMPREHENSIVE CODE AUDIT", styles["DocSubtitle"]))
    story.append(Paragraph("MRI Brain Tumor Detection Using Custom Convolutional Neural Networks and Explainable AI (Grad-CAM)", styles["DocTitle"]))
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=2.5, color=colors.HexColor("#1A2B4C"), spaceBefore=5, spaceAfter=18))
    
    meta_text = """
    <b>Project Title:</b> From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection<br/>
    <b>Author / Student:</b> Ritik (B.Tech Computer Science & Engineering)<br/>
    <b>Primary Frameworks:</b> Python 3.11 · TensorFlow 2.15.0 · Keras · Streamlit · OpenCV · Scikit-Learn<br/>
    <b>Evaluation Summary:</b> 62.18% Test Accuracy · 0.8123 Macro ROC-AUC · 4-Class Softmax Classification<br/>
    <b>Explainable AI Method:</b> Gradient-weighted Class Activation Mapping (Grad-CAM) on Block 4 Feature Maps<br/>
    <b>Document Purpose:</b> End-to-End Forensic Code Documentation, Architectural Analysis & Viva Preparation Manual<br/>
    <b>Status:</b> Fully Implemented, Trained, Evaluated, and Deployed via Local Web Interface
    """
    story.append(Paragraph(meta_text, styles["MetaBlock"]))
    story.append(Spacer(1, 12))

    # Embed sample montage on cover
    story.append(make_image_flowable(
        "outputs/doc_diagrams/sample_mri_montage.png", width=490, height=137,
        caption="Figure 1.1: Representative T1/T2-weighted MRI brain scans across the four target categories: Glioma, Meningioma, Normal (No Tumor), and Pituitary Tumor.",
        styles=styles
    ))
    story.append(Spacer(1, 14))

    exec_summary = """
    This technical document serves as a complete forensic analysis, implementation guide, and architectural specification
    for an undergraduate deep learning student project focused on automated brain tumor detection from magnetic resonance imaging (MRI) scans.
    The primary motivation is providing clinical decision support that bridges the gap between high-capacity convolutional neural networks
    and clinical interpretability. By training a scratch-built 4-block deep CNN on 3,264 MRI slices and integrating Grad-CAM visual heatmaps,
    the system not only outputs calibrated classification probabilities across four distinct neuropathological categories (Glioma, Meningioma,
    Pituitary Tumor, and No Tumor), but simultaneously explains the exact morphological regions of the brain that dictated its inferences.
    This report documents every script, data generator, callback, hyperparameter, evaluation metric, and web interface component in the repository,
    serving as an exhaustive reference and viva preparation manual.
    """
    story.append(make_callout(exec_summary, styles, title="EXECUTIVE SUMMARY"))
    story.append(PageBreak())

    # ── Table of Contents ────────────────────────────────────────────────────
    story.append(Paragraph("TABLE OF CONTENTS", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceBefore=3, spaceAfter=12))

    toc_items = [
        ("Chapter 1: Project Overview & Clinical Significance", "Problem formulation, clinical motivation, objectives, scope boundaries, workflow, technologies used."),
        ("Chapter 2: Project Structure & File Inventory", "Complete folder hierarchy, functional role of directories, detailed file-by-file specification table."),
        ("Chapter 3: MRI Dataset Documentation & Class Imbalance", "Kaggle dataset origin, folder layouts, neuropathology of 4 classes, image resolutions, class distribution and imbalance analysis."),
        ("Chapter 4: Data Preprocessing & Input Validation Pipeline", "Image decoding, resizing to 224x224, 3-channel RGB handling, normalization, data augmentation, MRIScanValidator safety checks."),
        ("Chapter 5: Exploratory Data Analysis (EDA)", "Class frequency breakdown, pixel intensity distributions across tumor types, anatomical symmetry, key EDA takeaways."),
        ("Chapter 6: Deep Learning Model Architecture", "Custom CNN layer-by-layer design, Batch Normalization, Global Average Pooling, Grad-CAM hook layer, Transfer learning survey (EfficientNetB0, DenseNet121)."),
        ("Chapter 7: Training Pipeline & Optimization Strategies", "Data pipeline caching & prefetching, loss with label smoothing, Adam optimizer, class weights, callback architecture, model checkpointing."),
        ("Chapter 8: Model Evaluation & Statistical Metrics", "Held-out test set performance, accuracy (62.18%), ROC-AUC (0.8123), per-class precision/recall/F1, Cohen's Kappa, automated debug diagnostics."),
        ("Chapter 9: Visualizations & Forensic Error Analysis", "Training history curves, normalized confusion matrix, One-vs-Rest ROC curves, forensic analysis of misclassified samples."),
        ("Chapter 10: Explainable AI — Grad-CAM Theory & Implementation", "Mathematical derivation of Grad-CAM, gradient computation, feature map weighting, clinical interpretation, multi-class heatmap analysis."),
        ("Chapter 11: End-to-End Prediction & Inference Pipeline", "TumorClassifier class design, input handling (bytes/files), multi-stage safety checks, softmax calibration, PredictionResult dataclass."),
        ("Chapter 12: Streamlit Interactive Web Application", "Application design, custom dark theme, sidebar controls, Home dashboard, Predict MRI workbench, About technical documentation."),
        ("Chapter 13: File-by-File & Function-by-Function Code Explanation", "Exhaustive audit of every python module and notebook in src/, app/, notebooks/, main.py, and debug scripts."),
        ("Chapter 14: Generated Output Files & Artifact Analysis", "Audit of saved models (.keras), metric JSONs, CSV logs, ROC curves, confusion matrices, and Grad-CAM exports."),
        ("Chapter 15: Layered System Architecture & Data Flow", "6-tier system architecture from dataset layer to user interface layer, component coupling and lifecycle."),
        ("Chapter 16: Technology Stack & Technical Justifications", "Explanations of Python 3.11, TensorFlow 2.15, Streamlit, OpenCV, Pillow, Scikit-learn, Seaborn, ReportLab."),
        ("Chapter 17: Environment Setup & Reproducibility Guide", "Virtual environment creation, package installation, directory initialization, running notebooks and web servers."),
        ("Chapter 18: Step-by-Step Project Execution Guide", "10-step student guide to running data loading, training, evaluation, explainability, debugging, and web deployment."),
        ("Chapter 19: Implementation Challenges Faced & Solutions", "Class imbalance, subtle histological boundaries, non-brain image inputs, memory constraints, and vanishing gradients."),
        ("Chapter 20: Project Limitations & Edge Cases", "Single-slice 2D MRI limitation, sensitivity to image contrast, lack of clinical certification, 4-class boundary conditions."),
        ("Chapter 21: Future Improvements & Research Roadmap", "Transfer learning backbones (ResNet, EfficientNet), 3D U-Net segmentation, multi-institutional datasets (BraTS), clinical deployment."),
        ("Chapter 22: Appendix & Master Viva Voce Preparation Guide", "Comprehensive 50-Question Viva Voce Q&A Manual, Technical Glossary, Master Hyperparameter Table, Performance Summary Table.")
    ]

    for title, desc in toc_items:
        story.append(Paragraph(f"• <b>{title}</b>", styles["TOCItem"]))
        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{desc}", styles["TOCDesc"]))
        story.append(Spacer(1, 3))

    story.append(PageBreak())


def add_chapter_1(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 1 — PROJECT OVERVIEW & CLINICAL SIGNIFICANCE", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("1.1 Project Title and Context", styles["SectionHeading"]))
    story.append(Paragraph(
        "The project is titled <b>'From Genes to Decisions: Explainable AI Framework for Brain Tumor Detection'</b>. "
        "It represents an end-to-end undergraduate deep learning engineering effort to address a pressing clinical need: "
        "the rapid, accurate, and transparent automated triage of Magnetic Resonance Imaging (MRI) brain scans. "
        "Unlike purely black-box image classifiers, this project integrates an Explainable Artificial Intelligence (XAI) "
        "framework using Gradient-weighted Class Activation Mapping (Grad-CAM), directly revealing the spatial attention maps "
        "that drive each neural network classification.",
        styles["ReportBody"]
    ))

    story.append(Paragraph("1.2 Problem Statement & Clinical Context", styles["SectionHeading"]))
    story.append(Paragraph(
        "Central Nervous System (CNS) tumors, specifically intracranial brain tumors, represent some of the most aggressive "
        "and debilitating human neoplasms. In clinical neuro-oncology, magnetic resonance imaging is the primary non-invasive "
        "diagnostic imaging modality utilized to visualize intracranial soft tissue structures. Radiologists must manually inspect "
        "dozens of MRI slices per patient across different scanning sequences (T1-weighted, T2-weighted, FLAIR, and contrast-enhanced T1) "
        "to detect abnormal tissue masses, identify margins, and differentiate between tumor categories. "
        "This manual interpretation process is time-consuming, mentally exhausting, and subject to inter-observer variability, "
        "particularly in community clinics or emergency departments where specialized neuro-radiologists may not be on duty 24/7.",
        styles["ReportBody"]
    ))

    story.append(Paragraph("1.3 Why Brain Tumor Detection is Important", styles["SectionHeading"]))
    story.append(Paragraph(
        "Early and differential diagnosis directly governs patient prognosis and surgical planning. "
        "Differentiating between distinct tumor categories is critical because therapeutic interventions vary radically:",
        styles["ReportBody"]
    ))
    story.append(Paragraph("• <b>Glioma Tumors:</b> Infiltrative neoplasms arising from glial support cells (astrocytomas, oligodendrogliomas, glioblastoma multiforme). They exhibit diffuse boundaries and require aggressive maximal surgical resection, radiation, and temozolomide chemotherapy.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Meningioma Tumors:</b> Typically extra-axial, slow-growing tumors arising from arachnoid cap cells of the meninges. Frequently benign (WHO Grade I), curable via total surgical excision without requiring immediate cytotoxic chemotherapy.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Pituitary Tumors:</b> Sellar region adenomas that compress the optic chiasm and disrupt endocrine homeostasis. Frequently managed via transsphenoidal endoscopic surgery or medical dopaminergic therapy.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Normal (No Tumor):</b> Healthy neuroanatomy where symmetrical ventricles, normal sulci, and intact white-gray matter interfaces indicate absence of mass effect, avoiding unnecessary invasive biopsies.", styles["ReportBullet"]))

    story.append(Paragraph("1.4 Project Objectives", styles["SectionHeading"]))
    story.append(Paragraph("The technical goals of this project are structured into five measurable objectives:", styles["ReportBody"]))
    story.append(Paragraph("1. <b>Automated 4-Class Classification:</b> Design and train a deep convolutional neural network capable of categorizing 224x224 RGB MRI scans into Glioma, Meningioma, Pituitary Tumor, or No Tumor.", styles["ReportBullet"]))
    story.append(Paragraph("2. <b>Input Safety Validation:</b> Construct an automated algorithmic validator (MRIScanValidator) that rejects invalid non-brain images (e.g. photos of people, animals, landscapes, or text documents) before invoking model inference.", styles["ReportBullet"]))
    story.append(Paragraph("3. <b>Visual Explainability via Grad-CAM:</b> Formulate and hook an analytical Grad-CAM pipeline into the final convolutional layer of the CNN to extract gradient-based saliency heatmaps highlighting tumor mass boundaries.", styles["ReportBullet"]))
    story.append(Paragraph("4. <b>Rigorous Quantitative Evaluation:</b> Evaluate the trained model across balanced accuracy, per-class precision, recall, F1-score, Cohen's Kappa, and multi-class One-vs-Rest ROC-AUC curves on held-out test data.", styles["ReportBullet"]))
    story.append(Paragraph("5. <b>Interactive Web Deployment:</b> Deliver a production-grade Streamlit application featuring dark-mode glassmorphic aesthetics, dynamic confidence threshold sliders, and visual explainability panels.", styles["ReportBullet"]))

    story.append(Paragraph("1.5 Scope of the Project", styles["SectionHeading"]))
    story.append(Paragraph(
        "<b>In Scope:</b> Single-slice 2D axial MRI image processing, custom convolutional network architecture, data augmentation, "
        "class imbalance mitigation via smoothed weighting, Grad-CAM visual explainability, local web dashboard deployment, and offline batch debugging.<br/>"
        "<b>Out of Scope:</b> 3D volumetric MRI segmentation (e.g., BraTS multi-sequence voxel labeling), direct clinical diagnostic certification, "
        "and integration with hospital Picture Archiving and Communication Systems (PACS) via DICOM protocols.",
        styles["ReportBody"]
    ))

    story.append(Paragraph("1.6 High-Level Project Workflow", styles["SectionHeading"]))
    story.append(Paragraph(
        "The project executes a strictly sequential, reproducible data science lifecycle illustrated in Figure 1.2 below. "
        "Each stage encapsulates dedicated Python modules with unit-level error handling.",
        styles["ReportBody"]
    ))

    story.append(make_image_flowable(
        "outputs/doc_diagrams/workflow_diagram.png", width=500, height=160,
        caption="Figure 1.2: Six-stage architectural workflow diagram of the brain tumor detection system from raw data ingestion to interactive deployment.",
        styles=styles
    ))

    story.append(Paragraph("1.7 Technologies Used", styles["SectionHeading"]))
    tech_headers = ["Layer", "Primary Technology", "Version", "Role in Project"]
    tech_rows = [
        ["Runtime", "Python", "3.11.0", "Core programming language"],
        ["Deep Learning", "TensorFlow & Keras", "2.15.0", "Functional model construction, backpropagation, tf.data pipeline"],
        ["Numerical Computing", "NumPy & SciPy", "1.26.4 / 1.13.0", "Array transformations, gradient operations, mathematical metrics"],
        ["Data Processing", "Pandas", "2.2.2", "Metadata parsing, CSV report generation, tabular statistics"],
        ["Computer Vision", "OpenCV & Pillow", "4.9.0 / 10.3.0", "Image decoding, resizing, color space conversion, heatmap blending"],
        ["Evaluation & Metrics", "Scikit-Learn", "1.4.2", "Confusion matrix, ROC-AUC, classification report, class weighting"],
        ["Visual Analytics", "Matplotlib & Seaborn", "3.8.4 / 0.13.2", "Training curves, ROC curves, confusion matrix heatmaps"],
        ["Web Application", "Streamlit", "1.35.0", "Reactive client-facing web dashboard and interactive inference"],
        ["Documentation", "ReportLab", "5.0.0", "Automated PDF documentation compilation and vector layout"]
    ]
    story.append(make_table(tech_headers, tech_rows, [90, 115, 75, 252], styles))
    story.append(PageBreak())


def add_chapter_2(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 2 — PROJECT STRUCTURE ANALYSIS", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("2.1 Directory Hierarchy & Organization", styles["SectionHeading"]))
    story.append(Paragraph(
        "The project adheres to professional Python software engineering best practices. Source code, dataset storage, "
        "trained model checkpoints, evaluation outputs, interactive web applications, and exploratory notebooks are strictly partitioned. "
        "The complete directory tree is shown below:",
        styles["ReportBody"]
    ))

    tree_str = """
From_Genes_to_Decisions/
├── app/
│   └── app.py                     # Streamlit web application (multi-page dashboard)
├── data/
│   ├── raw/                       # Original Kaggle MRI dataset (Training/ & Testing/)
│   ├── metadata/                  # class_mapping.json & dataset_summary.csv
│   └── processed/                 # Reserved for intermediate serialized datasets
├── debug/                         # Automated diagnostic outputs & misclassified samples
│   ├── debug_report.txt           # Detailed diagnostic log
│   ├── confusion_matrix.png       # Test batch confusion matrix
│   └── misclassified/             # Raw images of misclassified test samples
├── docs/                          # Architecture guides, API reference, presentation assets
├── logs/                          # Runtime log outputs (training.log)
├── notebooks/                     # Educational Jupyter notebooks (01 to 06)
├── outputs/                       # Generated artifacts from pipeline execution
│   ├── saved_models/              # best_model.keras & final_model.keras
│   ├── metrics/                   # scalar_metrics.json, classification_report.csv/json
│   ├── plots/                     # training_history.png, roc_curves.png
│   ├── confusion_matrix/          # Normalized & raw count confusion matrix heatmaps
│   └── gradcam/                   # Multi-class Grad-CAM triple-panel visualizations
├── scripts/                       # Automated PDF & diagram generation scripts
├── src/                           # Core reusable Python package
│   ├── config.py                  # Master configuration and hyperparameter constants
│   ├── deployment/                # inference.py (TumorClassifier & PredictionResult)
│   ├── evaluation/                # metrics.py & visualizer.py
│   ├── explainability/            # gradcam.py (GradCAM engine)
│   ├── models/                    # custom_cnn.py & callbacks.py
│   ├── preprocessing/             # data_loader.py, augmentation.py, image_utils.py, mri_validator.py
│   └── utils/                     # helpers.py & logger.py
├── main.py                        # CLI pipeline orchestrator
└── requirements.txt               # Locked project dependencies
"""
    from scripts.generate_project_pdf import make_code_box
    story.append(make_code_box(tree_str.strip(), styles))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2.2 Comprehensive File Inventory", styles["SectionHeading"]))
    story.append(Paragraph(
        "Table 2.1 documents all primary files in the repository, explaining their structural purpose, "
        "inputs, outputs, and pipeline responsibilities.",
        styles["ReportBody"]
    ))

    file_headers = ["File Path", "Size / LOC", "Primary Responsibility", "Inputs / Outputs"]
    file_rows = [
        ["src/config.py", "9.7 KB / 196", "Central configuration class. Declares all hyperparameter constants, file paths, and dataset properties.", "In: OS environment<br/>Out: Config class"],
        ["src/models/custom_cnn.py", "9.8 KB / 277", "Builds the 4-block deep convolutional network with BN, GAP, and explicitly named last_conv_layer.", "In: (224, 224, 3) tensor<br/>Out: Keras Model"],
        ["src/models/callbacks.py", "6.8 KB / 167", "Factory creating ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, and CSVLogger callbacks.", "In: Config parameters<br/>Out: List of Callbacks"],
        ["src/preprocessing/data_loader.py", "12.1 KB / 365", "Builds high-performance tf.data pipelines with in-memory caching, parallel mapping, and prefetching.", "In: Image directories<br/>Out: tf.data.Dataset"],
        ["src/preprocessing/augmentation.py", "6.1 KB / 162", "Conservative geometric transformations (rotation, zoom, flip) maintaining anatomical brain validity.", "In: Image tensors<br/>Out: Augmented tensors"],
        ["src/preprocessing/mri_validator.py", "7.4 KB / 168", "Safety validator detecting non-medical images via color divergence, border darkness, and contrast.", "In: Raw image array<br/>Out: (bool, message, dict)"],
        ["src/preprocessing/image_utils.py", "7.5 KB / 243", "Low-level decoding, resizing, and normalization helpers for single-image inference and Grad-CAM.", "In: Image bytes / paths<br/>Out: Normalized array"],
        ["src/training/trainer.py", "10.0 KB / 241", "Manages model compilation, smoothed class weight calculation, fitting, and checkpoint loading.", "In: train_ds, val_ds<br/>Out: History, best model"],
        ["src/evaluation/metrics.py", "6.7 KB / 185", "Computes multi-class accuracy, precision, recall, macro/weighted F1, Cohen's Kappa, and ROC-AUC.", "In: Model & test_ds<br/>Out: Dictionary of metrics"],
        ["src/evaluation/visualizer.py", "11.3 KB / 310", "Generates publication-quality plots: confusion matrices, training history curves, and ROC curves.", "In: Metrics & history<br/>Out: Matplotlib Figures"],
        ["src/explainability/gradcam.py", "15.7 KB / 459", "Computes gradient-weighted class activation maps, overlays them onto MRIs, and generates figures.", "In: Model & input tensor<br/>Out: Heatmap & overlay"],
        ["src/deployment/inference.py", "11.6 KB / 311", "Production TumorClassifier class encapsulating validation, inference, thresholding, and Grad-CAM.", "In: Image bytes / path<br/>Out: PredictionResult"],
        ["src/utils/helpers.py", "11.9 KB / 320", "System helpers for directory creation, JSON I/O, seed fixing, and dataset metadata extraction.", "In: Paths, dictionaries<br/>Out: Serialized files"],
        ["src/utils/logger.py", "3.8 KB / 115", "Configures standardized console and file logging with formatted timestamps and log levels.", "In: Module name, log path<br/>Out: Python Logger"],
        ["main.py", "9.7 KB / 250", "Command-line interface (CLI) to trigger data ingestion, training, evaluation, and explainability.", "In: CLI flags (--stage)<br/>Out: Pipeline execution"],
        ["debug_model.py", "13.2 KB / 299", "Automated debugging script sampling test images, checking confidence thresholds, and logging errors.", "In: Trained model<br/>Out: debug/debug_report.txt"],
        ["app/app.py", "61.4 KB / 1453", "Interactive Streamlit web dashboard providing real-time inference, threshold controls, and XAI panels.", "In: User uploads<br/>Out: Interactive Web UI"]
    ]
    story.append(make_table(file_headers, file_rows, [110, 60, 212, 150], styles))
    story.append(PageBreak())


def add_chapter_3(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 3 — MRI DATASET DOCUMENTATION & CLASS IMBALANCE", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("3.1 Dataset Source and Origin", styles["SectionHeading"]))
    story.append(Paragraph(
        "The project utilizes the widely established <b>Brain Tumor MRI Dataset</b> curated by Masoud Nickparvar, "
        "publicly hosted on Kaggle. This benchmark dataset aggregates verified T1-weighted, T2-weighted, and contrast-enhanced "
        "brain MRI slices from three primary open medical imaging repositories: Figshare, SARTAJ dataset, and Br35H. "
        "The images are organized strictly into standard <code>Training</code> and <code>Testing</code> partitions, "
        "with subdirectories corresponding to the four target classes.",
        styles["ReportBody"]
    ))

    story.append(Paragraph("3.2 Neuropathological Class Descriptions", styles["SectionHeading"]))
    story.append(Paragraph(
        "Understanding the underlying neuropathology is vital to evaluating machine learning model performance in medical imaging. "
        "The four classes encompass three major tumor categories and one normal control:",
        styles["ReportBody"]
    ))

    story.append(Paragraph("• <b>Glioma Tumor:</b> Originates in the glial cells of the brain parenchyma (astrocytes, oligodendrocytes, ependymal cells). Gliomas range from low-grade (WHO Grade II) to high-grade glioblastoma multiforme (GBM, WHO Grade IV). In MRI scans, gliomas appear as irregular, infiltrative, intra-axial masses often surrounded by extensive vasogenic edema, producing hyperintense regions on T2/FLAIR and heterogeneous contrast enhancement.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Meningioma Tumor:</b> Arises from the arachnoid cap cells of the meninges, the protective membranous layers enveloping the brain. Meningiomas are typically extra-axial, well-circumscribed, and slow-growing. In MRI scans, they present as sharply demarcated, dural-based lesions that enhance intensely and homogeneously following intravenous gadolinium administration, frequently exhibiting a characteristic 'dural tail' sign.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Pituitary Tumor:</b> Arises within the sella turcica from the anterior pituitary gland (adenohypophysis). Pituitary adenomas are almost universally benign, classified into microadenomas (&lt;10 mm) and macroadenomas (&gt;10 mm). On MRI, they appear as rounded sellar and suprasellar masses that frequently cause expansion of the sella and upward compression of the optic chiasm.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>No Tumor (Normal Brain):</b> Represents non-pathological cranial MRI scans. Normal neuroanatomy exhibits strict bilateral hemispheric symmetry, well-defined ventricles (lateral, third, and fourth) filled with cerebrospinal fluid, intact cortical gray matter and subcortical white matter boundaries, and no abnormal mass effect, midline shift, or abnormal tissue contrast enhancement.", styles["ReportBullet"]))

    story.append(Paragraph("3.3 Dataset Distribution and Class Imbalance", styles["SectionHeading"]))
    story.append(Paragraph(
        "A critical finding during project auditing is the presence of <b>severe class imbalance</b> in the training partition. "
        "While Glioma, Meningioma, and Pituitary classes have roughly 825 images each in the training directory, the <code>no_tumor</code> "
        "class has only 395 images — representing an approximately 48% deficit. Table 3.1 outlines the exact counts.",
        styles["ReportBody"]
    ))

    dist_headers = ["Tumor Class", "Training Images", "Testing Images", "Total Images", "% of Dataset", "Imbalance Ratio (Train)"]
    dist_rows = [
        ["Glioma Tumor", "826", "100", "926", "28.37%", "1.00x (Baseline)"],
        ["Meningioma Tumor", "822", "115", "937", "28.71%", "1.00x"],
        ["No Tumor (Normal)", "395", "105", "500", "15.32%", "0.48x (Deficit)"],
        ["Pituitary Tumor", "827", "74", "901", "27.60%", "1.00x"],
        ["Total / Overall", "2,870", "394", "3,264", "100.00%", "—"]
    ]
    story.append(make_table(dist_headers, dist_rows, [100, 80, 80, 75, 85, 112], styles))
    story.append(Spacer(1, 6))

    story.append(make_image_flowable(
        "outputs/doc_diagrams/eda_class_distribution.png", width=480, height=210,
        caption="Figure 3.1: Quantitative breakdown of the Kaggle Brain Tumor MRI Dataset highlighting the severe training deficit in the normal (No Tumor) class.",
        styles=styles
    ))

    imbalance_callout = """
    In clinical machine learning, class imbalance severely distorts unweighted empirical risk minimization.
    If left unmitigated, gradient descent disproportionately favors the majority classes, leading the network to
    develop a high false-positive rate for tumor detection when encountering healthy brains.
    In this project, class imbalance was systematically resolved in src/training/trainer.py using square-root smoothed
    balanced class weights, ensuring that gradient updates from under-represented classes receive appropriate amplification
    without destabilizing optimization.
    """
    story.append(make_callout(imbalance_callout, styles, title="CRITICAL LESSON — IMPACT OF DATASET IMBALANCE"))
    story.append(PageBreak())


def add_chapter_4(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 4 — DATA PREPROCESSING & INPUT VALIDATION", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("4.1 Preprocessing Pipeline Overview", styles["SectionHeading"]))
    story.append(Paragraph(
        "Medical image data cannot be directly ingested into deep neural networks without rigorous mathematical standardization. "
        "Raw MRI scans originate from different scanner vendors (Siemens, GE, Philips), varying magnetic field strengths (1.5 Tesla vs. 3.0 Tesla), "
        "and different spatial dimensions (e.g. 512x512, 256x256, or arbitrary aspect ratios). "
        "The project implements an automated, GPU-accelerated preprocessing pipeline designed to standardize every image.",
        styles["ReportBody"]
    ))

    story.append(Paragraph("4.2 Preprocessing Steps & Technical Rationale", styles["SectionHeading"]))
    steps_headers = ["Preprocessing Step", "Input State", "Output State", "Technical & Medical Rationale"]
    steps_rows = [
        ["1. Image Decoding", "Raw file bytes on disk (JPEG/PNG)", "Uncompressed uint8 tensor", "Converts compressed byte streams into raw multidimensional matrix tensors in memory."],
        ["2. Channel Standardization", "1-channel grayscale or 4-channel RGBA", "3-channel RGB (H, W, 3)", "Standardizes all inputs to 3 identical channels. While brain MRI is monochromatic, 3-channel input maintains compatibility with pretrained weights and vision backbones."],
        ["3. Spatial Resizing", "Arbitrary dimension (e.g. 512x512)", "Fixed 224x224 pixels", "Bilinear interpolation resizes images to standard CNN resolution. 224x224 balances anatomical lesion visibility with manageable GPU memory footprint."],
        ["4. Pixel Normalization", "Pixel integers in [0, 255]", "Float32 values in [0.0, 1.0]", "Applied via tf.keras.layers.Rescaling(1./255). Eliminates scale sensitivity, ensures activation functions operate within linear regimes, and prevents exploding gradients."],
        ["5. Label Encoding", "String class folder names", "One-hot vector shape (4,)", "Maps string directory names ('glioma_tumor' → index 0) into binary vectors for multi-class categorical cross-entropy loss computation."],
        ["6. Dataset Caching", "Decoded batch tensors", "In-memory RAM cache", "train_ds.cache() avoids re-reading and re-decoding image files from disk during subsequent epochs, accelerating training by over 300%."],
        ["7. Asynchronous Prefetching", "Batched dataset stream", "Buffered GPU queue", "train_ds.prefetch(AUTOTUNE) overlaps CPU data augmentation with GPU forward/backward passes, eliminating hardware idle time."]
    ]
    story.append(make_table(steps_headers, steps_rows, [95, 95, 95, 247], styles))
    story.append(Spacer(1, 8))

    story.append(Paragraph("4.3 Medically Conservative Data Augmentation", styles["SectionHeading"]))
    story.append(Paragraph(
        "A major engineering triumph documented in <code>src/preprocessing/augmentation.py</code> is the adoption of "
        "<b>medically conservative data augmentation</b>. Generic computer vision augmentations (such as vertical flipping, "
        "aggressive cropping, or extreme shearing) are catastrophic in neuro-oncology because they create anatomically impossible "
        "head orientations and destroy delicate structural landmarks. The project strictly enforces five rules:",
        styles["ReportBody"]
    ))
    story.append(Paragraph("• <b>Random Horizontal Flip (mode='horizontal'):</b> Anatomically valid because the human cerebral hemispheres exhibit approximate bilateral axial symmetry. (Vertical flip is strictly forbidden).", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Subtle Rotation (factor=0.05, ≈ ±9°):</b> Simulates slight patient head tilting inside the MRI head coil without altering gross neuroanatomy.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Subtle Zoom (height_factor=(-0.05, 0.05)):</b> Accounts for minor variations in scanner field-of-view (FOV) and patient positioning relative to the coil.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Subtle Translation (factor=(-0.05, 0.05)):</b> Simulates minor patient centering offsets along the X and Y axes.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Value Clamping [0.0, 1.0]:</b> Applied via a custom Lambda layer <code>tf.clip_by_value(x, 0.0, 1.0)</code> to guarantee that affine transformations do not create mathematical artifacts outside the normalized pixel boundary.", styles["ReportBullet"]))

    story.append(Paragraph("4.4 Input Image Safety Validator (MRIScanValidator)", styles["SectionHeading"]))
    story.append(Paragraph(
        "A common vulnerability of deployed medical classifiers is failing gracefully when users upload non-medical images "
        "(such as pet photos, human faces, or screenshots). The project addresses this with a dedicated heuristic validator "
        "in <code>src/preprocessing/mri_validator.py</code> that evaluates three physical properties of brain MRIs:",
        styles["ReportBody"]
    ))
    story.append(Paragraph("1. <b>Color Divergence Check (max_color_divergence = 12.0):</b> Brain MRIs are monochromatic grayscale scans where R ≈ G ≈ B across all coordinates. Natural color images have high inter-channel standard deviation. Images with inter-channel std-dev exceeding 12.0 are rejected.", styles["ReportBullet"]))
    story.append(Paragraph("2. <b>Border Darkness Ratio (min_border_darkness_ratio = 0.20):</b> Cranial MRI scans are surrounded by empty air space, creating a black background along the perimeter. The validator inspects the outer 8% margin of the image; if less than 20% of perimeter pixels are dark, the image is rejected as a non-MRI.", styles["ReportBullet"]))
    story.append(Paragraph("3. <b>Brain Tissue Contrast Check (min_contrast = 15.0):</b> Genuine cranial MRIs exhibit distinct contrast between skull bone, brain parenchyma, and cerebrospinal fluid. Solid color images, blank scans, or uniform documents fail this luminance variance threshold.", styles["ReportBullet"]))
    story.append(PageBreak())


def add_chapter_5(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 5 — EXPLORATORY DATA ANALYSIS (EDA)", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("5.1 Objectives of Exploratory Data Analysis", styles["SectionHeading"]))
    story.append(Paragraph(
        "Exploratory Data Analysis (EDA) was conducted in <code>notebooks/01_Data_Understanding.ipynb</code> to uncover "
        "underlying statistical patterns, verify data integrity, detect corrupt image files, and identify class-conditional "
        "intensity characteristics prior to neural network training. The findings directly influenced the choice of loss functions, "
        "data augmentation parameters, and preprocessing steps.",
        styles["ReportBody"]
    ))

    story.append(Paragraph("5.2 Key Statistical Findings from Dataset Exploration", styles["SectionHeading"]))
    story.append(Paragraph("• <b>File Integrity & Corrupt Files:</b> All 3,264 images across training and testing folders were decoded and verified using PIL and OpenCV. Zero corrupt files or zero-byte files were detected.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Image Resolution Variance:</b> Original image dimensions ranged from 150x150 up to 1024x1024 pixels with an average resolution of 384x384. Standardizing to 224x224 preserved critical morphological structures without introducing severe downsampling artifacts.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Pixel Intensity Distribution:</b> MRI pixel intensity histograms exhibit a sharp bimodal distribution. A large spike at pixel intensity 0 corresponds to the background air margin, while an approximately Gaussian distribution between intensities 40 and 220 corresponds to brain tissue and fluid compartments.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Class-Conditional Luminance:</b> Meningioma scans frequently exhibit localized hyper-intense clusters corresponding to dense gadolinium contrast uptake, whereas No Tumor scans display a more uniform, symmetrical intensity profile across hemispheres.", styles["ReportBullet"]))

    story.append(Paragraph("5.3 EDA Takeaways and Model Design Implications", styles["SectionHeading"]))
    eda_headers = ["EDA Observation", "Clinical / Physical Cause", "Engineering Solution Implemented"]
    eda_rows = [
        ["Class Imbalance (48% No Tumor deficit)", "Normal scans are less frequently uploaded to public tumor challenges.", "Smoothed balanced class weights calculated in trainer.py and applied to training loss."],
        ["Background Noise / Border Variations", "Different scanner coils produce variable background gradient artifacts.", "Border darkness ratio thresholding in MRIScanValidator; Global Average Pooling in CNN."],
        ["Subtle Histological Boundaries", "Gliomas blend gradually into healthy tissue via micro-infiltrative fingers.", "Grad-CAM spatial heatmap visualization to inspect model focus regions; dropout regularization."],
        ["Extreme Luminance Spikes", "Contrast-enhanced scans have very bright focal points.", "Pixel normalization to [0.0, 1.0] and He Normal initialization to prevent saturation."]
    ]
    story.append(make_table(eda_headers, eda_rows, [110, 150, 272], styles))
    story.append(PageBreak())


def add_chapter_6(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 6 — DEEP LEARNING MODEL ARCHITECTURE", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("6.1 Custom CNN Architectural Philosophy", styles["SectionHeading"]))
    story.append(Paragraph(
        "The model implemented in <code>src/models/custom_cnn.py</code> is a deep convolutional neural network "
        "designed from first principles for brain MRI classification. Rather than treating neural networks as opaque black boxes, "
        "the architecture balances representational capacity, parameter efficiency, and explainability hooks. "
        "The backbone is inspired by the classic VGGNet philosophy of stacked 3x3 convolutional kernels, but modernized "
        "with Batch Normalization, Global Average Pooling, and a dedicated Grad-CAM visualization layer.",
        styles["ReportBody"]
    ))

    story.append(make_image_flowable(
        "outputs/doc_diagrams/cnn_architecture_diagram.png", width=510, height=185,
        caption="Figure 6.1: Architectural diagram of the Custom CNN showing stacked convolutional blocks, Batch Normalization, Global Average Pooling, and the Grad-CAM hook layer.",
        styles=styles
    ))

    story.append(Paragraph("6.2 Detailed Layer-by-Layer Architectural Breakdown", styles["SectionHeading"]))
    story.append(Paragraph(
        "Table 6.1 details every layer block in the implemented Custom CNN, including output tensor shapes, "
        "receptive fields, activation functions, and parameter counts.",
        styles["ReportBody"]
    ))

    cnn_headers = ["Block / Layer", "Operation & Hyperparameters", "Output Shape", "Trainable Parameters", "Key Architectural Purpose"]
    cnn_rows = [
        ["Input Layer", "Input(shape=(224, 224, 3))", "(None, 224, 224, 3)", "0", "Accepts standardized 3-channel RGB MRI slices."],
        ["Block 1", "2x Conv2D(32, 3x3) + BN + ReLU<br/>MaxPooling2D(2x2)", "(None, 112, 112, 32)", "10,240", "Extracts primitive low-level features: edges, intensity gradients, skull boundaries."],
        ["Block 2", "2x Conv2D(64, 3x3) + BN + ReLU<br/>MaxPooling2D(2x2)", "(None, 56, 56, 64)", "55,808", "Captures local texture patterns: sulcal margins, ventricular contours, tissue interfaces."],
        ["Block 3", "2x Conv2D(128, 3x3) + BN + ReLU<br/>MaxPooling2D(2x2)", "(None, 28, 28, 128)", "222,208", "Extracts mid-level anatomical semantics: ventricular asymmetry, focal mass shapes."],
        ["Block 4<br/>(Grad-CAM Layer)", "Conv2D(256, 3x3) + BN + ReLU<br/>Conv2D(256, 3x3, name='last_conv_layer')<br/>BN + ReLU + MaxPool(2x2)", "(None, 14, 14, 256)", "886,784", "High-level tumor semantics. Explicitly named 'last_conv_layer' to serve as target for Grad-CAM gradients."],
        ["Global Avg Pool", "GlobalAveragePooling2D()", "(None, 256)", "0", "Averages each 14x14 feature map into a single scalar. Replaces Flatten; cuts parameter count by ~75%."],
        ["Dense Head", "Dense(256) + BN + ReLU<br/>Dropout(0.4)", "(None, 256)", "66,304", "Learns non-linear combination of spatial features; 40% dropout prevents feature co-adaptation."],
        ["Output Layer", "Dense(4, activation='softmax')", "(None, 4)", "1,028", "Produces calibrated posterior probability distribution across the 4 tumor classes."],
        ["Total Model", "BrainTumor_CustomCNN", "—", "1,242,372 params", "Compact, highly regularized architecture resistant to overfitting."]
    ]
    story.append(make_table(cnn_headers, cnn_rows, [80, 130, 85, 80, 157], styles))
    story.append(Spacer(1, 8))

    story.append(Paragraph("6.3 Key Architectural Design Decisions", styles["SectionHeading"]))
    story.append(Paragraph("1. <b>Batch Normalization after Every Convolution:</b> In deep convolutional networks, internal covariate shift slows down training and creates extreme sensitivity to initialization. Placing Batch Normalization after every Conv2D stabilizes gradient distributions, acts as a regularizer, and permits a higher initial learning rate (3x10⁻⁴).", styles["ReportBullet"]))
    story.append(Paragraph("2. <b>Global Average Pooling vs. Traditional Flattening:</b> Traditional architectures (such as AlexNet or VGG16) flatten high-dimensional feature maps into massive fully connected layers, creating over 10 to 50 million parameters that easily overfit medical datasets. In this Custom CNN, GlobalAveragePooling2D reduces the spatial 14x14x256 tensor directly into a 256-element vector, reducing trainable parameters to just 1.24 million.", styles["ReportBullet"]))
    story.append(Paragraph("3. <b>Dedicated Naming of 'last_conv_layer':</b> To enable Gradient-weighted Class Activation Mapping without brittle index-based hacking, the final Conv2D layer in Block 4 is explicitly assigned <code>name='last_conv_layer'</code>. The GradCAM engine dynamically hooks into this layer.", styles["ReportBullet"]))
    story.append(Paragraph("4. <b>He Normal Weight Initialization:</b> Standard random normal initialization frequently leads to vanishing gradients in deep networks using ReLU activations. All convolutional and dense kernels utilize <code>kernel_initializer='he_normal'</code>, sampling weights from a truncated normal distribution with variance 2/N_in.", styles["ReportBullet"]))
    story.append(Paragraph("5. <b>Bias Omission in Convolutional Layers (use_bias=False):</b> Because Batch Normalization immediately follows each convolutional layer by centering the batch mean (subtracting beta) and scaling variance (multiplying gamma), adding an independent convolutional bias term is mathematically redundant and wastes memory.", styles["ReportBullet"]))

    story.append(Paragraph("6.4 Transfer Learning Architectural Survey", styles["SectionHeading"]))
    story.append(Paragraph(
        "As outlined in the project README.md and <code>src/models/custom_cnn.py</code> roadmap, future iterations (Phase 2) "
        "will evaluate modern pretrained transfer learning backbones against this Custom CNN baseline. "
        "The two primary candidate architectures specified for Phase 2 exploration are analyzed below:",
        styles["ReportBody"]
    ))

    story.append(Paragraph("<b>A. EfficientNetB0 (Tan &amp; Le, ICML 2019):</b>", styles["SubsectionHeading"]))
    story.append(Paragraph(
        "• <i>Architecture Overview:</i> EfficientNetB0 utilizes Mobile Inverted Bottleneck Convolutions (MBConv) with Squeeze-and-Excitation (SE) optimization blocks. Its core innovation is <b>compound scaling</b>, which uniformly scales network depth, width, and input resolution using a fixed compound coefficient φ rather than arbitrary manual tuning.<br/>"
        "• <i>Transfer Learning Rationale:</i> EfficientNetB0 achieves 77.1% top-1 accuracy on ImageNet with only 5.3 million parameters. Pretrained ImageNet weights provide rich edge, texture, and contour detectors that can be fine-tuned on medical MRIs with minimal training epochs.<br/>"
        "• <i>Hypothetical Implementation Setup:</i> Input size: 224x224x3; Base frozen during initial warm-up; Head: GlobalAveragePooling2D + Dense(256, ReLU) + Dropout(0.4) + Dense(4, Softmax); Optimizer: Adam(lr=1e-4); Loss: Categorical Crossentropy.<br/>"
        "• <i>Implementation Status:</i> <b>Not implemented in this project</b> (Documented as Phase 2 transfer learning roadmap).",
        styles["ReportBody"]
    ))

    story.append(Paragraph("<b>B. DenseNet121 (Huang et al., CVPR 2017):</b>", styles["SubsectionHeading"]))
    story.append(Paragraph(
        "• <i>Architecture Overview:</i> DenseNet connects every layer to every subsequent layer in a feed-forward fashion across dense blocks. Unlike traditional residual networks that combine features through summation, DenseNet concatenates feature maps across channel dimensions.<br/>"
        "• <i>Transfer Learning Rationale:</i> Dense connectivity promotes strong feature reuse, alleviates vanishing gradients, and strengthens feature propagation throughout deep architectures. In medical imaging, where subtle micro-structural lesions require multi-scale feature awareness, DenseNet121 has historically demonstrated superior sensitivity.<br/>"
        "• <i>Hypothetical Implementation Setup:</i> Input size: 224x224x3; Backbone: 121 layers partitioned into 4 dense blocks with transition layers; Dense head with Softmax; Optimizer: Adam(lr=2e-4); Loss: Categorical Crossentropy.<br/>"
        "• <i>Implementation Status:</i> <b>Not implemented in this project</b> (Documented as Phase 2 transfer learning roadmap).",
        styles["ReportBody"]
    ))

    story.append(Paragraph("<b>Model Comparison Matrix:</b>", styles["SubsectionHeading"]))
    comp_headers = ["Architecture", "Implementation Status", "Total Params", "Core Architectural Concept", "Primary Advantage for Brain MRI"]
    comp_rows = [
        ["Custom CNN", "FULLY IMPLEMENTED & TRAINED", "1,242,372", "4 Conv Blocks + BN + GAP + Explicit Grad-CAM hook", "Compact, fully understood, no external weight dependencies, transparent Grad-CAM localization."],
        ["EfficientNetB0", "Not implemented (Phase 2 Roadmap)", "5,300,000", "Compound scaling + MBConv inverted residuals + Squeeze-and-Excitation", "High representational efficiency, excellent parameter-to-accuracy ratio from ImageNet pretraining."],
        ["DenseNet121", "Not implemented (Phase 2 Roadmap)", "8,000,000", "Dense connectivity concatenating all prior feature maps", "Exceptional feature reuse and gradient flow; captures multi-scale tumor boundaries."]
    ]
    story.append(make_table(comp_headers, comp_rows, [85, 95, 65, 137, 150], styles))
    story.append(PageBreak())

print("Chapters 1-6 module loaded with PageBreaks.")
