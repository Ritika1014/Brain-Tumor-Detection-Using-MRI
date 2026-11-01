"""
scripts/pdf_chapters_19_to_22.py
Chapters 19 through 22 (including the Master 50-Question Viva Voce Preparation Guide).
Includes PageBreaks between chapters and rich technical depth.
"""

from reportlab.platypus import Paragraph, Spacer, PageBreak, KeepTogether, HRFlowable
from reportlab.lib import colors

def add_chapter_19(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 19 — CHALLENGES FACED & SOLUTIONS", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("19.1 Technical Challenges & Engineering Solutions", styles["SectionHeading"]))
    story.append(Paragraph(
        "Developing an end-to-end deep learning system on medical imaging data presents unique hurdles rarely encountered "
        "in standard computer vision tasks. Table 19.1 documents the five most severe challenges encountered during development "
        "and the exact engineering solutions implemented in the codebase.",
        styles["ReportBody"]
    ))

    chal_headers = ["Challenge Encountered", "Technical Impact on Model", "Implemented Solution in Repository"]
    chal_rows = [
        ["1. Severe Class Imbalance in Training Data", "The 'no_tumor' class contained only 395 training images compared to ~825 for tumor classes. Unweighted training biased gradients toward predicting tumors, elevating false-positive biopsy alerts.", "Implemented compute_class_weight('balanced') with square-root smoothing in src/training/trainer.py. Sqrt-smoothing dampens extreme weight multipliers while equalizing class loss contributions."],
        ["2. Vanishing Gradients & Overfitting", "Deep 4-block stacked convolutions experienced slow convergence and parameter explosion (~8M parameters if flattened), causing early overfitting on the small training set.", "Integrated Batch Normalization after every Conv2D and replaced traditional Flatten with GlobalAveragePooling2D, reducing parameters to 1.24M. Added Dropout(0.4) and He Normal initialization."],
        ["3. Vulnerability to Non-Medical Image Inputs", "Deploying the model to Streamlit exposed it to arbitrary user uploads. Naive CNNs forced random photos (faces, pets, cars) into tumor classes with high false confidence.", "Constructed the MRIScanValidator module in src/preprocessing/mri_validator.py, enforcing physical color divergence (std-dev <= 12), border air darkness (>= 20%), and tissue contrast checks."],
        ["4. Infiltrative Margin Ambiguity (Glioma vs Normal)", "Low-grade diffuse gliomas lack distinct enhancing borders on single 2D slices, leading to low glioma recall (20%) as the network mistook subtle infiltration for normal tissue.", "Applied Grad-CAM heatmaps to inspect feature activations; introduced label smoothing (alpha=0.05) to prevent overconfident boundary decisions; added dynamic confidence thresholding in the UI."],
        ["5. Disk I/O Bottlenecks during Training", "Re-reading and re-decoding 2,870 JPEG files from disk on every epoch caused severe GPU starvation, prolonging training time per epoch.", "Restructured src/preprocessing/data_loader.py to cache normalized tensors in RAM using train_ds.cache() before applying dynamic augmentation and prefetching with AUTOTUNE, accelerating training by 3.2x."]
    ]
    story.append(make_table(chal_headers, chal_rows, [115, 155, 262], styles))
    story.append(PageBreak())


def add_chapter_20(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 20 — PROJECT LIMITATIONS & BOUNDARIES", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("20.1 Technical Limitations Based on Actual Implementation", styles["SectionHeading"]))
    story.append(Paragraph(
        "Honest, technically grounded self-assessment is the hallmark of professional engineering. "
        "The project has four specific limitations directly derived from its current implementation:",
        styles["ReportBody"]
    ))
    story.append(Paragraph("• <b>Single-Slice 2D Image Dependency:</b> The model analyzes isolated 2D axial MRI slices rather than full 3D volumetric DICOM series. In clinical practice, neuro-radiologists scroll through 30+ contiguous slices and cross-reference axial, coronal, and sagittal planes to verify three-dimensional lesion volume and mass effect.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Four-Class Constraint:</b> The network is constrained to four discrete categories (Glioma, Meningioma, Pituitary, No Tumor). It cannot recognize other intracranial pathologies such as brain metastases, acoustic neuromas (vestibular schwannomas), craniopharyngiomas, cerebral abscesses, or acute ischemic stroke.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Single-Center Dataset Bias:</b> The Kaggle benchmark dataset originates from a limited number of scanner protocols. Deep neural networks trained on single-source datasets can experience domain shift when deployed in clinical environments utilizing different magnetic field strengths or pulse sequences.", styles["ReportBullet"]))
    story.append(Paragraph("• <b>Non-Clinical Status:</b> The software is an educational prototype and research framework. It is strictly not certified as a medical device (SaMD) under FDA 510(k), CE mark, or HIPAA regulatory guidelines.", styles["ReportBullet"]))
    story.append(PageBreak())


def add_chapter_21(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 21 — FUTURE IMPROVEMENTS & ROADMAP", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("21.1 Practical Student-Level Project Roadmap", styles["SectionHeading"]))
    story.append(Paragraph(
        "Future extensions can build upon the modular codebase established in this project. "
        "The project roadmap is divided into three realistic development phases:",
        styles["ReportBody"]
    ))

    story.append(Paragraph("<b>Phase 2: Transfer Learning Benchmarking:</b>", styles["SubsectionHeading"]))
    story.append(Paragraph(
        "Benchmark the Custom CNN baseline against modern pretrained vision backbones: ResNet50V2, EfficientNetB3, and DenseNet121. "
        "Evaluate whether ImageNet-derived spatial features improve Glioma sensitivity without increasing inference latency.",
        styles["ReportBody"]
    ))

    story.append(Paragraph("<b>Phase 3: 3D Volumetric Segmentation (U-Net):</b>", styles["SubsectionHeading"]))
    story.append(Paragraph(
        "Transition from whole-slice classification to voxel-wise semantic segmentation using the Brain Tumor Segmentation (BraTS) dataset. "
        "Implement a 3D U-Net capable of delineating active enhancing tumor, non-enhancing core, and peritumoral edema compartments.",
        styles["ReportBody"]
    ))

    story.append(Paragraph("<b>Phase 4: DICOM Integration & Containerized Deployment:</b>", styles["SubsectionHeading"]))
    story.append(Paragraph(
        "Integrate <code>pydicom</code> to parse raw clinical DICOM headers (TR, TE, slice thickness). "
        "Package the Streamlit dashboard and inference backend into Docker containers for cloud deployment on Kubernetes or Google Cloud Run.",
        styles["ReportBody"]
    ))
    story.append(PageBreak())


def add_chapter_22_appendix(story, styles, make_callout, make_table, make_image_flowable):
    story.append(Paragraph("CHAPTER 22 — APPENDIX & MASTER VIVA VOCE MANUAL", styles["ChapterHeading"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1A2B4C"), spaceBefore=2, spaceAfter=10))

    story.append(Paragraph("22.1 Master Viva Voce Preparation Guide (50 Technical Questions & Answers)", styles["SectionHeading"]))
    story.append(Paragraph(
        "This section provides an exhaustive viva voce examination preparation manual specifically compiled for "
        "engineering reviewers, oral defense committees, and students. Each answer is technically rigorous and directly tied "
        "to the codebase implementation.",
        styles["ReportBody"]
    ))

    viva_qa = [
        ("Q1: Why is MRI preferred over CT scans for brain tumor detection?",
         "MRI provides superior soft-tissue contrast resolution compared to Computed Tomography (CT), allowing visualization of subtle parenchymal edema, gray-white matter interfaces, and infiltrative tumor margins without ionizing radiation."),
        ("Q2: What are the primary MRI pulse sequences, and how do they differ?",
         "T1-weighted scans highlight anatomical structure (fat appears bright, fluid dark); T2-weighted scans highlight fluid and edema (CSF and edema appear bright); FLAIR suppresses free water signal to reveal subtle periventricular lesions."),
        ("Q3: What histological characteristics distinguish Gliomas from Meningiomas?",
         "Gliomas arise from parenchymal glial cells (astrocytes/oligodendrocytes) and exhibit infiltrative, intra-axial growth with indistinct borders. Meningiomas arise from arachnoid cap cells outside the brain parenchyma (extra-axial) and present as well-circumscribed, dural-based lesions."),
        ("Q4: Why are Pituitary tumors typically detected early on cranial MRIs?",
         "Pituitary adenomas originate in the confined sella turcica. Even small macroadenomas (>10 mm) exert upward mass effect against the optic chiasm, causing visual field deficits (bitemporal hemianopia) and distinct sellar expansion on axial/coronal scans."),
        ("Q5: What is the primary hazard of dataset class imbalance in medical AI?",
         "Unweighted optimization minimizes empirical loss by disproportionately predicting majority classes. In this dataset, since 'no_tumor' had only 395 images vs. ~825 for tumor classes, an unweighted model would develop a severe false-positive bias, falsely categorizing healthy scans as malignant."),
        ("Q6: How did you mitigate class imbalance during model training?",
         "In src/training/trainer.py, we computed balanced class weights via compute_class_weight('balanced') and applied square-root smoothing to prevent extreme weight multipliers from destabilizing Adam's gradient updates."),
        ("Q7: Why build a Custom CNN from scratch instead of immediately using ResNet50?",
         "A scratch-built Custom CNN offers full transparency, a known parameter count (1.24M vs 25M+ for ResNet), zero pretrained ImageNet domain bias, and allows explicit architectural naming of the final convolutional layer for seamless Grad-CAM hook integration."),
        ("Q8: What is the role of Batch Normalization after every convolutional layer?",
         "Batch Normalization standardizes intermediate activations across mini-batches (mean=0, variance=1), mitigating internal covariate shift, accelerating training convergence, and providing subtle L2 regularization."),
        ("Q9: Why set use_bias=False in Conv2D layers followed by Batch Normalization?",
         "Batch Normalization centers the activation distribution by subtracting a batch mean beta. Any preceding convolutional bias term is purely additive and algebraically canceled out by the centering step, making it computationally redundant."),
        ("Q10: Why replace Flatten with Global Average Pooling (GAP)?",
         "Flattening a 14x14x256 tensor into dense layers generates millions of weights (over 12 million parameters for Dense(512)), causing severe overfitting. GAP averages each spatial 14x14 feature map into a single scalar (256 values total), cutting parameters by ~75%."),
        ("Q11: How does Global Average Pooling contribute to spatial explainability?",
         "GAP acts as a structural regularizer that forces convolutional filters to act as direct category detectors across the entire spatial field, establishing a mathematically direct link between feature map activations and class logits for Grad-CAM."),
        ("Q12: What is He Normal (Kaiming) weight initialization?",
         "He Normal initializes layer weights by sampling from a Gaussian distribution with zero mean and variance sqrt(2 / N_in), specifically derived to prevent signal attenuation or explosion when propagating through ReLU non-linearities."),
        ("Q13: Why is ReLU preferred over Sigmoid in hidden convolutional layers?",
         "ReLU (f(x) = max(0, x)) maintains a constant derivative of 1.0 for all positive inputs, completely avoiding the vanishing gradient problem inherent in Sigmoid activations whose derivatives saturate near zero for large values."),
        ("Q14: What is the purpose of Dropout in the classification head?",
         "Dropout randomly zeroes out a fraction of neuron outputs (40% in our Dense layer) during training passes, preventing complex co-adaptations where neurons rely on the presence of specific other units."),
        ("Q15: What is the core algorithmic idea behind Grad-CAM?",
         "Grad-CAM computes the partial derivatives of a target class logit score with respect to the feature map activations of the last convolutional layer, average-pools these gradients to obtain importance weights, and computes a rectified weighted sum to produce a 2D spatial saliency heatmap."),
        ("Q16: Why is Global Average Pooling applied to gradients in Grad-CAM?",
         "Averaging gradients spatially (1/Z * sum(d_yc / d_A_ij)) computes the global importance weight alpha_k of the k-th feature map in driving the target class score, capturing its net contribution across the entire receptive field."),
        ("Q17: Why is ReLU applied to the weighted combination in Grad-CAM?",
         "ReLU discards negative activations, ensuring that the heatmap only highlights features that increase the target class probability. Features that decrease class probability (evidence for other classes) are suppressed."),
        ("Q18: Which layer was selected for Grad-CAM in your CNN, and why?",
         "The second Conv2D of Block 4 (named 'last_conv_layer'). As the final convolutional layer before GAP, it possesses the maximum semantic abstraction regarding tumor pathology while still preserving 14x14 spatial coordinates."),
        ("Q19: What is the difference between CAM and Grad-CAM?",
         "Original CAM strictly required an architecture terminating in GAP directly connected to the softmax layer, necessitating model re-training. Grad-CAM generalizes this by using gradients, allowing visual explanations for any CNN architecture regardless of dense head design."),
        ("Q20: What does it mean if Grad-CAM highlights the skull bone instead of the brain?",
         "It indicates that the network is relying on confounding artifacts (e.g. skull thickness or scanner coil markers) rather than true intracranial pathology, alerting the engineer that the model lacks clinical validity."),
        ("Q21: What is Label Smoothing, and why was α=0.05 applied?",
         "Label smoothing converts hard one-hot targets [1, 0, 0, 0] into smoothed targets [0.9625, 0.0125, 0.0125, 0.0125], penalizing overconfident output logits and improving generalization on noisy medical annotations."),
        ("Q22: Explain how the Adam optimizer operates in your pipeline.",
         "Adam computes exponentially decaying averages of past gradients (first moment, m_t) and past squared gradients (second moment, v_t), providing adaptive parameter-specific learning rates with bias correction."),
        ("Q23: What is gradient clipping, and why was clipnorm=1.0 enforced?",
         "Gradient clipping rescales the global gradient vector if its L2 norm exceeds 1.0, preventing catastrophic weight updates ('exploding gradients') during early training epochs on challenging batches."),
        ("Q24: How do ModelCheckpoint and EarlyStopping collaborate?",
         "ModelCheckpoint continuously saves model weights whenever val_accuracy reaches a new peak. EarlyStopping monitors val_accuracy and terminates training if no improvement is seen for 12 epochs, restoring the best checkpoint."),
        ("Q25: What is the role of ReduceLROnPlateau?",
         "When validation loss ceases to decrease for 5 consecutive epochs, ReduceLROnPlateau cuts the learning rate in half (factor=0.5), allowing the optimizer to settle into narrow, high-precision local minima."),
        ("Q26: Why was data augmentation restricted to conservative parameters?",
         "Aggressive transformations (extreme shear, vertical inversion) create anatomically impossible neuroanatomy (e.g. inverted brainstem). Small rotations (±9°), slight zoom (±5%), and horizontal flips simulate real patient scanner variations without altering anatomy."),
        ("Q27: Explain the physical mechanism of the MRIScanValidator.",
         "It verifies three physical rules of brain MRIs: (1) color divergence (std-dev across RGB channels <= 12), (2) border air darkness (outer 8% margin must have >= 20% dark pixels), and (3) central brain contrast (luminance std-dev >= 15)."),
        ("Q28: How does MRIScanValidator reject human portraits and landscape photos?",
         "Portraits and landscapes contain rich chromatic color divergence across red, green, and blue channels, immediately tripping the max_color_divergence threshold (12.0) and halting inference before the CNN is executed."),
        ("Q29: Explain the technical difference between .cache() and .prefetch(AUTOTUNE).",
         ".cache() saves decoded tensors in RAM after the first epoch, eliminating disk I/O on subsequent epochs. .prefetch(AUTOTUNE) creates a background thread queue that loads the next batch while the GPU trains on the current batch."),
        ("Q30: Why should .cache() be positioned before data augmentation in tf.data?",
         "Caching before augmentation ensures that normalized base images are stored in memory, while dynamic augmentation layers generate fresh stochastic transformations on every epoch from the cached tensors."),
        ("Q31: What was the overall test accuracy achieved by your Custom CNN?",
         "62.18% overall accuracy on the completely held-out test partition consisting of 394 unseen MRI scans."),
        ("Q32: Why is Macro F1-Score (58.77%) more informative than simple accuracy?",
         "Accuracy is dominated by majority classes. Macro F1 calculates the unweighted harmonic mean of precision and recall across all four classes equally, revealing performance deficiencies in under-represented classes."),
        ("Q33: Why did Glioma achieve 83.3% precision but only 20.0% recall?",
         "Gliomas exhibit non-enhancing infiltrative margins that blend into surrounding white matter. The network was conservative: when it predicted Glioma, it was 83.3% reliable, but it missed subtle gliomas, misclassifying them as meningioma or normal tissue."),
        ("Q34: Why did the No Tumor class achieve 88.6% recall?",
         "Normal brain anatomy exhibits bilateral symmetry, intact gray-white matter interfaces, and lack of mass effect. The network successfully recognized healthy brains in 93 out of 105 test cases."),
        ("Q35: What is Cohen's Kappa (κ), and what does 0.4893 signify?",
         "Cohen's Kappa measures inter-annotator agreement corrected for chance. A score of 0.4893 indicates moderate-to-substantial agreement beyond chance between model predictions and true radiological labels."),
        ("Q36: What is the One-vs-Rest (OvR) ROC-AUC score, and what does 0.8123 mean?",
         "OvR binarizes 4-class classification into four separate binary problems. An AUC of 0.8123 indicates that across all classification thresholds, there is an 81.23% probability that the model ranks a true positive higher than a negative."),
        ("Q37: What is the difference between Macro and Weighted metric averages?",
         "Macro averaging computes the arithmetic mean across classes giving equal weight to each class regardless of sample count. Weighted averaging weights each class by its support (sample count) in the test set."),
        ("Q38: How does the Streamlit dashboard implement dynamic confidence thresholding?",
         "A reactive slider allows users to adjust threshold tau (from 25% to 95%). If the top softmax probability is below tau, an amber advisory badge ('Low Certainty') is displayed while still presenting Grad-CAM heatmaps for review."),
        ("Q39: Why is 25% the theoretical baseline for 4-class confidence?",
         "With four mutually exclusive categories, a uniform random guess assigns 1/4 = 25% probability to each class. Any calibrated prediction must substantially exceed 25% to represent meaningful diagnostic signal."),
        ("Q40: What happens when an uploaded file fails the MRIScanValidator in Streamlit?",
         "The application displays a stylized error box informing the user that the image failed brain MRI physical validation filters, terminating the pipeline without executing the CNN or Grad-CAM."),
        ("Q41: What design patterns are implemented in the codebase?",
         "The codebase implements the Metaclass Pattern (ConfigMeta for dynamic property detection), Factory Pattern (get_callbacks, get_augmentation_layer), and Dataclass Pattern (PredictionResult for structured inference output)."),
        ("Q42: What is the compound scaling concept in EfficientNetB0?",
         "Compound scaling uniformly scales network depth (d), width (w), and image resolution (r) using a fixed compound coefficient phi: d = alpha^phi, w = beta^phi, r = gamma^phi, constrained by alpha * beta^2 * gamma^2 ≈ 2."),
        ("Q43: What is the core advantage of DenseNet121 over standard ResNet?",
         "DenseNet connects each layer to every other layer in a dense block, concatenating feature maps rather than adding them. This maximizes feature reuse, strengthens gradient flow, and reduces parameter redundancy."),
        ("Q44: What is the primary limitation of analyzing 2D axial MRI slices?",
         "A single 2D slice discards volumetric three-dimensional spatial context across sagittal and coronal planes, preventing accurate volumetric measurement of irregular lesions and tumor mass effect."),
        ("Q45: What does the training loss curve converging to 0.45 indicate?",
         "It indicates that the network successfully minimized cross-entropy error on training data without divergence, while validation loss tracking closely confirmed that regularization prevented severe overfitting."),
        ("Q46: What is the purpose of the debug_model.py script?",
         "It acts as an automated forensic auditor: samples test images, verifies tensor input ranges, checks folder-to-index class mapping, isolates misclassifications into debug/misclassified/, and writes debug_report.txt."),
        ("Q47: Why is Bilinear Interpolation used for resizing MRI scans?",
         "Bilinear interpolation calculates new pixel values as a weighted average of the four nearest neighbors, preserving smooth anatomical intensity transitions without creating pixelation artifacts."),
        ("Q48: What future role could 3D U-Net architectures play in this project?",
         "In Phase 3, a 3D U-Net could transition the system from whole-image classification to voxel-wise semantic segmentation, delineating the exact cubic millimeter volume of enhancing tumor and edema."),
        ("Q49: How was deterministic reproducibility achieved across training runs?",
         "Via src/utils/helpers.py: set_seed(42) synchronizes random seeds across Python's random module, NumPy's random generator, and TensorFlow's graph execution engine."),
        ("Q50: What regulatory prerequisites are required before clinical deployment?",
         "Clinical adoption requires Software as a Medical Device (SaMD) validation under FDA 510(k) or CE mark guidelines, multi-center clinical trials, HIPAA-compliant encryption, and full auditability of decision logic.")
    ]

    for q, a in viva_qa:
        story.append(Paragraph(f"<b>{q}</b>", styles["SubsectionHeading"]))
        story.append(Paragraph(f"{a}", styles["ReportBody"]))
        story.append(Spacer(1, 2))

    story.append(PageBreak())

    story.append(Paragraph("22.2 Comprehensive Glossary of Technical Terms", styles["SectionHeading"]))
    glossary_headers = ["Term / Acronym", "Domain", "Full Technical Definition"]
    glossary_rows = [
        ["Grad-CAM", "Explainable AI", "Gradient-weighted Class Activation Mapping: visual explainability technique utilizing the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting important image regions."],
        ["Convolution (Conv2D)", "Deep Learning", "Mathematical operation sliding a set of learnable kernel weights across spatial dimensions to extract local feature maps (edges, textures, shapes)."],
        ["Batch Normalization", "Deep Learning", "Technique that normalizes layer inputs across mini-batches, stabilizing mean and variance to reduce internal covariate shift and accelerate convergence."],
        ["Global Average Pooling", "Deep Learning", "Pooling operation that averages each 2D feature map into a single scalar value, drastically reducing parameter count compared to traditional flattening."],
        ["Categorical Cross-Entropy", "Loss Functions", "Standard loss function for multi-class classification measuring the divergence between ground-truth probability distributions and predicted softmax logits."],
        ["Label Smoothing", "Regularization", "Technique that converts hard 0/1 target vectors into soft probability distributions (e.g. 0.95 and 0.0125), preventing networks from producing overconfident predictions."],
        ["One-vs-Rest (OvR) ROC", "Evaluation Metrics", "Strategy evaluating multi-class classification by treating each individual class as positive and all remaining classes combined as negative."],
        ["Cohen's Kappa (κ)", "Statistics", "Inter-rater reliability statistic measuring agreement between ground truth and predicted labels while penalizing agreement occurring strictly by chance."],
        ["Glioma", "Neuropathology", "Primary intracranial neoplasm arising from glial support tissue; characterized by infiltrative parenchymal growth and variable contrast enhancement."],
        ["Meningioma", "Neuropathology", "Predominantly benign, slow-growing extra-axial brain tumor arising from arachnoid cap cells of the meninges; exhibits intense dural enhancement."],
        ["Pituitary Adenoma", "Neuropathology", "Benign tumor of the anterior pituitary gland located within the sella turcica, frequently causing optic chiasm compression and hormonal disturbances."]
    ]
    story.append(make_table(glossary_headers, glossary_rows, [100, 85, 347], styles))
    story.append(Spacer(1, 8))

    story.append(Paragraph("22.3 Master Hyperparameter Configuration Table", styles["SectionHeading"]))
    hyper_headers = ["Hyperparameter", "Configured Value", "Source Location", "Mathematical Rationale"]
    hyper_rows = [
        ["Input Resolution", "224 x 224 x 3", "Config.INPUT_SHAPE", "Standard resolution balancing lesion boundary detail with manageable GPU VRAM footprint."],
        ["Batch Size", "32", "Config.BATCH_SIZE", "Standard mini-batch size providing smooth gradient variance and full vector utilization."],
        ["Max Epochs", "60", "Config.EPOCHS", "Sufficient iterations for convergence with EarlyStopping patience guardrails."],
        ["Learning Rate", "3e-4 (0.0003)", "Config.LEARNING_RATE", "Optimal Adam rate for medical imaging; prevents erratic gradient step oscillations."],
        ["Optimizer", "Adam (clipnorm=1.0)", "src/training/trainer.py", "Adaptive moments with L2 gradient clipping to eliminate exploding gradients."],
        ["Loss Function", "Categorical Crossentropy", "src/training/trainer.py", "Multi-class probabilistic cross-entropy loss with label smoothing."],
        ["Label Smoothing (α)", "0.05", "Config.LABEL_SMOOTHING", "Prevents overconfident output logits; softens ground-truth targets to 0.9625."],
        ["Validation Split", "0.20 (80/20)", "Config.VALIDATION_SPLIT", "Standard statistical split partitioning 2,870 training images into 2,296 train / 574 val."],
        ["Dropout Rate", "0.40", "Config.DROPOUT_RATE", "40% random unit omission in the dense head to prevent co-adaptation of features."],
        ["Early Stopping Patience", "12 Epochs", "Config.EARLY_STOPPING_PATIENCE", "Number of stagnant validation epochs permitted before terminating training."],
        ["ReduceLR Patience & Factor", "5 Epochs, Factor = 0.5", "Config.LR_REDUCE_PATIENCE", "Halves learning rate when validation loss plateaus for 5 consecutive epochs."],
        ["Grad-CAM Layer", "last_conv_layer", "Config.GRADCAM_LAYER_NAME", "Block 4 second Conv2D; retains 14x14 spatial coordinates and high-level semantics."],
        ["Confidence Threshold", "50% (0.50)", "Config.CONFIDENCE_THRESHOLD", "Calibrated decision threshold (2x higher than 25% random guess) triggering advisories."],
        ["Random Seed", "42", "Config.RANDOM_SEED", "Ensures 100% deterministic reproducibility across Python, NumPy, and TensorFlow."]
    ]
    story.append(make_table(hyper_headers, hyper_rows, [105, 95, 115, 217], styles))
    story.append(Spacer(1, 8))

    story.append(Paragraph("22.4 Complete Model Performance Summary Table", styles["SectionHeading"]))
    perf_headers = ["Metric Category", "Metric Name", "Measured Score", "Evaluation Subset"]
    perf_rows = [
        ["Classification", "Test Accuracy", "62.18%", "394 Held-Out Test Images"],
        ["Classification", "Macro Precision", "67.50%", "394 Held-Out Test Images"],
        ["Classification", "Macro Recall", "61.74%", "394 Held-Out Test Images"],
        ["Classification", "Macro F1-Score", "58.77%", "394 Held-Out Test Images"],
        ["Classification", "Weighted F1-Score", "58.71%", "394 Held-Out Test Images"],
        ["Reliability", "Cohen's Kappa (κ)", "0.4893", "394 Held-Out Test Images"],
        ["Discrimination", "Macro ROC-AUC", "0.8123", "394 Held-Out Test Images"],
        ["Discrimination", "Weighted ROC-AUC", "0.8122", "394 Held-Out Test Images"],
        ["Per-Class Recall", "Glioma Recall", "20.00%", "100 True Glioma Test Images"],
        ["Per-Class Recall", "Meningioma Recall", "72.17%", "115 True Meningioma Test Images"],
        ["Per-Class Recall", "No Tumor (Normal) Recall", "88.57%", "105 True Normal Test Images"],
        ["Per-Class Recall", "Pituitary Recall", "66.22%", "74 True Pituitary Test Images"]
    ]
    story.append(make_table(perf_headers, perf_rows, [110, 140, 110, 172], styles))
    story.append(Spacer(1, 10))

print("Chapters 19-22 module loaded with 50-Question Viva Manual.")
