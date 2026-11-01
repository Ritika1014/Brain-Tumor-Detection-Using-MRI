"""
scripts/generate_diagrams.py
Generates high-resolution diagrams to embed in the project documentation PDF:
1. End-to-end project workflow diagram
2. Custom CNN architecture block diagram
3. 6-Layer system architecture diagram
4. EDA Class distribution and imbalance chart
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = Path("outputs/doc_diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set common style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# ---------------------------------------------------------------------------
# 1. End-to-End Workflow Diagram
# ---------------------------------------------------------------------------
def generate_workflow_diagram():
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=200)
    ax.axis('off')
    
    stages = [
        ("1. Data Ingestion", "Kaggle MRI Dataset\n3,264 Images\n4 Classes"),
        ("2. Validation & Preprocess", "MRIScanValidator\nResize 224x224 RGB\nNormalize [0, 1]"),
        ("3. Data Augmentation", "Random Flip (H)\nRotation (±9°)\nZoom (±5%), Clip"),
        ("4. Custom CNN Model", "4 Conv Blocks + BN\nGlobalAvgPool\nDense(256) + Dropout"),
        ("5. Explainable AI", "Grad-CAM Engine\nlast_conv_layer Hook\nHeatmap + Overlay"),
        ("6. Deployment / UI", "Streamlit Web App\nReal-time Inference\nDiagnostic Report")
    ]
    
    colors = ['#1E3A8A', '#0D9488', '#D97706', '#4F46E5', '#DC2626', '#059669']
    
    box_width = 1.35
    box_height = 1.8
    y_pos = 0.6
    spacing = 1.6
    
    for i, ((title, desc), color) in enumerate(zip(stages, colors)):
        x_pos = 0.2 + i * spacing
        
        # Main box
        rect = patches.FancyBboxPatch(
            (x_pos, y_pos), box_width, box_height,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor='#F8FAFC', edgecolor=color, linewidth=2.0
        )
        ax.add_patch(rect)
        
        # Header banner
        header_rect = patches.FancyBboxPatch(
            (x_pos, y_pos + box_height - 0.45), box_width, 0.45,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=color, edgecolor=color, linewidth=1.0
        )
        ax.add_patch(header_rect)
        
        # Header text
        ax.text(x_pos + box_width/2, y_pos + box_height - 0.25, title,
                ha='center', va='center', color='white', fontsize=7.5, fontweight='bold')
        
        # Body text
        ax.text(x_pos + box_width/2, y_pos + 0.65, desc,
                ha='center', va='center', color='#334155', fontsize=6.8, linespacing=1.3)
        
        # Arrow to next stage
        if i < len(stages) - 1:
            ax.annotate('', xy=(x_pos + box_width + 0.2, y_pos + box_height/2),
                        xytext=(x_pos + box_width + 0.05, y_pos + box_height/2),
                        arrowprops=dict(facecolor='#64748B', edgecolor='#64748B', arrowstyle="->", lw=2.0))
            
    ax.set_xlim(0, 9.8)
    ax.set_ylim(0.2, 2.7)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "workflow_diagram.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved:", save_path)


# ---------------------------------------------------------------------------
# 2. Custom CNN Architecture Block Diagram
# ---------------------------------------------------------------------------
def generate_cnn_architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=200)
    ax.axis('off')
    
    blocks = [
        ("Input", "224x224x3\nRGB MRI", "#475569"),
        ("Block 1", "Conv2D(32) x2\nBN + ReLU\nMaxPool(2x2)\n[112x112x32]", "#2563EB"),
        ("Block 2", "Conv2D(64) x2\nBN + ReLU\nMaxPool(2x2)\n[56x56x64]", "#3B82F6"),
        ("Block 3", "Conv2D(128) x2\nBN + ReLU\nMaxPool(2x2)\n[28x28x128]", "#6366F1"),
        ("Block 4\n(Grad-CAM)", "Conv2D(256)\nConv2D(256)*\nBN + ReLU + Pool\n[14x14x256]", "#DC2626"),
        ("Global Pooling", "GlobalAverage\nPooling2D\n[256 vector]", "#0D9488"),
        ("Classifier Head", "Dense(256)+BN\nDropout(0.4)\nDense(4, Softmax)\n[4 Probabilities]", "#059669")
    ]
    
    box_width = 1.25
    box_height = 2.4
    y_pos = 0.5
    spacing = 1.52
    
    for i, (title, desc, color) in enumerate(blocks):
        x_pos = 0.2 + i * spacing
        
        # Outer box
        rect = patches.FancyBboxPatch(
            (x_pos, y_pos), box_width, box_height,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor='#FFFFFF', edgecolor=color, linewidth=2.2
        )
        ax.add_patch(rect)
        
        # Header banner
        header_height = 0.65 if "\n" in title else 0.45
        header_rect = patches.FancyBboxPatch(
            (x_pos, y_pos + box_height - header_height), box_width, header_height,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=color, edgecolor=color, linewidth=1.0
        )
        ax.add_patch(header_rect)
        
        # Header text
        ax.text(x_pos + box_width/2, y_pos + box_height - header_height/2, title,
                ha='center', va='center', color='white', fontsize=7.2, fontweight='bold', linespacing=1.1)
        
        # Body text
        ax.text(x_pos + box_width/2, y_pos + (box_height - header_height)/2 + 0.1, desc,
                ha='center', va='center', color='#1E293B', fontsize=6.8, linespacing=1.25)
        
        # Connecting arrow
        if i < len(blocks) - 1:
            ax.annotate('', xy=(x_pos + box_width + 0.22, y_pos + box_height/2),
                        xytext=(x_pos + box_width + 0.05, y_pos + box_height/2),
                        arrowprops=dict(facecolor='#475569', edgecolor='#475569', arrowstyle="->", lw=2.2))
            
    # Note on Grad-CAM target
    ax.text(5.5, 0.15, "* 'last_conv_layer' is the second Conv2D in Block 4, serving as target for Grad-CAM gradients.",
            ha='center', va='center', color='#991B1B', fontsize=7.5, style='italic', fontweight='bold')
    
    ax.set_xlim(0, 11.0)
    ax.set_ylim(0.0, 3.2)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "cnn_architecture_diagram.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved:", save_path)


# ---------------------------------------------------------------------------
# 3. 6-Tier System Architecture Diagram
# ---------------------------------------------------------------------------
def generate_system_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    ax.axis('off')
    
    layers = [
        ("Layer 6: User Interface Layer", "Streamlit Web Application (app/app.py) | CLI Interface (main.py) | Jupyter Notebooks (01-06)", "#1E293B"),
        ("Layer 5: Deployment & Inference Layer", "TumorClassifier (src/deployment/inference.py) | PredictionResult Dataclass | Threshold Enforcement", "#2563EB"),
        ("Layer 4: Explainable AI & Visualization", "GradCAM Engine (src/explainability/gradcam.py) | Visualizer & Confusion Matrix (src/evaluation/)", "#D97706"),
        ("Layer 3: Model & Training Layer", "Custom CNN (src/models/custom_cnn.py) | Callbacks Factory (src/models/callbacks.py) | Trainer (src/training/)", "#7C3AED"),
        ("Layer 2: Preprocessing & Validation Layer", "MRIScanValidator (mri_validator.py) | DataLoader & tf.data (data_loader.py) | Augmentation (augmentation.py)", "#0D9488"),
        ("Layer 1: Dataset & Storage Layer", "Raw MRI Images (data/raw/) | Metadata & Class Mapping (data/metadata/) | Saved Models & Artifacts (outputs/)", "#059669")
    ]
    
    box_w = 9.2
    box_h = 0.58
    gap = 0.18
    
    for i, (title, desc, color) in enumerate(layers):
        y = 4.0 - i * (box_h + gap)
        
        rect = patches.FancyBboxPatch(
            (0.4, y), box_w, box_h,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            facecolor='#F8FAFC', edgecolor=color, linewidth=2.0
        )
        ax.add_patch(rect)
        
        # Indicator strip
        strip = patches.FancyBboxPatch(
            (0.4, y), 0.25, box_h,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor=color, edgecolor=color, linewidth=0
        )
        ax.add_patch(strip)
        
        # Text
        ax.text(0.85, y + box_h/2 + 0.1, title, ha='left', va='center',
                color=color, fontsize=8.0, fontweight='bold')
        ax.text(0.85, y + box_h/2 - 0.14, desc, ha='left', va='center',
                color='#334155', fontsize=7.0)
        
        # Downward bidirectional arrow
        if i < len(layers) - 1:
            ax.annotate('', xy=(5.0, y - gap + 0.05), xytext=(5.0, y - 0.03),
                        arrowprops=dict(facecolor='#94A3B8', edgecolor='#94A3B8', arrowstyle="<->", lw=1.5))
            
    ax.set_xlim(0, 10.0)
    ax.set_ylim(-0.3, 4.8)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "system_architecture_diagram.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved:", save_path)


# ---------------------------------------------------------------------------
# 4. EDA Class Distribution & Imbalance Chart
# ---------------------------------------------------------------------------
def generate_eda_class_distribution():
    classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
    train_counts = [826, 822, 395, 827]
    test_counts = [100, 115, 105, 74]
    
    x = np.arange(len(classes))
    width = 0.38
    
    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=200)
    
    rects1 = ax.bar(x - width/2, train_counts, width, label='Training Images', color='#2563EB', edgecolor='#1D4ED8')
    rects2 = ax.bar(x + width/2, test_counts, width, label='Testing Images', color='#D97706', edgecolor='#B45309')
    
    ax.set_ylabel('Number of MRI Scans', fontsize=9, fontweight='bold', color='#1E293B')
    ax.set_title('Dataset Class Distribution & Imbalance (Kaggle Brain Tumor MRI Dataset)', fontsize=10, fontweight='bold', color='#1E293B', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.legend(fontsize=8.5, frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    # Value annotations on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1E40AF')
        
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#92400E')
        
    # Class imbalance annotation
    ax.annotate('Severe Imbalance in Training:\n"No Tumor" has only 395 images\n(~48% fewer than tumor classes)',
                xy=(2 - width/2, 395), xytext=(2 - width/2, 620),
                arrowprops=dict(facecolor='#DC2626', edgecolor='#DC2626', arrowstyle="->", lw=1.5),
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#DC2626',
                bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#FCA5A5", lw=1.0))
    
    ax.set_ylim(0, 980)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "eda_class_distribution.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved:", save_path)


if __name__ == "__main__":
    generate_workflow_diagram()
    generate_cnn_architecture_diagram()
    generate_system_architecture_diagram()
    generate_eda_class_distribution()
    print("All diagrams generated successfully in", OUTPUT_DIR)
