# Dataset Documentation — Brain Tumor MRI Dataset

## Overview

| Property | Value |
|---|---|
| Source | [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Modality | MRI (T1-weighted, T2-weighted, FLAIR) |
| Classes | 4 (Glioma, Meningioma, Pituitary, No Tumor) |
| Format | JPEG/PNG |
| Resolution | Variable (resized to 224×224) |

## Class Descriptions

### Glioma Tumor
- **Origin**: Glial cells of the brain/spinal cord
- **Subtypes**: Glioblastoma (GBM, Grade IV), Astrocytoma, Oligodendroglioma
- **Frequency**: ~30% of all brain tumors; ~80% of malignant brain tumors
- **Prognosis**: Poor for high-grade (GBM median survival: 14 months)
- **MRI Characteristics**: Irregular borders, mass effect, surrounding edema, ring enhancement

### Meningioma Tumor
- **Origin**: Meninges (protective brain membranes)
- **Grade**: Mostly WHO Grade I (benign); Grade II/III are rare
- **Frequency**: ~37% of all brain tumors (most common primary brain tumor)
- **Prognosis**: Good for benign cases; surgical resection is curative
- **MRI Characteristics**: Well-defined, extra-axial mass, homogeneous enhancement

### Pituitary Tumor (Adenoma)
- **Origin**: Pituitary gland (anterior lobe most common)
- **Classification**: Microadenoma (<10mm), Macroadenoma (≥10mm)
- **Effects**: Hormonal imbalances (Cushing's, acromegaly), vision problems
- **MRI Characteristics**: Located at sella turcica, may compress optic chiasm

### No Tumor
- **Description**: Normal brain MRI with no pathological findings
- **Significance**: Establishes baseline for comparison with tumor classes

## Folder Structure

```
data/raw/
├── Training/
│   ├── glioma/           (images)
│   ├── meningioma/       (images)
│   ├── notumor/          (images)
│   └── pituitary/        (images)
└── Testing/
    ├── glioma/           (images)
    ├── meningioma/       (images)
    ├── notumor/          (images)
    └── pituitary/        (images)
```

## Preprocessing Pipeline

1. **Load**: OpenCV reads as BGR, converted to RGB
2. **Resize**: Bilinear interpolation to 224×224
3. **Normalize**: Pixel values divided by 255 → [0, 1] float32
4. **Augmentation** (training only):
   - Random horizontal flip
   - Random rotation ±15°
   - Random zoom ±10%
   - Random translation ±10%
   - Random contrast ±10%
5. **Batch**: 32 images per batch
6. **Prefetch**: AUTOTUNE for GPU throughput
