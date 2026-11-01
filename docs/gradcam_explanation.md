# Grad-CAM: Explainable AI Methodology

## What is Grad-CAM?

Gradient-weighted Class Activation Mapping (**Grad-CAM**) is an XAI technique that produces
visual explanations for CNN predictions by highlighting the image regions most important
for the model's classification decision.

**Reference:** Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
(2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.*
Proceedings of the IEEE ICCV.

---

## Mathematical Formulation

### Step 1: Forward Pass

Given an input image **I**, the CNN produces feature maps from the last convolutional layer:

$$A^k \in \mathbb{R}^{H \times W}, \quad k = 1, \ldots, K$$

where K is the number of filters, H×W is the spatial resolution.

### Step 2: Compute Gradients

Compute the gradient of the class score $y^c$ (before softmax) with respect to each
feature map activation:

$$\frac{\partial y^c}{\partial A^k_{ij}}$$

This tells us how much a small change in activation $A^k_{ij}$ at position (i,j)
would change the class score.

### Step 3: Global Average Pooling of Gradients

Pool the gradients spatially to obtain importance weights $\alpha^c_k$:

$$\alpha^c_k = \underbrace{\frac{1}{Z} \sum_i \sum_j}_{\text{global avg. pool}} \frac{\partial y^c}{\partial A^k_{ij}}$$

where Z = H × W is the number of spatial positions.

### Step 4: Weighted Combination + ReLU

Compute the Grad-CAM map:

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha^c_k A^k\right)$$

The **ReLU** is applied because:
- Positive activations increase the class score (relevant to our target class).
- Negative activations decrease it (relevant to other classes) — we discard these.

### Step 5: Upsample and Overlay

1. Resize $L^c_{\text{Grad-CAM}}$ to the original image size using bilinear interpolation.
2. Normalize to [0, 1].
3. Apply a colormap (JET: blue→cyan→green→yellow→red).
4. Blend with the original MRI image using weighted addition.

---

## Why the Last Convolutional Layer?

- **Earlier layers** detect low-level features (edges, textures) — not class-discriminative.
- **The last conv layer** contains high-level semantic features while retaining spatial information.
- This is why no pooling is applied in Block 4 of our CNN — larger spatial maps give finer Grad-CAM resolution.

---

## Implementation in This Project

```python
from src.explainability.gradcam import GradCAM

cam = GradCAM(model, layer_name="last_conv_layer")
heatmap = cam.compute_heatmap(image_array, class_idx=None)  # None = predicted class
overlay = cam.overlay_on_image(original_image, heatmap)
fig = cam.visualize(image_array, original_image)
```

---

## Interpreting the Output

| Region Color | Interpretation |
|---|---|
| 🔴 Red/Yellow (hot) | High gradient → model focuses heavily on this region |
| 🔵 Blue (cool) | Low gradient → model considers this region less relevant |
| ⚪ Gray/Black | Gradient near zero → neutral contribution |

### Clinical Significance

For brain tumor detection:
- Grad-CAM should highlight the tumor mass region.
- If the heatmap focuses on non-tumor regions → model may be relying on confounders.
- Use Grad-CAM to validate that the model is **clinically sensible**.
