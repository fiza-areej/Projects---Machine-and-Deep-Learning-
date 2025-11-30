UNet-based Segmentation (Breast Cancer Images)

## Objective

* Implement image segmentation models using UNet.
* Compare UNet **with** and **without** skip connections.
* Learn proper use of Dataset & DataLoader for large medical image datasets.
* Evaluate segmentation performance using Dice Score and visual analysis.

---

## What This Assignment Does

This assignment focuses on building an end-to-end medical image segmentation pipeline using the Breast Cancer dataset.
You will:

* Load images, masks, and labels (0 = Normal, 1 = Benign, 2 = Malignant) efficiently without loading everything into memory.
* Implement two architectures:

  * UNet with skip connections
  * UNet without skip connections
* Train both models with early stopping, best-weight saving, Dice Loss, and learning rate decay.
* Test the models and generate predictions.
* Run real-time inference on a user-provided image.
* Visualize loss curves, dice score curves, accurate vs inaccurate predictions, and TSNE representation of encoder output.
* Compare results of both architectures and explain their differences.

---

## Dataset

* Breast Cancer Image Dataset (PNG format)
* Contains:

  * Original images
  * Ground truth masks
  * Three classes: Normal, Benign, Malignant
* Must be split into training, validation, and testing.
* All images and masks must be normalized and loaded lazily using a custom Dataset class.

---

## References

1. UNet: Convolutional Networks for Biomedical Image Segmentation
   [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
2. Breast Cancer Medical Imaging segmentation literature (general reference)
3. PyTorch documentation for Dataset, DataLoader, optimizers
4. TSNE Visualization:
   [https://lvdmaaten.github.io/tsne/](https://lvdmaaten.github.io/tsne/)


