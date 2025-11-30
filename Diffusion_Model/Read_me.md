\# Diffusion Model Image Denoising



\## Objective

The purpose of this assignment is to provide a comprehensive understanding of diffusion models in machine learning. By completing this assignment, students should be able to:

\- Explain the concept of diffusion models.

\- Understand how noise is added to images and how images are denoised.

\- Implement a diffusion model architecture using neural networks.

\- Generate and evaluate images using diffusion models.



\## What This Assignment Does

This assignment implements a diffusion model for image denoising on a subset of an animal dataset. The workflow consists of:



1\. **Data Loading:** Custom PyTorch DataLoader reads images from folders and applies necessary transformations.

2\. **Forward Diffusion Process:** Gaussian noise is progressively added to input images over multiple steps to simulate a diffusion process.

3\. **Model Architecture:** A U-Net model is used to predict and remove noise in the reverse diffusion process.

4\. **Backward Process (Denoising):** The model learns to reverse the diffusion process and reconstruct the original image from noisy input.

5\. **Training and Evaluation:** The model is trained with MSE loss.



\## Dataset

The dataset consists of images of 15 animal classes. For training, a small subset (e.g., 20 images from 5 classes) is used. Images are resized and normalized before being fed to the model.



\## Key Components

\- **UNet Model:** Encoder-decoder architecture with skip connections to reconstruct images.

\- **Noise Scheduler:** Schedules Gaussian noise addition across multiple steps (`T` steps).

\- **Training Loop:** Optimizes the model to predict noise in images using MSE loss.



\## References / Helping Material

1\. \[A Very Short Introduction to Diffusion Models](https://kailashahirwar.medium.com/a-very-short-introduction-to-diffusion-models-a84235e4e9ae)  

2\. \[Diffusion Models Made Easy](https://towardsdatascience.com/diffusion-models-made-easy-8414298ce4da)  

3\. \[SuperAnnotate Blog on Diffusion Models](https://www.superannotate.com/blog/diffusion-models#:~:text=Diffusion%20models%20are%20advanced%20machine,learning%20to%20reverse%20this%20process.)  

4\. \[CMU Diffusion Models Slides](https://deeplearning.cs.cmu.edu/S24/document/slides/Diffusion\_Models.pdf)  



