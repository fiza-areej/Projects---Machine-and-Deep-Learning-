Deep Convolutional GAN (DCGAN) \& Multi-Task Discriminator



\## Objective



This assignment focuses on implementing and understanding Generative Adversarial Networks (GANs) and extending the discriminator into a multi-label classifier. After completing this assignment, you should be able to:



\* Understand how GANs work and how the generator and discriminator interact

\* Implement DCGAN using convolutional and transpose convolution layers

\* Train GANs using adversarial loss

\* Extend the discriminator into a multi-task model performing real/fake detection + image classification

\* Implement GradCAM/Heatmap visualization

\* Build a complete training and evaluation pipeline



---



\## What This Assignment Does



This assignment implements a **DCGAN** and a **multi-task discriminator** using the Malaria cell dataset. The workflow consists of:



1\. Data Loading



&nbsp;  \* Custom DataLoader reads HCM images

&nbsp;  \* Classification labels are read from annotation files (Ring, Trophozoite, Schizont, Gametocyte)



2\. DCGAN Implementation



&nbsp;  \* Generator:



&nbsp;    \* Takes 100-dimensional noise vector

&nbsp;    \* Upsamples using transpose convolutions

&nbsp;    \* Produces synthetic malaria cell images

&nbsp;  \* Discriminator:



&nbsp;    \* CNN-based model that downsamples features

&nbsp;    \* Outputs probability of \*real vs fake\*



3\. Training Loop



&nbsp;  \* Different optimizers for generator and discriminator

&nbsp;  \* Binary cross-entropy loss for adversarial training

&nbsp;  \* Generated samples are visualized every epoch



4\. Multi-Task Discriminator



&nbsp;  \* Added classification head before final layers

&nbsp;  \* Two fully connected layers classify the 4 malaria cell types

&nbsp;  \* Total discriminator loss = adversarial loss + classification loss



5\. GradCAM / Heatmap Visualization



&nbsp;  \* Extract last convolutional layer

&nbsp;  \* Visualize model attention and feature activation regions



6\. Testing \& Evaluation



&nbsp;  \* Test file loads model + weights

&nbsp;  \* For GAN → random noise

&nbsp;  \* For classifier → real test images

&nbsp;  \* Model checkpoints saved and loaded



---



\## Dataset



\* \*\*Malaria Cell HCM Dataset\*\*

\* Contains 4 classes:



&nbsp; \* Ring

&nbsp; \* Trophozoite

&nbsp; \* Schizont

&nbsp; \* Gametocyte

\* Includes bounding-box annotations

\* Train/val/test split available

\* GAN uses only images; classifier uses labels



---



\## Key Components



\* **Generator**



&nbsp; \* Linear → reshape to 128×4×4

&nbsp; \* ConvTranspose2d layers

&nbsp; \* ReLU activations

&nbsp; \* Tanh output image generation



\* **Discriminator**



&nbsp; \* Sequential convolution layers

&nbsp; \* Stride-2 downsampling

&nbsp; \* LeakyReLU activations

&nbsp; \* Real/Fake output



\* **Multi-Task Classification Head**



&nbsp; \* Input from second-last conv layer

&nbsp; \* Two FC layers

&nbsp; \* Outputs 4-class prediction

&nbsp; \* Cross-entropy loss



\* **Training Pipeline**



&nbsp; \* BCE for GAN

&nbsp; \* CE for classifier

&nbsp; \* Loss curves and generated images saved



\* **GradCAM**



&nbsp; \* Heatmaps showing which regions influence classification



---



\## Results



\* Generator and discriminator loss curves

\* Validation and training accuracy curves for classification

\* GradCAM heatmaps from discriminator

\* Visual samples of generated malaria images

\* Comparison before/after adding classification head

\* Visualization of generator and discriminator weights



---



\## References / Helping Material



1\. GAN Tutorial – Lilian Weng

&nbsp;  \[https://lilianweng.github.io/posts/2017-08-20-gan/](https://lilianweng.github.io/posts/2017-08-20-gan/)



2\. DCGAN Paper

&nbsp;  \[https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)



3\. Torch ConvTranspose2d

&nbsp;  \[https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)



4\. GradCAM Paper

&nbsp;  \[https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)



5\. Malaria Dataset (HCM) – CVPR 2022

&nbsp;  Dataset and annotation details provided by IMAL Lab



