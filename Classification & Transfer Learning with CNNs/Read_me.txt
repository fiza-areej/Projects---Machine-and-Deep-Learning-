 Classification & Transfer Learning with CNNs

## Objective

* Implement image classification using custom Convolutional Neural Networks (CNNs).
* Understand and apply transfer learning with pre-trained CNN models.
* Learn how to prepare a custom DataLoader for datasets.
* Evaluate classification performance using accuracy, confusion matrix, and F1 score.

---

## What This Assignment Does

This assignment focuses on classifying handwritten digits from the MNIST dataset. You will:

* Load and preprocess images and labels using a custom PyTorch Dataset and DataLoader.
* Split the training data into training and validation sets without using library functions.
* Implement a custom CNN model with multiple convolutional, batch normalization, pooling, and fully connected layers.
* Train the model while monitoring training and validation loss and accuracy.
* Apply early stopping and learning rate decay during training.
* Save and load the trained model for later use.
* Test the model on a separate test set and evaluate its performance using accuracy, confusion matrix, and F1 score.
* Perform runtime testing on user-provided images.
* Visualize loss curves, accuracy curves, correct and incorrect predictions, and model weights.
* Experiment with transfer learning using pre-trained models (optional for extended performance analysis).

---

## Dataset

* MNIST Handwritten Digit Dataset
* Contains 60,000 training images and 10,000 testing images of 28x28 grayscale digits.
* 10 classes representing digits from 0 to 9.
* Training data provided as images and corresponding CSV files with labels.
* Images must be normalized and labels one-hot encoded for model training.

---

## References

1. MNIST Dataset Overview
   [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
2. PyTorch Dataset and DataLoader Documentation
   [https://pytorch.org/docs/stable/data.html](https://pytorch.org/docs/stable/data.html)
3. CNN Basics and Transfer Learning Concepts
   [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
4. Evaluation Metrics: Accuracy, Confusion Matrix, F1 Score
   [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)


