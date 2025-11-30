
# **Assignment 1: Implementation of Linear and Logistic Regression**

## **Objective**

* Gain hands-on experience with Numpy arrays and PyTorch for regression tasks.
* Implement Linear Regression using Stochastic Gradient Descent (SGD).
* Implement Logistic Regression for binary classification using PyTorch.
* Understand feedforward, gradient computation, and optimization mechanisms.
* Analyze the effect of hyperparameters like learning rate, batch size, and initialization on model performance.

---

## **What This Assignment Does**

This assignment involves two main tasks:

### **Task 1: Linear Regression**

* Load and normalize the California Housing dataset using Numpy.
* Split data into training (70%), validation (20%), and testing (10%) sets.
* Implement a Linear Regression model from scratch using Numpy arrays.
* Train the model using mini-batch SGD.
* Compute L2 loss and update model parameters using gradient descent.
* Save the trained model and perform testing on the test set.
* Visualize training, validation, and test losses across epochs.
* Experiment with hyperparameters: learning rate, batch size, and weight initialization.

### **Task 2: Logistic Regression**

* Load and preprocess the Titanic dataset.
* Implement a Logistic Regression model in PyTorch for binary classification.
* Normalize data and split into training, validation, and test sets.
* Train the model using gradient descent or SGD with cross-entropy loss.
* Compute gradients and update model weights iteratively.
* Save and load the trained model for testing.
* Evaluate performance using accuracy, confusion matrix, and F1 score.
* Visualize loss curves and prediction results, and compare performance under different hyperparameters.

---

## **Datasets**

1. **California Housing Dataset**

   * 20,640 samples with 8 features each.
   * Target: Median house value (scaled in 100,000s).
   * Used for Linear Regression task.

2. **Titanic Dataset**

   * 891 training samples and 418 testing samples.
   * Features include age, sex, passenger class, etc.
   * Target: Survived or not (binary classification).

---

## **References**

1. California Housing Dataset Documentation
   [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
2. PyTorch Logistic Regression Tutorial
   [https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
3. Gradient Descent & SGD Concepts
   [https://cs231n.github.io/optimization-1/](https://cs231n.github.io/optimization-1/)
4. Evaluation Metrics: Accuracy, Confusion Matrix, F1 Score
   [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)

