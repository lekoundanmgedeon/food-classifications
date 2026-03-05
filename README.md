# 🍲 Food Image Recognition Challenge

## 📌 Description

This competition challenges  to train a neural network capable of recognizing different food dishes from images.

The objective is to classify each image into the correct food category using Deep Learning techniques.

Examples of possible classes include:

- Thieboudienne  
- Yassa Poulet  
- Mafé  
- Attiéké  
- Jollof Rice  

The goal is to practice **Computer Vision with Deep Learning**, using Convolutional Neural Networks and transfer learning techniques.

---

## 📂 Dataset

We will use a food image dataset composed of labeled images for each dish category.

Possible dataset sources include:

👉 African Food Dataset (Kaggle)  
👉 Food-101 Dataset (subset)

👉 Kaggle Dataset:  
https://www.kaggle.com/datasets/dansbecker/food-101


The dataset contains:

- 101 food categories
- 101,000 images
- 750 training images and 250 test images per class

For this competition, we will use a **subset of the dataset** focusing on selected dishes.


The dataset contains:

- Images of different food dishes
- Labels indicating the food category
- Training and test splits

Typical structure:
    data/
    ├── train/
    ├── test/
    ├── train.csv
    └── test.csv

Example `train.csv`:
id,label
img_001.jpg,thieboudienne
img_002.jpg,yassa

👉  Dataset (Option 2 ): 
https://data.mendeley.com/datasets/rrzhwbg3kw/1 (Ghana and cameroun food) 
 
https://data.mendeley.com/datasets/2vktdxfxv7/2 (Nigerian food)



---

## ⚙️ Learning Objectives

Through this competition, you will:

- Understand how **image classification with CNNs** works.
- Apply **data preprocessing and augmentation** techniques.
- Compare different neural network architectures.
- Learn how **transfer learning** improves performance on small datasets.
- Evaluate models using appropriate metrics.

---

## 🏆 Evaluation

Models will be evaluated based on:

- **Accuracy**
- **Macro F1-score**
- **Confusion Matrix** to visualize misclassifications

The primary ranking metric for the leaderboard will be:

**Macro F1-score**

This metric ensures that all classes are treated equally even if the dataset is imbalanced.

---

## 🚀 Instructions

### 1. Download the Dataset

Download the dataset and place it in the `data/` directory.

### 2. Preprocessing

Typical preprocessing steps include:

- Resize images (e.g., 224×224)
- Normalize pixel values
- Apply data augmentation techniques such as:
  - Random flip
  - Rotation
  - Color jitter

### 3. Modeling

Participants should:

- Implement a **baseline CNN model**
- Experiment with more advanced architectures such as:

  - ResNet
  - EfficientNet
  - MobileNet
  - Vision Transformers

### 4. Training

Recommended practices:

- Train for 10–30 epochs
- Use Adam or SGD optimizer
- Apply early stopping or regularization

### 5. Submission

Participants must submit:

- A **Jupyter Notebook (.ipynb)** containing:
  - preprocessing
  - model training
  - evaluation
- A **prediction file** in CSV format.

Example submission format:
    id,label
    img_901.jpg,thieboudienne
    img_902.jpg,yassa


---

## 📖 Useful Resources

- Deep Learning for Computer Vision
- PyTorch Image Classification Tutorial
- Transfer Learning with CNNs
- Food-101 Dataset

---

## 📌 Baseline Example

A simple baseline model could include:

- Image resizing to **224×224**
- A **CNN with 2–3 convolutional layers**
- ReLU activation
- MaxPooling layers
- Adam optimizer
- Training for **10 epochs**

Example architecture:


More advanced baselines include **ResNet18 with transfer learning**.

---

## 📅 Deadline

Submissions must be made before **[date to be defined]**.

Final results and model comparisons will be discussed in class.

---

## 🏆 Leaderboard

To participate:

1. Submit your predictions as: ``results/submission.csv``

2. Run the evaluation script: ``results/leaderboard.csv``

3. The leaderboard will automatically update in: ``results/leaderboard.csv``


Ranking is based on **Macro F1-score**.

---

Good luck and have fun experimenting with Deep Learning! 🚀


