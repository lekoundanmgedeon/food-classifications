# 🍲 Food Image Recognition Challenge

## 📌 Description

This competition challenges  to train a neural network capable of recognizing different food dishes from images.

The objective is to classify each image into the correct food category using Deep Learning techniques.

Examples of possible classes include:

- Akara  
- banga soup  
- Massa 
- ewedu soup 
- Jollof Rice  

The goal is to practice **Computer Vision with Deep Learning**, using Convolutional Neural Networks and transfer learning techniques.

---

## 📂 Dataset

We will use a food image dataset composed of labeled images for each dish category.

 https://data.mendeley.com/datasets/2vktdxfxv7/2 (Nigerian food) (USE THIS DATASET)

The dataset contains:

- Images of different food dishes
- Labels indicating the food category
- Training and test splits

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
    img_901.jpg,1
    img_902.jpg,2


---
## 📁 Structure du Projet

```text
food-classifications/
├── data/
│   ├── train/          # Images d'entraînement classées par dossiers
│   └── test/           # Images de test (votre but est de les prédire)
├── src/
│   ├── train.py        # Main script of training
│   ├── predict.py      # Main script of prediction
│   └── model_baseline.py # Here you define architecture of your model
├── evaluation/
│   ├── evaluate.py     # Compute you model metric (Acc/F1)
│   └── leaderboard.py  # Update of leaderboard
└── submissions/        # Folder were you will put your submit result

```
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

## ⚖️ Submission rules
- Your CSV file must contain exactly 303 lines.
- The format must be: id (image name) and prediction (class index).
- The final evaluation is based on Accuracy and the F1-Score Macro.
---


## Train model

``python src/train.py``

## Submission 

``python src/predict.py`` 

## Evaluation 
Evaluate score of your model
```python evaluation/evaluate.py```
Submit your evaluation on leaderboard 
```python evaluation/leaderboard.py```

## 📅 Deadline

Submissions must be made before **13-03-2026**.

Final results and model comparisons will be discussed in class.

---

## 🏆 Leaderboard

To participate:

1. Submit your predictions as: ``submissions/team_submission.csv``

2. Run the evaluation script: ``evaluation/leaderboard.csv``

3. The leaderboard will automatically update in: ``leaderbord/leaderboard.csv``


Ranking is based on **Macro F1-score**.

---

Good luck and have fun experimenting with Deep Learning! 🚀


