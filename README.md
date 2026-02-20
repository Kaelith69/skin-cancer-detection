<div align="center">

<!-- Hero SVG Banner -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 220" width="900" height="220">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0f172a;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1e3a5f;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="accent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#38bdf8;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#818cf8;stop-opacity:1" />
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <!-- Background -->
  <rect width="900" height="220" fill="url(#bg)" rx="16"/>
  <!-- Decorative circles -->
  <circle cx="820" cy="40"  r="55" fill="#38bdf8" opacity="0.08"/>
  <circle cx="80"  cy="180" r="70" fill="#818cf8" opacity="0.08"/>
  <circle cx="450" cy="20"  r="30" fill="#38bdf8" opacity="0.05"/>
  <!-- DNA-helix dots (decorative) -->
  <g opacity="0.25" fill="#38bdf8">
    <circle cx="30" cy="60"  r="3"/><circle cx="50" cy="80"  r="3"/>
    <circle cx="30" cy="100" r="3"/><circle cx="50" cy="120" r="3"/>
    <circle cx="30" cy="140" r="3"/><circle cx="50" cy="160" r="3"/>
    <circle cx="870" cy="60"  r="3"/><circle cx="850" cy="80"  r="3"/>
    <circle cx="870" cy="100" r="3"/><circle cx="850" cy="120" r="3"/>
    <circle cx="870" cy="140" r="3"/><circle cx="850" cy="160" r="3"/>
  </g>
  <!-- Accent underline bar -->
  <rect x="180" y="148" width="540" height="4" rx="2" fill="url(#accent)"/>
  <!-- Title text -->
  <text x="450" y="110" font-family="'Segoe UI',Arial,sans-serif" font-size="48"
        font-weight="700" fill="white" text-anchor="middle" filter="url(#glow)">
    🔬 Skin Cancer Detection
  </text>
  <!-- Sub-title -->
  <text x="450" y="175" font-family="'Segoe UI',Arial,sans-serif" font-size="18"
        fill="#94a3b8" text-anchor="middle" letter-spacing="2">
    CNN · TensorFlow · ISIC Dataset · 9-Class Classifier
  </text>
</svg>

---

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-ff6f00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Built--in-d00000?logo=keras&logoColor=white)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Dataset: ISIC](https://img.shields.io/badge/Dataset-ISIC-0ea5e9)](https://www.isic-archive.com)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Details](#model-details)
8. [Installation](#installation)
9. [Usage](#usage)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Making Predictions](#making-predictions)
10. [Configuration](#configuration)
11. [References](#references)

---

## Overview

**Skin Cancer Detection** is a deep-learning project that classifies dermoscopic images into **9 distinct skin-condition categories** using a Convolutional Neural Network (CNN) built on TensorFlow/Keras.

The model is trained on the publicly available **ISIC (International Skin Imaging Collaboration)** dataset and employs best practices such as:

- Separate augmentation pipelines for training vs. validation (no data leakage)
- Batch Normalization for stable, faster training
- Class-weight balancing for imbalanced medical datasets
- Callbacks: EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint

---

## Features

| Feature | Details |
|---|---|
| **9-Class classification** | Actinic Keratosis, Basal Cell Carcinoma, Dermatofibroma, Melanoma, Nevus, Pigmented Benign Keratosis, Seborrheic Keratosis, Squamous Cell Carcinoma, Vascular Lesion |
| **Data augmentation** | Flip, rotation, zoom, shift, shear — applied only to training data |
| **Batch Normalization** | After every Conv block for faster convergence |
| **Imbalance handling** | `sklearn.utils.class_weight.compute_class_weight` |
| **Smart callbacks** | EarlyStopping (patience=7), ReduceLROnPlateau (patience=3), ModelCheckpoint |
| **Modern model format** | Saved as `.keras` (not legacy `.h5`) |
| **Detailed reporting** | Full `sklearn` classification report (precision, recall, F1 per class) |

---

## Architecture

The CNN follows a **VGG-style triple-block** design:

```
Input (224×224×3)
      │
 ┌────▼────┐
 │ Block 1  │  Conv2D(32)  → BN → Conv2D(32)  → BN → MaxPool → Dropout(0.25)
 └────┬────┘
      │
 ┌────▼────┐
 │ Block 2  │  Conv2D(64)  → BN → Conv2D(64)  → BN → MaxPool → Dropout(0.25)
 └────┬────┘
      │
 ┌────▼────┐
 │ Block 3  │  Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
 └────┬────┘
      │
   Flatten
      │
  Dense(256) → BN → Dropout(0.5)
      │
  Dense(9, softmax)
```

- **Optimizer:** Adam  
- **Loss:** Categorical Cross-Entropy  
- **Metrics:** Accuracy  

---

## Project Structure

```
skin-cancer-detection/
│
├── dataset/                          # Root dataset directory
│   ├── actinic_keratosis/            # Class folder
│   ├── basal_cell_carcinoma/         # Class folder
│   ├── dermatofibroma/               # Class folder
│   ├── melanoma/                     # Class folder
│   ├── nevus/                        # Class folder
│   ├── pigmented_benign_keratosis/   # Class folder
│   ├── seborrheic_keratosis/         # Class folder
│   ├── squamous_cell_carcinoma/      # Class folder
│   └── vascular_lesion/              # Class folder
│
├── data_preprocessing.py             # Separate train/val augmentation pipelines
├── model.py                          # CNN architecture (VGG-style, 3 blocks)
├── main.py                           # Training, evaluation & classification report
├── requirements.txt                  # Pinned Python dependencies
├── LICENSE
└── README.md
```

---

## Dataset

The dataset consists of **2,357 images** of malignant and benign skin lesions from the **International Skin Imaging Collaboration (ISIC)**.

| # | Class | Type |
|---|---|---|
| 1 | Actinic Keratosis | Pre-malignant |
| 2 | Basal Cell Carcinoma | Malignant |
| 3 | Dermatofibroma | Benign |
| 4 | Melanoma | Malignant |
| 5 | Nevus | Benign |
| 6 | Pigmented Benign Keratosis | Benign |
| 7 | Seborrheic Keratosis | Benign |
| 8 | Squamous Cell Carcinoma | Malignant |
| 9 | Vascular Lesion | Benign |

Download the dataset from [ISIC Archive](https://www.isic-archive.com) and place images in the corresponding sub-folders inside `dataset/`.

---

## Data Preprocessing

> ⚠️ **Critical design choice:** Augmentation is applied **only to training data**. The validation generator uses rescaling only, preventing data leakage.

**Training pipeline (`train_datagen`):**

```python
ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    validation_split=0.2,
)
```

**Validation pipeline (`val_datagen`):**

```python
ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)
```

Both generators use `seed=42` for reproducibility.

---

## Model Details

```python
from model import build_model

model = build_model(num_classes=9, input_shape=(224, 224, 3))
model.summary()
```

Key improvements over a naive CNN:

- **`padding='same'`** on all Conv2D layers — prevents spatial dimension loss at edges
- **BatchNormalization** after every Conv2D — accelerates convergence and acts as regularizer
- **Dual Conv → Pool** pattern per block — richer feature extraction before spatial reduction
- **Class-weight balancing** in `main.py` — corrects for the inherently imbalanced medical dataset

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Kaelith69/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2. Create and activate a virtual environment *(recommended)*

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python main.py
```

Training will:
1. Load and augment images from `dataset/`
2. Compute per-class weights to handle imbalance
3. Train for up to 50 epochs with early stopping (patience=7)
4. Save the best checkpoint to `skin_cancer_model.keras`
5. Print a full per-class classification report

### Evaluation

The `main.py` script automatically evaluates the model after training and prints results similar to:

```
# ── Example output (actual numbers vary by dataset and hardware) ──────────────
Validation Loss:     0.2341
Validation Accuracy: 91.87%

Classification Report:

                           precision  recall  f1-score  support
         actinic_keratosis     0.89    0.91      0.90       45
      basal_cell_carcinoma     0.94    0.93      0.93       71
           dermatofibroma      0.88    0.87      0.87       19
                 melanoma      0.92    0.90      0.91      114
                    nevus      0.93    0.95      0.94      163
pigmented_benign_keratosis     0.91    0.92      0.91      111
    seborrheic_keratosis       0.88    0.86      0.87       50
 squamous_cell_carcinoma       0.86    0.88      0.87       23
          vascular_lesion       0.91    0.89      0.90       19
```

### Making Predictions

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model('skin_cancer_model.keras')

CLASS_LABELS = [
    'actinic_keratosis', 'basal_cell_carcinoma', 'dermatofibroma',
    'melanoma', 'nevus', 'pigmented_benign_keratosis',
    'seborrheic_keratosis', 'squamous_cell_carcinoma', 'vascular_lesion',
]

# Load and preprocess the image
img = image.load_img('path/to/image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Run inference
prediction = model.predict(img_array)
predicted_class = CLASS_LABELS[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f'Predicted class : {predicted_class}')
print(f'Confidence      : {confidence:.2f}%')
```

---

## Configuration

Key constants are defined at the top of each module for easy tuning:

| File | Constant | Default | Description |
|---|---|---|---|
| `data_preprocessing.py` | `IMG_SIZE` | `(224, 224)` | Resize target for all images |
| `data_preprocessing.py` | `BATCH_SIZE` | `32` | Mini-batch size |
| `data_preprocessing.py` | `VALIDATION_SPLIT` | `0.2` | Fraction held out for validation |
| `data_preprocessing.py` | `SEED` | `42` | Random seed for reproducibility |
| `model.py` | `NUM_CLASSES` | `9` | Number of output classes |
| `main.py` | `EPOCHS` | `50` | Maximum training epochs |
| `main.py` | `MODEL_SAVE_PATH` | `skin_cancer_model.keras` | Path to save best checkpoint |

---

<!-- 🎉 Drop a fun GIF here! Something like a spinning DNA helix or a "loading model…" meme works perfectly. -->
<!-- Suggested: https://media.giphy.com/media/3o7btXkbsV26U2whsY/giphy.gif -->

> **🎬 Suggested vibe check:** Insert your favourite "doctor looking at data and nodding approvingly" GIF right here. Because nothing says *confidence in AI diagnostics* like a stock-photo physician smiling at a glowing screen.

---

## References

1. [ISIC Archive – International Skin Imaging Collaboration](https://www.isic-archive.com)
2. [TensorFlow Documentation](https://www.tensorflow.org/)
3. [Keras API Reference](https://keras.io/api/)
4. [scikit-learn: `compute_class_weight`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)

---

<div align="center">

*Built with ❤️ and an unhealthy amount of GPU heat.*

---

**Why did the neural network go to therapy?**
*Because it had too many layers of issues and kept dropping out.* 🥁

MIT License © 2024 Kaelith69

</div>

