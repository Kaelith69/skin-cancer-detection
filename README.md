<div align="center">

<!-- SVG Hero Banner — Healthcare palette: #0B8F87 / #2563EB / #10B981 -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 240" width="960" height="240">
  <defs>
    <linearGradient id="heroBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#062a27"/>
      <stop offset="100%" stop-color="#0d2b55"/>
    </linearGradient>
    <linearGradient id="heroAccent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#0B8F87"/>
      <stop offset="50%"  stop-color="#2563EB"/>
      <stop offset="100%" stop-color="#10B981"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="softglow">
      <feGaussianBlur stdDeviation="2" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="960" height="240" fill="url(#heroBg)" rx="18"/>
  <g stroke="#0B8F87" stroke-width="0.4" opacity="0.12">
    <line x1="0" y1="60"  x2="960" y2="60"/>
    <line x1="0" y1="120" x2="960" y2="120"/>
    <line x1="0" y1="180" x2="960" y2="180"/>
    <line x1="240" y1="0" x2="240" y2="240"/>
    <line x1="480" y1="0" x2="480" y2="240"/>
    <line x1="720" y1="0" x2="720" y2="240"/>
  </g>
  <circle cx="880" cy="50"  r="70" fill="#0B8F87" opacity="0.07" filter="url(#softglow)"/>
  <circle cx="80"  cy="195" r="80" fill="#2563EB" opacity="0.07" filter="url(#softglow)"/>
  <circle cx="480" cy="15"  r="35" fill="#10B981" opacity="0.06"/>
  <polyline points="60,140 90,140 105,100 120,170 135,115 150,140 200,140"
            fill="none" stroke="#10B981" stroke-width="2" opacity="0.55"
            stroke-linecap="round" stroke-linejoin="round"/>
  <polyline points="760,140 790,140 805,105 820,165 835,118 850,140 900,140"
            fill="none" stroke="#0B8F87" stroke-width="2" opacity="0.55"
            stroke-linecap="round" stroke-linejoin="round"/>
  <rect x="200" y="163" width="560" height="5" rx="3" fill="url(#heroAccent)" opacity="0.9"/>
  <text x="480" y="125"
        font-family="'Segoe UI',system-ui,Arial,sans-serif"
        font-size="52" font-weight="800"
        fill="white" text-anchor="middle"
        filter="url(#glow)" letter-spacing="-1">
    Skin Cancer Detection
  </text>
  <text x="480" y="190"
        font-family="'Segoe UI',system-ui,Arial,sans-serif"
        font-size="17" fill="#94a3b8"
        text-anchor="middle" letter-spacing="3">
    CNN  |  TensorFlow  |  ISIC Dataset  |  9-Class Classifier
  </text>
</svg>

</div>

---

## What even is this?

So you found a repo where a neural network stares at skin images until it figures out whether that mole is "just a mole" or "please make a dermatologist appointment immediately." Built with TensorFlow, trained on the ISIC archive, and armed with a VGG-style CNN that has *opinions* about your dermoscopic images.

No frontend. No REST API. Just pure, unapologetic `python main.py` energy.

> **Not a medical device.** Seriously. Do not replace your doctor with this. This is a research/learning project. Your actual healthcare decisions should involve actual healthcare professionals.

---

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Built--in-d00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Dataset: ISIC](https://img.shields.io/badge/Dataset-ISIC-0ea5e9?style=for-the-badge)](https://www.isic-archive.com)

</div>

---

## Table of Contents

1. [Live Demo](#live-demo)
2. [System Overview](#system-overview)
3. [Features](#features)
4. [Capability Visualization](#capability-visualization)
5. [Architecture Diagram](#architecture-diagram)
6. [Data Flow](#data-flow)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [Performance Stats](#performance-stats)
11. [Privacy](#privacy)
12. [Roadmap](#roadmap)
13. [License](#license)

---

## Live Demo

> **Place demo GIF here:** `assets/demo.gif`

![Demo](assets/demo.gif)

*What the GIF should show:*
- Running `python main.py` in a terminal
- Epoch-by-epoch training output scrolling (the satisfying kind)
- EarlyStopping kicking in and saving the best checkpoint
- Final classification report being printed class by class

Keep it short — 15 to 30 seconds is enough. No one needs to watch 50 epochs in real time.

---

## System Overview

**Skin Cancer Detection** classifies dermoscopic images into **9 distinct skin-condition categories** using a custom VGG-style Convolutional Neural Network built on TensorFlow/Keras.

Trained on **2,357 images** from the **International Skin Imaging Collaboration (ISIC)** dataset, the model handles the inherently messy real-world problem of class imbalance via sklearn's `compute_class_weight`, which is basically telling the model "hey, rare diseases exist too, please don't ignore them."

Key design decisions (that weren't made at 2 AM, probably):

- **Augmentation only on training data** — no data leakage for the validation set
- **Class-weight balancing** — compensates for dataset skew across 9 classes
- **Batch Normalization** after every Conv layer — faster convergence, built-in regularisation
- **Three smart callbacks** — EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Modern `.keras` format** — not the old `.h5` we don't talk about anymore

---

## Features

| Feature | Details |
|---|---|
| **9-Class classification** | Actinic Keratosis, Basal Cell Carcinoma, Dermatofibroma, Melanoma, Nevus, Pigmented Benign Keratosis, Seborrheic Keratosis, Squamous Cell Carcinoma, Vascular Lesion |
| **Data augmentation** | Horizontal flip, rotation 20 deg, zoom 20%, shift 10%, shear 10% — training only |
| **Batch Normalization** | After every Conv2D layer — convergence goes brrr |
| **Imbalance handling** | `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` |
| **Smart callbacks** | EarlyStopping (patience=7), ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6), ModelCheckpoint |
| **Modern model format** | Saved as `.keras` (not legacy `.h5`, we have standards) |
| **Detailed reporting** | Full sklearn classification report — precision, recall, F1, support per class |
| **Reproducibility** | `seed=42` on all data generators (the answer to the universe and everything, apparently) |

---

## Capability Visualization

<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 340" width="760" height="340">
  <defs>
    <linearGradient id="capBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#0a1628"/>
      <stop offset="100%" stop-color="#0d2b2a"/>
    </linearGradient>
    <linearGradient id="bar1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0B8F87"/>
      <stop offset="100%" stop-color="#10B981"/>
    </linearGradient>
    <linearGradient id="bar2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#2563EB"/>
      <stop offset="100%" stop-color="#0B8F87"/>
    </linearGradient>
  </defs>
  <rect width="760" height="340" fill="url(#capBg)" rx="14"/>
  <text x="380" y="36" font-family="'Segoe UI',Arial,sans-serif" font-size="16"
        font-weight="700" fill="#e2e8f0" text-anchor="middle" letter-spacing="1">
    MODEL CAPABILITY OVERVIEW
  </text>
  <g font-family="'Segoe UI',Arial,sans-serif" font-size="12" fill="#64748b">
    <text x="18" y="80">Classes</text>
    <text x="18" y="126">Augment</text>
    <text x="18" y="172">BN Depth</text>
    <text x="18" y="218">Callbacks</text>
    <text x="18" y="264">Report</text>
    <text x="18" y="310">Imbalance</text>
  </g>
  <g fill="#1e293b">
    <rect x="140" y="60"  width="560" height="28" rx="5"/>
    <rect x="140" y="106" width="560" height="28" rx="5"/>
    <rect x="140" y="152" width="560" height="28" rx="5"/>
    <rect x="140" y="198" width="560" height="28" rx="5"/>
    <rect x="140" y="244" width="560" height="28" rx="5"/>
    <rect x="140" y="290" width="560" height="28" rx="5"/>
  </g>
  <rect x="140" y="60"  width="560" height="28" rx="5" fill="url(#bar1)" opacity="0.9"/>
  <rect x="140" y="106" width="476" height="28" rx="5" fill="url(#bar1)" opacity="0.85"/>
  <rect x="140" y="152" width="498" height="28" rx="5" fill="url(#bar2)" opacity="0.85"/>
  <rect x="140" y="198" width="560" height="28" rx="5" fill="url(#bar2)" opacity="0.9"/>
  <rect x="140" y="244" width="560" height="28" rx="5" fill="url(#bar1)" opacity="0.9"/>
  <rect x="140" y="290" width="504" height="28" rx="5" fill="url(#bar2)" opacity="0.85"/>
  <g font-family="'Segoe UI',Arial,sans-serif" font-size="13" font-weight="700" fill="white">
    <text x="710" y="80">9 classes</text>
    <text x="710" y="126">6 transforms</text>
    <text x="710" y="172">8 BN layers</text>
    <text x="710" y="218">3 callbacks</text>
    <text x="710" y="264">Full report</text>
    <text x="710" y="310">Balanced</text>
  </g>
</svg>

</div>

---

## Architecture Diagram

<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 580" width="760" height="580">
  <defs>
    <linearGradient id="archBg" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%"   stop-color="#0a1628"/>
      <stop offset="100%" stop-color="#062a27"/>
    </linearGradient>
    <linearGradient id="headGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#2563EB"/>
      <stop offset="100%" stop-color="#10B981"/>
    </linearGradient>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#0B8F87"/>
    </marker>
  </defs>
  <rect width="760" height="580" fill="url(#archBg)" rx="14"/>
  <text x="380" y="36" font-family="'Segoe UI',Arial,sans-serif" font-size="16"
        font-weight="700" fill="#e2e8f0" text-anchor="middle" letter-spacing="1">
    CNN ARCHITECTURE  --  VGG-STYLE  (3 CONV BLOCKS)
  </text>
  <rect x="280" y="55" width="200" height="40" rx="8" fill="#1e3a5f" stroke="#2563EB" stroke-width="1.5"/>
  <text x="380" y="80" font-family="'Segoe UI',Arial,sans-serif" font-size="13"
        font-weight="600" fill="#93c5fd" text-anchor="middle">Input  224 x 224 x 3</text>
  <line x1="380" y1="95" x2="380" y2="115" stroke="#0B8F87" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="140" y="115" width="480" height="80" rx="10" fill="#0B8F87" opacity="0.18" stroke="#0B8F87" stroke-width="1.5"/>
  <text x="380" y="138" font-family="'Segoe UI',Arial,sans-serif" font-size="13"
        font-weight="700" fill="#5eead4" text-anchor="middle">Block 1</text>
  <text x="380" y="158" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#94a3b8" text-anchor="middle">Conv2D(32) -- BN -- ReLU -- Conv2D(32) -- BN -- ReLU</text>
  <text x="380" y="178" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#94a3b8" text-anchor="middle">MaxPooling2D(2x2)  --  Dropout(0.25)</text>
  <line x1="380" y1="195" x2="380" y2="215" stroke="#0B8F87" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="140" y="215" width="480" height="80" rx="10" fill="#2563EB" opacity="0.15" stroke="#2563EB" stroke-width="1.5"/>
  <text x="380" y="238" font-family="'Segoe UI',Arial,sans-serif" font-size="13"
        font-weight="700" fill="#93c5fd" text-anchor="middle">Block 2</text>
  <text x="380" y="258" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#94a3b8" text-anchor="middle">Conv2D(64) -- BN -- ReLU -- Conv2D(64) -- BN -- ReLU</text>
  <text x="380" y="278" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#94a3b8" text-anchor="middle">MaxPooling2D(2x2)  --  Dropout(0.25)</text>
  <line x1="380" y1="295" x2="380" y2="315" stroke="#0B8F87" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="140" y="315" width="480" height="80" rx="10" fill="#10B981" opacity="0.13" stroke="#10B981" stroke-width="1.5"/>
  <text x="380" y="338" font-family="'Segoe UI',Arial,sans-serif" font-size="13"
        font-weight="700" fill="#6ee7b7" text-anchor="middle">Block 3</text>
  <text x="380" y="358" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#94a3b8" text-anchor="middle">Conv2D(128) -- BN -- ReLU -- Conv2D(128) -- BN -- ReLU</text>
  <text x="380" y="378" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#94a3b8" text-anchor="middle">MaxPooling2D(2x2)  --  Dropout(0.25)</text>
  <line x1="380" y1="395" x2="380" y2="415" stroke="#0B8F87" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="300" y="415" width="160" height="36" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="1.2"/>
  <text x="380" y="438" font-family="'Segoe UI',Arial,sans-serif" font-size="12"
        fill="#cbd5e1" text-anchor="middle">Flatten</text>
  <line x1="380" y1="451" x2="380" y2="468" stroke="#0B8F87" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="200" y="468" width="360" height="60" rx="10" fill="none" stroke="url(#headGrad)" stroke-width="1.8"/>
  <text x="380" y="491" font-family="'Segoe UI',Arial,sans-serif" font-size="13"
        font-weight="700" fill="#e2e8f0" text-anchor="middle">Classification Head</text>
  <text x="380" y="511" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#94a3b8" text-anchor="middle">Dense(256) -- BN -- Dropout(0.5) -- Dense(9, softmax)</text>
  <line x1="380" y1="528" x2="380" y2="548" stroke="#10B981" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="260" y="548" width="240" height="20" rx="6" fill="#10B981" opacity="0.25"/>
  <text x="380" y="562" font-family="'Segoe UI',Arial,sans-serif" font-size="12"
        font-weight="600" fill="#6ee7b7" text-anchor="middle">9-Class Probability Vector</text>
</svg>

</div>

---

## Data Flow

<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 220" width="860" height="220">
  <defs>
    <linearGradient id="dfBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#0a1628"/>
      <stop offset="100%" stop-color="#062a27"/>
    </linearGradient>
    <marker id="dfarrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#0B8F87"/>
    </marker>
  </defs>
  <rect width="860" height="220" fill="url(#dfBg)" rx="14"/>
  <text x="430" y="28" font-family="'Segoe UI',Arial,sans-serif" font-size="14"
        font-weight="700" fill="#e2e8f0" text-anchor="middle" letter-spacing="1">
    DATA FLOW -- FROM DISK TO PREDICTION
  </text>
  <rect x="20"  y="55" width="120" height="60" rx="8" fill="#0B8F87" opacity="0.25" stroke="#0B8F87" stroke-width="1.5"/>
  <text x="80"  y="82"  font-family="'Segoe UI',Arial,sans-serif" font-size="11" font-weight="700" fill="#5eead4" text-anchor="middle">Dataset Dir</text>
  <text x="80"  y="98"  font-family="'Segoe UI',Arial,sans-serif" font-size="10" fill="#94a3b8" text-anchor="middle">9 class folders</text>
  <text x="80"  y="112" font-family="'Segoe UI',Arial,sans-serif" font-size="10" fill="#94a3b8" text-anchor="middle">2,357 images</text>
  <line x1="140" y1="85" x2="162" y2="85" stroke="#0B8F87" stroke-width="1.8" marker-end="url(#dfarrow)"/>
  <rect x="162" y="55" width="140" height="60" rx="8" fill="#2563EB" opacity="0.2" stroke="#2563EB" stroke-width="1.5"/>
  <text x="232" y="79"  font-family="'Segoe UI',Arial,sans-serif" font-size="11" font-weight="700" fill="#93c5fd" text-anchor="middle">ImageData</text>
  <text x="232" y="93"  font-family="'Segoe UI',Arial,sans-serif" font-size="11" font-weight="700" fill="#93c5fd" text-anchor="middle">Generator</text>
  <text x="232" y="109" font-family="'Segoe UI',Arial,sans-serif" font-size="10" fill="#94a3b8" text-anchor="middle">rescale + augment</text>
  <line x1="302" y1="85" x2="324" y2="85" stroke="#0B8F87" stroke-width="1.8" marker-end="url(#dfarrow)"/>
  <rect x="324" y="55" width="130" height="60" rx="8" fill="#0B8F87" opacity="0.2" stroke="#0B8F87" stroke-width="1.5"/>
  <text x="389" y="79"  font-family="'Segoe UI',Arial,sans-serif" font-size="11" font-weight="700" fill="#5eead4" text-anchor="middle">80 / 20 Split</text>
  <text x="389" y="95"  font-family="'Segoe UI',Arial,sans-serif" font-size="10" fill="#94a3b8" text-anchor="middle">train + val</text>
  <text x="389" y="109" font-family="'Segoe UI',Arial,sans-serif" font-size="10" fill="#94a3b8" text-anchor="middle">seed=42</text>
  <line x1="454" y1="85" x2="476" y2="85" stroke="#0B8F87" stroke-width="1.8" marker-end="url(#dfarrow)"/>
  <rect x="476" y="55" width="120" height="60" rx="8" fill="#10B981" opacity="0.2" stroke="#10B981" stroke-width="1.5"/>
  <text x="536" y="79"  font-family="'Segoe UI',Arial,sans-serif" font-size="11" font-weight="700" fill="#6ee7b7" text-anchor="middle">CNN Model</text>
  <text x="536" y="95"  font-family="'Segoe UI',Arial,sans-serif" font-size="10" fill="#94a3b8" text-anchor="middle">3 conv blocks</text>
  <text x="536" y="109" font-family="'Segoe UI',Arial,sans-serif" font-size="10" fill="#94a3b8" text-anchor="middle">224x224x3 in</text>
  <line x1="596" y1="85" x2="618" y2="85" stroke="#0B8F87" stroke-width="1.8" marker-end="url(#dfarrow)"/>
  <rect x="618" y="55" width="120" height="60" rx="8" fill="#2563EB" opacity="0.2" stroke="#2563EB" stroke-width="1.5"/>
  <text x="678" y="79"  font-family="'Segoe UI',Arial,sans-serif" font-size="11" font-weight="700" fill="#93c5fd" text-anchor="middle">Softmax</text>
  <text x="678" y="95"  font-family="'Segoe UI',Arial,sans-serif" font-size="10" fill="#94a3b8" text-anchor="middle">9-class probs</text>
  <text x="678" y="109" font-family="'Segoe UI',Arial,sans-serif" font-size="10" fill="#94a3b8" text-anchor="middle">argmax =&gt; label</text>
  <line x1="738" y1="85" x2="760" y2="85" stroke="#0B8F87" stroke-width="1.8" marker-end="url(#dfarrow)"/>
  <rect x="760" y="55" width="85" height="60" rx="8" fill="#10B981" opacity="0.3" stroke="#10B981" stroke-width="1.8"/>
  <text x="802" y="82"  font-family="'Segoe UI',Arial,sans-serif" font-size="11" font-weight="700" fill="#6ee7b7" text-anchor="middle">Prediction</text>
  <text x="802" y="98"  font-family="'Segoe UI',Arial,sans-serif" font-size="10" fill="#d1fae5" text-anchor="middle">class + conf</text>
  <path d="M 536,115 Q 536,165 380,165 Q 224,165 224,115"
        fill="none" stroke="#2563EB" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.7" marker-end="url(#dfarrow)"/>
  <text x="380" y="185" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#93c5fd" text-anchor="middle">EarlyStopping  |  ReduceLROnPlateau  |  ModelCheckpoint</text>
</svg>

</div>

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/Kaelith69/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2. Set up a virtual environment (optional but you will thank yourself later)

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

That's it. Seriously. No Docker, no Kubernetes, no 47-step setup wizard. Just Python and vibes.

### 4. Download the dataset

Grab images from the [ISIC Archive](https://www.isic-archive.com) and drop them into the matching class sub-folders:

```
dataset/
├── actinic_keratosis/
├── basal_cell_carcinoma/
├── dermatofibroma/
├── melanoma/
├── nevus/
├── pigmented_benign_keratosis/
├── seborrheic_keratosis/
├── squamous_cell_carcinoma/
└── vascular_lesion/
```

---

## Usage

### Training

```bash
python main.py
```

What happens under the hood:
1. Loads images from `dataset/` with augmentation on train, rescale-only on val
2. Computes per-class weights so the rare classes don't get bullied
3. Trains up to 50 epochs — EarlyStopping will intervene if loss stops improving (patience=7)
4. Saves the best model to `skin_cancer_model.keras`
5. Prints a full per-class classification report

### Evaluation output

```
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

Numbers are illustrative — actual results vary by dataset split and hardware.

### Inference on a single image

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('skin_cancer_model.keras')

CLASS_LABELS = [
    'actinic_keratosis', 'basal_cell_carcinoma', 'dermatofibroma',
    'melanoma', 'nevus', 'pigmented_benign_keratosis',
    'seborrheic_keratosis', 'squamous_cell_carcinoma', 'vascular_lesion',
]

img = image.load_img('path/to/image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction      = model.predict(img_array)
predicted_class = CLASS_LABELS[np.argmax(prediction)]
confidence      = np.max(prediction) * 100

print(f'Predicted class : {predicted_class}')
print(f'Confidence      : {confidence:.2f}%')
```

### Configuration

All the knobs you might want to turn are constants at the top of each file:

| File | Constant | Default | What it does |
|---|---|---|---|
| `data_preprocessing.py` | `IMG_SIZE` | `(224, 224)` | Resize all images to this |
| `data_preprocessing.py` | `BATCH_SIZE` | `32` | Mini-batch size |
| `data_preprocessing.py` | `VALIDATION_SPLIT` | `0.2` | 20% held out for validation |
| `data_preprocessing.py` | `SEED` | `42` | Reproducibility seed |
| `model.py` | `NUM_CLASSES` | `9` | Number of output classes |
| `main.py` | `EPOCHS` | `50` | Max training epochs |
| `main.py` | `MODEL_SAVE_PATH` | `skin_cancer_model.keras` | Where to save the best model |

---

## Project Structure

```
skin-cancer-detection/
|
+-- dataset/                            # Root dataset directory
|   +-- actinic_keratosis/              # Class folder -- pre-malignant
|   +-- basal_cell_carcinoma/           # Class folder -- malignant
|   +-- dermatofibroma/                 # Class folder -- benign
|   +-- melanoma/                       # Class folder -- malignant
|   +-- nevus/                          # Class folder -- benign
|   +-- pigmented_benign_keratosis/     # Class folder -- benign
|   +-- seborrheic_keratosis/           # Class folder -- benign
|   +-- squamous_cell_carcinoma/        # Class folder -- malignant
|   +-- vascular_lesion/                # Class folder -- benign
|
+-- assets/                             # Static assets (demo GIF, etc.)
|   +-- demo.gif                        # Place your demo GIF here
|
+-- docs/
|   +-- wiki/                           # Wiki source files
|       +-- Home.md
|       +-- Architecture.md
|       +-- Installation.md
|       +-- Usage.md
|       +-- Privacy.md
|       +-- Troubleshooting.md
|       +-- Roadmap.md
|
+-- data_preprocessing.py               # Train/val augmentation pipelines
+-- model.py                            # CNN architecture (VGG-style, 3 blocks)
+-- main.py                             # Training, evaluation & classification report
+-- requirements.txt                    # Python dependencies
+-- CONTRIBUTING.md
+-- CHANGELOG.md
+-- SECURITY.md
+-- LICENSE
+-- README.md
```

---

## Performance Stats

<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 200" width="760" height="200">
  <defs>
    <linearGradient id="statsBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0a1628"/>
      <stop offset="100%" stop-color="#062a27"/>
    </linearGradient>
    <linearGradient id="statsBar" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0B8F87"/>
      <stop offset="100%" stop-color="#10B981"/>
    </linearGradient>
  </defs>
  <rect width="760" height="200" fill="url(#statsBg)" rx="14"/>
  <text x="380" y="30" font-family="'Segoe UI',Arial,sans-serif" font-size="14"
        font-weight="700" fill="#e2e8f0" text-anchor="middle" letter-spacing="1">
    ILLUSTRATIVE PERFORMANCE STATS  (results vary by dataset + hardware)
  </text>
  <rect x="30"  y="50" width="160" height="120" rx="10" fill="#0B8F87" opacity="0.15" stroke="#0B8F87" stroke-width="1.5"/>
  <text x="110" y="95"  font-family="'Segoe UI',Arial,sans-serif" font-size="30"
        font-weight="800" fill="#5eead4" text-anchor="middle">~92%</text>
  <text x="110" y="118" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#94a3b8" text-anchor="middle">Val Accuracy</text>
  <text x="110" y="155" font-family="'Segoe UI',Arial,sans-serif" font-size="10"
        fill="#64748b" text-anchor="middle">over 9 classes</text>
  <rect x="210" y="50" width="160" height="120" rx="10" fill="#2563EB" opacity="0.15" stroke="#2563EB" stroke-width="1.5"/>
  <text x="290" y="95"  font-family="'Segoe UI',Arial,sans-serif" font-size="30"
        font-weight="800" fill="#93c5fd" text-anchor="middle">2,357</text>
  <text x="290" y="118" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#94a3b8" text-anchor="middle">Training Images</text>
  <text x="290" y="155" font-family="'Segoe UI',Arial,sans-serif" font-size="10"
        fill="#64748b" text-anchor="middle">ISIC archive</text>
  <rect x="390" y="50" width="160" height="120" rx="10" fill="#10B981" opacity="0.15" stroke="#10B981" stroke-width="1.5"/>
  <text x="470" y="95"  font-family="'Segoe UI',Arial,sans-serif" font-size="24"
        font-weight="800" fill="#6ee7b7" text-anchor="middle">224x224</text>
  <text x="470" y="118" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#94a3b8" text-anchor="middle">Input Resolution</text>
  <text x="470" y="155" font-family="'Segoe UI',Arial,sans-serif" font-size="10"
        fill="#64748b" text-anchor="middle">3-channel RGB</text>
  <rect x="570" y="50" width="160" height="120" rx="10" fill="#0B8F87" opacity="0.15" stroke="#0B8F87" stroke-width="1.5"/>
  <text x="650" y="95"  font-family="'Segoe UI',Arial,sans-serif" font-size="30"
        font-weight="800" fill="#5eead4" text-anchor="middle">50</text>
  <text x="650" y="118" font-family="'Segoe UI',Arial,sans-serif" font-size="11"
        fill="#94a3b8" text-anchor="middle">Max Epochs</text>
  <text x="650" y="155" font-family="'Segoe UI',Arial,sans-serif" font-size="10"
        fill="#64748b" text-anchor="middle">EarlyStopping p=7</text>
</svg>

</div>

---

## Privacy

**No data leaves your machine.** Full stop.

- All training and inference runs **100% locally**
- No telemetry, no analytics, no "anonymous usage data" (there isn't any)
- No network calls during training or prediction
- Model weights are saved to your local disk (`skin_cancer_model.keras`)
- No PII is collected or stored

The only outbound connection this project makes is the one *you* make to download the ISIC dataset. After that, everything happens offline.

> Medical images are sensitive. This project was designed to work entirely offline to respect that. If you build something on top of this, please maintain that boundary.

---

## Roadmap

Things that would be cool to add (PR welcome!):

- [ ] Transfer learning with EfficientNet or MobileNetV2 for better accuracy
- [ ] TensorBoard integration for live training visualisation
- [ ] Grad-CAM visualisation — show your work, neural network
- [ ] Unit tests for data preprocessing and model building
- [ ] Dockerfile for reproducible environment
- [ ] Simple CLI inference tool (`predict.py`) for single-image inference
- [ ] `setup.py` / `pyproject.toml` for proper packaging
- [ ] Confusion matrix output after evaluation

---

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, learn from it. Just don't pretend a Python script replaces a dermatologist.

---

<div align="center">

*Built with love and an amount of GPU heat that definitely affected the room temperature.*

---

**Why did the neural network go to therapy?**
*Because it had too many layers of issues and kept dropping out.*

MIT License (c) 2024 Kaelith69 (Sayanth T M)

</div>
