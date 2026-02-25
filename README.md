<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 260" width="960" height="260" role="img" aria-label="Skin Cancer Detection project banner">
  <defs>
    <linearGradient id="heroBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0b1320;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#0f1f33;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#062127;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="heroLine" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#22c55e;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0ea5e9;stop-opacity:1" />
    </linearGradient>
    <filter id="heroGlow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="960" height="260" rx="16" fill="url(#heroBg)" />
  <rect x="0" y="0" width="960" height="5" fill="url(#heroLine)" />
  <circle cx="840" cy="60" r="90" fill="#10b981" opacity="0.08" />
  <circle cx="130" cy="210" r="85" fill="#2563eb" opacity="0.08" />
  <g opacity="0.03" stroke="#0ea5e9" stroke-width="1" fill="none">
    <line x1="0" y1="65" x2="960" y2="65"/><line x1="0" y1="130" x2="960" y2="130"/>
    <line x1="0" y1="195" x2="960" y2="195"/><line x1="192" y1="0" x2="192" y2="260"/>
    <line x1="384" y1="0" x2="384" y2="260"/><line x1="576" y1="0" x2="576" y2="260"/>
    <line x1="768" y1="0" x2="768" y2="260"/>
  </g>
  <text x="480" y="108" text-anchor="middle" fill="#ffffff" font-family="'Segoe UI',Arial,sans-serif" font-size="46" font-weight="800" filter="url(#heroGlow)">Skin Cancer Detection</text>
  <text x="480" y="148" text-anchor="middle" fill="#93c5fd" font-family="'Segoe UI',Arial,sans-serif" font-size="18">TensorFlow/Keras · 9-Class Dermoscopy Classifier · Research/Education Only</text>
  <text x="480" y="186" text-anchor="middle" fill="#a7f3d0" font-family="'Segoe UI',Arial,sans-serif" font-size="15">VGG-style CNN baseline trained on ISIC class-folder datasets</text>
  <rect x="0" y="255" width="960" height="5" fill="url(#heroLine)" opacity="0.5"/>
</svg>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Research Only](https://img.shields.io/badge/Use-Research%20Only-ef4444?style=flat-square)](wiki/Privacy.md)
[![Classes](https://img.shields.io/badge/Classes-9%20Lesion%20Types-0ea5e9?style=flat-square)](wiki/Architecture.md)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC-7c3aed?style=flat-square)](https://www.isic-archive.com)

</div>

> ⚠️ **Medical Disclaimer:** This repository is for research and educational use only. It is **not** a certified medical device and must not be used as a substitute for professional dermatological diagnosis.

## Project Overview

This project trains a custom CNN to classify dermoscopic images into 9 lesion categories:

`actinic_keratosis`, `basal_cell_carcinoma`, `dermatofibroma`, `melanoma`, `nevus`, `pigmented_benign_keratosis`, `seborrheic_keratosis`, `squamous_cell_carcinoma`, and `vascular_lesion`.

Core modules:

| Module | Responsibility |
|---|---|
| `data_preprocessing.py` | Image loading, normalization, augmentation, 80/20 train/validation split |
| `model.py` | VGG-style CNN architecture definition and compilation |
| `main.py` | Class weighting, training loop, callbacks, evaluation, classification report |

## Quick Start

```bash
git clone https://github.com/Kaelith69/skin-cancer-detection.git
cd skin-cancer-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The trained checkpoint is saved to `skin_cancer_model.keras` via `ModelCheckpoint`.

## Architecture at a Glance

```
Input (224×224×3)
  ↓
[Block 1]  Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)   → 112×112×32
[Block 2]  Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)   →  56×56×64
[Block 3]  Conv2D(128)→ BN → Conv2D(128)→ BN → MaxPool → Dropout(0.25)   →  28×28×128
  ↓
Flatten → Dense(256) → BN → Dropout(0.5) → Dense(9, softmax)
```

See [Architecture](wiki/Architecture.md) for the full visual diagram and design rationale.

## Documentation Suite

Full project documentation lives in `/wiki`:

| Page | Contents |
|---|---|
| [Home](wiki/Home.md) | Project summary, class table, repo layout, quick-start |
| [Architecture](wiki/Architecture.md) | CNN layer diagram, data pipeline, design decisions |
| [Installation](wiki/Installation.md) | Environment setup, dependency install, dataset prep |
| [Usage](wiki/Usage.md) | Training, evaluation, inference, configuration reference |
| [Troubleshooting](wiki/Troubleshooting.md) | Common errors and fixes |
| [Contributing](wiki/Contributing.md) | Workflow, code style, PR guidelines |
| [Roadmap](wiki/Roadmap.md) | Planned features and long-term vision |
| [Privacy](wiki/Privacy.md) | Local-only inference, data handling, security model |

## License

Licensed under the [MIT License](LICENSE).
