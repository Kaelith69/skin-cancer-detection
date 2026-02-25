<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 200" width="860" height="200">
  <defs>
    <linearGradient id="wikiHeroBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0a1628;stop-opacity:1"/>
      <stop offset="50%" style="stop-color:#0d2137;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#071e20;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="wikiTeal" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#0B8F87;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#10B981;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="wikiBlue" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#2563EB;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#0B8F87;stop-opacity:1"/>
    </linearGradient>
    <filter id="wikiGlow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="860" height="200" fill="url(#wikiHeroBg)" rx="14"/>
  <g opacity="0.04" stroke="#0B8F87" stroke-width="1" fill="none">
    <line x1="0" y1="40" x2="860" y2="40"/>
    <line x1="0" y1="80" x2="860" y2="80"/>
    <line x1="0" y1="120" x2="860" y2="120"/>
    <line x1="0" y1="160" x2="860" y2="160"/>
    <line x1="86" y1="0" x2="86" y2="200"/>
    <line x1="172" y1="0" x2="172" y2="200"/>
    <line x1="258" y1="0" x2="258" y2="200"/>
    <line x1="344" y1="0" x2="344" y2="200"/>
    <line x1="430" y1="0" x2="430" y2="200"/>
    <line x1="516" y1="0" x2="516" y2="200"/>
    <line x1="602" y1="0" x2="602" y2="200"/>
    <line x1="688" y1="0" x2="688" y2="200"/>
    <line x1="774" y1="0" x2="774" y2="200"/>
  </g>
  <circle cx="760" cy="40" r="80" fill="#0B8F87" opacity="0.06"/>
  <circle cx="100" cy="160" r="70" fill="#2563EB" opacity="0.06"/>
  <rect x="0" y="0" width="860" height="4" rx="0" fill="url(#wikiTeal)" opacity="0.9"/>
  <text x="430" y="88" font-family="'Segoe UI',system-ui,Arial,sans-serif" font-size="38"
        font-weight="800" fill="white" text-anchor="middle" filter="url(#wikiGlow)"
        letter-spacing="-0.5">Skin Cancer Detection — Wiki</text>
  <text x="430" y="124" font-family="'Segoe UI',system-ui,Arial,sans-serif" font-size="15"
        fill="#94a3b8" text-anchor="middle" letter-spacing="0.5">
    9-Class CNN · TensorFlow/Keras · ISIC Dataset · Research Use Only
  </text>
  <rect x="180" y="138" width="500" height="2" rx="1" fill="url(#wikiTeal)" opacity="0.6"/>
  <rect x="0" y="196" width="860" height="4" rx="0" fill="url(#wikiBlue)" opacity="0.5"/>
</svg>

</div>

---

Welcome to the official documentation wiki for the **Skin Cancer Detection** project. A Convolutional Neural Network that classifies dermoscopic images of skin lesions into 9 diagnostic categories using TensorFlow/Keras and the ISIC research dataset.

This model reads a skin image and outputs a probability distribution across 9 classes faster than a dermatologist can find parking. That said — **this is a research tool, not a clinical device**. Go see a real doctor.

> ⚠️ **Medical Disclaimer:** This project is for research and educational purposes only. It is not a certified medical device and must not replace professional dermatological diagnosis.

---

## Quick Links

| Page | What's inside |
|---|---|
| [Architecture](Architecture.md) | CNN design, layer breakdown, and why every decision was made |
| [Installation](Installation.md) | Environment setup, dependency installation, dataset preparation |
| [Usage](Usage.md) | Training, evaluation, inference, and configuration reference |
| [Privacy](Privacy.md) | No cloud. No spying. No villain origin story. |
| [Contributing](Contributing.md) | Development workflow, code style, and PR guidelines |
| [Troubleshooting](Troubleshooting.md) | Common errors and their solutions |
| [Roadmap](Roadmap.md) | Planned features and long-term vision |

---

## Project Summary

| Property | Value |
|---|---|
| **Language** | Python 3.9+ |
| **Framework** | TensorFlow 2.12+ / Keras |
| **Model type** | Custom VGG-style CNN |
| **Input size** | 224 × 224 × 3 |
| **Output classes** | 9 |
| **Dataset** | ISIC (~2,239 images) |
| **License** | MIT |

---

## The 9 Diagnostic Classes

| # | Class Name | Lesion Type |
|---|---|---|
| 0 | `actinic_keratosis` | Pre-malignant |
| 1 | `basal_cell_carcinoma` | Malignant |
| 2 | `dermatofibroma` | Benign |
| 3 | `melanoma` | Malignant |
| 4 | `nevus` | Benign |
| 5 | `pigmented_benign_keratosis` | Benign |
| 6 | `seborrheic_keratosis` | Benign |
| 7 | `squamous_cell_carcinoma` | Malignant |
| 8 | `vascular_lesion` | Benign |

---

## Repository Layout

```
skin-cancer-detection/
|-- data_preprocessing.py   # Data loading and augmentation — keep your pipelines honest
|-- model.py                # CNN architecture — VGG-style, built from scratch
|-- main.py                 # Training, evaluation, reporting — one file to rule them all
|-- requirements.txt        # Pinned dependencies
|-- dataset/                # Image folders (one per class, included in this repository)
|-- wiki/                   # You are here
`-- LICENSE
```

---

## Getting Started in 4 Steps

```bash
# 1. Clone
git clone https://github.com/Kaelith69/skin-cancer-detection.git
cd skin-cancer-detection

# 2. Virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train (go make coffee — GPU recommended)
python main.py
```

See [Installation](Installation.md) and [Usage](Usage.md) for full details.
