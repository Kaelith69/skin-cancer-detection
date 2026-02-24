# Skin Cancer Detection — Project Wiki

Welcome to the official documentation wiki for the **Skin Cancer Detection** project. This repository implements a Convolutional Neural Network (CNN) for classifying dermoscopic images of skin lesions into 9 diagnostic categories, using TensorFlow/Keras and the ISIC research dataset.

---

## What This Project Does

The system ingests a dermoscopic JPEG/PNG image, preprocesses it (resize to 224×224, normalize to [0,1]), and outputs a probability distribution across 9 skin-lesion classes. The class with the highest probability is returned as the predicted diagnosis along with a confidence score.

> ⚠️ **Medical Disclaimer:** This project is for research and educational purposes only. It is not a certified medical device and must not replace professional dermatological diagnosis.

---

## Quick Links

| Page | Description |
|---|---|
| [Architecture](Architecture.md) | CNN design, layer breakdown, and component interaction |
| [Installation](Installation.md) | Environment setup, dependency installation, dataset preparation |
| [Usage](Usage.md) | Training, evaluation, inference, and configuration reference |
| [Privacy](Privacy.md) | Data handling, security model, and regulatory considerations |
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
| **Dataset** | ISIC (~2,357 images) |
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
|-- data_preprocessing.py   # Data loading and augmentation
|-- model.py                # CNN architecture definition
|-- main.py                 # Training, evaluation, reporting
|-- requirements.txt        # Pinned dependencies
|-- dataset/                # Image folders (one per class)
|-- wiki/                   # This documentation
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

# 4. Train
python main.py
```

See [Installation](Installation.md) and [Usage](Usage.md) for full details.
