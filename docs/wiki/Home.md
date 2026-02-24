# Skin Cancer Detection — Wiki Home

Welcome to the wiki. If you're here you're either trying to understand the project, debug something, or you accidentally clicked the wrong tab. All three are valid.

---

## What is this?

A Python project that trains a Convolutional Neural Network to classify dermoscopic skin images into 9 categories using TensorFlow/Keras. It's built for learning and research. It is not a medical device.

The full stack is deliberately minimal:

- **Python 3.9+**
- **TensorFlow 2.12+** (with Keras built in)
- **scikit-learn** for class-weight balancing and evaluation reports
- **NumPy, Pillow, matplotlib, seaborn, pandas** for data handling and visualisation

---

## Quick Navigation

| Page | What's in it |
|---|---|
| [Architecture](Architecture) | How the CNN is built, layer by layer |
| [Installation](Installation) | Getting the project running from zero |
| [Usage](Usage) | Training, evaluation, single-image inference |
| [Privacy](Privacy) | Where your data goes (spoiler: nowhere) |
| [Troubleshooting](Troubleshooting) | Common errors and how to fix them |
| [Roadmap](Roadmap) | Features we'd like to add |

---

## Project Files at a Glance

```
skin-cancer-detection/
|
+-- data_preprocessing.py    # Loads and augments images — training pipeline only
+-- model.py                 # Defines the CNN architecture
+-- main.py                  # Runs the full training loop + evaluation
+-- requirements.txt         # Python dependencies
+-- dataset/                 # 9 class sub-folders of dermoscopic images
```

That's really it. Three Python files and a dataset folder. The simplicity is intentional.

---

## The 9 Classes

| Class | Malignancy |
|---|---|
| Actinic Keratosis | Pre-malignant |
| Basal Cell Carcinoma | Malignant |
| Dermatofibroma | Benign |
| Melanoma | Malignant |
| Nevus | Benign |
| Pigmented Benign Keratosis | Benign |
| Seborrheic Keratosis | Benign |
| Squamous Cell Carcinoma | Malignant |
| Vascular Lesion | Benign |

---

## Dataset

Images come from the [ISIC Archive](https://www.isic-archive.com) — roughly 2,357 images total. You download them yourself and place them in the matching sub-folders under `dataset/`. The project doesn't ship with images for obvious reasons.

---

## The Short Version

If you just want to run it:

```bash
git clone https://github.com/Kaelith69/skin-cancer-detection.git
cd skin-cancer-detection
pip install -r requirements.txt
# add your dataset images to dataset/
python main.py
```

For everything else, pick a page from the navigation above.
