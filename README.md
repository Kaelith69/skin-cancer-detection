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
  </defs>
  <rect width="960" height="260" rx="16" fill="url(#heroBg)" />
  <rect x="0" y="0" width="960" height="5" fill="url(#heroLine)" />
  <circle cx="840" cy="60" r="90" fill="#10b981" opacity="0.08" />
  <circle cx="130" cy="210" r="85" fill="#2563eb" opacity="0.08" />
  <text x="480" y="108" text-anchor="middle" fill="#ffffff" font-family="'Segoe UI',Arial,sans-serif" font-size="46" font-weight="800">Skin Cancer Detection</text>
  <text x="480" y="145" text-anchor="middle" fill="#93c5fd" font-family="'Segoe UI',Arial,sans-serif" font-size="18">TensorFlow/Keras - 9-Class Dermoscopy Classifier - Research/Education Only</text>
  <text x="480" y="186" text-anchor="middle" fill="#a7f3d0" font-family="'Segoe UI',Arial,sans-serif" font-size="15">VGG-style CNN baseline trained on ISIC class-folder datasets</text>
</svg>

</div>

> **WARNING (medical disclaimer):** This repository is for research and educational use only. It is **not** a certified medical device and must not be used as a substitute for professional diagnosis.

## Project Overview

This project trains a custom CNN to classify dermoscopic images into 9 lesion categories:

`actinic_keratosis`, `basal_cell_carcinoma`, `dermatofibroma`, `melanoma`, `nevus`, `pigmented_benign_keratosis`, `seborrheic_keratosis`, `squamous_cell_carcinoma`, and `vascular_lesion`.

Core modules:

- `data_preprocessing.py` - image loading, normalization, augmentation, train/validation split
- `model.py` - VGG-style CNN architecture and compilation
- `main.py` - class weighting, training loop, callbacks, evaluation, classification report

## Quick Start

```bash
git clone https://github.com/Kaelith69/skin-cancer-detection.git
cd skin-cancer-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The trained checkpoint is saved to `skin_cancer_model.keras` using `ModelCheckpoint`.

## Documentation Suite

Full project documentation lives in `/wiki`:

- [Home](wiki/Home.md)
- [Architecture](wiki/Architecture.md)
- [Installation](wiki/Installation.md)
- [Usage](wiki/Usage.md)
- [Troubleshooting](wiki/Troubleshooting.md)
- [Contributing](wiki/Contributing.md)
- [Roadmap](wiki/Roadmap.md)
- [Privacy](wiki/Privacy.md)

## License

Licensed under the [MIT License](LICENSE).
