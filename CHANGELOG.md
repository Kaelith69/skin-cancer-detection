# Changelog

All notable changes to this project will be documented here.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> "A changelog is a file which contains a curated, chronologically ordered list of notable changes for each version of a project." — also: proof that things have actually changed since you last looked.

---

## [Unreleased]

### Planned
- Transfer learning backbone (EfficientNet / MobileNetV2)
- TensorBoard integration
- Grad-CAM saliency visualisation
- `predict.py` CLI inference script
- Unit tests for preprocessing and model builder
- Dockerfile

---

## [1.1.0] — 2024-12-01

### Added
- Complete GitHub documentation suite: README, CONTRIBUTING, CHANGELOG, SECURITY, WIKI
- SVG hero banner, architecture diagram, data flow diagram, capability chart, and stats panel in README
- `assets/` directory with demo GIF placeholder
- `docs/wiki/` directory with seven wiki pages (Home, Architecture, Installation, Usage, Privacy, Troubleshooting, Roadmap)

### Changed
- README completely rewritten — now follows mandatory structure with all required SVG sections
- LICENSE year confirmed as 2024 (Sayanth T M)

---

## [1.0.0] — 2024-10-15

### Added
- `model.py` — VGG-style CNN with three convolutional blocks
  - Block 1: Conv2D(32) × 2, BN, MaxPool, Dropout(0.25)
  - Block 2: Conv2D(64) × 2, BN, MaxPool, Dropout(0.25)
  - Block 3: Conv2D(128) × 2, BN, MaxPool, Dropout(0.25)
  - Classification head: Dense(256), BN, Dropout(0.5), Dense(9, softmax)
  - Optimizer: Adam | Loss: Categorical Cross-Entropy
- `data_preprocessing.py` — separate augmentation pipelines for training and validation
  - Training: rescale, horizontal_flip, rotation_range=20, zoom_range=0.2, shift/shear
  - Validation: rescale only (no data leakage — we were raised right)
  - 80/20 train/val split, seed=42
- `main.py` — full training loop with:
  - Class-weight balancing via `sklearn.utils.class_weight.compute_class_weight`
  - EarlyStopping (monitor=val_loss, patience=7, restore_best_weights=True)
  - ReduceLROnPlateau (monitor=val_loss, factor=0.5, patience=3, min_lr=1e-6)
  - ModelCheckpoint (monitor=val_accuracy, save_best_only=True)
  - Full sklearn classification report (precision, recall, F1, support per class)
- `requirements.txt` — pinned dependencies: tensorflow, numpy, Pillow, matplotlib, seaborn, scikit-learn, pandas
- `dataset/` directory structure with 9 class sub-folders matching ISIC categories:
  - actinic_keratosis, basal_cell_carcinoma, dermatofibroma, melanoma, nevus,
    pigmented_benign_keratosis, seborrheic_keratosis, squamous_cell_carcinoma, vascular_lesion
- Initial `README.md` with SVG banner and project overview
- MIT License

---

## Legend

- **Added** — new features or files
- **Changed** — changes to existing functionality
- **Deprecated** — features that will be removed in a future version
- **Removed** — features that were removed
- **Fixed** — bug fixes
- **Security** — vulnerability patches
