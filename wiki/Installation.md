<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 200" width="860" height="200" role="img" aria-label="Installation page header">
  <defs>
    <linearGradient id="instHeroBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0a1628;stop-opacity:1"/>
      <stop offset="50%" style="stop-color:#0f1a10;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#071e20;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="instAccent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#22c55e;stop-opacity:1"/>
    </linearGradient>
    <filter id="instGlow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="860" height="200" rx="14" fill="url(#instHeroBg)"/>
  <g opacity="0.04" stroke="#f59e0b" stroke-width="1" fill="none">
    <line x1="0" y1="40" x2="860" y2="40"/><line x1="0" y1="80" x2="860" y2="80"/>
    <line x1="0" y1="120" x2="860" y2="120"/><line x1="0" y1="160" x2="860" y2="160"/>
    <line x1="86" y1="0" x2="86" y2="200"/><line x1="172" y1="0" x2="172" y2="200"/>
    <line x1="258" y1="0" x2="258" y2="200"/><line x1="344" y1="0" x2="344" y2="200"/>
    <line x1="430" y1="0" x2="430" y2="200"/><line x1="516" y1="0" x2="516" y2="200"/>
    <line x1="602" y1="0" x2="602" y2="200"/><line x1="688" y1="0" x2="688" y2="200"/>
    <line x1="774" y1="0" x2="774" y2="200"/>
  </g>
  <circle cx="760" cy="40" r="80" fill="#f59e0b" opacity="0.07"/>
  <circle cx="100" cy="160" r="70" fill="#22c55e" opacity="0.07"/>
  <rect x="0" y="0" width="860" height="4" fill="url(#instAccent)" opacity="0.9"/>
  <text x="430" y="88" font-family="'Segoe UI',system-ui,Arial,sans-serif" font-size="36"
        font-weight="800" fill="white" text-anchor="middle" filter="url(#instGlow)">Installation</text>
  <text x="430" y="122" font-family="'Segoe UI',system-ui,Arial,sans-serif" font-size="15"
        fill="#fde68a" text-anchor="middle">Clone · Virtual Environment · pip install · Dataset Setup</text>
  <rect x="180" y="138" width="500" height="2" rx="1" fill="url(#instAccent)" opacity="0.6"/>
  <rect x="0" y="196" width="860" height="4" fill="url(#instAccent)" opacity="0.5"/>
</svg>

</div>

# Installation

Everything you need to get this running — from cloning the repo to having a trained model ready. Follow these steps and you'll spend exactly zero time debugging "why does it say module not found."

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.10 or 3.11 |
| RAM | 8 GB | 16 GB+ |
| GPU | None (CPU fallback, bring patience) | CUDA-capable NVIDIA GPU |
| CUDA | — | 11.x or 12.x (matching TF version) |
| Disk | 2 GB (code + deps) | 5 GB+ (code + deps + dataset) |

### GPU Support

TensorFlow 2.12 and later have GPU support built-in on Linux. On Windows, use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/) or ensure you have the correct CUDA/cuDNN versions. See [TensorFlow GPU guide](https://www.tensorflow.org/install/pip) for details.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/Kaelith69/skin-cancer-detection.git
cd skin-cancer-detection
```

---

## Step 2 — Create a Virtual Environment

Using a virtual environment isolates project dependencies from your system Python. This is not optional — it's what separates "it works on my machine" from "it works on everyone's machine."

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (Command Prompt)

```cmd
python -m venv venv
venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` prefixed on your terminal prompt. If you don't, re-read this section. The venv didn't activate itself.

---

## Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependency Overview

| Package | Version | Purpose |
|---|---|---|
| `tensorflow` | ≥2.12.0 | Core deep learning framework |
| `numpy` | ≥1.24.0 | Numerical array operations |
| `Pillow` | ≥9.5.0 | Image I/O (JPEG, PNG) |
| `matplotlib` | ≥3.7.0 | Training curve visualization |
| `seaborn` | ≥0.12.0 | Confusion matrix heatmaps |
| `scikit-learn` | ≥1.2.0 | Class weights, classification report |
| `pandas` | ≥2.0.0 | Tabular data/metrics handling |

### Verify Installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
# Should print: 2.12.x or later

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Should list GPU(s) if available
```

---

## Step 4 — Prepare the Dataset

### Download

1. Visit [https://www.isic-archive.com](https://www.isic-archive.com)
2. Register for a free research account
3. Download the ISIC 2019 or 2020 training dataset (~2,239 total images used in this project; after the 80/20 split, approximately ~1,792 are used for training and ~447 for validation)

### Organize

Place the images into class-named sub-folders inside the `dataset/` directory at the project root:

```
dataset/
|-- actinic_keratosis/
|   |-- ISIC_0024306.jpg
|   |-- ISIC_0024307.jpg
|   ...
|-- basal_cell_carcinoma/
|   ...
|-- dermatofibroma/
|-- melanoma/
|-- nevus/
|-- pigmented_benign_keratosis/
|-- seborrheic_keratosis/
|-- squamous_cell_carcinoma/
`-- vascular_lesion/
```

> **Important:** Each sub-folder name must exactly match the class names listed above. TensorFlow's `flow_from_directory` infers class labels from folder names and it is case-sensitive. A folder named `Melanoma` is not the same as `melanoma`. Yes, this has bitten people before.

### Verify Dataset Structure

```bash
python - << 'EOF'
import os
dataset_path = 'dataset'
classes = sorted(os.listdir(dataset_path))
for cls in classes:
    count = len(os.listdir(os.path.join(dataset_path, cls)))
    print(f"  {cls:<35} {count:>5} images")
EOF
```

Expected output (counts vary by dataset version):

```
  actinic_keratosis                    114 images
  basal_cell_carcinoma                 376 images
  dermatofibroma                        95 images
  melanoma                             438 images
  nevus                                357 images
  pigmented_benign_keratosis           462 images
  seborrheic_keratosis                 253 images
  squamous_cell_carcinoma              181 images
  vascular_lesion                       81 images
```

---

## Optional — Using conda Instead of venv

```bash
conda create -n skin-cancer python=3.10
conda activate skin-cancer
pip install -r requirements.txt
```

---

## Common Installation Issues

See [Troubleshooting](Troubleshooting.md) for solutions to common problems such as:

- `ModuleNotFoundError: No module named 'tensorflow'`
- CUDA/cuDNN version mismatch
- Pillow import failures on Apple Silicon
