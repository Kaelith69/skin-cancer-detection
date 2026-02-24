# Installation

Getting this project running is refreshingly straightforward. No Docker, no environment variables with 30 values, no enterprise license. Just Python.

---

## Requirements

- Python 3.9 or higher
- pip
- A machine with enough RAM/VRAM to train a CNN (a modern CPU works, a GPU makes it faster)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/Kaelith69/skin-cancer-detection.git
cd skin-cancer-detection
```

---

## Step 2: Create a Virtual Environment (Recommended)

Keeps your system Python clean. You'll thank yourself when dependency hell arrives for someone else's project.

```bash
python -m venv venv
```

Activate it:

**Linux / macOS:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

You should see `(venv)` prepended to your shell prompt.

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Version | Role |
|---|---|---|
| `tensorflow` | >=2.12.0 | CNN training + inference |
| `numpy` | >=1.24.0 | Array operations |
| `Pillow` | >=9.5.0 | Image loading and preprocessing |
| `matplotlib` | >=3.7.0 | Plotting (optional for visualisation) |
| `seaborn` | >=0.12.0 | Statistical plots (optional) |
| `scikit-learn` | >=1.2.0 | Class weights + classification report |
| `pandas` | >=2.0.0 | Data manipulation utilities |

Installation takes a few minutes — TensorFlow is large. Get a coffee.

---

## Step 4: Prepare the Dataset

Download dermoscopic images from the [ISIC Archive](https://www.isic-archive.com) and organize them into sub-folders by class:

```
dataset/
+-- actinic_keratosis/
+-- basal_cell_carcinoma/
+-- dermatofibroma/
+-- melanoma/
+-- nevus/
+-- pigmented_benign_keratosis/
+-- seborrheic_keratosis/
+-- squamous_cell_carcinoma/
+-- vascular_lesion/
```

**The folder names must match exactly.** Keras `flow_from_directory` uses the folder names as class labels.

The dataset is ~2,357 images across 9 classes. The project uses an 80/20 train/val split defined by `VALIDATION_SPLIT = 0.2` in `data_preprocessing.py`.

---

## Step 5: Verify Everything Works

Run a quick sanity check before committing to a full training run:

```python
from data_preprocessing import load_data
train_data, val_data = load_data('dataset/')
print(f"Training batches: {len(train_data)}")
print(f"Validation batches: {len(val_data)}")
print(f"Classes: {train_data.class_indices}")
```

If this prints without error, you're good to go.

---

## GPU Support

TensorFlow will automatically use a CUDA-compatible GPU if one is available and the CUDA/cuDNN dependencies are installed. Check the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu) for your platform.

Training on CPU works but is significantly slower for 50 epochs on 2,357 images.

---

## Common Installation Issues

See the [Troubleshooting](Troubleshooting) page for common problems and their fixes.
