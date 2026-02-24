# Troubleshooting

This page covers the most common errors encountered when setting up or running the Skin Cancer Detection project, along with their solutions.

---

## Installation Issues

### `ModuleNotFoundError: No module named 'tensorflow'`

**Cause:** The virtual environment is not activated, or `requirements.txt` was not installed into the correct environment.

**Fix:**

```bash
source venv/bin/activate           # Linux/macOS
# or
venv\Scripts\activate              # Windows

pip install -r requirements.txt
python -c "import tensorflow; print('OK')"
```

---

### TensorFlow install fails on Apple Silicon (M1/M2/M3)

**Cause:** Standard TensorFlow wheels do not support Apple Silicon; the Apple-optimized `tensorflow-macos` package is required.

**Fix:**

```bash
pip install tensorflow-macos
pip install tensorflow-metal       # optional: GPU acceleration via Metal
```

For specific version compatibility, see the [Apple ML packages page](https://developer.apple.com/metal/tensorflow-plugin/). Check Apple's official documentation for the most current TensorFlow and Metal plugin version compatibility, as they evolve independently.

---

### `ImportError: libGL.so.1: cannot open shared object file`

**Cause:** Pillow's image display features require OpenGL libraries, which may be absent on headless servers.

**Fix (Debian/Ubuntu):**

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

**Alternative:** Install headless OpenCV if you added it as a dependency:

```bash
pip install opencv-python-headless
```

---

## Dataset Issues

### `Found 0 images belonging to 0 classes`

**Cause:** The `dataset/` directory is empty, does not exist, or the sub-folder names do not match what TensorFlow expects.

**Diagnosis:**

```bash
ls dataset/
# Should show: actinic_keratosis  basal_cell_carcinoma  dermatofibroma ...
```

**Fix:**
- Ensure `dataset/` exists at the project root
- Verify sub-folder names match exactly (case-sensitive on Linux): `actinic_keratosis`, `basal_cell_carcinoma`, etc.
- Confirm image files are inside sub-folders, not directly inside `dataset/`

---

### `FileNotFoundError: [Errno 2] No such file or directory: 'dataset/'`

**Cause:** `main.py` is being run from a different working directory.

**Fix:**

```bash
cd /path/to/skin-cancer-detection
python main.py
```

Or pass an absolute path in `main.py`:

```python
DATASET_PATH = '/absolute/path/to/skin-cancer-detection/dataset/'
```

---

### Images are loaded but accuracy is unexpectedly low

**Possible causes and fixes:**

| Cause | Fix |
|---|---|
| Augmentation applied to both train and val | Use two separate `ImageDataGenerator` instances (already done) |
| Wrong `seed` value | Ensure `seed=42` in both `flow_from_directory` calls |
| Class folders contain wrong images | Manually inspect a few images per class |
| Dataset too small | Download the full ISIC dataset |

---

## Training Issues

### `ResourceExhaustedError: OOM when allocating tensor`

**Cause:** GPU out-of-memory. Most common with large batch sizes or high-resolution images.

**Fix:**

```python
# In data_preprocessing.py, reduce BATCH_SIZE:
BATCH_SIZE = 16   # or 8
```

Or limit GPU memory growth:

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

### Training is very slow (CPU only)

**Cause:** TensorFlow is not detecting the GPU.

**Diagnosis:**

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))   # should not be empty
```

**Fix:**
1. Install the CUDA toolkit and cuDNN matching your TensorFlow version
2. On Linux: verify `nvidia-smi` works in terminal
3. On Windows: ensure CUDA PATH is set correctly
4. Consult [TensorFlow GPU installation guide](https://www.tensorflow.org/install/pip)

---

### `ValueError: Found input variables with inconsistent numbers of samples`

**Cause:** `y_train` and `class_weights` have different lengths, or class index mapping is wrong.

**Fix:** Ensure `class_weight_dict` is built from the same generator used for training:

```python
class_indices = train_data.class_indices
y_train = train_data.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train,
)
class_weight_dict = dict(enumerate(class_weights))
```

---

### Model training stops immediately (0 epochs)

**Cause:** The model's initial validation loss is already better than EarlyStopping expects, or `validation_data` returns empty batches.

**Fix:**
- Verify dataset is non-empty (see dataset issues above)
- Check `EarlyStopping(patience=7)` — a patience of 7 should be sufficient for any normal dataset

---

## Inference Issues

### `ValueError: Input 0 of layer is incompatible... expected shape (None, 224, 224, 3)`

**Cause:** The image was not resized to 224×224 before prediction.

**Fix:**

```python
img = image.load_img(img_path, target_size=(224, 224))   # must specify target_size
```

---

### Low confidence on all classes

**Cause:** The image was not normalized to [0, 1].

**Fix:**

```python
img_array = image.img_to_array(img) / 255.0   # divide by 255
```

---

### `OSError: SavedModel file does not exist`

**Cause:** `skin_cancer_model.keras` has not been created yet (training was never completed or was interrupted before ModelCheckpoint triggered).

**Fix:** Run `python main.py` to completion. The checkpoint is saved as soon as the first epoch completes.

---

## Getting Help

If your issue is not listed here:

1. Search existing [GitHub Issues](https://github.com/Kaelith69/skin-cancer-detection/issues)
2. Open a new issue with:
   - Python version, TensorFlow version, OS
   - Full error traceback
   - Minimal reproducible example
