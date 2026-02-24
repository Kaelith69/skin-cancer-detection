# Troubleshooting

Things go wrong. Here's a field guide to the most common disasters and how to survive them.

---

## Installation Issues

### `pip install -r requirements.txt` fails

**Symptom:** Errors during TensorFlow install, often involving wheel compatibility or missing build tools.

**Fix:**
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you're on Windows and getting MSVC errors, you may need [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

### `ImportError: No module named 'tensorflow'`

**Symptom:** Python can't find TensorFlow even after installing.

**Fix:** Make sure your virtual environment is activated:
```bash
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
python -c "import tensorflow; print(tensorflow.__version__)"
```

If TensorFlow imports fine but your script can't find it, you're running the wrong Python interpreter.

---

## Dataset Issues

### `FileNotFoundError: [Errno 2] No such file or directory: 'dataset/'`

**Symptom:** Training fails immediately with a path error.

**Fix:** Make sure the `dataset/` directory exists and contains at least one class sub-folder with images:
```
dataset/
+-- melanoma/
|   +-- image1.jpg
|   +-- image2.jpg
```

The script expects `DATASET_PATH = 'dataset/'` relative to where you run it. Run from the project root.

---

### `Found 0 images belonging to 0 classes.`

**Symptom:** `flow_from_directory` finds nothing.

**Causes and fixes:**
1. **Wrong path** — double-check `DATASET_PATH`
2. **Wrong image format** — Keras `flow_from_directory` supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.ppm`, `.tif`, `.tiff` by default. If your files have other extensions, they'll be silently ignored.
3. **Empty folders** — sub-folders exist but have no images in them
4. **Nested folders** — images are in `dataset/melanoma/batch1/img.jpg` instead of `dataset/melanoma/img.jpg`

---

### `ValueError: Found input variables with inconsistent numbers of samples`

**Symptom:** Crash when computing class weights.

**Fix:** Usually caused by an empty class folder. Remove or populate all sub-folders under `dataset/`. Every class folder listed must have at least one image.

---

## Training Issues

### Training is very slow

**Symptom:** Each epoch takes many minutes on CPU.

**Fix:**
- If you have a CUDA GPU, install the GPU version of TensorFlow and CUDA/cuDNN: see [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu)
- Reduce `BATCH_SIZE` if you're running out of memory
- Reduce `EPOCHS` for a quick test run

---

### Validation accuracy stays near chance (~11% for 9 classes)

**Symptom:** The model isn't learning anything useful.

**Possible causes:**
- Images aren't being rescaled (check `rescale=1.0/255` is set in the generator)
- Class folder names don't match expected labels — check `train_data.class_indices` output
- Dataset is too small or severely imbalanced — add more data or increase class weight influence
- Learning rate is too high — ReduceLROnPlateau will help, but a very bad start can take a while to recover from

---

### `OOM` (Out of Memory) error

**Symptom:** Process is killed or TensorFlow raises a ResourceExhaustedError.

**Fix:**
```python
# In data_preprocessing.py, reduce batch size:
BATCH_SIZE = 16  # or even 8
```

Also ensure no other GPU-heavy processes are running.

---

### EarlyStopping fires immediately or too soon

**Symptom:** Training stops after just a few epochs.

**Fix:** `patience=7` means it will wait 7 epochs without val_loss improvement before stopping. If it stops very early, your val_loss might be spiking, which usually means:
- Learning rate is too high initially (ReduceLROnPlateau will compensate over time)
- Validation set is too small or unrepresentative

---

## Inference Issues

### Model predicts the same class for everything

**Symptom:** Every image gets the same label with high confidence.

**Fix:** Almost always a preprocessing mismatch. Your inference preprocessing MUST match training:
```python
# This is correct:
img_array = image.img_to_array(img) / 255.0

# This is wrong (forgot to rescale):
img_array = image.img_to_array(img)
```

---

### `ValueError: Input 0 of layer is incompatible with the layer`

**Symptom:** Shape mismatch when calling `model.predict`.

**Fix:** Make sure input shape is `(1, 224, 224, 3)` — batch dimension + H + W + channels:
```python
img_array = np.expand_dims(img_array, axis=0)  # adds the batch dim
```

---

### `OSError: SavedModel file does not exist`

**Symptom:** Can't load `skin_cancer_model.keras`.

**Fix:** The model file is created during training. If you haven't trained yet, there's nothing to load. Run `python main.py` first.

---

## Still Stuck?

Open an issue on GitHub with:
1. The full error traceback
2. Your Python version (`python --version`)
3. Your TensorFlow version (`python -c "import tensorflow; print(tensorflow.__version__)"`)
4. Your OS
5. What you were doing when the error occurred

The more context, the faster the fix.
