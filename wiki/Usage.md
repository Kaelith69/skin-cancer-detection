<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 200" width="860" height="200" role="img" aria-label="Usage page header">
  <defs>
    <linearGradient id="usageHeroBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0a1628;stop-opacity:1"/>
      <stop offset="50%" style="stop-color:#071e20;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#082a2a;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="usageAccent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#0ea5e9;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#0b8f87;stop-opacity:1"/>
    </linearGradient>
    <filter id="usageGlow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="860" height="200" rx="14" fill="url(#usageHeroBg)"/>
  <g opacity="0.04" stroke="#0ea5e9" stroke-width="1" fill="none">
    <line x1="0" y1="40" x2="860" y2="40"/><line x1="0" y1="80" x2="860" y2="80"/>
    <line x1="0" y1="120" x2="860" y2="120"/><line x1="0" y1="160" x2="860" y2="160"/>
    <line x1="86" y1="0" x2="86" y2="200"/><line x1="172" y1="0" x2="172" y2="200"/>
    <line x1="258" y1="0" x2="258" y2="200"/><line x1="344" y1="0" x2="344" y2="200"/>
    <line x1="430" y1="0" x2="430" y2="200"/><line x1="516" y1="0" x2="516" y2="200"/>
    <line x1="602" y1="0" x2="602" y2="200"/><line x1="688" y1="0" x2="688" y2="200"/>
    <line x1="774" y1="0" x2="774" y2="200"/>
  </g>
  <circle cx="760" cy="40" r="80" fill="#0ea5e9" opacity="0.07"/>
  <circle cx="100" cy="160" r="70" fill="#0b8f87" opacity="0.07"/>
  <rect x="0" y="0" width="860" height="4" fill="url(#usageAccent)" opacity="0.9"/>
  <text x="430" y="88" font-family="'Segoe UI',system-ui,Arial,sans-serif" font-size="36"
        font-weight="800" fill="white" text-anchor="middle" filter="url(#usageGlow)">Usage</text>
  <text x="430" y="122" font-family="'Segoe UI',system-ui,Arial,sans-serif" font-size="15"
        fill="#bae6fd" text-anchor="middle">Training · Evaluation · Single-Image Inference · Configuration</text>
  <rect x="180" y="138" width="500" height="2" rx="1" fill="url(#usageAccent)" opacity="0.6"/>
  <rect x="0" y="196" width="860" height="4" fill="url(#usageAccent)" opacity="0.5"/>
</svg>

</div>

# Usage

All the runtime operations: training from scratch, evaluating, running inference on new images, and configuring the important knobs. One command gets you most of the way there.

---

## Training

### Run Training

With the dataset prepared and dependencies installed:

```bash
python main.py
```

That's the whole command. Go make coffee — the first epoch will take a while to warm up. On GPU it's tolerable. On CPU it's a commitment.

### What Happens Under the Hood

1. **Data loading** — `load_data('dataset/')` creates two `DirectoryIterator` objects: `train_data` (80%) and `validation_data` (20%)
2. **Class-weight computation** — `compute_class_weight('balanced', ...)` calculates per-class weights from `train_data.classes`
3. **Model build** — `build_model(num_classes=9)` constructs and compiles the VGG-style CNN
4. **Training loop** — `model.fit(...)` trains for up to 50 epochs with three callbacks
5. **Evaluation** — `model.evaluate(validation_data)` prints loss and accuracy
6. **Classification report** — `classification_report(y_true, y_pred, ...)` prints per-class precision, recall, and F1-score

### Expected Console Output

```
Found 1885 images belonging to 9 classes.
Found 472 images belonging to 9 classes.
Model: "sequential"
...
Epoch 1/50
59/59 [==============================] - 45s 754ms/step - loss: 1.8432 - accuracy: 0.3421 - val_loss: 1.6123 - val_accuracy: 0.4619
...
Epoch 00023: early stopping
Restoring model weights from the end of the best epoch: 16.

Validation Loss:     0.2341
Validation Accuracy: 91.87%

Classification Report:
...
```

---

## Evaluation Only

To re-evaluate a previously saved model without re-training:

```python
from tensorflow.keras.models import load_model
from data_preprocessing import load_data
import numpy as np
from sklearn.metrics import classification_report

model = load_model('skin_cancer_model.keras')
_, validation_data = load_data('dataset/')

loss, accuracy = model.evaluate(validation_data)
print(f'Validation Loss:     {loss:.4f}')
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

y_true = validation_data.classes
y_pred = np.argmax(model.predict(validation_data), axis=1)
class_labels = {v: k for k, v in validation_data.class_indices.items()}
target_names = [class_labels[i] for i in range(len(class_labels))]
print(classification_report(y_true, y_pred, target_names=target_names))
```

---

## Inference on a Single Image

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model('skin_cancer_model.keras')

CLASS_LABELS = [
    'actinic_keratosis', 'basal_cell_carcinoma', 'dermatofibroma',
    'melanoma', 'nevus', 'pigmented_benign_keratosis',
    'seborrheic_keratosis', 'squamous_cell_carcinoma', 'vascular_lesion',
]

def predict_image(img_path: str) -> dict:
    """Run inference on a single dermoscopic image.

    Args:
        img_path: Absolute or relative path to the image file.

    Returns:
        dict with 'class' (str) and 'confidence' (float, 0-100).
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # add batch dimension

    probs = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        'class': CLASS_LABELS[idx],
        'confidence': float(probs[idx]) * 100,
        'probabilities': dict(zip(CLASS_LABELS, probs.tolist())),
    }

result = predict_image('path/to/lesion.jpg')
print(f"Predicted: {result['class']}  ({result['confidence']:.2f}%)")
```

---

## Batch Inference

```python
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model('skin_cancer_model.keras')

CLASS_LABELS = [
    'actinic_keratosis', 'basal_cell_carcinoma', 'dermatofibroma',
    'melanoma', 'nevus', 'pigmented_benign_keratosis',
    'seborrheic_keratosis', 'squamous_cell_carcinoma', 'vascular_lesion',
]

def load_batch(image_paths, target_size=(224, 224)):
    batch = []
    for p in image_paths:
        img = load_img(p, target_size=target_size)
        batch.append(img_to_array(img) / 255.0)
    return np.array(batch)

image_files = ['img1.jpg', 'img2.jpg', 'img3.jpg']
batch = load_batch(image_files)
predictions = model.predict(batch)

for path, probs in zip(image_files, predictions):
    cls = CLASS_LABELS[np.argmax(probs)]
    conf = np.max(probs) * 100
    print(f'{os.path.basename(path):30s}  ->  {cls}  ({conf:.1f}%)')
```

---

## Configuration Reference

All constants are defined at the top of their respective modules. Change them there — don't hardcode values in the middle of functions and wonder why things break.

### `data_preprocessing.py`

| Constant | Default | Description |
|---|---|---|
| `IMG_SIZE` | `(224, 224)` | Target image dimensions |
| `BATCH_SIZE` | `32` | Mini-batch size for training |
| `VALIDATION_SPLIT` | `0.2` | Fraction of data used for validation |
| `SEED` | `42` | Random seed for reproducibility |

### `model.py`

| Constant | Default | Description |
|---|---|---|
| `NUM_CLASSES` | `9` | Number of output classes |
| `IMG_SHAPE` | `(224, 224, 3)` | Default input tensor shape |

### `main.py`

| Constant | Default | Description |
|---|---|---|
| `DATASET_PATH` | `'dataset/'` | Root path for image folders |
| `MODEL_SAVE_PATH` | `'skin_cancer_model.keras'` | Where to save the best model |
| `EPOCHS` | `50` | Maximum training epochs |

---

## Customization Examples

### Change Batch Size

In `data_preprocessing.py`:

```python
BATCH_SIZE = 16   # reduce if GPU OOM errors occur
```

### Add More Augmentation

In `data_preprocessing.py`, add to `train_datagen`:

```python
brightness_range=[0.8, 1.2],
channel_shift_range=20,
```

### Freeze and Unfreeze Layers

After loading a checkpoint for fine-tuning:

```python
model = load_model('skin_cancer_model.keras')
for layer in model.layers[:8]:   # freeze first 8 layers
    layer.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Change Optimizer or Learning Rate

In `model.py`:

```python
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
```
