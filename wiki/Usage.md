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
