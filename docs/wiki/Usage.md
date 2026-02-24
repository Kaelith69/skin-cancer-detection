# Usage

Everything you need to run this project is in three files: `data_preprocessing.py`, `model.py`, and `main.py`. Here's what each one does and how to use them.

---

## Training the Model

```bash
python main.py
```

That's it. One command. The script will:

1. **Load and split the dataset** — 80% training, 20% validation, `seed=42`
2. **Apply augmentation** — training data gets flipped, rotated, zoomed, shifted, and sheared. Validation data gets rescaled only.
3. **Compute class weights** — `sklearn` calculates per-class weights to handle the imbalanced ISIC dataset
4. **Build the CNN** — VGG-style 3-block model with the classification head
5. **Run up to 50 epochs** with three callbacks monitoring training
6. **Save the best checkpoint** to `skin_cancer_model.keras`
7. **Print a full classification report** after training completes

### What training output looks like

```
Epoch 1/50
59/59 [==============================] - 45s 734ms/step
  - loss: 2.1803 - accuracy: 0.2341
  - val_loss: 1.9215 - val_accuracy: 0.3012
Epoch 2/50
...
Epoch 23/50
Epoch 00023: val_accuracy improved from 0.9134 to 0.9187, saving model to skin_cancer_model.keras

Epoch 00030: ReduceLROnPlateau reducing learning rate to 0.0005.

Epoch 00034: early stopping
Restoring model weights from the end of the best epoch: 23.
```

The EarlyStopping callback (patience=7) will usually stop training well before epoch 50 once the model converges.

---

## Evaluation

Evaluation runs automatically at the end of `main.py`. Output looks like:

```
Validation Loss:     0.2341
Validation Accuracy: 91.87%

Classification Report:

                           precision  recall  f1-score  support
         actinic_keratosis     0.89    0.91      0.90       45
      basal_cell_carcinoma     0.94    0.93      0.93       71
           dermatofibroma      0.88    0.87      0.87       19
                 melanoma      0.92    0.90      0.91      114
                    nevus      0.93    0.95      0.94      163
pigmented_benign_keratosis     0.91    0.92      0.91      111
    seborrheic_keratosis       0.88    0.86      0.87       50
 squamous_cell_carcinoma       0.86    0.88      0.87       23
          vascular_lesion       0.91    0.89      0.90       19
```

Numbers are illustrative. Your actual results will vary based on dataset size, split randomness, hardware, and how many times you've muttered "just one more epoch."

---

## Inference on a Single Image

There's no dedicated inference script yet (it's on the roadmap), but you can do it directly:

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('skin_cancer_model.keras')

CLASS_LABELS = [
    'actinic_keratosis',
    'basal_cell_carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented_benign_keratosis',
    'seborrheic_keratosis',
    'squamous_cell_carcinoma',
    'vascular_lesion',
]

img = image.load_img('path/to/your/image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension

prediction      = model.predict(img_array)
predicted_class = CLASS_LABELS[np.argmax(prediction)]
confidence      = np.max(prediction) * 100

print(f'Predicted class : {predicted_class}')
print(f'Confidence      : {confidence:.2f}%')
```

**Important:** The image MUST be preprocessed identically to training data:
- Resized to 224x224
- Pixel values rescaled to [0, 1] (divide by 255)

If preprocessing differs from training, predictions will be garbage. The model will confidently be wrong, which is the worst kind of wrong.

---

## Using the Model Builder Directly

```python
from model import build_model

model = build_model(num_classes=9, input_shape=(224, 224, 3))
model.summary()
```

You can override `num_classes` if you're using a different dataset with a different number of classes.

---

## Using the Data Loader Directly

```python
from data_preprocessing import load_data

train_data, val_data = load_data('dataset/')

print(f"Classes: {train_data.class_indices}")
print(f"Training samples: {train_data.n}")
print(f"Validation samples: {val_data.n}")
```

---

## Configuration Knobs

All key constants are defined at the top of their respective modules. Change them there:

| File | Constant | Default | Effect |
|---|---|---|---|
| `data_preprocessing.py` | `IMG_SIZE` | `(224, 224)` | Input image dimensions |
| `data_preprocessing.py` | `BATCH_SIZE` | `32` | Training batch size |
| `data_preprocessing.py` | `VALIDATION_SPLIT` | `0.2` | Validation fraction |
| `data_preprocessing.py` | `SEED` | `42` | Reproducibility |
| `model.py` | `NUM_CLASSES` | `9` | Output classes |
| `main.py` | `EPOCHS` | `50` | Max training epochs |
| `main.py` | `DATASET_PATH` | `'dataset/'` | Path to image directory |
| `main.py` | `MODEL_SAVE_PATH` | `'skin_cancer_model.keras'` | Checkpoint save path |
