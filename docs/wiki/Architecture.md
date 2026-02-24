# Architecture

The model is a custom VGG-style CNN implemented in `model.py`. It doesn't use a pretrained backbone — it learns entirely from scratch on the ISIC dataset. Transfer learning is on the roadmap, but this was built to be understandable first.

---

## Overall Structure

```
Input (224 x 224 x 3)
        |
   [ Block 1 ]   Conv2D(32) -- BN -- ReLU -- Conv2D(32) -- BN -- ReLU
                 MaxPooling2D(2x2) -- Dropout(0.25)
        |
   [ Block 2 ]   Conv2D(64) -- BN -- ReLU -- Conv2D(64) -- BN -- ReLU
                 MaxPooling2D(2x2) -- Dropout(0.25)
        |
   [ Block 3 ]   Conv2D(128) -- BN -- ReLU -- Conv2D(128) -- BN -- ReLU
                 MaxPooling2D(2x2) -- Dropout(0.25)
        |
      Flatten
        |
   Dense(256) -- BatchNormalization -- Dropout(0.5)
        |
   Dense(9, softmax)
        |
   9-class probability vector
```

---

## Why VGG-style?

VGG-style architectures (repeated Conv pairs followed by pooling) are well understood, easy to debug, and produce solid baselines for image classification. They're not state of the art, but they're a great starting point and easy to reason about.

The pattern is: extract features with two Conv layers, then downsample with MaxPool. Repeat. Each block doubles the filter count (32 → 64 → 128), allowing the network to learn increasingly abstract representations.

---

## Layer-by-Layer Breakdown

### Convolutional Blocks (x3)

Each block follows the same pattern:

```python
Conv2D(filters, (3, 3), padding='same', activation='relu')
BatchNormalization()
Conv2D(filters, (3, 3), padding='same', activation='relu')
BatchNormalization()
MaxPooling2D(pool_size=(2, 2))
Dropout(0.25)
```

Key decisions:
- `padding='same'` — spatial dimensions are preserved through Conv2D, only MaxPool reduces them. This avoids shrinking the feature map too aggressively.
- Two Conv layers per block — richer feature extraction before downsampling.
- BatchNormalization after every Conv — normalises activations, speeds up training, acts as a mild regulariser.
- Dropout(0.25) after each pool — prevents early co-adaptation of features.

### Classification Head

```python
Flatten()
Dense(256, activation='relu')
BatchNormalization()
Dropout(0.5)
Dense(num_classes, activation='softmax')
```

- A single hidden Dense(256) layer bridges the convolutional features and the output.
- Dropout(0.5) here is higher — this is where overfitting pressure is greatest.
- `softmax` output gives a proper probability distribution across all 9 classes.

---

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam (default lr=0.001) |
| Loss | Categorical Cross-Entropy |
| Metric | Accuracy |
| Max epochs | 50 |
| Batch size | 32 |
| Input size | 224 x 224 x 3 |

---

## Callbacks

Three callbacks run during training:

**EarlyStopping**
- Monitors: `val_loss`
- Patience: 7 epochs
- Restores best weights on stop
- Translation: "if you haven't improved in 7 epochs, I'm done here"

**ReduceLROnPlateau**
- Monitors: `val_loss`
- Factor: 0.5 (halves the learning rate)
- Patience: 3 epochs
- Minimum LR: 1e-6
- Translation: "things slowing down? try smaller steps"

**ModelCheckpoint**
- Monitors: `val_accuracy`
- Saves only when validation accuracy improves
- Output path: `skin_cancer_model.keras`
- Translation: "only save when things are actually getting better"

---

## Class Imbalance Handling

The ISIC dataset is imbalanced — some conditions are rarer than others. Without compensation the model would just predict the majority class constantly and look fine on accuracy while being useless.

`sklearn.utils.class_weight.compute_class_weight('balanced', ...)` computes per-class weights inversely proportional to frequency. These are passed to `model.fit` via `class_weight=class_weight_dict`.

---

## Model Format

The trained model is saved in the `.keras` format (not `.h5`). This is the current recommended format for TensorFlow 2.12+ and supports the full SavedModel spec.

To load it:

```python
from tensorflow.keras.models import load_model
model = load_model('skin_cancer_model.keras')
```

---

## What's Not In Here (Yet)

- No transfer learning / pretrained backbone
- No attention mechanism
- No ensemble
- No data-efficient augmentation (Mixup, CutMix, etc.)

All of these are plausible improvements. See the [Roadmap](Roadmap) page.
