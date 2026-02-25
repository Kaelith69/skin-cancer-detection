# Architecture

This page is a deep-dive into the CNN architecture, module structure, and internal data flow. If you've ever wondered why a specific design decision was made instead of a dozen other reasonable alternatives, this is the place.

---

## High-Level Design

Three Python modules, each with exactly one job. Because the moment a module has two jobs, it has infinite jobs.

| Module | Responsibility |
|---|---|
| `data_preprocessing.py` | Load images from disk, apply augmentation (training only), split 80/20 |
| `model.py` | Define and compile the CNN — returns a ready-to-train `Sequential` model |
| `main.py` | Orchestrate training, compute class weights, run evaluation, print report |

---

## CNN Architecture

### Overview

The network follows a **VGG-style** design: repeated `[Conv → BN → Conv → BN → Pool → Dropout]` blocks that progressively halve the spatial resolution while doubling the filter count.

```
Input (224×224×3)
    |
[Block 1]  Conv2D(32)  → BN → Conv2D(32)  → BN → MaxPool(2×2) → Dropout(0.25)
    |       Output: 112×112×32
[Block 2]  Conv2D(64)  → BN → Conv2D(64)  → BN → MaxPool(2×2) → Dropout(0.25)
    |       Output: 56×56×64
[Block 3]  Conv2D(128) → BN → Conv2D(128) → BN → MaxPool(2×2) → Dropout(0.25)
    |       Output: 28×28×128
Flatten     Output: 100352
Dense(256) → BN → Dropout(0.5)
Dense(9, softmax)
```

### Layer-by-Layer Breakdown

#### Convolutional Blocks (×3)

Each block contains two convolutional layers followed by pooling and regularization:

- **Conv2D**: 3×3 kernels, `padding='same'` (preserves spatial size within the block), ReLU activation
- **BatchNormalization**: Normalizes feature maps per mini-batch, accelerating convergence and providing implicit L2-like regularization
- **MaxPooling2D(2×2)**: Downsamples by a factor of 2 in each spatial dimension
- **Dropout(0.25)**: Randomly zeroes 25% of feature maps to reduce co-adaptation

Filter progression (32 → 64 → 128) follows the standard practice of doubling filters each time the spatial size is halved, keeping the total parameter count per block roughly constant.

#### Classification Head

- **Flatten**: Converts the 28×28×128 = 100,352-dimensional feature volume into a 1D vector
- **Dense(256, ReLU)**: Learns global classification patterns from the convolutional features
- **BatchNormalization**: Stabilizes the dense representation
- **Dropout(0.5)**: Higher dropout rate in the dense head to prevent overfitting
- **Dense(9, softmax)**: Outputs a probability distribution over the 9 diagnostic classes

### Compilation

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
```

- **Adam**: Adaptive learning rate optimizer — generally robust to hyperparameter choices
- **Categorical cross-entropy**: Standard loss for multi-class soft-label targets (one-hot encoded)
- **Accuracy**: Top-1 classification accuracy on the validation set

---

## Data Preprocessing Architecture

### Separate Generator Design

Two independent `ImageDataGenerator` instances are created from the same dataset root:

```python
# train_datagen: augmentation + rescale
# val_datagen:   rescale only
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, ...)
val_datagen   = ImageDataGenerator(rescale=1./255, validation_split=0.2)
```

The `validation_split=0.2` split is determined by TensorFlow's internal file sorting (using `seed=42`). By using two separate generator objects, augmentation transforms are applied **only** to the training iterator, not to the validation iterator — a critical design choice to prevent data leakage.

### Augmentation Transforms (Training Only)

| Transform | Parameter | Effect |
|---|---|---|
| Rescale | `1/255` | Normalize pixel values to [0, 1] |
| Horizontal flip | `True` | Mirror lesions left-right |
| Rotation | ±20° | Simulate orientation variance |
| Zoom | ±20% | Simulate distance/magnification variance |
| Width shift | ±10% | Random horizontal translation |
| Height shift | ±10% | Random vertical translation |
| Shear | ±10% | Simulate perspective distortion |

---

## Training Architecture

### Class-Weight Balancing

The ISIC dataset is naturally imbalanced (e.g., nevus images greatly outnumber vascular lesion images). To correct for this:

```python
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train,
)
class_weight_dict = dict(enumerate(class_weights))
```

`sklearn`'s `balanced` strategy computes: `weight[i] = n_samples / (n_classes × count[i])`. This makes the model pay proportionally more attention to rare classes during backpropagation.

### Callbacks

| Callback | Configuration | Purpose |
|---|---|---|
| `EarlyStopping` | monitor=`val_loss`, patience=7, restore_best_weights=True | Halt training when validation loss stops improving; revert to best weights |
| `ModelCheckpoint` | monitor=`val_accuracy`, save_best_only=True | Persist only the epoch with the highest validation accuracy |
| `ReduceLROnPlateau` | monitor=`val_loss`, factor=0.5, patience=3, min_lr=1e-6 | Halve learning rate after 3 stagnant epochs |

---

## Design Decisions

### Why VGG-style instead of a pre-trained backbone?

This project is a purpose-built research baseline. A VGG-style architecture:
- Is entirely transparent — no hidden pretrained weights, no "trust me bro" ImageNet knowledge
- Trains from scratch on domain-specific ISIC data
- Provides a clear performance floor for comparing future transfer-learning experiments

Think of it as the control group. You need a boring baseline before you can have interesting results.

### Why `.keras` format instead of `.h5`?

The `.keras` format (introduced in TensorFlow 2.12) is the modern, recommended serialization format. It is more portable, supports more Keras objects, and will be the only supported format in future TF versions. The `.h5` format is like that old config file you keep around "just in case" — stop it.

### Why separate `ImageDataGenerator` instances?

If the same generator (with augmentation) were used with both `subset='training'` and `subset='validation'`, augmentation would be applied to both subsets, causing artificial performance inflation on the validation set (data leakage). Your metrics would look great. Your model would be lying to you. These are not the same thing.
