import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from data_preprocessing import load_data
from model import build_model

# ── Paths & hyper-parameters ──────────────────────────────────────────────────
DATASET_PATH = 'dataset/'
MODEL_SAVE_PATH = 'skin_cancer_model.keras'
EPOCHS = 50  # EarlyStopping will halt before this if the model converges

# ── Load data ─────────────────────────────────────────────────────────────────
train_data, validation_data = load_data(DATASET_PATH)

# ── Compute class weights to handle imbalanced data ───────────────────────────
class_indices = train_data.class_indices          # {'actinic_keratosis': 0, …}
class_labels = {v: k for k, v in class_indices.items()}
y_train = train_data.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train,
)
class_weight_dict = dict(enumerate(class_weights))

# ── Build model ───────────────────────────────────────────────────────────────
model = build_model(num_classes=len(class_indices))
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=validation_data,
    class_weight=class_weight_dict,
    callbacks=callbacks,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(validation_data)
print(f'\nValidation Loss:     {loss:.4f}')
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# ── Detailed classification report ────────────────────────────────────────────
y_true = validation_data.classes
y_pred = np.argmax(model.predict(validation_data), axis=1)
target_names = [class_labels[i] for i in range(len(class_labels))]
print('\nClassification Report:\n')
print(classification_report(y_true, y_pred, target_names=target_names))

