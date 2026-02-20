from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)

# Number of target skin-condition classes
NUM_CLASSES = 9
IMG_SHAPE = (224, 224, 3)


def build_model(num_classes=NUM_CLASSES, input_shape=IMG_SHAPE):
    """Build and compile a CNN model for skin-cancer classification.

    The architecture follows a standard VGG-style pattern:
      Conv → BN → ReLU → Conv → BN → ReLU → Pool → Dropout  (×3 blocks)
    followed by fully-connected classification head.

    Args:
        num_classes: Number of output classes (default: 9).
        input_shape: Input image shape as (H, W, C) tuple.

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu',
               input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Classification head
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
