from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image dimensions and batch size constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42


def load_data(dataset_path):
    """Load and preprocess training and validation data from a directory.

    Augmentation is applied only to the training set; the validation set is
    rescaled only to avoid data leakage.

    Args:
        dataset_path: Path to the root dataset directory whose sub-folders
            represent class labels.

    Returns:
        A tuple of (train_data, validation_data) Keras DirectoryIterators.
    """
    # Training generator: augmentation + rescaling
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        horizontal_flip=True,
        vertical_flip=False,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
    )

    # Validation generator: rescaling only (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    train_data = train_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=SEED,
        shuffle=True,
    )

    validation_data = val_datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=SEED,
        shuffle=False,
    )

    return train_data, validation_data
