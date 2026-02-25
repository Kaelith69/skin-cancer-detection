# Roadmap

Where this project is going, in rough priority order. The baseline works — now let's make it actually good.

---

## Current State (Baseline)

The current implementation is a solid research baseline — and intentionally humble:

- Custom VGG-style 3-block CNN (no pretrained weights, no shortcuts)
- 9-class ISIC dermoscopy classification
- Class-weighted training to handle dataset imbalance
- EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks
- Full sklearn classification report

It's not fancy. It's honest. That's the point.

---

## Near-Term (High Priority)

### Transfer Learning Backbone

Replace the custom CNN with a pre-trained backbone. ~2,357 images is a small dataset for a CNN trained from scratch — transfer learning is the obvious next step:

- **EfficientNetV2-S** or **MobileNetV3-Large** — strong accuracy/parameter ratio
- Freeze all backbone layers initially; fine-tune top layers with a small learning rate
- Expected accuracy improvement: 5–10 percentage points on the ISIC dataset

```python
from tensorflow.keras.applications import EfficientNetV2S

base_model = EfficientNetV2S(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False
```

### Grad-CAM Explainability

Add gradient-weighted class activation mapping (Grad-CAM) to produce heatmaps that highlight the image regions most influential for the model's prediction. Because "the model said melanoma" is not a satisfying answer on its own:

- Helps clinicians understand *why* the model predicts a given class
- Can detect if the model is focusing on imaging artifacts instead of lesion features
- Output: overlaid heatmap image alongside class prediction

---

## Medium-Term

### TensorFlow Lite Export

Export the trained model to TFLite for on-device mobile inference:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('skin_cancer_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

Enables integration into Android/iOS dermatology apps without a network connection.

### REST API (FastAPI)

Expose the model as a local HTTP API using FastAPI:

```
POST /predict
Content-Type: multipart/form-data
Body: image file

Response:
{
  "predicted_class": "melanoma",
  "confidence": 92.3,
  "probabilities": { ... }
}
```

### Evaluation Visualizations

Enrich the evaluation output with:

- **Confusion matrix** heatmap (matplotlib/seaborn) — shows per-class mis-classification patterns
- **ROC-AUC curves** for each class (one-vs-rest)
- **Training history plot** — loss and accuracy curves per epoch

### Data Augmentation with Albumentations

Replace TensorFlow's built-in `ImageDataGenerator` augmentation with [Albumentations](https://albumentations.ai/) for:

- Elastic deformation (better simulates skin lesion shape variance)
- CLAHE (contrast enhancement for dermoscopy)
- CoarseDropout (simulate occluded skin regions)
- Faster augmentation via CPU parallelism

---

## Long-Term

### Docker Container

Package the entire project in a Docker image for environment-agnostic deployment:

```dockerfile
FROM tensorflow/tensorflow:2.14.0-gpu
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### Hyperparameter Optimization

Integrate [Keras Tuner](https://keras.io/keras_tuner/) or [Optuna](https://optuna.org/) to search over:

- Learning rate, batch size
- Number of filters per block
- Dropout rates
- Optimizer choice

### CI/CD Pipeline

Add a GitHub Actions workflow that:

1. Runs linting (`flake8` / `ruff`)
2. Runs unit tests (`pytest`)
3. Evaluates model on a held-out test set and posts metrics as a PR comment
4. Blocks merges if accuracy drops below a configurable threshold

### Multi-Label Support

Extend the model to support multi-label classification, since some patients present with multiple concurrent conditions. This requires:

- Sigmoid activation in the output layer instead of softmax
- Binary cross-entropy loss
- Updated evaluation metrics (subset accuracy, Hamming loss)

---

## Version History

| Version | Status | Notes |
|---|---|---|
| v0.1 | Released | Baseline custom CNN, 9-class ISIC classifier |
| v0.2 | Planned | Transfer learning backbone (EfficientNetV2) |
| v0.3 | Planned | Grad-CAM + evaluation visualizations |
| v0.4 | Planned | FastAPI serving + TFLite export |
| v1.0 | Planned | Full CI/CD, Docker, hyperparameter tuning |

---

## Contributing to the Roadmap

Have an idea not listed here? Open a GitHub issue with:

- A clear description of the proposed feature
- The motivation (what problem does it solve?)
- Any relevant references or implementation ideas

See [Contributing](Contributing.md) for guidelines on submitting pull requests.
