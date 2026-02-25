<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 200" width="860" height="200" role="img" aria-label="Roadmap page header">
  <defs>
    <linearGradient id="roadHeroBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0a1628;stop-opacity:1"/>
      <stop offset="50%" style="stop-color:#1a1400;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#1a1200;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="roadAccent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#eab308;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#f97316;stop-opacity:1"/>
    </linearGradient>
    <filter id="roadGlow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="860" height="200" rx="14" fill="url(#roadHeroBg)"/>
  <g opacity="0.04" stroke="#eab308" stroke-width="1" fill="none">
    <line x1="0" y1="40" x2="860" y2="40"/><line x1="0" y1="80" x2="860" y2="80"/>
    <line x1="0" y1="120" x2="860" y2="120"/><line x1="0" y1="160" x2="860" y2="160"/>
    <line x1="86" y1="0" x2="86" y2="200"/><line x1="172" y1="0" x2="172" y2="200"/>
    <line x1="258" y1="0" x2="258" y2="200"/><line x1="344" y1="0" x2="344" y2="200"/>
    <line x1="430" y1="0" x2="430" y2="200"/><line x1="516" y1="0" x2="516" y2="200"/>
    <line x1="602" y1="0" x2="602" y2="200"/><line x1="688" y1="0" x2="688" y2="200"/>
    <line x1="774" y1="0" x2="774" y2="200"/>
  </g>
  <circle cx="760" cy="40" r="80" fill="#eab308" opacity="0.07"/>
  <circle cx="100" cy="160" r="70" fill="#f97316" opacity="0.07"/>
  <rect x="0" y="0" width="860" height="4" fill="url(#roadAccent)" opacity="0.9"/>
  <text x="430" y="88" font-family="'Segoe UI',system-ui,Arial,sans-serif" font-size="36"
        font-weight="800" fill="white" text-anchor="middle" filter="url(#roadGlow)">Roadmap</text>
  <text x="430" y="122" font-family="'Segoe UI',system-ui,Arial,sans-serif" font-size="15"
        fill="#fef08a" text-anchor="middle">Transfer Learning · Grad-CAM · TFLite · REST API · Benchmark Suite</text>
  <rect x="180" y="138" width="500" height="2" rx="1" fill="url(#roadAccent)" opacity="0.6"/>
  <rect x="0" y="196" width="860" height="4" fill="url(#roadAccent)" opacity="0.5"/>
</svg>

</div>

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

Replace the custom CNN with a pre-trained backbone. ~2,239 images is a small dataset for a CNN trained from scratch — transfer learning is the obvious next step:

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
