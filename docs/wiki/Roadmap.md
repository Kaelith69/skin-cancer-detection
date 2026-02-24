# Roadmap

What this project could become if time, motivation, and GPU credits align. Items are roughly ordered by how much they'd improve things.

PRs for any of these are very welcome. See [CONTRIBUTING.md](../../CONTRIBUTING.md).

---

## Near-Term (High Impact, Reasonable Effort)

### Grad-CAM Visualisation
Show which regions of the image the model is actually looking at when it makes a prediction. This is not a nice-to-have — it's the difference between "trust me" and "here's my reasoning." For medical-adjacent applications, this matters a lot.

Implementation: `tf.GradientTape` + last convolutional layer activations.

---

### `predict.py` CLI Script
Right now inference requires writing Python. A simple CLI tool would make the project much more usable:

```bash
python predict.py --image path/to/lesion.jpg
# Predicted: melanoma (confidence: 87.34%)
```

---

### TensorBoard Integration
Live training visualisation. Loss curves, accuracy, learning rate schedule — all in a browser instead of squinting at terminal scrollback.

```python
from tensorflow.keras.callbacks import TensorBoard
tensorboard_callback = TensorBoard(log_dir='./logs')
```

---

### Confusion Matrix Output
The classification report already gives per-class metrics. A confusion matrix would make it much easier to see which classes the model confuses with each other. (Spoiler: it will confuse melanoma and nevus. They're visually similar. So do dermatologists sometimes.)

---

### Unit Tests
`pytest` tests for:
- `load_data` returns iterators with the correct class indices
- `build_model` returns a compiled model with the right output shape
- `build_model` accepts custom `num_classes` and `input_shape`

Currently there are zero tests. This is a known gap.

---

## Medium-Term (Higher Impact, More Effort)

### Transfer Learning Backbone
Replace the scratch-trained CNN with a pretrained backbone (EfficientNetB0 or MobileNetV2). Fine-tuning on ISIC with a pretrained ImageNet backbone typically gives meaningfully better accuracy with less training time.

The current architecture is intentionally simple for learning purposes. A transfer learning version could live alongside it.

---

### Data-Efficient Augmentation
Current augmentation is basic (flip, rotate, zoom, shift, shear). More sophisticated techniques like Mixup, CutMix, or RandAugment tend to improve generalisation on small medical datasets.

---

### Dockerfile
Reproducible environment. One `docker run` and everything works regardless of what's installed locally. Especially useful for GPU environments.

---

## Longer-Term (Ambitious)

### Ensemble Model
Multiple models (e.g., different backbones or different augmentation regimes) voting on predictions. Tends to reduce variance and improve accuracy, especially on rare classes.

---

### Proper Packaging
`pyproject.toml` / `setup.py` so the project can be installed as a package. Makes imports cleaner and deployment easier.

---

### Web Interface
A simple Flask or FastAPI web UI that accepts an image upload and returns a prediction. Still local-only, still not a medical device, but much easier to demo.

---

## Not Planned

- Cloud deployment of model for arbitrary public use (medical liability concerns)
- Mobile app
- REST API for production use
- Anything that routes real patient images through third-party services

---

## How to Pick Something Up

1. Check open issues to see if someone is already working on it
2. Comment on the issue or open a new one to claim it
3. Fork, branch, implement, test, PR
4. See [CONTRIBUTING.md](../../CONTRIBUTING.md) for branch naming and commit style

Thanks for reading this far. You clearly care about the project.
