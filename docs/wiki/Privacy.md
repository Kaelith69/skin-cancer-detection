# Privacy

Short version: **your data stays on your machine.**

Longer version follows.

---

## What This Project Does Not Do

- Does not send images, model weights, or training metrics anywhere
- Does not phone home
- Does not collect usage analytics
- Does not log anything to an external service
- Does not have a server component
- Does not have network access during training or inference

This is a local Python script. It reads files from your disk, does math, writes a model file back to your disk. That's the entire I/O surface.

---

## What Leaves Your Machine

The only network activity this project intentionally triggers is **you manually downloading the ISIC dataset** from the ISIC Archive. After that download, no further network access is needed or performed.

When you run `python main.py`, the only files read or written are:
- Images from `dataset/` (read)
- `skin_cancer_model.keras` (written — your trained model)
- Terminal output (your screen)

---

## Medical Image Handling

Dermoscopic images are sensitive medical data. This project was designed with that in mind:

- No image uploading
- No cloud processing
- No third-party API calls
- Training and inference are fully air-gappable

If you are building a tool on top of this project that handles real patient images:

1. **Do not add network capabilities without explicit consent from data subjects**
2. **Comply with applicable regulations** — HIPAA (US), GDPR (EU), or equivalent in your jurisdiction
3. **Don't store identifiable data without appropriate safeguards**
4. **Understand that this code is not validated for clinical use**

---

## Dependencies and Telemetry

Some libraries used in this project may have their own telemetry defaults:

- **TensorFlow** — may collect crash reports unless opted out. See [TensorFlow's privacy policy](https://www.tensorflow.org/about/privacy) and disable with `TF_CPP_MIN_LOG_LEVEL` or appropriate environment variables.
- **pip** — may send anonymized install statistics. Use `pip install --no-index` or `pip install --quiet` if preferred.
- Other libraries (NumPy, scikit-learn, etc.) — no known telemetry.

---

## The ISIC Dataset

Images in this project come from the [ISIC Archive](https://www.isic-archive.com). The ISIC dataset has its own terms of use and privacy policy. Images are de-identified but you should review their terms before using the dataset in any context beyond personal research.

---

## Summary

If you run this project on your own machine with your own dataset copy, nothing about that run is observable by anyone else unless you deliberately share the output. It's as private as any Python script you run locally.
