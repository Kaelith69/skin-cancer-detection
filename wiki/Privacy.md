# Privacy

No cloud. No spying. No villain origin story.

This page documents the data handling practices, security model, and regulatory considerations for the Skin Cancer Detection project. Short version: everything stays local, nothing phones home, and this is a research tool not a clinical device.

---

## Data Handled by This Project

### Training / Validation Data

- **Source:** ISIC (International Skin Imaging Collaboration) public research archive
- **Nature:** Anonymized dermoscopic photographs contributed by research institutions
- **Consent model:** All ISIC images were collected under institutional review board (IRB) protocols with patient consent for research use
- **PII:** No personally identifiable information (name, date of birth, patient ID) is stored in or transmitted by this repository

### User-Supplied Inference Images

- Inference is performed entirely **locally** on the user's machine
- Images passed to `model.predict()` are never written to disk by the inference code
- No data is transmitted to any remote server, API, or cloud service

---

## What This Repository Stores

| Artefact | Location | Contains PII? |
|---|---|---|
| Model weights | `skin_cancer_model.keras` | No |
| Training images | `dataset/` (included in this repository) | No — ISIC only |
| Source code | `*.py` | No |
| Configuration | `requirements.txt` | No |

The `dataset/` directory is intentionally excluded from version control. Contributors must **never** commit patient images or any data with identifiable attributes to the repository.

---

## Security Model

### Offline-Only Inference

The trained model operates in a fully offline, local execution context. There are no:

- HTTP/HTTPS requests during inference
- WebSocket connections
- Cloud storage uploads
- Analytics or telemetry callbacks

If you put a network sniffer on this thing, you'll get bored quickly.

### Model Weight Security

Trained `.keras` model files contain learned numerical weights — no patient data is embedded. However:

- Restrict access to model weight files if they are deployed on a shared server
- Use filesystem permissions (chmod 640 or similar) to prevent world-readable model files
- Do not expose model weights over unauthenticated HTTP endpoints

### Dependency Security

This project depends on well-maintained open-source libraries (TensorFlow, NumPy, scikit-learn, Pillow). Keep dependencies up to date to receive security patches:

```bash
pip install --upgrade -r requirements.txt
```

---

## Regulatory Considerations

> **This project is a research tool, not a medical device.**

Any deployment in a clinical or patient-facing context would require — depending on jurisdiction — formal regulatory approval. This is not optional and not a formality. The table below is a starting point, not a checklist you can skip.

| Jurisdiction | Applicable Regulation |
|---|---|
| United States | FDA 510(k) premarket notification or De Novo classification (Software as a Medical Device) |
| European Union | CE marking under MDR 2017/745 |
| United Kingdom | UKCA marking under UK MDR 2002 (as amended) |
| Canada | Health Canada SaMD guidance |

The creators of this repository accept **no liability** for clinical decisions made using this software.

---

## GDPR Considerations

If this project is extended to process images from EU/EEA residents:

- Dermoscopic images may qualify as **health data** under GDPR Article 9 (special category data)
- Processing health data requires an explicit legal basis (e.g., explicit consent, scientific research exemption)
- A Data Protection Impact Assessment (DPIA) must be completed before any such deployment
- Images should be pseudonymized or anonymized before being used for model training or evaluation

---

## Responsible Disclosure

If you discover a security vulnerability in this project:

1. **Do not** open a public GitHub issue
2. Email the repository owner directly (see GitHub profile for contact)
3. Include a description of the issue and steps to reproduce
4. Allow a reasonable time for a fix before public disclosure

---

## Summary

| Property | Status |
|---|---|
| Patient data stored in repo | No |
| Network requests during inference | No |
| Telemetry / analytics | No |
| Suitable for clinical deployment | No — research only |
| ISIC data license compliance | Yes — research use |
