# Security Policy

## Supported Versions

This is a research/learning project. There is currently one active version and no formal release cycle for security patches.

| Version | Supported |
|---|---|
| latest (`main`) | Yes |
| older commits | No |

---

## Reporting a Vulnerability

Found a security issue? Good catch. Please **do not** open a public GitHub issue for it.

Instead:

1. **Email** the maintainer directly. Check the GitHub profile for contact info, or use GitHub's private vulnerability reporting if enabled on this repo.
2. **Describe the issue** clearly:
   - What is the vulnerability?
   - How can it be reproduced?
   - What is the potential impact?
   - Suggested fix (if you have one)
3. **Give us reasonable time** to respond — aim for at least 7 days before any public disclosure.

We'll acknowledge your report, investigate, and let you know when a fix is in place. We'll also credit you in the changelog if you want.

---

## What Counts as a Security Issue Here?

Since this is a local-only Python ML training script, the attack surface is small. But things that would genuinely concern us:

- **Dependency vulnerabilities** — a library in `requirements.txt` has a known CVE
- **Path traversal or arbitrary file write** — e.g. via a crafted dataset path
- **Model poisoning vectors** — anything that could corrupt training in a non-obvious way
- **Unsafe deserialization** — loading a model checkpoint in a way that could execute arbitrary code

---

## What is Explicitly Out of Scope

- "The model makes wrong predictions" — that's not a security issue, that's ML
- "The model could be biased" — valid and important, but not a CVE
- Social engineering of the maintainer
- Issues with the ISIC dataset itself (we don't own that)

---

## Dependencies

You can audit the full dependency list in `requirements.txt`. We use:

- `tensorflow` — primary ML framework
- `numpy` — array operations
- `Pillow` — image loading
- `matplotlib` / `seaborn` — visualisation (not used at inference time)
- `scikit-learn` — class weight computation and classification report
- `pandas` — data manipulation utilities

If you find a known vulnerability in any of these, please report it as described above.

---

## Notes on Medical Data

This project is designed to run **100% offline**. No patient data, no dermoscopic images, and no model weights should ever be transmitted over a network as part of normal operation. If you deploy this in any environment where medical images could be involved, you are responsible for ensuring appropriate data governance and compliance with applicable regulations (HIPAA, GDPR, etc.).

We are not responsible for misuse of this code in clinical settings. Seriously. Go see a dermatologist.
