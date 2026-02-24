# Contributing to Skin Cancer Detection

Hey, you want to contribute? Genuinely awesome. Pull requests are open, ideas are welcome, and judgment is mostly suspended (mostly).

Whether you're fixing a typo, adding a feature, or completely rethinking the data pipeline at 2 AM because you *just had a thought* — you're in the right place.

---

## Before You Start

A few things to keep in mind:

- **This is a research/learning project.** Not a production medical tool. Don't treat it like one, but do treat the code with care.
- **Don't hallucinate features.** Only build things that actually make sense for the existing stack (TensorFlow/Keras, scikit-learn, Python).
- **Read the code first.** It's only three Python files. Seriously. It'll take you ten minutes.

---

## How to Contribute

### 1. Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2. Create a branch

Use a descriptive name. `fix-typo-readme` is fine. `my-branch` is not fine.

```bash
git checkout -b feat/grad-cam-visualisation
# or
git checkout -b fix/validation-data-leak
# or
git checkout -b docs/add-wiki-troubleshooting
```

**Branch prefixes:**

| Prefix | Use for |
|---|---|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code cleanup without behaviour change |
| `test/` | Adding or updating tests |
| `chore/` | Dependency bumps, CI, tooling |

### 3. Make your changes

- Keep changes focused. One PR = one thing. Don't submit a PR that fixes a bug, adds a feature, and reformats the entire codebase.
- Follow the existing code style — type hints, docstrings, clean constants at the top of modules.
- If you're adding a feature, mention it in the README roadmap and check it off.

### 4. Test your changes

There are no automated tests yet (it's on the roadmap — maybe you're the one to add them). At minimum, run the pipeline and verify nothing is obviously broken:

```bash
pip install -r requirements.txt
python main.py
```

### 5. Commit your changes

Follow conventional commit messages — they're readable and make changelogs easy:

```
feat: add Grad-CAM saliency map visualisation
fix: prevent data leakage in validation generator
docs: update README with correct EPOCHS default
refactor: extract class label list into constants
test: add unit tests for load_data function
chore: bump tensorflow to 2.13.0
```

Format: `<type>(<optional scope>): <short description>`

- Use imperative mood: "add" not "added", "fix" not "fixed"
- Keep the subject line under 72 characters
- Add a body if context is needed

### 6. Open a Pull Request

- Title: same format as your commit message
- Description: explain *what* you changed and *why*
- Link any relevant issues

---

## What Makes a Good PR

- Solves one clear problem
- Doesn't break existing functionality
- Includes a clear description of what changed and why
- Is reasonably sized — no 2000-line monsters unless truly necessary

---

## What Gets Rejected

- Features that introduce dependencies not in `requirements.txt` without discussion
- Code that breaks the existing training pipeline
- "Improvements" that are just personal style preferences
- Anything that pretends this is a medical device

---

## Code Style

- Python 3.9+
- Follow PEP 8 (use a linter if you want — `flake8` or `ruff` are fine)
- Docstrings on public functions (Google or NumPy style, pick one and be consistent)
- Constants in UPPER_CASE at module level
- No magic numbers buried in logic

---

## Questions?

Open an issue. Label it `question`. We don't bite (the model might classify you though).

Thanks for contributing. You're making a research tool better — that matters.
