# Contributing

Thank you for your interest in contributing to Skin Cancer Detection. This page describes the development workflow, code standards, and review process.

---

## Code of Conduct

This project follows a standard open-source code of conduct:

- Be respectful and constructive in all communication
- Welcome contributors of all backgrounds and experience levels
- Focus criticism on code and ideas, not on individuals
- Report unacceptable behaviour to the repository maintainer

---

## How to Contribute

### Types of Contribution

| Type | Examples |
|---|---|
| Bug fixes | Incorrect preprocessing logic, broken evaluation code |
| Feature additions | Grad-CAM visualization, TFLite export, new augmentation options |
| Documentation | Improving wiki pages, adding docstrings, fixing typos |
| Performance | Faster data loading, memory optimization |
| Testing | Adding unit tests or integration tests |

---

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2. Set Up the Development Environment

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/grad-cam-visualization
# or for a bug fix:
git checkout -b fix/validation-split-seed
```

Branch naming conventions:

| Prefix | Use case |
|---|---|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `docs/` | Documentation-only changes |
| `refactor/` | Code restructuring with no behaviour change |
| `perf/` | Performance improvements |

### 4. Make Your Changes

Follow the code style guidelines below. Make small, atomic commits with descriptive messages:

```bash
git add data_preprocessing.py
git commit -m "fix: apply seed consistently to both train and val generators"
```

Commit message format (loosely follows [Conventional Commits](https://www.conventionalcommits.org/)):

```
<type>: <short description>

[optional body]
[optional footer]
```

Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`

### 5. Push and Open a Pull Request

```bash
git push origin feature/grad-cam-visualization
```

Then open a Pull Request on GitHub against the `main` branch. Include:

- A clear description of the change
- The motivation or linked issue
- How to test the change
- Any performance implications

---

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/) conventions
- Maximum line length: **88 characters** (compatible with Black)
- Use f-strings for string formatting
- Avoid bare `except:` clauses — catch specific exceptions

### Docstrings

All public functions must have docstrings in the Google style:

```python
def load_data(dataset_path: str):
    """Load and preprocess training and validation data.

    Args:
        dataset_path: Path to the root dataset directory.

    Returns:
        A tuple of (train_data, validation_data) DirectoryIterators.
    """
```

### Type Hints

Use type hints on all new function signatures:

```python
def build_model(num_classes: int = 9, input_shape: tuple = (224, 224, 3)):
    ...
```

### Imports

Group imports in this order, separated by blank lines:
1. Standard library
2. Third-party packages
3. Local modules

---

## Testing

There is no formal test suite at present. When adding new features, please include at minimum:

- A short docstring explaining the expected behaviour
- A comment showing example expected output
- Manual verification steps in your PR description

If you add tests, place them in a `tests/` directory at the project root:

```
tests/
|-- test_data_preprocessing.py
|-- test_model.py
`-- test_inference.py
```

Use `pytest` as the test runner:

```bash
pip install pytest
pytest tests/ -v
```

---

## Pull Request Review Criteria

PRs will be reviewed for:

| Criterion | Details |
|---|---|
| Correctness | Does the code do what it claims? |
| No data leakage | Does any new preprocessing correctly isolate val/train augmentation? |
| Code style | PEP 8 compliance, docstrings, type hints |
| Backwards compatibility | Does the change break existing workflows? |
| Documentation | Are wiki pages or docstrings updated? |

---

## Reporting Issues

Open a GitHub issue and include:

- Python version (`python --version`)
- TensorFlow version (`python -c "import tensorflow; print(tensorflow.__version__)"`)
- Operating system and hardware
- Full traceback if applicable
- Minimal reproducible example
