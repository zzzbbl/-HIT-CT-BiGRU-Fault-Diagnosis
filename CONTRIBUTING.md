# Contributing

Contributions are welcome. Please keep the repository focused on reproducible
source code and lightweight documentation.

## Development Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Guidelines

- Do not commit raw datasets, processed tensors, checkpoints, generated figures,
  logs, or other large binary outputs.
- Keep experiment settings in `configs/`.
- Keep reusable model and training logic under `src/hit_ct_bigru/`.
- Prefer small, focused pull requests.
- Run a syntax check before submitting changes:

```bash
python -m compileall src
```

## Reporting Issues

When reporting a bug, include:

- operating system
- Python version
- PyTorch version
- command used
- configuration file used
- relevant error message

