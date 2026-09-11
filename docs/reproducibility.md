# Reproducibility

This project provides code and configuration files for reproducing the
CT-BiGRU fault diagnosis workflow.

## Recommended Workflow

1. Create a clean Python environment.
2. Install dependencies from `requirements.txt`.
3. Prepare the dataset with `src/prepare_hit_dataset.py`.
4. Train the model with `src/train.py`.
5. Evaluate the checkpoint with `src/evaluate.py`.
6. Run noise robustness or ablation scripts as needed.

## Determinism

The training scripts set Python, NumPy, and PyTorch random seeds. Exact results
may still vary because of:

- GPU hardware differences
- CUDA and cuDNN versions
- PyTorch version differences
- floating point nondeterminism
- dataset preprocessing choices

When reporting results, include the random seed, PyTorch version, hardware, data
split strategy, and configuration file used for the run.

## Output Policy

All generated outputs should remain local unless a release explicitly chooses to
publish them. The default `.gitignore` excludes common output locations and
large binary formats.

