# HIT CT-BiGRU Fault Diagnosis

[English](README.md) | [中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

Official implementation of a 1DCNN-Transformer-BiGRU fault diagnosis pipeline
for aero-engine inter-shaft bearing vibration signals.

This repository accompanies the research article published in *Scientific
Reports*:

> Wang, Y., Zhang, B. A fault diagnosis method for aero-engine inter-shaft
> bearings based on 1DCNN-Transformer-BiGRU. *Scientific Reports* (2026).
> https://doi.org/10.1038/s41598-026-61710-4

The project provides reproducible source code for dataset preparation, model
training, evaluation, noise robustness testing, module ablation, and feature
visualization. Datasets, trained checkpoints, generated figures, and experiment
outputs are intentionally excluded from the repository.

## Highlights

- CT-BiGRU architecture combining a large-kernel 1DCNN front end, Transformer
  encoder, and bidirectional GRU.
- Full-sequence HIT data preparation with sample-level train, validation, and
  test splitting.
- Baseline and ablation model implementations for controlled comparison.
- Config-driven training and evaluation scripts.
- Noise robustness evaluation under configurable SNR levels.
- t-SNE feature visualization utility for trained checkpoints.

## Repository Structure

```text
configs/                  YAML experiment configurations
data/                     Dataset placement instructions
docs/                     Reproducibility and data policy notes
notebooks/                Optional notebooks, if added in future releases
src/hit_ct_bigru/         Reusable Python package
src/train.py              Train CT-BiGRU or baseline models
src/evaluate.py           Evaluate a saved checkpoint
src/noise_robustness.py   Evaluate robustness under injected noise
src/ablation.py           Run module ablation experiments
src/prepare_hit_dataset.py  Prepare processed HIT tensor splits
src/visualize_features.py   Generate t-SNE feature visualizations
```

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For editable package installation:

```bash
pip install -e .
```

For GPU training, install the PyTorch build that matches your CUDA version from
the official PyTorch installation guide.

## Dataset Preparation
The HIT aero-engine inter-shaft bearing dataset can be obtained from:

- Dataset page: <https://github.com/HouLeiHIT/HIT-dataset.>

Please follow the dataset provider's license and access terms. 

This repository does not redistribute the HIT dataset or derived tensor files.
Prepare the raw HIT arrays separately with the following expected filenames:

```text
data1.npy
data2.npy
data3.npy
data4.npy
data5.npy
```

Then build processed full-sequence splits:

```bash
python src/prepare_hit_dataset.py --raw-dir <path-to-hit-raw-arrays> --out-dir data/hit_full
```

The generated directory should contain:

```text
data/hit_full/
├── train_xdata
├── train_ylabel
├── val_xdata
├── val_ylabel
├── test_xdata
├── test_ylabel
└── metadata.json
```

See [data/README.md](data/README.md) and
[docs/data_policy.md](docs/data_policy.md) for details.

## Training

Train the default CT-BiGRU model:

```bash
python src/train.py --config configs/hit_ct_bigru.yaml
```

Training artifacts are written to `outputs/`, which is ignored by Git.

## Evaluation

Evaluate a trained checkpoint:

```bash
python src/evaluate.py \
  --config configs/hit_ct_bigru.yaml \
  --checkpoint outputs/hit_ct_bigru/best_model.pth
```

## Noise Robustness

Evaluate classification accuracy under injected Gaussian noise:

```bash
python src/noise_robustness.py \
  --config configs/hit_noise.yaml \
  --checkpoint outputs/hit_ct_bigru/best_model.pth
```

## Ablation Experiments

Run the module ablation experiments:

```bash
python src/ablation.py --config configs/hit_ablation.yaml
```

The ablation entry trains the following variants:

- `1dcnn_transformer`
- `1dcnn_bigru`
- `transformer_bigru`
- `ct_bigru`

## Default Model Configuration

The default configuration in [configs/hit_ct_bigru.yaml](configs/hit_ct_bigru.yaml)
uses:

- input channels: `4`
- sequence length: `20480`
- 1DCNN kernel sizes: `(32, 16)`
- 1DCNN strides: `(8, 4)`
- Transformer encoder layers: `2`
- attention heads: `8`
- embedding dimension: `128`
- feed-forward dimension: `512`
- BiGRU hidden units: `128`
- convolutional dropout: `0.4`
- batch size: `16`
- epochs: `50`

## Reproducibility

The scripts set random seeds and use config files to make runs easier to repeat.
Exact numerical results can still vary across hardware, PyTorch versions, CUDA
libraries, and dataset preprocessing choices.

For recommended reporting practice, see
[docs/reproducibility.md](docs/reproducibility.md).

## Citation

If this repository helps your research, please cite the paper:

```bibtex
@article{wang2026fault,
  title = {A fault diagnosis method for aero-engine inter-shaft bearings based on 1DCNN-Transformer-BiGRU},
  author = {Wang, Yang and Zhang, Boliang},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-61710-4},
  url = {https://doi.org/10.1038/s41598-026-61710-4}
}
```

A machine-readable citation template is provided in [CITATION.cff](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE). The license applies
to the source code and documentation in this repository. External datasets,
third-party papers, and user-generated experiment outputs are governed by their
own terms.
