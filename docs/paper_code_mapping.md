# Paper To Code Mapping

This document maps the main experimental tasks to repository entry points. The
repository is code-only and does not track result images, trained checkpoints,
or large data files.

The code accompanies the following published article:

Wang, Y., Zhang, B. A fault diagnosis method for aero-engine inter-shaft
bearings based on 1DCNN-Transformer-BiGRU. *Scientific Reports* (2026).
https://doi.org/10.1038/s41598-026-61710-4

| Paper experiment | Reproduction entry |
| --- | --- |
| CT-BiGRU main HIT experiment | `python src/train.py --config configs/hit_ct_bigru.yaml` |
| Module ablation | `python src/ablation.py --config configs/hit_ablation.yaml` |
| Noise robustness | `python src/noise_robustness.py --config configs/hit_noise.yaml --checkpoint outputs/hit_ct_bigru/best_model.pth` |
| t-SNE feature visualization | `python src/visualize_features.py --config configs/hit_ct_bigru.yaml --checkpoint outputs/hit_ct_bigru/best_model.pth` |
| HIT full-sequence data split | `python src/prepare_hit_dataset.py --raw-dir <raw_dir> --out-dir data/hit_full` |

Implementation notes:

- The default config follows the paper table: BiGRU hidden units are set to `128`.
- Generated figures, confusion matrices, summaries, and trained weights are
  produced locally during reproduction and are not tracked by Git.
