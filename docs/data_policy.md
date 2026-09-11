# Data Policy

This repository contains source code only. It does not redistribute raw datasets,
processed tensors, trained checkpoints, generated figures, or experiment output
directories.

## Rationale

Large binary files make Git repositories difficult to clone, review, and
maintain. Dataset redistribution may also be subject to the dataset provider's
license or access terms. For these reasons, this project keeps data preparation
code in the repository and leaves dataset acquisition to the user.

## Expected Files

The HIT preparation script expects five raw NumPy arrays:

```text
data1.npy
data2.npy
data3.npy
data4.npy
data5.npy
```

Run:

```bash
python src/prepare_hit_dataset.py --raw-dir <path-to-hit-raw-arrays> --out-dir data/hit_full
```

The generated files under `data/hit_full/` are ignored by Git.

## Tracked And Ignored Files

Tracked:

- source code
- configuration files
- documentation
- license and citation metadata

Ignored:

- raw datasets
- processed datasets
- model checkpoints
- generated figures
- training logs
- experiment outputs
- notebook checkpoints and Python caches

## License Scope

The repository license applies to the source code and documentation maintained
in this repository. It does not grant redistribution rights for external
datasets or third-party materials.
