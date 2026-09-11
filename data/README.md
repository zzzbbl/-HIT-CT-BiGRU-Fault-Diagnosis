# Data

This repository does not redistribute raw datasets, processed tensors, model
checkpoints, or generated figures. Users should obtain the dataset according to
the dataset provider's terms and prepare the files locally.

Expected processed HIT layout:

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

To build these files from HIT raw arrays, prepare `data1.npy` through
`data5.npy` and run:

```bash
python src/prepare_hit_dataset.py --raw-dir /path/to/raw_hit_arrays --out-dir data/hit_full
```

The processed data directory is ignored by Git.
