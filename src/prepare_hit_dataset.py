from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


DEFAULT_FILE_LABELS = {
    "data1.npy": 0,
    "data2.npy": 0,
    "data3.npy": 1,
    "data4.npy": 1,
    "data5.npy": 2,
}

DEFAULT_LABEL_NAMES = {
    0: "Healthy",
    1: "Inner Fault",
    2: "Outer Fault",
}


def load_hit_arrays(raw_dir: Path, channel_start: int, channel_end: int):
    samples = []
    labels = []
    sample_ids = []
    for filename, label in DEFAULT_FILE_LABELS.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing raw HIT array: {path}")
        data = np.load(path)
        if data.ndim != 3:
            raise ValueError(f"{path} should have shape [samples, channels, time], got {data.shape}")
        for idx in range(data.shape[0]):
            samples.append(data[idx, channel_start:channel_end, :])
            labels.append(label)
            sample_ids.append(f"{filename}:{idx}")
    return np.asarray(samples), np.asarray(labels), np.asarray(sample_ids)


def save_split(x, y, out_dir: Path, prefix: str) -> None:
    dump(torch.tensor(x, dtype=torch.float32), out_dir / f"{prefix}_xdata")
    dump(torch.tensor(y, dtype=torch.long), out_dir / f"{prefix}_ylabel")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed HIT full-sequence splits from raw .npy arrays.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing data1.npy ... data5.npy.")
    parser.add_argument("--out-dir", default="data/hit_full")
    parser.add_argument("--channel-start", type=int, default=2)
    parser.add_argument("--channel-end", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, sample_ids = load_hit_arrays(raw_dir, args.channel_start, args.channel_end)
    indices = np.arange(len(x))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=args.seed, stratify=y)
    subtrain_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=args.seed, stratify=y[train_idx])

    if set(sample_ids[subtrain_idx]) & set(sample_ids[val_idx]):
        raise RuntimeError("Train/validation leakage detected.")
    if set(sample_ids[subtrain_idx]) & set(sample_ids[test_idx]):
        raise RuntimeError("Train/test leakage detected.")
    if set(sample_ids[val_idx]) & set(sample_ids[test_idx]):
        raise RuntimeError("Validation/test leakage detected.")

    x_train, y_train = x[subtrain_idx], y[subtrain_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    scaler = RobustScaler()
    scaler.fit(x_train.transpose(0, 2, 1).reshape(-1, x_train.shape[1]))

    def transform(arr):
        original_shape = arr.shape
        arr2 = arr.transpose(0, 2, 1).reshape(-1, arr.shape[1])
        arr2 = scaler.transform(arr2).reshape(original_shape[0], original_shape[2], original_shape[1])
        return arr2.transpose(0, 2, 1)

    save_split(transform(x_train), y_train, out_dir, "train")
    save_split(transform(x_val), y_val, out_dir, "val")
    save_split(transform(x_test), y_test, out_dir, "test")

    metadata = {
        "input_shape": [int(x.shape[1]), int(x.shape[2])],
        "class_labels": DEFAULT_LABEL_NAMES,
        "split_ratio": "train 56%, validation 14%, test 30%",
        "channel_slice": [args.channel_start, args.channel_end],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Full-sequence split by original sample id to avoid window leakage.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved processed HIT splits to {out_dir}")


if __name__ == "__main__":
    main()

