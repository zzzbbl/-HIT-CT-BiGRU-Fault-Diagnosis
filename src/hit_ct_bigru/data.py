from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from joblib import load
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .augmentation import SafeSignalAugmenter


SPLIT_FILES = {
    "train_x": "train_xdata",
    "train_y": "train_ylabel",
    "val_x": "val_xdata",
    "val_y": "val_ylabel",
    "test_x": "test_xdata",
    "test_y": "test_ylabel",
}


class AugmentedTensorDataset(Dataset):
    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, augmenter=None) -> None:
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.x_tensor)

    def __getitem__(self, idx: int):
        x = self.x_tensor[idx]
        y = self.y_tensor[idx]
        if self.augmenter is not None:
            x = self.augmenter(x)
        return x, y


def load_tensor_splits(data_dir: str | Path) -> dict[str, torch.Tensor]:
    data_dir = Path(data_dir)
    missing = [name for name in SPLIT_FILES.values() if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed split files in {data_dir}: {missing}. "
            "Run src/prepare_hit_dataset.py first or place the files manually."
        )
    return {key: load(data_dir / filename) for key, filename in SPLIT_FILES.items()}


def compute_class_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.cpu().numpy())
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 16,
    augment_train: bool = True,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    splits = load_tensor_splits(data_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_workers is None:
        num_workers = 0 if sys.platform == "win32" else 2
    if pin_memory is None:
        pin_memory = device.type == "cuda"

    train_dataset = AugmentedTensorDataset(
        splits["train_x"],
        splits["train_y"],
        augmenter=SafeSignalAugmenter() if augment_train else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(splits["val_x"], splits["val_y"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(splits["test_x"], splits["test_y"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    class_weights = compute_class_weights(splits["train_y"], device=device)
    return train_loader, val_loader, test_loader, class_weights

