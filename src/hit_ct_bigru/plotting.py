from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import ensure_dir


def plot_training_curves(history: list[dict], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["loss"] for row in history]
    val_loss = [row["val"]["loss"] for row in history]
    train_acc = [row["train"]["accuracy"] for row in history]
    val_acc = [row["val"]["accuracy"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, train_acc, label="Train")
    axes[1].plot(epochs, val_acc, label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrix(matrix, output_dir: str | Path, class_names=None, filename: str = "confusion_matrix.png") -> None:
    output_dir = ensure_dir(output_dir)
    class_names = class_names or [str(i) for i in range(len(matrix))]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(np.asarray(matrix), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)

