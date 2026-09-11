from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from tqdm import tqdm

from .utils import ensure_dir, model_output, save_json


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(cfg.get("name", "adam")).lower()
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model_output(model(inputs))
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device: torch.device,
    output_dir: str | Path,
    epochs: int = 50,
    min_epochs: int = 0,
    early_stop_patience: int | None = None,
):
    output_dir = ensure_dir(output_dir)
    best_acc = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"train_loss={train_metrics['loss']:.6f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            stale_epochs += 1

        if early_stop_patience and epoch >= min_epochs and stale_epochs >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    save_json(
        {"best_val_accuracy": best_acc, "best_epoch": best_epoch, "seconds": time.time() - start, "history": history},
        output_dir / "training_history.json",
    )
    return history


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: torch.device, class_names=None) -> dict[str, Any]:
    model.eval()
    y_true = []
    y_pred = []
    for inputs, labels in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        logits = model_output(model(inputs))
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
    }

