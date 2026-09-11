from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hit_ct_bigru.config import load_config
from hit_ct_bigru.data import build_dataloaders
from hit_ct_bigru.models import build_model
from hit_ct_bigru.plotting import plot_confusion_matrix, plot_training_curves
from hit_ct_bigru.training import build_optimizer, evaluate_model, train_model
from hit_ct_bigru.utils import ensure_dir, get_device, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CT-BiGRU or a baseline model.")
    parser.add_argument("--config", default="configs/hit_ct_bigru.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "auto")))

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    data_dir = Path(args.data_dir or data_cfg.get("data_dir", "data/hit_full"))
    output_dir = ensure_dir(args.output_dir or train_cfg.get("output_dir", "outputs/hit_ct_bigru"))

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        data_dir=data_dir,
        batch_size=int(train_cfg.get("batch_size", 16)),
        augment_train=bool(train_cfg.get("augment_train", True)),
        device=device,
    )

    model_name = model_cfg.pop("name", "ct_bigru")
    model = build_model(model_name, **model_cfg).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights if train_cfg.get("class_weight", True) else None)
    optimizer = build_optimizer(model, train_cfg.get("optimizer", {}))

    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        output_dir,
        epochs=int(train_cfg.get("epochs", 50)),
        min_epochs=int(train_cfg.get("min_epochs", 0)),
        early_stop_patience=train_cfg.get("early_stop_patience"),
    )
    plot_training_curves(history, output_dir)

    best_path = output_dir / "best_model.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    class_names = data_cfg.get("class_names", ["Healthy", "Inner Fault", "Outer Fault"])
    metrics = evaluate_model(model, test_loader, device, class_names=class_names)
    save_json(metrics, output_dir / "test_metrics.json")
    plot_confusion_matrix(metrics["confusion_matrix"], output_dir, class_names=class_names)
    print(metrics["classification_report"])
    print(f"Saved run artifacts to {output_dir}")


if __name__ == "__main__":
    main()

