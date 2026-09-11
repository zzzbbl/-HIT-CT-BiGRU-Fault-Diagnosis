from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from hit_ct_bigru.config import load_config
from hit_ct_bigru.data import build_dataloaders
from hit_ct_bigru.models import build_model
from hit_ct_bigru.utils import ensure_dir, get_device, set_seed


@torch.no_grad()
def collect_final_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    for inputs, batch_labels in loader:
        inputs = inputs.to(device)
        if hasattr(model, "extract_features"):
            batch_features = model.extract_features(inputs)["final"]
        else:
            batch_features = model(inputs)
        features.append(batch_features.detach().cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(features), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create t-SNE visualization from a saved checkpoint.")
    parser.add_argument("--config", default="configs/hit_ct_bigru.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/features")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "auto")))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    model_cfg["return_stage_features"] = True

    _, _, test_loader, _ = build_dataloaders(
        data_dir=Path(args.data_dir or data_cfg.get("data_dir", "data/hit_full")),
        batch_size=int(cfg.get("training", {}).get("batch_size", 16)),
        augment_train=False,
        device=device,
    )
    model_name = model_cfg.pop("name", "ct_bigru")
    model = build_model(model_name, **model_cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    features, labels = collect_final_features(model, test_loader, device)
    points = TSNE(n_components=2, random_state=int(cfg.get("seed", 42)), init="pca", learning_rate="auto").fit_transform(features)

    output_dir = ensure_dir(args.output_dir)
    class_names = data_cfg.get("class_names", ["Healthy", "Inner Fault", "Outer Fault"])
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax.scatter(points[mask, 0], points[mask, 1], s=18, label=class_names[int(label)], alpha=0.8)
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_final_features.png", dpi=300)
    plt.close(fig)
    print(f"Saved feature visualization to {output_dir}")


if __name__ == "__main__":
    main()

