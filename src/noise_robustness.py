from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hit_ct_bigru.augmentation import add_awgn
from hit_ct_bigru.config import load_config
from hit_ct_bigru.data import build_dataloaders
from hit_ct_bigru.models import build_model
from hit_ct_bigru.utils import ensure_dir, get_device, model_output, save_json, set_seed


@torch.no_grad()
def evaluate_noise(model, loader, device, snr_levels):
    model.eval()
    results = {}
    for snr in snr_levels:
        total = 0
        correct = 0
        for inputs, labels in loader:
            inputs = add_awgn(inputs.to(device), float(snr))
            labels = labels.to(device)
            pred = model_output(model(inputs)).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        results[str(snr)] = correct / total
        print(f"SNR={snr} dB accuracy={results[str(snr)]:.4f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint robustness under injected Gaussian noise.")
    parser.add_argument("--config", default="configs/hit_noise.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/noise")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "auto")))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    noise_cfg = cfg.get("noise", {})

    _, _, test_loader, _ = build_dataloaders(
        data_dir=Path(args.data_dir or data_cfg.get("data_dir", "data/hit_full")),
        batch_size=int(cfg.get("training", {}).get("batch_size", 16)),
        augment_train=False,
        device=device,
    )
    model_name = model_cfg.pop("name", "ct_bigru")
    model = build_model(model_name, **model_cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    results = evaluate_noise(model, test_loader, device, noise_cfg.get("snr_levels", [-5, 0, 5, 10, 15, 20]))
    output_dir = ensure_dir(args.output_dir)
    save_json({"noise_accuracy": results}, output_dir / "noise_robustness.json")


if __name__ == "__main__":
    main()

