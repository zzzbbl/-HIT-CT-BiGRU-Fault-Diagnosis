from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

from hit_ct_bigru.config import load_config
from hit_ct_bigru.utils import ensure_dir


ABLATION_MODELS = {
    "1dcnn_transformer": "cnn_transformer",
    "1dcnn_bigru": "cnn_bigru",
    "transformer_bigru": "transformer_bigru",
    "ct_bigru": "ct_bigru",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run module ablation experiments.")
    parser.add_argument("--config", default="configs/hit_ablation.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = ensure_dir(cfg.get("training", {}).get("output_dir", "outputs/ablation"))
    for experiment_name, model_name in ABLATION_MODELS.items():
        run_cfg = copy.deepcopy(cfg)
        run_cfg.setdefault("model", {})["name"] = model_name
        run_cfg.setdefault("training", {})["output_dir"] = str(output_root / experiment_name)
        tmp_cfg = output_root / f"{experiment_name}.yaml"
        tmp_cfg.write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
        cmd = [sys.executable, "src/train.py", "--config", str(tmp_cfg)]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

