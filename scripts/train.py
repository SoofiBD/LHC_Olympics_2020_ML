"""CLI entry-point: train a model for LHC Olympics 2020."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Allow running as `python scripts/train.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import LHCDataset, SyntheticLHCDataset, build_dataloaders
from src.training.trainer import TrainConfig, train
from src.utils.config import load_config, get_model


def _sanitize_token(value: str) -> str:
    """Keep artifact names filesystem-safe and concise."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _model_short_name(model_type: str) -> str:
    mapping = {
        "autoencoder": "AE",
        "classifier": "CLF",
        "part_autoencoder": "parT_AE",
        "part_classifier": "parT_CLF",
    }
    return mapping.get(model_type.lower(), model_type)


def _build_run_tag(
    *,
    model_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    user_run_name: str | None,
) -> str:
    if user_run_name:
        return _sanitize_token(user_run_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr_token = f"{lr:.0e}" if lr < 0.01 else str(lr).replace(".", "p")
    raw = (
        f"{_model_short_name(model_type)}_ep{epochs}_bs{batch_size}_"
        f"lr{lr_token}_seed{seed}_{device}_{timestamp}"
    )
    return _sanitize_token(raw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a model for LHC Olympics 2020.")
    p.add_argument("--config", type=Path, default=None,
                    help="Path to YAML config file (e.g. configs/config.yaml)")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--data", type=Path, default=None,
                    help="Path to HDF5 data file. If not given, synthetic data is used.")
    p.add_argument("--model-type", type=str, default=None,
                    choices=["autoencoder", "classifier", "part_autoencoder", "part_classifier"])
    p.add_argument("--run-name", type=str, default=None,
                    help="Optional run name used in output artifact filenames")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.config is not None:
        cfg = load_config(args.config)
    else:
        cfg = {}

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    seed = int(cfg.get("seed", 42))
    output_root = cfg.get("outputs", {}).get("root", "outputs")

    model_type = args.model_type or model_cfg.get("type", "autoencoder")
    batch_size = args.batch_size or train_cfg.get("batch_size", 512)
    lr = args.lr or train_cfg.get("lr", 1e-3)
    epochs = args.epochs or train_cfg.get("epochs", 10)
    device = args.device or train_cfg.get("device", "cpu")
    output_base = args.output or Path(output_root)

    run_tag = _build_run_tag(
        model_type=model_type,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        seed=seed,
        device=str(device),
        user_run_name=args.run_name,
    )

    models_dir = Path(output_base) / "models"
    figures_dir = Path(output_base) / "figures"
    logs_dir = Path(output_base) / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.data is not None:
        print(f"Loading data from {args.data} ...")
        dataset = LHCDataset(args.data)
        input_dim = dataset.input_dim
    else:
        print("No data file specified — using synthetic dataset for demonstration.")
        input_dim = model_cfg.get("input_dim", 128)
        dataset = SyntheticLHCDataset(n_samples=10_000, input_dim=input_dim)

    train_loader, val_loader = build_dataloaders(
        dataset, batch_size=batch_size, seed=seed
    )

    cfg.setdefault("model", {})
    cfg["model"]["type"] = model_type
    cfg["model"]["input_dim"] = input_dim
    model = get_model(cfg)

    effective_cfg = copy.deepcopy(cfg)
    effective_cfg.setdefault("train", {})
    effective_cfg.setdefault("outputs", {})
    effective_cfg["train"].update(
        {
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "device": device,
        }
    )
    effective_cfg["train"]["seed"] = seed
    effective_cfg["model"]["type"] = model_type
    effective_cfg["model"]["input_dim"] = input_dim
    effective_cfg["outputs"]["root"] = str(output_base)

    print(f"Model: {model.__class__.__name__}  |  input_dim={input_dim}")
    print(f"Training: epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}")

    tc = TrainConfig(
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        device=device,
        model_type=model_type,
    )

    trained_model, loss_log = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=tc,
        output_dir=models_dir,
        artifact_prefix=run_tag,
    )

    from src.analysis.plotting import plot_loss_curves

    train_losses = [e["train_loss"] for e in loss_log]
    val_losses = [e["val_loss"] for e in loss_log]
    loss_curve_path = figures_dir / f"loss_curves_{run_tag}.png"
    plot_loss_curves(train_losses, val_losses, loss_curve_path)
    print(f"Loss curves saved to {loss_curve_path}")

    run_meta = {
        "run_tag": run_tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_type": model_type,
        "model": model_cfg,
        "train": {
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "device": device,
            "seed": seed,
        },
        "data": str(args.data) if args.data is not None else "synthetic",
        "config": str(args.config) if args.config is not None else None,
        "artifacts": {
            "best_model": str(models_dir / f"best_model_{run_tag}.pt"),
            "final_model": str(models_dir / f"final_model_{run_tag}.pt"),
            "loss_log": str(models_dir / f"loss_log_{run_tag}.csv"),
            "loss_curve": str(loss_curve_path),
        },
    }
    run_meta_path = logs_dir / f"run_meta_{run_tag}.json"
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run metadata saved to {run_meta_path}")

    effective_cfg_path = logs_dir / f"config_effective_{run_tag}.yaml"
    with open(effective_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(effective_cfg, f, sort_keys=False)
    print(f"Effective config saved to {effective_cfg_path}")


if __name__ == "__main__":
    main()
