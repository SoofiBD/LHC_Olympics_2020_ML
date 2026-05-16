"""CLI entry-point: evaluate a trained model on the LHC Olympics 2020 data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Allow running as `python scripts/evaluate.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import LHCDataset, SyntheticLHCDataset, build_dataloaders
from src.utils.config import load_config, get_model
from src.analysis.plotting import (
    plot_anomaly_scores,
    plot_roc_curve,
    plot_mass_distribution,
)
from src.analysis.bump_hunt import bump_hunt


def _sanitize_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _model_short_name(model_type: str) -> str:
    mapping = {
        "autoencoder": "AE",
        "classifier": "CLF",
        "part_autoencoder": "parT_AE",
        "part_classifier": "parT_CLF",
    }
    return mapping.get(model_type.lower(), model_type)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained model.")
    p.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to model checkpoint (.pt file)")
    p.add_argument("--config", type=Path, default=None,
                    help="Path to YAML config file")
    p.add_argument("--data", type=Path, default=None,
                    help="Path to HDF5 data file. If not given, synthetic data is used.")
    p.add_argument("--output", type=Path, default=Path("outputs"),
                    help="Output directory for figures and results")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--model-type", type=str, default=None,
                    choices=["autoencoder", "classifier", "part_autoencoder", "part_classifier"])
    p.add_argument("--tag", type=str, default=None,
                    help="Optional tag appended to output figure filenames")
    return p.parse_args()


def evaluate_autoencoder(
    model: nn.Module,
    dataloader,
    device: torch.device,
    figures_dir: Path,
    eval_tag: str,
) -> None:
    """Evaluate an autoencoder by computing reconstruction loss as anomaly score."""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            x_hat, _ = model(x)
            # Per-sample MSE: anomaly score
            mse = ((x_hat - x) ** 2).mean(dim=1)
            all_scores.append(mse.cpu().numpy())
            all_labels.append(y.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    has_bg = np.any(labels == 0)
    has_sig = np.any(labels == 1)
    labeled = bool(has_bg and has_sig)

    plot_anomaly_scores(
        scores,
        labels if labeled else None,
        figures_dir / f"anomaly_scores_{eval_tag}.png",
    )
    print(f"Anomaly score plot saved to {figures_dir / f'anomaly_scores_{eval_tag}.png'}")

    if labeled:
        auc_val = plot_roc_curve(labels, scores, figures_dir / f"roc_curve_{eval_tag}.png")
        print(f"ROC curve saved. AUC = {auc_val:.4f}")
    else:
        present = "background" if has_bg else "signal" if has_sig else "(none)"
        print(
            f"ROC skipped: dataset has only one class present ({present}). "
            "Provide labeled bg+sig data to compute AUC."
        )

    result = bump_hunt(scores)
    print(f"\nBump Hunt Results:")
    print(f"  Z-score:             {result.z_score}")
    print(f"  p-value:             {result.p_value}")
    print(f"  Signal count:        {result.signal_count}")
    print(f"  Background estimate: {result.background_estimate}")


def evaluate_classifier(
    model: nn.Module,
    dataloader,
    device: torch.device,
    figures_dir: Path,
    eval_tag: str,
) -> None:
    """Evaluate a classifier by computing ROC/AUC."""
    model.eval()
    all_probs = []
    all_labels = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            logits = model(x)
            probs = softmax(logits)[:, 1]  # probability of signal class
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    has_bg = np.any(labels == 0)
    has_sig = np.any(labels == 1)
    labeled = bool(has_bg and has_sig)

    if labeled:
        auc_val = plot_roc_curve(labels, probs, figures_dir / f"roc_curve_{eval_tag}.png")
        print(f"ROC curve saved. AUC = {auc_val:.4f}")
    else:
        present = "background" if has_bg else "signal" if has_sig else "(none)"
        print(
            f"ROC skipped: dataset has only one class present ({present}). "
            "Provide labeled bg+sig data to compute AUC."
        )

    plot_anomaly_scores(
        probs,
        labels if labeled else None,
        figures_dir / f"classifier_scores_{eval_tag}.png",
        title="Classifier Score Distribution",
    )
    print(f"Score distribution saved to {figures_dir / f'classifier_scores_{eval_tag}.png'}")


def main() -> None:
    args = parse_args()

    if args.config is not None:
        cfg = load_config(args.config)
    else:
        cfg = {}

    model_cfg = cfg.get("model", {})
    model_type = args.model_type or model_cfg.get("type", "autoencoder")
    input_dim = model_cfg.get("input_dim", 128)

    if args.data is not None:
        print(f"Loading data from {args.data} ...")
        dataset = LHCDataset(args.data)
        input_dim = dataset.input_dim
    else:
        print("No data file specified — using synthetic dataset for evaluation.")
        dataset = SyntheticLHCDataset(n_samples=5_000, input_dim=input_dim)

    _, eval_loader = build_dataloaders(
        dataset, batch_size=512, val_fraction=1.0, seed=cfg.get("seed", 42)
    )

    cfg.setdefault("model", {})
    cfg["model"]["type"] = model_type
    cfg["model"]["input_dim"] = input_dim
    model = get_model(cfg)

    device = torch.device(args.device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    figures_dir = args.output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_tag = args.data.stem if args.data is not None else "synthetic"
    checkpoint_tag = args.checkpoint.stem
    default_tag = f"{_model_short_name(model_type)}_{data_tag}_{checkpoint_tag}"
    eval_tag = _sanitize_token(args.tag or default_tag)

    if model_type == "autoencoder":
        evaluate_autoencoder(model, eval_loader, device, figures_dir, eval_tag)
    else:
        evaluate_classifier(model, eval_loader, device, figures_dir, eval_tag)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
