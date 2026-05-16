"""Training loop and validation for LHC Olympics 2020 models.

Supports both autoencoder (MSE loss) and classifier (CrossEntropy loss)
training through a unified interface.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from time import perf_counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    """All hyper-parameters needed by the training loop."""
    batch_size: int = 512
    lr: float = 1e-3
    epochs: int = 10
    seed: int = 42
    device: str = "cpu"
    model_type: str = "autoencoder"  # "autoencoder" | "classifier" | "part_autoencoder" | "part_classifier"


def _artifact_name(base_name: str, prefix: Optional[str] = None) -> str:
    """Build artifact filenames with optional run prefix.

    Examples
    --------
    - base_name='best_model.pt' -> 'best_model.pt'
    - base_name='best_model.pt', prefix='partAE_ep20' -> 'best_model_partAE_ep20.pt'
    """
    if not prefix:
        return base_name
    stem, suffix = base_name.rsplit(".", 1)
    return f"{stem}_{prefix}.{suffix}"



def _save_loss_log(
    log: List[Dict[str, float]], path: Path
) -> None:
    """Write per-epoch train/val loss to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log)


def _format_duration(seconds: float) -> str:
    """Format a duration in HH:MM:SS."""
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"



def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    model_type: str = "autoencoder",
) -> float:
    """Run one validation pass and return the mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if model_type in ("autoencoder", "part_autoencoder"):
                x = batch[0].to(device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(device, non_blocking=True)
                x_hat, _ = model(x)
                loss = criterion(x_hat, x)
            else:  # classifier
                x, y = batch
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)



def train(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    output_dir: Path,
    artifact_prefix: Optional[str] = None,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """Full training loop with checkpointing and logging.

    Parameters
    ----------
    model : nn.Module
        Model to train (SimpleAutoencoder or MLPClassifier).
    train_loader, val_loader : DataLoader
        Training and validation data loaders.
    config : TrainConfig
        Training hyper-parameters.
    output_dir : Path
        Where to save training artifacts.
    artifact_prefix : str | None
        Optional suffix injected into saved artifact names to avoid overwriting.

    Returns
    -------
    model : nn.Module
        The trained model (best checkpoint loaded).
    loss_log : list[dict]
        Per-epoch train/val losses.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_name = _artifact_name("best_model.pt", artifact_prefix)
    final_model_name = _artifact_name("final_model.pt", artifact_prefix)
    loss_log_name = _artifact_name("loss_log.csv", artifact_prefix)

    best_model_path = output_dir / best_model_name
    final_model_path = output_dir / final_model_name
    loss_log_path = output_dir / loss_log_name

    device = torch.device(config.device)
    model = model.to(device)

    if config.model_type in ("autoencoder", "part_autoencoder"):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    loss_log: List[Dict[str, float]] = []
    training_start = perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start = perf_counter()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if config.model_type in ("autoencoder", "part_autoencoder"):
                x = batch[0].to(device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(device, non_blocking=True)
                x_hat, _ = model(x)
                loss = criterion(x_hat, x)
            else:
                x, y = batch
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(n_batches, 1)

        avg_val_loss = validate(
            model, val_loader, criterion, device, model_type=config.model_type
        )

        loss_log.append(
            {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
        )

        epoch_elapsed = perf_counter() - epoch_start
        total_elapsed = perf_counter() - training_start
        completed_epochs = epoch
        avg_epoch_time = total_elapsed / max(completed_epochs, 1)
        remaining_epochs = max(config.epochs - epoch, 0)
        eta_seconds = avg_epoch_time * remaining_epochs

        print(
            f"Epoch {epoch:>3d}/{config.epochs}  |  "
            f"train_loss: {avg_train_loss:.6f}  |  "
            f"val_loss: {avg_val_loss:.6f}  |  "
            f"epoch_time: {_format_duration(epoch_elapsed)}  |  "
            f"elapsed: {_format_duration(total_elapsed)}  |  "
            f"ETA: {_format_duration(eta_seconds)}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    torch.save(model.state_dict(), final_model_path)

    _save_loss_log(loss_log, loss_log_path)

    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to {best_model_path}")

    return model, loss_log
