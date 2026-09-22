"""
train_autocall.py — Training Pipeline for Autocall Residual MLP Surrogate.

Features:
  - Multi-objective weighted loss (NPV, early call prob, expected life)
  - Basis point error tracking (1 bp = 0.01% = 0.0001)
  - AdamW optimizer + Cosine Annealing learning rate schedule
  - Early stopping on validation RMSE (bps)
  - Normalizer persistence (.npz) alongside model weights (.pth)
"""

import logging
import os
import time
from typing import Dict, Any, Tuple
logger = logging.getLogger(__name__)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from deepvol.surrogates.autocall_normalizer import (
    AutocallInputNormalizer,
    AutocallOutputNormalizer,
)
from deepvol.surrogates.autocall_mlp import AutocallMLP


class AutocallDataset(Dataset):
    """PyTorch Dataset yielding pre-normalized input and target tensors."""

    def __init__(
        self,
        npz_path: str,
        norm_in: AutocallInputNormalizer,
        norm_out: AutocallOutputNormalizer,
    ) -> None:
        data = np.load(npz_path)
        X = np.stack(
            [data[f] for f in AutocallInputNormalizer.FEATURE_NAMES], axis=1
        ).astype(np.float32)
        Y = np.stack(
            [data["npv"], data["call_prob"], data["exp_life"]], axis=1
        ).astype(np.float32)

        self.X = torch.tensor(norm_in.transform(X), dtype=torch.float32)
        self.Y = torch.tensor(norm_out.transform(Y), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


class AutocallLoss(nn.Module):
    """
    Multi-objective loss function with unnormalized basis point error metrics.
    """

    def __init__(
        self, w_npv: float = 1.0, w_call: float = 0.1, w_life: float = 0.1
    ) -> None:
        super().__init__()
        self.w_npv = w_npv
        self.w_call = w_call
        self.w_life = w_life
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        norm_out: AutocallOutputNormalizer,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_npv = self.mse(pred[:, 0], target[:, 0])
        loss_call = self.mse(pred[:, 1], target[:, 1])
        loss_life = self.mse(pred[:, 2], target[:, 2])

        total_loss = self.w_npv * loss_npv + self.w_call * loss_call + self.w_life * loss_life

        with torch.no_grad():
            pred_real = norm_out.inverse_transform_tensor(pred)
            target_real = norm_out.inverse_transform_tensor(target)

            rmse_bps = float(
                torch.sqrt(torch.mean((pred_real[:, 0] - target_real[:, 0]) ** 2)).item()
                * 10000.0
            )
            call_mae = float(torch.mean(torch.abs(pred_real[:, 1] - target_real[:, 1])).item())
            life_mae = float(torch.mean(torch.abs(pred_real[:, 2] - target_real[:, 2])).item())

        metrics = {
            "rmse_bps": rmse_bps,
            "call_mae": call_mae,
            "life_mae": life_mae,
        }
        return total_loss, metrics


def train(config: Dict[str, Any]) -> AutocallMLP:
    """
    Train AutocallMLP surrogate with AdamW, CosineAnnealingLR, and early stopping.
    """
    device_str = config.get("device_str", "cuda")
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Training AutocallMLP surrogate on {device}...")

    # 1. Load train split and fit normalizers
    train_data = np.load(config["train_path"])
    X_train = np.stack(
        [train_data[f] for f in AutocallInputNormalizer.FEATURE_NAMES], axis=1
    )
    Y_train = np.stack(
        [train_data["npv"], train_data["call_prob"], train_data["exp_life"]], axis=1
    )

    norm_in = AutocallInputNormalizer().fit(X_train)
    norm_out = AutocallOutputNormalizer().fit(Y_train)

    os.makedirs(os.path.dirname(config["norm_in_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["norm_out_path"]), exist_ok=True)
    norm_in.save(config["norm_in_path"])
    norm_out.save(config["norm_out_path"])
    print(f"Saved normalizers to {config['norm_in_path']} and {config['norm_out_path']}")

    # 2. Datasets & Loaders
    train_ds = AutocallDataset(config["train_path"], norm_in, norm_out)
    val_ds = AutocallDataset(config["val_path"], norm_in, norm_out)

    num_workers = config.get("num_workers", 2 if (os.cpu_count() or 1) > 2 else 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
    )

    # 3. Model
    model = AutocallMLP(
        in_dim=10,
        hidden=config["hidden"],
        n_layers=config["n_layers"],
        out_dim=3,
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["n_epochs"], eta_min=1e-6
    )
    criterion = AutocallLoss()

    best_val_rmse = float("inf")
    best_state_dict = None
    patience = config["patience"]
    patience_counter = 0

    print(f"Starting training for {config['n_epochs']} epochs (patience={patience})...")

    for epoch in range(1, config["n_epochs"] + 1):
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        train_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss, metrics = criterion(preds, batch_y, norm_out)

            # Guard against NaN/Inf loss to prevent wasting compute
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at epoch {epoch}, skipping batch")
                continue

            loss.backward()
            optimizer.step()

            train_loss += loss.detach()
            train_count += len(batch_x)

        train_loss = float(train_loss.item()) / max(1, train_count)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        val_call_mae = 0.0
        val_life_mae = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                preds = model(batch_x)
                loss, metrics = criterion(preds, batch_y, norm_out)

                val_loss += loss.item() * len(batch_x)
                val_rmse += metrics["rmse_bps"] * len(batch_x)
                val_call_mae += metrics["call_mae"] * len(batch_x)
                val_life_mae += metrics["life_mae"] * len(batch_x)

        val_loss /= len(val_ds)
        val_rmse /= len(val_ds)
        val_call_mae /= len(val_ds)
        val_life_mae /= len(val_ds)

        if epoch % 5 == 0 or epoch == 1 or val_rmse < best_val_rmse:
            print(
                f"Epoch {epoch:3d}/{config['n_epochs']} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val RMSE: {val_rmse:.2f} bps | Call MAE: {val_call_mae:.4f} | Life MAE: {val_life_mae:.4f}"
            )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            os.makedirs(os.path.dirname(config["weights_save_path"]), exist_ok=True)
            torch.save(best_state_dict, config["weights_save_path"])
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (best val RMSE: {best_val_rmse:.2f} bps)")
                break

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    print("\n=======================================================")
    print(f"Training Complete! Best Val RMSE: {best_val_rmse:.2f} bps")
    print(f"Saved weights to: {config['weights_save_path']}")
    print("=======================================================\n")

    return model


if __name__ == "__main__":
    config = {
        "train_path": "data/autocall/train_100k.npz",
        "val_path": "data/autocall/val_10k.npz",
        "n_epochs": 200,
        "batch_size": 512,
        "hidden": 256,
        "n_layers": 5,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 25,
        "device_str": "cuda",
        "weights_save_path": "artifacts/weights/autocall_mlp_best.pth",
        "norm_in_path": "artifacts/scalers/autocall_input_normalizer.npz",
        "norm_out_path": "artifacts/scalers/autocall_output_normalizer.npz",
    }
    os.makedirs("artifacts/weights", exist_ok=True)
    os.makedirs("artifacts/scalers", exist_ok=True)
    train(config)
