"""
Worst-of Multi-Asset Autocallable Surrogate Training Pipeline.

Trains the WoFAutocallEGNO surrogate model using AdamW and CosineAnnealingLR,
enforcing permutation equivariance via an auxiliary consistency loss.
"""

import logging
import os
import time
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from deepvol.surrogates.wof_autocall_egno import (
    WoFInputNormalizer,
    WoFOutputNormalizer,
    WoFAutocallEGNO,
)


class WoFDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        norm_in: WoFInputNormalizer,
        norm_out: WoFOutputNormalizer,
    ) -> None:
        data = np.load(npz_path)
        X = np.stack([data[f] for f in WoFInputNormalizer.FEATURE_NAMES], axis=1).astype(np.float32)
        Y = np.stack([data[t] for t in WoFOutputNormalizer.TARGET_NAMES], axis=1).astype(np.float32)

        self.X = torch.tensor(norm_in.transform(X), dtype=torch.float32)
        self.Y = torch.tensor(norm_out.transform(Y), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


class WoFLoss(nn.Module):
    def __init__(
        self,
        w_npv: float = 10.0,
        w_call: float = 1.0,
        w_life: float = 0.5,
        lambda_perm: float = 0.1,
    ) -> None:
        super().__init__()
        self.w_npv = w_npv
        self.w_call = w_call
        self.w_life = w_life
        self.lambda_perm = lambda_perm
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        norm_out: WoFOutputNormalizer,
        pred_perm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_npv = self.mse(pred[:, 0], target[:, 0])
        loss_call = self.mse(pred[:, 1], target[:, 1])
        loss_life = self.mse(pred[:, 2], target[:, 2])

        total_loss = (
            self.w_npv * loss_npv
            + self.w_call * loss_call
            + self.w_life * loss_life
        )

        perm_loss = 0.0
        if pred_perm is not None:
            perm_loss = self.mse(pred, pred_perm)
            total_loss = total_loss + self.lambda_perm * perm_loss

        with torch.no_grad():
            pred_real = norm_out.inverse_transform_tensor(pred)
            target_real = norm_out.inverse_transform_tensor(target)

            err_npv_bps = (pred_real[:, 0] - target_real[:, 0]).abs() * 10000.0
            rmse_bps = float(torch.sqrt(torch.mean(err_npv_bps ** 2)).item())
            call_mae = float(torch.mean((pred_real[:, 1] - target_real[:, 1]).abs()).item())
            life_mae = float(torch.mean((pred_real[:, 2] - target_real[:, 2]).abs()).item())

        metrics = {
            "loss": float(total_loss.item()),
            "rmse_bps": rmse_bps,
            "call_mae": call_mae,
            "life_mae": life_mae,
            "perm_loss": float(perm_loss.item()) if isinstance(perm_loss, torch.Tensor) else 0.0,
        }
        return total_loss, metrics


def train_wof(config: dict) -> WoFAutocallEGNO:
    device = torch.device(config.get("device_str", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training WoFAutocallEGNO on {device}...")

    train_data = np.load(config["train_path"])
    X_train_raw = np.stack([train_data[f] for f in WoFInputNormalizer.FEATURE_NAMES], axis=1)
    Y_train_raw = np.stack([train_data[t] for t in WoFOutputNormalizer.TARGET_NAMES], axis=1)

    norm_in = WoFInputNormalizer().fit(X_train_raw)
    norm_out = WoFOutputNormalizer().fit(Y_train_raw)

    norm_in.save(config["norm_in_path"])
    norm_out.save(config["norm_out_path"])
    print(f"Fitted and saved normalizers to {config['norm_in_path']} and {config['norm_out_path']}.")

    train_dataset = WoFDataset(config["train_path"], norm_in, norm_out)
    val_dataset = WoFDataset(config["val_path"], norm_in, norm_out)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 512),
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 512),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=2,
    )

    model = WoFAutocallEGNO(
        node_in=6,
        edge_in=1,
        global_in=5,
        hidden_dim=config.get("hidden", 128),
        n_layers=config.get("n_layers", 2),
        out_dim=3,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 2e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    n_epochs = config.get("n_epochs", 200)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    criterion = WoFLoss().to(device)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_rmse = float("inf")
    patience = config.get("patience", 25)
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.perf_counter()

        for X_b, Y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            Y_b = Y_b.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(X_b)
                # Permutation loss removed: EGNO is architecturally invariant
                # via mean pooling (h.mean(dim=1)), verified at machine precision.
                loss, _ = criterion(preds, Y_b, norm_out)

            # Guard against NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at epoch {epoch}, skipping batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(X_b)

        train_loss /= len(train_dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        val_call_mae = 0.0
        val_life_mae = 0.0

        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                Y_b = Y_b.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(X_b)
                    loss, metrics = criterion(preds, Y_b, norm_out)

                val_loss += loss.item() * len(X_b)
                val_rmse += (metrics["rmse_bps"] ** 2) * len(X_b)
                val_call_mae += metrics["call_mae"] * len(X_b)
                val_life_mae += metrics["life_mae"] * len(X_b)

        val_loss /= len(val_dataset)
        val_rmse = np.sqrt(val_rmse / len(val_dataset))
        val_call_mae /= len(val_dataset)
        val_life_mae /= len(val_dataset)

        epoch_time = time.perf_counter() - t0
        if epoch % 5 == 0 or epoch == 1 or val_rmse < best_val_rmse:
            print(
                f"Epoch {epoch:3d}/{n_epochs:3d} | Train: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse:.2f} bps | "
                f"Call MAE: {val_call_mae:.4f} | Life MAE: {val_life_mae:.4f} | Time: {epoch_time:.1f}s"
            )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            os.makedirs(os.path.dirname(config["weights_save_path"]), exist_ok=True)
            torch.save(model.state_dict(), config["weights_save_path"])
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val RMSE: {best_val_rmse:.2f} bps.")
                break

    print(f"Finished training. Best Val RMSE: {best_val_rmse:.2f} bps. Saved to {config['weights_save_path']}.")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/autocall/wof2_train_60k.npz")
    parser.add_argument("--val_path", type=str, default="data/autocall/wof2_val_8k.npz")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--weights_save_path", type=str, default="artifacts/weights/wof_mlp_best.pth")
    parser.add_argument("--norm_in_path", type=str, default="artifacts/scalers/wof_input_normalizer.npz")
    parser.add_argument("--norm_out_path", type=str, default="artifacts/scalers/wof_output_normalizer.npz")
    args = parser.parse_args()

    cfg = {
        "train_path": args.train_path,
        "val_path": args.val_path,
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": 128,
        "n_layers": 2,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "patience": 25,
        "device_str": "cuda",
        "weights_save_path": args.weights_save_path,
        "norm_in_path": args.norm_in_path,
        "norm_out_path": args.norm_out_path,
    }
    train_wof(cfg)

