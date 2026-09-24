"""
train_correction.py - Training Pipeline for Autocall Correction MLP Surrogate.

Features:
  - Learns the residual delta_v = mc_npv - pde_npv
  - Z-score normalization for both input (13 features) and output (1 residual)
  - Unbounded output (no sigmoid)
  - Basis point error tracking for total NPV RMSE
  - AdamW + CosineAnnealingLR
"""

import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Tuple

from deepvol.surrogates.correction_mlp import CorrectionMLP


class CorrectionInputNormalizer:
    """Z-score standardizer for the 13-dimensional Autocall Correction input vector."""
    FEATURE_NAMES: List[str] = [
        "kappa", "theta", "sigma", "rho", "v0", "B", "coupon", "T", "n_obs_per_year", "r",
        "pde_npv", "pde_delta", "pde_gamma"
    ]

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "CorrectionInputNormalizer":
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean = np.mean(X_arr, axis=0).astype(np.float32)
        self.std = np.std(X_arr, axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionInputNormalizer is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        return ((X_arr - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionInputNormalizer is not fitted yet.")
        X_arr = np.asarray(X_norm, dtype=np.float32)
        return (X_arr * self.std + self.mean).astype(np.float32)

    def to_tensor(self, X: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
        transformed = self.transform(X)
        return torch.tensor(transformed, dtype=torch.float32, device=device)

    def transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionInputNormalizer is not fitted yet.")
        mean_t = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std_t = torch.tensor(self.std, dtype=t.dtype, device=t.device)
        return (t - mean_t) / std_t

    def inverse_transform_tensor(self, t_norm: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionInputNormalizer is not fitted yet.")
        mean_t = torch.tensor(self.mean, dtype=t_norm.dtype, device=t_norm.device)
        std_t = torch.tensor(self.std, dtype=t_norm.dtype, device=t_norm.device)
        return t_norm * std_t + mean_t

    def save(self, path: str) -> None:
        if self.mean is None or self.std is None:
            raise ValueError("Cannot save an unfitted normalizer.")
        np.savez_compressed(
            path,
            mean=self.mean,
            std=self.std,
            feature_names=np.array(self.FEATURE_NAMES),
        )

    @classmethod
    def load(cls, path: str) -> "CorrectionInputNormalizer":
        data = np.load(path)
        inst = cls()
        inst.mean = np.asarray(data["mean"], dtype=np.float32)
        inst.std = np.asarray(data["std"], dtype=np.float32)
        return inst


class CorrectionOutputNormalizer:
    """Z-score normalizer for the residual_npv (which can be negative)."""
    TARGET_NAMES: List[str] = ["residual_npv"]

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, Y: np.ndarray) -> "CorrectionOutputNormalizer":
        Y_arr = np.asarray(Y, dtype=np.float64)
        self.mean = np.mean(Y_arr, axis=0).astype(np.float32)
        self.std = np.std(Y_arr, axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionOutputNormalizer is not fitted yet.")
        Y_arr = np.asarray(Y, dtype=np.float32)
        return ((Y_arr - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, Y_norm: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionOutputNormalizer is not fitted yet.")
        Y_arr = np.asarray(Y_norm, dtype=np.float32)
        return (Y_arr * self.std + self.mean).astype(np.float32)

    def to_tensor(self, Y: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
        transformed = self.transform(Y)
        return torch.tensor(transformed, dtype=torch.float32, device=device)

    def transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionOutputNormalizer is not fitted yet.")
        mean_t = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std_t = torch.tensor(self.std, dtype=t.dtype, device=t.device)
        return (t - mean_t) / std_t

    def inverse_transform_tensor(self, t_norm: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionOutputNormalizer is not fitted yet.")
        mean_t = torch.tensor(self.mean, dtype=t_norm.dtype, device=t_norm.device)
        std_t = torch.tensor(self.std, dtype=t_norm.dtype, device=t_norm.device)
        return t_norm * std_t + mean_t

    def save(self, path: str) -> None:
        if self.mean is None or self.std is None:
            raise ValueError("Cannot save an unfitted normalizer.")
        np.savez_compressed(
            path,
            mean=self.mean,
            std=self.std,
            target_names=np.array(self.TARGET_NAMES),
        )

    @classmethod
    def load(cls, path: str) -> "CorrectionOutputNormalizer":
        data = np.load(path)
        inst = cls()
        inst.mean = np.asarray(data["mean"], dtype=np.float32)
        inst.std = np.asarray(data["std"], dtype=np.float32)
        return inst


class CorrectionDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        norm_in: CorrectionInputNormalizer,
        norm_out: CorrectionOutputNormalizer,
    ) -> None:
        data = np.load(npz_path)
        X = np.stack(
            [data[f] for f in CorrectionInputNormalizer.FEATURE_NAMES], axis=1
        ).astype(np.float32)
        
        # Residual is mc_npv - pde_npv
        residual = (data["npv"] - data["pde_npv"]).astype(np.float32).reshape(-1, 1)

        self.X = torch.tensor(norm_in.transform(X), dtype=torch.float32)
        self.Y = torch.tensor(norm_out.transform(residual), dtype=torch.float32)
        
        self.pde_npv = data["pde_npv"].astype(np.float64)
        self.mc_npv = data["npv"].astype(np.float64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        return self.X[idx], self.Y[idx], self.pde_npv[idx], self.mc_npv[idx]


def train(config: Dict[str, Any]) -> CorrectionMLP:
    device_str = config.get("device_str", "cuda")
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Training CorrectionMLP surrogate on {device}...")

    # Load data for fitting normalizers
    train_data = np.load(config["train_path"])
    X_train = np.stack(
        [train_data[f] for f in CorrectionInputNormalizer.FEATURE_NAMES], axis=1
    )
    Y_train = (train_data["npv"] - train_data["pde_npv"]).reshape(-1, 1)

    norm_in = CorrectionInputNormalizer().fit(X_train)
    norm_out = CorrectionOutputNormalizer().fit(Y_train)

    os.makedirs(os.path.dirname(config["norm_in_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["norm_out_path"]), exist_ok=True)
    norm_in.save(config["norm_in_path"])
    norm_out.save(config["norm_out_path"])

    # Datasets
    train_ds = CorrectionDataset(config["train_path"], norm_in, norm_out)
    val_ds = CorrectionDataset(config["val_path"], norm_in, norm_out)

    num_workers = config.get("num_workers", 2 if (os.cpu_count() or 1) > 2 else 0)
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        pin_memory=(device.type == "cuda"), num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        pin_memory=(device.type == "cuda"), num_workers=num_workers,
    )

    model = CorrectionMLP(
        in_dim=13,
        hidden=config["hidden"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["n_epochs"], eta_min=1e-6
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state_dict = None
    patience = config["patience"]
    patience_counter = 0

    print(f"Starting training for {config['n_epochs']} epochs (patience={patience})...")

    for epoch in range(1, config["n_epochs"] + 1):
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        train_count = 0

        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            optimizer.step()

            train_loss += loss.detach() * len(batch_x)
            train_count += len(batch_x)

        train_loss = float(train_loss.item()) / max(1, train_count)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_pde = []
        all_mc = []

        with torch.no_grad():
            for batch_x, batch_y, pde_npv, mc_npv in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)
                
                delta_v_pred = norm_out.inverse_transform_tensor(preds)
                all_preds.append(delta_v_pred.cpu().numpy().flatten())
                all_pde.append(pde_npv.numpy())
                all_mc.append(mc_npv.numpy())

        val_loss /= len(val_ds)
        
        all_preds = np.concatenate(all_preds)
        all_pde = np.concatenate(all_pde)
        all_mc = np.concatenate(all_mc)

        total_pred = all_pde + all_preds
        val_rmse_bps = np.sqrt(np.mean((total_pred - all_mc)**2)) * 10000
        
        residual_actual = all_mc - all_pde
        residual_rmse_bps = np.sqrt(np.mean((all_preds - residual_actual)**2)) * 10000

        if epoch % 5 == 0 or epoch == 1 or val_rmse_bps < best_val_rmse:
            print(
                f"Epoch {epoch:3d}/{config['n_epochs']} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val RMSE: {val_rmse_bps:.2f} bps | Res RMSE: {residual_rmse_bps:.2f} bps"
            )

        if val_rmse_bps < best_val_rmse:
            best_val_rmse = val_rmse_bps
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

DEFAULT_CORRECTION_CONFIG = {
    'train_path': 'data/autocall/train_100k_sobol_pde.npz',
    'val_path': 'data/autocall/val_10k_sobol_pde.npz',
    'n_epochs': 250, 'batch_size': 256, 'hidden': 256, 'n_layers': 4,
    'dropout': 0.0, 'lr': 3e-4, 'weight_decay': 1e-5, 'patience': 40,
    'device_str': 'cuda',
    'weights_save_path': 'artifacts/weights/autocall_correction_mlp.pth',
    'norm_in_path': 'artifacts/scalers/correction_input_normalizer.npz',
    'norm_out_path': 'artifacts/scalers/correction_output_normalizer.npz',
}

if __name__ == "__main__":
    os.makedirs("artifacts/weights", exist_ok=True)
    os.makedirs("artifacts/scalers", exist_ok=True)
    
    if os.path.exists(DEFAULT_CORRECTION_CONFIG['train_path']) and os.path.exists(DEFAULT_CORRECTION_CONFIG['val_path']):
        train(DEFAULT_CORRECTION_CONFIG)
    else:
        print(f"Train/Val data not found. Skipping training.")
