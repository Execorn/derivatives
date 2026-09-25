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
    """Z-score standardizer for the 19-dimensional Autocall Correction input vector."""
    FEATURE_NAMES: List[str] = [
        # 10 base params
        "kappa", "theta", "sigma", "rho", "v0", "B", "coupon", "T", "n_obs_per_year", "r",
        # 3 PDE outputs
        "pde_npv", "pde_delta", "pde_gamma",
        # 6 barrier-aware derived features
        "log_moneyness", "sigma_adj_dist", "barrier_vol_interaction",
        "leverage_skew", "vov_impact", "pde_gamma_v0_ratio",
    ]

    @staticmethod
    def compute_derived_features(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Compute barrier-aware derived features from raw parameters.

        Derived features:
          1. log_moneyness = -ln(B) = ln(S0/B) with S0=1
          2. sigma_adj_dist = -ln(B) / (sqrt(v0) * sqrt(T))
          3. barrier_vol_interaction = B * sqrt(v0)
          4. leverage_skew = rho * sigma * sqrt(T)
          5. vov_impact = sigma^2 * T
          6. pde_gamma_v0_ratio = (pde_gamma * v0) / max(|pde_npv|, 1e-6)
        """
        v0 = np.asarray(data["v0"], dtype=np.float32)
        B = np.asarray(data["B"], dtype=np.float32)
        T = np.asarray(data["T"], dtype=np.float32)
        sigma = np.asarray(data["sigma"], dtype=np.float32)
        rho = np.asarray(data["rho"], dtype=np.float32)
        pde_gamma = np.asarray(data["pde_gamma"], dtype=np.float32)
        pde_npv = np.asarray(data["pde_npv"], dtype=np.float32)

        sqrt_v0 = np.sqrt(np.maximum(v0, 1e-8))
        sqrt_T = np.sqrt(np.maximum(T, 1e-8))

        log_moneyness = -np.log(np.maximum(B, 1e-8))
        sigma_adj_dist = log_moneyness / (sqrt_v0 * sqrt_T)
        barrier_vol_interaction = B * sqrt_v0
        leverage_skew = rho * sigma * sqrt_T
        vov_impact = (sigma ** 2) * T
        pde_gamma_v0_ratio = (pde_gamma * v0) / np.maximum(np.abs(pde_npv), 1e-6)

        return {
            "log_moneyness": log_moneyness.astype(np.float32),
            "sigma_adj_dist": sigma_adj_dist.astype(np.float32),
            "barrier_vol_interaction": barrier_vol_interaction.astype(np.float32),
            "leverage_skew": leverage_skew.astype(np.float32),
            "vov_impact": vov_impact.astype(np.float32),
            "pde_gamma_v0_ratio": pde_gamma_v0_ratio.astype(np.float32),
        }

    @classmethod
    def build_feature_matrix(cls, data: Dict[str, Any]) -> np.ndarray:
        """Construct the 19-dimensional feature matrix from data dictionary."""
        derived = cls.compute_derived_features(data)
        feature_dict = {f: np.asarray(data[f], dtype=np.float32) for f in cls.FEATURE_NAMES if f in data}
        feature_dict.update(derived)
        return np.stack([feature_dict[f] for f in cls.FEATURE_NAMES], axis=1).astype(np.float32)

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
        alpha: float = 10.0,
        beta: float = 5.0,
    ) -> None:
        data = np.load(npz_path)
        X = CorrectionInputNormalizer.build_feature_matrix(data)

        # Residual is mc_npv - pde_npv
        residual = (data["npv"] - data["pde_npv"]).astype(np.float32).reshape(-1, 1)

        self.X = torch.tensor(norm_in.transform(X), dtype=torch.float32)
        self.Y = torch.tensor(norm_out.transform(residual), dtype=torch.float32)

        self.pde_npv = data["pde_npv"].astype(np.float64)
        self.mc_npv = data["npv"].astype(np.float64)

        # Barrier-proximity weights: w_i = 1 + alpha * exp(-beta * |ln(B_i)|)
        B_arr = np.maximum(np.asarray(data["B"], dtype=np.float32), 1e-8)
        weights = 1.0 + alpha * np.exp(-beta * np.abs(np.log(B_arr)))
        self.weights = torch.tensor(weights, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, float, torch.Tensor]:
        return self.X[idx], self.Y[idx], self.pde_npv[idx], self.mc_npv[idx], self.weights[idx]


def train(config: Dict[str, Any]) -> CorrectionMLP:
    from deepvol.utils.gpu_lock import acquire_gpu_lock
    acquire_gpu_lock()

    device_str = config.get("device_str", "cuda")
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Training CorrectionMLP surrogate on {device}...")

    # Load data for fitting normalizers
    train_data = np.load(config["train_path"])
    X_train = CorrectionInputNormalizer.build_feature_matrix(train_data)
    Y_train = (train_data["npv"] - train_data["pde_npv"]).reshape(-1, 1)

    norm_in = CorrectionInputNormalizer().fit(X_train)
    norm_out = CorrectionOutputNormalizer().fit(Y_train)

    os.makedirs(os.path.dirname(config["norm_in_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["norm_out_path"]), exist_ok=True)
    norm_in.save(config["norm_in_path"])
    norm_out.save(config["norm_out_path"])

    # Datasets with barrier proximity weighting
    alpha = config.get("barrier_weight_alpha", 10.0)
    beta = config.get("barrier_weight_beta", 5.0)
    train_ds = CorrectionDataset(config["train_path"], norm_in, norm_out, alpha=alpha, beta=beta)
    val_ds = CorrectionDataset(config["val_path"], norm_in, norm_out, alpha=alpha, beta=beta)

    torch.set_float32_matmul_precision("high")

    num_workers = config.get("num_workers", 2 if (os.cpu_count() or 1) > 2 else 0)
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        pin_memory=(device.type == "cuda"), num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        pin_memory=(device.type == "cuda"), num_workers=num_workers,
    )

    in_dim = config.get("in_dim", 19)
    model = CorrectionMLP(
        in_dim=in_dim,
        hidden=config["hidden"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    ).to(device)

    best_val_rmse = float("inf")
    best_state_dict = None

    # Checkpoint resumption if available
    weights_path = config.get("weights_save_path")
    if config.get("resume", True) and weights_path and os.path.exists(weights_path):
        try:
            saved_weights = torch.load(weights_path, map_location=device, weights_only=True)
            if "in_proj.0.weight" in saved_weights and saved_weights["in_proj.0.weight"].shape[1] == in_dim:
                model.load_state_dict(saved_weights)
                print(f"Resumed checkpoint from {weights_path}")
                # Evaluate resumed model to establish baseline best_val_rmse
                model.eval()
                all_preds = []
                all_pde = []
                all_mc = []
                with torch.no_grad():
                    for batch_x, batch_y, pde_npv, mc_npv, _ in val_loader:
                        batch_x = batch_x.to(device, non_blocking=True)
                        preds = model(batch_x)
                        delta_v_pred = norm_out.inverse_transform_tensor(preds)
                        all_preds.append(delta_v_pred.cpu().numpy().flatten())
                        all_pde.append(pde_npv.numpy())
                        all_mc.append(mc_npv.numpy())
                all_preds = np.concatenate(all_preds)
                all_pde = np.concatenate(all_pde)
                all_mc = np.concatenate(all_mc)
                total_pred = all_pde + all_preds
                best_val_rmse = float(np.sqrt(np.mean((total_pred - all_mc) ** 2)) * 10000)
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"Resumed checkpoint baseline Val RMSE: {best_val_rmse:.2f} bps")
        except Exception as e:
            print(f"Could not resume weights: {e}, starting from scratch.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["n_epochs"], eta_min=1e-6
    )
    criterion = nn.MSELoss()

    patience = config["patience"]
    patience_counter = 0

    lambda_smooth = config.get("lambda_smooth", 0.0)
    lambda_mono = config.get("lambda_mono", 0.0)

    print(
        f"Starting training for {config['n_epochs']} epochs "
        f"(patience={patience}, lambda_smooth={lambda_smooth}, lambda_mono={lambda_mono}, "
        f"alpha={alpha}, beta={beta})..."
    )

    for epoch in range(1, config["n_epochs"] + 1):
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        train_count = 0

        for batch_x, batch_y, _, _, batch_w in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_w = batch_w.to(device, non_blocking=True)

            optimizer.zero_grad()

            if lambda_smooth > 0 or lambda_mono > 0:
                batch_x.requires_grad_(True)
                preds = model._forward_uncompiled(batch_x)
                raw_loss = (preds - batch_y) ** 2
                loss = (raw_loss * batch_w).mean()

                jac = torch.autograd.grad(
                    preds.sum(), batch_x, create_graph=True
                )[0]  # shape (batch, 19)

                if lambda_smooth > 0:
                    # Select key financial columns: sigma=2, rho=3, v0=4, B=5, T=7
                    key_cols = [2, 3, 4, 5, 7]
                    jac_penalty = (jac[:, key_cols] ** 2).mean()
                    loss = loss + lambda_smooth * jac_penalty

                if lambda_mono > 0:
                    B_col = 5  # B column index in 19-dim input
                    grad_B = jac[:, B_col]
                    mono_penalty = torch.relu(grad_B).pow(2).mean()
                    loss = loss + lambda_mono * mono_penalty
            else:
                preds = model(batch_x)
                raw_loss = (preds - batch_y) ** 2
                loss = (raw_loss * batch_w).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            for batch_x, batch_y, pde_npv, mc_npv, _ in val_loader:
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
        val_rmse_bps = np.sqrt(np.mean((total_pred - all_mc) ** 2)) * 10000

        residual_actual = all_mc - all_pde
        residual_rmse_bps = np.sqrt(np.mean((all_preds - residual_actual) ** 2)) * 10000

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
    'n_epochs': 350,
    'batch_size': 256,
    'hidden': 256,
    'n_layers': 4,
    'dropout': 0.0,
    'lr': 2e-4,
    'weight_decay': 1e-5,
    'patience': 50,
    'device_str': 'cuda',
    'in_dim': 19,
    'lambda_smooth': 0.05,
    'lambda_mono': 0.05,
    'barrier_weight_alpha': 10.0,
    'barrier_weight_beta': 5.0,
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
