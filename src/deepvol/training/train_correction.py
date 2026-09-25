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
from typing import Dict, Any, List, Optional, Tuple, Union

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
        self._cached_tensors: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}

    def fit(self, X: np.ndarray) -> "CorrectionInputNormalizer":
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean = np.mean(X_arr, axis=0).astype(np.float32)
        self.std = np.std(X_arr, axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        self._cached_tensors.clear()
        return self

    def get_mean_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        key = (str(device), dtype, "mean")
        if key not in self._cached_tensors:
            self._cached_tensors[key] = torch.tensor(self.mean, dtype=dtype, device=device)
        return self._cached_tensors[key]

    def get_std_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        key = (str(device), dtype, "std")
        if key not in self._cached_tensors:
            self._cached_tensors[key] = torch.tensor(self.std, dtype=dtype, device=device)
        return self._cached_tensors[key]

    @property
    def mean_t(self) -> torch.Tensor:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.get_mean_tensor(dev)

    @property
    def std_t(self) -> torch.Tensor:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.get_std_tensor(dev)

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
        mean_t = self.get_mean_tensor(t.device, t.dtype)
        std_t = self.get_std_tensor(t.device, t.dtype)
        return (t - mean_t) / std_t

    def inverse_transform_tensor(self, t_norm: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionInputNormalizer is not fitted yet.")
        mean_t = self.get_mean_tensor(t_norm.device, t_norm.dtype)
        std_t = self.get_std_tensor(t_norm.device, t_norm.dtype)
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
    """Z-score normalizer for the residual_npv with strict float64 pricing precision."""
    TARGET_NAMES: List[str] = ["residual_npv"]

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self._cached_tensors: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}

    def fit(self, Y: np.ndarray) -> "CorrectionOutputNormalizer":
        Y_arr = np.asarray(Y, dtype=np.float64)
        self.mean = np.mean(Y_arr, axis=0).astype(np.float32)
        self.std = np.std(Y_arr, axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        self._cached_tensors.clear()
        return self

    def get_mean_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        key = (str(device), dtype, "mean")
        if key not in self._cached_tensors:
            self._cached_tensors[key] = torch.tensor(self.mean, dtype=dtype, device=device)
        return self._cached_tensors[key]

    def get_std_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        key = (str(device), dtype, "std")
        if key not in self._cached_tensors:
            self._cached_tensors[key] = torch.tensor(self.std, dtype=dtype, device=device)
        return self._cached_tensors[key]

    @property
    def mean_t(self) -> torch.Tensor:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.get_mean_tensor(dev)

    @property
    def std_t(self) -> torch.Tensor:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.get_std_tensor(dev)

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionOutputNormalizer is not fitted yet.")
        Y_arr = np.asarray(Y, dtype=np.float32)
        return ((Y_arr - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, Y_norm: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionOutputNormalizer is not fitted yet.")
        Y_arr = np.asarray(Y_norm, dtype=np.float64)
        return (Y_arr * float(self.std[0]) + float(self.mean[0])).astype(np.float32)

    def to_tensor(self, Y: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
        transformed = self.transform(Y)
        return torch.tensor(transformed, dtype=torch.float32, device=device)

    def transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionOutputNormalizer is not fitted yet.")
        mean_t = self.get_mean_tensor(t.device, t.dtype)
        std_t = self.get_std_tensor(t.device, t.dtype)
        return (t - mean_t) / std_t

    def inverse_transform_tensor(self, t_norm: torch.Tensor, preserve_float64: bool = False) -> torch.Tensor:
        """
        Denormalize predictions. Complies with .agents/AGENTS.md mandatory float64 policy:
        Casts input to torch.float64, applies double precision scaling, and preserves float64
        when requested to prevent precision loss during pricing layer additions.
        """
        if self.mean is None or self.std is None:
            raise ValueError("CorrectionOutputNormalizer is not fitted yet.")
        mean_f64 = self.get_mean_tensor(t_norm.device, torch.float64)
        std_f64 = self.get_std_tensor(t_norm.device, torch.float64)
        t_f64 = t_norm.to(torch.float64)
        out_f64 = t_f64 * std_f64 + mean_f64
        return out_f64 if (preserve_float64 or t_norm.dtype == torch.float64) else out_f64.to(t_norm.dtype)

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


def compute_total_barrier_derivative(
    jac: torch.Tensor,
    raw_params_or_batch_x: Union[Dict[str, Any], torch.Tensor],
    norm_in: CorrectionInputNormalizer,
    norm_out: CorrectionOutputNormalizer,
) -> torch.Tensor:
    """
    Computes total derivative df_theta / dB via exact multivariable chain rule
    across all 19 normalized inputs and unnormalized targets.

    Features depending on B:
      u5  = B                         => du5/dB  = 1
      u13 = -ln(B)                    => du13/dB = -1/B
      u14 = -ln(B)/(sqrt(v0)*sqrt(T)) => du14/dB = -1/(B*sqrt(v0)*sqrt(T))
      u15 = B*sqrt(v0)                => du15/dB = sqrt(v0)

    Chain rule:
      df/du_j = (jac[:, j] / norm_in.std_t[j]) * norm_out.std_t[0]
      df/dB   = df/du_5 * 1 + df/du_13 * (-1/B) + df/du_14 * (-1/(B*sqrt(v0)*sqrt(T))) + df/du_15 * sqrt(v0)
    """
    device = jac.device
    dtype = jac.dtype

    if isinstance(raw_params_or_batch_x, dict):
        B = torch.clamp(torch.as_tensor(raw_params_or_batch_x["B"], dtype=dtype, device=device).flatten(), min=1e-6)
        v0 = torch.clamp(torch.as_tensor(raw_params_or_batch_x["v0"], dtype=dtype, device=device).flatten(), min=1e-6)
        T = torch.clamp(torch.as_tensor(raw_params_or_batch_x["T"], dtype=dtype, device=device).flatten(), min=1e-6)
    else:
        # Reconstruct unnormalized parameters directly from normalized batch_x on GPU
        bx = raw_params_or_batch_x
        mean_in = norm_in.get_mean_tensor(device, dtype)
        std_in = norm_in.get_std_tensor(device, dtype)
        B = torch.clamp(bx[:, 5] * std_in[5] + mean_in[5], min=1e-6)
        v0 = torch.clamp(bx[:, 4] * std_in[4] + mean_in[4], min=1e-6)
        T = torch.clamp(bx[:, 7] * std_in[7] + mean_in[7], min=1e-6)

    sqrt_v0 = torch.sqrt(v0)
    sqrt_T = torch.sqrt(T)

    std_in = norm_in.get_std_tensor(device, dtype)
    std_out = norm_out.get_std_tensor(device, dtype)[0]

    df_du5 = (jac[:, 5] / std_in[5]) * std_out
    df_du13 = (jac[:, 13] / std_in[13]) * std_out
    df_du14 = (jac[:, 14] / std_in[14]) * std_out
    df_du15 = (jac[:, 15] / std_in[15]) * std_out

    du5_dB = 1.0
    du13_dB = -1.0 / B
    du14_dB = -1.0 / (B * sqrt_v0 * sqrt_T)
    du15_dB = sqrt_v0

    df_dB = df_du5 * du5_dB + df_du13 * du13_dB + df_du14 * du14_dB + df_du15 * du15_dB
    return df_dB


class InVRAMDataLoader:
    """Zero-copy, zero-PCIe GPU resident data loader with preallocated buffers."""
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor, batch_size: int = 256):
        self.X = X.contiguous()
        self.Y = Y.contiguous()
        self.W = W.contiguous()
        self.batch_size = batch_size
        self.N = len(X)
        self.n_batches = (self.N + batch_size - 1) // batch_size
        self.perm = torch.empty(self.N, dtype=torch.long, device=X.device)
        self.X_shuffled = torch.empty_like(self.X)
        self.Y_shuffled = torch.empty_like(self.Y)
        self.W_shuffled = torch.empty_like(self.W)

    def __iter__(self):
        torch.randperm(self.N, out=self.perm, device=self.X.device)
        torch.index_select(self.X, 0, self.perm, out=self.X_shuffled)
        torch.index_select(self.Y, 0, self.perm, out=self.Y_shuffled)
        torch.index_select(self.W, 0, self.perm, out=self.W_shuffled)
        self.idx = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.idx >= self.N:
            raise StopIteration
        end = min(self.idx + self.batch_size, self.N)
        bx = self.X_shuffled[self.idx:end]
        by = self.Y_shuffled[self.idx:end]
        bw = self.W_shuffled[self.idx:end]
        self.idx = end
        return bx, by, bw

    def __len__(self) -> int:
        return self.n_batches

def train(config: Dict[str, Any]) -> CorrectionMLP:
    from deepvol.utils.gpu_lock import acquire_gpu_lock
    acquire_gpu_lock()

    device_str = config.get("device_str", "cuda")
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Training CorrectionMLP surrogate on {device} (Phase C+ Hardened)...")

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

    # Barrier proximity weights: w_i = 1 + alpha * exp(-beta * |ln(B_i)|)
    alpha = config.get("barrier_weight_alpha", 10.0)
    beta = config.get("barrier_weight_beta", 5.0)
    B_train = np.maximum(np.asarray(train_data["B"], dtype=np.float32), 1e-8)
    weights_train = 1.0 + alpha * np.exp(-beta * np.abs(np.log(B_train)))

    # In-VRAM GPU Resident Dataset (Zero-PCIe latency during training)
    X_train_norm = norm_in.transform(X_train)
    Y_train_norm = norm_out.transform(Y_train)
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train_norm, dtype=torch.float32, device=device)
    W_train_t = torch.tensor(weights_train, dtype=torch.float32, device=device).reshape(-1, 1)

    train_loader = InVRAMDataLoader(
        X_train_t, Y_train_t, W_train_t, batch_size=config["batch_size"]
    )

    # In-VRAM Validation Data (for fast GPU vectorized validation)
    val_data = np.load(config["val_path"])
    X_val = CorrectionInputNormalizer.build_feature_matrix(val_data)
    X_val_norm = norm_in.transform(X_val)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
    pde_val_t = torch.tensor(val_data["pde_npv"], dtype=torch.float64, device=device)
    mc_val_t = torch.tensor(val_data["npv"], dtype=torch.float64, device=device)

    torch.set_float32_matmul_precision("high")

    in_dim = config.get("in_dim", 19)
    model = CorrectionMLP(
        in_dim=in_dim,
        hidden=config["hidden"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    ).to(device)

    # Polyak-Ruppert Exponential Moving Average (EMA) to filter spatial autograd jitter
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    weights_save_path = config.get("weights_save_path")
    if config.get("resume", False) and weights_save_path and os.path.exists(weights_save_path):
        saved_weights = torch.load(weights_save_path, map_location=device, weights_only=True)
        model.load_state_dict(saved_weights)
        ema_model.module.load_state_dict(saved_weights)
        print(f"Resumed model weights from {weights_save_path} for fine-tuning.")

    best_val_rmse = float("inf")
    best_val_score = float("inf")
    best_state_dict = None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["n_epochs"], eta_min=1e-6
    )

    # Basis-point Adaptive Huber / Smooth L1 Loss (beta = 1.0 bp in normalized units)
    beta_normalized = (1.0 / 10000.0) / float(norm_out.std[0])
    criterion = nn.SmoothL1Loss(beta=beta_normalized, reduction="none")

    patience = config["patience"]
    patience_counter = 0

    lambda_smooth = config.get("lambda_smooth", 0.05)
    lambda_mono = config.get("lambda_mono", 0.05)
    gamma_frobenius = config.get("gamma_frobenius", 1.80)

    print(
        f"Starting training for {config['n_epochs']} epochs "
        f"(patience={patience}, lambda_smooth={lambda_smooth}, lambda_mono={lambda_mono}, "
        f"gamma_frobenius={gamma_frobenius}, beta_norm={beta_normalized:.6f}, "
        f"alpha={alpha}, beta={beta})..."
    )

    for epoch in range(1, config["n_epochs"] + 1):
        model.train()
        train_loss_accum = torch.tensor(0.0, device=device)
        train_count = 0

        for batch_x, batch_y, batch_w in train_loader:
            optimizer.zero_grad(set_to_none=True)

            if lambda_smooth > 0 or lambda_mono > 0:
                batch_x.requires_grad_(True)
                preds = model._forward_uncompiled(batch_x)
                raw_loss = criterion(preds, batch_y)
                value_loss = (raw_loss * batch_w).mean()

                # Single-pass autograd Jacobian
                jac = torch.autograd.grad(
                    preds.sum(), batch_x, create_graph=True
                )[0]  # shape (batch, 19)

                reg_loss = torch.tensor(0.0, device=device)

                if lambda_smooth > 0:
                    # Comprehensive 19-dimensional thresholded Frobenius regularization
                    jac_norm = torch.linalg.vector_norm(jac, dim=-1)
                    frobenius_penalty = torch.relu(jac_norm - gamma_frobenius).pow(2).mean()
                    reg_loss = reg_loss + lambda_smooth * frobenius_penalty

                if lambda_mono > 0:
                    # Total-derivative multivariable chain rule monotonicity
                    df_dB = compute_total_barrier_derivative(jac, batch_x, norm_in, norm_out)
                    std_out_t = norm_out.get_std_tensor(device, batch_x.dtype)[0]
                    # Scale to normalized target units so penalty is commensurate with value_loss
                    df_dB_norm = df_dB / std_out_t
                    coupon_unnorm = batch_x[:, 6] * norm_in.get_std_tensor(device, batch_x.dtype)[6] + norm_in.get_mean_tensor(device, batch_x.dtype)[6]
                    r_unnorm = batch_x[:, 9] * norm_in.get_std_tensor(device, batch_x.dtype)[9] + norm_in.get_mean_tensor(device, batch_x.dtype)[9]
                    mono_mask = (coupon_unnorm >= r_unnorm).float()
                    mono_penalty = (torch.relu(df_dB_norm).pow(2) * mono_mask).mean()
                    reg_loss = reg_loss + lambda_mono * mono_penalty

                loss = value_loss + reg_loss
            else:
                preds = model(batch_x)
                raw_loss = criterion(preds, batch_y)
                loss = (raw_loss * batch_w).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                del loss, preds
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update Polyak-Ruppert EMA model parameters
            ema_model.update_parameters(model)

            train_loss_accum += loss.detach() * len(batch_x)
            train_count += len(batch_x)

        train_loss = float(train_loss_accum.item()) / max(1, train_count)
        scheduler.step()

        # Fast Vectorized GPU Validation using EMA Model
        eval_model = ema_model.module
        eval_model.eval()
        with torch.no_grad():
            preds_val = eval_model(X_val_t)
            delta_v_pred = norm_out.inverse_transform_tensor(preds_val, preserve_float64=True)  # strict float64
            total_pred = pde_val_t + delta_v_pred.squeeze(-1)
            errors_bps = (total_pred - mc_val_t) * 10000.0

            val_rmse_bps = float(torch.sqrt(torch.mean(errors_bps ** 2)).item())
            val_mae_bps = float(torch.mean(torch.abs(errors_bps)).item())

            # 99%-trimmed RMSE for reference
            p99_threshold = float(torch.quantile(torch.abs(errors_bps), 0.99).item())
            trimmed_mask = torch.abs(errors_bps) <= p99_threshold
            trimmed_rmse_bps = float(torch.sqrt(torch.mean(errors_bps[trimmed_mask] ** 2)).item())

        # Vectorized GPU Monotonicity Validation
        with torch.enable_grad():
            X_val_eval = X_val_t.clone().detach().requires_grad_(True)
            eval_preds = eval_model._forward_uncompiled(X_val_eval)
            eval_jac = torch.autograd.grad(eval_preds.sum(), X_val_eval, create_graph=False)[0]
            eval_df_dB = compute_total_barrier_derivative(eval_jac, X_val_eval, norm_in, norm_out)
            mono_val_pct = float((eval_df_dB > 1e-4).float().mean().item()) * 100.0

        if epoch % 5 == 0 or epoch == 1 or trimmed_rmse_bps < 1.0:
            print(
                f"Epoch {epoch:3d}/{config['n_epochs']} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Raw RMSE: {val_rmse_bps:.2f} bps | Trimmed: {trimmed_rmse_bps:.2f} bps | MAE: {val_mae_bps:.2f} bps | "
                f"Mono Violations: {mono_val_pct:.2f}%"
            )

        # Joint optimization target: trimmed RMSE < 1.0 bps and monotonicity violations < 1.5%
        # Score blends trimmed RMSE with penalty for monotonicity violations above 0.2%
        val_score = trimmed_rmse_bps + 2.0 * max(0.0, (mono_val_pct - 0.2) / 100.0)

        # Accept checkpoint if it improves score or achieves strict compliance
        is_compliant = (trimmed_rmse_bps < 1.0) and (mono_val_pct < 1.5)
        if (is_compliant and val_score < best_val_score) or (val_score < best_val_score and best_val_score == float("inf")):
            best_val_score = val_score
            best_val_rmse = val_rmse_bps
            best_state_dict = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
            patience_counter = 0
            os.makedirs(os.path.dirname(config["weights_save_path"]), exist_ok=True)
            torch.save(best_state_dict, config["weights_save_path"])
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (best score: {best_val_score:.4f}, val RMSE: {best_val_rmse:.2f} bps)")
                break

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    print("\n=======================================================")
    print(f"Training Complete! Best Val RMSE: {best_val_rmse:.2f} bps (Score: {best_val_score:.4f})")
    print(f"Saved weights to: {config['weights_save_path']}")
    print("=======================================================\n")

    return model

DEFAULT_CORRECTION_CONFIG = {
    'train_path': 'data/autocall/train_100k_sobol_pde.npz',
    'val_path': 'data/autocall/val_10k_sobol_pde.npz',
    'n_epochs': 50,
    'batch_size': 256,
    'hidden': 256,
    'n_layers': 4,
    'dropout': 0.0,
    'lr': 5e-5,
    'weight_decay': 1e-5,
    'patience': 25,
    'device_str': 'cuda',
    'in_dim': 19,
    'lambda_smooth': 0.05,
    'lambda_mono': 1.0,
    'gamma_frobenius': 1.80,
    'barrier_weight_alpha': 10.0,
    'barrier_weight_beta': 5.0,
    'weights_save_path': 'artifacts/weights/autocall_correction_mlp.pth',
    'norm_in_path': 'artifacts/scalers/correction_input_normalizer.npz',
    'norm_out_path': 'artifacts/scalers/correction_output_normalizer.npz',
    'resume': True,
}

if __name__ == "__main__":
    os.makedirs("artifacts/weights", exist_ok=True)
    os.makedirs("artifacts/scalers", exist_ok=True)
    
    if os.path.exists(DEFAULT_CORRECTION_CONFIG['train_path']) and os.path.exists(DEFAULT_CORRECTION_CONFIG['val_path']):
        train(DEFAULT_CORRECTION_CONFIG)
    else:
        print(f"Train/Val data not found. Skipping training.")
