"""
Phoenix Autocallable Neural Surrogate Architecture and Normalizers.

Implements:
  - PhoenixInputNormalizer: Z-score normalization for 12 input features.
  - PhoenixOutputNormalizer: Min-max scaling for 4 target outputs:
    [npv, call_prob, cpn_prob, exp_life].
  - PhoenixResidualBlock: Residual block with LayerNorm and SiLU.
  - PhoenixMLP: 6-layer residual surrogate neural network.
  - compute_phoenix_greeks: Autograd sensitivities (Delta_call, Delta_cpn, Vega, Theta).
"""

import os
import warnings
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhoenixInputNormalizer:
    FEATURE_NAMES = [
        "kappa", "theta", "sigma", "rho", "v0",
        "B_call", "B_cpn", "coupon", "T", "n_obs_per_year", "r", "memory"
    ]

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PhoenixInputNormalizer":
        self.mean = np.mean(X, axis=0, dtype=np.float64).astype(np.float32)
        self.std = np.std(X, axis=0, dtype=np.float64).astype(np.float32)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted yet.")
        return ((X - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted yet.")
        return (X_norm * self.std + self.mean).astype(np.float32)

    def transform_tensor(self, X: torch.Tensor) -> torch.Tensor:
        mean_t = torch.tensor(self.mean, dtype=X.dtype, device=X.device)
        std_t = torch.tensor(self.std, dtype=X.dtype, device=X.device)
        return (X - mean_t) / std_t

    def inverse_transform_tensor(self, X_norm: torch.Tensor) -> torch.Tensor:
        mean_t = torch.tensor(self.mean, dtype=X_norm.dtype, device=X_norm.device)
        std_t = torch.tensor(self.std, dtype=X_norm.dtype, device=X_norm.device)
        return X_norm * std_t + mean_t

    def to_tensor(self, X: np.ndarray, device: torch.device) -> torch.Tensor:
        X_norm = self.transform(X)
        return torch.tensor(X_norm, dtype=torch.float32, device=device)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "PhoenixInputNormalizer":
        data = np.load(path)
        norm = cls()
        norm.mean = data["mean"].astype(np.float32)
        norm.std = data["std"].astype(np.float32)
        return norm


class PhoenixOutputNormalizer:
    TARGET_NAMES = ["npv", "call_prob", "cpn_prob", "exp_life"]

    def __init__(self) -> None:
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None

    def fit(self, Y: np.ndarray) -> "PhoenixOutputNormalizer":
        self.min_val = np.min(Y, axis=0).astype(np.float32)
        self.max_val = np.max(Y, axis=0).astype(np.float32)
        diff = self.max_val - self.min_val
        diff = np.where(diff < 1e-8, 1.0, diff)
        self.max_val = self.min_val + diff
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Normalizer has not been fitted yet.")
        diff = self.max_val - self.min_val
        return ((Y - self.min_val) / diff).astype(np.float32)

    def inverse_transform(self, Y_norm: np.ndarray) -> np.ndarray:
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Normalizer has not been fitted yet.")
        diff = self.max_val - self.min_val
        return (Y_norm * diff + self.min_val).astype(np.float32)

    def transform_tensor(self, Y: torch.Tensor) -> torch.Tensor:
        min_t = torch.tensor(self.min_val, dtype=Y.dtype, device=Y.device)
        max_t = torch.tensor(self.max_val, dtype=Y.dtype, device=Y.device)
        return (Y - min_t) / (max_t - min_t)

    def inverse_transform_tensor(self, Y_norm: torch.Tensor) -> torch.Tensor:
        min_t = torch.tensor(self.min_val, dtype=Y_norm.dtype, device=Y_norm.device)
        max_t = torch.tensor(self.max_val, dtype=Y_norm.dtype, device=Y_norm.device)
        return Y_norm * (max_t - min_t) + min_t

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, min_val=self.min_val, max_val=self.max_val)

    @classmethod
    def load(cls, path: str) -> "PhoenixOutputNormalizer":
        data = np.load(path)
        norm = cls()
        norm.min_val = data["min_val"].astype(np.float32)
        norm.max_val = data["max_val"].astype(np.float32)
        return norm


class PhoenixResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.act(self.ln(self.fc1(x)))
        h = self.drop(h)
        h = self.fc2(h)
        return res + h


class PhoenixMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 12,
        hidden: int = 256,
        n_layers: int = 6,
        out_dim: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [PhoenixResidualBlock(hidden, dropout=dropout) for _ in range(n_layers)]
        )
        self.out_head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_head(h)


def compute_phoenix_greeks(
    model: PhoenixMLP,
    x_raw: np.ndarray,
    norm_in: PhoenixInputNormalizer,
    norm_out: PhoenixOutputNormalizer,
) -> Dict[str, float]:
    """Compute first-order sensitivities (Delta_call, Delta_cpn, Vega, Theta)."""
    was_training = model.training
    model.eval()

    x_np = np.asarray(x_raw, dtype=np.float32).reshape(1, 12)
    device = next(model.parameters()).device
    x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)

    x_norm = norm_in.transform_tensor(x_tensor)
    out_norm = model(x_norm)
    out_real = norm_out.inverse_transform_tensor(out_norm)
    npv = out_real[0, 0]

    npv.backward()
    grad = x_tensor.grad

    # Feature indices: 4=v0, 5=B_call, 6=B_cpn, 7=coupon, 8=T
    raw_dB_call = float(grad[0, 5].item()) if grad is not None else 0.0
    # dNPV/dB_call < 0 (higher call barrier delays redemption).
    # Delta_call = -dNPV/dB_call. Sign is NOT forced to enable model risk detection.
    delta_call = -raw_dB_call
    raw_dB_cpn = float(grad[0, 6].item()) if grad is not None else 0.0
    # dNPV/dB_cpn > 0 (higher coupon barrier reduces corridor coupons).
    # Delta_cpn = dNPV/dB_cpn. Sign is NOT forced to enable model risk detection.
    delta_cpn = raw_dB_cpn
    vega = float(grad[0, 4].item()) if grad is not None else 0.0

    # Theta via finite difference bumping T by -1/252
    dt_theta = 1.0 / 252.0
    x_bump = x_np.copy()
    x_bump[0, 8] = max(0.01, float(x_bump[0, 8]) - dt_theta)
    with torch.no_grad():
        x_bump_t = norm_in.to_tensor(x_bump, device=device)
        out_bump = norm_out.inverse_transform_tensor(model(x_bump_t))
        npv_bump = float(out_bump[0, 0].item())
        npv_base = float(out_real[0, 0].item())
    theta = float((npv_bump - npv_base) / dt_theta)

    if was_training:
        model.train()

    return {
        "delta_call": float(delta_call),
        "delta_cpn": float(delta_cpn),
        "vega": float(vega),
        "theta": float(theta),
    }

