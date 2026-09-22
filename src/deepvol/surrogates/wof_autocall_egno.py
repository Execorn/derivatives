"""
Worst-of Autocallable Equivariant Graph Neural Operator (EGNO) Surrogate.

Implements:
  - WoFInputNormalizer: Z-score normalization for 16 worst-of parameters.
  - WoFOutputNormalizer: Min-max scaling for [npv, call_prob, exp_life].
  - WoFAutocallEGNO: Permutation-equivariant graph neural operator.
"""

import sys
sys.path.insert(0, "src")

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from deepvol.surrogates.egno import EGNOLayer


class WoFInputNormalizer:
    FEATURE_NAMES = [
        "kappa1", "theta1", "sigma1", "rho_sv1", "v01",
        "kappa2", "theta2", "sigma2", "rho_sv2", "v02",
        "rho_12", "B", "coupon", "T", "n_obs_per_year", "r"
    ]

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "WoFInputNormalizer":
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
    def load(cls, path: str) -> "WoFInputNormalizer":
        data = np.load(path)
        norm = cls()
        norm.mean = data["mean"].astype(np.float32)
        norm.std = data["std"].astype(np.float32)
        return norm


class WoFOutputNormalizer:
    TARGET_NAMES = ["npv", "call_prob", "exp_life"]

    def __init__(self) -> None:
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None

    def fit(self, Y: np.ndarray) -> "WoFOutputNormalizer":
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
    def load(cls, path: str) -> "WoFOutputNormalizer":
        data = np.load(path)
        norm = cls()
        norm.min_val = data["min_val"].astype(np.float32)
        norm.max_val = data["max_val"].astype(np.float32)
        return norm


class WoFAutocallEGNO(nn.Module):
    def __init__(
        self,
        node_in: int = 6,
        edge_in: int = 1,
        global_in: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 2,
        out_dim: int = 3,
    ) -> None:
        super().__init__()
        self.node_in = node_in
        self.edge_in = edge_in
        self.global_in = global_in
        self.hidden_dim = hidden_dim

        self.node_proj = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList(
            [EGNOLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _build_graph(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_flat: (B, 16)
        # [kappa1, theta1, sigma1, rho_sv1, v01, kappa2, theta2, sigma2, rho_sv2, v02, rho_12, B, coupon, T, n_obs, r]
        B = x_flat.shape[0]
        device = x_flat.device
        dtype = x_flat.dtype

        # Node 1: [kappa1, theta1, sigma1, rho_sv1, v01, 1.0]
        node1 = torch.stack([
            x_flat[:, 0], x_flat[:, 1], x_flat[:, 2], x_flat[:, 3], x_flat[:, 4],
            torch.ones(B, device=device, dtype=dtype),
        ], dim=1)

        # Node 2: [kappa2, theta2, sigma2, rho_sv2, v02, 1.0]
        node2 = torch.stack([
            x_flat[:, 5], x_flat[:, 6], x_flat[:, 7], x_flat[:, 8], x_flat[:, 9],
            torch.ones(B, device=device, dtype=dtype),
        ], dim=1)

        x_nodes = torch.stack([node1, node2], dim=1)  # (B, 2, 6)

        # Edge: (B, 2, 2, 1)
        rho_12 = x_flat[:, 10].view(B, 1, 1, 1)
        ones = torch.ones_like(rho_12)
        row0 = torch.cat([ones, rho_12], dim=2)
        row1 = torch.cat([rho_12, ones], dim=2)
        e_edges = torch.cat([row0, row1], dim=1)  # (B, 2, 2, 1)

        # Global: [B, coupon, T, n_obs, r] -> (B, 5)
        g_global = x_flat[:, 11:]  # (B, 5)
        return x_nodes, e_edges, g_global

    def _forward_uncompiled(
        self,
        x: torch.Tensor,
        e: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if e is None or g is None:
            x, e, g = self._build_graph(x)

        h = self.node_proj(x)
        e_feat = self.edge_proj(e)
        g_feat = self.global_proj(g)

        for layer in self.layers:
            h, e_feat = layer(h, e_feat, g_feat)

        # Global mean pool across the 2 nodes
        h_pool = h.mean(dim=1)  # (B, hidden_dim)
        combined = torch.cat([h_pool, g_feat], dim=-1)  # (B, 2*hidden_dim)
        return self.out_mlp(combined)

    def forward(
        self,
        x: torch.Tensor,
        e: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._forward_uncompiled(x, e, g)

