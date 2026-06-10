"""
normalizers.py — Input/Output scalers for the FNO Surrogate.

Two normalizers are fit on the training split and saved to disk alongside
the model weights. Both must be loaded at inference time.

Design rationale
----------------
ParameterNormalizer (z-score per-parameter):
  κ ∈ [0.1, 5.0]  vs  H ∈ [0.02, 0.15]  — a 40× range ratio.
  Without standardization, AdamW weight decay prunes the small-range
  parameter connections faster than gradients can restore them.
  Z-score standardization makes all inputs unit-variance, ensuring
  isotropic gradient flow through the FiLM generator MLP.

IVSurfaceNormalizer (per-grid-point z-score):
  The T=0.1 slice has std=0.2017 vs std=0.1284 at T=2.0 — a 60% gap.
  Absolute MSE over the raw surface lets T=0.1 dominate gradients,
  starving the long-maturity slices of learning signal. Per-grid-point
  normalization forces every (T,K) point to contribute equally to the
  training loss, restoring gradient coverage for κ, σ, ρ, H.
"""

import os
import numpy as np
import torch


class ParameterNormalizer:
    """
    Z-score normalizer for the 6-dimensional Rough Heston parameter vector.
    Fit on the training split, applied at both train and inference time.
    """

    PARAM_NAMES = ["kappa", "theta", "sigma", "rho", "v0", "H"]

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ParameterNormalizer":
        """Fit on training data X of shape (N, 6)."""
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0)
        # Guard against degenerate constant parameters
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "Call fit() first"
        return (X - self.mean) / self.std

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "Call fit() first"
        return X_norm * self.std + self.mean

    def transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Transform a torch.Tensor on any device."""
        mean = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std  = torch.tensor(self.std,  dtype=t.dtype, device=t.device)
        return (t - mean) / std

    def inverse_transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std  = torch.tensor(self.std,  dtype=t.dtype, device=t.device)
        return t * std + mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "ParameterNormalizer":
        data = np.load(path)
        n = cls()
        n.mean = data["mean"]
        n.std  = data["std"]
        return n

    def summary(self) -> str:
        lines = ["ParameterNormalizer (μ ± σ):"]
        for name, m, s in zip(self.PARAM_NAMES, self.mean, self.std):
            lines.append(f"  {name:6s}: {m:.4f} ± {s:.4f}")
        return "\n".join(lines)


class IVSurfaceNormalizer:
    """
    Per-grid-point z-score normalizer for the (8, 11) implied volatility surface.

    Each of the 88 grid points (T_i, K_j) is independently standardized
    so that every point contributes equal variance to the training MSE.
    This prevents the T=0.1 roughness explosion from monopolizing gradients.

    After training, denormalization is applied before computing the
    calibration loss and before any display to the user.
    """

    def __init__(self):
        self.mean: np.ndarray | None = None   # shape (8, 11)
        self.std:  np.ndarray | None = None   # shape (8, 11)

    def fit(self, Y: np.ndarray) -> "IVSurfaceNormalizer":
        """Fit on training data Y of shape (N, 8, 11)."""
        self.mean = Y.mean(axis=0)   # (8, 11)
        self.std  = Y.std(axis=0)    # (8, 11)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "Call fit() first"
        return (Y - self.mean) / self.std

    def inverse_transform(self, Y_norm: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "Call fit() first"
        return Y_norm * self.std + self.mean

    def transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std  = torch.tensor(self.std,  dtype=t.dtype, device=t.device)
        return (t - mean) / std

    def inverse_transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std  = torch.tensor(self.std,  dtype=t.dtype, device=t.device)
        return t * std + mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "IVSurfaceNormalizer":
        data = np.load(path)
        n = cls()
        n.mean = data["mean"]
        n.std  = data["std"]
        return n

    def summary(self) -> str:
        maturities = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
        lines = ["IVSurfaceNormalizer — per-maturity avg std:"]
        for i, T in enumerate(maturities):
            lines.append(f"  T={T:.1f}: mean_IV={self.mean[i].mean():.4f}  "
                         f"avg_std={self.std[i].mean():.4f}")
        return "\n".join(lines)
