"""
autocall_normalizer.py — Feature and Target Normalizers for Autocall Surrogate.

Input normalizer applies z-score standardization across 10 contract and market parameters.
Output normalizer applies min-max scaling to [0, 1] across NPV, early call probability,
and expected note life.
"""

from typing import Any, List, Optional, cast
import numpy as np
import numpy.typing as npt
import torch


class AutocallInputNormalizer:
    """
    Z-score standardizer for the 10-dimensional Autocall input vector:
    [kappa, theta, sigma, rho, v0, B, coupon, T, n_obs_per_year, r].
    """

    FEATURE_NAMES: List[str] = [
        "kappa",
        "theta",
        "sigma",
        "rho",
        "v0",
        "B",
        "coupon",
        "T",
        "n_obs_per_year",
        "r",
    ]

    def __init__(self) -> None:
        self.mean: Optional[npt.NDArray[Any]] = None
        self.std: Optional[npt.NDArray[Any]] = None

    def fit(self, X: npt.NDArray[Any]) -> "AutocallInputNormalizer":
        """Fit mean and standard deviation from numpy array of shape (N, 10)."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean = np.mean(X_arr, axis=0).astype(np.float32)
        std = np.std(X_arr, axis=0).astype(np.float32)
        # Guard against zero variance
        std[std < 1e-8] = 1.0
        self.std = std
        return self

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Transform raw inputs to z-scored float32 array."""
        if self.mean is None or self.std is None:
            raise ValueError("AutocallInputNormalizer is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        return cast(npt.NDArray[Any], ((X_arr - self.mean) / self.std).astype(np.float32))

    def inverse_transform(self, X_norm: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Invert z-scored array back to real parameter space."""
        if self.mean is None or self.std is None:
            raise ValueError("AutocallInputNormalizer is not fitted yet.")
        X_arr = np.asarray(X_norm, dtype=np.float32)
        return cast(npt.NDArray[Any], (X_arr * self.std + self.mean).astype(np.float32))

    def to_tensor(
        self, X: npt.NDArray[Any], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Transform raw inputs to a float32 PyTorch tensor on target device."""
        transformed = self.transform(X)
        return torch.tensor(transformed, dtype=torch.float32, device=device)

    def transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Transform PyTorch tensor directly on its device."""
        if self.mean is None or self.std is None:
            raise ValueError("AutocallInputNormalizer is not fitted yet.")
        mean_t = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std_t = torch.tensor(self.std, dtype=t.dtype, device=t.device)
        return (t - mean_t) / std_t

    def inverse_transform_tensor(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Inverse transform PyTorch tensor directly on its device."""
        if self.mean is None or self.std is None:
            raise ValueError("AutocallInputNormalizer is not fitted yet.")
        mean_t = torch.tensor(self.mean, dtype=t_norm.dtype, device=t_norm.device)
        std_t = torch.tensor(self.std, dtype=t_norm.dtype, device=t_norm.device)
        return t_norm * std_t + mean_t

    def save(self, path: str) -> None:
        """Save fitted normalizer statistics to compressed .npz archive."""
        if self.mean is None or self.std is None:
            raise ValueError("Cannot save an unfitted normalizer.")
        np.savez_compressed(
            path,
            mean=self.mean,
            std=self.std,
            feature_names=np.array(self.FEATURE_NAMES),
        )

    @classmethod
    def load(cls, path: str) -> "AutocallInputNormalizer":
        """Load fitted normalizer statistics from compressed .npz archive."""
        data = np.load(path)
        inst = cls()
        inst.mean = np.asarray(data["mean"], dtype=np.float32)
        inst.std = np.asarray(data["std"], dtype=np.float32)
        return inst


class AutocallOutputNormalizer:
    """
    Min-Max normalizer for the 3-dimensional Autocall target vector:
    [npv, call_prob, exp_life] -> scaled into [0, 1].
    """

    TARGET_NAMES: List[str] = ["npv", "call_prob", "exp_life"]

    def __init__(self) -> None:
        self.min_val: Optional[npt.NDArray[Any]] = None
        self.max_val: Optional[npt.NDArray[Any]] = None

    def fit(self, Y: npt.NDArray[Any]) -> "AutocallOutputNormalizer":
        """Fit min and max values from numpy array of shape (N, 3)."""
        Y_arr = np.asarray(Y, dtype=np.float64)
        self.min_val = np.min(Y_arr, axis=0).astype(np.float32)
        self.max_val = np.max(Y_arr, axis=0).astype(np.float32)
        # Guard against zero range
        diff = self.max_val - self.min_val
        diff[diff < 1e-8] = 1.0
        return self

    def transform(self, Y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Scale targets into [0, 1]."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("AutocallOutputNormalizer is not fitted yet.")
        Y_arr = np.asarray(Y, dtype=np.float32)
        range_val = self.max_val - self.min_val
        return cast(npt.NDArray[Any], ((Y_arr - self.min_val) / range_val).astype(np.float32))

    def inverse_transform(self, Y_norm: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Unscale targets from [0, 1] back to real domain."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("AutocallOutputNormalizer is not fitted yet.")
        Y_arr = np.asarray(Y_norm, dtype=np.float32)
        range_val = self.max_val - self.min_val
        return cast(npt.NDArray[Any], (Y_arr * range_val + self.min_val).astype(np.float32))

    def to_tensor(
        self, Y: npt.NDArray[Any], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Transform raw targets to a float32 PyTorch tensor on target device."""
        transformed = self.transform(Y)
        return torch.tensor(transformed, dtype=torch.float32, device=device)

    def transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Transform PyTorch tensor directly on its device."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("AutocallOutputNormalizer is not fitted yet.")
        min_t = torch.tensor(self.min_val, dtype=t.dtype, device=t.device)
        range_t = torch.tensor(self.max_val - self.min_val, dtype=t.dtype, device=t.device)
        return (t - min_t) / range_t

    def inverse_transform_tensor(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Inverse transform PyTorch tensor directly on its device."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("AutocallOutputNormalizer is not fitted yet.")
        min_t = torch.tensor(self.min_val, dtype=t_norm.dtype, device=t_norm.device)
        range_t = torch.tensor(self.max_val - self.min_val, dtype=t_norm.dtype, device=t_norm.device)
        return t_norm * range_t + min_t

    def save(self, path: str) -> None:
        """Save fitted normalizer statistics to compressed .npz archive."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("Cannot save an unfitted normalizer.")
        np.savez_compressed(
            path,
            min_val=self.min_val,
            max_val=self.max_val,
            target_names=np.array(self.TARGET_NAMES),
        )

    @classmethod
    def load(cls, path: str) -> "AutocallOutputNormalizer":
        """Load fitted normalizer statistics from compressed .npz archive."""
        data = np.load(path)
        inst = cls()
        inst.min_val = np.asarray(data["min_val"], dtype=np.float32)
        inst.max_val = np.asarray(data["max_val"], dtype=np.float32)
        return inst
