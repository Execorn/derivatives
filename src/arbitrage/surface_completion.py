"""
§1.2 Arbitrage-free IV surface completion.

TODO (fill in after deep research results):
  - SVI slice-by-slice fitting (Gatheral 2004)
  - Durrleman condition check
  - Monotone rearrangement (Chernozhukov 2010)
  - FNO-based completion with soft arbitrage penalties
  - Evaluation against held-out quotes
"""
from __future__ import annotations
import numpy as np
from typing import Optional


def check_calendar_spread(iv_surface: np.ndarray,
                          T_grid: np.ndarray) -> np.ndarray:
    """
    Check calendar spread arbitrage: C(T1,K) <= C(T2,K) for T1 < T2.
    In IV space: σ²(T2) * T2 >= σ²(T1) * T1  (total variance monotone in T).

    Returns boolean mask (n_T-1, n_K): True = violation.
    """
    total_var = iv_surface**2 * T_grid[:, None]           # (nT, nK)
    violations = np.diff(total_var, axis=0) < 0           # (nT-1, nK)
    return violations


def check_butterfly(iv_surface: np.ndarray,
                    K_grid: np.ndarray,
                    T_grid: np.ndarray,
                    S: float = 1.0) -> np.ndarray:
    """
    Check butterfly arbitrage: d²C/dK² >= 0.
    Approximate via finite differences on call prices.

    Returns boolean mask (nT, nK-2): True = violation.
    """
    raise NotImplementedError("TODO: implement after §1.2 deep research results")


def fit_svi_slice(k: np.ndarray, total_var: np.ndarray) -> dict:
    """
    Fit raw SVI parametrization to a single maturity slice.

    SVI: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    where w = sigma_IV^2 * T (total variance).

    Returns: {"a": ..., "b": ..., "rho": ..., "m": ..., "sigma": ...}
    """
    raise NotImplementedError("TODO: implement after §1.2 deep research results")


def monotone_rearrangement(f: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Project f onto the set of monotone (non-decreasing) functions
    along the given axis, using sorting-based rearrangement.
    (Chernozhukov, Fernandez-Val, Galichon 2010)
    """
    raise NotImplementedError("TODO: implement after §1.2 deep research results")


def complete_surface(sparse_iv: np.ndarray,
                     mask: np.ndarray,
                     T_grid: np.ndarray,
                     K_grid: np.ndarray,
                     method: str = "fno") -> np.ndarray:
    """
    Complete a sparse IV surface (NaN where mask=False) using the specified method.

    Args:
        sparse_iv:  (nT, nK) array with NaN at missing quotes
        mask:       (nT, nK) bool — True = observed
        method:     "fno" | "svi" | "cubic_spline"

    Returns: (nT, nK) complete arbitrage-free IV surface
    """
    raise NotImplementedError("TODO: implement after §1.2 deep research results")
