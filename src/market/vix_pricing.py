"""
§1.3 VIX futures and variance swap pricing under Rough Heston.

TODO (fill in after deep research results):
  - E[∫₀ᵀ v_t dt] computation via Laplace transform of Rough Heston variance
  - VIX futures: F(t, T_VIX) = E[VIX_T | ℱ_t]
  - VIX options: distribution of VIX_T
  - Joint calibration loss: SPX options + VIX futures + variance swaps
  - Download CBOE VIX futures data
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

CACHE_DIR = Path(__file__).parents[2] / "data" / "market" / "vix"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def model_variance_swap_rate(kappa: float, theta: float, sigma: float,
                              rho: float, v0: float, H: float,
                              T: float) -> float:
    """
    Compute the fair strike of a variance swap under Rough Heston.

    K_var = (1/T) * E[∫₀ᵀ v_t dt]

    Uses the Laplace transform approach from El Euch & Rosenbaum (2019).
    """
    raise NotImplementedError("TODO: implement after §1.3 deep research results")


def model_vix(kappa: float, theta: float, sigma: float,
               rho: float, v0: float, H: float,
               t: float = 0.0, delta: float = 30/365) -> float:
    """
    Compute model VIX at time t.

    VIX(t)² = (1/delta) * E[∫ₜ^{t+delta} v_s ds | ℱ_t]
    where delta = 30/365 years (30-day window).
    """
    raise NotImplementedError("TODO: implement after §1.3 deep research results")


def vix_futures_curve(kappa: float, theta: float, sigma: float,
                       rho: float, v0: float, H: float,
                       maturities: np.ndarray) -> np.ndarray:
    """
    Compute VIX futures prices across a term structure of maturities.

    Args:
        maturities: array of futures expiry times in years
    Returns:
        array of VIX futures prices (in VIX points, i.e., annualised %)
    """
    raise NotImplementedError("TODO: implement after §1.3 deep research results")


def download_vix_futures(snapshot_date: str) -> dict:
    """
    Download VIX futures term structure from CBOE.

    Returns: {"maturities": [...], "prices": [...]}
    """
    raise NotImplementedError("TODO: implement after §1.3 deep research results")


def joint_calibration_loss(params: np.ndarray,
                            spx_iv_observed: np.ndarray,
                            vix_futures_observed: np.ndarray,
                            vix_maturities: np.ndarray,
                            w_spx: float = 1.0,
                            w_vix: float = 0.5) -> float:
    """
    Combined loss for joint SPX + VIX calibration.

    L = w_spx * RMSE_SPX + w_vix * RMSE_VIX
    """
    raise NotImplementedError("TODO: implement after §1.3 deep research results")
