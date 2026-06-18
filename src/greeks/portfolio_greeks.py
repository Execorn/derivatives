"""
§1.4 Portfolio-level Greeks via PyTorch autograd through FNO.

All Greeks are analytic (no finite differences) — computed via torch.func.jacfwd/jacrev.

TODO (fill in after deep research results):
  - Black-Scholes in differentiable PyTorch ops (for chain rule)
  - Full Greek surface computation (Δ, Γ, Vega, Vanna, Volga, Theta)
  - Batched portfolio Greek aggregation
  - Dollar-delta hedge ratio computation
  - Vega bucketing by maturity
  - P&L attribution decomposition
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Optional


# ── Black-Scholes in differentiable PyTorch (needed for chain rule) ──────────

def bs_call_price(S: torch.Tensor, K: torch.Tensor,
                  T: torch.Tensor, r: torch.Tensor,
                  sigma: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Black-Scholes call price.
    All inputs are torch tensors — no numpy operations inside.
    """
    raise NotImplementedError("TODO: implement after §1.4 deep research results")


def bs_greeks(S: float, K: float, T: float, r: float,
              sigma_iv: float) -> dict:
    """
    Closed-form Black-Scholes Greeks for a single option.

    Returns:
        delta, gamma, vega, theta, rho,
        vanna (d_delta/d_sigma), volga (d_vega/d_sigma)
    """
    raise NotImplementedError("TODO: implement after §1.4 deep research results")


# ── FNO Model Greeks ──────────────────────────────────────────────────────────

def fno_parameter_jacobian(model: torch.nn.Module,
                            theta: torch.Tensor,
                            spatial: torch.Tensor) -> torch.Tensor:
    """
    Compute full Jacobian of IV surface w.r.t. model parameters.

    Uses torch.func.jacfwd for efficiency (6 params → 88 outputs).

    Returns: (nT, nK, n_params) Jacobian tensor
    """
    raise NotImplementedError("TODO: implement after §1.4 deep research results")


def fno_surface_greeks(model: torch.nn.Module,
                        theta: np.ndarray,
                        pn, yn,
                        S: float, r: float = 0.05,
                        T_grid: Optional[np.ndarray] = None,
                        K_grid: Optional[np.ndarray] = None) -> dict:
    """
    Compute full Greek surface for all (T, K) grid points.

    Returns dict with keys: delta, gamma, vega, vanna, volga, theta
    Each value is np.ndarray of shape (nT, nK).
    """
    raise NotImplementedError("TODO: implement after §1.4 deep research results")


# ── Portfolio Greeks ──────────────────────────────────────────────────────────

def portfolio_greeks(positions: list[dict],
                     model: torch.nn.Module,
                     theta: np.ndarray,
                     pn, yn, S: float) -> dict:
    """
    Aggregate Greeks across a portfolio of option positions.

    Each position dict: {"K": float, "T": float, "type": "call"|"put",
                         "notional": float, "quantity": float}

    Returns:
        {
          "total_delta":  float,  # in $ per $1 move in S
          "total_gamma":  float,
          "vega_bucket":  np.ndarray (nT,),  # vega by maturity
          "total_vanna":  float,
          "total_volga":  float,
          "hedge_contracts": int,  # SPX futures to buy for delta-neutral
        }
    """
    raise NotImplementedError("TODO: implement after §1.4 deep research results")


def pnl_attribution(S_before: float, S_after: float,
                     sigma_before: float, sigma_after: float,
                     greeks: dict) -> dict:
    """
    Decompose daily P&L using Taylor expansion:
    ΔC ≈ Δ*ΔS + ½Γ*ΔS² + Vega*Δσ + Vanna*ΔS*Δσ + ½Volga*Δσ²

    Returns: {"delta_pnl": ..., "gamma_pnl": ..., "vega_pnl": ...,
              "vanna_pnl": ..., "volga_pnl": ..., "unexplained": ...}
    """
    raise NotImplementedError("TODO: implement after §1.4 deep research results")
