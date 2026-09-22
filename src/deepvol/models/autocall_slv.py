"""
Local Volatility (LV) and Stochastic Local Volatility (SLV) Autocall Integration.

Integrates Dupire local volatility surfaces and McKean-Vlasov SDE particle solvers
with the GPU Monte Carlo autocall payoff kernel.

Mathematical References:
  - Dupire, B. (1994). Pricing with a smile. Risk, 7(1), 18-20.
  - Guyon, J., & Henry-Labordère, P. (2012). Being particular about local volatility. Risk, 25(1), 78-83.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor

from deepvol.models.local_vol import svi_to_lv_surface
from deepvol.models.mlsv_gpu import MLSVSolverGPU
from deepvol.models.autocall import price_autocall_mc, make_obs_indices
from deepvol.hedging.d_xva import simulate_heston_paths


def build_dupire_vol_fn(
    T_grid: Tensor,       # (nT,) float64
    K_grid: Tensor,       # (nK,) float64 (log-moneyness or strikes)
    svi_params: Tensor,   # (nT, 5) float64
    device: Optional[torch.device] = None,
    S0_ref: float = 100.0,
) -> Callable[[float, Tensor], Tensor]:
    """
    Builds a GPU-accelerated local volatility interpolation closure from SVI parameters.
    Enforces minimum vol clamp to 0.01 (100 bps) to prevent mathematical singularities.
    """
    if device is None:
        device = svi_params.device

    T_t = T_grid.to(device=device, dtype=torch.float64)
    K_t = K_grid.to(device=device, dtype=torch.float64)
    svi_t = svi_params.to(device=device, dtype=torch.float64)

    # Precompute local volatility surface on (nT, nK)
    lv_surface = svi_to_lv_surface(T_t, K_t, svi_t)
    if isinstance(lv_surface, np.ndarray):
        lv_surface = torch.tensor(lv_surface, dtype=torch.float64, device=device)
    else:
        lv_surface = lv_surface.to(device=device, dtype=torch.float64)

    # Squeeze batch dimension if present
    if lv_surface.ndim == 3 and lv_surface.shape[0] == 1:
        lv_surface = lv_surface.squeeze(0)

    nT = len(T_t)
    nK = len(K_t)

    def vol_fn(t: float, S: Tensor) -> Tensor:
        # S: shape (N,) or (1, N)
        S_flat = S.reshape(-1).to(dtype=torch.float64, device=device)
        k_val = torch.log(torch.clamp(S_flat / S0_ref, min=1e-6))

        # Clamp t and k to grid boundaries
        t_clamped = torch.clamp(torch.as_tensor(t, dtype=torch.float64, device=device), T_t[0], T_t[-1])
        k_clamped = torch.clamp(k_val, K_t[0], K_t[-1])

        # 1D index for T
        t_idx = torch.bucketize(t_clamped, T_t) - 1
        t_idx = torch.clamp(t_idx, 0, nT - 2)
        t_left, t_right = T_t[t_idx], T_t[t_idx + 1]
        t_w = (t_clamped - t_left) / torch.clamp(t_right - t_left, min=1e-8)

        # 1D index for K
        k_idx = torch.bucketize(k_clamped, K_t) - 1
        k_idx = torch.clamp(k_idx, 0, nK - 2)
        k_left, k_right = K_t[k_idx], K_t[k_idx + 1]
        k_w = (k_clamped - k_left) / torch.clamp(k_right - k_left, min=1e-8)

        # Bilinear interpolation
        v00 = lv_surface[t_idx, k_idx]
        v01 = lv_surface[t_idx, k_idx + 1]
        v10 = lv_surface[t_idx + 1, k_idx]
        v11 = lv_surface[t_idx + 1, k_idx + 1]

        v0 = v00 * (1.0 - k_w) + v01 * k_w
        v1 = v10 * (1.0 - k_w) + v11 * k_w
        vol_interp = v0 * (1.0 - t_w) + v1 * t_w

        # Hardening: Clamp vol to [0.01, 2.0]
        vol_clamped = torch.clamp(vol_interp, min=0.01, max=2.0)
        return vol_clamped.view_as(S).to(dtype=S.dtype)

    return vol_fn


def simulate_lv_paths(
    S0: float,
    r: float,
    T: float,
    N_steps: int,
    N_paths: int,
    vol_fn: Callable[[float, Tensor], Tensor],
    device: torch.device,
) -> Tensor:
    """Simulate spot paths under Local Volatility using log-Euler scheme."""
    dt = T / N_steps
    sqrt_dt = dt ** 0.5

    S = torch.empty(1, N_paths, N_steps + 1, dtype=torch.float64, device=device)
    S[:, :, 0] = S0

    curr_S = torch.full((N_paths,), S0, dtype=torch.float64, device=device)
    r_val = float(r)

    Z_all = torch.randn(N_steps, N_paths, dtype=torch.float64, device=device)

    for k in range(N_steps):
        t_k = k * dt
        vol_k = vol_fn(t_k, curr_S)
        Z = Z_all[k]

        # Log-Euler step: S_{k+1} = S_k * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        drift = (r_val - 0.5 * (vol_k ** 2)) * dt
        diffusion = vol_k * sqrt_dt * Z
        curr_S = curr_S * torch.exp(drift + diffusion)
        S[:, :, k + 1] = curr_S

    return S


def price_autocall_lv_mc(
    S0: float,
    r: float,
    T: float,
    N_steps: int,
    N_paths: int,
    obs_indices: List[int],
    B_scalar: float,
    coupon_scalar: float,
    svi_params: Tensor,
    T_grid: Tensor,
    K_grid: Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Price an autocallable note under pure Dupire Local Volatility."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if N_steps <= 0:
        N_steps = max(252, int(round(T * 252)))

    vol_fn = build_dupire_vol_fn(T_grid, K_grid, svi_params, device=device, S0_ref=S0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    S_paths = simulate_lv_paths(S0, r, T, N_steps, N_paths, vol_fn, device)

    B_t = torch.tensor([B_scalar], dtype=torch.float64, device=device)
    coupon_t = torch.tensor([coupon_scalar], dtype=torch.float64, device=device)
    r_t = torch.tensor([r], dtype=torch.float64, device=device)
    dt = T / N_steps

    npv, call_prob, exp_life = price_autocall_mc(
        S=S_paths,
        obs_indices=obs_indices,
        B=B_t,
        coupon=coupon_t,
        r=r_t,
        T=T,
        dt=dt,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    timing_s = time.perf_counter() - t0

    return {
        "npv": float(npv.item()),
        "call_prob": float(call_prob.item()),
        "exp_life": float(exp_life.item()),
        "timing_s": float(timing_s),
    }


def price_autocall_slv_mc(
    S0: float,
    r: float,
    q: float,
    slv_params: dict,
    T: float,
    N_steps: int,
    N_paths: int,
    obs_indices: List[int],
    B_scalar: float,
    coupon_scalar: float,
    svi_params: Tensor,
    T_grid: Tensor,
    K_grid: Tensor,
    device: Optional[torch.device] = None,
    method: str = "nadaraya_watson",
) -> Dict[str, float]:
    """Price an autocallable note under Stochastic Local Volatility (McKean-Vlasov SDE)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if N_steps <= 0:
        N_steps = max(252, int(round(T * 252)))

    vol_fn = build_dupire_vol_fn(T_grid, K_grid, svi_params, device=device, S0_ref=S0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    solver = MLSVSolverGPU(
        S0=S0,
        r=r,
        q=q,
        v0=slv_params.get("v0", 0.04),
        kappa=slv_params.get("kappa", 2.0),
        theta=slv_params.get("theta", 0.04),
        xi=slv_params.get("xi", slv_params.get("sigma", 0.3)),
        rho=slv_params.get("rho", -0.6),
        T=T,
        steps_per_unit=int(round(N_steps / T)),
        N_paths=N_paths,
        dupire_vol_fn=vol_fn,
        device=str(device),
        dtype=torch.float64,
    )
    solver.simulate(method=method)

    # Convert log-spot paths (N_steps+1, N_paths) -> (1, N_paths, N_steps+1)
    S_paths = (S0 * torch.exp(solver.X_paths)).T.unsqueeze(0)

    B_t = torch.tensor([B_scalar], dtype=torch.float64, device=device)
    coupon_t = torch.tensor([coupon_scalar], dtype=torch.float64, device=device)
    r_t = torch.tensor([r], dtype=torch.float64, device=device)
    dt = T / N_steps

    npv, call_prob, exp_life = price_autocall_mc(
        S=S_paths,
        obs_indices=obs_indices,
        B=B_t,
        coupon=coupon_t,
        r=r_t,
        T=T,
        dt=dt,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    timing_s = time.perf_counter() - t0

    return {
        "npv": float(npv.item()),
        "call_prob": float(call_prob.item()),
        "exp_life": float(exp_life.item()),
        "timing_s": float(timing_s),
    }


def lv_vs_heston_comparison(
    heston_params: dict,
    svi_params: Tensor,
    T_grid: Tensor,
    K_grid: Tensor,
    autocall_contract: dict,
    device: Optional[torch.device] = None,
    n_paths: int = 50000,
) -> Dict[str, Dict[str, float]]:
    """Compare pricing across Heston, LV, and SLV models."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    S0 = float(autocall_contract.get("S0", 100.0))
    B = float(autocall_contract["B"])
    coupon = float(autocall_contract["coupon"])
    T = float(autocall_contract["T"])
    r = float(autocall_contract["r"])
    n_obs = int(autocall_contract.get("n_obs", 4))
    N_steps = int(round(T * 252))
    obs_indices = make_obs_indices(n_obs, T, N_steps)

    # 1. Heston MC
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    theta_t = torch.tensor([[
        heston_params["kappa"], heston_params["theta"], heston_params["sigma"],
        heston_params["rho"], heston_params["v0"]
    ]], dtype=torch.float64, device=device)

    S_heston = simulate_heston_paths(
        theta=theta_t, S0=S0, T=T, N_steps=N_steps, N_paths=n_paths, r=r, device=device
    )
    B_t = torch.tensor([B], dtype=torch.float64, device=device)
    c_t = torch.tensor([coupon], dtype=torch.float64, device=device)
    r_t = torch.tensor([r], dtype=torch.float64, device=device)
    npv_h, cp_h, el_h = price_autocall_mc(S_heston, obs_indices, B_t, c_t, r_t, T, T / N_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_heston = time.perf_counter() - t0

    # 2. LV MC
    lv_res = price_autocall_lv_mc(
        S0=S0, r=r, T=T, N_steps=N_steps, N_paths=n_paths, obs_indices=obs_indices,
        B_scalar=B, coupon_scalar=coupon, svi_params=svi_params, T_grid=T_grid, K_grid=K_grid,
        device=device,
    )

    # 3. SLV MC
    slv_params = {
        "kappa": heston_params["kappa"],
        "theta": heston_params["theta"],
        "xi": heston_params["sigma"],
        "rho": heston_params["rho"],
        "v0": heston_params["v0"],
    }
    slv_res = price_autocall_slv_mc(
        S0=S0, r=r, q=0.0, slv_params=slv_params, T=T, N_steps=N_steps, N_paths=n_paths,
        obs_indices=obs_indices, B_scalar=B, coupon_scalar=coupon, svi_params=svi_params,
        T_grid=T_grid, K_grid=K_grid, device=device,
    )

    return {
        "heston": {
            "npv": float(npv_h.item()),
            "call_prob": float(cp_h.item()),
            "exp_life": float(el_h.item()),
            "timing_s": float(t_heston),
        },
        "lv": lv_res,
        "slv": slv_res,
    }

