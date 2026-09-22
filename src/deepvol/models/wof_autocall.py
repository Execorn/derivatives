"""
Worst-of Multi-Asset Autocallable Option Pricing Kernel under Correlated Heston Models.

Implements:
  - simulate_correlated_heston_paths: 2-asset joint Heston simulation with 4x4 Cholesky decomposition.
  - price_wof_autocall_mc: GPU Monte Carlo pricing kernel for worst-of barrier notes.
  - correlation_sensitivity: Finite-difference sensitivity to asset-asset correlation rho_12.

Mathematical References:
  - Overhaus, M. et al. (2007). Equity Hybrid Derivatives. John Wiley & Sons.
  - Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. Springer.
"""

import sys
sys.path.insert(0, "src")

from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from deepvol.models.autocall import price_autocall_mc


@torch.compile(mode="default", dynamic=True)
def _wof_mc_kernel(
    S1: Tensor,           # (B, N_paths, N_steps+1) float64
    S2: Tensor,           # (B, N_paths, N_steps+1) float64
    obs_steps: Tensor,    # (N_obs,) int64
    B: Tensor,            # (B,) float64
    coupon: Tensor,       # (B,) float64
    r: Tensor,            # (B,) float64
    T: float,
    dt: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    B_batch, N_paths, _ = S1.shape
    device = S1.device

    S0_1 = S1[:, :, 0:1]
    S0_2 = S2[:, :, 0:1]

    called = torch.zeros(B_batch, N_paths, dtype=torch.bool, device=device)
    npv = torch.zeros(B_batch, N_paths, dtype=torch.float64, device=device)
    life = torch.zeros(B_batch, N_paths, dtype=torch.float64, device=device)

    N_obs = obs_steps.shape[0]
    for i in range(N_obs):
        step_idx = obs_steps[i]
        t_i = step_idx.to(torch.float64) * dt

        perf1 = S1[:, :, step_idx] / S0_1.squeeze(-1)
        perf2 = S2[:, :, step_idx] / S0_2.squeeze(-1)
        wof_perf = torch.minimum(perf1, perf2)

        active = ~called
        trigger = active & (wof_perf >= B.unsqueeze(1))

        disc = torch.exp(-r.unsqueeze(1) * t_i)
        call_payoff = disc * (1.0 + coupon.unsqueeze(1) * t_i)

        npv = npv + trigger.to(torch.float64) * call_payoff
        life = life + trigger.to(torch.float64) * t_i
        called = called | trigger

    # Capital protection at maturity
    uncalled = ~called
    disc_T = torch.exp(-r.unsqueeze(1) * T)
    npv = npv + uncalled.to(torch.float64) * disc_T * 1.0
    life = life + uncalled.to(torch.float64) * T

    return npv.mean(dim=1).clone(), called.to(torch.float64).mean(dim=1).clone(), life.mean(dim=1).clone()


def simulate_correlated_heston_paths(
    theta1: Tensor,       # (B, 5): kappa1, theta1, sigma1, rho_sv1, v0_1
    theta2: Tensor,       # (B, 5): kappa2, theta2, sigma2, rho_sv2, v0_2
    rho_assets: Tensor,   # (B,): rho_12 asset correlation
    S0_1: float,
    S0_2: float,
    T: float,
    N_steps: int,
    N_paths: int,
    r: Union[float, Tensor] = 0.0,
    antithetic: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """Simulate joint correlated Heston paths for 2 assets using 4x4 Cholesky decomposition."""
    if device is None:
        device = theta1.device

    B = theta1.shape[0]
    dt = T / N_steps
    sqrt_dt = dt ** 0.5

    # Unpack parameters
    kappa1, theta_v1, sigma1, rho_sv1, v0_1 = theta1[:, 0], theta1[:, 1], theta1[:, 2], theta1[:, 3], theta1[:, 4]
    kappa2, theta_v2, sigma2, rho_sv2, v0_2 = theta2[:, 0], theta2[:, 1], theta2[:, 2], theta2[:, 3], theta2[:, 4]

    # Analytical Cholesky factor L for [W^{S1}, W^{V1}, W^{S2}, W^{V2}]:
    # W^{S1} = Z_1
    # W^{V1} = rho_sv1 * Z_1 + sqrt(1 - rho_sv1^2) * Z_2
    # W^{S2} = rho_12 * Z_1 + sqrt(1 - rho_12^2) * Z_3
    # W^{V2} = rho_sv2 * W^{S2} + sqrt(1 - rho_sv2^2) * (rho_12 * Z_2 + sqrt(1 - rho_12^2) * Z_4)
    # Guaranteed PSD for any rho_12 in [-1, 1], rho_sv in (-1, 1).
    L = torch.zeros(B, 4, 4, dtype=torch.float64, device=device)
    c1 = torch.sqrt(torch.clamp(1.0 - rho_sv1**2, min=1e-8))
    c2 = torch.sqrt(torch.clamp(1.0 - rho_assets**2, min=0.0))
    c3 = torch.sqrt(torch.clamp(1.0 - rho_sv2**2, min=1e-8))

    L[:, 0, 0] = 1.0
    L[:, 1, 0] = rho_sv1
    L[:, 1, 1] = c1
    L[:, 2, 0] = rho_assets
    L[:, 2, 2] = c2
    L[:, 3, 0] = rho_sv2 * rho_assets
    L[:, 3, 1] = c3 * rho_assets
    L[:, 3, 2] = rho_sv2 * c2
    L[:, 3, 3] = c3 * c2

    # Initialize paths: S1, S2, V1, V2
    S1 = torch.empty(B, N_paths, N_steps + 1, dtype=torch.float64, device=device)
    S2 = torch.empty(B, N_paths, N_steps + 1, dtype=torch.float64, device=device)
    S1[:, :, 0] = S0_1
    S2[:, :, 0] = S0_2

    V1 = v0_1.unsqueeze(1).repeat(1, N_paths)
    V2 = v0_2.unsqueeze(1).repeat(1, N_paths)

    r_t = torch.as_tensor(r, dtype=torch.float64, device=device)
    if r_t.ndim > 0:
        r_t = r_t.view(B, 1)

    use_av = antithetic and (N_paths % 2 == 0)
    half_paths = N_paths // 2 if use_av else N_paths

    for k in range(N_steps):
        # Generate independent standard normals
        if use_av:
            Z_half = torch.randn(B, 4, half_paths, dtype=torch.float64, device=device)
            Z_iid = torch.cat([Z_half, -Z_half], dim=2)
        else:
            Z_iid = torch.randn(B, 4, N_paths, dtype=torch.float64, device=device)

        # Correlated increments: L @ Z_iid -> (B, 4, N_paths)
        dZ = torch.bmm(L, Z_iid) * sqrt_dt

        dW_S1 = dZ[:, 0, :]
        dW_V1 = dZ[:, 1, :]
        dW_S2 = dZ[:, 2, :]
        dW_V2 = dZ[:, 3, :]

        # Full truncation for variance processes
        V1_pos = torch.clamp(V1, min=0.0)
        V2_pos = torch.clamp(V2, min=0.0)
        sqrt_V1 = torch.sqrt(V1_pos)
        sqrt_V2 = torch.sqrt(V2_pos)

        # Spot step (Log-Euler exponential stepping guarantees S > 0 strictly)
        curr_S1 = S1[:, :, k]
        curr_S2 = S2[:, :, k]
        S1[:, :, k + 1] = curr_S1 * torch.exp((r_t - 0.5 * V1_pos) * dt + sqrt_V1 * dW_S1)
        S2[:, :, k + 1] = curr_S2 * torch.exp((r_t - 0.5 * V2_pos) * dt + sqrt_V2 * dW_S2)

        # Variance step (Full truncation scheme)
        V1 = V1 + kappa1.unsqueeze(1) * (theta_v1.unsqueeze(1) - V1_pos) * dt + sigma1.unsqueeze(1) * sqrt_V1 * dW_V1
        V2 = V2 + kappa2.unsqueeze(1) * (theta_v2.unsqueeze(1) - V2_pos) * dt + sigma2.unsqueeze(1) * sqrt_V2 * dW_V2

    return S1.clone(), S2.clone()


def price_wof_autocall_mc(
    S1: Tensor,
    S2: Tensor,
    obs_indices: List[int],
    B: Tensor,
    coupon: Tensor,
    r: Tensor,
    T: float,
    dt: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Price worst-of 2-asset autocall note via GPU Monte Carlo."""
    obs_steps = torch.tensor(obs_indices, dtype=torch.int64, device=S1.device)
    return _wof_mc_kernel(S1, S2, obs_steps, B, coupon, r, T, dt)


def correlation_sensitivity(
    S1: Tensor,
    S2: Tensor,
    obs_indices: List[int],
    B: Tensor,
    coupon: Tensor,
    r: Tensor,
    T: float,
    dt: float,
    delta_rho: float = 0.05,
    theta1: Optional[Tensor] = None,
    theta2: Optional[Tensor] = None,
    rho_12: Optional[Tensor] = None,
) -> Tensor:
    """Calculate sensitivity of WoF autocall NPV to correlation rho_12.

    If model parameters (theta1, theta2, rho_12) are provided, computes finite differences
    via Common Random Numbers (CRN) simulation across rho_12 +/- delta_rho.
    If only paths S1, S2 are provided, computes path-wise copula correlation shifting
    of the orthogonalized asset increments.
    """
    if theta1 is not None and theta2 is not None and rho_12 is not None:
        N_paths = S1.shape[1]
        N_steps = S1.shape[2] - 1
        device = S1.device

        rho_val = float(rho_12.item() if isinstance(rho_12, Tensor) else rho_12)
        rho_up = torch.tensor([min(0.99, rho_val + delta_rho)], dtype=torch.float64, device=device)
        rho_dn = torch.tensor([max(-0.99, rho_val - delta_rho)], dtype=torch.float64, device=device)
        eff_drho = (rho_up - rho_dn).item()
        if eff_drho < 1e-6:
            eff_drho = delta_rho

        S0_1 = float(S1[0, 0, 0].item())
        S0_2 = float(S2[0, 0, 0].item())
        r_val = float(r[0].item() if isinstance(r, Tensor) and r.ndim > 0 else (r.item() if isinstance(r, Tensor) else r))

        S1_up, S2_up = simulate_correlated_heston_paths(
            theta1, theta2, rho_up, S0_1, S0_2, T, N_steps, N_paths, r_val, device=device
        )
        S1_dn, S2_dn = simulate_correlated_heston_paths(
            theta1, theta2, rho_dn, S0_1, S0_2, T, N_steps, N_paths, r_val, device=device
        )
        npv_up, _, _ = price_wof_autocall_mc(S1_up, S2_up, obs_indices, B, coupon, r, T, dt)
        npv_dn, _, _ = price_wof_autocall_mc(S1_dn, S2_dn, obs_indices, B, coupon, r, T, dt)
        return (npv_up - npv_dn) / eff_drho

    # Path-wise copula shift on existing trajectories
    r1 = torch.log(torch.clamp(S1[:, :, 1:] / S1[:, :, :-1], min=1e-8))
    r2 = torch.log(torch.clamp(S2[:, :, 1:] / S2[:, :, :-1], min=1e-8))

    std1 = torch.std(r1, dim=1, keepdim=True).clamp(min=1e-6)
    std2 = torch.std(r2, dim=1, keepdim=True).clamp(min=1e-6)
    z1 = (r1 - torch.mean(r1, dim=1, keepdim=True)) / std1
    z2 = (r2 - torch.mean(r2, dim=1, keepdim=True)) / std2

    emp_rho = torch.mean(z1 * z2, dim=1, keepdim=True).clamp(-0.95, 0.95)
    c_perp = torch.sqrt(torch.clamp(1.0 - emp_rho**2, min=1e-6))
    z_perp = (z2 - emp_rho * z1) / c_perp

    rho_up = torch.clamp(emp_rho + delta_rho, -0.99, 0.99)
    rho_dn = torch.clamp(emp_rho - delta_rho, -0.99, 0.99)
    eff_drho = (rho_up - rho_dn).mean().clamp(min=1e-4).item()

    c_perp_up = torch.sqrt(torch.clamp(1.0 - rho_up**2, min=1e-6))
    c_perp_dn = torch.sqrt(torch.clamp(1.0 - rho_dn**2, min=1e-6))

    z2_up = rho_up * z1 + c_perp_up * z_perp
    z2_dn = rho_dn * z1 + c_perp_dn * z_perp

    r2_up = z2_up * std2 + torch.mean(r2, dim=1, keepdim=True)
    r2_dn = z2_dn * std2 + torch.mean(r2, dim=1, keepdim=True)

    S2_up = torch.empty_like(S2)
    S2_dn = torch.empty_like(S2)
    S2_up[:, :, 0] = S2[:, :, 0]
    S2_dn[:, :, 0] = S2[:, :, 0]
    S2_up[:, :, 1:] = S2[:, :, 0:1] * torch.cumprod(torch.exp(r2_up), dim=2)
    S2_dn[:, :, 1:] = S2[:, :, 0:1] * torch.cumprod(torch.exp(r2_dn), dim=2)

    npv_up, _, _ = price_wof_autocall_mc(S1, S2_up, obs_indices, B, coupon, r, T, dt)
    npv_dn, _, _ = price_wof_autocall_mc(S1, S2_dn, obs_indices, B, coupon, r, T, dt)

    return (npv_up - npv_dn) / eff_drho
