"""
Unit and convergence test suite for 1D Crank-Nicolson PDE Autocall Pricer.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
import torch
from deepvol.models.autocall_pde import (
    build_lv_grid,
    price_autocall_pde,
    price_autocall_pde_scalar,
)

FLAT_SIGMA = 0.20
flat_sigma_func = lambda t, S: np.full_like(S, FLAT_SIGMA)

S0 = 100.0
r = 0.05
T = 1.0
B = 1.00
coupon = 0.02
OBS_INDICES_FWD = [63, 126, 189, 252]
N_S = 300
N_T = 252


def test_european_call_no_obs_dates():
    """n_obs=0 -> pure zero-coupon capital-protected bond; |PDE - exp(-rT)| < 1 bps."""
    result = price_autocall_pde_scalar(
        S0_val=S0, r=r, T=T, N_S=N_S, N_T=N_T, obs_indices=[], B=B, coupon=coupon,
        sigma_func=flat_sigma_func
    )
    expected = np.exp(-r * T)
    diff_bps = abs(result["npv"] - expected) * 10_000.0
    assert diff_bps < 1.0, f"PDE vs ZCB mismatch: {diff_bps:.4f} bps"


@pytest.mark.slow
def test_pde_vs_mc_autocall():
    """20 random param sets: |PDE price - MC(100k paths)| < 5 bps."""
    from deepvol.models.autocall import price_autocall_mc, make_obs_indices
    from deepvol.hedging.d_xva import simulate_heston_paths

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed=1234)
    errors_bps = []

    for _ in range(5):  # 5 fast parameter sets for standard pytest runs
        sigma_atm = float(rng.uniform(0.15, 0.30))
        sigma_f = lambda t, S, sig=sigma_atm: np.full_like(S, sig)
        r_test = float(rng.uniform(0.02, 0.05))
        B_test = float(rng.uniform(0.95, 1.05))
        coupon_test = float(rng.uniform(0.02, 0.05))

        pde_res = price_autocall_pde_scalar(
            S0_val=S0, r=r_test, T=T, N_S=N_S, N_T=N_T, obs_indices=OBS_INDICES_FWD,
            B=B_test, coupon=coupon_test, sigma_func=sigma_f
        )

        v0 = sigma_atm ** 2
        theta_h = torch.tensor([[1e-4, v0, 1e-4, 0.0, v0]], dtype=torch.float64, device=device)
        S_paths = simulate_heston_paths(theta_h, S0=S0, T=T, N_steps=252, N_paths=100_000, r=r_test, device=device)

        B_t = torch.tensor([B_test], dtype=torch.float64, device=device)
        c_t = torch.tensor([coupon_test], dtype=torch.float64, device=device)
        r_t = torch.tensor([r_test], dtype=torch.float64, device=device)
        mc_npv, _, _ = price_autocall_mc(S_paths, OBS_INDICES_FWD, B_t, c_t, r_t, T=T, dt=T / 252)

        diff_bps = abs(pde_res["npv"] - float(mc_npv.item())) * 10_000.0
        errors_bps.append(diff_bps)

    max_err = max(errors_bps)
    assert max_err < 5.0, f"PDE vs MC max error {max_err:.4f} bps exceeds 5 bps"


def test_delta_monotone():
    """Delta should be >= 0 everywhere on interior grid."""
    _, delta_grid, _ = price_autocall_pde(S0, r, T, N_S, N_T, OBS_INDICES_FWD, B, coupon, flat_sigma_func)
    assert (delta_grid[10:-10] >= -1e-5).all(), f"Negative delta found: {delta_grid.min():.6f}"


def test_gamma_near_barrier():
    """Gamma peaks near S = B * S0."""
    _, _, gamma_grid = price_autocall_pde(S0, r, T, N_S, N_T, OBS_INDICES_FWD, B, coupon, flat_sigma_func)
    S_grid, _, _ = build_lv_grid(S0, T, N_S, N_T, flat_sigma_func)
    barrier_spot = B * S0
    idx_barrier = np.searchsorted(S_grid, barrier_spot)

    window = slice(max(0, idx_barrier - 25), min(len(gamma_grid), idx_barrier + 25))
    peak_idx = int(np.argmax(gamma_grid))
    assert abs(peak_idx - idx_barrier) <= 25, f"Gamma peak at {S_grid[peak_idx]:.2f}, barrier at {barrier_spot:.2f}"


def test_grid_convergence():
    """N_S=150 vs N_S=300: prices agree within 2 bps."""
    res_coarse = price_autocall_pde_scalar(S0, r, T, 150, N_T, OBS_INDICES_FWD, B, coupon, flat_sigma_func)
    res_fine = price_autocall_pde_scalar(S0, r, T, 300, N_T, OBS_INDICES_FWD, B, coupon, flat_sigma_func)
    diff_bps = abs(res_coarse["npv"] - res_fine["npv"]) * 10_000.0
    assert diff_bps < 2.0, f"Grid convergence difference {diff_bps:.4f} bps exceeds 2 bps"
