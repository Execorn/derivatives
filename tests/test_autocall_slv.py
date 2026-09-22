"""
Unit and integration test suite for Autocall Local Volatility (LV) and
Stochastic Local Volatility (SLV) pricing engine.
"""

import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from deepvol.models.autocall_slv import (
    build_dupire_vol_fn,
    simulate_lv_paths,
    price_autocall_lv_mc,
    price_autocall_slv_mc,
)
from deepvol.models.autocall import make_obs_indices

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SKIP_CPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_dummy_svi_grid(device: str = DEVICE):
    """Create a standard SVI parameter grid across maturities."""
    T_grid = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float64, device=device)
    K_grid = torch.linspace(-0.5, 0.5, 21, dtype=torch.float64, device=device)
    # Raw SVI: (a, b, rho, m, sigma)
    nT = len(T_grid)
    svi_params = torch.zeros(nT, 5, dtype=torch.float64, device=device)
    for i in range(nT):
        # a=0.04, b=0.1, rho=-0.4, m=0.0, sigma=0.1
        svi_params[i] = torch.tensor([0.04, 0.1, -0.4, 0.0, 0.1], dtype=torch.float64, device=device)
    return T_grid, K_grid, svi_params


@SKIP_CPU
def test_build_dupire_vol_fn_CUDA():
    """Verify Dupire local volatility interpolation and 100 bps clamp."""
    T_grid, K_grid, svi_params = _make_dummy_svi_grid()
    vol_fn = build_dupire_vol_fn(T_grid, K_grid, svi_params, device=torch.device(DEVICE), S0_ref=100.0)

    # Test spot grid
    S = torch.tensor([50.0, 80.0, 100.0, 120.0, 200.0], dtype=torch.float64, device=DEVICE)
    vol_val = vol_fn(0.5, S)

    assert vol_val.shape == S.shape
    assert not torch.isnan(vol_val).any()
    assert (vol_val >= 0.01).all(), "Durrleman 100 bps minimum vol clamp violated"
    assert (vol_val <= 2.0).all(), "Upper volatility clamp exceeded"


@SKIP_CPU
def test_simulate_lv_paths_CUDA():
    """Simulate spot paths under Local Volatility using log-Euler scheme."""
    T_grid, K_grid, svi_params = _make_dummy_svi_grid()
    vol_fn = build_dupire_vol_fn(T_grid, K_grid, svi_params, device=torch.device(DEVICE), S0_ref=100.0)

    S0 = 100.0
    r = 0.03
    T = 1.0
    N_steps = 126
    N_paths = 2000

    S_paths = simulate_lv_paths(S0, r, T, N_steps, N_paths, vol_fn, device=torch.device(DEVICE))

    assert S_paths.shape == (1, N_paths, N_steps + 1)
    assert not torch.isnan(S_paths).any(), "NaN in LV simulated paths"
    assert (S_paths > 0.0).all(), "Negative asset price encountered in LV simulation"
    assert torch.allclose(S_paths[:, :, 0], torch.tensor(S0, dtype=torch.float64, device=DEVICE))


@SKIP_CPU
def test_price_autocall_lv_mc_CUDA():
    """Price autocallable note under pure Dupire Local Volatility."""
    T_grid, K_grid, svi_params = _make_dummy_svi_grid()

    S0 = 100.0
    r = 0.03
    T = 1.0
    N_steps = 126
    N_paths = 5000
    n_obs = 4
    obs_indices = make_obs_indices(n_obs, T, N_steps)
    B = 1.0
    coupon = 0.08

    res = price_autocall_lv_mc(
        S0=S0,
        r=r,
        T=T,
        N_steps=N_steps,
        N_paths=N_paths,
        obs_indices=obs_indices,
        B_scalar=B,
        coupon_scalar=coupon,
        svi_params=svi_params,
        T_grid=T_grid,
        K_grid=K_grid,
        device=torch.device(DEVICE),
    )

    npv = res["npv"]
    call_prob = res["call_prob"]
    exp_life = res["exp_life"]

    floor = math.exp(-r * T) * 1.0
    t1 = T / n_obs
    ceiling = math.exp(-r * t1) * (1.0 + coupon * t1)

    assert floor - 0.01 <= npv <= ceiling + 0.01, f"NPV {npv} outside valid bounds [{floor}, {ceiling}]"
    assert 0.0 <= call_prob <= 1.0, f"Call prob {call_prob} outside [0, 1]"
    assert 0.0 <= exp_life <= T + 1e-4, f"Exp life {exp_life} outside [0, {T}]"


@SKIP_CPU
def test_price_autocall_slv_mc_CUDA():
    """Price autocallable note under McKean-Vlasov SLV particle simulation."""
    T_grid, K_grid, svi_params = _make_dummy_svi_grid()

    S0 = 100.0
    r = 0.03
    q = 0.0
    T = 0.5
    N_steps = 63
    N_paths = 1000
    n_obs = 2
    obs_indices = make_obs_indices(n_obs, T, N_steps)
    B = 1.0
    coupon = 0.06

    slv_params = {
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.3,
        "rho": -0.6,
    }

    res = price_autocall_slv_mc(
        S0=S0,
        r=r,
        q=q,
        slv_params=slv_params,
        T=T,
        N_steps=N_steps,
        N_paths=N_paths,
        obs_indices=obs_indices,
        B_scalar=B,
        coupon_scalar=coupon,
        svi_params=svi_params,
        T_grid=T_grid,
        K_grid=K_grid,
        device=torch.device(DEVICE),
        method="nadaraya_watson",
    )

    npv = res["npv"]
    call_prob = res["call_prob"]
    exp_life = res["exp_life"]

    assert not math.isnan(npv), "NaN in SLV autocall NPV"
    assert 0.0 <= call_prob <= 1.0, f"SLV Call prob {call_prob} outside [0, 1]"
    assert 0.0 <= exp_life <= T + 1e-4, f"SLV Exp life {exp_life} outside [0, {T}]"
