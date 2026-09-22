"""
Unit and regression test suite for Worst-of Autocall MC and EGNO Surrogate.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from deepvol.models.wof_autocall import (
    simulate_correlated_heston_paths,
    price_wof_autocall_mc,
    correlation_sensitivity,
)
from deepvol.surrogates.wof_autocall_egno import WoFAutocallEGNO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B_BATCH = 2
N_PATHS = 5_000
N_STEPS = 63   # 3-month for speed
T = 0.25
OBS_INDICES = [21, 42, 63]   # monthly for 3m


def _make_heston_params(B_batch: int, device: str = DEVICE) -> torch.Tensor:
    theta = torch.zeros(B_batch, 5, dtype=torch.float64, device=device)
    theta[:, 0] = 2.0   # kappa
    theta[:, 1] = 0.04  # theta
    theta[:, 2] = 0.40  # sigma
    theta[:, 3] = -0.70 # rho_sv
    theta[:, 4] = 0.04  # v0
    return theta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_correlated_paths_shape():
    """Simulated paths have correct shape."""
    theta1 = _make_heston_params(B_BATCH)
    theta2 = _make_heston_params(B_BATCH)
    rho_12 = torch.tensor([0.5, 0.3], dtype=torch.float64, device=DEVICE)
    r = 0.03
    S1, S2 = simulate_correlated_heston_paths(theta1, theta2, rho_12, 100.0, 100.0, T, N_STEPS, N_PATHS, r, torch.device(DEVICE))
    assert S1.shape == (B_BATCH, N_PATHS, N_STEPS + 1), f"S1 shape mismatch: {S1.shape}"
    assert S2.shape == (B_BATCH, N_PATHS, N_STEPS + 1), f"S2 shape mismatch: {S2.shape}"
    assert not S1.isnan().any(), "NaN in S1"
    assert not S2.isnan().any(), "NaN in S2"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_identical_assets_matches_vanilla():
    """With identical asset params and rho_12=1.0, WoF NPV == single-asset autocall NPV."""
    from deepvol.models.autocall import price_autocall_mc
    theta = _make_heston_params(1)
    r = 0.03
    rho_12 = torch.tensor([1.0], dtype=torch.float64, device=DEVICE)
    S1, S2 = simulate_correlated_heston_paths(theta, theta, rho_12, 100.0, 100.0, T, N_STEPS, N_PATHS, r, torch.device(DEVICE))
    B = torch.tensor([1.0], dtype=torch.float64, device=DEVICE)
    coupon = torch.tensor([0.02], dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([r], dtype=torch.float64, device=DEVICE)

    npv_wof, cp_wof, el_wof = price_wof_autocall_mc(S1, S2, OBS_INDICES, B, coupon, r_t, T, T / N_STEPS)
    npv_va, cp_va, el_va = price_autocall_mc(S1, OBS_INDICES, B, coupon, r_t, T=T, dt=T / N_STEPS)
    assert torch.allclose(npv_wof, npv_va, atol=5e-3), f"WoF vs vanilla mismatch: {npv_wof.item():.6f} vs {npv_va.item():.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wof_le_vanilla():
    """WoF NPV <= vanilla single-asset NPV for same underlying params."""
    from deepvol.models.autocall import price_autocall_mc
    theta = _make_heston_params(B_BATCH)
    r = 0.03
    rho_12 = torch.tensor([0.3, 0.5], dtype=torch.float64, device=DEVICE)
    S1, S2 = simulate_correlated_heston_paths(theta, theta, rho_12, 100.0, 100.0, T, N_STEPS, N_PATHS, r, torch.device(DEVICE))
    B = torch.ones(B_BATCH, dtype=torch.float64, device=DEVICE)
    coupon = torch.full((B_BATCH,), 0.02, dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([r], dtype=torch.float64, device=DEVICE)

    npv_wof, _, _ = price_wof_autocall_mc(S1, S2, OBS_INDICES, B, coupon, r_t, T, T / N_STEPS)
    npv_va, _, _ = price_autocall_mc(S1, OBS_INDICES, B, coupon, r_t, T=T, dt=T / N_STEPS)
    assert (npv_wof <= npv_va + 5e-3).all(), "WoF must be <= vanilla for same asset"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_higher_rho_higher_price():
    """NPV with rho_12=0.9 > NPV with rho_12=0.1."""
    theta = _make_heston_params(1)
    r = 0.03
    B = torch.tensor([1.0], dtype=torch.float64, device=DEVICE)
    coupon = torch.tensor([0.02], dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([r], dtype=torch.float64, device=DEVICE)

    results = {}
    for rho_val in [0.1, 0.9]:
        rho_12 = torch.tensor([rho_val], dtype=torch.float64, device=DEVICE)
        S1, S2 = simulate_correlated_heston_paths(theta, theta, rho_12, 100.0, 100.0, T, N_STEPS, 15_000, r, torch.device(DEVICE))
        npv, _, _ = price_wof_autocall_mc(S1, S2, OBS_INDICES, B, coupon, r_t, T, T / N_STEPS)
        results[rho_val] = float(npv.item())

    assert results[0.9] > results[0.1] - 1e-4, f"High rho={results[0.9]:.6f} should exceed low rho={results[0.1]:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_permutation_equivariance():
    """Swapping asset 1 and asset 2 params in EGNO surrogate -> identical price."""
    model = WoFAutocallEGNO().to(DEVICE).eval()
    B_batch = 8
    x_nodes = torch.randn(B_batch, 2, 6, device=DEVICE)
    x_perm = x_nodes[:, [1, 0], :]
    # Symmetric asset correlation edge matrix: e[i, i] = 1, e[i, j] = rho
    rho = torch.rand(B_batch, 1, 1, 1, device=DEVICE)
    ones = torch.ones_like(rho)
    row0 = torch.cat([ones, rho], dim=2)
    row1 = torch.cat([rho, ones], dim=2)
    e = torch.cat([row0, row1], dim=1)
    g = torch.rand(B_batch, 5, device=DEVICE)

    with torch.no_grad():
        out = model(x_nodes, e, g)
        out_perm = model(x_perm, e, g)

    assert torch.allclose(out, out_perm, atol=1e-5), f"EGNO not permutation-equivariant: max diff={(out - out_perm).abs().max().item():.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_correlation_sensitivity_sign():
    """dcall_prob/drho_12 > 0."""
    theta = _make_heston_params(1)
    r = 0.03
    B = torch.tensor([0.95], dtype=torch.float64, device=DEVICE)
    coupon = torch.tensor([0.02], dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([r], dtype=torch.float64, device=DEVICE)
    rho_12 = torch.tensor([0.5], dtype=torch.float64, device=DEVICE)

    S1, S2 = simulate_correlated_heston_paths(theta, theta, rho_12, 100.0, 100.0, T, N_STEPS, 10_000, r, torch.device(DEVICE))
    sens = correlation_sensitivity(S1, S2, OBS_INDICES, B, coupon, r_t, T, T / N_STEPS)
    assert sens.item() > 0.0
