"""
Unit and regression test suite for Phoenix Autocall Monte Carlo Engine.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from deepvol.models.phoenix import price_phoenix_mc, phoenix_decomposition

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B_BATCH = 4
N_PATHS = 10_000
N_STEPS = 252
OBS_INDICES = [63, 126, 189, 252]  # quarterly for T=1y
DT = 1.0 / 252


def make_paths(B_batch: int, N_paths: int, N_steps: int, device: str = DEVICE) -> torch.Tensor:
    """Flat paths at S=1.0."""
    return torch.ones(B_batch, N_paths, N_steps + 1, device=device, dtype=torch.float64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_payoff_always_called():
    """B_call=0.001 -> S/S0 always >= B_call -> call_prob ≈ 1.0, npv ≈ exp(-r*t1)*(1+coupon*1)."""
    S = make_paths(B_BATCH, N_PATHS, N_STEPS)
    B_call = torch.full((B_BATCH,), 0.001, dtype=torch.float64, device=DEVICE)
    B_cpn = torch.full((B_BATCH,), 0.0005, dtype=torch.float64, device=DEVICE)
    coupon = torch.full((B_BATCH,), 0.02, dtype=torch.float64, device=DEVICE)
    r = torch.full((B_BATCH,), 0.05, dtype=torch.float64, device=DEVICE)

    npv, call_prob, cpn_prob, exp_life = price_phoenix_mc(
        S, OBS_INDICES, B_call, B_cpn, coupon, r, T=1.0, dt=DT, memory=False
    )
    assert (call_prob > 0.99).all(), f"Expected call_prob≈1, got {call_prob}"
    expected_npv = torch.exp(-r * (OBS_INDICES[0] * DT)) * (1.0 + coupon * 1.0)
    assert torch.allclose(npv, expected_npv, atol=1e-3), f"NPV mismatch: {npv} vs {expected_npv}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coupon_corridor():
    """B_cpn=0.001, B_call=1e6 -> coupon always triggered, product never called."""
    S = make_paths(B_BATCH, N_PATHS, N_STEPS)
    B_call = torch.full((B_BATCH,), 1e6, dtype=torch.float64, device=DEVICE)
    B_cpn = torch.full((B_BATCH,), 0.001, dtype=torch.float64, device=DEVICE)
    coupon = torch.full((B_BATCH,), 0.02, dtype=torch.float64, device=DEVICE)
    r = torch.full((B_BATCH,), 0.05, dtype=torch.float64, device=DEVICE)

    npv, call_prob, cpn_prob, exp_life = price_phoenix_mc(
        S, OBS_INDICES, B_call, B_cpn, coupon, r, T=1.0, dt=DT, memory=False
    )
    assert (call_prob < 0.01).all(), "Expected call_prob≈0"
    assert (cpn_prob > 0.99).all(), "Expected cpn_prob≈1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_memory_premium():
    """memory=True price >= memory=False price for same paths (monotone)."""
    S = make_paths(B_BATCH, N_PATHS, N_STEPS)
    B_call = torch.full((B_BATCH,), 1.0, dtype=torch.float64, device=DEVICE)
    B_cpn = torch.full((B_BATCH,), 0.75, dtype=torch.float64, device=DEVICE)
    coupon = torch.full((B_BATCH,), 0.02, dtype=torch.float64, device=DEVICE)
    r = torch.full((B_BATCH,), 0.05, dtype=torch.float64, device=DEVICE)

    npv_mem, *_ = price_phoenix_mc(S, OBS_INDICES, B_call, B_cpn, coupon, r, 1.0, DT, memory=True)
    npv_no, *_ = price_phoenix_mc(S, OBS_INDICES, B_call, B_cpn, coupon, r, 1.0, DT, memory=False)
    assert (npv_mem >= npv_no - 1e-6).all(), "Memory premium violated"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_phoenix_dominates_vanilla():
    """Phoenix NPV >= vanilla autocall NPV."""
    from deepvol.models.autocall import price_autocall_mc
    S = make_paths(B_BATCH, N_PATHS, N_STEPS)
    B_call = torch.full((B_BATCH,), 1.0, dtype=torch.float64, device=DEVICE)
    B_cpn = torch.full((B_BATCH,), 0.75, dtype=torch.float64, device=DEVICE)
    coupon = torch.full((B_BATCH,), 0.02, dtype=torch.float64, device=DEVICE)
    r = torch.full((B_BATCH,), 0.05, dtype=torch.float64, device=DEVICE)

    npv_ph, call_prob, cpn_prob, _ = price_phoenix_mc(S, OBS_INDICES, B_call, B_cpn, coupon, r, 1.0, DT, memory=False)
    npv_va, *_ = price_autocall_mc(S, OBS_INDICES, B_call, coupon, r, T=1.0, dt=DT)
    assert (npv_ph >= npv_va - 1e-6).all(), "Phoenix must dominate vanilla autocall"

    decomp = phoenix_decomposition(npv_ph, npv_va, call_prob, cpn_prob)
    assert (decomp["coupon_corridor_leg"] >= -1e-6).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_b_cpn_must_be_below_b_call():
    """ValueError raised if B_cpn >= B_call."""
    S = make_paths(1, N_PATHS, N_STEPS)
    B_call = torch.tensor([0.80], dtype=torch.float64, device=DEVICE)
    B_cpn = torch.tensor([0.90], dtype=torch.float64, device=DEVICE)
    with pytest.raises(ValueError, match="B_cpn must be strictly less than B_call"):
        price_phoenix_mc(
            S, OBS_INDICES, B_call, B_cpn,
            torch.tensor([0.02], device=DEVICE, dtype=torch.float64),
            torch.tensor([0.05], device=DEVICE, dtype=torch.float64),
            1.0, DT, memory=False
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_call_prob_bounds():
    """call_prob and cpn_prob must lie in [0, 1]."""
    S = make_paths(B_BATCH, N_PATHS, N_STEPS)
    B_call = torch.rand(B_BATCH, dtype=torch.float64, device=DEVICE) * 0.4 + 0.8
    B_cpn = B_call - 0.10
    coupon = torch.rand(B_BATCH, dtype=torch.float64, device=DEVICE) * 0.05
    r = torch.rand(B_BATCH, dtype=torch.float64, device=DEVICE) * 0.05

    npv, call_prob, cpn_prob, exp_life = price_phoenix_mc(
        S, OBS_INDICES, B_call, B_cpn, coupon, r, 1.0, DT, memory=False
    )
    assert (call_prob >= 0.0).all() and (call_prob <= 1.0).all()
    assert (cpn_prob >= 0.0).all() and (cpn_prob <= 1.0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exp_life_le_T():
    """Expected life must not exceed tenor T."""
    S = make_paths(B_BATCH, N_PATHS, N_STEPS)
    B_call = torch.rand(B_BATCH, dtype=torch.float64, device=DEVICE) * 0.4 + 0.8
    B_cpn = B_call - 0.15
    coupon = torch.rand(B_BATCH, dtype=torch.float64, device=DEVICE) * 0.04
    r = torch.zeros(B_BATCH, dtype=torch.float64, device=DEVICE)
    T = 1.0
    _, _, _, exp_life = price_phoenix_mc(
        S, OBS_INDICES, B_call, B_cpn, coupon, r, T, DT, memory=False
    )
    assert (exp_life <= T + 1e-6).all(), f"exp_life exceeds T: {exp_life}"
