"""
Unit and behavior test suite for Autocall Deep Hedging Environment and Policy.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from deepvol.hedging.autocall_hedging import (
    AutocallHedgingEnv,
    AutocallHedgePolicy,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PATHS = 200
N_T = 63     # 3-month horizon
D = 1        # underlying only
STATE_DIM = 7
OBS_INDICES = [21, 42, 63]


def _make_env(device: str = DEVICE) -> AutocallHedgingEnv:
    torch.manual_seed(42)
    H = torch.randn(N_PATHS, N_T + 1, D, device=device, dtype=torch.float32)
    H = torch.cumsum(H * 0.01, dim=1) + 100.0
    cost_coeffs = torch.tensor([0.001], device=device, dtype=torch.float32)
    return AutocallHedgingEnv(
        H=H,
        cost_coeffs=cost_coeffs,
        S0=100.0,
        strike=100.0,
        B_call=1.0,
        coupon=0.02,
        r=0.05,
        T=0.25,
        obs_indices=OBS_INDICES,
        risk_aversion=1.0,
        risk_measure="entropic",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_state_dim():
    """get_state returns (N_paths, STATE_DIM) tensor."""
    env = _make_env()
    state = env.get_state(0)
    assert state.shape == (N_PATHS, STATE_DIM), f"State dim mismatch: {state.shape}"
    assert not state.isnan().any(), "NaN in initial state"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_episode_no_crash():
    """simulate_episode completes without NaN/Inf."""
    env = _make_env()
    policy = AutocallHedgePolicy(state_dim=STATE_DIM, n_instruments=D).to(DEVICE)
    wealth, payoff, pnl_per_step = env.simulate_episode(policy)
    assert not wealth.isnan().any(), "NaN in terminal wealth"
    assert not payoff.isnan().any(), "NaN in payoff"
    assert pnl_per_step.shape == (N_PATHS, N_T), f"pnl_per_step shape wrong: {pnl_per_step.shape}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_policy_output_clamped():
    """All delta outputs lie in [-2, 2]."""
    policy = AutocallHedgePolicy(state_dim=STATE_DIM, n_instruments=D).to(DEVICE)
    state = torch.randn(N_PATHS, STATE_DIM, device=DEVICE) * 100.0
    delta, _, _ = policy(state)
    assert (delta >= -2.0 - 1e-5).all() and (delta <= 2.0 + 1e-5).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_loss_backward():
    """Gradient flows through entropic loss."""
    env = _make_env()
    policy = AutocallHedgePolicy(state_dim=STATE_DIM, n_instruments=D).to(DEVICE)
    wealth, payoff, _ = env.simulate_episode(policy)
    loss = env.compute_loss(wealth, payoff)
    loss.backward()
    for name, param in policy.named_parameters():
        assert param.grad is not None, f"No gradient for param: {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_loss_backward_cvar():
    """Gradient flows through CVaR loss."""
    env = _make_env()
    env.risk_measure = "cvar"
    policy = AutocallHedgePolicy(state_dim=STATE_DIM, n_instruments=D).to(DEVICE)
    wealth, payoff, _ = env.simulate_episode(policy)
    loss = env.compute_loss(wealth, payoff)
    loss.backward()
    for name, param in policy.named_parameters():
        assert param.grad is not None, f"No gradient for param: {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_cost_baseline():
    """Zero delta policy has zero trading wealth."""
    env = _make_env()
    zero_policy = AutocallHedgePolicy(state_dim=STATE_DIM, n_instruments=D).to(DEVICE)
    for p in zero_policy.parameters():
        p.data.zero_()
    wealth, payoff, _ = env.simulate_episode(zero_policy)
    assert wealth.abs().max() < 1e-3, f"Expected near-zero wealth, got {wealth.abs().max():.4f}"
