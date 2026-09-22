"""
test_autocall_mc.py — Unit and Numerical Verification Tests for Autocall MC Engine.
"""

import math
import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from deepvol.models.autocall import (
    make_obs_indices,
    price_autocall_mc,
    autocall_upper_bound,
)
from deepvol.hedging.d_xva import simulate_heston_paths

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SKIP_CPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_make_obs_indices_quarterly():
    """Quarterly schedule: 4 obs over 1.0 year with 252 steps -> [63, 126, 189, 252]."""
    indices = make_obs_indices(4, 1.0, 252)
    assert indices == [63, 126, 189, 252]


def test_make_obs_indices_boundary():
    """Observation at maturity T must always yield last index equal to N_steps."""
    for n_obs in [2, 4, 8, 12]:
        for N_steps in [100, 252, 504]:
            indices = make_obs_indices(n_obs, 2.0, N_steps)
            assert indices[-1] == N_steps
            assert len(indices) == n_obs
            assert all(1 <= idx <= N_steps for idx in indices)
            assert indices == sorted(indices)


@SKIP_CPU
def test_payoff_always_called_CUDA():
    """Extreme low barrier (B=0.0001) -> call at first obs t_1 with prob ~ 1.0."""
    B_val = 0.0001
    coupon_val = 0.10
    r_val = 0.03
    T = 1.0
    n_obs = 4
    N_steps = 252
    dt = T / N_steps
    obs_indices = make_obs_indices(n_obs, T, N_steps)
    t1 = T / n_obs

    # Heston parameters
    theta = torch.tensor([[2.0, 0.04, 0.3, -0.7, 0.04]], dtype=torch.float64, device=DEVICE)
    S = simulate_heston_paths(theta, 100.0, T, N_steps, N_paths=10000, r=0.0, device=DEVICE)

    B_t = torch.tensor([B_val], dtype=torch.float64, device=DEVICE)
    coupon_t = torch.tensor([coupon_val], dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([r_val], dtype=torch.float64, device=DEVICE)

    npv, call_prob, exp_life = price_autocall_mc(
        S, obs_indices, B_t, coupon_t, r_t, T, dt
    )

    expected_npv = math.exp(-r_val * t1) * (1.0 + coupon_val * t1)

    assert abs(call_prob.item() - 1.0) < 1e-4
    assert abs(exp_life.item() - t1) < 1e-3
    assert abs(npv.item() - expected_npv) < 1e-3


@SKIP_CPU
def test_payoff_never_called_CUDA():
    """Extreme high barrier (B=1e6) -> never called, capital protected at maturity T."""
    B_val = 1e6
    coupon_val = 0.10
    r_val = 0.05
    T = 1.0
    n_obs = 4
    N_steps = 252
    dt = T / N_steps
    obs_indices = make_obs_indices(n_obs, T, N_steps)

    theta = torch.tensor([[2.0, 0.04, 0.3, -0.7, 0.04]], dtype=torch.float64, device=DEVICE)
    S = simulate_heston_paths(theta, 100.0, T, N_steps, N_paths=10000, r=0.0, device=DEVICE)

    B_t = torch.tensor([B_val], dtype=torch.float64, device=DEVICE)
    coupon_t = torch.tensor([coupon_val], dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([r_val], dtype=torch.float64, device=DEVICE)

    npv, call_prob, exp_life = price_autocall_mc(
        S, obs_indices, B_t, coupon_t, r_t, T, dt
    )

    expected_npv = math.exp(-r_val * T) * 1.0

    assert abs(call_prob.item() - 0.0) < 1e-4
    assert abs(exp_life.item() - T) < 1e-3
    assert abs(npv.item() - expected_npv) < 1e-3


@SKIP_CPU
def test_call_prob_bounds_CUDA():
    """Verify all call probabilities lie strictly in [0, 1] for 20 random parameter sets."""
    torch.manual_seed(42)
    T = 1.0
    N_steps = 252
    dt = T / N_steps
    obs_indices = make_obs_indices(4, T, N_steps)

    for _ in range(20):
        kappa = float(torch.empty(1).uniform_(0.5, 5.0))
        theta_v = float(torch.empty(1).uniform_(0.01, 0.15))
        sigma_v = float(torch.empty(1).uniform_(0.1, 1.0))
        rho = float(torch.empty(1).uniform_(-0.9, -0.1))
        v0 = float(torch.empty(1).uniform_(0.01, 0.15))
        theta = torch.tensor([[kappa, theta_v, sigma_v, rho, v0]], dtype=torch.float64, device=DEVICE)

        b = float(torch.empty(1).uniform_(0.85, 1.15))
        coupon = float(torch.empty(1).uniform_(0.03, 0.25))
        r = float(torch.empty(1).uniform_(0.0, 0.08))

        B_t = torch.tensor([b], dtype=torch.float64, device=DEVICE)
        coupon_t = torch.tensor([coupon], dtype=torch.float64, device=DEVICE)
        r_t = torch.tensor([r], dtype=torch.float64, device=DEVICE)

        S = simulate_heston_paths(theta, 100.0, T, N_steps, N_paths=5000, r=0.0, device=DEVICE)
        npv, call_prob, exp_life = price_autocall_mc(
            S, obs_indices, B_t, coupon_t, r_t, T, dt
        )

        assert 0.0 <= call_prob.item() <= 1.0
        assert 0.0 <= exp_life.item() <= T + 1e-4


@SKIP_CPU
def test_exp_life_monotone_CUDA():
    """Higher barrier B -> lower call_prob, higher exp_life."""
    T = 1.0
    N_steps = 252
    dt = T / N_steps
    obs_indices = make_obs_indices(4, T, N_steps)

    theta = torch.tensor([[2.0, 0.04, 0.3, -0.7, 0.04]], dtype=torch.float64, device=DEVICE)
    S = simulate_heston_paths(theta, 100.0, T, N_steps, N_paths=20000, r=0.0, device=DEVICE)

    barriers = [0.85, 0.95, 1.00, 1.05, 1.15]
    call_probs = []
    exp_lifes = []

    coupon_t = torch.tensor([0.10], dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([0.03], dtype=torch.float64, device=DEVICE)

    for b in barriers:
        B_t = torch.tensor([b], dtype=torch.float64, device=DEVICE)
        _, cp, el = price_autocall_mc(S, obs_indices, B_t, coupon_t, r_t, T, dt)
        call_probs.append(cp.item())
        exp_lifes.append(el.item())

    # Verify monotonic trends
    for i in range(len(barriers) - 1):
        assert call_probs[i] >= call_probs[i + 1] - 0.01, f"Call prob not decreasing: {call_probs}"
        assert exp_lifes[i] <= exp_lifes[i + 1] + 0.01, f"Exp life not increasing: {exp_lifes}"


@SKIP_CPU
def test_mc_low_variance_CUDA():
    """With 50k paths, std of independent NPV estimates is small (< 0.005)."""
    T = 1.0
    N_steps = 252
    dt = T / N_steps
    obs_indices = make_obs_indices(4, T, N_steps)

    theta = torch.tensor([[2.0, 0.04, 0.3, -0.7, 0.04]], dtype=torch.float64, device=DEVICE)
    B_t = torch.tensor([1.0], dtype=torch.float64, device=DEVICE)
    coupon_t = torch.tensor([0.10], dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([0.03], dtype=torch.float64, device=DEVICE)

    estimates = []
    for _ in range(3):
        S = simulate_heston_paths(theta, 100.0, T, N_steps, N_paths=50000, r=0.0, device=DEVICE)
        npv, _, _ = price_autocall_mc(S, obs_indices, B_t, coupon_t, r_t, T, dt)
        estimates.append(npv.item())

    std_est = float(torch.tensor(estimates).std().item())
    assert std_est < 0.005, f"MC variance too high: std={std_est}"


@SKIP_CPU
def test_upper_bound_CUDA():
    """MC NPV must be <= analytical upper bound for 50 random parameter sets."""
    torch.manual_seed(123)
    n_sets = 50
    T = 1.0
    N_steps = 252
    dt = T / N_steps
    n_obs = 4
    obs_indices = make_obs_indices(n_obs, T, N_steps)

    theta = torch.tensor([[2.0, 0.04, 0.3, -0.7, 0.04]], dtype=torch.float64, device=DEVICE)
    S = simulate_heston_paths(theta, 100.0, T, N_steps, N_paths=10000, r=0.0, device=DEVICE)

    for _ in range(n_sets):
        b = float(torch.empty(1).uniform_(0.85, 1.15))
        coupon = float(torch.empty(1).uniform_(0.03, 0.25))
        r = float(torch.empty(1).uniform_(0.00, 0.08))

        B_t = torch.tensor([b], dtype=torch.float64, device=DEVICE)
        coupon_t = torch.tensor([coupon], dtype=torch.float64, device=DEVICE)
        r_t = torch.tensor([r], dtype=torch.float64, device=DEVICE)

        npv, _, _ = price_autocall_mc(S, obs_indices, B_t, coupon_t, r_t, T, dt)
        ub = autocall_upper_bound(n_obs, coupon, T, r)

        # Allow small MC sampling tolerance
        assert npv.item() <= ub + 0.015, f"MC NPV {npv.item()} exceeded upper bound {ub}"
