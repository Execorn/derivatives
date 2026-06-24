import torch
import numpy as np
import pytest
from deepvol.hedging.deep_hedging import HedgingPolicy
from deepvol.hedging.barrier_hedging import BarrierHedgingEnv


def simulate_gbm_paths(S0, mu, sigma, T, steps, N_paths, device="cpu"):
    dt = T / steps
    t_grid = torch.arange(steps + 1, device=device) * dt
    W = torch.randn(N_paths, steps, device=device)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
    S = S0 * torch.exp(torch.cumsum(log_returns, dim=-1))
    S0_col = torch.full((N_paths, 1), S0, device=device)
    S_full = torch.cat([S0_col, S], dim=-1)
    
    # Prepend dummy volatility
    vol = torch.full_like(S_full, sigma)
    
    H = torch.stack([S_full, vol], dim=-1)
    return H, t_grid


def test_barrier_hedging_env_knockout():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulate 50 paths of 20 steps
    # We will enforce that some paths breach the barrier of 85.0
    H, t_grid = simulate_gbm_paths(S0=100.0, mu=-0.1, sigma=0.3, T=0.1, steps=20, N_paths=50, device=device)
    
    # Manually overwrite the first path to guarantee it breaches the barrier
    H[0, 10:, 0] = 80.0
    
    # Manually overwrite the second path to guarantee it stays above the barrier
    H[1, :, 0] = 110.0
    
    cost_coeffs = torch.tensor([0.0001, 0.0], device=device)
    
    env = BarrierHedgingEnv(
        H=H,
        cost_coeffs=cost_coeffs,
        strike=100.0,
        barrier=85.0,
        expiry=0.1,
        risk_aversion=1.0,
        risk_measure="entropic",
        t_grid=t_grid
    )
    
    policy = HedgingPolicy(input_dim=5, hidden_dim=16, output_dim=2).to(device)  # input dim: log(S/K), log(S/B), T-t, active, prev_delta (2) = 6?
    # Wait, in get_state, features are: log_moneyness (1), log_barrier_dist (1), time_to_expiry (1), active_mask (1), prev_delta (d)
    # Total input size: 4 + d. Since d = 2 here, input_dim is 6.
    policy = HedgingPolicy(input_dim=6, hidden_dim=16, output_dim=2).to(device)
    
    wealth, total_costs, all_deltas = env.simulate_hedging_episode(policy)
    
    # Verify outputs
    assert wealth.shape == (50,)
    assert total_costs.shape == (50,)
    assert all_deltas.shape == (50, 20, 2)
    
    # Verify knockout indicator
    # Path 0: breached barrier -> final payoff must be 0
    assert env.payoff[0].item() == 0.0
    
    # Path 1: spot 110 at maturity, did not breach 85 -> final payoff must be Call payoff = 110 - 100 = 10.0
    assert abs(env.payoff[1].item() - 10.0) < 1e-4
    
    loss = env.compute_loss(wealth)
    assert not torch.isnan(loss)
    
    
def test_barrier_state_integrity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, t_grid = simulate_gbm_paths(S0=100.0, mu=0.0, sigma=0.2, T=0.1, steps=5, N_paths=10, device=device)
    
    env = BarrierHedgingEnv(
        H=H,
        cost_coeffs=torch.tensor([0.0], device=device),
        strike=100.0,
        barrier=85.0,
        expiry=0.1,
        t_grid=t_grid
    )
    
    # Test state construction for step 0
    prev_delta = torch.zeros(10, 1, device=device)
    active_mask = torch.ones(10, 1, device=device)
    
    state = env.get_state(k=0, prev_delta=prev_delta, active_mask=active_mask)
    
    # state dimensions: log(S_k/K) (1) + log(S_k/B) (1) + T-t_k (1) + active_mask (1) + prev_delta (1) = 5
    assert state.shape == (10, 5)
    assert not torch.any(torch.isnan(state))
    assert not torch.any(torch.isinf(state))
