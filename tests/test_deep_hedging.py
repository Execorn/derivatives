import torch
import torch.nn as nn
import numpy as np
import pytest
from scipy.stats import norm
from deepvol.hedging.deep_hedging import HedgingPolicy, DeepHedgingEnv, train_deep_hedger


def bs_delta_cpu(S, K, T, t, sigma):
    tau = T - t
    if tau < 1e-6:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def simulate_gbm_paths(S0, mu, sigma, T, steps, N_paths, d=1, device="cpu"):
    dt = T / steps
    t_grid = torch.arange(steps + 1, device=device) * dt
    
    # Simulate log-returns
    W = torch.randn(N_paths, steps, device=device)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
    
    # Cumsum to get stock prices
    S = S0 * torch.exp(torch.cumsum(log_returns, dim=-1))
    S0_col = torch.full((N_paths, 1), S0, device=device)
    S_full = torch.cat([S0_col, S], dim=-1)
    
    if d == 1:
        H = S_full.unsqueeze(-1)
    else:
        # Prepend dummy volatility proxy (constant volatility)
        vol = torch.full_like(S_full, sigma)
        H = torch.stack([S_full, vol], dim=-1)
    return H, t_grid


def test_hedging_policy_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # input_dim = 3 (moneyness, expiry, vol) + 1 (prev_delta) = 4
    policy = HedgingPolicy(input_dim=4, hidden_dim=16, output_dim=1).to(device)
    
    # Batch of 10 samples
    x = torch.randn(10, 4, device=device)
    delta, lstm_state = policy(x)
    
    assert delta.shape == (10, 1)
    assert lstm_state[0].shape == (10, 16)
    assert lstm_state[1].shape == (10, 16)
    assert torch.all(delta >= -2.0) and torch.all(delta <= 2.0)


def test_deep_hedging_env_loop():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulate 100 paths of 30 steps with d=2 instruments
    H, t_grid = simulate_gbm_paths(S0=100.0, mu=0.0, sigma=0.2, T=0.1, steps=30, N_paths=100, d=2, device=device)
    
    # Terminal option payoff: Call option at strike 100
    S_T = H[:, -1, 0]
    payoff = torch.clamp(S_T - 100.0, min=0.0)
    
    cost_coeffs = torch.tensor([0.0001, 0.0005], device=device)  # 1 bp cost on stock, 5 bps on vol
    
    env = DeepHedgingEnv(
        H=H,
        payoff=payoff,
        cost_coeffs=cost_coeffs,
        strike=100.0,
        expiry=0.1,
        risk_aversion=1.0,
        risk_measure="entropic",
        t_grid=t_grid
    )
    
    # input_dim = 3 (moneyness, expiry, vol) + 2 (d=2, prev_delta) = 5
    policy = HedgingPolicy(input_dim=5, hidden_dim=16, output_dim=2).to(device)
    
    wealth, total_costs, all_deltas = env.simulate_hedging_episode(policy)
    
    assert wealth.shape == (100,)
    assert total_costs.shape == (100,)
    assert all_deltas.shape == (100, 30, 2)
    
    loss = env.compute_loss(wealth)
    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_delta_convergence():
    """
    Fast training test: verifies that under zero transaction costs, the neural policy
    converges towards the analytic Black-Scholes delta.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    S0 = 100.0
    K = 100.0
    T = 0.05
    steps = 10
    sigma = 0.2
    N_paths = 512
    
    # Generate GBM paths with d=1 (stock only)
    H, t_grid = simulate_gbm_paths(S0=S0, mu=0.0, sigma=sigma, T=T, steps=steps, N_paths=N_paths, d=1, device=device)
    S_T = H[:, -1, 0]
    payoff = torch.clamp(S_T - K, min=0.0)
    
    # Zero cost coefficients
    cost_coeffs = torch.tensor([0.0], device=device)
    
    env = DeepHedgingEnv(
        H=H,
        payoff=payoff,
        cost_coeffs=cost_coeffs,
        strike=K,
        expiry=T,
        risk_aversion=1.0,
        risk_measure="quad",  # Mean squared error
        t_grid=t_grid
    )
    
    # LSTM policy (input_dim = 3 + 1 = 4, output_dim = 1)
    policy = HedgingPolicy(input_dim=4, hidden_dim=32, output_dim=1).to(device)
    
    # Train for a few epochs
    train_deep_hedger(env, policy, lr=5e-3, epochs=80, batch_size=256, device=device)
    
    # Evaluate and compare with analytic delta
    policy.eval()
    with torch.no_grad():
        wealth, _, all_deltas = env.simulate_hedging_episode(policy)
        
    # Calculate analytic delta along a path
    S_paths = H[:, :, 0].cpu().numpy()
    analytic_deltas = np.zeros((N_paths, steps))
    dt = T / steps
    
    for i in range(N_paths):
        for k in range(steps):
            t = k * dt
            analytic_deltas[i, k] = bs_delta_cpu(S_paths[i, k], K, T, t, sigma)
            
    # Compare generated deltas
    learned_deltas = all_deltas.squeeze(-1).cpu().numpy()
    
    # Compute Mean Squared Error between learned and analytic delta
    mse = np.mean((learned_deltas - analytic_deltas) ** 2)
    print(f"Learned vs BS Delta MSE: {mse:.6f}")
    
    # The learned policy should exhibit strong correlation and low MSE (e.g. < 0.05)
    assert mse < 0.05


def test_cost_awareness():
    """
    Verifies that average hedging turnover decreases as transaction cost coefficients increase.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    
    S0 = 100.0
    K = 100.0
    T = 0.05
    steps = 10
    sigma = 0.2
    N_paths = 512
    
    H, t_grid = simulate_gbm_paths(S0=S0, mu=0.0, sigma=sigma, T=T, steps=steps, N_paths=N_paths, d=1, device=device)
    S_T = H[:, -1, 0]
    payoff = torch.clamp(S_T - K, min=0.0)
    
    # 1. Environment and policy with low transaction cost
    cost_coeffs_low = torch.tensor([0.0], device=device)
    env_low = DeepHedgingEnv(
        H=H,
        payoff=payoff,
        cost_coeffs=cost_coeffs_low,
        strike=K,
        expiry=T,
        risk_aversion=1.0,
        risk_measure="quad",
        t_grid=t_grid
    )
    policy_low = HedgingPolicy(input_dim=4, hidden_dim=32, output_dim=1).to(device)
    
    # 2. Environment and policy with high transaction cost
    cost_coeffs_high = torch.tensor([0.02], device=device)
    env_high = DeepHedgingEnv(
        H=H,
        payoff=payoff,
        cost_coeffs=cost_coeffs_high,
        strike=K,
        expiry=T,
        risk_aversion=1.0,
        risk_measure="quad",
        t_grid=t_grid
    )
    policy_high = HedgingPolicy(input_dim=4, hidden_dim=32, output_dim=1).to(device)
    policy_high.load_state_dict(policy_low.state_dict())
    
    # Train both policies
    train_deep_hedger(env_low, policy_low, lr=5e-3, epochs=15, batch_size=256, device=device)
    train_deep_hedger(env_high, policy_high, lr=5e-3, epochs=15, batch_size=256, device=device)
    
    # Evaluate
    policy_low.eval()
    policy_high.eval()
    
    with torch.no_grad():
        _, _, deltas_low = env_low.simulate_hedging_episode(policy_low)
        _, _, deltas_high = env_high.simulate_hedging_episode(policy_high)
        
    # Calculate turnover (average sum of absolute delta changes)
    zeros = torch.zeros(deltas_low.shape[0], 1, deltas_low.shape[2], device=device)
    deltas_low_extended = torch.cat([zeros, deltas_low], dim=1)
    deltas_high_extended = torch.cat([zeros, deltas_high], dim=1)
    
    turnover_low = torch.mean(torch.sum(torch.abs(deltas_low_extended[:, 1:] - deltas_low_extended[:, :-1]), dim=1)).item()
    turnover_high = torch.mean(torch.sum(torch.abs(deltas_high_extended[:, 1:] - deltas_high_extended[:, :-1]), dim=1)).item()
    
    print(f"\nTurnover Low Cost: {turnover_low:.6f}")
    print(f"Turnover High Cost: {turnover_high:.6f}")
    
    assert turnover_high < turnover_low


