import torch
import torch.nn as nn
import numpy as np
import pytest
from deepvol.hedging.frictional_env import (
    FrictionalHedgingEnv,
    _step_wealth_and_cost_compiled,
    _terminal_unwind_compiled
)
from deepvol.hedging.indifference_pricing import (
    IndifferencePricingEngine,
    invert_implied_volatility_hybrid,
    bs_call_price,
    bs_call_vega,
    train_frictional_hedger,
    evaluate_loss
)
from deepvol.hedging.deep_hedging import HedgingPolicy


def simulate_gbm_paths_double(S0, mu, sigma, T, steps, N_paths, d=1, device="cpu"):
    """
    Simulates stock price paths in double precision.
    """
    dt = T / steps
    t_grid = torch.arange(steps + 1, device=device, dtype=torch.float64) * dt
    W = torch.randn(N_paths, steps, device=device, dtype=torch.float64)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
    S = S0 * torch.exp(torch.cumsum(log_returns, dim=-1))
    S0_col = torch.full((N_paths, 1), S0, device=device, dtype=torch.float64)
    S_full = torch.cat([S0_col, S], dim=-1)
    if d == 1:
        H = S_full.unsqueeze(-1)
    else:
        vols = torch.full_like(S_full, sigma)
        H = torch.stack([S_full, vols], dim=-1)
    return H, t_grid


# Test Case 1: Environment Initialization with Scalars
def test_frictional_env_init_scalars():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.1, 10, 50, d=1, device=device)
    env = FrictionalHedgingEnv(H, gamma_0=0.001, gamma_1=0.005, alpha=0.5)
    assert env.gamma_0.shape == (1,)
    assert env.gamma_1.shape == (1,)
    assert env.alpha.shape == (1,)
    assert env.gamma_0.item() == 0.001
    assert env.gamma_1.item() == 0.005
    assert env.alpha.item() == 0.5


# Test Case 2: Environment Initialization with Tensors
def test_frictional_env_init_tensors():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.1, 10, 50, d=2, device=device)
    gamma_0 = torch.tensor([0.001, 0.002], device=device, dtype=torch.float64)
    gamma_1 = torch.tensor([0.003, 0.004], device=device, dtype=torch.float64)
    alpha = torch.tensor([0.5, 0.6], device=device, dtype=torch.float64)
    
    env = FrictionalHedgingEnv(H, gamma_0=gamma_0, gamma_1=gamma_1, alpha=alpha)
    assert torch.allclose(env.gamma_0, gamma_0)
    assert torch.allclose(env.gamma_1, gamma_1)
    assert torch.allclose(env.alpha, alpha)


# Test Case 3: Invalid Coefficient Type
def test_frictional_env_invalid_coeff_type():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.1, 10, 50, d=1, device=device)
    with pytest.raises(TypeError):
        FrictionalHedgingEnv(H, gamma_0="invalid")


# Test Case 4: Invalid Coefficient Shape
def test_frictional_env_invalid_coeff_shape():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.1, 10, 50, d=1, device=device)
    with pytest.raises(ValueError):
        FrictionalHedgingEnv(H, gamma_0=torch.tensor([0.01, 0.02], device=device))


# Test Case 5: State Representation Structure
def test_frictional_env_get_state():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.1, 10, 50, d=1, device=device)
    env = FrictionalHedgingEnv(H, strike=100.0, expiry=0.1, t_grid=t_grid)
    prev_delta = torch.zeros(50, 1, device=device, dtype=torch.float64)
    state = env.get_state(0, prev_delta)
    assert state.dtype == torch.float32
    assert state.shape == (50, 4)  # moneyness, expiry, vol, prev_delta


# Test Case 6: Simulation Episode Shape Correctness
def test_frictional_env_simulate_hedging_episode():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.1, 10, 50, d=1, device=device)
    env = FrictionalHedgingEnv(H, gamma_0=0.001, gamma_1=0.002, alpha=0.5, strike=100.0, expiry=0.1, t_grid=t_grid)
    policy = HedgingPolicy(input_dim=4, hidden_dim=8, output_dim=1).to(device)
    wealth, total_costs, all_deltas = env.simulate_hedging_episode(policy)
    
    assert wealth.dtype == torch.float64
    assert total_costs.dtype == torch.float64
    assert all_deltas.dtype == torch.float64
    assert wealth.shape == (50,)
    assert total_costs.shape == (50,)
    assert all_deltas.shape == (50, 10, 1)


# Test Case 7: Compiled Helper Wealth and Cost step calculations
def test_frictional_env_step_wealth_and_cost_compiled():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wealth = torch.zeros(10, dtype=torch.float64, device=device)
    total_costs = torch.zeros(10, dtype=torch.float64, device=device)
    H_k = torch.full((10, 1), 100.0, dtype=torch.float64, device=device)
    H_k_next = torch.full((10, 1), 101.0, dtype=torch.float64, device=device)
    delta = torch.full((10, 1), 1.0, dtype=torch.float64, device=device)
    prev_delta = torch.full((10, 1), 0.0, dtype=torch.float64, device=device)
    gamma_0 = torch.tensor([0.01], dtype=torch.float64, device=device)
    gamma_1 = torch.tensor([0.02], dtype=torch.float64, device=device)
    alpha = torch.tensor([0.5], dtype=torch.float64, device=device)
    
    # Cost = 100 * 1 * (0.01 + 0.02 * 1^0.5) = 100 * 0.03 = 3.0
    # Gain = 1.0 * (101 - 100) = 1.0
    # Wealth = 0 + 1.0 - 3.0 = -2.0
    new_w, new_c = _step_wealth_and_cost_compiled(wealth, total_costs, H_k, H_k_next, delta, prev_delta, gamma_0, gamma_1, alpha)
    assert torch.allclose(new_w, torch.full((10,), -2.0, dtype=torch.float64, device=device))
    assert torch.allclose(new_c, torch.full((10,), 3.0, dtype=torch.float64, device=device))


# Test Case 8: Compiled Helper Terminal Unwind calculations
def test_frictional_env_terminal_unwind_compiled():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wealth = torch.zeros(10, dtype=torch.float64, device=device)
    total_costs = torch.zeros(10, dtype=torch.float64, device=device)
    H_T = torch.full((10, 1), 100.0, dtype=torch.float64, device=device)
    prev_delta = torch.full((10, 1), 0.5, dtype=torch.float64, device=device)
    gamma_0 = torch.tensor([0.01], dtype=torch.float64, device=device)
    gamma_1 = torch.tensor([0.02], dtype=torch.float64, device=device)
    alpha = torch.tensor([0.5], dtype=torch.float64, device=device)
    
    # Cost = 100 * 0.5 * (0.01 + 0.02 * 0.5^0.5) = 50 * (0.01 + 0.02 * 0.7071) = 50 * 0.024142 = 1.20710678
    new_w, new_c = _terminal_unwind_compiled(wealth, total_costs, H_T, prev_delta, gamma_0, gamma_1, alpha)
    expected_cost = 50.0 * (0.01 + 0.02 * np.sqrt(0.5))
    assert torch.allclose(new_w, torch.full((10,), -expected_cost, dtype=torch.float64, device=device))
    assert torch.allclose(new_c, torch.full((10,), expected_cost, dtype=torch.float64, device=device))


# Test Case 9: Compute Loss for Entropic Risk
def test_frictional_env_compute_loss_entropic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, _ = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.1, 10, 50, d=1, device=device)
    payoff = torch.zeros(50, device=device, dtype=torch.float64)
    env = FrictionalHedgingEnv(H, payoff=payoff, risk_aversion=1.5, risk_measure="entropic")
    wealth = torch.ones(50, device=device, dtype=torch.float64) * 2.0
    
    loss = env.compute_loss(wealth)
    # loss = E[exp(-1.5 * (2.0 - 0.0))] = exp(-3.0)
    assert torch.allclose(loss, torch.tensor(np.exp(-3.0), device=device, dtype=torch.float64))


# Test Case 10: Compute Loss for Quadratic Risk
def test_frictional_env_compute_loss_quad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, _ = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.1, 10, 50, d=1, device=device)
    payoff = torch.ones(50, device=device, dtype=torch.float64) * 5.0
    env = FrictionalHedgingEnv(H, payoff=payoff, risk_measure="quad")
    wealth = torch.ones(50, device=device, dtype=torch.float64) * 2.0
    
    loss = env.compute_loss(wealth)
    # loss = E[(2.0 - 5.0)^2] = 9.0
    assert torch.allclose(loss, torch.tensor(9.0, device=device, dtype=torch.float64))


# Test Case 11: Black-Scholes Call Price
def test_bs_call_price():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = torch.tensor([100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0], device=device, dtype=torch.float64)
    sigma = torch.tensor([0.2], device=device, dtype=torch.float64)
    
    price = bs_call_price(S, K, T, sigma)
    # Analytic BS Call price for S=100, K=100, T=1, r=0, sigma=0.2 is approx 7.965567
    assert torch.allclose(price, torch.tensor(7.965567455, device=device, dtype=torch.float64), atol=1e-6)


# Test Case 12: Black-Scholes Vega
def test_bs_call_vega():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = torch.tensor([100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0], device=device, dtype=torch.float64)
    sigma = torch.tensor([0.2], device=device, dtype=torch.float64)
    
    vega = bs_call_vega(S, K, T, sigma)
    # d1 = (0 + 0.5 * 0.04) / 0.2 = 0.1
    # pdf(0.1) = exp(-0.005) / sqrt(2pi) = 0.995012 / 2.506628 = 0.396952
    # vega = 100 * 1 * 0.396952 = 39.6952
    assert torch.allclose(vega, torch.tensor(39.6952, device=device, dtype=torch.float64), atol=1e-3)


# Test Case 13: Implied Volatility Solver Exact Pricing
def test_implied_vol_solver_exact():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = torch.tensor([100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0], device=device, dtype=torch.float64)
    price = torch.tensor([7.965567455], device=device, dtype=torch.float64)
    
    sigma = invert_implied_volatility_hybrid(price, S, K, T)
    assert torch.allclose(sigma, torch.tensor(0.2, device=device, dtype=torch.float64), atol=1e-5)


# Test Case 14: Implied Volatility Solver Clamping
def test_implied_vol_solver_clamping():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = torch.tensor([100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0], device=device, dtype=torch.float64)
    
    # Very small price close to intrinsic (0.0)
    price = torch.tensor([0.0001], device=device, dtype=torch.float64)
    
    sigma = invert_implied_volatility_hybrid(price, S, K, T)
    # Should clamp to 0.01 (minimum volatility parameter) to prevent singularities
    assert torch.allclose(sigma, torch.tensor([0.01], device=device, dtype=torch.float64))


# Test Case 15: Implied Volatility Solver Bounds
def test_implied_vol_solver_bounds():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = torch.tensor([100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0], device=device, dtype=torch.float64)
    
    price_high = torch.tensor([105.0], device=device, dtype=torch.float64) # Above spot
    price_low = torch.tensor([-1.0], device=device, dtype=torch.float64)   # Below intrinsic
    
    sigma_high = invert_implied_volatility_hybrid(price_high, S, K, T)
    sigma_low = invert_implied_volatility_hybrid(price_low, S, K, T)
    
    assert torch.isnan(sigma_high).all()
    assert torch.isnan(sigma_low).all()


# Test Case 16: Indifference Pricing Engine Init
def test_indifference_pricing_engine_init():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.05, 5, 20, d=1, device=device)
    payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0)
    engine = IndifferencePricingEngine(H, payoff, gamma_0=0.001, strike=100.0, expiry=0.05, t_grid=t_grid)
    
    assert engine.env_pure.H.dtype == torch.float64
    assert engine.env_short.payoff is payoff
    assert torch.allclose(engine.env_long.payoff, -payoff)


# Test Case 17: Pricing Engine Invalid Risk Measure
def test_indifference_pricing_engine_invalid_risk_measure():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.05, 5, 20, d=1, device=device)
    payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0)
    engine = IndifferencePricingEngine(H, payoff, risk_measure="quad", strike=100.0, expiry=0.05, t_grid=t_grid)
    
    with pytest.raises(NotImplementedError):
        engine.compute_prices()


# Test Case 18: Frictional Env Precompute Toggle
def test_frictional_env_precompute_toggle():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.1, 10, 50, d=1, device=device)
    env = FrictionalHedgingEnv(H, strike=100.0, expiry=0.1, t_grid=t_grid)
    env.precompute = False
    policy = HedgingPolicy(input_dim=4, hidden_dim=8, output_dim=1).to(device)
    wealth, total_costs, all_deltas = env.simulate_hedging_episode(policy)
    
    assert wealth.shape == (50,)
    assert total_costs.shape == (50,)


# Test Case 19: Double Precision Compliance
def test_double_precision_compliance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.2, 0.1, 10, 50, d=1, device=device)
    env = FrictionalHedgingEnv(H, strike=100.0, expiry=0.1, t_grid=t_grid)
    policy = HedgingPolicy(input_dim=4, hidden_dim=8, output_dim=1).to(device)
    wealth, total_costs, all_deltas = env.simulate_hedging_episode(policy)
    
    # Internal variables must be double precision
    assert env.H.dtype == torch.float64
    assert env.payoff.dtype == torch.float64
    assert env.t_grid.dtype == torch.float64
    assert env.gamma_0.dtype == torch.float64
    assert wealth.dtype == torch.float64
    assert total_costs.dtype == torch.float64
    assert all_deltas.dtype == torch.float64


# Test Case 20: Solver NaN propagation
def test_implied_vol_solver_nan_propagation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = torch.tensor([100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0], device=device, dtype=torch.float64)
    price = torch.tensor([float('nan')], device=device, dtype=torch.float64)
    
    sigma = invert_implied_volatility_hybrid(price, S, K, T)
    assert torch.isnan(sigma).all()


# Test Case 21: Pricing Engine Training and Spread Positivity
def test_pricing_engine_train_and_spread_positivity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.25, 0.05, 5, 200, d=1, device=device)
    payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0)
    
    engine = IndifferencePricingEngine(
        H=H, payoff=payoff, gamma_0=0.002, gamma_1=0.005, alpha=0.5,
        risk_aversion=1.0, risk_measure="entropic", strike=100.0, expiry=0.05, t_grid=t_grid
    )
    
    # Fast training to verify training loop runs without crashing
    histories = engine.train_policies(epochs=5, batch_size=64, lr=1e-2, device=device)
    assert "losses_pure" in histories
    assert len(histories["losses_pure"]) == 5
    
    bid, ask, spread = engine.compute_prices(batch_size=64)
    assert spread >= 0.0
    # Bid should be less than or equal to ask
    assert bid <= ask


# Test Case 22: Spread Monotonicity with Transaction Costs
def test_pricing_engine_spread_monotonicity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    # Generate same paths
    H, t_grid = simulate_gbm_paths_double(100.0, 0.0, 0.25, 0.05, 5, 250, d=1, device=device)
    payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0)
    
    # Low cost engine
    engine_low = IndifferencePricingEngine(
        H=H, payoff=payoff, gamma_0=0.0, gamma_1=0.0, alpha=1.0,
        risk_aversion=1.0, risk_measure="entropic", strike=100.0, expiry=0.05, t_grid=t_grid
    )
    
    # High cost engine
    engine_high = IndifferencePricingEngine(
        H=H, payoff=payoff, gamma_0=0.02, gamma_1=0.05, alpha=0.5,
        risk_aversion=1.0, risk_measure="entropic", strike=100.0, expiry=0.05, t_grid=t_grid
    )
    
    # Share policy weights initially
    engine_high.policy_pure.load_state_dict(engine_low.policy_pure.state_dict())
    engine_high.policy_short.load_state_dict(engine_low.policy_short.state_dict())
    engine_high.policy_long.load_state_dict(engine_low.policy_long.state_dict())
    
    # Train both
    engine_low.train_policies(epochs=10, batch_size=128, lr=1e-2, device=device)
    engine_high.train_policies(epochs=10, batch_size=128, lr=1e-2, device=device)
    
    bid_low, ask_low, spread_low = engine_low.compute_prices(batch_size=128)
    bid_high, ask_high, spread_high = engine_high.compute_prices(batch_size=128)
    
    print(f"\nLow cost: Bid={bid_low:.4f}, Ask={ask_low:.4f}, Spread={spread_low:.4f}")
    print(f"High cost: Bid={bid_high:.4f}, Ask={ask_high:.4f}, Spread={spread_high:.4f}")
    
    # In a frictional market, the bid-ask spread must increase with higher transaction costs
    assert spread_high > spread_low
