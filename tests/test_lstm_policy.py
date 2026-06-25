import torch
import torch.nn as nn
import numpy as np
import pytest
from deepvol.hedging.deep_hedging import RecurrentHedgingPolicy, DeepHedgingEnv
from deepvol.models.signature_vol import compute_path_signature


def test_recurrent_policy_precision_and_device():
    """
    Verify that the RecurrentHedgingPolicy is instantiated in double precision (float64)
    and runs correctly on the active device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = RecurrentHedgingPolicy(input_dim=5, hidden_dim=16, output_dim=2).to(device)
    
    # Check parameter data types
    for param in policy.parameters():
        assert param.dtype == torch.float64, "All parameters must be in torch.float64"
        
    # Check forward pass on active device with float64 inputs
    x = torch.randn(10, 5, device=device, dtype=torch.float64)
    delta, h = policy(x)
    
    assert delta.dtype == torch.float64
    assert delta.shape == (10, 2)
    assert h[0][0].dtype == torch.float64
    assert h[1].dtype == torch.float64
    assert h[2].dtype == torch.float64
    assert h[3].dtype == torch.float64
    assert h[4].dtype == torch.float64


def test_signature_extraction_correctness():
    """
    Verify that the online signature extraction matches the offline signature computation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    seq_len = 10
    input_dim = 6  # e.g., 3 + d (d=3)
    
    policy = RecurrentHedgingPolicy(input_dim=input_dim, hidden_dim=16, output_dim=3).to(device)
    
    # Generate mock inputs
    # Index 0: log-spot, Index 2: vol_proxy
    x_sequence = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=torch.float64)
    
    # Extract the 2D path: (log_spot, vol_proxy)
    path = torch.cat([x_sequence[:, :, 0:1], x_sequence[:, :, 2:3]], dim=-1)
    
    # Compute offline signature
    offline_sig = compute_path_signature(path, depth=3)
    
    # Run the policy step-by-step (online signature computation)
    h = None
    for t in range(seq_len):
        delta_t, h = policy(x_sequence[:, t, :], h)
        
    # Extract online signature components from state h
    _, S1, S2, S3, _ = h
    online_sig = torch.cat([
        S1.reshape(batch_size, -1),
        S2.reshape(batch_size, -1),
        S3.reshape(batch_size, -1)
    ], dim=-1)
    
    # Assert correctness
    torch.testing.assert_close(online_sig, offline_sig, rtol=1e-6, atol=1e-6)


def test_recurrent_policy_hedging_episode():
    """
    Verify the recurrent policy in a full DeepHedgingEnv simulation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    N_paths = 100
    N_t = 20
    d = 2  # 2 hedging instruments
    
    # Simulate paths
    # Spot price & vol proxy
    from deepvol.hedging.deep_hedging import simulate_gbm_paths
    H, t_grid = simulate_gbm_paths(S0=100.0, mu=0.05, sigma=0.2, T=0.1, steps=N_t, N_paths=N_paths, d=d, device=device)
    H = H.to(torch.float64)
    t_grid = t_grid.to(torch.float64)
    
    # Payoff
    payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0).to(torch.float64)
    cost_coeffs = torch.tensor([0.001, 0.002], device=device, dtype=torch.float64)
    
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
    
    policy = RecurrentHedgingPolicy(input_dim=3 + d, hidden_dim=32, output_dim=d).to(device)
    
    wealth, total_costs, all_deltas = env.simulate_hedging_episode(policy)
    
    assert wealth.shape == (N_paths,)
    assert total_costs.shape == (N_paths,)
    assert all_deltas.shape == (N_paths, N_t, d)
    assert wealth.dtype == torch.float64
    assert not torch.isnan(wealth).any()
    
    loss = env.compute_loss(wealth)
    assert loss.dtype == torch.float64
    assert not torch.isnan(loss)
