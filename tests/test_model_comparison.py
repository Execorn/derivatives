import pytest
import numpy as np
import torch
from deepvol.analysis.model_comparison import (
    ssvi_to_svi_surface,
    newey_west_variance,
    diebold_mariano_test,
    make_dupire_vol_fn,
    invert_prices_to_iv
)

def test_ssvi_to_svi_mapping():
    # Setup dummy SSVI parameters: 8 ATM variances and rho=-0.4, eta=1.2, gamma=0.3
    theta_atm = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18])
    rho, eta, gamma = -0.4, 1.2, 0.3
    ssvi_params = np.concatenate([theta_atm, [rho, eta, gamma]])
    
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    
    svi_params = ssvi_to_svi_surface(ssvi_params, T_grid)
    
    assert svi_params.shape == (8, 5)
    for i in range(8):
        a, b, r, m, s = svi_params[i]
        assert b >= 0
        assert s > 0
        assert abs(r - rho) < 1e-9


def test_newey_west_variance():
    # Setup a dummy series with autocorrelation
    np.random.seed(42)
    N = 100
    noise = np.random.randn(N)
    d = np.zeros(N)
    d[0] = noise[0]
    for t in range(1, N):
        d[t] = 0.5 * d[t-1] + noise[t]
        
    var_nw = newey_west_variance(d, lag=4)
    assert var_nw > 0.0
    
    # Variance with lag > 0 should be larger than sample variance for positive autocorrelation
    sample_var = np.var(d, ddof=0)
    assert var_nw > sample_var


def test_diebold_mariano_test():
    np.random.seed(42)
    N = 200
    
    # Model A: smaller errors
    errors_a = 0.01 * np.random.randn(N)
    # Model B: larger errors
    errors_b = 0.05 * np.random.randn(N)
    
    dm_stat, p_val = diebold_mariano_test(errors_a, errors_b, lag=3)
    
    # Since Model A is better (lower error), DM stat (loss_a - loss_b) should be negative
    assert dm_stat < 0.0
    assert 0.0 <= p_val <= 1.0


def test_vectorized_dupire_vol_fn():
    # Setup dummy flat local vol surface
    nT, nK = 8, 11
    local_vol_surface = np.full((nT, nK), 0.20)
    
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    S0 = 100.0
    r = 0.05
    q = 0.015
    device = "cpu"
    
    dup_vol_fn = make_dupire_vol_fn(local_vol_surface, T_grid, K_grid, S0, r, q, device)
    
    # Test with dummy stock price paths tensor
    S_t = torch.tensor([90.0, 100.0, 110.0], dtype=torch.float32)
    vol = dup_vol_fn(0.5, S_t)
    
    assert vol.shape == (3,)
    # Since surface is flat at 0.20, interpolated vol should be close to 0.20
    np.testing.assert_allclose(vol.numpy(), 0.20, rtol=1e-4)


def test_invert_prices_to_iv():
    T_grid = np.array([0.1, 0.3])
    K_grid = np.array([-0.1, 0.0, 0.1])
    S0 = 100.0
    r = 0.05
    q = 0.015
    
    # flat implied vol = 0.20
    market_iv = np.full((2, 3), 0.20)
    
    # Calculate Black-Scholes prices
    T_mesh = np.tile(T_grid[:, np.newaxis], (1, 3))
    strikes_mesh = S0 * np.exp((r - q) * T_mesh + np.tile(K_grid[np.newaxis, :], (2, 1)))
    
    import py_vollib_vectorized
    prices_flat = py_vollib_vectorized.vectorized_black_scholes_merton(
        "c", S0, strikes_mesh.ravel(), T_mesh.ravel(), r, 0.20, q, return_as="numpy"
    )
    prices = prices_flat.reshape(2, 3)
    
    # Invert back to IV
    iv_inverted = invert_prices_to_iv(prices, S0, T_grid, K_grid, r, q, market_iv)
    
    np.testing.assert_allclose(iv_inverted, 0.20, rtol=1e-3)
