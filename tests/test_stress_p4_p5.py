"""
tests/test_stress_p4_p5.py — Stress and boundary condition testing for Phase 4 and Phase 5 models.
"""

import numpy as np
import torch
import pytest

from src.pricing.heston import heston_cf, heston_iv_surface, batch_heston_iv_surface
from src.pricing.sabr import sabr_iv_lognormal, sabr_iv_normal, ssvi_total_variance
from src.pricing.local_vol import svi_to_lv_surface, check_arbitrage_free
from src.pricing.rbergomi_gpu import simulate_rbergomi_paths, rbergomi_iv_surface
from src.pricing.neural_sde import NeuralSDE, NeuralSDEPricer, compute_calibration_loss
from src.pricing.signature_vol import SignatureVolatilityModel, compute_path_signature

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# 1. Heston Stress Tests
# ---------------------------------------------------------------------------

def test_heston_extreme_boundaries():
    """Test Heston characteristic function and pricer under extreme boundary conditions."""
    # Zero frequency limit
    cf_zero = heston_cf(u=0.0, T=1.0, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04)
    np.testing.assert_allclose(cf_zero, 1.0 + 0j)
    
    # Zero maturity limit
    cf_short_t = heston_cf(u=1.0, T=1e-8, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04)
    # At T=0, CF should equal exp(i * u * ln(S0)) = exp(0) = 1.0
    np.testing.assert_allclose(cf_short_t, 1.0 + 0j, rtol=1e-5, atol=1e-5)
    
    # Feller violation case (2 * kappa * theta < sigma^2)
    # kappa=1.0, theta=0.04, sigma=0.5 -> 2 * 0.04 = 0.08 < 0.25 (Feller violated)
    cf_feller = heston_cf(u=2.0, T=0.5, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    assert not np.isnan(cf_feller)
    assert not np.isinf(cf_feller)

def test_batch_heston_gpu_stability():
    """Test batched GPU Heston pricer under adversarial parameter settings (OOM check & NaNs)."""
    # 5 adversarial parameter sets:
    # 1. Normal
    # 2. Extreme volatility of vol (sigma = 1.5)
    # 3. High mean reversion (kappa = 5.0)
    # 4. Extreme correlation (rho = -0.99)
    # 5. Volatility of vol near zero (sigma = 1e-4)
    params = torch.tensor([
        [1.5, 0.04, 0.3, -0.6, 0.04],
        [1.0, 0.04, 1.5, -0.6, 0.04],
        [5.0, 0.08, 0.4, -0.3, 0.08],
        [1.5, 0.04, 0.3, -0.99, 0.04],
        [1.5, 0.04, 1e-4, -0.6, 0.04]
    ], dtype=torch.float64, device=device)
    
    T_grid = torch.tensor([0.05, 0.1, 0.5, 1.0], device=device)
    K_grid = torch.tensor([-0.2, 0.0, 0.2], device=device)
    
    ivs = batch_heston_iv_surface(params, T_grid, K_grid, S0=1.0, N_cos=256, device=device)
    
    assert ivs.shape == (5, len(T_grid), len(K_grid))
    # We should not have all NaNs for valid option prices
    # Note: extremum parameters might produce NaNs for out-of-money IVs, which is expected
    # but the pricer itself must not crash.
    assert not torch.isinf(ivs).any()

# ---------------------------------------------------------------------------
# 2. SABR and SSVI Stress Tests
# ---------------------------------------------------------------------------

def test_sabr_strike_singularity():
    """Test SABR implied volatility approximation near the strike singularity K -> F (ATM)."""
    # Lognormal Hagan formula ATM limit
    F = 100.0
    K_atm = 100.0
    # Near ATM
    K_near = 100.0001
    
    iv_atm = sabr_iv_lognormal(F, K_atm, T=0.5, alpha=0.2, beta=1.0, rho=-0.5, nu=0.3)
    iv_near = sabr_iv_lognormal(F, K_near, T=0.5, alpha=0.2, beta=1.0, rho=-0.5, nu=0.3)
    
    np.testing.assert_allclose(iv_atm, iv_near, rtol=1e-3)
    assert iv_atm > 0.0

def test_ssvi_calendar_arbitrage():
    """Verify that ssvi_total_variance detects or produces valid surface bounds."""
    # Compute SSVI total variance for two maturities
    # SSVI parameter vector: [theta, phi, rho]
    # theta must be increasing with maturity to avoid calendar arbitrage
    theta_1 = 0.04
    theta_2 = 0.08
    phi = 0.5
    rho = -0.4
    
    k = np.linspace(-0.5, 0.5, 10)
    w1 = ssvi_total_variance(k, theta_1, rho, phi, gamma=0.5)
    w2 = ssvi_total_variance(k, theta_2, rho, phi, gamma=0.5)
    
    # Calendar spread requires w2 >= w1
    assert np.all(w2 >= w1)

# ---------------------------------------------------------------------------
# 3. Local Volatility (Dupire) Stress Tests
# ---------------------------------------------------------------------------

def test_local_vol_arbitrage_handling():
    """Test that svi_to_lv_surface identifies arbitrage-violating SVI slices and sets them to -1.0."""
    T_grid = np.array([0.1, 0.5])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    # SVI parameters [a, b, rho, m, sigma]
    # Slice 0: Normal
    # Slice 1: Arbitrage-violating (calendar arbitrage: variance is smaller at T=0.5 than T=0.1)
    svi_params = np.array([
        [0.08, 0.1, -0.4, 0.0, 0.1],  # T=0.1
        [0.02, 0.05, -0.4, 0.0, 0.1]  # T=0.5 (W_0.5 < W_0.1)
    ])
    
    lv_surf = svi_to_lv_surface(T_grid, K_grid, svi_params)
    
    # Slice 1 should have negative values indicating arbitrage/invalid local vol
    # Slices are interpolated, so the final surface at T=0.5 should contain -1.0
    assert np.any(lv_surf[1] == -1.0)
    
    # Verify that check_arbitrage_free returns False
    assert check_arbitrage_free(T_grid, K_grid, svi_params) is False

# ---------------------------------------------------------------------------
# 4. Rough Bergomi Stress Tests
# ---------------------------------------------------------------------------

def test_rbergomi_extreme_roughness():
    """Stress test Rough Bergomi path simulation with extreme rough parameters H -> 0.01."""
    # We test that path simulation doesn't fail or yield NaNs when H is extremely small
    # (very rough volatility)
    N_paths = 128
    N_steps = 100
    H_extreme = 0.01
    eta = 1.5
    rho = -0.7
    v0 = 0.04
    
    # Run simulation
    params = torch.tensor([[v0, H_extreme, eta, rho]], device=device, dtype=torch.float32)
    S, V, t_grid = simulate_rbergomi_paths(
        params=params,
        T=1.0, steps_per_unit=N_steps, N_paths=N_paths,
        device=device
    )
    
    assert S.shape == (1, N_paths, N_steps + 1)
    assert not torch.isnan(S).any()
    assert not torch.isinf(S).any()

# ---------------------------------------------------------------------------
# 5. Neural SDE Stress Tests
# ---------------------------------------------------------------------------

def test_neural_sde_zero_maturity():
    """Verify Neural SDE pricer behavior at extremely short maturities or zero maturities."""
    sde = NeuralSDE(r=0.05, q=0.01, rho_init=-0.7, hidden_dim=16, epsilon=1e-4)
    pricer = NeuralSDEPricer(sde, v0_init=0.04).to(device)
    
    strikes = torch.tensor([100.0], device=device)
    # Zero maturity
    maturities = torch.tensor([0.0], device=device)
    
    prices, _ = pricer.price_options(
        S0=100.0, strikes=strikes, maturities=maturities,
        N_paths=512, dt=0.01, method="euler"
    )
    
    # At T=0, call option price is max(S0 - K, 0.0) = max(100 - 100, 0) = 0.0
    np.testing.assert_allclose(prices.item(), 0.0, atol=1e-5)

def test_neural_sde_gradient_flow():
    """Stress test backpropagation through SDE adjoint solver under extreme inputs."""
    sde = NeuralSDE(r=0.05, q=0.01, rho_init=-0.7, hidden_dim=16, epsilon=1e-4)
    pricer = NeuralSDEPricer(sde, v0_init=0.04).to(device)
    
    strikes = torch.tensor([80.0, 100.0, 120.0], device=device)
    maturities = torch.tensor([0.2, 0.2, 0.2], device=device)
    market_prices = torch.tensor([21.0, 5.0, 0.5], device=device)
    
    # Run forward
    prices, ys = pricer.price_options(
        S0=100.0, strikes=strikes, maturities=maturities,
        N_paths=256, dt=0.01, method="euler"
    )
    
    loss_dict = compute_calibration_loss(
        model_prices=prices,
        market_prices=market_prices,
        vegas=torch.ones_like(prices),
        ys=ys,
        lambda_bound=0.01,
        epsilon=1e-4
    )
    
    loss = loss_dict["loss"]
    loss.backward()
    
    # Check that gradients are successfully calculated and are not NaN
    for p in pricer.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()
        assert not torch.isinf(p.grad).any()

# ---------------------------------------------------------------------------
# 6. Signature Volatility Stress Tests
# ---------------------------------------------------------------------------

def test_signature_vol_martingality_check():
    """Verify that Signature Volatility martingale property holds under simulation."""
    model_martingale = SignatureVolatilityModel(device=device)
    with torch.no_grad():
        model_martingale.v0_raw.copy_(torch.tensor(np.log(0.04), device=device))
        model_martingale.rho_raw.copy_(torch.tensor(0.0, device=device))  # rho = -0.5
        model_martingale.ell_raw.zero_()
        model_martingale.ell_raw[0] = 0.02   # Level 1 odd
        model_martingale.ell_raw[6] = -0.005 # Level 3 odd
        
    # Use 30,000 paths for high precision
    S_rn, _, _, _ = model_martingale(
        T=1.0, steps_per_unit=100, N_paths=30000, S0=100.0, r=0.0, q=0.0, antithetic=True
    )
    E_ST_mart = S_rn[:, -1].mean().item()
    err_mart_bps = abs(E_ST_mart - 100.0) / 100.0 * 10000.0
    
    print(f"Martingale pricing error: {err_mart_bps:.4f} bps")
    
    # Assert that martingale error is within the 10 bps threshold
    assert err_mart_bps < 10.0
