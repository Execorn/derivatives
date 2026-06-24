import os
import sys
import numpy as np
import pytest

# Ensure src path is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from deepvol.benchmarks.hedging_backtest import (
    bs_call_price,
    bs_delta,
    bs_vega,
    interpolate_bilinear_np,
    simulate_heston_paths,
    evaluate_bs_delta,
    evaluate_fno_delta,
    evaluate_bs_delta_vega,
    evaluate_fno_delta_vega
)

def test_bs_formulas_vectorized():
    """Verify that Black-Scholes formulas handle numpy arrays correctly."""
    S = np.array([90.0, 100.0, 110.0])
    K = 100.0
    T = np.array([0.5, 0.5, 0.5])
    sigma = np.array([0.2, 0.2, 0.2])
    
    prices = bs_call_price(S, K, T, sigma)
    deltas = bs_delta(S, K, T, sigma)
    vegas = bs_vega(S, K, T, sigma)
    
    assert len(prices) == 3
    assert len(deltas) == 3
    assert len(vegas) == 3
    
    # Check that in-the-money options have higher delta and price
    assert deltas[2] > deltas[0]
    assert prices[2] > prices[0]
    
    # Verify edge cases: T=0
    T_zero = np.array([0.0, 0.0, 0.0])
    prices_zero = bs_call_price(S, K, T_zero, sigma)
    deltas_zero = bs_delta(S, K, T_zero, sigma)
    vegas_zero = bs_vega(S, K, T_zero, sigma)
    
    assert np.allclose(prices_zero, np.maximum(S - K, 0.0))
    assert np.allclose(deltas_zero, np.where(S > K, 1.0, 0.0))
    assert np.allclose(vegas_zero, 0.0)

def test_interpolate_bilinear_np():
    """Verify that bilinear interpolation maps correctly on a grid."""
    T_grid = np.array([0.5, 1.5])
    K_grid = np.array([-0.5, 0.5])
    iv_surface = np.array([
        [0.2, 0.3], # T=0.5, K=[-0.5, 0.5]
        [0.25, 0.35] # T=1.5, K=[-0.5, 0.5]
    ])
    
    # Exact grid point
    val_exact = interpolate_bilinear_np(T_grid, K_grid, iv_surface, 0.5, -0.5)
    assert np.isclose(val_exact, 0.2)
    
    # Midpoint interpolation
    val_mid = interpolate_bilinear_np(T_grid, K_grid, iv_surface, 1.0, 0.0)
    # Expected value is average of all 4 grid values: (0.2 + 0.3 + 0.25 + 0.35) / 4 = 0.275
    assert np.isclose(val_mid, 0.275)

def test_simulate_heston_paths():
    """Verify that Heston path simulation generates correct shapes."""
    rng = np.random.default_rng(42)
    S0 = 100.0
    v0 = 0.07
    kappa = 1.0
    theta = 0.08
    sigma = 0.5
    rho = -0.6
    T = 1.0
    N_steps = 10
    N_paths = 5
    
    S, V = simulate_heston_paths(S0, v0, kappa, theta, sigma, rho, T, N_steps, N_paths, rng)
    
    assert S.shape == (5, 11)
    assert V.shape == (5, 11)
    assert np.allclose(S[:, 0], S0)
    assert np.allclose(V[:, 0], v0)
    # Volatilities and prices should be positive or non-negative
    assert np.all(S > 0)

def test_hedging_evaluators():
    """Verify that the evaluation functions run without errors."""
    M, N_steps = 3, 5
    dt = 1.0 / N_steps
    S = np.full((M, N_steps + 1), 100.0)
    S[:, 1] = 101.0
    S[:, 2] = 99.0
    
    sigma_init = np.array([0.2, 0.2, 0.2])
    sigma_cal = np.full((M, N_steps + 1), 0.2)
    V0_price = 10.0
    K_V = 100.0
    T_V = 1.0
    
    errors_bs, tc_bs = evaluate_bs_delta(S, K_V, T_V, dt, sigma_init, V0_price)
    errors_fno, tc_fno = evaluate_fno_delta(S, K_V, T_V, dt, sigma_cal, V0_price)
    
    assert len(errors_bs) == M
    assert len(tc_bs) == M
    assert len(errors_fno) == M
    assert len(tc_fno) == M

    # Test delta-vega evaluators with standard inputs
    A_BS = np.full((M, N_steps + 1), 5.0)
    K_A = 100.0
    T_A = 2.0
    sigma_A_init = np.array([0.2, 0.2, 0.2])
    sigma_A_cal = np.full((M, N_steps + 1), 0.2)
    
    errors_bs_dv, tc_bs_dv = evaluate_bs_delta_vega(
        S, A_BS, K_V, K_A, T_V, T_A, dt, sigma_init, sigma_A_init, V0_price
    )
    errors_fno_dv, tc_fno_dv = evaluate_fno_delta_vega(
        S, A_BS, K_V, K_A, T_V, T_A, dt, sigma_cal, sigma_A_cal, V0_price
    )
    
    assert len(errors_bs_dv) == M
    assert len(tc_bs_dv) == M
    assert len(errors_fno_dv) == M
    assert len(tc_fno_dv) == M

def test_vega_collapse_position_capping():
    """Verify that delta-vega evaluators handle vega collapse gracefully using position capping."""
    M, N_steps = 3, 5
    dt = 1.0 / N_steps
    S = np.full((M, N_steps + 1), 100.0)
    A_BS = np.full((M, N_steps + 1), 5.0)
    
    K_V = 100.0
    K_A = 100.0
    T_V = 1.0
    T_A = 2.0
    V0_price = 10.0
    
    # We set sigma_A_init to a tiny value to induce vega collapse (v_A -> 0)
    sigma_V_init = np.array([0.2, 0.2, 0.2])
    sigma_A_init = np.array([1e-10, 1e-10, 1e-10]) # tiny vol -> vega collapse
    
    # Call BS evaluator
    errors_bs, tc_bs = evaluate_bs_delta_vega(
        S, A_BS, K_V, K_A, T_V, T_A, dt, sigma_V_init, sigma_A_init, V0_price
    )
    
    # Without position capping, delta_A = v_V / (v_A + 1e-8) would be extremely large,
    # leading to huge transaction costs (e.g. > 1e6).
    # With clipping to [-2.0, 2.0], delta_A and delta_S are capped.
    # The transaction cost tc is bounded by:
    # 0.0001 * S_t * |delta_S| + 0.0005 * A_t * |delta_A|
    # <= 0.0001 * 100 * 2.0 * 6 + 0.0005 * 5 * 2.0 * 6 = 0.12 + 0.03 = 0.15.
    # So tc_bs should be strictly bounded (e.g., less than 10.0).
    assert np.all(tc_bs < 10.0), f"Transaction cost exploded: {tc_bs}"
    assert np.all(np.isfinite(errors_bs)), "Errors contain non-finite values"

    # Also test FNO evaluator
    sigma_V_cal = np.full((M, N_steps + 1), 0.2)
    sigma_A_cal = np.full((M, N_steps + 1), 1e-10) # tiny vol -> vega collapse
    
    errors_fno, tc_fno = evaluate_fno_delta_vega(
        S, A_BS, K_V, K_A, T_V, T_A, dt, sigma_V_cal, sigma_A_cal, V0_price
    )
    assert np.all(tc_fno < 10.0), f"FNO Transaction cost exploded: {tc_fno}"
    assert np.all(np.isfinite(errors_fno)), "FNO Errors contain non-finite values"
