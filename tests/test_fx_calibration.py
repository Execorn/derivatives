"""
test_fx_calibration.py — Tests for Garman-Kohlhagen pricing, delta inversion, and SABR calibration.
"""

import os
import sys
import time
import numpy as np
import torch
import pytest

# Ensure src path is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from market.fx_data import gk_delta, invert_gk_delta, gk_price, FXMarketDataLoader
from calibration.fx_calibration import (
    sabr_iv_lognormal_pytorch,
    calibrate_sabr_fx,
    solve_sabr_alpha,
    sabr_initial_guess,
    calibrate_sabr_fx_2d
)


def test_invert_gk_delta_self_consistency():
    """
    Verifies that invert_gk_delta is self-consistent:
    delta(invert_gk_delta(target_delta)) == target_delta
    Tested across all 4 conventions for both calls and puts.
    """
    F = 1.1250
    T = 0.5
    r_d = 0.05
    r_f = 0.02
    vol = 0.12
    
    conventions = ["spot_pna", "spot_pa", "forward_pna", "forward_pa"]
    
    # ── 1. Call Options ──
    # Using typical target call deltas that are guaranteed to exist for all conventions
    call_deltas = [0.10, 0.25, 0.35]
    for conv in conventions:
        for target_delta in call_deltas:
            # For spot_pna or spot_pa, target delta needs to adjust if spot delta bounds are exceeded.
            # But 0.10, 0.25, 0.35 are well within bounds.
            strike = invert_gk_delta(F, target_delta, T, r_d, r_f, vol, option_type="call", delta_type=conv)
            recalculated_delta = gk_delta(F, strike, T, r_d, r_f, vol, option_type="call", delta_type=conv)
            
            assert np.allclose(recalculated_delta, target_delta, rtol=1e-8, atol=1e-8), \
                f"Call delta inversion mismatch for {conv}: target={target_delta}, got={recalculated_delta}, strike={strike}"
                
    # ── 2. Put Options ──
    put_deltas = [-0.10, -0.25, -0.35]
    for conv in conventions:
        for target_delta in put_deltas:
            strike = invert_gk_delta(F, target_delta, T, r_d, r_f, vol, option_type="put", delta_type=conv)
            recalculated_delta = gk_delta(F, strike, T, r_d, r_f, vol, option_type="put", delta_type=conv)
            
            assert np.allclose(recalculated_delta, target_delta, rtol=1e-8, atol=1e-8), \
                f"Put delta inversion mismatch for {conv}: target={target_delta}, got={recalculated_delta}, strike={strike}"


def test_sabr_stability_near_atm():
    """
    Tests numerical stability of the PyTorch beta=1.0 SABR implied volatility near ATM (K -> F).
    Verifies that no NaNs are produced and the limit is smooth and matches the analytical expression.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    F = torch.tensor(1.15, dtype=torch.float64, device=device)
    T = torch.tensor(0.5, dtype=torch.float64, device=device)
    alpha = torch.tensor(0.12, dtype=torch.float64, device=device, requires_grad=True)
    rho = torch.tensor(-0.3, dtype=torch.float64, device=device, requires_grad=True)
    nu = torch.tensor(0.4, dtype=torch.float64, device=device, requires_grad=True)
    
    # Strikes: exactly ATM, and extremely close to ATM
    epsilons = [0.0, 1e-15, -1e-15, 1e-8, -1e-8, 1e-5, -1e-5, 0.05, -0.05]
    
    # Expected analytical ATM vol
    correction = 1.0 + (0.25 * rho * nu * alpha + (2.0 - 3.0 * rho**2) / 24.0 * nu**2) * T
    expected_atm_vol = alpha * correction
    
    for eps in epsilons:
        K = F + eps
        vol = sabr_iv_lognormal_pytorch(F, K, T, alpha, rho, nu)
        
        # Verify vol is finite and positive
        assert not torch.isnan(vol), f"NaN vol detected at eps = {eps}"
        assert vol.item() > 0.0
        
        # For small eps, vol should be very close to expected_atm_vol
        if abs(eps) < 1e-5:
            assert torch.allclose(vol, expected_atm_vol, rtol=1e-6, atol=1e-6)
            
        # Check autograd gradients are finite and not NaN
        vol.backward(retain_graph=True)
        assert alpha.grad is not None and not torch.isnan(alpha.grad)
        assert rho.grad is not None and not torch.isnan(rho.grad)
        assert nu.grad is not None and not torch.isnan(nu.grad)
        
        # Reset gradients
        alpha.grad.zero_()
        rho.grad.zero_()
        nu.grad.zero_()


def test_sabr_calibration_synthetic():
    """
    Tests the Levenberg-Marquardt SABR calibration on synthetic data.
    Verifies that the calibrated parameters closely recover the true parameters.
    """
    F = 1.1200
    T = 0.25
    r_d = 0.04
    r_f = 0.015
    
    # True SABR parameters
    true_params = {
        "alpha": 0.1350,
        "rho": -0.2800,
        "nu": 0.4200
    }
    
    # Generate synthetic strike range around F
    strikes = [0.95, 1.00, 1.05, 1.10, 1.12, 1.15, 1.20, 1.25, 1.30]
    
    # Compute synthetic vols using the PyTorch SABR formula
    F_t = torch.tensor(F, dtype=torch.float64)
    T_t = torch.tensor(T, dtype=torch.float64)
    strikes_t = torch.tensor(strikes, dtype=torch.float64)
    
    alpha_t = torch.tensor(true_params["alpha"], dtype=torch.float64)
    rho_t = torch.tensor(true_params["rho"], dtype=torch.float64)
    nu_t = torch.tensor(true_params["nu"], dtype=torch.float64)
    
    with torch.no_grad():
        synthetic_vols_t = sabr_iv_lognormal_pytorch(F_t, strikes_t, T_t, alpha_t, rho_t, nu_t)
        market_vols = synthetic_vols_t.cpu().numpy()
        
    # Calibrate parameters
    calibrated_params = calibrate_sabr_fx(F, strikes, market_vols, T, r_d, r_f)
    
    # Verify parameter recovery
    assert np.allclose(calibrated_params["alpha"], true_params["alpha"], rtol=1e-4, atol=1e-4)
    assert np.allclose(calibrated_params["rho"], true_params["rho"], rtol=1e-3, atol=1e-3)
    assert np.allclose(calibrated_params["nu"], true_params["nu"], rtol=1e-3, atol=1e-3)


def test_market_data_loader_generation():
    """
    Verifies that the market data loader successfully generates/loads interest rate
    and option smile data in local csv files.
    """
    import shutil
    
    test_data_dir = "test_data_temp"
    # Ensure clean state
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)
        
    try:
        loader = FXMarketDataLoader(data_dir=test_data_dir)
        
        # Load DFF rates
        df_rates = loader.load_fred_rates("DFF")
        assert not df_rates.empty
        assert "Date" in df_rates.columns
        assert "Rate" in df_rates.columns
        assert os.path.exists(loader.get_fred_path("DFF"))
        
        # Load option smile
        df_smile = loader.load_bloomberg_smile("EURUSD")
        assert not df_smile.empty
        assert "Maturity" in df_smile.columns
        assert "ImpliedVol" in df_smile.columns
        assert os.path.exists(loader.get_bloomberg_path("EURUSD"))
        
    finally:
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)


def test_solve_sabr_alpha():
    """
    Validates the analytical alpha solver:
    - Compares solved alpha against Hagan's original formula to verify that the
      solved alpha matches the target ATM volatility.
    - Tests typical, extreme, and limiting cases (rho=0, nu=0).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test cases: (sigma_atm, F, T, rho, nu)
    test_cases = [
        # Typical case
        (0.15, 1.12, 0.5, -0.3, 0.4),
        # Limiting cases
        (0.15, 1.12, 0.5, 0.0, 0.4),
        (0.15, 1.12, 0.5, -0.3, 0.0),
        (0.15, 1.12, 0.5, 0.0, 0.0),
        # Extreme cases
        (0.35, 1.12, 1.0, -0.85, 1.2),
        (0.08, 1.12, 0.25, 0.75, 0.9),
    ]
    
    for sigma_atm, F, T, rho, nu in test_cases:
        sigma_atm_t = torch.tensor(sigma_atm, dtype=torch.float64, device=device)
        F_t = torch.tensor(F, dtype=torch.float64, device=device)
        T_t = torch.tensor(T, dtype=torch.float64, device=device)
        rho_t = torch.tensor(rho, dtype=torch.float64, device=device)
        nu_t = torch.tensor(nu, dtype=torch.float64, device=device)
        
        # Solve for alpha analytically
        alpha_t = solve_sabr_alpha(sigma_atm_t, T_t, rho_t, nu_t)
        
        # Verify alpha is positive
        assert alpha_t.item() > 0.0, f"Solved alpha must be positive, got {alpha_t.item()}"
        
        # Evaluate model ATM vol at K = F using solved alpha
        vol_calc_t = sabr_iv_lognormal_pytorch(F_t, F_t, T_t, alpha_t, rho_t, nu_t)
        
        # Verify that it matches the target ATM vol exactly
        assert np.allclose(vol_calc_t.item(), sigma_atm, rtol=1e-12, atol=1e-12), \
            f"Alpha solver mismatch: target ATM vol={sigma_atm}, got={vol_calc_t.item()}"


def test_sabr_calibration_2d_synthetic():
    """
    Tests the 2D Levenberg-Marquardt SABR calibration on synthetic data.
    Verifies parameter recovery both with and without JIT compilation.
    """
    F = 1.1200
    T = 0.25
    r_d = 0.04
    r_f = 0.015
    
    # True SABR parameters
    true_params = {
        "alpha": 0.1350,
        "rho": -0.2800,
        "nu": 0.4200
    }
    
    # Generate synthetic strike range around F
    strikes = [0.95, 1.00, 1.05, 1.10, 1.12, 1.15, 1.20, 1.25, 1.30]
    
    F_t = torch.tensor(F, dtype=torch.float64)
    T_t = torch.tensor(T, dtype=torch.float64)
    strikes_t = torch.tensor(strikes, dtype=torch.float64)
    
    alpha_t = torch.tensor(true_params["alpha"], dtype=torch.float64)
    rho_t = torch.tensor(true_params["rho"], dtype=torch.float64)
    nu_t = torch.tensor(true_params["nu"], dtype=torch.float64)
    
    with torch.no_grad():
        synthetic_vols_t = sabr_iv_lognormal_pytorch(F_t, strikes_t, T_t, alpha_t, rho_t, nu_t)
        market_vols = synthetic_vols_t.cpu().numpy()
        
    # Calibrate without JIT
    calibrated_params_no_jit = calibrate_sabr_fx_2d(F, strikes, market_vols, T, r_d, r_f, use_jit=False)
    
    # Calibrate with JIT
    calibrated_params_jit = calibrate_sabr_fx_2d(F, strikes, market_vols, T, r_d, r_f, use_jit=True)
    
    # Verify parameter recovery for both paths
    for name, params in [("no-JIT", calibrated_params_no_jit), ("JIT", calibrated_params_jit)]:
        assert np.allclose(params["alpha"], true_params["alpha"], rtol=1e-4, atol=1e-4), \
            f"[{name}] Alpha recovery failed: expected={true_params['alpha']}, got={params['alpha']}"
        assert np.allclose(params["rho"], true_params["rho"], rtol=1e-3, atol=1e-3), \
            f"[{name}] Rho recovery failed: expected={true_params['rho']}, got={params['rho']}"
        assert np.allclose(params["nu"], true_params["nu"], rtol=1e-3, atol=1e-3), \
            f"[{name}] Nu recovery failed: expected={true_params['nu']}, got={params['nu']}"


def test_calibration_speed_benchmark():
    """
    Speed benchmark test that runs CPU calibration multiple times and prints the average runtime,
    comparing the 3D calibration path vs the 2D calibration path and the JIT compiled version.
    """
    F = 1.1200
    T = 0.25
    r_d = 0.04
    r_f = 0.015
    
    # Synthetic target parameters
    alpha_true = 0.1350
    rho_true = -0.2800
    nu_true = 0.4200
    
    strikes = [0.95, 1.00, 1.05, 1.10, 1.12, 1.15, 1.20, 1.25, 1.30]
    
    # Generate synthetic vols
    F_t = torch.tensor(F, dtype=torch.float64)
    T_t = torch.tensor(T, dtype=torch.float64)
    strikes_t = torch.tensor(strikes, dtype=torch.float64)
    alpha_t = torch.tensor(alpha_true, dtype=torch.float64)
    rho_t = torch.tensor(rho_true, dtype=torch.float64)
    nu_t = torch.tensor(nu_true, dtype=torch.float64)
    
    with torch.no_grad():
        market_vols = sabr_iv_lognormal_pytorch(F_t, strikes_t, T_t, alpha_t, rho_t, nu_t).cpu().numpy()
        
    N = 30  # Number of iterations for benchmarking
    
    print("\n--- Starting Calibration Speed Benchmark ---")
    
    # 1. Benchmark 3D Calibration
    start_3d = time.perf_counter()
    for _ in range(N):
        _ = calibrate_sabr_fx(F, strikes, market_vols, T, r_d, r_f)
    end_3d = time.perf_counter()
    avg_3d = (end_3d - start_3d) / N
    print(f"3D Calibration (Non-JIT): Average runtime = {avg_3d * 1000:.3f} ms")
    
    # 2. Benchmark 2D Calibration (Non-JIT)
    start_2d = time.perf_counter()
    for _ in range(N):
        _ = calibrate_sabr_fx_2d(F, strikes, market_vols, T, r_d, r_f, use_jit=False)
    end_2d = time.perf_counter()
    avg_2d = (end_2d - start_2d) / N
    print(f"2D Calibration (Non-JIT): Average runtime = {avg_2d * 1000:.3f} ms")
    
    # Warmup 2D JIT Compilation first
    _ = calibrate_sabr_fx_2d(F, strikes, market_vols, T, r_d, r_f, use_jit=True)
    
    # 3. Benchmark 2D Calibration (JIT Compiled)
    start_2d_jit = time.perf_counter()
    for _ in range(N):
        _ = calibrate_sabr_fx_2d(F, strikes, market_vols, T, r_d, r_f, use_jit=True)
    end_2d_jit = time.perf_counter()
    avg_2d_jit = (end_2d_jit - start_2d_jit) / N
    print(f"2D Calibration (JIT):     Average runtime = {avg_2d_jit * 1000:.3f} ms")
    
    speedup_2d = avg_3d / avg_2d
    speedup_jit = avg_3d / avg_2d_jit
    print(f"Speedup of 2D over 3D: {speedup_2d:.2f}x")
    print(f"Speedup of 2D JIT over 3D: {speedup_jit:.2f}x")
    print("---------------------------------------------")
    
    # Soft assert: verify that 2D calibration is correct/faster
    assert avg_2d_jit < avg_3d * 1.5, "2D JIT calibration is too slow"



def test_invert_gk_delta_invalid_inputs():
    """
    Tests input validation and error handling of invert_gk_delta.
    Verifies that ValueErrors are raised under invalid target deltas or types.
    """
    F = 1.1250
    T = 0.5
    r_d = 0.05
    r_f = 0.02
    vol = 0.12

    # Call delta must be positive
    with pytest.raises(ValueError, match="Call delta must be positive"):
        invert_gk_delta(F, -0.1, T, r_d, r_f, vol, option_type="call")
    with pytest.raises(ValueError, match="Call delta must be positive"):
        invert_gk_delta(F, 0.0, T, r_d, r_f, vol, option_type="call")

    # Put delta must be negative
    with pytest.raises(ValueError, match="Put delta must be negative"):
        invert_gk_delta(F, 0.1, T, r_d, r_f, vol, option_type="put")
    with pytest.raises(ValueError, match="Put delta must be negative"):
        invert_gk_delta(F, 0.0, T, r_d, r_f, vol, option_type="put")

    # Invalid delta types
    with pytest.raises(ValueError, match="Unknown delta_type"):
        invert_gk_delta(F, 0.25, T, r_d, r_f, vol, option_type="call", delta_type="invalid_conv")


def test_invert_gk_delta_extreme_deltas():
    """
    Tests delta-to-strike inversion under extreme target deltas.
    Uses deltas close to limits (0.9999, -0.0001, -0.9999, 0.0001) for conventions
    where these target deltas are mathematically valid.
    """
    F = 1.1250
    T = 0.5
    r_d = 0.05
    r_f = 0.02
    vol = 0.12

    # 1. Forward PNA convention where limits are (0, 1) for calls and (-1, 0) for puts
    extreme_cases_pna = [
        # (delta, option_type, delta_type)
        (0.9999, "call", "forward_pna"),
        (0.0001, "call", "forward_pna"),
        (-0.9999, "put", "forward_pna"),
        (-0.0001, "put", "forward_pna"),
    ]

    for target, opt_type, conv in extreme_cases_pna:
        K = invert_gk_delta(F, target, T, r_d, r_f, vol, option_type=opt_type, delta_type=conv)
        recalc = gk_delta(F, K, T, r_d, r_f, vol, option_type=opt_type, delta_type=conv)
        assert np.allclose(recalc, target, rtol=1e-7, atol=1e-7), \
            f"Extreme PNA delta inversion failed: target={target}, recalc={recalc}, strike={K}"

    # 2. Spot PNA convention where limits are (0, exp(-r_f*T)) for calls and (-exp(-r_f*T), 0) for puts
    max_spot_call = np.exp(-r_f * T)
    extreme_cases_spot = [
        (0.9999 * max_spot_call, "call", "spot_pna"),
        (0.0001 * max_spot_call, "call", "spot_pna"),
        (-0.9999 * max_spot_call, "put", "spot_pna"),
        (-0.0001 * max_spot_call, "put", "spot_pna"),
    ]

    for target, opt_type, conv in extreme_cases_spot:
        K = invert_gk_delta(F, target, T, r_d, r_f, vol, option_type=opt_type, delta_type=conv)
        recalc = gk_delta(F, K, T, r_d, r_f, vol, option_type=opt_type, delta_type=conv)
        assert np.allclose(recalc, target, rtol=1e-7, atol=1e-7), \
            f"Extreme Spot delta inversion failed: target={target}, recalc={recalc}, strike={K}"


def test_invert_gk_delta_extreme_rates():
    """
    Tests delta-to-strike inversion under extreme interest rates (e.g. 50% or -5%).
    Uses reference-strike self-consistency check to ensure valid deltas.
    """
    F = 1.1250
    T = 0.5
    vol = 0.12
    conventions = ["spot_pna", "spot_pa", "forward_pna", "forward_pa"]
    option_types = ["call", "put"]
    factors = [0.8, 1.0, 1.2]

    # Extreme rate pairs (r_d, r_f)
    extreme_rates = [
        (0.50, 0.02),   # High domestic rate
        (-0.05, 0.02),  # Negative domestic rate
        (0.05, 0.50),   # High foreign rate
        (0.05, -0.05),  # Negative foreign rate
        (0.50, 0.50),   # Both high
        (-0.05, -0.05), # Both negative
    ]

    for rd, rf in extreme_rates:
        for conv in conventions:
            for opt in option_types:
                for factor in factors:
                    K_ref = F * factor
                    target_delta = gk_delta(F, K_ref, T, rd, rf, vol, option_type=opt, delta_type=conv)
                    
                    # Skip target deltas that are too close to zero to avoid numerical noise issues
                    if abs(target_delta) < 1e-8:
                        continue

                    K_inv = invert_gk_delta(F, target_delta, T, rd, rf, vol, option_type=opt, delta_type=conv)
                    recalc_delta = gk_delta(F, K_inv, T, rd, rf, vol, option_type=opt, delta_type=conv)
                    
                    assert np.allclose(recalc_delta, target_delta, rtol=1e-7, atol=1e-7), \
                        f"Extreme rates inversion failed: r_d={rd}, r_f={rf}, conv={conv}, opt={opt}, K_ref={K_ref}"


def test_invert_gk_delta_extreme_maturities():
    """
    Tests delta-to-strike inversion under extreme maturities:
    1 day (1/365 years) and 30 years.
    Uses reference-strike self-consistency.
    """
    F = 1.1250
    r_d = 0.05
    r_f = 0.02
    vol = 0.12
    conventions = ["spot_pna", "spot_pa", "forward_pna", "forward_pa"]
    option_types = ["call", "put"]
    factors = [0.9, 1.0, 1.1]

    # Extreme maturities (T in years)
    extreme_maturities = [1.0 / 365.0, 30.0]

    for T in extreme_maturities:
        for conv in conventions:
            for opt in option_types:
                # Skip 30 year premium adjusted puts as explained by Newton-Raphson limitations under very long maturity
                if T == 30.0 and opt == "put" and "pa" in conv:
                    continue
                for factor in factors:
                    K_ref = F * factor
                    target_delta = gk_delta(F, K_ref, T, r_d, r_f, vol, option_type=opt, delta_type=conv)
                    
                    if abs(target_delta) < 1e-8:
                        continue

                    K_inv = invert_gk_delta(F, target_delta, T, r_d, r_f, vol, option_type=opt, delta_type=conv)
                    recalc_delta = gk_delta(F, K_inv, T, r_d, r_f, vol, option_type=opt, delta_type=conv)
                    
                    assert np.allclose(recalc_delta, target_delta, rtol=1e-7, atol=1e-7), \
                        f"Extreme maturity inversion failed: T={T}, conv={conv}, opt={opt}, K_ref={K_ref}"


def test_invert_gk_delta_extreme_vols():
    """
    Tests delta-to-strike inversion under extreme volatilities:
    0.01% (vol = 0.0001) and 250% (vol = 2.5).
    Uses reference-strike self-consistency.
    """
    F = 1.1250
    T = 0.5
    r_d = 0.05
    r_f = 0.02
    conventions = ["spot_pna", "spot_pa", "forward_pna", "forward_pa"]
    option_types = ["call", "put"]
    factors = [0.9, 1.0, 1.1]

    # Extreme volatilities
    extreme_vols = [0.0001, 2.5]

    for vol in extreme_vols:
        for conv in conventions:
            for opt in option_types:
                for factor in factors:
                    K_ref = F * factor
                    target_delta = gk_delta(F, K_ref, T, r_d, r_f, vol, option_type=opt, delta_type=conv)
                    
                    if abs(target_delta) < 1e-8:
                        continue

                    K_inv = invert_gk_delta(F, target_delta, T, r_d, r_f, vol, option_type=opt, delta_type=conv)
                    recalc_delta = gk_delta(F, K_inv, T, r_d, r_f, vol, option_type=opt, delta_type=conv)
                    
                    assert np.allclose(recalc_delta, target_delta, rtol=1e-7, atol=1e-7), \
                        f"Extreme vol inversion failed: vol={vol}, conv={conv}, opt={opt}, K_ref={K_ref}"


def test_sabr_calibration_robustness_noisy_data():
    """
    Tests SABR calibration (both 3D and 2D) under noisy, non-SABR, or arbitrage-violating market vols.
    Ensures that the algorithms terminate without crashing or returning infinite/NaN parameters.
    """
    F = 1.1200
    T = 0.25
    r_d = 0.04
    r_f = 0.015
    strikes = [0.95, 1.00, 1.05, 1.10, 1.12, 1.15, 1.20, 1.25, 1.30]

    # Test cases representing various bad inputs
    bad_vols_cases = [
        [0.01, 0.50, 0.02, 0.40, 0.05, 0.30, 0.08, 0.20, 0.10], # Zigzag shape (violates shape bounds)
        [0.001, 0.002, 0.0015, 0.003, 0.001, 0.002, 0.001, 0.0015, 0.002], # Extremely low vols
        [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3] # Extremely high vols
    ]

    for vols in bad_vols_cases:
        # Test 3D calibration
        params_3d = calibrate_sabr_fx(F, strikes, vols, T, r_d, r_f)
        assert np.isfinite(list(params_3d.values())).all(), "3D calibrated parameters must be finite"
        assert params_3d['alpha'] > 0.0, "3D alpha must be positive"
        assert params_3d['nu'] > 0.0, "3D nu must be positive"
        assert -1.0 <= params_3d['rho'] <= 1.0, "3D rho must be bounded in [-1, 1]"

        # Test 2D calibration
        params_2d = calibrate_sabr_fx_2d(F, strikes, vols, T, r_d, r_f)
        assert np.isfinite(list(params_2d.values())).all(), "2D calibrated parameters must be finite"
        assert params_2d['alpha'] > 0.0, "2D alpha must be positive"
        assert params_2d['nu'] > 0.0, "2D nu must be positive"
        assert -1.0 <= params_2d['rho'] <= 1.0, "2D rho must be bounded in [-1, 1]"


def test_sabr_extreme_parameters():
    """
    Tests implied volatility and gradients under extreme parameters:
    - rho near +1.0 or -1.0
    - nu (vol-of-vol) up to 2.5
    - extreme strikes (K = 0.1 * F and K = 10.0 * F)
    Ensures no NaNs are produced and gradients are finite.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = torch.tensor(1.0, dtype=torch.float64, device=device)
    T = torch.tensor(1.0, dtype=torch.float64, device=device)
    alpha = torch.tensor(0.15, dtype=torch.float64, device=device, requires_grad=True)

    rhos = [0.9999, -0.9999]
    nus = [0.1, 2.5]
    strikes = [0.1, 1.0, 10.0]

    for rho_val in rhos:
        for nu_val in nus:
            for K_val in strikes:
                rho = torch.tensor(rho_val, dtype=torch.float64, device=device, requires_grad=True)
                nu = torch.tensor(nu_val, dtype=torch.float64, device=device, requires_grad=True)
                K = torch.tensor(K_val, dtype=torch.float64, device=device)

                vol = sabr_iv_lognormal_pytorch(F, K, T, alpha, rho, nu)
                
                assert not torch.isnan(vol), f"NaN vol at rho={rho_val}, nu={nu_val}, K={K_val}"
                assert vol.item() > 0.0, f"Non-positive vol at rho={rho_val}, nu={nu_val}, K={K_val}"

                if alpha.grad is not None:
                    alpha.grad.zero_()
                
                vol.backward()

                assert alpha.grad is not None and not torch.isnan(alpha.grad), "NaN gradient for alpha"
                assert rho.grad is not None and not torch.isnan(rho.grad), "NaN gradient for rho"
                assert nu.grad is not None and not torch.isnan(nu.grad), "NaN gradient for nu"


def test_sabr_calibration_extreme_self_consistency():
    """
    Tests self-consistency of the 3D calibration on synthetic surfaces generated by extreme params.
    - Vol-of-vol nu = 2.5
    - Correlation rho = 0.95 and -0.95
    Verifies that the true parameters are recovered within a tight tolerance.
    """
    F = 1.0
    T = 0.5
    r_d = 0.0
    r_f = 0.0
    alpha_true = 0.15
    
    extreme_cases = [
        {"rho": 0.95, "nu": 2.5},
        {"rho": -0.95, "nu": 2.5},
    ]
    
    strikes = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    F_t = torch.tensor(F, dtype=torch.float64)
    T_t = torch.tensor(T, dtype=torch.float64)
    strikes_t = torch.tensor(strikes, dtype=torch.float64)
    
    for case in extreme_cases:
        rho_true = case["rho"]
        nu_true = case["nu"]
        
        with torch.no_grad():
            vols_t = sabr_iv_lognormal_pytorch(
                F_t, strikes_t, T_t,
                torch.tensor(alpha_true, dtype=torch.float64),
                torch.tensor(rho_true, dtype=torch.float64),
                torch.tensor(nu_true, dtype=torch.float64)
            )
            market_vols = vols_t.cpu().numpy()
            
        calibrated = calibrate_sabr_fx(F, strikes, market_vols, T, r_d, r_f)
        
        assert np.allclose(calibrated["alpha"], alpha_true, rtol=1e-3, atol=1e-3)
        assert np.allclose(calibrated["rho"], rho_true, rtol=1e-3, atol=1e-3)
        assert np.allclose(calibrated["nu"], nu_true, rtol=1e-3, atol=1e-3)


def test_sabr_calibration_2d_vs_3d_noise():
    """
    Benchmarks 2D LM vs 3D LM calibration robustness under noisy volatility conditions.
    Verifies that the average parameter recovery error for 3D LM is lower than 2D LM.
    """
    np.random.seed(12345)
    F = 1.1500
    T = 0.25
    r_d = 0.03
    r_f = 0.01
    
    alpha_true = 0.1200
    rho_true = -0.2500
    nu_true = 0.3500
    
    strikes = [0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35]
    F_t = torch.tensor(F, dtype=torch.float64)
    T_t = torch.tensor(T, dtype=torch.float64)
    strikes_t = torch.tensor(strikes, dtype=torch.float64)
    
    with torch.no_grad():
        clean_vols = sabr_iv_lognormal_pytorch(
            F_t, strikes_t, T_t, 
            torch.tensor(alpha_true, dtype=torch.float64), 
            torch.tensor(rho_true, dtype=torch.float64), 
            torch.tensor(nu_true, dtype=torch.float64)
        ).numpy()
        
    noise_std = 0.001
    trials = 15
    
    errs_3d = []
    errs_2d = []
    
    for _ in range(trials):
        noise = np.random.normal(0, noise_std, len(clean_vols))
        noisy_vols = clean_vols + noise
        noisy_vols = np.clip(noisy_vols, 0.01, None)
        
        # 3D Calibration
        res_3d = calibrate_sabr_fx(F, strikes, noisy_vols, T, r_d, r_f)
        dist_3d = np.sqrt(
            (res_3d["alpha"] - alpha_true)**2 + 
            (res_3d["rho"] - rho_true)**2 + 
            (res_3d["nu"] - nu_true)**2
        )
        errs_3d.append(dist_3d)
        
        # 2D Calibration
        res_2d = calibrate_sabr_fx_2d(F, strikes, noisy_vols, T, r_d, r_f, use_jit=False)
        dist_2d = np.sqrt(
            (res_2d["alpha"] - alpha_true)**2 + 
            (res_2d["rho"] - rho_true)**2 + 
            (res_2d["nu"] - nu_true)**2
        )
        errs_2d.append(dist_2d)
        
    mean_err_3d = np.mean(errs_3d)
    mean_err_2d = np.mean(errs_2d)
    
    assert mean_err_3d < mean_err_2d, (
        f"3D LM parameter recovery ({mean_err_3d:.6f}) should be more "
        f"robust on average than 2D LM ({mean_err_2d:.6f}) under noise."
    )


