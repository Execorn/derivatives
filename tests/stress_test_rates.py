"""
stress_test_rates.py — Extensive adversarial verification and stress testing of
the Interest Rate Swaptions LMM-SABR calibration engine.
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import patch

# Inject src path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pricing.bachelier import (
    bachelier_price,
    black_price,
    shifted_black_price,
    bachelier_implied_vol,
    black_implied_vol
)
from pricing.sabr_rates import (
    displaced_sabr_vol,
    calibrate_sabr_node,
    SwaptionVolCube,
    bilinear_interpolate,
    solve_alpha_from_atm
)
from market.rates_data import (
    get_synthetic_forward_rates,
    load_swaption_vol_cube
)


def test_implied_vol_solvers_robustness():
    """
    Stress test implied volatility solvers over 100,000 combinations of options.
    Verifies that solvers never crash, never hang, and handle extreme cases
    (e.g. prices close to intrinsic value, extreme vols, very short maturities,
    negative interest rates, and invalid inputs).
    """
    np.random.seed(42)
    N_half = 50000
    
    # ==========================================
    # 1. Bachelier Solver Stress (50,000 cases)
    # ==========================================
    F_bac = np.random.uniform(-0.05, 0.15, N_half)
    K_bac = np.random.uniform(-0.05, 0.15, N_half)
    T_bac = np.random.exponential(1.0, N_half) + 1e-6
    # Extreme maturities
    T_bac[:5000] = 1e-6
    T_bac[5000:10000] = 1e-5
    
    vol_bac = np.random.exponential(0.3, N_half) + 1e-5
    # Extreme volatilities
    vol_bac[:5000] = 1e-5
    vol_bac[5000:10000] = 5.0
    
    option_types_bac = np.random.choice(['call', 'put'], N_half)
    
    call_mask = option_types_bac == 'call'
    put_mask = ~call_mask
    
    prices_bac = np.zeros(N_half)
    prices_bac[call_mask] = bachelier_price(F_bac[call_mask], K_bac[call_mask], T_bac[call_mask], vol_bac[call_mask], 'call')
    prices_bac[put_mask] = bachelier_price(F_bac[put_mask], K_bac[put_mask], T_bac[put_mask], vol_bac[put_mask], 'put')
    
    intrinsic_bac = np.zeros(N_half)
    intrinsic_bac[call_mask] = np.maximum(F_bac[call_mask] - K_bac[call_mask], 0.0)
    intrinsic_bac[put_mask] = np.maximum(K_bac[put_mask] - F_bac[put_mask], 0.0)
    
    # Introduce adversarial cases in a portion of data
    # 5,000 cases: price strictly below intrinsic (should return NaN)
    prices_bac[10000:15000] = intrinsic_bac[10000:15000] - np.random.uniform(1e-15, 0.05, 5000)
    # 5,000 cases: price within 1e-15 of intrinsic (should return 0.0 or NaN)
    prices_bac[15000:20000] = intrinsic_bac[15000:20000] + np.random.uniform(-1e-15, 1e-15, 5000)
    # 2,000 cases: negative prices (should return NaN)
    prices_bac[20000:22000] = -np.random.uniform(0.001, 0.1, 2000)
    # 2,000 cases: zero or negative maturities (should return NaN)
    T_bac[22000:24000] = -np.random.uniform(0.0, 1.0, 2000)
    # 800 cases: NaN/Inf inputs
    F_bac[24000:24200] = np.nan
    K_bac[24200:24400] = np.inf
    prices_bac[24400:24600] = np.nan
    T_bac[24600:24800] = np.nan
    
    # Run Bachelier Solver for calls and puts
    recovered_vol_bac_call = bachelier_implied_vol(prices_bac[call_mask], F_bac[call_mask], K_bac[call_mask], T_bac[call_mask], 'call')
    recovered_vol_bac_put = bachelier_implied_vol(prices_bac[put_mask], F_bac[put_mask], K_bac[put_mask], T_bac[put_mask], 'put')
    
    # Verify Bachelier call cases
    for i, idx in enumerate(np.where(call_mask)[0]):
        vol_true = vol_bac[idx]
        p = prices_bac[idx]
        f = F_bac[idx]
        k = K_bac[idx]
        t = T_bac[idx]
        v_rec = recovered_vol_bac_call[i]
        
        if idx >= 10000 and idx < 15000:
            assert np.isnan(v_rec) or v_rec == 0.0
        elif idx >= 20000 and idx < 22000:
            assert np.isnan(v_rec)
        elif idx >= 22000 and idx < 24000:
            assert np.isnan(v_rec)
        elif idx >= 24000 and idx < 24800:
            assert np.isnan(v_rec) or v_rec == 0.0
        elif np.isnan(v_rec):
            vega = np.sqrt(t) * np.exp(-0.5 * ((f - k)/(vol_true * np.sqrt(t)))**2) / np.sqrt(2 * np.pi)
            assert vega < 1e-4 or vol_true < 1e-4 or t < 1e-4 or p <= intrinsic_bac[idx] + 1e-12
        else:
            p_rec = bachelier_price(f, k, t, v_rec, 'call')
            assert np.allclose(p_rec, p, atol=1e-5)
            
    # Verify Bachelier put cases
    for i, idx in enumerate(np.where(put_mask)[0]):
        vol_true = vol_bac[idx]
        p = prices_bac[idx]
        f = F_bac[idx]
        k = K_bac[idx]
        t = T_bac[idx]
        v_rec = recovered_vol_bac_put[i]
        
        if idx >= 10000 and idx < 15000:
            assert np.isnan(v_rec) or v_rec == 0.0
        elif idx >= 20000 and idx < 22000:
            assert np.isnan(v_rec)
        elif idx >= 22000 and idx < 24000:
            assert np.isnan(v_rec)
        elif idx >= 24000 and idx < 24800:
            assert np.isnan(v_rec) or v_rec == 0.0
        elif np.isnan(v_rec):
            vega = np.sqrt(t) * np.exp(-0.5 * ((f - k)/(vol_true * np.sqrt(t)))**2) / np.sqrt(2 * np.pi)
            assert vega < 1e-4 or vol_true < 1e-4 or t < 1e-4 or p <= intrinsic_bac[idx] + 1e-12
        else:
            p_rec = bachelier_price(f, k, t, v_rec, 'put')
            assert np.allclose(p_rec, p, atol=1e-5)

    # ==========================================
    # 2. Shifted Black Solver Stress (50,000 cases)
    # ==========================================
    F_blk = np.random.uniform(-0.02, 0.12, N_half)
    shift_blk = np.random.uniform(0.03, 0.06, N_half)
    F_s = F_blk + shift_blk
    
    K_blk = np.random.uniform(-0.02, 0.12, N_half)
    K_s = K_blk + shift_blk
    
    T_blk = np.random.exponential(1.0, N_half) + 1e-6
    T_blk[:5000] = 1e-6
    T_blk[5000:10000] = 1e-5
    
    vol_blk = np.random.exponential(0.3, N_half) + 1e-5
    vol_blk[:5000] = 1e-5
    vol_blk[5000:10000] = 3.0
    
    option_types_blk = np.random.choice(['call', 'put'], N_half)
    call_mask_blk = option_types_blk == 'call'
    put_mask_blk = ~call_mask_blk
    
    prices_blk = np.zeros(N_half)
    prices_blk[call_mask_blk] = shifted_black_price(F_blk[call_mask_blk], K_blk[call_mask_blk], T_blk[call_mask_blk], vol_blk[call_mask_blk], shift_blk[call_mask_blk], 'call')
    prices_blk[put_mask_blk] = shifted_black_price(F_blk[put_mask_blk], K_blk[put_mask_blk], T_blk[put_mask_blk], vol_blk[put_mask_blk], shift_blk[put_mask_blk], 'put')
    
    intrinsic_blk = np.zeros(N_half)
    intrinsic_blk[call_mask_blk] = np.maximum(F_s[call_mask_blk] - K_s[call_mask_blk], 0.0)
    intrinsic_blk[put_mask_blk] = np.maximum(K_s[put_mask_blk] - F_s[put_mask_blk], 0.0)
    
    max_bound_blk = np.where(call_mask_blk, F_s, K_s)
    
    # Introduce adversarial cases
    # 5,000 cases: price strictly below intrinsic
    prices_blk[10000:15000] = intrinsic_blk[10000:15000] - np.random.uniform(1e-15, 0.05, 5000)
    # 3,000 cases: price exceeding maximum theoretical bound
    prices_blk[15000:18000] = max_bound_blk[15000:18000] + np.random.uniform(1e-15, 0.1, 3000)
    # 2,000 cases: negative prices
    prices_blk[18000:20000] = -np.random.uniform(0.001, 0.1, 2000)
    # 2,000 cases: zero or negative maturities
    T_blk[20000:22000] = -np.random.uniform(0.0, 1.0, 2000)
    # 2,000 cases: shifted rate/strike is non-positive
    shift_blk[22000:24000] = -0.2
    # 800 cases: NaN/Inf
    F_blk[24000:24200] = np.nan
    K_blk[24200:24400] = np.inf
    prices_blk[24400:24600] = np.nan
    T_blk[24600:24800] = np.nan
    
    # Run Shifted Black Solver for calls and puts
    recovered_vol_blk_call = black_implied_vol(prices_blk[call_mask_blk], F_blk[call_mask_blk], K_blk[call_mask_blk], T_blk[call_mask_blk], 'call', shift_blk[call_mask_blk])
    recovered_vol_blk_put = black_implied_vol(prices_blk[put_mask_blk], F_blk[put_mask_blk], K_blk[put_mask_blk], T_blk[put_mask_blk], 'put', shift_blk[put_mask_blk])
    
    # Verify Shifted Black call cases
    for i, idx in enumerate(np.where(call_mask_blk)[0]):
        vol_true = vol_blk[idx]
        p = prices_blk[idx]
        f = F_blk[idx]
        k = K_blk[idx]
        t = T_blk[idx]
        sh = shift_blk[idx]
        v_rec = recovered_vol_blk_call[i]
        
        if idx >= 10000 and idx < 15000:
            assert np.isnan(v_rec) or v_rec == 0.0
        elif idx >= 15000 and idx < 18000:
            assert np.isnan(v_rec)
        elif idx >= 18000 and idx < 20000:
            assert np.isnan(v_rec)
        elif idx >= 20000 and idx < 22000:
            assert np.isnan(v_rec)
        elif idx >= 22000 and idx < 24000:
            assert np.isnan(v_rec)
        elif idx >= 24000 and idx < 24800:
            assert np.isnan(v_rec) or v_rec == 0.0
        elif np.isnan(v_rec):
            f_s = f + sh
            k_s = k + sh
            if f_s > 0 and k_s > 0 and t > 0:
                d1 = (np.log(f_s/k_s) + 0.5 * vol_true**2 * t) / (vol_true * np.sqrt(t))
                vega = f_s * np.sqrt(t) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
                assert vega < 1e-4 or vol_true < 1e-4 or t < 1e-4 or p <= intrinsic_blk[idx] + 1e-12 or p >= max_bound_blk[idx] - 1e-12
        else:
            p_rec = shifted_black_price(f, k, t, v_rec, sh, 'call')
            assert np.allclose(p_rec, p, atol=1e-5)
            
    # Verify Shifted Black put cases
    for i, idx in enumerate(np.where(put_mask_blk)[0]):
        vol_true = vol_blk[idx]
        p = prices_blk[idx]
        f = F_blk[idx]
        k = K_blk[idx]
        t = T_blk[idx]
        sh = shift_blk[idx]
        v_rec = recovered_vol_blk_put[i]
        
        if idx >= 10000 and idx < 15000:
            assert np.isnan(v_rec) or v_rec == 0.0
        elif idx >= 15000 and idx < 18000:
            assert np.isnan(v_rec)
        elif idx >= 18000 and idx < 20000:
            assert np.isnan(v_rec)
        elif idx >= 20000 and idx < 22000:
            assert np.isnan(v_rec)
        elif idx >= 22000 and idx < 24000:
            assert np.isnan(v_rec)
        elif idx >= 24000 and idx < 24800:
            assert np.isnan(v_rec) or v_rec == 0.0
        elif np.isnan(v_rec):
            f_s = f + sh
            k_s = k + sh
            if f_s > 0 and k_s > 0 and t > 0:
                d1 = (np.log(f_s/k_s) + 0.5 * vol_true**2 * t) / (vol_true * np.sqrt(t))
                vega = f_s * np.sqrt(t) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
                assert vega < 1e-4 or vol_true < 1e-4 or t < 1e-4 or p <= intrinsic_blk[idx] + 1e-12 or p >= max_bound_blk[idx] - 1e-12
        else:
            p_rec = shifted_black_price(f, k, t, v_rec, sh, 'put')
            assert np.allclose(p_rec, p, atol=1e-5)


def test_calibration_robustness():
    """
    Stress test calibration over 100 random forward curves and volatility cubes.
    Asserts that:
    1. 100% of the calibration nodes converge successfully (do not return NaN/Inf).
    2. Calibrated parameters respect physical bounds.
    3. Fallbacks to 3D calibration work correctly if the cubic 2D reduction fails.
    """
    np.random.seed(123)
    expiries = np.array([1.0, 5.0, 10.0])
    tenors = np.array([2.0, 10.0])
    relative_strikes = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
    
    num_exp = len(expiries)
    num_ten = len(tenors)
    num_str = len(relative_strikes)
    
    for count in range(100):
        vol_type = 'normal' if count < 50 else 'lognormal'
        
        # Generate random forward curves
        if vol_type == 'normal':
            forward_rates = np.random.uniform(-0.01, 0.06, (num_exp, num_ten))
            beta = 0.0  # Safe for negative rates
            shift = 0.02
        else:
            forward_rates = np.random.uniform(0.01, 0.06, (num_exp, num_ten))
            beta = 0.5
            shift = 0.01
            
        market_vols = np.zeros((num_exp, num_ten, num_str))
        
        for i, T in enumerate(expiries):
            for j, tenor in enumerate(tenors):
                F = forward_rates[i, j]
                strikes = F + relative_strikes * 1e-4
                
                # Generate true parameters
                if vol_type == 'normal':
                    true_alpha = np.random.uniform(0.005, 0.015)
                    true_rho = np.random.uniform(-0.7, 0.7)
                    true_nu = np.random.uniform(0.2, 0.8)
                else:
                    true_alpha = np.random.uniform(0.1, 0.3)
                    true_rho = np.random.uniform(-0.7, 0.7)
                    true_nu = np.random.uniform(0.2, 0.8)
                    
                vols = displaced_sabr_vol(F, strikes, T, true_alpha, beta, true_rho, true_nu, shift, vol_type)
                noise = np.random.normal(0.0, 0.0001, num_str)
                vols_noisy = vols + noise
                market_vols[i, j, :] = np.maximum(vols_noisy, 1e-4)
                
        cube = SwaptionVolCube(expiries, tenors, relative_strikes)
        parallel_flag = (count % 2 == 0)
        cube.calibrate(market_vols, forward_rates, beta=beta, shift=shift, vol_type=vol_type, parallel=parallel_flag)
        
        # Assert convergence (no NaNs or Infs in calibrated parameters)
        assert not np.any(np.isnan(cube.alpha))
        assert not np.any(np.isnan(cube.rho))
        assert not np.any(np.isnan(cube.nu))
        
        assert not np.any(np.isinf(cube.alpha))
        assert not np.any(np.isinf(cube.rho))
        assert not np.any(np.isinf(cube.nu))
        
        # Assert parameter physical bounds
        assert np.all(cube.alpha > 0.0)
        assert np.all(cube.rho >= -0.999)
        assert np.all(cube.rho <= 0.999)
        assert np.all(cube.nu >= 0.0)
        
    # --- Test 3D Fallback ---
    F = 0.03
    T = 1.0
    beta = 0.5
    shift = 0.01
    vol_type = 'normal'
    strikes = F + np.array([-100.0, -50.0, 0.0, 50.0, 100.0]) * 1e-4
    
    # 1. Fallback via exception in solve_alpha_from_atm
    market_vols_normal = np.array([0.0085, 0.0082, 0.0080, 0.0082, 0.0085])
    with patch('pricing.sabr_rates.solve_alpha_from_atm', side_effect=RuntimeError("2D failed")):
        alpha, rho, nu = calibrate_sabr_node(F, strikes, market_vols_normal, T, beta, shift, vol_type)
        assert alpha > 0.0
        assert -0.999 <= rho <= 0.999
        assert nu >= 0.0
        
    # 2. Fallback via high MSE (poor 2D fit)
    # ATM vol is 0.0080, but other vols are extremely large, causing 2D fit to fail the MSE < 1e-4 threshold
    market_vols_bad = np.array([0.5, 0.5, 0.0080, 0.5, 0.5])
    alpha_bad, rho_bad, nu_bad = calibrate_sabr_node(F, strikes, market_vols_bad, T, beta, shift, vol_type)
    assert alpha_bad > 0.0
    assert -0.999 <= rho_bad <= 0.999
    assert nu_bad >= 0.0


def test_arbitrage_freeness():
    """
    Arbitrage-freeness check: for a large set of interpolated smiles (both normal and lognormal SABR),
    checks that they do not contain butterfly or calendar arbitrage.
    
    Uses standard SOFR swaption data for the normal cube, and flat Black-equivalent SABR data
    for the lognormal cube (to ensure a theoretically arbitrage-free reference baseline for the
    finite differences second derivative checks).
    """
    # 1. Load synthetic SOFR swap data
    expiries, tenors, relative_strikes, market_vols = load_swaption_vol_cube()
    forward_rates = get_synthetic_forward_rates(expiries, tenors)
    
    # 2. Calibrate normal cube (realistic parameters)
    cube_normal = SwaptionVolCube(expiries, tenors, relative_strikes)
    cube_normal.calibrate(market_vols, forward_rates, beta=0.0, shift=0.02, vol_type='normal')
    
    # 3. Create and calibrate lognormal cube using Black-equivalent parameters
    # (beta=1.0, rho=0.0, nu=1e-5 is mathematically guaranteed to be free of butterfly arbitrage)
    np.random.seed(42)
    market_vols_ln = np.zeros((len(expiries), len(tenors), len(relative_strikes)))
    for i, T in enumerate(expiries):
        for j, tenor in enumerate(tenors):
            F = forward_rates[i, j]
            strikes = F + relative_strikes * 1e-4
            # Flat lognormal vols
            vols = displaced_sabr_vol(F, strikes, T, 0.15 - 0.005*T, 1.0, 0.0, 1e-5, 0.01, 'lognormal')
            market_vols_ln[i, j, :] = vols
            
    cube_lognormal = SwaptionVolCube(expiries, tenors, relative_strikes)
    cube_lognormal.calibrate(market_vols_ln, forward_rates, beta=1.0, shift=0.01, vol_type='lognormal')
    
    # Define dense interpolation coordinates
    dense_expiries = np.linspace(1.5, 9.5, 10)
    dense_tenors = np.linspace(1.5, 25.0, 10)
    
    # Audit both normal and lognormal surfaces
    for vol_type, cube in [('normal', cube_normal), ('lognormal', cube_lognormal)]:
        shift_val = 0.02 if vol_type == 'normal' else 0.01
        
        # --- Strike/Butterfly Arbitrage Checks ---
        for T_exp in dense_expiries:
            for T_tenor in dense_tenors:
                F = bilinear_interpolate(T_exp, T_tenor, expiries, tenors, forward_rates)
                alpha, beta, rho, nu, shift = cube.interpolate_params(T_exp, T_tenor)
                
                # Estimate ATM vol to define strike bounds
                atm_vol = displaced_sabr_vol(F, F, T_exp, alpha, beta, rho, nu, shift, vol_type)
                
                # Setup strike grid (+/- 100 bps for normal, +/- 1.0 std dev for lognormal)
                if vol_type == 'normal':
                    K_min = F - 0.01
                    K_max = F + 0.01
                else:
                    std_dev = atm_vol * F * np.sqrt(T_exp)
                    K_min = F - 1.0 * std_dev
                    K_max = F + 1.0 * std_dev
                    K_min = max(K_min, -shift + 0.005) # buffer above shifted boundary
                    
                # 200 strike grid points for high-density CFD
                strikes = np.linspace(K_min, K_max, 200)
                dK = strikes[1] - strikes[0]
                
                # Generate SABR vols and call option prices
                vols = displaced_sabr_vol(F, strikes, T_exp, alpha, beta, rho, nu, shift, vol_type)
                
                if vol_type == 'normal':
                    call_prices = bachelier_price(F, strikes, T_exp, vols, 'call')
                else:
                    call_prices = shifted_black_price(F, strikes, T_exp, vols, shift, 'call')
                    
                # 1. Call prices must be strictly decreasing (dC/dK <= 0)
                # Allowing a tiny numerical tolerance for floating point limits
                dC = np.diff(call_prices)
                assert np.all(dC <= 1e-15), f"Strike arbitrage: Call prices not decreasing at T_exp={T_exp}, tenor={T_tenor}"
                
                # 2. Call prices must be convex in strike (d^2C/dK^2 >= 0)
                # This corresponds to implied PDF being non-negative
                pdf = (call_prices[2:] - 2.0 * call_prices[1:-1] + call_prices[:-2]) / (dK ** 2)
                assert np.all(pdf >= -1e-9), f"Butterfly arbitrage: negative implied PDF at T_exp={T_exp}, tenor={T_tenor}"

        # --- Calendar Arbitrage Checks ---
        for tenor in dense_tenors:
            for k in range(len(dense_expiries) - 1):
                T1 = dense_expiries[k]
                T2 = dense_expiries[k+1]
                
                F = bilinear_interpolate(T1, tenor, expiries, tenors, forward_rates)
                
                # Check option prices are increasing in maturity for strikes around ATM
                for rel_k in [-50.0, 0.0, 50.0]:
                    K = F + rel_k * 1e-4
                    
                    if vol_type == 'lognormal' and K + shift_val <= 0.0:
                        continue
                        
                    # Price at T1
                    alpha1, beta1, rho1, nu1, shift1 = cube.interpolate_params(T1, tenor)
                    vol1 = displaced_sabr_vol(F, K, T1, alpha1, beta1, rho1, nu1, shift1, vol_type)
                    
                    # Price at T2
                    alpha2, beta2, rho2, nu2, shift2 = cube.interpolate_params(T2, tenor)
                    vol2 = displaced_sabr_vol(F, K, T2, alpha2, beta2, rho2, nu2, shift2, vol_type)
                    
                    if vol_type == 'normal':
                        p1 = bachelier_price(F, K, T1, vol1, 'call')
                        p2 = bachelier_price(F, K, T2, vol2, 'call')
                    else:
                        p1 = shifted_black_price(F, K, T1, vol1, shift1, 'call')
                        p2 = shifted_black_price(F, K, T2, vol2, shift2, 'call')
                        
                    assert p2 - p1 >= -1e-9, (
                        f"Calendar arbitrage at tenor={tenor}, strike={K}: "
                        f"Price at T2={T2} ({p2}) is less than price at T1={T1} ({p1})"
                    )
