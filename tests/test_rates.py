"""
test_rates.py — Unit and stress tests for Interest Rate Swaptions LMM-SABR calibration engine.
"""

import os
import sys
import numpy as np
import pytest

# Inject src path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pricing.bachelier import (
    bachelier_price,
    black_price,
    shifted_black_price,
    bachelier_implied_vol,
    black_implied_vol
)
from market.rates_data import (
    load_sofr_swap_rates,
    get_synthetic_forward_rates,
    load_swaption_vol_cube
)
from pricing.sabr_rates import (
    displaced_sabr_vol,
    calibrate_sabr_node,
    SwaptionVolCube,
    bilinear_interpolate
)


def test_analytical_models():
    """
    Test analytical Bachelier and Black pricing models.
    Covers happy paths, call vs put pricing, boundary cases (T=0, sigma=0, very small vols),
    and put-call parity.
    """
    F = 0.03
    K = 0.03
    T = 1.0
    sigma_n = 0.0080  # 80 bps normal vol
    sigma_ln = 0.25   # 25% lognormal vol
    
    # 1. Happy path call vs put pricing (both must be strictly positive)
    c_n = bachelier_price(F, K, T, sigma_n, 'call')
    p_n = bachelier_price(F, K, T, sigma_n, 'put')
    assert c_n > 0.0
    assert p_n > 0.0
    
    c_ln = black_price(F, K, T, sigma_ln, 'call')
    p_ln = black_price(F, K, T, sigma_ln, 'put')
    assert c_ln > 0.0
    assert p_ln > 0.0

    # 2. Boundary cases: T = 0 (should return intrinsic value)
    assert np.allclose(bachelier_price(0.04, 0.03, 0.0, sigma_n, 'call'), 0.01)
    assert np.allclose(bachelier_price(0.04, 0.03, 0.0, sigma_n, 'put'), 0.0)
    assert np.allclose(black_price(0.04, 0.03, 0.0, sigma_ln, 'call'), 0.01)
    assert np.allclose(black_price(0.04, 0.03, 0.0, sigma_ln, 'put'), 0.0)
    
    # Boundary cases: sigma = 0 (should return intrinsic value)
    assert np.allclose(bachelier_price(0.04, 0.03, 1.0, 0.0, 'call'), 0.01)
    assert np.allclose(bachelier_price(0.04, 0.03, 1.0, 0.0, 'put'), 0.0)
    assert np.allclose(black_price(0.04, 0.03, 1.0, 0.0, 'call'), 0.01)
    assert np.allclose(black_price(0.04, 0.03, 1.0, 0.0, 'put'), 0.0)

    # Boundary cases: Very small vols (e.g. 1e-9)
    assert np.allclose(bachelier_price(0.04, 0.03, 1.0, 1e-9, 'call'), 0.01, atol=1e-8)
    assert np.allclose(black_price(0.04, 0.03, 1.0, 1e-9, 'call'), 0.01, atol=1e-8)
    
    # 3. Put-call parity: C - P = F - K (since discount factor r = 0 is assumed in formulas)
    strikes = [0.01, 0.02, 0.03, 0.04, 0.05]
    for k in strikes:
        # Bachelier
        c = bachelier_price(F, k, T, sigma_n, 'call')
        p = bachelier_price(F, k, T, sigma_n, 'put')
        assert np.allclose(c - p, F - k, atol=1e-15)
        
        # Black
        c_b = black_price(F, k, T, sigma_ln, 'call')
        p_b = black_price(F, k, T, sigma_ln, 'put')
        assert np.allclose(c_b - p_b, F - k, atol=1e-15)
        
        # Shifted Black
        c_sb = shifted_black_price(F, k, T, sigma_ln, 0.02, 'call')
        p_sb = shifted_black_price(F, k, T, sigma_ln, 0.02, 'put')
        assert np.allclose(c_sb - p_sb, F - k, atol=1e-15)


def test_shifted_black_equivalence():
    """
    Verify that shifted Black option pricing behaves mathematically equivalent
    to adjusting the forward and strike rates.
    """
    F = 0.03
    K = 0.04
    T = 2.0
    sigma = 0.35
    shift = 0.02
    
    p1_call = shifted_black_price(F, K, T, sigma, shift, 'call')
    p2_call = black_price(F + shift, K + shift, T, sigma, 'call')
    assert np.allclose(p1_call, p2_call, atol=1e-15)
    
    p1_put = shifted_black_price(F, K, T, sigma, shift, 'put')
    p2_put = black_price(F + shift, K + shift, T, sigma, 'put')
    assert np.allclose(p1_put, p2_put, atol=1e-15)


def test_implied_vol_solvers():
    """
    Verify that implied volatility solvers accurately recover input volatilities
    and return NaN/handle errors on invalid prices.
    """
    F = 0.03
    K = 0.035
    T = 1.5
    
    # 1. Recover normal vol (50-100 bps)
    target_vol_n = 0.0075  # 75 bps
    p_n = bachelier_price(F, K, T, target_vol_n, 'call')
    recovered_vol_n = bachelier_implied_vol(p_n, F, K, T, 'call')
    assert np.allclose(recovered_vol_n, target_vol_n, atol=1e-7)
    
    # 2. Recover lognormal vol (10-50%)
    target_vol_ln = 0.30   # 30%
    p_ln = black_price(F, K, T, target_vol_ln, 'call')
    recovered_vol_ln = black_implied_vol(p_ln, F, K, T, 'call')
    assert np.allclose(recovered_vol_ln, target_vol_ln, atol=1e-7)
    
    # 3. Handle errors and invalid/non-arbitrage-free prices
    # Price below intrinsic
    bad_p_low = -0.01
    assert np.isnan(bachelier_implied_vol(bad_p_low, F, K, T, 'call'))
    assert np.isnan(black_implied_vol(bad_p_low, F, K, T, 'call'))
    
    # ITM price below intrinsic
    K_itm = 0.02
    int_itm = F - K_itm  # 0.01
    bad_itm_p = int_itm - 0.002
    assert np.isnan(bachelier_implied_vol(bad_itm_p, F, K_itm, T, 'call'))
    assert np.isnan(black_implied_vol(bad_itm_p, F, K_itm, T, 'call'))
    
    # Black price above maximum theoretical option price (which is F for call)
    bad_p_high = F + 0.10
    assert np.isnan(black_implied_vol(bad_p_high, F, K, T, 'call'))
    
    # Verify vectorization/array handling
    prices = np.array([p_n, p_n * 1.1])
    recovered_vols = bachelier_implied_vol(prices, F, K, T, 'call')
    assert recovered_vols.shape == (2,)
    assert not np.any(np.isnan(recovered_vols))


def test_displaced_sabr_vol():
    """
    Test displaced SABR implied volatilities under normal and lognormal vol types.
    Covers ATM vs non-ATM paths, correctness of shifts, and edge conditions (negative rates).
    """
    F = 0.03
    T = 1.0
    alpha = 0.015
    beta = 0.5
    rho = -0.3
    nu = 0.4
    shift = 0.02
    
    # ATM vs non-ATM paths
    vol_n_atm = displaced_sabr_vol(F, F, T, alpha, beta, rho, nu, shift, 'normal')
    vol_n_natm = displaced_sabr_vol(F, F + 0.01, T, alpha, beta, rho, nu, shift, 'normal')
    assert vol_n_atm > 0.0
    assert vol_n_natm > 0.0
    
    vol_ln_atm = displaced_sabr_vol(F, F, T, alpha, beta, rho, nu, shift, 'lognormal')
    vol_ln_natm = displaced_sabr_vol(F, F + 0.01, T, alpha, beta, rho, nu, shift, 'lognormal')
    assert vol_ln_atm > 0.0
    assert vol_ln_natm > 0.0
    
    # Correctness of shifts: passing shift should be mathematically identical to
    # passing (F + shift) and (K + shift) with zero shift.
    vol_shifted_n = displaced_sabr_vol(F, F + 0.01, T, alpha, beta, rho, nu, shift, 'normal')
    vol_base_n = displaced_sabr_vol(F + shift, F + 0.01 + shift, T, alpha, beta, rho, nu, 0.0, 'normal')
    assert np.allclose(vol_shifted_n, vol_base_n, atol=1e-15)
    
    # Edge conditions: negative rates in normal/displaced models
    # Normal SABR with beta=0 allows negative rates/strikes without shift
    vol_neg_n = displaced_sabr_vol(-0.01, -0.005, T, alpha, 0.0, rho, nu, 0.0, 'normal')
    assert vol_neg_n > 0.0
    assert not np.isnan(vol_neg_n)
    
    # With a positive shift, we can handle negative rates even with beta > 0
    vol_neg_shifted = displaced_sabr_vol(-0.01, -0.005, T, alpha, beta, rho, nu, 0.02, 'normal')
    assert vol_neg_shifted > 0.0
    assert not np.isnan(vol_neg_shifted)


def test_sabr_node_calibration():
    """
    Test single node SABR calibration.
    Verifies convergence and that parameter bounds (alpha > 0, rho in [-0.999, 0.999], nu >= 0)
    are strictly enforced.
    """
    F = 0.03
    T = 1.0
    beta = 0.5
    shift = 0.01
    vol_type = 'normal'
    
    # Synthetic market vols from known SABR parameters
    true_alpha = 0.0080
    true_rho = -0.25
    true_nu = 0.40
    
    strikes = F + np.array([-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0]) * 1e-4
    market_vols = displaced_sabr_vol(F, strikes, T, true_alpha, beta, true_rho, true_nu, shift, vol_type)
    
    # Run calibration
    cal_alpha, cal_rho, cal_nu = calibrate_sabr_node(F, strikes, market_vols, T, beta, shift, vol_type)
    
    # Check convergence: fitted vols should closely match market vols
    fitted_vols = displaced_sabr_vol(F, strikes, T, cal_alpha, beta, cal_rho, cal_nu, shift, vol_type)
    assert np.allclose(fitted_vols, market_vols, atol=1e-5)
    
    # Check parameter bounds
    assert cal_alpha > 0.0
    assert -0.999 <= cal_rho <= 0.999
    assert cal_nu >= 0.0


def test_swaption_vol_cube():
    """
    Test SwaptionVolCube class calibration, parameter bilinear interpolation,
    smile retrieval, and swaption pricing.
    """
    # 1. Load synthetic grid data
    expiries, tenors, relative_strikes, market_vols = load_swaption_vol_cube()
    forward_rates = get_synthetic_forward_rates(expiries, tenors)
    
    # 2. Initialize and calibrate cube
    cube = SwaptionVolCube(expiries, tenors, relative_strikes)
    cube.calibrate(market_vols, forward_rates, beta=0.5, shift=0.01, vol_type='normal')
    
    # Verify calibration results
    assert cube.alpha.shape == (len(expiries), len(tenors))
    assert cube.rho.shape == (len(expiries), len(tenors))
    assert cube.nu.shape == (len(expiries), len(tenors))
    
    # 3. Interpolation at non-grid coordinates
    T_exp = 1.5
    T_tenor = 3.5
    alpha, beta, rho, nu, shift = cube.interpolate_params(T_exp, T_tenor)
    
    assert alpha > 0.0
    assert -0.999 <= rho <= 0.999
    assert nu >= 0.0
    assert np.isclose(beta, 0.5)
    assert np.isclose(shift, 0.01)
    
    # 4. Smile retrieval
    F_interp = bilinear_interpolate(T_exp, T_tenor, expiries, tenors, forward_rates)
    strikes = F_interp + np.array([-100.0, 0.0, 100.0]) * 1e-4
    smile_vols = cube.get_smile(T_exp, T_tenor, strikes, vol_type='normal')
    
    assert len(smile_vols) == 3
    assert not np.any(np.isnan(smile_vols))
    assert np.all(smile_vols > 0.0)
    
    # 5. Option pricing and parity
    price_c = cube.price_swaption(T_exp, T_tenor, strikes[0], F_interp, 'call', 'normal')
    price_p = cube.price_swaption(T_exp, T_tenor, strikes[0], F_interp, 'put', 'normal')
    assert price_c > 0.0
    assert price_p > 0.0
    assert np.allclose(price_c - price_p, F_interp - strikes[0], atol=1e-12)


def test_extrapolation_behavior():
    """
    Verify flat extrapolation behavior outside coordinate bounds.
    """
    expiries, tenors, relative_strikes, market_vols = load_swaption_vol_cube()
    forward_rates = get_synthetic_forward_rates(expiries, tenors)
    
    cube = SwaptionVolCube(expiries, tenors, relative_strikes)
    cube.calibrate(market_vols, forward_rates, beta=0.5, shift=0.01, vol_type='normal')
    
    # Expiries are [1.0, 2.0, 5.0, 10.0]
    # Tenors are [1.0, 2.0, 5.0, 10.0, 30.0]
    
    # Under boundary check (T_exp = 0.5, T_tenor = 0.5 should map to T_exp = 1.0, T_tenor = 1.0)
    alpha_under, _, rho_under, nu_under, _ = cube.interpolate_params(0.5, 0.5)
    alpha_ref_under, _, rho_ref_under, nu_ref_under, _ = cube.interpolate_params(1.0, 1.0)
    assert np.allclose(alpha_under, alpha_ref_under)
    assert np.allclose(rho_under, rho_ref_under)
    assert np.allclose(nu_under, nu_ref_under)
    
    # Over boundary check (T_exp = 15.0, T_tenor = 40.0 should map to T_exp = 10.0, T_tenor = 30.0)
    alpha_over, _, rho_over, nu_over, _ = cube.interpolate_params(15.0, 40.0)
    alpha_ref_over, _, rho_ref_over, nu_ref_over, _ = cube.interpolate_params(10.0, 30.0)
    assert np.allclose(alpha_over, alpha_ref_over)
    assert np.allclose(rho_over, rho_ref_over)
    assert np.allclose(nu_over, nu_ref_over)
    
    # Mixed bounds check (T_exp = 0.5, T_tenor = 40.0 should map to T_exp = 1.0, T_tenor = 30.0)
    alpha_mixed, _, rho_mixed, nu_mixed, _ = cube.interpolate_params(0.5, 40.0)
    alpha_ref_mixed, _, rho_ref_mixed, nu_ref_mixed, _ = cube.interpolate_params(1.0, 30.0)
    assert np.allclose(alpha_mixed, alpha_ref_mixed)
    assert np.allclose(rho_mixed, rho_ref_mixed)
    assert np.allclose(nu_mixed, nu_ref_mixed)


def test_optimizations():
    """
    Verify the optimized engine components: ATM alpha solver, and parallel calibration equivalence.
    """
    from pricing.sabr_rates import solve_alpha_from_atm
    
    # 1. Test solve_alpha_from_atm
    F = 0.03
    T = 1.0
    beta = 0.5
    rho = -0.25
    nu = 0.40
    shift = 0.01
    vol_type = 'normal'
    
    # Normal ATM vol from alpha=0.008
    alpha_target = 0.008
    atm_vol = displaced_sabr_vol(F, F, T, alpha_target, beta, rho, nu, shift, vol_type)
    solved_alpha = solve_alpha_from_atm(F, T, beta, rho, nu, shift, atm_vol, vol_type)
    assert np.isclose(solved_alpha, alpha_target, rtol=1e-6)
    
    # Lognormal ATM vol from alpha=0.04
    vol_type_ln = 'lognormal'
    alpha_target_ln = 0.04
    atm_vol_ln = displaced_sabr_vol(F, F, T, alpha_target_ln, beta, rho, nu, shift, vol_type_ln)
    solved_alpha_ln = solve_alpha_from_atm(F, T, beta, rho, nu, shift, atm_vol_ln, vol_type_ln)
    assert np.isclose(solved_alpha_ln, alpha_target_ln, rtol=1e-6)

    # 2. Test parallel calibration equivalence
    expiries, tenors, relative_strikes, market_vols = load_swaption_vol_cube()
    forward_rates = get_synthetic_forward_rates(expiries, tenors)
    
    cube_seq = SwaptionVolCube(expiries, tenors, relative_strikes)
    cube_seq.calibrate(market_vols, forward_rates, beta=0.5, shift=0.01, vol_type='normal', parallel=False)
    
    cube_par = SwaptionVolCube(expiries, tenors, relative_strikes)
    cube_par.calibrate(market_vols, forward_rates, beta=0.5, shift=0.01, vol_type='normal', parallel=True)
    
    # Check that parallel calibration is identical to sequential
    assert np.allclose(cube_seq.alpha, cube_par.alpha, atol=1e-15)
    assert np.allclose(cube_seq.rho, cube_par.rho, atol=1e-15)
    assert np.allclose(cube_seq.nu, cube_par.nu, atol=1e-15)

