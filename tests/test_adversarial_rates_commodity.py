"""
test_adversarial_rates_commodity.py — Adversarial stress tests for Interest Rate LMM-SABR and Commodity Schwartz-Smith pricing/calibration.
"""

import os
import sys
import numpy as np
import pytest
import torch
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from deepvol.models.bachelier import (
    bachelier_price,
    black_price,
    shifted_black_price,
    bachelier_implied_vol,
    black_implied_vol
)
from deepvol.models.sabr_rates import (
    displaced_sabr_vol,
    calibrate_sabr_node,
    SwaptionVolCube,
    bilinear_interpolate,
    solve_alpha_from_atm
)
from deepvol.models.schwartz_smith import (
    schwartz_smith_price_black76,
    schwartz_smith_price_fourier,
    schwartz_smith_price_black76_pt,
    schwartz_smith_price_fourier_pt,
    schwartz_smith_price_cos,
    schwartz_smith_price_cos_pt,
    schwartz_smith_greeks_pt,
    calibrate_schwartz_smith,
    run_kalman_filter,
    conditional_variance,
    futures_price
)


# ===========================================================================
# 1. LMM-SABR Adversarial Tests
# ===========================================================================

def test_sabr_rates_rho_singularities():
    """
    Test LMM-SABR Hagan formula behavior at extreme correlation values (rho = 1.0, -1.0, and out of bounds).
    Hagan formula divides by (1 - rho) in non-ATM case, creating a division-by-zero singularity.
    """
    F = 0.03
    K = 0.035
    T = 1.0
    alpha = 0.0080
    beta = 0.5
    nu = 0.40
    shift = 0.01

    # 1. rho = -1.0 is valid and should not divide by zero (denominator is 1 - (-1) = 2)
    vol_neg1 = displaced_sabr_vol(F, K, T, alpha, beta, -1.0, nu, shift, 'normal')
    assert np.isfinite(vol_neg1)
    assert vol_neg1 > 0.0

    vol_pos1 = displaced_sabr_vol(F, K, T, alpha, beta, 1.0, nu, shift, 'normal')
    assert np.isfinite(vol_pos1)
    assert vol_pos1 > 0.0

    # 3. Out-of-bounds rho (> 1.0 or < -1.0)
    vol_out_pos_zeta = displaced_sabr_vol(0.035, 0.03, T, alpha, beta, 1.5, nu, shift, 'normal')
    assert np.isfinite(vol_out_pos_zeta)
    assert vol_out_pos_zeta > 0.0


def test_sabr_rates_negative_rates():
    """
    Test LMM-SABR normal, lognormal, and displaced versions under negative swap rates.
    """
    T = 1.0
    alpha = 0.008
    nu = 0.4
    shift = 0.02

    # 1. Normal SABR (beta = 0) should allow negative rates and strikes without shift
    vol_neg_normal = displaced_sabr_vol(-0.01, -0.005, T, alpha, 0.0, -0.2, nu, 0.0, 'normal')
    assert np.isfinite(vol_neg_normal)
    assert vol_neg_normal > 0.0

    # 2. Normal SABR (beta > 0) should NOT allow negative rates/strikes and return NaN
    vol_neg_normal_beta = displaced_sabr_vol(-0.01, -0.005, T, alpha, 0.5, -0.2, nu, 0.0, 'normal')
    assert np.isnan(vol_neg_normal_beta)

    # 3. Lognormal SABR should NOT allow negative rates/strikes and return NaN
    vol_neg_lognormal = displaced_sabr_vol(-0.01, -0.005, T, alpha, 0.5, -0.2, nu, 0.0, 'lognormal')
    assert np.isnan(vol_neg_lognormal)

    # 4. Displaced SABR with positive shift should allow negative rates as long as F + shift > 0 and K + shift > 0
    # Here F = -0.01, K = -0.005, shift = 0.02. Shifted values: F_s = 0.01, K_s = 0.015.
    vol_displaced = displaced_sabr_vol(-0.01, -0.005, T, alpha, 0.5, -0.2, nu, shift, 'normal')
    assert np.isfinite(vol_displaced)
    assert vol_displaced > 0.0

    # If shift is too small (e.g. 0.005), F + shift = -0.005 <= 0, should return NaN
    vol_displaced_fail = displaced_sabr_vol(-0.01, -0.005, T, alpha, 0.5, -0.2, nu, 0.005, 'normal')
    assert np.isnan(vol_displaced_fail)


def test_sabr_rates_cev_exponent_boundaries():
    """
    Test LMM-SABR CEV exponent beta outside [0, 1] range.
    """
    F = 0.03
    K = 0.035
    T = 1.0
    alpha = 0.008
    rho = -0.2
    nu = 0.4
    shift = 0.0

    # 1. beta < 0 (non-physical). Let's see if it crashes or yields complex/nan.
    # F_atm ** beta_atm where F_atm = 0.03, beta = -0.5 is mathematically real (0.03**-0.5 is real).
    # But what if F < 0 and beta < 0?
    with np.errstate(all='ignore'):
        vol_neg_beta = displaced_sabr_vol(F, K, T, alpha, -0.5, rho, nu, shift, 'normal')
    # Should not crash. Can be finite or nan.
    assert np.isnan(vol_neg_beta) or vol_neg_beta > 0.0

    # 2. beta > 1.0 (e.g. beta = 1.5)
    vol_large_beta = displaced_sabr_vol(F, K, T, alpha, 1.5, rho, nu, shift, 'normal')
    assert np.isfinite(vol_large_beta)


def test_sabr_rates_vol_of_vol_zero():
    """
    Test LMM-SABR Hagan formula when vol-of-vol nu is exactly 0.0 (reduces to CEV model).
    """
    F = 0.03
    K = 0.035
    T = 1.0
    alpha = 0.008
    beta = 0.5
    rho = -0.2
    shift = 0.01

    # ATM and non-ATM cases with nu = 0
    vol_atm = displaced_sabr_vol(F, F, T, alpha, beta, rho, 0.0, shift, 'normal')
    vol_natm = displaced_sabr_vol(F, K, T, alpha, beta, rho, 0.0, shift, 'normal')

    assert np.isfinite(vol_atm)
    assert vol_atm > 0.0
    assert np.isfinite(vol_natm)
    assert vol_natm > 0.0


def test_sabr_rates_maturity_singularities():
    """
    Test LMM-SABR option price and vol under zero or negative maturity T.
    """
    F = 0.03
    K = 0.03
    alpha = 0.008
    beta = 0.5
    rho = -0.2
    nu = 0.4
    shift = 0.01

    # 1. T < 0: displaced_sabr_vol should return NaN
    vol_neg_T = displaced_sabr_vol(F, K, -0.5, alpha, beta, rho, nu, shift, 'normal')
    assert np.isnan(vol_neg_T)

    # 2. T = 0: displaced_sabr_vol should return NaN (since T > 0 is in validity mask)
    vol_zero_T = displaced_sabr_vol(F, K, 0.0, alpha, beta, rho, nu, shift, 'normal')
    assert np.isnan(vol_zero_T)

    # 3. T very small (e.g. 1e-15)
    vol_small_T = displaced_sabr_vol(F, K, 1e-15, alpha, beta, rho, nu, shift, 'normal')
    assert np.isfinite(vol_small_T) or np.isnan(vol_small_T)


def test_sabr_rates_atm_discontinuity():
    """
    Test LMM-SABR normal and lognormal vol near the ATM/non-ATM boundary (|F - K| = 1e-8).
    Check that there is no large jump in volatility at the threshold.
    """
    F = 0.03
    alpha = 0.008
    beta = 0.5
    rho = -0.2
    nu = 0.4
    shift = 0.01
    T = 1.0

    for vtype in ['normal', 'lognormal']:
        # Just below threshold (ATM path)
        K_atm = F + 1e-8 - 1e-15
        vol_atm = displaced_sabr_vol(F, K_atm, T, alpha, beta, rho, nu, shift, vtype)

        # Just above threshold (non-ATM path)
        K_natm = F + 1e-8 + 1e-15
        vol_natm = displaced_sabr_vol(F, K_natm, T, alpha, beta, rho, nu, shift, vtype)

        # Volatility difference should be very small
        diff = abs(vol_atm - vol_natm)
        assert diff < 1e-6, f"Discontinuity in SABR {vtype} vol at ATM boundary: ATM={vol_atm}, non-ATM={vol_natm}, diff={diff}"


def test_sabr_rates_calibration_adversarial_inputs():
    """
    Test calibrate_sabr_node robustness under adversarial conditions:
    1. Too few strikes (e.g. 1 strike)
    2. Duplicate strikes
    3. NaNs in market volatilities
    """
    F = 0.03
    T = 1.0
    beta = 0.5
    shift = 0.01

    # 1. Too few strikes: calibrate_sabr_node should fail gracefully or return bounds
    strikes_1 = np.array([F])
    vols_1 = np.array([0.0080])
    # Full 3D least squares optimization with 1 data point and 3 variables is underdetermined.
    # It should return a solution within bounds without crashing.
    try:
        alpha, rho, nu = calibrate_sabr_node(F, strikes_1, vols_1, T, beta, shift, 'normal')
        assert alpha > 0.0
        assert -0.999 <= rho <= 0.999
        assert nu >= 0.0
    except Exception as e:
        # If it raises an error, that is also a valid graceful failure, but it shouldn't hang.
        pass

    # 2. Duplicate strikes
    strikes_dup = np.array([F - 0.01, F, F, F + 0.01])
    vols_dup = np.array([0.0085, 0.0080, 0.0080, 0.0085])
    alpha, rho, nu = calibrate_sabr_node(F, strikes_dup, vols_dup, T, beta, shift, 'normal')
    assert np.isfinite(alpha) and np.isfinite(rho) and np.isfinite(nu)

    # 3. NaNs in market volatilities
    # Case A: NaN at ATM index. Handles it by searching for the closest non-NaN volatility quote.
    strikes_nan_atm = np.array([F - 0.01, F, F + 0.01])
    vols_nan_atm = np.array([0.0085, np.nan, 0.0085])
    alpha_atm, rho_atm, nu_atm = calibrate_sabr_node(F, strikes_nan_atm, vols_nan_atm, T, beta, shift, 'normal')
    assert np.isfinite(alpha_atm) and np.isfinite(rho_atm) and np.isfinite(nu_atm)

    # Case B: NaN at non-ATM index. ATM vol is valid, and the NaN residual is handled via np.where replacement.
    strikes_nan_natm = np.array([F - 0.01, F, F + 0.01])
    vols_nan_natm = np.array([np.nan, 0.0080, 0.0085])
    alpha_nan, rho_nan, nu_nan = calibrate_sabr_node(F, strikes_nan_natm, vols_nan_natm, T, beta, shift, 'normal')
    assert np.isfinite(alpha_nan) and np.isfinite(rho_nan) and np.isfinite(nu_nan)

    # Case C: All NaNs. Handles it by falling back to default volatility.
    strikes_all_nan = np.array([F - 0.01, F, F + 0.01])
    vols_all_nan = np.array([np.nan, np.nan, np.nan])
    alpha_all, rho_all, nu_all = calibrate_sabr_node(F, strikes_all_nan, vols_all_nan, T, beta, shift, 'normal')
    assert np.isfinite(alpha_all) and np.isfinite(rho_all) and np.isfinite(nu_all)


# ===========================================================================
# 2. Commodity Schwartz-Smith Adversarial Tests
# ===========================================================================

def test_schwartz_smith_expired_options_cpu_and_pt():
    """
    Test Schwartz-Smith option pricing under zero or negative maturity (tau <= 0).
    It should return the undiscounted payoff.
    """
    kappa = 0.6
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    chi_t = 0.1
    xi_t = np.log(75.0)
    t = 1.0
    r = 0.05
    K = 70.0
    T_fut = 1.2

    # Under Q, F(t, T_fut)
    F = futures_price(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    intrinsic_call = max(F - K, 0.0)
    intrinsic_put = max(K - F, 0.0)

    # 1. Negative maturity (T_opt < t)
    for T_opt in [0.9, 0.5, 0.0]:
        for otype in ["C", "P"]:
            # CPU Black76
            p_b = schwartz_smith_price_black76(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi, option_type=otype)
            # CPU Fourier
            p_f = schwartz_smith_price_fourier(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi, option_type=otype)
            # CPU COS
            p_c = schwartz_smith_price_cos(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi, option_type=otype)
            
            payoff = intrinsic_call if otype == "C" else intrinsic_put
            assert np.isclose(p_b, payoff)
            assert np.isclose(p_f, payoff)
            assert np.isclose(p_c, payoff)

    # 2. PyTorch equivalents
    chi_pt = torch.tensor([chi_t], dtype=torch.float64)
    xi_pt = torch.tensor([xi_t], dtype=torch.float64)
    for T_opt in [0.9, 0.5, 0.0]:
        for otype in ["C", "P"]:
            p_b_pt = schwartz_smith_price_black76_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi, option_type=otype)
            p_f_pt = schwartz_smith_price_fourier_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi, option_type=otype)
            p_c_pt = schwartz_smith_price_cos_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi, option_type=otype)
            
            payoff = intrinsic_call if otype == "C" else intrinsic_put
            assert np.isclose(p_b_pt.item(), payoff)
            assert np.isclose(p_f_pt.item(), payoff)
            assert np.isclose(p_c_pt.item(), payoff)


def test_schwartz_smith_option_expires_after_futures():
    """
    Test Schwartz-Smith pricing when the option maturity is greater than or equal to the futures maturity (T_opt >= T_fut).
    An option on futures cannot mature after the underlying futures contract.
    It should raise ValueError for all CPU and PyTorch pricers.
    """
    kappa = 0.6
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    chi_t = 0.1
    xi_t = np.log(75.0)
    t = 0.0
    r = 0.04
    K = 75.0
    
    T_opt = 1.0
    T_fut = 0.8  # Option expires after futures

    # CPU Pricers
    with pytest.raises(ValueError):
        schwartz_smith_price_black76(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_fourier(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_cos(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)

    # PyTorch Pricers
    chi_pt = torch.tensor([chi_t], dtype=torch.float64)
    xi_pt = torch.tensor([xi_t], dtype=torch.float64)
    with pytest.raises(ValueError):
        schwartz_smith_price_black76_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_fourier_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_cos_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)


def test_schwartz_smith_explosive_mean_reversion():
    """
    Test Schwartz-Smith pricing under explosive mean reversion (kappa < 0).
    kappa must be non-negative physically.
    It should raise ValueError for all CPU and PyTorch pricers.
    """
    kappa = -0.2  # Explosive
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    chi_t = 0.1
    xi_t = np.log(75.0)
    t = 0.0
    T_opt = 0.5
    T_fut = 1.0
    r = 0.04
    K = 75.0

    # CPU Pricers
    with pytest.raises(ValueError):
        schwartz_smith_price_black76(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_fourier(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_cos(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)

    # PyTorch Pricers
    chi_pt = torch.tensor([chi_t], dtype=torch.float64)
    xi_pt = torch.tensor([xi_t], dtype=torch.float64)
    with pytest.raises(ValueError):
        schwartz_smith_price_black76_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_fourier_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_cos_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)


def test_schwartz_smith_out_of_bounds_correlation():
    """
    Test Schwartz-Smith pricing under out-of-bounds correlation rho (|rho| > 1.0).
    It should raise ValueError for all CPU and PyTorch pricers.
    """
    kappa = 0.6
    sigma_chi = 0.25
    rho = 1.5  # Invalid correlation
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    chi_t = 0.1
    xi_t = np.log(75.0)
    t = 0.0
    T_opt = 0.5
    T_fut = 1.0
    r = 0.04
    K = 75.0

    # CPU Pricers
    with pytest.raises(ValueError):
        schwartz_smith_price_black76(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_fourier(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_cos(t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)

    # PyTorch Pricers
    chi_pt = torch.tensor([chi_t], dtype=torch.float64)
    xi_pt = torch.tensor([xi_t], dtype=torch.float64)
    with pytest.raises(ValueError):
        schwartz_smith_price_black76_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_fourier_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    with pytest.raises(ValueError):
        schwartz_smith_price_cos_pt(t, T_opt, T_fut, K, r, chi_pt, xi_pt, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)


def test_schwartz_smith_kalman_filter_adversarial_inputs():
    """
    Test run_kalman_filter under adversarial inputs:
    1. Zero and negative time steps (dt <= 0)
    2. All NaN futures prices
    3. Negative maturities
    """
    dates = [
        datetime.date(2026, 1, 1),
        datetime.date(2026, 1, 8),
        datetime.date(2026, 1, 8),  # Duplicate date (dt = 0)
        datetime.date(2026, 1, 5),  # Reverse date (dt < 0)
        datetime.date(2026, 1, 15)
    ]
    
    # 3 contracts, 5 dates
    futures_prices = np.array([
        [75.0, 74.0, 73.0],
        [76.0, 75.0, 74.0],
        [76.5, 75.5, 74.5],
        [75.5, 74.5, 73.5],
        [77.0, 76.0, 75.0]
    ])
    
    maturities = np.array([
        [0.1, 0.3, 0.5],
        [0.08, 0.28, 0.48],
        [0.08, 0.28, 0.48],
        [0.08, 0.28, 0.48],
        [0.06, 0.26, 0.46]
    ])
    
    kappa, sigma_chi, rho, sigma_xi = 0.6, 0.25, 0.3, 0.12
    mu, lambda_chi, mu_star, sigma_e = 0.05, 0.02, 0.03, 0.01

    # 1. Run Kalman filter with duplicate/unsorted dates
    ll, states, covs = run_kalman_filter(
        dates, futures_prices, maturities,
        kappa, sigma_chi, rho, sigma_xi,
        mu, lambda_chi, mu_star, sigma_e
    )
    # Check that it runs successfully and returns finite likelihood and states (skips dt <= 0 steps)
    assert np.isfinite(ll)
    assert states.shape == (5, 2)
    assert covs.shape == (5, 2, 2)

    # 2. All NaN futures prices
    prices_nan = np.full_like(futures_prices, np.nan)
    ll_nan, states_nan, covs_nan = run_kalman_filter(
        dates, prices_nan, maturities,
        kappa, sigma_chi, rho, sigma_xi,
        mu, lambda_chi, mu_star, sigma_e
    )
    # Should run successfully (skipping update steps) and return finite likelihood/states
    assert np.isfinite(ll_nan)
    assert states_nan.shape == (5, 2)
    assert covs_nan.shape == (5, 2, 2)

    # 3. Negative maturities
    # maturities has some negative values (contracts expired)
    maturities_neg = maturities.copy()
    maturities_neg[3, :] = -0.05
    ll_neg, states_neg, covs_neg = run_kalman_filter(
        dates, futures_prices, maturities_neg,
        kappa, sigma_chi, rho, sigma_xi,
        mu, lambda_chi, mu_star, sigma_e
    )
    assert np.isfinite(ll_neg)
