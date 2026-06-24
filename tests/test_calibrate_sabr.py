"""
test_calibrate_sabr.py — Tests formulas, SSVI arbitrage-free conditions, and self-consistency.
"""

import os
import sys
import numpy as np
import pytest

# Ensure src path is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from deepvol.models.sabr import (
    sabr_iv_lognormal,
    sabr_iv_normal,
    ssvi_total_variance,
    sabr_iv_surface,
    ssvi_iv_surface,
)


def test_sabr_lognormal_atm_limit():
    """
    Checks that the ATM limit (K -> F) in sabr_iv_lognormal is handled correctly
    and matches the analytic limit.
    """
    F = 1.0
    T = 0.5
    alpha = 0.3
    beta = 0.8
    rho = -0.5
    nu = 0.4
    
    # Analytic ATM formula
    one_minus_beta = 1.0 - beta
    term1 = alpha / (F ** one_minus_beta)
    num_c1 = ((one_minus_beta ** 2) / 24.0) * (alpha ** 2) / (F ** (2.0 * one_minus_beta))
    num_c2 = 0.25 * rho * beta * nu * alpha / (F ** one_minus_beta)
    num_c3 = ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
    analytic_atm_vol = term1 * (1.0 + (num_c1 + num_c2 + num_c3) * T)
    
    # Computed ATM vol
    computed_atm_vol = sabr_iv_lognormal(F, F, T, alpha, beta, rho, nu)
    
    # Near-ATM vol
    near_atm_vol_up = sabr_iv_lognormal(F, F + 1e-6, T, alpha, beta, rho, nu)
    near_atm_vol_dn = sabr_iv_lognormal(F, F - 1e-6, T, alpha, beta, rho, nu)
    
    assert np.allclose(computed_atm_vol, analytic_atm_vol, rtol=1e-8)
    assert np.allclose(near_atm_vol_up, analytic_atm_vol, rtol=1e-4)
    assert np.allclose(near_atm_vol_dn, analytic_atm_vol, rtol=1e-4)


def test_sabr_normal_atm_limit():
    """
    Checks that the ATM limit (K -> F) in sabr_iv_normal is handled correctly
    and matches the analytic limit.
    """
    F = 1.0
    T = 0.5
    alpha = 0.3
    beta = 0.5
    rho = -0.5
    nu = 0.4
    
    # Analytic ATM formula
    term1 = alpha * (F ** beta)
    T1 = -beta * (2.0 - beta) * (alpha ** 2) / (24.0 * (F ** (2.0 - 2.0 * beta)))
    T2 = rho * beta * nu * alpha / (4.0 * (F ** (1.0 - beta)))
    T3 = ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
    analytic_atm_vol = term1 * (1.0 + (T1 + T2 + T3) * T)
    
    # Computed ATM vol
    computed_atm_vol = sabr_iv_normal(F, F, T, alpha, beta, rho, nu)
    
    # Near-ATM vol
    near_atm_vol_up = sabr_iv_normal(F, F + 1e-6, T, alpha, beta, rho, nu)
    near_atm_vol_dn = sabr_iv_normal(F, F - 1e-6, T, alpha, beta, rho, nu)
    
    assert np.allclose(computed_atm_vol, analytic_atm_vol, rtol=1e-8)
    assert np.allclose(near_atm_vol_up, analytic_atm_vol, rtol=1e-4)
    assert np.allclose(near_atm_vol_dn, analytic_atm_vol, rtol=1e-4)


def test_sabr_normal_beta_zero():
    """
    Checks that when beta=0, sabr_iv_normal works correctly and allows
    negative strikes/forwards.
    """
    F = -0.01  # negative forward rate
    K = -0.005 # negative strike rate
    T = 0.25
    alpha = 0.02
    beta = 0.0
    rho = -0.2
    nu = 0.3
    
    vol = sabr_iv_normal(F, K, T, alpha, beta, rho, nu)
    assert not np.isnan(vol)
    assert vol > 0.0
    
    # ATM case
    vol_atm = sabr_iv_normal(F, F, T, alpha, beta, rho, nu)
    assert not np.isnan(vol_atm)
    assert vol_atm > 0.0


def test_ssvi_no_arbitrage_guarantees():
    """
    Generates random SSVI surfaces satisfying Gatheral-Jacquier conditions
    and verifies that they numerically satisfy:
    1. Calendar spread arbitrage-free condition: w(k, t_i) <= w(k, t_j) for t_i < t_j
    2. Butterfly arbitrage-free condition: risk-neutral density g(k) >= 0
    """
    np.random.seed(42)
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    k_grid = np.linspace(-0.5, 0.5, 20)
    
    for _ in range(50):
        rho = np.random.uniform(-0.9, 0.9)
        gamma = np.random.uniform(0.1, 0.5)
        # eta * (1 + |rho|) <= 2.0
        eta = np.random.uniform(0.05, 2.0 / (1.0 + abs(rho)))
        
        # Monotone ATM vols
        vols = np.sort(np.random.uniform(0.08, 0.8, len(T_grid)))
        theta_grid = vols ** 2 * T_grid
        
        # Verify monotone total variance at ATM (k=0)
        assert np.all(np.diff(theta_grid) >= 0.0)
        
        # Compute total variance surface
        w_surf = np.zeros((len(T_grid), len(k_grid)))
        for i, theta in enumerate(theta_grid):
            w_surf[i] = ssvi_total_variance(k_grid, theta, rho, eta, gamma)
            
        # 1. Check calendar arbitrage: w_surf[i, j] must be non-decreasing in i
        for i in range(1, len(T_grid)):
            diff = w_surf[i] - w_surf[i-1]
            # Allow tiny numerical tolerance
            assert np.all(diff >= -1e-15), f"Calendar arbitrage detected: diff = {diff.min()}"
            
        # 2. Check butterfly arbitrage (density g(k) >= 0)
        # g(k) = (1 - k*w'/(2w))^2 - (w'^2 / 4)*(1/w + 1/4) + w'' / 2
        dk = k_grid[1] - k_grid[0]
        
        for i in range(len(T_grid)):
            w = w_surf[i]
            
            # Central differences for derivatives
            w_prime = np.zeros_like(w)
            w_prime[1:-1] = (w[2:] - w[:-2]) / (2.0 * dk)
            w_prime[0] = (w[1] - w[0]) / dk
            w_prime[-1] = (w[-1] - w[-2]) / dk
            
            w_prime2 = np.zeros_like(w)
            w_prime2[1:-1] = (w[2:] - 2.0 * w[1:-1] + w[:-2]) / (dk ** 2)
            w_prime2[0] = w_prime2[1]
            w_prime2[-1] = w_prime2[-2]
            
            term1 = (1.0 - (k_grid * w_prime) / (2.0 * w)) ** 2
            term2 = (w_prime ** 2 / 4.0) * (1.0 / w + 0.25)
            term3 = 0.5 * w_prime2
            
            g_k = term1 - term2 + term3
            
            # Risk-neutral density should be positive (allowing tiny numerical tolerance near boundaries)
            assert np.all(g_k >= -1e-6), f"Butterfly arbitrage detected: g_k = {g_k.min()} for rho={rho}, eta={eta}, gamma={gamma}"


def test_sabr_iv_surface_shape():
    """
    Checks that sabr_iv_surface returns correct dimensions and values.
    """
    T_grid = np.array([0.1, 0.5, 1.0])
    k_grid = np.array([-0.2, 0.0, 0.2])
    
    surface = sabr_iv_surface(
        F=1.0,
        T_grid=T_grid,
        k_grid=k_grid,
        alpha=0.3,
        beta=1.0,
        rho=-0.5,
        nu=0.4,
        iv_type="lognormal"
    )
    
    assert surface.shape == (3, 3)
    assert np.all(surface > 0)
    assert not np.any(np.isnan(surface))


def test_ssvi_iv_surface_shape():
    """
    Checks that ssvi_iv_surface returns correct dimensions and values.
    """
    T_grid = np.array([0.1, 0.5, 1.0])
    k_grid = np.array([-0.2, 0.0, 0.2])
    theta_grid = np.array([0.02, 0.05, 0.1])
    
    surface = ssvi_iv_surface(
        T_grid=T_grid,
        k_grid=k_grid,
        theta_grid=theta_grid,
        rho=-0.5,
        eta=0.5,
        gamma=0.3
    )
    
    assert surface.shape == (3, 3)
    assert np.all(surface > 0)
    assert not np.any(np.isnan(surface))


def test_calibrate_sabr_fast_self_consistency():
    """Verify that calibrate_sabr (Newton) recovers synthetic parameters."""
    from deepvol.calibration.calibrate_newton import calibrate_sabr as calibrate_sabr_fast
    from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MirrorPaddedFNO2d(param_dim=3).to(device)
    model.load_state_dict(torch.load("artifacts/weights/fno_sabr_final_prod.pth", map_location=device, weights_only=True))
    model.eval()
    
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    alpha_t = 0.08
    rho_t = -0.4
    nu_t = 0.3
    
    # Generate target using pricing formula
    iv_target = sabr_iv_surface(
        F=1.0,
        T_grid=T_grid,
        k_grid=K_grid,
        alpha=alpha_t,
        beta=1.0,
        rho=rho_t,
        nu=nu_t,
        iv_type="lognormal"
    )
    
    res = calibrate_sabr_fast(model, iv_target, T_grid, K_grid, max_iter=25, n_starts=2)
    
    assert res["final_mse"] < 1e-4
    assert abs(res["alpha"] - alpha_t) < 0.015
    assert abs(res["rho"] - rho_t) < 0.15
    assert abs(res["nu"] - nu_t) < 0.10


def test_calibrate_ssvi_fast_self_consistency():
    """Verify that calibrate_ssvi (Newton) recovers synthetic parameters."""
    from deepvol.calibration.calibrate_newton import calibrate_ssvi as calibrate_ssvi_fast
    from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MirrorPaddedFNO2d(param_dim=11).to(device)
    model.load_state_dict(torch.load("artifacts/weights/fno_ssvi_final_prod.pth", map_location=device, weights_only=True))
    model.eval()
    
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    rho_t = -0.3
    eta_t = 0.6
    gamma_t = 0.3
    # Monotone increasing ATM variances
    theta_grid = np.array([0.01, 0.02, 0.035, 0.05, 0.065, 0.08, 0.09, 0.10])
    
    from deepvol.calibration.calibrate_newton import _fno_predict_real_iv, _make_spatial_input, _load_normalizers
    _load_normalizers("ssvi")
    spatial = _make_spatial_input(T_grid, K_grid, device)
    
    raw_params = torch.cat([
        torch.tensor(theta_grid, dtype=torch.float32, device=device),
        torch.tensor([rho_t, eta_t, gamma_t], dtype=torch.float32, device=device)
    ]).unsqueeze(0)
    
    with torch.no_grad():
        iv_target = _fno_predict_real_iv(model, raw_params, spatial).cpu().numpy()
    
    # Decoupled calibration with known theta_atm_init
    res = calibrate_ssvi_fast(model, iv_target, T_grid, K_grid, theta_atm_init=theta_grid, max_iter=25, n_starts=2)
    
    assert res["final_mse"] < 1e-4
    assert abs(res["rho"] - rho_t) < 0.05
    assert abs(res["eta"] - eta_t) < 0.05
    assert abs(res["gamma"] - gamma_t) < 0.05


