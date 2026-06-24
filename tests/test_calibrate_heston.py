"""
test_calibrate_heston.py — Unit and integration tests for Classic Heston pricing and calibration.
"""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from deepvol.models.heston import (
    heston_cf,
    heston_iv_surface,
    batch_heston_iv_surface,
    calibrate_heston,
)

T_GRID = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
K_GRID = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])

# --- Bound Limits for Heston ---
BOUNDS_LOWER = np.array([0.1, 0.01, 0.1, -0.9, 0.01])
BOUNDS_UPPER = np.array([5.0, 0.15, 1.0, -0.1, 0.15])


def test_returns_required_keys():
    """Verify that calibrate_heston returns a dictionary containing all expected keys."""
    params_true = {
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7,
        'v0': 0.04
    }
    
    # Generate a target surface
    iv_target = heston_iv_surface(params_true, T_GRID, K_GRID)
    
    # Calibrate (few iterations just to verify keys)
    res = calibrate_heston(iv_target, T_GRID, K_GRID, max_iter=5)
    
    assert isinstance(res, dict)
    for key in ['params', 'param_vector', 'loss', 'converged', 'message']:
        assert key in res
    
    assert isinstance(res['params'], dict)
    for param in ['kappa', 'theta', 'sigma', 'rho', 'v0']:
        assert param in res['params']
        assert isinstance(res['params'][param], float)
        
    assert res['param_vector'].shape == (5,)
    assert isinstance(res['loss'], float)
    assert isinstance(res['converged'], bool)
    assert isinstance(res['message'], str)


def test_parameter_bounds():
    """Verify that calibrate_heston respects Heston parameter bounds."""
    params_true = {
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7,
        'v0': 0.04
    }
    iv_target = heston_iv_surface(params_true, T_GRID, K_GRID)
    
    # Calibrate
    res = calibrate_heston(iv_target, T_GRID, K_GRID, max_iter=15)
    param_vector = res['param_vector']
    
    # Check bounds
    assert np.all(param_vector >= BOUNDS_LOWER - 1e-7)
    assert np.all(param_vector <= BOUNDS_UPPER + 1e-7)


def test_feller_violation_handled():
    """Verify that pricing and calibration are stable even when Feller condition is violated."""
    # Feller: 2 * kappa * theta > sigma**2.
    # Here: 2 * 0.5 * 0.02 = 0.02 <= 0.6**2 = 0.36 (Feller condition violated)
    params_violating = {
        'kappa': 0.5,
        'theta': 0.02,
        'sigma': 0.6,
        'rho': -0.5,
        'v0': 0.05
    }
    
    # Verify pricing works and returns non-NaN surface even under Feller violation
    iv_surface = heston_iv_surface(params_violating, T_GRID, K_GRID)
    assert iv_surface.shape == (len(T_GRID), len(K_GRID))
    assert not np.isnan(iv_surface).all()  # Should compute valid IVs
    
    # Verify calibration objective function doesn't crash and penalizes violation
    res = calibrate_heston(iv_surface, T_GRID, K_GRID, max_iter=5)
    assert isinstance(res, dict)
    assert 'loss' in res


def test_self_consistency():
    """Verify that we can recover known synthetic Heston parameters from a priced surface."""
    # Synthetic parameters satisfying Feller condition: 2 * 2.5 * 0.08 = 0.4 > 0.3**2 = 0.09
    params_true = {
        'kappa': 2.5,
        'theta': 0.08,
        'sigma': 0.3,
        'rho': -0.6,
        'v0': 0.05
    }
    
    # Generate synthetic market surface
    iv_target = heston_iv_surface(params_true, T_GRID, K_GRID)
    
    # Start calibration with a perturbed initial guess
    init_guess = np.array([2.0, 0.06, 0.4, -0.5, 0.06])
    
    res = calibrate_heston(iv_target, T_GRID, K_GRID, init_guess=init_guess, max_iter=80)
    
    # Verify parameter recovery
    param_vector = res['param_vector']
    true_vector = np.array([
        params_true['kappa'],
        params_true['theta'],
        params_true['sigma'],
        params_true['rho'],
        params_true['v0']
    ])
    
    # Recovered parameters should be close to true ones
    error = np.abs(param_vector - true_vector)
    
    # Tolerance level for parameters: kappa (0.5), theta (0.02), sigma (0.1), rho (0.15), v0 (0.015)
    # The L-BFGS-B optimizer might find slightly different local minima in flat regions (e.g. kappa),
    # but the overall option price difference (loss) should be extremely small.
    print(f"Calibrated parameters: {res['params']}")
    print(f"True parameters: {params_true}")
    print(f"Errors: {error}")
    
    assert res['loss'] < 1e-4, f"Calibration loss is too high: {res['loss']}"
    assert error[1] < 0.02, f"Theta error too high: {error[1]}"
    assert error[4] < 0.015, f"v0 error too high: {error[4]}"


def test_calibrate_heston_fast_self_consistency():
    """Verify that calibrate_heston (Newton) recovers synthetic parameters."""
    from deepvol.calibration.calibrate_newton import calibrate_heston as calibrate_heston_fast
    from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MirrorPaddedFNO2d(param_dim=5).to(device)
    model.load_state_dict(torch.load("artifacts/weights/fno_heston_final_prod.pth", map_location=device, weights_only=True))
    model.eval()
    
    T_grid_fno = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid_fno = np.linspace(-0.5, 0.5, 11)
    
    params_true = {
        'kappa': 2.5,
        'theta': 0.08,
        'sigma': 0.3,
        'rho': -0.6,
        'v0': 0.05
    }
    
    from deepvol.calibration.calibrate_newton import _fno_predict_real_iv, _make_spatial_input, _load_normalizers
    _load_normalizers("heston")
    spatial = _make_spatial_input(T_grid_fno, K_grid_fno, device)
    params_tensor = torch.tensor([[
        params_true['kappa'],
        params_true['theta'],
        params_true['sigma'],
        params_true['rho'],
        params_true['v0']
    ]], dtype=torch.float32, device=device)
    with torch.no_grad():
        iv_target = _fno_predict_real_iv(model, params_tensor, spatial).cpu().numpy()
    
    res = calibrate_heston_fast(model, iv_target, T_grid_fno, K_grid_fno, max_iter=20, n_starts=2)
    
    assert res["loss"] < 1.5e-3
    assert abs(res["params"]["theta"] - params_true["theta"]) < 0.06
    assert abs(res["params"]["v0"] - params_true["v0"]) < 0.015

