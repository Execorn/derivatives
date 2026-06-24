import os
import sys
import json
import numpy as np
import pytest
import torch
from pathlib import Path

# Inject src path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from deepvol.arbitrage.surface_completion import (
    check_calendar_spread,
    check_butterfly,
    fit_svi_slice,
    enforce_calendar_spread_monotonicity,
    complete_surface
)
from deepvol.market.spx_data import T_GRID, K_GRID


def test_enforce_calendar_spread_monotonicity():
    # Setup a variance surface with calendar spread violations along T-axis (axis=0)
    var = np.array([
        [0.1, 0.1, 0.1],
        [0.08, 0.08, 0.08],  # Violation: 0.08 < 0.1
        [0.15, 0.15, 0.15],
        [0.12, 0.12, 0.12]   # Violation: 0.12 < 0.15
    ])
    rearranged = enforce_calendar_spread_monotonicity(var, axis=0)
    
    # Check that it is non-decreasing along the T-axis
    diffs = np.diff(rearranged, axis=0)
    assert np.all(diffs >= 0.0), f"Monotone rearrangement failed, diffs: {diffs}"


def test_check_butterfly_arbitrage():
    T_grid = np.array([1.0])
    K_grid = np.array([-0.5, 0.0, 0.5])
    
    # Slice with a butterfly violation: middle strike has huge IV, outer strikes have tiny IV
    iv_violation = np.array([[0.05, 0.95, 0.05]])
    mask = check_butterfly(iv_violation, K_grid, T_grid, S=1.0)
    assert np.all(mask == True), "Butterfly spread violation check failed to detect arbitrage."
    
    # Slice with no violation (flat IV)
    iv_ok = np.array([[0.20, 0.20, 0.20]])
    mask_ok = check_butterfly(iv_ok, K_grid, T_grid, S=1.0)
    assert np.all(mask_ok == False), "Butterfly spread violation check detected false arbitrage on flat IV surface."


def test_fit_svi_slice_bounds():
    k = np.linspace(-0.5, 0.5, 11)
    # Generate SVI-like total variance slice
    total_var = 0.04 + 0.1 * (-0.3 * k + np.sqrt(k**2 + 0.05))
    
    params = fit_svi_slice(k, total_var)
    assert params["a"] >= 0.0, "SVI parameter 'a' is negative."
    assert params["b"] >= 0.0, "SVI parameter 'b' is negative."
    assert -1.0 <= params["rho"] <= 1.0, "SVI parameter 'rho' is out of bounds."
    assert params["sigma"] >= 1e-4, "SVI parameter 'sigma' is below minimum bound."
    assert params["b"] * (1.0 + abs(params["rho"])) <= 4.0001, "SVI parameter 'b(1+|rho|)' constraint violated."


def test_complete_surface_cubic_spline():
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    nT, nK = len(T_grid), len(K_grid)
    
    # Create flat IV surface (completely arbitrage-free)
    sparse_iv = np.full((nT, nK), 0.25)
    
    mask = np.ones((nT, nK), dtype=bool)
    mask[2, 3] = False
    mask[4, 5] = False
    
    completed = complete_surface(sparse_iv, mask, T_grid, K_grid, method="cubic_spline")
    
    assert completed.shape == (nT, nK)
    assert not np.any(np.isnan(completed))
    # Check that observed entries match closely
    assert np.allclose(completed[mask], sparse_iv[mask], atol=1e-2)


def test_mask_completion_and_rmse(fno_v2_model):
    device = next(fno_v2_model.parameters()).device
    
    import deepvol.calibration.calibrate_bfgs as calibrate
    orig_v1 = calibrate._NORM_VERSIONS["v1"]
    try:
        # Load v2 normalizers and generate synthetic target surface for 2024-01-02 using FNO v2
        calibrate._NORM_VERSIONS["v1"] = calibrate._NORM_VERSIONS["v2"]
        calibrate._param_norm = None
        calibrate._iv_norm = None
        
        theta_raw = torch.tensor([[1.0, 0.08, 0.5, -0.7, 0.08, 0.08]], dtype=torch.float32, device=device)
        spatial = calibrate._make_spatial_input(T_GRID, K_GRID, device)
        with torch.no_grad():
            iv_surface_t = calibrate._fno_predict_real_iv(fno_v2_model, theta_raw, spatial)
        target_surface = iv_surface_t.cpu().numpy()
    finally:
        calibrate._NORM_VERSIONS["v1"] = orig_v1
        calibrate._param_norm = None
        calibrate._iv_norm = None

    # Randomly mask 40% of the surface
    rng = np.random.default_rng(42)
    mask = rng.random(target_surface.shape) >= 0.40
    
    sparse_iv = target_surface.copy()
    sparse_iv[~mask] = np.nan
    
    results = {}
    
    # Run completion for each method
    methods = ["cubic_spline", "svi", "fno"]
    for method in methods:
        completed = complete_surface(sparse_iv, mask, T_GRID, K_GRID, method=method)
        
        # Calculate RMSE on held-out quotes
        rmse = float(np.sqrt(np.mean((completed[~mask] - target_surface[~mask])**2)))
        
        # Check violations
        cal_viols = int(np.sum(check_calendar_spread(completed, T_GRID)))
        butt_viols = int(np.sum(check_butterfly(completed, K_GRID, T_GRID, S=1.0)))
        
        results[method] = {
            "rmse": rmse,
            "calendar_violations": cal_viols,
            "butterfly_violations": butt_viols
        }
        
    # Save comparison results
    out_dir = Path("results/spx_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "completion_test.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Print results to stdout
    print("\nSurface Completion Comparison Results:")
    print(json.dumps(results, indent=4))
    
    # Assertions: FNO and SVI completion have no calendar spread or butterfly spread violations
    assert results["fno"]["calendar_violations"] == 0, "FNO completion has calendar spread arbitrage violations!"
    assert results["fno"]["butterfly_violations"] == 0, "FNO completion has butterfly spread arbitrage violations!"
    assert results["svi"]["calendar_violations"] == 0, "SVI completion has calendar spread arbitrage violations!"
    assert results["svi"]["butterfly_violations"] == 0, "SVI completion has butterfly spread arbitrage violations!"
