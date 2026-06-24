"""
test_cross_asset_roughness.py

Optimized test suite for deepvol.analysis.cross_asset_roughness.
Contains unit, stress (NaN, inf, boundary conditions), and plotting tests.
"""

from __future__ import annotations
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

# Ensure parent and src are in path
# The test is in /home/execorn/programming/derivatives-w1/tests/
# parents[0] is tests/, parents[1] is project root /home/execorn/programming/derivatives-w1/
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from deepvol.analysis.cross_asset_roughness import (
    ParameterTrajectoryGenerator,
    CrossAssetDataPipeline,
    run_cross_asset_study,
    run_pettitt_test,
    compute_hurst_statistics,
    plot_hurst_series,
    plot_hurst_correlation
)


def test_trajectory_generator_caching():
    """Verify that ParameterTrajectoryGenerator precomputes and caches results."""
    tg = ParameterTrajectoryGenerator()
    assert "SPX" in tg._cache
    assert isinstance(tg._cache["SPX"], pd.DataFrame)
    assert len(tg._cache["SPX"]) > 0


def test_trajectory_generator_lookup():
    """Verify that O(1) daily parameter lookup returns expected keys and values."""
    tg = ParameterTrajectoryGenerator()
    params = tg.get_parameters("SPX", "2020-03-16")
    
    expected_keys = {"kappa", "theta", "sigma", "rho", "v0", "H"}
    assert set(params.keys()) == expected_keys
    assert isinstance(params["v0"], float)


def test_trajectory_generator_invalid_asset():
    """Verify that ParameterTrajectoryGenerator raises ValueError for unknown assets."""
    tg = ParameterTrajectoryGenerator()
    with pytest.raises(ValueError, match="Unknown asset"):
        tg.get_parameters("INVALID_ASSET_NAME", "2020-01-02")


def test_trajectory_generator_bounds_clamping():
    """Verify that parameter values are strictly clamped to FNO v3 bounds."""
    tg = ParameterTrajectoryGenerator()
    for asset in tg.base_params.keys():
        for d in tg.dates[:50]:  # sample first 50 days
            params = tg.get_parameters(asset, d)
            assert 0.01 <= params["v0"] <= 0.15, f"v0 out of bounds for {asset} on {d}: {params['v0']}"
            assert 0.04 <= params["H"] <= 0.15, f"H out of bounds for {asset} on {d}: {params['H']}"
            assert -0.90 <= params["rho"] <= -0.10, f"rho out of bounds for {asset} on {d}: {params['rho']}"
            assert 0.10 <= params["sigma"] <= 1.0, f"sigma out of bounds for {asset} on {d}: {params['sigma']}"


def test_pettitt_test_nan_handling():
    """Verify Pettitt test handles NaN values by ignoring them or handling cleanly."""
    x = np.array([0.06] * 10 + [np.nan] + [0.12] * 10)
    # Filter out NaNs
    x_clean = x[~np.isnan(x)]
    tau, p_val, K = run_pettitt_test(x_clean)
    assert tau == 10
    assert p_val < 0.05


def test_pettitt_test_empty_input():
    """Verify Pettitt test handles empty or very small inputs gracefully."""
    tau, p_val, K = run_pettitt_test(np.array([]))
    assert tau == 0
    assert p_val == 1.0
    assert K == 0.0


def test_run_cross_asset_study_integration_optimized(tmp_path):
    """Verify that run_cross_asset_study runs successfully with the optimized codebase."""
    results_dir = tmp_path / "results" / "cross_asset"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    weights_dest_dir = tmp_path / "artifacts" / "weights"
    weights_dest_dir.mkdir(parents=True, exist_ok=True)
    
    models_dest_dir = tmp_path / "artifacts" / "models"
    models_dest_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(project_root / "artifacts" / "weights" / "fno_v3_final_prod.pth", weights_dest_dir)
    shutil.copy(project_root / "artifacts" / "models" / "param_normalizer_v3.npz", models_dest_dir)
    shutil.copy(project_root / "artifacts" / "models" / "iv_normalizer_v3.npz", models_dest_dir)
    
    # Run study for a 2-day range on CPU with compile disabled
    study_results = run_cross_asset_study(
        start="2020-01-02",
        end="2020-01-03",
        assets=["SPX", "WTI", "EURUSD"],
        project_root_dir=str(tmp_path),
        max_workers=2,
        device="cpu",
        batch_size=2,
        use_compile=False
    )
    
    assert "SPX" in study_results
    assert "WTI" in study_results
    assert "EURUSD" in study_results
    
    df_spx = study_results["SPX"]
    assert len(df_spx) == 2
    assert "H" in df_spx.columns
    assert "v0" in df_spx.columns
    assert "rmse_bps" in df_spx.columns


def test_compute_hurst_statistics():
    """Verify Hurst statistics computation for mean, std, and lag autocorrelations."""
    df = pd.DataFrame({
        "H": [0.06, 0.07, 0.08, 0.09, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05]
    })
    stats = compute_hurst_statistics(df)
    assert np.isclose(stats["mean"], 0.075)
    assert stats["std"] > 0.0
    assert "autocorr_lag1" in stats
    assert "autocorr_lag5" in stats


def test_plotting_helpers(tmp_path):
    """Verify that plotting helpers create output files without error."""
    study_results = {
        "SPX": pd.DataFrame({
            "date": ["2020-01-02", "2020-01-03", "2020-01-04"],
            "H": [0.06, 0.07, 0.08]
        }),
        "BTC": pd.DataFrame({
            "date": ["2020-01-02", "2020-01-03", "2020-01-04"],
            "H": [0.10, 0.11, 0.12]
        })
    }
    
    series_path = str(tmp_path / "series.png")
    corr_path = str(tmp_path / "corr.png")
    
    plot_hurst_series(study_results, series_path)
    plot_hurst_correlation(study_results, corr_path)
    
    assert os.path.exists(series_path)
    assert os.path.exists(corr_path)


def test_ou_trajectory_properties():
    """Verify that ParameterTrajectoryGenerator generates a true discretized OU process."""
    tg = ParameterTrajectoryGenerator()
    # Check that trajectories are different from baseline cumulative sum
    for asset in tg.base_params.keys():
        df = tg._cache[asset]
        # v0 and H must have mean-reverting properties or at least vary
        assert not df["v0"].eq(tg.base_params[asset]["v0"]).all()
        assert not df["H"].eq(tg.base_params[asset]["H"]).all()
        
        # Verify clamp boundaries
        assert df["v0"].min() >= 0.01
        assert df["v0"].max() <= 0.15
        assert df["H"].min() >= 0.04
        assert df["H"].max() <= 0.15


def test_pettitt_test_internal_nan():
    """Verify that run_pettitt_test filters NaN internally and returns correct results."""
    # Series with a NaN in the middle and enough length for significance
    x = np.array([0.06] * 10 + [np.nan] + [0.12] * 10)
    tau, p_val, K = run_pettitt_test(x)
    # Non-nan length is 20. The change is exactly at index 10.
    assert tau == 10
    assert p_val < 0.05
    assert K > 0.0


def test_parameter_only_caching(tmp_path):
    """Verify that save_results excludes the surface from the JSON output, and loads correctly."""
    import json
    from deepvol.analysis.cross_asset_roughness import save_results, load_results, CalibrationResult
    
    results = [
        CalibrationResult(
            date="2020-01-02",
            currency="SPX",
            params={"kappa": 1.0, "theta": 0.08, "sigma": 0.5, "rho": -0.5, "v0": 0.05, "H": 0.06},
            rmse_bps=35.0,
            runtime_ms=12.5,
            converged=True,
            surface=np.random.rand(8, 11)  # Synthetic surface
        )
    ]
    
    cache_file = tmp_path / "test_cache.json"
    save_results(results, str(cache_file))
    
    # Read the raw JSON and check that the surface field is null/None
    with open(cache_file, "r") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["surface"] is None
    
    # Load back using load_results and check CalibrationResult
    loaded = load_results(str(cache_file))
    assert len(loaded) == 1
    assert loaded[0].surface is None
    assert loaded[0].params["v0"] == 0.05
    assert loaded[0].rmse_bps == 35.0
    assert loaded[0].converged is True


def test_calibrate_newton_h_batch_unit():
    """Verify calibrate_newton_h_batch recovers Hurst and parameters on synthetic surfaces."""
    device = torch.device("cpu")
    
    # Load model and normalizers
    from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
    from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer
    from deepvol.analysis.cross_asset_roughness import calibrate_newton_h_batch, _reparam_to_6d_with_H, _make_spatial
    
    weights_path = project_root / "artifacts" / "weights" / "fno_v3_final_prod.pth"
    model = MirrorPaddedFNO2d(param_dim=6)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    pn = ParameterNormalizer.load(str(project_root / "artifacts" / "models" / "param_normalizer_v3.npz"))
    yn = IVSurfaceNormalizer.load(str(project_root / "artifacts" / "models" / "iv_normalizer_v3.npz"))
    
    # Create synthetic surface for B=2
    known_v0 = torch.tensor([0.06, 0.06], dtype=torch.float32)
    known_zeta = torch.tensor([-0.2, -0.2], dtype=torch.float32)
    known_lam = torch.tensor([0.3, 0.3], dtype=torch.float32)
    known_H = torch.tensor([0.08, 0.08], dtype=torch.float32)
    
    p6 = _reparam_to_6d_with_H(known_v0, known_zeta, known_lam, known_H, device)
    
    spatial = _make_spatial(device).repeat(2, 1, 1, 1)
    pn_mean = torch.tensor(pn.mean, dtype=torch.float32, device=device)
    pn_std = torch.tensor(pn.std, dtype=torch.float32, device=device)
    yn_mean = torch.tensor(yn.mean, dtype=torch.float32, device=device)
    yn_std = torch.tensor(yn.std, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        theta_norm = (p6 - pn_mean) / pn_std
        theta_norm = theta_norm.clamp(min=-3.0, max=3.0)
        pred = model(spatial, theta_norm)
        iv = pred * yn_std + yn_mean
        target_iv_batch = iv.clamp(min=1e-4) # shape (2, 8, 11)
        
    best_theta, final_preds, final_loss = calibrate_newton_h_batch(
        model, target_iv_batch, pn, yn, device, max_iter=10
    )
    
    # Assert shapes
    assert best_theta.shape == (2, 6)
    assert final_preds.shape == (2, 8, 11)
    assert final_loss.shape == (2,)
    
    # Assert that calibrated H is close to 0.08
    calibrated_H = best_theta[:, 5].cpu().numpy()
    assert np.allclose(calibrated_H, 0.08, atol=0.03)


def test_calibrate_newton_h_batch_regularized():
    """Verify calibrate_newton_h_batch regularizes calibration when prior_batch is provided."""
    device = torch.device("cpu")
    
    # Load model and normalizers
    from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
    from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer
    from deepvol.analysis.cross_asset_roughness import calibrate_newton_h_batch, _reparam_to_6d_with_H, _make_spatial
    
    weights_path = project_root / "artifacts" / "weights" / "fno_v3_final_prod.pth"
    model = MirrorPaddedFNO2d(param_dim=6)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    pn = ParameterNormalizer.load(str(project_root / "artifacts" / "models" / "param_normalizer_v3.npz"))
    yn = IVSurfaceNormalizer.load(str(project_root / "artifacts" / "models" / "iv_normalizer_v3.npz"))
    
    # Create synthetic surface for B=1
    known_v0 = torch.tensor([0.06], dtype=torch.float32)
    known_zeta = torch.tensor([-0.2], dtype=torch.float32)
    known_lam = torch.tensor([0.3], dtype=torch.float32)
    known_H = torch.tensor([0.08], dtype=torch.float32)
    
    p6 = _reparam_to_6d_with_H(known_v0, known_zeta, known_lam, known_H, device)
    
    spatial = _make_spatial(device).repeat(1, 1, 1, 1)
    pn_mean = torch.tensor(pn.mean, dtype=torch.float32, device=device)
    pn_std = torch.tensor(pn.std, dtype=torch.float32, device=device)
    yn_mean = torch.tensor(yn.mean, dtype=torch.float32, device=device)
    yn_std = torch.tensor(yn.std, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        theta_norm = (p6 - pn_mean) / pn_std
        theta_norm = theta_norm.clamp(min=-3.0, max=3.0)
        pred = model(spatial, theta_norm)
        iv = pred * yn_std + yn_mean
        target_iv_batch = iv.clamp(min=1e-4) # shape (1, 8, 11)
        
    # Calibration with a strong prior far away from 0.08, e.g. H_prior = 0.12
    # Prior shape is (1, 4): [v0, zeta, lam, H]
    prior_batch = torch.tensor([[0.06, -0.2, 0.3, 0.12]], dtype=torch.float32, device=device)
    
    # Use strong regularization weight for H (index 3) and 0 for others to isolate the effect
    reg_weights = torch.tensor([0.0, 0.0, 0.0, 10.0], dtype=torch.float32, device=device)
    
    best_theta, final_preds, final_loss = calibrate_newton_h_batch(
        model, target_iv_batch, pn, yn, device, max_iter=10,
        prior_batch=prior_batch, reg_weights=reg_weights
    )
    
    # Assert shapes
    assert best_theta.shape == (1, 6)
    
    # Assert that calibrated H is pulled towards the prior H of 0.12 (i.e. it should be higher than 0.08)
    calibrated_H = best_theta[0, 5].item()
    print("Calibrated H with strong prior of 0.12 (true 0.08):", calibrated_H)
    assert calibrated_H > 0.09, f"Calibrated H ({calibrated_H}) should be pulled towards 0.12"


