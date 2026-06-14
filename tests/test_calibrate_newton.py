import pytest
import torch
import numpy as np
import calibrate
from calibrate import _make_spatial_input, _fno_predict_real_iv, compute_fim_ellipsoid
from calibrate_fast import _reparam_to_6d, calibrate_newton

# Mark the whole module to skip if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

def test_calibrate_newton_self_consistency(fno_v2_model):
    device = next(fno_v2_model.parameters()).device
    orig_norm_versions = calibrate._NORM_VERSIONS.copy()
    calibrate._NORM_VERSIONS["v1"] = calibrate._NORM_VERSIONS["v2"]
    calibrate._param_norm = None
    calibrate._iv_norm = None
    
    try:
        T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
        K_grid = np.linspace(-0.5, 0.5, 11)
        spatial = _make_spatial_input(T_grid, K_grid, device)
        
        v0_val = 0.10
        zeta_val = -0.4
        lambda_val = 0.3
        
        v0_t = torch.tensor([v0_val], dtype=torch.float32, device=device)
        zeta_t = torch.tensor([zeta_val], dtype=torch.float32, device=device)
        lam_t = torch.tensor([lambda_val], dtype=torch.float32, device=device)
        p6 = _reparam_to_6d(v0_t, zeta_t, lam_t, device)
        
        with torch.no_grad():
            target_iv_t = _fno_predict_real_iv(fno_v2_model, p6, spatial)
        target_iv = target_iv_t.cpu().numpy()
        
        res = calibrate_newton(fno_v2_model, target_iv, T_grid, K_grid, max_iter=30, tol=1e-5)
        
        assert res["final_mse"] < 1e-4
    finally:
        calibrate._NORM_VERSIONS = orig_norm_versions
        calibrate._param_norm = None
        calibrate._iv_norm = None

def test_calibrate_newton_theta_history(fno_v2_model):
    device = next(fno_v2_model.parameters()).device
    orig_norm_versions = calibrate._NORM_VERSIONS.copy()
    calibrate._NORM_VERSIONS["v1"] = calibrate._NORM_VERSIONS["v2"]
    calibrate._param_norm = None
    calibrate._iv_norm = None
    
    try:
        T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
        K_grid = np.linspace(-0.5, 0.5, 11)
        spatial = _make_spatial_input(T_grid, K_grid, device)
        
        v0_t = torch.tensor([0.10], dtype=torch.float32, device=device)
        zeta_t = torch.tensor([-0.4], dtype=torch.float32, device=device)
        lam_t = torch.tensor([0.3], dtype=torch.float32, device=device)
        p6 = _reparam_to_6d(v0_t, zeta_t, lam_t, device)
        
        with torch.no_grad():
            target_iv_t = _fno_predict_real_iv(fno_v2_model, p6, spatial)
        target_iv = target_iv_t.cpu().numpy()
        
        res = calibrate_newton(fno_v2_model, target_iv, T_grid, K_grid, max_iter=5)
        
        assert "theta_history" in res
        assert isinstance(res["theta_history"], list)
        assert len(res["theta_history"]) > 0
        for theta in res["theta_history"]:
            assert isinstance(theta, np.ndarray)
            assert theta.shape == (3,)
    finally:
        calibrate._NORM_VERSIONS = orig_norm_versions
        calibrate._param_norm = None
        calibrate._iv_norm = None

def test_fim_properties(fno_v2_model):
    device = next(fno_v2_model.parameters()).device
    orig_norm_versions = calibrate._NORM_VERSIONS.copy()
    calibrate._NORM_VERSIONS["v1"] = calibrate._NORM_VERSIONS["v2"]
    calibrate._param_norm = None
    calibrate._iv_norm = None
    
    try:
        T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
        K_grid = np.linspace(-0.5, 0.5, 11)
        
        v0 = 0.10
        zeta = -0.4
        lam = 0.3
        
        res = compute_fim_ellipsoid(fno_v2_model, v0, zeta, lam, T_grid, K_grid)
        
        expected_keys = {"fim_matrix", "cov_matrix", "std_errors", "corr_matrix", "ci_95"}
        assert expected_keys.issubset(res.keys())
        
        std_errors = res["std_errors"]
        assert np.all(std_errors > 0)
        
        corr = res["corr_matrix"]
        assert corr.shape == (3, 3)
        np.testing.assert_allclose(corr, corr.T, atol=1e-7)
        np.testing.assert_allclose(np.diagonal(corr), np.ones(3), atol=1e-6)
        
    finally:
        calibrate._NORM_VERSIONS = orig_norm_versions
        calibrate._param_norm = None
        calibrate._iv_norm = None
