import os
import pytest
import numpy as np
import torch
import calibrate
from normalizers import ParameterNormalizer, IVSurfaceNormalizer

def test_parameter_normalizer_roundtrip():
    path = "artifacts/models/param_normalizer_v2.npz"
    assert os.path.exists(path), f"File {path} does not exist"
    norm = ParameterNormalizer.load(path)
    
    rng = np.random.default_rng(42)
    kappa = rng.uniform(0.1, 5.0, 10)
    theta = rng.uniform(0.01, 0.15, 10)
    sigma = rng.uniform(0.1, 1.0, 10)
    rho = rng.uniform(-0.9, -0.1, 10)
    v0 = rng.uniform(0.01, 0.15, 10)
    H = rng.uniform(0.02, 0.15, 10)
    
    params = np.stack([kappa, theta, sigma, rho, v0, H], axis=-1)
    params_tensor = torch.tensor(params, dtype=torch.float32)
    
    transformed = norm.transform_tensor(params_tensor)
    reconstructed = norm.inverse_transform_tensor(transformed)
    
    torch.testing.assert_close(reconstructed, params_tensor, atol=1e-5, rtol=1e-5)

def test_iv_normalizer_roundtrip():
    path = "artifacts/models/iv_normalizer_v2.npz"
    assert os.path.exists(path), f"File {path} does not exist"
    norm = IVSurfaceNormalizer.load(path)
    
    rng = np.random.default_rng(42)
    iv = rng.uniform(0.05, 1.50, (10, 8, 11))
    iv_tensor = torch.tensor(iv, dtype=torch.float32)
    
    transformed = norm.transform_tensor(iv_tensor)
    reconstructed = norm.inverse_transform_tensor(transformed)
    
    torch.testing.assert_close(reconstructed, iv_tensor, atol=1e-5, rtol=1e-5)

def test_load_normalizers_v1_identity_fallback():
    orig_versions = calibrate._NORM_VERSIONS.copy()
    orig_param = calibrate._param_norm
    orig_iv = calibrate._iv_norm
    orig_param_path = calibrate._PARAM_NORM_PATH
    orig_iv_path = calibrate._IV_NORM_PATH
    
    try:
        calibrate._NORM_VERSIONS = {
            "v1": ("nonexistent_param.npz", "nonexistent_iv.npz")
        }
        calibrate._param_norm = None
        calibrate._iv_norm = None
        
        calibrate._load_normalizers(version='v1')
        
        assert isinstance(calibrate._param_norm, calibrate._IdentityParamNorm)
        assert isinstance(calibrate._iv_norm, calibrate._IdentityIVNorm)
        
    finally:
        calibrate._NORM_VERSIONS = orig_versions
        calibrate._param_norm = orig_param
        calibrate._iv_norm = orig_iv
        calibrate._PARAM_NORM_PATH = orig_param_path
        calibrate._IV_NORM_PATH = orig_iv_path

def test_load_normalizers_v2_non_identity():
    calibrate._param_norm = None
    calibrate._iv_norm = None
    
    calibrate._load_normalizers(version='v2')
    
    assert not isinstance(calibrate._param_norm, calibrate._IdentityParamNorm)
    assert not isinstance(calibrate._iv_norm, calibrate._IdentityIVNorm)
    assert isinstance(calibrate._param_norm, ParameterNormalizer)
    assert isinstance(calibrate._iv_norm, IVSurfaceNormalizer)

def test_version_switching():
    calibrate._param_norm = None
    calibrate._iv_norm = None
    
    calibrate._load_normalizers(version='v1')
    p_v1 = calibrate._param_norm
    iv_v1 = calibrate._iv_norm
    
    calibrate._load_normalizers(version='v2')
    p_v2 = calibrate._param_norm
    iv_v2 = calibrate._iv_norm
    
    assert p_v1 is not p_v2
    assert iv_v1 is not iv_v2

def test_parameter_normalizer_summary():
    path = "artifacts/models/param_normalizer_v2.npz"
    norm = ParameterNormalizer.load(path)
    s = norm.summary()
    assert isinstance(s, str)
    assert len(s) > 0

def test_iv_normalizer_summary():
    path = "artifacts/models/iv_normalizer_v2.npz"
    norm = IVSurfaceNormalizer.load(path)
    s = norm.summary()
    assert isinstance(s, str)
    assert len(s) > 0
