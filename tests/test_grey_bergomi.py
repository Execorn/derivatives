import sys
import os
import pytest
import torch
import numpy as np
import scipy.special
import math

# Add deepvol C++ extension path to import path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cpp_path = os.path.join(project_root, "src/deepvol/cpp")
if cpp_path not in sys.path:
    sys.path.insert(0, cpp_path)

# Verify import
try:
    import deepvol_cuda
except ImportError as e:
    raise ImportError(f"Could not import deepvol_cuda. Is it compiled? Error: {e}")


def mittag_leffler_py(z, beta, max_iter=500, tol=1e-9):
    if z <= 0.0:
        return 1.0
    val_check = z ** (1.0 / beta)
    if val_check <= 35.0:
        s = 0.0
        log_z = math.log(z)
        for k in range(max_iter):
            val = k * log_z - math.lgamma(beta * k + 1.0)
            term = math.exp(val)
            s += term
            if term < tol * s and k > 0:
                break
        return s
    else:
        sum_terms = 0.0
        for j in range(1, 5):
            arg = 1.0 - beta * j
            gamma_val = scipy.special.gamma(arg)
            term = (z ** (-j)) / gamma_val
            sum_terms += term
        return (1.0 / beta) * math.exp(val_check) - sum_terms


def test_mittag_leffler_cuda():
    """
    Test Mittag-Leffler kernel correctness on CUDA against reference Python implementation.
    """
    assert torch.cuda.is_available(), "CUDA must be available for testing"
    
    # Test values spanning both power-series (z^(1/beta) <= 35) and asymptotic (z^(1/beta) > 35) regimes
    beta = 0.8
    z_vals = torch.tensor([0.0, 1.0, 5.0, 15.0, 25.0, 30.0, 40.0, 50.0, 100.0], dtype=torch.float64, device="cuda")
    
    out_cuda = deepvol_cuda.mittag_leffler_cuda(z_vals, beta, 500, 1e-9)
    out_cuda_cpu = out_cuda.cpu().numpy()
    
    for i, z in enumerate(z_vals.cpu().numpy()):
        expected = mittag_leffler_py(z, beta)
        actual = out_cuda_cpu[i]
        # Allow small relative error due to floating point differences
        assert np.allclose(actual, expected, rtol=1e-6), f"Mismatch at z={z}: expected={expected}, got={actual}"


def test_generate_grey_paths_cuda_shapes_and_device():
    """
    Test that generate_grey_paths_cuda runs on GPU and returns correct shapes, dtypes, and devices.
    """
    assert torch.cuda.is_available()
    
    params = torch.tensor([0.04, 0.1, 1.5, -0.7, 0.85], device="cuda")
    steps = 100
    paths = 2000
    T = 0.5
    dt = T / steps
    
    S, V, B_H = deepvol_cuda.generate_grey_paths_cuda(params, steps, paths, T, dt)
    
    # Shape checks
    assert S.shape == (paths, steps + 1)
    assert V.shape == (paths, steps + 1)
    assert B_H.shape == (paths, steps + 1)
    
    # Dtype and device checks (converted to float32 at output boundary)
    assert S.dtype == torch.float32
    assert V.dtype == torch.float32
    assert B_H.dtype == torch.float32
    
    assert S.is_cuda
    assert V.is_cuda
    assert B_H.is_cuda
    
    # Initial value checks
    assert torch.allclose(S[:, 0], torch.tensor(1.0, device="cuda"))
    assert torch.allclose(V[:, 0], torch.tensor(0.04, device="cuda"))
    assert torch.allclose(B_H[:, 0], torch.tensor(0.0, device="cuda"))
    
    # Finite check
    assert torch.isfinite(S).all()
    assert torch.isfinite(V).all()
    assert torch.isfinite(B_H).all()


def test_generate_grey_paths_cuda_batched():
    """
    Test batched generation (B > 1) of Grey Rough Bergomi paths.
    """
    assert torch.cuda.is_available()
    
    # B = 3
    params = torch.tensor([
        [0.04, 0.1, 1.5, -0.7, 0.85],
        [0.09, 0.15, 2.0, -0.5, 0.9],
        [0.06, 0.08, 1.2, -0.9, 0.8]
    ], device="cuda")
    
    steps = 50
    paths = 1000
    T = 0.25
    dt = T / steps
    
    S, V, B_H = deepvol_cuda.generate_grey_paths_cuda(params, steps, paths, T, dt)
    
    assert S.shape == (3, paths, steps + 1)
    assert V.shape == (3, paths, steps + 1)
    assert B_H.shape == (3, paths, steps + 1)
    
    assert S.dtype == torch.float32
    assert S.is_cuda
    
    # Initial value checks per batch element
    assert torch.allclose(S[:, :, 0], torch.tensor(1.0, device="cuda"))
    assert torch.allclose(V[0, :, 0], torch.tensor(0.04, device="cuda"))
    assert torch.allclose(V[1, :, 0], torch.tensor(0.09, device="cuda"))
    assert torch.allclose(V[2, :, 0], torch.tensor(0.06, device="cuda"))


def test_fbm_variance_and_martingale():
    """
    Verify mathematical properties:
    1. Stock price S_t has expectation close to 1.0 (martingale property).
    2. Fractional Brownian motion B_T^H has variance close to T^(2H) / Gamma(H + 0.5)^2.
    """
    assert torch.cuda.is_available()
    
    params = torch.tensor([0.04, 0.15, 1.0, -0.6, 0.95], device="cuda")
    steps = 200
    paths = 15000  # large number of paths for statistical convergence
    T = 1.0
    dt = T / steps
    
    S, V, B_H = deepvol_cuda.generate_grey_paths_cuda(params, steps, paths, T, dt)
    
    # Martingale property check: E[S_T] should be close to 1.0
    s_mean = S[:, -1].mean().item()
    print(f"Empirical E[S_T]: {s_mean}")
    # Allow some statistical deviation
    assert abs(s_mean - 1.0) < 0.05, f"Martingale property violated: E[S_T]={s_mean}"
    
    # fBm variance check: Var(B_T^H) = T^(2H) / Gamma(H + 0.5)^2
    H = 0.15
    gamma_val = scipy.special.gamma(H + 0.5)
    expected_var = (T ** (2 * H)) / (gamma_val ** 2)
    
    actual_var = B_H[:, -1].var().item()
    print(f"Expected fBm Var: {expected_var}, Actual: {actual_var}")
    # Check within 5% relative tolerance
    assert abs(actual_var - expected_var) / expected_var < 0.05
