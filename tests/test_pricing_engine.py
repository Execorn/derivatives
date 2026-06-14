import pytest
import torch
import numpy as np
from scipy.stats import qmc
from pricing_engine_gpu import price_batch_gpu

# Mark the whole module to skip if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)

def test_pricing_engine_nan_regression():
    # Canonical parameter: kappa=5.0, theta=0.08, sigma=0.8, rho=-0.7, v0=0.10
    params = np.array([[5.0, 0.08, 0.8, -0.7, 0.10]])
    
    ivs = price_batch_gpu(
        params, T_GRID, K_GRID,
        N_factors=40, N_cos=128, N_steps_per_unit=200,
        device='cuda'
    )
    
    # Verify that the NaN fraction across the 8x11 IV surface is <= 5%
    nan_fraction = np.isnan(ivs[0]).mean()
    assert nan_fraction <= 0.05

def test_pricing_engine_monotonicity():
    # Draw 10 Sobol samples from the parameter space
    sampler = qmc.Sobol(d=5, seed=42)
    u_samples = sampler.random(16)[:10]  # generate 16, keep first 10
    
    # Parameter bounds: kappa [0.1, 5.0], theta [0.01, 0.15], sigma [0.1, 1.0], rho [-0.9, -0.1], v0 [0.01, 0.15]
    LO = np.array([0.1, 0.01, 0.1, -0.9, 0.01])
    HI = np.array([5.0, 0.15, 1.0, -0.1, 0.15])
    samples = qmc.scale(u_samples, LO, HI)
    
    ivs = price_batch_gpu(samples, T_GRID, K_GRID, device='cuda')
    
    atm_idx = int(np.argmin(np.abs(K_GRID)))
    
    def is_monotonic(arr):
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) < 2:
            return True
        diffs = np.diff(arr_clean)
        return np.all(diffs >= 0) or np.all(diffs <= 0)

    monotonic_count = 0
    for i in range(10):
        atm_ivs = ivs[i, :, atm_idx]
        if is_monotonic(atm_ivs):
            monotonic_count += 1
            
    assert monotonic_count >= 8

def test_pricing_engine_batch_consistency():
    canonical_param = np.array([[5.0, 0.08, 0.8, -0.7, 0.10]])
    params_batch = np.repeat(canonical_param, 5, axis=0)
    
    ivs = price_batch_gpu(params_batch, T_GRID, K_GRID, device='cuda')
    
    for i in range(5):
        for j in range(i + 1, 5):
            np.testing.assert_equal(np.isnan(ivs[i]), np.isnan(ivs[j]))
            mask = ~np.isnan(ivs[i])
            if np.any(mask):
                max_diff = np.max(np.abs(ivs[i][mask] - ivs[j][mask]))
                assert max_diff < 1e-6


def test_pricing_engine_adaptive_n_cos():
    canonical_param = np.array([[3.0, 0.08, 0.5, -0.5, 0.08]])
    test_t_grid = np.array([0.04, 0.1, 0.3])
    
    # Call with adaptive N_cos_per_T dict
    N_cos_per_T = {0.04: 128, 0.1: 64, 0.3: 64}
    ivs_adaptive = price_batch_gpu(
        canonical_param, test_t_grid, K_GRID,
        N_cos=64, N_cos_per_T=N_cos_per_T,
        device='cuda'
    )
    
    # Compute manually for each sub-grid to compare
    iv_004 = price_batch_gpu(
        canonical_param, np.array([0.04]), K_GRID,
        N_cos=128, device='cuda'
    )
    iv_others = price_batch_gpu(
        canonical_param, np.array([0.1, 0.3]), K_GRID,
        N_cos=64, device='cuda'
    )
    
    # Verify shape
    assert ivs_adaptive.shape == (1, 3, len(K_GRID))
    
    # Verify that values match the manual runs
    np.testing.assert_allclose(ivs_adaptive[0, 0], iv_004[0, 0], rtol=1e-5, atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(ivs_adaptive[0, 1:], iv_others[0], rtol=1e-5, atol=1e-5, equal_nan=True)

