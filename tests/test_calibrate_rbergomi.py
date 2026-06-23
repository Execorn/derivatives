"""
test_calibrate_rbergomi.py — Test suite for Rough Bergomi model.
Verifies self-consistency, hybrid scheme vs naive Euler, and variance reduction.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from src.pricing.rbergomi_gpu import (
    simulate_rbergomi_paths,
    batch_rbergomi_iv_surface,
    rbergomi_iv_surface,
)


def test_simulate_rbergomi_self_consistency():
    """
    Test self-consistency and reproducibility of Rough Bergomi simulation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = torch.tensor([[0.04, 0.07, 1.5, -0.7]], device=device, dtype=torch.float32)
    T = 1.0
    steps_per_unit = 100
    N_paths = 1000

    # Test seed reproducibility
    torch.manual_seed(42)
    S1, V1, t1 = simulate_rbergomi_paths(
        params, T, steps_per_unit, N_paths, antithetic=True, device=device
    )

    torch.manual_seed(42)
    S2, V2, t2 = simulate_rbergomi_paths(
        params, T, steps_per_unit, N_paths, antithetic=True, device=device
    )

    assert torch.allclose(S1, S2, atol=1e-5)
    assert torch.allclose(V1, V2, atol=1e-5)
    assert torch.allclose(t1, t2, atol=1e-5)

    # Test basic properties
    assert S1.shape == (1, N_paths, steps_per_unit + 1)
    assert V1.shape == (1, N_paths, steps_per_unit + 1)
    assert (S1 > 0).all(), "Stock price should be strictly positive"
    assert (V1 > 0).all(), "Variance should be strictly positive"
    assert torch.allclose(S1[:, :, 0], torch.tensor(1.0, device=device)), "Initial stock price must be 1.0"


def test_hybrid_vs_naive_euler():
    """
    Compare Bennedsen hybrid scheme against a naive Euler scheme for fBm.
    Verifies that the hybrid scheme matches the theoretical variance t^(2H)
    much more closely than the naive Euler scheme, which underestimates it.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 1
    N_paths = 5000
    T = 1.0
    steps_per_unit = 200
    N_t = int(T * steps_per_unit)
    dt = 1.0 / steps_per_unit

    H = 0.05  # low H to amplify singularity effect
    params = torch.tensor([[0.04, H, 1.5, -0.7]], device=device, dtype=torch.float32)

    # 1. Simulate using Hybrid Scheme
    torch.manual_seed(123)
    _, _, _ = simulate_rbergomi_paths(
        params, T, steps_per_unit, N_paths, antithetic=False, device=device
    )

    # Let's extract Y_t from the hybrid scheme directly by duplicating the logic:
    H_val = torch.tensor([[H]], device=device, dtype=torch.float32)
    torch.manual_seed(123)
    Z1 = torch.randn(B, N_paths, N_t, device=device, dtype=torch.float32)
    Z2 = torch.randn(B, N_paths, N_t, device=device, dtype=torch.float32)

    # Hybrid convolution
    k_vec = torch.arange(1, N_t, device=device, dtype=torch.float32).unsqueeze(0)
    w = (dt ** H_val) * ((k_vec + 1) ** (H_val + 0.5) - k_vec ** (H_val + 0.5)) / (H_val + 0.5)
    zeros = torch.zeros(B, 1, device=device, dtype=torch.float32)
    w_full = torch.cat([zeros, w], dim=1)
    w_rev = torch.flip(w_full, dims=[1]).unsqueeze(1)

    Z1_reshaped = Z1.view(1, B * N_paths, N_t)
    Z1_padded = F.pad(Z1_reshaped, (N_t - 1, 0))
    w_rev_repeated = w_rev.repeat_interleave(N_paths, dim=0)
    conv_out = F.conv1d(Z1_padded, w_rev_repeated, groups=B * N_paths).view(B, N_paths, N_t)

    c1 = 1.0 / (H_val + 0.5)
    c2 = torch.sqrt(1.0 / (2.0 * H_val) - 1.0 / ((H_val + 0.5) ** 2))
    Y_hybrid = torch.sqrt(2.0 * H_val) * (conv_out + (dt ** H_val) * (c1 * Z1 + c2 * Z2))

    # 2. Simulate using Naive Euler Scheme
    # Y_naive_t_i = sqrt(2H) * sum_{j=1}^i dt^H * (i-j+1)^(H-0.5) * Z1_j
    k_naive = torch.arange(1, N_t + 1, device=device, dtype=torch.float32)
    w_naive = (dt ** H) * (k_naive ** (H - 0.5))
    w_naive_rev = torch.flip(w_naive, dims=[0]).view(1, 1, N_t)

    # Convolution for naive Euler
    w_naive_repeated = w_naive_rev.repeat_interleave(N_paths, dim=0)
    conv_naive = F.conv1d(Z1_padded, w_naive_repeated, groups=B * N_paths).view(B, N_paths, N_t)
    Y_naive = torch.sqrt(torch.tensor(2.0 * H, device=device)) * conv_naive

    # Measure variance at t = dt (index 0) and t = T (index N_t - 1)
    # Theoretical variance at t_i is t_i^(2H)
    t_grid = torch.arange(1, N_t + 1, device=device, dtype=torch.float32) * dt
    var_theory = t_grid ** (2.0 * H)

    var_hybrid = torch.var(Y_hybrid, dim=1).squeeze(0)  # (N_t,)
    var_naive = torch.var(Y_naive, dim=1).squeeze(0)    # (N_t,)

    # Error at t = dt (index 0)
    err_hybrid_t1 = torch.abs(var_hybrid[0] - var_theory[0]).item()
    err_naive_t1 = torch.abs(var_naive[0] - var_theory[0]).item()

    print(f"Variance at t=dt: Theory={var_theory[0].item():.6f}, Hybrid={var_hybrid[0].item():.6f}, Naive={var_naive[0].item():.6f}")
    assert err_hybrid_t1 < err_naive_t1, "Hybrid scheme should be more accurate than naive Euler near singularity"
    assert err_hybrid_t1 < 0.05, "Hybrid scheme variance error should be small"


def test_variance_reduction_antithetic():
    """
    Test that antithetic variables reduce option pricing estimator variance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = torch.tensor([[0.04, 0.07, 1.5, -0.7]], device=device, dtype=torch.float32)
    T = 1.0
    steps_per_unit = 100
    N_paths = 2000
    K = 1.0  # ATM strike

    n_runs = 10
    prices_std = []
    prices_anti = []

    # Run multiple times with different seeds
    for run in range(n_runs):
        torch.manual_seed(run)
        S_std, _, _ = simulate_rbergomi_paths(
            params, T, steps_per_unit, N_paths, antithetic=False, device=device
        )
        payoff_std = torch.clamp(S_std[:, :, -1] - K, min=0.0)
        prices_std.append(payoff_std.mean().item())

        torch.manual_seed(run)
        S_anti, _, _ = simulate_rbergomi_paths(
            params, T, steps_per_unit, N_paths, antithetic=True, device=device
        )
        payoff_anti = torch.clamp(S_anti[:, :, -1] - K, min=0.0)
        prices_anti.append(payoff_anti.mean().item())

    var_std = np.var(prices_std)
    var_anti = np.var(prices_anti)

    print(f"Standard MC Estimator Variance:   {var_std:.8e}")
    print(f"Antithetic MC Estimator Variance: {var_anti:.8e}")

    assert var_anti < var_std, "Antithetic variables should reduce estimator variance"


def test_calibrate_rbergomi_fast_self_consistency():
    """Verify that calibrate_rbergomi (Newton) recovers synthetic parameters."""
    from calibrate_fast import calibrate_rbergomi as calibrate_rbergomi_fast
    from fno_model import MirrorPaddedFNO2d
    from calibrate import _load_normalizers, _make_spatial_input, _fno_predict_real_iv
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MirrorPaddedFNO2d(param_dim=4).to(device)
    model.load_state_dict(torch.load("artifacts/weights/fno_rbergomi_final_prod.pth", map_location=device, weights_only=True))
    model.eval()
    
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.linspace(-0.5, 0.5, 11)
    
    v0_t = 0.04
    H_t = 0.07
    eta_t = 1.5
    rho_t = -0.7
    
    _load_normalizers("rbergomi")
    spatial = _make_spatial_input(T_grid, K_grid, device)
    
    raw_params = torch.tensor([[v0_t, H_t, eta_t, rho_t]], dtype=torch.float32, device=device)
    with torch.no_grad():
        target_iv_t = _fno_predict_real_iv(model, raw_params, spatial)
    target_iv = target_iv_t.cpu().numpy()
    
    res = calibrate_rbergomi_fast(model, target_iv, T_grid, K_grid, max_iter=25, n_starts=2)
    
    assert res["final_mse"] < 1e-4
    assert abs(res["v0"] - v0_t) < 0.015
    assert abs(res["H"] - H_t) < 0.02
    assert abs(res["eta"] - eta_t) < 0.20
    assert abs(res["rho"] - rho_t) < 0.15

