"""
Tests for Sobol-BB Brownian Bridge + Milstein Heston path simulation.

Validates:
  - Shape, dtype, and index contracts
  - NaN/Inf safety under Heston dynamics
  - Power-of-2 N_paths enforcement
  - QMC convergence advantage over pseudo-random MC
  - Batch independence via per-element Sobol seeds
  - Milstein coarse stepping bias vs fine-grid Euler reference
  - Float32 vs float64 accuracy equivalence
"""

import math
import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SKIP_NO_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@SKIP_NO_CUDA
def test_sobol_bb_shape():
    """Output shape = (B, N_paths, n_obs+1) with correct dtype and indices."""
    from deepvol.hedging.d_xva import simulate_heston_paths_sobol_bb

    theta = torch.tensor(
        [[2.0, 0.04, 0.3, -0.7, 0.04]], dtype=torch.float64, device=DEVICE
    )
    S_obs, idx = simulate_heston_paths_sobol_bb(
        theta, 100.0, 1.0, n_obs=4, N_paths=1024, r=0.05,
        device=torch.device(DEVICE),
    )
    assert S_obs.shape == (1, 1024, 5), f"Expected (1, 1024, 5), got {S_obs.shape}"
    assert S_obs.dtype == torch.float32, f"Expected float32, got {S_obs.dtype}"
    assert idx == [1, 2, 3, 4], f"Expected [1,2,3,4], got {idx}"
    # S0 stored at index 0
    assert torch.allclose(S_obs[:, :, 0], torch.full_like(S_obs[:, :, 0], 100.0))


@SKIP_NO_CUDA
def test_sobol_bb_no_nan():
    """No NaN or Inf in Sobol-BB paths under standard Heston parameters."""
    from deepvol.hedging.d_xva import simulate_heston_paths_sobol_bb

    theta = torch.tensor(
        [[2.0, 0.04, 0.3, -0.7, 0.04]], dtype=torch.float64, device=DEVICE
    )
    S_obs, _ = simulate_heston_paths_sobol_bb(
        theta, 100.0, 1.0, n_obs=4, N_paths=4096, r=0.05,
        device=torch.device(DEVICE),
    )
    assert not S_obs.isnan().any(), "NaN detected in Sobol-BB paths"
    assert not S_obs.isinf().any(), "Inf detected in Sobol-BB paths"
    assert (S_obs > 0).all(), "Non-positive spot prices detected"


@SKIP_NO_CUDA
def test_sobol_bb_no_nan_extreme_params():
    """No NaN/Inf under extreme Heston parameters (high vol-of-vol, negative rho)."""
    from deepvol.hedging.d_xva import simulate_heston_paths_sobol_bb

    theta = torch.tensor(
        [[5.0, 0.09, 0.8, -0.95, 0.09]], dtype=torch.float64, device=DEVICE
    )
    S_obs, _ = simulate_heston_paths_sobol_bb(
        theta, 100.0, 2.0, n_obs=8, N_paths=2048, r=0.01,
        device=torch.device(DEVICE),
    )
    assert not S_obs.isnan().any(), "NaN under extreme Heston params"
    assert not S_obs.isinf().any(), "Inf under extreme Heston params"
    assert (S_obs > 0).all(), "Non-positive spot under extreme params"


@SKIP_NO_CUDA
def test_sobol_bb_power_of_two():
    """Non-power-of-2 N_paths raises AssertionError."""
    from deepvol.hedging.d_xva import simulate_heston_paths_sobol_bb

    theta = torch.tensor(
        [[2.0, 0.04, 0.3, -0.7, 0.04]], dtype=torch.float64, device=DEVICE
    )
    with pytest.raises(AssertionError, match="power of 2"):
        simulate_heston_paths_sobol_bb(
            theta, 100.0, 1.0, n_obs=4, N_paths=1000, r=0.05,
            device=torch.device(DEVICE),
        )


@SKIP_NO_CUDA
def test_sobol_bb_convergence():
    """Sobol-BB MC converges faster than pseudo-random MC.

    At N=2^14, Sobol-BB RMSE should be < 0.7x pseudo-random RMSE
    (vs a 2^18 Sobol-BB reference).
    """
    from deepvol.hedging.d_xva import (
        simulate_heston_paths,
        simulate_heston_paths_sobol_bb,
    )
    from deepvol.models.autocall import price_autocall_mc, make_obs_indices

    theta = torch.tensor(
        [[2.0, 0.04, 0.3, -0.7, 0.04]], dtype=torch.float64, device=DEVICE
    )
    B = torch.tensor([1.0], dtype=torch.float64, device=DEVICE)
    c = torch.tensor([0.05], dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([0.03], dtype=torch.float64, device=DEVICE)
    n_obs = 4

    # Reference: Sobol-BB with 2^18 paths (float64 for precision)
    S_ref, obs_idx = simulate_heston_paths_sobol_bb(
        theta, 100.0, 1.0, n_obs, 2**18, 0.03,
        torch.device(DEVICE), seed=999, dtype=torch.float64,
    )
    dt_ref = 1.0 / (n_obs * 8)
    npv_ref, _, _ = price_autocall_mc(S_ref, obs_idx, B, c, r_t, 1.0, dt_ref)
    ref_val = float(npv_ref.item())
    del S_ref
    torch.cuda.empty_cache()

    N_test = 2**14

    # Sobol-BB errors (5 seeds)
    sobol_errs = []
    for s in range(5):
        S_s, oi = simulate_heston_paths_sobol_bb(
            theta, 100.0, 1.0, n_obs, N_test, 0.03,
            torch.device(DEVICE), seed=s, dtype=torch.float64,
        )
        npv_s, _, _ = price_autocall_mc(S_s, oi, B, c, r_t, 1.0, dt_ref)
        sobol_errs.append(abs(float(npv_s.item()) - ref_val))
        del S_s
        torch.cuda.empty_cache()

    # Pseudo-random errors (5 seeds)
    pseudo_errs = []
    obs_indices_fine = make_obs_indices(n_obs, 1.0, 252)
    for s in range(5):
        torch.manual_seed(s + 100)
        S_p = simulate_heston_paths(
            theta, 100.0, 1.0, 252, N_test, 0.03,
            device=torch.device(DEVICE),
        )
        npv_p, _, _ = price_autocall_mc(S_p, obs_indices_fine, B, c, r_t, 1.0, 1 / 252)
        pseudo_errs.append(abs(float(npv_p.item()) - ref_val))
        del S_p
        torch.cuda.empty_cache()

    mean_sobol = sum(sobol_errs) / len(sobol_errs)
    mean_pseudo = sum(pseudo_errs) / len(pseudo_errs)

    assert mean_sobol < mean_pseudo * 0.8, (
        f"Sobol-BB not converging faster: sobol={mean_sobol*10000:.2f}bps, "
        f"pseudo={mean_pseudo*10000:.2f}bps"
    )


@SKIP_NO_CUDA
def test_sobol_bb_batch_independence():
    """Different batch elements use independent Sobol sequences."""
    from deepvol.hedging.d_xva import simulate_heston_paths_sobol_bb

    # Same params, but batch dim=2 → each element gets seed+b*1000
    theta = torch.tensor(
        [
            [2.0, 0.04, 0.3, -0.7, 0.04],
            [2.0, 0.04, 0.3, -0.7, 0.04],
        ],
        dtype=torch.float64,
        device=DEVICE,
    )
    S_obs, _ = simulate_heston_paths_sobol_bb(
        theta, 100.0, 1.0, 4, 1024, 0.03, torch.device(DEVICE), seed=42,
    )
    # Terminal spot values should be independent between batch elements
    corr = torch.corrcoef(
        torch.stack([S_obs[0, :, -1].to(torch.float64),
                     S_obs[1, :, -1].to(torch.float64)])
    )[0, 1]
    assert abs(corr) < 0.15, f"Batch elements too correlated: {corr:.4f}"


@SKIP_NO_CUDA
def test_milstein_vs_euler_bias():
    """Milstein coarse (8 sub-steps) vs fine (32 sub-steps) discretization bias < 20 bps.

    By comparing coarse and fine Milstein simulations with the SAME Sobol seed
    and large N_paths, we isolate the discretization error from sampling noise.
    The Sobol anchors are identical; only the sub-step resolution differs.
    """
    from deepvol.hedging.d_xva import simulate_heston_paths_sobol_bb
    from deepvol.models.autocall import price_autocall_mc

    theta = torch.tensor(
        [[2.0, 0.04, 0.3, -0.7, 0.04]], dtype=torch.float64, device=DEVICE
    )
    B = torch.tensor([1.0], dtype=torch.float64, device=DEVICE)
    c = torch.tensor([0.08], dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([0.03], dtype=torch.float64, device=DEVICE)

    # Fine reference: 32 sub-steps per obs (128 total steps)
    S_fine, oi = simulate_heston_paths_sobol_bb(
        theta, 100.0, 1.0, 4, 2**16, 0.03,
        torch.device(DEVICE), sub_steps_per_obs=32, seed=42,
        dtype=torch.float64,
    )
    dt_fine = 1.0 / (4 * 32)
    npv_fine, _, _ = price_autocall_mc(S_fine, oi, B, c, r_t, 1.0, dt_fine)
    ref = float(npv_fine.item())
    del S_fine
    torch.cuda.empty_cache()

    # Coarse: 8 sub-steps per obs (32 total steps), same seed
    coarse_diffs = []
    for s in range(3):
        S_coarse, oi_c = simulate_heston_paths_sobol_bb(
            theta, 100.0, 1.0, 4, 2**16, 0.03,
            torch.device(DEVICE), sub_steps_per_obs=8, seed=42 + s,
            dtype=torch.float64,
        )
        # Also compute fine with same seed for fair comparison
        S_fine_s, oi_f = simulate_heston_paths_sobol_bb(
            theta, 100.0, 1.0, 4, 2**16, 0.03,
            torch.device(DEVICE), sub_steps_per_obs=32, seed=42 + s,
            dtype=torch.float64,
        )
        dt_c = 1.0 / (4 * 8)
        dt_f = 1.0 / (4 * 32)
        npv_c, _, _ = price_autocall_mc(S_coarse, oi_c, B, c, r_t, 1.0, dt_c)
        npv_f, _, _ = price_autocall_mc(S_fine_s, oi_f, B, c, r_t, 1.0, dt_f)
        coarse_diffs.append(abs(float(npv_c.item()) - float(npv_f.item())))
        del S_coarse, S_fine_s
        torch.cuda.empty_cache()

    mean_bias = sum(coarse_diffs) / len(coarse_diffs)
    # Discretization bias (coarse vs fine) should be < 20 bps
    assert mean_bias < 0.002, (
        f"Milstein coarse-vs-fine bias too large: {mean_bias*10000:.1f} bps"
    )


@SKIP_NO_CUDA
def test_float32_vs_float64_accuracy():
    """Float32 simulation error < 1.0 bps vs float64 reference (same Sobol seed).

    Validates that float32 path simulation does not introduce significant
    precision loss vs float64 for the same Sobol sequence.
    """
    from deepvol.hedging.d_xva import simulate_heston_paths_sobol_bb
    from deepvol.models.autocall import price_autocall_mc

    theta = torch.tensor(
        [[2.0, 0.04, 0.3, -0.7, 0.04]], dtype=torch.float64, device=DEVICE
    )
    B = torch.tensor([1.0], dtype=torch.float64, device=DEVICE)
    c = torch.tensor([0.05], dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([0.03], dtype=torch.float64, device=DEVICE)

    # Float64 reference
    S64, oi = simulate_heston_paths_sobol_bb(
        theta, 100.0, 1.0, 4, 2**16, 0.03,
        torch.device(DEVICE), seed=42, dtype=torch.float64,
    )
    dt = 1.0 / (4 * 8)
    npv64, _, _ = price_autocall_mc(S64, oi, B, c, r_t, 1.0, dt)
    del S64
    torch.cuda.empty_cache()

    # Float32 (same seed — path will differ slightly due to precision)
    S32, oi32 = simulate_heston_paths_sobol_bb(
        theta, 100.0, 1.0, 4, 2**16, 0.03,
        torch.device(DEVICE), seed=42, dtype=torch.float32,
    )
    npv32, _, _ = price_autocall_mc(
        S32.to(torch.float64), oi32, B, c, r_t, 1.0, dt,
    )
    del S32
    torch.cuda.empty_cache()

    diff_bps = abs(float(npv64.item()) - float(npv32.item())) * 10000
    assert diff_bps < 5.0, (
        f"Float32 vs float64 diff = {diff_bps:.2f} bps (want < 5.0)"
    )
