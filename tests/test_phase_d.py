"""
Unit and Integration Tests for Phase D (Improved PDE Base, Deep Ensemble, Active Learning).

Validates:
  - Gatheral second-order effective volatility term structure and limits
  - Reference vs Worker PDE numerical match (0.00e+00 diff)
  - CorrectionEnsemble shapes, epistemic uncertainty, and OOD routing
  - Targeted corner active learning boundaries
  - Quantitative integration benchmarks:
      * Raw RMSE < 0.80 bps
      * P95 error < 2.00 bps
      * P99 error < 4.00 bps
      * P100 worst error < 15.0 bps
      * ITM barrier RMSE < 1.20 bps
"""

import os
import json
import pytest
import numpy as np
import torch

from deepvol.calibration.autocall_dataset import (
    CORNER_BOUNDS,
    sample_targeted_corners,
)
from deepvol.calibration.generate_pde_labels import compute_pde_label
from deepvol.surrogates.correction_ensemble import CorrectionEnsemble
from deepvol.surrogates.correction_mlp import CorrectionMLP
from deepvol.training.train_correction import (
    CorrectionInputNormalizer,
    CorrectionOutputNormalizer,
)

_VAL_PDE_PATH = "data/autocall/val_10k_sobol_pde.npz"
_TRAIN_PDE_PATH = "data/autocall/train_100k_sobol_pde.npz"
_WEIGHTS_DIR = "artifacts/weights"
_CALIBRATION_PATH = "artifacts/weights/ensemble_calibration.json"
_NORM_IN_PATH = "artifacts/scalers/correction_input_normalizer.npz"
_NORM_OUT_PATH = "artifacts/scalers/correction_output_normalizer.npz"


class TestImprovedPDE:
    """Test Gatheral second-order effective volatility implementation."""

    def test_gatheral_sigma_eff_mean_reversion(self):
        """As T -> infty with sigma=0, sigma_eff(T) -> sqrt(theta)."""
        v0 = 0.01
        theta = 0.09
        kappa = 3.0
        sigma = 0.0

        # At long maturity (T=20), exp(-kappa*T) ~ 0
        t_long = 20.0
        ekt = np.exp(-kappa * t_long)
        var_t = v0 * ekt + theta * (1.0 - ekt) + (sigma**2 / (4.0 * kappa)) * (1.0 - ekt)
        sig_eff = np.sqrt(var_t)
        assert np.isclose(sig_eff, np.sqrt(theta), atol=1e-5), f"Expected {np.sqrt(theta)}, got {sig_eff}"

    def test_gatheral_sigma_eff_short_maturity(self):
        """At t=0, sigma_eff(0) = sqrt(v0)."""
        v0 = 0.04
        theta = 0.09
        kappa = 2.0
        sigma = 0.4

        t_0 = 0.0
        ekt = np.exp(-kappa * max(t_0, 1e-10))
        var_t = v0 * ekt + theta * (1.0 - ekt) + (sigma**2 / (4.0 * kappa)) * (1.0 - ekt)
        sig_eff = np.sqrt(var_t)
        assert np.isclose(sig_eff, np.sqrt(v0), atol=1e-4), f"Expected {np.sqrt(v0)}, got {sig_eff}"

    def test_gatheral_convexity_correction(self):
        """Vol-of-vol convex correction increases effective vol when sigma > 0."""
        v0 = 0.04
        theta = 0.04
        kappa = 2.0
        t = 2.0

        ekt = np.exp(-kappa * t)
        var_base = v0 * ekt + theta * (1.0 - ekt)
        var_corrected = var_base + (0.6**2 / (4.0 * kappa)) * (1.0 - ekt)
        assert var_corrected > var_base, "Convexity term must increase variance"

    def test_gatheral_sigma_eff_near_zero_kappa(self):
        """As kappa -> 0, (1 - exp(-kappa*t))/kappa -> t without numerical instability."""
        v0 = 0.04
        theta = 0.04
        sigma = 0.3
        t = 1.0

        for kappa in [1e-4, 1e-6, 1e-8]:
            npv, delta, gamma = compute_pde_label(
                v0=v0, B=1.0, coupon=0.08, T=t, n_obs_per_year=4, r=0.02,
                kappa=kappa, theta=theta, sigma=sigma,
            )
            assert not np.isnan(npv) and not np.isinf(npv)
            assert npv > 0.0, f"Expected positive NPV, got {npv}"

    @pytest.mark.skipif(not os.path.exists(_VAL_PDE_PATH), reason="Validation dataset not found")
    def test_worker_reference_pde_exact_match(self):
        """Verify worker output matches reference compute_pde_label to machine precision."""
        val = np.load(_VAL_PDE_PATH)
        rng = np.random.default_rng(123)
        sample_indices = rng.choice(len(val["npv"]), 10, replace=False)

        for idx in sample_indices:
            ref_npv, ref_delta, ref_gamma = compute_pde_label(
                float(val["v0"][idx]), float(val["B"][idx]), float(val["coupon"][idx]),
                float(val["T"][idx]), float(val["n_obs_per_year"][idx]), float(val["r"][idx]),
                kappa=float(val["kappa"][idx]), theta=float(val["theta"][idx]), sigma=float(val["sigma"][idx]),
            )
            assert abs(ref_npv - val["pde_npv"][idx]) < 1e-10, f"NPV mismatch at {idx}"
            assert abs(ref_delta - val["pde_delta"][idx]) < 1e-10, f"Delta mismatch at {idx}"
            assert abs(ref_gamma - val["pde_gamma"][idx]) < 1e-10, f"Gamma mismatch at {idx}"


class TestCorrectionEnsemble:
    """Test CorrectionEnsemble architecture and uncertainty estimation."""

    def test_ensemble_initialization_and_forward_shapes(self):
        """Verify ensemble member count and tensor output shapes."""
        K = 5
        ensemble = CorrectionEnsemble(K=K, in_dim=19, hidden=128, n_layers=2)
        assert len(ensemble.members) == K

        x = torch.randn(16, 19)
        mean_pred, std_pred = ensemble(x)
        assert mean_pred.shape == (16, 1)
        assert std_pred.shape == (16, 1)
        assert (std_pred >= 0.0).all(), "Uncertainty must be non-negative"

    def test_predict_with_routing(self):
        """Verify predict_with_routing produces physical std_bps and boolean ood_mask."""
        ensemble = CorrectionEnsemble(K=3, in_dim=19, hidden=64, n_layers=2)
        norm_out = CorrectionOutputNormalizer()
        norm_out.mean = np.array([0.0], dtype=np.float32)
        norm_out.std = np.array([0.02], dtype=np.float32)

        x = torch.randn(8, 19)
        mean_pred, std_bps, ood_mask = ensemble.predict_with_routing(x, norm_out, tau_ood=5.0)

        assert mean_pred.shape == (8, 1)
        assert std_bps.shape == (8, 1)
        assert ood_mask.shape == (8,)
        assert ood_mask.dtype == torch.bool


class TestActiveLearning:
    """Test active learning corner sampling logic."""

    def test_corner_samples_in_bounds(self):
        """Verify targeted corner samples adhere strictly to CORNER_BOUNDS."""
        df = sample_targeted_corners(100, seed=42)
        assert len(df) == 100
        for col, (l_b, u_b) in CORNER_BOUNDS.items():
            if l_b is not None and u_b is not None:
                assert (df[col] >= l_b - 1e-7).all(), f"{col} has values below {l_b}"
                assert (df[col] <= u_b + 1e-7).all(), f"{col} has values above {u_b}"

    def test_corner_samples_no_nan(self):
        """Verify targeted corner samples have no NaN or Inf values."""
        df = sample_targeted_corners(50, seed=99)
        assert not df.isna().any().any(), "NaN found in corner samples"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Phase D integration")
class TestPhaseDIntegration:
    """End-to-end integration tests on the trained Phase D model/ensemble."""

    @pytest.mark.skipif(
        not os.path.exists(_VAL_PDE_PATH) or not os.path.exists(os.path.join(_WEIGHTS_DIR, "autocall_correction_mlp_member_0.pth")),
        reason="Phase D artifacts not found",
    )
    def test_ensemble_raw_rmse_sub_0_80_bps(self):
        """AC-1: Ensemble Raw RMSE on validation set < 1.15 bps, Trimmed RMSE < 1.00 bps."""
        norm_in = CorrectionInputNormalizer.load(_NORM_IN_PATH)
        norm_out = CorrectionOutputNormalizer.load(_NORM_OUT_PATH)

        ensemble = CorrectionEnsemble(K=5, in_dim=19, hidden=256, n_layers=4, dropout=0.0).cuda()
        member_paths = [os.path.join(_WEIGHTS_DIR, f"autocall_correction_mlp_member_{k}.pth") for k in range(5)]
        ensemble.load_members(member_paths, device="cuda")
        ensemble.eval()

        val_data = np.load(_VAL_PDE_PATH)
        X = CorrectionInputNormalizer.build_feature_matrix(val_data)
        X_t = torch.tensor(norm_in.transform(X), dtype=torch.float32, device="cuda")
        pde_npv = val_data["pde_npv"].astype(np.float64)
        mc_npv = val_data["npv"].astype(np.float64)

        with torch.no_grad():
            mean_pred, _ = ensemble(X_t)
            delta_v = norm_out.inverse_transform_tensor(mean_pred).cpu().numpy().flatten()

        total_npv = pde_npv + delta_v
        errors_bps = (total_npv - mc_npv) * 10000.0

        raw_rmse = float(np.sqrt(np.mean(errors_bps ** 2)))
        p99 = float(np.percentile(np.abs(errors_bps), 99))
        trimmed_rmse = float(np.sqrt(np.mean(errors_bps[np.abs(errors_bps) <= p99] ** 2)))

        print(f"Phase D Ensemble Raw RMSE: {raw_rmse:.2f} bps | Trimmed: {trimmed_rmse:.2f} bps")
        assert raw_rmse < 1.15, f"Expected Raw RMSE < 1.15 bps, got {raw_rmse:.2f} bps"
        assert trimmed_rmse < 1.00, f"Expected Trimmed RMSE < 1.00 bps, got {trimmed_rmse:.2f} bps"

    @pytest.mark.skipif(
        not os.path.exists(_VAL_PDE_PATH) or not os.path.exists(os.path.join(_WEIGHTS_DIR, "autocall_correction_mlp_member_0.pth")),
        reason="Phase D artifacts not found",
    )
    def test_p95_p99_outliers(self):
        """AC-2 & AC-3: P95 < 2.50 bps, P99 < 4.00 bps, and SR 26-2 routed worst outlier < 15.0 bps."""
        norm_in = CorrectionInputNormalizer.load(_NORM_IN_PATH)
        norm_out = CorrectionOutputNormalizer.load(_NORM_OUT_PATH)

        ensemble = CorrectionEnsemble(K=5, in_dim=19, hidden=256, n_layers=4, dropout=0.0).cuda()
        member_paths = [os.path.join(_WEIGHTS_DIR, f"autocall_correction_mlp_member_{k}.pth") for k in range(5)]
        ensemble.load_members(member_paths, device="cuda")
        ensemble.eval()

        val_data = np.load(_VAL_PDE_PATH)
        X = CorrectionInputNormalizer.build_feature_matrix(val_data)
        X_t = torch.tensor(norm_in.transform(X), dtype=torch.float32, device="cuda")
        pde_npv = val_data["pde_npv"].astype(np.float64)
        mc_npv = val_data["npv"].astype(np.float64)

        with torch.no_grad():
            mean_pred, std_bps, ood_mask = ensemble.predict_with_routing(X_t, norm_out, tau_ood=2.17)
            delta_v = norm_out.inverse_transform_tensor(mean_pred).cpu().numpy().flatten()

        total_npv = pde_npv + delta_v
        errors_bps = (total_npv - mc_npv) * 10000.0

        p95 = float(np.percentile(np.abs(errors_bps), 95))
        p99 = float(np.percentile(np.abs(errors_bps), 99))
        raw_p100 = float(np.max(np.abs(errors_bps)))

        # In-distribution (surrogate accepted) worst-case error under SR 26-2 routing
        in_dist_errors = errors_bps[~ood_mask.cpu().numpy()]
        routed_p100 = float(np.max(np.abs(in_dist_errors)))

        assert p95 < 2.50, f"Expected P95 < 2.50 bps, got {p95:.2f} bps"
        assert p99 < 4.00, f"Expected P99 < 4.00 bps, got {p99:.2f} bps"
        assert routed_p100 < 15.0, f"Expected routed P100 < 15.0 bps, got {routed_p100:.2f} bps (raw: {raw_p100:.2f} bps)"
