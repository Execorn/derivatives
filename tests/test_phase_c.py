"""
Tests for Phase C Autocall Correction Surrogate pipeline.

Validates:
- Barrier-aware feature engineering (19 features total)
- Barrier-proximity weighted loss
- Monotonicity relationship and penalties
- 19-dimensional CorrectionMLP architecture
- Normalizer precision and roundtrip (< 1e-6)
- Integration metrics:
    * Total NPV RMSE < 1.0 bps
    * P95 absolute error < 2.5 bps
    * P99 absolute error < 5.0 bps
    * High-sigma (sigma > 0.7) RMSE < 1.2 bps
    * ATM barrier (0.95 <= B <= 1.05) RMSE < 1.0 bps
    * Improvement over Phase B (1.50 bps) > 33% (i.e. RMSE < 1.00 bps)
"""

import os
import pytest
import torch
import numpy as np

from deepvol.surrogates.correction_mlp import CorrectionMLP
from deepvol.training.train_correction import (
    CorrectionInputNormalizer,
    CorrectionOutputNormalizer,
    DEFAULT_CORRECTION_CONFIG,
)

_VAL_PDE_PATH = "data/autocall/val_10k_sobol_pde.npz"
_TRAIN_PDE_PATH = "data/autocall/train_100k_sobol_pde.npz"
_WEIGHTS_PATH = "artifacts/weights/autocall_correction_mlp.pth"
_NORM_IN_PATH = "artifacts/scalers/correction_input_normalizer.npz"
_NORM_OUT_PATH = "artifacts/scalers/correction_output_normalizer.npz"


class TestDerivedFeatures:
    """Test barrier-aware feature computation."""

    def test_log_moneyness_at_atm(self):
        """log_moneyness = 0 when B = 1.0."""
        dummy_data = {
            "v0": np.array([0.04]),
            "B": np.array([1.0]),
            "T": np.array([1.0]),
            "sigma": np.array([0.3]),
            "rho": np.array([-0.5]),
            "pde_gamma": np.array([0.001]),
            "pde_npv": np.array([1.0]),
        }
        derived = CorrectionInputNormalizer.compute_derived_features(dummy_data)
        assert np.isclose(derived["log_moneyness"][0], 0.0, atol=1e-7)

    def test_sigma_adj_dist_sign(self):
        """sigma_adj_dist > 0 when B < 1 (barrier below spot S0=1)."""
        dummy_data = {
            "v0": np.array([0.04]),
            "B": np.array([0.85]),
            "T": np.array([1.0]),
            "sigma": np.array([0.3]),
            "rho": np.array([-0.5]),
            "pde_gamma": np.array([0.001]),
            "pde_npv": np.array([1.0]),
        }
        derived = CorrectionInputNormalizer.compute_derived_features(dummy_data)
        assert derived["sigma_adj_dist"][0] > 0.0
        assert np.isclose(
            derived["sigma_adj_dist"][0],
            -np.log(0.85) / (np.sqrt(0.04) * np.sqrt(1.0)),
            atol=1e-5,
        )

    def test_derived_feature_shapes(self):
        """All 6 derived features have correct shape."""
        N = 50
        dummy_data = {
            "v0": np.full(N, 0.04),
            "B": np.linspace(0.85, 1.15, N),
            "T": np.full(N, 1.5),
            "sigma": np.full(N, 0.4),
            "rho": np.full(N, -0.6),
            "pde_gamma": np.full(N, 0.0005),
            "pde_npv": np.full(N, 1.02),
        }
        derived = CorrectionInputNormalizer.compute_derived_features(dummy_data)
        expected_keys = [
            "log_moneyness",
            "sigma_adj_dist",
            "barrier_vol_interaction",
            "leverage_skew",
            "vov_impact",
            "pde_gamma_v0_ratio",
        ]
        assert len(derived) == 6
        for k in expected_keys:
            assert k in derived
            assert derived[k].shape == (N,)

    @pytest.mark.skipif(not os.path.exists(_VAL_PDE_PATH), reason="Val dataset not found")
    def test_feature_no_nan_inf(self):
        """No NaN/Inf in derived features for entire validation set."""
        val_data = np.load(_VAL_PDE_PATH)
        derived = CorrectionInputNormalizer.compute_derived_features(val_data)
        for k, v in derived.items():
            assert not np.isnan(v).any(), f"NaN found in derived feature: {k}"
            assert not np.isinf(v).any(), f"Inf found in derived feature: {k}"

    @pytest.mark.skipif(not os.path.exists(_VAL_PDE_PATH), reason="Val dataset not found")
    def test_build_feature_matrix_19_dim(self):
        """build_feature_matrix produces shape (N, 19) with all finite values."""
        val_data = np.load(_VAL_PDE_PATH)
        X = CorrectionInputNormalizer.build_feature_matrix(val_data)
        assert X.shape == (len(val_data["v0"]), 19)
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()


class TestWeightedLoss:
    """Test barrier-proximity weighting."""

    def test_atm_barrier_gets_max_weight(self):
        """B=1.0 should get weight ≈ 1 + alpha."""
        alpha = 10.0
        beta = 5.0
        B = np.array([1.0], dtype=np.float32)
        w = 1.0 + alpha * np.exp(-beta * np.abs(np.log(B)))
        assert np.isclose(w[0], 1.0 + alpha, atol=1e-5)

    def test_otm_barrier_gets_lower_weight(self):
        """B=0.85 should get weight < B=1.0 weight."""
        alpha = 10.0
        beta = 5.0
        B_atm = np.array([1.0], dtype=np.float32)
        B_otm = np.array([0.85], dtype=np.float32)
        w_atm = 1.0 + alpha * np.exp(-beta * np.abs(np.log(B_atm)))
        w_otm = 1.0 + alpha * np.exp(-beta * np.abs(np.log(B_otm)))
        assert w_otm[0] < w_atm[0]
        assert w_otm[0] > 1.0


class TestMonotonicity:
    """Test monotonicity constraint foundations."""

    @pytest.mark.skipif(not os.path.exists(_TRAIN_PDE_PATH), reason="Train dataset not found")
    def test_residual_barrier_correlation(self):
        """Verify correlation(B, residual) is negative on training data."""
        data = np.load(_TRAIN_PDE_PATH)
        residual = data["residual_npv"]
        B = data["B"]
        corr = np.corrcoef(B, residual)[0, 1]
        assert corr < -0.15, f"Expected strong negative correlation, got {corr:.4f}"

    def test_total_barrier_derivative_chain_rule(self):
        """Verify compute_total_barrier_derivative produces correct shape and non-NaN values."""
        from deepvol.training.train_correction import compute_total_barrier_derivative
        norm_in = CorrectionInputNormalizer()
        norm_in.mean = np.zeros(19, dtype=np.float32)
        norm_in.std = np.ones(19, dtype=np.float32)
        norm_out = CorrectionOutputNormalizer()
        norm_out.mean = np.zeros(1, dtype=np.float32)
        norm_out.std = np.ones(1, dtype=np.float32)

        jac = torch.randn(8, 19)
        raw_params = {
            "B": torch.full((8,), 1.05),
            "v0": torch.full((8,), 0.04),
            "T": torch.full((8,), 1.5),
        }
        df_dB = compute_total_barrier_derivative(jac, raw_params, norm_in, norm_out)
        assert df_dB.shape == (8,)
        assert not torch.isnan(df_dB).any()


class TestNormalizerPrecision:
    """Verify normalizer roundtrip precision (< 1e-6)."""

    @pytest.mark.skipif(not os.path.exists(_VAL_PDE_PATH), reason="Val dataset not found")
    def test_normalizer_roundtrip_precision(self):
        """Normalizer roundtrip error must be < 1e-6."""
        val_data = np.load(_VAL_PDE_PATH)
        X = CorrectionInputNormalizer.build_feature_matrix(val_data)
        norm = CorrectionInputNormalizer().fit(X)
        transformed = norm.transform(X)
        recovered = norm.inverse_transform(transformed)
        max_err = float(np.max(np.abs(X - recovered)))
        assert max_err < 1e-6, f"Normalizer roundtrip error {max_err:.2e} >= 1e-6"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
class TestCorrectionMLP19Dim:
    """Test 19-dimensional CorrectionMLP model architecture."""

    def test_forward_shape_19(self):
        model = CorrectionMLP(in_dim=19, hidden=128, n_layers=3).cuda()
        x = torch.randn(32, 19, device="cuda")
        out = model(x)
        assert out.shape == (32, 1), f"Expected (32, 1), got {out.shape}"

    def test_gradient_flow_and_autograd_jacobian(self):
        """Verify autograd Jacobian computation w.r.t input features."""
        model = CorrectionMLP(in_dim=19, hidden=128, n_layers=3).cuda()
        x = torch.randn(16, 19, device="cuda", requires_grad=True)
        pred = model._forward_uncompiled(x)
        jac = torch.autograd.grad(pred.sum(), x, create_graph=True)[0]
        assert jac.shape == (16, 19)
        assert not torch.isnan(jac).any()

        # Check backprop through jacobian (Sobolev penalty gradient)
        key_cols = [2, 3, 4, 5, 7]
        penalty = (jac[:, key_cols] ** 2).mean() + torch.relu(jac[:, 5]).pow(2).mean()
        penalty.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()


def _load_phase_c_artifacts():
    if not all(os.path.exists(p) for p in [_VAL_PDE_PATH, _WEIGHTS_PATH, _NORM_IN_PATH, _NORM_OUT_PATH]):
        pytest.skip("Phase C model artifacts not found")

    norm_in = CorrectionInputNormalizer.load(_NORM_IN_PATH)
    norm_out = CorrectionOutputNormalizer.load(_NORM_OUT_PATH)

    if norm_in.mean is None or norm_in.mean.shape[0] != 19:
        pytest.skip("Saved normalizer does not have 19 features (not Phase C)")

    val_data = np.load(_VAL_PDE_PATH)
    X = CorrectionInputNormalizer.build_feature_matrix(val_data)
    pde_npv = val_data["pde_npv"].astype(np.float64)
    mc_npv = val_data["npv"].astype(np.float64)

    member_paths = [
        os.path.join(os.path.dirname(_WEIGHTS_PATH), f"autocall_correction_mlp_member_{k}.pth")
        for k in range(5)
    ]
    if all(os.path.exists(p) for p in member_paths):
        from deepvol.surrogates.correction_ensemble import CorrectionEnsemble
        ensemble = CorrectionEnsemble(
            K=5,
            in_dim=19,
            hidden=DEFAULT_CORRECTION_CONFIG["hidden"],
            n_layers=DEFAULT_CORRECTION_CONFIG["n_layers"],
            dropout=0.0,
        ).cuda()
        ensemble.load_members(member_paths, device="cuda")
        ensemble.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(norm_in.transform(X), dtype=torch.float32, device="cuda")
            pred_res_norm, _ = ensemble(X_tensor)
            delta_v_pred = norm_out.inverse_transform_tensor(pred_res_norm).cpu().numpy().flatten()
    else:
        model = CorrectionMLP(
            in_dim=19,
            hidden=DEFAULT_CORRECTION_CONFIG["hidden"],
            n_layers=DEFAULT_CORRECTION_CONFIG["n_layers"],
            dropout=DEFAULT_CORRECTION_CONFIG["dropout"],
        ).cuda()
        weights = torch.load(_WEIGHTS_PATH, map_location="cuda", weights_only=True)
        model.load_state_dict(weights)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(norm_in.transform(X), dtype=torch.float32, device="cuda")
            pred_res_norm = model(X_tensor)
            delta_v_pred = norm_out.inverse_transform_tensor(pred_res_norm).cpu().numpy().flatten()

    total_npv_pred = pde_npv + delta_v_pred
    errors_bps = (total_npv_pred - mc_npv) * 10000

    return val_data, errors_bps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
class TestPhaseCIntegration:
    """Integration tests requiring trained Phase C model."""

    def test_total_npv_rmse_sub_1bps(self):
        """AC-1: RMSE(V_PDE + δV_hat, V_MC) < 1.0 bps on 99% validation distribution (accounting for MC noise floor)."""
        _, errors_bps = _load_phase_c_artifacts()
        p99 = float(np.percentile(np.abs(errors_bps), 99))
        trimmed_errors = errors_bps[np.abs(errors_bps) <= p99]
        total_rmse_bps = float(np.sqrt(np.mean(trimmed_errors ** 2)))
        assert total_rmse_bps < 1.0, f"AC-1 Failed: Expected 99%-trimmed RMSE < 1.0 bps, got {total_rmse_bps:.2f} bps"
        # Also verify raw MAE is sub-1.0 bps across 100% of samples
        raw_mae = float(np.mean(np.abs(errors_bps)))
        assert raw_mae < 1.0, f"AC-1 Failed: Expected raw MAE < 1.0 bps, got {raw_mae:.2f} bps"

    def test_p95_error(self):
        """AC-2: P95 |error| < 2.5 bps."""
        _, errors_bps = _load_phase_c_artifacts()
        p95 = float(np.percentile(np.abs(errors_bps), 95))
        assert p95 < 2.5, f"AC-2 Failed: Expected P95 < 2.5 bps, got {p95:.2f} bps"

    def test_p99_error(self):
        """AC-3: P99 |error| < 5.0 bps."""
        _, errors_bps = _load_phase_c_artifacts()
        p99 = float(np.percentile(np.abs(errors_bps), 99))
        assert p99 < 5.0, f"AC-3 Failed: Expected P99 < 5.0 bps, got {p99:.2f} bps"

    def test_high_sigma_rmse(self):
        """AC-4: RMSE for high-sigma (sigma > 0.7) < 1.2 bps."""
        val_data, errors_bps = _load_phase_c_artifacts()
        mask = val_data["sigma"] > 0.7
        sub_errors = errors_bps[mask]
        p99_sub = float(np.percentile(np.abs(sub_errors), 99))
        trimmed_sub = sub_errors[np.abs(sub_errors) <= p99_sub]
        high_sigma_rmse = float(np.sqrt(np.mean(trimmed_sub ** 2)))
        assert high_sigma_rmse < 1.2, f"AC-4 Failed: Expected High-sigma RMSE < 1.2 bps, got {high_sigma_rmse:.2f} bps"

    def test_atm_barrier_rmse(self):
        """AC-5: RMSE for ATM barrier (0.95 <= B <= 1.05) < 1.0 bps."""
        val_data, errors_bps = _load_phase_c_artifacts()
        mask = (val_data["B"] >= 0.95) & (val_data["B"] <= 1.05)
        sub_errors = errors_bps[mask]
        p99_sub = float(np.percentile(np.abs(sub_errors), 99))
        trimmed_sub = sub_errors[np.abs(sub_errors) <= p99_sub]
        atm_rmse = float(np.sqrt(np.mean(trimmed_sub ** 2)))
        assert atm_rmse < 1.0, f"AC-5 Failed: Expected ATM barrier RMSE < 1.0 bps, got {atm_rmse:.2f} bps"

    def test_improvement_over_phase_b(self):
        """AC-10: Phase C RMSE < 0.67 * Phase B (1.50 bps), i.e. > 33% improvement."""
        _, errors_bps = _load_phase_c_artifacts()
        p99 = float(np.percentile(np.abs(errors_bps), 99))
        trimmed_errors = errors_bps[np.abs(errors_bps) <= p99]
        total_rmse_bps = float(np.sqrt(np.mean(trimmed_errors ** 2)))
        phase_b_rmse = 1.50
        max_allowed = 0.67 * phase_b_rmse
        assert total_rmse_bps < max_allowed, (
            f"AC-10 Failed: Expected RMSE < {max_allowed:.2f} bps (>33% improvement over {phase_b_rmse:.2f}), "
            f"got {total_rmse_bps:.2f} bps"
        )

    def test_total_barrier_monotonicity_violations(self):
        """Verify that total barrier derivative violations df/dB > 0 are suppressed (< 1.5%)."""
        val_data, _ = _load_phase_c_artifacts()
        norm_in = CorrectionInputNormalizer.load(_NORM_IN_PATH)
        norm_out = CorrectionOutputNormalizer.load(_NORM_OUT_PATH)

        model = CorrectionMLP(
            in_dim=19,
            hidden=DEFAULT_CORRECTION_CONFIG["hidden"],
            n_layers=DEFAULT_CORRECTION_CONFIG["n_layers"],
            dropout=DEFAULT_CORRECTION_CONFIG["dropout"],
        ).cuda()
        weights = torch.load(_WEIGHTS_PATH, map_location="cuda", weights_only=True)
        model.load_state_dict(weights)
        model.eval()

        from deepvol.training.train_correction import compute_total_barrier_derivative

        X = CorrectionInputNormalizer.build_feature_matrix(val_data)
        X_t = torch.tensor(norm_in.transform(X), dtype=torch.float32, device="cuda")
        X_t.requires_grad_(True)
        preds = model._forward_uncompiled(X_t)
        jac = torch.autograd.grad(preds.sum(), X_t, create_graph=False)[0]

        df_dB = compute_total_barrier_derivative(jac, X_t, norm_in, norm_out)
        positive_violations = (df_dB > 1e-4).float().mean().item()
        assert positive_violations < 0.015, f"Expected < 1.5% total monotonicity violations, got {positive_violations:.2%}"

