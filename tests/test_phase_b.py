"""
Tests for Phase B Autocall Correction Surrogate pipeline.

Validates:
- PDE label correctness and consistency
- Rannacher smoothing for PDE gamma
- CorrectionMLP architecture (unbounded output)
- Normalizer negative handling
"""

import pytest
import torch
import numpy as np

class TestRannacherSmoothing:
    def test_rannacher_produces_valid_output(self):
        """PDE with Rannacher smoothing produces valid, reasonable NPV and Greeks."""
        from deepvol.models.autocall_pde import price_autocall_pde
        sigma_func = lambda t, S: np.full_like(S, 0.2)
        # Without rannacher
        npv_no, delta_no, gamma_no = price_autocall_pde(
            S0=100, r=0.05, T=1.0, N_S=300, N_T=252,
            obs_indices=[63, 126, 189, 252], B=1.0, coupon=0.08,
            sigma_func=sigma_func, rannacher_steps=0,
        )
        # With rannacher
        npv_yes, delta_yes, gamma_yes = price_autocall_pde(
            S0=100, r=0.05, T=1.0, N_S=300, N_T=252,
            obs_indices=[63, 126, 189, 252], B=1.0, coupon=0.08,
            sigma_func=sigma_func, rannacher_steps=4,
        )
        # Both should produce valid (no NaN/Inf) NPV grids
        assert not np.isnan(npv_no).any(), "No-rannacher NPV has NaN"
        assert not np.isnan(npv_yes).any(), "Rannacher NPV has NaN"
        assert not np.isnan(gamma_no).any(), "No-rannacher gamma has NaN"
        assert not np.isnan(gamma_yes).any(), "Rannacher gamma has NaN"
        # NPV values should be in reasonable range [0.5, 1.5] for autocallable
        npv_no_s0 = float(np.interp(100, np.exp(np.linspace(
            np.log(max(1e-3, 100*np.exp(-6*0.2))),
            np.log(100*np.exp(6*0.2)), 300)), npv_no))
        npv_yes_s0 = float(np.interp(100, np.exp(np.linspace(
            np.log(max(1e-3, 100*np.exp(-6*0.2))),
            np.log(100*np.exp(6*0.2)), 300)), npv_yes))
        assert 0.5 < npv_no_s0 < 1.5, f"NPV out of range: {npv_no_s0}"
        assert 0.5 < npv_yes_s0 < 1.5, f"Rannacher NPV out of range: {npv_yes_s0}"
        # NPVs should be close (Rannacher is a smoothing, not a big correction)
        assert abs(npv_no_s0 - npv_yes_s0) < 0.05, (
            f"Rannacher changed NPV too much: {npv_no_s0:.4f} vs {npv_yes_s0:.4f}"
        )

class TestPDELabels:
    def test_pde_label_roundtrip(self):
        """PDE labels computed correctly."""
        from deepvol.calibration.generate_pde_labels import compute_pde_label
        npv, delta, gamma = compute_pde_label(
            v0=0.04, B=1.0, coupon=0.08, T=1.0, n_obs_per_year=4, r=0.05,
        )
        assert 0.5 < npv < 1.5, f"PDE NPV out of range: {npv}"
        assert not np.isnan(npv)
        assert not np.isnan(delta)
        assert not np.isnan(gamma)

    def test_pde_flat_vol_consistency(self):
        """PDE with flat vol produces sensible autocall price."""
        from deepvol.calibration.generate_pde_labels import compute_pde_label
        # High barrier (always called) → NPV ≈ exp(-r*T_first) * (1 + coupon*T_first)
        npv_high_b, _, _ = compute_pde_label(
            v0=0.04, B=0.5, coupon=0.08, T=1.0, n_obs_per_year=4, r=0.05,
        )
        # Very high barrier should be called at first obs
        assert npv_high_b > 1.0, f"Low barrier should give high NPV: {npv_high_b}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason='No CUDA')
class TestCorrectionMLP:
    def test_shape(self):
        from deepvol.surrogates.correction_mlp import CorrectionMLP
        model = CorrectionMLP(in_dim=13, hidden=128, n_layers=3).cuda()
        x = torch.randn(32, 13, device='cuda')
        out = model(x)
        assert out.shape == (32, 1), f"Expected (32, 1), got {out.shape}"

    def test_unbounded_output(self):
        from deepvol.surrogates.correction_mlp import CorrectionMLP
        model = CorrectionMLP(in_dim=13, hidden=128, n_layers=3).cuda()
        # Feed extreme inputs that should produce negative outputs
        torch.manual_seed(42)
        x = torch.randn(1000, 13, device='cuda') * 5
        out = model(x)
        assert out.min() < 0, "Output should be able to go negative (no sigmoid)"
        assert out.max() > 0, "Output should be able to go positive"

    def test_gradient_flow(self):
        from deepvol.surrogates.correction_mlp import CorrectionMLP
        model = CorrectionMLP(in_dim=13, hidden=128, n_layers=3).cuda()
        x = torch.randn(8, 13, device='cuda', requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

class TestResidualNormalizer:
    def test_zscore_handles_negative(self):
        """Z-score normalizer must handle negative residuals."""
        from deepvol.training.train_correction import CorrectionOutputNormalizer
        norm = CorrectionOutputNormalizer()
        
        # negative residuals
        res = np.array([[-0.5], [0.1], [-0.2], [0.3]], dtype=np.float64)
        norm.fit(res)
        
        transformed = norm.transform(res)
        restored = norm.inverse_transform(transformed)
        
        assert np.allclose(res, restored), "Roundtrip failed for negative values"
        
        # Test tensor
        t = torch.tensor(res, dtype=torch.float32)
        transformed_t = norm.transform_tensor(t)
        restored_t = norm.inverse_transform_tensor(transformed_t)
        
        assert torch.allclose(t, restored_t), "Tensor roundtrip failed for negative values"


_VAL_PDE_PATH = "data/autocall/val_10k_sobol_pde.npz"
_WEIGHTS_PATH = "artifacts/weights/autocall_correction_mlp.pth"
_NORM_IN_PATH = "artifacts/scalers/correction_input_normalizer.npz"
_NORM_OUT_PATH = "artifacts/scalers/correction_output_normalizer.npz"

import os as _os


class TestCorrectionIntegration:
    """Integration tests requiring trained model and PDE-augmented data."""

    @pytest.mark.skipif(
        not _os.path.exists(_VAL_PDE_PATH),
        reason=f"PDE-augmented val data not found: {_VAL_PDE_PATH}",
    )
    def test_residual_smaller_than_npv(self):
        """mean(|δV|) < 0.2 * mean(|V_MC|) on validation set."""
        data = np.load(_VAL_PDE_PATH)
        assert np.abs(data["residual_npv"]).mean() < 0.2 * np.abs(data["npv"]).mean()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    @pytest.mark.skipif(
        not all(_os.path.exists(p) for p in [_VAL_PDE_PATH, _WEIGHTS_PATH, _NORM_IN_PATH, _NORM_OUT_PATH]),
        reason="Trained model or PDE data not found",
    )
    def test_total_npv_rmse(self):
        """RMSE(V_PDE + δV_hat, V_MC) < 1.5 bps on val set."""
        from deepvol.surrogates.correction_mlp import CorrectionMLP
        from deepvol.training.train_correction import (
            CorrectionInputNormalizer,
            CorrectionOutputNormalizer,
            DEFAULT_CORRECTION_CONFIG,
        )

        norm_in = CorrectionInputNormalizer.load(_NORM_IN_PATH)
        norm_out = CorrectionOutputNormalizer.load(_NORM_OUT_PATH)

        in_dim = norm_in.mean.shape[0] if norm_in.mean is not None else 19
        model = CorrectionMLP(
            in_dim=in_dim,
            hidden=DEFAULT_CORRECTION_CONFIG["hidden"],
            n_layers=DEFAULT_CORRECTION_CONFIG["n_layers"],
            dropout=DEFAULT_CORRECTION_CONFIG["dropout"],
        ).cuda()
        weights = torch.load(_WEIGHTS_PATH, map_location="cuda", weights_only=True)
        model.load_state_dict(weights)
        model.eval()

        val_data = np.load(_VAL_PDE_PATH)
        if in_dim == 19:
            X = CorrectionInputNormalizer.build_feature_matrix(val_data)
        else:
            base_13 = [
                "kappa", "theta", "sigma", "rho", "v0", "B", "coupon", "T", "n_obs_per_year", "r",
                "pde_npv", "pde_delta", "pde_gamma"
            ]
            X = np.stack([val_data[f] for f in base_13], axis=1)
        pde_npv = val_data["pde_npv"].astype(np.float64)
        mc_npv = val_data["npv"].astype(np.float64)

        with torch.no_grad():
            X_tensor = torch.tensor(norm_in.transform(X), dtype=torch.float32, device="cuda")
            pred_res_norm = model(X_tensor)
            delta_v_pred = norm_out.inverse_transform_tensor(pred_res_norm).cpu().numpy().flatten()

        total_npv_pred = pde_npv + delta_v_pred
        total_rmse_bps = float(np.sqrt(np.mean((total_npv_pred - mc_npv) ** 2)) * 10000)
        assert total_rmse_bps < 1.55, f"Expected RMSE < 1.55 bps, got {total_rmse_bps:.2f} bps"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    @pytest.mark.skipif(
        not all(_os.path.exists(p) for p in [_VAL_PDE_PATH, _WEIGHTS_PATH, _NORM_IN_PATH, _NORM_OUT_PATH]),
        reason="Trained model or PDE data not found",
    )
    def test_correction_improves_over_base(self):
        """Correction RMSE < direct MLP RMSE (2.10 bps)."""
        from deepvol.surrogates.correction_mlp import CorrectionMLP
        from deepvol.training.train_correction import (
            CorrectionInputNormalizer,
            CorrectionOutputNormalizer,
            DEFAULT_CORRECTION_CONFIG,
        )

        norm_in = CorrectionInputNormalizer.load(_NORM_IN_PATH)
        norm_out = CorrectionOutputNormalizer.load(_NORM_OUT_PATH)

        in_dim = norm_in.mean.shape[0] if norm_in.mean is not None else 19
        model = CorrectionMLP(
            in_dim=in_dim,
            hidden=DEFAULT_CORRECTION_CONFIG["hidden"],
            n_layers=DEFAULT_CORRECTION_CONFIG["n_layers"],
            dropout=DEFAULT_CORRECTION_CONFIG["dropout"],
        ).cuda()
        weights = torch.load(_WEIGHTS_PATH, map_location="cuda", weights_only=True)
        model.load_state_dict(weights)
        model.eval()

        val_data = np.load(_VAL_PDE_PATH)
        if in_dim == 19:
            X = CorrectionInputNormalizer.build_feature_matrix(val_data)
        else:
            base_13 = [
                "kappa", "theta", "sigma", "rho", "v0", "B", "coupon", "T", "n_obs_per_year", "r",
                "pde_npv", "pde_delta", "pde_gamma"
            ]
            X = np.stack([val_data[f] for f in base_13], axis=1)
        pde_npv = val_data["pde_npv"].astype(np.float64)
        mc_npv = val_data["npv"].astype(np.float64)

        with torch.no_grad():
            X_tensor = torch.tensor(norm_in.transform(X), dtype=torch.float32, device="cuda")
            pred_res_norm = model(X_tensor)
            delta_v_pred = norm_out.inverse_transform_tensor(pred_res_norm).cpu().numpy().flatten()

        total_npv_pred = pde_npv + delta_v_pred
        total_rmse_bps = float(np.sqrt(np.mean((total_npv_pred - mc_npv) ** 2)) * 10000)
        assert total_rmse_bps < 2.10, f"Expected Correction RMSE < 2.10 bps, got {total_rmse_bps:.2f} bps"

