"""
test_autocall_guardian.py — Tests for Autocall 3-Tier Model Risk Guardian.

Validates:
  - Tier 1: Input OOD filter gate (Feller singularity, integrated vol distortion, corner mask)
  - Tier 2: Output arbitrage gate (barrier monotonicity df/dB > 0, magnitude clamp, price bounds)
  - Tier 3: Operational fallback execution, SR 26-2 compliance audit logging, PSI tracking
  - Interception rate on 100 Phase C corner outlier contracts
"""

import os
import pytest
import numpy as np
import torch

from deepvol.mrm.autocall_guardian import AutocallModelGuardian
from deepvol.training.train_correction import (
    CorrectionInputNormalizer,
    CorrectionOutputNormalizer,
)

_VAL_PDE_PATH = "data/autocall/val_10k_sobol_pde.npz"


class TestAutocallGuardianTier1:
    """Test Tier 1: Pre-inference Input OOD Filter Gate."""

    def test_feller_violation_triggers_ood(self):
        guardian = AutocallModelGuardian(feller_threshold=0.20)
        # Feller ratio = 2 * 0.5 * 0.02 / (0.8^2) = 0.02 / 0.64 = 0.03125 < 0.20
        bad_params = {
            "kappa": 0.5, "theta": 0.02, "sigma": 0.8,
            "rho": -0.5, "v0": 0.04, "T": 1.0, "B": 1.0,
        }
        res = guardian.check_input_ood(bad_params)
        assert res["is_ood"] is True
        assert any("Feller" in r for r in res["reasons"])

    def test_integrated_vol_distortion_triggers_ood(self):
        guardian = AutocallModelGuardian(integrated_vol_threshold=0.40)
        # Long maturity with v0 << theta: v0=0.01, theta=0.08, T=3.0, kappa=2.0
        distorted_params = {
            "kappa": 2.0, "theta": 0.08, "sigma": 0.3,
            "rho": -0.5, "v0": 0.01, "T": 3.0, "B": 1.0,
        }
        res = guardian.check_input_ood(distorted_params)
        assert res["is_ood"] is True
        assert any("Integrated volatility" in r for r in res["reasons"])

    def test_normal_params_pass_tier_1(self):
        guardian = AutocallModelGuardian()
        normal_params = {
            "kappa": 2.0, "theta": 0.04, "sigma": 0.3,
            "rho": -0.6, "v0": 0.04, "T": 1.0, "B": 1.0,
        }
        res = guardian.check_input_ood(normal_params)
        assert res["is_ood"] is False
        assert len(res["reasons"]) == 0

    def test_compound_corner_triggers_ood(self):
        guardian = AutocallModelGuardian()
        corner_params = {
            "kappa": 2.0, "theta": 0.04, "sigma": 0.7,
            "rho": -0.85, "v0": 0.015, "T": 3.0, "B": 1.10,
        }
        res = guardian.check_input_ood(corner_params)
        assert res["is_ood"] is True
        assert any("extrapolation corner" in r.lower() for r in res["reasons"])


class TestAutocallGuardianTier2:
    """Test Tier 2: Post-inference Output Arbitrage & Plausibility Gate."""

    def test_barrier_monotonicity_violation_triggers_arbitrage(self):
        guardian = AutocallModelGuardian()
        params = {"T": 1.0, "coupon": 0.08, "S0": 1.0}
        # Artificial positive derivative df/dB = +0.02
        res = guardian.check_output_arbitrage(params, pde_npv=1.0, delta_v=-0.05, df_dB=0.02)
        assert res["is_arbitrage"] is True
        assert any("monotonicity" in r.lower() for r in res["reasons"])

    def test_valid_monotonicity_passes_tier_2(self):
        guardian = AutocallModelGuardian()
        params = {"T": 1.0, "coupon": 0.08, "S0": 1.0}
        # Negative derivative df/dB = -0.05 <= 0
        res = guardian.check_output_arbitrage(params, pde_npv=1.0, delta_v=-0.05, df_dB=-0.05)
        assert res["is_arbitrage"] is False

    def test_price_bounds_violation_triggers_arbitrage(self):
        guardian = AutocallModelGuardian()
        params = {"T": 1.0, "coupon": 0.08, "S0": 1.0}
        # Negative price
        res = guardian.check_output_arbitrage(params, pde_npv=0.1, delta_v=-0.2)
        assert res["is_arbitrage"] is True
        assert any("boundary" in r.lower() for r in res["reasons"])

    def test_magnitude_clamping_with_normalizer(self):
        norm_out = CorrectionOutputNormalizer()
        norm_out.mean = np.array([0.0], dtype=np.float32)
        norm_out.std = np.array([0.02], dtype=np.float32)  # 200 bps std
        guardian = AutocallModelGuardian(norm_out=norm_out, residual_clamp_std=3.0)

        params = {"T": 1.0, "coupon": 0.08, "S0": 1.0}
        # |delta_v| = 0.08 > 3 * 0.02 = 0.06
        res = guardian.check_output_arbitrage(params, pde_npv=1.0, delta_v=0.08, df_dB=-0.01)
        assert res["is_arbitrage"] is True
        assert any("magnitude" in r.lower() for r in res["reasons"])


class TestAutocallGuardianTier3AndGovernance:
    """Test Tier 3: Operational Fallback Routing, SR 26-2 Logging, and PSI Tracking."""

    def test_fallback_execution_and_latency(self):
        guardian = AutocallModelGuardian()
        bad_params = {
            "kappa": 0.2, "theta": 0.01, "sigma": 0.9,
            "rho": -0.9, "v0": 0.01, "T": 1.0, "B": 1.0,
            "r": 0.02, "coupon": 0.08, "n_obs_per_year": 4, "pde_npv": 1.0,
        }
        res = guardian.predict_with_guardian(bad_params)
        assert res["is_fallback"] is True
        assert res["trigger"] == "Tier_1_Input_OOD"
        assert len(guardian.audit_log) == 1
        assert "fallback_npv" in guardian.audit_log[0]
        assert guardian.audit_log[0]["fallback_latency_ms"] < 20.0  # sub-20 ms latency

    def test_psi_tracking(self):
        guardian = AutocallModelGuardian()
        # Reference: N(0, 1)
        np.random.seed(42)
        ref_data = np.random.randn(1000, 19).astype(np.float32)
        guardian.set_reference_distribution(ref_data)

        # Same distribution batch: PSI should be very small (< 0.05)
        same_batch = np.random.randn(500, 19).astype(np.float32)
        psi_same = guardian.compute_psi(same_batch)
        assert psi_same < 0.05, f"Expected small PSI for same distribution, got {psi_same:.4f}"

        # Drastically shifted batch: PSI should be large (>= 0.25)
        shifted_batch = np.random.randn(500, 19).astype(np.float32) + 2.0
        psi_shifted = guardian.compute_psi(shifted_batch)
        assert psi_shifted >= 0.25, f"Expected large PSI for shifted distribution, got {psi_shifted:.4f}"

    @pytest.mark.skipif(not os.path.exists(_VAL_PDE_PATH), reason="Val dataset not found")
    def test_intercepts_corner_outlier_contracts(self):
        """Verify Guardian intercepts outlier corner contracts from validation set."""
        guardian = AutocallModelGuardian()
        val_data = np.load(_VAL_PDE_PATH)

        # Identify extreme corner contracts (B > 1.05, T >= 2.5, v0 <= 0.03, sigma >= 0.6)
        mask = (
            (val_data["B"] > 1.05)
            & (val_data["T"] >= 2.5)
            & (val_data["v0"] <= 0.03)
            & (val_data["sigma"] >= 0.6)
        )
        indices = np.where(mask)[0]
        assert len(indices) > 0, "No corner contracts found in val set"

        intercepted = 0
        for idx in indices:
            contract_params = {k: float(val_data[k][idx]) for k in ["kappa", "theta", "sigma", "rho", "v0", "B", "T", "r", "coupon"]}
            ood = guardian.check_input_ood(contract_params)
            if ood["is_ood"]:
                intercepted += 1

        interception_rate = intercepted / len(indices)
        assert interception_rate == 1.0, f"Expected 100% corner interception, got {interception_rate:.1%}"
