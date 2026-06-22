"""
tests/test_calibrate_newton_h.py — Tests for §5.1 Learnable Hurst calibration.

Tests the 4D reparameterised Newton-Raphson calibrator (v0, zeta, lam, H).
The self-consistency tests use the v2 model (H=0.08 fixed in training) to
verify that calibrate_newton_h recovers the correct H when H_target=0.08.
"""

import os, sys
import numpy as np
import torch
import pytest

CUDA_OK = torch.cuda.is_available()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from calibrate_fast import (
    _reparam_to_6d_with_H,
    _BOUNDS_LOWER_4D,
    _BOUNDS_UPPER_4D,
    calibrate_newton_h,
)

T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)


# ─── _reparam_to_6d_with_H ────────────────────────────────────────────────────

class TestReparamWithH:
    """Unit tests for the 4D → 6D back-transform."""

    CASES = [
        (0.07, -0.30, 0.40, 0.08),
        (0.05, -0.20, 0.30, 0.06),
        (0.10, -0.40, 0.50, 0.10),
        (0.12, -0.10, 0.80, 0.12),
        (0.03, -0.70, 0.20, 0.05),
    ]

    def _run(self, v0, zeta, lam, H):
        v0_t   = torch.tensor([v0])
        zeta_t = torch.tensor([zeta])
        lam_t  = torch.tensor([lam])
        H_t    = torch.tensor([H])
        p6 = _reparam_to_6d_with_H(v0_t, zeta_t, lam_t, H_t, device='cpu')
        return p6[0]   # (6,)

    def test_sigma_is_sqrt_zeta2_plus_lam2(self):
        for v0, zeta, lam, H in self.CASES:
            p6 = self._run(v0, zeta, lam, H)
            sigma_expected = max(np.sqrt(zeta**2 + lam**2), 0.01)
            assert abs(float(p6[2]) - sigma_expected) < 1e-5, \
                f"sigma mismatch: got {float(p6[2]):.6f}, expected {sigma_expected:.6f}"

    def test_rho_is_clamped(self):
        # rho = zeta / sigma, should be clamped to [-0.9, -0.1]
        for v0, zeta, lam, H in self.CASES:
            p6 = self._run(v0, zeta, lam, H)
            rho = float(p6[3])
            assert -0.9 <= rho <= -0.1, f"rho={rho} outside clamp"

    def test_H_is_clamped(self):
        # H should be clamped to [0.04, 0.15] — allow float32 rounding (±1e-6)
        for H_raw in [0.0, 0.01, 0.03, 0.16, 0.5]:
            p6 = self._run(0.07, -0.30, 0.40, H_raw)
            H_out = float(p6[5])
            assert H_out >= 0.04 - 1e-6, f"H_out={H_out} below lower bound for H_raw={H_raw}"
            assert H_out <= 0.15 + 1e-6, f"H_out={H_out} above upper bound for H_raw={H_raw}"

    def test_kappa_and_theta_are_fixed(self):
        for v0, zeta, lam, H in self.CASES:
            p6 = self._run(v0, zeta, lam, H)
            assert abs(float(p6[0]) - 1.0) < 1e-6, "kappa should be 1.0"
            assert abs(float(p6[1]) - 0.08) < 1e-6, "theta should be 0.08"

    def test_v0_is_preserved(self):
        for v0, zeta, lam, H in self.CASES:
            p6 = self._run(v0, zeta, lam, H)
            assert abs(float(p6[4]) - v0) < 1e-5, f"v0 mismatch"

    def test_H_is_preserved_when_in_range(self):
        for _, _, _, H in self.CASES:
            if 0.04 <= H <= 0.15:
                p6 = self._run(0.07, -0.30, 0.40, H)
                assert abs(float(p6[5]) - H) < 1e-5, f"H mismatch (no clamp expected)"


# ─── Bound checks ─────────────────────────────────────────────────────────────

class TestBounds4D:
    def test_lower_bounds_shape(self):
        assert _BOUNDS_LOWER_4D.shape == (4,)

    def test_upper_bounds_shape(self):
        assert _BOUNDS_UPPER_4D.shape == (4,)

    def test_lower_lt_upper(self):
        assert ((_BOUNDS_LOWER_4D < _BOUNDS_UPPER_4D).all()), \
            "All lower bounds must be strictly less than upper bounds"

    def test_H_bounds(self):
        # H is 4th dimension (index 3)
        assert abs(float(_BOUNDS_LOWER_4D[3]) - 0.04) < 1e-6
        assert abs(float(_BOUNDS_UPPER_4D[3]) - 0.15) < 1e-6

    def test_v0_bounds(self):
        # v0 is 1st dimension (index 0)
        assert float(_BOUNDS_LOWER_4D[0]) > 0.0    # v0 must be positive
        assert float(_BOUNDS_UPPER_4D[0]) <= 0.20  # v0 < 20% variance


# ─── Self-consistency (requires v3 model + CUDA) ─────────────────────────────

@pytest.mark.skipif(not CUDA_OK, reason="CUDA not available")
class TestCalibrateNewtonH:
    """Self-consistency: calibrate_newton_h recovers params from v3 FNO surfaces."""

    @pytest.fixture(scope="class")
    def fno_v3_model(self):
        """Load FNO v3 (learnable H ∈ [0.04, 0.15])."""
        from fno_model import MirrorPaddedFNO2d
        from calibrate import _load_normalizers

        weights = "artifacts/weights/fno_v3_final_prod.pth"
        if not os.path.exists(weights):
            pytest.skip(f"FNO v3 weights not found: {weights}")

        model = MirrorPaddedFNO2d(param_dim=6)
        import torch as _t
        model.load_state_dict(_t.load(weights, map_location="cpu", weights_only=True))
        model.eval()
        _load_normalizers(version="v3")  # calibrate_newton_h requires v3
        return model

    def test_self_consistency(self, fno_v3_model):
        """calibrate_newton_h on FNO v3 surface should recover H≈0.10 with MSE<1e-3."""
        from calibrate import _make_spatial_input, _fno_predict_real_iv

        spatial = _make_spatial_input(T_GRID, K_GRID, device=torch.device("cpu"))
        true_p6 = torch.tensor([[2.0, 0.04, 0.50, -0.70, 0.04, 0.10]])
        with torch.no_grad():
            iv_target = _fno_predict_real_iv(fno_v3_model, true_p6, spatial).numpy()

        res = calibrate_newton_h(fno_v3_model, iv_target, T_GRID, K_GRID,
                                 max_iter=20, verbose=False,
                                 init_H=0.08)

        # MSE should be low (self-consistency on v3 FNO surface)
        assert res["final_mse"] < 1e-3, f"High MSE: {res['final_mse']:.4e}"

    def test_returns_H_key(self, fno_v3_model):
        """Return dict must contain 'H' key."""
        from calibrate import _make_spatial_input, _fno_predict_real_iv

        spatial  = _make_spatial_input(T_GRID, K_GRID, device=torch.device("cpu"))
        true_p6  = torch.tensor([[2.0, 0.04, 0.50, -0.70, 0.04, 0.10]])
        with torch.no_grad():
            iv_target = _fno_predict_real_iv(fno_v3_model, true_p6, spatial).numpy()

        res = calibrate_newton_h(fno_v3_model, iv_target, T_GRID, K_GRID, max_iter=5)
        assert "H" in res
        assert 0.04 <= res["H"] <= 0.15

    def test_theta_history_has_4D(self, fno_v3_model):
        """theta_history entries must have shape (4,) for 4D calibration."""
        from calibrate import _make_spatial_input, _fno_predict_real_iv

        spatial  = _make_spatial_input(T_GRID, K_GRID, device=torch.device("cpu"))
        true_p6  = torch.tensor([[2.0, 0.04, 0.40, -0.50, 0.08, 0.10]])
        with torch.no_grad():
            iv_target = _fno_predict_real_iv(fno_v3_model, true_p6, spatial).numpy()

        res = calibrate_newton_h(fno_v3_model, iv_target, T_GRID, K_GRID, max_iter=5)
        assert "theta_history" in res
        assert all(np.asarray(h).shape == (4,) for h in res["theta_history"])
