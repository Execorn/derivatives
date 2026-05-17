"""
Unit tests — Phase 2: MC Dropout uncertainty estimation.

Tests cover:
  1. Output shapes of (mean_iv, std_iv)
  2. std_iv is non-negative everywhere
  3. Model returns to eval() mode after the call
  4. Dropout actually produces variance (std_iv > 0)
  5. The inverse transform IV = sqrt(W/T) yields physically plausible values
  6. num_samples parameter is respected (smoke test with small N)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import joblib

# ── Path setup ─────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from model import HestonSurrogateMLP
from calibrator import HestonCalibrator

# ── Fixtures ───────────────────────────────────────────────────────────────────

ARTIFACTS = PROJECT_ROOT / "artifacts"


@pytest.fixture(scope="module")
def calibrator() -> HestonCalibrator:
    """Load trained model + scalers and return a HestonCalibrator instance."""
    f_scaler = joblib.load(ARTIFACTS / "scalers" / "feature_scaler.pkl")
    t_scaler = joblib.load(ARTIFACTS / "scalers" / "target_scaler.pkl")

    model = HestonSurrogateMLP()
    model.load_state_dict(
        torch.load(ARTIFACTS / "weights" / "heston_best.pth", map_location="cpu",
                   weights_only=False)
    )
    model.eval()

    return HestonCalibrator(model, f_scaler, t_scaler, method="L-BFGS-B")


# Physically valid Heston parameter set satisfying the Feller condition:
#   data order: [v0, rho, sigma, theta, kappa]
#   Feller: 2 * kappa * theta > sigma^2  →  2*5*0.10 = 1.0 > 0.25 ✓
VALID_PARAMS = np.array([0.02, -0.50, 0.50, 0.10, 5.0])


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestPredictWithUncertainty:

    def test_output_shapes(self, calibrator):
        """mean_iv and std_iv must both have shape (88,)."""
        mean_iv, std_iv = calibrator.predict_with_uncertainty(VALID_PARAMS, num_samples=10)
        assert mean_iv.shape == (88,), f"mean_iv shape {mean_iv.shape} ≠ (88,)"
        assert std_iv.shape == (88,), f"std_iv shape {std_iv.shape} ≠ (88,)"

    def test_std_non_negative(self, calibrator):
        """Standard deviation must be ≥ 0 at every grid point."""
        _, std_iv = calibrator.predict_with_uncertainty(VALID_PARAMS, num_samples=20)
        assert np.all(std_iv >= 0), "std_iv has negative entries"

    def test_model_restored_to_eval(self, calibrator):
        """Calibrator must restore model.training = False after the call."""
        calibrator.predict_with_uncertainty(VALID_PARAMS, num_samples=5)
        assert not calibrator.surrogate_model.training, (
            "surrogate_model is still in train() mode after predict_with_uncertainty"
        )

    def test_dropout_produces_nonzero_variance(self, calibrator):
        """With p=0.1 dropout and 50 samples the std should be > 0 somewhere."""
        _, std_iv = calibrator.predict_with_uncertainty(VALID_PARAMS, num_samples=50)
        assert np.any(std_iv > 0), (
            "std_iv is zero everywhere — dropout may be inactive or p=0"
        )

    def test_iv_physically_plausible(self, calibrator):
        """Mean IV values should be in the range (0, 2] for a well-trained model."""
        mean_iv, _ = calibrator.predict_with_uncertainty(VALID_PARAMS, num_samples=20)
        assert np.all(mean_iv > 0), "mean_iv has non-positive entries"
        assert np.all(mean_iv <= 2.0), f"mean_iv has implausibly large entries: {mean_iv.max():.4f}"

    def test_num_samples_respected(self, calibrator):
        """Smoke-test with num_samples=1; must return without error."""
        mean_iv, std_iv = calibrator.predict_with_uncertainty(VALID_PARAMS, num_samples=1)
        assert mean_iv.shape == (88,)
        # With 1 sample std must be 0 (single observation has no variance)
        assert np.allclose(std_iv, 0.0), "std_iv should be 0 for num_samples=1"

    def test_two_sigma_bounds_order(self, calibrator):
        """upper_iv ≥ mean_iv ≥ lower_iv must hold element-wise."""
        mean_iv, std_iv = calibrator.predict_with_uncertainty(VALID_PARAMS, num_samples=30)
        upper_iv = mean_iv + 2.0 * std_iv
        lower_iv = np.maximum(mean_iv - 2.0 * std_iv, 1e-6)
        assert np.all(upper_iv >= mean_iv), "upper bound is below mean"
        assert np.all(mean_iv >= lower_iv), "mean is below lower bound"
