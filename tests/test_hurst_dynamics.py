import os
import sys
import json
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

# Add project root and src to path for robust imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from analysis.hurst_dynamics import (
    run_historical_study,
    compute_hurst_statistics,
    detect_regime_changes,
)
from analysis.crypto_hurst import (
    align_crypto_inputs,
    generate_mock_crypto_data,
    run_crypto_historical_study,
)
from calibration.batch_calibration import CalibrationResult


# ============================================================================
# 1. Pettitt test correctness on synthetic step-change series
# ============================================================================

def test_pettitt_test_synthetic_step_change():
    """
    Verify Pettitt test correctly identifies the change point location
    in a synthetic series with a sharp step change.
    """
    # 15 elements of 0.05, followed by 15 elements of 0.12 (total length 30)
    x = np.array([0.05] * 15 + [0.12] * 15)
    df = pd.DataFrame({"date": [f"2024-01-{i:02d}" for i in range(1, 31)], "H": x})
    
    res = detect_regime_changes(df)
    assert not res.empty
    
    # The change point occurs after the 15th element (index 14).
    # The new regime starts at index 15 (16th element).
    assert res.loc[0, "change_point_index"] == 15
    assert res.loc[0, "change_point_date"] == "2024-01-16"
    assert res.loc[0, "p_value"] < 0.05  # highly significant
    assert res.loc[0, "is_significant"] == True


# ============================================================================
# 2. Pettitt test on noise-only series
# ============================================================================

def test_pettitt_test_no_change():
    """
    Verify Pettitt test on a flat / noise-only series has a high p-value
    and is not marked as significant.
    """
    # Random noise around 0.08
    rng = np.random.default_rng(42)
    x = 0.08 + rng.normal(0, 0.005, 100)
    df = pd.DataFrame({"date": [f"2024-01-{i:02d}" for i in range(1, 101)], "H": x})
    
    res = detect_regime_changes(df)
    assert not res.empty
    assert res.loc[0, "p_value"] > 0.05  # should not be significant
    assert not res.loc[0, "is_significant"]


# ============================================================================
# 3. Pettitt test on too short series
# ============================================================================

def test_pettitt_test_too_short():
    """
    Verify Pettitt test handles very short series gracefully.
    """
    x = np.array([0.08, 0.09, 0.08])
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"], "H": x})
    
    res = detect_regime_changes(df)
    assert res.empty  # should return empty df if less than 4 elements


# ============================================================================
# 4. Hurst statistics computation
# ============================================================================

def test_hurst_statistics_computation():
    """
    Verify that compute_hurst_statistics returns the correct mean, std,
    and autocorrelation values.
    """
    # Create simple sine wave series
    h_vals = [0.08, 0.09, 0.10, 0.11, 0.10, 0.09, 0.08, 0.07, 0.08, 0.09]
    df = pd.DataFrame({"H": h_vals})
    
    stats = compute_hurst_statistics(df)
    
    assert abs(stats["mean"] - np.mean(h_vals)) < 1e-5
    assert abs(stats["std"] - np.std(h_vals, ddof=1)) < 1e-5
    assert "autocorr_lag_1" in stats
    assert "autocorr_lag_5" in stats
    assert "autocorr_lag_10" in stats
    assert "autocorr_lag_20" in stats
    assert np.isnan(stats["autocorr_lag_20"])  # series length is 10, so lag 20 is nan


# ============================================================================
# 5. Incremental resume functionality
# ============================================================================

def test_incremental_resume_functionality(tmp_path):
    """
    Test that run_historical_study successfully loads existing results,
    skips completed dates, calls calibrate_batch on missing dates, and saves.
    """
    results_dir = tmp_path / "results" / "hurst_dynamics"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = results_dir / "SPX_hurst_study.json"
    
    # Save some pre-existing results
    pre_existing = [
        CalibrationResult(
            date="2024-01-02",
            currency="SPX",
            params={"kappa": 1.0, "theta": 0.08, "sigma": 0.5, "rho": -0.7, "v0": 0.08, "H": 0.08},
            rmse_bps=10.0,
            runtime_ms=100.0,
            converged=True,
        ).to_dict()
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(pre_existing, f, indent=2)
        
    # Mock calibrate_batch
    mock_new_result = CalibrationResult(
        date="2024-01-03",
        currency="SPX",
        params={"kappa": 1.1, "theta": 0.09, "sigma": 0.6, "rho": -0.6, "v0": 0.09, "H": 0.09},
        rmse_bps=12.0,
        runtime_ms=110.0,
        converged=True,
    )
    
    with patch("analysis.hurst_dynamics.project_root", tmp_path), \
         patch("analysis.hurst_dynamics.calibrate_batch", return_value=[mock_new_result]) as mock_calib:
         
        # Run study from 2024-01-02 to 2024-01-03.
        # 2024-01-02 is already present, so it should only calibrate 2024-01-03.
        df = run_historical_study(start="2024-01-02", end="2024-01-03", currency="SPX")
        
        # Verify calibrate_batch was called with only 2024-01-03
        mock_calib.assert_called_once()
        called_dates = mock_calib.call_args[1]["dates"]
        assert called_dates == ["2024-01-03"]
        
        # Verify output dataframe has both dates
        assert len(df) == 2
        assert list(df["date"].values) == ["2024-01-02", "2024-01-03"]
        
        # Verify file contains both dates
        with open(file_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert len(saved_data) == 2
        assert saved_data[0]["date"] == "2024-01-02"
        assert saved_data[1]["date"] == "2024-01-03"


# ============================================================================
# 6. Align crypto inputs
# ============================================================================

def test_align_crypto_inputs():
    """
    Verify align_crypto_inputs divides mark_iv, bid_iv, ask_iv by 100
    if they are in percent (>2.0), but leaves them alone if they are in decimal.
    """
    # Case 1: Percent format
    df_pct = pd.DataFrame({
        "mark_iv": [50.0, 60.0],
        "bid_iv": [48.0, 58.0],
        "ask_iv": [52.0, 62.0],
    })
    aligned_pct = align_crypto_inputs(df_pct)
    assert aligned_pct["mark_iv"].iloc[0] == pytest.approx(0.50)
    assert aligned_pct["bid_iv"].iloc[0] == pytest.approx(0.48)
    assert aligned_pct["ask_iv"].iloc[0] == pytest.approx(0.52)
    
    # Case 2: Decimal format
    df_dec = pd.DataFrame({
        "mark_iv": [0.50, 0.60],
        "bid_iv": [0.48, 0.58],
        "ask_iv": [0.52, 0.62],
    })
    aligned_dec = align_crypto_inputs(df_dec)
    assert aligned_dec["mark_iv"].iloc[0] == pytest.approx(0.50)
    assert aligned_dec["bid_iv"].iloc[0] == pytest.approx(0.48)
    assert aligned_dec["ask_iv"].iloc[0] == pytest.approx(0.52)


# ============================================================================
# 7. Crypto v0 clipping warning / clamping
# ============================================================================

def test_crypto_v0_clipping_and_warning(tmp_path):
    """
    Verify that run_crypto_historical_study issues a warning and clips v0
    when the calibrated parameter exceeds [0.01, 0.25].
    """
    # Result with v0 = 0.35 (outside range [0.01, 0.25])
    mock_res = CalibrationResult(
        date="2027-01-04",
        currency="BTC",
        params={"kappa": 1.0, "theta": 0.08, "sigma": 0.5, "rho": -0.7, "v0": 0.35, "H": 0.08},
        rmse_bps=10.0,
        runtime_ms=100.0,
        converged=True,
    )
    
    with patch("analysis.crypto_hurst.project_root", tmp_path), \
         patch("analysis.crypto_hurst.calibrate_batch", return_value=[mock_res]):
         
        # Expect warning about clipping
        with pytest.warns(UserWarning, match="exceeds FNO training range bounds"):
            df = run_crypto_historical_study(
                start="2027-01-04",
                end="2027-01-04",
                currency="BTC",
                test_mode=True
            )
            
        # Verify it was clipped to FNO bounds (max 0.25)
        assert df["v0"].iloc[0] == pytest.approx(0.25)


# ============================================================================
# 8. Crypto historical study test mode with mock tickers
# ============================================================================

def test_crypto_historical_study_test_mode(tmp_path):
    """
    Verify run_crypto_historical_study runs in test mode using mock tickers
    with year 2027+ and returns correct schemas.
    """
    # Let's generate a mock result with realistic values
    mock_res = CalibrationResult(
        date="2027-01-04",
        currency="BTC",
        params={"kappa": 1.0, "theta": 0.08, "sigma": 0.5, "rho": -0.7, "v0": 0.08, "H": 0.08},
        rmse_bps=12.0,
        runtime_ms=150.0,
        converged=True,
    )
    
    with patch("analysis.crypto_hurst.project_root", tmp_path), \
         patch("analysis.crypto_hurst.calibrate_batch", return_value=[mock_res]) as mock_calib:
         
        df = run_crypto_historical_study(
            start="2027-01-04",
            end="2027-01-04",
            currency="BTC",
            test_mode=True
        )
        
        # Verify calibrate_batch was called with the target surfaces populated
        mock_calib.assert_called_once()
        target_surfs = mock_calib.call_args[1]["target_surfaces"]
        assert "2027-01-04" in target_surfs
        assert target_surfs["2027-01-04"].shape == (8, 11)
        
        # Verify schema
        assert len(df) == 1
        assert "date" in df.columns
        assert "currency" in df.columns
        assert "H" in df.columns
        assert "v0" in df.columns


# ============================================================================
# 9. Verify run_historical_study runs on SPX short subset
# ============================================================================

def test_run_historical_study_spx_convergence_subset(tmp_path):
    """
    Verify run_historical_study runs and converges for SPX on a 3-day subset.
    """
    # We use a short 3-day subset to keep the test fast
    start_date = "2024-01-02"
    end_date = "2024-01-04"
    
    # Use patch to store results in tmp_path
    with patch("analysis.hurst_dynamics.project_root", tmp_path):
        df = run_historical_study(
            start=start_date,
            end=end_date,
            currency="SPX",
            chunk_size=3,
        )
        
        assert len(df) == 3
        # Check convergence rate (converged should be true or rmse < 50 bps)
        convergence_rate = (df["converged"].sum() / len(df))
        assert convergence_rate >= 0.70, f"Convergence rate {convergence_rate} is below 70%"
        
        # Calibrated H values stay within bounds [0.04, 0.15]
        for H_val in df["H"].values:
            assert 0.04 <= H_val <= 0.15, f"H={H_val} is outside [0.04, 0.15]"


# ============================================================================
# 10. Convergence rate check for a longer period
# ============================================================================

def test_run_historical_study_spx_convergence_rate_check(tmp_path):
    """
    Verify run_historical_study runs and converges for at least 70% of days
    on a larger subset (e.g. 2024-01-01 to 2024-01-15, which has 10 business days).
    Also checks H values are within [0.04, 0.15].
    """
    start_date = "2024-01-01"
    end_date = "2024-01-15" # 10 business days
    
    with patch("analysis.hurst_dynamics.project_root", tmp_path):
        df = run_historical_study(
            start=start_date,
            end=end_date,
            currency="SPX",
            chunk_size=5,
        )
        
        assert len(df) >= 7  # at least 10 business days, so should be 10 (or min 7 business days depending on holiday)
        
        convergence_rate = (df["converged"].sum() / len(df))
        print(f"Convergence rate: {convergence_rate:.2%}")
        assert convergence_rate >= 0.70, f"Convergence rate {convergence_rate:.2%} is below 70%"
        
        for H_val in df["H"].values:
            assert 0.04 <= H_val <= 0.15, f"H={H_val} is outside [0.04, 0.15]"
