import pytest
import numpy as np
import torch
from deepvol.mrm.arbitrage import check_calendar_arbitrage, check_butterfly_arbitrage_durrleman, check_butterfly_arbitrage_price, check_arbitrage
from deepvol.mrm.guardian import ModelRiskGuardian

def test_calendar_arbitrage_detection():
    T_grid = np.array([0.1, 0.5, 1.0])
    K_grid = np.array([-0.2, 0.0, 0.2])
    
    # Safe surface: total variance w = iv^2 * T increases with T
    # T=0.1: w = 0.2^2 * 0.1 = 0.004
    # T=0.5: w = 0.2^2 * 0.5 = 0.02
    # T=1.0: w = 0.2^2 * 1.0 = 0.04
    safe_iv = np.full((3, 3), 0.2)
    res_safe = check_calendar_arbitrage(safe_iv, T_grid)
    assert not res_safe["has_arbitrage"]
    
    # Anomalous surface: total variance decreases with T
    # T=0.1: w = 0.4^2 * 0.1 = 0.016
    # T=0.5: w = 0.1^2 * 0.5 = 0.005 (decreases!)
    bad_iv = np.array([
        [0.4, 0.4, 0.4],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2]
    ])
    res_bad = check_calendar_arbitrage(bad_iv, T_grid)
    assert res_bad["has_arbitrage"]
    assert res_bad["violations"][0, 0]  # First transition decreases

def test_butterfly_arbitrage_detection():
    T_grid = np.array([0.5, 1.0])
    K_grid = np.array([-0.5, 0.0, 0.5])
    
    # Flat smile (no arbitrage)
    safe_iv = np.full((2, 3), 0.2)
    res_safe_dur = check_butterfly_arbitrage_durrleman(safe_iv, K_grid, T_grid)
    res_safe_prc = check_butterfly_arbitrage_price(safe_iv, K_grid, T_grid)
    assert not res_safe_dur["has_arbitrage"]
    assert not res_safe_prc["has_arbitrage"]
    
    # Severe butterfly arbitrage: put a massive hump in the middle strike IV
    # This creates a non-convex option price surface / negative probability density
    bad_iv = np.array([
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.1]
    ])
    res_bad_dur = check_butterfly_arbitrage_durrleman(bad_iv, K_grid, T_grid)
    res_bad_prc = check_butterfly_arbitrage_price(bad_iv, K_grid, T_grid)
    assert res_bad_dur["has_arbitrage"]
    assert res_bad_prc["has_arbitrage"]

def test_guardian_parameter_checks():
    guardian = ModelRiskGuardian(vol_of_vol_limit=0.99, hurst_limit=0.015, residual_limit=0.0150)
    
    # 1. Clean Heston
    res_clean_heston = guardian.check_parameters(
        "heston",
        {"kappa": 1.5, "theta": 0.04, "sigma": 0.3, "rho": -0.6, "v0": 0.04}
    )
    assert not res_clean_heston["anomaly_detected"]
    
    # 2. Pinned Heston vol-of-vol
    res_pinned_heston = guardian.check_parameters(
        "heston",
        {"kappa": 1.5, "theta": 0.04, "sigma": 0.995, "rho": -0.6, "v0": 0.04}
    )
    assert res_pinned_heston["anomaly_detected"]
    assert any("vol-of-vol pinned" in a for a in res_pinned_heston["anomalies"])
    
    # 3. Clean Rough Bergomi
    res_clean_rb = guardian.check_parameters(
        "rbergomi",
        {"v0": 0.04, "H": 0.1, "eta": 0.5, "rho": -0.7}
    )
    assert not res_clean_rb["anomaly_detected"]
    
    # 4. Pinned Hurst exponent
    res_pinned_rb = guardian.check_parameters(
        "rbergomi",
        {"v0": 0.04, "H": 0.01, "eta": 0.5, "rho": -0.7}
    )
    assert res_pinned_rb["anomaly_detected"]
    assert any("Hurst exponent pinned" in a for a in res_pinned_rb["anomalies"])
    
    # 5. Pinned vol-of-vol rough Bergomi
    res_pinned_rb_vol = guardian.check_parameters(
        "rbergomi",
        {"v0": 0.04, "H": 0.1, "eta": 1.0, "rho": -0.7}
    )
    assert res_pinned_rb_vol["anomaly_detected"]
    assert any("vol-of-vol pinned" in a for a in res_pinned_rb_vol["anomalies"])
    
    # 6. High residuals (> 150 bps)
    res_high_res = guardian.check_parameters(
        "heston",
        {"kappa": 1.5, "theta": 0.04, "sigma": 0.3, "rho": -0.6, "v0": 0.04},
        rmse=0.0180
    )
    assert res_high_res["anomaly_detected"]
    assert any("residual exceeds" in a for a in res_high_res["anomalies"])

def test_guardian_price_or_fallback_clean():
    guardian = ModelRiskGuardian()
    T_grid = np.array([0.1, 0.5, 1.0])
    K_grid = np.array([-0.2, 0.0, 0.2])
    params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04}
    
    res = guardian.price_or_fallback(
        model_name="heston",
        parameters=params,
        spot=1.0,
        strikes=K_grid,
        maturities=T_grid,
        fallback_route="fourier"
    )
    assert not res["fallback_triggered"]
    assert res["guardian_status"] == "passed"
    assert res["prices"].shape == (3, 3)
    assert res["ivs"].shape == (3, 3)
    assert not np.isnan(res["prices"]).any()

def test_guardian_price_or_fallback_anomalous_fourier():
    guardian = ModelRiskGuardian(vol_of_vol_limit=0.99)
    T_grid = np.array([0.1, 0.5, 1.0])
    K_grid = np.array([-0.2, 0.0, 0.2])
    
    # Pinned parameters to trigger fallback
    params = {"kappa": 2.0, "theta": 0.04, "sigma": 1.0, "rho": -0.7, "v0": 0.04}
    
    res = guardian.price_or_fallback(
        model_name="heston",
        parameters=params,
        spot=1.0,
        strikes=K_grid,
        maturities=T_grid,
        fallback_route="fourier"
    )
    assert res["fallback_triggered"]
    assert res["guardian_status"] == "fallback_fourier_safe_params"
    assert len(res["anomalies"]) > 0
    assert res["prices"].shape == (3, 3)
    assert res["ivs"].shape == (3, 3)
    assert not np.isnan(res["prices"]).any()
    
def test_guardian_price_or_fallback_anomalous_particle():
    # Only run MLSV particle solver if CPU/GPU supports it
    guardian = ModelRiskGuardian(vol_of_vol_limit=0.99)
    T_grid = np.array([0.1, 0.5])
    K_grid = np.array([-0.1, 0.0, 0.1])
    
    # Pinned parameters to trigger fallback
    params = {"kappa": 2.0, "theta": 0.04, "sigma": 1.0, "rho": -0.7, "v0": 0.04}
    
    res = guardian.price_or_fallback(
        model_name="heston",
        parameters=params,
        spot=1.0,
        strikes=K_grid,
        maturities=T_grid,
        fallback_route="particle"
    )
    assert res["fallback_triggered"]
    assert res["guardian_status"] == "fallback_particle_safe_params"
    assert len(res["anomalies"]) > 0
    assert res["prices"].shape == (2, 3)
    assert res["ivs"].shape == (2, 3)


# ── F-07: Dedicated price_to_iv edge-case tests ──────────────────────────────

from deepvol.mrm.guardian import price_to_iv


def test_price_to_iv_round_trip():
    """Verify that price_to_iv inverts a known BS price back to the correct IV."""
    import scipy.stats as stats
    S, K, T, sigma_true = 100.0, 100.0, 1.0, 0.25
    d1 = (np.log(S / K) + 0.5 * sigma_true**2 * T) / (sigma_true * np.sqrt(T))
    d2 = d1 - sigma_true * np.sqrt(T)
    price = S * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)

    iv_recovered = price_to_iv(price, S, K, T)
    assert abs(iv_recovered - sigma_true) < 1e-6, f"Expected {sigma_true}, got {iv_recovered}"


def test_price_to_iv_below_intrinsic():
    """Price ≤ intrinsic should return the floor IV (1e-4)."""
    S, K, T = 110.0, 100.0, 0.5
    intrinsic = S - K  # = 10.0
    iv = price_to_iv(intrinsic - 1.0, S, K, T)
    assert iv == 1e-4, f"Expected floor 1e-4 for sub-intrinsic price, got {iv}"

    iv_exact = price_to_iv(intrinsic, S, K, T)
    assert iv_exact == 1e-4, f"Expected floor 1e-4 for exact intrinsic price, got {iv_exact}"


def test_price_to_iv_at_spot():
    """Price ≥ spot should return the ceiling IV (5.0)."""
    S, K, T = 100.0, 100.0, 1.0
    iv = price_to_iv(S, S, K, T)
    assert iv == 5.0, f"Expected ceiling 5.0 for price=spot, got {iv}"

    iv_above = price_to_iv(S + 10.0, S, K, T)
    assert iv_above == 5.0, f"Expected ceiling 5.0 for price>spot, got {iv_above}"


def test_price_to_iv_deep_otm():
    """Deep OTM option (very small price) should return a small but valid IV."""
    S, K, T = 100.0, 200.0, 0.5  # Very deep OTM
    price = 0.001  # Tiny premium
    iv = price_to_iv(price, S, K, T)
    assert 1e-4 <= iv <= 5.0, f"IV {iv} outside valid range"
    assert np.isfinite(iv), f"IV is not finite: {iv}"


def test_price_to_iv_zero_and_negative():
    """Zero or negative prices should return floor IV without error."""
    S, K, T = 100.0, 100.0, 1.0
    iv_zero = price_to_iv(0.0, S, K, T)
    assert iv_zero == 1e-4

    iv_neg = price_to_iv(-5.0, S, K, T)
    assert iv_neg == 1e-4
