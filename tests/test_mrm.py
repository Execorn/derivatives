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
