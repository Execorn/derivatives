import pytest
import numpy as np
from deepvol.market.vix_pricing import (
    model_vix,
    vix_futures_curve,
    download_vix_futures,
    joint_calibration_loss,
    model_variance_swap_rate
)

def test_model_vix_sanity():
    # Test that VIX index value is within reasonable range for baseline parameters
    vix = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08)
    assert 10.0 < vix < 40.0, f"VIX value {vix} out of expected range"
    
    # Test that VIX increases when initial variance v0 increases (monotonicity check)
    vix_low = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.05, H=0.08)
    vix_high = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.15, H=0.08)
    assert vix_high > vix_low, "VIX should increase when initial variance increases"


def test_vix_futures_curve_sanity():
    maturities = np.array([0.1, 0.3, 0.5])
    curve = vix_futures_curve(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, maturities=maturities)
    
    # Check shape and types
    assert isinstance(curve, np.ndarray)
    assert curve.shape == (3,)
    assert np.all(curve > 0.0)
    
    # Contango test (v0 < theta): expect upward sloping curve
    curve_contango = vix_futures_curve(kappa=2.0, theta=0.12, sigma=0.8, rho=-0.34, v0=0.06, H=0.08, maturities=maturities)
    assert curve_contango[2] > curve_contango[0], "VIX futures curve should be in contango when v0 < theta"
    
    # Backwardation test (v0 > theta): expect downward sloping curve
    curve_back = vix_futures_curve(kappa=2.0, theta=0.06, sigma=0.8, rho=-0.34, v0=0.15, H=0.08, maturities=maturities)
    assert curve_back[2] < curve_back[0], "VIX futures curve should be in backwardation when v0 > theta"


def test_download_vix_futures():
    res = download_vix_futures("2024-01-02")
    assert isinstance(res, dict)
    assert "maturities" in res
    assert "prices" in res
    assert len(res["maturities"]) == len(res["prices"])
    assert res["prices"][0] > 0.0


def test_joint_calibration_loss():
    # Setup dummy data matching training shapes
    dummy_spx = np.full((8, 11), 0.20) # 20% IV everywhere
    dummy_vix_fut = np.array([14.0, 15.0, 16.0])
    vix_maturities = np.array([0.1, 0.3, 0.5])
    
    params = np.array([1.0, 0.08, 0.8, -0.34, 0.10, 0.08])
    loss = joint_calibration_loss(params, dummy_spx, dummy_vix_fut, vix_maturities)
    
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_model_variance_swap_rate_sanity():
    # Test variance swap rate for default parameters
    vs_rate = model_variance_swap_rate(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, T=1.0)
    assert vs_rate > 0.0, f"Variance swap rate {vs_rate} should be positive"
    
    # Monotonicity check in initial variance v0
    vs_low = model_variance_swap_rate(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.05, H=0.08, T=1.0)
    vs_high = model_variance_swap_rate(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.15, H=0.08, T=1.0)
    assert vs_high > vs_low, "Variance swap rate should increase as v0 increases"
