import pytest
import numpy as np
import time
from market.vix_pricing import (
    model_vix,
    vix_futures_curve,
    model_variance_swap_rate
)

def test_model_vix_performance():
    """
    Verify that VIX index calculation is fast (<10ms) over a range of typical parameter values.
    """
    latencies = []
    np.random.seed(42)
    
    # Run 100 trials to get stable statistics
    for _ in range(100):
        kappa = np.random.uniform(0.1, 5.0)
        theta = np.random.uniform(0.01, 0.25)
        sigma = np.random.uniform(0.1, 1.5)
        rho = np.random.uniform(-0.9, 0.0)
        v0 = np.random.uniform(0.01, 0.25)
        H = np.random.uniform(0.02, 0.48)
        
        t0 = time.perf_counter()
        vix = model_vix(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0, H=H)
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0) # to ms
        assert vix >= 0.0, f"VIX value {vix} cannot be negative"
        
    mean_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p90_lat = np.percentile(latencies, 90)
    p99_lat = np.percentile(latencies, 99)
    max_lat = np.max(latencies)
    
    print(f"\n[PERFORMANCE] model_vix latency (ms):")
    print(f"  Mean: {mean_lat:.3f} ms")
    print(f"  p50:  {p50_lat:.3f} ms")
    print(f"  p90:  {p90_lat:.3f} ms")
    print(f"  p99:  {p99_lat:.3f} ms")
    print(f"  Max:  {max_lat:.3f} ms")
    
    # Assert latency target
    assert p99_lat < 10.0, f"99th percentile latency ({p99_lat:.3f}ms) exceeds 10ms target"


def test_model_vix_extreme_parameters():
    """
    Test convergence and stability of model_vix under extreme but mathematically valid parameters.
    """
    # 1. Very high mean reversion speed kappa
    vix_high_kappa = model_vix(kappa=100.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08)
    assert np.isfinite(vix_high_kappa)
    assert vix_high_kappa > 0.0
    
    # 2. Very low mean reversion speed kappa (almost 0)
    vix_low_kappa = model_vix(kappa=1e-5, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08)
    assert np.isfinite(vix_low_kappa)
    assert vix_low_kappa > 0.0
    
    # 3. Very low Hurst index H -> 0 (extremely rough)
    vix_rough = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.005)
    assert np.isfinite(vix_rough)
    assert vix_rough > 0.0
    
    # 4. Hurst index H -> 0.5 (approaching standard Heston)
    vix_standard = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.495)
    assert np.isfinite(vix_standard)
    assert vix_standard > 0.0
    
    # 5. Very high initial variance v0
    vix_high_v0 = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=5.0, H=0.08)
    assert np.isfinite(vix_high_v0)
    assert vix_high_v0 > 100.0  # highly elevated VIX
    
    # 6. Very low initial variance v0
    vix_low_v0 = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=1e-5, H=0.08)
    assert np.isfinite(vix_low_v0)
    assert vix_low_v0 > 0.0


def test_model_vix_invalid_parameters():
    """
    Test behaviour of model_vix under invalid / non-physical parameters.
    """
    # 1. Negative initial variance v0
    # Should either raise ValueError, or if it runs, check if it handles it (e.g. returns 0 or raises exception)
    try:
        vix_neg_v0 = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=-0.05, H=0.08)
        print(f"Warning: model_vix did not raise exception for negative v0=-0.05, returned {vix_neg_v0}")
    except Exception as e:
        print(f"Fitted exception for negative v0: {e}")
        
    # 2. Negative theta (mean reversion level)
    try:
        vix_neg_theta = model_vix(kappa=1.0, theta=-0.05, sigma=0.8, rho=-0.34, v0=0.10, H=0.08)
        print(f"Warning: model_vix did not raise exception for negative theta=-0.05, returned {vix_neg_theta}")
    except Exception as e:
        print(f"Fitted exception for negative theta: {e}")
        
    # 3. Negative kappa (mean reversion speed)
    # This might cause exponential divergence in the ODE. Let's see if solve_ivp fails or diverges.
    try:
        vix_neg_kappa = model_vix(kappa=-1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08)
        print(f"Warning: model_vix did not raise exception for negative kappa=-1.0, returned {vix_neg_kappa}")
    except Exception as e:
        print(f"Fitted exception for negative kappa: {e}")

    # 4. Hurst index H outside (0, 0.5)
    # H <= 0 or H >= 0.5. Under Rough Heston, H must be in (0, 0.5).
    # If H >= 0.5, let's see what happens.
    try:
        vix_h_large = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.6)
        print(f"Warning: model_vix did not raise exception for H=0.6 (outside bounds), returned {vix_h_large}")
    except Exception as e:
        print(f"Fitted exception for H=0.6: {e}")

    try:
        vix_h_neg = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=-0.1)
        print(f"Warning: model_vix did not raise exception for negative H=-0.1, returned {vix_h_neg}")
    except Exception as e:
        print(f"Fitted exception for negative H: {e}")


def test_vix_futures_curve_stress():
    """
    Test vix_futures_curve under extreme and invalid parameters.
    """
    maturities = np.array([0.083, 0.164, 0.246, 0.328, 0.411, 0.493, 0.575, 0.657])
    
    # 1. Performance test: measuring curve computation latency
    t0 = time.perf_counter()
    curve = vix_futures_curve(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, maturities=maturities)
    t1 = time.perf_counter()
    duration = (t1 - t0) * 1000.0
    print(f"\n[PERFORMANCE] vix_futures_curve latency for 8 maturities: {duration:.2f} ms")
    
    assert len(curve) == len(maturities)
    assert np.all(np.isfinite(curve))
    
    # 2. Extreme volatility of volatility (sigma)
    # Riccati equations have 0.5 * (sigma**2) * (Phi**2). Extremely large sigma could cause blow-up.
    try:
        curve_high_sigma = vix_futures_curve(kappa=1.0, theta=0.08, sigma=5.0, rho=-0.34, v0=0.10, H=0.08, maturities=maturities)
        print(f"VIX futures curve with sigma=5.0: {curve_high_sigma}")
    except Exception as e:
        print(f"Fitted exception for high sigma=5.0: {e}")
        
    # 3. High mean reversion kappa
    curve_high_kappa = vix_futures_curve(kappa=20.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, maturities=maturities)
    assert np.all(np.isfinite(curve_high_kappa))

    # 4. Negative parameter testing for vix_futures_curve
    try:
        curve_neg_v0 = vix_futures_curve(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=-0.05, H=0.08, maturities=maturities)
        print(f"Warning: vix_futures_curve did not raise exception for negative v0=-0.05, returned {curve_neg_v0}")
    except Exception as e:
        print(f"Fitted exception for negative v0 in futures: {e}")

    try:
        curve_neg_theta = vix_futures_curve(kappa=1.0, theta=-0.05, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, maturities=maturities)
        print(f"Warning: vix_futures_curve did not raise exception for negative theta=-0.05, returned {curve_neg_theta}")
    except Exception as e:
        print(f"Fitted exception for negative theta in futures: {e}")

    try:
        # Negative kappa might cause exponential blow up in Riccati / ODE solver. Let's see if it handles it.
        curve_neg_kappa = vix_futures_curve(kappa=-1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, maturities=maturities)
        print(f"Warning: vix_futures_curve did not raise exception for negative kappa=-1.0, returned {curve_neg_kappa}")
    except Exception as e:
        print(f"Fitted exception/failure for negative kappa in futures: {e}")


def test_model_variance_swap_rate_stress():
    """
    Test model_variance_swap_rate under extreme and invalid parameters.
    """
    # 1. Baseline swap rate
    vs_rate = model_variance_swap_rate(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, T=1.0)
    assert vs_rate > 0.0
    
    # 2. Extreme T (very short vs very long)
    vs_short = model_variance_swap_rate(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, T=0.001)
    assert np.isfinite(vs_short)
    
    vs_long = model_variance_swap_rate(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, T=20.0)
    assert np.isfinite(vs_long)
    
    # 3. Negative parameters
    try:
        vs_neg_v0 = model_variance_swap_rate(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=-0.05, H=0.08, T=1.0)
        print(f"Warning: model_variance_swap_rate did not raise exception for negative v0=-0.05, returned {vs_neg_v0}")
    except Exception as e:
        print(f"Fitted exception for negative v0 in var swap: {e}")

    try:
        vs_neg_theta = model_variance_swap_rate(kappa=1.0, theta=-0.05, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, T=1.0)
        print(f"Warning: model_variance_swap_rate did not raise exception for negative theta=-0.05, returned {vs_neg_theta}")
    except Exception as e:
        print(f"Fitted exception for negative theta in var swap: {e}")

    try:
        vs_neg_kappa = model_variance_swap_rate(kappa=-1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, T=1.0)
        print(f"Warning: model_variance_swap_rate did not raise exception for negative kappa=-1.0, returned {vs_neg_kappa}")
    except Exception as e:
        print(f"Fitted exception for negative kappa in var swap: {e}")
