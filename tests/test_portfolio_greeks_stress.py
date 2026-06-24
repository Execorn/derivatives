import pytest
import numpy as np
import torch
import time
from deepvol.greeks.portfolio_greeks import (
    bs_greeks,
    fno_parameter_jacobian,
    fno_surface_greeks,
    portfolio_greeks,
    pnl_attribution
)
from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer

def test_bs_greeks_extreme_t_sigma():
    """Test bs_greeks for extreme small and large inputs to check robustness."""
    S_vals = [0.0, 1e-15, 50.0, 100.0, 150.0, 1e6, 1e300, float('inf')]
    K_vals = [1e-15, 50.0, 100.0, 150.0, 1e6]
    T_vals = [0.0, -1e-5, 1e-5, 1e-10, 1e-15, 1e-30, 1e-300, 1e-324]
    sigma_vals = [0.0, -0.05, 1e-5, 1e-10, 1e-15, 1e-30, 1e-300, 1e-324, 10.0, 100.0]
    r = 0.05
    
    # We want to check for crashes (exceptions) or unexpected NaNs/Infs
    warnings_or_crashes = []
    
    for S in S_vals:
        for K in K_vals:
            for T in T_vals:
                for sigma in sigma_vals:
                    try:
                        g = bs_greeks(S, K, T, r, sigma)
                        # Check for NaN or Inf in outputs
                        for name, val in g.items():
                            if np.isnan(val) or np.isinf(val):
                                warnings_or_crashes.append(
                                    f"NaN/Inf detected: S={S}, K={K}, T={T}, sigma={sigma} -> Greek '{name}' = {val}"
                                )
                    except Exception as e:
                        warnings_or_crashes.append(
                            f"Exception raised: S={S}, K={K}, T={T}, sigma={sigma} -> {type(e).__name__}: {str(e)}"
                        )
                        
    # Let's print out the issues found (if any)
    if warnings_or_crashes:
        print(f"\n--- Detected {len(warnings_or_crashes)} potential vulnerabilities in bs_greeks ---")
        for issue in warnings_or_crashes[:30]:  # Show first 30
            print(issue)
        if len(warnings_or_crashes) > 30:
            print(f"... and {len(warnings_or_crashes) - 30} more.")
            
    # We don't fail immediately, we want to collect them all. But we can assert or just report.
    # In the final handoff we will report all findings.


def test_fno_greeks_extreme_parameters(fno_v2_model):
    """Test FNO surface Greeks with extreme parameters (inside/outside bounds)."""
    model = fno_v2_model
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    
    # Standard parameters (inside bounds)
    # Bounds: kappa in [0.1, 5.0], theta in [0.01, 0.15], sigma in [0.1, 1.0], rho in [-0.9, -0.1], v0 in [0.01, 0.15], H in [0.02, 0.15]
    theta_in = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    
    # Extremely small parameters
    theta_tiny = np.array([1e-5, 1e-5, 1e-5, -0.99, 1e-5, 1e-5])
    
    # Extremely large parameters
    theta_huge = np.array([100.0, 10.0, 20.0, -0.01, 5.0, 0.99])
    
    for theta_name, theta in [("in-bounds", theta_in), ("tiny", theta_tiny), ("huge", theta_huge)]:
        try:
            g_surf = fno_surface_greeks(model, theta, pn, yn, S=100.0, r=0.05)
            for Greek, surf in g_surf.items():
                num_nan = np.isnan(surf).sum()
                num_inf = np.isinf(surf).sum()
                assert num_nan == 0, f"NaNs found in FNO Greek surface '{Greek}' for {theta_name} parameters: {num_nan} points"
                assert num_inf == 0, f"Infs found in FNO Greek surface '{Greek}' for {theta_name} parameters: {num_inf} points"
        except Exception as e:
            pytest.fail(f"FNO surface Greeks failed for {theta_name} parameters: {type(e).__name__}: {str(e)}")


def test_portfolio_greeks_empty_and_extremes(fno_v2_model):
    """Test portfolio_greeks with empty list and extreme position parameters."""
    model = fno_v2_model
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    theta = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    
    # Empty portfolio
    res_empty = portfolio_greeks([], model, theta, pn, yn, S=100.0)
    assert res_empty["total_delta"] == 0.0
    assert res_empty["total_gamma"] == 0.0
    assert res_empty["total_vanna"] == 0.0
    assert res_empty["total_volga"] == 0.0
    assert np.all(res_empty["vega_bucket"] == 0.0)
    assert res_empty["hedge_contracts"] == 0
    
    # Extreme positions
    extreme_positions = [
        {"K": 100.0, "T": 0.0, "type": "call", "quantity": 1.0},        # T = 0
        {"K": 100.0, "T": -0.5, "type": "call", "quantity": 1.0},       # T < 0
        {"K": 100.0, "T": 1e-15, "type": "call", "quantity": 1.0},      # T very small positive
        {"K": 1e-5, "T": 0.5, "type": "call", "quantity": 1.0},         # Strike very small
        {"K": 1e6, "T": 0.5, "type": "call", "quantity": 1.0},          # Strike very large
        {"K": 100.0, "T": 0.5, "type": "put", "quantity": -2.5},        # Short put
    ]
    
    try:
        res = portfolio_greeks(extreme_positions, model, theta, pn, yn, S=100.0)
        # Check that outputs are finite
        assert np.isfinite(res["total_delta"])
        assert np.isfinite(res["total_gamma"])
        assert np.isfinite(res["total_vanna"])
        assert np.isfinite(res["total_volga"])
        assert np.all(np.isfinite(res["vega_bucket"]))
        assert isinstance(res["hedge_contracts"], int)
    except Exception as e:
        pytest.fail(f"portfolio_greeks failed with extreme positions: {type(e).__name__}: {str(e)}")


def test_performance_benchmarks(fno_v2_model):
    """Benchmark the execution time of Greeks calculation."""
    model = fno_v2_model
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    theta = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    
    # Large portfolio of 100 random positions
    np.random.seed(42)
    large_portfolio = []
    for _ in range(100):
        large_portfolio.append({
            "K": float(np.random.uniform(80, 120)),
            "T": float(np.random.uniform(0.05, 2.0)),
            "type": "call" if np.random.rand() > 0.5 else "put",
            "quantity": float(np.random.uniform(-5.0, 5.0))
        })
        
    # Benchmark portfolio_greeks
    t_start = time.perf_counter()
    iterations = 50
    for _ in range(iterations):
        _ = portfolio_greeks(large_portfolio, model, theta, pn, yn, S=100.0)
    t_end = time.perf_counter()
    avg_latency_ms = (t_end - t_start) / iterations * 1000.0
    
    print(f"\n--- Portfolio Greeks Performance Benchmark ---")
    print(f"Average latency for 100 positions: {avg_latency_ms:.2f} ms")
    
    # Ensure it meets reasonable real-time requirements (e.g., < 100ms for a 100-position portfolio)
    assert avg_latency_ms < 100.0, f"Aggregation latency too high: {avg_latency_ms:.2f} ms"
