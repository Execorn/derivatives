import pytest
import numpy as np
import torch
import time
from deepvol.greeks.portfolio_greeks import (
    bs_greeks,
    fno_parameter_jacobian,
    fno_surface_greeks,
    portfolio_greeks,
    pnl_attribution,
    portfolio_price_tensor,
    interpolate_bilinear,
    _bilinear_interp
)
from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer

def test_adversarial_bs_greeks_failures():
    """Systematically explore failure modes of bs_greeks under extreme inputs."""
    # Test cases that are likely to cause overflow, division by zero, or NaN
    S_vals = [1e-15, 1e-5, 100.0, 1e15, float('inf'), float('nan')]
    K_vals = [1e-15, 100.0, 1e15, float('nan')]
    T_vals = [0.0, -1.0, 1e-15, 1e-300, 1.0, 1e5, float('inf'), float('nan')]
    sigma_vals = [0.0, -0.1, 1e-15, 1e-300, 0.2, 10.0, 1e5, float('inf'), float('nan')]
    r_vals = [0.0, 0.05, 1e5, float('nan')]
    
    failures = []
    nan_infs = []
    
    for S in S_vals:
        for K in K_vals:
            for T in T_vals:
                for sigma in sigma_vals:
                    for r in r_vals:
                        try:
                            res = bs_greeks(S, K, T, r, sigma)
                            for k, v in res.items():
                                if not np.isfinite(v):
                                    nan_infs.append((S, K, T, r, sigma, k, v))
                        except Exception as e:
                            failures.append((S, K, T, r, sigma, type(e).__name__, str(e)))
                            
    print(f"\n[Adversarial BS Greeks] Total combinations tested: {len(S_vals)*len(K_vals)*len(T_vals)*len(sigma_vals)*len(r_vals)}")
    print(f"[Adversarial BS Greeks] Total unhandled exceptions: {len(failures)}")
    print(f"[Adversarial BS Greeks] Total finite violations (NaN/Inf): {len(nan_infs)}")
    
    if len(failures) > 0:
        print("\n--- Example Unhandled Exceptions ---")
        for f in failures[:10]:
            print(f"Inputs: S={f[0]}, K={f[1]}, T={f[2]}, r={f[3]}, sigma={f[4]} | Error: {f[5]}: {f[6]}")
            
    if len(nan_infs) > 0:
        print("\n--- Example NaN/Inf Outputs ---")
        for ni in nan_infs[:10]:
            print(f"Inputs: S={ni[0]}, K={ni[1]}, T={ni[2]}, r={ni[3]}, sigma={ni[4]} | Output '{ni[5]}' = {ni[6]}")

    # Assert that no exceptions are raised (robustness requirement)
    assert len(failures) == 0, f"Detected {len(failures)} unhandled exceptions in bs_greeks"
    # Assert that no NaNs/Infs are generated (robustness requirement)
    assert len(nan_infs) == 0, f"Detected {len(nan_infs)} NaN/Inf outputs in bs_greeks"


def test_adversarial_fno_greeks_boundaries(fno_v2_model):
    """Stress test FNO surface greeks and parameter Jacobian with invalid/extreme inputs."""
    model = fno_v2_model
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    
    # 1. Test parameter combinations with NaNs and Infs
    theta_nan = np.array([2.5, 0.08, 0.5, float('nan'), 0.08, 0.08])
    theta_inf = np.array([2.5, 0.08, float('inf'), -0.5, 0.08, 0.08])
    
    # 2. Test parameter combinations extremely out-of-bounds (e.g. very negative, huge)
    theta_huge = np.array([1e10, 1e10, 1e10, -100.0, 1e10, 1e10])
    theta_neg = np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0])
    
    test_thetas = [
        ("nan_theta", theta_nan),
        ("inf_theta", theta_inf),
        ("huge_theta", theta_huge),
        ("neg_theta", theta_neg)
    ]
    
    failures = []
    nan_infs = []
    
    for name, theta in test_thetas:
        try:
            res = fno_surface_greeks(model, theta, pn, yn, S=100.0)
            for k, surf in res.items():
                if isinstance(surf, np.ndarray):
                    num_nonfinite = np.isnan(surf).sum() + np.isinf(surf).sum()
                    if num_nonfinite > 0:
                        nan_infs.append((name, k, num_nonfinite))
        except Exception as e:
            failures.append((name, type(e).__name__, str(e)))
            
    print(f"\n[Adversarial FNO Greeks] Total failures: {len(failures)}")
    print(f"[Adversarial FNO Greeks] Total NaN/Inf surfaces: {len(nan_infs)}")
    
    for f in failures:
        print(f"Scenario: {f[0]} | Error: {f[1]}: {f[2]}")
    for ni in nan_infs:
        print(f"Scenario: {ni[0]} | Surface '{ni[1]}' has {ni[2]} non-finite values")
        
    assert len(failures) == 0, f"Detected FNO surface greeks crash under extreme parameters"
    assert len(nan_infs) == 0, f"Detected FNO surface greeks NaN/Infs"


def test_portfolio_greeks_robustness(fno_v2_model):
    """Stress test portfolio aggregation with various pathological positions."""
    model = fno_v2_model
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    theta = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    
    pathological_portfolios = [
        # Zero strike options
        [{"K": 0.0, "T": 0.5, "type": "call", "quantity": 10.0}],
        # Negative strike options
        [{"K": -10.0, "T": 0.5, "type": "call", "quantity": 10.0}],
        # Nan/inf strike/tenor options
        [{"K": float('nan'), "T": 0.5, "type": "call", "quantity": 1.0}],
        [{"K": 100.0, "T": float('inf'), "type": "call", "quantity": 1.0}],
        # Gigantic quantity or notional
        [{"K": 100.0, "T": 0.5, "type": "call", "quantity": 1e20, "notional": 1e20}],
        # Extremely short dated (T near zero or negative)
        [{"K": 100.0, "T": -1e-15, "type": "call", "quantity": 1.0}],
        # Option type case-insensitivity and invalid types
        [{"K": 100.0, "T": 0.5, "type": "CALL", "quantity": 1.0}],
        [{"K": 100.0, "T": 0.5, "type": "invalid_type", "quantity": 1.0}]
    ]
    
    for idx, port in enumerate(pathological_portfolios):
        try:
            res = portfolio_greeks(port, model, theta, pn, yn, S=100.0)
            print(f"Portfolio {idx} succeeded. total_delta={res['total_delta']}, total_gamma={res['total_gamma']}")
            for k, v in res.items():
                if k != "vega_bucket" and k != "hedge_contracts":
                    assert np.isfinite(v), f"Portfolio {idx} produced non-finite output for {k}: {v}"
                elif k == "vega_bucket":
                    assert np.all(np.isfinite(v)), f"Portfolio {idx} produced non-finite vega bucket: {v}"
        except Exception as e:
            # Let's check if it's the expected ValueError for invalid type
            if idx == 7 and isinstance(e, ValueError) and "Unsupported option type" in str(e):
                print(f"Portfolio {idx} correctly raised ValueError for invalid option type")
            else:
                pytest.fail(f"Portfolio {idx} failed with unexpected exception: {type(e).__name__}: {str(e)}")


def test_differentiability_robustness(fno_v2_model):
    """Verify that portfolio_price_tensor autograd does not produce NaNs/Infs under extreme parameters."""
    model = fno_v2_model
    device = next(model.parameters()).device
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    
    # Inputs that could trigger autograd NaNs/Infs
    S_t = torch.tensor(100.0, dtype=torch.float32, device=device, requires_grad=True)
    theta_t = torch.tensor([2.5, 0.08, 0.5, -0.5, 0.08, 0.08], dtype=torch.float32, device=device, requires_grad=True)
    r_t = torch.tensor(0.05, dtype=torch.float32, device=device)
    
    # Portfolio containing extreme / boundary tenors
    positions = [
        {"K": 100.0, "T": 1e-15, "type": "call", "notional": 100.0, "quantity": 1.0},
        {"K": 95.0,  "T": 2.5, "type": "put",  "notional": 100.0, "quantity": 1.0} # K log-moneyness out of bounds
    ]
    
    price = portfolio_price_tensor(positions, model, theta_t, pn, yn, S_t, r_t)
    
    # Compute gradients
    grad_S = torch.autograd.grad(price, S_t, create_graph=True, retain_graph=True)[0]
    grad_theta = torch.autograd.grad(price, theta_t, retain_graph=False)[0]
    
    print(f"\n[Autograd Robustness] price={price.item()}, grad_S={grad_S.item()}, grad_theta={grad_theta.cpu().numpy()}")
    
    assert torch.isfinite(price), f"Price is not finite: {price}"
    assert torch.isfinite(grad_S), f"grad_S is not finite: {grad_S}"
    assert torch.all(torch.isfinite(grad_theta)), f"grad_theta is not finite: {grad_theta}"


def test_detailed_performance_benchmark(fno_v2_model):
    """Detailed benchmarking of portfolio aggregation latency for different sizes."""
    model = fno_v2_model
    pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v2.npz")
    yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v2.npz")
    theta = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    
    sizes = [10, 50, 100, 500, 1000]
    np.random.seed(1337)
    
    for size in sizes:
        portfolio = []
        for _ in range(size):
            portfolio.append({
                "K": float(np.random.uniform(80, 120)),
                "T": float(np.random.uniform(0.05, 2.0)),
                "type": "call" if np.random.rand() > 0.5 else "put",
                "quantity": float(np.random.uniform(-5.0, 5.0)),
                "notional": 100.0
            })
            
        # Warmup
        for _ in range(5):
            _ = portfolio_greeks(portfolio, model, theta, pn, yn, S=100.0)
            
        latencies = []
        for _ in range(50):
            t_start = time.perf_counter()
            _ = portfolio_greeks(portfolio, model, theta, pn, yn, S=100.0)
            latencies.append((time.perf_counter() - t_start) * 1000.0)
            
        latencies = np.array(latencies)
        avg_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)
        p99_lat = np.percentile(latencies, 99)
        min_lat = np.min(latencies)
        max_lat = np.max(latencies)
        
        print(f"\n[Performance Benchmark - {size} positions]")
        print(f"  Avg Latency: {avg_lat:.2f} ms")
        print(f"  Min/Max:     {min_lat:.2f} / {max_lat:.2f} ms")
        print(f"  p95 / p99:   {p95_lat:.2f} / {p99_lat:.2f} ms")
        
        if size == 100:
            assert avg_lat < 100.0, f"Aggregation latency for 100 positions exceeded 100ms budget: {avg_lat:.2f} ms"
