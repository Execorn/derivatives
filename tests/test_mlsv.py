import pytest
import torch
import numpy as np
import math
from deepvol.models.mlsv_gpu import MLSVSolverGPU, compute_conditional_expectation


def test_mlsv_solver_initialization():
    # Valid initialization
    dup_vol_fn = lambda t, s: torch.full_like(s, 0.2)
    solver = MLSVSolverGPU(
        S0=100.0,
        r=0.05,
        q=0.02,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        T=0.5,
        steps_per_unit=100,
        N_paths=1000,
        dupire_vol_fn=dup_vol_fn,
        device="cpu",
    )
    assert solver.S0 == 100.0
    assert solver.device == "cpu"
    assert solver.N_steps == 50

    # Invalid initializations should raise ValueError
    with pytest.raises(ValueError):
        MLSVSolverGPU(
            S0=-100.0, r=0.05, q=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, T=0.5, steps_per_unit=100, N_paths=1000
        )
    with pytest.raises(ValueError):
        MLSVSolverGPU(
            S0=100.0, r=0.05, q=0.02, v0=-0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, T=0.5, steps_per_unit=100, N_paths=1000
        )
    with pytest.raises(ValueError):
        MLSVSolverGPU(
            S0=100.0, r=0.05, q=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=1.5, T=0.5, steps_per_unit=100, N_paths=1000
        )


def test_compute_conditional_expectation_cpu():
    device = "cpu"
    dtype = torch.float32
    
    # Setup dummy paths
    N_paths = 1000
    X_t = torch.randn(N_paths, device=device, dtype=dtype)
    V_t = 0.04 + 0.1 * torch.randn(N_paths, device=device, dtype=dtype).abs()
    
    # Test targets
    targets = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
    
    # Test Nadaraya-Watson
    est_nw = compute_conditional_expectation(
        X_t=X_t,
        V_t=V_t,
        targets=targets,
        method="nadaraya_watson",
        block_size=512,
    )
    assert est_nw.shape == (3,)
    assert not torch.isnan(est_nw).any()
    
    # Test Muguruza
    mu_i = X_t + 0.01
    sigma_cond = torch.full_like(X_t, 0.02)
    est_mug = compute_conditional_expectation(
        X_t=X_t,
        V_t=V_t,
        targets=targets,
        method="muguruza",
        mu_i=mu_i,
        sigma_cond=sigma_cond,
        block_size=512,
    )
    assert est_mug.shape == (3,)
    assert not torch.isnan(est_mug).any()


def test_mlsv_solver_simulation_cpu():
    dup_vol_fn = lambda t, s: torch.full_like(s, 0.2)
    solver = MLSVSolverGPU(
        S0=100.0,
        r=0.05,
        q=0.02,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        T=0.25,
        steps_per_unit=100,
        N_paths=500,
        dupire_vol_fn=dup_vol_fn,
        device="cpu",
    )
    
    # Simulate using Nadaraya-Watson
    solver.simulate(method="nadaraya_watson")
    assert solver.X_paths.shape == (26, 500)
    assert solver.V_paths.shape == (26, 500)
    assert (solver.V_paths >= 1e-6).all()
    
    # Price option
    call_price = solver.price_european_option(strike=100.0, maturity=0.25, is_call=True)
    put_price = solver.price_european_option(strike=100.0, maturity=0.25, is_call=False)
    assert call_price > 0.0
    assert put_price > 0.0
    
    # Simulate using Muguruza
    solver.simulate(method="muguruza")
    assert solver.X_paths.shape == (26, 500)
    assert solver.V_paths.shape == (26, 500)
    assert (solver.V_paths >= 1e-6).all()
    
    # Price option
    call_price_mug = solver.price_european_option(strike=100.0, maturity=0.25, is_call=True)
    assert call_price_mug > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mlsv_solver_simulation_gpu():
    dup_vol_fn = lambda t, s: torch.full_like(s, 0.2)
    solver = MLSVSolverGPU(
        S0=100.0,
        r=0.05,
        q=0.02,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        T=0.25,
        steps_per_unit=100,
        N_paths=2000,
        dupire_vol_fn=dup_vol_fn,
        device="cuda",
    )
    
    solver.simulate(method="muguruza", block_size=1024)
    assert solver.X_paths.is_cuda
    assert solver.V_paths.is_cuda
    
    prices = solver.price_european_option(
        strike=np.array([90.0, 100.0, 110.0]),
        maturity=0.25,
    )
    assert len(prices) == 3
    assert (prices > 0.0).all()


def test_mlsv_call_put_parity():
    # Test for both NW and Muguruza methods
    for method in ["nadaraya_watson", "muguruza"]:
        solver = MLSVSolverGPU(
            S0=100.0,
            r=0.05,
            q=0.02,
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            T=0.5,
            steps_per_unit=100,
            N_paths=5000,
            device="cpu",
        )
        solver.simulate(method=method)
        
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        calls = solver.price_european_option(strike=strikes, maturity=0.5, is_call=True)
        puts = solver.price_european_option(strike=strikes, maturity=0.5, is_call=False)
        
        # 1. Pathwise parity check: C - P = e^{-r T} * (mean(S_T) - K)
        # Extract stock prices at maturity
        S_T = torch.exp(solver.X_paths[-1])
        mean_S_T = S_T.mean().item()
        
        for idx, K in enumerate(strikes):
            C = calls[idx].item()
            P = puts[idx].item()
            pathwise_parity_rhs = math.exp(-solver.r * solver.T) * (mean_S_T - K)
            assert abs((C - P) - pathwise_parity_rhs) < 5e-5, f"Pathwise call-put parity violated for K={K}, method={method}"
            
        # 2. Statistical parity check: C - P = S_0 e^{-q T} - K e^{-r T} within statistical error tolerance
        # Standard error of discounted stock price: SE = std(e^{-r T} * S_T) / sqrt(N_paths)
        discounted_S_T = math.exp(-solver.r * solver.T) * S_T
        std_error = (discounted_S_T.std() / math.sqrt(solver.N_paths)).item()
        
        for idx, K in enumerate(strikes):
            C = calls[idx].item()
            P = puts[idx].item()
            theoretical_parity = solver.S0 * math.exp(-solver.q * solver.T) - K * math.exp(-solver.r * solver.T)
            diff = abs((C - P) - theoretical_parity)
            z_score = diff / std_error
            # Z-score of 3.5 corresponds to ~99.95% confidence level.
            # Since Monte Carlo paths are simulated, they should be well within statistical error.
            assert z_score < 3.5, f"Statistical call-put parity failed for K={K}, method={method}, z-score={z_score:.2f}"


def test_mlsv_pricing_monotonicity():
    for method in ["nadaraya_watson", "muguruza"]:
        solver = MLSVSolverGPU(
            S0=100.0,
            r=0.05,
            q=0.02,
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            T=0.5,
            steps_per_unit=100,
            N_paths=2000,
            device="cpu",
        )
        solver.simulate(method=method)
        
        # Use a narrower strike grid around S0 to ensure non-zero option prices
        strikes = np.linspace(85.0, 115.0, 15)
        calls = solver.price_european_option(strike=strikes, maturity=0.5, is_call=True)
        puts = solver.price_european_option(strike=strikes, maturity=0.5, is_call=False)
        
        # Verify call prices are strictly decreasing and put prices are strictly increasing
        for i in range(len(strikes) - 1):
            assert calls[i] > calls[i+1], f"Call price is not strictly decreasing: {calls[i]} <= {calls[i+1]} at strike index {i}"
            assert puts[i] < puts[i+1], f"Put price is not strictly increasing: {puts[i]} >= {puts[i+1]} at strike index {i}"


def test_mlsv_extreme_parameters():
    # Test high vol-of-vol, high/extreme correlations, and 0 dividend yield
    scenarios = [
        # xi, rho, q
        {"xi": 1.0, "rho": 0.9, "q": 0.0},
        {"xi": 1.0, "rho": -0.9, "q": 0.0},
        # Also test extreme boundary correlations
        {"xi": 0.5, "rho": 1.0, "q": 0.01},
        {"xi": 0.5, "rho": -1.0, "q": 0.01},
    ]
    
    for idx, params in enumerate(scenarios):
        for method in ["nadaraya_watson", "muguruza"]:
            solver = MLSVSolverGPU(
                S0=100.0,
                r=0.04,
                q=params["q"],
                v0=0.04,
                kappa=2.0,
                theta=0.04,
                xi=params["xi"],
                rho=params["rho"],
                T=0.25,
                steps_per_unit=100,
                N_paths=1000,
                device="cpu",
            )
            # Verify simulation runs without NaN
            solver.simulate(method=method)
            assert not torch.isnan(solver.X_paths).any(), f"NaNs found in X_paths for scenario {idx}, method={method}"
            assert not torch.isnan(solver.V_paths).any(), f"NaNs found in V_paths for scenario {idx}, method={method}"
            
            # Check price values
            strikes = np.array([90.0, 100.0, 110.0])
            calls = solver.price_european_option(strike=strikes, maturity=0.25, is_call=True)
            puts = solver.price_european_option(strike=strikes, maturity=0.25, is_call=False)
            
            assert (calls > 0.0).all(), f"Some calls are non-positive in scenario {idx}, method={method}"
            assert (puts > 0.0).all(), f"Some puts are non-positive in scenario {idx}, method={method}"
            
            # Check monotonicity
            assert calls[0] > calls[1] > calls[2]
            assert puts[0] < puts[1] < puts[2]


def test_mlsv_extreme_maturities():
    maturities = [0.01, 3.0]
    for T in maturities:
        for method in ["nadaraya_watson", "muguruza"]:
            solver = MLSVSolverGPU(
                S0=100.0,
                r=0.05,
                q=0.02,
                v0=0.04,
                kappa=2.0,
                theta=0.04,
                xi=0.3,
                rho=-0.7,
                T=T,
                steps_per_unit=200 if T == 0.01 else 50,  # Ensure at least some steps for short, and reasonable for long
                N_paths=1000,
                device="cpu",
            )
            solver.simulate(method=method)
            
            assert not torch.isnan(solver.X_paths).any()
            assert not torch.isnan(solver.V_paths).any()
            
            # Verify options can be priced
            c = solver.price_european_option(strike=100.0, maturity=T, is_call=True)
            p = solver.price_european_option(strike=100.0, maturity=T, is_call=False)
            assert c > 0.0
            assert p > 0.0


def test_mlsv_oom_safety():
    # Large strike grid (200 strikes) and 10,000 paths
    # Verify that block tiling works and does not throw allocation errors on CPU
    for method in ["nadaraya_watson", "muguruza"]:
        solver = MLSVSolverGPU(
            S0=100.0,
            r=0.05,
            q=0.02,
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            T=0.1,
            steps_per_unit=100,  # 10 steps
            N_paths=10000,
            device="cpu",
        )
        # Using a small block size to force multiple tiles/chunks to execute
        solver.simulate(method=method, block_size=512)
        
        # Price on a large grid of 200 strikes
        strikes = np.linspace(50.0, 150.0, 200)
        calls = solver.price_european_option(strike=strikes, maturity=0.1, is_call=True)
        puts = solver.price_european_option(strike=strikes, maturity=0.1, is_call=False)
        
        assert len(calls) == 200
        assert len(puts) == 200
        assert (calls > 0.0).any()
        assert (puts > 0.0).any()

