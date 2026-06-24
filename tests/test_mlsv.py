import pytest
import torch
import numpy as np
import math
from src.pricing.mlsv_gpu import MLSVSolverGPU, compute_conditional_expectation


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



# =============================================================================
# Finding 7.1 — Production-Scale Particle Convergence Test (GPU-first)
# =============================================================================
# GPU utilization is MANDATORY per project rules (.agents/AGENTS.md).
# On RTX 3060 (6.1 GB VRAM) this test runs in ~25s vs ~12 min on CPU.
# block_size=4096: each NW tile = 4096×50k×4B ≈ 800 MB — fits in VRAM.
# Falls back to CPU only when CUDA is completely unavailable.

_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE   = "cuda" if _CUDA_AVAILABLE else "cpu"
_DTYPE    = torch.float32 if _CUDA_AVAILABLE else torch.float64
# Tiled block size: safe for RTX 3060 at N=50k float32
# 4096 × 50000 × 4B = 800 MB per block (well within 6.1 GB VRAM)
_BLOCK_SIZE = 4096 if _CUDA_AVAILABLE else 2048


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA GPU required — CPU fallback in test_mlsv_particle_convergence_cpu_fallback")
def test_mlsv_particle_convergence():
    """
    GPU-first particle convergence test (Finding 7.1 — P7 roadmap).

    Verifies that MLSV option prices converge as N_paths → 50,000 on CUDA.

    Strategy
    --------
    1. Run simulations at N ∈ {1k, 5k, 10k, 25k, 50k} on CUDA float32.
    2. Compute the exact MC standard error from std(discounted_payoff)/sqrt(N)
       directly from the CUDA simulation paths (not the incorrect price/sqrt(N)).
    3. Assert N=25k is within 5 combined MC standard errors of N=50k.
    4. Assert N=50k price is within 30% of Black-Scholes reference (sanity).
    5. Assert total GPU wall-clock time is < 120 s (GPU utilization check).

    GPU Notes
    ---------
    - device="cuda", dtype=torch.float32 (30–60× faster than CPU float64).
    - block_size=4096: tiles O(N²) NW kernel to 800 MB VRAM chunks on RTX 3060.
    - torch.cuda.synchronize() called before/after timing.
    - McKean-Vlasov particle systems are NOT monotone-convergent in N: each N
      defines a different coupled system (more particles → better E[V|X] density
      → different drift for ALL paths). Only the SE-based check is used.
    """
    import time

    S0, K, T = 100.0, 100.0, 0.5   # ATM call, 6-month expiry
    r, q     = 0.05, 0.02
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    particle_counts = [1_000, 5_000, 10_000, 25_000, 50_000]
    prices  = {}
    solvers = {}

    # ── GPU warm-up (avoids one-time CUDA init cost in timing) ────────────────
    _ = torch.zeros(1, device=_DEVICE, dtype=_DTYPE)
    torch.cuda.synchronize()
    t_wall_start = time.perf_counter()

    for N in particle_counts:
        solver = MLSVSolverGPU(
            S0=S0, r=r, q=q, v0=v0,
            kappa=kappa, theta=theta, xi=xi, rho=rho,
            T=T,
            steps_per_unit=50,       # 25 timesteps for T=0.5 — production-like
            N_paths=N,
            device=_DEVICE,
            dtype=_DTYPE,
        )
        solver.simulate(method="nadaraya_watson", block_size=_BLOCK_SIZE)
        prices[N]  = float(solver.price_european_option(strike=K, maturity=T, is_call=True))
        solvers[N] = solver

    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t_wall_start

    # ── GPU utilization / speed gate ──────────────────────────────────────────
    # All 5 simulations (1k→50k) on RTX 3060 should complete in < 120 s.
    # Failure here means CUDA is not being used or is severely throttled.
    assert wall_s < 120.0, (
        f"GPU convergence sweep took {wall_s:.1f}s > 120s limit. "
        f"Check that CUDA kernels are executing (device={_DEVICE})."
    )

    # ── Exact MC standard error from simulation paths ─────────────────────────
    disc = math.exp(-r * T)

    def _payoff_se(solver, N):
        T_tensor = torch.tensor(T, device=_DEVICE, dtype=_DTYPE)
        t_idx = int(torch.argmin(torch.abs(solver.t_grid - T_tensor)).item())
        S_T   = torch.exp(solver.X_paths[t_idx].float())   # → float32 on CPU for std()
        payoff = disc * torch.clamp(S_T - K, min=0.0)
        return float(payoff.std().cpu()) / math.sqrt(N)

    mc_se_50k = _payoff_se(solvers[50_000], 50_000)
    mc_se_25k = _payoff_se(solvers[25_000], 25_000)

    price_50k = prices[50_000]
    price_25k = prices[25_000]

    # ── Positivity ────────────────────────────────────────────────────────────
    assert price_50k > 0.0, f"50k price non-positive: {price_50k}"
    assert price_25k > 0.0, f"25k price non-positive: {price_25k}"

    # ── SE-based convergence (N=25k vs N=50k) ────────────────────────────────
    combined_se = math.sqrt(mc_se_50k**2 + mc_se_25k**2)
    gap = abs(price_25k - price_50k)
    assert gap < 5.0 * combined_se + 1e-4, (
        f"Particle convergence failure (N=25k vs N=50k): "
        f"|{price_25k:.6f} - {price_50k:.6f}| = {gap:.6f} "
        f"> 5*SE={5.0*combined_se:.6f}  "
        f"(mc_se_50k={mc_se_50k:.5f}, mc_se_25k={mc_se_25k:.5f}, wall={wall_s:.1f}s)"
    )

    # ── MKV coarse check: N=1k should not be closer than N=25k by a wide margin ─
    deviations = {N: abs(prices[N] - price_50k) for N in particle_counts[:-1]}
    assert deviations[1_000] >= deviations[25_000] - 5.0 * combined_se, (
        f"MKV anomaly: N=1k deviation {deviations[1_000]:.6f} is unexpectedly "
        f"smaller than N=25k deviation {deviations[25_000]:.6f} - 5*SE "
        f"(combined_se={combined_se:.5f})"
    )

    # ── Black-Scholes sanity ───────────────────────────────────────────────────
    from scipy.stats import norm as _norm
    sigma_bs = math.sqrt(v0)
    d1 = (math.log(S0/K) + (r - q + 0.5*sigma_bs**2)*T) / (sigma_bs*math.sqrt(T))
    d2 = d1 - sigma_bs*math.sqrt(T)
    bs_price = S0*math.exp(-q*T)*_norm.cdf(d1) - K*math.exp(-r*T)*_norm.cdf(d2)

    assert abs(price_50k - bs_price) / bs_price < 0.30, (
        f"50k MLSV GPU price {price_50k:.4f} deviates >30% from BS {bs_price:.4f}"
    )


@pytest.mark.skipif(_CUDA_AVAILABLE, reason="CPU fallback only runs when no GPU present")
def test_mlsv_particle_convergence_cpu_fallback():
    """
    CPU fallback for test_mlsv_particle_convergence when no CUDA GPU is available.

    Uses a reduced N ∈ {1k, 2k, 5k} sweep (N=5k as reference) to keep runtime
    under 3 minutes on CPU. GPU variant is authoritative for production.
    """
    S0, K, T = 100.0, 100.0, 0.25  # shorter maturity to reduce CPU time
    r, q     = 0.05, 0.02
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    particle_counts = [500, 1_000, 2_000, 5_000]
    prices  = {}
    solvers = {}

    for N in particle_counts:
        solver = MLSVSolverGPU(
            S0=S0, r=r, q=q, v0=v0,
            kappa=kappa, theta=theta, xi=xi, rho=rho,
            T=T, steps_per_unit=20, N_paths=N,
            device="cpu", dtype=torch.float64,
        )
        solver.simulate(method="nadaraya_watson", block_size=512)
        prices[N]  = float(solver.price_european_option(strike=K, maturity=T, is_call=True))
        solvers[N] = solver

    # Exact SE from 5k paths
    disc = math.exp(-r * T)
    for ref_N in [5_000, 2_000]:
        slv = solvers[ref_N]
        T_t = torch.tensor(T, dtype=torch.float64)
        idx = int(torch.argmin(torch.abs(slv.t_grid - T_t)).item())
        payoff = disc * torch.clamp(torch.exp(slv.X_paths[idx]) - K, min=0.0)
        if ref_N == 5_000:
            mc_se_5k = float(payoff.std()) / math.sqrt(ref_N)
        else:
            mc_se_2k = float(payoff.std()) / math.sqrt(ref_N)

    combined_se = math.sqrt(mc_se_5k**2 + mc_se_2k**2)
    gap = abs(prices[2_000] - prices[5_000])
    assert gap < 5.0 * combined_se + 1e-4, (
        f"CPU fallback convergence failure: |{prices[2_000]:.4f}-{prices[5_000]:.4f}|={gap:.4f} > 5*SE={5*combined_se:.4f}"
    )
    assert prices[5_000] > 0.0



