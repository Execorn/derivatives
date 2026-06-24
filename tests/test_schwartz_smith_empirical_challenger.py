import pytest
import numpy as np
import torch
import time
from deepvol.models.schwartz_smith import (
    schwartz_smith_price_black76,
    schwartz_smith_price_black76_pt,
    schwartz_smith_price_cos,
    schwartz_smith_price_cos_pt,
    futures_price
)

# Standard base parameters for tests
BASE_PARAMS = {
    "kappa": 0.6,
    "sigma_chi": 0.25,
    "rho": 0.35,
    "sigma_xi": 0.12,
    "mu_star": 0.02,
    "lambda_chi": 0.05,
    "chi_t": 0.1,
    "xi_t": np.log(75.0),
    "t": 0.0,
    "r": 0.04,
    "K": 75.0,
    "T_opt": 0.5,
    "T_fut": 1.0
}

def test_challenger_extreme_short_maturity():
    """Verify stability under extremely short maturities down to 1e-12."""
    p = BASE_PARAMS.copy()
    maturities = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    
    for T_opt in maturities:
        p["T_opt"] = T_opt
        p["T_fut"] = T_opt + 0.1 # futures maturity slightly after option
        
        for otype in ["C", "P"]:
            # CPU
            p_black = schwartz_smith_price_black76(
                p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], p["chi_t"], p["xi_t"],
                p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
                option_type=otype
            )
            p_cos = schwartz_smith_price_cos(
                p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], p["chi_t"], p["xi_t"],
                p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
                option_type=otype, N=256, L=10.0
            )
            
            assert np.isfinite(p_cos), f"CPU COS price non-finite for T_opt={T_opt}, type={otype}"
            assert p_cos >= 0.0, f"CPU COS price negative ({p_cos}) for T_opt={T_opt}, type={otype}"
            # For extremely short maturity, price should be extremely close to Black-76
            assert np.isclose(p_cos, p_black, rtol=1e-5, atol=1e-5), \
                f"CPU COS/Black mismatch at T_opt={T_opt}, type={otype}: Black={p_black}, COS={p_cos}"

            # PyTorch (CPU and GPU)
            devices = [torch.device("cpu")]
            if torch.cuda.is_available():
                devices.append(torch.device("cuda"))
                
            for dev in devices:
                chi_pt = torch.tensor([p["chi_t"]], dtype=torch.float64, device=dev)
                xi_pt = torch.tensor([p["xi_t"]], dtype=torch.float64, device=dev)
                
                p_black_pt = schwartz_smith_price_black76_pt(
                    p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], chi_pt, xi_pt,
                    p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
                    option_type=otype
                )
                p_cos_pt = schwartz_smith_price_cos_pt(
                    p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], chi_pt, xi_pt,
                    p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
                    option_type=otype, N=256, L=10.0
                )
                
                assert torch.isfinite(p_cos_pt).all(), f"PyTorch {dev} COS price non-finite for T_opt={T_opt}"
                assert (p_cos_pt >= 0.0).all(), f"PyTorch {dev} COS price negative for T_opt={T_opt}"
                assert np.isclose(p_cos_pt.item(), p_black_pt.item(), rtol=1e-5, atol=1e-5), \
                    f"PyTorch {dev} COS/Black mismatch at T_opt={T_opt}: Black={p_black_pt.item()}, COS={p_cos_pt.item()}"


def test_challenger_extreme_volatilities():
    """Verify stability under extreme volatilities up to 2.0 and down to 0.0."""
    p = BASE_PARAMS.copy()
    vols = [
        (0.0, 0.0),       # Zero vol
        (1e-8, 1e-8),     # Near-zero vol
        (1e-5, 1e-5),     # Low vol
        (0.5, 0.5),       # Moderate vol
        (1.0, 1.0),       # High vol
        (2.0, 2.0),       # Extreme high vol (requested bound)
        (2.0, 0.05),      # Highly asymmetric vol (factor 1 dominant)
        (0.05, 2.0),      # Highly asymmetric vol (factor 2 dominant)
    ]
    
    for sig_chi, sig_xi in vols:
        for otype in ["C", "P"]:
            # CPU
            p_black = schwartz_smith_price_black76(
                p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], p["chi_t"], p["xi_t"],
                p["kappa"], sig_chi, p["rho"], sig_xi, p["mu_star"], p["lambda_chi"],
                option_type=otype
            )
            p_cos = schwartz_smith_price_cos(
                p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], p["chi_t"], p["xi_t"],
                p["kappa"], sig_chi, p["rho"], sig_xi, p["mu_star"], p["lambda_chi"],
                option_type=otype, N=256, L=10.0
            )
            
            assert np.isfinite(p_cos), f"CPU COS price non-finite for sig_chi={sig_chi}, sig_xi={sig_xi}"
            assert p_cos >= 0.0, f"CPU COS price negative ({p_cos}) for sig_chi={sig_chi}, sig_xi={sig_xi}"
            
            rtol = 1e-4 if max(sig_chi, sig_xi) >= 2.0 else 1e-5
            atol = 1e-4 if max(sig_chi, sig_xi) >= 2.0 else 1e-5
            assert np.isclose(p_cos, p_black, rtol=rtol, atol=atol), \
                f"CPU COS/Black mismatch at vol=({sig_chi},{sig_xi}), type={otype}: Black={p_black}, COS={p_cos}"

            # PyTorch
            devices = [torch.device("cpu")]
            if torch.cuda.is_available():
                devices.append(torch.device("cuda"))
                
            for dev in devices:
                chi_pt = torch.tensor([p["chi_t"]], dtype=torch.float64, device=dev)
                xi_pt = torch.tensor([p["xi_t"]], dtype=torch.float64, device=dev)
                
                p_black_pt = schwartz_smith_price_black76_pt(
                    p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], chi_pt, xi_pt,
                    p["kappa"], sig_chi, p["rho"], sig_xi, p["mu_star"], p["lambda_chi"],
                    option_type=otype
                )
                p_cos_pt = schwartz_smith_price_cos_pt(
                    p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], chi_pt, xi_pt,
                    p["kappa"], sig_chi, p["rho"], sig_xi, p["mu_star"], p["lambda_chi"],
                    option_type=otype, N=256, L=10.0
                )
                
                assert torch.isfinite(p_cos_pt).all(), f"PyTorch {dev} COS price non-finite for vols=({sig_chi},{sig_xi})"
                assert (p_cos_pt >= 0.0).all(), f"PyTorch {dev} COS price negative for vols=({sig_chi},{sig_xi})"
                assert np.isclose(p_cos_pt.item(), p_black_pt.item(), rtol=rtol, atol=atol), \
                    f"PyTorch {dev} COS/Black mismatch at vols=({sig_chi},{sig_xi}): Black={p_black_pt.item()}, COS={p_cos_pt.item()}"


def test_challenger_extreme_correlations():
    """Verify stability under extreme correlation parameters +/- 0.999, +/- 0.9999, and +/- 1.0."""
    p = BASE_PARAMS.copy()
    correlations = [-1.0, -0.9999, -0.999, 0.0, 0.999, 0.9999, 1.0]
    
    for rho in correlations:
        for otype in ["C", "P"]:
            # CPU
            p_black = schwartz_smith_price_black76(
                p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], p["chi_t"], p["xi_t"],
                p["kappa"], p["sigma_chi"], rho, p["sigma_xi"], p["mu_star"], p["lambda_chi"],
                option_type=otype
            )
            p_cos = schwartz_smith_price_cos(
                p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], p["chi_t"], p["xi_t"],
                p["kappa"], p["sigma_chi"], rho, p["sigma_xi"], p["mu_star"], p["lambda_chi"],
                option_type=otype, N=256, L=12.0
            )
            
            assert np.isfinite(p_cos), f"CPU COS price non-finite for rho={rho}"
            assert p_cos >= 0.0, f"CPU COS price negative ({p_cos}) for rho={rho}"
            assert np.isclose(p_cos, p_black, rtol=1e-5, atol=1e-5), \
                f"CPU COS/Black mismatch at rho={rho}, type={otype}: Black={p_black}, COS={p_cos}"

            # PyTorch
            devices = [torch.device("cpu")]
            if torch.cuda.is_available():
                devices.append(torch.device("cuda"))
                
            for dev in devices:
                chi_pt = torch.tensor([p["chi_t"]], dtype=torch.float64, device=dev)
                xi_pt = torch.tensor([p["xi_t"]], dtype=torch.float64, device=dev)
                
                p_black_pt = schwartz_smith_price_black76_pt(
                    p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], chi_pt, xi_pt,
                    p["kappa"], p["sigma_chi"], rho, p["sigma_xi"], p["mu_star"], p["lambda_chi"],
                    option_type=otype
                )
                p_cos_pt = schwartz_smith_price_cos_pt(
                    p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], chi_pt, xi_pt,
                    p["kappa"], p["sigma_chi"], rho, p["sigma_xi"], p["mu_star"], p["lambda_chi"],
                    option_type=otype, N=256, L=12.0
                )
                
                assert torch.isfinite(p_cos_pt).all(), f"PyTorch {dev} COS price non-finite for rho={rho}"
                assert (p_cos_pt >= 0.0).all(), f"PyTorch {dev} COS price negative for rho={rho}"
                assert np.isclose(p_cos_pt.item(), p_black_pt.item(), rtol=1e-5, atol=1e-5), \
                    f"PyTorch {dev} COS/Black mismatch at rho={rho}: Black={p_black_pt.item()}, COS={p_cos_pt.item()}"


def test_challenger_convergence_high_N():
    """Test convergence rates as N goes from 32 to 1024."""
    p = BASE_PARAMS.copy()
    Ns = [32, 64, 128, 256, 512, 1024]
    
    # We will print or check that error strictly decreases or remains extremely small
    for otype in ["C", "P"]:
        p_black = schwartz_smith_price_black76(
            p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], p["chi_t"], p["xi_t"],
            p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
            option_type=otype
        )
        
        errors = []
        for N in Ns:
            p_cos = schwartz_smith_price_cos(
                p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], p["chi_t"], p["xi_t"],
                p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
                option_type=otype, N=N, L=10.0
            )
            err = abs(p_cos - p_black)
            errors.append(err)
            
        # Check that error is extremely small for large N
        assert errors[-1] < 1e-7, f"COS failed to converge to Black-76, error={errors[-1]} for N=1024"
        # Check that error generally decreases or is at machine precision
        # Since COS converges exponentially, by N=128 the error is usually very small (< 1e-7).
        assert errors[3] < 1e-6, f"COS error at N=256 is too high: {errors[3]}"


def test_challenger_gpu_correctness_and_speedup():
    """Verify GPU correct results vs CPU, speedup, and device mismatch resilience."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    device = torch.device("cuda")
    p = BASE_PARAMS.copy()
    
    # Check correctness
    chi_pt_cpu = torch.tensor([p["chi_t"]], dtype=torch.float64)
    xi_pt_cpu = torch.tensor([p["xi_t"]], dtype=torch.float64)
    
    chi_pt_gpu = chi_pt_cpu.to(device)
    xi_pt_gpu = xi_pt_cpu.to(device)
    
    p_cpu = schwartz_smith_price_cos_pt(
        p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], chi_pt_cpu, xi_pt_cpu,
        p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"]
    )
    p_gpu = schwartz_smith_price_cos_pt(
        p["t"], p["T_opt"], p["T_fut"], p["K"], p["r"], chi_pt_gpu, xi_pt_gpu,
        p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"]
    )
    
    assert p_gpu.device.type == "cuda"
    assert np.isclose(p_cpu.item(), p_gpu.item(), rtol=1e-7, atol=1e-7), "CPU and GPU option prices do not match!"
    
    # Test batch speedup
    size = 15000
    np.random.seed(42)
    chi_arr = np.random.uniform(-0.3, 0.3, size)
    xi_arr = np.random.uniform(np.log(60.0), np.log(90.0), size)
    T_opts = np.random.uniform(0.1, 1.5, size)
    T_futs = T_opts + np.random.uniform(0.1, 0.5, size)
    Ks = np.random.uniform(60.0, 90.0, size)
    
    chi_cpu_b = torch.tensor(chi_arr, dtype=torch.float64)
    xi_cpu_b = torch.tensor(xi_arr, dtype=torch.float64)
    T_opt_cpu_b = torch.tensor(T_opts, dtype=torch.float64)
    T_fut_cpu_b = torch.tensor(T_futs, dtype=torch.float64)
    K_cpu_b = torch.tensor(Ks, dtype=torch.float64)
    
    chi_gpu_b = chi_cpu_b.to(device)
    xi_gpu_b = xi_cpu_b.to(device)
    T_opt_gpu_b = T_opt_cpu_b.to(device)
    T_fut_gpu_b = T_fut_cpu_b.to(device)
    K_gpu_b = K_cpu_b.to(device)
    
    # Warmup
    _ = schwartz_smith_price_cos_pt(
        0.0, T_opt_gpu_b[:100], T_fut_gpu_b[:100], K_gpu_b[:100], p["r"],
        chi_gpu_b[:100], xi_gpu_b[:100], p["kappa"], p["sigma_chi"], p["rho"],
        p["sigma_xi"], p["mu_star"], p["lambda_chi"]
    )
    torch.cuda.synchronize()
    
    # Time CPU
    t0 = time.perf_counter()
    _ = schwartz_smith_price_cos_pt(
        0.0, T_opt_cpu_b, T_fut_cpu_b, K_cpu_b, p["r"],
        chi_cpu_b, xi_cpu_b, p["kappa"], p["sigma_chi"], p["rho"],
        p["sigma_xi"], p["mu_star"], p["lambda_chi"]
    )
    t_cpu = time.perf_counter() - t0
    
    # Time GPU
    t1 = time.perf_counter()
    _ = schwartz_smith_price_cos_pt(
        0.0, T_opt_gpu_b, T_fut_gpu_b, K_gpu_b, p["r"],
        chi_gpu_b, xi_gpu_b, p["kappa"], p["sigma_chi"], p["rho"],
        p["sigma_xi"], p["mu_star"], p["lambda_chi"]
    )
    torch.cuda.synchronize()
    t_gpu = time.perf_counter() - t1
    
    print(f"\nBatch CPU Time for {size} items: {t_cpu:.5f}s")
    print(f"Batch GPU Time for {size} items: {t_gpu:.5f}s")
    print(f"GPU Speedup: {t_cpu / t_gpu:.2f}x")
    
    assert t_gpu < t_cpu, "GPU is not faster than CPU for large batch size!"
    
    # Verify Device Mismatch Resilience
    # Call PyTorch pricing where chi_t is on GPU, but K is on CPU.
    # The function should cast them internally without raising RuntimeError.
    mixed_price = schwartz_smith_price_cos_pt(
        t=0.0, T_opt=p["T_opt"], T_fut=p["T_fut"], K=torch.tensor([p["K"]], dtype=torch.float64, device="cpu"), 
        r=p["r"], chi_t=chi_gpu_b[:1], xi_t=xi_gpu_b[:1], kappa=p["kappa"], 
        sigma_chi=p["sigma_chi"], rho=p["rho"], sigma_xi=p["sigma_xi"], 
        mu_star=p["mu_star"], lambda_chi=p["lambda_chi"]
    )
    assert mixed_price.device.type == "cuda", "Output of mixed device execution should be on the same device as chi_t"


def test_challenger_monotonicity_and_convexity():
    """Verify monotonicity and convexity of options prices with respect to strike K and maturity T_opt."""
    p = BASE_PARAMS.copy()
    
    # 1. Monotonicity and convexity w.r.t Strike K
    strikes = np.linspace(40.0, 110.0, 100)
    
    # CPU Check
    call_prices_cpu = []
    put_prices_cpu = []
    for K in strikes:
        c_p = schwartz_smith_price_cos(
            p["t"], p["T_opt"], p["T_fut"], K, p["r"], p["chi_t"], p["xi_t"],
            p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
            option_type="C", N=256, L=10.0
        )
        p_p = schwartz_smith_price_cos(
            p["t"], p["T_opt"], p["T_fut"], K, p["r"], p["chi_t"], p["xi_t"],
            p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
            option_type="P", N=256, L=10.0
        )
        call_prices_cpu.append(c_p)
        put_prices_cpu.append(p_p)
        
    call_prices_cpu = np.array(call_prices_cpu)
    put_prices_cpu = np.array(put_prices_cpu)
    
    # Call option should decrease/non-increase with K: diff <= 0
    assert np.all(np.diff(call_prices_cpu) <= 1e-7), "CPU Call option price not non-increasing with strike K"
    # Put option should increase/non-decrease with K: diff >= 0
    assert np.all(np.diff(put_prices_cpu) >= -1e-7), "CPU Put option price not non-decreasing with strike K"
    
    # Convexity check: second diff >= 0
    assert np.all(np.diff(call_prices_cpu, n=2) >= -1e-9), "CPU Call option price not convex in strike K"
    assert np.all(np.diff(put_prices_cpu, n=2) >= -1e-9), "CPU Put option price not convex in strike K"

    # PyTorch Check (CPU and GPU)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        
    for dev in devices:
        chi_pt = torch.tensor([p["chi_t"]], dtype=torch.float64, device=dev)
        xi_pt = torch.tensor([p["xi_t"]], dtype=torch.float64, device=dev)
        K_pt = torch.tensor(strikes, dtype=torch.float64, device=dev)
        
        call_prices_pt = schwartz_smith_price_cos_pt(
            p["t"], p["T_opt"], p["T_fut"], K_pt, p["r"], chi_pt, xi_pt,
            p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
            option_type="C", N=256, L=10.0
        )
        put_prices_pt = schwartz_smith_price_cos_pt(
            p["t"], p["T_opt"], p["T_fut"], K_pt, p["r"], chi_pt, xi_pt,
            p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
            option_type="P", N=256, L=10.0
        )
        
        c_diff = torch.diff(call_prices_pt)
        p_diff = torch.diff(put_prices_pt)
        assert torch.all(c_diff <= 1e-7), f"PyTorch {dev} Call option price not non-increasing in K"
        assert torch.all(p_diff >= -1e-7), f"PyTorch {dev} Put option price not non-decreasing in K"
        
        c_diff2 = torch.diff(call_prices_pt, n=2)
        p_diff2 = torch.diff(put_prices_pt, n=2)
        assert torch.all(c_diff2 >= -1e-9), f"PyTorch {dev} Call option price not convex in K"
        assert torch.all(p_diff2 >= -1e-9), f"PyTorch {dev} Put option price not convex in K"

    # 2. Monotonicity w.r.t Maturity T_opt (when r = 0, fixed futures maturity)
    maturities = np.linspace(0.05, 0.95, 50)
    T_fut_fixed = 1.0
    r_zero = 0.0
    
    # CPU Check
    call_prices_mat_cpu = []
    put_prices_mat_cpu = []
    for T_opt in maturities:
        c_p = schwartz_smith_price_cos(
            p["t"], T_opt, T_fut_fixed, p["K"], r_zero, p["chi_t"], p["xi_t"],
            p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
            option_type="C", N=256, L=10.0
        )
        p_p = schwartz_smith_price_cos(
            p["t"], T_opt, T_fut_fixed, p["K"], r_zero, p["chi_t"], p["xi_t"],
            p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
            option_type="P", N=256, L=10.0
        )
        call_prices_mat_cpu.append(c_p)
        put_prices_mat_cpu.append(p_p)
        
    call_prices_mat_cpu = np.array(call_prices_mat_cpu)
    put_prices_mat_cpu = np.array(put_prices_mat_cpu)
    
    assert np.all(np.diff(call_prices_mat_cpu) >= -1e-7), "CPU Call price is not non-decreasing in T_opt (r=0)"
    assert np.all(np.diff(put_prices_mat_cpu) >= -1e-7), "CPU Put price is not non-decreasing in T_opt (r=0)"
    
    # PyTorch Check (Corrected version of the original test with diff calculation)
    for dev in devices:
        chi_pt = torch.tensor([p["chi_t"]], dtype=torch.float64, device=dev)
        xi_pt = torch.tensor([p["xi_t"]], dtype=torch.float64, device=dev)
        T_opt_pt = torch.tensor(maturities, dtype=torch.float64, device=dev)
        
        call_prices_pt = schwartz_smith_price_cos_pt(
            p["t"], T_opt_pt, T_fut_fixed, p["K"], r_zero, chi_pt, xi_pt,
            p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
            option_type="C", N=256, L=10.0
        )
        put_prices_pt = schwartz_smith_price_cos_pt(
            p["t"], T_opt_pt, T_fut_fixed, p["K"], r_zero, chi_pt, xi_pt,
            p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
            option_type="P", N=256, L=10.0
        )
        
        m_call_diffs = torch.diff(call_prices_pt)
        m_put_diffs = torch.diff(put_prices_pt)
        
        assert torch.all(m_call_diffs >= -1e-7), f"PyTorch {dev} Call option price not non-decreasing in T_opt (r=0)"
        assert torch.all(m_put_diffs >= -1e-7), f"PyTorch {dev} Put option price not non-decreasing in T_opt (r=0)"


def test_challenger_expired_options():
    """Verify that expired options (tau <= 0) return undiscounted payoffs."""
    p = BASE_PARAMS.copy()
    p["t"] = 0.5
    p["T_fut"] = 1.0
    
    # Calculate underlying futures price at t = 0.5
    F = futures_price(
        p["t"], p["T_fut"], p["chi_t"], p["xi_t"],
        p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"]
    )
    
    # Test tau = 0 (T_opt = t = 0.5) and tau < 0 (T_opt < t)
    for T_opt in [0.5, 0.4, 0.0]:
        tau = T_opt - p["t"]
        for otype in ["C", "P"]:
            # CPU
            p_cos = schwartz_smith_price_cos(
                p["t"], T_opt, p["T_fut"], p["K"], p["r"], p["chi_t"], p["xi_t"],
                p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
                option_type=otype, N=128, L=10.0
            )
            payoff = max(F - p["K"], 0.0) if otype == "C" else max(p["K"] - F, 0.0)
            
            assert np.isclose(p_cos, payoff, atol=1e-15), \
                f"CPU Expired option at tau={tau} did not return undiscounted payoff. Got {p_cos}, expected {payoff}"

    # PyTorch batch check (CPU and GPU)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        
    for dev in devices:
        chi_pt = torch.tensor([p["chi_t"]], dtype=torch.float64, device=dev)
        xi_pt = torch.tensor([p["xi_t"]], dtype=torch.float64, device=dev)
        
        # Batch of negative, zero, and positive maturities (T_opt = 0.4, 0.5, 0.6 with t = 0.5)
        T_opts = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64, device=dev)
        
        for otype in ["C", "P"]:
            prices = schwartz_smith_price_cos_pt(
                p["t"], T_opts, p["T_fut"], p["K"], p["r"], chi_pt, xi_pt,
                p["kappa"], p["sigma_chi"], p["rho"], p["sigma_xi"], p["mu_star"], p["lambda_chi"],
                option_type=otype, N=128, L=10.0
            )
            
            payoff = max(F - p["K"], 0.0) if otype == "C" else max(p["K"] - F, 0.0)
            
            # Check expired elements return undiscounted payoff
            assert np.isclose(prices[0].item(), payoff, atol=1e-15), \
                f"PyTorch {dev} expired tau < 0 fail for {otype}: {prices[0].item()} vs {payoff}"
            assert np.isclose(prices[1].item(), payoff, atol=1e-15), \
                f"PyTorch {dev} expired tau = 0 fail for {otype}: {prices[1].item()} vs {payoff}"
            
            # Check active option element is priced properly (different from payoff, greater than 0)
            active_price = prices[2].item()
            assert active_price > 0.0
            assert not np.isclose(active_price, payoff, atol=1e-7)
