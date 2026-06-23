import pytest
import numpy as np
import torch
import time
from src.pricing.schwartz_smith import (
    schwartz_smith_price_black76,
    schwartz_smith_price_black76_pt,
    schwartz_smith_price_cos,
    schwartz_smith_price_cos_pt,
    futures_price
)


def test_extreme_short_maturity():
    """Verify numerical stability and accuracy under extremely short option maturities."""
    kappa = 0.6
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    chi_t = 0.1
    xi_t = np.log(70.0)
    t = 0.0
    r = 0.04
    K = 70.0
    
    # We test various small maturities down to 1e-8
    for T_opt in [1e-2, 1e-4, 1e-6, 1e-8]:
        T_fut = T_opt + 0.1
        for otype in ["C", "P"]:
            # CPU
            p_black = schwartz_smith_price_black76(
                t, T_opt, T_fut, K, r, chi_t, xi_t,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype
            )
            p_cos = schwartz_smith_price_cos(
                t, T_opt, T_fut, K, r, chi_t, xi_t,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype, N=128, L=10.0
            )
            assert np.isfinite(p_cos), f"COS price non-finite for T_opt={T_opt}, type={otype}"
            assert p_cos >= 0.0, f"COS price negative ({p_cos}) for T_opt={T_opt}, type={otype}"
            assert np.isclose(p_cos, p_black, rtol=1e-5, atol=1e-5), f"Mismatch at T_opt={T_opt}, type={otype}: Black={p_black}, COS={p_cos}"

            # PyTorch
            chi_pt = torch.tensor([chi_t], dtype=torch.float64)
            xi_pt = torch.tensor([xi_t], dtype=torch.float64)
            p_black_pt = schwartz_smith_price_black76_pt(
                t, T_opt, T_fut, K, r, chi_pt, xi_pt,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype
            )
            p_cos_pt = schwartz_smith_price_cos_pt(
                t, T_opt, T_fut, K, r, chi_pt, xi_pt,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype, N=128, L=10.0
            )
            assert torch.isfinite(p_cos_pt).all(), f"COS_pt non-finite for T_opt={T_opt}, type={otype}"
            assert (p_cos_pt >= 0.0).all(), f"COS_pt negative for T_opt={T_opt}, type={otype}"
            assert np.isclose(p_cos_pt.item(), p_black_pt.item(), rtol=1e-5, atol=1e-5), f"Mismatch at T_opt={T_opt}, type={otype}: Black_pt={p_black_pt.item()}, COS_pt={p_cos_pt.item()}"

def test_extreme_volatilities():
    """Verify numerical stability and accuracy under extreme volatility parameters."""
    kappa = 0.6
    rho = 0.35
    mu_star = 0.02
    lambda_chi = 0.05
    chi_t = 0.1
    xi_t = np.log(70.0)
    t = 0.0
    T_opt = 0.5
    T_fut = 0.6
    r = 0.04
    K = 70.0

    vols = [
        (0.0001, 0.0001), # ultra-low vol
        (2.0, 2.0),       # extremely high vol
        (0.0, 0.0),       # zero vol
        (5.0, 0.05),      # highly asymmetric vol
        (0.05, 5.0),      # highly asymmetric vol
    ]

    for sig_chi, sig_xi in vols:
        for otype in ["C", "P"]:
            # CPU
            p_black = schwartz_smith_price_black76(
                t, T_opt, T_fut, K, r, chi_t, xi_t,
                kappa, sig_chi, rho, sig_xi, mu_star, lambda_chi,
                option_type=otype
            )
            p_cos = schwartz_smith_price_cos(
                t, T_opt, T_fut, K, r, chi_t, xi_t,
                kappa, sig_chi, rho, sig_xi, mu_star, lambda_chi,
                option_type=otype, N=256, L=10.0
            )
            assert np.isfinite(p_cos), f"COS price non-finite for sig_chi={sig_chi}, sig_xi={sig_xi}, type={otype}"
            assert p_cos >= 0.0, f"COS price negative ({p_cos}) for sig_chi={sig_chi}, sig_xi={sig_xi}, type={otype}"
            
            # Using relative tolerance because absolute difference can scale with the very large price of options at high vol
            assert np.isclose(p_cos, p_black, rtol=1e-4, atol=1e-4), f"Mismatch at sig_chi={sig_chi}, sig_xi={sig_xi}, type={otype}: Black={p_black}, COS={p_cos}, diff={abs(p_cos - p_black)}"

            # PyTorch
            chi_pt = torch.tensor([chi_t], dtype=torch.float64)
            xi_pt = torch.tensor([xi_t], dtype=torch.float64)
            p_black_pt = schwartz_smith_price_black76_pt(
                t, T_opt, T_fut, K, r, chi_pt, xi_pt,
                kappa, sig_chi, rho, sig_xi, mu_star, lambda_chi,
                option_type=otype
            )
            p_cos_pt = schwartz_smith_price_cos_pt(
                t, T_opt, T_fut, K, r, chi_pt, xi_pt,
                kappa, sig_chi, rho, sig_xi, mu_star, lambda_chi,
                option_type=otype, N=256, L=10.0
            )
            assert torch.isfinite(p_cos_pt).all(), f"COS_pt non-finite for sig_chi={sig_chi}, sig_xi={sig_xi}, type={otype}"
            assert (p_cos_pt >= 0.0).all(), f"COS_pt negative for sig_chi={sig_chi}, sig_xi={sig_xi}, type={otype}"
            assert np.isclose(p_cos_pt.item(), p_black_pt.item(), rtol=1e-4, atol=1e-4), f"Mismatch at sig_chi={sig_chi}, sig_xi={sig_xi}, type={otype}: Black_pt={p_black_pt.item()}, COS_pt={p_cos_pt.item()}"

def test_extreme_correlations():
    """Verify numerical stability and accuracy under extreme correlation parameters."""
    kappa = 0.6
    sigma_chi = 0.25
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    chi_t = 0.1
    xi_t = np.log(70.0)
    t = 0.0
    T_opt = 0.5
    T_fut = 0.6
    r = 0.04
    K = 70.0

    correlations = [-0.999, -0.99, -0.5, 0.0, 0.5, 0.99, 0.999]

    for rho in correlations:
        for otype in ["C", "P"]:
            # CPU
            p_black = schwartz_smith_price_black76(
                t, T_opt, T_fut, K, r, chi_t, xi_t,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype
            )
            p_cos = schwartz_smith_price_cos(
                t, T_opt, T_fut, K, r, chi_t, xi_t,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype, N=256, L=12.0
            )
            assert np.isfinite(p_cos), f"COS price non-finite for rho={rho}, type={otype}"
            assert p_cos >= 0.0, f"COS price negative ({p_cos}) for rho={rho}, type={otype}"
            assert np.isclose(p_cos, p_black, rtol=1e-5, atol=1e-5), f"Mismatch at rho={rho}, type={otype}: Black={p_black}, COS={p_cos}"

            # PyTorch
            chi_pt = torch.tensor([chi_t], dtype=torch.float64)
            xi_pt = torch.tensor([xi_t], dtype=torch.float64)
            p_black_pt = schwartz_smith_price_black76_pt(
                t, T_opt, T_fut, K, r, chi_pt, xi_pt,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype
            )
            p_cos_pt = schwartz_smith_price_cos_pt(
                t, T_opt, T_fut, K, r, chi_pt, xi_pt,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype, N=256, L=12.0
            )
            assert torch.isfinite(p_cos_pt).all(), f"COS_pt non-finite for rho={rho}, type={otype}"
            assert (p_cos_pt >= 0.0).all(), f"COS_pt negative for rho={rho}, type={otype}"
            assert np.isclose(p_cos_pt.item(), p_black_pt.item(), rtol=1e-5, atol=1e-5), f"Mismatch at rho={rho}, type={otype}: Black_pt={p_black_pt.item()}, COS_pt={p_cos_pt.item()}"

def test_high_N():
    """Verify numerical stability and accuracy with large N values."""
    kappa = 0.6
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    chi_t = 0.1
    xi_t = np.log(70.0)
    t = 0.0
    T_opt = 0.5
    T_fut = 0.6
    r = 0.04
    K = 70.0

    for N in [512, 1024, 2048, 4096]:
        for otype in ["C", "P"]:
            p_black = schwartz_smith_price_black76(
                t, T_opt, T_fut, K, r, chi_t, xi_t,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype
            )
            p_cos = schwartz_smith_price_cos(
                t, T_opt, T_fut, K, r, chi_t, xi_t,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype, N=N, L=10.0
            )
            assert np.isfinite(p_cos), f"COS price non-finite for N={N}, type={otype}"
            assert np.isclose(p_cos, p_black, rtol=1e-5, atol=1e-5), f"Mismatch at N={N}, type={otype}: Black={p_black}, COS={p_cos}"

def test_gpu_acceleration():
    """Test the behavior on GPU if CUDA is available, confirming speedup and device robustness."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    # Set up large arrays to measure speedup
    size = 20000
    np.random.seed(1234)

    chi_t = np.random.uniform(-0.5, 0.5, size)
    xi_t = np.random.uniform(np.log(50), np.log(100), size)
    T_opt = np.random.uniform(0.1, 2.0, size)
    T_fut = T_opt + np.random.uniform(0.1, 0.5, size)
    K = np.random.uniform(50, 100, size)

    r = 0.04
    kappa = 0.6
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05

    # PyTorch CPU (using float64)
    chi_pt_cpu = torch.tensor(chi_t, dtype=torch.float64)
    xi_pt_cpu = torch.tensor(xi_t, dtype=torch.float64)
    T_opt_cpu = torch.tensor(T_opt, dtype=torch.float64)
    T_fut_cpu = torch.tensor(T_fut, dtype=torch.float64)
    K_cpu = torch.tensor(K, dtype=torch.float64)

    # PyTorch GPU
    chi_pt_gpu = chi_pt_cpu.to(device)
    xi_pt_gpu = xi_pt_cpu.to(device)
    T_opt_gpu = T_opt_cpu.to(device)
    T_fut_gpu = T_fut_cpu.to(device)
    K_gpu = K_cpu.to(device)

    # Warm up
    _ = schwartz_smith_price_cos_pt(
        t=0.0, T_opt=T_opt_gpu[:100], T_fut=T_fut_gpu[:100], K=K_gpu[:100], r=r,
        chi_t=chi_pt_gpu[:100], xi_t=xi_pt_gpu[:100], kappa=kappa,
        sigma_chi=sigma_chi, rho=rho, sigma_xi=sigma_xi, mu_star=mu_star,
        lambda_chi=lambda_chi, N=128
    )
    torch.cuda.synchronize()

    # Time CPU
    t0 = time.perf_counter()
    p_cpu = schwartz_smith_price_cos_pt(
        t=0.0, T_opt=T_opt_cpu, T_fut=T_fut_cpu, K=K_cpu, r=r,
        chi_t=chi_pt_cpu, xi_t=xi_pt_cpu, kappa=kappa,
        sigma_chi=sigma_chi, rho=rho, sigma_xi=sigma_xi, mu_star=mu_star,
        lambda_chi=lambda_chi, N=128
    )
    t_cpu = time.perf_counter() - t0

    # Time GPU
    t1 = time.perf_counter()
    p_gpu = schwartz_smith_price_cos_pt(
        t=0.0, T_opt=T_opt_gpu, T_fut=T_fut_gpu, K=K_gpu, r=r,
        chi_t=chi_pt_gpu, xi_t=xi_pt_gpu, kappa=kappa,
        sigma_chi=sigma_chi, rho=rho, sigma_xi=sigma_xi, mu_star=mu_star,
        lambda_chi=lambda_chi, N=128
    )
    torch.cuda.synchronize()
    t_gpu = time.perf_counter() - t1

    assert p_gpu.device.type == "cuda"
    np.testing.assert_allclose(p_cpu.numpy(), p_gpu.cpu().numpy(), rtol=1e-7, atol=1e-7)
    
    print(f"Large-scale CPU Time: {t_cpu:.6f} s")
    print(f"Large-scale GPU Time: {t_gpu:.6f} s")
    print(f"Speedup: {t_cpu / t_gpu:.2f}x")
    assert t_gpu < t_cpu

def test_monotonicity_and_convexity():
    """Verify monotonicity of option prices with respect to strike and maturity, and convexity in strike."""
    kappa = 0.6
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    chi_t = 0.1
    xi_t = np.log(75.0)
    t = 0.0
    
    # 1. Strike Monotonicity & Convexity for both CPU and PyTorch
    r = 0.04
    T_opt = 0.5
    T_fut = 1.0
    strikes = np.linspace(40.0, 110.0, 100)
    
    # Check CPU
    call_prices_cpu = []
    put_prices_cpu = []
    for K in strikes:
        c_price = schwartz_smith_price_cos(
            t, T_opt, T_fut, K, r, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="C", N=256, L=12.0
        )
        p_price = schwartz_smith_price_cos(
            t, T_opt, T_fut, K, r, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="P", N=256, L=12.0
        )
        call_prices_cpu.append(c_price)
        put_prices_cpu.append(p_price)
        
    call_prices_cpu = np.array(call_prices_cpu)
    put_prices_cpu = np.array(put_prices_cpu)
    
    # Monotonicity check: derivative of call is negative, derivative of put is positive
    assert np.all(np.diff(call_prices_cpu) <= 1e-7), "CPU Call option price is not non-increasing with strike"
    assert np.all(np.diff(put_prices_cpu) >= -1e-7), "CPU Put option price is not non-decreasing with strike"
    
    # Convexity check (f''(x) >= 0)
    assert np.all(np.diff(call_prices_cpu, n=2) >= -1e-9), "CPU Call option price is not convex in strike"
    assert np.all(np.diff(put_prices_cpu, n=2) >= -1e-9), "CPU Put option price is not convex in strike"

    # Check PyTorch (CPU and GPU if available)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        
    for device in devices:
        chi_pt = torch.tensor([chi_t], dtype=torch.float64, device=device)
        xi_pt = torch.tensor([xi_t], dtype=torch.float64, device=device)
        K_pt = torch.tensor(strikes, dtype=torch.float64, device=device)
        
        call_prices_pt = schwartz_smith_price_cos_pt(
            t, T_opt, T_fut, K_pt, r, chi_pt, xi_pt,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="C", N=256, L=12.0
        )
        put_prices_pt = schwartz_smith_price_cos_pt(
            t, T_opt, T_fut, K_pt, r, chi_pt, xi_pt,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="P", N=256, L=12.0
        )
        
        call_diffs = torch.diff(call_prices_pt)
        put_diffs = torch.diff(put_prices_pt)
        assert torch.all(call_diffs <= 1e-7), f"PyTorch ({device}) Call option price not non-increasing in K"
        assert torch.all(put_diffs >= -1e-7), f"PyTorch ({device}) Put option price not non-decreasing in K"
        
        call_second_diffs = torch.diff(call_prices_pt, n=2)
        put_second_diffs = torch.diff(put_prices_pt, n=2)
        assert torch.all(call_second_diffs >= -1e-9), f"PyTorch ({device}) Call option price not convex in K"
        assert torch.all(put_second_diffs >= -1e-9), f"PyTorch ({device}) Put option price not convex in K"

    # 3. Maturity Monotonicity (r = 0, fixed T_fut)
    # When r = 0, the option value should be a non-decreasing function of T_opt.
    r_zero = 0.0
    T_fut_fixed = 2.0
    maturities = np.linspace(0.05, 1.95, 50)
    
    for device in devices:
        chi_pt = torch.tensor([chi_t], dtype=torch.float64, device=device)
        xi_pt = torch.tensor([xi_t], dtype=torch.float64, device=device)
        T_opt_pt = torch.tensor(maturities, dtype=torch.float64, device=device)
        
        call_prices_pt = schwartz_smith_price_cos_pt(
            t, T_opt_pt, T_fut_fixed, 75.0, r_zero, chi_pt, xi_pt,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="C", N=256, L=12.0
        )
        put_prices_pt = schwartz_smith_price_cos_pt(
            t, T_opt_pt, T_fut_fixed, 75.0, r_zero, chi_pt, xi_pt,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="P", N=256, L=12.0
        )
        
        m_call_diffs = torch.diff(call_prices_pt)
        m_put_diffs = torch.diff(put_prices_pt)
        assert torch.all(m_call_diffs >= -1e-7), f"PyTorch ({device}) Call option price not non-decreasing in T_opt (r=0)"
        assert torch.all(m_put_diffs >= -1e-7), f"PyTorch ({device}) Put option price not non-decreasing in T_opt (r=0)"


def test_expired_options():
    """Verify that expired options (tau <= 0) return undiscounted payoffs."""
    kappa = 0.6
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    chi_t = 0.1
    xi_t = np.log(75.0)
    t = 0.5
    r = 0.04
    K = 70.0
    
    # Futures price at t=0.5 with T_fut=1.0
    T_fut = 1.0
    F = futures_price(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    
    # 1. CPU check
    for T_opt in [0.5, 0.4, 0.0]:  # tau = 0.0, -0.1, -0.5
        for otype in ["C", "P"]:
            p_cos = schwartz_smith_price_cos(
                t, T_opt, T_fut, K, r, chi_t, xi_t,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype, N=128, L=10.0
            )
            payoff = max(F - K, 0.0) if otype == "C" else max(K - F, 0.0)
            assert np.isclose(p_cos, payoff, atol=1e-15), f"Expired option did not return undiscounted payoff for T_opt={T_opt}, type={otype}. Got {p_cos}, expected {payoff}"

    # 2. PyTorch batch check (CPU and GPU if available)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        
    for device in devices:
        chi_pt = torch.tensor([chi_t], dtype=torch.float64, device=device)
        xi_pt = torch.tensor([xi_t], dtype=torch.float64, device=device)
        
        # Test a batch with negative, zero, and positive maturities
        T_opts = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64, device=device)
        
        for otype in ["C", "P"]:
            prices = schwartz_smith_price_cos_pt(
                t, T_opts, T_fut, K, r, chi_pt, xi_pt,
                kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                option_type=otype, N=128, L=10.0
            )
            
            payoff = max(F - K, 0.0) if otype == "C" else max(K - F, 0.0)
            
            # Check expired elements
            assert np.isclose(prices[0].item(), payoff, atol=1e-15), f"tau < 0 fail on {device} ({otype}): {prices[0].item()} vs {payoff}"
            assert np.isclose(prices[1].item(), payoff, atol=1e-15), f"tau = 0 fail on {device} ({otype}): {prices[1].item()} vs {payoff}"
            
            # Check active option element is priced properly (different from payoff, greater than 0)
            active_price = prices[2].item()
            assert active_price > 0.0
            assert not np.isclose(active_price, payoff, atol=1e-7)


def test_cpu_maturity_monotonicity():
    """Verify maturity monotonicity of option prices on CPU (r=0)."""
    kappa = 0.6
    sigma_chi = 0.25
    rho = 0.35
    sigma_xi = 0.12
    mu_star = 0.02
    lambda_chi = 0.05
    chi_t = 0.1
    xi_t = np.log(75.0)
    t = 0.0
    r_zero = 0.0
    T_fut_fixed = 2.0
    maturities = np.linspace(0.05, 1.95, 50)
    
    call_prices = []
    put_prices = []
    for T_opt in maturities:
        c_price = schwartz_smith_price_cos(
            t, T_opt, T_fut_fixed, 75.0, r_zero, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="C", N=256, L=12.0
        )
        p_price = schwartz_smith_price_cos(
            t, T_opt, T_fut_fixed, 75.0, r_zero, chi_t, xi_t,
            kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
            option_type="P", N=256, L=12.0
        )
        call_prices.append(c_price)
        put_prices.append(p_price)
        
    call_prices = np.array(call_prices)
    put_prices = np.array(put_prices)
    
    assert np.all(np.diff(call_prices) >= -1e-7), "CPU Call option price is not non-decreasing in T_opt (r=0)"
    assert np.all(np.diff(put_prices) >= -1e-7), "CPU Put option price is not non-decreasing in T_opt (r=0)"

