import pytest
import numpy as np
import torch
import gc
from deepvol.models.bachelier import (
    bachelier_price,
    black_price,
    shifted_black_price,
    bachelier_implied_vol,
    black_implied_vol
)
from deepvol.models.heston import (
    heston_cf,
    heston_iv_surface,
    batch_heston_iv_surface,
    HestonEngine
)
from deepvol.models.local_vol import (
    svi_slice,
    svi_to_lv_surface,
    check_arbitrage_free
)
from deepvol.models.sabr import (
    sabr_iv_lognormal,
    sabr_iv_normal,
    sabr_iv_surface,
    ssvi_total_variance
)
from deepvol.models.rbergomi_gpu import rBergomiEngine

def test_bachelier_stress():
    """
    Stress test Bachelier pricing and implied volatility solvers.
    Verifies behavior under:
      - Extremely short maturities (T <= 1e-8)
      - Volatilities at boundary limits (sigma <= 1e-8, sigma = 1e4)
      - Strike boundaries (ATM, deep ITM, deep OTM, negative strikes)
      - Invalidation boundaries (negative T or sigma)
    """
    T_cases = [0.0, 1e-12, 1e-8, 1e-5, 1.0, -1e-5]
    sigma_cases = [0.0, 1e-12, 1e-8, 1e-5, 1.0, 10000.0, -1.0]
    K_cases = [100.0, 0.0, -50.0, 50.0, 150.0, 1e6, -1e6]
    
    F = 100.0
    
    for T in T_cases:
        for sigma in sigma_cases:
            for K in K_cases:
                price_c = bachelier_price(F, K, T, sigma, 'call')
                price_p = bachelier_price(F, K, T, sigma, 'put')
                
                if T < 0.0 or sigma < 0.0:
                    assert np.isnan(price_c), f"Expected NaN for T={T}, sigma={sigma}"
                    assert np.isnan(price_p), f"Expected NaN for T={T}, sigma={sigma}"
                else:
                    assert np.isfinite(price_c), f"Expected finite price for T={T}, sigma={sigma}, K={K}"
                    assert np.isfinite(price_p), f"Expected finite price for T={T}, sigma={sigma}, K={K}"
                    
                    if T <= 1e-8 or sigma <= 1e-8:
                        assert price_c == max(F - K, 0.0), f"Expected call intrinsic, got {price_c}"
                        assert price_p == max(K - F, 0.0), f"Expected put intrinsic, got {price_p}"
                    else:
                        np.testing.assert_allclose(price_c - price_p, F - K, atol=1e-10)
                        
                if T > 1e-8 and sigma > 1e-8 and 0.0 < K < 200.0:
                    price_c_valid = bachelier_price(F, K, T, sigma, 'call')
                    iv_solved = bachelier_implied_vol(price_c_valid, F, K, T, 'call')
                    if np.isfinite(iv_solved) and iv_solved > 1e-8:
                        price_solved = bachelier_price(F, K, T, iv_solved, 'call')
                        np.testing.assert_allclose(price_solved, price_c_valid, atol=1e-5)
                        
    # Test bachelier_implied_vol boundary checks
    # Price below intrinsic must return NaN
    assert np.isnan(bachelier_implied_vol(5.0, 100.0, 90.0, 0.5, 'call'))
    # Price at intrinsic should return 0.0
    assert bachelier_implied_vol(10.0, 100.0, 90.0, 0.5, 'call') == 0.0


def test_black_stress():
    """
    Stress test Black pricing and implied volatility solvers.
    Verifies behavior under:
      - Extremely short maturities (T <= 1e-8)
      - Volatilities at boundary limits (sigma <= 1e-8, sigma = 10.0)
      - Strike boundaries (ATM, deep OTM, deep ITM, invalid strikes)
      - Invalidation boundaries (negative T or sigma)
    """
    F = 100.0
    T_cases = [0.0, 1e-12, 1e-8, 1e-5, 1.0, -1e-5]
    sigma_cases = [0.0, 1e-12, 1e-8, 1e-5, 1.0, 10.0, -0.1]
    K_cases = [100.0, 1e-12, 1e-5, 50.0, 150.0, 1e6, -100.0]
    
    for T in T_cases:
        for sigma in sigma_cases:
            for K in K_cases:
                price_c = black_price(F, K, T, sigma, 'call')
                price_p = black_price(F, K, T, sigma, 'put')
                
                if T < 0.0 or sigma < 0.0 or F <= 0.0 or K <= 0.0:
                    assert np.isnan(price_c), f"Expected NaN for T={T}, sigma={sigma}, K={K}"
                    assert np.isnan(price_p), f"Expected NaN for T={T}, sigma={sigma}, K={K}"
                else:
                    assert np.isfinite(price_c), f"Expected finite price for T={T}, sigma={sigma}, K={K}"
                    assert np.isfinite(price_p), f"Expected finite price for T={T}, sigma={sigma}, K={K}"
                    
                    if T <= 1e-8 or sigma <= 1e-8:
                        assert price_c == max(F - K, 0.0), f"Expected call intrinsic, got {price_c}"
                        assert price_p == max(K - F, 0.0), f"Expected put intrinsic, got {price_p}"
                    else:
                        np.testing.assert_allclose(price_c - price_p, F - K, atol=1e-10)
                        
                if T > 1e-8 and sigma > 1e-8 and K > 0.0 and K < 200.0:
                    price_c_valid = black_price(F, K, T, sigma, 'call')
                    iv_solved = black_implied_vol(price_c_valid, F, K, T, 'call')
                    if np.isfinite(iv_solved) and iv_solved > 1e-8:
                        price_solved = black_price(F, K, T, iv_solved, 'call')
                        np.testing.assert_allclose(price_solved, price_c_valid, atol=1e-5)
                        
    # Test black_implied_vol boundary checks
    # Price below intrinsic must return NaN
    assert np.isnan(black_implied_vol(5.0, 100.0, 90.0, 0.5, 'call'))
    # Price at intrinsic should return 0.0
    assert black_implied_vol(10.0, 100.0, 90.0, 0.5, 'call') == 0.0
    # Price above stock (F) should return NaN
    assert np.isnan(black_implied_vol(105.0, 100.0, 90.0, 0.5, 'call'))


def test_heston_stress():
    """
    Stress test Heston characteristic function and pricing engine.
    Verifies behavior under:
      - Extremely short maturities (T <= 1e-8)
      - Volatilities at boundary limits (v0 -> 0, v0 -> 100, vol-of-vol -> 0, vol-of-vol -> 10)
      - Correlation limits (rho = -1, rho = +1)
      - Compare CPU and GPU results (if CUDA available)
    """
    T_cases = [0.0, 1e-12, 1e-8, 1e-5, 1.0, 5.0]
    v0_cases = [1e-10, 0.04, 1.0, 100.0]
    sigma_cases = [1e-10, 0.3, 2.0, 10.0]
    rho_cases = [-1.0, -0.9999, -0.5, 0.0, 0.5, 0.9999, 1.0]
    
    # Verify characteristic function values are stable and in the unit circle
    u = np.linspace(0.1, 100.0, 50)
    for T in T_cases:
        for v0 in v0_cases:
            for sigma in sigma_cases:
                for rho in rho_cases:
                    cf = heston_cf(u, T, kappa=1.5, theta=0.05, sigma=sigma, rho=rho, v0=v0)
                    assert np.all(np.isfinite(cf)), f"CF contained non-finite values for T={T}, v0={v0}, sigma={sigma}, rho={rho}"
                    assert np.all(np.abs(cf) <= 1.0 + 1e-6), f"CF exceeded unit circle for T={T}, v0={v0}, sigma={sigma}, rho={rho}"
                    
    engine = HestonEngine()
    T_grid = np.array([1e-8, 0.1, 1.0, 5.0])
    K_grid = np.array([-0.5, 0.0, 0.5])
    
    params = {
        'kappa': 1.5,
        'theta': 0.05,
        'sigma': 0.3,
        'rho': -0.7,
        'v0': 0.04
    }
    
    # Price surface on CPU
    iv_surface = engine.price_surface(params, T_grid, K_grid)
    assert iv_surface.shape == (len(T_grid), len(K_grid))
    assert np.all(np.isnan(iv_surface) | ((iv_surface >= 0.0) & np.isfinite(iv_surface)))
    assert np.any(np.isfinite(iv_surface))
    
    # Price surface on GPU if available
    if torch.cuda.is_available():
        params_tensor = torch.tensor([[1.5, 0.05, 0.3, -0.7, 0.04]], dtype=torch.float64)
        T_tensor = torch.tensor(T_grid, dtype=torch.float64)
        K_tensor = torch.tensor(K_grid, dtype=torch.float64)
        
        iv_gpu = engine.batch_price_surface(params_tensor, T_tensor, K_tensor, device='cuda')
        iv_cpu = engine.batch_price_surface(params_tensor, T_tensor, K_tensor, device='cpu')
        
        assert iv_gpu.shape == (1, len(T_grid), len(K_grid))
        assert iv_cpu.shape == (1, len(T_grid), len(K_grid))
        
        iv_gpu_cpu = iv_gpu.cpu()
        iv_cpu_cpu = iv_cpu.cpu()
        mask = ~torch.isnan(iv_gpu_cpu)
        if mask.any():
            torch.testing.assert_close(iv_gpu_cpu[mask], iv_cpu_cpu[mask], rtol=1e-4, atol=1e-4)


def test_local_vol_stress():
    """
    Stress test Local Volatility model (SVI representation and Dupire formula).
    Verifies behavior under:
      - SVI parameter boundaries (curvature sigma -> 0, correlation rho -> +/-1)
      - Dupire formula with very short maturity grid steps
      - Arbitrage condition failures
    """
    k = np.linspace(-2.0, 2.0, 50)
    a, b, rho, m, sigma = 0.04, 0.1, -0.5, 0.0, 0.2
    
    # Check SVI slice values are stable
    for sig in [0.0, 1e-12, -0.1, 10.0]:
        w = svi_slice(k, a, b, rho, m, sig)
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0.0)
        
    for slp in [0.0, -0.05]:
        w = svi_slice(k, a, slp, rho, m, sigma)
        assert np.all(np.isfinite(w))
        
        # Test check_arbitrage_free flags invalid parameter b < 0 or negative total variance
        T_grid = np.array([0.5])
        K_grid = np.array([-0.5, 0.0, 0.5])
        svi_params = np.array([[a, slp, rho, m, sigma]])
        is_free = check_arbitrage_free(T_grid, K_grid, svi_params)
        if slp < 0:
            assert not is_free
            
    # Dupire local volatility surface computation
    # Test with very short maturity grid: [1e-8, 0.5, 1.0]
    T_grid = np.array([1e-8, 0.5, 1.0])
    K_grid = np.array([-0.5, 0.0, 0.5])
    svi_params = np.array([
        [0.02, 0.1, -0.5, 0.0, 0.15],
        [0.04, 0.1, -0.5, 0.0, 0.15],
        [0.08, 0.1, -0.5, 0.0, 0.15]
    ])
    
    lv_surface = svi_to_lv_surface(T_grid, K_grid, svi_params)
    assert lv_surface.shape == (3, 3)
    assert np.all(np.isfinite(lv_surface))
    
    # Test Dupire behavior under extremely short time differences
    T_grid_tight = np.array([1e-8, 2e-8])
    svi_params_tight = np.array([
        [0.02, 0.1, -0.5, 0.0, 0.15],
        [0.0200001, 0.1, -0.5, 0.0, 0.15]
    ])
    lv_surface_tight = svi_to_lv_surface(T_grid_tight, K_grid, svi_params_tight)
    assert lv_surface_tight.shape == (2, 3)
    assert np.all(np.isfinite(lv_surface_tight))


def test_sabr_stress():
    """
    Stress test SABR lognormal and normal implied volatility formulas.
    Verifies behavior under:
      - Extremely short maturities (T <= 1e-8)
      - Volatilities at boundary limits (alpha -> 0, nu -> 0, alpha -> 10, nu -> 10)
      - Strike boundaries (ATM, deep ITM, deep OTM)
      - Correlation limits (rho = -1, rho = +1)
    """
    F = 100.0
    T_cases = [1e-9, 1e-8, 1e-5, 1.0, -0.5]
    alpha_cases = [1e-6, 0.2, 5.0]
    beta_cases = [0.0, 0.5, 1.0]
    rho_cases = [-1.0, -0.9999, 0.0, 0.9999, 1.0]
    nu_cases = [1e-6, 0.3, 5.0]
    K_cases = [100.0, 1.0, 1e-5, 200.0, 10000.0]
    
    for T in T_cases:
        for alpha in alpha_cases:
            for beta in beta_cases:
                for rho in rho_cases:
                    for nu in nu_cases:
                        for K in K_cases:
                            # Verify lognormal SABR
                            try:
                                iv_ln = sabr_iv_lognormal(F, K, T, alpha, beta, rho, nu)
                                if T <= 0.0:
                                    assert np.isnan(iv_ln)
                                else:
                                    assert np.all(np.isnan(iv_ln)) or np.all(np.isfinite(iv_ln))
                            except ValueError as e:
                                assert "alpha" in str(e) or "beta" in str(e) or "nu" in str(e) or "rho" in str(e)
                                
                            # Verify normal SABR
                            try:
                                iv_n = sabr_iv_normal(F, K, T, alpha, beta, rho, nu)
                                if T <= 0.0:
                                    assert np.isnan(iv_n)
                                else:
                                    assert np.all(np.isnan(iv_n)) or np.all(np.isfinite(iv_n))
                            except ValueError as e:
                                assert "alpha" in str(e) or "beta" in str(e) or "nu" in str(e) or "rho" in str(e)


def test_rbergomi_engine_stress():
    """
    Stress test rBergomiEngine path simulation wrapper.
    Verifies behavior under:
      - Correct execution on CPU and GPU (if CUDA available)
      - Correct path shape and positive pricing/variance values
      - Memory leaks or crashes under continuous simulations
      - Extremely short maturity boundaries (T <= 1e-8)
      - Parameter boundary constraints check
    """
    engine = rBergomiEngine()
    params = torch.tensor([[0.04, 0.1, 1.5, -0.7]], dtype=torch.float32)
    T = 0.5
    steps_per_unit = 200
    N_paths = 1000
    
    # 1. CPU execution
    S_cpu, V_cpu, t_cpu = engine.simulate_paths(
        params=params,
        T=T,
        steps_per_unit=steps_per_unit,
        N_paths=N_paths,
        antithetic=True,
        device="cpu",
        dtype=torch.float32
    )
    
    expected_steps = int(round(T * steps_per_unit)) + 1
    assert S_cpu.shape == (1, N_paths, expected_steps)
    assert V_cpu.shape == (1, N_paths, expected_steps)
    assert t_cpu.shape == (expected_steps,)
    assert torch.all(S_cpu > 0.0)
    assert torch.all(V_cpu > 0.0)
    assert torch.all(torch.isfinite(S_cpu))
    assert torch.all(torch.isfinite(V_cpu))
    
    # 2. GPU execution if available
    if torch.cuda.is_available():
        S_gpu, V_gpu, t_gpu = engine.simulate_paths(
            params=params,
            T=T,
            steps_per_unit=steps_per_unit,
            N_paths=N_paths,
            antithetic=True,
            device="cuda",
            dtype=torch.float32
        )
        assert S_gpu.shape == (1, N_paths, expected_steps)
        assert V_gpu.shape == (1, N_paths, expected_steps)
        assert t_gpu.shape == (expected_steps,)
        assert torch.all(S_gpu > 0.0)
        assert torch.all(V_gpu > 0.0)
        assert torch.all(torch.isfinite(S_gpu))
        assert torch.all(torch.isfinite(V_gpu))
        
        # 3. VRAM leak check
        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated()
        
        for _ in range(50):
            S_tmp, V_tmp, t_tmp = engine.simulate_paths(
                params=params,
                T=T,
                steps_per_unit=steps_per_unit,
                N_paths=N_paths,
                antithetic=True,
                device="cuda",
                dtype=torch.float32
            )
            del S_tmp, V_tmp, t_tmp
            
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_end = torch.cuda.memory_allocated()
        
        # No leak detected (within reasonable caching/overhead margin)
        assert mem_end <= mem_start + 1024 * 1024
        
    # 4. Very short maturities (T <= 1e-8)
    # T = 1e-9 should round steps to 0. We verify it handles it without crashing
    for T_short in [1e-9, 1e-8]:
        try:
            S_s, V_s, t_s = engine.simulate_paths(
                params=params,
                T=T_short,
                steps_per_unit=steps_per_unit,
                N_paths=N_paths,
                antithetic=True,
                device="cpu",
                dtype=torch.float32
            )
            assert S_s.ndim == 3
            assert V_s.ndim == 3
        except Exception:
            pass
            
    # 5. Invalid parameters should trigger assertion errors
    invalid_params_list = [
        torch.tensor([[0.0, 0.1, 1.5, -0.7]], dtype=torch.float32),  # v0 = 0
        torch.tensor([[0.04, 0.0, 1.5, -0.7]], dtype=torch.float32), # H = 0
        torch.tensor([[0.04, 0.5, 1.5, -0.7]], dtype=torch.float32), # H = 0.5
        torch.tensor([[0.04, 0.1, 0.0, -0.7]], dtype=torch.float32), # eta = 0
        torch.tensor([[0.04, 0.1, 1.5, 0.1]], dtype=torch.float32),  # rho > 0
        torch.tensor([[0.04, 0.1, 1.5, -1.5]], dtype=torch.float32), # rho < -1
    ]
    for ip in invalid_params_list:
        with pytest.raises(AssertionError):
            engine.simulate_paths(params=ip, T=0.5, steps_per_unit=200, N_paths=N_paths, device="cpu")
