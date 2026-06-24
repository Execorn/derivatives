import pytest
import time
import numpy as np
from scipy.integrate import solve_ivp
from deepvol.market.vix_pricing import (
    model_vix,
    vix_futures_curve,
    model_variance_swap_rate,
    download_vix_futures,
    joint_calibration_loss
)

# -------------------------------------------------------------------------
# Performance & Speed tests
# -------------------------------------------------------------------------
def test_vix_performance():
    # Benchmark model_vix speed
    params = {
        'kappa': 1.0, 'theta': 0.08, 'sigma': 0.8, 'rho': -0.34, 'v0': 0.10, 'H': 0.08
    }
    
    # Warm-up
    _ = model_vix(**params)
    
    # Run 50 iterations to get average speed
    start_time = time.perf_counter()
    runs = 50
    for _ in range(runs):
        _ = model_vix(**params)
    end_time = time.perf_counter()
    
    avg_time_ms = ((end_time - start_time) / runs) * 1000.0
    print(f"\nAverage model_vix execution time: {avg_time_ms:.4f} ms")
    
    # Check if VIX index calculation is fast (<10ms)
    assert avg_time_ms < 10.0, f"Average VIX pricing time is {avg_time_ms:.2f}ms, which exceeds the 10ms threshold"


def test_vix_futures_performance():
    # Benchmark futures curve speed
    maturities = np.array([0.083, 0.164, 0.246, 0.328, 0.411, 0.493, 0.575, 0.657])
    params = {
        'kappa': 1.0, 'theta': 0.08, 'sigma': 0.8, 'rho': -0.34, 'v0': 0.10, 'H': 0.08,
        'maturities': maturities
    }
    
    # Warm-up
    _ = vix_futures_curve(**params)
    
    start_time = time.perf_counter()
    runs = 10
    for _ in range(runs):
        _ = vix_futures_curve(**params)
    end_time = time.perf_counter()
    
    avg_time_ms = ((end_time - start_time) / runs) * 1000.0
    print(f"\nAverage vix_futures_curve execution time: {avg_time_ms:.4f} ms")


# -------------------------------------------------------------------------
# Parameter extremes & Error boundaries
# -------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [1e-3, 0.1, 1.0, 10.0, 50.0])
@pytest.mark.parametrize("theta", [1e-3, 0.05, 0.2, 0.5])
@pytest.mark.parametrize("v0", [1e-3, 0.05, 0.2, 0.5])
@pytest.mark.parametrize("H", [0.01, 0.1, 0.25, 0.45])
def test_vix_robust_valid_combinations(kappa, theta, v0, H):
    # Test typical/boundary ranges of model_vix
    vix = model_vix(kappa=kappa, theta=theta, sigma=0.8, rho=-0.34, v0=v0, H=H)
    assert np.isfinite(vix)
    assert vix > 0.0


@pytest.mark.parametrize("H", [-0.1, 0.0, 0.5, 0.6, 1.0])
def test_vix_invalid_h(H):
    # What happens for H outside (0, 0.5)?
    # Specifically, H=0.5 or H>0.5 are standard Heston or super-smooth, H<=0 is not defined or invalid.
    try:
        vix = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=H)
        print(f"H={H} succeeded and returned VIX={vix}")
    except Exception as e:
        print(f"H={H} raised: {type(e).__name__}: {e}")
        # If it raises, that's fine, but let's document it
        pass


def test_vix_negative_parameters():
    # Test behaviour when parameters are negative (non-physical but can occur during optimizer steps)
    # kappa <= 0
    vix_neg_kappa = model_vix(kappa=-1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08)
    assert np.isfinite(vix_neg_kappa)
    
    # v0 < 0 (should return 0.0 or positive value depending on max(vix_sq, 0.0))
    vix_neg_v0 = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=-0.05, H=0.08)
    assert np.isfinite(vix_neg_v0)
    assert vix_neg_v0 >= 0.0
    
    # theta < 0
    vix_neg_theta = model_vix(kappa=1.0, theta=-0.05, sigma=0.8, rho=-0.34, v0=0.10, H=0.08)
    assert np.isfinite(vix_neg_theta)
    assert vix_neg_theta >= 0.0


def test_vix_futures_blowup_risks():
    # Let's see if we can trigger Riccati ODE blowup in vix_futures_curve
    # Blowup occurs when quadratic term (0.5 * sigma^2 * Phi^2) dominates mean reversion.
    # We test with large sigma and long maturities.
    maturities = np.array([1.0, 2.0, 5.0])
    
    # Case A: Very large volatility of volatility (sigma)
    try:
        curve_high_sigma = vix_futures_curve(
            kappa=1.0, theta=0.08, sigma=5.0, rho=-0.34, v0=0.10, H=0.08, maturities=maturities
        )
        print(f"Large sigma (5.0) succeeded: {curve_high_sigma}")
    except Exception as e:
        print(f"Large sigma (5.0) failed as expected/unexpected: {type(e).__name__}: {e}")
        
    # Case B: Very large initial variance (v0)
    try:
        curve_high_v0 = vix_futures_curve(
            kappa=1.0, theta=0.08, sigma=1.0, rho=-0.34, v0=5.0, H=0.08, maturities=maturities
        )
        print(f"Large v0 (5.0) succeeded: {curve_high_v0}")
    except Exception as e:
        print(f"Large v0 (5.0) failed: {type(e).__name__}: {e}")

    # Case C: Long maturity (T = 10.0)
    try:
        curve_long_t = vix_futures_curve(
            kappa=1.0, theta=0.08, sigma=1.5, rho=-0.34, v0=0.10, H=0.08, maturities=np.array([10.0])
        )
        print(f"Long maturity (10.0) succeeded: {curve_long_t}")
    except Exception as e:
        print(f"Long maturity (10.0) failed: {type(e).__name__}: {e}")


def test_vix_futures_edge_cases():
    # Empty maturities
    with pytest.raises(Exception):
        vix_futures_curve(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, maturities=np.array([]))
        
    # Zero maturity
    curve_zero = vix_futures_curve(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, maturities=np.array([0.0]))
    print(f"Zero maturity succeeded: {curve_zero}")
    # Note: at T=0, state_Tk is state_init.
    # psi_init at T=0 is - (y_nodes[m]**2) * psi_vix / delta
    # So vix_fut should be the spot VIX. Let's verify if curve_zero is close to spot VIX
    spot_vix = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08)
    assert np.allclose(curve_zero, spot_vix, rtol=1e-2)



# -------------------------------------------------------------------------
# Convergence tests
# -------------------------------------------------------------------------
def test_vix_ode_convergence():
    # Test convergence/sensitivity of ODE solution vs tolerances
    # By default, solve_ivp in model_vix uses rtol=1e-8, atol=1e-8.
    # Let's compare the default with looser/tighter tolerances to see if the value is stable.
    # We do a manual ODE solve inside to replicate and compare.
    kappa, theta, v0, H = 1.0, 0.08, 0.10, 0.08
    delta = 30/365
    from deepvol.models.lifted_heston import bernstein_factors
    x, c = bernstein_factors(H, N=20)
    N = len(x)
    
    def rhs_vix(s, state):
        psi = state[:N]
        Phi = np.sum(c * psi)
        dpsi = -kappa * x * psi - kappa * Phi + 1.0
        dI = Phi
        return np.concatenate([dpsi, [dI]])
        
    y0 = np.zeros(N + 1)
    
    # Tighter tol
    sol_tight = solve_ivp(rhs_vix, [0.0, delta], y0, method='RK45', rtol=1e-12, atol=1e-12)
    psi_tight = sol_tight.y[:N, -1]
    I_psi_tight = sol_tight.y[-1, -1]
    vix_sq_tight = (1.0 / delta) * (v0 * np.sum(c * psi_tight) + kappa * theta * I_psi_tight)
    vix_tight = np.sqrt(vix_sq_tight) * 100.0
    
    # Default tol
    sol_default = solve_ivp(rhs_vix, [0.0, delta], y0, method='RK45', rtol=1e-8, atol=1e-8)
    psi_default = sol_default.y[:N, -1]
    I_psi_default = sol_default.y[-1, -1]
    vix_sq_default = (1.0 / delta) * (v0 * np.sum(c * psi_default) + kappa * theta * I_psi_default)
    vix_default = np.sqrt(vix_sq_default) * 100.0
    
    diff = abs(vix_tight - vix_default)
    print(f"\nODE tolerance convergence diff: {diff:.2e}")
    assert diff < 1e-5, f"ODE solution is not converged! Diff: {diff:.2e}"


def test_vix_futures_zero_vol_lower_bound():
    # As VIX futures tend to zero volatility, they should approach 0.0.
    # We test with kappa=1.0, theta=0.0, sigma=0.8, rho=-0.34, v0=0.0, H=0.08.
    # The true mathematical value should be 0.0.
    # However, due to fixed y_max = 20.0 and Laplace transform tail approximation,
    # the code yields ~2.82%.
    curve = vix_futures_curve(
        kappa=1.0, theta=0.0, sigma=0.8, rho=-0.34, v0=0.0, H=0.08,
        maturities=np.array([0.1, 0.5])
    )
    print(f"\nZero-vol VIX futures values: {curve}")
    
    # We assert it should be close to 0.0 (this is expected to fail in the current implementation!)
    # To avoid breaking the entire test suite completely, let's print a warning or check it,
    # but let's assert curve > 2.8 to document the bug, or let's assert it is < 0.1 and mark it as failing.
    assert np.all(curve < 0.1), f"VIX futures for zero vol are not zero: {curve}"


def test_vix_calibration_normalizer_mismatch():
    import deepvol.calibration.calibrate_bfgs as calibrate
    # Initialise/clear cached normalizers
    calibrate._param_norm = None
    calibrate._iv_norm = None
    
    dummy_spx = np.full((8, 11), 0.20)
    dummy_vix_fut = np.array([14.0, 15.0, 16.0])
    vix_maturities = np.array([0.1, 0.3, 0.5])
    
    params = np.array([1.0, 0.08, 0.8, -0.34, 0.10, 0.08])
    _ = joint_calibration_loss(params, dummy_spx, dummy_vix_fut, vix_maturities)
    
    # Check what param normalizer path was loaded inside joint_calibration_loss
    loaded_path = calibrate._PARAM_NORM_PATH
    print(f"\nLoaded normalizer path: {loaded_path}")
    
    # We expect version v2 to be loaded because the model is FNO v2,
    # but the code loads v1 by default ('artifacts/models/param_normalizer.npz').
    # Let's assert it is v2. If it is v1, this test will fail, highlighting the bug.
    assert "param_normalizer_v2.npz" in loaded_path, f"Mismatched normalizer loaded: {loaded_path} (expected v2)"

