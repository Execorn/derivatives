import pytest
import torch
import numpy as np
import math

from deepvol.calibration.rkhs_mlsv import (
    compute_rkhs_conditional_expectation,
    RKHSMLSVSolver,
    RKHSMLSVEngine,
)
from deepvol.models.mlsv_gpu import compute_conditional_expectation


def test_rkhs_vs_nadaraya_watson_synthetic():
    """
    Test comparing RKHS sparse landmark ridge regression against the Nadaraya-Watson
    reference on a synthetic function E[V_t | X_t] = f(X_t).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    N_paths = 10000
    L = 50
    
    # Generate synthetic conditioning variable X_t
    X_t = torch.randn(N_paths, device=device, dtype=torch.float64)
    # Target function: f(x) = 0.04 + 0.02 * sin(x)
    def true_f(x):
        return 0.04 + 0.02 * torch.sin(x)
    # Generate dependent variable V_t with noise
    noise = 0.005 * torch.randn(N_paths, device=device, dtype=torch.float64)
    V_t = true_f(X_t) + noise
    # Clamp to avoid non-positive variance
    V_t = torch.clamp(V_t, min=1e-6)
    
    # Evaluation grid (targets)
    targets = torch.linspace(-2.0, 2.0, 100, device=device, dtype=torch.float64)
    
    # 1. Nadaraya-Watson reference estimation (using float32 as in the source)
    est_nw = compute_conditional_expectation(
        X_t=X_t.to(torch.float32),
        V_t=V_t.to(torch.float32),
        targets=targets.to(torch.float32),
        method="nadaraya_watson",
        block_size=512,
    ).to(torch.float64)
    
    # 2. RKHS sparse landmark ridge regression
    est_rkhs = compute_rkhs_conditional_expectation(
        X_t=X_t,
        V_t=V_t,
        targets=targets,
        num_landmarks=L,
        lambda_reg=1e-5,
    )
    
    # Check shape and no NaNs
    assert est_rkhs.shape == targets.shape
    assert not torch.isnan(est_rkhs).any()
    
    # Verify that RKHS recovers the true underlying expectation function with low MSE
    mse_to_true = torch.mean((est_rkhs - true_f(targets)) ** 2).item()
    print(f"RKHS MSE to True Function: {mse_to_true:.6f}")
    assert mse_to_true < 1e-4, f"RKHS failed to recover true expectation, MSE: {mse_to_true:.6f}"
    
    # Verify that RKHS and Nadaraya-Watson are close (MSE < 1e-4)
    mse_to_nw = torch.mean((est_rkhs - est_nw) ** 2).item()
    print(f"RKHS MSE to Nadaraya-Watson: {mse_to_nw:.6f}")
    assert mse_to_nw < 1e-4, f"RKHS and Nadaraya-Watson diverged, MSE: {mse_to_nw:.6f}"


def test_rkhs_solver_simulation_cpu():
    """
    Test RKHSMLSVSolver simulation and option pricing on CPU.
    """
    def dup_vol_fn(t, s):
        return torch.full_like(s, 0.2)
    solver = RKHSMLSVSolver(
        S0=100.0,
        r=0.05,
        q=0.02,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        T=0.25,
        steps_per_unit=50,
        N_paths=1000,
        dupire_vol_fn=dup_vol_fn,
        device="cpu",
        dtype=torch.float64,
    )
    
    # Simulate path using RKHS method
    solver.simulate(method="rkhs", num_landmarks=30, lambda_reg=1e-4)
    
    assert solver.X_paths.shape == (13, 1000)
    assert solver.V_paths.shape == (13, 1000)
    assert (solver.V_paths >= 1e-6).all()
    assert not torch.isnan(solver.X_paths).any()
    assert not torch.isnan(solver.V_paths).any()
    
    # Price European options
    call_price = solver.price_european_option(strike=100.0, maturity=0.25, is_call=True)
    put_price = solver.price_european_option(strike=100.0, maturity=0.25, is_call=False)
    
    assert call_price > 0.0
    assert put_price > 0.0
    
    # Call-put parity check: C - P = S_0 * e^{-q T} - K * e^{-r T} (approximately within statistical error)
    expected_parity = solver.S0 * math.exp(-solver.q * solver.T) - 100.0 * math.exp(-solver.r * solver.T)
    actual_parity = call_price - put_price
    assert abs(actual_parity - expected_parity) < 1.0  # within 1 USD tolerance for 1000 paths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rkhs_solver_simulation_gpu():
    """
    Test RKHSMLSVSolver simulation and option pricing on GPU.
    """
    def dup_vol_fn(t, s):
        return torch.full_like(s, 0.2)
    solver = RKHSMLSVSolver(
        S0=100.0,
        r=0.05,
        q=0.02,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        T=0.25,
        steps_per_unit=50,
        N_paths=2000,
        dupire_vol_fn=dup_vol_fn,
        device="cuda",
        dtype=torch.float64,
    )
    
    solver.simulate(method="rkhs", num_landmarks=50, lambda_reg=1e-4)
    
    assert solver.X_paths.is_cuda
    assert solver.V_paths.is_cuda
    
    # Check that option prices are computed on GPU and are smooth
    strikes = np.array([90.0, 100.0, 110.0])
    prices = solver.price_european_option(strike=strikes, maturity=0.25)
    
    assert len(prices) == 3
    assert (prices > 0.0).all()
    assert prices.is_cuda


def test_rkhs_local_vol_recovery():
    """
    Verify that RKHS recovers a smooth local volatility surface.
    """
    engine = RKHSMLSVEngine(
        kappa=2.0,
        theta=0.04,
        epsilon=0.3,
        rho=-0.7,
        device="cpu",
    )
    
    spot_grid = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    time_grid = np.array([0.1, 0.2])
    market_prices = np.zeros((len(time_grid), len(spot_grid)))  # Dummy
    
    local_vol = engine.calibrate_local_vol(
        spot_grid=spot_grid,
        time_grid=time_grid,
        market_prices=market_prices,
        num_landmarks=30,
        lambda_reg=1e-4,
    )
    
    assert local_vol.shape == (2, 5)
    assert np.all(local_vol >= 0.05)
    assert np.all(local_vol <= 1.5)
    assert not np.isnan(local_vol).any()
    
    # Check that the local volatility is relatively smooth along strikes
    # (i.e. second derivative is not exploding)
    diff2 = np.diff(local_vol, n=2, axis=1)
    assert np.all(np.abs(diff2) < 0.2)
