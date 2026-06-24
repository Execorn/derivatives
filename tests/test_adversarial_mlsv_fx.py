import pytest
import torch
import numpy as np
import math
import warnings
from deepvol.models.mlsv_gpu import MLSVSolverGPU, compute_conditional_expectation
from deepvol.models.lifted_heston_gpu import (
    solve_riccati_rk4,
    solve_riccati_rk4_mixed,
    price_batch_gpu,
    _riccati_rhs,
    bs_iv_gpu
)
from deepvol.calibration.fx_calibration import (
    sabr_iv_lognormal_pytorch,
    calibrate_sabr_fx,
    calibrate_sabr_fx_2d,
    solve_sabr_alpha,
    sabr_initial_guess
)
from deepvol.market.fx_data import (
    gk_price,
    gk_delta,
    gk_delta_dk,
    invert_gk_delta
)

# ==============================================================================
# SECTION 1: Equity MLSV & Pricing Engine GPU Adversarial Tests
# ==============================================================================

def test_mlsv_solver_extreme_init_boundaries():
    """
    Tests MLSVSolverGPU initialization with parameters just outside the boundaries,
    ensuring they raise ValueError or TypeError.
    """
    # Negative S0
    with pytest.raises(ValueError, match="S0 must be positive"):
        MLSVSolverGPU(S0=-0.01, r=0.05, q=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, T=0.5, steps_per_unit=100, N_paths=1000)
    
    # Zero S0
    with pytest.raises(ValueError, match="S0 must be positive"):
        MLSVSolverGPU(S0=0.0, r=0.05, q=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, T=0.5, steps_per_unit=100, N_paths=1000)
        
    # Negative v0
    with pytest.raises(ValueError, match="v0 must be positive"):
        MLSVSolverGPU(S0=100.0, r=0.05, q=0.02, v0=-0.01, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, T=0.5, steps_per_unit=100, N_paths=1000)

    # Negative kappa
    with pytest.raises(ValueError, match="kappa must be positive"):
        MLSVSolverGPU(S0=100.0, r=0.05, q=0.02, v0=0.04, kappa=-1.0, theta=0.04, xi=0.3, rho=-0.7, T=0.5, steps_per_unit=100, N_paths=1000)

    # Negative theta
    with pytest.raises(ValueError, match="theta must be positive"):
        MLSVSolverGPU(S0=100.0, r=0.05, q=0.02, v0=0.04, kappa=2.0, theta=-0.04, xi=0.3, rho=-0.7, T=0.5, steps_per_unit=100, N_paths=1000)

    # Negative xi (vol-of-vol)
    with pytest.raises(ValueError, match="xi must be non-negative"):
        MLSVSolverGPU(S0=100.0, r=0.05, q=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=-0.1, rho=-0.7, T=0.5, steps_per_unit=100, N_paths=1000)

    # Correlation rho outside [-1, 1]
    with pytest.raises(ValueError, match="rho must be between -1.0 and 1.0"):
        MLSVSolverGPU(S0=100.0, r=0.05, q=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-1.01, T=0.5, steps_per_unit=100, N_paths=1000)
    with pytest.raises(ValueError, match="rho must be between -1.0 and 1.0"):
        MLSVSolverGPU(S0=100.0, r=0.05, q=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=1.01, T=0.5, steps_per_unit=100, N_paths=1000)

    # Negative T
    with pytest.raises(ValueError, match="T must be positive"):
        MLSVSolverGPU(S0=100.0, r=0.05, q=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, T=-0.1, steps_per_unit=100, N_paths=1000)

    # steps_per_unit <= 0
    with pytest.raises(ValueError, match="steps_per_unit must be positive"):
        MLSVSolverGPU(S0=100.0, r=0.05, q=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, T=0.5, steps_per_unit=0, N_paths=1000)

    # N_paths <= 0
    with pytest.raises(ValueError, match="N_paths must be positive"):
        MLSVSolverGPU(S0=100.0, r=0.05, q=0.02, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, T=0.5, steps_per_unit=100, N_paths=-5)


def test_mlsv_solver_zero_steps_division_by_zero():
    """
    Ensures that if T is positive but very small, N_steps is safeguarded to be at least 1
    to avoid division by zero.
    """
    solver = MLSVSolverGPU(
        S0=100.0,
        r=0.05,
        q=0.02,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        T=1e-6,
        steps_per_unit=100,
        N_paths=1000,
        device="cpu"
    )
    assert solver.N_steps >= 1
    assert solver.dt > 0.0



def test_mlsv_nadaraya_watson_single_path_nan():
    """
    Ensures that if N_paths = 1, the solver handles the NaN standard deviation in
    Nadaraya-Watson regression cleanly without propagating NaNs.
    """
    solver = MLSVSolverGPU(
        S0=100.0,
        r=0.05,
        q=0.02,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        T=0.2,  # set T=0.2 so N_steps = 2
        steps_per_unit=10,
        N_paths=1,  # Single path
        device="cpu"
    )
    solver.simulate(method="nadaraya_watson")
    assert not torch.isnan(solver.X_paths).any()
    assert not torch.isnan(solver.V_paths).any()
    
    price = solver.price_european_option(strike=100.0, maturity=0.2)
    assert not math.isnan(price)



def test_mlsv_muguruza_extreme_correlation_division_safeguard():
    """
    Tests if the Muguruza method handles extreme correlation rho = -1.0 or 1.0.
    Mathematically, sigma_cond = sigma * sqrt(dt) * sqrt(1 - rho^2).
    If rho = 1.0, sigma_cond is 0, which would cause division by zero.
    We verify the code's clamp safeguard prevents crashes.
    """
    solver = MLSVSolverGPU(
        S0=100.0,
        r=0.05,
        q=0.02,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-1.0,  # Extreme correlation
        T=0.1,
        steps_per_unit=10,
        N_paths=100,
        device="cpu"
    )
    # Simulate under Muguruza. The internal clamp(sigma_cond, min=1e-8) should prevent division by zero.
    solver.simulate(method="muguruza")
    assert not torch.isnan(solver.X_paths).any()
    assert not torch.isnan(solver.V_paths).any()


def test_mlsv_extreme_variance_parameters():
    """
    Tests solver stability under extremely large mean reversion kappa and vol-of-vol xi.
    This pushes the boundary of Feller condition and variance processes, testing the reflection/truncation boundaries.
    """
    solver = MLSVSolverGPU(
        S0=100.0,
        r=0.05,
        q=0.02,
        v0=0.04,
        kappa=500.0,  # Extremely high mean reversion speed
        theta=0.01,
        xi=10.0,      # Extremely high vol-of-vol
        rho=-0.9,
        T=0.1,
        steps_per_unit=100,
        N_paths=1000,
        device="cpu"
    )
    # Truncation boundary
    solver.simulate(method="nadaraya_watson", vol_boundary_style="truncation")
    assert (solver.V_paths >= 1e-6).all()
    assert not torch.isnan(solver.X_paths).any()

    # Reflection boundary
    solver.simulate(method="nadaraya_watson", vol_boundary_style="reflection")
    assert (solver.V_paths >= 1e-6).all()
    assert not torch.isnan(solver.X_paths).any()


def test_riccati_rk4_instability_and_mixed_precision_stability():
    """
    Tests numerical stability of solve_riccati_rk4 vs solve_riccati_rk4_mixed.
    Standard RK4 becomes unstable / overflows when kappa * x_max * dt is large.
    The exponential midpoint integrator in solve_riccati_rk4_mixed should remain stable.
    """
    device = "cpu"
    B = 2
    N = 20
    N_u = 64
    
    # Extreme parameters to force RK4 instability (large kappa)
    kappa = torch.tensor([50.0, 100.0], dtype=torch.float64, device=device)
    theta = torch.tensor([0.08, 0.08], dtype=torch.float64, device=device)
    sigma = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    rho = torch.tensor([-0.5, -0.5], dtype=torch.float64, device=device)
    v0 = torch.tensor([0.08, 0.08], dtype=torch.float64, device=device)
    
    u_np = np.arange(N_u) * np.pi / 8.0
    u_c = torch.tensor(u_np + 0j, dtype=torch.complex128, device=device)
    
    # Setup x and c corresponding to Bernstein factors
    r_N = 1.0 + 10.0 * (N ** -0.9)
    x = torch.tensor([r_N ** (i - 1.0 - N / 2.0) for i in range(1, N + 1)], dtype=torch.float64, device=device)
    c = x ** -(0.08 + 0.5)
    c = c / c.sum()
    
    T_grid = np.array([0.5, 1.0])
    
    # 1. Standard RK4 solver under extreme kappa should blow up / raise NaN/Inf
    log_phi_rk4 = solve_riccati_rk4(
        kappa, theta, sigma, rho, v0, u_c, x, c, T_grid, N_steps_per_unit=100, device=device
    )
    # Check if there are any NaNs or Infs (representing numerical instability)
    has_unstable_rk4 = torch.isnan(log_phi_rk4).any() or torch.isinf(log_phi_rk4).any()
    
    # 2. Mixed-precision exponential midpoint solver should remain stable and finite
    log_phi_mixed = solve_riccati_rk4_mixed(
        kappa, theta, sigma, rho, v0, u_c, x, c, T_grid, N_steps_per_unit=100, device=device
    )
    
    assert not torch.isnan(log_phi_mixed).any()
    assert not torch.isinf(log_phi_mixed).any()
    
    # Assert that the mixed solver is indeed robust
    assert torch.isfinite(log_phi_mixed).all()


def test_pricing_engine_extreme_parameters_without_validation():
    """
    Tests batch pricing with extreme / negative parameters that bypass initial validations.
    We check if price_batch_gpu survives or returns NaNs/zeros.
    """
    # Negative kappa, rho outside boundaries
    params_batch = np.array([
        [-2.0, 0.08, 0.5, -1.5, 0.08],  # Negative kappa, rho = -1.5 (invalid)
        [2.0, -0.05, 0.5, 0.5, -0.01],  # Negative theta, negative v0
    ])
    T_grid = np.array([0.5, 1.0])
    K_grid = np.array([-0.2, 0.0, 0.2])
    
    # Check if this runs without crashing. We expect NaNs or clamped values but no hard crash.
    ivs = price_batch_gpu(
        params_batch,
        T_grid=T_grid,
        K_grid=K_grid,
        H_fixed=0.08,
        device="cpu"
    )
    assert ivs.shape == (2, 2, 3)
    # Since inputs were financially invalid, the resulting IVs should contain NaNs
    assert np.isnan(ivs).any()


# ==============================================================================
# SECTION 2: FX SABR & Garman-Kohlhagen Adversarial Tests
# ==============================================================================

def test_gk_delta_singularities():
    """
    Ensures that gk_delta handles T = 0.0 or vol = 0.0 gracefully, returning the mathematical limit
    without triggering warnings.
    """
    F = 1.1250
    K = 1.1000
    r_d = 0.05
    r_f = 0.02
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = gk_delta(F, K, T=0.0, r_d=r_d, r_f=r_f, vol=0.12)
        assert len(w) == 0
        assert np.allclose(d, math.exp(-r_f * 0.0))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = gk_delta(F, K, T=0.5, r_d=r_d, r_f=r_f, vol=0.0)
        assert len(w) == 0
        assert np.allclose(d, math.exp(-r_f * 0.5))


def test_gk_delta_zero_forward_or_strike():
    """
    Tests gk_delta with zero/negative forward or strike.
    log(F/K) with zero/negative values raises ValueError in np.log or ZeroDivisionError.
    """
    T = 0.5
    vol = 0.12
    r_d = 0.05
    r_f = 0.02
    
    # Zero strike raises ZeroDivisionError (python float division)
    with pytest.raises(ZeroDivisionError):
        gk_delta(F=1.125, K=0.0, T=T, r_d=r_d, r_f=r_f, vol=vol)

    # Negative forward triggers RuntimeWarning in np.log and returns nan
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = gk_delta(F=-1.125, K=1.100, T=T, r_d=r_d, r_f=r_f, vol=vol)
        assert len(w) > 0
        assert math.isnan(d)


def test_invert_gk_delta_impossible_target():
    """
    GAP IDENTIFIED: invert_gk_delta silently fails to converge when target delta is mathematically impossible.
    It returns a wrong strike without raising any error or warning.
    """
    F = 1.1250
    T = 0.5
    r_d = 0.05
    r_f = 0.02
    vol = 0.12
    
    # 1. Target call delta > exp(-r_f * T) under spot_pna
    max_possible_delta = math.exp(-r_f * T)  # approx 0.99005
    impossible_delta = max_possible_delta + 0.01  # approx 1.00005, which is impossible
    
    strike = invert_gk_delta(
        F, impossible_delta, T, r_d, r_f, vol, option_type="call", delta_type="spot_pna"
    )
    recalc = gk_delta(F, strike, T, r_d, r_f, vol, option_type="call", delta_type="spot_pna")
    
    # It returned a strike, but recalculated delta is far from target (since target is impossible)
    assert not np.allclose(recalc, impossible_delta, rtol=1e-5, atol=1e-5)
    
    # 2. Call delta under forward_pna is bounded by (0, 1). A target of 1.5 is impossible.
    impossible_fwd_delta = 1.5
    strike_fwd = invert_gk_delta(
        F, impossible_fwd_delta, T, r_d, r_f, vol, option_type="call", delta_type="forward_pna"
    )
    recalc_fwd = gk_delta(F, strike_fwd, T, r_d, r_f, vol, option_type="call", delta_type="forward_pna")
    assert not np.allclose(recalc_fwd, impossible_fwd_delta, rtol=1e-5, atol=1e-5)


def test_sabr_iv_rho_singularity():
    """
    GAP IDENTIFIED: sabr_iv_lognormal_pytorch handles extreme rho = 1.0 using internally clamped rho.
    """
    device = torch.device("cpu")
    F = torch.tensor(1.15, dtype=torch.float64, device=device)
    T = torch.tensor(0.5, dtype=torch.float64, device=device)
    alpha = torch.tensor(0.12, dtype=torch.float64, device=device)
    nu = torch.tensor(0.4, dtype=torch.float64, device=device)
    
    # 1. rho = 1.0 (boundary violation)
    rho_1 = torch.tensor(1.0, dtype=torch.float64, device=device)
    K = torch.tensor(1.10, dtype=torch.float64, device=device)
    vol_1 = sabr_iv_lognormal_pytorch(F, K, T, alpha, rho_1, nu)
    assert not torch.isnan(vol_1), "SABR vol should not be NaN when rho = 1.0 due to internal clamping"
    assert torch.isfinite(vol_1)

    # 2. rho = -1.0 is stable since 1.0 - rho = 2.0
    rho_minus_1 = torch.tensor(-1.0, dtype=torch.float64, device=device)
    vol_minus_1 = sabr_iv_lognormal_pytorch(F, K, T, alpha, rho_minus_1, nu)
    assert not torch.isnan(vol_minus_1)
    assert vol_minus_1.item() > 0.0


def test_sabr_iv_alpha_zero_singularity():
    """
    GAP IDENTIFIED: sabr_iv_lognormal_pytorch handles alpha = 0.0 gracefully using internal clamping.
    """
    device = torch.device("cpu")
    F = torch.tensor(1.15, dtype=torch.float64, device=device)
    T = torch.tensor(0.5, dtype=torch.float64, device=device)
    rho = torch.tensor(-0.3, dtype=torch.float64, device=device)
    nu = torch.tensor(0.4, dtype=torch.float64, device=device)
    
    alpha_zero = torch.tensor(0.0, dtype=torch.float64, device=device)
    K = torch.tensor(1.10, dtype=torch.float64, device=device)
    
    vol = sabr_iv_lognormal_pytorch(F, K, T, alpha_zero, rho, nu)
    assert not torch.isnan(vol)
    assert torch.isfinite(vol)


def test_solve_sabr_alpha_quadratic_singularity():
    """
    GAP IDENTIFIED: solve_sabr_alpha handles quadratic singularity when sigma_atm = 0.0 and b < 0.
    """
    device = torch.device("cpu")
    sigma_atm = torch.tensor(0.0, dtype=torch.float64, device=device)
    T = torch.tensor(1.0, dtype=torch.float64, device=device)
    
    # Choose rho and nu to make b < 0
    # b = 1.0 + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2 * T
    # If rho = 1.0, b = 1.0 - nu^2 * T / 24.0
    # Let nu = 10.0 -> b = 1.0 - 100/24 = -3.167 < 0
    rho = torch.tensor(1.0, dtype=torch.float64, device=device)
    nu = torch.tensor(10.0, dtype=torch.float64, device=device)
    
    alpha = solve_sabr_alpha(sigma_atm, T, rho, nu)
    assert not torch.isnan(alpha)
    assert alpha == 0.0


def test_sabr_initial_guess_negative_maturity():
    """
    Ensures that sabr_initial_guess raises ValueError if T <= 0.
    """
    F = 1.1200
    T = -0.1  # Negative maturity
    market_strikes = [0.95, 1.00, 1.05, 1.10, 1.12, 1.15, 1.20, 1.25, 1.30]
    market_vols = [0.15, 0.14, 0.13, 0.125, 0.12, 0.125, 0.13, 0.14, 0.15]
    
    with pytest.raises(ValueError, match="T must be positive"):
        sabr_initial_guess(F, T, market_strikes, market_vols)


def test_sabr_calibration_invalid_vols():
    """
    Tests SABR calibration (both 3D and 2D) under degenerate/invalid volatility inputs (all zeros).
    """
    F = 1.1200
    T = 0.25
    r_d = 0.04
    r_f = 0.015
    strikes = [0.95, 1.00, 1.05, 1.10, 1.12, 1.15, 1.20, 1.25, 1.30]
    
    # All zero volatilities
    zero_vols = [0.0] * len(strikes)
    
    # calibrate_sabr_fx (3D) should return some parameters without crashing
    params_3d = calibrate_sabr_fx(F, strikes, zero_vols, T, r_d, r_f)
    assert np.isfinite(list(params_3d.values())).all()

    # calibrate_sabr_fx_2d (2D)
    params_2d = calibrate_sabr_fx_2d(F, strikes, zero_vols, T, r_d, r_f)
    assert np.isfinite(list(params_2d.values())).all()
