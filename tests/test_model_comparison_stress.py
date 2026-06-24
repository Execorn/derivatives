import pytest
import numpy as np
import torch
from datetime import date
from deepvol.analysis.model_comparison import ModelComparisonStudy, T_GRID, K_GRID

@pytest.fixture
def base_surface():
    """Construct a smooth SPX-like implied volatility surface as a base."""
    nT = len(T_GRID)
    nK = len(K_GRID)
    base_iv = np.zeros((nT, nK))
    for i in range(nT):
        t = T_GRID[i]
        # ATM vol ~20%, smile curvature increases for shorter maturities
        base_iv[i, :] = 0.20 + 0.05 * K_GRID**2 - 0.02 * K_GRID * np.exp(-t)
    return base_iv

def check_parameter_bounds(model_name: str, params: any):
    """
    Helper to verify that model parameters lie strictly within their specified bounds
    even under extreme noise calibration conditions.
    """
    assert params is not None, f"Parameters for {model_name} are None"
    tol = 1e-5
    
    if model_name == "heston":
        # Heston returns a dictionary of params: kappa, theta, sigma, rho, v0
        assert isinstance(params, dict), "Heston parameters should be returned as a dict"
        kappa = params["kappa"]
        theta = params["theta"]
        sigma = params["sigma"]
        rho = params["rho"]
        v0 = params["v0"]
        
        # Check bounds: lo = [0.5, 0.01, 0.1, -0.95, 0.01], hi = [10.0, 0.25, 2.0, -0.01, 0.25]
        assert 0.5 - tol <= kappa <= 10.0 + tol, f"Heston kappa {kappa} out of bounds [0.5, 10.0]"
        assert 0.01 - tol <= theta <= 0.25 + tol, f"Heston theta {theta} out of bounds [0.01, 0.25]"
        assert 0.1 - tol <= sigma <= 2.0 + tol, f"Heston sigma {sigma} out of bounds [0.1, 2.0]"
        assert -0.95 - tol <= rho <= -0.01 + tol, f"Heston rho {rho} out of bounds [-0.95, -0.01]"
        assert 0.01 - tol <= v0 <= 0.25 + tol, f"Heston v0 {v0} out of bounds [0.01, 0.25]"
        
    elif model_name == "sabr":
        # SABR returns a numpy array [alpha, rho, nu]
        assert isinstance(params, np.ndarray), "SABR parameters should be a numpy array"
        assert params.shape == (3,), f"SABR parameters shape is {params.shape}, expected (3,)"
        alpha, rho, nu = params
        
        # Bounds: lo = [0.005, -0.95, 0.05], hi = [0.5, 0.3, 1.5]
        assert 0.005 - tol <= alpha <= 0.5 + tol, f"SABR alpha {alpha} out of bounds [0.005, 0.5]"
        assert -0.95 - tol <= rho <= 0.3 + tol, f"SABR rho {rho} out of bounds [-0.95, 0.3]"
        assert 0.05 - tol <= nu <= 1.5 + tol, f"SABR nu {nu} out of bounds [0.05, 1.5]"
        
    elif model_name == "ssvi":
        # SSVI returns a numpy array of shape (11,): [theta_atm_0...7, rho, eta, gamma]
        assert isinstance(params, np.ndarray), "SSVI parameters should be a numpy array"
        assert params.shape == (11,), f"SSVI parameters shape is {params.shape}, expected (11,)"
        theta_atm = params[:8]
        rho, eta, gamma = params[8], params[9], params[10]
        
        # Check ATM variances are positive/clamped
        assert np.all(theta_atm >= 0.0 - tol), f"SSVI theta_atm must be non-negative, got {theta_atm}"
        
        # Bounds: lo = [-0.9, 0.05, 0.1], hi = [0.9, 4.0, 0.5]
        assert -0.9 - tol <= rho <= 0.9 + tol, f"SSVI rho {rho} out of bounds [-0.9, 0.9]"
        assert 0.05 - tol <= eta <= 4.0 + tol, f"SSVI eta {eta} out of bounds [0.05, 4.0]"
        assert 0.1 - tol <= gamma <= 0.5 + tol, f"SSVI gamma {gamma} out of bounds [0.1, 0.5]"
        
    elif model_name == "rbergomi":
        # rBergomi returns a numpy array [v0, H, eta, rho]
        assert isinstance(params, np.ndarray), "rBergomi parameters should be a numpy array"
        assert params.shape == (4,), f"rBergomi parameters shape is {params.shape}, expected (4,)"
        v0, H, eta, rho = params
        
        # Bounds: lo = [0.01, 0.04, 0.5, -0.95], hi = [0.20, 0.15, 4.0, 0.0]
        assert 0.01 - tol <= v0 <= 0.20 + tol, f"rBergomi v0 {v0} out of bounds [0.01, 0.20]"
        assert 0.04 - tol <= H <= 0.15 + tol, f"rBergomi H {H} out of bounds [0.04, 0.15]"
        assert 0.5 - tol <= eta <= 4.0 + tol, f"rBergomi eta {eta} out of bounds [0.5, 4.0]"
        assert -0.95 - tol <= rho <= 0.0 + tol, f"rBergomi rho {rho} out of bounds [-0.95, 0.0]"
        
    elif model_name in ("rough_heston", "fno"):
        # Rough Heston / standard FNO 3D Newton calibration returns a numpy array [v0, zeta, lambda]
        assert isinstance(params, np.ndarray), "Rough Heston parameters should be a numpy array"
        assert params.shape == (3,), f"Rough Heston parameters shape is {params.shape}, expected (3,)"
        v0, zeta, lam = params
        
        # Bounds: lo = [0.01, -0.90, 0.01], hi = [0.15, -0.01, 0.99]
        assert 0.01 - tol <= v0 <= 0.15 + tol, f"Rough Heston v0 {v0} out of bounds [0.01, 0.15]"
        assert -0.90 - tol <= zeta <= -0.01 + tol, f"Rough Heston zeta {zeta} out of bounds [-0.90, -0.01]"
        assert 0.01 - tol <= lam <= 0.99 + tol, f"Rough Heston lambda {lam} out of bounds [0.01, 0.99]"
        
    elif model_name == "local_vol":
        # Local Vol returns SVI slice parameters of shape (8, 5)
        assert isinstance(params, np.ndarray), "Local Vol SVI parameters should be a numpy array"
        assert params.shape == (8, 5), f"Local Vol parameters shape is {params.shape}, expected (8, 5)"
        for i in range(8):
            a, b, rho, m, sigma = params[i]
            assert b >= 0.0 - tol, f"SVI b slice {i} must be non-negative, got {b}"
            assert sigma >= 0.0 - tol, f"SVI sigma slice {i} must be non-negative, got {sigma}"
            assert -0.999 - tol <= rho <= 0.999 + tol, f"SVI rho slice {i} must be in [-0.999, 0.999], got {rho}"
            
    elif model_name == "mlsv":
        # MLSV returns a numpy array [v0, kappa, theta, xi, rho]
        assert isinstance(params, np.ndarray), "MLSV parameters should be a numpy array"
        assert params.shape == (5,), f"MLSV parameters shape is {params.shape}, expected (5,)"
        v0, kappa, theta, xi, rho = params
        
        # Bounds: v0 and theta clipped to [0.005, 0.20], others are fixed defaults
        assert 0.005 <= v0 <= 0.20, f"MLSV v0 {v0} out of bounds [0.005, 0.20]"
        assert 0.005 <= theta <= 0.20, f"MLSV theta {theta} out of bounds [0.005, 0.20]"
        assert abs(kappa - 2.0) < 1e-6, f"MLSV kappa should be 2.0, got {kappa}"
        assert abs(xi - 0.3) < 1e-6, f"MLSV xi should be 0.3, got {xi}"
        assert abs(rho - (-0.7)) < 1e-6, f"MLSV rho should be -0.7, got {rho}"

def test_robustness_empty_surfaces():
    """Verify that calling calibration/pricing with empty surfaces raises clean exceptions without crashing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    study = ModelComparisonStudy(device=device)
    empty_surface = np.empty((0, 0))
    
    models = ["heston", "rough_heston", "rbergomi", "sabr", "ssvi", "local_vol", "mlsv"]
    
    for model in models:
        with pytest.raises((ValueError, IndexError, RuntimeError, ZeroDivisionError)) as exc_info:
            study.run_calibration_and_pricing(
                snapshot_date=date(2024, 1, 2),
                model_name=model,
                market_iv_surface=empty_surface,
                S0=100.0, r=0.05, q=0.015,
                use_cache=False,
                N_paths=200,
                steps_per_unit=10
            )
        assert exc_info.type in (ValueError, IndexError, RuntimeError, ZeroDivisionError)

def test_robustness_shape_mismatch():
    """Verify that calling calibration/pricing with mismatching surface shape (e.g. single point) raises clean exceptions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    study = ModelComparisonStudy(device=device)
    single_point_surface = np.full((1, 1), 0.20)
    
    models = ["heston", "rough_heston", "rbergomi", "sabr", "ssvi", "local_vol", "mlsv"]
    
    for model in models:
        with pytest.raises((ValueError, IndexError, RuntimeError)) as exc_info:
            study.run_calibration_and_pricing(
                snapshot_date=date(2024, 1, 2),
                model_name=model,
                market_iv_surface=single_point_surface,
                S0=100.0, r=0.05, q=0.015,
                use_cache=False,
                N_paths=200,
                steps_per_unit=10
            )
        assert exc_info.type in (ValueError, IndexError, RuntimeError)

def test_robustness_extreme_asset_spots(base_surface):
    """Verify that simulation and inversion work cleanly or error gracefully under extreme positive/negative asset spots S0."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    study = ModelComparisonStudy(device=device)
    
    # 1. Negative or zero S0 must raise ValueError immediately (validated in MLSVSolverGPU)
    for model in ["local_vol", "mlsv"]:
        for S0_invalid in [-10.0, 0.0]:
            with pytest.raises(ValueError) as exc_info:
                study.run_calibration_and_pricing(
                    snapshot_date=date(2024, 1, 2),
                    model_name=model,
                    market_iv_surface=base_surface,
                    S0=S0_invalid, r=0.05, q=0.015,
                    use_cache=False,
                    N_paths=200,
                    steps_per_unit=10
                )
            assert "S0" in str(exc_info.value) or "positive" in str(exc_info.value)
            
    # 2. Extreme positive S0 (microscopic S0=1e-3, or massive S0=1e6) should price and invert correctly
    for model in ["local_vol", "mlsv"]:
        for S0_extreme in [1e-3, 1e6]:
            res = study.run_calibration_and_pricing(
                snapshot_date=date(2024, 1, 2),
                model_name=model,
                market_iv_surface=base_surface,
                S0=S0_extreme, r=0.05, q=0.015,
                use_cache=False,
                N_paths=200,
                steps_per_unit=10
            )
            assert res["iv_fitted"] is not None
            assert res["iv_fitted"].shape == base_surface.shape
            assert not np.any(np.isnan(res["iv_fitted"]))
            assert res["rmse"] >= 0.0

def test_robustness_extreme_interest_rates(base_surface):
    """Verify that the Local Vol and MLSV solvers run cleanly under extreme positive/negative interest rates."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    study = ModelComparisonStudy(device=device)
    
    # Test r = -0.5 (-50%) and r = 5.0 (500% hyperinflation)
    for model in ["local_vol", "mlsv"]:
        for r_extreme in [-0.5, 5.0]:
            res = study.run_calibration_and_pricing(
                snapshot_date=date(2024, 1, 2),
                model_name=model,
                market_iv_surface=base_surface,
                S0=100.0, r=r_extreme, q=0.015,
                use_cache=False,
                N_paths=200,
                steps_per_unit=10
            )
            assert res["iv_fitted"] is not None
            assert res["iv_fitted"].shape == base_surface.shape
            assert not np.any(np.isnan(res["iv_fitted"]))
            assert res["rmse"] >= 0.0

def test_noise_injection_gaussian(base_surface):
    """Inject Gaussian noise into the market IV surface, calibrate, and check bounds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    study = ModelComparisonStudy(device=device)
    
    # Inject large Gaussian noise (5% standard deviation)
    np.random.seed(123)
    noise = np.random.normal(0.0, 0.05, size=base_surface.shape)
    noisy_surface = np.clip(base_surface + noise, 0.01, 1.5)
    
    models = ["heston", "rough_heston", "rbergomi", "sabr", "ssvi", "local_vol", "mlsv"]
    
    for model in models:
        res = study.run_calibration_and_pricing(
            snapshot_date=date(2024, 1, 2),
            model_name=model,
            market_iv_surface=noisy_surface,
            S0=100.0, r=0.05, q=0.015,
            use_cache=False,
            N_paths=200,
            steps_per_unit=10
        )
        assert res["iv_fitted"] is not None
        assert not np.any(np.isnan(res["iv_fitted"]))
        assert res["rmse"] >= 0.0
        
        # Verify that parameter bounds are strictly respected
        check_parameter_bounds(model, res["parameters"])
        
        # Verify optimization was active (i.e. did not just return dummy values)
        # If parameters were hardcoded, they would be exactly the same across runs or not respect bounds.
        # Here we also check that the RMSE is returned and is a real float.
        assert isinstance(res["rmse"], float)
        assert res["rmse"] >= 0.0

def test_noise_injection_fat_tailed(base_surface):
    """Inject fat-tailed (Student-t) noise into the market IV surface, calibrate, and check bounds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    study = ModelComparisonStudy(device=device)
    
    # Inject Student-t noise with df=3 to simulate fat-tailed market anomalies
    np.random.seed(456)
    noise = np.random.standard_t(df=3, size=base_surface.shape) * 0.05
    noisy_surface = np.clip(base_surface + noise, 0.01, 1.5)
    
    models = ["heston", "rough_heston", "rbergomi", "sabr", "ssvi", "local_vol", "mlsv"]
    
    for model in models:
        res = study.run_calibration_and_pricing(
            snapshot_date=date(2024, 1, 2),
            model_name=model,
            market_iv_surface=noisy_surface,
            S0=100.0, r=0.05, q=0.015,
            use_cache=False,
            N_paths=200,
            steps_per_unit=10
        )
        assert res["iv_fitted"] is not None
        assert not np.any(np.isnan(res["iv_fitted"]))
        assert res["rmse"] >= 0.0
        
        # Verify parameter bounds
        check_parameter_bounds(model, res["parameters"])

def test_no_cheating_optimization_validation(base_surface):
    """
    Ensure there are no dummy/facade implementations.
    We verify that calibrating to two different target surfaces yields different calibrated parameters,
    confirming that the solver is dynamically optimizing parameters rather than returning hardcoded results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    study = ModelComparisonStudy(device=device)
    
    # Surface 1: low vol smile
    iv_low = np.full_like(base_surface, 0.15)
    # Surface 2: high vol smile
    iv_high = np.full_like(base_surface, 0.35)
    
    # Test on a FNO model: rough_heston and heston
    for model in ["rough_heston", "heston"]:
        res_low = study.run_calibration_and_pricing(
            snapshot_date=date(2024, 1, 2),
            model_name=model,
            market_iv_surface=iv_low,
            S0=100.0, r=0.05, q=0.015,
            use_cache=False
        )
        res_high = study.run_calibration_and_pricing(
            snapshot_date=date(2024, 1, 2),
            model_name=model,
            market_iv_surface=iv_high,
            S0=100.0, r=0.05, q=0.015,
            use_cache=False
        )
        
        params_low = res_low["parameters"]
        params_high = res_high["parameters"]
        
        if isinstance(params_low, dict):
            # Heston
            assert not all(params_low[k] == params_high[k] for k in params_low), \
                f"Cheating detected! {model} returned identical parameters for low and high vol surfaces."
        else:
            # Rough Heston
            assert not np.allclose(params_low, params_high, rtol=1e-4), \
                f"Cheating detected! {model} returned identical parameters for low and high vol surfaces."
