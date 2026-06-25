import pytest
import numpy as np
import torch
from deepvol.mrm.arbitrage import check_arbitrage, project_arbitrage_free
from deepvol.calibration.fallbacks import (
    FourierCOSEngine,
    McKeanVlasovFallbackEngine,
    calibrate_tikhonov,
    calculate_psi,
    check_ood_parameters,
    get_drift_report
)

# Standard grids expected by FNO models
T_GRID_STD = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID_STD = np.linspace(-0.5, 0.5, 11)

def test_project_arbitrage_free():
    T_grid = np.array([0.1, 0.5, 1.0])
    K_grid = np.array([-0.2, 0.0, 0.2])
    
    # Create surface with severe calendar spread violation
    bad_iv = np.array([
        [0.4, 0.4, 0.4],
        [0.1, 0.1, 0.1],  # total variance decreases here!
        [0.2, 0.2, 0.2]
    ])
    
    # Check that initial has arbitrage
    res_bad = check_arbitrage(bad_iv, K_grid, T_grid)
    assert res_bad["has_arbitrage"]
    
    # Project it onto arbitrage-free space
    clean_iv = project_arbitrage_free(bad_iv, K_grid, T_grid, S=1.0)
    
    # Check that projected surface has no arbitrage
    res_clean = check_arbitrage(clean_iv, K_grid, T_grid)
    assert not res_clean["has_arbitrage"]
    assert np.all(clean_iv >= 0.01)

def test_fourier_cos_engine():
    engine = FourierCOSEngine(device="cpu")
    params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04}
    
    res = engine.price_surface(params, T_GRID_STD, K_GRID_STD, S0=1.0)
    assert "prices" in res
    assert "ivs" in res
    assert res["prices"].shape == (8, 11)
    assert res["ivs"].shape == (8, 11)
    assert np.all(res["ivs"] >= 0.01)
    assert not np.isnan(res["prices"]).any()
    assert not np.isnan(res["ivs"]).any()

def test_mckean_vlasov_fallback_engine():
    engine = McKeanVlasovFallbackEngine(device="cpu")
    T_grid = np.array([0.1, 0.5, 1.0])
    K_grid = np.array([-0.2, 0.0, 0.2])
    params = {"kappa": 2.0, "theta": 0.04, "epsilon": 0.3, "rho": -0.7}
    
    res = engine.price_surface(params, T_grid, K_grid, S0=1.0)
    assert "prices" in res
    assert "ivs" in res
    assert res["prices"].shape == (3, 3)
    assert res["ivs"].shape == (3, 3)
    assert np.all(res["ivs"] >= 0.01)
    assert not np.isnan(res["prices"]).any()
    assert not np.isnan(res["ivs"]).any()

def test_calibrate_tikhonov_heston():
    p_prior = np.array([2.0, 0.04, 0.3, -0.7, 0.04])
    
    device = "cpu"
    from deepvol.calibration.interface import _get_default_model
    from deepvol.calibration.calibrate_bfgs import _fno_predict_real_iv, _load_normalizers, _make_spatial_input
    
    model = _get_default_model("heston", torch.device(device))
    _load_normalizers("heston")
    spatial = _make_spatial_input(T_GRID_STD, K_GRID_STD, torch.device(device))
    
    with torch.no_grad():
        target_iv_t = _fno_predict_real_iv(
            model,
            torch.tensor(p_prior, dtype=torch.float32).unsqueeze(0),
            spatial
        )
        market_iv = target_iv_t.numpy()
    
    # Run Tikhonov calibration
    res = calibrate_tikhonov(
        market_iv_surface=market_iv,
        model_name="heston",
        p_prior=p_prior,
        lmbda=0.01,
        T_grid=T_GRID_STD,
        K_grid=K_GRID_STD,
        device=device,
        model=model
    )
    
    assert res.status == "converged"
    assert res.rmse < 1e-3
    assert len(res.parameters) == 5

def test_calibrate_tikhonov_rbergomi():
    p_prior = np.array([0.04, 0.07, 1.5, -0.7])
    
    device = "cpu"
    from deepvol.calibration.interface import _get_default_model
    from deepvol.calibration.calibrate_bfgs import _fno_predict_real_iv, _load_normalizers, _make_spatial_input
    
    model = _get_default_model("rbergomi", torch.device(device))
    _load_normalizers("rbergomi")
    spatial = _make_spatial_input(T_GRID_STD, K_GRID_STD, torch.device(device))
    
    with torch.no_grad():
        target_iv_t = _fno_predict_real_iv(
            model,
            torch.tensor(p_prior, dtype=torch.float32).unsqueeze(0),
            spatial
        )
        target_iv = target_iv_t.numpy()
        
    # Calibrate with regularized Tikhonov (free H)
    res = calibrate_tikhonov(
        market_iv_surface=target_iv,
        model_name="rbergomi",
        p_prior=p_prior,
        lmbda=0.01,
        T_grid=T_GRID_STD,
        K_grid=K_GRID_STD,
        device=device,
        model=model
    )
    assert res.status == "converged"
    assert res.rmse < 1e-3
    assert len(res.parameters) == 4
    
    # Calibrate with fixed H
    res_fixed = calibrate_tikhonov(
        market_iv_surface=target_iv,
        model_name="rbergomi",
        p_prior=p_prior,
        lmbda=0.01,
        T_grid=T_GRID_STD,
        K_grid=K_GRID_STD,
        fixed_H=0.07,
        device=device,
        model=model
    )
    assert res_fixed.status == "converged"
    assert res_fixed.rmse < 1e-3
    assert len(res_fixed.parameters) == 4
    assert abs(res_fixed.parameters[1] - 0.07) < 1e-7

def test_model_governance_compliance():
    # Test PSI calculation
    baseline = np.random.normal(0.04, 0.01, 100)
    actual = np.random.normal(0.045, 0.01, 100)
    psi = calculate_psi(baseline, actual)
    assert isinstance(psi, float)
    assert psi >= 0.0
    
    # Test OOD Parameter Clamping & Logging
    bad_heston_params = np.array([0.2, 0.30, 2.5, -0.99, 0.30]) # Out of bounds
    res_heston = check_ood_parameters("heston", bad_heston_params)
    assert res_heston["is_ood"]
    assert len(res_heston["logs"]) > 0
    # verify clamping
    assert 0.5 <= res_heston["clamped_params"][0] <= 10.0
    assert 0.01 <= res_heston["clamped_params"][1] <= 0.25
    assert 0.1 <= res_heston["clamped_params"][2] <= 2.0
    assert -0.95 <= res_heston["clamped_params"][3] <= -0.01
    assert 0.01 <= res_heston["clamped_params"][4] <= 0.25
    
    # Rough Bergomi OOD test
    bad_rb_params = np.array([0.30, 0.005, 5.0, 0.1]) # Out of bounds
    res_rb = check_ood_parameters("rbergomi", bad_rb_params)
    assert res_rb["is_ood"]
    assert len(res_rb["logs"]) > 0
    # verify clamping
    assert 0.01 <= res_rb["clamped_params"][0] <= 0.20
    assert 0.01 <= res_rb["clamped_params"][1] <= 0.5
    assert 0.5 <= res_rb["clamped_params"][2] <= 4.0
    assert -0.95 <= res_rb["clamped_params"][3] <= 0.0
