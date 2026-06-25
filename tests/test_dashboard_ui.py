import os
import sys
import numpy as np
from streamlit.testing.v1 import AppTest

# Add src/ to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from deepvol.app.dashboard import (  # noqa: E402
    decimate_grid_to_30x30,
    compute_psi,
    check_ood_and_clamp,
    check_arbitrage_violations,
    compute_greeks_surface
)

def test_decimate_grid_to_30x30():
    # Test grid smaller than 30x30 is unchanged
    z_small = np.random.rand(10, 15)
    x_small = np.linspace(-0.5, 0.5, 15)
    y_small = np.linspace(0.1, 2.0, 10)
    z_dec, x_dec, y_dec = decimate_grid_to_30x30(z_small, x_small, y_small)
    assert z_dec.shape == (10, 15)
    assert len(x_dec) == 15
    assert len(y_dec) == 10

    # Test grid larger than 30x30 is decimated
    z_large = np.random.rand(50, 45)
    x_large = np.linspace(-0.5, 0.5, 45)
    y_large = np.linspace(0.1, 2.0, 50)
    z_dec, x_dec, y_dec = decimate_grid_to_30x30(z_large, x_large, y_large)
    assert z_dec.shape[0] <= 30
    assert z_dec.shape[1] <= 30
    assert len(x_dec) == z_dec.shape[1]
    assert len(y_dec) == z_dec.shape[0]

def test_compute_psi():
    # Identical distributions should have low/near-zero PSI
    rng = np.random.default_rng(seed=42)
    expected = rng.normal(0.5, 0.05, 2000)
    actual = rng.normal(0.5, 0.05, 2000)
    psi = compute_psi(actual, expected)
    assert psi < 0.1

    # Significantly drifted distributions should have high PSI (>0.1)
    actual_drifted = rng.normal(0.7, 0.05, 2000)
    psi_drifted = compute_psi(actual_drifted, expected)
    assert psi_drifted > 0.1

def test_check_ood_and_clamp():
    # Test valid parameters are not modified
    valid_params = {"kappa": 2.0, "theta": 0.05, "sigma": 0.3, "rho": -0.6, "v0": 0.05}
    clamped, logs = check_ood_and_clamp("Classic Heston", valid_params)
    assert clamped == valid_params
    assert len(logs) == 0

    # Test out-of-distribution parameters are clamped and logged
    ood_params = {"kappa": 6.0, "theta": 0.001, "sigma": 2.0, "rho": 0.5, "v0": -0.05}
    clamped, logs = check_ood_and_clamp("Classic Heston", ood_params)
    assert clamped["kappa"] == 5.0
    assert clamped["theta"] == 0.005
    assert clamped["sigma"] == 1.5
    assert clamped["rho"] == 0.0
    assert clamped["v0"] == 0.005
    assert len(logs) > 0
    assert "OOD COMPLIANCE ALERT" in logs[0]

def test_check_arbitrage_violations():
    # 1. Clean surface should have no violations
    iv_clean = np.full((8, 11), 0.20)
    violations = check_arbitrage_violations(iv_clean, S=100.0, r=0.05, q=0.01)
    assert len(violations) == 0

    # 2. Calendar arbitrage (short maturity has higher total variance than long maturity)
    iv_cal = np.full((8, 11), 0.20)
    # w = sigma^2 * T
    # T[0] = 0.1, T[1] = 0.3
    # let's set iv[0] = 0.60, so w[0] = 0.36 * 0.1 = 0.036
    # let's set iv[1] = 0.20, so w[1] = 0.04 * 0.3 = 0.012
    # w[0] > w[1], calendar arbitrage violation!
    iv_cal[0, :] = 0.60
    iv_cal[1, :] = 0.20
    violations_cal = check_arbitrage_violations(iv_cal, S=100.0, r=0.05, q=0.01)
    calendar_alerts = [v for v in violations_cal if v["Type"] == "Calendar Arbitrage"]
    assert len(calendar_alerts) > 0

    # 3. Butterfly arbitrage (convexity breach in strike)
    iv_but = np.full((8, 11), 0.20)
    # Put a massive spike in implied vol at strike index 5
    iv_but[:, 5] = 0.90
    violations_but = check_arbitrage_violations(iv_but, S=100.0, r=0.05, q=0.01)
    butterfly_alerts = [v for v in violations_but if v["Type"] == "Butterfly Arbitrage"]
    assert len(butterfly_alerts) > 0

def test_compute_greeks_surface():
    iv_surf = np.full((8, 11), 0.25)
    greeks = compute_greeks_surface(iv_surf, S=100.0, r=0.05, q=0.01)
    for name in ["delta", "gamma", "vega", "theta", "vanna", "volga"]:
        assert name in greeks
        assert greeks[name].shape == (8, 11)
        assert np.all(np.isfinite(greeks[name]))

def test_app_test_console_render():
    app_file = os.path.join(project_root, "src", "deepvol/app/dashboard.py")
    at = AppTest.from_file(app_file)
    # Streamlit testing run
    at.run(timeout=15)
    assert not at.exception
    
    # Check tabs are present
    assert at.tabs[0].label == "Calibration Sandbox"
    assert at.tabs[1].label == "Live Greeks & Arbitrage Console"

def test_app_test_simulation_run():
    app_file = os.path.join(project_root, "src", "deepvol/app/dashboard.py")
    at = AppTest.from_file(app_file)
    at.run(timeout=15)
    assert not at.exception

    # Select Classic Heston model
    at.sidebar.selectbox("model_selector").select("Classic Heston").run()
    assert not at.exception

    # Toggle live simulation feed checkbox
    checkbox = at.checkbox("run_sim_checkbox")
    assert checkbox is not None
    checkbox.check().run()
    assert not at.exception
    
    # Check that live spot and arbitrage alerts are in session state
    assert "live_spot" in at.session_state
    assert "arbitrage_alerts" in at.session_state
