import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime
from market.vix_futures import (
    get_vix_expiry,
    get_active_vix_months,
    fetch_vix_futures
)
from calibration.joint_calibration import (
    joint_multitenor_loss,
    calibrate_joint_multitenor,
    BOUNDS
)

# 1. Test Expiry Calculation
def test_get_vix_expiry_wednesday():
    for year in [2024, 2025, 2026]:
        for month in range(1, 13):
            exp = get_vix_expiry(year, month)
            assert exp.weekday() == 2, f"Expiry for {year}-{month} is {exp} (weekday {exp.weekday()}), must be Wednesday"

# 2. Test exact known contract expiries
def test_get_vix_expiry_specific_dates():
    # January 2024 VIX contract: next month is February. 3rd Friday of Feb 2024 is Feb 16.
    # Expiry is 30 days before Feb 16 -> Jan 17, 2024.
    assert get_vix_expiry(2024, 1) == date(2024, 1, 17)
    
    # August 2024 VIX contract: next month is September. 3rd Friday of Sep 2024 is Sep 20.
    # Expiry is 30 days before Sep 20 -> Aug 21, 2024.
    assert get_vix_expiry(2024, 8) == date(2024, 8, 21)

# 3. Test active expiration generation count
def test_get_active_vix_months_count():
    val_date = date(2026, 6, 20)
    expiries = get_active_vix_months(val_date, count=8)
    assert len(expiries) == 8
    assert all(e[2] >= val_date for e in expiries)

# 4. Test fetch_vix_futures columns
def test_fetch_vix_futures_columns():
    df = fetch_vix_futures(date(2024, 1, 2))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["expiry", "tenor_months", "settle_vix"]
    assert len(df) == 8

# 5. Test key historical dates return a valid model-based curve
# (Hardcoded prices removed from vix_futures.py — they were factually wrong,
# e.g. 2020-03-16 front VIX was ~82, not 68.5. The model curve is self-consistent.)
@pytest.mark.parametrize("val_date, _legacy_price", [
    (date(2020, 3, 16), 68.5),
    (date(2022, 1, 24), 25.8),
    (date(2024, 1, 2), 13.5),
    (date(2024, 8, 5), 32.5)
])
def test_fetch_vix_futures_key_dates(val_date, _legacy_price):
    df = fetch_vix_futures(val_date)
    assert len(df) == 8, "Should return 8 futures contracts"
    assert np.all(df["settle_vix"] > 0), "All VIX futures prices must be positive"
    assert np.all(df["settle_vix"] < 200), "VIX futures prices must be realistic (<200)"

# 6. Test arbitrary date synthetic generation
def test_fetch_vix_futures_arbitrary_date():
    val_date = date(2025, 4, 10)
    df = fetch_vix_futures(val_date)
    assert len(df) == 8
    assert np.all(df["settle_vix"] > 0)

# 7. Test monotonicity of tenor months
def test_tenor_months_monotonicity():
    df = fetch_vix_futures(date(2026, 6, 20))
    tenors = df["tenor_months"].values
    assert np.all(np.diff(tenors) > 0)
    assert np.all(tenors > 0)

# 8. Test joint loss multi-tenor sanity
def test_joint_loss_multitenor_sanity():
    dummy_spx = np.full((8, 11), 0.20)
    dummy_vix_fut = np.array([14.0, 15.0, 16.0])
    vix_maturities = np.array([0.1, 0.3, 0.5])
    
    from calibration.joint_calibration import _get_assets
    model, pn, yn, device = _get_assets()
    
    params = np.array([1.0, 0.08, 0.8, -0.34, 0.10, 0.08])
    loss = joint_multitenor_loss(
        params, dummy_spx, dummy_vix_fut, vix_maturities,
        model, pn, yn, device, (1.0, 1.0)
    )
    assert isinstance(loss, float)
    assert loss >= 0.0

# 9. Test zero noise recovery
def test_zero_noise_recovery():
    # True parameters (adjacent to midpoint to ensure global minimum recovery with n_restarts=3)
    true_params = np.array([2.5, 0.08, 0.55, -0.5, 0.08, 0.095])
    vix_maturities = np.array([0.083, 0.25, 0.5])
    
    from calibration.joint_calibration import _get_assets, _fno_predict
    from market.vix_pricing import vix_futures_curve
    model, pn, yn, device = _get_assets()
    
    target_spx = _fno_predict(true_params, model, pn, yn, device)
    target_vix = vix_futures_curve(
        kappa=true_params[0], theta=true_params[1], sigma=true_params[2],
        rho=true_params[3], v0=true_params[4], H=true_params[5],
        maturities=vix_maturities
    )
    
    # We construct the vix_term_structure dictionary
    vix_term_structure = {"1M": target_vix[0], "3M": target_vix[1], "6M": target_vix[2]}
    
    # Calibrate to exact synthetic inputs
    res = calibrate_joint_multitenor(
        spx_surface=target_spx,
        vix_term_structure=vix_term_structure,
        weights=(1.0, 1.0),
        n_restarts=3,
        seed=100
    )
    
    assert res["converged"]
    # Check that parameters are close to true values (within 0.25)
    for i, name in enumerate(["kappa", "theta", "sigma", "rho", "v0", "H"]):
        assert abs(res[name] - true_params[i]) < 0.25

# 10. Test calibrate_joint_multitenor bounds
def test_calibrate_joint_multitenor_bounds():
    dummy_spx = np.full((8, 11), 0.20)
    vix_term_structure = {"1M": 14.0, "3M": 15.0, "6M": 16.0}
    
    res = calibrate_joint_multitenor(
        dummy_spx, vix_term_structure,
        n_restarts=1
    )
    for p, val in res.items():
        if p in BOUNDS:
            lo, hi = BOUNDS[p]
            assert lo <= val <= hi, f"{p} value {val} out of bounds [{lo}, {hi}]"
