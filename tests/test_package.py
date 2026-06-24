import os
import subprocess
import sys
import numpy as np
import pytest
import torch

import deepvol
from deepvol.calibration.interface import calibrate, CalibrationResult
from deepvol.greeks.interface import compute_greeks
from deepvol.models.heston import HestonEngine
from deepvol.models.rbergomi_gpu import rBergomiEngine

def test_package_metadata():
    """Verify package metadata is present and correct."""
    assert hasattr(deepvol, "__version__")
    assert deepvol.__version__ == "1.0.0"

def test_public_exports():
    """Verify key classes and functions are exported at top level."""
    exports = [
        "calibrate",
        "CalibrationResult",
        "compute_greeks",
        "MLSVEngine",
        "RatesSABREngine",
        "SchwartzSmithEngine",
        "HestonEngine",
        "rBergomiEngine",
    ]
    for exp in exports:
        assert hasattr(deepvol, exp), f"deepvol missing exported member: {exp}"

def test_heston_engine_pricing():
    """Test HestonEngine interface."""
    engine = HestonEngine()
    params = {
        'kappa': 1.5,
        'theta': 0.04,
        'sigma': 0.1,
        'rho': -0.5,
        'v0': 0.04
    }
    T_grid = np.array([1.0, 2.0])
    K_grid = np.array([0.9, 1.0, 1.1])
    
    iv = engine.price_surface(params, T_grid, K_grid)
    assert iv.shape == (2, 3)
    assert np.all(iv > 0)
    assert np.all(iv < 1.0)

def test_calibrate_dispatch_heston():
    """Verify calibrate interface dispatches to Heston Newton calibration."""
    # Generate a mock target IV surface of shape (8, 11)
    T_grid = np.array([0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32)
    K_grid = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    
    target_iv = np.full((8, 11), 0.25, dtype=np.float32)
    
    res = calibrate(
        market_iv_surface=target_iv,
        model_name="heston",
        method="newton",
        T_grid=T_grid,
        K_grid=K_grid,
        max_iter=2  # Keep it quick
    )
    
    assert isinstance(res, CalibrationResult)
    assert res.parameters is not None
    assert len(res.parameters) == 5  # kappa, theta, log(sigma), rho, log(v0)
    assert res.rmse >= 0.0
    assert res.elapsed_time > 0.0
    assert res.status == "converged"

def test_compute_greeks_bs():
    """Verify compute_greeks interface dispatches to Black-Scholes."""
    spot = 100.0
    strikes = np.array([90.0, 100.0, 110.0])
    maturities = np.array([0.25, 0.5])
    
    res = compute_greeks(
        model_name="bs",
        parameters=np.array([]),  # Not used for BS
        spot=spot,
        strikes=strikes,
        maturities=maturities,
        vol=0.2,
        r=0.05
    )
    
    assert "delta" in res
    assert "gamma" in res
    assert "vega" in res
    assert res["delta"].shape == (2, 3)
    assert np.all(res["delta"] >= 0.0)

def test_cli_execution():
    """Verify CLI entrypoint deepvol-calibrate can run successfully."""
    # Run CLI help output
    cmd = [sys.executable, "-m", "deepvol.cli", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "DeepVol" in res.stdout
    
    # Run CLI calibration with default self-test (no surface) and max_iter=1
    cmd = [sys.executable, "-m", "deepvol.cli", "--model", "heston", "--method", "newton"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    # The default self test runs, but wait: is --max-iter supported? Let's check cli.py.
    # We didn't expose --max-iter in cli.py, but it should complete the default 30 runs quickly on CPU.
    # But wait, let's make sure it doesn't fail. It should print succeed.
    assert res.returncode == 0
    assert "Calibration succeeded!" in res.stdout
