import pytest
import numpy as np
import os
import sys

# Ensure src path is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from greeks.pnl_attribution import pnl_attribution

def test_pnl_attribution_scalar_div():
    """
    Verify portfolio P&L attribution with a scalar d_iv_surface.
    """
    portfolio = [
        {
            "quantity": 2.0,
            "notional": 100.0,
            "delta": 0.60,
            "gamma": 0.02,
            "vega": 15.0,
            "vanna": -0.5,
            "volga": 0.8,
            "actual_pnl": 2.5
        },
        {
            "quantity": -1.0,
            "notional": 100.0,
            "delta": -0.40,
            "gamma": 0.015,
            "vega": 12.0,
            "vanna": 0.3,
            "volga": 0.6,
            "actual_pnl": -1.2
        }
    ]
    
    dS = 1.5
    d_iv = 0.02
    
    # Run attribution
    res = pnl_attribution(portfolio, dS, d_iv)
    
    # Check keys
    assert "explained_pnl" in res
    assert "actual_pnl" in res
    assert "residual" in res
    assert "breakdown" in res
    
    # Verify math
    # Option 1 weights: quantity=2.0, notional=100.0 -> weight=200.0
    # Option 2 weights: quantity=-1.0, notional=100.0 -> weight=-100.0
    
    w_delta_1 = 0.60 * 200.0
    w_gamma_1 = 0.02 * 200.0
    w_vega_1 = 15.0 * 200.0
    w_vanna_1 = -0.5 * 200.0
    w_volga_1 = 0.8 * 200.0
    
    w_delta_2 = -0.40 * (-100.0)
    w_gamma_2 = 0.015 * (-100.0)
    w_vega_2 = 12.0 * (-100.0)
    w_vanna_2 = 0.3 * (-100.0)
    w_volga_2 = 0.6 * (-100.0)
    
    delta_pnl_1 = w_delta_1 * dS
    gamma_pnl_1 = 0.5 * w_gamma_1 * (dS ** 2)
    vega_pnl_1 = w_vega_1 * d_iv
    vanna_pnl_1 = w_vanna_1 * dS * d_iv
    volga_pnl_1 = w_volga_1 * (d_iv ** 2)
    
    delta_pnl_2 = w_delta_2 * dS
    gamma_pnl_2 = 0.5 * w_gamma_2 * (dS ** 2)
    vega_pnl_2 = w_vega_2 * d_iv
    vanna_pnl_2 = w_vanna_2 * dS * d_iv
    volga_pnl_2 = w_volga_2 * (d_iv ** 2)
    
    expected_delta = delta_pnl_1 + delta_pnl_2
    expected_gamma = gamma_pnl_1 + gamma_pnl_2
    expected_vega = vega_pnl_1 + vega_pnl_2
    expected_vanna = vanna_pnl_1 + vanna_pnl_2
    expected_volga = volga_pnl_1 + volga_pnl_2
    
    expected_explained = expected_delta + expected_gamma + expected_vega + expected_vanna + expected_volga
    expected_actual = 2.5 + (-1.2)
    expected_residual = expected_actual - expected_explained
    
    bd = res["breakdown"]
    assert np.isclose(bd["delta_pnl"], expected_delta)
    assert np.isclose(bd["gamma_pnl"], expected_gamma)
    assert np.isclose(bd["vega_pnl"], expected_vega)
    assert np.isclose(bd["vanna_pnl"], expected_vanna)
    assert np.isclose(bd["volga_pnl"], expected_volga)
    
    assert np.isclose(res["explained_pnl"], expected_explained)
    assert np.isclose(res["actual_pnl"], expected_actual)
    assert np.isclose(res["residual"], expected_residual)

def test_pnl_attribution_1d_div():
    """
    Verify portfolio P&L attribution with a 1D d_iv_surface.
    """
    portfolio = [
        {
            "total_delta": 120.0,
            "total_gamma": 4.0,
            "total_vega": 3000.0,
            "total_vanna": -100.0,
            "total_volga": 160.0,
            "price_before": 10.0,
            "price_after": 10.15,
            "quantity": 10.0,
            "notional": 100.0
        }
    ]
    dS = 0.5
    d_iv = np.array([0.01])
    
    res = pnl_attribution(portfolio, dS, d_iv)
    
    # Verify math (already scaled since total_ prefix is used)
    delta_pnl = 120.0 * dS
    gamma_pnl = 0.5 * 4.0 * (dS ** 2)
    vega_pnl = 3000.0 * d_iv[0]
    vanna_pnl = -100.0 * dS * d_iv[0]
    volga_pnl = 160.0 * (d_iv[0] ** 2)
    
    expected_explained = delta_pnl + gamma_pnl + vega_pnl + vanna_pnl + volga_pnl
    expected_actual = (10.15 - 10.0) * 10.0 * 100.0 # = 150.0
    
    assert np.isclose(res["explained_pnl"], expected_explained)
    assert np.isclose(res["actual_pnl"], expected_actual)
    assert np.isclose(res["residual"], expected_actual - expected_explained)
