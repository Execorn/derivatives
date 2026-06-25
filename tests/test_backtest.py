import pytest
import torch
import numpy as np
from datetime import date
from unittest.mock import patch
import pandas as pd

from deepvol.hedging.backtest import (
    compute_bs_greeks,
    get_whalley_wilmott_beta,
    attribute_pnl_daily,
    run_empirical_backtest
)

def test_compute_bs_greeks_double():
    """Verify that Black-Scholes Greeks are computed in double precision and handle clamping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = torch.tensor(100.0, dtype=torch.float64, device=device)
    K = torch.tensor(100.0, dtype=torch.float64, device=device)
    T = torch.tensor(0.5, dtype=torch.float64, device=device)
    r = torch.tensor(0.05, dtype=torch.float64, device=device)
    q = torch.tensor(0.01, dtype=torch.float64, device=device)
    
    # 1. Standard Volatility
    sigma = torch.tensor(0.20, dtype=torch.float64, device=device)
    is_call = torch.tensor(1.0, dtype=torch.float64, device=device)
    
    g_call = compute_bs_greeks(S, K, T, r, q, sigma, is_call)
    
    assert g_call["price"].dtype == torch.float64
    assert g_call["delta"].dtype == torch.float64
    assert g_call["gamma"].dtype == torch.float64
    
    # Delta of ATM call should be around 0.5 - 0.6
    assert 0.5 < g_call["delta"].item() < 0.6
    
    # 2. Check Put Delta
    is_call_put = torch.tensor(0.0, dtype=torch.float64, device=device)
    g_put = compute_bs_greeks(S, K, T, r, q, sigma, is_call_put)
    assert -0.5 < g_put["delta"].item() < -0.4
    
    # Put-call parity for delta: Delta_Call - Delta_Put = exp(-q*T)
    expected_diff = torch.exp(-q * T).item()
    assert np.isclose(g_call["delta"].item() - g_put["delta"].item(), expected_diff)

    # 3. Check Volatility Clamping (Durrleman mathematical singularity check)
    tiny_sigma = torch.tensor(0.001, dtype=torch.float64, device=device)
    g_clamp = compute_bs_greeks(S, K, T, r, q, tiny_sigma, is_call)
    # The kernel must clamp sigma to 0.01, so we should not get NaN or division-by-zero
    assert not torch.isnan(g_clamp["price"])
    assert not torch.isnan(g_clamp["delta"])
    assert not torch.isnan(g_clamp["gamma"])
    assert g_clamp["delta"].item() > 0.0


def test_get_whalley_wilmott_beta():
    """Verify Whalley-Wilmott band half-width calculation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = torch.tensor(100.0, dtype=torch.float64, device=device)
    gamma = torch.tensor(0.05, dtype=torch.float64, device=device)
    c_S = 0.0001
    
    beta = get_whalley_wilmott_beta(S, gamma, c_S, risk_aversion=1.0)
    assert beta.dtype == torch.float64
    assert beta.item() > 0.0
    
    # If gamma increases, beta should increase
    gamma_large = torch.tensor(0.10, dtype=torch.float64, device=device)
    beta_large = get_whalley_wilmott_beta(S, gamma_large, c_S, risk_aversion=1.0)
    assert beta_large.item() > beta.item()


def test_attribute_pnl_daily_identity():
    """Verify that daily P&L attribution identity holds exactly: explained + residual == actual."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_prev = torch.tensor(100.0, dtype=torch.float64, device=device)
    S_curr = torch.tensor(102.0, dtype=torch.float64, device=device)
    sig_prev = torch.tensor(0.20, dtype=torch.float64, device=device)
    sig_curr = torch.tensor(0.21, dtype=torch.float64, device=device)
    dt = torch.tensor(1.0 / 365.0, dtype=torch.float64, device=device)
    
    g_prev = {
        "delta": torch.tensor(0.55, dtype=torch.float64, device=device),
        "gamma": torch.tensor(0.04, dtype=torch.float64, device=device),
        "vega": torch.tensor(15.0, dtype=torch.float64, device=device),
        "vanna": torch.tensor(-0.5, dtype=torch.float64, device=device),
        "volga": torch.tensor(2.0, dtype=torch.float64, device=device),
        "theta": torch.tensor(-10.0, dtype=torch.float64, device=device)
    }
    
    actual_pnl = torch.tensor(1.50, dtype=torch.float64, device=device)
    
    attr = attribute_pnl_daily(S_prev, S_curr, sig_prev, sig_curr, dt, g_prev, actual_pnl)
    
    explained = attr["explained_pnl"]
    residual = attr["residual"]
    
    # Explained + Residual must equal Actual P&L exactly
    assert np.isclose((explained + residual).item(), actual_pnl.item())
    
    # Check individual terms
    assert np.isclose(attr["delta_pnl"].item(), 0.55 * 2.0)
    assert np.isclose(attr["gamma_pnl"].item(), 0.5 * 0.04 * (2.0 ** 2))
    assert np.isclose(attr["vega_pnl"].item(), 15.0 * 0.01)


@patch("deepvol.hedging.backtest.download_spx_chain")
@patch("deepvol.hedging.backtest.clean_chain")
def test_run_empirical_backtest_savings(mock_clean, mock_download):
    """Verify that NTB hedging saves transaction costs compared to BS daily rebalancing."""
    # Create mock option chain dataframes for 5 consecutive business days
    dates_list = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8)
    ]
    
    # Define a clean option chain mock
    # Spot price is 5000.0, r=0.05, q=0.015
    # T = (expiry - date) / 365
    expiry = date(2024, 6, 21)
    expiry_pd = pd.to_datetime(expiry)
    
    # We mock download and clean chain to return a DataFrame with a single option matching our criteria
    mock_dfs = []
    spots = [5000.0, 5010.0, 5005.0, 4995.0, 4980.0]
    vols = [0.20, 0.205, 0.198, 0.202, 0.210]
    
    for idx, d in enumerate(dates_list):
        S_t = spots[idx]
        sig_t = vols[idx]
        T_t = (expiry_pd - pd.to_datetime(d)).days / 365.0
        
        # We need log_moneyness matching S_t
        log_mon = np.log(5000.0 / (S_t * np.exp(0.035 * T_t)))
        
        df = pd.DataFrame([{
            "strike": 5000.0,
            "expiry": expiry_pd,
            "type": "call",
            "bid": 250.0,
            "ask": 252.0,
            "mid_price": 251.0,
            "mid_iv": sig_t,
            "open_interest": 100,
            "openInterest": 100,
            "volume": 50,
            "T": T_t,
            "log_moneyness": log_mon,
            "is_synthetic": True
        }])
        mock_dfs.append(df)
        
    mock_download.side_effect = lambda d, cache: mock_dfs[dates_list.index(d)]
    mock_clean.side_effect = lambda df: df
    
    res = run_empirical_backtest(
        dates_list=dates_list,
        strike=5000.0,
        expiry=expiry,
        opt_type="call",
        r_val=0.05,
        q_val=0.015,
        c_S=0.0002,  # 2 bps transaction cost
        ntb_type="constant",
        ntb_beta=0.015,  # 1.5% delta band
        device="cpu"
    )
    
    # Check that keys exist
    assert "bs" in res
    assert "ntb" in res
    assert "attribution" in res
    
    # Check cost savings: NTB must have lower transaction cost than BS
    bs_cost = res["bs"]["total_cost"]
    ntb_cost = res["ntb"]["total_cost"]
    savings = res["ntb"]["cost_savings_pct"]
    
    print(f"BS Cost: {bs_cost:.5f}, NTB Cost: {ntb_cost:.5f}, Savings: {savings:.2f}%")
    
    assert bs_cost > 0.0
    assert ntb_cost > 0.0
    assert savings > 0.0
    assert ntb_cost < bs_cost


def test_stress_greeks_extreme_inputs():
    """Stress test: verify numerical stability under extreme near-zero boundaries (OOD)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extreme low boundaries
    S = torch.tensor(1e-12, dtype=torch.float64, device=device)
    K = torch.tensor(1e-12, dtype=torch.float64, device=device)
    T = torch.tensor(1e-12, dtype=torch.float64, device=device)
    r = torch.tensor(-0.05, dtype=torch.float64, device=device)
    q = torch.tensor(-0.05, dtype=torch.float64, device=device)
    sigma = torch.tensor(1e-12, dtype=torch.float64, device=device)
    is_call = torch.tensor(1.0, dtype=torch.float64, device=device)
    
    g = compute_bs_greeks(S, K, T, r, q, sigma, is_call)
    
    # Must not contain NaN or Inf
    for name, tensor in g.items():
        assert torch.isfinite(tensor), f"Non-finite value found in Greek '{name}': {tensor}"
        
    # Price and Delta should be sensible (near zero/one)
    assert g["price"].item() >= 0.0
    assert 0.0 <= g["delta"].item() <= 1.0


@patch("deepvol.hedging.backtest.download_spx_chain")
@patch("deepvol.hedging.backtest.clean_chain")
def test_backtest_zero_costs_identity(mock_clean, mock_download):
    """Verify that under zero transaction costs, NTB rebalancing is identical to BS daily rebalancing."""
    dates_list = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    expiry = date(2024, 6, 21)
    expiry_pd = pd.to_datetime(expiry)
    
    mock_dfs = []
    spots = [5000.0, 5010.0, 5005.0]
    
    for idx, d in enumerate(dates_list):
        S_t = spots[idx]
        T_t = (expiry_pd - pd.to_datetime(d)).days / 365.0
        log_mon = np.log(5000.0 / (S_t * np.exp(0.035 * T_t)))
        
        df = pd.DataFrame([{
            "strike": 5000.0,
            "expiry": expiry_pd,
            "type": "call",
            "bid": 250.0,
            "ask": 252.0,
            "mid_price": 251.0,
            "mid_iv": 0.20,
            "open_interest": 100,
            "openInterest": 100,
            "volume": 50,
            "T": T_t,
            "log_moneyness": log_mon,
            "is_synthetic": True
        }])
        mock_dfs.append(df)
        
    mock_download.side_effect = lambda d, cache: mock_dfs[dates_list.index(d)]
    mock_clean.side_effect = lambda df: df
    
    # Run backtest with c_S = 0.0
    res = run_empirical_backtest(
        dates_list=dates_list,
        strike=5000.0,
        expiry=expiry,
        opt_type="call",
        r_val=0.05,
        q_val=0.015,
        c_S=0.0,  # Zero transaction costs!
        ntb_type="constant",
        ntb_beta=0.0,
        device="cpu"
    )
    
    # With zero costs, transaction cost must be exactly 0.0
    assert res["bs"]["total_cost"] == 0.0
    assert res["ntb"]["total_cost"] == 0.0
    assert res["ntb"]["cost_savings_pct"] == 0.0
    
    # The deltas must be identical at all steps
    assert np.allclose(res["bs"]["deltas"], res["ntb"]["deltas"])

