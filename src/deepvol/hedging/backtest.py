"""
backtest.py — Dynamic options portfolio backtesting, P&L attribution, and No-Transaction-Band (NTB) extraction on empirical options data.
All pricing and greeks computations are carried out strictly in torch.float64 (double precision) on GPU when available.
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import date
from typing import List, Dict, Any, Tuple, Optional
from deepvol.market.spx_data import download_spx_chain, clean_chain

# Default grids matching FNO training data
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float64)
K_GRID = np.linspace(-0.5, 0.5, 11, dtype=np.float64)

# Differentiable normal distribution in double precision for CUDA/CPU
class NormalCDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.special.erf(x / math.sqrt(2.0)) * 0.5 + 0.5

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        pdf = torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)
        return grad_output * pdf

def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return NormalCDF.apply(x)

# ── Double Precision Black-Scholes Greeks Solver ──────────────────────────────
@torch.compile(mode="reduce-overhead")
def _bs_greeks_kernel(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    sigma: torch.Tensor,
    is_call: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compiled kernel for Black-Scholes prices and Greeks in double precision (torch.float64).
    Clamps minimum volatility parameter to 0.01 to prevent mathematical singularities.
    Returns: (price, delta, gamma, vega, vanna, volga, theta)
    """
    # Enforce float64 representation
    S = S.to(torch.float64)
    K = K.to(torch.float64)
    T = T.to(torch.float64)
    r = r.to(torch.float64)
    q = q.to(torch.float64)
    sigma = torch.clamp(sigma.to(torch.float64), min=0.01)

    # Avoid division-by-zero or negative time-to-maturity
    T_safe = torch.clamp(T, min=1e-6)
    S_safe = torch.clamp(S, min=1e-8)
    K_safe = torch.clamp(K, min=1e-8)

    sqrt_T = torch.sqrt(T_safe)
    denom = sigma * sqrt_T

    d1 = (torch.log(S_safe / K_safe) + (r - q + 0.5 * sigma**2) * T_safe) / denom
    d2 = d1 - denom

    # CDF calculations
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64, device=S.device),
        torch.tensor(1.0, dtype=torch.float64, device=S.device)
    )
    N_d1 = normal.cdf(d1)
    N_d2 = normal.cdf(d2)
    N_minus_d1 = normal.cdf(-d1)
    N_minus_d2 = normal.cdf(-d2)

    phi_d1 = torch.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)

    # Discount factors
    exp_rt = torch.exp(-r * T_safe)
    exp_qt = torch.exp(-q * T_safe)

    # Price and Delta based on call/put type
    call_price = S * exp_qt * N_d1 - K * exp_rt * N_d2
    put_price = K * exp_rt * N_minus_d2 - S * exp_qt * N_minus_d1
    price = torch.where(is_call == 1.0, call_price, put_price)

    call_delta = exp_qt * N_d1
    put_delta = -exp_qt * N_minus_d1
    delta = torch.where(is_call == 1.0, call_delta, put_delta)

    # Gamma, Vega, Vanna, Volga, Theta
    gamma = (exp_qt * phi_d1) / (S_safe * denom)
    vega = S * exp_qt * sqrt_T * phi_d1
    vanna = -exp_qt * phi_d1 * (d2 / sigma)
    volga = vega * d1 * d2 / sigma

    call_theta = - (S * exp_qt * sigma * phi_d1) / (2.0 * sqrt_T) + q * S * exp_qt * N_d1 - r * K * exp_rt * N_d2
    put_theta = - (S * exp_qt * sigma * phi_d1) / (2.0 * sqrt_T) - q * S * exp_qt * N_minus_d1 + r * K * exp_rt * N_minus_d2
    theta = torch.where(is_call == 1.0, call_theta, put_theta)

    # Clamping extreme values/NaNs to prevent propagation
    price = torch.nan_to_num(price, nan=0.0)
    delta = torch.nan_to_num(delta, nan=0.0)
    gamma = torch.nan_to_num(gamma, nan=0.0, posinf=1e6, neginf=-1e6)
    vega = torch.nan_to_num(vega, nan=0.0)
    vanna = torch.nan_to_num(vanna, nan=0.0)
    volga = torch.nan_to_num(volga, nan=0.0)
    theta = torch.nan_to_num(theta, nan=0.0)

    # Clone returned tensors to prevent CUDA Graph static buffer overwrite
    return (
        price.clone(),
        delta.clone(),
        gamma.clone(),
        vega.clone(),
        vanna.clone(),
        volga.clone(),
        theta.clone()
    )


def compute_bs_greeks(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    sigma: torch.Tensor,
    is_call: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    User-facing interface for Black-Scholes Greeks.
    Ensures internal float64 precision and returns results as dict of tensors.
    """
    res = _bs_greeks_kernel(S, K, T, r, q, sigma, is_call)
    return {
        "price": res[0].clone(),
        "delta": res[1].clone(),
        "gamma": res[2].clone(),
        "vega": res[3].clone(),
        "vanna": res[4].clone(),
        "volga": res[5].clone(),
        "theta": res[6].clone()
    }


# ── Whalley-Wilmott No-Transaction-Band (NTB) boundaries ──────────────────────
def get_whalley_wilmott_beta(
    S: torch.Tensor,
    gamma: torch.Tensor,
    c_S: float,
    risk_aversion: float = 1.0
) -> torch.Tensor:
    """
    Extract NTB half-width using the Whalley-Wilmott asymptotic formula:
    beta = ( (3/2) * (c_S * S * gamma^2) / risk_aversion )^(1/3)
    """
    numerator = 1.5 * c_S * S * (gamma ** 2)
    beta = (numerator / risk_aversion) ** (1.0 / 3.0)
    return torch.clamp(beta, min=1e-5)


# ── P&L Attribution Engine ────────────────────────────────────────────────────
def attribute_pnl_daily(
    S_prev: torch.Tensor,
    S_curr: torch.Tensor,
    sigma_prev: torch.Tensor,
    sigma_curr: torch.Tensor,
    dt: torch.Tensor,
    greeks_prev: Dict[str, torch.Tensor],
    actual_pnl: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Perform daily P&L attribution decomposing option value change into delta, gamma,
    vega, vanna, volga, theta, and residual.
    Operates strictly in double precision.
    """
    dS = (S_curr - S_prev).to(torch.float64)
    dsigma = (sigma_curr - sigma_prev).to(torch.float64)
    dt = dt.to(torch.float64)

    delta = greeks_prev["delta"].to(torch.float64)
    gamma = greeks_prev["gamma"].to(torch.float64)
    vega = greeks_prev["vega"].to(torch.float64)
    vanna = greeks_prev["vanna"].to(torch.float64)
    volga = greeks_prev["volga"].to(torch.float64)
    theta = greeks_prev["theta"].to(torch.float64)

    # Taylor expansion components
    delta_pnl = delta * dS
    gamma_pnl = 0.5 * gamma * (dS ** 2)
    vega_pnl = vega * dsigma
    vanna_pnl = vanna * dS * dsigma
    volga_pnl = 0.5 * volga * (dsigma ** 2)
    theta_pnl = theta * dt

    explained = delta_pnl + gamma_pnl + vega_pnl + vanna_pnl + volga_pnl + theta_pnl
    residual = actual_pnl.to(torch.float64) - explained

    return {
        "delta_pnl": delta_pnl,
        "gamma_pnl": gamma_pnl,
        "vega_pnl": vega_pnl,
        "vanna_pnl": vanna_pnl,
        "volga_pnl": volga_pnl,
        "theta_pnl": theta_pnl,
        "explained_pnl": explained,
        "residual": residual
    }


# ── Option Path Extraction ───────────────────────────────────────────────────
def extract_empirical_option_path(
    dates_list: List[date],
    strike: float,
    expiry: date,
    opt_type: str = "call",
    r_val: float = 0.05,
    q_val: float = 0.015
) -> Dict[str, np.ndarray]:
    """
    Loads daily parquet files and extracts the underlying spot price, option price, and IV.
    If the exact strike/expiry does not exist on some date, uses bilinear interpolation
    on the daily IV surface.
    """
    N = len(dates_list)
    spot_arr = np.zeros(N)
    price_arr = np.zeros(N)
    iv_arr = np.zeros(N)
    T_arr = np.zeros(N)

    # Convert expiry to pandas timestamp for alignment
    expiry_pd = pd.to_datetime(expiry)

    for idx, d in enumerate(dates_list):
        df_raw = download_spx_chain(d, cache=True)
        df_clean = clean_chain(df_raw)

        # Retrieve/Estimate S0
        row = df_clean.iloc[0]
        S0 = row['strike'] * np.exp(-row['log_moneyness'] - (r_val - q_val) * row['T'])
        S0 = np.round(S0, 2)
        spot_arr[idx] = S0

        # Calculate time to maturity on this date
        T_rem = (expiry_pd - pd.to_datetime(d)).days / 365.0
        T_rem = max(T_rem, 0.0)
        T_arr[idx] = T_rem

        # Search for exact match
        df_opt = df_clean[
            (np.isclose(df_clean["strike"], strike, atol=0.1)) &
            (pd.to_datetime(df_clean["expiry"]) == expiry_pd) &
            (df_clean["type"].str.lower() == opt_type.lower())
        ]

        if len(df_opt) > 0:
            price_arr[idx] = df_opt["mid_price"].mean()
            iv_arr[idx] = df_opt["mid_iv"].mean()
        else:
            # Fallback to interpolation on the daily IV surface
            from deepvol.market.spx_data import to_iv_surface, T_GRID, K_GRID
            iv_surface = to_iv_surface(df_clean, S0, r_val, q_val)

            # Map to log-moneyness
            F = S0 * np.exp((r_val - q_val) * T_rem) if T_rem > 0 else S0
            k_mon = np.log(strike / F)

            # Interpolate
            from deepvol.benchmarks.hedging_backtest import interpolate_bilinear_np
            sig_interp = interpolate_bilinear_np(T_GRID, K_GRID, iv_surface, T_rem, k_mon)
            iv_arr[idx] = max(sig_interp, 0.01)

            # Price option using interpolated IV
            is_c = 1.0 if opt_type.lower() == "call" else 0.0
            S_t = torch.tensor([S0], dtype=torch.float64)
            K_t = torch.tensor([strike], dtype=torch.float64)
            T_t = torch.tensor([T_rem], dtype=torch.float64)
            r_t = torch.tensor([r_val], dtype=torch.float64)
            q_t = torch.tensor([q_val], dtype=torch.float64)
            sig_t = torch.tensor([sig_interp], dtype=torch.float64)
            is_c_t = torch.tensor([is_c], dtype=torch.float64)

            with torch.no_grad():
                res_price = _bs_greeks_kernel(S_t, K_t, T_t, r_t, q_t, sig_t, is_c_t)[0]
            price_arr[idx] = float(res_price.item())

    return {
        "spot": spot_arr,
        "price": price_arr,
        "iv": iv_arr,
        "T": T_arr
    }


# ── Full Portfolio Backtesting Simulator ──────────────────────────────────────
def run_empirical_backtest(
    dates_list: List[date],
    strike: float,
    expiry: date,
    opt_type: str = "call",
    r_val: float = 0.05,
    q_val: float = 0.015,
    c_S: float = 0.0001,  # 1 bp proportional stock cost
    ntb_type: str = "whalley_wilmott",  # "whalley_wilmott" or "constant"
    ntb_beta: float = 0.02,  # Used if ntb_type is "constant"
    risk_aversion: float = 1.0,  # Used for Whalley-Wilmott
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Run empirical option hedging backtest comparing:
      1. Frictionless Black-Scholes daily rebalancing
      2. No-Transaction-Band (NTB) hedging
    """
    device_obj = torch.device(device)

    # 1. Extract path data
    path_data = extract_empirical_option_path(dates_list, strike, expiry, opt_type, r_val, q_val)

    # Move arrays to GPU/CPU tensors in float64
    S = torch.tensor(path_data["spot"], dtype=torch.float64, device=device_obj)
    V = torch.tensor(path_data["price"], dtype=torch.float64, device=device_obj)
    iv = torch.tensor(path_data["iv"], dtype=torch.float64, device=device_obj)
    T = torch.tensor(path_data["T"], dtype=torch.float64, device=device_obj)

    N_steps = len(dates_list) - 1
    is_c = torch.tensor(1.0 if opt_type.lower() == "call" else 0.0, dtype=torch.float64, device=device_obj)
    r = torch.tensor(r_val, dtype=torch.float64, device=device_obj)
    q = torch.tensor(q_val, dtype=torch.float64, device=device_obj)
    K = torch.tensor(strike, dtype=torch.float64, device=device_obj)

    # 2. Compute greeks for all days
    greeks_list = []
    for idx in range(len(dates_list)):
        g = compute_bs_greeks(S[idx], K, T[idx], r, q, iv[idx], is_c)
        greeks_list.append(g)

    # Pre-build SoA for Greeks
    delta_target = torch.stack([g["delta"] for g in greeks_list])
    gamma_all = torch.stack([g["gamma"] for g in greeks_list])
    vega_all = torch.stack([g["vega"] for g in greeks_list])
    vanna_all = torch.stack([g["vanna"] for g in greeks_list])
    volga_all = torch.stack([g["volga"] for g in greeks_list])
    theta_all = torch.stack([g["theta"] for g in greeks_list])

    # 3. Simulate BS Daily Rebalancing
    bs_deltas = torch.zeros(len(dates_list), dtype=torch.float64, device=device_obj)
    bs_cash = torch.zeros(len(dates_list), dtype=torch.float64, device=device_obj)
    bs_costs = torch.zeros(len(dates_list), dtype=torch.float64, device=device_obj)

    # t0
    bs_deltas[0] = delta_target[0]
    bs_cash[0] = V[0] - bs_deltas[0] * S[0] - c_S * S[0] * torch.abs(bs_deltas[0])
    bs_costs[0] = c_S * S[0] * torch.abs(bs_deltas[0])

    # t_i
    for idx in range(1, len(dates_list)):
        dt_step = (dates_list[idx] - dates_list[idx-1]).days / 365.0
        # Reinvest cash at risk-free rate
        cash_growth = bs_cash[idx-1] * math.exp(r_val * dt_step)

        if idx < len(dates_list) - 1:
            bs_deltas[idx] = delta_target[idx]
            trade_cost = c_S * S[idx] * torch.abs(bs_deltas[idx] - bs_deltas[idx-1])
            bs_cash[idx] = cash_growth - (bs_deltas[idx] - bs_deltas[idx-1]) * S[idx] - trade_cost
            bs_costs[idx] = trade_cost
        else:
            # Unwind portfolio at maturity
            unwind_cost = c_S * S[idx] * torch.abs(bs_deltas[idx-1])
            bs_cash[idx] = cash_growth + bs_deltas[idx-1] * S[idx] - unwind_cost
            bs_costs[idx] = unwind_cost
            bs_deltas[idx] = 0.0

    # 4. Simulate NTB Rebalancing
    ntb_deltas = torch.zeros(len(dates_list), dtype=torch.float64, device=device_obj)
    ntb_cash = torch.zeros(len(dates_list), dtype=torch.float64, device=device_obj)
    ntb_costs = torch.zeros(len(dates_list), dtype=torch.float64, device=device_obj)
    ntb_beta_val = torch.zeros(len(dates_list), dtype=torch.float64, device=device_obj)

    # t0: Rebalance to center (target)
    ntb_deltas[0] = delta_target[0]
    ntb_cash[0] = V[0] - ntb_deltas[0] * S[0] - c_S * S[0] * torch.abs(ntb_deltas[0])
    ntb_costs[0] = c_S * S[0] * torch.abs(ntb_deltas[0])

    for idx in range(1, len(dates_list)):
        dt_step = (dates_list[idx] - dates_list[idx-1]).days / 365.0
        cash_growth = ntb_cash[idx-1] * math.exp(r_val * dt_step)

        if idx < len(dates_list) - 1:
            # Compute NTB half-width
            if ntb_type.lower() == "whalley_wilmott":
                beta = get_whalley_wilmott_beta(S[idx], gamma_all[idx], c_S, risk_aversion)
            else:
                beta = torch.tensor(ntb_beta, dtype=torch.float64, device=device_obj)
            ntb_beta_val[idx] = beta

            prev_d = ntb_deltas[idx-1]
            target_d = delta_target[idx]

            # Rebalance to boundary of the band if outside
            if prev_d < target_d - beta:
                new_d = target_d - beta
            elif prev_d > target_d + beta:
                new_d = target_d + beta
            else:
                new_d = prev_d

            ntb_deltas[idx] = new_d
            trade_cost = c_S * S[idx] * torch.abs(new_d - prev_d)
            ntb_cash[idx] = cash_growth - (new_d - prev_d) * S[idx] - trade_cost
            ntb_costs[idx] = trade_cost
        else:
            # Unwind portfolio
            unwind_cost = c_S * S[idx] * torch.abs(ntb_deltas[idx-1])
            ntb_cash[idx] = cash_growth + ntb_deltas[idx-1] * S[idx] - unwind_cost
            ntb_costs[idx] = unwind_cost
            ntb_deltas[idx] = 0.0

    # 5. P&L Attribution Calculation
    daily_attrs = []
    for idx in range(1, len(dates_list)):
        dt_step = torch.tensor((dates_list[idx] - dates_list[idx-1]).days / 365.0, dtype=torch.float64, device=device_obj)
        actual_opt_change = V[idx] - V[idx-1]

        g_prev = {
            "delta": delta_target[idx-1],
            "gamma": gamma_all[idx-1],
            "vega": vega_all[idx-1],
            "vanna": vanna_all[idx-1],
            "volga": volga_all[idx-1],
            "theta": theta_all[idx-1]
        }
        attr = attribute_pnl_daily(
            S[idx-1], S[idx], iv[idx-1], iv[idx], dt_step, g_prev, actual_opt_change
        )
        daily_attrs.append(attr)

    # 6. Performance Evaluation Metrics
    payoff = torch.clamp(S[-1] - K if opt_type.lower() == "call" else K - S[-1], min=0.0)

    # Tracking error (wealth - payoff)
    bs_tracking_error = bs_cash[-1] - payoff
    ntb_tracking_error = ntb_cash[-1] - payoff

    total_cost_bs = torch.sum(bs_costs)
    total_cost_ntb = torch.sum(ntb_costs)

    # Cost savings compare NTB to BS rebalancing
    cost_savings = (total_cost_bs - total_cost_ntb) / torch.clamp(total_cost_bs, min=1e-8) * 100.0

    # Rebalancing cost savings (excluding index 0 and index -1)
    if N_steps > 1:
        bs_rebal = torch.sum(bs_costs[1:-1])
        ntb_rebal = torch.sum(ntb_costs[1:-1])
        rebal_savings = (bs_rebal - ntb_rebal) / torch.clamp(bs_rebal, min=1e-8) * 100.0
    else:
        rebal_savings = torch.tensor(0.0, dtype=torch.float64, device=device_obj)

    # Output dictionary containing results
    return {
        "spot": S.cpu().numpy(),
        "price": V.cpu().numpy(),
        "iv": iv.cpu().numpy(),
        "T": T.cpu().numpy(),
        "greeks": {
            "delta": delta_target.cpu().numpy(),
            "gamma": gamma_all.cpu().numpy(),
            "vega": vega_all.cpu().numpy(),
            "vanna": vanna_all.cpu().numpy(),
            "volga": volga_all.cpu().numpy(),
            "theta": theta_all.cpu().numpy(),
        },
        "bs": {
            "deltas": bs_deltas.cpu().numpy(),
            "cash": bs_cash.cpu().numpy(),
            "costs": bs_costs.cpu().numpy(),
            "total_cost": float(total_cost_bs.item()),
            "tracking_error": float(bs_tracking_error.item())
        },
        "ntb": {
            "deltas": ntb_deltas.cpu().numpy(),
            "cash": ntb_cash.cpu().numpy(),
            "costs": ntb_costs.cpu().numpy(),
            "betas": ntb_beta_val.cpu().numpy(),
            "total_cost": float(total_cost_ntb.item()),
            "tracking_error": float(ntb_tracking_error.item()),
            "cost_savings_pct": float(cost_savings.item()),
            "rebalancing_cost_savings_pct": float(rebal_savings.item())
        },
        "attribution": {
            "delta_pnl": np.array([float(a["delta_pnl"].item()) for a in daily_attrs]),
            "gamma_pnl": np.array([float(a["gamma_pnl"].item()) for a in daily_attrs]),
            "vega_pnl": np.array([float(a["vega_pnl"].item()) for a in daily_attrs]),
            "vanna_pnl": np.array([float(a["vanna_pnl"].item()) for a in daily_attrs]),
            "volga_pnl": np.array([float(a["volga_pnl"].item()) for a in daily_attrs]),
            "theta_pnl": np.array([float(a["theta_pnl"].item()) for a in daily_attrs]),
            "explained_pnl": np.array([float(a["explained_pnl"].item()) for a in daily_attrs]),
            "residual": np.array([float(a["residual"].item()) for a in daily_attrs]),
        }
    }
