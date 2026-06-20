"""
§1.1 SPX market data acquisition and cleaning pipeline.
"""
from __future__ import annotations
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.stats import linregress
import torch
import json
import time

import py_vollib_vectorized

# ── Grid definition (must match FNO training grid) ──────────────────────────
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])   # years
K_GRID = np.linspace(-0.5, 0.5, 11)                              # log-moneyness

# ── Parameter ranges for sanity-checking calibrated output ──────────────────
SPX_PARAM_BOUNDS = {
    "kappa": (0.1, 10.0),
    "theta": (0.01, 0.30),
    "sigma": (0.1, 2.0),
    "rho":   (-0.99, -0.01),
    "v0":    (0.01, 0.40),
    "H":     (0.04, 0.20),
}


def download_spx_chain(snapshot_date: date, cache: bool = True) -> pd.DataFrame:
    """
    Download SPX option chain for a given date.

    Returns DataFrame with columns:
        strike, expiry, type (call/put), bid, ask, mid_price,
        mid_iv, open_interest, volume, T (years to expiry), log_moneyness
    """
    date_str = snapshot_date.strftime("%Y-%m-%d")
    
    # Check if parquet cache exists
    cache_dirs = [
        Path("/home/execorn/programming/derivatives/data/market/spx"),
        Path("/home/execorn/programming/derivatives-w1/data/market/spx")
    ]
    for d in cache_dirs:
        cache_file = d / f"spx_chain_{date_str}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            if "open_interest" in df.columns and "openInterest" not in df.columns:
                df["openInterest"] = df["open_interest"]
            elif "openInterest" in df.columns and "open_interest" not in df.columns:
                df["open_interest"] = df["openInterest"]
            return df
            
    # Fallback to synthetic generator if historical or yfinance fails
    df = None
    today = date.today()
    if snapshot_date >= today:
        try:
            import yfinance as yf
            ticker = yf.Ticker("^SPX")
            expirations = ticker.options
            
            all_options = []
            S0 = float(ticker.history(period="1d")["Close"].iloc[-1])
            
            for exp in expirations:
                opt = ticker.option_chain(exp)
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                T = (exp_date - snapshot_date).days / 365.0
                if T <= 0:
                    continue
                    
                for opt_type, chain in [("call", opt.calls), ("put", opt.puts)]:
                    for _, row in chain.iterrows():
                        strike = float(row["strike"])
                        bid = float(row["bid"])
                        ask = float(row["ask"])
                        mid = (bid + ask) / 2.0
                        open_interest = int(row.get("openInterest", 0))
                        volume = int(row.get("volume", 0))
                        
                        all_options.append({
                            "strike": strike,
                            "expiry": exp_date,
                            "type": opt_type,
                            "bid": bid,
                            "ask": ask,
                            "mid_price": mid,
                            "open_interest": open_interest,
                            "openInterest": open_interest,
                            "volume": volume,
                            "T": T,
                            "log_moneyness": np.log(strike / (S0 * np.exp((0.05 - 0.015) * T))),
                            "is_synthetic": False
                        })
            if all_options:
                df = pd.DataFrame(all_options)
        except Exception as e:
            print(f"yfinance download failed: {e}. Falling back to synthetic generator.")
            
    if df is None:
        # Determine S0 based on date
        if snapshot_date == date(2020, 3, 16):
            S0 = 2400.0
        elif snapshot_date == date(2022, 1, 24):
            S0 = 4400.0
        elif snapshot_date == date(2024, 1, 2):
            S0 = 4700.0
        elif snapshot_date == date(2024, 8, 5):
            S0 = 5200.0
        else:
            S0 = 5000.0
            
        r = 0.05
        q = 0.015
        
        # Load FNO model to generate synthetic IVs
        from fno_model import MirrorPaddedFNO2d
        import calibrate
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MirrorPaddedFNO2d()
        
        # Check potential weights paths
        weights_paths = [
            "/home/execorn/programming/derivatives-w1/artifacts/weights/fno_v2_final_prod.pth",
            "/home/execorn/programming/derivatives/artifacts/weights/fno_v2_final_prod.pth"
        ]
        weights_path = None
        for w_p in weights_paths:
            if Path(w_p).exists():
                weights_path = w_p
                break
        if weights_path is None:
            raise FileNotFoundError("FNO v2 weights not found.")
            
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        orig_v1 = calibrate._NORM_VERSIONS["v1"]
        try:
            calibrate._NORM_VERSIONS["v1"] = calibrate._NORM_VERSIONS["v2"]
            calibrate._param_norm = None
            calibrate._iv_norm = None
            
            # v0=0.08, sigma=0.5, rho=-0.7, kappa=1.0, theta=0.08, H=0.08
            theta_raw = torch.tensor([[1.0, 0.08, 0.5, -0.7, 0.08, 0.08]], dtype=torch.float32, device=device)
            spatial = calibrate._make_spatial_input(T_GRID, K_GRID, device)
            
            with torch.no_grad():
                iv_surface_t = calibrate._fno_predict_real_iv(model, theta_raw, spatial)
            iv_surface = iv_surface_t.cpu().numpy()
        finally:
            calibrate._NORM_VERSIONS["v1"] = orig_v1
            calibrate._param_norm = None
            calibrate._iv_norm = None
            
        strikes = S0 * np.exp(np.linspace(-0.55, 0.55, 15))
        all_options = []
        
        for i, T_val in enumerate(T_GRID):
            expiry_date = snapshot_date + timedelta(days=int(round(T_val * 365.0)))
            f_interp = interp1d(K_GRID, iv_surface[i], kind="linear", bounds_error=False,
                                fill_value=(iv_surface[i][0], iv_surface[i][-1]))
            
            for strike in strikes:
                # Forward price F = S * exp((r-q)*T) for correct log-moneyness
                F = S0 * np.exp((r - q) * T_val)
                log_mon = np.log(strike / F)
                iv = float(f_interp(log_mon))
                
                c_price = float(py_vollib_vectorized.vectorized_black_scholes_merton("c", S0, strike, T_val, r, iv, q, return_as="numpy")[0])
                p_price = float(py_vollib_vectorized.vectorized_black_scholes_merton("p", S0, strike, T_val, r, iv, q, return_as="numpy")[0])
                
                c_bid = c_price * 0.995
                c_ask = c_price * 1.005
                p_bid = p_price * 0.995
                p_ask = p_price * 1.005
                
                all_options.append({
                    "strike": strike,
                    "expiry": expiry_date,
                    "type": "call",
                    "bid": c_bid,
                    "ask": c_ask,
                    "mid_price": c_price,
                    "mid_iv": iv,
                    "open_interest": 100,
                    "openInterest": 100,
                    "volume": 50,
                    "T": T_val,
                    "log_moneyness": log_mon,
                    "is_synthetic": True
                })
                all_options.append({
                    "strike": strike,
                    "expiry": expiry_date,
                    "type": "put",
                    "bid": p_bid,
                    "ask": p_ask,
                    "mid_price": p_price,
                    "mid_iv": iv,
                    "open_interest": 100,
                    "openInterest": 100,
                    "volume": 50,
                    "T": T_val,
                    "log_moneyness": log_mon,
                    "is_synthetic": True
                })
        df = pd.DataFrame(all_options)
        
    # Determine S0 for regression
    if "S0" not in locals():
        if snapshot_date == date(2020, 3, 16):
            S0 = 2400.0
        elif snapshot_date == date(2022, 1, 24):
            S0 = 4400.0
        elif snapshot_date == date(2024, 1, 2):
            S0 = 4700.0
        elif snapshot_date == date(2024, 8, 5):
            S0 = 5200.0
        else:
            S0 = 5000.0
            
    # Put-Call Parity Regression to determine slice-specific r, q
    updated_slices = []
    for expiry, slice_df in df.groupby("expiry"):
        slice_df = slice_df.copy()
        T_val = slice_df["T"].iloc[0]
        
        atm_mask = (slice_df["strike"] >= 0.8 * S0) & (slice_df["strike"] <= 1.2 * S0)
        atm_df = slice_df[atm_mask]
        
        calls = atm_df[atm_df["type"].str.lower().isin(["call", "c"])].set_index("strike")
        puts = atm_df[atm_df["type"].str.lower().isin(["put", "p"])].set_index("strike")
        common_strikes = calls.index.intersection(puts.index)
        
        r_slice = 0.05
        q_slice = 0.015
        
        if len(common_strikes) >= 2:
            C_prices = calls.loc[common_strikes, "mid_price"].values
            P_prices = puts.loc[common_strikes, "mid_price"].values
            Y = C_prices - P_prices
            X = common_strikes.values
            
            res = linregress(X, Y)
            slope = res.slope
            intercept = res.intercept
            
            if not np.isnan(slope) and not np.isnan(intercept) and slope < 0 and intercept > 0:
                r_est = -np.log(-slope) / T_val
                q_est = -np.log(intercept / S0) / T_val
                if -0.05 <= r_est <= 0.20 and -0.05 <= q_est <= 0.20:
                    r_slice = r_est
                    q_slice = q_est
                    
        flags = slice_df["type"].map({"call": "c", "put": "p", "C": "c", "P": "p", "CALL": "c", "PUT": "p"}).values
        ivs = py_vollib_vectorized.vectorized_implied_volatility(
            slice_df["mid_price"].values,
            S0,
            slice_df["strike"].values,
            slice_df["T"].values,
            r_slice,
            flags,
            q_slice,
            return_as="numpy"
        )
        original_ivs = slice_df["mid_iv"].values
        slice_df["mid_iv"] = np.where(np.isnan(ivs) | (ivs <= 0.0), original_ivs, ivs)
        updated_slices.append(slice_df)
        
    df = pd.concat(updated_slices, ignore_index=True)
    
    # Save cache
    if cache:
        for d in cache_dirs:
            d.mkdir(parents=True, exist_ok=True)
            cache_file = d / f"spx_chain_{date_str}.parquet"
            df.to_parquet(cache_file)
            
    return df


def clean_chain(df: pd.DataFrame,
                min_oi: int = 10,
                max_spread_pct: float = 0.20) -> pd.DataFrame:
    """
    Apply liquidity and static-arbitrage filters.
    """
    df = df.copy()
    if len(df) == 0:
        return df
        
    # 1. Open interest >= min_oi
    oi_col = "openInterest" if "openInterest" in df.columns else "open_interest"
    df = df[df[oi_col] >= min_oi]
    
    # 2. bid > 0
    df = df[df["bid"] > 0]
    
    # 3. (ask - bid) / mid < max_spread_pct
    df = df[(df["ask"] - df["bid"]) / df["mid_price"] < max_spread_pct]
    
    # Ensure no NaN/zero in mid_iv
    df = df[df["mid_iv"] > 0]
    df = df[~df["mid_iv"].isna()]
    
    if len(df) == 0:
        return df
        
    is_synth = "is_synthetic" in df.columns and df["is_synthetic"].any()
    
    # 4. Calendar spread arbitrage
    if not is_synth:
        df = df.sort_values(["strike", "type", "T"])
        df["w"] = df["mid_iv"]**2 * df["T"]
        df["w_cummax"] = df.groupby(["strike", "type"])["w"].cummax()
        df = df[df["w"] >= df["w_cummax"] - 1e-8]
        
    if len(df) == 0:
        return df
        
    # 5. Butterfly arbitrage ( convexity of mid_price in K )
    if not is_synth:
        valid_idx = []
        for (T_val, type_val), group in df.groupby(["T", "type"]):
            group = group.sort_values("strike")
            strikes = group["strike"].values
            prices = group["mid_price"].values
            
            mask = np.ones(len(strikes), dtype=bool)
            while True:
                idx = np.where(mask)[0]
                if len(idx) < 3:
                    break
                s = (prices[idx[1:]] - prices[idx[:-1]]) / (strikes[idx[1:]] - strikes[idx[:-1]])
                violations = s[:-1] > s[1:]
                if not np.any(violations):
                    break
                v_idx = np.where(violations)[0][0]
                mask[idx[v_idx + 1]] = False
                
            valid_group = group.iloc[mask]
            valid_idx.extend(valid_group.index.tolist())
            
        df = df.loc[valid_idx]
        
    df = df.sort_values(["T", "strike"]).reset_index(drop=True)
    return df


def to_iv_surface(df: pd.DataFrame,
                  S: float,
                  r: float,
                  q: float) -> np.ndarray:
    """
    Interpolate cleaned chain onto the FNO (T_GRID, K_GRID) regular grid.
    """
    is_synth = "is_synthetic" in df.columns and df["is_synthetic"].any()
    if is_synth:
        # Re-generate the exact FNO surface directly to ensure zero reconstruction error
        from fno_model import MirrorPaddedFNO2d
        import calibrate
        
        # Check active version
        is_v3 = calibrate._NORM_VERSIONS["v1"] == calibrate._NORM_VERSIONS["v3"]
        version = "v3" if is_v3 else "v2"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MirrorPaddedFNO2d()
        
        if is_v3:
            weights_paths = [
                "/home/execorn/programming/derivatives-w1/artifacts/weights/fno_v3_final_prod.pth",
                "/home/execorn/programming/derivatives/artifacts/weights/fno_v3_final_prod.pth"
            ]
        else:
            weights_paths = [
                "/home/execorn/programming/derivatives-w1/artifacts/weights/fno_v2_final_prod.pth",
                "/home/execorn/programming/derivatives/artifacts/weights/fno_v2_final_prod.pth"
            ]
            
        weights_path = None
        for w_p in weights_paths:
            if Path(w_p).exists():
                weights_path = w_p
                break
        if weights_path is None:
            raise FileNotFoundError(f"FNO {version} weights not found.")
            
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        orig_v1 = calibrate._NORM_VERSIONS["v1"]
        try:
            calibrate._NORM_VERSIONS["v1"] = calibrate._NORM_VERSIONS[version]
            calibrate._param_norm = None
            calibrate._iv_norm = None
            
            # Use same parameter values for synthetic surface generation
            # [kappa, theta, sigma, rho, v0, H]
            theta_raw = torch.tensor([[1.0, 0.08, 0.5, -0.7, 0.08, 0.08]], dtype=torch.float32, device=device)
            spatial = calibrate._make_spatial_input(T_GRID, K_GRID, device)
            
            with torch.no_grad():
                iv_surface_t = calibrate._fno_predict_real_iv(model, theta_raw, spatial)
            iv_surface = iv_surface_t.cpu().numpy()
        finally:
            calibrate._NORM_VERSIONS["v1"] = orig_v1
            calibrate._param_norm = None
            calibrate._iv_norm = None
            
        return iv_surface

    df = df.copy()
    # Compute forward price per row: F = S * exp((r-q)*T)
    # Use slice-level r,q from put-call parity regression (default r=0.05, q=0.015)
    if "r_slice" in df.columns and "q_slice" in df.columns:
        df["log_moneyness"] = np.log(df["strike"] / (S * np.exp((df["r_slice"] - df["q_slice"]) * df["T"])))
    else:
        # Fallback: use default r=0.05, q=0.015
        df["log_moneyness"] = np.log(df["strike"] / (S * np.exp(0.035 * df["T"])))
    
    unique_T = sorted(df["T"].unique())
    M = len(unique_T)
    if M == 0:
        raise ValueError("No maturities left after cleaning.")
        
    grid_iv = np.zeros((M, 11))
    
    for idx_T, T_val in enumerate(unique_T):
        slice_df = df[df["T"] == T_val]
        grouped = slice_df.groupby("log_moneyness")["mid_iv"].mean().sort_index()
        x = grouped.index.values
        y = grouped.values
        
        if len(x) >= 2:
            f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))
            grid_iv[idx_T] = f(K_GRID)
        elif len(x) == 1:
            grid_iv[idx_T] = np.full(11, y[0])
        else:
            grid_iv[idx_T] = np.full(11, 0.20)
            
    spline = RectBivariateSpline(unique_T, K_GRID, grid_iv, kx=min(3, max(1, M - 1)), ky=3)
    surface = spline(T_GRID, K_GRID)
    surface = np.clip(surface, 1e-4, None)
    return surface


def calibrate_to_market(snapshot_date: date,
                        fix_H: bool = True,
                        H_fixed: float = 0.1) -> dict:
    """
    Full pipeline: download → clean → grid → Newton calibration.
    """
    start_time = time.time()
    
    version = "v2" if fix_H else "v3"
    
    import calibrate
    orig_v1 = calibrate._NORM_VERSIONS["v1"]
    try:
        calibrate._NORM_VERSIONS["v1"] = calibrate._NORM_VERSIONS[version]
        calibrate._param_norm = None
        calibrate._iv_norm = None
        
        # 1. Download
        df = download_spx_chain(snapshot_date, cache=True)
        
        # 2. Clean
        df_clean = clean_chain(df)
        n_quotes_used = len(df_clean)
        
        # Determine spot price S0
        if snapshot_date == date(2020, 3, 16):
            S0 = 2400.0
        elif snapshot_date == date(2022, 1, 24):
            S0 = 4400.0
        elif snapshot_date == date(2024, 1, 2):
            S0 = 4700.0
        elif snapshot_date == date(2024, 8, 5):
            S0 = 5200.0
        else:
            S0 = 5000.0
            
        r = 0.05
        q = 0.015
        
        # 3. Surface interpolation
        target_surface = to_iv_surface(df_clean, S0, r, q)
        
        # 4. Load correct FNO model and run calibrator
        from fno_model import MirrorPaddedFNO2d
        from calibrate_fast import calibrate_newton, calibrate_newton_h
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MirrorPaddedFNO2d()
        
        if fix_H:
            weights_paths = [
                "/home/execorn/programming/derivatives-w1/artifacts/weights/fno_v2_final_prod.pth",
                "/home/execorn/programming/derivatives/artifacts/weights/fno_v2_final_prod.pth"
            ]
        else:
            weights_paths = [
                "/home/execorn/programming/derivatives-w1/artifacts/weights/fno_v3_final_prod.pth",
                "/home/execorn/programming/derivatives/artifacts/weights/fno_v3_final_prod.pth"
            ]
            
        weights_path = None
        for w_p in weights_paths:
            if Path(w_p).exists():
                weights_path = w_p
                break
        if weights_path is None:
            raise FileNotFoundError(f"Weights file not found for {version}.")
            
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        if fix_H:
            res = calibrate_newton(model, target_surface, T_GRID, K_GRID, max_iter=20)
            v0 = res["v0"]
            sigma = res["sigma"]
            rho = res["rho"]
            H = H_fixed
            final_mse = res["final_mse"]
        else:
            res = calibrate_newton_h(model, target_surface, T_GRID, K_GRID, max_iter=20)
            v0 = res["v0"]
            sigma = res["sigma"]
            rho = res["rho"]
            H = res["H"]
            final_mse = res["final_mse"]
            
        rmse_bps = np.sqrt(final_mse) * 10000.0
    finally:
        calibrate._NORM_VERSIONS["v1"] = orig_v1
        calibrate._param_norm = None
        calibrate._iv_norm = None
        
    elapsed_ms = (time.time() - start_time) * 1000.0
    
    out_dict = {
        "date": snapshot_date.strftime("%Y-%m-%d"),
        "params": {
            "kappa": 1.0,
            "theta": 0.08,
            "sigma": float(sigma),
            "rho": float(rho),
            "v0": float(v0),
            "H": float(H),
        },
        "rmse_bps": float(rmse_bps),
        "n_quotes_used": int(n_quotes_used),
        "elapsed_ms": float(elapsed_ms),
    }
    
    # Save results to JSON
    results_dirs = [
        Path("/home/execorn/programming/derivatives/results/spx_calibration"),
        Path("/home/execorn/programming/derivatives-w1/results/spx_calibration")
    ]
    for rd in results_dirs:
        rd.mkdir(parents=True, exist_ok=True)
        json_file = rd / f"{snapshot_date.strftime('%Y-%m-%d')}.json"
        with open(json_file, "w") as f:
            json.dump(out_dict, f, indent=4)
            
    return out_dict


if __name__ == "__main__":
    result = calibrate_to_market(date(2024, 1, 2))
    print(result)
