"""
uploader.py — Implied volatility surface uploader and validation component.
Handles:
  1. Parsing pivot and flat CSV/Excel volatility sheets.
  2. Calendar spread and butterfly arbitrage checking.
  3. Interpolation/mapping to the FNO 8x11 grid.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import norm

# Standard FNO grid
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
STRIKES = np.linspace(-0.5, 0.5, 11)  # Log-moneyness grid

def bs_call_price(S0, K, T, sigma, r=0.0, q=0.0):
    """Compute Black-Scholes call option price."""
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def check_arbitrage(T_grid, K_grid, iv_matrix, S0=100.0, r=0.0):
    """
    Check discrete calendar and butterfly arbitrage on the given surface.
    
    Parameters:
    -----------
    T_grid : np.ndarray, sorted maturities
    K_grid : np.ndarray, sorted strikes (or log-moneyness converted to strikes)
    iv_matrix : np.ndarray, shape (len(T_grid), len(K_grid))
    """
    nT = len(T_grid)
    nK = len(K_grid)
    
    # 1. Positivity
    if np.any(iv_matrix <= 0):
        return {
            "is_free": False,
            "calendar_violations": [],
            "butterfly_violations": [{"reason": "Negative or zero implied volatility detected."}]
        }

    # 2. Calendar spread check: w(k, T_i) <= w(k, T_{i+1})
    # w = sigma^2 * T
    calendar_violations = []
    for j in range(nK):
        for i in range(nT - 1):
            w_curr = iv_matrix[i, j]**2 * T_grid[i]
            w_next = iv_matrix[i+1, j]**2 * T_grid[i+1]
            if w_next < w_curr - 1e-7:
                calendar_violations.append({
                    "maturity_1": float(T_grid[i]),
                    "maturity_2": float(T_grid[i+1]),
                    "strike": float(K_grid[j]),
                    "variance_1": float(w_curr),
                    "variance_2": float(w_next),
                    "diff": float(w_curr - w_next)
                })

    # 3. Butterfly arbitrage check: option pricing convexity
    butterfly_violations = []
    # If K_grid is log-moneyness, map to absolute strikes relative to S0
    abs_strikes = np.array(K_grid)
    if np.all(abs_strikes <= 2.0) and np.all(abs_strikes >= -2.0):
        # Interpret K_grid as log-moneyness
        abs_strikes = S0 * np.exp(abs_strikes)
        
    for i in range(nT):
        T = T_grid[i]
        # Compute calls
        calls = []
        for j in range(nK):
            C = bs_call_price(S0, abs_strikes[j], T, iv_matrix[i, j], r=r)
            calls.append(C)
            
        # Check calls are non-negative and non-increasing
        for j in range(nK):
            if calls[j] < -1e-7:
                butterfly_violations.append({
                    "maturity": float(T),
                    "strike": float(abs_strikes[j]),
                    "reason": f"Negative call option price: {calls[j]:.6f}"
                })
            if j > 0 and calls[j] > calls[j-1] + 1e-7:
                butterfly_violations.append({
                    "maturity": float(T),
                    "strike_1": float(abs_strikes[j-1]),
                    "strike_2": float(abs_strikes[j]),
                    "reason": f"Call price is increasing in strike: C(K1)={calls[j-1]:.6f}, C(K2)={calls[j]:.6f}"
                })
                
        # Check convexity: slope must be non-decreasing
        for j in range(1, nK - 1):
            dK1 = abs_strikes[j] - abs_strikes[j-1]
            dK2 = abs_strikes[j+1] - abs_strikes[j]
            slope1 = (calls[j] - calls[j-1]) / dK1
            slope2 = (calls[j+1] - calls[j]) / dK2
            if slope2 < slope1 - 1e-7:
                butterfly_violations.append({
                    "maturity": float(T),
                    "strike_1": float(abs_strikes[j-1]),
                    "strike_2": float(abs_strikes[j]),
                    "strike_3": float(abs_strikes[j+1]),
                    "slope_1": float(slope1),
                    "slope_2": float(slope2),
                    "diff": float(slope1 - slope2)
                })

    is_free = (len(calendar_violations) == 0) and (len(butterfly_violations) == 0)
    return {
        "is_free": is_free,
        "calendar_violations": calendar_violations,
        "butterfly_violations": butterfly_violations
    }

def parse_iv_sheet(uploaded_file) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse an uploaded CSV or Excel sheet containing implied volatilities.
    Returns:
    --------
    T_grid : np.ndarray, maturities
    K_grid : np.ndarray, strikes or log-moneyness
    iv_matrix : np.ndarray, shape (nT, nK)
    """
    name = uploaded_file.name.lower()
    if name.endswith('.xlsx') or name.endswith('.xls'):
        df = pd.read_excel(uploaded_file, index_col=None)
    else:
        df = pd.read_csv(uploaded_file, index_col=None)
        
    # Check if this is a flat table or pivot table
    # Flat tables usually have columns like Maturity/T, Strike/K, Vol/IV
    cols = [str(c).lower().strip() for c in df.columns]
    
    # Try to find flat table columns
    t_col = None
    k_col = None
    v_col = None
    
    t_names = ['maturity', 'maturities', 't', 'tenor', 'expiry', 'expiries']
    k_names = ['strike', 'strikes', 'k', 'moneyness', 'logmoneyness', 'log_moneyness']
    v_names = ['iv', 'impliedvol', 'vol', 'volatility', 'implied_volatility', 'implied_vol']
    
    for c in df.columns:
        c_clean = str(c).lower().strip()
        if c_clean in t_names:
            t_col = c
        elif c_clean in k_names:
            k_col = c
        elif c_clean in v_names:
            v_col = c
            
    if t_col is not None and k_col is not None and v_col is not None:
        # It's a flat table!
        df_clean = df[[t_col, k_col, v_col]].dropna()
        # Pivot the flat table
        pivot_df = df_clean.pivot_table(index=t_col, columns=k_col, values=v_col)
        T_grid = np.array(sorted(pivot_df.index))
        K_grid = np.array(sorted(pivot_df.columns))
        iv_matrix = pivot_df.values
        return T_grid, K_grid, iv_matrix
        
    # If not a flat table, treat it as a pivot/matrix table
    # If the first column is named, but others are numeric, it's a pivot table
    # We assume the index (first column or actual index) represents maturities, and columns represent strikes.
    df = df.set_index(df.columns[0])
    
    # Attempt to convert index and columns to floats
    try:
        T_grid = np.array([float(x) for x in df.index])
        K_grid = np.array([float(x) for x in df.columns])
        iv_matrix = df.values.astype(float)
        
        # Ensure maturities are sorted
        t_sort = np.argsort(T_grid)
        T_grid = T_grid[t_sort]
        iv_matrix = iv_matrix[t_sort, :]
        
        # Ensure strikes are sorted
        k_sort = np.argsort(K_grid)
        K_grid = K_grid[k_sort]
        iv_matrix = iv_matrix[:, k_sort]
        
        return T_grid, K_grid, iv_matrix
    except Exception as e:
        raise ValueError(f"Could not parse pivot sheet format: {e}. Please ensure rows are maturities, columns are strikes, and cells are numbers.")

def interpolate_to_model_grid(T_src, K_src, iv_src, S0=100.0) -> np.ndarray:
    """
    Interpolates a source implied volatility surface onto the FNO model grid:
    MATURITIES (8,) and STRIKES (11,) [log-moneyness].
    """
    # Create coordinate grid for source
    T_mesh, K_mesh = np.meshgrid(T_src, K_src, indexing="ij")
    
    # Flatten the source coordinates and values
    points = np.stack([T_mesh.ravel(), K_mesh.ravel()], axis=-1)
    values = iv_src.ravel()
    
    # If K_src looks like absolute strikes, map the target STRIKES (log-moneyness) to absolute strikes
    # or map K_src to log-moneyness: k = ln(K/S0)
    K_src_log = np.array(K_src)
    if np.any(K_src > 2.0) or np.any(K_src < -2.0):
        # K_src represents absolute strikes, convert to log-moneyness
        K_src_log = np.log(K_src / S0)
        
    # Rebuild points with log-moneyness
    T_mesh_log, K_mesh_log = np.meshgrid(T_src, K_src_log, indexing="ij")
    points_log = np.stack([T_mesh_log.ravel(), K_mesh_log.ravel()], axis=-1)
    
    # Target grid coordinates
    T_tgt_mesh, K_tgt_mesh = np.meshgrid(MATURITIES, STRIKES, indexing="ij")
    grid_tgt = np.stack([T_tgt_mesh.ravel(), K_tgt_mesh.ravel()], axis=-1)
    
    # Run linear interpolation
    iv_tgt = griddata(points_log, values, grid_tgt, method="linear")
    
    # Fill any NaNs (due to extrapolation) with nearest neighbor
    if np.any(np.isnan(iv_tgt)):
        iv_nearest = griddata(points_log, values, grid_tgt, method="nearest")
        iv_tgt = np.where(np.isnan(iv_tgt), iv_nearest, iv_tgt)
        
    # Reshape back to target grid shape (8, 11)
    return iv_tgt.reshape(len(MATURITIES), len(STRIKES))
