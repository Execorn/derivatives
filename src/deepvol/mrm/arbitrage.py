"""
arbitrage.py — Vectorized static calendar and butterfly arbitrage checks for implied volatility surfaces.
"""

import numpy as np
import scipy.stats as stats

def check_calendar_arbitrage(iv_surface: np.ndarray, T_grid: np.ndarray) -> dict:
    """
    Check calendar spread arbitrage: total variance w(T, k) = iv^2 * T must be non-decreasing in T.
    
    Parameters:
        iv_surface (np.ndarray): Shape (nT, nK) grid of implied volatilities.
        T_grid (np.ndarray): Shape (nT,) times to maturity.
        
    Returns:
        dict: A dictionary containing:
            "has_arbitrage": bool, True if any calendar spread violation is detected.
            "violations": np.ndarray, Shape (nT-1, nK) boolean mask of violations.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    iv_surface = np.asarray(iv_surface, dtype=float)
    nT, nK = iv_surface.shape
    
    assert len(T_grid) == nT, f"Maturity grid length {len(T_grid)} must match surface rows {nT}"
    
    # Calculate total variance: w(T, k) = iv^2 * T
    total_var = (iv_surface ** 2) * T_grid[:, None]
    
    # Check if total variance decreases over time: w(T_i+1) - w(T_i) < 0
    diff = np.diff(total_var, axis=0)
    
    # A small tolerance to handle floating-point precision issues
    violations = diff < -1e-8
    has_arbitrage = bool(np.any(violations))
    
    return {
        "has_arbitrage": has_arbitrage,
        "violations": violations
    }

def check_butterfly_arbitrage_durrleman(iv_surface: np.ndarray, K_grid: np.ndarray, T_grid: np.ndarray) -> dict:
    """
    Check butterfly arbitrage using Durrleman's condition on total variance w(k, T):
    g(k) = (1 - k*w'/(2*w))^2 - (w')^2/4 * (1/w + 1/4) + w''/2 >= 0
    
    Here K_grid represents forward log-moneyness: k = ln(K / F).
    This implementation is fully vectorized across both strikes and maturities.
    
    Parameters:
        iv_surface (np.ndarray): Shape (nT, nK) grid of implied volatilities.
        K_grid (np.ndarray): Shape (nK,) grid of log-moneyness.
        T_grid (np.ndarray): Shape (nT,) times to maturity.
        
    Returns:
        dict: A dictionary containing:
            "has_arbitrage": bool, True if Durrleman condition is violated.
            "violations": np.ndarray, Shape (nT, nK) boolean mask of violations (interior checked).
    """
    k = np.asarray(K_grid, dtype=float)
    T = np.asarray(T_grid, dtype=float)
    iv = np.asarray(iv_surface, dtype=float)
    nT, nK = iv.shape
    
    assert len(k) == nK, f"Strike grid length {len(k)} must match surface columns {nK}"
    assert len(T) == nT, f"Maturity grid length {len(T)} must match surface rows {nT}"
    
    # Calculate total variance: w = iv^2 * T
    w = (iv ** 2) * T[:, None]
    
    dk = np.diff(k)
    w_prime = np.zeros_like(w)
    w_prime_prime = np.zeros_like(w)
    
    # Vectorized central differences for interior points (1 to nK-2)
    h_l = dk[:-1]  # K_j - K_j-1
    h_r = dk[1:]   # K_j+1 - K_j
    
    w_l = w[:, :-2]
    w_c = w[:, 1:-1]
    w_r = w[:, 2:]
    
    w_prime[:, 1:-1] = (w_r - w_l) / (h_l + h_r)
    w_prime_prime[:, 1:-1] = 2.0 * ((w_r - w_c) / h_r - (w_c - w_l) / h_l) / (h_l + h_r)
    
    # Boundary points: forward/backward differences
    w_prime[:, 0] = (w[:, 1] - w[:, 0]) / dk[0]
    w_prime[:, -1] = (w[:, -1] - w[:, -2]) / dk[-1]
    
    w_prime_prime[:, 0] = w_prime_prime[:, 1]
    w_prime_prime[:, -1] = w_prime_prime[:, -2]
    
    # Compute Durrleman's g(k) function
    w_safe = np.clip(w, 1e-9, None)
    
    term1 = (1.0 - (k[None, :] * w_prime) / (2.0 * w_safe)) ** 2
    term2 = (w_prime ** 2) / 4.0 * (1.0 / w_safe + 0.25)
    term3 = w_prime_prime / 2.0
    
    g = term1 - term2 + term3
    
    # Violation if g < -1e-8
    violations = g < -1e-8
    
    # Ignore boundary coordinates for second derivative
    violations_interior = violations.copy()
    violations_interior[:, 0] = False
    violations_interior[:, -1] = False
    
    has_arbitrage = bool(np.any(violations_interior))
    
    return {
        "has_arbitrage": has_arbitrage,
        "violations": violations_interior,
        "g_values": g
    }

def construct_butterfly_matrix(K: np.ndarray) -> np.ndarray:
    """
    Constructs the tridiagonal convexity check matrix M_but of shape (m-2, m).
    M_but @ C.T computes the convexity term at interior points.
    """
    m = len(K)
    h = np.diff(K)
    M = np.zeros((m - 2, m))
    for i in range(m - 2):
        M[i, i] = 1.0 / h[i]
        M[i, i + 1] = - (1.0 / h[i] + 1.0 / h[i+1])
        M[i, i + 2] = 1.0 / h[i+1]
    return M

def check_butterfly_arbitrage_price(iv_surface: np.ndarray, K_grid: np.ndarray, T_grid: np.ndarray, S: float = 1.0) -> dict:
    """
    Check butterfly arbitrage by pricing European call options and verifying convexity of prices.
    Uses tridiagonal convexity matrix operator M_but for fast vectorization.
    
    Parameters:
        iv_surface (np.ndarray): Shape (nT, nK) grid of implied volatilities.
        K_grid (np.ndarray): Shape (nK,) grid of log-moneyness or absolute strikes.
        T_grid (np.ndarray): Shape (nT,) times to maturity.
        S (float): Current underlying asset price.
        
    Returns:
        dict: A dictionary containing:
            "has_arbitrage": bool, True if any price convexity violation is detected.
            "violations": np.ndarray, Shape (nT, nK-2) boolean mask of violations.
    """
    nT, nK = iv_surface.shape
    
    # If K_grid is log-moneyness, convert to absolute strikes
    if np.any(K_grid < 0) or np.max(np.abs(K_grid)) < 5.0:
        K_abs = S * np.exp(K_grid)
    else:
        K_abs = K_grid
        
    if K_abs.ndim == 1:
        K_abs = np.tile(K_abs, (nT, 1))
        
    # Helper vectorized BS call price calculation (r=0, q=0)
    T_m = T_grid[:, None]
    vol_std = iv_surface * np.sqrt(T_m)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K_abs) + 0.5 * vol_std**2) / np.clip(vol_std, 1e-9, None)
        d2 = d1 - vol_std
        
    C = S * stats.norm.cdf(d1) - K_abs * stats.norm.cdf(d2)
    # Handle zero maturity/vol edge cases
    mask_zero = vol_std <= 1e-8
    intrinsic = np.maximum(S - K_abs, 0.0)
    C = np.where(mask_zero, intrinsic, C)
    
    # Fast vectorized matrix check using M_but operator
    K_vector = K_abs[0]
    M_but = construct_butterfly_matrix(K_vector)
    
    # M_but @ C.T has shape (nK-2, nT)
    conv_check = M_but @ C.T
    
    h_minus = K_vector[1:-1] - K_vector[:-2]
    h_plus = K_vector[2:] - K_vector[1:-1]
    scale = 2.0 / (h_plus + h_minus)
    
    # Compute actual d2C matrix of shape (nT, nK-2)
    d2C = (scale[:, None] * conv_check).T
    
    # Violation counts if second derivative is negative relative to option price
    violations = d2C < -1e-6 * np.abs(C[:, 1:-1]).clip(1e-10)
    has_arbitrage = bool(np.any(violations))
    
    return {
        "has_arbitrage": has_arbitrage,
        "violations": violations,
        "d2C": d2C
    }

def check_arbitrage(iv_surface: np.ndarray, K_grid: np.ndarray, T_grid: np.ndarray, S: float = 1.0) -> dict:
    """
    Combined check for calendar and butterfly arbitrage.
    
    Returns:
        dict: Summary of checks and results.
    """
    cal_res = check_calendar_arbitrage(iv_surface, T_grid)
    but_dur = check_butterfly_arbitrage_durrleman(iv_surface, K_grid, T_grid)
    but_prc = check_butterfly_arbitrage_price(iv_surface, K_grid, T_grid, S)
    
    has_arbitrage = cal_res["has_arbitrage"] or but_dur["has_arbitrage"] or but_prc["has_arbitrage"]
    
    return {
        "has_arbitrage": has_arbitrage,
        "calendar": cal_res,
        "butterfly_durrleman": but_dur,
        "butterfly_price": but_prc
    }
