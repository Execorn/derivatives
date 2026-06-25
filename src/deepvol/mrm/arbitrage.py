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

import torch
import scipy.optimize as optimize

def invert_black_scholes_vectorized(
    C: torch.Tensor,
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    max_iters: int = 100,
    tol: float = 1e-8
) -> torch.Tensor:
    """
    Vectorized and differentiable implied volatility solver using hybrid bisection + Newton-Raphson.
    Clamps the minimum volatility parameter to 0.01 to prevent Durrleman singularities.
    Operates strictly in torch.float64.
    """
    import math
    # 1. Cast all inputs to torch.float64
    C_d = C.to(torch.float64)
    S_d = S.to(torch.float64) if isinstance(S, torch.Tensor) else torch.tensor(S, dtype=torch.float64, device=C.device)
    K_d = K.to(torch.float64)
    T_d = T.to(torch.float64)
    
    # Initialize bounds for bisection: low = 0.01, high = 5.0
    low = torch.full_like(C_d, 0.01)
    high = torch.full_like(C_d, 5.0)
    
    vol = 0.5 * (low + high)
    
    for i in range(max_iters):
        vol_std = vol * torch.sqrt(T_d)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (torch.log(S_d / K_d) + 0.5 * vol_std**2) / torch.clamp(vol_std, min=1e-12)
            d2 = d1 - vol_std
        
        nd1 = 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))
        nd2 = 0.5 * (1.0 + torch.erf(d2 / math.sqrt(2.0)))
        
        price = S_d * nd1 - K_d * nd2
        
        intrinsic = torch.clamp(S_d - K_d, min=0.0)
        price = torch.where(T_d <= 1e-12, intrinsic, price)
        
        diff = price - C_d
        
        if torch.max(torch.abs(diff)) < tol:
            break
            
        pdf_d1 = torch.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
        vega = S_d * pdf_d1 * torch.sqrt(T_d)
        vega = torch.clamp(vega, min=1e-12)
        
        low = torch.where(diff < 0.0, vol, low)
        high = torch.where(diff > 0.0, vol, high)
        
        vol_new = vol - diff / vega
        
        fallback_mask = (vol_new <= low) | (vol_new >= high) | (vega < 1e-8)
        vol = torch.where(fallback_mask, 0.5 * (low + high), vol_new)
        vol = torch.clamp(vol, min=0.01, max=5.0)
        
    return vol

def project_arbitrage_free(
    iv_surface: np.ndarray,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    S: float = 1.0
) -> np.ndarray:
    """
    Project an arbitrated implied volatility surface onto the space of arbitrage-free surfaces
    using a convex Quadratic Programming (QP) projection of Call option prices.
    All internal computations are performed in double precision (float64).
    
    Parameters:
        iv_surface (np.ndarray): Shape (nT, nK) grid of implied volatilities.
        K_grid (np.ndarray): Shape (nK,) grid of log-moneyness.
        T_grid (np.ndarray): Shape (nT,) times to maturity.
        S (float): Current underlying asset price.
        
    Returns:
        np.ndarray: Shape (nT, nK) projected implied volatility surface.
    """
    iv = np.asarray(iv_surface, dtype=np.float64)
    K = np.asarray(K_grid, dtype=np.float64)
    T = np.asarray(T_grid, dtype=np.float64)
    S_val = float(S)
    
    M, N = iv.shape
    
    if np.any(K < 0) or np.max(np.abs(K)) < 5.0:
        K_abs = S_val * np.exp(K)
    else:
        K_abs = K.copy()
        
    T_m = T[:, None]
    vol_std = iv * np.sqrt(T_m)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S_val / K_abs) + 0.5 * vol_std**2) / np.clip(vol_std, 1e-9, None)
        d2 = d1 - vol_std
        
    C = S_val * stats.norm.cdf(d1) - K_abs * stats.norm.cdf(d2)
    mask_zero = vol_std <= 1e-8
    intrinsic = np.maximum(S_val - K_abs, 0.0)
    C = np.where(mask_zero, intrinsic, C)
    
    c_flat = C.flatten()
    
    def objective(x):
        diff = x - c_flat
        return 0.5 * np.sum(diff ** 2)
        
    def jacobian(x):
        return x - c_flat
        
    bounds = []
    for i in range(M):
        for j in range(N):
            lb = max(S_val - K_abs[j], 0.0)
            bounds.append((lb, S_val))
            
    A_list = []
    # Butterfly constraints
    for i in range(M):
        h = np.diff(K_abs)
        for j in range(1, N - 1):
            row = np.zeros(M * N)
            offset = i * N
            row[offset + j - 1] = 1.0 / h[j - 1]
            row[offset + j] = - (1.0 / h[j - 1] + 1.0 / h[j])
            row[offset + j + 1] = 1.0 / h[j]
            A_list.append(row)
            
    # Calendar constraints
    for i in range(M - 1):
        for j in range(N):
            row = np.zeros(M * N)
            row[i * N + j] = -1.0
            row[(i + 1) * N + j] = 1.0
            A_list.append(row)
            
    if len(A_list) > 0:
        A = np.vstack(A_list)
        lb_con = np.zeros(A.shape[0])
        ub_con = np.full(A.shape[0], np.inf)
        linear_constraint = optimize.LinearConstraint(A, lb_con, ub_con)
        constraints = [linear_constraint]
    else:
        constraints = []
        
    res = optimize.minimize(
        fun=objective,
        x0=c_flat,
        jac=jacobian,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-9}
    )
    
    C_proj = res.x.reshape((M, N))
    
    device = iv_surface.device if isinstance(iv_surface, torch.Tensor) else "cpu"
    C_proj_t = torch.tensor(C_proj, dtype=torch.float64, device=device)
    K_abs_t = torch.tensor(np.tile(K_abs, (M, 1)), dtype=torch.float64, device=device)
    T_t = torch.tensor(np.tile(T[:, None], (1, N)), dtype=torch.float64, device=device)
    
    iv_proj_t = invert_black_scholes_vectorized(
        C=C_proj_t,
        S=torch.tensor(S_val, dtype=torch.float64, device=device),
        K=K_abs_t,
        T=T_t
    )
    
    iv_proj = iv_proj_t.cpu().numpy()
    iv_proj = np.clip(iv_proj, 0.01, None)
    
    if isinstance(iv_surface, torch.Tensor):
        return torch.tensor(iv_proj, device=iv_surface.device, dtype=iv_surface.dtype)
    return iv_proj

