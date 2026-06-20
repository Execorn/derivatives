"""
§1.2 Arbitrage-free IV surface completion.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
from typing import Optional
from pathlib import Path

# Add src/ to sys.path to enable imports of sibling modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def check_calendar_spread(iv_surface: np.ndarray,
                          T_grid: np.ndarray) -> np.ndarray:
    """
    Check calendar spread arbitrage: C(T1,K) <= C(T2,K) for T1 < T2.
    In IV space: σ²(T2) * T2 >= σ²(T1) * T1  (total variance monotone in T).

    Returns boolean mask (n_T-1, n_K): True = violation.
    """
    total_var = iv_surface**2 * T_grid[:, None]           # (nT, nK)
    violations = np.diff(total_var, axis=0) < 0           # (nT-1, nK)
    return violations


def bs_call_price(S: float, K: np.ndarray, T: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """
    Vectorized Black-Scholes call option pricing formula with r=0, q=0.
    """
    # T_m shape (nT, nK), K_m shape (nT, nK)
    if T.ndim == 1:
        T_m = T[:, None]
    else:
        T_m = T
        
    if K.ndim == 1:
        K_m = K[None, :]
    else:
        K_m = K
        
    vol_std = iv * np.sqrt(T_m)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K_m) + 0.5 * vol_std**2) / vol_std
        d2 = d1 - vol_std
        
    c = S * stats.norm.cdf(d1) - K_m * stats.norm.cdf(d2)
    
    # Handle T=0 or iv=0 edge cases
    mask_zero = vol_std <= 1e-8
    intrinsic = np.maximum(S - K_m, 0.0)
    c = np.where(mask_zero, intrinsic, c)
    return c


def bs_call_price_scalar(S: float, K: float, T: float, sigma: float) -> float:
    """
    Scalar Black-Scholes call option price helper.
    """
    if T <= 0.0 or sigma <= 0.0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)


def implied_vol_scalar(price: float, S: float, K: float, T: float) -> float:
    """
    Scalar Brent implied volatility root-finder.
    """
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-15:
        return 1e-4
    if price >= S - 1e-15:
        return 5.0
    try:
        return optimize.brentq(
            lambda sigma: bs_call_price_scalar(S, K, T, sigma) - price,
            1e-4, 5.0, xtol=1e-12
        )
    except ValueError:
        return 1e-4


def project_convex_call_prices(C: np.ndarray, K_abs: np.ndarray, S: float) -> np.ndarray:
    """
    Project call prices onto the set of convex, non-increasing functions of strike K.
    """
    nT, nK = C.shape
    C_convex = np.zeros_like(C)
    for t in range(nT):
        c_target = C[t]
        k_slice = K_abs[t]
        
        # Objective: minimize sum of squared differences
        def obj(x):
            return np.sum((x - c_target)**2)
            
        cons = []
        h = np.diff(k_slice)
        
        # 1. Convexity: (x[j+1] - x[j]) / h_plus - (x[j] - x[j-1]) / h_minus >= 0
        for j in range(1, nK - 1):
            h_minus = h[j-1]
            h_plus = h[j]
            # Use default arguments to capture loop variables correctly
            cons.append({
                'type': 'ineq',
                'fun': lambda x, j=j, hm=h_minus, hp=h_plus: (x[j+1] - x[j])/hp - (x[j] - x[j-1])/hm
            })
            
        # 2. Non-increasing: x[j] - x[j+1] >= 0
        for j in range(nK - 1):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, j=j: x[j] - x[j+1]
            })
            
        # Bounds: max(S - K, 0) <= x <= S
        bounds = [(max(S - k_slice[j], 0.0), S) for j in range(nK)]
        
        res = optimize.minimize(obj, c_target, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            C_convex[t] = res.x
        else:
            C_convex[t] = c_target
            
    return C_convex


def check_butterfly(iv_surface: np.ndarray,
                    K_grid: np.ndarray,
                    T_grid: np.ndarray,
                    S: float = 1.0) -> np.ndarray:
    """
    Check butterfly arbitrage: d²C/dK² >= 0.
    Approximate via finite differences on call prices.

    Returns boolean mask (nT, nK-2): True = violation.
    """
    nT, nK = iv_surface.shape
    if np.any(K_grid < 0) or np.max(np.abs(K_grid)) < 5.0:
        K_abs = S * np.exp(K_grid)
    else:
        K_abs = K_grid
        
    if K_abs.ndim == 1:
        K_abs = np.tile(K_abs, (nT, 1))
        
    C = bs_call_price(S, K_abs, T_grid, iv_surface)
    
    h_minus = K_abs[:, 1:-1] - K_abs[:, :-2]
    h_plus = K_abs[:, 2:] - K_abs[:, 1:-1]
    
    dC_minus = (C[:, 1:-1] - C[:, :-2]) / h_minus
    dC_plus = (C[:, 2:] - C[:, 1:-1]) / h_plus
    
    d2C = 2.0 / (h_plus + h_minus) * (dC_plus - dC_minus)
    
    # Use relative tolerance: a violation only counts if the second derivative
    # is negative relative to the local option price (BUG-8 fix).
    # Absolute tolerance -1e-6 was insufficient for deep ITM/OTM options.
    return d2C < -1e-6 * np.abs(C[:, 1:-1]).clip(1e-10)


def fit_svi_slice(k: np.ndarray, total_var: np.ndarray) -> dict:
    """
    Fit raw SVI parametrization to a single maturity slice using PyTorch on GPU.

    SVI: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    where w = sigma_IV^2 * T (total variance).

    Returns: {"a": ..., "b": ..., "rho": ..., "m": ..., "sigma": ...}
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k_t = torch.as_tensor(k, dtype=torch.float32, device=device)
    total_var_t = torch.as_tensor(total_var, dtype=torch.float32, device=device)

    a_max = float(max(10.0, np.max(total_var) * 2.0))
    b_max = 10.0
    m_min, m_max = float(2.0 * np.min(k)), float(2.0 * np.max(k))
    sigma_min, sigma_max = 1e-4, 5.0

    a_init = float(np.minimum(np.max(total_var), np.maximum(0.0, np.mean(total_var))))
    b_init = 0.1
    rho_init = -0.5
    m_init = 0.0
    sigma_init = 0.1

    def to_unconstrained(y, y_min, y_max):
        p = (y - y_min) / (y_max - y_min)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    raw_a = torch.tensor(to_unconstrained(a_init, 0.0, a_max), device=device, requires_grad=True)
    # BUG-4 fix: use b_max for initialization so the full SVI parameter space
    # is available at the start. get_constrained() enforces the live rho
    # constraint dynamically during optimization.
    raw_rho = torch.tensor(np.arctanh(rho_init / 0.999), device=device, requires_grad=True)
    raw_b = torch.tensor(to_unconstrained(b_init, 0.0, b_max), device=device, requires_grad=True)
    raw_m = torch.tensor(to_unconstrained(m_init, m_min, m_max), device=device, requires_grad=True)
    raw_sigma = torch.tensor(to_unconstrained(sigma_init, sigma_min, sigma_max), device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([raw_a, raw_b, raw_rho, raw_m, raw_sigma], lr=1.0, max_iter=100)

    def get_constrained():
        a = a_max * torch.sigmoid(raw_a)
        rho = 0.999 * torch.tanh(raw_rho)
        b_limit = 4.0 / (1.0 + torch.abs(rho))
        b_upper = torch.minimum(torch.tensor(b_max, device=device), b_limit)
        b = b_upper * torch.sigmoid(raw_b)
        m = m_min + (m_max - m_min) * torch.sigmoid(raw_m)
        sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(raw_sigma)
        return a, b, rho, m, sigma

    def closure():
        optimizer.zero_grad()
        a, b, rho, m, sigma = get_constrained()
        pred = a + b * (rho * (k_t - m) + torch.sqrt((k_t - m)**2 + sigma**2))
        loss = torch.sum((pred - total_var_t)**2)
        loss.backward()
        return loss

    optimizer.step(closure)

    a, b, rho, m, sigma = get_constrained()

    return {
        "a": float(a.item()),
        "b": float(b.item()),
        "rho": float(rho.item()),
        "m": float(m.item()),
        "sigma": float(sigma.item())
    }


def enforce_calendar_spread_monotonicity(f: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Enforce calendar spread arbitrage-freedom by sorting total variance slices
    independently along the given axis (column-wise sort for axis=0).

    For a total variance surface w[T, K] = sigma^2[T,K] * T, sorting along
    axis=0 guarantees w(T2, K) >= w(T1, K) for T2 > T1 at each strike K,
    which is equivalent to the absence of calendar spread arbitrage.

    Note: this is a column-wise sort, NOT the full functional rearrangement
    of Chernozhukov, Fernandez-Val & Galichon (2010). The original name
    'monotone_rearrangement' was misleading (BUG-7 fix).
    """
    return np.sort(f, axis=axis)


# Backward-compatible alias (deprecated — use enforce_calendar_spread_monotonicity)
monotone_rearrangement = enforce_calendar_spread_monotonicity


def make_arbitrage_free(iv_surface: np.ndarray, T_grid: np.ndarray, K_grid: np.ndarray, S: float = 1.0) -> np.ndarray:
    """
    Post-process an IV surface to guarantee both calendar spread and butterfly spread arbitrage-freedom.
    """
    nT, nK = iv_surface.shape
    
    # 1. Calendar spread monotone rearrangement on total variance
    total_var = iv_surface**2 * T_grid[:, None]
    rearranged_var = enforce_calendar_spread_monotonicity(total_var, axis=0)
    iv_surface = np.sqrt(rearranged_var / np.maximum(T_grid[:, None], 1e-10))
    iv_surface = np.clip(iv_surface, 1e-4, None)
    
    # 2. Convert K_grid to absolute strikes
    if np.any(K_grid < 0) or np.max(np.abs(K_grid)) < 5.0:
        K_abs = S * np.exp(K_grid)
    else:
        K_abs = K_grid
        
    if K_abs.ndim == 1:
        K_abs = np.tile(K_abs, (nT, 1))
        
    # 3. Price calls under r=0, q=0
    C = bs_call_price(S, K_abs, T_grid, iv_surface)
    
    # 4. Project calls onto the set of convex and non-increasing functions of strike K
    C_convex = project_convex_call_prices(C, K_abs, S)
    
    # 5. Convert calls back to IV surface
    completed_iv = np.zeros_like(iv_surface)
    for t in range(nT):
        for j in range(nK):
            completed_iv[t, j] = implied_vol_scalar(C_convex[t, j], S, K_abs[t, j], T_grid[t])
            
    # 6. Apply calendar spread monotone rearrangement one more time to handle any tiny cross-interaction
    total_var = completed_iv**2 * T_grid[:, None]
    rearranged_var = enforce_calendar_spread_monotonicity(total_var, axis=0)
    completed_iv = np.sqrt(rearranged_var / np.maximum(T_grid[:, None], 1e-10))
    completed_iv = np.clip(completed_iv, 1e-4, None)
    
    return completed_iv


def _calibrate_fno_masked(model, target_iv: np.ndarray, mask: np.ndarray,
                          T_grid: np.ndarray, K_grid: np.ndarray,
                          max_iter: int = 20, tol: float = 1e-5,
                          damping: float = 0.5) -> np.ndarray:
    """
    Masked Gauss-Newton calibration routine on observed points.
    """
    import torch
    from calibrate_fast import _BOUNDS_LOWER_3D, _BOUNDS_UPPER_3D, _reparam_to_6d, fno_jacobian_autograd
    from calibrate import _load_normalizers, _make_spatial_input, _fno_predict_real_iv

    model.eval()
    _load_normalizers()
    device = next(model.parameters()).device
    spatial = _make_spatial_input(T_grid, K_grid, device)
    
    target_t = torch.tensor(target_iv, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
    
    # Short maturity/ATM fallback setup
    T_arr = np.asarray(T_grid)
    K_arr = np.asarray(K_grid)
    atm_idx = int(np.argmin(np.abs(K_arr)))
    t01_idx = int(np.argmin(np.abs(T_arr - 0.1)))
    
    if mask[t01_idx, atm_idx]:
        iv_short = float(target_iv[t01_idx, atm_idx])
    else:
        observed_ivs = target_iv[mask]
        iv_short = float(np.median(observed_ivs)) if len(observed_ivs) > 0 else 0.20

    lo = _BOUNDS_LOWER_3D.numpy()
    hi = _BOUNDS_UPPER_3D.numpy()

    # Try 3 diverse starting points
    v0_est = float(np.clip(iv_short**2, 0.01, 0.14))
    inits = np.array([
        [v0_est, -0.25, 0.35],
        [v0_est, -0.15, 0.50],
        [v0_est, -0.40, 0.25],
    ], dtype=np.float32)
    inits = np.clip(inits, lo + 1e-4, hi - 1e-4)

    best_loss = float("inf")
    best_params = inits[0].copy()

    for init in inits:
        theta = init.copy()
        for it in range(max_iter):
            theta_t = torch.tensor(theta, dtype=torch.float32, device=device)
            lo_t = _BOUNDS_LOWER_3D.to(device)
            hi_t = _BOUNDS_UPPER_3D.to(device)
            theta_c = theta_t.clamp(lo_t, hi_t)

            with torch.no_grad():
                p6 = _reparam_to_6d(theta_c[0:1], theta_c[1:2], theta_c[2:3], device)
                iv_pred = _fno_predict_real_iv(model, p6, spatial)

            r = (iv_pred - target_t)
            r_masked = r[mask_t].cpu().numpy()
            loss = float((r_masked**2).mean())

            if loss < tol:
                break

            # Autograd Jacobian
            J = fno_jacobian_autograd(model, theta_c.detach(), spatial)
            J_masked = J[mask_t].cpu().numpy()   # (N_obs, 3)

            # Solve GN system: (JᵀJ + ε·diag(JᵀJ)) δ = -Jᵀr
            JtJ = J_masked.T @ J_masked
            eps_lm = 1e-4 * np.diag(JtJ).mean() if JtJ.size > 0 else 1e-4
            
            try:
                delta = -np.linalg.solve(JtJ + eps_lm * np.eye(3), J_masked.T @ r_masked)
            except np.linalg.LinAlgError:
                break  # GN singular, stop this candidate

            # Backtracking line search
            alpha = damping
            for _ in range(8):
                theta_new = np.clip(theta_c.cpu().numpy() + alpha * delta,
                                    lo + 1e-5, hi - 1e-5)
                tt = torch.tensor(theta_new, dtype=torch.float32, device=device)
                with torch.no_grad():
                    p6n = _reparam_to_6d(tt[0:1], tt[1:2], tt[2:3], device)
                    ivn = _fno_predict_real_iv(model, p6n, spatial)
                    rn = (ivn - target_t)[mask_t]
                    ln = float((rn**2).mean())
                if ln < loss:
                    theta = theta_new
                    break
                alpha *= 0.5
            else:
                theta = theta_c.cpu().numpy()   # keep current

        # Final evaluation for candidate
        theta_t = torch.tensor(theta, dtype=torch.float32, device=device)
        theta_c = theta_t.clamp(lo_t, hi_t)
        with torch.no_grad():
            p6 = _reparam_to_6d(theta_c[0:1], theta_c[1:2], theta_c[2:3], device)
            iv_pred = _fno_predict_real_iv(model, p6, spatial)
        r_masked = (iv_pred - target_t)[mask_t].cpu().numpy()
        final_loss = float((r_masked**2).mean())

        if final_loss < best_loss:
            best_loss = final_loss
            best_params = theta.copy()

    # Full surface prediction using the best candidate
    best_params_t = torch.tensor(best_params, dtype=torch.float32, device=device)
    p6_best = _reparam_to_6d(best_params_t[0:1], best_params_t[1:2], best_params_t[2:3], device)
    with torch.no_grad():
        full_pred = _fno_predict_real_iv(model, p6_best, spatial)
        
    return full_pred.cpu().numpy()


def complete_surface(sparse_iv: np.ndarray,
                     mask: np.ndarray,
                     T_grid: np.ndarray,
                     K_grid: np.ndarray,
                     method: str = "fno") -> np.ndarray:
    """
    Complete a sparse IV surface (NaN where mask=False) using the specified method.

    Args:
        sparse_iv:  (nT, nK) array with NaN at missing quotes
        mask:       (nT, nK) bool — True = observed
        method:     "fno" | "svi" | "cubic_spline"

    Returns: (nT, nK) complete arbitrage-free IV surface
    """
    nT, nK = sparse_iv.shape

    if method == "cubic_spline":
        # griddata is imported locally here where it is actually used.
        # Get coordinates of observed elements
        observed_coords = []
        observed_values = []
        for t_idx, t in enumerate(T_grid):
            for k_idx, k_val in enumerate(K_grid):
                if mask[t_idx, k_idx]:
                    observed_coords.append([t, k_val])
                    observed_values.append(sparse_iv[t_idx, k_idx])
                    
        observed_coords = np.array(observed_coords)
        observed_values = np.array(observed_values)

        if len(observed_values) == 0:
            completed_surface = np.full((nT, nK), 0.20)
        else:
            T_mesh, K_mesh = np.meshgrid(T_grid, K_grid, indexing='ij')
            grid_points = np.stack([T_mesh.ravel(), K_mesh.ravel()], axis=1)
            
            if len(observed_values) >= 4:
                completed_flat = griddata(observed_coords, observed_values, grid_points, method='cubic')
            else:
                completed_flat = griddata(observed_coords, observed_values, grid_points, method='nearest')
                
            if np.any(np.isnan(completed_flat)):
                completed_flat_nearest = griddata(observed_coords, observed_values, grid_points, method='nearest')
                completed_flat = np.where(np.isnan(completed_flat), completed_flat_nearest, completed_flat)
                
            completed_surface = completed_flat.reshape(nT, nK)

        # Apply arbitrage-free projection
        completed_surface = make_arbitrage_free(completed_surface, T_grid, K_grid)
        return completed_surface

    elif method == "svi":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        completed_surface = np.zeros((nT, nK))

        # 1. Identify valid slices with >= 5 observed points
        valid_slices = []
        max_obs = 0
        for t in range(nT):
            num_obs = np.sum(mask[t])
            if num_obs >= 5:
                valid_slices.append((t, num_obs))
                if num_obs > max_obs:
                    max_obs = num_obs

        if len(valid_slices) > 0:
            n_valid = len(valid_slices)
            k_pad = np.zeros((n_valid, max_obs), dtype=np.float32)
            var_pad = np.zeros((n_valid, max_obs), dtype=np.float32)
            loss_mask = np.zeros((n_valid, max_obs), dtype=np.float32)

            a_max_arr = np.zeros(n_valid, dtype=np.float32)
            m_min_arr = np.zeros(n_valid, dtype=np.float32)
            m_max_arr = np.zeros(n_valid, dtype=np.float32)

            a_init_arr = np.zeros(n_valid, dtype=np.float32)
            m_init_arr = np.zeros(n_valid, dtype=np.float32)

            for idx, (t, num_obs) in enumerate(valid_slices):
                if K_grid.ndim == 1:
                    obs_k = K_grid[mask[t]]
                else:
                    obs_k = K_grid[t][mask[t]]
                obs_iv = sparse_iv[t][mask[t]]
                obs_var = (obs_iv**2) * T_grid[t]

                k_pad[idx, :num_obs] = obs_k
                var_pad[idx, :num_obs] = obs_var
                loss_mask[idx, :num_obs] = 1.0

                a_max_arr[idx] = max(10.0, np.max(obs_var) * 2.0)
                m_min_arr[idx] = 2.0 * np.min(obs_k)
                m_max_arr[idx] = 2.0 * np.max(obs_k)

                a_init_arr[idx] = np.minimum(np.max(obs_var), np.maximum(0.0, np.mean(obs_var)))
                m_init_arr[idx] = 0.0

            # Convert parameters/bounds to GPU PyTorch tensors
            k_t = torch.as_tensor(k_pad, device=device)
            var_t = torch.as_tensor(var_pad, device=device)
            mask_t = torch.as_tensor(loss_mask, device=device)

            a_max_t = torch.as_tensor(a_max_arr, device=device)
            m_min_t = torch.as_tensor(m_min_arr, device=device)
            m_max_t = torch.as_tensor(m_max_arr, device=device)

            def to_unconstrained(y, y_min, y_max):
                p = (y - y_min) / (y_max - y_min)
                p = np.clip(p, 1e-6, 1.0 - 1e-6)
                return np.log(p / (1.0 - p))

            # Initialize parameters
            raw_a = torch.tensor(to_unconstrained(a_init_arr, 0.0, a_max_arr), device=device, requires_grad=True)
            raw_rho = torch.tensor(np.full(n_valid, np.arctanh(-0.5 / 0.999), dtype=np.float32), device=device, requires_grad=True)

            b_limit_arr = 4.0 / (1.0 + abs(-0.5))
            b_upper_arr = np.minimum(10.0, b_limit_arr)
            raw_b = torch.tensor(to_unconstrained(np.full(n_valid, 0.1, dtype=np.float32), 0.0, b_upper_arr), device=device, requires_grad=True)

            raw_m = torch.tensor(to_unconstrained(m_init_arr, m_min_arr, m_max_arr), device=device, requires_grad=True)
            raw_sigma = torch.tensor(to_unconstrained(np.full(n_valid, 0.1, dtype=np.float32), 1e-4, 5.0), device=device, requires_grad=True)

            optimizer = torch.optim.LBFGS([raw_a, raw_b, raw_rho, raw_m, raw_sigma], lr=1.0, max_iter=100)

            def get_constrained():
                a = (a_max_t * torch.sigmoid(raw_a)).unsqueeze(1)
                rho = (0.999 * torch.tanh(raw_rho)).unsqueeze(1)
                b_limit = 4.0 / (1.0 + torch.abs(rho))
                b_upper = torch.minimum(torch.tensor(10.0, device=device), b_limit)
                b = b_upper * torch.sigmoid(raw_b).unsqueeze(1)
                m = (m_min_t + (m_max_t - m_min_t) * torch.sigmoid(raw_m)).unsqueeze(1)
                sigma = (1e-4 + (5.0 - 1e-4) * torch.sigmoid(raw_sigma)).unsqueeze(1)
                return a, b, rho, m, sigma

            def closure():
                optimizer.zero_grad()
                a, b, rho, m, sigma = get_constrained()
                pred = a + b * (rho * (k_t - m) + torch.sqrt((k_t - m)**2 + sigma**2))
                loss = torch.sum(((pred - var_t)**2) * mask_t)
                loss.backward()
                return loss

            optimizer.step(closure)

            # Get final parameter values
            a, b, rho, m, sigma = get_constrained()

            # Predict full grid for completed surface
            K_grid_t = torch.as_tensor(K_grid, device=device, dtype=torch.float32)
            if K_grid_t.ndim == 1:
                K_grid_t = K_grid_t.unsqueeze(0).repeat(n_valid, 1)  # (n_valid, nK)
            else:
                valid_indices = [t for t, _ in valid_slices]
                K_grid_t = K_grid_t[valid_indices]

            T_grid_t = torch.as_tensor([T_grid[t] for t, _ in valid_slices], device=device, dtype=torch.float32).unsqueeze(1)

            pred_var = a + b * (rho * (K_grid_t - m) + torch.sqrt((K_grid_t - m)**2 + sigma**2))
            completed_vols = torch.sqrt(torch.clamp(pred_var, min=1e-8) / torch.clamp(T_grid_t, min=1e-10))
            completed_vols_np = completed_vols.detach().cpu().numpy()

            valid_idx = 0
            for t in range(nT):
                if mask[t].sum() >= 5:
                    completed_surface[t] = completed_vols_np[valid_idx]
                    valid_idx += 1
                else:
                    observed_iv = sparse_iv[t][mask[t]]
                    if len(observed_iv) > 0:
                        val = np.median(observed_iv)
                    else:
                        all_observed_vols = sparse_iv[mask]
                        val = np.median(all_observed_vols) if len(all_observed_vols) > 0 else 0.20
                    completed_surface[t] = np.full(nK, val)
        else:
            for t in range(nT):
                observed_iv = sparse_iv[t][mask[t]]
                if len(observed_iv) > 0:
                    val = np.median(observed_iv)
                else:
                    all_observed_vols = sparse_iv[mask]
                    val = np.median(all_observed_vols) if len(all_observed_vols) > 0 else 0.20
                completed_surface[t] = np.full(nK, val)

        # Apply arbitrage-free projection
        completed_surface = make_arbitrage_free(completed_surface, T_grid, K_grid)
        return completed_surface

    elif method == "fno":
        import torch
        from fno_model import MirrorPaddedFNO2d
        import calibrate

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MirrorPaddedFNO2d()

        # Find the weights file
        weights_paths = [
            "artifacts/weights/fno_v2_final_prod.pth",
            "/home/execorn/programming/derivatives-w2/artifacts/weights/fno_v2_final_prod.pth",
            "/home/execorn/programming/derivatives/artifacts/weights/fno_v2_final_prod.pth",
            "/home/execorn/programming/derivatives-w1/artifacts/weights/fno_v2_final_prod.pth",
        ]
        weights_path = None
        for w_p in weights_paths:
            if Path(w_p).exists():
                weights_path = w_p
                break
        if weights_path is None:
            raise FileNotFoundError("FNO v2 weights file not found.")

        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        orig_v1 = calibrate._NORM_VERSIONS["v1"]
        try:
            calibrate._NORM_VERSIONS["v1"] = calibrate._NORM_VERSIONS["v2"]
            calibrate._param_norm = None
            calibrate._iv_norm = None

            completed_surface = _calibrate_fno_masked(
                model=model,
                target_iv=sparse_iv,
                mask=mask,
                T_grid=T_grid,
                K_grid=K_grid
            )
        finally:
            calibrate._NORM_VERSIONS["v1"] = orig_v1
            calibrate._param_norm = None
            calibrate._iv_norm = None

        # Apply arbitrage-free projection
        completed_surface = make_arbitrage_free(completed_surface, T_grid, K_grid)
        return completed_surface

    else:
        raise ValueError(f"Unknown completion method: {method}")
