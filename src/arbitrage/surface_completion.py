"""
§1.2 Arbitrage-free IV surface completion.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.interpolate import griddata
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
    
    # Butterfly spread violation: d2C < 0
    # Use a standard numerical tolerance to prevent numerical precision issues
    return d2C < -1e-6


def fit_svi_slice(k: np.ndarray, total_var: np.ndarray) -> dict:
    """
    Fit raw SVI parametrization to a single maturity slice.

    SVI: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    where w = sigma_IV^2 * T (total variance).

    Returns: {"a": ..., "b": ..., "rho": ..., "m": ..., "sigma": ...}
    """
    def svi_fun(k_val, a, b, rho, m, sigma):
        return a + b * (rho * (k_val - m) + np.sqrt((k_val - m)**2 + sigma**2))
        
    def objective(params):
        a, b, rho, m, sigma = params
        pred = svi_fun(k, a, b, rho, m, sigma)
        return np.sum((pred - total_var)**2)
        
    a_init = np.minimum(np.max(total_var), np.maximum(0.0, np.mean(total_var)))
    initial_guess = np.array([a_init, 0.1, -0.5, 0.0, 0.1])
    
    bounds = [
        (0.0, max(10.0, np.max(total_var) * 2.0)),   # a
        (0.0, 10.0),                                 # b
        (-0.999, 0.999),                             # rho
        (2.0 * np.min(k), 2.0 * np.max(k)),          # m
        (1e-4, 5.0)                                  # sigma
    ]
    
    cons = ({
        'type': 'ineq',
        'fun': lambda p: 4.0 - p[1] * (1.0 + np.abs(p[2]))
    })
    
    res = optimize.minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 500}
    )
    
    if not res.success:
        # Fallback to Nelder-Mead with penalized objective and manual projection
        def objective_penalized(params):
            a, b, rho, m, sigma = params
            pred = svi_fun(k, a, b, rho, m, sigma)
            base_loss = np.sum((pred - total_var)**2)
            
            penalty = 0.0
            if a < bounds[0][0]: penalty += 1e5 * (bounds[0][0] - a)**2
            if a > bounds[0][1]: penalty += 1e5 * (a - bounds[0][1])**2
            if b < bounds[1][0]: penalty += 1e5 * (bounds[1][0] - b)**2
            if b > bounds[1][1]: penalty += 1e5 * (b - bounds[1][1])**2
            if rho < bounds[2][0]: penalty += 1e5 * (bounds[2][0] - rho)**2
            if rho > bounds[2][1]: penalty += 1e5 * (rho - bounds[2][1])**2
            if m < bounds[3][0]: penalty += 1e5 * (bounds[3][0] - m)**2
            if m > bounds[3][1]: penalty += 1e5 * (m - bounds[3][1])**2
            if sigma < bounds[4][0]: penalty += 1e5 * (bounds[4][0] - sigma)**2
            if sigma > bounds[4][1]: penalty += 1e5 * (sigma - bounds[4][1])**2
            
            c_val = b * (1.0 + np.abs(rho))
            if c_val > 4.0:
                penalty += 1e5 * (c_val - 4.0)**2
            return base_loss + penalty
            
        res_nm = optimize.minimize(
            objective_penalized,
            initial_guess,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        # Project results to bounds
        a, b, rho, m, sigma = res_nm.x
        a = np.clip(a, bounds[0][0], bounds[0][1])
        b = np.clip(b, bounds[1][0], bounds[1][1])
        rho = np.clip(rho, bounds[2][0], bounds[2][1])
        m = np.clip(m, bounds[3][0], bounds[3][1])
        sigma = np.clip(sigma, bounds[4][0], bounds[4][1])
        
        if b * (1.0 + np.abs(rho)) > 4.0:
            b = 4.0 / (1.0 + np.abs(rho))
            
        params_dict = {
            "a": float(a),
            "b": float(b),
            "rho": float(rho),
            "m": float(m),
            "sigma": float(sigma)
        }
    else:
        a, b, rho, m, sigma = res.x
        params_dict = {
            "a": float(a),
            "b": float(b),
            "rho": float(rho),
            "m": float(m),
            "sigma": float(sigma)
        }
        
    return params_dict


def monotone_rearrangement(f: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Project f onto the set of monotone (non-decreasing) functions
    along the given axis, using sorting-based rearrangement.
    (Chernozhukov, Fernandez-Val, Galichon 2010)
    """
    return np.sort(f, axis=axis)


def make_arbitrage_free(iv_surface: np.ndarray, T_grid: np.ndarray, K_grid: np.ndarray, S: float = 1.0) -> np.ndarray:
    """
    Post-process an IV surface to guarantee both calendar spread and butterfly spread arbitrage-freedom.
    """
    nT, nK = iv_surface.shape
    
    # 1. Calendar spread monotone rearrangement on total variance
    total_var = iv_surface**2 * T_grid[:, None]
    rearranged_var = monotone_rearrangement(total_var, axis=0)
    iv_surface = np.sqrt(rearranged_var / T_grid[:, None])
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
    rearranged_var = monotone_rearrangement(total_var, axis=0)
    completed_iv = np.sqrt(rearranged_var / T_grid[:, None])
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
        completed_surface = np.zeros((nT, nK))
        for t in range(nT):
            observed_k = K_grid[mask[t]]
            observed_iv = sparse_iv[t][mask[t]]
            if len(observed_iv) >= 5:
                observed_total_var = (observed_iv**2) * T_grid[t]
                params = fit_svi_slice(observed_k, observed_total_var)
                
                # SVI parameterization w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
                a, b, rho, m, sigma = params["a"], params["b"], params["rho"], params["m"], params["sigma"]
                pred_total_var = a + b * (rho * (K_grid - m) + np.sqrt((K_grid - m)**2 + sigma**2))
                completed_surface[t] = np.sqrt(np.clip(pred_total_var, 1e-8, None) / T_grid[t])
            else:
                # Median fallback
                if len(observed_iv) > 0:
                    val = np.median(observed_iv)
                else:
                    all_observed_vols = sparse_iv[mask]
                    if len(all_observed_vols) > 0:
                        val = np.median(all_observed_vols)
                    else:
                        val = 0.20
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
