"""
fx_calibration.py — SABR beta=1.0 model and Levenberg-Marquardt calibration using PyTorch forward-mode autograd.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Union

# ── 1. SABR Model Implied Volatility in PyTorch ──

def sabr_iv_lognormal_pytorch(
    F: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    alpha: torch.Tensor,
    rho: torch.Tensor,
    nu: torch.Tensor,
    beta: float = 1.0
) -> torch.Tensor:
    """
    Computes Hagan's approximate lognormal implied volatility for beta=1.0 in PyTorch.
    Differentiable and robust near ATM (K -> F).
    """
    # z = (nu / alpha) * ln(F/K)
    log_FK = torch.log(F / K)
    z = (nu / alpha) * log_FK
    
    # We use a Taylor series approximation for small |z| to avoid NaN or division-by-zero.
    z_abs = torch.abs(z)
    is_near_atm = z_abs < 1e-5
    
    # Safeguard z to avoid division by zero or NaN gradients in the division branch
    z_safe = torch.where(is_near_atm, torch.ones_like(z), z)
    
    # chi(z)
    temp = torch.sqrt(1.0 - 2.0 * rho * z_safe + z_safe**2) + z_safe - rho
    temp = torch.clamp(temp / (1.0 - rho), min=1e-15)
    chi_z = torch.log(temp)
    
    div_branch = z_safe / chi_z
    taylor_branch = 1.0 + 0.5 * rho * z + ((3.0 * rho**2 - 1.0) / 12.0) * z**2
    
    f_z = torch.where(is_near_atm, taylor_branch, div_branch)
    
    # Correction term for beta = 1.0:
    # 1 + [ 1/4 * rho * nu * alpha + (2 - 3*rho^2)/24 * nu^2 ] * T
    correction = 1.0 + (0.25 * rho * nu * alpha + (2.0 - 3.0 * rho**2) / 24.0 * nu**2) * T
    
    return alpha * f_z * correction


# ── 2. Levenberg-Marquardt Calibration ──

def calibrate_sabr_fx(
    F: float,
    strikes: Union[np.ndarray, list],
    market_vols: Union[np.ndarray, list],
    T: float,
    r_d: float,
    r_f: float,
    beta: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-8
) -> Dict[str, float]:
    """
    Calibrates SABR (beta=1.0) parameters (alpha, rho, nu) to market implied volatilities.
    Uses Levenberg-Marquardt in the unconstrained space with exact analytical Jacobians
    computed via PyTorch forward-mode autograd (torch.func.jacfwd).
    
    Parameters
    ----------
    F : float
        Forward price.
    strikes : array-like
        Strikes of option quotes.
    market_vols : array-like
        Implied volatilities of option quotes.
    T : float
        Time to maturity in years.
    r_d : float
        Domestic interest rate.
    r_f : float
        Foreign interest rate.
    beta : float
        CEV parameter (fixed to 1.0).
    max_iter : int
        Maximum Levenberg-Marquardt iterations.
    tol : float
        Convergence tolerance on J^T * residual.
        
    Returns
    -------
    params : dict
        Calibrated parameters {'alpha': alpha, 'rho': rho, 'nu': nu}.
    """
    # 1. Convert inputs to PyTorch tensors (double precision for root finding stability)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    F_t = torch.tensor(F, dtype=torch.float64, device=device)
    T_t = torch.tensor(T, dtype=torch.float64, device=device)
    strikes_t = torch.tensor(strikes, dtype=torch.float64, device=device)
    market_vols_t = torch.tensor(market_vols, dtype=torch.float64, device=device)
    
    # 2. Initial guess
    # Use closest to ATM strike vol as alpha estimate
    strikes_arr = np.asarray(strikes)
    atm_idx = np.argmin(np.abs(strikes_arr - F))
    atm_vol = float(market_vols[atm_idx])
    
    alpha_0 = max(atm_vol, 0.001)
    rho_0 = 0.0
    nu_0 = 0.3
    
    # Parameter transformations for unconstrained optimization:
    # alpha = exp(theta_0) -> alpha > 0
    # rho = tanh(theta_1) -> rho in (-1, 1)
    # nu = exp(theta_2) -> nu > 0
    theta = np.array([np.log(alpha_0), np.arctanh(rho_0), np.log(nu_0)], dtype=np.float64)
    theta_t = torch.tensor(theta, dtype=torch.float64, device=device)
    
    # 3. Define residual function for forward-mode autograd
    def residual_fn(theta_trans: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(theta_trans[0])
        rho = torch.tanh(theta_trans[1])
        nu = torch.exp(theta_trans[2])
        
        pred_vols = sabr_iv_lognormal_pytorch(F_t, strikes_t, T_t, alpha, rho, nu, beta=beta)
        return pred_vols - market_vols_t
        
    # 4. Levenberg-Marquardt Optimization Loop
    lambda_lm = 1e-3
    r = residual_fn(theta_t)
    loss = torch.sum(r**2).item()
    
    for it in range(max_iter):
        # Exact analytical Jacobian J of shape (N, 3) using forward autograd
        J = torch.func.jacfwd(residual_fn)(theta_t)
        
        J_np = J.cpu().numpy()
        r_np = r.cpu().numpy()
        
        JtJ = J_np.T @ J_np
        Jtr = J_np.T @ r_np
        
        # Check convergence on gradient norm
        if np.linalg.norm(Jtr) < tol:
            break
            
        success = False
        # Try damping values
        for search_it in range(10):
            # LM system: (J^T J + lambda_lm * I) * delta = -J^T r
            A = JtJ + lambda_lm * np.eye(3)
            try:
                delta = np.linalg.solve(A, -Jtr)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A, -Jtr, rcond=None)[0]
                
            theta_new = theta_t.cpu().numpy() + delta
            theta_new_t = torch.tensor(theta_new, dtype=torch.float64, device=device)
            
            with torch.no_grad():
                r_new = residual_fn(theta_new_t)
                loss_new = torch.sum(r_new**2).item()
                
            if loss_new < loss:
                # Accept step
                theta_t = theta_new_t
                r = r_new
                loss = loss_new
                lambda_lm *= 0.1
                success = True
                break
            else:
                # Reject step, increase damping
                lambda_lm *= 10.0
                
        if not success:
            # If no step reduces loss, terminate early
            break
            
    # Convert back to constrained space
    alpha_cal = torch.exp(theta_t[0]).item()
    rho_cal = torch.tanh(theta_t[1]).item()
    nu_cal = torch.exp(theta_t[2]).item()
    
    return {
        "alpha": alpha_cal,
        "rho": rho_cal,
        "nu": nu_cal
    }


# ── 3. 2D SABR Calibration with Analytical Alpha ──

def solve_sabr_alpha(
    sigma_atm: torch.Tensor,
    T: torch.Tensor,
    rho: torch.Tensor,
    nu: torch.Tensor
) -> torch.Tensor:
    """
    Solves the quadratic equation a * alpha^2 + b * alpha - sigma_atm = 0 for alpha.
    Differentiable under PyTorch and robust near the rho=0/nu=0 limit.
    """
    a = 0.25 * rho * nu * T
    b = 1.0 + ((2.0 - 3.0 * rho**2) / 24.0) * (nu**2) * T
    
    discriminant = b**2 + 4.0 * a * sigma_atm
    discriminant_safe = torch.clamp(discriminant, min=1e-15)
    
    alpha = (2.0 * sigma_atm) / (b + torch.sqrt(discriminant_safe))
    return alpha


def sabr_initial_guess(
    F: float,
    T: float,
    market_strikes: Union[np.ndarray, list],
    market_vols: Union[np.ndarray, list]
) -> Tuple[float, float]:
    """
    Generates initial guesses for rho and nu from market strikes and vols using
    25-Delta Risk Reversals and Butterfly quotes.
    """
    strikes_arr = np.asarray(market_strikes)
    vols_arr = np.asarray(market_vols)
    
    # Extract ATM vol (closest strike to F)
    atm_idx = np.argmin(np.abs(strikes_arr - F))
    sigma_atm = float(vols_arr[atm_idx])
    
    # Approximate Delta_k = sigma_atm * sqrt(T) * N^-1(0.75)
    # N^-1(0.75) is approx 0.6744897501960817
    inv_N_75 = 0.6744897501960817
    Delta_k = sigma_atm * np.sqrt(T) * inv_N_75
    
    if Delta_k < 1e-8 or len(strikes_arr) < 3:
        return 0.0, 0.3
        
    # 25-Delta strikes
    half_vol_T = 0.5 * sigma_atm**2 * T
    K_25C = F * np.exp(Delta_k + half_vol_T)
    K_25P = F * np.exp(-Delta_k + half_vol_T)
    
    # Find closest strikes in market_strikes
    c_idx = np.argmin(np.abs(strikes_arr - K_25C))
    p_idx = np.argmin(np.abs(strikes_arr - K_25P))
    
    if c_idx == p_idx or c_idx == atm_idx or p_idx == atm_idx:
        # Fallback to simple strike-based search if index collision
        # e.g., look for strikes that are larger/smaller than F
        above_atm = strikes_arr[strikes_arr > F]
        below_atm = strikes_arr[strikes_arr < F]
        if len(above_atm) > 0 and len(below_atm) > 0:
            K_25C = above_atm[np.argmin(np.abs(above_atm - (F * 1.05)))]
            K_25P = below_atm[np.argmin(np.abs(below_atm - (F * 0.95)))]
            c_idx = np.where(strikes_arr == K_25C)[0][0]
            p_idx = np.where(strikes_arr == K_25P)[0][0]
        else:
            return 0.0, 0.3
            
    sigma_c = float(vols_arr[c_idx])
    sigma_p = float(vols_arr[p_idx])
    
    # Risk Reversal and Butterfly
    RR_25 = sigma_c - sigma_p
    BF_25 = 0.5 * (sigma_c + sigma_p) - sigma_atm
    
    # nu_init and rho_init
    val_sqrt = (2.0 * BF_25) / (Delta_k**2) + (RR_25 / Delta_k)**2
    nu_init = np.sqrt(max(val_sqrt, 1e-6))
    rho_init = RR_25 / (nu_init * Delta_k)
    
    # Clamp to physical domains
    rho_init = np.clip(rho_init, -0.999, 0.999)
    nu_init = max(nu_init, 1e-3)
    
    return float(rho_init), float(nu_init)


def _residual_fn_2d_raw(
    theta_trans: torch.Tensor,
    F_t: torch.Tensor,
    strikes_t: torch.Tensor,
    T_t: torch.Tensor,
    market_vols_t: torch.Tensor,
    sigma_atm_t: torch.Tensor,
    beta: float
) -> torch.Tensor:
    rho = torch.tanh(theta_trans[0])
    nu = torch.exp(theta_trans[1])
    alpha = solve_sabr_alpha(sigma_atm_t, T_t, rho, nu)
    pred_vols = sabr_iv_lognormal_pytorch(F_t, strikes_t, T_t, alpha, rho, nu, beta=beta)
    return pred_vols - market_vols_t


_jacobian_fn_2d_raw = torch.func.jacfwd(_residual_fn_2d_raw, argnums=0)

# Caches for compiled functions to avoid compilation overhead on every run
_compiled_res_cache_2d = {}
_compiled_jac_cache_2d = {}


def _get_compiled_fns_2d(device):
    key = str(device)
    if key not in _compiled_res_cache_2d:
        _compiled_res_cache_2d[key] = torch.compile(_residual_fn_2d_raw)
        _compiled_jac_cache_2d[key] = torch.compile(_jacobian_fn_2d_raw)
    return _compiled_res_cache_2d[key], _compiled_jac_cache_2d[key]


def calibrate_sabr_fx_2d(
    F: float,
    strikes: Union[np.ndarray, list],
    market_vols: Union[np.ndarray, list],
    T: float,
    r_d: float,
    r_f: float,
    beta: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-8,
    use_jit: bool = False
) -> Dict[str, float]:
    """
    Calibrates SABR (beta=1.0) parameters (alpha, rho, nu) to market implied volatilities.
    Reduces the optimization search space to 2D by solving for alpha analytically at each step.
    Uses Levenberg-Marquardt in the unconstrained space with exact analytical Jacobians
    computed via PyTorch forward-mode autograd (torch.func.jacfwd).
    
    Parameters
    ----------
    F : float
        Forward price.
    strikes : array-like
        Strikes of option quotes.
    market_vols : array-like
        Implied volatilities of option quotes.
    T : float
        Time to maturity in years.
    r_d : float
        Domestic interest rate.
    r_f : float
        Foreign interest rate.
    beta : float
        CEV parameter (fixed to 1.0).
    max_iter : int
        Maximum Levenberg-Marquardt iterations.
    tol : float
        Convergence tolerance on J^T * residual.
    use_jit : bool
        Whether to use JIT compilation via torch.compile.
        
    Returns
    -------
    params : dict
        Calibrated parameters {'alpha': alpha, 'rho': rho, 'nu': nu}.
    """
    # 1. Convert inputs to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    F_t = torch.tensor(F, dtype=torch.float64, device=device)
    T_t = torch.tensor(T, dtype=torch.float64, device=device)
    strikes_t = torch.tensor(strikes, dtype=torch.float64, device=device)
    market_vols_t = torch.tensor(market_vols, dtype=torch.float64, device=device)
    
    # Extract ATM vol
    strikes_arr = np.asarray(strikes)
    atm_idx = np.argmin(np.abs(strikes_arr - F))
    sigma_atm = float(market_vols[atm_idx])
    sigma_atm_t = torch.tensor(sigma_atm, dtype=torch.float64, device=device)
    
    # 2. Initial guess
    rho_0, nu_0 = sabr_initial_guess(F, T, strikes, market_vols)
    
    # Parameter transformations for unconstrained 2D optimization:
    # rho = tanh(theta_0) -> rho in (-1, 1)
    # nu = exp(theta_1) -> nu > 0
    theta = np.array([np.arctanh(rho_0), np.log(nu_0)], dtype=np.float64)
    theta_t = torch.tensor(theta, dtype=torch.float64, device=device)
    
    # 3. Setup Residual & Jacobian Functions
    if use_jit:
        compiled_res, compiled_jac = _get_compiled_fns_2d(device)
        def residual_fn(theta_trans: torch.Tensor) -> torch.Tensor:
            return compiled_res(theta_trans, F_t, strikes_t, T_t, market_vols_t, sigma_atm_t, beta)
        def jacobian_fn(theta_trans: torch.Tensor) -> torch.Tensor:
            return compiled_jac(theta_trans, F_t, strikes_t, T_t, market_vols_t, sigma_atm_t, beta)
    else:
        def residual_fn(theta_trans: torch.Tensor) -> torch.Tensor:
            return _residual_fn_2d_raw(theta_trans, F_t, strikes_t, T_t, market_vols_t, sigma_atm_t, beta)
        def jacobian_fn(theta_trans: torch.Tensor) -> torch.Tensor:
            return _jacobian_fn_2d_raw(theta_trans, F_t, strikes_t, T_t, market_vols_t, sigma_atm_t, beta)
            
    # 4. Levenberg-Marquardt Optimization Loop
    lambda_lm = 1e-3
    r = residual_fn(theta_t)
    loss = torch.sum(r**2).item()
    
    for it in range(max_iter):
        J = jacobian_fn(theta_t)
        
        J_np = J.cpu().numpy()
        r_np = r.cpu().numpy()
        
        JtJ = J_np.T @ J_np
        Jtr = J_np.T @ r_np
        
        # Check convergence on gradient norm
        if np.linalg.norm(Jtr) < tol:
            break
            
        success = False
        # Try damping values
        for search_it in range(10):
            A = JtJ + lambda_lm * np.eye(2)
            try:
                delta = np.linalg.solve(A, -Jtr)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A, -Jtr, rcond=None)[0]
                
            theta_new = theta_t.cpu().numpy() + delta
            theta_new_t = torch.tensor(theta_new, dtype=torch.float64, device=device)
            
            with torch.no_grad():
                r_new = residual_fn(theta_new_t)
                loss_new = torch.sum(r_new**2).item()
                
            if loss_new < loss:
                # Accept step
                theta_t = theta_new_t
                r = r_new
                loss = loss_new
                lambda_lm *= 0.1
                success = True
                break
            else:
                # Reject step, increase damping
                lambda_lm *= 10.0
                
        if not success:
            # If no step reduces loss, terminate early
            break
            
    # Convert back to constrained space
    rho_cal = torch.tanh(theta_t[0]).item()
    nu_cal = torch.exp(theta_t[1]).item()
    alpha_cal = solve_sabr_alpha(sigma_atm_t, T_t, torch.tanh(theta_t[0]), torch.exp(theta_t[1])).item()
    
    return {
        "alpha": alpha_cal,
        "rho": rho_cal,
        "nu": nu_cal
    }
