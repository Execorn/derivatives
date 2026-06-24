"""
heston.py — Exact Fourier-COS pricing for the Classic Heston model on CPU and GPU.

Implements:
  1. heston_cf: Stable Gatheral characteristic function for the Heston model using log1p.
  2. heston_iv_surface: CPU option pricer and IV surface generator.
  3. batch_heston_iv_surface: GPU-batched vectorized version using PyTorch.
"""

from typing import Union, Dict, Any, Tuple, Optional
import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import minimize
from functools import lru_cache

_A = -4.0
_B = 4.0
_SQRT2 = 1.4142135623730951
_INVSQRT2PI = 0.3989422804014327

# ---------------------------------------------------------------------------
# COS payoff helpers (NumPy)
# ---------------------------------------------------------------------------

def cos_payoff_coeffs_np(N_cos: int, a: float = _A, b: float = _B) -> np.ndarray:
    """
    Exact COS call option payoff coefficients on [0, b].
    
    Formula:
      V_k = \\frac{2}{b-a} (\\chi_k(0, b) - \\psi_k(0, b))
      
    Academic Reference:
      Fang, F., & Oosterlee, C. W. (2008). A novel pricing method for European options 
      based on Fourier-cosine series expansions. SIAM Journal on Scientific Computing, 
      31(2), 826-848.
      
    Parameters
    ----------
    N_cos : int
        Number of cosine terms.
    a : float
        Lower integration boundary.
    b : float
        Upper integration boundary.
        
    Returns
    -------
    coefficients : np.ndarray
        Payoff coefficients of shape (N_cos,).
    """
    k = np.arange(N_cos, dtype=np.float64)
    uk = k * np.pi / (b - a)

    with np.errstate(divide='ignore', invalid='ignore'):
        chi = np.real(
            np.exp(-1j * uk * a)
            * (np.exp((1.0 + 1j * uk) * b) - 1.0)
            / (1.0 + 1j * uk)
        )
    chi[0] = np.exp(b) - 1.0

    safe_uk = np.where(k == 0, 1.0, uk)
    psi = np.where(
        k == 0,
        b,
        (np.sin(uk * (b - a)) + np.sin(uk * a)) / safe_uk,
    )

    Vk = (2.0 / (b - a)) * (chi - psi)
    Vk[0] *= 0.5
    return Vk


def cos_payoff_coeffs_put_np(N_cos: int, a: float = _A, b: float = _B) -> np.ndarray:
    """
    Exact COS put option payoff coefficients on [a, 0].
    
    Formula:
      V_k = \\frac{2}{b-a} (\\psi_k(a, 0) - \\chi_k(a, 0))
      
    Academic Reference:
      Fang, F., & Oosterlee, C. W. (2008). A novel pricing method for European options 
      based on Fourier-cosine series expansions. SIAM Journal on Scientific Computing, 
      31(2), 826-848.
      
    Parameters
    ----------
    N_cos : int
        Number of cosine terms.
    a : float
        Lower integration boundary.
    b : float
        Upper integration boundary.
        
    Returns
    -------
    coefficients : np.ndarray
        Payoff coefficients of shape (N_cos,).
    """
    k = np.arange(N_cos, dtype=np.float64)
    uk = k * np.pi / (b - a)

    with np.errstate(divide='ignore', invalid='ignore'):
        chi_put = np.real(
            np.exp(-1j * uk * a)
            * (1.0 - np.exp((1.0 + 1j * uk) * a))
            / (1.0 + 1j * uk)
        )
    chi_put[0] = 1.0 - np.exp(a)

    safe_uk = np.where(k == 0, 1.0, uk)
    psi_put = np.where(
        k == 0,
        -a,
        -np.sin(uk * a) / safe_uk,
    )

    Vk_put = (2.0 / (b - a)) * (psi_put - chi_put)
    Vk_put[0] *= 0.5
    return Vk_put


# ---------------------------------------------------------------------------
# Black-Scholes helpers (CPU)
# ---------------------------------------------------------------------------

def bs_call_cpu(
    S: float,
    K: float,
    T: float,
    sigma: float
) -> float:
    """
    Black-Scholes analytical call price.
    """
    if sigma <= 1e-10 or T <= 1e-10:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * norm.cdf(d2))


def bs_put_cpu(
    S: float,
    K: float,
    T: float,
    sigma: float
) -> float:
    """
    Black-Scholes analytical put price.
    """
    if sigma <= 1e-10 or T <= 1e-10:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_vega_cpu(
    S: float,
    K: float,
    T: float,
    sigma: float
) -> float:
    """
    Black-Scholes option vega.
    """
    if sigma <= 1e-10 or T <= 1e-10:
        return 0.0
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return float(S * np.sqrt(T) * norm.pdf(d1))


def implied_vol_cpu(
    price: float,
    S: float,
    K: float,
    T: float,
    max_iter: int = 50,
    r: float = 0.0,
    q: float = 0.0
) -> float:
    """
    Solve for Black-Scholes implied volatility on CPU.
    """
    if T <= 1e-10:
        return np.nan
    is_put = K < S * np.exp((r - q) * T)  # use forward for correct OTM selection
    if is_put:
        eff_price = price - S + K
        intrinsic = max(K - S, 0.0)
    else:
        eff_price = price
        intrinsic = max(S - K, 0.0)

    if eff_price <= intrinsic + 1e-12:
        return np.nan
    if eff_price >= (K if is_put else S):
        return np.nan

    sigma = 0.3
    for _ in range(max_iter):
        p = bs_put_cpu(S, K, T, sigma) if is_put else bs_call_cpu(S, K, T, sigma)
        v = bs_vega_cpu(S, K, T, sigma)
        if abs(v) < 1e-15:
            break
        sigma -= (p - eff_price) / v
        sigma = np.clip(sigma, 1e-6, 5.0)
        if abs(p - eff_price) < 1e-10:
            break
    return float(sigma) if 1e-6 < sigma < 5.0 else np.nan


# ---------------------------------------------------------------------------
# Black-Scholes helpers (GPU)
# ---------------------------------------------------------------------------

def _ncdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _npdf(x: torch.Tensor) -> torch.Tensor:
    return _INVSQRT2PI * torch.exp(-0.5 * x * x)


def bs_iv_gpu(
    prices: torch.Tensor,
    S0: float,
    K_arr: torch.Tensor,
    T_arr: torch.Tensor,
    n_iter: int = 40,
) -> torch.Tensor:
    """
    Solve for Black-Scholes implied volatility on GPU.
    """
    dev = prices.device
    S = torch.tensor(S0, dtype=torch.float64, device=dev)
    K = K_arr.view(1, 1, -1)
    T = T_arr.view(1, -1, 1)
    sqT = torch.sqrt(T.clamp(min=1e-10))

    itm = (K < S)
    put_prices = prices - (S - K).clamp(min=0.0)
    eff_prices = torch.where(itm, put_prices, prices)

    invalid = (eff_prices <= 1e-12) | (T < 1e-10)

    sigma = torch.full_like(prices, 0.30)

    for _ in range(n_iter):
        s = sigma.clamp(min=1e-8)
        denom_d1 = (s * sqT).clamp(min=1e-15)
        d1 = (torch.log(S / K) + 0.5 * s**2 * T) / denom_d1
        d2 = d1 - s * sqT
        call_p = S * _ncdf(d1) - K * _ncdf(d2)
        put_p = K * _ncdf(-d2) - S * _ncdf(-d1)
        model_p = torch.where(itm, put_p, call_p)
        v = S * sqT * _npdf(d1)
        sigma = (sigma - (model_p - eff_prices) / v.clamp(min=1e-15)).clamp(1e-7, 5.0)

    sigma[invalid] = float('nan')
    sigma[(sigma < 1e-5) | (sigma > 4.9)] = float('nan')

    return sigma.float()


# ---------------------------------------------------------------------------
# Heston Characteristic Function
# ---------------------------------------------------------------------------

def heston_cf(
    u: np.ndarray,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float
) -> np.ndarray:
    """
    Stable Gatheral formulation of the characteristic function for Heston model on CPU (NumPy).
    Uses log1p to avoid complex branch-cut issues.
    
    Formula:
      \\phi_t(u) = \\exp(C(u, T) + D(u, T) v_0)
      
    Academic Reference:
      Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. John Wiley & Sons.
      
    Parameters
    ----------
    u : np.ndarray
        Fourier transform variable.
    T : float
        Maturity time.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term variance.
    sigma : float
        Volatility of volatility.
    rho : float
        Correlation coefficient.
    v0 : float
        Initial variance.
        
    Returns
    -------
    cf_values : np.ndarray
        Characteristic function values.
    """
    # Guard against invalid inputs and numerical issues
    if kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0 or v0 <= 0.0:
        raise ValueError("Heston parameters kappa, theta, sigma, v0 must be strictly positive.")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("Heston correlation rho must be in [-1, 1].")

    sigma = max(sigma, 1e-8)
    v0 = max(v0, 1e-8)
    kappa = max(kappa, 1e-8)
    theta = max(theta, 1e-8)
    rho = np.clip(rho, -0.9999, 0.9999)
    
    xi = kappa - 1j * rho * sigma * u
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    
    # Avoid division by zero when xi + d is zero
    denom_g = xi + d
    denom_g = np.where(np.abs(denom_g) < 1e-12, 1e-12 + 0j, denom_g)
    g = (xi - d) / denom_g
    
    exp_mindT = np.exp(-d * T)
    denom = 1.0 - g
    denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12 + 0j, denom)
    z = g * (1.0 - exp_mindT) / denom_safe
    
    C = (kappa * theta / sigma**2) * ((xi - d) * T - 2.0 * np.log1p(z))
    
    denom_D = 1.0 - g * exp_mindT
    denom_D_safe = np.where(np.abs(denom_D) < 1e-12, 1e-12 + 0j, denom_D)
    D = ((xi - d) / sigma**2) * ((1.0 - exp_mindT) / denom_D_safe)
    
    return np.exp(C + D * v0)


# ---------------------------------------------------------------------------
# Heston Pricing Engines
# ---------------------------------------------------------------------------

def heston_iv_surface(
    params: dict,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    S0: float = 1.0,
    N_cos: int = 128
) -> np.ndarray:
    """
    Computes option implied volatility surface on CPU for a single Heston parameter set.
    
    Parameters
    ----------
    params : dict
        Dict with keys ['kappa', 'theta', 'sigma', 'rho', 'v0']
    T_grid : np.ndarray
        Maturities.
    K_grid : np.ndarray
        Log-moneyness K = log(Strike / S0).
    S0 : float
        Spot price.
    N_cos : int
        Number of terms in Fourier series.
        
    Returns
    -------
    iv_surface : np.ndarray
        Implied volatility surface.
    """
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho = params['rho']
    v0 = params['v0']
    
    if kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0 or v0 <= 0.0:
        raise ValueError("Heston parameters kappa, theta, sigma, v0 must be strictly positive.")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("Heston correlation rho must be in [-1, 1].")
    
    a, b = _A, _B
    k_arr = np.arange(N_cos)
    u_k = k_arr * np.pi / (b - a)
    
    Vk_call = cos_payoff_coeffs_np(N_cos, a, b)
    Vk_put = cos_payoff_coeffs_put_np(N_cos, a, b)
    
    iv_surface = np.full((len(T_grid), len(K_grid)), np.nan)
    
    for i, T in enumerate(T_grid):
        phi_k = heston_cf(u_k, T, kappa, theta, sigma, rho, v0)
        phi_k[0] = 1.0 + 0j  # Normalization: E[e^{i·0·X}] = 1 by definition (Martingale condition)
        
        for j, log_moneyness in enumerate(K_grid):
            K = S0 * np.exp(log_moneyness)
            x0 = np.log(S0 / K)
            
            if K < S0:
                price_put = K * np.real(np.sum(phi_k * np.exp(1j * u_k * (x0 - a)) * Vk_put))
                price_put = max(price_put, max(K - S0, 0.0))
                price = price_put + S0 - K
            else:
                price_call = K * np.real(np.sum(phi_k * np.exp(1j * u_k * (x0 - a)) * Vk_call))
                price = max(price_call, max(S0 - K, 0.0))
                
            iv_surface[i, j] = implied_vol_cpu(price, S0, K, T)
            
    return iv_surface


@lru_cache(maxsize=128)
def _get_cos_payoff_coeffs_gpu_cached(N_cos: int, a: float, b: float, device_str: str, is_put: bool = False) -> torch.Tensor:
    device = torch.device(device_str)
    k = torch.arange(N_cos, dtype=torch.float64, device=device)
    uk = k * np.pi / (b - a)
    uk_c = uk.to(torch.complex128)
    
    if is_put:
        chi_put = torch.real(
            torch.exp(-1j * uk_c * a)
            * (1.0 - torch.exp((1.0 + 1j * uk_c) * a))
            / (1.0 + 1j * uk_c)
        )
        chi_put[0] = 1.0 - np.exp(a)
        
        safe_uk = torch.where(k == 0, torch.tensor(1.0, dtype=torch.float64, device=device), uk)
        psi_put = torch.where(
            k == 0,
            torch.tensor(-a, dtype=torch.float64, device=device),
            -torch.sin(uk * a) / safe_uk,
        )
        Vk = (2.0 / (b - a)) * (psi_put - chi_put)
        Vk[0] *= 0.5
    else:
        chi = torch.real(
            torch.exp(-1j * uk_c * a)
            * (torch.exp((1.0 + 1j * uk_c) * b) - 1.0)
            / (1.0 + 1j * uk_c)
        )
        chi[0] = np.exp(b) - 1.0
        
        safe_uk = torch.where(k == 0, torch.tensor(1.0, dtype=torch.float64, device=device), uk)
        psi = torch.where(
            k == 0,
            torch.tensor(b, dtype=torch.float64, device=device),
            (torch.sin(uk * (b - a)) + torch.sin(uk * a)) / safe_uk,
        )
        Vk = (2.0 / (b - a)) * (chi - psi)
        Vk[0] *= 0.5
        
    return Vk


def get_cos_payoff_coeffs_gpu(N_cos: int, a: float, b: float, device: torch.device, is_put: bool = False) -> torch.Tensor:
    """
    Get GPU COS payoff coefficients.
    """
    return _get_cos_payoff_coeffs_gpu_cached(N_cos, a, b, str(device), is_put)


def batch_heston_iv_surface(
    params: torch.Tensor,
    T_grid: torch.Tensor,
    K_grid: torch.Tensor,
    S0: float = 1.0,
    N_cos: int = 128,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    GPU-batched vectorized version to compute Heston implied volatility surfaces.
    
    Parameters
    ----------
    params : torch.Tensor
        Tensor of shape (B, 5): [kappa, theta, sigma, rho, v0]
    T_grid : torch.Tensor
        Maturities of shape (nT,).
    K_grid : torch.Tensor
        Log-moneyness of shape (nK,).
    S0 : float
        Spot price.
    N_cos : int
        Number of terms in Fourier series.
    device : str
        Target hardware device ('cuda' or 'cpu').
        
    Returns
    -------
    ivs : torch.Tensor
        Implied volatility surfaces of shape (B, nT, nK).
    """
    device_obj = torch.device(device)
    params = params.to(device_obj)
    T_grid = torch.as_tensor(T_grid, dtype=torch.float64, device=device_obj)
    K_grid = torch.as_tensor(K_grid, dtype=torch.float64, device=device_obj)
    
    B = params.shape[0]
    
    a, b = _A, _B
    k = torch.arange(N_cos, dtype=torch.float64, device=device_obj)
    u_k = k * np.pi / (b - a)
    
    Vk_call = get_cos_payoff_coeffs_gpu(N_cos, a, b, device_obj, is_put=False)
    Vk_put = get_cos_payoff_coeffs_gpu(N_cos, a, b, device_obj, is_put=True)
    
    kappa = params[:, 0:1].clamp(min=1e-8)
    theta = params[:, 1:2].clamp(min=1e-8)
    sigma = params[:, 2:3].clamp(min=1e-8)
    rho = params[:, 3:4].clamp(-0.9999, 0.9999)
    v0 = params[:, 4:5].clamp(min=1e-8)
    
    u_c = u_k.view(1, 1, -1)
    T_c = T_grid.view(1, -1, 1)
    
    kappa_e = kappa.view(-1, 1, 1)
    theta_e = theta.view(-1, 1, 1)
    sigma_e = sigma.view(-1, 1, 1)
    rho_e = rho.view(-1, 1, 1)
    v0_e = v0.view(-1, 1, 1)
    
    xi = kappa_e - 1j * rho_e * sigma_e * u_c
    d = torch.sqrt(xi**2 + sigma_e**2 * (u_c**2 + 1j * u_c))
    
    denom_g = xi + d
    denom_g = torch.where(denom_g.abs() < 1e-12,
        torch.tensor(1e-12 + 0j, dtype=denom_g.dtype, device=denom_g.device), denom_g)
    g = (xi - d) / denom_g
    
    exp_mindT = torch.exp(-d * T_c)
    denom = 1.0 - g
    denom_safe = torch.where(denom.abs() < 1e-12,
        torch.tensor(1e-12 + 0j, dtype=denom.dtype, device=denom.device), denom)
    z = g * (1.0 - exp_mindT) / denom_safe
    log_term = torch.log1p(z)
    
    C = (kappa_e * theta_e / sigma_e**2) * ((xi - d) * T_c - 2.0 * log_term)
    
    denom_D = 1.0 - g * exp_mindT
    denom_D_safe = torch.where(denom_D.abs() < 1e-12,
        torch.tensor(1e-12 + 0j, dtype=denom_D.dtype, device=denom_D.device), denom_D)
    D = ((xi - d) / sigma_e**2) * ((1.0 - exp_mindT) / denom_D_safe)
    
    phi = torch.exp(C + D * v0_e)
    phi[:, :, 0] = 1.0 + 0.0j
    
    S0t = torch.tensor(S0, dtype=torch.float64, device=device_obj)
    K_arr = S0t * torch.exp(K_grid)
    x0 = -K_grid
    phase = torch.exp(1j * u_k.unsqueeze(1) * (x0 - a).unsqueeze(0))
    
    phi_w_call = phi * Vk_call.to(torch.complex128)
    phi_w_put = phi * Vk_put.to(torch.complex128)
    
    result_call = torch.einsum('btn,nk->btk', phi_w_call, phase)
    result_put = torch.einsum('btn,nk->btk', phi_w_put, phase)
    
    K_v = K_arr.view(1, 1, -1)
    call_prices = K_v * result_call.real
    put_prices = K_v * result_put.real
    
    call_from_put = put_prices + (S0t - K_arr).clamp(min=0.0).view(1, 1, -1)
    itm = (K_arr < S0t).view(1, 1, -1)
    prices = torch.where(itm, call_from_put, call_prices)
    
    intrinsic = (S0t - K_v).clamp(min=0.0)
    prices = torch.max(prices, intrinsic)
    
    ivs = bs_iv_gpu(prices, S0, K_arr, T_grid)
    return ivs


# ---------------------------------------------------------------------------
# Heston Calibration
# ---------------------------------------------------------------------------

def calibrate_heston(
    iv_target: np.ndarray,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    init_guess: Optional[np.ndarray] = None,
    max_iter: int = 100,
) -> dict:
    """
    Calibrate Classic Heston parameters to a market IV surface using L-BFGS-B.
    
    Parameters
    ----------
    iv_target : np.ndarray
        Target market implied volatility surface.
    T_grid : np.ndarray
        Maturities.
    K_grid : np.ndarray
        Log-strikes.
    init_guess : np.ndarray, optional
        Initial parameters guess.
    max_iter : int
        Maximum iterations.
        
    Returns
    -------
    calibration_result : dict
        Dict with keys ['params', 'param_vector', 'loss', 'converged', 'message'].
    """
    bounds = [
        (0.1, 5.0),    # kappa
        (0.01, 0.15),  # theta
        (0.1, 1.0),    # sigma
        (-0.9, -0.1),  # rho
        (0.01, 0.15),  # v0
    ]
    
    if init_guess is None:
        init_guess = np.array([1.5, 0.05, 0.3, -0.5, 0.05])
        
    def objective(x):
        kappa, theta, sigma, rho, v0 = x
        
        # Soft Feller penalty
        feller_violation = sigma**2 - 2.0 * kappa * theta
        penalty = 0.0
        if feller_violation > 0:
            penalty = 100.0 * feller_violation
            
        p_dict = {
            'kappa': kappa,
            'theta': theta,
            'sigma': sigma,
            'rho': rho,
            'v0': v0
        }
        
        iv_pred = heston_iv_surface(p_dict, T_grid, K_grid)
        
        if np.isnan(iv_pred).all():
            return 1e6
            
        mask = np.isnan(iv_pred)
        if mask.any():
            diff = np.where(mask, 1.0, iv_pred - iv_target)
        else:
            diff = iv_pred - iv_target
            
        return np.mean(diff**2) + penalty

    res = minimize(
        objective,
        init_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-6}
    )
    
    calibrated_params = {
        'kappa': float(res.x[0]),
        'theta': float(res.x[1]),
        'sigma': float(res.x[2]),
        'rho': float(res.x[3]),
        'v0': float(res.x[4]),
    }
    
    return {
        'params': calibrated_params,
        'param_vector': res.x,
        'loss': float(res.fun),
        'converged': bool(res.success),
        'message': str(res.message),
    }


class HestonEngine:
    """
    Heston Engine wrapper class.
    """
    def price_surface(
        self,
        params: dict,
        T_grid: np.ndarray,
        K_grid: np.ndarray,
        S0: float = 1.0,
        N_cos: int = 128
    ) -> np.ndarray:
        """
        Price single Heston implied volatility surface.
        """
        return heston_iv_surface(params, T_grid, K_grid, S0, N_cos)
        
    def batch_price_surface(
        self,
        params: torch.Tensor,
        T_grid: torch.Tensor,
        K_grid: torch.Tensor,
        S0: float = 1.0,
        N_cos: int = 128,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Price batched Heston implied volatility surfaces on CPU or GPU.
        """
        return batch_heston_iv_surface(params, T_grid, K_grid, S0, N_cos, device)
        
    def calibrate(
        self,
        iv_target: np.ndarray,
        T_grid: np.ndarray,
        K_grid: np.ndarray,
        init_guess: Optional[np.ndarray] = None,
        max_iter: int = 100
    ) -> dict:
        """
        Calibrate Heston model parameters to target surface.
        """
        return calibrate_heston(iv_target, T_grid, K_grid, init_guess, max_iter)
