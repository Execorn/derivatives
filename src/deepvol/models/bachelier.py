"""
bachelier.py — Analytical option pricing and implied volatility solvers.
Contains normal (Bachelier), lognormal (Black), and shifted Black pricing
along with root-finding implied volatility solvers.
"""

from typing import Union
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt

def bachelier_price(
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    option_type: str = 'call'
) -> Union[float, np.ndarray]:
    """
    Price a normal (Bachelier) option using the Bachelier (1900) model.
    
    The Bachelier model assumes the underlying asset follows a normal distribution
    without mean reversion (Arithmetic Brownian Motion):
    
        dF_t = \\sigma dW_t
        
    Formula:
      d = \\frac{F - K}{\\sigma \\sqrt{T}}
      C = (F - K) \\Phi(d) + \\sigma \\sqrt{T} \\phi(d)
      P = (K - F) \\Phi(-d) + \\sigma \\sqrt{T} \\phi(d)
      
    where \\Phi is the standard normal cumulative distribution function (CDF)
    and \\phi is the standard normal probability density function (PDF).
    
    Academic Reference:
      Bachelier, L. (1900). Théorie de la spéculation. Annales Scientifiques de 
      l'École Normale Supérieure, 17, 21-86.
      
    Parameters
    ----------
    F : float or ndarray
        Forward price/rate.
    K : float or ndarray
        Strike price/rate.
    T : float or ndarray
        Time to maturity.
    sigma : float or ndarray
        Normal volatility (absolute vol).
    option_type : str
        'call' or 'put' (case-insensitive).
        
    Returns
    -------
    price : float or ndarray
        Bachelier option price.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Broadcast to same shape
    F, K, T, sigma = np.broadcast_arrays(F, K, T, sigma)
    shape = F.shape
    
    price = np.zeros(shape)
    
    # Valid condition: T > 1e-8 and sigma > 1e-8
    valid = (T > 1e-8) & (sigma > 1e-8)
    
    # Error cases: T < 0 or sigma < 0 should return NaN
    nan_mask = (T < 0.0) | (sigma < 0.0)
    price[nan_mask] = np.nan
    
    # Zero/boundary cases: T <= 1e-8 or sigma <= 1e-8 (intrinsic value)
    zero_mask = ~nan_mask & (~valid)
    if np.any(zero_mask):
        if option_type.lower() in ['call', 'c']:
            price[zero_mask] = np.maximum(F[zero_mask] - K[zero_mask], 0.0)
        elif option_type.lower() in ['put', 'p']:
            price[zero_mask] = np.maximum(K[zero_mask] - F[zero_mask], 0.0)
        else:
            raise ValueError(f"Unknown option_type: {option_type}")
            
    if np.any(valid):
        F_v = F[valid]
        K_v = K[valid]
        T_v = T[valid]
        sigma_v = sigma[valid]
        
        vol_sqrt_T = np.maximum(sigma_v * np.sqrt(T_v), 1e-15)
        d = (F_v - K_v) / vol_sqrt_T
        
        N_d = stats.norm.cdf(d)
        n_d = stats.norm.pdf(d)
        
        if option_type.lower() in ['call', 'c']:
            price[valid] = (F_v - K_v) * N_d + vol_sqrt_T * n_d
        elif option_type.lower() in ['put', 'p']:
            price[valid] = (K_v - F_v) * stats.norm.cdf(-d) + vol_sqrt_T * n_d
        else:
            raise ValueError(f"Unknown option_type: {option_type}")
            
    if len(shape) == 0:
        return float(price)
    return price


def black_price(
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    option_type: str = 'call'
) -> Union[float, np.ndarray]:
    """
    Price a lognormal (Black) option using the Black (1976) model.
    
    The Black model assumes the forward price of the underlying follows a Geometric
    Brownian Motion under the forward measure:
    
        dF_t = \\sigma F_t dW_t
        
    Formula:
      d1 = \\frac{\\ln(F/K) + 0.5 \\sigma^2 T}{\\sigma \\sqrt{T}}
      d2 = d1 - \\sigma \\sqrt{T}
      C = F \\Phi(d1) - K \\Phi(d2)
      P = K \\Phi(-d2) - F \\Phi(-d1)
      
    Academic Reference:
      Black, F. (1976). The pricing of commodity contracts. Journal of Financial 
      Economics, 3(1-2), 167-179.
      
    Parameters
    ----------
    F : float or ndarray
        Forward price/rate. Must be > 0.
    K : float or ndarray
        Strike price/rate. Must be > 0.
    T : float or ndarray
        Time to maturity. Must be >= 0.
    sigma : float or ndarray
        Lognormal volatility (percentage vol). Must be >= 0.
    option_type : str
        'call' or 'put' (case-insensitive).
        
    Returns
    -------
    price : float or ndarray
        Black option price.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Broadcast to same shape
    F, K, T, sigma = np.broadcast_arrays(F, K, T, sigma)
    shape = F.shape
    
    price = np.zeros(shape)
    
    # Valid condition for Black pricing
    valid = (F > 1e-15) & (K > 1e-15) & (T > 1e-8) & (sigma > 1e-8)
    
    # Error cases: F <= 0, K <= 0, T < 0, or sigma < 0 should return NaN
    nan_mask = (F <= 0.0) | (K <= 0.0) | (T < 0.0) | (sigma < 0.0)
    price[nan_mask] = np.nan
    
    # Zero/boundary cases: T <= 1e-8 or sigma <= 1e-8 (intrinsic value)
    zero_mask = ~nan_mask & (~valid)
    if np.any(zero_mask):
        if option_type.lower() in ['call', 'c']:
            price[zero_mask] = np.maximum(F[zero_mask] - K[zero_mask], 0.0)
        elif option_type.lower() in ['put', 'p']:
            price[zero_mask] = np.maximum(K[zero_mask] - F[zero_mask], 0.0)
        else:
            raise ValueError(f"Unknown option_type: {option_type}")
            
    if np.any(valid):
        F_v = F[valid]
        K_v = K[valid]
        T_v = T[valid]
        sigma_v = sigma[valid]
        
        vol_sqrt_T = np.maximum(sigma_v * np.sqrt(T_v), 1e-15)
        # Avoid log of negative or division by zero, though F_v and K_v are > 1e-15
        F_v = np.maximum(F_v, 1e-15)
        K_v = np.maximum(K_v, 1e-15)
        d1 = (np.log(F_v / K_v) + 0.5 * (sigma_v ** 2) * T_v) / vol_sqrt_T
        d2 = d1 - vol_sqrt_T
        
        if option_type.lower() in ['call', 'c']:
            price[valid] = F_v * stats.norm.cdf(d1) - K_v * stats.norm.cdf(d2)
        elif option_type.lower() in ['put', 'p']:
            price[valid] = K_v * stats.norm.cdf(-d2) - F_v * stats.norm.cdf(-d1)
        else:
            raise ValueError(f"Unknown option_type: {option_type}")
            
    if len(shape) == 0:
        return float(price)
    return price


def shifted_black_price(
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    shift: Union[float, np.ndarray],
    option_type: str = 'call'
) -> Union[float, np.ndarray]:
    """
    Price a shifted (displaced) Black option using the displaced diffusion model.
    
    Displaced lognormal pricing, where both forward and strike are shifted by a shift parameter.
    
    Formula:
      F_shifted = F + shift
      K_shifted = K + shift
      Price = BlackPrice(F_shifted, K_shifted, T, sigma, option_type)
      
    Academic Reference:
      Rubinstein, M. (1983). Displaced Diffusion Option Pricing. The Journal of 
      Finance, 38(1), 213-217.
    
    Parameters
    ----------
    F : float or ndarray
        Forward price/rate.
    K : float or ndarray
        Strike price/rate.
    T : float or ndarray
        Time to maturity.
    sigma : float or ndarray
        Shifted lognormal volatility.
    shift : float or ndarray
        Displacement/shift parameter (theta).
    option_type : str
        'call' or 'put' (case-insensitive).
        
    Returns
    -------
    price : float or ndarray
        Shifted Black option price.
    """
    F_shifted = np.asarray(F) + shift
    K_shifted = np.asarray(K) + shift
    return black_price(F_shifted, K_shifted, T, sigma, option_type=option_type)


def bachelier_implied_vol(
    price: Union[float, np.ndarray],
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    option_type: str = 'call',
    tol: float = 1e-8,
    max_iter: int = 100
) -> Union[float, np.ndarray]:
    """
    Solve for normal (Bachelier) implied volatility from option price.
    Uses Halley's cubic convergence method with Brent's method fallback.
    
    Parameters
    ----------
    price : float or ndarray
        Option price.
    F : float or ndarray
        Forward price/rate.
    K : float or ndarray
        Strike price/rate.
    T : float or ndarray
        Time to maturity.
    option_type : str
        'call' or 'put' (case-insensitive).
    tol : float
        Root-finding tolerance.
    max_iter : int
        Maximum number of iterations.
        
    Returns
    -------
    iv : float or ndarray
        Bachelier implied volatility (normal vol). Returns NaN if solver fails or price is out of bounds.
    """
    price = np.asarray(price, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    
    # Broadcast to same shape
    price, F, K, T = np.broadcast_arrays(price, F, K, T)
    shape = price.shape
    
    price_flat = price.ravel()
    F_flat = F.ravel()
    K_flat = K.ravel()
    T_flat = T.ravel()
    
    iv_flat = np.full_like(price_flat, np.nan)
    
    for i in range(len(price_flat)):
        p = price_flat[i]
        f = F_flat[i]
        k = K_flat[i]
        t = T_flat[i]
        
        if np.isnan(p) or np.isnan(f) or np.isnan(k) or np.isnan(t) or t <= 1e-8:
            continue
            
        # Intrinsic value
        intrinsic = max(f - k, 0.0) if option_type.lower() in ['call', 'c'] else max(k - f, 0.0)
        
        if p < intrinsic - 1e-12:
            iv_flat[i] = np.nan
            continue
        elif p <= intrinsic + 1e-12:
            iv_flat[i] = 0.0
            continue
            
        # Initial guess: ATM normal volatility formula
        sigma_n = p * np.sqrt(2.0 * np.pi / t)
        if sigma_n <= 1e-15:
            sigma_n = 1e-4
            
        sqrt_t = np.sqrt(t)
        converged = False
        
        # Halley's method loop
        for _ in range(30):
            if sigma_n <= 1e-15 or np.isnan(sigma_n) or np.isinf(sigma_n):
                break
                
            vol_sqrt_t = max(sigma_n * sqrt_t, 1e-15)
            d = (f - k) / vol_sqrt_t
            n_d = np.exp(-0.5 * d * d) / np.sqrt(2.0 * np.pi)
            N_d = stats.norm.cdf(d)
            
            if option_type.lower() in ['call', 'c']:
                p_model = (f - k) * N_d + sigma_n * sqrt_t * n_d
            else:
                p_model = (k - f) * stats.norm.cdf(-d) + sigma_n * sqrt_t * n_d
                
            diff = p_model - p
            if abs(diff) < tol:
                iv_flat[i] = sigma_n
                converged = True
                break
                
            vega = sqrt_t * n_d
            if vega < 1e-15:
                break
                
            vomma = vega * d * d / max(sigma_n, 1e-15)
            denom = 2.0 * vega * vega - diff * vomma
            if abs(denom) < 1e-30:
                break
                
            step = (2.0 * diff * vega) / denom
            sigma_next = sigma_n - step
            
            if abs(step) < tol:
                if sigma_next > 1e-15 and not np.isnan(sigma_next) and not np.isinf(sigma_next):
                    vol_sqrt_t_next = max(sigma_next * sqrt_t, 1e-15)
                    d_next = (f - k) / vol_sqrt_t_next
                    n_d_next = np.exp(-0.5 * d_next * d_next) / np.sqrt(2.0 * np.pi)
                    if option_type.lower() in ['call', 'c']:
                        p_next = (f - k) * stats.norm.cdf(d_next) + sigma_next * sqrt_t * n_d_next
                    else:
                        p_next = (k - f) * stats.norm.cdf(-d_next) + sigma_next * sqrt_t * n_d_next
                    if abs(p_next - p) < tol:
                        sigma_n = sigma_next
                        iv_flat[i] = sigma_n
                        converged = True
                        break
                break
                
            sigma_n = sigma_next
            
        if not converged:
            # Fallback to Brent's method
            def obj(sigma):
                return bachelier_price(f, k, t, sigma, option_type=option_type) - p
            high = 0.1
            try:
                iter_limit = 0
                while obj(high) < 0.0 and high < 1000.0 and iter_limit < 20:
                    high *= 2.0
                    iter_limit += 1
                iv_flat[i] = opt.brentq(obj, 1e-15, high, xtol=tol, maxiter=max_iter)
            except ValueError:
                iv_flat[i] = np.nan
                
    if len(shape) == 0:
        return float(iv_flat[0])
    return iv_flat.reshape(shape)


def black_implied_vol(
    price: Union[float, np.ndarray],
    F: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    option_type: str = 'call',
    shift: Union[float, np.ndarray] = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100
) -> Union[float, np.ndarray]:
    """
    Solve for lognormal (Black) implied volatility from option price, with optional shift.
    Uses Halley's cubic convergence method with Brent's method fallback.
    
    Parameters
    ----------
    price : float or ndarray
        Option price.
    F : float or ndarray
        Forward price/rate.
    K : float or ndarray
        Strike price/rate.
    T : float or ndarray
        Time to maturity.
    option_type : str
        'call' or 'put' (case-insensitive).
    shift : float or ndarray
        Shift/displacement parameter.
    tol : float
        Root-finding tolerance.
    max_iter : int
        Maximum number of iterations.
        
    Returns
    -------
    iv : float or ndarray
        Black implied volatility (lognormal or shifted lognormal vol).
        Returns NaN if solver fails or price is out of bounds.
    """
    price = np.asarray(price, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    shift = np.asarray(shift, dtype=float)
    
    # Broadcast to same shape
    price, F, K, T, shift = np.broadcast_arrays(price, F, K, T, shift)
    shape = price.shape
    
    price_flat = price.ravel()
    F_flat = F.ravel()
    K_flat = K.ravel()
    T_flat = T.ravel()
    shift_flat = shift.ravel()
    
    iv_flat = np.full_like(price_flat, np.nan)
    
    for i in range(len(price_flat)):
        p = price_flat[i]
        f = F_flat[i]
        k = K_flat[i]
        t = T_flat[i]
        sh = shift_flat[i]
        
        f_s = f + sh
        k_s = k + sh
        
        if np.isnan(p) or np.isnan(f_s) or np.isnan(k_s) or np.isnan(t) or t <= 1e-8 or f_s <= 1e-15 or k_s <= 1e-15:
            continue
            
        # Intrinsic value
        intrinsic = max(f_s - k_s, 0.0) if option_type.lower() in ['call', 'c'] else max(k_s - f_s, 0.0)
        
        # Max possible Black option price is the shifted forward (call) or shifted strike (put)
        max_p = f_s if option_type.lower() in ['call', 'c'] else k_s
        
        if p < intrinsic - 1e-12:
            iv_flat[i] = np.nan
            continue
        elif p <= intrinsic + 1e-12:
            iv_flat[i] = 0.0
            continue
        if p >= max_p - 1e-12:
            iv_flat[i] = np.nan
            continue
            
        # Initial guess: ATM Black volatility formula
        denom_init = max(f_s * np.sqrt(t), 1e-15)
        sigma_n = p / denom_init * np.sqrt(2.0 * np.pi)
        if sigma_n <= 1e-15:
            sigma_n = 1e-4
            
        sqrt_t = np.sqrt(t)
        converged = False
        
        # Halley's method loop
        for _ in range(30):
            if sigma_n <= 1e-15 or np.isnan(sigma_n) or np.isinf(sigma_n):
                break
                
            vol_sqrt_t = max(sigma_n * sqrt_t, 1e-15)
            # Avoid division by zero and log of zero
            f_s = max(f_s, 1e-15)
            k_s = max(k_s, 1e-15)
            d1 = (np.log(f_s / k_s) + 0.5 * sigma_n * sigma_n * t) / vol_sqrt_t
            d2 = d1 - vol_sqrt_t
            
            n_d1 = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
            
            if option_type.lower() in ['call', 'c']:
                p_model = f_s * stats.norm.cdf(d1) - k_s * stats.norm.cdf(d2)
            else:
                p_model = k_s * stats.norm.cdf(-d2) - f_s * stats.norm.cdf(-d1)
                
            diff = p_model - p
            if abs(diff) < tol:
                iv_flat[i] = sigma_n
                converged = True
                break
                
            vega = f_s * sqrt_t * n_d1
            if vega < 1e-15:
                break
                
            vomma = vega * d1 * d2 / max(sigma_n, 1e-15)
            denom = 2.0 * vega * vega - diff * vomma
            if abs(denom) < 1e-30:
                break
                
            step = (2.0 * diff * vega) / denom
            sigma_next = sigma_n - step
            
            if abs(step) < tol:
                if sigma_next > 1e-15 and not np.isnan(sigma_next) and not np.isinf(sigma_next):
                    vol_sqrt_t_next = max(sigma_next * sqrt_t, 1e-15)
                    d1_next = (np.log(f_s / k_s) + 0.5 * sigma_next * sigma_next * t) / vol_sqrt_t_next
                    d2_next = d1_next - vol_sqrt_t_next
                    if option_type.lower() in ['call', 'c']:
                        p_next = f_s * stats.norm.cdf(d1_next) - k_s * stats.norm.cdf(d2_next)
                    else:
                        p_next = k_s * stats.norm.cdf(-d2_next) - f_s * stats.norm.cdf(-d1_next)
                    if abs(p_next - p) < tol:
                        sigma_n = sigma_next
                        iv_flat[i] = sigma_n
                        converged = True
                        break
                break
                
            sigma_n = sigma_next
            
        if not converged:
            # Fallback to Brent's method
            def obj(sigma):
                return black_price(f_s, k_s, t, sigma, option_type=option_type) - p
            high = 0.5
            try:
                iter_limit = 0
                while obj(high) < 0.0 and high < 100.0 and iter_limit < 20:
                    high *= 2.0
                    iter_limit += 1
                iv_flat[i] = opt.brentq(obj, 1e-15, high, xtol=tol, maxiter=max_iter)
            except ValueError:
                iv_flat[i] = np.nan
                
    if len(shape) == 0:
        return float(iv_flat[0])
    return iv_flat.reshape(shape)
