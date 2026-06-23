"""
Schwartz-Smith (2000) two-factor commodity pricing model.

Implements:
  1. Futures pricing under the risk-neutral measure Q.
  2. Analytical option pricing on futures using Black-76 with the correct conditional variance.
  3. Characteristic function of the log futures price under Q.
  4. Option pricing via Fourier inversion (Lewis integration method) on CPU and PyTorch/GPU.
  5. PyTorch/CUDA batch pricing and parallel sensitivity (Greeks) calculations.
  6. Kalman Filter calibrator maximizing the log-likelihood of innovations.
"""

from __future__ import annotations

import datetime
import numpy as np
import scipy.integrate
import scipy.optimize
from scipy.stats import norm
import torch

# ---------------------------------------------------------------------------
# 1. CPU (NumPy / SciPy) Implementation
# ---------------------------------------------------------------------------

def A_factor(
    tau: float | np.ndarray,
    kappa: float,
    sigma_chi: float,
    rho: float,
    sigma_xi: float,
    mu_star: float,
    lambda_chi: float = 0.0
) -> float | np.ndarray:
    """
    Computes the A(tau) term in the Schwartz-Smith futures pricing formula.
    
    A(tau) = (mu* + 0.5 * sigma_xi^2) * tau 
             + (sigma_chi^2 / (4 * kappa)) * (1 - e^{-2*kappa*tau}) 
             + (rho * sigma_chi * sigma_xi / kappa) * (1 - e^{-kappa*tau})
             - (lambda_chi / kappa) * (1 - e^{-kappa*tau})
    """
    term1 = (mu_star + 0.5 * sigma_xi**2) * tau
    if kappa < 1e-5:
        term2_coeff = 0.5 * tau - 0.5 * kappa * tau**2 + (1.0 / 3.0) * (kappa**2) * (tau**3)
        term3_coeff = tau - 0.5 * kappa * tau**2 + (1.0 / 6.0) * (kappa**2) * (tau**3)
    else:
        term2_coeff = (1.0 - np.exp(-2.0 * kappa * tau)) / (4.0 * kappa)
        term3_coeff = (1.0 - np.exp(-kappa * tau)) / kappa
        
    term2 = sigma_chi**2 * term2_coeff
    term3 = rho * sigma_chi * sigma_xi * term3_coeff
    term_risk = lambda_chi * term3_coeff
    
    return term1 + term2 + term3 - term_risk


def conditional_variance(
    t: float | np.ndarray,
    T_opt: float | np.ndarray,
    T_fut: float | np.ndarray,
    kappa: float,
    sigma_chi: float,
    rho: float,
    sigma_xi: float
) -> float | np.ndarray:
    """
    Computes the conditional variance of the log futures price ln F(T_opt, T_fut) at time t.
    
    v^2(t, T_opt, T_fut) = e^{-2*kappa*(T_fut - T_opt)} * (sigma_chi^2 / (2*kappa)) * (1 - e^{-2*kappa*(T_opt - t)})
                           + sigma_xi^2 * (T_opt - t)
                           + 2 * e^{-kappa*(T_fut - T_opt)} * (rho * sigma_chi * sigma_xi / kappa) * (1 - e^{-kappa*(T_opt - t)})
    """
    tau_opt = T_opt - t
    tau_diff = T_fut - T_opt
    
    if kappa < 1e-5:
        factor1 = tau_opt - kappa * tau_opt**2 + (2.0 / 3.0) * (kappa**2) * (tau_opt**3)
        factor3 = tau_opt - 0.5 * kappa * tau_opt**2 + (1.0 / 6.0) * (kappa**2) * (tau_opt**3)
    else:
        factor1 = (1.0 - np.exp(-2.0 * kappa * tau_opt)) / (2.0 * kappa)
        factor3 = (1.0 - np.exp(-kappa * tau_opt)) / kappa
        
    term1 = np.exp(-2.0 * kappa * tau_diff) * sigma_chi**2 * factor1
    term2 = sigma_xi**2 * tau_opt
    term3 = 2.0 * np.exp(-kappa * tau_diff) * rho * sigma_chi * sigma_xi * factor3
    return term1 + term2 + term3


def futures_price(
    t: float | np.ndarray,
    T_fut: float | np.ndarray,
    chi_t: float | np.ndarray,
    xi_t: float | np.ndarray,
    kappa: float,
    sigma_chi: float,
    rho: float,
    sigma_xi: float,
    mu_star: float,
    lambda_chi: float = 0.0
) -> float | np.ndarray:
    """
    Computes the futures price F(t, T_fut) at time t.
    
    F(t, T_fut) = exp( e^{-kappa*(T_fut - t)} * chi_t + xi_t + A(T_fut - t) )
    """
    tau = T_fut - t
    log_F = np.exp(-kappa * tau) * chi_t + xi_t + A_factor(tau, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    return np.exp(log_F)


def schwartz_smith_cf(
    u: complex | np.ndarray,
    t: float,
    T_opt: float,
    T_fut: float,
    chi_t: float,
    xi_t: float,
    kappa: float,
    sigma_chi: float,
    rho: float,
    sigma_xi: float,
    mu_star: float,
    lambda_chi: float = 0.0
) -> complex | np.ndarray:
    """
    Computes the characteristic function of the log futures price ln F(T_opt, T_fut) under Q.
    
    phi(u) = E_t^Q[ exp(i * u * ln F(T_opt, T_fut)) ]
           = exp( i * u * m - 0.5 * u^2 * v^2 )
    where:
      v^2 = conditional_variance(t, T_opt, T_fut)
      m = ln F(t, T_fut) - 0.5 * v^2
    """
    F_t = futures_price(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    v2 = conditional_variance(t, T_opt, T_fut, kappa, sigma_chi, rho, sigma_xi)
    m = np.log(F_t) - 0.5 * v2
    return np.exp(1j * u * m - 0.5 * (u**2) * v2)


def schwartz_smith_price_black76(
    t: float,
    T_opt: float,
    T_fut: float,
    K: float,
    r: float,
    chi_t: float,
    xi_t: float,
    kappa: float,
    sigma_chi: float,
    rho: float,
    sigma_xi: float,
    mu_star: float,
    lambda_chi: float = 0.0,
    option_type: str = "C"
) -> float:
    """
    Analytical European option pricing on futures using Black-76 with the correct conditional variance.
    """
    if K <= 0.0:
        raise ValueError("Strike must be positive")
    if r < 0.0:
        raise ValueError("Risk free rate must be non-negative")
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    if sigma_chi < 0.0:
        raise ValueError("sigma_chi must be non-negative")
    if sigma_xi < 0.0:
        raise ValueError("sigma_xi must be non-negative")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be between -1.0 and 1.0")
    if T_opt > T_fut:
        raise ValueError("Option maturity cannot exceed futures maturity")

    tau = T_opt - t
    if tau <= 0:
        F = futures_price(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
        return max(F - K, 0.0) if option_type == "C" else max(K - F, 0.0)
        
    F = futures_price(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    v2 = conditional_variance(t, T_opt, T_fut, kappa, sigma_chi, rho, sigma_xi)
    v = np.sqrt(v2)
    
    if v < 1e-10:
        return np.exp(-r * tau) * (max(F - K, 0.0) if option_type == "C" else max(K - F, 0.0))
        
    d1 = (np.log(F / K) + 0.5 * v2) / v
    d2 = d1 - v
    
    if option_type == "C":
        price = np.exp(-r * tau) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = np.exp(-r * tau) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return float(price)


def schwartz_smith_price_fourier(
    t: float,
    T_opt: float,
    T_fut: float,
    K: float,
    r: float,
    chi_t: float,
    xi_t: float,
    kappa: float,
    sigma_chi: float,
    rho: float,
    sigma_xi: float,
    mu_star: float,
    lambda_chi: float = 0.0,
    option_type: str = "C",
    limit: int = 100
) -> float:
    """
    Option pricing via Fourier inversion (Lewis 2001 method) on CPU.
    """
    if K <= 0.0:
        raise ValueError("Strike must be positive")
    if r < 0.0:
        raise ValueError("Risk free rate must be non-negative")
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    if sigma_chi < 0.0:
        raise ValueError("sigma_chi must be non-negative")
    if sigma_xi < 0.0:
        raise ValueError("sigma_xi must be non-negative")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be between -1.0 and 1.0")
    if T_opt > T_fut:
        raise ValueError("Option maturity cannot exceed futures maturity")

    F = futures_price(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    tau = T_opt - t
    
    if tau <= 0:
        return max(F - K, 0.0) if option_type == "C" else max(K - F, 0.0)
        
    v2 = conditional_variance(t, T_opt, T_fut, kappa, sigma_chi, rho, sigma_xi)
    if v2 < 1e-15:
        return float(np.exp(-r * tau) * (max(F - K, 0.0) if option_type == "C" else max(K - F, 0.0)))
        
    # Integrand for Lewis formula:
    # Re[ e^{-i * u * ln K} * phi(u - i/2) / (u^2 + 1/4) ]
    def integrand(u):
        cf_val = schwartz_smith_cf(u - 0.5j, t, T_opt, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
        val = np.exp(-1j * u * np.log(K)) * cf_val / (u**2 + 0.25)
        return np.real(val)
        
    integral, _ = scipy.integrate.quad(integrand, 0.0, np.inf, limit=limit)
    call_price = np.exp(-r * tau) * (F - (np.sqrt(K) / np.pi) * integral)
    
    if option_type == "C":
        return max(call_price, 0.0)
    else:
        # Put-call parity: C - P = e^{-r*tau} * (F - K)
        put_price = call_price - np.exp(-r * tau) * (F - K)
        return max(put_price, 0.0)


def _cos_payoff_call(a: float, b: float, strike: float, N: int) -> np.ndarray:
    k = np.arange(N)
    c = np.maximum(a, 0.0)
    d = np.maximum(b, 0.0)
    
    # Compute psi_k(c, d)
    psi = np.zeros(N)
    psi[0] = d - c
    if N > 1:
        psi[1:] = (b - a) / (k[1:] * np.pi) * (
            np.sin(k[1:] * np.pi * (d - a) / (b - a)) - np.sin(k[1:] * np.pi * (c - a) / (b - a))
        )
    
    # Compute chi_k(c, d)
    denom = 1.0 + (k * np.pi / (b - a)) ** 2
    cos_d = np.cos(k * np.pi * (d - a) / (b - a))
    sin_d = np.sin(k * np.pi * (d - a) / (b - a))
    cos_c = np.cos(k * np.pi * (c - a) / (b - a))
    sin_c = np.sin(k * np.pi * (c - a) / (b - a))
    
    chi = (1.0 / denom) * (
        cos_d * np.exp(d) - cos_c * np.exp(c) 
        + (k * np.pi / (b - a)) * sin_d * np.exp(d) 
        - (k * np.pi / (b - a)) * sin_c * np.exp(c)
    )
    
    return (2.0 / (b - a)) * strike * (chi - psi)


def _cos_payoff_put(a: float, b: float, strike: float, N: int) -> np.ndarray:
    k = np.arange(N)
    c = np.minimum(a, 0.0)
    d = np.minimum(b, 0.0)
    
    # Compute psi_k(c, d)
    psi = np.zeros(N)
    psi[0] = d - c
    if N > 1:
        psi[1:] = (b - a) / (k[1:] * np.pi) * (
            np.sin(k[1:] * np.pi * (d - a) / (b - a)) - np.sin(k[1:] * np.pi * (c - a) / (b - a))
        )
    
    # Compute chi_k(c, d)
    denom = 1.0 + (k * np.pi / (b - a)) ** 2
    cos_d = np.cos(k * np.pi * (d - a) / (b - a))
    sin_d = np.sin(k * np.pi * (d - a) / (b - a))
    cos_c = np.cos(k * np.pi * (c - a) / (b - a))
    sin_c = np.sin(k * np.pi * (c - a) / (b - a))
    
    chi = (1.0 / denom) * (
        cos_d * np.exp(d) - cos_c * np.exp(c) 
        + (k * np.pi / (b - a)) * sin_d * np.exp(d) 
        - (k * np.pi / (b - a)) * sin_c * np.exp(c)
    )
    
    return (2.0 / (b - a)) * strike * (-chi + psi)


def price_option_cos(
    t: float,
    T_opt: float,
    T_fut: float,
    K: float,
    r: float,
    chi_t: float,
    xi_t: float,
    kappa: float,
    sigma_chi: float,
    rho: float,
    sigma_xi: float,
    mu_star: float,
    lambda_chi: float = 0.0,
    option_type: str = "C",
    N: int = 128,
    L: float = 10.0
) -> float:
    """
    European option pricing on futures using the Fourier-Cosine (COS) method
    under the Schwartz-Smith two-factor model.
    """
    if not (np.isfinite(t) and np.isfinite(T_opt) and np.isfinite(T_fut) and np.isfinite(K) and np.isfinite(r)
            and np.isfinite(chi_t) and np.isfinite(xi_t) and np.isfinite(kappa)
            and np.isfinite(sigma_chi) and np.isfinite(rho) and np.isfinite(sigma_xi)
            and np.isfinite(mu_star) and np.isfinite(lambda_chi)):
        raise ValueError("All inputs must be finite")
    if K <= 0.0:
        raise ValueError("Strike must be positive")
    if r < 0.0:
        raise ValueError("Risk free rate must be non-negative")
        
    tau = T_opt - t
    if tau <= 0.0:
        F = futures_price(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
        return max(F - K, 0.0) if option_type == "C" else max(K - F, 0.0)
        
    v2 = conditional_variance(t, T_opt, T_fut, kappa, sigma_chi, rho, sigma_xi)
    if v2 < 1e-15:
        F = futures_price(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
        return float(np.exp(-r * tau) * (max(F - K, 0.0) if option_type == "C" else max(K - F, 0.0)))
        
    F = futures_price(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    
    # Truncation limits using cumulants c_1 and c_2
    c1 = np.log(F / K) - 0.5 * v2
    c2 = v2
    
    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)
    
    # Grid parameters
    k = np.arange(N)
    u = k * np.pi / (b - a)
    
    # Characteristic function of log-ratio y = ln(F_T/K)
    phi_y = np.exp(1j * u * c1 - 0.5 * (u ** 2) * v2)
    
    # Compute payoff coefficients
    if option_type == "C":
        V = _cos_payoff_call(a, b, K, N)
    else:
        V = _cos_payoff_put(a, b, K, N)
        
    # Reconstruct pricing formula
    terms = np.real(phi_y * np.exp(-1j * u * a)) * V
    terms[0] *= 0.5 # multiply k=0 term by 0.5
    
    price = np.exp(-r * tau) * np.sum(terms)
    return max(0.0, float(price))


def schwartz_smith_price_cos(
    t: float,
    T_opt: float,
    T_fut: float,
    K: float,
    r: float,
    chi_t: float,
    xi_t: float,
    kappa: float,
    sigma_chi: float,
    rho: float,
    sigma_xi: float,
    mu_star: float,
    lambda_chi: float = 0.0,
    option_type: str = "C",
    N: int = 128,
    L: float = 10.0
) -> float:
    """
    European option pricing on futures using the Fourier-Cosine (COS) method
    under the Schwartz-Smith two-factor model (CPU implementation).
    Enforces constraints on inputs (e.g. non-negative inputs, positive strikes/spots).
    """
    if not (np.isfinite(t) and np.isfinite(T_opt) and np.isfinite(T_fut) and np.isfinite(K) and np.isfinite(r)
            and np.isfinite(chi_t) and np.isfinite(xi_t) and np.isfinite(kappa)
            and np.isfinite(sigma_chi) and np.isfinite(rho) and np.isfinite(sigma_xi)
            and np.isfinite(mu_star) and np.isfinite(lambda_chi)):
        raise ValueError("All inputs must be finite")
    if K <= 0.0:
        raise ValueError("Strike must be positive")
    if r < 0.0:
        raise ValueError("Risk free rate must be non-negative")
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    if sigma_chi < 0.0:
        raise ValueError("sigma_chi must be non-negative")
    if sigma_xi < 0.0:
        raise ValueError("sigma_xi must be non-negative")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be between -1.0 and 1.0")
    if T_opt > T_fut:
        raise ValueError("Option maturity cannot exceed futures maturity")
    if N <= 0:
        raise ValueError("N must be positive")
    if L <= 0.0:
        raise ValueError("L must be positive")
        
    return price_option_cos(
        t=t, T_opt=T_opt, T_fut=T_fut, K=K, r=r, chi_t=chi_t, xi_t=xi_t,
        kappa=kappa, sigma_chi=sigma_chi, rho=rho, sigma_xi=sigma_xi,
        mu_star=mu_star, lambda_chi=lambda_chi, option_type=option_type,
        N=N, L=L
    )


# ---------------------------------------------------------------------------
# 2. PyTorch / CUDA GPU Implementation
# ---------------------------------------------------------------------------

def A_factor_pt(
    tau: torch.Tensor,
    kappa: torch.Tensor | float,
    sigma_chi: torch.Tensor | float,
    rho: torch.Tensor | float,
    sigma_xi: torch.Tensor | float,
    mu_star: torch.Tensor | float,
    lambda_chi: torch.Tensor | float = 0.0
) -> torch.Tensor:
    """PyTorch version of A_factor."""
    kappa_t = torch.as_tensor(kappa, device=tau.device, dtype=tau.dtype)
    sigma_chi_t = torch.as_tensor(sigma_chi, device=tau.device, dtype=tau.dtype)
    rho_t = torch.as_tensor(rho, device=tau.device, dtype=tau.dtype)
    sigma_xi_t = torch.as_tensor(sigma_xi, device=tau.device, dtype=tau.dtype)
    mu_star_t = torch.as_tensor(mu_star, device=tau.device, dtype=tau.dtype)
    lambda_chi_t = torch.as_tensor(lambda_chi, device=tau.device, dtype=tau.dtype)
    
    term1 = (mu_star_t + 0.5 * sigma_xi_t**2) * tau
    
    term2_coeff_small = 0.5 * tau - 0.5 * kappa_t * tau**2 + (1.0 / 3.0) * (kappa_t**2) * (tau**3)
    term3_coeff_small = tau - 0.5 * kappa_t * tau**2 + (1.0 / 6.0) * (kappa_t**2) * (tau**3)
    
    kappa_safe = torch.where(kappa_t < 1e-5, torch.ones_like(kappa_t), kappa_t)
    term2_coeff_large = (1.0 - torch.exp(-2.0 * kappa_t * tau)) / (4.0 * kappa_safe)
    term3_coeff_large = (1.0 - torch.exp(-kappa_t * tau)) / kappa_safe
    
    term2_coeff = torch.where(kappa_t < 1e-5, term2_coeff_small, term2_coeff_large)
    term3_coeff = torch.where(kappa_t < 1e-5, term3_coeff_small, term3_coeff_large)
    
    term2 = sigma_chi_t**2 * term2_coeff
    term3 = rho_t * sigma_chi_t * sigma_xi_t * term3_coeff
    term_risk = lambda_chi_t * term3_coeff
    
    return term1 + term2 + term3 - term_risk


def conditional_variance_pt(
    t: torch.Tensor | float,
    T_opt: torch.Tensor | float,
    T_fut: torch.Tensor | float,
    kappa: torch.Tensor | float,
    sigma_chi: torch.Tensor | float,
    rho: torch.Tensor | float,
    sigma_xi: torch.Tensor | float
) -> torch.Tensor:
    """PyTorch version of conditional_variance."""
    t_t = torch.as_tensor(t)
    T_opt_t = torch.as_tensor(T_opt, device=t_t.device, dtype=t_t.dtype)
    T_fut_t = torch.as_tensor(T_fut, device=t_t.device, dtype=t_t.dtype)
    kappa_t = torch.as_tensor(kappa, device=t_t.device, dtype=t_t.dtype)
    sigma_chi_t = torch.as_tensor(sigma_chi, device=t_t.device, dtype=t_t.dtype)
    rho_t = torch.as_tensor(rho, device=t_t.device, dtype=t_t.dtype)
    sigma_xi_t = torch.as_tensor(sigma_xi, device=t_t.device, dtype=t_t.dtype)
    
    tau_opt = T_opt_t - t_t
    tau_diff = T_fut_t - T_opt_t
    
    factor1_small = tau_opt - kappa_t * tau_opt**2 + (2.0 / 3.0) * (kappa_t**2) * (tau_opt**3)
    factor3_small = tau_opt - 0.5 * kappa_t * tau_opt**2 + (1.0 / 6.0) * (kappa_t**2) * (tau_opt**3)
    
    kappa_safe = torch.where(kappa_t < 1e-5, torch.ones_like(kappa_t), kappa_t)
    factor1_large = (1.0 - torch.exp(-2.0 * kappa_t * tau_opt)) / (2.0 * kappa_safe)
    factor3_large = (1.0 - torch.exp(-kappa_t * tau_opt)) / kappa_safe
    
    factor1 = torch.where(kappa_t < 1e-5, factor1_small, factor1_large)
    factor3 = torch.where(kappa_t < 1e-5, factor3_small, factor3_large)
    
    term1 = torch.exp(-2.0 * kappa_t * tau_diff) * (sigma_chi_t**2) * factor1
    term2 = sigma_xi_t**2 * tau_opt
    term3 = 2.0 * torch.exp(-kappa_t * tau_diff) * (rho_t * sigma_chi_t * sigma_xi_t) * factor3
    return term1 + term2 + term3


def futures_price_pt(
    t: torch.Tensor | float,
    T_fut: torch.Tensor | float,
    chi_t: torch.Tensor,
    xi_t: torch.Tensor,
    kappa: torch.Tensor | float,
    sigma_chi: torch.Tensor | float,
    rho: torch.Tensor | float,
    sigma_xi: torch.Tensor | float,
    mu_star: torch.Tensor | float,
    lambda_chi: torch.Tensor | float = 0.0
) -> torch.Tensor:
    """PyTorch version of futures_price."""
    t = torch.as_tensor(t, device=chi_t.device, dtype=chi_t.dtype)
    T_fut = torch.as_tensor(T_fut, device=chi_t.device, dtype=chi_t.dtype)
    kappa = torch.as_tensor(kappa, device=chi_t.device, dtype=chi_t.dtype)
    sigma_chi = torch.as_tensor(sigma_chi, device=chi_t.device, dtype=chi_t.dtype)
    rho = torch.as_tensor(rho, device=chi_t.device, dtype=chi_t.dtype)
    sigma_xi = torch.as_tensor(sigma_xi, device=chi_t.device, dtype=chi_t.dtype)
    mu_star = torch.as_tensor(mu_star, device=chi_t.device, dtype=chi_t.dtype)
    lambda_chi = torch.as_tensor(lambda_chi, device=chi_t.device, dtype=chi_t.dtype)
    xi_t = torch.as_tensor(xi_t, device=chi_t.device, dtype=chi_t.dtype)
    
    tau = T_fut - t
    log_F = torch.exp(-kappa * tau) * chi_t + xi_t + A_factor_pt(tau, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    return torch.exp(log_F)


def schwartz_smith_cf_pt(
    u: torch.Tensor,
    t: torch.Tensor | float,
    T_opt: torch.Tensor | float,
    T_fut: torch.Tensor | float,
    chi_t: torch.Tensor,
    xi_t: torch.Tensor,
    kappa: torch.Tensor | float,
    sigma_chi: torch.Tensor | float,
    rho: torch.Tensor | float,
    sigma_xi: torch.Tensor | float,
    mu_star: torch.Tensor | float,
    lambda_chi: torch.Tensor | float = 0.0
) -> torch.Tensor:
    """
    PyTorch version of characteristic function.
    Supports complex tensor u and broadcasts with state variables.
    """
    t = torch.as_tensor(t, device=chi_t.device, dtype=chi_t.dtype)
    T_opt = torch.as_tensor(T_opt, device=chi_t.device, dtype=chi_t.dtype)
    T_fut = torch.as_tensor(T_fut, device=chi_t.device, dtype=chi_t.dtype)
    kappa = torch.as_tensor(kappa, device=chi_t.device, dtype=chi_t.dtype)
    sigma_chi = torch.as_tensor(sigma_chi, device=chi_t.device, dtype=chi_t.dtype)
    rho = torch.as_tensor(rho, device=chi_t.device, dtype=chi_t.dtype)
    sigma_xi = torch.as_tensor(sigma_xi, device=chi_t.device, dtype=chi_t.dtype)
    mu_star = torch.as_tensor(mu_star, device=chi_t.device, dtype=chi_t.dtype)
    lambda_chi = torch.as_tensor(lambda_chi, device=chi_t.device, dtype=chi_t.dtype)
    xi_t = torch.as_tensor(xi_t, device=chi_t.device, dtype=chi_t.dtype)
    
    u_dtype = torch.complex128 if chi_t.dtype == torch.float64 else torch.complex64
    u = torch.as_tensor(u, device=chi_t.device, dtype=u_dtype)
    
    F_t = futures_price_pt(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    v2 = conditional_variance_pt(t, T_opt, T_fut, kappa, sigma_chi, rho, sigma_xi)
    
    # Broadcast across batch dimensions
    # u is typically of shape (N_grid,)
    # F_t, v2 are of shape (batch_size,)
    u_exp = u.unsqueeze(0) if len(u.shape) == 1 else u
    F_exp = F_t.unsqueeze(-1) if len(u.shape) == 1 else F_t
    v2_exp = v2.unsqueeze(-1) if len(u.shape) == 1 else v2
    
    m = torch.log(F_exp) - 0.5 * v2_exp
    return torch.exp(1j * u_exp * m - 0.5 * (u_exp**2) * v2_exp)


def schwartz_smith_price_black76_pt(
    t: torch.Tensor | float,
    T_opt: torch.Tensor | float,
    T_fut: torch.Tensor | float,
    K: torch.Tensor | float,
    r: torch.Tensor | float,
    chi_t: torch.Tensor,
    xi_t: torch.Tensor,
    kappa: torch.Tensor | float,
    sigma_chi: torch.Tensor | float,
    rho: torch.Tensor | float,
    sigma_xi: torch.Tensor | float,
    mu_star: torch.Tensor | float,
    lambda_chi: torch.Tensor | float = 0.0,
    option_type: str = "C"
) -> torch.Tensor:
    """
    Batch option pricing on futures using Black-76 in PyTorch.
    Enables GPU acceleration and auto-diff Greeks.
    """
    t = torch.as_tensor(t, device=chi_t.device, dtype=chi_t.dtype)
    T_opt = torch.as_tensor(T_opt, device=chi_t.device, dtype=chi_t.dtype)
    T_fut = torch.as_tensor(T_fut, device=chi_t.device, dtype=chi_t.dtype)
    K = torch.as_tensor(K, device=chi_t.device, dtype=chi_t.dtype)
    r = torch.as_tensor(r, device=chi_t.device, dtype=chi_t.dtype)
    kappa = torch.as_tensor(kappa, device=chi_t.device, dtype=chi_t.dtype)
    sigma_chi = torch.as_tensor(sigma_chi, device=chi_t.device, dtype=chi_t.dtype)
    rho = torch.as_tensor(rho, device=chi_t.device, dtype=chi_t.dtype)
    sigma_xi = torch.as_tensor(sigma_xi, device=chi_t.device, dtype=chi_t.dtype)
    mu_star = torch.as_tensor(mu_star, device=chi_t.device, dtype=chi_t.dtype)
    lambda_chi = torch.as_tensor(lambda_chi, device=chi_t.device, dtype=chi_t.dtype)
    xi_t = torch.as_tensor(xi_t, device=chi_t.device, dtype=chi_t.dtype)
    
    if torch.any(K <= 0.0):
        raise ValueError("Strike must be positive")
    if torch.any(r < 0.0):
        raise ValueError("Risk free rate must be non-negative")
    if torch.any(kappa < 0.0):
        raise ValueError("kappa must be non-negative")
    if torch.any(sigma_chi < 0.0):
        raise ValueError("sigma_chi must be non-negative")
    if torch.any(sigma_xi < 0.0):
        raise ValueError("sigma_xi must be non-negative")
    if torch.any(torch.abs(rho) > 1.0):
        raise ValueError("rho must be between -1.0 and 1.0")
    if torch.any(T_opt > T_fut):
        raise ValueError("Option maturity cannot exceed futures maturity")
        
    tau = T_opt - t
    F = futures_price_pt(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    v2 = conditional_variance_pt(t, T_opt, T_fut, kappa, sigma_chi, rho, sigma_xi)
    v = torch.sqrt(v2.clamp(min=1e-15))
    
    d1 = (torch.log(F / K) + 0.5 * v2) / v
    d2 = d1 - v
    
    # Normal CDF via erf
    ncdf_d1 = 0.5 * (1.0 + torch.erf(d1 / np.sqrt(2.0)))
    ncdf_d2 = 0.5 * (1.0 + torch.erf(d2 / np.sqrt(2.0)))
    ncdf_md1 = 0.5 * (1.0 + torch.erf(-d1 / np.sqrt(2.0)))
    ncdf_md2 = 0.5 * (1.0 + torch.erf(-d2 / np.sqrt(2.0)))
    
    call_price = torch.exp(-r * tau) * (F * ncdf_d1 - K * ncdf_d2)
    put_price = torch.exp(-r * tau) * (K * ncdf_md2 - F * ncdf_md1)
    
    price = call_price if option_type == "C" else put_price
    
    # Handle near-maturity or zero-vol boundary cases
    intrinsic = (F - K).clamp(min=0.0) if option_type == "C" else (K - F).clamp(min=0.0)
    price = torch.where((tau <= 1e-8) | (v2 < 1e-15), intrinsic, price)
    
    return price


def schwartz_smith_price_fourier_pt(
    t: torch.Tensor | float,
    T_opt: torch.Tensor | float,
    T_fut: torch.Tensor | float,
    K: torch.Tensor | float,
    r: torch.Tensor | float,
    chi_t: torch.Tensor,
    xi_t: torch.Tensor,
    kappa: torch.Tensor | float,
    sigma_chi: torch.Tensor | float,
    rho: torch.Tensor | float,
    sigma_xi: torch.Tensor | float,
    mu_star: torch.Tensor | float,
    lambda_chi: torch.Tensor | float = 0.0,
    option_type: str = "C",
    N_grid: int = 500,
    u_max: float = 100.0
) -> torch.Tensor:
    """
    Batch option pricing via Fourier inversion (Lewis method) in PyTorch.
    """
    t = torch.as_tensor(t, device=chi_t.device, dtype=chi_t.dtype)
    T_opt = torch.as_tensor(T_opt, device=chi_t.device, dtype=chi_t.dtype)
    T_fut = torch.as_tensor(T_fut, device=chi_t.device, dtype=chi_t.dtype)
    K = torch.as_tensor(K, device=chi_t.device, dtype=chi_t.dtype)
    r = torch.as_tensor(r, device=chi_t.device, dtype=chi_t.dtype)
    kappa = torch.as_tensor(kappa, device=chi_t.device, dtype=chi_t.dtype)
    sigma_chi = torch.as_tensor(sigma_chi, device=chi_t.device, dtype=chi_t.dtype)
    rho = torch.as_tensor(rho, device=chi_t.device, dtype=chi_t.dtype)
    sigma_xi = torch.as_tensor(sigma_xi, device=chi_t.device, dtype=chi_t.dtype)
    mu_star = torch.as_tensor(mu_star, device=chi_t.device, dtype=chi_t.dtype)
    lambda_chi = torch.as_tensor(lambda_chi, device=chi_t.device, dtype=chi_t.dtype)
    xi_t = torch.as_tensor(xi_t, device=chi_t.device, dtype=chi_t.dtype)
    
    if torch.any(K <= 0.0):
        raise ValueError("Strike must be positive")
    if torch.any(r < 0.0):
        raise ValueError("Risk free rate must be non-negative")
    if torch.any(kappa < 0.0):
        raise ValueError("kappa must be non-negative")
    if torch.any(sigma_chi < 0.0):
        raise ValueError("sigma_chi must be non-negative")
    if torch.any(sigma_xi < 0.0):
        raise ValueError("sigma_xi must be non-negative")
    if torch.any(torch.abs(rho) > 1.0):
        raise ValueError("rho must be between -1.0 and 1.0")
    if torch.any(T_opt > T_fut):
        raise ValueError("Option maturity cannot exceed futures maturity")
        
    F = futures_price_pt(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    tau = T_opt - t
    
    # u grid setup
    u = torch.linspace(0.0, u_max, N_grid, dtype=torch.float64, device=chi_t.device)
    u_complex = u.to(torch.complex128) - 0.5j
    
    # Get characteristic function values: (batch_size, N_grid)
    cf_val = schwartz_smith_cf_pt(
        u_complex, t, T_opt, T_fut, chi_t, xi_t,
        kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
    )
    
    # Integrand
    u_exp = u.unsqueeze(0)
    K_exp = K.unsqueeze(-1) if isinstance(K, torch.Tensor) and len(K.shape) > 0 else K
    
    # val: (batch_size, N_grid)
    val = torch.exp(-1j * u_exp * torch.log(K_exp)) * cf_val / (u_exp**2 + 0.25)
    integrand = torch.real(val)
    
    # Trapezoidal integration in PyTorch
    dx = u[1] - u[0]
    y_mean = 0.5 * (integrand[..., 1:] + integrand[..., :-1])
    integral = torch.sum(y_mean * dx, dim=-1)
    
    call_price = torch.exp(-r * tau) * (F - (torch.sqrt(K) / np.pi) * integral)
    call_price = call_price.clamp(min=0.0)
    
    price = call_price if option_type == "C" else (call_price - torch.exp(-r * tau) * (F - K)).clamp(min=0.0)
        
    intrinsic = (F - K).clamp(min=0.0) if option_type == "C" else (K - F).clamp(min=0.0)
    v2 = conditional_variance_pt(t, T_opt, T_fut, kappa, sigma_chi, rho, sigma_xi)
    price = torch.where((tau <= 1e-8) | (v2 < 1e-15), intrinsic, price)
    return price


def _cos_payoff_call_pt(a: torch.Tensor, b: torch.Tensor, strike: torch.Tensor, N: int) -> torch.Tensor:
    device = a.device
    dtype = a.dtype
    
    # Ensure inputs are at least 1D for batching consistency
    a_exp = a.reshape(-1, 1)
    b_exp = b.reshape(-1, 1)
    strike_exp = strike.reshape(-1, 1)
    
    k = torch.arange(N, dtype=dtype, device=device).unsqueeze(0) # shape: (1, N)
    
    c = torch.clamp(a_exp, min=0.0)
    d = torch.clamp(b_exp, min=0.0)
    
    # Compute psi_k(c, d)
    psi = torch.zeros((a_exp.shape[0], N), dtype=dtype, device=device)
    psi[:, 0] = (d - c).squeeze(-1)
    
    if N > 1:
        k_sub = k[:, 1:] # shape: (1, N-1)
        term_d = k_sub * np.pi * (d - a_exp) / (b_exp - a_exp)
        term_c = k_sub * np.pi * (c - a_exp) / (b_exp - a_exp)
        psi[:, 1:] = (b_exp - a_exp) / (k_sub * np.pi) * (torch.sin(term_d) - torch.sin(term_c))
        
    # Compute chi_k(c, d)
    denom = 1.0 + (k * np.pi / (b_exp - a_exp)) ** 2 # shape: (batch_size, N)
    term_d_all = k * np.pi * (d - a_exp) / (b_exp - a_exp)
    term_c_all = k * np.pi * (c - a_exp) / (b_exp - a_exp)
    
    cos_d = torch.cos(term_d_all)
    sin_d = torch.sin(term_d_all)
    cos_c = torch.cos(term_c_all)
    sin_c = torch.sin(term_c_all)
    
    exp_d = torch.exp(d)
    exp_c = torch.exp(c)
    
    chi = (1.0 / denom) * (
        cos_d * exp_d - cos_c * exp_c
        + (k * np.pi / (b_exp - a_exp)) * sin_d * exp_d
        - (k * np.pi / (b_exp - a_exp)) * sin_c * exp_c
    )
    
    V = (2.0 / (b_exp - a_exp)) * strike_exp * (chi - psi)
    
    if len(a.shape) == 0:
        return V.squeeze(0)
    return V.reshape(*a.shape, N)


def _cos_payoff_put_pt(a: torch.Tensor, b: torch.Tensor, strike: torch.Tensor, N: int) -> torch.Tensor:
    device = a.device
    dtype = a.dtype
    
    # Ensure inputs are at least 1D for batching consistency
    a_exp = a.reshape(-1, 1)
    b_exp = b.reshape(-1, 1)
    strike_exp = strike.reshape(-1, 1)
    
    k = torch.arange(N, dtype=dtype, device=device).unsqueeze(0) # shape: (1, N)
    
    c = torch.clamp(a_exp, max=0.0)
    d = torch.clamp(b_exp, max=0.0)
    
    # Compute psi_k(c, d)
    psi = torch.zeros((a_exp.shape[0], N), dtype=dtype, device=device)
    psi[:, 0] = (d - c).squeeze(-1)
    
    if N > 1:
        k_sub = k[:, 1:] # shape: (1, N-1)
        term_d = k_sub * np.pi * (d - a_exp) / (b_exp - a_exp)
        term_c = k_sub * np.pi * (c - a_exp) / (b_exp - a_exp)
        psi[:, 1:] = (b_exp - a_exp) / (k_sub * np.pi) * (torch.sin(term_d) - torch.sin(term_c))
        
    # Compute chi_k(c, d)
    denom = 1.0 + (k * np.pi / (b_exp - a_exp)) ** 2 # shape: (batch_size, N)
    term_d_all = k * np.pi * (d - a_exp) / (b_exp - a_exp)
    term_c_all = k * np.pi * (c - a_exp) / (b_exp - a_exp)
    
    cos_d = torch.cos(term_d_all)
    sin_d = torch.sin(term_d_all)
    cos_c = torch.cos(term_c_all)
    sin_c = torch.sin(term_c_all)
    
    exp_d = torch.exp(d)
    exp_c = torch.exp(c)
    
    chi = (1.0 / denom) * (
        cos_d * exp_d - cos_c * exp_c
        + (k * np.pi / (b_exp - a_exp)) * sin_d * exp_d
        - (k * np.pi / (b_exp - a_exp)) * sin_c * exp_c
    )
    
    V = (2.0 / (b_exp - a_exp)) * strike_exp * (-chi + psi)
    
    if len(a.shape) == 0:
        return V.squeeze(0)
    return V.reshape(*a.shape, N)


def price_option_cos_pt(
    t: torch.Tensor | float,
    T_opt: torch.Tensor | float,
    T_fut: torch.Tensor | float,
    K: torch.Tensor | float,
    r: torch.Tensor | float,
    chi_t: torch.Tensor,
    xi_t: torch.Tensor,
    kappa: torch.Tensor | float,
    sigma_chi: torch.Tensor | float,
    rho: torch.Tensor | float,
    sigma_xi: torch.Tensor | float,
    mu_star: torch.Tensor | float,
    lambda_chi: torch.Tensor | float = 0.0,
    option_type: str = "C",
    N: int = 128,
    L: float = 10.0
) -> torch.Tensor:
    """
    Batch option pricing on futures using the Fourier-Cosine (COS) method in PyTorch.
    Supports GPU acceleration and auto-diff Greeks.
    """
    t = torch.as_tensor(t, device=chi_t.device, dtype=chi_t.dtype)
    T_opt = torch.as_tensor(T_opt, device=chi_t.device, dtype=chi_t.dtype)
    T_fut = torch.as_tensor(T_fut, device=chi_t.device, dtype=chi_t.dtype)
    K = torch.as_tensor(K, device=chi_t.device, dtype=chi_t.dtype)
    r = torch.as_tensor(r, device=chi_t.device, dtype=chi_t.dtype)
    kappa = torch.as_tensor(kappa, device=chi_t.device, dtype=chi_t.dtype)
    sigma_chi = torch.as_tensor(sigma_chi, device=chi_t.device, dtype=chi_t.dtype)
    rho = torch.as_tensor(rho, device=chi_t.device, dtype=chi_t.dtype)
    sigma_xi = torch.as_tensor(sigma_xi, device=chi_t.device, dtype=chi_t.dtype)
    mu_star = torch.as_tensor(mu_star, device=chi_t.device, dtype=chi_t.dtype)
    lambda_chi = torch.as_tensor(lambda_chi, device=chi_t.device, dtype=chi_t.dtype)
    xi_t = torch.as_tensor(xi_t, device=chi_t.device, dtype=chi_t.dtype)
    
    if torch.any(K <= 0.0):
        raise ValueError("Strike must be positive")
    if torch.any(r < 0.0):
        raise ValueError("Risk free rate must be non-negative")
        
    # Broadcast all input tensors to a common shape
    t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi = torch.broadcast_tensors(
        t, T_opt, T_fut, K, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi
    )
    
    tau = T_opt - t
    F = futures_price_pt(t, T_fut, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
    v2 = conditional_variance_pt(t, T_opt, T_fut, kappa, sigma_chi, rho, sigma_xi)
    
    # We will clamp v2 to a safe minimum for the main COS calculations
    v2_safe = torch.clamp(v2, min=1e-15)
    
    c1 = torch.log(F / K) - 0.5 * v2_safe
    c2 = v2_safe
    
    a = c1 - L * torch.sqrt(c2)
    b = c1 + L * torch.sqrt(c2)
    
    orig_shape = F.shape
    F_flat = F.reshape(-1)
    v2_flat = v2.reshape(-1)
    v2_safe_flat = v2_safe.reshape(-1)
    c1_flat = c1.reshape(-1)
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    tau_flat = tau.reshape(-1)
    
    K_flat = K.reshape(-1)
    r_flat = r.reshape(-1)
    
    # Grid parameters
    k = torch.arange(N, dtype=chi_t.dtype, device=chi_t.device) # shape: (N,)
    u = k.unsqueeze(0) * np.pi / (b_flat.unsqueeze(-1) - a_flat.unsqueeze(-1)) # shape: (batch_size, N)
    
    # Characteristic function: phi_y(u) = exp(i * u * c1 - 0.5 * u^2 * c2)
    phi_y = torch.exp(1j * u * c1_flat.unsqueeze(-1) - 0.5 * (u ** 2) * v2_safe_flat.unsqueeze(-1))
    
    if option_type == "C":
        V = _cos_payoff_call_pt(a_flat, b_flat, K_flat, N)
    else:
        V = _cos_payoff_put_pt(a_flat, b_flat, K_flat, N)
        
    # Reconstruct pricing formula
    terms = torch.real(phi_y * torch.exp(-1j * u * a_flat.unsqueeze(-1))) * V
    
    # Multiply the k=0 term by 0.5 without in-place modification
    multiplier = torch.ones_like(terms)
    multiplier[..., 0] = 0.5
    terms = terms * multiplier
    
    price_flat = torch.exp(-r_flat * tau_flat) * torch.sum(terms, dim=-1)
    price_flat = torch.clamp(price_flat, min=0.0)
    
    # Handle limiting/boundary cases
    intrinsic = (F_flat - K_flat).clamp(min=0.0) if option_type == "C" else (K_flat - F_flat).clamp(min=0.0)
    
    price_flat = torch.where((tau_flat <= 1e-8) | (v2_flat < 1e-15), intrinsic, price_flat)
    
    return price_flat.reshape(orig_shape)


def schwartz_smith_price_cos_pt(
    t: torch.Tensor | float,
    T_opt: torch.Tensor | float,
    T_fut: torch.Tensor | float,
    K: torch.Tensor | float,
    r: torch.Tensor | float,
    chi_t: torch.Tensor,
    xi_t: torch.Tensor,
    kappa: torch.Tensor | float,
    sigma_chi: torch.Tensor | float,
    rho: torch.Tensor | float,
    sigma_xi: torch.Tensor | float,
    mu_star: torch.Tensor | float,
    lambda_chi: torch.Tensor | float = 0.0,
    option_type: str = "C",
    N: int = 128,
    L: float = 10.0
) -> torch.Tensor:
    """
    Batch option pricing on futures using the Fourier-Cosine (COS) method in PyTorch.
    Supports GPU acceleration, auto-diff Greeks, and checks device mismatches.
    Enforces constraints on inputs (e.g. non-negative inputs, positive strikes/spots).
    """
    # Enforce device/dtype matching by casting first
    t_t = torch.as_tensor(t, device=chi_t.device, dtype=chi_t.dtype)
    T_opt_t = torch.as_tensor(T_opt, device=chi_t.device, dtype=chi_t.dtype)
    T_fut_t = torch.as_tensor(T_fut, device=chi_t.device, dtype=chi_t.dtype)
    K_t = torch.as_tensor(K, device=chi_t.device, dtype=chi_t.dtype)
    r_t = torch.as_tensor(r, device=chi_t.device, dtype=chi_t.dtype)
    kappa_t = torch.as_tensor(kappa, device=chi_t.device, dtype=chi_t.dtype)
    sigma_chi_t = torch.as_tensor(sigma_chi, device=chi_t.device, dtype=chi_t.dtype)
    rho_t = torch.as_tensor(rho, device=chi_t.device, dtype=chi_t.dtype)
    sigma_xi_t = torch.as_tensor(sigma_xi, device=chi_t.device, dtype=chi_t.dtype)
    mu_star_t = torch.as_tensor(mu_star, device=chi_t.device, dtype=chi_t.dtype)
    lambda_chi_t = torch.as_tensor(lambda_chi, device=chi_t.device, dtype=chi_t.dtype)
    xi_t_t = torch.as_tensor(xi_t, device=chi_t.device, dtype=chi_t.dtype)

    # Validate finite values on all inputs
    for val in [t_t, T_opt_t, T_fut_t, K_t, r_t, chi_t, xi_t_t,
                kappa_t, sigma_chi_t, rho_t, sigma_xi_t, mu_star_t, lambda_chi_t]:
        if not torch.all(torch.isfinite(val)):
            raise ValueError("All inputs must be finite")

    # Perform validation checks
    if torch.any(K_t <= 0.0):
        raise ValueError("Strike must be positive")
    if torch.any(r_t < 0.0):
        raise ValueError("Risk free rate must be non-negative")
    if torch.any(kappa_t < 0.0):
        raise ValueError("kappa must be non-negative")
    if torch.any(sigma_chi_t < 0.0):
        raise ValueError("sigma_chi must be non-negative")
    if torch.any(sigma_xi_t < 0.0):
        raise ValueError("sigma_xi must be non-negative")
    if torch.any(torch.abs(rho_t) > 1.0):
        raise ValueError("rho must be between -1.0 and 1.0")
    if torch.any(T_opt_t > T_fut_t):
        raise ValueError("Option maturity cannot exceed futures maturity")
    if N <= 0:
        raise ValueError("N must be positive")
    if L <= 0.0:
        raise ValueError("L must be positive")

    return price_option_cos_pt(
        t=t_t, T_opt=T_opt_t, T_fut=T_fut_t, K=K_t, r=r_t, chi_t=chi_t, xi_t=xi_t_t,
        kappa=kappa_t, sigma_chi=sigma_chi_t, rho=rho_t, sigma_xi=sigma_xi_t,
        mu_star=mu_star_t, lambda_chi=lambda_chi_t, option_type=option_type,
        N=N, L=L
    )


def schwartz_smith_greeks_pt(
    t: float,
    T_opt: float,
    T_fut: float,
    K: float,
    r: float,
    chi_t: float | torch.Tensor,
    xi_t: float | torch.Tensor,
    kappa: float,
    sigma_chi: float,
    rho: float,
    sigma_xi: float,
    mu_star: float,
    lambda_chi: float = 0.0,
    option_type: str = "C",
    target_greek: str = "delta"
) -> dict[str, torch.Tensor]:
    """
    Computes option price and Greeks (Delta w.r.t chi & xi, Gamma, Vega, etc.) 
    using PyTorch autograd.
    """
    # Force float values into tensors requiring grad
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    chi_tensor = torch.tensor(chi_t, dtype=torch.float64, device=device, requires_grad=True) if not isinstance(chi_t, torch.Tensor) else chi_t.clone().detach().requires_grad_(True)
    xi_tensor = torch.tensor(xi_t, dtype=torch.float64, device=device, requires_grad=True) if not isinstance(xi_t, torch.Tensor) else xi_t.clone().detach().requires_grad_(True)
    
    kappa_t = torch.tensor(kappa, dtype=torch.float64, device=device, requires_grad=True)
    sigma_chi_t = torch.tensor(sigma_chi, dtype=torch.float64, device=device, requires_grad=True)
    sigma_xi_t = torch.tensor(sigma_xi, dtype=torch.float64, device=device, requires_grad=True)
    rho_t = torch.tensor(rho, dtype=torch.float64, device=device, requires_grad=True)
    lambda_chi_t = torch.tensor(lambda_chi, dtype=torch.float64, device=device, requires_grad=True)
    
    # Calculate price
    price = schwartz_smith_price_black76_pt(
        t, T_opt, T_fut, K, r, chi_tensor, xi_tensor,
        kappa_t, sigma_chi_t, rho_t, sigma_xi_t, mu_star, lambda_chi_t,
        option_type=option_type
    )
    
    greeks = {"price": price.detach().cpu()}
    
    # We want to compute gradients of the sum of prices (if batched)
    grad_outputs = torch.ones_like(price)
    
    if target_greek.lower() == "all" or target_greek.lower() == "delta":
        # Deltas
        grads = torch.autograd.grad(price, (chi_tensor, xi_tensor), grad_outputs=grad_outputs, create_graph=True)
        greeks["delta_chi"] = grads[0]
        greeks["delta_xi"] = grads[1]
        
        # Gammas (second derivative)
        grads_gamma_chi = torch.autograd.grad(grads[0], chi_tensor, grad_outputs=grad_outputs, retain_graph=True)[0]
        grads_gamma_xi = torch.autograd.grad(grads[1], xi_tensor, grad_outputs=grad_outputs, retain_graph=True)[0]
        greeks["gamma_chi"] = grads_gamma_chi
        greeks["gamma_xi"] = grads_gamma_xi
        
    if target_greek.lower() == "all" or target_greek.lower() == "vega":
        grads_vega = torch.autograd.grad(price, (sigma_chi_t, sigma_xi_t), grad_outputs=grad_outputs, retain_graph=True)
        greeks["vega_sigma_chi"] = grads_vega[0]
        greeks["vega_sigma_xi"] = grads_vega[1]
        
    if target_greek.lower() == "all" or target_greek.lower() == "kappa":
        grads_kappa = torch.autograd.grad(price, kappa_t, grad_outputs=grad_outputs, retain_graph=True)[0]
        greeks["vega_kappa"] = grads_kappa
        
    # Detach everything
    for k in greeks:
        if k != "price":
            greeks[k] = greeks[k].detach().cpu()
            
    return greeks


# ---------------------------------------------------------------------------
# 3. Kalman Filter & Calibration
# ---------------------------------------------------------------------------

def run_kalman_filter(
    dates: list[datetime.date] | np.ndarray,
    futures_prices: np.ndarray,
    maturities: np.ndarray,
    kappa: float,
    sigma_chi: float,
    rho: float,
    sigma_xi: float,
    mu: float,
    lambda_chi: float,
    mu_star: float,
    sigma_e: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Runs the Kalman Filter for the Schwartz-Smith model on a panel of futures prices.
    
    Parameters
    ----------
    dates : list/array of dates of length N_dates
    futures_prices : array of shape (N_dates, N_contracts)
    maturities : array of shape (N_dates, N_contracts) (time-to-maturity of each contract)
    
    Returns
    -------
    log_likelihood : float
    filtered_states : array of shape (N_dates, 2)
    filtered_covariances : array of shape (N_dates, 2, 2)
    """
    num_dates, num_contracts = futures_prices.shape
    
    # Initial state estimate: [chi_0, xi_0]^T
    # We initialize chi_0 = 0.0, and xi_0 to the mean of the first date's valid log futures prices
    first_date_prices = futures_prices[0]
    valid_first_prices = first_date_prices[~np.isnan(first_date_prices) & (first_date_prices > 0)]
    if len(valid_first_prices) > 0:
        xi_0 = np.mean(np.log(valid_first_prices))
    else:
        xi_0 = 0.0
    x_hat = np.array([0.0, xi_0])
    
    P = np.array([
        [sigma_chi**2 / (2.0 * kappa) if kappa >= 1e-5 else sigma_chi**2, 0.0],
        [0.0, 10.0]
    ])
    
    filtered_states = np.zeros((num_dates, 2))
    filtered_covs = np.zeros((num_dates, 2, 2))
    
    filtered_states[0] = x_hat
    filtered_covs[0] = P
    
    log_lik = 0.0
    
    for t in range(1, num_dates):
        dt = (dates[t] - dates[t-1]).days / 365.0
        if dt <= 0:
            # Skip duplicate dates or zero dt
            filtered_states[t] = x_hat
            filtered_covs[t] = P
            continue
            
        exp_k = np.exp(-kappa * dt)
        
        # State transition: x_t = c + G * x_{t-1} + omega_t
        c = np.array([
            0.0,
            mu * dt
        ])
        G = np.array([
            [exp_k, 0.0],
            [0.0, 1.0]
        ])
        
        # Covariance of state transition noise: Q
        var_chi = (sigma_chi**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * dt)) if kappa >= 1e-5 else sigma_chi**2 * dt
        var_xi = sigma_xi**2 * dt
        cov_chi_xi = (rho * sigma_chi * sigma_xi / kappa) * (1.0 - exp_k) if kappa >= 1e-5 else rho * sigma_chi * sigma_xi * dt
        Q = np.array([
            [var_chi, cov_chi_xi],
            [cov_chi_xi, var_xi]
        ])
        
        # 1. Predict Step
        x_hat_pred = c + G @ x_hat
        P_pred = G @ P @ G.T + Q
        
        # 2. Measurement Update Step
        valid_mask = ~np.isnan(futures_prices[t]) & (futures_prices[t] > 0)
        num_valid = np.sum(valid_mask)
        
        if num_valid == 0:
            # Skip measurement update step if all contracts are NaN
            x_hat = x_hat_pred
            P = P_pred
            filtered_states[t] = x_hat
            filtered_covs[t] = P
            continue
            
        tau_valid = maturities[t][valid_mask]
        y_observed_valid = np.log(futures_prices[t][valid_mask])
        
        # Compute A(tau) for each valid contract
        A_val_valid = A_factor(tau_valid, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)
        
        # Measurement matrix H_t and displacement d_t
        d_t_valid = A_val_valid
        H_t_valid = np.column_stack([np.exp(-kappa * tau_valid), np.ones(num_valid)])
        
        # Predicted measurement
        y_pred_valid = d_t_valid + H_t_valid @ x_hat_pred
        
        # Innovation
        v_valid = y_observed_valid - y_pred_valid
        
        # Innovation Covariance
        R_valid = (sigma_e**2) * np.eye(num_valid)
        F_cov_valid = H_t_valid @ P_pred @ H_t_valid.T + R_valid
        
        try:
            F_inv_valid = np.linalg.inv(F_cov_valid)
            det_F_valid = np.linalg.det(F_cov_valid)
        except np.linalg.LinAlgError:
            return -1e10, filtered_states, filtered_covs
            
        if det_F_valid <= 0:
            return -1e10, filtered_states, filtered_covs
            
        # Update State
        K_gain_valid = P_pred @ H_t_valid.T @ F_inv_valid
        x_hat = x_hat_pred + K_gain_valid @ v_valid
        P = (np.eye(2) - K_gain_valid @ H_t_valid) @ P_pred
        
        filtered_states[t] = x_hat
        filtered_covs[t] = P
        
        # Log-likelihood contribution
        term1 = -0.5 * num_valid * np.log(2.0 * np.pi)
        term2 = -0.5 * np.log(det_F_valid)
        term3 = -0.5 * v_valid.T @ F_inv_valid @ v_valid
        
        log_lik += term1 + term2 + term3
        
    return log_lik, filtered_states, filtered_covs


def calibrate_schwartz_smith(
    dates: list[datetime.date] | np.ndarray,
    futures_prices: np.ndarray,
    maturities: np.ndarray,
    init_guess: list[float] | None = None
) -> dict[str, float]:
    """
    Calibrates the Schwartz-Smith parameters from historical futures prices using the Kalman Filter.
    
    Parameters
    ----------
    dates : list/array of dates
    futures_prices : array of shape (N_dates, N_contracts)
    maturities : array of shape (N_dates, N_contracts)
    init_guess : list of length 8 representing initial values for:
                 [kappa, sigma_chi, rho, sigma_xi, mu, lambda_chi, mu_star, sigma_e]
                 
    Returns
    -------
    calibrated_params : dict of parameter names to values
    """
    if init_guess is None:
        # Reasonable WTI Crude Oil default parameters
        # kappa, sigma_chi, rho, sigma_xi, mu, lambda_chi, mu_star, sigma_e
        init_guess = [0.5, 0.20, 0.30, 0.10, 0.05, 0.02, 0.03, 0.01]
        
    # Bounds to enforce constraints (e.g. kappa > 0, sigma > 0, -1 < rho < 1, sigma_e > 0)
    bounds = [
        (1e-4, 5.0),    # kappa
        (1e-4, 2.0),    # sigma_chi
        (-0.99, 0.99),  # rho
        (1e-4, 2.0),    # sigma_xi
        (-2.0, 2.0),    # mu
        (-2.0, 2.0),    # lambda_chi
        (-2.0, 2.0),    # mu_star
        (1e-5, 0.5)     # sigma_e
    ]
    
    # Objective function: negative log-likelihood
    def objective(x):
        kappa, sigma_chi, rho, sigma_xi, mu, lambda_chi, mu_star, sigma_e = x
        ll, _, _ = run_kalman_filter(
            dates, futures_prices, maturities,
            kappa, sigma_chi, rho, sigma_xi,
            mu, lambda_chi, mu_star, sigma_e
        )
        return -ll
        
    res = scipy.optimize.minimize(
        objective,
        x0=init_guess,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200}
    )
    
    p = res.x
    return {
        "kappa": float(p[0]),
        "sigma_chi": float(p[1]),
        "rho": float(p[2]),
        "sigma_xi": float(p[3]),
        "mu": float(p[4]),
        "lambda_chi": float(p[5]),
        "mu_star": float(p[6]),
        "sigma_e": float(p[7]),
        "success": bool(res.success),
        "log_likelihood": float(-res.fun)
    }


class SchwartzSmithEngine:
    def __init__(self, kappa: float, mu_y: float, sigma_x: float, sigma_y: float, rho_xy: float):
        if not (np.isfinite(kappa) and np.isfinite(mu_y) and np.isfinite(sigma_x) and np.isfinite(sigma_y) and np.isfinite(rho_xy)):
            raise ValueError("All inputs must be finite")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if sigma_x <= 0.0:
            raise ValueError("sigma_x must be positive")
        if sigma_y <= 0.0:
            raise ValueError("sigma_y must be positive")
        if not (-1.0 <= rho_xy <= 1.0):
            raise ValueError("Correlation (rho_xy) must be in range [-1.0, 1.0]")
            
        self.kappa = kappa
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho_xy = rho_xy

    def price_option(self, spot: float, strike: float, maturity: float, risk_free_rate: float = 0.0, is_call: bool = True) -> float:
        if not (np.isfinite(spot) and np.isfinite(strike) and np.isfinite(maturity) and np.isfinite(risk_free_rate)):
            raise ValueError("All inputs must be finite")
        if spot <= 0.0:
            raise ValueError("Spot must be positive")
        if strike <= 0.0:
            raise ValueError("Strike must be positive")
        if maturity <= 0.0:
            raise ValueError("Maturity must be positive")
        if risk_free_rate < 0.0:
            raise ValueError("Risk free rate must be non-negative")
            
        from scipy.stats import norm
        # Variance of ln(S_T) under Schwartz-Smith
        v2 = conditional_variance(0.0, maturity, maturity, self.kappa, self.sigma_x, self.rho_xy, self.sigma_y)
        if v2 <= 0.0:
            return max(spot - strike * np.exp(-risk_free_rate * maturity) if is_call else strike * np.exp(-risk_free_rate * maturity) - spot, 0.0)
            
        vol = np.sqrt(v2 / maturity)
        
        # Black-Scholes pricing
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * vol**2) * maturity) / (vol * np.sqrt(maturity))
        d2 = d1 - vol * np.sqrt(maturity)
        
        if is_call:
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * maturity) * norm.cdf(d2)
            return max(price, 0.0)
        else:
            price = strike * np.exp(-risk_free_rate * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            return max(price, 0.0)

    def heston_price(self, spot: float, strike: float, maturity: float, risk_free_rate: float, heston_params: dict, is_call: bool = True) -> float:
        if not (np.isfinite(spot) and np.isfinite(strike) and np.isfinite(maturity) and np.isfinite(risk_free_rate)):
            raise ValueError("All inputs must be finite")
        if spot <= 0.0:
            raise ValueError("Spot must be positive")
        if strike <= 0.0:
            raise ValueError("Strike must be positive")
        if maturity <= 0.0:
            raise ValueError("Maturity must be positive")
        if risk_free_rate < 0.0:
            raise ValueError("Risk free rate must be non-negative")
            
        # Validate heston_params
        required_keys = ['kappa', 'theta', 'sigma', 'rho', 'v0']
        for k in required_keys:
            if k not in heston_params:
                raise ValueError("Heston parameters must contain 'kappa', 'theta', 'sigma', 'rho', 'v0'")
            if not np.isfinite(heston_params[k]):
                raise ValueError("Heston parameters must be finite")
                
        if heston_params['kappa'] <= 0.0 or heston_params['theta'] <= 0.0 or heston_params['sigma'] <= 0.0 or heston_params['v0'] <= 0.0:
            raise ValueError("Heston parameters (kappa, theta, sigma, v0) must be positive")
            
        if not (-1.0 <= heston_params['rho'] <= 1.0):
            raise ValueError("Heston correlation (rho) must be between -1.0 and 1.0")
            
        # Price using Heston COS method
        from pricing.heston import heston_cf, cos_payoff_coeffs_np, cos_payoff_coeffs_put_np
        
        N_cos = 128
        a, b = -4.0, 4.0
        k_arr = np.arange(N_cos)
        u_k = k_arr * np.pi / (b - a)
        
        # Spot expectation drifts at r: E[S_T] = S0 * exp(r * T)
        # CF of ln(S_T/K) = CF of ln(S_T/S0) + 1j * u * ln(S0/K)
        # Under risk-neutral Heston, log-price is ln(S_T/S0) = (r - q - 0.5*v0)T + ...
        # heston_cf handles the characteristic function of the dynamic parts.
        # Let's add the forward drift:
        x0 = np.log(spot / strike) + risk_free_rate * maturity
        
        phi_k = heston_cf(
            u_k, maturity,
            heston_params['kappa'],
            heston_params['theta'],
            heston_params['sigma'],
            heston_params['rho'],
            heston_params['v0']
        ) * np.exp(1j * u_k * x0)
        phi_k[0] = 1.0 + 0j
        
        if is_call:
            Vk = cos_payoff_coeffs_np(N_cos, a, b)
            price = strike * np.real(np.sum(phi_k * np.exp(-1j * u_k * a) * Vk))
            price = np.exp(-risk_free_rate * maturity) * price
            return max(price, 0.0)
        else:
            Vk = cos_payoff_coeffs_put_np(N_cos, a, b)
            price = strike * np.real(np.sum(phi_k * np.exp(-1j * u_k * a) * Vk))
            price = np.exp(-risk_free_rate * maturity) * price
            return max(price, 0.0)

    def compare_vs_heston(self, spot: float, strike: float, maturity: float, heston_params: dict, risk_free_rate: float = 0.0, is_call: bool = True) -> dict:
        ss_price = self.price_option(spot, strike, maturity, risk_free_rate, is_call)
        h_price = self.heston_price(spot, strike, maturity, risk_free_rate, heston_params, is_call)
        abs_err = abs(ss_price - h_price)
        rel_err = abs_err / max(h_price, 1e-8)
        return {
            "schwartz_smith_price": ss_price,
            "heston_price": h_price,
            "absolute_error": abs_err,
            "relative_error": rel_err
        }
