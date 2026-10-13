"""
artifacts/audit/scripts/oracles.py — Clean-room float64 reference oracles for deepvol FX cross pricing audit.
Derived strictly from literature (Black 1976, Garman-Kohlhagen 1983, Margrabe 1978, Gatheral 2004,
Reiswich & Wystup 2010, Clark 2011) and SPEC-DEEPVOL-FX-CROSS-003 v3.0.0.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri


def std_norm_pdf(x: float | np.ndarray) -> float | np.ndarray:
    """Float64 standard normal PDF."""
    return np.exp(-0.5 * np.square(x)) / math.sqrt(2.0 * math.pi)


def black76_price(
    F: float,
    K: float,
    sigma: float,
    T: float,
    D_set: float,
    phi: int = 1,
) -> float:
    """Float64 Black-76 European vanilla option price."""
    if T <= 0.0 or sigma <= 0.0:
        intrinsic = max(0.0, float(phi) * (F - K))
        return D_set * intrinsic
    sigma_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    if phi == 1:
        return D_set * (F * float(ndtr(d1)) - K * float(ndtr(d2)))
    else:
        return D_set * (K * float(ndtr(-d2)) - F * float(ndtr(-d1)))


def garman_kohlhagen_price(
    S: float,
    K: float,
    sigma: float,
    T: float,
    D_foreign: float,
    D_domestic: float,
    phi: int = 1,
) -> float:
    """Float64 Garman-Kohlhagen European FX option price."""
    F = S * D_foreign / D_domestic
    return black76_price(F, K, sigma, T, D_domestic, phi)


def implied_vol_brent(
    c: float,
    F: float,
    K: float,
    T: float,
    D_set: float = 1.0,
    phi: int = 1,
    vol_min: float = 1e-4,
    vol_max: float = 5.0,
) -> float:
    """Independent float64 implied volatility solver via Brent's method to 1e-13."""
    undisc_c = c / D_set
    intrinsic = max(0.0, float(phi) * (F - K))
    upper_bound = F if phi == 1 else K
    if undisc_c <= intrinsic + 1e-14 * F:
        return 0.0
    if undisc_c >= upper_bound - 1e-14 * F:
        return vol_max

    def objective(s: float) -> float:
        return black76_price(F, K, s, T, 1.0, phi) - undisc_c

    try:
        return float(brentq(objective, vol_min, vol_max, xtol=1e-13, rtol=1e-13))
    except ValueError:
        return float(brentq(objective, 1e-6, 10.0, xtol=1e-13, rtol=1e-13))


def svi_total_variance(
    x: float | np.ndarray,
    a: float,
    b: float,
    rho_s: float,
    m: float,
    s: float,
) -> Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Raw SVI total variance w(x) and its exact first and second derivatives."""
    diff = x - m
    hyp = np.sqrt(np.square(diff) + s * s)
    w = a + b * (rho_s * diff + hyp)
    w_prime = b * (rho_s + diff / hyp)
    w_second = b * (s * s) / (hyp**3)
    return w, w_prime, w_second


def durrleman_g(
    x: float | np.ndarray,
    a: float,
    b: float,
    rho_s: float,
    m: float,
    s: float,
) -> float | np.ndarray:
    """Durrleman butterfly non-negativity certificate g(x)."""
    w, w_p, w_pp = svi_total_variance(x, a, b, rho_s, m, s)
    term1 = 1.0 - (x * w_p) / (2.0 * w)
    term2 = 0.25 * np.square(w_p) * (1.0 / w + 0.25)
    term3 = 0.5 * w_pp
    return np.square(term1) - term2 + term3


def ssvi_to_raw_svi(
    theta: float,
    rho: float,
    phi: float,
) -> Tuple[float, float, float, float, float]:
    """Map Gatheral-Jacquier SSVI parameters to raw-SVI parameters (a, b, rho_s, m, s)."""
    a = 0.5 * theta * (1.0 - rho * rho)
    b = 0.5 * theta * phi
    rho_s = rho
    m = -rho / phi
    s = math.sqrt(max(0.0, 1.0 - rho * rho)) / phi
    return a, b, rho_s, m, s


def beta_split_h1(
    k: float,
    T: float,
    rho: float,
    slice1_fn: Callable[[float], float],
    slice2_fn: Callable[[float], float],
    n_fp: int = 30,
    eps_beta: float = 1e-8,
) -> Tuple[float, float, float, float, float]:
    """Clean-room Beta-split H1 triangulation with n_fp fixed-point steps.

    Returns:
        (sigma_H1, sigma_1, sigma_2, beta_1, beta_2)
    """
    s1 = slice1_fn(0.0)
    s2 = slice2_fn(0.0)
    beta1 = 0.5
    beta2 = 0.5
    for _ in range(n_fp):
        var_cross = s1 * s1 + s2 * s2 + 2.0 * rho * s1 * s2
        denom = var_cross + eps_beta
        beta1 = (s1 * s1 + rho * s1 * s2) / denom
        beta2 = 1.0 - beta1
        s1 = slice1_fn(beta1 * k)
        s2 = slice2_fn(beta2 * k)

    var_final = max(1e-8, s1 * s1 + s2 * s2 + 2.0 * rho * s1 * s2)
    sigma_h1 = math.sqrt(var_final)
    return sigma_h1, s1, s2, beta1, beta2


def closed_form_greeks(
    F: float,
    K: float,
    sigma: float,
    T: float,
    D_set: float,
    g: float,
    S_cross: float,
    r_C: float,
    r_A: float,
    r_set: float,
    phi: int = 1,
) -> dict[str, float]:
    """Float64 closed-form Black sensitivities at the served sigma."""
    if T <= 0.0 or sigma <= 0.0:
        return {}

    sigma_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    nd1 = float(ndtr(float(phi) * d1))
    nd2 = float(ndtr(float(phi) * d2))
    pdf_d1 = float(std_norm_pdf(d1))

    price = D_set * (float(phi) * F * nd1 - float(phi) * K * nd2) if phi == 1 else D_set * (K * float(ndtr(-d2)) - F * float(ndtr(-d1)))
    cash_delta = float(phi) * D_set * F * nd1
    delta_cross = float(phi) * D_set * g * nd1
    non_adapted_delta = float(phi) * D_set * nd1

    # Gamma contains carry factor g
    gamma_cross = D_set * (g * g) * pdf_d1 / (F * sigma_sqrt_T)

    # Cross Vega
    vega_cross = D_set * F * math.sqrt(T) * pdf_d1

    # Vanna = -D_set * g * n(d1) * d2 / sigma
    vanna = -D_set * g * pdf_d1 * d2 / sigma

    # Vomma = Vega * d1 * d2 / sigma
    vomma = vega_cross * d1 * d2 / sigma

    # Theta per calendar day (divide by 365.0)
    theta_annual = (
        r_set * price
        - float(phi) * D_set * F * nd1 * (r_C - r_A)
        - (D_set * F * sigma * pdf_d1) / (2.0 * math.sqrt(T))
    )
    theta_daily = theta_annual / 365.0

    return {
        "price": price,
        "delta": delta_cross,
        "cash_delta": cash_delta,
        "non_adapted_delta": non_adapted_delta,
        "gamma": gamma_cross,
        "vega": vega_cross,
        "vanna": vanna,
        "vomma": vomma,
        "theta": theta_daily,
        "theta_annual": theta_annual,
    }


def smile_inclusive_greeks(
    F: float,
    K: float,
    sigma: float,
    T: float,
    D_set: float,
    g: float,
    s_k: float,
    s_kk: float,
    r_C: float,
    r_A: float,
    r_set: float,
    phi: int = 1,
    sticky_moneyness: bool = True,
) -> dict[str, float]:
    """Exact smile-inclusive Greeks under sticky-moneyness or sticky-strike."""
    sigma_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    pdf_d1 = float(std_norm_pdf(d1))
    nd1 = float(ndtr(float(phi) * d1))
    nd2 = float(ndtr(float(phi) * d2))

    B_F = float(phi) * D_set * nd1
    B_FF = D_set * pdf_d1 / (F * sigma_sqrt_T)
    B_s = D_set * F * math.sqrt(T) * pdf_d1
    B_Fs = -D_set * pdf_d1 * d2 / sigma
    B_ss = B_s * d1 * d2 / sigma

    if not sticky_moneyness:
        s_k = 0.0
        s_kk = 0.0

    k_F = -1.0 / F
    k_FF = 1.0 / (F * F)

    dVdF = B_F + B_s * s_k * k_F
    d2VdF2 = (
        B_FF
        + 2.0 * B_Fs * s_k * k_F
        + B_ss * ((s_k * k_F) ** 2)
        + B_s * (s_kk * (k_F**2) + s_k * k_FF)
    )

    delta = g * dVdF
    gamma = (g * g) * d2VdF2
    vanna = g * (B_Fs + B_ss * s_k * k_F)
    vomma = B_ss

    price = D_set * (float(phi) * F * nd1 - float(phi) * K * nd2) if phi == 1 else D_set * (K * float(ndtr(-d2)) - F * float(ndtr(-d1)))
    theta_annual = (
        r_set * price
        - float(phi) * D_set * F * nd1 * (r_C - r_A)
        - (D_set * F * sigma * pdf_d1) / (2.0 * math.sqrt(T))
    )
    if sticky_moneyness:
        theta_annual += B_s * s_k * (r_C - r_A)
    theta_daily = theta_annual / 365.0

    return {
        "delta": delta,
        "gamma": gamma,
        "vanna": vanna,
        "vomma": vomma,
        "theta": theta_daily,
        "dVdF": dVdF,
        "d2VdF2": d2VdF2,
    }


def adaptive_stencil_gamma(
    pricing_fn: Callable[[float], float],
    S: float,
    sigma_ref: float,
    T: float,
) -> float:
    """Adaptive 3-point stencil gamma test oracle: h = S * min(1e-3, 0.02 * sigma_ref * sqrt(T))."""
    h = S * min(1e-3, 0.02 * sigma_ref * math.sqrt(T))
    v_plus = pricing_fn(S + h)
    v_mid = pricing_fn(S)
    v_minus = pricing_fn(S - h)
    return (v_plus - 2.0 * v_mid + v_minus) / (h * h)


def gaussian_copula_cross_pricer_lognormal(
    F1: float,
    F2: float,
    K: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    T: float,
    n_nodes: int = 1601,
    L: float = 8.5,
) -> float:
    """Clean-room Gaussian-copula pricer under Q^B with lognormal legs.

    Evaluates:
        c(K) = F2 * E^{Q^B}[(X - K*Y)^+]
    where X = S1(T), Y = 1/S2(T), corr(W_1^B, W_Y) = -rho.
    """
    z = np.linspace(-L, L, n_nodes)
    wq = std_norm_pdf(z)
    wq = wq / wq.sum()

    vol1_sqT = sigma1 * math.sqrt(T)
    vol2_sqT = sigma2 * math.sqrt(T)
    rho_XY = -rho
    rho_perp = math.sqrt(max(0.0, 1.0 - rho_XY * rho_XY))

    X = F1 * np.exp(-0.5 * sigma1 * sigma1 * T + vol1_sqT * z)
    Z2 = rho_XY * z[:, None] + rho_perp * z[None, :]
    Y = (1.0 / F2) * np.exp(-0.5 * sigma2 * sigma2 * T + vol2_sqT * Z2)

    W = wq[:, None] * wq[None, :]
    payoff = np.maximum(X[:, None] - K * Y, 0.0)
    return float(F2 * (W * payoff).sum())


class MarginalSVI:
    """Risk-neutral marginal law of an FX rate from an SVI slice under Q^B."""

    def __init__(self, F: float, a: float, b: float, rho_s: float, m: float, s: float, T: float, reflect: bool = False):
        self.F = 1.0 / F if reflect else F
        self.T = T
        sd = math.sqrt(a + b * (rho_s * (-m) + math.sqrt(m * m + s * s)))
        kg = np.linspace(-14.0 * sd, 14.0 * sd, 40001)
        ks = -kg if reflect else kg
        w, w_p, _ = svi_total_variance(ks, a, b, rho_s, m, s)
        sig = np.sqrt(np.maximum(w, 1e-12) / T)
        dsdk_s = w_p / (2.0 * sig * T)
        dsdk = -dsdk_s if reflect else dsdk_s
        K = self.F * np.exp(kg)
        sq = sig * math.sqrt(T)
        d1 = (-kg + 0.5 * sq * sq) / sq
        d2 = d1 - sq
        vega = self.F * std_norm_pdf(d1) * math.sqrt(T)
        dsdK = dsdk / K
        self.cdf = ndtr(-d2) + vega * dsdK
        self.sf = ndtr(d2) - vega * dsdK
        self.kg = kg

    def quantile_from_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        out = np.empty_like(z)
        lo = z < 0
        lc = np.log(np.maximum(self.cdf, 1e-300))
        ls = np.log(np.maximum(self.sf, 1e-300))
        out[lo] = np.interp(np.log(ndtr(z[lo])), lc, self.kg)
        out[~lo] = np.interp(np.log(ndtr(-z[~lo])), ls[::-1], self.kg[::-1])
        return self.F * np.exp(out)


def copula_cross_price_svi(
    mX: MarginalSVI,
    mY: MarginalSVI,
    F2: float,
    K: float,
    rho_z: float,
    n_nodes: int = 1601,
    L: float = 8.5,
) -> float:
    """Clean-room Gaussian-copula pricer over general SVI marginals."""
    z = np.linspace(-L, L, n_nodes)
    wq = std_norm_pdf(z)
    wq = wq / wq.sum()

    X = mX.quantile_from_z(z)
    rho_perp = math.sqrt(max(0.0, 1.0 - rho_z * rho_z))
    Z2 = rho_z * z[:, None] + rho_perp * z[None, :]
    Y = mY.quantile_from_z(Z2.ravel()).reshape(n_nodes, n_nodes)

    W = wq[:, None] * wq[None, :]
    payoff = np.maximum(X[:, None] - K * Y, 0.0)
    return float(F2 * (W * payoff).sum())


def fx_delta_solve_strike(
    delta_target: float,
    sigma: float,
    F: float,
    T: float,
    D_f: float,
    phi: int,
    convention: str,
    tol: float = 1e-14,
    max_iter: int = 50,
) -> float:
    """Clean-room float64 FX delta to strike solver for all 4 conventions."""
    sigma_sqrt_T = sigma * math.sqrt(T)

    if convention == "forward_pna":
        arg = float(phi) * delta_target
        if not (0.0 < arg < 1.0):
            raise ValueError(f"Invalid delta {delta_target} for phi={phi}")
        z = float(ndtri(arg))
        d1 = float(phi) * z
        return F * math.exp(-d1 * sigma_sqrt_T + 0.5 * sigma * sigma * T)

    elif convention == "spot_pna":
        arg = (float(phi) * delta_target) / D_f
        if not (0.0 < arg < 1.0):
            raise ValueError(f"Invalid spot delta {delta_target}")
        z = float(ndtri(arg))
        d1 = float(phi) * z
        return F * math.exp(-d1 * sigma_sqrt_T + 0.5 * sigma * sigma * T)

    elif convention in ("forward_pa", "spot_pa"):
        target = delta_target if convention == "forward_pa" else delta_target / D_f

        def delta_fn(K_cand: float) -> Tuple[float, float]:
            d2 = (math.log(F / K_cand) - 0.5 * sigma * sigma * T) / sigma_sqrt_T
            nd2 = float(ndtr(float(phi) * d2))
            val = float(phi) * (K_cand / F) * nd2
            pdf_d2 = float(std_norm_pdf(d2))
            deriv = (float(phi) / F) * nd2 - (K_cand / F) * pdf_d2 / (K_cand * sigma_sqrt_T)
            return val, deriv

        k_pna = fx_delta_solve_strike(
            delta_target, sigma, F, T, D_f, phi, "forward_pna" if convention == "forward_pa" else "spot_pna"
        )
        if phi == 1:
            low = 1e-4 * F
            high = 2.0 * max(F, k_pna)
        else:
            low = 1e-5 * min(F, k_pna)
            high = 2.0 * F

        K_curr = k_pna * 0.95
        for _ in range(max_iter):
            val, deriv = delta_fn(K_curr)
            err = val - target
            if abs(err) < tol:
                return K_curr
            if abs(deriv) > 1e-12:
                K_next = K_curr - err / deriv
                if low <= K_next <= high:
                    K_curr = K_next
                    continue
            if err * (delta_fn(low)[0] - target) < 0:
                high = K_curr
            else:
                low = K_curr
            K_curr = 0.5 * (low + high)

        return K_curr

    else:
        raise ValueError(f"Unknown convention {convention}")


def check_arbitrage_grid(
    strikes: Sequence[float],
    expiries: Sequence[float],
    call_surface: np.ndarray,
    D_grid: np.ndarray,
    total_var_surface: np.ndarray | None = None,
    tau_df: float = 1e-12,
    tau_totvar: float = 1e-14,
) -> Tuple[int, int, int]:
    """Independent float64 grid certifier for Vertical, Butterfly, and Calendar arbitrage.

    Args:
        strikes: (n_K,) sorted strikes
        expiries: (n_T,) sorted expiries
        call_surface: (n_T, n_K) discounted call prices
        D_grid: (n_T,) discount factors
        total_var_surface: optional (n_T, n_K) total variance w(k, T)
        tau_df: tolerance threshold * DF
        tau_totvar: tolerance for calendar total variance monotonicity

    Returns:
        (n_vertical_violations, n_butterfly_violations, n_calendar_violations)
    """
    n_T, n_K = call_surface.shape
    n_vert = 0
    n_butt = 0
    n_cal = 0

    for t_idx in range(n_T):
        D = D_grid[t_idx]
        tau = tau_df * D
        c = call_surface[t_idx]

        # 1. Vertical monotonicity: -D - tau <= dC/dK <= tau
        # (C[i] - C[i+1]) / (K[i+1] - K[i]) in [-tau, D + tau]
        for k_idx in range(n_K - 1):
            dK = strikes[k_idx + 1] - strikes[k_idx]
            dC = c[k_idx + 1] - c[k_idx]
            slope = dC / dK
            if slope > tau or slope < -D - tau:
                n_vert += 1

        # 2. Butterfly convexity: C[i] - 2 C[i+1] + C[i+2] >= -tau
        for k_idx in range(n_K - 2):
            dK1 = strikes[k_idx + 1] - strikes[k_idx]
            dK2 = strikes[k_idx + 2] - strikes[k_idx + 1]
            c0, c1, c2 = c[k_idx], c[k_idx + 1], c[k_idx + 2]
            # Standard central second difference when spaced
            d2C = (c2 - c1) / dK2 - (c1 - c0) / dK1
            if d2C < -tau / min(dK1, dK2):
                n_butt += 1

    # 3. Calendar monotonicity
    if total_var_surface is not None:
        for t_idx in range(n_T - 1):
            w_curr = total_var_surface[t_idx]
            w_next = total_var_surface[t_idx + 1]
            diff = w_next - w_curr
            if np.any(diff < -tau_totvar):
                n_cal += int(np.sum(diff < -tau_totvar))
    else:
        # Undiscounted call monotonicity in calendar time for fixed forward
        for t_idx in range(n_T - 1):
            c_curr = call_surface[t_idx] / D_grid[t_idx]
            c_next = call_surface[t_idx + 1] / D_grid[t_idx + 1]
            diff = c_next - c_curr
            if np.any(diff < -tau_df):
                n_cal += int(np.sum(diff < -tau_df))

    return n_vert, n_butt, n_cal
