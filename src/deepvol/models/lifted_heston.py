"""
lifted_heston.py — Exact Fourier-COS pricing for the Lifted Rough Heston model.

CPU reference implementation (SciPy BDF solver). Kept for validation / unit-test
comparison against the GPU RK4 version.  NOT used in production — use
lifted_heston_gpu.price_batch_gpu() instead.

Mathematical details of the Lifted Rough Heston model:
  - The Riccati ODE decay term includes the mean reversion parameter kappa.
  - Bernstein factor weights c are normalized (sum(c)=1).
  - The volatility state V is computed as the c-weighted aggregate sum_i c_i * psi_i.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Bernstein kernel factors  (must match GPU version)
# ---------------------------------------------------------------------------

def bernstein_factors(H: float, N: int = 20):
    """
    r_N = 1 + 10·N^{-0.9}
    x_i = r_N^{i-1-N/2}
    c_i = x_i^{-(H+0.5)},  then normalised: c_i /= sum(c)

    CRITICAL: normalise so sum(c)=1.  Without this, sum(c)≈26 and the
    Riccati quadratic term blows up by a factor of ~676.
    """
    r_N = 1.0 + 10.0 * (N ** -0.9)
    x = np.array([r_N ** (i - 1 - N / 2.0) for i in range(1, N + 1)])
    c = x ** -(H + 0.5)
    c = c / c.sum()   # Normalise weights
    return x, c


# ---------------------------------------------------------------------------
# COS payoff helpers
# ---------------------------------------------------------------------------

def _chi(a, b, c_lo, d_hi, k):
    kpi_ba = k * np.pi / (b - a)
    if k == 0:
        return np.exp(d_hi) - np.exp(c_lo)
    denom = 1.0 + kpi_ba**2
    return (np.cos(kpi_ba*(d_hi - a))*np.exp(d_hi)
            - np.cos(kpi_ba*(c_lo - a))*np.exp(c_lo)
            + kpi_ba*(np.sin(kpi_ba*(d_hi - a))*np.exp(d_hi)
                      - np.sin(kpi_ba*(c_lo - a))*np.exp(c_lo))) / denom


def _psi(a, b, c_lo, d_hi, k):
    if k == 0:
        return d_hi - c_lo
    kpi_ba = k * np.pi / (b - a)
    return (np.sin(kpi_ba*(d_hi - a)) - np.sin(kpi_ba*(c_lo - a))) / kpi_ba


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_call(S, K, T, sigma):
    if sigma < 1e-10 or T < 1e-10:
        return max(S - K, 0.0)
    d1 = (np.log(S/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*norm.cdf(d2)


def bs_vega(S, K, T, sigma):
    if sigma < 1e-10 or T < 1e-10:
        return 0.0
    d1 = (np.log(S/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    return S*np.sqrt(T)*norm.pdf(d1)


def implied_vol(price, S, K, T, max_iter=50):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-12:
        return np.nan
    if price >= S:
        return np.nan
    sigma = 0.3
    for _ in range(max_iter):
        p = bs_call(S, K, T, sigma)
        v = bs_vega(S, K, T, sigma)
        if abs(v) < 1e-15:
            break
        sigma -= (p - price) / v
        sigma = np.clip(sigma, 1e-6, 5.0)
        if abs(p - price) < 1e-10:
            break
    return sigma if 1e-6 < sigma < 5.0 else np.nan


# ---------------------------------------------------------------------------
# Lifted Heston Fourier-COS pricer
# ---------------------------------------------------------------------------

def price_iv_surface(params: dict, T_grid, K_grid, S0: float = 1.0,
                     N_factors: int = 20, N_cos: int = 64):
    """
    Price an IV surface using the Lifted Rough Heston model via Fourier-COS.

    Returns ndarray shape (nT, nK) of implied volatilities.
    NaN is returned for unquotable options (deep OTM / numerical failure).

    Note: This CPU implementation is slow (BDF solver on large system).
    Use price_batch_gpu() for production.
    """
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho   = params['rho']
    v0    = params['v0']

    x, c = bernstein_factors(params.get('H', 0.08), N_factors)  # c is normalised
    N = len(x)

    # COS domain [-4, 4]: The historical default was N_cos=64, but the current production
    # value is N_cos=128. This higher value is required due to the slowly decaying
    # characteristic function at very short maturities like T=0.1 and rough volatility
    # H=0.08, which causes N_cos=64 to truncate too early and produce a 264bp ATM error,
    # whereas N_cos=128 reduces this error to ~4bp.
    a, b = -4.0, 4.0
    k_arr = np.arange(N_cos)
    u_k   = k_arr * np.pi / (b - a)
    u_c   = u_k.astype(complex)

    # Vectorised RHS for SciPy BDF solver
    def rhs_vec(t, state):
        # State layout: [psi_real (N_cos*N), psi_imag (N_cos*N), int_cv_real (N_cos), int_cv_imag (N_cos)]
        psi_c = (state[:N_cos*N] + 1j * state[N_cos*N:2*N_cos*N]).reshape(N_cos, N)

        # c-weighted aggregate V(t) = sum_i c_i * psi_i
        V = psi_c @ c     # (N_cos,)  — matrix-vector with c as weights

        # Lifted Heston Riccati RHS: g(u, V)
        F = (-0.5*(u_c**2 + 1j*u_c)
             + rho * sigma * 1j * u_c * V
             + 0.5 * sigma**2 * V**2)

        # Include kappa in mean-reversion decay
        dpsi = F[:, None] - kappa * x[None, :] * psi_c   # (N_cos, N)

        # Accumulate int_0^t V(s) ds  (for kappa*theta*integral term)
        # V = c·psi already computed above
        return np.concatenate([
            dpsi.real.flatten(),
            dpsi.imag.flatten(),
            V.real.flatten(),    # d(int_V)/dt = V(t)
            V.imag.flatten(),
        ])

    # Initial state: psi(0) = 0, int_cv(0) = 0
    y0 = np.zeros(2 * N_cos * N + 2 * N_cos)
    sol = solve_ivp(rhs_vec, [0.0, float(np.max(T_grid))], y0,
                    method='BDF', t_eval=np.asarray(T_grid, dtype=float),
                    rtol=1e-5, atol=1e-7)

    # COS call payoff coefficients on [0, b]
    Vk = np.zeros(N_cos)
    for k in k_arr:
        Vk[k] = (2.0 / (b - a)) * (_chi(a, b, 0, b, k) - _psi(a, b, 0, b, k))
    Vk[0] *= 0.5

    iv_surface = np.full((len(T_grid), len(K_grid)), np.nan)

    for i, T in enumerate(T_grid):
        y_T       = sol.y[:, i]
        psi_T     = (y_T[:N_cos*N] + 1j * y_T[N_cos*N:2*N_cos*N]).reshape(N_cos, N)
        # c-weighted aggregate at terminal time T
        V_T       = psi_T @ c                                     # (N_cos,)
        int_cv_T  = y_T[2*N_cos*N:2*N_cos*N+N_cos] + 1j * y_T[2*N_cos*N+N_cos:]

        # log characteristic function (Abi Jaber 2022, Eq 4.2)
        log_phi   = v0 * V_T + kappa * theta * int_cv_T           # (N_cos,)
        phi_k     = np.exp(log_phi)
        phi_k[0]  = 1.0 + 0j                                      # martingale condition

        for j, log_moneyness in enumerate(K_grid):
            K = S0 * np.exp(log_moneyness)
            x0 = np.log(S0 / K)
            price = K * np.real(np.sum(phi_k * np.exp(1j * u_k * (x0 - a)) * Vk))
            price = max(price, max(S0 - K, 0.0))
            iv_surface[i, j] = implied_vol(price, S0, K, T)

    return iv_surface
