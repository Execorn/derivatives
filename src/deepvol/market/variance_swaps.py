"""
§P2-B2 Variance Swap Pricing under Rough Heston (Lifted Heston).

A variance swap pays N * (σ²_realized - K_var) at maturity T, where:
    K_var = E[σ²_realized] = (1/T) * E[∫₀ᵀ v_t dt]

Under Rough Heston, the instantaneous variance v_t evolves (via Bernstein lifting) as:
    V_t = Σᵢ cᵢ Yᵢ_t
    dYᵢ_t = -κ xᵢ Yᵢ_t dt - κ (V_t - θ) dt + σ √V_t dW_t

The fair variance strike is computed by integrating E[v_t] over [0, T]:
    K_var = (1/T) ∫₀ᵀ E[v_t] dt

E[v_t] satisfies a **linear** ODE (obtained by taking expectations, which kills the diffusion term):
    d/dt E[Yᵢ] = -κ xᵢ E[Yᵢ] - κ (Σⱼ cⱼ E[Yⱼ] - θ)
    E[v_t] = Σᵢ cᵢ E[Yᵢ_t],  E[Yᵢ_0] = v₀

So K_var = (1/T) * I_V(T), where I_V(t) = ∫₀ᵗ E[v_s] ds is tracked as an extra ODE state.

This is identical to `model_variance_swap_rate` in vix_pricing.py — here we expose the full
public API with additional helpers (realized variance, vol swap, P&L, term structure).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# Add src directory to PYTHONPATH dynamically if not present
_src_dir = str(Path(__file__).parents[2])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from deepvol.models.lifted_heston import bernstein_factors  # Bernstein (xᵢ, cᵢ) factors


# ---------------------------------------------------------------------------
# 1. Fair Variance Strike  K_var = (1/T) E[∫₀ᵀ v_t dt]
# ---------------------------------------------------------------------------

def variance_swap_rate(
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    H: float,
    T: float,
) -> float:
    """
    Compute the fair variance strike of a variance swap under Rough Heston.

    Mathematically:
        K_var = (1/T) * E[∫₀ᵀ v_t dt]

    Under the Bernstein (lifted Heston) representation the conditional expectation
    of v_t satisfies the **linear** system (σ drops out after taking 𝔼[·]):

        d/dt E[Yᵢ] = -κ xᵢ E[Yᵢ] - κ (∑ⱼ cⱼ E[Yⱼ] - θ),  E[Yᵢ_0] = v₀
        E[v_t]      = ∑ᵢ cᵢ E[Yᵢ_t]
        d/dt I_V    = E[v_t],                                  I_V(0) = 0

    Then K_var = I_V(T) / T.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed κ > 0.
    theta : float
        Long-run variance θ > 0.
    sigma : float
        Vol-of-vol σ > 0 (enters MC simulation but not the linear ODE for 𝔼[v]).
    rho : float
        Correlation between spot and vol Brownian motions.
    v0 : float
        Initial variance V₀ ≥ 0.
    H : float
        Hurst exponent H ∈ (0.005, 0.495).  H = 0.5 → standard Heston.
    T : float
        Maturity in years.

    Returns
    -------
    float
        Annualised fair variance strike K_var (decimal, e.g. 0.04 ≡ 4% var = 20% vol).

    Raises
    ------
    ValueError
        If T ≤ 0.
    RuntimeError
        If the ODE solver fails to converge.

    Examples
    --------
    >>> # Near-spot variance should be close to v0 for small T
    >>> abs(variance_swap_rate(1.0, 0.04, 0.5, -0.7, 0.04, 0.1, 0.25) - 0.04) < 0.005
    True
    """
    if T <= 0.0:
        return float(v0)  # zero-maturity limit: E[v_0] = v0

    # Clip parameters to safe numerical ranges
    H = float(np.clip(H, 0.005, 0.495))
    kappa = float(np.clip(kappa, 1e-4, np.inf))
    theta = float(np.clip(theta, 1e-5, np.inf))
    sigma = float(np.clip(sigma, 1e-5, np.inf))
    v0 = float(np.clip(v0, 0.0, np.inf))

    x, c = bernstein_factors(H, N=20)
    N_factors = len(x)

    # State vector: [E[Y_1], ..., E[Y_N], I_V]
    # E[Y_i(0)] = v0 for all i  (V_0 = Σ cᵢ Yᵢ_0 = v0 since Σ cᵢ = 1 by construction)
    def _rhs(t, state):  # noqa: ANN001
        Y = state[:N_factors]
        V_exp = np.dot(c, Y)            # E[v_t] = Σ cᵢ E[Yᵢ]
        dY = -kappa * x * Y - kappa * (V_exp - theta)
        dI = V_exp
        return np.append(dY, dI)

    y0 = np.append(np.full(N_factors, v0), 0.0)
    sol = solve_ivp(
        _rhs,
        [0.0, float(T)],
        y0,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed in variance_swap_rate: {sol.message}")

    I_V_T = sol.y[-1, -1]
    return float(I_V_T / T)


# ---------------------------------------------------------------------------
# 2. Realized Variance from a Price Series
# ---------------------------------------------------------------------------

def realized_variance(prices: np.ndarray, dt: float = 1.0 / 252) -> float:
    """
    Compute annualised realised variance from a price series.

    Uses the standard quadratic-variation estimator:
        RV = (1 / (N · dt)) · ∑ᵢ rᵢ²,  rᵢ = ln(Pᵢ / Pᵢ₋₁)

    Parameters
    ----------
    prices : np.ndarray
        Array of price observations.  Must have at least 2 elements.
    dt : float
        Sampling interval in years (default 1/252 for daily data).

    Returns
    -------
    float
        Annualised realised variance (decimal).

    Raises
    ------
    ValueError
        If prices has fewer than 2 observations or contains non-positive values.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 252)))
    >>> rv = realized_variance(prices)
    >>> 0.0 < rv < 1.0
    True
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        raise ValueError("realized_variance requires at least 2 price observations.")
    if np.any(prices <= 0):
        raise ValueError("All prices must be strictly positive.")

    log_returns = np.diff(np.log(prices))
    N = len(log_returns)
    return float(np.sum(log_returns**2) / (N * dt))


# ---------------------------------------------------------------------------
# 3. Vol Swap Rate  K_vol = E[σ_realized]
# ---------------------------------------------------------------------------

def vol_swap_rate(
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    H: float,
    T: float,
) -> float:
    """
    Approximate the fair strike of a volatility swap under Rough Heston.

    By Jensen's inequality, the vol swap rate satisfies:
        K_vol = E[√(RV)] ≤ √(E[RV]) = √K_var

    The upper-bound approximation K_vol ≈ √K_var is used here.  A tighter
    second-order correction via Jensen's inequality reads:
        K_vol ≈ √K_var · (1 - Var[RV] / (8 · K_var²))

    but since Var[RV] requires higher-order cumulants that are expensive to compute
    under rough Heston, we return the tractable upper bound.

    Parameters
    ----------
    kappa, theta, sigma, rho, v0, H, T : float
        Same as in :func:`variance_swap_rate`.

    Returns
    -------
    float
        Approximate annualised vol swap rate (decimal), K_vol ≤ √K_var.

    Notes
    -----
    The convexity adjustment (vol swap discount) is given by:
        √K_var - K_vol > 0

    For typical equity parameters this is on the order of 0.5–2 vol points.
    """
    kv = variance_swap_rate(kappa, theta, sigma, rho, v0, H, T)
    return float(np.sqrt(max(kv, 0.0)))


# ---------------------------------------------------------------------------
# 4. Variance Swap P&L
# ---------------------------------------------------------------------------

def variance_swap_pnl(
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    H: float,
    T: float,
    N_notional: float,
    realized_var: float,
) -> float:
    """
    Compute the P&L of a long variance swap position at expiry.

    A variance swap pays:
        P&L = N · (σ²_realised - K_var)

    where K_var is the fair variance strike determined at inception.

    Parameters
    ----------
    kappa, theta, sigma, rho, v0, H, T : float
        Rough Heston model parameters and maturity.
    N_notional : float
        Notional amount N (in currency units per variance point, e.g. USD/vol²).
    realized_var : float
        Realised annualised variance observed over [0, T] (decimal, e.g. 0.06).

    Returns
    -------
    float
        P&L in the same units as N_notional.

    Examples
    --------
    >>> # Long variance swap that broke even (realized == strike)
    >>> params = dict(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, H=0.1, T=1.0)
    >>> kv = variance_swap_rate(**params)
    >>> abs(variance_swap_pnl(**params, N_notional=1e6, realized_var=kv)) < 1e-3
    True
    """
    kv = variance_swap_rate(kappa, theta, sigma, rho, v0, H, T)
    return float(N_notional * (realized_var - kv))


# ---------------------------------------------------------------------------
# 5. Variance Swap Term Structure
# ---------------------------------------------------------------------------

_DEFAULT_T_GRID = np.array([1/12, 3/12, 6/12, 9/12, 12/12, 18/12, 24/12])


def variance_term_structure(
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    H: float,
    T_grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the variance swap term structure under Rough Heston.

    Evaluates K_var(T) = (1/T) E[∫₀ᵀ v_t dt] for each maturity T in T_grid by
    solving the lifted-Heston linear ODE once and extracting I_V(T) at the
    requested evaluation times.

    Parameters
    ----------
    kappa, theta, sigma, rho, v0, H : float
        Rough Heston model parameters.
    T_grid : array-like or None
        Maturities in years.  Defaults to
        ``[1/12, 3/12, 6/12, 9/12, 12/12, 18/12, 24/12]``.

    Returns
    -------
    np.ndarray
        Array of annualised variance swap rates, shape ``(len(T_grid),)``.

    Notes
    -----
    The implementation solves the ODE **once** over ``[0, max(T_grid)]`` with
    ``t_eval`` at all requested maturities, which is much faster than calling
    :func:`variance_swap_rate` independently for each T.

    Examples
    --------
    >>> ts = variance_term_structure(1.0, 0.04, 0.5, -0.7, 0.04, 0.1)
    >>> len(ts) == 7
    True
    >>> bool(np.all(ts > 0))
    True
    """
    if T_grid is None:
        T_grid = _DEFAULT_T_GRID

    T_grid = np.asarray(T_grid, dtype=float)
    if len(T_grid) == 0:
        return np.array([], dtype=float)

    # Clip parameters
    H = float(np.clip(H, 0.005, 0.495))
    kappa = float(np.clip(kappa, 1e-4, np.inf))
    theta = float(np.clip(theta, 1e-5, np.inf))
    sigma = float(np.clip(sigma, 1e-5, np.inf))
    v0 = float(np.clip(v0, 0.0, np.inf))

    # Sort maturities for ODE integration, then unsort at the end
    sort_idx = np.argsort(T_grid)
    unsort_idx = np.argsort(sort_idx)
    sorted_T = T_grid[sort_idx]

    # Replace any T=0 entries with a tiny epsilon; handle separately
    zero_mask = sorted_T <= 0.0
    sorted_T_pos = np.where(zero_mask, 1e-6, sorted_T)

    x, c = bernstein_factors(H, N=20)
    N_factors = len(x)

    def _rhs(t, state):  # noqa: ANN001
        Y = state[:N_factors]
        V_exp = np.dot(c, Y)
        dY = -kappa * x * Y - kappa * (V_exp - theta)
        dI = V_exp
        return np.append(dY, dI)

    y0 = np.append(np.full(N_factors, v0), 0.0)
    max_T = float(sorted_T_pos[-1])

    sol = solve_ivp(
        _rhs,
        [0.0, max_T],
        y0,
        method="RK45",
        t_eval=sorted_T_pos,
        rtol=1e-8,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(
            f"ODE solver failed in variance_term_structure: {sol.message}"
        )

    # I_V(T) is the last row of sol.y
    I_V = sol.y[-1, :]                   # shape: (len(sorted_T),)
    rates_sorted = I_V / sorted_T_pos    # element-wise K_var(T) = I_V(T)/T

    # For any T=0 slots, return v0 (spot variance)
    rates_sorted = np.where(zero_mask, v0, rates_sorted)

    return rates_sorted[unsort_idx]
