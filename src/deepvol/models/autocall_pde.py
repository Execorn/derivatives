"""
1D Crank-Nicolson Finite Difference PDE Reference Pricer for Autocallable Notes.

Solves the Black-Scholes PDE with local volatility via Crank-Nicolson backward
induction, applying discrete barrier jump conditions at observation dates.

Mathematical Formulation:
  dV/dt + 0.5 * sigma^2(t,S) * S^2 * d^2V/dS^2 + r * S * dV/dS - r * V = 0
  with Crank-Nicolson time-stepping and non-uniform grid spatial discretization.

References:
  - Crank, J., & Nicolson, P. (1947). A practical method for numerical evaluation of solutions of partial differential equations.
  - Tavella, D., & Randall, C. (2000). Pricing Financial Instruments: The Finite Difference Method. John Wiley & Sons.
"""

import sys
sys.path.insert(0, "src")

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.linalg import solve_banded


def build_lv_grid(
    S0: float,
    T: float,
    N_S: int,
    N_T: int,
    sigma_func: Callable[[float, np.ndarray], np.ndarray],
    sigma_atm: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precomputes log-uniform spot grid, uniform time grid, and local volatility matrix."""
    S_min = max(1e-3, S0 * np.exp(-6.0 * sigma_atm * np.sqrt(T)))
    S_max = S0 * np.exp(+6.0 * sigma_atm * np.sqrt(T))

    S_grid = np.exp(np.linspace(np.log(S_min), np.log(S_max), N_S))
    t_grid = np.linspace(0.0, T, N_T + 1)

    sigma_grid = np.zeros((N_T + 1, N_S), dtype=np.float64)
    for k in range(N_T + 1):
        vols = sigma_func(float(t_grid[k]), S_grid)
        sigma_grid[k] = np.clip(np.asarray(vols, dtype=np.float64), 0.01, 2.0)

    return S_grid, t_grid, sigma_grid


def cn_step(
    V: np.ndarray,
    S_grid: np.ndarray,
    dt: float,
    r: float,
    sigma_slice: np.ndarray,
    V_0_k: float,
    V_end_k: float,
) -> np.ndarray:
    """
    Performs one Crank-Nicolson backward time step from t_{k+1} to t_k.
    Solves (I - 0.5 * dt * L) V_new = (I + 0.5 * dt * L) V_old.
    """
    N_S = len(S_grid)
    h_minus = S_grid[1:-1] - S_grid[:-2]
    h_plus = S_grid[2:] - S_grid[1:-1]
    h_sum = h_minus + h_plus

    S_mid = S_grid[1:-1]
    sig_mid = sigma_slice[1:-1]
    sig2_S2 = (sig_mid ** 2) * (S_mid ** 2)

    # Spatial differential operator: L V_j = a_j V_{j-1} + b_j V_j + c_j V_{j+1}
    diff_coeff = 0.5 * sig2_S2
    drift_coeff = r * S_mid

    a = (2.0 * diff_coeff / (h_minus * h_sum)) - (drift_coeff / h_sum)
    c = (2.0 * diff_coeff / (h_plus * h_sum)) + (drift_coeff / h_sum)
    b = -(2.0 * diff_coeff / (h_plus * h_minus)) - r

    A_sub = -0.5 * dt * a
    A_diag = 1.0 - 0.5 * dt * b
    A_sup = -0.5 * dt * c

    # Right-hand side vector: (I + 0.5 * dt * L) V_old
    rhs = (
        (0.5 * dt * a) * V[:-2]
        + (1.0 + 0.5 * dt * b) * V[1:-1]
        + (0.5 * dt * c) * V[2:]
    )

    # Apply Dirichlet boundary conditions to RHS at t_k
    rhs[0] -= A_sub[0] * V_0_k
    rhs[-1] -= A_sup[-1] * V_end_k

    # Prepare banded matrix for scipy.linalg.solve_banded (1 upper, 1 lower diagonal)
    ab = np.zeros((3, N_S - 2), dtype=np.float64)
    ab[0, 1:] = A_sup[:-1]
    ab[1, :] = A_diag
    ab[2, :-1] = A_sub[1:]

    V_inner = solve_banded((1, 1), ab, rhs)

    V_new = np.empty_like(V)
    V_new[0] = V_0_k
    V_new[-1] = V_end_k
    V_new[1:-1] = V_inner
    return V_new


def price_autocall_pde(
    S0: float,
    r: float,
    T: float,
    N_S: int,
    N_T: int,
    obs_indices: List[int],
    B: float,
    coupon: float,
    sigma_func: Callable[[float, np.ndarray], np.ndarray],
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves Black-Scholes PDE backward in time to price autocallable note.
    Returns:
        (npv_grid, delta_grid, gamma_grid) at t=0 on S_grid.
    """
    S_grid, t_grid, sigma_grid = build_lv_grid(S0, T, N_S, N_T, sigma_func)
    dt = T / N_T
    barrier_level = B * S0
    obs_set = set(obs_indices)

    # Terminal condition at t = T: capital protection pays 1.0 (or call payoff if N_T is obs date)
    V = np.ones(N_S, dtype=np.float64)
    if N_T in obs_set:
        V[S_grid >= barrier_level] = 1.0 + coupon * T

    # Precompute observation times for upper boundary evaluation
    obs_times = [float(idx) * dt for idx in sorted(obs_indices)]

    def get_upper_bdry(t: float) -> float:
        future_obs = [to for to in obs_times if to >= t - 1e-8]
        if future_obs:
            t_next = future_obs[0]
            return float(np.exp(-r * (t_next - t)) * (1.0 + coupon * t_next))
        return float(np.exp(-r * (T - t)))

    # Backward induction from k = N_T - 1 down to 0
    for k in range(N_T - 1, -1, -1):
        t_k = float(t_grid[k])
        V_0_k = float(np.exp(-r * (T - t_k)))
        V_end_k = get_upper_bdry(t_k)
        sigma_slice = sigma_grid[k]

        # Crank-Nicolson backward step from t_{k+1} to t_k
        V = cn_step(V, S_grid, dt, r, sigma_slice, V_0_k, V_end_k)

        # If t_k is an observation date (k > 0), apply autocall barrier jump condition
        if k > 0 and k in obs_set:
            call_payoff = 1.0 + coupon * t_k
            V[S_grid >= barrier_level] = call_payoff

    # At t=0, compute spatial Greeks via central differences
    h_minus = S_grid[1:-1] - S_grid[:-2]
    h_plus = S_grid[2:] - S_grid[1:-1]
    h_sum = h_minus + h_plus

    delta_inner = (V[2:] - V[:-2]) / h_sum
    # Pad boundary deltas
    delta_0 = (V[1] - V[0]) / (S_grid[1] - S_grid[0])
    delta_end = (V[-1] - V[-2]) / (S_grid[-1] - S_grid[-2])
    delta_grid = np.concatenate([[delta_0], delta_inner, [delta_end]])

    gamma_inner = 2.0 * (
        (V[2:] - V[1:-1]) / h_plus - (V[1:-1] - V[:-2]) / h_minus
    ) / h_sum
    gamma_grid = np.concatenate([[0.0], gamma_inner, [0.0]])

    return V, delta_grid, gamma_grid


def price_autocall_pde_scalar(
    S0_val: float,
    r: float,
    T: float,
    N_S: int,
    N_T: int,
    obs_indices: List[int],
    B: float,
    coupon: float,
    sigma_func: Callable[[float, np.ndarray], np.ndarray],
    device: str = "cpu",
) -> Dict[str, Union[float, np.ndarray]]:
    """Convenience wrapper evaluating PDE at spot level S0_val."""
    npv_grid, delta_grid, gamma_grid = price_autocall_pde(
        S0=S0_val,
        r=r,
        T=T,
        N_S=N_S,
        N_T=N_T,
        obs_indices=obs_indices,
        B=B,
        coupon=coupon,
        sigma_func=sigma_func,
        device=device,
    )
    S_grid, _, _ = build_lv_grid(S0_val, T, N_S, N_T, sigma_func)

    npv_val = float(np.interp(S0_val, S_grid, npv_grid))
    delta_val = float(np.interp(S0_val, S_grid, delta_grid))
    gamma_val = float(np.interp(S0_val, S_grid, gamma_grid))

    return {
        "npv": npv_val,
        "delta": delta_val,
        "gamma": gamma_val,
        "S_grid": S_grid,
        "npv_grid": npv_grid,
        "delta_grid": delta_grid,
        "gamma_grid": gamma_grid,
    }

