"""
Option portfolio stress-testing scenario generator and Delta-Vega stress grid generation.
"""

from __future__ import annotations
import math
import numpy as np
import torch
from typing import Dict, List, Union, Tuple, Optional

# Pre-defined historical stress scenarios (Black Monday, Lehman, COVID, Flash Crash)
HISTORICAL_SCENARIOS = {
    "Black Monday 1987": {
        "spot_shift": -0.226,
        "flat_shift": 0.25,
        "skew_shift": -0.05,
        "term_shift": 0.15,
        "term_decay": 1.5,
        "description": "Replay of October 19, 1987 crash: spot crash -22.6%, vol spike +25% abs."
    },
    "Lehman Default 2008": {
        "spot_shift": -0.150,
        "flat_shift": 0.30,
        "skew_shift": -0.08,
        "term_shift": 0.20,
        "term_decay": 1.0,
        "description": "Replay of September 2008 Lehman crisis: spot crash -15.0%, vol spike +30% abs."
    },
    "COVID-19 Crash 2020": {
        "spot_shift": -0.120,
        "flat_shift": 0.20,
        "skew_shift": -0.04,
        "term_shift": 0.15,
        "term_decay": 2.0,
        "description": "Replay of March 2020 liquidity panic: spot crash -12.0%, vol spike +20% abs."
    },
    "Flash Crash 2010": {
        "spot_shift": -0.090,
        "flat_shift": 0.15,
        "skew_shift": -0.03,
        "term_shift": 0.10,
        "term_decay": 3.0,
        "description": "Replay of May 6, 2010 intraday crash: spot crash -9.0%, vol spike +15% abs."
    }
}


def apply_surface_shifts(
    T_grid: Union[np.ndarray, torch.Tensor],
    K_grid: Union[np.ndarray, torch.Tensor],
    iv_surface: Union[np.ndarray, torch.Tensor],
    flat_shift: float = 0.0,
    skew_shift: float = 0.0,
    term_shift: float = 0.0,
    term_decay: float = 1.0,
    S_ref: Optional[float] = None,
    min_vol: float = 1e-4,
    sota_mode: bool = False,
    skew_steepness: float = 2.0,
    skew_decay: float = 1.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply stress shifts (flat, skew rotation, term structure spike) to an implied volatility surface.

    Formulas:
        If sota_mode is False:
            d_flat(T, K) = flat_shift
            d_skew(T, K) = skew_shift * ln(K / S_ref)
            d_term(T, K) = term_shift * exp(-term_decay * T)
            Sigma_stressed(T, K) = max(Sigma_0(T, K) + d_flat + d_skew + d_term, min_vol)

        If sota_mode is True (SOTA Eq. 18):
            d_flat(T, K) = flat_shift * exp(-term_decay * T)
            d_skew(T, K) = skew_shift * [ tanh(-skew_steepness * k) * (1 - sgn(k)) ] * exp(-skew_decay * T)
            Sigma_stressed(T, K) = max(Sigma_0(T, K) + d_flat + d_skew, min_vol)

    Args:
        T_grid: Maturing grid, shape (nT,)
        K_grid: Strike (or log-moneyness) grid, shape (nK,)
        iv_surface: Implied vol surface, shape (nT, nK)
        flat_shift: Parallel shift in volatility (absolute) or ATM vol spike scale
        skew_shift: Skew twist/rotation coefficient or scale of asymmetric skew
        term_shift: Term structure shift coefficient (short-term spike) [not used in sota_mode]
        term_decay: Exponential decay parameter for term structure shift
        S_ref: Reference spot price to convert strike grid to log-moneyness
        min_vol: Lower bound to keep volatility strictly positive and stable
        sota_mode: Enable the SOTA asymmetric joint shift formulation
        skew_steepness: Steepness of the left-wing skew (beta in Eq. 18)
        skew_decay: Maturity decay of the skew deformation (gamma_T in Eq. 18)

    Returns:
        Stressed implied volatility surface of the same type and shape as iv_surface.
    """
    is_torch = isinstance(iv_surface, torch.Tensor)

    # Convert NumPy arrays to torch tensors internally if needed for unified computation
    if is_torch:
        device = iv_surface.device
        dtype = iv_surface.dtype
        T_t = torch.as_tensor(T_grid, device=device, dtype=dtype)
        K_t = torch.as_tensor(K_grid, device=device, dtype=dtype)
        iv_t = iv_surface
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        T_t = torch.tensor(T_grid, device=device, dtype=dtype)
        K_t = torch.tensor(K_grid, device=device, dtype=dtype)
        iv_t = torch.tensor(iv_surface, device=device, dtype=dtype)

    # Determine reference log-moneyness
    if S_ref is None:
        if torch.any(K_t > 2.0):
            S_ref = float(torch.median(K_t).item())
        else:
            S_ref = 1.0

    if torch.any(K_t > 2.0):
        k_t = torch.log(K_t / S_ref)
    else:
        k_t = K_t

    if sota_mode:
        # SOTA Shift Formula (Eq. 18)
        # d_flat = flat_shift * exp(-term_decay * T) -> shape (nT, 1)
        shift_flat = (flat_shift * torch.exp(-term_decay * T_t)).unsqueeze(1)

        # d_skew = skew_shift * [ tanh(-skew_steepness * k) * (1 - sgn(k)) ] * exp(-skew_decay * T)
        sgn_k = torch.sign(k_t)
        skew_factor = torch.tanh(-skew_steepness * k_t) * (1.0 - sgn_k)
        decay_factor = torch.exp(-skew_decay * T_t)

        # Outer product to shape (nT, nK)
        shift_skew = skew_shift * (decay_factor.unsqueeze(1) @ skew_factor.unsqueeze(0))

        stressed_iv = iv_t + shift_flat + shift_skew
    else:
        # Standard Shifts
        shift_flat = flat_shift
        shift_skew = skew_shift * k_t.unsqueeze(0)
        shift_term = (term_shift * torch.exp(-term_decay * T_t)).unsqueeze(1)

        stressed_iv = iv_t + shift_flat + shift_skew + shift_term

    stressed_iv = torch.clamp(stressed_iv, min=min_vol)

    if is_torch:
        return stressed_iv
    else:
        return stressed_iv.cpu().numpy().astype(iv_surface.dtype)


def _interpolate_bilinear_torch(
    T_grid: torch.Tensor,
    K_grid: torch.Tensor,
    iv_surface: torch.Tensor,
    T: torch.Tensor,
    k: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized 2D bilinear interpolation for a query tensor (T, k)
    on a grid (T_grid, K_grid) with values iv_surface of shape (nT, nK).
    """
    nT = T_grid.size(0)
    nK = K_grid.size(0)

    # Clamping query points with margin to prevent out-of-bounds indexing
    T_clip = torch.clamp(T, min=T_grid[0] + 1e-4, max=T_grid[-1] - 1e-4)
    k_clip = torch.clamp(k, min=K_grid[0] + 1e-4, max=K_grid[-1] - 1e-4)

    t_idx = torch.bucketize(T_clip, T_grid) - 1
    t_idx = torch.clamp(t_idx, min=0, max=nT - 2)

    k_idx = torch.bucketize(k_clip, K_grid) - 1
    k_idx = torch.clamp(k_idx, min=0, max=nK - 2)

    t0 = T_grid[t_idx]
    t1 = T_grid[t_idx + 1]
    k0 = K_grid[k_idx]
    k1 = K_grid[k_idx + 1]

    dt = torch.clamp(t1 - t0, min=1e-8)
    dk = torch.clamp(k1 - k0, min=1e-8)

    wt = (T_clip - t0) / dt
    wk = (k_clip - k0) / dk

    # Gather surface values using advanced indexing
    val00 = iv_surface[t_idx, k_idx]
    val10 = iv_surface[t_idx + 1, k_idx]
    val01 = iv_surface[t_idx, k_idx + 1]
    val11 = iv_surface[t_idx + 1, k_idx + 1]

    val = (1.0 - wt) * (1.0 - wk) * val00 + \
          wt * (1.0 - wk) * val10 + \
          (1.0 - wt) * wk * val01 + \
          wt * wk * val11

    return val


def bs_call_price_batch(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized Black-Scholes call option pricer.
    Inputs must broadcast to the same shape.
    """
    S_safe = S.clamp(min=1e-8)
    K_safe = K.clamp(min=1e-8)
    sigma_safe = sigma.clamp(min=1e-8)
    T_safe = T.clamp(min=1e-8)

    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=S.dtype, device=S.device),
        torch.tensor(1.0, dtype=S.dtype, device=S.device)
    )

    d1 = (torch.log(S_safe / K_safe) + (r + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * torch.sqrt(T_safe))
    d2 = d1 - sigma_safe * torch.sqrt(T_safe)

    call = S_safe * normal.cdf(d1) - K_safe * torch.exp(-r * T_safe) * normal.cdf(d2)
    intrinsic = torch.clamp(S - K, min=0.0)

    return torch.where((T <= 0.0) | (sigma <= 0.0), intrinsic, call)


def stress_portfolio(
    positions: List[Dict],
    S0: float,
    r: float,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    iv_surface: Union[np.ndarray, torch.Tensor],
    spot_shift: float = 0.0,
    flat_shift: float = 0.0,
    skew_shift: float = 0.0,
    term_shift: float = 0.0,
    term_decay: float = 1.0,
    min_vol: float = 1e-4,
    sota_mode: bool = False,
    skew_steepness: float = 2.0,
    skew_decay: float = 1.0
) -> Dict[str, float]:
    """
    Stress-test an option portfolio under a spot shift and surface shift scenario.
    """
    valid_positions = []
    for pos in positions:
        K_pos = float(pos.get("K", 0.0))
        T_pos = float(pos.get("T", 0.0))
        qty = float(pos.get("quantity", 1.0))
        notional = float(pos.get("notional", 100.0))
        opt_type = pos.get("type", "call").lower()

        if opt_type not in ["call", "put"]:
            raise ValueError(f"Unsupported option type: {opt_type}")

        if math.isfinite(K_pos) and K_pos > 0 and math.isfinite(T_pos) and T_pos > 0 and math.isfinite(qty):
            valid_positions.append({
                "K": K_pos,
                "T": T_pos,
                "quantity": qty,
                "notional": notional,
                "is_call": 1.0 if opt_type == "call" else 0.0
            })

    if not valid_positions:
        return {
            "baseline_price": 0.0,
            "stressed_price": 0.0,
            "portfolio_pnl": 0.0
        }

    if isinstance(iv_surface, torch.Tensor):
        device = iv_surface.device
        dtype = iv_surface.dtype
        iv_t = iv_surface
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        iv_t = torch.tensor(iv_surface, device=device, dtype=dtype)

    T_grid_t = torch.as_tensor(T_grid, device=device, dtype=dtype)
    K_grid_t = torch.as_tensor(K_grid, device=device, dtype=dtype)

    K_p = torch.tensor([p["K"] for p in valid_positions], device=device, dtype=dtype)
    T_p = torch.tensor([p["T"] for p in valid_positions], device=device, dtype=dtype)
    qty_p = torch.tensor([p["quantity"] for p in valid_positions], device=device, dtype=dtype)
    notional_p = torch.tensor([p["notional"] for p in valid_positions], device=device, dtype=dtype)
    is_call_p = torch.tensor([p["is_call"] for p in valid_positions], device=device, dtype=dtype)

    r_t = torch.tensor(r, device=device, dtype=dtype)
    S0_t = torch.tensor(S0, device=device, dtype=dtype)

    # Compute baseline pricing
    with torch.no_grad():
        k0_p = torch.log(K_p / S0_t)
        sigma0_p = _interpolate_bilinear_torch(T_grid_t, K_grid_t, iv_t, T_p, k0_p)
        sigma0_p = torch.clamp(sigma0_p, min=min_vol)

        call_prices0 = bs_call_price_batch(S0_t, K_p, T_p, r_t, sigma0_p)
        put_prices0 = call_prices0 + K_p * torch.exp(-r_t * T_p) - S0_t
        prices0 = torch.where(is_call_p == 1.0, call_prices0, put_prices0)
        baseline_price = torch.sum(prices0 * qty_p * notional_p).item()

    # Compute stressed pricing
    with torch.no_grad():
        S_stressed_t = S0_t * (1.0 + spot_shift)

        # Get baseline volatility at spot-shifted log-moneyness
        k_stressed_p = torch.log(K_p / S_stressed_t)
        sigma_base_p = _interpolate_bilinear_torch(T_grid_t, K_grid_t, iv_t, T_p, k_stressed_p)

        # Apply shifts analytically for each position
        if sota_mode:
            # SOTA asymmetric shifts
            shift_flat = flat_shift * torch.exp(-term_decay * T_p)

            # Left-wing skew rotation
            k_ref_p = torch.log(K_p / S0_t)
            sgn_k = torch.sign(k_ref_p)
            shift_skew = skew_shift * torch.tanh(-skew_steepness * k_ref_p) * (1.0 - sgn_k) * torch.exp(-skew_decay * T_p)

            sigma_stressed_p = sigma_base_p + shift_flat + shift_skew
        else:
            # Standard shifts
            sigma_stressed_p = (
                sigma_base_p
                + flat_shift
                + skew_shift * torch.log(K_p / S0_t)
                + term_shift * torch.exp(-term_decay * T_p)
            )

        sigma_stressed_p = torch.clamp(sigma_stressed_p, min=min_vol)

        call_prices_stressed = bs_call_price_batch(S_stressed_t, K_p, T_p, r_t, sigma_stressed_p)
        put_prices_stressed = call_prices_stressed + K_p * torch.exp(-r_t * T_p) - S_stressed_t
        prices_stressed = torch.where(is_call_p == 1.0, call_prices_stressed, put_prices_stressed)
        stressed_price = torch.sum(prices_stressed * qty_p * notional_p).item()

    portfolio_pnl = stressed_price - baseline_price

    return {
        "baseline_price": baseline_price,
        "stressed_price": stressed_price,
        "portfolio_pnl": portfolio_pnl
    }


def generate_stress_grid(
    positions: List[Dict],
    S0: float,
    r: float,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    iv_surface: Union[np.ndarray, torch.Tensor],
    spot_shifts: Union[List[float], np.ndarray],
    vol_shifts: Union[List[float], np.ndarray],
    min_vol: float = 1e-4
) -> Tuple[np.ndarray, float]:
    """
    Generate a 2D Delta-Vega stress grid (spot shifts vs flat vol shifts)
    using a fully vectorized GPU/CPU PyTorch batch calculation.
    """
    valid_positions = []
    for pos in positions:
        K_pos = float(pos.get("K", 0.0))
        T_pos = float(pos.get("T", 0.0))
        qty = float(pos.get("quantity", 1.0))
        notional = float(pos.get("notional", 100.0))
        opt_type = pos.get("type", "call").lower()

        if opt_type not in ["call", "put"]:
            raise ValueError(f"Unsupported option type: {opt_type}")

        if math.isfinite(K_pos) and K_pos > 0 and math.isfinite(T_pos) and T_pos > 0 and math.isfinite(qty):
            valid_positions.append({
                "K": K_pos,
                "T": T_pos,
                "quantity": qty,
                "notional": notional,
                "is_call": 1.0 if opt_type == "call" else 0.0
            })

    M = len(spot_shifts)
    N = len(vol_shifts)

    if not valid_positions:
        return np.zeros((M, N), dtype=np.float32), 0.0

    if isinstance(iv_surface, torch.Tensor):
        device = iv_surface.device
        dtype = iv_surface.dtype
        iv_t = iv_surface
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        iv_t = torch.tensor(iv_surface, device=device, dtype=dtype)

    T_grid_t = torch.as_tensor(T_grid, device=device, dtype=dtype)
    K_grid_t = torch.as_tensor(K_grid, device=device, dtype=dtype)

    spot_shifts_t = torch.tensor(spot_shifts, device=device, dtype=dtype)
    vol_shifts_t = torch.tensor(vol_shifts, device=device, dtype=dtype)

    K_p = torch.tensor([p["K"] for p in valid_positions], device=device, dtype=dtype)
    T_p = torch.tensor([p["T"] for p in valid_positions], device=device, dtype=dtype)
    qty_p = torch.tensor([p["quantity"] for p in valid_positions], device=device, dtype=dtype)
    notional_p = torch.tensor([p["notional"] for p in valid_positions], device=device, dtype=dtype)
    is_call_p = torch.tensor([p["is_call"] for p in valid_positions], device=device, dtype=dtype)

    r_t = torch.tensor(r, device=device, dtype=dtype)
    S0_t = torch.tensor(S0, device=device, dtype=dtype)

    # Compute baseline pricing
    with torch.no_grad():
        k0_p = torch.log(K_p / S0_t)
        sigma0_p = _interpolate_bilinear_torch(T_grid_t, K_grid_t, iv_t, T_p, k0_p)
        sigma0_p = torch.clamp(sigma0_p, min=min_vol)

        call_prices0 = bs_call_price_batch(S0_t, K_p, T_p, r_t, sigma0_p)
        put_prices0 = call_prices0 + K_p * torch.exp(-r_t * T_p) - S0_t
        prices0 = torch.where(is_call_p == 1.0, call_prices0, put_prices0)
        baseline_price = torch.sum(prices0 * qty_p * notional_p).item()

    # Vectorized Stress Grid Calculation
    with torch.no_grad():
        S_stressed = S0_t * (1.0 + spot_shifts_t)

        k_stressed_pm = torch.log(K_p.unsqueeze(1) / S_stressed.unsqueeze(0))
        T_pm = T_p.unsqueeze(1).expand(-1, M)

        sigma_base_pm = _interpolate_bilinear_torch(T_grid_t, K_grid_t, iv_t, T_pm, k_stressed_pm)

        sigma_stressed_pmn = sigma_base_pm.unsqueeze(2) + vol_shifts_t.unsqueeze(0).unsqueeze(1)
        sigma_stressed_pmn = torch.clamp(sigma_stressed_pmn, min=min_vol)

        S_pmn = S_stressed.unsqueeze(0).unsqueeze(2).expand(len(valid_positions), -1, N)
        K_pmn = K_p.unsqueeze(1).unsqueeze(2).expand(-1, M, N)
        T_pmn = T_p.unsqueeze(1).unsqueeze(2).expand(-1, M, N)
        r_pmn = r_t.expand_as(S_pmn)

        call_prices_pmn = bs_call_price_batch(S_pmn, K_pmn, T_pmn, r_pmn, sigma_stressed_pmn)
        put_prices_pmn = call_prices_pmn + K_pmn * torch.exp(-r_pmn * T_pmn) - S_pmn

        is_call_pmn = is_call_p.unsqueeze(1).unsqueeze(2).expand(-1, M, N)
        prices_pmn = torch.where(is_call_pmn == 1.0, call_prices_pmn, put_prices_pmn)

        qty_pmn = qty_p.unsqueeze(1).unsqueeze(2).expand(-1, M, N)
        notional_pmn = notional_p.unsqueeze(1).unsqueeze(2).expand(-1, M, N)
        weighted_prices_pmn = prices_pmn * qty_pmn * notional_pmn

        portfolio_prices_mn = torch.sum(weighted_prices_pmn, dim=0)
        grid_pnl_mn = portfolio_prices_mn - baseline_price
        grid_pnl = grid_pnl_mn.cpu().numpy()

    return grid_pnl, baseline_price


class OptionPortfolioStressTester:
    """
    Class wrapper for option portfolio stress-testing operations.
    """
    def __init__(
        self,
        positions: List[Dict],
        S0: float,
        r: float,
        T_grid: np.ndarray,
        K_grid: np.ndarray,
        iv_surface: Union[np.ndarray, torch.Tensor]
    ):
        self.positions = positions
        self.S0 = S0
        self.r = r
        self.T_grid = T_grid
        self.K_grid = K_grid
        self.iv_surface = iv_surface

    def stress_scenario(
        self,
        spot_shift: float = 0.0,
        flat_shift: float = 0.0,
        skew_shift: float = 0.0,
        term_shift: float = 0.0,
        term_decay: float = 1.0,
        min_vol: float = 1e-4,
        sota_mode: bool = False,
        skew_steepness: float = 2.0,
        skew_decay: float = 1.0
    ) -> Dict[str, float]:
        return stress_portfolio(
            positions=self.positions,
            S0=self.S0,
            r=self.r,
            T_grid=self.T_grid,
            K_grid=self.K_grid,
            iv_surface=self.iv_surface,
            spot_shift=spot_shift,
            flat_shift=flat_shift,
            skew_shift=skew_shift,
            term_shift=term_shift,
            term_decay=term_decay,
            min_vol=min_vol,
            sota_mode=sota_mode,
            skew_steepness=skew_steepness,
            skew_decay=skew_decay
        )

    def historical_replay(self, scenario_name: str) -> Dict[str, Union[float, str]]:
        if scenario_name not in HISTORICAL_SCENARIOS:
            raise ValueError(f"Historical scenario '{scenario_name}' not found. "
                             f"Available options: {list(HISTORICAL_SCENARIOS.keys())}")

        scenario = HISTORICAL_SCENARIOS[scenario_name]
        res = self.stress_scenario(
            spot_shift=scenario["spot_shift"],
            flat_shift=scenario["flat_shift"],
            skew_shift=scenario["skew_shift"],
            term_shift=scenario["term_shift"],
            term_decay=scenario["term_decay"]
        )

        return {
            "scenario_name": scenario_name,
            "description": scenario["description"],
            **res
        }

    def delta_vega_grid(
        self,
        spot_shifts: Union[List[float], np.ndarray],
        vol_shifts: Union[List[float], np.ndarray],
        min_vol: float = 1e-4
    ) -> Tuple[np.ndarray, float]:
        return generate_stress_grid(
            positions=self.positions,
            S0=self.S0,
            r=self.r,
            T_grid=self.T_grid,
            K_grid=self.K_grid,
            iv_surface=self.iv_surface,
            spot_shifts=spot_shifts,
            vol_shifts=vol_shifts,
            min_vol=min_vol
        )
