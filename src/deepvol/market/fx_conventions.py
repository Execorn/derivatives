from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple, cast

import scipy.optimize
import torch
from torch import Tensor


class DeltaConvention(str, Enum):
    SPOT_PNA = "spot_pna"
    SPOT_PA = "spot_pa"
    FORWARD_PNA = "forward_pna"
    FORWARD_PA = "forward_pa"


class AtmConvention(str, Enum):
    DNS = "DNS"
    ATMF = "ATMF"


class StrangleConvention(str, Enum):
    SMILE = "SMILE"
    MARKET = "MARKET"


class ConventionSolveError(RuntimeError):
    """Raised when market convention strike or strangle inversion fails to converge."""


@dataclass(frozen=True)
class LegMarket:
    spot: Tensor
    df_foreign: Tensor
    df_domestic: Tensor
    T: Tensor

    @property
    def forward(self) -> Tensor:
        return self.spot * self.df_foreign / self.df_domestic


def fx_delta(
    K: Tensor,
    sigma: Tensor,
    leg: LegMarket,
    phi: Tensor,
    conv: DeltaConvention,
) -> Tensor:
    """Evaluates option delta across market quoting conventions.

    Args:
        K: Strike tensor of shape (S,) or broadcastable, float64.
        sigma: Volatility tensor of shape (S,) or broadcastable, float64.
        leg: LegMarket container holding spot, discount factors, and maturity T.
        phi: Call (+1.0) / Put (-1.0) indicator tensor of shape (S,), float64.
        conv: Delta convention enum.

    Returns:
        Delta tensor of shape (S,), float64.
    """
    F = leg.forward
    tau = sigma * torch.sqrt(leg.T)
    d1 = (torch.log(F / K) + 0.5 * (tau ** 2)) / tau
    d2 = d1 - tau

    N1 = torch.special.ndtr(phi * d1)
    N2 = torch.special.ndtr(phi * d2)

    if conv == DeltaConvention.FORWARD_PNA:
        return cast(Tensor, phi * N1)
    elif conv == DeltaConvention.SPOT_PNA:
        return cast(Tensor, phi * leg.df_foreign * N1)
    elif conv == DeltaConvention.FORWARD_PA:
        return cast(Tensor, phi * (K / F) * N2)
    elif conv == DeltaConvention.SPOT_PA:
        return cast(Tensor, phi * leg.df_foreign * (K / F) * N2)
    else:
        raise ValueError(f"Unsupported delta convention: {conv}")


def pa_call_delta_argmax(
    sigma: Tensor,
    leg: LegMarket,
    n_iter: int = 30,
) -> Tensor:
    """Solves for strike K_max maximizing premium-adjusted call delta via sigma sqrt(T) N(d2) = n(d2).

    Args:
        sigma: Volatility tensor of shape (S,), float64.
        leg: LegMarket container.
        n_iter: Fixed Newton iteration count.

    Returns:
        K_max strike tensor of shape (S,), float64.
    """
    F = leg.forward
    tau = sigma * torch.sqrt(leg.T)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    # Initial guess for z = d2 solving n(z) / N(z) = tau
    z = torch.where(tau > 1.0, -tau + 1.0 / tau, 0.54 - 0.68 * tau)

    for _ in range(n_iter):
        Nz = torch.special.ndtr(z)
        nz = inv_sqrt_2pi * torch.exp(-0.5 * (z ** 2))
        m = nz / torch.clamp(Nz, min=1e-300)
        # m'(z) = -m(z) * (z + m(z)) (Sampford identity for hazard rate)
        dm = -m * (z + m)
        step = (m - tau) / dm
        z = z - step

    # d2 = (ln(F / K_max) - 0.5 * tau^2) / tau -> K_max = F * exp(-z * tau - 0.5 * tau^2)
    return F * torch.exp(-z * tau - 0.5 * (tau ** 2))


def strike_from_delta(
    delta: Tensor,
    sigma: Tensor,
    leg: LegMarket,
    phi: Tensor,
    conv: DeltaConvention,
    *,
    n_iter: int = 12,
    tol: float = 1e-12,
) -> Tensor:
    """Inverts option delta to strike K across market quoting conventions.

    Args:
        delta: Delta target tensor of shape (S,), float64.
        sigma: Volatility tensor of shape (S,), float64.
        leg: LegMarket container.
        phi: Call (+1.0) / Put (-1.0) tensor of shape (S,), float64.
        conv: Delta convention enum.
        n_iter: Fixed Newton iteration count for premium-adjusted conventions.
        tol: Convergence residual tolerance.

    Returns:
        Strike price tensor K of shape (S,), float64.
    """
    F = leg.forward
    tau = sigma * torch.sqrt(leg.T)

    if conv in (DeltaConvention.SPOT_PNA, DeltaConvention.SPOT_PA):
        delta_fwd = delta / leg.df_foreign
    else:
        delta_fwd = delta

    if conv in (DeltaConvention.FORWARD_PNA, DeltaConvention.SPOT_PNA):
        # Non-adjusted (PNA) closed form inversion
        val_pna = torch.clamp(phi * delta_fwd, min=1e-15, max=1.0 - 1e-15)
        d1_pna = phi * torch.special.ndtri(val_pna)
        K = F * torch.exp(-d1_pna * tau + 0.5 * (tau ** 2))
    else:
        # Premium-adjusted (PA) bracketed Newton-bisection solver
        K_max = pa_call_delta_argmax(sigma, leg)
        val_pna_call = torch.clamp(delta_fwd, min=1e-15, max=1.0 - 1e-15)
        K_pna_call = F * torch.exp(-torch.special.ndtri(val_pna_call) * tau + 0.5 * (tau ** 2))
        u_lo_call = torch.log(K_max / F)
        u_hi_call = torch.log(K_pna_call / F)

        val_pna_put = torch.clamp(-delta_fwd, min=1e-15, max=1.0 - 1e-15)
        K_pna_put = F * torch.exp(torch.special.ndtri(val_pna_put) * tau + 0.5 * (tau ** 2))
        u_lo_put = torch.log(K_pna_put / F) - 8.0 * tau
        u_hi_put = torch.log(K_pna_put / F)

        u_lo = torch.where(phi > 0.0, u_lo_call, u_lo_put)
        u_hi = torch.where(phi > 0.0, u_hi_call, u_hi_put)

        u = 0.5 * (u_lo + u_hi)
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

        for _ in range(n_iter):
            d2 = (-u - 0.5 * (tau ** 2)) / tau
            N2 = torch.special.ndtr(phi * d2)
            n2 = inv_sqrt_2pi * torch.exp(-0.5 * (d2 ** 2))
            delta_curr = phi * torch.exp(u) * N2
            J = torch.exp(u) * (phi * N2 - n2 / tau)

            f = delta_curr - delta_fwd

            # Update bracket monotonically: f is decreasing in u on the bracket
            u_lo = torch.where(f > 0.0, torch.maximum(u_lo, u), u_lo)
            u_hi = torch.where(f <= 0.0, torch.minimum(u_hi, u), u_hi)

            step = f / J
            u_cand = u - step

            # Non-strict containment prevents bisecting away once converged to boundary at machine precision
            in_bracket = (u_cand >= u_lo) & (u_cand <= u_hi)
            u_mid = 0.5 * (u_lo + u_hi)
            u = torch.where(in_bracket, u_cand, u_mid)

        K = F * torch.exp(u)

    # Runtime residual validation; skipped when torch.compile is active to guarantee fullgraph capture
    if not torch.compiler.is_compiling():
        recovered_delta = fx_delta(K, sigma, leg, phi, conv)
        residual = torch.abs(recovered_delta - delta)
        eps_mach = torch.finfo(torch.float64).eps
        tau_safe = torch.clamp(tau, min=1e-15)
        tol_tensor = torch.as_tensor(tol, dtype=torch.float64, device=tau.device)
        tol_eff = torch.maximum(tol_tensor, 10.0 * eps_mach / tau_safe)
        if bool(torch.any(residual > tol_eff)):
            raise ConventionSolveError(
                f"Delta solver failed to reach tolerance: max residual {residual.max().item():.3e} > effective tol {tol_eff.max().item():.3e}"
            )

    return K


def atm_strike(
    sigma_atm: Tensor,
    leg: LegMarket,
    atm: AtmConvention,
    conv: DeltaConvention,
) -> Tensor:
    """Computes at-the-money strike across ATMF and DNS conventions.

    Args:
        sigma_atm: ATM volatility tensor of shape (S,), float64.
        leg: LegMarket container.
        atm: ATM convention enum (ATMF or DNS).
        conv: Delta convention enum.

    Returns:
        ATM strike tensor of shape (S,), float64.
    """
    F = leg.forward
    if atm == AtmConvention.ATMF:
        return F
    elif atm == AtmConvention.DNS:
        shift = 0.5 * (sigma_atm ** 2) * leg.T
        if conv in (DeltaConvention.SPOT_PNA, DeltaConvention.FORWARD_PNA):
            return F * torch.exp(shift)
        else:
            return F * torch.exp(-shift)
    else:
        raise ValueError(f"Unsupported ATM convention: {atm}")


def _black_price(
    F: Tensor,
    K: Tensor,
    sigma: Tensor,
    T: Tensor,
    df_d: Tensor,
    phi: float,
) -> Tensor:
    tau = sigma * torch.sqrt(T)
    d1 = (torch.log(F / K) + 0.5 * (tau ** 2)) / tau
    d2 = d1 - tau
    if phi > 0:
        return cast(Tensor, df_d * (F * torch.special.ndtr(d1) - K * torch.special.ndtr(d2)))
    else:
        return cast(Tensor, df_d * (K * torch.special.ndtr(-d2) - F * torch.special.ndtr(-d1)))


def market_to_smile_strangle(
    bf_market: Tensor,
    rr: Tensor,
    sigma_atm: Tensor,
    delta: float,
    leg: LegMarket,
    conv: DeltaConvention,
    smile_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Inverts broker market strangle premium to smile strangle BF_smile.

    Args:
        bf_market: Broker market butterfly tensor of shape (S,), float64.
        rr: Risk reversal tensor of shape (S,), float64.
        sigma_atm: ATM volatility tensor of shape (S,), float64.
        delta: Absolute strangle delta target (e.g. 0.25).
        leg: LegMarket container.
        conv: Delta convention enum.
        smile_fn: Callable (K, bf) -> vol evaluating the parameterised smile.

    Returns:
        Smile butterfly tensor BF_smile of shape (S,), float64.
    """
    F = leg.forward
    T = leg.T
    df_d = leg.df_domestic

    sigma_mkt = sigma_atm + bf_market
    delta_c = torch.full_like(sigma_mkt, delta)
    delta_p = torch.full_like(sigma_mkt, -delta)
    phi_c = torch.ones_like(sigma_mkt)
    phi_p = -torch.ones_like(sigma_mkt)

    K_c = strike_from_delta(delta_c, sigma_mkt, leg, phi_c, conv)
    K_p = strike_from_delta(delta_p, sigma_mkt, leg, phi_p, conv)

    prem_c_mkt = _black_price(F, K_c, sigma_mkt, T, df_d, 1.0)
    prem_p_mkt = _black_price(F, K_p, sigma_mkt, T, df_d, -1.0)
    target_prem = prem_c_mkt + prem_p_mkt

    bf_smile_list = []
    S_len = bf_market.shape[0]

    for i in range(S_len):
        target = target_prem[i].item()
        Kc_i = K_c[i:i+1]
        Kp_i = K_p[i:i+1]
        F_i = F[i:i+1]
        T_i = T[i:i+1]
        df_d_i = df_d[i:i+1]

        def obj(bf_val: float) -> float:
            bf_tensor = torch.tensor([bf_val], dtype=torch.float64)
            vc = smile_fn(Kc_i, bf_tensor)
            vp = smile_fn(Kp_i, bf_tensor)
            pc = _black_price(F_i, Kc_i, vc, T_i, df_d_i, 1.0)
            pp = _black_price(F_i, Kp_i, vp, T_i, df_d_i, -1.0)
            return (pc + pp).item() - target

        b_start = bf_market[i].item()
        bracket_width = 0.05
        bracket = [b_start - bracket_width, b_start + bracket_width]
        f_lo = obj(bracket[0])
        f_hi = obj(bracket[1])

        expand_iter = 0
        while f_lo * f_hi > 0 and expand_iter < 10:
            bracket_width *= 2.0
            bracket = [b_start - bracket_width, b_start + bracket_width]
            f_lo = obj(bracket[0])
            f_hi = obj(bracket[1])
            expand_iter += 1

        if f_lo * f_hi > 0:
            raise ConventionSolveError("Failed to bracket root in market_to_smile_strangle")

        root_val = cast(float, scipy.optimize.brentq(obj, bracket[0], bracket[1], xtol=1e-12))
        bf_smile_list.append(root_val)

    return torch.tensor(bf_smile_list, dtype=torch.float64, device=bf_market.device)


def pillar_strikes(
    quotes: Tensor,
    leg: LegMarket,
    conv: DeltaConvention,
    atm: AtmConvention,
) -> Tuple[Tensor, Tensor]:
    """Maps 5-point smile quotes [atm, rr25, bf25, rr10, bf10] to pillar strikes and vols.

    Args:
        quotes: Quote matrix of shape (S, 5) float64.
        leg: LegMarket container.
        conv: Delta convention enum.
        atm: ATM convention enum.

    Returns:
        Tuple (K, sigma) each of shape (S, 5) ordered as [10P, 25P, ATM, 25C, 10C].
    """
    sigma_atm = quotes[:, 0]
    rr25 = quotes[:, 1]
    bf25 = quotes[:, 2]
    rr10 = quotes[:, 3]
    bf10 = quotes[:, 4]

    sigma_10p = sigma_atm + bf10 - 0.5 * rr10
    sigma_25p = sigma_atm + bf25 - 0.5 * rr25
    sigma_25c = sigma_atm + bf25 + 0.5 * rr25
    sigma_10c = sigma_atm + bf10 + 0.5 * rr10

    sigmas = torch.stack([sigma_10p, sigma_25p, sigma_atm, sigma_25c, sigma_10c], dim=-1)

    delta_10 = torch.full_like(sigma_atm, 0.10)
    delta_25 = torch.full_like(sigma_atm, 0.25)
    phi_call = torch.ones_like(sigma_atm)
    phi_put = -torch.ones_like(sigma_atm)

    K_10p = strike_from_delta(-delta_10, sigma_10p, leg, phi_put, conv)
    K_25p = strike_from_delta(-delta_25, sigma_25p, leg, phi_put, conv)
    K_atm = atm_strike(sigma_atm, leg, atm, conv)
    K_25c = strike_from_delta(delta_25, sigma_25c, leg, phi_call, conv)
    K_10c = strike_from_delta(delta_10, sigma_10c, leg, phi_call, conv)

    K = torch.stack([K_10p, K_25p, K_atm, K_25c, K_10c], dim=-1)
    return K, sigmas
