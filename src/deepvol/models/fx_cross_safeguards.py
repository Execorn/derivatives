"""fx_cross_safeguards.py — Numerical defense, asymptotic taper, and cross-slice audit.

Implements:
  - taper_jet: C2 quintic smoothstep moneyness taper on Jet triples
  - z_squash: Smooth moneyness squashing via c * tanh(z / c)
  - nn_tapered_mask: Warning mask for 5 < |z| < 7
  - smooth_vol_floor: Softplus200 smooth lower bound floor on volatility
  - smooth_time_floor: Softplus1e5 smooth lower bound floor on maturity
  - AuditResult & cross_slice_audit: 401-point slice arbitrage audit
  - wing_slope_certificate: Roger Lee cross-wing asymptotic moment bound certificate (L_bar <= 1.9)
"""
from __future__ import annotations

from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from deepvol.market.svi_slice import durrleman_g
from deepvol.models import taylor_jets as tj
from deepvol.models.fx_cross_analytic import Col, lee_slope_bound


class AuditResult(NamedTuple):
    """Container for per-state cross-slice arbitrage audit outputs."""

    min_g: Tensor           # (S,) float64 minimum Durrleman butterfly density
    call_spread_ok: Tensor  # (S,) bool call-spread slope bounds [-D, 0] satisfied
    slope_bound: Tensor     # (S,) float64 Roger Lee asymptotic wing slope bound bar_L
    ok: Tensor              # (S,) bool composite slice certification passed


def taper_jet(z: tj.Jet, z0: float = 5.0, z1: float = 7.0) -> tj.Jet:
    """C2 quintic smoothstep moneyness taper on Jet z.

    Tapers smoothly from 1 on |z| <= z0 to exact 0 on |z| >= z1 with continuous 1st and 2nd derivatives.
    Uses factored derivatives -30 t^2 (1 - t)^2 and -60 t (1 - t) (1 - 2t).
    """
    a = z.v.abs()
    w = z1 - z0
    inv_w = 1.0 / w
    t = torch.clamp((a - z0) * inv_w, 0.0, 1.0)
    inside = (a > z0) & (a < z1)

    # S(t) = 1 - (10 t^3 - 15 t^4 + 6 t^5) = 1 - t^3 (10 - 15 t + 6 t^2)
    S = 1.0 - t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)

    # Factored derivatives with respect to a = |z|
    one_minus_t = 1.0 - t
    dS_da = torch.where(
        inside,
        -30.0 * inv_w * (t * t) * (one_minus_t * one_minus_t),
        torch.zeros_like(t),
    )
    d2S_da2 = torch.where(
        inside,
        -60.0 * (inv_w * inv_w) * t * one_minus_t * (1.0 - 2.0 * t),
        torch.zeros_like(t),
    )

    sgn = torch.sign(z.v)
    dchi_dz = dS_da * sgn
    d2chi_dz2 = d2S_da2

    return tj.unary(z, S, dchi_dz, d2chi_dz2)


def z_squash(z: tj.Jet, c: float = 7.0) -> tj.Jet:
    """Squashes moneyness Jet z via c * tanh(z / c) with exact Taylor jet derivatives."""
    u = z.v / c
    th = torch.tanh(u)
    sech2 = 1.0 - th * th

    f = c * th
    f1 = sech2
    f2 = -(2.0 / c) * th * sech2

    return tj.unary(z, f, f1, f2)


def nn_tapered_mask(z: Tensor, z0: float = 5.0, z1: float = 7.0) -> Tensor:
    """Boolean mask indicating contracts in transition region z0 < |z| < z1."""
    a = z.abs()
    return (a > z0) & (a < z1)


def smooth_vol_floor(s: tj.Jet, lo: float = 0.01, beta: float = 200.0) -> tj.Jet:
    """Softplus200 smooth lower bound floor on volatility Jet s ensuring strictly positive gradient."""
    return tj.softfloor(s, lo=lo, beta=beta)


def smooth_time_floor(T: Tensor, lo: float = 1e-4, beta: float = 1e5) -> Tensor:
    """Softplus1e5 smooth lower bound floor on maturity T ensuring T >= 1e-4."""
    return lo + F.softplus(T - lo, beta=beta)


def wing_slope_certificate(
    L1: Tensor,
    L2: Tensor,
    beta_wing: Tensor,
    rho: Tensor,
    limit: float = 1.9,
) -> Tensor:
    """Evaluates Roger Lee's asymptotic cross total-variance wing slope bound against limit (default 1.9)."""
    if beta_wing.ndim > 1 and beta_wing.shape[-1] == 2:
        beta1 = beta_wing[..., 0]
        beta2 = beta_wing[..., 1]
    else:
        beta1 = beta_wing
        beta2 = 1.0 - beta1

    slope = lee_slope_bound(L1, L2, beta1, beta2, rho)
    return slope <= limit


def cross_slice_audit(
    engine: nn.Module,
    X_states: Tensor,
    n_grid: int = 401,
    z_max: float = 12.0,
) -> AuditResult:
    """Audits market states across a 401-point collocation grid for Durrleman, call-spread, and wing bounds.

    Args:
        engine: Pricing engine module (e.g. AnalyticTriangulated or CrossVolSurrogateEngine).
        X_states: Tensor of shape (S, 34) containing market states.
        n_grid: Number of points on log-moneyness grid (default 401).
        z_max: Truncation boundary in standardized moneyness z (default 12.0).

    Returns:
        AuditResult named tuple containing min_g (S,), call_spread_ok (S,), slope_bound (S,), ok (S,).
    """
    if X_states.dim() == 1:
        X_states = X_states.unsqueeze(0)

    S_cnt = X_states.shape[0]
    device = X_states.device
    dtype = torch.float64

    min_g_list = []
    call_spread_ok_list = []
    slope_bound_list = []
    ok_list = []

    z_grid = torch.linspace(-z_max, z_max, n_grid, dtype=dtype, device=device)

    for s in range(S_cnt):
        state = X_states[s]

        F_val = float(state[Col.F].item())
        T_raw = state[Col.T]
        T_val = smooth_time_floor(T_raw)
        T_float = float(T_val.item())
        D_val = float(state[Col.D].item())
        rho_val = float(state[Col.RHO].item())

        p1 = state[Col.SVI1 : Col.SVI1 + 5]
        p2 = state[Col.SVI2 : Col.SVI2 + 5]

        # Calculate reference ATMF volatility sigma_ref
        # w_i(0) = a_i + b_i * (rho_s * (-m) + sqrt(m^2 + s^2))
        a1, b1, rho_s1, m1, s1 = [float(x.item()) for x in p1]
        a2, b2, rho_s2, m2, s2 = [float(x.item()) for x in p2]

        w1_0 = a1 + b1 * (rho_s1 * (-m1) + (m1 * m1 + s1 * s1) ** 0.5)
        w2_0 = a2 + b2 * (rho_s2 * (-m2) + (m2 * m2 + s2 * s2) ** 0.5)

        sig1_0 = max(w1_0 / T_float, 1e-8) ** 0.5
        sig2_0 = max(w2_0 / T_float, 1e-8) ** 0.5
        sig_ref = max(sig1_0 * sig1_0 + sig2_0 * sig2_0 + 2.0 * rho_val * sig1_0 * sig2_0, 1e-8) ** 0.5

        # Collocation strikes
        k_vals = z_grid * sig_ref * (T_float ** 0.5)
        K_vals = F_val * torch.exp(k_vals)

        # Build grid tensor (n_grid, 34)
        X_grid = state.unsqueeze(0).expand(n_grid, -1).clone()
        X_grid[:, Col.K] = K_vals
        X_grid[:, Col.PHI] = 1.0  # Call

        with torch.no_grad():
            out = engine(X_grid)

        # 1. Durrleman density non-negativity
        sig_out = out.sigma
        sk_out = out.s_k
        skk_out = out.s_kk

        w_grid = (sig_out ** 2) * T_val
        w1_grid = 2.0 * sig_out * sk_out * T_val
        w2_grid = 2.0 * (sk_out ** 2 + sig_out * skk_out) * T_val

        g_grid = durrleman_g(k_vals.unsqueeze(0), w_grid.unsqueeze(0), w1_grid.unsqueeze(0), w2_grid.unsqueeze(0)).squeeze(0)
        min_g = float(torch.min(g_grid).item())

        # 2. Call spread condition: -D <= dV/dK <= 0
        # dV_call / dK = -D * N(d2) + B_s * s_k / K
        sd = sig_out * (T_val ** 0.5)
        d1 = (-k_vals + 0.5 * sd * sd) / sd
        d2 = d1 - sd
        dK_dV = -D_val * torch.special.ndtr(d2) + out.B_s * sk_out / K_vals

        tau = 1e-10
        call_spread_pass = bool(((dK_dV >= -D_val - tau) & (dK_dV <= tau)).all().item())

        # 3. Roger Lee wing slope bound
        L1_val = b1 * (1.0 + abs(rho_s1))
        L2_val = b2 * (1.0 + abs(rho_s2))

        # Asymptotic beta weights
        sqrt_prod = (L1_val * L2_val) ** 0.5
        denom_beta = L1_val + L2_val + 2.0 * rho_val * sqrt_prod
        if denom_beta > 1e-8:
            beta1_val = (L1_val + rho_val * sqrt_prod) / denom_beta
            beta2_val = 1.0 - beta1_val
        else:
            beta1_val = 0.5
            beta2_val = 0.5

        bar_L = float(lee_slope_bound(
            torch.tensor(L1_val, dtype=dtype),
            torch.tensor(L2_val, dtype=dtype),
            torch.tensor(beta1_val, dtype=dtype),
            torch.tensor(beta2_val, dtype=dtype),
            torch.tensor(rho_val, dtype=dtype),
        ).item())

        min_g_pass = (min_g >= -1e-12)
        slope_pass = (bar_L <= 1.9)
        ok = min_g_pass and call_spread_pass and slope_pass

        min_g_list.append(min_g)
        call_spread_ok_list.append(call_spread_pass)
        slope_bound_list.append(bar_L)
        ok_list.append(ok)

    return AuditResult(
        min_g=torch.tensor(min_g_list, dtype=dtype, device=device),
        call_spread_ok=torch.tensor(call_spread_ok_list, dtype=torch.bool, device=device),
        slope_bound=torch.tensor(slope_bound_list, dtype=dtype, device=device),
        ok=torch.tensor(ok_list, dtype=torch.bool, device=device),
    )
