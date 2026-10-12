"""fx_cross_analytic.py — Smile-consistent Analytic Triangulated pricing engine and Greeks.

Implements beta-split H1 triangulation with 6 fixed-point iterations, Taylor-mode jets,
closed-form Black partials, smile-inclusive Greek assembly, and multi-curve bucketed Rho.
"""
from __future__ import annotations

from enum import IntEnum
import math
from typing import Dict, Final, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from deepvol.models import taylor_jets as tj


class Col(IntEnum):
    """Column indices for the 34-dimensional analytic engine input tensor."""

    F = 0
    K = 1
    T = 2
    T_SET = 3
    D = 4
    G = 5
    RHO = 6
    PHI = 7
    SVI1 = 8
    SVI2 = 13
    XI1 = 18
    XI2 = 23
    DELTA1 = 28
    DELTA2 = 29
    R_C = 30
    R_A = 31
    R_SET = 32
    NN_GATE = 33


N_COLS: Final[int] = 34
_SQ2PI: Final[float] = math.sqrt(2.0 * math.pi)


class EngineOutputs(NamedTuple):
    """Outputs from the AnalyticTriangulated pricing engine."""

    V: Tensor
    sigma: Tensor
    s_k: Tensor
    s_kk: Tensor
    s_t: Tensor
    B_F: Tensor
    B_FF: Tensor
    B_Fs: Tensor
    B_s: Tensor
    B_ss: Tensor
    chi: Tensor


def smooth_T(T: Tensor) -> Tensor:
    """Smooth maturity floor guaranteeing T >= 1e-4 without hard kink."""
    return 1e-4 + F.softplus(T - 1e-4, beta=1e5)


def svi_vol_jet(x: tj.Jet, p: Tensor, T: Tensor, delta: tj.Jet) -> tj.Jet:
    """Evaluates raw SVI implied volatility sqrt(w(x)/T) + delta as a Jet."""
    P = x.t.shape[1]
    a = p[:, 0]
    b = p[:, 1]
    rho_s = p[:, 2]
    m = p[:, 3]
    s = p[:, 4]

    delta_x = tj.add(x, tj.const(-m, P))
    hypot_sq = tj.add(tj.mul(delta_x, delta_x), tj.const(s * s, P))
    hypot = tj.sqrt(hypot_sq)
    bracket = tj.add(tj.scale(delta_x, rho_s), hypot)
    w = tj.add(tj.const(a, P), tj.scale(bracket, b))
    w_t = tj.scale(w, 1.0 / T)
    return tj.add(tj.sqrt(w_t), delta)


def beta_split_jet(
    k: tj.Jet,
    T: Tensor,
    rho: tj.Jet,
    p1: Tensor,
    p2: Tensor,
    d1: tj.Jet,
    d2: tj.Jet,
    n_fp: int = 6,
) -> Tuple[tj.Jet, tj.Jet, tj.Jet]:
    """Beta-split H1 fixed-point triangulation with exact strike additivity."""
    P = k.t.shape[1]
    zero_jet = tj.const(torch.zeros_like(T), P)

    s1 = svi_vol_jet(zero_jet, p1, T, d1)
    s2 = svi_vol_jet(zero_jet, p2, T, d2)

    for _ in range(n_fp):
        s11 = tj.mul(s1, s1)
        s22 = tj.mul(s2, s2)
        s12 = tj.mul(tj.mul(s1, s2), rho)
        sc2 = tj.add(tj.add(s11, s22), tj.scale(s12, 2.0))
        sc2_safe = tj.add(sc2, tj.const(torch.full_like(T, 1e-8), P))
        inv_sc2 = tj.recip(sc2_safe)

        beta1 = tj.mul(tj.add(s11, s12), inv_sc2)
        beta2 = tj.add(tj.const(torch.ones_like(T), P), tj.scale(beta1, -1.0))

        k1 = tj.mul(k, beta1)
        k2 = tj.mul(k, beta2)

        s1 = svi_vol_jet(k1, p1, T, d1)
        s2 = svi_vol_jet(k2, p2, T, d2)

    s11 = tj.mul(s1, s1)
    s22 = tj.mul(s2, s2)
    s12 = tj.mul(tj.mul(s1, s2), rho)
    sc2 = tj.add(tj.add(s11, s22), tj.scale(s12, 2.0))
    sigma_H1 = tj.sqrt(sc2)

    return sigma_H1, s1, s2


def black76_otm_parity(
    F_val: Tensor,
    K_val: Tensor,
    sigma: Tensor,
    T_val: Tensor,
    D_val: Tensor,
    phi: Tensor,
) -> Tensor:
    """Evaluates option price via OTM Black-76 with exact put-call parity reconstruction."""
    k = torch.log(K_val / F_val)
    sd = sigma * torch.sqrt(T_val)
    d1 = (-k + 0.5 * sd * sd) / sd
    d2 = d1 - sd

    phi_otm = torch.where(k >= 0.0, torch.ones_like(k), -torch.ones_like(k))
    c_otm = phi_otm * D_val * (F_val * torch.special.ndtr(phi_otm * d1) - K_val * torch.special.ndtr(phi_otm * d2))
    return c_otm + 0.5 * (phi - phi_otm) * D_val * (F_val - K_val)


def black_partials(
    F_val: Tensor,
    K_val: Tensor,
    sigma: Tensor,
    T_val: Tensor,
    D_val: Tensor,
    phi: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Computes closed-form Black-76 partials (B_F, B_FF, B_Fs, B_s, B_ss)."""
    k = torch.log(K_val / F_val)
    sq = torch.sqrt(T_val)
    sd = sigma * sq
    d1 = (-k + 0.5 * sd * sd) / sd
    d2 = d1 - sd

    n1 = torch.exp(-0.5 * d1 * d1) / _SQ2PI

    B_F = phi * D_val * torch.special.ndtr(phi * d1)
    B_FF = D_val * n1 / (F_val * sd)
    B_Fs = -D_val * n1 * d2 / sigma
    B_s = D_val * F_val * sq * n1
    B_ss = B_s * d1 * d2 / sigma

    return B_F, B_FF, B_Fs, B_s, B_ss


class AnalyticTriangulated(nn.Module):
    """Analytic triangulated cross pricing engine with hand-written Taylor jets."""

    def forward(self, X: Tensor) -> EngineOutputs:
        """Evaluates price, served vol, jets, and Black partials for device tensor X (B, 34)."""
        F_val = X[:, Col.F]
        K_val = X[:, Col.K]
        T_raw = X[:, Col.T]
        D_val = X[:, Col.D]
        rho_val = X[:, Col.RHO]
        phi_val = X[:, Col.PHI]

        p1 = X[:, Col.SVI1 : Col.SVI1 + 5]
        p2 = X[:, Col.SVI2 : Col.SVI2 + 5]

        T_val = smooth_T(T_raw)
        k_val = torch.log(K_val / F_val)

        P = 3
        eye = torch.eye(P, dtype=torch.float64, device=X.device)

        k_jet = tj.seed_k(k_val, P)
        rho_jet = tj.seed_tangent(rho_val, 0, P, eye=eye)
        d1_jet = tj.seed_tangent(X[:, Col.DELTA1], 1, P, eye=eye)
        d2_jet = tj.seed_tangent(X[:, Col.DELTA2], 2, P, eye=eye)

        sigma_jet, _, _ = beta_split_jet(k_jet, T_val, rho_jet, p1, p2, d1_jet, d2_jet, n_fp=6)

        sigma = sigma_jet.v
        s_k = sigma_jet.d
        s_kk = sigma_jet.dd
        s_t = sigma_jet.t

        V = black76_otm_parity(F_val, K_val, sigma, T_val, D_val, phi_val)
        B_F, B_FF, B_Fs, B_s, B_ss = black_partials(F_val, K_val, sigma, T_val, D_val, phi_val)
        chi = torch.ones_like(V)

        return EngineOutputs(
            V=V,
            sigma=sigma,
            s_k=s_k,
            s_kk=s_kk,
            s_t=s_t,
            B_F=B_F,
            B_FF=B_FF,
            B_Fs=B_Fs,
            B_s=B_s,
            B_ss=B_ss,
            chi=chi,
        )


def greeks_from_outputs(
    X: Tensor,
    out: EngineOutputs,
    grad: Optional[Tensor] = None,
    dynamics: str = "STICKY_MONEYNESS",
) -> Dict[str, Tensor]:
    """Assembles smile-inclusive hedging Greeks from engine outputs."""
    F_val = X[:, Col.F]
    K_val = X[:, Col.K]
    T_val = smooth_T(X[:, Col.T])
    D_val = X[:, Col.D]
    g_val = X[:, Col.G]
    phi_val = X[:, Col.PHI]
    r_C = X[:, Col.R_C]
    r_A = X[:, Col.R_A]
    r_set = X[:, Col.R_SET]

    V = out.V
    sigma = out.sigma
    s_t = out.s_t
    B_F = out.B_F
    B_FF = out.B_FF
    B_Fs = out.B_Fs
    B_s = out.B_s
    B_ss = out.B_ss

    if dynamics.upper() == "STICKY_STRIKE":
        s_k = torch.zeros_like(out.s_k)
        s_kk = torch.zeros_like(out.s_kk)
    else:
        s_k = out.s_k
        s_kk = out.s_kk

    k_F = -1.0 / F_val
    k_FF = 1.0 / (F_val * F_val)

    dVdF = B_F + B_s * s_k * k_F
    d2VdF2 = B_FF + 2.0 * B_Fs * s_k * k_F + B_ss * (s_k * k_F) ** 2 + B_s * (s_kk * k_F * k_F + s_k * k_FF)

    delta = g_val * dVdF
    cash_delta = F_val * dVdF
    gamma = g_val * g_val * d2VdF2
    vega = B_s
    vanna = g_val * (B_Fs + B_ss * s_k * k_F)
    vomma = B_ss

    sd = sigma * torch.sqrt(T_val)
    d1 = (-torch.log(K_val / F_val) + 0.5 * sd * sd) / sd
    n1 = torch.exp(-0.5 * d1 * d1) / _SQ2PI

    carry_term = phi_val * D_val * F_val * torch.special.ndtr(phi_val * d1) * (r_C - r_A)
    decay_term = (D_val * F_val * sigma * n1) / (2.0 * torch.sqrt(T_val))
    theta_frozen = r_set * V - carry_term - decay_term
    roll_term = B_s * s_k * (r_C - r_A) if dynamics.upper() == "STICKY_MONEYNESS" else torch.zeros_like(V)
    theta = (theta_frozen + roll_term) / 365.0

    if grad is not None:
        d_rho = grad[:, Col.RHO]
        d_delta1 = grad[:, Col.DELTA1]
        d_delta2 = grad[:, Col.DELTA2]
    else:
        d_rho = B_s * s_t[:, 0]
        d_delta1 = B_s * s_t[:, 1]
        d_delta2 = B_s * s_t[:, 2]

    return {
        "price": V,
        "delta": delta,
        "cash_delta": cash_delta,
        "gamma": gamma,
        "vega": vega,
        "vanna": vanna,
        "vomma": vomma,
        "theta": theta,
        "rho": d_rho,
        "vega_leg1": d_delta1,
        "vega_leg2": d_delta2,
    }


def bucketed_rho(
    dVdF: Tensor,
    V: Tensor,
    F_val: Tensor,
    T_val: Tensor,
    T_set: Tensor,
    pillars_C: Tensor,
    pillars_A: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Computes exact bucketed interest rate sensitivities via linear zero-rate chain rule."""
    def _linear_weights(t: Tensor, pillars: Tensor) -> Tensor:
        B_dim = t.shape[0]
        P_dim = pillars.shape[0]
        w = torch.zeros(B_dim, P_dim, dtype=t.dtype, device=t.device)
        for b in range(B_dim):
            tb = t[b]
            if tb <= pillars[0]:
                w[b, 0] = 1.0
            elif tb >= pillars[-1]:
                w[b, -1] = 1.0
            else:
                idx = int(torch.searchsorted(pillars, tb))
                t_lo, t_hi = pillars[idx - 1], pillars[idx]
                frac = (tb - t_lo) / (t_hi - t_lo)
                w[b, idx - 1] = 1.0 - frac
                w[b, idx] = frac
        return w

    wC_T = _linear_weights(T_val, pillars_C)
    wC_Tset = _linear_weights(T_set, pillars_C)
    wA_T = _linear_weights(T_val, pillars_A)

    dF_term = dVdF * F_val * T_val
    rho_C = dF_term.unsqueeze(-1) * wC_T - (T_set * V).unsqueeze(-1) * wC_Tset
    rho_A = -dF_term.unsqueeze(-1) * wA_T

    return rho_C, rho_A


def lee_slope_bound(
    L1: Tensor,
    L2: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    rho: Tensor,
) -> Tensor:
    """Evaluates Roger Lee's asymptotic total variance wing slope bound."""
    prod = torch.clamp(L1 * L2 * torch.abs(beta1 * beta2), min=0.0)
    return L1 * torch.abs(beta1) + L2 * torch.abs(beta2) + 2.0 * torch.abs(rho) * torch.sqrt(prod)
