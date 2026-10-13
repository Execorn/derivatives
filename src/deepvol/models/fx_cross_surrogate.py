"""fx_cross_surrogate.py — Neural residual volatility surrogate and fused pricing graph.

Implements ResidualCrossVolNet (stacked ensemble MLP in float32 with SiLU activations,
zero-initialized output layer, and Taylor jet propagation via tensor contractions)
and CrossVolSurrogateEngine (smooth C2 quintic moneyness taper, softplus200 volatility floor,
epistemic uncertainty estimation, fused Black-76 with ITM parity, and forward-only quote graph).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, cast
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from deepvol.models import taylor_jets as tj
from deepvol.models.fx_cross_analytic import (
    Col,
    EngineOutputs,
    beta_split_jet,
    black76_otm_parity,
    black_partials,
    smooth_T,
    svi_vol_jet,
)


def jtaper(z: tj.Jet, z0: float = 5.0, z1: float = 7.0) -> tj.Jet:
    """C2 quintic smoothstep moneyness taper on Jet z.

    Tapers smoothly from 1 on |z| <= z0 to 0 on |z| >= z1 with continuous first and second derivatives.
    """
    a = z.v.abs()
    inv_w = 1.0 / (z1 - z0)
    t = torch.clamp((a - z0) * inv_w, 0.0, 1.0)
    inside = (a > z0) & (a < z1)

    # S(t) = 1 - (10 t^3 - 15 t^4 + 6 t^5)
    S = 1.0 - t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    dS = torch.where(inside, -30.0 * inv_w * t * t * ((1.0 - t) ** 2), torch.zeros_like(t))
    d2S = torch.where(
        inside,
        -60.0 * (inv_w ** 2) * t * (1.0 - t) * (1.0 - 2.0 * t),
        torch.zeros_like(t),
    )

    sgn = torch.sign(z.v)
    return tj.unary(z, S, dS * sgn, d2S)


class ResidualCrossVolNet(nn.Module):
    """Stacked ensemble residual cross volatility network with Taylor jet propagation."""

    feature_mean: Tensor
    feature_std: Tensor

    def __init__(self, width: int = 64, depth: int = 3, members: int = 1) -> None:
        super().__init__()
        self.width = width
        self.depth = depth
        self.members = members

        self.register_buffer("feature_mean", torch.zeros(13, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(13, dtype=torch.float32))

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for layer_idx in range(depth + 1):
            in_dim = 13 if layer_idx == 0 else width
            out_dim = 1 if layer_idx == depth else width

            w = torch.empty(members, out_dim, in_dim, dtype=torch.float32)
            b = torch.empty(members, out_dim, dtype=torch.float32)

            if layer_idx == depth:
                nn.init.zeros_(w)
                nn.init.zeros_(b)
            else:
                for member_idx in range(members):
                    nn.init.kaiming_uniform_(w[member_idx], a=math.sqrt(5))
                    bound = 1.0 / math.sqrt(in_dim) if in_dim > 0 else 0.0
                    nn.init.uniform_(b[member_idx], -bound, bound)

            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))

    def standardize(self, x: Tensor) -> Tensor:
        """Standardizes raw 13-D input features using frozen training statistics."""
        return (x - self.feature_mean) / self.feature_std

    def forward(self, x: Tensor) -> Tensor:
        """Evaluates standardized features (B, 13) float32 to raw residual output (B, E) float32."""
        h = x.unsqueeze(0).expand(self.members, -1, -1)

        for layer_idx in range(self.depth):
            w = self.weights[layer_idx]
            b = self.biases[layer_idx]
            h = torch.einsum("eoi,ebi->ebo", w, h) + b.unsqueeze(1)
            h = F.silu(h)

        w_last = self.weights[-1]
        b_last = self.biases[-1]
        out = torch.einsum("eoi,ebi->ebo", w_last, h) + b_last.unsqueeze(1)
        return cast(Tensor, out.squeeze(-1).t())

    def forward_jet(self, cols: Sequence[tj.Jet]) -> tj.Jet:
        """Propagates 13 feature Jets through the ensemble MLP via stacked GEMMs."""
        B = cols[0].v.shape[0]
        P = 0
        for c in cols:
            if c.t is not None:
                P = c.t.shape[1]
                break

        v = torch.stack([c.v for c in cols], dim=-1)
        d = torch.stack([c.d for c in cols], dim=-1)
        dd = torch.stack([c.dd for c in cols], dim=-1)
        dt = self.weights[0].dtype

        if P > 0:
            zt = torch.zeros((B, P), dtype=v.dtype, device=v.device)
            t = torch.stack([c.t if c.t is not None else zt for c in cols], dim=-1)
            t_flat = t.reshape(B * P, -1)
            h_base = torch.cat([v, d, dd, t_flat], dim=0).to(dtype=dt)
        else:
            h_base = torch.cat([v, d, dd], dim=0).to(dtype=dt)

        if self.members == 1:
            H = h_base
            for layer_idx in range(self.depth):
                w = self.weights[layer_idx][0]
                b = self.biases[layer_idx][0]
                H = H @ w.t()
                H = torch.cat([H[:B] + b, H[B:]], dim=0)
                t_in = H[3 * B :].reshape(B, P, -1).transpose(1, 2) if P > 0 else None
                a = tj.silu(tj.Jet(v=H[:B], d=H[B : 2 * B], dd=H[2 * B : 3 * B], t=t_in))
                if P > 0 and a.t is not None:
                    H = torch.cat([a.v, a.d, a.dd, a.t.transpose(1, 2).reshape(B * P, -1)], dim=0)
                else:
                    H = torch.cat([a.v, a.d, a.dd], dim=0)

            w_last = self.weights[-1][0]
            b_last = self.biases[-1][0]
            H = H @ w_last.t()
            H = torch.cat([H[:B] + b_last, H[B:]], dim=0)
            H_64 = H.double()
            return tj.Jet(
                v=H_64[:B, 0],
                d=H_64[B : 2 * B, 0],
                dd=H_64[2 * B : 3 * B, 0],
                t=H_64[3 * B :, 0].reshape(B, P) if P > 0 else None,
            )

        H = h_base.unsqueeze(0).expand(self.members, -1, -1)

        for layer_idx in range(self.depth):
            w = self.weights[layer_idx]
            b = self.biases[layer_idx]
            H = torch.einsum("eoi,ebi->ebo", w, H)

            # Bias is added exclusively to primal rows
            H = torch.cat([H[:, :B, :] + b.unsqueeze(1), H[:, B:, :]], dim=1)

            # SiLU derivative evaluation
            v_b = H[:, :B, :]
            d_b = H[:, B : 2 * B, :]
            dd_b = H[:, 2 * B : 3 * B, :]

            s = torch.sigmoid(v_b)
            f = v_b * s
            f1 = s + v_b * s * (1.0 - s)
            f2 = s * (1.0 - s) * (2.0 + v_b * (1.0 - 2.0 * s))

            v_new = f
            d_new = f1 * d_b
            dd_new = f2 * (d_b * d_b) + f1 * dd_b

            if P > 0:
                t_b = H[:, 3 * B :, :].reshape(self.members, B, P, -1)
                t_new = (t_b * f1.unsqueeze(2)).reshape(self.members, B * P, -1)
                H = torch.cat([v_new, d_new, dd_new, t_new], dim=1)
            else:
                H = torch.cat([v_new, d_new, dd_new], dim=1)

        w_last = self.weights[-1]
        b_last = self.biases[-1]
        H = torch.einsum("eoi,ebi->ebo", w_last, H)
        H = torch.cat([H[:, :B, :] + b_last.unsqueeze(1), H[:, B:, :]], dim=1)
        H = H.squeeze(-1)

        H_64 = H.double()
        v_out = H_64[:, :B].t()
        d_out = H_64[:, B : 2 * B].t()
        dd_out = H_64[:, 2 * B : 3 * B].t()

        if P > 0:
            t_out: Optional[Tensor] = H_64[:, 3 * B :].reshape(self.members, B, P).permute(1, 0, 2)
        else:
            t_out = None

        return tj.Jet(v=v_out, d=d_out, dd=dd_out, t=t_out)


class CrossVolSurrogateEngine(nn.Module):
    """Fused pricing and Greeks engine combining analytic triangulation with residual neural surrogate."""

    eye3: Tensor

    def __init__(self, net: ResidualCrossVolNet, eps_bound: float = 0.25) -> None:
        super().__init__()
        self.net = net
        self.eps_bound = eps_bound
        self.register_buffer("eye3", torch.eye(3, dtype=torch.float64), persistent=False)

    def _prepare_features(
        self,
        X: Tensor,
        eye: Optional[Tensor] = None,
        with_tangents: bool = True,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        tj.Jet,
        tj.Jet,
        tj.Jet,
        tj.Jet,
        Sequence[tj.Jet],
    ]:
        F_val = X[:, Col.F]
        K_val = X[:, Col.K]
        T_raw = X[:, Col.T]
        D_val = X[:, Col.D]
        rho_val = X[:, Col.RHO]
        phi_val = X[:, Col.PHI]

        p1 = X[:, Col.SVI1 : Col.SVI1 + 5]
        p2 = X[:, Col.SVI2 : Col.SVI2 + 5]
        xi1 = X[:, Col.XI1 : Col.XI1 + 5]
        xi2 = X[:, Col.XI2 : Col.XI2 + 5]

        T_val = smooth_T(T_raw)
        k_val = torch.log(K_val / F_val)

        k_jet = tj.seed_k(k_val)
        if with_tangents:
            P = 3
            eye_mat = eye if eye is not None else self.eye3.to(device=X.device)
            rho_jet = tj.seed_tangent(rho_val, 0, P, eye=eye_mat)
            d1_jet = tj.seed_tangent(X[:, Col.DELTA1], 1, P, eye=eye_mat)
            d2_jet = tj.seed_tangent(X[:, Col.DELTA2], 2, P, eye=eye_mat)
        else:
            rho_jet = tj.const(rho_val)
            d1_jet = tj.const(X[:, Col.DELTA1])
            d2_jet = tj.const(X[:, Col.DELTA2])

        sigma_H1, _, _ = beta_split_jet(k_jet, T_val, rho_jet, p1, p2, d1_jet, d2_jet, n_fp=6)

        zero_jet = tj.const(torch.zeros_like(T_val))
        s1_atm = svi_vol_jet(zero_jet, p1, T_val, d1_jet)
        s2_atm = svi_vol_jet(zero_jet, p2, T_val, d2_jet)
        s11 = tj.mul(s1_atm, s1_atm)
        s22 = tj.mul(s2_atm, s2_atm)
        s12 = tj.mul(tj.mul(s1_atm, s2_atm), rho_jet)
        sc2 = tj.add(tj.add(s11, s22), tj.scale(s12, 2.0))
        sc2_safe = tj.add(sc2, tj.const(torch.full_like(T_val, 1e-8)))
        sigma_ref = tj.sqrt(sc2_safe)

        sq_T = torch.sqrt(T_val)
        den = tj.scale(sigma_ref, sq_T)
        z = tj.mul(k_jet, tj.recip(den))

        # Squashed moneyness z_in = 7 tanh(z / 7)
        z_in = tj.scale(tj.tanh(tj.scale(z, 1.0 / 7.0)), 7.0)
        chi = jtaper(z)

        # 13 features: z_in, sqrt(T), rho, xi1 (5), xi2 (5)
        cols_raw = [
            z_in,
            tj.const(sq_T),
            rho_jet,
            *[tj.const(xi1[:, i]) for i in range(5)],
            *[tj.const(xi2[:, i]) for i in range(5)],
        ]

        mean = self.net.feature_mean.to(device=X.device, dtype=torch.float64)
        std = self.net.feature_std.to(device=X.device, dtype=torch.float64)
        cols_std = [
            tj.Jet(
                v=(c.v - mean[j]) / std[j],
                d=c.d / std[j],
                dd=c.dd / std[j],
                t=c.t / std[j] if c.t is not None else None,
            )
            for j, c in enumerate(cols_raw)
        ]

        return (
            F_val,
            K_val,
            T_val,
            D_val,
            phi_val,
            k_val,
            sigma_H1,
            z,
            chi,
            rho_jet,
            cols_std,
        )

    def _forward_impl(
        self,
        X: Tensor,
        eye: Optional[Tensor] = None,
        with_tangents: bool = True,
    ) -> Tuple[EngineOutputs, Tensor]:
        """Internal single-pass evaluation returning both EngineOutputs and epistemic_std_bps."""
        (
            F_val,
            K_val,
            T_val,
            D_val,
            phi_val,
            _,
            sigma_H1,
            _,
            chi,
            _,
            cols_std,
        ) = self._prepare_features(X, eye=eye, with_tangents=with_tangents)

        gate = X[:, Col.NN_GATE]
        out_jet = self.net.forward_jet(cols_std)

        if self.net.members == 1:
            eps_raw = tj.scale(tj.tanh(out_jet), self.eps_bound)
            eps_tapered = tj.mul(eps_raw, chi)
            bar_eps = tj.scale(eps_tapered, gate)
            eps_std = torch.zeros_like(F_val)
        else:
            v_tanh = torch.tanh(out_jet.v)
            u = 1.0 - v_tanh * v_tanh
            d_tanh = u * out_jet.d
            dd_tanh = (-2.0 * v_tanh * u) * (out_jet.d * out_jet.d) + u * out_jet.dd

            v_eps = self.eps_bound * v_tanh
            d_eps = self.eps_bound * d_tanh
            dd_eps = self.eps_bound * dd_tanh

            v_chi = chi.v.unsqueeze(-1)
            d_chi = chi.d.unsqueeze(-1)
            dd_chi = chi.dd.unsqueeze(-1)

            g_col = gate.unsqueeze(-1)
            v_out = v_eps * v_chi * g_col
            d_out = (d_eps * v_chi + v_eps * d_chi) * g_col
            dd_out = (dd_eps * v_chi + 2.0 * d_eps * d_chi + v_eps * dd_chi) * g_col
            eps_std = v_out.std(dim=-1)

            if out_jet.t is not None and chi.t is not None:
                t_tanh = out_jet.t * u.unsqueeze(-1)
                t_eps = self.eps_bound * t_tanh
                t_chi = chi.t.unsqueeze(1)
                t_out_ens = (
                    t_eps * v_chi.unsqueeze(-1) + t_chi * v_eps.unsqueeze(-1)
                ) * g_col.unsqueeze(-1)
                t_bar: Optional[Tensor] = t_out_ens.mean(dim=1)
            else:
                t_bar = None

            bar_eps = tj.Jet(
                v=v_out.mean(dim=-1),
                d=d_out.mean(dim=-1),
                dd=dd_out.mean(dim=-1),
                t=t_bar,
            )

        sigma_raw = tj.mul(sigma_H1, tj.exp(bar_eps))
        sigma_jet = tj.softfloor(sigma_raw, lo=0.01, beta=200.0)

        sigma = sigma_jet.v
        s_k = sigma_jet.d
        s_kk = sigma_jet.dd
        s_t = (
            sigma_jet.t
            if sigma_jet.t is not None
            else torch.zeros(X.shape[0], 0, dtype=torch.float64, device=X.device)
        )

        V = black76_otm_parity(F_val, K_val, sigma, T_val, D_val, phi_val)
        B_F, B_FF, B_Fs, B_s, B_ss = black_partials(
            F_val, K_val, sigma, T_val, D_val, phi_val
        )

        vega_norm = B_s / torch.clamp(D_val * F_val, min=1e-30)
        eps_std_bps = 1e4 * eps_std * sigma * vega_norm

        return (
            EngineOutputs(
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
                chi=chi.v,
            ),
            eps_std_bps,
        )

    def forward(self, X: Tensor) -> EngineOutputs:
        """Forward evaluation of pricing engine returning EngineOutputs tuple with P=3 tangents."""
        out, _ = self._forward_impl(X, eye=None, with_tangents=True)
        return out

    def forward_batch(self, X: Tensor) -> Tuple[EngineOutputs, Tensor]:
        """Batch forward evaluation with P=0 tangents returning (EngineOutputs, epistemic_std_bps)."""
        return self._forward_impl(X, eye=None, with_tangents=False)

    def forward_single(self, X: Tensor, eye: Optional[Tensor] = None) -> Tensor:
        """Forward-only quote graph carrying P=3 tangents, returning packed (B, 9)."""
        out, _ = self._forward_impl(X, eye=eye, with_tangents=True)
        F_val = X[:, Col.F]
        s_k = out.s_k
        s_kk = out.s_kk
        s_t = out.s_t

        B_F = out.B_F
        B_FF = out.B_FF
        B_Fs = out.B_Fs
        B_s = out.B_s
        B_ss = out.B_ss

        k_F = -1.0 / F_val
        k_FF = 1.0 / (F_val * F_val)

        dVdF = B_F + B_s * s_k * k_F
        gF = (
            B_FF
            + 2.0 * B_Fs * s_k * k_F
            + B_ss * (s_k * k_F) ** 2
            + B_s * (s_kk * k_F * k_F + s_k * k_FF)
        )
        dV_t = B_s.unsqueeze(-1) * s_t

        return torch.cat(
            [torch.stack([out.V, out.sigma, gF, dVdF, B_Fs, B_ss], dim=1), dV_t],
            dim=1,
        )

    def epistemic_std_bps(self, X: Tensor) -> Tensor:
        """Calculates epistemic standard deviation across ensemble members in basis points of DF."""
        if self.net.members <= 1:
            return torch.zeros(X.shape[0], dtype=torch.float64, device=X.device)
        _, eps_std_bps = self._forward_impl(X, eye=None, with_tangents=False)
        return eps_std_bps
