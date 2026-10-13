"""src/deepvol/models/fx_cross_copula.py

Gaussian-copula spread option teacher engine under intermediate currency measure Q^B.
Computes market-implied cross option benchmarks, calibrated copula correlation rho_z,
and differentiable implied volatility surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from deepvol.hedging.pivot_iv import pivot_implied_vol
from deepvol.market.svi_slice import RawSVI, durrleman_g


@dataclass(frozen=True)
class CopulaConfig:
    """Configuration container for GaussianCopulaCrossEngine.

    Attributes:
        n_nodes: Number of 1D quadrature nodes (must be odd, default 1601).
        z_max: Truncation boundary for standard normal drivers (default 8.5).
        strike_chunk: Maximum strikes per working chunk to cap memory at <= 256 MB (default 8).
        device: Target execution device ('cuda' or 'cpu').
    """
    n_nodes: int = 1601
    z_max: float = 8.5
    strike_chunk: int = 8
    device: str = "cuda"


@dataclass(frozen=True)
class CrossState:
    """Market state container for triangulated cross option valuation.

    Attributes:
        F1: Float64 tensor of shape (S,) storing Leg 1 forwards (asset A in ccy B).
        F2: Float64 tensor of shape (S,) storing Leg 2 forwards (ccy B in ccy C).
        T: Float64 tensor of shape (S,) storing time to expiry in years.
        svi1: RawSVI slice container for Leg 1 (params shape (S, 5)).
        svi2: RawSVI slice container for Leg 2 (params shape (S, 5)).
        rho: Float64 tensor of shape (S,) storing market triangulation correlation rho_12.
        rho_z: Optional float64 tensor of shape (S,) storing calibrated copula correlation.
    """
    F1: Tensor
    F2: Tensor
    T: Tensor
    svi1: RawSVI
    svi2: RawSVI
    rho: Tensor
    rho_z: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.F1.dim() == 0:
            object.__setattr__(self, "F1", self.F1.unsqueeze(0))
        if self.F2.dim() == 0:
            object.__setattr__(self, "F2", self.F2.unsqueeze(0))
        if self.T.dim() == 0:
            object.__setattr__(self, "T", self.T.unsqueeze(0))
        if self.rho.dim() == 0:
            object.__setattr__(self, "rho", self.rho.unsqueeze(0))
        if self.rho_z is not None and self.rho_z.dim() == 0:
            object.__setattr__(self, "rho_z", self.rho_z.unsqueeze(0))

    @property
    def F(self) -> Tensor:
        """Synthetic cross forward F = F1 * F2."""
        return self.F1 * self.F2

    @property
    def sigma_ref(self) -> Tensor:
        """Reference ATMF cross volatility from leg ATMF volatilities."""
        w1_0 = self.svi1.w(torch.zeros_like(self.F1))
        w2_0 = self.svi2.w(torch.zeros_like(self.F2))
        s1 = torch.sqrt(torch.clamp(w1_0 / self.T, min=1e-8))
        s2 = torch.sqrt(torch.clamp(w2_0 / self.T, min=1e-8))
        var = s1 ** 2 + s2 ** 2 + 2.0 * self.rho * s1 * s2
        return torch.sqrt(torch.clamp(var, min=1e-8))


class _QuantileInversionFunction(torch.autograd.Function):
    """Autograd function implementing quantile inversion with implicit backward for AAD."""

    @staticmethod
    def forward(
        ctx: Any,
        u: Tensor,
        F: Tensor,
        T: Tensor,
        svi_params: Tensor,
        reflect: bool,
    ) -> Tensor:
        original_shape = u.shape
        S_cnt = F.shape[0]
        device = u.device
        dtype = torch.float64

        u_in = u
        if u_in.dim() > 1 and u_in.shape[0] != S_cnt:
            u_in = u_in.unsqueeze(0)

        if u_in.dim() == 1:
            u_2d = u_in.unsqueeze(0).expand(S_cnt, -1).to(dtype=dtype)
        elif u_in.dim() == 2 and u_in.shape[0] == S_cnt:
            u_2d = u_in.to(dtype=dtype)
        else:
            u_2d = u_in.reshape(S_cnt, -1).to(dtype=dtype)

        U_cnt = u_2d.shape[1]
        F_f64 = F.to(device=device, dtype=dtype)
        T_f64 = T.to(device=device, dtype=dtype)
        params_f64 = svi_params.to(device=device, dtype=dtype)

        if reflect:
            F_eff = 1.0 / F_f64
            params_eff = params_f64.clone()
            params_eff[:, 2] = -params_eff[:, 2]  # rho_s -> -rho_s
            params_eff[:, 3] = -params_eff[:, 3]  # m -> -m
        else:
            F_eff = F_f64
            params_eff = params_f64

        svi = RawSVI(params_eff)
        N_grid = 4097
        w0 = svi.w(torch.zeros((S_cnt, 1), device=device, dtype=dtype))
        sd = torch.sqrt(torch.clamp(w0, min=1e-8))
        k_max = 14.0 * sd

        alpha_lin = torch.linspace(0.0, 1.0, N_grid, device=device, dtype=dtype).unsqueeze(0)
        kg = -k_max + 2.0 * k_max * alpha_lin

        w, w1, _ = svi.derivs(kg)
        sq = torch.sqrt(torch.clamp(w, min=1e-8))
        d1 = (-kg + 0.5 * w) / sq
        d2 = d1 - sq

        nd_minus_d2 = torch.special.ndtr(-d2)
        SQRT_2PI = 2.5066282746310005
        phi_d2 = torch.exp(-0.5 * d2 ** 2) / SQRT_2PI

        cdf_grid = nd_minus_d2 + phi_d2 * (w1 / (2.0 * sq))
        cdf_grid = torch.clamp(cdf_grid, min=1e-45, max=1.0 - 1e-16)
        cdf_grid = torch.cummax(cdf_grid, dim=-1).values

        quantiles = torch.empty((S_cnt, U_cnt), device=device, dtype=dtype)
        chunk_len = 65536

        for s in range(S_cnt):
            if params_eff[s, 1] < 1e-12:
                # Exact lognormal Black-Scholes quantile for flat smiles (b == 0)
                w0_s = params_eff[s, 0]
                sd_s = torch.sqrt(torch.clamp(w0_s, min=1e-8))
                u_clamped = torch.clamp(u_2d[s], min=1e-45, max=1.0 - 1e-16)
                z_u = torch.special.ndtri(u_clamped)
                k_u = -0.5 * sd_s ** 2 + sd_s * z_u
                quantiles[s] = F_eff[s] * torch.exp(k_u)
                continue

            cdf_s = cdf_grid[s]
            kg_s = kg[s]
            u_s = u_2d[s]

            for start in range(0, U_cnt, chunk_len):
                end = min(start + chunk_len, U_cnt)
                u_chunk = u_s[start:end]

                idx = torch.searchsorted(cdf_s, u_chunk)
                idx = torch.clamp(idx, min=1, max=N_grid - 1)

                c_lo = cdf_s[idx - 1]
                c_hi = cdf_s[idx]
                k_lo = kg_s[idx - 1]
                k_hi = kg_s[idx]

                frac = (u_chunk - c_lo) / torch.clamp(c_hi - c_lo, min=1e-16)
                k_interp = k_lo + frac * (k_hi - k_lo)
                quantiles[s, start:end] = F_eff[s] * torch.exp(k_interp)

        ctx.save_for_backward(quantiles, F_eff, T_f64, params_eff)
        ctx.reflect = reflect
        ctx.u_shape = original_shape
        ctx.s_cnt = S_cnt

        if u.dim() == 1:
            return quantiles
        return quantiles.reshape(original_shape)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Optional[Tensor], ...]:
        grad_output = grad_outputs[0]
        quantiles, F_eff, _, params_eff = ctx.saved_tensors
        grad_out_2d = grad_output.reshape(quantiles.shape)
        S_cnt, U_cnt = quantiles.shape
        chunk_len = 65536

        grad_u = torch.empty_like(quantiles)
        grad_params = torch.zeros_like(params_eff)
        grad_F = torch.zeros_like(F_eff)
        SQRT_2PI = 2.5066282746310005

        for s in range(S_cnt):
            p_s = params_eff[s : s + 1]
            svi_s = RawSVI(p_s)
            F_s = F_eff[s]

            b = p_s[:, 1:2]
            rho_s = p_s[:, 2:3]
            m = p_s[:, 3:4]
            s_param = p_s[:, 4:5]

            grad_a_s = torch.zeros(1, dtype=torch.float64, device=quantiles.device)
            grad_b_s = torch.zeros(1, dtype=torch.float64, device=quantiles.device)
            grad_rho_s = torch.zeros(1, dtype=torch.float64, device=quantiles.device)
            grad_m_s = torch.zeros(1, dtype=torch.float64, device=quantiles.device)
            grad_s_param_s = torch.zeros(1, dtype=torch.float64, device=quantiles.device)
            grad_F_s = torch.zeros(1, dtype=torch.float64, device=quantiles.device)

            for start in range(0, U_cnt, chunk_len):
                end = min(start + chunk_len, U_cnt)
                q_chunk = quantiles[s : s + 1, start:end]
                grad_out_chunk = grad_out_2d[s : s + 1, start:end]

                k_chunk = torch.log(torch.clamp(q_chunk / F_s, min=1e-12))
                w_c, w1_c, w2_c = svi_s.derivs(k_chunk)

                sq_c = torch.sqrt(torch.clamp(w_c, min=1e-8))
                d1_c = (-k_chunk + 0.5 * w_c) / sq_c
                d2_c = d1_c - sq_c
                phi_d2_c = torch.exp(-0.5 * d2_c ** 2) / SQRT_2PI

                g_c = durrleman_g(k_chunk, w_c, w1_c, w2_c)
                pdf_c = (phi_d2_c / (q_chunk * sq_c)) * torch.clamp(g_c, min=1e-8)
                pdf_safe = torch.clamp(pdf_c, min=1e-12)

                grad_u_chunk = grad_out_chunk / pdf_safe
                grad_u[s, start:end] = grad_u_chunk.squeeze(0)

                # Parameter derivatives: dq / d params = - (1 / p(q)) * d F_cdf / d params
                d_F_dw1 = phi_d2_c / (2.0 * sq_c)
                d_d2_dw = (2.0 * k_chunk - w_c) / (4.0 * sq_c ** 3)
                d_F_dw = phi_d2_c * (-d_d2_dw * (1.0 + d2_c * w1_c / (2.0 * sq_c)) - w1_c / (4.0 * sq_c ** 3))

                delta = k_chunk - m
                hk = torch.sqrt(delta ** 2 + s_param ** 2)

                dw_da = torch.ones_like(k_chunk)
                dw1_da = torch.zeros_like(k_chunk)

                dw_db = rho_s * delta + hk
                dw1_db = rho_s + delta / hk

                dw_drho = b * delta
                dw1_drho = b.expand_as(k_chunk)

                dw_dm = -w1_c
                dw1_dm = -b * (s_param ** 2) / (hk ** 3)

                dw_ds = b * s_param / hk
                dw1_ds = -b * s_param * delta / (hk ** 3)

                dF_da = d_F_dw * dw_da + d_F_dw1 * dw1_da
                dF_db = d_F_dw * dw_db + d_F_dw1 * dw1_db
                dF_drho = d_F_dw * dw_drho + d_F_dw1 * dw1_drho
                dF_dm = d_F_dw * dw_dm + d_F_dw1 * dw1_dm
                dF_ds = d_F_dw * dw_ds + d_F_dw1 * dw1_ds

                grad_a_s = grad_a_s - (grad_u_chunk * dF_da).sum()
                grad_b_s = grad_b_s - (grad_u_chunk * dF_db).sum()
                grad_rho_s = grad_rho_s - (grad_u_chunk * dF_drho).sum()
                grad_m_s = grad_m_s - (grad_u_chunk * dF_dm).sum()
                grad_s_param_s = grad_s_param_s - (grad_u_chunk * dF_ds).sum()

                grad_F_s = grad_F_s + (grad_out_chunk * (q_chunk / F_s)).sum()

            if ctx.reflect:
                grad_rho_s = -grad_rho_s
                grad_m_s = -grad_m_s
                grad_F_s = -grad_F_s / (F_s ** 2)

            grad_params[s, 0] = grad_a_s.squeeze()
            grad_params[s, 1] = grad_b_s.squeeze()
            grad_params[s, 2] = grad_rho_s.squeeze()
            grad_params[s, 3] = grad_m_s.squeeze()
            grad_params[s, 4] = grad_s_param_s.squeeze()
            grad_F[s] = grad_F_s.squeeze()

        if len(ctx.u_shape) == 1:
            grad_u_out = grad_u.sum(dim=0)
        else:
            grad_u_out = grad_u.reshape(ctx.u_shape)

        return grad_u_out, grad_F, None, grad_params, None


class GaussianCopulaCrossEngine:
    """Production Gaussian-copula teacher engine under intermediate measure Q^B."""

    def __init__(self, cfg: CopulaConfig = CopulaConfig()) -> None:
        self.cfg = cfg
        target_dev = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.device = target_dev
        self._init_quadrature()

    def _init_quadrature(self) -> None:
        n = self.cfg.n_nodes
        if n % 2 == 0:
            raise ValueError(f"n_nodes must be odd for Simpson's quadrature, got {n}")
        L = self.cfg.z_max
        dev = self.device
        dtype = torch.float64

        z = torch.linspace(-L, L, n, device=dev, dtype=dtype)
        dz = 2.0 * L / (n - 1)

        w_simp = torch.empty(n, device=dev, dtype=dtype)
        w_simp[0] = 1.0
        w_simp[-1] = 1.0
        w_simp[1:-1:2] = 4.0
        w_simp[2:-1:2] = 2.0
        w_simp *= (dz / 3.0)

        SQRT_2PI = 2.5066282746310005
        phi_z = torch.exp(-0.5 * z ** 2) / SQRT_2PI
        w1d = w_simp * phi_z
        w1d = w1d / w1d.sum()

        self.z = z
        self.w1d = w1d
        self.u_1d = torch.special.ndtr(z)

    @property
    def W2d(self) -> Tensor:
        """On-demand 2D quadrature weight matrix for backward compatibility."""
        return self.w1d.unsqueeze(1) * self.w1d.unsqueeze(0)

    def quantiles(
        self,
        svi: RawSVI,
        F: Tensor,
        T: Tensor,
        u: Tensor,
        reflect: bool = False,
    ) -> Tensor:
        """Evaluates risk-neutral quantiles q(u) via monotonic 4,097-point grid inversion.

        Args:
            svi: RawSVI container.
            F: Forward tensor of shape (S,).
            T: Maturity tensor of shape (S,).
            u: Uniform probability tensor of shape (U,), (S, U), or (S, U1, U2).
            reflect: Whether to reflect the smile (for Y = 1 / S_2).

        Returns:
            Quantile values in float64.
        """
        F_in = F.to(self.device, dtype=torch.float64)
        T_in = T.to(self.device, dtype=torch.float64)
        u_in = u.to(self.device, dtype=torch.float64)
        params_in = svi.params.to(self.device, dtype=torch.float64)
        res: Tensor = _QuantileInversionFunction.apply(u_in, F_in, T_in, params_in, reflect)  # type: ignore[no-untyped-call]
        return res

    def price(
        self,
        state: CrossState,
        K: Tensor,
        is_call: bool = True,
    ) -> Tensor:
        """Prices undiscounted European cross options via 2D copula quadrature.

        Working memory strictly capped at <= 256 MB via strike chunking (<= 8 strikes per chunk).

        Args:
            state: CrossState containing market data and slices.
            K: Strike tensor of shape (S, M) or (M,).
            is_call: Whether to price European call (True) or put (False).

        Returns:
            Undiscounted prices of shape (S, M) in float64.
        """
        S_cnt = state.F1.shape[0]
        K_in = K.to(self.device, dtype=torch.float64)
        if K_in.dim() == 1:
            K_in = K_in.unsqueeze(0).expand(S_cnt, -1)
        elif K_in.dim() == 0:
            K_in = K_in.view(1, 1).expand(S_cnt, 1)
        M_cnt = K_in.shape[1]

        F1 = state.F1.to(self.device, dtype=torch.float64)
        F2 = state.F2.to(self.device, dtype=torch.float64)
        T = state.T.to(self.device, dtype=torch.float64)
        rho_z = (state.rho_z if state.rho_z is not None else -state.rho).to(self.device, dtype=torch.float64)

        n = self.cfg.n_nodes
        z = self.z
        u_1d = self.u_1d

        # Evaluate X(z_1): shape (S, n)
        X = self.quantiles(state.svi1, F1, T, u_1d, reflect=False)  # (S, n)

        # Build bivariate Gaussian driver Z2: (S, n, n)
        rho_z_col = rho_z.view(S_cnt, 1, 1)
        z1_mat = z.view(1, n, 1)
        z2_mat = z.view(1, 1, n)
        scale_perp = torch.sqrt(torch.clamp(1.0 - rho_z_col ** 2, min=1e-8))
        Z2_joint = rho_z_col * z1_mat + scale_perp * z2_mat
        u2_joint = torch.special.ndtr(Z2_joint)  # (S, n, n)

        # Evaluate Y(Z2_joint): shape (S, n, n)
        Y = self.quantiles(state.svi2, F2, T, u2_joint, reflect=True)  # (S, n, n)

        chunk_size = self.cfg.strike_chunk
        out_prices = torch.empty((S_cnt, M_cnt), device=self.device, dtype=torch.float64)

        X_b = X.view(S_cnt, 1, n, 1)
        Y_b = Y.view(S_cnt, 1, n, n)

        for start_idx in range(0, M_cnt, chunk_size):
            end_idx = min(start_idx + chunk_size, M_cnt)
            K_chunk = K_in[:, start_idx:end_idx]  # (S, M_c)
            M_c = end_idx - start_idx
            K_b = K_chunk.view(S_cnt, M_c, 1, 1)

            if is_call:
                payoff = torch.clamp(X_b - K_b * Y_b, min=0.0)
            else:
                payoff = torch.clamp(K_b * Y_b - X_b, min=0.0)

            p_int = (payoff * self.w1d.view(1, 1, 1, n)).sum(dim=-1)
            p_int = (p_int * self.w1d.view(1, 1, n)).sum(dim=-1)
            out_prices[:, start_idx:end_idx] = p_int * F2.unsqueeze(-1)

        return out_prices

    def implied_vol(
        self,
        price: Tensor,
        F: Tensor,
        K: Tensor,
        T: Tensor,
        is_call: bool = True,
    ) -> Tensor:
        """Inverts undiscounted cross prices to Black-76 implied vol via PIVOT.

        Args:
            price: Option price tensor of shape (S, M).
            F: Cross forward tensor of shape (S,) or (S, M).
            K: Strike tensor of shape (S, M).
            T: Maturity tensor of shape (S,) or (S, M).
            is_call: Whether price represents call (True) or put (False).

        Returns:
            Implied volatility tensor of shape (S, M) in float64.
        """
        p_f64 = price.to(self.device, dtype=torch.float64)
        K_f64 = K.to(self.device, dtype=torch.float64)
        if K_f64.shape != p_f64.shape:
            K_f64 = K_f64.expand_as(p_f64)
        F_f64 = F.to(self.device, dtype=torch.float64)
        if F_f64.shape != p_f64.shape:
            F_f64 = F_f64.expand_as(p_f64)
        T_f64 = T.to(self.device, dtype=torch.float64)
        if T_f64.shape != p_f64.shape:
            T_f64 = T_f64.expand_as(p_f64)

        return pivot_implied_vol(
            price=p_f64,
            S=F_f64,
            K=K_f64,
            T=T_f64,
            r=0.0,
            q=0.0,
            is_call=is_call,
            vega_epsilon=1e-4,
        )

    def calibrate_rho_z(
        self,
        state: CrossState,
        sigma_ref: Tensor,
        n_iter: int = 40,
    ) -> Tensor:
        """Calibrates copula correlation rho_z via fixed hybrid bisection + secant iterations.

        Objective: sigma_cop(F, rho_z) = sigma_ref.
        Convergence: Strictly decreasing monotonicity ensures superlinear convergence.

        Args:
            state: Market CrossState.
            sigma_ref: Target ATMF cross volatility tensor of shape (S,).
            n_iter: Total iterations (default 40: 10 bisection + 30 secant).

        Returns:
            Calibrated rho_z tensor of shape (S,) in float64.
        """
        S_cnt = state.F1.shape[0]
        dev = self.device
        dtype = torch.float64

        Fc = state.F.to(dev, dtype=dtype)
        target = sigma_ref.to(dev, dtype=dtype)
        if target.dim() == 0:
            target = target.unsqueeze(0)

        # Precompute X(z_1) once, since it does not depend on rho_z
        X = self.quantiles(state.svi1, state.F1, state.T, self.u_1d, reflect=False)  # (S, n)

        def _get_atmf_vol(rz_cand: Tensor) -> Tensor:
            rz_col = rz_cand.view(S_cnt, 1, 1)
            z1_mat = self.z.view(1, self.cfg.n_nodes, 1)
            z2_mat = self.z.view(1, 1, self.cfg.n_nodes)
            scale_perp = torch.sqrt(torch.clamp(1.0 - rz_col ** 2, min=1e-8))
            Z2 = rz_col * z1_mat + scale_perp * z2_mat
            u2 = torch.special.ndtr(Z2)
            Y = self.quantiles(state.svi2, state.F2, state.T, u2, reflect=True)  # (S, n, n)

            X_b = X.view(S_cnt, 1, self.cfg.n_nodes, 1)
            Y_b = Y.view(S_cnt, 1, self.cfg.n_nodes, self.cfg.n_nodes)
            K_b = Fc.view(S_cnt, 1, 1, 1)
            payoff = torch.clamp(X_b - K_b * Y_b, min=0.0)
            p_int = (payoff * self.w1d.view(1, 1, 1, self.cfg.n_nodes)).sum(dim=-1)
            p_int = (p_int * self.w1d.view(1, 1, self.cfg.n_nodes)).sum(dim=-1)
            c_atm = p_int.squeeze(-1) * state.F2.to(dev, dtype=dtype)
            iv = self.implied_vol(c_atm, Fc, Fc, state.T.to(dev, dtype=dtype), is_call=True)
            return iv.view(S_cnt)

        a = torch.full((S_cnt,), -0.99, device=dev, dtype=dtype)
        b = torch.full((S_cnt,), 0.99, device=dev, dtype=dtype)


        # 10 Bisection iterations
        for _ in range(10):
            mid = 0.5 * (a + b)
            f_mid = _get_atmf_vol(mid) - target
            a = torch.where(f_mid > 0.0, mid, a)
            b = torch.where(f_mid <= 0.0, mid, b)

        x0, f0 = a.clone(), (_get_atmf_vol(a) - target)
        x1, f1 = b.clone(), (_get_atmf_vol(b) - target)

        # 30 Safeguarded Secant iterations
        for _ in range(n_iter - 10):
            if float(torch.max(torch.abs(f1))) < 1e-12:
                break

            denom = f1 - f0
            denom_safe = torch.where(denom.abs() < 1e-15, torch.full_like(denom, 1e-15), denom)
            x_sec = x1 - f1 * (x1 - x0) / denom_safe

            valid = (x_sec > a + 1e-8) & (x_sec < b - 1e-8)
            x_next = torch.where(valid, x_sec, 0.5 * (a + b))

            f_next = _get_atmf_vol(x_next) - target

            a = torch.where(f_next > 0.0, x_next, a)
            b = torch.where(f_next <= 0.0, x_next, b)

            x0, f0 = x1, f1
            x1, f1 = x_next, f_next

        return 0.5 * (a + b)
