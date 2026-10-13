from __future__ import annotations

import hashlib
import math
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import Tensor

from deepvol.market.fx_conventions import (
    AtmConvention,
    DeltaConvention,
    LegMarket,
    StrangleConvention,
    pillar_strikes,
    strike_from_delta,
)
from deepvol.market.svi_slice import (
    RawSVI,
    certify_slice,
    fit_raw_svi_5pt,
)


@dataclass(frozen=True)
class CanonicalSmile:
    """Canonical leg smile representation.

    Attributes:
        slice_params: Float64 tensor of shape (S, 5) storing raw-SVI parameters [a, b, rho_s, m, s].
        xi: Float64 tensor of shape (S, 5) storing canonical features [sigma_ATM, RR25, BF25, RR10, BF10].
        jacobian: Float64 tensor of shape (S, 5, 5) storing d xi / d quotes.
        sigma_atmf: Float64 tensor of shape (S,) storing ATMF volatility.
        min_g: Float64 tensor of shape (S,) storing minimum Durrleman butterfly density.
        used_ssvi: Boolean tensor of shape (S,) indicating whether SSVI fallback was activated.
        lee_slope: Float64 tensor of shape (S,) storing asymptotic total-variance slope L.
    """
    slice_params: Tensor
    xi: Tensor
    jacobian: Tensor
    sigma_atmf: Tensor
    min_g: Tensor
    used_ssvi: Tensor
    lee_slope: Tensor
    _R_theta: Optional[Tensor] = None
    _R_q: Optional[Tensor] = None
    _Psi_theta: Optional[Tensor] = None

    @property
    def r_theta(self) -> Optional[Tensor]:
        return self._R_theta

    @property
    def r_q(self) -> Optional[Tensor]:
        return self._R_q

    @property
    def psi_theta(self) -> Optional[Tensor]:
        return self._Psi_theta


class SmileArbitrageError(Exception):
    """Raised when market smile quotes exhibit unrepairable vertical or butterfly arbitrage."""


def fit_ssvi_slice(x: Tensor, w: Tensor) -> Tuple[Tensor, Tensor]:
    """Fits 3-parameter Gatheral-Jacquier SSVI slice under strict no-arbitrage bounds.

    The Gatheral-Jacquier surface SVI slice has total variance:
        w(x) = 0.5 * theta * (1 + rho * phi * x + sqrt((phi * x + rho)^2 + 1 - rho^2))
    Sufficient no-arbitrage bounds theta * phi * (1 + |rho|) < 4 and theta * phi^2 * (1 + |rho|) <= 4
    are enforced through smooth unconstrained reparametrization:
        rho = 0.999 * tanh(u)
        phi = phi_max * sigmoid(v)
        phi_max = min(4 / (theta * (1 + |rho|)), sqrt(4 / (theta * (1 + |rho|)))) * (1 - 1e-6)

    Args:
        x: Log-moneyness tensor of shape (S, 5) or (5,), float64.
        w: Total variance tensor of shape (S, 5) or (5,), float64.

    Returns:
        Tuple of (params, rmse) where params has shape (S, 3) storing [theta, rho, phi],
        and rmse has shape (S,) storing vol fit RMSE.

    Raises:
        SmileArbitrageError: If inputs contain non-positive total variances or fit RMSE exceeds threshold.
    """
    x_in = (x.unsqueeze(0) if x.dim() == 1 else x).detach()
    w_in = (w.unsqueeze(0) if w.dim() == 1 else w).detach()
    S_len = x_in.shape[0]
    dev = x_in.device

    params_out: List[List[float]] = []
    rmse_out: List[float] = []

    for i in range(S_len):
        x_row = x_in[i].detach().cpu().numpy()
        w_row = w_in[i].detach().cpu().numpy()

        if np.any(w_row <= 0.0) or np.any(np.isnan(w_row)):
            raise SmileArbitrageError("Non-positive or NaN total variance encountered in SSVI fit")

        sig_row = np.sqrt(np.maximum(w_row, 1e-12))
        th_base = float(max(w_row[2], 1e-6))

        def get_params(p: Sequence[float]) -> Tuple[float, float, float]:
            zeta, u_val, v_val = p
            th = th_base * math.exp(zeta)
            r_val = 0.999 * math.tanh(u_val)
            denom = th * (1.0 + abs(r_val))
            phi_max = min(4.0 / denom, math.sqrt(4.0 / denom)) * (1.0 - 1e-6)
            sigm_v = 1.0 / (1.0 + math.exp(-v_val))
            ph = phi_max * sigm_v
            return th, r_val, ph

        def eval_w(th: float, r_val: float, ph: float, x_vals: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            arg = ph * x_vals + r_val
            rad = np.sqrt(arg ** 2 + 1.0 - r_val ** 2)
            return cast(np.ndarray[Any, Any], 0.5 * th * (1.0 + r_val * ph * x_vals + rad))

        def residuals(p: Sequence[float]) -> np.ndarray[Any, Any]:
            th, r_val, ph = get_params(p)
            w_pred = eval_w(th, r_val, ph, x_row)
            vol_pred = np.sqrt(np.maximum(w_pred, 1e-12))
            return cast(np.ndarray[Any, Any], vol_pred - sig_row)

        best_res = None
        for u0 in (-0.5, 0.0, 0.5):
            for v0 in (-1.0, 0.0, 1.0):
                sol = least_squares(residuals, [0.0, u0, v0], ftol=1e-12, xtol=1e-12)
                if best_res is None or sol.cost < best_res.cost:
                    best_res = sol

        assert best_res is not None
        th_opt, r_opt, ph_opt = get_params(best_res.x)
        w_fit = eval_w(th_opt, r_opt, ph_opt, x_row)
        vol_fit = np.sqrt(np.maximum(w_fit, 1e-12))
        rmse_val = float(np.sqrt(np.mean((vol_fit - sig_row) ** 2)))

        if rmse_val > 0.05:
            raise SmileArbitrageError(f"SSVI fit RMSE {rmse_val:.4f} exceeds 0.05 vol threshold")

        params_out.append([th_opt, r_opt, ph_opt])
        rmse_out.append(rmse_val)

    params_tensor = torch.tensor(params_out, dtype=torch.float64, device=dev)
    rmse_tensor = torch.tensor(rmse_out, dtype=torch.float64, device=dev)
    return params_tensor, rmse_tensor


def ssvi_to_raw_svi(theta: Tensor, rho: Tensor, phi: Tensor) -> RawSVI:
    """Maps Gatheral-Jacquier SSVI parameters to raw-SVI parameters [a, b, rho_s, m, s].

    Algebraic isomorphism:
        a = 0.5 * theta * (1 - rho^2)
        b = 0.5 * theta * phi
        rho_s = rho
        m = -rho / phi
        s = sqrt(1 - rho^2) / phi

    Args:
        theta: ATM total variance tensor of shape (S,), float64.
        rho: Correlation parameter tensor of shape (S,), float64.
        phi: Skew parameter tensor of shape (S,), float64.

    Returns:
        RawSVI slice container.
    """
    rho_safe = torch.clamp(rho, min=-0.999, max=0.999)
    a = 0.5 * theta * (1.0 - (rho_safe ** 2))
    b = 0.5 * theta * phi
    rho_s = rho_safe
    m = -rho_safe / phi
    s = torch.sqrt(torch.clamp(1.0 - (rho_safe ** 2), min=1e-15)) / phi

    params = torch.stack([a, b, rho_s, m, s], dim=-1)
    return RawSVI(params)


def canonical_features(svi: RawSVI, leg: LegMarket) -> Tensor:
    """Evaluates reference canonical features xi from a certified SVI slice.

    Canonical features xi = [sigma_ATM, RR25, BF25, RR10, BF10] are evaluated under
    reference conventions: DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE.

    Args:
        svi: Certified RawSVI slice container of shape (S, 5).
        leg: LegMarket container of shape (S,).

    Returns:
        Tensor of shape (S, 5), float64.
    """
    S_len = svi.params.shape[0]
    dev = svi.params.device

    # ATM DNS: solve x_atm = 0.5 * w(x_atm) via Newton
    x_atm = 0.5 * svi.w(torch.zeros(S_len, 1, dtype=torch.float64, device=dev)).squeeze(1)
    for _ in range(6):
        w_val, w1_val, _ = svi.derivs(x_atm.unsqueeze(1))
        f_val = x_atm.unsqueeze(1) - 0.5 * w_val
        f1_val = 1.0 - 0.5 * w1_val
        x_atm = (x_atm.unsqueeze(1) - f_val / f1_val).squeeze(1)

    min_var = (0.01 ** 2) * leg.T
    sig_atm = torch.clamp(
        torch.sqrt(torch.clamp(svi.w(x_atm.unsqueeze(1)).squeeze(1), min=min_var) / leg.T),
        min=0.01,
    )

    # Invert 25C, 25P, 10C, 10P strikes and vols
    d_targets = [0.25, -0.25, 0.10, -0.10]
    phis = [1.0, -1.0, 1.0, -1.0]
    v_out: List[Tensor] = []

    for d_val, p_val in zip(d_targets, phis):
        d_ten = torch.full((S_len,), d_val, dtype=torch.float64, device=dev)
        p_ten = torch.full((S_len,), p_val, dtype=torch.float64, device=dev)
        sig_curr = sig_atm.clone()
        for _ in range(16):
            K_curr = strike_from_delta(d_ten, sig_curr, leg, p_ten, DeltaConvention.FORWARD_PNA)
            x_curr = torch.log(K_curr / leg.forward)
            sig_curr = torch.clamp(
                torch.sqrt(torch.clamp(svi.w(x_curr.unsqueeze(1)).squeeze(1), min=min_var) / leg.T),
                min=0.01,
            )
        v_out.append(sig_curr)

    v25c, v25p, v10c, v10p = v_out
    rr25 = v25c - v25p
    bf25 = 0.5 * (v25c + v25p) - sig_atm
    rr10 = v10c - v10p
    bf10 = 0.5 * (v10c + v10p) - sig_atm

    return torch.stack([sig_atm, rr25, bf25, rr10, bf10], dim=-1)


def compute_raw_svi_residual_jacobian_theta(svi: RawSVI, x: Tensor) -> Tensor:
    """Evaluates analytical residual Jacobian R_theta = dR/dtheta of shape (S, 5, 5).

    Row k: [1, rho_s * (x_k - m) + h_k, b * (x_k - m), -w'(x_k), b * s / h_k]
    """
    S_len = svi.params.shape[0]
    dev = svi.params.device
    b = svi.params[:, 1:2]
    rho_s = svi.params[:, 2:3]
    m = svi.params[:, 3:4]
    s = svi.params[:, 4:5]

    delta = x - m
    hk = torch.sqrt(delta ** 2 + s ** 2)
    _, w1k, _ = svi.derivs(x)

    R_theta = torch.zeros(S_len, 5, 5, dtype=torch.float64, device=dev)
    R_theta[:, :, 0] = 1.0
    R_theta[:, :, 1] = rho_s * delta + hk
    R_theta[:, :, 2] = b * delta
    R_theta[:, :, 3] = -w1k
    R_theta[:, :, 4] = b * s / hk
    return R_theta


def compute_raw_svi_residual_jacobian_q(
    quotes: Tensor,
    leg: LegMarket,
    conv_list: Sequence[Tuple[DeltaConvention, AtmConvention, StrangleConvention]],
    svi: RawSVI,
    x_mkt: Tensor,
) -> Tensor:
    """Evaluates analytical residual Jacobian R_q = dR/dq of shape (S, 5, 5)."""
    S_len = quotes.shape[0]
    dev = quotes.device
    T = leg.T
    _, w1k, _ = svi.derivs(x_mkt)

    M_sigma = torch.tensor([
        [1.0, 0.0, 0.0, -0.5, 1.0],
        [1.0, -0.5, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.5, 1.0],
    ], dtype=torch.float64, device=dev)

    R_q_list: List[Tensor] = []
    d_targets = [-0.10, -0.25, 0.0, 0.25, 0.10]
    phis = [-1.0, -1.0, 0.0, 1.0, 1.0]

    for i in range(S_len):
        d_conv, atm_conv, _ = conv_list[i]
        T_i = float(T[i].item())
        sqrt_T = math.sqrt(T_i)
        df_f_i = float(leg.df_foreign[i].item())

        if d_conv in (DeltaConvention.FORWARD_PNA, DeltaConvention.SPOT_PNA):
            c_spot = df_f_i if d_conv == DeltaConvention.SPOT_PNA else 1.0
            diag_f = torch.zeros(5, dtype=torch.float64, device=dev)
            for k in range(5):
                w1_val = float(w1k[i, k].item())
                if k == 2:
                    sig_atm = float(quotes[i, 0].item())
                    if atm_conv == AtmConvention.ATMF:
                        diag_f[k] = -2.0 * sig_atm * T_i
                    else:
                        diag_f[k] = (w1_val - 2.0) * sig_atm * T_i
                else:
                    d_val = d_targets[k]
                    p_val = phis[k]
                    d1 = p_val * float(torch.special.ndtri(torch.tensor(p_val * d_val / c_spot, dtype=torch.float64)).item())
                    sig_k = float((M_sigma[k:k+1] @ quotes[i:i+1].T).item())
                    d2 = d1 - sig_k * sqrt_T
                    diag_f[k] = -w1_val * d2 * sqrt_T - 2.0 * sig_k * T_i
            R_q_i = torch.diag(diag_f) @ M_sigma
        else:
            leg_i = LegMarket(
                spot=leg.spot[i:i+1],
                df_foreign=leg.df_foreign[i:i+1],
                df_domestic=leg.df_domestic[i:i+1],
                T=leg.T[i:i+1],
            )
            p_i = svi.params[i]

            def eval_res(q_row: Tensor) -> Tensor:
                K_i, s_i = pillar_strikes(q_row.unsqueeze(0), leg_i, d_conv, atm_conv)
                x_eval = torch.log(K_i / leg_i.forward.unsqueeze(-1)).squeeze(0)
                w_eval = (s_i.squeeze(0) ** 2) * T_i
                delta_p = x_eval - p_i[3]
                hk_eval = torch.sqrt(delta_p ** 2 + p_i[4] ** 2)
                w_mod = p_i[0] + p_i[1] * (p_i[2] * delta_p + hk_eval)
                return w_mod - w_eval

            R_q_i = cast(Tensor, torch.autograd.functional.jacobian(eval_res, quotes[i]))  # type: ignore[no-untyped-call]

        R_q_list.append(R_q_i)

    return torch.stack(R_q_list, dim=0)


def compute_canonical_feature_jacobian_theta(svi: RawSVI, leg: LegMarket) -> Tensor:
    """Evaluates analytical Jacobian Psi_theta = d xi / d theta of shape (S, 5, 5)."""
    S_len = svi.params.shape[0]
    dev = svi.params.device
    T = leg.T
    sqrt_T = torch.sqrt(T)

    x_atm = 0.5 * svi.w(torch.zeros(S_len, 1, dtype=torch.float64, device=dev)).squeeze(1)
    for _ in range(8):
        w_val, w1_val, _ = svi.derivs(x_atm.unsqueeze(1))
        f_val = x_atm.unsqueeze(1) - 0.5 * w_val
        f1_val = 1.0 - 0.5 * w1_val
        x_atm = (x_atm.unsqueeze(1) - f_val / f1_val).squeeze(1)
    w_atm, _, _ = svi.derivs(x_atm.unsqueeze(1))
    min_var = (0.01 ** 2) * T
    v_atm = torch.clamp(torch.sqrt(torch.clamp(w_atm.squeeze(1), min=min_var) / T), min=0.01)

    def eval_gv(x_k: Tensor, v_k: Tensor, d2_k: Tensor) -> Tensor:
        delta_k = x_k - svi.params[:, 3]
        hk = torch.sqrt(delta_k ** 2 + svi.params[:, 4] ** 2)
        _, w1_k, _ = svi.derivs(x_k.unsqueeze(1))
        w1_s = w1_k.squeeze(1)
        grad_w = torch.stack([
            torch.ones_like(x_k),
            svi.params[:, 2] * delta_k + hk,
            svi.params[:, 1] * delta_k,
            -w1_s,
            svi.params[:, 1] * svi.params[:, 4] / hk,
        ], dim=-1)
        denom = 2.0 * v_k * T + w1_s * d2_k * sqrt_T
        return grad_w / denom.unsqueeze(-1)

    gv_atm = eval_gv(x_atm, v_atm, -v_atm * sqrt_T)

    d_targets = [0.25, -0.25, 0.10, -0.10]
    phis = [1.0, -1.0, 1.0, -1.0]
    gv_wings: List[Tensor] = []

    for d_val, p_val in zip(d_targets, phis):
        d1 = p_val * torch.special.ndtri(torch.tensor(p_val * d_val, dtype=torch.float64, device=dev))
        v_curr = v_atm.clone()
        x_curr = torch.zeros(S_len, dtype=torch.float64, device=dev)
        for _ in range(16):
            x_curr = -d1 * v_curr * sqrt_T + 0.5 * (v_curr ** 2) * T
            w_curr = svi.w(x_curr.unsqueeze(1)).squeeze(1)
            v_curr = torch.clamp(torch.sqrt(torch.clamp(w_curr, min=min_var) / T), min=0.01)
        d2 = d1 - v_curr * sqrt_T
        gv_wings.append(eval_gv(x_curr, v_curr, d2))

    gv_25c, gv_25p, gv_10c, gv_10p = gv_wings
    Psi_theta = torch.stack([
        gv_atm,
        gv_25c - gv_25p,
        0.5 * (gv_25c + gv_25p) - gv_atm,
        gv_10c - gv_10p,
        0.5 * (gv_10c + gv_10p) - gv_atm,
    ], dim=1)
    return Psi_theta


class ImplicitCanonicalMap(torch.autograd.Function):
    """Autograd function computing canonical features with analytical IFT reverse-mode solve."""

    @staticmethod
    def forward(
        ctx: Any,
        quotes: Tensor,
        conv: Sequence[Tuple[DeltaConvention, AtmConvention, StrangleConvention]],
        leg: LegMarket,
        canonicalizer: SmileCanonicalizer,
    ) -> Tensor:
        smile = canonicalizer.canonicalize_unmemoized(quotes, conv, leg)
        ctx.save_for_backward(smile.r_theta, smile.r_q, smile.psi_theta)
        return smile.xi

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        R_theta, R_q, Psi_theta = ctx.saved_tensors
        grad_output = cast(Tensor, grad_outputs[0])

        bar_theta = torch.einsum("si,sij->sj", grad_output, Psi_theta)
        lambda_vec = torch.linalg.solve(
            R_theta.mT,
            bar_theta.unsqueeze(-1),
        ).squeeze(-1)
        grad_quotes = -torch.einsum("si,sij->sj", lambda_vec, R_q)

        return grad_quotes, None, None, None


def calendar_check(w_cross: Tensor, T: Tensor, tol: float = 1e-14) -> Tensor:
    """Verifies monotonic total variance growth across calendar expiries.

    Args:
        w_cross: Total variance tensor of shape (S_T, M) across sorted expiries.
        T: Expiry tenors of shape (S_T,).
        tol: Permissible tolerance for numerical variance decrease (default 1e-14).

    Returns:
        Boolean tensor of shape (S_T,) indicating validity of each expiry transition.
    """
    S_T = w_cross.shape[0]
    dev = w_cross.device
    if S_T <= 1:
        return torch.ones((S_T,), dtype=torch.bool, device=dev)

    order = torch.argsort(T)
    w_sorted = w_cross[order]

    diffs = w_sorted[1:] - w_sorted[:-1]
    step_valid = torch.all(diffs >= -tol, dim=-1)

    valid_sorted = torch.cat([torch.tensor([True], dtype=torch.bool, device=dev), step_valid])
    inv_order = torch.argsort(order)
    return valid_sorted[inv_order]


class SmileCanonicalizer:
    """Canonicalizes raw market quotes into convention-invariant smile features and IFT Jacobian."""

    def __init__(self, cache_size: int = 4096) -> None:
        self.cache_size = cache_size
        self._cache: OrderedDict[bytes, CanonicalSmile] = OrderedDict()

    @staticmethod
    def cache_key(
        quotes_row: Sequence[float],
        leg_row: Sequence[float],
        conv: Tuple[str, str, str],
    ) -> bytes:
        """Computes 128-bit Blake2b hash key for a market state slice."""
        h = hashlib.blake2b(digest_size=16)
        for q in quotes_row:
            h.update(struct.pack(">d", float(q)))
        for val in leg_row:
            h.update(struct.pack(">d", float(val)))
        for c in conv:
            h.update(c.encode("utf-8"))
        return h.digest()

    def canonicalize(
        self,
        quotes: Tensor,
        conv: Sequence[Tuple[DeltaConvention, AtmConvention, StrangleConvention]],
        leg: LegMarket,
    ) -> CanonicalSmile:
        """Canonicalizes market smile quotes across conventions.

        Args:
            quotes: Market quotes tensor of shape (S, 5) or (5,), float64.
            conv: Sequence of convention tuples of length S.
            leg: LegMarket container of length S.

        Returns:
            CanonicalSmile container.
        """
        quotes_in = quotes.unsqueeze(0) if quotes.dim() == 1 else quotes
        S_len = quotes_in.shape[0]
        batch_key: Optional[bytes] = None

        if self.cache_size > 0:
            h_batch = hashlib.blake2b(digest_size=16)
            for i in range(S_len):
                q_row = [float(val) for val in quotes_in[i].tolist()]
                l_row = [
                    float(leg.spot[i].item()),
                    float(leg.df_foreign[i].item()),
                    float(leg.df_domestic[i].item()),
                    float(leg.T[i].item()),
                ]
                conv_str = (conv[i][0].value, conv[i][1].value, conv[i][2].value)
                h_batch.update(self.cache_key(q_row, l_row, conv_str))
            batch_key = h_batch.digest()

            if batch_key in self._cache:
                self._cache.move_to_end(batch_key)
                return self._cache[batch_key]

        result = self.canonicalize_unmemoized(quotes_in, conv, leg)

        if self.cache_size > 0 and batch_key is not None:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[batch_key] = result

        return result

    def canonicalize_unmemoized(
        self,
        quotes: Tensor,
        conv: Sequence[Tuple[DeltaConvention, AtmConvention, StrangleConvention]],
        leg: LegMarket,
    ) -> CanonicalSmile:
        S_len = quotes.shape[0]
        dev = quotes.device

        # Extract strikes and market vols
        all_same_conv = all(c == conv[0] for c in conv)
        if all_same_conv:
            d_c, a_c, _ = conv[0]
            try:
                K_all, sig_all = pillar_strikes(quotes, leg, d_c, a_c)
            except Exception as exc:
                raise SmileArbitrageError(f"Pillar strike resolution failed: {exc}") from exc
        else:
            K_list = []
            sig_list = []
            for i in range(S_len):
                d_c, a_c, _ = conv[i]
                leg_i = LegMarket(
                    spot=leg.spot[i:i+1],
                    df_foreign=leg.df_foreign[i:i+1],
                    df_domestic=leg.df_domestic[i:i+1],
                    T=leg.T[i:i+1],
                )
                try:
                    k_i, s_i = pillar_strikes(quotes[i:i+1], leg_i, d_c, a_c)
                except Exception as exc:
                    raise SmileArbitrageError(f"Pillar strike resolution failed for state {i}: {exc}") from exc
                K_list.append(k_i)
                sig_list.append(s_i)
            K_all = torch.cat(K_list, dim=0)
            sig_all = torch.cat(sig_list, dim=0)

        if torch.any(sig_all <= 0.0) or torch.any(torch.isnan(sig_all)):
            raise SmileArbitrageError("Market quotes yield non-positive or NaN volatilities")

        F = leg.forward
        x_mkt = torch.log(K_all / F.unsqueeze(-1))
        w_mkt = (sig_all ** 2) * leg.T.unsqueeze(-1)

        raw_svi, rmse_raw = fit_raw_svi_5pt(x_mkt, w_mkt)
        cert = certify_slice(raw_svi, quotes[:, 0], leg.T)

        slice_params_list: List[Tensor] = []
        used_ssvi_list: List[bool] = []
        min_g_list: List[float] = []

        for i in range(S_len):
            if bool(cert.ok[i].item()) and float(rmse_raw[i].item()) <= 0.01:
                slice_params_list.append(raw_svi.params[i])
                used_ssvi_list.append(False)
                min_g_list.append(float(cert.min_g[i].item()))
            else:
                # Trigger SSVI fallback
                try:
                    params_ssvi, _ = fit_ssvi_slice(x_mkt[i:i+1], w_mkt[i:i+1])
                    th_i = params_ssvi[:, 0]
                    r_i = params_ssvi[:, 1]
                    ph_i = params_ssvi[:, 2]
                    svi_fallback = ssvi_to_raw_svi(th_i, r_i, ph_i)
                    cert_fb = certify_slice(svi_fallback, quotes[i:i+1, 0], leg.T[i:i+1])
                    if not bool(cert_fb.ok[0].item()):
                        raise SmileArbitrageError(f"SSVI fallback failed certification on state {i}")
                    slice_params_list.append(svi_fallback.params[0])
                    used_ssvi_list.append(True)
                    min_g_list.append(float(cert_fb.min_g[0].item()))
                except Exception as exc:
                    raise SmileArbitrageError(f"Double infeasible quotes for state {i}: {exc}") from exc

        slice_params = torch.stack(slice_params_list, dim=0)
        final_svi = RawSVI(slice_params)
        used_ssvi = torch.tensor(used_ssvi_list, dtype=torch.bool, device=dev)
        min_g = torch.tensor(min_g_list, dtype=torch.float64, device=dev)

        # Compute canonical features xi
        ref_conv = (DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)
        xi_list: List[Tensor] = []
        for i in range(S_len):
            if conv[i] == ref_conv and not used_ssvi_list[i]:
                xi_list.append(quotes[i])
            else:
                leg_i = LegMarket(
                    spot=leg.spot[i:i+1],
                    df_foreign=leg.df_foreign[i:i+1],
                    df_domestic=leg.df_domestic[i:i+1],
                    T=leg.T[i:i+1],
                )
                xi_i = canonical_features(RawSVI(slice_params[i:i+1]), leg_i)
                xi_list.append(xi_i[0])
        xi = torch.stack(xi_list, dim=0)

        # Compute Jacobian d xi / d quotes via analytical IFT
        R_theta = compute_raw_svi_residual_jacobian_theta(final_svi, x_mkt)
        R_q = compute_raw_svi_residual_jacobian_q(quotes, leg, conv, final_svi, x_mkt)
        Psi_theta = compute_canonical_feature_jacobian_theta(final_svi, leg)

        jacobian = torch.zeros(S_len, 5, 5, dtype=torch.float64, device=dev)

        for i in range(S_len):
            if conv[i] == ref_conv and not used_ssvi_list[i]:
                jacobian[i] = torch.eye(5, dtype=torch.float64, device=dev)
                R_theta[i] = torch.eye(5, dtype=torch.float64, device=dev)
                R_q[i] = -torch.eye(5, dtype=torch.float64, device=dev)
                Psi_theta[i] = torch.eye(5, dtype=torch.float64, device=dev)
            elif used_ssvi_list[i]:
                # On SSVI fallback, evaluate Jacobian strictly on the SSVI manifold
                h = 1e-6
                J_mat = torch.zeros(5, 5, dtype=torch.float64, device=dev)
                q_base = quotes[i:i+1].detach()
                leg_i = LegMarket(
                    spot=leg.spot[i:i+1],
                    df_foreign=leg.df_foreign[i:i+1],
                    df_domestic=leg.df_domestic[i:i+1],
                    T=leg.T[i:i+1],
                )
                for j in range(5):
                    q_p = q_base.clone()
                    q_p[0, j] += h
                    K_p, sig_p = pillar_strikes(q_p, leg_i, conv[i][0], conv[i][1])
                    xp = torch.log(K_p / leg_i.forward.unsqueeze(-1))
                    wp = (sig_p ** 2) * leg_i.T.unsqueeze(-1)
                    params_p, _ = fit_ssvi_slice(xp, wp)
                    svi_p = ssvi_to_raw_svi(params_p[:, 0], params_p[:, 1], params_p[:, 2])
                    xi_p = canonical_features(svi_p, leg_i)[0]

                    q_m = q_base.clone()
                    q_m[0, j] -= h
                    K_m, sig_m = pillar_strikes(q_m, leg_i, conv[i][0], conv[i][1])
                    xm = torch.log(K_m / leg_i.forward.unsqueeze(-1))
                    wm = (sig_m ** 2) * leg_i.T.unsqueeze(-1)
                    params_m, _ = fit_ssvi_slice(xm, wm)
                    svi_m = ssvi_to_raw_svi(params_m[:, 0], params_m[:, 1], params_m[:, 2])
                    xi_m = canonical_features(svi_m, leg_i)[0]

                    J_mat[:, j] = (xi_p - xi_m) / (2.0 * h)
                jacobian[i] = J_mat
                R_theta[i] = torch.eye(5, dtype=torch.float64, device=dev)
                R_q[i] = -J_mat
                Psi_theta[i] = torch.eye(5, dtype=torch.float64, device=dev)
            else:
                dtheta_dq_i = -torch.linalg.solve(R_theta[i], R_q[i])
                jacobian[i] = Psi_theta[i] @ dtheta_dq_i

        # ATMF volatility: w(0) / T
        w_atmf = final_svi.w(torch.zeros(S_len, 1, dtype=torch.float64, device=dev)).squeeze(1)
        min_var_leg = (0.01 ** 2) * leg.T
        sigma_atmf = torch.clamp(
            torch.sqrt(torch.clamp(w_atmf, min=min_var_leg) / leg.T),
            min=0.01,
        )

        # Asymptotic Lee slope L = b * (1 + |rho_s|)
        lee_slope = slice_params[:, 1] * (1.0 + torch.abs(slice_params[:, 2]))

        return CanonicalSmile(
            slice_params=slice_params,
            xi=xi,
            jacobian=jacobian,
            sigma_atmf=sigma_atmf,
            min_g=min_g,
            used_ssvi=used_ssvi,
            lee_slope=lee_slope,
            _R_theta=R_theta,
            _R_q=R_q,
            _Psi_theta=Psi_theta,
        )
