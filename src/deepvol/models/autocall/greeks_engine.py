"""greeks_engine.py — End-to-End Differentiable Chain-Rule Greeks Assembler (Task 3.1).

Implements SPEC-DEEPVOL-EQ-AUTOCALL-001 §1.4, Symbolic Proofs W2, W3, W5, W6, and Gates
G1.8, G1.9, G2.3, G2.4 with strict precision boundary segregation (REM-S0-01):
  - All outer Greek inputs, spline/surface Jacobians, and outputs execute in IEEE-754
    double precision (`torch.float64` / `np.float64`) and convert to clean Python
    `float` / `dict` / `list` structures for API responses.
  - Only the internal `MaskedEQAutocallSurrogate` residual pass executes in `torch.float32`
    (casting via `.to_tier2_float32_tensors()` at entry and casting residual outputs back to
    `torch.float64` before W1 affine cash-flow and PV assembly).
  - Supports all 8 Greek queries (`Delta`, `Gamma`, `MultivariateAggregatedCashGamma`,
    `Vega`, `Rho`, `DividendsSensitivities`, `Theta`, `Cega`), fast O(1) 2-JVP directional
    Euler CashGamma (`compute_euler_cash_gamma`), and vectorized GPU batch Greeks
    (`compute_batched_surrogate_greeks`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch

from deepvol.models.autocall.analytical_baseline import (
    AutocallBaselineOutput,
    _black76_put_torch,
    _digital_call_spread_prob_torch,
    compute_analytical_autocall_baseline,
)
from deepvol.models.autocall.dupire_qmc_solver import RegularizedVolSurface
from deepvol.models.autocall.pre_post_processor import (
    AutocallEdgeCol,
    AutocallNodeCol,
    AutocallPostprocOutput,
    AutocallPreprocState,
    AutocallSchedCol,
    assemble_autocall_cashflows_and_pv,
    preprocess_eq_autocall,
)
from deepvol.models.autocall.seasoned_collapser import _year_fraction
from deepvol.surrogates.eq_autocall_egno import (
    AutocallSurrogateOutput,
    MaskedEQAutocallSurrogate,
    assemble_surrogate_product_value,
)


@dataclass
class AutocallGreeksResult:
    """Complete valuation, cash-flow schedule, and 8-Greek sensitivity bundle in float64."""

    product_value: float
    cash_flows: list[Any]
    delta: dict[str, float] | None = None
    gamma: dict[str, dict[str, float]] | None = None
    multivariate_aggregated_cash_gamma: float | None = None
    vega: dict[str, list[list[float]]] | None = None
    rho: dict[str, list[float]] | None = None
    dividends_sensitivities: dict[str, list[float]] | None = None
    theta: float | None = None
    cega: dict[str, dict[str, float]] | None = None

    # Tensor & NumPy float64 representations in original caller asset order (pi_lex^{-1})
    delta_tensor: torch.Tensor | None = None
    gamma_tensor: torch.Tensor | None = None
    vega_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    rho_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    dividends_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    cega_tensor: torch.Tensor | None = None

    reporting_currency: str = "RUB"
    standard_error: float = 0.0
    cash_to_settle_today: float = 0.0
    postproc_output: AutocallPostprocOutput | None = field(default=None, repr=False)

    @property
    def cashflows(self) -> list[Any]:
        """Alias for `cash_flows`."""
        return self.cash_flows


@dataclass
class BatchedSurrogateGreeksOutput:
    """Vectorized batched valuation and first-order Greek sensitivities in torch.float64."""

    product_value: torch.Tensor
    stopping_probs: torch.Tensor
    survival_probs: torch.Tensor
    coupon_probs: torch.Tensor
    put_loss: torch.Tensor
    delta: torch.Tensor
    vega_atm: torch.Tensor
    vega_surface_feats: torch.Tensor
    rho_df: torch.Tensor
    cega: torch.Tensor
    cash_gamma_euler: torch.Tensor | None = None


@dataclass
class _SurfaceTorchCache:
    """Cached torch.float64 spline tensors for a RegularizedVolSurface on a specific device."""

    basis_c: torch.Tensor
    slope_left: torch.Tensor
    slope_right: torch.Tensor
    e_first: torch.Tensor
    e_last: torch.Tensor
    quotes: torch.Tensor
    dedup_t: torch.Tensor
    slice_c: torch.Tensor | None = None
    slice_slope_left: torch.Tensor | None = None
    slice_slope_right: torch.Tensor | None = None


_GLOBAL_SURFACE_TORCH_CACHE: dict[tuple[bytes, bytes, bytes, float, torch.device], _SurfaceTorchCache] = {}


def _get_surface_torch_cache(
    surf: RegularizedVolSurface,
    device: torch.device,
) -> _SurfaceTorchCache:
    """Retrieves or builds the cached torch.float64 spline basis tensors for `surf` on `device`."""
    cache_attr = "_greeks_torch_cache"
    cache_dict: dict[torch.device, _SurfaceTorchCache] | None = getattr(surf, cache_attr, None)
    if cache_dict is None:
        cache_dict = {}
        setattr(surf, cache_attr, cache_dict)

    if device in cache_dict:
        return cache_dict[device]

    g_key = (
        surf.strikes.tobytes(),
        surf.tenors.tobytes(),
        surf.raw_quotes.tobytes(),
        float(surf.min_vol),
        device,
    )
    cached_global = _GLOBAL_SURFACE_TORCH_CACHE.get(g_key)
    if cached_global is not None:
        cache_dict[device] = cached_global
        return cached_global

    surf._ensure_basis_spline()
    if surf._fast_basis_coeffs is None or surf._basis_slope_left is None or surf._basis_slope_right is None:
        raise RuntimeError("RegularizedVolSurface basis spline failed to initialize.")

    n_u = surf.n_unique_strikes
    e_first = torch.zeros(n_u, dtype=torch.float64, device=device)
    e_first[0] = 1.0
    e_last = torch.zeros(n_u, dtype=torch.float64, device=device)
    e_last[-1] = 1.0

    slice_c = (
        torch.as_tensor(surf._fast_slice_coeffs, dtype=torch.float64, device=device)
        if getattr(surf, "_fast_slice_coeffs", None) is not None
        else None
    )
    slice_slope_left = (
        torch.as_tensor(surf._slopes_left, dtype=torch.float64, device=device)
        if getattr(surf, "_slopes_left", None) is not None
        else None
    )
    slice_slope_right = (
        torch.as_tensor(surf._slopes_right, dtype=torch.float64, device=device)
        if getattr(surf, "_slopes_right", None) is not None
        else None
    )

    entry = _SurfaceTorchCache(
        basis_c=torch.as_tensor(surf._fast_basis_coeffs, dtype=torch.float64, device=device),
        slope_left=torch.as_tensor(surf._basis_slope_left, dtype=torch.float64, device=device),
        slope_right=torch.as_tensor(surf._basis_slope_right, dtype=torch.float64, device=device),
        e_first=e_first,
        e_last=e_last,
        quotes=torch.as_tensor(surf.quotes, dtype=torch.float64, device=device),
        dedup_t=torch.as_tensor(surf.dedup_matrix.T, dtype=torch.float64, device=device),
        slice_c=slice_c,
        slice_slope_left=slice_slope_left,
        slice_slope_right=slice_slope_right,
    )
    _GLOBAL_SURFACE_TORCH_CACHE[g_key] = entry
    cache_dict[device] = entry
    return entry


def _strike_basis_u_torch(
    surf: RegularizedVolSurface,
    k_query: torch.Tensor,
    cache: _SurfaceTorchCache,
) -> torch.Tensor:
    """Evaluates the clamped cubic strike spline basis vector w_u(K) in R^{N_unique} in torch.float64."""
    k_det = float(k_query.detach().cpu().item())
    k_min = float(surf.strikes[0])
    k_max = float(surf.strikes[-1])

    if k_det < k_min:
        return cache.e_first + (k_query - k_min) * cache.slope_left
    if k_det > k_max:
        return cache.e_last + (k_query - k_max) * cache.slope_right

    seg_idx = int(np.searchsorted(surf.strikes, k_det, side="right")) - 1
    seg_idx = max(0, min(seg_idx, surf.n_unique_strikes - 2))
    dx = k_query - float(surf.strikes[seg_idx])
    c_seg = cache.basis_c[:, seg_idx, :]  # (4, N_unique)
    return ((c_seg[0] * dx + c_seg[1]) * dx + c_seg[2]) * dx + c_seg[3]


def _strike_basis_u_batch_torch(
    surf: RegularizedVolSurface,
    k_queries: torch.Tensor,
    cache: _SurfaceTorchCache,
) -> torch.Tensor:
    """Vectorized evaluation of strike spline basis vectors w_u(K) in R^{K x N_unique} in torch.float64."""
    device = k_queries.device
    k_det_np = k_queries.detach().cpu().numpy()
    k_min = float(surf.strikes[0])
    k_max = float(surf.strikes[-1])

    seg_idx = np.clip(
        np.searchsorted(surf.strikes, k_det_np, side="right") - 1,
        0,
        surf.n_unique_strikes - 2,
    )
    k_left = torch.as_tensor(surf.strikes[seg_idx], dtype=torch.float64, device=device)
    dx = (k_queries - k_left).unsqueeze(-1)  # (K, 1)
    c_seg = cache.basis_c[:, seg_idx, :]  # (4, K, N_unique)
    w_in = ((c_seg[0] * dx + c_seg[1]) * dx + c_seg[2]) * dx + c_seg[3]

    if bool(np.any(k_det_np < k_min)):
        w_left = cache.e_first.unsqueeze(0) + (k_queries - k_min).unsqueeze(-1) * cache.slope_left.unsqueeze(0)
        mask_left = torch.as_tensor(k_det_np < k_min, dtype=torch.bool, device=device).unsqueeze(-1)
        w_in = torch.where(mask_left, w_left, w_in)

    if bool(np.any(k_det_np > k_max)):
        w_right = cache.e_last.unsqueeze(0) + (k_queries - k_max).unsqueeze(-1) * cache.slope_right.unsqueeze(0)
        mask_right = torch.as_tensor(k_det_np > k_max, dtype=torch.bool, device=device).unsqueeze(-1)
        w_in = torch.where(mask_right, w_right, w_in)

    return w_in


def _eval_reg_surface_iv_batch_torch(
    surf: RegularizedVolSurface,
    t_queries_np: npt.NDArray[np.float64],
    k_queries: torch.Tensor,
    quote_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized torch.float64 evaluation of sigma_IV(t_j, K_j) across K query pairs simultaneously."""
    device = k_queries.device
    cache = _get_surface_torch_cache(surf, device)
    t_vals = np.maximum(np.asarray(t_queries_np, dtype=np.float64), 1e-8)

    if (
        cache.slice_c is not None
        and cache.slice_slope_left is not None
        and cache.slice_slope_right is not None
    ):
        k_det_np = k_queries.detach().cpu().numpy()
        k_min = float(surf.strikes[0])
        k_max = float(surf.strikes[-1])
        seg_idx = np.clip(
            np.searchsorted(surf.strikes, k_det_np, side="right") - 1,
            0,
            surf.n_unique_strikes - 2,
        )
        k_left = torch.as_tensor(surf.strikes[seg_idx], dtype=torch.float64, device=device)
        dx = k_queries - k_left

        w_raw: torch.Tensor | None = None
        if quote_delta is not None:
            w_u_delta = _strike_basis_u_batch_torch(surf, k_queries.detach(), cache)
            w_raw = torch.matmul(w_u_delta, cache.dedup_t.T)

        if surf.n_exp == 1:
            c0 = cache.slice_c[:, seg_idx, 0]
            val = ((c0[0] * dx + c0[1]) * dx + c0[2]) * dx + c0[3]
            if bool(np.any(k_det_np < k_min)):
                mask_left = torch.as_tensor(k_det_np < k_min, dtype=torch.bool, device=device)
                val_left = cache.quotes[0, 0] + cache.slice_slope_left[0] * (k_queries - k_min)
                val = torch.where(mask_left, val_left, val)
            if bool(np.any(k_det_np > k_max)):
                mask_right = torch.as_tensor(k_det_np > k_max, dtype=torch.bool, device=device)
                val_right = cache.quotes[0, -1] + cache.slice_slope_right[0] * (k_queries - k_max)
                val = torch.where(mask_right, val_right, val)
            if quote_delta is not None and w_raw is not None:
                val = val + (quote_delta[0].unsqueeze(0) * w_raw).sum(dim=-1)
            return torch.clamp(val, min=surf.min_vol)

        idx0 = np.clip(np.searchsorted(surf.tenors, t_vals, side="right") - 1, 0, surf.n_exp - 2)
        idx1 = idx0 + 1
        c0 = cache.slice_c[:, seg_idx, idx0]
        c1 = cache.slice_c[:, seg_idx, idx1]
        v0 = ((c0[0] * dx + c0[1]) * dx + c0[2]) * dx + c0[3]
        v1 = ((c1[0] * dx + c1[1]) * dx + c1[2]) * dx + c1[3]

        if bool(np.any(k_det_np < k_min)):
            mask_left = torch.as_tensor(k_det_np < k_min, dtype=torch.bool, device=device)
            v0_left = cache.quotes[idx0, 0] + cache.slice_slope_left[idx0] * (k_queries - k_min)
            v1_left = cache.quotes[idx1, 0] + cache.slice_slope_left[idx1] * (k_queries - k_min)
            v0 = torch.where(mask_left, v0_left, v0)
            v1 = torch.where(mask_left, v1_left, v1)
        if bool(np.any(k_det_np > k_max)):
            mask_right = torch.as_tensor(k_det_np > k_max, dtype=torch.bool, device=device)
            v0_right = cache.quotes[idx0, -1] + cache.slice_slope_right[idx0] * (k_queries - k_max)
            v1_right = cache.quotes[idx1, -1] + cache.slice_slope_right[idx1] * (k_queries - k_max)
            v0 = torch.where(mask_right, v0_right, v0)
            v1 = torch.where(mask_right, v1_right, v1)

        if quote_delta is not None and w_raw is not None:
            v0 = v0 + (quote_delta[idx0] * w_raw).sum(dim=-1)
            v1 = v1 + (quote_delta[idx1] * w_raw).sum(dim=-1)

        v0 = torch.clamp(v0, min=surf.min_vol)
        v1 = torch.clamp(v1, min=surf.min_vol)

        t0_np = surf.tenors[idx0]
        t1_np = surf.tenors[idx1]
        alpha_np = (t_vals - t0_np) / np.maximum(t1_np - t0_np, 1e-12)

        t0_t = torch.as_tensor(t0_np, dtype=torch.float64, device=device)
        t1_t = torch.as_tensor(t1_np, dtype=torch.float64, device=device)
        alpha_t = torch.as_tensor(alpha_np, dtype=torch.float64, device=device)
        t_vals_t = torch.as_tensor(t_vals, dtype=torch.float64, device=device)

        w0 = (v0 * v0) * t0_t
        w1 = (v1 * v1) * t1_t
        w = (1.0 - alpha_t) * w0 + alpha_t * w1
        v_interp = torch.sqrt(torch.clamp(w / t_vals_t, min=surf.min_vol * surf.min_vol))

        if bool(np.any(t_vals <= float(surf.tenors[0]))):
            mask_lo = torch.as_tensor(t_vals <= float(surf.tenors[0]), dtype=torch.bool, device=device)
            v_interp = torch.where(mask_lo, v0, v_interp)
        if bool(np.any(t_vals >= float(surf.tenors[-1]))):
            mask_hi = torch.as_tensor(t_vals >= float(surf.tenors[-1]), dtype=torch.bool, device=device)
            v_interp = torch.where(mask_hi, v1, v_interp)

        return v_interp

    w_u = _strike_basis_u_batch_torch(surf, k_queries, cache)  # (K, N_unique)
    w_raw = torch.matmul(w_u, cache.dedup_t.T) if quote_delta is not None else None

    if surf.n_exp == 1:
        val = (cache.quotes[0].unsqueeze(0) * w_u).sum(dim=-1)
        if quote_delta is not None and w_raw is not None:
            val = val + (quote_delta[0].unsqueeze(0) * w_raw).sum(dim=-1)
        return torch.clamp(val, min=surf.min_vol)

    idx0 = np.clip(np.searchsorted(surf.tenors, t_vals, side="right") - 1, 0, surf.n_exp - 2)
    idx1 = idx0 + 1

    v0 = (cache.quotes[idx0] * w_u).sum(dim=-1)
    v1 = (cache.quotes[idx1] * w_u).sum(dim=-1)
    if quote_delta is not None and w_raw is not None:
        v0 = v0 + (quote_delta[idx0] * w_raw).sum(dim=-1)
        v1 = v1 + (quote_delta[idx1] * w_raw).sum(dim=-1)
    v0 = torch.clamp(v0, min=surf.min_vol)
    v1 = torch.clamp(v1, min=surf.min_vol)

    t0_np = surf.tenors[idx0]
    t1_np = surf.tenors[idx1]
    alpha_np = (t_vals - t0_np) / np.maximum(t1_np - t0_np, 1e-12)

    t0_t = torch.as_tensor(t0_np, dtype=torch.float64, device=device)
    t1_t = torch.as_tensor(t1_np, dtype=torch.float64, device=device)
    alpha_t = torch.as_tensor(alpha_np, dtype=torch.float64, device=device)
    t_vals_t = torch.as_tensor(t_vals, dtype=torch.float64, device=device)

    w0 = (v0 * v0) * t0_t
    w1 = (v1 * v1) * t1_t
    w = (1.0 - alpha_t) * w0 + alpha_t * w1
    v_interp = torch.sqrt(torch.clamp(w / t_vals_t, min=surf.min_vol * surf.min_vol))

    if bool(np.any(t_vals <= float(surf.tenors[0]))):
        mask_lo = torch.as_tensor(t_vals <= float(surf.tenors[0]), dtype=torch.bool, device=device)
        v_interp = torch.where(mask_lo, v0, v_interp)
    if bool(np.any(t_vals >= float(surf.tenors[-1]))):
        mask_hi = torch.as_tensor(t_vals >= float(surf.tenors[-1]), dtype=torch.bool, device=device)
        v_interp = torch.where(mask_hi, v1, v_interp)

    return v_interp


def _eval_reg_surface_iv_torch(
    surf: RegularizedVolSurface,
    t: float,
    k_query: torch.Tensor,
    quote_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable torch.float64 evaluation of sigma_IV(t, K) wrt both K and raw surface quotes."""
    device = k_query.device
    cache = _get_surface_torch_cache(surf, device)
    t_val = max(float(t), 1e-8)
    w_u = _strike_basis_u_torch(surf, k_query, cache)
    w_raw = torch.mv(cache.dedup_t, w_u) if quote_delta is not None else None

    def _slice_vol(s_idx: int) -> torch.Tensor:
        val = torch.dot(cache.quotes[s_idx], w_u)
        if quote_delta is not None and w_raw is not None:
            val = val + torch.dot(quote_delta[s_idx], w_raw)
        return torch.clamp(val, min=surf.min_vol)

    if t_val <= float(surf.tenors[0]):
        return _slice_vol(0)
    if t_val >= float(surf.tenors[-1]):
        return _slice_vol(surf.n_exp - 1)

    idx = int(np.searchsorted(surf.tenors, t_val, side="right")) - 1
    idx = max(0, min(idx, surf.n_exp - 2))
    t0 = float(surf.tenors[idx])
    t1 = float(surf.tenors[idx + 1])
    v0 = _slice_vol(idx)
    v1 = _slice_vol(idx + 1)

    alpha = (t_val - t0) / max(t1 - t0, 1e-12)
    w0 = (v0 * v0) * t0
    w1 = (v1 * v1) * t1
    w = (1.0 - alpha) * w0 + alpha * w1
    return torch.sqrt(torch.clamp(w / t_val, min=surf.min_vol * surf.min_vol))


def _extract_node_surface_features_torch(
    surf: RegularizedVolSurface,
    t_short: float,
    f_short: torch.Tensor,
    t_term: float,
    f_term: torch.Tensor,
    quote_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable torch.float64 extraction of the 7 node volatility surface features."""
    f_stencil = f_term.detach()
    k_batch = torch.stack(
        [
            f_short,
            f_term,
            f_stencil,
            0.90 * f_stencil,
            1.10 * f_stencil,
            0.80 * f_stencil,
            1.20 * f_stencil,
        ],
        dim=0,
    )
    t_batch_np = np.array([t_short, t_term, t_term, t_term, t_term, t_term, t_term], dtype=np.float64)
    iv_batch = _eval_reg_surface_iv_batch_torch(surf, t_batch_np, k_batch, quote_delta)
    iv_short = iv_batch[0]
    iv_term = iv_batch[1]
    iv_c = iv_batch[2]
    iv_m10 = iv_batch[3]
    iv_p10 = iv_batch[4]
    iv_m20 = iv_batch[5]
    iv_p20 = iv_batch[6]

    iv_slope = iv_term - iv_short

    denom_skew = torch.clamp(0.20 * iv_c, min=1e-6)
    raw_skew = (iv_p10 - iv_m10) / denom_skew
    ssvi_rho = torch.tanh(raw_skew)

    denom_curv = torch.clamp(0.01 * iv_c, min=1e-6)
    raw_curv = (iv_p10 - 2.0 * iv_c + iv_m10) / denom_curv
    ssvi_eta = torch.clamp(raw_curv, min=1e-4)

    pc1 = iv_m20 - iv_c
    pc2 = iv_p20 - iv_c

    return torch.stack([iv_short, iv_term, iv_slope, ssvi_rho, ssvi_eta, pc1, pc2], dim=0)


def _eval_reg_surface_iv_torch_safe(
    surf: RegularizedVolSurface,
    t: float,
    k_query: torch.Tensor,
    quote_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    return _eval_reg_surface_iv_torch(surf, t, k_query, quote_delta)


def _curve_df_delta_multiplier(
    curve: Any,
    dates: Sequence[Any],
    pillar_deltas: torch.Tensor,
) -> torch.Tensor:
    """Computes the multiplicative discount factor adjustment exp(-Delta_r(d) * tau_365) from pillar quote deltas."""
    device = pillar_deltas.device
    if len(dates) == 0:
        return torch.ones(0, dtype=torch.float64, device=device)

    cache_attr = "_df_delta_mult_cache"
    cache_dict: dict[Any, tuple[torch.Tensor, torch.Tensor]] | None = getattr(curve, cache_attr, None)
    if cache_dict is None:
        cache_dict = {}
        try:
            setattr(curve, cache_attr, cache_dict)
        except Exception:
            cache_dict = None

    cache_key = (tuple(dates), device)
    if cache_dict is not None and cache_key in cache_dict:
        w_t, tau_t = cache_dict[cache_key]
    else:
        as_of = getattr(curve, "as_of_date", None)
        dcc_curve = getattr(curve, "dcc", "Act365")
        if hasattr(curve, "spline_weights") and callable(curve.spline_weights):
            w_np = np.asarray(curve.spline_weights(dates), dtype=np.float64)
        else:
            w_np = np.zeros((len(dates), pillar_deltas.numel()), dtype=np.float64)

        if as_of is not None:
            tau_np = np.array([_year_fraction(as_of, d, dcc_curve) for d in dates], dtype=np.float64)
        else:
            tau_np = np.asarray(dates, dtype=np.float64)

        w_t = torch.as_tensor(w_np, dtype=torch.float64, device=device)
        tau_t = torch.as_tensor(tau_np, dtype=torch.float64, device=device)
        if cache_dict is not None:
            cache_dict[cache_key] = (w_t, tau_t)

    scale = float(getattr(curve, "quote_scale", 0.01))
    space_coords = getattr(curve, "space_coords", "Rates")
    input_mode = getattr(curve, "_input_mode", None)

    if input_mode == "rates" or (input_mode is None and space_coords == "Rates"):
        delta_r = torch.mv(w_t, pillar_deltas) * scale
        return torch.exp(-delta_r * tau_t.clamp(min=0.0))
    if input_mode == "log_df":
        tenors_t = torch.as_tensor(curve.tenors, dtype=torch.float64, device=device).clamp(min=1e-12)
        delta_r_pil = -(pillar_deltas * scale) / tenors_t
        delta_r = torch.mv(w_t, delta_r_pil)
        return torch.exp(-delta_r * tau_t.clamp(min=0.0))
    if input_mode is None and space_coords == "LogarithmOfDiscountFactors":
        tenors_t = torch.as_tensor(curve.tenors, dtype=torch.float64, device=device)
        delta_y = -tenors_t * pillar_deltas * scale
        return torch.exp(torch.mv(w_t, delta_y))

    # DiscountFactors space
    df0_np = np.asarray(curve.df(dates), dtype=np.float64).reshape(-1)
    _, d_df_np = curve.df_and_d_df_d_pillar(dates, in_quote_units=True)
    df0_t = torch.as_tensor(df0_np, dtype=torch.float64, device=device).clamp(min=1e-15)
    d_df_t = torch.as_tensor(d_df_np, dtype=torch.float64, device=device)
    df_new = (df0_t + torch.mv(d_df_t, pillar_deltas)).clamp(min=1e-15)
    return df_new / df0_t


def evaluate_differentiable_pv(
    state_or_request: AutocallPreprocState | Any,
    *,
    spots_t0: torch.Tensor | None = None,
    corr_matrix: torch.Tensor | None = None,
    df_pay: torch.Tensor | None = None,
    df_obs: torch.Tensor | None = None,
    div_amounts: Mapping[str, torch.Tensor] | None = None,
    surface_quote_overrides: Mapping[str, torch.Tensor] | None = None,
    vol_quote_deltas: Mapping[str, torch.Tensor] | None = None,
    curve_pillar_deltas: torch.Tensor | None = None,
    surrogate: MaskedEQAutocallSurrogate | None = None,
    spots_in_original_order: bool = False,
    tau_shift: float = 0.0,
    return_postproc: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, AutocallPostprocOutput]:
    """Evaluates `ProductValue` with a 100% connected `torch.float64` computational graph (W2, W3, W6)."""
    if isinstance(state_or_request, AutocallPreprocState):
        base_state = state_or_request
    else:
        base_state = preprocess_eq_autocall(state_or_request)

    eff_surrogate = (
        None
        if (
            surrogate is not None
            and hasattr(surrogate, "is_zero_residual")
            and surrogate.is_zero_residual()
        )
        else surrogate
    )

    d_assets = base_state.d_assets
    m_rem = base_state.m_rem
    d_pad = base_state.d_max
    m_pad = base_state.m_max
    sorted_ids = base_state.sorted_identifiers

    # Determine execution device
    device = base_state.spots_t0.device
    for cand in (spots_t0, corr_matrix, df_pay, df_obs, curve_pillar_deltas):
        if isinstance(cand, torch.Tensor):
            device = cand.device
            break

    # 1. Canonical spot vector S_i(t_0) and moneyness M_{0,i} = S_i(t_0) / S_i(0)
    if spots_t0 is not None:
        sp_in = spots_t0.to(dtype=torch.float64, device=device)
        if spots_in_original_order:
            lex_idx = torch.as_tensor(base_state.lex_perm, dtype=torch.long, device=device)
            spots_sorted = sp_in.index_select(0, lex_idx)
        else:
            spots_sorted = sp_in
    else:
        spots_sorted = base_state.spots_t0.to(dtype=torch.float64, device=device)

    ref_spots = base_state.ref_spots.to(dtype=torch.float64, device=device)
    moneyness_t0 = spots_sorted / ref_spots

    # Handle seasoned trades with 0 remaining future observation periods
    if m_rem == 0 or base_state.already_autocalled:
        pv_transit = torch.tensor(base_state.pv_in_transit, dtype=torch.float64, device=device)
        if curve_pillar_deltas is not None and base_state.seasoned_state.in_transit_indices:
            for rec in base_state.seasoned_state.historical_records:
                if rec.status == "In-Transit" and (rec.coupon_paid != 0.0 or rec.notional_paid != 0.0):
                    mult = _curve_df_delta_multiplier(
                        base_state.discount_curve, [rec.payment_date], curve_pillar_deltas
                    )[0]
                    pv_transit = pv_transit + float(rec.discount_factor * (rec.coupon_paid + rec.notional_paid)) * (
                        mult - 1.0
                    )
        if return_postproc:
            post_zero = assemble_autocall_cashflows_and_pv(
                base_state,
                p_stop=torch.zeros(m_rem, dtype=torch.float64, device=device),
                q_cpn=torch.zeros(m_rem, dtype=torch.float64, device=device),
                l_put=torch.tensor(0.0, dtype=torch.float64, device=device),
            )
            post_zero.product_value = pv_transit
            return pv_transit, post_zero
        return pv_transit

    # 2. Observation & payment tenors (supporting optional 1-day Theta roll tau_shift)
    tau_obs_base = base_state.tau_obs.to(dtype=torch.float64, device=device)
    tau_pay_base = base_state.tau_pay.to(dtype=torch.float64, device=device)
    if tau_shift != 0.0:
        tau_obs_t = (tau_obs_base - float(tau_shift)).clamp(min=1e-6)
        tau_pay_t = (tau_pay_base - float(tau_shift)).clamp(min=1e-6)
        delta_tau_t = torch.cat([tau_obs_t[:1], tau_obs_t[1:] - tau_obs_t[:-1]], dim=0)
    else:
        tau_obs_t = tau_obs_base
        tau_pay_t = tau_pay_base
        delta_tau_t = base_state.delta_tau.to(dtype=torch.float64, device=device)

    # 3. Discount factors at observation dates (df_obs) and payment dates (df_pay)
    if df_obs is not None:
        df_obs_t = df_obs.to(dtype=torch.float64, device=device)
    else:
        if tau_shift != 0.0 and hasattr(base_state.discount_curve, "df"):
            df_obs_np = np.asarray(
                base_state.discount_curve.df(tau_obs_t.detach().cpu().numpy()),
                dtype=np.float64,
            ).reshape(-1)
            df_obs_t = torch.as_tensor(df_obs_np, dtype=torch.float64, device=device)
        else:
            df_obs_t = base_state.df_obs.to(dtype=torch.float64, device=device)
        if curve_pillar_deltas is not None:
            obs_mult = _curve_df_delta_multiplier(
                base_state.discount_curve,
                base_state.seasoned_state.future_obs_dates,
                curve_pillar_deltas,
            )
            df_obs_t = df_obs_t * obs_mult

    if df_pay is not None:
        df_pay_t = df_pay.to(dtype=torch.float64, device=device)
    else:
        if tau_shift != 0.0 and hasattr(base_state.discount_curve, "df"):
            df_pay_np = np.asarray(
                base_state.discount_curve.df(tau_pay_t.detach().cpu().numpy()),
                dtype=np.float64,
            ).reshape(-1)
            df_pay_t = torch.as_tensor(df_pay_np, dtype=torch.float64, device=device)
        else:
            df_pay_t = base_state.df_pay.to(dtype=torch.float64, device=device)
        if curve_pillar_deltas is not None:
            pay_mult = _curve_df_delta_multiplier(
                base_state.discount_curve,
                base_state.seasoned_state.future_pay_dates,
                curve_pillar_deltas,
            )
            df_pay_t = df_pay_t * pay_mult

    # 4. Buehler Escrowed Dividend Present Values, Forwards, and Cash-Dividend Leverage (W2 & W3)
    div_sched = base_state.dividend_schedule
    as_of_dt = base_state.seasoned_state.as_of_date

    if div_amounts is None and curve_pillar_deltas is None and tau_shift == 0.0:
        div_pv_t = div_sched.div_pv.to(dtype=torch.float64, device=device)
    else:
        div_pv_rows: list[torch.Tensor] = []
        for i, ticker in enumerate(sorted_ids):
            events_i = div_sched.dividend_events.get(ticker, [])
            if len(events_i) > 0:
                if div_amounts is not None and ticker in div_amounts:
                    amt_t = div_amounts[ticker].to(dtype=torch.float64, device=device)
                else:
                    amt_t = div_sched.dividend_amount_tensors[ticker].to(dtype=torch.float64, device=device)

                w_base_t = div_sched.dividend_pv_weights[ticker].to(dtype=torch.float64, device=device)
                inc_t = div_sched.dividend_obs_incidence[ticker].to(dtype=torch.float64, device=device)

                if curve_pillar_deltas is not None:
                    active_pay_dates = [ev.payment_date for ev in events_i]
                    div_df_mult = _curve_df_delta_multiplier(
                        base_state.discount_curve, active_pay_dates, curve_pillar_deltas
                    )
                    active_ex_mask = torch.tensor(
                        [1.0 if ev.ex_date > as_of_dt else 0.0 for ev in events_i],
                        dtype=torch.float64,
                        device=device,
                    )
                    eff_mult = torch.where(active_ex_mask > 0.5, div_df_mult, torch.ones_like(div_df_mult))
                    w_t = w_base_t * eff_mult
                else:
                    w_t = w_base_t

                if tau_shift != 0.0 and hasattr(base_state.discount_curve, "df"):
                    t_ex_arr = np.array([ev.tau_ex_act365 for ev in events_i], dtype=np.float64)
                    t_pay_arr = np.array([ev.tau_pay_act365 for ev in events_i], dtype=np.float64)
                    valid_mask = t_ex_arr > float(tau_shift)
                    t_pay_shift_arr = np.maximum(t_pay_arr - float(tau_shift), 1e-6)
                    df_new_arr = np.asarray(base_state.discount_curve.df(t_pay_shift_arr), dtype=np.float64).reshape(-1)
                    df_old_arr = np.asarray(base_state.discount_curve.df(t_pay_arr), dtype=np.float64).reshape(-1)
                    ratio_arr = np.where(valid_mask, df_new_arr / np.maximum(df_old_arr, 1e-15), 0.0)
                    w_t = w_base_t * torch.as_tensor(ratio_arr, dtype=torch.float64, device=device)

                pv_each = amt_t * w_t
                div_pv_row = torch.matmul(pv_each.unsqueeze(0), inc_t).squeeze(0)
            else:
                div_pv_row = torch.zeros(m_rem, dtype=torch.float64, device=device)
            div_pv_rows.append(div_pv_row)

        div_pv_t = torch.stack(div_pv_rows, dim=0)  # (D, M_rem)

    spots_col = spots_sorted.unsqueeze(1)       # (D, 1)
    ref_col = ref_spots.unsqueeze(1)            # (D, 1)

    escrowed_spot = torch.maximum(spots_col - div_pv_t, 0.01 * spots_col)
    r_tot_base = div_sched.r_tot_obs.to(dtype=torch.float64, device=device)
    if df_obs is None and curve_pillar_deltas is None and tau_shift == 0.0:
        growth_factor = torch.exp(r_tot_base)
    else:
        df_fin_base = div_sched.df_fin_obs.to(dtype=torch.float64, device=device).clamp(min=1e-15)
        df_repo_quanto_obs = torch.exp(r_tot_base) * df_fin_base
        if tau_shift != 0.0:
            tau_ratio = (tau_obs_t / tau_obs_base.clamp(min=1e-12)).unsqueeze(0)
            df_repo_quanto_obs = torch.pow(df_repo_quanto_obs.clamp(min=1e-15), tau_ratio)
        growth_factor = df_repo_quanto_obs / df_obs_t.clamp(min=1e-15).unsqueeze(0)

    forward_price = escrowed_spot * growth_factor
    forward_rel = forward_price / spots_col
    forward_perf = forward_price / ref_col
    cash_div_leverage = spots_col / escrowed_spot
    repo_spread_avg = div_sched.repo_spread_avg.to(dtype=torch.float64, device=device)

    first_param = next(eff_surrogate.parameters(), None) if eff_surrogate is not None else None
    surr_dtype = first_param.dtype if first_param is not None else torch.float32
    surr_dev = first_param.device if first_param is not None else device
    need_surr_grad_feats = eff_surrogate is not None and surr_dtype == torch.float64

    with torch.no_grad():
        base_spots_col = base_state.spots_t0.detach().to(dtype=torch.float64, device=device).unsqueeze(1)
        base_div_pv = div_sched.div_pv.detach().to(dtype=torch.float64, device=device)
        escrowed_base = torch.maximum(base_spots_col - base_div_pv, 0.01 * base_spots_col)
        forward_base_term = (escrowed_base * torch.exp(r_tot_base.detach()))[:, -1]

    # 5. Differentiable Sticky-Strike Implied Volatility & 7D Node Surface Features
    iv_rows: list[torch.Tensor] = []
    surf_feat_rows: list[torch.Tensor] = []
    tau_obs_np = tau_obs_t.detach().cpu().numpy()
    t_term = float(tau_obs_np[-1])
    if need_surr_grad_feats:
        t_queries_np = np.concatenate(
            [tau_obs_np, np.array([t_term, t_term, t_term, t_term, t_term], dtype=np.float64)]
        )
    else:
        t_queries_np = tau_obs_np

    for i, ticker in enumerate(sorted_ids):
        surf_i = base_state.vol_surfaces[ticker]
        q_delta_i: torch.Tensor | None = None
        if vol_quote_deltas is not None and ticker in vol_quote_deltas:
            q_delta_i = vol_quote_deltas[ticker].to(dtype=torch.float64, device=device)
        elif surface_quote_overrides is not None and ticker in surface_quote_overrides:
            raw_q_base = torch.as_tensor(surf_i.raw_quotes, dtype=torch.float64, device=device)
            q_delta_i = surface_quote_overrides[ticker].to(dtype=torch.float64, device=device) - raw_q_base

        if need_surr_grad_feats:
            if q_delta_i is None and tau_shift == 0.0:
                iv_m_vec = _eval_reg_surface_iv_batch_torch(
                    surf_i, tau_obs_np, forward_price[i], None
                )
                iv_rows.append(iv_m_vec)
                iv_short = iv_m_vec[0]
                iv_term = iv_m_vec[-1]
                iv_slope = iv_term - iv_short
                smile_4 = base_state.node_feats[0, i, 7:11].detach().to(dtype=torch.float64, device=device)
                surf_feat_rows.append(
                    torch.cat([torch.stack([iv_short, iv_term, iv_slope], dim=0), smile_4], dim=0)
                )
            else:
                f_stencil = forward_base_term[i]
                k_queries = torch.cat(
                    [
                        forward_price[i],
                        torch.stack(
                            [
                                f_stencil,
                                0.90 * f_stencil,
                                1.10 * f_stencil,
                                0.80 * f_stencil,
                                1.20 * f_stencil,
                            ],
                            dim=0,
                        ),
                    ],
                    dim=0,
                )
                iv_all = _eval_reg_surface_iv_batch_torch(surf_i, t_queries_np, k_queries, q_delta_i)
                iv_m_vec = iv_all[:m_rem]
                iv_rows.append(iv_m_vec)

                iv_short = iv_m_vec[0]
                iv_term = iv_m_vec[-1]
                iv_c = iv_all[m_rem]
                iv_m10 = iv_all[m_rem + 1]
                iv_p10 = iv_all[m_rem + 2]
                iv_m20 = iv_all[m_rem + 3]
                iv_p20 = iv_all[m_rem + 4]

                iv_slope = iv_term - iv_short
                denom_skew = torch.clamp(0.20 * iv_c, min=1e-6)
                ssvi_rho = torch.tanh((iv_p10 - iv_m10) / denom_skew)
                denom_curv = torch.clamp(0.01 * iv_c, min=1e-6)
                ssvi_eta = torch.clamp((iv_p10 - 2.0 * iv_c + iv_m10) / denom_curv, min=1e-4)
                pc1 = iv_m20 - iv_c
                pc2 = iv_p20 - iv_c
                surf_feat_rows.append(
                    torch.stack([iv_short, iv_term, iv_slope, ssvi_rho, ssvi_eta, pc1, pc2], dim=0)
                )
        else:
            iv_m_vec = _eval_reg_surface_iv_batch_torch(
                surf_i, tau_obs_np, forward_price[i], q_delta_i
            )
            iv_rows.append(iv_m_vec)

    iv_at_fwd_t = torch.stack(iv_rows, dim=0)  # (D, M_rem)
    v_perf_t = iv_at_fwd_t * torch.sqrt(tau_obs_t.clamp(min=1e-12)).unsqueeze(0)

    # 6. Correlation Matrix (symmetrized off-diagonal for Cega differentiability)
    if corr_matrix is not None:
        corr_in = corr_matrix.to(dtype=torch.float64, device=device)
        eye_d = torch.eye(d_assets, dtype=torch.float64, device=device)
        corr_sym = 0.5 * (corr_in + corr_in.transpose(0, 1)) * (1.0 - eye_d) + eye_d
    else:
        corr_in = base_state.corr_matrix.to(dtype=torch.float64, device=device)
        corr_sym = corr_in

    ko_barriers_t = base_state.ko_barriers.to(dtype=torch.float64, device=device)
    is_callable_t = base_state.is_callable.to(dtype=torch.float64, device=device)
    cpn_barriers_t = base_state.cpn_barriers.to(dtype=torch.float64, device=device)
    is_guaranteed_t = base_state.is_guaranteed_cpn.to(dtype=torch.float64, device=device)
    coupon_amounts_t = base_state.coupon_amounts.to(dtype=torch.float64, device=device)

    # 7. Analytical Stulz-Margrabe Moment-Matched Basket Proxy in torch.float64
    proxy_inputs = {
        "forward_perf": forward_perf,
        "v_perf": v_perf_t,
        "corr_matrix": corr_sym,
        "df_pay": df_pay_t,
        "ko_barriers": ko_barriers_t,
        "is_callable": is_callable_t,
        "cpn_barriers": cpn_barriers_t,
        "is_guaranteed_cpn": is_guaranteed_t,
        "coupon_amounts": coupon_amounts_t,
        "strike": base_state.strike,
        "option_gearing": base_state.option_gearing,
        "lower_strike": base_state.lower_strike,
        "ki_barrier": base_state.ki_barrier,
        "i_ki_past": base_state.i_ki_past,
        "c_mem_0": base_state.c_mem_0,
        "is_memory": base_state.is_memory,
        "ko_spread": base_state.ko_spread,
        "cpn_spread": base_state.cpn_spread,
        "ki_spread": base_state.ki_spread,
        "no_notional_repayment": base_state.no_notional_repayment,
        "autocall_coupon": base_state.autocall_coupon,
        "force_coupons_on_autocall": base_state.force_coupons_on_autocall,
        "coupon_barrier_type": base_state.coupon_barrier_type,
        "coupon_barrier_direction": base_state.coupon_barrier_direction,
        "is_long_basket": base_state.is_long_basket,
        "participation_call_strike": base_state.participation_call_strike,
        "participation_call_rate": base_state.participation_call_rate,
        "one_star_strike": base_state.one_star_strike,
        "one_star_coupon": base_state.one_star_coupon,
        "pv_in_transit": base_state.pv_in_transit,
        "already_autocalled": base_state.already_autocalled,
        "callability_index": base_state.callability_index,
    }
    baseline: AutocallBaselineOutput = compute_analytical_autocall_baseline(
        proxy_inputs, use_fast_cpu=None
    )

    # 8. If surrogate is provided, build Tier 2 tensors, run forward, and cast back to float64
    if eff_surrogate is not None:
        if not need_surr_grad_feats:
            cached_f32 = getattr(base_state, "_cached_f32_surr_residuals", None)
            if cached_f32 is not None and cached_f32[0] == id(eff_surrogate):
                d_haz, d_cpn, d_put = cached_f32[1], cached_f32[2], cached_f32[3]
            else:
                with torch.no_grad():
                    surr_out = eff_surrogate(
                        node_feats=base_state.node_feats[:, :d_assets, :].to(device=surr_dev, dtype=surr_dtype),
                        edge_feats=base_state.edge_feats[:, :d_assets, :d_assets, :].to(
                            device=surr_dev, dtype=surr_dtype
                        ),
                        asset_mask=base_state.asset_mask[:, :d_assets].to(device=surr_dev, dtype=surr_dtype),
                        sched_feats=base_state.sched_feats[:, :m_rem, :].to(device=surr_dev, dtype=surr_dtype),
                        sched_mask=base_state.sched_mask[:, :m_rem].to(device=surr_dev, dtype=surr_dtype),
                        global_feats=base_state.global_feats.to(device=surr_dev, dtype=surr_dtype),
                        detach_gate_weights=True,
                        return_residuals_only=True,
                        all_active=True,
                    )
                    d_haz = surr_out.delta_hazard_logits[0, :m_rem].to(dtype=torch.float64, device=device)
                    d_cpn = surr_out.delta_coupon_logits[0, :m_rem].to(dtype=torch.float64, device=device)
                    d_put = surr_out.delta_put_loss[0].to(dtype=torch.float64, device=device)
                    res_entry = (id(eff_surrogate), d_haz, d_cpn, d_put)
                    setattr(base_state, "_cached_f32_surr_residuals", res_entry)
                    origin_st = getattr(base_state, "_cache_origin", None)
                    if origin_st is not None and origin_st.corr_matrix is base_state.corr_matrix:
                        setattr(origin_st, "_cached_f32_surr_residuals", res_entry)
        else:
            surf_feats_t = torch.stack(surf_feat_rows, dim=0)  # (D, 7)
            log_m0 = torch.log(moneyness_t0.clamp(min=1e-12))
            tau_term = tau_obs_t[-1].clamp(min=1e-6)
            log_fwd_carry_1y = torch.log(forward_rel[:, -1].clamp(min=1e-12)) / tau_term
            log_div_lev = torch.log(cash_div_leverage[:, -1].clamp(min=1e-6))
            f_p_term = forward_perf[:, -1].clamp(min=1e-12)
            v_term = v_perf_t[:, -1].clamp(min=1e-6)
            k_eff = max(float(base_state.strike), 1e-4)
            dist_to_put = torch.log(f_p_term / k_eff) / v_term
            active_ones = torch.ones(d_assets, dtype=torch.float64, device=device)

            node_active = torch.stack(
                [
                    log_m0,
                    log_fwd_carry_1y,
                    log_div_lev,
                    repo_spread_avg,
                    surf_feats_t[:, 0],
                    surf_feats_t[:, 1],
                    surf_feats_t[:, 2],
                    surf_feats_t[:, 3],
                    surf_feats_t[:, 4],
                    surf_feats_t[:, 5],
                    surf_feats_t[:, 6],
                    dist_to_put,
                    baseline.proxy_delta_weights,
                    active_ones,
                ],
                dim=1,
            )
            node_feats = node_active.unsqueeze(0)
            asset_mask = active_ones.unsqueeze(0)

            v_term_vec = v_perf_t[:, -1]
            log_fp_term = torch.log(f_p_term)
            cov_active = corr_sym * v_term_vec.unsqueeze(1) * v_term_vec.unsqueeze(0)
            log_perf_diff_active = torch.abs(log_fp_term.unsqueeze(1) - log_fp_term.unsqueeze(0))
            off_diag_active = 1.0 - torch.eye(d_assets, dtype=torch.float64, device=device)
            edge_active = torch.stack(
                [corr_in, cov_active, log_perf_diff_active, off_diag_active],
                dim=-1,
            )
            edge_feats = edge_active.unsqueeze(0)

            sb = base_state.seasoned_state.sentinel_barriers
            if sb is None:
                raise RuntimeError("Seasoned state sentinel_barriers must be initialized before evaluating PV.")
            log_ko_feat_t = torch.as_tensor(sb.log_ko_barrier_feat, dtype=torch.float64, device=device)
            log_cpn_feat_t = torch.as_tensor(sb.log_cpn_barrier_feat, dtype=torch.float64, device=device)
            log_fw_t = torch.log(baseline.f_w.clamp(min=1e-12))
            sched_active = torch.stack(
                [
                    tau_obs_t,
                    delta_tau_t,
                    df_pay_t,
                    log_ko_feat_t,
                    is_callable_t,
                    log_cpn_feat_t,
                    is_guaranteed_t,
                    coupon_amounts_t,
                    log_fw_t,
                    baseline.v_w,
                    baseline.p_proxy,
                    baseline.q_proxy,
                ],
                dim=1,
            )
            sched_ones = torch.ones(m_rem, dtype=torch.float64, device=device)
            sched_feats = sched_active.unsqueeze(0)
            sched_mask = sched_ones.unsqueeze(0)

            has_ki = 1.0 if (base_state.ki_barrier is not None and base_state.ki_barrier > 0.0) else 0.0
            b_ki_val = (
                float(base_state.ki_barrier)
                if (base_state.ki_barrier is not None and base_state.ki_barrier > 0.0)
                else float(base_state.strike)
            )
            global_feats = torch.stack(
                [
                    torch.tensor(float(base_state.strike), dtype=torch.float64, device=device),
                    torch.tensor(b_ki_val, dtype=torch.float64, device=device),
                    torch.tensor(has_ki, dtype=torch.float64, device=device),
                    torch.tensor(float(base_state.i_ki_past), dtype=torch.float64, device=device),
                    torch.tensor(float(base_state.option_gearing), dtype=torch.float64, device=device),
                    torch.tensor(float(base_state.c_mem_0), dtype=torch.float64, device=device),
                    torch.tensor(1.0 if base_state.is_memory else 0.0, dtype=torch.float64, device=device),
                    torch.tensor(abs(float(base_state.ko_spread)), dtype=torch.float64, device=device),
                    torch.tensor(abs(float(base_state.cpn_spread)), dtype=torch.float64, device=device),
                    baseline.l_put_proxy,
                ],
                dim=0,
            ).unsqueeze(0)

            surr_out = eff_surrogate(
                node_feats=node_feats.to(device=surr_dev, dtype=surr_dtype),
                edge_feats=edge_feats.to(device=surr_dev, dtype=surr_dtype),
                asset_mask=asset_mask.to(device=surr_dev, dtype=surr_dtype),
                sched_feats=sched_feats.to(device=surr_dev, dtype=surr_dtype),
                sched_mask=sched_mask.to(device=surr_dev, dtype=surr_dtype),
                global_feats=global_feats.to(device=surr_dev, dtype=surr_dtype),
                return_residuals_only=True,
                all_active=True,
            )

            d_haz = surr_out.delta_hazard_logits[0, :m_rem].to(dtype=torch.float64, device=device)
            d_cpn = surr_out.delta_coupon_logits[0, :m_rem].to(dtype=torch.float64, device=device)
            d_put = surr_out.delta_put_loss[0].to(dtype=torch.float64, device=device)

        p_proxy_c = baseline.p_proxy.clamp(0.0, 1.0) * is_callable_t
        cum_p_proxy = torch.cumsum(p_proxy_c, dim=0)
        surv_prev_proxy = (1.0 - (cum_p_proxy - p_proxy_c)).clamp(min=1e-6)
        h_proxy_c = (p_proxy_c / surv_prev_proxy).clamp(min=1e-6, max=1.0 - 1e-6)
        logit_h_proxy = torch.log(h_proxy_c / (1.0 - h_proxy_c))

        hazard_rates = is_callable_t * torch.sigmoid(logit_h_proxy + d_haz)
        survival_probs = torch.cumprod(1.0 - hazard_rates, dim=0)
        surv_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64, device=device), survival_probs[:-1]],
            dim=0,
        )
        p_stop = surv_prev * hazard_rates

        q_proxy_c = baseline.q_proxy.clamp(min=1e-6, max=1.0 - 1e-6)
        logit_q_proxy = torch.log(q_proxy_c / (1.0 - q_proxy_c))
        q_gate = (baseline.q_proxy / 1e-5).clamp(0.0, 1.0)
        q_pred = q_gate * torch.sigmoid(logit_q_proxy + d_cpn)
        q_cpn = is_guaranteed_t + (1.0 - is_guaranteed_t) * q_pred

        surv_term = survival_probs[-1]
        k_put_val = max(float(base_state.strike), 0.0)
        l_put_proxy_c = baseline.l_put_proxy.clamp(min=0.0)
        max_put_loss = k_put_val * surv_term
        raw_put_loss = l_put_proxy_c + surv_term * k_put_val * 0.25 * torch.tanh(d_put)
        l_put = torch.minimum(
            torch.maximum(raw_put_loss, torch.zeros_like(raw_put_loss)),
            max_put_loss,
        )
        node_feats_post = node_feats if need_surr_grad_feats else base_state.node_feats
        edge_feats_post = edge_feats if need_surr_grad_feats else base_state.edge_feats
        asset_mask_post = asset_mask if need_surr_grad_feats else base_state.asset_mask
        sched_feats_post = sched_feats if need_surr_grad_feats else base_state.sched_feats
        sched_mask_post = sched_mask if need_surr_grad_feats else base_state.sched_mask
        global_feats_post = global_feats if need_surr_grad_feats else base_state.global_feats
    else:
        node_feats_post = base_state.node_feats
        edge_feats_post = base_state.edge_feats
        asset_mask_post = base_state.asset_mask
        sched_feats_post = base_state.sched_feats
        sched_mask_post = base_state.sched_mask
        global_feats_post = base_state.global_feats
        p_stop = baseline.p_proxy
        q_cpn = baseline.q_proxy
        l_put = baseline.l_put_proxy

    # 9. Assemble W1 cash flows and ProductValue in torch.float64
    state_for_post = AutocallPreprocState(
        node_feats=node_feats_post,
        edge_feats=edge_feats_post,
        asset_mask=asset_mask_post,
        sched_feats=sched_feats_post,
        sched_mask=sched_mask_post,
        global_feats=global_feats_post,
        d_assets=d_assets,
        m_rem=m_rem,
        d_max=d_pad,
        m_max=m_pad,
        original_identifiers=base_state.original_identifiers,
        sorted_identifiers=sorted_ids,
        lex_perm=base_state.lex_perm,
        inv_lex_perm=base_state.inv_lex_perm,
        spots_t0=spots_sorted,
        ref_spots=ref_spots,
        moneyness_t0=moneyness_t0,
        corr_matrix=corr_sym,
        tau_obs=tau_obs_t,
        delta_tau=delta_tau_t,
        tau_pay=tau_pay_t,
        df_obs=df_obs_t,
        df_pay=df_pay_t,
        ko_barriers=ko_barriers_t,
        is_callable=is_callable_t,
        cpn_barriers=cpn_barriers_t,
        is_guaranteed_cpn=is_guaranteed_t,
        coupon_amounts=coupon_amounts_t,
        forward_rel=forward_rel,
        forward_perf=forward_perf,
        cash_div_leverage=cash_div_leverage,
        iv_at_fwd=iv_at_fwd_t,
        v_perf=v_perf_t,
        seasoned_state=base_state.seasoned_state,
        dividend_schedule=div_sched,
        vol_surfaces=base_state.vol_surfaces,
        surface_feature_jacobians=base_state.surface_feature_jacobians,
        iv_at_fwd_jacobians=base_state.iv_at_fwd_jacobians,
        discount_curve=base_state.discount_curve,
        baseline=baseline,
        strike=base_state.strike,
        option_gearing=base_state.option_gearing,
        lower_strike=base_state.lower_strike,
        ki_barrier=base_state.ki_barrier,
        i_ki_past=base_state.i_ki_past,
        c_mem_0=base_state.c_mem_0,
        is_memory=base_state.is_memory,
        ko_spread=base_state.ko_spread,
        cpn_spread=base_state.cpn_spread,
        ki_spread=base_state.ki_spread,
        no_notional_repayment=base_state.no_notional_repayment,
        autocall_coupon=base_state.autocall_coupon,
        force_coupons_on_autocall=base_state.force_coupons_on_autocall,
        coupon_barrier_type=base_state.coupon_barrier_type,
        coupon_barrier_direction=base_state.coupon_barrier_direction,
        allow_autocall_on_expiry=base_state.allow_autocall_on_expiry,
        participation_call_strike=base_state.participation_call_strike,
        participation_call_rate=base_state.participation_call_rate,
        one_star_strike=base_state.one_star_strike,
        one_star_coupon=base_state.one_star_coupon,
        is_long_basket=base_state.is_long_basket,
        payment_currency=base_state.payment_currency,
        callability_index=base_state.callability_index,
        coupon_index=base_state.coupon_index,
        option_index=base_state.option_index,
        funding_leg=base_state.funding_leg,
        pv_in_transit=base_state.pv_in_transit,
        already_autocalled=base_state.already_autocalled,
        trade_info=base_state.trade_info,
        queries=base_state.queries,
    )

    df_fin_fn_override: Callable[[Any], torch.Tensor | float] | None = None
    if curve_pillar_deltas is not None and base_state.funding_leg is not None:
        f_curve_in_leg = base_state.funding_leg.get("funding_curve")
        fin_curve_obj = f_curve_in_leg if f_curve_in_leg is not None else base_state.discount_curve
        disc_id = str(getattr(base_state.discount_curve, "identifier", ""))
        fin_id = str(getattr(fin_curve_obj, "identifier", disc_id))
        if fin_id == disc_id or f_curve_in_leg is None:
            def _df_fin_override(fin_dates: Any) -> torch.Tensor:
                is_scalar = not isinstance(fin_dates, (list, tuple, np.ndarray))
                date_list = [fin_dates] if is_scalar else list(fin_dates)
                base_dfs_np = np.asarray(base_state.discount_curve.df(date_list), dtype=np.float64).reshape(-1)
                base_dfs_t = torch.as_tensor(base_dfs_np, dtype=torch.float64, device=device)
                mult = _curve_df_delta_multiplier(base_state.discount_curve, date_list, curve_pillar_deltas)
                res = base_dfs_t * mult
                return res[0] if is_scalar else res

            df_fin_fn_override = _df_fin_override

    post_out = assemble_autocall_cashflows_and_pv(
        state_for_post,
        p_stop=p_stop,
        q_cpn=q_cpn,
        l_put=l_put,
        l_call=baseline.l_call_proxy,
        l_onestar=baseline.l_onestar_proxy,
        df_fin_fn_override=df_fin_fn_override,
        build_cashflow_items=return_postproc,
    )
    pv_total = post_out.product_value

    # Include curve sensitivity of any non-zero In-Transit payment if curve_pillar_deltas is active
    if curve_pillar_deltas is not None and base_state.seasoned_state.in_transit_indices:
        for rec in base_state.seasoned_state.historical_records:
            if rec.status == "In-Transit" and (rec.coupon_paid != 0.0 or rec.notional_paid != 0.0):
                mult = _curve_df_delta_multiplier(
                    base_state.discount_curve, [rec.payment_date], curve_pillar_deltas
                )[0]
                pv_total = pv_total + float(rec.discount_factor * (rec.coupon_paid + rec.notional_paid)) * (
                    mult - 1.0
                )
        post_out.product_value = pv_total

    if return_postproc:
        return pv_total, post_out
    return pv_total


def _evaluate_batched_spot_perturbed_pvs(
    base_state: AutocallPreprocState,
    sp_batch: torch.Tensor,
    eff_surrogate: MaskedEQAutocallSurrogate,
) -> torch.Tensor:
    """Evaluates B_p spot-perturbed PVs simultaneously in torch.float64 with a single surrogate pass."""
    b_sz = sp_batch.shape[0]
    d_assets = base_state.d_assets
    m_rem = base_state.m_rem
    sorted_ids = base_state.sorted_identifiers
    device = sp_batch.device

    ref_spots = base_state.ref_spots.to(dtype=torch.float64, device=device)
    moneyness_t0 = sp_batch / ref_spots.unsqueeze(0)

    tau_obs_t = base_state.tau_obs.to(dtype=torch.float64, device=device)
    delta_tau_t = base_state.delta_tau.to(dtype=torch.float64, device=device)
    df_pay_t = base_state.df_pay.to(dtype=torch.float64, device=device)

    div_sched = base_state.dividend_schedule
    div_pv_t = div_sched.div_pv.to(dtype=torch.float64, device=device)
    spots_col = sp_batch.unsqueeze(-1)
    ref_col = ref_spots.view(1, d_assets, 1)

    escrowed_spot = torch.maximum(spots_col - div_pv_t.unsqueeze(0), 0.01 * spots_col)
    r_tot_base = div_sched.r_tot_obs.to(dtype=torch.float64, device=device)
    growth_factor = torch.exp(r_tot_base).unsqueeze(0)

    forward_price = escrowed_spot * growth_factor
    forward_rel = forward_price / spots_col
    forward_perf = forward_price / ref_col
    cash_div_leverage = spots_col / escrowed_spot
    repo_spread_avg = div_sched.repo_spread_avg.to(dtype=torch.float64, device=device)

    tau_obs_np = tau_obs_t.detach().cpu().numpy()
    iv_cols: list[torch.Tensor] = []
    for i, ticker in enumerate(sorted_ids):
        surf_i = base_state.vol_surfaces[ticker]
        iv_i = _eval_reg_surface_iv_batch_torch(surf_i, tau_obs_np, forward_price[:, i, :], None)
        iv_cols.append(iv_i)

    iv_at_fwd_t = torch.stack(iv_cols, dim=1)  # (B_p, D, M_rem)
    v_perf_t = iv_at_fwd_t * torch.sqrt(tau_obs_t.clamp(min=1e-12)).view(1, 1, m_rem)

    iv_short = iv_at_fwd_t[:, :, 0]
    iv_term = iv_at_fwd_t[:, :, -1]
    iv_slope = iv_term - iv_short
    base_smile_4 = (
        base_state.node_feats[0, :d_assets, 7:11]
        .detach()
        .to(dtype=torch.float64, device=device)
        .unsqueeze(0)
        .expand(b_sz, d_assets, 4)
    )

    corr_in = base_state.corr_matrix.to(dtype=torch.float64, device=device)
    corr_sym = corr_in
    ko_barriers_t = base_state.ko_barriers.to(dtype=torch.float64, device=device)
    is_callable_t = base_state.is_callable.to(dtype=torch.float64, device=device)
    cpn_barriers_t = base_state.cpn_barriers.to(dtype=torch.float64, device=device)
    is_guaranteed_t = base_state.is_guaranteed_cpn.to(dtype=torch.float64, device=device)
    coupon_amounts_t = base_state.coupon_amounts.to(dtype=torch.float64, device=device)

    proxy_inputs = {
        "forward_perf": forward_perf,
        "v_perf": v_perf_t,
        "corr_matrix": corr_sym,
        "df_pay": df_pay_t,
        "ko_barriers": ko_barriers_t,
        "is_callable": is_callable_t,
        "cpn_barriers": cpn_barriers_t,
        "is_guaranteed_cpn": is_guaranteed_t,
        "coupon_amounts": coupon_amounts_t,
        "strike": base_state.strike,
        "option_gearing": base_state.option_gearing,
        "lower_strike": base_state.lower_strike,
        "ki_barrier": base_state.ki_barrier,
        "i_ki_past": base_state.i_ki_past,
        "c_mem_0": base_state.c_mem_0,
        "is_memory": base_state.is_memory,
        "ko_spread": base_state.ko_spread,
        "cpn_spread": base_state.cpn_spread,
        "ki_spread": base_state.ki_spread,
        "no_notional_repayment": base_state.no_notional_repayment,
        "autocall_coupon": base_state.autocall_coupon,
        "force_coupons_on_autocall": base_state.force_coupons_on_autocall,
        "coupon_barrier_type": base_state.coupon_barrier_type,
        "coupon_barrier_direction": base_state.coupon_barrier_direction,
        "is_long_basket": base_state.is_long_basket,
        "participation_call_strike": base_state.participation_call_strike,
        "participation_call_rate": base_state.participation_call_rate,
        "one_star_strike": base_state.one_star_strike,
        "one_star_coupon": base_state.one_star_coupon,
        "pv_in_transit": base_state.pv_in_transit,
        "already_autocalled": base_state.already_autocalled,
        "callability_index": base_state.callability_index,
    }
    baseline = compute_analytical_autocall_baseline(proxy_inputs, use_fast_cpu=False)

    first_param = next(eff_surrogate.parameters(), None)
    surr_dtype = first_param.dtype if first_param is not None else torch.float32
    surr_dev = first_param.device if first_param is not None else device

    cached_f32 = getattr(base_state, "_cached_f32_surr_residuals", None) if surr_dtype != torch.float64 else None
    if cached_f32 is not None and cached_f32[0] == id(eff_surrogate):
        d_haz = cached_f32[1].unsqueeze(0)
        d_cpn = cached_f32[2].unsqueeze(0)
        d_put = cached_f32[3].unsqueeze(0).expand(b_sz)
    else:
        log_m0 = torch.log(moneyness_t0.clamp(min=1e-12))
        tau_term = tau_obs_t[-1].clamp(min=1e-6)
        log_fwd_carry_1y = torch.log(forward_rel[:, :, -1].clamp(min=1e-12)) / tau_term
        log_div_lev = torch.log(cash_div_leverage[:, :, -1].clamp(min=1e-6))
        f_p_term = forward_perf[:, :, -1].clamp(min=1e-12)
        v_term = v_perf_t[:, :, -1].clamp(min=1e-6)
        k_eff = max(float(base_state.strike), 1e-4)
        dist_to_put = torch.log(f_p_term / k_eff) / v_term
        repo_b = repo_spread_avg.unsqueeze(0).expand(b_sz, d_assets)
        active_ones = torch.ones((b_sz, d_assets), dtype=torch.float64, device=device)

        node_active = torch.stack(
            [
                log_m0,
                log_fwd_carry_1y,
                log_div_lev,
                repo_b,
                iv_short,
                iv_term,
                iv_slope,
                base_smile_4[:, :, 0],
                base_smile_4[:, :, 1],
                base_smile_4[:, :, 2],
                base_smile_4[:, :, 3],
                dist_to_put,
                baseline.proxy_delta_weights,
                active_ones,
            ],
            dim=-1,
        )

        v_term_vec = v_perf_t[:, :, -1]
        log_fp_term = torch.log(f_p_term)
        cov_active = corr_sym.unsqueeze(0) * v_term_vec.unsqueeze(2) * v_term_vec.unsqueeze(1)
        log_perf_diff_active = torch.abs(log_fp_term.unsqueeze(2) - log_fp_term.unsqueeze(1))
        corr_b = corr_in.unsqueeze(0).expand(b_sz, d_assets, d_assets)
        off_diag_b = (1.0 - torch.eye(d_assets, dtype=torch.float64, device=device)).unsqueeze(0).expand(
            b_sz, d_assets, d_assets
        )
        edge_active = torch.stack([corr_b, cov_active, log_perf_diff_active, off_diag_b], dim=-1)

        sb = base_state.seasoned_state.sentinel_barriers
        if sb is None:
            raise RuntimeError("Seasoned state sentinel_barriers must be initialized before evaluating PV.")
        log_ko_feat_t = torch.as_tensor(sb.log_ko_barrier_feat, dtype=torch.float64, device=device)
        log_cpn_feat_t = torch.as_tensor(sb.log_cpn_barrier_feat, dtype=torch.float64, device=device)
        log_fw_t = torch.log(baseline.f_w.clamp(min=1e-12))

        sched_active = torch.stack(
            [
                tau_obs_t.unsqueeze(0).expand(b_sz, m_rem),
                delta_tau_t.unsqueeze(0).expand(b_sz, m_rem),
                df_pay_t.unsqueeze(0).expand(b_sz, m_rem),
                log_ko_feat_t.unsqueeze(0).expand(b_sz, m_rem),
                is_callable_t.unsqueeze(0).expand(b_sz, m_rem),
                log_cpn_feat_t.unsqueeze(0).expand(b_sz, m_rem),
                is_guaranteed_t.unsqueeze(0).expand(b_sz, m_rem),
                coupon_amounts_t.unsqueeze(0).expand(b_sz, m_rem),
                log_fw_t,
                baseline.v_w,
                baseline.p_proxy,
                baseline.q_proxy,
            ],
            dim=-1,
        )
        sched_ones = torch.ones((b_sz, m_rem), dtype=torch.float64, device=device)

        global_static = (
            base_state.global_feats[:, :9]
            .detach()
            .to(dtype=torch.float64, device=device)
            .expand(b_sz, 9)
        )
        global_feats = torch.cat([global_static, baseline.l_put_proxy.unsqueeze(1)], dim=1)

        surr_out = eff_surrogate(
            node_feats=node_active.to(device=surr_dev, dtype=surr_dtype),
            edge_feats=edge_active.to(device=surr_dev, dtype=surr_dtype),
            asset_mask=active_ones.to(device=surr_dev, dtype=surr_dtype),
            sched_feats=sched_active.to(device=surr_dev, dtype=surr_dtype),
            sched_mask=sched_ones.to(device=surr_dev, dtype=surr_dtype),
            global_feats=global_feats.to(device=surr_dev, dtype=surr_dtype),
            return_residuals_only=True,
            all_active=True,
        )

        d_haz = surr_out.delta_hazard_logits[:, :m_rem].to(dtype=torch.float64, device=device)
        d_cpn = surr_out.delta_coupon_logits[:, :m_rem].to(dtype=torch.float64, device=device)
        d_put = surr_out.delta_put_loss.to(dtype=torch.float64, device=device)

    p_proxy_c = baseline.p_proxy.clamp(0.0, 1.0) * is_callable_t.unsqueeze(0)
    cum_p_proxy = torch.cumsum(p_proxy_c, dim=1)
    surv_prev_proxy = (1.0 - (cum_p_proxy - p_proxy_c)).clamp(min=1e-6)
    h_proxy_c = (p_proxy_c / surv_prev_proxy).clamp(min=1e-6, max=1.0 - 1e-6)
    logit_h_proxy = torch.log(h_proxy_c / (1.0 - h_proxy_c))

    hazard_rates = is_callable_t.unsqueeze(0) * torch.sigmoid(logit_h_proxy + d_haz)
    survival_probs = torch.cumprod(1.0 - hazard_rates, dim=1)
    surv_prev = torch.cat(
        [torch.ones((b_sz, 1), dtype=torch.float64, device=device), survival_probs[:, :-1]],
        dim=1,
    )
    p_stop = surv_prev * hazard_rates

    q_proxy_c = baseline.q_proxy.clamp(min=1e-6, max=1.0 - 1e-6)
    logit_q_proxy = torch.log(q_proxy_c / (1.0 - q_proxy_c))
    q_gate = (baseline.q_proxy / 1e-5).clamp(0.0, 1.0)
    q_pred = q_gate * torch.sigmoid(logit_q_proxy + d_cpn)
    q_cpn = is_guaranteed_t.unsqueeze(0) + (1.0 - is_guaranteed_t.unsqueeze(0)) * q_pred

    surv_term = survival_probs[:, -1]
    k_put_val = max(float(base_state.strike), 0.0)
    l_put_proxy_c = baseline.l_put_proxy.clamp(min=0.0)
    max_put_loss = k_put_val * surv_term
    raw_put_loss = l_put_proxy_c + surv_term * k_put_val * 0.25 * torch.tanh(d_put)
    l_put = torch.minimum(
        torch.maximum(raw_put_loss, torch.zeros_like(raw_put_loss)),
        max_put_loss,
    )

    p_t = p_stop * is_callable_t.unsqueeze(0)
    cum_p = torch.cumsum(p_t, dim=1)
    surv_w1 = (1.0 - cum_p).clamp(min=0.0)
    surv_prev_w1 = torch.cat(
        [torch.ones((b_sz, 1), dtype=torch.float64, device=device), surv_w1[:, :-1]],
        dim=1,
    )

    if base_state.no_notional_repayment:
        notional_pv = torch.zeros(b_sz, dtype=torch.float64, device=device)
    else:
        notional_cf = df_pay_t.unsqueeze(0) * p_t
        notional_cf_term = df_pay_t[-1] * (p_t[:, -1] + surv_w1[:, -1])
        if m_rem > 1:
            notional_pv = notional_cf[:, :-1].sum(dim=1) + notional_cf_term
        else:
            notional_pv = notional_cf_term

    cpn_cf_list: list[torch.Tensor] = []
    mem_acc = torch.full((b_sz,), float(base_state.c_mem_0), dtype=torch.float64, device=device)
    for m in range(m_rem):
        c_m = coupon_amounts_t[m]
        eff_c_m = (c_m + mem_acc) if base_state.is_memory else c_m
        if base_state.force_coupons_on_autocall:
            eff_q_m = torch.maximum(q_cpn[:, m], p_t[:, m] / surv_prev_w1[:, m].clamp(min=1e-12))
        else:
            eff_q_m = q_cpn[:, m]
        cf_cpn_m = df_pay_t[m] * (
            surv_prev_w1[:, m] * eff_q_m * eff_c_m + p_t[:, m] * float(base_state.autocall_coupon)
        )
        cpn_cf_list.append(cf_cpn_m)
        if base_state.is_memory:
            mem_acc = eff_c_m * (1.0 - eff_q_m)

    coupon_pv = torch.stack(cpn_cf_list, dim=1).sum(dim=1)

    if base_state.funding_leg is not None:
        cached_w_pair = base_state.funding_leg.get("_cached_w_fund_per_period")
        if cached_w_pair is not None:
            w_fund_t = torch.as_tensor(cached_w_pair[1], dtype=torch.float64, device=device)
            funding_pv = (surv_prev_w1 * w_fund_t.unsqueeze(0)).sum(dim=1)
        else:
            funding_pv = torch.zeros(b_sz, dtype=torch.float64, device=device)
    else:
        funding_pv = torch.zeros(b_sz, dtype=torch.float64, device=device)

    put_sign = 1.0 if base_state.is_long_basket else -1.0
    option_pv = df_pay_t[-1] * (
        put_sign * float(base_state.option_gearing) * l_put
        + float(base_state.participation_call_rate) * baseline.l_call_proxy
        + baseline.l_onestar_proxy
    )

    return float(base_state.pv_in_transit) + notional_pv + coupon_pv + option_pv - funding_pv


def compute_euler_cash_gamma(
    pv_of_eps_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Evaluates scalar directional Euler CashGamma d^2/d_eps^2 PV(S_0 * (1 + eps))|_{eps=0} in O(1) calls (W2 & Gate G1.9).

    By Euler's homogeneous directional derivative identity:
        d^2 / d_eps^2 PV(S_0 * (1 + eps))|_{eps=0} == sum_{i=1}^D sum_{j=1}^D S_{0,i} S_{0,j} d^2 PV / (dS_i dS_j)
    """
    eps = torch.tensor(0.0, dtype=torch.float64, device=device, requires_grad=True)
    pv = pv_of_eps_fn(eps)
    grad_eps = torch.autograd.grad(pv, eps, create_graph=True)[0]
    d2_pv = torch.autograd.grad(grad_eps, eps, retain_graph=False)[0]
    return d2_pv.to(dtype=torch.float64)


def compute_eq_autocall_greeks(
    state_or_request: AutocallPreprocState | Any,
    *,
    queries: Sequence[str] | None = None,
    surrogate: MaskedEQAutocallSurrogate | None = None,
    theta_day_fraction: float = 1.0 / 365.0,
) -> AutocallGreeksResult:
    """Computes `ProductValue`, `CashFlows`, and any subset of the 8 Greek metrics in `torch.float64`.

    Supported Greek queries (§1.4):
      - `Delta`: dPV / dS_i(t_0) mapped to `original_identifiers` order via pi_lex^{-1}.
      - `Gamma`: d^2 PV / (dS_i(t_0) dS_j(t_0)) mapped to `original_identifiers` order via pi_lex^{-1}.
      - `MultivariateAggregatedCashGamma`: sum_{i,j} S_i(t_0) S_j(t_0) Gamma_{ij} (computed via O(1)
        `compute_euler_cash_gamma` when full `Gamma` is not requested, or directly from `Gamma` when both are requested).
      - `Vega`: dPV / dSigma_{quote,i} of shape (N_exp, N_strike) per asset in `original_identifiers` order.
      - `Rho`: dPV / dr_k^{pillar} of shape (K_pillars,) per discount curve via analytical W_spline.
      - `DividendsSensitivities`: dPV / dd_{i,k} per discrete dividend event of each asset in `original_identifiers` order.
      - `Theta`: 1-calendar-day roll PV(t_0 + 1d) - PV(t_0).
      - `Cega`: Symmetrized off-diagonal correlation sensitivity dPV/d_rho_{ij} + dPV/d_rho_{ji} in `original_identifiers` order.
    """
    if isinstance(state_or_request, AutocallPreprocState):
        state = state_or_request
    else:
        state = preprocess_eq_autocall(state_or_request)

    raw_queries = queries if queries is not None else state.queries
    q_set = {str(getattr(q, "value", q)) for q in raw_queries}

    want_delta = "Delta" in q_set
    want_gamma = "Gamma" in q_set
    want_cash_gamma = "MultivariateAggregatedCashGamma" in q_set
    want_vega = "Vega" in q_set
    want_rho = "Rho" in q_set
    want_divs = "DividendsSensitivities" in q_set
    want_theta = "Theta" in q_set
    want_cega = "Cega" in q_set or "ConvexCega" in q_set

    d_assets = state.d_assets
    m_rem = state.m_rem
    orig_ids = state.original_identifiers
    sorted_ids = state.sorted_identifiers
    inv_perm = state.inv_lex_perm
    device = state.spots_t0.device

    curve = state.discount_curve
    curve_id = str(getattr(curve, "identifier", getattr(curve, "Identifier", "DEFAULT")))
    if hasattr(curve, "tenors"):
        k_pillars = len(curve.tenors)
    elif isinstance(curve, Mapping) and "RatesSchedule" in curve:
        k_pillars = len(curve["RatesSchedule"].get("Rates", []))
    else:
        k_pillars = 0

    eff_surrogate = (
        None
        if (
            surrogate is not None
            and hasattr(surrogate, "is_zero_residual")
            and surrogate.is_zero_residual()
        )
        else surrogate
    )

    need_spot_grad = want_delta or want_gamma
    surr_first_param = next(eff_surrogate.parameters(), None) if eff_surrogate is not None else None
    use_second_order_ad_gamma = want_gamma and (
        eff_surrogate is None
        or (surr_first_param is not None and surr_first_param.dtype == torch.float64)
    )

    origin_st = getattr(state, "_cache_origin", state)
    g_cache_key: Any | None = None
    if origin_st.spots_t0 is state.spots_t0 and origin_st.corr_matrix is state.corr_matrix:
        param_sig = (
            float(surr_first_param.detach().view(-1)[0].item())
            if surr_first_param is not None and surr_first_param.numel() > 0
            else None
        )
        g_cache_key = (
            tuple(orig_ids),
            frozenset(q_set),
            id(eff_surrogate) if eff_surrogate is not None else None,
            param_sig,
            float(theta_day_fraction),
        )
        g_cache: dict[Any, AutocallGreeksResult] | None = getattr(origin_st, "_greeks_result_cache", None)
        if g_cache is not None and g_cache_key in g_cache:
            return g_cache[g_cache_key]
    spots_var = state.spots_t0.detach().clone().to(dtype=torch.float64, device=device)
    spots_var.requires_grad_(need_spot_grad)

    corr_var = state.corr_matrix.detach().clone().to(dtype=torch.float64, device=device)
    corr_var.requires_grad_(want_cega and d_assets > 1)

    pillar_deltas: torch.Tensor | None = None
    if want_rho and k_pillars > 0:
        pillar_deltas = torch.zeros(k_pillars, dtype=torch.float64, device=device, requires_grad=True)

    div_vars: dict[str, torch.Tensor] = {}
    div_grad_tickers: list[str] = []
    if want_divs:
        for ticker in sorted_ids:
            base_amt = state.dividend_schedule.dividend_amount_tensors.get(ticker)
            if base_amt is not None and base_amt.numel() > 0:
                t_div = base_amt.detach().clone().to(dtype=torch.float64, device=device).requires_grad_(True)
                div_vars[ticker] = t_div
                div_grad_tickers.append(ticker)

    vol_deltas: dict[str, torch.Tensor] = {}
    vol_grad_tickers: list[str] = []
    if want_vega and m_rem > 0:
        for ticker in sorted_ids:
            surf = state.vol_surfaces[ticker]
            v_del = torch.zeros(
                (surf.n_exp, surf.n_raw_strikes),
                dtype=torch.float64,
                device=device,
                requires_grad=True,
            )
            vol_deltas[ticker] = v_del
            vol_grad_tickers.append(ticker)

    pv_res = evaluate_differentiable_pv(
        state,
        spots_t0=spots_var,
        corr_matrix=corr_var,
        div_amounts=div_vars if want_divs else None,
        vol_quote_deltas=vol_deltas if want_vega else None,
        curve_pillar_deltas=pillar_deltas,
        surrogate=eff_surrogate,
        return_postproc=True,
    )
    if not isinstance(pv_res, tuple):
        raise RuntimeError("Expected tuple (pv_t, post_out) when return_postproc=True.")
    pv_t, post_out = pv_res

    # Collect active leaf inputs; run non-spot 1st-order gradients with create_graph=False first
    other_inputs: list[torch.Tensor] = []
    other_tags: list[tuple[str, str]] = []

    if want_cega and d_assets > 1 and pv_t.requires_grad:
        other_inputs.append(corr_var)
        other_tags.append(("corr", ""))
    if want_rho and pillar_deltas is not None and pv_t.requires_grad:
        other_inputs.append(pillar_deltas)
        other_tags.append(("rho", curve_id))
    if want_divs and pv_t.requires_grad:
        for ticker in div_grad_tickers:
            other_inputs.append(div_vars[ticker])
            other_tags.append(("div", ticker))
    if want_vega and pv_t.requires_grad:
        for ticker in vol_grad_tickers:
            other_inputs.append(vol_deltas[ticker])
            other_tags.append(("vega", ticker))

    grad_map: dict[tuple[str, str], torch.Tensor] = {}
    if not use_second_order_ad_gamma:
        grad_inputs = ([spots_var] if (need_spot_grad and pv_t.requires_grad) else []) + other_inputs
        input_tags = ([("spots", "")] if (need_spot_grad and pv_t.requires_grad) else []) + other_tags
        if len(grad_inputs) > 0:
            grads = torch.autograd.grad(
                pv_t,
                grad_inputs,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            for tag, g_val in zip(input_tags, grads):
                if g_val is not None:
                    grad_map[tag] = g_val
    else:
        if len(other_inputs) > 0:
            other_grads = torch.autograd.grad(
                pv_t,
                other_inputs,
                create_graph=False,
                retain_graph=True,
                allow_unused=True,
            )
            for tag, g_val in zip(other_tags, other_grads):
                if g_val is not None:
                    grad_map[tag] = g_val
        if pv_t.requires_grad:
            g_spot = torch.autograd.grad(
                pv_t,
                spots_var,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g_spot is not None:
                grad_map[("spots", "")] = g_spot

    inv_idx_t = torch.as_tensor(inv_perm, dtype=torch.long, device=device)
    spots_orig_t = state.spots_t0.detach().index_select(0, inv_idx_t)

    # 1. Delta (in original asset order pi_lex^{-1})
    delta_dict: dict[str, float] | None = None
    delta_tensor_orig: torch.Tensor | None = None
    grad_spots_sorted = grad_map.get(("spots", ""))

    if want_delta:
        if grad_spots_sorted is not None:
            delta_tensor_orig = grad_spots_sorted.detach().index_select(0, inv_idx_t).to(dtype=torch.float64)
        else:
            delta_tensor_orig = torch.zeros(d_assets, dtype=torch.float64, device=device)
        delta_dict = {
            orig_ids[i]: float(delta_tensor_orig[i].cpu().item()) for i in range(d_assets)
        }

    # 2. Gamma & 3. MultivariateAggregatedCashGamma
    gamma_dict: dict[str, dict[str, float]] | None = None
    gamma_tensor_orig: torch.Tensor | None = None
    cash_gamma_agg: float | None = None

    if want_gamma:
        if use_second_order_ad_gamma and grad_spots_sorted is not None and grad_spots_sorted.requires_grad:
            hess_rows: list[torch.Tensor] = []
            for i in range(d_assets):
                row_i = torch.autograd.grad(
                    grad_spots_sorted[i],
                    spots_var,
                    retain_graph=(i < d_assets - 1),
                    allow_unused=True,
                )[0]
                if row_i is None:
                    hess_rows.append(torch.zeros(d_assets, dtype=torch.float64, device=device))
                else:
                    hess_rows.append(row_i.detach().to(dtype=torch.float64))
            hess_sorted = torch.stack(hess_rows, dim=0)
            hess_sym = 0.5 * (hess_sorted + hess_sorted.transpose(0, 1))
            gamma_tensor_orig = hess_sym.index_select(0, inv_idx_t).index_select(1, inv_idx_t)
        elif not use_second_order_ad_gamma and eff_surrogate is not None and m_rem > 0 and not state.already_autocalled:
            with torch.no_grad():
                base_spots_g = state.spots_t0.detach().to(dtype=torch.float64, device=device)
                h_rel = 1.0e-4
                h_vec = torch.clamp(base_spots_g * h_rel, min=1e-6)
                sp_list: list[torch.Tensor] = []
                for i in range(d_assets):
                    h_i = h_vec[i]
                    sp_p = base_spots_g.clone()
                    sp_p[i] = sp_p[i] + h_i
                    sp_m = base_spots_g.clone()
                    sp_m[i] = sp_m[i] - h_i
                    sp_list.append(sp_p)
                    sp_list.append(sp_m)
                for i in range(d_assets):
                    for j in range(i + 1, d_assets):
                        sp_pp = base_spots_g.clone()
                        sp_pp[i] = sp_pp[i] + h_vec[i]
                        sp_pp[j] = sp_pp[j] + h_vec[j]
                        sp_mm = base_spots_g.clone()
                        sp_mm[i] = sp_mm[i] - h_vec[i]
                        sp_mm[j] = sp_mm[j] - h_vec[j]
                        sp_list.append(sp_pp)
                        sp_list.append(sp_mm)
                sp_batch = torch.stack(sp_list, dim=0)
                pv_batch = _evaluate_batched_spot_perturbed_pvs(state, sp_batch, eff_surrogate)
                pv0 = pv_t.detach()
                hess_sorted = torch.zeros((d_assets, d_assets), dtype=torch.float64, device=device)
                for i in range(d_assets):
                    hess_sorted[i, i] = (pv_batch[2 * i] - 2.0 * pv0 + pv_batch[2 * i + 1]) / (
                        h_vec[i] * h_vec[i]
                    )
                pair_offset = 2 * d_assets
                for i in range(d_assets):
                    for j in range(i + 1, d_assets):
                        d2_ij = pv_batch[pair_offset] - 2.0 * pv0 + pv_batch[pair_offset + 1]
                        pair_offset += 2
                        g_ij = (
                            d2_ij
                            - (h_vec[i] * h_vec[i]) * hess_sorted[i, i]
                            - (h_vec[j] * h_vec[j]) * hess_sorted[j, j]
                        ) / (2.0 * h_vec[i] * h_vec[j])
                        hess_sorted[i, j] = g_ij
                        hess_sorted[j, i] = g_ij
                gamma_tensor_orig = hess_sorted.index_select(0, inv_idx_t).index_select(1, inv_idx_t)
        else:
            gamma_tensor_orig = torch.zeros((d_assets, d_assets), dtype=torch.float64, device=device)

        gamma_dict = {
            orig_ids[i]: {
                orig_ids[j]: float(gamma_tensor_orig[i, j].cpu().item()) for j in range(d_assets)
            }
            for i in range(d_assets)
        }
        if want_cash_gamma:
            s_outer = spots_orig_t.unsqueeze(1) * spots_orig_t.unsqueeze(0)
            cash_gamma_agg = float((s_outer * gamma_tensor_orig).sum().cpu().item())

    elif want_cash_gamma:
        if m_rem == 0 or state.already_autocalled:
            cash_gamma_agg = 0.0
        else:
            base_spots = state.spots_t0.detach().to(dtype=torch.float64, device=device)

            def _pv_of_eps(eps_scalar: torch.Tensor) -> torch.Tensor:
                res = evaluate_differentiable_pv(
                    state,
                    spots_t0=base_spots * (1.0 + eps_scalar),
                    surrogate=eff_surrogate,
                    return_postproc=False,
                )
                if not isinstance(res, torch.Tensor):
                    raise RuntimeError("Expected torch.Tensor when return_postproc=False.")
                return res

            cash_gamma_agg = float(
                compute_euler_cash_gamma(_pv_of_eps, device=device).detach().cpu().item()
            )

    # 4. Vega (in original asset order pi_lex^{-1})
    vega_dict: dict[str, list[list[float]]] | None = None
    vega_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    if want_vega:
        vega_dict = {}
        vega_arrays = {}
        for ticker in orig_ids:
            surf = state.vol_surfaces[ticker]
            g_v = grad_map.get(("vega", ticker))
            if g_v is not None:
                arr_v = g_v.detach().cpu().numpy().astype(np.float64)
            else:
                arr_v = np.zeros((surf.n_exp, surf.n_raw_strikes), dtype=np.float64)
            vega_arrays[ticker] = arr_v
            vega_dict[ticker] = [[float(x) for x in row] for row in arr_v]

    # 5. Rho
    rho_dict: dict[str, list[float]] | None = None
    rho_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    if want_rho:
        g_rho = grad_map.get(("rho", curve_id))
        if g_rho is not None:
            arr_rho = g_rho.detach().cpu().numpy().astype(np.float64)
        else:
            arr_rho = np.zeros(k_pillars, dtype=np.float64)
        rho_arrays = {curve_id: arr_rho}
        rho_dict = {curve_id: [float(x) for x in arr_rho]}

    # 6. DividendsSensitivities (in original asset order pi_lex^{-1})
    divs_dict: dict[str, list[float]] | None = None
    divs_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    if want_divs:
        divs_dict = {}
        divs_arrays = {}
        for ticker in orig_ids:
            g_d = grad_map.get(("div", ticker))
            if g_d is not None:
                arr_d = g_d.detach().cpu().numpy().astype(np.float64)
            else:
                n_ev = len(state.dividend_schedule.dividend_events.get(ticker, []))
                arr_d = np.zeros(n_ev, dtype=np.float64)
            divs_arrays[ticker] = arr_d
            divs_dict[ticker] = [float(x) for x in arr_d]

    # 7. Cega (symmetrized off-diagonal dPV/d_rho_{ij} + dPV/d_rho_{ji} in original asset order)
    cega_dict: dict[str, dict[str, float]] | None = None
    cega_tensor_orig: torch.Tensor | None = None
    if want_cega:
        g_corr = grad_map.get(("corr", ""))
        if g_corr is not None and d_assets > 1:
            eye_d = torch.eye(d_assets, dtype=torch.float64, device=device)
            cega_sorted = (g_corr.detach() + g_corr.detach().transpose(0, 1)) * (1.0 - eye_d)
            cega_tensor_orig = cega_sorted.index_select(0, inv_idx_t).index_select(1, inv_idx_t)
        else:
            cega_tensor_orig = torch.zeros((d_assets, d_assets), dtype=torch.float64, device=device)
        cega_dict = {
            orig_ids[i]: {
                orig_ids[j]: float(cega_tensor_orig[i, j].cpu().item()) for j in range(d_assets)
            }
            for i in range(d_assets)
        }

    # 8. Theta (1-calendar-day roll PV(t_0 + 1d) - PV(t_0))
    theta_val: float | None = None
    if want_theta:
        with torch.no_grad():
            pv_rolled = evaluate_differentiable_pv(
                state,
                surrogate=eff_surrogate,
                tau_shift=float(theta_day_fraction),
                return_postproc=False,
            )
            if not isinstance(pv_rolled, torch.Tensor):
                raise RuntimeError("Expected torch.Tensor from rolled Theta evaluation.")
            theta_val = float((pv_rolled - pv_t.detach()).cpu().item())

    res_out = AutocallGreeksResult(
        product_value=float(pv_t.detach().cpu().item()),
        cash_flows=post_out.cashflows,
        delta=delta_dict,
        gamma=gamma_dict,
        multivariate_aggregated_cash_gamma=cash_gamma_agg,
        vega=vega_dict,
        rho=rho_dict,
        dividends_sensitivities=divs_dict,
        theta=theta_val,
        cega=cega_dict,
        delta_tensor=delta_tensor_orig,
        gamma_tensor=gamma_tensor_orig,
        vega_arrays=vega_arrays,
        rho_arrays=rho_arrays,
        dividends_arrays=divs_arrays,
        cega_tensor=cega_tensor_orig,
        reporting_currency=post_out.reporting_currency,
        standard_error=post_out.standard_error,
        cash_to_settle_today=post_out.cash_to_settle_today,
        postproc_output=post_out,
    )
    if g_cache_key is not None:
        g_cache_dict: dict[Any, AutocallGreeksResult] | None = getattr(origin_st, "_greeks_result_cache", None)
        if g_cache_dict is None:
            g_cache_dict = {}
            setattr(origin_st, "_greeks_result_cache", g_cache_dict)
        g_cache_dict[g_cache_key] = res_out
    return res_out


def compute_batched_surrogate_greeks(
    surrogate: MaskedEQAutocallSurrogate,
    node_feats: torch.Tensor,
    edge_feats: torch.Tensor,
    asset_mask: torch.Tensor,
    sched_feats: torch.Tensor,
    sched_mask: torch.Tensor,
    global_feats: torch.Tensor,
    *,
    spots_t0: torch.Tensor | None = None,
    ref_spots: torch.Tensor | None = None,
    pv_in_transit: torch.Tensor | float | None = None,
    no_notional_repayment: torch.Tensor | bool = False,
    autocall_coupon: torch.Tensor | float = 0.0,
    force_coupons_on_autocall: torch.Tensor | bool = False,
    compute_cash_gamma: bool = False,
) -> BatchedSurrogateGreeksOutput:
    """Vectorized GPU/CPU batched valuation and first-order Greeks in a single autograd pass (Gate G2.4).

    Outer inputs, Tier-1 Stulz-Margrabe proxy chain rule, and returned Greek tensors are strictly
    `torch.float64` (REM-S0-01), while the inner `MaskedEQAutocallSurrogate` residual pass runs in
    `torch.float32` (or parameter dtype). Achieves <= 0.05 ms/trade for B = 1024 on CUDA.
    """
    first_param = next(surrogate.parameters(), None)
    surr_dtype = first_param.dtype if first_param is not None else torch.float32
    device = node_feats.device

    batch_size, d_max, d_node = node_feats.shape
    _, m_max, d_sched = sched_feats.shape

    amask_bool = asset_mask > 0.5
    smask_bool = sched_mask > 0.5
    amask_f64 = amask_bool.to(dtype=torch.float64, device=device)
    smask_f64 = smask_bool.to(dtype=torch.float64, device=device)

    # Compact trailing all-zero padded asset and schedule dimensions (preserving exact Gate G1.7 invariance)
    if d_max > 1 and not bool(amask_bool[:, -1].all()):
        any_asset = amask_bool.any(dim=0)
        d_act = int(any_asset.nonzero()[-1].item()) + 1 if any_asset.any() else 1
    else:
        d_act = d_max
    if m_max > 1 and not bool(smask_bool[:, -1].all()):
        any_sched = smask_bool.any(dim=0)
        m_act = int(any_sched.nonzero()[-1].item()) + 1 if any_sched.any() else 1
    else:
        m_act = m_max

    node_in = node_feats[:, :d_act, :].detach().to(device=device, dtype=surr_dtype).requires_grad_(True)
    edge_in = (
        edge_feats[:, :d_act, :d_act, :].detach().to(device=device, dtype=surr_dtype).requires_grad_(True)
    )
    sched_in = sched_feats[:, :m_act, :].detach().to(device=device, dtype=surr_dtype).requires_grad_(True)
    glob_in = global_feats.detach().to(device=device, dtype=surr_dtype).requires_grad_(True)
    amask_act = amask_bool[:, :d_act].to(device=device, dtype=surr_dtype)
    smask_act = smask_bool[:, :m_act].to(device=device, dtype=surr_dtype)

    saved_req_grad: list[tuple[torch.nn.Parameter, bool]] = []
    if first_param is not None and first_param.requires_grad:
        for p in surrogate.parameters():
            if p.requires_grad:
                saved_req_grad.append((p, True))
                p.requires_grad_(False)
    try:
        out = surrogate(
            node_in,
            edge_in,
            amask_act,
            sched_in,
            smask_act,
            glob_in,
            detach_gate_weights=True,
        )
        pv = assemble_surrogate_product_value(
            out,
            sched_in,
            smask_act,
            glob_in,
            no_notional_repayment=no_notional_repayment,
            autocall_coupon=autocall_coupon,
            force_coupons_on_autocall=force_coupons_on_autocall,
            pv_in_transit=0.0,
        )
        g_node_act, g_edge_act, g_sched_act, g_glob_act = torch.autograd.grad(
            pv.sum(),
            (node_in, edge_in, sched_in, glob_in),
            create_graph=False,
            retain_graph=False,
        )
    finally:
        for p, req in saved_req_grad:
            p.requires_grad_(req)

    with torch.no_grad():
        amask_64 = amask_f64[:, :d_act]
        smask_64 = smask_f64[:, :m_act]
        all_sched_act = bool(smask_bool[:, :m_act].all())
        node_64 = node_feats[:, :d_act, :].detach().to(dtype=torch.float64, device=device)
        sched_64 = sched_feats[:, :m_act, :].detach().to(dtype=torch.float64, device=device)
        glob_64 = global_feats.detach().to(dtype=torch.float64, device=device)

        g_node_64 = g_node_act.detach().to(dtype=torch.float64)
        g_sched_64 = g_sched_act.detach().to(dtype=torch.float64)
        g_glob_64 = g_glob_act.detach().to(dtype=torch.float64)

        log_div_lev = node_64[:, :, int(AutocallNodeCol.CASH_DIV_LEVERAGE)].clamp(-10.0, 10.0)
        cash_div_lev = torch.exp(log_div_lev) * amask_64
        iv_term_node = node_64[:, :, int(AutocallNodeCol.IV_TERM_ATM)].clamp(min=1e-4)
        tau_all = sched_64[:, :, int(AutocallSchedCol.TAU_OBS)]
        if all_sched_act:
            idx_last: torch.Tensor | None = None
            tau_term = tau_all[:, -1:].clamp(min=1e-6)
        else:
            idx_last = (smask_64.sum(dim=1).long() - 1).clamp(min=0)
            tau_term = tau_all.gather(1, idx_last.unsqueeze(1)).clamp(min=1e-6)
        v_i_term = (iv_term_node * torch.sqrt(tau_term)).clamp(min=1e-6)
        g_dist_put = g_node_64[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)] * (
            cash_div_lev / v_i_term
        )

        w_proxy = node_64[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)].clamp(min=0.0) * amask_64
        w_lev = w_proxy * cash_div_lev

        log_fw_base = sched_64[:, :, int(AutocallSchedCol.STULZ_FWD_W)].clamp(-20.0, 20.0)
        v_w_base = sched_64[:, :, int(AutocallSchedCol.STULZ_VOL_W)].clamp(min=1e-6)
        b_ko = torch.exp(sched_64[:, :, int(AutocallSchedCol.LOG_KO_BARRIER)].clamp(-20.0, 20.0))
        b_cpn = torch.exp(sched_64[:, :, int(AutocallSchedCol.LOG_CPN_BARRIER)].clamp(-20.0, 20.0))
        is_call_base = sched_64[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)] * smask_64
        is_guar_base = sched_64[:, :, int(AutocallSchedCol.IS_GUARANTEED_CPN)]
        ko_spread_mag = glob_64[:, 7:8]
        cpn_spread_mag = glob_64[:, 8:9]
        k_strike = glob_64[:, 0]
        b_ki = glob_64[:, 1]
        has_ki = glob_64[:, 2]
        i_ki_past = glob_64[:, 3]
        active_ki = (has_ki > 0.5) & (i_ki_past < 0.5) & (b_ki > 0.0)
        k_eff = torch.where(active_ki, torch.minimum(k_strike, b_ki), k_strike)

        h_fd = 1e-4
        n_pts = 3 if compute_cash_gamma else 2
        if compute_cash_gamma:
            shifts = torch.tensor([-h_fd, 0.0, h_fd], dtype=torch.float64, device=device).view(3, 1, 1)
        else:
            shifts = torch.tensor([-h_fd, h_fd], dtype=torch.float64, device=device).view(2, 1, 1)

        # Broadcast 3D shape (n_pts, batch_size, m_act) directly without .expand().reshape() copies
        f_w_pts = torch.exp(log_fw_base.unsqueeze(0) + shifts)
        h_raw_pts = is_call_base * _digital_call_spread_prob_torch(
            f_w_pts, v_w_base, b_ko, -ko_spread_mag
        )
        h_mod_pts = h_raw_pts.clamp(0.0, 1.0 - 1e-6)
        if m_act == 1:
            surv_mod_pts = 1.0 - h_mod_pts
            p_mod_pts = h_mod_pts * smask_64
        else:
            surv_mod_pts = torch.cumprod(1.0 - h_mod_pts, dim=-1)
            surv_prev_pts = torch.cat(
                [torch.ones((n_pts, batch_size, 1), dtype=torch.float64, device=device), surv_mod_pts[:, :, :-1]],
                dim=-1,
            )
            p_mod_pts = surv_prev_pts * h_mod_pts * smask_64

        q_raw_pts = _digital_call_spread_prob_torch(f_w_pts, v_w_base, b_cpn, -cpn_spread_mag)
        q_mod_pts = (is_guar_base + (1.0 - is_guar_base) * q_raw_pts) * smask_64

        if all_sched_act:
            f_w_term_pts = f_w_pts[:, :, -1]
            v_w_term_pts = v_w_base[:, -1]
            surv_term_pts = surv_mod_pts[:, :, -1]
        else:
            assert idx_last is not None
            idx_last_3d = idx_last.view(1, batch_size, 1).expand(n_pts, batch_size, 1)
            f_w_term_pts = f_w_pts.gather(2, idx_last_3d).squeeze(2)
            v_w_term_pts = v_w_base.gather(1, idx_last.unsqueeze(1)).squeeze(1)
            surv_term_pts = surv_mod_pts.gather(2, idx_last_3d).squeeze(2)

        put_part_pts = _black76_put_torch(f_w_term_pts, k_eff, v_w_term_pts)
        ki_up_pts = _digital_call_spread_prob_torch(
            f_w_term_pts, v_w_term_pts, b_ki, -ko_spread_mag.squeeze(1)
        )
        gap_part_pts = torch.where(
            active_ki,
            torch.clamp(k_strike - b_ki, min=0.0) * (1.0 - ki_up_pts).clamp(0.0, 1.0),
            torch.zeros_like(put_part_pts),
        )
        l_put_pts = torch.where(
            k_strike > 1e-6,
            surv_term_pts * (put_part_pts + gap_part_pts),
            torch.zeros_like(put_part_pts),
        )

        idx_hi = n_pts - 1
        dp_dfw = (p_mod_pts[idx_hi] - p_mod_pts[0]) / (2.0 * h_fd)
        dq_dfw = (q_mod_pts[idx_hi] - q_mod_pts[0]) / (2.0 * h_fd)
        dlput_dfw = (l_put_pts[idx_hi] - l_put_pts[0]) / (2.0 * h_fd)

        d_pv_d_fw = (
            (
                g_sched_64[:, :, int(AutocallSchedCol.STULZ_FWD_W)] * smask_64
                + g_sched_64[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] * dp_dfw
                + g_sched_64[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)] * dq_dfw
            ).sum(dim=1)
            + g_glob_64[:, 9] * dlput_dfw
        )
        g_node_64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)] = (
            g_node_64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)]
            + g_dist_put
            + d_pv_d_fw.unsqueeze(1) * w_lev
        )

        cash_gamma_t: torch.Tensor | None = None
        if compute_cash_gamma:
            g_log_m0_sum = (g_node_64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)] * amask_64).sum(dim=1)
            h_proxy_cg = h_raw_pts.clamp(1e-6, 1.0 - 1e-6)
            logit_h_cg = torch.log(h_proxy_cg / (1.0 - h_proxy_cg))
            dh_det = out.delta_hazard_logits[:, :m_act].detach().to(dtype=torch.float64)
            h_tot_cg = is_call_base * smask_64 * torch.sigmoid(logit_h_cg + dh_det)
            if m_act == 1:
                surv_cg = 1.0 - h_tot_cg
                p_stop_cg = h_tot_cg * smask_64
            else:
                surv_cg = torch.cumprod(1.0 - h_tot_cg, dim=-1)
                surv_prev_cg = torch.cat(
                    [torch.ones((3, batch_size, 1), dtype=torch.float64, device=device), surv_cg[:, :, :-1]],
                    dim=-1,
                )
                p_stop_cg = (surv_prev_cg * h_tot_cg) * smask_64

            q_raw_cg = q_raw_pts.clamp(1e-6, 1.0 - 1e-6)
            logit_q_cg = torch.log(q_raw_cg / (1.0 - q_raw_cg))
            dq_det = out.delta_coupon_logits[:, :m_act].detach().to(dtype=torch.float64)
            q_cpn_cg = (is_guar_base + (1.0 - is_guar_base) * torch.sigmoid(logit_q_cg + dq_det)) * smask_64

            if all_sched_act:
                surv_term_cg = surv_cg[:, :, -1]
            else:
                assert idx_last is not None
                surv_term_cg = surv_cg.gather(
                    2, idx_last.view(1, batch_size, 1).expand(3, batch_size, 1)
                ).squeeze(2)

            dput_det = out.delta_put_loss.detach().to(dtype=torch.float64)
            l_put_cg = (
                torch.where(
                    k_strike > 1e-6,
                    surv_term_cg * (put_part_pts + gap_part_pts),
                    torch.zeros_like(put_part_pts),
                )
                + dput_det
            ).clamp(min=0.0)

            surr_cg = AutocallSurrogateOutput(
                stopping_probs=p_stop_cg.reshape(3 * batch_size, m_act),
                survival_probs=surv_cg.reshape(3 * batch_size, m_act),
                hazard_rates=h_tot_cg.reshape(3 * batch_size, m_act),
                coupon_probs=q_cpn_cg.reshape(3 * batch_size, m_act),
                put_loss=l_put_cg.reshape(3 * batch_size),
                delta_hazard_logits=dh_det.unsqueeze(0).expand(3, batch_size, m_act).reshape(3 * batch_size, m_act),
                delta_coupon_logits=dq_det.unsqueeze(0).expand(3, batch_size, m_act).reshape(3 * batch_size, m_act),
                delta_put_loss=dput_det.unsqueeze(0).expand(3, batch_size).reshape(3 * batch_size),
            )
            sched_pts = sched_64.unsqueeze(0).expand(3, batch_size, m_act, d_sched).reshape(
                3 * batch_size, m_act, d_sched
            )
            glob_pts = glob_64.unsqueeze(0).expand(3, batch_size, glob_64.shape[-1]).reshape(
                3 * batch_size, glob_64.shape[-1]
            )
            smask_pts = smask_64.unsqueeze(0).expand(3, batch_size, m_act).reshape(3 * batch_size, m_act)
            pv_cg = assemble_surrogate_product_value(
                surr_cg,
                sched_pts,
                smask_pts,
                glob_pts,
                no_notional_repayment=no_notional_repayment,
                autocall_coupon=autocall_coupon,
                force_coupons_on_autocall=force_coupons_on_autocall,
                pv_in_transit=0.0,
            ).view(3, batch_size)
            s_w = w_lev.sum(dim=1)
            d2_pv_dfw2 = (pv_cg[2] - 2.0 * pv_cg[1] + pv_cg[0]) / (h_fd * h_fd)
            d2_cg = d2_pv_dfw2 * (s_w * s_w)
            cash_gamma_t = (d2_cg - g_log_m0_sum).to(dtype=torch.float64)

        g_node_f64 = torch.zeros((batch_size, d_max, d_node), dtype=torch.float64, device=device)
        g_node_f64[:, :d_act, :] = g_node_64

        rho_df_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
        rho_df_f64[:, :m_act] = g_sched_64[:, :, int(AutocallSchedCol.DF_PAY)] * smask_64

        stop_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
        stop_f64[:, :m_act] = out.stopping_probs[:, :m_act].detach().to(dtype=torch.float64)

        surv_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
        surv_f64[:, :m_act] = out.survival_probs[:, :m_act].detach().to(dtype=torch.float64)
        if m_act < m_max:
            surv_f64[:, m_act:] = surv_f64[:, m_act - 1 : m_act]

        cpn_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
        cpn_f64[:, :m_act] = out.coupon_probs[:, :m_act].detach().to(dtype=torch.float64)
        put_loss_f64 = out.put_loss.detach().to(dtype=torch.float64)

        d_log_m0_f64 = g_node_f64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)] * amask_f64
        if spots_t0 is not None:
            sp_f64 = spots_t0.to(dtype=torch.float64, device=device).clamp(min=1e-12)
            delta_f64 = (d_log_m0_f64 / sp_f64) * amask_f64
        else:
            log_m0_f64 = node_feats[:, :, int(AutocallNodeCol.LOG_MONEYNESS)].to(dtype=torch.float64)
            m0_f64 = torch.exp(log_m0_f64).clamp(min=1e-12)
            if ref_spots is not None:
                m0_f64 = m0_f64 * ref_spots.to(dtype=torch.float64, device=device).clamp(min=1e-12)
            delta_f64 = (d_log_m0_f64 / m0_f64) * amask_f64

        vega_surface_f64 = g_node_f64[:, :, 4:11] * amask_f64.unsqueeze(-1)
        vega_atm_f64 = g_node_f64[:, :, 4:6] * amask_f64.unsqueeze(-1)

        eye_act = torch.eye(d_act, dtype=surr_dtype, device=device).unsqueeze(0)
        pair_mask_act = amask_act.unsqueeze(1) * amask_act.unsqueeze(2) * (1.0 - eye_act)
        g_corr_act = g_edge_act[:, :, :, int(AutocallEdgeCol.CORR_RHO)].detach()
        cega_act = (g_corr_act + g_corr_act.transpose(1, 2)) * pair_mask_act
        cega_f64 = torch.zeros((batch_size, d_max, d_max), dtype=torch.float64, device=device)
        cega_f64[:, :d_act, :d_act] = cega_act.to(dtype=torch.float64)

        pv_f64 = pv.detach().to(dtype=torch.float64)
        if pv_in_transit is not None:
            pv_f64 = pv_f64 + torch.as_tensor(pv_in_transit, dtype=torch.float64, device=device)

    return BatchedSurrogateGreeksOutput(
        product_value=pv_f64,
        stopping_probs=stop_f64,
        survival_probs=surv_f64,
        coupon_probs=cpn_f64,
        put_loss=put_loss_f64.detach(),
        delta=delta_f64,
        vega_atm=vega_atm_f64,
        vega_surface_feats=vega_surface_f64,
        rho_df=rho_df_f64,
        cega=cega_f64,
        cash_gamma_euler=cash_gamma_t,
    )


__all__ = [
    "AutocallGreeksResult",
    "BatchedSurrogateGreeksOutput",
    "evaluate_differentiable_pv",
    "compute_euler_cash_gamma",
    "compute_eq_autocall_greeks",
    "compute_batched_surrogate_greeks",
]
