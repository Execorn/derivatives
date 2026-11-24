"""analytical_baseline.py: Analytical Multi-Asset Autocallable Basket Proxy.

Implements IEEE-754 float64 (`torch.float64` / `np.float64`) moment-matched pricing:
  - Tie-symmetrized Stulz-Margrabe recursive moments (`WorstOf`, `BestOf`), exact
    Levy-Wilkinson (`AverageOf`), exact lognormal (`RatioAB`/`RatioBA` respecting
    `inv_lex_perm`), shifted-normal (`Diff*`), and explicit flagging of exotic indices.
  - Exact closed-form truncated-normal + ramp survival moments via branch-cut-free
    `erfcx` Inverse Mills Ratio + continued-fraction tail, with Euler-weighted
    inter-period correlation and distinct `coupon_index` / `option_index` conditioning.
  - Exact piecewise terminal put/call integration over `LowerStrike`, conditional KI ramp,
    `AllowAutocallOnExpiry`, and 16-point bivariate Gauss-Legendre `OneStar` coupling.
  - Stateless CPU execution delegated to `Tier1MathBackend` (C++23 `-fno-fast-math` with
    automatic pure-NumPy fallback) and parity-matched differentiable `torch.float64` path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch

from deepvol.models.autocall._tier1_backend import (
    _GL8_W01,
    _GL8_X01,
    _GL16_W,
    _GL16_X,
    _GL16_Y_NORM,
    _GL16_YNORM,
    _GW4,
    _GX4,
    _LAG6_W,
    _LAG6_X,
    _eval_index_moments,
    get_tier1_backend,
    is_exact_analytical_index,
    monitoring_index_to_code,
)

_INV_SQRT2: float = 0.70710678118654752440084436210485
_INV_SQRT2PI: float = 0.39894228040143267793994605993438
_SQRT_HALF_PI: float = 1.2533141373155002512078826424055
_SQRT_TWO_OVER_PI: float = 0.79788456080286535587989211986876
_TIE_TAU: float = 0.008

_BALANCED_TREES_D4: Tuple[Tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),
    (2, 3, 0, 1),
    (0, 2, 1, 3),
    (1, 3, 0, 2),
    (0, 3, 1, 2),
    (1, 2, 0, 3),
)


def _build_balanced_trees_d5() -> Tuple[Tuple[int, int, int, int, int], ...]:
    trees: List[Tuple[int, int, int, int, int]] = []
    for e in range(5):
        r = [x for x in range(5) if x != e]
        for a, b, c, d in _BALANCED_TREES_D4:
            trees.append((r[a], r[b], r[c], r[d], e))
    return tuple(trees)


_BALANCED_TREES_D5: Tuple[Tuple[int, int, int, int, int], ...] = _build_balanced_trees_d5()


def _ndtr_scalar(x: float) -> float:
    """Fast IEEE-754 float64 standard normal CDF Phi(x) via math.erfc."""
    return 0.5 * math.erfc(-x * _INV_SQRT2)


def _bvn_cdf_pos_tail_torch(x: torch.Tensor, y: torch.Tensor, r_pos: torch.Tensor) -> torch.Tensor:
    """Genz upper-correlation bivariate normal CDF branch for r >= 0.925."""
    two_pi = 2.0 * math.pi
    h = -x
    k = -y
    hk = h * k
    a = torch.sqrt(((1.0 - r_pos) * (1.0 + r_pos)).clamp(min=1e-12))
    b = (h - k).abs()
    bs = b * b
    c = (4.0 - hk) * 0.125
    d = (12.0 - hk) * 0.0625
    asr = -0.5 * (bs / (a * a) + hk)
    bvn = torch.where(
        asr > -100.0,
        a * torch.exp(asr.clamp(min=-100.0)) * (1.0 - c * (bs - a * a) * (1.0 - d * bs * 0.2) / 3.0 + c * d * (a**4) * 0.2),
        torch.zeros_like(hk),
    )
    bvn = bvn - torch.where(
        -hk < 100.0,
        torch.exp((-0.5 * hk).clamp(min=-100.0))
        * math.sqrt(two_pi)
        * torch.special.ndtr(-b / a)
        * b
        * (1.0 - c * bs * (1.0 - d * bs * 0.2) / 3.0),
        torch.zeros_like(hk),
    )
    a_half = 0.5 * a
    gx_t = torch.tensor(_GL16_X, dtype=torch.float64, device=x.device)
    gw_t = torch.tensor(_GL16_W, dtype=torch.float64, device=x.device)
    xs = a_half.unsqueeze(-1) * (gx_t + 1.0)
    xs2 = xs * xs
    rs = torch.sqrt((1.0 - xs2).clamp(min=1e-30))
    asr_q = -0.5 * (bs.unsqueeze(-1) / xs2.clamp(min=1e-30) + hk.unsqueeze(-1))
    integrand = torch.where(
        asr_q > -100.0,
        torch.exp(asr_q.clamp(min=-100.0))
        * (torch.exp((-hk.unsqueeze(-1) * (1.0 - rs) / (2.0 * (1.0 + rs))).clamp(max=80.0)) / rs - (1.0 + c.unsqueeze(-1) * xs2 * (1.0 + d.unsqueeze(-1) * xs2))),
        torch.zeros_like(xs2),
    )
    bvn = bvn - a_half * (gw_t * integrand).sum(dim=-1)
    return cast(torch.Tensor, torch.special.ndtr(-torch.maximum(h, k)) - bvn / two_pi)


def _bvn_cdf_genz_torch(a: torch.Tensor, b: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Differentiable Genz bivariate normal CDF P(Z1 <= a, Z2 <= b; rho)."""
    z_bcast = torch.zeros_like(a) + torch.zeros_like(b) + torch.zeros_like(rho)
    a_t = a.expand_as(z_bcast)
    b_t = b.expand_as(z_bcast)
    rho_t = rho.expand_as(z_bcast)
    r = rho_t.clamp(-0.999999, 0.999999)
    small_r = r.abs() < 0.925
    r_sm = torch.where(small_r, r, torch.zeros_like(r))
    asr = torch.asin(r_sm)
    hs = 0.5 * (a_t * a_t + b_t * b_t)
    ab = a_t * b_t
    gx_t = torch.tensor(_GL16_X, dtype=torch.float64, device=a_t.device)
    gw_t = torch.tensor(_GL16_W, dtype=torch.float64, device=a_t.device)
    sn = torch.sin(0.5 * asr.unsqueeze(-1) * (gx_t + 1.0))
    int_sm = (0.5 * gw_t * torch.exp(((sn * ab.unsqueeze(-1) - hs.unsqueeze(-1)) / (1.0 - sn * sn).clamp(min=1e-12)).clamp(max=80.0))).sum(dim=-1)
    bvn_sm = torch.special.ndtr(a_t) * torch.special.ndtr(b_t) + int_sm * (asr / (2.0 * math.pi))

    r_pos = torch.where(~small_r, r.abs(), torch.full_like(r, 0.95))
    b_tail_in = torch.where(r < 0.0, -b_t, b_t)
    tail_pos = _bvn_cdf_pos_tail_torch(a_t, b_tail_in, r_pos)
    bvn_lg = torch.where(r < 0.0, torch.special.ndtr(a_t) - tail_pos, tail_pos)

    raw = torch.where(small_r, bvn_sm, bvn_lg)
    pa = torch.special.ndtr(a_t)
    pb = torch.special.ndtr(b_t)
    lo = (pa + pb - 1.0).clamp(min=0.0)
    hi = torch.minimum(pa, pb)
    return torch.minimum(torch.maximum(raw, lo), hi)


@dataclass
class AutocallBaselineOutput:
    """Analytical Stulz-Margrabe proxy outputs in torch.float64."""

    f_w: torch.Tensor
    v_w: torch.Tensor
    h_proxy: torch.Tensor
    p_proxy: torch.Tensor
    survival_proxy: torch.Tensor
    q_proxy: torch.Tensor
    l_put_proxy: torch.Tensor
    proxy_delta_weights: torch.Tensor
    option_pv_proxy: torch.Tensor
    notional_pv_proxy: torch.Tensor
    coupon_pv_proxy: torch.Tensor
    product_value_proxy: torch.Tensor
    l_call_proxy: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0, dtype=torch.float64))
    l_onestar_proxy: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0, dtype=torch.float64))
    p_onestar_proxy: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0, dtype=torch.float64))
    has_unsupported_exotic_index: bool = False
    unsupported_indices: Tuple[str, ...] = ()


def _black76_call_scalar(f: float, k: float, v: float) -> float:
    """Fast scalar float64 Black-76 call price."""
    f_c = f if f > 1e-16 else 1e-16
    k_c = k if k > 1e-16 else 1e-16
    v_c = v if v > 1e-12 else 1e-12
    d1 = (math.log(f_c / k_c) + 0.5 * v_c * v_c) / v_c
    d2 = d1 - v_c
    return f_c * _ndtr_scalar(d1) - k_c * _ndtr_scalar(d2)


def _black76_put_scalar(f: float, k: float, v: float) -> float:
    """Fast scalar float64 Black-76 put price."""
    f_c = f if f > 1e-16 else 1e-16
    k_c = k if k > 1e-16 else 1e-16
    v_c = v if v > 1e-12 else 1e-12
    d1 = (math.log(f_c / k_c) + 0.5 * v_c * v_c) / v_c
    d2 = d1 - v_c
    return k_c * _ndtr_scalar(-d2) - f_c * _ndtr_scalar(-d1)


def _digital_call_spread_prob_scalar(f: float, v: float, barrier: float, spread: float) -> float:
    """Exact lognormal expectation of a call-spread digital ramp in float64."""
    f_c = f if f > 1e-16 else 1e-16
    v_c = v if v > 1e-12 else 1e-12
    b_c = barrier if barrier > 1e-8 else 1e-8
    if abs(spread) <= 1e-6:
        d2_pt = (math.log(f_c / b_c) - 0.5 * v_c * v_c) / v_c
        return min(max(_ndtr_scalar(d2_pt), 0.0), 1.0)
    b_low = min(b_c, b_c + spread)
    b_high = max(b_c, b_c + spread)
    if b_high <= 0.0:
        return 1.0
    c_low = _black76_call_scalar(f_c, b_low, v_c) if b_low > 0.0 else (f_c - b_low)
    c_high = _black76_call_scalar(f_c, b_high, v_c)
    return min(max((c_low - c_high) / (b_high - b_low), 0.0), 1.0)


def fast_stulz_margrabe_cpu(
    f_arr: npt.NDArray[np.float64] | list[float],
    v_arr: npt.NDArray[np.float64] | list[float],
    corr: npt.NDArray[np.float64] | list[list[float]],
    k_strike: float = 0.90,
    monitoring_index: str = "WorstOf",
    inv_lex_perm: Optional[Sequence[int] | npt.NDArray[np.int64]] = None,
) -> tuple[float, float, float, npt.NDArray[np.float64]]:
    """Fast scalar float64 Stulz-Margrabe moment matching for single-trade CPU execution."""
    code = monitoring_index_to_code(monitoring_index)
    backend = get_tier1_backend()
    eval_mom = getattr(backend, "eval_index_moments_np", None)
    if (
        eval_mom is not None
        and isinstance(f_arr, np.ndarray)
        and isinstance(v_arr, np.ndarray)
        and isinstance(corr, np.ndarray)
    ):
        perm_np: Optional[npt.NDArray[np.int64]] = (
            inv_lex_perm
            if isinstance(inv_lex_perm, np.ndarray)
            else (np.asarray(inv_lex_perm, dtype=np.int64) if inv_lex_perm is not None else None)
        )
        f_w, v_w, w_out = eval_mom(f_arr, v_arr, corr, code, perm_np)
    else:
        f_list = f_arr.tolist() if isinstance(f_arr, np.ndarray) else [float(x) for x in f_arr]
        v_list = v_arr.tolist() if isinstance(v_arr, np.ndarray) else [float(x) for x in v_arr]
        c_list = corr.tolist() if isinstance(corr, np.ndarray) else [[float(z) for z in row] for row in corr]
        d_assets = len(f_list)
        perm_seq: Optional[Sequence[int]] = (
            inv_lex_perm.tolist() if isinstance(inv_lex_perm, np.ndarray) else inv_lex_perm
        )
        f_w, v_w, w_list = _eval_index_moments(d_assets, f_list, v_list, c_list, perm_seq, code)
        w_out = np.asarray(w_list, dtype=np.float64)
    if k_strike <= 0.0 and not (9 <= code <= 13):
        return 0.0, f_w, v_w, w_out
    if 9 <= code <= 13:
        s_c = max(v_w, 1e-12)
        z = (k_strike - f_w) / s_c
        put_val = (k_strike - f_w) * _ndtr_scalar(z) + s_c * _INV_SQRT2PI * math.exp(-0.5 * z * z)
        return put_val, f_w, v_w, w_out
    put_val = _black76_put_scalar(f_w, k_strike, v_w)
    return put_val, f_w, v_w, w_out


def _truncated_upper_std_moments_torch(alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact C^inf posterior (mu_u, var_u) of U ~ N(0,1) | U <= alpha via erfcx IMR + CF."""
    use_cf = alpha <= -4.0
    use_zero = alpha >= 26.0
    # CF branch (safe clamp keeps inactive branch finite)
    x_cf = -torch.where(use_cf, alpha, torch.full_like(alpha, -4.0))
    v = torch.zeros_like(x_cf)
    for k in range(80, 2, -1):
        v = float(k) / (x_cf + v)
    r2_cf = 2.0 / (x_cf + v)
    psi_cf = 1.0 / (x_cf + r2_cf)
    mu_cf = -x_cf - psi_cf
    var_cf = (psi_cf * (r2_cf - psi_cf)).clamp(min=1e-12)

    # erfcx IMR branch for -4.0 < alpha < 26.0
    a_mid = alpha.clamp(min=-4.0, max=25.999)
    lam_mid = _SQRT_TWO_OVER_PI / torch.special.erfcx(-a_mid * _INV_SQRT2)
    mu_mid = -lam_mid
    var_mid = (1.0 - a_mid * lam_mid - lam_mid * lam_mid).clamp(min=1e-12)

    mu = torch.where(use_cf, mu_cf, torch.where(use_zero, torch.zeros_like(alpha), mu_mid))
    var = torch.where(use_cf, var_cf, torch.where(use_zero, torch.ones_like(alpha), var_mid))
    return mu, var


def _clark_one_order_torch(
    f_in: torch.Tensor,
    v_in: torch.Tensor,
    c_in: torch.Tensor,
    order: torch.Tensor,
    is_best_of: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched differentiable Clark reduction along explicit integer permutation `order` (B, D)."""
    b_size, d_assets = f_in.shape
    device = f_in.device
    inv_order = torch.argsort(order, dim=1)
    f_ord = torch.gather(f_in, 1, order).clamp(min=1e-16)
    v_ord = torch.gather(v_in, 1, order).clamp(min=1e-12)
    ord_row = order.unsqueeze(2).expand(b_size, d_assets, d_assets)
    ord_col = order.unsqueeze(1).expand(b_size, d_assets, d_assets)
    c_ord = torch.gather(torch.gather(c_in, 1, ord_row), 2, ord_col)

    f_cur = f_ord[:, 0]
    v_cur = v_ord[:, 0]
    rho_rem = c_ord[:, 0, 1:]
    w1_steps: List[torch.Tensor] = []
    w2_steps: List[torch.Tensor] = []

    for j in range(1, d_assets):
        f_2 = f_ord[:, j]
        v_2 = v_ord[:, j]
        rho_12 = rho_rem[:, 0].clamp(-0.999999, 0.999999)
        v_cur_sq = v_cur * v_cur
        v_2_sq = v_2 * v_2
        var_m = (v_cur_sq + v_2_sq - 2.0 * rho_12 * v_cur * v_2).clamp(min=1e-16)
        sig_m = torch.sqrt(var_m)
        log_f_ratio = torch.log(f_cur / f_2)
        d1 = (log_f_ratio + 0.5 * var_m) / sig_m
        d2 = d1 - sig_m

        if is_best_of:
            w1 = torch.special.ndtr(d1)
            w2 = torch.special.ndtr(-d2)
        else:
            w1 = torch.special.ndtr(-d1)
            w2 = torch.special.ndtr(d2)

        e_m1 = (f_cur * w1 + f_2 * w2).clamp(min=1e-16)
        w1_steps.append(w1)
        w2_steps.append(w2)

        f1_sq = (f_cur * f_cur) * torch.exp(v_cur_sq)
        f2_sq = (f_2 * f_2) * torch.exp(v_2_sq)
        sig_m2 = 2.0 * sig_m
        var_m2 = sig_m2 * sig_m2
        d1_sq = (torch.log(f1_sq.clamp(min=1e-30) / f2_sq.clamp(min=1e-30)) + 0.5 * var_m2) / sig_m2
        d2_sq = d1_sq - sig_m2

        if is_best_of:
            e_m2 = f1_sq * torch.special.ndtr(d1_sq) + f2_sq * torch.special.ndtr(-d2_sq)
        else:
            e_m2 = f1_sq * torch.special.ndtr(-d1_sq) + f2_sq * torch.special.ndtr(d2_sq)

        v_new = torch.sqrt(
            torch.clamp(torch.log(torch.clamp(e_m2 / (e_m1 * e_m1).clamp(min=1e-30), min=1.0)), min=1e-16)
        )

        if j + 1 < d_assets:
            v_rem = v_ord[:, j + 1 :]
            rho_1_rem = rho_rem[:, 1:]
            rho_2_rem = c_ord[:, j, j + 1 :]
            exp_arg_1 = rho_1_rem * v_cur.unsqueeze(1) * v_rem
            exp_arg_2 = rho_2_rem * v_2.unsqueeze(1) * v_rem
            f1_k = f_cur.unsqueeze(1) * torch.exp(exp_arg_1)
            f2_k = f_2.unsqueeze(1) * torch.exp(exp_arg_2)
            var_m_col = var_m.unsqueeze(1)
            sig_m_col = sig_m.unsqueeze(1)
            d1_k = (log_f_ratio.unsqueeze(1) + (exp_arg_1 - exp_arg_2) + 0.5 * var_m_col) / sig_m_col
            d2_k = d1_k - sig_m_col
            if is_best_of:
                e_m_pk = f1_k * torch.special.ndtr(d1_k) + f2_k * torch.special.ndtr(-d2_k)
            else:
                e_m_pk = f1_k * torch.special.ndtr(-d1_k) + f2_k * torch.special.ndtr(d2_k)
            cov_log = torch.log(torch.clamp(e_m_pk / e_m1.unsqueeze(1), min=1e-16))
            rho_rem = torch.clamp(
                cov_log / (v_new.unsqueeze(1) * v_rem).clamp(min=1e-16),
                -0.999999,
                0.999999,
            )

        f_cur = e_m1
        v_cur = v_new

    w1_rev = torch.stack(w1_steps[::-1], dim=1)
    suff = torch.cumprod(w1_rev, dim=1).flip(1)
    ones_col = torch.ones((b_size, 1), dtype=torch.float64, device=device)
    w2_all = torch.cat([ones_col, torch.stack(w2_steps, dim=1)], dim=1)
    suff_all = torch.cat([suff, ones_col], dim=1)
    w_ord_t = w2_all * suff_all
    w_unsorted = torch.gather(w_ord_t, 1, inv_order)
    return f_cur, v_cur, w_unsorted


def _clark_symmetrized_torch(
    f_in: torch.Tensor,
    v_in: torch.Tensor,
    c_in: torch.Tensor,
    mask_in: Optional[torch.Tensor],
    is_best_of: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth tie-symmetrized Clark reduction eliminating dynamic argsort step jumps."""
    b_size, d_max = f_in.shape
    device = f_in.device

    if b_size == 0:
        z0 = torch.zeros(0, dtype=torch.float64, device=device)
        return z0, z0.clone(), torch.zeros((0, d_max), dtype=torch.float64, device=device)

    if d_max == 1:
        f_0 = f_in[:, 0].clamp(min=1e-16)
        v_0 = v_in[:, 0].clamp(min=1e-12)
        w_0 = torch.ones_like(f_in) if mask_in is None else mask_in
        return f_0, v_0, w_0

    f_safe = f_in.clamp(min=1e-16)
    v_safe = v_in.clamp(min=1e-12)
    if mask_in is not None:
        active_b = mask_in > 0.5
        f_safe = torch.where(active_b, f_safe, torch.ones_like(f_safe))
        v_safe = torch.where(active_b, v_safe, torch.full_like(v_safe, 0.1))
        pair_active = active_b.unsqueeze(2) & active_b.unsqueeze(1)
        eye_d = torch.eye(d_max, dtype=torch.float64, device=device).unsqueeze(0)
        c_safe = torch.where(pair_active, c_in, eye_d)
    else:
        active_b = torch.ones((b_size, d_max), dtype=torch.bool, device=device)
        c_safe = c_in

    # Smooth C^inf tie indicator across active asset pairs
    g_sep = torch.ones(b_size, dtype=torch.float64, device=device)
    for i in range(d_max):
        for j in range(i + 1, d_max):
            dl = torch.log(f_safe[:, i] / f_safe[:, j])
            p_ij = torch.exp(-(dl * dl) / (2.0 * _TIE_TAU * _TIE_TAU))
            one_m = 1.0 - p_ij
            pair_on = active_b[:, i] & active_b[:, j]
            factor = torch.where(pair_on, one_m * one_m, torch.ones_like(one_m))
            g_sep = g_sep * factor
    w_tie = 1.0 - g_sep

    # Move inactive slots to the end so active prefix reduces cleanly
    if mask_in is None:
        sort_key_f = -f_safe.detach() if is_best_of else f_safe.detach()
    else:
        base_key = -f_safe.detach() if is_best_of else f_safe.detach()
        sort_key_f = torch.where(active_b, base_key, torch.full_like(base_key, 1e12))
    ord_f = torch.argsort(sort_key_f, dim=1, stable=True)

    # Fast path when all rows have well-separated forwards and no mask padding complications
    w_tie_max = float(w_tie.detach().max().item()) if b_size > 0 else 0.0

    def _eval_masked_clark_order(order_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask_in is None:
            return _clark_one_order_torch(f_safe, v_safe, c_safe, order_b, is_best_of)
        # Compact active slots first while respecting order_b on active elements
        inv_o = torch.argsort(order_b, dim=1)
        f_o = torch.gather(f_safe, 1, order_b)
        v_o = torch.gather(v_safe, 1, order_b)
        m_o = torch.gather(mask_in, 1, order_b)
        ord_row = order_b.unsqueeze(2).expand(b_size, d_max, d_max)
        ord_col = order_b.unsqueeze(1).expand(b_size, d_max, d_max)
        c_o = torch.gather(torch.gather(c_safe, 1, ord_row), 2, ord_col)

        f_cur = f_o[:, 0]
        v_cur = v_o[:, 0]
        rho_cur = c_o[:, 0, :]
        w_list: List[torch.Tensor] = [torch.ones((b_size,), dtype=torch.float64, device=device)]
        for _ in range(1, d_max):
            w_list.append(torch.zeros((b_size,), dtype=torch.float64, device=device))

        for j in range(1, d_max):
            active_j = m_o[:, j] > 0.5
            if not bool(torch.any(active_j)):
                break
            f_2 = f_o[:, j]
            v_2 = v_o[:, j]
            rho_12 = rho_cur[:, j].clamp(-0.999999, 0.999999)
            var_m = (v_cur * v_cur + v_2 * v_2 - 2.0 * rho_12 * v_cur * v_2).clamp(min=1e-16)
            sig_m = torch.sqrt(var_m)
            d1 = (torch.log(f_cur / f_2) + 0.5 * var_m) / sig_m
            d2 = d1 - sig_m
            if is_best_of:
                w1 = torch.special.ndtr(d1)
                w2 = torch.special.ndtr(-d2)
            else:
                w1 = torch.special.ndtr(-d1)
                w2 = torch.special.ndtr(d2)
            e_m1 = (f_cur * w1 + f_2 * w2).clamp(min=1e-16)
            f1_sq = (f_cur * f_cur) * torch.exp(v_cur * v_cur)
            f2_sq = (f_2 * f_2) * torch.exp(v_2 * v_2)
            sig_m2 = 2.0 * sig_m
            var_m2 = sig_m2 * sig_m2
            d1_sq = (torch.log(f1_sq.clamp(min=1e-30) / f2_sq.clamp(min=1e-30)) + 0.5 * var_m2) / sig_m2
            d2_sq = d1_sq - sig_m2
            if is_best_of:
                e_m2 = f1_sq * torch.special.ndtr(d1_sq) + f2_sq * torch.special.ndtr(-d2_sq)
            else:
                e_m2 = f1_sq * torch.special.ndtr(-d1_sq) + f2_sq * torch.special.ndtr(d2_sq)
            v_new = torch.sqrt(
                torch.clamp(torch.log(torch.clamp(e_m2 / (e_m1 * e_m1).clamp(min=1e-30), min=1.0)), min=1e-16)
            )
            for prev_idx in range(j):
                w_list[prev_idx] = torch.where(active_j, w_list[prev_idx] * w1, w_list[prev_idx])
            w_list[j] = torch.where(active_j, w2, torch.zeros_like(w2))
            if j + 1 < d_max:
                v_rem = v_o[:, j + 1 :]
                rho_1_rem = rho_cur[:, j + 1 :]
                rho_2_rem = c_o[:, j, j + 1 :]
                f1_k = f_cur.unsqueeze(1) * torch.exp(rho_1_rem * v_cur.unsqueeze(1) * v_rem)
                f2_k = f_2.unsqueeze(1) * torch.exp(rho_2_rem * v_2.unsqueeze(1) * v_rem)
                var_m_col = var_m.unsqueeze(1)
                sig_m_col = sig_m.unsqueeze(1)
                d1_k = (torch.log(f1_k.clamp(min=1e-16) / f2_k.clamp(min=1e-16)) + 0.5 * var_m_col) / sig_m_col
                d2_k = d1_k - sig_m_col
                if is_best_of:
                    e_m_pk = f1_k * torch.special.ndtr(d1_k) + f2_k * torch.special.ndtr(-d2_k)
                else:
                    e_m_pk = f1_k * torch.special.ndtr(-d1_k) + f2_k * torch.special.ndtr(d2_k)
                cov_log = torch.log(torch.clamp(e_m_pk / e_m1.unsqueeze(1), min=1e-16))
                rho_new_rem = torch.clamp(
                    cov_log / (v_new.unsqueeze(1) * v_rem).clamp(min=1e-16),
                    -0.999999,
                    0.999999,
                )
                updated_rem = torch.where(active_j.unsqueeze(1), rho_new_rem, rho_1_rem)
                rho_cur = torch.cat([rho_cur[:, : j + 1], updated_rem], dim=1)
            f_cur = torch.where(active_j, e_m1, f_cur)
            v_cur = torch.where(active_j, v_new, v_cur)

        w_o_t = torch.stack(w_list, dim=1) * m_o
        return f_cur, v_cur, torch.gather(w_o_t, 1, inv_o)

    if w_tie_max < 1e-12:
        return _eval_masked_clark_order(ord_f)

    def _clark_pair_merge(
        f1: torch.Tensor,
        v1: torch.Tensor,
        f2: torch.Tensor,
        v2: torch.Tensor,
        rho_12: torch.Tensor,
        v_rem: Optional[torch.Tensor],
        rho_1_rem: Optional[torch.Tensor],
        rho_2_rem: Optional[torch.Tensor],
        best_of: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        rho_c = rho_12.clamp(-0.999999, 0.999999)
        var_m = (v1 * v1 + v2 * v2 - 2.0 * rho_c * v1 * v2).clamp(min=1e-16)
        sig_m = torch.sqrt(var_m)
        log_f_ratio = torch.log(f1 / f2)
        d1 = (log_f_ratio + 0.5 * var_m) / sig_m
        d2 = d1 - sig_m
        if best_of:
            w1 = torch.special.ndtr(d1)
            w2 = torch.special.ndtr(-d2)
        else:
            w1 = torch.special.ndtr(-d1)
            w2 = torch.special.ndtr(d2)
        e_m1 = (f1 * w1 + f2 * w2).clamp(min=1e-16)

        f1_sq = (f1 * f1) * torch.exp(v1 * v1)
        f2_sq = (f2 * f2) * torch.exp(v2 * v2)
        sig_m2 = 2.0 * sig_m
        var_m2 = sig_m2 * sig_m2
        d1_sq = (torch.log(f1_sq.clamp(min=1e-30) / f2_sq.clamp(min=1e-30)) + 0.5 * var_m2) / sig_m2
        d2_sq = d1_sq - sig_m2
        if best_of:
            e_m2 = f1_sq * torch.special.ndtr(d1_sq) + f2_sq * torch.special.ndtr(-d2_sq)
        else:
            e_m2 = f1_sq * torch.special.ndtr(-d1_sq) + f2_sq * torch.special.ndtr(d2_sq)

        v_new = torch.sqrt(
            torch.clamp(torch.log(torch.clamp(e_m2 / (e_m1 * e_m1).clamp(min=1e-30), min=1.0)), min=1e-16)
        )

        rho_m_rem: Optional[torch.Tensor] = None
        if v_rem is not None and rho_1_rem is not None and rho_2_rem is not None:
            exp_arg_1 = rho_1_rem * v1.unsqueeze(1) * v_rem
            exp_arg_2 = rho_2_rem * v2.unsqueeze(1) * v_rem
            f1_k = f1.unsqueeze(1) * torch.exp(exp_arg_1)
            f2_k = f2.unsqueeze(1) * torch.exp(exp_arg_2)
            var_m_col = var_m.unsqueeze(1)
            sig_m_col = sig_m.unsqueeze(1)
            d1_k = (log_f_ratio.unsqueeze(1) + (exp_arg_1 - exp_arg_2) + 0.5 * var_m_col) / sig_m_col
            d2_k = d1_k - sig_m_col
            if best_of:
                e_m_pk = f1_k * torch.special.ndtr(d1_k) + f2_k * torch.special.ndtr(-d2_k)
            else:
                e_m_pk = f1_k * torch.special.ndtr(-d1_k) + f2_k * torch.special.ndtr(d2_k)
            cov_log = torch.log(torch.clamp(e_m_pk / e_m1.unsqueeze(1), min=1e-16))
            rho_m_rem = torch.clamp(
                cov_log / (v_new.unsqueeze(1) * v_rem).clamp(min=1e-16),
                -0.999999,
                0.999999,
            )
        return e_m1, v_new, w1, w2, rho_m_rem

    def _symmetrized_balanced_clark_batch(
        f_b: torch.Tensor,
        v_b: torch.Tensor,
        c_b: torch.Tensor,
        best_of: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """S_D-symmetric, C^inf balanced Clark tree reduction without argsort(v)."""
        b_sub, d_sub = f_b.shape
        if d_sub == 0:
            z1 = torch.zeros(b_sub, dtype=torch.float64, device=device)
            return z1, z1, torch.zeros((b_sub, 0), dtype=torch.float64, device=device)
        if d_sub == 1:
            f1 = f_b[:, 0]
            v1 = v_b[:, 0]
            return f1, f1 * f1 * torch.exp(v1 * v1), torch.ones((b_sub, 1), dtype=torch.float64, device=device)
        if d_sub == 2:
            f_m, v_m, w1, w2, _ = _clark_pair_merge(
                f_b[:, 0], v_b[:, 0], f_b[:, 1], v_b[:, 1], c_b[:, 0, 1], None, None, None, best_of
            )
            return f_m, f_m * f_m * torch.exp(v_m * v_m), torch.stack([w1, w2], dim=1)
        if d_sub == 3:
            perms_3 = torch.tensor([[0, 1, 2], [0, 2, 1], [1, 2, 0]], dtype=torch.long, device=device)
            n_t = 3
            f_rep = f_b.unsqueeze(0).expand(n_t, b_sub, 3).reshape(n_t * b_sub, 3)
            v_rep = v_b.unsqueeze(0).expand(n_t, b_sub, 3).reshape(n_t * b_sub, 3)
            c_rep = c_b.unsqueeze(0).expand(n_t, b_sub, 3, 3).reshape(n_t * b_sub, 3, 3)
            p_rep = perms_3.unsqueeze(1).expand(n_t, b_sub, 3).reshape(n_t * b_sub, 3)
            ft, vt, wt = _clark_one_order_torch(f_rep, v_rep, c_rep, p_rep, best_of)
            ft_v = ft.view(n_t, b_sub)
            vt_v = vt.view(n_t, b_sub)
            wt_v = wt.view(n_t, b_sub, 3)
            return ft_v.mean(dim=0), (ft_v * ft_v * torch.exp(vt_v * vt_v)).mean(dim=0), wt_v.mean(dim=0)
        if d_sub == 4:
            trees_4 = torch.tensor(_BALANCED_TREES_D4, dtype=torch.long, device=device)
            n_t = 6
            f_rep = f_b.unsqueeze(0).expand(n_t, b_sub, 4).reshape(n_t * b_sub, 4)
            v_rep = v_b.unsqueeze(0).expand(n_t, b_sub, 4).reshape(n_t * b_sub, 4)
            c_rep = c_b.unsqueeze(0).expand(n_t, b_sub, 4, 4).reshape(n_t * b_sub, 4, 4)
            p_rep = trees_4.unsqueeze(1).expand(n_t, b_sub, 4).reshape(n_t * b_sub, 4)
            fo = torch.gather(f_rep, 1, p_rep)
            vo = torch.gather(v_rep, 1, p_rep)
            co = torch.gather(
                torch.gather(c_rep, 1, p_rep.unsqueeze(2).expand(n_t * b_sub, 4, 4)),
                2,
                p_rep.unsqueeze(1).expand(n_t * b_sub, 4, 4),
            )
            f_ab, v_ab, w_a, w_b, rho_ab_cd = _clark_pair_merge(
                fo[:, 0], vo[:, 0], fo[:, 1], vo[:, 1], co[:, 0, 1],
                vo[:, 2:4], co[:, 0, 2:4], co[:, 1, 2:4], best_of,
            )
            assert rho_ab_cd is not None
            f_cd, v_cd, w_c, w_d, rho_cd_ab = _clark_pair_merge(
                fo[:, 2], vo[:, 2], fo[:, 3], vo[:, 3], co[:, 2, 3],
                v_ab.unsqueeze(1), rho_ab_cd[:, 0:1], rho_ab_cd[:, 1:2], best_of,
            )
            assert rho_cd_ab is not None
            f_root, v_root, w_ab, w_cd, _ = _clark_pair_merge(
                f_ab, v_ab, f_cd, v_cd, rho_cd_ab[:, 0], None, None, None, best_of
            )
            w_leaves = torch.stack([w_ab * w_a, w_ab * w_b, w_cd * w_c, w_cd * w_d], dim=1)
            w_unsorted = torch.zeros_like(w_leaves).scatter(1, p_rep, w_leaves)
            fr_v = f_root.view(n_t, b_sub)
            vr_v = v_root.view(n_t, b_sub)
            return fr_v.mean(dim=0), (fr_v * fr_v * torch.exp(vr_v * vr_v)).mean(dim=0), w_unsorted.view(n_t, b_sub, 4).mean(dim=0)
        if d_sub == 5:
            trees_5 = torch.tensor(_BALANCED_TREES_D5, dtype=torch.long, device=device)
            n_t = 30
            f_rep = f_b.unsqueeze(0).expand(n_t, b_sub, 5).reshape(n_t * b_sub, 5)
            v_rep = v_b.unsqueeze(0).expand(n_t, b_sub, 5).reshape(n_t * b_sub, 5)
            c_rep = c_b.unsqueeze(0).expand(n_t, b_sub, 5, 5).reshape(n_t * b_sub, 5, 5)
            p_rep = trees_5.unsqueeze(1).expand(n_t, b_sub, 5).reshape(n_t * b_sub, 5)
            fo = torch.gather(f_rep, 1, p_rep)
            vo = torch.gather(v_rep, 1, p_rep)
            co = torch.gather(
                torch.gather(c_rep, 1, p_rep.unsqueeze(2).expand(n_t * b_sub, 5, 5)),
                2,
                p_rep.unsqueeze(1).expand(n_t * b_sub, 5, 5),
            )
            f_ab, v_ab, w_a, w_b, rho_ab_cde = _clark_pair_merge(
                fo[:, 0], vo[:, 0], fo[:, 1], vo[:, 1], co[:, 0, 1],
                vo[:, 2:5], co[:, 0, 2:5], co[:, 1, 2:5], best_of,
            )
            assert rho_ab_cde is not None
            v_rem2 = torch.stack([v_ab, vo[:, 4]], dim=1)
            r1_rem2 = torch.stack([rho_ab_cde[:, 0], co[:, 2, 4]], dim=1)
            r2_rem2 = torch.stack([rho_ab_cde[:, 1], co[:, 3, 4]], dim=1)
            f_cd, v_cd, w_c, w_d, rho_cd_abe = _clark_pair_merge(
                fo[:, 2], vo[:, 2], fo[:, 3], vo[:, 3], co[:, 2, 3],
                v_rem2, r1_rem2, r2_rem2, best_of,
            )
            assert rho_cd_abe is not None
            f_abcd, v_abcd, w_ab, w_cd, rho_abcd_e = _clark_pair_merge(
                f_ab, v_ab, f_cd, v_cd, rho_cd_abe[:, 0],
                vo[:, 4:5], rho_ab_cde[:, 2:3], rho_cd_abe[:, 1:2], best_of,
            )
            assert rho_abcd_e is not None
            f_root, v_root, w_abcd, w_e, _ = _clark_pair_merge(
                f_abcd, v_abcd, fo[:, 4], vo[:, 4], rho_abcd_e[:, 0], None, None, None, best_of
            )
            w_leaves = torch.stack(
                [w_abcd * w_ab * w_a, w_abcd * w_ab * w_b, w_abcd * w_cd * w_c, w_abcd * w_cd * w_d, w_e],
                dim=1,
            )
            w_unsorted = torch.zeros_like(w_leaves).scatter(1, p_rep, w_leaves)
            fr_v = f_root.view(n_t, b_sub)
            vr_v = v_root.view(n_t, b_sub)
            return fr_v.mean(dim=0), (fr_v * fr_v * torch.exp(vr_v * vr_v)).mean(dim=0), w_unsorted.view(n_t, b_sub, 5).mean(dim=0)

        # D >= 6: canonical cyclic + reverse shifts without data-dependent argsort(v)
        base_ord = torch.arange(d_sub, dtype=torch.long, device=device).unsqueeze(0).expand(b_sub, d_sub)
        perms_cyc = [torch.roll(base_ord, shifts=-s, dims=1) for s in range(d_sub)]
        perms_cyc = perms_cyc + [torch.flip(p, dims=[1]) for p in perms_cyc]
        n_p = len(perms_cyc)
        p_cat = torch.cat(perms_cyc, dim=0)
        f_rep = f_b.repeat(n_p, 1)
        v_rep = v_b.repeat(n_p, 1)
        c_rep = c_b.repeat(n_p, 1, 1)
        ft, vt, wt = _clark_one_order_torch(f_rep, v_rep, c_rep, p_cat, best_of)
        ft_v = ft.view(n_p, b_sub)
        vt_v = vt.view(n_p, b_sub)
        return ft_v.mean(dim=0), (ft_v * ft_v * torch.exp(vt_v * vt_v)).mean(dim=0), wt.view(n_p, b_sub, d_sub).mean(dim=0)

    if mask_in is None:
        f_sym, m2_sym, w_sym = _symmetrized_balanced_clark_batch(f_safe, v_safe, c_safe, is_best_of)
        v_sym = torch.sqrt(
            torch.clamp(torch.log(torch.clamp(m2_sym / (f_sym * f_sym).clamp(min=1e-30), min=1.0)), min=1e-16)
        )
    else:
        ord_act = torch.argsort((~active_b).to(dtype=torch.int64), dim=1, stable=True)
        active_cnt = active_b.sum(dim=1).to(dtype=torch.long)
        min_act = int(active_cnt.min().item()) if b_size > 0 else 0
        max_act = int(active_cnt.max().item()) if b_size > 0 else 0
        if min_act == max_act:
            act_idx = ord_act[:, :min_act]
            f_act = torch.gather(f_safe, 1, act_idx)
            v_act = torch.gather(v_safe, 1, act_idx)
            c_act = torch.gather(
                torch.gather(c_safe, 1, act_idx.unsqueeze(2).expand(b_size, min_act, d_max)),
                2,
                act_idx.unsqueeze(1).expand(b_size, min_act, min_act),
            )
            f_sym, m2_sym, w_act = _symmetrized_balanced_clark_batch(f_act, v_act, c_act, is_best_of)
            w_sym = torch.zeros((b_size, d_max), dtype=torch.float64, device=device).scatter(1, act_idx, w_act)
        else:
            f_rows: List[torch.Tensor] = []
            m2_rows: List[torch.Tensor] = []
            w_rows: List[torch.Tensor] = []
            for r in range(b_size):
                d_act = int(active_cnt[r].item())
                act_idx_r = ord_act[r : r + 1, :d_act]
                f_act_r = torch.gather(f_safe[r : r + 1], 1, act_idx_r)
                v_act_r = torch.gather(v_safe[r : r + 1], 1, act_idx_r)
                c_act_r = torch.gather(
                    torch.gather(c_safe[r : r + 1], 1, act_idx_r.unsqueeze(2).expand(1, d_act, d_max)),
                    2,
                    act_idx_r.unsqueeze(1).expand(1, d_act, d_act),
                )
                f_r, m2_r, w_act_r = _symmetrized_balanced_clark_batch(f_act_r, v_act_r, c_act_r, is_best_of)
                w_r = torch.zeros((1, d_max), dtype=torch.float64, device=device).scatter(1, act_idx_r, w_act_r)
                f_rows.append(f_r)
                m2_rows.append(m2_r)
                w_rows.append(w_r)
            f_sym = torch.cat(f_rows, dim=0)
            m2_sym = torch.cat(m2_rows, dim=0)
            w_sym = torch.cat(w_rows, dim=0)
        v_sym = torch.sqrt(
            torch.clamp(torch.log(torch.clamp(m2_sym / (f_sym * f_sym).clamp(min=1e-30), min=1.0)), min=1e-16)
        )

    if b_size == 0 or float(g_sep.detach().max().item()) < 1e-15:
        return f_sym, v_sym, w_sym

    f_f, v_f, w_f = _eval_masked_clark_order(ord_f)
    f_out = g_sep * f_f + w_tie * f_sym
    m2_out = g_sep * (f_f * f_f * torch.exp(v_f * v_f)) + w_tie * m2_sym
    v_out = torch.sqrt(torch.clamp(torch.log(torch.clamp(m2_out / (f_out * f_out).clamp(min=1e-30), min=1.0)), min=1e-16))
    w_out = g_sep.unsqueeze(1) * w_f + w_tie.unsqueeze(1) * w_sym
    return f_out, v_out, w_out


def _clark_corr_min_max_torch(
    f_in: torch.Tensor,
    v_in: torch.Tensor,
    corr: torch.Tensor,
) -> torch.Tensor:
    """Computes batched Stulz-Clark correlation between ln(W_min) and ln(W_max) for (B, D) or (D,)."""
    squeeze_batch = f_in.ndim == 1
    f_b = f_in.unsqueeze(0) if squeeze_batch else f_in
    v_b = v_in.unsqueeze(0) if squeeze_batch else v_in
    c_b = corr.unsqueeze(0) if corr.ndim == 2 else corr
    b_size, d = int(f_b.shape[0]), int(f_b.shape[1])
    device = f_b.device
    if b_size == 0:
        out0 = torch.zeros(0, dtype=torch.float64, device=device)
        return out0
    if d <= 1:
        out1 = torch.ones(b_size, dtype=torch.float64, device=device)
        return out1.squeeze(0) if squeeze_batch else out1

    f_s = f_b.clamp(min=1e-16)
    v_s = v_b.clamp(min=1e-12)
    v_ord_0 = torch.argsort(v_s.detach(), dim=1, stable=True)
    f_round = torch.round(f_s.detach() * 1e9)
    f_tie = torch.gather(f_round, 1, v_ord_0)
    idx_min = torch.gather(v_ord_0, 1, torch.argsort(f_tie, dim=1, stable=True))
    idx_max = torch.gather(v_ord_0, 1, torch.argsort(-f_tie, dim=1, stable=True))

    b_idx = torch.arange(b_size, device=device).unsqueeze(1)
    fo = torch.gather(f_s, 1, idx_min)
    vo = torch.gather(v_s, 1, idx_min)
    c_min_full = c_b[b_idx, idx_min, :]
    f_cur = fo[:, 0]
    v_cur = vo[:, 0]
    rho_m_p = c_min_full[:, 0, :]

    for j in range(1, d):
        f_2 = fo[:, j]
        v_2 = vo[:, j]
        oj = idx_min[:, j]
        rho_12 = torch.gather(rho_m_p, 1, oj.unsqueeze(1)).squeeze(1).clamp(-0.999999, 0.999999)
        var_m = (v_cur * v_cur + v_2 * v_2 - 2.0 * rho_12 * v_cur * v_2).clamp(min=1e-16)
        sig_m = torch.sqrt(var_m)
        d1 = (torch.log(f_cur / f_2) + 0.5 * var_m) / sig_m
        d2 = d1 - sig_m
        e_m1 = (f_cur * torch.special.ndtr(-d1) + f_2 * torch.special.ndtr(d2)).clamp(min=1e-16)
        f1_sq = f_cur * f_cur * torch.exp(v_cur * v_cur)
        f2_sq = f_2 * f_2 * torch.exp(v_2 * v_2)
        sig_m2 = 2.0 * sig_m
        var_m2 = sig_m2 * sig_m2
        d1_sq = (torch.log(f1_sq.clamp(min=1e-30) / f2_sq.clamp(min=1e-30)) + 0.5 * var_m2) / sig_m2
        d2_sq = d1_sq - sig_m2
        e_m2 = f1_sq * torch.special.ndtr(-d1_sq) + f2_sq * torch.special.ndtr(d2_sq)
        v_new = torch.sqrt(torch.clamp(torch.log(torch.clamp(e_m2 / (e_m1 * e_m1).clamp(min=1e-30), min=1.0)), min=1e-16))

        c_oj = c_b[torch.arange(b_size, device=device), oj, :]
        f1_i = f_cur.unsqueeze(1) * torch.exp(rho_m_p * (v_cur.unsqueeze(1) * v_s))
        f2_i = f_2.unsqueeze(1) * torch.exp(c_oj * (v_2.unsqueeze(1) * v_s))
        d1_i = (torch.log(f1_i.clamp(min=1e-16) / f2_i.clamp(min=1e-16)) + 0.5 * var_m.unsqueeze(1)) / sig_m.unsqueeze(1)
        d2_i = d1_i - sig_m.unsqueeze(1)
        e_mp_i = f1_i * torch.special.ndtr(-d1_i) + f2_i * torch.special.ndtr(d2_i)
        rho_m_p = (
            torch.log(torch.clamp(e_mp_i / e_m1.unsqueeze(1), min=1e-16))
            / (v_new.unsqueeze(1) * v_s).clamp(min=1e-16)
        ).clamp(-0.999999, 0.999999)
        f_cur, v_cur = e_m1, v_new

    v_min = v_cur
    fob = torch.gather(f_s, 1, idx_max)
    vob = torch.gather(v_s, 1, idx_max)
    cob = c_b[b_idx.unsqueeze(2), idx_max.unsqueeze(2), idx_max.unsqueeze(1)]
    f_b_cur = fob[:, 0]
    v_b_cur = vob[:, 0]
    rho_cur = cob[:, 0, :]
    rho_b_min = torch.gather(rho_m_p, 1, idx_max[:, :1]).squeeze(1)

    for j in range(1, d):
        f_2 = fob[:, j]
        v_2 = vob[:, j]
        rho_12 = rho_cur[:, j].clamp(-0.999999, 0.999999)
        var_m = (v_b_cur * v_b_cur + v_2 * v_2 - 2.0 * rho_12 * v_b_cur * v_2).clamp(min=1e-16)
        sig_m = torch.sqrt(var_m)
        d1 = (torch.log(f_b_cur / f_2) + 0.5 * var_m) / sig_m
        d2 = d1 - sig_m
        e_m1 = (f_b_cur * torch.special.ndtr(d1) + f_2 * torch.special.ndtr(-d2)).clamp(min=1e-16)
        f1_sq = f_b_cur * f_b_cur * torch.exp(v_b_cur * v_b_cur)
        f2_sq = f_2 * f_2 * torch.exp(v_2 * v_2)
        sig_m2 = 2.0 * sig_m
        var_m2 = sig_m2 * sig_m2
        d1_sq = (torch.log(f1_sq.clamp(min=1e-30) / f2_sq.clamp(min=1e-30)) + 0.5 * var_m2) / sig_m2
        d2_sq = d1_sq - sig_m2
        e_m2 = f1_sq * torch.special.ndtr(d1_sq) + f2_sq * torch.special.ndtr(-d2_sq)
        v_new = torch.sqrt(torch.clamp(torch.log(torch.clamp(e_m2 / (e_m1 * e_m1).clamp(min=1e-30), min=1.0)), min=1e-16))

        oj = idx_max[:, j]
        rho_oj_min = torch.gather(rho_m_p, 1, oj.unsqueeze(1)).squeeze(1)
        f1_min = f_b_cur * torch.exp(rho_b_min * v_b_cur * v_min)
        f2_min = f_2 * torch.exp(rho_oj_min * v_2 * v_min)
        d1_min = (torch.log(f1_min.clamp(min=1e-16) / f2_min.clamp(min=1e-16)) + 0.5 * var_m) / sig_m
        d2_min = d1_min - sig_m
        e_b_min = f1_min * torch.special.ndtr(d1_min) + f2_min * torch.special.ndtr(-d2_min)
        rho_b_min = (torch.log(torch.clamp(e_b_min / e_m1, min=1e-16)) / (v_new * v_min).clamp(min=1e-16)).clamp(
            -0.999999, 0.999999
        )

        if j + 1 < d:
            v_rem = vob[:, j + 1 :]
            rho_rem_cur = rho_cur[:, j + 1 :]
            c_j_rem = cob[:, j, j + 1 :]
            f1_k = f_b_cur.unsqueeze(1) * torch.exp(rho_rem_cur * (v_b_cur.unsqueeze(1) * v_rem))
            f2_k = f_2.unsqueeze(1) * torch.exp(c_j_rem * (v_2.unsqueeze(1) * v_rem))
            d1_k = (torch.log(f1_k.clamp(min=1e-16) / f2_k.clamp(min=1e-16)) + 0.5 * var_m.unsqueeze(1)) / sig_m.unsqueeze(1)
            d2_k = d1_k - sig_m.unsqueeze(1)
            e_mp_k = f1_k * torch.special.ndtr(d1_k) + f2_k * torch.special.ndtr(-d2_k)
            rho_rem = (
                torch.log(torch.clamp(e_mp_k / e_m1.unsqueeze(1), min=1e-16))
                / (v_new.unsqueeze(1) * v_rem).clamp(min=1e-16)
            ).clamp(-0.999999, 0.999999)
            rho_cur = torch.cat([rho_cur[:, : j + 1], rho_rem], dim=1)
        f_b_cur, v_b_cur = e_m1, v_new
    res = rho_b_min.clamp(-0.9999, 0.9999)
    return res.squeeze(0) if squeeze_batch else res


def _fit_1factor_loadings_torch(corr_b: torch.Tensor) -> torch.Tensor:
    """Fits differentiable rank-1 correlation loadings for batched correlation matrices (B, D, D)."""
    b_size, d_assets, _ = corr_b.shape
    device = corr_b.device
    if d_assets <= 1 or b_size == 0:
        return torch.ones((b_size, d_assets), dtype=torch.float64, device=device)
    corr_sum = corr_b.sum(dim=(1, 2))
    rho_off = ((corr_sum - float(d_assets)) / float(d_assets * (d_assets - 1))).clamp(0.0, 0.999)
    b_vec = torch.sqrt(rho_off).unsqueeze(1).expand(b_size, d_assets).clone()
    eye_d = torch.eye(d_assets, dtype=torch.float64, device=device).unsqueeze(0)
    corr_off = corr_b * (1.0 - eye_d)
    for _ in range(8):
        mb = torch.bmm(corr_off, b_vec.unsqueeze(2)).squeeze(2) + (b_vec * b_vec) * b_vec
        mb_norm = torch.sqrt((mb * mb).sum(dim=1, keepdim=True).clamp(min=1e-30))
        u_vec = mb / mb_norm
        sgn = torch.where(u_vec.sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
        u_vec = u_vec * sgn
        mu_vec = torch.bmm(corr_off, u_vec.unsqueeze(2)).squeeze(2) + (b_vec * b_vec) * u_vec
        lam = (u_vec * mu_vec).sum(dim=1, keepdim=True).clamp(min=0.0)
        b_vec = (u_vec * torch.sqrt(lam)).clamp(0.0, 0.999)
    return b_vec


def stulz_margrabe_moments_with_weights(
    f_perf: torch.Tensor,
    v_perf: torch.Tensor,
    corr: torch.Tensor,
    asset_mask: torch.Tensor | None = None,
    monitoring_index: str = "WorstOf",
    inv_lex_perm: Optional[Sequence[int] | npt.NDArray[np.int64]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable torch.float64 moment matching returning (F_W, v_W, delta_weights)."""
    squeeze_batch = f_perf.ndim == 1
    f_in = f_perf if f_perf.dtype == torch.float64 else f_perf.to(dtype=torch.float64)
    v_in = v_perf if v_perf.dtype == torch.float64 else v_perf.to(dtype=torch.float64)
    c_in = corr if corr.dtype == torch.float64 else corr.to(dtype=torch.float64)

    if squeeze_batch:
        f_in = f_in.unsqueeze(0)
        v_in = v_in.unsqueeze(0)
        c_in = c_in.unsqueeze(0)
        mask_in = (
            None
            if asset_mask is None
            else asset_mask.to(dtype=torch.float64, device=f_in.device).unsqueeze(0)
        )
    else:
        mask_in = (
            None
            if asset_mask is None
            else asset_mask.to(dtype=torch.float64, device=f_in.device)
        )

    b_size, d_max = f_in.shape
    device = f_in.device
    code = monitoring_index_to_code(monitoring_index)

    if b_size == 0 or d_max == 0:
        f_z = torch.zeros(b_size, dtype=torch.float64, device=device)
        v_z = torch.full((b_size,), 1e-12, dtype=torch.float64, device=device)
        w_z = torch.zeros((b_size, d_max), dtype=torch.float64, device=device)
        if squeeze_batch and b_size > 0:
            return f_z.squeeze(0), v_z.squeeze(0), w_z.squeeze(0)
        return f_z, v_z, w_z

    # Clean inactive slots upfront so NaN/Inf in dummy assets never propagate
    if mask_in is not None:
        active_b = mask_in > 0.5
        f_in = torch.where(active_b, f_in, torch.ones_like(f_in))
        v_in = torch.where(active_b, v_in, torch.full_like(v_in, 0.1))
        pair_active = active_b.unsqueeze(2) & active_b.unsqueeze(1)
        eye_d = torch.eye(d_max, dtype=torch.float64, device=device).unsqueeze(0)
        c_in = torch.where(pair_active, c_in, eye_d)

    if d_max == 1:
        if 9 <= code <= 13:
            f_0 = torch.zeros(b_size, dtype=torch.float64, device=device)
            v_0 = torch.full((b_size,), 1e-12, dtype=torch.float64, device=device)
            w_0 = torch.zeros_like(f_in)
        else:
            f_0 = f_in[:, 0].clamp(min=1e-16)
            v_0 = v_in[:, 0].clamp(min=1e-12)
            w_0 = torch.ones_like(f_in) if mask_in is None else mask_in
        if squeeze_batch:
            return f_0.squeeze(0), v_0.squeeze(0), w_0.squeeze(0)
        return f_0, v_0, w_0

    def _ab_indices() -> Tuple[int, int]:
        if inv_lex_perm is not None and len(inv_lex_perm) >= 2:
            return int(inv_lex_perm[0]), int(inv_lex_perm[1])
        return 0, 1

    if code in (0, 14):
        f_w, v_w, weights = _clark_symmetrized_torch(f_in, v_in, c_in, mask_in, False)
    elif code == 1:
        f_w, v_w, weights = _clark_symmetrized_torch(f_in, v_in, c_in, mask_in, True)
    elif code == 2:
        if mask_in is None:
            inv_d = 1.0 / float(d_max)
            f_w = f_in.sum(dim=1) * inv_d
            cov_mat = c_in * v_in.unsqueeze(2) * v_in.unsqueeze(1)
            f_outer = f_in.unsqueeze(2) * f_in.unsqueeze(1)
            m2 = (f_outer * torch.exp(cov_mat)).sum(dim=(1, 2)) * (inv_d * inv_d)
            v_w = torch.sqrt(
                torch.clamp(torch.log(torch.clamp(m2 / (f_w * f_w).clamp(min=1e-30), min=1.0)), min=1e-16)
            )
            weights = torch.full_like(f_in, inv_d)
        else:
            active_cnt = mask_in.sum(dim=1).clamp(min=1.0)
            f_masked = f_in * mask_in
            f_w = f_masked.sum(dim=1) / active_cnt
            pair_mask = mask_in.unsqueeze(2) * mask_in.unsqueeze(1)
            cov_mat = c_in * v_in.unsqueeze(2) * v_in.unsqueeze(1)
            f_outer = f_masked.unsqueeze(2) * f_masked.unsqueeze(1)
            m2 = (f_outer * torch.exp(cov_mat) * pair_mask).sum(dim=(1, 2)) / (active_cnt * active_cnt)
            v_w = torch.sqrt(
                torch.clamp(torch.log(torch.clamp(m2 / (f_w * f_w).clamp(min=1e-30), min=1.0)), min=1e-16)
            )
            weights = mask_in / active_cnt.unsqueeze(1)
    elif code == 3:
        m_w = torch.ones_like(f_in) if mask_in is None else mask_in
        active_cnt = m_w.sum(dim=1).clamp(min=1.0)
        f_masked = f_in * m_w
        f_mean = f_masked.sum(dim=1) / active_cnt
        m2_p = ((f_masked * f_masked) * torch.exp(v_in * v_in)).sum(dim=1) / active_cnt
        pair_mask = m_w.unsqueeze(2) * m_w.unsqueeze(1)
        cov_mat = c_in * v_in.unsqueeze(2) * v_in.unsqueeze(1)
        m2_avg = ((f_masked.unsqueeze(2) * f_masked.unsqueeze(1)) * torch.exp(cov_mat) * pair_mask).sum(
            dim=(1, 2)
        ) / (active_cnt * active_cnt)
        ew2 = (m2_p - m2_avg).clamp(min=1e-12)
        var_f = (((f_in - f_mean.unsqueeze(1)) ** 2) * m_w).sum(dim=1) / active_cnt
        sig_approx = torch.sqrt(ew2)
        use_var = var_f > 1e-10
        var_f_safe = var_f.clamp(min=1e-10)
        f_w = torch.where(use_var, torch.sqrt(var_f_safe), sig_approx)
        v_w = torch.where(
            use_var,
            torch.sqrt(torch.clamp(torch.log(torch.clamp(ew2 / var_f_safe, min=1.0)), min=1e-16)),
            torch.full_like(ew2, 0.35),
        )
        weights = (m_w / active_cnt.unsqueeze(1)) * (f_in - f_mean.unsqueeze(1)) / sig_approx.clamp(min=1e-8).unsqueeze(1)
    elif code in (4, 5):
        ia, ib = _ab_indices()
        if code == 5:
            ia, ib = ib, ia
        fa = f_in[:, ia].clamp(min=1e-16)
        fb = f_in[:, ib].clamp(min=1e-16)
        va = v_in[:, ia].clamp(min=1e-12)
        vb = v_in[:, ib].clamp(min=1e-12)
        rab = c_in[:, ia, ib]
        var_r = (va * va + vb * vb - 2.0 * rab * va * vb).clamp(min=1e-16)
        f_w = (fa / fb) * torch.exp(vb * vb - rab * va * vb)
        v_w = torch.sqrt(var_r)
        weights = torch.zeros_like(f_in)
        weights[:, ia] = 1.0
        weights[:, ib] = -1.0
    elif code in (6, 7, 8):
        if code == 6:
            f_num, v_num, w_num = _clark_symmetrized_torch(f_in, v_in, c_in, mask_in, True)
            f_den, v_den, w_den = _clark_symmetrized_torch(f_in, v_in, c_in, mask_in, False)
            rho_nd = _clark_corr_min_max_torch(f_in, v_in, c_in)
        else:
            if code == 7:
                f_num, v_num, w_num = _clark_symmetrized_torch(f_in, v_in, c_in, mask_in, True)
                f_den, v_den, w_den = stulz_margrabe_moments_with_weights(
                    f_in, v_in, c_in, asset_mask=mask_in, monitoring_index="AverageOf", inv_lex_perm=inv_lex_perm
                )
            else:
                f_num, v_num, w_num = stulz_margrabe_moments_with_weights(
                    f_in, v_in, c_in, asset_mask=mask_in, monitoring_index="AverageOf", inv_lex_perm=inv_lex_perm
                )
                f_den, v_den, w_den = _clark_symmetrized_torch(f_in, v_in, c_in, mask_in, False)
            cij = c_in * v_in.unsqueeze(2) * v_in.unsqueeze(1)
            cov = (w_num.unsqueeze(2) * w_den.unsqueeze(1) * cij).sum(dim=(1, 2))
            vn2 = (w_num.unsqueeze(2) * w_num.unsqueeze(1) * cij).sum(dim=(1, 2))
            vd2 = (w_den.unsqueeze(2) * w_den.unsqueeze(1) * cij).sum(dim=(1, 2))
            rho_nd = (cov / torch.sqrt((vn2 * vd2).clamp(min=1e-30))).clamp(-0.9999, 0.9999)
        var_r = (v_num * v_num + v_den * v_den - 2.0 * rho_nd * v_num * v_den).clamp(min=1e-16)
        f_w = (f_num / f_den.clamp(min=1e-16)) * torch.exp(v_den * v_den - rho_nd * v_num * v_den)
        v_w = torch.sqrt(var_r)
        weights = w_num - w_den
    elif code in (9, 10):
        ia, ib = _ab_indices()
        if code == 10:
            ia, ib = ib, ia
        fa = f_in[:, ia]
        fb = f_in[:, ib]
        va = v_in[:, ia]
        vb = v_in[:, ib]
        rab = c_in[:, ia, ib]
        f_w = fa - fb
        var_d = (
            fa * fa * torch.expm1(va * va)
            + fb * fb * torch.expm1(vb * vb)
            - 2.0 * fa * fb * torch.expm1(rab * va * vb)
        ).clamp(min=1e-16)
        v_w = torch.sqrt(var_d)
        weights = torch.zeros_like(f_in)
        weights[:, ia] = 1.0
        weights[:, ib] = -1.0
    elif code in (11, 12, 13):
        if code == 11:
            f_a, v_a, w_a = _clark_symmetrized_torch(f_in, v_in, c_in, mask_in, True)
            f_b, v_b, w_b = _clark_symmetrized_torch(f_in, v_in, c_in, mask_in, False)
            rho_ab = _clark_corr_min_max_torch(f_in, v_in, c_in)
        else:
            if code == 12:
                f_a, v_a, w_a = _clark_symmetrized_torch(f_in, v_in, c_in, mask_in, True)
                f_b, v_b, w_b = stulz_margrabe_moments_with_weights(
                    f_in, v_in, c_in, asset_mask=mask_in, monitoring_index="AverageOf", inv_lex_perm=inv_lex_perm
                )
            else:
                f_a, v_a, w_a = stulz_margrabe_moments_with_weights(
                    f_in, v_in, c_in, asset_mask=mask_in, monitoring_index="AverageOf", inv_lex_perm=inv_lex_perm
                )
                f_b, v_b, w_b = _clark_symmetrized_torch(f_in, v_in, c_in, mask_in, False)
            cij = c_in * v_in.unsqueeze(2) * v_in.unsqueeze(1)
            cov = (w_a.unsqueeze(2) * w_b.unsqueeze(1) * cij).sum(dim=(1, 2))
            va2 = (w_a.unsqueeze(2) * w_a.unsqueeze(1) * cij).sum(dim=(1, 2))
            vb2 = (w_b.unsqueeze(2) * w_b.unsqueeze(1) * cij).sum(dim=(1, 2))
            rho_ab = (cov / torch.sqrt((va2 * vb2).clamp(min=1e-30))).clamp(-0.9999, 0.9999)
        f_w = f_a - f_b
        var_d = (
            f_a * f_a * torch.expm1(v_a * v_a)
            + f_b * f_b * torch.expm1(v_b * v_b)
            - 2.0 * f_a * f_b * torch.expm1(rho_ab * v_a * v_b)
        ).clamp(min=1e-16)
        v_w = torch.sqrt(var_d)
        weights = w_a - w_b
    else:
        # Order statistics: SecondWorstOf (15), ThirdWorstOf (16), SecondBestOf (17)
        if d_max == 2 or (d_max == 3 and code == 16):
            f_w, v_w, weights = _clark_symmetrized_torch(f_in, v_in, c_in, mask_in, is_best_of=(code != 17))
        else:
            b_vec = _fit_1factor_loadings_torch(c_in)
            f_s = f_in.clamp(min=1e-16)
            v_s = v_in.clamp(min=1e-12)
            mu_bd = torch.log(f_s) - 0.5 * v_s * v_s
            vb_bd = v_s * b_vec
            vs_bd = (v_s * torch.sqrt((1.0 - b_vec * b_vec).clamp(min=1e-12))).clamp(min=1e-12)
            y_g = torch.tensor(_GL16_YNORM, dtype=torch.float64, device=device).view(1, 16, 1)
            z_g = torch.tensor(_GL16_YNORM, dtype=torch.float64, device=device).view(1, 1, 16)
            w_yz = (
                0.25
                * torch.tensor(_GL16_W, dtype=torch.float64, device=device).view(1, 16, 1)
                * torch.tensor(_GL16_W, dtype=torch.float64, device=device).view(1, 1, 16)
            )
            # SecondBestOf asks for 1 larger neighbor; SecondWorstOf/ThirdWorstOf ask for 1/2 smaller.
            count_above = code == 17
            target_cnt = 2 if code == 16 else 1
            f_sum = torch.zeros(b_size, dtype=torch.float64, device=device)
            m2_sum = torch.zeros(b_size, dtype=torch.float64, device=device)
            w_list: List[torch.Tensor] = []
            for i in range(d_max):
                vi = v_s[:, i].view(b_size, 1, 1)
                vbi = vb_bd[:, i].view(b_size, 1, 1)
                vsi = vs_bd[:, i].view(b_size, 1, 1)
                mui = mu_bd[:, i].view(b_size, 1, 1)
                lx = mui + vbi * y_g + vsi * z_g
                unit_p = torch.exp(-0.5 * vi * vi + vbi * y_g + vsi * z_g)
                p_val = f_s[:, i].view(b_size, 1, 1) * unit_p
                dp: List[torch.Tensor] = [torch.ones((b_size, 16, 16), dtype=torch.float64, device=device)]
                for _ in range(target_cnt):
                    dp.append(torch.zeros((b_size, 16, 16), dtype=torch.float64, device=device))
                for j in range(d_max):
                    if j == i:
                        continue
                    z_j = (lx - mu_bd[:, j].view(b_size, 1, 1) - vb_bd[:, j].view(b_size, 1, 1) * y_g) / vs_bd[:, j].view(
                        b_size, 1, 1
                    )
                    prob = torch.special.ndtr(-z_j if count_above else z_j)
                    if mask_in is not None:
                        prob = prob * mask_in[:, j].view(b_size, 1, 1)
                    qrob = 1.0 - prob
                    next_dp: List[torch.Tensor] = [dp[0] * qrob]
                    for r in range(1, target_cnt + 1):
                        next_dp.append(dp[r] * qrob + dp[r - 1] * prob)
                    dp = next_dp
                pr = w_yz * dp[target_cnt]
                if mask_in is not None:
                    pr = pr * mask_in[:, i].view(b_size, 1, 1)
                w_list.append((pr * unit_p).sum(dim=(1, 2)))
                f_sum = f_sum + (pr * p_val).sum(dim=(1, 2))
                m2_sum = m2_sum + (pr * p_val * p_val).sum(dim=(1, 2))
            f_w = f_sum.clamp(min=1e-16)
            v_w = torch.sqrt(
                torch.clamp(torch.log(torch.clamp(m2_sum / (f_w * f_w).clamp(min=1e-30), min=1.0)), min=1e-16)
            )
            weights = torch.stack(w_list, dim=1)

    if squeeze_batch:
        return f_w.squeeze(0), v_w.squeeze(0), weights.squeeze(0)
    return f_w, v_w, weights


def stulz_margrabe_worst_of_moments(
    f_perf: torch.Tensor,
    v_perf: torch.Tensor,
    corr: torch.Tensor,
    asset_mask: torch.Tensor | None = None,
    monitoring_index: str = "WorstOf",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normative entrypoint returning effective moments (F_W, v_W) in torch.float64."""
    f_w, v_w, _ = stulz_margrabe_moments_with_weights(
        f_perf=f_perf,
        v_perf=v_perf,
        corr=corr,
        asset_mask=asset_mask,
        monitoring_index=monitoring_index,
    )
    return f_w, v_w


def _black76_call_torch(f: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Undiscounted Black-76 call price E[max(W - K, 0)] for W ~ LogNormal(f, v^2) in torch.float64."""
    f_c = f.clamp(min=1e-16)
    k_c = k.clamp(min=1e-16)
    v_c = v.clamp(min=1e-12)
    d1 = (torch.log(f_c / k_c) + 0.5 * v_c * v_c) / v_c
    d2 = d1 - v_c
    res: torch.Tensor = f_c * torch.special.ndtr(d1) - k_c * torch.special.ndtr(d2)
    return res


def _black76_put_torch(f: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Undiscounted Black-76 put price E[max(K - W, 0)] for W ~ LogNormal(f, v^2) in torch.float64."""
    f_c = f.clamp(min=1e-16)
    k_c = k.clamp(min=1e-16)
    v_c = v.clamp(min=1e-12)
    d1 = (torch.log(f_c / k_c) + 0.5 * v_c * v_c) / v_c
    d2 = d1 - v_c
    res: torch.Tensor = k_c * torch.special.ndtr(-d2) - f_c * torch.special.ndtr(-d1)
    return res


def _normal_call_torch(mu: torch.Tensor, sig: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    s_c = sig.clamp(min=1e-12)
    z = (mu - k) / s_c
    phi_z = torch.exp(-0.5 * z * z) * _INV_SQRT2PI
    return cast(torch.Tensor, (mu - k) * torch.special.ndtr(z) + s_c * phi_z)


def _digital_call_spread_prob_torch(
    f: torch.Tensor,
    v: torch.Tensor,
    barrier: torch.Tensor,
    spread: float | torch.Tensor,
) -> torch.Tensor:
    """NaN-safe exact lognormal expectation of a call-spread digital ramp in torch.float64."""
    f_c = f.clamp(min=1e-16)
    v_c = v.clamp(min=1e-12)
    b_c = barrier.clamp(min=1e-8)

    if not isinstance(spread, torch.Tensor) and abs(float(spread)) <= 1e-6:
        d2_pt = (torch.log(f_c / b_c) - 0.5 * v_c * v_c) / v_c
        return cast(torch.Tensor, torch.special.ndtr(d2_pt).clamp(0.0, 1.0))

    s_t = (
        spread.to(dtype=torch.float64, device=f.device)
        if isinstance(spread, torch.Tensor)
        else torch.tensor(float(spread), dtype=torch.float64, device=f.device)
    )
    abs_s = torch.abs(s_t)
    is_spread = abs_s > 1e-6

    b_low_raw = torch.minimum(b_c, b_c + s_t)
    b_high_raw = torch.maximum(b_c, b_c + s_t)

    # Guard b_high_safe and positivity BEFORE Black-76 evaluation and division (zero 0/0 NaNs)
    b_low_pos = torch.where(b_low_raw > 0.0, b_low_raw, torch.ones_like(b_low_raw))
    c_low_call = _black76_call_torch(f_c, b_low_pos, v_c)
    c_low = torch.where(b_low_raw > 0.0, c_low_call, f_c - b_low_raw)

    b_high_pos = torch.where(b_high_raw > 0.0, b_high_raw, torch.ones_like(b_high_raw))
    b_high_safe = torch.where(is_spread & (b_high_raw > 0.0), b_high_pos, b_low_pos + 1.0)
    c_high_call = _black76_call_torch(f_c, b_high_safe, v_c)
    c_high = torch.where(b_high_raw > 0.0, c_high_call, f_c - b_high_raw)

    denom_safe = torch.where(is_spread & (b_high_raw > 0.0), b_high_raw - b_low_raw, torch.ones_like(b_low_raw))
    prob_spread = torch.where(
        b_high_raw <= 0.0,
        torch.ones_like(f_c),
        ((c_low - c_high) / denom_safe).clamp(0.0, 1.0),
    )

    d2_pt = (torch.log(f_c / b_c) - 0.5 * v_c * v_c) / v_c
    prob_pt = torch.special.ndtr(d2_pt).clamp(0.0, 1.0)
    return torch.where(is_spread, prob_spread, prob_pt).clamp(0.0, 1.0)


def _digital_call_spread_normal_torch(
    mu: torch.Tensor,
    sig: torch.Tensor,
    barrier: torch.Tensor,
    spread: float | torch.Tensor,
) -> torch.Tensor:
    s_c = sig.clamp(min=1e-12)
    if not isinstance(spread, torch.Tensor) and abs(float(spread)) <= 1e-6:
        return cast(torch.Tensor, torch.special.ndtr((mu - barrier) / s_c).clamp(0.0, 1.0))
    s_t = (
        spread.to(dtype=torch.float64, device=mu.device)
        if isinstance(spread, torch.Tensor)
        else torch.tensor(float(spread), dtype=torch.float64, device=mu.device)
    )
    is_spread = torch.abs(s_t) > 1e-6
    b_low = torch.minimum(barrier, barrier + s_t)
    b_high = torch.maximum(barrier, barrier + s_t)
    b_high_safe = torch.where(is_spread, b_high, b_low + 1.0)
    c_low = _normal_call_torch(mu, s_c, b_low)
    c_high = _normal_call_torch(mu, s_c, b_high_safe)
    denom_safe = torch.where(is_spread, b_high_safe - b_low, torch.ones_like(b_low))
    prob_spread = ((c_low - c_high) / denom_safe).clamp(0.0, 1.0)
    prob_pt = torch.special.ndtr((mu - barrier) / s_c).clamp(0.0, 1.0)
    return torch.where(is_spread, prob_spread, prob_pt).clamp(0.0, 1.0)


def _eval_coupon_prob_torch(
    f: torch.Tensor,
    v: torch.Tensor,
    barrier: torch.Tensor,
    spread: float | torch.Tensor,
    direction: str = "Up",
    b_type: str = "In",
    *,
    is_normal: bool = False,
    index_code: int = 0,
) -> torch.Tensor:
    dir_down = direction.strip().lower() == "down"
    type_out = b_type.strip().lower() == "out"
    eff_spread = -spread if dir_down else spread
    if index_code == 14:
        b_lo = torch.minimum(barrier, barrier + eff_spread)
        b_hi = torch.maximum(barrier, barrier + eff_spread)
        db = b_hi - b_lo
        f_c = f.clamp(min=1e-16)
        v_c = v.clamp(min=1e-12)
        a_dn = (2.0 - b_hi).clamp(min=0.0)
        b_dn = (2.0 - b_lo).clamp(0.0, 1.0)
        a_up = b_lo.clamp(min=1.0)
        if abs(float(eff_spread) if not isinstance(eff_spread, torch.Tensor) else 1.0) <= 1e-6:
            p_surv = _lognormal_moments_interval_torch(f_c, v_c, a_dn, b_hi, 0)[0]
        else:
            inv_db = 1.0 / db.clamp(min=1e-12)
            m_dn = _lognormal_moments_interval_torch(f_c, v_c, a_dn, b_dn, 1)
            m_mid = _lognormal_moments_interval_torch(f_c, v_c, b_dn, a_up, 0)
            m_up = _lognormal_moments_interval_torch(f_c, v_c, a_up, b_hi, 1)
            p_ramp = (
                (-(2.0 - b_hi) * m_dn[0] + m_dn[1]) * inv_db
                + m_mid[0]
                + (b_hi * m_up[0] - m_up[1]) * inv_db
            )
            p_hard = _lognormal_moments_interval_torch(f_c, v_c, a_dn, b_hi, 0)[0]
            p_surv = torch.where(db <= 1e-6, p_hard, p_ramp)
        p_ge = torch.where(b_hi <= 1.0, torch.ones_like(f_c), (1.0 - p_surv).clamp(0.0, 1.0))
        p_ev = (1.0 - p_ge) if dir_down else p_ge
        return (1.0 - p_ev) if type_out else p_ev
    p_up = (
        _digital_call_spread_normal_torch(f, v, barrier, eff_spread)
        if is_normal
        else _digital_call_spread_prob_torch(f, v, barrier, eff_spread)
    )
    p_ev = (1.0 - p_up) if dir_down else p_up
    return (1.0 - p_ev) if type_out else p_ev


def _ramp_survival_post_moments_lognormal_torch(
    f_cond: torch.Tensor,
    lam: torch.Tensor,
    b_ko: torch.Tensor,
    ko_spread: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact closed-form truncated-lognormal posterior (mu_u, var_u) under KO ramp survival."""
    f_c = f_cond.clamp(min=1e-16)
    l_c = lam.clamp(min=1e-12)
    b_c = b_ko.clamp(min=1e-8)
    m_u = torch.log(f_c) - 0.5 * l_c * l_c

    if abs(ko_spread) <= 1e-6:
        u_star = (torch.log(b_c) - m_u) / l_c
        return _truncated_upper_std_moments_torch(u_star)

    b_lo = torch.minimum(b_c, b_c + ko_spread)
    b_hi = torch.maximum(b_c, b_c + ko_spread)
    b_hi_safe = b_hi.clamp(min=1e-16)
    db = (b_hi_safe - b_lo).clamp(min=1e-12)
    u_hi = (torch.log(b_hi_safe) - m_u) / l_c
    has_lo = b_lo > 0.0
    b_lo_pos = torch.where(has_lo, b_lo, torch.ones_like(b_lo))
    u_lo = torch.where(has_lo, (torch.log(b_lo_pos) - m_u) / l_c, torch.full_like(u_hi, -100.0))

    use_tail = u_hi <= -20.0
    use_scaled = (u_hi < -4.0) & (~use_tail)

    f_cond_d = torch.where(u_hi < -4.0, b_hi_safe, f_c)
    u_hi_d = torch.where(u_hi < -4.0, torch.zeros_like(u_hi), u_hi)
    u_lo_d = torch.where((u_hi < -4.0) | (~has_lo), torch.full_like(u_hi, -100.0), u_lo)
    Phi_hi = torch.special.ndtr(u_hi_d)
    Phi_hi_m = torch.special.ndtr(u_hi_d - l_c)
    phi_hi = torch.exp(-0.5 * u_hi_d * u_hi_d) * _INV_SQRT2PI
    Phi_lo = torch.where(has_lo, torch.special.ndtr(u_lo_d), torch.zeros_like(u_hi))
    Phi_lo_m = torch.where(has_lo, torch.special.ndtr(u_lo_d - l_c), torch.zeros_like(u_hi))
    phi_lo = torch.where(has_lo, torch.exp(-0.5 * u_lo_d * u_lo_d) * _INV_SQRT2PI, torch.zeros_like(u_hi))

    M0_d = (b_hi_safe * Phi_hi - b_lo * Phi_lo - f_cond_d * (Phi_hi_m - Phi_lo_m)) / db
    M1_d = -(l_c * f_cond_d / db) * (Phi_hi_m - Phi_lo_m)
    M2_d = M0_d + l_c * M1_d + (l_c / db) * (b_hi_safe * phi_hi - b_lo * phi_lo)
    d_tiny = M0_d <= 1e-300
    M0_d_safe = torch.where((u_hi < -4.0) | d_tiny, torch.ones_like(M0_d), M0_d)
    mu_d = M1_d / M0_d_safe
    var_d = (M2_d / M0_d_safe - mu_d * mu_d).clamp(min=1e-12)

    u_hi_s = torch.where(use_scaled, u_hi, torch.full_like(u_hi, -4.0))
    u_lo_s = torch.where(use_scaled & has_lo, u_lo, torch.full_like(u_hi, -1e9))
    exp_lo = torch.where(has_lo, torch.exp((-0.5 * (u_lo_s - u_hi_s) * (u_lo_s + u_hi_s)).clamp(max=0.0)), torch.zeros_like(u_hi))
    r_hi = _SQRT_HALF_PI * torch.special.erfcx(-u_hi_s * _INV_SQRT2)
    r_hi_m = _SQRT_HALF_PI * torch.special.erfcx(-(u_hi_s - l_c) * _INV_SQRT2)
    r_lo = torch.where(has_lo, _SQRT_HALF_PI * torch.special.erfcx(-u_lo_s * _INV_SQRT2), torch.zeros_like(u_hi))
    r_lo_m = torch.where(
        has_lo, _SQRT_HALF_PI * torch.special.erfcx(-(u_lo_s - l_c) * _INV_SQRT2), torch.zeros_like(u_hi)
    )

    M0_s = (b_hi_safe * (r_hi - r_hi_m) - b_lo * exp_lo * (r_lo - r_lo_m)) / db
    M1_s = -(l_c / db) * (b_hi_safe * r_hi_m - b_lo * exp_lo * r_lo_m)
    M2_s = M0_s + l_c * M1_s + (l_c / db) * (b_hi_safe - b_lo * exp_lo)
    s_tiny = M0_s <= 1e-300
    M0_s_safe = torch.where((~use_scaled) | s_tiny, torch.ones_like(M0_s), M0_s)
    mu_s = M1_s / M0_s_safe
    var_s = (M2_s / M0_s_safe - mu_s * mu_s).clamp(min=1e-12)

    # Shifted-Y = -(U + x) quadrature eliminates erfcx subtraction in deep left tail (u_hi <= -20).
    x_t = torch.where(use_tail, -u_hi, torch.full_like(u_hi, 20.0))
    y_lo = torch.where(has_lo, (u_hi - u_lo).clamp(min=0.0), torch.full_like(x_t, 1e100))
    y_end = torch.minimum(y_lo, 45.0 / x_t)
    inv_ratio = b_hi_safe / db
    m0_t = torch.zeros_like(x_t)
    n1_t = torch.zeros_like(x_t)
    n2_t = torch.zeros_like(x_t)
    for gx, gw in zip(_GL8_X01, _GL8_W01):
        yq = y_end * gx
        wt = y_end * gw * torch.exp((-yq * (x_t + 0.5 * yq)).clamp(min=-80.0)) * inv_ratio * (-torch.expm1((-l_c * yq).clamp(min=-80.0)))
        m0_t = m0_t + wt
        n1_t = n1_t + wt * yq
        n2_t = n2_t + wt * (yq * yq)
    lag_on = has_lo & (y_lo * (x_t + 0.5 * y_lo) < 55.0)
    y_lo_safe = torch.where(lag_on, y_lo, torch.zeros_like(x_t))
    alpha_l = x_t + y_lo_safe
    inv_alpha = 1.0 / alpha_l
    pref_l = torch.where(lag_on, torch.exp((-y_lo_safe * (x_t + 0.5 * y_lo_safe)).clamp(min=-80.0)) * inv_alpha, torch.zeros_like(x_t))
    for lx, lw in zip(_LAG6_X, _LAG6_W):
        sq = lx * inv_alpha
        yq = y_lo_safe + sq
        wt = pref_l * lw * torch.exp((-0.5 * sq * sq).clamp(min=-80.0))
        m0_t = m0_t + wt
        n1_t = n1_t + wt * yq
        n2_t = n2_t + wt * (yq * yq)
    t_tiny = m0_t <= 1e-300
    m0_t_safe = torch.where((~use_tail) | t_tiny, torch.ones_like(m0_t), m0_t)
    ey_t = n1_t / m0_t_safe
    mu_t = -x_t - ey_t
    var_t = (n2_t / m0_t_safe - ey_t * ey_t).clamp(min=1e-12)

    mu_fb, var_fb = _truncated_upper_std_moments_torch(u_hi)
    mu = torch.where(use_tail, mu_t, torch.where(use_scaled, mu_s, mu_d))
    var = torch.where(use_tail, var_t, torch.where(use_scaled, var_s, var_d))
    tiny_mask = torch.where(use_tail, t_tiny, torch.where(use_scaled, s_tiny, d_tiny))
    mu = torch.where(tiny_mask, mu_fb, mu)
    var = torch.where(tiny_mask, var_fb, var)
    return torch.where(b_hi <= 0.0, torch.zeros_like(mu), mu), torch.where(b_hi <= 0.0, torch.ones_like(var), var)


def _ramp_survival_post_moments_twinwin_lognormal_torch(
    f_cond: torch.Tensor,
    lam: torch.Tensor,
    b_ko: torch.Tensor,
    ko_spread: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable truncated-lognormal posterior (mu_u, var_u) over TwinWin survival band."""
    f_c = f_cond.clamp(min=1e-16)
    l_c = lam.clamp(min=1e-12)
    b_lo = torch.minimum(b_ko, b_ko + ko_spread)
    b_hi = torch.maximum(b_ko, b_ko + ko_spread)
    m_u = torch.log(f_c) - 0.5 * l_c * l_c
    db = b_hi - b_lo
    z0 = torch.zeros_like(l_c)
    o1 = torch.ones_like(l_c)

    def _add_seg(seg_a: torch.Tensor, seg_b: torch.Tensor, c0: float | torch.Tensor, c1: float | torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = seg_b > seg_a + 1e-14
        a_pos = seg_a > 0.0
        a_s = torch.where(valid & a_pos, seg_a, torch.ones_like(f_c))
        b_s = torch.where(valid, seg_b.clamp(min=1e-16), torch.ones_like(f_c))
        u_b = (torch.log(b_s) - m_u) / l_c
        u_a = torch.where(a_pos, (torch.log(a_s) - m_u) / l_c, torch.full_like(f_c, -1e60))
        u_b_e = torch.where(valid, u_b, torch.zeros_like(f_c))
        u_a_e = torch.where(valid, u_a, torch.full_like(f_c, -1.0))
        jc = _normal_moments_interval_torch(z0, o1, u_a_e, u_b_e, 2)
        jl = _normal_moments_interval_torch(l_c, o1, u_a_e, u_b_e, 2)
        w1 = c1 * f_c
        s0 = torch.where(valid, c0 * jc[0] + w1 * jl[0], z0)
        s1 = torch.where(valid, c0 * jc[1] + w1 * jl[1], z0)
        s2 = torch.where(valid, c0 * jc[2] + w1 * jl[2], z0)
        return s0, s1, s2

    a_dn = (2.0 - b_hi).clamp(min=0.0)
    if abs(ko_spread) <= 1e-6:
        m0, m1, m2 = _add_seg(a_dn, b_hi, 1.0, 0.0)
    else:
        inv_db = 1.0 / db.clamp(min=1e-12)
        b_dn = (2.0 - b_lo).clamp(0.0, 1.0)
        a_up = b_lo.clamp(min=1.0)
        s0a, s1a, s2a = _add_seg(a_dn, b_dn, -(2.0 - b_hi) * inv_db, inv_db)
        s0b, s1b, s2b = _add_seg(b_dn, a_up, 1.0, 0.0)
        s0c, s1c, s2c = _add_seg(a_up, b_hi, b_hi * inv_db, -inv_db)
        m0_r = s0a + s0b + s0c
        m1_r = s1a + s1b + s1c
        m2_r = s2a + s2b + s2c
        m0_h, m1_h, m2_h = _add_seg(a_dn, b_hi, 1.0, 0.0)
        hard_m = db <= 1e-6
        m0 = torch.where(hard_m, m0_h, m0_r)
        m1 = torch.where(hard_m, m1_h, m1_r)
        m2 = torch.where(hard_m, m2_h, m2_r)

    tiny = m0 <= 1e-300
    m0_safe = torch.where(tiny, o1, m0)
    mu_raw = m1 / m0_safe
    var_raw = (m2 / m0_safe - mu_raw * mu_raw).clamp(min=1e-12)

    u_top = (torch.log(b_hi.clamp(min=1e-16)) - m_u) / l_c
    mu_top, var_top = _truncated_upper_std_moments_torch(u_top)
    u_bot = -(torch.log(a_dn.clamp(min=1e-16)) - m_u) / l_c
    mu_neg, var_bot = _truncated_upper_std_moments_torch(u_bot)
    mu_fb = torch.where(m_u > torch.log(b_hi.clamp(min=1e-16)), mu_top, torch.where(a_dn > 0.0, -mu_neg, z0))
    var_fb = torch.where(m_u > torch.log(b_hi.clamp(min=1e-16)), var_top, torch.where(a_dn > 0.0, var_bot, o1))

    mu = torch.where(tiny, mu_fb, mu_raw)
    var = torch.where(tiny, var_fb, var_raw)
    no_band = b_hi <= 1.0
    return torch.where(no_band, z0, mu), torch.where(no_band, torch.full_like(var, 1e-12), var)


def _shifted_y_jk_torch(k: int, t: torch.Tensor) -> torch.Tensor:
    inv_t = 1.0 / t
    inv_t2 = inv_t * inv_t
    if k == 0:
        coeffs = (1.0, -1.0, 3.0, -15.0, 105.0, -945.0, 10395.0)
    elif k == 1:
        coeffs = (1.0, -3.0, 15.0, -105.0, 945.0, -10395.0, 135135.0)
    elif k == 2:
        coeffs = (2.0, -12.0, 90.0, -840.0, 9450.0, -124740.0, 1891890.0)
    else:
        coeffs = (6.0, -60.0, 630.0, -7560.0, 103950.0, -1621620.0, 28378350.0)
    acc = torch.full_like(t, coeffs[6])
    for n in range(5, -1, -1):
        acc = acc * inv_t2 + coeffs[n]
    return cast(torch.Tensor, acc * (inv_t ** (k + 1)))


def _ramp_survival_post_moments_normal_torch(
    mu_cond: torch.Tensor,
    sig_cond: torch.Tensor,
    b_ko: torch.Tensor,
    ko_spread: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_c = sig_cond.clamp(min=1e-12)
    if abs(ko_spread) <= 1e-6:
        return _truncated_upper_std_moments_torch((b_ko - mu_cond) / s_c)
    b_lo = torch.minimum(b_ko, b_ko + ko_spread)
    b_hi = torch.maximum(b_ko, b_ko + ko_spread)
    u_lo = (b_lo - mu_cond) / s_c
    u_hi = (b_hi - mu_cond) / s_c
    du = (u_hi - u_lo).clamp(min=1e-12)
    use_tail = u_hi <= -20.0
    use_scaled = (u_hi < -4.0) & (~use_tail)

    u_hi_d = torch.where(u_hi < -4.0, torch.zeros_like(u_hi), u_hi)
    u_lo_d = torch.where(u_hi < -4.0, torch.full_like(u_hi, -1.0), u_lo)
    phi_hi = torch.exp(-0.5 * u_hi_d * u_hi_d) * _INV_SQRT2PI
    phi_lo = torch.exp(-0.5 * u_lo_d * u_lo_d) * _INV_SQRT2PI
    Phi_hi = torch.special.ndtr(u_hi_d)
    Phi_lo = torch.special.ndtr(u_lo_d)
    M0_d = ((u_hi_d * Phi_hi + phi_hi) - (u_lo_d * Phi_lo + phi_lo)) / du
    M1_d = -(Phi_hi - Phi_lo) / du
    M2_d = M0_d + (phi_hi - phi_lo) / du
    d_tiny = M0_d <= 1e-300
    M0_d_safe = torch.where((u_hi < -4.0) | d_tiny, torch.ones_like(M0_d), M0_d)
    mu_d = M1_d / M0_d_safe
    var_d = (M2_d / M0_d_safe - mu_d * mu_d).clamp(min=1e-12)

    u_hi_s = torch.where(use_scaled, u_hi, torch.full_like(u_hi, -4.0))
    u_lo_s = torch.where(use_scaled, u_lo, torch.full_like(u_hi, -5.0))
    exp_lo = torch.exp(-0.5 * (u_lo_s - u_hi_s) * (u_lo_s + u_hi_s))
    r_hi = _SQRT_HALF_PI * torch.special.erfcx(-u_hi_s * _INV_SQRT2)
    r_lo = _SQRT_HALF_PI * torch.special.erfcx(-u_lo_s * _INV_SQRT2)
    M0_s = ((u_hi_s * r_hi + 1.0) - exp_lo * (u_lo_s * r_lo + 1.0)) / du
    M1_s = -(r_hi - exp_lo * r_lo) / du
    M2_s = M0_s + (1.0 - exp_lo) / du
    s_tiny = M0_s <= 1e-300
    M0_s_safe = torch.where((~use_scaled) | s_tiny, torch.ones_like(M0_s), M0_s)
    mu_s = M1_s / M0_s_safe
    var_s = (M2_s / M0_s_safe - mu_s * mu_s).clamp(min=1e-12)

    # Shifted-Y = -(U + x) asymptotic integrals eliminate erfcx subtraction for x >= 20.
    x_safe = torch.where(use_tail, -u_hi, torch.full_like(u_hi, 20.0))
    eta = du * (x_safe + 0.5 * du)
    use_eta_big = eta >= 1e-3

    t0 = x_safe
    t1 = x_safe + du
    e_eta = torch.exp(-eta)
    j1_0 = _shifted_y_jk_torch(1, t0)
    j2_0 = _shifted_y_jk_torch(2, t0)
    j3_0 = _shifted_y_jk_torch(3, t0)
    j1_1 = _shifted_y_jk_torch(1, t1)
    j2_1 = _shifted_y_jk_torch(2, t1)
    j3_1 = _shifted_y_jk_torch(3, t1)
    M0_tb = (j1_0 - e_eta * j1_1) / du
    N1_tb = (j2_0 - e_eta * (j2_1 + du * j1_1)) / du
    N2_tb = (j3_0 - e_eta * (j3_1 + 2.0 * du * j2_1 + du * du * j1_1)) / du

    # 3-point Gauss-Legendre on [0, du] keeps small-eta (eta < 1e-3) derivatives smooth and subtraction-free.
    gl_nodes = (0.11270166537925831148, 0.5, 0.88729833462074168852)
    gl_weights = (0.27777777777777777778, 0.44444444444444444444, 0.27777777777777777778)
    M0_ts = torch.zeros_like(x_safe)
    N1_ts = torch.zeros_like(x_safe)
    N2_ts = torch.zeros_like(x_safe)
    for gx, gw in zip(gl_nodes, gl_weights):
        s_q = du * gx
        t_q = x_safe + s_q
        we = gw * torch.exp(-(x_safe * s_q + 0.5 * s_q * s_q))
        j0_q = _shifted_y_jk_torch(0, t_q)
        j1_q = _shifted_y_jk_torch(1, t_q)
        j2_q = _shifted_y_jk_torch(2, t_q)
        M0_ts = M0_ts + we * j0_q
        N1_ts = N1_ts + we * (j1_q + s_q * j0_q)
        N2_ts = N2_ts + we * (j2_q + 2.0 * s_q * j1_q + (s_q * s_q) * j0_q)

    M0_t = torch.where(use_eta_big, M0_tb, M0_ts)
    N1_t = torch.where(use_eta_big, N1_tb, N1_ts)
    N2_t = torch.where(use_eta_big, N2_tb, N2_ts)
    t_tiny = M0_t <= 1e-300
    M0_t_safe = torch.where((~use_tail) | t_tiny, torch.ones_like(M0_t), M0_t)
    ey_t = N1_t / M0_t_safe
    mu_t_raw = -x_safe - ey_t
    var_t_raw = (N2_t / M0_t_safe - ey_t * ey_t).clamp(min=1e-12)

    mu_sharp, var_sharp = _truncated_upper_std_moments_torch(u_hi)
    mu_t = torch.where(t_tiny, mu_sharp, mu_t_raw)
    var_t = torch.where(t_tiny, var_sharp, var_t_raw)

    mu_res = torch.where(use_tail, mu_t, torch.where(use_scaled, mu_s, mu_d))
    var_res = torch.where(use_tail, var_t, torch.where(use_scaled, var_s, var_d))
    return mu_res, var_res


def _lognormal_moments_interval_torch(
    f: torch.Tensor,
    v: torch.Tensor,
    a: float | torch.Tensor,
    b: float | torch.Tensor,
    max_order: int,
) -> List[torch.Tensor]:
    device = f.device
    a_t = torch.as_tensor(a, dtype=torch.float64, device=device)
    b_t = torch.as_tensor(b, dtype=torch.float64, device=device)
    valid = (b_t > a_t) & (b_t > 0.0)
    b_safe = torch.where(valid, b_t.clamp(min=1e-16), torch.ones_like(f))
    has_a = a_t > 0.0
    a_safe = torch.where(valid & has_a, a_t.clamp(min=1e-16), torch.full_like(f, 0.5))

    f_c = f.clamp(min=1e-16)
    v_c = v.clamp(min=1e-12)
    m_u = torch.log(f_c) - 0.5 * v_c * v_c
    u_b = (torch.log(b_safe) - m_u) / v_c
    u_a = (torch.log(a_safe) - m_u) / v_c
    log_f = torch.log(f_c)
    log_b = torch.log(b_safe)
    log_a = torch.log(a_safe)

    out: List[torch.Tensor] = []
    for k in range(min(max_order, 4) + 1):
        kf = float(k)
        zb = u_b - kf * v_c
        za = u_a - kf * v_c
        use_left_bounded = has_a & (zb <= -6.0)
        use_left_unbounded = (~has_a) & (zb <= -6.0)
        use_right = has_a & (~use_left_bounded) & (za >= 6.0)

        zb_lb = torch.where(use_left_bounded, zb, torch.full_like(zb, -6.0))
        za_lb = torch.where(use_left_bounded, za, torch.full_like(za, -7.0))
        exp_ba = torch.exp((-0.5 * (za_lb - zb_lb) * (za_lb + zb_lb)).clamp(max=0.0))
        val_lb = 0.5 * torch.exp((kf * log_b - 0.5 * u_b * u_b).clamp(min=-740.0, max=600.0)) * (
            torch.special.erfcx(-zb_lb * _INV_SQRT2) - exp_ba * torch.special.erfcx(-za_lb * _INV_SQRT2)
        ).clamp(min=0.0)

        zb_lu = torch.where(use_left_unbounded, zb, torch.full_like(zb, -6.0))
        val_lu = 0.5 * torch.exp((kf * log_b - 0.5 * u_b * u_b).clamp(min=-740.0, max=600.0)) * (
            torch.special.erfcx(-zb_lu * _INV_SQRT2)
        )

        za_r = torch.where(use_right, za, torch.full_like(za, 6.0))
        zb_r = torch.where(use_right, zb, torch.full_like(zb, 7.0))
        exp_ab = torch.exp((-0.5 * (zb_r - za_r) * (zb_r + za_r)).clamp(max=0.0))
        val_r = 0.5 * torch.exp((kf * log_a - 0.5 * u_a * u_a).clamp(min=-740.0, max=600.0)) * (
            torch.special.erfcx(za_r * _INV_SQRT2) - exp_ab * torch.special.erfcx(zb_r * _INV_SQRT2)
        ).clamp(min=0.0)

        diff_mid = torch.where(has_a, torch.special.ndtr(zb) - torch.special.ndtr(za), torch.special.ndtr(zb))
        diff_pos = diff_mid.clamp(min=0.0)
        log_scale = (kf * log_f + 0.5 * float(k * (k - 1)) * (v_c * v_c)).clamp(max=600.0)
        val_mid = torch.where(diff_pos > 0.0, torch.exp(log_scale) * diff_pos, torch.zeros_like(f_c))

        val = torch.where(
            use_left_bounded,
            val_lb,
            torch.where(use_left_unbounded, val_lu, torch.where(use_right, val_r, val_mid)),
        )
        out.append(torch.where(valid, val, torch.zeros_like(f_c)))
    return out


def _normal_moments_interval_torch(
    mu: torch.Tensor,
    sig: torch.Tensor,
    a: float | torch.Tensor,
    b: float | torch.Tensor,
    max_order: int,
) -> List[torch.Tensor]:
    device = mu.device
    a_t = torch.as_tensor(a, dtype=torch.float64, device=device)
    b_t = torch.as_tensor(b, dtype=torch.float64, device=device)
    valid = b_t > a_t
    a_v = torch.where(valid, a_t, torch.zeros_like(mu))
    b_v = torch.where(valid, b_t, torch.ones_like(mu))

    def _eval_one_pass(mu_i: torch.Tensor, s_i: torch.Tensor, a_i: torch.Tensor, b_i: torch.Tensor) -> List[torch.Tensor]:
        s_c = s_i.clamp(min=1e-12)
        u_b = (b_i - mu_i) / s_c
        has_a = a_i > -1e50
        u_a = torch.where(has_a, (a_i - mu_i) / s_c, torch.zeros_like(mu_i))
        phi_b = torch.exp(-0.5 * u_b * u_b) * _INV_SQRT2PI
        phi_a = torch.where(has_a, torch.exp((-0.5 * u_a * u_a).clamp(min=-700.0)) * _INV_SQRT2PI, torch.zeros_like(mu_i))
        j0 = torch.special.ndtr(u_b) - torch.where(has_a, torch.special.ndtr(u_a), torch.zeros_like(mu_i))
        j1 = phi_a - phi_b
        j2 = j0 + torch.where(has_a, u_a * phi_a, torch.zeros_like(mu_i)) - u_b * phi_b
        j3 = 2.0 * j1 + torch.where(has_a, (u_a * u_a) * phi_a, torch.zeros_like(mu_i)) - (u_b * u_b) * phi_b
        mid_m = [j0]
        if max_order >= 1:
            mid_m.append(mu_i * j0 + s_c * j1)
        if max_order >= 2:
            mid_m.append(mu_i * mu_i * j0 + 2.0 * mu_i * s_c * j1 + (s_c * s_c) * j2)
        if max_order >= 3:
            mid_m.append(
                mu_i * mu_i * mu_i * j0
                + 3.0 * mu_i * mu_i * s_c * j1
                + 3.0 * mu_i * (s_c * s_c) * j2
                + (s_c * s_c * s_c) * j3
            )

        left_tail = u_b <= -15.0
        x_t = torch.where(left_tail, -u_b, torch.full_like(u_b, 15.0))
        pref = _INV_SQRT2PI * torch.exp((-0.5 * x_t * x_t).clamp(min=-740.0))
        du = torch.where(has_a, (u_b - u_a).clamp(min=0.0), torch.full_like(u_b, 1e100))
        use_gl = has_a & (du * x_t < 45.0)
        du_gl = torch.where(use_gl, du, torch.zeros_like(du))
        zm_gl = [torch.zeros_like(mu_i) for _ in range(4)]
        for gx, gw in zip(_GL8_X01, _GL8_W01):
            zq = du_gl * gx
            wt = pref * du_gl * gw * torch.exp((-zq * (x_t + 0.5 * zq)).clamp(min=-80.0))
            zm_gl[0] = zm_gl[0] + wt
            z_p = wt * zq
            zm_gl[1] = zm_gl[1] + z_p
            z_p = z_p * zq
            zm_gl[2] = zm_gl[2] + z_p
            zm_gl[3] = zm_gl[3] + z_p * zq
        inv_x = 1.0 / x_t
        p_scale = pref * inv_x
        zm_lg = [torch.zeros_like(mu_i) for _ in range(4)]
        for lx, lw in zip(_LAG6_X, _LAG6_W):
            zq = lx * inv_x
            wt = p_scale * lw * torch.exp((-0.5 * zq * zq).clamp(min=-80.0))
            zm_lg[0] = zm_lg[0] + wt
            z_p = wt * zq
            zm_lg[1] = zm_lg[1] + z_p
            z_p = z_p * zq
            zm_lg[2] = zm_lg[2] + z_p
            zm_lg[3] = zm_lg[3] + z_p * zq
        zm = [torch.where(use_gl, zm_gl[r], zm_lg[r]) for r in range(4)]
        tail_m = [zm[0]]
        if max_order >= 1:
            tail_m.append(b_i * zm[0] - s_c * zm[1])
        if max_order >= 2:
            tail_m.append(b_i * b_i * zm[0] - 2.0 * b_i * s_c * zm[1] + s_c * s_c * zm[2])
        if max_order >= 3:
            tail_m.append(
                b_i * b_i * b_i * zm[0]
                - 3.0 * b_i * b_i * s_c * zm[1]
                + 3.0 * b_i * s_c * s_c * zm[2]
                - s_c * s_c * s_c * zm[3]
            )
        return [torch.where(left_tail, tail_m[k], mid_m[k]) for k in range(max_order + 1)]

    s_c0 = sig.clamp(min=1e-12)
    has_a0 = a_v > -1e50
    u_a0 = torch.where(has_a0, (a_v - mu) / s_c0, torch.zeros_like(mu))
    right_tail = has_a0 & (u_a0 >= 15.0)
    fwd_m = _eval_one_pass(mu, s_c0, a_v, b_v)
    rev_m = _eval_one_pass(-mu, s_c0, -b_v, -a_v)
    res: List[torch.Tensor] = []
    for k in range(max_order + 1):
        rev_k = -rev_m[k] if (k % 2 == 1) else rev_m[k]
        res.append(torch.where(valid, torch.where(right_tail, rev_k, fwd_m[k]), torch.zeros_like(mu)))
    return res


def _clip_interval_with_ko_ramp_torch(
    a_s: float,
    b_s: float,
    has_expiry_ko: bool,
    ko_lo_t: torch.Tensor,
    ko_hi_t: torch.Tensor,
    hard_ko: bool,
) -> List[Tuple[torch.Tensor, torch.Tensor, float | torch.Tensor, float | torch.Tensor]]:
    """Splits scalar [a_s, b_s] against possibly batched KO ramp [ko_lo_t, ko_hi_t]."""
    one_t = torch.ones_like(ko_lo_t)
    zero_t = torch.zeros_like(ko_lo_t)
    if not has_expiry_ko:
        return [(torch.full_like(ko_lo_t, a_s), torch.full_like(ko_lo_t, b_s), 1.0, 0.0)]
    a_t = torch.full_like(ko_lo_t, a_s)
    b_t = torch.full_like(ko_lo_t, b_s)
    a1 = torch.minimum(a_t, ko_lo_t)
    b1 = torch.minimum(b_t, ko_lo_t)
    if hard_ko:
        return [(a1, b1, one_t, zero_t)]
    a2 = torch.maximum(a_t, ko_lo_t)
    b2 = torch.minimum(b_t, ko_hi_t)
    db_k = (ko_hi_t - ko_lo_t).clamp(min=1e-12)
    return [
        (a1, b1, one_t, zero_t),
        (a2, b2, ko_hi_t / db_k, -one_t / db_k),
    ]


def _exact_piecewise_terminal_put_torch(
    f_or_mu: torch.Tensor,
    v_or_sig: torch.Tensor,
    is_normal: bool,
    strike: float,
    has_lower_strike: bool,
    lower_strike: float,
    has_active_ki: bool,
    ki_barrier: float,
    ki_spread: float,
    has_expiry_ko: bool,
    ko_barrier: float | torch.Tensor,
    ko_spread: float,
) -> torch.Tensor:
    if (not is_normal) and strike <= 1e-8:
        return torch.zeros_like(f_or_mu)
    k_low = float(lower_strike) if has_lower_strike else 0.0
    if has_lower_strike and k_low >= strike:
        return torch.zeros_like(f_or_mu)
    low_bound = -1e60 if is_normal else 0.0
    pts: List[float] = [low_bound, strike]
    if has_lower_strike and k_low > low_bound:
        pts.append(k_low)
    if has_active_ki:
        if abs(ki_spread) <= 1e-6:
            pts.append(ki_barrier)
        else:
            pts.append(min(ki_barrier, ki_barrier + ki_spread))
            pts.append(max(ki_barrier, ki_barrier + ki_spread))
    pts = sorted(x for x in pts if low_bound <= x <= strike)

    ko_t = torch.as_tensor(ko_barrier, dtype=torch.float64, device=f_or_mu.device)
    hard_ko = abs(ko_spread) <= 1e-6
    ko_lo_t = ko_t if hard_ko else torch.minimum(ko_t, ko_t + ko_spread)
    ko_hi_t = ko_t if hard_ko else torch.maximum(ko_t, ko_t + ko_spread)

    total = torch.zeros_like(f_or_mu)
    for idx in range(len(pts) - 1):
        a_s = pts[idx]
        b_s = pts[idx + 1]
        if b_s - a_s <= 1e-14:
            continue
        mid = (b_s - 1.0) if a_s < -1e30 else 0.5 * (a_s + b_s)
        c0, c1 = ((strike - k_low, 0.0) if (has_lower_strike and mid <= k_low) else (strike, -1.0))

        d0, d1 = 1.0, 0.0
        if has_active_ki:
            if abs(ki_spread) <= 1e-6:
                if mid >= ki_barrier:
                    continue
            else:
                b_ki_lo = min(ki_barrier, ki_barrier + ki_spread)
                b_ki_hi = max(ki_barrier, ki_barrier + ki_spread)
                if mid >= b_ki_hi:
                    continue
                if mid > b_ki_lo:
                    db = b_ki_hi - b_ki_lo
                    d0, d1 = b_ki_hi / db, -1.0 / db

        cd0 = c0 * d0
        cd1 = c0 * d1 + c1 * d0
        cd2 = c1 * d1

        for a_sub, b_sub, e0, e1 in _clip_interval_with_ko_ramp_torch(a_s, b_s, has_expiry_ko, ko_lo_t, ko_hi_t, hard_ko):
            q0 = cd0 * e0
            q1 = cd0 * e1 + cd1 * e0
            q2 = cd1 * e1 + cd2 * e0
            q3 = cd2 * e1
            max_k = 3 if (not hard_ko and has_expiry_ko and abs(ki_spread) > 1e-6) else 2
            moms = (
                _normal_moments_interval_torch(f_or_mu, v_or_sig, a_sub, b_sub, max_k)
                if is_normal
                else _lognormal_moments_interval_torch(f_or_mu, v_or_sig, a_sub, b_sub, max_k)
            )
            seg_v = q0 * moms[0] + q1 * moms[1] + q2 * moms[2] + (q3 * moms[3] if max_k >= 3 else 0.0)
            total = total + torch.where(b_sub > a_sub + 1e-14, seg_v, torch.zeros_like(f_or_mu))
    return total


def _exact_piecewise_terminal_call_torch(
    f_or_mu: torch.Tensor,
    v_or_sig: torch.Tensor,
    is_normal: bool,
    k_call: float,
    has_expiry_ko: bool,
    ko_barrier: float | torch.Tensor,
    ko_spread: float,
) -> torch.Tensor:
    if not has_expiry_ko:
        k_t = torch.tensor(k_call, dtype=torch.float64, device=f_or_mu.device)
        return _normal_call_torch(f_or_mu, v_or_sig, k_t) if is_normal else _black76_call_torch(f_or_mu, k_t, v_or_sig)
    ko_t = torch.as_tensor(ko_barrier, dtype=torch.float64, device=f_or_mu.device)
    hard_ko = abs(ko_spread) <= 1e-6
    ko_lo_t = ko_t if hard_ko else torch.minimum(ko_t, ko_t + ko_spread)
    ko_hi_t = ko_t if hard_ko else torch.maximum(ko_t, ko_t + ko_spread)
    k_t = torch.full_like(ko_lo_t, k_call)

    total = torch.zeros_like(f_or_mu)
    if hard_ko:
        a1 = k_t
        b1 = ko_lo_t
        m1 = (
            _normal_moments_interval_torch(f_or_mu, v_or_sig, a1, b1, 1)
            if is_normal
            else _lognormal_moments_interval_torch(f_or_mu, v_or_sig, a1, b1, 1)
        )
        total = torch.where(b1 > a1 + 1e-14, m1[1] - k_call * m1[0], torch.zeros_like(f_or_mu))
    else:
        db = (ko_hi_t - ko_lo_t).clamp(min=1e-12)
        a1, b1 = k_t, torch.minimum(ko_lo_t, ko_hi_t)
        a2, b2 = torch.maximum(k_t, ko_lo_t), ko_hi_t
        m1 = (
            _normal_moments_interval_torch(f_or_mu, v_or_sig, a1, b1, 1)
            if is_normal
            else _lognormal_moments_interval_torch(f_or_mu, v_or_sig, a1, b1, 1)
        )
        m2 = (
            _normal_moments_interval_torch(f_or_mu, v_or_sig, a2, b2, 2)
            if is_normal
            else _lognormal_moments_interval_torch(f_or_mu, v_or_sig, a2, b2, 2)
        )
        p1 = torch.where(b1 > a1 + 1e-14, m1[1] - k_call * m1[0], torch.zeros_like(f_or_mu))
        e0 = ko_hi_t / db
        e1 = -1.0 / db
        q0 = -k_call * e0
        q1 = e0 - k_call * e1
        q2 = e1
        p2 = torch.where(b2 > a2 + 1e-14, q0 * m2[0] + q1 * m2[1] + q2 * m2[2], torch.zeros_like(f_or_mu))
        total = p1 + p2
    return total


def _exact_twinwin_terminal_put_torch(
    f: torch.Tensor,
    v: torch.Tensor,
    strike: float,
    has_lower_strike: bool,
    lower_strike: float,
    has_active_ki: bool,
    ki_barrier: float,
    ki_spread: float,
    has_expiry_ko: bool,
    ko_barrier: float | torch.Tensor,
    ko_spread: float,
) -> torch.Tensor:
    if strike <= 1.0:
        return torch.zeros_like(f)
    k_low = float(lower_strike) if has_lower_strike else 0.0
    if has_lower_strike and k_low >= strike:
        return torch.zeros_like(f)
    b_ki_lo, b_ki_hi = min(ki_barrier, ki_barrier + ki_spread), max(ki_barrier, ki_barrier + ki_spread)
    ko_t = torch.as_tensor(ko_barrier, dtype=torch.float64, device=f.device)
    b_ko_lo = torch.minimum(ko_t, ko_t + ko_spread)
    b_ko_hi = torch.maximum(ko_t, ko_t + ko_spread)

    x_dn_min = max(2.0 - strike, 0.0)
    pts: List[float] = [x_dn_min, 1.0, strike]
    if has_lower_strike and k_low > 1.0:
        pts.extend((2.0 - k_low, k_low))
    if has_active_ki:
        if abs(ki_spread) <= 1e-6:
            pts.extend((2.0 - ki_barrier, ki_barrier))
        else:
            pts.extend((2.0 - b_ki_lo, 2.0 - b_ki_hi, b_ki_lo, b_ki_hi))
    pts = sorted(x for x in pts if x_dn_min <= x <= strike)

    total = torch.zeros_like(f)
    for idx in range(len(pts) - 1):
        a_s = pts[idx]
        b_s = pts[idx + 1]
        if b_s - a_s <= 1e-14:
            continue
        mid = 0.5 * (a_s + b_s)
        s_sgn = -1.0 if mid < 1.0 else 1.0
        lin_0 = 2.0 if mid < 1.0 else 0.0
        if has_lower_strike and abs(mid - 1.0) <= k_low - 1.0:
            c0, c1 = strike - k_low, 0.0
        else:
            c0, c1 = strike - lin_0, -s_sgn

        d0, d1 = 1.0, 0.0
        if has_active_ki:
            if abs(ki_spread) <= 1e-6:
                if abs(mid - 1.0) >= ki_barrier - 1.0:
                    continue
            else:
                if abs(mid - 1.0) >= b_ki_hi - 1.0:
                    continue
                if abs(mid - 1.0) > b_ki_lo - 1.0:
                    inv_db = 1.0 / (b_ki_hi - b_ki_lo)
                    d0, d1 = (b_ki_hi - lin_0) * inv_db, -s_sgn * inv_db

        cd0, cd1, cd2 = c0 * d0, c0 * d1 + c1 * d0, c1 * d1

        if not has_expiry_ko:
            max_k = 2 if cd2 != 0.0 else 1
            moms = _lognormal_moments_interval_torch(f, v, a_s, b_s, max_k)
            total = total + cd0 * moms[0] + cd1 * moms[1] + (cd2 * moms[2] if max_k >= 2 else 0.0)
            continue

        a_t = torch.full_like(ko_t, a_s)
        b_t = torch.full_like(ko_t, b_s)
        if mid < 1.0:
            edge_hi = (2.0 - b_ko_hi).clamp(min=0.0)
            edge_lo = 2.0 - b_ko_lo
            a_pass = torch.maximum(a_t, edge_lo)
            b_pass = b_t
            a_ramp = torch.maximum(a_t, edge_hi)
            b_ramp = torch.minimum(b_t, edge_lo)
            inv_dbk = 1.0 / (b_ko_hi - b_ko_lo).clamp(min=1e-12)
            e0_r, e1_r = -(2.0 - b_ko_hi) * inv_dbk, inv_dbk
        else:
            a_pass = a_t
            b_pass = torch.minimum(b_t, b_ko_lo)
            a_ramp = torch.maximum(a_t, b_ko_lo)
            b_ramp = torch.minimum(b_t, b_ko_hi)
            inv_dbk = 1.0 / (b_ko_hi - b_ko_lo).clamp(min=1e-12)
            e0_r, e1_r = b_ko_hi * inv_dbk, -inv_dbk

        m_p = _lognormal_moments_interval_torch(f, v, a_pass, b_pass, 2)
        seg_p = torch.where(b_pass > a_pass + 1e-14, cd0 * m_p[0] + cd1 * m_p[1] + cd2 * m_p[2], torch.zeros_like(f))
        if abs(ko_spread) <= 1e-6:
            total = total + seg_p
        else:
            m_r = _lognormal_moments_interval_torch(f, v, a_ramp, b_ramp, 3)
            q0, q1, q2, q3 = cd0 * e0_r, cd0 * e1_r + cd1 * e0_r, cd1 * e1_r + cd2 * e0_r, cd2 * e1_r
            seg_r = torch.where(
                b_ramp > a_ramp + 1e-14, q0 * m_r[0] + q1 * m_r[1] + q2 * m_r[2] + q3 * m_r[3], torch.zeros_like(f)
            )
            total = total + seg_p + seg_r
    return total


def _exact_twinwin_terminal_call_torch(
    f: torch.Tensor,
    v: torch.Tensor,
    k_call: float,
    has_expiry_ko: bool,
    ko_barrier: float | torch.Tensor,
    ko_spread: float,
) -> torch.Tensor:
    f_c = f.clamp(min=1e-16)
    v_c = v.clamp(min=1e-12)
    k_eff = max(float(k_call), 1.0)
    k_shift = float(k_call)
    a_dn_k = max(2.0 - k_eff, 0.0)

    def _side_call(a_s: float | torch.Tensor, b_s: float | torch.Tensor, e0: float | torch.Tensor, e1: float | torch.Tensor, lin_0: float, s_sgn: float) -> torch.Tensor:
        valid = torch.as_tensor(b_s, dtype=torch.float64, device=f.device) > torch.as_tensor(a_s, dtype=torch.float64, device=f.device) + 1e-14
        c0 = lin_0 - k_shift
        c1 = s_sgn
        m = _lognormal_moments_interval_torch(f_c, v_c, a_s, b_s, 2)
        q0 = c0 * e0
        q1 = c0 * e1 + c1 * e0
        q2 = c1 * e1
        return torch.where(valid, q0 * m[0] + q1 * m[1] + q2 * m[2], torch.zeros_like(f_c))

    if not has_expiry_ko:
        p1 = _lognormal_moments_interval_torch(f_c, v_c, 0.0, a_dn_k, 1)
        left_ev = ((2.0 - k_shift) * p1[0] - p1[1]) if a_dn_k > 0.0 else torch.zeros_like(f_c)
        right_ev = _black76_call_torch(f_c, torch.tensor(k_eff, dtype=torch.float64, device=f.device), v_c) + (k_eff - k_shift) * torch.special.ndtr(
            (torch.log(f_c) - 0.5 * v_c * v_c - math.log(k_eff)) / v_c
        )
        return cast(torch.Tensor, left_ev + right_ev)

    ko_t = torch.as_tensor(ko_barrier, dtype=torch.float64, device=f.device)
    b_ko_lo = torch.minimum(ko_t, ko_t + ko_spread)
    b_ko_hi = torch.maximum(ko_t, ko_t + ko_spread)
    a_dn_ko_hi = (2.0 - b_ko_hi).clamp(min=0.0)
    a_dn_ko_lo = (2.0 - b_ko_lo).clamp(min=0.0)
    inv_db = 1.0 / (b_ko_hi - b_ko_lo).clamp(min=1e-12)

    a_dn_k_t = torch.full_like(ko_t, a_dn_k)
    k_eff_t = torch.full_like(ko_t, k_eff)
    if abs(ko_spread) <= 1e-6:
        s_dn = _side_call(a_dn_ko_hi, a_dn_k_t, 1.0, 0.0, 2.0, -1.0)
        s_up = _side_call(k_eff_t, b_ko_hi, 1.0, 0.0, 0.0, 1.0)
        return torch.where(b_ko_hi > k_eff, s_dn + s_up, torch.zeros_like(f_c))

    zero_t = torch.zeros_like(ko_t)
    one_t = torch.ones_like(ko_t)
    s_dn_ramp = _side_call(a_dn_ko_hi, torch.minimum(a_dn_k_t, a_dn_ko_lo), -(2.0 - b_ko_hi) * inv_db, inv_db, 2.0, -1.0)
    s_dn_full = _side_call(a_dn_ko_lo, a_dn_k_t, one_t, zero_t, 2.0, -1.0)
    s_up_full = _side_call(k_eff_t, torch.maximum(k_eff_t, b_ko_lo), one_t, zero_t, 0.0, 1.0)
    s_up_ramp = _side_call(torch.maximum(k_eff_t, b_ko_lo), b_ko_hi, b_ko_hi * inv_db, -inv_db, 0.0, 1.0)
    return torch.where(b_ko_hi > k_eff, s_dn_ramp + s_dn_full + s_up_full + s_up_ramp, torch.zeros_like(f_c))


def _onestar_payoff_fn_scalar(
    w: float,
    strike: float,
    k_low: float,
    has_active_ki: bool,
    ki_barrier: float,
    ki_spread: float,
    use_direct_ko: bool,
    ko_b_term: float,
    ko_spread: float,
) -> float:
    if w >= strike:
        return 0.0
    pay = strike - w
    if k_low > 0.0:
        pay = min(pay, max(strike - k_low, 0.0))
    if has_active_ki:
        if abs(ki_spread) <= 1e-6:
            if w >= ki_barrier:
                return 0.0
        else:
            blo = min(ki_barrier, ki_barrier + ki_spread)
            bhi = max(ki_barrier, ki_barrier + ki_spread)
            pay *= min(max((bhi - w) / (bhi - blo), 0.0), 1.0)
    if use_direct_ko:
        if abs(ko_spread) <= 1e-6:
            if w >= ko_b_term:
                return 0.0
        else:
            blo = min(ko_b_term, ko_b_term + ko_spread)
            bhi = max(ko_b_term, ko_b_term + ko_spread)
            pay *= min(max((bhi - w) / (bhi - blo), 0.0), 1.0)
    return pay


def _eval_1factor_onestar_torch(
    f_c: torch.Tensor,
    v_c: torch.Tensor,
    corr: torch.Tensor,
    strike: float,
    has_lower_strike: bool,
    lower_strike: float,
    has_active_ki: bool,
    ki_barrier: float,
    ki_spread: float,
    use_direct_ko: bool,
    ko_b_term: float | torch.Tensor,
    ko_spread: float,
    one_star_strike: float,
    one_star_coupon: float,
    surv_cur: torch.Tensor,
    surv_prev_term: torch.Tensor,
    option_gearing: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable 1-factor CDF-mapped GL16 OneStar put & coupon expectation for (B, D)."""
    b_sz, d_assets = int(f_c.shape[0]), int(f_c.shape[1])
    device = f_c.device
    if b_sz == 0 or d_assets == 0:
        z0 = torch.zeros(b_sz, dtype=torch.float64, device=device)
        return z0, z0.clone(), z0.clone()
    k_low = float(lower_strike) if has_lower_strike else 0.0
    log_ks = math.log(max(one_star_strike, 1e-16))
    ko_t = torch.as_tensor(ko_b_term, dtype=torch.float64, device=device)

    if d_assets == 1:
        eps_os = 0.02
        f0 = f_c[:, 0].clamp(min=1e-16)
        v0 = v_c[:, 0].clamp(min=1e-12)
        os_lo = max(one_star_strike - 0.5 * eps_os, 0.0)
        os_hi = one_star_strike + 0.5 * eps_os
        inv_os_eps = 1.0 / (os_hi - os_lo)
        surv_mult = surv_prev_term if use_direct_ko else surv_cur

        p_onestar_out = torch.zeros(b_sz, dtype=torch.float64, device=device)
        l_onestar_out = torch.zeros(b_sz, dtype=torch.float64, device=device)
        if one_star_strike > 0.0:
            if not use_direct_ko:
                b_lo_t = torch.full_like(f0, os_lo)
                b_hi_t = torch.full_like(f0, os_hi)
                m_r = _lognormal_moments_interval_torch(f0, v0, b_lo_t, b_hi_t, 1)
                m_u = torch.log(f0) - 0.5 * v0 * v0
                p_tail = torch.special.ndtr(-(math.log(max(os_hi, 1e-16)) - m_u) / v0)
                p_os = (-os_lo * inv_os_eps * m_r[0] + inv_os_eps * m_r[1] + p_tail).clamp(min=0.0)
            else:
                hard_ko = abs(ko_spread) <= 1e-6
                blo_ko = ko_t if hard_ko else torch.minimum(ko_t, ko_t + ko_spread)
                bhi_ko = ko_t if hard_ko else torch.maximum(ko_t, ko_t + ko_spread)
                p_os = torch.zeros_like(f0)
                os_lo_t = torch.full_like(f0, os_lo)
                zero_t = torch.zeros_like(f0)
                for seg_a_s, seg_b_s, s0, s1 in ((os_lo, os_hi, -os_lo * inv_os_eps, inv_os_eps), (os_hi, 1e18, 1.0, 0.0)):
                    for a_sub, b_sub, e0, e1 in _clip_interval_with_ko_ramp_torch(seg_a_s, seg_b_s, True, blo_ko, bhi_ko, hard_ko):
                        valid = b_sub > a_sub + 1e-14
                        if seg_b_s > 1e15 and hard_ko:
                            a_c = torch.maximum(os_lo_t, torch.full_like(f0, os_hi))
                            b_c = bhi_ko
                            val = torch.where(b_c > a_c + 1e-14, _lognormal_moments_interval_torch(f0, v0, a_c, b_c, 0)[0], zero_t)
                            p_os = p_os + val
                        else:
                            q0 = s0 * e0
                            q1 = s0 * e1 + s1 * e0
                            q2 = s1 * e1
                            moms = _lognormal_moments_interval_torch(f0, v0, a_sub, b_sub, 2)
                            seg_v = q0 * moms[0] + q1 * moms[1] + q2 * moms[2]
                            p_os = p_os + torch.where(valid, seg_v, zero_t)
                p_os = torch.where(bhi_ko > os_lo, p_os.clamp(min=0.0), zero_t)
            p_onestar_out = surv_mult * p_os
            l_onestar_out = p_onestar_out * one_star_coupon

        l_put_out = torch.zeros(b_sz, dtype=torch.float64, device=device)
        if strike > 1e-6 and option_gearing > 0.0 and one_star_strike > 1e-16:
            if os_lo >= strike:
                l_put_out = surv_mult * _exact_piecewise_terminal_put_torch(
                    f0, v0, False, strike, has_lower_strike, lower_strike,
                    has_active_ki, ki_barrier, ki_spread, use_direct_ko, ko_t, ko_spread
                )
                return l_put_out, l_onestar_out, p_onestar_out

            hi = min(strike, os_hi)
            pts_p: List[float] = [0.0, hi]
            if has_lower_strike and k_low > 0.0:
                pts_p.append(k_low)
            if os_lo > 0.0:
                pts_p.append(os_lo)
            if has_active_ki:
                if abs(ki_spread) <= 1e-6:
                    pts_p.append(ki_barrier)
                else:
                    pts_p.extend((min(ki_barrier, ki_barrier + ki_spread), max(ki_barrier, ki_barrier + ki_spread)))
            pts_p = sorted(x for x in pts_p if 0.0 <= x <= hi)
            hard_ko = abs(ko_spread) <= 1e-6
            ko_lo_t = ko_t if hard_ko else torch.minimum(ko_t, ko_t + ko_spread)
            ko_hi_t = ko_t if hard_ko else torch.maximum(ko_t, ko_t + ko_spread)
            total_put = torch.zeros(b_sz, dtype=torch.float64, device=device)
            for idx in range(len(pts_p) - 1):
                a_s, b_s = pts_p[idx], pts_p[idx + 1]
                if b_s - a_s <= 1e-14:
                    continue
                mid = 0.5 * (a_s + b_s)
                c0, c1 = ((strike - k_low, 0.0) if (has_lower_strike and mid <= k_low) else (strike, -1.0))
                d0, d1 = 1.0, 0.0
                if has_active_ki:
                    if abs(ki_spread) <= 1e-6:
                        if mid >= ki_barrier:
                            continue
                    else:
                        blo, bhi = min(ki_barrier, ki_barrier + ki_spread), max(ki_barrier, ki_barrier + ki_spread)
                        if mid >= bhi:
                            continue
                        if mid > blo:
                            d0, d1 = bhi / (bhi - blo), -1.0 / (bhi - blo)
                r0, r1 = (os_hi * inv_os_eps, -inv_os_eps) if mid > os_lo else (1.0, 0.0)
                cd0, cd1, cd2 = c0 * d0, c0 * d1 + c1 * d0, c1 * d1
                cdr0, cdr1, cdr2, cdr3 = cd0 * r0, cd0 * r1 + cd1 * r0, cd1 * r1 + cd2 * r0, cd2 * r1
                for a_sub, b_sub, e0, e1 in _clip_interval_with_ko_ramp_torch(a_s, b_s, use_direct_ko, ko_lo_t, ko_hi_t, hard_ko):
                    q0, q1 = cdr0 * e0, cdr0 * e1 + cdr1 * e0
                    q2, q3, q4 = cdr1 * e1 + cdr2 * e0, cdr2 * e1 + cdr3 * e0, cdr3 * e1
                    moms = _lognormal_moments_interval_torch(f0, v0, a_sub, b_sub, 4)
                    seg_v = q0 * moms[0] + q1 * moms[1] + q2 * moms[2] + q3 * moms[3] + q4 * moms[4]
                    total_put = total_put + torch.where(b_sub > a_sub + 1e-14, seg_v, torch.zeros_like(f0))
            l_put_out = surv_mult * total_put.clamp(min=0.0)
        return l_put_out, l_onestar_out, p_onestar_out

    corr_b = corr.unsqueeze(0) if corr.ndim == 2 else corr
    b_vec_b = _fit_1factor_loadings_torch(corr_b)

    f_s = f_c.clamp(min=1e-16)
    v_s = v_c.clamp(min=1e-12)
    s_eps = torch.sqrt((1.0 - b_vec_b * b_vec_b).clamp(min=1e-12))
    mu_bd = torch.log(f_s) - 0.5 * v_s * v_s
    vb_bd = v_s * b_vec_b
    vs_bd = (v_s * s_eps).clamp(min=1e-12)

    y_q = torch.tensor(_GL16_Y_NORM, dtype=torch.float64, device=device).view(1, 16, 1)
    w_q = torch.tensor(_GL16_W, dtype=torch.float64, device=device).view(1, 16)
    c_ks = torch.special.ndtr(
        (log_ks - mu_bd.unsqueeze(1) - vb_bd.unsqueeze(1) * y_q) / vs_bd.unsqueeze(1)
    )
    p_max_lt_ks_y = c_ks.prod(dim=2)

    if one_star_strike > 0.0:
        if use_direct_ko:
            if abs(ko_spread) <= 1e-6:
                ko_pts_t = ko_t.expand(b_sz).unsqueeze(1)
                n_ko = 1
            else:
                blo_t = torch.minimum(ko_t, ko_t + ko_spread).expand(b_sz)
                bhi_t = torch.maximum(ko_t, ko_t + ko_spread).expand(b_sz)
                mid_t = 0.5 * (blo_t + bhi_t)
                hw_t = 0.5 * (bhi_t - blo_t)
                gx4_t = torch.tensor(_GX4, dtype=torch.float64, device=device).view(1, 4)
                ko_pts_t = (mid_t.unsqueeze(1) + hw_t.unsqueeze(1) * gx4_t).clamp(min=1e-16)
                n_ko = 4
            p_sub = torch.zeros((b_sz, 16), dtype=torch.float64, device=device)
            for r in range(n_ko):
                ko_r = ko_pts_t[:, r]
                log_ko = torch.log(ko_r.clamp(min=1e-16)).view(b_sz, 1, 1)
                c_ko = torch.special.ndtr(
                    (log_ko - mu_bd.unsqueeze(1) - vb_bd.unsqueeze(1) * y_q) / vs_bd.unsqueeze(1)
                )
                p_ge_ko = (1.0 - c_ko).prod(dim=2)
                p_in = torch.where(
                    (ko_r < one_star_strike).unsqueeze(1),
                    (c_ks - c_ko).clamp(min=0.0).prod(dim=2),
                    torch.zeros_like(p_ge_ko),
                )
                p_term = ((1.0 - p_ge_ko) - (p_max_lt_ks_y - p_in)).clamp(min=0.0)
                wt_r = 1.0 if n_ko == 1 else (0.5 * _GW4[r])
                p_sub = p_sub + wt_r * p_term
            os_acc = (0.5 * w_q * p_sub).sum(dim=1)
            p_onestar_out = surv_prev_term * os_acc
        else:
            os_acc = (0.5 * w_q * (1.0 - p_max_lt_ks_y)).sum(dim=1)
            p_onestar_out = surv_cur * os_acc
        l_onestar_out = p_onestar_out * one_star_coupon
    else:
        p_onestar_out = torch.zeros(b_sz, dtype=torch.float64, device=device)
        l_onestar_out = torch.zeros(b_sz, dtype=torch.float64, device=device)

    l_put_out = torch.zeros(b_sz, dtype=torch.float64, device=device)
    if strike > 1e-6 and option_gearing > 0.0:
        def _eval_cum_joint_t(x_val: float | torch.Tensor) -> torch.Tensor:
            x_t = torch.as_tensor(x_val, dtype=torch.float64, device=device)
            pos = x_t > 0.0
            x_safe = torch.where(pos, x_t.clamp(min=1e-16), torch.ones_like(x_t))
            log_x = torch.log(x_safe).view(-1, 1, 1) if x_t.ndim >= 1 else torch.log(x_safe).view(1, 1, 1)
            c_x = torch.special.ndtr(
                (log_x - mu_bd.unsqueeze(1) - vb_bd.unsqueeze(1) * y_q) / vs_bd.unsqueeze(1)
            )
            p_in = (c_ks - c_x).clamp(min=0.0).prod(dim=2)
            out_v = cast(torch.Tensor, (0.5 * w_q * (p_max_lt_ks_y - p_in).clamp(min=0.0)).sum(dim=1))
            return torch.where(pos, out_v, torch.zeros_like(out_v))

        def _onestar_payoff_fn_tensor(x_t: torch.Tensor) -> torch.Tensor:
            p_base = (strike - x_t).clamp(min=0.0)
            if has_lower_strike:
                p_base = torch.minimum(p_base, torch.full_like(p_base, max(strike - k_low, 0.0)))
            if has_active_ki:
                if abs(ki_spread) <= 1e-6:
                    p_base = torch.where(x_t < ki_barrier, p_base, torch.zeros_like(p_base))
                else:
                    blo_k, bhi_k = min(ki_barrier, ki_barrier + ki_spread), max(ki_barrier, ki_barrier + ki_spread)
                    p_base = p_base * ((bhi_k - x_t) / (bhi_k - blo_k)).clamp(0.0, 1.0)
            if use_direct_ko:
                if abs(ko_spread) <= 1e-6:
                    p_base = torch.where(x_t < ko_t, p_base, torch.zeros_like(p_base))
                else:
                    blo_o = torch.minimum(ko_t, ko_t + ko_spread)
                    bhi_o = torch.maximum(ko_t, ko_t + ko_spread)
                    p_base = p_base * ((bhi_o - x_t) / (bhi_o - blo_o).clamp(min=1e-12)).clamp(0.0, 1.0)
            return cast(torch.Tensor, p_base)

        pts_list: List[float] = [0.0, strike]
        if has_lower_strike and 0.0 < k_low < strike:
            pts_list.append(k_low)
        if has_active_ki:
            if abs(ki_spread) <= 1e-6:
                if 0.0 <= ki_barrier <= strike:
                    pts_list.append(ki_barrier)
            else:
                for p_v in (min(ki_barrier, ki_barrier + ki_spread), max(ki_barrier, ki_barrier + ki_spread)):
                    if 0.0 <= p_v <= strike:
                        pts_list.append(p_v)
        pts_list.sort()

        put_acc = torch.zeros(b_sz, dtype=torch.float64, device=device)
        hard_ko = abs(ko_spread) <= 1e-6
        ko_lo_t = (ko_t if hard_ko else torch.minimum(ko_t, ko_t + ko_spread)).expand(b_sz)
        ko_hi_t = (ko_t if hard_ko else torch.maximum(ko_t, ko_t + ko_spread)).expand(b_sz)

        for idx in range(len(pts_list) - 1):
            a_s = pts_list[idx]
            b_pt_s = pts_list[idx + 1]
            span_s = b_pt_s - a_s
            if span_s <= 1e-12:
                continue
            if not use_direct_ko:
                mid_x = 0.5 * (a_s + b_pt_s)
                hw_x = 0.5 * span_s
                next_span = (pts_list[idx + 2] - b_pt_s) if (idx + 2 < len(pts_list)) else 1.0
                probe_right = b_pt_s + min(1e-10, 0.25 * max(next_span, 0.0))
                ga = _onestar_payoff_fn_scalar(a_s + 1e-10 * span_s, strike, k_low, has_active_ki, ki_barrier, ki_spread, False, 0.0, 0.0)
                gb_m = _onestar_payoff_fn_scalar(b_pt_s - 1e-10 * span_s, strike, k_low, has_active_ki, ki_barrier, ki_spread, False, 0.0, 0.0)
                gb_p = _onestar_payoff_fn_scalar(probe_right, strike, k_low, has_active_ki, ki_barrier, ki_spread, False, 0.0, 0.0)
                if ga > 0.0 or gb_m > 0.0:
                    linear_unit = (not has_active_ki) and ((not has_lower_strike) or a_s >= k_low - 1e-12)
                    for r in range(16):
                        xr = mid_x + hw_x * _GL16_X[r]
                        if linear_unit:
                            neg_gp = 1.0
                        else:
                            h_fd = 1e-7 * max(span_s, 1e-3)
                            xp = min(xr + h_fd, b_pt_s - 1e-12 * span_s)
                            xm = max(xr - h_fd, a_s + 1e-12 * span_s)
                            dx_fd = xp - xm
                            g_p = _onestar_payoff_fn_scalar(xp, strike, k_low, has_active_ki, ki_barrier, ki_spread, False, 0.0, 0.0)
                            g_m = _onestar_payoff_fn_scalar(xm, strike, k_low, has_active_ki, ki_barrier, ki_spread, False, 0.0, 0.0)
                            neg_gp = (-(g_p - g_m) / dx_fd) if dx_fd > 0.0 else 0.0
                        if abs(neg_gp) > 1e-15:
                            put_acc = put_acc + hw_x * _GL16_W[r] * neg_gp * _eval_cum_joint_t(xr)
                jump = gb_m - gb_p
                if abs(jump) > 1e-12:
                    put_acc = put_acc + jump * _eval_cum_joint_t(b_pt_s)
                continue

            # When use_direct_ko=True and ko_t is scalar or batched tensor, split [a_s, b_pt_s] around [ko_lo_t, ko_hi_t].
            a_t = torch.full((b_sz,), a_s, dtype=torch.float64, device=device)
            b_t = torch.full((b_sz,), b_pt_s, dtype=torch.float64, device=device)
            sub_segs = [(torch.minimum(a_t, ko_lo_t), torch.minimum(b_t, ko_lo_t))]
            if not hard_ko:
                sub_segs.append((torch.maximum(a_t, ko_lo_t), torch.minimum(b_t, ko_hi_t)))
            for a_sub, b_sub in sub_segs:
                span_sub = b_sub - a_sub
                valid = span_sub > 1e-12
                mid_sub = 0.5 * (a_sub + b_sub)
                hw_sub = 0.5 * span_sub
                h_fd_t = (1e-7 * span_sub.clamp(min=1e-3)).clamp(max=0.25 * span_sub)
                seg_acc = torch.zeros(b_sz, dtype=torch.float64, device=device)
                for r in range(16):
                    xr_t = mid_sub + hw_sub * _GL16_X[r]
                    xp_t = torch.minimum(xr_t + h_fd_t, b_sub - 1e-12 * span_sub)
                    xm_t = torch.maximum(xr_t - h_fd_t, a_sub + 1e-12 * span_sub)
                    dx_t = (xp_t - xm_t).clamp(min=1e-30)
                    neg_gp_t = -(_onestar_payoff_fn_tensor(xp_t) - _onestar_payoff_fn_tensor(xm_t)) / dx_t
                    seg_acc = seg_acc + hw_sub * _GL16_W[r] * neg_gp_t * _eval_cum_joint_t(xr_t)
                put_acc = put_acc + torch.where(valid, seg_acc, torch.zeros_like(put_acc))
            # Handle step jump at static endpoint b_pt_s (e.g. hard KI step).
            next_span = (pts_list[idx + 2] - b_pt_s) if (idx + 2 < len(pts_list)) else 1.0
            probe_r = b_pt_s + min(1e-10, 0.25 * max(next_span, 0.0))
            if hard_ko:
                jump_noko_s = _onestar_payoff_fn_scalar(
                    b_pt_s - 1e-10 * span_s, strike, k_low, has_active_ki, ki_barrier, ki_spread, False, 0.0, 0.0
                ) - _onestar_payoff_fn_scalar(
                    probe_r, strike, k_low, has_active_ki, ki_barrier, ki_spread, False, 0.0, 0.0
                )
                if abs(jump_noko_s) > 1e-12:
                    put_acc = put_acc + torch.where(
                        ko_lo_t > b_pt_s,
                        jump_noko_s * _eval_cum_joint_t(b_pt_s),
                        torch.zeros_like(put_acc),
                    )
            else:
                jump_s = _onestar_payoff_fn_tensor(torch.full((b_sz,), b_pt_s - 1e-10 * span_s, dtype=torch.float64, device=device)) - _onestar_payoff_fn_tensor(
                    torch.full((b_sz,), probe_r, dtype=torch.float64, device=device)
                )
                put_acc = put_acc + jump_s * _eval_cum_joint_t(b_pt_s)

        if use_direct_ko and hard_ko:
            ko_in = (ko_lo_t > 0.0) & (ko_lo_t <= strike)
            left_ko_val = (strike - ko_lo_t).clamp(min=0.0)
            if has_lower_strike:
                left_ko_val = torch.minimum(left_ko_val, torch.full_like(left_ko_val, max(strike - k_low, 0.0)))
            if has_active_ki:
                if abs(ki_spread) <= 1e-6:
                    left_ko_val = torch.where(ko_lo_t <= ki_barrier, left_ko_val, torch.zeros_like(left_ko_val))
                else:
                    blo_k, bhi_k = min(ki_barrier, ki_barrier + ki_spread), max(ki_barrier, ki_barrier + ki_spread)
                    left_ko_val = left_ko_val * ((bhi_k - ko_lo_t) / (bhi_k - blo_k)).clamp(0.0, 1.0)
            put_acc = put_acc + torch.where(ko_in, left_ko_val * _eval_cum_joint_t(ko_lo_t), torch.zeros_like(put_acc))
        l_put_out = (surv_prev_term if use_direct_ko else surv_cur) * put_acc

    return l_put_out, l_onestar_out, p_onestar_out


def _euler_next_corr_torch(
    f_cur: torch.Tensor,
    v_cur: torch.Tensor,
    w_cur: torch.Tensor,
    f_nxt: torch.Tensor,
    v_nxt: torch.Tensor,
    w_nxt: torch.Tensor,
    corr: torch.Tensor,
    is_normal: bool,
) -> torch.Tensor:
    """Computes Euler-weighted inter-period correlation rho_{m,m+1}^{eff} clipped to [-0.9999, 0.9999]."""
    wi_c = w_cur * f_cur if is_normal else w_cur
    wj_n = w_nxt * f_nxt if is_normal else w_nxt
    v_min = torch.minimum(v_cur.unsqueeze(-2), v_nxt.unsqueeze(-2))
    # w_cur[..., i] * w_nxt[..., j] * corr[..., i, j] * v_cur[..., i] * min(v_cur[..., j], v_nxt[..., j])
    num = (
        wi_c.unsqueeze(-1)
        * wj_n.unsqueeze(-2)
        * corr
        * v_cur.unsqueeze(-1)
        * v_min
    ).sum(dim=(-2, -1))
    den_cur = (
        wi_c.unsqueeze(-1) * wi_c.unsqueeze(-2) * corr * v_cur.unsqueeze(-1) * v_cur.unsqueeze(-2)
    ).sum(dim=(-2, -1)).clamp(min=1e-30)
    den_nxt = (
        wj_n.unsqueeze(-1) * wj_n.unsqueeze(-2) * corr * v_nxt.unsqueeze(-1) * v_nxt.unsqueeze(-2)
    ).sum(dim=(-2, -1)).clamp(min=1e-30)
    denom = (torch.sqrt(den_cur) * torch.sqrt(den_nxt)).clamp(min=1e-30)
    return (num / denom).clamp(-0.9999, 0.9999)


def _compute_cross_index_corr_torch(
    f_in: torch.Tensor,
    v_in: torch.Tensor,
    corr: torch.Tensor,
    code_a: int,
    w_a: torch.Tensor,
    code_b: int,
    w_b: torch.Tensor,
) -> torch.Tensor:
    """Cross-index correlation for 1D (D,), 2D (B, D), or 3D (B, M, D) inputs."""
    if code_a == code_b:
        return torch.ones(f_in.shape[:-1], dtype=torch.float64, device=f_in.device)
    if (code_a == 0 and code_b == 1) or (code_a == 1 and code_b == 0):
        if f_in.ndim == 1:
            return _clark_corr_min_max_torch(f_in, v_in, corr)
        lead_shape = f_in.shape[:-1]
        d_assets = int(f_in.shape[-1])
        f_flat = f_in.reshape(-1, d_assets)
        v_flat = v_in.reshape(-1, d_assets)
        c_bc = corr.expand(*lead_shape, d_assets, d_assets).reshape(-1, d_assets, d_assets)
        vals = torch.stack(
            [_clark_corr_min_max_torch(f_flat[k], v_flat[k], c_bc[k]) for k in range(int(f_flat.shape[0]))],
            dim=0,
        )
        return vals.view(lead_shape)
    da_diff = 9 <= code_a <= 13
    db_diff = 9 <= code_b <= 13
    wa = w_a * f_in if da_diff else w_a
    wb = w_b * f_in if db_diff else w_b
    cij = corr * v_in.unsqueeze(-1) * v_in.unsqueeze(-2)
    num = (wa.unsqueeze(-1) * wb.unsqueeze(-2) * cij).sum(dim=(-2, -1))
    da = (wa.unsqueeze(-1) * wa.unsqueeze(-2) * cij).sum(dim=(-2, -1)).clamp(min=1e-30)
    db = (wb.unsqueeze(-1) * wb.unsqueeze(-2) * cij).sum(dim=(-2, -1)).clamp(min=1e-30)
    return (num / torch.sqrt(da * db)).clamp(-0.9999, 0.9999)


def _coalesce_float(val: Any, default: float) -> float:
    return default if val is None else float(val)


def _coalesce_bool(val: Any, default: bool) -> bool:
    return default if val is None else bool(val)


def _perm_to_np(perm: Any, d_assets: int) -> npt.NDArray[np.int64]:
    if perm is None:
        return np.arange(d_assets, dtype=np.int64)
    arr = np.asarray(
        perm.detach().cpu().numpy() if isinstance(perm, torch.Tensor) else perm,
        dtype=np.int64,
    ).reshape(-1)
    return arr if arr.size >= 2 else np.arange(d_assets, dtype=np.int64)


def _extract_state_fields(
    preproc_state: Any,
    coupon_index: Optional[str],
    option_index: Optional[str],
    lower_strike: float,
    allow_autocall_on_expiry: bool,
    inv_lex_perm: Optional[Sequence[int] | npt.NDArray[np.int64]],
    coupon_barrier_type: str,
    coupon_barrier_direction: str,
) -> Dict[str, Any]:
    is_mapping = isinstance(preproc_state, dict) or (
        not hasattr(preproc_state, "forward_perf") and isinstance(preproc_state, Mapping)
    )

    def _get(key: str, default: Any = None) -> Any:
        if is_mapping:
            return preproc_state.get(key, default)
        return getattr(preproc_state, key, default)

    is_long_basket_flag = _coalesce_bool(_get("is_long_basket", False), False)
    call_idx = str(_get("callability_index", "WorstOf") or "WorstOf").strip() or "WorstOf"
    raw_cpn_idx = coupon_index if coupon_index is not None else _get("coupon_index", None)
    raw_opt_idx = option_index if option_index is not None else _get("option_index", None)
    eff_cpn_idx = str(raw_cpn_idx).strip() if (raw_cpn_idx is not None and str(raw_cpn_idx).strip()) else call_idx
    opt_empty = raw_opt_idx is None or not str(raw_opt_idx).strip()
    base_opt_idx = call_idx if opt_empty else str(raw_opt_idx).strip()
    eff_opt_idx = (
        "AverageOf"
        if (is_long_basket_flag and (opt_empty or monitoring_index_to_code(base_opt_idx) == 0))
        else base_opt_idx
    )

    raw_ls_state = _get("lower_strike", None)
    if lower_strike != 0.0:
        eff_lower_strike = float(lower_strike)
        has_lower_strike = math.isfinite(eff_lower_strike)
    elif raw_ls_state is not None:
        eff_lower_strike = float(raw_ls_state)
        has_lower_strike = math.isfinite(eff_lower_strike)
    else:
        eff_lower_strike = 0.0
        has_lower_strike = False

    eff_allow_exp = bool(allow_autocall_on_expiry) or _coalesce_bool(_get("allow_autocall_on_expiry", False), False)
    eff_perm = inv_lex_perm if inv_lex_perm is not None else _get("inv_lex_perm", None)

    state_cbt = _get("coupon_barrier_type", "In") or "In"
    state_cbd = _get("coupon_barrier_direction", "Up") or "Up"
    eff_cbt = str(coupon_barrier_type if coupon_barrier_type != "In" else state_cbt)
    eff_cbd = str(coupon_barrier_direction if coupon_barrier_direction != "Up" else state_cbd)

    ki_raw = _get("ki_barrier", None)
    ki_val = float(ki_raw) if ki_raw is not None else 0.0
    part_raw = _get("participation_call_strike", None)
    part_val = float(part_raw) if part_raw is not None else -1.0
    os_raw = _get("one_star_strike", None)
    os_val = float(os_raw) if os_raw is not None else 0.0

    return {
        "forward_perf": preproc_state["forward_perf"] if is_mapping else preproc_state.forward_perf,
        "v_perf": preproc_state["v_perf"] if is_mapping else preproc_state.v_perf,
        "corr_matrix": preproc_state["corr_matrix"] if is_mapping else preproc_state.corr_matrix,
        "df_pay": preproc_state["df_pay"] if is_mapping else preproc_state.df_pay,
        "tau_obs": _get("tau_obs", None),
        "ko_barriers": preproc_state["ko_barriers"] if is_mapping else preproc_state.ko_barriers,
        "is_callable": preproc_state["is_callable"] if is_mapping else preproc_state.is_callable,
        "cpn_barriers": preproc_state["cpn_barriers"] if is_mapping else preproc_state.cpn_barriers,
        "is_guaranteed_cpn": preproc_state["is_guaranteed_cpn"] if is_mapping else preproc_state.is_guaranteed_cpn,
        "coupon_amounts": preproc_state["coupon_amounts"] if is_mapping else preproc_state.coupon_amounts,
        "strike": _coalesce_float(_get("strike", 0.90), 0.90),
        "option_gearing": _coalesce_float(_get("option_gearing", 1.11111), 1.11111),
        "lower_strike": eff_lower_strike,
        "has_lower_strike": has_lower_strike,
        "ki_barrier": ki_val,
        "has_ki": ki_raw is not None and ki_val > 0.0,
        "i_ki_past": _coalesce_float(_get("i_ki_past", 0.0), 0.0),
        "c_mem_0": _coalesce_float(_get("c_mem_0", 0.0), 0.0),
        "is_memory": _coalesce_bool(_get("is_memory", True), True),
        "ko_spread": _coalesce_float(_get("ko_spread", -0.15), -0.15),
        "cpn_spread": _coalesce_float(_get("cpn_spread", -0.15), -0.15),
        "ki_spread": _coalesce_float(_get("ki_spread", -0.15), -0.15),
        "no_notional_repayment": _coalesce_bool(_get("no_notional_repayment", False), False),
        "autocall_coupon": _coalesce_float(_get("autocall_coupon", 0.0), 0.0),
        "force_coupons_on_autocall": _coalesce_bool(_get("force_coupons_on_autocall", False), False),
        "allow_autocall_on_expiry": eff_allow_exp,
        "inv_lex_perm": eff_perm,
        "coupon_barrier_type": eff_cbt,
        "coupon_barrier_direction": eff_cbd,
        "is_long_basket": is_long_basket_flag,
        "pv_in_transit": _coalesce_float(_get("pv_in_transit", 0.0), 0.0),
        "already_autocalled": _coalesce_bool(_get("already_autocalled", False), False),
        "callability_index": call_idx,
        "coupon_index": eff_cpn_idx,
        "option_index": eff_opt_idx,
        "part_call_strike": part_val,
        "has_part_call_strike": part_raw is not None,
        "part_call_rate": _coalesce_float(_get("participation_call_rate", 0.0), 0.0),
        "one_star_strike": os_val,
        "one_star_coupon": _coalesce_float(_get("one_star_coupon", 0.0), 0.0),
    }


_EMPTY_F64 = np.empty(0, dtype=np.float64)


def _tensor_to_np_ro(t: torch.Tensor) -> npt.NDArray[np.float64]:
    return t.detach().numpy() if t.requires_grad else t.numpy()


def _tensor_ptr_f64(t: torch.Tensor, keep: List[Any]) -> int:
    if t.device.type == "cpu" and t.dtype is torch.float64 and t.is_contiguous():
        return int(t.data_ptr())
    arr = np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float64)
    keep.append(arr)
    return int(arr.ctypes.data)


def _baseline_cpu_via_cpp(
    backend: Any,
    d_assets: int,
    m_rem: int,
    fp_t: torch.Tensor,
    vp_t: torch.Tensor,
    cm_t: torch.Tensor,
    tau_raw: Any,
    df_t: torch.Tensor,
    ko_t: torch.Tensor,
    ic_t: torch.Tensor,
    cb_t: torch.Tensor,
    ig_t: torch.Tensor,
    cr_t: torch.Tensor,
    cbt_i: int,
    cbd_i: int,
    eff_perm: Any,
    strike: float,
    lower_strike: float,
    ki_val: float,
    has_ki: bool,
    i_ki_past: float,
    ko_spread: float,
    cpn_spread: float,
    ki_spread: float,
    call_code: int,
    cpn_code: int,
    opt_code: int,
    allow_autocall_on_expiry: bool,
    c_mem_0: float,
    is_memory: bool,
    autocall_coupon: float,
    force_coupons_on_autocall: bool,
    no_notional_repayment: bool,
    is_long_basket: bool,
    option_gearing: float,
    part_call_strike: float,
    part_call_rate: float,
    one_star_strike: float,
    one_star_coupon: float,
    pv_in_transit: float,
    has_unsupported: bool,
    unsupported_tup: Tuple[str, ...],
) -> AutocallBaselineOutput:
    if not (
        fp_t.is_cpu
        and vp_t.is_cpu
        and cm_t.is_cpu
        and df_t.is_cpu
        and ko_t.is_cpu
        and ic_t.is_cpu
        and cb_t.is_cpu
        and ig_t.is_cpu
        and cr_t.is_cpu
    ):
        raise ValueError("_baseline_cpu_via_cpp requires CPU tensors.")
    if d_assets <= 0 or m_rem <= 0:
        z_m = torch.zeros(max(m_rem, 0), dtype=torch.float64)
        z_s = torch.zeros((), dtype=torch.float64)
        return AutocallBaselineOutput(
            f_w=z_m,
            v_w=z_m.clone(),
            h_proxy=z_m.clone(),
            p_proxy=z_m.clone(),
            survival_proxy=z_m.clone(),
            q_proxy=z_m.clone(),
            l_put_proxy=z_s,
            proxy_delta_weights=torch.zeros(max(d_assets, 0), dtype=torch.float64),
            option_pv_proxy=z_s.clone(),
            notional_pv_proxy=z_s.clone(),
            coupon_pv_proxy=z_s.clone(),
            product_value_proxy=torch.tensor(pv_in_transit, dtype=torch.float64),
            l_call_proxy=z_s.clone(),
            l_onestar_proxy=z_s.clone(),
            p_onestar_proxy=z_s.clone(),
            has_unsupported_exotic_index=has_unsupported,
            unsupported_indices=unsupported_tup,
        )
    if (
        fp_t.ndim != 2
        or vp_t.shape != fp_t.shape
        or cm_t.ndim != 2
        or cm_t.shape[0] != d_assets
        or cm_t.shape[1] != d_assets
        or df_t.ndim < 1
        or ko_t.ndim < 1
        or ic_t.ndim < 1
        or cb_t.ndim < 1
        or ig_t.ndim < 1
        or cr_t.ndim < 1
        or df_t.shape[0] < m_rem
        or ko_t.shape[0] < m_rem
        or ic_t.shape[0] < m_rem
        or cb_t.shape[0] < m_rem
        or ig_t.shape[0] < m_rem
        or cr_t.shape[0] < m_rem
    ):
        raise ValueError(f"Mismatched dimensions in _baseline_cpu_via_cpp for d={d_assets}, m={m_rem}.")

    keep: List[Any] = []
    if (
        fp_t.dtype is torch.float64
        and fp_t.is_contiguous()
        and vp_t.dtype is torch.float64
        and vp_t.is_contiguous()
        and cm_t.dtype is torch.float64
        and cm_t.is_contiguous()
        and df_t.dtype is torch.float64
        and df_t.is_contiguous()
    ):
        fp_p = fp_t.data_ptr()
        vp_p = vp_t.data_ptr()
        cm_p = cm_t.data_ptr()
        df_p = df_t.data_ptr()
        ko_p = ko_t.data_ptr() if (ko_t.dtype is torch.float64 and ko_t.is_contiguous()) else _tensor_ptr_f64(ko_t, keep)
        ic_p = ic_t.data_ptr() if (ic_t.dtype is torch.float64 and ic_t.is_contiguous()) else _tensor_ptr_f64(ic_t, keep)
        cb_p = cb_t.data_ptr() if (cb_t.dtype is torch.float64 and cb_t.is_contiguous()) else _tensor_ptr_f64(cb_t, keep)
        ig_p = ig_t.data_ptr() if (ig_t.dtype is torch.float64 and ig_t.is_contiguous()) else _tensor_ptr_f64(ig_t, keep)
        cr_p = cr_t.data_ptr() if (cr_t.dtype is torch.float64 and cr_t.is_contiguous()) else _tensor_ptr_f64(cr_t, keep)
    else:
        fp_p = _tensor_ptr_f64(fp_t, keep)
        vp_p = _tensor_ptr_f64(vp_t, keep)
        cm_p = _tensor_ptr_f64(cm_t, keep)
        df_p = _tensor_ptr_f64(df_t, keep)
        ko_p = _tensor_ptr_f64(ko_t, keep)
        ic_p = _tensor_ptr_f64(ic_t, keep)
        cb_p = _tensor_ptr_f64(cb_t, keep)
        ig_p = _tensor_ptr_f64(ig_t, keep)
        cr_p = _tensor_ptr_f64(cr_t, keep)

    if tau_raw is None:
        tau_p = 0
    elif isinstance(tau_raw, torch.Tensor):
        tau_p = (
            tau_raw.data_ptr()
            if (tau_raw.is_cpu and tau_raw.dtype is torch.float64 and tau_raw.is_contiguous())
            else _tensor_ptr_f64(tau_raw, keep)
        )
    else:
        tau_arr = np.ascontiguousarray(tau_raw, dtype=np.float64)
        keep.append(tau_arr)
        tau_p = int(tau_arr.ctypes.data) if tau_arr.size > 0 else 0

    if (
        eff_perm is None
        or (
            call_code not in (4, 5, 9, 10)
            and cpn_code not in (4, 5, 9, 10)
            and opt_code not in (4, 5, 9, 10)
        )
    ):
        perm_p = 0
    elif isinstance(eff_perm, torch.Tensor):
        if eff_perm.numel() >= 2:
            if eff_perm.is_cpu and eff_perm.dtype is torch.int64 and eff_perm.is_contiguous():
                perm_p = int(eff_perm.data_ptr())
            else:
                perm_arr = np.ascontiguousarray(eff_perm.detach().cpu().numpy(), dtype=np.int64).reshape(-1)
                keep.append(perm_arr)
                perm_p = int(perm_arr.ctypes.data)
        else:
            perm_p = 0
    elif isinstance(eff_perm, np.ndarray) and eff_perm.dtype == np.int64 and eff_perm.flags.c_contiguous:
        perm_p = int(eff_perm.__array_interface__["data"][0]) if eff_perm.size >= 2 else 0
    else:
        perm_arr = np.ascontiguousarray(eff_perm, dtype=np.int64).reshape(-1)
        if perm_arr.size >= 2:
            keep.append(perm_arr)
            perm_p = int(perm_arr.ctypes.data)
        else:
            perm_p = 0

    m_bytes = m_rem * 8
    arr_6m = torch.empty((6, m_rem), dtype=torch.float64)
    dw_all = torch.empty((d_assets, m_rem), dtype=torch.float64)
    sc_buf = torch.empty(8, dtype=torch.float64)
    arr_ptr = arr_6m.data_ptr()
    packed = backend._args_pack(
        d_assets,
        m_rem,
        fp_p,
        vp_p,
        cm_p,
        tau_p,
        df_p,
        ko_p,
        ic_p,
        cb_p,
        ig_p,
        cr_p,
        perm_p,
        strike,
        lower_strike,
        ki_val,
        i_ki_past,
        ko_spread,
        cpn_spread,
        ki_spread,
        c_mem_0,
        autocall_coupon,
        option_gearing,
        part_call_strike,
        part_call_rate,
        one_star_strike,
        one_star_coupon,
        pv_in_transit,
        cbt_i,
        cbd_i,
        1 if has_ki else 0,
        call_code,
        cpn_code,
        opt_code,
        1 if allow_autocall_on_expiry else 0,
        1 if is_memory else 0,
        1 if force_coupons_on_autocall else 0,
        1 if no_notional_repayment else 0,
        1 if is_long_basket else 0,
        0,
        arr_ptr,
        arr_ptr + m_bytes,
        dw_all.data_ptr(),
        arr_ptr + 2 * m_bytes,
        arr_ptr + 3 * m_bytes,
        arr_ptr + 4 * m_bytes,
        arr_ptr + 5 * m_bytes,
        sc_buf.data_ptr(),
    )
    backend._fn_struct(packed)
    f_w_t, v_w_t, p_t, surv_t, h_t, q_t = torch.unbind(arr_6m, 0)
    dw_t = dw_all.squeeze(1) if m_rem == 1 else dw_all[:, -1].contiguous()
    if one_star_strike == 0.0 and part_call_rate == 0.0:
        s0, s1, s2, s3, s4, s5 = torch.unbind(sc_buf[:6], 0)
        s6 = s5
        s7 = s5
    else:
        s0, s1, s2, s3, s4, s5, s6, s7 = torch.unbind(sc_buf, 0)
    return AutocallBaselineOutput(
        f_w=f_w_t,
        v_w=v_w_t,
        h_proxy=h_t,
        p_proxy=p_t,
        survival_proxy=surv_t,
        q_proxy=q_t,
        l_put_proxy=s0,
        proxy_delta_weights=dw_t,
        option_pv_proxy=s1,
        notional_pv_proxy=s2,
        coupon_pv_proxy=s3,
        product_value_proxy=s4,
        l_call_proxy=s5,
        l_onestar_proxy=s6,
        p_onestar_proxy=s7,
        has_unsupported_exotic_index=has_unsupported,
        unsupported_indices=unsupported_tup,
    )


_UNPACK_DUAL = getattr(torch.autograd.forward_ad, "unpack_dual", None)


def _is_ad_active(*items: Any) -> bool:
    if torch._C._are_functorch_transforms_active():
        return True
    unpack_fn = _UNPACK_DUAL
    for t in items:
        if isinstance(t, torch.Tensor):
            if t.requires_grad:
                return True
            if unpack_fn is not None and unpack_fn(t).tangent is not None:
                return True
    return False


def _has_near_tied_forwards(fp_t: torch.Tensor) -> bool:
    if fp_t.ndim != 2 or fp_t.device.type != "cpu" or _is_ad_active(fp_t):
        return False
    d_assets, m_rem = int(fp_t.shape[0]), int(fp_t.shape[1])
    if d_assets < 2 or m_rem == 0:
        return False
    arr = _tensor_to_np_ro(fp_t)
    if m_rem == 1:
        col = sorted(max(float(x), 1e-16) for x in arr[:, 0].tolist())
        return any(col[i + 1] < col[i] * 1.0618365465453596 for i in range(d_assets - 1))
    s_arr = np.sort(np.maximum(arr, 1e-16), axis=0)
    return bool(np.any(s_arr[1:] < s_arr[:-1] * 1.0618365465453596))


def compute_analytical_autocall_baseline(
    preproc_state: Any,
    *,
    use_fast_cpu: bool | None = None,
    coupon_index: str | None = None,
    option_index: str | None = None,
    lower_strike: float = 0.0,
    allow_autocall_on_expiry: bool = False,
    inv_lex_perm: Sequence[int] | npt.NDArray[np.int64] | None = None,
    coupon_barrier_type: str = "In",
    coupon_barrier_direction: str = "Up",
) -> AutocallBaselineOutput:
    """Evaluates the exact analytical moment-matched baseline across all future periods.

    Uses closed-form truncated-normal + ramp survival moments via `erfcx` Inverse Mills Ratio,
    Euler-weighted inter-period basket correlation, distinct multi-index conditioning, exact
    piecewise terminal put/call integrals, and 16-point bivariate Gauss-Legendre `OneStar`.
    Stateless across calls (no mutable buffer caching).
    """
    is_mapping = isinstance(preproc_state, dict) or (
        not hasattr(preproc_state, "forward_perf") and isinstance(preproc_state, Mapping)
    )
    if not is_mapping and use_fast_cpu is not False:
        fp_t: torch.Tensor = preproc_state.forward_perf
        vp_t: torch.Tensor = preproc_state.v_perf
        cm_t: torch.Tensor = preproc_state.corr_matrix
        df_t: torch.Tensor = preproc_state.df_pay
        ko_t: torch.Tensor = preproc_state.ko_barriers
        ic_t: torch.Tensor = preproc_state.is_callable
        cb_t: torch.Tensor = preproc_state.cpn_barriers
        ig_t: torch.Tensor = preproc_state.is_guaranteed_cpn
        cr_t: torch.Tensor = preproc_state.coupon_amounts
        is_preproc = hasattr(preproc_state, "baseline")
        can_cpu = (
            fp_t.ndim == 2
            and fp_t.is_cpu
            and vp_t.is_cpu
            and cm_t.is_cpu
            and df_t.is_cpu
            and ko_t.is_cpu
            and ic_t.is_cpu
            and cb_t.is_cpu
            and ig_t.is_cpu
            and cr_t.is_cpu
            and df_t.ndim == 1
            and ko_t.ndim == 1
            and ic_t.ndim == 1
            and cb_t.ndim == 1
            and ig_t.ndim == 1
            and cr_t.ndim == 1
            and (
                use_fast_cpu is True
                or not _is_ad_active(
                    fp_t, vp_t, cm_t, df_t, ko_t, ic_t, cb_t, ig_t, cr_t,
                    preproc_state.tau_obs if is_preproc else getattr(preproc_state, "tau_obs", None),
                )
            )
        )
        if can_cpu:
            tau_raw = preproc_state.tau_obs if is_preproc else getattr(preproc_state, "tau_obs", None)
            if is_preproc:
                call_idx = preproc_state.callability_index or "WorstOf"
                raw_cpn_idx = coupon_index if coupon_index is not None else preproc_state.coupon_index
                raw_opt_idx = option_index if option_index is not None else preproc_state.option_index
                already_autocalled = (
                    False if preproc_state.already_autocalled is None else bool(preproc_state.already_autocalled)
                )
                pv_in_transit = 0.0 if preproc_state.pv_in_transit is None else float(preproc_state.pv_in_transit)
                is_long_flag = False if preproc_state.is_long_basket is None else bool(preproc_state.is_long_basket)
            else:
                call_idx = str(getattr(preproc_state, "callability_index", "WorstOf") or "WorstOf")
                raw_cpn_idx = coupon_index if coupon_index is not None else getattr(preproc_state, "coupon_index", None)
                raw_opt_idx = option_index if option_index is not None else getattr(preproc_state, "option_index", None)
                already_autocalled = _coalesce_bool(getattr(preproc_state, "already_autocalled", False), False)
                pv_in_transit = _coalesce_float(getattr(preproc_state, "pv_in_transit", 0.0), 0.0)
                is_long_flag = _coalesce_bool(getattr(preproc_state, "is_long_basket", False), False)

            call_code = 0 if call_idx == "WorstOf" else monitoring_index_to_code(call_idx)
            if raw_cpn_idx is None:
                cpn_idx = call_idx
                cpn_code = call_code
            else:
                cpn_s = str(raw_cpn_idx).strip()
                if cpn_s:
                    cpn_idx = cpn_s
                    cpn_code = 0 if cpn_idx == "WorstOf" else monitoring_index_to_code(cpn_idx)
                else:
                    cpn_idx = call_idx
                    cpn_code = call_code
            if raw_opt_idx is None:
                if is_long_flag:
                    opt_idx = "AverageOf"
                    opt_code = 2
                else:
                    opt_idx = call_idx
                    opt_code = call_code
            else:
                opt_s = str(raw_opt_idx).strip()
                if not opt_s:
                    if is_long_flag:
                        opt_idx = "AverageOf"
                        opt_code = 2
                    else:
                        opt_idx = call_idx
                        opt_code = call_code
                else:
                    base_opt_code = 0 if opt_s == "WorstOf" else monitoring_index_to_code(opt_s)
                    if is_long_flag and base_opt_code == 0:
                        opt_idx = "AverageOf"
                        opt_code = 2
                    else:
                        opt_idx = opt_s
                        opt_code = base_opt_code

            if call_code <= 2 and cpn_code <= 2 and opt_code <= 2:
                has_unsupported = False
                unsupported_tup: Tuple[str, ...] = ()
            else:
                unsupported_list: List[str] = []
                for idx_str, idx_c in ((call_idx, call_code), (cpn_idx, cpn_code), (opt_idx, opt_code)):
                    if not is_exact_analytical_index(idx_c) and idx_str not in unsupported_list:
                        unsupported_list.append(idx_str)
                has_unsupported = len(unsupported_list) > 0
                unsupported_tup = tuple(unsupported_list)

            d_assets, m_rem = int(fp_t.shape[0]), int(fp_t.shape[1])
            if m_rem == 0 or already_autocalled or d_assets == 0:
                device = fp_t.device
                z_m = torch.zeros(m_rem, dtype=torch.float64, device=device)
                z_s = torch.tensor(0.0, dtype=torch.float64, device=device)
                return AutocallBaselineOutput(
                    f_w=z_m,
                    v_w=z_m.clone(),
                    h_proxy=z_m.clone(),
                    p_proxy=z_m.clone(),
                    survival_proxy=z_m.clone(),
                    q_proxy=z_m.clone(),
                    l_put_proxy=z_s,
                    proxy_delta_weights=torch.zeros(d_assets, dtype=torch.float64, device=device),
                    option_pv_proxy=z_s.clone(),
                    notional_pv_proxy=z_s.clone(),
                    coupon_pv_proxy=z_s.clone(),
                    product_value_proxy=torch.tensor(pv_in_transit, dtype=torch.float64, device=device),
                    l_call_proxy=z_s.clone(),
                    l_onestar_proxy=z_s.clone(),
                    p_onestar_proxy=z_s.clone(),
                    has_unsupported_exotic_index=has_unsupported,
                    unsupported_indices=unsupported_tup,
                )

            if is_preproc:
                raw_ls_state = preproc_state.lower_strike
                if lower_strike != 0.0:
                    eff_lower_strike = float(lower_strike)
                    has_ls_flag = math.isfinite(eff_lower_strike)
                elif raw_ls_state is not None:
                    eff_lower_strike = float(raw_ls_state)
                    has_ls_flag = math.isfinite(eff_lower_strike)
                else:
                    eff_lower_strike = 0.0
                    has_ls_flag = False
                eff_allow_exp = bool(allow_autocall_on_expiry) or preproc_state.allow_autocall_on_expiry
                eff_perm = inv_lex_perm if inv_lex_perm is not None else preproc_state.inv_lex_perm
                eff_cbt = coupon_barrier_type if coupon_barrier_type != "In" else (preproc_state.coupon_barrier_type or "In")
                eff_cbd = coupon_barrier_direction if coupon_barrier_direction != "Up" else (preproc_state.coupon_barrier_direction or "Up")
                ki_raw = preproc_state.ki_barrier
                ki_val = float(ki_raw) if ki_raw is not None else 0.0
                part_raw = preproc_state.participation_call_strike
                os_raw = preproc_state.one_star_strike
                strike_v = 0.90 if preproc_state.strike is None else float(preproc_state.strike)
                i_ki_past_v = 0.0 if preproc_state.i_ki_past is None else float(preproc_state.i_ki_past)
                ko_sp_v = -0.15 if preproc_state.ko_spread is None else float(preproc_state.ko_spread)
                cpn_sp_v = -0.15 if preproc_state.cpn_spread is None else float(preproc_state.cpn_spread)
                ki_sp_v = -0.15 if preproc_state.ki_spread is None else float(preproc_state.ki_spread)
                c_mem_0_v = 0.0 if preproc_state.c_mem_0 is None else float(preproc_state.c_mem_0)
                is_mem_v = True if preproc_state.is_memory is None else bool(preproc_state.is_memory)
                ac_cpn_v = 0.0 if preproc_state.autocall_coupon is None else float(preproc_state.autocall_coupon)
                force_cpn_v = (
                    False
                    if preproc_state.force_coupons_on_autocall is None
                    else bool(preproc_state.force_coupons_on_autocall)
                )
                no_not_v = (
                    False if preproc_state.no_notional_repayment is None else bool(preproc_state.no_notional_repayment)
                )
                is_long_v = is_long_flag
                opt_gear_v = 1.11111 if preproc_state.option_gearing is None else float(preproc_state.option_gearing)
                part_rate_v = (
                    0.0 if preproc_state.participation_call_rate is None else float(preproc_state.participation_call_rate)
                )
                os_cpn_v = 0.0 if preproc_state.one_star_coupon is None else float(preproc_state.one_star_coupon)
            else:
                raw_ls_state = getattr(preproc_state, "lower_strike", None)
                if lower_strike != 0.0:
                    eff_lower_strike = float(lower_strike)
                    has_ls_flag = math.isfinite(eff_lower_strike)
                elif raw_ls_state is not None:
                    eff_lower_strike = float(raw_ls_state)
                    has_ls_flag = math.isfinite(eff_lower_strike)
                else:
                    eff_lower_strike = 0.0
                    has_ls_flag = False
                eff_allow_exp = bool(allow_autocall_on_expiry) or _coalesce_bool(
                    getattr(preproc_state, "allow_autocall_on_expiry", False), False
                )
                eff_perm = inv_lex_perm if inv_lex_perm is not None else getattr(preproc_state, "inv_lex_perm", None)
                state_cbt = getattr(preproc_state, "coupon_barrier_type", "In") or "In"
                state_cbd = getattr(preproc_state, "coupon_barrier_direction", "Up") or "Up"
                eff_cbt = str(coupon_barrier_type if coupon_barrier_type != "In" else state_cbt)
                eff_cbd = str(coupon_barrier_direction if coupon_barrier_direction != "Up" else state_cbd)
                ki_raw = getattr(preproc_state, "ki_barrier", None)
                ki_val = float(ki_raw) if ki_raw is not None else 0.0
                part_raw = getattr(preproc_state, "participation_call_strike", None)
                os_raw = getattr(preproc_state, "one_star_strike", None)
                strike_v = _coalesce_float(getattr(preproc_state, "strike", 0.90), 0.90)
                i_ki_past_v = _coalesce_float(getattr(preproc_state, "i_ki_past", 0.0), 0.0)
                ko_sp_v = _coalesce_float(getattr(preproc_state, "ko_spread", -0.15), -0.15)
                cpn_sp_v = _coalesce_float(getattr(preproc_state, "cpn_spread", -0.15), -0.15)
                ki_sp_v = _coalesce_float(getattr(preproc_state, "ki_spread", -0.15), -0.15)
                c_mem_0_v = _coalesce_float(getattr(preproc_state, "c_mem_0", 0.0), 0.0)
                is_mem_v = _coalesce_bool(getattr(preproc_state, "is_memory", True), True)
                ac_cpn_v = _coalesce_float(getattr(preproc_state, "autocall_coupon", 0.0), 0.0)
                force_cpn_v = _coalesce_bool(getattr(preproc_state, "force_coupons_on_autocall", False), False)
                no_not_v = _coalesce_bool(getattr(preproc_state, "no_notional_repayment", False), False)
                is_long_v = _coalesce_bool(getattr(preproc_state, "is_long_basket", False), False)
                opt_gear_v = _coalesce_float(getattr(preproc_state, "option_gearing", 1.11111), 1.11111)
                part_rate_v = _coalesce_float(getattr(preproc_state, "participation_call_rate", 0.0), 0.0)
                os_cpn_v = _coalesce_float(getattr(preproc_state, "one_star_coupon", 0.0), 0.0)

            diff_nonpos_ls = (9 <= opt_code <= 13) and has_ls_flag and (eff_lower_strike <= 0.0)
            part_strike_c = float(part_raw) if part_raw is not None else -1.0
            if not diff_nonpos_ls:
                backend = get_tier1_backend()
                if hasattr(backend, "_fn_struct"):
                    cbt_i = 0 if eff_cbt == "In" else (1 if (eff_cbt == "Out" or eff_cbt.strip().lower() == "out") else 0)
                    cbd_i = 0 if eff_cbd == "Up" else (1 if (eff_cbd == "Down" or eff_cbd.strip().lower() == "down") else 0)
                    return _baseline_cpu_via_cpp(
                        backend,
                        d_assets,
                        m_rem,
                        fp_t,
                        vp_t,
                        cm_t,
                        tau_raw,
                        df_t,
                        ko_t,
                        ic_t,
                        cb_t,
                        ig_t,
                        cr_t,
                        cbt_i,
                        cbd_i,
                        eff_perm,
                        strike_v,
                        eff_lower_strike,
                        ki_val,
                        ki_raw is not None and ki_val > 0.0,
                        i_ki_past_v,
                        ko_sp_v,
                        cpn_sp_v,
                        ki_sp_v,
                        call_code,
                        cpn_code,
                        opt_code,
                        eff_allow_exp,
                        c_mem_0_v,
                        is_mem_v,
                        ac_cpn_v,
                        force_cpn_v,
                        no_not_v,
                        is_long_v,
                        opt_gear_v,
                        part_strike_c,
                        part_rate_v,
                        float(os_raw) if os_raw is not None else 0.0,
                        os_cpn_v,
                        pv_in_transit,
                        has_unsupported,
                        unsupported_tup,
                    )

                eval_fn = getattr(backend, "eval_autocall_baseline_cpu", None)
                if eval_fn is not None:
                    tau_np = (
                        _tensor_to_np_ro(tau_raw)
                        if isinstance(tau_raw, torch.Tensor)
                        else (np.asarray(tau_raw, dtype=np.float64) if tau_raw is not None else _EMPTY_F64)
                    )
                    perm_np = _perm_to_np(eff_perm, d_assets)
                    f_w_np, v_w_np, dw_np, p_np, surv_np, h_np, q_np, sc_np = eval_fn(
                        fwd_perf=_tensor_to_np_ro(fp_t),
                        vol_perf=_tensor_to_np_ro(vp_t),
                        corr=_tensor_to_np_ro(cm_t),
                        tau_obs=tau_np,
                        df_pay=_tensor_to_np_ro(df_t),
                        ko_barriers=_tensor_to_np_ro(ko_t),
                        is_callable=_tensor_to_np_ro(ic_t),
                        cpn_barriers=_tensor_to_np_ro(cb_t),
                        is_guaranteed_cpn=_tensor_to_np_ro(ig_t),
                        coupon_rates=_tensor_to_np_ro(cr_t),
                        coupon_barrier_type=eff_cbt,
                        coupon_barrier_direction=eff_cbd,
                        inv_lex_perm=perm_np,
                        strike=strike_v,
                        lower_strike=eff_lower_strike,
                        ki_barrier=ki_val,
                        has_ki=ki_raw is not None and ki_val > 0.0,
                        i_ki_past=i_ki_past_v,
                        ko_spread=ko_sp_v,
                        cpn_spread=cpn_sp_v,
                        ki_spread=ki_sp_v,
                        call_index_code=call_code,
                        cpn_index_code=cpn_code,
                        opt_index_code=opt_code,
                        allow_autocall_on_expiry=eff_allow_exp,
                        c_mem_0=c_mem_0_v,
                        is_memory=is_mem_v,
                        autocall_coupon=ac_cpn_v,
                        force_coupons_on_autocall=force_cpn_v,
                        no_notional_repayment=no_not_v,
                        is_long_basket=is_long_v,
                        option_gearing=opt_gear_v,
                        part_call_strike=part_strike_c,
                        part_call_rate=part_rate_v,
                        one_star_strike=float(os_raw) if os_raw is not None else 0.0,
                        one_star_coupon=os_cpn_v,
                        pv_in_transit=pv_in_transit,
                    )
                    sc_t = torch.from_numpy(sc_np)
                    dw_col = torch.from_numpy(dw_np)[:, -1]
                    if m_rem > 1:
                        dw_col = dw_col.contiguous()
                    sc6 = sc_t[6]
                    return AutocallBaselineOutput(
                        f_w=torch.from_numpy(f_w_np),
                        v_w=torch.from_numpy(v_w_np),
                        h_proxy=torch.from_numpy(h_np),
                        p_proxy=torch.from_numpy(p_np),
                        survival_proxy=torch.from_numpy(surv_np),
                        q_proxy=torch.from_numpy(q_np),
                        l_put_proxy=sc_t[0],
                        proxy_delta_weights=dw_col,
                        option_pv_proxy=sc_t[1],
                        notional_pv_proxy=sc_t[2],
                        coupon_pv_proxy=sc_t[3],
                        product_value_proxy=sc_t[4],
                        l_call_proxy=sc_t[5],
                        l_onestar_proxy=sc6,
                        p_onestar_proxy=sc_t[7] if (os_raw is not None and float(os_raw) > 0.0) else sc6,
                        has_unsupported_exotic_index=has_unsupported,
                        unsupported_indices=unsupported_tup,
                    )

    fields = _extract_state_fields(
        preproc_state,
        coupon_index=coupon_index,
        option_index=option_index,
        lower_strike=lower_strike,
        allow_autocall_on_expiry=allow_autocall_on_expiry,
        inv_lex_perm=inv_lex_perm,
        coupon_barrier_type=coupon_barrier_type,
        coupon_barrier_direction=coupon_barrier_direction,
    )

    fp_t = fields["forward_perf"]
    vp_t = fields["v_perf"]
    cm_t = fields["corr_matrix"]
    df_t = fields["df_pay"]
    ko_t = fields["ko_barriers"]
    ic_t = fields["is_callable"]
    cb_t = fields["cpn_barriers"]
    ig_t = fields["is_guaranteed_cpn"]
    cr_t = fields["coupon_amounts"]
    tau_raw = fields["tau_obs"]

    call_idx = fields["callability_index"]
    cpn_idx = fields["coupon_index"]
    opt_idx = fields["option_index"]
    call_code = monitoring_index_to_code(call_idx)
    cpn_code = monitoring_index_to_code(cpn_idx)
    opt_code = monitoring_index_to_code(opt_idx)

    unsupported_list = []
    for idx_str, idx_c in ((call_idx, call_code), (cpn_idx, cpn_code), (opt_idx, opt_code)):
        if not is_exact_analytical_index(idx_c) and idx_str not in unsupported_list:
            unsupported_list.append(idx_str)
    has_unsupported = len(unsupported_list) > 0
    unsupported_tup = tuple(unsupported_list)

    device = fp_t.device
    any_grad = _is_ad_active(fp_t, vp_t, cm_t, df_t, ko_t, ic_t, cb_t, ig_t, cr_t, tau_raw)
    all_cpu_1d_sched = (
        fp_t.ndim == 2
        and all(t.device.type == "cpu" for t in (fp_t, vp_t, cm_t, df_t, ko_t, ic_t, cb_t, ig_t, cr_t))
        and all(t.ndim == 1 for t in (df_t, ko_t, ic_t, cb_t, ig_t, cr_t))
    )
    can_fast_cpu = all_cpu_1d_sched and (use_fast_cpu is True or (use_fast_cpu is None and not any_grad))
    diff_nonpos_ls = (9 <= opt_code <= 13) and bool(fields["has_lower_strike"]) and (float(fields["lower_strike"]) <= 0.0)

    if can_fast_cpu and not diff_nonpos_ls:
        d_assets, m_rem = int(fp_t.shape[0]), int(fp_t.shape[1])
        already_autocalled = bool(fields["already_autocalled"])
        pv_in_transit = float(fields["pv_in_transit"])
        if m_rem == 0 or already_autocalled or d_assets == 0:
            z_m = torch.zeros(m_rem, dtype=torch.float64, device=device)
            z_s = torch.tensor(0.0, dtype=torch.float64, device=device)
            pv_t = torch.tensor(pv_in_transit, dtype=torch.float64, device=device)
            return AutocallBaselineOutput(
                f_w=z_m,
                v_w=z_m.clone(),
                h_proxy=z_m.clone(),
                p_proxy=z_m.clone(),
                survival_proxy=z_m.clone(),
                q_proxy=z_m.clone(),
                l_put_proxy=z_s,
                proxy_delta_weights=torch.zeros(d_assets, dtype=torch.float64, device=device),
                option_pv_proxy=z_s.clone(),
                notional_pv_proxy=z_s.clone(),
                coupon_pv_proxy=z_s.clone(),
                product_value_proxy=pv_t,
                l_call_proxy=z_s.clone(),
                l_onestar_proxy=z_s.clone(),
                p_onestar_proxy=z_s.clone(),
                has_unsupported_exotic_index=has_unsupported,
                unsupported_indices=unsupported_tup,
            )

        tau_np = (
            _tensor_to_np_ro(tau_raw)
            if isinstance(tau_raw, torch.Tensor)
            else (np.asarray(tau_raw, dtype=np.float64) if tau_raw is not None else _EMPTY_F64)
        )
        perm_np = _perm_to_np(fields["inv_lex_perm"], d_assets)

        backend = get_tier1_backend()
        eval_fn = getattr(backend, "eval_autocall_baseline_cpu", None)
        if eval_fn is not None:
            f_w_np, v_w_np, dw_np, p_np, surv_np, h_np, q_np, sc_np = eval_fn(
                fwd_perf=_tensor_to_np_ro(fp_t),
                vol_perf=_tensor_to_np_ro(vp_t),
                corr=_tensor_to_np_ro(cm_t),
                tau_obs=tau_np,
                df_pay=_tensor_to_np_ro(df_t),
                ko_barriers=_tensor_to_np_ro(ko_t),
                is_callable=_tensor_to_np_ro(ic_t),
                cpn_barriers=_tensor_to_np_ro(cb_t),
                is_guaranteed_cpn=_tensor_to_np_ro(ig_t),
                coupon_rates=_tensor_to_np_ro(cr_t),
                coupon_barrier_type=fields["coupon_barrier_type"],
                coupon_barrier_direction=fields["coupon_barrier_direction"],
                inv_lex_perm=perm_np,
                strike=fields["strike"],
                lower_strike=fields["lower_strike"],
                ki_barrier=fields["ki_barrier"],
                has_ki=fields["has_ki"],
                i_ki_past=fields["i_ki_past"],
                ko_spread=fields["ko_spread"],
                cpn_spread=fields["cpn_spread"],
                ki_spread=fields["ki_spread"],
                call_index_code=call_code,
                cpn_index_code=cpn_code,
                opt_index_code=opt_code,
                allow_autocall_on_expiry=fields["allow_autocall_on_expiry"],
                c_mem_0=fields["c_mem_0"],
                is_memory=fields["is_memory"],
                autocall_coupon=fields["autocall_coupon"],
                force_coupons_on_autocall=fields["force_coupons_on_autocall"],
                no_notional_repayment=fields["no_notional_repayment"],
                is_long_basket=fields["is_long_basket"],
                option_gearing=fields["option_gearing"],
                part_call_strike=fields["part_call_strike"],
                part_call_rate=fields["part_call_rate"],
                one_star_strike=fields["one_star_strike"],
                one_star_coupon=fields["one_star_coupon"],
                pv_in_transit=pv_in_transit,
            )
            sc_t = torch.from_numpy(sc_np)
            dw_col = torch.from_numpy(dw_np)[:, -1]
            if m_rem > 1:
                dw_col = dw_col.contiguous()
            return AutocallBaselineOutput(
                f_w=torch.from_numpy(f_w_np),
                v_w=torch.from_numpy(v_w_np),
                h_proxy=torch.from_numpy(h_np),
                p_proxy=torch.from_numpy(p_np),
                survival_proxy=torch.from_numpy(surv_np),
                q_proxy=torch.from_numpy(q_np),
                l_put_proxy=sc_t[0],
                proxy_delta_weights=dw_col,
                option_pv_proxy=sc_t[1],
                notional_pv_proxy=sc_t[2],
                coupon_pv_proxy=sc_t[3],
                product_value_proxy=sc_t[4],
                l_call_proxy=sc_t[5],
                l_onestar_proxy=sc_t[6],
                p_onestar_proxy=sc_t[7] if sc_t.numel() > 7 else torch.zeros_like(sc_t[6]),
                has_unsupported_exotic_index=has_unsupported,
                unsupported_indices=unsupported_tup,
            )

    # Differentiable PyTorch float64 path (2D single-trade or 3D batched)
    is_batched = fp_t.ndim == 3
    if not is_batched:
        forward_perf = fp_t.unsqueeze(0)
        v_perf = vp_t.unsqueeze(0)
    else:
        forward_perf = fp_t
        v_perf = vp_t
    cm_b = cm_t.unsqueeze(0).expand(int(forward_perf.shape[0]), int(forward_perf.shape[1]), int(forward_perf.shape[1])) if cm_t.ndim == 2 else cm_t

    b_sz, d_assets, m_rem = forward_perf.shape
    already_autocalled = bool(fields["already_autocalled"])
    pv_in_transit = float(fields["pv_in_transit"])
    if m_rem == 0 or already_autocalled or d_assets == 0 or b_sz == 0:
        zero_ad_s = 0.0 * (fp_t.sum() + vp_t.sum() + cm_t.sum())
        if is_batched:
            z_m_b = torch.zeros((b_sz, m_rem), dtype=torch.float64, device=device) + zero_ad_s
            z_s_b = torch.zeros(b_sz, dtype=torch.float64, device=device) + zero_ad_s
            pv_t_b = z_s_b + pv_in_transit
            return AutocallBaselineOutput(
                f_w=z_m_b,
                v_w=z_m_b.clone(),
                h_proxy=z_m_b.clone(),
                p_proxy=z_m_b.clone(),
                survival_proxy=z_m_b.clone(),
                q_proxy=z_m_b.clone(),
                l_put_proxy=z_s_b,
                proxy_delta_weights=torch.zeros((b_sz, d_assets), dtype=torch.float64, device=device),
                option_pv_proxy=z_s_b.clone(),
                notional_pv_proxy=z_s_b.clone(),
                coupon_pv_proxy=z_s_b.clone(),
                product_value_proxy=pv_t_b,
                l_call_proxy=z_s_b.clone(),
                l_onestar_proxy=z_s_b.clone(),
                p_onestar_proxy=z_s_b.clone(),
                has_unsupported_exotic_index=has_unsupported,
                unsupported_indices=unsupported_tup,
            )
        z_m = torch.zeros(m_rem, dtype=torch.float64, device=device) + zero_ad_s
        z_s = torch.zeros((), dtype=torch.float64, device=device) + zero_ad_s
        pv_t = z_s + pv_in_transit
        return AutocallBaselineOutput(
            f_w=z_m,
            v_w=z_m.clone(),
            h_proxy=z_m.clone(),
            p_proxy=z_m.clone(),
            survival_proxy=z_m.clone(),
            q_proxy=z_m.clone(),
            l_put_proxy=z_s,
            proxy_delta_weights=torch.zeros(d_assets, dtype=torch.float64, device=device),
            option_pv_proxy=z_s.clone(),
            notional_pv_proxy=z_s.clone(),
            coupon_pv_proxy=z_s.clone(),
            product_value_proxy=pv_t,
            l_call_proxy=z_s.clone(),
            l_onestar_proxy=z_s.clone(),
            p_onestar_proxy=z_s.clone(),
            has_unsupported_exotic_index=has_unsupported,
            unsupported_indices=unsupported_tup,
        )

    def _sched_col(sched: torch.Tensor, idx_m: int) -> torch.Tensor:
        if sched.ndim >= 2:
            return sched[:, idx_m].to(dtype=torch.float64, device=device)
        return sched[idx_m].to(dtype=torch.float64, device=device).expand(b_sz)

    raw_perm = fields["inv_lex_perm"]
    perm_seq: Optional[Sequence[int]] = (
        raw_perm.detach().cpu().reshape(-1).tolist()
        if isinstance(raw_perm, torch.Tensor)
        else (raw_perm.tolist() if isinstance(raw_perm, np.ndarray) else raw_perm)
    )
    f_batch_all = forward_perf.transpose(1, 2).reshape(b_sz * m_rem, d_assets)
    v_batch_all = v_perf.transpose(1, 2).reshape(b_sz * m_rem, d_assets)
    c_batch_all = cm_b.unsqueeze(1).expand(b_sz, m_rem, d_assets, d_assets).reshape(b_sz * m_rem, d_assets, d_assets)
    f_w_flat, v_w_flat, w_all_flat = stulz_margrabe_moments_with_weights(
        f_perf=f_batch_all,
        v_perf=v_batch_all,
        corr=c_batch_all,
        asset_mask=None,
        monitoring_index=call_idx,
        inv_lex_perm=perm_seq,
    )
    f_w = f_w_flat.view(b_sz, m_rem)
    v_w = v_w_flat.view(b_sz, m_rem)
    w_call_all = w_all_flat.view(b_sz, m_rem, d_assets)
    proxy_delta_weights = w_call_all[:, -1, :]

    call_is_normal = 9 <= call_code <= 13
    cpn_is_normal = 9 <= cpn_code <= 13
    opt_is_normal = 9 <= opt_code <= 13
    cpn_differs = cpn_code != call_code
    opt_differs = opt_code != call_code
    cpn_b_type: str = fields["coupon_barrier_type"]
    cpn_b_dir: str = fields["coupon_barrier_direction"]
    cpn_dir_down = cpn_b_dir.strip().lower() == "down"
    cpn_type_out = cpn_b_type.strip().lower() == "out"
    force_cpn = bool(fields["force_coupons_on_autocall"])
    ko_spread = float(fields["ko_spread"])
    cpn_spread = float(fields["cpn_spread"])
    ki_spread = float(fields["ki_spread"])

    if cpn_differs:
        f_cpn_flat, v_cpn_flat, w_cpn_flat = stulz_margrabe_moments_with_weights(
            f_perf=f_batch_all,
            v_perf=v_batch_all,
            corr=c_batch_all,
            asset_mask=None,
            monitoring_index=cpn_idx,
            inv_lex_perm=perm_seq,
        )
        f_cpn_raw = f_cpn_flat.view(b_sz, m_rem)
        v_cpn_raw = v_cpn_flat.view(b_sz, m_rem)
        w_cpn_all = w_cpn_flat.view(b_sz, m_rem, d_assets)
        rho_cc_all = _compute_cross_index_corr_torch(
            forward_perf.transpose(1, 2),
            v_perf.transpose(1, 2),
            cm_b.unsqueeze(1),
            call_code,
            w_call_all,
            cpn_code,
            w_cpn_all,
        )
    else:
        f_cpn_raw = f_w
        v_cpn_raw = v_w
        rho_cc_all = torch.ones((b_sz, m_rem), dtype=torch.float64, device=device)

    z_mean = torch.zeros(b_sz, dtype=torch.float64, device=device)
    z_std = torch.ones(b_sz, dtype=torch.float64, device=device)
    surv_cur = torch.ones(b_sz, dtype=torch.float64, device=device)

    h_list: List[torch.Tensor] = []
    p_list: List[torch.Tensor] = []
    surv_list: List[torch.Tensor] = []
    surv_prev_list: List[torch.Tensor] = []
    q_list: List[torch.Tensor] = []

    z_mean_term = z_mean
    z_std_term = z_std
    f_cond_term = f_w[:, -1]
    v_cond_term = v_w[:, -1]

    for m in range(m_rem):
        fw_m = f_w[:, m] if call_is_normal else f_w[:, m].clamp(min=1e-16)
        vw_m = v_w[:, m].clamp(min=1e-12)
        if m == 0:
            f_cond_m = fw_m
            v_cond_m = vw_m
        elif call_is_normal:
            f_cond_m = fw_m + vw_m * z_mean
            v_cond_m = (vw_m * z_std).clamp(min=1e-12)
        else:
            f_cond_m = fw_m * torch.exp(vw_m * z_mean - 0.5 * (vw_m * vw_m) * (1.0 - z_std * z_std))
            v_cond_m = (vw_m * z_std).clamp(min=1e-12)

        if m == m_rem - 1:
            f_cond_term = f_cond_m
            v_cond_term = v_cond_m
            z_mean_term = z_mean
            z_std_term = z_std

        ko_b_m = _sched_col(ko_t, m)
        call_m = _sched_col(ic_t, m)
        is_call_mask = call_m > 0.5
        if call_code == 14:
            ko_raw_m = _eval_coupon_prob_torch(
                f_cond_m, v_cond_m, ko_b_m, ko_spread, direction="Up", b_type="In", index_code=14
            )
        elif call_is_normal:
            ko_raw_m = _digital_call_spread_normal_torch(f_cond_m, v_cond_m, ko_b_m, ko_spread)
        else:
            ko_raw_m = _digital_call_spread_prob_torch(f_cond_m, v_cond_m, ko_b_m, ko_spread)
        h_raw_m = torch.where(is_call_mask, ko_raw_m.clamp(0.0, 1.0), torch.zeros_like(ko_raw_m))
        h_m = torch.where(is_call_mask, call_m * ko_raw_m.clamp(0.0, 1.0 - 1e-7), torch.zeros_like(ko_raw_m))

        p_m = surv_cur * h_m
        surv_next = surv_cur * (1.0 - h_m)

        if cpn_differs:
            fc_raw = f_cpn_raw[:, m]
            vc_raw = v_cpn_raw[:, m].clamp(min=1e-12)
            rho_cc = rho_cc_all[:, m]
            z_mc = rho_cc * z_mean
            z_vc = rho_cc * rho_cc * (z_std * z_std) + (1.0 - rho_cc * rho_cc)
            z_sc = torch.sqrt(z_vc.clamp(min=1e-12))
            rho_ck_cond = (rho_cc * z_std / z_sc).clamp(-0.999999, 0.999999)
            if cpn_is_normal:
                f_cpn_cond = fc_raw + vc_raw * z_mc
                v_cpn_cond = (vc_raw * z_sc).clamp(min=1e-12)
            else:
                f_cpn_cond = fc_raw.clamp(min=1e-16) * torch.exp(vc_raw * z_mc - 0.5 * (vc_raw * vc_raw) * (1.0 - z_vc))
                v_cpn_cond = (vc_raw * z_sc).clamp(min=1e-12)
        else:
            f_cpn_cond = f_cond_m
            v_cpn_cond = v_cond_m
            rho_ck_cond = torch.ones(b_sz, dtype=torch.float64, device=device)

        cb_m = _sched_col(cb_t, m)
        q_base = _eval_coupon_prob_torch(
            f_cpn_cond,
            v_cpn_cond,
            cb_m,
            cpn_spread,
            direction=cpn_b_dir,
            b_type=cpn_b_type,
            is_normal=cpn_is_normal,
            index_code=cpn_code,
        ).clamp(0.0, 1.0)
        if force_cpn:
            h_prob = h_raw_m.clamp(0.0, 1.0)
            s_c = -1.0 if (cpn_dir_down != cpn_type_out) else 1.0
            z_c = torch.special.ndtri(q_base.clamp(1e-12, 1.0 - 1e-12))
            z_k = torch.special.ndtri(h_prob.clamp(1e-12, 1.0 - 1e-12))
            p_ck = _bvn_cdf_genz_torch(z_c, z_k, s_c * rho_ck_cond)
            q_comb = torch.minimum(
                torch.maximum(q_base + h_prob - p_ck, torch.maximum(q_base, h_prob)),
                (q_base + h_prob).clamp(max=1.0),
            )
            q_forced = torch.where(
                (h_prob >= 1.0 - 1e-12) | (q_base >= 1.0 - 1e-12),
                torch.ones_like(q_base),
                torch.where(
                    h_prob <= 1e-12,
                    q_base,
                    torch.where(q_base <= 1e-12, h_prob, q_comb),
                ),
            )
            q_raw_m = torch.where(is_call_mask, q_forced, q_base)
        else:
            q_raw_m = q_base
        ig_m = _sched_col(ig_t, m)
        q_m = torch.where(ig_m > 0.5, torch.ones_like(q_raw_m), q_raw_m)

        h_list.append(h_m)
        p_list.append(p_m)
        surv_prev_list.append(surv_cur)
        surv_list.append(surv_next)
        q_list.append(q_m)

        if m < m_rem - 1:
            if call_code == 14:
                mu_u, var_u = _ramp_survival_post_moments_twinwin_lognormal_torch(f_cond_m, v_cond_m, ko_b_m, ko_spread)
            elif call_is_normal:
                mu_u, var_u = _ramp_survival_post_moments_normal_torch(f_cond_m, v_cond_m, ko_b_m, ko_spread)
            else:
                mu_u, var_u = _ramp_survival_post_moments_lognormal_torch(f_cond_m, v_cond_m, ko_b_m, ko_spread)
            z_post_mean = torch.where(is_call_mask, z_mean + z_std * mu_u, z_mean)
            z_post_var = torch.where(is_call_mask, (z_std * z_std * var_u).clamp(min=1e-12), z_std * z_std)

            r_next = _euler_next_corr_torch(
                forward_perf[:, :, m],
                v_perf[:, :, m].clamp(min=1e-12),
                w_call_all[:, m, :],
                forward_perf[:, :, m + 1],
                v_perf[:, :, m + 1].clamp(min=1e-12),
                w_call_all[:, m + 1, :],
                cm_b,
                call_is_normal,
            )
            z_mean_next = r_next * z_post_mean
            z_std_next = torch.sqrt((r_next * r_next * z_post_var + (1.0 - r_next * r_next)).clamp(min=1e-12))
            if m == 0:
                z_mean = torch.where(is_call_mask, z_mean_next, z_mean)
                z_std = torch.where(is_call_mask, z_std_next, z_std)
            else:
                z_mean = z_mean_next
                z_std = z_std_next

        surv_cur = surv_next

    h_proxy = torch.stack(h_list, dim=1)
    p_proxy = torch.stack(p_list, dim=1)
    survival_proxy = torch.stack(surv_list, dim=1)
    surv_prev = torch.stack(surv_prev_list, dim=1)
    q_proxy = torch.stack(q_list, dim=1)

    surv_term = survival_proxy[:, -1]
    surv_prev_term = surv_prev[:, -1]
    allow_exp_flag = bool(fields["allow_autocall_on_expiry"])
    ic_last = _sched_col(ic_t, m_rem - 1)
    ko_b_term_t = _sched_col(ko_t, m_rem - 1)
    allow_direct_ko = allow_exp_flag and not opt_differs
    exp_ko_mask = (ic_last > 0.5) if allow_direct_ko else torch.zeros(b_sz, dtype=torch.bool, device=device)

    if opt_differs:
        f_opt_raw, v_opt_raw, w_opt_t = stulz_margrabe_moments_with_weights(
            f_perf=forward_perf[:, :, -1],
            v_perf=v_perf[:, :, -1],
            corr=cm_b,
            asset_mask=None,
            monitoring_index=opt_idx,
            inv_lex_perm=perm_seq,
        )
        rho_co = _compute_cross_index_corr_torch(
            forward_perf[:, :, -1],
            v_perf[:, :, -1],
            cm_b,
            call_code,
            w_call_all[:, -1, :],
            opt_code,
            w_opt_t,
        )
        z_mo = rho_co * z_mean_term
        z_vo = rho_co * rho_co * (z_std_term * z_std_term) + (1.0 - rho_co * rho_co)
        z_so = torch.sqrt(z_vo.clamp(min=1e-12))
        v_opt_r = v_opt_raw.clamp(min=1e-12)
        if opt_is_normal:
            f_opt_cond = f_opt_raw + v_opt_r * z_mo
            v_opt_cond = (v_opt_r * z_so).clamp(min=1e-12)
        else:
            f_opt_cond = f_opt_raw.clamp(min=1e-16) * torch.exp(
                v_opt_r * z_mo - 0.5 * (v_opt_r * v_opt_r) * (1.0 - z_vo)
            )
            v_opt_cond = (v_opt_r * z_so).clamp(min=1e-12)
    else:
        f_opt_cond = f_cond_term
        v_opt_cond = v_cond_term

    strike = float(fields["strike"])
    option_gearing = float(fields["option_gearing"])
    has_lower_strike = bool(fields["has_lower_strike"])
    lower_strike_val = float(fields["lower_strike"])
    ki_barrier_val = float(fields["ki_barrier"])
    has_active_ki = bool(fields["has_ki"]) and (float(fields["i_ki_past"]) < 0.5)
    one_star_strike = float(fields["one_star_strike"])
    one_star_coupon = float(fields["one_star_coupon"])
    part_call_strike = float(fields["part_call_strike"])
    part_call_rate = float(fields["part_call_rate"])

    if one_star_strike > 0.0:
        f_ot_c = forward_perf[:, :, -1]
        v_ot_c = v_perf[:, :, -1]
        if m_rem > 1:
            wc_term = w_call_all[:, -1, :]
            wi_c = wc_term * f_ot_c if call_is_normal else wc_term
            vc_n2 = (
                wi_c.unsqueeze(-1) * wi_c.unsqueeze(-2) * cm_b * v_ot_c.unsqueeze(-1) * v_ot_c.unsqueeze(-2)
            ).sum(dim=(-2, -1)).clamp(min=1e-30)
            vc_n = torch.sqrt(vc_n2).unsqueeze(-1)
            cov_i = (
                cm_b * v_ot_c.unsqueeze(-1) * (v_ot_c * wi_c).unsqueeze(-2)
            ).sum(dim=-1)
            v_r = v_ot_c.clamp(min=1e-12)
            rho_ic = (cov_i / (v_r * vc_n)).clamp(-0.9999, 0.9999)
            z_mc = rho_ic * z_mean_term.unsqueeze(-1)
            z_vc = rho_ic * rho_ic * (z_std_term * z_std_term).unsqueeze(-1) + (1.0 - rho_ic * rho_ic)
            f_ot_c = f_ot_c.clamp(min=1e-16) * torch.exp(
                v_r * z_mc - 0.5 * (v_r * v_r) * (1.0 - z_vc)
            )
            v_ot_c = (v_r * torch.sqrt(z_vc.clamp(min=1e-12))).clamp(min=1e-12)
        corr_os = cm_b
        lp_no, los_no, pos_no = _eval_1factor_onestar_torch(
            f_ot_c, v_ot_c, corr_os, strike, has_lower_strike, lower_strike_val,
            has_active_ki, ki_barrier_val, ki_spread, False, ko_b_term_t, ko_spread,
            one_star_strike, one_star_coupon, surv_term, surv_prev_term, option_gearing,
        )
        if allow_direct_ko:
            lp_ko, los_ko, pos_ko = _eval_1factor_onestar_torch(
                f_ot_c, v_ot_c, corr_os, strike, has_lower_strike, lower_strike_val,
                has_active_ki, ki_barrier_val, ki_spread, True, ko_b_term_t, ko_spread,
                one_star_strike, one_star_coupon, surv_term, surv_prev_term, option_gearing,
            )
            l_put_proxy = torch.where(exp_ko_mask, lp_ko, lp_no)
            l_onestar_proxy = torch.where(exp_ko_mask, los_ko, los_no)
            p_onestar_proxy = torch.where(exp_ko_mask, pos_ko, pos_no)
        else:
            l_put_proxy, l_onestar_proxy, p_onestar_proxy = lp_no, los_no, pos_no
    else:
        l_onestar_proxy = torch.zeros(b_sz, dtype=torch.float64, device=device)
        p_onestar_proxy = torch.zeros(b_sz, dtype=torch.float64, device=device)
        has_put = (option_gearing > 0.0) and (
            (not has_lower_strike or strike > lower_strike_val)
            if opt_is_normal
            else (strike > 1e-6 and (not has_lower_strike or strike > lower_strike_val))
        )
        if not has_put:
            l_put_proxy = torch.zeros(b_sz, dtype=torch.float64, device=device)
        elif opt_code == 14:
            p_no = surv_term * _exact_twinwin_terminal_put_torch(
                f_opt_cond, v_opt_cond, strike, has_lower_strike, lower_strike_val,
                has_active_ki, ki_barrier_val, ki_spread, False, ko_b_term_t, ko_spread,
            )
            if allow_direct_ko:
                p_ko = surv_prev_term * _exact_twinwin_terminal_put_torch(
                    f_opt_cond, v_opt_cond, strike, has_lower_strike, lower_strike_val,
                    has_active_ki, ki_barrier_val, ki_spread, True, ko_b_term_t, ko_spread,
                )
                l_put_proxy = torch.where(exp_ko_mask, p_ko, p_no)
            else:
                l_put_proxy = p_no
        else:
            p_no = surv_term * _exact_piecewise_terminal_put_torch(
                f_opt_cond, v_opt_cond, opt_is_normal, strike, has_lower_strike, lower_strike_val,
                has_active_ki, ki_barrier_val, ki_spread, False, ko_b_term_t, ko_spread,
            )
            if allow_direct_ko:
                p_ko = surv_prev_term * _exact_piecewise_terminal_put_torch(
                    f_opt_cond, v_opt_cond, opt_is_normal, strike, has_lower_strike, lower_strike_val,
                    has_active_ki, ki_barrier_val, ki_spread, True, ko_b_term_t, ko_spread,
                )
                l_put_proxy = torch.where(exp_ko_mask, p_ko, p_no)
            else:
                l_put_proxy = p_no

    has_part_call = math.isfinite(part_call_strike) and (part_call_rate != 0.0) and (
        opt_is_normal or part_call_strike >= 0.0
    )
    if has_part_call:
        if opt_code == 14:
            c_no = surv_term * _exact_twinwin_terminal_call_torch(
                f_opt_cond, v_opt_cond, part_call_strike, False, ko_b_term_t, ko_spread
            )
            if allow_direct_ko:
                c_ko = surv_prev_term * _exact_twinwin_terminal_call_torch(
                    f_opt_cond, v_opt_cond, part_call_strike, True, ko_b_term_t, ko_spread
                )
                l_call_proxy = torch.where(exp_ko_mask, c_ko, c_no)
            else:
                l_call_proxy = c_no
        else:
            c_no = surv_term * _exact_piecewise_terminal_call_torch(
                f_opt_cond, v_opt_cond, opt_is_normal, part_call_strike, False, ko_b_term_t, ko_spread
            )
            if allow_direct_ko:
                c_ko = surv_prev_term * _exact_piecewise_terminal_call_torch(
                    f_opt_cond, v_opt_cond, opt_is_normal, part_call_strike, True, ko_b_term_t, ko_spread
                )
                l_call_proxy = torch.where(exp_ko_mask, c_ko, c_no)
            else:
                l_call_proxy = c_no
    else:
        l_call_proxy = torch.zeros(b_sz, dtype=torch.float64, device=device)

    df_last = _sched_col(df_t, m_rem - 1)
    if bool(fields["no_notional_repayment"]):
        notional_pv_proxy = torch.zeros(b_sz, dtype=torch.float64, device=device)
    else:
        df_mat = df_t.to(dtype=torch.float64, device=device)
        if df_mat.ndim == 1:
            df_mat = df_mat.unsqueeze(0).expand(b_sz, m_rem)
        notional_flows = df_mat * p_proxy
        notional_flows_term = notional_flows[:, -1] + df_last * surv_term
        if m_rem > 1:
            notional_pv_proxy = notional_flows[:, :-1].sum(dim=1) + notional_flows_term
        else:
            notional_pv_proxy = notional_flows_term

    cpn_pv_terms: List[torch.Tensor] = []
    mem_acc = torch.full((b_sz,), float(fields["c_mem_0"]), dtype=torch.float64, device=device)
    is_memory = bool(fields["is_memory"])
    autocall_coupon = float(fields["autocall_coupon"])
    for m in range(m_rem):
        c_m_t = _sched_col(cr_t, m)
        df_m_t = _sched_col(df_t, m)
        eff_c_m_t = (c_m_t + mem_acc) if is_memory else c_m_t
        cf_cpn_m = df_m_t * (surv_prev[:, m] * q_proxy[:, m] * eff_c_m_t + p_proxy[:, m] * autocall_coupon)
        cpn_pv_terms.append(cf_cpn_m)
        if is_memory:
            mem_acc = eff_c_m_t * (1.0 - q_proxy[:, m])

    coupon_pv_proxy = torch.stack(cpn_pv_terms, dim=1).sum(dim=1)
    opt_sign = 1.0
    option_pv_proxy = df_last * (
        -opt_sign * option_gearing * l_put_proxy + part_call_rate * l_call_proxy + l_onestar_proxy
    )
    product_value_proxy = (
        torch.tensor(pv_in_transit, dtype=torch.float64, device=device)
        + notional_pv_proxy
        + coupon_pv_proxy
        + option_pv_proxy
    )

    if not is_batched:
        return AutocallBaselineOutput(
            f_w=f_w.squeeze(0),
            v_w=v_w.squeeze(0),
            h_proxy=h_proxy.squeeze(0),
            p_proxy=p_proxy.squeeze(0),
            survival_proxy=survival_proxy.squeeze(0),
            q_proxy=q_proxy.squeeze(0),
            l_put_proxy=l_put_proxy.squeeze(0),
            proxy_delta_weights=proxy_delta_weights.squeeze(0),
            option_pv_proxy=option_pv_proxy.squeeze(0),
            notional_pv_proxy=notional_pv_proxy.squeeze(0),
            coupon_pv_proxy=coupon_pv_proxy.squeeze(0),
            product_value_proxy=product_value_proxy.squeeze(0),
            l_call_proxy=l_call_proxy.squeeze(0),
            l_onestar_proxy=l_onestar_proxy.squeeze(0),
            p_onestar_proxy=p_onestar_proxy.squeeze(0),
            has_unsupported_exotic_index=has_unsupported,
            unsupported_indices=unsupported_tup,
        )

    return AutocallBaselineOutput(
        f_w=f_w,
        v_w=v_w,
        h_proxy=h_proxy,
        p_proxy=p_proxy,
        survival_proxy=survival_proxy,
        q_proxy=q_proxy,
        l_put_proxy=l_put_proxy,
        proxy_delta_weights=proxy_delta_weights,
        option_pv_proxy=option_pv_proxy,
        notional_pv_proxy=notional_pv_proxy,
        coupon_pv_proxy=coupon_pv_proxy,
        product_value_proxy=product_value_proxy,
        l_call_proxy=l_call_proxy,
        l_onestar_proxy=l_onestar_proxy,
        p_onestar_proxy=p_onestar_proxy,
        has_unsupported_exotic_index=has_unsupported,
        unsupported_indices=unsupported_tup,
    )


compute_analytical_autocall_proxy = compute_analytical_autocall_baseline


__all__ = [
    "AutocallBaselineOutput",
    "fast_stulz_margrabe_cpu",
    "stulz_margrabe_moments_with_weights",
    "stulz_margrabe_worst_of_moments",
    "compute_analytical_autocall_baseline",
    "compute_analytical_autocall_proxy",
]
