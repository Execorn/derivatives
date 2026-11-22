"""dupire_qmc_solver.py — Durrleman Surface Regularizer & Multi-Asset Dupire QMC Solver.

Implements SPEC-DEEPVOL-EQ-AUTOCALL-002 §5.5.1 in strict IEEE-754 float64 precision:
  - `RegularizedVolSurface`:
      * Rejects negative implied volatility quotes (`SMILE_ARBITRAGE`) and screens
        spot-independent Durrleman butterfly density g(y) with monotone convex-hull
        wing projection and core sawtooth rejection.
      * Deduplicates identical/non-increasing strikes, enforces calendar total-variance
        monotonicity (d_T w >= 1e-5), records the exact calendar-projection ancestor
        Jacobian `calendar_proj_jacobian`, and freezes all internal NumPy arrays.
      * Honors surface DayCountConvention, SmileInterpolationMethod, and
        SmileExtrapolationMethod (`Flat` vs `Linear`), with vectorized `.iv_batch()` and
        single-pass analytical cubic Dupire local-volatility derivatives.
  - `price_eq_autocall_dupire_qmc` / `price_eq_autocall_qmc_fallback`:
      * Routes CUDA/GPU requests cleanly to `DupireQMCGpuSolver` while preserving exact
        CPU float64 Owen-scrambled RQMC (R=8 replicates, power-of-2 paths, Sobol<=21201
        + antithetic PCG64 hybrid for dim>21201) with exact pathwise memory coupons,
        FundingLeg SE, lex un-permutation, and dividend ex-date forward stepping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import hashlib
import inspect
import math
import threading
from typing import Any, Callable, List, Literal, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import scipy.interpolate
from scipy.special import ndtr
from scipy.stats import norm, qmc
import torch

from deepvol.market.eq_curves import (
    eval_clamped_cubic_coeffs,
    fast_clamped_cubic_coeffs,
    year_fraction as eq_year_fraction,
)
from deepvol.models.autocall.seasoned_collapser import (
    CashFlowItem,
    _to_utc_datetime,
    evaluate_monitoring_index_np,
)

_SOBOL_MAX_DIM: int = 21201
_ASYMMETRIC_PAIR_INDICES = frozenset(
    {"RatioAB", "RatioBA", "DiffAB", "DiffBA", "Outperformance", "Spread"}
)

_ASM_ACCEPTS_COUPON_CF: Optional[bool] = None


def _asm_supports_coupon_cf(asm_fn: Callable[..., Any]) -> bool:
    """Checks once whether `assemble_autocall_cashflows_and_pv` accepts `coupon_cf_undisc`."""
    global _ASM_ACCEPTS_COUPON_CF
    if _ASM_ACCEPTS_COUPON_CF is None:
        _ASM_ACCEPTS_COUPON_CF = "coupon_cf_undisc" in inspect.signature(asm_fn).parameters
    return _ASM_ACCEPTS_COUPON_CF


def _surface_year_fraction(
    d1: Union[datetime, date, str],
    d2: Union[datetime, date, str],
    dcc: str = "Act365",
) -> float:
    """Computes day-count fraction via canonical `eq_curves.year_fraction` with Murex alias support."""
    dt1 = _to_utc_datetime(d1)
    dt2 = _to_utc_datetime(d2)
    dcc_raw = str(dcc or "Act365").strip()
    dcc_norm = dcc_raw.upper().replace("/", "").replace("_", "").replace("-", "").replace(".", "")

    try:
        return float(eq_year_fraction(dt1, dt2, dcc_raw))
    except ValueError:
        pass

    days = float((dt2 - dt1).total_seconds() / 86400.0)
    if dcc_norm in ("ACT365", "ACT365F", "ACT365FIXED", "ACTUAL365"):
        return days / 365.0
    if dcc_norm in ("ACT36525",):
        return days / 365.25
    if dcc_norm in ("ACT360", "ACTUAL360"):
        return days / 360.0
    if dcc_norm in ("ACT364",):
        return days / 364.0
    if dcc_norm in ("ACT366", "ACT3661", "ACT3662", "ACT3663"):
        return days / 366.0
    if dcc_norm in ("ACT252",):
        return days / 252.0
    if dcc_norm in ("BUS252", "BUSINESS252"):
        return float(np.busday_count(dt1.date().isoformat(), dt2.date().isoformat())) / 252.0
    if dcc_norm in ("ACTACT", "ACTACTISDA"):
        return float(eq_year_fraction(dt1, dt2, "ActAct"))
    if dcc_norm in ("30360", "THIRTY360", "30E360", "30360ISDA"):
        return float(eq_year_fraction(dt1, dt2, "Thirty360"))
    return days / 365.0


def _b76_norm_call(y: npt.NDArray[np.float64], w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Black-76 normalized call price C(K)/F with log-strike y = ln(K/F) and total variance w."""
    sqrt_w = np.sqrt(np.maximum(w, 1e-14))
    d1 = (-y + 0.5 * w) / sqrt_w
    d2 = d1 - sqrt_w
    return np.asarray(ndtr(d1) - np.exp(y) * ndtr(d2), dtype=np.float64)


def _invert_b76_norm_call_to_vol(
    y: float,
    c_target: float,
    t: float,
    min_vol: float,
    max_vol: float,
) -> float:
    """Inverts normalized Black-76 call price C/F back to implied vol via safeguarded bisection."""
    k_over_f = math.exp(y)
    intrinsic = max(1.0 - k_over_f, 0.0)
    if c_target <= intrinsic + 1e-14:
        return float(min_vol)
    if c_target >= 1.0 - 1e-14:
        return float(max_vol)

    lo = float(min_vol)
    hi = float(max_vol)
    sqrt_t = math.sqrt(max(t, 1e-12))

    def _call(sig: float) -> float:
        sw = sig * sqrt_t
        d1 = (-y + 0.5 * sw * sw) / max(sw, 1e-12)
        d2 = d1 - sw
        return float(ndtr(d1) - k_over_f * ndtr(d2))

    if _call(lo) >= c_target:
        return lo
    if _call(hi) <= c_target:
        return hi
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        if _call(mid) < c_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return float(0.5 * (lo + hi))


def _project_wing_calls_convex(
    strikes: npt.NDArray[np.float64],
    calls: npt.NDArray[np.float64],
    k_ref: float,
    core_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Projects offending wing call prices onto the monotone decreasing convex hull."""
    n = len(strikes)
    # Andrew's lower convex hull over (strikes, calls) keeping interior core nodes anchored when convex
    pts_idx = np.arange(n, dtype=np.int64)
    # Build lower convex envelope of (strikes, calls)
    hull: List[int] = []
    for idx in pts_idx:
        k_i = float(strikes[idx])
        c_i = max(float(calls[idx]), max(1.0 - k_i / k_ref, 0.0))
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            k1, k2 = float(strikes[i1]), float(strikes[i2])
            c1, c2 = float(calls[i1]), float(calls[i2])
            slope_12 = (c2 - c1) / max(k2 - k1, 1e-12)
            slope_23 = (c_i - c2) / max(k_i - k2, 1e-12)
            if slope_23 < slope_12 - 1e-12:
                hull.pop()
            else:
                break
        hull.append(int(idx))

    k_hull = strikes[hull]
    c_hull = calls[hull]
    c_proj = np.interp(strikes, k_hull, c_hull)
    # Enforce monotone decreasing -1/K_ref <= dc/dK <= 0 and intrinsic floor
    intrinsic = np.maximum(1.0 - strikes / k_ref, 0.0)
    c_proj = np.maximum(c_proj, intrinsic)
    # Preserve core strikes if core was already convex and consistent
    out = np.where(core_mask, calls, c_proj)
    # Ensure global monotonicity non-increasing after merging
    for j in range(1, n):
        if out[j] > out[j - 1]:
            out[j] = out[j - 1]
    return out


def _eval_durrleman_batch(
    y: npt.NDArray[np.float64],
    w_mat: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Vectorized Durrleman butterfly density g(y) across all slices using fast_clamped_cubic_coeffs."""
    coeffs = fast_clamped_cubic_coeffs(y, w_mat.T)
    c0 = coeffs[0].T
    c1 = coeffs[1].T
    c2 = coeffs[2].T
    c3 = coeffs[3].T

    y_mid = 0.5 * (y[:-1] + y[1:])
    dx_mid = (y_mid - y[:-1])[None, :]
    dx_end = float(y[-1] - y[-2])
    y_eval = np.concatenate([y, y_mid])

    c0_e = c0[:, -1:]
    c1_e = c1[:, -1:]
    c2_e = c2[:, -1:]
    c3_e = c3[:, -1:]
    w_end = ((c0_e * dx_end + c1_e) * dx_end + c2_e) * dx_end + c3_e
    wp_end = (3.0 * c0_e * dx_end + 2.0 * c1_e) * dx_end + c2_e
    wpp_end = 6.0 * c0_e * dx_end + 2.0 * c1_e

    w_mid = ((c0 * dx_mid + c1) * dx_mid + c2) * dx_mid + c3
    wp_mid = (3.0 * c0 * dx_mid + 2.0 * c1) * dx_mid + c2
    wpp_mid = 6.0 * c0 * dx_mid + 2.0 * c1

    w_e = np.maximum(np.concatenate([c3, w_end, w_mid], axis=1), 1e-12)
    wp_e = np.concatenate([c2, wp_end, wp_mid], axis=1)
    wpp_e = np.concatenate([2.0 * c1, wpp_end, wpp_mid], axis=1)

    term1 = (1.0 - (y_eval[None, :] * wp_e) / (2.0 * w_e)) ** 2
    term2 = 0.25 * (wp_e * wp_e) * (1.0 / w_e + 0.25)
    g_mat = term1 - term2 + 0.5 * wpp_e
    return y_eval, g_mat


def _eval_durrleman_slice(
    y: npt.NDArray[np.float64],
    w_slice: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluates Durrleman butterfly density g(y) at knots and midpoints using spot-independent log-strikes."""
    y_eval, g_mat = _eval_durrleman_batch(y, np.asarray(w_slice, dtype=np.float64)[None, :])
    return y_eval, g_mat[0]


def enforce_no_arbitrage_projection_with_jacobian(
    tenors: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    quotes: npt.NDArray[np.float64],
    *,
    min_vol: float = 0.01,
    max_vol: float = 5.00,
    min_calendar_dvar_dt: float = 1e-5,
    surface_id: str = "UNKNOWN",
    warnings_out: Optional[List[str]] = None,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    bool,
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Screens and projects an implied volatility surface and returns exact projection Jacobians.

    Returns:
        unique_strikes: (N_unique,)
        projected_quotes: (N_exp, N_unique)
        dedup_matrix: (N_unique, N_strike_raw)
        was_projected: bool
        cal_proj_ancestor: (N_exp, N_unique) int64 binding source slice index (-1 if floored/wing-projected)
        cal_proj_scale: (N_exp, N_unique) float64 chain-rule scale factor
        cal_proj_jacobian: (N_exp, N_exp, N_unique) float64 d(quotes[m,u]) / d(q_dedup[m_src,u])
    """
    t_arr = np.asarray(tenors, dtype=np.float64)
    k_arr = np.asarray(strikes, dtype=np.float64)
    q_arr = np.asarray(quotes, dtype=np.float64)

    if t_arr.ndim != 1 or k_arr.ndim != 1 or q_arr.ndim != 2:
        raise ValueError(f"SMILE_ARBITRAGE: Invalid surface dimensions on surface={surface_id}.")
    if len(t_arr) == 0 or len(k_arr) < 2:
        raise ValueError(
            f"SMILE_ARBITRAGE: Surface must have at least 1 expiry and 2 strikes on surface={surface_id}."
        )
    if q_arr.shape != (len(t_arr), len(k_arr)):
        raise ValueError(
            f"SMILE_ARBITRAGE: Quotes shape {q_arr.shape} does not match ({len(t_arr)}, {len(k_arr)}) on surface={surface_id}."
        )
    if not (np.all(np.isfinite(t_arr)) and np.all(np.isfinite(k_arr)) and np.all(np.isfinite(q_arr))):
        bad_q = np.argwhere(~np.isfinite(q_arr))
        if len(bad_q) > 0:
            m_b, j_b = int(bad_q[0, 0]), int(bad_q[0, 1])
            j_p = max(j_b - 1, 0)
            t_b = float(t_arr[m_b]) if np.isfinite(t_arr[m_b]) else -1.0
            k_b = float(k_arr[j_b]) if np.isfinite(k_arr[j_b]) else -1.0
            k_p = float(k_arr[j_p]) if np.isfinite(k_arr[j_p]) else -1.0
            v_p = float(q_arr[m_b, j_p])
            v_b = float(q_arr[m_b, j_b])
            raise ValueError(
                f"SMILE_ARBITRAGE: Non-finite (NaN/Inf) values in volatility surface={surface_id}, "
                f"slice={m_b} (T={t_b:.4f}Y), K={k_b:.4f} (prev_K={k_p:.4f} -> K={k_b:.4f}, "
                f"vol_jump={v_p:.5f}->{v_b:.5f}), min_d2c=nan, min_g=nan."
            )
        raise ValueError(
            f"SMILE_ARBITRAGE: Non-finite (NaN/Inf) values in volatility surface={surface_id}."
        )
    if np.any(t_arr <= 0.0) or np.any(np.diff(t_arr) <= 0.0):
        raise ValueError(
            f"SMILE_ARBITRAGE: Expiration tenors must be strictly positive and increasing on surface={surface_id}."
        )
    if np.any(k_arr <= 0.0):
        raise ValueError(f"SMILE_ARBITRAGE: Strikes must be strictly positive on surface={surface_id}.")
    if np.any(q_arr < 1e-4):
        bad_low = np.argwhere(q_arr < 1e-4)
        m_b, j_b = int(bad_low[0, 0]), int(bad_low[0, 1])
        j_p = max(j_b - 1, 0)
        raise ValueError(
            f"SMILE_ARBITRAGE: Negative or sub-1e-4 implied volatility quote on surface={surface_id}, "
            f"slice={m_b} (T={float(t_arr[m_b]):.4f}Y), K={float(k_arr[j_b]):.4f} "
            f"(prev_K={float(k_arr[j_p]):.4f} -> K={float(k_arr[j_b]):.4f}, "
            f"vol_jump={float(q_arr[m_b, j_p]):.5f}->{float(q_arr[m_b, j_b]):.5f}), "
            f"min_d2c=0.000000, min_g=0.0000."
        )
    if np.any(q_arr > max_vol):
        bad_hi = np.argwhere(q_arr > max_vol)
        m_b, j_b = int(bad_hi[0, 0]), int(bad_hi[0, 1])
        j_p = max(j_b - 1, 0)
        raise ValueError(
            f"SMILE_ARBITRAGE: Implied volatility quotes exceed max_vol ({max_vol}) on surface={surface_id}, "
            f"slice={m_b} (T={float(t_arr[m_b]):.4f}Y), K={float(k_arr[j_b]):.4f} "
            f"(prev_K={float(k_arr[j_p]):.4f} -> K={float(k_arr[j_b]):.4f}, "
            f"vol_jump={float(q_arr[m_b, j_p]):.5f}->{float(q_arr[m_b, j_b]):.5f}), "
            f"min_d2c=0.000000, min_g=0.0000."
        )

    was_projected = False

    # 1. Deduplicate identical or non-strictly-increasing strikes
    if len(k_arr) >= 2 and bool(
        np.all(np.diff(k_arr) > 1e-9 * np.maximum(1.0, np.abs(k_arr[:-1])))
    ):
        unique_strikes = k_arr.copy()
        dedup_matrix = np.eye(len(k_arr), dtype=np.float64)
        q_dedup = q_arr.copy()
    else:
        sort_idx = np.argsort(k_arr, kind="mergesort")
        k_sorted = k_arr[sort_idx]

        groups: List[List[int]] = [[int(sort_idx[0])]]
        unique_k_list: List[float] = [float(k_sorted[0])]
        for idx_pos in range(1, len(k_sorted)):
            orig_i = int(sort_idx[idx_pos])
            if abs(float(k_sorted[idx_pos]) - unique_k_list[-1]) <= 1e-9 * max(
                1.0, abs(unique_k_list[-1])
            ):
                groups[-1].append(orig_i)
                was_projected = True
            else:
                groups.append([orig_i])
                unique_k_list.append(float(k_sorted[idx_pos]))

        n_unique = len(unique_k_list)
        if n_unique < 2:
            raise ValueError("SMILE_ARBITRAGE: Fewer than 2 unique strikes after deduplication.")

        dedup_matrix = np.zeros((n_unique, len(k_arr)), dtype=np.float64)
        for u_idx, g in enumerate(groups):
            w_grp = 1.0 / len(g)
            for raw_idx in g:
                dedup_matrix[u_idx, raw_idx] = w_grp

        unique_strikes = np.asarray(unique_k_list, dtype=np.float64)
        q_dedup = q_arr @ dedup_matrix.T

    n_exp = len(t_arr)
    n_u = len(unique_strikes)
    cal_ancestor = np.broadcast_to(np.arange(n_exp, dtype=np.int64)[:, None], (n_exp, n_u)).copy()
    cal_scale = np.ones((n_exp, n_u), dtype=np.float64)

    # 2. Floor implied volatility at min_vol
    floor_mask = q_dedup < min_vol
    if np.any(floor_mask):
        q_dedup = np.maximum(q_dedup, min_vol)
        cal_ancestor[floor_mask] = -1
        cal_scale[floor_mask] = 0.0
        was_projected = True

    # 3. Enforce calendar total-variance monotonicity: w_m(K) >= w_{m-1}(K) + min_calendar_dvar_dt * dT
    w_mat = (q_dedup * q_dedup) * t_arr[:, None]
    sig_curr = q_dedup.copy()
    for m in range(1, n_exp):
        dt_m = t_arr[m] - t_arr[m - 1]
        min_w_m = w_mat[m - 1] + min_calendar_dvar_dt * dt_m
        viol = w_mat[m] < min_w_m
        if np.any(viol):
            w_mat[m, viol] = min_w_m[viol]
            was_projected = True
            for u in np.flatnonzero(viol):
                sig_new = math.sqrt(max(w_mat[m, u] / t_arr[m], min_vol * min_vol))
                if w_mat[m, u] / t_arr[m] <= min_vol * min_vol or cal_ancestor[m - 1, u] < 0:
                    cal_ancestor[m, u] = -1
                    cal_scale[m, u] = 0.0
                else:
                    cal_ancestor[m, u] = cal_ancestor[m - 1, u]
                    cal_scale[m, u] = cal_scale[m - 1, u] * (
                        (t_arr[m - 1] * sig_curr[m - 1, u]) / max(t_arr[m] * sig_new, 1e-12)
                    )
                sig_curr[m, u] = sig_new

    projected_quotes = np.sqrt(np.maximum(w_mat / t_arr[:, None], min_vol * min_vol))

    # 4. Spot-independent Durrleman butterfly convexity screening & localized wing projection
    if n_u >= 3:
        k_ref = float(np.median(unique_strikes))
        y_u = np.log(unique_strikes / k_ref)
        core_strike_mask = (unique_strikes >= 0.5 * k_ref) & (unique_strikes <= 1.5 * k_ref)

        w_all_init = (projected_quotes * projected_quotes) * t_arr[:, None]
        y_eval, g_mat_all = _eval_durrleman_batch(y_u, w_all_init)
        calls_mat_all = _b76_norm_call(y_u[None, :], w_all_init)
        core_eval_mask = (y_eval >= math.log(0.5)) & (y_eval <= math.log(1.5))
        has_core_eval = bool(np.any(core_eval_mask))
        dk_step = np.maximum(np.diff(unique_strikes), 1e-12)

        def _compute_surface_worst_metrics() -> Tuple[float, float]:
            w_chk_all = (projected_quotes * projected_quotes) * t_arr[:, None]
            _, g_chk_all = _eval_durrleman_batch(y_u, w_chk_all)
            if has_core_eval:
                surf_min_g = float(np.min(g_chk_all[:, core_eval_mask]))
            else:
                surf_min_g = float(np.min(g_chk_all))
            c_chk_all = _b76_norm_call(y_u[None, :], w_chk_all)
            sl_chk_all = np.diff(c_chk_all, axis=1) / dk_step[None, :]
            d2_chk_all = np.diff(sl_chk_all, axis=1)
            core_int_mask = core_strike_mask[1:-1]
            if np.any(core_int_mask):
                surf_min_d2c = float(np.min(d2_chk_all[:, core_int_mask]))
            else:
                surf_min_d2c = 0.0
            if not math.isfinite(surf_min_d2c):
                surf_min_d2c = 0.0
            if not math.isfinite(surf_min_g):
                surf_min_g = 0.0
            return surf_min_d2c, surf_min_g

        for m in range(n_exp):
            t_m = float(t_arr[m])
            g_eval = g_mat_all[m]

            min_g_core = (
                float(np.min(g_eval[core_eval_mask]))
                if has_core_eval
                else float(np.min(g_eval))
            )
            min_g_all = float(np.min(g_eval))

            # Check discrete Black-76 normalized call convexity C''(K) >= 0 at F = K_ref
            calls_m = calls_mat_all[m]
            slopes_m = np.diff(calls_m) / dk_step
            d2c_m = np.diff(slopes_m)

            min_d2c_core = float("inf")
            j_first_severe: Optional[int] = None
            j_first_mild: Optional[int] = None
            j_worst_core: int = 1
            for j in range(1, n_u - 1):
                if core_strike_mask[j]:
                    val_d2c = float(d2c_m[j - 1])
                    if val_d2c < min_d2c_core:
                        min_d2c_core = val_d2c
                        j_worst_core = j
                    if val_d2c < -1e-3 and j_first_severe is None:
                        j_first_severe = j
                    elif val_d2c < -1e-4 and j_first_mild is None:
                        j_first_mild = j
            if not math.isfinite(min_d2c_core):
                min_d2c_core = float(np.min(d2c_m)) if len(d2c_m) > 0 else 0.0
                j_worst_core = int(np.argmin(d2c_m)) + 1 if len(d2c_m) > 0 else 1

            j_off = (
                j_first_severe
                if j_first_severe is not None
                else (j_first_mild if j_first_mild is not None else j_worst_core)
            )
            j_prev = max(j_off - 1, 0)
            strike_val = float(unique_strikes[j_off])
            prev_strike_val = float(unique_strikes[j_prev])
            v_prev = float(projected_quotes[m, j_prev])
            v_curr = float(projected_quotes[m, j_off])

            if min_d2c_core < -1e-3 or min_g_core < -10.0 or min_g_all < -25.0:
                surf_min_d2c, surf_min_g = _compute_surface_worst_metrics()
                min_d2c_val = min(min_d2c_core, surf_min_d2c)
                min_g_val = min(min_g_core, surf_min_g)
                raise ValueError(
                    f"SMILE_ARBITRAGE: Core call-price convexity violation at slice {m}: "
                    f"surface={surface_id}, slice={m} (T={t_m:.4f}Y), "
                    f"K={strike_val:.4f} (prev_K={prev_strike_val:.4f} -> K={strike_val:.4f}, "
                    f"vol_jump={v_prev:.5f}->{v_curr:.5f}), "
                    f"min_d2c={min_d2c_val:.6f}, min_g={min_g_val:.4f}."
                )

            if min_d2c_core < -1e-4:
                # Two-tier C2 ruling: mild numerical quote noise (-1e-3 <= min_d2c_core < -1e-4
                # and min_g_core >= -10.0) is projected onto convex call prices with warning.
                all_wing_mask = np.zeros_like(core_strike_mask)
                calls_proj = _project_wing_calls_convex(unique_strikes, calls_m, k_ref, all_wing_mask)
                for u in range(n_u):
                    if abs(calls_proj[u] - calls_m[u]) > 1e-11:
                        sig_u = _invert_b76_norm_call_to_vol(
                            float(y_u[u]), float(calls_proj[u]), t_m, min_vol, max_vol
                        )
                        projected_quotes[m, u] = sig_u
                        cal_ancestor[m, u] = -1
                        cal_scale[m, u] = 0.0
                # If a downward quote dip left any residual d2c < -1e-4 after lower-hull projection,
                # smooth offending interior vol wiggle iteratively until d2c >= -1e-4.
                for _ in range(16):
                    w_chk = (projected_quotes[m] * projected_quotes[m]) * t_m
                    c_chk = _b76_norm_call(y_u, w_chk)
                    sl_chk = np.diff(c_chk) / np.maximum(np.diff(unique_strikes), 1e-12)
                    d2_chk = np.diff(sl_chk)
                    fixed_any = False
                    for j_c in range(1, n_u - 1):
                        if core_strike_mask[j_c] and float(d2_chk[j_c - 1]) < -1e-4:
                            w_interp = (unique_strikes[j_c] - unique_strikes[j_c - 1]) / max(
                                unique_strikes[j_c + 1] - unique_strikes[j_c - 1], 1e-12
                            )
                            v_sec = (1.0 - w_interp) * projected_quotes[m, j_c - 1] + w_interp * projected_quotes[m, j_c + 1]
                            projected_quotes[m, j_c] = 0.5 * (projected_quotes[m, j_c] + v_sec)
                            cal_ancestor[m, j_c] = -1
                            cal_scale[m, j_c] = 0.0
                            fixed_any = True
                    if not fixed_any:
                        break
                was_projected = True
                if warnings_out is not None and "SMILE_ARBITRAGE_PROJECTED" not in warnings_out:
                    warnings_out.append("SMILE_ARBITRAGE_PROJECTED")
            elif np.min(d2c_m) < -1e-9 or np.max(slopes_m) > 1e-9:
                calls_proj = _project_wing_calls_convex(unique_strikes, calls_m, k_ref, core_strike_mask)
                if np.max(np.abs(calls_proj - calls_m)) > 1e-10:
                    for u in range(n_u):
                        if not core_strike_mask[u] and abs(calls_proj[u] - calls_m[u]) > 1e-11:
                            sig_wing = _invert_b76_norm_call_to_vol(
                                float(y_u[u]), float(calls_proj[u]), t_m, min_vol, max_vol
                            )
                            projected_quotes[m, u] = sig_wing
                            cal_ancestor[m, u] = -1
                            cal_scale[m, u] = 0.0
                    was_projected = True

        # Re-enforce calendar monotonicity only when variance falls meaningfully below m-1
        # (ignore ~1e-16 round-trip noise from sqrt(w/t)**2 * t)
        w_re = (projected_quotes * projected_quotes) * t_arr[:, None]
        for m in range(1, n_exp):
            dt_m = t_arr[m] - t_arr[m - 1]
            min_w_m = w_re[m - 1] + min_calendar_dvar_dt * dt_m
            tol_m = 1e-14 * np.maximum(1.0, np.abs(min_w_m))
            viol = w_re[m] < (min_w_m - tol_m)
            if np.any(viol):
                w_re[m, viol] = min_w_m[viol]
                was_projected = True
                for u in np.flatnonzero(viol):
                    sig_prev_u = float(projected_quotes[m - 1, u])
                    sig_new = math.sqrt(max(w_re[m, u] / t_arr[m], min_vol * min_vol))
                    projected_quotes[m, u] = sig_new
                    if w_re[m, u] / t_arr[m] <= min_vol * min_vol or cal_ancestor[m - 1, u] < 0:
                        cal_ancestor[m, u] = -1
                        cal_scale[m, u] = 0.0
                    else:
                        cal_ancestor[m, u] = cal_ancestor[m - 1, u]
                        cal_scale[m, u] = cal_scale[m - 1, u] * (
                            (t_arr[m - 1] * sig_prev_u) / max(t_arr[m] * sig_new, 1e-12)
                        )

    cal_jacobian = np.zeros((n_exp, n_exp, n_u), dtype=np.float64)
    for m in range(n_exp):
        for u in range(n_u):
            src_m = int(cal_ancestor[m, u])
            if src_m >= 0:
                cal_jacobian[m, src_m, u] = cal_scale[m, u]

    return (
        unique_strikes,
        projected_quotes,
        dedup_matrix,
        was_projected,
        cal_ancestor,
        cal_scale,
        cal_jacobian,
    )


def enforce_no_arbitrage_projection(
    tenors: npt.NDArray[np.float64],
    strikes: npt.NDArray[np.float64],
    quotes: npt.NDArray[np.float64],
    *,
    min_vol: float = 0.01,
    max_vol: float = 5.00,
    min_calendar_dvar_dt: float = 1e-5,
    surface_id: str = "UNKNOWN",
    warnings_out: Optional[List[str]] = None,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    bool,
]:
    """Screens and projects an implied volatility surface to satisfy no-arbitrage invariants."""
    u_k, proj_q, dedup_m, was_proj, _, _, _ = enforce_no_arbitrage_projection_with_jacobian(
        tenors,
        strikes,
        quotes,
        min_vol=min_vol,
        max_vol=max_vol,
        min_calendar_dvar_dt=min_calendar_dvar_dt,
        surface_id=surface_id,
        warnings_out=warnings_out,
    )
    return u_k, proj_q, dedup_m, was_proj


def _build_piecewise_linear_coeffs(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Builds `(4, K-1, ...)` coefficients for piecewise linear strike interpolation."""
    k = len(x)
    h = np.diff(x).reshape((k - 1,) + (1,) * (y.ndim - 1))
    c3 = y[:-1].copy()
    c2 = (y[1:] - y[:-1]) / h
    c1 = np.zeros_like(c2)
    c0 = np.zeros_like(c2)
    return np.stack([c0, c1, c2, c3], axis=0)


class RegularizedVolSurface:
    """Arbitrage-regularized implied & Dupire local volatility surface in IEEE-754 float64.

    All internal NumPy arrays are frozen (`.setflags(write=False)`) upon construction so
    instances are safe to share across threads and content-addressed market caches.
    """

    def __init__(
        self,
        surf_spec: Any,
        as_of_date: Union[datetime, date, str, None] = None,
        *,
        bc_type: str = "clamped",
        min_vol: float = 0.01,
        max_local_vol: float = 2.50,
        dcc: Optional[str] = None,
        smile_interp: Optional[str] = None,
        smile_extrap: Optional[str] = None,
        surface_id: Optional[str] = None,
    ) -> None:
        self.min_vol = float(min_vol)
        self.max_local_vol = float(max_local_vol)
        self.bc_type = str(bc_type)
        self._basis_lock = threading.RLock()
        self.warnings: List[str] = []

        if isinstance(surf_spec, Mapping):
            self.identifier = str(surf_spec.get("Identifier", surf_spec.get("identifier", "ASSET")))
            raw_dcc = (
                dcc
                if dcc is not None
                else surf_spec.get("DayCountConvention", surf_spec.get("day_count_convention", "Act365"))
            )
            eff_dcc = str(getattr(raw_dcc, "value", raw_dcc))
            raw_interp = (
                smile_interp
                if smile_interp is not None
                else surf_spec.get(
                    "SmileInterpolationMethod",
                    surf_spec.get("smile_interpolation_method", "CubicSpline"),
                )
            )
            eff_interp = str(getattr(raw_interp, "value", raw_interp))
            raw_extrap = (
                smile_extrap
                if smile_extrap is not None
                else surf_spec.get(
                    "SmileExtrapolationMethod",
                    surf_spec.get("smile_extrapolation_method", "Flat"),
                )
            )
            eff_extrap = str(getattr(raw_extrap, "value", raw_extrap))
            raw_as_of = as_of_date if as_of_date is not None else surf_spec.get("AsOfDate", surf_spec.get("as_of_date"))
            raw_strikes = surf_spec.get("Strikes", surf_spec.get("strikes", []))
            raw_quotes = surf_spec.get(
                "VolatilityQuotes",
                surf_spec.get("volatility_quotes", surf_spec.get("quotes", [])),
            )
            exp_dates: Any = surf_spec.get(
                "ExpirationDates",
                surf_spec.get("expiration_dates", surf_spec.get("exp_dates")),
            )
            if "tenors" in surf_spec and surf_spec["tenors"] is not None:
                raw_tenors = np.asarray(surf_spec["tenors"], dtype=np.float64)
                raw_tenors_365 = raw_tenors.copy()
            else:
                if raw_as_of is None:
                    raise ValueError(
                        "as_of_date is required when ExpirationDates are provided instead of tenors."
                    )
                if exp_dates is None:
                    exp_dates = []
                raw_tenors = np.asarray(
                    [_surface_year_fraction(raw_as_of, d, eff_dcc) for d in exp_dates],
                    dtype=np.float64,
                )
                raw_tenors_365 = np.asarray(
                    [_surface_year_fraction(raw_as_of, d, "Act365") for d in exp_dates],
                    dtype=np.float64,
                )
        else:
            self.identifier = str(
                getattr(surf_spec, "Identifier", getattr(surf_spec, "identifier", "ASSET"))
            )
            raw_dcc = (
                dcc
                if dcc is not None
                else getattr(
                    surf_spec,
                    "DayCountConvention",
                    getattr(surf_spec, "day_count_convention", "Act365"),
                )
            )
            eff_dcc = str(getattr(raw_dcc, "value", raw_dcc))
            raw_interp = (
                smile_interp
                if smile_interp is not None
                else getattr(
                    surf_spec,
                    "SmileInterpolationMethod",
                    getattr(surf_spec, "smile_interpolation_method", "CubicSpline"),
                )
            )
            eff_interp = str(getattr(raw_interp, "value", raw_interp))
            raw_extrap = (
                smile_extrap
                if smile_extrap is not None
                else getattr(
                    surf_spec,
                    "SmileExtrapolationMethod",
                    getattr(surf_spec, "smile_extrapolation_method", "Flat"),
                )
            )
            eff_extrap = str(getattr(raw_extrap, "value", raw_extrap))
            raw_as_of = (
                as_of_date
                if as_of_date is not None
                else getattr(surf_spec, "AsOfDate", getattr(surf_spec, "as_of_date", None))
            )
            raw_strikes = getattr(surf_spec, "Strikes", getattr(surf_spec, "strikes", []))
            raw_quotes = getattr(
                surf_spec,
                "VolatilityQuotes",
                getattr(surf_spec, "volatility_quotes", getattr(surf_spec, "quotes", [])),
            )
            exp_dates = getattr(
                surf_spec,
                "ExpirationDates",
                getattr(surf_spec, "expiration_dates", None),
            )
            if hasattr(surf_spec, "tenors") and getattr(surf_spec, "tenors") is not None:
                raw_tenors = np.asarray(getattr(surf_spec, "tenors"), dtype=np.float64)
                raw_tenors_365 = raw_tenors.copy()
            else:
                if raw_as_of is None:
                    raise ValueError(
                        "as_of_date is required when ExpirationDates are provided instead of tenors."
                    )
                if exp_dates is None:
                    exp_dates = []
                raw_tenors = np.asarray(
                    [_surface_year_fraction(raw_as_of, d, eff_dcc) for d in exp_dates],
                    dtype=np.float64,
                )
                raw_tenors_365 = np.asarray(
                    [_surface_year_fraction(raw_as_of, d, "Act365") for d in exp_dates],
                    dtype=np.float64,
                )

        if surface_id is not None and str(surface_id).strip() and (
            not self.identifier or self.identifier in ("ASSET", "UNKNOWN")
        ):
            self.identifier = str(surface_id).strip()

        self.dcc: str = eff_dcc
        self.as_of_date: Optional[datetime] = _to_utc_datetime(raw_as_of) if raw_as_of is not None else None
        self.smile_interp: str = eff_interp
        self.interp_method: str = eff_interp
        self.smile_extrap: str = eff_extrap
        self.extrap_method: str = eff_extrap

        self.raw_strikes = np.asarray(raw_strikes, dtype=np.float64).copy()
        self.raw_quotes = np.asarray(raw_quotes, dtype=np.float64).copy()
        self.tenors = np.asarray(raw_tenors, dtype=np.float64).copy()
        self.tenors_act365 = np.asarray(raw_tenors_365, dtype=np.float64).copy()

        (
            self.strikes,
            self.quotes,
            self.dedup_matrix,
            self.was_projected,
            self.calendar_proj_ancestor,
            self.calendar_proj_scale,
            self.calendar_proj_jacobian,
        ) = enforce_no_arbitrage_projection_with_jacobian(
            self.tenors,
            self.raw_strikes,
            self.raw_quotes,
            min_vol=self.min_vol,
            surface_id=self.identifier,
            warnings_out=self.warnings,
        )

        self.calendar_proj_digest: str = hashlib.blake2b(
            self.calendar_proj_ancestor.tobytes() + self.calendar_proj_scale.tobytes(),
            digest_size=16,
        ).hexdigest()

        self.n_exp = len(self.tenors)
        self.n_unique_strikes = len(self.strikes)
        self.n_raw_strikes = len(self.raw_strikes)

        self._basis_cs: Any = None
        self._fast_basis_coeffs: Optional[npt.NDArray[np.float64]] = None
        self._basis_slope_left: Optional[npt.NDArray[np.float64]] = None
        self._basis_slope_right: Optional[npt.NDArray[np.float64]] = None

        interp_norm = self.smile_interp.strip().lower()
        extrap_norm = self.smile_extrap.strip().lower()
        use_linear_extrap = extrap_norm == "linear"

        self._fast_slice_coeffs: Optional[npt.NDArray[np.float64]] = None
        self._slice_cs: Any = None

        if interp_norm == "linear":
            self._fast_slice_coeffs = _build_piecewise_linear_coeffs(self.strikes, self.quotes.T)
            if use_linear_extrap:
                h0 = max(float(self.strikes[1] - self.strikes[0]), 1e-12)
                h1 = max(float(self.strikes[-1] - self.strikes[-2]), 1e-12)
                self._slopes_left = (self.quotes[:, 1] - self.quotes[:, 0]) / h0
                self._slopes_right = (self.quotes[:, -1] - self.quotes[:, -2]) / h1
            else:
                self._slopes_left = np.zeros(self.n_exp, dtype=np.float64)
                self._slopes_right = np.zeros(self.n_exp, dtype=np.float64)
        elif interp_norm in ("pchip", "monotonecubic"):
            self._slice_cs = scipy.interpolate.PchipInterpolator(self.strikes, self.quotes, axis=1)
            self._fast_slice_coeffs = np.transpose(self._slice_cs.c, (0, 1, 2)).copy()
            if use_linear_extrap:
                self._slopes_left = np.asarray(self._slice_cs(self.strikes[0], 1), dtype=np.float64)
                self._slopes_right = np.asarray(self._slice_cs(self.strikes[-1], 1), dtype=np.float64)
            else:
                self._slopes_left = np.zeros(self.n_exp, dtype=np.float64)
                self._slopes_right = np.zeros(self.n_exp, dtype=np.float64)
        elif self.bc_type == "clamped" and interp_norm in ("cubicspline", "spline", ""):
            self._fast_slice_coeffs = fast_clamped_cubic_coeffs(self.strikes, self.quotes.T)
            if use_linear_extrap:
                h0 = max(float(self.strikes[1] - self.strikes[0]), 1e-12)
                h1 = max(float(self.strikes[-1] - self.strikes[-2]), 1e-12)
                self._slopes_left = (self.quotes[:, 1] - self.quotes[:, 0]) / h0
                self._slopes_right = (self.quotes[:, -1] - self.quotes[:, -2]) / h1
            else:
                self._slopes_left = np.zeros(self.n_exp, dtype=np.float64)
                self._slopes_right = np.zeros(self.n_exp, dtype=np.float64)
        else:
            eff_bc = self.bc_type
            if interp_norm == "naturalcubicspline":
                eff_bc = "natural"
            elif interp_norm in ("notaknot", "notaknotcubicspline", "polynomial", "lme"):
                eff_bc = "not-a-knot"
            self._slice_cs = scipy.interpolate.CubicSpline(
                self.strikes, self.quotes, bc_type=eff_bc, axis=1
            )
            self._fast_slice_coeffs = np.asarray(self._slice_cs.c, dtype=np.float64).copy()
            if use_linear_extrap:
                self._slopes_left = np.asarray(self._slice_cs(self.strikes[0], 1), dtype=np.float64)
                self._slopes_right = np.asarray(self._slice_cs(self.strikes[-1], 1), dtype=np.float64)
            else:
                self._slopes_left = np.zeros(self.n_exp, dtype=np.float64)
                self._slopes_right = np.zeros(self.n_exp, dtype=np.float64)

        self._ensure_basis_spline()
        self._freeze_arrays()

    def _freeze_arrays(self) -> None:
        """Marks all internal NumPy arrays read-only (`CHR-010`)."""
        arrays: List[Optional[npt.NDArray[Any]]] = [
            self.raw_strikes,
            self.raw_quotes,
            self.tenors,
            self.tenors_act365,
            self.strikes,
            self.quotes,
            self.dedup_matrix,
            self.calendar_proj_ancestor,
            self.calendar_proj_scale,
            self.calendar_proj_jacobian,
            self._fast_slice_coeffs,
            self._slopes_left,
            self._slopes_right,
            self._fast_basis_coeffs,
            self._basis_slope_left,
            self._basis_slope_right,
        ]
        for arr in arrays:
            if arr is not None:
                arr.setflags(write=False)

    def __deepcopy__(self, memo: Any) -> RegularizedVolSurface:
        _ = memo
        return self

    def _ensure_basis_spline(self) -> None:
        """Initializes and freezes the strike basis spline weights thread-safely."""
        with self._basis_lock:
            if self._fast_basis_coeffs is not None or self._basis_cs is not None:
                return
            eye_u = np.eye(self.n_unique_strikes, dtype=np.float64)
            interp_norm = self.smile_interp.strip().lower()
            extrap_norm = self.smile_extrap.strip().lower()
            use_linear_extrap = extrap_norm == "linear"

            if interp_norm == "linear":
                self._fast_basis_coeffs = _build_piecewise_linear_coeffs(self.strikes, eye_u)
                if use_linear_extrap:
                    sl = np.zeros(self.n_unique_strikes, dtype=np.float64)
                    sr = np.zeros(self.n_unique_strikes, dtype=np.float64)
                    h0 = max(float(self.strikes[1] - self.strikes[0]), 1e-12)
                    h1 = max(float(self.strikes[-1] - self.strikes[-2]), 1e-12)
                    sl[0], sl[1] = -1.0 / h0, 1.0 / h0
                    sr[-2], sr[-1] = -1.0 / h1, 1.0 / h1
                    self._basis_slope_left = sl
                    self._basis_slope_right = sr
                else:
                    self._basis_slope_left = np.zeros(self.n_unique_strikes, dtype=np.float64)
                    self._basis_slope_right = np.zeros(self.n_unique_strikes, dtype=np.float64)
            elif self.bc_type == "clamped" and interp_norm in ("cubicspline", "spline", "", "pchip", "monotonecubic"):
                self._fast_basis_coeffs = fast_clamped_cubic_coeffs(self.strikes, eye_u)
                if use_linear_extrap:
                    sl = np.zeros(self.n_unique_strikes, dtype=np.float64)
                    sr = np.zeros(self.n_unique_strikes, dtype=np.float64)
                    h0 = max(float(self.strikes[1] - self.strikes[0]), 1e-12)
                    h1 = max(float(self.strikes[-1] - self.strikes[-2]), 1e-12)
                    sl[0], sl[1] = -1.0 / h0, 1.0 / h0
                    sr[-2], sr[-1] = -1.0 / h1, 1.0 / h1
                    self._basis_slope_left = sl
                    self._basis_slope_right = sr
                else:
                    self._basis_slope_left = np.zeros(self.n_unique_strikes, dtype=np.float64)
                    self._basis_slope_right = np.zeros(self.n_unique_strikes, dtype=np.float64)
            else:
                eff_bc = self.bc_type
                if interp_norm == "naturalcubicspline":
                    eff_bc = "natural"
                elif interp_norm in ("notaknot", "notaknotcubicspline", "polynomial", "lme"):
                    eff_bc = "not-a-knot"
                self._basis_cs = scipy.interpolate.CubicSpline(
                    self.strikes, eye_u, bc_type=eff_bc, axis=0
                )
                self._fast_basis_coeffs = np.asarray(self._basis_cs.c, dtype=np.float64).copy()
                if use_linear_extrap:
                    self._basis_slope_left = np.asarray(
                        self._basis_cs(self.strikes[0], 1), dtype=np.float64
                    ).copy()
                    self._basis_slope_right = np.asarray(
                        self._basis_cs(self.strikes[-1], 1), dtype=np.float64
                    ).copy()
                else:
                    self._basis_slope_left = np.zeros(self.n_unique_strikes, dtype=np.float64)
                    self._basis_slope_right = np.zeros(self.n_unique_strikes, dtype=np.float64)

            for arr in (self._fast_basis_coeffs, self._basis_slope_left, self._basis_slope_right):
                if arr is not None:
                    arr.setflags(write=False)

    def tenor_from_date(self, dt: Union[datetime, date, str]) -> float:
        """Converts an observation date to the surface's DayCountConvention year fraction."""
        if self.as_of_date is None:
            raise ValueError("as_of_date is not set on RegularizedVolSurface.")
        return _surface_year_fraction(self.as_of_date, dt, self.dcc)

    def map_act365_to_surface_tenor(self, t_act365: float) -> float:
        """Maps an Act365 year fraction to the surface's native DCC coordinate."""
        t_val = float(t_act365)
        if self.dcc.strip().lower() in ("act365", "act365f", "act_365", "act_365f"):
            return t_val
        t365_knots = np.concatenate([np.array([0.0], dtype=np.float64), self.tenors_act365])
        t_dcc_knots = np.concatenate([np.array([0.0], dtype=np.float64), self.tenors])
        if t_val <= t365_knots[-1]:
            return float(np.interp(t_val, t365_knots, t_dcc_knots))
        ratio = float(self.tenors[-1] / max(self.tenors_act365[-1], 1e-12))
        return float(self.tenors[-1] + ratio * (t_val - self.tenors_act365[-1]))

    def strike_basis_weights(
        self, k_query: Union[float, npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """Returns the strike interpolation weight vector w(K) in R^{N_raw_strikes}."""
        self._ensure_basis_spline()
        k_arr = np.asarray(k_query, dtype=np.float64)
        scalar_in = k_arr.ndim == 0
        if scalar_in:
            k_arr = k_arr[None]

        k_min = float(self.strikes[0])
        k_max = float(self.strikes[-1])
        k_clip = np.clip(k_arr, k_min, k_max)
        if self._fast_basis_coeffs is not None:
            w_u = eval_clamped_cubic_coeffs(self.strikes, self._fast_basis_coeffs, k_clip)
        else:
            w_u = np.asarray(self._basis_cs(k_clip), dtype=np.float64)

        left_mask = k_arr < k_min
        right_mask = k_arr > k_max
        if np.any(left_mask) and self._basis_slope_left is not None:
            w_left = np.zeros_like(w_u)
            w_left[..., 0] = 1.0
            w_left = w_left + (k_arr[..., None] - k_min) * self._basis_slope_left
            w_u = np.where(left_mask[..., None], w_left, w_u)
        if np.any(right_mask) and self._basis_slope_right is not None:
            w_right = np.zeros_like(w_u)
            w_right[..., -1] = 1.0
            w_right = w_right + (k_arr[..., None] - k_max) * self._basis_slope_right
            w_u = np.where(right_mask[..., None], w_right, w_u)

        w_raw: npt.NDArray[np.float64] = np.asarray(w_u @ self.dedup_matrix, dtype=np.float64)
        if scalar_in:
            return np.asarray(w_raw[0], dtype=np.float64)
        return w_raw

    def _eval_slice_with_derivs(
        self,
        slice_idx: int,
        k_arr: npt.NDArray[np.float64],
        seg_idx: Optional[npt.NDArray[np.int64]] = None,
        dx: Optional[npt.NDArray[np.float64]] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Evaluates (val, dval/dK, d2val/dK2) on slice `slice_idx` in a single searchsorted pass."""
        k_min = float(self.strikes[0])
        k_max = float(self.strikes[-1])
        if seg_idx is None or dx is None:
            k_clip = np.clip(k_arr, k_min, k_max)
            seg_idx = np.clip(
                np.searchsorted(self.strikes, k_clip, side="right") - 1,
                0,
                self.n_unique_strikes - 2,
            )
            dx = k_clip - self.strikes[seg_idx]

        assert self._fast_slice_coeffs is not None
        c0 = self._fast_slice_coeffs[0, seg_idx, slice_idx]
        c1 = self._fast_slice_coeffs[1, seg_idx, slice_idx]
        c2 = self._fast_slice_coeffs[2, seg_idx, slice_idx]
        c3 = self._fast_slice_coeffs[3, seg_idx, slice_idx]

        val = ((c0 * dx + c1) * dx + c2) * dx + c3
        dval = (3.0 * c0 * dx + 2.0 * c1) * dx + c2
        d2val = 6.0 * c0 * dx + 2.0 * c1

        left_mask = k_arr < k_min
        right_mask = k_arr > k_max
        if np.any(left_mask):
            sl = float(self._slopes_left[slice_idx])
            val = np.where(left_mask, self.quotes[slice_idx, 0] + sl * (k_arr - k_min), val)
            dval = np.where(left_mask, sl, dval)
            d2val = np.where(left_mask, 0.0, d2val)
        if np.any(right_mask):
            sr = float(self._slopes_right[slice_idx])
            val = np.where(right_mask, self.quotes[slice_idx, -1] + sr * (k_arr - k_max), val)
            dval = np.where(right_mask, sr, dval)
            d2val = np.where(right_mask, 0.0, d2val)

        clamped_low = val < self.min_vol
        if np.any(clamped_low):
            val = np.maximum(val, self.min_vol)
            dval = np.where(clamped_low, 0.0, dval)
            d2val = np.where(clamped_low, 0.0, d2val)
        return val, dval, d2val

    def eval_slice(
        self,
        slice_idx: int,
        k_query: Union[float, npt.NDArray[np.float64]],
    ) -> Union[npt.NDArray[np.float64], float]:
        """Evaluates implied volatility on maturity slice `slice_idx` at strike(s) `k_query`."""
        k_arr = np.asarray(k_query, dtype=np.float64)
        scalar_in = k_arr.ndim == 0
        if scalar_in:
            k_arr = k_arr[None]

        k_min = float(self.strikes[0])
        k_max = float(self.strikes[-1])
        k_clip = np.clip(k_arr, k_min, k_max)
        if self._fast_slice_coeffs is not None:
            val = eval_clamped_cubic_coeffs(
                self.strikes, self._fast_slice_coeffs[:, :, slice_idx], k_clip
            )
        else:
            val = np.asarray(self._slice_cs(k_clip)[slice_idx], dtype=np.float64)

        left_mask = k_arr < k_min
        right_mask = k_arr > k_max
        if np.any(left_mask):
            val = np.where(
                left_mask,
                self.quotes[slice_idx, 0] + self._slopes_left[slice_idx] * (k_arr - k_min),
                val,
            )
        if np.any(right_mask):
            val = np.where(
                right_mask,
                self.quotes[slice_idx, -1] + self._slopes_right[slice_idx] * (k_arr - k_max),
                val,
            )

        out = np.maximum(val, self.min_vol)
        if scalar_in:
            return float(out[0])
        return out

    def iv(
        self,
        t: Union[float, datetime, date],
        k_query: Union[float, npt.NDArray[np.float64]],
        *,
        time_interp: Literal["variance", "vol"] = "variance",
    ) -> Union[npt.NDArray[np.float64], float]:
        """Evaluates implied volatility sigma_IV(t, K) in float64."""
        if isinstance(t, (datetime, date)):
            t_raw = self.tenor_from_date(t)
        else:
            t_raw = float(t)
        t_val = max(t_raw, 1e-8)
        if t_val <= float(self.tenors[0]):
            return self.eval_slice(0, k_query)
        if t_val >= float(self.tenors[-1]):
            return self.eval_slice(self.n_exp - 1, k_query)

        idx = int(np.searchsorted(self.tenors, t_val, side="right")) - 1
        idx = max(0, min(idx, self.n_exp - 2))
        t0 = float(self.tenors[idx])
        t1 = float(self.tenors[idx + 1])
        v0 = np.asarray(self.eval_slice(idx, k_query), dtype=np.float64)
        v1 = np.asarray(self.eval_slice(idx + 1, k_query), dtype=np.float64)

        alpha = (t_val - t0) / max(t1 - t0, 1e-12)
        if time_interp == "variance":
            w0 = (v0 * v0) * t0
            w1 = (v1 * v1) * t1
            w = (1.0 - alpha) * w0 + alpha * w1
            res = np.sqrt(np.maximum(w / t_val, self.min_vol * self.min_vol))
        else:
            res = np.maximum((1.0 - alpha) * v0 + alpha * v1, self.min_vol)

        if np.ndim(k_query) == 0:
            return float(res)
        return np.asarray(res, dtype=np.float64)

    def iv_batch(
        self,
        tenors: Union[Sequence[float], Sequence[datetime], Sequence[date], npt.NDArray[np.float64]],
        strikes: Union[Sequence[float], npt.NDArray[np.float64]],
        *,
        time_interp: Literal["variance", "vol"] = "variance",
    ) -> npt.NDArray[np.float64]:
        """Vectorized float64 evaluation of sigma_IV(t_j, K_j) across arbitrary arrays."""
        if isinstance(tenors, Sequence) and len(tenors) > 0 and isinstance(tenors[0], (datetime, date)):
            t_arr = np.asarray([self.tenor_from_date(d) for d in tenors], dtype=np.float64)  # type: ignore[arg-type]
        else:
            t_arr = np.asarray(tenors, dtype=np.float64)
        k_arr = np.asarray(strikes, dtype=np.float64)
        t_b, k_b = np.broadcast_arrays(t_arr, k_arr)
        orig_shape = t_b.shape
        t_flat = np.maximum(t_b.reshape(-1), 1e-8)
        k_flat = k_b.reshape(-1)
        n_q = len(t_flat)
        if n_q == 0:
            return np.empty(orig_shape, dtype=np.float64)

        k_min = float(self.strikes[0])
        k_max = float(self.strikes[-1])
        k_clip = np.clip(k_flat, k_min, k_max)
        seg_idx = np.clip(
            np.searchsorted(self.strikes, k_clip, side="right") - 1,
            0,
            self.n_unique_strikes - 2,
        )
        dx = k_clip - self.strikes[seg_idx]

        # Evaluate all slices at k_flat in one vectorized gather
        assert self._fast_slice_coeffs is not None
        c = self._fast_slice_coeffs[:, seg_idx, :]  # (4, N_q, N_exp)
        dx_c = dx[:, None]
        vals_all = ((c[0] * dx_c + c[1]) * dx_c + c[2]) * dx_c + c[3]  # (N_q, N_exp)

        left_mask = k_flat < k_min
        right_mask = k_flat > k_max
        if np.any(left_mask):
            vals_left = self.quotes[None, :, 0] + self._slopes_left[None, :] * (k_flat[:, None] - k_min)
            vals_all = np.where(left_mask[:, None], vals_left, vals_all)
        if np.any(right_mask):
            vals_right = self.quotes[None, :, -1] + self._slopes_right[None, :] * (k_flat[:, None] - k_max)
            vals_all = np.where(right_mask[:, None], vals_right, vals_all)
        vals_all = np.maximum(vals_all, self.min_vol)

        if self.n_exp == 1:
            return np.asarray(vals_all[:, 0].reshape(orig_shape), dtype=np.float64)

        idx0 = np.clip(np.searchsorted(self.tenors, t_flat, side="right") - 1, 0, self.n_exp - 2)
        idx1 = idx0 + 1
        rows = np.arange(n_q, dtype=np.int64)
        v0 = vals_all[rows, idx0]
        v1 = vals_all[rows, idx1]
        t0 = self.tenors[idx0]
        t1 = self.tenors[idx1]
        alpha = (t_flat - t0) / np.maximum(t1 - t0, 1e-12)

        if time_interp == "variance":
            w = (1.0 - alpha) * (v0 * v0 * t0) + alpha * (v1 * v1 * t1)
            res = np.sqrt(np.maximum(w / t_flat, self.min_vol * self.min_vol))
        else:
            res = np.maximum((1.0 - alpha) * v0 + alpha * v1, self.min_vol)

        res = np.where(t_flat <= float(self.tenors[0]), vals_all[:, 0], res)
        res = np.where(t_flat >= float(self.tenors[-1]), vals_all[:, -1], res)
        return res.reshape(orig_shape)

    def iv_and_d_iv_d_quotes(
        self,
        t: Union[float, datetime, date],
        k_query: float,
    ) -> Tuple[float, npt.NDArray[np.float64]]:
        """Computes sigma_IV(t, K) and its exact analytical Jacobian wrt raw quotes (N_exp, N_raw_strikes)."""
        if isinstance(t, (datetime, date)):
            t_raw = self.tenor_from_date(t)
        else:
            t_raw = float(t)
        t_val = max(t_raw, 1e-8)
        k_val = float(k_query)
        w_k = self.strike_basis_weights(k_val)  # (N_raw_strikes,)
        w_u = self.dedup_matrix @ w_k
        # Actually w_k = w_u_raw @ dedup_matrix, so compute directly in unique strike space then contract
        jac_u = np.zeros((self.n_exp, self.n_unique_strikes), dtype=np.float64)

        # Obtain exact basis weights in unique strike space
        k_arr = np.array([k_val], dtype=np.float64)
        k_min = float(self.strikes[0])
        k_max = float(self.strikes[-1])
        k_clip = np.clip(k_arr, k_min, k_max)
        self._ensure_basis_spline()
        if self._fast_basis_coeffs is not None:
            w_u_vec = eval_clamped_cubic_coeffs(self.strikes, self._fast_basis_coeffs, k_clip)[0]
        else:
            w_u_vec = np.asarray(self._basis_cs(k_clip)[0], dtype=np.float64)
        if k_val < k_min and self._basis_slope_left is not None:
            w_u_vec = np.zeros(self.n_unique_strikes, dtype=np.float64)
            w_u_vec[0] = 1.0
            w_u_vec = w_u_vec + (k_val - k_min) * self._basis_slope_left
        elif k_val > k_max and self._basis_slope_right is not None:
            w_u_vec = np.zeros(self.n_unique_strikes, dtype=np.float64)
            w_u_vec[-1] = 1.0
            w_u_vec = w_u_vec + (k_val - k_max) * self._basis_slope_right
        _ = w_u

        if t_val <= float(self.tenors[0]):
            sig = float(self.eval_slice(0, k_val))
            jac_u[0, :] = w_u_vec
        elif t_val >= float(self.tenors[-1]):
            sig = float(self.eval_slice(self.n_exp - 1, k_val))
            jac_u[self.n_exp - 1, :] = w_u_vec
        else:
            idx = int(np.searchsorted(self.tenors, t_val, side="right")) - 1
            idx = max(0, min(idx, self.n_exp - 2))
            t0 = float(self.tenors[idx])
            t1 = float(self.tenors[idx + 1])
            v0 = float(self.eval_slice(idx, k_val))
            v1 = float(self.eval_slice(idx + 1, k_val))

            alpha = (t_val - t0) / max(t1 - t0, 1e-12)
            w0 = (v0 * v0) * t0
            w1 = (v1 * v1) * t1
            w = (1.0 - alpha) * w0 + alpha * w1
            sig = math.sqrt(max(w / t_val, self.min_vol * self.min_vol))

            d_sig_dv0 = (1.0 - alpha) * t0 * v0 / max(t_val * sig, 1e-12)
            d_sig_dv1 = alpha * t1 * v1 / max(t_val * sig, 1e-12)
            jac_u[idx, :] = d_sig_dv0 * w_u_vec
            jac_u[idx + 1, :] = d_sig_dv1 * w_u_vec

        # Contract through calendar-projection + min-vol floor Jacobian (T1M-013)
        jac_dedup = np.einsum("mu,mnu->nu", jac_u, self.calendar_proj_jacobian)
        jac = np.asarray(jac_dedup @ self.dedup_matrix, dtype=np.float64)
        return sig, jac

    def local_vol(
        self,
        t: float,
        s_spot: Union[float, npt.NDArray[np.float64]],
        *,
        forward_at_t: Optional[float] = None,
        r_zero: float = 0.0,
        q_repo: float = 0.0,
        spot_t0: Optional[float] = None,
        cash_div_leverage: float = 1.0,
    ) -> Union[npt.NDArray[np.float64], float]:
        """Evaluates Dupire local volatility sigma_loc^X(t, S) using analytical cubic derivatives."""
        t_eff = max(float(t), 1e-4)
        s_arr = np.maximum(np.asarray(s_spot, dtype=np.float64), 1e-4)
        scalar_in = s_arr.ndim == 0
        if scalar_in:
            s_arr = s_arr[None]

        k_min = float(self.strikes[0])
        k_max = float(self.strikes[-1])
        k_clip = np.clip(s_arr, k_min, k_max)
        seg_idx = np.clip(
            np.searchsorted(self.strikes, k_clip, side="right") - 1,
            0,
            self.n_unique_strikes - 2,
        )
        dx = k_clip - self.strikes[seg_idx]

        if t_eff <= float(self.tenors[0]):
            sig, dsig_dk, d2sig_dk2 = self._eval_slice_with_derivs(0, s_arr, seg_idx, dx)
            w = (sig * sig) * t_eff
            dw_dt = sig * sig
            dw_dk = 2.0 * t_eff * sig * dsig_dk
            d2w_dk2 = 2.0 * t_eff * (dsig_dk * dsig_dk + sig * d2sig_dk2)
        elif t_eff >= float(self.tenors[-1]):
            sig, dsig_dk, d2sig_dk2 = self._eval_slice_with_derivs(self.n_exp - 1, s_arr, seg_idx, dx)
            w = (sig * sig) * t_eff
            dw_dt = sig * sig
            dw_dk = 2.0 * t_eff * sig * dsig_dk
            d2w_dk2 = 2.0 * t_eff * (dsig_dk * dsig_dk + sig * d2sig_dk2)
        else:
            idx = int(np.searchsorted(self.tenors, t_eff, side="right")) - 1
            idx = max(0, min(idx, self.n_exp - 2))
            t0 = float(self.tenors[idx])
            t1 = float(self.tenors[idx + 1])
            dt_inv = 1.0 / max(t1 - t0, 1e-12)
            alpha = (t_eff - t0) * dt_inv

            v0, dv0_dk, d2v0_dk2 = self._eval_slice_with_derivs(idx, s_arr, seg_idx, dx)
            v1, dv1_dk, d2v1_dk2 = self._eval_slice_with_derivs(idx + 1, s_arr, seg_idx, dx)

            w0 = (v0 * v0) * t0
            w1 = (v1 * v1) * t1
            w = (1.0 - alpha) * w0 + alpha * w1
            dw_dt = (w1 - w0) * dt_inv

            dw0_dk = 2.0 * t0 * v0 * dv0_dk
            dw1_dk = 2.0 * t1 * v1 * dv1_dk
            dw_dk = (1.0 - alpha) * dw0_dk + alpha * dw1_dk

            d2w0_dk2 = 2.0 * t0 * (dv0_dk * dv0_dk + v0 * d2v0_dk2)
            d2w1_dk2 = 2.0 * t1 * (dv1_dk * dv1_dk + v1 * d2v1_dk2)
            d2w_dk2 = (1.0 - alpha) * d2w0_dk2 + alpha * d2w1_dk2

        if forward_at_t is not None:
            f_t = max(float(forward_at_t), 1e-4)
        elif spot_t0 is not None:
            f_t = max(float(spot_t0) * math.exp((r_zero - q_repo) * t_eff), 1e-4)
        else:
            f_t = float(np.mean(s_arr))

        y = np.log(s_arr / f_t)
        w_safe = np.maximum(w, 1e-8)
        dw_dy = s_arr * dw_dk
        dw_dt_y = dw_dt + (float(r_zero) - float(q_repo)) * dw_dy
        d2w_dy2 = (s_arr * s_arr) * d2w_dk2 + dw_dy

        denom = (
            1.0
            - (y / w_safe) * dw_dy
            + 0.25 * (-0.25 - 1.0 / w_safe + (y * y) / (w_safe * w_safe)) * (dw_dy * dw_dy)
            + 0.5 * d2w_dy2
        )
        denom_safe = np.maximum(denom, 0.05)
        loc_var = np.maximum(dw_dt_y / denom_safe, self.min_vol * self.min_vol)
        lv_s = np.sqrt(loc_var)
        lv_x = np.clip(float(cash_div_leverage) * lv_s, self.min_vol, self.max_local_vol)

        if scalar_in:
            return float(lv_x[0])
        return np.asarray(lv_x, dtype=np.float64)

    def extract_node_surface_features_values_only(
        self,
        t_short: float,
        f_short: float,
        t_term: float,
        f_term: float,
        *,
        iv_short_val: Optional[float] = None,
        iv_term_val: Optional[float] = None,
    ) -> npt.NDArray[np.float64]:
        """Fast value-only extraction of the 7 surface features for AutocallNodeCol (cols 4..10)."""
        iv_short = (
            float(self.iv(t_short, float(f_short)))
            if iv_short_val is None
            else float(iv_short_val)
        )
        if iv_term_val is None:
            k_term_vec = np.array(
                [f_term, 0.90 * f_term, 1.10 * f_term, 0.80 * f_term, 1.20 * f_term],
                dtype=np.float64,
            )
            iv_term_vec = np.asarray(self.iv(t_term, k_term_vec), dtype=np.float64)
            iv_term = float(iv_term_vec[0])
            iv_m10 = float(iv_term_vec[1])
            iv_p10 = float(iv_term_vec[2])
            iv_m20 = float(iv_term_vec[3])
            iv_p20 = float(iv_term_vec[4])
        else:
            iv_term = float(iv_term_val)
            k_term_vec = np.array(
                [0.90 * f_term, 1.10 * f_term, 0.80 * f_term, 1.20 * f_term],
                dtype=np.float64,
            )
            iv_term_vec = np.asarray(self.iv(t_term, k_term_vec), dtype=np.float64)
            iv_m10 = float(iv_term_vec[0])
            iv_p10 = float(iv_term_vec[1])
            iv_m20 = float(iv_term_vec[2])
            iv_p20 = float(iv_term_vec[3])

        iv_slope = iv_term - iv_short
        denom_skew = max(0.20 * iv_term, 1e-6)
        raw_skew = (iv_p10 - iv_m10) / denom_skew
        ssvi_rho = math.tanh(raw_skew)

        denom_curv = max(0.01 * iv_term, 1e-6)
        raw_curv = (iv_p10 - 2.0 * iv_term + iv_m10) / denom_curv
        ssvi_eta = raw_curv if raw_curv > 1e-4 else 1e-4

        pc1 = iv_m20 - iv_term
        pc2 = iv_p20 - iv_term

        return np.array(
            [iv_short, iv_term, iv_slope, ssvi_rho, ssvi_eta, pc1, pc2],
            dtype=np.float64,
        )

    def extract_node_surface_features(
        self,
        t_short: float,
        f_short: float,
        t_term: float,
        f_term: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Extracts the 7 surface features for AutocallNodeCol (cols 4..10) + exact quote Jacobian."""
        iv_short, j_short = self.iv_and_d_iv_d_quotes(t_short, f_short)
        iv_term, j_term = self.iv_and_d_iv_d_quotes(t_term, f_term)
        iv_m10, j_m10 = self.iv_and_d_iv_d_quotes(t_term, 0.90 * f_term)
        iv_p10, j_p10 = self.iv_and_d_iv_d_quotes(t_term, 1.10 * f_term)
        iv_m20, j_m20 = self.iv_and_d_iv_d_quotes(t_term, 0.80 * f_term)
        iv_p20, j_p20 = self.iv_and_d_iv_d_quotes(t_term, 1.20 * f_term)

        iv_slope = iv_term - iv_short
        j_slope = j_term - j_short

        denom_skew = max(0.20 * iv_term, 1e-6)
        raw_skew = (iv_p10 - iv_m10) / denom_skew
        ssvi_rho = math.tanh(raw_skew)
        sech2 = 1.0 - ssvi_rho * ssvi_rho
        j_raw_skew = (j_p10 - j_m10) / denom_skew - (raw_skew / max(iv_term, 1e-6)) * j_term
        j_rho = sech2 * j_raw_skew

        denom_curv = max(0.01 * iv_term, 1e-6)
        raw_curv = (iv_p10 - 2.0 * iv_term + iv_m10) / denom_curv
        if raw_curv > 1e-4:
            ssvi_eta = raw_curv
            j_eta = (j_p10 - 2.0 * j_term + j_m10) / denom_curv - (
                raw_curv / max(iv_term, 1e-6)
            ) * j_term
        else:
            ssvi_eta = 1e-4
            j_eta = np.zeros_like(j_term)

        pc1 = iv_m20 - iv_term
        j_pc1 = j_m20 - j_term

        pc2 = iv_p20 - iv_term
        j_pc2 = j_p20 - j_term

        feats = np.array(
            [iv_short, iv_term, iv_slope, ssvi_rho, ssvi_eta, pc1, pc2],
            dtype=np.float64,
        )
        jac = np.stack([j_short, j_term, j_slope, j_rho, j_eta, j_pc1, j_pc2], axis=0)
        return feats, jac


@dataclass
class DupireQMCResult:
    """Output of the multi-asset Dupire Local Volatility / Terminal IV Sobol QMC solver."""

    product_value: float
    standard_error: float
    option_pv: float
    option_se: float
    notional_pv: float
    coupon_pv: float
    funding_pv: float
    p_stop: npt.NDArray[np.float64]
    q_cpn: npt.NDArray[np.float64]
    l_put: float
    l_call: float
    cashflows: List[CashFlowItem] = field(default_factory=list)
    n_paths: int = 32768
    mode: str = "dupire_lv"
    l_onestar: float = 0.0
    p_onestar: float = 0.0


DupireQMCOutput = DupireQMCResult


def _resolve_r_native_obs_tensor(state: Any) -> torch.Tensor:
    """Resolves native-currency cumulative forward carry integral (excluding quanto drift)."""
    r_nat = getattr(state, "r_native_obs", None)
    if r_nat is None:
        r_nat = getattr(state.dividend_schedule, "r_native_obs", None)
    if r_nat is not None:
        return cast(torch.Tensor, r_nat)
    q_meta = getattr(state.dividend_schedule, "quanto_meta", None)
    df_fin = getattr(state.dividend_schedule, "df_fin_obs", None)
    df_repo = getattr(state.dividend_schedule, "df_repo_obs", None)
    if q_meta and df_fin is not None and df_repo is not None:
        return -torch.log(torch.clamp(df_fin, min=1e-15)) + torch.log(torch.clamp(df_repo, min=1e-15))
    return cast(torch.Tensor, state.dividend_schedule.r_tot_obs)


def _digital_ramp_np(
    w_vals: npt.NDArray[np.float64],
    barrier: float,
    spread: float,
) -> npt.NDArray[np.float64]:
    """Evaluates the call-spread digital ramp I_{Delta B}(W, B) on pathwise index values in float64."""
    if abs(spread) < 1e-9:
        return (w_vals >= barrier).astype(np.float64)
    b_low = min(barrier, barrier + spread)
    b_high = max(barrier, barrier + spread)
    return np.clip((w_vals - b_low) / (b_high - b_low), 0.0, 1.0)


def _next_power_of_two(n: int) -> int:
    """Returns the smallest power of 2 >= max(n, 2)."""
    n_val = max(int(n), 2)
    return 1 << (n_val - 1).bit_length()


def _generate_sobol_normals(
    n_paths: int,
    dim: int,
    *,
    scramble: bool = True,
    seed: Optional[int] = None,
) -> npt.NDArray[np.float64]:
    """Generates power-of-2 Sobol normal variates (N_paths, dim), falling back to antithetic PCG64 above 21,201."""
    if dim <= 0:
        return np.empty((n_paths, 0), dtype=np.float64)

    n_p2 = _next_power_of_two(n_paths)
    m_pow = int(round(math.log2(n_p2)))

    if dim <= _SOBOL_MAX_DIM:
        sampler = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
        u_raw = sampler.random_base2(m=m_pow)
        if not scramble:
            u_shifted = (u_raw + 0.5 / float(n_p2)) % 1.0
        else:
            u_shifted = u_raw
        u_clipped = np.clip(u_shifted[:n_paths], 1e-12, 1.0 - 1e-12)
        return np.asarray(norm.ppf(u_clipped), dtype=np.float64)

    # Hybrid: Sobol for first 21,201 dimensions + antithetic PCG64 normals for trailing dimensions
    sampler = qmc.Sobol(d=_SOBOL_MAX_DIM, scramble=scramble, seed=seed)
    u_sobol = sampler.random_base2(m=m_pow)
    if not scramble:
        u_sobol = (u_sobol + 0.5 / float(n_p2)) % 1.0
    u_sobol = np.clip(u_sobol[:n_paths], 1e-12, 1.0 - 1e-12)
    z_sobol = np.asarray(norm.ppf(u_sobol), dtype=np.float64)

    rem_dim = dim - _SOBOL_MAX_DIM
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2
    z_half = rng.standard_normal((half, rem_dim))
    z_anti = np.concatenate([z_half, -z_half], axis=0)[:n_paths]
    return np.concatenate([z_sobol, z_anti], axis=1)


def _compute_funding_weights_np(state: Any) -> npt.NDArray[np.float64]:
    """Computes deterministic per-period discounted FundingLeg weights w_fund[m] in float64."""
    m_rem = int(state.m_rem)
    w_fund = np.zeros(m_rem, dtype=np.float64)
    if state.funding_leg is None or m_rem == 0:
        return w_fund

    from deepvol.models.autocall.escrowed_dividend import build_discount_rate_evaluator

    f_leg = state.funding_leg
    f_rid = f_leg.get("rate_identifier")
    f_curve = f_leg.get("funding_curve")
    as_of_dt = state.seasoned_state.as_of_date
    dcc = state.seasoned_state.day_count_convention

    _, df_fin_fn = build_discount_rate_evaluator(state.discount_curve, as_of_dt)
    disc_id = getattr(state.discount_curve, "identifier", None)
    f_curve_id = getattr(f_curve, "identifier", None)
    if (
        f_rid is not None
        and f_curve is not None
        and (f_curve is not state.discount_curve)
        and (disc_id is None or f_curve_id is None or f_curve_id != disc_id)
    ):
        _, df_fwd_fn = build_discount_rate_evaluator(f_curve, as_of_dt)
    else:
        df_fwd_fn = df_fin_fn

    f_spread = float(f_leg.get("funding_spread", 0.0))
    f_dates = f_leg.get("payment_dates", [])
    prev_f_dt = getattr(state.seasoned_state, "start_date", as_of_dt)
    for f_dt in f_dates:
        if f_dt <= as_of_dt:
            prev_f_dt = f_dt
            continue
        alpha_j = _surface_year_fraction(prev_f_dt, f_dt, dcc)
        df_curr = float(df_fin_fn(f_dt))
        if f_rid is None:
            r_fwd_j = 0.0
        else:
            df_prev_fwd = float(df_fwd_fn(prev_f_dt))
            df_curr_fwd = float(df_fwd_fn(f_dt))
            r_fwd_j = (df_prev_fwd / max(df_curr_fwd, 1e-12) - 1.0) / max(alpha_j, 1e-8)

        m_idx = m_rem - 1
        for m in range(m_rem):
            if state.seasoned_state.future_pay_dates[m] >= f_dt:
                m_idx = m
                break
        w_fund[m_idx] += df_curr * (r_fwd_j + f_spread) * alpha_j
        prev_f_dt = f_dt
    return w_fund


def _evaluate_index_with_unperm_np(
    perf_sorted: npt.NDArray[np.float64],
    index_name: str,
    inv_lex_perm: Optional[npt.NDArray[np.int64]],
) -> npt.NDArray[np.float64]:
    """Evaluates monitoring functional, restoring original caller ordering for asymmetric pair indices."""
    idx_clean = str(index_name).strip()
    if idx_clean in _ASYMMETRIC_PAIR_INDICES and inv_lex_perm is not None:
        perf_in = perf_sorted[..., inv_lex_perm]
    else:
        perf_in = perf_sorted
    return np.asarray(evaluate_monitoring_index_np(perf_in, idx_clean), dtype=np.float64)


def _compute_forward_at_time_np(
    spots_t0: npt.NDArray[np.float64],
    r_tot_obs: npt.NDArray[np.float64],
    tau_obs: npt.NDArray[np.float64],
    div_events_list: Sequence[Sequence[Any]],
    t_step: float,
    period_idx: int,
) -> npt.NDArray[np.float64]:
    """Computes exact escrowed forward F_i(0, t_step) honoring discrete dividend ex-date jumps."""
    d_assets = len(spots_t0)
    t_prev = 0.0 if period_idx == 0 else float(tau_obs[period_idx - 1])
    t_curr = float(tau_obs[period_idx])
    alpha = (t_step - t_prev) / max(t_curr - t_prev, 1e-12)
    r_prev = np.zeros(d_assets, dtype=np.float64) if period_idx == 0 else r_tot_obs[:, period_idx - 1]
    r_curr = r_tot_obs[:, period_idx]
    r_t = r_prev + alpha * (r_curr - r_prev)

    d_pv = np.zeros(d_assets, dtype=np.float64)
    for i in range(d_assets):
        acc = 0.0
        for ev in div_events_list[i]:
            tau_ex = float(getattr(ev, "tau_ex_act365", 0.0))
            if 0.0 < tau_ex <= t_step + 1e-12:
                acc += float(getattr(ev, "amount", 0.0)) * float(
                    getattr(ev, "pv_discount_weight", 0.0)
                )
        d_pv[i] = acc

    escrowed = np.maximum(spots_t0 - d_pv, 0.01 * spots_t0)
    return np.asarray(escrowed * np.exp(r_t), dtype=np.float64)


def _assemble_qmc_postproc(
    state: Any,
    *,
    p_stop_arr: npt.NDArray[np.float64],
    q_cpn_arr: npt.NDArray[np.float64],
    cpn_cf_expected: npt.NDArray[np.float64],
    df_pay: npt.NDArray[np.float64],
    l_put_val: float,
    l_call_val: float,
    l_onestar_val: float,
    p_onestar_val: float = 0.0,
    option_se: float,
    total_se: float,
    assemble_fn: Callable[..., Any],
) -> Any:
    """Assembles postproc cashflows & PV, handling pre-merge and post-merge signatures cleanly."""
    p_t = torch.as_tensor(p_stop_arr, dtype=torch.float64)
    q_t = torch.as_tensor(q_cpn_arr, dtype=torch.float64)
    l_put_t = torch.tensor(l_put_val, dtype=torch.float64)
    l_call_t = torch.tensor(l_call_val, dtype=torch.float64)
    l_onestar_t = torch.tensor(l_onestar_val, dtype=torch.float64)
    sig_params = inspect.signature(assemble_fn).parameters
    extra_kwargs: dict[str, Any] = {}
    if "p_onestar" in sig_params:
        extra_kwargs["p_onestar"] = torch.tensor(float(p_onestar_val), dtype=torch.float64)

    if _asm_supports_coupon_cf(assemble_fn):
        asm = assemble_fn(
            state,
            p_stop=p_t,
            q_cpn=q_t,
            l_put=l_put_t,
            l_call=l_call_t,
            l_onestar=l_onestar_t,
            option_se=option_se,
            total_se=total_se,
            coupon_cf_undisc=torch.as_tensor(cpn_cf_expected, dtype=torch.float64),
            **extra_kwargs,
        )
        if "p_onestar" not in sig_params and bool(getattr(state, "is_long_basket", False)) and len(df_pay) > 0:
            put_fix_delta = -2.0 * float(df_pay[-1]) * float(state.option_gearing) * float(l_put_val)
            asm.option_pv = asm.option_pv + put_fix_delta
            asm.product_value = asm.product_value + put_fix_delta
        return asm

    asm = assemble_fn(
        state,
        p_stop=p_t,
        q_cpn=q_t,
        l_put=l_put_t,
        l_call=l_call_t,
        l_onestar=l_onestar_t,
        option_se=option_se,
        total_se=total_se,
        **extra_kwargs,
    )
    # Pre-merge compatibility patch: overwrite marginal-independence coupon rows with pathwise expectations
    disc_cpn_cfs = df_pay * cpn_cf_expected
    true_cpn_pv = float(np.sum(disc_cpn_cfs))
    cpn_delta = true_cpn_pv - float(asm.coupon_pv.item())
    asm.coupon_pv = torch.tensor(true_cpn_pv, dtype=torch.float64)
    asm.product_value = asm.product_value + cpn_delta

    cpn_row_idx = 0
    for idx_cf, cf in enumerate(asm.cashflows):
        if cf.Description == "Coupon" and cf.StatusText == "Estimate":
            if cpn_row_idx < len(disc_cpn_cfs):
                asm.cashflows[idx_cf] = CashFlowItem(
                    Description=cf.Description,
                    Date=cf.Date,
                    Amount=float(disc_cpn_cfs[cpn_row_idx]),
                    StandardError=cf.StandardError,
                    Currency=cf.Currency,
                    Leg=cf.Leg,
                    StatusText=cf.StatusText,
                )
                cpn_row_idx += 1
    return asm


def price_eq_autocall_dupire_qmc(
    state_or_request: Any,
    *,
    n_paths: int = 32768,
    n_replicates: int = 8,
    mode: Literal["dupire_lv", "terminal_iv", "auto"] = "auto",
    n_substeps_per_period: int = 16,
    scramble: bool = True,
    seed: Optional[int] = None,
    compute_greeks: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    use_gpu: Optional[bool] = None,
    precision: Literal["mixed_fp32_fp64", "fp64_audit"] = "mixed_fp32_fp64",
    cv_mode: Literal["bs1d_exact", "multifidelity_gbm", "none"] = "bs1d_exact",
) -> DupireQMCResult:
    """Vectorized Owen-scrambled RQMC solver for Multi-Asset Equity Autocallables.

    Routes GPU requests (`device="cuda"` or `use_gpu=True`) cleanly to `DupireQMCGpuSolver`
    while providing an exact IEEE-754 float64 CPU Owen-scrambled RQMC reference path.
    """
    from deepvol.models.autocall.pre_post_processor import (
        AutocallPreprocState,
        assemble_autocall_cashflows_and_pv,
        preprocess_eq_autocall,
    )

    _ = compute_greeks

    if isinstance(state_or_request, AutocallPreprocState):
        state = state_or_request
    else:
        state = preprocess_eq_autocall(state_or_request)

    spots_np_chk = state.spots_t0.detach().cpu().numpy().astype(np.float64)
    ref_np_chk = state.ref_spots.detach().cpu().numpy().astype(np.float64)
    if not (np.all(np.isfinite(spots_np_chk)) and np.all(spots_np_chk > 0.0)):
        raise ValueError("Spots must be strictly positive and finite.")
    if not (np.all(np.isfinite(ref_np_chk)) and np.all(ref_np_chk > 0.0)):
        raise ValueError("UnderlyingReferenceSpots must be strictly positive and finite.")

    want_gpu = False
    if use_gpu is True:
        want_gpu = True
    elif device is not None and str(device).startswith("cuda"):
        want_gpu = True
    elif use_gpu is None and device is None and state.spots_t0.is_cuda:
        want_gpu = True

    if want_gpu and precision != "fp64_audit" and torch.cuda.is_available():
        from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCConfig, DupireQMCGpuSolver

        gpu_eff_mode: Literal["dupire_lv", "terminal_iv"] = (
            "terminal_iv" if mode == "terminal_iv" else "dupire_lv"
        )
        cfg = DupireQMCConfig(
            n_paths=int(n_paths),
            n_replicates=int(n_replicates),
            n_substeps=int(n_substeps_per_period),
            mode=gpu_eff_mode,
            precision=precision,
            cv_mode=cv_mode,
            seed=42 if seed is None else int(seed),
        )
        return DupireQMCGpuSolver(device=device or "cuda").simulate_and_price(state, config=cfg)

    d_assets = state.d_assets
    m_rem = state.m_rem

    if m_rem == 0 or state.seasoned_state.already_autocalled:
        z_m = np.zeros(m_rem, dtype=np.float64)
        asm = assemble_autocall_cashflows_and_pv(
            state,
            p_stop=torch.zeros(m_rem, dtype=torch.float64),
            q_cpn=torch.zeros(m_rem, dtype=torch.float64),
            l_put=torch.tensor(0.0, dtype=torch.float64),
            option_se=0.0,
        )
        return DupireQMCResult(
            product_value=float(asm.product_value.item()),
            standard_error=0.0,
            option_pv=0.0,
            option_se=0.0,
            notional_pv=float(asm.notional_pv.item()),
            coupon_pv=float(asm.coupon_pv.item()),
            funding_pv=float(asm.funding_pv.item()),
            p_stop=z_m,
            q_cpn=z_m,
            l_put=0.0,
            l_call=0.0,
            cashflows=asm.cashflows,
            n_paths=n_paths,
            mode=mode,
            l_onestar=0.0,
        )

    # Power-of-2 RQMC replicate layout (T1M-009, T1M-024)
    eff_r = max(1, int(n_replicates)) if scramble else 1
    if eff_r > 1:
        n_rep = _next_power_of_two(max(2, int(n_paths) // eff_r))
        n_total_paths = eff_r * n_rep
    else:
        n_rep = _next_power_of_two(max(2, int(n_paths)))
        n_total_paths = n_rep

    corr_np = state.corr_matrix.detach().cpu().numpy().astype(np.float64)
    corr_sym = 0.5 * (corr_np + corr_np.T)
    eigvals, eigvecs = np.linalg.eigh(corr_sym)
    if np.min(eigvals) < 1e-8:
        eigvals = np.maximum(eigvals, 1e-6)
        corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d_inv = 1.0 / np.sqrt(np.diag(corr_psd))
        corr_sym = corr_psd * np.outer(d_inv, d_inv)
    chol_l = np.linalg.cholesky(corr_sym)

    tau_obs = state.tau_obs.detach().cpu().numpy().astype(np.float64)
    df_pay = state.df_pay.detach().cpu().numpy().astype(np.float64)
    ko_barriers = state.ko_barriers.detach().cpu().numpy().astype(np.float64)
    is_callable = state.is_callable.detach().cpu().numpy().astype(np.float64)
    cpn_barriers = state.cpn_barriers.detach().cpu().numpy().astype(np.float64)
    is_guaranteed_cpn = state.is_guaranteed_cpn.detach().cpu().numpy().astype(np.float64)
    coupon_amounts = state.coupon_amounts.detach().cpu().numpy().astype(np.float64)

    fwd_prices = state.dividend_schedule.forward_price.detach().cpu().numpy().astype(np.float64)
    iv_at_fwd = state.iv_at_fwd.detach().cpu().numpy().astype(np.float64)
    ref_spots = ref_np_chk
    spots_t0 = spots_np_chk

    base_seed = 42 if seed is None else int(seed)
    total_dim = m_rem * d_assets
    eff_mode = "dupire_lv" if mode == "auto" else mode
    n_sub = max(int(n_substeps_per_period), 1)

    perf_paths = np.empty((n_total_paths, m_rem, d_assets), dtype=np.float64)
    surfs_ordered = [state.vol_surfaces[ticker] for ticker in state.sorted_identifiers]
    div_events_map = getattr(state.dividend_schedule, "dividend_events", {})
    div_events_ordered = [div_events_map.get(ticker, []) for ticker in state.sorted_identifiers]
    df_fin_obs_np = state.dividend_schedule.df_fin_obs.detach().cpu().numpy()
    r_tot_obs_np = state.dividend_schedule.r_tot_obs.detach().cpu().numpy()
    r_native_obs_np = _resolve_r_native_obs_tensor(state).detach().cpu().numpy().astype(np.float64)
    has_quanto_diff = not bool(np.allclose(r_native_obs_np, r_tot_obs_np, rtol=1e-14, atol=1e-14))
    escrowed_spot_t = getattr(state.dividend_schedule, "escrowed_spot", None)
    escrowed_spot_np = (
        escrowed_spot_t.detach().cpu().numpy().astype(np.float64)
        if escrowed_spot_t is not None
        else None
    )
    future_obs_dates = getattr(state.seasoned_state, "future_obs_dates", [])

    # Precompute surface DCC tenors and Act365-to-DCC variance-time scale factors (T1M-012)
    tau_surf_obs = np.empty((d_assets, m_rem), dtype=np.float64)
    for i, surf_i in enumerate(surfs_ordered):
        if surf_i.as_of_date is not None and len(future_obs_dates) == m_rem:
            tau_surf_obs[i, :] = [surf_i.tenor_from_date(dt) for dt in future_obs_dates]
        else:
            tau_surf_obs[i, :] = [surf_i.map_act365_to_surface_tenor(float(t_m)) for t_m in tau_obs]

    # Generate Owen-scrambled Sobol nets for all replicates and diffuse vectorized across (eff_r, n_rep)
    z_sobol_all = np.empty((eff_r, n_rep, total_dim), dtype=np.float64)
    for rep_idx in range(eff_r):
        rep_seed = base_seed + rep_idx * 1009
        z_sobol_all[rep_idx] = _generate_sobol_normals(
            n_rep, total_dim, scramble=scramble, seed=rep_seed
        )

    if eff_mode == "terminal_iv":
        log_x_prev = np.zeros((eff_r, n_rep, d_assets), dtype=np.float64)
        var_prev = np.zeros(d_assets, dtype=np.float64)
        for m in range(m_rem):
            z_m = z_sobol_all[:, :, m * d_assets : (m + 1) * d_assets] @ chol_l.T
            var_curr = np.empty(d_assets, dtype=np.float64)
            for i, surf_i in enumerate(surfs_ordered):
                if surf_i.dcc.strip().lower() not in ("act365", "act365f", "act_365", "act_365f"):
                    iv_i_m = float(surf_i.iv(float(tau_surf_obs[i, m]), float(fwd_prices[i, m])))
                    var_curr[i] = (iv_i_m * iv_i_m) * float(tau_surf_obs[i, m])
                else:
                    var_curr[i] = (iv_at_fwd[i, m] ** 2) * tau_obs[m]
            d_var = np.maximum(var_curr - var_prev, 1e-12)
            std_step = np.sqrt(d_var)
            log_x_curr = log_x_prev - 0.5 * d_var[None, None, :] + std_step[None, None, :] * z_m
            s_m = fwd_prices[:, m][None, None, :] * np.exp(log_x_curr)
            perf_paths[:, m, :] = (s_m / ref_spots[None, None, :]).reshape(n_total_paths, d_assets)
            log_x_prev = log_x_curr
            var_prev = var_prev + d_var  # T1M-011: monotone running max cumulative variance
    else:
        rng_bridges = [
            np.random.default_rng(base_seed + rep_idx * 1009 + 777) for rep_idx in range(eff_r)
        ]
        log_x = np.zeros((eff_r, n_rep, d_assets), dtype=np.float64)
        t_prev = 0.0
        has_div_events = len(div_events_ordered) == d_assets and any(
            len(ev) > 0 for ev in div_events_ordered
        )

        for m in range(m_rem):
            t_curr = float(tau_obs[m])
            dt_period = max(t_curr - t_prev, 1e-8)
            dt_sub = dt_period / float(n_sub)
            sqrt_dt_sub = math.sqrt(dt_sub)

            df_fin_prev = (
                np.ones(d_assets, dtype=np.float64) if m == 0 else df_fin_obs_np[:, m - 1]
            )
            df_fin_curr = df_fin_obs_np[:, m]
            r_tot_prev = (
                np.zeros(d_assets, dtype=np.float64) if m == 0 else r_tot_obs_np[:, m - 1]
            )
            r_tot_curr = r_tot_obs_np[:, m]
            r_nat_prev = (
                np.zeros(d_assets, dtype=np.float64) if m == 0 else r_native_obs_np[:, m - 1]
            )
            r_nat_curr = r_native_obs_np[:, m]
            esc_prev = (
                spots_t0
                if (m == 0 or escrowed_spot_np is None or escrowed_spot_np.ndim < 2)
                else escrowed_spot_np[:, m - 1]
            )
            esc_curr = (
                spots_t0
                if (escrowed_spot_np is None or escrowed_spot_np.ndim < 2)
                else escrowed_spot_np[:, m]
            )
            q_div_m = (
                -np.log(np.maximum(esc_curr / np.maximum(esc_prev, 1e-12), 1e-12))
                / dt_period
            )
            r_zero_m = (
                -np.log(np.maximum(df_fin_curr / np.maximum(df_fin_prev, 1e-15), 1e-15))
                / dt_period
            )
            drift_net_m = (r_nat_curr - r_nat_prev) / dt_period - q_div_m
            q_repo_m = r_zero_m - drift_net_m

            z_coarse = z_sobol_all[:, :, m * d_assets : (m + 1) * d_assets] @ chol_l.T
            if n_sub == 1:
                z_steps = z_coarse[None, :, :, :]
            else:
                z_steps = np.empty((n_sub, eff_r, n_rep, d_assets), dtype=np.float64)
                inv_sqrt_sub = 1.0 / math.sqrt(float(n_sub))
                for rep_idx in range(eff_r):
                    raw_sub = rng_bridges[rep_idx].standard_normal((n_sub, n_rep, d_assets)) @ chol_l.T
                    mean_sub = np.mean(raw_sub, axis=0, keepdims=True)
                    z_steps[:, rep_idx, :, :] = (
                        raw_sub - mean_sub + z_coarse[rep_idx][None, :, :] * inv_sqrt_sub
                    )

            f_curr = fwd_prices[:, m]
            f_prev = spots_t0 if m == 0 else fwd_prices[:, m - 1]

            for s_idx in range(n_sub):
                alpha_s = (s_idx + 0.5) / float(n_sub)
                t_step = t_prev + alpha_s * dt_period
                if has_div_events:
                    f_step = _compute_forward_at_time_np(
                        spots_t0, r_tot_obs_np, tau_obs, div_events_ordered, t_step, m
                    )
                else:
                    f_step = f_prev + alpha_s * (f_curr - f_prev)
                s_curr_est = np.exp(log_x) * f_step[None, None, :]
                if has_quanto_diff:
                    r_tot_step = r_tot_prev + alpha_s * (r_tot_curr - r_tot_prev)
                    r_nat_step = r_nat_prev + alpha_s * (r_nat_curr - r_nat_prev)
                    f_step_native = f_step * np.exp(r_nat_step - r_tot_step)
                else:
                    f_step_native = f_step

                for i in range(d_assets):
                    t_surf_prev = 0.0 if m == 0 else float(tau_surf_obs[i, m - 1])
                    t_surf_curr = float(tau_surf_obs[i, m])
                    dt_surf_ratio = (t_surf_curr - t_surf_prev) / dt_period
                    t_surf_step = t_surf_prev + alpha_s * (
                        t_surf_curr - t_surf_prev
                    )
                    lv_dcc = np.asarray(
                        surfs_ordered[i].local_vol(
                            t_surf_step,
                            s_curr_est[:, :, i].reshape(-1),
                            forward_at_t=float(f_step_native[i]),
                            r_zero=float(r_zero_m[i]),
                            q_repo=float(q_repo_m[i]),
                            cash_div_leverage=1.0,  # T1M-001: never double-count dividend leverage
                        ),
                        dtype=np.float64,
                    ).reshape(eff_r, n_rep)
                    lv_i = lv_dcc * math.sqrt(max(dt_surf_ratio, 1e-12))
                    log_x[:, :, i] += (
                        -0.5 * (lv_i * lv_i) * dt_sub
                        + lv_i * sqrt_dt_sub * z_steps[s_idx, :, :, i]
                    )

            # Martingale drift enforcement per replicate
            mean_x = np.mean(np.exp(log_x), axis=1, keepdims=True)
            log_x = log_x - np.log(np.maximum(mean_x, 1e-12))

            s_m = f_curr[None, None, :] * np.exp(log_x)
            perf_paths[:, m, :] = (s_m / ref_spots[None, None, :]).reshape(n_total_paths, d_assets)
            t_prev = t_curr

    # Pathwise payoff & cashflow evaluation
    alive = np.ones(n_total_paths, dtype=np.float64)
    unpaid_mem = np.full(n_total_paths, float(state.seasoned_state.c_mem_0), dtype=np.float64)
    total_pv_paths = np.zeros(n_total_paths, dtype=np.float64)

    p_stop_arr = np.zeros(m_rem, dtype=np.float64)
    q_cpn_arr = np.zeros(m_rem, dtype=np.float64)
    cpn_cf_expected = np.zeros(m_rem, dtype=np.float64)

    eff_cpn_idx = state.coupon_index if state.coupon_index else state.callability_index
    eff_opt_idx = (
        "AverageOf"
        if bool(getattr(state, "is_long_basket", False))
        else (state.option_index if state.option_index else state.callability_index)
    )
    cpn_dir = str(getattr(state, "coupon_barrier_direction", "Up") or "Up").strip().lower()
    cpn_type = str(getattr(state, "coupon_barrier_type", "In") or "In").strip().lower()
    inv_lex_perm_raw = getattr(state, "inv_lex_perm", None)
    inv_lex_perm = np.asarray(inv_lex_perm_raw, dtype=np.int64) if inv_lex_perm_raw is not None else None
    w_fund = _compute_funding_weights_np(state)

    for m in range(m_rem):
        perf_m = perf_paths[:, m, :]
        w_call = _evaluate_index_with_unperm_np(perf_m, state.callability_index, inv_lex_perm)
        w_cpn = _evaluate_index_with_unperm_np(perf_m, eff_cpn_idx, inv_lex_perm)

        if is_callable[m] > 0.5:
            ko_ind = _digital_ramp_np(w_call, float(ko_barriers[m]), float(state.ko_spread))
        else:
            ko_ind = np.zeros(n_total_paths, dtype=np.float64)

        if is_guaranteed_cpn[m] > 0.5:
            cpn_ind = np.ones(n_total_paths, dtype=np.float64)
        else:
            b_cpn_m = float(cpn_barriers[m])
            s_cpn = float(state.cpn_spread)
            if cpn_dir == "down":
                base_cpn_ind = 1.0 - _digital_ramp_np(w_cpn, b_cpn_m, -s_cpn)
            else:
                base_cpn_ind = _digital_ramp_np(w_cpn, b_cpn_m, s_cpn)
            cpn_ind = (1.0 - base_cpn_ind) if cpn_type == "out" else base_cpn_ind

        if state.force_coupons_on_autocall:
            cpn_ind = np.maximum(cpn_ind, ko_ind)

        # Pathwise funding leg cashflow before period m autocall exit (T1M-009)
        if w_fund[m] != 0.0:
            total_pv_paths -= alive * w_fund[m]

        stop_m = alive * ko_ind
        p_stop_arr[m] = float(np.mean(stop_m))

        surv_before_m = float(np.mean(alive))
        if surv_before_m > 1e-12:
            q_cpn_arr[m] = float(np.mean(alive * cpn_ind) / surv_before_m)
        else:
            q_cpn_arr[m] = 0.0

        eff_c_paths = (
            (float(coupon_amounts[m]) + unpaid_mem) if state.is_memory else float(coupon_amounts[m])
        )
        cpn_undisc_m = alive * cpn_ind * eff_c_paths + stop_m * float(state.autocall_coupon)
        cpn_cf_expected[m] = float(np.mean(cpn_undisc_m))
        total_pv_paths += df_pay[m] * cpn_undisc_m

        if state.is_memory:
            unpaid_mem = (unpaid_mem + float(coupon_amounts[m])) * (1.0 - cpn_ind)

        alive = alive * (1.0 - ko_ind)
        if not state.no_notional_repayment:
            if m < m_rem - 1:
                total_pv_paths += df_pay[m] * stop_m
            else:
                total_pv_paths += df_pay[m] * (stop_m + alive)

    # Terminal option evaluation at m = M_rem - 1
    perf_term = perf_paths[:, -1, :]
    w_opt = _evaluate_index_with_unperm_np(perf_term, eff_opt_idx, inv_lex_perm)
    w_best: Optional[npt.NDArray[np.float64]] = None
    star_ind: Optional[npt.NDArray[np.float64]] = None
    if state.one_star_strike is not None and float(state.one_star_strike) > 0.0:
        w_best = _evaluate_index_with_unperm_np(perf_term, "BestOf", inv_lex_perm)
        star_ind = (w_best >= float(state.one_star_strike)).astype(np.float64)

    if state.strike > 1e-6 and state.option_gearing > 0.0:
        put_intrinsic = np.maximum(float(state.strike) - w_opt, 0.0)
        if state.lower_strike is not None and float(state.lower_strike) > 0.0:
            put_intrinsic = np.minimum(
                put_intrinsic, max(float(state.strike) - float(state.lower_strike), 0.0)
            )

        if (
            state.ki_barrier is not None
            and float(state.ki_barrier) > 0.0
            and state.seasoned_state.i_ki_past < 0.5
        ):
            ki_up = _digital_ramp_np(w_opt, float(state.ki_barrier), float(state.ki_spread))
            ki_ind = 1.0 - ki_up
            put_intrinsic = put_intrinsic * ki_ind

        if star_ind is not None:
            put_intrinsic = put_intrinsic * (1.0 - star_ind)

        unit_put_paths = alive * put_intrinsic
    else:
        unit_put_paths = np.zeros(n_total_paths, dtype=np.float64)

    if state.participation_call_strike is not None and float(state.participation_call_rate) > 0.0:
        call_intrinsic = np.maximum(w_opt - float(state.participation_call_strike), 0.0)
        unit_call_paths = alive * call_intrinsic
    else:
        unit_call_paths = np.zeros(n_total_paths, dtype=np.float64)

    l_put_val = float(np.mean(unit_put_paths))
    l_call_val = float(np.mean(unit_call_paths))
    l_onestar_val = 0.0
    p_onestar_val = float(np.mean(alive * star_ind)) if star_ind is not None else 0.0

    put_sign = -1.0
    opt_payoff_paths = df_pay[-1] * (
        put_sign * float(state.option_gearing) * unit_put_paths
        + float(state.participation_call_rate) * unit_call_paths
    )
    if star_ind is not None and float(state.one_star_coupon) > 0.0:
        onestar_paths = alive * star_ind * float(state.one_star_coupon)
        opt_payoff_paths = opt_payoff_paths + df_pay[-1] * onestar_paths
        l_onestar_val = float(np.mean(onestar_paths))

    total_pv_paths += opt_payoff_paths

    if eff_r > 1:
        rep_pv = total_pv_paths.reshape(eff_r, n_rep).mean(axis=1)
        rep_opt = opt_payoff_paths.reshape(eff_r, n_rep).mean(axis=1)
        total_se = float(np.std(rep_pv, ddof=1) / math.sqrt(float(eff_r)))
        option_se = float(np.std(rep_opt, ddof=1) / math.sqrt(float(eff_r)))
    else:
        total_se = float(np.std(total_pv_paths, ddof=1) / math.sqrt(float(n_total_paths)))
        option_se = float(np.std(opt_payoff_paths, ddof=1) / math.sqrt(float(n_total_paths)))

    asm = _assemble_qmc_postproc(
        state,
        p_stop_arr=p_stop_arr,
        q_cpn_arr=q_cpn_arr,
        cpn_cf_expected=cpn_cf_expected,
        df_pay=df_pay,
        l_put_val=l_put_val,
        l_call_val=l_call_val,
        l_onestar_val=l_onestar_val,
        p_onestar_val=p_onestar_val,
        option_se=option_se,
        total_se=total_se,
        assemble_fn=assemble_autocall_cashflows_and_pv,
    )

    return DupireQMCResult(
        product_value=float(asm.product_value.item()),
        standard_error=total_se,
        option_pv=float(asm.option_pv.item()),
        option_se=option_se,
        notional_pv=float(asm.notional_pv.item()),
        coupon_pv=float(asm.coupon_pv.item()),
        funding_pv=float(asm.funding_pv.item()),
        p_stop=p_stop_arr,
        q_cpn=q_cpn_arr,
        l_put=l_put_val,
        l_call=l_call_val,
        cashflows=asm.cashflows,
        n_paths=n_total_paths,
        mode=eff_mode,
        l_onestar=l_onestar_val,
        p_onestar=p_onestar_val,
    )


price_eq_autocall_qmc_fallback = price_eq_autocall_dupire_qmc
price_autocall_dupire_qmc = price_eq_autocall_dupire_qmc


__all__ = [
    "RegularizedVolSurface",
    "DupireQMCResult",
    "DupireQMCOutput",
    "enforce_no_arbitrage_projection",
    "enforce_no_arbitrage_projection_with_jacobian",
    "price_eq_autocall_dupire_qmc",
    "price_autocall_dupire_qmc",
    "price_eq_autocall_qmc_fallback",
]
