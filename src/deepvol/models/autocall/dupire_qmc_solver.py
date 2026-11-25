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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import inspect
import math
import threading
from typing import Any, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import scipy.interpolate
from scipy.linalg import lapack as scipy_lapack
from scipy.special import ndtr
from scipy.stats import norm, qmc
import torch

from deepvol.market.eq_curves import (
    eval_clamped_cubic_coeffs,
    fast_clamped_cubic_coeffs,
    year_fraction as eq_year_fraction,
    year_fractions as eq_year_fractions,
)
from deepvol.models.autocall.feature_layout import (
    PAIR_ORDER_DEPENDENT_INDICES,
    CanonicalTier3Config,
    MonitoringIndexCode,
    monitoring_index_to_code,
)
from deepvol.models.autocall.seasoned_collapser import (
    CashFlowItem,
    _to_utc_datetime,
)

_SOBOL_MAX_DIM: int = 21201
_ASYMMETRIC_PAIR_INDICES = PAIR_ORDER_DEPENDENT_INDICES
_DIFF_INDEX_CODES = frozenset(
    {
        int(MonitoringIndexCode.DIFF_AB),
        int(MonitoringIndexCode.DIFF_BA),
        int(MonitoringIndexCode.DIFF_BEST_WORST),
        int(MonitoringIndexCode.DIFF_BEST_AVERAGE),
        int(MonitoringIndexCode.DIFF_AVERAGE_WORST),
    }
)
_PAIR_ORDER_INDEX_CODES = frozenset(
    {
        int(MonitoringIndexCode.RATIO_AB),
        int(MonitoringIndexCode.RATIO_BA),
        int(MonitoringIndexCode.DIFF_AB),
        int(MonitoringIndexCode.DIFF_BA),
    }
)
_INDEX_NORM_ALIAS_MAP: dict[str, int] = {
    "worst": int(MonitoringIndexCode.WORST_OF),
    "worstof": int(MonitoringIndexCode.WORST_OF),
    "lookback": int(MonitoringIndexCode.WORST_OF),
    "min": int(MonitoringIndexCode.WORST_OF),
    "best": int(MonitoringIndexCode.BEST_OF),
    "bestof": int(MonitoringIndexCode.BEST_OF),
    "max": int(MonitoringIndexCode.BEST_OF),
    "average": int(MonitoringIndexCode.AVERAGE_OF),
    "averageof": int(MonitoringIndexCode.AVERAGE_OF),
    "multiperf": int(MonitoringIndexCode.AVERAGE_OF),
    "equallyweightedbasket": int(MonitoringIndexCode.AVERAGE_OF),
    "mean": int(MonitoringIndexCode.AVERAGE_OF),
    "dispersion": int(MonitoringIndexCode.DISPERSION),
    "ratioab": int(MonitoringIndexCode.RATIO_AB),
    "ratio": int(MonitoringIndexCode.RATIO_AB),
    "ratioba": int(MonitoringIndexCode.RATIO_BA),
    "ratiobestworst": int(MonitoringIndexCode.RATIO_BEST_WORST),
    "ratiobestaverage": int(MonitoringIndexCode.RATIO_BEST_AVERAGE),
    "ratioaverageworst": int(MonitoringIndexCode.RATIO_AVERAGE_WORST),
    "diffab": int(MonitoringIndexCode.DIFF_AB),
    "outperformance": int(MonitoringIndexCode.DIFF_AB),
    "spread": int(MonitoringIndexCode.DIFF_AB),
    "diffba": int(MonitoringIndexCode.DIFF_BA),
    "diffbestworst": int(MonitoringIndexCode.DIFF_BEST_WORST),
    "rainbow": int(MonitoringIndexCode.DIFF_BEST_WORST),
    "spreadbestworst": int(MonitoringIndexCode.DIFF_BEST_WORST),
    "diffbestaverage": int(MonitoringIndexCode.DIFF_BEST_AVERAGE),
    "spreadbestaverage": int(MonitoringIndexCode.DIFF_BEST_AVERAGE),
    "diffaverageworst": int(MonitoringIndexCode.DIFF_AVERAGE_WORST),
    "spreadaverageworst": int(MonitoringIndexCode.DIFF_AVERAGE_WORST),
    "twinwin": int(MonitoringIndexCode.TWIN_WIN),
    "secondworstof": int(MonitoringIndexCode.SECOND_WORST_OF),
    "2ndworstof": int(MonitoringIndexCode.SECOND_WORST_OF),
    "thirdworstof": int(MonitoringIndexCode.THIRD_WORST_OF),
    "3rdworstof": int(MonitoringIndexCode.THIRD_WORST_OF),
    "secondbestof": int(MonitoringIndexCode.SECOND_BEST_OF),
    "secondbest": int(MonitoringIndexCode.SECOND_BEST_OF),
    "2ndbestof": int(MonitoringIndexCode.SECOND_BEST_OF),
}


def resolve_monitoring_index_code(index_name: Optional[str]) -> int:
    """Resolves any canonical monitoring index name or alias to code 0..17."""
    if not index_name:
        return int(MonitoringIndexCode.WORST_OF)
    cleaned = str(index_name).strip()
    if not cleaned:
        return int(MonitoringIndexCode.WORST_OF)
    norm = cleaned.lower().replace("_", "").replace("-", "").replace(" ", "")
    mapped = _INDEX_NORM_ALIAS_MAP.get(norm)
    if mapped is not None:
        return mapped
    return monitoring_index_to_code(cleaned)

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
    # Normalize strikes by K_ref so slope/convexity checks are invariant to asset price units.
    k_scale = max(float(k_ref), 1e-12)
    x_norm = strikes / k_scale
    intrinsic = np.maximum(1.0 - x_norm, 0.0)
    calls_floored = np.maximum(calls, intrinsic)
    pts_idx = np.arange(n, dtype=np.int64)
    hull: List[int] = []
    for idx in pts_idx:
        x_i = float(x_norm[idx])
        c_i = float(calls_floored[idx])
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            x1, x2 = float(x_norm[i1]), float(x_norm[i2])
            c1, c2 = float(calls_floored[i1]), float(calls_floored[i2])
            slope_12 = (c2 - c1) / max(x2 - x1, 1e-12)
            slope_23 = (c_i - c2) / max(x_i - x2, 1e-12)
            if slope_23 < slope_12 - 1e-12:
                hull.pop()
            else:
                break
        hull.append(int(idx))

    k_hull = strikes[hull]
    c_hull = calls_floored[hull]
    c_proj = np.interp(strikes, k_hull, c_hull)
    c_proj = np.maximum(c_proj, intrinsic)
    # Preserve core strikes if core was already convex and consistent
    out = np.where(core_mask, calls, c_proj)
    # Ensure global monotonicity non-increasing after merging
    for j in range(1, n):
        if out[j] > out[j - 1]:
            out[j] = out[j - 1]
    return out


def repair_arbitrage_quotes(
    strikes: npt.NDArray[np.float64],
    quotes: npt.NDArray[np.float64],
    t_exp: npt.NDArray[np.float64],
    k_ref: float,
    *,
    tol_slope: float = 0.0,
    tol_d2c: float = 1e-6,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Repairs calendar/smile call-convexity arbitrage quotes via secant lower-hull projection.

    For each expiry slice, computes normalized Black-76 call prices at ``k_ref``, finds the
    earliest strike index ``j`` where ``slope[j] > tol_slope`` or ``(j >= 1 and d2c[j - 1] < -tol_d2c)``,
    holds lower strikes ``0 .. j_anom - 1`` fixed, and projects ``[j_anom - 1 .. n_u - 1]``
    onto the monotone non-increasing convex hull followed by Black-76 IV inversion.
    """
    strikes_arr = np.asarray(strikes, dtype=np.float64)
    quotes_arr = np.asarray(quotes, dtype=np.float64)
    t_arr = np.asarray(t_exp, dtype=np.float64)
    if strikes_arr.ndim != 1 or t_arr.ndim != 1 or quotes_arr.ndim != 2:
        raise ValueError("repair_arbitrage_quotes requires 1D strikes/t_exp and 2D quotes.")
    n_exp, n_u = quotes_arr.shape
    if n_exp != len(t_arr) or n_u != len(strikes_arr):
        raise ValueError(
            f"repair_arbitrage_quotes shape mismatch: quotes={quotes_arr.shape}, "
            f"t_exp={t_arr.shape}, strikes={strikes_arr.shape}."
        )

    repaired = quotes_arr.copy()
    k_scale = max(float(k_ref), 1e-12)
    repaired_slices: list[int] = []
    j_anom_by_slice: list[Optional[int]] = []
    n_modified = 0

    if n_exp == 0 or n_u < 2:
        return repaired, {
            "repaired": False,
            "repaired_slices": repaired_slices,
            "j_anom_by_slice": j_anom_by_slice,
            "max_quote_delta": 0.0,
            "n_quotes_modified": 0,
            "k_ref": float(k_scale),
        }

    y_u = np.log(np.maximum(strikes_arr, 1e-12) / k_scale)
    dk_step = np.maximum((strikes_arr[1:] - strikes_arr[:-1]) / k_scale, 1e-12)

    for m in range(n_exp):
        t_m = max(float(t_arr[m]), 1e-12)
        w_m = (repaired[m] * repaired[m]) * t_m
        calls_m = _b76_norm_call(y_u, w_m)
        slope = (calls_m[1:] - calls_m[:-1]) / dk_step
        d2c = slope[1:] - slope[:-1]

        j_anom: Optional[int] = None
        for j in range(n_u - 1):
            if (float(slope[j]) > tol_slope) or (j >= 1 and float(d2c[j - 1]) < -tol_d2c):
                j_anom = j
                break
        j_anom_by_slice.append(j_anom)
        if j_anom is None:
            continue

        core_mask = np.zeros(n_u, dtype=np.bool_)
        if j_anom > 0:
            core_mask[:j_anom] = True
        calls_hull = _project_wing_calls_convex(strikes_arr, calls_m, k_scale, core_mask)
        q_hull = repaired[m].copy()
        slice_changed = False
        for u in range(max(j_anom - 1, 0), n_u):
            if abs(float(calls_hull[u]) - float(calls_m[u])) > 1e-12:
                q_hull[u] = _invert_b76_norm_call_to_vol(
                    float(y_u[u]), float(calls_hull[u]), t_m, 0.01, 5.00
                )
                slice_changed = True

        q_final = q_hull
        if j_anom >= 2 and slice_changed and n_u >= 3:
            # Piecewise-linear convex-hull chords have zero density (C'' = 0), causing negative/near-zero
            # Durrleman denominators and local-vol cliffs; prefer a C1-smooth damped-slope wing continuation
            # when it stays arbitrage-free with strictly positive Durrleman butterfly density.
            _, g_hull = _eval_durrleman_slice(y_u, (q_hull * q_hull) * t_m)
            min_g_hull = float(np.min(g_hull))
            is_slope_jump = float(slope[j_anom]) > tol_slope and float(d2c[j_anom - 1]) >= -tol_d2c
            u_a = j_anom if is_slope_jump else (j_anom - 1)
            if 1 <= u_a < n_u - 1:
                dy_prev = max(float(y_u[u_a] - y_u[u_a - 1]), 1e-6)
                s_a = float(np.clip((repaired[m, u_a] - repaired[m, u_a - 1]) / dy_prev, -0.5, 0.5))
                l_decay = 0.30
                q_c1 = repaired[m].copy()
                for u in range(u_a + 1, n_u):
                    dy_u = max(float(y_u[u] - y_u[u_a]), 0.0)
                    q_c1[u] = float(
                        np.clip(
                            repaired[m, u_a] + s_a * l_decay * (1.0 - math.exp(-dy_u / l_decay)),
                            0.01,
                            5.00,
                        )
                    )
                w_c1 = (q_c1 * q_c1) * t_m
                calls_c1 = _b76_norm_call(y_u, w_c1)
                sl_c1 = (calls_c1[1:] - calls_c1[:-1]) / dk_step
                d2_c1 = sl_c1[1:] - sl_c1[:-1]
                _, g_c1 = _eval_durrleman_slice(y_u, w_c1)
                if (
                    float(np.max(sl_c1)) <= tol_slope
                    and float(np.min(d2_c1)) >= -tol_d2c
                    and float(np.min(g_c1)) >= 0.05
                    and min_g_hull < 0.05
                ):
                    q_final = q_c1

        slice_changed = False
        for u in range(max(j_anom - 1, 0), n_u):
            if abs(float(q_final[u]) - float(repaired[m, u])) > 1e-12:
                repaired[m, u] = q_final[u]
                n_modified += 1
                slice_changed = True
        if slice_changed:
            repaired_slices.append(m)

    if repaired_slices and n_exp >= 2:
        w_rep = (repaired * repaired) * t_arr[:, None]
        for m in range(1, n_exp):
            dt_m = max(float(t_arr[m] - t_arr[m - 1]), 1e-12)
            min_w_m = w_rep[m - 1] + 1e-5 * dt_m
            viol = w_rep[m] < min_w_m
            if np.any(viol):
                w_rep[m, viol] = min_w_m[viol]
                for u in np.flatnonzero(viol):
                    repaired[m, u] = math.sqrt(max(float(w_rep[m, u]) / max(float(t_arr[m]), 1e-12), 1e-4))
                    n_modified += 1
                if m not in repaired_slices:
                    repaired_slices.append(m)
        repaired_slices.sort()

    max_delta = float(np.max(np.abs(repaired - quotes_arr))) if repaired.size > 0 else 0.0
    return repaired, {
        "repaired": bool(repaired_slices),
        "repaired_slices": repaired_slices,
        "j_anom_by_slice": j_anom_by_slice,
        "max_quote_delta": max_delta,
        "n_quotes_modified": int(n_modified),
        "k_ref": float(k_scale),
    }


def _surface_year_fractions_batch(
    as_of: Union[datetime, date, str],
    dates: Sequence[Union[datetime, date, str]],
    dcc: str = "Act365",
) -> npt.NDArray[np.float64]:
    """Vectorized year fractions across surface expirations with Murex alias fallback."""
    if len(dates) == 0:
        return np.zeros(0, dtype=np.float64)
    dt_cls = datetime
    utc_tz = timezone.utc
    dt1 = as_of if (type(as_of) is dt_cls and as_of.tzinfo is utc_tz) else _to_utc_datetime(as_of)
    exp_dts = [
        d if (type(d) is dt_cls and d.tzinfo is not None) else _to_utc_datetime(d)
        for d in dates
    ]
    dcc_raw = str(dcc or "Act365").strip()
    try:
        return np.asarray(eq_year_fractions(dt1, exp_dts, dcc_raw), dtype=np.float64)
    except ValueError:
        return np.asarray(
            [_surface_year_fraction(dt1, d, dcc_raw) for d in exp_dts],
            dtype=np.float64,
        )


def _fast_clamped_cubic_coeffs_lapack(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Computes clamped cubic spline polynomial coefficients `(4, K - 1, ...)` via LAPACK dgtsv."""
    k = len(x)
    if k < 2:
        raise ValueError("fast_clamped_cubic_coeffs requires at least 2 knots.")
    h = x[1:] - x[:-1]
    h_b = h.reshape((k - 1,) + (1,) * (y.ndim - 1))
    s = (y[1:] - y[:-1]) / h_b
    d = np.zeros_like(y, dtype=np.float64)
    n_int = k - 2
    if n_int > 0:
        hp = h[:-1]
        hn = h[1:]
        diag = 2.0 * (hp + hn)
        int_shape = (n_int,) + (1,) * (y.ndim - 1)
        rhs = 3.0 * (hn.reshape(int_shape) * s[:-1] + hp.reshape(int_shape) * s[1:])
        if n_int == 1:
            d[1:-1] = rhs[0] / diag[0]
        else:
            _, _, _, x_sol, info = scipy_lapack.dgtsv(
                hn[1:].copy(),
                diag,
                hp[:-1].copy(),
                rhs,
                overwrite_dl=1,
                overwrite_d=1,
                overwrite_du=1,
                overwrite_b=1,
            )
            if info != 0:
                return fast_clamped_cubic_coeffs(x, y)
            d[1:-1] = x_sol
    c3 = y[:-1]
    c2 = d[:-1]
    c1 = (3.0 * s - 2.0 * d[:-1] - d[1:]) / h_b
    c0 = (d[:-1] + d[1:] - 2.0 * s) / (h_b * h_b)
    return np.stack([c0, c1, c2, c3], axis=0)


def _eval_durrleman_batch(
    y: npt.NDArray[np.float64],
    w_mat: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Vectorized Durrleman butterfly density g(y) across all slices using LAPACK tridiagonal solve."""
    coeffs = _fast_clamped_cubic_coeffs_lapack(y, w_mat.T)
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
    t_min = float(t_arr.min())
    t_max = float(t_arr.max())
    k_min = float(k_arr.min())
    k_max = float(k_arr.max())
    q_min = float(q_arr.min())
    q_max = float(q_arr.max())
    if not (
        math.isfinite(t_min)
        and math.isfinite(t_max)
        and math.isfinite(k_min)
        and math.isfinite(k_max)
        and math.isfinite(q_min)
        and math.isfinite(q_max)
    ):
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
    if t_min <= 0.0:
        m_b = int(np.flatnonzero(t_arr <= 0.0)[0])
        raise ValueError(
            f"SMILE_ARBITRAGE: Expiration tenor must be strictly positive on surface={surface_id}, "
            f"slice={m_b} (T={float(t_arr[m_b]):.4f}Y)."
        )
    if len(t_arr) >= 2:
        t_diff = t_arr[1:] - t_arr[:-1]
        if float(t_diff.min()) <= 0.0:
            m_b = int(np.flatnonzero(t_diff <= 0.0)[0]) + 1
            raise ValueError(
                f"SMILE_ARBITRAGE: Expiration tenors must be strictly increasing on surface={surface_id}, "
                f"slice={m_b} (prev_T={float(t_arr[m_b - 1]):.4f}Y -> T={float(t_arr[m_b]):.4f}Y)."
            )
    else:
        t_diff = np.empty(0, dtype=np.float64)
    if k_min <= 0.0:
        j_b = int(np.flatnonzero(k_arr <= 0.0)[0])
        raise ValueError(
            f"SMILE_ARBITRAGE: Strikes must be strictly positive on surface={surface_id}, "
            f"K={float(k_arr[j_b]):.4f}."
        )
    if q_min < 1e-4:
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
    if q_max > max_vol:
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
        np.all((k_arr[1:] - k_arr[:-1]) > 1e-9 * np.maximum(1.0, np.abs(k_arr[:-1])))
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
            raise ValueError(
                f"SMILE_ARBITRAGE: Fewer than 2 unique strikes after deduplication on surface={surface_id}."
            )

        dedup_matrix = np.zeros((n_unique, len(k_arr)), dtype=np.float64)
        for u_idx, g in enumerate(groups):
            w_grp = 1.0 / len(g)
            for raw_idx in g:
                dedup_matrix[u_idx, raw_idx] = w_grp

        unique_strikes = np.asarray(unique_k_list, dtype=np.float64)
        q_dedup = q_arr @ dedup_matrix.T

    n_exp = len(t_arr)
    n_u = len(unique_strikes)
    cal_ancestor: Optional[npt.NDArray[np.int64]] = None
    cal_scale: Optional[npt.NDArray[np.float64]] = None

    def _ensure_cal_state() -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        nonlocal cal_ancestor, cal_scale
        if cal_ancestor is None or cal_scale is None:
            cal_ancestor = np.broadcast_to(
                np.arange(n_exp, dtype=np.int64)[:, None], (n_exp, n_u)
            ).copy()
            cal_scale = np.ones((n_exp, n_u), dtype=np.float64)
        return cal_ancestor, cal_scale

    # 2. Floor implied volatility at min_vol
    if float(q_dedup.min()) < min_vol:
        ca, cs = _ensure_cal_state()
        floor_mask = q_dedup < min_vol
        q_dedup = np.maximum(q_dedup, min_vol)
        ca[floor_mask] = -1
        cs[floor_mask] = 0.0
        was_projected = True

    # 3. Enforce calendar total-variance monotonicity: w_m(K) >= w_{m-1}(K) + min_calendar_dvar_dt * dT
    w_mat = (q_dedup * q_dedup) * t_arr[:, None]
    if n_exp >= 2 and bool(np.any((w_mat[1:] - w_mat[:-1]) < (min_calendar_dvar_dt * t_diff)[:, None])):
        ca, cs = _ensure_cal_state()
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
                    if w_mat[m, u] / t_arr[m] <= min_vol * min_vol or ca[m - 1, u] < 0:
                        ca[m, u] = -1
                        cs[m, u] = 0.0
                    else:
                        ca[m, u] = ca[m - 1, u]
                        cs[m, u] = cs[m - 1, u] * (
                            (t_arr[m - 1] * sig_curr[m - 1, u]) / max(t_arr[m] * sig_new, 1e-12)
                        )
                    sig_curr[m, u] = sig_new
        projected_quotes = np.sqrt(np.maximum(w_mat / t_arr[:, None], min_vol * min_vol))
        w_all_init = (projected_quotes * projected_quotes) * t_arr[:, None]
    else:
        projected_quotes = q_dedup
        w_all_init = w_mat

    # 4. Spot-independent Durrleman butterfly convexity screening & localized wing projection
    if n_u >= 3:
        k_ref = float(np.median(unique_strikes))
        y_u = np.log(unique_strikes / k_ref)
        core_strike_mask = (unique_strikes >= 0.5 * k_ref) & (unique_strikes <= 1.5 * k_ref)

        y_eval, g_mat_all = _eval_durrleman_batch(y_u, w_all_init)
        calls_mat_all = _b76_norm_call(y_u[None, :], w_all_init)
        core_eval_mask = (y_eval >= math.log(0.5)) & (y_eval <= math.log(1.5))
        has_core_eval = bool(np.any(core_eval_mask))
        k_scale = max(float(k_ref), 1e-12)
        dk_step = np.maximum((unique_strikes[1:] - unique_strikes[:-1]) / k_scale, 1e-12)

        def _compute_surface_worst_metrics() -> Tuple[float, float]:
            w_chk_all = (projected_quotes * projected_quotes) * t_arr[:, None]
            _, g_chk_all = _eval_durrleman_batch(y_u, w_chk_all)
            if has_core_eval:
                surf_min_g = float(np.min(g_chk_all[:, core_eval_mask]))
            else:
                surf_min_g = float(np.min(g_chk_all))
            c_chk_all = _b76_norm_call(y_u[None, :], w_chk_all)
            sl_chk_all = (c_chk_all[:, 1:] - c_chk_all[:, :-1]) / dk_step[None, :]
            d2_chk_all = sl_chk_all[:, 1:] - sl_chk_all[:, :-1]
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

        slopes_all = (calls_mat_all[:, 1:] - calls_mat_all[:, :-1]) / dk_step[None, :]
        d2c_all = slopes_all[:, 1:] - slopes_all[:, :-1]
        core_d2_mask = core_strike_mask[1:-1]
        has_core_d2 = bool(np.any(core_d2_mask))
        min_g_core_all = (
            float(np.min(g_mat_all[:, core_eval_mask]))
            if has_core_eval
            else float(np.min(g_mat_all))
        )
        min_g_all_global = float(np.min(g_mat_all))
        d2c_core_mat = d2c_all[:, core_d2_mask] if has_core_d2 else d2c_all
        min_d2c_core_global = float(np.min(d2c_core_mat)) if d2c_core_mat.size > 0 else 0.0
        needs_per_slice_pass = (
            min_d2c_core_global < -1e-4
            or min_g_core_all < -10.0
            or min_g_all_global < -25.0
            or bool(np.min(d2c_all) < -1e-9)
            or bool(np.max(slopes_all) > 1e-9)
        )

        if needs_per_slice_pass:
            ca, cs = _ensure_cal_state()
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
                slopes_m = slopes_all[m]
                d2c_m = d2c_all[m]

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

                if j_first_severe is not None:
                    j_off = j_first_severe
                elif j_first_mild is not None:
                    j_off = j_first_mild
                elif min_d2c_core < 0.0:
                    j_off = j_worst_core
                elif (min_g_core < -10.0 or min_g_all < -25.0) and len(g_eval) > 0:
                    g_idx = int(np.argmin(g_eval))
                    j_off = g_idx if g_idx < n_u else min((g_idx - n_u) + 1, n_u - 1)
                else:
                    j_off = j_worst_core
                j_prev = max(j_off - 1, 0)
                strike_val = float(unique_strikes[j_off])
                prev_strike_val = float(unique_strikes[j_prev])
                v_prev = float(projected_quotes[m, j_prev])
                v_curr = float(projected_quotes[m, j_off])
                min_slope_m = float(np.min(slopes_m)) if len(slopes_m) > 0 else 0.0
                max_slope_m = float(np.max(slopes_m)) if len(slopes_m) > 0 else 0.0

                if min_d2c_core < -1e-3 or min_g_core < -10.0 or min_g_all < -25.0:
                    surf_min_d2c, surf_min_g = _compute_surface_worst_metrics()
                    min_d2c_val = min(min_d2c_core, surf_min_d2c)
                    min_g_val = min(min_g_core, surf_min_g)
                    raise ValueError(
                        f"SMILE_ARBITRAGE: Core call-price convexity violation at slice {m}: "
                        f"surface={surface_id}, slice={m} (T={t_m:.4f}Y), "
                        f"K={strike_val:.4f} (prev_K={prev_strike_val:.4f} -> K={strike_val:.4f}, "
                        f"vol_jump={v_prev:.5f}->{v_curr:.5f}), "
                        f"min_d2c={min_d2c_val:.6f}, min_g={min_g_val:.4f}, "
                        f"slope_range=[{min_slope_m:.4f}, {max_slope_m:.4f}]."
                    )

                if min_d2c_core < -1e-4:
                    # Mild numerical quote noise (-1e-3 <= min_d2c_core < -1e-4
                    # and min_g_core >= -10.0) is projected onto convex call prices with warning.
                    all_wing_mask = np.zeros_like(core_strike_mask)
                    calls_proj = _project_wing_calls_convex(unique_strikes, calls_m, k_ref, all_wing_mask)
                    for u in range(n_u):
                        if abs(calls_proj[u] - calls_m[u]) > 1e-11:
                            sig_u = _invert_b76_norm_call_to_vol(
                                float(y_u[u]), float(calls_proj[u]), t_m, min_vol, max_vol
                            )
                            projected_quotes[m, u] = sig_u
                            ca[m, u] = -1
                            cs[m, u] = 0.0
                    # If a downward quote dip left any residual d2c < -1e-4 after lower-hull projection,
                    # smooth offending interior vol wiggle iteratively until d2c >= -1e-4.
                    for _ in range(16):
                        w_chk = (projected_quotes[m] * projected_quotes[m]) * t_m
                        c_chk = _b76_norm_call(y_u, w_chk)
                        sl_chk = (c_chk[1:] - c_chk[:-1]) / dk_step
                        d2_chk = sl_chk[1:] - sl_chk[:-1]
                        fixed_any = False
                        for j_c in range(1, n_u - 1):
                            if core_strike_mask[j_c] and float(d2_chk[j_c - 1]) < -1e-4:
                                w_interp = (unique_strikes[j_c] - unique_strikes[j_c - 1]) / max(
                                    unique_strikes[j_c + 1] - unique_strikes[j_c - 1], 1e-12
                                )
                                v_sec = (1.0 - w_interp) * projected_quotes[m, j_c - 1] + w_interp * projected_quotes[m, j_c + 1]
                                projected_quotes[m, j_c] = 0.5 * (projected_quotes[m, j_c] + v_sec)
                                ca[m, j_c] = -1
                                cs[m, j_c] = 0.0
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
                                ca[m, u] = -1
                                cs[m, u] = 0.0
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
                ca_re, cs_re = _ensure_cal_state()
                w_re[m, viol] = min_w_m[viol]
                was_projected = True
                for u in np.flatnonzero(viol):
                    sig_prev_u = float(projected_quotes[m - 1, u])
                    sig_new = math.sqrt(max(w_re[m, u] / t_arr[m], min_vol * min_vol))
                    projected_quotes[m, u] = sig_new
                    if w_re[m, u] / t_arr[m] <= min_vol * min_vol or ca_re[m - 1, u] < 0:
                        ca_re[m, u] = -1
                        cs_re[m, u] = 0.0
                    else:
                        ca_re[m, u] = ca_re[m - 1, u]
                        cs_re[m, u] = cs_re[m - 1, u] * (
                            (t_arr[m - 1] * sig_prev_u) / max(t_arr[m] * sig_new, 1e-12)
                        )

        if was_projected:
            w_post = (projected_quotes * projected_quotes) * t_arr[:, None]
            _, g_post_all = _eval_durrleman_batch(y_u, w_post)
            c_post_all = _b76_norm_call(y_u[None, :], w_post)
            for m in range(n_exp):
                t_m = float(t_arr[m])
                g_post = g_post_all[m]
                min_g_post_core = float(np.min(g_post[core_eval_mask])) if has_core_eval else float(np.min(g_post))
                min_g_post_all = float(np.min(g_post))
                sl_post = (c_post_all[m, 1:] - c_post_all[m, :-1]) / dk_step
                d2_post = sl_post[1:] - sl_post[:-1]
                core_d2_post = d2_post[core_strike_mask[1:-1]] if np.any(core_strike_mask[1:-1]) else d2_post
                min_d2_post_core = float(np.min(core_d2_post)) if len(core_d2_post) > 0 else 0.0
                if min_d2_post_core < -1e-3 or min_g_post_core < -10.0 or min_g_post_all < -25.0:
                    surf_min_d2c, surf_min_g = _compute_surface_worst_metrics()
                    j_post = int(np.argmin(d2_post)) + 1 if len(d2_post) > 0 else 1
                    k_post = float(unique_strikes[min(j_post, n_u - 1)])
                    min_sl_p = float(np.min(sl_post)) if len(sl_post) > 0 else 0.0
                    max_sl_p = float(np.max(sl_post)) if len(sl_post) > 0 else 0.0
                    raise ValueError(
                        f"SMILE_ARBITRAGE: Post-projection slice {m} fails convexity: "
                        f"surface={surface_id}, slice={m} (T={t_m:.4f}Y), K={k_post:.4f}, "
                        f"min_d2c={min(min_d2_post_core, surf_min_d2c):.6f}, "
                        f"min_g={min(min_g_post_core, surf_min_g):.4f}, "
                        f"slope_range=[{min_sl_p:.4f}, {max_sl_p:.4f}]."
                    )
                if min_d2_post_core < -1e-4:
                    ca_post, cs_post = _ensure_cal_state()
                    for j_c in range(1, n_u - 1):
                        if core_strike_mask[j_c] and float(d2_post[j_c - 1]) < -1e-4:
                            w_interp = (unique_strikes[j_c] - unique_strikes[j_c - 1]) / max(
                                unique_strikes[j_c + 1] - unique_strikes[j_c - 1], 1e-12
                            )
                            v_sec = (1.0 - w_interp) * projected_quotes[m, j_c - 1] + w_interp * projected_quotes[m, j_c + 1]
                            projected_quotes[m, j_c] = 0.5 * (projected_quotes[m, j_c] + v_sec)
                            ca_post[m, j_c] = -1
                            cs_post[m, j_c] = 0.0
                    if warnings_out is not None and "SMILE_ARBITRAGE_PROJECTED" not in warnings_out:
                        warnings_out.append("SMILE_ARBITRAGE_PROJECTED")
                    c_post_chk = _b76_norm_call(y_u, (projected_quotes[m] * projected_quotes[m]) * t_m)
                    sl_post_chk = (c_post_chk[1:] - c_post_chk[:-1]) / dk_step
                    d2_post_chk = sl_post_chk[1:] - sl_post_chk[:-1]
                    core_chk = d2_post_chk[core_strike_mask[1:-1]] if np.any(core_strike_mask[1:-1]) else d2_post_chk
                    if len(core_chk) > 0 and float(np.min(core_chk)) < -1e-3:
                        surf_min_d2c, surf_min_g = _compute_surface_worst_metrics()
                        j_chk = int(np.argmin(d2_post_chk)) + 1 if len(d2_post_chk) > 0 else 1
                        k_chk = float(unique_strikes[min(j_chk, n_u - 1)])
                        raise ValueError(
                            f"SMILE_ARBITRAGE: Post-projection slice {m} remains non-convex: "
                            f"surface={surface_id}, slice={m} (T={t_m:.4f}Y), K={k_chk:.4f}, "
                            f"min_d2c={min(float(np.min(core_chk)), surf_min_d2c):.6f}, min_g={surf_min_g:.4f}, "
                            f"slope_range=[{float(np.min(sl_post_chk)):.4f}, {float(np.max(sl_post_chk)):.4f}]."
                        )

    ca_out, cs_out = _ensure_cal_state()

    cal_jacobian = np.zeros((n_exp, n_exp, n_u), dtype=np.float64)
    if not was_projected:
        m_idx = np.arange(n_exp)
        cal_jacobian[m_idx, m_idx, :] = 1.0
    else:
        m_grid, u_grid = np.nonzero(ca_out >= 0)
        cal_jacobian[m_grid, ca_out[m_grid, u_grid], u_grid] = cs_out[m_grid, u_grid]

    return (
        unique_strikes,
        projected_quotes,
        dedup_matrix,
        was_projected,
        ca_out,
        cs_out,
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
            raw_tenors_in = surf_spec.get("Tenors", surf_spec.get("tenors"))
            if raw_tenors_in is not None:
                raw_tenors = np.asarray(raw_tenors_in, dtype=np.float64)
                raw_tenors_365 = raw_tenors.copy()
            else:
                if raw_as_of is None:
                    raise ValueError(
                        "as_of_date is required when ExpirationDates are provided instead of tenors."
                    )
                if exp_dates is None:
                    exp_dates = []
                raw_tenors = _surface_year_fractions_batch(raw_as_of, exp_dates, eff_dcc)
                raw_tenors_365 = (
                    raw_tenors.copy()
                    if eff_dcc.strip().upper().replace("/", "").replace("_", "") in ("ACT365", "ACT365F", "ACT365FIXED", "ACTUAL365")
                    else _surface_year_fractions_batch(raw_as_of, exp_dates, "Act365")
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
            raw_tenors_in = getattr(surf_spec, "Tenors", getattr(surf_spec, "tenors", None))
            if raw_tenors_in is not None:
                raw_tenors = np.asarray(raw_tenors_in, dtype=np.float64)
                raw_tenors_365 = raw_tenors.copy()
            else:
                if raw_as_of is None:
                    raise ValueError(
                        "as_of_date is required when ExpirationDates are provided instead of tenors."
                    )
                if exp_dates is None:
                    exp_dates = []
                raw_tenors = _surface_year_fractions_batch(raw_as_of, exp_dates, eff_dcc)
                raw_tenors_365 = (
                    raw_tenors.copy()
                    if eff_dcc.strip().upper().replace("/", "").replace("_", "") in ("ACT365", "ACT365F", "ACT365FIXED", "ACTUAL365")
                    else _surface_year_fractions_batch(raw_as_of, exp_dates, "Act365")
                )

        if surface_id is not None and str(surface_id).strip() and (
            not self.identifier or self.identifier in ("ASSET", "UNKNOWN")
        ):
            self.identifier = str(surface_id).strip()

        self.dcc: str = eff_dcc
        self.as_of_date: Optional[datetime] = (
            raw_as_of
            if (type(raw_as_of) is datetime and raw_as_of.tzinfo is timezone.utc)
            else (_to_utc_datetime(raw_as_of) if raw_as_of is not None else None)
        )
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
        self._fast_basis_coeffs_per_slice: Optional[npt.NDArray[np.float64]] = None
        self._basis_slope_left_per_slice: Optional[npt.NDArray[np.float64]] = None
        self._basis_slope_right_per_slice: Optional[npt.NDArray[np.float64]] = None

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
            eye_u = np.eye(self.n_unique_strikes, dtype=np.float64)
            both_c = _fast_clamped_cubic_coeffs_lapack(
                self.strikes, np.hstack([self.quotes.T, eye_u])
            )
            self._fast_slice_coeffs = np.ascontiguousarray(both_c[:, :, : self.n_exp])
            self._fast_basis_coeffs = np.ascontiguousarray(both_c[:, :, self.n_exp :])
            if use_linear_extrap:
                h0 = max(float(self.strikes[1] - self.strikes[0]), 1e-12)
                h1 = max(float(self.strikes[-1] - self.strikes[-2]), 1e-12)
                self._slopes_left = (self.quotes[:, 1] - self.quotes[:, 0]) / h0
                self._slopes_right = (self.quotes[:, -1] - self.quotes[:, -2]) / h1
                bsl = np.zeros(self.n_unique_strikes, dtype=np.float64)
                bsr = np.zeros(self.n_unique_strikes, dtype=np.float64)
                bsl[0], bsl[1] = -1.0 / h0, 1.0 / h0
                bsr[-2], bsr[-1] = -1.0 / h1, 1.0 / h1
                self._basis_slope_left = bsl
                self._basis_slope_right = bsr
            else:
                self._slopes_left = np.zeros(self.n_exp, dtype=np.float64)
                self._slopes_right = np.zeros(self.n_exp, dtype=np.float64)
                self._basis_slope_left = np.zeros(self.n_unique_strikes, dtype=np.float64)
                self._basis_slope_right = np.zeros(self.n_unique_strikes, dtype=np.float64)
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
        h_surf = hashlib.blake2b(digest_size=16)
        h_surf.update(self.quotes.tobytes())
        h_surf.update(self.strikes.tobytes())
        h_surf.update(self.tenors.tobytes())
        h_surf.update(self.tenors_act365.tobytes())
        h_surf.update(repr((self.dcc, self.smile_interp, self.smile_extrap, self.bc_type, float(self.min_vol))).encode("utf-8"))
        self.content_digest: str = h_surf.hexdigest()

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
            self._fast_basis_coeffs_per_slice,
            self._basis_slope_left_per_slice,
            self._basis_slope_right_per_slice,
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
            if (
                self._fast_basis_coeffs is not None
                and self._basis_slope_left is not None
                and self._basis_slope_right is not None
            ):
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
            elif interp_norm in ("pchip", "monotonecubic"):
                # PCHIP derivatives are nonlinear in quote levels, so compute central FD basis
                # weights per maturity slice for exact Jacobian propagation alongside the mean basis.
                eps_fd = 1e-6
                q_plus = self.quotes[:, None, :] + eps_fd * eye_u[None, :, :]
                q_minus = self.quotes[:, None, :] - eps_fd * eye_u[None, :, :]
                q_both = np.concatenate([q_plus, q_minus], axis=1).reshape(
                    self.n_exp * 2 * self.n_unique_strikes, self.n_unique_strikes
                )
                pchip_fd = scipy.interpolate.PchipInterpolator(self.strikes, q_both, axis=1)
                c_both = np.asarray(pchip_fd.c, dtype=np.float64).reshape(
                    4, max(self.n_unique_strikes - 1, 1), self.n_exp, 2, self.n_unique_strikes
                )
                basis_per_slice = np.ascontiguousarray(
                    np.transpose(
                        (c_both[:, :, :, 0, :] - c_both[:, :, :, 1, :]) / (2.0 * eps_fd),
                        (2, 0, 1, 3),
                    ),
                    dtype=np.float64,
                )
                self._fast_basis_coeffs_per_slice = basis_per_slice
                self._fast_basis_coeffs = np.ascontiguousarray(
                    np.mean(basis_per_slice, axis=0), dtype=np.float64
                )
                if use_linear_extrap:
                    sl_both = np.asarray(pchip_fd(self.strikes[0], 1), dtype=np.float64).reshape(
                        self.n_exp, 2, self.n_unique_strikes
                    )
                    sr_both = np.asarray(pchip_fd(self.strikes[-1], 1), dtype=np.float64).reshape(
                        self.n_exp, 2, self.n_unique_strikes
                    )
                    sl_per = np.ascontiguousarray(
                        (sl_both[:, 0, :] - sl_both[:, 1, :]) / (2.0 * eps_fd), dtype=np.float64
                    )
                    sr_per = np.ascontiguousarray(
                        (sr_both[:, 0, :] - sr_both[:, 1, :]) / (2.0 * eps_fd), dtype=np.float64
                    )
                else:
                    sl_per = np.zeros((self.n_exp, self.n_unique_strikes), dtype=np.float64)
                    sr_per = np.zeros((self.n_exp, self.n_unique_strikes), dtype=np.float64)
                self._basis_slope_left_per_slice = sl_per
                self._basis_slope_right_per_slice = sr_per
                self._basis_slope_left = np.ascontiguousarray(np.mean(sl_per, axis=0), dtype=np.float64)
                self._basis_slope_right = np.ascontiguousarray(np.mean(sr_per, axis=0), dtype=np.float64)
            elif self.bc_type == "clamped" and interp_norm in ("cubicspline", "spline", ""):
                self._fast_basis_coeffs = _fast_clamped_cubic_coeffs_lapack(self.strikes, eye_u)
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
                if self._basis_cs is None:
                    eff_bc = self.bc_type
                    if interp_norm == "naturalcubicspline":
                        eff_bc = "natural"
                    elif interp_norm in ("notaknot", "notaknotcubicspline", "polynomial", "lme"):
                        eff_bc = "not-a-knot"
                    self._basis_cs = scipy.interpolate.CubicSpline(
                        self.strikes, eye_u, bc_type=eff_bc, axis=0
                    )
                self._fast_basis_coeffs = np.ascontiguousarray(self._basis_cs.c, dtype=np.float64)
                if use_linear_extrap:
                    self._basis_slope_left = np.ascontiguousarray(
                        self._basis_cs(self.strikes[0], 1), dtype=np.float64
                    )
                    self._basis_slope_right = np.ascontiguousarray(
                        self._basis_cs(self.strikes[-1], 1), dtype=np.float64
                    )
                else:
                    self._basis_slope_left = np.zeros(self.n_unique_strikes, dtype=np.float64)
                    self._basis_slope_right = np.zeros(self.n_unique_strikes, dtype=np.float64)

            for arr in (
                self._fast_basis_coeffs,
                self._basis_slope_left,
                self._basis_slope_right,
                self._fast_basis_coeffs_per_slice,
                self._basis_slope_left_per_slice,
                self._basis_slope_right_per_slice,
            ):
                if arr is not None:
                    arr.setflags(write=False)

    def ensure_basis_spline_coeffs(self) -> npt.NDArray[np.float64]:
        """Thread-safely initializes and returns frozen `(4, n_unique - 1, n_unique)` basis coeffs."""
        self._ensure_basis_spline()
        if self._fast_basis_coeffs is None:
            raise RuntimeError("RegularizedVolSurface basis spline failed to initialize.")
        return self._fast_basis_coeffs

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

    def _eval_slice_raw(
        self,
        slice_idx: int,
        k_arr: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
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
        return val

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
        out = np.maximum(self._eval_slice_raw(slice_idx, k_arr), self.min_vol)
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
        if isinstance(tenors, np.ndarray):
            t_arr = tenors if tenors.dtype == np.float64 else tenors.astype(np.float64)
        elif isinstance(tenors, Sequence) and len(tenors) > 0 and isinstance(tenors[0], (datetime, date)):
            t_arr = np.asarray([self.tenor_from_date(d) for d in tenors], dtype=np.float64)  # type: ignore[arg-type]
        else:
            t_arr = np.asarray(tenors, dtype=np.float64)
        k_arr = strikes if (isinstance(strikes, np.ndarray) and strikes.dtype == np.float64) else np.asarray(strikes, dtype=np.float64)
        t_b, k_b = np.broadcast_arrays(t_arr, k_arr)
        orig_shape = t_b.shape
        t_flat = np.maximum(t_b.reshape(-1), 1e-8)
        k_flat = k_b.reshape(-1)
        n_q = len(t_flat)
        if n_q == 0:
            return np.empty(orig_shape, dtype=np.float64)

        k_min = float(self.strikes[0])
        k_max = float(self.strikes[-1])
        k_lo_val = float(k_flat.min())
        k_hi_val = float(k_flat.max())
        in_k_range = k_lo_val >= k_min and k_hi_val <= k_max
        k_clip = k_flat if in_k_range else np.clip(k_flat, k_min, k_max)
        seg_idx = np.clip(
            np.searchsorted(self.strikes, k_clip, side="right") - 1,
            0,
            self.n_unique_strikes - 2,
        )
        dx = k_clip - self.strikes[seg_idx]

        assert self._fast_slice_coeffs is not None
        if self.n_exp == 1:
            c = self._fast_slice_coeffs[:, seg_idx, 0]
            v = np.maximum(((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3], self.min_vol)
            if not in_k_range:
                if k_lo_val < k_min:
                    v = np.where(
                        k_flat < k_min,
                        np.maximum(self.quotes[0, 0] + self._slopes_left[0] * (k_flat - k_min), self.min_vol),
                        v,
                    )
                if k_hi_val > k_max:
                    v = np.where(
                        k_flat > k_max,
                        np.maximum(self.quotes[0, -1] + self._slopes_right[0] * (k_flat - k_max), self.min_vol),
                        v,
                    )
            return np.asarray(v.reshape(orig_shape), dtype=np.float64)

        idx0 = np.clip(np.searchsorted(self.tenors, t_flat, side="right") - 1, 0, self.n_exp - 2)
        idx1 = idx0 + 1
        coeffs = self._fast_slice_coeffs
        c0 = coeffs[:, seg_idx, idx0]
        c1 = coeffs[:, seg_idx, idx1]
        v0 = ((c0[0] * dx + c0[1]) * dx + c0[2]) * dx + c0[3]
        v1 = ((c1[0] * dx + c1[1]) * dx + c1[2]) * dx + c1[3]
        if not in_k_range:
            if k_lo_val < k_min:
                left_mask = k_flat < k_min
                v0 = np.where(left_mask, self.quotes[idx0, 0] + self._slopes_left[idx0] * (k_flat - k_min), v0)
                v1 = np.where(left_mask, self.quotes[idx1, 0] + self._slopes_left[idx1] * (k_flat - k_min), v1)
            if k_hi_val > k_max:
                right_mask = k_flat > k_max
                v0 = np.where(right_mask, self.quotes[idx0, -1] + self._slopes_right[idx0] * (k_flat - k_max), v0)
                v1 = np.where(right_mask, self.quotes[idx1, -1] + self._slopes_right[idx1] * (k_flat - k_max), v1)
        v0 = np.maximum(v0, self.min_vol)
        v1 = np.maximum(v1, self.min_vol)

        t0 = self.tenors[idx0]
        t1 = self.tenors[idx1]
        alpha = (t_flat - t0) / np.maximum(t1 - t0, 1e-12)

        if time_interp == "variance":
            w = (1.0 - alpha) * (v0 * v0 * t0) + alpha * (v1 * v1 * t1)
            res = np.sqrt(np.maximum(w / t_flat, self.min_vol * self.min_vol))
        else:
            res = np.maximum((1.0 - alpha) * v0 + alpha * v1, self.min_vol)

        t_lo_val = float(t_flat.min())
        t_hi_val = float(t_flat.max())
        if t_lo_val <= float(self.tenors[0]):
            res = np.where(t_flat <= float(self.tenors[0]), v0, res)
        if t_hi_val >= float(self.tenors[-1]):
            res = np.where(t_flat >= float(self.tenors[-1]), v1, res)
        return np.asarray(res.reshape(orig_shape), dtype=np.float64)

    def _eval_basis_u_vec(
        self,
        k_val: float,
        slice_idx: Optional[int] = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluates strike basis weights in unique strike space for optional slice index."""
        self._ensure_basis_spline()
        k_arr = np.array([k_val], dtype=np.float64)
        k_min = float(self.strikes[0])
        k_max = float(self.strikes[-1])
        k_clip = np.clip(k_arr, k_min, k_max)

        coeffs = self._fast_basis_coeffs
        sl_left = self._basis_slope_left
        sl_right = self._basis_slope_right
        if (
            slice_idx is not None
            and self._fast_basis_coeffs_per_slice is not None
            and self._basis_slope_left_per_slice is not None
            and self._basis_slope_right_per_slice is not None
        ):
            coeffs = self._fast_basis_coeffs_per_slice[slice_idx]
            sl_left = self._basis_slope_left_per_slice[slice_idx]
            sl_right = self._basis_slope_right_per_slice[slice_idx]

        if coeffs is not None:
            w_u_vec = eval_clamped_cubic_coeffs(self.strikes, coeffs, k_clip)[0]
        else:
            w_u_vec = np.asarray(self._basis_cs(k_clip)[0], dtype=np.float64)

        if k_val < k_min and sl_left is not None:
            w_u_vec = np.zeros(self.n_unique_strikes, dtype=np.float64)
            w_u_vec[0] = 1.0
            w_u_vec = w_u_vec + (k_val - k_min) * sl_left
        elif k_val > k_max and sl_right is not None:
            w_u_vec = np.zeros(self.n_unique_strikes, dtype=np.float64)
            w_u_vec[-1] = 1.0
            w_u_vec = w_u_vec + (k_val - k_max) * sl_right
        return np.asarray(w_u_vec, dtype=np.float64)

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
        jac_u = np.zeros((self.n_exp, self.n_unique_strikes), dtype=np.float64)

        k_arr_1 = np.array([k_val], dtype=np.float64)
        min_var_floor = self.min_vol * self.min_vol

        if t_val <= float(self.tenors[0]):
            v_raw = float(self._eval_slice_raw(0, k_arr_1)[0])
            sig = max(v_raw, self.min_vol)
            if v_raw > self.min_vol:
                jac_u[0, :] = self._eval_basis_u_vec(k_val, slice_idx=0)
        elif t_val >= float(self.tenors[-1]):
            last_idx = self.n_exp - 1
            v_raw = float(self._eval_slice_raw(last_idx, k_arr_1)[0])
            sig = max(v_raw, self.min_vol)
            if v_raw > self.min_vol:
                jac_u[last_idx, :] = self._eval_basis_u_vec(k_val, slice_idx=last_idx)
        else:
            idx = int(np.searchsorted(self.tenors, t_val, side="right")) - 1
            idx = max(0, min(idx, self.n_exp - 2))
            t0 = float(self.tenors[idx])
            t1 = float(self.tenors[idx + 1])
            v0_raw = float(self._eval_slice_raw(idx, k_arr_1)[0])
            v1_raw = float(self._eval_slice_raw(idx + 1, k_arr_1)[0])
            v0 = max(v0_raw, self.min_vol)
            v1 = max(v1_raw, self.min_vol)

            alpha = (t_val - t0) / max(t1 - t0, 1e-12)
            w0 = (v0 * v0) * t0
            w1 = (v1 * v1) * t1
            w = (1.0 - alpha) * w0 + alpha * w1
            w_over_t = w / t_val
            sig = math.sqrt(max(w_over_t, min_var_floor))

            if w_over_t > min_var_floor:
                if v0_raw > self.min_vol:
                    d_sig_dv0 = (1.0 - alpha) * t0 * v0 / max(t_val * sig, 1e-12)
                    jac_u[idx, :] = d_sig_dv0 * self._eval_basis_u_vec(k_val, slice_idx=idx)
                if v1_raw > self.min_vol:
                    d_sig_dv1 = alpha * t1 * v1 / max(t_val * sig, 1e-12)
                    jac_u[idx + 1, :] = d_sig_dv1 * self._eval_basis_u_vec(k_val, slice_idx=idx + 1)

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
        iv_wing_vals: Optional[Sequence[float] | npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Fast value-only extraction of the 7 surface features for AutocallNodeCol (cols 4..10)."""
        iv_short = (
            float(self.iv(t_short, float(f_short)))
            if iv_short_val is None
            else float(iv_short_val)
        )
        if iv_term_val is not None and iv_wing_vals is not None and len(iv_wing_vals) >= 4:
            iv_term = float(iv_term_val)
            iv_m10 = float(iv_wing_vals[0])
            iv_p10 = float(iv_wing_vals[1])
            iv_m20 = float(iv_wing_vals[2])
            iv_p20 = float(iv_wing_vals[3])
        elif iv_term_val is None:
            t_vec = np.full(5, float(t_term), dtype=np.float64)
            k_term_vec = np.array(
                [f_term, 0.90 * f_term, 1.10 * f_term, 0.80 * f_term, 1.20 * f_term],
                dtype=np.float64,
            )
            iv_term_vec = self.iv_batch(t_vec, k_term_vec)
            iv_term = float(iv_term_vec[0])
            iv_m10 = float(iv_term_vec[1])
            iv_p10 = float(iv_term_vec[2])
            iv_m20 = float(iv_term_vec[3])
            iv_p20 = float(iv_term_vec[4])
        else:
            iv_term = float(iv_term_val)
            t_vec = np.full(4, float(t_term), dtype=np.float64)
            k_term_vec = np.array(
                [0.90 * f_term, 1.10 * f_term, 0.80 * f_term, 1.20 * f_term],
                dtype=np.float64,
            )
            iv_term_vec = self.iv_batch(t_vec, k_term_vec)
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


def ensure_basis_spline_coeffs(surface: Any) -> npt.NDArray[np.float64]:
    """Thread-safe RegularizedVolSurface._ensure_basis_spline() accessor (clamped, _basis_cs & PCHIP)."""
    if hasattr(surface, "ensure_basis_spline_coeffs") and callable(surface.ensure_basis_spline_coeffs):
        return cast(npt.NDArray[np.float64], surface.ensure_basis_spline_coeffs())
    if hasattr(surface, "_ensure_basis_spline") and callable(surface._ensure_basis_spline):
        surface._ensure_basis_spline()
        coeffs = getattr(surface, "_fast_basis_coeffs", None)
        if coeffs is not None:
            return cast(npt.NDArray[np.float64], coeffs)
    raise RuntimeError("Surface does not provide basis spline initialization.")


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


def _build_bb_orthogonal_np(tau_obs_np: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Constructs exact orthogonal (M, M) dyadic Brownian-bridge temporal basis."""
    m_rem = int(len(tau_obs_np))
    if m_rem <= 1:
        return np.ones((max(m_rem, 1), max(m_rem, 1)), dtype=np.float64)
    dt_arr = np.maximum(np.diff(np.concatenate([[0.0], tau_obs_np.astype(np.float64)])), 1e-12)
    t_cum = np.concatenate([[0.0], np.cumsum(dt_arr)])
    c_path = np.zeros((m_rem + 1, m_rem), dtype=np.float64)
    c_path[m_rem, 0] = math.sqrt(float(t_cum[m_rem]))
    queue: list[tuple[int, int]] = [(0, m_rem)]
    col_idx = 1
    head = 0
    while head < len(queue):
        l_idx, r_idx = queue[head]
        head += 1
        if r_idx - l_idx <= 1:
            continue
        mid = (l_idx + r_idx) // 2
        tl, tm, tr = float(t_cum[l_idx]), float(t_cum[mid]), float(t_cum[r_idx])
        denom = max(tr - tl, 1e-12)
        c_path[mid, :] = ((tr - tm) / denom) * c_path[l_idx, :] + ((tm - tl) / denom) * c_path[r_idx, :]
        c_path[mid, col_idx] = math.sqrt(max((tm - tl) * (tr - tm) / denom, 0.0))
        col_idx += 1
        queue.append((l_idx, mid))
        queue.append((mid, r_idx))
    inv_sqrt_dt = 1.0 / np.sqrt(dt_arr)[:, None]
    return np.ascontiguousarray((c_path[1:, :] - c_path[:-1, :]) * inv_sqrt_dt, dtype=np.float64)


def _lookup_funding_fixing_utc(
    fixings_pairs: Sequence[Tuple[datetime, float]],
    target_dt: datetime,
) -> Optional[float]:
    """Finds an exact or same-calendar-day UTC fixing for a funding reset date."""
    if not fixings_pairs:
        return None
    target_date = target_dt.date()
    fallback: Optional[float] = None
    for f_dt, f_val in fixings_pairs:
        if f_dt == target_dt:
            return float(f_val)
        if f_dt.date() == target_date:
            fallback = float(f_val)
    return fallback


def _compute_funding_weights_np(state: Any) -> npt.NDArray[np.float64]:
    """Computes deterministic per-period discounted FundingLeg weights w_fund[m] in float64."""
    m_rem = int(state.m_rem)
    w_fund = np.zeros(m_rem, dtype=np.float64)
    if state.funding_leg is None or m_rem == 0:
        return w_fund

    from deepvol.models.autocall.escrowed_dividend import build_discount_rate_evaluator

    f_leg = state.funding_leg
    raw_f_rid = f_leg.get("rate_identifier")
    f_rid = str(raw_f_rid).strip() if raw_f_rid is not None else ""
    f_curve = f_leg.get("funding_curve")
    as_of_dt = _to_utc_datetime(state.seasoned_state.as_of_date)
    dcc_raw = f_leg.get(
        "day_count",
        f_leg.get("day_count_convention", state.seasoned_state.day_count_convention),
    )
    dcc = str(getattr(dcc_raw, "value", dcc_raw) or state.seasoned_state.day_count_convention)

    _, df_fin_fn = build_discount_rate_evaluator(state.discount_curve, as_of_dt)
    disc_id = getattr(state, "discount_curve_id", None) or getattr(
        state.discount_curve, "identifier", None
    )
    f_curve_id = f_leg.get("funding_curve_id") or getattr(f_curve, "identifier", None)
    if (
        bool(f_rid)
        and f_curve is not None
        and (f_curve is not state.discount_curve)
        and (disc_id is None or f_curve_id is None or f_curve_id != disc_id)
    ):
        _, df_fwd_fn = build_discount_rate_evaluator(f_curve, as_of_dt)
    else:
        df_fwd_fn = df_fin_fn

    raw_spread = f_leg.get("funding_spread", 0.0)
    f_spread = 0.0 if raw_spread is None else float(raw_spread)
    f_dates = f_leg.get("payment_dates") or ()
    f_fixings_raw = f_leg.get("funding_fixings") or ()
    if isinstance(f_fixings_raw, Mapping):
        f_fixings = [(_to_utc_datetime(dt_r), float(val_r)) for dt_r, val_r in f_fixings_raw.items()]
    else:
        f_fixings = [(_to_utc_datetime(dt_r), float(val_r)) for dt_r, val_r in f_fixings_raw]
    pay_dates_utc = [_to_utc_datetime(p_dt) for p_dt in state.seasoned_state.future_pay_dates[:m_rem]]
    if not pay_dates_utc:
        return w_fund
    max_future_pay_dt = pay_dates_utc[-1]
    hist_recs = getattr(state.seasoned_state, "historical_records", ()) or ()
    max_past_pay_dt = max((_to_utc_datetime(r.payment_date) for r in hist_recs), default=None)
    prev_f_dt = _to_utc_datetime(getattr(state.seasoned_state, "start_date", as_of_dt) or as_of_dt)

    for f_dt_raw in f_dates:
        f_dt = _to_utc_datetime(f_dt_raw)
        if f_dt <= as_of_dt:
            prev_f_dt = f_dt
            continue
        if max_past_pay_dt is not None and f_dt <= max_past_pay_dt:
            prev_f_dt = f_dt
            continue
        if f_dt > max_future_pay_dt:
            prev_f_dt = f_dt
            continue
        alpha_j = _surface_year_fraction(prev_f_dt, f_dt, dcc)
        df_curr = float(df_fin_fn(f_dt))
        if not f_rid:
            r_fwd_j = 0.0
        else:
            fixed_reset = (
                _lookup_funding_fixing_utc(f_fixings, prev_f_dt) if prev_f_dt < as_of_dt else None
            )
            if fixed_reset is not None:
                r_fwd_j = fixed_reset
            elif prev_f_dt < as_of_dt:
                alpha_rem = max(_surface_year_fraction(as_of_dt, f_dt, dcc), 1e-8)
                df_curr_fwd = float(df_fwd_fn(f_dt))
                r_fwd_j = (1.0 / max(df_curr_fwd, 1e-12) - 1.0) / alpha_rem
            else:
                df_prev_fwd = float(df_fwd_fn(prev_f_dt))
                df_curr_fwd = float(df_fwd_fn(f_dt))
                r_fwd_j = (df_prev_fwd / max(df_curr_fwd, 1e-12) - 1.0) / max(alpha_j, 1e-8)

        m_idx = m_rem - 1
        for m, pay_dt in enumerate(pay_dates_utc):
            if pay_dt >= f_dt:
                m_idx = m
                break
        w_fund[m_idx] += df_curr * (r_fwd_j + f_spread) * alpha_j
        prev_f_dt = f_dt
    return w_fund


def _eval_index_by_code_np(
    perf_sorted: npt.NDArray[np.float64],
    code: int,
    inv_lex_perm: Optional[npt.NDArray[np.int64]],
) -> npt.NDArray[np.float64]:
    """Evaluates any of the 18 canonical monitoring functionals on sorted paths in float64."""
    if code in _PAIR_ORDER_INDEX_CODES and inv_lex_perm is not None:
        p = perf_sorted[..., inv_lex_perm]
    else:
        p = perf_sorted
    d = p.shape[-1]

    if code == int(MonitoringIndexCode.WORST_OF):
        return np.asarray(np.min(p, axis=-1), dtype=np.float64)
    if code == int(MonitoringIndexCode.BEST_OF):
        return np.asarray(np.max(p, axis=-1), dtype=np.float64)
    if code == int(MonitoringIndexCode.AVERAGE_OF):
        return np.asarray(np.mean(p, axis=-1), dtype=np.float64)
    if code == int(MonitoringIndexCode.DISPERSION):
        if d <= 1:
            return np.zeros(p.shape[:-1], dtype=np.float64)
        mu = np.mean(p, axis=-1, keepdims=True)
        return np.asarray(np.sqrt(np.maximum(np.mean((p - mu) ** 2, axis=-1), 0.0)), dtype=np.float64)
    if code == int(MonitoringIndexCode.RATIO_AB):
        return np.asarray(p[..., 0] if d < 2 else p[..., 0] / np.maximum(p[..., 1], 1e-12), dtype=np.float64)
    if code == int(MonitoringIndexCode.RATIO_BA):
        return np.asarray(p[..., 0] if d < 2 else p[..., 1] / np.maximum(p[..., 0], 1e-12), dtype=np.float64)
    if code == int(MonitoringIndexCode.RATIO_BEST_WORST):
        return np.asarray(np.max(p, axis=-1) / np.maximum(np.min(p, axis=-1), 1e-12), dtype=np.float64)
    if code == int(MonitoringIndexCode.RATIO_BEST_AVERAGE):
        return np.asarray(np.max(p, axis=-1) / np.maximum(np.mean(p, axis=-1), 1e-12), dtype=np.float64)
    if code == int(MonitoringIndexCode.RATIO_AVERAGE_WORST):
        return np.asarray(np.mean(p, axis=-1) / np.maximum(np.min(p, axis=-1), 1e-12), dtype=np.float64)
    if code == int(MonitoringIndexCode.DIFF_AB):
        return np.zeros(p.shape[:-1], dtype=np.float64) if d < 2 else np.asarray(p[..., 0] - p[..., 1], dtype=np.float64)
    if code == int(MonitoringIndexCode.DIFF_BA):
        return np.zeros(p.shape[:-1], dtype=np.float64) if d < 2 else np.asarray(p[..., 1] - p[..., 0], dtype=np.float64)
    if code == int(MonitoringIndexCode.DIFF_BEST_WORST):
        return np.asarray(np.max(p, axis=-1) - np.min(p, axis=-1), dtype=np.float64)
    if code == int(MonitoringIndexCode.DIFF_BEST_AVERAGE):
        return np.asarray(np.max(p, axis=-1) - np.mean(p, axis=-1), dtype=np.float64)
    if code == int(MonitoringIndexCode.DIFF_AVERAGE_WORST):
        return np.asarray(np.mean(p, axis=-1) - np.min(p, axis=-1), dtype=np.float64)
    if code == int(MonitoringIndexCode.TWIN_WIN):
        w_min = np.min(p, axis=-1)
        return np.asarray(np.where(w_min < 1.0, 2.0 - w_min, w_min), dtype=np.float64)
    if code == int(MonitoringIndexCode.SECOND_WORST_OF):
        k = min(1, max(d - 1, 0))
        return np.asarray(np.partition(p, k, axis=-1)[..., k], dtype=np.float64)
    if code == int(MonitoringIndexCode.THIRD_WORST_OF):
        k = min(2, max(d - 1, 0))
        return np.asarray(np.partition(p, k, axis=-1)[..., k], dtype=np.float64)
    if code == int(MonitoringIndexCode.SECOND_BEST_OF):
        k = max(0, d - 2)
        return np.asarray(np.partition(p, k, axis=-1)[..., k], dtype=np.float64)
    return np.asarray(np.min(p, axis=-1), dtype=np.float64)


def _evaluate_index_with_unperm_np(
    perf_sorted: npt.NDArray[np.float64],
    index_name: str,
    inv_lex_perm: Optional[npt.NDArray[np.int64]],
) -> npt.NDArray[np.float64]:
    """Evaluates monitoring functional, restoring caller ordering for pair indices."""
    code = resolve_monitoring_index_code(index_name)
    return _eval_index_by_code_np(perf_sorted, code, inv_lex_perm)


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


def _black76_put_np(f: float, k: float, total_vol: float) -> float:
    """Exact undiscounted Black-76 put price E[max(K - F*exp(-0.5 v^2 + v Z), 0)]."""
    if k <= 0.0:
        return 0.0
    if f <= 0.0:
        return k
    if total_vol <= 1e-12:
        return max(k - f, 0.0)
    d1 = (math.log(f / k) + 0.5 * total_vol * total_vol) / total_vol
    d2 = d1 - total_vol
    return k * float(ndtr(-d2)) - f * float(ndtr(-d1))


def _apply_loo_control_variate_np(
    y_paths: npt.NDArray[np.float64],
    x_paths: npt.NDArray[np.float64],
    mu_x: float,
) -> Tuple[float, float]:
    """Leave-one-replicate-out unbiased RQMC control variate estimator."""
    eff_r, n_rep = y_paths.shape
    y_bar = np.mean(y_paths, axis=1)
    if eff_r < 2:
        raw_mean = float(np.mean(y_bar))
        raw_se = float(np.std(y_paths.reshape(-1), ddof=1) / math.sqrt(max(n_rep, 1)))
        return raw_mean, raw_se

    x_bar = np.mean(x_paths, axis=1)
    # Estimate slopes from (R - 1) * n_rep path deviations, then shrink on held-out
    # replicate means so CV never amplifies RQMC replicate variance on tail-heavy payoffs.
    dy = y_paths - y_bar[:, None]
    dx = x_paths - x_bar[:, None]
    cov_xy_r = np.sum(dy * dx, axis=1)
    var_xx_r = np.sum(dx * dx, axis=1)
    var_yy_r = np.sum(dy * dy, axis=1)
    tot_cov = float(np.sum(cov_xy_r))
    tot_var = float(np.sum(var_xx_r))
    tot_yvar = float(np.sum(var_yy_r))

    loo_cov = tot_cov - cov_xy_r
    loo_var = tot_var - var_xx_r
    loo_yvar = tot_yvar - var_yy_r
    var_thresh = 1e-18 * max(1.0, tot_var)
    beta_path = np.where(loo_var > var_thresh, loo_cov / np.maximum(loo_var, var_thresh), 0.0)

    if eff_r >= 3:
        tot_y = float(np.sum(y_bar))
        tot_x = float(np.sum(x_bar))
        sum_yy = float(np.sum(y_bar * y_bar))
        sum_yx = float(np.sum(y_bar * x_bar))
        sum_xx = float(np.sum(x_bar * x_bar))
        rm1 = float(eff_r - 1)
        my = (tot_y - y_bar) / rm1
        mx = (tot_x - x_bar) / rm1
        s_yy = np.maximum((sum_yy - y_bar * y_bar) - rm1 * my * my, 0.0)
        s_yx = (sum_yx - y_bar * x_bar) - rm1 * my * mx
        s_xx = np.maximum((sum_xx - x_bar * x_bar) - rm1 * mx * mx, 0.0)
        var_cv_path = s_yy - 2.0 * beta_path * s_yx + beta_path * beta_path * s_xx
        denom_bx = beta_path * s_xx
        safe_denom = np.where(np.abs(denom_bx) > 1e-30, denom_bx, 1.0)
        alpha = np.where(
            var_cv_path <= s_yy,
            1.0,
            np.where(
                (beta_path * s_yx > 0.0) & (s_xx > 1e-24 * max(1.0, sum_xx)),
                np.clip(s_yx / safe_denom, 0.0, 1.0),
                0.0,
            ),
        )
        beta_loo = alpha * beta_path
    else:
        y_thresh = 1e-18 * max(1.0, tot_yvar)
        r2_path = np.where(
            (loo_var > var_thresh) & (loo_yvar > y_thresh),
            (loo_cov * loo_cov) / np.maximum(loo_var * loo_yvar, 1e-30),
            0.0,
        )
        beta_loo = np.where(r2_path >= 0.1, beta_path, 0.0)

    z_rep = y_bar - beta_loo * (x_bar - mu_x)
    est = float(np.mean(z_rep))
    se = float(np.std(z_rep, ddof=1) / math.sqrt(float(eff_r)))
    return est, se


def _eval_index_log_weights_np(
    perf: npt.NDArray[np.float64],
    code: int,
    inv_lex_perm: Optional[npt.NDArray[np.int64]],
) -> npt.NDArray[np.float64]:
    """Returns approximate dW/d(log S_i) sensitivity weights along last axis for BB KI variance."""
    d = perf.shape[-1]
    if code in _PAIR_ORDER_INDEX_CODES and inv_lex_perm is not None:
        p = perf[..., inv_lex_perm]
    else:
        p = perf
    w = np.zeros_like(p, dtype=np.float64)

    def _set_onehot(idx_arr: npt.NDArray[np.intp], val: Union[float, npt.NDArray[np.float64]]) -> None:
        np.put_along_axis(w, idx_arr[..., None], np.asarray(val, dtype=np.float64)[..., None], axis=-1)

    if code == int(MonitoringIndexCode.WORST_OF):
        _set_onehot(np.argmin(p, axis=-1), np.min(p, axis=-1))
    elif code == int(MonitoringIndexCode.BEST_OF):
        _set_onehot(np.argmax(p, axis=-1), np.max(p, axis=-1))
    elif code == int(MonitoringIndexCode.AVERAGE_OF):
        w = p / float(max(d, 1))
    elif code == int(MonitoringIndexCode.RATIO_AB):
        if d >= 2:
            r_val = p[..., 0] / np.maximum(p[..., 1], 1e-12)
            w[..., 0] = r_val
            w[..., 1] = -r_val
        else:
            w[..., 0] = p[..., 0]
    elif code == int(MonitoringIndexCode.RATIO_BA):
        if d >= 2:
            r_val = p[..., 1] / np.maximum(p[..., 0], 1e-12)
            w[..., 1] = r_val
            w[..., 0] = -r_val
        else:
            w[..., 0] = p[..., 0]
    elif code == int(MonitoringIndexCode.RATIO_BEST_WORST):
        r_val = np.max(p, axis=-1) / np.maximum(np.min(p, axis=-1), 1e-12)
        np.put_along_axis(w, np.argmax(p, axis=-1)[..., None], r_val[..., None], axis=-1)
        cur_min = np.take_along_axis(w, np.argmin(p, axis=-1)[..., None], axis=-1)
        np.put_along_axis(w, np.argmin(p, axis=-1)[..., None], cur_min - r_val[..., None], axis=-1)
    elif code == int(MonitoringIndexCode.RATIO_BEST_AVERAGE):
        mean_p = np.maximum(np.mean(p, axis=-1), 1e-12)
        max_p = np.max(p, axis=-1)
        r_val = max_p / mean_p
        w = -(r_val[..., None] * p) / (float(max(d, 1)) * mean_p[..., None])
        cur_max = np.take_along_axis(w, np.argmax(p, axis=-1)[..., None], axis=-1)
        np.put_along_axis(w, np.argmax(p, axis=-1)[..., None], cur_max + r_val[..., None], axis=-1)
    elif code == int(MonitoringIndexCode.RATIO_AVERAGE_WORST):
        min_p = np.maximum(np.min(p, axis=-1), 1e-12)
        mean_p = np.mean(p, axis=-1)
        r_val = mean_p / min_p
        w = p / (float(max(d, 1)) * min_p[..., None])
        cur_min = np.take_along_axis(w, np.argmin(p, axis=-1)[..., None], axis=-1)
        np.put_along_axis(w, np.argmin(p, axis=-1)[..., None], cur_min - r_val[..., None], axis=-1)
    elif code == int(MonitoringIndexCode.DIFF_AB):
        if d >= 2:
            w[..., 0] = p[..., 0]
            w[..., 1] = -p[..., 1]
    elif code == int(MonitoringIndexCode.DIFF_BA):
        if d >= 2:
            w[..., 1] = p[..., 1]
            w[..., 0] = -p[..., 0]
    elif code == int(MonitoringIndexCode.DIFF_BEST_WORST):
        np.put_along_axis(w, np.argmax(p, axis=-1)[..., None], np.max(p, axis=-1)[..., None], axis=-1)
        cur_min = np.take_along_axis(w, np.argmin(p, axis=-1)[..., None], axis=-1)
        np.put_along_axis(w, np.argmin(p, axis=-1)[..., None], cur_min - np.min(p, axis=-1)[..., None], axis=-1)
    elif code == int(MonitoringIndexCode.DIFF_BEST_AVERAGE):
        w = -p / float(max(d, 1))
        cur_max = np.take_along_axis(w, np.argmax(p, axis=-1)[..., None], axis=-1)
        np.put_along_axis(w, np.argmax(p, axis=-1)[..., None], cur_max + np.max(p, axis=-1)[..., None], axis=-1)
    elif code == int(MonitoringIndexCode.DIFF_AVERAGE_WORST):
        w = p / float(max(d, 1))
        cur_min = np.take_along_axis(w, np.argmin(p, axis=-1)[..., None], axis=-1)
        np.put_along_axis(w, np.argmin(p, axis=-1)[..., None], cur_min - np.min(p, axis=-1)[..., None], axis=-1)
    elif code == int(MonitoringIndexCode.TWIN_WIN):
        w_min = np.min(p, axis=-1)
        sgn = np.where(w_min < 1.0, -w_min, w_min)
        _set_onehot(np.argmin(p, axis=-1), sgn)
    elif code == int(MonitoringIndexCode.SECOND_WORST_OF):
        order = np.argsort(p, axis=-1)
        idx = order[..., min(1, max(d - 1, 0))]
        _set_onehot(idx, np.take_along_axis(p, idx[..., None], axis=-1)[..., 0])
    elif code == int(MonitoringIndexCode.THIRD_WORST_OF):
        order = np.argsort(p, axis=-1)
        idx = order[..., min(2, max(d - 1, 0))]
        _set_onehot(idx, np.take_along_axis(p, idx[..., None], axis=-1)[..., 0])
    elif code == int(MonitoringIndexCode.SECOND_BEST_OF):
        order = np.argsort(p, axis=-1)
        idx = order[..., max(0, d - 2)]
        _set_onehot(idx, np.take_along_axis(p, idx[..., None], axis=-1)[..., 0])
    elif code == int(MonitoringIndexCode.DISPERSION) and d > 1:
        mu = np.mean(p, axis=-1, keepdims=True)
        disp = np.maximum(np.sqrt(np.maximum(np.mean((p - mu) ** 2, axis=-1, keepdims=True), 0.0)), 1e-6)
        w = ((p - mu) / (float(d) * disp)) * p

    if code in _PAIR_ORDER_INDEX_CODES and inv_lex_perm is not None:
        w_out = np.empty_like(w)
        w_out[..., inv_lex_perm] = w
        return w_out
    return w


def _eval_continuous_ki_step_survival_np(
    perf_prev: npt.NDArray[np.float64],
    perf_curr: npt.NDArray[np.float64],
    corr_np: npt.NDArray[np.float64],
    lv_step: npt.NDArray[np.float64],
    dt_step: float,
    opt_code: int,
    ki_barrier: float,
    inv_lex_perm: Optional[npt.NDArray[np.int64]],
    *,
    lv_step_end: Optional[npt.NDArray[np.float64]] = None,
    lv_barrier: Optional[npt.NDArray[np.float64]] = None,
    bgk_daily: bool = False,
) -> npt.NDArray[np.float64]:
    """Shevchenko (2009) & BGK (1997) Brownian-bridge non-knock-in survival over [t0, t1]."""
    if opt_code == int(MonitoringIndexCode.WORST_OF) and ki_barrier > 0.0:
        lv0 = np.maximum(lv_step, 1e-6)
        lv1 = np.maximum(lv_step_end, 1e-6) if lv_step_end is not None else lv0
        lv_b = np.maximum(lv_barrier, 1e-6) if lv_barrier is not None else lv0
        if bgk_daily:
            b_eff = ki_barrier * np.exp(-0.5826 * lv_b * math.sqrt(1.0 / 252.0))
        else:
            b_eff = np.full_like(lv_b, ki_barrier, dtype=np.float64)
        alive_i = (perf_prev > b_eff) & (perf_curr > b_eff)
        if not np.any(np.all(alive_i, axis=-1)):
            return np.zeros(perf_curr.shape[:-1], dtype=np.float64)
        # Gobet-Menozzi (2010) Simpson-rule barrier-pinned local variance
        var_simpson = (lv0 * lv0 + 4.0 * (lv_b * lv_b) + lv1 * lv1) / 6.0
        p_prev_s = np.where(alive_i, np.maximum(perf_prev, b_eff + 1e-12), b_eff + 0.1)
        p_curr_s = np.where(alive_i, np.maximum(perf_curr, b_eff + 1e-12), b_eff + 0.1)
        eta_i = (
            2.0
            * np.log(p_prev_s / b_eff)
            * np.log(p_curr_s / b_eff)
            / np.maximum(var_simpson * max(dt_step, 1e-12), 1e-12)
        )
        surv_i = np.where(alive_i, 1.0 - np.exp(-np.maximum(eta_i, 0.0)), 0.0)
        return np.asarray(np.prod(surv_i, axis=-1), dtype=np.float64)

    w_prev = _eval_index_by_code_np(perf_prev, opt_code, inv_lex_perm)
    w_curr = _eval_index_by_code_np(perf_curr, opt_code, inv_lex_perm)
    perf_mid = 0.5 * (perf_prev + perf_curr)
    dw_dlog = _eval_index_log_weights_np(perf_mid, opt_code, inv_lex_perm)
    weighted = dw_dlog * lv_step
    var_abs = np.einsum("...i,ij,...j->...", weighted, corr_np, weighted)

    if opt_code in _DIFF_INDEX_CODES or ki_barrier <= 0.0:
        b_eff_idx = np.full_like(w_curr, ki_barrier, dtype=np.float64)
        alive_pair = (w_prev > b_eff_idx) & (w_curr > b_eff_idx)
        if not np.any(alive_pair):
            return np.zeros_like(w_curr, dtype=np.float64)
        eta = (
            2.0
            * np.maximum(w_prev - b_eff_idx, 0.0)
            * np.maximum(w_curr - b_eff_idx, 0.0)
            / np.maximum(var_abs * max(dt_step, 1e-12), 1e-12)
        )
    else:
        w_mid = np.maximum(_eval_index_by_code_np(perf_mid, opt_code, inv_lex_perm), 1e-12)
        var_log = var_abs / (w_mid * w_mid)
        if bgk_daily:
            b_eff_idx = ki_barrier * np.exp(-0.5826 * np.sqrt(np.maximum(var_log, 1e-12)) * math.sqrt(1.0 / 252.0))
        else:
            b_eff_idx = np.full_like(w_curr, ki_barrier, dtype=np.float64)
        alive_pair = (w_prev > b_eff_idx) & (w_curr > b_eff_idx)
        if not np.any(alive_pair):
            return np.zeros_like(w_curr, dtype=np.float64)
        w_p_s = np.where(alive_pair, np.maximum(w_prev, b_eff_idx + 1e-12), b_eff_idx + 0.1)
        w_c_s = np.where(alive_pair, np.maximum(w_curr, b_eff_idx + 1e-12), b_eff_idx + 0.1)
        eta = (
            2.0
            * np.log(w_p_s / b_eff_idx)
            * np.log(w_c_s / b_eff_idx)
            / np.maximum(var_log * max(dt_step, 1e-12), 1e-12)
        )
    return np.where(alive_pair, 1.0 - np.exp(-np.maximum(eta, 0.0)), 0.0)


def _evaluate_autocall_payoff_np(
    state: Any,
    perf_paths: npt.NDArray[np.float64],
    *,
    df_pay: npt.NDArray[np.float64],
    ko_barriers: npt.NDArray[np.float64],
    is_callable: npt.NDArray[np.float64],
    cpn_barriers: npt.NDArray[np.float64],
    is_guaranteed_cpn: npt.NDArray[np.float64],
    coupon_amounts: npt.NDArray[np.float64],
    w_fund: npt.NDArray[np.float64],
    inv_lex_perm: Optional[npt.NDArray[np.int64]],
    ki_ind_override: Optional[npt.NDArray[np.float64]] = None,
    ki_mode: str = "European",
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
    float,
    float,
    float,
]:
    """Evaluates pathwise autocall cashflows and terminal option payoff in float64."""
    n_total_paths, m_rem, _ = perf_paths.shape
    alive = np.ones(n_total_paths, dtype=np.float64)
    unpaid_mem = np.full(n_total_paths, float(state.seasoned_state.c_mem_0), dtype=np.float64)
    total_pv_paths = np.zeros(n_total_paths, dtype=np.float64)

    p_stop_arr = np.zeros(m_rem, dtype=np.float64)
    q_cpn_arr = np.zeros(m_rem, dtype=np.float64)
    cpn_cf_expected = np.zeros(m_rem, dtype=np.float64)

    call_code = resolve_monitoring_index_code(state.callability_index)
    eff_cpn_idx = state.coupon_index if state.coupon_index else state.callability_index
    cpn_code = resolve_monitoring_index_code(eff_cpn_idx)
    raw_opt_idx = state.option_index
    is_long_flag = bool(getattr(state, "is_long_basket", False))
    eff_opt_idx = (
        "AverageOf"
        if (is_long_flag and (raw_opt_idx is None or str(raw_opt_idx).strip() in ("", "WorstOf")))
        else (raw_opt_idx if raw_opt_idx else state.callability_index)
    )
    opt_code = resolve_monitoring_index_code(eff_opt_idx)
    ki_opt_idx = (
        "WorstOf"
        if is_long_flag
        else (raw_opt_idx if raw_opt_idx else state.callability_index)
    )
    ki_opt_code = resolve_monitoring_index_code(ki_opt_idx)
    is_diff_opt = opt_code in _DIFF_INDEX_CODES

    cpn_dir = str(getattr(state, "coupon_barrier_direction", "Up") or "Up").strip().lower()
    cpn_type = str(getattr(state, "coupon_barrier_type", "In") or "In").strip().lower()

    has_ki = (
        state.ki_barrier is not None
        and (float(state.ki_barrier) > 1e-6 or (is_diff_opt and float(state.ki_barrier) < -1e-6))
        and float(state.seasoned_state.i_ki_past) < 0.5
    )
    discrete_ki_active = has_ki and ki_ind_override is None and ki_mode == "Discrete"
    not_ki_disc = np.ones(n_total_paths, dtype=np.float64)

    for m in range(m_rem):
        perf_m = perf_paths[:, m, :]
        w_call = _eval_index_by_code_np(perf_m, call_code, inv_lex_perm)
        w_cpn = (
            w_call if cpn_code == call_code else _eval_index_by_code_np(perf_m, cpn_code, inv_lex_perm)
        )
        if discrete_ki_active and state.ki_barrier is not None:
            w_opt_m = _eval_index_by_code_np(perf_m, ki_opt_code, inv_lex_perm)
            not_ki_disc *= _digital_ramp_np(w_opt_m, float(state.ki_barrier), float(state.ki_spread))

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
            unpaid_mem = alive * (unpaid_mem + float(coupon_amounts[m])) * (1.0 - cpn_ind)

        alive = alive * (1.0 - ko_ind)
        if not state.no_notional_repayment:
            if m < m_rem - 1:
                total_pv_paths += df_pay[m] * stop_m
            else:
                total_pv_paths += df_pay[m] * (stop_m + alive)

    perf_term = perf_paths[:, -1, :]
    w_opt = _eval_index_by_code_np(perf_term, opt_code, inv_lex_perm)
    star_ind: Optional[npt.NDArray[np.float64]] = None
    if state.one_star_strike is not None and float(state.one_star_strike) > 0.0:
        w_best = _eval_index_by_code_np(perf_term, int(MonitoringIndexCode.BEST_OF), inv_lex_perm)
        k_os = float(state.one_star_strike)
        ko_sp = float(getattr(state, "ko_spread", 0.0) or 0.0)
        if state.ki_barrier is not None and ko_sp < 0.0:
            k_os_eff = k_os + 0.5 * ko_sp
            eps_os = abs(ko_sp)
        else:
            k_os_eff = k_os
            eps_os = 0.02
        star_ind = np.clip((w_best - (k_os_eff - 0.5 * eps_os)) / max(eps_os, 1e-12), 0.0, 1.0)

    k_put = float(state.strike)
    gearing = float(state.option_gearing)
    has_put = ((k_put > 1e-6) or (is_diff_opt and k_put < -1e-6)) and gearing != 0.0
    if is_long_flag:
        w_avg_term = _eval_index_by_code_np(
            perf_term, int(MonitoringIndexCode.AVERAGE_OF), inv_lex_perm
        )
        w_wof_term = _eval_index_by_code_np(
            perf_term, int(MonitoringIndexCode.WORST_OF), inv_lex_perm
        )
        if ki_ind_override is not None:
            ki_wof_k = ki_ind_override
        elif discrete_ki_active and state.ki_barrier is not None:
            ki_wof_k = 1.0 - not_ki_disc
        else:
            ki_wof_k = 1.0 - _digital_ramp_np(w_wof_term, k_put, float(state.ki_spread))
        if star_ind is not None:
            ki_wof_k = ki_wof_k * (1.0 - star_ind)
        eff_gearing = gearing if abs(gearing) > 1e-12 else 1.0
        unit_put_paths = alive * ((1.0 - w_avg_term) * ki_wof_k) / eff_gearing
    elif has_put:
        put_intrinsic = np.maximum(k_put - w_opt, 0.0)
        lk = state.lower_strike
        has_lower_strike = lk is not None and (
            float(lk) > 1e-6 or (is_diff_opt and float(lk) < -1e-6)
        )
        if has_lower_strike and lk is not None:
            put_intrinsic = np.minimum(put_intrinsic, max(k_put - float(lk), 0.0))

        if has_ki and state.ki_barrier is not None:
            if ki_ind_override is not None:
                ki_ind = ki_ind_override
            elif discrete_ki_active:
                ki_ind = 1.0 - not_ki_disc
            else:
                ki_ind = 1.0 - _digital_ramp_np(w_opt, float(state.ki_barrier), float(state.ki_spread))
            put_intrinsic = put_intrinsic * ki_ind

        if star_ind is not None:
            put_intrinsic = put_intrinsic * (1.0 - star_ind)

        unit_put_paths = alive * put_intrinsic
    else:
        unit_put_paths = np.zeros(n_total_paths, dtype=np.float64)

    pcall_rate = float(state.participation_call_rate)
    if state.participation_call_strike is not None and pcall_rate != 0.0:
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
        put_sign * gearing * unit_put_paths + pcall_rate * unit_call_paths
    )
    os_cpn = float(state.one_star_coupon)
    if star_ind is not None and os_cpn != 0.0:
        onestar_paths = alive * star_ind * os_cpn
        opt_payoff_paths = opt_payoff_paths + df_pay[-1] * onestar_paths
        l_onestar_val = float(np.mean(onestar_paths))

    total_pv_paths = total_pv_paths + opt_payoff_paths
    return (
        total_pv_paths,
        opt_payoff_paths,
        p_stop_arr,
        q_cpn_arr,
        cpn_cf_expected,
        l_put_val,
        l_call_val,
        l_onestar_val,
        p_onestar_val,
    )


def _eval_cheap_gbm_expectation_np(
    state: Any,
    *,
    chol_l: npt.NDArray[np.float64],
    d_var_gbm: npt.NDArray[np.float64],
    fwd_rel: npt.NDArray[np.float64],
    df_pay: npt.NDArray[np.float64],
    ko_barriers: npt.NDArray[np.float64],
    is_callable: npt.NDArray[np.float64],
    cpn_barriers: npt.NDArray[np.float64],
    is_guaranteed_cpn: npt.NDArray[np.float64],
    coupon_amounts: npt.NDArray[np.float64],
    w_fund: npt.NDArray[np.float64],
    inv_lex_perm: Optional[npt.NDArray[np.int64]],
    ki_mode: str,
    n_cheap: int,
    seed: int,
) -> float:
    """Evaluates independent cheap M_cheap=4*N scrambled-Sobol flat-ATM GBM expectation on CPU."""
    d_assets, m_rem = fwd_rel.shape
    n_rep_mf = _next_power_of_two(min(max(n_cheap // 4, 512), 4096))
    r_mf = max(1, (n_cheap + n_rep_mf - 1) // n_rep_mf)
    sqrt_dv = np.sqrt(d_var_gbm)
    pv_sum = 0.0
    total_cnt = 0
    for r_idx in range(r_mf):
        z_raw = _generate_sobol_normals(
            n_rep_mf, m_rem * d_assets, scramble=True, seed=seed + r_idx * 997
        ).reshape(n_rep_mf, m_rem, d_assets)
        z_corr = z_raw @ chol_l.T
        d_log = -0.5 * d_var_gbm.T[None, :, :] + sqrt_dv.T[None, :, :] * z_corr
        log_x_mf = np.cumsum(d_log, axis=1)
        perf_mf = fwd_rel.T[None, :, :] * np.exp(log_x_mf)
        pv_r, *_ = _evaluate_autocall_payoff_np(
            state,
            perf_mf,
            df_pay=df_pay,
            ko_barriers=ko_barriers,
            is_callable=is_callable,
            cpn_barriers=cpn_barriers,
            is_guaranteed_cpn=is_guaranteed_cpn,
            coupon_amounts=coupon_amounts,
            w_fund=w_fund,
            inv_lex_perm=inv_lex_perm,
            ki_mode="Discrete" if ki_mode in ("Continuous", "Daily") else ki_mode,
        )
        pv_sum += float(np.sum(pv_r))
        total_cnt += n_rep_mf
    return pv_sum / float(max(total_cnt, 1))


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
    config: Optional[CanonicalTier3Config] = None,
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

    n_cheap_mf_override: Optional[int] = None
    if config is not None:
        n_paths = int(getattr(config, "n_paths", n_paths))
        n_replicates = int(getattr(config, "n_replicates", n_replicates))
        n_substeps_per_period = int(
            getattr(config, "n_substeps", getattr(config, "n_substeps_per_period", n_substeps_per_period))
        )
        cv_mode = cast(
            Literal["bs1d_exact", "multifidelity_gbm", "none"],
            getattr(config, "cv_mode", cv_mode),
        )
        seed = int(getattr(config, "seed", 42 if seed is None else seed))
        raw_n_cheap = getattr(config, "n_cheap_mf", None)
        if raw_n_cheap is not None:
            n_cheap_mf_override = int(raw_n_cheap)

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

    explicit_cpu = (use_gpu is False) or (
        device is not None and not str(device).startswith("cuda") and use_gpu is not True
    )
    want_gpu = False
    if use_gpu is True:
        want_gpu = True
    elif device is not None and str(device).startswith("cuda"):
        want_gpu = True
    elif use_gpu is None and device is None and state.spots_t0.is_cuda:
        want_gpu = True

    if precision != "fp64_audit" and not explicit_cpu and (
        (want_gpu and torch.cuda.is_available()) or int(n_paths) >= 32768
    ):
        from deepvol.models.autocall.dupire_qmc_gpu import (
            DupireQMCConfig,
            DupireQMCGpuSolver,
            GpuTorchDupireQMC,
        )

        gpu_eff_mode: Literal["dupire_lv", "terminal_iv"] = (
            "terminal_iv" if mode == "terminal_iv" else "dupire_lv"
        )
        cfg_kwargs: dict[str, Any] = {
            "n_paths": int(n_paths),
            "n_replicates": int(n_replicates),
            "n_substeps": int(n_substeps_per_period),
            "mode": gpu_eff_mode,
            "precision": precision,
            "cv_mode": cv_mode,
            "seed": 42 if seed is None else int(seed),
        }
        if config is not None:
            for opt_k in ("lv_grid_nt", "lv_grid_ny", "adaptive_substeps"):
                if hasattr(config, opt_k):
                    cfg_kwargs[opt_k] = getattr(config, opt_k)
        cfg = DupireQMCConfig(**cfg_kwargs)
        if want_gpu and torch.cuda.is_available():
            return DupireQMCGpuSolver(device=device or "cuda").simulate_and_price(state, config=cfg)
        return GpuTorchDupireQMC(device="cpu", use_triton=False).simulate_and_price(state, config=cfg)

    d_assets = state.d_assets
    m_rem = state.m_rem

    if m_rem == 0 or bool(getattr(state, "already_autocalled", False)) or state.seasoned_state.already_autocalled:
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
    if not np.all(np.isfinite(corr_np)):
        raise ValueError("Correlation matrix contains non-finite values.")
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

    if d_assets > 1:
        idx_desc = np.argsort(eigvals)[::-1]
        v_eig = np.ascontiguousarray(eigvecs[:, idx_desc], dtype=np.float64)
    else:
        v_eig = np.ones((1, 1), dtype=np.float64)
    u_time = _build_bb_orthogonal_np(tau_obs)

    base_seed = (42 if seed is None else int(seed)) & 0x7FFFFFFF
    eff_mode = "dupire_lv" if mode == "auto" else mode
    n_sub = max(int(n_substeps_per_period), 1)
    dim_coarse = m_rem * d_assets
    dim_fine = (m_rem * n_sub * d_assets) if (eff_mode == "dupire_lv" and n_sub > 1) else 0
    total_dim = dim_coarse + dim_fine

    perf_paths = np.empty((n_total_paths, m_rem, d_assets), dtype=np.float64)
    surfs_ordered = [state.vol_surfaces[ticker] for ticker in state.sorted_identifiers]
    div_events_map = getattr(state.dividend_schedule, "dividend_events", {})
    div_events_ordered = [div_events_map.get(ticker, []) for ticker in state.sorted_identifiers]
    df_fin_obs_np = state.dividend_schedule.df_fin_obs.detach().cpu().numpy()
    r_tot_obs_np = state.dividend_schedule.r_tot_obs.detach().cpu().numpy()
    r_native_obs_np = _resolve_r_native_obs_tensor(state).detach().cpu().numpy().astype(np.float64)
    has_quanto_diff = not bool(np.allclose(r_native_obs_np, r_tot_obs_np, rtol=1e-14, atol=1e-14))
    future_obs_dates = getattr(state.seasoned_state, "future_obs_dates", [])

    # Precompute surface DCC tenors and Act365-to-DCC variance-time scale factors.
    tau_surf_obs = np.empty((d_assets, m_rem), dtype=np.float64)
    for i, surf_i in enumerate(surfs_ordered):
        if surf_i.as_of_date is not None and len(future_obs_dates) == m_rem:
            tau_surf_obs[i, :] = [surf_i.tenor_from_date(dt) for dt in future_obs_dates]
        else:
            tau_surf_obs[i, :] = [surf_i.map_act365_to_surface_tenor(float(t_m)) for t_m in tau_obs]

    inv_lex_perm_raw = getattr(state, "inv_lex_perm", None)
    inv_lex_perm = np.asarray(inv_lex_perm_raw, dtype=np.int64) if inv_lex_perm_raw is not None else None
    raw_opt_idx = state.option_index
    is_long_flag = bool(getattr(state, "is_long_basket", False))
    eff_opt_idx = (
        "AverageOf"
        if (is_long_flag and (raw_opt_idx is None or str(raw_opt_idx).strip() in ("", "WorstOf")))
        else (raw_opt_idx if raw_opt_idx else state.callability_index)
    )
    opt_code = resolve_monitoring_index_code(eff_opt_idx)
    is_diff_opt = opt_code in _DIFF_INDEX_CODES

    ki_obs_raw = getattr(state, "ki_obs_type", None)
    if ki_obs_raw is None and hasattr(state, "seasoned_state"):
        ki_obs_raw = getattr(state.seasoned_state, "ki_obs_type", None)
    ki_mode_raw = str(getattr(ki_obs_raw, "value", ki_obs_raw) or "European").strip()
    ki_mode_low = ki_mode_raw.lower()
    if ki_mode_low in ("european", "terminal", "atmaturity", "maturity"):
        ki_mode = "European"
    elif ki_mode_low in ("discrete", "periodic", "observation"):
        ki_mode = "Discrete"
    elif ki_mode_low == "daily":
        ki_mode = "Daily"
    elif ki_mode_low in ("continuous", "american"):
        ki_mode = "Continuous"
    else:
        ki_mode = "European"

    ki_opt_idx = (
        "WorstOf"
        if is_long_flag
        else (raw_opt_idx if raw_opt_idx else state.callability_index)
    )
    ki_opt_code = resolve_monitoring_index_code(ki_opt_idx)

    has_ki = (
        state.ki_barrier is not None
        and (float(state.ki_barrier) > 1e-6 or (is_diff_opt and float(state.ki_barrier) < -1e-6))
        and float(state.seasoned_state.i_ki_past) < 0.5
    )
    cont_ki_active = bool(has_ki and ki_mode in ("Continuous", "Daily") and state.ki_barrier is not None)
    p_noki_cont = np.ones((eff_r, n_rep), dtype=np.float64) if cont_ki_active else None
    ki_bar_val = float(state.ki_barrier) if state.ki_barrier is not None else 0.0

    # Generate Owen-scrambled Sobol nets for all replicates and apply dyadic BB + PCA rotation
    z_sobol_all = np.empty((eff_r, n_rep, total_dim), dtype=np.float64)
    for rep_idx in range(eff_r):
        rep_seed = (base_seed + rep_idx * 1009) & 0x7FFFFFFF
        z_sobol_all[rep_idx] = _generate_sobol_normals(
            n_rep, total_dim, scramble=scramble, seed=rep_seed
        )
    z_coarse_raw = z_sobol_all[:, :, :dim_coarse].reshape(eff_r, n_rep, m_rem, d_assets)
    z_coarse_indep = np.ascontiguousarray(
        np.einsum("mt,rntd,jd->rmnj", u_time, z_coarse_raw, v_eig),
        dtype=np.float64,
    )
    z_fine_raw = (
        z_sobol_all[:, :, dim_coarse:].reshape(eff_r, n_rep, m_rem, n_sub, d_assets)
        if dim_fine > 0
        else None
    )

    # Precompute cumulative and incremental terminal/GBM variances honoring surface DCC.
    var_cum_gbm = np.empty((d_assets, m_rem), dtype=np.float64)
    for i, surf_i in enumerate(surfs_ordered):
        non_365 = surf_i.dcc.strip().lower() not in ("act365", "act365f", "act_365", "act_365f")
        for m in range(m_rem):
            if non_365:
                iv_i_m = float(surf_i.iv(float(tau_surf_obs[i, m]), float(fwd_prices[i, m])))
                var_cum_gbm[i, m] = (iv_i_m * iv_i_m) * float(tau_surf_obs[i, m])
            else:
                var_cum_gbm[i, m] = (iv_at_fwd[i, m] ** 2) * tau_obs[m]
    d_var_gbm = np.empty((d_assets, m_rem), dtype=np.float64)
    var_run = np.zeros(d_assets, dtype=np.float64)
    for m in range(m_rem):
        dv_col = np.maximum(var_cum_gbm[:, m] - var_run, 1e-12)
        d_var_gbm[:, m] = dv_col
        var_run = var_run + dv_col

    dom_idx = int(np.argmax(var_cum_gbm[:, -1]))
    sqrt_dvar_dom_np = np.sqrt(d_var_gbm[dom_idx, :])
    z_flat_cum = np.zeros((eff_r, n_rep), dtype=np.float64)

    need_gbm_fine = eff_mode == "dupire_lv" and cv_mode == "multifidelity_gbm"
    log_x_gbm = np.zeros((eff_r, n_rep, d_assets), dtype=np.float64) if need_gbm_fine else None
    perf_gbm_fine = np.empty((n_total_paths, m_rem, d_assets), dtype=np.float64) if need_gbm_fine else None

    if eff_mode == "terminal_iv":
        log_x_prev = np.zeros((eff_r, n_rep, d_assets), dtype=np.float64)
        perf_step_prev = np.broadcast_to((spots_t0 / ref_spots)[None, None, :], (eff_r, n_rep, d_assets)).copy()
        t_prev_ti = 0.0
        for m in range(m_rem):
            t_curr_ti = float(tau_obs[m])
            dt_ti = max(t_curr_ti - t_prev_ti, 1e-8)
            z_m = z_coarse_indep[:, m, :, :] @ chol_l.T
            z_flat_cum += sqrt_dvar_dom_np[m] * z_m[:, :, dom_idx]
            d_var = d_var_gbm[:, m]
            std_step = np.sqrt(d_var)
            log_x_curr = log_x_prev - 0.5 * d_var[None, None, :] + std_step[None, None, :] * z_m
            s_m = fwd_prices[:, m][None, None, :] * np.exp(log_x_curr)
            perf_step_curr = s_m / ref_spots[None, None, :]
            if p_noki_cont is not None:
                lv_ti = np.broadcast_to(np.sqrt(d_var / dt_ti)[None, None, :], (eff_r, n_rep, d_assets))
                p_noki_cont *= _eval_continuous_ki_step_survival_np(
                    perf_step_prev,
                    perf_step_curr,
                    corr_sym,
                    lv_ti,
                    dt_ti,
                    ki_opt_code,
                    ki_bar_val,
                    inv_lex_perm,
                    lv_step_end=lv_ti,
                    lv_barrier=lv_ti,
                    bgk_daily=(ki_mode == "Daily"),
                )
            perf_paths[:, m, :] = perf_step_curr.reshape(n_total_paths, d_assets)
            perf_step_prev = perf_step_curr
            log_x_prev = log_x_curr
            t_prev_ti = t_curr_ti
    else:
        log_x = np.zeros((eff_r, n_rep, d_assets), dtype=np.float64)
        perf_sub_prev = (
            np.broadcast_to((spots_t0 / ref_spots)[None, None, :], (eff_r, n_rep, d_assets)).copy()
            if p_noki_cont is not None
            else None
        )
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
            f_curr = fwd_prices[:, m]
            f_prev = spots_t0 if m == 0 else fwd_prices[:, m - 1]
            if has_quanto_diff:
                f_curr_nat = f_curr * np.exp(r_nat_curr - r_tot_curr)
                f_prev_nat = spots_t0 if m == 0 else f_prev * np.exp(r_nat_prev - r_tot_prev)
            else:
                f_curr_nat = f_curr
                f_prev_nat = f_prev
            # Buehler effective native-currency drift: cancels div_yield * |dw_dy| double-subtraction.
            r_zero_m = (
                -np.log(np.maximum(df_fin_curr / np.maximum(df_fin_prev, 1e-15), 1e-15))
                / dt_period
            )
            q_repo_eff = (
                r_zero_m
                - np.log(np.maximum(f_curr_nat / np.maximum(f_prev_nat, 1e-12), 1e-12)) / dt_period
            )

            z_coarse_i = z_coarse_indep[:, m, :, :]
            z_coarse = z_coarse_i @ chol_l.T
            z_flat_cum += sqrt_dvar_dom_np[m] * z_coarse[:, :, dom_idx]
            if log_x_gbm is not None and perf_gbm_fine is not None:
                dv_m = d_var_gbm[:, m]
                log_x_gbm = (
                    log_x_gbm
                    - 0.5 * dv_m[None, None, :]
                    + np.sqrt(dv_m)[None, None, :] * z_coarse
                )
                perf_gbm_fine[:, m, :] = (
                    (fwd_prices[:, m][None, None, :] * np.exp(log_x_gbm)) / ref_spots[None, None, :]
                ).reshape(n_total_paths, d_assets)

            if n_sub == 1 or z_fine_raw is None:
                z_steps = z_coarse[None, :, :, :]
            else:
                inv_sqrt_sub = 1.0 / math.sqrt(float(n_sub))
                zf_m = np.ascontiguousarray(
                    z_fine_raw[:, :, m, :, :].transpose(2, 0, 1, 3), dtype=np.float64
                )
                mean_sub = np.mean(zf_m, axis=0, keepdims=True)
                z_steps_indep = zf_m - mean_sub + z_coarse_i[None, :, :, :] * inv_sqrt_sub
                z_steps = z_steps_indep @ chol_l.T

            log_f_ratio = np.log(np.maximum(f_curr / np.maximum(f_prev, 1e-12), 1e-12))

            lv_step_ki = (
                np.empty((eff_r, n_rep, d_assets), dtype=np.float64)
                if p_noki_cont is not None
                else None
            )
            lv_end_ki = (
                np.empty((eff_r, n_rep, d_assets), dtype=np.float64)
                if p_noki_cont is not None
                else None
            )
            lv_bar_ki = (
                np.empty((1, 1, d_assets), dtype=np.float64)
                if p_noki_cont is not None
                else None
            )

            for s_idx in range(n_sub):
                alpha_s0 = float(s_idx) / float(n_sub)
                alpha_s1 = float(s_idx + 1) / float(n_sub)
                t_s0 = t_prev + alpha_s0 * dt_period
                t_s1 = t_prev + alpha_s1 * dt_period
                if has_div_events:
                    f_s0 = _compute_forward_at_time_np(
                        spots_t0, r_tot_obs_np, tau_obs, div_events_ordered, t_s0, m
                    )
                    f_s1 = _compute_forward_at_time_np(
                        spots_t0, r_tot_obs_np, tau_obs, div_events_ordered, t_s1, m
                    )
                else:
                    f_s0 = f_prev * np.exp(alpha_s0 * log_f_ratio)
                    f_s1 = f_prev * np.exp(alpha_s1 * log_f_ratio)

                if has_quanto_diff:
                    r_tot_s0 = r_tot_prev + alpha_s0 * (r_tot_curr - r_tot_prev)
                    r_nat_s0 = r_nat_prev + alpha_s0 * (r_nat_curr - r_nat_prev)
                    f_s0_native = f_s0 * np.exp(r_nat_s0 - r_tot_s0)
                    r_tot_s1 = r_tot_prev + alpha_s1 * (r_tot_curr - r_tot_prev)
                    r_nat_s1 = r_nat_prev + alpha_s1 * (r_nat_curr - r_nat_prev)
                    f_s1_native = f_s1 * np.exp(r_nat_s1 - r_tot_s1)
                else:
                    f_s0_native = f_s0
                    f_s1_native = f_s1

                for i in range(d_assets):
                    t_surf_prev = 0.0 if m == 0 else float(tau_surf_obs[i, m - 1])
                    t_surf_curr = float(tau_surf_obs[i, m])
                    dt_surf = t_surf_curr - t_surf_prev
                    dt_surf_ratio = dt_surf / dt_period
                    scale_i = math.sqrt(max(dt_surf_ratio, 1e-12))
                    t_surf_s0 = t_surf_prev + alpha_s0 * dt_surf
                    t_surf_s1 = t_surf_prev + alpha_s1 * dt_surf

                    lx_i = log_x[:, :, i]
                    lv0 = np.asarray(
                        surfs_ordered[i].local_vol(
                            t_surf_s0,
                            (f_s0_native[i] * np.exp(lx_i)).reshape(-1),
                            forward_at_t=float(f_s0_native[i]),
                            r_zero=float(r_zero_m[i]),
                            q_repo=float(q_repo_eff[i]),
                            cash_div_leverage=1.0,
                        ),
                        dtype=np.float64,
                    ).reshape(eff_r, n_rep) * scale_i
                    if lv_step_ki is not None:
                        lv_step_ki[:, :, i] = lv0

                    # Componentwise Derivative-Free Platen Weak-Order-2.0 update in log-relative coordinates.
                    mu0 = -0.5 * (lv0 * lv0)
                    y_h = lx_i + mu0 * dt_sub
                    sig_step = lv0 * sqrt_dt_sub
                    lv_p = np.asarray(
                        surfs_ordered[i].local_vol(
                            t_surf_s1,
                            (f_s1_native[i] * np.exp(y_h + sig_step)).reshape(-1),
                            forward_at_t=float(f_s1_native[i]),
                            r_zero=float(r_zero_m[i]),
                            q_repo=float(q_repo_eff[i]),
                            cash_div_leverage=1.0,
                        ),
                        dtype=np.float64,
                    ).reshape(eff_r, n_rep) * scale_i
                    lv_m = np.asarray(
                        surfs_ordered[i].local_vol(
                            t_surf_s1,
                            (f_s1_native[i] * np.exp(y_h - sig_step)).reshape(-1),
                            forward_at_t=float(f_s1_native[i]),
                            r_zero=float(r_zero_m[i]),
                            q_repo=float(q_repo_eff[i]),
                            cash_div_leverage=1.0,
                        ),
                        dtype=np.float64,
                    ).reshape(eff_r, n_rep) * scale_i
                    mu_p = -0.5 * (lv_p * lv_p)
                    mu_m = -0.5 * (lv_m * lv_m)
                    z_i = z_steps[s_idx, :, :, i]
                    log_x[:, :, i] = lx_i + (
                        0.5 * (mu0 + 0.5 * (mu_p + mu_m)) * dt_sub
                        + 0.25 * (mu_p - mu_m) * dt_sub * z_i
                        + 0.25 * (lv_p + lv_m + 2.0 * lv0) * sqrt_dt_sub * z_i
                        + 0.25 * (lv_p - lv_m) * (z_i * z_i - 1.0) * sqrt_dt_sub
                    )
                    if lv_end_ki is not None and lv_bar_ki is not None:
                        lv_end_ki[:, :, i] = (
                            np.asarray(
                                surfs_ordered[i].local_vol(
                                    t_surf_s1,
                                    (f_s1_native[i] * np.exp(log_x[:, :, i])).reshape(-1),
                                    forward_at_t=float(f_s1_native[i]),
                                    r_zero=float(r_zero_m[i]),
                                    q_repo=float(q_repo_eff[i]),
                                    cash_div_leverage=1.0,
                                ),
                                dtype=np.float64,
                            ).reshape(eff_r, n_rep)
                            * scale_i
                        )
                        k_bar_i = max(ki_bar_val * ref_spots[i], 1e-4)
                        lv_bar_ki[0, 0, i] = (
                            float(
                                surfs_ordered[i].local_vol(
                                    0.5 * (t_surf_s0 + t_surf_s1),
                                    k_bar_i,
                                    forward_at_t=0.5 * float(f_s0_native[i] + f_s1_native[i]),
                                    r_zero=float(r_zero_m[i]),
                                    q_repo=float(q_repo_eff[i]),
                                    cash_div_leverage=1.0,
                                )
                            )
                            * scale_i
                        )

                if (
                    p_noki_cont is not None
                    and perf_sub_prev is not None
                    and lv_step_ki is not None
                    and lv_end_ki is not None
                    and lv_bar_ki is not None
                ):
                    perf_sub_curr = (f_s1[None, None, :] * np.exp(log_x)) / ref_spots[None, None, :]
                    p_noki_cont *= _eval_continuous_ki_step_survival_np(
                        perf_sub_prev,
                        perf_sub_curr,
                        corr_sym,
                        lv_step_ki,
                        dt_sub,
                        ki_opt_code,
                        ki_bar_val,
                        inv_lex_perm,
                        lv_step_end=lv_end_ki,
                        lv_barrier=lv_bar_ki,
                        bgk_daily=(ki_mode == "Daily"),
                    )
                    perf_sub_prev = perf_sub_curr

            # Martingale drift enforcement per replicate
            mean_x = np.mean(np.exp(log_x), axis=1, keepdims=True)
            log_x = log_x - np.log(np.maximum(mean_x, 1e-12))

            s_m = f_curr[None, None, :] * np.exp(log_x)
            perf_m_curr = s_m / ref_spots[None, None, :]
            perf_paths[:, m, :] = perf_m_curr.reshape(n_total_paths, d_assets)
            if perf_sub_prev is not None:
                perf_sub_prev = perf_m_curr
            t_prev = t_curr

    w_fund = _compute_funding_weights_np(state)
    ki_ind_override = (1.0 - p_noki_cont).reshape(n_total_paths) if p_noki_cont is not None else None
    (
        total_pv_paths,
        opt_payoff_paths,
        p_stop_arr,
        q_cpn_arr,
        cpn_cf_expected,
        l_put_val,
        l_call_val,
        l_onestar_val,
        p_onestar_val,
    ) = _evaluate_autocall_payoff_np(
        state,
        perf_paths,
        df_pay=df_pay,
        ko_barriers=ko_barriers,
        is_callable=is_callable,
        cpn_barriers=cpn_barriers,
        is_guaranteed_cpn=is_guaranteed_cpn,
        coupon_amounts=coupon_amounts,
        w_fund=w_fund,
        inv_lex_perm=inv_lex_perm,
        ki_ind_override=ki_ind_override,
        ki_mode=ki_mode,
    )

    if eff_r > 1:
        rep_opt = opt_payoff_paths.reshape(eff_r, n_rep).mean(axis=1)
        option_se = float(np.std(rep_opt, ddof=1) / math.sqrt(float(eff_r)))
    else:
        option_se = float(np.std(opt_payoff_paths, ddof=1) / math.sqrt(float(n_total_paths)))

    raw_pv_mean = float(np.mean(total_pv_paths))
    total_pv_mat = total_pv_paths.reshape(eff_r, n_rep)
    if cv_mode == "bs1d_exact" and eff_r > 1:
        v_dom_term = math.sqrt(max(float(np.sum(sqrt_dvar_dom_np * sqrt_dvar_dom_np)), 1e-12))
        f_dom = float(fwd_prices[dom_idx, -1] / max(ref_spots[dom_idx], 1e-12))
        k_cv = float(state.strike) if float(state.strike) > 1e-6 else 1.0
        k_low = (
            float(state.lower_strike)
            if (state.lower_strike is not None and float(state.lower_strike) > 0.0 and float(state.strike) > 1e-6)
            else -1.0
        )
        df_term = float(df_pay[-1])
        mu_cv = df_term * _black76_put_np(f_dom, k_cv, v_dom_term)
        if k_low > 0.0:
            mu_cv -= df_term * _black76_put_np(f_dom, k_low, v_dom_term)
        p_dom = f_dom * np.exp(-0.5 * (v_dom_term * v_dom_term) + z_flat_cum)
        x_cv = df_term * np.maximum(k_cv - p_dom, 0.0)
        if k_low > 0.0:
            x_cv = x_cv - df_term * np.maximum(k_low - p_dom, 0.0)
        cv_pv_est, total_se = _apply_loo_control_variate_np(total_pv_mat, x_cv, mu_cv)
    elif cv_mode == "multifidelity_gbm" and eff_r > 1 and perf_gbm_fine is not None:
        pv_gbm_fine, *_ = _evaluate_autocall_payoff_np(
            state,
            perf_gbm_fine,
            df_pay=df_pay,
            ko_barriers=ko_barriers,
            is_callable=is_callable,
            cpn_barriers=cpn_barriers,
            is_guaranteed_cpn=is_guaranteed_cpn,
            coupon_amounts=coupon_amounts,
            w_fund=w_fund,
            inv_lex_perm=inv_lex_perm,
            ki_mode="Discrete" if ki_mode in ("Continuous", "Daily") else ki_mode,
        )
        mu_mf = _eval_cheap_gbm_expectation_np(
            state,
            chol_l=chol_l,
            d_var_gbm=d_var_gbm,
            fwd_rel=fwd_prices / ref_spots[:, None],
            df_pay=df_pay,
            ko_barriers=ko_barriers,
            is_callable=is_callable,
            cpn_barriers=cpn_barriers,
            is_guaranteed_cpn=is_guaranteed_cpn,
            coupon_amounts=coupon_amounts,
            w_fund=w_fund,
            inv_lex_perm=inv_lex_perm,
            ki_mode=ki_mode,
            n_cheap=(n_cheap_mf_override if n_cheap_mf_override is not None else 4 * n_total_paths),
            seed=base_seed + 88888,
        )
        cv_pv_est, total_se = _apply_loo_control_variate_np(
            total_pv_mat, pv_gbm_fine.reshape(eff_r, n_rep), mu_mf
        )
    else:
        cv_pv_est = raw_pv_mean
        if eff_r > 1:
            rep_pv = np.mean(total_pv_mat, axis=1)
            total_se = float(np.std(rep_pv, ddof=1) / math.sqrt(float(eff_r)))
        else:
            total_se = float(np.std(total_pv_paths, ddof=1) / math.sqrt(float(n_total_paths)))

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

    cv_shift = cv_pv_est - raw_pv_mean
    adj_prod = float(asm.product_value.item()) + cv_shift
    has_onestar_bonus = (
        getattr(state, "one_star_strike", None) is not None
        and float(getattr(state, "one_star_strike", 0.0) or 0.0) > 0.0
        and float(getattr(state, "one_star_coupon", 0.0) or 0.0) != 0.0
    )
    has_zero_opt = (
        float(getattr(state, "option_gearing", 1.0)) == 0.0
        and float(getattr(state, "participation_call_rate", 0.0)) == 0.0
        and not is_long_flag
        and not has_onestar_bonus
    )
    no_notional = bool(getattr(state, "no_notional_repayment", False))
    shift_to_notional = has_zero_opt and not no_notional
    adj_opt = float(asm.option_pv.item()) if shift_to_notional else (float(asm.option_pv.item()) + cv_shift)
    adj_notional = (float(asm.notional_pv.item()) + cv_shift) if shift_to_notional else float(asm.notional_pv.item())
    if cv_shift != 0.0 and asm.cashflows:
        target_order: Tuple[str, ...]
        if is_long_flag:
            target_order = ("IsLongBasket", "Option", "Notional")
        elif shift_to_notional:
            target_order = ("Notional", "Option", "Coupon")
        else:
            target_order = ("Option", "OneStarCoupon", "Coupon", "Notional")
        for target_desc in target_order:
            replaced = False
            for idx_cf in range(len(asm.cashflows) - 1, -1, -1):
                cf = asm.cashflows[idx_cf]
                if cf.Description == target_desc and cf.StatusText == "Estimate":
                    asm.cashflows[idx_cf] = CashFlowItem(
                        Description=cf.Description,
                        Date=cf.Date,
                        Amount=float(cf.Amount + cv_shift),
                        StandardError=float(option_se),
                        Currency=cf.Currency,
                        Leg=cf.Leg,
                        StatusText=cf.StatusText,
                    )
                    replaced = True
                    break
            if replaced:
                break

    return DupireQMCResult(
        product_value=adj_prod,
        standard_error=total_se,
        option_pv=adj_opt,
        option_se=option_se,
        notional_pv=adj_notional,
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
    "ensure_basis_spline_coeffs",
    "DupireQMCResult",
    "DupireQMCOutput",
    "enforce_no_arbitrage_projection",
    "enforce_no_arbitrage_projection_with_jacobian",
    "repair_arbitrage_quotes",
    "price_eq_autocall_dupire_qmc",
    "price_autocall_dupire_qmc",
    "price_eq_autocall_qmc_fallback",
]
