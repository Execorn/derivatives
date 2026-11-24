"""High-precision float64 multi-curve yield, repo interpolation, and day-count engine."""

from __future__ import annotations

import calendar
import datetime
from enum import Enum
from functools import lru_cache
import re
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
    make_interp_spline,
)

from deepvol.api.schemas_eq_autocall import (
    CurveInterpolationMethodEnum,
    DayCountConventionEnum,
    DiscountFactorCurveSpec,
    RepoSpreadCurveSpec,
    _normalize_datetime,
)


# ── Day Count Conventions (IEEE-754 float64) ──────────────────────────────────

_REPO_PERIOD_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([DWMYdwmy])\s*$")


def _to_date_and_frac_days(d: datetime.date | datetime.datetime) -> Tuple[datetime.date, float]:
    """Extracts calendar date and any sub-day fractional seconds as days."""
    if type(d) is datetime.datetime or isinstance(d, datetime.datetime):
        if d.tzinfo is None or d.tzinfo is datetime.timezone.utc:
            if d.hour == 0 and d.minute == 0 and d.second == 0 and d.microsecond == 0:
                return d.date(), 0.0
            sec = d.hour * 3600.0 + d.minute * 60.0 + d.second + d.microsecond * 1e-6
            return d.date(), sec / 86400.0
        dt_utc = d.astimezone(datetime.timezone.utc)
        sec = (
            dt_utc.hour * 3600.0
            + dt_utc.minute * 60.0
            + dt_utc.second
            + dt_utc.microsecond * 1e-6
        )
        return dt_utc.date(), sec / 86400.0
    if isinstance(d, datetime.date):
        return d, 0.0
    raise TypeError(f"Expected datetime.date or datetime.datetime, got {type(d).__name__}")


def _exact_days(
    d1: datetime.date | datetime.datetime,
    d2: datetime.date | datetime.datetime,
) -> float:
    date1, frac1 = _to_date_and_frac_days(d1)
    date2, frac2 = _to_date_and_frac_days(d2)
    return float((date2 - date1).days) + (frac2 - frac1)


def _act_act_isda(
    d1: datetime.date | datetime.datetime,
    d2: datetime.date | datetime.datetime,
) -> float:
    """Exact ISDA Actual/Actual year fraction in float64."""
    date1, frac1 = _to_date_and_frac_days(d1)
    date2, frac2 = _to_date_and_frac_days(d2)
    if date1 == date2 and frac1 == frac2:
        return 0.0
    if (date2, frac2) < (date1, frac1):
        return -_act_act_isda(d2, d1)

    denom1 = 366.0 if calendar.isleap(date1.year) else 365.0
    denom2 = 366.0 if calendar.isleap(date2.year) else 365.0

    if date1.year == date2.year:
        return (float((date2 - date1).days) + (frac2 - frac1)) / denom1

    end_of_y1 = datetime.date(date1.year + 1, 1, 1)
    start_of_y2 = datetime.date(date2.year, 1, 1)

    rem_y1 = (float((end_of_y1 - date1).days) - frac1) / denom1
    full_years = float(date2.year - date1.year - 1)
    elap_y2 = (float((date2 - start_of_y2).days) + frac2) / denom2
    return rem_y1 + full_years + elap_y2


def _thirty_360_us(
    d1: datetime.date | datetime.datetime,
    d2: datetime.date | datetime.datetime,
) -> float:
    """US 30/360 Bond Basis day-count fraction in float64."""
    date1, _ = _to_date_and_frac_days(d1)
    date2, _ = _to_date_and_frac_days(d2)
    day1 = min(date1.day, 30)
    day2 = 30 if (date2.day == 31 and day1 == 30) else date2.day
    days_360 = (
        360.0 * (date2.year - date1.year)
        + 30.0 * (date2.month - date1.month)
        + float(day2 - day1)
    )
    return days_360 / 360.0


def _thirty_e_360(
    d1: datetime.date | datetime.datetime,
    d2: datetime.date | datetime.datetime,
) -> float:
    """European 30E/360 ISDA day-count fraction in float64."""
    date1, _ = _to_date_and_frac_days(d1)
    date2, _ = _to_date_and_frac_days(d2)
    day1 = min(date1.day, 30)
    day2 = min(date2.day, 30)
    days_360 = (
        360.0 * (date2.year - date1.year)
        + 30.0 * (date2.month - date1.month)
        + float(day2 - day1)
    )
    return days_360 / 360.0


@lru_cache(maxsize=64)
def _normalize_dcc_str(dcc_str: str) -> str:
    return dcc_str.strip().upper().replace("/", "").replace("_", "").replace("-", "").replace(".", "")


def _normalize_dcc_token(dcc: str | DayCountConventionEnum) -> str:
    return _normalize_dcc_str(dcc.value if isinstance(dcc, Enum) else str(dcc))


_FIXED_DENOM_DCCS: dict[str, float] = {
    "ACT365": 365.0,
    "ACT365F": 365.0,
    "ACTUAL365": 365.0,
    "ACTUAL365FIXED": 365.0,
    "ACT365FIXED": 365.0,
    "ACT36525": 365.25,
    "ACTUAL36525": 365.25,
    "ACT360": 360.0,
    "ACTUAL360": 360.0,
    "ACT364": 364.0,
    "ACTUAL364": 364.0,
    "ACT366": 366.0,
    "ACT3661": 366.0,
    "ACT3662": 366.0,
    "ACT3663": 366.0,
    "ACTUAL366": 366.0,
    "ACT252": 252.0,
    "ACTUAL252": 252.0,
}


def dcc_denominator(dcc: str | DayCountConventionEnum) -> float:
    """Returns the nominal annual day-count denominator for a DCC string."""
    norm = _normalize_dcc_token(dcc)
    if norm in _FIXED_DENOM_DCCS:
        return _FIXED_DENOM_DCCS[norm]
    if norm in ("BUSINESS252", "BUS252"):
        return 252.0
    if norm in ("THIRTY360", "30360", "30360US", "30U360", "30E360", "THIRTYE360", "EURO30360", "30360ISDA"):
        return 360.0
    if norm in ("ACTACT", "ACTUALACTUAL", "ACTACTISDA", "ACTUALACTUALISDA"):
        return 365.0
    raise ValueError(f"Unsupported day count convention: {dcc!r}")


def year_fraction(
    d1: datetime.date | datetime.datetime,
    d2: datetime.date | datetime.datetime,
    dcc: str | DayCountConventionEnum = "Act365",
) -> float:
    """Computes the exact year fraction tau(d1, d2; dcc) in IEEE-754 float64."""
    norm = _normalize_dcc_token(dcc)
    denom = _FIXED_DENOM_DCCS.get(norm)
    if denom is not None:
        return float(_exact_days(d1, d2) / denom)
    if norm in ("ACTACT", "ACTUALACTUAL", "ACTACTISDA", "ACTUALACTUALISDA"):
        return float(_act_act_isda(d1, d2))
    if norm in ("THIRTY360", "30360", "30360US", "30U360"):
        return float(_thirty_360_us(d1, d2))
    if norm in ("30E360", "THIRTYE360", "EURO30360", "30360ISDA"):
        return float(_thirty_e_360(d1, d2))
    if norm in ("BUSINESS252", "BUS252"):
        date1, frac1 = _to_date_and_frac_days(d1)
        date2, frac2 = _to_date_and_frac_days(d2)
        b_days = float(int(np.busday_count(date1.isoformat(), date2.isoformat())))
        if frac1 != 0.0 or frac2 != 0.0:
            if frac2 != 0.0 and bool(np.is_busday(date2.isoformat())):
                b_days += frac2
            if frac1 != 0.0 and bool(np.is_busday(date1.isoformat())):
                b_days -= frac1
        return float(b_days / 252.0)

    dcc_str = dcc.value if isinstance(dcc, Enum) else str(dcc)
    raise ValueError(f"Unsupported day count convention: {dcc_str!r}")


def year_fractions(
    as_of: datetime.date | datetime.datetime,
    dates: Sequence[datetime.date | datetime.datetime] | npt.NDArray[Any],
    dcc: str | DayCountConventionEnum = "Act365",
) -> npt.NDArray[np.float64]:
    """Vectorized year fraction computation returning a 1D `np.float64` array."""
    norm = _normalize_dcc_token(dcc)
    denom = _FIXED_DENOM_DCCS.get(norm)
    if denom is not None:
        d1_date, d1_frac = _to_date_and_frac_days(as_of)
        out = np.empty(len(dates), dtype=np.float64)
        for i, d2 in enumerate(dates):
            d2_date, d2_frac = _to_date_and_frac_days(d2)
            out[i] = (float((d2_date - d1_date).days) + (d2_frac - d1_frac)) / denom
        return out
    return np.array([year_fraction(as_of, d, norm) for d in dates], dtype=np.float64)


def year_fraction_batch_epoch_sec(
    as_of_sec: float,
    target_secs: npt.NDArray[np.float64] | Sequence[float],
    dcc: str | DayCountConventionEnum = "Act365",
) -> npt.NDArray[np.float64]:
    """Vectorized year fraction from UTC epoch seconds in IEEE-754 float64."""
    targets = np.asarray(target_secs, dtype=np.float64)
    if targets.size == 0:
        return np.zeros(0, dtype=np.float64)
    norm = _normalize_dcc_token(dcc)
    denom = _FIXED_DENOM_DCCS.get(norm)
    if denom is not None:
        return np.asarray((targets - float(as_of_sec)) / (86400.0 * denom), dtype=np.float64)
    as_of_dt = datetime.datetime.fromtimestamp(float(as_of_sec), tz=datetime.timezone.utc)
    target_dts = [
        datetime.datetime.fromtimestamp(float(s), tz=datetime.timezone.utc)
        for s in targets.tolist()
    ]
    return year_fractions(as_of_dt, target_dts, dcc)


# ── Analytical Spline & Hermite Helpers ───────────────────────────────────────


def _extrap_sides(extrapolation: str) -> Tuple[bool, bool]:
    """Returns (left_is_flat, right_is_flat) for Flat/Linear/FlatLeft/FlatRight."""
    ex = extrapolation.strip()
    if ex == "Linear":
        return False, False
    if ex == "FlatLeft":
        return True, False
    if ex == "FlatRight":
        return False, True
    return True, True


def _bessel_hermite_basis(
    tenors: npt.NDArray[np.float64],
    tau_eval: npt.NDArray[np.float64],
    *,
    extrapolation: str = "Flat",
) -> npt.NDArray[np.float64]:
    """Computes the exact Bessel cubic Hermite basis weight matrix W in R^{M_eval x K}."""
    k_len = len(tenors)
    m_eval = len(tau_eval)
    eye_k = np.eye(k_len, dtype=np.float64)
    if k_len == 1:
        return np.ones((m_eval, 1), dtype=np.float64)

    left_flat, right_flat = _extrap_sides(extrapolation)
    dx = np.diff(tenors)
    s_sec = (eye_k[1:, :] - eye_k[:-1, :]) / dx[:, None]

    m_slope = np.zeros((k_len, k_len), dtype=np.float64)
    if k_len == 2:
        m_slope[0, :] = s_sec[0, :]
        m_slope[1, :] = s_sec[0, :]
    elif k_len > 2:
        denom = dx[:-1] + dx[1:]
        m_slope[1:-1, :] = (
            dx[:-1, None] * s_sec[1:, :] + dx[1:, None] * s_sec[:-1, :]
        ) / denom[:, None]
        # Endpoint slopes depend only on interior knots so Flat vs Linear extrapolation does not alter the interior spline.
        m_slope[0, :] = ((2.0 * dx[0] + dx[1]) * s_sec[0, :] - dx[0] * s_sec[1, :]) / (dx[0] + dx[1])
        m_slope[-1, :] = (
            (2.0 * dx[-1] + dx[-2]) * s_sec[-1, :] - dx[-1] * s_sec[-2, :]
        ) / (dx[-1] + dx[-2])

    w_out = np.zeros((m_eval, k_len), dtype=np.float64)
    t_min = tenors[0]
    t_max = tenors[-1]

    for i, t_raw in enumerate(tau_eval):
        if t_raw <= t_min:
            if left_flat:
                w_out[i, 0] = 1.0
            else:
                w_out[i, :] = eye_k[0, :] + (t_raw - t_min) * m_slope[0, :]
        elif t_raw >= t_max:
            if right_flat:
                w_out[i, -1] = 1.0
            else:
                w_out[i, :] = eye_k[-1, :] + (t_raw - t_max) * m_slope[-1, :]
        else:
            idx = int(np.searchsorted(tenors, t_raw, side="right")) - 1
            idx = max(0, min(idx, k_len - 2))
            h = tenors[idx + 1] - tenors[idx]
            u = (t_raw - tenors[idx]) / h
            h00 = (1.0 + 2.0 * u) * ((1.0 - u) ** 2)
            h10 = u * ((1.0 - u) ** 2)
            h01 = (u ** 2) * (3.0 - 2.0 * u)
            h11 = (u ** 2) * (u - 1.0)
            w_out[i, :] = (
                h00 * eye_k[idx, :]
                + h10 * h * m_slope[idx, :]
                + h01 * eye_k[idx + 1, :]
                + h11 * h * m_slope[idx + 1, :]
            )
    return w_out


def _linear_basis(
    tenors: npt.NDArray[np.float64],
    tau_eval: npt.NDArray[np.float64],
    *,
    extrapolation: str = "Flat",
) -> npt.NDArray[np.float64]:
    """Computes the exact piecewise-linear basis weight matrix W in R^{M_eval x K}."""
    k_len = len(tenors)
    m_eval = len(tau_eval)
    w_out = np.zeros((m_eval, k_len), dtype=np.float64)
    if k_len == 1:
        w_out[:, 0] = 1.0
        return w_out

    left_flat, right_flat = _extrap_sides(extrapolation)
    t_min = tenors[0]
    t_max = tenors[-1]
    for i, t_raw in enumerate(tau_eval):
        if t_raw <= t_min:
            if left_flat or k_len < 2:
                w_out[i, 0] = 1.0
            else:
                dt = tenors[1] - tenors[0]
                alpha = (t_raw - tenors[0]) / dt
                w_out[i, 0] = 1.0 - alpha
                w_out[i, 1] = alpha
        elif t_raw >= t_max:
            if right_flat or k_len < 2:
                w_out[i, -1] = 1.0
            else:
                dt = tenors[-1] - tenors[-2]
                alpha = (t_raw - tenors[-2]) / dt
                w_out[i, -2] = 1.0 - alpha
                w_out[i, -1] = alpha
        else:
            idx = int(np.searchsorted(tenors, t_raw, side="right")) - 1
            idx = max(0, min(idx, k_len - 2))
            dt = tenors[idx + 1] - tenors[idx]
            alpha = (t_raw - tenors[idx]) / dt
            w_out[i, idx] = 1.0 - alpha
            w_out[i, idx + 1] = alpha
    return w_out


def _log_linear_rate_basis(
    tenors: npt.NDArray[np.float64],
    tau_eval: npt.NDArray[np.float64],
    *,
    extrapolation: str = "Flat",
    space_coords: str = "Rates",
) -> npt.NDArray[np.float64]:
    """Computes the basis weight matrix W in R^{M_eval x K} for LogLinear / PiecewiseConstantForward."""
    if space_coords == "LogarithmOfDiscountFactors":
        return _linear_basis(tenors, tau_eval, extrapolation=extrapolation)
    k_len = len(tenors)
    m_eval = len(tau_eval)
    w_out = np.zeros((m_eval, k_len), dtype=np.float64)
    if k_len == 1:
        w_out[:, 0] = 1.0
        return w_out

    left_flat, right_flat = _extrap_sides(extrapolation)
    t_min = tenors[0]
    t_max = tenors[-1]
    for i, t_raw in enumerate(tau_eval):
        if t_raw <= t_min:
            if left_flat or k_len < 2:
                w_out[i, 0] = 1.0
            else:
                dt = tenors[1] - tenors[0]
                alpha = (t_raw - tenors[0]) / dt
                w_out[i, 0] = 1.0 - alpha
                w_out[i, 1] = alpha
        elif t_raw >= t_max:
            if right_flat or k_len < 2:
                w_out[i, -1] = 1.0
            else:
                dt = tenors[-1] - tenors[-2]
                alpha = (t_raw - tenors[-2]) / dt
                w_out[i, -2] = 1.0 - alpha
                w_out[i, -1] = alpha
        else:
            idx = int(np.searchsorted(tenors, t_raw, side="right")) - 1
            idx = max(0, min(idx, k_len - 2))
            t_l = tenors[idx]
            t_r = tenors[idx + 1]
            alpha = (t_raw - t_l) / (t_r - t_l)
            if abs(t_raw) < 1e-12:
                w_out[i, idx] = 1.0 - alpha
                w_out[i, idx + 1] = alpha
            else:
                w_out[i, idx] = (1.0 - alpha) * (t_l / t_raw)
                w_out[i, idx + 1] = alpha * (t_r / t_raw)
    return w_out


def _fwd_fwd_linear_basis(
    tenors: npt.NDArray[np.float64],
    tau_eval: npt.NDArray[np.float64],
    *,
    extrapolation: str = "Flat",
    space_coords: str = "Rates",
) -> npt.NDArray[np.float64]:
    """Computes the exact basis weight matrix W for linear forward-rate interpolation preserving pillar DFs."""
    k_len = len(tenors)
    m_eval = len(tau_eval)
    if k_len == 1:
        return np.ones((m_eval, 1), dtype=np.float64)

    # Map pillar zero rates r_k -> average interval forwards f_bar -> knot instantaneous forwards f_k
    eye_k = np.eye(k_len, dtype=np.float64)
    f_bar = np.zeros((k_len, k_len), dtype=np.float64)
    if tenors[0] > 1e-12:
        f_bar[0, 0] = 1.0
    else:
        f_bar[0, 1] = 1.0
    for k in range(1, k_len):
        dt_k = tenors[k] - tenors[k - 1]
        f_bar[k, k] = tenors[k] / dt_k
        f_bar[k, k - 1] = -tenors[k - 1] / dt_k

    f_knot = np.zeros((k_len, k_len), dtype=np.float64)
    f_knot[0, :] = f_bar[0, :]
    for k in range(1, k_len):
        f_knot[k, :] = 2.0 * f_bar[k, :] - f_knot[k - 1, :]

    left_flat, right_flat = _extrap_sides(extrapolation)
    w_out = np.zeros((m_eval, k_len), dtype=np.float64)
    t_min = tenors[0]
    t_max = tenors[-1]

    def _eval_interior_weights(tau_val: float) -> npt.NDArray[np.float64]:
        if tau_val <= t_min:
            return eye_k[0, :]
        idx = int(np.searchsorted(tenors, tau_val, side="right")) - 1
        idx = max(0, min(idx, k_len - 2))
        t_l = tenors[idx]
        t_r = tenors[idx + 1]
        h = t_r - t_l
        dx = tau_val - t_l
        # Integral of linear f(u) = f_l + (f_{l+1} - f_l) * (u - t_l)/h from t_l to tau_val:
        slope_frac = 0.5 * (dx / h)
        int_seg = dx * ((1.0 - slope_frac) * f_knot[idx, :] + slope_frac * f_knot[idx + 1, :])
        total_int = t_l * eye_k[idx, :] + int_seg
        if abs(tau_val) < 1e-12:
            return np.asarray(f_knot[0, :], dtype=np.float64)
        return np.asarray(total_int / tau_val, dtype=np.float64)

    inv_tenors = np.where(tenors > 1e-12, 1.0 / np.maximum(tenors, 1e-12), 0.0)
    for i, t_raw in enumerate(tau_eval):
        if t_raw <= t_min:
            if left_flat:
                w_out[i, :] = eye_k[0, :]
            else:
                slope_l = (eye_k[1, :] - eye_k[0, :]) / (tenors[1] - tenors[0])
                w_out[i, :] = eye_k[0, :] + (t_raw - t_min) * slope_l
        elif t_raw >= t_max:
            if right_flat:
                w_out[i, :] = eye_k[-1, :]
            else:
                slope_r = (eye_k[-1, :] - eye_k[-2, :]) / (tenors[-1] - tenors[-2])
                w_out[i, :] = eye_k[-1, :] + (t_raw - t_max) * slope_r
        else:
            w_r = _eval_interior_weights(float(t_raw))
            if space_coords == "LogarithmOfDiscountFactors":
                w_out[i, :] = float(t_raw) * w_r * inv_tenors
            else:
                w_out[i, :] = w_r
    return w_out


def _monotone_convex_eval(
    tenors: npt.NDArray[np.float64],
    y_pillars: npt.NDArray[np.float64],
    tau_eval: npt.NDArray[np.float64],
    *,
    space_coords: str = "Rates",
    extrapolation: str = "Flat",
) -> npt.NDArray[np.float64]:
    """Hagan-West (2006) Monotone Convex forward-rate curve evaluation in float64."""
    k_len = len(tenors)
    m_eval = len(tau_eval)
    if k_len == 1:
        return np.full(m_eval, float(y_pillars[0]), dtype=np.float64)

    # Convert pillars to zero rates r_k for forward-rate integration
    if space_coords == "Rates":
        r_pil = y_pillars
    elif space_coords == "LogarithmOfDiscountFactors":
        t_safe = np.maximum(np.abs(tenors), 1e-12)
        r_pil = np.where(tenors > 1e-12, -y_pillars / t_safe, 0.0)
    else:
        t_safe = np.maximum(np.abs(tenors), 1e-12)
        r_pil = np.where(tenors > 1e-12, -np.log(np.maximum(y_pillars, 1e-15)) / t_safe, 0.0)

    if tenors[0] > 1e-12:
        t_hw = np.concatenate([[0.0], tenors])
        rt_hw = np.concatenate([[0.0], r_pil * tenors])
    else:
        t_hw = tenors
        rt_hw = r_pil * tenors
    n_seg = len(t_hw) - 1
    dt_hw = np.diff(t_hw)
    f_d = np.diff(rt_hw) / np.maximum(dt_hw, 1e-12)  # length n_seg, 0-based

    # Estimate knot forwards f_knot[0..n_seg] and apply Hagan-West §3.3 positivity bounds
    f_knot = np.zeros(n_seg + 1, dtype=np.float64)
    for i in range(1, n_seg):
        denom = dt_hw[i - 1] + dt_hw[i]
        f_knot[i] = (dt_hw[i - 1] * f_d[i] + dt_hw[i] * f_d[i - 1]) / denom
    if n_seg >= 2:
        f_knot[0] = f_d[0] - 0.5 * (f_knot[1] - f_d[0])
        f_knot[n_seg] = f_d[n_seg - 1] - 0.5 * (f_knot[n_seg - 1] - f_d[n_seg - 1])
    else:
        f_knot[0] = f_d[0]
        f_knot[1] = f_d[0]

    if f_d[0] > 0.0:
        f_knot[0] = min(max(f_knot[0], 0.0), 2.0 * f_d[0])
    for i in range(1, n_seg):
        if f_d[i - 1] > 0.0 and f_d[i] > 0.0:
            f_knot[i] = min(max(f_knot[i], 0.0), 2.0 * min(f_d[i - 1], f_d[i]))
    if f_d[n_seg - 1] > 0.0:
        f_knot[n_seg] = min(max(f_knot[n_seg], 0.0), 2.0 * f_d[n_seg - 1])

    def _hw_segment_integral(i_seg: int, x: float) -> float:
        """Evaluates int_0^x (f(u) - f_d[i_seg]) du on unit interval [0, 1] for Hagan-West §3.2."""
        g0 = f_knot[i_seg] - f_d[i_seg]
        g1 = f_knot[i_seg + 1] - f_d[i_seg]
        if abs(g0) < 1e-15 and abs(g1) < 1e-15:
            return 0.0

        # Case (i): opposite signs with ratio in [0.5, 2.0]
        if abs(g0) <= 1e-15 and abs(g1) <= 1e-15:
            return 0.0
        # Case (i): single cubic polynomial (covers |g1| in [0.5|g0|, 2|g0|] and g0==0 or g1==0 limits)
        if (
            (g0 > 0.0 and -2.0 * g0 <= g1 <= -0.5 * g0)
            or (g0 < 0.0 and -2.0 * g0 >= g1 >= -0.5 * g0)
            or abs(g0) <= 1e-15
            or abs(g1) <= 1e-15
        ):
            return float(g0 * (x - 2.0 * x * x + x * x * x) + g1 * (-x * x + x * x * x))

        # Case (ii): opposite signs with |g1| > 2*|g0|
        if (g0 < 0.0 and g1 > -2.0 * g0) or (g0 > 0.0 and g1 < -2.0 * g0):
            eta = 1.0 + (3.0 * g0) / (g1 - g0)
            eta = min(max(eta, 0.0), 1.0)
            if x <= eta or eta >= 1.0 - 1e-15:
                return float(g0 * x)
            u = (x - eta) / (1.0 - eta)
            return float(g0 * x + (g1 - g0) * (1.0 - eta) * (u * u * u) / 3.0)

        # Case (iii): opposite signs with |g0| > 2*|g1|
        if (g0 > 0.0 and -0.5 * g0 < g1 < 0.0) or (g0 < 0.0 and 0.0 < g1 < -0.5 * g0):
            eta = (3.0 * g1) / (g1 - g0)
            eta = min(max(eta, 0.0), 1.0)
            if eta <= 1e-15:
                return float(g1 * x)
            if x <= eta:
                return float(g1 * x + (g0 - g1) * eta * (1.0 - ((1.0 - x / eta) ** 3)) / 3.0)
            return float(g1 * x + (g0 - g1) * eta / 3.0)

        # Case (iv): strictly same-sign (g0 * g1 > 0) two-piece quadratic preserving G(0) = G(1) = 0
        eta = g1 / (g1 + g0)
        a_val = -(g0 * g1) / (g0 + g1)
        if x <= eta:
            return float(a_val * x + ((g0 - a_val) / (3.0 * eta * eta)) * ((x - eta) ** 3 + eta ** 3))
        return float(
            a_val * x
            + ((g0 - a_val) / 3.0) * eta
            + ((g1 - a_val) / (3.0 * (1.0 - eta) ** 2)) * ((x - eta) ** 3)
        )

    def _eval_zero_rate_at(tau_val: float) -> float:
        if tau_val <= t_hw[0] + 1e-12:
            return float(f_knot[0])
        if tau_val >= t_hw[-1]:
            if abs(t_hw[-1]) < 1e-12:
                return float(f_knot[-1])
            return float(rt_hw[-1] / t_hw[-1])
        idx = int(np.searchsorted(t_hw, tau_val, side="right")) - 1
        idx = max(0, min(idx, n_seg - 1))
        if abs(tau_val) < 1e-12:
            return float(f_knot[idx]) if abs(t_hw[idx]) < 1e-12 else float(f_knot[0])
        h = dt_hw[idx]
        x = (tau_val - t_hw[idx]) / max(h, 1e-12)
        rt_val = rt_hw[idx] + h * (f_d[idx] * x + _hw_segment_integral(idx, x))
        return float(rt_val / tau_val)

    def _zero_to_space(tau_val: float, r_val: float) -> float:
        if space_coords == "Rates":
            return r_val
        if space_coords == "LogarithmOfDiscountFactors":
            return -r_val * tau_val
        return float(np.exp(-r_val * tau_val))

    left_flat, right_flat = _extrap_sides(extrapolation)
    t_min = float(tenors[0])
    t_max = float(tenors[-1])
    y_min = float(y_pillars[0])
    y_max = float(y_pillars[-1])
    slope_l = float((y_pillars[1] - y_pillars[0]) / (tenors[1] - tenors[0]))
    slope_r = float((y_pillars[-1] - y_pillars[-2]) / (tenors[-1] - tenors[-2]))

    out = np.empty(m_eval, dtype=np.float64)
    for i, t_raw in enumerate(tau_eval):
        if t_raw <= t_min:
            out[i] = y_min if left_flat else (y_min + slope_l * (t_raw - t_min))
        elif t_raw >= t_max:
            out[i] = y_max if right_flat else (y_max + slope_r * (t_raw - t_max))
        else:
            r_val = _eval_zero_rate_at(float(t_raw))
            out[i] = _zero_to_space(float(t_raw), r_val)
    return out


def _nelson_siegel_basis(
    tenors: npt.NDArray[np.float64],
    tau_eval: npt.NDArray[np.float64],
    *,
    svensson: bool = False,
    extrapolation: str = "Flat",
) -> npt.NDArray[np.float64]:
    """Computes the exact linear least-squares Nelson-Siegel / Svensson projection basis W in R^{M_eval x K}."""
    k_len = len(tenors)
    left_flat, right_flat = _extrap_sides(extrapolation)
    t_min = float(tenors[0])
    t_max = float(tenors[-1])
    t_eval = tau_eval.copy()
    if left_flat:
        t_eval = np.maximum(t_eval, t_min)
    if right_flat:
        t_eval = np.minimum(t_eval, t_max)

    def _ns_factors(u: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        small = u < 1e-5
        u_safe = np.where(small, 1.0, u)
        expm1_u = np.expm1(-u_safe)
        f1_large = -expm1_u / u_safe
        f2_large = f1_large - (1.0 + expm1_u)
        f1_small = 1.0 - 0.5 * u + (u * u) / 6.0 - (u * u * u) / 24.0
        f2_small = 0.5 * u - (u * u) / 3.0 + (u * u * u) / 8.0 - (u * u * u * u) / 30.0
        return np.where(small, f1_small, f1_large), np.where(small, f2_small, f2_large)

    def _design(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        t_pos = np.maximum(t, 0.0)
        f0 = np.ones_like(t_pos)
        f1, f2 = _ns_factors(t_pos / 1.5)
        if not svensson or k_len < 4:
            return np.column_stack([f0, f1, f2])[:, : min(3, k_len)]
        _, f3 = _ns_factors(t_pos / 0.5)
        return np.column_stack([f0, f1, f2, f3])[:, : min(4, k_len)]

    x_pil = _design(tenors)
    x_eval = _design(t_eval)
    pinv = np.linalg.pinv(x_pil)
    return np.asarray(x_eval @ pinv, dtype=np.float64)


def fast_clamped_cubic_coeffs(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Computes clamped cubic spline polynomial coefficients of shape `(4, K - 1, ...)` in pure NumPy."""
    k = len(x)
    if k < 2:
        raise ValueError("fast_clamped_cubic_coeffs requires at least 2 knots.")
    h = np.diff(x)
    h_shape = (k - 1,) + (1,) * (y.ndim - 1)
    h_b = h.reshape(h_shape)
    s = (y[1:] - y[:-1]) / h_b
    d = np.zeros_like(y, dtype=np.float64)
    n_int = k - 2
    if n_int > 0:
        h_prev = h[:-1]
        h_next = h[1:]
        diag = 2.0 * (h_prev + h_next)
        int_shape = (n_int,) + (1,) * (y.ndim - 1)
        rhs = 3.0 * (h_next.reshape(int_shape) * s[:-1] + h_prev.reshape(int_shape) * s[1:])
        c_star = np.empty(n_int, dtype=np.float64)
        d_star = np.empty_like(rhs)
        c_star[0] = h_prev[0] / diag[0] if n_int > 1 else 0.0
        d_star[0] = rhs[0] / diag[0]
        for i in range(1, n_int):
            m_piv = diag[i] - h_next[i] * c_star[i - 1]
            if i < n_int - 1:
                c_star[i] = h_prev[i] / m_piv
            d_star[i] = (rhs[i] - h_next[i] * d_star[i - 1]) / m_piv
        d[k - 2] = d_star[n_int - 1]
        for i in range(n_int - 2, -1, -1):
            d[i + 1] = d_star[i] - c_star[i] * d[i + 2]
    c3 = y[:-1]
    c2 = d[:-1]
    c1 = (3.0 * s - 2.0 * d[:-1] - d[1:]) / h_b
    c0 = (d[:-1] + d[1:] - 2.0 * s) / (h_b * h_b)
    return np.stack([c0, c1, c2, c3], axis=0)


def eval_clamped_cubic_coeffs(
    x_knots: npt.NDArray[np.float64],
    coeffs: npt.NDArray[np.float64],
    x_eval: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluates piecewise cubic polynomial with coefficients `(4, K - 1, ...)` on clamped `x_eval`."""
    k = len(x_knots)
    t_clip = np.clip(x_eval, x_knots[0], x_knots[-1])
    idx = np.clip(np.searchsorted(x_knots, t_clip, side="right") - 1, 0, k - 2)
    dx = t_clip - x_knots[idx]
    if coeffs.ndim > 2:
        dx_b = dx.reshape((len(dx),) + (1,) * (coeffs.ndim - 2))
    else:
        dx_b = dx
    c0 = coeffs[0, idx]
    c1 = coeffs[1, idx]
    c2 = coeffs[2, idx]
    c3 = coeffs[3, idx]
    return np.asarray(((c0 * dx_b + c1) * dx_b + c2) * dx_b + c3, dtype=np.float64)


def _extract_curve_field(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return default


def _convert_dates_or_tenors(
    dates_or_tenors: Any,
    *,
    as_of_date: datetime.date | datetime.datetime | None,
    dcc: str,
    caller_name: str,
) -> Tuple[npt.NDArray[np.float64], bool]:
    if isinstance(dates_or_tenors, np.ndarray):
        if dates_or_tenors.ndim == 0:
            return _convert_dates_or_tenors(
                dates_or_tenors.item(),
                as_of_date=as_of_date,
                dcc=dcc,
                caller_name=caller_name,
            )
        if dates_or_tenors.ndim == 1 and np.issubdtype(dates_or_tenors.dtype, np.number):
            return dates_or_tenors.astype(np.float64, copy=False), False
    if isinstance(dates_or_tenors, (datetime.date, datetime.datetime)):
        if as_of_date is None:
            raise ValueError(f"{caller_name} requires `as_of_date` to evaluate date inputs.")
        t = year_fraction(as_of_date, dates_or_tenors, dcc)
        return np.array([t], dtype=np.float64), True
    if isinstance(dates_or_tenors, str):
        if as_of_date is None:
            raise ValueError(f"{caller_name} requires `as_of_date` to evaluate date string inputs.")
        t = year_fraction(as_of_date, _normalize_datetime(dates_or_tenors), dcc)
        return np.array([t], dtype=np.float64), True
    if isinstance(dates_or_tenors, (int, float, np.floating, np.integer)):
        return np.array([float(dates_or_tenors)], dtype=np.float64), True

    seq = list(dates_or_tenors)
    if len(seq) == 0:
        return np.zeros(0, dtype=np.float64), False
    if isinstance(seq[0], (datetime.date, datetime.datetime, str)):
        if as_of_date is None:
            raise ValueError(f"{caller_name} requires `as_of_date` to evaluate date inputs.")
        dt_seq = [_normalize_datetime(d) if isinstance(d, str) else d for d in seq]
        return year_fractions(as_of_date, dt_seq, dcc), False
    return np.asarray(seq, dtype=np.float64), False


# ── ClampedYieldCurve ─────────────────────────────────────────────────────────


class ClampedYieldCurve:
    """High-precision float64 yield/discount curve engine with immutable state and spline basis weights."""

    def __init__(
        self,
        spec: DiscountFactorCurveSpec | Mapping[str, Any] | datetime.date | datetime.datetime | None = None,
        pillar_dates: Sequence[datetime.date | datetime.datetime] | None = None,
        rates: Sequence[float] | npt.NDArray[np.float64] | None = None,
        *,
        as_of_date: datetime.date | datetime.datetime | None = None,
        pillar_tenors: Sequence[float] | npt.NDArray[np.float64] | None = None,
        spreads: Sequence[float] | npt.NDArray[np.float64] | None = None,
        dcc: str | DayCountConventionEnum = "Act365",
        interpolation_method: str | CurveInterpolationMethodEnum = "CubicSpline",
        space_coords: str = "Rates",
        extrapolation_method: str = "Flat",
        quote_expression: str = "Percentage",
        input_mode: str | None = None,
        identifier: str = "DEFAULT",
        currency: str = "RUB",
    ) -> None:
        if isinstance(spec, DiscountFactorCurveSpec):
            self.identifier = spec.Identifier
            self.currency = spec.Currency
            self.as_of_date: datetime.date | datetime.datetime | None = spec.AsOfDate
            self.dcc = spec.InterpolationDayCountConvention.value
            self.interpolation_method = spec.InterpolationMethod.value
            self.space_coords = spec.InterpolationSpaceCoordinates.value
            self.extrapolation_method = spec.ExtrapolationMethod.value
            self.quote_expression = spec.RateQuoteExpression.value
            self.pillar_dates: list[datetime.date | datetime.datetime] = list(
                spec.RatesSchedule.Dates
            )
            tenors_arr = year_fractions(self.as_of_date, self.pillar_dates, self.dcc)
            raw_rates = (
                np.asarray(spec.RatesSchedule.Rates, dtype=np.float64)
                if len(spec.RatesSchedule.Rates) > 0
                else np.zeros(len(self.pillar_dates), dtype=np.float64)
            )
            if spec.RatesSchedule.Spreads is not None and len(spec.RatesSchedule.Spreads) > 0:
                raw_rates = raw_rates + np.asarray(spec.RatesSchedule.Spreads, dtype=np.float64)
        elif isinstance(spec, Mapping):
            self.identifier = str(_extract_curve_field(spec, "Identifier", "identifier", default=identifier))
            self.currency = str(_extract_curve_field(spec, "Currency", "currency", default=currency))
            raw_as_of = _extract_curve_field(spec, "AsOfDate", "as_of_date", default=as_of_date)
            self.as_of_date = _normalize_datetime(raw_as_of) if raw_as_of is not None else None
            raw_dcc = _extract_curve_field(
                spec, "InterpolationDayCountConvention", "DayCountConvention", "dcc", "day_count_convention", default=dcc
            )
            self.dcc = str(getattr(raw_dcc, "value", raw_dcc))
            raw_interp = _extract_curve_field(
                spec, "InterpolationMethod", "interpolation_method", default=interpolation_method
            )
            self.interpolation_method = str(getattr(raw_interp, "value", raw_interp))
            raw_space = _extract_curve_field(
                spec, "InterpolationSpaceCoordinates", "space_coords", default=space_coords
            )
            self.space_coords = str(getattr(raw_space, "value", raw_space))
            raw_extrap = _extract_curve_field(
                spec, "ExtrapolationMethod", "extrapolation_method", default=extrapolation_method
            )
            self.extrapolation_method = str(getattr(raw_extrap, "value", raw_extrap))
            raw_qe = _extract_curve_field(
                spec, "RateQuoteExpression", "rate_quote_expression", "quote_expression", default=quote_expression
            )
            self.quote_expression = str(getattr(raw_qe, "value", raw_qe))

            default_dates = pillar_dates if pillar_dates is not None else []
            sched = _extract_curve_field(spec, "RatesSchedule", "rates_schedule", default=spec)
            raw_dates = _extract_curve_field(sched, "Dates", "dates", default=default_dates)
            raw_tenors = _extract_curve_field(sched, "Tenors", "tenors", "pillar_tenors", default=pillar_tenors)
            raw_r = _extract_curve_field(sched, "Rates", "rates", "rates_pct", default=rates)
            raw_sp = _extract_curve_field(sched, "Spreads", "spreads", default=spreads)
            raw_dfs = _extract_curve_field(sched, "DiscountFactors", "discount_factors", default=None)

            self.pillar_dates = (
                [_normalize_datetime(d) for d in raw_dates]
                if (raw_dates is not None and len(raw_dates) > 0)
                else []
            )
            if raw_tenors is not None:
                tenors_arr = np.asarray(raw_tenors, dtype=np.float64)
            elif len(self.pillar_dates) > 0 and self.as_of_date is not None:
                tenors_arr = year_fractions(self.as_of_date, self.pillar_dates, self.dcc)
            else:
                raise ValueError(
                    "ClampedYieldCurve dict spec requires Dates + AsOfDate or pillar_tenors."
                )
            if raw_r is None and raw_sp is None and raw_dfs is None:
                raise ValueError("ClampedYieldCurve dict spec requires Rates, Spreads, or DiscountFactors.")
            if (raw_r is None or len(raw_r) == 0) and raw_dfs is not None:
                raw_rates = np.asarray(raw_dfs, dtype=np.float64)
                if input_mode is None:
                    input_mode = "df"
                if self.quote_expression == "Percentage":
                    self.quote_expression = "Standard"
            else:
                raw_rates = (
                    np.asarray(raw_r, dtype=np.float64)
                    if (raw_r is not None and len(raw_r) > 0)
                    else np.zeros(len(tenors_arr), dtype=np.float64)
                )
            if raw_sp is not None and len(raw_sp) > 0:
                raw_rates = raw_rates + np.asarray(raw_sp, dtype=np.float64)
        else:
            if isinstance(spec, (datetime.date, datetime.datetime)) and as_of_date is None:
                as_of_date = spec
            self.identifier = identifier
            self.currency = currency
            self.as_of_date = _normalize_datetime(as_of_date) if as_of_date is not None else None
            self.dcc = dcc.value if isinstance(dcc, Enum) else str(dcc)
            self.interpolation_method = (
                interpolation_method.value
                if isinstance(interpolation_method, Enum)
                else str(interpolation_method)
            )
            self.space_coords = (
                space_coords.value if isinstance(space_coords, Enum) else str(space_coords)
            )
            self.extrapolation_method = (
                extrapolation_method.value
                if isinstance(extrapolation_method, Enum)
                else str(extrapolation_method)
            )
            self.quote_expression = (
                quote_expression.value
                if isinstance(quote_expression, Enum)
                else str(quote_expression)
            )
            self.pillar_dates = list(pillar_dates) if pillar_dates is not None else []
            if pillar_tenors is not None:
                tenors_arr = np.asarray(pillar_tenors, dtype=np.float64)
            elif len(self.pillar_dates) > 0 and self.as_of_date is not None:
                tenors_arr = year_fractions(self.as_of_date, self.pillar_dates, self.dcc)
            else:
                raise ValueError(
                    "ClampedYieldCurve requires either `spec`, `pillar_tenors`, or (`as_of_date` and `pillar_dates`)."
                )
            if rates is None and spreads is None:
                raise ValueError("ClampedYieldCurve requires `rates` or `spreads`.")
            raw_rates = (
                np.asarray(rates, dtype=np.float64)
                if (rates is not None and len(rates) > 0)
                else np.zeros(len(tenors_arr), dtype=np.float64)
            )
            if spreads is not None and len(spreads) > 0:
                raw_rates = raw_rates + np.asarray(spreads, dtype=np.float64)

        if len(tenors_arr) == 0:
            raise ValueError("ClampedYieldCurve requires at least one pillar.")
        if len(tenors_arr) != len(raw_rates):
            raise ValueError(
                f"Pillar tenors length ({len(tenors_arr)}) != rates length ({len(raw_rates)})"
            )
        if not np.all(np.isfinite(tenors_arr)):
            raise ValueError("ClampedYieldCurve pillar tenors must be finite.")
        if not np.all(np.isfinite(raw_rates)):
            raise ValueError("ClampedYieldCurve rates/quotes must be finite.")
        if len(tenors_arr) > 1 and np.any(np.diff(tenors_arr) <= 0.0):
            raise ValueError("ClampedYieldCurve pillar tenors must be strictly increasing.")

        self._tenors: npt.NDArray[np.float64] = np.array(tenors_arr, dtype=np.float64, copy=True)
        self._raw_quotes: npt.NDArray[np.float64] = np.array(raw_rates, dtype=np.float64, copy=True)
        self._quote_scale: float = 0.01 if self.quote_expression == "Percentage" else 1.0

        # Resolve quote semantics cleanly without value-range heuristics
        if input_mode is not None:
            mode_norm = input_mode.strip().lower()
            if mode_norm not in ("rates", "log_df", "df"):
                raise ValueError(f"Unsupported input_mode: {input_mode!r}")
            self._input_mode: str = mode_norm
        elif self.space_coords == "Rates":
            self._input_mode = "rates"
        elif self.space_coords == "LogarithmOfDiscountFactors":
            self._input_mode = "rates" if self.quote_expression == "Percentage" else "log_df"
        elif self.space_coords in ("DiscountFactors", "Factors"):
            self._input_mode = "rates" if self.quote_expression == "Percentage" else "df"
        else:
            raise ValueError(f"Unsupported InterpolationSpaceCoordinates: {self.space_coords!r}")

        t_safe = np.maximum(np.abs(self._tenors), 1e-12)
        if self._input_mode == "rates":
            rates_dec = self._raw_quotes * self._quote_scale
        elif self._input_mode == "log_df":
            rates_dec = np.where(
                self._tenors > 1e-12, -(self._raw_quotes * self._quote_scale) / t_safe, 0.0
            )
        else:
            df_pil = np.maximum(self._raw_quotes * self._quote_scale, 1e-15)
            rates_dec = np.where(
                self._tenors > 1e-12, -np.log(df_pil) / t_safe, 0.0
            )
        self._rates_decimal: npt.NDArray[np.float64] = np.asarray(rates_dec, dtype=np.float64)

        # Interpolate natively in the requested InterpolationSpaceCoordinates
        if self.space_coords == "Rates":
            y_pil = self._rates_decimal.copy()
        elif self.space_coords == "LogarithmOfDiscountFactors":
            if self._input_mode == "log_df":
                y_pil = self._raw_quotes * self._quote_scale
            elif self._input_mode == "df":
                y_pil = np.log(np.maximum(self._raw_quotes * self._quote_scale, 1e-15))
            else:
                y_pil = -self._rates_decimal * self._tenors
        elif self.space_coords in ("DiscountFactors", "Factors"):
            if self._input_mode == "df":
                y_pil = self._raw_quotes * self._quote_scale
            elif self._input_mode == "log_df":
                y_pil = np.exp(self._raw_quotes * self._quote_scale)
            else:
                y_pil = np.exp(-self._rates_decimal * self._tenors)
        else:
            raise ValueError(f"Unsupported InterpolationSpaceCoordinates: {self.space_coords!r}")

        self._y_pillars: npt.NDArray[np.float64] = np.asarray(y_pil, dtype=np.float64)

        self._fast_coeffs: npt.NDArray[np.float64] | None = None
        self._fast_basis_coeffs: npt.NDArray[np.float64] | None = None
        self._cs_val: Any = None
        self._cs_basis: Any = None
        self._nonlin_interp: Any = None
        self.interpolation_metadata: dict[str, Any] = {}
        self._build_interpolator()

        # Freeze immutable backing arrays
        self._tenors.setflags(write=False)
        self._raw_quotes.setflags(write=False)
        self._rates_decimal.setflags(write=False)
        self._y_pillars.setflags(write=False)
        if self._fast_coeffs is not None:
            self._fast_coeffs.setflags(write=False)

    def _build_interpolator(self) -> None:
        k_len = len(self._tenors)
        method = self.interpolation_method
        approx_aliases = {
            "LME": "NotAKnotCubicSpline",
            "Polynomial": "NotAKnotCubicSpline",
            "Rational": "NotAKnotCubicSpline",
            "FwdFwdQuartic": "NotAKnotCubicSpline",
            "ExpSpline": "ClampedCubicSpline",
            "ConstrainedCubicSpline": "ClampedCubicSpline",
            "AdaptiveSpline": "ClampedCubicSpline",
            "Kruger": "PCHIP",
            "Harmonic": "PCHIP",
            "Smooth": "PCHIP",
        }
        if method in approx_aliases:
            self.interpolation_metadata = {"exact": False, "surrogate_method": approx_aliases[method]}
        else:
            self.interpolation_metadata = {"exact": True, "surrogate_method": method}

        if k_len == 1:
            return

        if method in (
            "CubicSpline",
            "ClampedCubicSpline",
            "Spline",
            "ConstrainedCubicSpline",
            "AdaptiveSpline",
            "ExpSpline",
        ):
            self._fast_coeffs = fast_clamped_cubic_coeffs(self._tenors, self._y_pillars)
        elif method == "NaturalCubicSpline":
            self._cs_val = CubicSpline(self._tenors, self._y_pillars, bc_type="natural")
        elif method in (
            "NotAKnotCubicSpline",
            "Polynomial",
            "LME",
            "Rational",
            "FwdFwdQuartic",
        ):
            if k_len >= 4:
                self._cs_val = CubicSpline(self._tenors, self._y_pillars, bc_type="not-a-knot")
            else:
                self._fast_coeffs = fast_clamped_cubic_coeffs(self._tenors, self._y_pillars)
        elif method in ("QuadraticSpline", "Quadratic"):
            deg = min(2, k_len - 1)
            self._cs_val = make_interp_spline(self._tenors, self._y_pillars, k=deg)
        elif method in ("MonotoneCubic", "PCHIP", "Kruger", "Harmonic", "Smooth"):
            self._nonlin_interp = PchipInterpolator(self._tenors, self._y_pillars)
        elif method == "Akima":
            if k_len >= 5:
                self._nonlin_interp = Akima1DInterpolator(self._tenors, self._y_pillars)
            else:
                self._nonlin_interp = PchipInterpolator(self._tenors, self._y_pillars)
        elif method in (
            "MonotoneConvex",
            "Linear",
            "FwdFwdLinear",
            "LogLinear",
            "LogDiscountLinear",
            "PiecewiseConstantForward",
            "FlatLeft",
            "PiecewiseConstantToTheLeft",
            "FlatRight",
            "PiecewiseConstantToTheRight",
            "Bessel",
            "NelsonSiegel",
            "NelsonSiegelContinuous",
            "NelsonSiegelDiscrete",
            "Svensson",
        ):
            pass
        else:
            raise ValueError(f"Unsupported CurveInterpolationMethod: {method!r}")

    def _ensure_basis_interpolator(self) -> None:
        if self._fast_basis_coeffs is not None or self._cs_basis is not None:
            return
        k_len = len(self._tenors)
        if k_len <= 1:
            return
        method = self.interpolation_method
        eye_k = np.eye(k_len, dtype=np.float64)
        if self._fast_coeffs is not None:
            self._fast_basis_coeffs = fast_clamped_cubic_coeffs(self._tenors, eye_k)
            self._fast_basis_coeffs.setflags(write=False)
        elif method == "NaturalCubicSpline":
            self._cs_basis = CubicSpline(self._tenors, eye_k, axis=0, bc_type="natural")
        elif method in (
            "NotAKnotCubicSpline",
            "Polynomial",
            "LME",
            "Rational",
            "FwdFwdQuartic",
        ):
            bc = "not-a-knot" if k_len >= 4 else "clamped"
            self._cs_basis = CubicSpline(self._tenors, eye_k, axis=0, bc_type=bc)
        elif method in ("QuadraticSpline", "Quadratic"):
            deg = min(2, k_len - 1)
            self._cs_basis = make_interp_spline(self._tenors, eye_k, k=deg, axis=0)

    @property
    def tenors(self) -> npt.NDArray[np.float64]:
        """Pillar tenors in curve day-count convention (float64)."""
        return self._tenors

    @property
    def pillar_tenors(self) -> npt.NDArray[np.float64]:
        """Alias for `tenors`."""
        return self._tenors

    @property
    def rates(self) -> npt.NDArray[np.float64]:
        """Decimal zero rates at pillar dates (float64)."""
        return self._rates_decimal

    @property
    def quote_scale(self) -> float:
        """Quote scaling factor (0.01 for Percentage, 1.0 for Decimal)."""
        return self._quote_scale

    def _to_tenors_and_scalar_flag(
        self, dates_or_tenors: Any
    ) -> Tuple[npt.NDArray[np.float64], bool]:
        return _convert_dates_or_tenors(
            dates_or_tenors,
            as_of_date=self.as_of_date,
            dcc=self.dcc,
            caller_name="ClampedYieldCurve",
        )

    def _spline_weights_interior_and_extrap(self, tau_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        m_eval = len(tau_arr)
        k_len = len(self._tenors)
        t_min = self._tenors[0]
        t_max = self._tenors[-1]
        left_flat, right_flat = _extrap_sides(self.extrapolation_method)
        t_clip = np.clip(tau_arr, t_min, t_max)

        self._ensure_basis_interpolator()
        if self._fast_basis_coeffs is not None:
            w = eval_clamped_cubic_coeffs(self._tenors, self._fast_basis_coeffs, t_clip)
            if not left_flat:
                mask_l = tau_arr < t_min
                if np.any(mask_l):
                    dt0 = self._tenors[1] - t_min
                    alpha_l = (tau_arr[mask_l] - t_min)[:, None] / dt0
                    w[mask_l] = 0.0
                    w[mask_l, 0] = 1.0 - alpha_l[:, 0]
                    w[mask_l, 1] = alpha_l[:, 0]
            if not right_flat:
                mask_r = tau_arr > t_max
                if np.any(mask_r):
                    dtn = t_max - self._tenors[-2]
                    alpha_r = (tau_arr[mask_r] - t_max)[:, None] / dtn
                    w[mask_r] = 0.0
                    w[mask_r, -2] = -alpha_r[:, 0]
                    w[mask_r, -1] = 1.0 + alpha_r[:, 0]
            return w

        if self._cs_basis is not None:
            w = np.asarray(self._cs_basis(t_clip), dtype=np.float64)
            if not left_flat:
                mask_l = tau_arr < t_min
                if np.any(mask_l):
                    w0 = np.asarray(self._cs_basis(t_min), dtype=np.float64)
                    dw0 = np.asarray(self._cs_basis.derivative()(t_min), dtype=np.float64)
                    w[mask_l] = w0[None, :] + (tau_arr[mask_l] - t_min)[:, None] * dw0[None, :]
            if not right_flat:
                mask_r = tau_arr > t_max
                if np.any(mask_r):
                    wn = np.asarray(self._cs_basis(t_max), dtype=np.float64)
                    dwn = np.asarray(self._cs_basis.derivative()(t_max), dtype=np.float64)
                    w[mask_r] = wn[None, :] + (tau_arr[mask_r] - t_max)[:, None] * dwn[None, :]
            return w

        method = self.interpolation_method
        if method == "Linear":
            return _linear_basis(self._tenors, tau_arr, extrapolation=self.extrapolation_method)
        if method == "FwdFwdLinear":
            if self.space_coords in ("Rates", "LogarithmOfDiscountFactors"):
                return _fwd_fwd_linear_basis(
                    self._tenors,
                    tau_arr,
                    extrapolation=self.extrapolation_method,
                    space_coords=self.space_coords,
                )
            w_r = _fwd_fwd_linear_basis(
                self._tenors,
                tau_arr,
                extrapolation=self.extrapolation_method,
                space_coords="Rates",
            )
            s_vals = self._eval_space_coord(tau_arr)
            y_pos = np.maximum(self._y_pillars, 1e-15)
            pos_pil = (self._tenors > 1e-12).astype(np.float64)
            t_safe = np.maximum(np.abs(self._tenors), 1e-12)
            jac = (s_vals[:, None] * (tau_arr[:, None] * w_r)) * (pos_pil / (t_safe * y_pos))[None, :]
            mask_l = tau_arr <= t_min
            if np.any(mask_l):
                jac[mask_l] = 0.0
                if left_flat or k_len < 2:
                    jac[mask_l, 0] = 1.0
                else:
                    alpha_l = (tau_arr[mask_l] - t_min) / (self._tenors[1] - t_min)
                    jac[mask_l, 0] = 1.0 - alpha_l
                    jac[mask_l, 1] = alpha_l
            mask_r = tau_arr >= t_max
            if np.any(mask_r):
                jac[mask_r] = 0.0
                if right_flat or k_len < 2:
                    jac[mask_r, -1] = 1.0
                else:
                    alpha_r = (tau_arr[mask_r] - t_max) / (t_max - self._tenors[-2])
                    jac[mask_r, -2] = -alpha_r
                    jac[mask_r, -1] = 1.0 + alpha_r
            return np.asarray(jac, dtype=np.float64)
        if method in ("LogLinear", "LogDiscountLinear", "PiecewiseConstantForward"):
            if self.space_coords in ("Rates", "LogarithmOfDiscountFactors"):
                return _log_linear_rate_basis(
                    self._tenors,
                    tau_arr,
                    extrapolation=self.extrapolation_method,
                    space_coords=self.space_coords,
                )
            w_lin = _linear_basis(self._tenors, tau_arr, extrapolation=self.extrapolation_method)
            y_pos = np.maximum(self._y_pillars, 1e-15)
            s_vals = np.exp(w_lin @ np.log(y_pos))
            return np.asarray((s_vals[:, None] * w_lin) / y_pos[None, :], dtype=np.float64)
        if method == "FlatLeft":
            w = np.zeros((m_eval, k_len), dtype=np.float64)
            idx = np.clip(np.searchsorted(self._tenors, tau_arr, side="right") - 1, 0, k_len - 1)
            w[np.arange(m_eval), idx] = 1.0
            return w
        if method == "FlatRight":
            w = np.zeros((m_eval, k_len), dtype=np.float64)
            idx = np.clip(np.searchsorted(self._tenors, tau_arr, side="left"), 0, k_len - 1)
            w[np.arange(m_eval), idx] = 1.0
            return w
        if method == "PiecewiseConstantToTheLeft":
            w = np.zeros((m_eval, k_len), dtype=np.float64)
            idx = np.clip(np.searchsorted(self._tenors, tau_arr, side="left"), 0, k_len - 1)
            w[np.arange(m_eval), idx] = 1.0
            return w
        if method == "PiecewiseConstantToTheRight":
            w = np.zeros((m_eval, k_len), dtype=np.float64)
            idx = np.clip(np.searchsorted(self._tenors, tau_arr, side="right") - 1, 0, k_len - 1)
            w[np.arange(m_eval), idx] = 1.0
            return w
        if method == "Bessel":
            return _bessel_hermite_basis(
                self._tenors, tau_arr, extrapolation=self.extrapolation_method
            )
        if method in ("NelsonSiegel", "NelsonSiegelContinuous", "NelsonSiegelDiscrete"):
            return _nelson_siegel_basis(
                self._tenors, tau_arr, svensson=False, extrapolation=self.extrapolation_method
            )
        if method == "Svensson":
            return _nelson_siegel_basis(
                self._tenors, tau_arr, svensson=True, extrapolation=self.extrapolation_method
            )

        # Non-linear interpolators (PCHIP / Akima / MonotoneConvex): exact central-difference dS/dy
        eps = 1e-7
        jac = np.zeros((m_eval, k_len), dtype=np.float64)
        for k in range(k_len):
            y_up = self._y_pillars.copy()
            y_dn = self._y_pillars.copy()
            y_up[k] += eps
            y_dn[k] -= eps
            s_up = self._eval_nonlin_with_y(tau_arr, y_up)
            s_dn = self._eval_nonlin_with_y(tau_arr, y_dn)
            jac[:, k] = (s_up - s_dn) / (2.0 * eps)
        return jac

    def _eval_nonlin_with_y(
        self,
        tau_arr: npt.NDArray[np.float64],
        y_pil: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        method = self.interpolation_method
        if method == "MonotoneConvex":
            return _monotone_convex_eval(
                self._tenors,
                y_pil,
                tau_arr,
                space_coords=self.space_coords,
                extrapolation=self.extrapolation_method,
            )
        k_len = len(self._tenors)
        if method == "Akima" and k_len >= 5:
            interp = Akima1DInterpolator(self._tenors, y_pil)
        else:
            interp = PchipInterpolator(self._tenors, y_pil)
        return self._eval_scipy_with_extrap(interp, tau_arr)

    def _eval_scipy_with_extrap(
        self,
        interp: Any,
        tau_arr: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        t_min = self._tenors[0]
        t_max = self._tenors[-1]
        left_flat, right_flat = _extrap_sides(self.extrapolation_method)
        t_clip = np.clip(tau_arr, t_min, t_max)
        vals = np.asarray(interp(t_clip), dtype=np.float64)
        if not left_flat:
            mask_l = tau_arr < t_min
            if np.any(mask_l):
                s0 = float(interp(t_min))
                ds0 = float(interp.derivative()(t_min))
                vals[mask_l] = s0 + ds0 * (tau_arr[mask_l] - t_min)
        if not right_flat:
            mask_r = tau_arr > t_max
            if np.any(mask_r):
                sn = float(interp(t_max))
                dsn = float(interp.derivative()(t_max))
                vals[mask_r] = sn + dsn * (tau_arr[mask_r] - t_max)
        return vals

    def spline_weights(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Returns the M_eval x K basis weight matrix dS / dy_pillars in float64."""
        tau_arr, _ = self._to_tenors_and_scalar_flag(dates_or_tenors)
        m_eval = len(tau_arr)
        k_len = len(self._tenors)
        if m_eval == 0:
            return np.zeros((0, k_len), dtype=np.float64)
        if k_len == 1:
            return np.ones((m_eval, 1), dtype=np.float64)
        return self._spline_weights_interior_and_extrap(tau_arr)

    def _eval_space_coord(self, tau_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluates the interpolated curve coordinate S(tau) in float64 honoring ExtrapolationMethod."""
        if len(tau_arr) == 0:
            return np.zeros(0, dtype=np.float64)
        if len(self._tenors) == 1:
            return np.full(len(tau_arr), self._y_pillars[0], dtype=np.float64)

        t_min = self._tenors[0]
        t_max = self._tenors[-1]
        left_flat, right_flat = _extrap_sides(self.extrapolation_method)

        if self._fast_coeffs is not None:
            t_clip = np.clip(tau_arr, t_min, t_max)
            vals = eval_clamped_cubic_coeffs(self._tenors, self._fast_coeffs, t_clip)
            if not left_flat:
                mask_l = tau_arr < t_min
                if np.any(mask_l):
                    slope_l = (self._y_pillars[1] - self._y_pillars[0]) / (self._tenors[1] - t_min)
                    vals[mask_l] = self._y_pillars[0] + slope_l * (tau_arr[mask_l] - t_min)
            if not right_flat:
                mask_r = tau_arr > t_max
                if np.any(mask_r):
                    slope_r = (self._y_pillars[-1] - self._y_pillars[-2]) / (t_max - self._tenors[-2])
                    vals[mask_r] = self._y_pillars[-1] + slope_r * (tau_arr[mask_r] - t_max)
            return vals

        if self._cs_val is not None:
            return self._eval_scipy_with_extrap(self._cs_val, tau_arr)

        if self._nonlin_interp is not None:
            return self._eval_scipy_with_extrap(self._nonlin_interp, tau_arr)

        if self.interpolation_method == "MonotoneConvex":
            return _monotone_convex_eval(
                self._tenors,
                self._y_pillars,
                tau_arr,
                space_coords=self.space_coords,
                extrapolation=self.extrapolation_method,
            )

        if self.space_coords in ("DiscountFactors", "Factors"):
            if self.interpolation_method in ("LogLinear", "LogDiscountLinear", "PiecewiseConstantForward"):
                w_lin = _linear_basis(self._tenors, tau_arr, extrapolation=self.extrapolation_method)
                y_pos = np.maximum(self._y_pillars, 1e-15)
                return np.asarray(np.exp(w_lin @ np.log(y_pos)), dtype=np.float64)
            if self.interpolation_method == "FwdFwdLinear":
                t_safe = np.maximum(np.abs(self._tenors), 1e-12)
                y_pos = np.maximum(self._y_pillars, 1e-15)
                r_pil = np.where(self._tenors > 1e-12, -np.log(y_pos) / t_safe, 0.0)
                w_r = _fwd_fwd_linear_basis(
                    self._tenors,
                    tau_arr,
                    extrapolation=self.extrapolation_method,
                    space_coords="Rates",
                )
                vals = np.exp(-tau_arr * (w_r @ r_pil))
                mask_l = tau_arr <= t_min
                if np.any(mask_l):
                    if left_flat:
                        vals[mask_l] = self._y_pillars[0]
                    else:
                        slope_l = (self._y_pillars[1] - self._y_pillars[0]) / (self._tenors[1] - t_min)
                        vals[mask_l] = self._y_pillars[0] + slope_l * (tau_arr[mask_l] - t_min)
                mask_r = tau_arr >= t_max
                if np.any(mask_r):
                    if right_flat:
                        vals[mask_r] = self._y_pillars[-1]
                    else:
                        slope_r = (self._y_pillars[-1] - self._y_pillars[-2]) / (t_max - self._tenors[-2])
                        vals[mask_r] = self._y_pillars[-1] + slope_r * (tau_arr[mask_r] - t_max)
                return np.asarray(vals, dtype=np.float64)

        w = self._spline_weights_interior_and_extrap(tau_arr)
        return np.asarray(w @ self._y_pillars, dtype=np.float64)

    def zero_rate(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Evaluates continuously compounded decimal zero rate r(tau) in float64."""
        tau_arr, is_scalar = self._to_tenors_and_scalar_flag(dates_or_tenors)
        y_arr = self._eval_space_coord(tau_arr)
        if self.space_coords == "Rates":
            r_arr = y_arr
        elif self.space_coords == "LogarithmOfDiscountFactors":
            t_safe = np.maximum(np.abs(tau_arr), 1e-12)
            r0_fallback = (
                float(self._rates_decimal[1])
                if self._tenors[0] <= 1e-12 and len(self._tenors) > 1
                else float(self._rates_decimal[0])
            )
            r_arr = np.where(tau_arr > 1e-12, -y_arr / t_safe, r0_fallback)
        else:
            t_safe = np.maximum(np.abs(tau_arr), 1e-12)
            df_safe = np.maximum(y_arr, 1e-15)
            r0_fallback = (
                float(self._rates_decimal[1])
                if self._tenors[0] <= 1e-12 and len(self._tenors) > 1
                else float(self._rates_decimal[0])
            )
            r_arr = np.where(tau_arr > 1e-12, -np.log(df_safe) / t_safe, r0_fallback)

        if is_scalar:
            return np.float64(r_arr[0])  # type: ignore[return-value]
        return np.asarray(r_arr, dtype=np.float64)

    def df(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Evaluates the discount factor DF(tau) in float64."""
        tau_arr, is_scalar = self._to_tenors_and_scalar_flag(dates_or_tenors)
        y_val = self._eval_space_coord(tau_arr)
        if self.space_coords == "Rates":
            df_arr = np.where(tau_arr <= 0.0, 1.0, np.exp(-y_val * tau_arr))
        elif self.space_coords == "LogarithmOfDiscountFactors":
            df_arr = np.where(tau_arr <= 0.0, 1.0, np.exp(y_val))
        else:
            df_arr = np.where(tau_arr <= 0.0, 1.0, np.maximum(y_val, 1e-15))

        if is_scalar:
            return np.float64(df_arr[0])  # type: ignore[return-value]
        return np.asarray(df_arr, dtype=np.float64)

    def df_and_d_df_d_pillar(
        self,
        dates_or_tenors: Any,
        *,
        in_quote_units: bool = True,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Computes `(df, d_df_d_pillar_quote)` in float64 using the spline basis matrix."""
        tau_arr, _ = self._to_tenors_and_scalar_flag(dates_or_tenors)
        df_arr = np.asarray(self.df(tau_arr), dtype=np.float64)
        w_mat = self.spline_weights(tau_arr)
        scale = self._quote_scale if in_quote_units else 1.0
        pos_eval = (tau_arr > 0.0).astype(np.float64)[:, None]

        # 1. d DF(tau_m) / d y_pil
        if self.space_coords == "Rates":
            d_df_d_y = -(np.maximum(tau_arr, 0.0)[:, None] * df_arr[:, None]) * w_mat
        elif self.space_coords == "LogarithmOfDiscountFactors":
            d_df_d_y = (pos_eval * df_arr[:, None]) * w_mat
        else:
            d_df_d_y = pos_eval * w_mat

        # 2. d y_pil / d quote_k (masked to 0 at tau_k <= 1e-12 whenever 1/tau_k appears)
        pos_pil = self._tenors > 1e-12
        t_safe = np.maximum(np.abs(self._tenors), 1e-12)
        if self.space_coords == "Rates":
            if self._input_mode == "log_df":
                dy_dq = np.where(pos_pil, -1.0 / t_safe, 0.0) * scale
            elif self._input_mode == "df":
                df_pil = np.maximum(self._raw_quotes * self._quote_scale, 1e-15)
                dy_dq = np.where(pos_pil, -1.0 / (t_safe * df_pil), 0.0) * scale
            else:
                dy_dq = np.full(len(self._tenors), scale, dtype=np.float64)
        elif self.space_coords == "LogarithmOfDiscountFactors":
            if self._input_mode == "log_df":
                dy_dq = np.full(len(self._tenors), scale, dtype=np.float64)
            elif self._input_mode == "df":
                df_pil = np.maximum(self._raw_quotes * self._quote_scale, 1e-15)
                dy_dq = (1.0 / df_pil) * scale
            else:
                dy_dq = (-self._tenors) * scale
        else:
            if self._input_mode == "df":
                dy_dq = np.full(len(self._tenors), scale, dtype=np.float64)
            elif self._input_mode == "log_df":
                dy_dq = self._y_pillars * scale
            else:
                dy_dq = (-self._tenors * self._y_pillars) * scale

        d_df = d_df_d_y * dy_dq[None, :]
        return df_arr, np.asarray(d_df, dtype=np.float64)


# ── RepoCurve ─────────────────────────────────────────────────────────────────

_ISO_DATE_PREFIX_RE = re.compile(r"^\s*\d{4}-\d{1,2}-\d{1,2}")


def _parse_repo_periods_days(periods: Sequence[Any]) -> npt.NDArray[np.float64]:
    days: list[float] = []
    for p in periods:
        if isinstance(p, (int, float, np.integer, np.floating)) and not isinstance(p, bool):
            val = float(p)
            if not np.isfinite(val):
                raise ValueError(f"Non-finite repo period value {p!r}")
            days.append(val)
            continue
        m = _REPO_PERIOD_RE.match(str(p))
        if m is None:
            raise ValueError(f"Invalid repo period format {p!r}; expected '<n>[DWMY]'")
        val = float(m.group(1))
        unit = m.group(2).upper()
        if unit == "D":
            days.append(val)
        elif unit == "W":
            days.append(val * 7.0)
        elif unit == "M":
            days.append(val * 30.0)
        elif unit == "Y":
            days.append(val * 365.0)
    return np.asarray(days, dtype=np.float64)


class RepoCurve:
    """Equity repo spread curve with automatic flat-0.0% fallback when omitted."""

    def __init__(
        self,
        spec: RepoSpreadCurveSpec | Mapping[str, Any] | None = None,
        *,
        as_of_date: datetime.date | datetime.datetime | None = None,
        periods_days: Sequence[float] | npt.NDArray[np.float64] | None = None,
        rates: Sequence[float] | npt.NDArray[np.float64] | None = None,
        dcc: str | DayCountConventionEnum = "Act360",
        interpolation_method: str | CurveInterpolationMethodEnum = "CubicSpline",
        extrapolation_method: str = "Flat",
        quote_expression: str = "Percentage",
        identifier: str = "DEFAULT",
    ) -> None:
        self.as_of_date = _normalize_datetime(as_of_date) if as_of_date is not None else None
        if isinstance(spec, RepoSpreadCurveSpec):
            raw_as_of = getattr(spec, "AsOfDate", None)
            if self.as_of_date is None and raw_as_of is not None:
                self.as_of_date = _normalize_datetime(raw_as_of)
            self.identifier = spec.Identifier
            self.dcc = spec.InterpolationDayCountConvention.value
            self.interpolation_method = spec.InterpolationMethod.value
            self.extrapolation_method = spec.ExtrapolationMethod.value
            self.quote_expression = spec.RateQuoteExpression.value
            days_arr = np.asarray(spec.RatesSchedule.period_days(), dtype=np.float64)
            denom = dcc_denominator(self.dcc)
            tenors_arr = days_arr / denom
            rates_arr = np.asarray(spec.RatesSchedule.Rates, dtype=np.float64)
            self._is_flat_zero = False
            self._inner_curve: ClampedYieldCurve | None = ClampedYieldCurve(
                pillar_tenors=tenors_arr,
                rates=rates_arr,
                as_of_date=self.as_of_date,
                dcc=self.dcc,
                interpolation_method=self.interpolation_method,
                space_coords=spec.InterpolationSpaceCoordinates.value,
                extrapolation_method=self.extrapolation_method,
                quote_expression=self.quote_expression,
                identifier=self.identifier,
            )
        elif isinstance(spec, Mapping):
            raw_as_of = _extract_curve_field(spec, "AsOfDate", "as_of_date", default=None)
            if self.as_of_date is None and raw_as_of is not None:
                self.as_of_date = _normalize_datetime(raw_as_of)
            self.identifier = str(_extract_curve_field(spec, "Identifier", "identifier", default=identifier))
            raw_dcc = _extract_curve_field(
                spec, "InterpolationDayCountConvention", "DayCountConvention", "day_count_convention", "dcc", default=dcc
            )
            self.dcc = str(getattr(raw_dcc, "value", raw_dcc))
            raw_interp = _extract_curve_field(
                spec, "InterpolationMethod", "interpolation_method", default=interpolation_method
            )
            self.interpolation_method = str(getattr(raw_interp, "value", raw_interp))
            raw_extrap = _extract_curve_field(
                spec, "ExtrapolationMethod", "extrapolation_method", default=extrapolation_method
            )
            self.extrapolation_method = str(getattr(raw_extrap, "value", raw_extrap))
            raw_space = _extract_curve_field(
                spec, "InterpolationSpaceCoordinates", "space_coords", default="Rates"
            )
            space_coords = str(getattr(raw_space, "value", raw_space))
            raw_qe = _extract_curve_field(
                spec, "RateQuoteExpression", "rate_quote_expression", "quote_expression", default=quote_expression
            )
            self.quote_expression = str(getattr(raw_qe, "value", raw_qe))
            sched = _extract_curve_field(spec, "RatesSchedule", "rates_schedule", default=spec)
            raw_periods = _extract_curve_field(sched, "Periods", "periods_days", "Dates", "dates", default=[])
            raw_rates = _extract_curve_field(sched, "Rates", "rates_pct", "rates", default=[])
            rates_arr = np.asarray(raw_rates, dtype=np.float64)
            periods_list = list(raw_periods)
            if rates_arr.size == 0:
                self._is_flat_zero = True
                self._inner_curve = None
            else:
                first_p = periods_list[0] if len(periods_list) > 0 else None
                is_date_period = isinstance(first_p, (datetime.date, datetime.datetime)) or (
                    isinstance(first_p, str)
                    and (bool(_ISO_DATE_PREFIX_RE.match(first_p)) or ":" in first_p)
                )
                if is_date_period:
                    if self.as_of_date is None:
                        raise ValueError("RepoCurve requires `as_of_date` when Dates are provided.")
                    tenors_arr = year_fractions(
                        self.as_of_date, [_normalize_datetime(d) for d in periods_list], self.dcc
                    )
                else:
                    days_arr = _parse_repo_periods_days(periods_list)
                    tenors_arr = days_arr / dcc_denominator(self.dcc)
                self._is_flat_zero = False
                self._inner_curve = ClampedYieldCurve(
                    pillar_tenors=tenors_arr,
                    rates=rates_arr,
                    as_of_date=self.as_of_date,
                    dcc=self.dcc,
                    interpolation_method=self.interpolation_method,
                    space_coords=space_coords,
                    extrapolation_method=self.extrapolation_method,
                    quote_expression=self.quote_expression,
                    identifier=self.identifier,
                )
        elif periods_days is not None and rates is not None:
            self.identifier = identifier
            self.dcc = dcc.value if isinstance(dcc, Enum) else str(dcc)
            self.interpolation_method = (
                interpolation_method.value
                if isinstance(interpolation_method, Enum)
                else str(interpolation_method)
            )
            self.extrapolation_method = (
                extrapolation_method.value
                if isinstance(extrapolation_method, Enum)
                else str(extrapolation_method)
            )
            self.quote_expression = (
                quote_expression.value
                if isinstance(quote_expression, Enum)
                else str(quote_expression)
            )
            days_arr = np.asarray(periods_days, dtype=np.float64)
            tenors_arr = days_arr / dcc_denominator(self.dcc)
            self._is_flat_zero = False
            self._inner_curve = ClampedYieldCurve(
                pillar_tenors=tenors_arr,
                rates=rates,
                as_of_date=self.as_of_date,
                dcc=self.dcc,
                interpolation_method=self.interpolation_method,
                space_coords="Rates",
                extrapolation_method=self.extrapolation_method,
                quote_expression=self.quote_expression,
                identifier=self.identifier,
            )
        else:
            self.identifier = identifier
            self.dcc = dcc.value if isinstance(dcc, Enum) else str(dcc)
            self.interpolation_method = "CubicSpline"
            self.extrapolation_method = "Flat"
            self.quote_expression = "Percentage"
            self._is_flat_zero = True
            self._inner_curve = None

    @property
    def is_flat_zero(self) -> bool:
        """True when constructed with `spec=None` (flat 0.0% repo curve)."""
        return self._is_flat_zero

    def _to_repo_tenors_and_scalar_flag(
        self, dates_or_tenors: Any
    ) -> Tuple[npt.NDArray[np.float64], bool]:
        return _convert_dates_or_tenors(
            dates_or_tenors,
            as_of_date=self.as_of_date,
            dcc=self.dcc,
            caller_name="RepoCurve",
        )

    def repo_rate(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Evaluates decimal repo spread rate q_repo(t) in float64."""
        tau_arr, is_scalar = self._to_repo_tenors_and_scalar_flag(dates_or_tenors)
        if self._is_flat_zero or self._inner_curve is None:
            res = np.zeros_like(tau_arr, dtype=np.float64)
        else:
            res = np.asarray(self._inner_curve.zero_rate(tau_arr), dtype=np.float64)
        if is_scalar:
            return np.float64(res[0])  # type: ignore[return-value]
        return res

    def integrated_repo(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Evaluates integrated repo spread int_{t_0}^t q_repo(u) du in float64 (0 for tau <= 0)."""
        tau_arr, is_scalar = self._to_repo_tenors_and_scalar_flag(dates_or_tenors)
        if self._is_flat_zero or self._inner_curve is None:
            res = np.zeros_like(tau_arr, dtype=np.float64)
        else:
            r_arr = np.asarray(self._inner_curve.zero_rate(tau_arr), dtype=np.float64)
            res = np.where(tau_arr <= 0.0, 0.0, r_arr * tau_arr)
        if is_scalar:
            return np.float64(res[0])  # type: ignore[return-value]
        return res

    def repo_df(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Evaluates repo discount factor exp(-int_{t_0}^t q_repo(u) du) in float64 (1.0 for tau <= 0)."""
        tau_arr, is_scalar = self._to_repo_tenors_and_scalar_flag(dates_or_tenors)
        int_repo = np.asarray(self.integrated_repo(tau_arr), dtype=np.float64)
        df_arr = np.where(tau_arr <= 0.0, 1.0, np.exp(-int_repo))
        if is_scalar:
            return np.float64(df_arr[0])  # type: ignore[return-value]
        return df_arr
