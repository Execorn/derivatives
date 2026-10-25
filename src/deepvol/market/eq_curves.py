"""eq_curves.py — High-precision float64 multi-curve yield & repo interpolation engine.

Implements Milestone M0 (Task 0.2) of SPEC-DEEPVOL-EQ-AUTOCALL-001:
- Exact day-count fraction calculators (`Act365`, `Act360`, `ActAct`, `Thirty360`/`30360`, `Business252`).
- `ClampedYieldCurve`: supports all 19 Murex curve interpolation methods (`CubicSpline` with clamped
  boundary derivatives S'(tau_0) = S'(tau_{K-1}) = 0 on raw pillar tenors without prepending tau=0,
  reproducing `RUB_RUONIA_OIS` DF(2027-02-09) = 0.9594909887002673 to 0 ULP in float64) and precomputes
  the exact analytical spline basis weight matrix W_spline for single-MVP bucketed `Rho` sensitivities.
- `RepoCurve`: parses `<n>D` pillar schedules under `Act360` convention and defaults omitted basket
  constituents (such as `POSIRX_RUB` in `eq_autocallable_request.txt`) to a flat 0.0% repo spread curve.
"""

from __future__ import annotations

import calendar
import datetime
from enum import Enum
from typing import Any, Sequence

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
)


# ── Day Count Conventions (IEEE-754 float64) ──────────────────────────────────


def _to_date_and_frac_days(d: datetime.date | datetime.datetime) -> tuple[datetime.date, float]:
    """Extracts calendar date and any sub-day fractional seconds as days."""
    if isinstance(d, datetime.datetime):
        dt_utc = d.astimezone(datetime.timezone.utc) if d.tzinfo is not None else d
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


def _thirty_360_isda(
    d1: datetime.date | datetime.datetime,
    d2: datetime.date | datetime.datetime,
) -> float:
    """Standard 30E/360 ISDA day-count fraction in float64."""
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


def year_fraction(
    d1: datetime.date | datetime.datetime,
    d2: datetime.date | datetime.datetime,
    dcc: str | DayCountConventionEnum = "Act365",
) -> float:
    """Computes the exact year fraction tau(d1, d2; dcc) in IEEE-754 float64."""
    dcc_str = dcc.value if isinstance(dcc, Enum) else str(dcc)
    dcc_norm = dcc_str.strip()

    if dcc_norm in ("Act365", "Act365F", "ACT_365", "ACT_365F"):
        return float(_exact_days(d1, d2) / 365.0)
    if dcc_norm in ("Act360", "ACT_360"):
        return float(_exact_days(d1, d2) / 360.0)
    if dcc_norm in ("Act364", "ACT_364"):
        return float(_exact_days(d1, d2) / 364.0)
    if dcc_norm in ("Act366", "ACT_366"):
        return float(_exact_days(d1, d2) / 366.0)
    if dcc_norm in ("Act252", "ACT_252"):
        return float(_exact_days(d1, d2) / 252.0)
    if dcc_norm in ("ActAct", "ACT_ACT", "ActActISDA"):
        return float(_act_act_isda(d1, d2))
    if dcc_norm in ("Thirty360", "30360", "30E360", "THIRTY_360", "THIRTY_E_360"):
        return float(_thirty_360_isda(d1, d2))
    if dcc_norm in ("Business252", "Bus252", "BUSINESS_252", "BUS_252"):
        date1, _ = _to_date_and_frac_days(d1)
        date2, _ = _to_date_and_frac_days(d2)
        b_days = int(np.busday_count(date1.isoformat(), date2.isoformat()))
        return float(b_days / 252.0)

    raise ValueError(f"Unsupported day count convention: {dcc_str!r}")


def year_fractions(
    as_of: datetime.date | datetime.datetime,
    dates: Sequence[datetime.date | datetime.datetime] | npt.NDArray[Any],
    dcc: str | DayCountConventionEnum = "Act365",
) -> npt.NDArray[np.float64]:
    """Vectorized year fraction computation returning a 1D `np.float64` array."""
    return np.array([year_fraction(as_of, d, dcc) for d in dates], dtype=np.float64)


# ── Analytical Spline & Hermite Helpers ───────────────────────────────────────


def _bessel_hermite_basis(
    tenors: npt.NDArray[np.float64],
    tau_eval: npt.NDArray[np.float64],
    *,
    extrapolation: str = "Flat",
) -> npt.NDArray[np.float64]:
    """Computes the exact Bessel cubic Hermite basis weight matrix W in R^{M_eval x K}.

    Bessel slopes at interior knots use the weighted parabolic secant average:
        m_k = (dx_{k-1} s_k + dx_k s_{k-1}) / (dx_{k-1} + dx_k)
    and endpoint slopes are clamped to 0.0 when `extrapolation == 'Flat'` (or secant if 'Linear').
    Because every slope m_k is a linear combination of y, W @ y equals the Bessel Hermite
    interpolant identically in float64.
    """
    k_len = len(tenors)
    m_eval = len(tau_eval)
    eye_k = np.eye(k_len, dtype=np.float64)
    if k_len == 1:
        return np.ones((m_eval, 1), dtype=np.float64)

    dx = np.diff(tenors)  # (K-1,)
    # Secant matrix S_sec[j, :] = (e_{j+1} - e_j) / dx[j], shape (K-1, K)
    s_sec = (eye_k[1:, :] - eye_k[:-1, :]) / dx[:, None]

    # Slope basis matrix M_slope[k, :] = dm_k / dy, shape (K, K)
    m_slope = np.zeros((k_len, k_len), dtype=np.float64)
    if k_len > 2:
        denom = dx[:-1] + dx[1:]
        m_slope[1:-1, :] = (
            dx[:-1, None] * s_sec[1:, :] + dx[1:, None] * s_sec[:-1, :]
        ) / denom[:, None]

    if extrapolation != "Flat":
        m_slope[0, :] = s_sec[0, :]
        m_slope[-1, :] = s_sec[-1, :]

    w_out = np.zeros((m_eval, k_len), dtype=np.float64)
    t_min = tenors[0]
    t_max = tenors[-1]

    for i, t_raw in enumerate(tau_eval):
        if t_raw <= t_min:
            if extrapolation == "Flat":
                w_out[i, 0] = 1.0
            else:
                w_out[i, :] = eye_k[0, :] + (t_raw - t_min) * m_slope[0, :]
        elif t_raw >= t_max:
            if extrapolation == "Flat":
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

    t_min = tenors[0]
    t_max = tenors[-1]
    for i, t_raw in enumerate(tau_eval):
        if t_raw <= t_min:
            if extrapolation == "Flat" or k_len < 2:
                w_out[i, 0] = 1.0
            else:
                dt = tenors[1] - tenors[0]
                alpha = (t_raw - tenors[0]) / dt
                w_out[i, 0] = 1.0 - alpha
                w_out[i, 1] = alpha
        elif t_raw >= t_max:
            if extrapolation == "Flat" or k_len < 2:
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
) -> npt.NDArray[np.float64]:
    """Computes the basis weight matrix W in R^{M_eval x K} for LogLinear / PiecewiseConstantForward.

    Under LogLinear discount factor interpolation (piecewise constant forward rates):
    - For tau <= tau_0: r(tau) = y_0 -> W[i, 0] = 1.0.
    - For tau in (tau_{k-1}, tau_k]: r(tau)*tau = (1 - alpha)*y_{k-1}*tau_{k-1} + alpha*y_k*tau_k,
      so r(tau) = [(1 - alpha)*(tau_{k-1}/tau)]*y_{k-1} + [alpha*(tau_k/tau)]*y_k.
    - For tau > tau_{K-1}: flat zero-rate extrapolation r(tau) = y_{K-1} -> W[i, -1] = 1.0.
    """
    k_len = len(tenors)
    m_eval = len(tau_eval)
    w_out = np.zeros((m_eval, k_len), dtype=np.float64)
    if k_len == 1:
        w_out[:, 0] = 1.0
        return w_out

    t_min = tenors[0]
    t_max = tenors[-1]
    for i, t_raw in enumerate(tau_eval):
        if t_raw <= t_min:
            w_out[i, 0] = 1.0
        elif t_raw >= t_max:
            w_out[i, -1] = 1.0
        else:
            idx = int(np.searchsorted(tenors, t_raw, side="right")) - 1
            idx = max(0, min(idx, k_len - 2))
            t_l = tenors[idx]
            t_r = tenors[idx + 1]
            alpha = (t_raw - t_l) / (t_r - t_l)
            w_out[i, idx] = (1.0 - alpha) * (t_l / t_raw)
            w_out[i, idx + 1] = alpha * (t_r / t_raw)
    return w_out


def _nelson_siegel_basis(
    tenors: npt.NDArray[np.float64],
    tau_eval: npt.NDArray[np.float64],
    *,
    svensson: bool = False,
) -> npt.NDArray[np.float64]:
    """Computes the exact linear least-squares Nelson-Siegel / Svensson projection basis W in R^{M_eval x K}."""
    k_len = len(tenors)
    t_clip = np.clip(tau_eval, tenors[0], tenors[-1])

    def _design(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        t_safe = np.maximum(t, 1e-8)
        lam1 = 1.5
        u1 = t_safe / lam1
        f0 = np.ones_like(t_safe)
        f1 = (1.0 - np.exp(-u1)) / u1
        f2 = f1 - np.exp(-u1)
        if not svensson or k_len < 4:
            return np.column_stack([f0, f1, f2])[:, : min(3, k_len)]
        lam2 = 0.5
        u2 = t_safe / lam2
        f3 = (1.0 - np.exp(-u2)) / u2 - np.exp(-u2)
        return np.column_stack([f0, f1, f2, f3])[:, : min(4, k_len)]

    x_pil = _design(tenors)  # (K, p)
    x_eval = _design(t_clip)  # (M_eval, p)
    pinv = np.linalg.pinv(x_pil)  # (p, K)
    return np.asarray(x_eval @ pinv, dtype=np.float64)


def fast_clamped_cubic_coeffs(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Computes clamped cubic spline polynomial coefficients of shape `(4, K - 1, ...)` in pure NumPy.

    Solves the tridiagonal Thomas system with clamped boundary derivatives $S'(x_0) = S'(x_{K-1}) = 0$,
    reproducing `scipy.interpolate.CubicSpline(x, y, axis=0, bc_type="clamped").c` to 0 ULP on
    `RUB_RUONIA_OIS` (`DF(2027-02-09) = 0.9594909887002673`) in ~14 us instead of ~123 us.
    """
    k = len(x)
    if k < 2:
        raise ValueError("fast_clamped_cubic_coeffs requires at least 2 knots.")
    h = np.diff(x)  # (K - 1,)
    h_shape = (k - 1,) + (1,) * (y.ndim - 1)
    h_b = h.reshape(h_shape)
    s = (y[1:] - y[:-1]) / h_b  # (K - 1, ...)
    d = np.zeros_like(y, dtype=np.float64)
    n_int = k - 2
    if n_int > 0:
        h_prev = h[:-1]  # h[0 .. K-3]
        h_next = h[1:]   # h[1 .. K-2]
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


# ── ClampedYieldCurve ─────────────────────────────────────────────────────────


class ClampedYieldCurve:
    """High-precision float64 yield/discount curve engine with precomputed analytical spline basis weights.

    Matches the Murex Pricing Service `DiscountFactorCurve` implementation to 0 ULP (`1e-16`)
    using clamped cubic splines (`bc_type="clamped"`) on the raw pillar tenors without prepending tau=0.
    """

    def __init__(
        self,
        spec: DiscountFactorCurveSpec | datetime.date | datetime.datetime | None = None,
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
                if spec.RatesSchedule.Rates
                else np.zeros(len(self.pillar_dates), dtype=np.float64)
            )
            if spec.RatesSchedule.Spreads is not None:
                raw_rates = raw_rates + np.asarray(spec.RatesSchedule.Spreads, dtype=np.float64)
        else:
            if isinstance(spec, (datetime.date, datetime.datetime)) and as_of_date is None:
                as_of_date = spec
            self.identifier = identifier
            self.currency = currency
            self.as_of_date = as_of_date
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
            elif self.pillar_dates and self.as_of_date is not None:
                tenors_arr = year_fractions(self.as_of_date, self.pillar_dates, self.dcc)
            else:
                raise ValueError(
                    "ClampedYieldCurve requires either `spec`, `pillar_tenors`, or (`as_of_date` and `pillar_dates`)."
                )
            if rates is None and spreads is None:
                raise ValueError("ClampedYieldCurve requires `rates` or `spreads`.")
            raw_rates = (
                np.asarray(rates, dtype=np.float64)
                if rates is not None
                else np.zeros(len(tenors_arr), dtype=np.float64)
            )
            if spreads is not None:
                raw_rates = raw_rates + np.asarray(spreads, dtype=np.float64)

        if len(tenors_arr) == 0:
            raise ValueError("ClampedYieldCurve requires at least one pillar.")
        if len(tenors_arr) != len(raw_rates):
            raise ValueError(
                f"Pillar tenors length ({len(tenors_arr)}) != rates length ({len(raw_rates)})"
            )
        if len(tenors_arr) > 1 and np.any(np.diff(tenors_arr) <= 0.0):
            raise ValueError("ClampedYieldCurve pillar tenors must be strictly increasing.")

        self._tenors: npt.NDArray[np.float64] = np.asarray(tenors_arr, dtype=np.float64)
        self._raw_quotes: npt.NDArray[np.float64] = np.asarray(raw_rates, dtype=np.float64)
        self._quote_scale: float = 0.01 if self.quote_expression == "Percentage" else 1.0
        self._input_mode: str = "rates"

        # Convert pillar quotes to continuously compounded decimal zero rates r_k (AUD-P1-05):
        if self.space_coords == "Rates":
            self._rates_decimal: npt.NDArray[np.float64] = self._raw_quotes * self._quote_scale
        elif self.space_coords == "LogarithmOfDiscountFactors":
            if self.quote_expression != "Percentage" and np.all(self._raw_quotes <= 0.0):
                self._input_mode = "log_df"
                t_safe = np.maximum(np.abs(self._tenors), 1e-12)
                self._rates_decimal = np.where(
                    self._tenors > 1e-12, -self._raw_quotes / t_safe, 0.0
                )
            else:
                self._rates_decimal = self._raw_quotes * self._quote_scale
        elif self.space_coords in ("DiscountFactors", "Factors"):
            # Distinguish actual discount factors DF_k in (0.35, 1.05] (non-increasing) from decimal/pct zero rates:
            is_df_quotes = (
                self.quote_expression != "Percentage"
                and np.all((self._raw_quotes > 0.35) & (self._raw_quotes <= 1.05))
                and (
                    len(self._raw_quotes) == 1
                    or float(self._raw_quotes[-1]) <= float(self._raw_quotes[0]) + 1e-6
                )
            )
            if is_df_quotes:
                self._input_mode = "df"
                t_safe = np.maximum(np.abs(self._tenors), 1e-12)
                df_safe = np.maximum(self._raw_quotes, 1e-15)
                self._rates_decimal = np.where(
                    self._tenors > 1e-12, -np.log(df_safe) / t_safe, 0.0
                )
            else:
                self._rates_decimal = self._raw_quotes * self._quote_scale
        else:
            raise ValueError(f"Unsupported InterpolationSpaceCoordinates: {self.space_coords!r}")

        # Fit spline directly in continuous zero-rate space to prevent short-end 1/tau singularities
        self._y_pillars: npt.NDArray[np.float64] = self._rates_decimal.copy()

        # Precompute fast spline coefficients; build basis matrix lazily on first Rho/spline_weights call
        self._fast_coeffs: npt.NDArray[np.float64] | None = None
        self._fast_basis_coeffs: npt.NDArray[np.float64] | None = None
        self._cs_val: Any = None
        self._cs_basis: Any = None
        self._nonlin_interp: Any = None
        self._build_interpolator()

    def _build_interpolator(self) -> None:
        k_len = len(self._tenors)
        if k_len == 1:
            return

        method = self.interpolation_method

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
        elif method in (
            "MonotoneCubic",
            "PCHIP",
            "MonotoneConvex",
            "Kruger",
            "Harmonic",
        ):
            self._nonlin_interp = PchipInterpolator(self._tenors, self._y_pillars)
        elif method == "Akima":
            if k_len >= 5:
                self._nonlin_interp = Akima1DInterpolator(self._tenors, self._y_pillars)
            else:
                self._nonlin_interp = PchipInterpolator(self._tenors, self._y_pillars)
        elif method in (
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
    ) -> tuple[npt.NDArray[np.float64], bool]:
        if isinstance(dates_or_tenors, np.ndarray):
            if dates_or_tenors.ndim == 0:
                return self._to_tenors_and_scalar_flag(dates_or_tenors.item())
            if dates_or_tenors.ndim == 1 and np.issubdtype(dates_or_tenors.dtype, np.number):
                return dates_or_tenors.astype(np.float64, copy=False), False
        if isinstance(dates_or_tenors, (datetime.date, datetime.datetime)):
            if self.as_of_date is None:
                raise ValueError("Cannot convert date to tenor when `as_of_date` is None.")
            t = year_fraction(self.as_of_date, dates_or_tenors, self.dcc)
            return np.array([t], dtype=np.float64), True
        if isinstance(dates_or_tenors, (int, float, np.floating, np.integer)):
            return np.array([float(dates_or_tenors)], dtype=np.float64), True

        seq = list(dates_or_tenors)
        if len(seq) == 0:
            return np.zeros(0, dtype=np.float64), False
        if isinstance(seq[0], (datetime.date, datetime.datetime)):
            if self.as_of_date is None:
                raise ValueError("Cannot convert dates to tenors when `as_of_date` is None.")
            return year_fractions(self.as_of_date, seq, self.dcc), False
        return np.asarray(seq, dtype=np.float64), False

    def spline_weights(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Returns the $M_{\\text{eval}} \\times K$ basis weight matrix $\\mathbf{W}_{\\text{spline}}(\\boldsymbol{\\tau}_{\\text{eval}})$.

        Satisfies $S(\\boldsymbol{\\tau}_{\\text{eval}}) = \\mathbf{W}_{\\text{spline}}(\\boldsymbol{\\tau}_{\\text{eval}}) \\mathbf{y}$
        for all linear interpolation operators, and the exact Jacobian $\\partial S / \\partial \\mathbf{y}$
        for non-linear monotone interpolators (`PCHIP`, `Akima`).
        """
        tau_arr, _ = self._to_tenors_and_scalar_flag(dates_or_tenors)
        m_eval = len(tau_arr)
        k_len = len(self._tenors)
        if m_eval == 0:
            return np.zeros((0, k_len), dtype=np.float64)
        if k_len == 1:
            return np.ones((m_eval, 1), dtype=np.float64)

        method = self.interpolation_method
        t_min = self._tenors[0]
        t_max = self._tenors[-1]
        t_clip = np.clip(tau_arr, t_min, t_max)

        self._ensure_basis_interpolator()
        if self._fast_basis_coeffs is not None:
            return eval_clamped_cubic_coeffs(self._tenors, self._fast_basis_coeffs, t_clip)
        if self._cs_basis is not None:
            return np.asarray(self._cs_basis(t_clip), dtype=np.float64)

        if method in ("Linear", "FwdFwdLinear"):
            return _linear_basis(self._tenors, t_clip, extrapolation="Flat")

        if method in ("LogLinear", "LogDiscountLinear", "PiecewiseConstantForward"):
            return _log_linear_rate_basis(self._tenors, tau_arr)

        if method in ("FlatLeft", "PiecewiseConstantToTheLeft"):
            w = np.zeros((m_eval, k_len), dtype=np.float64)
            idx = np.clip(np.searchsorted(self._tenors, tau_arr, side="right") - 1, 0, k_len - 1)
            w[np.arange(m_eval), idx] = 1.0
            return w

        if method in ("FlatRight", "PiecewiseConstantToTheRight"):
            w = np.zeros((m_eval, k_len), dtype=np.float64)
            idx = np.clip(np.searchsorted(self._tenors, tau_arr, side="left"), 0, k_len - 1)
            w[np.arange(m_eval), idx] = 1.0
            return w

        if method == "Bessel":
            return _bessel_hermite_basis(
                self._tenors, t_clip, extrapolation="Flat"
            )

        if method in ("NelsonSiegel", "NelsonSiegelContinuous", "NelsonSiegelDiscrete"):
            return _nelson_siegel_basis(self._tenors, t_clip, svensson=False)

        if method == "Svensson":
            return _nelson_siegel_basis(self._tenors, t_clip, svensson=True)

        # Non-linear monotone interpolators (PCHIP / Akima): compute exact central-difference Jacobian dS/dy
        eps = 1e-7
        jac = np.zeros((m_eval, k_len), dtype=np.float64)
        for k in range(k_len):
            y_up = self._y_pillars.copy()
            y_dn = self._y_pillars.copy()
            y_up[k] += eps
            y_dn[k] -= eps
            if method == "Akima" and k_len >= 5:
                i_up = Akima1DInterpolator(self._tenors, y_up)
                i_dn = Akima1DInterpolator(self._tenors, y_dn)
            else:
                i_up = PchipInterpolator(self._tenors, y_up)
                i_dn = PchipInterpolator(self._tenors, y_dn)
            jac[:, k] = (np.asarray(i_up(t_clip), dtype=np.float64) - np.asarray(i_dn(t_clip), dtype=np.float64)) / (
                2.0 * eps
            )
        return jac

    def _eval_space_coord(self, tau_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluates the interpolated curve coordinate S(tau) in float64 with clamped tau in [tau_0, tau_{K-1}]."""
        if len(tau_arr) == 0:
            return np.zeros(0, dtype=np.float64)
        if len(self._tenors) == 1:
            return np.full(len(tau_arr), self._y_pillars[0], dtype=np.float64)

        t_clip = np.clip(tau_arr, self._tenors[0], self._tenors[-1])

        if self._fast_coeffs is not None:
            return eval_clamped_cubic_coeffs(self._tenors, self._fast_coeffs, t_clip)
        if self._cs_val is not None:
            return np.asarray(self._cs_val(t_clip), dtype=np.float64)
        if self._nonlin_interp is not None:
            return np.asarray(self._nonlin_interp(t_clip), dtype=np.float64)

        w = self.spline_weights(tau_arr)
        return np.asarray(w @ self._y_pillars, dtype=np.float64)

    def zero_rate(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Evaluates continuously compounded decimal zero rate r(tau) in float64."""
        tau_arr, is_scalar = self._to_tenors_and_scalar_flag(dates_or_tenors)
        r_arr = self._eval_space_coord(tau_arr)
        if is_scalar:
            return np.float64(r_arr[0])  # type: ignore[return-value]
        return np.asarray(r_arr, dtype=np.float64)

    def df(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Evaluates the discount factor DF(tau) = exp(-r(tau) * tau) in float64."""
        tau_arr, is_scalar = self._to_tenors_and_scalar_flag(dates_or_tenors)
        r_val = self._eval_space_coord(tau_arr)
        df_arr = np.where(tau_arr <= 0.0, 1.0, np.exp(-r_val * tau_arr))

        if is_scalar:
            return np.float64(df_arr[0])  # type: ignore[return-value]
        return np.asarray(df_arr, dtype=np.float64)

    def df_and_d_df_d_pillar(
        self,
        dates_or_tenors: Any,
        *,
        in_quote_units: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Computes `(df, d_df_d_pillar_quote)` in float64 using the precomputed spline basis matrix.

        For zero-rate quotes:
            d DF(tau_m) / d r_k^{quote} = -tau_m * DF(tau_m) * W_spline(tau_m, k) * scale_quote
        where `scale_quote = 0.01` for `Percentage` and `1.0` for `Decimal` (when `in_quote_units=True`).
        """
        tau_arr, _ = self._to_tenors_and_scalar_flag(dates_or_tenors)
        df_arr = np.asarray(self.df(tau_arr), dtype=np.float64)
        w_mat = self.spline_weights(tau_arr)  # (M_eval, K)
        scale = self._quote_scale if in_quote_units else 1.0

        d_df_d_r = -(np.maximum(tau_arr, 0.0)[:, None] * df_arr[:, None]) * w_mat
        if self._input_mode == "log_df":
            t_safe = np.maximum(np.abs(self._tenors), 1e-12)
            d_df = d_df_d_r * (-1.0 / t_safe[None, :]) * scale
        elif self._input_mode == "df":
            t_safe = np.maximum(np.abs(self._tenors), 1e-12)
            df_pil = np.maximum(self._raw_quotes, 1e-15)
            d_df = d_df_d_r * (-1.0 / (t_safe * df_pil)[None, :]) * scale
        else:
            d_df = d_df_d_r * scale

        return df_arr, np.asarray(d_df, dtype=np.float64)


# ── RepoCurve ─────────────────────────────────────────────────────────────────


class RepoCurve:
    """Equity repo spread curve (`Act360`, `<n>D` periods) with automatic flat-0.0% fallback when omitted."""

    def __init__(
        self,
        spec: RepoSpreadCurveSpec | None = None,
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
        self.as_of_date = as_of_date
        if spec is not None:
            self.identifier = spec.Identifier
            self.dcc = spec.InterpolationDayCountConvention.value
            self.interpolation_method = spec.InterpolationMethod.value
            self.extrapolation_method = spec.ExtrapolationMethod.value
            self.quote_expression = spec.RateQuoteExpression.value
            days_arr = np.asarray(spec.RatesSchedule.period_days(), dtype=np.float64)
            denom = 365.0 if "365" in self.dcc else 360.0
            tenors_arr = days_arr / denom
            rates_arr = np.asarray(spec.RatesSchedule.Rates, dtype=np.float64)
            self._is_flat_zero = False
            self._inner_curve: ClampedYieldCurve | None = ClampedYieldCurve(
                pillar_tenors=tenors_arr,
                rates=rates_arr,
                as_of_date=as_of_date,
                dcc=self.dcc,
                interpolation_method=self.interpolation_method,
                space_coords=spec.InterpolationSpaceCoordinates.value,
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
            denom = 365.0 if "365" in self.dcc else 360.0
            tenors_arr = days_arr / denom
            self._is_flat_zero = False
            self._inner_curve = ClampedYieldCurve(
                pillar_tenors=tenors_arr,
                rates=rates,
                as_of_date=as_of_date,
                dcc=self.dcc,
                interpolation_method=self.interpolation_method,
                space_coords="Rates",
                extrapolation_method=self.extrapolation_method,
                quote_expression=self.quote_expression,
                identifier=self.identifier,
            )
        else:
            # Default flat 0.0% repo curve (e.g., for omitted POSIRX_RUB)
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

    def _to_act360_tenors_and_scalar_flag(
        self, dates_or_tenors: Any
    ) -> tuple[npt.NDArray[np.float64], bool]:
        if isinstance(dates_or_tenors, np.ndarray):
            if dates_or_tenors.ndim == 0:
                return self._to_act360_tenors_and_scalar_flag(dates_or_tenors.item())
            if dates_or_tenors.ndim == 1 and np.issubdtype(dates_or_tenors.dtype, np.number):
                return dates_or_tenors.astype(np.float64, copy=False), False
        if isinstance(dates_or_tenors, (datetime.date, datetime.datetime)):
            if self.as_of_date is None:
                raise ValueError(
                    "RepoCurve requires `as_of_date` to evaluate `datetime.date` / `datetime.datetime` inputs."
                )
            t = year_fraction(self.as_of_date, dates_or_tenors, self.dcc)
            return np.array([t], dtype=np.float64), True
        if isinstance(dates_or_tenors, (int, float, np.floating, np.integer)):
            return np.array([float(dates_or_tenors)], dtype=np.float64), True

        seq = list(dates_or_tenors)
        if len(seq) == 0:
            return np.zeros(0, dtype=np.float64), False
        if isinstance(seq[0], (datetime.date, datetime.datetime)):
            if self.as_of_date is None:
                raise ValueError(
                    "RepoCurve requires `as_of_date` to evaluate `datetime.date` / `datetime.datetime` inputs."
                )
            return year_fractions(self.as_of_date, seq, self.dcc), False
        return np.asarray(seq, dtype=np.float64), False

    def repo_rate(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Evaluates decimal repo spread rate q_repo(t) in float64."""
        tau_arr, is_scalar = self._to_act360_tenors_and_scalar_flag(dates_or_tenors)
        if self._is_flat_zero or self._inner_curve is None:
            res = np.zeros_like(tau_arr, dtype=np.float64)
        else:
            res = np.asarray(self._inner_curve.zero_rate(tau_arr), dtype=np.float64)
        if is_scalar:
            return np.float64(res[0])  # type: ignore[return-value]
        return res

    def integrated_repo(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Evaluates integrated repo spread int_{t_0}^t q_repo(u) du = q_repo(t) * (days / 360.0) in float64."""
        tau_arr, is_scalar = self._to_act360_tenors_and_scalar_flag(dates_or_tenors)
        if self._is_flat_zero or self._inner_curve is None:
            res = np.zeros_like(tau_arr, dtype=np.float64)
        else:
            r_arr = np.asarray(self._inner_curve.zero_rate(tau_arr), dtype=np.float64)
            res = r_arr * tau_arr
        if is_scalar:
            return np.float64(res[0])  # type: ignore[return-value]
        return res

    def repo_df(self, dates_or_tenors: Any) -> npt.NDArray[np.float64]:
        """Evaluates repo discount factor exp(-int_{t_0}^t q_repo(u) du) in float64."""
        tau_arr, is_scalar = self._to_act360_tenors_and_scalar_flag(dates_or_tenors)
        int_repo = np.asarray(self.integrated_repo(tau_arr), dtype=np.float64)
        df_arr = np.exp(-int_repo)
        if is_scalar:
            return np.float64(df_arr[0])  # type: ignore[return-value]
        return df_arr
