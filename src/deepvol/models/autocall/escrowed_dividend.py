"""escrowed_dividend.py — Buehler Escrowed/Affine Cash Dividend Forward & Leverage Engine (Task 1.1b).

Implements SPEC-DEEPVOL-EQ-AUTOCALL-001 §1.2.C and Symbolic Proofs W2 & W3 in strict
IEEE-754 float64 precision (`torch.float64` / `np.float64`):
  - Merges MarketDataSet.Dividends and Product.PredefinedDividends (PredefinedDividends overrides).
  - Defaults omitted RepoSpreadCurves entries (e.g. POSIRX_RUB) to a flat 0.0% Act360 curve (REM-S0-02).
  - Evaluates discrete cash dividend present values D_{i,pv}(t_0, t_m), escrowed spot floor
    S_i^*(t_0, t_m) = max(S_i(t_0) - D_{i,pv}(t_0, t_m), 0.01 * S_i(t_0)), forward prices F_i(t_0, t_m),
    dimensionless relative forward ratios F_rel^{(i)}(t_m) = F_i(t_0, t_m) / S_i(t_0), forward
    performances F_{P,i}(t_m) = F_i(t_0, t_m) / S_i(0), and cash-dividend leverage factors
    delta_cash^{(i)}(t_m) = S_i(t_0) / S_i^*(t_0, t_m) >= 1.0.
  - Preserves full PyTorch autograd differentiability with respect to `spots_t0` and per-asset
    dividend amount tensors `div_amounts` for exact Delta, Gamma, CashGamma, and DividendsSensitivities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch

from deepvol.models.autocall.seasoned_collapser import _to_utc_datetime, _year_fraction

_PERIOD_DAYS_RE = re.compile(r"^\s*(\d+)\s*([DWMYdwmy])\s*$")


def _parse_period_days(period_str: str | int | float) -> float:
    """Parses a Murex period string such as '31D', '90D', '1277D' into calendar days."""
    if isinstance(period_str, (int, float)):
        return float(period_str)
    m = _PERIOD_DAYS_RE.match(str(period_str))
    if not m:
        return float(str(period_str).rstrip("Dd"))
    val = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "D":
        return val
    if unit == "W":
        return val * 7.0
    if unit == "M":
        return val * 30.0
    if unit == "Y":
        return val * 365.0
    return val


@dataclass(frozen=True)
class ParsedDividendEvent:
    """Single discrete cash dividend event for an underlying asset."""

    ex_date: datetime
    payment_date: datetime
    amount: float
    tau_ex_act365: float
    tau_pay_act365: float
    tau_ex_act360: float
    pv_discount_weight: float


@dataclass
class EscrowedDividendOutput:
    """Differentiable torch.float64 outputs of the Buehler escrowed dividend forward engine."""

    identifiers: list[str]
    as_of_date: datetime
    observation_dates: list[datetime]

    # Tensors of shape (D, M_rem) in torch.float64
    div_pv: torch.Tensor
    escrowed_spot: torch.Tensor
    forward_price: torch.Tensor
    forward_rel: torch.Tensor
    forward_perf: torch.Tensor
    cash_div_leverage: torch.Tensor
    escrowed_div_fraction: torch.Tensor

    # Tensors of shape (D, M_rem) for financing and repo discount factors
    df_fin_obs: torch.Tensor
    df_repo_obs: torch.Tensor
    r_tot_obs: torch.Tensor

    # Average repo spread per asset over the remaining horizon: shape (D,) in torch.float64
    repo_spread_avg: torch.Tensor

    # Per-asset differentiable dividend amount tensors and parsed events for DividendsSensitivities
    dividend_events: dict[str, list[ParsedDividendEvent]] = field(default_factory=dict)
    dividend_amount_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    dividend_pv_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    dividend_obs_incidence: dict[str, torch.Tensor] = field(default_factory=dict)


def build_repo_rate_evaluator(
    repo_spec: Any | None,
    as_of_date: datetime | date | str,
) -> Callable[[datetime | float], float]:
    """Builds a float64 repo rate evaluator q_repo(t) from a RepoSpreadCurve spec or dict.

    Per REM-S0-02: if repo_spec is None (e.g. POSIRX_RUB omitted in RepoSpreadCurves),
    deterministically defaults to a flat 0.0% Act360 repo spread curve.
    """
    if repo_spec is None:
        def fn_zero(_t: datetime | float) -> float:
            return 0.0

        fn_zero._eval_tau_batch = lambda tau_arr: np.zeros(len(tau_arr), dtype=np.float64)  # type: ignore[attr-defined]
        return fn_zero

    if hasattr(repo_spec, "repo_rate") and callable(repo_spec.repo_rate):
        def fn_rc(dt_or_tau: datetime | float) -> float:
            return float(np.asarray(repo_spec.repo_rate(dt_or_tau), dtype=np.float64).reshape(-1)[0])

        fn_rc._eval_tau_batch = lambda tau_arr: np.asarray(repo_spec.repo_rate(tau_arr), dtype=np.float64).reshape(-1)  # type: ignore[attr-defined]
        return fn_rc

    # If repo_spec already exposes a .zero_rate(...) method, use it
    if hasattr(repo_spec, "zero_rate") and callable(repo_spec.zero_rate):
        def _eval_curve_obj(dt_or_tau: datetime | float) -> float:
            if isinstance(dt_or_tau, datetime):
                res = repo_spec.zero_rate([dt_or_tau])
            else:
                res = repo_spec.zero_rate(np.asarray([float(dt_or_tau)], dtype=np.float64))
            return float(np.asarray(res, dtype=np.float64).reshape(-1)[0])

        _eval_curve_obj._eval_tau_batch = lambda tau_arr: np.asarray(repo_spec.zero_rate(tau_arr), dtype=np.float64).reshape(-1)  # type: ignore[attr-defined]
        return _eval_curve_obj

    t0_dt = _to_utc_datetime(as_of_date)

    # Extract fields from dict or Pydantic model
    if isinstance(repo_spec, Mapping):
        raw_qe = repo_spec.get("RateQuoteExpression", repo_spec.get("rate_quote_expression", "Percentage"))
        quote_expr = str(getattr(raw_qe, "value", raw_qe))
        raw_dcc = repo_spec.get(
            "InterpolationDayCountConvention",
            repo_spec.get("day_count_convention", "Act360"),
        )
        dcc = str(getattr(raw_dcc, "value", raw_dcc))
        raw_interp = repo_spec.get("InterpolationMethod", repo_spec.get("interpolation_method", "CubicSpline"))
        interp_method = str(getattr(raw_interp, "value", raw_interp))
        raw_space = repo_spec.get("InterpolationSpaceCoordinates", repo_spec.get("space_coords", "Rates"))
        space_coords = str(getattr(raw_space, "value", raw_space))
        sched = repo_spec.get("RatesSchedule", repo_spec)
        if isinstance(sched, Mapping):
            raw_periods = sched.get("Periods", sched.get("periods_days", sched.get("Dates", [])))
            raw_rates = sched.get("Rates", sched.get("rates_pct", sched.get("rates", [])))
        else:
            raw_periods = getattr(sched, "Periods", getattr(sched, "periods_days", getattr(sched, "Dates", [])))
            raw_rates = getattr(sched, "Rates", getattr(sched, "rates_pct", getattr(sched, "rates", [])))
    else:
        raw_qe = getattr(repo_spec, "RateQuoteExpression", getattr(repo_spec, "rate_quote_expression", "Percentage"))
        quote_expr = str(getattr(raw_qe, "value", raw_qe))
        raw_dcc = getattr(
            repo_spec,
            "InterpolationDayCountConvention",
            getattr(repo_spec, "day_count_convention", "Act360"),
        )
        dcc = str(getattr(raw_dcc, "value", raw_dcc))
        raw_interp = getattr(repo_spec, "InterpolationMethod", getattr(repo_spec, "interpolation_method", "CubicSpline"))
        interp_method = str(getattr(raw_interp, "value", raw_interp))
        raw_space = getattr(repo_spec, "InterpolationSpaceCoordinates", getattr(repo_spec, "space_coords", "Rates"))
        space_coords = str(getattr(raw_space, "value", raw_space))
        sched = getattr(repo_spec, "RatesSchedule", repo_spec)
        raw_periods = getattr(sched, "Periods", getattr(sched, "periods_days", getattr(sched, "Dates", [])))
        raw_rates = getattr(sched, "Rates", getattr(sched, "rates_pct", getattr(sched, "rates", [])))

    if interp_method != "CubicSpline" or space_coords != "Rates":
        try:
            from deepvol.market.eq_curves import RepoCurve
            from deepvol.api.schemas_eq_autocall import RepoSpreadCurveSpec

            if isinstance(repo_spec, RepoSpreadCurveSpec):
                rc = RepoCurve(repo_spec, as_of_date=t0_dt)

                def fn_rc2(dt_or_tau: datetime | float) -> float:
                    return float(np.asarray(rc.repo_rate(dt_or_tau), dtype=np.float64).reshape(-1)[0])

                fn_rc2._eval_tau_batch = lambda tau_arr: np.asarray(rc.repo_rate(tau_arr), dtype=np.float64).reshape(-1)  # type: ignore[attr-defined]
                return fn_rc2
        except Exception:
            pass

    if not raw_rates:
        def fn_empty(_t: datetime | float) -> float:
            return 0.0

        fn_empty._eval_tau_batch = lambda tau_arr: np.zeros(len(tau_arr), dtype=np.float64)  # type: ignore[attr-defined]
        return fn_empty

    scale = 0.01 if "perc" in quote_expr.lower() else 1.0
    rates_arr = np.asarray([float(r) * scale for r in raw_rates], dtype=np.float64)

    denom = 360.0 if "360" in dcc else 365.0
    tenors_list: list[float] = []
    for p in raw_periods:
        if isinstance(p, (datetime, date)):
            tenors_list.append(_year_fraction(t0_dt, p, dcc))
        else:
            tenors_list.append(_parse_period_days(p) / denom)
    tenors_arr = np.asarray(tenors_list, dtype=np.float64)

    if len(tenors_arr) <= 1 or np.allclose(rates_arr, rates_arr[0], atol=1e-15):
        const_rate = float(rates_arr[0])

        def fn_const(_t: datetime | float) -> float:
            return const_rate

        fn_const._eval_tau_batch = lambda tau_arr: np.full(len(tau_arr), const_rate, dtype=np.float64)  # type: ignore[attr-defined]
        return fn_const

    from deepvol.market.eq_curves import eval_clamped_cubic_coeffs, fast_clamped_cubic_coeffs

    coeffs = fast_clamped_cubic_coeffs(tenors_arr, rates_arr)
    t_min = float(tenors_arr[0])
    t_max = float(tenors_arr[-1])

    def _eval_repo(dt_or_tau: datetime | float) -> float:
        if isinstance(dt_or_tau, (datetime, date)):
            tau = _year_fraction(t0_dt, dt_or_tau, dcc)
        else:
            tau = float(dt_or_tau)
        tau_c = min(max(tau, t_min), t_max)
        return float(eval_clamped_cubic_coeffs(tenors_arr, coeffs, np.array([tau_c], dtype=np.float64))[0])

    def _eval_repo_tau_batch(tau_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if len(tau_arr) == 0:
            return np.zeros(0, dtype=np.float64)
        tau_c = np.clip(tau_arr, t_min, t_max)
        return np.asarray(eval_clamped_cubic_coeffs(tenors_arr, coeffs, tau_c), dtype=np.float64)

    _eval_repo._eval_tau_batch = _eval_repo_tau_batch  # type: ignore[attr-defined]
    return _eval_repo


def _extract_curve_dcc(curve_obj: Any, default_dcc: str = "Act365") -> str:
    if curve_obj is None:
        return default_dcc
    if isinstance(curve_obj, Mapping):
        raw = curve_obj.get(
            "InterpolationDayCountConvention",
            curve_obj.get("DayCountConvention", curve_obj.get("day_count_convention", default_dcc)),
        )
        return str(getattr(raw, "value", raw))
    for attr in ("dcc", "InterpolationDayCountConvention", "DayCountConvention", "day_count_convention"):
        if hasattr(curve_obj, attr):
            val = getattr(curve_obj, attr)
            if val is not None:
                return str(getattr(val, "value", val))
    return default_dcc


def build_discount_rate_evaluator(
    discount_curve: Any,
    as_of_date: datetime | date | str,
) -> tuple[Callable[[datetime], float], Callable[[datetime], float]]:
    """Returns (r_zero_fn(dt), df_fn(dt)) in float64 from a ClampedYieldCurve, spec, or dict."""
    t0_dt = _to_utc_datetime(as_of_date)
    curve_dcc = _extract_curve_dcc(discount_curve, "Act365")

    if hasattr(discount_curve, "df") and callable(discount_curve.df):
        def _df_from_obj(dt: datetime) -> float:
            dt_utc = _to_utc_datetime(dt)
            if dt_utc <= t0_dt:
                return 1.0
            val = discount_curve.df(dt_utc)
            return float(np.asarray(val, dtype=np.float64).reshape(-1)[0])

        def _r_from_obj(dt: datetime) -> float:
            dt_utc = _to_utc_datetime(dt)
            tau = _year_fraction(t0_dt, dt_utc, curve_dcc)
            if tau <= 0.0:
                return 0.0
            if hasattr(discount_curve, "zero_rate") and callable(discount_curve.zero_rate):
                val = discount_curve.zero_rate(dt_utc)
                return float(np.asarray(val, dtype=np.float64).reshape(-1)[0])
            df_val = _df_from_obj(dt_utc)
            return float(-np.log(max(df_val, 1e-15)) / tau)

        def _df_tau_batch_obj(tau_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            if len(tau_arr) == 0:
                return np.zeros(0, dtype=np.float64)
            return np.where(
                tau_arr <= 0.0,
                1.0,
                np.asarray(discount_curve.df(tau_arr), dtype=np.float64).reshape(-1),
            )

        if hasattr(discount_curve, "zero_rate") and callable(discount_curve.zero_rate):
            def _r_tau_batch_obj(tau_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                if len(tau_arr) == 0:
                    return np.zeros(0, dtype=np.float64)
                return np.where(
                    tau_arr <= 0.0,
                    0.0,
                    np.asarray(discount_curve.zero_rate(tau_arr), dtype=np.float64).reshape(-1),
                )
            _r_from_obj._eval_tau_batch = _r_tau_batch_obj  # type: ignore[attr-defined]
        _df_from_obj._eval_tau_batch = _df_tau_batch_obj  # type: ignore[attr-defined]

        return _r_from_obj, _df_from_obj

    if isinstance(discount_curve, Mapping):
        raw_qe = discount_curve.get("RateQuoteExpression", discount_curve.get("rate_quote_expression", "Percentage"))
        quote_expr = str(getattr(raw_qe, "value", raw_qe))
        raw_dcc = discount_curve.get(
            "InterpolationDayCountConvention",
            discount_curve.get("day_count_convention", "Act365"),
        )
        dcc = str(getattr(raw_dcc, "value", raw_dcc))
        sched = discount_curve.get("RatesSchedule", discount_curve)
        if isinstance(sched, Mapping):
            raw_dates = sched.get("Dates", sched.get("dates", []))
            raw_rates = sched.get("Rates", sched.get("rates_pct", sched.get("rates", [])))
        else:
            raw_dates = getattr(sched, "Dates", getattr(sched, "dates", []))
            raw_rates = getattr(sched, "Rates", getattr(sched, "rates_pct", getattr(sched, "rates", [])))
    else:
        raw_qe = getattr(
            discount_curve,
            "RateQuoteExpression",
            getattr(discount_curve, "rate_quote_expression", "Percentage"),
        )
        quote_expr = str(getattr(raw_qe, "value", raw_qe))
        raw_dcc = getattr(
            discount_curve,
            "InterpolationDayCountConvention",
            getattr(discount_curve, "day_count_convention", "Act365"),
        )
        dcc = str(getattr(raw_dcc, "value", raw_dcc))
        sched = getattr(discount_curve, "RatesSchedule", discount_curve)
        raw_dates = getattr(sched, "Dates", getattr(sched, "dates", []))
        raw_rates = getattr(sched, "Rates", getattr(sched, "rates_pct", getattr(sched, "rates", [])))

    scale = 0.01 if "perc" in quote_expr.lower() else 1.0
    tenors = np.asarray([_year_fraction(t0_dt, d, dcc) for d in raw_dates], dtype=np.float64)
    rates = np.asarray([float(r) * scale for r in raw_rates], dtype=np.float64)

    if len(tenors) == 0:
        def r_z(_dt: datetime) -> float:
            return 0.0

        def df_z(_dt: datetime) -> float:
            return 1.0

        r_z._eval_tau_batch = lambda tau_arr: np.zeros(len(tau_arr), dtype=np.float64)  # type: ignore[attr-defined]
        df_z._eval_tau_batch = lambda tau_arr: np.ones(len(tau_arr), dtype=np.float64)  # type: ignore[attr-defined]
        return r_z, df_z
    if len(tenors) == 1 or np.allclose(rates, rates[0], atol=1e-15):
        r0 = float(rates[0])

        def _r_flat(_dt: datetime) -> float:
            return r0

        def _df_flat(dt: datetime) -> float:
            tau = _year_fraction(t0_dt, dt, dcc)
            return 1.0 if tau <= 0.0 else float(np.exp(-r0 * tau))

        _r_flat._eval_tau_batch = lambda tau_arr: np.full(len(tau_arr), r0, dtype=np.float64)  # type: ignore[attr-defined]
        _df_flat._eval_tau_batch = lambda tau_arr: np.where(tau_arr <= 0.0, 1.0, np.exp(-r0 * tau_arr))  # type: ignore[attr-defined]
        return _r_flat, _df_flat

    from deepvol.market.eq_curves import eval_clamped_cubic_coeffs, fast_clamped_cubic_coeffs

    coeffs = fast_clamped_cubic_coeffs(tenors, rates)
    t_min = float(tenors[0])
    t_max = float(tenors[-1])

    def _r_fn(dt: datetime) -> float:
        tau = _year_fraction(t0_dt, dt, dcc)
        tau_c = min(max(tau, t_min), t_max)
        return float(eval_clamped_cubic_coeffs(tenors, coeffs, np.array([tau_c], dtype=np.float64))[0])

    def _df_fn(dt: datetime) -> float:
        tau = _year_fraction(t0_dt, dt, dcc)
        if tau <= 0.0:
            return 1.0
        r_val = _r_fn(dt)
        return float(np.exp(-r_val * tau))

    def _r_tau_batch(tau_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if len(tau_arr) == 0:
            return np.zeros(0, dtype=np.float64)
        tau_c = np.clip(tau_arr, t_min, t_max)
        return np.asarray(eval_clamped_cubic_coeffs(tenors, coeffs, tau_c), dtype=np.float64)

    def _df_tau_batch(tau_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if len(tau_arr) == 0:
            return np.zeros(0, dtype=np.float64)
        r_arr = _r_tau_batch(tau_arr)
        return np.where(tau_arr <= 0.0, 1.0, np.exp(-r_arr * tau_arr))

    _r_fn._eval_tau_batch = _r_tau_batch  # type: ignore[attr-defined]
    _df_fn._eval_tau_batch = _df_tau_batch  # type: ignore[attr-defined]
    return _r_fn, _df_fn


def _parse_div_item(item: Any) -> tuple[datetime, datetime, float]:
    """Extracts (ex_date, payment_date, amount) from a DividendItemSpec or dict."""
    if isinstance(item, Mapping):
        ex_val: Any = item.get("ExDate", item.get("ex_date"))
        ex_d = _to_utc_datetime(ex_val)
        pay_val: Any = item.get("PaymentDate", item.get("payment_date", ex_d))
        pay_d = _to_utc_datetime(pay_val)
        amt = float(item.get("Amount", item.get("amount", 0.0)))
        return ex_d, pay_d, amt

    ex_raw: Any = getattr(item, "ExDate", None)
    if ex_raw is None:
        ex_raw = getattr(item, "ex_date")
    ex_d = _to_utc_datetime(ex_raw)
    pay_raw: Any = getattr(item, "PaymentDate", None)
    if pay_raw is None:
        pay_raw = getattr(item, "payment_date", ex_d)
    pay_d = _to_utc_datetime(pay_raw if pay_raw is not None else ex_d)
    amt_raw: Any = getattr(item, "Amount", None)
    if amt_raw is None:
        amt_raw = getattr(item, "amount", 0.0)
    return ex_d, pay_d, float(amt_raw if amt_raw is not None else 0.0)


def merge_dividend_schedules(
    identifiers: Sequence[str],
    market_dividends: Mapping[str, Any] | None = None,
    predefined_dividends: Mapping[str, Any] | None = None,
) -> dict[str, list[tuple[datetime, datetime, float]]]:
    """Merges MarketDataSet.Dividends and Product.PredefinedDividends per asset.

    Per SPEC-DEEPVOL-EQ-AUTOCALL-001 §1.2.C:
      If an asset has a non-empty entry in `PredefinedDividends`, it overrides
      `MarketDataSet.Dividends` for that asset.
    """
    merged: dict[str, list[tuple[datetime, datetime, float]]] = {}

    for ticker in identifiers:
        source_obj: Any = None
        if predefined_dividends and ticker in predefined_dividends and predefined_dividends[ticker] is not None:
            source_obj = predefined_dividends[ticker]
        elif market_dividends and ticker in market_dividends and market_dividends[ticker] is not None:
            source_obj = market_dividends[ticker]

        events: list[tuple[datetime, datetime, float]] = []
        if source_obj is not None:
            if isinstance(source_obj, Sequence) and not isinstance(source_obj, (str, bytes, Mapping)):
                for item in source_obj:
                    events.append(_parse_div_item(item))
            elif isinstance(source_obj, Mapping):
                if "CashDividendList" in source_obj or "cash_dividend_list" in source_obj:
                    cd_list = source_obj.get("CashDividendList", source_obj.get("cash_dividend_list", []))
                    for item in cd_list:
                        events.append(_parse_div_item(item))
                elif "ExDates" in source_obj or "ex_dates" in source_obj:
                    ex_dates = source_obj.get("ExDates", source_obj.get("ex_dates", []))
                    pay_dates = source_obj.get("PaymentDates", source_obj.get("payment_dates", ex_dates))
                    amounts = source_obj.get("Amounts", source_obj.get("amounts", []))
                    for ex_d, pay_d, amt in zip(ex_dates, pay_dates, amounts):
                        events.append((_to_utc_datetime(ex_d), _to_utc_datetime(pay_d), float(amt)))
            else:
                if hasattr(source_obj, "CashDividendList") or hasattr(source_obj, "cash_dividend_list"):
                    cd_list = getattr(source_obj, "CashDividendList", None)
                    if cd_list is None:
                        cd_list = getattr(source_obj, "cash_dividend_list", [])
                    for item in cd_list:
                        events.append(_parse_div_item(item))
                elif hasattr(source_obj, "ExDates") or hasattr(source_obj, "ex_dates"):
                    ex_dates = getattr(source_obj, "ExDates", None)
                    if ex_dates is None:
                        ex_dates = getattr(source_obj, "ex_dates", [])
                    pay_dates = getattr(source_obj, "PaymentDates", None)
                    if pay_dates is None:
                        pay_dates = getattr(source_obj, "payment_date", ex_dates)
                    amounts = getattr(source_obj, "Amounts", None)
                    if amounts is None:
                        amounts = getattr(source_obj, "amounts", [])
                    for ex_d, pay_d, amt in zip(ex_dates, pay_dates, amounts):
                        events.append((_to_utc_datetime(ex_d), _to_utc_datetime(pay_d), float(amt)))

        merged[str(ticker)] = events

    return merged


def _infer_asset_currency(ticker: str, default_ccy: str) -> str:
    parts = ticker.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 3 and parts[1].isalpha():
        return parts[1].upper()
    return default_ccy.upper()


def _eval_fx_atm_vol(fx_spec: Any, tau_eval: float) -> float:
    if isinstance(fx_spec, Mapping):
        tenors_raw = fx_spec.get("Tenors", fx_spec.get("tenors", []))
        deltas_raw = fx_spec.get("Deltas", fx_spec.get("deltas", ["ATM"]))
        quotes_raw = fx_spec.get("VolatilityQuotes", fx_spec.get("volatility_quotes", []))
        raw_dcc = fx_spec.get("DayCountConvention", fx_spec.get("day_count_convention", "Act365"))
    else:
        tenors_raw = getattr(fx_spec, "Tenors", getattr(fx_spec, "tenors", []))
        deltas_raw = getattr(fx_spec, "Deltas", getattr(fx_spec, "deltas", ["ATM"]))
        quotes_raw = getattr(fx_spec, "VolatilityQuotes", getattr(fx_spec, "volatility_quotes", []))
        raw_dcc = getattr(fx_spec, "DayCountConvention", getattr(fx_spec, "day_count_convention", "Act365"))
    dcc = str(getattr(raw_dcc, "value", raw_dcc))
    denom = 252.0 if "252" in dcc else (360.0 if "360" in dcc else 365.0)
    if not tenors_raw or not quotes_raw:
        return 0.0
    deltas_list = [str(d).upper() for d in deltas_raw]
    atm_idx = deltas_list.index("ATM") if "ATM" in deltas_list else len(deltas_list) // 2
    t_arr = np.asarray([_parse_period_days(t) / denom for t in tenors_raw], dtype=np.float64)
    v_arr = np.asarray([float(row[atm_idx]) for row in quotes_raw], dtype=np.float64)
    if len(t_arr) == 1:
        return float(v_arr[0])
    return float(np.interp(float(tau_eval), t_arr, v_arr))


def _eval_eq_atm_vol(eq_spec: Any, as_of_dt: datetime, obs_dt: datetime, spot_val: float) -> float:
    if hasattr(eq_spec, "iv") and callable(eq_spec.iv):
        tau = _year_fraction(as_of_dt, obs_dt, "Act365")
        return float(eq_spec.iv(tau, spot_val))
    if isinstance(eq_spec, Mapping):
        strikes_raw = eq_spec.get("Strikes", eq_spec.get("strikes", []))
        quotes_raw = eq_spec.get("VolatilityQuotes", eq_spec.get("volatility_quotes", []))
        exp_dates = eq_spec.get("ExpirationDates", eq_spec.get("expiration_dates", []))
        raw_dcc = eq_spec.get("DayCountConvention", eq_spec.get("day_count_convention", "Act365"))
    else:
        strikes_raw = getattr(eq_spec, "Strikes", getattr(eq_spec, "strikes", []))
        quotes_raw = getattr(eq_spec, "VolatilityQuotes", getattr(eq_spec, "volatility_quotes", []))
        exp_dates = getattr(eq_spec, "ExpirationDates", getattr(eq_spec, "expiration_dates", []))
        raw_dcc = getattr(eq_spec, "DayCountConvention", getattr(eq_spec, "day_count_convention", "Act365"))
    if not strikes_raw or not quotes_raw or not exp_dates:
        return 0.0
    dcc = str(getattr(raw_dcc, "value", raw_dcc))
    t_eval = _year_fraction(as_of_dt, obs_dt, dcc)
    tenors = np.asarray([_year_fraction(as_of_dt, d, dcc) for d in exp_dates], dtype=np.float64)
    strikes = np.asarray(strikes_raw, dtype=np.float64)
    quotes = np.asarray(quotes_raw, dtype=np.float64)
    slice_vols = np.array([float(np.interp(spot_val, strikes, row)) for row in quotes], dtype=np.float64)
    if len(tenors) == 1:
        return float(slice_vols[0])
    return float(np.interp(t_eval, tenors, slice_vols))


def _resolve_quanto_drift_rate(
    *,
    ticker: str,
    spot_val: float,
    as_of_dt: datetime,
    obs_dt: datetime,
    payment_currency: str,
    vol_surfaces: Mapping[str, Any] | None,
    correlations: Mapping[str, float] | None,
) -> float:
    """Computes instantaneous quanto drift rate -rho_{S,FX} * sigma_{S,ATM} * sigma_{FX,ATM} when AssetCurrency != PayoffCurrency."""
    if not vol_surfaces or not correlations:
        return 0.0
    pay_ccy = payment_currency.strip().upper()
    asset_ccy = _infer_asset_currency(ticker, pay_ccy)
    if asset_ccy == pay_ccy:
        return 0.0

    # Find matching FXVolatilitySmileCollection in vol_surfaces
    fx_candidates = (
        f"{asset_ccy}_{pay_ccy}",
        f"{asset_ccy}{pay_ccy}",
        f"{asset_ccy}/{pay_ccy}",
        f"{pay_ccy}_{asset_ccy}",
        f"{pay_ccy}{asset_ccy}",
        f"{pay_ccy}/{asset_ccy}",
    )
    fx_id: str | None = None
    fx_spec: Any = None
    fx_sign: float = 1.0
    for cand in fx_candidates:
        if cand in vol_surfaces:
            fx_id = cand
            fx_spec = vol_surfaces[cand]
            if cand.startswith(pay_ccy):
                fx_sign = -1.0
            break
    if fx_spec is None:
        for k_id, v_surf in vol_surfaces.items():
            t_name = (
                v_surf.get("TypeName")
                if isinstance(v_surf, Mapping)
                else getattr(v_surf, "TypeName", None)
            )
            if t_name == "FXVolatilitySmileCollection" and asset_ccy in k_id.upper() and pay_ccy in k_id.upper():
                fx_id = k_id
                fx_spec = v_surf
                if k_id.upper().index(pay_ccy) < k_id.upper().index(asset_ccy):
                    fx_sign = -1.0
                break
    if fx_spec is None or fx_id is None:
        return 0.0

    # Look up correlation rho_{S, FX}
    k1 = f"{ticker}&{fx_id}"
    k2 = f"{fx_id}&{ticker}"
    if k1 in correlations:
        rho_s_fx = float(correlations[k1]) * fx_sign
    elif k2 in correlations:
        rho_s_fx = float(correlations[k2]) * fx_sign
    else:
        return 0.0

    tau_365 = _year_fraction(as_of_dt, obs_dt, "Act365")
    sig_fx = _eval_fx_atm_vol(fx_spec, tau_365)
    eq_spec = vol_surfaces.get(ticker)
    sig_eq = _eval_eq_atm_vol(eq_spec, as_of_dt, obs_dt, spot_val) if eq_spec is not None else 0.0
    return -rho_s_fx * sig_eq * sig_fx


def compute_escrowed_dividend_forwards(
    *,
    identifiers: Sequence[str],
    as_of_date: datetime | date | str,
    observation_dates: Sequence[datetime | date | str],
    spots_t0: torch.Tensor | Sequence[float],
    ref_spots: torch.Tensor | Sequence[float],
    discount_curve: Any,
    repo_curves: Mapping[str, Any] | None = None,
    market_dividends: Mapping[str, Any] | None = None,
    predefined_dividends: Mapping[str, Any] | None = None,
    dividend_amount_overrides: Mapping[str, torch.Tensor] | None = None,
    escrow_floor_fraction: float = 0.01,
    payment_currency: str = "RUB",
    vol_surfaces: Mapping[str, Any] | None = None,
    correlations: Mapping[str, float] | None = None,
) -> EscrowedDividendOutput:
    """Computes Buehler escrowed dividend forwards, relative forwards, and cash-dividend leverage in torch.float64.

    Exact Murex/Pricing Service forward & dividend discounting convention verified in E2/E3:
      r_tot,i(t) = r_fin(t) * tau_disc(t_0, t) - q_repo,i(t) * tau_repo(t_0, t) - rho_{S,FX} * sigma_S * sigma_FX * tau(t_0, t)
      D_{i,pv}(t_0, t_m) = sum_{t_0 < tau_{i,k}^{ex} <= t_m} d_{i,k} * DF_fin(t_0, tau_{i,k}^{pay}) * exp(q_repo,i * tau_repo(t_0, tau_{i,k}^{ex}))
      S_i^*(t_0, t_m) = max(S_i(t_0) - D_{i,pv}(t_0, t_m), 0.01 * S_i(t_0))
      F_i(t_0, t_m) = S_i^*(t_0, t_m) * exp(r_tot,i(t_m))
      F_rel^{(i)}(t_m) = F_i(t_0, t_m) / S_i(t_0)
      F_{P,i}(t_m) = F_i(t_0, t_m) / S_i(0) = M_{0,i} * F_rel^{(i)}(t_m)
      delta_cash^{(i)}(t_m) = S_i(t_0) / S_i^*(t_0, t_m) >= 1.0
    """
    t0_dt = _to_utc_datetime(as_of_date)
    obs_dts = [_to_utc_datetime(d) for d in observation_dates]
    id_list = [str(x) for x in identifiers]
    d_assets = len(id_list)
    m_rem = len(obs_dts)

    if isinstance(spots_t0, torch.Tensor):
        spots_t = spots_t0.to(dtype=torch.float64)
    else:
        spots_t = torch.as_tensor(spots_t0, dtype=torch.float64)
    device = spots_t.device

    if isinstance(ref_spots, torch.Tensor):
        ref_t = ref_spots.to(dtype=torch.float64, device=device)
    else:
        ref_t = torch.as_tensor(ref_spots, dtype=torch.float64, device=device)

    disc_dcc = _extract_curve_dcc(discount_curve, "Act365")
    r_fin_fn, df_fin_fn = build_discount_rate_evaluator(discount_curve, t0_dt)
    merged_divs = merge_dividend_schedules(id_list, market_dividends, predefined_dividends)

    if m_rem == 0:
        empty_dm = torch.zeros((d_assets, 0), dtype=torch.float64, device=device)
        return EscrowedDividendOutput(
            identifiers=id_list,
            as_of_date=t0_dt,
            observation_dates=obs_dts,
            div_pv=empty_dm,
            escrowed_spot=empty_dm,
            forward_price=empty_dm,
            forward_rel=empty_dm,
            forward_perf=empty_dm,
            cash_div_leverage=empty_dm,
            escrowed_div_fraction=empty_dm,
            df_fin_obs=empty_dm,
            df_repo_obs=empty_dm,
            r_tot_obs=empty_dm,
            repo_spread_avg=torch.zeros(d_assets, dtype=torch.float64, device=device),
            dividend_events={ticker: [] for ticker in id_list},
            dividend_amount_tensors={},
            dividend_pv_weights={},
            dividend_obs_incidence={},
        )

    df_fin_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    df_repo_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    r_tot_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    repo_avg_np = np.zeros(d_assets, dtype=np.float64)

    dividend_events_map: dict[str, list[ParsedDividendEvent]] = {}
    div_amount_tensors: dict[str, torch.Tensor] = {}
    div_pv_weights_map: dict[str, torch.Tensor] = {}
    div_incidence_map: dict[str, torch.Tensor] = {}
    div_pv_rows: list[torch.Tensor] = []

    spots_cpu = spots_t.detach().cpu().numpy()

    tau_disc_obs = np.array([_year_fraction(t0_dt, d, disc_dcc) for d in obs_dts], dtype=np.float64)
    tau_365_obs = (
        tau_disc_obs
        if disc_dcc == "Act365"
        else np.array([_year_fraction(t0_dt, d, "Act365") for d in obs_dts], dtype=np.float64)
    )
    tau_360_obs = (
        tau_disc_obs
        if disc_dcc == "Act360"
        else np.array([_year_fraction(t0_dt, d, "Act360") for d in obs_dts], dtype=np.float64)
    )

    r_batch_fn = getattr(r_fin_fn, "_eval_tau_batch", None)
    df_batch_fn = getattr(df_fin_fn, "_eval_tau_batch", None)
    if r_batch_fn is not None:
        r_obs_arr = r_batch_fn(tau_disc_obs)
    else:
        r_obs_arr = np.array([r_fin_fn(d) for d in obs_dts], dtype=np.float64)
    r_times_tau_obs = r_obs_arr * tau_disc_obs
    df_fin_obs_row = np.exp(-r_times_tau_obs)
    df_fin_np[:] = df_fin_obs_row[None, :]
    pay_ccy_upper = payment_currency.strip().upper()
    has_quanto_data = bool(vol_surfaces and correlations)

    for i, ticker in enumerate(id_list):
        repo_spec_i = repo_curves.get(ticker) if repo_curves else None
        repo_dcc = _extract_curve_dcc(repo_spec_i, "Act360")
        q_repo_fn = build_repo_rate_evaluator(repo_spec_i, t0_dt)
        q_batch_fn = getattr(q_repo_fn, "_eval_tau_batch", None)
        spot_i_val = float(spots_cpu[i])

        if repo_dcc == "Act360":
            tau_repo_obs = tau_360_obs
        elif repo_dcc == "Act365":
            tau_repo_obs = tau_365_obs
        else:
            tau_repo_obs = np.array([_year_fraction(t0_dt, d, repo_dcc) for d in obs_dts], dtype=np.float64)

        raw_events = merged_divs.get(ticker, [])
        n_div = len(raw_events)
        if n_div > 0:
            div_meta: list[tuple[datetime, datetime, float, float, float, float, bool]] = []
            fut_pay_taus: list[float] = []
            fut_ex_taus: list[float] = []
            for ex_dt, pay_dt, amt in raw_events:
                tau_ex_365 = _year_fraction(t0_dt, ex_dt, disc_dcc)
                tau_pay_365 = _year_fraction(t0_dt, pay_dt, disc_dcc)
                tau_ex_360 = _year_fraction(t0_dt, ex_dt, repo_dcc)
                is_fut = ex_dt > t0_dt
                div_meta.append((ex_dt, pay_dt, amt, tau_ex_365, tau_pay_365, tau_ex_360, is_fut))
                if is_fut:
                    fut_pay_taus.append(tau_pay_365)
                    fut_ex_taus.append(tau_ex_360)

            if q_batch_fn is not None:
                if fut_ex_taus:
                    q_all = q_batch_fn(
                        np.concatenate([tau_repo_obs, np.asarray(fut_ex_taus, dtype=np.float64)])
                    )
                    q_obs_arr = q_all[:m_rem]
                    q_fut_arr = q_all[m_rem:]
                else:
                    q_obs_arr = q_batch_fn(tau_repo_obs)
                    q_fut_arr = np.zeros(0, dtype=np.float64)
            else:
                q_obs_arr = np.array([q_repo_fn(d) for d in obs_dts], dtype=np.float64)
                q_fut_arr = np.array(
                    [q_repo_fn(m[0]) for m in div_meta if m[6]], dtype=np.float64
                )

            if fut_pay_taus:
                if df_batch_fn is not None:
                    df_fut_arr = df_batch_fn(np.asarray(fut_pay_taus, dtype=np.float64))
                else:
                    df_fut_arr = np.array(
                        [df_fin_fn(m[1]) for m in div_meta if m[6]], dtype=np.float64
                    )
            else:
                df_fut_arr = np.zeros(0, dtype=np.float64)
        else:
            div_meta = []
            if q_batch_fn is not None:
                q_obs_arr = q_batch_fn(tau_repo_obs)
            else:
                q_obs_arr = np.array([q_repo_fn(d) for d in obs_dts], dtype=np.float64)
            q_fut_arr = np.zeros(0, dtype=np.float64)
            df_fut_arr = np.zeros(0, dtype=np.float64)

        df_repo_np[i, :] = np.exp(-q_obs_arr * tau_repo_obs)
        if has_quanto_data and _infer_asset_currency(ticker, pay_ccy_upper) != pay_ccy_upper:
            quanto_arr = np.array(
                [
                    _resolve_quanto_drift_rate(
                        ticker=ticker,
                        spot_val=spot_i_val,
                        as_of_dt=t0_dt,
                        obs_dt=obs_dt,
                        payment_currency=payment_currency,
                        vol_surfaces=vol_surfaces,
                        correlations=correlations,
                    )
                    for obs_dt in obs_dts
                ],
                dtype=np.float64,
            )
            r_tot_np[i, :] = r_times_tau_obs - q_obs_arr * tau_repo_obs + quanto_arr * tau_365_obs
        else:
            r_tot_np[i, :] = r_times_tau_obs - q_obs_arr * tau_repo_obs

        repo_avg_np[i] = float(np.mean(q_obs_arr)) if m_rem > 0 else 0.0

        # Process discrete cash dividends for asset i
        parsed_events_i: list[ParsedDividendEvent] = []
        weights_i: list[float] = []
        amounts_i: list[float] = []
        incidence_rows_i: list[list[float]] = []

        fut_idx = 0
        for ex_dt, pay_dt, amt, tau_ex_365, tau_pay_365, tau_ex_360, is_fut in div_meta:
            if not is_fut:
                w_k = 0.0
                inc_k = [0.0] * m_rem
            else:
                df_fin_pay = float(df_fut_arr[fut_idx])
                q_ex = float(q_fut_arr[fut_idx])
                fut_idx += 1
                w_k = float(df_fin_pay * np.exp(q_ex * tau_ex_360))
                inc_k = [1.0 if (t0_dt < ex_dt <= obs_dt) else 0.0 for obs_dt in obs_dts]

            parsed_events_i.append(
                ParsedDividendEvent(
                    ex_date=ex_dt,
                    payment_date=pay_dt,
                    amount=amt,
                    tau_ex_act365=tau_ex_365,
                    tau_pay_act365=tau_pay_365,
                    tau_ex_act360=tau_ex_360,
                    pv_discount_weight=w_k,
                )
            )
            weights_i.append(w_k)
            amounts_i.append(amt)
            incidence_rows_i.append(inc_k)

        dividend_events_map[ticker] = parsed_events_i

        if len(parsed_events_i) > 0:
            if dividend_amount_overrides and ticker in dividend_amount_overrides:
                amt_t = dividend_amount_overrides[ticker].to(dtype=torch.float64, device=device)
            else:
                amt_t = torch.tensor(amounts_i, dtype=torch.float64, device=device)
            w_t = torch.tensor(weights_i, dtype=torch.float64, device=device)
            inc_t = torch.tensor(incidence_rows_i, dtype=torch.float64, device=device)  # (K_div, M_rem)

            div_amount_tensors[ticker] = amt_t
            div_pv_weights_map[ticker] = w_t
            div_incidence_map[ticker] = inc_t

            # D_{i,pv}(t_0, t_m) = sum_k d_{i,k} * w_{i,k} * 1_{t_0 < tau_{i,k}^{ex} <= t_m}
            pv_each_div = amt_t * w_t  # (K_div,)
            div_pv_row = torch.matmul(pv_each_div.unsqueeze(0), inc_t).squeeze(0)  # (M_rem,)
        else:
            empty_k = torch.zeros(0, dtype=torch.float64, device=device)
            div_amount_tensors[ticker] = empty_k
            div_pv_weights_map[ticker] = empty_k
            div_incidence_map[ticker] = torch.zeros((0, m_rem), dtype=torch.float64, device=device)
            div_pv_row = torch.zeros(m_rem, dtype=torch.float64, device=device)

        div_pv_rows.append(div_pv_row)

    div_pv_t = torch.stack(div_pv_rows, dim=0)  # (D, M_rem)
    df_fin_t = torch.as_tensor(df_fin_np, dtype=torch.float64, device=device)
    df_repo_t = torch.as_tensor(df_repo_np, dtype=torch.float64, device=device)
    r_tot_t = torch.as_tensor(r_tot_np, dtype=torch.float64, device=device)
    repo_avg_t = torch.as_tensor(repo_avg_np, dtype=torch.float64, device=device)

    spots_col = spots_t.unsqueeze(1)  # (D, 1)
    ref_col = ref_t.unsqueeze(1)      # (D, 1)

    # Escrowed spot floor: S_i^*(t_0, t_m) = max(S_i(t_0) - D_{i,pv}(t_0, t_m), 0.01 * S_i(t_0))
    floor_spot = escrow_floor_fraction * spots_col
    escrowed_spot = torch.maximum(spots_col - div_pv_t, floor_spot)

    growth_factor = torch.exp(r_tot_t)
    forward_price = escrowed_spot * growth_factor
    forward_rel = forward_price / spots_col
    forward_perf = forward_price / ref_col

    cash_div_leverage = spots_col / escrowed_spot
    escrowed_div_fraction = div_pv_t / spots_col

    return EscrowedDividendOutput(
        identifiers=id_list,
        as_of_date=t0_dt,
        observation_dates=obs_dts,
        div_pv=div_pv_t,
        escrowed_spot=escrowed_spot,
        forward_price=forward_price,
        forward_rel=forward_rel,
        forward_perf=forward_perf,
        cash_div_leverage=cash_div_leverage,
        escrowed_div_fraction=escrowed_div_fraction,
        df_fin_obs=df_fin_t,
        df_repo_obs=df_repo_t,
        r_tot_obs=r_tot_t,
        repo_spread_avg=repo_avg_t,
        dividend_events=dividend_events_map,
        dividend_amount_tensors=div_amount_tensors,
        dividend_pv_weights=div_pv_weights_map,
        dividend_obs_incidence=div_incidence_map,
    )


__all__ = [
    "ParsedDividendEvent",
    "EscrowedDividendOutput",
    "build_repo_rate_evaluator",
    "build_discount_rate_evaluator",
    "merge_dividend_schedules",
    "compute_escrowed_dividend_forwards",
]
