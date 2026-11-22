"""Buehler escrowed/affine cash dividend forward, quanto, and leverage engine in IEEE-754 float64."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import logging
import re
from typing import Any, Callable, Mapping, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
import torch
import torch.nn.functional as F

from deepvol.market.eq_curves import (
    ClampedYieldCurve,
    RepoCurve,
    dcc_denominator,
    year_fraction,
)
from deepvol.models.autocall.seasoned_collapser import _to_utc_datetime

_LOGGER = logging.getLogger(__name__)
_PERIOD_DAYS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([DWMYdwmy])\s*$")


def _parse_period_days(period_str: str | int | float) -> float:
    """Parses a Murex period string such as '31D', '1.5Y', or 'O/N' into calendar days."""
    if isinstance(period_str, (int, float)):
        return float(period_str)
    s = str(period_str).strip().upper()
    if s in ("O/N", "ON"):
        return 1.0
    if s in ("T/N", "TN"):
        return 2.0
    if s in ("S/N", "SN"):
        return 3.0
    m = _PERIOD_DAYS_RE.match(s)
    if not m:
        return float(s.rstrip("D"))
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


def _parse_tenor_years(tenor_str: str | int | float, dcc: str = "Act365") -> float:
    """Converts a Murex FX/equity tenor label directly to year-fraction units without 365/252 distortion."""
    if isinstance(tenor_str, (int, float)):
        return float(tenor_str) / 365.0
    s = str(tenor_str).strip().upper()
    denom = dcc_denominator(dcc)
    if s in ("O/N", "ON"):
        return 1.0 / denom
    if s in ("T/N", "TN"):
        return 2.0 / denom
    if s in ("S/N", "SN"):
        return 3.0 / denom
    m = _PERIOD_DAYS_RE.match(s)
    if not m:
        return float(s.rstrip("D")) / 365.0
    val = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "Y":
        return val
    if unit == "M":
        return val / 12.0
    if unit == "W":
        return val / 52.0
    if unit == "D":
        # Murex labels FX tenors as calendar days (e.g. '365D' for 1Y) even on Bus252 surfaces.
        return val / denom if (val <= 5.0 and "252" in dcc) else (val / 365.0)
    return val / 365.0


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

    div_pv: torch.Tensor
    escrowed_spot: torch.Tensor
    forward_price: torch.Tensor
    forward_rel: torch.Tensor
    forward_perf: torch.Tensor
    cash_div_leverage: torch.Tensor
    escrowed_div_fraction: torch.Tensor

    df_fin_obs: torch.Tensor
    df_repo_obs: torch.Tensor
    r_tot_obs: torch.Tensor

    repo_spread_avg: torch.Tensor

    dividend_events: dict[str, list[ParsedDividendEvent]] = field(default_factory=dict)
    dividend_amount_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    dividend_pv_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    dividend_obs_incidence: dict[str, torch.Tensor] = field(default_factory=dict)
    quanto_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    high_dividend_warning: bool = False
    r_native_obs: torch.Tensor | None = None


def build_repo_rate_evaluator(
    repo_spec: Any | None,
    as_of_date: datetime | date | str,
) -> Callable[[datetime | float], float]:
    """Builds a float64 repo rate evaluator q_repo(t) delegating directly to RepoCurve."""
    if repo_spec is None:
        def fn_zero(_t: datetime | float) -> float:
            return 0.0

        fn_zero._eval_tau_batch = lambda tau_arr: np.zeros(len(tau_arr), dtype=np.float64)  # type: ignore[attr-defined]
        return fn_zero

    if hasattr(repo_spec, "repo_rate") and callable(repo_spec.repo_rate):
        rc_existing = repo_spec

        def fn_rc(dt_or_tau: datetime | float) -> float:
            return float(np.asarray(rc_existing.repo_rate(dt_or_tau), dtype=np.float64).reshape(-1)[0])

        fn_rc._eval_tau_batch = lambda tau_arr: np.asarray(rc_existing.repo_rate(tau_arr), dtype=np.float64).reshape(-1)  # type: ignore[attr-defined]
        return fn_rc

    if hasattr(repo_spec, "zero_rate") and callable(repo_spec.zero_rate):
        def _eval_curve_obj(dt_or_tau: datetime | float) -> float:
            res = repo_spec.zero_rate([dt_or_tau] if isinstance(dt_or_tau, datetime) else np.asarray([float(dt_or_tau)], dtype=np.float64))
            return float(np.asarray(res, dtype=np.float64).reshape(-1)[0])

        _eval_curve_obj._eval_tau_batch = lambda tau_arr: np.asarray(repo_spec.zero_rate(tau_arr), dtype=np.float64).reshape(-1)  # type: ignore[attr-defined]
        return _eval_curve_obj

    t0_dt = _to_utc_datetime(as_of_date)
    rc = RepoCurve(repo_spec, as_of_date=t0_dt)
    if rc.is_flat_zero:
        def fn_empty(_t: datetime | float) -> float:
            return 0.0

        fn_empty._eval_tau_batch = lambda tau_arr: np.zeros(len(tau_arr), dtype=np.float64)  # type: ignore[attr-defined]
        return fn_empty

    def _eval_repo(dt_or_tau: datetime | float) -> float:
        return float(np.asarray(rc.repo_rate(dt_or_tau), dtype=np.float64).reshape(-1)[0])

    _eval_repo._eval_tau_batch = lambda tau_arr: np.asarray(rc.repo_rate(tau_arr), dtype=np.float64).reshape(-1)  # type: ignore[attr-defined]
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
    """Returns (r_zero_fn(dt), df_fn(dt)) in float64 delegating directly to ClampedYieldCurve."""
    t0_dt = _to_utc_datetime(as_of_date)
    if not (hasattr(discount_curve, "df") and callable(discount_curve.df)):
        if discount_curve is None:
            def r_z(_dt: datetime) -> float:
                return 0.0

            def df_z(_dt: datetime) -> float:
                return 1.0

            r_z._eval_tau_batch = lambda tau_arr: np.zeros(len(tau_arr), dtype=np.float64)  # type: ignore[attr-defined]
            df_z._eval_tau_batch = lambda tau_arr: np.ones(len(tau_arr), dtype=np.float64)  # type: ignore[attr-defined]
            return r_z, df_z
        discount_curve = ClampedYieldCurve(discount_curve, as_of_date=t0_dt)

    curve_dcc = _extract_curve_dcc(discount_curve, "Act365")

    def _df_from_obj(dt: datetime) -> float:
        dt_utc = _to_utc_datetime(dt)
        if dt_utc <= t0_dt:
            return 1.0
        val = discount_curve.df(dt_utc)
        return float(np.asarray(val, dtype=np.float64).reshape(-1)[0])

    def _r_from_obj(dt: datetime) -> float:
        dt_utc = _to_utc_datetime(dt)
        tau = year_fraction(t0_dt, dt_utc, curve_dcc)
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


def _extract_events_from_div_source(source_obj: Any) -> list[tuple[datetime, datetime, float]]:
    events: list[tuple[datetime, datetime, float]] = []
    if source_obj is None:
        return events
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
        elif "ExDate" in source_obj and isinstance(source_obj["ExDate"], Sequence) and not isinstance(source_obj["ExDate"], (str, bytes)):
            ex_dates = source_obj["ExDate"]
            pay_dates = source_obj.get("PaymentDate", ex_dates)
            amounts = source_obj.get("Amount", [])
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
                pay_dates = getattr(source_obj, "payment_dates", ex_dates)
            amounts = getattr(source_obj, "Amounts", None)
            if amounts is None:
                amounts = getattr(source_obj, "amounts", [])
            for ex_d, pay_d, amt in zip(ex_dates, pay_dates, amounts):
                events.append((_to_utc_datetime(ex_d), _to_utc_datetime(pay_d), float(amt)))
    return events


def merge_dividend_schedules(
    identifiers: Sequence[str],
    market_dividends: Mapping[str, Any] | None = None,
    predefined_dividends: Mapping[str, Any] | None = None,
) -> dict[str, list[tuple[datetime, datetime, float]]]:
    """Merges MarketDataSet.Dividends and Product.PredefinedDividends per asset.

    A non-empty entry in `PredefinedDividends` overrides `MarketDataSet.Dividends` for that asset.
    """
    merged: dict[str, list[tuple[datetime, datetime, float]]] = {}

    for ticker in identifiers:
        events: list[tuple[datetime, datetime, float]] = []
        if predefined_dividends and ticker in predefined_dividends and predefined_dividends[ticker] is not None:
            events = _extract_events_from_div_source(predefined_dividends[ticker])
        if not events and market_dividends and ticker in market_dividends and market_dividends[ticker] is not None:
            events = _extract_events_from_div_source(market_dividends[ticker])
        merged[str(ticker)] = events

    return merged


_KNOWN_ISO_CURRENCIES: frozenset[str] = frozenset(
    {
        "RUB",
        "USD",
        "CNH",
        "CNY",
        "EUR",
        "GBP",
        "HKD",
        "CHF",
        "JPY",
        "AED",
        "KZT",
        "TRY",
        "INR",
        "BRL",
        "ZAR",
        "BYN",
        "CAD",
        "AUD",
        "SGD",
    }
)
_ISO_CCY_CODES: frozenset[str] = _KNOWN_ISO_CURRENCIES

_NON_CCY_TICKER_SUFFIXES: frozenset[str] = frozenset(
    {
        "OLD",
        "NEW",
        "REG",
        "PREF",
        "P",
        "DR",
        "ADR",
        "GDR",
        "CLS",
        "LOCAL",
        "SHS",
        "ORD",
        "A",
        "B",
        "C",
        "1",
        "2",
        "RTS",
        "MOEX",
    }
)
_VERSION_TAG_SUFFIXES: frozenset[str] = _NON_CCY_TICKER_SUFFIXES

_COUNTRY_SUFFIX_TO_CCY: dict[str, str] = {
    "US": "USD",
    "HK": "HKD",
    "CN": "CNH",
    "EU": "EUR",
    "LN": "GBP",
    "GR": "EUR",
    "FP": "EUR",
    "GY": "EUR",
    "SW": "CHF",
    "JP": "JPY",
    "KZ": "KZT",
}

_OIS_CURVE_KEYWORDS: tuple[str, ...] = (
    "OIS",
    "SOFR",
    "RUONIA",
    "ESTR",
    "SONIA",
    "TONAR",
    "SARON",
    "SHIBOR",
)
_OIS_KEYWORDS: tuple[str, ...] = _OIS_CURVE_KEYWORDS

_FUNDING_CURVE_KEYWORDS: tuple[str, ...] = (
    "FUNDING",
    "FUND",
    "SECD",
)


def strip_version_tags(ticker: str) -> list[str]:
    """Splits a ticker on '_' and strips trailing share-class or version tags."""
    tokens = [tok.strip().upper() for tok in str(ticker).split("_") if tok.strip()]
    while len(tokens) > 1:
        tail = tokens[-1]
        if tail in _NON_CCY_TICKER_SUFFIXES:
            tokens.pop()
        elif tail == "RX" and len(tokens) >= 3 and len(tokens[-2]) == 3 and tokens[-2].isalpha():
            tokens.pop()
        else:
            break
    return tokens


def _extract_explicit_ccy(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        raw = obj.get("Currency", obj.get("currency"))
    else:
        raw = getattr(obj, "Currency", getattr(obj, "currency", None))
    if raw is None:
        return None
    ccy = str(getattr(raw, "value", raw)).strip().upper()
    if len(ccy) == 3 and ccy.isalpha() and ccy not in _NON_CCY_TICKER_SUFFIXES:
        return ccy
    return None


def _is_equity_vol_surface_obj(obj: Any) -> bool:
    if obj is None:
        return False
    t_name = obj.get("TypeName") if isinstance(obj, Mapping) else getattr(obj, "TypeName", None)
    if t_name == "FXVolatilitySmileCollection":
        return False
    if t_name == "EquityVolatilitySmileCollection":
        return True
    has_strikes = (
        ("Strikes" in obj or "strikes" in obj)
        if isinstance(obj, Mapping)
        else (hasattr(obj, "Strikes") or hasattr(obj, "strikes"))
    )
    has_deltas = (
        ("Deltas" in obj or "deltas" in obj)
        if isinstance(obj, Mapping)
        else (hasattr(obj, "Deltas") or hasattr(obj, "deltas"))
    )
    return bool(has_strikes and not has_deltas)


def _infer_asset_currency(
    ticker: str,
    default_ccy: str = "RUB",
    div_spec: Any = None,
    fx_smiles: Any = None,
    correlations: Any = None,
    rate_curve_ids: Sequence[str] = (),
    *,
    dividends_map: Mapping[str, Any] | None = None,
    predefined_dividends_map: Mapping[str, Any] | None = None,
    vol_surfaces: Mapping[str, Any] | None = None,
    rates_bundles: Mapping[str, Any] | Sequence[str] | None = None,
) -> str:
    """5-tier deterministic asset currency resolution from metadata, tokens, FX pairs, and curves."""
    pay_ccy = str(default_ccy).strip().upper()
    ticker_str = str(ticker).strip()
    ticker_u = ticker_str.upper()

    explicit_ccy = _extract_explicit_ccy(div_spec)
    if explicit_ccy is not None:
        return explicit_ccy
    for src_map in (predefined_dividends_map, dividends_map, vol_surfaces):
        if isinstance(src_map, Mapping) and ticker_str in src_map and src_map[ticker_str] is not None:
            map_ccy = _extract_explicit_ccy(src_map[ticker_str])
            if map_ccy is not None:
                return map_ccy

    known_ccys: set[str] = set(_KNOWN_ISO_CURRENCIES)
    if len(pay_ccy) == 3 and pay_ccy.isalpha() and pay_ccy not in _NON_CCY_TICKER_SUFFIXES:
        known_ccys.add(pay_ccy)

    active_rate_ccys: set[str] = set()
    all_curve_ids: list[str] = [str(cid) for cid in (rate_curve_ids or ())]
    if isinstance(rates_bundles, Mapping):
        for k_cid, c_obj in rates_bundles.items():
            all_curve_ids.append(str(k_cid))
            c_ccy = _extract_explicit_ccy(c_obj)
            if c_ccy is not None:
                known_ccys.add(c_ccy)
                active_rate_ccys.add(c_ccy)
            c_ident = (
                c_obj.get("Identifier", c_obj.get("identifier"))
                if isinstance(c_obj, Mapping)
                else getattr(c_obj, "Identifier", getattr(c_obj, "identifier", None))
            )
            if c_ident is not None:
                all_curve_ids.append(str(c_ident))
    elif isinstance(rates_bundles, Sequence) and not isinstance(rates_bundles, (str, bytes)):
        all_curve_ids.extend(str(cid) for cid in rates_bundles)

    for cid_raw in all_curve_ids:
        cid_u = cid_raw.strip().upper()
        if len(cid_u) >= 3 and cid_u[:3].isalpha():
            prefix3 = cid_u[:3]
            if prefix3 in _KNOWN_ISO_CURRENCIES:
                active_rate_ccys.add(prefix3)
                known_ccys.add(prefix3)
            elif (len(cid_u) == 3 or cid_u[3] in ("_", "-", "/")) and prefix3 not in _NON_CCY_TICKER_SUFFIXES:
                active_rate_ccys.add(prefix3)
                known_ccys.add(prefix3)

    active_fx_ccys: set[str] = set()
    fx_pair_entries: list[tuple[str, str]] = []
    for surf_src, is_dedicated_fx in ((fx_smiles, True), (vol_surfaces, False)):
        if surf_src is None:
            continue
        if isinstance(surf_src, Mapping):
            items_iter = list(surf_src.items())
        elif isinstance(surf_src, Sequence) and not isinstance(surf_src, (str, bytes)):
            items_iter = [(str(k), None) for k in surf_src]
        else:
            continue
        for sid, s_obj in items_iter:
            if _is_equity_vol_surface_obj(s_obj):
                continue
            t_name = (
                s_obj.get("TypeName")
                if isinstance(s_obj, Mapping)
                else getattr(s_obj, "TypeName", None)
            )
            sid_clean = str(sid).strip().upper().replace("_", "").replace("/", "").replace("-", "")
            if len(sid_clean) == 6 and sid_clean.isalpha():
                c1, c2 = sid_clean[:3], sid_clean[3:]
                is_fx_pair = (
                    t_name == "FXVolatilitySmileCollection"
                    or (c1 in known_ccys and c2 in known_ccys)
                    or (is_dedicated_fx and s_obj is None)
                )
                if is_fx_pair and c1 not in _NON_CCY_TICKER_SUFFIXES and c2 not in _NON_CCY_TICKER_SUFFIXES:
                    known_ccys.add(c1)
                    known_ccys.add(c2)
                    active_fx_ccys.add(c1)
                    active_fx_ccys.add(c2)
                    fx_pair_entries.append((str(sid), sid_clean))

    tokens = strip_version_tags(ticker_u)
    for tok in reversed(tokens[1:]):
        if len(tok) == 3 and tok.isalpha() and tok in known_ccys:
            return tok

    if isinstance(correlations, Mapping) and len(correlations) > 0:
        for raw_fx_id, fx_clean in fx_pair_entries:
            if f"{ticker_str}&{raw_fx_id}" in correlations or f"{raw_fx_id}&{ticker_str}" in correlations:
                c1, c2 = fx_clean[:3], fx_clean[3:]
                if c1 == pay_ccy and c2 != pay_ccy:
                    return c2
                if c2 == pay_ccy and c1 != pay_ccy:
                    return c1
        for corr_key in correlations:
            key_s = str(corr_key)
            if "&" not in key_s:
                continue
            left_id, right_id = key_s.split("&", 1)
            if left_id.strip().upper() == ticker_u:
                other_id = right_id.strip()
            elif right_id.strip().upper() == ticker_u:
                other_id = left_id.strip()
            else:
                continue
            p_clean = other_id.upper().replace("/", "").replace("_", "")
            if len(p_clean) == 6 and p_clean.isalpha():
                c1, c2 = p_clean[:3], p_clean[3:]
                if c1 in known_ccys and c2 in known_ccys:
                    if c1 == pay_ccy and c2 != pay_ccy:
                        return c2
                    if c2 == pay_ccy and c1 != pay_ccy:
                        return c1

    base_tok = tokens[0] if tokens else ticker_u
    if len(base_tok) > 4:
        mapped_ccy = _COUNTRY_SUFFIX_TO_CCY.get(base_tok[-2:])
        if mapped_ccy is not None and (mapped_ccy in active_rate_ccys or mapped_ccy in active_fx_ccys):
            return mapped_ccy

    return pay_ccy


def infer_asset_currency_from_context(
    ticker: str,
    payment_currency: str = "RUB",
    *,
    div_spec: Any = None,
    fx_smiles: Any = None,
    dividends_map: Mapping[str, Any] | None = None,
    predefined_dividends_map: Mapping[str, Any] | None = None,
    vol_surfaces: Mapping[str, Any] | None = None,
    correlations: Mapping[str, float] | None = None,
    rates_bundles: Mapping[str, Any] | Sequence[str] | None = None,
    rate_curve_ids: Sequence[str] = (),
) -> str:
    """Context-aware 5-tier deterministic asset currency resolver."""
    return _infer_asset_currency(
        ticker,
        payment_currency,
        div_spec=div_spec,
        fx_smiles=fx_smiles,
        correlations=correlations,
        rate_curve_ids=rate_curve_ids,
        dividends_map=dividends_map,
        predefined_dividends_map=predefined_dividends_map,
        vol_surfaces=vol_surfaces,
        rates_bundles=rates_bundles,
    )


_infer_asset_currency_from_context = infer_asset_currency_from_context


def equity_forward_curve_priority(
    curve_id: str,
    financing_curve_ids: Sequence[str] = (),
) -> tuple[int, str]:
    """Ranks candidate same-currency curves for equity forward growth (0=OIS, 1=Std, 2=XCCY, 3=Funding)."""
    cid_u = str(curve_id).strip().upper()
    fin_set = {str(fid).strip().upper() for fid in (financing_curve_ids or ())}
    has_xccy = "XCCY" in cid_u
    has_funding_tag = any(k in cid_u for k in _FUNDING_CURVE_KEYWORDS)

    if not has_xccy and not has_funding_tag and any(k in cid_u for k in _OIS_CURVE_KEYWORDS):
        return (0, cid_u)
    if not has_xccy and not has_funding_tag and cid_u not in fin_set:
        return (1, cid_u)
    if has_xccy and not has_funding_tag:
        return (2, cid_u)
    return (3, cid_u)


curve_ois_priority_rank = equity_forward_curve_priority


def _eval_fx_atm_vol_batch(fx_spec: Any, tau_eval_arr: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], str]:
    if isinstance(fx_spec, Mapping):
        tenors_raw = fx_spec.get("Tenors", fx_spec.get("tenors", []))
        deltas_raw = fx_spec.get("Deltas", fx_spec.get("deltas", ["ATM"]))
        quotes_raw = fx_spec.get(
            "VolatilityQuotes", fx_spec.get("volatility_quotes", fx_spec.get("Quotes", fx_spec.get("quotes", [])))
        )
        raw_dcc = fx_spec.get("DayCountConvention", fx_spec.get("day_count_convention", "Act365"))
        raw_interp = fx_spec.get("SmileInterpolationMethod", fx_spec.get("smile_interpolation_method", "Linear"))
        raw_extrap = fx_spec.get("SmileExtrapolationMethod", fx_spec.get("smile_extrapolation_method", "Flat"))
    else:
        tenors_raw = getattr(fx_spec, "Tenors", getattr(fx_spec, "tenors", []))
        deltas_raw = getattr(fx_spec, "Deltas", getattr(fx_spec, "deltas", ["ATM"]))
        quotes_raw = getattr(
            fx_spec,
            "VolatilityQuotes",
            getattr(fx_spec, "volatility_quotes", getattr(fx_spec, "Quotes", getattr(fx_spec, "quotes", []))),
        )
        raw_dcc = getattr(fx_spec, "DayCountConvention", getattr(fx_spec, "day_count_convention", "Act365"))
        raw_interp = getattr(fx_spec, "SmileInterpolationMethod", getattr(fx_spec, "smile_interpolation_method", "Linear"))
        raw_extrap = getattr(fx_spec, "SmileExtrapolationMethod", getattr(fx_spec, "smile_extrapolation_method", "Flat"))

    dcc = str(getattr(raw_dcc, "value", raw_dcc))
    interp_method = str(getattr(raw_interp, "value", raw_interp))
    extrap_method = str(getattr(raw_extrap, "value", raw_extrap))
    if not tenors_raw or not quotes_raw:
        return np.zeros(len(tau_eval_arr), dtype=np.float64), dcc

    deltas_list = [str(d).upper() for d in deltas_raw]
    atm_idx = deltas_list.index("ATM") if "ATM" in deltas_list else len(deltas_list) // 2
    t_arr = np.asarray([_parse_tenor_years(t, dcc) for t in tenors_raw], dtype=np.float64)
    v_arr = np.asarray(
        [
            float(row[atm_idx]) if isinstance(row, Sequence) and not isinstance(row, (str, bytes)) else float(row)
            for row in quotes_raw
        ],
        dtype=np.float64,
    )

    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    v_arr = v_arr[order]

    if len(t_arr) == 1:
        return np.full(len(tau_eval_arr), float(v_arr[0]), dtype=np.float64), dcc

    t_clip = np.clip(tau_eval_arr, t_arr[0], t_arr[-1])
    if interp_method in ("CubicSpline", "Smooth") and len(t_arr) >= 3:
        cs = CubicSpline(t_arr, v_arr, bc_type="natural")
        out = np.asarray(cs(t_clip), dtype=np.float64)
    else:
        out = np.interp(t_clip, t_arr, v_arr)

    if extrap_method == "Linear":
        mask_l = tau_eval_arr < t_arr[0]
        if np.any(mask_l):
            sl = (v_arr[1] - v_arr[0]) / max(t_arr[1] - t_arr[0], 1e-12)
            out[mask_l] = v_arr[0] + sl * (tau_eval_arr[mask_l] - t_arr[0])
        mask_r = tau_eval_arr > t_arr[-1]
        if np.any(mask_r):
            sr = (v_arr[-1] - v_arr[-2]) / max(t_arr[-1] - t_arr[-2], 1e-12)
            out[mask_r] = v_arr[-1] + sr * (tau_eval_arr[mask_r] - t_arr[-1])

    return np.maximum(out, 1e-6), dcc


def _eval_fx_atm_vol(fx_spec: Any, tau_eval: float) -> float:
    vals, _ = _eval_fx_atm_vol_batch(fx_spec, np.array([float(tau_eval)], dtype=np.float64))
    return float(vals[0])


def _eval_eq_atm_vol(eq_spec: Any, as_of_dt: datetime, obs_dt: datetime, spot_val: float) -> float:
    if hasattr(eq_spec, "iv") and callable(eq_spec.iv):
        surf_dcc = str(getattr(eq_spec, "dcc", "Act365"))
        tau = year_fraction(as_of_dt, obs_dt, surf_dcc)
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
    t_eval = year_fraction(as_of_dt, _to_utc_datetime(obs_dt), dcc)
    tenors = np.asarray([year_fraction(as_of_dt, _to_utc_datetime(d), dcc) for d in exp_dates], dtype=np.float64)
    strikes = np.asarray(strikes_raw, dtype=np.float64)
    quotes = np.asarray(quotes_raw, dtype=np.float64)
    slice_vols = np.array([float(np.interp(spot_val, strikes, row)) for row in quotes], dtype=np.float64)
    if len(tenors) == 1:
        return float(slice_vols[0])
    order = np.argsort(tenors)
    tenors = tenors[order]
    slice_vols = slice_vols[order]
    if t_eval <= float(tenors[0]):
        return float(slice_vols[0])
    if t_eval >= float(tenors[-1]):
        return float(slice_vols[-1])
    # Interpolate linearly in total variance (sigma^2 * T) to match RegularizedVolSurface.iv
    slice_vars = (slice_vols * slice_vols) * np.maximum(tenors, 0.0)
    w_eval = float(np.interp(t_eval, tenors, slice_vars))
    return float(np.sqrt(max(w_eval / max(t_eval, 1e-12), 0.0)))


class QuantoFXSpec(tuple[str, Any, float]):
    """3-tuple (fx_id, fx_spec, rho_s_fx) exposing .fx_id, .fx_spec, .rho_s_fx, .fx_sign, and .rho_quoted."""

    fx_id: str
    fx_spec: Any
    rho_s_fx: float
    fx_sign: float
    rho_quoted: float

    def __new__(
        cls,
        fx_id: str,
        fx_spec: Any,
        rho_s_fx: float,
        fx_sign: float = 1.0,
        rho_quoted: float = 0.0,
    ) -> QuantoFXSpec:
        obj = super().__new__(cls, (str(fx_id), fx_spec, float(rho_s_fx)))
        obj.fx_id = str(fx_id)
        obj.fx_spec = fx_spec
        obj.rho_s_fx = float(rho_s_fx)
        obj.fx_sign = float(fx_sign)
        obj.rho_quoted = float(rho_quoted)
        return obj


def _lookup_quanto_fx_spec(
    *,
    ticker: str,
    payment_currency: str,
    vol_surfaces: Mapping[str, Any] | None,
    correlations: Mapping[str, float] | None,
    asset_ccy: str | None = None,
    div_spec: Any = None,
    rate_curve_ids: Sequence[str] = (),
) -> QuantoFXSpec | None:
    """Resolves (fx_id, fx_spec, signed_rho_s_fx) with .fx_sign and .fx_id for a foreign-currency equity."""
    if not vol_surfaces or not correlations:
        return None
    pay_ccy = payment_currency.strip().upper()
    resolved_asset_ccy = (
        asset_ccy.strip().upper()
        if (isinstance(asset_ccy, str) and len(asset_ccy.strip()) == 3)
        else _infer_asset_currency(
            ticker,
            pay_ccy,
            div_spec=div_spec,
            fx_smiles=vol_surfaces,
            correlations=correlations,
            rate_curve_ids=rate_curve_ids,
        )
    )
    if resolved_asset_ccy == pay_ccy:
        return None

    fx_candidates = (
        f"{resolved_asset_ccy}_{pay_ccy}",
        f"{resolved_asset_ccy}{pay_ccy}",
        f"{resolved_asset_ccy}/{pay_ccy}",
        f"{pay_ccy}_{resolved_asset_ccy}",
        f"{pay_ccy}{resolved_asset_ccy}",
        f"{pay_ccy}/{resolved_asset_ccy}",
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
            if (
                t_name == "FXVolatilitySmileCollection"
                and resolved_asset_ccy in k_id.upper()
                and pay_ccy in k_id.upper()
            ):
                fx_id = k_id
                fx_spec = v_surf
                if k_id.upper().index(pay_ccy) < k_id.upper().index(resolved_asset_ccy):
                    fx_sign = -1.0
                break
    if fx_spec is None or fx_id is None:
        return None

    k1 = f"{ticker}&{fx_id}"
    k2 = f"{fx_id}&{ticker}"
    if k1 in correlations:
        rho_quoted = float(correlations[k1])
    elif k2 in correlations:
        rho_quoted = float(correlations[k2])
    else:
        return None
    rho_s_fx = rho_quoted * fx_sign
    return QuantoFXSpec(
        str(fx_id),
        fx_spec,
        rho_s_fx,
        fx_sign=fx_sign,
        rho_quoted=rho_quoted,
    )


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
    """Computes instantaneous quanto drift rate -rho_{S,FX} * sigma_{S,ATM} * sigma_{FX,ATM}."""
    resolved = _lookup_quanto_fx_spec(
        ticker=ticker,
        payment_currency=payment_currency,
        vol_surfaces=vol_surfaces,
        correlations=correlations,
    )
    if resolved is None:
        return 0.0
    _, fx_spec, rho_s_fx = resolved
    tau_365 = year_fraction(as_of_dt, obs_dt, "Act365")
    sig_fx = _eval_fx_atm_vol(fx_spec, tau_365)
    assert vol_surfaces is not None
    eq_spec = vol_surfaces.get(ticker)
    sig_eq = _eval_eq_atm_vol(eq_spec, as_of_dt, obs_dt, spot_val) if eq_spec is not None else 0.0
    return -rho_s_fx * sig_eq * sig_fx


def compute_escrowed_dividend_forwards(
    *,
    identifiers: Sequence[str],
    as_of_date: datetime | date | str,
    observation_dates: Sequence[datetime | date | str],
    spots_t0: torch.Tensor | Sequence[float] | Mapping[str, float],
    ref_spots: torch.Tensor | Sequence[float],
    discount_curve: Any,
    asset_financing_curves: Mapping[str, Any] | None = None,
    repo_curves: Mapping[str, Any] | None = None,
    market_dividends: Mapping[str, Any] | None = None,
    predefined_dividends: Mapping[str, Any] | None = None,
    dividend_amount_overrides: Mapping[str, torch.Tensor] | None = None,
    escrow_floor_fraction: float = 0.01,
    payment_currency: str = "RUB",
    vol_surfaces: Mapping[str, Any] | None = None,
    correlations: Mapping[str, float] | None = None,
) -> EscrowedDividendOutput:
    """Computes Buehler escrowed dividend forwards, multi-curve quanto drifts, and leverage in float64."""
    t0_dt = _to_utc_datetime(as_of_date)
    obs_dts = [_to_utc_datetime(d) for d in observation_dates]
    id_list = [str(x) for x in identifiers]
    d_assets = len(id_list)
    m_rem = len(obs_dts)

    if isinstance(spots_t0, torch.Tensor):
        spots_t = spots_t0.to(dtype=torch.float64)
    elif isinstance(spots_t0, Mapping):
        spots_t = torch.tensor([float(spots_t0[t]) for t in id_list], dtype=torch.float64)
    else:
        spots_t = torch.as_tensor(spots_t0, dtype=torch.float64)
    device = spots_t.device

    if not bool(torch.all(torch.isfinite(spots_t) & (spots_t > 0.0)).item()):
        raise ValueError("Spots must be strictly positive and finite.")

    if isinstance(ref_spots, torch.Tensor):
        ref_t = ref_spots.to(dtype=torch.float64, device=device)
    else:
        ref_t = torch.as_tensor(ref_spots, dtype=torch.float64, device=device)

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
            quanto_meta={},
            r_native_obs=empty_dm,
        )

    df_fin_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    df_repo_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    r_tot_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    r_native_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    repo_avg_np = np.zeros(d_assets, dtype=np.float64)

    dividend_events_map: dict[str, list[ParsedDividendEvent]] = {}
    div_amount_tensors: dict[str, torch.Tensor] = {}
    div_pv_weights_map: dict[str, torch.Tensor] = {}
    div_incidence_map: dict[str, torch.Tensor] = {}
    quanto_meta_map: dict[str, dict[str, Any]] = {}
    div_pv_rows: list[torch.Tensor] = []

    spots_cpu = spots_t.detach().cpu().numpy()

    tau_365_obs = np.array([year_fraction(t0_dt, d, "Act365") for d in obs_dts], dtype=np.float64)
    tau_360_obs = np.array([year_fraction(t0_dt, d, "Act360") for d in obs_dts], dtype=np.float64)

    pay_ccy_upper = payment_currency.strip().upper()
    has_quanto_data = bool(vol_surfaces and correlations)

    rate_curve_id_list: list[str] = []
    if asset_financing_curves is not None:
        for k_cid, v_cobj in asset_financing_curves.items():
            rate_curve_id_list.append(str(k_cid))
            for attr_name in ("identifier", "Identifier", "currency", "Currency"):
                attr_val = (
                    v_cobj.get(attr_name)
                    if isinstance(v_cobj, Mapping)
                    else getattr(v_cobj, attr_name, None)
                )
                if attr_val is not None:
                    rate_curve_id_list.append(str(attr_val))
    if discount_curve is not None:
        for attr_name in ("identifier", "Identifier", "currency", "Currency"):
            attr_val = (
                discount_curve.get(attr_name)
                if isinstance(discount_curve, Mapping)
                else getattr(discount_curve, attr_name, None)
            )
            if attr_val is not None:
                rate_curve_id_list.append(str(attr_val))
    rate_curve_ids_tuple = tuple(rate_curve_id_list)

    # Cache per-curve evaluators inside this call so domestic assets share one evaluation
    fin_eval_cache: dict[
        int,
        Tuple[
            str,
            Callable[[datetime], float],
            Callable[[datetime], float],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
    ] = {}

    for i, ticker in enumerate(id_list):
        fin_curve_i = (
            asset_financing_curves.get(ticker, discount_curve)
            if asset_financing_curves is not None
            else discount_curve
        )
        curve_key = id(fin_curve_i)
        if curve_key in fin_eval_cache:
            fin_dcc_i, r_fin_i_fn, df_fin_i_fn, tau_fin_obs, r_times_tau_fin_obs = fin_eval_cache[curve_key]
        else:
            fin_dcc_i = _extract_curve_dcc(fin_curve_i, "Act365")
            r_fin_i_fn, df_fin_i_fn = build_discount_rate_evaluator(fin_curve_i, t0_dt)
            if fin_dcc_i == "Act365":
                tau_fin_obs = tau_365_obs
            elif fin_dcc_i == "Act360":
                tau_fin_obs = tau_360_obs
            else:
                tau_fin_obs = np.array([year_fraction(t0_dt, d, fin_dcc_i) for d in obs_dts], dtype=np.float64)
            r_batch_fn = getattr(r_fin_i_fn, "_eval_tau_batch", None)
            if r_batch_fn is not None:
                r_obs_arr = r_batch_fn(tau_fin_obs)
            else:
                r_obs_arr = np.array([r_fin_i_fn(d) for d in obs_dts], dtype=np.float64)
            r_times_tau_fin_obs = r_obs_arr * tau_fin_obs
            fin_eval_cache[curve_key] = (
                fin_dcc_i,
                r_fin_i_fn,
                df_fin_i_fn,
                tau_fin_obs,
                r_times_tau_fin_obs,
            )

        df_fin_np[i, :] = np.exp(-r_times_tau_fin_obs)
        df_batch_fn = getattr(df_fin_i_fn, "_eval_tau_batch", None)

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
            tau_repo_obs = np.array([year_fraction(t0_dt, d, repo_dcc) for d in obs_dts], dtype=np.float64)

        raw_events = merged_divs.get(ticker, [])
        n_div = len(raw_events)
        div_meta: list[tuple[datetime, datetime, float, float, float, float, bool]] = []
        fut_ex_dates: list[datetime] = []
        fut_pay_taus: list[float] = []
        fut_ex_taus: list[float] = []
        fut_ex_taus_365: list[float] = []

        if n_div > 0:
            for ex_dt, pay_dt, amt in raw_events:
                tau_ex_365 = year_fraction(t0_dt, ex_dt, "Act365")
                tau_pay_fin = year_fraction(t0_dt, pay_dt, fin_dcc_i)
                tau_ex_repo = year_fraction(t0_dt, ex_dt, repo_dcc)
                is_fut = ex_dt > t0_dt
                div_meta.append((ex_dt, pay_dt, amt, tau_ex_365, tau_pay_fin, tau_ex_repo, is_fut))
                if is_fut:
                    fut_ex_dates.append(ex_dt)
                    fut_pay_taus.append(tau_pay_fin)
                    fut_ex_taus.append(tau_ex_repo)
                    fut_ex_taus_365.append(tau_ex_365)

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
                        [df_fin_i_fn(m[1]) for m in div_meta if m[6]], dtype=np.float64
                    )
            else:
                df_fut_arr = np.zeros(0, dtype=np.float64)
        else:
            if q_batch_fn is not None:
                q_obs_arr = q_batch_fn(tau_repo_obs)
            else:
                q_obs_arr = np.array([q_repo_fn(d) for d in obs_dts], dtype=np.float64)
            q_fut_arr = np.zeros(0, dtype=np.float64)
            df_fut_arr = np.zeros(0, dtype=np.float64)

        df_repo_np[i, :] = np.exp(-q_obs_arr * tau_repo_obs)
        r_native_i = r_times_tau_fin_obs - q_obs_arr * tau_repo_obs
        r_native_np[i, :] = r_native_i

        div_spec_i = (
            predefined_dividends.get(ticker)
            if (predefined_dividends and ticker in predefined_dividends and predefined_dividends[ticker] is not None)
            else (market_dividends.get(ticker) if market_dividends else None)
        )
        asset_ccy_i = _infer_asset_currency(
            ticker,
            pay_ccy_upper,
            div_spec=div_spec_i,
            fx_smiles=vol_surfaces,
            correlations=correlations,
            rate_curve_ids=rate_curve_ids_tuple,
        )

        quanto_resolved = (
            _lookup_quanto_fx_spec(
                ticker=ticker,
                payment_currency=payment_currency,
                vol_surfaces=vol_surfaces,
                correlations=correlations,
                asset_ccy=asset_ccy_i,
                div_spec=div_spec_i,
                rate_curve_ids=rate_curve_ids_tuple,
            )
            if has_quanto_data and asset_ccy_i != pay_ccy_upper
            else None
        )
        if quanto_resolved is not None:
            fx_id, fx_spec, rho_s_fx = quanto_resolved
            sig_fx_obs, fx_dcc = _eval_fx_atm_vol_batch(fx_spec, tau_365_obs)
            assert vol_surfaces is not None
            eq_spec = vol_surfaces.get(ticker)
            sig_eq_obs = np.array(
                [
                    _eval_eq_atm_vol(eq_spec, t0_dt, obs_dt, spot_i_val) if eq_spec is not None else 0.0
                    for obs_dt in obs_dts
                ],
                dtype=np.float64,
            )
            quanto_obs_arr = -rho_s_fx * sig_eq_obs * sig_fx_obs
            r_tot_np[i, :] = r_native_i + quanto_obs_arr * tau_365_obs
            quanto_meta_map[ticker] = {
                "fx_id": str(quanto_resolved.fx_id),
                "fx_sign": float(quanto_resolved.fx_sign),
                "rho_quoted": float(quanto_resolved.rho_quoted),
                "rho_s_fx": float(rho_s_fx),
                "sig_fx_obs": sig_fx_obs.tolist(),
                "fx_dcc": fx_dcc,
                "asset_ccy": asset_ccy_i,
            }
            if fut_ex_dates:
                sig_fx_ex, _ = _eval_fx_atm_vol_batch(fx_spec, np.asarray(fut_ex_taus_365, dtype=np.float64))
                sig_eq_ex = np.array(
                    [
                        _eval_eq_atm_vol(eq_spec, t0_dt, ex_dt, spot_i_val) if eq_spec is not None else 0.0
                        for ex_dt in fut_ex_dates
                    ],
                    dtype=np.float64,
                )
                quanto_fut_arr = -rho_s_fx * sig_eq_ex * sig_fx_ex
            else:
                quanto_fut_arr = np.zeros(0, dtype=np.float64)
        else:
            r_tot_np[i, :] = r_native_i
            quanto_fut_arr = np.zeros(0, dtype=np.float64)

        repo_avg_np[i] = float(np.mean(q_obs_arr)) if m_rem > 0 else 0.0

        # Process discrete cash dividends for asset i using native curve DF_f and quanto adjustment
        parsed_events_i: list[ParsedDividendEvent] = []
        weights_i: list[float] = []
        amounts_i: list[float] = []
        incidence_rows_i: list[list[float]] = []

        fut_idx = 0
        for ex_dt, pay_dt, amt, tau_ex_365, tau_pay_fin, tau_ex_repo, is_fut in div_meta:
            if not is_fut:
                w_k = 0.0
                inc_k = [0.0] * m_rem
            else:
                df_fin_pay = float(df_fut_arr[fut_idx])
                q_ex = float(q_fut_arr[fut_idx])
                q_quanto_ex = float(quanto_fut_arr[fut_idx]) if fut_idx < len(quanto_fut_arr) else 0.0
                fut_idx += 1
                w_k = float(df_fin_pay * np.exp(q_ex * tau_ex_repo - q_quanto_ex * tau_ex_365))
                inc_k = [1.0 if (t0_dt < ex_dt <= obs_dt) else 0.0 for obs_dt in obs_dts]

            parsed_events_i.append(
                ParsedDividendEvent(
                    ex_date=ex_dt,
                    payment_date=pay_dt,
                    amount=amt,
                    tau_ex_act365=tau_ex_365,
                    tau_pay_act365=tau_pay_fin,
                    tau_ex_act360=tau_ex_repo,
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
            inc_t = torch.tensor(incidence_rows_i, dtype=torch.float64, device=device)

            div_amount_tensors[ticker] = amt_t
            div_pv_weights_map[ticker] = w_t
            div_incidence_map[ticker] = inc_t

            pv_each_div = amt_t * w_t
            div_pv_row = torch.matmul(pv_each_div.unsqueeze(0), inc_t).squeeze(0)
        else:
            empty_k = torch.zeros(0, dtype=torch.float64, device=device)
            div_amount_tensors[ticker] = empty_k
            div_pv_weights_map[ticker] = empty_k
            div_incidence_map[ticker] = torch.zeros((0, m_rem), dtype=torch.float64, device=device)
            div_pv_row = torch.zeros(m_rem, dtype=torch.float64, device=device)

        div_pv_rows.append(div_pv_row)

    div_pv_t = torch.stack(div_pv_rows, dim=0)
    df_fin_t = torch.as_tensor(df_fin_np, dtype=torch.float64, device=device)
    df_repo_t = torch.as_tensor(df_repo_np, dtype=torch.float64, device=device)
    r_tot_t = torch.as_tensor(r_tot_np, dtype=torch.float64, device=device)
    r_native_t = torch.as_tensor(r_native_np, dtype=torch.float64, device=device)
    repo_avg_t = torch.as_tensor(repo_avg_np, dtype=torch.float64, device=device)

    spots_col = spots_t.unsqueeze(1)
    ref_col = ref_t.unsqueeze(1)

    high_div_warn = bool(torch.any(div_pv_t >= 0.80 * spots_col).item())
    if high_div_warn:
        _LOGGER.warning("Escrowed dividend PV >= 80%% of spot; using smooth softplus floor.")

    # Smooth softplus floor transition preserves 0-ULP exactness below 70% div yield while eliminating kinks near 99%
    floor_spot = escrow_floor_fraction * spots_col
    scale_sp = (0.01 * spots_col).clamp(min=1e-12)
    escrowed_spot = floor_spot + scale_sp * F.softplus(
        (spots_col - div_pv_t - floor_spot) / scale_sp, beta=1.0, threshold=40.0
    )

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
        quanto_meta=quanto_meta_map,
        high_dividend_warning=high_div_warn,
        r_native_obs=r_native_t,
    )


__all__ = [
    "ParsedDividendEvent",
    "EscrowedDividendOutput",
    "QuantoFXSpec",
    "strip_version_tags",
    "_infer_asset_currency",
    "infer_asset_currency_from_context",
    "_infer_asset_currency_from_context",
    "equity_forward_curve_priority",
    "curve_ois_priority_rank",
    "build_repo_rate_evaluator",
    "build_discount_rate_evaluator",
    "merge_dividend_schedules",
    "compute_escrowed_dividend_forwards",
]
