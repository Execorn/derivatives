"""Seasoned schedule partitioning, historical fixing collapse, and Markovian state construction."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch

from deepvol.market.eq_curves import year_fraction as _eq_year_fraction
from deepvol.models.autocall.feature_layout import PAIR_ORDER_DEPENDENT_INDICES


def _to_utc_datetime(dt_or_str: datetime | date | str) -> datetime:
    """Normalizes a date, datetime, or ISO string to a timezone-aware UTC datetime."""
    if type(dt_or_str) is datetime:
        if dt_or_str.tzinfo is timezone.utc:
            return dt_or_str
        if dt_or_str.tzinfo is None:
            return dt_or_str.replace(tzinfo=timezone.utc)
        return dt_or_str.astimezone(timezone.utc)
    if isinstance(dt_or_str, str):
        s = dt_or_str.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        parsed = datetime.datetime.fromisoformat(s) if hasattr(datetime, "datetime") else datetime.fromisoformat(s)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    if isinstance(dt_or_str, datetime):
        if dt_or_str.tzinfo is None:
            return dt_or_str.replace(tzinfo=timezone.utc)
        return dt_or_str.astimezone(timezone.utc)
    if isinstance(dt_or_str, date):
        return datetime(dt_or_str.year, dt_or_str.month, dt_or_str.day, tzinfo=timezone.utc)
    raise TypeError(f"Unsupported date/datetime type: {type(dt_or_str)!r}")


def _year_fraction(
    d1: datetime | date | str,
    d2: datetime | date | str,
    dcc: str = "Act365",
) -> float:
    """Computes day-count fraction between d1 and d2 in float64 via canonical eq_curves."""
    return float(_eq_year_fraction(_to_utc_datetime(d1), _to_utc_datetime(d2), dcc))


@dataclass(frozen=True)
class CashFlowItem:
    """Canonical cash-flow row matching Murex Pricing Service CashFlows schema."""

    Description: Literal[
        "FundingLeg",
        "Coupon",
        "Notional",
        "Option",
        "OneStarCoupon",
        "IsLongBasket",
    ]
    Date: datetime
    Amount: float
    StandardError: float = 0.0
    Currency: str = "RUB"
    Leg: int = 0
    StatusText: Literal["Settled", "Estimate"] = "Estimate"

    @property
    def PaymentValue(self) -> float:
        """Alias for Amount used in audit gates."""
        return float(self.Amount)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the cash-flow row to a plain Murex-compatible dictionary."""
        return {
            "Description": str(self.Description),
            "Date": self.Date.isoformat() if isinstance(self.Date, (datetime, date)) else str(self.Date),
            "Amount": float(self.Amount),
            "StandardError": float(self.StandardError),
            "Currency": str(self.Currency),
            "Leg": int(self.Leg),
            "StatusText": str(self.StatusText),
        }


@dataclass(frozen=True)
class SentinelBarrierResult:
    """Normalized barrier schedule features after sentinel lock-out and guaranteed coupon handling."""

    ko_barriers_raw: npt.NDArray[np.float64]
    cpn_barriers_raw: npt.NDArray[np.float64]
    is_callable: npt.NDArray[np.float64]
    log_ko_barrier_feat: npt.NDArray[np.float64]
    is_guaranteed_cpn: npt.NDArray[np.float64]
    log_cpn_barrier_feat: npt.NDArray[np.float64]


@dataclass(frozen=True)
class HistoricalPeriodRecord:
    """Audit record for a settled or in-transit observation period (t_m <= t_0)."""

    period_index: int
    observation_date: datetime
    payment_date: datetime
    status: Literal["Settled", "In-Transit"]
    asset_performances: npt.NDArray[np.float64]
    call_index_value: float
    coupon_index_value: float
    ko_barrier: float
    cpn_barrier: float
    coupon_rate: float
    coupon_triggered: bool
    autocall_triggered: bool
    coupon_paid: float
    notional_paid: float
    unpaid_memory_after: float
    discount_factor: float


@dataclass
class SeasonedScheduleState:
    """Markovian state and partitioned schedule at AsOfDate t_0 (all tensors in float64)."""

    as_of_date: datetime
    start_date: datetime
    payment_currency: str
    day_count_convention: str

    # Canonical asset ordering pi_lex and inverse pi_lex^{-1}
    original_identifiers: list[str]
    sorted_identifiers: list[str]
    lex_perm: npt.NDArray[np.int64]
    inv_lex_perm: npt.NDArray[np.int64]

    # Spot & reference spot vectors in sorted canonical order (torch.float64)
    spots_t0: torch.Tensor
    ref_spots: torch.Tensor
    moneyness_t0: torch.Tensor

    # Partitioned schedule indices (0-based into original M-period schedule)
    settled_indices: list[int]
    in_transit_indices: list[int]
    future_indices: list[int]

    # Markovian scalars collapsed from Fixings (t_m <= t_0)
    c_mem_0: float
    i_ki_past: float
    already_autocalled: bool
    autocalled_period_index: int | None
    pv_in_transit: float

    # Historical period records & pre-terminal cash flow rows (Periods with t_m <= t_0)
    historical_records: list[HistoricalPeriodRecord] = field(default_factory=list)
    past_cashflows: list[CashFlowItem] = field(default_factory=list)

    # Remaining future schedule (length M_rem)
    future_obs_dates: list[datetime] = field(default_factory=list)
    future_pay_dates: list[datetime] = field(default_factory=list)
    tau_obs: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    delta_tau: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    tau_pay: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    future_coupon_rates: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    sentinel_barriers: SentinelBarrierResult | None = None
    cash_to_settle_today: float = 0.0
    warnings: list[str] = field(default_factory=list)


def canonicalize_asset_order(
    identifiers: Sequence[str],
) -> tuple[list[str], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Sorts UnderlyingIdentifiers into strict ASCII lexicographical order."""
    n = len(identifiers)
    if n == 0:
        raise ValueError("UnderlyingIdentifiers must be non-empty.")
    lex_perm = np.argsort(np.asarray(identifiers, dtype=object), kind="mergesort").astype(np.int64)
    inv_lex_perm = np.empty(n, dtype=np.int64)
    inv_lex_perm[lex_perm] = np.arange(n, dtype=np.int64)
    sorted_ids = [str(identifiers[int(i)]) for i in lex_perm]
    return sorted_ids, lex_perm, inv_lex_perm


def evaluate_monitoring_index_np(
    perf: npt.NDArray[np.float64],
    index_name: str = "WorstOf",
    inv_lex_perm: npt.NDArray[np.int64] | Sequence[int] | None = None,
) -> npt.NDArray[np.float64] | float:
    """Evaluates any of the Murex basket performance functionals along the last axis in float64."""
    p = np.asarray(perf, dtype=np.float64)
    scalar_input = p.ndim == 1
    if scalar_input:
        p = p[None, :]

    idx = index_name.strip()
    if inv_lex_perm is not None and idx in PAIR_ORDER_DEPENDENT_INDICES:
        p = p[..., np.asarray(inv_lex_perm, dtype=np.int64)]

    if idx == "WorstOf" or idx == "Lookback":
        out = np.min(p, axis=-1)
    elif idx == "BestOf":
        out = np.max(p, axis=-1)
    elif idx in ("Average", "AverageOf", "MultiPerf"):
        out = np.mean(p, axis=-1)
    elif idx == "Dispersion":
        mean_p = np.mean(p, axis=-1, keepdims=True)
        out = np.sqrt(np.mean((p - mean_p) ** 2, axis=-1))
    elif idx == "RatioBestWorst":
        out = np.max(p, axis=-1) / np.maximum(np.min(p, axis=-1), 1e-12)
    elif idx == "RatioBestAverage":
        out = np.max(p, axis=-1) / np.maximum(np.mean(p, axis=-1), 1e-12)
    elif idx == "RatioAverageWorst":
        out = np.mean(p, axis=-1) / np.maximum(np.min(p, axis=-1), 1e-12)
    elif idx in ("DiffBestWorst", "Rainbow"):
        out = np.max(p, axis=-1) - np.min(p, axis=-1)
    elif idx == "DiffBestAverage":
        out = np.max(p, axis=-1) - np.mean(p, axis=-1)
    elif idx == "DiffAverageWorst":
        out = np.mean(p, axis=-1) - np.min(p, axis=-1)
    elif idx == "RatioAB":
        out = p[..., 0] if p.shape[-1] < 2 else p[..., 0] / np.maximum(p[..., 1], 1e-12)
    elif idx == "RatioBA":
        out = p[..., 0] if p.shape[-1] < 2 else p[..., 1] / np.maximum(p[..., 0], 1e-12)
    elif idx in ("DiffAB", "Outperformance", "Spread"):
        out = p[..., 0] if p.shape[-1] < 2 else p[..., 0] - p[..., 1]
    elif idx == "DiffBA":
        out = p[..., 0] if p.shape[-1] < 2 else p[..., 1] - p[..., 0]
    elif idx == "TwinWin":
        w_min = np.min(p, axis=-1)
        out = np.where(w_min < 1.0, 2.0 - w_min, w_min)
    else:
        raise ValueError(f"Unsupported MonitoringIndex: {index_name!r}")

    if scalar_input:
        return float(out[0])
    res_arr: npt.NDArray[np.float64] = np.asarray(out, dtype=np.float64)
    return res_arr


def evaluate_monitoring_index_torch(
    perf: torch.Tensor,
    index_name: str = "WorstOf",
    asset_mask: torch.Tensor | None = None,
    inv_lex_perm: torch.Tensor | Sequence[int] | None = None,
) -> torch.Tensor:
    """Differentiable torch.float64 evaluation of the Murex basket performance functionals."""
    p = perf.to(dtype=torch.float64)
    idx = index_name.strip()
    if inv_lex_perm is not None and idx in PAIR_ORDER_DEPENDENT_INDICES:
        perm_t = torch.as_tensor(inv_lex_perm, dtype=torch.long, device=p.device)
        p = p.index_select(-1, perm_t)
        if asset_mask is not None:
            asset_mask = asset_mask.index_select(-1, perm_t)

    if asset_mask is None:
        mask = torch.ones_like(p, dtype=torch.float64)
    else:
        mask = asset_mask.to(dtype=torch.float64, device=p.device)

    active_count = mask.sum(dim=-1).clamp(min=1.0)
    p_for_min = torch.where(mask > 0.5, p, torch.full_like(p, 1e12))
    p_for_max = torch.where(mask > 0.5, p, torch.full_like(p, -1e12))
    p_mean = (p * mask).sum(dim=-1) / active_count

    if idx in ("WorstOf", "Lookback"):
        return torch.min(p_for_min, dim=-1).values
    if idx == "BestOf":
        return torch.max(p_for_max, dim=-1).values
    if idx in ("Average", "AverageOf", "MultiPerf"):
        return p_mean
    if idx == "Dispersion":
        diff_sq = ((p - p_mean.unsqueeze(-1)) ** 2) * mask
        return torch.sqrt((diff_sq.sum(dim=-1) / active_count).clamp(min=1e-16))
    if idx == "RatioBestWorst":
        return torch.max(p_for_max, dim=-1).values / torch.min(p_for_min, dim=-1).values.clamp(min=1e-12)
    if idx == "RatioBestAverage":
        return torch.max(p_for_max, dim=-1).values / p_mean.clamp(min=1e-12)
    if idx == "RatioAverageWorst":
        return p_mean / torch.min(p_for_min, dim=-1).values.clamp(min=1e-12)
    if idx in ("DiffBestWorst", "Rainbow"):
        return torch.max(p_for_max, dim=-1).values - torch.min(p_for_min, dim=-1).values
    if idx == "DiffBestAverage":
        return torch.max(p_for_max, dim=-1).values - p_mean
    if idx == "DiffAverageWorst":
        return p_mean - torch.min(p_for_min, dim=-1).values
    if idx == "RatioAB":
        return p[..., 0] / p[..., 1].clamp(min=1e-12) if p.shape[-1] >= 2 else p[..., 0]
    if idx == "RatioBA":
        return p[..., 1] / p[..., 0].clamp(min=1e-12) if p.shape[-1] >= 2 else p[..., 0]
    if idx in ("DiffAB", "Outperformance", "Spread"):
        return p[..., 0] - p[..., 1] if p.shape[-1] >= 2 else p[..., 0]
    if idx == "DiffBA":
        return p[..., 1] - p[..., 0] if p.shape[-1] >= 2 else p[..., 0]
    if idx == "TwinWin":
        w_min = torch.min(p_for_min, dim=-1).values
        return torch.where(w_min < 1.0, 2.0 - w_min, w_min)
    raise ValueError(f"Unsupported MonitoringIndex: {index_name!r}")


def normalize_sentinel_barriers(
    ko_barriers: Sequence[float] | npt.NDArray[np.float64],
    cpn_barriers: Sequence[float] | npt.NDArray[np.float64],
    *,
    allow_autocall_on_expiry: bool = False,
    includes_expiry_period: bool = True,
    sentinel_ko_threshold: float = 10.0,
    max_ko_feature_level: float = 3.0,
    guaranteed_cpn_threshold: float = 1e-3,
    min_cpn_feature_level: float = 0.05,
) -> SentinelBarrierResult:
    """Normalizes sentinel lock-out autocall barriers and guaranteed coupon barriers."""
    ko_arr = np.asarray(ko_barriers, dtype=np.float64).copy()
    cpn_arr = np.asarray(cpn_barriers, dtype=np.float64).copy()
    m_len = len(ko_arr)
    if len(cpn_arr) != m_len:
        raise ValueError(
            f"Length mismatch between ko_barriers ({m_len}) and cpn_barriers ({len(cpn_arr)})."
        )

    is_callable = (ko_arr < sentinel_ko_threshold).astype(np.float64)
    if includes_expiry_period and m_len > 0 and not allow_autocall_on_expiry:
        is_callable[-1] = 0.0

    ko_clamped = np.clip(ko_arr, 1e-4, max_ko_feature_level)
    log_ko_feat = np.log(ko_clamped).astype(np.float64)

    is_guaranteed = (cpn_arr <= guaranteed_cpn_threshold).astype(np.float64)
    cpn_clamped = np.maximum(cpn_arr, min_cpn_feature_level)
    log_cpn_feat = np.log(cpn_clamped).astype(np.float64)

    return SentinelBarrierResult(
        ko_barriers_raw=ko_arr,
        cpn_barriers_raw=cpn_arr,
        is_callable=is_callable,
        log_ko_barrier_feat=log_ko_feat,
        is_guaranteed_cpn=is_guaranteed,
        log_cpn_barrier_feat=log_cpn_feat,
    )


_IndexedFixing = Tuple[dict[date, float], list[int], list[float]]


def _build_fixing_index(raw_dates: Sequence[Any], raw_vals: Sequence[Any]) -> _IndexedFixing:
    date_map: dict[date, float] = {}
    pairs: list[Tuple[int, float]] = []
    for d, v in zip(raw_dates, raw_vals):
        d_date = _to_utc_datetime(d).date()
        f_val = float(v)
        if d_date not in date_map:
            date_map[d_date] = f_val
        pairs.append((d_date.toordinal(), f_val))
    pairs.sort(key=lambda item: item[0])
    return date_map, [p[0] for p in pairs], [p[1] for p in pairs]


def _lookup_fixing_indexed(indexed: _IndexedFixing, target_date: datetime) -> float:
    date_map, sorted_ords, sorted_vals = indexed
    target_d = target_date.date()
    exact = date_map.get(target_d)
    if exact is not None:
        return exact
    target_ord = target_d.toordinal()
    pos = bisect_right(sorted_ords, target_ord) - 1
    if pos >= 0 and (target_ord - sorted_ords[pos]) <= 7:
        return sorted_vals[pos]
    raise KeyError(f"Missing historical fixing for observation date {target_d.isoformat()}.")


def _lookup_fixing_optional(indexed: _IndexedFixing | None, target_date: datetime) -> float | None:
    if indexed is None or not indexed[1]:
        return None
    date_map, sorted_ords, sorted_vals = indexed
    target_d = target_date.date()
    exact = date_map.get(target_d)
    if exact is not None:
        return exact
    target_ord = target_d.toordinal()
    pos = bisect_right(sorted_ords, target_ord) - 1
    if pos >= 0 and (target_ord - sorted_ords[pos]) <= 7:
        return sorted_vals[pos]
    return None


def _extract_fixings_entry(f_obj: Any) -> _IndexedFixing:
    if isinstance(f_obj, Mapping):
        raw_dates = f_obj.get("Dates", f_obj.get("dates", []))
        raw_vals = f_obj.get("Values", f_obj.get("values", []))
    else:
        raw_dates = getattr(f_obj, "Dates", getattr(f_obj, "dates", []))
        raw_vals = getattr(f_obj, "Values", getattr(f_obj, "values", []))
    return _build_fixing_index(raw_dates, raw_vals)


def _eval_terminal_option_payoff(
    *,
    perf_m: npt.NDArray[np.float64],
    w_opt: float,
    strike: float,
    lower_strike: float | None,
    option_gearing: float,
    ki_barrier: float | None,
    i_ki_past: float,
    is_long_basket: bool,
    participation_call_strike: float | None,
    participation_call_rate: float,
    one_star_strike: float | None,
    one_star_coupon: float,
) -> float:
    """Evaluates deterministic terminal Option payoff from observed expiry performances."""
    opt_paid = 0.0
    eff_w_opt = float(evaluate_monitoring_index_np(perf_m, "AverageOf")) if is_long_basket else float(w_opt)
    if strike > 1e-6 and option_gearing > 0.0:
        put_intrinsic = max(float(strike) - eff_w_opt, 0.0)
        if lower_strike is not None and float(lower_strike) > 0.0:
            put_intrinsic = min(put_intrinsic, max(float(strike) - float(lower_strike), 0.0))
        if ki_barrier is not None and float(ki_barrier) > 0.0 and i_ki_past < 0.5:
            if eff_w_opt >= float(ki_barrier):
                put_intrinsic = 0.0
        if one_star_strike is not None and float(one_star_strike) > 0.0:
            w_best = float(evaluate_monitoring_index_np(perf_m, "BestOf"))
            if w_best >= float(one_star_strike):
                put_intrinsic = 0.0
        put_sign = -1.0
        opt_paid += put_sign * float(option_gearing) * put_intrinsic

    if participation_call_strike is not None and float(participation_call_rate) > 0.0:
        opt_paid += float(participation_call_rate) * max(
            eff_w_opt - float(participation_call_strike), 0.0
        )

    if one_star_strike is not None and float(one_star_coupon) > 0.0:
        w_best = float(evaluate_monitoring_index_np(perf_m, "BestOf"))
        if w_best >= float(one_star_strike):
            opt_paid += float(one_star_coupon)

    return float(opt_paid)


def collapse_seasoned_schedule(
    *,
    as_of_date: datetime | date | str,
    start_date: datetime | date | str,
    observation_dates: Sequence[datetime | date | str],
    payment_dates: Sequence[datetime | date | str],
    underlying_identifiers: Sequence[str],
    underlying_reference_spots: Sequence[float],
    spots_t0: Mapping[str, float] | Sequence[float] | torch.Tensor,
    ko_barriers: Sequence[float],
    cpn_barriers: Sequence[float] | None = None,
    fixed_coupon_amount: float | Sequence[float] | None = None,
    fixed_rate: float | Sequence[float] | None = None,
    coupon_amounts: Sequence[float] | None = None,
    day_count_convention: str = "Act360",
    time_day_count_convention: str = "Act365",
    is_memory: bool = True,
    callability_index: str = "WorstOf",
    coupon_index: str | None = None,
    option_index: str | None = None,
    fixings: Mapping[str, Any] | None = None,
    ki_barrier: float | None = None,
    ki_monitored_past: bool = False,
    allow_autocall_on_expiry: bool = False,
    force_coupons_on_autocall: bool = False,
    autocall_coupon: float = 0.0,
    no_notional_repayment: bool = False,
    payment_currency: str = "RUB",
    df_pay_fn: Callable[[datetime], float] | None = None,
    coupon_barrier_type: str = "In",
    coupon_barrier_direction: str = "Up",
    strike: float = 0.90,
    option_gearing: float = 1.0,
    lower_strike: float | None = None,
    is_long_basket: bool = False,
    participation_call_strike: float | None = None,
    participation_call_rate: float = 0.0,
    one_star_strike: float | None = None,
    one_star_coupon: float = 0.0,
    funding_leg: Mapping[str, Any] | None = None,
) -> SeasonedScheduleState:
    """Partitions the observation schedule relative to AsOfDate and collapses past fixings in float64."""
    t0_dt = _to_utc_datetime(as_of_date)
    start_dt = _to_utc_datetime(start_date)
    obs_dts = [_to_utc_datetime(d) for d in observation_dates]
    pay_dts = [_to_utc_datetime(d) for d in payment_dates]
    m_total = len(obs_dts)
    if len(pay_dts) != m_total:
        raise ValueError(f"ObservationDates ({m_total}) and PaymentDates ({len(pay_dts)}) length mismatch.")

    sorted_ids, lex_perm, inv_lex_perm = canonicalize_asset_order(underlying_identifiers)
    d_assets = len(sorted_ids)

    ref_spots_orig = np.asarray(underlying_reference_spots, dtype=np.float64)
    if len(ref_spots_orig) != d_assets:
        raise ValueError("UnderlyingReferenceSpots length must match UnderlyingIdentifiers.")
    ref_spots_sorted = ref_spots_orig[lex_perm]
    ref_spots_t = torch.as_tensor(ref_spots_sorted, dtype=torch.float64)

    if isinstance(spots_t0, torch.Tensor):
        if spots_t0.dtype != torch.float64:
            spots_t0 = spots_t0.to(dtype=torch.float64)
        perm_t = torch.as_tensor(lex_perm, dtype=torch.long, device=spots_t0.device)
        spots_sorted_t = spots_t0.index_select(0, perm_t)
        ref_spots_t = ref_spots_t.to(device=spots_t0.device)
    elif isinstance(spots_t0, Mapping):
        spots_list = [float(spots_t0[ticker]) for ticker in sorted_ids]
        spots_sorted_t = torch.tensor(spots_list, dtype=torch.float64)
    else:
        spots_arr = np.asarray(spots_t0, dtype=np.float64)[lex_perm]
        spots_sorted_t = torch.as_tensor(spots_arr, dtype=torch.float64)

    moneyness_t0 = spots_sorted_t / ref_spots_t.clamp(min=1e-12)

    # Build per-period coupon amounts c_m supporting scalar and sequence inputs
    if coupon_amounts is not None:
        cpn_rates = np.asarray(coupon_amounts, dtype=np.float64)
    elif fixed_coupon_amount is not None:
        if isinstance(fixed_coupon_amount, Sequence) and not isinstance(fixed_coupon_amount, (str, bytes)):
            if len(fixed_coupon_amount) != m_total:
                raise ValueError(
                    f"FixedCouponAmount length ({len(fixed_coupon_amount)}) != number of periods ({m_total})."
                )
            cpn_rates = np.asarray(fixed_coupon_amount, dtype=np.float64)
        else:
            cpn_rates = np.full(m_total, float(fixed_coupon_amount), dtype=np.float64)
    elif fixed_rate is not None:
        cpn_rates = np.empty(m_total, dtype=np.float64)
        rate_seq: list[float] | None = (
            [float(x) for x in fixed_rate]
            if isinstance(fixed_rate, Sequence) and not isinstance(fixed_rate, (str, bytes))
            else None
        )
        if rate_seq is not None and len(rate_seq) != m_total:
            raise ValueError(
                f"FixedRate length ({len(rate_seq)}) != number of periods ({m_total})."
            )
        prev_d = start_dt
        for m in range(m_total):
            dcf_m = _year_fraction(prev_d, pay_dts[m], day_count_convention)
            rate_m = rate_seq[m] if rate_seq is not None else float(fixed_rate)  # type: ignore[arg-type]
            cpn_rates[m] = rate_m * dcf_m
            prev_d = pay_dts[m]
    else:
        cpn_rates = np.zeros(m_total, dtype=np.float64)

    ko_arr = np.asarray(ko_barriers, dtype=np.float64)
    if cpn_barriers is not None and len(cpn_barriers) > 0:
        cpn_arr = np.asarray(cpn_barriers, dtype=np.float64)
    else:
        cpn_arr = np.full(m_total, 1e-5, dtype=np.float64)

    eff_cpn_index = coupon_index if coupon_index else callability_index
    raw_opt_index = option_index if option_index else callability_index
    eff_opt_index = "AverageOf" if (is_long_basket and raw_opt_index == "WorstOf") else raw_opt_index
    type_out = coupon_barrier_type.strip().lower() == "out"
    dir_down = coupon_barrier_direction.strip().lower() == "down"

    # Pre-index Fixings dictionary for O(1) lookup
    indexed_fixings: dict[str, _IndexedFixing] = {}
    if fixings:
        for ticker in sorted_ids:
            if ticker in fixings:
                indexed_fixings[ticker] = _extract_fixings_entry(fixings[ticker])

    # Precompute past/in-transit FundingLeg amounts mapped to product periods
    past_funding_by_period: dict[int, float] = {}
    if funding_leg is not None and m_total > 0:
        f_rid = funding_leg.get("rate_identifier")
        f_spread = float(funding_leg.get("funding_spread", 0.0) or 0.0)
        f_dates = [_to_utc_datetime(d) for d in funding_leg.get("payment_dates", [])]
        rate_fix_idx: _IndexedFixing | None = None
        if f_rid is not None and fixings and f_rid in fixings:
            rate_fix_idx = _extract_fixings_entry(fixings[f_rid])
        f_curve = funding_leg.get("funding_curve")
        prev_f_dt = start_dt
        for f_dt in f_dates:
            if f_dt > pay_dts[-1] and f_dt > t0_dt:
                break
            if f_dt <= t0_dt:
                alpha_j = _year_fraction(prev_f_dt, f_dt, day_count_convention)
                r_fix = _lookup_fixing_optional(rate_fix_idx, prev_f_dt) if f_rid is not None else 0.0
                if r_fix is None and f_rid is not None:
                    r_fix = _lookup_fixing_optional(rate_fix_idx, f_dt)
                if r_fix is None and f_rid is not None and f_curve is not None and hasattr(f_curve, "zero_rate"):
                    r_fix = float(f_curve.zero_rate(f_dt)) if f_dt > t0_dt else 0.0
                r_val = float(r_fix or 0.0)
                amt_j = (r_val + f_spread) * alpha_j
                m_idx = m_total - 1
                for m_cand in range(m_total):
                    if pay_dts[m_cand] >= f_dt:
                        m_idx = m_cand
                        break
                past_funding_by_period[m_idx] = past_funding_by_period.get(m_idx, 0.0) + amt_j
            prev_f_dt = f_dt

    settled_indices: list[int] = []
    in_transit_indices: list[int] = []
    future_indices: list[int] = []

    c_mem_0 = 0.0
    i_ki_past = 0.0
    already_autocalled = False
    autocalled_period_index: int | None = None
    pv_in_transit = 0.0

    historical_records: list[HistoricalPeriodRecord] = []
    past_cashflows: list[CashFlowItem] = []

    for m in range(m_total):
        t_m = obs_dts[m]
        T_m = pay_dts[m]

        if T_m <= t0_dt:
            settled_indices.append(m)
            period_status: Literal["Settled", "In-Transit"] = "Settled"
            cf_status: Literal["Settled", "Estimate"] = "Settled"
        elif t_m <= t0_dt < T_m:
            in_transit_indices.append(m)
            period_status = "In-Transit"
            cf_status = "Estimate"
        else:
            future_indices.append(m)
            continue

        # Reconstruct fixing performances at t_m for all D assets in sorted_ids order
        perf_m = np.empty(d_assets, dtype=np.float64)
        for i, ticker in enumerate(sorted_ids):
            idx_fix = indexed_fixings.get(ticker)
            if idx_fix is not None and idx_fix[1]:
                fix_val = _lookup_fixing_indexed(idx_fix, t_m)
            elif t_m.date() == t0_dt.date():
                fix_val = float(spots_sorted_t[i].detach().cpu().item())
            else:
                raise KeyError(
                    f"Fixings missing for asset {ticker!r} on past observation date {t_m.date().isoformat()}."
                )
            perf_m[i] = fix_val / ref_spots_sorted[i]

        w_call = float(evaluate_monitoring_index_np(perf_m, callability_index, inv_lex_perm=inv_lex_perm))
        w_cpn = float(evaluate_monitoring_index_np(perf_m, eff_cpn_index, inv_lex_perm=inv_lex_perm))
        w_opt = float(evaluate_monitoring_index_np(perf_m, eff_opt_index, inv_lex_perm=inv_lex_perm))

        if ki_monitored_past and ki_barrier is not None and w_opt < float(ki_barrier):
            i_ki_past = 1.0

        b_ko = float(ko_arr[m])
        b_cpn = float(cpn_arr[m])
        c_m = float(cpn_rates[m])

        is_callable_m = (b_ko < 10.0) and (m < m_total - 1 or allow_autocall_on_expiry)
        df_m = float(df_pay_fn(T_m)) if (period_status == "In-Transit" and df_pay_fn is not None) else 1.0
        fund_amt_m = float(past_funding_by_period.get(m, 0.0))

        opt_paid = 0.0
        if already_autocalled:
            cpn_trig = False
            ko_trig = False
            cpn_paid = 0.0
            notional_paid = 0.0
            fund_amt_m = 0.0
        else:
            ko_trig = bool(is_callable_m and (w_call >= b_ko))
            hit_dir = (w_cpn <= b_cpn) if dir_down else (w_cpn >= b_cpn)
            base_cpn = (not hit_dir) if type_out else hit_dir
            is_guaranteed_m = (b_cpn <= 1e-3) and (not type_out) and (not dir_down)
            cpn_trig = bool(is_guaranteed_m or base_cpn or (ko_trig and force_coupons_on_autocall))

            if cpn_trig:
                cpn_paid = (c_m + c_mem_0) if is_memory else c_m
                c_mem_0 = 0.0
            else:
                cpn_paid = 0.0
                if is_memory:
                    c_mem_0 = float(c_mem_0 + c_m)

            if ko_trig:
                cpn_paid = float(cpn_paid + autocall_coupon)
                notional_paid = 0.0 if no_notional_repayment else 1.0
                already_autocalled = True
                autocalled_period_index = m
            elif m == m_total - 1:
                notional_paid = 0.0 if no_notional_repayment else 1.0
                opt_paid = _eval_terminal_option_payoff(
                    perf_m=perf_m,
                    w_opt=w_opt,
                    strike=strike,
                    lower_strike=lower_strike,
                    option_gearing=option_gearing,
                    ki_barrier=ki_barrier,
                    i_ki_past=i_ki_past,
                    is_long_basket=is_long_basket,
                    participation_call_strike=participation_call_strike,
                    participation_call_rate=participation_call_rate,
                    one_star_strike=one_star_strike,
                    one_star_coupon=one_star_coupon,
                )
            else:
                notional_paid = 0.0

        if period_status == "In-Transit":
            disc_cpn = float(df_m * cpn_paid)
            disc_notional = float(df_m * notional_paid)
            disc_opt = float(df_m * opt_paid)
            disc_fund = float(df_m * fund_amt_m)
            pv_in_transit = float(pv_in_transit + disc_cpn + disc_notional + disc_opt - disc_fund)
            cf_cpn_amt = disc_cpn
            cf_notional_amt = disc_notional
            cf_opt_amt = disc_opt
            cf_fund_amt = -disc_fund if disc_fund != 0.0 else 0.0
        else:
            cf_cpn_amt = float(cpn_paid)
            cf_notional_amt = float(notional_paid)
            cf_opt_amt = float(opt_paid)
            cf_fund_amt = -float(fund_amt_m) if fund_amt_m != 0.0 else 0.0

        historical_records.append(
            HistoricalPeriodRecord(
                period_index=m,
                observation_date=t_m,
                payment_date=T_m,
                status=period_status,
                asset_performances=perf_m,
                call_index_value=w_call,
                coupon_index_value=w_cpn,
                ko_barrier=b_ko,
                cpn_barrier=b_cpn,
                coupon_rate=c_m,
                coupon_triggered=cpn_trig,
                autocall_triggered=ko_trig,
                coupon_paid=cpn_paid,
                notional_paid=notional_paid,
                unpaid_memory_after=c_mem_0,
                discount_factor=df_m,
            )
        )

        past_cashflows.append(
            CashFlowItem(
                Description="FundingLeg",
                Date=T_m,
                Amount=cf_fund_amt,
                StandardError=0.0,
                Currency=payment_currency,
                Leg=0,
                StatusText=cf_status,
            )
        )
        past_cashflows.append(
            CashFlowItem(
                Description="Coupon",
                Date=T_m,
                Amount=cf_cpn_amt,
                StandardError=0.0,
                Currency=payment_currency,
                Leg=0,
                StatusText=cf_status,
            )
        )
        if m == m_total - 1:
            past_cashflows.append(
                CashFlowItem(
                    Description="Option",
                    Date=T_m,
                    Amount=cf_opt_amt,
                    StandardError=0.0,
                    Currency=payment_currency,
                    Leg=0,
                    StatusText=cf_status,
                )
            )
        past_cashflows.append(
            CashFlowItem(
                Description="Notional",
                Date=T_m,
                Amount=cf_notional_amt,
                StandardError=0.0,
                Currency=payment_currency,
                Leg=0,
                StatusText=cf_status,
            )
        )

    cash_to_settle_today = float(
        sum(cf.Amount for cf in past_cashflows if cf.Date.date() == t0_dt.date())
    )

    future_obs_dates = [obs_dts[m] for m in future_indices]
    future_pay_dates = [pay_dts[m] for m in future_indices]
    m_rem = len(future_indices)

    tau_obs = np.array(
        [_year_fraction(t0_dt, d, time_day_count_convention) for d in future_obs_dates],
        dtype=np.float64,
    )
    tau_pay = np.array(
        [_year_fraction(t0_dt, d, time_day_count_convention) for d in future_pay_dates],
        dtype=np.float64,
    )
    if m_rem > 0:
        delta_tau = np.empty(m_rem, dtype=np.float64)
        delta_tau[0] = tau_obs[0]
        if m_rem > 1:
            delta_tau[1:] = np.diff(tau_obs)
    else:
        delta_tau = np.empty(0, dtype=np.float64)

    future_ko = ko_arr[future_indices] if m_rem > 0 else np.empty(0, dtype=np.float64)
    future_cpn_bar = cpn_arr[future_indices] if m_rem > 0 else np.empty(0, dtype=np.float64)
    future_cpn_rates = cpn_rates[future_indices] if m_rem > 0 else np.empty(0, dtype=np.float64)

    sentinel_res = normalize_sentinel_barriers(
        future_ko,
        future_cpn_bar,
        allow_autocall_on_expiry=allow_autocall_on_expiry,
        includes_expiry_period=(m_rem > 0 and future_indices[-1] == m_total - 1),
    )

    return SeasonedScheduleState(
        as_of_date=t0_dt,
        start_date=start_dt,
        payment_currency=payment_currency,
        day_count_convention=day_count_convention,
        original_identifiers=list(underlying_identifiers),
        sorted_identifiers=sorted_ids,
        lex_perm=lex_perm,
        inv_lex_perm=inv_lex_perm,
        spots_t0=spots_sorted_t,
        ref_spots=ref_spots_t,
        moneyness_t0=moneyness_t0,
        settled_indices=settled_indices,
        in_transit_indices=in_transit_indices,
        future_indices=future_indices,
        c_mem_0=float(c_mem_0),
        i_ki_past=float(i_ki_past),
        already_autocalled=already_autocalled,
        autocalled_period_index=autocalled_period_index,
        pv_in_transit=float(pv_in_transit),
        historical_records=historical_records,
        past_cashflows=past_cashflows,
        future_obs_dates=future_obs_dates,
        future_pay_dates=future_pay_dates,
        tau_obs=tau_obs,
        delta_tau=delta_tau,
        tau_pay=tau_pay,
        future_coupon_rates=future_cpn_rates,
        sentinel_barriers=sentinel_res,
        cash_to_settle_today=cash_to_settle_today,
    )


__all__ = [
    "CashFlowItem",
    "SentinelBarrierResult",
    "HistoricalPeriodRecord",
    "SeasonedScheduleState",
    "canonicalize_asset_order",
    "evaluate_monitoring_index_np",
    "evaluate_monitoring_index_torch",
    "normalize_sentinel_barriers",
    "collapse_seasoned_schedule",
]
