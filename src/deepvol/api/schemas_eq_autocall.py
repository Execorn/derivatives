"""schemas_eq_autocall.py — Canonical Pydantic V2 schemas & Murex DAG parser for Multi-Asset Equity Autocallables.

Provides canonical validation models and serialization helpers:
- Strict Pydantic V2 models for EQAutocallableTradeRequest, AutoCallableProductSpec, DupireModelSpec,
  MarketDataSetSpec, DiscountFactorCurveSpec, RepoSpreadCurveSpec, DividendsSpec,
  EquityVolSmileCollectionSpec, FixingsSpec, CashFlowRow, and EQAutocallableTradeResponse.
- Tier 0 legacy MC/engine parameter stripping (CalculationMethod, NumberOfPaths, BatchSizeForMCPath
  at both root and GreekMetricsSettings, StaticDataSet, GreekMetricsSettings and its sub-fields,
  UseNumeraireCurrencyParameterForCrossCurrency, InterpolationLeftLength, SpreadType, BaseCurve)
  with audit tracking in `stripped_legacy_fields` and `extra="forbid"` on unknown typos.
- Lossless parser `parse_murex_autocall_dag()` for Murex `ConstantMessageMap` content-addressable DAGs.
"""

from __future__ import annotations

from collections.abc import Mapping as AbcMapping, Sequence as AbcSequence
import datetime
from enum import Enum
import functools
import math
from pathlib import Path
import re
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────


class AutocallQueryMetric(str, Enum):
    """Supported valuation and Greek sensitivity query metrics."""

    ProductValue = "ProductValue"
    CashFlows = "CashFlows"
    Delta = "Delta"
    Gamma = "Gamma"
    MultivariateAggregatedCashGamma = "MultivariateAggregatedCashGamma"
    Vega = "Vega"
    Rho = "Rho"
    DividendsSensitivities = "DividendsSensitivities"
    Theta = "Theta"
    Cega = "Cega"
    ConvexCega = "ConvexCega"

    # Uppercase aliases
    PRODUCT_VALUE = "ProductValue"
    CASH_FLOWS = "CashFlows"
    DELTA = "Delta"
    GAMMA = "Gamma"
    MULTIVARIATE_AGGREGATED_CASH_GAMMA = "MultivariateAggregatedCashGamma"
    VEGA = "Vega"
    RHO = "Rho"
    DIVIDENDS_SENSITIVITIES = "DividendsSensitivities"
    THETA = "Theta"
    CEGA = "Cega"
    CONVEX_CEGA = "ConvexCega"


class MonitoringIndexEnum(str, Enum):
    """Basket performance monitoring functionals for callability, coupon, and option barriers."""

    WorstOf = "WorstOf"
    BestOf = "BestOf"
    Average = "Average"
    AverageOf = "AverageOf"
    Outperformance = "Outperformance"
    TwinWin = "TwinWin"
    Rainbow = "Rainbow"
    Lookback = "Lookback"
    Spread = "Spread"
    Dispersion = "Dispersion"
    RatioAB = "RatioAB"
    Ratio = "Ratio"
    RatioBA = "RatioBA"
    RatioBestWorst = "RatioBestWorst"
    RatioBestAverage = "RatioBestAverage"
    RatioAverageWorst = "RatioAverageWorst"
    DiffAB = "DiffAB"
    DiffBA = "DiffBA"
    DiffBestWorst = "DiffBestWorst"
    SpreadBestWorst = "SpreadBestWorst"
    DiffBestAverage = "DiffBestAverage"
    SpreadBestAverage = "SpreadBestAverage"
    DiffAverageWorst = "DiffAverageWorst"
    SpreadAverageWorst = "SpreadAverageWorst"
    EquallyWeightedBasket = "EquallyWeightedBasket"
    SecondWorstOf = "SecondWorstOf"
    ThirdWorstOf = "ThirdWorstOf"
    SecondBestOf = "SecondBestOf"
    MultiPerf = "MultiPerf"

    # Uppercase aliases
    WORST_OF = "WorstOf"
    BEST_OF = "BestOf"
    AVERAGE = "Average"
    AVERAGE_OF = "AverageOf"
    OUTPERFORMANCE = "Outperformance"
    TWIN_WIN = "TwinWin"
    RAINBOW = "Rainbow"
    LOOKBACK = "Lookback"
    SPREAD = "Spread"
    DISPERSION = "Dispersion"
    RATIO_AB = "RatioAB"
    RATIO = "Ratio"
    RATIO_BA = "RatioBA"
    RATIO_BEST_WORST = "RatioBestWorst"
    RATIO_BEST_AVERAGE = "RatioBestAverage"
    RATIO_AVERAGE_WORST = "RatioAverageWorst"
    DIFF_AB = "DiffAB"
    DIFF_BA = "DiffBA"
    DIFF_BEST_WORST = "DiffBestWorst"
    SPREAD_BEST_WORST = "SpreadBestWorst"
    DIFF_BEST_AVERAGE = "DiffBestAverage"
    SPREAD_BEST_AVERAGE = "SpreadBestAverage"
    DIFF_AVERAGE_WORST = "DiffAverageWorst"
    SPREAD_AVERAGE_WORST = "SpreadAverageWorst"
    EQUALLY_WEIGHTED_BASKET = "EquallyWeightedBasket"
    SECOND_WORST_OF = "SecondWorstOf"
    THIRD_WORST_OF = "ThirdWorstOf"
    SECOND_BEST_OF = "SecondBestOf"
    MULTI_PERF = "MultiPerf"


class DayCountConventionEnum(str, Enum):
    """Supported day-count conventions across product schedules, yield/repo curves, and vol surfaces."""

    Act365 = "Act365"
    Act360 = "Act360"
    ActAct = "ActAct"
    Thirty360 = "Thirty360"
    D30360 = "30360"
    Business252 = "Business252"
    Bus252 = "Bus252"
    Act252 = "Act252"
    ThirtyE360 = "30E360"
    Act364 = "Act364"
    Act366 = "Act366"
    Act365F = "Act365F"
    Act365_25 = "Act365_25"
    Act366_1 = "Act366_1"
    Act366_2 = "Act366_2"
    Act366_3 = "Act366_3"

    # Uppercase aliases
    ACT_365 = "Act365"
    ACT_360 = "Act360"
    ACT_ACT = "ActAct"
    THIRTY_360 = "Thirty360"
    BUSINESS_252 = "Business252"
    BUS_252 = "Bus252"
    ACT_252 = "Act252"
    THIRTY_E_360 = "30E360"
    ACT_364 = "Act364"
    ACT_366 = "Act366"
    ACT_365F = "Act365F"
    ACT_365_25 = "Act365_25"
    ACT_366_1 = "Act366_1"
    ACT_366_2 = "Act366_2"
    ACT_366_3 = "Act366_3"


class CurveInterpolationMethodEnum(str, Enum):
    """Supported discount/repo/smile curve interpolation methods."""

    CubicSpline = "CubicSpline"
    Linear = "Linear"
    Smooth = "Smooth"
    LogLinear = "LogLinear"
    FlatLeft = "FlatLeft"
    FlatRight = "FlatRight"
    MonotoneCubic = "MonotoneCubic"
    PCHIP = "PCHIP"
    Bessel = "Bessel"
    Harmonic = "Harmonic"
    PiecewiseConstantForward = "PiecewiseConstantForward"
    NaturalCubicSpline = "NaturalCubicSpline"
    NotAKnotCubicSpline = "NotAKnotCubicSpline"
    ClampedCubicSpline = "ClampedCubicSpline"
    Akima = "Akima"
    Kruger = "Kruger"
    NelsonSiegel = "NelsonSiegel"
    Svensson = "Svensson"
    QuadraticSpline = "QuadraticSpline"
    LogDiscountLinear = "LogDiscountLinear"

    # Additional Murex schema enum aliases
    PiecewiseConstantToTheRight = "PiecewiseConstantToTheRight"
    PiecewiseConstantToTheLeft = "PiecewiseConstantToTheLeft"
    LME = "LME"
    Polynomial = "Polynomial"
    Spline = "Spline"
    Rational = "Rational"
    FwdFwdQuartic = "FwdFwdQuartic"
    FwdFwdLinear = "FwdFwdLinear"
    ExpSpline = "ExpSpline"
    ConstrainedCubicSpline = "ConstrainedCubicSpline"
    Quadratic = "Quadratic"
    AdaptiveSpline = "AdaptiveSpline"
    NelsonSiegelContinuous = "NelsonSiegelContinuous"
    NelsonSiegelDiscrete = "NelsonSiegelDiscrete"
    MonotoneConvex = "MonotoneConvex"

    # Uppercase aliases
    CUBIC_SPLINE = "CubicSpline"
    LINEAR = "Linear"
    SMOOTH = "Smooth"
    LOG_LINEAR = "LogLinear"
    FLAT_LEFT = "FlatLeft"
    FLAT_RIGHT = "FlatRight"
    MONOTONE_CUBIC = "MonotoneCubic"
    BESSEL = "Bessel"
    HARMONIC = "Harmonic"
    PIECEWISE_CONSTANT_FORWARD = "PiecewiseConstantForward"
    NATURAL_CUBIC_SPLINE = "NaturalCubicSpline"
    NOT_A_KNOT_CUBIC_SPLINE = "NotAKnotCubicSpline"
    CLAMPED_CUBIC_SPLINE = "ClampedCubicSpline"
    AKIMA = "Akima"
    KRUGER = "Kruger"
    NELSON_SIEGEL = "NelsonSiegel"
    SVENSSON = "Svensson"
    QUADRATIC_SPLINE = "QuadraticSpline"
    LOG_DISCOUNT_LINEAR = "LogDiscountLinear"


class InterpolationSpaceCoordinatesEnum(str, Enum):
    """Curve interpolation coordinate spaces."""

    Rates = "Rates"
    LogarithmOfDiscountFactors = "LogarithmOfDiscountFactors"
    Factors = "Factors"
    DiscountFactors = "DiscountFactors"


class ExtrapolationMethodEnum(str, Enum):
    """Curve extrapolation methods."""

    Flat = "Flat"
    Linear = "Linear"
    FlatLeft = "FlatLeft"
    FlatRight = "FlatRight"


class RateQuoteExpressionEnum(str, Enum):
    """Rate quote unit expressions."""

    Percentage = "Percentage"
    Decimal = "Decimal"
    Absolute = "Absolute"


# ── Tier 0 Legacy Field Sets ──────────────────────────────────────────────────

_ROOT_TIER0_FIELDS: frozenset[str] = frozenset(
    {
        "CalculationMethod",
        "NumberOfPaths",
        "BatchSizeForMCPath",
        "RandomNumberGenerator",
        "TimeSteps",
        "StaticDataSet",
        "GreekMetricsSettings",
    }
)

_GREEK_SETTINGS_TIER0_FIELDS: frozenset[str] = frozenset(
    {
        "DifferentiationMethod",
        "BumpParametersOverrides",
        "MetricMapping",
        "StepSize",
        "BatchSizeForMCPath",
        "NumberGenerationMethod",
        "NumberGenerationClass",
        "PathGenerationMethod",
    }
)

_BUMP_OVERRIDE_TIER0_FIELDS: frozenset[str] = frozenset(
    {
        "MetricMapping",
        "StepSize",
    }
)

_MODEL_TIER0_FIELDS: frozenset[str] = frozenset(
    {
        "UseNumeraireCurrencyParameterForCrossCurrency",
    }
)

_CURVE_TIER0_FIELDS: frozenset[str] = frozenset(
    {
        "InterpolationLeftLength",
        "SpreadType",
        "BaseCurve",
    }
)


@functools.lru_cache(maxsize=4096)
def _parse_datetime_str(val: str) -> datetime.datetime:
    s = val.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    if dt.tzinfo is datetime.timezone.utc:
        return dt
    return dt.astimezone(datetime.timezone.utc)


def _normalize_datetime(val: Any) -> datetime.datetime:
    """Normalizes a date, datetime, or ISO string to a UTC-aware datetime.datetime."""
    if type(val) is str:
        return _parse_datetime_str(val)
    if isinstance(val, datetime.datetime):
        if val.tzinfo is datetime.timezone.utc:
            return val
        if val.tzinfo is None:
            return val.replace(tzinfo=datetime.timezone.utc)
        return val.astimezone(datetime.timezone.utc)
    if isinstance(val, datetime.date):
        return datetime.datetime(val.year, val.month, val.day, 0, 0, tzinfo=datetime.timezone.utc)
    if isinstance(val, str):
        return _parse_datetime_str(str(val))
    raise TypeError(f"Cannot convert {type(val).__name__} to datetime.datetime")


def _normalize_datetime_list(vals: Any) -> List[datetime.datetime]:
    if not isinstance(vals, (list, tuple)) and (
        isinstance(vals, (str, bytes)) or not isinstance(vals, AbcSequence)
    ):
        raise TypeError("Expected a sequence of dates/datetimes")
    parse_str = _parse_datetime_str
    norm_dt = _normalize_datetime
    return [parse_str(v) if type(v) is str else norm_dt(v) for v in vals]


def _append_unique(target: List[str], items: Sequence[str]) -> None:
    for item in items:
        if item not in target:
            target.append(item)


_STRICT_CONFIG: ConfigDict = ConfigDict(extra="forbid", allow_inf_nan=False)
_FLOAT_INT_TYPES: frozenset[type] = frozenset((float, int))


def _coalesce_stripped_legacy_fields(raw_stripped: Any) -> Any:
    if raw_stripped is None:
        return []
    if isinstance(raw_stripped, (list, tuple)):
        return [str(x) for x in raw_stripped]
    return raw_stripped


def _check_finite_sequence(vals: Sequence[float], label: str) -> None:
    if isinstance(vals, np.ndarray):
        if vals.dtype == np.bool_ or not np.issubdtype(vals.dtype, np.number):
            raise ValueError(f"{label} elements must be finite floats")
        if vals.size > 0 and not bool(np.isfinite(vals).all()):
            raise ValueError(f"{label} elements must be finite floats")
        return
    if not vals:
        return
    if set(map(type, vals)).issubset(_FLOAT_INT_TYPES):
        if all(map(math.isfinite, vals)):
            return
        raise ValueError(f"{label} elements must be finite floats")
    isfinite = math.isfinite
    for x in vals:
        if isinstance(x, bool):
            raise ValueError(f"{label} elements must be finite floats")
        try:
            fx = float(x)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} elements must be finite floats") from exc
        if not isfinite(fx):
            raise ValueError(f"{label} elements must be finite floats")


def _check_finite_nested(obj: Any, path: str = "response") -> None:
    if obj is None or isinstance(obj, (str, bool)):
        return
    if isinstance(obj, (int, float)):
        if not math.isfinite(float(obj)):
            raise ValueError(f"{path} contains non-finite numeric value: {obj!r}")
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            _check_finite_nested(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            _check_finite_nested(item, f"{path}[{idx}]")


# ── Market Data Leaf Specifications ───────────────────────────────────────────


class RatesScheduleSpec(BaseModel):
    """Pillar dates, percentage/decimal zero rates, and optional basis spreads for a DiscountFactorCurve."""

    model_config = _STRICT_CONFIG

    Dates: List[datetime.datetime]
    Rates: List[float] = Field(default_factory=list)
    Spreads: Optional[List[float]] = None

    @field_validator("Dates", mode="before")
    @classmethod
    def _parse_dates(cls, v: Any) -> List[datetime.datetime]:
        return _normalize_datetime_list(v)

    @model_validator(mode="after")
    def _validate_lengths(self) -> RatesScheduleSpec:
        if len(self.Dates) == 0:
            raise ValueError("RatesSchedule.Dates must not be empty")
        if self.Rates:
            if len(self.Rates) != len(self.Dates):
                raise ValueError(
                    f"RatesSchedule.Rates length ({len(self.Rates)}) != Dates length ({len(self.Dates)})"
                )
            _check_finite_sequence(self.Rates, "RatesSchedule.Rates")
        if self.Spreads is not None:
            if len(self.Spreads) != len(self.Dates):
                raise ValueError(
                    f"RatesSchedule.Spreads length ({len(self.Spreads)}) != Dates length ({len(self.Dates)})"
                )
            _check_finite_sequence(self.Spreads, "RatesSchedule.Spreads")
        if not self.Rates and self.Spreads is None:
            raise ValueError("RatesSchedule requires either Rates or Spreads")
        return self


class DiscountFactorCurveSpec(BaseModel):
    """Discount factor / yield curve specification with legacy field stripping."""

    model_config = _STRICT_CONFIG

    TypeName: Literal["DiscountFactorCurve"] = "DiscountFactorCurve"
    Identifier: str
    Currency: str = "RUB"
    AsOfDate: datetime.datetime
    InterpolationDayCountConvention: DayCountConventionEnum = DayCountConventionEnum.Act365
    InterpolationMethod: CurveInterpolationMethodEnum = CurveInterpolationMethodEnum.CubicSpline
    InterpolationSpaceCoordinates: InterpolationSpaceCoordinatesEnum = (
        InterpolationSpaceCoordinatesEnum.Rates
    )
    ExtrapolationMethod: ExtrapolationMethodEnum = ExtrapolationMethodEnum.Flat
    RateQuoteExpression: RateQuoteExpressionEnum = RateQuoteExpressionEnum.Percentage
    RatesSchedule: RatesScheduleSpec
    stripped_legacy_fields: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _strip_tier0_curve_fields(cls, data: Any) -> Any:
        if not isinstance(data, (dict, AbcMapping)):
            return data
        out = dict(data)
        stripped = _coalesce_stripped_legacy_fields(out.get("stripped_legacy_fields"))
        if isinstance(stripped, list):
            for field_name in _CURVE_TIER0_FIELDS:
                if field_name in out:
                    out.pop(field_name)
                    _append_unique(stripped, [field_name])
        out["stripped_legacy_fields"] = stripped
        return out

    @field_validator("InterpolationDayCountConvention", mode="before")
    @classmethod
    def _coalesce_dcc(cls, v: Any) -> Any:
        return DayCountConventionEnum.Act365 if v is None else v

    @field_validator("AsOfDate", mode="before")
    @classmethod
    def _parse_as_of(cls, v: Any) -> datetime.datetime:
        return _normalize_datetime(v)


_PERIOD_DAYS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([DWMYdwmy])\s*$")


@functools.lru_cache(maxsize=512)
def _parse_repo_period_days(p: str) -> float:
    m = _PERIOD_DAYS_RE.match(p)
    if m is None:
        raise ValueError(f"Invalid repo period format '{p}'; expected '<n>D'")
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
    raise ValueError(f"Invalid repo period format '{p}'; expected '<n>D'")


class RepoScheduleSpec(BaseModel):
    """Pillar periods ('<n>D') and percentage repo spread rates for a RepoSpreadCurve."""

    model_config = _STRICT_CONFIG

    Periods: List[str]
    Rates: List[float]

    @model_validator(mode="after")
    def _validate_schedule(self) -> RepoScheduleSpec:
        if len(self.Periods) != len(self.Rates):
            raise ValueError(
                f"RepoSchedule.Periods length ({len(self.Periods)}) != Rates length ({len(self.Rates)})"
            )
        if len(self.Periods) == 0:
            raise ValueError("RepoSchedule.Periods must not be empty")
        _check_finite_sequence(self.Rates, "RepoSchedule.Rates")
        _ = self.period_days()
        return self

    def period_days(self) -> List[float]:
        """Parses '<n>D' period strings into day counts in float64."""
        parse_p = _parse_repo_period_days
        return [parse_p(p) for p in self.Periods]


class RepoSpreadCurveSpec(BaseModel):
    """Equity repo spread curve specification (`Act360`, `<n>D` pillar periods)."""

    model_config = _STRICT_CONFIG

    TypeName: Literal["RepoSpreadCurve"] = "RepoSpreadCurve"
    Identifier: str
    InterpolationDayCountConvention: DayCountConventionEnum = DayCountConventionEnum.Act360
    InterpolationMethod: CurveInterpolationMethodEnum = CurveInterpolationMethodEnum.CubicSpline
    ExtrapolationMethod: ExtrapolationMethodEnum = ExtrapolationMethodEnum.Flat
    InterpolationSpaceCoordinates: InterpolationSpaceCoordinatesEnum = (
        InterpolationSpaceCoordinatesEnum.Rates
    )
    RateQuoteExpression: RateQuoteExpressionEnum = RateQuoteExpressionEnum.Percentage
    RatesSchedule: RepoScheduleSpec


class DividendItemSpec(BaseModel):
    """Single discrete cash dividend entry (`ExDate`, `PaymentDate`, `Amount`)."""

    model_config = _STRICT_CONFIG

    ExDate: datetime.datetime
    PaymentDate: datetime.datetime
    Amount: float

    @field_validator("ExDate", "PaymentDate", mode="before")
    @classmethod
    def _parse_dt(cls, v: Any) -> datetime.datetime:
        return _normalize_datetime(v)

    @model_validator(mode="after")
    def _validate_amount(self) -> DividendItemSpec:
        if isinstance(self.Amount, bool) or not math.isfinite(self.Amount) or self.Amount < 0.0:
            raise ValueError(f"DividendItem.Amount must be finite and non-negative, got {self.Amount!r}")
        return self


class DividendsSpec(BaseModel):
    """Discrete cash dividend schedule for a single equity underlying."""

    model_config = _STRICT_CONFIG

    TypeName: Literal["Dividends"] = "Dividends"
    Identifier: str
    CashDividendList: List[DividendItemSpec] = Field(default_factory=list)
    Currency: Optional[str] = None

    @field_validator("CashDividendList", mode="before")
    @classmethod
    def _coalesce_cash_divs(cls, v: Any) -> Any:
        return [] if v is None else v


class PredefinedDividendsSpec(BaseModel):
    """Contract-level predefined dividend schedule (`ExDates`, `Amounts`) from the AUTOCALL PERF block."""

    model_config = _STRICT_CONFIG

    ExDates: List[datetime.datetime]
    Amounts: List[float]
    Currency: Optional[str] = None

    @field_validator("ExDates", mode="before")
    @classmethod
    def _parse_dates(cls, v: Any) -> List[datetime.datetime]:
        return _normalize_datetime_list(v)

    @model_validator(mode="after")
    def _validate_lengths(self) -> PredefinedDividendsSpec:
        if len(self.ExDates) != len(self.Amounts):
            raise ValueError(
                f"PredefinedDividends ExDates length ({len(self.ExDates)}) != Amounts length ({len(self.Amounts)})"
            )
        if any((not math.isfinite(a)) or a < 0.0 for a in self.Amounts):
            raise ValueError("PredefinedDividends.Amounts must be finite and non-negative")
        for idx in range(1, len(self.ExDates)):
            if self.ExDates[idx] < self.ExDates[idx - 1]:
                raise ValueError("PredefinedDividends.ExDates must be sorted in ascending order")
        return self


class EquityVolSmileCollectionSpec(BaseModel):
    """Implied volatility surface (`[maturity][strike]` decimal volatility quotes) for an equity underlying."""

    model_config = _STRICT_CONFIG

    Identifier: str
    TypeName: Literal["EquityVolatilitySmileCollection"] = "EquityVolatilitySmileCollection"
    DayCountConvention: DayCountConventionEnum = DayCountConventionEnum.Act365
    SmileInterpolationMethod: CurveInterpolationMethodEnum = CurveInterpolationMethodEnum.CubicSpline
    SmileExtrapolationMethod: ExtrapolationMethodEnum = ExtrapolationMethodEnum.Linear
    ExpirationDates: List[datetime.datetime]
    Strikes: List[float]
    VolatilityQuotes: List[List[float]]
    Currency: Optional[str] = None

    @field_validator("DayCountConvention", mode="before")
    @classmethod
    def _coalesce_dcc(cls, v: Any) -> Any:
        return DayCountConventionEnum.Act365 if v is None else v

    @field_validator("ExpirationDates", mode="before")
    @classmethod
    def _parse_exp_dates(cls, v: Any) -> List[datetime.datetime]:
        return _normalize_datetime_list(v)

    @model_validator(mode="after")
    def _validate_surface_dimensions(self) -> EquityVolSmileCollectionSpec:
        n_exp = len(self.ExpirationDates)
        n_str = len(self.Strikes)
        if n_exp == 0 or n_str == 0:
            raise ValueError("EquityVolatilitySmileCollection requires non-empty ExpirationDates and Strikes")
        _check_finite_sequence(self.Strikes, "EquityVolatilitySmileCollection.Strikes")
        if len(self.VolatilityQuotes) != n_exp:
            raise ValueError(
                f"VolatilityQuotes row count ({len(self.VolatilityQuotes)}) != ExpirationDates length ({n_exp})"
            )
        for row_idx, row in enumerate(self.VolatilityQuotes):
            if len(row) != n_str:
                raise ValueError(
                    f"VolatilityQuotes[{row_idx}] length ({len(row)}) != Strikes length ({n_str})"
                )
            _check_finite_sequence(row, f"EquityVolatilitySmileCollection.VolatilityQuotes[{row_idx}]")
        return self


class FXVolatilitySmileCollectionSpec(BaseModel):
    """FX implied volatility surface (`[maturity][delta]` volatility quotes) covering all 12 Murex FX surface fields."""

    model_config = _STRICT_CONFIG

    Identifier: str
    TypeName: Literal["FXVolatilitySmileCollection"] = "FXVolatilitySmileCollection"
    AsOfDate: datetime.datetime
    Calendar: Optional[str] = None
    DayCountConvention: DayCountConventionEnum = DayCountConventionEnum.Act365
    SmileInterpolationMethod: CurveInterpolationMethodEnum = CurveInterpolationMethodEnum.CubicSpline
    SmileExtrapolationMethod: ExtrapolationMethodEnum = ExtrapolationMethodEnum.Flat
    ForwardObject: Optional[str] = None
    FinancingCurve: Optional[str] = None
    Deltas: List[str]
    Tenors: List[str]
    VolatilityQuotes: List[List[float]]

    @field_validator("DayCountConvention", mode="before")
    @classmethod
    def _coalesce_dcc(cls, v: Any) -> Any:
        return DayCountConventionEnum.Act365 if v is None else v

    @field_validator("AsOfDate", mode="before")
    @classmethod
    def _parse_as_of_date(cls, v: Any) -> datetime.datetime:
        return _normalize_datetime(v)

    @model_validator(mode="after")
    def _validate_fx_surface_dimensions(self) -> FXVolatilitySmileCollectionSpec:
        n_ten = len(self.Tenors)
        n_del = len(self.Deltas)
        if n_ten == 0 or n_del == 0:
            raise ValueError("FXVolatilitySmileCollection requires non-empty Tenors and Deltas")
        if len(self.VolatilityQuotes) != n_ten:
            raise ValueError(
                f"VolatilityQuotes row count ({len(self.VolatilityQuotes)}) != Tenors length ({n_ten})"
            )
        for row_idx, row in enumerate(self.VolatilityQuotes):
            if len(row) != n_del:
                raise ValueError(
                    f"VolatilityQuotes[{row_idx}] length ({len(row)}) != Deltas length ({n_del})"
                )
            _check_finite_sequence(row, f"FXVolatilitySmileCollection.VolatilityQuotes[{row_idx}]")
        return self


VolatilitySurfaceSpec = Annotated[
    Union[EquityVolSmileCollectionSpec, FXVolatilitySmileCollectionSpec],
    Field(discriminator="TypeName"),
]


class FixingsSpec(BaseModel):
    """Historical fixing dates and observed price/rate values for a seasoned underlying or rate index."""

    model_config = _STRICT_CONFIG

    TypeName: Literal["Fixings"] = "Fixings"
    Identifier: str
    Dates: List[datetime.datetime]
    Values: List[float]

    @field_validator("Dates", mode="before")
    @classmethod
    def _parse_dates(cls, v: Any) -> List[datetime.datetime]:
        return _normalize_datetime_list(v)

    @model_validator(mode="after")
    def _validate_lengths(self) -> FixingsSpec:
        if len(self.Dates) != len(self.Values):
            raise ValueError(
                f"Fixings Dates length ({len(self.Dates)}) != Values length ({len(self.Values)})"
            )
        _check_finite_sequence(self.Values, "Fixings.Values")
        return self


class CorrelationPairSpec(BaseModel):
    """Structured representation of a pairwise correlation entry `<asset_1>&<asset_2>`."""

    model_config = _STRICT_CONFIG

    asset_1: str
    asset_2: str
    correlation: float = Field(..., ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def _validate_finite_corr(self) -> CorrelationPairSpec:
        if isinstance(self.correlation, bool) or not math.isfinite(self.correlation):
            raise ValueError(f"Correlation must be finite, got {self.correlation!r}")
        return self

    @property
    def key(self) -> str:
        """Canonical lexicographically sorted `<name1>&<name2>` key."""
        a, b = sorted((self.asset_1, self.asset_2))
        return f"{a}&{b}"

    @classmethod
    def from_key(cls, pair_key: str, correlation: float) -> CorrelationPairSpec:
        """Constructs a `CorrelationPairSpec` from a `'NAME1&NAME2'` key and value."""
        parts = pair_key.split("&")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid correlation key '{pair_key}'; expected '<asset_1>&<asset_2>'")
        if isinstance(correlation, bool):
            raise ValueError(f"Correlation must be numeric, got {correlation!r}")
        c_val = float(correlation)
        if not math.isfinite(c_val):
            raise ValueError(f"Correlation for {pair_key!r} must be finite, got {correlation!r}")
        return cls(asset_1=parts[0], asset_2=parts[1], correlation=c_val)


class MarketDataSetSpec(BaseModel):
    """Complete multi-asset market data bundle (`RatesCurvesBundles`, `VolatilitySurfaces`, `Spots`, `Correlations`, etc.)."""

    model_config = _STRICT_CONFIG

    TypeName: Literal["MarketDataSet"] = "MarketDataSet"
    AsOfDate: datetime.datetime
    RatesCurvesBundles: Dict[str, DiscountFactorCurveSpec]
    VolatilitySurfaces: Dict[str, VolatilitySurfaceSpec]
    Spots: Dict[str, float]
    Correlations: Dict[str, float] = Field(default_factory=dict)
    RepoSpreadCurves: Dict[str, RepoSpreadCurveSpec] = Field(default_factory=dict)
    Dividends: Dict[str, DividendsSpec] = Field(default_factory=dict)
    Fixings: Dict[str, FixingsSpec] = Field(default_factory=dict)

    @field_validator("Correlations", "RepoSpreadCurves", "Dividends", mode="before")
    @classmethod
    def _coalesce_none_dicts(cls, v: Any) -> Any:
        return {} if v is None else v

    @field_validator("Fixings", mode="before")
    @classmethod
    def _coalesce_and_enrich_fixings(cls, v: Any) -> Any:
        if v is None:
            return {}
        if not isinstance(v, (dict, AbcMapping)):
            return v
        out_fix: Dict[str, Any] = {}
        for ticker, item in v.items():
            if isinstance(item, (dict, AbcMapping)) and not item.get("Identifier"):
                item_copy = dict(item)
                item_copy["Identifier"] = str(ticker)
                out_fix[ticker] = item_copy
            else:
                out_fix[ticker] = item
        return out_fix

    @field_validator("AsOfDate", mode="before")
    @classmethod
    def _parse_as_of(cls, v: Any) -> datetime.datetime:
        return _normalize_datetime(v)

    @field_validator("RatesCurvesBundles")
    @classmethod
    def _validate_rates_bundles(
        cls, v: Dict[str, DiscountFactorCurveSpec]
    ) -> Dict[str, DiscountFactorCurveSpec]:
        if len(v) == 0:
            raise ValueError("MarketDataSet.RatesCurvesBundles must not be empty")
        return v

    @field_validator("VolatilitySurfaces")
    @classmethod
    def _validate_vol_surfaces(
        cls, v: Dict[str, VolatilitySurfaceSpec]
    ) -> Dict[str, VolatilitySurfaceSpec]:
        if len(v) == 0:
            raise ValueError("MarketDataSet.VolatilitySurfaces must not be empty")
        return v

    @field_validator("Spots")
    @classmethod
    def _validate_spots(cls, v: Dict[str, float]) -> Dict[str, float]:
        if len(v) == 0:
            raise ValueError("MarketDataSet.Spots must not be empty")
        for ticker, spot in v.items():
            if isinstance(spot, bool):
                raise ValueError(f"MarketDataSet.Spots[{ticker!r}] must be float, got bool")
            s_val = float(spot)
            if not math.isfinite(s_val) or s_val <= 0.0:
                raise ValueError(
                    f"MarketDataSet.Spots[{ticker!r}] must be strictly positive and finite, got {spot!r}"
                )
        return v

    @field_validator("Correlations")
    @classmethod
    def _validate_correlations(cls, v: Dict[str, float]) -> Dict[str, float]:
        seen_pairs: Dict[Tuple[str, str], float] = {}
        for k, corr in v.items():
            pair = CorrelationPairSpec.from_key(k, corr)
            if pair.asset_1 == pair.asset_2 and abs(pair.correlation - 1.0) > 1e-12:
                raise ValueError(
                    f"Self-correlation for '{pair.asset_1}' must equal 1.0, got {pair.correlation}"
                )
            canon = (pair.asset_1, pair.asset_2) if pair.asset_1 <= pair.asset_2 else (pair.asset_2, pair.asset_1)
            prev = seen_pairs.get(canon)
            if prev is not None and abs(prev - pair.correlation) > 1e-12:
                raise ValueError(
                    f"Contradictory correlation entries for '{canon[0]}&{canon[1]}': {prev} vs {pair.correlation}"
                )
            seen_pairs[canon] = pair.correlation
        return v

    def get_correlation_pairs(self) -> List[CorrelationPairSpec]:
        """Returns all pairwise correlation entries as `CorrelationPairSpec` objects."""
        return [CorrelationPairSpec.from_key(k, val) for k, val in self.Correlations.items()]

    def get_correlation_matrix(self, identifiers: Sequence[str]) -> npt.NDArray[np.float64]:
        """Constructs the $D \\times D$ `np.float64` correlation matrix in the exact order of `identifiers`."""
        d = len(identifiers)
        mat = np.eye(d, dtype=np.float64)
        for i in range(d):
            id_i = identifiers[i]
            for j in range(i + 1, d):
                id_j = identifiers[j]
                if id_i == id_j:
                    mat[i, j] = 1.0
                    mat[j, i] = 1.0
                    continue
                k1 = f"{id_i}&{id_j}"
                k2 = f"{id_j}&{id_i}"
                if k1 in self.Correlations:
                    c_val = float(self.Correlations[k1])
                elif k2 in self.Correlations:
                    c_val = float(self.Correlations[k2])
                else:
                    raise KeyError(
                        f"Missing correlation pair for ({id_i!r}, {id_j!r}) in MarketDataSet.Correlations"
                    )
                mat[i, j] = c_val
                mat[j, i] = c_val
        return mat


class DupireModelSpec(BaseModel):
    """Dupire local volatility model specification with cross-currency flag stripping."""

    model_config = _STRICT_CONFIG

    TypeName: Literal["DupireLocalVolatilityModel"] = "DupireLocalVolatilityModel"
    MarketDataSet: MarketDataSetSpec
    FinancingCurveIdentifiers: List[str] = Field(default_factory=list)
    stripped_legacy_fields: List[str] = Field(default_factory=list)

    @field_validator("TypeName", mode="before")
    @classmethod
    def _coalesce_typename(cls, v: Any) -> Any:
        return "DupireLocalVolatilityModel" if v is None else v

    @field_validator("FinancingCurveIdentifiers", mode="before")
    @classmethod
    def _coalesce_fin_curves(cls, v: Any) -> Any:
        return [] if v is None else v

    @model_validator(mode="before")
    @classmethod
    def _strip_tier0_model_fields(cls, data: Any) -> Any:
        if not isinstance(data, (dict, AbcMapping)):
            return data
        out = dict(data)
        stripped = _coalesce_stripped_legacy_fields(out.get("stripped_legacy_fields"))
        if isinstance(stripped, list):
            for field_name in _MODEL_TIER0_FIELDS:
                if field_name in out:
                    out.pop(field_name)
                    _append_unique(stripped, [field_name])
        out["stripped_legacy_fields"] = stripped
        return out

    @model_validator(mode="after")
    def _collect_nested_stripped(self) -> DupireModelSpec:
        for curve in self.MarketDataSet.RatesCurvesBundles.values():
            _append_unique(self.stripped_legacy_fields, curve.stripped_legacy_fields)
        return self


class AutoCallableProductSpec(BaseModel):
    """Multi-asset Equity Autocallable product specification covering all 34 Murex contract fields."""

    model_config = _STRICT_CONFIG

    TypeName: Literal["AutoCallableProduct"] = "AutoCallableProduct"
    PaymentCurrency: str
    StartDate: datetime.datetime
    PaymentDates: List[datetime.datetime]
    ObservationDates: List[datetime.datetime]
    DayCountConvention: DayCountConventionEnum = DayCountConventionEnum.Act360
    AutoCallTriggerAbsoluteDigitalSpread: float
    Strike: float
    OptionGearing: float
    CallabilityMonitoringIndex: MonitoringIndexEnum = MonitoringIndexEnum.WorstOf
    AutoCallTriggerLevels: List[float]
    UnderlyingIdentifiers: List[str]
    UnderlyingReferenceSpots: List[float]

    # Optional payoff & schedule fields
    LowerStrike: Optional[float] = None
    ParticipationCallStrike: Optional[float] = None
    ParticipationCallRate: Optional[float] = None
    OneStarStrike: Optional[float] = None
    OneStarCoupon: Optional[float] = None
    ForceCouponsOnAutocall: bool = False
    AllowAutocallOnExpiry: bool = False
    IsLongBasket: bool = False
    FixedRate: Optional[Union[float, List[float]]] = None
    FixedCouponAmount: Optional[Union[float, List[float]]] = None
    CouponBarrierLevels: Optional[List[float]] = None
    CouponBarrierType: Optional[Literal["In", "Out"]] = "In"
    CouponBarrierDirection: Optional[Literal["Up", "Down"]] = "Up"
    CouponBarrierAbsoluteDigitalSpread: Optional[float] = None
    OptionDownAndInBarrierLevel: Optional[float] = None
    OptionDownAndInBarrierAbsoluteDigitalSpread: Optional[float] = None
    OptionDownAndInBarrierObservationType: Optional[
        Literal["European", "Discrete", "American", "Continuous"]
    ] = None
    CouponBarrierMonitoringIndex: Optional[MonitoringIndexEnum] = None
    OptionMonitoringIndex: Optional[MonitoringIndexEnum] = None
    RateIdentifier: Optional[str] = None
    FundingSpread: Optional[float] = None
    FundingLegPaymentDates: Optional[List[datetime.datetime]] = None
    NoNotionalRepayment: bool = False
    IsMemory: bool = False
    AutoCallCoupon: Optional[float] = None
    PredefinedDividends: Optional[Dict[str, PredefinedDividendsSpec]] = None

    @field_validator("DayCountConvention", mode="before")
    @classmethod
    def _coalesce_dcc(cls, v: Any) -> Any:
        return DayCountConventionEnum.Act360 if v is None else v

    @field_validator("CallabilityMonitoringIndex", mode="before")
    @classmethod
    def _coalesce_call_idx(cls, v: Any) -> Any:
        return MonitoringIndexEnum.WorstOf if v is None else v

    @field_validator("StartDate", mode="before")
    @classmethod
    def _parse_start_date(cls, v: Any) -> datetime.datetime:
        return _normalize_datetime(v)

    @field_validator("PaymentDates", "ObservationDates", mode="before")
    @classmethod
    def _parse_schedule_dates(cls, v: Any) -> List[datetime.datetime]:
        return _normalize_datetime_list(v)

    @field_validator("FundingLegPaymentDates", mode="before")
    @classmethod
    def _parse_funding_dates(cls, v: Any) -> Optional[List[datetime.datetime]]:
        if v is None:
            return None
        return _normalize_datetime_list(v)

    @model_validator(mode="after")
    def _validate_product_invariants(self) -> AutoCallableProductSpec:
        if self.FixedRate is not None and self.FixedCouponAmount is not None:
            raise ValueError(
                "FixedRate and FixedCouponAmount are mutually exclusive in AutoCallableProductSpec"
            )
        m_obs = len(self.ObservationDates)
        if m_obs == 0:
            raise ValueError("ObservationDates must not be empty")
        if len(self.PaymentDates) != m_obs:
            raise ValueError(
                f"PaymentDates length ({len(self.PaymentDates)}) != ObservationDates length ({m_obs})"
            )
        if len(self.AutoCallTriggerLevels) != m_obs:
            raise ValueError(
                f"AutoCallTriggerLevels length ({len(self.AutoCallTriggerLevels)}) != ObservationDates length ({m_obs})"
            )
        if self.CouponBarrierLevels is not None:
            if len(self.CouponBarrierLevels) != m_obs:
                raise ValueError(
                    f"CouponBarrierLevels length ({len(self.CouponBarrierLevels)}) != ObservationDates length ({m_obs})"
                )
            _check_finite_sequence(self.CouponBarrierLevels, "CouponBarrierLevels")
        if len(self.UnderlyingIdentifiers) == 0:
            raise ValueError("UnderlyingIdentifiers must not be empty")
        if len(set(self.UnderlyingIdentifiers)) != len(self.UnderlyingIdentifiers):
            raise ValueError("UnderlyingIdentifiers must not contain duplicates")
        if len(self.UnderlyingIdentifiers) != len(self.UnderlyingReferenceSpots):
            raise ValueError(
                f"UnderlyingIdentifiers length ({len(self.UnderlyingIdentifiers)}) != "
                f"UnderlyingReferenceSpots length ({len(self.UnderlyingReferenceSpots)})"
            )
        if any((not math.isfinite(s)) or s <= 0.0 for s in self.UnderlyingReferenceSpots):
            raise ValueError("UnderlyingReferenceSpots must all be strictly positive and finite")
        pair_indices = {"RatioAB", "Ratio", "RatioBA", "DiffAB", "DiffBA", "Outperformance", "Spread"}
        for idx_val in (
            self.CallabilityMonitoringIndex,
            self.CouponBarrierMonitoringIndex,
            self.OptionMonitoringIndex,
        ):
            if idx_val is not None and idx_val.value in pair_indices and len(self.UnderlyingIdentifiers) != 2:
                raise ValueError(
                    f"MonitoringIndex {idx_val.value!r} requires exactly 2 underlyings, got {len(self.UnderlyingIdentifiers)}"
                )
        if any((not math.isfinite(lvl)) or lvl <= 0.0 for lvl in self.AutoCallTriggerLevels):
            raise ValueError("AutoCallTriggerLevels must all be strictly positive and finite")
        if not math.isfinite(self.Strike) or self.Strike < 0.0:
            raise ValueError(f"Strike must be finite and non-negative, got {self.Strike!r}")
        for spread_name, spread_val in (
            ("AutoCallTriggerAbsoluteDigitalSpread", self.AutoCallTriggerAbsoluteDigitalSpread),
            ("CouponBarrierAbsoluteDigitalSpread", self.CouponBarrierAbsoluteDigitalSpread),
            ("OptionDownAndInBarrierAbsoluteDigitalSpread", self.OptionDownAndInBarrierAbsoluteDigitalSpread),
            ("OptionDownAndInBarrierLevel", self.OptionDownAndInBarrierLevel),
            ("FundingSpread", self.FundingSpread),
        ):
            if spread_val is not None and (isinstance(spread_val, bool) or not math.isfinite(spread_val)):
                raise ValueError(f"{spread_name} must be finite, got {spread_val!r}")
        if self.LowerStrike is not None:
            if not math.isfinite(self.LowerStrike) or self.LowerStrike < 0.0:
                raise ValueError(f"LowerStrike must be finite and non-negative, got {self.LowerStrike!r}")
            if self.LowerStrike > self.Strike:
                raise ValueError(f"LowerStrike ({self.LowerStrike}) must be <= Strike ({self.Strike})")
        if not math.isfinite(self.OptionGearing) or self.OptionGearing < 0.0:
            raise ValueError(f"OptionGearing must be finite and non-negative, got {self.OptionGearing!r}")
        for name_opt, val_opt in (
            ("AutoCallCoupon", self.AutoCallCoupon),
            ("OneStarStrike", self.OneStarStrike),
            ("OneStarCoupon", self.OneStarCoupon),
            ("ParticipationCallStrike", self.ParticipationCallStrike),
            ("ParticipationCallRate", self.ParticipationCallRate),
        ):
            if val_opt is not None and (not math.isfinite(val_opt) or val_opt < 0.0):
                raise ValueError(f"{name_opt} must be finite and non-negative, got {val_opt!r}")
        for name_cpn, val_cpn in (
            ("FixedCouponAmount", self.FixedCouponAmount),
            ("FixedRate", self.FixedRate),
        ):
            if isinstance(val_cpn, list):
                if len(val_cpn) != m_obs:
                    raise ValueError(f"{name_cpn} length ({len(val_cpn)}) != ObservationDates length ({m_obs})")
                if any((not math.isfinite(x)) or x < 0.0 for x in val_cpn):
                    raise ValueError(f"{name_cpn} elements must be finite and non-negative")
            elif val_cpn is not None and (not math.isfinite(val_cpn) or val_cpn < 0.0):
                raise ValueError(f"{name_cpn} must be finite and non-negative, got {val_cpn!r}")
        return self

    def get_coupon_rates(self, *, apply_dcc: bool = False) -> List[float]:
        """Returns the per-period coupon vector of length M (applying DCC when apply_dcc=True and FixedRate is used)."""
        m = len(self.ObservationDates)
        source = self.FixedCouponAmount if self.FixedCouponAmount is not None else self.FixedRate
        if source is None:
            return [0.0] * m
        if isinstance(source, list):
            if len(source) != m:
                raise ValueError(f"Coupon vector length ({len(source)}) != number of periods ({m})")
            rates = [float(x) for x in source]
        else:
            rates = [float(source)] * m
        if apply_dcc and self.FixedRate is not None:
            from deepvol.market.eq_curves import year_fraction

            dcc = self.DayCountConvention.value
            prev_dt = self.StartDate
            scaled: List[float] = []
            for r_m, pay_dt in zip(rates, self.PaymentDates):
                scaled.append(r_m * year_fraction(prev_dt, pay_dt, dcc))
                prev_dt = pay_dt
            return scaled
        return rates


_DESCRIPTION_TO_SHORT_CODE: Dict[str, str] = {
    "FundingLeg": "F",
    "Coupon": "C",
    "Notional": "N",
    "Option": "O",
    "OneStarCoupon": "S",
    "IsLongBasket": "L",
}


class CashFlowRow(BaseModel):
    """Single cash-flow table entry in `EQAutocallableTradeResponse`."""

    model_config = _STRICT_CONFIG

    Description: Union[
        Literal["FundingLeg", "Coupon", "Notional", "Option", "OneStarCoupon", "IsLongBasket"],
        str,
    ]
    Date: datetime.datetime
    Amount: float
    StandardError: float = 0.0
    Currency: str = "RUB"
    Leg: int = 0
    StatusText: Union[Literal["Settled", "In-Transit", "InTransit", "Estimate"], str] = "Estimate"
    ShortCode: Optional[
        Union[Literal["F", "C", "N", "O", "S", "B", "L"], str]
    ] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        if "PaymentDate" in out:
            pay_dt = out.pop("PaymentDate")
            if "Date" not in out:
                out["Date"] = pay_dt
        if "ExpectedValue" in out:
            exp_val = out.pop("ExpectedValue")
            if "Amount" not in out and exp_val is not None:
                out["Amount"] = exp_val
        if "PaymentValue" in out:
            pay_val = out.pop("PaymentValue")
            if "Amount" not in out and pay_val is not None:
                out["Amount"] = pay_val
        if out.get("ShortCode") is None and "Description" in out:
            sc = _DESCRIPTION_TO_SHORT_CODE.get(str(out["Description"]))
            if sc is not None:
                out["ShortCode"] = sc
        return out

    @field_validator("Date", mode="before")
    @classmethod
    def _parse_date(cls, v: Any) -> datetime.datetime:
        return _normalize_datetime(v)

    @model_validator(mode="after")
    def _validate_finite_amounts(self) -> CashFlowRow:
        if isinstance(self.Amount, bool) or not math.isfinite(self.Amount):
            raise ValueError(f"CashFlowRow.Amount must be finite, got {self.Amount!r}")
        if isinstance(self.StandardError, bool) or not math.isfinite(self.StandardError):
            raise ValueError(f"CashFlowRow.StandardError must be finite, got {self.StandardError!r}")
        return self

    @property
    def PaymentValue(self) -> float:
        """Alias for `Amount`."""
        return float(self.Amount)


_DEFAULT_AUTOCALL_QUERIES: Tuple[AutocallQueryMetric, ...] = (
    AutocallQueryMetric.ProductValue,
    AutocallQueryMetric.CashFlows,
)


class EQAutocallableTradeRequest(BaseModel):
    """Canonical top-level pricing request with legacy MC/engine parameter stripping."""

    model_config = _STRICT_CONFIG

    Product: AutoCallableProductSpec
    Model: DupireModelSpec
    Queries: List[AutocallQueryMetric] = Field(
        default_factory=lambda: list(_DEFAULT_AUTOCALL_QUERIES)
    )
    TradeInfo: Optional[str] = None
    stripped_legacy_fields: List[str] = Field(default_factory=list)
    legacy_solver_config: Optional[Dict[str, Any]] = Field(default=None, exclude=True)

    @field_validator("Queries", mode="before")
    @classmethod
    def _coalesce_queries(cls, v: Any) -> Any:
        if v is None:
            return list(_DEFAULT_AUTOCALL_QUERIES)
        return v

    @model_validator(mode="before")
    @classmethod
    def _strip_tier0_root_fields(cls, data: Any) -> Any:
        if not isinstance(data, (dict, AbcMapping)):
            return data
        out = dict(data)
        if "Queries" in out and out["Queries"] is None:
            out["Queries"] = list(_DEFAULT_AUTOCALL_QUERIES)
        stripped = _coalesce_stripped_legacy_fields(out.get("stripped_legacy_fields"))
        existing_solver_cfg = out.get("legacy_solver_config")
        solver_cfg: Dict[str, Any] = (
            dict(existing_solver_cfg) if isinstance(existing_solver_cfg, (dict, AbcMapping)) else {}
        )

        for field_name in (
            "CalculationMethod",
            "NumberOfPaths",
            "BatchSizeForMCPath",
            "RandomNumberGenerator",
            "TimeSteps",
            "StaticDataSet",
            "GreekMetricsSettings",
        ):
            if field_name in out:
                val = out.pop(field_name)
                if isinstance(stripped, list):
                    _append_unique(stripped, [field_name])
                if field_name in (
                    "CalculationMethod",
                    "NumberOfPaths",
                    "BatchSizeForMCPath",
                    "RandomNumberGenerator",
                    "TimeSteps",
                    "GreekMetricsSettings",
                ):
                    solver_cfg[field_name] = val
                if field_name == "GreekMetricsSettings" and isinstance(val, (dict, AbcMapping)):
                    for gk_key, gk_val in val.items():
                        if gk_key not in _GREEK_SETTINGS_TIER0_FIELDS:
                            raise ValueError(
                                f"Unknown field '{gk_key}' inside GreekMetricsSettings"
                            )
                        if isinstance(stripped, list):
                            _append_unique(stripped, [gk_key])
                        if gk_key == "BumpParametersOverrides" and isinstance(gk_val, (dict, AbcMapping)):
                            for metric_name, bump_cfg in gk_val.items():
                                if isinstance(bump_cfg, (dict, AbcMapping)):
                                    for b_key in bump_cfg:
                                        if b_key not in _BUMP_OVERRIDE_TIER0_FIELDS:
                                            raise ValueError(
                                                f"Unknown field '{b_key}' in BumpParametersOverrides[{metric_name!r}]"
                                            )
                                        if isinstance(stripped, list):
                                            _append_unique(stripped, [b_key])

        out["stripped_legacy_fields"] = stripped
        if solver_cfg:
            out["legacy_solver_config"] = solver_cfg
        return out

    @model_validator(mode="after")
    def _collect_nested_stripped_fields(self) -> EQAutocallableTradeRequest:
        if len(self.Queries) == 0:
            raise ValueError("Queries must not be empty")
        mds = self.Model.MarketDataSet
        uids = self.Product.UnderlyingIdentifiers
        for uid in uids:
            if uid not in mds.Spots:
                raise ValueError(f"Missing spot for underlying '{uid}' in MarketDataSet.Spots")
            if uid not in mds.VolatilitySurfaces:
                raise ValueError(f"Missing volatility surface for underlying '{uid}' in MarketDataSet.VolatilitySurfaces")
        if len(uids) >= 2:
            if len(mds.Correlations) == 0:
                raise ValueError("MarketDataSet.Correlations must not be empty for multi-asset baskets")
            for i in range(len(uids)):
                u_i = uids[i]
                for j in range(i + 1, len(uids)):
                    u_j = uids[j]
                    if f"{u_i}&{u_j}" not in mds.Correlations and f"{u_j}&{u_i}" not in mds.Correlations:
                        raise ValueError(
                            f"Missing correlation pair for ({u_i!r}, {u_j!r}) in MarketDataSet.Correlations"
                        )
        _append_unique(self.stripped_legacy_fields, self.Model.stripped_legacy_fields)
        return self


class EQAutocallableTradeResponse(BaseModel):
    """Canonical top-level response containing `ProductValue`, `CashFlows`, Greeks, and SR 26-2 telemetry."""

    model_config = _STRICT_CONFIG

    ProductValue: float
    ReportingCurrency: str = "RUB"
    StandardError: float = 0.0
    CashToSettleToday: float = 0.0
    CashFlows: List[CashFlowRow] = Field(default_factory=list)

    # Optional Greek sensitivity outputs
    Delta: Optional[Dict[str, float]] = None
    Gamma: Optional[Union[Dict[str, Dict[str, float]], Dict[str, float], List[List[float]]]] = None
    MultivariateAggregatedCashGamma: Optional[Union[float, List[List[float]]]] = None
    Vega: Optional[Dict[str, List[List[float]]]] = None
    Rho: Optional[Dict[str, List[float]]] = None
    RepoRho: Optional[Dict[str, List[float]]] = None
    rho_unit_convention: Literal["quote", "decimal"] = "quote"
    DividendsSensitivities: Optional[Dict[str, List[float]]] = None
    Theta: Optional[float] = None
    Cega: Optional[Union[Dict[str, Dict[str, float]], Dict[str, float], List[List[float]]]] = None

    # Metadata & SR 11-7 / SR 26-2 governance telemetry
    TradeInfo: Optional[str] = None
    engine_used: str = "ML_SURROGATE"
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    latency_ms: float = 0.0

    @model_validator(mode="after")
    def _validate_finite_outputs(self) -> EQAutocallableTradeResponse:
        for attr in ("ProductValue", "StandardError", "CashToSettleToday", "latency_ms", "Theta"):
            val = getattr(self, attr)
            if val is not None and (isinstance(val, bool) or not math.isfinite(float(val))):
                raise ValueError(f"EQAutocallableTradeResponse.{attr} must be finite, got {val!r}")
        for attr in (
            "Delta",
            "Gamma",
            "MultivariateAggregatedCashGamma",
            "Vega",
            "Rho",
            "RepoRho",
            "DividendsSensitivities",
            "Cega",
        ):
            _check_finite_nested(getattr(self, attr), attr)
        return self


def to_decimal_merged_rho(
    response: Any,
    quote_scales: Optional[Mapping[str, float]] = None,
    *,
    repo_quote_scales: Optional[Mapping[str, float]] = None,
    merge_repo: bool = True,
    request: Any = None,
) -> Dict[str, List[float]]:
    """Converts quote-unit Rho and optional RepoRho sensitivities to decimal dPV/dr_dec."""
    out: Dict[str, List[float]] = {}
    is_map = isinstance(response, Mapping)
    conv = response.get("rho_unit_convention", "quote") if is_map else getattr(response, "rho_unit_convention", "quote")
    rho_map = response.get("Rho") if is_map else getattr(response, "Rho", None)
    if request is not None:
        mds_obj = getattr(getattr(request, "Model", None), "MarketDataSet", None)
        if quote_scales is None and mds_obj is not None:
            bundles = getattr(mds_obj, "RatesCurvesBundles", None) or {}
            quote_scales = {
                str(cid): (
                    0.01
                    if "Percentage" in str(getattr(c_spec, "RateQuoteExpression", "Percentage"))
                    else 1.0
                )
                for cid, c_spec in bundles.items()
            }
        if repo_quote_scales is None and mds_obj is not None:
            repos = getattr(mds_obj, "RepoSpreadCurves", None) or {}
            repo_quote_scales = {
                str(tk): (
                    0.01
                    if "Percentage" in str(getattr(r_spec, "RateQuoteExpression", "Percentage"))
                    else 1.0
                )
                for tk, r_spec in repos.items()
            }
    scales = quote_scales or {}
    if rho_map:
        for cid, vec in rho_map.items():
            raw_scale = scales.get(cid)
            scale_val = 0.01 if raw_scale is None else float(raw_scale)
            if not math.isfinite(scale_val):
                raise ValueError(f"Non-finite quote scale for '{cid}': {raw_scale!r}")
            scale = max(scale_val, 1e-12)
            inv_s = 1.0 if conv == "decimal" else (1.0 / scale)
            converted: List[float] = []
            for v in vec:
                v_f = float(v)
                if not math.isfinite(v_f):
                    raise ValueError(f"Non-finite Rho sensitivity for '{cid}': {v!r}")
                converted.append(v_f * inv_s)
            out[cid] = converted
    repo_map = response.get("RepoRho") if is_map else getattr(response, "RepoRho", None)
    if merge_repo and repo_map:
        r_scales = repo_quote_scales or {}
        for asset_id, vec in repo_map.items():
            if asset_id not in out or not out[asset_id]:
                raw_r_scale = r_scales.get(asset_id)
                r_scale_val = 0.01 if raw_r_scale is None else float(raw_r_scale)
                if not math.isfinite(r_scale_val):
                    raise ValueError(f"Non-finite repo quote scale for '{asset_id}': {raw_r_scale!r}")
                r_scale = max(r_scale_val, 1e-12)
                inv_rs = 1.0 if conv == "decimal" else (1.0 / r_scale)
                converted_r: List[float] = []
                for v in vec:
                    v_f = float(v)
                    if not math.isfinite(v_f):
                        raise ValueError(f"Non-finite RepoRho sensitivity for '{asset_id}': {v!r}")
                    converted_r.append(v_f * inv_rs)
                out[asset_id] = converted_r
    return out


# ── Murex `ConstantMessageMap` DAG Tokenizer & Parser ─────────────────────────

_FAST_TOK_RE = re.compile(
    r"datetime\.(?:datetime|date)\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)"
    r"|<skipped>"
    r"|'[^']*'"
    r'|"[^"]*"'
    r"|[\{\}\[\]:]"
    r"|[^\s,\{\}\[\]:'\"]+"
    r"|['\"]"
)

_TOKEN_RE = re.compile(
    r"""
    (?:\s+|,+)
  | (?P<SKIPPED><skipped>)
  | (?P<DATETIME>datetime\.(?:datetime|date)\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\))
  | (?P<LBRACE>\{)
  | (?P<RBRACE>\})
  | (?P<LBRACKET>\[)
  | (?P<RBRACKET>\])
  | (?P<COLON>:)
  | (?P<SQUOTE>'[^']*')
  | (?P<DQUOTE>"[^"]*")
  | (?P<REF>[@&!][0-9a-fA-F]{40})
  | (?P<WORD>[^\s,\{\}\[\]:'"]+)
  | (?P<ERR>.)
    """,
    re.VERBOSE,
)

_TOK_SKIPPED = 1
_TOK_DATETIME = 2
_TOK_LBRACE = 3
_TOK_RBRACE = 4
_TOK_LBRACKET = 5
_TOK_RBRACKET = 6
_TOK_COLON = 7
_TOK_SQUOTE = 8
_TOK_DQUOTE = 9
_TOK_REF = 10
_TOK_WORD = 11
_TOK_ERR = 12

_KIND_TO_NAME: Tuple[str, ...] = (
    "",
    "SKIPPED",
    "DATETIME",
    "LBRACE",
    "RBRACE",
    "LBRACKET",
    "RBRACKET",
    "COLON",
    "SQUOTE",
    "DQUOTE",
    "REF",
    "WORD",
    "ERR",
)

_FAST_UTC_DT_RE = re.compile(
    r"^datetime\.datetime\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+))?,\s*tzinfo=datetime\.timezone\.utc\)$"
)
_SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_REF_STR_RE = re.compile(r"^[@&!]([0-9a-fA-F]{40})$")
_DT_ARGS_RE = re.compile(r"^datetime\.(datetime|date)\((.*)\)$")
_TD_ARGS_RE = re.compile(r"(?:datetime\.)?timedelta\((.*)\)$")
_EXTERNAL_OR_TIER0_ID_KEYS: frozenset[str] = frozenset(
    {"StaticDataSet", "BaseCurve", "ForwardObject", "FinancingCurve"}
)


def _split_top_level_args(inner: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            seg = inner[start:idx].strip()
            if seg:
                parts.append(seg)
            start = idx + 1
    tail = inner[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_tzinfo_expr(expr: str) -> datetime.timezone:
    s = expr.strip()
    if "utc" in s.lower():
        return datetime.timezone.utc
    if s.startswith(("datetime.timezone(", "timezone(")) and s.endswith(")"):
        inner = s[s.index("(") + 1 : -1].strip()
        td_match = _TD_ARGS_RE.match(inner)
        if td_match is not None:
            td_parts = _split_top_level_args(td_match.group(1))
            pos_vals: List[float] = []
            kw_vals: Dict[str, float] = {}
            for tp in td_parts:
                if "=" in tp:
                    tk, tv = [x.strip() for x in tp.split("=", 1)]
                    kw_vals[tk] = float(tv)
                else:
                    pos_vals.append(float(tp))
            td = datetime.timedelta(*pos_vals, **kw_vals)
            return datetime.timezone(td)
    raise ValueError(f"Unsupported tzinfo literal in datetime expression: {expr!r}")


def _parse_datetime_literal(text: str) -> datetime.datetime:
    """Safely parses a `datetime.datetime(...)` or `datetime.date(...)` literal without `eval()`."""
    if text.startswith("datetime.datetime(") and text.endswith(", tzinfo=datetime.timezone.utc)"):
        nums = text[18:-31].split(",")
        n_nums = len(nums)
        if 3 <= n_nums <= 6:
            try:
                vals = [int(x) for x in nums]
                return datetime.datetime(
                    vals[0],
                    vals[1],
                    vals[2],
                    vals[3] if n_nums > 3 else 0,
                    vals[4] if n_nums > 4 else 0,
                    vals[5] if n_nums > 5 else 0,
                    tzinfo=datetime.timezone.utc,
                )
            except ValueError:
                pass
    fast_m = _FAST_UTC_DT_RE.match(text)
    if fast_m is not None:
        g1, g2, g3, g4, g5, g6 = fast_m.groups()
        return datetime.datetime(
            int(g1),
            int(g2),
            int(g3),
            int(g4),
            int(g5),
            int(g6) if g6 is not None else 0,
            tzinfo=datetime.timezone.utc,
        )
    m = _DT_ARGS_RE.match(text.strip())
    if m is None:
        raise ValueError(f"Malformed datetime literal: {text!r}")
    kind = m.group(1)
    inner = m.group(2)
    parts = _split_top_level_args(inner)
    int_args: List[int] = []
    kw_int: Dict[str, int] = {}
    tz = datetime.timezone.utc
    for p in parts:
        if "=" in p:
            k, v = [x.strip() for x in p.split("=", 1)]
            if k == "tzinfo":
                tz = _parse_tzinfo_expr(v)
            elif k in ("year", "month", "day", "hour", "minute", "second", "microsecond"):
                kw_int[k] = int(v)
        else:
            int_args.append(int(p))
    if kind == "date":
        if not kw_int and len(int_args) < 3:
            raise ValueError(f"Malformed date literal: {text!r}")
        return datetime.datetime(
            kw_int.get("year", int_args[0] if len(int_args) > 0 else 1),
            kw_int.get("month", int_args[1] if len(int_args) > 1 else 1),
            kw_int.get("day", int_args[2] if len(int_args) > 2 else 1),
            0,
            0,
            tzinfo=tz,
        ).astimezone(datetime.timezone.utc)
    while len(int_args) < 7:
        int_args.append(0)
    dt = datetime.datetime(
        kw_int.get("year", int_args[0]),
        kw_int.get("month", int_args[1]),
        kw_int.get("day", int_args[2]),
        kw_int.get("hour", int_args[3]),
        kw_int.get("minute", int_args[4]),
        kw_int.get("second", int_args[5]),
        kw_int.get("microsecond", int_args[6]),
        tzinfo=tz,
    )
    return dt.astimezone(datetime.timezone.utc)


class _DAGTokenStream:
    """Lexer stream over Murex `ConstantMessageMap` text dumps."""

    def __init__(self, text: str) -> None:
        raw_tokens: List[Tuple[int, str]] = [
            (m.lastindex, m.group())
            for m in _TOKEN_RE.finditer(text)
            if m.lastindex is not None
        ]
        for k_id, val in raw_tokens:
            if k_id == _TOK_ERR:
                raise ValueError(f"Unexpected character in DAG input: {val!r}")
        self._raw_tokens = raw_tokens
        self.tokens: List[Tuple[str, str]] = [(_KIND_TO_NAME[k_id], val) for k_id, val in raw_tokens]
        self.idx = 0

    def peek(self, offset: int = 0) -> Optional[Tuple[str, str]]:
        pos = self.idx + offset
        if 0 <= pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def next(self) -> Tuple[str, str]:
        if self.idx >= len(self.tokens):
            raise ValueError("Unexpected end of token stream while parsing Murex DAG")
        tok = self.tokens[self.idx]
        self.idx += 1
        return tok

    def expect(self, expected_kind: str) -> str:
        kind, val = self.next()
        if kind != expected_kind:
            raise ValueError(f"Expected token {expected_kind}, got ({kind}, {val!r})")
        return val


def _tokenize_dag_fast(text: str) -> List[Tuple[int, str]]:
    tokens: List[Tuple[int, str]] = [
        (m.lastindex, m.group())
        for m in _TOKEN_RE.finditer(text)
        if m.lastindex is not None
    ]
    for k_id, val in tokens:
        if k_id == _TOK_ERR:
            raise ValueError(f"Unexpected character in DAG input: {val!r}")
    return tokens


_NUM_START = frozenset("0123456789-+")


def _parse_scalar_token(val: str) -> Any:
    ch = val[0]
    if ch == "'" or ch == '"':
        if len(val) < 2 or val[-1] != ch:
            raise ValueError(f"Unclosed string literal {val!r}")
        return val[1:-1]
    if ch in _NUM_START:
        if len(val) == 40 and _SHA1_RE.fullmatch(val) is not None:
            return val
        try:
            if "." not in val and "e" not in val and "E" not in val:
                return int(val)
            return float(val)
        except ValueError:
            return val
    if val == "<skipped>":
        return None
    if ch == "d" and (val.startswith("datetime.datetime(") or val.startswith("datetime.date(")):
        return _parse_datetime_literal(val)
    if (ch == "@" or ch == "&" or ch == "!") and len(val) == 41 and _SHA1_RE.fullmatch(val[1:]) is not None:
        return val
    if val in ("}", "]", ":"):
        raise ValueError(f"Unexpected delimiter {val!r} while parsing value")
    if val == "True" or val == "true":
        return True
    if val == "False" or val == "false":
        return False
    if val == "None" or val == "null":
        return None
    if len(val) == 40 and _SHA1_RE.fullmatch(val) is not None:
        return val
    return val


def _parse_str_dag_at(tokens: List[str], idx: int, n_toks: int) -> Tuple[Any, int]:
    """Iterative Murex DAG block parser avoiding per-scalar Python call overhead."""
    if idx >= n_toks:
        raise ValueError("Unexpected end of token stream while parsing Murex DAG")

    num_start = _NUM_START
    sha1_full = _SHA1_RE.fullmatch
    dt_cls = datetime.datetime
    utc_tz = datetime.timezone.utc

    stack: List[Any] = []
    # Stack entries are either (dict_obj, pending_key_or_None) or list_obj
    root_assigned = False
    root_val: Any = None

    while idx < n_toks:
        val = tokens[idx]
        idx += 1
        ch = val[0]

        if ch == "}":
            if not stack or type(stack[-1]) is not tuple:
                raise ValueError("Unexpected '}' in Murex DAG")
            completed_dict, pending_k = stack.pop()
            if pending_k is not None:
                raise ValueError(f"Missing value for key {pending_k!r} before '}}'")
            parsed = completed_dict
        elif ch == "]":
            if not stack or type(stack[-1]) is not list:
                raise ValueError("Unexpected ']' in Murex DAG")
            parsed = stack.pop()
        elif stack and type(stack[-1]) is tuple and stack[-1][1] is None:
            if ch in "{[:" or ch == "}":
                raise ValueError(f"Invalid map key token {val!r} in Murex DAG")
            if ch == "'" or ch == '"':
                if len(val) < 2 or val[-1] != ch:
                    raise ValueError(f"Unclosed string key token {val!r}")
                key_str = val[1:-1]
            else:
                key_str = val
            if idx >= n_toks or tokens[idx] != ":":
                raise ValueError("Expected token COLON after map key")
            idx += 1
            stack[-1] = (stack[-1][0], key_str)
            continue
        elif ch == "{":
            stack.append(({}, None))
            continue
        elif ch == "[":
            stack.append([])
            continue
        else:
            if ch in num_start:
                if len(val) == 40 and sha1_full(val) is not None:
                    parsed = val
                elif "." in val or "e" in val or "E" in val:
                    try:
                        parsed = float(val)
                    except ValueError:
                        parsed = val
                else:
                    try:
                        parsed = int(val)
                    except ValueError:
                        parsed = val
            elif ch == "'" or ch == '"':
                if len(val) < 2 or val[-1] != ch:
                    raise ValueError(f"Unclosed string literal {val!r}")
                parsed = val[1:-1]
            elif ch == "d" and (
                val.startswith("datetime.datetime(") or val.startswith("datetime.date(")
            ):
                if val.startswith("datetime.datetime(") and val.endswith(", tzinfo=datetime.timezone.utc)"):
                    nums = val[18:-31].split(",")
                    n_nums = len(nums)
                    try:
                        if n_nums == 5:
                            parsed = dt_cls(
                                int(nums[0]), int(nums[1]), int(nums[2]),
                                int(nums[3]), int(nums[4]), tzinfo=utc_tz,
                            )
                        elif n_nums == 6:
                            parsed = dt_cls(
                                int(nums[0]), int(nums[1]), int(nums[2]),
                                int(nums[3]), int(nums[4]), int(nums[5]), tzinfo=utc_tz,
                            )
                        else:
                            parsed = _parse_datetime_literal(val)
                    except ValueError:
                        parsed = _parse_datetime_literal(val)
                else:
                    parsed = _parse_datetime_literal(val)
            elif val == "<skipped>":
                parsed = None
            elif (ch == "@" or ch == "&" or ch == "!") and len(val) == 41 and sha1_full(val[1:]) is not None:
                parsed = val
            elif val == ":":
                raise ValueError("Unexpected ':' while parsing value")
            elif val == "True" or val == "true":
                parsed = True
            elif val == "False" or val == "false":
                parsed = False
            elif val == "None" or val == "null":
                parsed = None
            elif len(val) == 40 and sha1_full(val) is not None:
                parsed = val
            else:
                parsed = val

        if not stack:
            root_val = parsed
            root_assigned = True
            break
        top = stack[-1]
        if type(top) is list:
            top.append(parsed)
        else:
            d_obj, p_key = top
            d_obj[p_key] = parsed
            stack[-1] = (d_obj, None)

    if not root_assigned:
        raise ValueError("Unclosed '{' or '[' in Murex DAG")
    return root_val, idx


def _parse_dag_at(tokens: List[Tuple[int, str]], idx: int, n_toks: int) -> Tuple[Any, int]:
    if idx >= n_toks:
        raise ValueError("Unexpected end of token stream while parsing Murex DAG")
    kind, val = tokens[idx]
    idx += 1
    if kind == _TOK_LBRACE:
        obj: Dict[str, Any] = {}
        while idx < n_toks:
            k_kind, k_val = tokens[idx]
            if k_kind == _TOK_RBRACE:
                return obj, idx + 1
            if k_kind == _TOK_SQUOTE or k_kind == _TOK_DQUOTE:
                key_str = k_val[1:-1]
            elif k_kind == _TOK_WORD:
                key_str = k_val
            else:
                raise ValueError(f"Invalid map key token ({_KIND_TO_NAME[k_kind]}, {k_val!r})")
            idx += 1
            if idx >= n_toks or tokens[idx][0] != _TOK_COLON:
                raise ValueError("Expected token COLON after map key")
            idx += 1
            parsed_val, idx = _parse_dag_at(tokens, idx, n_toks)
            obj[key_str] = parsed_val
        raise ValueError("Unclosed '{' in Murex DAG")
    if kind == _TOK_LBRACKET:
        arr: List[Any] = []
        while idx < n_toks:
            if tokens[idx][0] == _TOK_RBRACKET:
                return arr, idx + 1
            item_val, idx = _parse_dag_at(tokens, idx, n_toks)
            arr.append(item_val)
        raise ValueError("Unclosed '[' in Murex DAG")
    if kind == _TOK_SKIPPED:
        return None, idx
    if kind == _TOK_DATETIME:
        return _parse_datetime_literal(val), idx
    if kind == _TOK_SQUOTE or kind == _TOK_DQUOTE:
        return val[1:-1], idx
    if kind == _TOK_REF:
        return val, idx
    if kind == _TOK_WORD:
        if val == "True" or val == "true":
            return True, idx
        if val == "False" or val == "false":
            return False, idx
        if val == "None" or val == "null":
            return None, idx
        if len(val) == 40 and _SHA1_RE.fullmatch(val) is not None:
            return val, idx
        try:
            if "." not in val and "e" not in val and "E" not in val:
                return int(val), idx
            return float(val), idx
        except ValueError:
            return val, idx
    raise ValueError(f"Unexpected token ({_KIND_TO_NAME[kind]}, {val!r}) while parsing value")


def _parse_dag_value(stream: _DAGTokenStream) -> Any:
    val, new_idx = _parse_dag_at(stream._raw_tokens, stream.idx, len(stream._raw_tokens))
    stream.idx = new_idx
    return val


def _extract_reference_hash(val: Any, nodes: Mapping[str, Any]) -> Optional[str]:
    if type(val) is not str and not isinstance(val, str):
        return None
    s = val.strip()
    if len(s) == 41 and s[0] in ("@", "&", "!") and _SHA1_RE.fullmatch(s[1:]) is not None:
        return s[1:].lower()
    if len(s) == 40 and _SHA1_RE.fullmatch(s) is not None:
        s_low = s.lower()
        if s_low in nodes:
            return s_low
    return None


def _resolve_dag_node(
    obj: Any,
    nodes: Mapping[str, Dict[str, Any]],
    visiting: set[str],
    memo: Optional[Dict[str, Any]] = None,
) -> Any:
    if memo is None:
        memo = {}
    obj_type = type(obj)
    if obj_type is float or obj_type is int or obj_type is bool or obj is None:
        return obj
    if obj_type is dict:
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in _EXTERNAL_OR_TIER0_ID_KEYS:
                ext_ref = _extract_reference_hash(v, nodes)
                if ext_ref is not None and ext_ref not in nodes:
                    out[k] = v
                    continue
            out[k] = _resolve_dag_node(v, nodes, visiting, memo)
        return out
    if obj_type is list:
        return [
            item
            if (type(item) is float or type(item) is int or type(item) is bool or item is None)
            else _resolve_dag_node(item, nodes, visiting, memo)
            for item in obj
        ]
    if obj_type is str:
        ref_hash = _extract_reference_hash(obj, nodes)
        if ref_hash is not None:
            if ref_hash in visiting:
                raise ValueError(f"Circular dependency cycle detected in Murex DAG at {ref_hash}")
            if ref_hash in memo:
                return memo[ref_hash]
            if ref_hash not in nodes:
                raise KeyError(f"Dangling Murex DAG reference to missing hash {ref_hash!r}")
            visiting.add(ref_hash)
            resolved = _resolve_dag_node(nodes[ref_hash], nodes, visiting, memo)
            visiting.remove(ref_hash)
            memo[ref_hash] = resolved
            return resolved
        return obj
    if isinstance(obj, (dict, AbcMapping)):
        out_m: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in _EXTERNAL_OR_TIER0_ID_KEYS:
                ext_ref = _extract_reference_hash(v, nodes)
                if ext_ref is not None and ext_ref not in nodes:
                    out_m[k] = v
                    continue
            out_m[k] = _resolve_dag_node(v, nodes, visiting, memo)
        return out_m
    return obj


def _enrich_expected_result(
    result_dict: Dict[str, Any],
    req: EQAutocallableTradeRequest,
) -> Dict[str, Any]:
    """Ensures `Option` and `TradeInfo` convenience keys are populated in `expected_result_dict`."""
    enriched = dict(result_dict)
    if "Option" not in enriched and isinstance(enriched.get("CashFlows"), list):
        for cf in enriched["CashFlows"]:
            if isinstance(cf, (dict, AbcMapping)) and cf.get("Description") == "Option":
                amt = cf.get("Amount")
                if amt is None:
                    amt = cf.get("ExpectedValue")
                if amt is None:
                    amt = cf.get("PaymentValue")
                if amt is not None:
                    enriched["Option"] = float(amt)
                break
    if "TradeInfo" not in enriched and req.TradeInfo is not None:
        enriched["TradeInfo"] = req.TradeInfo
    return enriched


def parse_murex_autocall_dag(
    raw_text_or_dict: Union[str, Path, Mapping[str, Any]],
    *,
    allow_file_path: bool = False,
) -> Tuple[EQAutocallableTradeRequest, Dict[str, Any]]:
    """Parses a Murex `ConstantMessageMap` DAG text dump (or resolved Python dict) into a validated request and reference result dict."""
    if isinstance(raw_text_or_dict, (dict, AbcMapping)):
        raw_map = dict(raw_text_or_dict)
        if "Product" in raw_map and "Model" in raw_map:
            raw_res = raw_map.pop("Result", None)
            if raw_res is not None and not isinstance(raw_res, (dict, AbcMapping)):
                raise ValueError("Murex DAG 'Result' entry must be a mapping or null")
            res_block = dict(raw_res) if isinstance(raw_res, (dict, AbcMapping)) else {}
            req = EQAutocallableTradeRequest.model_validate(raw_map)
            return req, _enrich_expected_result(res_block, req)

        nodes_from_dict: Dict[str, Dict[str, Any]] = {}
        result_from_dict: Dict[str, Any] = {}
        root_hash_from_dict: Optional[str] = None
        for k, v in raw_map.items():
            if k == "Result" and isinstance(v, (dict, AbcMapping)):
                result_from_dict = dict(v)
            elif k in ("Root", "root_id") and isinstance(v, str):
                root_hash_from_dict = v.lstrip("@&!").lower()
            elif isinstance(v, (dict, AbcMapping)):
                h_key = k.lstrip("@&!").lower()
                nodes_from_dict[h_key] = dict(v)
                if "Product" in v and "Model" in v:
                    root_hash_from_dict = h_key

        if root_hash_from_dict is None or root_hash_from_dict not in nodes_from_dict:
            raise ValueError("Could not identify root trade request node in DAG dictionary")
        resolved_root = _resolve_dag_node(nodes_from_dict[root_hash_from_dict], nodes_from_dict, set())
        req = EQAutocallableTradeRequest.model_validate(resolved_root)
        return req, _enrich_expected_result(result_from_dict, req)

    if isinstance(raw_text_or_dict, Path):
        p = raw_text_or_dict
        try:
            if not p.is_file():
                repo_root = Path(__file__).resolve().parents[3]
                for candidate in [repo_root / "data" / "autocall" / p.name, repo_root / p.name]:
                    if candidate.is_file():
                        p = candidate
                        break
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Failed to read Murex DAG file {raw_text_or_dict!r}: {exc}") from exc
    elif isinstance(raw_text_or_dict, str):
        text = raw_text_or_dict
        if allow_file_path and "\n" not in raw_text_or_dict and 0 < len(raw_text_or_dict) <= 255:
            try:
                p = Path(raw_text_or_dict)
                if not p.is_file():
                    repo_root = Path(__file__).resolve().parents[3]
                    for candidate in [repo_root / "data" / "autocall" / p.name, repo_root / p.name]:
                        if candidate.is_file():
                            p = candidate
                            break
                if p.is_file():
                    text = p.read_text(encoding="utf-8")
            except OSError:
                text = raw_text_or_dict
    else:
        raise TypeError(
            f"Unsupported input type for parse_murex_autocall_dag: {type(raw_text_or_dict).__name__}"
        )

    if not text.strip():
        raise ValueError("Empty Murex DAG input")

    tokens = _FAST_TOK_RE.findall(text)
    n_toks = len(tokens)
    if n_toks == 0:
        raise ValueError("No valid tokens found in Murex DAG input")
    idx = 0
    nodes: Dict[str, Dict[str, Any]] = {}
    result_dict: Dict[str, Any] = {}
    anonymous_dicts: List[Dict[str, Any]] = []
    root_hash: Optional[str] = None

    while idx < n_toks:
        val = tokens[idx]
        ch = val[0]

        if ch == "{":
            parsed_map, idx = _parse_str_dag_at(tokens, idx, n_toks)
            if isinstance(parsed_map, (dict, AbcMapping)) and (
                "ProductValue" in parsed_map or "CashFlows" in parsed_map
            ):
                result_dict = dict(parsed_map)
            elif isinstance(parsed_map, (dict, AbcMapping)):
                anonymous_dicts.append(dict(parsed_map))
        elif ch in ("@", "&", "!") and len(val) == 41 and _SHA1_RE.fullmatch(val[1:]) is not None:
            idx += 1
            root_hash = val[1:].lower()
        elif val == "Result" and idx + 1 < n_toks and tokens[idx + 1] == ":":
            idx += 2
            res_val, idx = _parse_str_dag_at(tokens, idx, n_toks)
            if isinstance(res_val, (dict, AbcMapping)):
                result_dict = dict(res_val)
        elif len(val) == 40 and _SHA1_RE.fullmatch(val) is not None:
            node_hash = val.lower()
            idx += 1
            if idx < n_toks:
                nv = tokens[idx]
                if nv == "class":
                    idx += 1
                    if idx >= n_toks or tokens[idx] != ":":
                        raise ValueError("Expected COLON after 'class' in Murex DAG header")
                    idx += 1
                    if idx >= n_toks or tokens[idx] in ("{", "[", ":", "}", "]"):
                        raise ValueError("Expected class identifier after 'class:' in Murex DAG header")
                    idx += 1
                elif nv == ":":
                    idx += 1
            node_body, idx = _parse_str_dag_at(tokens, idx, n_toks)
            if not isinstance(node_body, (dict, AbcMapping)):
                raise ValueError(f"Expected map body for DAG node {node_hash}")
            nodes[node_hash] = dict(node_body)
        else:
            raise ValueError(f"Unexpected top-level token in Murex DAG: {val!r}")

    if root_hash is None:
        for h, body in nodes.items():
            if "Product" in body and "Model" in body:
                root_hash = h
                break

    if root_hash is not None:
        if root_hash not in nodes:
            raise ValueError(f"Root hash {root_hash!r} not found in parsed ConstantMessageMap nodes")
        resolved_req_dict = _resolve_dag_node(nodes[root_hash], nodes, {root_hash})
    elif anonymous_dicts:
        root_anon = next(
            (d for d in anonymous_dicts if "Product" in d and "Model" in d),
            anonymous_dicts[0],
        )
        resolved_req_dict = _resolve_dag_node(root_anon, nodes, set())
    else:
        raise ValueError("No root request node found in Murex DAG input")

    req = EQAutocallableTradeRequest.model_validate(resolved_req_dict)
    return req, _enrich_expected_result(result_dict, req)
