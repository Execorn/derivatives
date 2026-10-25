"""schemas_eq_autocall.py — Canonical Pydantic V2 schemas & Murex DAG parser for Multi-Asset Equity Autocallables.

Implements Milestone M0 (Task 0.1) of SPEC-DEEPVOL-EQ-AUTOCALL-001:
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

import datetime
from enum import Enum
import os
from pathlib import Path
import re
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union, cast

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
    RatioBA = "RatioBA"
    RatioBestWorst = "RatioBestWorst"
    RatioBestAverage = "RatioBestAverage"
    RatioAverageWorst = "RatioAverageWorst"
    DiffAB = "DiffAB"
    DiffBA = "DiffBA"
    DiffBestWorst = "DiffBestWorst"
    DiffBestAverage = "DiffBestAverage"
    DiffAverageWorst = "DiffAverageWorst"

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
    RATIO_BA = "RatioBA"
    RATIO_BEST_WORST = "RatioBestWorst"
    RATIO_BEST_AVERAGE = "RatioBestAverage"
    RATIO_AVERAGE_WORST = "RatioAverageWorst"
    DIFF_AB = "DiffAB"
    DIFF_BA = "DiffBA"
    DIFF_BEST_WORST = "DiffBestWorst"
    DIFF_BEST_AVERAGE = "DiffBestAverage"
    DIFF_AVERAGE_WORST = "DiffAverageWorst"


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


class CurveInterpolationMethodEnum(str, Enum):
    """Supported discount/repo/smile curve interpolation methods."""

    CubicSpline = "CubicSpline"
    Linear = "Linear"
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


def _normalize_datetime(val: Any) -> datetime.datetime:
    """Normalizes a date, datetime, or ISO string to a UTC-aware datetime.datetime."""
    if isinstance(val, datetime.datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=datetime.timezone.utc)
        return val
    if isinstance(val, datetime.date):
        return datetime.datetime(val.year, val.month, val.day, 0, 0, tzinfo=datetime.timezone.utc)
    if isinstance(val, str):
        s = val.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    raise TypeError(f"Cannot convert {type(val).__name__} to datetime.datetime")


def _normalize_datetime_list(vals: Any) -> List[datetime.datetime]:
    if not isinstance(vals, Sequence) or isinstance(vals, (str, bytes)):
        raise TypeError("Expected a sequence of dates/datetimes")
    return [_normalize_datetime(v) for v in vals]


def _append_unique(target: List[str], items: Sequence[str]) -> None:
    for item in items:
        if item not in target:
            target.append(item)


# ── Market Data Leaf Specifications ───────────────────────────────────────────


class RatesScheduleSpec(BaseModel):
    """Pillar dates, percentage/decimal zero rates, and optional basis spreads for a DiscountFactorCurve."""

    model_config = ConfigDict(extra="forbid")

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
        if self.Rates and len(self.Rates) != len(self.Dates):
            raise ValueError(
                f"RatesSchedule.Rates length ({len(self.Rates)}) != Dates length ({len(self.Dates)})"
            )
        if self.Spreads is not None and len(self.Spreads) != len(self.Dates):
            raise ValueError(
                f"RatesSchedule.Spreads length ({len(self.Spreads)}) != Dates length ({len(self.Dates)})"
            )
        if not self.Rates and self.Spreads is None:
            raise ValueError("RatesSchedule requires either Rates or Spreads")
        return self


class DiscountFactorCurveSpec(BaseModel):
    """Discount factor / yield curve specification with Tier 0 basis-placeholder stripping."""

    model_config = ConfigDict(extra="forbid")

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
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        stripped: List[str] = list(out.get("stripped_legacy_fields", []))
        for field_name in _CURVE_TIER0_FIELDS:
            if field_name in out:
                out.pop(field_name)
                _append_unique(stripped, [field_name])
        out["stripped_legacy_fields"] = stripped
        return out

    @field_validator("AsOfDate", mode="before")
    @classmethod
    def _parse_as_of(cls, v: Any) -> datetime.datetime:
        return _normalize_datetime(v)


_PERIOD_DAYS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([DWMYdwmy])\s*$")


class RepoScheduleSpec(BaseModel):
    """Pillar periods ('<n>D') and percentage repo spread rates for a RepoSpreadCurve."""

    model_config = ConfigDict(extra="forbid")

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
        _ = self.period_days()
        return self

    def period_days(self) -> List[float]:
        """Parses '<n>D' period strings into day counts in float64."""
        days: List[float] = []
        for p in self.Periods:
            m = _PERIOD_DAYS_RE.match(p)
            if m is None:
                raise ValueError(f"Invalid repo period format '{p}'; expected '<n>D'")
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
        return days


class RepoSpreadCurveSpec(BaseModel):
    """Equity repo spread curve specification (`Act360`, `<n>D` pillar periods)."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    ExDate: datetime.datetime
    PaymentDate: datetime.datetime
    Amount: float

    @field_validator("ExDate", "PaymentDate", mode="before")
    @classmethod
    def _parse_dt(cls, v: Any) -> datetime.datetime:
        return _normalize_datetime(v)


class DividendsSpec(BaseModel):
    """Discrete cash dividend schedule for a single equity underlying."""

    model_config = ConfigDict(extra="forbid")

    TypeName: Literal["Dividends"] = "Dividends"
    Identifier: str
    CashDividendList: List[DividendItemSpec] = Field(default_factory=list)


class PredefinedDividendsSpec(BaseModel):
    """Contract-level predefined dividend schedule (`ExDates`, `Amounts`) from the AUTOCALL PERF block."""

    model_config = ConfigDict(extra="forbid")

    ExDates: List[datetime.datetime]
    Amounts: List[float]

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
        return self


class EquityVolSmileCollectionSpec(BaseModel):
    """Implied volatility surface (`[maturity][strike]` decimal volatility quotes) for an equity underlying."""

    model_config = ConfigDict(extra="forbid")

    Identifier: str
    TypeName: Literal["EquityVolatilitySmileCollection"] = "EquityVolatilitySmileCollection"
    DayCountConvention: DayCountConventionEnum = DayCountConventionEnum.Act365
    SmileInterpolationMethod: CurveInterpolationMethodEnum = CurveInterpolationMethodEnum.CubicSpline
    SmileExtrapolationMethod: ExtrapolationMethodEnum = ExtrapolationMethodEnum.Linear
    ExpirationDates: List[datetime.datetime]
    Strikes: List[float]
    VolatilityQuotes: List[List[float]]

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
        if len(self.VolatilityQuotes) != n_exp:
            raise ValueError(
                f"VolatilityQuotes row count ({len(self.VolatilityQuotes)}) != ExpirationDates length ({n_exp})"
            )
        for row_idx, row in enumerate(self.VolatilityQuotes):
            if len(row) != n_str:
                raise ValueError(
                    f"VolatilityQuotes[{row_idx}] length ({len(row)}) != Strikes length ({n_str})"
                )
        return self


class FXVolatilitySmileCollectionSpec(BaseModel):
    """FX implied volatility surface (`[maturity][delta]` volatility quotes) covering all 12 Murex FX surface fields."""

    model_config = ConfigDict(extra="forbid")

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
        return self


VolatilitySurfaceSpec = Annotated[
    Union[EquityVolSmileCollectionSpec, FXVolatilitySmileCollectionSpec],
    Field(discriminator="TypeName"),
]


class FixingsSpec(BaseModel):
    """Historical fixing dates and observed price/rate values for a seasoned underlying or rate index."""

    model_config = ConfigDict(extra="forbid")

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
        return self


class CorrelationPairSpec(BaseModel):
    """Structured representation of a pairwise correlation entry `<asset_1>&<asset_2>`."""

    model_config = ConfigDict(extra="forbid")

    asset_1: str
    asset_2: str
    correlation: float = Field(..., ge=-1.0, le=1.0)

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
        return cls(asset_1=parts[0], asset_2=parts[1], correlation=float(correlation))


class MarketDataSetSpec(BaseModel):
    """Complete multi-asset market data bundle (`RatesCurvesBundles`, `VolatilitySurfaces`, `Spots`, `Correlations`, etc.)."""

    model_config = ConfigDict(extra="forbid")

    TypeName: Literal["MarketDataSet"] = "MarketDataSet"
    AsOfDate: datetime.datetime
    RatesCurvesBundles: Dict[str, DiscountFactorCurveSpec]
    VolatilitySurfaces: Dict[str, VolatilitySurfaceSpec]
    Spots: Dict[str, float]
    Correlations: Dict[str, float] = Field(default_factory=dict)
    RepoSpreadCurves: Dict[str, RepoSpreadCurveSpec] = Field(default_factory=dict)
    Dividends: Dict[str, DividendsSpec] = Field(default_factory=dict)
    Fixings: Dict[str, FixingsSpec] = Field(default_factory=dict)

    @field_validator("AsOfDate", mode="before")
    @classmethod
    def _parse_as_of(cls, v: Any) -> datetime.datetime:
        return _normalize_datetime(v)

    @field_validator("Correlations")
    @classmethod
    def _validate_correlations(cls, v: Dict[str, float]) -> Dict[str, float]:
        for k, corr in v.items():
            _ = CorrelationPairSpec.from_key(k, corr)
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
    """Dupire local volatility model specification with Tier 0 cross-currency flag stripping."""

    model_config = ConfigDict(extra="forbid")

    TypeName: Literal["DupireLocalVolatilityModel"] = "DupireLocalVolatilityModel"
    MarketDataSet: MarketDataSetSpec
    FinancingCurveIdentifiers: List[str] = Field(default_factory=list)
    stripped_legacy_fields: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _strip_tier0_model_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        stripped: List[str] = list(out.get("stripped_legacy_fields", []))
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

    model_config = ConfigDict(extra="forbid")

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
    CouponBarrierType: Optional[str] = "In"
    CouponBarrierDirection: Optional[str] = "Up"
    CouponBarrierAbsoluteDigitalSpread: Optional[float] = None
    OptionDownAndInBarrierLevel: Optional[float] = None
    OptionDownAndInBarrierAbsoluteDigitalSpread: Optional[float] = None
    CouponBarrierMonitoringIndex: Optional[MonitoringIndexEnum] = None
    OptionMonitoringIndex: Optional[MonitoringIndexEnum] = None
    RateIdentifier: Optional[str] = None
    FundingSpread: Optional[float] = None
    FundingLegPaymentDates: Optional[List[datetime.datetime]] = None
    NoNotionalRepayment: bool = False
    IsMemory: bool = False
    AutoCallCoupon: Optional[float] = None
    PredefinedDividends: Optional[Dict[str, PredefinedDividendsSpec]] = None

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
        if self.CouponBarrierLevels is not None and len(self.CouponBarrierLevels) != m_obs:
            raise ValueError(
                f"CouponBarrierLevels length ({len(self.CouponBarrierLevels)}) != ObservationDates length ({m_obs})"
            )
        if len(self.UnderlyingIdentifiers) == 0:
            raise ValueError("UnderlyingIdentifiers must not be empty")
        if len(self.UnderlyingIdentifiers) != len(self.UnderlyingReferenceSpots):
            raise ValueError(
                f"UnderlyingIdentifiers length ({len(self.UnderlyingIdentifiers)}) != "
                f"UnderlyingReferenceSpots length ({len(self.UnderlyingReferenceSpots)})"
            )
        if any(s <= 0.0 for s in self.UnderlyingReferenceSpots):
            raise ValueError("UnderlyingReferenceSpots must all be strictly positive")
        if self.LowerStrike is not None and self.LowerStrike > self.Strike:
            raise ValueError(f"LowerStrike ({self.LowerStrike}) must be <= Strike ({self.Strike})")
        return self

    def get_coupon_rates(self) -> List[float]:
        """Returns the per-period coupon vector of length $M$ (broadcasting scalar if needed)."""
        m = len(self.ObservationDates)
        source = self.FixedCouponAmount if self.FixedCouponAmount is not None else self.FixedRate
        if source is None:
            return [0.0] * m
        if isinstance(source, list):
            if len(source) != m:
                raise ValueError(f"Coupon vector length ({len(source)}) != number of periods ({m})")
            return [float(x) for x in source]
        return [float(source)] * m


class CashFlowRow(BaseModel):
    """Single cash-flow table entry in `EQAutocallableTradeResponse`."""

    model_config = ConfigDict(extra="forbid")

    Description: Union[Literal["FundingLeg", "Coupon", "Notional", "Option"], str]
    Date: datetime.datetime
    Amount: float
    StandardError: float = 0.0
    Currency: str = "RUB"
    Leg: int = 0
    StatusText: Union[Literal["Settled", "Estimate"], str] = "Estimate"

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
            if "Amount" not in out:
                out["Amount"] = float(exp_val)
        if "PaymentValue" in out:
            pay_val = out.pop("PaymentValue")
            if "Amount" not in out:
                out["Amount"] = float(pay_val)
        return out

    @field_validator("Date", mode="before")
    @classmethod
    def _parse_date(cls, v: Any) -> datetime.datetime:
        return _normalize_datetime(v)

    @property
    def PaymentValue(self) -> float:
        """Alias for `Amount`."""
        return float(self.Amount)


class EQAutocallableTradeRequest(BaseModel):
    """Canonical top-level pricing request with Tier 0 legacy MC/engine parameter stripping."""

    model_config = ConfigDict(extra="forbid")

    Product: AutoCallableProductSpec
    Model: DupireModelSpec
    Queries: List[AutocallQueryMetric] = Field(
        default_factory=lambda: [AutocallQueryMetric.ProductValue, AutocallQueryMetric.CashFlows]
    )
    TradeInfo: Optional[str] = None
    stripped_legacy_fields: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _strip_tier0_root_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        stripped: List[str] = list(out.get("stripped_legacy_fields", []))

        for field_name in (
            "CalculationMethod",
            "NumberOfPaths",
            "BatchSizeForMCPath",
            "StaticDataSet",
            "GreekMetricsSettings",
        ):
            if field_name in out:
                val = out.pop(field_name)
                _append_unique(stripped, [field_name])
                if field_name == "GreekMetricsSettings" and isinstance(val, Mapping):
                    for gk_key, gk_val in val.items():
                        if gk_key not in _GREEK_SETTINGS_TIER0_FIELDS:
                            raise ValueError(
                                f"Unknown field '{gk_key}' inside GreekMetricsSettings"
                            )
                        _append_unique(stripped, [gk_key])
                        if gk_key == "BumpParametersOverrides" and isinstance(gk_val, Mapping):
                            for metric_name, bump_cfg in gk_val.items():
                                if isinstance(bump_cfg, Mapping):
                                    for b_key in bump_cfg:
                                        if b_key not in _BUMP_OVERRIDE_TIER0_FIELDS:
                                            raise ValueError(
                                                f"Unknown field '{b_key}' in BumpParametersOverrides[{metric_name!r}]"
                                            )
                                        _append_unique(stripped, [b_key])

        out["stripped_legacy_fields"] = stripped
        return out

    @model_validator(mode="after")
    def _collect_nested_stripped_fields(self) -> EQAutocallableTradeRequest:
        _append_unique(self.stripped_legacy_fields, self.Model.stripped_legacy_fields)
        return self


class EQAutocallableTradeResponse(BaseModel):
    """Canonical top-level response containing `ProductValue`, `CashFlows`, Greeks, and SR 26-2 telemetry."""

    model_config = ConfigDict(extra="forbid")

    ProductValue: float
    ReportingCurrency: str = "RUB"
    StandardError: float = 0.0
    CashToSettleToday: float = 0.0
    CashFlows: List[CashFlowRow] = Field(default_factory=list)

    # Optional Greek sensitivity outputs
    Delta: Optional[Dict[str, float]] = None
    Gamma: Optional[Union[Dict[str, float], List[List[float]]]] = None
    MultivariateAggregatedCashGamma: Optional[Union[float, List[List[float]]]] = None
    Vega: Optional[Dict[str, List[List[float]]]] = None
    Rho: Optional[Dict[str, List[float]]] = None
    DividendsSensitivities: Optional[Dict[str, List[float]]] = None
    Theta: Optional[float] = None
    Cega: Optional[Union[Dict[str, float], List[List[float]]]] = None

    # Metadata & SR 11-7 / SR 26-2 governance telemetry
    TradeInfo: Optional[str] = None
    engine_used: str = "ML_SURROGATE"
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    latency_ms: float = 0.0


# ── Murex `ConstantMessageMap` DAG Tokenizer & Parser ─────────────────────────

_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+|,+)
  | (?P<SKIPPED><skipped>)
  | (?P<DATETIME>datetime\.(?:datetime|date)\([^)]*\))
  | (?P<LBRACE>\{)
  | (?P<RBRACE>\})
  | (?P<LBRACKET>\[)
  | (?P<RBRACKET>\])
  | (?P<COLON>:)
  | (?P<SQUOTE>'[^']*')
  | (?P<DQUOTE>"[^"]*")
  | (?P<REF>[@&!][0-9a-fA-F]{40})
  | (?P<WORD>[^\s,\{\}\[\]:'"]+)
    """,
    re.VERBOSE,
)

_SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_REF_STR_RE = re.compile(r"^[@&!]([0-9a-fA-F]{40})$")
_DT_ARGS_RE = re.compile(r"^datetime\.(datetime|date)\((.*)\)$")


def _parse_datetime_literal(text: str) -> datetime.datetime:
    """Safely parses a `datetime.datetime(...)` or `datetime.date(...)` literal without `eval()`."""
    m = _DT_ARGS_RE.match(text.strip())
    if m is None:
        raise ValueError(f"Malformed datetime literal: {text!r}")
    kind = m.group(1)
    inner = m.group(2)
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    int_args: List[int] = []
    tz = datetime.timezone.utc
    for p in parts:
        if "=" in p:
            k, v = [x.strip() for x in p.split("=", 1)]
            if k == "tzinfo" and "utc" in v.lower():
                tz = datetime.timezone.utc
        else:
            int_args.append(int(p))
    if kind == "date":
        return datetime.datetime(int_args[0], int_args[1], int_args[2], 0, 0, tzinfo=tz)
    while len(int_args) < 6:
        int_args.append(0)
    return datetime.datetime(
        int_args[0],
        int_args[1],
        int_args[2],
        int_args[3],
        int_args[4],
        int_args[5],
        tzinfo=tz,
    )


class _DAGTokenStream:
    """Lexer stream over Murex `ConstantMessageMap` text dumps."""

    def __init__(self, text: str) -> None:
        self.tokens: List[Tuple[str, str]] = []
        pos = 0
        n = len(text)
        while pos < n:
            m = _TOKEN_RE.match(text, pos)
            if m is None:
                raise ValueError(
                    f"Unexpected character in DAG input at position {pos}: {text[pos:pos+32]!r}"
                )
            pos = m.end()
            kind = m.lastgroup
            if kind is None or kind == "WS":
                continue
            self.tokens.append((kind, m.group(kind)))
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


def _parse_dag_value(stream: _DAGTokenStream) -> Any:
    kind, val = stream.next()
    if kind == "LBRACE":
        obj: Dict[str, Any] = {}
        while True:
            pk = stream.peek()
            if pk is None:
                raise ValueError("Unclosed '{' in Murex DAG")
            if pk[0] == "RBRACE":
                stream.next()
                return obj
            k_kind, k_val = stream.next()
            if k_kind in ("SQUOTE", "DQUOTE"):
                key_str = k_val[1:-1]
            elif k_kind == "WORD":
                key_str = k_val
            else:
                raise ValueError(f"Invalid map key token ({k_kind}, {k_val!r})")
            stream.expect("COLON")
            obj[key_str] = _parse_dag_value(stream)
    if kind == "LBRACKET":
        arr: List[Any] = []
        while True:
            pk = stream.peek()
            if pk is None:
                raise ValueError("Unclosed '[' in Murex DAG")
            if pk[0] == "RBRACKET":
                stream.next()
                return arr
            arr.append(_parse_dag_value(stream))
    if kind == "SKIPPED":
        return None
    if kind == "DATETIME":
        return _parse_datetime_literal(val)
    if kind in ("SQUOTE", "DQUOTE"):
        return val[1:-1]
    if kind == "REF":
        return val
    if kind == "WORD":
        if val in ("True", "true"):
            return True
        if val in ("False", "false"):
            return False
        if val in ("None", "null"):
            return None
        try:
            if "." not in val and "e" not in val.lower():
                return int(val)
            return float(val)
        except ValueError:
            return val
    raise ValueError(f"Unexpected token ({kind}, {val!r}) while parsing value")


def _extract_reference_hash(val: Any, nodes: Mapping[str, Any]) -> Optional[str]:
    if not isinstance(val, str):
        return None
    m = _REF_STR_RE.match(val.strip())
    if m is not None:
        h = m.group(1).lower()
        return h
    s = val.strip().lower()
    if _SHA1_RE.match(s) and s in nodes:
        return s
    return None


def _resolve_dag_node(
    obj: Any,
    nodes: Mapping[str, Dict[str, Any]],
    visiting: set[str],
) -> Any:
    ref_hash = _extract_reference_hash(obj, nodes)
    if ref_hash is not None:
        if ref_hash in visiting:
            raise ValueError(f"Circular dependency cycle detected in Murex DAG at {ref_hash}")
        if ref_hash not in nodes:
            raise KeyError(f"Dangling Murex DAG reference to missing hash {ref_hash!r}")
        visiting.add(ref_hash)
        resolved = _resolve_dag_node(nodes[ref_hash], nodes, visiting)
        visiting.remove(ref_hash)
        return resolved
    if isinstance(obj, Mapping):
        return {k: _resolve_dag_node(v, nodes, visiting) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_dag_node(item, nodes, visiting) for item in obj]
    return obj


def _enrich_expected_result(
    result_dict: Dict[str, Any],
    req: EQAutocallableTradeRequest,
) -> Dict[str, Any]:
    """Ensures `Option` and `TradeInfo` convenience keys are populated in `expected_result_dict`."""
    enriched = dict(result_dict)
    if "Option" not in enriched and isinstance(enriched.get("CashFlows"), list):
        for cf in enriched["CashFlows"]:
            if isinstance(cf, Mapping) and cf.get("Description") == "Option":
                enriched["Option"] = float(cf["Amount"])
                break
    if "TradeInfo" not in enriched and req.TradeInfo is not None:
        enriched["TradeInfo"] = req.TradeInfo
    return enriched


def parse_murex_autocall_dag(
    raw_text_or_dict: Union[str, Path, Dict[str, Any]],
) -> Tuple[EQAutocallableTradeRequest, Dict[str, Any]]:
    """Parses a Murex `ConstantMessageMap` DAG text dump (or resolved Python dict) into a validated request and reference result dict.

    Supports:
    - Raw text dumps such as `eq_autocallable_request.txt` (with top or bottom `Result:` block,
      `@<sha1>`, `'&<sha1>'`, or `'!<sha1>'` hash pointers, and `class: ConstantMessageMap` headers).
    - File paths (`str` or `Path`) pointing to a Murex DAG file.
    - Already-resolved Python dicts (either direct request dicts or `{sha1: node_dict}` maps).
    """
    if isinstance(raw_text_or_dict, Mapping):
        raw_map = dict(raw_text_or_dict)
        if "Product" in raw_map and "Model" in raw_map:
            res_block = dict(cast(Mapping[str, Any], raw_map.pop("Result", {})))
            req = EQAutocallableTradeRequest.model_validate(raw_map)
            return req, _enrich_expected_result(res_block, req)

        # Otherwise treat as {sha1_or_key: node_dict}
        nodes_from_dict: Dict[str, Dict[str, Any]] = {}
        result_from_dict: Dict[str, Any] = {}
        root_hash_from_dict: Optional[str] = None
        for k, v in raw_map.items():
            if k == "Result" and isinstance(v, Mapping):
                result_from_dict = dict(v)
            elif k in ("Root", "root_id") and isinstance(v, str):
                root_hash_from_dict = v.lstrip("@&!").lower()
            elif isinstance(v, Mapping):
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
        text = raw_text_or_dict.read_text(encoding="utf-8")
    elif isinstance(raw_text_or_dict, str):
        if "\n" not in raw_text_or_dict and len(raw_text_or_dict) < 4096 and os.path.isfile(raw_text_or_dict):
            text = Path(raw_text_or_dict).read_text(encoding="utf-8")
        else:
            text = raw_text_or_dict
    else:
        raise TypeError(
            f"Unsupported input type for parse_murex_autocall_dag: {type(raw_text_or_dict).__name__}"
        )

    stream = _DAGTokenStream(text)
    nodes: Dict[str, Dict[str, Any]] = {}
    result_dict: Dict[str, Any] = {}
    anonymous_dicts: List[Dict[str, Any]] = []
    root_hash: Optional[str] = None

    while stream.peek() is not None:
        pk0 = stream.peek(0)
        assert pk0 is not None
        kind, val = pk0

        if kind == "LBRACE":
            parsed_map = cast(Dict[str, Any], _parse_dag_value(stream))
            if "ProductValue" in parsed_map or "CashFlows" in parsed_map:
                result_dict = parsed_map
            else:
                anonymous_dicts.append(parsed_map)
        elif kind == "REF":
            stream.next()
            root_hash = val[1:].lower()
        elif kind == "WORD" and val == "Result" and stream.peek(1) is not None and stream.peek(1)[0] == "COLON":  # type: ignore[index]
            stream.next()  # Result
            stream.next()  # :
            res_val = _parse_dag_value(stream)
            if isinstance(res_val, Mapping):
                result_dict = dict(res_val)
        elif kind == "WORD" and _SHA1_RE.match(val):
            node_hash = val.lower()
            stream.next()
            # Optional `class : ConstantMessageMap` or `:` header
            pk_next = stream.peek(0)
            if pk_next is not None and pk_next[0] == "WORD" and pk_next[1] == "class":
                stream.next()  # class
                stream.expect("COLON")
                stream.next()  # ConstantMessageMap
            elif pk_next is not None and pk_next[0] == "COLON":
                stream.next()
            node_body = _parse_dag_value(stream)
            if not isinstance(node_body, Mapping):
                raise ValueError(f"Expected map body for DAG node {node_hash}")
            nodes[node_hash] = dict(node_body)
        else:
            raise ValueError(f"Unexpected top-level token in Murex DAG: ({kind}, {val!r})")

    if root_hash is None:
        for h, body in nodes.items():
            if "Product" in body and "Model" in body:
                root_hash = h
                break

    if root_hash is not None:
        if root_hash not in nodes:
            raise KeyError(f"Root hash {root_hash!r} not found in parsed ConstantMessageMap nodes")
        resolved_req_dict = _resolve_dag_node(nodes[root_hash], nodes, {root_hash})
    elif anonymous_dicts:
        resolved_req_dict = _resolve_dag_node(anonymous_dicts[0], nodes, set())
    else:
        raise ValueError("No root request node found in Murex DAG input")

    req = EQAutocallableTradeRequest.model_validate(resolved_req_dict)
    return req, _enrich_expected_result(result_dict, req)
