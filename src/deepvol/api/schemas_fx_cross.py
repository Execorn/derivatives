"""schemas_fx_cross.py — Request/response schemas for triangulated FX/commodity cross vanilla options (SPEC v3.0).

Supports the analytic smile-consistent triangulation engine, the residual-volatility differential surrogate,
the Gaussian-copula benchmark, and SR 26-2 Model Risk Guardian telemetry.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Final, List, Literal, Optional

from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator


class ParityEnum(str, Enum):
    CALL = "Call"
    PUT = "Put"


class SettlementTypeEnum(str, Enum):
    CASH = "CASH"
    PHYSICAL = "PHYSICAL"


class PricingEngineEnum(str, Enum):
    DIFFERENTIAL_ML = "Differential_ML"
    ANALYTIC_TRIANGULATED = "Analytic_Triangulated"
    ANALYTIC_GK = "Analytic_GK"  # v1.0 alias, resolved to Analytic_Triangulated
    HYBRID_ENSEMBLE = "Hybrid_Ensemble"
    COPULA_BENCHMARK = "Copula_Benchmark"  # validation engine, not latency-bound


class PrecisionEnum(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class DeltaConventionEnum(str, Enum):
    SPOT_PNA = "spot_pna"
    SPOT_PA = "spot_pa"
    FORWARD_PNA = "forward_pna"
    FORWARD_PA = "forward_pa"


class AtmConventionEnum(str, Enum):
    DNS = "DNS"  # delta-neutral straddle
    ATMF = "ATMF"  # at-the-money forward


class StrangleConventionEnum(str, Enum):
    SMILE = "SMILE"  # BF quoted as smile strangle: 0.5(s25C + s25P) - sATM
    MARKET = "MARKET"  # broker / market strangle (single-vol strangle premium), needs inversion


class CorrelationTypeEnum(str, Enum):
    ATM_TRIANGULATION = "ATM_TRIANGULATION"  # implied from an ATM cross vol quote
    HISTORICAL_LOGRETURN = "HISTORICAL_LOGRETURN"


class SmileDynamicsEnum(str, Enum):
    STICKY_MONEYNESS = (
        "STICKY_MONEYNESS"  # smile moves with the forward (sticky-delta proxy)
    )
    STICKY_STRIKE = "STICKY_STRIKE"


class QueryEnum(str, Enum):
    PRODUCT_VALUE = "ProductValue"
    DELTA = "Delta"
    CASH_DELTA = "CashDelta"
    GAMMA = "Gamma"
    VEGA = "Vega"
    THETA = "Theta"
    VANNA = "Vanna"
    VOMMA = "Vomma"
    RHO = "Rho"
    CORRELATION_SENSITIVITY = "CorrelationSensitivity"
    SMILE_VEGA = "SmileVega"
    MODEL_RESERVE = "ModelReserve"


class FallbackReasonEnum(str, Enum):
    EXTREME_MONEYNESS = "EXTREME_MONEYNESS"  # |z| >= 7: NN contribution is exactly zero
    OOD_MARKET_REGIME = "OOD_MARKET_REGIME"
    EPISTEMIC_UNCERTAINTY = "EPISTEMIC_UNCERTAINTY"
    SMILE_ARBITRAGE = "SMILE_ARBITRAGE"
    OUTPUT_AUDIT_FAILED = (
        "OUTPUT_AUDIT_FAILED"  # served cross slice failed the Durrleman / spread audit
    )
    ENGINE_REQUESTED = "ENGINE_REQUESTED"


class WarningCodeEnum(str, Enum):
    SMILE_SSVI_FALLBACK = "SMILE_SSVI_FALLBACK"
    COLD_MARKET_STATE = (
        "COLD_MARKET_STATE"  # state canonicalized on the request path (cache miss)
    )
    MODEL_RESERVE_STALE = (
        "MODEL_RESERVE_STALE"  # copula reserve older than the market state
    )
    NN_TAPERED = "NN_TAPERED"  # 5 < |z| < 7: NN contribution partially tapered


_PAIR = r"^[A-Z]{6}$"
_CCY = r"^[A-Z]{3}$"
PAIR_PATTERN: Final[str] = _PAIR
CCY_PATTERN: Final[str] = _CCY


class SmileQuotes(BaseModel):
    """5-point market smile at the option expiry, with the quoting conventions needed to map deltas to strikes."""

    atm_vol: float = Field(..., ge=0.01, le=3.0)
    rr25: float = Field(..., ge=-0.50, le=0.50)
    bf25: float = Field(..., ge=-0.05, le=0.30)
    rr10: float = Field(..., ge=-0.80, le=0.80)
    bf10: float = Field(..., ge=-0.05, le=0.50)
    delta_convention: DeltaConventionEnum = Field(default=DeltaConventionEnum.SPOT_PNA)
    atm_convention: AtmConventionEnum = Field(default=AtmConventionEnum.DNS)
    strangle_convention: StrangleConventionEnum = Field(
        default=StrangleConventionEnum.SMILE
    )

    @model_validator(mode="after")
    def pillar_vols_positive(self) -> SmileQuotes:
        for bf, rr, tag in ((self.bf25, self.rr25, "25"), (self.bf10, self.rr10, "10")):
            if self.atm_vol + bf - 0.5 * abs(rr) < 0.01:
                raise ValueError(
                    f"{tag}-delta wing vol below 1% floor: atm + bf{tag} - |rr{tag}|/2 < 0.01"
                )
        return self


class FXLegData(BaseModel):
    pair: str = Field(..., pattern=_PAIR)
    spot: float = Field(..., gt=0.0)
    foreign_discount_factor: float = Field(
        ..., ge=1e-4, le=2.5, description="Base ccy/asset DF to T_exp."
    )
    domestic_discount_factor: float = Field(
        ..., ge=1e-4, le=2.5, description="Quote ccy DF to T_exp."
    )
    smile: SmileQuotes

    @property
    def base(self) -> str:
        return self.pair[:3]

    @property
    def quote(self) -> str:
        return self.pair[3:]


class CurvePillar(BaseModel):
    pillar_date: AwareDatetime
    tenor_years: float = Field(..., ge=0.0, le=60.0)
    zero_rate: float = Field(
        ...,
        ge=-0.20,
        le=2.0,
        description="Continuously compounded zero rate (decimal).",
    )


class ContractTerms(BaseModel):
    identifier: str = Field(..., pattern=_PAIR)
    parity: ParityEnum
    strike: float = Field(..., ge=1e-6, le=1e9)
    as_of_date: AwareDatetime
    expiration_date: AwareDatetime
    settlement_date: AwareDatetime
    payment_currency: str = Field(..., pattern=_CCY)
    settlement_type: SettlementTypeEnum = SettlementTypeEnum.CASH
    calendar: str = Field(default="NYB,MWB,LNB")

    @model_validator(mode="after")
    def date_order(self) -> ContractTerms:
        if not (self.as_of_date < self.expiration_date <= self.settlement_date):
            raise ValueError("require as_of_date < expiration_date <= settlement_date")
        return self


class MarketData(BaseModel):
    leg1: FXLegData = Field(
        ..., description="Asset A in intermediate ccy B, e.g. XAUUSD."
    )
    leg2: FXLegData = Field(..., description="Ccy B in target ccy C, e.g. USDRUB.")
    correlation: float = Field(..., ge=-0.999, le=0.999)
    correlation_type: CorrelationTypeEnum = CorrelationTypeEnum.HISTORICAL_LOGRETURN
    settlement_discount_factor: float = Field(
        ..., ge=1e-4, le=2.5, description="D_C(T_set)."
    )

    @model_validator(mode="after")
    def triangle_consistency(self) -> MarketData:
        if self.leg1.quote != self.leg2.base:
            raise ValueError(
                f"leg1 quote ccy {self.leg1.quote} must equal leg2 base ccy {self.leg2.base}"
            )
        d1, d2 = self.leg1.domestic_discount_factor, self.leg2.foreign_discount_factor
        if abs(d1 - d2) > 1e-6 * max(d1, d2):
            raise ValueError(
                "leg1.domestic_discount_factor and leg2.foreign_discount_factor are both D_B(T_exp) and must agree"
            )
        return self


class InterestRates(BaseModel):
    domestic_curve: List[CurvePillar] = Field(
        default_factory=list,
        max_length=60,
        description="Ccy C curve, e.g. RUB_RUONIA_OIS.",
    )
    foreign_curve: List[CurvePillar] = Field(
        default_factory=list,
        max_length=60,
        description="Asset A curve, e.g. XAU lease.",
    )

    @field_validator("domestic_curve", "foreign_curve")
    @classmethod
    def increasing_tenors(cls, v: List[CurvePillar]) -> List[CurvePillar]:
        t = [p.tenor_years for p in v]
        if any(b <= a for a, b in zip(t, t[1:])):
            raise ValueError("curve pillars must have strictly increasing tenor_years")
        return v


def _default_queries() -> List[QueryEnum]:
    return [q for q in QueryEnum if q is not QueryEnum.MODEL_RESERVE]


class ExecutionOptions(BaseModel):
    engine: PricingEngineEnum = PricingEngineEnum.DIFFERENTIAL_ML
    precision: PrecisionEnum = Field(
        default=PrecisionEnum.FLOAT64,
        description="Analytic-layer precision; the network runs in float32.",
    )
    use_guardian: bool = True
    smile_dynamics: SmileDynamicsEnum = Field(
        default=SmileDynamicsEnum.STICKY_MONEYNESS,
        description="Smile dynamics assumed for Delta/Gamma/Vanna.",
    )
    requested_queries: List[QueryEnum] = Field(default_factory=_default_queries)


class TriangulatedFXOptionRequest(BaseModel):
    contract: ContractTerms
    market_data: MarketData
    interest_rates: InterestRates = Field(default_factory=InterestRates)
    execution_options: ExecutionOptions = Field(default_factory=ExecutionOptions)

    @model_validator(mode="after")
    def cross_identity(self) -> TriangulatedFXOptionRequest:
        md, c = self.market_data, self.contract
        if c.identifier != md.leg1.base + md.leg2.quote:
            raise ValueError(
                f"identifier {c.identifier} != leg1 base + leg2 quote ({md.leg1.base + md.leg2.quote})"
            )
        if c.payment_currency != md.leg2.quote:
            raise ValueError(
                "payment_currency must equal the cross quote ccy C; quanto/compo payoffs are out of scope"
            )
        return self


class MarketStateColumnar(BaseModel):
    """One (cross, expiry) market state shared by many contracts in a columnar batch."""

    state_id: str
    as_of_date: AwareDatetime
    expiration_date: AwareDatetime
    settlement_date: AwareDatetime
    market_data: MarketData


_MAX_STATES = 64
_MAX_CONTRACTS = 10_000
MAX_STATES: Final[int] = _MAX_STATES
MAX_CONTRACTS: Final[int] = _MAX_CONTRACTS
_Strike = Annotated[float, Field(ge=1e-6, le=1e9, allow_inf_nan=False)]
_Notional = Annotated[float, Field(allow_inf_nan=False)]
_StateIdx = Annotated[int, Field(ge=0, lt=_MAX_STATES)]


class TriangulatedFXOptionColumnarBatchRequest(BaseModel):
    """Columnar (SoA) batch: contracts reference shared market states by index. Up to 10,000 contracts."""

    identifier: str = Field(..., pattern=_PAIR)
    payment_currency: str = Field(..., pattern=_CCY)
    states: List[MarketStateColumnar] = Field(..., min_length=1, max_length=_MAX_STATES)
    state_index: List[_StateIdx] = Field(..., min_length=1, max_length=_MAX_CONTRACTS)
    strike: List[_Strike] = Field(..., min_length=1, max_length=_MAX_CONTRACTS)
    parity_sign: List[Literal[-1, 1]] = Field(
        ..., min_length=1, max_length=_MAX_CONTRACTS, description="+1 call, -1 put"
    )
    notional: Optional[List[_Notional]] = Field(default=None, max_length=_MAX_CONTRACTS)
    execution_options: ExecutionOptions = Field(default_factory=ExecutionOptions)

    @model_validator(mode="after")
    def consistency(self) -> TriangulatedFXOptionColumnarBatchRequest:
        n = len(self.strike)
        if (
            len(self.state_index) != n
            or len(self.parity_sign) != n
            or (self.notional is not None and len(self.notional) != n)
        ):
            raise ValueError("columnar arrays must have equal length")
        num_states = len(self.states)
        if max(self.state_index) >= num_states:
            raise ValueError("state_index out of range")
        ident = self.identifier
        ident_base = ident[:3]
        ident_quote = ident[3:]
        pay_ccy = self.payment_currency
        for st in self.states:
            md = st.market_data
            l1_pair = md.leg1.pair
            l2_pair = md.leg2.pair
            if ident_base != l1_pair[:3] or ident_quote != l2_pair[3:]:
                raise ValueError(
                    f"state {st.state_id}: identifier != leg1 base + leg2 quote"
                )
            if pay_ccy != l2_pair[3:]:
                raise ValueError(
                    f"state {st.state_id}: payment_currency must equal the cross quote ccy C"
                )
            if not (st.as_of_date < st.expiration_date <= st.settlement_date):
                raise ValueError(
                    f"state {st.state_id}: require as_of_date < expiration_date <= settlement_date"
                )
        return self


# ── Response models ───────────────────────────────────────────────────────────


class DeltaLegs(BaseModel):
    cross: float
    leg1: float
    leg2: float


class CashDeltaLegs(BaseModel):
    cross: float
    leg1: float
    leg2: float


class VegaLegs(BaseModel):
    cross: float = Field(
        ..., description="dV/d(sigma_cross), parallel shift of the cross smile."
    )
    leg1: float = Field(
        ..., description="dV/d(sigma_1) parallel shift of the leg-1 smile."
    )
    leg2: float
    smile_matrix_leg1: Optional[List[float]] = Field(
        None, min_length=5, max_length=5, description="[ATM, RR25, BF25, RR10, BF10]"
    )
    smile_matrix_leg2: Optional[List[float]] = Field(
        None, min_length=5, max_length=5, description="[ATM, RR25, BF25, RR10, BF10]"
    )


class ModelRiskTelemetry(BaseModel):
    engine_used: PricingEngineEnum
    standardized_moneyness: float = Field(
        ..., description="z = ln(K/F)/(sigma_ref sqrt(T))."
    )
    mahalanobis_distance: float
    ood_threshold: float
    is_out_of_distribution: bool
    ensemble_epistemic_std_bps: float
    fallback_activated: bool
    fallback_reason: Optional[FallbackReasonEnum] = None
    durrleman_min_g: Optional[float] = Field(
        None, description="min g(k) of the output cross smile on the audit grid."
    )
    smile_dynamics: SmileDynamicsEnum = SmileDynamicsEnum.STICKY_MONEYNESS
    model_version: str = Field(
        ..., description="Weights hash + code version, for SR 11-7 traceability."
    )


class TriangulatedFXOptionResponse(BaseModel):
    status: str = "done"
    product_value: float
    implied_vol_cross: float = Field(
        ..., description="Black-76 implied cross vol used for valuation."
    )
    forward_cross: float
    delta: DeltaLegs
    cash_delta: CashDeltaLegs
    gamma: float = Field(
        ...,
        description="Smile-adjusted cross gamma d2V/dS_cross^2 under smile_dynamics.",
    )
    vega: VegaLegs
    theta: float = Field(
        ..., description="dV/dt per calendar day, spot and smile held fixed."
    )
    vanna: float = Field(..., description="d2V/(dS_cross d sigma_cross).")
    vomma: float = Field(..., description="d2V/d sigma_cross^2.")
    correlation_sensitivity: float
    rho_domestic: Optional[List[float]] = None
    rho_foreign: Optional[List[float]] = None
    model_reserve_bps: Optional[float] = Field(
        None, description="|Copula benchmark - served price| in bp of D_set*F."
    )
    warnings: List[WarningCodeEnum] = Field(default_factory=list)
    latency_ms: float
    telemetry: ModelRiskTelemetry


class TriangulatedFXOptionColumnarBatchResponse(BaseModel):
    """Columnar response aligned with the request arrays (index i <-> request contract i)."""

    status: str = "done"
    product_value: List[float]
    implied_vol_cross: List[float]
    delta_cross: List[float]
    cash_delta: List[float] = Field(
        ..., description="phi D_set F N(phi d1); equal for the cross and both legs."
    )
    gamma_cross: List[float]
    vega_cross: List[float]
    vega_leg1: List[float]
    vega_leg2: List[float]
    theta: List[float] = Field(..., description="Per calendar day.")
    vanna_cross: List[float]
    vomma_cross: List[float]
    correlation_sensitivity: List[float]
    fallback_mask: List[bool]
    fallback_reason: List[Optional[FallbackReasonEnum]]
    state_model_reserve_bps: List[Optional[float]] = Field(
        ..., description="Per market state, aligned with request.states."
    )
    state_durrleman_min_g: List[Optional[float]] = Field(
        ..., description="Per market state, aligned with request.states."
    )
    cold_states: int = Field(
        0,
        ge=0,
        description="Market states canonicalized on this request (cache misses).",
    )
    engine_used: PricingEngineEnum
    model_version: str
    latency_ms: float


# v2.0 backward-compatibility aliases
TriangulatedFXOptionBatchRequest = TriangulatedFXOptionColumnarBatchRequest
TriangulatedFXOptionBatchResponse = TriangulatedFXOptionColumnarBatchResponse
