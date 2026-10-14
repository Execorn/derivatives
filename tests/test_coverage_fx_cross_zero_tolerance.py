"""Comprehensive 100% line and branch coverage tests for zero-tolerance FX cross modules:
- src/deepvol/api/schemas_fx_cross.py
- src/deepvol/models/fx_cross_safeguards.py
- src/deepvol/mrm/fx_cross_guardian.py
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Any, List, Optional
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# ── Module Imports ─────────────────────────────────────────────────────────────
from deepvol.api.schemas_fx_cross import (
    AtmConventionEnum,
    CashDeltaLegs,
    ContractTerms,
    CorrelationTypeEnum,
    CurvePillar,
    DeltaConventionEnum,
    DeltaLegs,
    ExecutionOptions,
    FallbackReasonEnum,
    FXLegData,
    InterestRates,
    MarketData,
    MarketStateColumnar,
    ModelRiskTelemetry,
    ParityEnum,
    PrecisionEnum,
    PricingEngineEnum,
    QueryEnum,
    SettlementTypeEnum,
    SmileDynamicsEnum,
    SmileQuotes,
    StrangleConventionEnum,
    TriangulatedFXOptionBatchRequest,
    TriangulatedFXOptionBatchResponse,
    TriangulatedFXOptionColumnarBatchRequest,
    TriangulatedFXOptionColumnarBatchResponse,
    TriangulatedFXOptionRequest,
    TriangulatedFXOptionResponse,
    VegaLegs,
    WarningCodeEnum,
    _default_queries,
)
from deepvol.market.svi_slice import RawSVI
from deepvol.models import taylor_jets as tj
from deepvol.models.fx_cross_analytic import (
    AnalyticTriangulated,
    Col,
)
from deepvol.models.fx_cross_copula import CrossState, GaussianCopulaCrossEngine
from deepvol.models.fx_cross_safeguards import (
    AuditResult,
    cross_slice_audit,
    nn_tapered_mask,
    smooth_time_floor,
    smooth_vol_floor,
    taper_jet,
    wing_slope_certificate,
    z_squash,
)
from deepvol.mrm.fx_cross_guardian import (
    FEATURE_NAMES_12D,
    FXCrossGuardian,
    MarketStateDriftMonitor,
    MarketStateOODDetector,
    ModelReserveEntry,
    ModelReserveTracker,
    RoutingDecision,
)


# ==============================================================================
# SECTION 1: SCHEMAS ZERO-TOLERANCE COVERAGE
# ==============================================================================

def test_schemas_enums_and_constants() -> None:
    """Verify all enums, members, and module constants."""
    enums = [
        ParityEnum,
        SettlementTypeEnum,
        PricingEngineEnum,
        PrecisionEnum,
        DeltaConventionEnum,
        AtmConventionEnum,
        StrangleConventionEnum,
        CorrelationTypeEnum,
        SmileDynamicsEnum,
        QueryEnum,
        FallbackReasonEnum,
        WarningCodeEnum,
    ]
    for enum_cls in enums:
        for member in enum_cls:
            assert str(member.value) == member.value

    # Test default queries
    dq = _default_queries()
    assert QueryEnum.MODEL_RESERVE not in dq
    assert QueryEnum.PRODUCT_VALUE in dq


def test_smile_quotes_validation() -> None:
    """Exercise SmileQuotes validation branches including wing floor violations."""
    sq = SmileQuotes(
        atm_vol=0.20,
        rr25=0.02,
        bf25=0.015,
        rr10=0.04,
        bf10=0.05,
    )
    assert sq.atm_vol == 0.20

    # 25-delta wing vol below 0.01 floor
    with pytest.raises(ValueError, match="25-delta wing vol below 1% floor"):
        SmileQuotes(
            atm_vol=0.02,
            rr25=0.04,
            bf25=-0.015,
            rr10=0.01,
            bf10=0.01,
        )

    # 10-delta wing vol below 0.01 floor
    with pytest.raises(ValueError, match="10-delta wing vol below 1% floor"):
        SmileQuotes(
            atm_vol=0.02,
            rr25=0.01,
            bf25=0.01,
            rr10=0.05,
            bf10=-0.02,
        )


def test_fx_leg_data_properties() -> None:
    """Exercise FXLegData properties base and quote."""
    sq = SmileQuotes(atm_vol=0.15, rr25=0.0, bf25=0.01, rr10=0.0, bf10=0.02)
    leg = FXLegData(
        pair="XAUUSD",
        spot=2600.0,
        foreign_discount_factor=0.98,
        domestic_discount_factor=0.95,
        smile=sq,
    )
    assert leg.base == "XAU"
    assert leg.quote == "USD"


def test_contract_terms_date_validation() -> None:
    """Exercise ContractTerms date ordering validator branches."""
    t0 = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2027, 10, 11, 12, 0, tzinfo=timezone.utc)
    t2 = datetime(2027, 10, 13, 12, 0, tzinfo=timezone.utc)

    # Valid
    c = ContractTerms(
        identifier="XAURUB",
        parity=ParityEnum.CALL,
        strike=2600.0,
        as_of_date=t0,
        expiration_date=t1,
        settlement_date=t2,
        payment_currency="RUB",
    )
    assert c.identifier == "XAURUB"

    # as_of >= exp
    with pytest.raises(ValueError, match="require as_of_date < expiration_date"):
        ContractTerms(
            identifier="XAURUB",
            parity=ParityEnum.CALL,
            strike=2600.0,
            as_of_date=t1,
            expiration_date=t1,
            settlement_date=t2,
            payment_currency="RUB",
        )

    # exp > set
    with pytest.raises(ValueError, match="require as_of_date < expiration_date"):
        ContractTerms(
            identifier="XAURUB",
            parity=ParityEnum.CALL,
            strike=2600.0,
            as_of_date=t0,
            expiration_date=t2,
            settlement_date=t1,
            payment_currency="RUB",
        )


def test_market_data_triangle_validation() -> None:
    """Exercise MarketData triangle consistency and discount agreement branches."""
    sq = SmileQuotes(atm_vol=0.15, rr25=0.0, bf25=0.01, rr10=0.0, bf10=0.02)
    leg1 = FXLegData(
        pair="XAUUSD",
        spot=2600.0,
        foreign_discount_factor=0.98,
        domestic_discount_factor=0.95,
        smile=sq,
    )
    leg2 = FXLegData(
        pair="USDRUB",
        spot=95.0,
        foreign_discount_factor=0.95,
        domestic_discount_factor=0.85,
        smile=sq,
    )

    # Valid
    md = MarketData(
        leg1=leg1,
        leg2=leg2,
        correlation=0.25,
        settlement_discount_factor=0.85,
    )
    assert md.correlation == 0.25

    # Currency mismatch: leg1.quote != leg2.base
    leg2_bad_ccy = FXLegData(
        pair="EURRUB",
        spot=95.0,
        foreign_discount_factor=0.95,
        domestic_discount_factor=0.85,
        smile=sq,
    )
    with pytest.raises(ValueError, match="must equal leg2 base ccy"):
        MarketData(
            leg1=leg1,
            leg2=leg2_bad_ccy,
            correlation=0.25,
            settlement_discount_factor=0.85,
        )

    # Discount factor mismatch: d1 != d2
    leg2_bad_df = FXLegData(
        pair="USDRUB",
        spot=95.0,
        foreign_discount_factor=0.90,  # diff from leg1.domestic 0.95
        domestic_discount_factor=0.85,
        smile=sq,
    )
    with pytest.raises(ValueError, match="are both D_B.*and must agree"):
        MarketData(
            leg1=leg1,
            leg2=leg2_bad_df,
            correlation=0.25,
            settlement_discount_factor=0.85,
        )


def test_interest_rates_tenor_monotonicity() -> None:
    """Exercise InterestRates increasing_tenors validator (Lines 223-226 in schemas_fx_cross.py)."""
    now = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
    p1 = CurvePillar(pillar_date=now, tenor_years=0.5, zero_rate=0.05)
    p2 = CurvePillar(pillar_date=now, tenor_years=1.0, zero_rate=0.06)

    # Valid increasing
    ir = InterestRates(domestic_curve=[p1, p2], foreign_curve=[p1, p2])
    assert len(ir.domestic_curve) == 2

    # Non-increasing domestic curve
    with pytest.raises(ValueError, match="strictly increasing tenor_years"):
        InterestRates(domestic_curve=[p2, p1])

    # Equal tenors
    p2_dup = CurvePillar(pillar_date=now, tenor_years=0.5, zero_rate=0.06)
    with pytest.raises(ValueError, match="strictly increasing tenor_years"):
        InterestRates(foreign_curve=[p1, p2_dup])


def test_triangulated_fx_option_request_validation() -> None:
    """Exercise TriangulatedFXOptionRequest cross_identity and payment_currency validators."""
    t0 = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2027, 10, 11, 12, 0, tzinfo=timezone.utc)
    t2 = datetime(2027, 10, 13, 12, 0, tzinfo=timezone.utc)

    sq = SmileQuotes(atm_vol=0.15, rr25=0.0, bf25=0.01, rr10=0.0, bf10=0.02)
    leg1 = FXLegData(pair="XAUUSD", spot=2600.0, foreign_discount_factor=0.98, domestic_discount_factor=0.95, smile=sq)
    leg2 = FXLegData(pair="USDRUB", spot=95.0, foreign_discount_factor=0.95, domestic_discount_factor=0.85, smile=sq)
    md = MarketData(leg1=leg1, leg2=leg2, correlation=0.25, settlement_discount_factor=0.85)

    # Valid
    c_ok = ContractTerms(
        identifier="XAURUB",
        parity=ParityEnum.CALL,
        strike=2600.0,
        as_of_date=t0,
        expiration_date=t1,
        settlement_date=t2,
        payment_currency="RUB",
    )
    req = TriangulatedFXOptionRequest(contract=c_ok, market_data=md)
    assert req.contract.identifier == "XAURUB"

    # Identifier mismatch
    c_bad_id = ContractTerms(
        identifier="EURRUB",
        parity=ParityEnum.CALL,
        strike=2600.0,
        as_of_date=t0,
        expiration_date=t1,
        settlement_date=t2,
        payment_currency="RUB",
    )
    with pytest.raises(ValueError, match="identifier EURRUB != leg1 base \\+ leg2 quote"):
        TriangulatedFXOptionRequest(contract=c_bad_id, market_data=md)

    # Payment currency mismatch
    c_bad_pay = ContractTerms(
        identifier="XAURUB",
        parity=ParityEnum.CALL,
        strike=2600.0,
        as_of_date=t0,
        expiration_date=t1,
        settlement_date=t2,
        payment_currency="USD",
    )
    with pytest.raises(ValueError, match="payment_currency must equal the cross quote ccy C"):
        TriangulatedFXOptionRequest(contract=c_bad_pay, market_data=md)


def test_columnar_batch_request_validation() -> None:
    """Exercise TriangulatedFXOptionColumnarBatchRequest validation branches including Line 318."""
    t0 = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2027, 10, 11, 12, 0, tzinfo=timezone.utc)
    t2 = datetime(2027, 10, 13, 12, 0, tzinfo=timezone.utc)

    sq = SmileQuotes(atm_vol=0.15, rr25=0.0, bf25=0.01, rr10=0.0, bf10=0.02)
    leg1 = FXLegData(pair="XAUUSD", spot=2600.0, foreign_discount_factor=0.98, domestic_discount_factor=0.95, smile=sq)
    leg2 = FXLegData(pair="USDRUB", spot=95.0, foreign_discount_factor=0.95, domestic_discount_factor=0.85, smile=sq)
    md = MarketData(leg1=leg1, leg2=leg2, correlation=0.25, settlement_discount_factor=0.85)

    st0 = MarketStateColumnar(
        state_id="S0",
        as_of_date=t0,
        expiration_date=t1,
        settlement_date=t2,
        market_data=md,
    )

    # Valid
    batch_ok = TriangulatedFXOptionColumnarBatchRequest(
        identifier="XAURUB",
        payment_currency="RUB",
        states=[st0],
        state_index=[0, 0],
        strike=[2500.0, 2600.0],
        parity_sign=[1, -1],
        notional=[1000.0, 2000.0],
    )
    assert len(batch_ok.strike) == 2

    # Array length mismatch: state_index vs strike
    with pytest.raises(ValueError, match="columnar arrays must have equal length"):
        TriangulatedFXOptionColumnarBatchRequest(
            identifier="XAURUB",
            payment_currency="RUB",
            states=[st0],
            state_index=[0],
            strike=[2500.0, 2600.0],
            parity_sign=[1, -1],
        )

    # Array length mismatch: parity_sign vs strike
    with pytest.raises(ValueError, match="columnar arrays must have equal length"):
        TriangulatedFXOptionColumnarBatchRequest(
            identifier="XAURUB",
            payment_currency="RUB",
            states=[st0],
            state_index=[0, 0],
            strike=[2500.0, 2600.0],
            parity_sign=[1],
        )

    # Array length mismatch: notional vs strike
    with pytest.raises(ValueError, match="columnar arrays must have equal length"):
        TriangulatedFXOptionColumnarBatchRequest(
            identifier="XAURUB",
            payment_currency="RUB",
            states=[st0],
            state_index=[0, 0],
            strike=[2500.0, 2600.0],
            parity_sign=[1, -1],
            notional=[1000.0],
        )

    # state_index out of range
    with pytest.raises(ValueError, match="state_index out of range"):
        TriangulatedFXOptionColumnarBatchRequest(
            identifier="XAURUB",
            payment_currency="RUB",
            states=[st0],
            state_index=[0, 1],  # 1 >= len(states)
            strike=[2500.0, 2600.0],
            parity_sign=[1, -1],
        )

    # State identifier mismatch
    leg1_other = FXLegData(pair="EURUSD", spot=1.10, foreign_discount_factor=0.98, domestic_discount_factor=0.95, smile=sq)
    md_other = MarketData(leg1=leg1_other, leg2=leg2, correlation=0.25, settlement_discount_factor=0.85)
    st_bad_id = MarketStateColumnar(state_id="S_BAD", as_of_date=t0, expiration_date=t1, settlement_date=t2, market_data=md_other)
    with pytest.raises(ValueError, match="identifier != leg1 base \\+ leg2 quote"):
        TriangulatedFXOptionColumnarBatchRequest(
            identifier="XAURUB",
            payment_currency="RUB",
            states=[st_bad_id],
            state_index=[0],
            strike=[2500.0],
            parity_sign=[1],
        )

    # State payment_currency mismatch (LINE 318 in schemas_fx_cross.py!)
    with pytest.raises(ValueError, match="payment_currency must equal the cross quote ccy C"):
        TriangulatedFXOptionColumnarBatchRequest(
            identifier="XAURUB",
            payment_currency="USD",  # USD != leg2.quote "RUB"
            states=[st0],
            state_index=[0],
            strike=[2500.0],
            parity_sign=[1],
        )

    # State date order violation
    st_bad_date = MarketStateColumnar(state_id="S_BAD_D", as_of_date=t1, expiration_date=t0, settlement_date=t2, market_data=md)
    with pytest.raises(ValueError, match="require as_of_date < expiration_date"):
        TriangulatedFXOptionColumnarBatchRequest(
            identifier="XAURUB",
            payment_currency="RUB",
            states=[st_bad_date],
            state_index=[0],
            strike=[2500.0],
            parity_sign=[1],
        )


def test_response_models_instantiation() -> None:
    """Exercise response models and aliases."""
    delta = DeltaLegs(cross=0.5, leg1=0.48, leg2=0.02)
    cash_delta = CashDeltaLegs(cross=125000.0, leg1=125000.0, leg2=125000.0)
    vega = VegaLegs(cross=100.0, leg1=80.0, leg2=60.0, smile_matrix_leg1=[0.1, 0.0, 0.0, 0.0, 0.0], smile_matrix_leg2=[0.2, 0.0, 0.0, 0.0, 0.0])
    telem = ModelRiskTelemetry(
        engine_used=PricingEngineEnum.DIFFERENTIAL_ML,
        standardized_moneyness=0.25,
        mahalanobis_distance=1.8,
        ood_threshold=3.5,
        is_out_of_distribution=False,
        ensemble_epistemic_std_bps=0.45,
        fallback_activated=False,
        model_version="v3.0.0",
    )
    resp = TriangulatedFXOptionResponse(
        product_value=120.0,
        implied_vol_cross=0.22,
        forward_cross=250000.0,
        delta=delta,
        cash_delta=cash_delta,
        gamma=0.005,
        vega=vega,
        theta=-12.5,
        vanna=0.01,
        vomma=0.02,
        correlation_sensitivity=15.0,
        latency_ms=1.2,
        telemetry=telem,
    )
    assert resp.product_value == 120.0

    batch_resp = TriangulatedFXOptionColumnarBatchResponse(
        product_value=[120.0],
        implied_vol_cross=[0.22],
        delta_cross=[0.5],
        cash_delta=[125000.0],
        gamma_cross=[0.005],
        vega_cross=[100.0],
        vega_leg1=[80.0],
        vega_leg2=[60.0],
        theta=[-12.5],
        vanna_cross=[0.01],
        vomma_cross=[0.02],
        correlation_sensitivity=[15.0],
        fallback_mask=[False],
        fallback_reason=[None],
        state_model_reserve_bps=[1.2],
        state_durrleman_min_g=[0.35],
        engine_used=PricingEngineEnum.DIFFERENTIAL_ML,
        model_version="v3.0.0",
        latency_ms=5.0,
    )
    assert len(batch_resp.product_value) == 1

    # Aliases
    assert TriangulatedFXOptionBatchRequest is TriangulatedFXOptionColumnarBatchRequest
    assert TriangulatedFXOptionBatchResponse is TriangulatedFXOptionColumnarBatchResponse


# ==============================================================================
# SECTION 2: SAFEGUARDS ZERO-TOLERANCE COVERAGE
# ==============================================================================

def test_safeguards_taper_jet_and_squash() -> None:
    """Exercise taper_jet across all moneyness regions and signs."""
    # |z| <= 5.0 (core region)
    z_core_pos = tj.const(torch.tensor([2.0], dtype=torch.float64))
    z_core_neg = tj.const(torch.tensor([-3.0], dtype=torch.float64))
    res_cp = taper_jet(z_core_pos)
    res_cn = taper_jet(z_core_neg)
    assert res_cp.v.item() == 1.0 and res_cp.d.item() == 0.0
    assert res_cn.v.item() == 1.0 and res_cn.d.item() == 0.0

    # 5.0 < |z| < 7.0 (transition zone)
    z_trans_pos = tj.seed_k(torch.tensor([6.0], dtype=torch.float64))
    z_trans_neg = tj.seed_k(torch.tensor([-5.5], dtype=torch.float64))
    res_tp = taper_jet(z_trans_pos)
    res_tn = taper_jet(z_trans_neg)
    assert 0.0 < res_tp.v.item() < 1.0
    assert 0.0 < res_tn.v.item() < 1.0
    assert res_tp.d.item() < 0.0  # decreasing for positive z
    assert res_tn.d.item() > 0.0  # increasing for negative z moving towards 0

    # |z| >= 7.0 (deep wings)
    z_wing_pos = tj.const(torch.tensor([8.0], dtype=torch.float64))
    z_wing_neg = tj.const(torch.tensor([-10.0], dtype=torch.float64))
    res_wp = taper_jet(z_wing_pos)
    res_wn = taper_jet(z_wing_neg)
    assert res_wp.v.item() == 0.0 and res_wp.d.item() == 0.0
    assert res_wn.v.item() == 0.0 and res_wn.d.item() == 0.0

    # z_squash
    z_sq = z_squash(z_trans_pos, c=7.0)
    assert z_sq.v.item() < 7.0

    # nn_tapered_mask
    mask = nn_tapered_mask(torch.tensor([4.0, 6.0, 8.0], dtype=torch.float64))
    assert mask.tolist() == [False, True, False]

    # smooth_vol_floor & smooth_time_floor
    s_floor = smooth_vol_floor(z_core_pos, lo=0.01)
    assert s_floor.v.item() > 0.01
    t_floor = smooth_time_floor(torch.tensor([0.0, 1.0], dtype=torch.float64), lo=1e-4)
    assert (t_floor >= 1e-4).all()


def test_wing_slope_certificate_1d_and_2d() -> None:
    """Exercise wing_slope_certificate with 1D and 2D tensors (Lines 109-110 in safeguards)."""
    L1 = torch.tensor([0.5], dtype=torch.float64)
    L2 = torch.tensor([0.6], dtype=torch.float64)
    rho = torch.tensor([0.2], dtype=torch.float64)

    # 1D beta_wing
    beta_1d = torch.tensor([0.4], dtype=torch.float64)
    cert_1d = wing_slope_certificate(L1, L2, beta_1d, rho, limit=1.9)
    assert cert_1d.item() is True

    # 2D beta_wing: ndim > 1 and shape[-1] == 2 (Lines 109-110)
    beta_2d = torch.tensor([[0.4, 0.6]], dtype=torch.float64)
    cert_2d = wing_slope_certificate(L1, L2, beta_2d, rho, limit=1.9)
    assert cert_2d.item() is True

    # Limit exceeded (False branch)
    cert_false = wing_slope_certificate(L1, L2, beta_1d, rho, limit=0.1)
    assert cert_false.item() is False


def test_cross_slice_audit_branches() -> None:
    """Exercise cross_slice_audit across all branches:
    - 1D input (Line 143)
    - S_cnt == 1 (Line 157, 251)
    - Degenerate denom_beta <= 1e-8 (Lines 230-231)
    - cross_ids as Tensor and list (Lines 270-275)
    - Group size <= 1 (Line 286)
    - Group expiries identical (Line 290)
    - Multi-expiry calendar check
    """
    engine = AnalyticTriangulated()

    def make_state(F: float = 250000.0, T: float = 1.0, rho: float = 0.25, degenerate: bool = False) -> Tensor:
        st = torch.zeros(34, dtype=torch.float64)
        st[Col.F] = F
        st[Col.T] = T
        st[Col.T_SET] = T
        st[Col.D] = 0.85
        st[Col.G] = 1.0
        st[Col.RHO] = rho
        if degenerate:
            # Zero out b1 and b2 to make L1_val = 0, L2_val = 0 -> denom_beta = 0 <= 1e-8
            st[Col.SVI1 : Col.SVI1 + 5] = torch.tensor([0.04, 0.0, 0.0, 0.0, 0.1], dtype=torch.float64)
            st[Col.SVI2 : Col.SVI2 + 5] = torch.tensor([0.04, 0.0, 0.0, 0.0, 0.1], dtype=torch.float64)
        else:
            st[Col.SVI1 : Col.SVI1 + 5] = torch.tensor([0.04, 0.08, -0.2, 0.0, 0.1], dtype=torch.float64)
            st[Col.SVI2 : Col.SVI2 + 5] = torch.tensor([0.06, 0.10, 0.1, 0.0, 0.1], dtype=torch.float64)
        return st

    # 1. 1D X_states (Line 143: X_states.dim() == 1)
    st_1d = make_state(T=1.0)
    res_1d = cross_slice_audit(engine, st_1d, n_grid=21)
    assert res_1d.min_g.shape == (1,)
    assert res_1d.ok.item() is True

    # 2. Degenerate denom_beta <= 1e-8 (Lines 230-231: beta1_val=0.5, beta2_val=0.5)
    st_degen = make_state(degenerate=True)
    res_degen = cross_slice_audit(engine, st_degen, n_grid=21)
    assert res_degen.slope_bound[0].item() >= 0.0

    # 3. S_cnt > 1 without cross_ids (grouping by (RHO, F/G))
    st_t1 = make_state(T=0.5)
    st_t2 = make_state(T=1.5)
    states_multi = torch.stack([st_t1, st_t2], dim=0)
    res_multi = cross_slice_audit(engine, states_multi, n_grid=21)
    assert res_multi.ok.all().item() is True

    # 4. cross_ids as Tensor (Lines 270-271)
    c_ids_tensor = torch.tensor([1, 1], dtype=torch.int64)
    res_c_tensor = cross_slice_audit(engine, states_multi, n_grid=21, cross_ids=c_ids_tensor)
    assert res_c_tensor.calendar_ok.all().item() is True

    # 5. cross_ids as list (Lines 272-273)
    res_c_list = cross_slice_audit(engine, states_multi, n_grid=21, cross_ids=["XAURUB", "XAURUB"])
    assert res_c_list.calendar_ok.all().item() is True

    # 6. Group with len(grp_indices) <= 1 (Lines 285-286: len(grp_indices) <= 1: continue)
    res_c_single_grps = cross_slice_audit(engine, states_multi, n_grid=21, cross_ids=["PAIR_A", "PAIR_B"])
    assert res_c_single_grps.calendar_ok.all().item() is True

    # 7. Group with identical expiries (Lines 289-290: T_grp.max() - T_grp.min() < 1e-6: continue)
    st_same1 = make_state(T=1.0)
    st_same2 = make_state(T=1.0)
    states_same = torch.stack([st_same1, st_same2], dim=0)
    res_same_exp = cross_slice_audit(engine, states_same, n_grid=21, cross_ids=[0, 0])
    assert res_same_exp.calendar_ok.all().item() is True


# ==============================================================================
# SECTION 3: GUARDIAN ZERO-TOLERANCE COVERAGE
# ==============================================================================

def test_market_state_ood_detector_validation_and_methods() -> None:
    """Exercise MarketStateOODDetector __init__, fit, score, and from_checkpoint."""
    mean_12d = torch.zeros(12, dtype=torch.float64)
    cov_inv_12d = torch.eye(12, dtype=torch.float64)
    box_lo = -2.0 * torch.ones(12, dtype=torch.float64)
    box_hi = 2.0 * torch.ones(12, dtype=torch.float64)

    # Wrong mean shape
    with pytest.raises(ValueError, match="Expected mean of shape \\(12,\\)"):
        MarketStateOODDetector(mean=torch.zeros(10), cov_inv=cov_inv_12d, q995=3.5, box_lo=box_lo, box_hi=box_hi)

    # Wrong cov_inv shape
    with pytest.raises(ValueError, match="Expected cov_inv of shape \\(12, 12\\)"):
        MarketStateOODDetector(mean=mean_12d, cov_inv=torch.eye(10), q995=3.5, box_lo=box_lo, box_hi=box_hi)

    # Wrong box shape
    with pytest.raises(ValueError, match="Expected box_lo and box_hi of shape \\(12,\\)"):
        MarketStateOODDetector(mean=mean_12d, cov_inv=cov_inv_12d, q995=3.5, box_lo=torch.zeros(10), box_hi=box_hi)

    # margin == 0.0 branch
    det_no_margin = MarketStateOODDetector(mean=mean_12d, cov_inv=cov_inv_12d, q995=3.5, box_lo=box_lo, box_hi=box_hi, margin=0.0)
    assert det_no_margin.eff_box_lo[0] == -2.0

    # score: 1D input (dim=1)
    m_1d = torch.zeros(12, dtype=torch.float64)
    d_m, box_v, is_ood = det_no_margin.score(m_1d)
    assert d_m.item() == 0.0 and is_ood.item() is False

    # score: wrong feature count
    with pytest.raises(ValueError, match="Expected 12 features per market state"):
        det_no_margin.score(torch.zeros(10))

    # score: box violations (low and high) & mahalanobis violation
    m_lo_viol = -5.0 * torch.ones(1, 12, dtype=torch.float64)
    m_hi_viol = 5.0 * torch.ones(1, 12, dtype=torch.float64)
    assert det_no_margin.score(m_lo_viol)[1].item() is True
    assert det_no_margin.score(m_hi_viol)[1].item() is True

    # fit: S_cnt == 1 branch (shrinkage = 0.1)
    fit_s1 = MarketStateOODDetector.fit(torch.randn(1, 12, dtype=torch.float64))
    assert fit_s1.mean.shape == (12,)

    # fit: S_cnt > 1 branch
    fit_s_multi = MarketStateOODDetector.fit(torch.randn(25, 12, dtype=torch.float64))
    assert fit_s_multi.cov_inv.shape == (12, 12)

    # fit: wrong dimension != 12
    with pytest.raises(ValueError, match="Expected 12-D market states"):
        MarketStateOODDetector.fit(torch.randn(10, 10))

    # from_checkpoint with json
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        card_file = tmp_path / "model_card.json"
        card_content = {
            "ood_statistics": {
                "mean_12d": [0.0] * 12,
                "cov_inv_12d": np.eye(12).tolist(),
                "q995_mahalanobis": 3.85,
                "box_bounds": {
                    "lo": [-2.0] * 12,
                    "hi": [2.0] * 12,
                },
            }
        }
        with open(card_file, "w") as f:
            json.dump(card_content, f)

        det_from_card = MarketStateOODDetector.from_checkpoint(card_file)
        assert det_from_card.q995 == 3.85

        # Directory loading model_card.json
        det_from_dir = MarketStateOODDetector.from_checkpoint(tmp_path)
        assert det_from_dir.q995 == 3.85

        # .pt checkpoint loading
        pt_file = tmp_path / "ckpt.pt"
        torch.save(
            {
                "ood_statistics": {
                    "mean_12d": torch.zeros(12),
                    "cov_inv_12d": torch.eye(12),
                    "q995_mahalanobis": torch.tensor(4.1),
                    "box_lo": -2.0 * torch.ones(12),
                    "box_hi": 2.0 * torch.ones(12),
                }
            },
            pt_file,
        )
        det_from_pt = MarketStateOODDetector.from_checkpoint(pt_file)
        assert abs(det_from_pt.q995 - 4.1) < 1e-6

        # Directory containing residual_vol_v3_m0.pt
        card_file.unlink()
        ckpt_pt = tmp_path / "residual_vol_v3_m0.pt"
        torch.save(
            {
                "ood_statistics": {
                    "mean_12d": torch.zeros(12),
                    "cov_inv_12d": torch.eye(12),
                    "q995_mahalanobis": 3.9,
                    "box_lo": -2.0 * torch.ones(12),
                    "box_hi": 2.0 * torch.ones(12),
                }
            },
            ckpt_pt,
        )
        det_dir_pt = MarketStateOODDetector.from_checkpoint(tmp_path)
        assert det_dir_pt.q995 == 3.9

        # Directory with neither
        ckpt_pt.unlink()
        with pytest.raises(FileNotFoundError, match="No model_card.json or .pt"):
            MarketStateOODDetector.from_checkpoint(tmp_path)

        # .pt without ood_statistics
        bad_pt = tmp_path / "bad.pt"
        torch.save({"dummy": 123}, bad_pt)
        with pytest.raises(KeyError, match="does not contain 'ood_statistics'"):
            MarketStateOODDetector.from_checkpoint(bad_pt)


def test_drift_monitor_branches() -> None:
    """Exercise MarketStateDriftMonitor branches."""
    dm = MarketStateDriftMonitor(window_size=25, psi_alert_threshold=0.25)

    # Empty history compute_psi
    assert dm.compute_psi() == {}
    assert dm.is_drift_detected() is False

    # Bad baseline shape
    with pytest.raises(ValueError, match="Expected baseline of shape \\(N, 12\\)"):
        dm.set_baseline(torch.zeros(10, 10))

    # Set valid baseline
    base = torch.randn(100, 12, dtype=torch.float64)
    dm.set_baseline(base)

    # Record 1D and overflow window_size
    dm.record(torch.randn(12, dtype=torch.float64))
    for _ in range(30):
        dm.record(torch.randn(1, 12, dtype=torch.float64))
    assert len(dm.history) == 25

    # Compute PSI with moderate / critical drift
    # Baseline normal, history shifted heavily by +5.0 to trigger critical drift logs
    shifted = torch.randn(30, 12, dtype=torch.float64) + 5.0
    for row in shifted:
        dm.record(row)
    psi_res = dm.compute_psi(num_bins=5)
    assert len(psi_res) == 12
    assert dm.is_drift_detected() is True


def test_fx_cross_guardian_routing_and_checks() -> None:
    """Exercise FXCrossGuardian routing, NaN weights, epistemic post-checks, and audit integration."""
    guardian = FXCrossGuardian(enable_slice_audit=True)

    # Audit engine property access
    assert isinstance(guardian.audit_engine, nn.Module)

    # Invalid state shapes
    with pytest.raises(ValueError, match="cross_slice_audit requires 34-column"):
        guardian.audit_slice(torch.zeros(1, 12))

    with pytest.raises(ValueError, match="Expected states tensor of shape"):
        guardian.route(torch.zeros(1, 20))

    # Extract 12d state from 34-col tensor
    st_34 = torch.zeros(2, 34, dtype=torch.float64)
    st_34[:, Col.T] = 1.0
    st_34[:, Col.RHO] = 0.2
    st_34[:, Col.XI1 : Col.XI1 + 5] = 0.15
    st_34[:, Col.XI2 : Col.XI2 + 5] = 0.20
    m_12d = guardian.extract_12d_state(st_34)
    assert m_12d.shape == (2, 12)

    # Route: ANALYTIC_TRIANGULATED & ANALYTIC_GK
    dec_ana = guardian.route(st_34, engine=PricingEngineEnum.ANALYTIC_TRIANGULATED)
    assert dec_ana.fallback_activated.all().item() is True
    assert dec_ana.fallback_reason[0] == FallbackReasonEnum.ENGINE_REQUESTED.value

    dec_gk = guardian.route(st_34, engine=PricingEngineEnum.ANALYTIC_GK)
    assert dec_gk.fallback_activated.all().item() is True

    # Route: COPULA_BENCHMARK
    dec_cop = guardian.route(st_34, engine=PricingEngineEnum.COPULA_BENCHMARK)
    assert dec_cop.fallback_activated.any().item() is False
    assert dec_cop.engine_used[0] == PricingEngineEnum.COPULA_BENCHMARK

    # Route: NaN/Inf weights failover
    class DummyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.tensor([float("nan")]))

    dummy_model = DummyNet()
    guardian.model = dummy_model
    dec_nan = guardian.route(m_12d, engine=PricingEngineEnum.DIFFERENTIAL_ML)
    assert dec_nan.fallback_activated.all().item() is True
    assert dec_nan.fallback_reason[0] == "NUMERICAL_INSTABILITY"
    guardian.model = None  # reset

    # Route: In-distribution with drift breached -> HYBRID_ENSEMBLE
    guardian.drift_monitor.psi_alert_threshold = 0.001  # force drift detected
    for _ in range(25):
        guardian.drift_monitor.history.append(np.ones(12))
    guardian.drift_monitor.baseline = np.zeros((25, 12))
    dec_drift = guardian.route(m_12d, engine=PricingEngineEnum.DIFFERENTIAL_ML)
    assert dec_drift.engine_used[0] == PricingEngineEnum.HYBRID_ENSEMBLE
    guardian.drift_monitor.psi_alert_threshold = 0.25  # reset

    # apply_audit_check: perfect vs failing audit
    dec_clean = RoutingDecision(
        nn_gate=torch.ones(2),
        fallback_activated=torch.zeros(2, dtype=torch.bool),
        fallback_reason=[None, None],
        d_m=torch.zeros(2),
        is_ood=torch.zeros(2, dtype=torch.bool),
        box_violation=torch.zeros(2, dtype=torch.bool),
        engine_used=[PricingEngineEnum.DIFFERENTIAL_ML, PricingEngineEnum.DIFFERENTIAL_ML],
    )
    audit_ok = AuditResult(
        min_g=torch.tensor([0.2, 0.2]),
        call_spread_ok=torch.tensor([True, True]),
        slope_bound=torch.tensor([0.4, 0.4]),
        ok=torch.tensor([True, True]),
        calendar_ok=torch.tensor([True, True]),
    )
    # Perfect audit: returns unmodified
    assert guardian.apply_audit_check(dec_clean, audit_ok) is dec_clean

    # Failing audit on state 1: forces OUTPUT_AUDIT_FAILED
    audit_fail = AuditResult(
        min_g=torch.tensor([0.2, -0.05]),
        call_spread_ok=torch.tensor([True, False]),
        slope_bound=torch.tensor([0.4, 2.5]),
        ok=torch.tensor([True, False]),
        calendar_ok=torch.tensor([True, True]),
    )
    dec_after_fail = guardian.apply_audit_check(dec_clean, audit_fail)
    assert dec_after_fail.fallback_activated.tolist() == [False, True]
    assert dec_after_fail.fallback_reason[1] == FallbackReasonEnum.OUTPUT_AUDIT_FAILED.value
    assert dec_after_fail.engine_used[1] == PricingEngineEnum.ANALYTIC_TRIANGULATED

    # Preserves existing OOD reason
    dec_ood = RoutingDecision(
        nn_gate=torch.tensor([1.0, 0.0]),
        fallback_activated=torch.tensor([False, True]),
        fallback_reason=[None, FallbackReasonEnum.OOD_MARKET_REGIME.value],
        d_m=torch.zeros(2),
        is_ood=torch.tensor([False, True]),
        box_violation=torch.zeros(2, dtype=torch.bool),
        engine_used=[PricingEngineEnum.DIFFERENTIAL_ML, PricingEngineEnum.ANALYTIC_TRIANGULATED],
    )
    dec_preserve = guardian.apply_audit_check(dec_ood, audit_fail)
    assert dec_preserve.fallback_reason[1] == FallbackReasonEnum.OOD_MARKET_REGIME.value

    # Post-check: epistemic threshold
    mask_none = guardian.post_check(torch.tensor([0.5]), torch.tensor([10.0]))
    assert mask_none.item() is False
    mask_breached = guardian.post_check(torch.tensor([2.5]), torch.tensor([10.0]))
    assert mask_breached.item() is True

    # apply_post_check
    assert guardian.apply_post_check(dec_clean, torch.tensor([0.5, 0.5]), torch.tensor([10.0, 10.0])) is dec_clean
    dec_post = guardian.apply_post_check(dec_clean, torch.tensor([0.5, 3.5]), torch.tensor([10.0, 10.0]))
    assert dec_post.fallback_activated.tolist() == [False, True]
    assert dec_post.fallback_reason[1] == FallbackReasonEnum.EPISTEMIC_UNCERTAINTY.value


def test_model_reserve_entry_and_tracker() -> None:
    """Exercise ModelReserveEntry __post_init__ and ModelReserveTracker cache methods."""
    now_utc = datetime.now(timezone.utc)
    entry_naive = ModelReserveEntry(reserve_bp=1.5, computed_at=datetime.now())
    assert entry_naive.computed_at_mono > 0.0

    entry_explicit = ModelReserveEntry(reserve_bp=2.0, computed_at=now_utc, computed_at_mono=123.45)
    assert entry_explicit.computed_at_mono == 123.45

    # Mock copula engine
    class MockCopula:
        def __init__(self) -> None:
            self.device = torch.device("cpu")

        def price(self, state: Any, strikes: Tensor, is_call: bool = True) -> Tensor:
            return torch.ones(1, len(strikes), dtype=torch.float64)

    tracker = ModelReserveTracker(
        copula=MockCopula(),  # type: ignore[arg-type]
        max_age=timedelta(seconds=2),
        max_capacity=2,
    )

    # get: cache miss
    val, stale = tracker.get("NONEXISTENT")
    assert val is None and stale is True

    # compute_reserve: D passed explicitly, state with D, without D
    svi_a = RawSVI(torch.tensor([[0.04, 0.08, -0.2, 0.0, 0.1]], dtype=torch.float64))
    svi_b = RawSVI(torch.tensor([[0.06, 0.10, 0.1, 0.0, 0.1]], dtype=torch.float64))
    state = CrossState(
        F1=torch.tensor([2600.0], dtype=torch.float64),
        F2=torch.tensor([95.0], dtype=torch.float64),
        T=torch.tensor([1.0], dtype=torch.float64),
        svi1=svi_a,
        svi2=svi_b,
        rho=torch.tensor([0.25], dtype=torch.float64),
    )
    # served returning object with .V
    class ServedObj:
        def __init__(self, v: Tensor) -> None:
            self.V = v

    r1 = tracker.compute_reserve(state, served=lambda k: ServedObj(torch.ones(1, len(k))), D=torch.tensor([0.85]))
    assert isinstance(r1, float)

    # served returning raw Tensor
    r2 = tracker.compute_reserve(state, served=lambda k: torch.ones(1, len(k)), D=None)
    assert isinstance(r2, float)

    # refresh async: first caller and concurrent deduplication
    async def run_async_test() -> None:
        key = "TEST_KEY"
        await tracker.refresh(key, state, served=lambda k: torch.ones(1, len(k)))
        assert len(tracker) == 1
        res, is_st = tracker.get(key)
        assert res is not None and is_st is False

        # Staleness by state_timestamp
        future_ts = datetime.now(timezone.utc) + timedelta(minutes=5)
        _, is_st_ts = tracker.get(key, state_timestamp=future_ts)
        assert is_st_ts is True

        # Staleness by max_age
        await asyncio.sleep(2.1)
        _, is_st_age = tracker.get(key)
        assert is_st_age is True

        # Concurrent deduplication path
        t1 = tracker.refresh("DEDUP_KEY", state, served=lambda k: torch.ones(1, len(k)))
        t2 = tracker.refresh("DEDUP_KEY", state, served=lambda k: torch.ones(1, len(k)))
        await asyncio.gather(t1, t2)

        # LRU eviction: capacity is 2
        tracker.invalidate()
        assert len(tracker) == 0
        await tracker.refresh("K1", state, served=lambda k: torch.ones(1, len(k)))
        await tracker.refresh("K2", state, served=lambda k: torch.ones(1, len(k)))
        await tracker.refresh("K3", state, served=lambda k: torch.ones(1, len(k)))
        assert len(tracker) <= 2

        # Invalidate specific key
        tracker.invalidate("K3")

    asyncio.run(run_async_test())


def test_guardian_additional_branches() -> None:
    """Cover all remaining lines and branches in fx_cross_guardian.py."""
    # 1. RoutingDecision properties reason and D_M
    rd = RoutingDecision(
        nn_gate=torch.tensor([1.0]),
        fallback_activated=torch.tensor([False]),
        fallback_reason=["TEST_REASON"],
        d_m=torch.tensor([1.23]),
        is_ood=torch.tensor([False]),
        box_violation=torch.tensor([False]),
        engine_used=[PricingEngineEnum.DIFFERENTIAL_ML],
    )
    assert rd.reason == ["TEST_REASON"]
    assert abs(rd.D_M.item() - 1.23) < 1e-5

    # 2. MarketStateDriftMonitor constructor with baseline directly
    base = torch.randn(100, 12, dtype=torch.float64)
    dm_with_base = MarketStateDriftMonitor(baseline=base)
    assert dm_with_base.baseline is not None

    # 3. Moderate drift log: psi_val >= 0.10
    dm_mod = MarketStateDriftMonitor(window_size=100, baseline=torch.randn(100, 12, dtype=torch.float64))
    for row in (torch.randn(100, 12, dtype=torch.float64) + 0.35):
        dm_mod.record(row)
    dm_mod.compute_psi(num_bins=5)

    # 4. Calling guardian.audit_engine twice (lines 347->349)
    guardian = FXCrossGuardian(enable_slice_audit=True)
    eng1 = guardian.audit_engine
    eng2 = guardian.audit_engine
    assert eng1 is eng2

    # 5. Calling guardian.audit_slice with 1D tensor of length 34 (lines 359-360)
    st_1d_34 = torch.zeros(34, dtype=torch.float64)
    st_1d_34[Col.T] = 1.0
    st_1d_34[Col.SVI1] = 0.04
    st_1d_34[Col.SVI1 + 4] = 0.1
    st_1d_34[Col.SVI2] = 0.04
    st_1d_34[Col.SVI2 + 4] = 0.1
    res_1d_audit = guardian.audit_slice(st_1d_34)
    assert res_1d_audit.min_g.shape == (1,)

    # 6. Calling guardian.route with 1D tensor of shape (12,) (lines 421-422)
    st_1d_12 = torch.zeros(12, dtype=torch.float64)
    dec_1d = guardian.route(st_1d_12)
    assert dec_1d.nn_gate.shape == (1,)

    # 7. Calling guardian.route with initialized ood_detector (line 437)
    ood_det = MarketStateOODDetector(
        mean=torch.zeros(12, dtype=torch.float64),
        cov_inv=torch.eye(12, dtype=torch.float64),
        q995=3.5,
        box_lo=-2.0 * torch.ones(12, dtype=torch.float64),
        box_hi=2.0 * torch.ones(12, dtype=torch.float64),
    )
    g_ood = FXCrossGuardian(ood_detector=ood_det, enable_slice_audit=False)
    dec_ood_eval = g_ood.route(st_1d_12)
    assert dec_ood_eval.d_m.shape == (1,)

    # 8. guardian.route: models loop (lines 456-462) & clean differential ML (lines 499-502)
    m_clean = nn.Linear(2, 2)
    m_inf = nn.Linear(2, 2)
    m_inf.weight.data.fill_(float("inf"))
    g_inf = FXCrossGuardian(enable_slice_audit=False)
    g_inf.model = m_clean
    dec_c = g_inf.route(st_1d_12, engine=PricingEngineEnum.DIFFERENTIAL_ML)
    assert dec_c.fallback_activated[0].item() is False
    assert dec_c.engine_used[0] == PricingEngineEnum.DIFFERENTIAL_ML

    g_inf.model = m_inf
    dec_i = g_inf.route(st_1d_12, engine=PricingEngineEnum.DIFFERENTIAL_ML)
    assert dec_i.fallback_activated[0].item() is True

    # 9. states.shape[-1] == 34 where ALL states are OOD (lines 527-530: non_ood_mask has no True)
    g_all_ood = FXCrossGuardian(ood_detector=ood_det, enable_slice_audit=True)
    st_ood_34 = 100.0 * torch.ones(2, 34, dtype=torch.float64)
    dec_all_ood = g_all_ood.route(st_ood_34, engine=PricingEngineEnum.DIFFERENTIAL_ML)
    assert dec_all_ood.fallback_activated.all().item() is True

    # 10. ModelReserveEntry timezone-aware without mono (lines 589->591)
    entry_aware = ModelReserveEntry(reserve_bp=1.0, computed_at=datetime.now(timezone.utc))
    assert entry_aware.computed_at_mono > 0.0

    # 11. Multi-element F, T, sigma_ref in generate_strikes (lines 640-647)
    class DummyMultiState:
        F = torch.tensor([250000.0, 250000.0], dtype=torch.float64)
        T = torch.tensor([1.0, 1.0], dtype=torch.float64)
        sigma_ref = torch.tensor([0.20, 0.20], dtype=torch.float64)

    class MockCopula:
        device = torch.device("cpu")

        def price(self, state: Any, strikes: Tensor, is_call: bool = True) -> Tensor:
            return torch.ones(1, len(strikes), dtype=torch.float64)

    tr = ModelReserveTracker(copula=MockCopula())  # type: ignore[arg-type]
    stks = tr.generate_strikes(DummyMultiState())  # type: ignore[arg-type]
    assert len(stks) == 64

    # 12. state with attribute D (lines 674-676)
    class DummyStateWithD:
        F = torch.tensor([250000.0])
        T = torch.tensor([1.0])
        sigma_ref = torch.tensor([0.20])
        D = torch.tensor([0.85])

    res_d = tr.compute_reserve(DummyStateWithD(), served=lambda k: torch.ones(1, len(k)), D=None)  # type: ignore[arg-type]
    assert isinstance(res_d, float)

    # 13. refresh exception handling (lines 726-728)
    async def run_err() -> None:
        def failing_served(k: Tensor) -> Any:
            raise RuntimeError("pricing error")

        with pytest.raises(RuntimeError):
            await tr.refresh("ERR_K", DummyMultiState(), served=failing_served)  # type: ignore[arg-type]

    asyncio.run(run_err())

    # 14. get state_timestamp comparisons (lines 760-764)
    now_utc = datetime.now(timezone.utc)
    tr._cache["TS_OK"] = ModelReserveEntry(reserve_bp=1.0, computed_at=now_utc, state_timestamp=now_utc, computed_at_mono=time.monotonic())
    _, is_st1 = tr.get("TS_OK", state_timestamp=now_utc - timedelta(seconds=10))
    assert is_st1 is False

    tr._cache["TS_NO_ST"] = ModelReserveEntry(reserve_bp=1.0, computed_at=now_utc - timedelta(seconds=10), state_timestamp=None, computed_at_mono=time.monotonic())
    _, is_st2 = tr.get("TS_NO_ST", state_timestamp=now_utc)
    assert is_st2 is True

    _, is_st3 = tr.get("TS_NO_ST", state_timestamp=now_utc - timedelta(seconds=20))
    assert is_st3 is False

    # 15. Slice audit integration on non-OOD states (lines 528-530)
    guardian_slice_audit = FXCrossGuardian(enable_slice_audit=True)
    st_34_valid = torch.zeros(1, 34, dtype=torch.float64)
    st_34_valid[:, Col.T] = 1.0
    st_34_valid[:, Col.SVI1] = 0.04
    st_34_valid[:, Col.SVI1 + 4] = 0.1
    st_34_valid[:, Col.SVI2] = 0.04
    st_34_valid[:, Col.SVI2 + 4] = 0.1
    dec_slice_audit = guardian_slice_audit.route(st_34_valid, engine=PricingEngineEnum.DIFFERENTIAL_ML)
    assert dec_slice_audit.fallback_reason[0] == FallbackReasonEnum.OUTPUT_AUDIT_FAILED.value

    # 16. Fall through branch for unknown engine (lines 504->510)
    dec_unknown = guardian.route(st_1d_12, engine="CUSTOM_ENGINE")  # type: ignore[arg-type]
    assert dec_unknown.engine_used[0] == "CUSTOM_ENGINE"

    # 17. Exception in surrogate models_to_check import (lines 454-455)
    import sys
    from unittest.mock import patch
    with patch.dict(sys.modules, {"deepvol.models.fx_cross_surrogate": None}):
        dec_imp_err = guardian.route(st_1d_12)
        assert dec_imp_err.engine_used[0] == PricingEngineEnum.DIFFERENTIAL_ML


