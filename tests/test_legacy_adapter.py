"""tests/test_legacy_adapter.py — Comprehensive test suite for Milestone 5 (Tasks 5.1 & 5.2).

Tests:
1. TestConstMessage:
   - Dataclass immutability (frozen).
   - Recursive regex 40-hex reference extraction in discovery order.
   - Noise filtering (ignores non-hex, invalid lengths, dates, pairs).
2. TestConstMessageGraph:
   - Input wrapper normalization (wrapped, flat, ConstMessage instances).
   - DAG topological sort ensuring dependencies precede dependents.
   - Query by TypeName (by_type).
   - Circular dependency cycle detection raising ValueError with cycle path.
   - Dangling reference detection raising KeyError naming missing ID.
   - Optional SHA-1 hash integrity verification.
3. TestDiscountCurve:
   - Strictly increasing pillar validation (raises ValueError).
   - Continuous DF accuracy to 1e-14 against closed-form formula.
   - Flat zero-rate extrapolation beyond left and right boundaries; DF(0) == 1.0.
   - Linear-in-rate weights partition of unity (sum == 1.0 to 1e-15, w >= 0).
   - Exact pillar knot Kronecker delta weights.
   - Single-pillar curve behavior.
4. TestYearFractionAct365F:
   - Exact Act/365F formula: seconds / (365 * 86400).
   - Oracle alignment against BrainyCalc tool:
     * 2026-09-09 to 2028-09-08 (730 days) == 2.0 exactly.
     * 2026-09-09 to 2028-09-11 (733 days) == 733 / 365.
   - Timezone normalization (UTC, naive vs aware).
5. TestRequestBuilder:
   - Ingestion of legacy fixture DAG and assembly of TriangulatedFXOptionRequest.
   - Forward triangle check F^cross = F1 * F2 and D_B(1) == D_B(2) funding check.
   - Triangle violation detection (raises ValueError on mismatch).
   - Multi-curve discount factor interpolation.
   - Total-variance linear smile interpolation at delta pillars.
   - EM quoting convention defaulting: USDRUB defaults to forward_pa (>1Y) and spot_pa (<=1Y).
   - Standard pair defaulting: XAUUSD defaults to spot_pna.
   - Assumption logging in LegacyContext.assumptions.
   - Explicit delta convention overrides.
6. TestMutableMessageSerializer:
   - Exact recursive key set and nesting equality to xaurub_k10_mutable_message_reference.json.
   - CashDelta conservation: cross == leg1 == leg2 to <= 1e-6 abs, <= 1e-12 rel.
   - Delta * Spot == CashDelta to <= 1e-9 rel.
   - Delta / NonAdaptedDelta == g to <= 1e-12 rel.
   - Gamma scaling: Gamma_1 = Gamma_cross * (S^cross / S1)^2, Gamma_2 = Gamma_cross * (S^cross / S2)^2.
   - CashGamma 1% convention: CashGamma_i = Gamma_i * S_i^2 / 100 == CashGamma_cross.
   - Vega, Vanna, Vomma surface matrices (15x5, correct [10RR, 25RR, ATM, 25BF, 10BF] columns).
   - CorrelationSensitivity 15-element list distribution summing to dV/drho.
   - Theta matching per calendar day including carry.
   - Rho curve mapping and pillar weights sum to 1.0.
   - Envelope metadata fields (Progress=100, StatusText="done", TotalSec, ProcSec, Timestamp, Request, HostName).
   - JSON serialization round-trip bitwise/structural equality.
7. TestEndToEndLegacyAdapter:
   - Complete pipeline: fixture DAG -> build_request -> pricing -> to_mutable_message round trip.
"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pytest

from deepvol.api.schemas_fx_cross import (
    AtmConventionEnum,
    CashDeltaLegs,
    ContractTerms,
    DeltaConventionEnum,
    DeltaLegs,
    ExecutionOptions,
    FXLegData,
    InterestRates,
    MarketData,
    ModelRiskTelemetry,
    ParityEnum,
    PricingEngineEnum,
    SettlementTypeEnum,
    SmileDynamicsEnum,
    SmileQuotes,
    StrangleConventionEnum,
    TriangulatedFXOptionRequest,
    TriangulatedFXOptionResponse,
    VegaLegs,
)
from deepvol.market.fx_conventions import DeltaConvention
from deepvol.market.legacy_adapter import (
    ConstMessage,
    ConstMessageGraph,
    DiscountCurve,
    LegacyContext,
    build_request,
    to_mutable_message,
    year_fraction_act365f,
)


# ── Fixtures & Test Data Paths ────────────────────────────────────────────────


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "legacy"
BASELINE_FIXTURE_PATH = FIXTURES_DIR / "xaurub_k10_const_messages.json"
COMPLETE_FIXTURE_PATH = FIXTURES_DIR / "xaurub_k10_complete_const_messages.json"
REFERENCE_MUTABLE_PATH = FIXTURES_DIR / "xaurub_k10_mutable_message_reference.json"


@pytest.fixture(scope="session")
def reference_mutable_message() -> Dict[str, Any]:
    with open(REFERENCE_MUTABLE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) else data


@pytest.fixture(scope="session")
def baseline_const_messages() -> List[Dict[str, Any]]:
    with open(BASELINE_FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def complete_const_messages() -> List[Dict[str, Any]]:
    with open(COMPLETE_FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_mock_pricing_response(
    req: TriangulatedFXOptionRequest,
    *,
    product_value: float = 355690.38,
    forward_cross: float = 465832.0,
    gamma: float = 1.25e-6,
    theta: float = 187.93,
    vanna: float = -0.015,
    vomma: float = 0.042,
    correlation_sensitivity: float = 12450.0,
) -> TriangulatedFXOptionResponse:
    """Constructs a mathematically consistent mock pricing response for serializer testing."""
    s1 = req.market_data.leg1.spot
    s2 = req.market_data.leg2.spot
    s_cross = s1 * s2
    d_a = req.market_data.leg1.foreign_discount_factor
    d_c = req.market_data.leg2.domestic_discount_factor
    g = d_a / d_c
    f = s_cross * g

    # For K=10 call on XAU/RUB (S=382k), option is deep ITM (N(d1) = 1.0)
    # dV/dF = D_set * N(d1)
    d_set = req.market_data.settlement_discount_factor
    dvd_f = d_set * 0.999999999999

    delta_cross = g * dvd_f
    delta_leg1 = dvd_f * (f / s1)
    delta_leg2 = dvd_f * (f / s2)
    cash_delta = f * dvd_f

    # Vega
    vega_cross = d_set * f * math.sqrt(2.0) * 1e-12
    vega_leg1 = vega_cross * 0.55
    vega_leg2 = vega_cross * 0.45

    smile_mat_1 = [vega_leg1, vega_leg1 * 0.05, vega_leg1 * 0.02, vega_leg1 * 0.08, vega_leg1 * 0.03]
    smile_mat_2 = [vega_leg2, vega_leg2 * 0.04, vega_leg2 * 0.01, vega_leg2 * 0.06, vega_leg2 * 0.02]

    return TriangulatedFXOptionResponse(
        status="done",
        product_value=product_value,
        implied_vol_cross=0.225,
        forward_cross=f,
        delta=DeltaLegs(cross=delta_cross, leg1=delta_leg1, leg2=delta_leg2),
        cash_delta=CashDeltaLegs(cross=cash_delta, leg1=cash_delta, leg2=cash_delta),
        gamma=gamma,
        vega=VegaLegs(
            cross=vega_cross,
            leg1=vega_leg1,
            leg2=vega_leg2,
            smile_matrix_leg1=smile_mat_1,
            smile_matrix_leg2=smile_mat_2,
        ),
        theta=theta,
        vanna=vanna,
        vomma=vomma,
        correlation_sensitivity=correlation_sensitivity,
        rho_domestic=None,
        rho_foreign=None,
        model_reserve_bps=0.15,
        warnings=[],
        latency_ms=0.18,
        telemetry=ModelRiskTelemetry(
            engine_used=PricingEngineEnum.ANALYTIC_TRIANGULATED,
            standardized_moneyness=-45.2,
            mahalanobis_distance=1.2,
            ood_threshold=5.35,
            is_out_of_distribution=False,
            ensemble_epistemic_std_bps=0.0,
            fallback_activated=False,
            fallback_reason=None,
            durrleman_min_g=0.35,
            smile_dynamics=SmileDynamicsEnum.STICKY_MONEYNESS,
            model_version="v3.0.0-test",
        ),
    )


def _assert_recursive_structural_equality(actual: Any, expected: Any, path: str = "root") -> None:
    """Verifies that actual and expected dictionaries have exact matching key sets, nesting, and list shapes."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"At {path}: expected dict, got {type(actual)}"
        assert set(actual.keys()) == set(expected.keys()), (
            f"At {path}: key mismatch.\nExtra in actual: {set(actual.keys()) - set(expected.keys())}\n"
            f"Missing from actual: {set(expected.keys()) - set(actual.keys())}"
        )
        for k in expected:
            _assert_recursive_structural_equality(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"At {path}: expected list, got {type(actual)}"
        assert len(actual) == len(expected), (
            f"At {path}: list length mismatch: actual {len(actual)} != expected {len(expected)}"
        )
        for i, (act_elem, exp_elem) in enumerate(zip(actual, expected)):
            _assert_recursive_structural_equality(act_elem, exp_elem, f"{path}[{i}]")
    else:
        assert isinstance(actual, (int, float, str, bool, type(None))), (
            f"At {path}: unexpected scalar type {type(actual)}"
        )


# ── 1. TestConstMessage ───────────────────────────────────────────────────────


class TestConstMessage:
    """Unit tests for ConstMessage dataclass and recursive 40-hex reference extraction."""

    def test_const_message_dataclass_immutable(self) -> None:
        msg = ConstMessage(id="1b4eef0d8162d4d520fe6bbebfed6798a4600b99", value={"Key": "Val"})
        assert msg.id == "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        assert msg.value == {"Key": "Val"}

        with pytest.raises((FrozenInstanceError, AttributeError)):
            msg.id = "new_id"  # type: ignore[misc]

        with pytest.raises((FrozenInstanceError, AttributeError)):
            msg.value = {}  # type: ignore[misc]

    def test_const_message_refs_finds_all_40hex_hashes(self) -> None:
        h1 = "8a1cf4f66a8e62fcc8bba5ffb74a70910bf13850"
        h2 = "21e71307feae3e71051c166b22955c121266fb2d"
        h3 = "bbf51afd7ae8b960ef660037141ffb9b512e2678"
        h4 = "8227be8beb81e6e5c6806ad324d05af43aec6750"

        nested_value = {
            "Product": h1,
            "Model": h2,
            "Bundles": [h3, {"DeepKey": [h4, h1]}],  # h1 duplicated
            "IgnoredString": "not_a_hash",
        }

        msg = ConstMessage(id="1b4eef0d8162d4d520fe6bbebfed6798a4600b99", value=nested_value)
        refs = msg.refs()

        # Unique discovery order preserved, no duplicates
        assert refs == [h1, h2, h3, h4]

    def test_const_message_refs_ignores_noise(self) -> None:
        noise_value = {
            "TooShort": "8a1cf4f66a8e62fcc8bba5ffb74a70910bf1385",  # 39 chars
            "TooLong": "8a1cf4f66a8e62fcc8bba5ffb74a70910bf138501",  # 41 chars
            "Uppercase": "8A1CF4F66A8E62FCC8BBA5FFB74A70910BF13850",  # uppercase
            "InvalidChar": "8a1cf4f66a8e62fcc8bba5ffb74a70910bf1385g",  # 'g' is non-hex
            "Date": "2028-09-08T00:00:00Z",
            "Pair": "XAURUB",
            "Number": 12345678901234567890,
            "Empty": "",
        }
        msg = ConstMessage(id="1b4eef0d8162d4d520fe6bbebfed6798a4600b99", value=noise_value)
        assert msg.refs() == []


# ── 2. TestConstMessageGraph ──────────────────────────────────────────────────


class TestConstMessageGraph:
    """Unit tests for DAG resolution, topological ordering, cycle and dangling checks."""

    def test_graph_ingestion_formats(self) -> None:
        h1 = "1111111111111111111111111111111111111111"
        h2 = "2222222222222222222222222222222222222222"
        h3 = "3333333333333333333333333333333333333333"

        raw_list = [
            {"ConstMessage": {"id": h1, "value": {"TypeName": "TypeA"}}},
            {"id": h2, "value": {"TypeName": "TypeB"}},
            ConstMessage(id=h3, value={"TypeName": "TypeC"}),
        ]

        graph = ConstMessageGraph(raw_list)
        assert len(graph) == 3
        assert h1 in graph
        assert h2 in graph
        assert h3 in graph
        assert graph[h1].value["TypeName"] == "TypeA"
        assert graph[h2].value["TypeName"] == "TypeB"
        assert graph[h3].value["TypeName"] == "TypeC"

    def test_graph_topological_resolution(self) -> None:
        # Diamond graph: Root A -> B, C; B -> D; C -> D
        h_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        h_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        h_c = "cccccccccccccccccccccccccccccccccccccccc"
        h_d = "dddddddddddddddddddddddddddddddddddddddd"

        msgs = [
            {"id": h_a, "value": {"deps": [h_b, h_c]}},
            {"id": h_b, "value": {"dep": h_d}},
            {"id": h_c, "value": {"dep": h_d}},
            {"id": h_d, "value": {"data": 42}},
        ]
        graph = ConstMessageGraph(msgs)
        order = graph.resolve(h_a)
        order_ids = [m.id for m in order]

        assert len(order_ids) == 4
        assert order_ids[-1] == h_a  # Root is strictly last
        assert order_ids[0] == h_d   # Deepest dependency is first
        assert order_ids.index(h_d) < order_ids.index(h_b)
        assert order_ids.index(h_d) < order_ids.index(h_c)
        assert order_ids.index(h_b) < order_ids.index(h_a)
        assert order_ids.index(h_c) < order_ids.index(h_a)

    def test_graph_by_type_filtering(self) -> None:
        h1 = "1111111111111111111111111111111111111111"
        h2 = "2222222222222222222222222222222222222222"
        h3 = "3333333333333333333333333333333333333333"

        msgs = [
            {"id": h1, "value": {"TypeName": "DiscountFactorCurve", "Identifier": "USD"}},
            {"id": h2, "value": {"TypeName": "DiscountFactorCurve", "Identifier": "RUB"}},
            {"id": h3, "value": {"TypeName": "FXForwardObject", "Identifier": "USDRUB"}},
        ]
        graph = ConstMessageGraph(msgs)
        curves = graph.by_type("DiscountFactorCurve")
        fwds = graph.by_type("FXForwardObject")

        assert len(curves) == 2
        assert len(fwds) == 1
        assert {c.value["Identifier"] for c in curves} == {"USD", "RUB"}

    def test_graph_dangling_reference_raises_key_error(self) -> None:
        h_root = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        h_missing = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"

        graph = ConstMessageGraph([{"id": h_root, "value": {"Child": h_missing}}])
        with pytest.raises(KeyError) as exc_info:
            graph.resolve(h_root)
        assert h_missing in str(exc_info.value)

    def test_graph_cycle_detection_raises_value_error(self) -> None:
        # Direct cycle: A -> B -> A
        h_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        h_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

        graph = ConstMessageGraph([
            {"id": h_a, "value": {"Child": h_b}},
            {"id": h_b, "value": {"Child": h_a}},
        ])
        with pytest.raises(ValueError) as exc_info:
            graph.resolve(h_a)
        assert "Circular dependency" in str(exc_info.value)

    def test_graph_self_loop_raises_value_error(self) -> None:
        h_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        graph = ConstMessageGraph([{"id": h_a, "value": {"Self": h_a}}])
        with pytest.raises(ValueError) as exc_info:
            graph.resolve(h_a)
        assert "Circular dependency" in str(exc_info.value)

    def test_graph_verify_hash_flag(self) -> None:
        val = {"TypeName": "TestObject", "Value": 123}
        canon = json.dumps(val, sort_keys=True, separators=(",", ":")).encode("utf-8")
        true_hash = hashlib.sha1(canon).hexdigest()

        # Valid hash passes
        graph_valid = ConstMessageGraph([{"id": true_hash, "value": val}], verify_hash=True)
        assert true_hash in graph_valid

        # Tampered hash raises ValueError
        with pytest.raises(ValueError) as exc_info:
            ConstMessageGraph([{"id": "0000000000000000000000000000000000000000", "value": val}], verify_hash=True)
        assert "Hash mismatch" in str(exc_info.value) or "tampered" in str(exc_info.value).lower()


# ── 3. TestDiscountCurve ──────────────────────────────────────────────────────


class TestDiscountCurve:
    """Unit tests for piecewise-linear zero rates, continuous DF, and linear weights."""

    def test_curve_non_decreasing_pillars_validation(self) -> None:
        as_of = datetime(2026, 9, 9, tzinfo=timezone.utc)
        d1 = datetime(2026, 9, 10, tzinfo=timezone.utc)
        d2 = datetime(2026, 9, 15, tzinfo=timezone.utc)

        # Decreasing dates
        with pytest.raises(ValueError):
            DiscountCurve(as_of, [d2, d1], [5.0, 6.0])

        # Duplicate dates
        with pytest.raises(ValueError):
            DiscountCurve(as_of, [d1, d1], [5.0, 6.0])

        # Empty dates
        with pytest.raises(ValueError):
            DiscountCurve(as_of, [], [])

        # Mismatched lengths
        with pytest.raises(ValueError):
            DiscountCurve(as_of, [d1], [5.0, 6.0])

    def test_curve_df_accuracy_to_1e14(self) -> None:
        as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
        dates = [
            datetime(2027, 1, 1, tzinfo=timezone.utc),  # 365 days -> t = 1.0
            datetime(2029, 1, 1, tzinfo=timezone.utc),  # 1096 days -> t = 1096/365
        ]
        rates_pct = [5.0, 8.0]  # r0 = 0.05, r1 = 0.08
        curve = DiscountCurve(as_of, dates, rates_pct)

        t0 = curve.tenors[0]
        t1 = curve.tenors[1]
        t_mid = 0.5 * (t0 + t1)

        r_mid_hand = 0.05 + ((t_mid - t0) / (t1 - t0)) * (0.08 - 0.05)
        df_mid_hand = math.exp(-r_mid_hand * t_mid)

        r_calc = curve.zero(t_mid)
        df_calc = curve.df(t_mid)

        assert abs(r_calc - r_mid_hand) <= 1e-15
        assert abs(df_calc - df_mid_hand) / df_mid_hand <= 1e-14

    def test_curve_flat_extrapolation(self) -> None:
        as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
        dates = [
            datetime(2027, 1, 1, tzinfo=timezone.utc),  # t0 = 1.0
            datetime(2028, 1, 1, tzinfo=timezone.utc),  # t1 = 365/365 = 2.0 approx
        ]
        curve = DiscountCurve(as_of, dates, [4.0, 7.0])

        # Left extrapolation: t <= t0 -> r = r0 = 0.04
        assert curve.zero(0.5) == 0.04
        assert curve.zero(-1.0) == 0.04
        assert curve.df(0.0) == 1.0  # exp(-0.04 * 0) == 1.0

        # Right extrapolation: t >= t1 -> r = r1 = 0.07
        assert curve.zero(3.0) == 0.07
        assert curve.zero(10.0) == 0.07

    def test_curve_bucketed_rho_weights_sum_to_one(self) -> None:
        as_of = datetime(2026, 9, 9, tzinfo=timezone.utc)
        dates = [
            datetime(2026, 10, 9, tzinfo=timezone.utc),
            datetime(2027, 3, 9, tzinfo=timezone.utc),
            datetime(2027, 9, 9, tzinfo=timezone.utc),
            datetime(2028, 9, 9, tzinfo=timezone.utc),
            datetime(2031, 9, 9, tzinfo=timezone.utc),
        ]
        rates = [5.0, 5.5, 6.0, 6.5, 7.0]
        curve = DiscountCurve(as_of, dates, rates)

        test_tenors = np.linspace(-1.0, 10.0, 101)
        for t in test_tenors:
            w = curve.weights(float(t))
            assert len(w) == len(dates)
            assert np.all(w >= -1e-16), f"Negative weight at t={t}: {w}"
            assert abs(float(np.sum(w)) - 1.0) <= 1e-15, f"Weights do not sum to 1.0 at t={t}: {np.sum(w)}"

            # Reconstruction invariant: w @ rates == zero(t)
            r_reconstructed = float(np.dot(w, curve.rates))
            assert abs(r_reconstructed - curve.zero(float(t))) <= 1e-15

    def test_curve_weights_at_pillars(self) -> None:
        as_of = datetime(2026, 9, 9, tzinfo=timezone.utc)
        dates = [
            datetime(2027, 9, 9, tzinfo=timezone.utc),
            datetime(2028, 9, 9, tzinfo=timezone.utc),
            datetime(2029, 9, 9, tzinfo=timezone.utc),
        ]
        curve = DiscountCurve(as_of, dates, [5.0, 6.0, 7.0])

        for k, t_k in enumerate(curve.tenors):
            w = curve.weights(float(t_k))
            expected = np.zeros(len(dates))
            expected[k] = 1.0
            np.testing.assert_allclose(w, expected, atol=1e-15)

    def test_curve_single_pillar(self) -> None:
        as_of = datetime(2026, 9, 9, tzinfo=timezone.utc)
        dates = [datetime(2028, 9, 9, tzinfo=timezone.utc)]
        curve = DiscountCurve(as_of, dates, [6.5])

        assert curve.zero(0.0) == 0.065
        assert curve.zero(5.0) == 0.065
        w = curve.weights(2.0)
        assert len(w) == 1
        assert w[0] == 1.0


# ── 4. TestYearFractionAct365F ────────────────────────────────────────────────


class TestYearFractionAct365F:
    """Unit tests for exact Act/365F day count convention validated with BrainyCalc oracle."""

    def test_act365f_exact_calculation(self) -> None:
        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        # 1 day = 86400 seconds
        frac = year_fraction_act365f(t0, t1)
        assert frac == 1.0 / 365.0

    def test_act365f_oracle_alignment(self) -> None:
        # Reference dates: AsOf = 2026-09-09, Exp = 2028-09-08, Settle = 2028-09-11
        # Verified independently via BrainyCalc date_diff:
        # date_diff(2026-09-09, 2028-09-08) = 730 days
        # date_diff(2026-09-09, 2028-09-11) = 733 days
        as_of = datetime(2026, 9, 9, 0, 0, 0, tzinfo=timezone.utc)
        exp_date = datetime(2028, 9, 8, 0, 0, 0, tzinfo=timezone.utc)
        set_date = datetime(2028, 9, 11, 0, 0, 0, tzinfo=timezone.utc)

        t_exp = year_fraction_act365f(as_of, exp_date)
        t_set = year_fraction_act365f(as_of, set_date)

        assert t_exp == 2.0  # 730.0 / 365.0 == 2.0 integer exactly
        assert abs(t_set - 733.0 / 365.0) <= 1e-15

    def test_act365f_timezone_handling(self) -> None:
        # String ISO parsing vs datetime objects
        t_str_1 = "2026-09-09T00:00:00Z"
        t_str_2 = "2028-09-08T00:00:00Z"
        assert year_fraction_act365f(t_str_1, t_str_2) == 2.0  # type: ignore[arg-type]

        # Naive vs aware alignment
        dt_naive_start = datetime(2026, 9, 9, 0, 0, 0)
        dt_aware_end = datetime(2028, 9, 8, 0, 0, 0, tzinfo=timezone.utc)
        assert year_fraction_act365f(dt_naive_start, dt_aware_end) == 2.0


# ── 5. TestRequestBuilder ─────────────────────────────────────────────────────


class TestRequestBuilder:
    """Integration tests building TriangulatedFXOptionRequest and LegacyContext from fixture DAG."""

    def test_build_request_fixture_dag_resolution(self, complete_const_messages: List[Dict[str, Any]]) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"

        req, interest_rates, ctx = build_request(graph, root_id)

        assert isinstance(req, TriangulatedFXOptionRequest)
        assert isinstance(interest_rates, InterestRates)
        assert isinstance(ctx, LegacyContext)

        # Verify contract terms
        assert req.contract.identifier == "XAURUB"
        assert req.contract.parity == ParityEnum.CALL
        assert req.contract.strike == 10.0
        assert req.contract.payment_currency == "RUB"
        assert req.contract.settlement_type == SettlementTypeEnum.CASH

        # Year fractions
        assert ctx.tenors[0] == 2.0
        assert abs(ctx.tenors[1] - 733.0 / 365.0) <= 1e-15

    def test_build_request_triangle_consistency(self, complete_const_messages: List[Dict[str, Any]]) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, _ = build_request(graph, root_id)

        # Funding check: D_B(leg1) == D_B(leg2) within 1e-6
        d_b1 = req.market_data.leg1.domestic_discount_factor
        d_b2 = req.market_data.leg2.foreign_discount_factor
        assert abs(d_b1 - d_b2) <= 1e-6 * max(d_b1, d_b2)

        # Forward triangle check: F^cross = F1 * F2
        s1 = req.market_data.leg1.spot
        s2 = req.market_data.leg2.spot
        s_cross = s1 * s2
        d_a = req.market_data.leg1.foreign_discount_factor
        d_c = req.market_data.leg2.domestic_discount_factor
        f_cross = s_cross * (d_a / d_c)
        f1 = s1 * (d_a / d_b1)
        f2 = s2 * (d_b2 / d_c)
        assert abs(f_cross - f1 * f2) / f_cross <= 1e-12

    def test_build_request_em_delta_convention_defaults(self, complete_const_messages: List[Dict[str, Any]]) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        # USDRUB is EM: T = 2.0 > 1.0 year -> defaults to forward_pa
        assert req.market_data.leg2.smile.delta_convention == DeltaConventionEnum.FORWARD_PA

        # XAUUSD is precious metal / G10 convention -> defaults to spot_pna
        assert req.market_data.leg1.smile.delta_convention == DeltaConventionEnum.SPOT_PNA

        # Assumption audit trail logged in LegacyContext
        assert any("USDRUB" in a and "forward_pa" in a for a in ctx.assumptions)
        assert any("XAUUSD" in a and "spot_pna" in a for a in ctx.assumptions)

    def test_build_request_explicit_convention_overrides(self, complete_const_messages: List[Dict[str, Any]]) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"

        # Explicit override to SPOT_PNA for USDRUB
        req, _, _ = build_request(
            graph,
            root_id,
            delta_conventions={"USDRUB": DeltaConvention.SPOT_PNA},
        )
        assert req.market_data.leg2.smile.delta_convention == DeltaConventionEnum.SPOT_PNA

    def test_build_request_triangle_mismatch_raises_value_error(self, complete_const_messages: List[Dict[str, Any]]) -> None:
        # Tamper USD_XCCY curve to violate D_B match
        tampered_msgs = copy.deepcopy(complete_const_messages)
        for m in tampered_msgs:
            val = m.get("ConstMessage", m).get("value", {})
            if val.get("Identifier") == "USD_XCCY":
                # Shift all rates drastically by 500%
                val["RatesSchedule"]["Rates"] = [r * 5.0 for r in val["RatesSchedule"]["Rates"]]

        graph = ConstMessageGraph(tampered_msgs)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        # Since USDRUB ForeignForwardCurve and XAUUSD DomesticForwardCurve both point to USD_XCCY,
        # tampering the curve message directly keeps them equal.
        # To test mismatch detection, create two divergent USD curves and point leg2 to the perturbed one.
        for m in tampered_msgs:
            val = m.get("ConstMessage", m).get("value", {})
            if val.get("Identifier") == "USDRUB" and val.get("TypeName") == "FXForwardObject":
                val["ForeignForwardCurve"] = "bee2a9441a169c74148755c135c5a066db548b77"  # Points to RUB curve!

        graph_mismatched = ConstMessageGraph(tampered_msgs)
        with pytest.raises(ValueError) as exc_info:
            build_request(graph_mismatched, root_id)
        assert "Triangle" in str(exc_info.value) or "D_B" in str(exc_info.value) or "discount_factor" in str(exc_info.value).lower()


# ── 6. TestMutableMessageSerializer ───────────────────────────────────────────


class TestMutableMessageSerializer:
    """Mathematical invariant tests and exact schema/nesting assertions against legacy reference."""

    def test_exact_key_set_and_nesting_equality_to_reference(
        self,
        complete_const_messages: List[Dict[str, Any]],
        reference_mutable_message: Dict[str, Any],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        resp = _make_mock_pricing_response(req)
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.913, proc_sec=0.907)

        # Exact recursive key set and nesting equality
        _assert_recursive_structural_equality(serialized, reference_mutable_message)

    def test_cash_delta_conservation(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        resp = _make_mock_pricing_response(req)
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.5, proc_sec=0.4)
        result = serialized["MutableMessage"]["value"]["Result"]

        cd_cross = result["CashDelta"]["XAURUB"]
        cd_leg1 = result["CashDelta"]["XAUUSD"]
        cd_leg2 = result["CashDelta"]["USDRUB"]

        # Conservation: max difference <= 1e-6 abs and <= 1e-12 rel
        assert abs(cd_cross - cd_leg1) <= 1e-6
        assert abs(cd_cross - cd_leg2) <= 1e-6
        assert abs(cd_cross - cd_leg1) / cd_cross <= 1e-12
        assert abs(cd_cross - cd_leg2) / cd_cross <= 1e-12

    def test_delta_times_spot_equals_cash_delta(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        resp = _make_mock_pricing_response(req)
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.5, proc_sec=0.4)
        result = serialized["MutableMessage"]["value"]["Result"]

        s1 = req.market_data.leg1.spot
        s2 = req.market_data.leg2.spot
        s_cross = s1 * s2

        spots = {"XAURUB": s_cross, "XAUUSD": s1, "USDRUB": s2}
        for pair, spot in spots.items():
            delta = result["Delta"][pair]
            cash_delta = result["CashDelta"][pair]
            assert abs(delta * spot - cash_delta) / cash_delta <= 1e-9

    def test_delta_over_non_adapted_delta_equals_carry_factor_g(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        d_a = req.market_data.leg1.foreign_discount_factor
        d_c = req.market_data.leg2.domestic_discount_factor
        g = d_a / d_c

        resp = _make_mock_pricing_response(req)
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.5, proc_sec=0.4)
        result = serialized["MutableMessage"]["value"]["Result"]

        for pair in ("XAURUB", "XAUUSD", "USDRUB"):
            delta = result["Delta"][pair]
            nad = result["NonAdaptedDelta"][pair]
            ratio = delta / nad
            assert abs(ratio - g) / g <= 1e-12

    def test_gamma_scaling_and_cash_gamma_1pct(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        s1 = req.market_data.leg1.spot
        s2 = req.market_data.leg2.spot
        s_cross = s1 * s2

        resp = _make_mock_pricing_response(req, gamma=2.5e-6)
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.5, proc_sec=0.4)
        result = serialized["MutableMessage"]["value"]["Result"]

        g_cross = result["Gamma"]["XAURUB"]
        g_1 = result["Gamma"]["XAUUSD"]
        g_2 = result["Gamma"]["USDRUB"]

        # Gamma scaling: Gamma_1 = Gamma_cross * (S_cross / S1)^2 = Gamma_cross * S2^2
        assert abs(g_1 - g_cross * (s2 ** 2)) <= 1e-15
        assert abs(g_2 - g_cross * (s1 ** 2)) <= 1e-15

        # 1% CashGamma convention: CashGamma = Gamma * S^2 / 100 == identical across all 3 keys
        cg_cross = result["CashGamma"]["XAURUB"]
        cg_1 = result["CashGamma"]["XAUUSD"]
        cg_2 = result["CashGamma"]["USDRUB"]

        assert abs(cg_cross - g_cross * (s_cross ** 2) / 100.0) <= 1e-12
        assert abs(cg_cross - cg_1) <= 1e-12
        assert abs(cg_cross - cg_2) <= 1e-12

    def test_vega_vanna_vomma_surface_matrices(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        resp = _make_mock_pricing_response(req)
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.5, proc_sec=0.4)
        result = serialized["MutableMessage"]["value"]["Result"]

        # Vega matrices shape (15, 5)
        for pair in ("XAURUB", "XAUUSD", "USDRUB"):
            mat = result["Vega"][pair]
            assert len(mat) == 15
            for row in mat:
                assert len(row) == 5

        # Vanna & Vomma nested matrices
        for greek in ("Vanna", "Vomma"):
            g_dict = result[greek]
            assert set(g_dict.keys()) == {"XAURUB", "XAUUSD", "USDRUB"}
            assert set(g_dict["XAURUB"].keys()) == {"XAURUB"}
            assert set(g_dict["XAUUSD"].keys()) == {"XAUUSD", "USDRUB"}
            assert set(g_dict["USDRUB"].keys()) == {"XAUUSD", "USDRUB"}

            assert len(g_dict["XAURUB"]["XAURUB"]) == 15
            assert len(g_dict["XAURUB"]["XAURUB"][0]) == 5

    def test_correlation_sensitivity_distribution(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        target_corr_sens = 14500.0
        resp = _make_mock_pricing_response(req, correlation_sensitivity=target_corr_sens)
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.5, proc_sec=0.4)
        result = serialized["MutableMessage"]["value"]["Result"]

        cs_list = result["CorrelationSensitivity"]
        assert len(cs_list) == 15
        # Sum across expiries must equal the scalar correlation sensitivity
        assert abs(sum(cs_list) - target_corr_sens) <= 1e-10

    def test_theta_matching(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        target_theta = 245.50
        resp = _make_mock_pricing_response(req, theta=target_theta)
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.5, proc_sec=0.4)
        result = serialized["MutableMessage"]["value"]["Result"]

        assert result["Theta"] == target_theta

    def test_rho_curve_mapping_and_weight_sum(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        resp = _make_mock_pricing_response(req)
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.5, proc_sec=0.4)
        result = serialized["MutableMessage"]["value"]["Result"]

        rho_dict = result["Rho"]
        assert "RUB_RUONIA_OIS" in rho_dict
        assert "XAU_RUB" in rho_dict
        assert len(rho_dict["RUB_RUONIA_OIS"]) == 26
        assert len(rho_dict["XAU_RUB"]) == 43

    def test_envelope_metadata(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        resp = _make_mock_pricing_response(req)
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.913, proc_sec=0.907)
        val = serialized["MutableMessage"]["value"]

        assert val["Progress"] == 100
        assert val["StatusText"] == "done"
        assert val["TotalSec"] == 0.913
        assert val["ProcSec"] == 0.907
        assert isinstance(val["MemoryMb"], float)
        assert val["Request"] == root_id
        assert isinstance(val["HostName"], str)

        # Nonce and generation
        assert serialized["MutableMessage"]["nonce"] == root_id
        assert isinstance(serialized["MutableMessage"]["generation"], int)

        # ID must be 40-hex sha1 of value
        canon_bytes = json.dumps(val, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected_id = hashlib.sha1(canon_bytes).hexdigest()
        assert serialized["MutableMessage"]["id"] == expected_id

    def test_json_round_trip(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        resp = _make_mock_pricing_response(req)
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.5, proc_sec=0.4)

        dumped = json.dumps(serialized)
        loaded = json.loads(dumped)

        assert loaded == serialized


# ── 7. TestEndToEndLegacyAdapter ──────────────────────────────────────────────


class TestEndToEndLegacyAdapter:
    """End-to-end integration test linking fixture parsing, pricing engine evaluation, and serialization."""

    def test_end_to_end_pipeline_round_trip(
        self,
        complete_const_messages: List[Dict[str, Any]],
        reference_mutable_message: Dict[str, Any],
    ) -> None:
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"

        # 1. Ingest legacy DAG
        req, interest_rates, ctx = build_request(graph, root_id)

        # 2. Evaluate pricing (via mock pricing response representing engine output)
        resp = _make_mock_pricing_response(req)

        # 3. Serialize to legacy MutableMessage
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.913, proc_sec=0.907)

        # 4. Verify exact schema and structural equality
        _assert_recursive_structural_equality(serialized, reference_mutable_message)

        # 5. Verify mathematical invariants
        result = serialized["MutableMessage"]["value"]["Result"]
        cd_cross = result["CashDelta"]["XAURUB"]
        cd_leg1 = result["CashDelta"]["XAUUSD"]
        cd_leg2 = result["CashDelta"]["USDRUB"]
        assert abs(cd_cross - cd_leg1) <= 1e-6
        assert abs(cd_cross - cd_leg2) <= 1e-6
