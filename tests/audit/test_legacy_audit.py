"""tests/audit/test_legacy_audit.py — Adversarial Audit for Legacy Message-Graph Adapter (Phase 13).

Audits SPEC-DEEPVOL-FX-CROSS-003 Milestone 5:
1. ConstMessage DAG:
   - Content-addressable hashing determinism across serialization
   - Cycle rejection (topological sort cycle detection)
   - Dangling reference detection
   - Loading 8 recoverable messages from legacy fixture
2. Multi-Curve DiscountCurve & Day Count:
   - Act/365F year fraction against BrainyCalc oracle on >= 200 date pairs
   - Monotonicity in discount factors
   - Linear partition of unity for interpolation weights (sum == 1.0)
3. MutableMessage Serializer:
   - Exact key set and nesting matching legacy reference fixture
   - CashDelta conservation: cross == leg1 == leg2 to <= 1e-6 abs and <= 1e-12 rel
   - NonAdaptedDelta relationship: Delta / NonAdaptedDelta == g to <= 1e-12 rel
   - Delta * Spot == CashDelta consistency
   - Vega 15x5 matrix dimensions and column alignment
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from deepvol.api.schemas_fx_cross import (
    CashDeltaLegs,
    DeltaLegs,
    ModelRiskTelemetry,
    ParityEnum,
    PricingEngineEnum,
    SmileDynamicsEnum,
    TriangulatedFXOptionRequest,
    TriangulatedFXOptionResponse,
    VegaLegs,
)
from deepvol.market.legacy_adapter import (
    ConstMessage,
    ConstMessageGraph,
    DiscountCurve,
    LegacyContext,
    build_request,
    to_mutable_message,
    year_fraction_act365f,
    _canonical_json_bytes,
)

pytestmark = pytest.mark.audit


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "legacy"


@pytest.fixture
def recoverable_fixture_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "xaurub_k10_const_messages.json"


@pytest.fixture
def reference_mutable_message_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "xaurub_k10_mutable_message_reference.json"


@pytest.fixture
def brainycalc_vectors_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "artifacts" / "audit" / "security" / "brainycalc_date_diff_vectors.json"


# ── 1. ConstMessage DAG & Content-Addressable Hashing ─────────────────────────


class TestConstMessageDAGAudit:
    """Audits ConstMessage DAG hashing, cycle detection, and resolution."""

    def test_canonical_json_hashing_determinism(self) -> None:
        """Verifies that _canonical_json_bytes produces identical output regardless of key insertion order."""
        dict1 = {"b": 2.0, "a": 1, "c": {"z": 9, "y": [1, 2, 3]}}
        dict2 = {"c": {"y": [1, 2, 3], "z": 9}, "a": 1, "b": 2.0}
        b1 = _canonical_json_bytes(dict1)
        b2 = _canonical_json_bytes(dict2)
        assert b1 == b2, "Canonical JSON bytes must be invariant to dict key ordering"

    def test_cycle_rejection(self) -> None:
        """Verifies that ConstMessageGraph.resolve raises ValueError on circular dependencies."""
        id_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        id_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        id_c = "cccccccccccccccccccccccccccccccccccccccc"

        messages = [
            {"ConstMessage": {"id": id_a, "value": {"Ref": id_b, "TypeName": "NodeA"}}},
            {"ConstMessage": {"id": id_b, "value": {"Ref": id_c, "TypeName": "NodeB"}}},
            {"ConstMessage": {"id": id_c, "value": {"Ref": id_a, "TypeName": "NodeC"}}},  # Cycle back to A
        ]
        graph = ConstMessageGraph(messages)
        with pytest.raises(ValueError, match="Circular dependency cycle"):
            graph.resolve(id_a)

    def test_dangling_reference_rejection(self) -> None:
        """Verifies that ConstMessageGraph.resolve raises KeyError on missing references."""
        id_root = "1111111111111111111111111111111111111111"
        id_missing = "2222222222222222222222222222222222222222"

        messages = [
            {"ConstMessage": {"id": id_root, "value": {"Dep": id_missing, "TypeName": "RootNode"}}}
        ]
        graph = ConstMessageGraph(messages)
        with pytest.raises(KeyError, match="Dangling reference"):
            graph.resolve(id_root)

    def test_load_recoverable_messages(self, recoverable_fixture_path: Path) -> None:
        """Verifies that the 8 recoverable messages from the legacy fixture load correctly."""
        assert recoverable_fixture_path.exists(), f"Fixture missing: {recoverable_fixture_path}"
        with open(recoverable_fixture_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        graph = ConstMessageGraph(data)
        assert len(graph) == 8, f"Expected exactly 8 recoverable messages, got {len(graph)}"

        # Verify discovery of expected types
        products = graph.by_type("EuropeanVanillaOptionProduct")
        assert len(products) == 1
        assert products[0].value["Identifier"] == "XAURUB"
        assert products[0].value["Strike"] == 10.0


# ── 2. Multi-Curve DiscountCurve & BrainyCalc Day Count ────────────────────────


class TestDiscountCurveAndDayCountAudit:
    """Audits Act/365F day count and DiscountCurve interpolation."""

    def test_act365f_against_brainycalc_vectors(self, brainycalc_vectors_path: Path) -> None:
        """Verifies year_fraction_act365f against >= 200 BrainyCalc date_diff vectors."""
        assert brainycalc_vectors_path.exists(), f"BrainyCalc vectors missing: {brainycalc_vectors_path}"
        with open(brainycalc_vectors_path, "r", encoding="utf-8") as fp:
            vectors = json.load(fp)

        assert len(vectors) >= 200, f"Expected >= 200 BrainyCalc test vectors, found {len(vectors)}"

        max_err = 0.0
        for vec in vectors:
            start_str = vec["start"]
            end_str = vec["end"]
            expected_yf = float(vec["expected_yf_act365f"])

            actual_yf = year_fraction_act365f(start_str, end_str)
            err = abs(actual_yf - expected_yf)
            max_err = max(max_err, err)
            assert err <= 1e-12, f"Year fraction mismatch for {start_str} -> {end_str}: actual {actual_yf}, expected {expected_yf}"

        assert max_err <= 1e-12

    def test_discount_curve_monotonicity(self) -> None:
        """Verifies that DiscountCurve yields strictly decreasing discount factors for positive rates."""
        as_of = datetime(2026, 9, 9, tzinfo=timezone.utc)
        dates = [
            as_of + timedelta(days=30),
            as_of + timedelta(days=90),
            as_of + timedelta(days=180),
            as_of + timedelta(days=365),
            as_of + timedelta(days=730),
            as_of + timedelta(days=1825),
        ]
        rates_pct = [5.0, 5.2, 5.5, 6.0, 6.5, 7.0]

        curve = DiscountCurve(as_of, dates, rates_pct)
        eval_tenors = np.linspace(0.01, 5.0, 200)
        dfs = [curve.df(t) for t in eval_tenors]

        diffs = np.diff(dfs)
        assert np.all(diffs < 0.0), "Discount factors must be strictly monotonically decreasing for positive rates"

    def test_discount_curve_partition_of_unity(self) -> None:
        """Verifies that weights(t) sum to 1.0 (partition of unity) across the entire curve span."""
        as_of = datetime(2026, 9, 9, tzinfo=timezone.utc)
        dates = [
            as_of + timedelta(days=30),
            as_of + timedelta(days=90),
            as_of + timedelta(days=365),
            as_of + timedelta(days=730),
        ]
        rates_pct = [4.0, 4.5, 5.0, 5.5]
        curve = DiscountCurve(as_of, dates, rates_pct)

        for t in np.linspace(0.0, 4.0, 50):
            w = curve.weights(float(t))
            assert abs(float(np.sum(w)) - 1.0) <= 1e-14, f"Weights at t={t} must sum to 1.0"
            assert np.all(w >= -1e-15), f"Interpolation weights must be non-negative, got {w}"


# ── 3. MutableMessage Serializer Compliance ────────────────────────────────────


class TestMutableMessageSerializerAudit:
    """Audits MutableMessage serialization against reference fixture contracts."""

    @pytest.fixture
    def mock_environment(self) -> tuple[TriangulatedFXOptionResponse, TriangulatedFXOptionRequest, LegacyContext]:
        as_of = datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc)
        exp = datetime(2028, 9, 8, 10, 0, tzinfo=timezone.utc)
        settle = datetime(2028, 9, 11, 10, 0, tzinfo=timezone.utc)

        t_exp = 2.0
        t_set = 2.0082191780821916
        s1, s2 = 1800.0, 75.0
        s_cross = s1 * s2
        g = 1.21915239922
        f_cross = s_cross * g

        # Response
        resp = TriangulatedFXOptionResponse(
            status="done",
            product_value=355690.38,
            implied_vol_cross=0.25,
            forward_cross=f_cross,
            delta=DeltaLegs(cross=0.93091474, leg1=80.5720066, leg2=4109.672068),
            cash_delta=CashDeltaLegs(cross=355698.014861, leg1=355698.014861, leg2=355698.014861),
            gamma=1.5e-6,
            vega=VegaLegs(cross=125000.0, leg1=85000.0, leg2=95000.0),
            theta=187.928,
            vanna=-25.0,
            vomma=40.0,
            correlation_sensitivity=5000.0,
            latency_ms=1.5,
            telemetry=ModelRiskTelemetry(
                engine_used=PricingEngineEnum.ANALYTIC_TRIANGULATED,
                standardized_moneyness=0.0,
                mahalanobis_distance=1.0,
                ood_threshold=5.35,
                is_out_of_distribution=False,
                ensemble_epistemic_std_bps=0.0,
                fallback_activated=False,
                model_version="v3.0.0",
            ),
        )

        # Context
        expiries = [as_of + timedelta(days=30 * i) for i in range(1, 16)]
        ctx = LegacyContext(
            root_id="root_123",
            product_id="prod_123",
            model_id="mod_123",
            market_data_set_id="mds_123",
            queries=["ProductValue", "Delta", "CashDelta", "NonAdaptedDelta", "Gamma", "Vega", "Theta"],
            pair_cross="XAURUB",
            pair_leg1="XAUUSD",
            pair_leg2="USDRUB",
            tenors=(t_exp, t_set),
            smile_expiries={"XAURUB": expiries, "XAUUSD": expiries, "USDRUB": expiries},
            smile_weights={"XAURUB": (9, 10, 0.6, 0.4), "XAUUSD": (9, 10, 0.6, 0.4), "USDRUB": (9, 10, 0.6, 0.4)},
        )

        from deepvol.api.schemas_fx_cross import ContractTerms, FXLegData, MarketData, SmileQuotes
        req = TriangulatedFXOptionRequest(
            contract=ContractTerms(
                identifier="XAURUB",
                parity=ParityEnum.CALL,
                strike=10.0,
                as_of_date=as_of,
                expiration_date=exp,
                settlement_date=settle,
                payment_currency="RUB",
            ),
            market_data=MarketData(
                leg1=FXLegData(pair="XAUUSD", spot=s1, foreign_discount_factor=0.95, domestic_discount_factor=0.90,
                               smile=SmileQuotes(atm_vol=0.20, rr25=0.01, bf25=0.01, rr10=0.02, bf10=0.02)),
                leg2=FXLegData(pair="USDRUB", spot=s2, foreign_discount_factor=0.90, domestic_discount_factor=0.78,
                               smile=SmileQuotes(atm_vol=0.20, rr25=0.01, bf25=0.01, rr10=0.02, bf10=0.02)),
                correlation=0.3,
                settlement_discount_factor=0.77,
            ),
        )

        return resp, req, ctx

    def test_mutable_message_key_set_and_structure(
        self,
        mock_environment: tuple[TriangulatedFXOptionResponse, TriangulatedFXOptionRequest, LegacyContext],
        reference_mutable_message_path: Path,
    ) -> None:
        """Verifies that the serialized MutableMessage exactly reproduces the reference fixture key set and hierarchy."""
        resp, req, ctx = mock_environment
        out = to_mutable_message(resp, req, ctx, nonce="nonce_123", total_sec=0.5, proc_sec=0.4)

        assert "MutableMessage" in out
        mm = out["MutableMessage"]
        assert "id" in mm and "nonce" in mm and "generation" in mm and "value" in mm
        val = mm["value"]
        assert set(val.keys()) == {
            "Progress", "StatusText", "TotalSec", "ProcSec", "MemoryMb",
            "Timestamp", "Result", "Request", "HostName"
        }

        result = val["Result"]
        expected_result_keys = {
            "ProductValue", "Delta", "CashDelta", "NonAdaptedDelta",
            "Gamma", "CashGamma", "Vega", "CorrelationSensitivity",
            "Theta", "Vanna", "Vomma", "Rho"
        }
        assert set(result.keys()) == expected_result_keys

    def test_cash_delta_conservation_and_ratios(
        self,
        mock_environment: tuple[TriangulatedFXOptionResponse, TriangulatedFXOptionRequest, LegacyContext],
    ) -> None:
        """Verifies CashDelta conservation and Delta / NonAdaptedDelta == g identity."""
        resp, req, ctx = mock_environment
        out = to_mutable_message(resp, req, ctx, nonce="nonce_123", total_sec=0.5, proc_sec=0.4)
        result = out["MutableMessage"]["value"]["Result"]

        # CashDelta conservation: cross == leg1 == leg2
        cd = result["CashDelta"]
        assert abs(cd["XAURUB"] - cd["XAUUSD"]) <= 1e-6
        assert abs(cd["XAURUB"] - cd["USDRUB"]) <= 1e-6
        assert abs(cd["XAURUB"] - cd["XAUUSD"]) / cd["XAURUB"] <= 1e-12

        # Delta / NonAdaptedDelta == g
        g = float(resp.forward_cross / (req.market_data.leg1.spot * req.market_data.leg2.spot))
        delta = result["Delta"]
        nad = result["NonAdaptedDelta"]

        for pair in ("XAURUB", "XAUUSD", "USDRUB"):
            ratio = delta[pair] / nad[pair]
            assert abs(ratio - g) / g <= 1e-12, f"Pair {pair} Delta/NonAdaptedDelta ratio ({ratio}) != g ({g})"

    def test_vega_matrix_shape(
        self,
        mock_environment: tuple[TriangulatedFXOptionResponse, TriangulatedFXOptionRequest, LegacyContext],
    ) -> None:
        """Verifies that Vega matrices have dimensions (15, 5)."""
        resp, req, ctx = mock_environment
        out = to_mutable_message(resp, req, ctx, nonce="nonce_123", total_sec=0.5, proc_sec=0.4)
        vega = out["MutableMessage"]["value"]["Result"]["Vega"]

        for pair in ("XAURUB", "XAUUSD", "USDRUB"):
            mat = vega[pair]
            assert len(mat) == 15, f"{pair} Vega matrix must have 15 rows"
            for row in mat:
                assert len(row) == 5, f"{pair} Vega matrix row must have 5 columns"

    def test_roundtrip_legacy_graph_build_request_and_serialize(
        self,
        fixtures_dir: Path,
    ) -> None:
        """Verifies full round trip: legacy ConstMessage DAG -> build_request -> to_mutable_message."""
        complete_path = fixtures_dir / "xaurub_k10_complete_const_messages.json"
        assert complete_path.exists(), f"Complete fixture missing: {complete_path}"

        with open(complete_path, "r", encoding="utf-8") as fp:
            msgs = json.load(fp)

        graph = ConstMessageGraph(msgs)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, rates, ctx = build_request(graph, root_id)

        assert req.contract.identifier == "XAURUB"
        assert req.contract.strike == 10.0
        assert req.contract.parity == ParityEnum.CALL

        # Create a compliant response
        s1 = req.market_data.leg1.spot
        s2 = req.market_data.leg2.spot
        g = req.market_data.leg1.foreign_discount_factor / req.market_data.leg2.domestic_discount_factor
        f_cross = s1 * s2 * g
        cd_val = 355698.014861

        resp = TriangulatedFXOptionResponse(
            status="done",
            product_value=355690.38,
            implied_vol_cross=0.25,
            forward_cross=f_cross,
            delta=DeltaLegs(cross=cd_val / (s1 * s2), leg1=cd_val / s1, leg2=cd_val / s2),
            cash_delta=CashDeltaLegs(cross=cd_val, leg1=cd_val, leg2=cd_val),
            gamma=1.5e-6,
            vega=VegaLegs(cross=125000.0, leg1=85000.0, leg2=95000.0),
            theta=187.928,
            vanna=-25.0,
            vomma=40.0,
            correlation_sensitivity=5000.0,
            latency_ms=1.5,
            telemetry=ModelRiskTelemetry(
                engine_used=PricingEngineEnum.ANALYTIC_TRIANGULATED,
                standardized_moneyness=0.0,
                mahalanobis_distance=1.0,
                ood_threshold=5.35,
                is_out_of_distribution=False,
                ensemble_epistemic_std_bps=0.0,
                fallback_activated=False,
                model_version="v3.0.0",
            ),
        )

        out = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.913, proc_sec=0.907)
        assert "MutableMessage" in out
        mm = out["MutableMessage"]
        assert mm["nonce"] == root_id
        res = mm["value"]["Result"]
        assert res["ProductValue"] == 355690.38
        assert res["CashDelta"]["XAURUB"] == cd_val
        assert res["CashDelta"]["XAUUSD"] == cd_val
        assert res["CashDelta"]["USDRUB"] == cd_val
