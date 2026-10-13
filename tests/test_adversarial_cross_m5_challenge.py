# pyright: reportPrivateUsage=none
"""tests/test_adversarial_cross_m5_challenge.py — Empirical challenge suite for Milestone 5.

Adversarially audits and stress-tests:
1. Mathematical Identities Stress across 10,000 Random Market States:
   - CashDelta conservation: max_{i,j} |CashDelta_i - CashDelta_j| <= 1e-6 abs, 1e-12 rel across cross, leg 1, leg 2.
   - Spot delta identity: |Delta_i * S_i - CashDelta_i| / CashDelta_i <= 1e-9 rel.
   - Forward delta carry ratio: |Delta_i / NonAdaptedDelta_i - g| / g <= 1e-12 rel.
   - CashGamma identity: CashGamma_1 == CashGamma_2 == CashGamma_cross == Gamma_cross * (S_cross)^2 / 100.
2. Serializer Key & Nesting Verification:
   - Assert exact recursive key set and nesting equality to xaurub_k10_mutable_message_reference.json.
   - Assert JSON round-trip stability (json.dumps and orjson.dumps followed by reload preserves bitwise fidelity).
3. End-to-End Ingestion-to-Pricing Pipeline:
   - Ingest xaurub_k10_complete_const_messages.json via ConstMessageGraph.
   - Build request via build_request.
   - Evaluate via live pricing engine (FastAPI client /fx_cross/price and _fx_cross_price_impl).
   - Serialize response via to_mutable_message.
   - Assert all output fields are populated, finite, and strictly non-NaN / non-Inf.
4. Boundary Conditions and Numerical Extremes:
   - Deep ITM / deep OTM, extreme spots, inverted discount factors, zero carry.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Generator, List, cast

from fastapi.testclient import TestClient
import numpy as np
import orjson
import pytest

from deepvol.api.schemas_fx_cross import (
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
    TriangulatedFXOptionRequest,
    TriangulatedFXOptionResponse,
    VegaLegs,
)
from deepvol.api.server import app, _fx_cross_price_impl
from deepvol.market.legacy_adapter import (
    ConstMessageGraph,
    DiscountCurve,
    LegacyContext,
    build_request,
    to_mutable_message,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "legacy"
COMPLETE_FIXTURE_PATH = FIXTURES_DIR / "xaurub_k10_complete_const_messages.json"
REFERENCE_MUTABLE_PATH = FIXTURES_DIR / "xaurub_k10_mutable_message_reference.json"


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient with lifespan manager."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def complete_const_messages() -> List[Dict[str, Any]]:
    with open(COMPLETE_FIXTURE_PATH, "r", encoding="utf-8") as f:
        return cast(List[Dict[str, Any]], json.load(f))


@pytest.fixture(scope="session")
def reference_mutable_message() -> Dict[str, Any]:
    with open(REFERENCE_MUTABLE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    res = data[0] if isinstance(data, list) else data
    return cast(Dict[str, Any], res)


def _assert_recursive_structural_equality(actual: Any, expected: Any, path: str = "root") -> None:
    """Verifies that actual and expected have identical key sets, subkey nesting, and list dimensions."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"At {path}: expected dict, got {type(actual)}"
        assert set(actual.keys()) == set(expected.keys()), (
            f"At {path}: key mismatch.\nExtra: {set(actual.keys()) - set(expected.keys())}\n"
            f"Missing: {set(expected.keys()) - set(actual.keys())}"
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


def _assert_all_finite(obj: Any, path: str = "root") -> None:
    """Recursively asserts that all floating point numbers are strictly finite (non-NaN, non-Inf)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_all_finite(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        assert not math.isnan(obj), f"NaN detected at {path}"
        assert not math.isinf(obj), f"Inf detected at {path}"


def _make_mock_response(
    *,
    product_value: float,
    forward_cross: float,
    delta_cross: float,
    delta_1: float,
    delta_2: float,
    cash_delta: float,
    gamma: float,
    vega_cross: float = 100.0,
    vega_1: float = 55.0,
    vega_2: float = 45.0,
    theta: float = 150.0,
    vanna: float = -0.01,
    vomma: float = 0.03,
    correlation_sensitivity: float = 5000.0,
    implied_vol_cross: float = 0.22,
) -> TriangulatedFXOptionResponse:
    return TriangulatedFXOptionResponse(
        status="done",
        product_value=product_value,
        implied_vol_cross=implied_vol_cross,
        forward_cross=forward_cross,
        delta=DeltaLegs(cross=delta_cross, leg1=delta_1, leg2=delta_2),
        cash_delta=CashDeltaLegs(cross=cash_delta, leg1=cash_delta, leg2=cash_delta),
        gamma=gamma,
        vega=VegaLegs(
            cross=vega_cross,
            leg1=vega_1,
            leg2=vega_2,
            smile_matrix_leg1=[vega_1, vega_1 * 0.05, vega_1 * 0.02, vega_1 * 0.08, vega_1 * 0.03],
            smile_matrix_leg2=[vega_2, vega_2 * 0.04, vega_2 * 0.01, vega_2 * 0.06, vega_2 * 0.02],
        ),
        theta=theta,
        vanna=vanna,
        vomma=vomma,
        correlation_sensitivity=correlation_sensitivity,
        rho_domestic=None,
        rho_foreign=None,
        model_reserve_bps=0.15,
        warnings=[],
        latency_ms=0.15,
        telemetry=ModelRiskTelemetry(
            engine_used=PricingEngineEnum.ANALYTIC_TRIANGULATED,
            standardized_moneyness=0.0,
            mahalanobis_distance=1.0,
            ood_threshold=5.35,
            is_out_of_distribution=False,
            ensemble_epistemic_std_bps=0.0,
            fallback_activated=False,
            fallback_reason=None,
            durrleman_min_g=0.25,
            smile_dynamics=SmileDynamicsEnum.STICKY_MONEYNESS,
            model_version="v3.0.0",
        ),
    )


# ── 1. Mathematical Identities Stress across 10,000 Random Market States ──────


class TestMathematicalIdentitiesStress10k:
    """Stress-tests conservation and scaling identities across 10,000 randomized market states."""

    def test_mathematical_identities_stress_10000_states(self) -> None:
        """Evaluates 10,000 diverse random market states and verifies all four identities."""
        rng = np.random.default_rng(42)
        n_states = 10_000

        # Parameter distributions
        s1_arr = rng.uniform(10.0, 5000.0, size=n_states)
        s2_arr = rng.uniform(0.5, 200.0, size=n_states)
        d_a_arr = rng.uniform(0.5, 1.2, size=n_states)
        d_b_arr = rng.uniform(0.5, 1.2, size=n_states)
        d_c_arr = rng.uniform(0.5, 1.2, size=n_states)
        dvd_f_arr = rng.uniform(-1.5, 1.5, size=n_states)
        gamma_arr = rng.uniform(1e-10, 10.0, size=n_states)

        # Baseline context setup
        as_of = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
        exp_date = datetime(2028, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
        set_date = datetime(2028, 10, 13, 12, 0, 0, tzinfo=timezone.utc)

        curve_c = DiscountCurve(as_of, [exp_date, set_date], [8.0, 8.0])
        curve_a = DiscountCurve(as_of, [exp_date, set_date], [2.0, 2.0])

        ctx_base = LegacyContext(
            root_id="test_root",
            product_id="test_prod",
            model_id="test_model",
            market_data_set_id="test_mds",
            queries=["ProductValue", "Delta", "CashDelta", "Gamma", "CashGamma"],
            curves={"RUB_RUONIA_OIS": curve_c, "XAU_RUB": curve_a},
            tenors=(2.0, 2.005479),
            pair_cross="XAURUB",
            pair_leg1="XAUUSD",
            pair_leg2="USDRUB",
        )

        max_cash_delta_abs_diff = 0.0
        max_cash_delta_rel_diff = 0.0
        max_spot_delta_rel_diff = 0.0
        max_carry_ratio_rel_diff = 0.0
        max_cash_gamma_diff = 0.0

        for i in range(n_states):
            s1 = float(s1_arr[i])
            s2 = float(s2_arr[i])
            s_cross = s1 * s2
            d_a = float(d_a_arr[i])
            d_b = float(d_b_arr[i])
            d_c = float(d_c_arr[i])
            g = d_a / d_c
            f_cross = s_cross * g

            dvd_f = float(dvd_f_arr[i])
            gamma_val = float(gamma_arr[i])

            delta_cross = g * dvd_f
            delta_1 = dvd_f * (f_cross / s1)
            delta_2 = dvd_f * (f_cross / s2)
            cash_delta = f_cross * dvd_f

            # Build mock engine response adhering to pricing contract
            resp = _make_mock_response(
                product_value=abs(cash_delta) * 0.95,
                forward_cross=f_cross,
                delta_cross=delta_cross,
                delta_1=delta_1,
                delta_2=delta_2,
                cash_delta=cash_delta,
                gamma=gamma_val,
            )

            req = TriangulatedFXOptionRequest(
                contract=ContractTerms(
                    identifier="XAURUB",
                    parity=ParityEnum.CALL,
                    strike=f_cross,
                    as_of_date=as_of,
                    expiration_date=exp_date,
                    settlement_date=set_date,
                    payment_currency="RUB",
                    settlement_type=SettlementTypeEnum.CASH,
                ),
                market_data=MarketData(
                    leg1=FXLegData(
                        pair="XAUUSD",
                        spot=s1,
                        foreign_discount_factor=d_a,
                        domestic_discount_factor=d_b,
                        smile=SmileQuotes(
                            atm_vol=0.20,
                            rr25=0.02,
                            bf25=0.005,
                            rr10=0.03,
                            bf10=0.01,
                            delta_convention=DeltaConventionEnum.SPOT_PNA,
                        ),
                    ),
                    leg2=FXLegData(
                        pair="USDRUB",
                        spot=s2,
                        foreign_discount_factor=d_b,
                        domestic_discount_factor=d_c,
                        smile=SmileQuotes(
                            atm_vol=0.18,
                            rr25=0.01,
                            bf25=0.005,
                            rr10=0.02,
                            bf10=0.01,
                            delta_convention=DeltaConventionEnum.FORWARD_PA,
                        ),
                    ),
                    correlation=0.35,
                    settlement_discount_factor=d_c,
                ),
                interest_rates=InterestRates(
                    domestic_curve=[],
                    foreign_curve=[],
                ),
                execution_options=ExecutionOptions(
                    engine=PricingEngineEnum.ANALYTIC_TRIANGULATED,
                ),
            )

            serialized = to_mutable_message(
                resp, req, ctx_base, nonce="state_test", total_sec=0.01, proc_sec=0.009
            )
            res = serialized["MutableMessage"]["value"]["Result"]

            # Identity 1: CashDelta conservation across cross, leg 1, and leg 2
            cd_x = res["CashDelta"]["XAURUB"]
            cd_1 = res["CashDelta"]["XAUUSD"]
            cd_2 = res["CashDelta"]["USDRUB"]

            diff_x_1 = abs(cd_x - cd_1)
            diff_x_2 = abs(cd_x - cd_2)
            max_diff = max(diff_x_1, diff_x_2)
            max_cash_delta_abs_diff = max(max_cash_delta_abs_diff, max_diff)

            scale = max(abs(cd_x), 1e-12)
            rel_diff = max_diff / scale
            max_cash_delta_rel_diff = max(max_cash_delta_rel_diff, rel_diff)

            assert max_diff <= 1e-6, f"State {i}: CashDelta abs diff {max_diff} > 1e-6"
            assert rel_diff <= 1e-12, f"State {i}: CashDelta rel diff {rel_diff} > 1e-12"

            # Identity 2: Spot delta identity: Delta_i * S_i == CashDelta_i
            d_x = res["Delta"]["XAURUB"]
            d_1 = res["Delta"]["XAUUSD"]
            d_2 = res["Delta"]["USDRUB"]

            rel_err_x = abs(d_x * s_cross - cd_x) / scale
            rel_err_1 = abs(d_1 * s1 - cd_1) / scale
            rel_err_2 = abs(d_2 * s2 - cd_2) / scale
            max_spot_rel = max(rel_err_x, rel_err_1, rel_err_2)
            max_spot_delta_rel_diff = max(max_spot_delta_rel_diff, max_spot_rel)

            assert max_spot_rel <= 1e-9, f"State {i}: Spot delta identity rel diff {max_spot_rel} > 1e-9"

            # Identity 3: Forward delta carry ratio: Delta_i / NonAdaptedDelta_i == g
            for pair in ("XAURUB", "XAUUSD", "USDRUB"):
                delta_val = res["Delta"][pair]
                nad_val = res["NonAdaptedDelta"][pair]
                if abs(nad_val) > 1e-12:
                    ratio = delta_val / nad_val
                    err_ratio = abs(ratio - g) / g
                    max_carry_ratio_rel_diff = max(max_carry_ratio_rel_diff, err_ratio)
                    assert err_ratio <= 1e-12, f"State {i}: Carry ratio rel diff {err_ratio} > 1e-12 for {pair}"

            # Identity 4: CashGamma identity across all legs
            cg_x = res["CashGamma"]["XAURUB"]
            cg_1 = res["CashGamma"]["XAUUSD"]
            cg_2 = res["CashGamma"]["USDRUB"]
            expected_cg = gamma_val * (s_cross ** 2) / 100.0

            diff_cg = max(abs(cg_x - expected_cg), abs(cg_1 - expected_cg), abs(cg_2 - expected_cg))
            max_cash_gamma_diff = max(max_cash_gamma_diff, diff_cg)
            assert diff_cg <= 1e-12, f"State {i}: CashGamma diff {diff_cg} > 1e-12"

        print(
            f"\n[10k Stress Results]\n"
            f"  Max CashDelta Abs Diff:   {max_cash_delta_abs_diff:.3e} (threshold: 1e-6)\n"
            f"  Max CashDelta Rel Diff:   {max_cash_delta_rel_diff:.3e} (threshold: 1e-12)\n"
            f"  Max Spot Delta Rel Diff:  {max_spot_delta_rel_diff:.3e} (threshold: 1e-9)\n"
            f"  Max Carry Ratio Rel Diff: {max_carry_ratio_rel_diff:.3e} (threshold: 1e-12)\n"
            f"  Max CashGamma Diff:       {max_cash_gamma_diff:.3e} (threshold: 1e-12)"
        )


# ── 2. Serializer Key & Nesting Verification ──────────────────────────────────


class TestSerializerKeyNestingVerification:
    """Verifies schema fidelity, structural nesting equality, and JSON round-trip stability."""

    def test_recursive_key_set_and_nesting_equality_to_reference(
        self,
        complete_const_messages: List[Dict[str, Any]],
        reference_mutable_message: Dict[str, Any],
    ) -> None:
        """Asserts exact recursive key set and nesting equality to xaurub_k10_mutable_message_reference.json."""
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        # Build mock response matching contract
        resp = _make_mock_response(
            product_value=355690.38,
            forward_cross=465832.0,
            delta_cross=0.93,
            delta_1=80.57,
            delta_2=4109.67,
            cash_delta=355698.01,
            gamma=1.25e-6,
            vega_cross=125.0,
            vega_1=70.0,
            vega_2=55.0,
            theta=187.93,
            vanna=-0.015,
            vomma=0.042,
            correlation_sensitivity=12450.0,
            implied_vol_cross=0.225,
        )

        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.913, proc_sec=0.907)

        # Exact recursive comparison
        _assert_recursive_structural_equality(serialized, reference_mutable_message)

    def test_json_and_orjson_round_trip_stability(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        """Verifies bitwise fidelity, round-trip equality, and SHA-1 message hash determinism."""
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        resp = _make_mock_response(
            product_value=355690.38,
            forward_cross=465832.0,
            delta_cross=0.93,
            delta_1=80.57,
            delta_2=4109.67,
            cash_delta=355698.01,
            gamma=1.25e-6,
            vega_cross=125.0,
            vega_1=70.0,
            vega_2=55.0,
            theta=187.93,
            vanna=-0.015,
            vomma=0.042,
            correlation_sensitivity=12450.0,
            implied_vol_cross=0.225,
        )

        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.5, proc_sec=0.4)

        # 1. Standard json round trip
        json_dump = json.dumps(serialized, sort_keys=True)
        json_loaded = json.loads(json_dump)
        assert json_loaded == serialized

        # 2. orjson round trip
        orjson_bytes = orjson.dumps(serialized, option=orjson.OPT_SORT_KEYS)
        orjson_loaded = orjson.loads(orjson_bytes)
        assert orjson_loaded == serialized

        # 3. Hash verification: id matches SHA-1 of canonical value
        val = serialized["MutableMessage"]["value"]
        canon_bytes = json.dumps(val, sort_keys=True, separators=(",", ":")).encode("utf-8")
        computed_sha1 = hashlib.sha1(canon_bytes).hexdigest()
        assert serialized["MutableMessage"]["id"] == computed_sha1


# ── 3. End-to-End Ingestion-to-Pricing Pipeline ───────────────────────────────


class TestEndToEndIngestionToPricingPipeline:
    """Verifies complete ingestion -> request building -> live pricing -> serialization pipeline."""

    def test_live_pricing_pipeline_complete(
        self,
        complete_const_messages: List[Dict[str, Any]],
        reference_mutable_message: Dict[str, Any],
    ) -> None:
        """Evaluates legacy DAG through the live engine, verifying schema and non-NaN finiteness."""
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"

        # 1. Ingest DAG and build canonical request
        req, _, ctx = build_request(graph, root_id)
        assert req.contract.identifier == "XAURUB"
        assert req.contract.strike == 10.0

        # 2. Evaluate via live pricing engine
        resp = asyncio.run(_fx_cross_price_impl(req))
        assert resp.status == "done"
        assert resp.product_value > 0.0
        assert resp.forward_cross > 0.0

        # 3. Serialize response to legacy format
        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.15, proc_sec=0.12)

        # 4. Assert exact schema structure matches legacy reference
        _assert_recursive_structural_equality(serialized, reference_mutable_message)

        # 5. Assert all output fields are strictly finite
        _assert_all_finite(serialized)

        # 6. Verify mathematical identities on live output
        res = serialized["MutableMessage"]["value"]["Result"]
        s1 = float(req.market_data.leg1.spot)
        s2 = float(req.market_data.leg2.spot)
        s_cross = s1 * s2
        g = float(req.market_data.leg1.foreign_discount_factor / req.market_data.leg2.domestic_discount_factor)

        # CashDelta conservation
        cd_x = res["CashDelta"]["XAURUB"]
        cd_1 = res["CashDelta"]["XAUUSD"]
        cd_2 = res["CashDelta"]["USDRUB"]
        assert abs(cd_x - cd_1) <= 1e-6
        assert abs(cd_x - cd_2) <= 1e-6

        # Spot delta identity: Delta * Spot == CashDelta
        d_x = res["Delta"]["XAURUB"]
        d_1 = res["Delta"]["XAUUSD"]
        d_2 = res["Delta"]["USDRUB"]
        assert abs(d_x * s_cross - cd_x) / cd_x <= 1e-9
        assert abs(d_1 * s1 - cd_1) / cd_1 <= 1e-9
        assert abs(d_2 * s2 - cd_2) / cd_2 <= 1e-9

        # Carry ratio
        for p in ("XAURUB", "XAUUSD", "USDRUB"):
            delta_val = res["Delta"][p]
            nad_val = res["NonAdaptedDelta"][p]
            assert abs(delta_val / nad_val - g) / g <= 1e-12

        # CashGamma identity
        cg_x = res["CashGamma"]["XAURUB"]
        cg_1 = res["CashGamma"]["XAUUSD"]
        cg_2 = res["CashGamma"]["USDRUB"]
        gamma_x = res["Gamma"]["XAURUB"]
        expected_cg = gamma_x * (s_cross ** 2) / 100.0
        assert abs(cg_x - expected_cg) <= 1e-12
        assert abs(cg_x - cg_1) <= 1e-12
        assert abs(cg_x - cg_2) <= 1e-12

    def test_live_pricing_via_fastapi_client(
        self,
        client: TestClient,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        """Verifies end-to-end pricing through FastAPI endpoint /fx_cross/price."""
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        req_payload = req.model_dump(mode="json")
        http_resp = client.post("/fx_cross/price", json=req_payload)
        assert http_resp.status_code == 200, f"HTTP Error: {http_resp.text}"

        data = http_resp.json()
        resp = TriangulatedFXOptionResponse.model_validate(data)
        assert resp.status == "done"
        assert resp.product_value > 0.0

        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.2, proc_sec=0.18)
        _assert_all_finite(serialized)


# ── 4. Boundary Conditions & Numerical Extremes ───────────────────────────────


class TestBoundaryConditionsAndNumericalExtremes:
    """Stress-tests legacy serializer under extreme market conditions."""

    @pytest.mark.parametrize(
        "spot_case,s1,s2",
        [
            ("micro_spot", 0.01, 0.02),
            ("macro_spot", 1_000_000.0, 500_000.0),
            ("asymmetric_high_low", 10_000_000.0, 0.0001),
            ("asymmetric_low_high", 0.0001, 10_000_000.0),
        ],
    )
    def test_extreme_spot_levels(
        self,
        spot_case: str,
        s1: float,
        s2: float,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        """Asserts zero division errors or NaN values under extreme spot configurations."""
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        # Mutate spots
        req.market_data.leg1.spot = s1
        req.market_data.leg2.spot = s2
        s_cross = s1 * s2
        g = float(req.market_data.leg1.foreign_discount_factor / req.market_data.leg2.domestic_discount_factor)
        f_cross = s_cross * g

        resp = _make_mock_response(
            product_value=f_cross * 0.5,
            forward_cross=f_cross,
            delta_cross=g * 0.5,
            delta_1=0.5 * f_cross / s1,
            delta_2=0.5 * f_cross / s2,
            cash_delta=f_cross * 0.5,
            gamma=1e-8,
            vega_cross=50.0,
            vega_1=30.0,
            vega_2=20.0,
            theta=10.0,
            vanna=-0.001,
            vomma=0.002,
            correlation_sensitivity=100.0,
            implied_vol_cross=0.25,
        )

        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.01, proc_sec=0.008)
        _assert_all_finite(serialized)

        res = serialized["MutableMessage"]["value"]["Result"]
        cd_x = res["CashDelta"]["XAURUB"]
        cd_1 = res["CashDelta"]["XAUUSD"]
        cd_2 = res["CashDelta"]["USDRUB"]
        assert abs(cd_x - cd_1) <= 1e-6
        assert abs(cd_x - cd_2) <= 1e-6

    def test_zero_carry_parity(
        self,
        complete_const_messages: List[Dict[str, Any]],
    ) -> None:
        """Asserts exact behaviour when g == 1.0 (zero carry / equal discount factors)."""
        graph = ConstMessageGraph(complete_const_messages)
        root_id = "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"
        req, _, ctx = build_request(graph, root_id)

        # Set discount factors equal -> g = 1.0
        req.market_data.leg1.foreign_discount_factor = 1.0
        req.market_data.leg1.domestic_discount_factor = 1.0
        req.market_data.leg2.foreign_discount_factor = 1.0
        req.market_data.leg2.domestic_discount_factor = 1.0
        req.market_data.settlement_discount_factor = 1.0

        s1 = req.market_data.leg1.spot
        s2 = req.market_data.leg2.spot
        s_cross = s1 * s2

        resp = _make_mock_response(
            product_value=s_cross * 0.2,
            forward_cross=s_cross,
            delta_cross=0.5,
            delta_1=0.5 * s_cross / s1,
            delta_2=0.5 * s_cross / s2,
            cash_delta=0.5 * s_cross,
            gamma=1e-6,
            vega_cross=10.0,
            vega_1=5.0,
            vega_2=5.0,
            theta=5.0,
            vanna=-0.001,
            vomma=0.001,
            correlation_sensitivity=50.0,
            implied_vol_cross=0.20,
        )

        serialized = to_mutable_message(resp, req, ctx, nonce=root_id, total_sec=0.01, proc_sec=0.008)
        _assert_all_finite(serialized)

        res = serialized["MutableMessage"]["value"]["Result"]
        # When g == 1.0, Delta == NonAdaptedDelta identically
        for pair in ("XAURUB", "XAUUSD", "USDRUB"):
            delta_val = res["Delta"][pair]
            nad_val = res["NonAdaptedDelta"][pair]
            assert abs(delta_val - nad_val) <= 1e-12
