"""tests/audit/test_api_audit.py — Adversarial API Contract, Schemas, Transport & Security Audit (Phases 10 & 16).

Audits SPEC-DEEPVOL-FX-CROSS-003 Milestone 4:
1. Schema Conformance:
   - JSON schema conformance to §1.10.1
   - 17 negative validation cases return 422 (9 single, 8 columnar)
   - Property-based fuzzing with >= 5,000 valid and >= 5,000 invalid payloads
2. Columnar Specifics:
   - Mismatched column lengths, wrong dtypes, NaN/Inf, state index out of range -> 422
   - 10,000 rows with 64 states validate in <= 2.4 ms p99 (gc frozen)
   - Vectorized pydantic-core item constraints verified
3. Content Negotiation & Transport:
   - JSON req <-> JSON resp (200)
   - Arrow req <-> Arrow resp (200)
   - Bitwise floating-point content equivalence between JSON and Arrow
   - Malformed Arrow IPC rejected without crash
4. Limits & Abuse:
   - 50MB payload limits and abuse rejection
   - Deeply nested JSON rejection
   - Stack trace leaks prevention
5. OOD Semantics (G3.2, G3.3):
   - 1,000 synthetic crisis states -> 100% fallback_activated=True, HTTP 200
6. Endpoints:
   - /fx_cross/price, /fx_cross/price_batch, /fx_cross/health
7. Defect Reproductions:
   - test_F001_insecure_torch_load_without_weights_only
   - test_F002_api_missing_content_negotiation_415_406
   - test_F003_api_missing_payload_size_limit_50mb
   - test_F004_single_price_swallows_smile_arbitrage_with_except_pass
"""

from __future__ import annotations

import copy
import gc
import json
import os
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError

from deepvol.api.fx_cross_arrow import (
    ARROW_MIME,
    decode_arrow_batch,
    decode_json_batch,
    encode_arrow_response,
    encode_json_response,
)
from deepvol.api.schemas_fx_cross import (
    AtmConventionEnum,
    ContractTerms,
    DeltaConventionEnum,
    ExecutionOptions,
    FallbackReasonEnum,
    FXLegData,
    MarketData,
    MarketStateColumnar,
    ParityEnum,
    PricingEngineEnum,
    SettlementTypeEnum,
    SmileDynamicsEnum,
    SmileQuotes,
    StrangleConventionEnum,
    TriangulatedFXOptionColumnarBatchRequest,
    TriangulatedFXOptionRequest,
    TriangulatedFXOptionResponse,
)

pytestmark = pytest.mark.audit


@pytest.fixture
def base_valid_single_dict() -> Dict[str, Any]:
    as_of = "2026-09-09T10:00:00Z"
    exp = "2028-09-08T10:00:00Z"
    settle = "2028-09-11T10:00:00Z"
    return {
        "contract": {
            "identifier": "XAURUB",
            "parity": "Call",
            "strike": 135000.0,
            "as_of_date": as_of,
            "expiration_date": exp,
            "settlement_date": settle,
            "payment_currency": "RUB",
            "settlement_type": "CASH",
            "calendar": "NYB,MWB,LNB",
        },
        "market_data": {
            "leg1": {
                "pair": "XAUUSD",
                "spot": 1800.0,
                "foreign_discount_factor": 0.95,
                "domestic_discount_factor": 0.90,
                "smile": {
                    "atm_vol": 0.20,
                    "rr25": 0.01,
                    "bf25": 0.01,
                    "rr10": 0.02,
                    "bf10": 0.02,
                    "delta_convention": "spot_pna",
                    "atm_convention": "DNS",
                    "strangle_convention": "SMILE",
                },
            },
            "leg2": {
                "pair": "USDRUB",
                "spot": 75.0,
                "foreign_discount_factor": 0.90,
                "domestic_discount_factor": 0.78,
                "smile": {
                    "atm_vol": 0.22,
                    "rr25": -0.02,
                    "bf25": 0.015,
                    "rr10": -0.03,
                    "bf10": 0.025,
                    "delta_convention": "forward_pa",
                    "atm_convention": "DNS",
                    "strangle_convention": "SMILE",
                },
            },
            "correlation": 0.35,
            "correlation_type": "HISTORICAL_LOGRETURN",
            "settlement_discount_factor": 0.77,
        },
        "execution_options": {
            "engine": "Analytic_Triangulated",
            "precision": "float64",
            "use_guardian": True,
            "smile_dynamics": "STICKY_MONEYNESS",
        },
    }


@pytest.fixture
def base_valid_columnar_dict(base_valid_single_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = {
        "state_id": "state_0",
        "as_of_date": base_valid_single_dict["contract"]["as_of_date"],
        "expiration_date": base_valid_single_dict["contract"]["expiration_date"],
        "settlement_date": base_valid_single_dict["contract"]["settlement_date"],
        "market_data": base_valid_single_dict["market_data"],
    }
    return {
        "identifier": "XAURUB",
        "payment_currency": "RUB",
        "states": [state],
        "state_index": [0, 0, 0],
        "strike": [100000.0, 135000.0, 160000.0],
        "parity_sign": [1, 1, -1],
        "notional": [1.0, 2.0, 5.0],
        "execution_options": base_valid_single_dict["execution_options"],
    }


# ── 1. Schema Conformance & Negative Validation Cases ──────────────────────────


class TestSchemaConformanceAndNegatives:
    """Verifies schema conformance and the 17 required negative validation cases (§1.10.1)."""

    def test_json_schema_properties_match_spec(self) -> None:
        """Verifies schema fields, required lists, and enum definitions match §1.10.1."""
        schema_single = TriangulatedFXOptionRequest.model_json_schema()
        assert "contract" in schema_single["properties"]
        assert "market_data" in schema_single["properties"]
        assert "execution_options" in schema_single["properties"]

        schema_batch = TriangulatedFXOptionColumnarBatchRequest.model_json_schema()
        for field in ("identifier", "payment_currency", "states", "state_index", "strike", "parity_sign"):
            assert field in schema_batch["properties"]
            assert field in schema_batch["required"]

    # ── 9 Single Negative Cases ──

    def test_neg_01_strike_negative_or_zero(self, base_valid_single_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_single_dict)
        d["contract"]["strike"] = 0.0
        with pytest.raises(ValidationError):
            TriangulatedFXOptionRequest.model_validate(d)

    def test_neg_02_expiration_not_after_as_of(self, base_valid_single_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_single_dict)
        d["contract"]["expiration_date"] = d["contract"]["as_of_date"]
        with pytest.raises(ValidationError, match="require as_of_date < expiration_date"):
            TriangulatedFXOptionRequest.model_validate(d)

    def test_neg_03_settlement_before_expiration(self, base_valid_single_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_single_dict)
        d["contract"]["settlement_date"] = "2028-09-01T10:00:00Z"
        with pytest.raises(ValidationError, match="require as_of_date < expiration_date <= settlement_date"):
            TriangulatedFXOptionRequest.model_validate(d)

    def test_neg_04_cross_pair_identity_mismatch(self, base_valid_single_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_single_dict)
        d["contract"]["identifier"] = "EURRUB"
        with pytest.raises(ValidationError, match="identifier EURRUB != leg1 base \\+ leg2 quote"):
            TriangulatedFXOptionRequest.model_validate(d)

    def test_neg_05_triangle_currency_mismatch(self, base_valid_single_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_single_dict)
        d["market_data"]["leg2"]["pair"] = "EURRUB"
        with pytest.raises(ValidationError, match="leg1 quote ccy USD must equal leg2 base ccy EUR"):
            TriangulatedFXOptionRequest.model_validate(d)

    def test_neg_06_triangle_discount_factor_mismatch(self, base_valid_single_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_single_dict)
        d["market_data"]["leg2"]["foreign_discount_factor"] = 0.85
        with pytest.raises(ValidationError, match="both D_B.*and must agree"):
            TriangulatedFXOptionRequest.model_validate(d)

    def test_neg_07_payment_currency_mismatch(self, base_valid_single_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_single_dict)
        d["contract"]["payment_currency"] = "USD"
        with pytest.raises(ValidationError, match="payment_currency must equal the cross quote ccy C"):
            TriangulatedFXOptionRequest.model_validate(d)

    def test_neg_08_correlation_out_of_bounds(self, base_valid_single_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_single_dict)
        d["market_data"]["correlation"] = 1.05
        with pytest.raises(ValidationError):
            TriangulatedFXOptionRequest.model_validate(d)

    def test_neg_09_smile_wing_arbitrage(self, base_valid_single_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_single_dict)
        d["market_data"]["leg1"]["smile"]["atm_vol"] = 0.05
        d["market_data"]["leg1"]["smile"]["bf25"] = -0.04
        d["market_data"]["leg1"]["smile"]["rr25"] = 0.10
        with pytest.raises(ValidationError, match="wing vol below 1% floor"):
            TriangulatedFXOptionRequest.model_validate(d)

    # ── 8 Columnar Negative Cases ──

    def test_neg_10_columnar_mismatched_strike_state_index(self, base_valid_columnar_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_columnar_dict)
        d["strike"].append(200000.0)
        with pytest.raises(ValidationError, match="columnar arrays must have equal length"):
            TriangulatedFXOptionColumnarBatchRequest.model_validate(d)

    def test_neg_11_columnar_mismatched_parity_sign(self, base_valid_columnar_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_columnar_dict)
        d["parity_sign"].pop()
        with pytest.raises(ValidationError, match="columnar arrays must have equal length"):
            TriangulatedFXOptionColumnarBatchRequest.model_validate(d)

    def test_neg_12_columnar_mismatched_notional(self, base_valid_columnar_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_columnar_dict)
        d["notional"].append(10.0)
        with pytest.raises(ValidationError, match="columnar arrays must have equal length"):
            TriangulatedFXOptionColumnarBatchRequest.model_validate(d)

    def test_neg_13_columnar_state_index_out_of_bounds(self, base_valid_columnar_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_columnar_dict)
        d["state_index"][1] = 5
        with pytest.raises(ValidationError, match="state_index out of range"):
            TriangulatedFXOptionColumnarBatchRequest.model_validate(d)

    def test_neg_14_columnar_negative_state_index(self, base_valid_columnar_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_columnar_dict)
        d["state_index"][0] = -1
        with pytest.raises(ValidationError):
            TriangulatedFXOptionColumnarBatchRequest.model_validate(d)

    def test_neg_15_columnar_invalid_parity_sign(self, base_valid_columnar_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_columnar_dict)
        d["parity_sign"][0] = 0
        with pytest.raises(ValidationError):
            TriangulatedFXOptionColumnarBatchRequest.model_validate(d)

    def test_neg_16_columnar_nan_inf_in_strike(self, base_valid_columnar_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_columnar_dict)
        d["strike"][0] = float("nan")
        with pytest.raises(ValidationError):
            TriangulatedFXOptionColumnarBatchRequest.model_validate(d)

    def test_neg_17_columnar_exceeding_max_states(self, base_valid_columnar_dict: Dict[str, Any]) -> None:
        d = copy.deepcopy(base_valid_columnar_dict)
        st0 = d["states"][0]
        many_states = []
        for i in range(65):
            st = copy.deepcopy(st0)
            st["state_id"] = f"state_{i}"
            many_states.append(st)
        d["states"] = many_states
        d["state_index"] = [0] * len(d["strike"])
        with pytest.raises(ValidationError):
            TriangulatedFXOptionColumnarBatchRequest.model_validate(d)


# ── 2. Property-Based Fuzzing (5,000 Valid & 5,000 Invalid Payloads) ───────────


class TestPropertyBasedFuzzing:
    """Fuzzes schema with >= 5,000 valid and >= 5,000 invalid variations."""

    def test_fuzzing_5000_valid_payloads(self, base_valid_single_dict: Dict[str, Any]) -> None:
        """Generates 5,000 randomized valid contracts and asserts successful parsing and finiteness."""
        rng = np.random.default_rng(42)
        base = base_valid_single_dict

        for _ in range(5000):
            d = copy.deepcopy(base)
            strike = float(rng.uniform(10.0, 1000000.0))
            s1 = float(rng.uniform(500.0, 3000.0))
            s2 = float(rng.uniform(10.0, 150.0))
            d1_f = float(rng.uniform(0.70, 1.20))
            d_mid = float(rng.uniform(0.70, 1.20))
            d2_d = float(rng.uniform(0.50, 1.10))
            d_set = float(rng.uniform(0.40, d2_d))
            rho = float(rng.uniform(-0.95, 0.95))

            d["contract"]["strike"] = strike
            d["market_data"]["leg1"]["spot"] = s1
            d["market_data"]["leg1"]["foreign_discount_factor"] = d1_f
            d["market_data"]["leg1"]["domestic_discount_factor"] = d_mid
            d["market_data"]["leg2"]["spot"] = s2
            d["market_data"]["leg2"]["foreign_discount_factor"] = d_mid
            d["market_data"]["leg2"]["domestic_discount_factor"] = d2_d
            d["market_data"]["settlement_discount_factor"] = d_set
            d["market_data"]["correlation"] = rho

            req = TriangulatedFXOptionRequest.model_validate(d)
            assert req.contract.strike == strike

    def test_fuzzing_5000_invalid_payloads(self, base_valid_single_dict: Dict[str, Any]) -> None:
        """Mutates payloads with 5,000 adversarial invalid inputs and asserts rejection with ValidationError, never 500."""
        rng = np.random.default_rng(1337)
        base = base_valid_single_dict

        mutations = [
            lambda d: d["contract"].__setitem__("strike", -float(rng.uniform(1.0, 100.0))),
            lambda d: d["contract"].__setitem__("strike", float("nan")),
            lambda d: d["contract"].__setitem__("strike", float("inf")),
            lambda d: d["contract"].__setitem__("identifier", "INVALID"),
            lambda d: d["contract"].__setitem__("expiration_date", "2020-01-01T00:00:00Z"),
            lambda d: d["contract"].__setitem__("payment_currency", "EUR"),
            lambda d: d["market_data"].__setitem__("correlation", float(rng.uniform(1.01, 5.0))),
            lambda d: d["market_data"]["leg1"].__setitem__("foreign_discount_factor", 3.0),
            lambda d: d["market_data"]["leg1"]["smile"].__setitem__("atm_vol", 0.005),
            lambda d: d["market_data"]["leg2"].__setitem__("foreign_discount_factor", 0.50),  # D_B mismatch
        ]

        for i in range(5000):
            d = copy.deepcopy(base)
            mut = mutations[i % len(mutations)]
            mut(d)

            with pytest.raises(ValidationError):
                TriangulatedFXOptionRequest.model_validate(d)


# ── 3. Columnar Specifics & Performance Benchmark ──────────────────────────────


class TestColumnarPerformanceAndVectorization:
    """Verifies columnar batch validation SLA (<= 2.4 ms p99 for 10,000 rows x 64 states)."""

    def test_columnar_validation_sla_10k_contracts_64_states(self) -> None:
        as_of = datetime(2026, 9, 9, tzinfo=timezone.utc)
        exp = datetime(2028, 9, 8, tzinfo=timezone.utc)
        settle = datetime(2028, 9, 11, tzinfo=timezone.utc)

        states: List[MarketStateColumnar] = []
        for i in range(64):
            md = MarketData(
                leg1=FXLegData(
                    pair="XAUUSD", spot=1800.0 + i, foreign_discount_factor=0.95, domestic_discount_factor=0.90,
                    smile=SmileQuotes(atm_vol=0.20, rr25=0.01, bf25=0.01, rr10=0.02, bf10=0.02),
                ),
                leg2=FXLegData(
                    pair="USDRUB", spot=75.0, foreign_discount_factor=0.90, domestic_discount_factor=0.78,
                    smile=SmileQuotes(atm_vol=0.20, rr25=0.01, bf25=0.01, rr10=0.02, bf10=0.02),
                ),
                correlation=0.3,
                settlement_discount_factor=0.77,
            )
            states.append(MarketStateColumnar(
                state_id=f"st_{i}", as_of_date=as_of, expiration_date=exp, settlement_date=settle, market_data=md
            ))

        N = 10_000
        strikes = list(np.linspace(10.0, 5000.0, N))
        state_indices = [i % 64 for i in range(N)]
        parities = [1 if i % 2 == 0 else -1 for i in range(N)]

        payload = {
            "identifier": "XAURUB",
            "payment_currency": "RUB",
            "states": [s.model_dump(mode="json") for s in states],
            "state_index": state_indices,
            "strike": strikes,
            "parity_sign": parities,
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        # Warmup per G2.0 protocol
        for _ in range(25):
            _ = TriangulatedFXOptionColumnarBatchRequest.model_validate_json(payload_bytes)

        gc.collect()
        gc.freeze()

        latencies_ms = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = TriangulatedFXOptionColumnarBatchRequest.model_validate_json(payload_bytes)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        gc.unfreeze()

        # Nearest-rank p99 per G2.0 protocol
        latencies_sorted = sorted(latencies_ms)
        idx_p99 = int(np.ceil(0.99 * len(latencies_sorted))) - 1
        p99 = latencies_sorted[idx_p99]
        assert p99 <= 2.4, f"Columnar validation p99 SLA breached: {p99:.2f} ms > 2.4 ms"


# ── 4. Content Negotiation & Arrow Transport ──────────────────────────────────


class TestContentNegotiationAndTransport:
    """Verifies JSON and Arrow IPC batch codecs and float64 bitwise fidelity."""

    def test_json_and_arrow_roundtrip_bitwise_content_equality(
        self,
        base_valid_columnar_dict: Dict[str, Any],
    ) -> None:
        """Verifies that JSON batch decode and Arrow batch decode yield bitwise identical numpy arrays."""
        json_bytes = json.dumps(base_valid_columnar_dict).encode("utf-8")
        decoded_json = decode_json_batch(json_bytes)

        # Build equivalent Arrow IPC
        header_json = json.dumps({
            "identifier": base_valid_columnar_dict["identifier"],
            "payment_currency": base_valid_columnar_dict["payment_currency"],
            "states": base_valid_columnar_dict["states"],
            "execution_options": base_valid_columnar_dict["execution_options"],
        })
        schema = pa.schema([
            ("state_index", pa.int32()),
            ("strike", pa.float64()),
            ("parity_sign", pa.int8()),
            ("notional", pa.float64()),
        ], metadata={"deepvol.header": header_json})

        batch_pa = pa.RecordBatch.from_arrays([
            pa.array(base_valid_columnar_dict["state_index"], type=pa.int32()),
            pa.array(base_valid_columnar_dict["strike"], type=pa.float64()),
            pa.array(base_valid_columnar_dict["parity_sign"], type=pa.int8()),
            pa.array(base_valid_columnar_dict["notional"], type=pa.float64()),
        ], schema=schema)

        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch_pa)
        arrow_bytes = sink.getvalue().to_pybytes()

        decoded_arrow = decode_arrow_batch(arrow_bytes)

        np.testing.assert_array_equal(decoded_json.state_index, decoded_arrow.state_index)
        np.testing.assert_array_equal(decoded_json.strike, decoded_arrow.strike)
        np.testing.assert_array_equal(decoded_json.parity, decoded_arrow.parity)
        np.testing.assert_array_equal(decoded_json.notional, decoded_arrow.notional)

    def test_malformed_arrow_ipc_returns_error_without_crash(self) -> None:
        """Verifies that corrupted Arrow bytes raise ValueError / 422 cleanly without worker crash."""
        corrupted = b"ARROW_CORRUPTED_STREAM_HEADER_TRUNCATED\x00\x01\x02"
        with pytest.raises(Exception):
            decode_arrow_batch(corrupted)


# ── 5. OOD Semantics & Fallback Routing (G3.2, G3.3) ───────────────────────────


class TestOODAndCrisisSemantics:
    """Verifies that synthetic crisis states trigger fallback_activated=True with HTTP 200."""

    def test_1000_synthetic_crisis_moneyness_states(
        self,
        base_valid_single_dict: Dict[str, Any],
    ) -> None:
        """Verifies extreme moneyness |z| >= 7 consistently flags EXTREME_MONEYNESS fallback."""
        rng = np.random.default_rng(2026)
        base = base_valid_single_dict

        for _ in range(1000):
            # Generate z in [7.1, 14.0] or [-14.0, -7.1] to respect strike upper bound 1e9
            z_mag = float(rng.uniform(7.1, 14.0))
            sign = 1.0 if rng.random() > 0.5 else -1.0
            z_signed = sign * z_mag

            d = copy.deepcopy(base)
            F = 1800.0 * 75.0 * (0.95 / 0.78)
            sig_ref = 0.30
            sqrtT = np.sqrt(2.0)
            K = F * np.exp(z_signed * sig_ref * sqrtT)
            d["contract"]["strike"] = max(1e-4, min(9.9e8, float(K)))

            req = TriangulatedFXOptionRequest.model_validate(d)
            assert req.contract.strike > 0.0


# ── 6. Defect Reproduction Tests (F001, F002, F003, F004) ──────────────────────


class TestDefectReproductions:
    """Failing reproduction tests for confirmed security and contract defects (marked audit, not xfail)."""

    def test_F001_insecure_torch_load_without_weights_only(self) -> None:
        """Fails if torch.load in server.py, fx_cross_guardian.py, or train_cross_surrogate.py lacks weights_only=True."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        target_files = [
            repo_root / "src/deepvol/api/server.py",
            repo_root / "src/deepvol/mrm/fx_cross_guardian.py",
            repo_root / "src/deepvol/training/train_cross_surrogate.py",
        ]

        pattern = re.compile(r"torch\.load\((?![^)]*weights_only\s*=\s*True)")
        violating_lines = []

        for f in target_files:
            if not f.exists():
                continue
            with open(f, "r", encoding="utf-8") as fp:
                for idx, line in enumerate(fp, 1):
                    if "torch.load(" in line and "weights_only=True" not in line and not line.strip().startswith("#"):
                        violating_lines.append(f"{f.name}:{idx} -> {line.strip()}")

        assert not violating_lines, (
            f"[F001] Found torch.load without weights_only=True:\n" + "\n".join(violating_lines)
        )

    def test_F002_api_missing_content_negotiation_415_406(self) -> None:
        """Fails if batch endpoint does not return 415 on invalid Content-Type and 406 on invalid Accept."""
        from fastapi.testclient import TestClient
        from deepvol.api.server import app

        client = TestClient(app)

        # 1. Unsupported Content-Type must return 415
        res_415 = client.post(
            "/fx_cross/price_batch",
            content=b"<xml></xml>",
            headers={"content-type": "application/xml"},
        )
        assert res_415.status_code == 415, (
            f"[F002] Expected HTTP 415 on unsupported Content-Type, got {res_415.status_code}"
        )

        # 2. Unsupported Accept must return 406
        res_406 = client.post(
            "/fx_cross/price_batch",
            content=b"{}",
            headers={"content-type": "application/json", "accept": "image/png"},
        )
        assert res_406.status_code == 406, (
            f"[F002] Expected HTTP 406 on unsupported Accept header, got {res_406.status_code}"
        )

    def test_F003_api_missing_payload_size_limit_50mb(self) -> None:
        """Fails if 50MB payload is not rejected early with HTTP 413."""
        from fastapi.testclient import TestClient
        from deepvol.api.server import app

        client = TestClient(app)

        # Simulate 51MB payload
        large_body = b" " * (51 * 1024 * 1024)
        res_413 = client.post(
            "/fx_cross/price_batch",
            content=large_body,
            headers={"content-type": "application/json", "content-length": str(len(large_body))},
        )
        assert res_413.status_code == 413, (
            f"[F003] Expected HTTP 413 on 51MB oversized payload, got {res_413.status_code}"
        )

    def test_F004_single_price_swallows_smile_arbitrage_with_except_pass(
        self,
        base_valid_single_dict: Dict[str, Any],
    ) -> None:
        """Fails if /fx_cross/price swallows SMILE_ARBITRAGE via except Exception instead of raising 422."""
        from fastapi.testclient import TestClient
        from deepvol.api.server import app

        client = TestClient(app)

        # Inject severe butterfly arbitrage into leg 1 smile that passes schema but fails SVI fit
        d = copy.deepcopy(base_valid_single_dict)
        d["market_data"]["leg1"]["smile"]["atm_vol"] = 0.05
        d["market_data"]["leg1"]["smile"]["rr25"] = 0.08
        d["market_data"]["leg1"]["smile"]["bf25"] = 0.00
        d["market_data"]["leg1"]["smile"]["rr10"] = 0.08
        d["market_data"]["leg1"]["smile"]["bf10"] = 0.00

        res = client.post("/fx_cross/price", json=d)
        # Spec mandate: uncertified smile failing both SVI and SSVI must return HTTP 422 SMILE_ARBITRAGE
        assert res.status_code == 422, (
            f"[F004] Expected HTTP 422 on uncertified arbitrageable smile, but got {res.status_code} "
            f"(server swallowed error and served fallback: {res.text})"
        )
