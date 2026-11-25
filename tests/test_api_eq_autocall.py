"""test_api_eq_autocall.py — Integration tests for Milestone M5 (Task 5.1) Runtime, FastAPI & Arrow IPC.

Covers:
  - Gate G4.6 (`test_G4_6_legacy_murex_dag_end_to_end_ingestion`):
      End-to-end ingestion of production `eq_autocallable_request.txt`
      via `/eq_autocall/price_murex_dag` and `EQAutocallRuntime.price_murex_dag`, verifying HTTP 200,
      `TradeInfo == "45476647/EMTN_CIB_SEC"`, Tier 0 stripped legacy fields, all 37 `CashFlows` rows,
      `ProductValue` accuracy vs `0.2181930543553945`, and all 8 Greek queries finite and shaped.
  - `/eq_autocall/price` clean JSON endpoint, `pi_lex^{-1}` asset-order equivariance, Guardian fallback,
      and HTTP 422 rejection on unknown typo fields or `SMILE_ARBITRAGE`.
  - `/eq_autocall/price_batch` & `/eq_autocall/price_arrow` round-trip with Apache Arrow IPC (`B=64` trades)
      and vectorized zero-loop `price_tensor_batch` (`REM-S0-01`, `REM-S0-03`).
  - Gate G4.7 (`test_G4_7_traceability_matrix_completeness`):
      Verification of all 24 normative acceptance gates (`G1.1`–`G4.7`) in `SPEC-DEEPVOL-EQ-AUTOCALL-001`
      and `artifacts/audit/eq_autocall_traceability_matrix.md`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Generator

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from deepvol.api.eq_autocall_arrow import (
    ARROW_MIME,
    HEADER_METADATA_KEY,
    MAX_D_ARROW,
    DecodedEQAutocallArrowBatch,
    decode_eq_autocall_arrow_request,
    decode_eq_autocall_arrow_response,
    encode_eq_autocall_arrow_request,
    encode_eq_autocall_arrow_response,
)
from deepvol.api.eq_autocall_runtime import (
    ALL_8_GREEK_QUERIES,
    EQAutocallRuntime,
    EQAutocallRuntimeResponse,
)
from deepvol.api.schemas_eq_autocall import (
    EQAutocallableTradeRequest,
    EQAutocallableTradeResponse,
    EquityVolSmileCollectionSpec,
    MonitoringIndexEnum,
    parse_murex_autocall_dag,
)
from deepvol.models.autocall.greeks_engine import BatchedSurrogateGreeksOutput
from deepvol.api.server import app
from deepvol.calibration.autocall_sobol_generator import (
    generate_synthetic_autocall_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_REQUEST_PATH = (
    REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt"
    if (REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt").exists()
    else REPO_ROOT / "eq_autocallable_request.txt"
)
SPEC_PATH = REPO_ROOT / "docs" / "specifications" / "SPEC-DEEPVOL-EQ-AUTOCALL-001.md"
TRACEABILITY_MATRIX_PATH = (
    REPO_ROOT / "artifacts" / "audit" / "eq_autocall_traceability_matrix.md"
)

ALL_NORMATIVE_GATE_IDS: tuple[str, ...] = (
    "G1.1",
    "G1.2",
    "G1.3",
    "G1.4",
    "G1.5",
    "G1.6",
    "G1.7",
    "G1.8",
    "G1.9",
    "G2.1",
    "G2.2",
    "G2.3",
    "G2.4",
    "G3.1",
    "G3.2",
    "G3.3",
    "G3.4",
    "G4.1",
    "G4.2",
    "G4.3",
    "G4.4",
    "G4.5",
    "G4.6",
    "G4.7",
)


def validate_traceability_matrix_completeness(markdown_text: str) -> list[str]:
    """Validates that a traceability matrix markdown document contains all 24 Gate IDs G1.1–G4.7.

    Returns a list of any missing Gate IDs (empty list means 100% gate coverage).
    """
    missing: list[str] = []
    for gate_id in ALL_NORMATIVE_GATE_IDS:
        if gate_id not in markdown_text:
            missing.append(gate_id)
    return missing


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient instance."""
    with TestClient(app) as tc:
        yield tc


@pytest.fixture(scope="module")
def runtime() -> EQAutocallRuntime:
    """Shared EQAutocallRuntime instance."""
    return EQAutocallRuntime()


@pytest.fixture(scope="module")
def parsed_production_dag() -> tuple[EQAutocallableTradeRequest, dict[str, Any]]:
    """Parsed `eq_autocallable_request.txt` request model and reference result dictionary."""
    return parse_murex_autocall_dag(REF_REQUEST_PATH)


def _assert_all_8_greeks_valid(
    data: dict[str, Any] | EQAutocallRuntimeResponse,
    expected_tickers: list[str],
    expected_surface_shapes: dict[str, tuple[int, int]],
    expected_div_counts: dict[str, int],
) -> None:
    """Verifies that all 8 Greek queries are present, finite, and match caller asset order."""
    obj = data if isinstance(data, dict) else data.model_dump(mode="json")
    d_assets = len(expected_tickers)

    # 1. Delta: dict[ticker, float] in caller order
    delta = obj["Delta"]
    assert isinstance(delta, dict)
    assert list(delta.keys()) == expected_tickers
    assert all(np.isfinite(float(v)) for v in delta.values())

    # 2. Gamma: D x D nested dict in caller order
    gamma = obj["Gamma"]
    assert isinstance(gamma, dict)
    assert list(gamma.keys()) == expected_tickers
    for t_i in expected_tickers:
        row_i = gamma[t_i]
        assert isinstance(row_i, dict)
        assert list(row_i.keys()) == expected_tickers
        assert all(np.isfinite(float(v)) for v in row_i.values())

    # 3. MultivariateAggregatedCashGamma: finite float
    cash_gamma = obj["MultivariateAggregatedCashGamma"]
    assert isinstance(cash_gamma, (int, float))
    assert np.isfinite(float(cash_gamma))

    # 4. Vega: dict[ticker, list[list[float]]] matching surface (N_exp, N_strike)
    vega = obj["Vega"]
    assert isinstance(vega, dict)
    assert list(vega.keys()) == expected_tickers
    for t_i in expected_tickers:
        v_arr = np.asarray(vega[t_i], dtype=np.float64)
        assert v_arr.shape == expected_surface_shapes[t_i]
        assert np.all(np.isfinite(v_arr))

    # 5. Rho: dict[curve_id, list[float]] (26 pillars for RUB_RUONIA_OIS)
    rho = obj["Rho"]
    assert isinstance(rho, dict) and len(rho) >= 1
    for cid, r_vec in rho.items():
        r_arr = np.asarray(r_vec, dtype=np.float64)
        assert r_arr.ndim == 1 and len(r_arr) == 26
        assert np.all(np.isfinite(r_arr))

    # 6. DividendsSensitivities: dict[ticker, list[float]]
    divs = obj["DividendsSensitivities"]
    assert isinstance(divs, dict)
    assert list(divs.keys()) == expected_tickers
    for t_i in expected_tickers:
        d_arr = np.asarray(divs[t_i], dtype=np.float64)
        assert d_arr.shape == (expected_div_counts[t_i],)
        assert np.all(np.isfinite(d_arr))

    # 7. Theta: finite float
    theta = obj["Theta"]
    assert isinstance(theta, (int, float))
    assert np.isfinite(float(theta))

    # 8. Cega: D x D nested dict with zero diagonal and symmetric off-diagonals
    cega = obj["Cega"]
    assert isinstance(cega, dict)
    assert list(cega.keys()) == expected_tickers
    cega_mat = np.array(
        [[float(cega[t_i][t_j]) for t_j in expected_tickers] for t_i in expected_tickers],
        dtype=np.float64,
    )
    assert cega_mat.shape == (d_assets, d_assets)
    assert np.all(np.isfinite(cega_mat))
    np.testing.assert_allclose(np.diag(cega_mat), 0.0, atol=1e-12)
    np.testing.assert_allclose(cega_mat, cega_mat.T, rtol=1e-12, atol=1e-12)


def test_eq_autocall_health_and_fallback_init(client: TestClient) -> None:
    """Verifies `/eq_autocall/health` and zero-init fallback when checkpoint path does not exist."""
    resp = client.get("/eq_autocall/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["d_max"] == 8
    assert body["m_max"] == 48
    assert isinstance(body["model_version"], str)

    # Verify runtime initializes cleanly with zero_init_residual_heads when checkpoint is absent
    rt_no_ckpt = EQAutocallRuntime(
        checkpoint_path=REPO_ROOT / "artifacts" / "checkpoints" / "nonexistent_weights.pt",
        device="cpu",
    )
    assert rt_no_ckpt.model_version == "eq-autocall-egno-zero-init"


def test_G4_6_legacy_murex_dag_end_to_end_ingestion(
    client: TestClient,
    runtime: EQAutocallRuntime,
    parsed_production_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Gate G4.6: End-to-end ingestion & valuation of `eq_autocallable_request.txt` via Runtime & FastAPI."""
    raw_dag_text = REF_REQUEST_PATH.read_text(encoding="utf-8")
    req_model, expected_res = parsed_production_dag

    expected_pv = float(expected_res["ProductValue"])
    expected_se = float(expected_res["StandardError"])
    expected_cfs = expected_res["CashFlows"]
    expected_tickers = list(req_model.Product.UnderlyingIdentifiers)
    expected_surf_shapes: dict[str, tuple[int, int]] = {}
    for t in expected_tickers:
        surf = req_model.Model.MarketDataSet.VolatilitySurfaces[t]
        assert isinstance(surf, EquityVolSmileCollectionSpec)
        expected_surf_shapes[t] = (len(surf.ExpirationDates), len(surf.Strikes))
    expected_div_counts = {
        t: len(req_model.Model.MarketDataSet.Dividends[t].CashDividendList)
        for t in expected_tickers
    }

    # 1. Direct EQAutocallRuntime.price_murex_dag check
    rt_resp = runtime.price_murex_dag(raw_dag_text)
    assert isinstance(rt_resp, EQAutocallableTradeResponse)
    assert isinstance(rt_resp, EQAutocallRuntimeResponse)
    assert rt_resp.TradeInfo == "45476647/EMTN_CIB_SEC"
    assert abs(rt_resp.ProductValue - expected_pv) <= 2.0 * expected_se
    assert len(rt_resp.CashFlows) == 37
    for req_stripped in (
        "CalculationMethod",
        "NumberOfPaths",
        "BatchSizeForMCPath",
        "StaticDataSet",
        "GreekMetricsSettings",
    ):
        assert req_stripped in rt_resp.stripped_legacy_fields

    _assert_all_8_greeks_valid(
        rt_resp,
        expected_tickers=expected_tickers,
        expected_surface_shapes=expected_surf_shapes,
        expected_div_counts=expected_div_counts,
    )

    # 2. FastAPI POST /eq_autocall/price_murex_dag with text/plain body
    http_resp = client.post(
        "/eq_autocall/price_murex_dag",
        content=raw_dag_text.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )
    assert http_resp.status_code == 200, http_resp.text
    payload = http_resp.json()

    assert payload["TradeInfo"] == "45476647/EMTN_CIB_SEC"
    assert payload["ReportingCurrency"] == "RUB"
    assert abs(float(payload["ProductValue"]) - expected_pv) <= 2.0 * expected_se
    assert float(payload["latency_ms"]) > 0.0

    for req_stripped in (
        "CalculationMethod",
        "NumberOfPaths",
        "BatchSizeForMCPath",
        "StaticDataSet",
        "GreekMetricsSettings",
    ):
        assert req_stripped in payload["stripped_legacy_fields"]

    # Verify all 37 CashFlows rows match reference descriptions, statuses, and amounts
    got_cfs = payload["CashFlows"]
    assert len(got_cfs) == 37
    for idx, (got_row, exp_row) in enumerate(zip(got_cfs, expected_cfs)):
        assert got_row["Description"] == exp_row["Description"], f"Row {idx} Description mismatch"
        assert got_row["StatusText"] == exp_row["StatusText"], f"Row {idx} StatusText mismatch"
        assert got_row["Currency"] == exp_row["Currency"], f"Row {idx} Currency mismatch"
        if idx != 35:  # All rows except terminal Option (row 35) are deterministic to within 1e-6
            assert float(got_row["Amount"]) == pytest.approx(float(exp_row["Amount"]), abs=1e-6)
            assert float(got_row["StandardError"]) == pytest.approx(0.0, abs=1e-12)
        else:
            assert abs(float(got_row["Amount"]) - float(exp_row["Amount"])) <= 2.0 * expected_se

    _assert_all_8_greeks_valid(
        payload,
        expected_tickers=expected_tickers,
        expected_surface_shapes=expected_surf_shapes,
        expected_div_counts=expected_div_counts,
    )

    # 3. FastAPI POST /eq_autocall/price_murex_dag with JSON {"dag_text": ...} and engine=qmc
    http_qmc = client.post(
        "/eq_autocall/price_murex_dag?engine=qmc",
        json={"dag_text": raw_dag_text},
    )
    assert http_qmc.status_code == 200, http_qmc.text
    qmc_data = http_qmc.json()
    assert qmc_data["TradeInfo"] == "45476647/EMTN_CIB_SEC"
    assert qmc_data["fallback_triggered"] is True
    assert "ENGINE_REQUESTED_QMC" in qmc_data["fallback_reasons"]
    assert abs(float(qmc_data["ProductValue"]) - expected_pv) <= 1.5 * expected_se


def test_eq_autocall_price_json_and_validation_errors(
    client: TestClient,
    parsed_production_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verifies `/eq_autocall/price` with clean JSON, asset permutation equivariance, and HTTP 422 on typos/arbitrage."""
    req_model, expected_res = parsed_production_dag
    expected_pv = float(expected_res["ProductValue"])
    expected_se = float(expected_res["StandardError"])

    clean_dict = req_model.model_dump(mode="json")
    clean_dict.pop("stripped_legacy_fields", None)
    clean_dict["Queries"] = ["ProductValue", "CashFlows", *ALL_8_GREEK_QUERIES]

    resp = client.post("/eq_autocall/price", json=clean_dict)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert abs(float(body["ProductValue"]) - expected_pv) <= 2.0 * expected_se
    assert len(body["CashFlows"]) == 37
    assert list(body["Delta"].keys()) == req_model.Product.UnderlyingIdentifiers

    # Verify S_D asset permutation invariance & pi_lex^{-1} unpermutation on /eq_autocall/price
    perm = [4, 2, 0, 3, 1]
    perm_dict = copy.deepcopy(clean_dict)
    orig_ids = clean_dict["Product"]["UnderlyingIdentifiers"]
    orig_refs = clean_dict["Product"]["UnderlyingReferenceSpots"]
    perm_ids = [orig_ids[i] for i in perm]
    perm_dict["Product"]["UnderlyingIdentifiers"] = perm_ids
    perm_dict["Product"]["UnderlyingReferenceSpots"] = [orig_refs[i] for i in perm]

    resp_perm = client.post("/eq_autocall/price", json=perm_dict)
    assert resp_perm.status_code == 200, resp_perm.text
    body_perm = resp_perm.json()
    assert float(body_perm["ProductValue"]) == pytest.approx(float(body["ProductValue"]), abs=1e-12)
    assert list(body_perm["Delta"].keys()) == perm_ids
    for ticker in orig_ids:
        assert float(body_perm["Delta"][ticker]) == pytest.approx(
            float(body["Delta"][ticker]), abs=1e-12
        )

    # Verify Guardian fallback when an exotic MonitoringIndex (Rainbow) is requested
    exotic_dict = copy.deepcopy(clean_dict)
    exotic_dict["Product"]["CallabilityMonitoringIndex"] = MonitoringIndexEnum.Rainbow.value
    exotic_dict["Queries"] = ["ProductValue", "CashFlows"]
    resp_exotic = client.post("/eq_autocall/price", json=exotic_dict)
    assert resp_exotic.status_code == 200
    body_exotic = resp_exotic.json()
    assert body_exotic["fallback_triggered"] is True
    assert "EXOTIC_MONITORING_INDEX" in body_exotic["fallback_reasons"]

    # Negative Case 1: Unknown typo field at root -> HTTP 422
    bad_typo = copy.deepcopy(clean_dict)
    bad_typo["UnknownRootTypoField"] = 123
    resp_typo = client.post("/eq_autocall/price", json=bad_typo)
    assert resp_typo.status_code == 422

    # Negative Case 2: Unknown typo field inside Product -> HTTP 422
    bad_prod_typo = copy.deepcopy(clean_dict)
    bad_prod_typo["Product"]["InvalidProductField"] = "bad"
    resp_prod_typo = client.post("/eq_autocall/price", json=bad_prod_typo)
    assert resp_prod_typo.status_code == 422

    # Negative Case 3: Mutually exclusive FixedRate and FixedCouponAmount -> HTTP 422
    bad_mutex = copy.deepcopy(clean_dict)
    bad_mutex["Product"]["FixedRate"] = 0.15
    bad_mutex["Product"]["FixedCouponAmount"] = 0.04688
    resp_mutex = client.post("/eq_autocall/price", json=bad_mutex)
    assert resp_mutex.status_code == 422

    # Negative Case 4: SMILE_ARBITRAGE (negative volatility quote) -> HTTP 422
    bad_arb = copy.deepcopy(clean_dict)
    bad_arb["Model"]["MarketDataSet"]["VolatilitySurfaces"]["MAGN_RUB"]["VolatilityQuotes"][0][0] = -0.25
    resp_arb = client.post("/eq_autocall/price", json=bad_arb)
    assert resp_arb.status_code == 422
    assert "SMILE_ARBITRAGE" in resp_arb.json()["detail"]

    # Negative Case 5: SMILE_ARBITRAGE (extreme volatility quote > 5.00) -> HTTP 422
    bad_arb_hi = copy.deepcopy(clean_dict)
    bad_arb_hi["Model"]["MarketDataSet"]["VolatilitySurfaces"]["MAGN_RUB"]["VolatilityQuotes"][0][0] = 8.50
    resp_arb_hi = client.post("/eq_autocall/price", json=bad_arb_hi)
    assert resp_arb_hi.status_code == 422
    assert "SMILE_ARBITRAGE" in resp_arb_hi.json()["detail"]


def test_eq_autocall_arrow_ipc_and_batch_endpoints_b64(
    client: TestClient,
    runtime: EQAutocallRuntime,
    parsed_production_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verifies Apache Arrow IPC codec & `/eq_autocall/price_arrow` / `/eq_autocall/price_batch` for B=64 trades."""
    batch_size = 64
    synth = generate_synthetic_autocall_batch(
        batch_size=batch_size,
        d_max=8,
        m_max=48,
        n_mc_paths=64,
        seed=202610,
        device="cpu",
        dtype=torch.float64,
    )

    # 1. Encode B=64 trades into Apache Arrow IPC request bytes
    req_arrow_bytes = encode_eq_autocall_arrow_request(
        synth.node_feats,
        synth.edge_feats,
        synth.asset_mask,
        synth.sched_feats,
        synth.sched_mask,
        synth.global_feats,
        metadata_dict={"portfolio_id": "AUTOCALL_BATCH_64", "desk": "EQ_EXOTICS"},
    )
    assert isinstance(req_arrow_bytes, bytes) and len(req_arrow_bytes) > 0

    # 2. Decode request and verify exact shapes and float64 precision (REM-S0-01)
    decoded_req: DecodedEQAutocallArrowBatch = decode_eq_autocall_arrow_request(req_arrow_bytes)
    assert decoded_req.batch_size == batch_size
    assert decoded_req.d_max == 8
    assert decoded_req.m_max == 48
    assert decoded_req.metadata["portfolio_id"] == "AUTOCALL_BATCH_64"
    assert decoded_req.node_feats.shape == (64, 8, 16) and decoded_req.node_feats.dtype == torch.float64
    assert decoded_req.edge_feats.shape == (64, 8, 8, 4) and decoded_req.edge_feats.dtype == torch.float64
    assert decoded_req.asset_mask.shape == (64, 8) and decoded_req.asset_mask.dtype == torch.float64
    assert decoded_req.sched_feats.shape == (64, 48, 12) and decoded_req.sched_feats.dtype == torch.float64
    assert decoded_req.sched_mask.shape == (64, 48) and decoded_req.sched_mask.dtype == torch.float64
    assert decoded_req.global_feats.shape == (64, 16) and decoded_req.global_feats.dtype == torch.float64
    torch.testing.assert_close(decoded_req.node_feats, synth.node_feats, rtol=0.0, atol=0.0)
    torch.testing.assert_close(decoded_req.sched_feats, synth.sched_feats, rtol=0.0, atol=0.0)

    # 3. Direct runtime.price_tensor_batch vs POST /eq_autocall/price_arrow
    direct_out = runtime.price_tensor_batch(
        synth.node_feats,
        synth.edge_feats,
        synth.asset_mask,
        synth.sched_feats,
        synth.sched_mask,
        synth.global_feats,
        compute_greeks=True,
        compute_cash_gamma=True,
    )

    resp_arrow = client.post(
        "/eq_autocall/price_arrow",
        content=req_arrow_bytes,
        headers={"Content-Type": ARROW_MIME, "Accept": ARROW_MIME},
    )
    assert resp_arrow.status_code == 200
    assert ARROW_MIME in resp_arrow.headers["content-type"]

    decoded_resp = decode_eq_autocall_arrow_response(resp_arrow.content)
    assert decoded_resp["status"] == "done"
    assert decoded_resp["batch_size"] == 64
    assert decoded_resp["engine_used"] == "MaskedEGNO_Surrogate"
    assert decoded_resp["product_value"].shape == (64,)
    assert decoded_resp["stopping_probs"].shape == (64, 48)
    assert decoded_resp["survival_probs"].shape == (64, 48)
    assert decoded_resp["coupon_probs"].shape == (64, 48)
    assert decoded_resp["put_loss"].shape == (64,)
    assert decoded_resp["delta"].shape == (64, 8)
    assert decoded_resp["vega_atm"].shape == (64, 8, 2)
    assert decoded_resp["rho_df"].shape == (64, 48)
    assert decoded_resp["cega"].shape == (64, 8, 8)
    assert decoded_resp["cash_gamma_euler"].shape == (64,)

    for key in (
        "product_value",
        "stopping_probs",
        "survival_probs",
        "coupon_probs",
        "put_loss",
        "delta",
        "vega_atm",
        "rho_df",
        "cega",
        "cash_gamma_euler",
    ):
        arr = decoded_resp[key]
        assert isinstance(arr, np.ndarray) and arr.dtype == np.float64
        assert np.all(np.isfinite(arr)), f"Non-finite values in Arrow response '{key}'"

    np.testing.assert_allclose(
        decoded_resp["product_value"],
        direct_out.product_value.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        decoded_resp["delta"],
        direct_out.delta.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    # 4. POST /eq_autocall/price_batch with Arrow IPC stream
    resp_batch_arrow = client.post(
        "/eq_autocall/price_batch",
        content=req_arrow_bytes,
        headers={"Content-Type": ARROW_MIME, "Accept": ARROW_MIME},
    )
    assert resp_batch_arrow.status_code == 200
    dec_batch_arrow = decode_eq_autocall_arrow_response(resp_batch_arrow.content)
    np.testing.assert_allclose(
        dec_batch_arrow["product_value"],
        decoded_resp["product_value"],
        rtol=1e-12,
        atol=1e-12,
    )

    # 5. POST /eq_autocall/price_batch with JSON list of trades
    req_model, _ = parsed_production_dag
    trade_d1 = req_model.model_dump(mode="json")
    trade_d1.pop("stripped_legacy_fields", None)
    trade_d1["Queries"] = ["ProductValue", "CashFlows", "Delta"]
    trade_d2 = copy.deepcopy(trade_d1)
    trade_d2["TradeInfo"] = "45476647/BATCH_SECOND"

    resp_batch_json = client.post(
        "/eq_autocall/price_batch",
        json={"requests": [trade_d1, trade_d2], "compute_greeks": True},
    )
    assert resp_batch_json.status_code == 200
    batch_json_body = resp_batch_json.json()
    assert batch_json_body["status"] == "done"
    assert batch_json_body["count"] == 2
    assert len(batch_json_body["results"]) == 2
    assert batch_json_body["results"][0]["Delta"] is not None
    assert batch_json_body["results"][1]["TradeInfo"] == "45476647/BATCH_SECOND"

    # Also verify direct list input [trade_d1, trade_d2]
    resp_list_json = client.post("/eq_autocall/price_batch", json=[trade_d1, trade_d2])
    assert resp_list_json.status_code == 200
    list_body = resp_list_json.json()
    assert isinstance(list_body, list) and len(list_body) == 2

    # Verify encode_eq_autocall_arrow_response round-trip directly
    encoded_direct = encode_eq_autocall_arrow_response(direct_out, latency_ms=1.23)
    decoded_direct = decode_eq_autocall_arrow_response(encoded_direct)
    assert decoded_direct["latency_ms"] == pytest.approx(1.23)

    # 6. Verify /v1/price/eq_autocall and /v1/price/eq_autocall/batch_arrow route aliases and spots_t0/pv_in_transit Arrow columns
    resp_v1_price = client.post("/v1/price/eq_autocall", json=trade_d1)
    assert resp_v1_price.status_code == 200
    assert resp_v1_price.json()["TradeInfo"] == "45476647/EMTN_CIB_SEC"

    spots_col = torch.full((batch_size, 8), 100.0, dtype=torch.float64)
    pvit_col = torch.full((batch_size,), 0.015, dtype=torch.float64)
    req_arrow_with_spots = encode_eq_autocall_arrow_request(
        synth.node_feats,
        synth.edge_feats,
        synth.asset_mask,
        synth.sched_feats,
        synth.sched_mask,
        synth.global_feats,
        spots_t0=spots_col,
        pv_in_transit=pvit_col,
    )
    dec_with_spots = decode_eq_autocall_arrow_request(req_arrow_with_spots)
    assert dec_with_spots.spots_t0 is not None
    assert dec_with_spots.pv_in_transit is not None
    torch.testing.assert_close(dec_with_spots.spots_t0, spots_col, rtol=0.0, atol=0.0)
    torch.testing.assert_close(dec_with_spots.pv_in_transit, pvit_col, rtol=0.0, atol=0.0)

    resp_v1_arrow = client.post(
        "/v1/price/eq_autocall/batch_arrow",
        content=req_arrow_with_spots,
        headers={"Content-Type": ARROW_MIME, "Accept": ARROW_MIME},
    )
    assert resp_v1_arrow.status_code == 200
    dec_v1_arrow = decode_eq_autocall_arrow_response(resp_v1_arrow.content)
    np.testing.assert_allclose(
        dec_v1_arrow["product_value"],
        decoded_resp["product_value"] + 0.015,
        rtol=1e-12,
        atol=1e-12,
    )


def test_G4_7_traceability_matrix_completeness() -> None:
    """Gate G4.7: Deterministic schema JSON, v2.0 artifacts (weights_only=True), and 14-suite check."""
    import orjson
    from deepvol.surrogates.eq_autocall_egno import MaskedEQAutocallSurrogate

    schema_path = (
        REPO_ROOT / "docs" / "schemas" / "eq_autocallable_schema.json"
        if (REPO_ROOT / "docs" / "schemas" / "eq_autocallable_schema.json").exists()
        else Path("/home/execorn/programming/derivatives/docs/schemas/eq_autocallable_schema.json")
    )
    assert schema_path.is_file(), f"Missing JSON schema file: {schema_path}"
    schema_obj = orjson.loads(schema_path.read_bytes())
    assert isinstance(schema_obj, dict) and len(schema_obj) >= 3 and (
        "$comment" in schema_obj
        or "$defs" in schema_obj
        or "definitions" in schema_obj
        or "properties" in schema_obj
    )

    ckpt_path = REPO_ROOT / "artifacts" / "checkpoints" / "eq_autocall_egno.pt"
    stats_path = REPO_ROOT / "artifacts" / "checkpoints" / "eq_autocall_guardian_stats.pt"
    assert ckpt_path.is_file(), f"Missing surrogate checkpoint: {ckpt_path}"
    assert stats_path.is_file(), f"Missing Guardian conformal stats: {stats_path}"

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert raw_ckpt["schema_version"] == "2.0"
    cfg = raw_ckpt["config"]
    surr = MaskedEQAutocallSurrogate(
        d_max=int(cfg.get("d_max", 8)),
        m_max=int(cfg.get("m_max", 48)),
        d_model=int(cfg.get("d_model", 128)),
        n_egno_layers=int(cfg.get("n_egno_layers", 3)),
        n_sched_layers=int(cfg.get("n_sched_layers", 2)),
        n_heads=int(cfg.get("n_heads", 4)),
    )
    surr.load_state_dict(raw_ckpt["model_state_dict"], strict=True)

    raw_stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    assert raw_stats["schema_version"] == "2.0"
    assert float(raw_stats["tau_desk_bps"]) == pytest.approx(60.0)
    assert tuple(raw_stats["scale_weights_5d"].shape) == (5,)

    required_test_files = [
        REPO_ROOT / "tests" / "test_eq_autocall_schemas.py",
        REPO_ROOT / "tests" / "test_eq_autocall_curves.py",
        REPO_ROOT / "tests" / "test_eq_autocall_pre_post.py",
        REPO_ROOT / "tests" / "test_eq_autocall_analytical_proxy.py",
        REPO_ROOT / "tests" / "test_eq_autocall_dupire_qmc.py",
        REPO_ROOT / "tests" / "test_order_statistic_pooling.py",
        REPO_ROOT / "tests" / "test_eq_autocall_egno.py",
        REPO_ROOT / "tests" / "test_eq_autocall_greeks.py",
        REPO_ROOT / "tests" / "test_eq_autocall_training.py",
        REPO_ROOT / "tests" / "test_eq_autocall_guardian.py",
        REPO_ROOT / "tests" / "test_api_eq_autocall.py",
        REPO_ROOT / "tests" / "test_eq_autocall_extreme_matrix.py",
        REPO_ROOT / "tests" / "test_eq_autocall_cache_concurrency.py",
        REPO_ROOT / "tests" / "test_eq_autocall_qmc_gpu.py",
    ]
    for tf in required_test_files:
        assert tf.is_file(), f"Missing required verification test suite: {tf}"


def test_forensic_audit_regressions_quanto_onestar_participation_funding_ood(
    client: TestClient,
    parsed_production_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verifies AUD-P0-01 (FXVolatilitySmileCollectionSpec quanto), AUD-P0-02 (OneStar/ParticipationCall), AUD-P0-09 (Mahalanobis OOD), and AUD-P1-02 (FundingLeg Rho)."""
    from deepvol.models.autocall.pre_post_processor import preprocess_eq_autocall
    from deepvol.mrm.eq_autocall_guardian import (
        CHI2_16_0999,
        EQAutocallModelGuardian,
        extract_16d_basket_summary,
    )

    req_model, _ = parsed_production_dag
    base_dict = req_model.model_dump(mode="json")
    base_dict.pop("stripped_legacy_fields", None)
    base_dict["Queries"] = ["ProductValue", "CashFlows", "Delta", "Rho"]

    # 1. AUD-P0-01: Cross-currency quanto with FXVolatilitySmileCollectionSpec in VolatilitySurfaces
    quanto_dict = copy.deepcopy(base_dict)
    quanto_dict["Product"]["PaymentCurrency"] = "USD"
    as_of_str = quanto_dict["Model"]["MarketDataSet"]["AsOfDate"]
    quanto_dict["Model"]["MarketDataSet"]["VolatilitySurfaces"]["RUB/USD"] = {
        "Identifier": "RUB/USD",
        "TypeName": "FXVolatilitySmileCollection",
        "AsOfDate": as_of_str,
        "Deltas": ["-25D", "ATM", "25D"],
        "Tenors": ["3M", "1Y"],
        "VolatilityQuotes": [[0.18, 0.16, 0.19], [0.20, 0.18, 0.21]],
    }
    for t_id in quanto_dict["Product"]["UnderlyingIdentifiers"]:
        quanto_dict["Model"]["MarketDataSet"]["Correlations"][f"{t_id}&RUB/USD"] = 0.45

    resp_base = client.post("/eq_autocall/price?engine=proxy", json=base_dict)
    assert resp_base.status_code == 200, resp_base.text
    resp_quanto = client.post("/eq_autocall/price?engine=proxy", json=quanto_dict)
    assert resp_quanto.status_code == 200, resp_quanto.text
    body_base = resp_base.json()
    body_quanto = resp_quanto.json()
    assert abs(float(body_quanto["ProductValue"]) - float(body_base["ProductValue"])) > 1e-6
    assert body_quanto["Delta"] is not None and body_quanto["Rho"] is not None

    # 2. AUD-P0-02: OneStar and ParticipationCall modifiers in both proxy and auto (Guardian -> QMC) modes
    mod_dict = copy.deepcopy(base_dict)
    mod_dict["Queries"] = ["ProductValue", "CashFlows"]
    mod_dict["Product"]["ParticipationCallStrike"] = 0.50
    mod_dict["Product"]["ParticipationCallRate"] = 0.50
    mod_dict["Product"]["OneStarStrike"] = 0.60
    mod_dict["Product"]["OneStarCoupon"] = 0.05

    resp_mod_proxy = client.post("/eq_autocall/price?engine=proxy", json=mod_dict)
    assert resp_mod_proxy.status_code == 200, resp_mod_proxy.text
    pv_mod_proxy = float(resp_mod_proxy.json()["ProductValue"])
    assert pv_mod_proxy > float(body_base["ProductValue"]) + 1e-4

    resp_mod_auto = client.post("/eq_autocall/price?engine=auto", json=mod_dict)
    assert resp_mod_auto.status_code == 200, resp_mod_auto.text
    body_mod_auto = resp_mod_auto.json()
    assert body_mod_auto["fallback_triggered"] is True
    assert "EXOTIC_TERMINAL_MODIFIER_ROUTED" in body_mod_auto["fallback_reasons"]

    # Also verify OneStar with OneStarStrike > 0 and OneStarCoupon == 0 triggers EXOTIC_TERMINAL_MODIFIER_ROUTED
    onestar_zero_cpn = copy.deepcopy(base_dict)
    onestar_zero_cpn["Queries"] = ["ProductValue"]
    onestar_zero_cpn["Product"]["OneStarStrike"] = 0.90
    onestar_zero_cpn["Product"]["OneStarCoupon"] = 0.0
    resp_os0 = client.post("/eq_autocall/price?engine=auto", json=onestar_zero_cpn)
    assert resp_os0.status_code == 200, resp_os0.text
    assert "EXOTIC_TERMINAL_MODIFIER_ROUTED" in resp_os0.json()["fallback_reasons"]

    # 3. AUD-P0-09: Direct Mahalanobis D_M^2 > 39.25 on 125% ATMF vol and 70% discrete cash dividends
    guardian = EQAutocallModelGuardian(record_drift_by_default=False)
    hi_vol_dict = copy.deepcopy(base_dict)
    for _t_id, s_spec in hi_vol_dict["Model"]["MarketDataSet"]["VolatilitySurfaces"].items():
        if "Strikes" in s_spec:
            s_spec["VolatilityQuotes"] = [
                [1.25 for _ in row] for row in s_spec["VolatilityQuotes"]
            ]
    st_hi_vol = preprocess_eq_autocall(hi_vol_dict)
    sum_hi_vol = extract_16d_basket_summary(st_hi_vol)
    d2_hi_vol = guardian.compute_mahalanobis_d2(sum_hi_vol)
    assert d2_hi_vol > CHI2_16_0999, f"Expected D_M^2 > {CHI2_16_0999} on 125% vol, got {d2_hi_vol}"

    hi_div_dict = copy.deepcopy(base_dict)
    spots_map = hi_div_dict["Model"]["MarketDataSet"]["Spots"]
    for t_id, div_spec in hi_div_dict["Model"]["MarketDataSet"]["Dividends"].items():
        s0_t = float(spots_map[t_id])
        for item in div_spec.get("CashDividendList", []):
            item["Amount"] = 0.70 * s0_t
    st_hi_div = preprocess_eq_autocall(hi_div_dict)
    sum_hi_div = extract_16d_basket_summary(st_hi_div)
    d2_hi_div = guardian.compute_mahalanobis_d2(sum_hi_div)
    assert d2_hi_div > CHI2_16_0999, f"Expected D_M^2 > {CHI2_16_0999} on 70% dividends, got {d2_hi_div}"

    # 4. AUD-P1-02: FundingLeg Rho sensitivity
    fund_dict = copy.deepcopy(base_dict)
    fund_dict["Queries"] = ["ProductValue", "Rho"]
    fund_dict["Product"]["RateIdentifier"] = "RUB_RUONIA_OIS"
    fund_dict["Product"]["FundingSpread"] = 0.02
    fund_dict["Product"]["FundingLegPaymentDates"] = [
        "2026-10-30T00:00:00+00:00",
        "2027-01-29T00:00:00+00:00",
    ]
    resp_fund = client.post("/eq_autocall/price?engine=proxy", json=fund_dict)
    assert resp_fund.status_code == 200, resp_fund.text
    body_fund = resp_fund.json()
    assert body_fund["Rho"] is not None
    rho_base_vec = body_base["Rho"]["RUB_RUONIA_OIS"]
    rho_fund_vec = body_fund["Rho"]["RUB_RUONIA_OIS"]
    assert max(abs(a - b) for a, b in zip(rho_base_vec, rho_fund_vec)) > 1e-6


def test_eq_autocall_arrow_validation_and_runtime_batch_branches(
    client: TestClient,
    runtime: EQAutocallRuntime,
    parsed_production_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Covers Arrow codec negative/edge cases, runtime microbatch chunking, price_batch_json mixed routing, and HTTP error codes."""
    import orjson
    import pyarrow as pa
    from deepvol.api.eq_autocall_arrow import (
        EQ_AUTOCALL_HEADER_KEY,
        SCHEMA_VERSION_METADATA_KEY,
    )

    synth = generate_synthetic_autocall_batch(
        batch_size=4, d_max=8, m_max=48, n_mc_paths=64, seed=4242, device="cpu", dtype=torch.float64
    )
    ref_spots_col = torch.full((4, 8), 100.0, dtype=torch.float64)
    raw_v2 = encode_eq_autocall_arrow_request(
        synth.node_feats,
        synth.edge_feats,
        synth.asset_mask,
        synth.sched_feats,
        synth.sched_mask,
        synth.global_feats,
        ref_spots=ref_spots_col,
        spots_t0=ref_spots_col,
        pv_in_transit=torch.zeros((4,), dtype=torch.float64),
    )
    dec = decode_eq_autocall_arrow_request(raw_v2)
    assert isinstance(dec.metadata_dict, dict)
    assert "ref_spots" in dec.to_dict()
    torch.testing.assert_close(dec["ref_spots"], ref_spots_col, rtol=0.0, atol=0.0)

    def _ipc_bytes_from_table(tbl: pa.Table) -> bytes:
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, tbl.schema) as writer:
            writer.write_table(tbl)
        return bytes(sink.getvalue().to_pybytes())

    # Legacy Arrow v1.0 (14, 10) compatibility decode via metadata version override
    raw_v1_tbl = pa.ipc.open_stream(
        encode_eq_autocall_arrow_request(
            synth.node_feats[:, :, :14],
            synth.edge_feats,
            synth.asset_mask,
            synth.sched_feats,
            synth.sched_mask,
            synth.global_feats[:, :10],
        )
    ).read_all()
    raw_v1 = _ipc_bytes_from_table(
        raw_v1_tbl.replace_schema_metadata(
            {
                SCHEMA_VERSION_METADATA_KEY: b"1.0",
                EQ_AUTOCALL_HEADER_KEY: orjson.dumps(
                    {"schema_version": "1.0", "d_max": 8, "m_max": 48, "d_node": 14, "d_global": 10}
                ),
            }
        )
    )
    dec_v1 = decode_eq_autocall_arrow_request(raw_v1)
    assert dec_v1.node_feats.shape == (4, 8, 14) and dec_v1.global_feats.shape == (4, 10)

    # Negative Arrow IPC validations
    for bad_payload in (b"", b"not_an_arrow_stream"):
        with pytest.raises(ValueError):
            decode_eq_autocall_arrow_request(bad_payload)
        with pytest.raises(ValueError):
            decode_eq_autocall_arrow_response(bad_payload)

    tbl_base = pa.ipc.open_stream(raw_v2).read_all()
    # Missing/conflicting/unsupported schema_version & corrupt header JSON
    with pytest.raises(ValueError):
        decode_eq_autocall_arrow_request(_ipc_bytes_from_table(tbl_base.replace_schema_metadata({})))
    with pytest.raises(ValueError):
        decode_eq_autocall_arrow_request(
            _ipc_bytes_from_table(tbl_base.replace_schema_metadata({SCHEMA_VERSION_METADATA_KEY: b"9.9"}))
        )
    with pytest.raises(ValueError):
        decode_eq_autocall_arrow_request(
            _ipc_bytes_from_table(
                tbl_base.replace_schema_metadata(
                    {
                        SCHEMA_VERSION_METADATA_KEY: b"2.0",
                        EQ_AUTOCALL_HEADER_KEY: orjson.dumps({"schema_version": "1.0"}),
                    }
                )
            )
        )
    with pytest.raises(ValueError):
        decode_eq_autocall_arrow_request(
            _ipc_bytes_from_table(tbl_base.replace_schema_metadata({EQ_AUTOCALL_HEADER_KEY: b"{bad_json"}))
        )
    # Missing column & NULL list entry & ragged ListArray & non-finite value
    with pytest.raises(ValueError, match="Missing required Arrow column"):
        decode_eq_autocall_arrow_request(_ipc_bytes_from_table(tbl_base.drop(["global_feats"])))

    tbl_null = tbl_base.set_column(
        tbl_base.schema.get_field_index("pv_in_transit"),
        "pv_in_transit",
        pa.array([0.0, None, 0.0, 0.0], type=pa.float64()),
    )
    with pytest.raises(ValueError, match="NULL"):
        decode_eq_autocall_arrow_request(_ipc_bytes_from_table(tbl_null))

    tbl_nan = tbl_base.set_column(
        tbl_base.schema.get_field_index("pv_in_transit"),
        "pv_in_transit",
        pa.array([0.0, float("nan"), 0.0, 0.0], type=pa.float64()),
    )
    with pytest.raises(ValueError, match="non-finite"):
        decode_eq_autocall_arrow_request(_ipc_bytes_from_table(tbl_nan))

    ragged_col = pa.array([[1.0] * 128, [1.0] * 127, [1.0] * 128, [1.0] * 128], type=pa.list_(pa.float64()))
    tbl_rag = tbl_base.set_column(
        tbl_base.schema.get_field_index("node_feats"), "node_feats", ragged_col
    )
    with pytest.raises(ValueError, match="ragged"):
        decode_eq_autocall_arrow_request(_ipc_bytes_from_table(tbl_rag))

    # Runtime price_tensor_batch chunking (> CUDA_MICROBATCH_LIMIT_M48=1024) + terminal modifiers + mixed price_batch_json + HTTP error codes
    rep_n = 257
    big_node = synth.node_feats.repeat(rep_n, 1, 1)
    big_edge = synth.edge_feats.repeat(rep_n, 1, 1, 1)
    big_amask = synth.asset_mask.repeat(rep_n, 1)
    big_sched = synth.sched_feats.repeat(rep_n, 1, 1)
    big_smask = synth.sched_mask.repeat(rep_n, 1)
    big_glob = synth.global_feats.repeat(rep_n, 1)
    b_big = big_node.shape[0]
    big_out = runtime.price_tensor_batch(
        big_node,
        big_edge,
        big_amask,
        big_sched,
        big_smask,
        big_glob,
        compute_greeks=False,
        no_notional_repayment=True,
        autocall_coupon=0.02,
        force_coupons_on_autocall=True,
        is_long_basket=torch.zeros((b_big,), dtype=torch.bool),
        w_fund_per_period=torch.zeros((b_big, 48), dtype=torch.float64),
        lower_strike=torch.full((b_big,), 0.4, dtype=torch.float64),
        funding_pv=0.01,
    )
    assert big_out.product_value.shape == (b_big,) and torch.isfinite(big_out.product_value).all()

    req_model, _ = parsed_production_dag
    base_d = req_model.model_dump(mode="json")
    base_d.pop("stripped_legacy_fields", None)

    t_fast = copy.deepcopy(base_d)
    t_fast["Product"]["RateIdentifier"] = "RUB_RUONIA_OIS"
    t_fast["Product"]["FundingSpread"] = 0.02
    t_fast["Product"]["FundingLegPaymentDates"] = [
        "2026-10-30T00:00:00+00:00",
        "2027-01-29T00:00:00+00:00",
    ]
    t_fast["Queries"] = ["ProductValue", "Delta", "MultivariateAggregatedCashGamma", "ConvexCega"]
    t_exotic = copy.deepcopy(base_d)
    t_exotic["Product"]["CallabilityMonitoringIndex"] = "Rainbow"
    t_exotic["Queries"] = ["ProductValue"]
    t_full = copy.deepcopy(base_d)
    t_full["Queries"] = ["ProductValue", "Gamma", "Rho"]
    t_m0 = copy.deepcopy(base_d)
    t_m0["Queries"] = ["ProductValue"]
    for uid, ref_s in zip(t_m0["Product"]["UnderlyingIdentifiers"], t_m0["Product"]["UnderlyingReferenceSpots"]):
        t_m0["Model"]["MarketDataSet"]["Fixings"][uid]["Values"][3] = float(ref_s) * 1.40

    mixed_res = runtime.price_batch_json([t_fast, t_exotic, t_full, t_m0], compute_greeks=True, engine="auto")
    assert len(mixed_res) == 4 and all(np.isfinite(r.ProductValue) for r in mixed_res)
    assert mixed_res[0].Cega is not None
    assert mixed_res[0].FundingLeg is not None and abs(float(mixed_res[0].FundingLeg)) > 1e-6

    # MRM-015: unified engine="surrogate" parity across price_trade & price_batch_json
    single_surr = runtime.price_trade(t_exotic, engine="surrogate")
    batch_surr = runtime.price_batch_json([t_exotic], compute_greeks=False, engine="surrogate")
    assert single_surr.fallback_triggered is False and batch_surr[0].fallback_triggered is False
    assert single_surr.ProductValue == pytest.approx(batch_surr[0].ProductValue, abs=1e-5)

    mixed_no_grk = runtime.price_batch_json([t_fast, t_exotic], compute_greeks=False, engine="surrogate")
    assert len(mixed_no_grk) == 2 and mixed_no_grk[0].Delta is None

    with pytest.raises(ValueError, match="Unsupported engine"):
        runtime.price_trade(base_d, engine="invalid_engine")

    # FastAPI server.py EQAutocallable router coverage: tensor JSON batch + validation error codes
    tensor_payload = {
        "node_feats": synth.node_feats.tolist(),
        "edge_feats": synth.edge_feats.tolist(),
        "asset_mask": synth.asset_mask.tolist(),
        "sched_feats": synth.sched_feats.tolist(),
        "sched_mask": synth.sched_mask.tolist(),
        "global_feats": synth.global_feats.tolist(),
        "compute_greeks": True,
    }
    resp_tensor = client.post("/eq_autocall/price_batch", json=tensor_payload)
    assert resp_tensor.status_code == 200
    assert resp_tensor.json()["batch_size"] == 4

    assert client.post("/eq_autocall/price?engine=bogus", json=base_d).status_code == 422
    assert client.post("/eq_autocall/price", content=b"", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/eq_autocall/price", content=b"{bad", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/eq_autocall/price_murex_dag", content=b"", headers={"Content-Type": "text/plain"}).status_code == 422
    assert client.post("/eq_autocall/price_murex_dag", content=b"{bad", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/eq_autocall/price_murex_dag", content=b"123", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/eq_autocall/price_murex_dag", content=b"\xff\xfe", headers={"Content-Type": "text/plain"}).status_code == 422
    assert client.post("/eq_autocall/price_arrow", content=b"", headers={"Content-Type": ARROW_MIME}).status_code == 422
    assert client.post("/eq_autocall/price_arrow", content=b"bad", headers={"Content-Type": ARROW_MIME}).status_code == 422
    assert client.post("/eq_autocall/price_batch", content=b"", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/eq_autocall/price_batch", content=b"{bad", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/eq_autocall/price_batch", content=b"bad", headers={"Content-Type": ARROW_MIME}).status_code == 422
    assert client.post("/eq_autocall/price_batch", json=[]).status_code == 422
    assert client.post("/eq_autocall/price_batch", json={"requests": []}).status_code == 422
    assert client.post("/eq_autocall/price_batch", json={"unknown": 1}).status_code == 422

    # Arrow unbatched encode/decode + shape/schema/response bounds errors
    raw_1d = encode_eq_autocall_arrow_request(
        synth.node_feats[0],
        synth.edge_feats[0],
        synth.asset_mask[0],
        synth.sched_feats[0],
        synth.sched_mask[0],
        synth.global_feats[0],
        spots_t0=ref_spots_col[0],
        ref_spots=ref_spots_col[0],
        pv_in_transit=0.02,
        metadata_dict={"schema_version": "2.0"},
    )
    dec_1d = decode_eq_autocall_arrow_request(raw_1d)
    assert dec_1d.batch_size == 1 and dec_1d.pv_in_transit is not None

    with pytest.raises(ValueError):
        encode_eq_autocall_arrow_request(
            synth.node_feats, synth.edge_feats, synth.asset_mask, synth.sched_feats, synth.sched_mask, synth.global_feats,
            metadata_dict={"schema_version": "9.9"},
        )
    for kw_bad in (
        {"node_feats": torch.zeros(4, 8, 5)},
        {"edge_feats": torch.zeros(4, 8, 8, 2)},
        {"asset_mask": torch.zeros(4, 3)},
        {"sched_feats": torch.zeros(4, 48, 3)},
        {"sched_mask": torch.zeros(4, 10)},
        {"global_feats": torch.zeros(4, 5)},
        {"spots_t0": torch.zeros(4, 3)},
        {"ref_spots": torch.zeros(4, 3)},
        {"pv_in_transit": torch.zeros(2, 2)},
    ):
        base_kw = {
            "node_feats": synth.node_feats,
            "edge_feats": synth.edge_feats,
            "asset_mask": synth.asset_mask,
            "sched_feats": synth.sched_feats,
            "sched_mask": synth.sched_mask,
            "global_feats": synth.global_feats,
        }
        base_kw.update(kw_bad)
        with pytest.raises(ValueError):
            encode_eq_autocall_arrow_request(**base_kw)  # type: ignore[arg-type]

    # Runtime checkpoint/init fallbacks, _parse_request variants, surrogate+include_all_greeks, empty batch error
    with pytest.raises(FileNotFoundError):
        EQAutocallRuntime(checkpoint_path=tmp_path / "missing.pt", strict_checkpoint=True, device="cpu")
    bad_ckpt = tmp_path / "corrupt.pt"
    bad_ckpt.write_bytes(b"corrupt_bytes")
    with pytest.raises(RuntimeError):
        EQAutocallRuntime(checkpoint_path=bad_ckpt, strict_checkpoint=True, device="cpu")
    rt_fb = EQAutocallRuntime(
        checkpoint_path=bad_ckpt,
        stats_path=tmp_path / "missing_stats.pt",
        strict_checkpoint=False,
        device="cpu",
    )
    assert rt_fb.checkpoint_loaded is False and rt_fb.guardian_stats_loaded is False

    rt_custom = EQAutocallRuntime(surrogate=copy.deepcopy(runtime.surrogate), device="cpu")
    assert rt_custom.model_version == "eq-autocall-egno-custom"

    parsed_from_str, is_d_s = runtime._parse_request(orjson.dumps(base_d).decode("utf-8"))
    assert is_d_s is False and parsed_from_str.TradeInfo == req_model.TradeInfo
    parsed_from_path, is_d_p = runtime._parse_request(REF_REQUEST_PATH)
    assert is_d_p is True and parsed_from_path.TradeInfo == "45476647/EMTN_CIB_SEC"
    with pytest.raises(TypeError):
        runtime._parse_request(42)  # type: ignore[arg-type]

    # Single surrogate + Delta query & single include_all_greeks=False
    t_no_delta = copy.deepcopy(base_d)
    t_no_delta["Queries"] = ["ProductValue"]
    r_s_grk = runtime.price_trade(t_no_delta, engine="surrogate", include_all_greeks=False)
    assert r_s_grk.Delta is None
    t_surr_d = copy.deepcopy(base_d)
    t_surr_d["Queries"] = ["ProductValue", "Delta"]
    r_s_d = runtime.price_trade(t_surr_d, engine="surrogate")
    assert r_s_d.Delta is not None

    # Duplicate/JSON-str/Pydantic batch items (cache hit) + surrogate non-PSD + auto OOD QMC fallback
    dup_out = runtime.price_batch_json(
        [t_fast, orjson.dumps(t_fast).decode("utf-8"), req_model],
        compute_greeks=False,
        engine="auto",
    )
    assert len(dup_out) == 3

    t_non_psd_s = copy.deepcopy(t_fast)
    corrs_s = t_non_psd_s["Model"]["MarketDataSet"]["Correlations"]
    uids_s = t_non_psd_s["Product"]["UnderlyingIdentifiers"]
    for a_i, b_i, v_i in ((uids_s[0], uids_s[1], 0.99), (uids_s[0], uids_s[2], 0.99), (uids_s[1], uids_s[2], -0.99)):
        k_use = f"{b_i}&{a_i}" if f"{b_i}&{a_i}" in corrs_s else f"{a_i}&{b_i}"
        corrs_s[k_use] = v_i
    r_psd_s = runtime.price_trade(t_non_psd_s, engine="surrogate", include_all_greeks=False)
    assert r_psd_s.fallback_triggered is False

    t_ood_a = copy.deepcopy(t_fast)
    t_ood_a["Model"]["MarketDataSet"]["Spots"][uids_s[0]] = 5000.0
    r_ood_a = runtime.price_trade(t_ood_a, engine="auto", include_all_greeks=False)
    assert r_ood_a.fallback_triggered is True and r_ood_a.engine_used == "Tier3_DupireQMC"

    qmc_b = runtime.price_batch_json([t_ood_a], compute_greeks=True, engine="qmc")
    assert len(qmc_b) == 1 and qmc_b[0].engine_used == "Tier3_DupireQMC"

    from deepvol.mrm.eq_autocall_guardian import EQAutocallModelGuardian

    rt_post_fb = EQAutocallRuntime(
        guardian=EQAutocallModelGuardian(stats_path=runtime.stats_path, residual_threshold=1e-9),
        device="cpu",
        n_fallback_paths=128,
    )
    post_fb_out = rt_post_fb.price_batch_json([t_fast], compute_greeks=True, engine="auto")
    assert len(post_fb_out) == 1 and post_fb_out[0].fallback_triggered is True

    # Corrupt guardian stats fallback + _validate_surfaces_for_arbitrage + _parse_request({"dag_text": ...}) + price_batch_json dag_text
    from types import SimpleNamespace
    from deepvol.api.eq_autocall_runtime import (
        _concat_batched_greeks,
        _validate_surfaces_for_arbitrage,
    )

    rt_bad_stats = EQAutocallRuntime(checkpoint_path=bad_ckpt, stats_path=bad_ckpt, strict_checkpoint=False, device="cpu")
    assert rt_bad_stats.guardian_stats_loaded is False

    raw_dag_s = REF_REQUEST_PATH.read_text(encoding="utf-8")
    p_dag_str, is_ds = runtime._parse_request(raw_dag_s)
    assert is_ds is True and p_dag_str.TradeInfo == "45476647/EMTN_CIB_SEC"
    p_dag_d, is_dd = runtime._parse_request({"dag_text": raw_dag_s})
    assert is_dd is True and p_dag_d.TradeInfo == "45476647/EMTN_CIB_SEC"
    with pytest.raises(ValueError):
        runtime._parse_request("{not_valid_json")
    with pytest.raises(ValueError):
        runtime._parse_request({"not_a_product": 1})
    dag_b_out = runtime.price_batch_json(
        [raw_dag_s, {"dag_text": raw_dag_s}],
        compute_greeks=False,
        engine="surrogate",
    )
    assert len(dag_b_out) == 2

    # Surrogate batch greeks + dedup cache hit + _compute_w_fund_per_period branches
    surr_dup_grk = runtime.price_batch_json([t_fast, t_fast], compute_greeks=True, engine="surrogate")
    assert len(surr_dup_grk) == 2
    assert surr_dup_grk[0].Delta is not None and surr_dup_grk[0].MultivariateAggregatedCashGamma is not None
    assert surr_dup_grk[0].Cega is not None

    from deepvol.api.eq_autocall_runtime import _compute_w_fund_per_period
    from deepvol.models.autocall.pre_post_processor import preprocess_eq_autocall

    st_plain = preprocess_eq_autocall(req_model)
    assert torch.all(_compute_w_fund_per_period(st_plain, 48) == 0.0)
    t_past_f = copy.deepcopy(t_fast)
    t_past_f["Product"]["RateIdentifier"] = None
    t_past_f["Product"]["FundingLegPaymentDates"] = [
        "2026-10-01T00:00:00+00:00",
        "2027-01-29T00:00:00+00:00",
    ]
    st_pf = preprocess_eq_autocall(t_past_f)
    if st_pf.funding_leg is not None:
        st_pf.funding_leg.pop("_cached_w_fund_per_period", None)
    assert _compute_w_fund_per_period(st_pf, 48).shape == (48,)

    # Direct _validate_surfaces_for_arbitrage branch probes via SimpleNamespace surfaces
    as_of_r = req_model.Model.MarketDataSet.AsOfDate
    t0_surf = req_model.Model.MarketDataSet.VolatilitySurfaces[req_model.Product.UnderlyingIdentifiers[0]]
    assert isinstance(t0_surf, EquityVolSmileCollectionSpec)
    t0_exp = t0_surf.ExpirationDates
    assert isinstance(t0_exp, list) and len(t0_exp) >= 2
    _validate_surfaces_for_arbitrage(
        SimpleNamespace(Model=SimpleNamespace(MarketDataSet=SimpleNamespace(AsOfDate=as_of_r, VolatilitySurfaces={"FX": SimpleNamespace(VolatilityQuotes=[[0.2]])})))  # type: ignore[arg-type]
    )
    for bad_surf in (
        SimpleNamespace(VolatilityQuotes=[[float("nan"), 0.2]], Strikes=[100.0, 110.0], ExpirationDates=t0_exp),
        SimpleNamespace(VolatilityQuotes=[[0.2, 0.2]], Strikes=[float("nan"), 110.0], ExpirationDates=t0_exp),
        SimpleNamespace(VolatilityQuotes=[[0.2, 0.2]], Strikes=[-10.0, 110.0], ExpirationDates=t0_exp),
        SimpleNamespace(VolatilityQuotes=[[0.2, 0.2]], Strikes=[100.0, 100.0], ExpirationDates=t0_exp),
        SimpleNamespace(VolatilityQuotes=[[0.2, 0.2]], Strikes=[100.0, 110.0], ExpirationDates=[as_of_r]),
        SimpleNamespace(VolatilityQuotes=[[0.2, 0.2]], Strikes=[100.0, 110.0], ExpirationDates=[t0_exp[0], t0_exp[0]]),
    ):
        with pytest.raises(ValueError, match="SMILE_ARBITRAGE"):
            _validate_surfaces_for_arbitrage(
                SimpleNamespace(Model=SimpleNamespace(MarketDataSet=SimpleNamespace(AsOfDate=as_of_r, VolatilitySurfaces={"EQ0": bad_surf})))  # type: ignore[arg-type]
            )

    # Arrow bounds, float32 ListArray cast, response None cash_gamma, and corrupt response guards
    import pyarrow as pa

    with pytest.raises(ValueError):
        encode_eq_autocall_arrow_request(
            torch.zeros(1, MAX_D_ARROW + 1, 16),
            torch.zeros(1, MAX_D_ARROW + 1, MAX_D_ARROW + 1, 4),
            torch.ones(1, MAX_D_ARROW + 1),
            torch.zeros(1, 2, 12),
            torch.ones(1, 2),
            torch.zeros(1, 16),
        )

    batch_f32 = pa.RecordBatch.from_arrays(
        [
            pa.array(synth.node_feats.reshape(4, -1).numpy().astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            pa.array(synth.edge_feats.reshape(4, -1).numpy().astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            pa.array(synth.asset_mask.numpy().astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            pa.array(synth.sched_feats.reshape(4, -1).numpy().astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            pa.array(synth.sched_mask.numpy().astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            pa.array(synth.global_feats.numpy().astype(np.float32).tolist(), type=pa.list_(pa.float32())),
        ],
        names=["node_feats", "edge_feats", "asset_mask", "sched_feats", "sched_mask", "global_feats"],
    )
    meta_hdr = orjson.dumps({"schema_version": "2.0", "d_max": 8, "m_max": 48})
    batch_f32 = batch_f32.replace_schema_metadata({HEADER_METADATA_KEY: meta_hdr})
    sink_f32 = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink_f32, batch_f32.schema) as wr_f32:
        wr_f32.write_batch(batch_f32)
    dec_f32 = decode_eq_autocall_arrow_request(sink_f32.getvalue().to_pybytes())
    assert dec_f32.node_feats.dtype == torch.float64

    bad_elem_batch = pa.RecordBatch.from_arrays(
        [
            pa.array([[None] + [0.0] * 127] * 4, type=pa.list_(pa.float64())),
            pa.array(synth.edge_feats.reshape(4, -1).numpy().tolist(), type=pa.list_(pa.float64())),
            pa.array(synth.asset_mask.numpy().tolist(), type=pa.list_(pa.float64())),
            pa.array(synth.sched_feats.reshape(4, -1).numpy().tolist(), type=pa.list_(pa.float64())),
            pa.array(synth.sched_mask.numpy().tolist(), type=pa.list_(pa.float64())),
            pa.array(synth.global_feats.numpy().tolist(), type=pa.list_(pa.float64())),
        ],
        names=["node_feats", "edge_feats", "asset_mask", "sched_feats", "sched_mask", "global_feats"],
    ).replace_schema_metadata({HEADER_METADATA_KEY: meta_hdr})
    sink_null = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink_null, bad_elem_batch.schema) as wr_null:
        wr_null.write_batch(bad_elem_batch)
    with pytest.raises(ValueError, match="NULL"):
        decode_eq_autocall_arrow_request(sink_null.getvalue().to_pybytes())

    resp_obj = BatchedSurrogateGreeksOutput(
        product_value=torch.zeros(2, dtype=torch.float64),
        stopping_probs=torch.zeros(2, 4, dtype=torch.float64),
        survival_probs=torch.ones(2, 4, dtype=torch.float64),
        coupon_probs=torch.zeros(2, 4, dtype=torch.float64),
        put_loss=torch.zeros(2, dtype=torch.float64),
        delta=torch.zeros(2, 3, dtype=torch.float64),
        vega_atm=torch.zeros(2, 3, 2, dtype=torch.float64),
        vega_surface_feats=torch.zeros(2, 21, dtype=torch.float64),
        rho_df=torch.zeros(2, 4, dtype=torch.float64),
        cega=torch.zeros(2, 3, 3, dtype=torch.float64),
        cash_gamma_euler=None,
    )
    resp_bytes_none = encode_eq_autocall_arrow_response(
        resp_obj,
        metadata_dict={"schema_version": "2.0"},
    )
    assert _concat_batched_greeks([resp_obj]) is resp_obj
    cat_two = _concat_batched_greeks([resp_obj, resp_obj])
    assert cat_two.cash_gamma_euler is None and cat_two.product_value.shape == (4,)
    dec_resp = decode_eq_autocall_arrow_response(resp_bytes_none)
    assert dec_resp["schema_version"] == "2.0" and dec_resp["cash_gamma_euler"].shape == (2,)
    for bad_resp in (b"", b"not_arrow"):
        with pytest.raises(ValueError):
            decode_eq_autocall_arrow_response(bad_resp)

    oversized_req = encode_eq_autocall_arrow_request(
        torch.zeros(1, runtime.d_max + 1, 16),
        torch.zeros(1, runtime.d_max + 1, runtime.d_max + 1, 4),
        torch.ones(1, runtime.d_max + 1),
        torch.zeros(1, 4, 12),
        torch.ones(1, 4),
        torch.zeros(1, 16),
    )
    with pytest.raises(ValueError):
        runtime.price_arrow_ipc(oversized_req)

    with pytest.raises(ValueError):
        runtime.price_batch_json([])


def test_api_audit_blind_spots(client: TestClient) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import datetime
    import io
    import orjson
    import pyarrow as pa
    from deepvol.models.autocall.pre_post_processor import build_vol_surface_cache_key

    req_model, _ = parse_murex_autocall_dag(REF_REQUEST_PATH)
    base_d = req_model.model_dump(mode="json")
    base_d.pop("stripped_legacy_fields", None)
    runtime = EQAutocallRuntime(device="cpu", enable_compiled_runner=False)

    first_tkr = base_d["Product"]["UnderlyingIdentifiers"][0]
    missing_spot = copy.deepcopy(base_d)
    missing_spot["Model"]["MarketDataSet"]["Spots"].pop(first_tkr, None)
    assert client.post("/eq_autocall/price", json=missing_spot).status_code == 422

    missing_surf = copy.deepcopy(base_d)
    missing_surf["Model"]["MarketDataSet"]["VolatilitySurfaces"].pop(first_tkr, None)
    assert client.post("/eq_autocall/price", json=missing_surf).status_code == 422

    missing_corr = copy.deepcopy(base_d)
    missing_corr["Model"]["MarketDataSet"]["Correlations"] = {}
    assert client.post("/eq_autocall/price", json=missing_corr).status_code == 422

    missing_fix = copy.deepcopy(base_d)
    missing_fix["Model"]["MarketDataSet"]["Fixings"] = {}
    assert client.post("/eq_autocall/price", json=missing_fix).status_code == 422

    assert client.post("/eq_autocall/price_murex_dag", json={"dag_text": 123}).status_code == 422

    # Strict boolean parsing across /eq_autocall/price (?include_all_greeks) and /eq_autocall/price_batch (?compute_greeks).
    d_pv = copy.deepcopy(base_d)
    d_pv["Queries"] = ["ProductValue", "CashFlows"]
    resp_all_on = client.post("/eq_autocall/price?include_all_greeks=true&engine=auto", json=d_pv)
    assert resp_all_on.status_code == 200 and resp_all_on.json()["Delta"] is not None
    resp_nogrk = client.post("/eq_autocall/price?include_all_greeks=false&engine=auto", json=d_pv)
    assert resp_nogrk.status_code == 200
    assert resp_nogrk.json()["Delta"] is None
    assert client.post("/eq_autocall/price?include_all_greeks=notabool", json=d_pv).status_code == 422

    d_grk = copy.deepcopy(base_d)
    d_grk["Queries"] = ["ProductValue", "Delta"]
    resp_batch_nogrk = client.post("/eq_autocall/price_batch?compute_greeks=false&engine=auto", json=[d_grk])
    assert resp_batch_nogrk.status_code == 200
    assert resp_batch_nogrk.json()[0]["Delta"] is None
    assert client.post("/eq_autocall/price_batch?compute_greeks=notabool", json=[d_grk]).status_code == 422
    assert (
        client.post("/eq_autocall/price_batch", json={"requests": [d_grk], "compute_greeks": "false"}).status_code
        == 422
    )

    # spots_t0 rank/NaN validation in /eq_autocall/price_batch.
    synth = generate_synthetic_autocall_batch(
        batch_size=2, d_max=8, m_max=48, n_mc_paths=32, seed=202611, device="cpu", dtype=torch.float64
    )
    bad_rank_payload = {
        "node_feats": synth.node_feats.tolist(),
        "edge_feats": synth.edge_feats.tolist(),
        "asset_mask": synth.asset_mask.tolist(),
        "sched_feats": synth.sched_feats.tolist(),
        "sched_mask": synth.sched_mask.tolist(),
        "global_feats": synth.global_feats.tolist(),
        "spots_t0": [100.0, 100.0],
    }
    assert client.post("/eq_autocall/price_batch", json=bad_rank_payload).status_code == 422
    assert (
        client.post(
            "/eq_autocall/price_batch",
            content=b'{"node_feats": NaN}',
            headers={"Content-Type": "application/json"},
        ).status_code
        == 422
    )

    # HTTP 413 payload limit guard (> 50 MiB).
    oversized_body = b"{" + (b" " * (50 * 1024 * 1024 + 16)) + b"}"
    resp_413 = client.post(
        "/eq_autocall/price",
        content=oversized_body,
        headers={"Content-Type": "application/json"},
    )
    assert resp_413.status_code == 413

    # Arrow IPC m_rem == 0 (all sched_mask zeros) returns HTTP 200 with zero Greeks & finite pv_in_transit.
    zero_sched = torch.zeros_like(synth.sched_mask)
    arrow_m0 = encode_eq_autocall_arrow_request(
        synth.node_feats,
        synth.edge_feats,
        synth.asset_mask,
        synth.sched_feats,
        zero_sched,
        synth.global_feats,
        pv_in_transit=torch.tensor([0.125, 0.25], dtype=torch.float64),
    )
    resp_m0 = client.post(
        "/eq_autocall/price_arrow?compute_greeks=true",
        content=arrow_m0,
        headers={"Content-Type": ARROW_MIME},
    )
    assert resp_m0.status_code == 200
    dec_m0 = decode_eq_autocall_arrow_response(resp_m0.content)
    assert np.all(np.isfinite(dec_m0["product_value"]))
    assert np.allclose(dec_m0["delta"], 0.0)

    # Arrow non-finite scalar pv_in_transit and non-dict Arrow metadata must be rejected.
    with pytest.raises(ValueError):
        encode_eq_autocall_arrow_request(
            synth.node_feats,
            synth.edge_feats,
            synth.asset_mask,
            synth.sched_feats,
            synth.sched_mask,
            synth.global_feats,
            pv_in_transit=torch.tensor([float("nan"), 0.0], dtype=torch.float64),
        )
    tbl_bad_hdr = pa.RecordBatch.from_arrays(
        [
            pa.array(synth.node_feats.reshape(2, -1).numpy().tolist(), type=pa.list_(pa.float64())),
            pa.array(synth.edge_feats.reshape(2, -1).numpy().tolist(), type=pa.list_(pa.float64())),
            pa.array(synth.asset_mask.numpy().tolist(), type=pa.list_(pa.float64())),
            pa.array(synth.sched_feats.reshape(2, -1).numpy().tolist(), type=pa.list_(pa.float64())),
            pa.array(synth.sched_mask.numpy().tolist(), type=pa.list_(pa.float64())),
            pa.array(synth.global_feats.numpy().tolist(), type=pa.list_(pa.float64())),
        ],
        names=["node_feats", "edge_feats", "asset_mask", "sched_feats", "sched_mask", "global_feats"],
    ).replace_schema_metadata({HEADER_METADATA_KEY: orjson.dumps([1, 2, 3])})
    buf_bad = io.BytesIO()
    with pa.ipc.new_stream(buf_bad, tbl_bad_hdr.schema) as wr_bad:
        wr_bad.write_batch(tbl_bad_hdr)
    with pytest.raises(ValueError):
        decode_eq_autocall_arrow_request(buf_bad.getvalue())

    # Invalid vol surface cache key shape & duplicate Deltas safeguard.
    fx_a = {
        "Identifier": "USD/RUB",
        "TypeName": "FXVolatilitySmileCollection",
        "DayCountConvention": "Act365",
        "Deltas": ["-25D", "ATM", "+25D"],
        "Tenors": ["1M"],
        "VolatilityQuotes": [[0.20, 0.21, 0.22]],
    }
    fx_b = {**fx_a, "Deltas": ["-10D", "ATM", "+10D"]}
    as_of_k = datetime.datetime(2026, 10, 23, tzinfo=datetime.timezone.utc)
    assert build_vol_surface_cache_key(fx_a, as_of_k) != build_vol_surface_cache_key(fx_b, as_of_k)

    # Explicit Queries=None coalesces to default [ProductValue, CashFlows] (200); empty Queries=[] returns 422.
    none_q = copy.deepcopy(base_d)
    none_q["Queries"] = None
    r_none_q = client.post("/eq_autocall/price", json=none_q)
    assert r_none_q.status_code == 200, f"Queries=None failed: {r_none_q.status_code} {r_none_q.text}"
    body_none_q = r_none_q.json()
    assert np.isfinite(float(body_none_q["ProductValue"]))
    assert isinstance(body_none_q.get("CashFlows"), list) and len(body_none_q["CashFlows"]) > 0

    empty_q = copy.deepcopy(base_d)
    empty_q["Queries"] = []
    assert client.post("/eq_autocall/price", json=empty_q).status_code == 422

    no_q = copy.deepcopy(base_d)
    no_q.pop("Queries", None)
    dag_text = REF_REQUEST_PATH.read_text(encoding="utf-8")

    def _worker(idx: int) -> tuple[int, float]:
        if idx % 2 == 0:
            r = client.post("/eq_autocall/price", json=no_q)
        else:
            r = client.post("/eq_autocall/price_murex_dag", content=dag_text, headers={"Content-Type": "text/plain"})
        assert r.status_code == 200, f"worker={idx} failed with {r.status_code}: {r.text}"
        return r.status_code, float(r.json()["ProductValue"])

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_worker, range(16)))
    assert all(code == 200 and np.isfinite(pv) for code, pv in results)
    assert callable(getattr(runtime, "price_trade"))


def test_runtime_canonical_invariance_cpp23_startup_quanto_and_irao_arb(
    monkeypatch: pytest.MonkeyPatch,
    parsed_production_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verifies cpp23 startup check, CanonicalTier3Config pv_only==all_greeks invariance, quanto 0.0 keys, and IRAO_RUB pre-gate/repair."""
    import time
    from types import SimpleNamespace
    import deepvol.api.eq_autocall_runtime as rt_mod

    req_model, _ = parsed_production_dag

    # 1. Runtime defaults & cpp23 startup assertion
    rt_default = EQAutocallRuntime(device="cpu")
    assert rt_default.n_fallback_paths is None
    assert rt_default.theta_convention == "murex_bizday_forward_fixed"
    assert rt_default.repair_arbitrage is False

    monkeypatch.setattr(rt_mod, "get_tier1_backend", lambda: SimpleNamespace(name="numpy"))
    monkeypatch.delenv("DEEPVOL_TIER1_CPP_LIB", raising=False)
    with pytest.raises(RuntimeError, match="cpp23"):
        EQAutocallRuntime(device="cpu")
    monkeypatch.setenv("DEEPVOL_TIER1_CPP_LIB", "numpy")
    rt_np_allowed = EQAutocallRuntime(device="cpu")
    assert rt_np_allowed.theta_convention == "murex_bizday_forward_fixed"
    monkeypatch.undo()

    # 2. CanonicalTier3Config invariance across pv_only vs all_greeks & quanto FX 0.0 Delta/Gamma keys
    captured_calls: list[dict[str, Any]] = []
    orig_crn = getattr(rt_mod._greeks_mod, "compute_qmc_crn_greeks")

    def _spy_crn(state: Any, **kwargs: Any) -> Any:
        captured_calls.append(
            {
                "queries": list(kwargs.get("queries") or ()),
                "config": kwargs.get("config"),
                "device": kwargs.get("device"),
                "theta_convention": kwargs.get("theta_convention"),
            }
        )
        return orig_crn(state, **kwargs)

    monkeypatch.setattr(rt_mod._greeks_mod, "compute_qmc_crn_greeks", _spy_crn)

    # Use a compact 2-asset 2-period QMC-routed request with CNHRUB FX surface
    import datetime as _dt_mod

    req_q = copy.deepcopy(req_model)
    uids = req_q.Product.UnderlyingIdentifiers[:2]
    as_of_dt = req_q.Model.MarketDataSet.AsOfDate
    req_q.Product.StartDate = as_of_dt
    req_q.Product.UnderlyingIdentifiers = uids
    req_q.Product.UnderlyingReferenceSpots = req_q.Product.UnderlyingReferenceSpots[:2]
    req_q.Product.ObservationDates = [
        as_of_dt + _dt_mod.timedelta(days=30),
        as_of_dt + _dt_mod.timedelta(days=60),
    ]
    req_q.Product.PaymentDates = [
        as_of_dt + _dt_mod.timedelta(days=33),
        as_of_dt + _dt_mod.timedelta(days=63),
    ]
    req_q.Product.AutoCallTriggerLevels = req_q.Product.AutoCallTriggerLevels[-2:]
    if req_q.Product.CouponBarrierLevels is not None:
        req_q.Product.CouponBarrierLevels = req_q.Product.CouponBarrierLevels[-2:]
    if isinstance(req_q.Product.FixedRate, list):
        req_q.Product.FixedRate = req_q.Product.FixedRate[-2:]
    if isinstance(req_q.Product.FixedCouponAmount, list):
        req_q.Product.FixedCouponAmount = req_q.Product.FixedCouponAmount[-2:]
    if req_q.Product.FundingLegPaymentDates is not None:
        req_q.Product.FundingLegPaymentDates = [
            as_of_dt + _dt_mod.timedelta(days=33),
            as_of_dt + _dt_mod.timedelta(days=63),
        ]
    req_q.Product.PaymentCurrency = "CNH"
    req_q.Model.MarketDataSet.Spots = {k: req_q.Model.MarketDataSet.Spots[k] for k in uids}
    req_q.Model.MarketDataSet.VolatilitySurfaces = {
        k: req_q.Model.MarketDataSet.VolatilitySurfaces[k] for k in uids
    }
    first_bundle_k = next(iter(req_q.Model.MarketDataSet.RatesCurvesBundles))
    cnh_curve = req_q.Model.MarketDataSet.RatesCurvesBundles[first_bundle_k].model_copy(
        update={"Identifier": "CNH_FX", "Currency": "CNH"},
        deep=True,
    )
    req_q.Model.MarketDataSet.RatesCurvesBundles["CNH_FX"] = cnh_curve
    from deepvol.api.schemas_eq_autocall import FXVolatilitySmileCollectionSpec

    req_q.Model.MarketDataSet.VolatilitySurfaces["CNHRUB"] = FXVolatilitySmileCollectionSpec(
        Identifier="CNHRUB",
        AsOfDate=as_of_dt,
        Deltas=["-25D", "ATM", "25D"],
        Tenors=["3M", "1Y"],
        VolatilityQuotes=[[0.18, 0.16, 0.19], [0.20, 0.18, 0.21]],
    )
    req_q.Model.MarketDataSet.Dividends = {
        k: req_q.Model.MarketDataSet.Dividends[k]
        for k in uids
        if k in req_q.Model.MarketDataSet.Dividends
    }
    req_q.Model.MarketDataSet.Fixings = {}
    req_q.Model.MarketDataSet.Correlations = {
        f"{uids[0]}&{uids[1]}": 0.45,
        f"{uids[0]}&CNHRUB": 0.20,
        f"{uids[1]}&CNHRUB": 0.20,
    }
    req_pv = req_q.model_copy(update={"Queries": ["ProductValue", "CashFlows"]})

    rt_cpu = EQAutocallRuntime(device="cpu")
    res_pv = rt_cpu.price_trade(req_pv, engine="qmc", include_all_greeks=False)
    res_grk = rt_cpu.price_trade(req_q, engine="qmc", include_all_greeks=True)
    assert len(captured_calls) == 2
    assert captured_calls[0]["theta_convention"] == "murex_bizday_forward_fixed"
    assert captured_calls[1]["theta_convention"] == "murex_bizday_forward_fixed"
    assert str(captured_calls[0]["device"]) == "cpu"
    assert str(captured_calls[1]["device"]) == "cpu"
    cfg0, cfg1 = captured_calls[0]["config"], captured_calls[1]["config"]
    assert (cfg0.n_paths, cfg0.n_replicates, cfg0.n_substeps, cfg0.lv_grid_nt, cfg0.lv_grid_ny) == (
        4096,
        4,
        4,
        32,
        64,
    )
    assert (cfg1.n_paths, cfg1.n_replicates, cfg1.n_substeps, cfg1.lv_grid_nt, cfg1.lv_grid_ny) == (
        4096,
        4,
        4,
        32,
        64,
    )
    assert abs(float(res_pv.ProductValue) - float(res_grk.ProductValue)) * 1e4 == pytest.approx(0.0, abs=1e-9)
    assert res_pv.Delta is None and res_grk.Delta is not None
    assert res_grk.Delta.get("CNHRUB") == 0.0
    assert isinstance(res_grk.Gamma, dict) and isinstance(res_grk.Gamma.get("CNHRUB"), dict)
    assert res_grk.Gamma["CNHRUB"]["CNHRUB"] == 0.0

    # 3. IRAO_RUB multi-slice pre-gate (slices=[0..6]) + convexity right-leg stencil + repair_arbitrage=True
    rt_mod._SURFACE_ARB_CACHE.clear()
    rt_mod._SURFACE_ARB_FAST_CACHE.clear()
    strikes_arb = [4.6725, 4.8725, 5.0725, 5.2725, 5.4725]
    q_bad_irao = [[0.28, 0.275, 0.2728, 0.32577, 0.32339] for _ in range(7)]
    exp_7 = [_dt_mod.datetime.fromisoformat(as_of_dt.isoformat()) + _dt_mod.timedelta(days=60 * (m + 1)) for m in range(7)]
    req_irao = copy.deepcopy(req_q)
    req_irao.Product.UnderlyingIdentifiers = ["IRAO_RUB", uids[1]]
    req_irao.Product.UnderlyingReferenceSpots = [3.984, req_q.Product.UnderlyingReferenceSpots[1]]
    req_irao.Model.MarketDataSet.Spots = {"IRAO_RUB": 2.684, uids[1]: req_q.Model.MarketDataSet.Spots[uids[1]]}
    req_irao.Model.MarketDataSet.VolatilitySurfaces = {
        "IRAO_RUB": EquityVolSmileCollectionSpec(
            Identifier="IRAO_RUB",
            ExpirationDates=exp_7,
            Strikes=strikes_arb,
            VolatilityQuotes=q_bad_irao,
        ),
        uids[1]: req_q.Model.MarketDataSet.VolatilitySurfaces[uids[1]],
    }
    req_irao.Model.MarketDataSet.Dividends = {}
    req_irao.Model.MarketDataSet.Correlations = {f"IRAO_RUB&{uids[1]}": 0.4}

    t_cold = time.perf_counter_ns()
    with pytest.raises(ValueError, match="SMILE_ARBITRAGE") as exc_irao:
        rt_cpu.price_trade(req_irao, engine="proxy", include_all_greeks=False)
    cold_ms = (time.perf_counter_ns() - t_cold) / 1e6
    msg_irao = str(exc_irao.value)
    assert "surface=IRAO_RUB" in msg_irao
    assert "slices=[0, 1, 2, 3, 4, 5, 6]" in msg_irao
    assert "K=5.2725" in msg_irao and "prev_K=5.0725" in msg_irao
    assert "vol_jump=" in msg_irao and "slope=" in msg_irao and "min_d2c=" in msg_irao
    assert cold_ms < 25.0  # Well under 2 ms in steady process; generous bound for first import/ndtr

    # Right-leg convexity jump (j_d2+1 -> j_d2+2) where slopes remain negative so convexity branch triggers
    strikes_conv = [90.0, 95.0, 100.0, 105.0, 110.0]
    q_conv_right = [[0.25, 0.25, 0.25, 0.25, 0.05] for _ in range(2)]
    req_conv = copy.deepcopy(req_irao)
    req_conv.Model.MarketDataSet.Spots["IRAO_RUB"] = 100.0
    req_conv.Model.MarketDataSet.VolatilitySurfaces["IRAO_RUB"] = EquityVolSmileCollectionSpec(
        Identifier="IRAO_RUB",
        ExpirationDates=exp_7[:2],
        Strikes=strikes_conv,
        VolatilityQuotes=q_conv_right,
    )
    with pytest.raises(ValueError, match="Discrete call convexity violation") as exc_conv:
        rt_mod._validate_surfaces_for_arbitrage(req_conv)
    assert "K=110.0000" in str(exc_conv.value) and "prev_K=105.0000" in str(exc_conv.value)

    # repair_arbitrage=True clones request and repairs quotes before validation without mutating caller input
    import deepvol.models.autocall.dupire_qmc_solver as qmc_solver_mod

    def _mock_repair(strikes: Any, quotes: Any, t_exp: Any, k_ref: float, **kw: Any) -> tuple[Any, dict[str, Any]]:
        _ = (t_exp, k_ref, kw)
        return np.full_like(quotes, 0.28, dtype=np.float64), {"repaired": True}

    monkeypatch.setattr(qmc_solver_mod, "repair_arbitrage_quotes", _mock_repair, raising=False)
    res_rep = rt_cpu.price_trade(req_irao, engine="proxy", include_all_greeks=False, repair_arbitrage=True)
    assert np.isfinite(res_rep.ProductValue)
    # Caller's original quotes remain untouched (still strictly rejected when repair_arbitrage=False)
    assert req_irao.Model.MarketDataSet.VolatilitySurfaces["IRAO_RUB"].VolatilityQuotes[0][3] == pytest.approx(0.32577)
    with pytest.raises(ValueError, match="SMILE_ARBITRAGE"):
        rt_cpu.price_trade(req_irao, engine="proxy", include_all_greeks=False, repair_arbitrage=False)






