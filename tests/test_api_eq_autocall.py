"""test_api_eq_autocall.py — Integration tests for Milestone M5 (Task 5.1) Runtime, FastAPI & Arrow IPC.

Covers:
  - Gate G4.6 (`test_G4_6_legacy_murex_dag_end_to_end_ingestion`):
      End-to-end ingestion of `/home/execorn/programming/derivatives/eq_autocallable_request.txt`
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
    MonitoringIndexEnum,
    parse_murex_autocall_dag,
)
from deepvol.api.server import app
from deepvol.calibration.autocall_sobol_generator import (
    generate_synthetic_autocall_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_REQUEST_PATH = REPO_ROOT / "eq_autocallable_request.txt"
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
    expected_surf_shapes = {
        t: (
            len(req_model.Model.MarketDataSet.VolatilitySurfaces[t].ExpirationDates),
            len(req_model.Model.MarketDataSet.VolatilitySurfaces[t].Strikes),
        )
        for t in expected_tickers
    }
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
    assert decoded_req.node_feats.shape == (64, 8, 14) and decoded_req.node_feats.dtype == torch.float64
    assert decoded_req.edge_feats.shape == (64, 8, 8, 4) and decoded_req.edge_feats.dtype == torch.float64
    assert decoded_req.asset_mask.shape == (64, 8) and decoded_req.asset_mask.dtype == torch.float64
    assert decoded_req.sched_feats.shape == (64, 48, 12) and decoded_req.sched_feats.dtype == torch.float64
    assert decoded_req.sched_mask.shape == (64, 48) and decoded_req.sched_mask.dtype == torch.float64
    assert decoded_req.global_feats.shape == (64, 10) and decoded_req.global_feats.dtype == torch.float64
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
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        decoded_resp["delta"],
        direct_out.delta.detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
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
    """Gate G4.7: Verify complete Gate ID coverage (G1.1–G4.7) across verification test suites."""
    if SPEC_PATH.is_file():
        spec_text = SPEC_PATH.read_text(encoding="utf-8")
        missing_in_spec = validate_traceability_matrix_completeness(spec_text)
        assert missing_in_spec == [], f"Gates missing from specification: {missing_in_spec}"

    # Verify all core EQ Autocall test suites exist on disk
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
    ]
    for tf in required_test_files:
        assert tf.is_file(), f"Missing required verification test suite: {tf}"

    if TRACEABILITY_MATRIX_PATH.is_file():
        matrix_text = TRACEABILITY_MATRIX_PATH.read_text(encoding="utf-8")
        missing_in_matrix = validate_traceability_matrix_completeness(matrix_text)
        assert missing_in_matrix == [], (
            f"Traceability matrix {TRACEABILITY_MATRIX_PATH} is missing gates: {missing_in_matrix}"
        )


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


