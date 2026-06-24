"""
Tests for §P2-B1 FastAPI REST pricing endpoint.

All tests use the FastAPI TestClient (synchronous), with the FNO model
and normalizers loaded from real artifact files where they exist.
Endpoints that require network (Deribit snapshot) are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fastapi.testclient import TestClient
from deepvol.api.server import app, _MODEL_STATE

# ── Fixtures ───────────────────────────────────────────────────────────────────

VALID_PARAMS = {
    "kappa": 1.5,
    "theta": 0.08,
    "sigma": 0.5,
    "rho":   -0.7,
    "v0":    0.08,
    "H":     0.08,
}


@pytest.fixture(scope="module")
def client():
    """TestClient with the FastAPI app."""
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_schema(client):
    body = client.get("/health").json()
    assert "status" in body
    assert "model_loaded" in body
    assert "uptime_s" in body
    assert "device" in body
    assert body["status"] == "ok"
    assert isinstance(body["uptime_s"], float)


# ── /iv_surface ────────────────────────────────────────────────────────────────

def test_iv_surface_valid(client):
    resp = client.post("/iv_surface", json=VALID_PARAMS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "surface" in body
    assert len(body["surface"]) == 8          # 8 maturities
    assert len(body["surface"][0]) == 11      # 11 strikes
    assert len(body["T_grid"]) == 8
    assert len(body["K_grid"]) == 11


def test_iv_surface_all_positive(client):
    resp = client.post("/iv_surface", json=VALID_PARAMS)
    surface = resp.json()["surface"]
    for row in surface:
        for val in row:
            assert val > 0.0, f"IV should be positive, got {val}"


def test_iv_surface_missing_field(client):
    bad = {k: v for k, v in VALID_PARAMS.items() if k != "sigma"}
    resp = client.post("/iv_surface", json=bad)
    assert resp.status_code == 422


def test_iv_surface_invalid_rho_positive(client):
    bad = {**VALID_PARAMS, "rho": 0.1}   # rho must be <= 0
    resp = client.post("/iv_surface", json=bad)
    assert resp.status_code == 422


def test_iv_surface_invalid_sigma_zero(client):
    bad = {**VALID_PARAMS, "sigma": 0.0}   # sigma must be > 0
    resp = client.post("/iv_surface", json=bad)
    assert resp.status_code == 422


# ── /vix ──────────────────────────────────────────────────────────────────────

def test_vix_valid(client):
    resp = client.post("/vix", json=VALID_PARAMS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "vix" in body
    vix = body["vix"]
    assert isinstance(vix, float)
    assert 0.0 < vix < 100.0, f"VIX should be between 0 and 100, got {vix}"


def test_vix_missing_field(client):
    bad = {k: v for k, v in VALID_PARAMS.items() if k != "H"}
    resp = client.post("/vix", json=bad)
    assert resp.status_code == 422


def test_vix_changes_with_v0(client):
    low_vol  = client.post("/vix", json={**VALID_PARAMS, "v0": 0.02}).json()["vix"]
    high_vol = client.post("/vix", json={**VALID_PARAMS, "v0": 0.20}).json()["vix"]
    assert high_vol > low_vol, "Higher v0 should produce higher model VIX"


# ── /greeks ───────────────────────────────────────────────────────────────────

def test_greeks_valid(client):
    req  = {**VALID_PARAMS, "S": 5000.0}
    resp = client.post("/greeks", json=req)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in ("delta", "gamma", "vega", "vanna", "volga", "iv_surface"):
        assert key in body, f"Missing key: {key}"
        assert len(body[key]) == 8
        assert len(body[key][0]) == 11


def test_greeks_default_spot(client):
    """S defaults to 5000 — should still work without explicit S field."""
    resp = client.post("/greeks", json=VALID_PARAMS)
    assert resp.status_code == 200, resp.text


def test_greeks_missing_field(client):
    bad = {k: v for k, v in VALID_PARAMS.items() if k != "kappa"}
    resp = client.post("/greeks", json=bad)
    assert resp.status_code == 422


# ── /deribit/snapshot ─────────────────────────────────────────────────────────

def test_deribit_snapshot_invalid_currency(client):
    resp = client.get("/deribit/snapshot?currency=INVALID")
    assert resp.status_code == 422


def test_deribit_snapshot_btc_mocked(client):
    """Mock fetch_option_snapshot to avoid real network calls."""
    mock_df = pd.DataFrame({
        "instrument_name": ["BTC-27JUN25-70000-C", "BTC-27JUN25-60000-P"],
        "coin":            ["BTC", "BTC"],
        "mark_iv":         [0.50, 0.55],
        "underlying_price": [65000.0, 65000.0],
        "log_moneyness":   [0.07, -0.08],
        "T":               [0.09, 0.09],   # ~1 month
        "option_type":     ["C", "P"],
    })

    with patch("deepvol.market.deribit_data.fetch_option_snapshot", new=AsyncMock(return_value=mock_df)):
        resp = client.get("/deribit/snapshot?currency=BTC")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["currency"] == "BTC"
    assert body["n_options"] == 2
    assert isinstance(body["atm_iv"], float)


# ── OpenAPI docs ───────────────────────────────────────────────────────────────

def test_openapi_docs_accessible(client):
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_openapi_schema(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = schema["paths"]
    assert "/health" in paths
    assert "/iv_surface" in paths
    assert "/greeks" in paths
    assert "/vix" in paths
    assert "/deribit/snapshot" in paths


# ── Phase 5-6 Endpoints Tests ──────────────────────────────────────────────────

def test_calibrate_neural_sde(client):
    # Setup dummy market IV surface (8x11)
    market_iv = [[0.2] * 11 for _ in range(8)]
    payload = {
        "market_iv": market_iv,
        "S0": 100.0,
        "r": 0.05,
        "q": 0.015,
        "epochs": 2,      # keep it very small for fast unit test
        "N_paths": 128    # small path count
    }
    resp = client.post("/calibrate_neural_sde", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "v0" in body
    assert "rho" in body
    assert "final_rmse" in body
    assert "loss_history" in body
    assert len(body["loss_history"]) == 2
    assert body["v0"] > 0.0
    assert body["rho"] <= 0.0


def test_predict_signature_vol(client):
    # Dummy coefficients (30 elements)
    ell = [0.0] * 30
    ell[0] = 0.01   # level 1 time
    ell[1] = -0.02  # level 1 W
    payload = {
        "v0": 0.04,
        "ell": ell,
        "rho": -0.5,
        "T": 0.25,
        "S0": 100.0,
        "r": 0.05,
        "q": 0.015,
        "N_paths": 128,
        "strikes": [90.0, 100.0, 110.0]
    }
    resp = client.post("/predict/signature_vol", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "strikes" in body
    assert "implied_vols" in body
    assert "option_prices" in body
    assert len(body["implied_vols"]) == 3
    assert len(body["option_prices"]) == 3


def test_hedge_simulate_european(client):
    payload = {
        "option_type": "european",
        "S0": 100.0,
        "strike": 100.0,
        "expiry": 0.1,
        "mu": 0.0,
        "sigma": 0.2,
        "steps": 10,
        "N_paths": 5,
        "cost_stock": 0.0001,
        "cost_vol": 0.0005
    }
    resp = client.post("/hedge/simulate", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "paths_S" in body
    assert "paths_vol" in body
    assert "deltas_stock" in body
    assert "deltas_vol" in body
    assert "costs" in body
    assert "wealth" in body
    assert "payoff" in body
    assert "pnl" in body
    assert "std_pnl" in body
    assert "final_loss" in body
    assert len(body["paths_S"]) == 5
    assert len(body["paths_S"][0]) == 11
    assert len(body["deltas_stock"][0]) == 10


# ── Phase 8 Endpoints Tests ───────────────────────────────────────────────────

def test_greeks_heston_v2(client):
    payload = {
        "S": 100.0,
        "kappa": 1.5,
        "theta": 0.08,
        "sigma": 0.5,
        "rho": -0.7,
        "v0": 0.08,
        "H": 0.08
    }
    resp = client.post("/greeks/heston", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "delta" in body
    assert "gamma" in body
    assert "vega" in body
    assert len(body["delta"]) == 8
    assert len(body["delta"][0]) == 11


def test_greeks_sabr(client):
    payload = {
        "S": 100.0,
        "alpha": 0.05,
        "rho": -0.5,
        "nu": 0.4
    }
    resp = client.post("/greeks/sabr", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "delta" in body
    assert "gamma" in body
    assert "vega" in body


def test_greeks_ssvi(client):
    payload = {
        "S": 100.0,
        "rho": -0.4,
        "eta": 0.5,
        "gamma": 0.3,
        "theta_atm": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
    }
    resp = client.post("/greeks/ssvi", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "delta" in body


def test_greeks_rbergomi(client):
    payload = {
        "S": 100.0,
        "v0": 0.04,
        "H": 0.07,
        "eta": 1.9,
        "rho": -0.7
    }
    resp = client.post("/greeks/rbergomi", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "delta" in body


def test_greeks_invalid_model(client):
    payload = {
        "S": 100.0,
        "alpha": 0.05,
        "rho": -0.5,
        "nu": 0.4
    }
    resp = client.post("/greeks/invalid_model", json=payload)
    assert resp.status_code == 400


def test_calibrate_heston_extended(client):
    market_iv = [[0.2] * 11 for _ in range(8)]
    payload = {
        "market_iv": market_iv,
        "n_starts": 1,
        "max_iter": 5,
        "tol": 1e-4
    }
    resp = client.post("/calibrate/heston", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "params" in body
    assert "final_mse" in body
    assert "rmse_bps" in body


def test_train_isolated_subprocess(client):
    payload = {
        "epochs": 1,
        "batch_size": 256,
        "lr": 1e-3
    }
    resp = client.post("/train/sabr", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "task_id" in body
    assert body["status"] == "running"
    task_id = body["task_id"]

    status_resp = client.get(f"/train/status/{task_id}")
    assert status_resp.status_code == 200, status_resp.text
    status_body = status_resp.json()
    assert status_body["task_id"] == task_id
    assert status_body["model_name"] == "sabr"
    assert status_body["status"] in ("running", "completed", "failed")


def test_session_calibration_cache_flow(client):
    market_iv = [[0.2] * 11 for _ in range(8)]
    payload = {
        "market_iv": market_iv,
        "n_starts": 1,
        "max_iter": 5
    }
    resp = client.post("/session/calibrate/sabr", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "session_id" in body
    assert "params" in body
    session_id = body["session_id"]

    greeks_payload = {
        "S": 100.0,
        "r": 0.05,
        "q": 0.01
    }
    greeks_resp = client.post(f"/session/{session_id}/greeks", json=greeks_payload)
    assert greeks_resp.status_code == 200, greeks_resp.text
    greeks_body = greeks_resp.json()
    assert "delta" in greeks_body
    assert "gamma" in greeks_body
    assert "vega" in greeks_body


def test_session_not_found(client):
    greeks_payload = {
        "S": 100.0
    }
    greeks_resp = client.post("/session/non_existent_session_id/greeks", json=greeks_payload)
    assert greeks_resp.status_code == 404

