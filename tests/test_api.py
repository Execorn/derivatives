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
from api.server import app, _MODEL_STATE

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

    with patch("market.deribit_data.fetch_option_snapshot", new=AsyncMock(return_value=mock_df)):
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
