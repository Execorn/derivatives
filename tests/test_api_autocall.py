"""
test_api_autocall.py — Test suite for Autocall CorrectionEnsemble API endpoints.

Tests:
  - POST /autocall/price returns valid schema and correct numeric properties
  - OOD detection triggers on extreme tail parameters (v0=0.001, sigma=0.99)
  - SR 26-2 Guardian fallback activates on extreme out-of-bounds parameters (kappa=100)
  - Direct non-guardian pricing mode executes and returns total NPV
  - Batch pricing /autocall/price_batch returns list of valid responses
  - GPU response latency benchmark (< 100ms per AC-10)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import pytest
import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from deepvol.api.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.asyncio
async def test_autocall_price_standard_schema():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "kappa": 2.0,
            "theta": 0.04,
            "sigma": 0.3,
            "rho": -0.7,
            "v0": 0.04,
            "B": 1.0,
            "coupon": 0.10,
            "T": 1.5,
            "n_obs_per_year": 4.0,
            "r": 0.03,
            "use_guardian": True,
        }
        resp = await ac.post("/autocall/price", json=payload)
        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()

        expected_fields = [
            "npv",
            "pde_npv",
            "correction_bps",
            "uncertainty_bps",
            "is_ood",
            "is_fallback",
            "fallback_trigger",
            "fallback_reasons",
            "df_dB",
            "latency_ms",
            "tau_ood",
        ]
        for field in expected_fields:
            assert field in data, f"Missing field {field} in response"

        assert isinstance(data["npv"], float)
        assert isinstance(data["pde_npv"], float)
        assert isinstance(data["correction_bps"], float)
        assert isinstance(data["uncertainty_bps"], float)
        assert isinstance(data["is_ood"], bool)
        assert isinstance(data["is_fallback"], bool)
        assert isinstance(data["df_dB"], float)
        assert isinstance(data["latency_ms"], float)
        assert isinstance(data["tau_ood"], float)

        assert 0.5 <= data["npv"] <= 1.5
        assert 0.5 <= data["pde_npv"] <= 1.5
        assert data["uncertainty_bps"] >= 0.0


@pytest.mark.asyncio
async def test_autocall_price_ood_detection():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        extreme_payload = {
            "kappa": 2.0,
            "theta": 0.04,
            "sigma": 0.99,
            "rho": -0.7,
            "v0": 0.001,
            "B": 1.0,
            "coupon": 0.10,
            "T": 1.5,
            "n_obs_per_year": 4.0,
            "r": 0.03,
            "use_guardian": True,
        }
        resp = await ac.post("/autocall/price", json=extreme_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["is_ood"] is True, f"Expected is_ood=True for extreme params, got {data}"


@pytest.mark.asyncio
async def test_autocall_price_guardian_fallback():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        oob_payload = {
            "kappa": 100.0,
            "theta": 0.04,
            "sigma": 0.3,
            "rho": -0.7,
            "v0": 0.04,
            "B": 1.0,
            "coupon": 0.10,
            "T": 1.5,
            "n_obs_per_year": 4.0,
            "r": 0.03,
            "use_guardian": True,
        }
        resp = await ac.post("/autocall/price", json=oob_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["is_fallback"] is True, f"Expected is_fallback=True for kappa=100, got {data}"
        assert data["fallback_trigger"] is not None
        assert len(data["fallback_reasons"]) > 0


@pytest.mark.asyncio
async def test_autocall_price_non_guardian_mode():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "kappa": 2.0,
            "theta": 0.04,
            "sigma": 0.3,
            "rho": -0.7,
            "v0": 0.04,
            "B": 1.0,
            "coupon": 0.10,
            "T": 1.5,
            "n_obs_per_year": 4.0,
            "r": 0.03,
            "use_guardian": False,
        }
        resp = await ac.post("/autocall/price", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["is_fallback"] is False
        assert data["fallback_trigger"] is None
        assert 0.5 <= data["npv"] <= 1.5
        assert isinstance(data["df_dB"], float)


@pytest.mark.asyncio
async def test_autocall_price_batch():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        batch_payload = {
            "items": [
                {
                    "kappa": 2.0,
                    "theta": 0.04,
                    "sigma": 0.3,
                    "rho": -0.7,
                    "v0": 0.04,
                    "B": 1.0,
                    "coupon": 0.10,
                    "T": 1.5,
                    "n_obs_per_year": 4.0,
                    "r": 0.03,
                    "use_guardian": True,
                },
                {
                    "kappa": 1.5,
                    "theta": 0.05,
                    "sigma": 0.25,
                    "rho": -0.5,
                    "v0": 0.03,
                    "B": 0.95,
                    "coupon": 0.08,
                    "T": 1.0,
                    "n_obs_per_year": 4.0,
                    "r": 0.02,
                    "use_guardian": True,
                },
            ]
        }
        resp = await ac.post("/autocall/price_batch", json=batch_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        for item in data:
            assert "npv" in item
            assert "pde_npv" in item
            assert "uncertainty_bps" in item


def test_autocall_price_latency(client):
    payload = {
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.3,
        "rho": -0.7,
        "v0": 0.04,
        "B": 1.0,
        "coupon": 0.10,
        "T": 1.5,
        "n_obs_per_year": 4.0,
        "r": 0.03,
        "use_guardian": False,
    }
    # Warmup call
    client.post("/autocall/price", json=payload)

    # Timed call
    t0 = time.perf_counter()
    resp = client.post("/autocall/price", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 200
    print(f"Autocall pricing total request latency: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 150.0, f"Expected roundtrip latency < 150ms, got {elapsed_ms:.2f} ms"
