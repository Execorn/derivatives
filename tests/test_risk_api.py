"""
tests/test_risk_api.py — Integration tests for FastAPI WebSocket risk streaming.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deepvol.api.server import app


def test_websocket_connect_and_ping():
    """Verify WebSocket connection succeeds and responds to ping-pong."""
    client = TestClient(app)
    with client.websocket_connect("/ws/risk") as websocket:
        websocket.send_json({"action": "ping"})
        resp = websocket.receive_json(mode="binary")
        assert resp["type"] == "pong"
        assert "timestamp" in resp


def test_websocket_invalid_action():
    """Verify WebSocket returns an error for invalid or missing action."""
    client = TestClient(app)
    with client.websocket_connect("/ws/risk") as websocket:
        websocket.send_json({"action": "invalid_action_foo"})
        resp = websocket.receive_json(mode="binary")
        assert resp["type"] == "error"
        assert "Unknown action" in resp["message"]

        websocket.send_json({})
        resp = websocket.receive_json(mode="binary")
        assert resp["type"] == "error"
        assert "Missing required field" in resp["message"]


def test_websocket_subscribe_and_stream():
    """Verify subscription starts a live stream of option Greeks."""
    client = TestClient(app)
    with client.websocket_connect("/ws/risk") as websocket:
        subscribe_payload = {
            "action": "subscribe",
            "currency": "BTC",
            "model_name": "rough_heston",
            "parameters": {
                "kappa": 2.0,
                "theta": 0.05,
                "sigma": 0.3,
                "rho": -0.6,
                "v0": 0.05,
                "H": 0.08
            },
            "interval": 0.1  # Fast updates for testing
        }
        websocket.send_json(subscribe_payload)
        
        # 1. First response should confirm subscription
        confirm = websocket.receive_json(mode="binary")
        assert confirm["type"] == "subscribed"
        assert confirm["currency"] == "BTC"
        assert confirm["model_name"] == "rough_heston"

        # 2. Wait and receive at least two periodic stream updates
        updates = []
        for _ in range(2):
            msg = websocket.receive_json(mode="binary")
            if msg.get("type") == "update":
                updates.append(msg)
            else:
                # If we get error or something else, fail
                pytest.fail(f"Expected update message, got: {msg}")

        assert len(updates) == 2
        for update in updates:
            assert "timestamp" in update
            assert "spot" in update
            assert "latency_ms" in update
            assert "greeks" in update
            
            greeks = update["greeks"]
            assert "iv_surface" in greeks
            assert "delta" in greeks
            assert "gamma" in greeks
            assert "vega" in greeks
            assert "vanna" in greeks
            assert "volga" in greeks
            
            # Grids should be 8 maturities by 11 strikes
            assert len(greeks["iv_surface"]) == 8
            assert len(greeks["iv_surface"][0]) == 11

        # 3. Test unsubscribe
        websocket.send_json({"action": "unsubscribe"})
        unsub_confirm = websocket.receive_json(mode="binary")
        assert unsub_confirm["type"] == "unsubscribed"


def test_websocket_stress_scenario():
    """Verify one-off stress tests return computed Greeks."""
    client = TestClient(app)
    with client.websocket_connect("/ws/risk") as websocket:
        stress_payload = {
            "action": "stress",
            "model_name": "rough_heston",
            "S": 70000.0,
            "r": 0.06,
            "q": 0.01,
            "parameters": {
                "kappa": 2.2,
                "theta": 0.06,
                "sigma": 0.35,
                "rho": -0.55,
                "v0": 0.06,
                "H": 0.07
            }
        }
        websocket.send_json(stress_payload)
        
        resp = websocket.receive_json(mode="binary")
        assert resp["type"] == "stress_result"
        assert resp["spot"] == 70000.0
        assert "latency_ms" in resp
        
        greeks = resp["greeks"]
        assert "iv_surface" in greeks
        assert "delta" in greeks
        assert len(greeks["iv_surface"]) == 8
        assert len(greeks["iv_surface"][0]) == 11


def test_websocket_rate_limiting():
    """Verify rate limits are enforced when messages are flooded."""
    client = TestClient(app)
    with client.websocket_connect("/ws/risk") as websocket:
        # Send many pings rapidly
        errors_received = 0
        for _ in range(40):
            try:
                websocket.send_json({"action": "ping"})
                resp = websocket.receive_json(mode="binary")
                if resp.get("type") == "error" and "Rate limit exceeded" in resp.get("message", ""):
                    errors_received += 1
            except Exception as e:
                # Log or print error to assist debugging
                print(f"Exception during flooded ping: {e}")
                break
        
        assert errors_received > 0, "Expected at least one rate limit error response"
