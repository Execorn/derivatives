import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import orjson
from deepvol.api.server import app, manager
from deepvol.api.websocket_server import ConflatedQueue, TaskGroup


def test_websocket_ping_pong():
    client = TestClient(app)
    with client.websocket_connect("/ws/risk") as websocket:
        websocket.send_json({"action": "ping"})
        data = websocket.receive_bytes()
        resp = orjson.loads(data)
        assert resp["type"] == "pong"
        assert "timestamp" in resp


def test_websocket_stress_calculation():
    client = TestClient(app)
    with client.websocket_connect("/ws/risk") as websocket:
        websocket.send_json({
            "action": "stress",
            "model_name": "rough_heston",
            "parameters": {
                "kappa": 1.5,
                "theta": 0.08,
                "sigma": 0.5,
                "rho": -0.7,
                "v0": 0.08,
                "H": 0.08
            },
            "S": 65000.0,
            "r": 0.05,
            "q": 0.0
        })
        data = websocket.receive_bytes()
        resp = orjson.loads(data)
        assert resp["type"] == "stress_result"
        assert resp["model_name"] == "rough_heston"
        assert "greeks" in resp
        assert "delta" in resp["greeks"]


def test_websocket_rate_limiting():
    client = TestClient(app)
    with client.websocket_connect("/ws/risk") as websocket:
        # Send 35 messages rapidly (limit is 30)
        throttled = False
        for _ in range(35):
            websocket.send_json({"action": "ping"})
            try:
                data = websocket.receive_bytes()
                resp = orjson.loads(data)
                if resp.get("type") == "error" and "Rate limit exceeded" in resp.get("message", ""):
                    throttled = True
                    break
            except Exception:
                break
        assert throttled, "Rate limit should have been triggered"


def test_websocket_connection_limits():
    client = TestClient(app)
    # Mock manager active_connections with an object whose length is 50
    with patch.object(manager, "active_connections", MagicMock(__len__=lambda s: 50)):
        with client.websocket_connect("/ws/risk") as websocket:
            data = websocket.receive_bytes()
            resp = orjson.loads(data)
            assert resp["type"] == "error"
            assert "Connection limit" in resp["message"]


def test_websocket_subscription_and_unsubscription():
    client = TestClient(app)
    with client.websocket_connect("/ws/risk") as websocket:
        # Subscribe
        websocket.send_json({
            "action": "subscribe",
            "currency": "BTC",
            "model_name": "rough_heston",
            "interval": 0.1,
            "parameters": {
                "kappa": 2.0,
                "theta": 0.05,
                "sigma": 0.3,
                "rho": -0.6,
                "v0": 0.05,
                "H": 0.08
            }
        })
        
        # Expect "subscribed" ack
        data = websocket.receive_bytes()
        resp = orjson.loads(data)
        assert resp["type"] == "subscribed"
        assert resp["currency"] == "BTC"

        # Expect at least one real-time update
        data = websocket.receive_bytes()
        resp = orjson.loads(data)
        assert resp["type"] == "update"
        assert resp["currency"] == "BTC"
        assert "greeks" in resp

        # Unsubscribe
        websocket.send_json({"action": "unsubscribe"})
        data = websocket.receive_bytes()
        resp = orjson.loads(data)
        assert resp["type"] == "unsubscribed"


def test_parameter_clamping_and_compliance():
    client = TestClient(app)
    with client.websocket_connect("/ws/risk") as websocket:
        # Send extreme parameters (OOD)
        websocket.send_json({
            "action": "stress",
            "model_name": "rough_heston",
            "parameters": {
                "kappa": 10.0,       # Max is 5.0
                "theta": 0.0001,     # Min is 0.01 (due to 100 bps clamp)
                "sigma": 5.0,        # Max is 1.5
                "rho": 0.5,          # Max is 0.0,
                "v0": 0.005,         # Min is 0.01 (due to 100 bps clamp)
                "H": 0.5             # Max is 0.15
            },
            "S": 65000.0
        })
        data = websocket.receive_bytes()
        resp = orjson.loads(data)
        assert resp["type"] == "stress_result"
        
        # Verify they got clamped
        params = resp["parameters"]
        assert params["kappa"] == 5.0
        assert params["theta"] == 0.010001
        assert params["sigma"] == 1.5
        assert params["rho"] == 0.0
        assert params["v0"] == 0.010001
        assert params["H"] == 0.15


@pytest.mark.asyncio
async def test_conflated_queue_conflation():
    queue = ConflatedQueue()
    queue.put("BTC", {"id": 1, "val": 10})
    queue.put("BTC", {"id": 2, "val": 20}) # Should overwrite the previous update
    queue.put("ETH", {"id": 3, "val": 30})
    
    batch = await queue.get()
    assert len(batch) == 2
    assert batch["BTC"]["val"] == 20
    assert batch["ETH"]["val"] == 30


@pytest.mark.asyncio
async def test_task_group_cancellation():
    tg = TaskGroup()
    task_run = False
    
    async def dummy_task():
        nonlocal task_run
        try:
            task_run = True
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
            
    async with tg:
        tg.create_task(dummy_task())
        # Let task run
        await asyncio.sleep(0.01)
        assert task_run is True
        
    # After exiting TaskGroup, the dummy task should have been cancelled and awaited
    # Verify no tasks are active in tg
    assert len(tg._tasks) == 0
