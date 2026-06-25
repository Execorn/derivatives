"""
Real-time WebSocket connection manager and JSON router.

Provides real-time streaming of options risk metrics (Greeks and IV surfaces)
using FNO models. Clients can subscribe to asset feeds and trigger stress-test
scenarios.

Geometric Brownian Motion for spot simulation:
S_t = S_{t-dt} * exp((r - q - 0.5 * sigma^2)*dt + sigma * sqrt(dt) * Z)
where Z ~ N(0, 1).
"""

from __future__ import annotations

import asyncio
import logging
import time
import math
import random
import socket
from typing import Any, Dict, List, Set

from fastapi import WebSocket, WebSocketDisconnect

try:
    import orjson
    def orjson_dumps(data: Any) -> bytes:
        return orjson.dumps(data)
except ImportError:
    import json
    logging.getLogger(__name__).warning(
        "orjson not installed — falling back to stdlib json. "
        "WebSocket serialization will be ~10x slower. Install orjson for production use."
    )
    def orjson_dumps(data: Any) -> bytes:
        return json.dumps(data).encode("utf-8")

# Import compute_model_greeks deferred to prevent circular import

log = logging.getLogger(__name__)

# Connection limits and rate limits
MAX_CONNECTIONS = 50
RATE_LIMIT_WINDOW = 10.0  # seconds
MAX_MESSAGES_PER_WINDOW = 30



import threading

class ConflatedQueue:
    """
    Thread-safe, async-native queue that retains only the latest update per key.
    Prevents memory leaks and backpressure lag when clients consume slowly.
    """
    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._event = asyncio.Event()
        self._lock = threading.Lock()

    def put(self, key: str, value: Any) -> None:
        """Upsert the latest value for the specified key and trigger wait event."""
        with self._lock:
            self._data[key] = value
        self._event.set()

    async def get(self) -> Dict[str, Any]:
        """Wait until data is available, then drain and return the current batch.

        Fix for CC-W1: _event.clear() is now inside the lock to prevent a
        lost-wakeup race where put() sets the event between lock-release and
        clear(), causing the consumer to block despite available data.
        """
        while True:
            with self._lock:
                if self._data:
                    batch = self._data.copy()
                    self._data.clear()
                    return batch
                # Clear event inside lock: if put() fires after this point,
                # it will re-set the event and we will see the data on the
                # next iteration after await returns.
                self._event.clear()
            await self._event.wait()


class ConnectionManager:
    """Manages active WebSocket connections, rate limits, and TCP optimization with thread-safety."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_connections: Set[WebSocket] = set()
        self.connection_queues: Dict[WebSocket, ConflatedQueue] = {}
        self.connection_tasks: Dict[WebSocket, List[asyncio.Task]] = {}
        # Client rate limiting: websocket -> list of message timestamps
        self.message_timestamps: Dict[WebSocket, List[float]] = {}

    def is_active(self, websocket: WebSocket) -> bool:
        """Checks if connection is active."""
        with self._lock:
            return websocket in self.active_connections

    def get_queue(self, websocket: WebSocket) -> Optional[ConflatedQueue]:
        """Retrieves conflated queue for a connection."""
        with self._lock:
            return self.connection_queues.get(websocket)

    def set_queue(self, websocket: WebSocket, queue: ConflatedQueue) -> None:
        """Sets conflated queue for a connection."""
        with self._lock:
            self.connection_queues[websocket] = queue

    def add_task(self, websocket: WebSocket, task: asyncio.Task) -> None:
        """Adds async task to a connection."""
        with self._lock:
            if websocket in self.connection_tasks:
                self.connection_tasks[websocket].append(task)

    def clear_tasks(self, websocket: WebSocket) -> List[asyncio.Task]:
        """Clears and returns connection tasks."""
        with self._lock:
            tasks = self.connection_tasks.get(websocket, [])
            self.connection_tasks[websocket] = []
            return tasks

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts a new connection and enforces limits."""
        with self._lock:
            conn_count = len(self.active_connections)
        if conn_count >= MAX_CONNECTIONS:
            await websocket.accept()
            await websocket.send_json({
                "type": "error",
                "message": f"Connection limit of {MAX_CONNECTIONS} reached. Reconnect later."
            })
            await websocket.close(code=1008)
            return

        await websocket.accept()
        
        # SOTA: Enable TCP_NODELAY to bypass Nagle's algorithm buffer latency
        try:
            transport = websocket.scope.get("transport")
            if transport:
                sock = transport.get_extra_info("socket")
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    log.info("Optimized WebSocket TCP socket with TCP_NODELAY")
        except Exception as exc:
            log.debug("Failed to set TCP_NODELAY on WebSocket socket: %s", exc)

        with self._lock:
            self.active_connections.add(websocket)
            self.connection_queues[websocket] = ConflatedQueue()
            self.connection_tasks[websocket] = []
            self.message_timestamps[websocket] = []
            active_count = len(self.active_connections)
        log.info("New WebSocket connection accepted. Active: %d", active_count)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Cleans up connection resources and cancels associated tasks."""
        with self._lock:
            self.active_connections.discard(websocket)
            self.connection_queues.pop(websocket, None)
            tasks = self.connection_tasks.pop(websocket, [])
            self.message_timestamps.pop(websocket, None)
            active_count = len(self.active_connections)
        
        # Cancel any active tasks for this connection
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        log.info("WebSocket disconnected. Active: %d", active_count)

    async def check_rate_limit(self, websocket: WebSocket) -> bool:
        """
        Enforces a simple sliding window rate limit.
        Returns True if allowed, False if throttled.
        """
        now = time.time()
        with self._lock:
            timestamps = self.message_timestamps.get(websocket, [])
            # Filter timestamps within current window
            timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
            self.message_timestamps[websocket] = timestamps

            if len(timestamps) >= MAX_MESSAGES_PER_WINDOW:
                return False

            self.message_timestamps[websocket].append(now)
            return True

    async def send_json(self, websocket: WebSocket, data: dict) -> None:
        """Sends JSON payload safely to a single connection."""
        with self._lock:
            active = websocket in self.active_connections
        if active:
            try:
                # SOTA: Use fast orjson serialization directly to bytes
                binary_frame = orjson_dumps(data)
                await websocket.send_bytes(binary_frame)
            except Exception as exc:
                log.warning("Failed to send message over WebSocket: %s", exc)
                await self.disconnect(websocket)



class JSONRouter:
    """Routes incoming JSON actions and manages subscription simulator loops."""

    def __init__(self, manager: ConnectionManager) -> None:
        self.manager = manager

    async def handle_message(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Parses action and routes to corresponding handler."""
        if not await self.manager.check_rate_limit(websocket):
            await self.manager.send_json(websocket, {
                "type": "error",
                "message": "Rate limit exceeded. Please throttle request frequency."
            })
            return

        action = data.get("action")
        if not action:
            await self.manager.send_json(websocket, {
                "type": "error",
                "message": "Missing required field 'action'."
            })
            return

        log.debug("WebSocket action received: %s", action)

        if action == "ping":
            await self.manager.send_json(websocket, {"type": "pong", "timestamp": time.time()})
        
        elif action == "subscribe":
            await self._handle_subscribe(websocket, data)

        elif action == "unsubscribe":
            await self._handle_unsubscribe(websocket)

        elif action == "stress":
            await self._handle_stress(websocket, data)

        else:
            await self.manager.send_json(websocket, {
                "type": "error",
                "message": f"Unknown action: {action}"
            })

    async def _handle_subscribe(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Subscribes the connection to a simulated live risk feed using ConflatedQueue."""
        # Cancel existing tasks first
        await self._handle_unsubscribe(websocket, send_ack=False)

        currency = data.get("currency", "BTC").upper()
        model_name = data.get("model_name", "rough_heston").lower()
        params = data.get("parameters", {})
        interval = float(data.get("interval", 1.0))

        # Validate parameters exist or use defaults
        if not params:
            if model_name == "sabr":
                params = {"alpha": 0.20, "rho": -0.40, "nu": 0.40}
            elif model_name == "rbergomi":
                params = {"v0": 0.08, "H": 0.07, "eta": 1.5, "rho": -0.70}
            elif model_name == "ssvi":
                params = {"theta_atm": [0.1]*8, "rho": -0.40, "eta": 1.0, "gamma": 0.5}
            else:  # heston or rough_heston
                params = {"kappa": 2.0, "theta": 0.05, "sigma": 0.3, "rho": -0.6, "v0": 0.05, "H": 0.08}

        # Seed initial spot price based on currency
        spot = 65000.0 if currency == "BTC" else 3500.0

        conflated_queue = self.manager.get_queue(websocket)
        if not conflated_queue:
            conflated_queue = ConflatedQueue()
            self.manager.set_queue(websocket, conflated_queue)

        # 1. Producer Task: Simulates spot, perturbs parameters, computes Greeks, puts to ConflatedQueue
        async def risk_metrics_producer() -> None:
            nonlocal spot
            sim_vol = 0.15  # annualized vol for simulation path
            r = 0.05
            q = 0.0
            
            try:
                while self.manager.is_active(websocket):
                    t_start = time.perf_counter()
                    
                    # Simulate spot price movement via GBM discretization step
                    dt = interval / 365.25  # time step in years
                    z = random.gauss(0.0, 1.0)
                    drift = (r - q - 0.5 * sim_vol**2) * dt
                    diffusion = sim_vol * (dt**0.5) * z
                    spot = spot * math.exp(drift + diffusion)

                    # Perturb parameters slightly within valid bounds
                    if "v0" in params:
                        params["v0"] = max(0.01, min(0.25, params["v0"] + random.gauss(0.0, 0.002)))
                    if "rho" in params:
                        params["rho"] = max(-0.95, min(-0.05, params["rho"] + random.gauss(0.0, 0.005)))
                    if "kappa" in params:
                        params["kappa"] = max(0.5, min(5.0, params["kappa"] + random.gauss(0.0, 0.05)))
                    if "sigma" in params:
                        params["sigma"] = max(0.1, min(1.5, params["sigma"] + random.gauss(0.0, 0.01)))
                    if "H" in params:
                        params["H"] = max(0.04, min(0.15, params["H"] + random.gauss(0.0, 0.001)))
                    if "alpha" in params:
                        params["alpha"] = max(0.05, min(0.8, params["alpha"] + random.gauss(0.0, 0.005)))
                    if "nu" in params:
                        params["nu"] = max(0.1, min(1.2, params["nu"] + random.gauss(0.0, 0.01)))

                    req_payload = {
                        "S": spot,
                        "r": r,
                        "q": q,
                        **params
                    }

                    try:
                        from deepvol.api.server import compute_model_greeks
                        res = await compute_model_greeks(model_name, req_payload)
                        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

                        update_msg = {
                            "type": "update",
                            "timestamp": time.time(),
                            "currency": currency,
                            "model_name": model_name,
                            "spot": spot,
                            "parameters": params,
                            "greeks": res.model_dump(),
                            "latency_ms": elapsed_ms
                        }
                        # Conflated write
                        conflated_queue.put(currency, update_msg)
                    except Exception as stream_exc:
                        log.warning("Stream step failed: %s", stream_exc)
                        conflated_queue.put("error", {
                            "type": "error",
                            "message": f"Stream step failed: {str(stream_exc)}"
                        })

                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                log.debug("Producer task cancelled.")

        # 2. Consumer Task: Reads from ConflatedQueue and sends over physical socket
        async def socket_writer_consumer() -> None:
            try:
                while self.manager.is_active(websocket):
                    batch = await conflated_queue.get()
                    # Broadcast/send each item in conflate batch
                    for item in batch.values():
                        await self.manager.send_json(websocket, item)
            except WebSocketDisconnect:
                log.info("WebSocket disconnect in writer task.")
            except asyncio.CancelledError:
                log.debug("Consumer task cancelled.")

        # Run tasks under structured concurrency with Python version fallback
        producer_task = asyncio.create_task(risk_metrics_producer())
        consumer_task = asyncio.create_task(socket_writer_consumer())
        self.manager.add_task(websocket, producer_task)
        self.manager.add_task(websocket, consumer_task)

        await self.manager.send_json(websocket, {
            "type": "subscribed",
            "currency": currency,
            "model_name": model_name,
            "parameters": params
        })

    async def _handle_unsubscribe(self, websocket: WebSocket, send_ack: bool = True) -> None:
        """Cancels any stream loop tasks for this connection."""
        tasks = self.manager.clear_tasks(websocket)
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if send_ack:
            await self.manager.send_json(websocket, {"type": "unsubscribed"})

    async def _handle_stress(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Calculates Greeks for a one-off stress scenario and returns them."""
        params = data.get("parameters", {})
        model_name = data.get("model_name", "rough_heston").lower()
        S = float(data.get("S", 5000.0 if model_name not in ("sabr", "rough_heston") else 65000.0))
        r = float(data.get("r", 0.05))
        q = float(data.get("q", 0.0))

        # Construct payload for compute_model_greeks
        req_payload = {
            "S": S,
            "r": r,
            "q": q,
            **params
        }

        t0 = time.perf_counter()
        try:
            from deepvol.api.server import compute_model_greeks
            res = await compute_model_greeks(model_name, req_payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            await self.manager.send_json(websocket, {
                "type": "stress_result",
                "timestamp": time.time(),
                "latency_ms": elapsed_ms,
                "model_name": model_name,
                "spot": S,
                "parameters": params,
                "greeks": res.model_dump()
            })
        except Exception as exc:
            log.exception("Error running stress test over WebSocket: %s", exc)
            await self.manager.send_json(websocket, {
                "type": "error",
                "message": f"Stress test failed: {str(exc)}"
            })
