"""
High-throughput FastAPI WebSocket server with ConflatedQueue, asyncio.TaskGroup
structured lifecycle management, TCP_NODELAY optimization, and Rust-based orjson binary frames.
Includes SR 26-2 model governance compliance (online drift monitoring via PSI and OOD parameter clamping).
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import orjson

from deepvol.api.server import compute_model_greeks

log = logging.getLogger(__name__)

# Connection limits and rate limits
MAX_CONNECTIONS = 50
RATE_LIMIT_WINDOW = 10.0  # seconds
MAX_MESSAGES_PER_WINDOW = 30

# Param bounds for OOD detection and clamping (SR 26-2)
PARAM_BOUNDS = {
    "kappa": (0.5, 5.0),
    "theta": (0.01, 0.25),
    "sigma": (0.1, 1.5),
    "rho": (-0.95, 0.0),
    "v0": (0.01, 0.25),
    "H": (0.04, 0.15),
    "alpha": (0.05, 0.8),
    "beta": (0.0, 1.0),
    "nu": (0.1, 1.5),
    "eta": (0.1, 2.0),
}


class TaskGroup:
    """
    A structured concurrency context manager for asyncio tasks (Python < 3.11 compatible).
    Ensures no dangling tasks survive the lifespan of a WebSocket connection.
    """
    def __init__(self) -> None:
        self._tasks: Set[asyncio.Task] = set()
        self._exiting = False

    def create_task(self, coro) -> asyncio.Task:
        if self._exiting:
            raise RuntimeError("TaskGroup has already exited or is exiting")
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def __aenter__(self) -> TaskGroup:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        self._exiting = True
        # Cancel all managed tasks
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        
        if self._tasks:
            # Wait for all tasks to complete or handle their cancellation
            await asyncio.gather(*self._tasks, return_exceptions=True)
        return False  # Propagate exception if any


class ConflatedQueue:
    """
    Thread-safe, async-native conflated queue that retains only the latest update per key.
    Prevents memory leaks and TCP backpressure lag when clients consume slowly.
    """
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._data: Dict[str, Any] = {}
        self._event = asyncio.Event()
        self._lock = threading.Lock()
        self._loop = loop or asyncio.get_event_loop()

    def put(self, key: str, value: Any) -> None:
        """Upsert the latest value for the specified key and trigger the wait event."""
        with self._lock:
            self._data[key] = value
        self._loop.call_soon_threadsafe(self._event.set)

    async def get(self) -> Dict[str, Any]:
        """Wait until data is available, then drain and return the current conflated batch."""
        while True:
            with self._lock:
                if self._data:
                    batch = self._data.copy()
                    self._data.clear()
                    self._event.clear()
                    return batch
            await self._event.wait()


class ConnectionManager:
    """Manages active WebSocket connections, rate limits, and TCP optimization with thread-safety."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_connections: Set[WebSocket] = set()
        self.connection_queues: Dict[WebSocket, ConflatedQueue] = {}
        self.connection_tgs: Dict[WebSocket, TaskGroup] = {}
        # Client rate limiting: websocket -> list of message timestamps
        self.message_timestamps: Dict[WebSocket, List[float]] = {}
        
        # Drift tracking state per websocket connection: param_name -> list of observations
        self.param_history: Dict[WebSocket, Dict[str, List[float]]] = {}
        self.param_baselines: Dict[WebSocket, Dict[str, List[float]]] = {}

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

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts a new connection, configures TCP options, and enforces limits."""
        with self._lock:
            conn_count = len(self.active_connections)
        if conn_count >= MAX_CONNECTIONS:
            await websocket.accept()
            # Send error via fast orjson binary frame
            err_msg = orjson.dumps({
                "type": "error",
                "message": f"Connection limit of {MAX_CONNECTIONS} reached. Reconnect later."
            })
            await websocket.send_bytes(err_msg)
            await websocket.close(code=1008)
            return

        await websocket.accept()
        
        # Set TCP socket options: TCP_NODELAY and socket buffer sizes
        try:
            transport = websocket.scope.get("transport")
            if transport:
                sock = transport.get_extra_info("socket")
                if sock:
                    # Disable Nagle's algorithm
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    # Optimize send and receive buffer sizes for throughput
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
                    log.info("Optimized WebSocket TCP socket with TCP_NODELAY, SO_SNDBUF, SO_RCVBUF")
        except Exception as exc:
            log.warning("Failed to optimize WebSocket TCP socket: %s", exc)

        with self._lock:
            self.active_connections.add(websocket)
            self.connection_queues[websocket] = ConflatedQueue()
            self.message_timestamps[websocket] = []
            self.param_history[websocket] = {}
            self.param_baselines[websocket] = {}
            active_count = len(self.active_connections)
        log.info("New WebSocket connection accepted. Active: %d", active_count)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Cleans up connection resources."""
        with self._lock:
            self.active_connections.discard(websocket)
            self.connection_queues.pop(websocket, None)
            self.connection_tgs.pop(websocket, None)
            self.message_timestamps.pop(websocket, None)
            self.param_history.pop(websocket, None)
            self.param_baselines.pop(websocket, None)
            active_count = len(self.active_connections)
        log.info("WebSocket disconnected. Active: %d", active_count)

    async def check_rate_limit(self, websocket: WebSocket) -> bool:
        """Enforces sliding window rate limit. Returns True if allowed."""
        now = time.time()
        with self._lock:
            timestamps = self.message_timestamps.get(websocket, [])
            timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
            self.message_timestamps[websocket] = timestamps

            if len(timestamps) >= MAX_MESSAGES_PER_WINDOW:
                return False

            self.message_timestamps[websocket].append(now)
            return True

    async def send_binary(self, websocket: WebSocket, data: dict) -> None:
        """Sends binary (orjson) payload safely to a connection."""
        with self._lock:
            active = websocket in self.active_connections
        if active:
            try:
                binary_frame = orjson.dumps(data)
                await websocket.send_bytes(binary_frame)
            except Exception as exc:
                log.warning("Failed to send message over WebSocket: %s", exc)
                await self.disconnect(websocket)

    def track_drift_and_ood(self, websocket: WebSocket, parameters: Dict[str, float]) -> Dict[str, float]:
        """
        Compliance Guardian (SR 26-2):
        1. Checks for out-of-distribution (OOD) parameters and clamps them.
        2. Clamps the minimum volatility parameter to 0.01 (100 bps) to prevent Durrleman singularities.
        3. Logs input parameter drift using the Population Stability Index (PSI).
        """
        clamped_params = {}
        for p, val in parameters.items():
            clamped_val = val
            # OOD check and clamp
            if p in PARAM_BOUNDS:
                lower, upper = PARAM_BOUNDS[p]
                # Special volatility parameter minimum clamping to 0.01 (100 bps)
                # theta and v0 have strict gt=0.01 constraints in HestonGreeksRequest, so they are clamped to 0.010001
                if p in ("v0", "theta"):
                    lower = max(lower, 0.010001)
                elif p in ("sigma", "nu", "eta"):
                    lower = max(lower, 0.01)
                
                if val < lower or val > upper:
                    clamped_val = max(lower, min(upper, val))
                    log.warning(
                        "COMPLIANCE OOD DETECTION: Parameter '%s' value %.6f is out of bounds (%s, %s). "
                        "Clamping to %.6f.", p, val, lower, upper, clamped_val
                    )
            clamped_params[p] = clamped_val

        # Online drift tracking (PSI)
        with self._lock:
            if websocket in self.active_connections:
                history = self.param_history.setdefault(websocket, {})
                baselines = self.param_baselines.setdefault(websocket, {})
                
                for p, val in clamped_params.items():
                    # If baseline does not exist, initialize it with a perturbed distribution around the initial value
                    if p not in baselines:
                        baselines[p] = [val + random.normalvariate(0, 0.05 * abs(val) if val != 0 else 0.05) for _ in range(100)]
                    
                    p_hist = history.setdefault(p, [])
                    p_hist.append(val)
                    
                    # Compute PSI on rolling window of 100 samples once we have at least 30 samples
                    if len(p_hist) >= 30:
                        if len(p_hist) > 100:
                            p_hist.pop(0)
                        
                        psi_score = self._calculate_psi(baselines[p], p_hist)
                        if psi_score > 0.25:
                            log.warning("SR 26-2 COMPLIANCE DRIFT WARNING: Parameter '%s' PSI = %.4f (> 0.25 threshold)", p, psi_score)
                        elif psi_score > 0.1:
                            log.info("SR 26-2 COMPLIANCE DRIFT INFO: Parameter '%s' PSI = %.4f (moderate drift)", p, psi_score)
                            
        return clamped_params

    def _calculate_psi(self, baseline: List[float], actual: List[float], num_bins: int = 10) -> float:
        """Helper to calculate Population Stability Index."""
        if not baseline or not actual:
            return 0.0
        try:
            combined = baseline + actual
            min_val, max_val = min(combined), max(combined)
            if max_val == min_val:
                max_val += 1e-5
            
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)
            baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
            actual_counts, _ = np.histogram(actual, bins=bin_edges)
            
            eps = 1e-4
            baseline_probs = (baseline_counts + eps) / (len(baseline) + eps * num_bins)
            actual_probs = (actual_counts + eps) / (len(actual) + eps * num_bins)
            
            psi_val = np.sum((actual_probs - baseline_probs) * np.log(actual_probs / baseline_probs))
            return float(psi_val)
        except Exception as e:
            log.error("Failed to calculate PSI: %s", e)
            return 0.0


class JSONRouter:
    """Routes incoming JSON actions and manages subscription loops within the TaskGroup."""

    def __init__(self, manager: ConnectionManager) -> None:
        self.manager = manager
        # Save reference to subscription tasks so they can be explicitly unsubscribed/cancelled
        self.subscription_tasks: Dict[WebSocket, List[asyncio.Task]] = {}

    async def handle_message(self, websocket: WebSocket, data: Dict[str, Any], tg: TaskGroup) -> None:
        """Parses action and routes to corresponding handler."""
        if not await self.manager.check_rate_limit(websocket):
            await self.manager.send_binary(websocket, {
                "type": "error",
                "message": "Rate limit exceeded. Please throttle request frequency."
            })
            return

        action = data.get("action")
        if not action:
            await self.manager.send_binary(websocket, {
                "type": "error",
                "message": "Missing required field 'action'."
            })
            return

        log.debug("WebSocket action received: %s", action)

        if action == "ping":
            await self.manager.send_binary(websocket, {"type": "pong", "timestamp": time.time()})
        
        elif action == "subscribe":
            await self._handle_subscribe(websocket, data, tg)

        elif action == "unsubscribe":
            await self._handle_unsubscribe(websocket)

        elif action == "stress":
            await self._handle_stress(websocket, data)

        else:
            await self.manager.send_binary(websocket, {
                "type": "error",
                "message": f"Unknown action: {action}"
            })

    async def _handle_subscribe(self, websocket: WebSocket, data: Dict[str, Any], tg: TaskGroup) -> None:
        """Subscribes the connection to a simulated live risk feed using ConflatedQueue."""
        # Cancel any existing subscription tasks first
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

        # Compliance Check (OOD and Clamping)
        params = self.manager.track_drift_and_ood(websocket, params)

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
                        params["v0"] += random.gauss(0.0, 0.002)
                    if "rho" in params:
                        params["rho"] += random.gauss(0.0, 0.005)
                    if "kappa" in params:
                        params["kappa"] += random.gauss(0.0, 0.05)
                    if "sigma" in params:
                        params["sigma"] += random.gauss(0.0, 0.01)
                    if "H" in params:
                        params["H"] += random.gauss(0.0, 0.001)
                    if "alpha" in params:
                        params["alpha"] += random.gauss(0.0, 0.005)
                    if "nu" in params:
                        params["nu"] += random.gauss(0.0, 0.01)

                    # Verify compliance, track drift and clamp parameters
                    clamped_params = self.manager.track_drift_and_ood(websocket, params)

                    req_payload = {
                        "S": spot,
                        "r": r,
                        "q": q,
                        **clamped_params
                    }

                    try:
                        res = await compute_model_greeks(model_name, req_payload)
                        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

                        update_msg = {
                            "type": "update",
                            "timestamp": time.time(),
                            "currency": currency,
                            "model_name": model_name,
                            "spot": spot,
                            "parameters": clamped_params,
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

        # Spawn the producer task in the structured TaskGroup
        prod_task = tg.create_task(risk_metrics_producer())
        
        # Track the subscription tasks for this connection so we can cancel them on unsubscribe
        self.subscription_tasks.setdefault(websocket, []).append(prod_task)

        await self.manager.send_binary(websocket, {
            "type": "subscribed",
            "currency": currency,
            "model_name": model_name,
            "parameters": params
        })

    async def _handle_unsubscribe(self, websocket: WebSocket, send_ack: bool = True) -> None:
        """Cancels stream loop tasks associated with this connection."""
        tasks = self.subscription_tasks.pop(websocket, [])
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if send_ack:
            await self.manager.send_binary(websocket, {"type": "unsubscribed"})

    async def _handle_stress(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Calculates Greeks for a one-off stress scenario and returns them."""
        params = data.get("parameters", {})
        model_name = data.get("model_name", "rough_heston").lower()
        S = float(data.get("S", 5000.0 if model_name not in ("sabr", "rough_heston") else 65000.0))
        r = float(data.get("r", 0.05))
        q = float(data.get("q", 0.0))

        # Check compliance & clamp stress-testing parameters
        params = self.manager.track_drift_and_ood(websocket, params)

        req_payload = {
            "S": S,
            "r": r,
            "q": q,
            **params
        }

        t0 = time.perf_counter()
        try:
            res = await compute_model_greeks(model_name, req_payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            await self.manager.send_binary(websocket, {
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
            await self.manager.send_binary(websocket, {
                "type": "error",
                "message": f"Stress test failed: {str(exc)}"
            })
