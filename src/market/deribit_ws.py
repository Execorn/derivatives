"""
§P2-B3  Deribit WebSocket Streaming Client.

Real-time IV surface updates via Deribit's JSON-RPC-over-WebSocket API.

WebSocket endpoint: wss://www.deribit.com/ws/api/v2

Message flow
------------
1. connect() → open WebSocket, start heartbeat task
2. subscribe_ticker(instrument) → JSON-RPC subscribe to ticker.{inst}.any
3. Incoming "subscription" messages → _handle_tick() → update in-memory surface
4. stream_iv_surface(currency, callback) → async generator: yield DataFrame on
   every surface update
5. disconnect() / async-with → clean shutdown

Data schema (same as fetch_option_snapshot)
-------------------------------------------
Columns: instrument_name, coin, expiry, strike, option_type, mark_iv (decimal),
         underlying_price, log_moneyness, T, bid_iv, ask_iv, delta, open_interest
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_src = str(Path(__file__).parents[1])
if _src not in sys.path:
    sys.path.insert(0, _src)

try:
    import aiohttp
except ImportError as exc:  # pragma: no cover
    raise ImportError("aiohttp is required: pip install aiohttp") from exc

from market.deribit_data import parse_instrument_name

log = logging.getLogger(__name__)

__all__ = [
    "DeribitWSClient",
    "stream_realtime_surface",
]

# ── Constants ─────────────────────────────────────────────────────────────────
WS_URL          = "wss://www.deribit.com/ws/api/v2"
HEARTBEAT_SECS  = 10
MAX_RETRIES     = 3
RECONNECT_DELAY = 2.0   # seconds between retries
_MIN_T          = 0.05  # minimum time-to-expiry kept in surface (years)
_MIN_IV         = 1e-4  # minimum IV kept (decimal)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tte_years(expiry: date, today: Optional[date] = None) -> float:
    """Calendar days remaining / 365.25."""
    today = today or datetime.now(timezone.utc).date()
    return max((expiry - today).days / 365.25, 0.0)


def _parse_tick(data: Dict[str, Any]) -> Optional[dict]:
    """
    Parse one Deribit ticker subscription payload into a row dict.

    Returns None if the tick is malformed or below quality thresholds.
    """
    try:
        inst      = data["instrument_name"]
        mark_iv   = float(data.get("mark_iv", 0.0)) / 100.0   # % → decimal
        und_price = float(data.get("underlying_price", data.get("mark_price", 0.0)))
        bid_iv    = float(data.get("bid_iv", 0.0)) / 100.0
        ask_iv    = float(data.get("ask_iv", 0.0)) / 100.0
        delta     = float(data.get("greeks", {}).get("delta", float("nan")))
        oi        = int(data.get("open_interest", 0))

        parsed    = parse_instrument_name(inst)
        expiry    = parsed["expiry"]
        strike    = float(parsed["strike"])
        opt_type  = parsed["option_type"]
        coin      = parsed["coin"]

        T = _tte_years(expiry)
        if T < _MIN_T or mark_iv < _MIN_IV or und_price <= 0:
            return None

        F              = und_price                          # underlying IS forward for crypto
        log_moneyness  = math.log(strike / F)

        return {
            "instrument_name": inst,
            "coin":            coin,
            "expiry":          expiry,
            "strike":          strike,
            "option_type":     opt_type,
            "mark_iv":         mark_iv,
            "underlying_price": und_price,
            "log_moneyness":   log_moneyness,
            "T":               T,
            "bid_iv":          bid_iv,
            "ask_iv":          ask_iv,
            "delta":           delta,
            "open_interest":   oi,
        }
    except Exception as exc:
        log.debug("Malformed tick %s: %s", data.get("instrument_name", "?"), exc)
        return None


# ── Main class ────────────────────────────────────────────────────────────────

class DeribitWSClient:
    """
    Async WebSocket client for Deribit real-time option data.

    Usage::

        async with DeribitWSClient() as client:
            await client.subscribe_index("BTC")
            # subscribe to a few instruments or use stream_iv_surface
            async for df in client.stream_iv_surface("BTC"):
                print(df.head())
    """

    def __init__(self) -> None:
        self._ws:          Optional[aiohttp.ClientWebSocketResponse] = None
        self._session:     Optional[aiohttp.ClientSession]           = None
        self._rows:        Dict[str, dict]                           = {}  # inst → row
        self._spot:        Optional[float]                           = None
        self._hb_task:     Optional[asyncio.Task]                    = None
        self._recv_task:   Optional[asyncio.Task]                    = None
        self._surface_evt: asyncio.Event                             = asyncio.Event()
        self._connected:   bool                                      = False
        self._msg_id:      int                                       = 0
        self._subscribed:  set[str]                                  = set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open WebSocket connection and start background tasks."""
        self._session = aiohttp.ClientSession()
        self._ws      = await self._session.ws_connect(WS_URL)
        self._connected = True
        log.info("Connected to Deribit WS: %s", WS_URL)
        self._hb_task   = asyncio.create_task(self._heartbeat_loop())
        self._recv_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self) -> None:
        """Gracefully close the connection and cancel background tasks."""
        self._connected = False
        for task in (self._hb_task, self._recv_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        log.info("Disconnected from Deribit WS")

    async def __aenter__(self) -> "DeribitWSClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    # ── Subscriptions ─────────────────────────────────────────────────────────

    async def subscribe_ticker(self, instrument_name: str) -> None:
        """Subscribe to ticker.{instrument_name}.any channel."""
        channel = f"ticker.{instrument_name}.any"
        if channel in self._subscribed:
            return
        await self._send({
            "jsonrpc": "2.0",
            "id":      self._next_id(),
            "method":  "public/subscribe",
            "params":  {"channels": [channel]},
        })
        self._subscribed.add(channel)

    async def subscribe_index(self, currency: str) -> None:
        """Subscribe to deribit_price_index.{currency}_usd for live spot."""
        channel = f"deribit_price_index.{currency.lower()}_usd"
        if channel in self._subscribed:
            return
        await self._send({
            "jsonrpc": "2.0",
            "id":      self._next_id(),
            "method":  "public/subscribe",
            "params":  {"channels": [channel]},
        })
        self._subscribed.add(channel)

    # ── Surface access ────────────────────────────────────────────────────────

    def get_current_surface(self) -> Optional[pd.DataFrame]:
        """Return the latest in-memory IV surface as a DataFrame, or None."""
        if not self._rows:
            return None
        return pd.DataFrame(list(self._rows.values()))

    def get_spot(self) -> Optional[float]:
        """Return the latest spot price, or None."""
        return self._spot

    # ── Streaming generator ────────────────────────────────────────────────────

    async def stream_iv_surface(
        self,
        currency: str,
        callback: Optional[Callable[[pd.DataFrame], None]] = None,
    ):
        """
        Async generator that yields an updated IV surface DataFrame each time
        any subscribed ticker fires.

        Automatically fetches the current instrument list via REST, subscribes
        to all of them, and subscribes to the spot index.

        Parameters
        ----------
        currency : str
            'BTC' or 'ETH'
        callback : Callable, optional
            Called with the new DataFrame on every update (in addition to yield)
        """
        await self.subscribe_index(currency)

        # Fetch active instruments via REST and bulk-subscribe
        instruments = await self._fetch_instruments(currency)
        for inst in instruments:
            await self.subscribe_ticker(inst)

        while self._connected:
            await self._surface_evt.wait()
            self._surface_evt.clear()
            df = self.get_current_surface()
            if df is not None and len(df) > 0:
                if callback is not None:
                    callback(df)
                yield df

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def _send(self, payload: dict) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.send_str(json.dumps(payload))

    async def _heartbeat_loop(self) -> None:
        """Send JSON-RPC ping every HEARTBEAT_SECS to keep the connection alive."""
        try:
            while self._connected:
                await asyncio.sleep(HEARTBEAT_SECS)
                await self._send({
                    "jsonrpc": "2.0",
                    "id":      self._next_id(),
                    "method":  "public/test",
                    "params":  {},
                })
        except asyncio.CancelledError:
            pass

    async def _receive_loop(self) -> None:
        """Dispatch incoming WebSocket messages."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_message(json.loads(msg.data))
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    log.warning("WS closed/error: %s", msg)
                    break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.exception("WS receive error: %s", exc)
        finally:
            self._connected = False
            self._surface_evt.set()  # Unblock waiters if connection drops

    def _handle_message(self, msg: dict) -> None:
        """Route a parsed JSON message."""
        method = msg.get("method")
        if method == "subscription":
            params  = msg.get("params", {})
            channel = params.get("channel", "")
            data    = params.get("data", {})

            if channel.startswith("ticker."):
                row = _parse_tick(data)
                if row is not None:
                    self._rows[row["instrument_name"]] = row
                    self._surface_evt.set()

            elif "price_index" in channel:
                price = data.get("price")
                if price is not None:
                    self._spot = float(price)

    async def _fetch_instruments(self, currency: str) -> list[str]:
        """Fetch active option instrument names from Deribit REST."""
        url = (
            f"https://www.deribit.com/api/v2/public/get_instruments"
            f"?currency={currency}&kind=option&expired=false"
        )
        try:
            async with self._session.get(url) as resp:
                data = await resp.json()
            return [i["instrument_name"] for i in data.get("result", [])]
        except Exception as exc:
            log.warning("Could not fetch instruments for %s: %s", currency, exc)
            return []


# ── Convenience function ───────────────────────────────────────────────────────

async def stream_realtime_surface(
    currency: str = "BTC",
    on_update: Optional[Callable[[pd.DataFrame], None]] = None,
    max_duration_seconds: float = 60.0,
) -> pd.DataFrame:
    """
    Stream live Deribit IV surface for up to `max_duration_seconds` seconds.

    Parameters
    ----------
    currency : str
        'BTC' or 'ETH'
    on_update : Callable, optional
        Called with updated DataFrame on each surface update
    max_duration_seconds : float
        Stop streaming after this many seconds (default 60)

    Returns
    -------
    pd.DataFrame
        Final snapshot of the IV surface
    """
    final_df: Optional[pd.DataFrame] = None
    retries = 0

    while retries <= MAX_RETRIES:
        try:
            async with DeribitWSClient() as client:
                deadline = asyncio.get_event_loop().time() + max_duration_seconds
                async for df in client.stream_iv_surface(currency, callback=on_update):
                    final_df = df
                    if asyncio.get_event_loop().time() >= deadline:
                        break
                if not client._connected and asyncio.get_event_loop().time() < deadline:
                    raise ConnectionError("WebSocket disconnected prematurely")
            break
        except Exception as exc:
            retries += 1
            log.warning("WS error (attempt %d/%d): %s", retries, MAX_RETRIES, exc)
            if retries > MAX_RETRIES:
                raise
            await asyncio.sleep(RECONNECT_DELAY)

    if final_df is None:
        return pd.DataFrame()
    return final_df
