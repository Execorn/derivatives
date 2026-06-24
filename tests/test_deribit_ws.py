"""
Tests for §P2-B3 Deribit WebSocket streaming client.

All tests use mocked WebSocket connections — no real network calls.
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from deepvol.market.deribit_ws import (
    DeribitWSClient,
    _parse_tick,
    _tte_years,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_btc_tick(
    instrument: str = "BTC-27JUN27-70000-C",
    mark_iv: float = 50.0,       # percent
    underlying_price: float = 65000.0,
    bid_iv: float = 49.0,
    ask_iv: float = 51.0,
    open_interest: int = 100,
) -> dict:
    return {
        "instrument_name": instrument,
        "mark_iv": mark_iv,
        "underlying_price": underlying_price,
        "bid_iv": bid_iv,
        "ask_iv": ask_iv,
        "greeks": {"delta": 0.5},
        "open_interest": open_interest,
    }


def make_subscription_msg(channel: str, data: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "subscription",
        "params": {"channel": channel, "data": data},
    }


# ── Unit: _tte_years ───────────────────────────────────────────────────────────

def test_tte_years_future():
    today = date(2025, 1, 1)
    exp   = date(2025, 7, 1)
    tte   = _tte_years(exp, today)
    assert 0.4 < tte < 0.6


def test_tte_years_past_returns_zero():
    today = date(2025, 6, 1)
    exp   = date(2025, 1, 1)
    tte   = _tte_years(exp, today)
    assert tte == 0.0


# ── Unit: _parse_tick ─────────────────────────────────────────────────────────

def test_parse_tick_valid():
    tick = make_btc_tick()
    row  = _parse_tick(tick)
    assert row is not None
    assert row["coin"] == "BTC"
    assert row["option_type"] == "C"
    assert abs(row["mark_iv"] - 0.50) < 1e-6   # % → decimal
    # log_moneyness = log(70000 / 65000)
    expected_lm = math.log(70000 / 65000)
    assert abs(row["log_moneyness"] - expected_lm) < 1e-6


def test_parse_tick_log_moneyness_atm():
    """ATM option → log_moneyness ≈ 0."""
    tick = make_btc_tick(
        instrument="BTC-27JUN27-65000-C",
        underlying_price=65000.0,
    )
    row = _parse_tick(tick)
    assert row is not None
    assert abs(row["log_moneyness"]) < 1e-6


def test_parse_tick_malformed_returns_none():
    assert _parse_tick({}) is None
    assert _parse_tick({"instrument_name": "NOT-VALID"}) is None


def test_parse_tick_expired_returns_none():
    """Instruments with T < 0.05 should be filtered."""
    # expiry in the past → T ≈ 0
    tick = make_btc_tick(instrument="BTC-01JAN22-70000-C")
    row  = _parse_tick(tick)
    assert row is None


def test_parse_tick_zero_iv_returns_none():
    tick = make_btc_tick(mark_iv=0.0)
    row  = _parse_tick(tick)
    assert row is None


def test_parse_tick_columns():
    tick = make_btc_tick()
    row  = _parse_tick(tick)
    expected = {
        "instrument_name", "coin", "expiry", "strike", "option_type",
        "mark_iv", "underlying_price", "log_moneyness", "T",
        "bid_iv", "ask_iv", "delta", "open_interest",
    }
    assert expected <= set(row.keys())


# ── Unit: DeribitWSClient instantiation ───────────────────────────────────────

def test_client_instantiation():
    client = DeribitWSClient()
    assert client.get_spot() is None
    assert client.get_current_surface() is None
    assert not client._connected


# ── Unit: _handle_message ─────────────────────────────────────────────────────

def test_handle_ticker_message_updates_rows():
    client = DeribitWSClient()
    tick   = make_btc_tick()
    msg    = make_subscription_msg(
        f"ticker.{tick['instrument_name']}.any", tick
    )
    client._handle_message(msg)
    df = client.get_current_surface()
    assert df is not None
    assert len(df) == 1
    assert df.iloc[0]["coin"] == "BTC"


def test_handle_price_index_message_updates_spot():
    client = DeribitWSClient()
    msg    = make_subscription_msg(
        "deribit_price_index.btc_usd",
        {"price": 65432.10},
    )
    client._handle_message(msg)
    assert abs(client.get_spot() - 65432.10) < 1e-6


def test_handle_malformed_ticker_message_no_crash():
    client = DeribitWSClient()
    msg    = make_subscription_msg("ticker.BTC-JUNK.any", {"bad": "data"})
    # Should not raise
    client._handle_message(msg)
    assert client.get_current_surface() is None


def test_handle_unknown_method_no_crash():
    client = DeribitWSClient()
    client._handle_message({"method": "heartbeat", "params": {}})
    # Nothing should break


# ── Unit: multiple ticks accumulate ───────────────────────────────────────────

def test_multiple_ticks_accumulate_rows():
    client = DeribitWSClient()
    instruments = [
        "BTC-27JUN27-60000-C",
        "BTC-27JUN27-65000-C",
        "BTC-27JUN27-70000-C",
    ]
    for inst in instruments:
        tick = make_btc_tick(instrument=inst)
        msg  = make_subscription_msg(f"ticker.{inst}.any", tick)
        client._handle_message(msg)

    df = client.get_current_surface()
    assert df is not None
    assert len(df) == 3


def test_tick_overwrites_same_instrument():
    client = DeribitWSClient()
    inst   = "BTC-27JUN27-70000-C"
    for iv in [50.0, 55.0, 60.0]:
        tick = make_btc_tick(instrument=inst, mark_iv=iv)
        msg  = make_subscription_msg(f"ticker.{inst}.any", tick)
        client._handle_message(msg)

    df = client.get_current_surface()
    assert len(df) == 1                                 # one unique instrument
    assert abs(df.iloc[0]["mark_iv"] - 0.60) < 1e-6   # latest IV kept


# ── Async: connect/disconnect with mock WS ────────────────────────────────────

@pytest.mark.asyncio
async def test_connect_disconnect_mock():
    """Verify connect/disconnect lifecycle with mocked aiohttp."""
    mock_ws   = AsyncMock()
    mock_ws.closed = False
    mock_ws.__aiter__ = MagicMock(return_value=iter([]))   # no messages

    mock_session = AsyncMock()
    mock_session.ws_connect = AsyncMock(return_value=mock_ws)
    mock_session.closed = False

    with patch("deepvol.market.deribit_ws.aiohttp.ClientSession", return_value=mock_session):
        client = DeribitWSClient()
        await client.connect()
        assert client._connected

        await client.disconnect()
        assert not client._connected


@pytest.mark.asyncio
async def test_subscribe_ticker_sends_jsonrpc():
    """subscribe_ticker should send a public/subscribe JSON-RPC call."""
    mock_ws   = AsyncMock()
    mock_ws.closed = False
    sent_messages = []

    async def fake_send_str(data: str) -> None:
        sent_messages.append(json.loads(data))

    mock_ws.send_str = fake_send_str
    mock_ws.__aiter__ = MagicMock(return_value=iter([]))

    mock_session = AsyncMock()
    mock_session.ws_connect = AsyncMock(return_value=mock_ws)
    mock_session.closed = False

    with patch("deepvol.market.deribit_ws.aiohttp.ClientSession", return_value=mock_session):
        client = DeribitWSClient()
        client._ws = mock_ws
        client._connected = True
        await client.subscribe_ticker("BTC-27JUN25-70000-C")

    assert any(
        m.get("method") == "public/subscribe" for m in sent_messages
    ), "Expected public/subscribe message"
    assert any(
        "ticker.BTC-27JUN25-70000-C.any" in str(m) for m in sent_messages
    ), "Expected correct channel in message"
