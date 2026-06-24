"""
tests/test_deribit_data.py — pytest tests for §1.5 Deribit crypto data module.

Tests:
  1. test_parse_instrument_name        — deterministic, no network
  2. test_parse_instrument_name_eth    — ETH put, different date
  3. test_fetch_option_snapshot_schema — mock network, check column schema
  4. test_fetch_option_snapshot_filters— mock network, verify T>0.05 filter
  5. test_build_iv_surface_shape       — given synthetic DF, check (8,11) output
  6. test_build_iv_surface_clip        — values clipped to [0.05, 1.80]
  7. test_estimate_hurst_variogram     — synthetic fBm-like series → H~0.1
  8. test_check_and_clip_params_warning— warns when v0 > FNO bound
"""
import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Path setup (mirrors conftest.py) ────────────────────────────────────────
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from deepvol.market.deribit_data import (
    parse_instrument_name,
    build_iv_surface,
    estimate_hurst_exponent,
    MATURITIES,
    STRIKES,
    _check_and_clip_params,
    _add_log_moneyness,
    fetch_option_snapshot,
)


# ============================================================================
# 1. parse_instrument_name — BTC call
# ============================================================================

def test_parse_instrument_name_btc_call():
    """Known input: BTC-28JUN24-70000-C → exact output."""
    result = parse_instrument_name("BTC-28JUN24-70000-C")
    assert result["coin"] == "BTC"
    assert result["expiry"] == date(2024, 6, 28)
    assert result["strike"] == 70000
    assert result["option_type"] == "C"


def test_parse_instrument_name_eth_put():
    """Known input: ETH-27DEC24-3500-P → exact output."""
    result = parse_instrument_name("ETH-27DEC24-3500-P")
    assert result["coin"] == "ETH"
    assert result["expiry"] == date(2024, 12, 27)
    assert result["strike"] == 3500
    assert result["option_type"] == "P"


def test_parse_instrument_name_invalid():
    """Malformed name raises ValueError."""
    with pytest.raises(ValueError):
        parse_instrument_name("BTC-INVALID")


# ============================================================================
# 2. fetch_option_snapshot — mocked network
# ============================================================================

def _make_fake_api_response(n: int = 150) -> list[dict]:
    """Create synthetic Deribit-style API response with n options."""
    rng = np.random.default_rng(42)
    records = []
    expiries = ["28JUN26", "26SEP26", "26DEC26", "26MAR27"]
    strikes  = [50000, 55000, 60000, 65000, 70000, 75000, 80000]

    for i in range(n):
        exp    = expiries[i % len(expiries)]
        strike = strikes[i % len(strikes)]
        opt_t  = "C" if i % 2 == 0 else "P"
        name   = f"BTC-{exp}-{strike}-{opt_t}"
        mark_iv_pct = float(rng.uniform(40, 90))  # percent, e.g. 55.3
        records.append({
            "instrument_name": name,
            "mark_iv":         mark_iv_pct,
            "bid_iv":          mark_iv_pct - 1.0,
            "ask_iv":          mark_iv_pct + 1.0,
            "underlying_price": 65000.0,
            "open_interest":    float(rng.integers(1, 500)),
            "mark_price":       0.01 * float(rng.uniform(0.5, 5.0)),
        })
    return records


def test_fetch_option_snapshot_schema():
    """
    fetch_option_snapshot returns a DataFrame with required columns.
    Network is mocked — no real HTTP calls.
    """
    fake_data = _make_fake_api_response(150)

    with patch("deepvol.market.deribit_data._async_fetch_snapshot",
               new=AsyncMock(return_value=fake_data)):
        df = asyncio.run(fetch_option_snapshot("BTC"))

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert len(df) > 0, "DataFrame should be non-empty"

    required_cols = [
        "instrument_name", "expiry", "strike", "option_type",
        "mark_iv", "log_moneyness", "T", "underlying_price",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_fetch_option_snapshot_filters():
    """
    Rows with T <= 0.05 or mark_iv <= 0 are excluded.
    """
    fake_data = _make_fake_api_response(150)

    with patch("deepvol.market.deribit_data._async_fetch_snapshot",
               new=AsyncMock(return_value=fake_data)):
        df = asyncio.run(fetch_option_snapshot("BTC"))

    # All surviving rows should have T > 0.05
    assert (df["T"] > 0.05).all(), "All rows must have T > 0.05"
    # All surviving rows should have mark_iv > 0
    assert (df["mark_iv"] > 0).all(), "All rows must have mark_iv > 0"
    # mark_iv should be decimal (< 10, not percent > 40)
    assert df["mark_iv"].max() < 5.0, "mark_iv should be decimal, not percent"


def test_fetch_option_snapshot_min_rows():
    """
    With 150 mocked options (all future-dated), expect >100 surviving rows.
    """
    fake_data = _make_fake_api_response(150)

    with patch("deepvol.market.deribit_data._async_fetch_snapshot",
               new=AsyncMock(return_value=fake_data)):
        df = asyncio.run(fetch_option_snapshot("BTC"))

    assert len(df) > 100, f"Expected >100 rows, got {len(df)}"
    assert "log_moneyness" in df.columns


# ============================================================================
# 3. build_iv_surface — shape & clipping
# ============================================================================

def _make_synthetic_df(n_rows: int = 300) -> pd.DataFrame:
    """Build a synthetic option DataFrame with realistic IV values."""
    rng  = np.random.default_rng(0)
    T    = rng.uniform(0.1, 2.1, n_rows)
    K    = rng.uniform(-0.5, 0.5, n_rows)
    IV   = rng.uniform(0.35, 0.70, n_rows)   # BTC-like 40–70%
    return pd.DataFrame({
        "T":              T,
        "log_moneyness":  K,
        "mark_iv":        IV,
        "option_type":    ["C"] * n_rows,
    })


def test_build_iv_surface_shape():
    """build_iv_surface returns (8, 11) float32 array."""
    df = _make_synthetic_df(300)
    surface = build_iv_surface(df, currency="BTC")
    assert surface.shape == (8, 11), f"Expected (8,11), got {surface.shape}"
    assert surface.dtype == np.float32


def test_build_iv_surface_no_nans():
    """No NaN values in the output surface (NN fallback fills all gaps)."""
    df = _make_synthetic_df(300)
    surface = build_iv_surface(df, currency="BTC")
    assert not np.any(np.isnan(surface)), "Surface contains NaN values"


def test_build_iv_surface_clip():
    """Values are clipped to [0.05, 1.80]."""
    df = _make_synthetic_df(300)
    surface = build_iv_surface(df, currency="BTC")
    assert surface.min() >= 0.04, "Surface min below 0.04"
    assert surface.max() <= 1.81, "Surface max above 1.81"


def test_build_iv_surface_too_few_rows():
    """Raises ValueError if fewer than 10 valid rows."""
    df = pd.DataFrame({"T": [0.5, 1.0], "log_moneyness": [0.0, 0.1],
                        "mark_iv": [0.4, 0.5]})
    with pytest.raises(ValueError, match="Too few valid option quotes"):
        build_iv_surface(df, currency="BTC")


# ============================================================================
# 4. estimate_hurst_exponent
# ============================================================================

def test_estimate_hurst_variogram_rough():
    """
    Anti-persistent (rough) series should give H < 0.5.
    We use alternating-sign increments (extreme anti-persistence).
    """
    rng = np.random.default_rng(123)
    n   = 500
    # Rough series: alternating-sign random walk (H << 0.5)
    increments = rng.choice([-1.0, 1.0], size=n)
    series     = np.cumsum(increments).astype(float)
    H = estimate_hurst_exponent(series, method="variogram")
    assert 0.01 <= H <= 0.49, f"H={H} out of expected range (0.01, 0.49)"


def test_estimate_hurst_rs():
    """R/S method should also return a valid H in (0, 0.5)."""
    rng = np.random.default_rng(42)
    series = np.cumsum(rng.standard_normal(400))
    H = estimate_hurst_exponent(series, method="rs")
    assert 0.01 <= H <= 0.49


def test_estimate_hurst_too_few():
    """Raises ValueError for series shorter than 20 points."""
    with pytest.raises(ValueError, match="at least 20"):
        estimate_hurst_exponent(np.array([0.1, 0.2, 0.3]))


# ============================================================================
# 5. Parameter clipping / warning
# ============================================================================

def test_check_and_clip_params_in_range():
    """No warning when all params are within FNO training range."""
    params  = {"v0": 0.06, "sigma": 0.8, "rho": -0.50, "H": 0.08}
    clipped = _check_and_clip_params(params, currency="BTC")
    assert clipped["v0"] == pytest.approx(0.06)
    assert clipped["sigma"] == pytest.approx(0.8)


def test_check_and_clip_params_out_of_range(recwarn):
    """
    v0=0.55 exceeds FNO bound of 0.25 → UserWarning issued, value clipped.
    """
    params  = {"v0": 0.55, "sigma": 2.0, "rho": -0.30, "H": 0.08}
    clipped = _check_and_clip_params(params, currency="BTC")
    # Clipped to FNO training bounds
    assert clipped["v0"] == pytest.approx(0.25)   # FNO upper bound
    assert clipped["sigma"] == pytest.approx(1.5)  # FNO sigma upper bound


# ============================================================================
# 6. Grids match expected MATURITIES / STRIKES
# ============================================================================

def test_grid_constants():
    """MATURITIES and STRIKES match documented FNO grid."""
    assert len(MATURITIES) == 8
    assert len(STRIKES) == 11
    assert MATURITIES[0] == pytest.approx(0.1)
    assert MATURITIES[-1] == pytest.approx(2.0)
    assert STRIKES[0] == pytest.approx(-0.5)
    assert STRIKES[-1] == pytest.approx(0.5)
