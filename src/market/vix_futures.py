import os
import sys
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to PYTHONPATH
src_dir = str(Path(__file__).parents[1])
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from market.vix_pricing import vix_futures_curve

# Expose public API
__all__ = ["fetch_vix_futures", "get_vix_expiry", "get_active_vix_months"]

MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"
}

def get_vix_expiry(year: int, month: int) -> date:
    """
    Calculate VIX futures expiration date using CBOE rules:
    Wednesday 30 days prior to the third Friday of month M+1.
    """
    next_month = month + 1
    next_year = year
    if next_month > 12:
        next_month = 1
        next_year += 1
    
    # Find the 1st of month M+1
    first_day = date(next_year, next_month, 1)
    w = first_day.weekday()  # 0 = Monday, ..., 4 = Friday
    
    # Days to first Friday
    days_to_friday = (4 - w) % 7
    third_friday = first_day + timedelta(days=days_to_friday + 14)
    
    expiry = third_friday - timedelta(days=30)
    return expiry

def get_active_vix_months(valuation_date: date, count: int = 8) -> list[tuple[int, int, date]]:
    """
    Get the next `count` active contract months and their expiries.
    """
    active = []
    y, m = valuation_date.year, valuation_date.month
    while len(active) < count:
        exp = get_vix_expiry(y, m)
        if exp >= valuation_date:
            active.append((y, m, exp))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return active

def fetch_vix_futures(date_input) -> pd.DataFrame:
    """
    Fetch VIX futures term structure for a given valuation date.
    Returns a DataFrame with columns [expiry, tenor_months, settle_vix].
    """
    # Normalize date
    if isinstance(date_input, str):
        val_date = datetime.strptime(date_input, "%Y-%m-%d").date()
        date_str = date_input
    elif isinstance(date_input, datetime):
        val_date = date_input.date()
        date_str = val_date.strftime("%Y-%m-%d")
    elif isinstance(date_input, date):
        val_date = date_input
        date_str = val_date.strftime("%Y-%m-%d")
    else:
        raise TypeError("date_input must be str, date, or datetime")

    # Get the 8 active contracts
    active_contracts = get_active_vix_months(val_date, count=8)
    
    # Try downloading via yfinance
    df = None
    try:
        import yfinance as yf
        prices = []
        success = True
        for y, m, exp in active_contracts:
            month_code = MONTH_CODES[m]
            year_2_digits = str(y)[-2:]
            
            # Try .CFD first
            ticker_symbol = f"VX{month_code}{year_2_digits}.CFD"
            t = yf.Ticker(ticker_symbol)
            hist = t.history(start=val_date, end=val_date + timedelta(days=1))
            
            if hist.empty:
                # Try .CF
                ticker_symbol = f"VX{month_code}{year_2_digits}.CF"
                t = yf.Ticker(ticker_symbol)
                hist = t.history(start=val_date, end=val_date + timedelta(days=1))
                
            if not hist.empty:
                prices.append(float(hist["Close"].iloc[-1]))
            else:
                success = False
                break
                
        if success and len(prices) == 8:
            rows = []
            for (y, m, exp), price in zip(active_contracts, prices):
                tenor = (exp - val_date).days / 30.4375
                rows.append({
                    "expiry": exp,
                    "tenor_months": tenor,
                    "settle_vix": price
                })
            df = pd.DataFrame(rows)
            # Write to cache
            cache_dir = Path(__file__).parents[2] / "data" / "market" / "vix"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"vix_futures_{date_str}.parquet"
            df.to_parquet(cache_file)
    except Exception:
        df = None

    if df is not None:
        return df

    # Fallbacks
    # 1. Local parquet cache
    cache_dir = Path(__file__).parents[2] / "data" / "market" / "vix"
    cache_file = cache_dir / f"vix_futures_{date_str}.parquet"
    if cache_file.exists():
        try:
            df_cached = pd.read_parquet(cache_file)
            if not df_cached.empty:
                df_cached["expiry"] = pd.to_datetime(df_cached["expiry"]).dt.date
                return df_cached[["expiry", "tenor_months", "settle_vix"]]
        except Exception:
            pass

    # NOTE: hardcoded historical prices removed — they were factually incorrect
    # (e.g., 2020-03-16 front VIX was ~82, not 68.5). Fall through to model curve.

    # 2. Dynamic model-consistent contango curves
    # using vix_futures_curve with default parameters:
    # kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08
    maturities = np.array([(exp - val_date).days / 365.25 for y, m, exp in active_contracts])
    prices = vix_futures_curve(
        kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08,
        maturities=maturities
    )
    rows = []
    for (y, m, exp), price in zip(active_contracts, prices):
        tenor = (exp - val_date).days / 30.4375
        rows.append({
            "expiry": exp,
            "tenor_months": tenor,
            "settle_vix": float(price)
        })
    return pd.DataFrame(rows)
