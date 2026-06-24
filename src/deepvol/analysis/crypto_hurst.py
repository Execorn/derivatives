"""
Crypto Hurst Exponent Dynamics Study analysis tools.
"""
from __future__ import annotations

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

# Add project root and src to path for robust imports
project_root = Path(__file__).resolve().parents[3]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from deepvol.market.deribit_data import build_iv_surface, fetch_option_snapshot
from deepvol.calibration.batch_calibration import calibrate_batch, CalibrationResult, results_to_dataframe


def align_crypto_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure mark_iv and other IV columns are in decimal format (divided by 100).
    If values are large (e.g. > 2.0), divide by 100.
    """
    df = df.copy()
    for col in ["mark_iv", "bid_iv", "ask_iv"]:
        if col in df.columns:
            if df[col].max() > 2.0:
                df[col] = df[col] / 100.0
    return df


def generate_mock_crypto_data(date_str: str, currency: str = "BTC") -> pd.DataFrame:
    """
    Generate mock options chain for crypto with expiry year 2027+ for test stability.
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    study_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    # FNO Grid values
    maturities = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
    strikes = np.linspace(-0.5, 0.5, 11)
    
    underlying_price = 50000.0 if currency.upper() == "BTC" else 3000.0
    
    records = []
    for T_val in maturities:
        # Force expiry year to be 2027+ for test stability
        expiry_date = study_date + timedelta(days=int(T_val * 365.25))
        year_shift = max(0, 2027 - expiry_date.year)
        if year_shift > 0:
            expiry_date = expiry_date.replace(year=expiry_date.year + year_shift)
            
        T_actual = (expiry_date - study_date).days / 365.25
        
        # Format expiry string as e.g. 28JUN27
        month_str = expiry_date.strftime("%b").upper()
        day_str = expiry_date.strftime("%d")
        year_str = expiry_date.strftime("%y")
        expiry_str = f"{day_str}{month_str}{year_str}"
        
        for log_mon in strikes:
            strike_val = underlying_price * np.exp(log_mon)
            for opt_type in ["C", "P"]:
                name = f"{currency.upper()}-{expiry_str}-{int(strike_val)}-{opt_type}"
                # Generate realistic IV in percent (e.g. 55.0) to test alignment
                iv_percent = 50.0 + 10.0 * log_mon**2 - 5.0 * log_mon
                
                records.append({
                    "instrument_name": name,
                    "coin": currency.upper(),
                    "expiry": expiry_date,
                    "strike": float(strike_val),
                    "option_type": opt_type,
                    "mark_iv": iv_percent,  # represented in percent as in raw Deribit API
                    "bid_iv": iv_percent - 2.0,
                    "ask_iv": iv_percent + 2.0,
                    "underlying_price": underlying_price,
                    "open_interest": 100.0,
                    "mark_price": 0.05 * underlying_price,
                    "log_moneyness": log_mon,
                    "T": float(T_actual),
                })
                
    return pd.DataFrame(records)


def run_crypto_historical_study(
    start: str,
    end: str,
    currency: str = "BTC",
    test_mode: bool = False,
    chunk_size: int = 5,
    max_workers: int = 4,
    device: str = "auto",
) -> pd.DataFrame:
    """
    Run historical study for crypto option data from Deribit (BTC/ETH).
    
    This function is resume-capable. Results are saved incrementally to a JSON file.
    
    Parameters
    ----------
    start : str
        Start date in ISO format.
    end : str
        End date in ISO format.
    currency : str
        'BTC' or 'ETH'
    test_mode : bool
        If True, runs in test mode using mock tickers with year 2027+ for test stability.
    chunk_size : int
        Number of dates to calibrate in a single batch before saving.
    max_workers : int
        Number of worker threads.
    device : str
        PyTorch device ('auto', 'cuda', 'cpu')
        
    Returns
    -------
    pd.DataFrame
        DataFrame of calibrated parameters and metrics.
    """
    from deepvol.calibration import calibrate_bfgs as calibrate
    calibrate._NORM_VERSIONS["v1"] = calibrate._NORM_VERSIONS["v3"]
    calibrate._param_norm = None
    calibrate._iv_norm = None

    currency_upper = currency.upper()
    results_dir = project_root / "results" / "hurst_dynamics"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = "_test" if test_mode else ""
    file_path = results_dir / f"{currency_upper}_hurst_study{suffix}.json"
    
    existing_results: List[CalibrationResult] = []
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            existing_results = [CalibrationResult.from_dict(d) for d in data]
            print(f"Loaded {len(existing_results)} existing results from {file_path}")
        except Exception as e:
            warnings.warn(f"Failed to load existing results from {file_path}: {e}. Starting fresh.")
            
    completed_dates = {r.date for r in existing_results}
    
    # Generate business dates range
    all_dates = pd.bdate_range(start=start, end=end).strftime("%Y-%m-%d").tolist()
    missing_dates = [d for d in all_dates if d not in completed_dates]
    
    if missing_dates:
        print(f"Running crypto study for {len(missing_dates)} missing dates.")
        for i in range(0, len(missing_dates), chunk_size):
            chunk = missing_dates[i : i + chunk_size]
            print(f"Calibrating chunk: {chunk}")
            
            # Prepare pre-computed target surfaces for the chunk
            target_surfaces = {}
            for d in chunk:
                # 1. Fetch/generate data
                cache_file = project_root / "data" / "market" / "deribit" / f"{currency_upper.lower()}_chain_{d}.parquet"
                if test_mode:
                    df = generate_mock_crypto_data(d, currency=currency_upper)
                elif cache_file.exists():
                    df = pd.read_parquet(cache_file)
                    print(f"Loaded cached data for {d} from {cache_file}")
                else:
                    # Fallback to generating mock data if not in test mode but cache is missing
                    # since historical Deribit API is not available
                    warnings.warn(f"Cache file {cache_file} not found. Using synthetic/mock generator for {d}.")
                    df = generate_mock_crypto_data(d, currency=currency_upper)
                
                # 2. Align inputs: divide mark_iv by 100 before any computation
                df = align_crypto_inputs(df)
                
                # 3. Build IV surface
                surface = build_iv_surface(df, currency=currency_upper)
                target_surfaces[d] = surface
                
            # 4. Calibrate batch
            chunk_results = calibrate_batch(
                dates=chunk,
                currency=currency_upper,
                max_workers=max_workers,
                device=device,
                target_surfaces=target_surfaces,
                verbose=True,
            )
            
            # 5. Warn and clip v0 parameters if they exceed FNO training range bounds
            for r in chunk_results:
                v0 = r.params.get("v0", 0.08)
                # FNO training range for v0 is [0.01, 0.25]
                if v0 < 0.01 or v0 > 0.25:
                    msg = (
                        f"WARNING [crypto]: Calibrated v0={v0:.4f} for {currency_upper} on {r.date} "
                        f"exceeds FNO training range bounds [0.01, 0.25]. Clipping to bounds."
                    )
                    warnings.warn(msg, UserWarning)
                    print(msg)
                    r.params["v0"] = float(np.clip(v0, 0.01, 0.25))
                    
            existing_results.extend(chunk_results)
            existing_results.sort(key=lambda r: r.date)
            
            # Incremental save
            serialized = [r.to_dict() for r in existing_results]
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2, default=str)
            print(f"Saved {len(existing_results)} total results to {file_path}")
    else:
        print(f"All {len(all_dates)} dates already calibrated. Returning cached results.")
        
    return results_to_dataframe(existing_results)
