"""
Hurst Exponent Dynamics Study analysis tools.
"""
from __future__ import annotations

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root and src to path for robust imports
project_root = Path(__file__).resolve().parents[3]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from deepvol.calibration.batch_calibration import calibrate_batch, CalibrationResult, results_to_dataframe


def run_historical_study(
    start: str,
    end: str,
    currency: str = "SPX",
    chunk_size: int = 5,
    max_workers: int = 4,
    device: str = "auto",
) -> pd.DataFrame:
    """
    Run historical study of Hurst exponent dynamics by calibrating Rough Heston day-by-day.
    
    This function is resume-capable. Results are saved incrementally to a JSON file
    in results/hurst_dynamics/.
    
    Parameters
    ----------
    start : str
        Start date in ISO format, e.g. '2024-01-01'
    end : str
        End date in ISO format, e.g. '2024-03-31'
    currency : str
        Currency to study ('SPX', 'BTC', or 'ETH')
    chunk_size : int
        Number of dates to calibrate in a single batch before saving
    max_workers : int
        Number of worker threads for parallel fetching
    device : str
        PyTorch device ('auto', 'cuda', 'cpu')
        
    Returns
    -------
    pd.DataFrame
        DataFrame of calibrated parameters and metrics for all dates.
    """
    # BUG-13 fix: explicitly reload v3 normalizers before calibrating.
    # The old code mutated calibrate._NORM_VERSIONS globally (side effect).
    # Now we use _load_normalizers("v3") which is idempotent and safe.
    from deepvol.calibration import calibrate_bfgs as _calibrate_mod
    _calibrate_mod._load_normalizers("v3")

    currency_upper = currency.upper()
    results_dir = project_root / "results" / "hurst_dynamics"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = results_dir / f"{currency_upper}_hurst_study.json"
    
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
    
    # Generate dates range (business days for options trading)
    all_dates = pd.bdate_range(start=start, end=end).strftime("%Y-%m-%d").tolist()
    missing_dates = [d for d in all_dates if d not in completed_dates]
    
    if missing_dates:
        print(f"Running study for {len(missing_dates)} missing dates out of {len(all_dates)} total dates.")
        for i in range(0, len(missing_dates), chunk_size):
            chunk = missing_dates[i : i + chunk_size]
            print(f"Calibrating chunk: {chunk}")
            chunk_results = calibrate_batch(
                dates=chunk,
                currency=currency_upper,
                max_workers=max_workers,
                device=device,
                verbose=True,
            )
            existing_results.extend(chunk_results)
            # Sort results by date
            existing_results.sort(key=lambda r: r.date)
            
            # Incremental save
            serialized = [r.to_dict() for r in existing_results]
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2, default=str)
            print(f"Saved {len(existing_results)} total results to {file_path}")
    else:
        print(f"All {len(all_dates)} dates already calibrated. Returning cached results.")
        
    return results_to_dataframe(existing_results)


def compute_hurst_statistics(df: pd.DataFrame) -> dict:
    """
    Compute statistical properties of H (mean, std, autocorrelation at various lags).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing calibration results, must have an 'H' column.
        
    Returns
    -------
    dict
        Dictionary containing statistical metrics.
    """
    if df.empty or "H" not in df.columns:
        return {
            "mean": np.nan,
            "std": np.nan,
            "autocorr_lag_1": np.nan,
            "autocorr_lag_5": np.nan,
            "autocorr_lag_10": np.nan,
            "autocorr_lag_20": np.nan,
        }
        
    h_series = df["H"].dropna()
    if len(h_series) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "autocorr_lag_1": np.nan,
            "autocorr_lag_5": np.nan,
            "autocorr_lag_10": np.nan,
            "autocorr_lag_20": np.nan,
        }
        
    stats = {
        "mean": float(h_series.mean()),
        "std": float(h_series.std()),
    }
    
    for lag in [1, 5, 10, 20]:
        if len(h_series) > lag:
            # Series.autocorr computes pearson correlation between series and its lagged version
            ac = h_series.autocorr(lag=lag)
            stats[f"autocorr_lag_{lag}"] = float(ac) if np.isfinite(ac) else np.nan
        else:
            stats[f"autocorr_lag_{lag}"] = np.nan
            
    return stats


def detect_regime_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect structural regime changes in the H time series using the Pettitt test.
    
    Implemented in pure Python/NumPy (no external statistical packages).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing calibration results, must have 'date' and 'H' columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing change point details:
        Columns: change_point_date, change_point_index, p_value, statistic,
                 mean_before, mean_after, is_significant
    """
    cols = [
        "change_point_date",
        "change_point_index",
        "p_value",
        "statistic",
        "mean_before",
        "mean_after",
        "is_significant",
    ]
    
    if df.empty or "H" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=cols)
        
    clean_df = df.dropna(subset=["H"]).copy()
    if len(clean_df) < 4:
        return pd.DataFrame(columns=cols)
        
    X = clean_df["H"].values
    dates = clean_df["date"].values
    n = len(X)
    
    # Pettitt test
    # D[i, j] = sign(X_i - X_j)
    diff = X[:, None] - X[None, :]
    D = np.sign(diff)
    V = D.sum(axis=1)
    U = np.cumsum(V)[:-1]
    
    abs_U = np.abs(U)
    K = np.max(abs_U)
    # argmax returns index of max value in U, which is 1-indexed at index + 1
    tau = int(np.argmax(abs_U)) + 1
    
    # Calculate p-value
    p_value = 2.0 * np.exp(-6.0 * (K ** 2) / (n ** 3 + n ** 2))
    p_value = min(p_value, 1.0)
    
    mean_before = float(np.mean(X[:tau]))
    mean_after = float(np.mean(X[tau:]))
    change_date = str(dates[tau])
    
    result_row = {
        "change_point_date": change_date,
        "change_point_index": tau,
        "p_value": float(p_value),
        "statistic": float(K),
        "mean_before": mean_before,
        "mean_after": mean_after,
        "is_significant": bool(p_value < 0.05),
    }
    
    return pd.DataFrame([result_row])
