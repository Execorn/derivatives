"""
fx_data.py — Garman-Kohlhagen pricing, delta conventions, strike inversion, and market data loaders.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton
from typing import Dict, Tuple, Union, Optional

# ── 1. Garman-Kohlhagen Option Pricing & Delta Conventions ──

def gk_price(
    F: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float,
    option_type: str = "call"
) -> float:
    """
    Computes the Garman-Kohlhagen currency option price.
    
    Parameters
    ----------
    F : float
        Forward exchange rate (S * exp((r_d - r_f)*T)).
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r_d : float
        Domestic risk-free interest rate.
    r_f : float
        Foreign risk-free interest rate.
    vol : float
        Implied volatility.
    option_type : str
        'call' or 'put'.
        
    Returns
    -------
    price : float
        Option price.
    """
    if T <= 0.0 or vol <= 0.0 or K <= 0.0 or F <= 0.0:
        return 0.0
        
    phi = 1.0 if option_type.lower() in ("call", "c") else -1.0
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    # V = phi * e^{-r_d * T} * (F * N(phi * d1) - K * N(phi * d2))
    price = phi * np.exp(-r_d * T) * (F * norm.cdf(phi * d1) - K * norm.cdf(phi * d2))
    return max(price, 0.0)


def gk_delta(
    F: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float,
    option_type: str = "call",
    delta_type: str = "spot_pna"
) -> float:
    """
    Computes the Garman-Kohlhagen option delta according to one of the 4 conventions.
    
    Conventions:
    - spot_pna     : Spot Delta Premium-Non-Adjusted
    - spot_pa      : Spot Delta Premium-Adjusted
    - forward_pna  : Forward Delta Premium-Non-Adjusted
    - forward_pa   : Forward Delta Premium-Adjusted
    """
    if T < 0.0 or vol < 0.0:
        raise ValueError("T and vol must be non-negative")
    if T == 0.0 or vol == 0.0:
        is_call = option_type.lower() in ("call", "c")
        factor = np.exp(-r_f * T) if "spot" in delta_type else 1.0
        if F > K:
            return factor if is_call else 0.0
        elif F < K:
            return 0.0 if is_call else -factor
        else:
            return 0.5 * factor if is_call else -0.5 * factor

    phi = 1.0 if option_type.lower() in ("call", "c") else -1.0
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    if delta_type == "spot_pna":
        return phi * np.exp(-r_f * T) * norm.cdf(phi * d1)
    elif delta_type == "spot_pa":
        return phi * (K / F) * np.exp(-r_f * T) * norm.cdf(phi * d2)
    elif delta_type == "forward_pna":
        return phi * norm.cdf(phi * d1)
    elif delta_type == "forward_pa":
        return phi * (K / F) * norm.cdf(phi * d2)
    else:
        raise ValueError(f"Unknown delta_type: {delta_type}")


def gk_delta_dk(
    F: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float,
    option_type: str = "call",
    delta_type: str = "spot_pna"
) -> float:
    """
    Computes the exact analytical derivative of delta with respect to strike (dDelta / dK).
    """
    if T < 0.0 or vol < 0.0:
        raise ValueError("T and vol must be non-negative")
    if T == 0.0 or vol == 0.0:
        return 0.0

    phi = 1.0 if option_type.lower() in ("call", "c") else -1.0
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    n_d1 = norm.pdf(d1)
    n_d2 = norm.pdf(d2)
    
    if delta_type == "spot_pna":
        return -np.exp(-r_f * T) * n_d1 / (K * vol * np.sqrt(T))
    elif delta_type == "spot_pa":
        term = phi * norm.cdf(phi * d2) / F - n_d2 / (F * vol * np.sqrt(T))
        return np.exp(-r_f * T) * term
    elif delta_type == "forward_pna":
        return -n_d1 / (K * vol * np.sqrt(T))
    elif delta_type == "forward_pa":
        return phi * norm.cdf(phi * d2) / F - n_d2 / (F * vol * np.sqrt(T))
    else:
        raise ValueError(f"Unknown delta_type: {delta_type}")


# ── 2. Newton-Raphson Strike Inversion ──

def invert_gk_delta(
    F: float,
    delta: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float,
    option_type: str = "call",
    delta_type: str = "spot_pna",
    max_iter: int = 100,
    tol: float = 1e-12
) -> float:
    """
    Inverts the Garman-Kohlhagen delta to find the strike price K.
    Uses Newton-Raphson with exact analytical derivatives and step-limiting safeguards.
    """
    phi = 1.0 if option_type.lower() in ("call", "c") else -1.0
    
    # Validate target delta boundaries
    if option_type.lower() in ("call", "c"):
        if delta <= 0.0:
            raise ValueError(f"Call delta must be positive, got {delta}")
    else:
        if delta >= 0.0:
            raise ValueError(f"Put delta must be negative, got {delta}")
            
    # Initial guess K0 using the closed-form non-adjusted formula
    # If the requested delta_type is adjusted, we still use the non-adjusted strike as a starting point.
    if "pna" in delta_type or delta_type == "spot_pna":
        if delta_type == "spot_pna":
            adj_delta = delta * np.exp(r_f * T)
        else:
            adj_delta = delta
            
        val = np.clip(phi * adj_delta, 1e-14, 1.0 - 1e-14)
        d1 = phi * norm.ppf(val)
        K0 = F * np.exp(-d1 * vol * np.sqrt(T) + 0.5 * vol**2 * T)
        
        # Non-adjusted is exact analytically!
        if delta_type in ("spot_pna", "forward_pna"):
            return K0
    else:
        # For adjusted, guess based on non-adjusted
        val = np.clip(phi * delta, 1e-14, 1.0 - 1e-14)
        d1 = phi * norm.ppf(val)
        K0 = F * np.exp(-d1 * vol * np.sqrt(T) + 0.5 * vol**2 * T)

    # Newton-Raphson Loop with Safeguards
    K = K0
    for _ in range(max_iter):
        curr_delta = gk_delta(F, K, T, r_d, r_f, vol, option_type, delta_type)
        diff = curr_delta - delta
        
        if np.abs(diff) < tol:
            return K
            
        dk = gk_delta_dk(F, K, T, r_d, r_f, vol, option_type, delta_type)
        
        if np.abs(dk) < 1e-15:
            # Fallback to a tiny step in the direction of the target
            step = -np.sign(diff) * 0.01 * K
        else:
            step = diff / dk
            
        # Restrict step to at most 50% of current K to ensure K remains strictly positive
        max_step = 0.5 * K
        if np.abs(step) > max_step:
            step = np.sign(step) * max_step
            
        K_new = K - step
        
        if np.abs(K_new - K) < tol:
            return K_new
            
        K = K_new
        
    warnings.warn("Newton-Raphson failed to converge within max_iter iterations", UserWarning)
    return K


# ── 3. Market Data Loaders (FRED / Bloomberg Mock & Cache) ──

class FXMarketDataLoader:
    """
    A robust market data loader that handles local files/caches.
    If the data files do not exist, it generates realistic historical interest rate
    and option smile data, saves them, and loads them to maintain real state and behavior.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def get_fred_path(self, series_id: str) -> str:
        return os.path.join(self.data_dir, f"fred_{series_id.lower()}.csv")
        
    def get_bloomberg_path(self, ticker: str) -> str:
        return os.path.join(self.data_dir, f"bloomberg_{ticker.lower()}_smile.csv")
        
    def load_fred_rates(self, series_id: str = "DFF", force_generate: bool = False) -> pd.DataFrame:
        """
        Loads Federal Funds Rate (or other benchmark interest rates) from local CSV cache.
        If the file does not exist, it generates a realistic dataset first.
        """
        file_path = self.get_fred_path(series_id)
        
        if force_generate or not os.path.exists(file_path):
            # Generate realistic rates starting from 2026-01-01 to 2026-06-23
            dates = pd.date_range(start="2026-01-01", end="2026-06-23", freq="D")
            # Simulating interest rates around 5.25% with minor fluctuations
            np.random.seed(42)
            rates = 0.0525 + np.random.normal(0, 0.0005, len(dates)).cumsum()
            rates = np.clip(rates, 0.01, 0.08) # clamp to realistic rates
            
            df = pd.DataFrame({"Date": dates, "Rate": rates})
            df.to_csv(file_path, index=False)
            
        return pd.read_csv(file_path, parse_dates=["Date"])
        
    def load_bloomberg_smile(self, ticker: str = "EURUSD", force_generate: bool = False) -> pd.DataFrame:
        """
        Loads option smile (deltas and implied volatilities) from local CSV cache.
        If the file does not exist, it generates a realistic volatility smile.
        """
        file_path = self.get_bloomberg_path(ticker)
        
        if force_generate or not os.path.exists(file_path):
            # Generate realistic EURUSD option smile for maturities: 1M, 3M, 6M, 1Y
            # Standard quotes in FX are 25D Put, 10D Put, ATM, 25D Call, 10D Call.
            maturities = [1/12, 3/12, 6/12, 1.0]
            records = []
            
            # Base parameters for smile generation:
            # ATM vol around 10%, skew (25D Call - 25D Put) around -1.0% (EURUSD has negative skew/risk reversal),
            # kurtosis (smile curvature) around 0.5% (butterfly)
            np.random.seed(100)
            
            for T in maturities:
                # Target deltas (signed)
                deltas = [-0.10, -0.25, 0.50, 0.25, 0.10]
                delta_labels = ["10D Put", "25D Put", "ATM", "25D Call", "10D Call"]
                
                atm_vol = 0.095 + 0.01 * np.sqrt(T)
                rr_25d = -0.012  # Risk reversal (Call vol - Put vol)
                bf_25d = 0.004   # Butterfly (average wings - ATM)
                
                rr_10d = -0.021
                bf_10d = 0.009
                
                # Implied vols for each quote
                vols = [
                    atm_vol + bf_10d - 0.5 * rr_10d, # 10D Put
                    atm_vol + bf_25d - 0.5 * rr_25d, # 25D Put
                    atm_vol,                         # ATM
                    atm_vol + bf_25d + 0.5 * rr_25d, # 25D Call
                    atm_vol + bf_10d + 0.5 * rr_10d, # 10D Call
                ]
                
                for label, delta, vol in zip(delta_labels, deltas, vols):
                    records.append({
                        "Maturity": T,
                        "DeltaLabel": label,
                        "DeltaValue": delta,
                        "ImpliedVol": vol
                    })
                    
            df = pd.DataFrame(records)
            df.to_csv(file_path, index=False)
            
        return pd.read_csv(file_path)


class FXDataLoader:
    def __init__(self):
        self._db = {
            "EUR/USD": {
                "spot": 1.10,
                "domestic_rate": 0.03,
                "foreign_rate": 0.01,
                "tenors": [0.25, 0.5, 1.0],
                "atm": [0.08, 0.08, 0.08],
                "rr25": [-0.005, -0.005, -0.005],
                "bf25": [0.002, 0.002, 0.002],
                "rr10": [-0.010, -0.010, -0.010],
                "bf10": [0.004, 0.004, 0.004],
            },
            "GBP/USD": {
                "spot": 1.30,
                "domestic_rate": 0.03,
                "foreign_rate": 0.02,
                "tenors": [0.25, 0.5, 1.0],
                "atm": [0.09, 0.09, 0.09],
                "rr25": [-0.003, -0.003, -0.003],
                "bf25": [0.002, 0.002, 0.002],
                "rr10": [-0.006, -0.006, -0.006],
                "bf10": [0.003, 0.003, 0.003],
            }
        }
        
    def load_quotes(self, pair: str) -> dict:
        if not pair:
            raise ValueError("Currency pair cannot be empty")
        import re
        if not re.match(r"^[A-Z]{3}/[A-Z]{3}$", pair):
            raise ValueError("FX pair must be in format 'XXX/YYY'")
        if pair not in self._db:
            raise ValueError(f"No quotes available for currency pair: {pair}")
        import copy
        return copy.deepcopy(self._db[pair])
