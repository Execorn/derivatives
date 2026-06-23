"""
CME Commodity Options and Futures Data Adapter.

Handles CME WTI Crude Oil (CL/LO) options delivery calendars, contract parsing, 
strike cleaning, options-to-futures matching, and synthetic data generation.
"""

from __future__ import annotations

import datetime
import re
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from scipy.stats import norm

from src.pricing.schwartz_smith import (
    schwartz_smith_price_black76,
    conditional_variance,
    A_factor
)

MONTH_CODES = {
    'F': 1,  # January
    'G': 2,  # February
    'H': 3,  # March
    'J': 4,  # April
    'K': 5,  # May
    'M': 6,  # June
    'N': 7,  # July
    'Q': 8,  # August
    'U': 9,  # September
    'V': 10, # October
    'X': 11, # November
    'Z': 12  # December
}

MONTH_TO_CODE = {v: k for k, v in MONTH_CODES.items()}


def get_cme_calendar() -> CustomBusinessDay:
    """Returns a pandas business day offset based on the US Federal Holiday Calendar."""
    us_cal = USFederalHolidayCalendar()
    return CustomBusinessDay(calendar=us_cal)


def wti_futures_expiry(year: int, month: int) -> datetime.date:
    """
    Calculates the WTI Crude Oil (CL) futures expiration date.
    
    Trading terminates at the close of business on the 3rd business day prior to the 
    25th calendar day of the month preceding the delivery month. If the 25th calendar 
    day of the month is not a business day, trading terminates on the 3rd business day 
    prior to the business day preceding the 25th calendar day.
    """
    usb = get_cme_calendar()
    
    # Month preceding the delivery month
    if month == 1:
        prev_month = 12
        prev_year = year - 1
    else:
        prev_month = month - 1
        prev_year = year
        
    d25 = pd.Timestamp(datetime.date(prev_year, prev_month, 25))
    
    # Find the latest business day on or before the 25th
    start_date = d25 - pd.Timedelta(days=10)
    end_date = d25 + pd.Timedelta(days=5)
    bdays = pd.date_range(start=start_date, end=end_date, freq=usb)
    
    valid_bdays = bdays[bdays <= d25]
    if len(valid_bdays) == 0:
        b_pre = d25
    else:
        b_pre = valid_bdays[-1]
        
    # Expiry is 3 business days prior to b_pre
    expiry = b_pre - 3 * usb
    return expiry.date()


def wti_options_expiry(year: int, month: int) -> datetime.date:
    """
    Calculates the WTI Crude Oil option (LO) expiration date.
    
    Trading terminates 3 business days prior to the underlying futures contract 
    termination date.
    """
    usb = get_cme_calendar()
    fut_expiry = wti_futures_expiry(year, month)
    expiry = pd.Timestamp(fut_expiry) - 3 * usb
    return expiry.date()


def parse_futures_code(code: str, ref_year: int | None = None) -> dict:
    """
    Parses a CME futures contract code (e.g. 'CLZ26', 'CLZ6', 'CL Z26')
    into underlying, month code, month (1-12), and year.
    """
    if ref_year is None:
        ref_year = datetime.date.today().year
        
    # Clean up and normalize
    code = code.replace(" ", "").upper()
    
    m = re.match(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d+)$", code)
    if not m:
        raise ValueError(f"Could not parse futures contract code: {code}")
        
    underlying = m.group(1)
    month_code = m.group(2)
    year_str = m.group(3)
    
    month = MONTH_CODES[month_code]
    
    # Resolve 1-digit, 2-digit, or 4-digit year
    if len(year_str) == 1:
        digit = int(year_str)
        ref_century = (ref_year // 10) * 10
        year = ref_century + digit
        if year < ref_year - 5:
            year += 10
        elif year > ref_year + 5:
            year -= 10
    elif len(year_str) == 2:
        year = 2000 + int(year_str)
    elif len(year_str) == 4:
        year = int(year_str)
    else:
        raise ValueError(f"Invalid year in contract code: {code}")
        
    return {
        "underlying": underlying,
        "month_code": month_code,
        "month": month,
        "year": year
    }


def parse_options_code(code: str, ref_year: int | None = None) -> dict:
    """
    Parses a CME options contract code.
    E.g. 'LOZ26 C7500', 'LO Z26 75.0 C', 'CLZ26 C7500'
    Returns dict with keys: underlying, month, year, strike, option_type.
    """
    if ref_year is None:
        ref_year = datetime.date.today().year
        
    # Standardize spaces and uppercase
    code = " ".join(code.split()).upper()
    
    # Try to determine option type
    option_type = None
    if " CALL" in code or " C " in code or code.endswith("C") or " C" in code:
        option_type = "C"
    elif " PUT" in code or " P " in code or code.endswith("P") or " P" in code:
        option_type = "P"
        
    # Extract strike and option type if adjacent
    strike = None
    m_prefix = re.search(r"([CP])\s*([0-9A-Z.]+)", code)
    m_suffix = re.search(r"([0-9A-Z.]+)\s*([CP])\b", code)
    
    if m_prefix and any(c.isdigit() for c in m_prefix.group(2)):
        option_type = m_prefix.group(1)
        strike_raw = m_prefix.group(2)
        strike_clean = "".join(c for c in strike_raw if c.isdigit() or c == '.')
        if strike_clean:
            strike = float(strike_clean)
    elif m_suffix and any(c.isdigit() for c in m_suffix.group(1)):
        strike_raw = m_suffix.group(1)
        option_type = m_suffix.group(2)
        strike_clean = "".join(c for c in strike_raw if c.isdigit() or c == '.')
        if strike_clean:
            strike = float(strike_clean)
            
    if strike is None:
        parts = code.split()
        for p in parts[1:]:
            p_clean = "".join(c for c in p if c.isdigit() or c == '.')
            if p_clean and not any(c in p for c in "FGHJKMNQUVXZ" if c not in "CP"):
                try:
                    strike = float(p_clean)
                    break
                except ValueError:
                    pass
                    
    # Extract month and year
    m_my = re.search(r"([FGHJKMNQUVXZ])(\d+)", code)
    if not m_my:
        raise ValueError(f"Could not parse option month/year from: {code}")
        
    month_code = m_my.group(1)
    year_str = m_my.group(2)
    month = MONTH_CODES[month_code]
    
    if len(year_str) == 1:
        digit = int(year_str)
        ref_century = (ref_year // 10) * 10
        year = ref_century + digit
        if year < ref_year - 5:
            year += 10
        elif year > ref_year + 5:
            year -= 10
    elif len(year_str) == 2:
        year = 2000 + int(year_str)
    elif len(year_str) == 4:
        year = int(year_str)
    else:
        raise ValueError(f"Invalid year in options code: {code}")
        
    idx = m_my.start()
    prefix = code[:idx].strip()
    underlying = "CL" if prefix == "LO" or not prefix else prefix
        
    return {
        "underlying": underlying,
        "month_code": month_code,
        "month": month,
        "year": year,
        "strike": strike,
        "option_type": option_type
    }


def clean_strike(strike: float | str, underlying_price: float | None = None) -> float:
    """
    Cleans and standardizes option strike prices.
    Handles cents/multiplier scaling (e.g. 7500 -> 75.0).
    """
    if isinstance(strike, str):
        # Extract digits and periods
        strike = "".join(c for c in strike if c.isdigit() or c == '.')
        strike = float(strike)
    else:
        strike = float(strike)
        
    if underlying_price is not None:
        ratio = strike / underlying_price
        if 50.0 < ratio < 200.0:
            strike = strike / 100.0
        elif 500.0 < ratio < 2000.0:
            strike = strike / 1000.0
    else:
        if strike > 500.0:
            strike = strike / 100.0
            
    return strike


def implied_vol_black76(
    price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "C"
) -> float:
    """
    Helper to calculate Black-76 implied volatility.
    """
    if T <= 0:
        return np.nan
        
    eff_price = price / np.exp(-r * T)
    is_put = (option_type == "P")
    
    intrinsic = max(K - F, 0.0) if is_put else max(F - K, 0.0)
    if eff_price <= intrinsic + 1e-12:
        return np.nan
    if eff_price >= (K if is_put else F):
        return np.nan
        
    sigma = 0.3
    for _ in range(50):
        d1 = (np.log(F / K) + 0.5 * (sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if is_put:
            p = K * norm.cdf(-d2) - F * norm.cdf(-d1)
        else:
            p = F * norm.cdf(d1) - K * norm.cdf(d2)
            
        vega = F * np.sqrt(T) * norm.pdf(d1)
        if abs(vega) < 1e-15:
            break
            
        diff = p - eff_price
        if abs(diff) < 1e-10:
            return sigma
        sigma -= diff / vega
        sigma = np.clip(sigma, 1e-6, 5.0)
        
    return sigma if 1e-5 < sigma < 4.9 else np.nan


class CMECommodityDataAdapter:
    """
    Adapter class for CME commodity options and futures contracts.
    """
    def __init__(self, ref_year: int | None = None):
        self.ref_year = ref_year or datetime.date.today().year
        
    def match_options_to_futures(
        self,
        options_df: pd.DataFrame,
        futures_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Cleans option strike prices and matches options to their correct underlying futures contracts.
        Calculates time-to-maturity (T) correctly based on contract specs.
        """
        # Make working copies
        opts = options_df.copy()
        futs = futures_df.copy()
        
        # Ensure dates are datetime.date
        opts["val_dt"] = pd.to_datetime(opts["valuation_date"]).dt.date
        futs["val_dt"] = pd.to_datetime(futs["valuation_date"]).dt.date
        
        # Parse futures codes
        fut_parsed = []
        for code in futs["contract_code"]:
            try:
                fut_parsed.append(parse_futures_code(code, self.ref_year))
            except Exception:
                fut_parsed.append({"underlying": None, "month": None, "year": None})
                
        futs["month"] = [p["month"] for p in fut_parsed]
        futs["year"] = [p["year"] for p in fut_parsed]
        futs = futs.dropna(subset=["month", "year"])
        
        # Parse options codes
        opt_parsed = []
        for code in opts["option_code"]:
            try:
                opt_parsed.append(parse_options_code(code, self.ref_year))
            except Exception:
                opt_parsed.append({"underlying": None, "month": None, "year": None, "strike": None, "option_type": None})
                
        opts["month"] = [p["month"] for p in opt_parsed]
        opts["year"] = [p["year"] for p in opt_parsed]
        opts["raw_strike_parsed"] = [p["strike"] for p in opt_parsed]
        opts["option_type_parsed"] = [p["option_type"] for p in opt_parsed]
        
        # Fallbacks for missing columns
        if "strike" not in opts.columns:
            opts["strike"] = opts["raw_strike_parsed"]
        if "option_type" not in opts.columns:
            opts["option_type"] = opts["option_type_parsed"]
            
        opts = opts.dropna(subset=["month", "year", "strike", "option_type"])
        
        # Merge options and futures
        merged = pd.merge(
            opts,
            futs,
            on=["val_dt", "month", "year"],
            suffixes=("_opt", "_fut")
        )
        
        # Clean strikes
        clean_strikes = []
        for _, row in merged.iterrows():
            clean_strikes.append(clean_strike(row["strike"], row["price_fut"]))
        merged["clean_strike"] = clean_strikes
        
        # Calculate expirations and time-to-maturities
        t_opt_list = []
        t_fut_list = []
        opt_expiry_list = []
        fut_expiry_list = []
        
        for _, row in merged.iterrows():
            m = int(row["month"])
            y = int(row["year"])
            
            opt_exp = wti_options_expiry(y, m)
            fut_exp = wti_futures_expiry(y, m)
            
            val_dt = row["val_dt"]
            
            t_opt = (opt_exp - val_dt).days / 365.0
            t_fut = (fut_exp - val_dt).days / 365.0
            
            opt_expiry_list.append(opt_exp)
            fut_expiry_list.append(fut_exp)
            t_opt_list.append(t_opt)
            t_fut_list.append(t_fut)
            
        merged["option_expiry_date"] = opt_expiry_list
        merged["futures_expiry_date"] = fut_expiry_list
        merged["T_opt"] = t_opt_list
        merged["T_fut"] = t_fut_list
        
        # Select and rename final columns
        result = merged[[
            "valuation_date_opt",
            "option_code",
            "contract_code",
            "clean_strike",
            "option_type",
            "price_opt",
            "price_fut",
            "T_opt",
            "T_fut",
            "option_expiry_date",
            "futures_expiry_date"
        ]].rename(columns={
            "valuation_date_opt": "valuation_date",
            "contract_code": "underlying_code",
            "clean_strike": "strike",
            "price_opt": "option_price",
            "price_fut": "futures_price"
        })
        
        # Drop rows with non-positive times to maturity
        result = result[result["T_opt"] > 0]
        
        return result.reset_index(drop=True)


def generate_synthetic_options_data(
    valuation_dates: list[datetime.date] | np.ndarray,
    months_ahead: list[int] | np.ndarray,
    strike_pcts: list[float] | np.ndarray,
    ss_params: dict[str, float],
    init_chi: float = 0.0,
    init_xi: float = np.log(75.0),
    r: float = 0.04,
    noise_std: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic options and futures data under the Schwartz-Smith model.
    """
    # Parameters
    kappa = ss_params["kappa"]
    sigma_chi = ss_params["sigma_chi"]
    rho = ss_params["rho"]
    sigma_xi = ss_params["sigma_xi"]
    mu = ss_params["mu"]
    lambda_chi = ss_params["lambda_chi"]
    mu_star = ss_params["mu_star"]
    
    # State variable simulation
    num_dates = len(valuation_dates)
    states_chi = np.zeros(num_dates)
    states_xi = np.zeros(num_dates)
    
    states_chi[0] = init_chi
    states_xi[0] = init_xi
    
    # Simulating under P
    for t in range(1, num_dates):
        dt = (valuation_dates[t] - valuation_dates[t-1]).days / 365.0
        
        # Bivariate transition covariance Q
        exp_k = np.exp(-kappa * dt)
        var_chi = (sigma_chi**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * dt)) if kappa >= 1e-5 else sigma_chi**2 * dt
        var_xi = sigma_xi**2 * dt
        cov_chi_xi = (rho * sigma_chi * sigma_xi / kappa) * (1.0 - exp_k) if kappa >= 1e-5 else rho * sigma_chi * sigma_xi * dt
        
        cov_matrix = np.array([
            [var_chi, cov_chi_xi],
            [cov_chi_xi, var_xi]
        ])
        
        noise = np.random.multivariate_normal(np.zeros(2), cov_matrix)
        
        states_chi[t] = exp_k * states_chi[t-1] + noise[0]
        states_xi[t] = states_xi[t-1] + mu * dt + noise[1]
        
    futures_records = []
    options_records = []
    
    for t_idx, val_dt in enumerate(valuation_dates):
        chi_t = states_chi[t_idx]
        xi_t = states_xi[t_idx]
        
        for m_ahead in months_ahead:
            # Determine contract delivery month and year
            # val_dt.month + m_ahead
            tgt_month = val_dt.month + m_ahead
            tgt_year = val_dt.year
            while tgt_month > 12:
                tgt_month -= 12
                tgt_year += 1
                
            month_code = MONTH_TO_CODE[tgt_month]
            yr_str = str(tgt_year)[2:]
            
            fut_code = f"CL{month_code}{yr_str}"
            fut_exp = wti_futures_expiry(tgt_year, tgt_month)
            T_fut = (fut_exp - val_dt).days / 365.0
            
            # Futures price
            fut_price = float(np.exp(np.exp(-kappa * T_fut) * chi_t + xi_t + A_factor(T_fut, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi)))
            
            # Add a bit of noise if specified
            if noise_std > 0:
                fut_price = max(fut_price + np.random.normal(0, noise_std), 1.0)
                
            futures_records.append({
                "valuation_date": val_dt,
                "contract_code": fut_code,
                "tenor": f"{m_ahead}M",
                "price": fut_price
            })
            
            opt_exp = wti_options_expiry(tgt_year, tgt_month)
            T_opt = (opt_exp - val_dt).days / 365.0
            
            # Strike grid around the futures price
            for pct in strike_pcts:
                strike = round(fut_price * pct, 1)
                # Format strike to CME style, e.g. 75.0 -> 7500, or just standard
                strike_cme = int(strike * 100)
                
                for otype in ["C", "P"]:
                    opt_code = f"LO {month_code}{yr_str} {strike_cme} {otype}"
                    
                    price = schwartz_smith_price_black76(
                        (val_dt - val_dt).days / 365.0, T_opt, T_fut, strike, r,
                        chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi,
                        option_type=otype
                    )
                    
                    # Add noise
                    if noise_std > 0:
                        price = max(price + np.random.normal(0, noise_std * 0.1), 0.01)
                        
                    iv = implied_vol_black76(price, fut_price, strike, T_opt, r, otype)
                    
                    options_records.append({
                        "valuation_date": val_dt,
                        "option_code": opt_code,
                        "option_type": otype,
                        "strike": strike_cme, # Generate raw cents style strike
                        "tenor": f"{m_ahead}M",
                        "price": price,
                        "implied_vol": iv
                    })
                    
    futures_df = pd.DataFrame(futures_records)
    options_df = pd.DataFrame(options_records)
    return options_df, futures_df
