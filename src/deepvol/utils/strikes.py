"""
strikes.py — Centralized strike type resolution utility.

Replaces the fragile heuristic ``np.any(strikes < 0) or np.max(np.abs(strikes)) < 5.0``
that was duplicated across fallbacks.py, guardian.py, and arbitrage.py.
"""

import numpy as np
from typing import Literal, Optional


def resolve_strikes(
    strikes: np.ndarray,
    S: float,
    strike_type: Optional[Literal["log_moneyness", "absolute"]] = None,
) -> np.ndarray:
    """
    Convert strike array to absolute strike prices.

    Parameters
    ----------
    strikes : np.ndarray
        Strike values — either log-moneyness k = ln(K/S) or absolute prices.
    S : float
        Current spot price of the underlying.
    strike_type : {"log_moneyness", "absolute"}, optional
        If provided, uses the explicit type. If None, falls back to the legacy
        heuristic: treats strikes as log-moneyness if any value is negative or
        all absolute values are below 5.0.

    Returns
    -------
    np.ndarray
        Absolute strike prices.
    """
    strikes = np.asarray(strikes, dtype=np.float64)

    if strike_type == "absolute":
        return strikes
    elif strike_type == "log_moneyness":
        return S * np.exp(strikes)
    else:
        # Legacy heuristic fallback — kept for backward compatibility
        if np.any(strikes < 0) or np.max(np.abs(strikes)) < 5.0:
            return S * np.exp(strikes)
        else:
            return strikes
