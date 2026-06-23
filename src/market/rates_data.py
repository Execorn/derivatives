"""
rates_data.py — SOFR swap rates and swaption volatility cube data.
Contains functions to load realistic forward swap rates and synthetic
swaption volatility grids.
"""

import numpy as np

def load_sofr_swap_rates():
    """
    Load a structured realistic curve of SOFR forward swap rates.
    
    Returns
    -------
    swap_rates : dict
        Dictionary of swap tenors (float, in years) and their corresponding swap rates (float).
    """
    # Realistic SOFR swap curve (yield curve)
    return {
        1.0: 0.0350,
        2.0: 0.0330,
        3.0: 0.0315,
        4.0: 0.0305,
        5.0: 0.0300,
        7.0: 0.0290,
        10.0: 0.0280,
        15.0: 0.0270,
        20.0: 0.0260,
        30.0: 0.0250
    }

def get_synthetic_forward_rates(expiries, tenors):
    """
    Generate a realistic 2D grid of forward swap rates F(T_expiry, T_tenor).
    
    Parameters
    ----------
    expiries : ndarray
        1D array of option expiries.
    tenors : ndarray
        1D array of swap tenors.
        
    Returns
    -------
    forward_rates : ndarray
        2D array of shape (len(expiries), len(tenors)) containing forward swap rates.
    """
    expiries = np.asarray(expiries, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    
    # We use a simple parametric model to generate realistic forward swap rates:
    # F(T_exp, T_tenor) = 0.04 - 0.0005 * T_exp - 0.0002 * T_tenor
    # This reflects a slightly inverted/declining curve typical of recent markets.
    T_exp_mesh, T_tenor_mesh = np.meshgrid(expiries, tenors, indexing='ij')
    forward_rates = 0.04 - 0.0005 * T_exp_mesh - 0.0002 * T_tenor_mesh
    return np.maximum(forward_rates, 0.001)  # Ensure positive rates

def load_swaption_vol_cube():
    """
    Generate a structured synthetic swaption volatility cube grid over option expiries,
    swap tenors, and relative strikes (moneyness).
    
    Returns
    -------
    expiries : ndarray
        1D array of option expiries (1Y, 2Y, 5Y, 10Y).
    tenors : ndarray
        1D array of swap tenors (1Y, 2Y, 5Y, 10Y, 30Y).
    relative_strikes : ndarray
        1D array of relative strikes/moneyness in bps (-200bps, -100bps, -50bps, ATM, +50bps, +100bps, +200bps).
    market_vols : ndarray
        3D array of shape (len(expiries), len(tenors), len(relative_strikes))
        containing realistic Bachelier (normal) implied volatilities in decimal format (e.g. 0.0080 for 80 bps).
    """
    expiries = np.array([1.0, 2.0, 5.0, 10.0])
    tenors = np.array([1.0, 2.0, 5.0, 10.0, 30.0])
    relative_strikes = np.array([-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0])  # in bps
    
    num_exp = len(expiries)
    num_ten = len(tenors)
    num_str = len(relative_strikes)
    
    market_vols = np.zeros((num_exp, num_ten, num_str))
    
    for i, T_exp in enumerate(expiries):
        for j, T_tenor in enumerate(tenors):
            # Base ATM vol: starting at 80 bps (0.0080), decreasing with maturity/tenor
            atm_vol_bps = 80.0 - 1.0 * T_exp - 0.3 * T_tenor
            atm_vol_bps = max(atm_vol_bps, 40.0)  # floor at 40 bps
            
            for k, rel_strike in enumerate(relative_strikes):
                # Vol smile: downward skew (negative linear term) and U-shaped smile (quadratic term)
                skew_term = -0.05 * rel_strike
                smile_term = 0.0002 * (rel_strike ** 2)
                vol_bps = atm_vol_bps + skew_term + smile_term
                
                # Convert from basis points to decimal absolute volatility (e.g. 80 bps -> 0.0080)
                market_vols[i, j, k] = vol_bps * 1e-4
                
    return expiries, tenors, relative_strikes, market_vols


class SOFRSwaptionLoader:
    def load_swaption_cube(self, date=None):
        if date == "":
            raise ValueError("Date cannot be empty")
        expiries = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
        tenors = np.array([1.0, 2.0, 5.0, 10.0, 30.0])
        strikes_bps = np.array([-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0])
        
        forward_rates = get_synthetic_forward_rates(expiries, tenors)
        
        num_exp = len(expiries)
        num_ten = len(tenors)
        num_str = len(strikes_bps)
        
        vol_cube = np.zeros((num_exp, num_ten, num_str))
        for i, T_exp in enumerate(expiries):
            for j, T_tenor in enumerate(tenors):
                atm_vol_bps = 80.0 - 1.0 * T_exp - 0.3 * T_tenor
                atm_vol_bps = max(atm_vol_bps, 40.0)
                for k, rel_strike in enumerate(strikes_bps):
                    skew_term = -0.05 * rel_strike
                    smile_term = 0.0002 * (rel_strike ** 2)
                    vol_bps = atm_vol_bps + skew_term + smile_term
                    vol_cube[i, j, k] = vol_bps * 1e-4
                    
        return {
            "expiries": expiries,
            "tenors": tenors,
            "strikes_bps": strikes_bps,
            "forward_rates": forward_rates,
            "vol_cube": vol_cube
        }
