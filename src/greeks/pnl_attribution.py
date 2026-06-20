import numpy as np
from typing import List, Dict, Any, Optional

def pnl_attribution(portfolio: List[Dict[str, Any]], dS: float, d_iv_surface: np.ndarray, S: Optional[float] = None) -> dict:
    """
    Portfolio-level P&L attribution using Taylor expansion:
    explained_pnl = Delta * dS + 0.5 * Gamma * dS^2 + Vega * dIV + Vanna * dS * dIV + Volga * dIV^2
    """
    # Import interpolation helpers from portfolio_greeks
    from greeks.portfolio_greeks import _bilinear_interp, MATURITIES, STRIKES
    
    total_explained = 0.0
    total_actual = 0.0
    
    delta_pnl_total = 0.0
    gamma_pnl_total = 0.0
    vega_pnl_total = 0.0
    vanna_pnl_total = 0.0
    volga_pnl_total = 0.0
    
    is_d_iv_scalar = np.isscalar(d_iv_surface) or (isinstance(d_iv_surface, np.ndarray) and d_iv_surface.ndim == 0)
    is_d_iv_1d = isinstance(d_iv_surface, (list, np.ndarray)) and not is_d_iv_scalar and len(d_iv_surface) == len(portfolio)
    
    for i, pos in enumerate(portfolio):
        qty = float(pos.get("quantity", 1.0))
        notional = float(pos.get("notional", 1.0))
        weight = qty * notional
        
        # Read Greeks (handling both raw and total/pre-scaled)
        delta = float(pos.get("total_delta", pos.get("delta", 0.0)))
        gamma = float(pos.get("total_gamma", pos.get("gamma", 0.0)))
        vega = float(pos.get("total_vega", pos.get("vega", 0.0)))
        vanna = float(pos.get("total_vanna", pos.get("vanna", 0.0)))
        volga = float(pos.get("total_volga", pos.get("volga", 0.0)))
        
        # Determine if the Greeks are pre-scaled or raw
        # If "total_delta" is present in the position, we assume the Greeks are pre-scaled.
        if "total_delta" in pos:
            w_delta = delta
            w_gamma = gamma
            w_vega = vega
            w_vanna = vanna
            w_volga = volga
        else:
            w_delta = delta * weight
            w_gamma = gamma * weight
            w_vega = vega * weight
            w_vanna = vanna * weight
            w_volga = volga * weight
            
        # Get actual P&L for this position
        if "actual_pnl" in pos:
            pos_actual = float(pos["actual_pnl"])
        elif "price_before" in pos and "price_after" in pos:
            pos_actual = (float(pos["price_after"]) - float(pos["price_before"])) * weight
        else:
            pos_actual = 0.0
            
        total_actual += pos_actual
        
        # Determine dIV for this position
        if is_d_iv_scalar:
            dIV = float(d_iv_surface)
        elif is_d_iv_1d:
            dIV = float(d_iv_surface[i])
        else:
            # 2D surface interpolation
            T_pos = float(pos.get("T", 0.5))
            K_pos = float(pos.get("K", 100.0))
            # Resolve spot price
            S_pos = S if S is not None else float(pos.get("S", pos.get("S_before", pos.get("spot", 5000.0))))
            k_pos = np.log(K_pos / S_pos)
            
            # Interpolate
            dIV = _bilinear_interp(MATURITIES, STRIKES, d_iv_surface, T_pos, k_pos)
            
        # Taylor expansion terms for this position
        delta_pnl = w_delta * dS
        gamma_pnl = 0.5 * w_gamma * (dS ** 2)
        vega_pnl = w_vega * dIV
        vanna_pnl = w_vanna * dS * dIV
        volga_pnl = w_volga * (dIV ** 2)
        
        # Accumulate
        delta_pnl_total += delta_pnl
        gamma_pnl_total += gamma_pnl
        vega_pnl_total += vega_pnl
        vanna_pnl_total += vanna_pnl
        volga_pnl_total += volga_pnl
        
    explained_pnl = delta_pnl_total + gamma_pnl_total + vega_pnl_total + vanna_pnl_total + volga_pnl_total
    
    # If actual P&L is explicitly passed or total_actual was computed:
    has_actual_keys = any("actual_pnl" in pos or ("price_before" in pos and "price_after" in pos) for pos in portfolio)
    if not has_actual_keys:
        # Default to explained_pnl so residual is 0
        actual_pnl = explained_pnl
    else:
        actual_pnl = total_actual
        
    residual = actual_pnl - explained_pnl
    
    return {
        "explained_pnl": explained_pnl,
        "actual_pnl": actual_pnl,
        "residual": residual,
        "breakdown": {
            "delta": delta_pnl_total,
            "gamma": gamma_pnl_total,
            "vega": vega_pnl_total,
            "vanna": vanna_pnl_total,
            "volga": volga_pnl_total,
            "delta_pnl": delta_pnl_total,
            "gamma_pnl": gamma_pnl_total,
            "vega_pnl": vega_pnl_total,
            "vanna_pnl": vanna_pnl_total,
            "volga_pnl": volga_pnl_total,
        }
    }
