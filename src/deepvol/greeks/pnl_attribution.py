import numpy as np
import torch
from typing import List, Dict, Any, Optional

def pnl_attribution(portfolio: List[Dict[str, Any]], dS: float, d_iv_surface: np.ndarray, S: Optional[float] = None) -> dict:
    """
    Portfolio-level P&L attribution using Taylor expansion on GPU:
    explained_pnl = Delta * dS + 0.5 * Gamma * dS^2 + Vega * dIV + Vanna * dS * dIV + Volga * dIV^2
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Pre-extract values into lists on CPU (very fast)
    qty_list = []
    notional_list = []
    delta_list = []
    gamma_list = []
    vega_list = []
    vanna_list = []
    volga_list = []
    total_delta_flag_list = []
    actual_pnl_list = []
    has_actual_list = []
    T_list = []
    K_list = []
    S_list = []

    for pos in portfolio:
        qty_list.append(float(pos.get("quantity", 1.0)))
        notional_list.append(float(pos.get("notional", 1.0)))
        
        delta_list.append(float(pos.get("total_delta", pos.get("delta", 0.0))))
        gamma_list.append(float(pos.get("total_gamma", pos.get("gamma", 0.0))))
        vega_list.append(float(pos.get("total_vega", pos.get("vega", 0.0))))
        vanna_list.append(float(pos.get("total_vanna", pos.get("vanna", 0.0))))
        volga_list.append(float(pos.get("total_volga", pos.get("volga", 0.0))))
        
        total_delta_flag_list.append(1.0 if "total_delta" in pos else 0.0)
        
        # Actual P&L extraction
        weight = float(pos.get("quantity", 1.0)) * float(pos.get("notional", 1.0))
        if "actual_pnl" in pos:
            actual_pnl_list.append(float(pos["actual_pnl"]))
            has_actual_list.append(1.0)
        elif "price_before" in pos and "price_after" in pos:
            actual_pnl_list.append((float(pos["price_after"]) - float(pos["price_before"])) * weight)
            has_actual_list.append(1.0)
        else:
            actual_pnl_list.append(0.0)
            has_actual_list.append(0.0)
            
        T_list.append(float(pos.get("T", 0.5)))
        K_list.append(float(pos.get("K", 100.0)))
        
        S_pos = S if S is not None else float(pos.get("S", pos.get("S_before", pos.get("spot", 5000.0))))
        S_list.append(S_pos)

    # Convert to GPU tensors
    qty_t = torch.tensor(qty_list, dtype=torch.float32, device=device)
    notional_t = torch.tensor(notional_list, dtype=torch.float32, device=device)
    delta_t = torch.tensor(delta_list, dtype=torch.float32, device=device)
    gamma_t = torch.tensor(gamma_list, dtype=torch.float32, device=device)
    vega_t = torch.tensor(vega_list, dtype=torch.float32, device=device)
    vanna_t = torch.tensor(vanna_list, dtype=torch.float32, device=device)
    volga_t = torch.tensor(volga_list, dtype=torch.float32, device=device)
    total_delta_flag_t = torch.tensor(total_delta_flag_list, dtype=torch.float32, device=device)
    actual_pnl_t = torch.tensor(actual_pnl_list, dtype=torch.float32, device=device)
    has_actual_t = torch.tensor(has_actual_list, dtype=torch.float32, device=device)
    T_t = torch.tensor(T_list, dtype=torch.float32, device=device)
    K_t = torch.tensor(K_list, dtype=torch.float32, device=device)
    S_t = torch.tensor(S_list, dtype=torch.float32, device=device)
    
    dS_t = torch.tensor(dS, dtype=torch.float32, device=device)

    # Apply scaling weights based on "total_delta" presence
    weight_t = qty_t * notional_t
    w_delta_t = torch.where(total_delta_flag_t == 1.0, delta_t, delta_t * weight_t)
    w_gamma_t = torch.where(total_delta_flag_t == 1.0, gamma_t, gamma_t * weight_t)
    w_vega_t = torch.where(total_delta_flag_t == 1.0, vega_t, vega_t * weight_t)
    w_vanna_t = torch.where(total_delta_flag_t == 1.0, vanna_t, vanna_t * weight_t)
    w_volga_t = torch.where(total_delta_flag_t == 1.0, volga_t, volga_t * weight_t)
    
    total_actual_t = torch.sum(actual_pnl_t)

    # Determine dIV for each position
    is_d_iv_scalar = np.isscalar(d_iv_surface) or (isinstance(d_iv_surface, np.ndarray) and d_iv_surface.ndim == 0)
    is_d_iv_1d = isinstance(d_iv_surface, (list, np.ndarray)) and not is_d_iv_scalar and len(d_iv_surface) == len(portfolio)

    if is_d_iv_scalar:
        dIV_t = torch.tensor(float(d_iv_surface), dtype=torch.float32, device=device).expand_as(T_t)
    elif is_d_iv_1d:
        dIV_t = torch.tensor(d_iv_surface, dtype=torch.float32, device=device)
    else:
        # 2D surface interpolation vectorially on GPU
        d_iv_surf_t = torch.tensor(d_iv_surface, dtype=torch.float32, device=device)
        
        # Import interpolation grid and function
        from deepvol.greeks.portfolio_greeks import interpolate_bilinear, MATURITIES, STRIKES
        
        T_grid_t = torch.tensor(MATURITIES, dtype=torch.float32, device=device)
        K_grid_t = torch.tensor(STRIKES, dtype=torch.float32, device=device)
        
        k_t = torch.log(K_t / S_t)
        dIV_t = interpolate_bilinear(T_grid_t, K_grid_t, d_iv_surf_t, T_t, k_t)

    # Taylor expansion terms for all positions vectorially
    delta_pnl_t = w_delta_t * dS_t
    gamma_pnl_t = 0.5 * w_gamma_t * (dS_t ** 2)
    vega_pnl_t = w_vega_t * dIV_t
    vanna_pnl_t = w_vanna_t * dS_t * dIV_t
    volga_pnl_t = w_volga_t * (dIV_t ** 2)

    delta_pnl_total = torch.sum(delta_pnl_t)
    gamma_pnl_total = torch.sum(gamma_pnl_t)
    vega_pnl_total = torch.sum(vega_pnl_t)
    vanna_pnl_total = torch.sum(vanna_pnl_t)
    volga_pnl_total = torch.sum(volga_pnl_t)
    
    explained_pnl = delta_pnl_total + gamma_pnl_total + vega_pnl_total + vanna_pnl_total + volga_pnl_total

    has_actual_keys = torch.any(has_actual_t == 1.0).item()
    if not has_actual_keys:
        actual_pnl = explained_pnl
    else:
        actual_pnl = total_actual_t
        
    residual = actual_pnl - explained_pnl

    return {
        "explained_pnl": float(explained_pnl.item()),
        "actual_pnl": float(actual_pnl.item() if isinstance(actual_pnl, torch.Tensor) else actual_pnl),
        "residual": float(residual.item() if isinstance(residual, torch.Tensor) else residual),
        # Use the _pnl-suffixed keys consistently.
        # Duplicate alias keys ("delta", "gamma", etc.) are removed.
        "breakdown": {
            "delta_pnl": float(delta_pnl_total.item()),
            "gamma_pnl": float(gamma_pnl_total.item()),
            "vega_pnl":  float(vega_pnl_total.item()),
            "vanna_pnl": float(vanna_pnl_total.item()),
            "volga_pnl": float(volga_pnl_total.item()),
        }
    }
