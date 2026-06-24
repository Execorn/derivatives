"""
local_vol.py — Local Volatility (Dupire) and SVI representation using PyTorch GPU acceleration.

Contains:
- `svi_slice`: computes total variance w(k) for a given SVI parameterization.
- `svi_to_lv_surface`: applies Dupire formula to SVI representations using finite differences on fine grids.
- `check_arbitrage_free`: checks calendar spread and butterfly arbitrage (g(k) >= 0).
"""

from typing import Union
import numpy as np
import torch

def svi_slice(
    k: Union[float, np.ndarray, torch.Tensor],
    a: Union[float, np.ndarray, torch.Tensor],
    b: Union[float, np.ndarray, torch.Tensor],
    rho: Union[float, np.ndarray, torch.Tensor],
    m: Union[float, np.ndarray, torch.Tensor],
    sigma: Union[float, np.ndarray, torch.Tensor]
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes total variance w(k) for a single SVI slice.
    Supports both NumPy arrays and PyTorch tensors.
    
    Formula:
      w(k) = a + b * (\\rho * (k - m) + \\sqrt{(k - m)^2 + \\sigma^2})
      
    Academic Reference:
      Gatheral, J. (2004). A arbitrage-free SVI volatility surface. Presentation at 
      Global Derivatives.
      
    Parameters
    ----------
    k : float, np.ndarray, or torch.Tensor
        Log-moneyness k = log(Strike / Spot).
    a : float, np.ndarray, or torch.Tensor
        SVI parameter controlling general level of variance.
    b : float, np.ndarray, or torch.Tensor
        SVI parameter controlling slope/angle of wings.
    rho : float, np.ndarray, or torch.Tensor
        SVI parameter controlling rotation/skewness.
    m : float, np.ndarray, or torch.Tensor
        SVI parameter controlling horizontal translation.
    sigma : float, np.ndarray, or torch.Tensor
        SVI parameter controlling curvature at ATM.
        
    Returns
    -------
    w : float, np.ndarray, or torch.Tensor
        Total variance at k.
    """
    # Guard to prevent division by zero in SVI square root and SVI parameters
    if isinstance(k, torch.Tensor):
        sigma_safe = torch.clamp(torch.as_tensor(sigma), min=1e-8)
        return a + b * (rho * (k - m) + torch.sqrt((k - m) ** 2 + sigma_safe ** 2))
    else:
        sigma_safe = np.maximum(np.asarray(sigma), 1e-8)
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma_safe ** 2))


def check_arbitrage_free(
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    svi_params: np.ndarray
) -> bool:
    """
    Checks if the SVI surface defined by svi_params is free of calendar spread and butterfly arbitrage.
    
    Academic Reference:
      Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. 
      Quantitative Finance, 14(1), 59-71.
      
    Parameters
    ----------
    T_grid : np.ndarray
        Maturities grid, shape (nT,)
    K_grid : np.ndarray
        Strikes grid, shape (nK,)
    svi_params : np.ndarray
        SVI parameters, shape (nT, 5) -> [a, b, rho, m, sigma] for each slice.
        
    Returns
    -------
    is_arbitrage_free : bool
        True if the surface has no static arbitrage.
    """
    assert np.all(np.diff(T_grid) > 0), "T_grid must be strictly increasing and sorted"
    
    # Use a dense strike grid to ensure we don't miss arbitrage between grid points
    k_min = min(K_grid.min(), -2.0)
    k_max = max(K_grid.max(), 2.0)
    K_dense = np.linspace(k_min, k_max, 400)
    nT = len(T_grid)
    
    # 1. Compute total variance for all slices on K_dense
    w_all = []
    for i in range(nT):
        a, b, rho, m, sigma = svi_params[i]
        
        # Basic parameter checks
        if b < 0 or sigma <= 0 or np.abs(rho) > 1.0:
            return False
            
        w_i = svi_slice(K_dense, a, b, rho, m, sigma)
        w_all.append(w_i)
        
    w_all = np.array(w_all) # shape (nT, len(K_dense))
    
    # 2. Check positivity of total variance
    if np.any(w_all < 0):
        return False
        
    # 3. Calendar spread check: w(k, T_i) <= w(k, T_{i+1})
    for i in range(nT - 1):
        if np.any(w_all[i+1] < w_all[i] - 1e-10):
            return False
            
    # 4. Butterfly arbitrage check: g(k) >= 0 for all slices
    for i in range(nT):
        a, b, rho, m, sigma = svi_params[i]
        u = K_dense - m
        sigma = max(sigma, 1e-8)
        sqrt_term = np.sqrt(u ** 2 + sigma ** 2)
        
        # Analytical derivatives
        w = a + b * (rho * u + sqrt_term)
        w_prime = b * (rho + u / sqrt_term)
        w_prime2 = b * (sigma ** 2) / (sqrt_term ** 3)
        
        w_safe = np.maximum(w, 1e-8)
        term1 = (1.0 - (K_dense * w_prime) / (2.0 * w_safe)) ** 2
        term2 = (w_prime ** 2 / 4.0) * (1.0 / w_safe + 0.25)
        term3 = 0.5 * w_prime2
        g_k = term1 - term2 + term3
        
        if np.any(g_k < -1e-10):
            return False
            
    return True


def check_arbitrage_free_batch(
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    svi_params: np.ndarray
) -> np.ndarray:
    """
    Checks calendar spread and butterfly arbitrage in batch.
    
    Parameters
    ----------
    T_grid : np.ndarray
        Maturities grid, shape (nT,)
    K_grid : np.ndarray
        Strikes grid, shape (nK,)
    svi_params : np.ndarray
        SVI parameters, shape (B, nT, 5) -> [a, b, rho, m, sigma] for each slice.
        
    Returns
    -------
    is_arbitrage_free : np.ndarray
        Boolean array of shape (B,) indicating which batch items are arbitrage-free.
    """
    B = svi_params.shape[0]
    nT = len(T_grid)
    
    k_min = min(K_grid.min(), -2.0)
    k_max = max(K_grid.max(), 2.0)
    K_dense = np.linspace(k_min, k_max, 400)
    
    # 0. Basic parameter checks (b >= 0, sigma > 0, |rho| <= 1.0)
    b = svi_params[..., 1]      # (B, nT)
    sigma = svi_params[..., 4]  # (B, nT)
    rho = svi_params[..., 2]    # (B, nT)
    
    param_ok = np.all((b >= 0) & (sigma > 0) & (np.abs(rho) <= 1.0), axis=1) # (B,)
    
    valid_mask = param_ok.copy()
    if not np.any(valid_mask):
        return valid_mask
        
    active_indices = np.where(valid_mask)[0]
    sub_svi = svi_params[active_indices] # (B_active, nT, 5)
    
    # 1. Compute total variance for all slices on K_dense
    a = sub_svi[..., 0, np.newaxis] # (B_active, nT, 1)
    b = sub_svi[..., 1, np.newaxis] # (B_active, nT, 1)
    rho = sub_svi[..., 2, np.newaxis] # (B_active, nT, 1)
    m = sub_svi[..., 3, np.newaxis] # (B_active, nT, 1)
    sigma = sub_svi[..., 4, np.newaxis] # (B_active, nT, 1)
    
    sigma = np.maximum(sigma, 1e-8)
    
    k_exp = K_dense[np.newaxis, np.newaxis, :] # (1, 1, 400)
    
    u = k_exp - m # (B_active, nT, 400)
    sqrt_term = np.sqrt(u ** 2 + sigma ** 2) # (B_active, nT, 400)
    w_all = a + b * (rho * u + sqrt_term) # (B_active, nT, 400)
    
    # 2. Check positivity of total variance
    positivity_ok = ~np.any(w_all < 0, axis=(1, 2)) # (B_active,)
    
    # 3. Calendar spread check: w(k, T_i) <= w(k, T_{i+1})
    calendar_ok = ~np.any(w_all[:, 1:, :] < w_all[:, :-1, :] - 1e-10, axis=(1, 2)) # (B_active,)
    
    # 4. Butterfly arbitrage check: g(k) >= 0 for all slices
    w_prime = b * (rho + u / sqrt_term) # (B_active, nT, 400)
    w_prime2 = b * (sigma ** 2) / (sqrt_term ** 3) # (B_active, nT, 400)
    
    w_safe = np.maximum(w_all, 1e-8)
    term1 = (1.0 - (k_exp * w_prime) / (2.0 * w_safe)) ** 2
    term2 = (w_prime ** 2 / 4.0) * (1.0 / w_safe + 0.25)
    term3 = 0.5 * w_prime2
    g_k = term1 - term2 + term3 # (B_active, nT, 400)
    
    butterfly_ok = ~np.any(g_k < -1e-10, axis=(1, 2)) # (B_active,)
    
    active_valid = positivity_ok & calendar_ok & butterfly_ok
    valid_mask[active_indices] = active_valid
    
    return valid_mask


def svi_to_lv_surface(
    T_grid: Union[np.ndarray, torch.Tensor],
    K_grid: Union[np.ndarray, torch.Tensor],
    svi_params: Union[np.ndarray, torch.Tensor],
    fine_T_points: int = 100,
    fine_K_points: int = 200
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes local volatility surface by applying Dupire formula to SVI representation
    using PyTorch (GPU/CPU) tensor operations.
    Supports both numpy arrays and PyTorch tensors, single and batched inputs.
    
    Formula:
      \\sigma^2_{loc}(T, K) = \\frac{\\partial W / \\partial T}{g(k)}
      
    Academic Reference:
      Dupire, B. (1994). Pricing with a smile. Risk, 7(1), 18-20.
      
    Parameters
    ----------
    T_grid : np.ndarray or torch.Tensor
        Target maturities grid, shape (nT,)
    K_grid : np.ndarray or torch.Tensor
        Target strikes grid, shape (nK,)
    svi_params : np.ndarray or torch.Tensor
        SVI parameters, shape (nT, 5) or (B, nT, 5)
    fine_T_points : int
        Number of points in the fine maturity grid.
    fine_K_points : int
        Number of points in the fine strike grid.
        
    Returns
    -------
    lv_surface : np.ndarray or torch.Tensor
        Local volatility surface of shape (nT, nK) or (B, nT, nK).
    """
    is_numpy = isinstance(svi_params, np.ndarray)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Convert to PyTorch tensors on GPU/CPU (with float64 precision to prevent FD noise)
    if is_numpy:
        T_t = torch.tensor(T_grid, dtype=torch.float64, device=device)
        K_t = torch.tensor(K_grid, dtype=torch.float64, device=device)
        svi_params_t = torch.tensor(svi_params, dtype=torch.float64, device=device)
    else:
        T_t = T_grid.to(device=device, dtype=torch.float64)
        K_t = K_grid.to(device=device, dtype=torch.float64)
        svi_params_t = svi_params.to(device=device, dtype=torch.float64)
        
    # Check if input is batched or single
    is_batched = (svi_params_t.ndim == 3)
    if not is_batched:
        svi_params_t = svi_params_t.unsqueeze(0)  # Shape: (1, nT, 5)
        
    B = svi_params_t.shape[0]
    nT = len(T_t)
    nK = len(K_t)
    
    # 2. Define fine uniform grids
    fine_T = torch.linspace(T_t[0].item(), T_t[-1].item(), fine_T_points, device=device)
    # We extend the strike grid slightly to avoid boundary effects
    k_min = K_t[0].item() - 0.2
    k_max = K_t[-1].item() + 0.2
    fine_K = torch.linspace(k_min, k_max, fine_K_points, device=device)
    
    dT = fine_T[1] - fine_T[0]
    dK = fine_K[1] - fine_K[0]
    
    # 3. Vectorized linear interpolation of SVI parameters in T direction
    indices = torch.bucketize(fine_T, T_t) - 1
    indices = torch.clamp(indices, 0, nT - 2)
    
    T_left = T_t[indices]
    T_right = T_t[indices + 1]
    T_diff = torch.clamp(T_right - T_left, min=1e-8)
    weights = (fine_T - T_left) / T_diff
    weights = torch.clamp(weights, 0.0, 1.0)
    
    # Interpolated parameters of shape (B, fine_T_points)
    a_fine = svi_params_t[:, indices, 0] * (1.0 - weights) + svi_params_t[:, indices + 1, 0] * weights
    b_fine = svi_params_t[:, indices, 1] * (1.0 - weights) + svi_params_t[:, indices + 1, 1] * weights
    rho_fine = svi_params_t[:, indices, 2] * (1.0 - weights) + svi_params_t[:, indices + 1, 2] * weights
    m_fine = svi_params_t[:, indices, 3] * (1.0 - weights) + svi_params_t[:, indices + 1, 3] * weights
    sigma_fine = svi_params_t[:, indices, 4] * (1.0 - weights) + svi_params_t[:, indices + 1, 4] * weights
    
    # 4. Compute total variance W(T, K) on 2D fine grid for all batch elements
    a_exp = a_fine.unsqueeze(-1)      # (B, fine_T_points, 1)
    b_exp = b_fine.unsqueeze(-1)      # (B, fine_T_points, 1)
    rho_exp = rho_fine.unsqueeze(-1)  # (B, fine_T_points, 1)
    m_exp = m_fine.unsqueeze(-1)      # (B, fine_T_points, 1)
    sigma_exp = sigma_fine.unsqueeze(-1).clamp(min=1e-8)  # (B, fine_T_points, 1)
    k_exp = fine_K.unsqueeze(0).unsqueeze(0)  # (1, 1, fine_K_points)
    
    u = k_exp - m_exp
    sqrt_term = torch.sqrt(u ** 2 + sigma_exp ** 2)
    W = a_exp + b_exp * (rho_exp * u + sqrt_term)  # Shape: (B, fine_T_points, fine_K_points)
    
    # 5. Finite differences on the fine grid w.r.t T and k
    dW_dT = torch.zeros_like(W)
    dW_dT[:, 1:-1, :] = (W[:, 2:, :] - W[:, :-2, :]) / (2.0 * dT)
    dW_dT[:, 0, :] = (W[:, 1, :] - W[:, 0, :]) / dT
    dW_dT[:, -1, :] = (W[:, -1, :] - W[:, -2, :]) / dT
    
    dW_dk = torch.zeros_like(W)
    dW_dk[:, :, 1:-1] = (W[:, :, 2:] - W[:, :, :-2]) / (2.0 * dK)
    dW_dk[:, :, 0] = (W[:, :, 1] - W[:, :, 0]) / dK
    dW_dk[:, :, -1] = (W[:, :, -1] - W[:, :, -2]) / dK
    
    d2W_dk2 = torch.zeros_like(W)
    d2W_dk2[:, :, 1:-1] = (W[:, :, 2:] - 2.0 * W[:, :, 1:-1] + W[:, :, :-2]) / (dK ** 2)
    # First-order linear boundary extrapolation to improve stability
    d2W_dk2[:, :, 0] = 2.0 * d2W_dk2[:, :, 1] - d2W_dk2[:, :, 2]
    d2W_dk2[:, :, -1] = 2.0 * d2W_dk2[:, :, -2] - d2W_dk2[:, :, -3]
    
    # 6. Compute density condition denominator g(k)
    W_safe = torch.clamp(W, min=1e-8)
    term1 = (1.0 - (k_exp * dW_dk) / (2.0 * W_safe)) ** 2
    term2 = (dW_dk ** 2 / 4.0) * (1.0 / W_safe + 0.25)
    term3 = 0.5 * d2W_dk2
    g_k = term1 - term2 + term3
    
    # 7. Apply Dupire formula: local_variance = dW_dT / g_k
    local_var = torch.zeros_like(W)
    valid_mask = (g_k > 1e-8) & (dW_dT >= 0)
    local_var[valid_mask] = dW_dT[valid_mask] / g_k[valid_mask]
    local_var[~valid_mask] = -1.0  # Mark as invalid
    
    # Take square root to get local volatility
    local_vol = torch.zeros_like(local_var)
    local_vol[local_var >= 0] = torch.sqrt(local_var[local_var >= 0])
    local_vol[local_var < 0] = -1.0
    
    # 8. Vectorized bilinear interpolation to target grid (T_grid, K_grid)
    t_indices = torch.bucketize(T_t, fine_T) - 1
    t_indices = torch.clamp(t_indices, 0, fine_T_points - 2)
    t_left = fine_T[t_indices]
    t_right = fine_T[t_indices + 1]
    t_diff = torch.clamp(t_right - t_left, min=1e-8)
    t_weights = (T_t - t_left) / t_diff
    t_weights = torch.clamp(t_weights, 0.0, 1.0)
    
    k_indices = torch.bucketize(K_t, fine_K) - 1
    k_indices = torch.clamp(k_indices, 0, fine_K_points - 2)
    k_left = fine_K[k_indices]
    k_right = fine_K[k_indices + 1]
    k_diff = torch.clamp(k_right - k_left, min=1e-8)
    k_weights = (K_t - k_left) / k_diff
    k_weights = torch.clamp(k_weights, 0.0, 1.0)
    
    # Advanced indexing to extract the 4 corners for all batch elements
    t_idx_exp = t_indices.unsqueeze(-1)
    k_idx_exp = k_indices.unsqueeze(0)
    
    c00 = local_vol[:, t_idx_exp, k_idx_exp]
    c10 = local_vol[:, t_idx_exp + 1, k_idx_exp]
    c01 = local_vol[:, t_idx_exp, k_idx_exp + 1]
    c11 = local_vol[:, t_idx_exp + 1, k_idx_exp + 1]
    
    w_t = t_weights.unsqueeze(-1)  # (nT, 1)
    w_k = k_weights.unsqueeze(0)   # (1, nK)
    
    # Interpolate in T first
    c0 = c00 * (1.0 - w_t) + c10 * w_t
    c1 = c01 * (1.0 - w_t) + c11 * w_t
    # Interpolate in K
    lv_surface = c0 * (1.0 - w_k) + c1 * w_k
    
    if not is_batched:
        lv_surface = lv_surface[0]
        
    if is_numpy:
        return lv_surface.cpu().numpy()
    return lv_surface
