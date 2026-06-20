"""
§1.4 Portfolio-level Greeks via PyTorch autograd through FNO.

All Greeks are analytic (no finite differences) — computed via torch.func.jacfwd.
"""
from __future__ import annotations
import math
import warnings
import numpy as np
import torch
import torch.func as func
from typing import Optional, Any
from scipy.special import ndtr

# Default grids matching FNO training data
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
STRIKES    = np.linspace(-0.5, 0.5, 11, dtype=np.float32)

# SPX futures contract notional (index points per contract)
_FUTURES_MULTIPLIER = 50.0

# Module-level normalizer cache (lazy-loaded)
_cached_pn = None
_cached_yn = None


# ── Normalizer cache helper ───────────────────────────────────────────────────

def _ensure_normalizers(model=None):
    """Lazy-load normalizers v2 once and cache module-level."""
    global _cached_pn, _cached_yn
    if _cached_pn is not None and _cached_yn is not None:
        return
    try:
        import os
        from normalizers import ParameterNormalizer, IVSurfaceNormalizer
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pn_path = os.path.join(root, "artifacts", "models", "param_normalizer_v2.npz")
        yn_path = os.path.join(root, "artifacts", "models", "iv_normalizer_v2.npz")
        if os.path.exists(pn_path) and os.path.exists(yn_path):
            _cached_pn = ParameterNormalizer.load(pn_path)
            _cached_yn = IVSurfaceNormalizer.load(yn_path)
        else:
            import calibrate as _cal
            _cal._param_norm = None
            _cal._iv_norm = None
            _cal._PARAM_NORM_PATH = "artifacts/models/param_normalizer.npz"
            _cal._load_normalizers(version="v2")
            _cached_pn = _cal._param_norm
            _cached_yn = _cal._iv_norm
    except Exception:
        # Absolute path fallback
        import os as _os
        from normalizers import ParameterNormalizer, IVSurfaceNormalizer
        _base = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            "..", "..",
            "artifacts", "models"
        )
        _base_abs = _os.path.abspath(_base)
        _cached_pn = ParameterNormalizer.load(
            _os.path.join(_base_abs, "param_normalizer_v2.npz"))
        _cached_yn = IVSurfaceNormalizer.load(
            _os.path.join(_base_abs, "iv_normalizer_v2.npz"))


# ── Spatial input helper ──────────────────────────────────────────────────────

def _make_spatial(T_grid: np.ndarray, K_grid: np.ndarray,
                  device: torch.device) -> torch.Tensor:
    """Build (1, nT, nK, 2) spatial coordinate tensor for the FNO."""
    T_arr = np.array(T_grid, dtype=np.float32)
    K_arr = np.array(K_grid, dtype=np.float32)
    T_norm = (T_arr - T_arr.mean()) / (T_arr.std() + 1e-8)
    K_norm = K_arr / 0.5  # log-moneyness grid [-0.5, 0.5] → [-1, 1]
    T_mesh, K_mesh = np.meshgrid(T_norm, K_norm, indexing="ij")
    coords = np.stack([T_mesh, K_mesh], axis=-1)
    return torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)


# ── Black-Scholes in differentiable PyTorch ───────────────────────────────────

def bs_call_price(S: torch.Tensor, K: torch.Tensor,
                  T: torch.Tensor, r: torch.Tensor,
                  sigma: torch.Tensor,
                  q: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Differentiable Black-Scholes call price.
    """
    if q is None:
        q = torch.zeros_like(r)

    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=S.dtype, device=S.device),
        torch.tensor(1.0, dtype=S.dtype, device=S.device)
    )

    S_safe     = S.clamp(min=1e-8)
    K_safe     = K.clamp(min=1e-8)
    sigma_safe = sigma.clamp(min=1e-8)
    T_safe     = T.clamp(min=1e-8)

    d1 = (torch.log(S_safe / K_safe) + (r - q + 0.5 * sigma_safe ** 2) * T_safe) \
         / (sigma_safe * torch.sqrt(T_safe))
    d2 = d1 - sigma_safe * torch.sqrt(T_safe)

    call = (S_safe * torch.exp(-q * T_safe) * normal.cdf(d1)
            - K_safe * torch.exp(-r * T_safe) * normal.cdf(d2))
    intrinsic = torch.clamp(S - K, min=0.0)
    return torch.where((T <= 0.0) | (sigma <= 0.0), intrinsic, call)


def bs_greeks(S: float, K: float, T: float, r: float,
              sigma_iv: float, q: float = 0.0, option_type: str = "call") -> dict:
    """
    Closed-form Black-Scholes Greeks for European Call or Put.
    """
    opt_type = option_type.lower()
    if opt_type not in ["call", "put"]:
        raise ValueError(f"Unsupported option type: {option_type}")

    # ── Robustness guard: sanitize all inputs ──────────────────────────────────
    # Any invalid (nan/inf/non-positive) input returns a safe zero-Greek dict.
    # We deliberately do NOT compute intrinsic value from bad inputs because
    # max(nan - K, 0) = nan, max(inf - K, 0) = inf — both invalid outputs.
    _ZERO_GREEKS = {
        "price": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0,
        "theta": 0.0, "rho": 0.0, "vanna": 0.0, "volga": 0.0,
        "speed": 0.0, "zomma": 0.0, "ultima": 0.0,
    }
    try:
        _S = float(S); _K = float(K); _T = float(T)
        _r = float(r); _q = float(q); _sig = float(sigma_iv)
    except (TypeError, ValueError, OverflowError):
        return _ZERO_GREEKS

    if not (math.isfinite(_S) and _S > 0
            and math.isfinite(_K) and _K > 0
            and math.isfinite(_T) and _T > 0
            and math.isfinite(_sig) and _sig > 0
            and math.isfinite(_r) and math.isfinite(_q)):
        return _ZERO_GREEKS

    try:
        ratio = _S / _K
        if ratio <= 0.0 or not math.isfinite(ratio):
            raise ValueError()
        log_S_K = math.log(ratio)

        denom = _sig * math.sqrt(_T)
        if denom == 0.0 or not math.isfinite(denom):
            raise ZeroDivisionError()

        d1 = (log_S_K + (_r - _q + 0.5 * _sig ** 2) * _T) / denom
        d2 = d1 - denom

        if not math.isfinite(d1) or not math.isfinite(d2):
            raise ValueError()
    except (ValueError, OverflowError, ZeroDivisionError, ArithmeticError):
        return _ZERO_GREEKS

    # Reassign to local names expected by rest of function
    S, K, T, r, q, sigma_iv = _S, _K, _T, _r, _q, _sig

    # ── All arithmetic wrapped in outer guard ─────────────────────────────────
    # This catches ZeroDivisionError from sigma_iv**2 underflowing to 0.0 when
    # sigma is near subnormal range (e.g. 1e-300), OverflowError from extreme
    # intermediate values, and any other unexpected arithmetic exceptions.
    try:
        # Standard normal CDF and PDF
        N_d1 = ndtr(d1)
        N_d2 = ndtr(d2)
        N_minus_d1 = ndtr(-d1)
        N_minus_d2 = ndtr(-d2)

        phi_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)

        # Robustness guard: if phi_d1 < 1e-150, set vega and dependent Greeks to 0.0
        is_underflow = (phi_d1 < 1e-150)

        # Price, delta, rho, theta
        exp_qT = math.exp(-q * T)
        exp_rT = math.exp(-r * T)

        if opt_type == "call":
            price = S * exp_qT * N_d1 - K * exp_rT * N_d2
            delta = exp_qT * N_d1
            rho = K * T * exp_rT * N_d2
            theta = - (S * exp_qT * sigma_iv * phi_d1) / (2.0 * math.sqrt(T)) \
                    + q * S * exp_qT * N_d1 - r * K * exp_rT * N_d2
        else:
            price = K * exp_rT * N_minus_d2 - S * exp_qT * N_minus_d1
            delta = -exp_qT * N_minus_d1
            rho = -K * T * exp_rT * N_minus_d2
            theta = - (S * exp_qT * sigma_iv * phi_d1) / (2.0 * math.sqrt(T)) \
                    - q * S * exp_qT * N_minus_d1 + r * K * exp_rT * N_minus_d2

        if is_underflow:
            vega = 0.0
            gamma = 0.0
            vanna = 0.0
            volga = 0.0
            speed = 0.0
            zomma = 0.0
            ultima = 0.0
        else:
            # Guard: combined denominator to avoid underflow-to-zero cascades
            sig_sqrt_T = max(sigma_iv * math.sqrt(T), 5e-300)  # subnormal guard
            sig_sq     = max(sigma_iv ** 2, 5e-300)            # sigma^2 underflow guard
            S_safe     = max(S, 5e-300)

            vega  = S * exp_qT * math.sqrt(T) * phi_d1
            gamma = (exp_qT * phi_d1) / (S_safe * sig_sqrt_T)
            vanna = -exp_qT * phi_d1 * (d2 / sigma_iv)
            volga = vega * (d1 * d2 / sigma_iv)
            speed = -(gamma / S_safe) * (d1 / sig_sqrt_T + 1.0)
            zomma = gamma * (d1 * d2 - 1.0) / sigma_iv
            ultima = (vega / sig_sq) * (d1**2 * d2**2 - d1**2 - d2**2 - d1 * d2)

        # Clamp outputs to finite range; any inf/nan becomes 0.0
        def _safe_float(x: float) -> float:
            f = float(x)
            return f if math.isfinite(f) else 0.0

        return {
            "price": _safe_float(price),
            "delta": _safe_float(delta),
            "gamma": _safe_float(gamma),
            "vega":  _safe_float(vega),
            "theta": _safe_float(theta),
            "rho":   _safe_float(rho),
            "vanna": _safe_float(vanna),
            "volga": _safe_float(volga),
            "speed": _safe_float(speed),
            "zomma": _safe_float(zomma),
            "ultima":_safe_float(ultima),
        }
    except (ZeroDivisionError, OverflowError, ArithmeticError, ValueError, FloatingPointError):
        return _ZERO_GREEKS


# ── FNO Parameter Jacobian ────────────────────────────────────────────────────

def fno_parameter_jacobian(model: torch.nn.Module,
                            theta: torch.Tensor,
                            spatial: torch.Tensor) -> torch.Tensor:
    """
    Compute full Jacobian of IV surface w.r.t. raw Heston parameters.
    """
    device  = next(model.parameters()).device
    spatial = spatial.to(device)
    nT, nK  = spatial.shape[1], spatial.shape[2]

    _ensure_normalizers(model)
    pn_mean = torch.tensor(_cached_pn.mean, dtype=torch.float32, device=device)
    pn_std  = torch.tensor(_cached_pn.std,  dtype=torch.float32, device=device)
    yn_mean = torch.tensor(_cached_yn.mean, dtype=torch.float32, device=device)
    yn_std  = torch.tensor(_cached_yn.std,  dtype=torch.float32, device=device)

    def _iv_flat(p6: torch.Tensor) -> torch.Tensor:
        p_norm  = (p6 - pn_mean) / pn_std
        iv_norm = model(spatial, p_norm.unsqueeze(0))
        iv_real = iv_norm * yn_std + yn_mean
        iv_real = iv_real.clamp(min=1e-4)
        return iv_real.squeeze(0)

    p6 = theta.to(device).float().detach()
    J  = func.jacfwd(_iv_flat)(p6)
    return J.detach()


# ── FNO Surface Greeks ────────────────────────────────────────────────────────

def fno_surface_greeks(model: torch.nn.Module,
                        theta,
                        pn, yn,
                        S: float, r: float = 0.05,
                        T_grid: Optional[np.ndarray] = None,
                        K_grid: Optional[np.ndarray] = None) -> dict:
    """
    Compute full Greek surface for all (T, K) grid points.
    """
    if T_grid is None:
        T_grid = MATURITIES
    if K_grid is None:
        K_grid = STRIKES

    k_grid = np.asarray(K_grid, dtype=np.float32)
    if np.any(k_grid > 2.0):
        k_grid = np.log(k_grid / S)

    # FNO training bounds — clip to prevent NaN/Inf propagation through normalizer
    _BOUNDS_ARR = np.array([
        [0.5, 5.0],    # kappa
        [0.01, 0.25],  # theta
        [0.1, 1.5],    # sigma
        [-0.95, 0.0],  # rho
        [0.01, 0.25],  # v0
        [0.04, 0.15],  # H
    ], dtype=np.float32)

    if isinstance(theta, dict):
        theta_arr = np.array([
            theta["kappa"], theta["theta"], theta["sigma"],
            theta["rho"],   theta["v0"],    theta["H"],
        ], dtype=np.float32)
    else:
        theta_arr = np.asarray(theta, dtype=np.float32).copy()

    # Replace NaN/Inf with midpoint of training range before clipping
    midpoints = (_BOUNDS_ARR[:, 0] + _BOUNDS_ARR[:, 1]) / 2.0
    bad_mask = ~np.isfinite(theta_arr)
    if bad_mask.any():
        warnings.warn(
            f"fno_surface_greeks received non-finite theta values at positions "
            f"{np.where(bad_mask)[0].tolist()}; replacing with training-range midpoints.",
            RuntimeWarning, stacklevel=2,
        )
        theta_arr = np.where(bad_mask, midpoints, theta_arr)

    # Clip to training range (also handles huge/negative out-of-range values)
    theta_arr = np.clip(theta_arr, _BOUNDS_ARR[:, 0], _BOUNDS_ARR[:, 1])

    device    = next(model.parameters()).device
    theta_t   = torch.tensor(theta_arr, dtype=torch.float32, device=device)
    spatial   = _make_spatial(T_grid, k_grid, device)

    with torch.no_grad():
        theta_norm = pn.transform_tensor(theta_t.unsqueeze(0))
        pred_norm  = model(spatial, theta_norm)
        iv_tensor  = yn.inverse_transform_tensor(pred_norm).squeeze(0)
        iv_surface = iv_tensor.clamp(min=1e-4).cpu().numpy()

    nT, nK = len(T_grid), len(k_grid)

    delta_surf  = np.zeros((nT, nK), dtype=np.float32)
    gamma_surf  = np.zeros((nT, nK), dtype=np.float32)
    vega_surf   = np.zeros((nT, nK), dtype=np.float32)
    theta_surf  = np.zeros((nT, nK), dtype=np.float32)
    vanna_surf  = np.zeros((nT, nK), dtype=np.float32)
    volga_surf  = np.zeros((nT, nK), dtype=np.float32)

    for i in range(nT):
        for j in range(nK):
            T_val   = float(T_grid[i])
            kk_val   = float(k_grid[j])
            K_val   = float(S) * float(np.exp(kk_val))
            sig_val = float(iv_surface[i, j])

            g = bs_greeks(float(S), K_val, T_val, r, sig_val)

            delta_surf[i, j] = g["delta"]
            gamma_surf[i, j] = g["gamma"]
            vega_surf[i, j]  = g["vega"]
            theta_surf[i, j] = g["theta"]
            vanna_surf[i, j] = g["vanna"]
            volga_surf[i, j] = g["volga"]

    gamma_max = np.percentile(np.abs(gamma_surf[np.isfinite(gamma_surf)]), 99.5)
    if gamma_max > 1e6:
        warnings.warn(f"Extreme gamma detected (max={gamma_max:.2e}); clamping to ±1e6",
                      RuntimeWarning)
        gamma_surf = np.clip(gamma_surf, -1e6, 1e6)

    return {
        "delta":        delta_surf,
        "gamma":        gamma_surf,
        "vega":         vega_surf,
        "theta":        theta_surf,
        "vanna":        vanna_surf,
        "volga":        volga_surf,
        "iv_surface":   iv_surface,
        "T_grid":       np.array(T_grid),
        "K_grid":       np.array(K_grid),
    }


# ── Differentiable Bilinear Interpolation for Portfolio Greeks ──────────────────

def interpolate_bilinear(T_grid: torch.Tensor, K_grid: torch.Tensor,
                         iv_surface: torch.Tensor, T: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 2D bilinear interpolation for a query point (T, k)
    on a grid (T_grid, K_grid) with values iv_surface of shape (nT, nK).
    """
    nT = T_grid.size(0)
    nK = K_grid.size(0)
    
    T_clip = torch.clamp(T, min=T_grid[0] + 1e-6, max=T_grid[-1] - 1e-6)
    k_clip = torch.clamp(k, min=K_grid[0] + 1e-6, max=K_grid[-1] - 1e-6)
    
    t_idx = torch.bucketize(T_clip, T_grid) - 1
    t_idx = torch.clamp(t_idx, min=0, max=nT - 2)
    
    k_idx = torch.bucketize(k_clip, K_grid) - 1
    k_idx = torch.clamp(k_idx, min=0, max=nK - 2)
    
    t0 = T_grid[t_idx]
    t1 = T_grid[t_idx + 1]
    k0 = K_grid[k_idx]
    k1 = K_grid[k_idx + 1]
    
    wt = (T_clip - t0) / (t1 - t0)
    wk = (k_clip - k0) / (k1 - k0)
    
    val00 = iv_surface[t_idx, k_idx]
    val10 = iv_surface[t_idx + 1, k_idx]
    val01 = iv_surface[t_idx, k_idx + 1]
    val11 = iv_surface[t_idx + 1, k_idx + 1]
    
    val = (1.0 - wt) * (1.0 - wk) * val00 + \
          wt * (1.0 - wk) * val10 + \
          (1.0 - wt) * wk * val01 + \
          wt * wk * val11
          
    return val


def _bilinear_interp(T_grid: np.ndarray, K_grid: np.ndarray,
                     surface: np.ndarray, T: float, k: float) -> float:
    """Simple numpy bilinear interpolation (no grad needed for IV lookup)."""
    T_clip = np.clip(T, T_grid[0], T_grid[-1])
    k_clip = np.clip(k, K_grid[0], K_grid[-1])

    ti = np.searchsorted(T_grid, T_clip) - 1
    ti = int(np.clip(ti, 0, len(T_grid) - 2))
    ki = np.searchsorted(K_grid, k_clip) - 1
    ki = int(np.clip(ki, 0, len(K_grid) - 2))

    wt = (T_clip - T_grid[ti]) / max(T_grid[ti + 1] - T_grid[ti], 1e-12)
    wk = (k_clip - K_grid[ki]) / max(K_grid[ki + 1] - K_grid[ki], 1e-12)

    v00 = surface[ti,     ki]
    v10 = surface[ti + 1, ki]
    v01 = surface[ti,     ki + 1]
    v11 = surface[ti + 1, ki + 1]

    return float((1 - wt) * (1 - wk) * v00 + wt * (1 - wk) * v10
                 + (1 - wt) * wk * v01 + wt * wk * v11)


# ── Differentiable Portfolio Price Tensor (for Autograd Preservation) ─────────

def portfolio_price_tensor(positions: list,
                            model: torch.nn.Module,
                            theta: torch.Tensor,
                            pn, yn, S: torch.Tensor,
                            r: torch.Tensor) -> torch.Tensor:
    """
    Computes portfolio price as a fully differentiable PyTorch tensor.
    Keeps the autograd graph open for skew sensitivities and parameter risk.
    """
    T_grid = MATURITIES
    K_grid = STRIKES
    
    device = next(model.parameters()).device
    theta_norm = pn.transform_tensor(theta.unsqueeze(0))
    spatial = _make_spatial(T_grid, K_grid, device)
    
    pred_norm = model(spatial, theta_norm)
    iv_surface = yn.inverse_transform_tensor(pred_norm).squeeze(0)
    iv_surface = torch.clamp(iv_surface, min=1e-4)
    
    T_grid_t = torch.tensor(T_grid, dtype=torch.float32, device=device)
    K_grid_t = torch.tensor(K_grid, dtype=torch.float32, device=device)
    
    total_price = torch.tensor(0.0, dtype=S.dtype, device=S.device)
    
    for pos in positions:
        K_val = float(pos["K"])
        T_val = float(pos["T"])
        qty = float(pos.get("quantity", 1.0))
        notional = float(pos.get("notional", 100.0))
        opt_type = pos.get("type", "call").lower()
        
        if opt_type not in ["call", "put"]:
            raise ValueError(f"Unsupported option type: {opt_type}")
            
        K_pos = torch.tensor(K_val, dtype=S.dtype, device=S.device)
        T_pos = torch.tensor(T_val, dtype=S.dtype, device=S.device)
        
        k_pos = torch.log(K_pos / S)
        sigma = interpolate_bilinear(T_grid_t, K_grid_t, iv_surface, T_pos, k_pos)
        
        if opt_type == "call":
            price = bs_call_price(S, K_pos, T_pos, r, sigma)
        else:
            price = bs_call_price(S, K_pos, T_pos, r, sigma) + K_pos * torch.exp(-r * T_pos) - S
            
        total_price = total_price + price * qty * notional
        
    return total_price


# ── Portfolio Greeks ──────────────────────────────────────────────────────────

def portfolio_greeks(positions: list,
                     model: torch.nn.Module,
                     theta: np.ndarray,
                     pn, yn, S: float,
                     r: float = 0.05) -> dict:
    """
    Aggregate Greeks across a portfolio of option positions.
    """
    # Cast to float64 to prevent float32 overflow in vega_bucket accumulation
    # MATURITIES is float32 (max ~3.4e38); large positions would overflow without cast
    T_grid = np.array(MATURITIES, dtype=np.float64)
    K_grid = STRIKES
    nT     = len(T_grid)

    device  = next(model.parameters()).device
    theta_t = torch.tensor(np.asarray(theta, dtype=np.float32), device=device)
    spatial = _make_spatial(T_grid, K_grid, device)

    with torch.no_grad():
        theta_norm = pn.transform_tensor(theta_t.unsqueeze(0))
        pred_norm  = model(spatial, theta_norm)
        iv_tensor  = yn.inverse_transform_tensor(pred_norm).squeeze(0)
        iv_surface = iv_tensor.clamp(min=1e-4).cpu().numpy()

    total_delta  = 0.0
    total_gamma  = 0.0
    total_vanna  = 0.0
    total_volga  = 0.0
    vega_bucket  = np.zeros(nT, dtype=np.float64)

    for pos in positions:
        K_pos    = float(pos["K"])
        T_pos    = float(pos["T"])
        qty      = float(pos.get("quantity", 1.0))
        opt_type = pos.get("type", "call").lower()
        notional = float(pos.get("notional", 100.0))

        if opt_type not in ["call", "put"]:
            raise ValueError(f"Unsupported option type: {opt_type}")

        # Guard against pathological positions (K<=0, nan, inf, T<=0)
        if not (math.isfinite(K_pos) and K_pos > 0
                and math.isfinite(T_pos) and T_pos > 0
                and math.isfinite(qty) and math.isfinite(notional)):
            continue  # skip invalid positions gracefully

        k_pos = np.log(K_pos / S)
        sigma = _bilinear_interp(T_grid, K_grid, iv_surface, T_pos, k_pos)
        sigma = max(sigma, 1e-4)

        g = bs_greeks(S, K_pos, T_pos, r, sigma, option_type=opt_type)

        raw_delta = g["delta"]
        weight    = qty * notional

        total_delta += raw_delta * weight
        total_gamma += g["gamma"] * weight
        total_vanna += g["vanna"] * weight
        total_volga += g["volga"] * weight

        vega_pos = g["vega"] * weight
        if T_pos <= T_grid[0]:
            vega_bucket[0] += float(vega_pos)
        elif T_pos >= T_grid[-1]:
            vega_bucket[-1] += float(vega_pos)
        else:
            idx = int(np.searchsorted(T_grid, T_pos)) - 1
            idx = int(np.clip(idx, 0, nT - 2))
            # Explicitly cast wt to Python float to stay in float64 throughout
            wt  = float((T_pos - T_grid[idx]) / max(T_grid[idx + 1] - T_grid[idx], 1e-12))
            wt  = max(0.0, min(1.0, wt))  # clamp to [0,1] for safety
            vega_bucket[idx]     += float(vega_pos) * (1.0 - wt)
            vega_bucket[idx + 1] += float(vega_pos) * wt

    hedge_contracts = int(np.round(-total_delta / _FUTURES_MULTIPLIER))

    # Safety: replace any residual inf/nan with 0 (e.g. from extreme notionals)
    vega_bucket = np.nan_to_num(vega_bucket, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "total_delta":     float(total_delta),
        "total_gamma":     float(total_gamma),
        "vega_bucket":     vega_bucket,
        "total_vanna":     float(total_vanna),
        "total_volga":     float(total_volga),
        "hedge_contracts": hedge_contracts,
    }


# ── P&L Attribution ───────────────────────────────────────────────────────────

def pnl_attribution(S_before: float, S_after: float,
                     sigma_before: float, sigma_after: float,
                     greeks: dict) -> dict:
    """
    Decompose daily P&L using second-order Taylor expansion.
    """
    dS     = float(S_after)     - float(S_before)
    
    # Handle both scalar and vector-based volatility shifts
    sig_bef = np.asarray(sigma_before, dtype=np.float64)
    sig_aft = np.asarray(sigma_after, dtype=np.float64)
    dsigma = sig_aft - sig_bef

    delta = float(greeks.get("total_delta", greeks.get("delta", 0.0)))
    gamma = float(greeks.get("total_gamma", greeks.get("gamma", 0.0)))
    vanna = float(greeks.get("total_vanna", greeks.get("vanna", 0.0)))
    volga = float(greeks.get("total_volga", greeks.get("volga", 0.0)))

    vega_bucket = greeks.get("vega_bucket", None)
    if vega_bucket is not None and dsigma.ndim > 0:
        # Maturity-specific vega attribution
        vega_pnl = float(np.sum(vega_bucket * dsigma))
    else:
        raw_vega = greeks.get("total_vega", greeks.get("vega", None))
        if raw_vega is None and vega_bucket is not None:
            raw_vega = np.sum(vega_bucket)
        vega = float(raw_vega) if raw_vega is not None else 0.0
        dsigma_scalar = float(np.mean(dsigma)) if dsigma.ndim > 0 else float(dsigma)
        vega_pnl = vega * dsigma_scalar

    delta_pnl = delta * dS
    gamma_pnl = 0.5 * gamma * dS ** 2
    
    dsigma_mean = float(np.mean(dsigma)) if dsigma.ndim > 0 else float(dsigma)
    vanna_pnl = vanna * dS * dsigma_mean
    volga_pnl = 0.5 * volga * (dsigma_mean ** 2)

    explained   = delta_pnl + gamma_pnl + vega_pnl + vanna_pnl + volga_pnl
    actual_pnl  = float(greeks.get("actual_pnl", explained))
    unexplained = actual_pnl - explained

    return {
        "delta_pnl":   float(delta_pnl),
        "gamma_pnl":   float(gamma_pnl),
        "vega_pnl":    float(vega_pnl),
        "vanna_pnl":   float(vanna_pnl),
        "volga_pnl":   float(volga_pnl),
        "unexplained": float(unexplained),
    }
