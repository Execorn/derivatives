"""
fallbacks.py — Fallback pricing engines, regularized Tikhonov calibration solvers, and SR 26-2 model governance compliance.
"""

import time
import math
import logging
import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats
import torch
from typing import Dict, Any, Union, Optional, List

from deepvol.calibration.interface import CalibrationResult, _get_default_model
from deepvol.calibration.calibrate_bfgs import (
    _load_normalizers,
    _make_spatial_input,
    _fno_predict_real_iv
)
from deepvol.models.heston import batch_heston_iv_surface, implied_vol_cpu
from deepvol.models.mlsv_gpu import MLSVSolverGPU
from deepvol.mrm.guardian import price_to_iv

# Bounding limits
_BOUNDS_LOWER_HESTON = torch.tensor([0.5, 0.01, 0.1, -0.95, 0.01])
_BOUNDS_UPPER_HESTON = torch.tensor([10.0, 0.25, 2.0, -0.01, 0.25])
_BOUNDS_LOWER_RBERGOMI = torch.tensor([0.01, 0.04, 0.5, -0.95])
_BOUNDS_UPPER_RBERGOMI = torch.tensor([0.20, 0.15, 4.0, 0.0])

# Global historical buffer for online drift tracking
_HISTORICAL_CALIBRATIONS: Dict[str, List[np.ndarray]] = {
    "heston": [],
    "rbergomi": []
}

# ---------------------------------------------------------------------------
# Fallback Pricing Engines
# ---------------------------------------------------------------------------

class FourierCOSEngine:
    """
    Exact analytical Fourier-COS pricing engine for the Heston model.
    Runs on CPU/GPU in double precision (float64) and clamps min vol to 0.01.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        
    def price_surface(
        self,
        params: dict,
        maturities: np.ndarray,
        strikes: np.ndarray,
        S0: float = 1.0,
        device: Optional[str] = None
    ) -> dict:
        dev = device or self.device
        device_obj = torch.device(dev)
        
        # Clamp parameters to prevent singularities
        kappa = max(float(params.get('kappa', 2.0)), 0.01)
        theta = max(float(params.get('theta', 0.04)), 1e-4) # sqrt(theta) >= 0.01 => theta >= 1e-4
        sigma = max(float(params.get('sigma', 0.3)), 0.01)
        rho = max(-0.9999, min(float(params.get('rho', -0.7)), 0.9999))
        v0 = max(float(params.get('v0', 0.04)), 1e-4)
        
        # Format params to tensor shape (1, 5)
        param_vector = torch.tensor([
            [kappa, theta, sigma, rho, v0]
        ], dtype=torch.float64, device=device_obj)
        
        T_t = torch.tensor(maturities, dtype=torch.float64, device=device_obj)
        K_t = torch.tensor(strikes, dtype=torch.float64, device=device_obj)
        
        # Price surface using deepvol's batch_heston_iv_surface
        ivs_t = batch_heston_iv_surface(
            params=param_vector,
            T_grid=T_t,
            K_grid=K_t,
            S0=S0,
            N_cos=256,
            device=str(device_obj)
        )
        
        ivs = ivs_t.squeeze(0).detach().cpu().numpy()
        
        # Fill NaNs with 0.20 and clamp min vol to 0.01 to prevent Durrleman singularities
        ivs = np.where(np.isnan(ivs), 0.20, ivs)
        ivs = np.clip(ivs, 0.01, None)
        
        # Calculate Option Prices in float64 using Black-Scholes call pricing formula
        T_m = maturities[:, None]
        K_m = S0 * np.exp(strikes)[None, :] if np.any(strikes < 0) or np.max(np.abs(strikes)) < 5.0 else strikes[None, :]
        vol_std = ivs * np.sqrt(T_m)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (np.log(S0 / K_m) + 0.5 * vol_std**2) / np.clip(vol_std, 1e-9, None)
            d2 = d1 - vol_std
            
        prices = S0 * stats.norm.cdf(d1) - K_m * stats.norm.cdf(d2)
        prices = np.where(vol_std <= 1e-8, np.maximum(S0 - K_m, 0.0), prices)
        
        return {
            "prices": prices,
            "ivs": ivs
        }


class McKeanVlasovFallbackEngine:
    """
    McKean-Vlasov SDE particle solver fallback pricing engine.
    Uses GPU acceleration when available, operates in float64, and clamps min vol to 0.01.
    Prices the entire grid using a single contiguous simulation run.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
    def price_surface(
        self,
        params: dict,
        maturities: np.ndarray,
        strikes: np.ndarray,
        S0: float = 1.0,
        device: Optional[str] = None
    ) -> dict:
        dev = device or self.device
        
        # Extract parameters
        kappa = max(float(params.get("kappa", 2.0)), 0.01)
        theta = max(float(params.get("theta", 0.04)), 1e-4)
        epsilon = max(float(params.get("epsilon", params.get("sigma", 0.3))), 0.01)
        rho = max(-0.9999, min(float(params.get("rho", -0.7)), 0.9999))
        vol_init = max(float(np.sqrt(theta)), 0.01)
        
        # Ensure strikes are absolute
        if np.any(strikes < 0) or np.max(np.abs(strikes)) < 5.0:
            strikes_abs = S0 * np.exp(strikes)
        else:
            strikes_abs = strikes.copy()
            
        # Initialize MLSVSolverGPU in float64
        solver = MLSVSolverGPU(
            S0=S0,
            r=0.0,
            q=0.0,
            v0=vol_init**2,
            kappa=kappa,
            theta=theta,
            xi=epsilon,
            rho=rho,
            T=float(max(maturities)),
            steps_per_unit=50,
            N_paths=2000,
            dupire_vol_fn=lambda t, s: torch.full_like(s, vol_init),
            device=dev,
            dtype=torch.float64
        )
        
        # Simulate paths using Nadaraya-Watson kernel density regression
        solver.simulate(method="nadaraya_watson")
        
        # Option prices tensor shape (nT, nK)
        prices_t = solver.price_european_option(strike=strikes_abs, maturity=maturities)
        prices = prices_t.detach().cpu().numpy()
        
        nT, nK = len(maturities), len(strikes)
        ivs = np.zeros((nT, nK))
        
        # Invert Call prices to implied volatilities
        for i, T in enumerate(maturities):
            for j, K_val in enumerate(strikes_abs):
                price = prices[i, j]
                try:
                    iv = price_to_iv(price, S0, K_val, T)
                    if np.isnan(iv) or not np.isfinite(iv):
                        iv = vol_init
                except Exception:
                    iv = vol_init
                ivs[i, j] = iv
                
        # Clamp min vol to 0.01 to prevent Durrleman singularities
        ivs = np.clip(ivs, 0.01, None)
        
        return {
            "prices": prices,
            "ivs": ivs
        }


# ---------------------------------------------------------------------------
# Regularized Calibration Solver
# ---------------------------------------------------------------------------

def calibrate_tikhonov(
    market_iv_surface: np.ndarray,
    model_name: str,
    p_prior: np.ndarray,
    lmbda: float,
    T_grid: np.ndarray,
    K_grid: np.ndarray,
    fixed_H: Optional[float] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model = None,
    max_iter: int = 150,
    **kwargs
) -> CalibrationResult:
    """
    Calibrate volatility model (Heston or Rough Bergomi) using regularized Tikhonov solver.
    Performs optimization in logit-transformed space to enforce strict parameter bounds.
    """
    t_start = time.time()
    device_obj = torch.device(device)
    model_name = model_name.lower()
    
    # 1. Resolve/Load the FNO model
    if model is None:
        model = _get_default_model(model_name, device_obj)
    else:
        model.to(device_obj)
    model.eval()
    
    # Load normalizers and boundaries based on model name
    if model_name == "heston":
        _load_normalizers("heston")
        lo = _BOUNDS_LOWER_HESTON.to(device_obj)
        hi = _BOUNDS_UPPER_HESTON.to(device_obj)
    elif model_name in ("rbergomi", "rough_bergomi"):
        _load_normalizers("rbergomi")
        lo = _BOUNDS_LOWER_RBERGOMI.to(device_obj)
        hi = _BOUNDS_UPPER_RBERGOMI.to(device_obj)
    else:
        raise ValueError(f"Tikhonov calibration not supported for model: {model_name}")
        
    spatial = _make_spatial_input(T_grid, K_grid, device_obj)
    target_t = torch.tensor(market_iv_surface, dtype=torch.float32, device=device_obj)
    
    # Helper logit mappings
    def to_logit(x, a, b):
        s = (x - a) / (b - a)
        s = torch.clamp(s, 1e-6, 1.0 - 1e-6)
        return torch.log(s / (1.0 - s))
        
    def from_logit(y, a, b):
        s = torch.sigmoid(y)
        return a + s * (b - a)
        
    p_prior_t = torch.tensor(p_prior, dtype=torch.float64, device=device_obj)
    
    if model_name == "heston":
        # 5D: kappa, theta, sigma, rho, v0
        prior_reparam = to_logit(p_prior_t, lo, hi)
        init_reparam = prior_reparam.detach().cpu().numpy()
        
        def loss_and_grad(x_arr):
            x_t = torch.tensor(x_arr, dtype=torch.float64, device=device_obj, requires_grad=True)
            raw = from_logit(x_t, lo, hi)
            pred_iv = _fno_predict_real_iv(model, raw.unsqueeze(0).to(torch.float32), spatial)
            
            # MSE loss + Tikhonov penalty
            mse = torch.nn.functional.mse_loss(pred_iv.to(torch.float64), target_t.to(torch.float64))
            penalty = lmbda * torch.sum((x_t - prior_reparam) ** 2)
            
            # Feller condition penalty
            feller = raw[2]**2 - 2.0 * raw[0] * raw[1]
            feller_penalty = 10.0 * torch.clamp(feller, min=0.0)**2
            
            loss_val = mse + penalty + feller_penalty
            loss_val.backward()
            
            grad = x_t.grad.detach().cpu().numpy()
            return loss_val.item(), grad
            
    else:  # rbergomi
        if fixed_H is not None:
            # 3D: v0, eta, rho (H is fixed)
            lo_3d = torch.tensor([lo[0], lo[2], lo[3]], device=device_obj)
            hi_3d = torch.tensor([hi[0], hi[2], hi[3]], device=device_obj)
            
            prior_active = torch.tensor([p_prior[0], p_prior[2], p_prior[3]], dtype=torch.float64, device=device_obj)
            prior_reparam = to_logit(prior_active, lo_3d, hi_3d)
            init_reparam = prior_reparam.detach().cpu().numpy()
            
            def loss_and_grad(x_arr):
                x_t = torch.tensor(x_arr, dtype=torch.float64, device=device_obj, requires_grad=True)
                raw_active = from_logit(x_t, lo_3d, hi_3d)
                
                v0_v = raw_active[0]
                H_v = torch.tensor(fixed_H, dtype=torch.float64, device=device_obj)
                eta_v = raw_active[1]
                rho_v = raw_active[2]
                
                raw = torch.stack([v0_v, H_v, eta_v, rho_v])
                pred_iv = _fno_predict_real_iv(model, raw.unsqueeze(0).to(torch.float32), spatial)
                
                mse = torch.nn.functional.mse_loss(pred_iv.to(torch.float64), target_t.to(torch.float64))
                penalty = lmbda * torch.sum((x_t - prior_reparam) ** 2)
                
                loss_val = mse + penalty
                loss_val.backward()
                
                grad = x_t.grad.detach().cpu().numpy()
                return loss_val.item(), grad
        else:
            # 4D: v0, H, eta, rho
            prior_reparam = to_logit(p_prior_t, lo, hi)
            init_reparam = prior_reparam.detach().cpu().numpy()
            
            def loss_and_grad(x_arr):
                x_t = torch.tensor(x_arr, dtype=torch.float64, device=device_obj, requires_grad=True)
                raw = from_logit(x_t, lo, hi)
                pred_iv = _fno_predict_real_iv(model, raw.unsqueeze(0).to(torch.float32), spatial)
                
                mse = torch.nn.functional.mse_loss(pred_iv.to(torch.float64), target_t.to(torch.float64))
                penalty = lmbda * torch.sum((x_t - prior_reparam) ** 2)
                
                loss_val = mse + penalty
                loss_val.backward()
                
                grad = x_t.grad.detach().cpu().numpy()
                return loss_val.item(), grad
                
    # Solve using L-BFGS-B solver
    res = optimize.minimize(
        fun=loss_and_grad,
        x0=init_reparam,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-6}
    )
    
    # Decode optimal parameters
    x_opt = torch.tensor(res.x, dtype=torch.float64, device=device_obj)
    if model_name == "heston":
        raw_opt = from_logit(x_opt, lo, hi)
    else:
        if fixed_H is not None:
            lo_3d = torch.tensor([lo[0], lo[2], lo[3]], device=device_obj)
            hi_3d = torch.tensor([hi[0], hi[2], hi[3]], device=device_obj)
            raw_active = from_logit(x_opt, lo_3d, hi_3d)
            raw_opt = torch.stack([raw_active[0], torch.tensor(fixed_H, dtype=torch.float64, device=device_obj), raw_active[1], raw_active[2]])
        else:
            raw_opt = from_logit(x_opt, lo, hi)
            
    final_params = raw_opt.detach().cpu().numpy()
    
    # Calculate final RMSE
    with torch.no_grad():
        final_raw_t = torch.tensor(final_params, dtype=torch.float32, device=device_obj)
        pred_iv_final = _fno_predict_real_iv(model, final_raw_t.unsqueeze(0), spatial)
        final_rmse = float(torch.sqrt(torch.nn.functional.mse_loss(pred_iv_final, target_t)).item())
        
    elapsed = time.time() - t_start
    status = "converged" if res.success else "failed"
    
    # Model governance tracking: OOD check + Online drift tracking
    compliance = check_ood_parameters(model_name, final_params)
    final_params_clamped = compliance["clamped_params"]
    
    # Update historical buffer for online drift tracking
    _HISTORICAL_CALIBRATIONS[model_name].append(final_params_clamped)
    if len(_HISTORICAL_CALIBRATIONS[model_name]) > 100:
         _HISTORICAL_CALIBRATIONS[model_name].pop(0)
         
    return CalibrationResult(
        parameters=final_params_clamped,
        rmse=final_rmse,
        elapsed_time=elapsed,
        status=status,
        info={
            "loss": res.fun,
            "message": res.message,
            "iterations": res.nit,
            "ood_details": compliance,
            "drift_logged": True
        }
    )


# ---------------------------------------------------------------------------
# SR 26-2 Model Governance & Compliance
# ---------------------------------------------------------------------------

def calculate_psi(baseline: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI) between baseline and actual distributions.
    """
    baseline = np.asarray(baseline)
    actual = np.asarray(actual)
    
    # Define bin edges using percentiles of baseline
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline, percentiles)
    
    # Adjust outer edges to avoid boundary issues
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    
    # Calculate counts in each bin
    b_counts, _ = np.histogram(baseline, bins=bin_edges)
    a_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Convert to fractions with a tiny epsilon to avoid division by zero / log(0)
    eps = 1e-4
    b_fracs = (b_counts + eps) / (len(baseline) + num_bins * eps)
    a_fracs = (a_counts + eps) / (len(actual) + num_bins * eps)
    
    # Calculate PSI
    psi = np.sum((a_fracs - b_fracs) * np.log(a_fracs / b_fracs))
    return float(psi)


def check_ood_parameters(model_name: str, parameters: np.ndarray) -> dict:
    """
    Check if parameters are out-of-distribution (OOD) and return clamping decisions and compliance logs.
    """
    model_name = model_name.lower()
    params = np.asarray(parameters, dtype=float)
    clamped_params = params.copy()
    logs = []
    is_ood = False
    
    if model_name == "heston":
        kappa, theta, sigma, rho, v0 = params
        if kappa < 0.5 or kappa > 10.0:
            is_ood = True
            clamped_params[0] = np.clip(kappa, 0.5, 10.0)
            logs.append(f"Heston kappa={kappa:.4f} is OOD. Clamped to {clamped_params[0]:.4f}")
        if theta < 0.01 or theta > 0.25:
            is_ood = True
            clamped_params[1] = np.clip(theta, 0.01, 0.25)
            logs.append(f"Heston theta={theta:.4f} is OOD. Clamped to {clamped_params[1]:.4f}")
        if sigma < 0.1 or sigma > 2.0:
            is_ood = True
            clamped_params[2] = np.clip(sigma, 0.1, 2.0)
            logs.append(f"Heston sigma={sigma:.4f} is OOD. Clamped to {clamped_params[2]:.4f}")
        if rho < -0.95 or rho > -0.01:
            is_ood = True
            clamped_params[3] = np.clip(rho, -0.95, -0.01)
            logs.append(f"Heston rho={rho:.4f} is OOD. Clamped to {clamped_params[3]:.4f}")
        if v0 < 0.01 or v0 > 0.25:
            is_ood = True
            clamped_params[4] = np.clip(v0, 0.01, 0.25)
            logs.append(f"Heston v0={v0:.4f} is OOD. Clamped to {clamped_params[4]:.4f}")
            
    elif model_name in ("rbergomi", "rough_bergomi"):
        v0, H, eta, rho = params
        if v0 < 0.01 or v0 > 0.20:
            is_ood = True
            clamped_params[0] = np.clip(v0, 0.01, 0.20)
            logs.append(f"Rough Bergomi v0={v0:.4f} is OOD. Clamped to {clamped_params[0]:.4f}")
        if H < 0.01 or H > 0.5:
            is_ood = True
            clamped_params[1] = np.clip(H, 0.01, 0.5)
            logs.append(f"Rough Bergomi H={H:.4f} is OOD. Clamped to {clamped_params[1]:.4f}")
        if eta < 0.5 or eta > 4.0:
            is_ood = True
            clamped_params[2] = np.clip(eta, 0.5, 4.0)
            logs.append(f"Rough Bergomi eta={eta:.4f} is OOD. Clamped to {clamped_params[2]:.4f}")
        if rho < -0.95 or rho > 0.0:
            is_ood = True
            clamped_params[3] = np.clip(rho, -0.95, 0.0)
            logs.append(f"Rough Bergomi rho={rho:.4f} is OOD. Clamped to {clamped_params[3]:.4f}")
            
    for log_msg in logs:
        logging.warning("SR 26-2 Compliance Warning: %s", log_msg)
        
    return {
        "is_ood": is_ood,
        "clamped_params": clamped_params,
        "logs": logs
    }


def get_drift_report(model_name: str, baseline_parameters: np.ndarray) -> dict:
    """
    Generate Population Stability Index (PSI) drift report comparing current history against baseline.
    """
    model_name = model_name.lower()
    history = _HISTORICAL_CALIBRATIONS.get(model_name, [])
    if len(history) < 10:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 10 observations, current count: {len(history)}",
            "psi": {}
        }
        
    history_arr = np.vstack(history)
    baseline_arr = np.asarray(baseline_parameters)
    
    psi_dict = {}
    is_drift_detected = False
    
    # Compute PSI for each parameter
    n_params = history_arr.shape[1]
    for idx in range(n_params):
        psi = calculate_psi(baseline_arr[:, idx], history_arr[:, idx])
        psi_dict[f"param_{idx}"] = psi
        if psi > 0.25:
             is_drift_detected = True
             
    return {
        "status": "drift_detected" if is_drift_detected else "stable",
        "psi": psi_dict,
        "count": len(history)
    }
