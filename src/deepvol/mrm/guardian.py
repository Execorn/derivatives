"""
guardian.py — Model risk guardian that monitors parameters and intercepts pricing calls to trigger fallbacks.
"""

import logging
import numpy as np
import torch
import scipy.stats as stats
from typing import Union, Dict, Any, List, Optional
from deepvol.utils.strikes import resolve_strikes
from deepvol.mrm.arbitrage import check_arbitrage
from deepvol.models.heston import HestonEngine
from deepvol.models.mlsv_gpu import MLSVEngine

class ModelRiskGuardian:
    """
    ModelRiskGuardian monitors volatility model calibration parameters and surface behavior
    for anomalies (such as parameter pinning or residual limits) and static arbitrage violations.
    It intercepts pricing requests and automatically routes them to robust fallback solvers:
    - Analytical Fourier-COS Solver (HestonEngine)
    - McKean-Vlasov SDE Particle Solver (MLSVEngine)
    """
    def __init__(
        self,
        vol_of_vol_limit: float = 0.99,
        hurst_limit: float = 0.015,
        residual_limit: float = 0.0150,  # 150 bps
        safe_heston_params: Optional[Dict[str, float]] = None,
        safe_mlsv_params: Optional[Dict[str, float]] = None
    ):
        self.vol_of_vol_limit = vol_of_vol_limit
        self.hurst_limit = hurst_limit
        self.residual_limit = residual_limit
        
        # Default safe recovery parameters if calibrated parameters are unusable
        self.safe_heston_params = safe_heston_params or {
            "kappa": 2.0,
            "theta": 0.04,
            "sigma": 0.3,
            "rho": -0.7,
            "v0": 0.04
        }
        self.safe_mlsv_params = safe_mlsv_params or {
            "kappa": 2.0,
            "theta": 0.04,
            "epsilon": 0.3,
            "rho": -0.7
        }

    def check_parameters(self, model_name: str, parameters: Any, rmse: float = 0.0) -> dict:
        """
        Check calibrated parameters for boundary pinning anomalies and high residuals.
        """
        anomalies = []
        params_dict = {}
        m_name = model_name.lower()
        
        # 1. Parse parameter inputs to dictionary
        if isinstance(parameters, dict):
            params_dict = parameters
        elif isinstance(parameters, (list, np.ndarray, torch.Tensor)):
            if isinstance(parameters, torch.Tensor):
                param_vals = parameters.detach().cpu().numpy().tolist()
            else:
                param_vals = list(parameters)
                
            if m_name in ("heston", "rough_heston"):
                if len(param_vals) >= 5:
                    params_dict = {
                        "kappa": param_vals[0],
                        "theta": param_vals[1],
                        "sigma": param_vals[2],
                        "rho": param_vals[3],
                        "v0": param_vals[4]
                    }
            elif m_name in ("rbergomi", "rough_bergomi"):
                if len(param_vals) >= 4:
                    params_dict = {
                        "v0": param_vals[0],
                        "H": param_vals[1],
                        "eta": param_vals[2],
                        "rho": param_vals[3]
                    }
            elif m_name == "sabr":
                if len(param_vals) >= 4:
                    params_dict = {
                        "alpha": param_vals[0],
                        "beta": param_vals[1],
                        "rho": param_vals[2],
                        "nu": param_vals[3]
                    }
                    
        # 2. Check for boundary pinning anomalies
        if m_name in ("heston", "rough_heston"):
            sigma = params_dict.get("sigma", 0.0)
            if sigma >= self.vol_of_vol_limit:
                anomalies.append(
                    f"Heston vol-of-vol pinned at boundary: sigma={sigma:.4f} (limit={self.vol_of_vol_limit})"
                )
                
        elif m_name in ("rbergomi", "rough_bergomi"):
            H = params_dict.get("H", 0.5)
            eta = params_dict.get("eta", 0.0)
            if H <= self.hurst_limit:
                anomalies.append(
                    f"Rough Bergomi Hurst exponent pinned at boundary: H={H:.4f} (limit={self.hurst_limit})"
                )
            if eta >= self.vol_of_vol_limit:
                anomalies.append(
                    f"Rough Bergomi vol-of-vol pinned at boundary: eta={eta:.4f} (limit={self.vol_of_vol_limit})"
                )
                
        elif m_name == "sabr":
            nu = params_dict.get("nu", 0.0)
            if nu >= self.vol_of_vol_limit:
                anomalies.append(
                    f"SABR vol-of-vol pinned at boundary: nu={nu:.4f} (limit={self.vol_of_vol_limit})"
                )
                
        # 3. Check for residual anomalies
        if rmse > self.residual_limit:
            anomalies.append(
                f"Calibration residual exceeds safety limit: rmse={rmse * 10000:.1f} bps (limit={self.residual_limit * 10000:.1f} bps)"
            )
            
        return {
            "anomaly_detected": len(anomalies) > 0,
            "anomalies": anomalies,
            "params_dict": params_dict
        }

    def check_surface(self, iv_surface: np.ndarray, T_grid: np.ndarray, K_grid: np.ndarray, S: float = 1.0) -> dict:
        """
        Check fitted or predicted implied volatility surface for calendar/butterfly arbitrage.
        """
        arb_res = check_arbitrage(iv_surface, K_grid, T_grid, S=S)
        
        anomalies = []
        if arb_res["calendar"]["has_arbitrage"]:
            anomalies.append("Static calendar spread arbitrage detected on surface.")
        if arb_res["butterfly_durrleman"]["has_arbitrage"] or arb_res["butterfly_price"]["has_arbitrage"]:
            anomalies.append("Static butterfly spread arbitrage detected on surface.")
            
        return {
            "anomaly_detected": len(anomalies) > 0,
            "anomalies": anomalies,
            "arb_details": arb_res
        }

    def price_or_fallback(
        self,
        model_name: str,
        parameters: Union[dict, np.ndarray, torch.Tensor],
        spot: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_iv_surface: Optional[np.ndarray] = None,
        calibration_rmse: float = 0.0,
        fallback_route: str = "fourier",  # 'fourier' or 'particle'
        **kwargs
    ) -> dict:
        """
        Inspect parameters and predicted surface for model risk anomalies and arbitrage.
        If clean: price options normally.
        If anomalous: trigger fallback routing.
        
        Returns:
            dict: pricing results and diagnostic logs.
        """
        # 1. Run parameter checks
        param_check = self.check_parameters(model_name, parameters, calibration_rmse)
        anomalies = list(param_check["anomalies"])
        params_dict = param_check["params_dict"]
        
        # 2. Try generating initial surface to check for arbitrage
        iv_surface = None
        has_surface_arb = False
        
        try:
            if model_name.lower() in ("heston", "rough_heston") and len(params_dict) >= 5:
                iv_surface = HestonEngine().price_surface(params_dict, maturities, strikes, S0=spot)
            elif market_iv_surface is not None:
                iv_surface = market_iv_surface
                
            if iv_surface is not None:
                # Calculate RMSE against market if market surface was provided and not checked yet
                if market_iv_surface is not None and calibration_rmse == 0.0:
                    valid_mask = ~np.isnan(iv_surface) & ~np.isnan(market_iv_surface)
                    if np.any(valid_mask):
                        calc_rmse = np.sqrt(np.mean((iv_surface[valid_mask] - market_iv_surface[valid_mask])**2))
                        if calc_rmse > self.residual_limit and not any("Calibration residual" in a for a in anomalies):
                            anomalies.append(
                                f"Calculated surface residual exceeds safety limit: rmse={calc_rmse * 10000:.1f} bps (limit={self.residual_limit * 10000:.1f} bps)"
                            )
                
                # Check for static arbitrage
                surf_check = self.check_surface(iv_surface, maturities, strikes, S=spot)
                if surf_check["anomaly_detected"]:
                    anomalies.extend(surf_check["anomalies"])
                    has_surface_arb = True
        except Exception as e:
            anomalies.append(f"Failed to generate check surface: {str(e)}")
            
        anomaly_detected = len(anomalies) > 0
        
        # 3. Handle Fallback Routing if anomalies found
        if anomaly_detected:
            fallback_triggered = True
            
            if fallback_route == "fourier":
                # Route to exact analytical Fourier-COS solver (HestonEngine)
                # If parameters were pinned/anomalous, use safe Heston parameters
                if param_check["anomaly_detected"]:
                    active_params = self.safe_heston_params
                    status = "fallback_fourier_safe_params"
                else:
                    active_params = params_dict
                    status = "fallback_fourier_calibrated_params"
                    
                engine = HestonEngine()
                fallback_ivs = engine.price_surface(active_params, maturities, strikes, S0=spot)
                
                # Convert implied volatilities to call prices
                T_m = maturities[:, None]
                K_m = resolve_strikes(strikes, spot)[None, :]
                vol_std = fallback_ivs * np.sqrt(T_m)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    d1 = (np.log(spot / K_m) + 0.5 * vol_std**2) / np.clip(vol_std, 1e-9, None)
                    d2 = d1 - vol_std
                    
                fallback_prices = spot * stats.norm.cdf(d1) - K_m * stats.norm.cdf(d2)
                fallback_prices = np.where(vol_std <= 1e-8, np.maximum(spot - K_m, 0.0), fallback_prices)
                
                return {
                    "prices": fallback_prices,
                    "ivs": fallback_ivs,
                    "guardian_status": status,
                    "anomalies": anomalies,
                    "fallback_triggered": True,
                    "active_parameters": active_params
                }
                
            elif fallback_route == "particle":
                # Route to McKean-Vlasov SDE Particle Solver on GPU/CPU
                if param_check["anomaly_detected"]:
                    active_params = self.safe_mlsv_params
                    status = "fallback_particle_safe_params"
                else:
                    # Map Heston or other params to MLSV if needed
                    active_params = {
                        "kappa": params_dict.get("kappa", self.safe_mlsv_params["kappa"]),
                        "theta": params_dict.get("theta", self.safe_mlsv_params["theta"]),
                        "epsilon": params_dict.get("sigma", self.safe_mlsv_params["epsilon"]),
                        "rho": params_dict.get("rho", self.safe_mlsv_params["rho"])
                    }
                    status = "fallback_particle_calibrated_params"
                    
                # Setup MLSV engine
                engine = MLSVEngine(
                    kappa=active_params["kappa"],
                    theta=active_params["theta"],
                    epsilon=active_params["epsilon"],
                    rho=active_params["rho"]
                )
                
                nT = len(maturities)
                nK = len(strikes)
                fallback_prices = np.zeros((nT, nK))
                fallback_ivs = np.zeros((nT, nK))
                
                # F-06: PERFORMANCE WARNING — The particle solver fallback prices
                # each option individually in a nested O(nT × nK) loop, which is
                # orders of magnitude slower than the vectorized Fourier path.
                # For a typical 8×11 grid this is ~88 serial MC simulations.
                logging.warning(
                    "MRM Guardian: Particle solver fallback triggered for %d × %d = %d "
                    "grid points. Expect significant latency increase vs. Fourier path.",
                    nT, nK, nT * nK
                )
                for i, T in enumerate(maturities):
                    for j, K_val in enumerate(strikes):
                        K_abs = spot * np.exp(K_val) if (K_val < 0 or K_val < 5.0) else K_val
                        vol_init = np.sqrt(active_params["theta"])
                        
                        price = engine.price_option(
                            spot=spot,
                            strike=K_abs,
                            maturity=T,
                            vol=vol_init,
                            is_call=True
                        )
                        fallback_prices[i, j] = price
                        
                        # Invert Black-Scholes to find IV
                        # Helper Brent solver
                        try:
                            iv = price_to_iv(price, spot, K_abs, T)
                        except Exception:
                            iv = vol_init
                        fallback_ivs[i, j] = iv
                        
                return {
                    "prices": fallback_prices,
                    "ivs": fallback_ivs,
                    "guardian_status": status,
                    "anomalies": anomalies,
                    "fallback_triggered": True,
                    "active_parameters": active_params
                }
            else:
                raise ValueError(f"Unknown fallback_route: {fallback_route}")
        else:
            # Clean case: price normally
            # For simplicity, we price with HestonEngine if it is a Heston model
            if model_name.lower() in ("heston", "rough_heston"):
                engine = HestonEngine()
                ivs = engine.price_surface(params_dict, maturities, strikes, S0=spot)
                
                T_m = maturities[:, None]
                K_m = resolve_strikes(strikes, spot)[None, :]
                vol_std = ivs * np.sqrt(T_m)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    d1 = (np.log(spot / K_m) + 0.5 * vol_std**2) / np.clip(vol_std, 1e-9, None)
                    d2 = d1 - vol_std
                    
                prices = spot * stats.norm.cdf(d1) - K_m * stats.norm.cdf(d2)
                prices = np.where(vol_std <= 1e-8, np.maximum(spot - K_m, 0.0), prices)
                
                return {
                    "prices": prices,
                    "ivs": ivs,
                    "guardian_status": "passed",
                    "anomalies": [],
                    "fallback_triggered": False,
                    "active_parameters": params_dict
                }
            else:
                # Return empty/standard output for other models
                return {
                    "prices": np.zeros((len(maturities), len(strikes))),
                    "ivs": np.zeros((len(maturities), len(strikes))),
                    "guardian_status": "passed",
                    "anomalies": [],
                    "fallback_triggered": False,
                    "active_parameters": params_dict
                }

def price_to_iv(price: float, S: float, K: float, T: float) -> float:
    """
    Invert Black-Scholes call option price to implied volatility.
    """
    import scipy.optimize as optimize
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-15:
        return 1e-4
    if price >= S - 1e-15:
        return 5.0
    
    def bs_call(sigma):
        if T <= 0.0 or sigma <= 0.0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)
        
    try:
        return optimize.brentq(
            lambda sigma: bs_call(sigma) - price,
            1e-4, 5.0, xtol=1e-12
        )
    except ValueError:
        return 1e-4
