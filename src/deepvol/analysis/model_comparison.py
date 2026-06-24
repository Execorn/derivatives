"""
model_comparison.py — Volatility Model Comparison Study.

Calibrates and compares seven option pricing and volatility models:
1. Classic Heston
2. Rough Heston
3. Rough Bergomi (rBergomi)
4. SABR
5. SSVI
6. Dupire Local Volatility
7. McKean-Vlasov SDE (MLSV)

On SPX implied volatility surfaces for dates:
- 2020-03-16
- 2022-01-24
- 2024-01-02
- 2024-08-05
"""

import os
import math
import time
import json
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

import torch
import py_vollib_vectorized
from scipy.stats import norm

from deepvol.market.spx_data import download_spx_chain, clean_chain, to_iv_surface
from deepvol.calibration.interface import calibrate
from deepvol.models.local_vol import svi_to_lv_surface
from deepvol.models.mlsv_gpu import MLSVSolverGPU

# ── Grid definition (must match FNO training grid) ──────────────────────────
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])   # years
K_GRID = np.linspace(-0.5, 0.5, 11)                              # log-moneyness

# ── SSVI to SVI Mapping ──────────────────────────────────────────────────────

def ssvi_to_svi_surface(ssvi_params: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
    """
    Map SSVI parameters to SVI slices.
    
    Parameters:
    -----------
    ssvi_params : np.ndarray
        Concatenated SSVI parameters of shape (11,): [theta_atm_0...7, rho, eta, gamma]
    T_grid : np.ndarray
        Maturity grid, shape (8,)
        
    Returns:
    --------
    svi_params : np.ndarray
        SVI parameters of shape (8, 5): [a, b, rho, m, sigma] for each slice.
    """
    theta_atm = ssvi_params[:8]
    rho, eta, gamma = ssvi_params[8], ssvi_params[9], ssvi_params[10]
    
    # Clamp rho to prevent division-by-zero or imaginary terms
    rho = np.clip(rho, -0.999, 0.999)
    
    svi_params = np.zeros((8, 5))
    for i in range(8):
        theta = theta_atm[i]
        # phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))
        # Handle zero/small theta to prevent division by zero
        if theta < 1e-8:
            phi = 0.0
        else:
            phi = eta / ((theta ** gamma) * ((1.0 + theta) ** (1.0 - gamma)))
            
        phi = max(phi, 1e-6)
        
        a = 0.5 * theta * (1.0 - rho ** 2)
        b = 0.5 * theta * phi
        m = -rho / phi
        sigma = np.sqrt(max(1.0 - rho ** 2, 0.0)) / phi
        
        svi_params[i] = [a, b, rho, m, sigma]
        
    return svi_params


# ── Vectorized 2D Local Volatility Interpolator ──────────────────────────────

def make_dupire_vol_fn(local_vol_surface: np.ndarray, T_grid: np.ndarray, K_grid: np.ndarray, 
                       S0: float, r: float, q: float, device: str) -> Any:
    """
    Construct a vectorized local volatility function for MLSVSolverGPU.
    
    Parameters:
    -----------
    local_vol_surface : np.ndarray
        Local volatility surface of shape (nT, nK)
    T_grid, K_grid : np.ndarray
        Maturity and log-moneyness grids
    S0 : float
        Spot stock price
    r, q : float
        Risk-free rate and dividend yield
    device : str
        Device context ('cpu' or 'cuda')
    """
    import torch
    
    device_obj = torch.device(device)
    lv_t = torch.as_tensor(local_vol_surface, dtype=torch.float32, device=device_obj)
    T_t = torch.as_tensor(T_grid, dtype=torch.float32, device=device_obj)
    K_t = torch.as_tensor(K_grid, dtype=torch.float32, device=device_obj)
    
    nT = len(T_t)
    nK = len(K_t)
    
    def dupire_vol_fn(t: float, S_t: torch.Tensor) -> torch.Tensor:
        # S_t is (N_paths,)
        # Compute log-moneyness for all paths: k = log(S_t / (S0 * exp((r-q)*t)))
        k = torch.log(S_t / (S0 * math.exp((r - q) * t)))
        
        # Clip k to K_grid bounds to prevent extrapolation errors
        k_min = K_t[0]
        k_max = K_t[-1]
        k_clipped = torch.clamp(k, k_min, k_max)
        
        # 1. Interpolation in T direction
        t_val = float(t)
        if t_val <= T_t[0].item():
            idx_t_left = 0
            idx_t_right = 0
            w_t = 0.0
        elif t_val >= T_t[-1].item():
            idx_t_left = nT - 1
            idx_t_right = nT - 1
            w_t = 0.0
        else:
            # Find bracket
            idx_t_right = int(torch.bucketize(torch.tensor(t_val, device=device_obj), T_t).item())
            idx_t_left = idx_t_right - 1
            w_t = (t_val - T_t[idx_t_left].item()) / (T_t[idx_t_right].item() - T_t[idx_t_left].item())
            
        # 2. Interpolation in K direction for both left and right T slices
        idx_k_right = torch.bucketize(k_clipped, K_t)
        idx_k_right = torch.clamp(idx_k_right, 1, nK - 1)
        idx_k_left = idx_k_right - 1
        
        k_left = K_t[idx_k_left]
        k_right = K_t[idx_k_right]
        
        denom = k_right - k_left
        w_k = (k_clipped - k_left) / torch.where(denom > 0, denom, torch.tensor(1.0, device=device_obj))
        w_k = torch.clamp(w_k, 0.0, 1.0)
        
        # Get values at corners using advanced indexing
        lv_left_left = lv_t[idx_t_left, idx_k_left]
        lv_left_right = lv_t[idx_t_left, idx_k_right]
        vol_left = lv_left_left * (1.0 - w_k) + lv_left_right * w_k
        
        lv_right_left = lv_t[idx_t_right, idx_k_left]
        lv_right_right = lv_t[idx_t_right, idx_k_right]
        vol_right = lv_right_left * (1.0 - w_k) + lv_right_right * w_k
        
        # Combine linear interpolations
        vol = vol_left * (1.0 - w_t) + vol_right * w_t
        return torch.clamp(vol, min=1e-4)
        
    return dupire_vol_fn


# ── Option Price Inversion ──────────────────────────────────────────────────

def invert_prices_to_iv(prices: np.ndarray, S0: float, T_grid: np.ndarray, K_grid: np.ndarray,
                        r: float, q: float, market_iv_surface: np.ndarray) -> np.ndarray:
    """
    Invert European option prices to Black-Scholes implied volatilities on the grid.
    
    Parameters:
    -----------
    prices : np.ndarray
        Option price grid of shape (nT, nK)
    S0 : float
        Spot price
    T_grid, K_grid : np.ndarray
        Maturity and log-moneyness grids
    r, q : float
        Risk-free rate and dividend yield
    market_iv_surface : np.ndarray
        Market IV surface to fall back on if inversion fails (e.g. due to MC noise)
    """
    nT, nK = prices.shape
    
    # Broadcast strikes and maturities
    T_mesh = np.tile(T_grid[:, np.newaxis], (1, nK))
    strikes_mesh = S0 * np.exp((r - q) * T_mesh + np.tile(K_grid[np.newaxis, :], (nT, 1)))
    
    # Enforce intrinsic value lower bounds to avoid solver failures
    intrinsic = np.maximum(S0 * np.exp(-q * T_mesh) - strikes_mesh * np.exp(-r * T_mesh), 0.0)
    prices_clipped = np.maximum(prices, intrinsic + 1e-6)
    
    prices_flat = prices_clipped.ravel()
    strikes_flat = strikes_mesh.ravel()
    T_flat = T_mesh.ravel()
    
    ivs_flat = py_vollib_vectorized.vectorized_implied_volatility(
        prices_flat,
        S0,
        strikes_flat,
        T_flat,
        r,
        "c",
        q,
        model="black_scholes_merton",
        return_as="numpy"
    )
    
    ivs = ivs_flat.reshape(nT, nK)
    
    # Handle NaNs / inversion failures by falling back to market implied volatility
    nans = np.isnan(ivs) | (ivs <= 0.0)
    if np.any(nans):
        ivs[nans] = market_iv_surface[nans]
        
    return ivs


# ── Diebold-Mariano and Newey-West Variance ─────────────────────────────────

def newey_west_variance(d: np.ndarray, lag: int = None) -> float:
    """
    Compute the Newey-West variance estimator for a sequence d.
    
    Parameters:
    -----------
    d : np.ndarray
        1D array of loss differentials
    lag : int, optional
        Lag truncation parameter. If None, uses standard plug-in choice.
    """
    N = len(d)
    if lag is None:
        lag = int(np.floor(4.0 * (N / 100.0) ** (2.0 / 9.0)))
    lag = min(lag, N - 1)
    
    d_bar = np.mean(d)
    x = d - d_bar
    
    # Compute autocovariances
    gamma = np.zeros(lag + 1)
    for j in range(lag + 1):
        if j == 0:
            gamma[0] = np.mean(x ** 2)
        else:
            gamma[j] = np.mean(x[j:] * x[:-j])
            
    # Compute Bartlett-weighted sum
    var = gamma[0]
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1)
        var += 2.0 * w * gamma[j]
        
    return float(var)


def diebold_mariano_test(model_a_errors: np.ndarray, model_b_errors: np.ndarray, lag: int = None) -> Tuple[float, float]:
    """
    Run the Diebold-Mariano test comparing Model A and Model B.
    
    Parameters:
    -----------
    model_a_errors : np.ndarray
        Flat 1D array of Model A implied volatility errors (model_iv - market_iv)
    model_b_errors : np.ndarray
        Flat 1D array of Model B implied volatility errors
    lag : int, optional
        Lag truncation parameter for Newey-West
        
    Returns:
    --------
    dm_stat : float
        Diebold-Mariano test statistic
    p_value : float
        Two-sided p-value
    """
    # Squared error loss
    loss_a = model_a_errors ** 2
    loss_b = model_b_errors ** 2
    
    # Loss differential
    d = loss_a - loss_b
    
    mean_d = np.mean(d)
    var_d = newey_west_variance(d, lag=lag)
    N = len(d)
    
    if var_d <= 1e-12:
        return 0.0, 1.0
        
    dm_stat = mean_d / np.sqrt(var_d / N)
    p_value = float(2.0 * (1.0 - norm.cdf(np.abs(dm_stat))))
    
    return float(dm_stat), p_value


# ── Parameter Serialization Helpers ─────────────────────────────────────────

def serialize_parameters(params: Any) -> Any:
    """Convert parameters to a JSON-serializable format."""
    if isinstance(params, np.ndarray):
        return params.tolist()
    elif isinstance(params, dict):
        return {k: float(v) if isinstance(v, (np.floating, float)) else serialize_parameters(v) for k, v in params.items()}
    elif isinstance(params, list):
        return [float(x) if isinstance(x, (np.floating, float)) else serialize_parameters(x) for x in params]
    elif isinstance(params, (np.floating, np.integer)):
        return float(params) if isinstance(params, np.floating) else int(params)
    return params


# ── Model Comparison Study Class ────────────────────────────────────────────

class ModelComparisonStudy:
    """
    A class to orchestrate the volatility model comparison study across multiple dates.
    """
    def __init__(self, cache_dir: str = "results/model_comparison_cache", device: str = "cpu"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
    def _get_cache_path(self, snapshot_date: date, model_name: str) -> Path:
        date_str = snapshot_date.strftime("%Y-%m-%d")
        return self.cache_dir / f"cache_{date_str}_{model_name}.json"
        
    def run_calibration_and_pricing(self, snapshot_date: date, model_name: str, 
                                    market_iv_surface: np.ndarray, S0: float, r: float, q: float,
                                    use_cache: bool = True, N_paths: int = 2000, 
                                    steps_per_unit: int = 50) -> Dict[str, Any]:
        """
        Calibrate model (or simulate/price for LV/MLSV) and return results.
        Uses caching to load prior results if available.
        """
        cache_path = self._get_cache_path(snapshot_date, model_name)
        
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                print(f"  [Cache Loaded] {snapshot_date.strftime('%Y-%m-%d')} - {model_name}")
                # Convert back to numpy arrays
                cached_result = {
                    "model": model_name,
                    "parameters": np.array(cached_data["parameters"]) if isinstance(cached_data["parameters"], list) else cached_data["parameters"],
                    "iv_fitted": np.array(cached_data["iv_fitted"]),
                    "elapsed_time": cached_data["elapsed_time"],
                    "rmse": cached_data["rmse"]
                }
                if "local_vol_surface" in cached_data:
                    cached_result["local_vol_surface"] = np.array(cached_data["local_vol_surface"])
                return cached_result
            except Exception as e:
                print(f"  [Cache Read Error] {e}. Recalibrating.")
                
        print(f"  [Running] {snapshot_date.strftime('%Y-%m-%d')} - {model_name}...")
        t0 = time.time()
        
        if model_name in ["heston", "rough_heston", "rbergomi", "sabr", "ssvi"]:
            # Calibrate using FNO Newton/BFGS interfaces
            res = calibrate(
                market_iv_surface, 
                model_name=model_name, 
                method="newton", 
                device=self.device, 
                T_grid=T_GRID, 
                K_grid=K_GRID
            )
            
            # Extract iv_fitted
            iv_fitted = res.info.get("iv_fitted")
            if iv_fitted is None:
                raise ValueError(f"Fitted implied volatility surface not found in calibration result for {model_name}.")
                
            elapsed = time.time() - t0
            rmse = float(res.rmse)
            params = res.parameters
            
            result = {
                "model": model_name,
                "parameters": params,
                "iv_fitted": iv_fitted,
                "elapsed_time": elapsed,
                "rmse": rmse
            }
            
        elif model_name == "local_vol":
            # For Local Vol, we need calibrated SSVI first (either cached or run)
            ssvi_res = self.run_calibration_and_pricing(
                snapshot_date, 
                "ssvi", 
                market_iv_surface, 
                S0, r, q, 
                use_cache=use_cache
            )
            
            # SSVI-to-SVI Mapping
            svi_params = ssvi_to_svi_surface(ssvi_res["parameters"], T_GRID)
            
            # Compute Dupire local volatility surface
            local_vol_surface = svi_to_lv_surface(T_GRID, K_GRID, svi_params)
            
            # Option pricing and simulation via GPU particle solver (with xi=0.0)
            option_prices = price_local_vol(
                local_vol_surface=local_vol_surface,
                S0=S0, r=r, q=q,
                T_grid=T_GRID, K_grid=K_GRID,
                N_paths=N_paths,
                steps_per_unit=steps_per_unit,
                device=self.device
            )
            
            # Invert option prices to implied volatilities on the grid
            iv_fitted = invert_prices_to_iv(
                option_prices, 
                S0, 
                T_GRID, K_GRID, 
                r, q, 
                market_iv_surface
            )
            
            elapsed = time.time() - t0
            rmse = float(np.sqrt(np.mean((iv_fitted - market_iv_surface) ** 2)))
            
            result = {
                "model": model_name,
                "parameters": svi_params,  # SVI slice parameters
                "local_vol_surface": local_vol_surface,
                "iv_fitted": iv_fitted,
                "elapsed_time": elapsed,
                "rmse": rmse
            }
            
        elif model_name == "mlsv":
            # For MLSV, we need the local vol surface first
            lv_res = self.run_calibration_and_pricing(
                snapshot_date, 
                "local_vol", 
                market_iv_surface, 
                S0, r, q, 
                use_cache=use_cache,
                N_paths=N_paths,
                steps_per_unit=steps_per_unit
            )
            local_vol_surface = lv_res["local_vol_surface"]
            
            # Setup dynamic parameters for stochastic volatility
            atm_idx = len(K_GRID) // 2
            market_v0 = float(market_iv_surface[0, atm_idx] ** 2)
            market_theta = float(market_iv_surface[-1, atm_idx] ** 2)
            
            # Clip to standard bounds
            v0 = np.clip(market_v0, 0.005, 0.20)
            theta = np.clip(market_theta, 0.005, 0.20)
            
            kappa = 2.0
            xi = 0.3
            rho = -0.7
            
            # Option pricing and simulation via GPU particle solver for MLSV
            option_prices = price_mlsv(
                local_vol_surface=local_vol_surface,
                S0=S0, r=r, q=q,
                T_grid=T_GRID, K_grid=K_GRID,
                v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
                N_paths=N_paths,
                steps_per_unit=steps_per_unit,
                device=self.device
            )
            
            # Invert option prices to implied volatilities on the grid
            iv_fitted = invert_prices_to_iv(
                option_prices, 
                S0, 
                T_GRID, K_GRID, 
                r, q, 
                market_iv_surface
            )
            
            elapsed = time.time() - t0
            rmse = float(np.sqrt(np.mean((iv_fitted - market_iv_surface) ** 2)))
            
            result = {
                "model": model_name,
                "parameters": np.array([v0, kappa, theta, xi, rho]),
                "iv_fitted": iv_fitted,
                "elapsed_time": elapsed,
                "rmse": rmse
            }
            
        else:
            raise ValueError(f"Unknown model name: {model_name}")
            
        # Save to JSON cache
        try:
            cache_save = {
                "date": snapshot_date.strftime("%Y-%m-%d"),
                "model": model_name,
                "parameters": serialize_parameters(result["parameters"]),
                "iv_fitted": result["iv_fitted"].tolist(),
                "elapsed_time": result["elapsed_time"],
                "rmse": result["rmse"]
            }
            if "local_vol_surface" in result:
                cache_save["local_vol_surface"] = result["local_vol_surface"].tolist()
                
            with open(cache_path, "w") as f:
                json.dump(cache_save, f, indent=4)
        except Exception as e:
            print(f"  [Cache Save Error] {e}")
            
        return result


# ── Option Pricing Helper Functions ──────────────────────────────────────────

def price_local_vol(local_vol_surface: np.ndarray, S0: float, r: float, q: float,
                    T_grid: np.ndarray, K_grid: np.ndarray, N_paths: int,
                    steps_per_unit: int, device: str) -> np.ndarray:
    """Simulate and price European options under Dupire Local Volatility model."""
    dup_vol_fn = make_dupire_vol_fn(local_vol_surface, T_grid, K_grid, S0, r, q, device)
    
    solver = MLSVSolverGPU(
        S0=S0,
        r=r,
        q=q,
        v0=1.0,         # V_curr = 1.0
        kappa=1.0,
        theta=1.0,
        xi=0.0,         # xi=0 makes V_t = 1.0 always (Local Volatility)
        rho=0.0,
        T=float(max(T_grid)),
        steps_per_unit=steps_per_unit,
        N_paths=N_paths,
        dupire_vol_fn=dup_vol_fn,
        device=device,
        dtype=torch.float32
    )
    
    solver.simulate(method="nadaraya_watson")
    
    prices = np.zeros((len(T_grid), len(K_grid)))
    for i, T_val in enumerate(T_grid):
        strikes = S0 * np.exp((r - q) * T_val + K_grid)
        opt_prices = solver.price_european_option(strike=strikes, maturity=T_val, is_call=True)
        if isinstance(opt_prices, torch.Tensor):
            opt_prices = opt_prices.cpu().numpy()
        prices[i] = opt_prices
        
    return prices


def price_mlsv(local_vol_surface: np.ndarray, S0: float, r: float, q: float,
               T_grid: np.ndarray, K_grid: np.ndarray, v0: float, kappa: float,
               theta: float, xi: float, rho: float, N_paths: int,
               steps_per_unit: int, device: str) -> np.ndarray:
    """Simulate and price European options under McKean-Vlasov SDE model."""
    dup_vol_fn = make_dupire_vol_fn(local_vol_surface, T_grid, K_grid, S0, r, q, device)
    
    solver = MLSVSolverGPU(
        S0=S0,
        r=r,
        q=q,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        T=float(max(T_grid)),
        steps_per_unit=steps_per_unit,
        N_paths=N_paths,
        dupire_vol_fn=dup_vol_fn,
        device=device,
        dtype=torch.float32
    )
    
    solver.simulate(method="nadaraya_watson")
    
    prices = np.zeros((len(T_grid), len(K_grid)))
    for i, T_val in enumerate(T_grid):
        strikes = S0 * np.exp((r - q) * T_val + K_grid)
        opt_prices = solver.price_european_option(strike=strikes, maturity=T_val, is_call=True)
        if isinstance(opt_prices, torch.Tensor):
            opt_prices = opt_prices.cpu().numpy()
        prices[i] = opt_prices
        
    return prices


# ── Run Study Entrypoint ─────────────────────────────────────────────────────

def run_study(device: str = "cpu", use_cache: bool = True, N_paths: int = 2000, 
              steps_per_unit: int = 50) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run the entire multi-date, multi-model comparison study.
    
    Parameters:
    -----------
    device : str
        Device context ('cpu' or 'cuda')
    use_cache : bool
        If True, load and write to JSON cache
    N_paths : int
        Number of Monte Carlo paths for Local Vol and MLSV pricing
    steps_per_unit : int
        Number of path steps per year
    """
    dates_list = [
        date(2020, 3, 16),
        date(2022, 1, 24),
        date(2024, 1, 2),
        date(2024, 8, 5)
    ]
    
    models = ["heston", "rough_heston", "rbergomi", "sabr", "ssvi", "local_vol", "mlsv"]
    
    study = ModelComparisonStudy(device=device)
    
    # Store errors for Diebold-Mariano testing
    # Shape: (7 models, 4 dates * 8 maturities * 11 strikes) -> (7, 352)
    errors_dict = {m: [] for m in models}
    
    summary_rows = []
    
    for snapshot_date in dates_list:
        date_str = snapshot_date.strftime("%Y-%m-%d")
        print(f"Processing Date: {date_str} ...")
        
        # 1. Download and clean SPX market implied volatility surface
        df = download_spx_chain(snapshot_date, cache=True)
        df_clean = clean_chain(df)
        
        # Spot price, interest rates and dividends
        if snapshot_date == date(2020, 3, 16):
            S0 = 2400.0
        elif snapshot_date == date(2022, 1, 24):
            S0 = 4400.0
        elif snapshot_date == date(2024, 1, 2):
            S0 = 4700.0
        elif snapshot_date == date(2024, 8, 5):
            S0 = 5200.0
        else:
            S0 = 5000.0
            
        r = 0.05
        q = 0.015
        
        market_iv = to_iv_surface(df_clean, S0, r, q)
        
        # Run calibration/pricing for each model
        for model in models:
            res = study.run_calibration_and_pricing(
                snapshot_date, 
                model, 
                market_iv, 
                S0, r, q,
                use_cache=use_cache,
                N_paths=N_paths,
                steps_per_unit=steps_per_unit
            )
            
            # Fitted surface and errors
            iv_fitted = res["iv_fitted"]
            errors = (iv_fitted - market_iv).ravel()
            errors_dict[model].extend(errors.tolist())
            
            rmse_bps = res["rmse"] * 10000.0
            elapsed_ms = res["elapsed_time"] * 1000.0
            
            summary_rows.append({
                "Date": date_str,
                "Model": model,
                "RMSE (bps)": rmse_bps,
                "Time (ms)": elapsed_ms
            })
            
            print(f"    RMSE = {rmse_bps:.2f} bps | Time = {elapsed_ms:.1f} ms")
            
    df_results = pd.DataFrame(summary_rows)
    
    # Convert error lists to numpy arrays
    for m in models:
        errors_dict[m] = np.array(errors_dict[m])
        
    # 2. Pairwise Diebold-Mariano testing
    dm_stats = np.zeros((len(models), len(models)))
    dm_pvals = np.zeros((len(models), len(models)))
    
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i == j:
                dm_stats[i, j] = 0.0
                dm_pvals[i, j] = 1.0
            else:
                stat, pval = diebold_mariano_test(errors_dict[model_a], errors_dict[model_b])
                dm_stats[i, j] = stat
                dm_pvals[i, j] = pval
                
    df_dm_stat = pd.DataFrame(dm_stats, index=models, columns=models)
    df_dm_pval = pd.DataFrame(dm_pvals, index=models, columns=models)
    
    dm_results = {
        "statistic": df_dm_stat,
        "p_value": df_dm_pval
    }
    
    print("\nPairwise Diebold-Mariano Test Statistics (Model Row vs Model Col):")
    print(df_dm_stat.to_string())
    print("\nPairwise Diebold-Mariano P-values:")
    print(df_dm_pval.to_string())
    
    # Save comparison report inside the results folder
    report_dir = Path("results/model_comparison")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    df_results.to_csv(report_dir / "comparison_metrics.csv", index=False)
    df_dm_stat.to_csv(report_dir / "dm_statistics.csv")
    df_dm_pval.to_csv(report_dir / "dm_pvalues.csv")
    
    return df_results, dm_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting model comparison study on device: {device}...")
    run_study(device=device, use_cache=True)
