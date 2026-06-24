"""
models.py — Volatility model pricing and calibration wrappers.
Integrates SABR, Heston, Rough Bergomi, Neural SDE, MLSV, and Schwartz-Smith.
Supports FNO surrogate evaluation, direct simulation, and custom optimization.
"""

import time
import os
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st
from scipy.stats import norm
import scipy.optimize

from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.calibration.calibrate_bfgs import _load_normalizers, _make_spatial_input, _fno_predict_real_iv
from deepvol.calibration.calibrate_newton import (
    calibrate_heston,
    calibrate_sabr,
    calibrate_rbergomi,
)
from deepvol.models.heston import heston_iv_surface
from deepvol.models.sabr import sabr_iv_surface
from deepvol.models.mlsv_gpu import MLSVSolverGPU
from deepvol.models.schwartz_smith import schwartz_smith_price_black76
from deepvol.models.neural_sde import NeuralSDE, NeuralSDEPricer, compute_calibration_loss

# Grid constants
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
STRIKES = np.linspace(-0.5, 0.5, 11)  # Log-moneyness

# ─── Black-Scholes & Implied Vol Inversion ───────────────────────────────────

def bs_call_price(S0, K, T, sigma, r=0.0, q=0.0):
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-8)
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_vega(S0, K, T, sigma, r=0.0, q=0.0):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-8)
    return S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def invert_implied_vol(price, S0, K, T, r=0.0, q=0.0, max_iter=100, tol=1e-6):
    intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-6:
        return 1e-4
    sigma = 0.3
    for _ in range(max_iter):
        p = bs_call_price(S0, K, T, sigma, r, q)
        diff = p - price
        if abs(diff) < tol:
            return float(sigma)
        vega = bs_vega(S0, K, T, sigma, r, q)
        if vega < 1e-6:
            sigma = sigma - 0.5 * diff / S0
        else:
            sigma = sigma - diff / vega
        sigma = np.clip(sigma, 1e-4, 5.0)
    return float(sigma)

# ─── FNO Model Loader ────────────────────────────────────────────────────────

@st.cache_resource
def load_fno_model(model_name: str):
    """Load and cache the appropriate FNO surrogate and normalizers on CPU."""
    if model_name == "Classic Heston":
        model = MirrorPaddedFNO2d(param_dim=5)
        path = "artifacts/weights/fno_heston_final_prod.pth"
        norm_key = "heston"
    elif model_name == "SABR":
        model = MirrorPaddedFNO2d(param_dim=3)
        path = "artifacts/weights/fno_sabr_final_prod.pth"
        norm_key = "sabr"
    elif model_name == "Rough Bergomi":
        model = MirrorPaddedFNO2d(param_dim=4)
        path = "artifacts/weights/fno_rbergomi_final_prod.pth"
        norm_key = "rbergomi"
    elif model_name == "Rough Heston":
        model = MirrorPaddedFNO2d(param_dim=6)
        path = "artifacts/weights/fno_v2_final_prod.pth"
        norm_key = "v2"
    else:
        return None

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    _load_normalizers(norm_key)
    return model

# ─── Model Surface Rebuilders ───────────────────────────────────────────────

def reconstruct_sabr_surface(alpha, rho, nu, F0=1.0):
    """Direct pricing of SABR surface."""
    return sabr_iv_surface(
        F=F0, T_grid=MATURITIES, k_grid=STRIKES,
        alpha=alpha, beta=1.0, rho=rho, nu=nu,
        iv_type="lognormal"
    )

def reconstruct_heston_surface(kappa, theta, sigma, rho, v0):
    """Direct pricing of Classic Heston surface."""
    if kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0 or v0 <= 0.0:
        raise ValueError("Heston parameters kappa, theta, sigma, v0 must be strictly positive.")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("Heston correlation rho must be in [-1, 1].")
    p_dict = {'kappa': kappa, 'theta': theta, 'sigma': sigma, 'rho': rho, 'v0': v0}
    target_iv = heston_iv_surface(p_dict, MATURITIES, STRIKES)
    # Fill NaNs
    for t_idx in range(len(MATURITIES)):
        slice_t = target_iv[t_idx, :]
        valid_vals = slice_t[np.isfinite(slice_t)]
        med = np.median(valid_vals) if len(valid_vals) > 0 else 0.3
        slice_t[~np.isfinite(slice_t)] = med
        target_iv[t_idx, :] = slice_t
    return target_iv

def reconstruct_rbergomi_surface(model, v0, H, eta, rho):
    """Evaluate Rough Bergomi surface using FNO."""
    device = torch.device("cpu")
    spatial = _make_spatial_input(MATURITIES, STRIKES, device=device)
    params_t = torch.tensor([v0, H, eta, rho], dtype=torch.float32).unsqueeze(0)
    _load_normalizers("rbergomi")
    with torch.no_grad():
        iv = _fno_predict_real_iv(model, params_t, spatial).numpy()
    return iv

def reconstruct_rheston_surface(model, kappa, theta, sigma, rho, v0, H):
    """Evaluate Rough Heston surface using FNO."""
    device = torch.device("cpu")
    spatial = _make_spatial_input(MATURITIES, STRIKES, device=device)
    params_t = torch.tensor([kappa, theta, sigma, rho, v0, H], dtype=torch.float32).unsqueeze(0)
    _load_normalizers("v2")
    with torch.no_grad():
        iv = _fno_predict_real_iv(model, params_t, spatial).numpy()
    return iv

def reconstruct_schwartz_smith_surface(S0, r, chi_t, xi_t, kappa, sigma_chi, rho, sigma_xi, mu_star, lambda_chi):
    """Price options under Schwartz-Smith and invert to IV surface."""
    iv_surface = np.zeros((len(MATURITIES), len(STRIKES)))
    for i, T in enumerate(MATURITIES):
        for j, k in enumerate(STRIKES):
            K = S0 * np.exp(k)
            price = schwartz_smith_price_black76(
                t=0.0, T_opt=T, T_fut=T, K=K, r=r,
                chi_t=chi_t, xi_t=xi_t, kappa=kappa, sigma_chi=sigma_chi,
                rho=rho, sigma_xi=sigma_xi, mu_star=mu_star, lambda_chi=lambda_chi,
                option_type="C"
            )
            iv_surface[i, j] = invert_implied_vol(price, S0, K, T, r, q=0.0)
    return iv_surface

def reconstruct_mlsv_surface(S0, r, q, v0, kappa, theta, xi, rho, method="nadaraya_watson", N_paths=1000):
    """Simulate particle paths and reconstruct MLSV IV surface."""
    T_max = float(np.max(MATURITIES))
    dup_vol_fn = lambda t, s: torch.full_like(s, 0.2)
    
    solver = MLSVSolverGPU(
        S0=S0, r=r, q=q, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        T=T_max, steps_per_unit=50, N_paths=N_paths, dupire_vol_fn=dup_vol_fn, device="cpu"
    )
    solver.simulate(method=method)
    
    iv_surface = np.zeros((len(MATURITIES), len(STRIKES)))
    for i, T in enumerate(MATURITIES):
        abs_strikes = S0 * np.exp(STRIKES)
        prices = solver.price_european_option(strike=abs_strikes, maturity=T, is_call=True)
        for j in range(len(STRIKES)):
            price_val = float(prices[j].item())
            iv_surface[i, j] = invert_implied_vol(price_val, S0, abs_strikes[j], T, r, q)
            
    return iv_surface

# ─── Local Calibration Routines ──────────────────────────────────────────────

def calibrate_neural_sde_local(market_iv, S0=100.0, r=0.05, q=0.015, epochs=30, N_paths=1024):
    """Run local Neural SDE calibration using PyTorch on CPU."""
    device = torch.device("cpu")
    
    # Pre-build target option prices
    T_mkt_list = []
    K_mkt_list = []
    prices_mkt_list = []
    
    for i, t in enumerate(MATURITIES):
        for j, k in enumerate(STRIKES):
            strike_val = S0 * np.exp(k)
            iv_val = market_iv[i][j]
            T_mkt_list.append(t)
            K_mkt_list.append(strike_val)
            
            # Map target IV to option price
            price_val = bs_call_price(S0, strike_val, t, iv_val, r, q)
            prices_mkt_list.append(price_val)
            
    strikes_mkt = torch.tensor(K_mkt_list, dtype=torch.float32, device=device)
    prices_mkt = torch.tensor(prices_mkt_list, dtype=torch.float32, device=device)
    maturities_mkt = torch.tensor(T_mkt_list, dtype=torch.float32, device=device)
    
    # Model
    epsilon = 1e-4
    sde = NeuralSDE(r=r, q=q, rho_init=-0.7, hidden_dim=16, epsilon=epsilon)
    pricer = NeuralSDEPricer(sde, v0_init=0.04)
    pricer.to(device)
    
    optimizer = torch.optim.Adam(pricer.parameters(), lr=0.02)
    loss_history = []
    
    t0 = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred, ys = pricer.price_options(
            S0=S0, strikes=strikes_mkt, maturities=maturities_mkt,
            N_paths=N_paths, dt=0.01, method="euler"
        )
        loss_dict = compute_calibration_loss(
            model_prices=pred, market_prices=prices_mkt,
            vegas=torch.ones_like(prices_mkt), ys=ys,
            lambda_bound=0.01, epsilon=epsilon
        )
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))
        
    elapsed = time.time() - t0
    
    # Calculate final fitted IV surface
    with torch.no_grad():
        pred_final, _ = pricer.price_options(
            S0=S0, strikes=strikes_mkt, maturities=maturities_mkt,
            N_paths=N_paths, dt=0.01, method="euler"
        )
        
    pred_final_np = pred_final.cpu().numpy()
    fitted_iv = np.zeros((8, 11))
    idx = 0
    for i, t in enumerate(MATURITIES):
        for j, k in enumerate(STRIKES):
            strike_val = S0 * np.exp(k)
            price_val = float(pred_final_np[idx])
            fitted_iv[i, j] = invert_implied_vol(price_val, S0, strike_val, t, r, q)
            idx += 1
            
    return {
        "v0": float(pricer.v0.item()),
        "rho": float(sde.rho.item()),
        "loss_history": loss_history,
        "elapsed_ms": elapsed * 1000.0,
        "fitted_iv": fitted_iv,
        "final_rmse": float(np.sqrt(np.mean((pred_final_np - prices_mkt.cpu().numpy())**2)))
    }
