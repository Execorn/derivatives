"""
hedging_backtest.py — Benchmark comparing Greeks-based FNO Delta/Delta-Vega hedging,
flat-vol Black-Scholes, and Deep Hedging policies under transaction costs.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.path.join(project_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.calibration.calibrate_newton import calibrate_newton
from deepvol.calibration.calibrate_bfgs import _load_normalizers, _make_spatial_input, _fno_predict_real_iv
from deepvol.hedging.deep_hedging import HedgingPolicy, DeepHedgingEnv, train_deep_hedger

# ─── Grids ─────────────────────────────────────────────────────────────────────
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
K_GRID = np.linspace(-0.5, 0.5, 11, dtype=np.float32)

# ─── Black-Scholes formulas ───────────────────────────────────────────────────

def bs_call_price(S, K, T, sigma, r=0.0):
    """Black-Scholes call price. Handles edge cases robustly."""
    S = np.maximum(S, 1e-8)
    if isinstance(T, (int, float)):
        T_arr = np.full_like(S, T)
    else:
        T_arr = np.asarray(T)
        
    sigma_arr = np.asarray(sigma)
    cond = (T_arr <= 1e-5) | (sigma_arr <= 1e-5)
    
    denom = np.maximum(sigma_arr * np.sqrt(T_arr), 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_arr**2) * T_arr) / denom
    d2 = d1 - denom
    call = S * norm.cdf(d1) - K * np.exp(-r * T_arr) * norm.cdf(d2)
    return np.where(cond, np.maximum(S - K, 0.0), call)

def bs_delta(S, K, T, sigma, r=0.0):
    """BS Delta for European Call."""
    S = np.maximum(S, 1e-8)
    if isinstance(T, (int, float)):
        T_arr = np.full_like(S, T)
    else:
        T_arr = np.asarray(T)
        
    sigma_arr = np.asarray(sigma)
    cond = (T_arr <= 1e-5) | (sigma_arr <= 1e-5)
    
    denom = np.maximum(sigma_arr * np.sqrt(T_arr), 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_arr**2) * T_arr) / denom
    delta = norm.cdf(d1)
    return np.where(cond, np.where(S > K, 1.0, 0.0), delta)

def bs_vega(S, K, T, sigma, r=0.0):
    """BS Vega."""
    S = np.maximum(S, 1e-8)
    if isinstance(T, (int, float)):
        T_arr = np.full_like(S, T)
    else:
        T_arr = np.asarray(T)
        
    sigma_arr = np.asarray(sigma)
    cond = (T_arr <= 1e-5) | (sigma_arr <= 1e-5)
    
    denom = np.maximum(sigma_arr * np.sqrt(T_arr), 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_arr**2) * T_arr) / denom
    vega = S * np.sqrt(T_arr) * norm.pdf(d1)
    return np.where(cond, 0.0, vega)


# ─── Bilinear Interpolation Helper ────────────────────────────────────────────

def interpolate_bilinear_np(T_grid, K_grid, iv_surface, T, k):
    """Bilinear interpolation in NumPy to map continuous query point to grid."""
    nT = len(T_grid)
    nK = len(K_grid)
    
    # Clip query points to the grid bounds
    T_clip = np.clip(T, T_grid[0], T_grid[-1])
    k_clip = np.clip(k, K_grid[0], K_grid[-1])
    
    # Find indices
    t_idx = np.searchsorted(T_grid, T_clip) - 1
    t_idx = np.clip(t_idx, 0, nT - 2)
    
    k_idx = np.searchsorted(K_grid, k_clip) - 1
    k_idx = np.clip(k_idx, 0, nK - 2)
    
    t0, t1 = T_grid[t_idx], T_grid[t_idx + 1]
    k0, k1 = K_grid[k_idx], K_grid[k_idx + 1]
    
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
          
    return float(val)

# ─── Heston path simulation ───────────────────────────────────────────────────

def simulate_heston_paths(S0, v0, kappa, theta, sigma, rho, T, N_steps, N_paths, rng):
    """Simulates spot price S_t and variance V_t using Full Truncation Euler scheme."""
    dt = T / N_steps
    S = np.zeros((N_paths, N_steps + 1))
    V = np.zeros((N_paths, N_steps + 1))
    S[:, 0] = S0
    V[:, 0] = v0
    
    for i in range(N_steps):
        Z_V = rng.normal(size=N_paths)
        Z_S = rng.normal(size=N_paths)
        
        dW_V = Z_V * np.sqrt(dt)
        dW_S = (rho * Z_V + np.sqrt(1.0 - rho**2) * Z_S) * np.sqrt(dt)
        
        V_prev = V[:, i]
        V_pos = np.maximum(V_prev, 0.0)
        
        V[:, i+1] = V_prev + kappa * (theta - V_pos) * dt + sigma * np.sqrt(V_pos) * dW_V
        S_prev = S[:, i]
        S[:, i+1] = S_prev * np.exp((-0.5 * V_pos) * dt + np.sqrt(V_pos) * dW_S)
        
    return S, V

# ─── Evaluator functions ──────────────────────────────────────────────────────

def evaluate_bs_delta(S, K_V, T_V, dt, sigma_V_init, V0_price):
    M, N_plus_1 = S.shape
    N_steps = N_plus_1 - 1
    
    cash = np.zeros(M)
    delta_prev = np.zeros(M)
    tc_total = np.zeros(M)
    
    # t_0
    delta_0 = bs_delta(S[:, 0], K_V, T_V, sigma_V_init)
    cash = V0_price - delta_0 * S[:, 0]
    tc = 0.0001 * S[:, 0] * np.abs(delta_0)
    cash -= tc
    tc_total += tc
    delta_prev = delta_0
    
    # t_i
    for i in range(1, N_steps):
        T_rem = T_V - i * dt
        delta_i = bs_delta(S[:, i], K_V, T_rem, sigma_V_init)
        tc = 0.0001 * S[:, i] * np.abs(delta_i - delta_prev)
        cash -= (delta_i - delta_prev) * S[:, i] + tc
        tc_total += tc
        delta_prev = delta_i
        
    # t_N
    tc = 0.0001 * S[:, -1] * np.abs(delta_prev)
    cash += delta_prev * S[:, -1] - tc
    tc_total += tc
    
    payoff = np.maximum(S[:, -1] - K_V, 0.0)
    return cash - payoff, tc_total

def evaluate_fno_delta(S, K_V, T_V, dt, sigma_V_cal, V0_price):
    M, N_plus_1 = S.shape
    N_steps = N_plus_1 - 1
    
    cash = np.zeros(M)
    delta_prev = np.zeros(M)
    tc_total = np.zeros(M)
    
    # t_0
    delta_0 = bs_delta(S[:, 0], K_V, T_V, sigma_V_cal[:, 0])
    cash = V0_price - delta_0 * S[:, 0]
    tc = 0.0001 * S[:, 0] * np.abs(delta_0)
    cash -= tc
    tc_total += tc
    delta_prev = delta_0
    
    # t_i
    for i in range(1, N_steps):
        T_rem = T_V - i * dt
        delta_i = bs_delta(S[:, i], K_V, T_rem, sigma_V_cal[:, i])
        tc = 0.0001 * S[:, i] * np.abs(delta_i - delta_prev)
        cash -= (delta_i - delta_prev) * S[:, i] + tc
        tc_total += tc
        delta_prev = delta_i
        
    # t_N
    tc = 0.0001 * S[:, -1] * np.abs(delta_prev)
    cash += delta_prev * S[:, -1] - tc
    tc_total += tc
    
    payoff = np.maximum(S[:, -1] - K_V, 0.0)
    return cash - payoff, tc_total

def evaluate_bs_delta_vega(S, A_BS, K_V, K_A, T_V, T_A, dt, sigma_V_init, sigma_A_init, V0_price):
    M, N_plus_1 = S.shape
    N_steps = N_plus_1 - 1
    
    cash = np.zeros(M)
    delta_S_prev = np.zeros(M)
    delta_A_prev = np.zeros(M)
    tc_total = np.zeros(M)
    
    # t_0
    A0 = A_BS[:, 0]
    
    v_V = bs_vega(S[:, 0], K_V, T_V, sigma_V_init)
    v_A = bs_vega(S[:, 0], K_A, T_A, sigma_A_init)
    d_V = bs_delta(S[:, 0], K_V, T_V, sigma_V_init)
    d_A = bs_delta(S[:, 0], K_A, T_A, sigma_A_init)
    
    delta_A = np.clip(v_V / (v_A + 1e-8), -2.0, 2.0)
    delta_S = np.clip(d_V - delta_A * d_A, -2.0, 2.0)
    
    cash = V0_price - delta_S * S[:, 0] - delta_A * A0
    tc = 0.0001 * S[:, 0] * np.abs(delta_S) + 0.0005 * A0 * np.abs(delta_A)
    cash -= tc
    tc_total += tc
    delta_S_prev = delta_S
    delta_A_prev = delta_A
    
    # t_i
    for i in range(1, N_steps):
        T_rem_V = T_V - i * dt
        T_rem_A = T_A - i * dt
        S_t = S[:, i]
        A_t = A_BS[:, i]
        
        v_V = bs_vega(S_t, K_V, T_rem_V, sigma_V_init)
        v_A = bs_vega(S_t, K_A, T_rem_A, sigma_A_init)
        d_V = bs_delta(S_t, K_V, T_rem_V, sigma_V_init)
        d_A = bs_delta(S_t, K_A, T_rem_A, sigma_A_init)
        
        delta_A = np.clip(v_V / (v_A + 1e-8), -2.0, 2.0)
        delta_S = np.clip(d_V - delta_A * d_A, -2.0, 2.0)
        
        tc = 0.0001 * S_t * np.abs(delta_S - delta_S_prev) + 0.0005 * A_t * np.abs(delta_A - delta_A_prev)
        cash -= (delta_S - delta_S_prev) * S_t + (delta_A - delta_A_prev) * A_t + tc
        tc_total += tc
        
        delta_S_prev = delta_S
        delta_A_prev = delta_A
        
    # t_N
    S_t = S[:, -1]
    A_t = A_BS[:, -1]
    
    tc = 0.0001 * S_t * np.abs(delta_S_prev) + 0.0005 * A_t * np.abs(delta_A_prev)
    cash += delta_S_prev * S_t + delta_A_prev * A_t - tc
    tc_total += tc
    
    payoff = np.maximum(S_t - K_V, 0.0)
    return cash - payoff, tc_total

def evaluate_fno_delta_vega(S, A_cal, K_V, K_A, T_V, T_A, dt, sigma_V_cal, sigma_A_cal, V0_price):
    M, N_plus_1 = S.shape
    N_steps = N_plus_1 - 1
    
    cash = np.zeros(M)
    delta_S_prev = np.zeros(M)
    delta_A_prev = np.zeros(M)
    tc_total = np.zeros(M)
    
    # t_0
    A0 = A_cal[:, 0]
    
    v_V = bs_vega(S[:, 0], K_V, T_V, sigma_V_cal[:, 0])
    v_A = bs_vega(S[:, 0], K_A, T_A, sigma_A_cal[:, 0])
    d_V = bs_delta(S[:, 0], K_V, T_V, sigma_V_cal[:, 0])
    d_A = bs_delta(S[:, 0], K_A, T_A, sigma_A_cal[:, 0])
    
    delta_A = np.clip(v_V / (v_A + 1e-8), -2.0, 2.0)
    delta_S = np.clip(d_V - delta_A * d_A, -2.0, 2.0)
    
    cash = V0_price - delta_S * S[:, 0] - delta_A * A0
    tc = 0.0001 * S[:, 0] * np.abs(delta_S) + 0.0005 * A0 * np.abs(delta_A)
    cash -= tc
    tc_total += tc
    delta_S_prev = delta_S
    delta_A_prev = delta_A
    
    # t_i
    for i in range(1, N_steps):
        T_rem_V = T_V - i * dt
        T_rem_A = T_A - i * dt
        S_t = S[:, i]
        A_t = A_cal[:, i]
        
        v_V = bs_vega(S_t, K_V, T_rem_V, sigma_V_cal[:, i])
        v_A = bs_vega(S_t, K_A, T_rem_A, sigma_A_cal[:, i])
        d_V = bs_delta(S_t, K_V, T_rem_V, sigma_V_cal[:, i])
        d_A = bs_delta(S_t, K_A, T_rem_A, sigma_A_cal[:, i])
        
        delta_A = np.clip(v_V / (v_A + 1e-8), -2.0, 2.0)
        delta_S = np.clip(d_V - delta_A * d_A, -2.0, 2.0)
        
        tc = 0.0001 * S_t * np.abs(delta_S - delta_S_prev) + 0.0005 * A_t * np.abs(delta_A - delta_A_prev)
        cash -= (delta_S - delta_S_prev) * S_t + (delta_A - delta_A_prev) * A_t + tc
        tc_total += tc
        
        delta_S_prev = delta_S
        delta_A_prev = delta_A
        
    # t_N
    S_t = S[:, -1]
    A_t = A_cal[:, -1]
    
    tc = 0.0001 * S_t * np.abs(delta_S_prev) + 0.0005 * A_t * np.abs(delta_A_prev)
    cash += delta_S_prev * S_t + delta_A_prev * A_t - tc
    tc_total += tc
    
    payoff = np.maximum(S_t - K_V, 0.0)
    return cash - payoff, tc_total

# ─── Metrics calculator ───────────────────────────────────────────────────────

def compute_metrics(errors, tc):
    mean_err = np.mean(errors)
    var_err = np.var(errors)
    std_err = np.std(errors)
    mean_tc = np.mean(tc)
    
    losses = -errors
    cutoff = np.percentile(losses, 95)
    cvar_95 = np.mean(losses[losses >= cutoff])
    
    return {
        "mean": mean_err,
        "var": var_err,
        "std": std_err,
        "tc": mean_tc,
        "cvar_95": cvar_95
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Milestone 2 Hedging effectiveness study benchmark.")
    parser.add_argument("--paths", type=int, default=50, help="Number of simulation paths")
    parser.add_argument("--steps", type=int, default=52, help="Number of weekly steps (52)")
    parser.add_argument("--noise", type=float, default=0.01, help="Multiplicative IV surface noise level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-deep", action="store_true", help="Train Deep Hedging policies")
    parser.add_argument("--deep-epochs", type=int, default=15, help="Number of Deep Hedging training epochs")
    parser.add_argument("--weights", type=str, default="artifacts/weights/fno_v2_final_prod.pth", help="FNO weights file")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("  Milestone 2: Initial Hedging Backtest Study Benchmark")
    print(f"  Paths: {args.paths}  |  Steps: {args.steps}  |  Noise Level: {args.noise*100:.1f}%")
    print(f"  Device: {device}  |  Seed: {args.seed}  |  Train Deep: {args.train_deep}")
    print("=" * 80)
    
    # 1. Load FNO model
    print("Loading FNO model...")
    model = MirrorPaddedFNO2d()
    w_path = args.weights
    if not os.path.exists(w_path):
        w_path = os.path.join(project_root, args.weights)
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"FNO weights not found at {args.weights}")
    
    model.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    _load_normalizers("v2")
    spatial = _make_spatial_input(T_GRID, K_GRID, device)
    
    rng = np.random.default_rng(args.seed)
    
    # 2. Simulate Heston paths
    print("Simulating Heston paths...")
    # Parameters
    S0, v0 = 100.0, 0.07
    kappa, theta, sigma, rho = 1.0, 0.08, 0.50, -0.60
    T_V, T_A = 1.0, 2.0
    K_V, K_A = 100.0, 100.0
    dt = T_V / args.steps
    
    S, V = simulate_heston_paths(S0, v0, kappa, theta, sigma, rho, T_V, args.steps, args.paths, rng)
    
    # 3. Generate observed surface & Calibrate
    print("Generating surfaces and calibrating Heston parameters...")
    # Precompute all volatilities and option prices
    sigma_V_init = np.zeros(args.paths)
    sigma_A_init = np.zeros(args.paths)
    
    sigma_V_cal = np.zeros((args.paths, args.steps + 1))
    sigma_A_cal = np.zeros((args.paths, args.steps + 1))
    A_cal = np.zeros((args.paths, args.steps + 1))
    A_BS = np.zeros((args.paths, args.steps + 1))
    
    # Build initial surface
    p6d_init = torch.tensor([[1.0, 0.08, 0.50, -0.60, 0.07, 0.08]], dtype=torch.float32, device=device)
    with torch.no_grad():
        iv_init = _fno_predict_real_iv(model, p6d_init, spatial)
        if iv_init.dim() == 3:
            iv_init = iv_init.squeeze(0)
        iv_init_np = iv_init.cpu().numpy()
        
    s_init_V = interpolate_bilinear_np(T_GRID, K_GRID, iv_init_np, T_V, 0.0)
    s_init_A = interpolate_bilinear_np(T_GRID, K_GRID, iv_init_np, T_A, 0.0)
    
    # Pre-populate initial premium
    V0_price = bs_call_price(S0, K_V, T_V, s_init_V)
    
    B = args.paths * (args.steps + 1)
    v0_params_flat = np.clip(V.reshape(-1), 0.01, 0.15)
    
    p6d_batch = torch.zeros((B, 6), dtype=torch.float32, device=device)
    p6d_batch[:, 0] = 1.0   # kappa
    p6d_batch[:, 1] = 0.08  # theta
    p6d_batch[:, 2] = 0.50  # sigma
    p6d_batch[:, 3] = -0.60 # rho
    p6d_batch[:, 4] = torch.tensor(v0_params_flat, dtype=torch.float32, device=device) # v0
    p6d_batch[:, 5] = 0.08  # H
    
    print(f"Generating true surfaces in one forward pass (batch size: {B})...")
    with torch.no_grad():
        iv_true_batch = _fno_predict_real_iv(model, p6d_batch, spatial)
        # Note: _fno_predict_real_iv returns shape (B, 8, 11) because B > 1
        iv_true_batch_np = iv_true_batch.cpu().numpy()
        
    # Add noise
    noise = rng.normal(0, args.noise * iv_true_batch_np, iv_true_batch_np.shape)
    iv_noisy_batch_np = np.maximum(iv_true_batch_np + noise, 1e-4)
    iv_noisy_tensor = torch.tensor(iv_noisy_batch_np, dtype=torch.float32, device=device)
    
    # Batched Newton Calibration
    print(f"Running batched Newton calibration (batch size: {B})...")
    from deepvol.calibration.batch_calibration import calibrate_newton_batch
    import deepvol.calibration.calibrate_bfgs as cb
    pn = cb._param_norm
    yn = cb._iv_norm
    
    cal_theta, _, _ = calibrate_newton_batch(model, iv_noisy_tensor, pn, yn, device, max_iter=15)
    cal_theta_np = cal_theta.cpu().numpy()
    
    # Handle non-finite values safely
    mask = ~np.isfinite(cal_theta_np).any(axis=1)
    if mask.any():
        cal_theta_np[mask, 2] = 0.50  # sigma
        cal_theta_np[mask, 3] = -0.60 # rho
        cal_theta_np[mask, 4] = v0_params_flat[mask]
        
    sigma_cal_flat = cal_theta_np[:, 2]
    rho_cal_flat = cal_theta_np[:, 3]
    v0_cal_flat = cal_theta_np[:, 4]
    
    # Generate calibrated surfaces in one forward pass
    p6d_cal_batch = torch.zeros((B, 6), dtype=torch.float32, device=device)
    p6d_cal_batch[:, 0] = 1.0
    p6d_cal_batch[:, 1] = 0.08
    p6d_cal_batch[:, 2] = torch.tensor(sigma_cal_flat, dtype=torch.float32, device=device)
    p6d_cal_batch[:, 3] = torch.tensor(rho_cal_flat, dtype=torch.float32, device=device)
    p6d_cal_batch[:, 4] = torch.tensor(v0_cal_flat, dtype=torch.float32, device=device)
    p6d_cal_batch[:, 5] = 0.08
    
    print(f"Generating calibrated surfaces in one forward pass (batch size: {B})...")
    with torch.no_grad():
        iv_cal_batch = _fno_predict_real_iv(model, p6d_cal_batch, spatial)
        iv_cal_batch_np = iv_cal_batch.cpu().numpy()
        
    print("Interpolating vols and pre-populating option prices...")
    for p in range(args.paths):
        sigma_V_init[p] = s_init_V
        sigma_A_init[p] = s_init_A
        
        for i in range(args.steps + 1):
            t_i = i * dt
            S_t = S[p, i]
            
            T_rem_V = max(T_V - t_i, 0.0)
            T_rem_A = max(T_A - t_i, 0.0)
            k_V = np.log(S_t / K_V)
            k_A = np.log(S_t / K_A)
            
            idx = p * (args.steps + 1) + i
            iv_cal_np = iv_cal_batch_np[idx]
            
            # Interpolate vols
            sig_V = interpolate_bilinear_np(T_GRID, K_GRID, iv_cal_np, T_rem_V, k_V)
            sig_A = interpolate_bilinear_np(T_GRID, K_GRID, iv_cal_np, T_rem_A, k_A)
            
            sigma_V_cal[p, i] = sig_V
            sigma_A_cal[p, i] = sig_A
            
            # Auxiliary Option Prices
            A_cal[p, i] = bs_call_price(S_t, K_A, T_rem_A, sig_A)
            A_BS[p, i] = bs_call_price(S_t, K_A, T_rem_A, s_init_A)
            
    # 4. Evaluate analytical strategies
    print("\nEvaluating analytical strategies...")
    err_bs_d, tc_bs_d = evaluate_bs_delta(S, K_V, T_V, dt, sigma_V_init, V0_price)
    err_fno_d, tc_fno_d = evaluate_fno_delta(S, K_V, T_V, dt, sigma_V_cal, V0_price)
    err_bs_dv, tc_bs_dv = evaluate_bs_delta_vega(S, A_BS, K_V, K_A, T_V, T_A, dt, sigma_V_init, sigma_A_init, V0_price)
    err_fno_dv, tc_fno_dv = evaluate_fno_delta_vega(S, A_cal, K_V, K_A, T_V, T_A, dt, sigma_V_cal, sigma_A_cal, V0_price)
    
    # 5. Deep Hedging (Stock-only and Delta-Vega)
    results_deep_stock = None
    results_deep_dv = None
    
    if args.train_deep:
        # Stock-only Deep Hedging
        print("\nTraining Deep Hedging Policy (Stock-only)...")
        # Prepare environment
        H_stock = torch.tensor(S, dtype=torch.float32, device=device).unsqueeze(-1)
        payoff_t = torch.clamp(torch.tensor(S[:, -1] - K_V, dtype=torch.float32, device=device), min=0.0)
        
        env_stock = DeepHedgingEnv(
            H=H_stock,
            payoff=payoff_t,
            cost_coeffs=torch.tensor([0.0001], dtype=torch.float32, device=device),
            risk_aversion=1.0,
            risk_measure="entropic",
            strike=K_V,
            expiry=T_V
        )
        
        policy_stock = HedgingPolicy(input_dim=4, hidden_dim=64, output_dim=1).to(device)
        train_deep_hedger(env_stock, policy_stock, lr=1e-3, epochs=args.deep_epochs, batch_size=256, device=str(device))
        
        policy_stock.eval()
        with torch.no_grad():
            wealth_stock, tc_stock, _ = env_stock.simulate_hedging_episode(policy_stock)
        
        # Deep Hedging error = wealth - payoff + V0_price (to match final cash comparison)
        err_deep_stock = (wealth_stock - payoff_t).cpu().numpy() + V0_price
        results_deep_stock = compute_metrics(err_deep_stock, tc_stock.cpu().numpy())
        
        # Delta-Vega Deep Hedging
        print("\nTraining Deep Hedging Policy (Delta-Vega)...")
        # Prepare environment with Stock and Auxiliary Option
        H_dv = torch.zeros((args.paths, args.steps + 1, 2), dtype=torch.float32, device=device)
        H_dv[:, :, 0] = torch.tensor(S, dtype=torch.float32, device=device)
        H_dv[:, :, 1] = torch.tensor(A_cal, dtype=torch.float32, device=device)
        
        env_dv = DeepHedgingEnv(
            H=H_dv,
            payoff=payoff_t,
            cost_coeffs=torch.tensor([0.0001, 0.0005], dtype=torch.float32, device=device),
            risk_aversion=1.0,
            risk_measure="entropic",
            strike=K_V,
            expiry=T_V
        )
        
        policy_dv = HedgingPolicy(input_dim=5, hidden_dim=64, output_dim=2).to(device)
        train_deep_hedger(env_dv, policy_dv, lr=1e-3, epochs=args.deep_epochs, batch_size=256, device=str(device))
        
        policy_dv.eval()
        with torch.no_grad():
            wealth_dv, tc_dv, _ = env_dv.simulate_hedging_episode(policy_dv)
            
        err_deep_dv = (wealth_dv - payoff_t).cpu().numpy() + V0_price
        results_deep_dv = compute_metrics(err_deep_dv, tc_dv.cpu().numpy())
        
    # 6. Report metrics
    metrics_bs_d = compute_metrics(err_bs_d, tc_bs_d)
    metrics_fno_d = compute_metrics(err_fno_d, tc_fno_d)
    metrics_bs_dv = compute_metrics(err_bs_dv, tc_bs_dv)
    metrics_fno_dv = compute_metrics(err_fno_dv, tc_fno_dv)
    
    print("\n" + "=" * 90)
    print("  HEDGING BACKTEST PERFORMANCE COMPARISON")
    print("=" * 90)
    print(f"  {'Strategy':<26} | {'Mean Err':>10} | {'Std Err':>10} | {'Var Err':>10} | {'Avg Cost':>10} | {'95% CVaR':>10}")
    print("-" * 90)
    print(f"  {'BS Delta (Flat)':<26} | {metrics_bs_d['mean']:>10.5f} | {metrics_bs_d['std']:>10.5f} | {metrics_bs_d['var']:>10.5f} | {metrics_bs_d['tc']:>10.5f} | {metrics_bs_d['cvar_95']:>10.5f}")
    print(f"  {'FNO Greeks Delta':<26} | {metrics_fno_d['mean']:>10.5f} | {metrics_fno_d['std']:>10.5f} | {metrics_fno_d['var']:>10.5f} | {metrics_fno_d['tc']:>10.5f} | {metrics_fno_d['cvar_95']:>10.5f}")
    print(f"  {'BS Delta-Vega (Flat)':<26} | {metrics_bs_dv['mean']:>10.5f} | {metrics_bs_dv['std']:>10.5f} | {metrics_bs_dv['var']:>10.5f} | {metrics_bs_dv['tc']:>10.5f} | {metrics_bs_dv['cvar_95']:>10.5f}")
    print(f"  {'FNO Greeks Delta-Vega':<26} | {metrics_fno_dv['mean']:>10.5f} | {metrics_fno_dv['std']:>10.5f} | {metrics_fno_dv['var']:>10.5f} | {metrics_fno_dv['tc']:>10.5f} | {metrics_fno_dv['cvar_95']:>10.5f}")
    
    if results_deep_stock:
        print(f"  {'Deep Hedging (Stock-only)':<26} | {results_deep_stock['mean']:>10.5f} | {results_deep_stock['std']:>10.5f} | {results_deep_stock['var']:>10.5f} | {results_deep_stock['tc']:>10.5f} | {results_deep_stock['cvar_95']:>10.5f}")
    if results_deep_dv:
        print(f"  {'Deep Hedging (Delta-Vega)':<26} | {results_deep_dv['mean']:>10.5f} | {results_deep_dv['std']:>10.5f} | {results_deep_dv['var']:>10.5f} | {results_deep_dv['tc']:>10.5f} | {results_deep_dv['cvar_95']:>10.5f}")
    print("=" * 90)
    
    # Save results to txt file
    res_dir = os.path.join(project_root, "results", "hedging_benchmark")
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, "hedging_backtest_results.txt")
    
    with open(res_path, "w") as f:
        f.write("HEDGING BACKTEST PERFORMANCE COMPARISON\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Strategy':<26} | {'Mean Err':>10} | {'Std Err':>10} | {'Var Err':>10} | {'Avg Cost':>10} | {'95% CVaR':>10}\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'BS Delta (Flat)':<26} | {metrics_bs_d['mean']:>10.5f} | {metrics_bs_d['std']:>10.5f} | {metrics_bs_d['var']:>10.5f} | {metrics_bs_d['tc']:>10.5f} | {metrics_bs_d['cvar_95']:>10.5f}\n")
        f.write(f"{'FNO Greeks Delta':<26} | {metrics_fno_d['mean']:>10.5f} | {metrics_fno_d['std']:>10.5f} | {metrics_fno_d['var']:>10.5f} | {metrics_fno_d['tc']:>10.5f} | {metrics_fno_d['cvar_95']:>10.5f}\n")
        f.write(f"{'BS Delta-Vega (Flat)':<26} | {metrics_bs_dv['mean']:>10.5f} | {metrics_bs_dv['std']:>10.5f} | {metrics_bs_dv['var']:>10.5f} | {metrics_bs_dv['tc']:>10.5f} | {metrics_bs_dv['cvar_95']:>10.5f}\n")
        f.write(f"{'FNO Greeks Delta-Vega':<26} | {metrics_fno_dv['mean']:>10.5f} | {metrics_fno_dv['std']:>10.5f} | {metrics_fno_dv['var']:>10.5f} | {metrics_fno_dv['tc']:>10.5f} | {metrics_fno_dv['cvar_95']:>10.5f}\n")
        if results_deep_stock:
            f.write(f"{'Deep Hedging (Stock-only)':<26} | {results_deep_stock['mean']:>10.5f} | {results_deep_stock['std']:>10.5f} | {results_deep_stock['var']:>10.5f} | {results_deep_stock['tc']:>10.5f} | {results_deep_stock['cvar_95']:>10.5f}\n")
        if results_deep_dv:
            f.write(f"{'Deep Hedging (Delta-Vega)':<26} | {results_deep_dv['mean']:>10.5f} | {results_deep_dv['std']:>10.5f} | {results_deep_dv['var']:>10.5f} | {results_deep_dv['tc']:>10.5f} | {results_deep_dv['cvar_95']:>10.5f}\n")
            
    print(f"Results saved to {res_path}")

if __name__ == "__main__":
    main()
