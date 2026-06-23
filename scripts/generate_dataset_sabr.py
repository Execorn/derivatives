"""
generate_dataset_sabr.py — Dataset generation for SABR and SSVI models.

Generates:
1. data/SABRDataset_v1.npz (param_dim=3: alpha, rho, nu)
2. data/SSVIDataset_v1.npz (param_dim=11: theta_1..8, rho, eta, gamma)

Maturities (T) and log-moneyness (k) grids are identical to Heston FNO:
T = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]  (8 points)
k = np.linspace(-0.5, 0.5, 11)                (11 points)

Arbitrage constraints for SSVI are enforced during sampling:
- Monotone ATM term structure: theta_1 < theta_2 < ... < theta_8
- Butterfly arbitrage-free: eta * (1 + |rho|) <= 2.0, gamma in [0.1, 0.5]
"""

import os
import sys
import numpy as np
from scipy.stats import qmc

# Ensure src path is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from pricing.sabr import sabr_iv_surface, ssvi_iv_surface

# ─── Config ────────────────────────────────────────────────────────────────────
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)

N_SAMPLES = 2048 if '--smoke' in sys.argv else 65536  # 2^16 samples
BATCH_SIZE = 1024

SABR_OUT_PATH = "data/SABRDataset_v1.npz"
SSVI_OUT_PATH = "data/SSVIDataset_v1.npz"


def generate_sabr_dataset():
    print("=" * 60)
    print("  Generating SABR Dataset (param_dim=3: alpha, rho, nu)")
    print("=" * 60)
    
    # 3D Sobol sampler
    sampler = qmc.Sobol(d=3, scramble=True, seed=42)
    unit_pts = sampler.random(N_SAMPLES)
    
    # Scale bounds:
    # alpha: [0.05, 0.8]
    # rho: [-0.9, 0.9]
    # nu: [0.1, 1.2]
    bounds_lower = np.array([0.05, -0.9, 0.1])
    bounds_upper = np.array([0.8, 0.9, 1.2])
    
    params = qmc.scale(unit_pts, bounds_lower, bounds_upper).astype(np.float32)
    
    iv_surfaces = np.zeros((N_SAMPLES, len(T_GRID), len(K_GRID)), dtype=np.float32)
    
    # Compute surfaces in batches
    n_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    
    for b in range(n_batches):
        s, e = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, N_SAMPLES)
        batch_params = params[s:e]
        
        for i in range(e - s):
            alpha = batch_params[i, 0]
            rho = batch_params[i, 1]
            nu = batch_params[i, 2]
            
            # Compute SABR lognormal surface (beta=1.0 fixed)
            surface = sabr_iv_surface(
                F=1.0,
                T_grid=T_GRID,
                k_grid=K_GRID,
                alpha=alpha,
                beta=1.0,
                rho=rho,
                nu=nu,
                iv_type="lognormal"
            )
            iv_surfaces[s + i] = surface.astype(np.float32)
            
        if (b + 1) % 10 == 0 or b == n_batches - 1:
            print(f"  Processed {e:,} / {N_SAMPLES:,} samples...")
            
    # Check for NaNs and fill with mature-wise median
    nan_mask = ~np.isnan(iv_surfaces).any(axis=(1, 2))
    nan_pct = (~nan_mask).mean() * 100
    print(f"  NaN surface percentage: {nan_pct:.2f}%")
    
    for ti in range(len(T_GRID)):
        slice_data = iv_surfaces[:, ti, :]
        nan_rows = np.isnan(slice_data).any(axis=1)
        if nan_rows.any():
            valid_rows = slice_data[~nan_rows]
            if len(valid_rows) > 0:
                median_val = np.nanmedian(valid_rows, axis=0)
            else:
                median_val = np.full(len(K_GRID), 0.3)
            # Replace NaNs
            for col_idx in range(len(K_GRID)):
                nans = np.isnan(slice_data[:, col_idx])
                slice_data[nans, col_idx] = median_val[col_idx]
                
    # Flatten the IV surfaces for standard FNO dataset format
    iv_flat = iv_surfaces.reshape(N_SAMPLES, -1)
    dataset = np.concatenate([params, iv_flat], axis=1).astype(np.float32)
    
    os.makedirs(os.path.dirname(SABR_OUT_PATH), exist_ok=True)
    np.savez_compressed(
        SABR_OUT_PATH,
        dataset=dataset,
        params=params,
        iv=iv_surfaces,
        nan_mask=nan_mask,
        T_grid=T_GRID,
        K_grid=K_GRID,
        param_names=np.array(["alpha", "rho", "nu"]),
    )
    print(f"  Saved → {SABR_OUT_PATH} ({os.path.getsize(SABR_OUT_PATH)/1e6:.2f} MB)\n")


def generate_ssvi_dataset():
    print("=" * 60)
    print("  Generating SSVI Dataset (param_dim=11: theta_1..8, rho, eta, gamma)")
    print("=" * 60)
    
    # 11D Sobol sampler
    sampler = qmc.Sobol(d=11, scramble=True, seed=42)
    unit_pts = sampler.random(N_SAMPLES)
    
    # Map dimensions:
    # u_rho = unit_pts[:, 0]
    # u_gamma = unit_pts[:, 1]
    # u_eta = unit_pts[:, 2]
    # u_sigmas = unit_pts[:, 3:11]
    
    rho = -0.9 + 1.8 * unit_pts[:, 0]
    gamma = 0.1 + 0.4 * unit_pts[:, 1]
    
    # eta(1+|rho|) <= 2.0 to prevent butterfly arbitrage.
    # We sample a scaling factor u in [0.05, 1.0], giving eta = u * 2.0 / (1 + |rho|).
    u_scale = 0.05 + 0.95 * unit_pts[:, 2]
    eta = u_scale * 2.0 / (1.0 + np.abs(rho))
    
    # Sample sigmas for the ATM total variance increments.
    # This guarantees monotone ATM total variance term structure: theta_1 < theta_2 < ... < theta_8.
    sigmas = 0.08 + 0.72 * unit_pts[:, 3:11]  # (N, 8) in [0.08, 0.8]
    
    theta = np.zeros((N_SAMPLES, 8), dtype=np.float32)
    theta[:, 0] = (sigmas[:, 0] ** 2) * T_GRID[0]
    for i in range(1, 8):
        theta[:, i] = theta[:, i-1] + (T_GRID[i] - T_GRID[i-1]) * (sigmas[:, i] ** 2)
        
    # Assemble parameter vector of shape (N, 11)
    params = np.zeros((N_SAMPLES, 11), dtype=np.float32)
    params[:, 0:8] = theta
    params[:, 8] = rho
    params[:, 9] = eta
    params[:, 10] = gamma
    
    iv_surfaces = np.zeros((N_SAMPLES, len(T_GRID), len(K_GRID)), dtype=np.float32)
    
    n_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    
    for b in range(n_batches):
        s, e = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, N_SAMPLES)
        batch_params = params[s:e]
        
        for i in range(e - s):
            theta_sample = batch_params[i, 0:8]
            rho_val = batch_params[i, 8]
            eta_val = batch_params[i, 9]
            gamma_val = batch_params[i, 10]
            
            surface = ssvi_iv_surface(
                T_grid=T_GRID,
                k_grid=K_GRID,
                theta_grid=theta_sample,
                rho=rho_val,
                eta=eta_val,
                gamma=gamma_val
            )
            iv_surfaces[s + i] = surface.astype(np.float32)
            
        if (b + 1) % 10 == 0 or b == n_batches - 1:
            print(f"  Processed {e:,} / {N_SAMPLES:,} samples...")
            
    # Verify no NaNs
    nan_mask = ~np.isnan(iv_surfaces).any(axis=(1, 2))
    nan_pct = (~nan_mask).mean() * 100
    print(f"  NaN surface percentage: {nan_pct:.2f}%")
    
    # Flatten and save
    iv_flat = iv_surfaces.reshape(N_SAMPLES, -1)
    dataset = np.concatenate([params, iv_flat], axis=1).astype(np.float32)
    
    os.makedirs(os.path.dirname(SSVI_OUT_PATH), exist_ok=True)
    np.savez_compressed(
        SSVI_OUT_PATH,
        dataset=dataset,
        params=params,
        iv=iv_surfaces,
        nan_mask=nan_mask,
        T_grid=T_GRID,
        K_grid=K_GRID,
        param_names=np.array(
            [f"theta_{i+1}" for i in range(8)] + ["rho", "eta", "gamma"]
        ),
    )
    print(f"  Saved → {SSVI_OUT_PATH} ({os.path.getsize(SSVI_OUT_PATH)/1e6:.2f} MB)\n")


if __name__ == "__main__":
    generate_sabr_dataset()
    generate_ssvi_dataset()
