"""
generate_dataset_heston.py — Dataset Generator for Classic Heston model.

Steps:
  1. Sobol low-discrepancy sampling of 5 parameters: [kappa, theta, sigma, rho, v0].
  2. Filter out Feller condition violations: 2*kappa*theta <= sigma**2.
  3. Batch-price on GPU using Fourier-COS.
  4. Fill single NaN columns with maturity medians.
  5. Keep only valid (non-NaN, non-inf, IV in [0, 5.0]) surfaces.
  6. Save to data/HestonDataset_v1.npz.
"""

import os
import sys
import time
import numpy as np
import torch
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pricing.heston import batch_heston_iv_surface

# --- Config ---
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)

PARAM_NAMES = ['kappa', 'theta', 'sigma', 'rho', 'v0']
BOUNDS_LOWER = np.array([0.1, 0.01, 0.1, -0.9, 0.01])
BOUNDS_UPPER = np.array([5.0, 0.15, 1.0, -0.1, 0.15])

N_SOBOL_SAMPLES = 2048 if '--smoke' in sys.argv else 131072  # 2^17
BATCH_SIZE = 4096  # Classic Heston is very fast, large batches are efficient
N_COS = 128
OUTPUT_PATH = 'data/HestonDataset_v1.npz'

def generate():
    print("=" * 60)
    print("  Heston Dataset Generator (Option 1)")
    print("=" * 60)
    print(f"  Sobol Samples : {N_SOBOL_SAMPLES:,}")
    print(f"  T grid        : {T_GRID.tolist()}")
    print(f"  K grid        : {K_GRID.tolist()}")
    print(f"  Free params   : {PARAM_NAMES}")
    print(f"  Output        : {OUTPUT_PATH}")
    print()

    # 1. Sobol sampling
    sampler = qmc.Sobol(d=5, scramble=True, seed=42)
    unit_pts = sampler.random(N_SOBOL_SAMPLES)
    params_all = qmc.scale(unit_pts, BOUNDS_LOWER, BOUNDS_UPPER).astype(np.float64)

    # 2. Filter out Feller condition violations
    kappa = params_all[:, 0]
    theta = params_all[:, 1]
    sigma = params_all[:, 2]
    feller_valid = (2 * kappa * theta > sigma**2)
    params = params_all[feller_valid]
    
    n_feller_valid = params.shape[0]
    print(f"  Feller filter: {n_feller_valid:,} / {N_SOBOL_SAMPLES:,} samples kept ({n_feller_valid/N_SOBOL_SAMPLES*100:.1f}%)")

    # 3. GPU batched pricing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Pricing on {device}...")
    
    n_batches = (n_feller_valid + BATCH_SIZE - 1) // BATCH_SIZE
    iv_chunks = []
    
    t0 = time.time()
    for b in range(n_batches):
        s = b * BATCH_SIZE
        e = min((b + 1) * BATCH_SIZE, n_feller_valid)
        batch_params = torch.tensor(params[s:e], dtype=torch.float64, device=device)
        
        # Price batch
        with torch.no_grad():
            iv_batch = batch_heston_iv_surface(
                batch_params,
                torch.tensor(T_GRID, device=device),
                torch.tensor(K_GRID, device=device),
                S0=1.0,
                N_cos=N_COS,
                device=device
            )
        iv_chunks.append(iv_batch.cpu().numpy())
        
        if (b + 1) % 5 == 0 or b == n_batches - 1:
            elapsed = time.time() - t0
            eta = (elapsed / (s + BATCH_SIZE)) * (n_feller_valid - (s + BATCH_SIZE)) if s + BATCH_SIZE < n_feller_valid else 0
            print(f"    Batch {b+1}/{n_batches} | Done: {e:,} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            
    iv_all = np.concatenate(iv_chunks, axis=0)  # (N_feller, 8, 11)
    
    # 4. Fill single NaN columns with maturity medians
    # Replace non-finite with NaN
    iv_all[~np.isfinite(iv_all)] = np.nan
    for t_idx in range(len(T_GRID)):
        slice_t = iv_all[:, t_idx, :]  # (N_feller, 11)
        valid_vals = slice_t[np.isfinite(slice_t)]
        med = np.median(valid_vals) if len(valid_vals) > 0 else 0.3
        
        nan_rows = np.isnan(slice_t).any(axis=1)
        if nan_rows.any():
            slice_t[nan_rows] = med
        iv_all[:, t_idx, :] = slice_t
        
    # 5. Keep only valid (non-NaN, non-inf, IV in [0, 5.0]) surfaces
    valid_mask = np.all((iv_all >= 0.0) & (iv_all <= 5.0) & np.isfinite(iv_all), axis=(1, 2))
    
    params_final = params[valid_mask]
    iv_final = iv_all[valid_mask]
    nan_mask_final = valid_mask  # True means genuine and kept
    
    n_final = params_final.shape[0]
    print(f"\n  Final valid dataset size: {n_final:,} samples ({n_final/n_feller_valid*100:.1f}% of Feller-valid)")
    print(f"  Total generation time: {(time.time() - t0)/60:.2f} minutes")
    
    # Save dataset
    iv_flat = iv_final.reshape(n_final, -1)
    dataset = np.concatenate([params_final, iv_flat], axis=1).astype(np.float32)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        dataset=dataset,
        params=params_final.astype(np.float32),
        iv=iv_final.astype(np.float32),
        nan_mask=nan_mask_final,
        T_grid=T_GRID,
        K_grid=K_GRID,
        param_names=np.array(PARAM_NAMES)
    )
    print(f"  Saved dataset to {OUTPUT_PATH}")

if __name__ == '__main__':
    # Since the user requested not to execute the full dataset generation yet,
    # we just run the main entry point if explicitly executed.
    generate()
