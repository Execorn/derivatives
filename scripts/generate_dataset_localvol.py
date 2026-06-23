"""
generate_dataset_localvol.py — Generate Local Volatility dataset from SVI parameters.

Samples SVI parameters uniformly (with term structure structure to avoid arbitrage),
computes the corresponding Dupire local volatility surfaces on a fine grid,
filters out surfaces that violate calendar spread or butterfly arbitrage (or have extreme LV values),
and saves the valid dataset to data/LocalVolDataset_v1.npz.
"""

import os
import sys
import time
import numpy as np

# Ensure project root is on PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pricing.local_vol import svi_to_lv_surface, check_arbitrage_free

# Config
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)
N_TARGET_SAMPLES = 2048 if '--smoke' in sys.argv else 60000
BATCH_SIZE = 2048 if '--smoke' in sys.argv else 1000
OUTPUT_PATH = 'data/LocalVolDataset_v1.npz'

def generate():
    print("=" * 60)
    print("  Local Volatility Dataset Generation  ")
    print("=" * 60)
    print(f"  Target Samples : {N_TARGET_SAMPLES}")
    print(f"  T Grid         : {T_GRID.tolist()}")
    print(f"  K Grid         : {K_GRID.tolist()}")
    print(f"  Output Path    : {OUTPUT_PATH}")
    print()

    np.random.seed(42)
    
    saved_params = []
    saved_lv = []
    
    total_sampled = 0
    t0 = time.time()
    
    while len(saved_params) < N_TARGET_SAMPLES:
        # Sample base SVI parameters
        a0 = np.random.uniform(0.01, 0.15, size=BATCH_SIZE)
        b0 = np.random.uniform(0.05, 0.4, size=BATCH_SIZE)
        rho_base = np.random.uniform(-0.85, -0.15, size=BATCH_SIZE)
        m_base = np.random.uniform(-0.15, 0.15, size=BATCH_SIZE)
        sigma_base = np.random.uniform(0.05, 0.35, size=BATCH_SIZE)
        
        # Build 8 slices of SVI parameters
        svi_params = np.zeros((BATCH_SIZE, 8, 5))
        for j in range(8):
            T = T_GRID[j]
            # scale grows with maturity
            scale = T * np.random.uniform(0.9, 1.1, size=BATCH_SIZE)
            
            svi_params[:, j, 0] = a0 * scale * np.random.uniform(0.95, 1.05, size=BATCH_SIZE)
            svi_params[:, j, 1] = b0 * scale * np.random.uniform(0.95, 1.05, size=BATCH_SIZE)
            svi_params[:, j, 2] = np.clip(rho_base + np.random.uniform(-0.02, 0.02, size=BATCH_SIZE), -0.95, -0.05)
            svi_params[:, j, 3] = m_base + np.random.uniform(-0.01, 0.01, size=BATCH_SIZE)
            svi_params[:, j, 4] = np.clip(sigma_base + np.random.uniform(-0.01, 0.01, size=BATCH_SIZE), 0.01, 0.5)
            
        total_sampled += BATCH_SIZE
        
        # Compute LV surfaces in batch
        lv_surfaces = svi_to_lv_surface(T_GRID, K_GRID, svi_params) # shape (BATCH_SIZE, 8, 11)
        
        # Filter based on bounds and NaNs
        min_vals = lv_surfaces.min(axis=(1, 2))
        max_vals = lv_surfaces.max(axis=(1, 2))
        has_nan = np.isnan(lv_surfaces).any(axis=(1, 2))
        
        # Check if min_val >= 0.0 and max_val <= 3.0 and no NaNs
        valid_mask = (min_vals >= 0.0) & (max_vals <= 3.0) & (~has_nan)
        
        # Double check with analytical no-arbitrage check
        for idx in np.where(valid_mask)[0]:
            if len(saved_params) >= N_TARGET_SAMPLES:
                break
            if check_arbitrage_free(T_GRID, K_GRID, svi_params[idx]):
                # Reshape SVI parameters to flat vector of size 40
                flat_params = svi_params[idx].flatten()
                saved_params.append(flat_params)
                saved_lv.append(lv_surfaces[idx])
                
        print(f"  Progress: {len(saved_params):,}/{N_TARGET_SAMPLES:,} valid samples | "
              f"Yield: {len(saved_params)/total_sampled * 100:.1f}% | "
              f"Time: {time.time() - t0:.1f}s")
              
    saved_params = np.array(saved_params[:N_TARGET_SAMPLES], dtype=np.float32)
    saved_lv = np.array(saved_lv[:N_TARGET_SAMPLES], dtype=np.float32)
    
    # Save the dataset
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        params=saved_params,
        lv=saved_lv,
        T_grid=T_GRID,
        K_grid=K_GRID
    )
    
    print(f"\n  Successfully saved dataset to {OUTPUT_PATH}")
    print(f"  Params shape: {saved_params.shape}")
    print(f"  LV shape    : {saved_lv.shape}")
    print(f"  Total time  : {time.time() - t0:.2f}s")

if __name__ == '__main__':
    generate()
