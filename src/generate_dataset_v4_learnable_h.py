"""
generate_dataset_v4_learnable_h.py — Option 4: Learnable Hurst Dataset

Extends v2 by treating H as a free calibration parameter: H ~ Uniform[0.04, 0.15].
The FNO v3 trained on this dataset can calibrate all 4 identifiable parameters
simultaneously: (v0, zeta=sigma*rho, lam=sigma*sqrt(1-rho^2), H).

Differences from v2:
  - PARAM_NAMES adds 'H' → 6 free params instead of 5
  - price_batch_gpu called with H_batch (per-sample) instead of H_fixed
  - Output dataset has shape (N, 7+88): [kappa, theta, sigma, rho, v0, H | 88 IVs]
  - Normalizer will have 6 free param dimensions (H included)

Usage (from repo root):
    .venv/bin/python src/generate_dataset_v4_learnable_h.py

Estimated time: ~35 min on RTX 3060 (same as v2).
Output: data/DeepRoughDataset_v4_learnable_h.npz
"""

import os, sys, time
import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pricing_engine_gpu import price_batch_gpu

# ─── Config ────────────────────────────────────────────────────────────────────
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])   # same as v2
K_GRID = np.linspace(-0.5, 0.5, 11)

# Free parameters: [kappa, theta, sigma, rho, v0, H]
PARAM_NAMES  = ['kappa', 'theta', 'sigma', 'rho', 'v0', 'H']
BOUNDS_LOWER = np.array([0.1,  0.01, 0.1,  -0.9, 0.01, 0.04])
BOUNDS_UPPER = np.array([5.0,  0.15, 1.0,  -0.1, 0.15, 0.15])

N_SAMPLES         = 65536   # next power-of-2 above 50k; Sobol requires 2^n
BATCH_SIZE        = 512     # smaller batches reduce OOM risk at high N_steps
N_COS             = 128
N_FACTORS         = 40
N_STEPS_PER_UNIT  = 500     # 200→500: H=0.04 is near-singular; needs finer grid

OUTPUT_PATH = 'data/DeepRoughDataset_v4_learnable_h.npz'


def generate():
    print("=" * 60)
    print("  Option 4 Dataset — Learnable Hurst  (H ∈ [0.04, 0.15])")
    print("=" * 60)
    print(f"  Samples     : {N_SAMPLES:,}")
    print(f"  T grid      : {T_GRID.tolist()}")
    print(f"  N_cos       : {N_COS}  N_factors : {N_FACTORS}")
    print(f"  Free params : {PARAM_NAMES}")
    print(f"  Output      : {OUTPUT_PATH}")
    print()

    # Scrambled Sobol over 6D unit hypercube
    sampler  = qmc.Sobol(d=6, scramble=True, seed=42)
    unit_pts = sampler.random(N_SAMPLES)
    params6  = qmc.scale(unit_pts, BOUNDS_LOWER, BOUNDS_UPPER).astype(np.float64)

    for i, name in enumerate(PARAM_NAMES):
        print(f"    {name:6s}: [{params6[:, i].min():.4f}, {params6[:, i].max():.4f}]")

    # Warmup pass (compile CUDA JIT)
    print("\n  Warming up CUDA JIT...")
    _ = price_batch_gpu(
        params6[:4, :5], T_GRID, K_GRID,
        H_batch=params6[:4, 5].astype(np.float32),
        N_factors=N_FACTORS, N_cos=N_COS,
        N_steps_per_unit=N_STEPS_PER_UNIT, device='cuda')
    print("  Warmup done.\n")

    n_batches  = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    iv_chunks  = []
    nan_chunks = []
    total_nan  = 0
    total_done = 0
    t0 = time.time()

    print(f"  {'Batch':>6} | {'Done':>8} | {'NaN%':>7} | {'Batch t':>8} | {'ETA':>8}")
    print("  " + "-" * 50)

    for b in range(n_batches):
        s, e = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, N_SAMPLES)
        batch_params  = params6[s:e, :5]   # kappa, theta, sigma, rho, v0
        batch_H       = params6[s:e,  5].astype(np.float32)

        bt = time.time()
        iv = price_batch_gpu(
            batch_params, T_GRID, K_GRID,
            H_batch=batch_H,
            N_factors=N_FACTORS, N_cos=N_COS,
            N_steps_per_unit=N_STEPS_PER_UNIT, device='cuda')
        bt = time.time() - bt

        nan_mask = ~np.isnan(iv).any(axis=(1, 2))   # True = fully valid surface
        nan_pct  = (~nan_mask).mean() * 100
        total_nan  += (~nan_mask).sum()
        total_done += (e - s)

        # Fill NaN rows (entire maturity slice) with the median IV for that maturity
        iv_filled = iv.copy()
        for ti in range(iv.shape[1]):
            col      = iv_filled[:, ti, :]              # (B, 11)
            nan_rows = np.isnan(col).any(axis=1)        # (B,) — rows with any NaN
            if nan_rows.any():
                med = float(np.nanmedian(col[~nan_rows])) if (~nan_rows).any() else 0.3
                iv_filled[nan_rows, ti, :] = med if not np.isnan(med) else 0.3

        iv_chunks.append(iv_filled.astype(np.float32))
        nan_chunks.append(nan_mask)

        rate  = total_done / (time.time() - t0)
        eta_s = (N_SAMPLES - total_done) / max(rate, 1e-6)
        print(f"  {b+1:>6}/{n_batches} | {total_done:>8,} | "
              f"{nan_pct:>6.1f}%  | "
              f"{bt:>7.2f}s  | {eta_s/60:>6.1f}min")

    iv_all   = np.concatenate(iv_chunks,  axis=0)   # (N, 8, 11)
    nan_all  = np.concatenate(nan_chunks, axis=0)   # (N,) bool

    print(f"\n  NaN fraction    : {(~nan_all).mean()*100:.2f}%  ({(~nan_all).sum()} / {N_SAMPLES})")
    print(f"  Valid fraction  : {nan_all.mean()*100:.2f}%")
    print(f"  Total time      : {(time.time()-t0)/60:.1f} min")

    # Build 7-column params array: [kappa, theta, sigma, rho, v0, H]
    params7  = params6.astype(np.float32)      # already 6D including H
    iv_flat  = iv_all.reshape(N_SAMPLES, -1)
    dataset  = np.concatenate([params7, iv_flat], axis=1).astype(np.float32)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        dataset=dataset,
        params=params7,
        iv=iv_all,
        nan_mask=nan_all,
        T_grid=T_GRID,
        K_grid=K_GRID,
        param_names=np.array(PARAM_NAMES),
    )
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"  Saved → {OUTPUT_PATH}  ({size_mb:.1f} MB)\n")

    # Spot-check
    print(f"  {'params (k,th,s,r,v0,H)':<36}  {'ATM IV @ T=0.1':>14}  {'ATM IV @ T=1.0':>14}")
    print(f"  {'-'*66}")
    atm_k = np.argmin(np.abs(K_GRID))
    for i in range(min(5, N_SAMPLES)):
        k, th, s, r, v, H = params7[i]
        iv01 = iv_all[i, 0, atm_k]
        iv10 = iv_all[i, 4, atm_k]
        print(f"  k={k:.2f} θ={th:.3f} σ={s:.2f} ρ={r:.2f} v₀={v:.3f} H={H:.3f}"
              f"  {iv01*100:>12.2f}%  {iv10*100:>12.2f}%")


if __name__ == '__main__':
    generate()
