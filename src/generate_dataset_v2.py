"""
generate_dataset_v2.py — Generate 50k exact IV surfaces using GPU Fourier-COS pricing.

Replaces CUDA Monte Carlo (O(dt^0.08) systematic bias ~15bp at T=0.1) with
deterministic Fourier-COS pricing via the corrected GPU Riccati solver.

NaN HANDLING
────────────
For extreme (K, T) combinations (e.g. K=±0.5 at T=0.1) the option price is
below float64 precision at low-vol parameters — no algorithm can recover an
IV there.  Rather than discarding samples (100% of samples have ≥1 such NaN),
we fill those positions by linear interpolation / extrapolation along the K
axis.  The smooth IV smile makes this physically sound; the neural network
learns the boundary behaviour from the interpolated values.

A 'nan_mask' boolean array (True = original COS price; False = interpolated)
is saved alongside the dataset so that downstream code can optionally weight
the interpolated cells lower in the loss function.

Dataset format (compatible with 8 mat × 11 strike = 88 output pipeline):
  'dataset': float32 ndarray shape (N, 94)
      Columns 0-5:  [kappa, theta, sigma, rho, v0, H]
      Columns 6-93: IV surface (8 × 11 = 88 values, flattened row-major)
  'params':   float32 (N, 6)
  'iv':       float32 (N, 8, 11)   -- interpolated where originally NaN
  'nan_mask': bool    (N, 8, 11)   -- True = valid COS price; False = interpolated

Usage:
    cd /home/execorn/programming/derivatives-option2
    /home/execorn/programming/derivatives/.venv/bin/python src/generate_dataset_v2.py
"""

import os
import sys
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from pricing_engine_gpu import price_batch_gpu

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])  # 8 maturities
K_GRID = np.linspace(-0.5, 0.5, 11)                             # 11 log-moneyness strikes

# Parameter bounds: [kappa, theta, sigma, rho, v0]
# Note: sigma here is vol-of-vol (Heston ν). The spot variance is v0/theta.
BOUNDS_LOWER = np.array([0.1,  0.01, 0.1,  -0.9, 0.01])
BOUNDS_UPPER = np.array([5.0,  0.15, 1.0,  -0.1, 0.15])
PARAM_NAMES  = ['kappa', 'theta', 'sigma', 'rho', 'v0']
H_FIXED      = 0.08

N_SAMPLES         = 50_000
BATCH_SIZE        = 2048     # smaller batch avoids OOM on 6 GB VRAM
N_STEPS_PER_UNIT  = 200      # dt = 0.005  →  400 steps for T_max=2.0
N_COS             = 64       # 64 terms on [-4,4] → machine-precision (err≈4e-15)
N_FACTORS         = 20

OUTPUT_PATH = 'data/DeepRoughDataset_v2_fourier.npz'


# ---------------------------------------------------------------------------
# NaN interpolation utilities
# ---------------------------------------------------------------------------

def fill_nan_row(iv_row: np.ndarray, K_grid: np.ndarray) -> np.ndarray:
    """
    Fill NaN values in a single maturity row (nK,) by linear interpolation /
    extrapolation along the K axis.  Clipped to [1e-3, 5.0] (0.1% – 500% IV).
    """
    nan_mask = np.isnan(iv_row)
    if not nan_mask.any():
        return iv_row
    valid = ~nan_mask
    if valid.sum() < 2:
        # Degenerate: only 0-1 valid point — fill with neighbour or 0.3
        if valid.sum() == 1:
            iv_row[nan_mask] = iv_row[valid][0]
        else:
            iv_row[:] = 0.3
        return iv_row
    f = interp1d(K_grid[valid], iv_row[valid],
                 kind='linear', fill_value='extrapolate')
    iv_row[nan_mask] = np.clip(f(K_grid[nan_mask]).astype(np.float32), 1e-3, 5.0)
    return iv_row


def fill_nan_surface(iv_surface: np.ndarray, K_grid: np.ndarray) -> np.ndarray:
    """
    Fill NaN in a (nT, nK) IV surface row-by-row (per maturity).
    Modifies in-place and returns the filled array.
    """
    for t in range(iv_surface.shape[0]):
        iv_surface[t] = fill_nan_row(iv_surface[t], K_grid)
    return iv_surface


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print(' Lifted Heston v2 Dataset — Fourier-COS Exact Pricing (GPU)')
    print('=' * 70)
    print(f'  Device      : {device}')
    if device == 'cuda':
        print(f'  GPU         : {torch.cuda.get_device_name(0)}')
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'  VRAM        : {mem_gb:.1f} GB')
    print(f'  N samples   : {N_SAMPLES:,}')
    print(f'  Batch size  : {BATCH_SIZE}')
    print(f'  N_cos       : {N_COS}')
    print(f'  dt          : {1/N_STEPS_PER_UNIT:.4f}  ({N_STEPS_PER_UNIT} steps/unit)')
    print(f'  H fixed     : {H_FIXED}')
    print(f'  T grid      : {T_GRID}')
    print(f'  K grid      : {K_GRID[0]:.2f} to {K_GRID[-1]:.2f}  ({len(K_GRID)} strikes)')
    print(f'  Output      : {OUTPUT_PATH}')
    print()

    # Sobol quasi-random sampling for space-filling uniformity
    n_pow2  = 2 ** int(np.ceil(np.log2(N_SAMPLES)))
    sampler = qmc.Sobol(d=5, scramble=True, seed=42)
    unit_pts = sampler.random(n=n_pow2)[:N_SAMPLES]
    params5  = qmc.scale(unit_pts, BOUNDS_LOWER, BOUNDS_UPPER).astype(np.float64)

    print('  Parameter ranges (Sobol sample):')
    for i, name in enumerate(PARAM_NAMES):
        print(f'    {name:6s}: [{params5[:, i].min():.4f}, {params5[:, i].max():.4f}]')
    print()

    # GPU warmup
    print('  Warming up GPU ...')
    _ = price_batch_gpu(
        params5[:min(64, N_SAMPLES)], T_GRID, K_GRID,
        H_fixed=H_FIXED, N_factors=N_FACTORS, N_cos=N_COS,
        N_steps_per_unit=N_STEPS_PER_UNIT, device=device,
    )
    print('  Warmup done.\n')

    nT, nK = len(T_GRID), len(K_GRID)
    iv_all      = np.full((N_SAMPLES, nT, nK), np.nan, dtype=np.float32)
    nan_mask    = np.zeros((N_SAMPLES, nT, nK), dtype=bool)   # True = valid COS
    n_batches   = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    total_nan   = 0
    t_start     = time.time()

    print(f"  {'Batch':>6} | {'Done':>8} | {'NaN%':>7} | {'Batch t':>8} | {'ETA':>8}")
    print(f"  {'-'*53}")

    for b in range(n_batches):
        s = b * BATCH_SIZE
        e = min(s + BATCH_SIZE, N_SAMPLES)

        t_b0 = time.time()
        iv_batch = price_batch_gpu(
            params5[s:e], T_GRID, K_GRID,
            H_fixed=H_FIXED, N_factors=N_FACTORS, N_cos=N_COS,
            N_steps_per_unit=N_STEPS_PER_UNIT, device=device,
        )                                       # (bsz, nT, nK) float32
        t_b1 = time.time()

        batch_nan_cells = int(np.isnan(iv_batch).sum())
        total_nan      += batch_nan_cells

        # Record valid mask BEFORE filling NaN
        nan_mask[s:e] = ~np.isnan(iv_batch)

        # Fill NaN by linear interpolation along K axis
        for i in range(e - s):
            if np.isnan(iv_batch[i]).any():
                fill_nan_surface(iv_batch[i], K_GRID)

        iv_all[s:e] = iv_batch

        rate    = e / (time.time() - t_start)
        eta_s   = (N_SAMPLES - e) / max(rate, 1e-6)
        eta_str = f'{eta_s/60:.1f}m' if eta_s < 3600 else f'{eta_s/3600:.1f}h'
        nan_pct = batch_nan_cells / ((e - s) * nT * nK) * 100

        print(f"  {b+1:>6}/{n_batches} | {e:>8,} | "
              f"{nan_pct:>6.1f}%  | "
              f"{t_b1 - t_b0:>7.2f}s | {eta_str:>8}")

    t_total  = time.time() - t_start
    total_iv = N_SAMPLES * nT * nK
    nan_frac = total_nan / total_iv

    print(f'\n{"="*70}')
    print(f'  Done in {t_total:.1f}s ({t_total/60:.1f} min)')
    print(f'  Raw NaN rate  : {nan_frac*100:.2f}%  ({total_nan:,}/{total_iv:,} cells)')
    print(f'  After interp  : {np.isnan(iv_all).sum()} NaN remaining')
    print(f'  Speed         : {N_SAMPLES/t_total:.0f} surfaces/sec')

    # Sanity check: no NaN should remain after interpolation
    if np.isnan(iv_all).any():
        remaining = np.isnan(iv_all).sum()
        print(f'  WARNING: {remaining} NaN remain — filling with 0.30 (fallback)')
        iv_all[np.isnan(iv_all)] = 0.30

    # Build output arrays
    params6  = np.column_stack([
        params5.astype(np.float32),
        np.full(N_SAMPLES, H_FIXED, np.float32),
    ])
    iv_flat  = iv_all.reshape(N_SAMPLES, nT * nK)
    dataset  = np.concatenate([params6, iv_flat], axis=1).astype(np.float32)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        dataset=dataset,
        params=params6,
        iv=iv_all,
        nan_mask=nan_mask,    # True = original COS value; False = interpolated
        T_grid=T_GRID,
        K_grid=K_GRID,
    )
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f'\n  Saved: {OUTPUT_PATH}')
    print(f'         shape={dataset.shape}  size={size_mb:.1f} MB')
    print(f'         nan_mask valid fraction = {nan_mask.mean()*100:.2f}%')

    # Spot check
    print('\n  Spot-check (first 5 samples):')
    print(f'  {"params (k,th,s,r,v0)":<35}  {"ATM IV @ T=0.1":>14}  {"ATM IV @ T=1.0":>14}')
    T10_idx = np.argmin(np.abs(T_GRID - 0.1))
    T10x_idx = np.argmin(np.abs(T_GRID - 0.9))
    ATM_idx = np.argmin(np.abs(K_GRID))
    for i in range(min(5, N_SAMPLES)):
        k, th, s, r, v = params5[i, :5]
        desc = f'k={k:.2f} th={th:.3f} s={s:.2f} r={r:.2f} v0={v:.3f}'
        iv01 = iv_all[i, T10_idx, ATM_idx] * 100
        iv10 = iv_all[i, T10x_idx, ATM_idx] * 100
        mask01 = nan_mask[i, T10_idx, ATM_idx]
        print(f'  {desc:<35}  {iv01:>13.2f}%  {iv10:>13.2f}%'
              + ('' if mask01 else '  [interp]'))

    return dataset


if __name__ == '__main__':
    generate()
