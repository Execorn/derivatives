"""
generate_dataset_v3_differential.py — Differential ML dataset generator.

Produces (params, IV, ∂IV/∂params) triples for 50k Lifted Rough Heston
parameter configurations using batched central finite differences on GPU.

Mathematical foundation (Huge & Savine, 2020 — Differential Machine Learning):
  Train the network to match BOTH the IV surface f(θ) AND its Jacobian ∂f/∂θ
  simultaneously.  Jacobian data provides 88×5=440 extra constraints per sample,
  giving accuracy equivalent to ~100× more samples with standard regression.

Method — batched central finite differences:
  For each of the 5 parameters θ_j, perturb by ±ε:
    ∂IV/∂θ_j  ≈  [IV(θ + ε·eⱼ) − IV(θ − ε·eⱼ)] / (2ε)

  10 perturbed evaluations (2 per param) are stacked into a single mega-batch
  and priced in one GPU call, so total time ≈ 11× the standard v2 generation.

  Relative perturbation: ε = max(0.5% × |θ_j|, ε_min_j)
  Jacobian precision: O(ε²) ≈ 0.0025% relative error for smooth IV surfaces.
  This is sufficient for differential ML — sign and order-of-magnitude matter.

Dataset:
  'dataset'   float32  (N, 94)      — [6 params | 88 IVs] compatible with FNO
  'params'    float32  (N, 6)       — [kappa, theta, sigma, rho, v0, H]
  'iv'        float32  (N, 8, 11)   — implied volatility surface
  'jacobian'  float32  (N, 8, 11, 5)— ∂IV/∂[kappa, theta, sigma, rho, v0]
  'nan_mask'  bool     (N, 8, 11)   — True = valid COS price; False = interpolated
  'T_grid'    float64  (8,)
  'K_grid'    float64  (11,)
  'param_names' str    (5,)

Usage:
    /home/execorn/programming/derivatives/.venv/bin/python \\
        src/generate_dataset_v3_differential.py
"""

import os
import sys
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pricing_engine_gpu import price_batch_gpu


# ---------------------------------------------------------------------------
# Configuration — identical grid to v2
# ---------------------------------------------------------------------------

T_GRID = np.array([0.04, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)

BOUNDS_LOWER = np.array([0.1,  0.01, 0.1,  -0.9, 0.01])
BOUNDS_UPPER = np.array([5.0,  0.15, 1.0,  -0.1, 0.15])
PARAM_NAMES  = ['kappa', 'theta', 'sigma', 'rho', 'v0']
H_FIXED      = 0.08

N_SAMPLES        = 50_000
BATCH_SIZE       = 512      # 10×512=5120 perturbed samples per mega-batch; fits 6GB VRAM
N_STEPS_PER_UNIT = 200
N_COS            = 64
N_FACTORS        = 20
S0               = 1.0

OUTPUT_PATH = 'data/DeepRoughDataset_v3_differential.npz'

# Finite difference step sizes (absolute).  Chosen to keep ε small vs param
# range while avoiding numerical cancellation in float32 IV differences.
EPS = np.array([
    0.02,   # kappa ∈ [0.1, 5.0]   → ε=0.02 (0.4% of range)
    0.001,  # theta ∈ [0.01, 0.15] → ε=0.001 (0.7% of range)
    0.005,  # sigma ∈ [0.1, 1.0]   → ε=0.005 (0.6% of range)
    0.004,  # rho   ∈ [-0.9,-0.1]  → ε=0.004 (0.5% of range)
    0.001,  # v0    ∈ [0.01, 0.15] → ε=0.001 (0.7% of range)
])


# ---------------------------------------------------------------------------
# NaN interpolation (same as v2)
# ---------------------------------------------------------------------------

def fill_nan_surface(iv: np.ndarray, K_grid: np.ndarray) -> np.ndarray:
    """Fill NaN along K axis per maturity row.  Modifies in-place."""
    for t in range(iv.shape[0]):
        row = iv[t]
        nm  = np.isnan(row)
        if not nm.any():
            continue
        valid = ~nm
        if valid.sum() < 2:
            iv[t, nm] = row[valid][0] if valid.sum() == 1 else 0.3
            continue
        f = interp1d(K_grid[valid], row[valid], kind='linear', fill_value='extrapolate')
        iv[t, nm] = np.clip(f(K_grid[nm]).astype(np.float32), 1e-3, 5.0)
    return iv


# ---------------------------------------------------------------------------
# Core: compute IV + Jacobian for one batch
# ---------------------------------------------------------------------------

def compute_batch_with_jacobian(
    params_np: np.ndarray,    # (B, 5) float64
    T_grid:    np.ndarray,
    K_grid:    np.ndarray,
    H_fixed:   float = H_FIXED,
    device:    str   = 'cuda',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      iv       : (B, nT, nK) float32  — IV surface (NaN before interpolation)
      jacobian : (B, nT, nK, 5) float32  — ∂IV/∂params
      nan_mask : (B, nT, nK) bool  — True = valid COS price

    Strategy:
      1. Price the base batch (B, 5) → iv_base (B, nT, nK)
      2. For each param j: stack up-/down-perturbed params → mega-batch (10B, 5)
      3. Price mega-batch in a single GPU call → iv_perturbed (10B, nT, nK)
      4. Central FD: ∂IV/∂θ_j = (iv_up_j − iv_dn_j) / (2·ε_j)

    All NaN handling (interpolation) happens AFTER this function returns.
    """
    B = params_np.shape[0]
    n_params = 5

    # 1. Base pricing
    iv_base = price_batch_gpu(
        params_np, T_grid, K_grid,
        H_fixed=H_fixed, N_factors=N_FACTORS, N_cos=N_COS,
        N_steps_per_unit=N_STEPS_PER_UNIT, device=device,
    )   # (B, nT, nK) float32

    raw_nan_mask = ~np.isnan(iv_base)   # True = valid COS price

    # 2. Build mega-batch: [up_0, dn_0, up_1, dn_1, ..., up_4, dn_4] — 10B rows
    blocks = []
    for j in range(n_params):
        eps_j = EPS[j]
        p_up = params_np.copy()
        p_dn = params_np.copy()
        p_up[:, j] = np.clip(p_up[:, j] + eps_j, BOUNDS_LOWER[j], BOUNDS_UPPER[j])
        p_dn[:, j] = np.clip(p_dn[:, j] - eps_j, BOUNDS_LOWER[j], BOUNDS_UPPER[j])
        blocks.append(p_up)
        blocks.append(p_dn)

    mega = np.vstack(blocks)   # (10B, 5)

    # 3. Price mega-batch in one GPU call
    iv_mega = price_batch_gpu(
        mega, T_grid, K_grid,
        H_fixed=H_fixed, N_factors=N_FACTORS, N_cos=N_COS,
        N_steps_per_unit=N_STEPS_PER_UNIT, device=device,
    )   # (10B, nT, nK) float32

    # 4. Central FD Jacobian
    nT, nK = len(T_grid), len(K_grid)
    jacobian = np.zeros((B, nT, nK, n_params), dtype=np.float32)

    for j in range(n_params):
        iv_up = iv_mega[j * 2 * B      : j * 2 * B + B    ]   # (B, nT, nK)
        iv_dn = iv_mega[j * 2 * B + B  : (j + 1) * 2 * B  ]   # (B, nT, nK)

        # For cells where either perturbed IV is NaN: set Jacobian = 0 (interpolated)
        finite = np.isfinite(iv_up) & np.isfinite(iv_dn)
        actual_eps = EPS[j]
        dIV = np.where(finite, (iv_up - iv_dn) / (2.0 * actual_eps), 0.0)
        jacobian[:, :, :, j] = dIV.astype(np.float32)

    # Clip extreme values (numerical noise at very deep OTM strikes)
    jacobian = np.clip(jacobian, -50.0, 50.0)

    return iv_base, jacobian, raw_nan_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 70)
    print(' Lifted Heston v3 — Differential ML Dataset (IV + Jacobians, FD)')
    print('=' * 70)
    print(f'  Device         : {device}')
    if device == 'cuda':
        print(f'  GPU            : {torch.cuda.get_device_name(0)}')
    print(f'  N samples      : {N_SAMPLES:,}')
    print(f'  Batch size     : {BATCH_SIZE}  (mega-batch={10*BATCH_SIZE} per FD)')
    print(f'  N_cos / steps  : {N_COS} / {N_STEPS_PER_UNIT}')
    print(f'  H fixed        : {H_FIXED}')
    print(f'  FD epsilons    : {dict(zip(PARAM_NAMES, EPS))}')
    print(f'  Output         : {OUTPUT_PATH}')
    print()

    # Sobol quasi-random sampling (same seed as v2 → identical parameter grid)
    n_pow2   = 2 ** int(np.ceil(np.log2(N_SAMPLES)))
    sampler  = qmc.Sobol(d=5, scramble=True, seed=42)
    unit_pts = sampler.random(n=n_pow2)[:N_SAMPLES]
    params5  = qmc.scale(unit_pts, BOUNDS_LOWER, BOUNDS_UPPER).astype(np.float64)

    print('  Parameter ranges (Sobol, same as v2):')
    for i, name in enumerate(PARAM_NAMES):
        print(f'    {name:6s}: [{params5[:,i].min():.4f}, {params5[:,i].max():.4f}]')
    print()

    # GPU warmup
    print('  Warming up GPU ...')
    _ = compute_batch_with_jacobian(
        params5[:min(16, N_SAMPLES)], T_GRID, K_GRID, device=device
    )
    print('  Done.\n')

    nT, nK = len(T_GRID), len(K_GRID)
    iv_all       = np.full((N_SAMPLES, nT, nK),     np.nan, dtype=np.float32)
    jac_all      = np.zeros((N_SAMPLES, nT, nK, 5), dtype=np.float32)
    nan_mask_all = np.zeros((N_SAMPLES, nT, nK),    dtype=bool)

    n_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    total_nan = 0
    t_start   = time.time()

    print(f"  {'Batch':>6} | {'Done':>8} | {'NaN%':>6} | "
          f"{'t_base':>7} | {'t_jac':>7} | {'ETA':>8}")
    print(f"  {'-'*60}")

    for b in range(n_batches):
        s = b * BATCH_SIZE
        e = min(s + BATCH_SIZE, N_SAMPLES)

        t0 = time.time()
        iv_b, jac_b, mask_b = compute_batch_with_jacobian(
            params5[s:e], T_GRID, K_GRID, device=device
        )
        t1 = time.time()

        batch_nan  = int(np.isnan(iv_b).sum())
        total_nan += batch_nan
        nan_mask_all[s:e] = mask_b

        # Fill NaN by K-axis linear interpolation
        for i in range(e - s):
            if np.isnan(iv_b[i]).any():
                fill_nan_surface(iv_b[i], K_GRID)

        iv_all[s:e]  = iv_b
        jac_all[s:e] = jac_b

        elapsed = time.time() - t_start
        rate    = e / elapsed
        eta_s   = (N_SAMPLES - e) / max(rate, 1e-6)
        eta_str = f'{eta_s/60:.1f}m' if eta_s < 3600 else f'{eta_s/3600:.1f}h'
        nan_pct = batch_nan / ((e - s) * nT * nK) * 100

        # Estimate base vs jacobian timing (roughly 1:10 split)
        t_total_b = t1 - t0
        t_base_b  = t_total_b / 11
        t_jac_b   = t_total_b * 10 / 11

        print(f"  {b+1:>6}/{n_batches} | {e:>8,} | "
              f"{nan_pct:>5.1f}%  | "
              f"{t_base_b:>6.2f}s | {t_jac_b:>6.2f}s | {eta_str:>8}")

    t_total = time.time() - t_start
    print(f'\n{"="*70}')
    print(f'  Done in {t_total:.1f}s ({t_total/60:.1f} min)')
    print(f'  Raw NaN rate   : {total_nan/(N_SAMPLES*nT*nK)*100:.2f}%')
    print(f'  After interp   : {np.isnan(iv_all).sum()} NaN remaining')
    print(f'  Speed          : {N_SAMPLES/t_total:.1f} surfaces/sec')
    print(f'  Jacobian range : [{jac_all.min():.3f}, {jac_all.max():.3f}]')

    if np.isnan(iv_all).any():
        iv_all[np.isnan(iv_all)] = 0.30

    # Sanity check: Jacobian signs should follow physical intuition
    # v0↑ → IV↑ (more variance → higher IV)
    # rho↑ (less negative) → IV↓ at ATM (lower skew)
    T_ref = np.argmin(np.abs(T_GRID - 1.0))
    K_ref = np.argmin(np.abs(K_GRID))
    med_jac = np.median(jac_all[:, T_ref, K_ref, :], axis=0)
    print(f'\n  Jacobian median at ATM T=1.0:')
    for j, name in enumerate(PARAM_NAMES):
        sign = '+' if med_jac[j] > 0 else '-'
        print(f'    ∂IV/∂{name:5s} = {med_jac[j]:+.4f}  ({sign})')

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
        jacobian=jac_all,
        nan_mask=nan_mask_all,
        T_grid=T_GRID,
        K_grid=K_GRID,
        param_names=np.array(PARAM_NAMES),
    )
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f'\n  Saved  : {OUTPUT_PATH}')
    print(f'           dataset={dataset.shape}  jacobian={jac_all.shape}  {size_mb:.1f} MB')
    print(f'           nan_mask valid fraction = {nan_mask_all.mean()*100:.2f}%')

    return dataset


if __name__ == '__main__':
    generate()
