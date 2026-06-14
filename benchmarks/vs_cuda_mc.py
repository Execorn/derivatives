"""
vs_cuda_mc.py — Dataset v1 quality benchmark: buggy COS vs corrected GPU COS.

Quantifies the systematic pricing error in the v1 training dataset
(DeepRoughDataset.npz, generated 2026-06-10) caused by the Bernstein
normalisation bug in the CPU COS engine:

  BUG (pre-2026-06-11): c_i = x_i^{-(H+0.5)}  with sum(c) ≈ 26 (unnormalised)
  FIX (2026-06-11+):    c_i /= sum(c)          so sum(c) = 1  (normalised)

With unnormalised c the quadratic Riccati term is amplified by sum(c)^2 ≈ 676,
blowing up the vol-of-vol coupling and producing wildly wrong IV surfaces.
The GPU COS engine (pricing_engine_gpu.py) was always normalised.

GPU acceleration (2026-06-14):
  Uses pricing_engine_gpu.price_batch_gpu with H_batch for all 200 samples
  in a single B=200 GPU kernel. Runtime: ~0.7s (vs ~30 min for serial CPU).

Expected finding (confirmed 2026-06-14):
  - Global mean error: ~1900-2000bp  (v1 dataset is corrupted)
  - Errors are UNIFORM across maturities (not maturity-dependent)
  - This motivates the v2 COS dataset (DeepRoughDataset_v2_fourier.npz)
    generated with the fixed, normalised GPU engine.

Thesis context: §4.1 — justification for discarding v1 and regenerating with
the corrected Fourier-COS pipeline.

Usage:
    /home/execorn/programming/derivatives/.venv/bin/python \
        benchmarks/vs_cuda_mc.py
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from pricing_engine_gpu import price_batch_gpu

# Grid — must match MC dataset
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)
N_PARAM_COLS = 6    # [kappa, theta, sigma, rho, v0, H]
N_IV_COLS    = 88   # 8 * 11

N_SAMPLES    = 200  # number of MC dataset samples to reprice
N_FACTORS    = 40   # 20→40 (2026-06-15): matches v2 dataset config
# N_COS=128 matches v2 dataset generation (N_cos=64 gives 264bp error at
# ATM/T=0.1 for rough Heston H=0.08 — CF decays too slowly for 64 terms).
N_COS        = 128  # 64→128 (2026-06-15)

# Dataset v1 — generated 2026-06-10 with buggy unnormalised Bernstein CPU engine
MC_DATA_PATH = "data/DeepRoughDataset.npz"

# H rounding precision for GPU batch grouping.
# price_batch_gpu takes a single H_fixed per call; we group samples by H rounded
# to 3 decimal places (≤0.001 error in H → negligible pricing error).
H_ROUND_DECIMALS = 3


def _reprice_batch_variable_h(
    params_np: np.ndarray,
    T_grid:    np.ndarray,
    K_grid:    np.ndarray,
    device:    str = "cuda",
    N_factors: int = 20,
    N_cos:     int = 64,
) -> tuple:
    """
    GPU-batched repricing that handles variable H across samples.

    price_batch_gpu takes a single H_fixed per call, so samples are grouped
    by rounded H and one GPU call is issued per group.

    Returns (iv_out, n_failed):
        iv_out   : (B, nT, nK) float32 — NaN for failed cells
        n_failed : int
    """
    B        = params_np.shape[0]
    iv_out   = np.full((B, len(T_grid), len(K_grid)), np.nan, dtype=np.float32)
    n_failed = 0

    H_rounded = np.round(params_np[:, 5], H_ROUND_DECIMALS)
    unique_H  = np.unique(H_rounded)

    for H_val in unique_H:
        mask         = H_rounded == H_val
        group_idx    = np.where(mask)[0]
        group_params = params_np[mask, :5]   # (G, 5): kappa,theta,sigma,rho,v0
        try:
            iv_group = price_batch_gpu(
                group_params, T_grid, K_grid,
                H_fixed=float(H_val),
                N_factors=N_factors,
                N_cos=N_cos,
                device=device,
            )                                # (G, nT, nK) float32
            iv_out[group_idx] = iv_group
        except Exception as exc:
            n_failed += len(group_idx)
            print(f"  WARN: GPU pricing failed for H={H_val:.3f} "
                  f"({len(group_idx)} samples): {exc}")

    return iv_out, n_failed


def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print(" Dataset v1 Quality Benchmark: Buggy CPU COS vs Corrected GPU COS")
    print(f" Repricing {N_SAMPLES} samples from v1 dataset with fixed GPU COS engine")
    print(f" COS config: N_factors={N_FACTORS}, N_cos={N_COS}")
    print(f" GPU device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (CPU fallback)"))
    print("=" * 72)
    print()
    print("  NOTE: v1 dataset (DeepRoughDataset.npz) was generated 2026-06-10")
    print("  with the BUGGY CPU COS engine (unnormalised Bernstein c, sum(c)≈26).")
    print("  GPU COS engine (pricing_engine_gpu) has always used normalised c.")
    print("  Errors below show the magnitude of the normalisation bug impact.")
    print()

    # Load MC dataset
    assert os.path.exists(MC_DATA_PATH), f"MC dataset not found: {MC_DATA_PATH}"
    data    = np.load(MC_DATA_PATH)["dataset"]   # (N_total, 94)
    N_total = data.shape[0]
    print(f"\n  MC dataset: {N_total:,} samples × {data.shape[1]} columns")

    # Random subsample
    rng     = np.random.default_rng(seed=1337)
    idx     = rng.choice(N_total, size=N_SAMPLES, replace=False)
    samples = data[idx]                           # (200, 94)

    params_mc = samples[:, :N_PARAM_COLS]         # (200, 6)
    iv_mc     = samples[:, N_PARAM_COLS:].reshape(N_SAMPLES, len(T_GRID), len(K_GRID))

    H_vals = params_mc[:, 5]
    print(f"\n  H range in subsample: [{H_vals.min():.3f}, {H_vals.max():.3f}]")

    # Single GPU batch call: all 200 samples in one B=200 kernel
    # H_batch enables per-sample Bernstein c=(B,N) instead of 103 separate calls
    print(f"\n  Repricing with GPU Fourier-COS (N_factors={N_FACTORS}, N_cos={N_COS}) ...")
    t_start  = time.perf_counter()
    try:
        iv_cos = price_batch_gpu(
            params_mc[:, :5], T_GRID, K_GRID,
            H_batch=params_mc[:, 5],
            N_factors=N_FACTORS, N_cos=N_COS,
            device=device,
        )                                     # (200, 8, 11) float32
        n_failed = 0
    except Exception as exc:
        print(f"  ERROR: {exc}")
        iv_cos   = np.full_like(iv_mc, np.nan)
        n_failed = N_SAMPLES
    t_total = time.perf_counter() - t_start
    print(f"  Done in {t_total:.2f}s  ({N_SAMPLES - n_failed}/{N_SAMPLES} successful)"
          f"  [{t_total / N_SAMPLES * 1000:.1f} ms/sample]")

    # Per-maturity absolute IV differences (only valid cells)
    valid = (np.isfinite(iv_cos) & np.isfinite(iv_mc)
             & (iv_mc > 0) & (iv_cos > 0))

    print(f"\n  {'Maturity':>10}  {'Mean|Err|(bp)':>14}  {'Max|Err|(bp)':>14}  "
          f"{'Median|Err|(bp)':>16}  {'Valid%':>7}")
    print(f"  {'-'*68}")

    results = {}
    for t_idx, T in enumerate(T_GRID):
        v = valid[:, t_idx, :]
        if v.sum() < 10:
            print(f"  {T:>10.1f}  {'N/A (too few valid)':>14}")
            continue
        err_iv    = np.abs(iv_cos[:, t_idx, :] - iv_mc[:, t_idx, :])
        err_valid = err_iv[v]

        mean_bp = err_valid.mean() * 1e4
        max_bp  = err_valid.max()  * 1e4
        med_bp  = np.median(err_valid) * 1e4
        pct     = v.mean() * 100

        results[T] = {"mean_bp": mean_bp, "max_bp": max_bp, "med_bp": med_bp}
        print(f"  {T:>10.2f}  {mean_bp:>14.2f}  {max_bp:>14.2f}  "
              f"{med_bp:>16.2f}  {pct:>6.1f}%")

    # Summary
    print(f"\n  Key findings (v1 dataset normalisation bug impact):")
    if 0.1 in results:
        r  = results[0.1]
        ok = "✓ bug confirmed (>>20bp)" if r['mean_bp'] > 100 else f"({r['mean_bp']:.1f}bp — unexpectedly small)"
        print(f"    T=0.1  mean error = {r['mean_bp']:8.2f}bp  {ok}")
    if 1.0 in results:
        r = results[1.0]
        print(f"    T=1.0  mean error = {r['mean_bp']:8.2f}bp")
    if 2.0 in results:
        r  = results[2.0]
        ok = "✓ bug confirmed (>>20bp)" if r['mean_bp'] > 100 else f"({r['mean_bp']:.1f}bp — unexpectedly small)"
        print(f"    T=2.0  mean error = {r['mean_bp']:8.2f}bp  {ok}")

    # Global statistics
    err_all       = np.abs(iv_cos - iv_mc)
    err_all_valid = err_all[valid]
    print(f"\n  Global statistics ({err_all_valid.size} valid cells):")
    print(f"    Mean absolute IV error : {err_all_valid.mean()*1e4:.2f} bp")
    print(f"    Max  absolute IV error : {err_all_valid.max()*1e4:.2f} bp")
    print(f"    Median IV error        : {np.median(err_all_valid)*1e4:.2f} bp")
    print(f"    P95 IV error           : {np.percentile(err_all_valid, 95)*1e4:.2f} bp")
    print("=" * 72)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "mc_vs_cos_results.txt")
    with open(out_path, "w") as f:
        f.write("Fourier-COS vs Monte Carlo Benchmark Results\n")
        f.write("=" * 72 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"N_samples repriced: {N_SAMPLES - n_failed}/{N_SAMPLES}\n")
        f.write(f"COS config: N_factors={N_FACTORS}, N_cos={N_COS}\n")
        f.write(f"Total time: {t_total:.2f}s  ({t_total/N_SAMPLES*1000:.1f} ms/sample)\n\n")
        f.write(f"{'Maturity':>10}  {'Mean(bp)':>10}  {'Max(bp)':>10}  {'Median(bp)':>12}\n")
        f.write("-" * 48 + "\n")
        for T, r in results.items():
            f.write(f"{T:>10.2f}  {r['mean_bp']:>10.2f}  {r['max_bp']:>10.2f}  "
                    f"{r['med_bp']:>12.2f}\n")
        if err_all_valid.size > 0:
            f.write(f"\nGlobal mean: {err_all_valid.mean()*1e4:.2f}bp\n")
            f.write(f"Global max:  {err_all_valid.max()*1e4:.2f}bp\n")
    print(f"\n  Saved: {out_path}")

    return results


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    run_benchmark()
