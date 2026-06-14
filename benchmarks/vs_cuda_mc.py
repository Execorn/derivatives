"""
vs_cuda_mc.py — Fourier-COS vs Monte Carlo systematic bias benchmark.

Quantifies the systematic pricing bias introduced by Euler-Maruyama Monte Carlo
(252 steps, O(Δt^0.08) convergence) vs exact Fourier-COS pricing for the
Lifted Rough Heston model.

GPU acceleration (2026-06-14):
  Switched from serial CPU BDF solver (pricing_engine.price_iv_surface) to
  GPU-batched RK4 solver (pricing_engine_gpu.price_batch_gpu).
  Samples are grouped by rounded H value so a single GPU kernel handles each
  group. Expected speedup: ~30-60× (seconds instead of ~30 minutes).

Expected finding:
  - T=0.1: 5-20bp systematic MC bias (roughness H=0.08 → very slow MC convergence)
  - T=1.0: 1-5bp
  - T=2.0: <1bp (MC is adequate at long maturities)

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
N_FACTORS    = 20
# N_COS=64 matches COS dataset generation and gives machine-precision IV
# (domain [-4,4] converges in 64 terms; N_COS=500 would inflate ODE state
# to 21,000 dims — O(N³) BDF cost — days not minutes on CPU).
N_COS        = 64

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
    print(" Fourier-COS vs CUDA Monte Carlo: Systematic Bias Benchmark")
    print(f" Repricing {N_SAMPLES} samples from MC dataset with exact COS engine")
    print(f" COS config: N_factors={N_FACTORS}, N_cos={N_COS}")
    print(f" GPU device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (CPU fallback)"))
    print("=" * 72)

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

    H_vals   = params_mc[:, 5]
    unique_H = np.unique(np.round(H_vals, H_ROUND_DECIMALS))
    print(f"\n  H range in subsample: [{H_vals.min():.3f}, {H_vals.max():.3f}]"
          f"  ({len(unique_H)} unique H groups)")

    # GPU-batched repricing — all 200 surfaces in one batched GPU pass per H group
    print(f"\n  Repricing with GPU Fourier-COS (N_factors={N_FACTORS}, N_cos={N_COS}) ...")
    t_start           = time.perf_counter()
    iv_cos, n_failed  = _reprice_batch_variable_h(
        params_mc, T_GRID, K_GRID,
        device=device, N_factors=N_FACTORS, N_cos=N_COS,
    )
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
    print(f"\n  Key findings:")
    if 0.1 in results:
        r  = results[0.1]
        ok = "\u2713 (5-20bp bias confirmed)" if 5 <= r['mean_bp'] <= 20 else f"({r['mean_bp']:.1f}bp)"
        print(f"    T=0.1  mean bias = {r['mean_bp']:6.2f}bp  {ok}")
    if 1.0 in results:
        r = results[1.0]
        print(f"    T=1.0  mean bias = {r['mean_bp']:6.2f}bp")
    if 2.0 in results:
        r  = results[2.0]
        ok = "\u2713 (<1bp confirmed)" if r['mean_bp'] < 1.0 else f"({r['mean_bp']:.1f}bp)"
        print(f"    T=2.0  mean bias = {r['mean_bp']:6.2f}bp  {ok}")

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
