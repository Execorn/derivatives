"""
vs_cuda_mc.py — Fourier-COS vs Monte Carlo systematic bias benchmark.

Quantifies the systematic pricing bias introduced by Euler-Maruyama Monte Carlo
(252 steps, O(Δt^0.08) convergence) vs exact Fourier-COS pricing for the
Lifted Rough Heston model.

Expected finding:
  - T=0.1: 5–20bp systematic MC bias (roughness H=0.08 → very slow MC convergence)
  - T=1.0: 1–5bp
  - T=2.0: <1bp (MC is adequate at long maturities)

Usage:
    /home/execorn/programming/derivatives/.venv/bin/python \
        benchmarks/vs_cuda_mc.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from pricing_engine import price_iv_surface

# Grid — must match MC dataset
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)
N_PARAM_COLS = 6    # [kappa, theta, sigma, rho, v0, H]
N_IV_COLS    = 88   # 8 * 11

N_SAMPLES    = 200  # number of MC dataset samples to reprice
N_FACTORS    = 20
N_COS        = 500

MC_DATA_PATH = "data/DeepRoughDataset.npz"


def run_benchmark():
    print("=" * 72)
    print(" Fourier-COS vs CUDA Monte Carlo: Systematic Bias Benchmark")
    print(f" Repricing {N_SAMPLES} samples from MC dataset with exact COS engine")
    print(f" COS config: N_factors={N_FACTORS}, N_cos={N_COS}")
    print("=" * 72)

    # Load MC dataset
    assert os.path.exists(MC_DATA_PATH), f"MC dataset not found: {MC_DATA_PATH}"
    data = np.load(MC_DATA_PATH)["dataset"]   # (N_total, 94)
    N_total = data.shape[0]
    print(f"\n  MC dataset: {N_total:,} samples × {data.shape[1]} columns")

    # Random subsample
    rng = np.random.default_rng(seed=1337)
    idx = rng.choice(N_total, size=N_SAMPLES, replace=False)
    samples = data[idx]                        # (200, 94)

    params_mc = samples[:, :N_PARAM_COLS]      # (200, 6)
    iv_mc     = samples[:, N_PARAM_COLS:].reshape(N_SAMPLES, len(T_GRID), len(K_GRID))

    # Reprice with Fourier-COS
    print(f"\n  Repricing with Fourier-COS (N_factors={N_FACTORS}, N_cos={N_COS}) ...")
    iv_cos = np.full_like(iv_mc, np.nan)

    t_start = time.time()
    n_failed = 0
    for i in range(N_SAMPLES):
        p = params_mc[i]
        pdict = dict(kappa=p[0], theta=p[1], sigma=p[2],
                     rho=p[3], v0=p[4], H=p[5])
        try:
            iv_cos[i] = price_iv_surface(
                pdict, T_GRID, K_GRID,
                N_factors=N_FACTORS, N_cos=N_COS,
            )
        except Exception as e:
            n_failed += 1
            continue

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (N_SAMPLES - i - 1)
            print(f"    {i+1}/{N_SAMPLES}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    t_total = time.time() - t_start
    print(f"  Done in {t_total:.1f}s  ({N_SAMPLES - n_failed}/{N_SAMPLES} successful)")

    # Compute per-maturity absolute IV differences
    # Only use samples where both MC and COS are valid
    valid = np.isfinite(iv_cos) & np.isfinite(iv_mc) & (iv_mc > 0) & (iv_cos > 0)

    print(f"\n  {'Maturity':>10}  {'Mean|Err|(bp)':>14}  {'Max|Err|(bp)':>14}  "
          f"{'Median|Err|(bp)':>16}  {'Valid%':>7}")
    print(f"  {'-'*68}")

    results = {}
    for t_idx, T in enumerate(T_GRID):
        v = valid[:, t_idx, :]              # (N, K) bool
        if v.sum() < 10:
            print(f"  {T:>10.1f}  {'N/A (too few valid)':>14}")
            continue
        err_iv = np.abs(iv_cos[:, t_idx, :] - iv_mc[:, t_idx, :])
        err_valid = err_iv[v]

        mean_bp = err_valid.mean() * 1e4
        max_bp  = err_valid.max()  * 1e4
        med_bp  = np.median(err_valid) * 1e4
        pct_valid = v.mean() * 100

        results[T] = {"mean_bp": mean_bp, "max_bp": max_bp, "med_bp": med_bp}
        print(f"  {T:>10.2f}  {mean_bp:>14.2f}  {max_bp:>14.2f}  {med_bp:>16.2f}  {pct_valid:>6.1f}%")

    # Summary
    print(f"\n  {'Key findings':}")
    if 0.1 in results:
        r = results[0.1]
        ok = "✓ (5-20bp bias confirmed)" if 5 <= r['mean_bp'] <= 20 else f"({r['mean_bp']:.1f}bp)"
        print(f"    T=0.1  mean bias = {r['mean_bp']:6.2f}bp  {ok}")
    if 1.0 in results:
        r = results[1.0]
        print(f"    T=1.0  mean bias = {r['mean_bp']:6.2f}bp")
    if 2.0 in results:
        r = results[2.0]
        ok = "✓ (<1bp confirmed)" if r['mean_bp'] < 1.0 else f"({r['mean_bp']:.1f}bp)"
        print(f"    T=2.0  mean bias = {r['mean_bp']:6.2f}bp  {ok}")

    # Global stats
    err_all = np.abs(iv_cos - iv_mc)
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
        f.write(f"N_samples repriced: {N_SAMPLES - n_failed}/{N_SAMPLES}\n")
        f.write(f"COS config: N_factors={N_FACTORS}, N_cos={N_COS}\n\n")
        f.write(f"{'Maturity':>10}  {'Mean(bp)':>10}  {'Max(bp)':>10}  {'Median(bp)':>12}\n")
        f.write("-" * 48 + "\n")
        for T, r in results.items():
            f.write(f"{T:>10.2f}  {r['mean_bp']:>10.2f}  {r['max_bp']:>10.2f}  "
                    f"{r['med_bp']:>12.2f}\n")
        f.write(f"\nGlobal mean: {err_all_valid.mean()*1e4:.2f}bp\n")
        f.write(f"Global max:  {err_all_valid.max()*1e4:.2f}bp\n")
    print(f"\n  Saved: {out_path}")

    return results


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    run_benchmark()
