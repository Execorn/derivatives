"""
convergence_N_factors.py — Bernstein factor convergence study (GPU, exp. midpoint).

Measures how IV error decays as N_factors increases, using N=128 as the
numerical reference. Shows that N=40 achieves <1bp global error.

Usage:
    .venv/bin/python benchmarks/convergence_N_factors.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from pricing_engine_gpu import price_batch_gpu

# Fixed test parameters (representative Rough Heston point)
TEST_PARAMS = np.array([[1.0, 0.08, 0.5, -0.5, 0.08]], dtype=np.float64)

T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)

N_VALUES  = [5, 10, 20, 40, 80]   # tested values; N=128 is the reference
N_REF     = 128                    # high-accuracy reference
N_COS     = 128                    # fixed; matches v2 dataset generation
DEVICE    = 'cuda'


def run_convergence():
    results = {}

    print("=" * 72)
    print(" Bernstein Factor Convergence: N-factor Study (GPU, exponential midpoint)")
    print(" Fixed params: κ=1.0, θ=0.08, σ=0.5, ρ=-0.5, v₀=0.08, H=0.08")
    print(f" N_cos={N_COS} (fixed)  |  Reference: N={N_REF}")
    print("=" * 72)

    # Compute reference with N=N_REF
    print(f"\n  Computing N={N_REF} reference surface ...", end=" ", flush=True)
    t0 = time.time()
    iv_ref = price_batch_gpu(TEST_PARAMS, T_GRID, K_GRID,
                             H_fixed=0.08, N_factors=N_REF, N_cos=N_COS,
                             device=DEVICE)[0]          # (8, 11)
    t_ref = time.time() - t0
    print(f"done in {t_ref:.2f}s")

    print(f"\n  {'N':>5}  {'T=0.1 MaxErr':>14}  {'T=1.0 MaxErr':>14}  "
          f"{'T=2.0 MaxErr':>14}  {'Global MaxErr':>14}  {'Time':>8}")
    print(f"  {'-'*80}")

    for N in N_VALUES:
        t0 = time.time()
        iv_N = price_batch_gpu(TEST_PARAMS, T_GRID, K_GRID,
                               H_fixed=0.08, N_factors=N, N_cos=N_COS,
                               device=DEVICE)[0]        # (8, 11)
        t_N = time.time() - t0

        err = np.abs(iv_N - iv_ref)

        t01 = np.argmin(np.abs(T_GRID - 0.1))
        t10 = np.argmin(np.abs(T_GRID - 1.0))
        t20 = np.argmin(np.abs(T_GRID - 2.0))

        err_t01    = np.nanmax(err[t01])
        err_t10    = np.nanmax(err[t10])
        err_t20    = np.nanmax(err[t20])
        err_global = np.nanmax(err)

        bp01 = err_t01 * 10000
        bp10 = err_t10 * 10000
        bp20 = err_t20 * 10000
        bp_g = err_global * 10000

        results[N] = {
            "err_t01": err_t01, "err_t10": err_t10,
            "err_t20": err_t20, "err_global": err_global,
            "time": t_N, "iv": iv_N,
        }

        flag = ""
        if N < 40:
            flag = " ✓" if bp01 < 1.0 else " ✗ (>1bp)"
        elif N == 40:
            flag = " ← PRODUCTION"

        print(f"  {N:>5}  {bp01:>12.4f}bp  {bp10:>12.4f}bp  "
              f"{bp20:>12.4f}bp  {bp_g:>12.4f}bp  {t_N:>7.3f}s{flag}")

    print(f"  {N_REF:>5}  {'0.0000bp':>14}  {'0.0000bp':>14}  "
          f"{'0.0000bp':>14}  {'0.0000bp':>14}  {t_ref:>7.3f}s  ← REFERENCE")

    n40 = results.get(40, {})
    n20 = results.get(20, {})
    print(f"\n  Conclusion:")
    if n40:
        print(f"    N=40 global max error   : {n40['err_global']*10000:.4f} bp  "
              f"({('<1bp ✓' if n40['err_global'] < 1e-4 else '>1bp ✗')})")
    if n20:
        print(f"    N=20 global max error   : {n20['err_global']*10000:.4f} bp")
    print("=" * 72)

    out_path = os.path.join(os.path.dirname(__file__), "convergence_results.txt")
    with open(out_path, "w") as f:
        f.write("Bernstein Factor Convergence Results (GPU, exponential midpoint)\n")
        f.write("=" * 72 + "\n")
        f.write(f"Reference: N={N_REF}  N_cos={N_COS}\n")
        f.write(f"Test params: kappa=1.0, theta=0.08, sigma=0.5, rho=-0.5, v0=0.08, H=0.08\n\n")
        f.write(f"{'N':>5}  {'T=0.1 MaxErr(bp)':>18}  {'T=1.0 MaxErr(bp)':>18}  "
                f"{'Global MaxErr(bp)':>18}  {'Time(s)':>8}\n")
        f.write("-" * 75 + "\n")
        for N in N_VALUES:
            r = results[N]
            f.write(f"{N:>5}  {r['err_t01']*1e4:>18.4f}  {r['err_t10']*1e4:>18.4f}  "
                    f"{r['err_global']*1e4:>18.4f}  {r['time']:>8.3f}\n")
        f.write(f"{N_REF:>5}  {'0.0000':>18}  {'0.0000':>18}  {'0.0000':>18}  "
                f"{t_ref:>8.3f}  (reference)\n")
    print(f"\n  Saved: {out_path}")

    return results


if __name__ == "__main__":
    run_convergence()
