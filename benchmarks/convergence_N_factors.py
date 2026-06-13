"""
convergence_N_factors.py — Bernstein factor convergence study.

Shows that N=20 factors achieves <1bp IV error vs the N=40 reference,
validating the choice of N=20 as the default for the Lifted Rough Heston engine.

Usage:
    /home/execorn/programming/derivatives/.venv/bin/python \
        benchmarks/convergence_N_factors.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from pricing_engine import price_iv_surface

# Fixed test parameters (representative Rough Heston point)
TEST_PARAMS = dict(kappa=1.0, theta=0.08, sigma=0.5, rho=-0.5, v0=0.08, H=0.08)

T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)

N_VALUES = [5, 10, 20, 40]
N_COS    = 128   # Fourier modes — same in all runs for isolation (engine default)


def run_convergence():
    results = {}

    print("=" * 68)
    print(" Bernstein Factor Convergence: N-factor Study")
    print(" Fixed params: κ=1.0, θ=0.08, σ=0.5, ρ=-0.5, v₀=0.08, H=0.08")
    print(" N_cos=500 (fixed); engine default")
    print("=" * 68)

    # Compute reference with N=40
    print("\n  Computing N=40 reference surface ...", end=" ", flush=True)
    t0 = time.time()
    iv_ref = price_iv_surface(
        TEST_PARAMS, T_GRID, K_GRID,
        N_factors=40, N_cos=N_COS,
    )
    t_ref = time.time() - t0
    print(f"done in {t_ref:.2f}s")

    print(f"\n  {'N':>5}  {'T=0.1 MaxErr':>14}  {'T=1.0 MaxErr':>14}  "
          f"{'T=2.0 MaxErr':>14}  {'Global MaxErr':>14}  {'Time':>8}")
    print(f"  {'-'*75}")

    for N in N_VALUES:
        t0 = time.time()
        iv_N = price_iv_surface(
            TEST_PARAMS, T_GRID, K_GRID,
            N_factors=N, N_cos=N_COS,
        )
        t_N = time.time() - t0

        # Absolute IV error vs N=40 reference
        err = np.abs(iv_N - iv_ref)   # (8, 11)

        # Find T indices closest to 0.1, 1.0, 2.0
        t01 = np.argmin(np.abs(T_GRID - 0.1))
        t10 = np.argmin(np.abs(T_GRID - 1.0))
        t20 = np.argmin(np.abs(T_GRID - 2.0))

        err_t01   = err[t01].max()
        err_t10   = err[t10].max()
        err_t20   = err[t20].max()
        err_global = err.max()

        ref_str = " ← REFERENCE" if N == 40 else ""
        bp01    = err_t01 * 10000   # convert to basis points
        bp10    = err_t10 * 10000
        bp20    = err_t20 * 10000
        bp_g    = err_global * 10000

        results[N] = {
            "err_t01": err_t01, "err_t10": err_t10,
            "err_t20": err_t20, "err_global": err_global,
            "time": t_N, "iv": iv_N,
        }

        flag = ""
        if N < 40:
            flag = " ✓" if bp01 < 1.0 else " ✗ (>1bp)"

        print(f"  {N:>5}  {bp01:>12.4f}bp  {bp10:>12.4f}bp  "
              f"{bp20:>12.4f}bp  {bp_g:>12.4f}bp  {t_N:>7.2f}s{ref_str}{flag}")

    results[40]["iv"] = iv_ref
    results[40]["err_t01"] = 0.0
    results[40]["time"] = t_ref

    print(f"\n  Conclusion:")
    n20 = results[20]
    print(f"    N=20 max error at T=0.1 : {n20['err_t01']*10000:.4f} bp  "
          f"({'< 1bp ✓' if n20['err_t01'] < 1e-4 else '> 1bp ✗'})")
    print(f"    N=20 global max error   : {n20['err_global']*10000:.4f} bp")
    print(f"    N=20 speedup vs N=40    : {results[40]['time']/n20['time']:.1f}×")
    print("=" * 68)

    # Save results to text file
    out_path = os.path.join(os.path.dirname(__file__), "convergence_results.txt")
    with open(out_path, "w") as f:
        f.write("Bernstein Factor Convergence Results\n")
        f.write("=" * 68 + "\n")
        f.write(f"Reference: N=40\n")
        f.write(f"Test params: {TEST_PARAMS}\n\n")
        f.write(f"{'N':>5}  {'T=0.1 MaxErr(bp)':>18}  {'T=1.0 MaxErr(bp)':>18}  "
                f"{'Global MaxErr(bp)':>18}  {'Time(s)':>8}\n")
        f.write("-" * 75 + "\n")
        for N in N_VALUES:
            r = results[N]
            f.write(f"{N:>5}  {r['err_t01']*1e4:>18.4f}  {r['err_t10']*1e4:>18.4f}  "
                    f"{r['err_global']*1e4:>18.4f}  {r['time']:>8.2f}\n")
        f.write("\n")
        f.write(f"N=20 vs N=40 error at T=0.1: {results[20]['err_t01']*1e4:.4f} bp\n")
        f.write(f"Conclusion: N=20 is {'SUFFICIENT (<1bp)' if results[20]['err_t01'] < 1e-4 else 'INSUFFICIENT (>1bp)'}\n")
    print(f"\n  Saved: {out_path}")

    return results


if __name__ == "__main__":
    run_convergence()
