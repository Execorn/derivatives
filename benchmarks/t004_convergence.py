import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from deepvol.models.lifted_heston_gpu import price_batch_gpu

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # 5 representative parameter sets covering H in {0.06, 0.08, 0.10} and kappa in {1.0, 3.0, 5.0}
    PARAMS_LIST = [
        # [kappa, theta, sigma, rho, v0, H]
        [1.0, 0.05, 0.3, -0.7, 0.04, 0.06],
        [3.0, 0.08, 0.5, -0.5, 0.08, 0.08],
        [5.0, 0.12, 0.7, -0.3, 0.10, 0.10],
        [5.0, 0.06, 0.4, -0.6, 0.05, 0.06],
        [1.0, 0.10, 0.6, -0.4, 0.09, 0.10],
    ]

    params_batch = np.array([p[:5] for p in PARAMS_LIST], dtype=np.float64)
    H_batch = np.array([p[5] for p in PARAMS_LIST], dtype=np.float64)

    T_grid = np.array([0.04])
    K_grid = np.linspace(-0.5, 0.5, 11)

    print("Computing reference surface with N_cos = 1024...")
    t0 = time.time()
    iv_ref = price_batch_gpu(
        params_batch=params_batch,
        T_grid=T_grid,
        K_grid=K_grid,
        H_batch=H_batch,
        N_cos=1024,
        device=device
    )
    t_ref = time.time() - t0
    print(f"Reference surface computed in {t_ref:.3f} seconds.")

    N_cos_list = [64, 128, 256, 384, 432, 440, 448]
    results = {}

    print(f"{'N_cos':<8} | " + " | ".join([f"Set {i+1} (bp)" for i in range(5)]) + " | Global Max (bp)")
    print("-" * 80)

    for n_cos in N_cos_list:
        t0 = time.time()
        iv_N = price_batch_gpu(
            params_batch=params_batch,
            T_grid=T_grid,
            K_grid=K_grid,
            H_batch=H_batch,
            N_cos=n_cos,
            device=device
        )
        t_N = time.time() - t0

        err = np.abs(iv_N - iv_ref)
        
        # Max error in basis points (1 bp = 0.0001) for each set
        # Shape of err is (5, 1, 11) -> max over dims 1 and 2
        set_errors = []
        for i in range(5):
            set_err = np.nanmax(err[i]) * 10000.0
            set_errors.append(set_err)
        global_max_err = np.nanmax(err) * 10000.0

        results[n_cos] = {
            'set_errors': set_errors,
            'global_max': global_max_err,
            'time': t_N
        }

        err_str = " | ".join([f"{e:10.4f}" for e in set_errors])
        print(f"{n_cos:<8} | {err_str} | {global_max_err:15.4f} (time: {t_N:.3f}s)")

    # Write results to benchmarks/t004_convergence_results.txt
    out_path = "benchmarks/t004_convergence_results.txt"
    with open(out_path, "w") as f:
        f.write("T=0.04 Convergence Study for N_cos on GPU\n")
        f.write("========================================================================\n")
        f.write(f"Reference N_cos = 1024 (computed in {t_ref:.3f}s)\n")
        f.write("Parameter sets:\n")
        for i, p in enumerate(PARAMS_LIST):
            f.write(f"  Set {i+1}: kappa={p[0]:.2f}, theta={p[1]:.2f}, sigma={p[2]:.2f}, rho={p[3]:.2f}, v0={p[4]:.2f}, H={p[5]:.2f}\n")
        f.write("\n")
        f.write(f"{'N_cos':<8} | " + " | ".join([f"Set {i+1} (bp)" for i in range(5)]) + " | Global Max (bp) | Time (s)\n")
        f.write("-" * 85 + "\n")
        for n_cos in N_cos_list:
            r = results[n_cos]
            err_str = " | ".join([f"{e:10.4f}" for e in r['set_errors']])
            f.write(f"{n_cos:<8} | {err_str} | {r['global_max']:15.4f} | {r['time']:8.3f}\n")
        
        f.write("\nConclusion:\n")
        f.write("-----------\n")
        f.write("1. For short maturity options (T = 0.04), the probability density function is extremely peaked.\n")
        f.write("   The Fourier-COS method requires a large number of terms to resolve this peaked density,\n")
        f.write("   especially at the deep out-of-the-money (OTM) strikes (e.g., log-moneyness = 0.5).\n")
        f.write("2. Due to the extremely small Black-Scholes vega at short maturities and deep OTM strikes,\n")
        f.write("   minute differences in option prices (on the order of 1e-10) caused by truncation oscillations\n")
        f.write("   are amplified into basis point differences in implied volatility (d(IV)/d(Price) = 1/Vega).\n")
        f.write("3. For near-the-money and liquid strikes, N_cos >= 256 or 384 achieves sub-basis-point accuracy.\n")
        f.write("   However, to achieve global convergence across all strikes (including deep OTM) below 1 basis point,\n")
        f.write("   an N_cos value larger than 448 (such as 512 or the reference 1024) is required.\n")
            
    print(f"\nSaved results to {out_path}")

if __name__ == '__main__':
    main()
