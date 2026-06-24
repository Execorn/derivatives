"""
lifted_heston_study.py — Lifted Heston Factor study script.
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.path.join(project_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from deepvol.models.lifted_heston_gpu import price_batch_gpu

# Representative parameters for the study
TEST_PARAMS = np.array([[1.5, 0.08, 0.5, -0.6, 0.08]], dtype=np.float64)

# Grid configurations
H_GRID = [0.04, 0.07, 0.10, 0.14]
N_GRID = [5, 10, 20, 40, 80, 160]
N_REF = 256

T_GRID = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.3, 0.5, 1.0, 2.0])
K_GRID = np.linspace(-0.2, 0.2, 11)
N_COS = 1024  # Large N_cos to prevent Fourier truncation errors at short T

def run_study(device='cuda'):
    results = {}
    
    reports_dir = Path(project_root) / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(" LIFTED HESTON FACTOR CONVERGENCE STUDY")
    print(f" Reference N={N_REF}  |  N_cos={N_COS}  |  Device={device}")
    print("=" * 80)
    
    for H in H_GRID:
        print(f"\nEvaluating H = {H:.2f}...")
        results[f"H_{H:.2f}"] = {}
        
        # 1. Compute reference surface (N = N_REF)
        t0 = time.time()
        iv_ref = price_batch_gpu(
            TEST_PARAMS, T_GRID, K_GRID,
            H_fixed=H, N_factors=N_REF, N_cos=N_COS, device=device
        )[0]
        t_ref = time.time() - t0
        print(f"  Reference N={N_REF} computed in {t_ref:.3f}s")
        
        for N in N_GRID:
            # 2. Compute N-factor surface
            t0 = time.time()
            iv_N = price_batch_gpu(
                TEST_PARAMS, T_GRID, K_GRID,
                H_fixed=H, N_factors=N, N_cos=N_COS, device=device
            )[0]
            t_N = time.time() - t0
            
            # 3. Calculate errors in basis points
            err = np.abs(iv_N - iv_ref)
            err_bp = err * 10000.0
            
            # Global metrics
            global_rmse = np.sqrt(np.nanmean(err_bp ** 2))
            global_max = np.nanmax(err_bp)
            
            # Ultra-short maturity metrics (T <= 0.1)
            short_idx = np.where(T_GRID <= 0.1)[0]
            short_rmse = np.sqrt(np.nanmean(err_bp[short_idx] ** 2))
            short_max = np.nanmax(err_bp[short_idx])
            
            results[f"H_{H:.2f}"][f"N_{N}"] = {
                "rmse_global_bp": float(global_rmse),
                "max_error_global_bp": float(global_max),
                "rmse_short_term_bp": float(short_rmse),
                "max_error_short_term_bp": float(short_max),
                "execution_time_seconds": float(t_N)
            }
            
            print(f"    N={N:<3} | Global RMSE: {global_rmse:6.4f}bp | Max: {global_max:6.4f}bp | Short-Term RMSE: {short_rmse:6.4f}bp | Time: {t_N:.3f}s")
            
    # 4. Save results to artifacts/reports/lifted_heston_convergence.json
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_N": N_REF,
        "N_cos": N_COS,
        "parameters": {
            "kappa": TEST_PARAMS[0, 0],
            "theta": TEST_PARAMS[0, 1],
            "sigma": TEST_PARAMS[0, 2],
            "rho": TEST_PARAMS[0, 3],
            "v0": TEST_PARAMS[0, 4]
        },
        "grids": {
            "maturities": T_GRID.tolist(),
            "log_moneyness": K_GRID.tolist()
        },
        "results": results
    }
    
    out_file = reports_dir / "lifted_heston_convergence.json"
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved report to: {out_file}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_study(device)
