import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market.spx_data import download_spx_chain, clean_chain, to_iv_surface, T_GRID, K_GRID
from fno_model import MirrorPaddedFNO2d
import calibrate
from calibrate_fast import calibrate_newton

def main():
    print("Running SPX calibration plots generator...")
    snapshot_date = date(2024, 1, 2)
    S0 = 4700.0
    r = 0.05
    q = 0.015
    
    # 1. Download & clean
    df = download_spx_chain(snapshot_date, cache=True)
    df_clean = clean_chain(df)
    
    # 2. Get target surface
    target_surface = to_iv_surface(df_clean, S0, r, q)
    
    # 3. Load model and run calibration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MirrorPaddedFNO2d()
    weights_path = Path(__file__).parent.parent / "artifacts" / "weights" / "fno_v2_final_prod.pth"
    
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    orig_v1 = calibrate._NORM_VERSIONS["v1"]
    try:
        calibrate._NORM_VERSIONS["v1"] = calibrate._NORM_VERSIONS["v2"]
        calibrate._param_norm = None
        calibrate._iv_norm = None
        
        res = calibrate_newton(model, target_surface, T_GRID, K_GRID, max_iter=20)
        
        # Get calibrated surface
        v0 = res["v0"]
        zeta = res["zeta"]
        lam = res["lambda"]
        sigma = res["sigma"]
        rho = res["rho"]
        
        from calibrate_fast import _reparam_to_6d
        p3 = torch.tensor([v0, zeta, lam], dtype=torch.float32, device=device)
        p6 = _reparam_to_6d(p3[0:1], p3[1:2], p3[2:3], device)
        spatial = calibrate._make_spatial_input(T_GRID, K_GRID, device)
        with torch.no_grad():
            calibrated_surface = calibrate._fno_predict_real_iv(model, p6, spatial).cpu().numpy()
            
    finally:
        calibrate._NORM_VERSIONS["v1"] = orig_v1
        calibrate._param_norm = None
        calibrate._iv_norm = None
        
    print(f"Calibration completed. RMSE: {res['final_mse']**0.5 * 10000:.2f} bps")
    print(f"Recovered parameters: v0={v0:.4f}, sigma={sigma:.4f}, rho={rho:.4f}")
    
    # Ensure images/generated directory exists
    img_dir = Path(__file__).parent.parent / "images" / "generated"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Target vs Calibrated Surface (1D slices across strikes for 4 maturities)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    maturities_indices = [0, 2, 4, 7] # 0.1, 0.6, 1.2, 2.0y
    
    for idx, t_idx in enumerate(maturities_indices):
        ax = axes[idx]
        T_val = T_GRID[t_idx]
        ax.plot(K_GRID, target_surface[t_idx], "o-", label="Target (Market)", color="blue", linewidth=1.5)
        ax.plot(K_GRID, calibrated_surface[t_idx], "x--", label="Calibrated (FNO)", color="red", linewidth=1.5)
        ax.set_title(f"Maturity T = {T_val:.2f}y")
        ax.set_xlabel("Log-moneyness k")
        ax.set_ylabel("Implied Volatility")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        
    plt.suptitle(f"SPX Volatility Surface Calibration - {snapshot_date}\nRMSE: {res['final_mse']**0.5 * 10000:.2f} bps", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(img_dir / "spx_calibration_surface.png", dpi=300)
    plt.close()
    
    # Plot 2: Gauss-Newton Loss Curve
    plt.figure(figsize=(8, 5))
    loss_history = res["history"]
    plt.plot(np.arange(1, len(loss_history) + 1), loss_history, "o-", color="darkblue", linewidth=2)
    plt.yscale("log")
    plt.title("Gauss-Newton Calibration Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(img_dir / "spx_calibration_loss.png", dpi=300)
    plt.close()
    
    print("Plots saved successfully to images/generated/")

if __name__ == "__main__":
    main()
