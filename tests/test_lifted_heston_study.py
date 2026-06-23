import os
import sys
import subprocess
import json
import numpy as np
import pytest
import torch

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.path.join(project_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from pricing_engine_gpu import price_batch_gpu

def test_script_execution_and_report_generation():
    # Path to report
    report_path = os.path.join(project_root, "artifacts", "reports", "lifted_heston_convergence.json")
    
    # Remove report if it already exists to ensure we verify generation
    if os.path.exists(report_path):
        os.remove(report_path)
        
    script_path = os.path.join(project_root, "src", "pricing", "lifted_heston_study.py")
    python_exe = os.path.join(project_root, ".venv", "bin", "python")
    
    # Run the script
    result = subprocess.run([python_exe, script_path], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with output:\n{result.stderr}\n{result.stdout}"
    
    # Verify report is generated
    assert os.path.exists(report_path), "Report was not generated"
    
    # Verify we can load the JSON
    with open(report_path, "r") as f:
        data = json.load(f)
        
    assert "results" in data
    assert "reference_N" in data
    assert data["reference_N"] == 256

def test_factor_80_vs_160_convergence():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Parameters & Grids
    TEST_PARAMS = np.array([[1.5, 0.08, 0.5, -0.6, 0.08]], dtype=np.float64)
    H_GRID = [0.04, 0.07, 0.10, 0.14]
    T_GRID = np.array([0.04])  # Short maturity test
    K_GRID = np.linspace(-0.2, 0.2, 11)
    N_COS = 1024
    
    rmses = []
    for H in H_GRID:
        # Price N = 80 surface
        iv_80 = price_batch_gpu(
            TEST_PARAMS, T_GRID, K_GRID,
            H_fixed=H, N_factors=80, N_cos=N_COS, device=device
        )[0]
        
        # Price N = 160 surface
        iv_160 = price_batch_gpu(
            TEST_PARAMS, T_GRID, K_GRID,
            H_fixed=H, N_factors=160, N_cos=N_COS, device=device
        )[0]
        
        # Calculate RMSE in basis points (1 bp = 1e-4)
        err = np.abs(iv_80 - iv_160)
        err_bp = err * 10000.0
        
        rmse_bp = np.sqrt(np.nanmean(err_bp ** 2))
        rmses.append(rmse_bp)
        print(f"H = {H:.2f}: N=80 vs N=160 RMSE = {rmse_bp:.4f} bp")
        
        # For H >= 0.07, verify that the RMSE is individually < 2.0 bp
        if H >= 0.07:
            assert rmse_bp < 2.0, f"RMSE of {rmse_bp:.4f} bp for H={H:.2f} is not < 2.0 bp"
            
    avg_rmse_bp = np.mean(rmses)
    print(f"Average RMSE across H grid: {avg_rmse_bp:.4f} bp")
    
    # Assert average RMSE is < 2.0 basis points
    assert avg_rmse_bp < 2.0, f"Average RMSE of {avg_rmse_bp:.4f} bp is not < 2.0 bp"
