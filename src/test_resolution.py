"""
Phase 6.1: Discretization Invariance Test
Zero-Shot Super-Resolution of the Mirror-Padded FNO

The FNO was trained on an 8x11 grid. We test its Operator learning capability
by feeding it a 50x50 coordinate grid without retraining.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_model import MirrorPaddedFNO2d

def main():
    print("Testing FNO Discretization Invariance...")
    
    # Load model
    model = MirrorPaddedFNO2d()
    weights_path = "artifacts/models/fno_best.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"Loaded weights from {weights_path}")
    else:
        print("Warning: Trained weights not found. Using untrained initialization.")
    
    model.eval()
    
    # Define arbitrary Deep Rough parameters: [kappa, theta, sigma, rho, v0, H]
    params = torch.tensor([2.5, 0.08, 0.5, -0.5, 0.08, 0.08], dtype=torch.float32)
    
    # 1. Baseline 8x11 Grid
    T_grid_8 = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float32)
    K_grid_11 = torch.linspace(-0.5, 0.5, 11, dtype=torch.float32)
    T_mesh_8, K_mesh_11 = torch.meshgrid(T_grid_8, K_grid_11, indexing='ij')
    
    params_exp_8 = params.view(1, 1, 1, 6).expand(1, 8, 11, 6)
    in_8x11 = torch.cat([params_exp_8, T_mesh_8.unsqueeze(0).unsqueeze(-1), K_mesh_11.unsqueeze(0).unsqueeze(-1)], dim=-1)
    
    with torch.no_grad():
        out_8x11 = model(in_8x11).squeeze(0).numpy()
        
    # 2. Super-Dense 50x50 Grid
    T_grid_50 = torch.linspace(0.1, 2.0, 50, dtype=torch.float32)
    K_grid_50 = torch.linspace(-0.5, 0.5, 50, dtype=torch.float32)
    T_mesh_50, K_mesh_50 = torch.meshgrid(T_grid_50, K_grid_50, indexing='ij')
    
    params_exp_50 = params.view(1, 1, 1, 6).expand(1, 50, 50, 6)
    in_50x50 = torch.cat([params_exp_50, T_mesh_50.unsqueeze(0).unsqueeze(-1), K_mesh_50.unsqueeze(0).unsqueeze(-1)], dim=-1)
    
    with torch.no_grad():
        out_50x50 = model(in_50x50).squeeze(0).numpy()
        
    # 3. Interpolated 8x11 to 50x50
    points = np.array([[t, k] for t in T_grid_8.numpy() for k in K_grid_11.numpy()])
    values = out_8x11.flatten()
    grid_t, grid_k = np.meshgrid(T_grid_50.numpy(), K_grid_50.numpy(), indexing='ij')
    out_interpolated = griddata(points, values, (grid_t, grid_k), method='linear')
    
    # 4. Plot Comparison
    os.makedirs("images/ai_generated", exist_ok=True)
    fig = plt.figure(figsize=(18, 6), facecolor="#0D1117")
    
    # Common 3D styling
    def style_ax(ax, title):
        ax.set_facecolor("#161B22")
        ax.set_xlabel("Log-Moneyness", color="#E6EDF3")
        ax.set_ylabel("Maturity", color="#E6EDF3")
        ax.set_zlabel("Total Variance", color="#E6EDF3")
        ax.set_title(title, color="#E6EDF3", pad=10)
        ax.tick_params(colors="#E6EDF3", labelsize=7)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#30363D")
        ax.view_init(elev=30, azim=-45)
    
    # Original 8x11
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(K_mesh_11.numpy(), T_mesh_8.numpy(), out_8x11, cmap='Blues', edgecolor='w', linewidth=0.2)
    style_ax(ax1, "Original (8x11 Grid)")
    
    # Linear Interpolation
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(grid_k, grid_t, out_interpolated, cmap='Blues', edgecolor='w', linewidth=0.2)
    style_ax(ax2, "Linear Interpolation (50x50)")
    
    # FNO Zero-Shot
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(K_mesh_50.numpy(), T_mesh_50.numpy(), out_50x50, cmap='Reds', edgecolor='w', linewidth=0.2)
    style_ax(ax3, "FNO Zero-Shot Super-Resolution (50x50)")
    
    plt.tight_layout()
    out_path = "images/ai_generated/fno_resolution_invariance.png"
    fig.savefig(out_path, dpi=200, facecolor="#0D1117")
    print(f"Zero-Shot Super-Resolution plot saved to {out_path}")

if __name__ == "__main__":
    main()
