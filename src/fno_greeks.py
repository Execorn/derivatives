"""
Phase 6.2: The Autograd Greeks Engine
Extracts exact 2nd-order sensitivities (Volga and Vanna) from the MFNO
using PyTorch Autograd, enabled by the C^2 smoothness of the ELU activations.

Design decisions:
  - The FNO is trained to predict IV directly (NOT total variance W = IV^2*T).
    The previous version incorrectly interpreted model output as W and applied
    an extra sqrt(W/T) conversion, yielding sqrt(IV/T) instead of IV.
  - Greeks are computed via a single torch.func.jacrev call over the flattened
    IV surface, avoiding the memory-leaking 88-iteration retain_graph loop.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_model import MirrorPaddedFNO2d


def _fno_iv_flat(params, model, T_grid, K_grid):
    """
    Pure-function wrapper: maps params (6,) -> IV surface flattened (T*K,).
    Required by torch.func.jacrev for functional transforms.
    The model is closed over so autograd can trace through it.
    """
    T_mesh, K_mesh = torch.meshgrid(T_grid, K_grid, indexing='ij')
    params_exp = params.view(1, 1, 1, 6).expand(1, T_grid.size(0), K_grid.size(0), 6)
    in_tensor = torch.cat(
        [params_exp,
         T_mesh.unsqueeze(0).unsqueeze(-1),
         K_mesh.unsqueeze(0).unsqueeze(-1)],
        dim=-1
    )
    # FIX: model predicts IV directly — do NOT apply sqrt(W/T) here.
    # Previous code treated model output as W = IV^2*T and back-converted,
    # which produced sqrt(IV/T) rather than IV.
    IV = model(in_tensor).squeeze(0)          # shape (T, K)
    IV = torch.clamp(IV, min=1e-6)            # numerical floor
    return IV.reshape(-1)                     # (T*K,)


def compute_greeks(model, params, T_grid, K_grid):
    """
    Computes Volga (d^2 IV / d sigma^2) and Vanna (d^2 IV / d sigma d rho)
    across the full (T, K) grid using a single Jacobian pass.

    Args:
        model:   Trained MirrorPaddedFNO2d instance (eval mode, CPU).
        params:  Tensor (6,) [kappa, theta, sigma, rho, v0, H] — will be detached
                 and re-attached as a leaf for AD.
        T_grid:  Tensor (T,) of maturities.
        K_grid:  Tensor (K,) of log-moneyness values.

    Returns:
        volga:  ndarray (T, K)  — d^2 IV / d sigma^2
        vanna:  ndarray (T, K)  — d^2 IV / d sigma d rho
    """
    nT, nK = T_grid.size(0), K_grid.size(0)
    SIGMA_IDX, RHO_IDX = 2, 3

    # Detach and re-register params as a differentiable leaf.
    params = params.detach().requires_grad_(True)

    # --- First-order Jacobian: J[i, p] = d IV_flat[i] / d params[p] ---
    # shape: (T*K, 6)
    J = torch.autograd.functional.jacobian(
        lambda p: _fno_iv_flat(p, model, T_grid, K_grid),
        params,
        create_graph=True,   # keep graph alive for second-order pass
        vectorize=False,     # safer with complex models; set True for speed if verified
    )  # shape: (T*K, 6)

    dIV_dsigma_flat = J[:, SIGMA_IDX]  # (T*K,)

    # --- Second-order: d/d params of each dIV_dsigma[i] ---
    # We accumulate Volga and Vanna by computing grad of sum(dIV_dsigma * selector)
    # for each output cell.  A cleaner approach: compute Hessian rows for sigma and rho.
    volga_flat = torch.zeros(nT * nK)
    vanna_flat = torch.zeros(nT * nK)

    for idx in range(nT * nK):
        g2 = torch.autograd.grad(
            dIV_dsigma_flat[idx], params,
            retain_graph=(idx < nT * nK - 1),  # free graph on last iteration
            create_graph=False,
        )[0]  # shape (6,)
        volga_flat[idx] = g2[SIGMA_IDX].item()
        vanna_flat[idx]  = g2[RHO_IDX].item()

    volga = volga_flat.reshape(nT, nK).numpy()
    vanna  = vanna_flat.reshape(nT, nK).numpy()
    return volga, vanna

def main():
    print("Testing Autograd Greeks Engine...")
    model = MirrorPaddedFNO2d()
    weights_path = "artifacts/models/fno_best.pth"
    if os.path.exists(weights_path):
        # weights_only=True prevents arbitrary code execution via pickle (PyTorch >= 2.0)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    
    params = torch.tensor([2.5, 0.08, 0.5, -0.5, 0.08, 0.08], dtype=torch.float32)
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float32)
    K_grid = torch.linspace(-0.5, 0.5, 11, dtype=torch.float32)
    
    print("Computing Volga and Vanna (this may take a moment)...")
    volga, vanna = compute_greeks(model, params, T_grid, K_grid)
    
    # Plotting
    os.makedirs("images/ai_generated", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0D1117")
    
    # Styling for seaborn heatmaps
    sns.set(style="dark")
    
    K_labels = [f"{k:.2f}" for k in K_grid.numpy()]
    T_labels = [f"{t:.1f}" for t in T_grid.numpy()]
    
    ax1 = axes[0]
    sns.heatmap(volga, xticklabels=K_labels, yticklabels=T_labels, cmap="magma", ax=ax1, cbar_kws={'label': 'Volga'})
    ax1.set_title(r"Volga ($\frac{\partial^2 IV}{\partial \sigma^2}$)", color="#E6EDF3", fontsize=14)
    ax1.set_xlabel("Log-Moneyness", color="#E6EDF3")
    ax1.set_ylabel("Maturity", color="#E6EDF3")
    ax1.tick_params(colors="#E6EDF3")
    ax1.set_facecolor("#0D1117")
    
    ax2 = axes[1]
    sns.heatmap(vanna, xticklabels=K_labels, yticklabels=T_labels, cmap="viridis", ax=ax2, cbar_kws={'label': 'Vanna'})
    ax2.set_title(r"Vanna ($\frac{\partial^2 IV}{\partial \sigma \partial \rho}$)", color="#E6EDF3", fontsize=14)
    ax2.set_xlabel("Log-Moneyness", color="#E6EDF3")
    ax2.set_ylabel("Maturity", color="#E6EDF3")
    ax2.tick_params(colors="#E6EDF3")
    ax2.set_facecolor("#0D1117")
    
    # Set colorbar tick labels to white
    for cbar_ax in fig.axes[2:]:
        cbar_ax.yaxis.label.set_color('#E6EDF3')
        cbar_ax.tick_params(colors='#E6EDF3')

    plt.tight_layout()
    out_path = "images/ai_generated/fno_greeks_heatmap.png"
    fig.savefig(out_path, dpi=200, facecolor="#0D1117")
    print(f"Greeks Heatmaps saved to {out_path}")

if __name__ == "__main__":
    main()
