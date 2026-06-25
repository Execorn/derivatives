"""
Phase 6.2: The Autograd Greeks Engine
Extracts exact 2nd-order sensitivities (Volga and Vanna) from the MFNO
using PyTorch Autograd, enabled by the C^2 smoothness of the ELU activations.

Design decisions:
  - Greeks are computed via vectorized torch.func transforms. First-order sensitivities
    use reverse-mode AD (jacrev / VJPs). Second-order sensitivities use forward-mode AD
    (jacfwd / JVPs) over the first-order VJPs to prevent OOM errors on Laptop GPU.
  - Supports second-order Greek supervision by tracing through the FNO parameters
    using torch.func.functional_call, maintaining the autograd graph for model weights.
  - Volatility parameters are clamped to prevent Durrleman singularities (>= 0.01).
  - CUDA execution is preferred when a GPU is available.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, jacfwd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d


def _fno_iv_flat(params: torch.Tensor, model: torch.nn.Module, T_grid: torch.Tensor, K_grid: torch.Tensor) -> torch.Tensor:
    """
    Pure-function wrapper: maps params (6,) -> IV surface flattened (T*K,).
    Uses functional_call to trace through model parameters.
    """
    params_dict = dict(model.named_parameters())
    device = params.device

    # Load normalizers v2 once
    from deepvol.calibration import calibrate_bfgs as cb
    if cb._ROOT_DIR.endswith("src/deepvol"):
        cb._ROOT_DIR = os.path.dirname(os.path.dirname(cb._ROOT_DIR))
    cb._load_normalizers(version="v2")


    pn_mean = torch.tensor(cb._param_norm.mean, dtype=torch.float32, device=device)
    pn_std  = torch.tensor(cb._param_norm.std,  dtype=torch.float32, device=device)
    yn_mean = torch.tensor(cb._iv_norm.mean, dtype=torch.float32, device=device)
    yn_std  = torch.tensor(cb._iv_norm.std,  dtype=torch.float32, device=device)

    # Clamp volatility parameters (sigma and v0) to prevent Durrleman singularities (>= 0.01)
    p6_clamped = params.clone()
    p6_clamped = torch.cat([
        p6_clamped[0:2],
        torch.clamp(p6_clamped[2:3], min=0.01),  # sigma
        p6_clamped[3:4],
        torch.clamp(p6_clamped[4:5], min=0.01),  # v0
        p6_clamped[5:6]
    ])

    # Normalize T and K coordinates matching the FNO training distribution
    T_norm = (T_grid - T_grid.mean()) / (T_grid.std() + 1e-8)
    K_norm = K_grid / 0.5
    T_mesh, K_mesh = torch.meshgrid(T_norm, K_norm, indexing='ij')
    spatial = torch.stack([T_mesh, K_mesh], dim=-1).unsqueeze(0).to(device)  # (1, T, K, 2)

    p_norm = (p6_clamped - pn_mean) / pn_std
    iv_norm = functional_call(model, params_dict, (spatial, p_norm.unsqueeze(0)))
    iv_real = iv_norm * yn_std + yn_mean
    iv_real = torch.clamp(iv_real, min=1e-6)
    return iv_real.squeeze(0).reshape(-1)


def compute_greeks(
    model: torch.nn.Module,
    params: torch.Tensor,
    T_grid: torch.Tensor,
    K_grid: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Volga (d^2 IV / d sigma^2) and Vanna (d^2 IV / d sigma d rho)
    across the full (T, K) grid using vectorized reverse-mode AD (VJPs)
    for first-order and forward-mode AD (JVPs) for second-order.

    Args:
        model:   Trained MirrorPaddedFNO2d instance.
        params:  Tensor (6,) [kappa, theta, sigma, rho, v0, H] — Leaf tensor for AD.
        T_grid:  Tensor (T,) of maturities.
        K_grid:  Tensor (K,) of log-moneyness values.

    Returns:
        volga:  Tensor (T, K)  — d^2 IV / d sigma^2
        vanna:  Tensor (T, K)  — d^2 IV / d sigma d rho
    """
    nT, nK = T_grid.size(0), K_grid.size(0)
    SIGMA_IDX, RHO_IDX = 2, 3

    device = params.device
    
    # Run Model Governance (SR 26-2) OOD checks and drift tracking
    with torch.no_grad():
        params_np = params.detach().cpu().numpy()
        from deepvol.mrm.compliance import check_compliance
        _ = check_compliance(params_np)
    
    # Load normalizers v2 once
    from deepvol.calibration import calibrate_bfgs as cb
    if cb._ROOT_DIR.endswith("src/deepvol"):
        cb._ROOT_DIR = os.path.dirname(os.path.dirname(cb._ROOT_DIR))
    cb._load_normalizers(version="v2")

    pn_mean = torch.tensor(cb._param_norm.mean, dtype=torch.float32, device=device)
    pn_std  = torch.tensor(cb._param_norm.std,  dtype=torch.float32, device=device)
    yn_mean = torch.tensor(cb._iv_norm.mean, dtype=torch.float32, device=device)
    yn_std  = torch.tensor(cb._iv_norm.std,  dtype=torch.float32, device=device)

    # Extract model parameters for functional tracing
    params_dict = dict(model.named_parameters())

    # Differentiable forward pass closing over params_dict
    def _iv_flat_fn(p):
        p6_clamped = p.clone()
        p6_clamped = torch.cat([
            p6_clamped[0:2],
            torch.clamp(p6_clamped[2:3], min=0.01),  # sigma
            p6_clamped[3:4],
            torch.clamp(p6_clamped[4:5], min=0.01),  # v0
            p6_clamped[5:6]
        ])

        # Normalize T and K coordinates matching the FNO training distribution
        T_norm = (T_grid - T_grid.mean()) / (T_grid.std() + 1e-8)
        K_norm = K_grid / 0.5
        T_mesh, K_mesh = torch.meshgrid(T_norm, K_norm, indexing='ij')
        spatial = torch.stack([T_mesh, K_mesh], dim=-1).unsqueeze(0).to(device)  # (1, T, K, 2)

        p_norm = (p6_clamped - pn_mean) / pn_std
        iv_norm = functional_call(model, params_dict, (spatial, p_norm.unsqueeze(0)))
        iv_real = iv_norm * yn_std + yn_mean
        iv_real = torch.clamp(iv_real, min=1e-6)
        return iv_real.squeeze(0).reshape(-1)

    # Free fragmented GPU memory before heavy AD computation (CC-C1 fix)
    if params.is_cuda:
        torch.cuda.empty_cache()

    # CC-C1 Fix: Chunk the Hessian computation to stay within 4 GB VRAM.
    # Both the inner jacrev AND the outer jacfwd operate on small output slices.
    # This reduces peak VRAM from ~5 GB (full 88-element surface) to ~1.5 GB.
    n_total = nT * nK
    chunk_size = 22  # Process ~22 grid points at a time (safe for 6 GB GPU)

    volga_chunks = []
    vanna_chunks = []

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)

        # Inner function outputs ONLY the chunk slice — jacrev differentiates
        # only chunk_size outputs, not the full 88-element surface.
        def _iv_chunk_fn(p, _s=start, _e=end):
            return _iv_flat_fn(p)[_s:_e]  # shape (chunk,)

        # First-order: jacrev over the small chunk → (chunk, 6) Jacobian
        def _dIV_dsigma_chunk(p, _fn=_iv_chunk_fn):
            J = jacrev(_fn, argnums=0)(p)
            return J[:, SIGMA_IDX]  # shape (chunk,)

        # Second-order: jacfwd over the small first-order → (chunk, 6)
        H_chunk = jacfwd(_dIV_dsigma_chunk, argnums=0)(params)

        volga_chunks.append(H_chunk[:, SIGMA_IDX])
        vanna_chunks.append(H_chunk[:, RHO_IDX])

        # Free intermediate computation graphs between chunks
        if params.is_cuda:
            torch.cuda.empty_cache()

    volga_flat = torch.cat(volga_chunks, dim=0)
    vanna_flat = torch.cat(vanna_chunks, dim=0)

    volga = volga_flat.reshape(nT, nK)
    vanna = vanna_flat.reshape(nT, nK)

    return volga, vanna


def main():
    print("Testing Vectorized Autograd Greeks Engine...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MirrorPaddedFNO2d()
    weights_path = "artifacts/weights/fno_v2_final_prod.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    from deepvol.calibration import calibrate_bfgs as cb
    if cb._ROOT_DIR.endswith("src/deepvol"):
        cb._ROOT_DIR = os.path.dirname(os.path.dirname(cb._ROOT_DIR))
    cb._load_normalizers(version="v2")
    
    params = torch.tensor([2.5, 0.08, 0.5, -0.5, 0.08, 0.08], dtype=torch.float32, device=device)
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float32, device=device)
    K_grid = torch.linspace(-0.5, 0.5, 11, dtype=torch.float32, device=device)
    
    print("Computing Volga and Vanna (vectorized)...")
    import time
    t0 = time.perf_counter()
    volga_t, vanna_t = compute_greeks(model, params, T_grid, K_grid)
    t1 = time.perf_counter()
    print(f"Vectorized Greeks computed in {(t1 - t0)*1000.0:.2f} ms")
    
    # Detach and convert to numpy for plotting
    volga = volga_t.detach().cpu().numpy()
    vanna = vanna_t.detach().cpu().numpy()
    
    # Plotting
    os.makedirs("images/ai_generated", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0D1117")
    
    sns.set(style="dark")
    
    K_labels = [f"{k:.2f}" for k in K_grid.cpu().numpy()]
    T_labels = [f"{t:.1f}" for t in T_grid.cpu().numpy()]
    
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
