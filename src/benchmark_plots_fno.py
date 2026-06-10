"""
Deep Rough Heston Calibration Benchmark & Visualization
=====================================================
Creates a publication-quality 3D surface comparison plot for thesis defense:
  - Target Market IV Surface (from test set)
  - Calibrated Model IV Surface (from FNO surrogate after L-BFGS calibration)

Grid: 8 maturities × 11 strikes = 88 points.
Output: images/ai_generated/surface_fit_fno.png
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fno_model import MirrorPaddedFNO2d
from calibrate import calibrate_parameters

# ─── Grid ─────────────────────────────────────────────────────────────────────
STRIKES = np.linspace(-0.5, 0.5, 11)  # log-moneyness in Deep Rough Dataset
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
N_MAT, N_STR = len(MATURITIES), len(STRIKES)

PARAM_NAMES = ["κ", "θ", "σ", "ρ", "v₀", "H"]
PARAM_LABELS = ["κ (kappa)", "θ (theta)", "σ (sigma)", "ρ (rho)", "v₀", "H (Hurst)"]

# ─── Dark-theme palette ───────────────────────────────────────────────────────
DARK_BG = "#0D1117"
PANEL_BG = "#161B22"
GRID_CLR = "#21262D"
EDGE_CLR = "#30363D"
TEXT_CLR = "#E6EDF3"
SUB_CLR = "#8B949E"
BLUE = "#58A6FF"
ORANGE = "#F0883E"

def style_3d_axis(ax, title: str):
    ax.set_facecolor(PANEL_BG)
    ax.set_xlabel("Log-Moneyness (ln(K/S))", labelpad=8, fontsize=8.5, color=TEXT_CLR)
    ax.set_ylabel("Maturity (T)", labelpad=8, fontsize=8.5, color=TEXT_CLR)
    ax.set_zlabel("Implied Volatility", labelpad=8, fontsize=8.5, color=TEXT_CLR)
    ax.set_title(title, color=TEXT_CLR, fontsize=11, pad=14, fontweight="bold")
    ax.tick_params(colors=TEXT_CLR, labelsize=6.5)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(EDGE_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.4)
    ax.view_init(elev=25, azim=-48)

def make_figure(
    Z_target: np.ndarray,
    Z_model: np.ndarray,
    true_params: np.ndarray,
    calibrated_params: np.ndarray,
    calib_time_ms: float,
    loss: float,
) -> plt.Figure:
    K, T = np.meshgrid(STRIKES, MATURITIES)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": TEXT_CLR,
        "axes.labelcolor": TEXT_CLR,
        "xtick.color": TEXT_CLR,
        "ytick.color": TEXT_CLR,
    })

    fig = plt.figure(figsize=(18, 9), facecolor=DARK_BG)

    # ── Left: 3D comparison surface ──
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.plot_surface(K, T, Z_target, color=BLUE, alpha=0.50, linewidth=0, antialiased=True)
    ax3d.plot_wireframe(K, T, Z_target, color=BLUE, linewidth=0.5, alpha=0.85)
    ax3d.plot_surface(K, T, Z_model, color=ORANGE, alpha=0.38, linewidth=0, antialiased=True)
    ax3d.plot_wireframe(K, T, Z_model, color=ORANGE, linewidth=0.5, alpha=0.85, linestyle="--")

    style_3d_axis(ax3d, "IV Surface — Target vs Calibrated (Deep Rough)")

    legend_elems = [
        Patch(facecolor=BLUE, alpha=0.8, label="Target Market Surface"),
        Patch(facecolor=ORANGE, alpha=0.8, label="Calibrated Model Surface"),
    ]
    ax3d.legend(handles=legend_elems, loc="upper left", facecolor=PANEL_BG, edgecolor=EDGE_CLR, labelcolor=TEXT_CLR, fontsize=8)

    # ── Right: per-maturity smile slices ──
    ax2d = fig.add_subplot(122, facecolor=PANEL_BG)
    n = len(MATURITIES)
    blues = plt.get_cmap("Blues")(np.linspace(0.35, 0.9, n))
    oranges = plt.get_cmap("Oranges")(np.linspace(0.35, 0.9, n))

    for i, mat in enumerate(MATURITIES):
        ax2d.plot(STRIKES, Z_target[i], color=blues[i], linewidth=1.9, label=f"T={mat:.1f}")
        ax2d.plot(STRIKES, Z_model[i], color=oranges[i], linewidth=1.4, linestyle="--")

    ax2d.set_xlabel("Log-Moneyness  ln(K/S)", color=TEXT_CLR, fontsize=9)
    ax2d.set_ylabel("Implied Volatility", color=TEXT_CLR, fontsize=9)
    ax2d.set_title("Smile Slices — All Maturities\n─── Target     - - - Calibrated", color=TEXT_CLR, fontsize=11, pad=12, fontweight="bold")
    ax2d.tick_params(colors=TEXT_CLR, labelsize=8)
    ax2d.spines[:].set_color(EDGE_CLR)
    ax2d.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--")

    mat_patches = [Patch(facecolor=blues[i], alpha=0.9, label=f"T={m:.1f}") for i, m in enumerate(MATURITIES)]
    ax2d.legend(handles=mat_patches, loc="upper right", facecolor=PANEL_BG, edgecolor=EDGE_CLR, labelcolor=TEXT_CLR, fontsize=7, ncol=2)

    # ── Metadata ──
    fig.suptitle("Deep Rough Heston FNO Calibration", color=TEXT_CLR, fontsize=14, fontweight="bold", y=0.99)

    sep = "   |   "
    true_str = sep.join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, true_params))
    calib_str = sep.join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, calibrated_params))

    fig.text(0.5, 0.94, f"True:       {true_str}", ha="center", color=BLUE, fontsize=7.5, family="monospace")
    fig.text(0.5, 0.915, f"Calibrated: {calib_str}", ha="center", color=ORANGE, fontsize=7.5, family="monospace")
    fig.text(0.5, 0.89, f"Calibration time: {calib_time_ms:.2f} ms   |   Final MSE: {loss:.4e}   |   Architecture: Mirror-Padded FNO2d (ELU)", ha="center", color=SUB_CLR, fontsize=7.5)

    plt.tight_layout(rect=[0, 0, 1, 0.89])
    return fig

def main():
    os.makedirs("images/ai_generated", exist_ok=True)
    
    print("[1/4] Loading model and dataset...")
    model = MirrorPaddedFNO2d()
    weights_path = "artifacts/models/fno_best.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print(f"Warning: {weights_path} not found. Using untrained model.")
        
    data = np.load("data/DeepRoughDataset.npz")['dataset']
    # Choose a random test sample
    np.random.seed(42)
    idx = np.random.randint(int(0.8 * len(data)), len(data))
    sample = data[idx]
    
    true_params = sample[:6]
    target_iv = sample[6:].reshape(8, 11)
    
    # Starting guess (e.g., from historical data or midpoints)
    init_params = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])
    
    print("[2/4] Calibrating to Deep Rough IV Surface...")
    calibrated_params, loss_history, elapsed = calibrate_parameters(model, target_iv, init_params, MATURITIES, STRIKES)
    calib_ms = elapsed * 1000
    
    print("\n" + "=" * 62)
    print(f"  {'Parameter':<16}  {'True':>10}  {'Calibrated':>12}  {'Rel Err':>8}")
    print("=" * 62)
    for name, tv, cv in zip(PARAM_LABELS, true_params, calibrated_params):
        pct = abs(tv - cv) / (abs(tv) + 1e-12) * 100
        print(f"  {name:<16}  {tv:>10.6f}  {cv:>12.6f}  {pct:>7.3f}%")
    print("=" * 62)
    
    print("[3/4] Reconstructing IV surface...")
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        params_t = torch.tensor(calibrated_params, dtype=torch.float32, device=device)
        T_mesh, K_mesh = torch.meshgrid(torch.tensor(MATURITIES, dtype=torch.float32), torch.tensor(STRIKES, dtype=torch.float32), indexing='ij')
        
        params_expanded = params_t.view(1, 1, 1, 6).expand(1, 8, 11, 6)
        T_mesh_expanded = T_mesh.unsqueeze(0).unsqueeze(-1)
        K_mesh_expanded = K_mesh.unsqueeze(0).unsqueeze(-1)
        
        fno_input = torch.cat([params_expanded, T_mesh_expanded, K_mesh_expanded], dim=-1)
        calibrated_iv = model(fno_input).squeeze(0).numpy()
        
    print("[4/4] Generating benchmark plots...")
    fig = make_figure(target_iv, calibrated_iv, true_params, calibrated_params, calib_ms, loss_history[-1])
    out_path = "images/ai_generated/surface_fit_fno.png"
    fig.savefig(out_path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved → {out_path}")

if __name__ == "__main__":
    main()
