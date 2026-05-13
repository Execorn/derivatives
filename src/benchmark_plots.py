"""
Heston Calibration Benchmark & Visualization
=============================================
Creates a publication-quality 3D surface comparison plot for thesis defense:
  - Target Market IV Surface (from test set, original scale)
  - Calibrated Model IV Surface (from surrogate NN after L-BFGS-B calibration)

Grid: 8 maturities × 11 strikes = 88 points  (row-major: surface[mat_i, strike_j])
Output: src/results/surface_fit.png

Usage:
    cd path/to/derivatives
    python src/benchmark_plots.py
"""

import sys
import gzip
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
import joblib
import torch
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─── Paths ─────────────────────────────────────────────────────────────────────

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from model import HestonSurrogateMLP
from calibrator import HestonCalibrator

WEIGHTS_PATH = PROJECT_ROOT / "artifacts" / "weights" / "heston_best.pth"
FEAT_SCALER = PROJECT_ROOT / "artifacts" / "scalers" / "feature_scaler.pkl"
TARGET_SCALER = PROJECT_ROOT / "artifacts" / "scalers" / "target_scaler.pkl"
DATA_PATH = (
    PROJECT_ROOT / "data" / "HestonTrainSet.txt.gz"
)
RESULTS_DIR = PROJECT_ROOT / "images" / "ai_generated"

# ─── Grid (must match training data layout) ────────────────────────────────────

STRIKES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])  # 11
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])  # 8
N_MAT, N_STR = len(MATURITIES), len(STRIKES)  # 8, 11
LOG_K = np.log(STRIKES)  # log-moneyness axis

PARAM_NAMES = ["κ", "θ", "σ", "ρ", "v₀"]
PARAM_LABELS = ["κ (kappa)", "θ (theta)", "σ (sigma)", "ρ (rho)", "v₀"]

# ─── Dark-theme palette ────────────────────────────────────────────────────────

DARK_BG = "#0D1117"
PANEL_BG = "#161B22"
GRID_CLR = "#21262D"
EDGE_CLR = "#30363D"
TEXT_CLR = "#E6EDF3"
SUB_CLR = "#8B949E"
BLUE = "#58A6FF"
ORANGE = "#F0883E"


# ─── Helpers ───────────────────────────────────────────────────────────────────


def unflatten_surface(vec: np.ndarray) -> np.ndarray:
    """Reshape 88-element flat vector → (8 maturities, 11 strikes), row-major."""
    assert vec.shape == (N_MAT * N_STR,), f"Expected ({N_MAT*N_STR},), got {vec.shape}"
    return vec.reshape(N_MAT, N_STR)


def nn_predict_iv(model, f_scaler, t_scaler, params_raw: np.ndarray) -> np.ndarray:
    """
    Forward-pass raw Heston params through the surrogate to get IV surface.

    Args:
        params_raw: shape (5,), unscaled Heston parameters.
    Returns:
        iv_surface: shape (88,), implied vols in original units.
    """
    params_scaled = f_scaler.transform(params_raw.reshape(1, -1))
    x_tensor = torch.tensor(params_scaled, dtype=torch.float32)
    with torch.no_grad():
        iv_scaled = model(x_tensor).numpy()
    return t_scaler.inverse_transform(iv_scaled).flatten()


def style_3d_axis(ax, title: str):
    """Apply dark-theme styling to a 3D axis."""
    ax.set_facecolor(PANEL_BG)
    ax.set_xlabel("Strike (K/S)", labelpad=8, fontsize=8.5, color=TEXT_CLR)
    ax.set_ylabel("Maturity (T)", labelpad=8, fontsize=8.5, color=TEXT_CLR)
    ax.set_zlabel("Implied Vol (σ_BS)", labelpad=8, fontsize=8.5, color=TEXT_CLR)
    ax.set_title(title, color=TEXT_CLR, fontsize=11, pad=14, fontweight="bold")
    ax.tick_params(colors=TEXT_CLR, labelsize=6.5)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(EDGE_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.4)
    ax.view_init(elev=25, azim=-48)


# ─── Plot function ─────────────────────────────────────────────────────────────


def make_figure(
    Z_target: np.ndarray,
    Z_model: np.ndarray,
    true_params: np.ndarray,
    calibrated_params: np.ndarray,
    calib_time_ms: float,
    converged: bool,
) -> plt.Figure:
    """
    Build the full 2-panel figure:
      Left : 3D dual-surface plot
      Right: per-maturity smile slices

    Args:
        Z_target / Z_model: (8, 11) arrays
    """
    K, T = np.meshgrid(STRIKES, MATURITIES)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": TEXT_CLR,
            "axes.labelcolor": TEXT_CLR,
            "xtick.color": TEXT_CLR,
            "ytick.color": TEXT_CLR,
        }
    )

    fig = plt.figure(figsize=(18, 9), facecolor=DARK_BG)

    # ── Left: 3D comparison surface ──────────────────────────────────────────
    ax3d = fig.add_subplot(121, projection="3d")

    # Target (blue): filled + wireframe
    ax3d.plot_surface(K, T, Z_target, color=BLUE, alpha=0.50, linewidth=0, antialiased=True)
    ax3d.plot_wireframe(K, T, Z_target, color=BLUE, linewidth=0.5, alpha=0.85)

    # Calibrated (orange): filled + wireframe
    ax3d.plot_surface(K, T, Z_model, color=ORANGE, alpha=0.38, linewidth=0, antialiased=True)
    ax3d.plot_wireframe(K, T, Z_model, color=ORANGE, linewidth=0.5, alpha=0.85, linestyle="--")

    style_3d_axis(ax3d, "IV Surface — Target vs Calibrated")

    legend_elems = [
        Patch(facecolor=BLUE, alpha=0.8, label="Target Market Surface"),
        Patch(facecolor=ORANGE, alpha=0.8, label="Calibrated Model Surface"),
    ]
    ax3d.legend(
        handles=legend_elems,
        loc="upper left",
        facecolor=PANEL_BG,
        edgecolor=EDGE_CLR,
        labelcolor=TEXT_CLR,
        fontsize=8,
    )

    # ── Right: per-maturity smile slices ─────────────────────────────────────
    ax2d = fig.add_subplot(122, facecolor=PANEL_BG)

    n = len(MATURITIES)
    blues = plt.get_cmap("Blues")(np.linspace(0.35, 0.9, n))
    oranges = plt.get_cmap("Oranges")(np.linspace(0.35, 0.9, n))

    for i, mat in enumerate(MATURITIES):
        ax2d.plot(LOG_K, Z_target[i], color=blues[i], linewidth=1.9, label=f"T={mat:.1f}")
        ax2d.plot(LOG_K, Z_model[i], color=oranges[i], linewidth=1.4, linestyle="--")

    ax2d.set_xlabel("Log-Moneyness  ln(K/S)", color=TEXT_CLR, fontsize=9)
    ax2d.set_ylabel("Implied Volatility", color=TEXT_CLR, fontsize=9)
    ax2d.set_title(
        "Smile Slices — All Maturities\n" "─── Target     - - - Calibrated",
        color=TEXT_CLR,
        fontsize=11,
        pad=12,
        fontweight="bold",
    )
    ax2d.tick_params(colors=TEXT_CLR, labelsize=8)
    ax2d.spines[:].set_color(EDGE_CLR)
    ax2d.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--")

    # Maturity legend
    mat_patches = [
        Patch(facecolor=blues[i], alpha=0.9, label=f"T={m:.1f}") for i, m in enumerate(MATURITIES)
    ]
    ax2d.legend(
        handles=mat_patches,
        loc="upper right",
        facecolor=PANEL_BG,
        edgecolor=EDGE_CLR,
        labelcolor=TEXT_CLR,
        fontsize=7,
        ncol=2,
    )

    # ── Suptitle & metadata ──────────────────────────────────────────────────
    fig.suptitle(
        "Heston Neural Network Calibration — Surface Fit Quality",
        color=TEXT_CLR,
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    sep = "   |   "
    true_str = sep.join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, true_params))
    calib_str = sep.join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, calibrated_params))

    fig.text(
        0.5,
        0.94,
        f"True:       {true_str}",
        ha="center",
        color=BLUE,
        fontsize=7.5,
        family="monospace",
    )
    fig.text(
        0.5,
        0.915,
        f"Calibrated: {calib_str}",
        ha="center",
        color=ORANGE,
        fontsize=7.5,
        family="monospace",
    )
    fig.text(
        0.5,
        0.89,
        f"Calibration time: {calib_time_ms:.2f} ms   |   "
        f"Converged: {converged}   |   "
        f"Val RMSE: 0.02605   |   "
        f"Architecture: 5→30→30→30→30→88 (ELU)",
        ha="center",
        color=SUB_CLR,
        fontsize=7.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.89])
    return fig


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load artefacts
    print("[1/5] Loading model and scalers …")
    f_scaler = joblib.load(FEAT_SCALER)
    t_scaler = joblib.load(TARGET_SCALER)
    model = HestonSurrogateMLP()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    # 2. Pick a reproducible test sample directly from raw data (avoids re-fitting scalers)
    print("[2/5] Loading test sample from raw data …")
    with gzip.open(DATA_PATH, "rb") as fh:
        data = np.load(fh)
    X_raw = data[:, :5]
    Y_raw = data[:, 5:]

    _, X_test, _, Y_test = train_test_split(X_raw, Y_raw, test_size=0.15, random_state=42)

    np.random.seed(7)
    idx = np.random.randint(0, len(X_test))
    true_params = X_test[idx]  # (5,) — unscaled Heston params
    target_iv = Y_test[idx]  # (88,) — IV surface in original units

    print(f"   Sample index: {idx}")
    for name, v in zip(PARAM_LABELS, true_params):
        print(f"   {name}: {v:.6f}")

    # 3. Calibrate
    print("[3/5] Running L-BFGS-B calibration …")
    calibrator = HestonCalibrator(model, f_scaler, t_scaler, method="L-BFGS-B")
    t0 = time.perf_counter()
    calibrated_params, opt_result = calibrator.calibrate(target_iv)
    calib_ms = (time.perf_counter() - t0) * 1000
    print(f"   Converged: {opt_result.success}   MSE (scaled): {opt_result.fun:.4e}")
    print(f"   Calibration time: {calib_ms:.2f} ms")

    # 4. Reconstruct calibrated IV surface
    print("[4/5] Reconstructing calibrated surface …")
    calibrated_iv = nn_predict_iv(model, f_scaler, t_scaler, calibrated_params)

    # Print comparison table
    print("\n" + "=" * 62)
    print(f"  {'Parameter':<16}  {'True':>10}  {'Calibrated':>12}  {'Rel Err':>8}")
    print("=" * 62)
    for name, tv, cv in zip(PARAM_LABELS, true_params, calibrated_params):
        pct = abs(tv - cv) / (abs(tv) + 1e-12) * 100
        print(f"  {name:<16}  {tv:>10.6f}  {cv:>12.6f}  {pct:>7.3f}%")
    print("=" * 62)
    print(f"  Calibration time: {calib_ms:.2f} ms\n")

    # 5. Plot & save
    print("[5/5] Generating figure …")
    Z_target = unflatten_surface(target_iv)
    Z_model = unflatten_surface(calibrated_iv)

    fig = make_figure(
        Z_target,
        Z_model,
        true_params,
        calibrated_params,
        calib_ms,
        opt_result.success,
    )

    out_path = RESULTS_DIR / "surface_fit.png"
    fig.savefig(out_path, dpi=200, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
