"""
Heston Neural Network Calibration — Streamlit Demo UI.

Interactive app for the Master's Thesis defense.
Allows the user to set Heston parameters via sliders, generate a synthetic
IV surface from the surrogate NN, add bid-ask noise, and run L-BFGS-B
calibration to recover the parameters.

Usage:
    cd path/to/derivatives
    streamlit run src/app.py

IMPORTANT: Column order in the data file is [v0, rho, sigma, theta, kappa].
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch
import joblib

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from model import HestonSurrogateMLP
from calibrator import HestonCalibrator

# ─── Grid (must match training data) ──────────────────────────────────────────

STRIKES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])  # 11
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])  # 8

# Column order in the training data file:  v0, rho, sigma, theta, kappa
# The scaler was fit on this exact order.  We present sliders in a more
# intuitive order (kappa, theta, sigma, rho, v0) but must build the
# parameter vector in the DATA order.
PARAM_DISPLAY_NAMES = ["κ (kappa)", "θ (theta)", "σ (sigma)", "ρ (rho)", "v₀"]


# ─── Model loading (cached) ───────────────────────────────────────────────────


@st.cache_resource
def load_model_and_scalers():
    """Load model weights and fitted scalers — cached across reruns."""
    f_scaler = joblib.load(PROJECT_ROOT / "artifacts" / "scalers" / "feature_scaler.pkl")
    t_scaler = joblib.load(PROJECT_ROOT / "artifacts" / "scalers" / "target_scaler.pkl")

    model = HestonSurrogateMLP()
    model.load_state_dict(
        torch.load(PROJECT_ROOT / "artifacts" / "weights" / "heston_best.pth", map_location="cpu")
    )
    model.eval()

    calibrator = HestonCalibrator(model, f_scaler, t_scaler, method="L-BFGS-B")
    return model, f_scaler, t_scaler, calibrator


model, f_scaler, t_scaler, calibrator = load_model_and_scalers()


# ─── Page config & header ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Heston NN Calibration",
    layout="wide",
)
st.title("Deep Learning Volatility — Heston Calibration")
st.caption("Horvath, Muguruza & Tomas (2019) • PyTorch surrogate + L-BFGS-B")

# ─── Sidebar: True Heston parameters ──────────────────────────────────────────

st.sidebar.header("True Heston Parameters")

# Slider ranges match the training data bounds
kappa = st.sidebar.slider("κ — Mean Reversion Speed", 1.0, 10.0, 5.0, step=0.1)
theta = st.sidebar.slider("θ — Long-Run Variance", 0.01, 0.20, 0.10, step=0.01)
sigma = st.sidebar.slider("σ — Vol of Vol", 0.01, 1.00, 0.50, step=0.01)
rho = st.sidebar.slider("ρ — Correlation", -0.95, -0.10, -0.50, step=0.01)
v0 = st.sidebar.slider("v₀ — Initial Variance", 0.0001, 0.04, 0.02, step=0.001, format="%.4f")

# Feller condition check
feller_ok = 2 * kappa * theta > sigma**2
if not feller_ok:
    st.sidebar.warning(
        f"Warning: Feller condition violated: 2κθ = {2*kappa*theta:.4f} ≤ σ² = {sigma**2:.4f}"
    )
else:
    st.sidebar.success(f"Feller passed: 2κθ = {2*kappa*theta:.4f} > σ² = {sigma**2:.4f}")


# ─── Build parameter vector in DATA column order ──────────────────────────────
# Data order: [v0, rho, sigma, theta, kappa]


def build_param_vector(kappa, theta, sigma, rho, v0):
    """Build param array in the order the scaler expects: [v0, rho, sigma, theta, kappa]."""
    return np.array([v0, rho, sigma, theta, kappa])


def display_order(vec_data_order):
    """Convert from data order [v0,rho,sigma,theta,kappa] to display order [kappa,theta,sigma,rho,v0]."""
    return np.array(
        [
            vec_data_order[4],
            vec_data_order[3],
            vec_data_order[2],
            vec_data_order[1],
            vec_data_order[0],
        ]
    )


# ─── Generate surface ─────────────────────────────────────────────────────────

if st.sidebar.button("Generate Target Surface", use_container_width=True):
    true_params = build_param_vector(kappa, theta, sigma, rho, v0)

    # Scale → forward pass → inverse-scale
    params_scaled = f_scaler.transform(true_params.reshape(1, -1))
    with torch.no_grad():
        iv_scaled = model(torch.tensor(params_scaled, dtype=torch.float32)).numpy()
    target_iv = t_scaler.inverse_transform(iv_scaled).flatten()

    # Clamp any tiny negative IVs from the linear output layer
    target_iv = np.maximum(target_iv, 1e-6)

    st.session_state["target_iv"] = target_iv
    st.session_state["true_params"] = true_params  # data order
    st.success("Target IV surface generated from neural network.")


# ─── Main view: Calibrate ─────────────────────────────────────────────────────

if "target_iv" in st.session_state:
    st.subheader("Calibration")

    if st.button("Calibrate", use_container_width=False):
        target_iv = st.session_state["target_iv"].copy()
        true_params = st.session_state["true_params"]  # data order

        # Add 1% Gaussian bid-ask noise (safe: use absolute value for scale)
        noise = np.random.normal(0, 0.01 * np.abs(target_iv), target_iv.shape)
        market_iv_noisy = np.maximum(target_iv + noise, 1e-6)

        # Calibrate via L-BFGS-B
        t0 = time.perf_counter()
        calibrated_params, opt_result = calibrator.calibrate(market_iv_noisy)
        elapsed = time.perf_counter() - t0

        st.success(
            f"Calibration took **{elapsed:.4f}s** ({elapsed*1000:.1f} ms)  "
            f"— Converged: {opt_result.success}"
        )

        # ── Comparison table (display order) ──
        true_disp = display_order(true_params)
        calib_disp = display_order(calibrated_params)

        df = pd.DataFrame(
            {
                "Parameter": PARAM_DISPLAY_NAMES,
                "True": true_disp,
                "Calibrated": calib_disp,
                "Abs Error": np.abs(true_disp - calib_disp),
            }
        )
        st.dataframe(
            df.style.format({"True": "{:.6f}", "Calibrated": "{:.6f}", "Abs Error": "{:.6f}"}),
            use_container_width=True,
        )

        # ── Reconstruct calibrated IV surface ──
        calib_scaled = f_scaler.transform(calibrated_params.reshape(1, -1))
        with torch.no_grad():
            calib_iv_scaled = model(torch.tensor(calib_scaled, dtype=torch.float32)).numpy()
        calibrated_iv = t_scaler.inverse_transform(calib_iv_scaled).flatten()
        calibrated_iv = np.maximum(calibrated_iv, 1e-6)

        # ── 3D Plotly surface ──
        K, T = np.meshgrid(STRIKES, MATURITIES)
        Z_target = market_iv_noisy.reshape(8, 11)
        Z_calib = calibrated_iv.reshape(8, 11)

        fig = go.Figure()

        fig.add_trace(
            go.Surface(
                x=K,
                y=T,
                z=Z_target,
                colorscale="Blues",
                opacity=0.7,
                name="Target (noisy)",
                showscale=False,
            )
        )
        fig.add_trace(
            go.Surface(
                x=K,
                y=T,
                z=Z_calib,
                colorscale="Reds",
                opacity=0.7,
                name="Calibrated",
                showscale=False,
            )
        )

        fig.update_layout(
            title="Target Market Surface vs Calibrated Surface",
            scene=dict(
                xaxis_title="Strike (K/S)",
                yaxis_title="Maturity (T)",
                zaxis_title="Implied Volatility",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)
