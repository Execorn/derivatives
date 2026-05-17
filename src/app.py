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
import streamlit.components.v1 as components
import plotly.graph_objects as go
import torch
import joblib

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from model import HestonSurrogateMLP
from calibrator import HestonCalibrator
from seq_model import HestonDynamicsLSTM

# ─── Grid (must match training data) ──────────────────────────────────────────

STRIKES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])  # 11
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])  # 8
T_VECTOR = np.repeat(MATURITIES, 11)  # shape (88,) — for W ↔ IV conversion

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
        torch.load(PROJECT_ROOT / "artifacts" / "weights" / "heston_best.pth",
                   map_location="cpu", weights_only=False)
    )
    model.eval()

    calibrator = HestonCalibrator(model, f_scaler, t_scaler, method="L-BFGS-B")
    return model, f_scaler, t_scaler, calibrator


model, f_scaler, t_scaler, calibrator = load_model_and_scalers()


@st.cache_resource
def load_lstm():
    """Load the trained HestonDynamicsLSTM and its label statistics."""
    lstm_path  = PROJECT_ROOT / "artifacts" / "weights" / "heston_lstm_best.pth"
    stats_path = PROJECT_ROOT / "artifacts" / "scalers" / "lstm_label_stats.npz"
    if not lstm_path.exists() or not stats_path.exists():
        return None, None, None
    lstm = HestonDynamicsLSTM()
    lstm.load_state_dict(torch.load(lstm_path, map_location="cpu", weights_only=False))
    lstm.eval()
    stats = np.load(stats_path)
    return lstm, stats["label_mean"], stats["label_std"]


def _simulate_history(center_params: np.ndarray, n_steps: int = 10) -> tuple:
    """
    Produce a (n_steps, 88) Total Variance surface history by simulating
    a short OU trajectory from ``center_params`` and running each step
    through the trained surrogate MLP.

    Args:
        center_params: Starting Heston params in data order [v0,rho,sigma,theta,kappa].
        n_steps:       Number of historical days to generate (default: 10).

    Returns:
        surfaces: numpy array of shape (n_steps, 88) — W = IV² × T.
        traj:     numpy array of shape (n_steps, 5)  — parameter trajectory.
    """
    # OU calibration constants (data order: v0, rho, sigma, theta, kappa)
    ou_kappa = np.array([2.50, 3.50, 0.80, 1.50, 1.20])
    ou_sigma = np.array([0.040, 0.100, 0.150, 0.020, 0.500])
    ou_lower = np.array([0.010, -0.90, 0.10, 0.020, 1.00])
    ou_upper = np.array([0.150, -0.40, 0.80, 0.120, 6.00])
    dt       = 1.0 / 252.0
    rng      = np.random.default_rng()

    x = np.clip(center_params.copy(), ou_lower, ou_upper)
    traj     = np.empty((n_steps, 5),  dtype=np.float32)
    surfaces = np.empty((n_steps, 88), dtype=np.float32)

    for t in range(n_steps):
        # OU Euler-Maruyama step toward center_params as the long-run mean
        drift     = ou_kappa * (center_params - x) * dt
        diffusion = ou_sigma * np.sqrt(dt) * rng.standard_normal(5)
        x         = np.clip(x + drift + diffusion, ou_lower, ou_upper)
        traj[t]   = x

        # Surrogate forward pass → W surface
        p_scaled = f_scaler.transform(x.reshape(1, -1))
        with torch.no_grad():
            w_sc = model(torch.tensor(p_scaled, dtype=torch.float32)).numpy()
        w = t_scaler.inverse_transform(w_sc).flatten()
        surfaces[t] = np.maximum(w, 1e-8)

    return surfaces, traj



# ─── Page config & header ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Heston NN Calibration",
    layout="wide",
)
st.title("Deep Learning Volatility — Heston Calibration")
st.caption("Horvath, Muguruza & Tomas (2019) • PyTorch surrogate + L-BFGS-B")

tab1, tab2 = st.tabs(["Calibration Demo", "Architecture & Methods"])

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

with tab1:

    if st.sidebar.button("Generate Target Surface", use_container_width=True):
        true_params = build_param_vector(kappa, theta, sigma, rho, v0)

        # Scale → forward pass → inverse-scale (output is Total Variance W)
        params_scaled = f_scaler.transform(true_params.reshape(1, -1))
        with torch.no_grad():
            w_scaled = model(torch.tensor(params_scaled, dtype=torch.float32)).numpy()
        target_w = t_scaler.inverse_transform(w_scaled).flatten()

        # Inverse Total Variance: IV = sqrt(W / T), clamped for stability
        target_iv = np.sqrt(np.maximum(target_w / T_VECTOR, 1e-8))
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

            # ── Reconstruct calibrated IV surface (W → IV) ──
            calib_scaled = f_scaler.transform(calibrated_params.reshape(1, -1))
            with torch.no_grad():
                calib_w_scaled = model(torch.tensor(calib_scaled, dtype=torch.float32)).numpy()
            calibrated_w = t_scaler.inverse_transform(calib_w_scaled).flatten()
            calibrated_iv = np.sqrt(np.maximum(calibrated_w / T_VECTOR, 1e-8))
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

        # ─── Epistemic Uncertainty via MC Dropout ─────────────────────────────────

        st.divider()
        st.subheader("Epistemic Uncertainty (MC Dropout)")
        st.caption(
            "Runs 100 stochastic forward passes with dropout active (MC Dropout — "
            "Gal & Ghahramani, 2016) to approximate the posterior predictive "
            "distribution. Shaded bounds show ±2σ around the mean IV surface."
        )

        if st.button("Estimate Uncertainty", use_container_width=False, key="btn_uncertainty"):
            true_params = st.session_state["true_params"]  # data order: [v0,rho,sigma,theta,kappa]

            with st.spinner("Running 100 MC Dropout forward passes…"):
                mean_iv, std_iv = calibrator.predict_with_uncertainty(
                    true_params, num_samples=100
                )

            # 2-sigma bounds (clamp lower bound to avoid negative IV)
            upper_iv = mean_iv + 2.0 * std_iv
            lower_iv = np.maximum(mean_iv - 2.0 * std_iv, 1e-6)

            K_unc, T_unc = np.meshgrid(STRIKES, MATURITIES)
            Z_mean = mean_iv.reshape(8, 11)
            Z_upper = upper_iv.reshape(8, 11)
            Z_lower = lower_iv.reshape(8, 11)

            fig_unc = go.Figure()

            # ── Upper bound (mean + 2σ) — translucent green ──
            fig_unc.add_trace(
                go.Surface(
                    x=K_unc,
                    y=T_unc,
                    z=Z_upper,
                    colorscale=[[0, "rgba(34,197,94,0.0)"], [1, "rgba(34,197,94,0.35)"]],
                    opacity=1.0,
                    name="Mean + 2σ",
                    showscale=False,
                    showlegend=True,
                    hovertemplate="K=%{x:.2f}<br>T=%{y:.2f}<br>IV (upper)=%{z:.4f}<extra>Mean + 2σ</extra>",
                )
            )

            # ── Mean surface — opaque Viridis ──
            fig_unc.add_trace(
                go.Surface(
                    x=K_unc,
                    y=T_unc,
                    z=Z_mean,
                    colorscale="Viridis",
                    opacity=0.90,
                    name="Mean IV",
                    showscale=True,
                    colorbar=dict(title="IV", thickness=15, x=1.02),
                    hovertemplate="K=%{x:.2f}<br>T=%{y:.2f}<br>IV (mean)=%{z:.4f}<extra>Mean IV</extra>",
                )
            )

            # ── Lower bound (mean − 2σ) — translucent red ──
            fig_unc.add_trace(
                go.Surface(
                    x=K_unc,
                    y=T_unc,
                    z=Z_lower,
                    colorscale=[[0, "rgba(239,68,68,0.0)"], [1, "rgba(239,68,68,0.35)"]],
                    opacity=1.0,
                    name="Mean − 2σ",
                    showscale=False,
                    showlegend=True,
                    hovertemplate="K=%{x:.2f}<br>T=%{y:.2f}<br>IV (lower)=%{z:.4f}<extra>Mean − 2σ</extra>",
                )
            )

            fig_unc.update_layout(
                title="MC Dropout — Mean IV Surface ± 2σ Epistemic Uncertainty Bounds",
                scene=dict(
                    xaxis_title="Strike (K/S)",
                    yaxis_title="Maturity (T)",
                    zaxis_title="Implied Volatility",
                ),
                margin=dict(l=0, r=0, b=0, t=50),
                height=680,
            )

            st.plotly_chart(fig_unc, use_container_width=True)

            # ── Summary metrics ──
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean IV (μ)", f"{mean_iv.mean():.4f}")
            with col2:
                st.metric("Mean Std (σ̄)", f"{std_iv.mean():.6f}")
            with col3:
                st.metric("Max Std (σ_max)", f"{std_iv.max():.6f}")


    # ─── Phase 3: LSTM Parameter Forecast ────────────────────────────────────────





    if "target_iv" in st.session_state:
        st.divider()
        st.subheader("Parameter Forecast (LSTM Temporal Dynamics)")
        st.caption(
            "Simulates a 10-day Heston parameter history via Ornstein\u2013Uhlenbeck dynamics, "
            "converts each day to a Total Variance surface through the surrogate MLP, "
            "then feeds the sequence into the trained LSTM to predict tomorrow\u2019s parameters "
            "(Cont & Da Fonseca, 2002; Bergomi, 2016)."
        )

        lstm_model, label_mean, label_std = load_lstm()

        if lstm_model is None:
            st.warning(
                "LSTM checkpoint not found. "
                "Run `python src/train_seq.py` to train the sequence model first."
            )
        else:
            if st.button("Forecast Next-Day Parameters", key="btn_lstm"):
                current_params = build_param_vector(kappa, theta, sigma, rho, v0)

                with st.spinner("Simulating 10-day history and running LSTM forecast (50 MC samples)\u2026"):
                    history_surfaces, history_traj = _simulate_history(current_params, n_steps=10)

                    # LSTM MC Dropout forward passes: (1, 10, 88) -> mean (1,5), std (1,5)
                    # Pass label statistics for correct Z-score denormalization: the LSTM
                    # is trained with MSE on Z-scored labels, so inference must reverse
                    # that normalization rather than applying sigmoid bounding.
                    x_seq = torch.tensor(history_surfaces[np.newaxis], dtype=torch.float32)
                    lm_t  = torch.tensor(label_mean, dtype=torch.float32)  # (5,)
                    ls_t  = torch.tensor(label_std,  dtype=torch.float32)  # (5,)
                    pred_mean, pred_std = lstm_model.predict_with_uncertainty(
                        x_seq, num_samples=50, label_mean=lm_t, label_std=ls_t
                    )
                    pred_params     = pred_mean.numpy().flatten()    # (5,) -- point estimate
                    pred_params_std = pred_std.numpy().flatten()     # (5,) -- +-1sigma per param

                # -- 5-panel subplot chart (one panel per parameter) ---------------
                # All 5 Heston parameters have incompatible natural scales:
                #   kappa ~ 1-6,  v0 ~ 0.01-0.15,  theta ~ 0.02-0.12
                #   rho ~ -0.9 to -0.4,  sigma ~ 0.1-0.8
                # A shared y-axis renders v0/theta as invisible flat lines dominated by kappa.
                # Each subplot has its own independent y-axis.
                from plotly.subplots import make_subplots

                display_labels = ["v\u2080", "\u03c1 (rho)", "\u03c3 (sigma)", "\u03b8 (theta)", "\u03ba (kappa)"]
                days_hist = list(range(1, 11))
                colors    = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

                fig_traj = make_subplots(
                    rows=5, cols=1,
                    shared_xaxes=True,
                    subplot_titles=display_labels,
                    vertical_spacing=0.06,
                )

                for row_idx, (label, color) in enumerate(zip(display_labels, colors), start=1):
                    p_idx     = row_idx - 1
                    two_sigma = 2.0 * float(pred_params_std[p_idx])

                    # Historical 10-day trajectory
                    fig_traj.add_trace(go.Scatter(
                        x=days_hist,
                        y=history_traj[:, p_idx].tolist(),
                        mode="lines+markers",
                        name=f"{label} history",
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        showlegend=False,
                    ), row=row_idx, col=1)

                    # Bridge + forecast star with +-2sigma error bar
                    fig_traj.add_trace(go.Scatter(
                        x=[10, 11],
                        y=[float(history_traj[-1, p_idx]), float(pred_params[p_idx])],
                        mode="lines+markers",
                        name=f"{label} forecast",
                        line=dict(color="orange", width=2, dash="dash"),
                        marker=dict(
                            size=9, symbol="star",
                            color="orange",
                            line=dict(color="white", width=1),
                        ),
                        showlegend=False,
                        error_y=dict(
                            type="data",
                            array=[0.0, two_sigma],
                            arrayminus=[0.0, two_sigma],
                            visible=True,
                            color="rgba(255,165,0,0.5)",
                            thickness=1.5,
                            width=5,
                        ),
                    ), row=row_idx, col=1)

                fig_traj.update_layout(
                    title=dict(
                        text="10-Day Parameter Trajectory + Day-11 LSTM Forecast (\u00b12\u03c3 MC Dropout)",
                        font=dict(size=14),
                    ),
                    height=720,
                    margin=dict(l=60, r=20, b=50, t=60),
                    showlegend=False,
                )
                fig_traj.update_xaxes(
                    title_text="Trading Day",
                    tickvals=list(range(1, 12)),
                    ticktext=[str(d) for d in range(1, 11)] + ["Day 11\u2605"],
                    row=5, col=1,
                )
                st.plotly_chart(fig_traj, use_container_width=True)

                # -- Comparison table with +-2sigma uncertainty --------------------
                st.markdown("**Current slider parameters vs. LSTM-predicted next-day parameters**")

                current_disp  = display_order(current_params)
                pred_disp     = display_order(pred_params)
                pred_std_disp = display_order(pred_params_std)

                df_lstm = pd.DataFrame({
                    "Parameter":              PARAM_DISPLAY_NAMES,
                    "Current":                current_disp,
                    "LSTM Forecast (Day +1)": pred_disp,
                    "\u00b12\u03c3 (epistemic)":         2.0 * pred_std_disp,
                    "\u0394 Change":               pred_disp - current_disp,
                })
                st.dataframe(
                    df_lstm.style.format({
                        "Current":                "{:.6f}",
                        "LSTM Forecast (Day +1)": "{:.6f}",
                        "\u00b12\u03c3 (epistemic)":         "{:.6f}",
                        "\u0394 Change":               "{:+.6f}",
                    }),
                    use_container_width=True,
                )

                # -- Per-parameter uncertainty metric cards -----------------------
                st.caption("Epistemic uncertainty (\u00b12\u03c3) per parameter \u2014 50 MC Dropout samples")
                unc_cols = st.columns(5)
                for uc, (pname, pmean, pstd) in zip(
                    unc_cols,
                    zip(PARAM_DISPLAY_NAMES, pred_disp, pred_std_disp),
                ):
                    uc.metric(
                        label=pname,
                        value=f"{pmean:.5f}",
                        delta=f"\u00b1{2*pstd:.5f}",
                    )

                # -- Feller condition on the forecast -----------------------------
                kp = float(pred_params[4])
                th = float(pred_params[3])
                sg = float(pred_params[2])
                if 2 * kp * th > sg ** 2:
                    st.success(
                        f"Forecast Feller: 2\u03ba\u03b8 = {2*kp*th:.4f} > \u03c3\u00b2 = {sg**2:.4f} \u2713"
                    )
                else:
                    st.warning(
                        f"Forecast Feller violated: 2\u03ba\u03b8 = {2*kp*th:.4f} \u2264 \u03c3\u00b2 = {sg**2:.4f}"
                    )


with tab2:
    # ─── Architecture Diagram Tab ─────────────────────────────────────────────────

    st.subheader("Architecture Diagram")
    
    components.html(
        r"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
            <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
            <style>
                body { font-family: 'Inter', 'Segoe UI', sans-serif; margin: 0; padding: 0; background-color: transparent; color: #fff; }
                #cy { width: 100%; height: 900px; background-color: #050505; border-radius: 12px; border: 1px solid #333; }
                #popup { 
                    position: absolute; right: 30px; top: 30px; width: 400px;
                    background-color: #1a1a1a; padding: 30px; border-radius: 12px;
                    display: none; box-shadow: 0 10px 40px rgba(0,0,0,0.9);
                    border: 1px solid #555; z-index: 10;
                }
                .badge { padding: 8px 16px; border-radius: 6px; font-size: 16px; font-weight: 700; margin-bottom: 20px; display: inline-block; text-transform: uppercase; letter-spacing: 1px;}
                .phase1 { background-color: #f1f1f1; color: #000; }
                .phase2 { background-color: #888888; color: #fff; }
                .phase3 { background-color: #333333; color: #fff; border: 1px solid #555; }
                
                h3 { font-size: 24px; font-weight: 700; margin-top: 0; margin-bottom: 20px; border-bottom: 2px solid #333; padding-bottom: 15px; }
                p { font-size: 18px; line-height: 1.6; color: #ddd; }
                pre { background-color: #000; padding: 20px; border-radius: 8px; overflow-x: auto; font-size: 16px; border: 1px solid #444; color: #eee; }
                .legend { position: absolute; top: 30px; left: 30px; z-index: 5; background: rgba(0,0,0,0.9); padding: 20px 25px; border-radius: 10px; display: flex; flex-direction: column; gap: 15px; font-size: 18px; border: 1px solid #444; }
                .legend-item { display: flex; align-items: center; gap: 12px; font-weight: 600; }
                .legend-color { width: 24px; height: 24px; border-radius: 6px; }
            </style>
        </head>
        <body>
            <div style="position: relative;">
                <div class="legend">
                    <div class="legend-item"><div class="legend-color phase1"></div>Phase 1 (MLP Surrogate)</div>
                    <div class="legend-item"><div class="legend-color phase2"></div>Phase 2 (MC Dropout)</div>
                    <div class="legend-item"><div class="legend-color phase3"></div>Phase 3 (LSTM)</div>
                </div>
                <div id="cy"></div>
                <div id="popup">
                    <div id="popup-badge" class="badge"></div>
                    <h3 id="popup-title"></h3>
                    <p id="popup-desc"></p>
                    <div id="popup-formula" style="margin-bottom: 20px; font-size: 20px;"></div>
                    <div id="popup-code"></div>
                </div>
            </div>
            
            <script>
                const elements = [
                    { data: { id: 'params_in', label: '5 Heston Params\n[v₀,ρ,σ,θ,κ]', phase: 1, desc: 'Parameter meanings, Feller condition 2κθ>σ², slider ranges', code: 'kappa = st.sidebar.slider(...)', formula: '' }, position: {x: 200, y: 100} },
                    { data: { id: 'minmax', label: 'MinMaxScaler\n→ [-1, 1]', phase: 1, desc: 'sklearn MinMaxScaler, why [-1,1] suits ELU', code: 'f_scaler.transform(p)', formula: '' }, position: {x: 200, y: 220} },
                    { data: { id: 'layer1', label: 'Linear(5→30)\nELU · Drop(0.1)', phase: 1, desc: 'Xavier init, ELU formula eˣ−1 for x≤0, C² smoothness proof', code: 'nn.Linear(5, 30)\nnn.ELU()\nnn.Dropout(0.1)', formula: 'f(x) = x \\text{ if } x > 0 \\text{ else } e^x - 1' }, position: {x: 200, y: 340} },
                    { data: { id: 'layer2', label: 'Linear(30→30)\nELU · Drop(0.1)', phase: 1, desc: 'Hidden layer 2', code: '', formula: '' }, position: {x: 200, y: 460} },
                    { data: { id: 'layer3', label: 'Linear(30→30)\nELU · Drop(0.1)', phase: 1, desc: 'Hidden layer 3', code: '', formula: '' }, position: {x: 200, y: 580} },
                    { data: { id: 'layer4', label: 'Linear(30→30)\nELU · Drop(0.1)', phase: 1, desc: 'Hidden layer 4', code: '', formula: '' }, position: {x: 200, y: 700} },
                    { data: { id: 'out_linear', label: 'Linear(30→88)\n(no activation)', phase: 1, desc: 'Why no output activation: W can be any positive real', code: 'nn.Linear(30, 88)', formula: '' }, position: {x: 200, y: 820} },
                    { data: { id: 'stdscaler_inv', label: 'StandardScaler⁻¹\n→ W surface', phase: 1, desc: 'Inverse: W = z·σ + μ; fit on training W values', code: 'target_w = t_scaler.inverse_transform(...)', formula: 'W = z \\cdot \\sigma + \\mu' }, position: {x: 200, y: 940} },
                    { data: { id: 'iv_recover', label: 'IV = √(W/T)\n88 grid points', phase: 1, desc: 'Total Variance definition, maturity grid 8×11', code: 'target_iv = np.sqrt(np.maximum(target_w / T_VECTOR, 1e-8))', formula: 'IV = \\sqrt{W/T}' }, position: {x: 200, y: 1060} },
                    
                    { data: { id: 'lbfgsb', label: 'L-BFGS-B\nOptimizer', phase: 1, desc: 'Quasi-Newton, bounded, uses autograd Jacobian; 47–135 ms typical', code: 'scipy.optimize.minimize(..., method="L-BFGS-B")', formula: '' }, position: {x: 550, y: 1060} },
                    { data: { id: 'feller', label: 'Feller Barrier\n2κθ > σ²', phase: 1, desc: 'Returns 1e6 loss if violated; hard penalty; ensures vₜ>0 a.s.', code: 'if 2 * kappa * theta <= sigma**2:\n    return 1e6', formula: '2\\kappa\\theta > \\sigma^2' }, position: {x: 900, y: 940} },
                    { data: { id: 'cal_arb', label: 'Calendar Arb\n∂W/∂T ≥ 0', phase: 1, desc: 'Carr-Madan condition on W; why NOT ∂IV/∂T (fails when v₀>θ)', code: 'loss += lambda * torch.sum(F.relu(-dW_dT)**2)', formula: '\\frac{\\partial W}{\\partial T} \\ge 0' }, position: {x: 900, y: 1060} },
                    { data: { id: 'but_arb', label: 'Butterfly Arb\n∂²IV/∂K² ≥ 0', phase: 1, desc: 'Convexity in strike; soft L2 penalty', code: 'loss += lambda * torch.sum(F.relu(-d2IV_dK2)**2)', formula: '\\frac{\\partial^2 IV}{\\partial K^2} \\ge 0' }, position: {x: 900, y: 1180} },

                    { data: { id: 'mc_train', label: 'model.train()\n(dropout active)', phase: 2, desc: 'Gal & Ghahramani (2016): dropout at test ≈ Bayesian posterior', code: 'model.train()', formula: '' }, position: {x: 550, y: 820} },
                    { data: { id: 'mc_passes', label: 'N=100 Forward\nPasses', phase: 2, desc: 'Each pass samples a different dropout mask; stochastic output', code: 'preds = [model(x) for _ in range(100)]', formula: '' }, position: {x: 850, y: 820} },
                    { data: { id: 'mc_stats', label: 'Mean ± 2σ\nIV Surface', phase: 2, desc: 'Posterior predictive: p(σ*|θ,D) ≈ (1/N)Σ Nwᵢ(θ)', code: 'mean_iv, std_iv = torch.mean(preds), torch.std(preds)', formula: 'p(\\sigma^*|\\theta,D) \\approx \\frac{1}{N}\\sum N_{w_i}(\\theta)' }, position: {x: 1150, y: 820} },

                    { data: { id: 'w_history', label: '10-day W history\n(10 × 88)', phase: 3, desc: 'Sliding window from OU simulation; each row = one W surface', code: 'x_seq = torch.tensor(history_surfaces[np.newaxis])', formula: '' }, position: {x: 200, y: 1350} },
                    { data: { id: 'lstm1', label: 'LSTM Layer 1\nhidden=64', phase: 3, desc: 'Forget-gate bias=1 trick; orthogonal recurrent init; captures vol regime', code: 'nn.LSTM(88, 64, num_layers=2)', formula: '' }, position: {x: 500, y: 1350} },
                    { data: { id: 'lstm2', label: 'LSTM Layer 2\nhidden=64', phase: 3, desc: 'Dropout(0.2) between layers; stacks temporal abstractions', code: '', formula: '' }, position: {x: 800, y: 1350} },
                    { data: { id: 'layer_norm', label: 'LayerNorm(64)', phase: 3, desc: 'Stabilises training on long sequences; applied to final hidden state', code: 'nn.LayerNorm(64)', formula: '' }, position: {x: 1100, y: 1350} },
                    { data: { id: 'head', label: 'Linear(64→5)\nraw logits', phase: 3, desc: 'Xavier init; outputs Z-score normalized param predictions', code: 'nn.Linear(64, 5)', formula: '' }, position: {x: 1400, y: 1350} },
                    { data: { id: 'denorm', label: 'Z-score Denorm\n+ Clamp', phase: 3, desc: 'params = raw*label_std + label_mean; clamp to [lower, upper]', code: 'params = raw * label_std + label_mean', formula: 'Z = \\frac{y-\\mu}{\\sigma}' }, position: {x: 1400, y: 1500} },
                    { data: { id: 'params_out', label: 'Next-day params\n[v₀,ρ,σ,θ,κ]', phase: 3, desc: 'Feller checked on output; MC Dropout: 50 passes → mean ± 2σ', code: '', formula: '' }, position: {x: 1100, y: 1500} },

                    { data: { source: 'params_in', target: 'minmax' } },
                    { data: { source: 'minmax', target: 'layer1' } },
                    { data: { source: 'layer1', target: 'layer2' } },
                    { data: { source: 'layer2', target: 'layer3' } },
                    { data: { source: 'layer3', target: 'layer4' } },
                    { data: { source: 'layer4', target: 'out_linear' } },
                    { data: { source: 'out_linear', target: 'stdscaler_inv' } },
                    { data: { source: 'stdscaler_inv', target: 'iv_recover' } },
                    
                    { data: { source: 'iv_recover', target: 'lbfgsb' }, classes: 'dashed' },
                    { data: { source: 'lbfgsb', target: 'feller' }, classes: 'dashed' },
                    { data: { source: 'lbfgsb', target: 'cal_arb' }, classes: 'dashed' },
                    { data: { source: 'lbfgsb', target: 'but_arb' }, classes: 'dashed' },
                    { data: { source: 'feller', target: 'lbfgsb' }, classes: 'dashed loop' },
                    { data: { source: 'cal_arb', target: 'lbfgsb' }, classes: 'dashed loop' },
                    { data: { source: 'but_arb', target: 'lbfgsb' }, classes: 'dashed loop' },

                    { data: { source: 'out_linear', target: 'mc_train' } },
                    { data: { source: 'mc_train', target: 'mc_passes' } },
                    { data: { source: 'mc_passes', target: 'mc_stats' } },

                    { data: { source: 'w_history', target: 'lstm1' } },
                    { data: { source: 'lstm1', target: 'lstm2' } },
                    { data: { source: 'lstm2', target: 'layer_norm' } },
                    { data: { source: 'layer_norm', target: 'head' } },
                    { data: { source: 'head', target: 'denorm' } },
                    { data: { source: 'denorm', target: 'params_out' } },
                ];

                const cy = cytoscape({
                    container: document.getElementById('cy'),
                    elements: elements,
                    style: [
                        {
                            selector: 'node',
                            style: {
                                'label': 'data(label)',
                                'text-wrap': 'wrap',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'font-size': '18px',
                                'font-weight': '600',
                                'width': '220px',
                                'height': '90px',
                                'shape': 'round-rectangle',
                                'border-width': '2px',
                                'border-color': '#444'
                            }
                        },
                        {
                            selector: 'node[phase = 1]',
                            style: { 'background-color': '#f1f1f1', 'color': '#000', 'border-color': '#aaa' }
                        },
                        {
                            selector: 'node[phase = 2]',
                            style: { 'background-color': '#888888', 'color': '#fff' }
                        },
                        {
                            selector: 'node[phase = 3]',
                            style: { 'background-color': '#333333', 'color': '#fff' }
                        },
                        {
                            selector: 'edge',
                            style: {
                                'width': 4,
                                'line-color': '#666',
                                'target-arrow-color': '#666',
                                'target-arrow-shape': 'triangle',
                                'curve-style': 'bezier',
                                'arrow-scale': 1.8
                            }
                        },
                        {
                            selector: '.dashed',
                            style: {
                                'line-style': 'dashed',
                                'line-dash-pattern': [10, 5]
                            }
                        }
                    ],
                    layout: { name: 'preset' },
                    userZoomingEnabled: true,
                    userPanningEnabled: true,
                    boxSelectionEnabled: false
                });

                // Crucial fix: wait for the container to become visible, or resize continuously.
                const resizeObserver = new ResizeObserver(() => {
                    if (document.getElementById('cy').clientWidth > 0) {
                        cy.resize();
                        cy.fit(undefined, 80);
                    }
                });
                resizeObserver.observe(document.getElementById('cy'));

                cy.on('tap', 'node', function(evt){
                    const node = evt.target;
                    const d = node.data();
                    
                    const popup = document.getElementById('popup');
                    const badge = document.getElementById('popup-badge');
                    
                    popup.style.display = 'block';
                    
                    badge.className = 'badge phase' + d.phase;
                    badge.innerText = 'Phase ' + d.phase;
                    
                    document.getElementById('popup-title').innerText = d.label.replace(/\n/g, ' ');
                    document.getElementById('popup-desc').innerText = d.desc;
                    
                    const fDiv = document.getElementById('popup-formula');
                    if(d.formula) {
                        katex.render(d.formula, fDiv, {displayMode: true});
                    } else {
                        fDiv.innerHTML = '';
                    }
                    
                    const cDiv = document.getElementById('popup-code');
                    if(d.code) {
                        cDiv.innerHTML = '<pre><code>' + d.code.replace(/</g, "&lt;").replace(/>/g, "&gt;") + '</code></pre>';
                    } else {
                        cDiv.innerHTML = '';
                    }
                });

                cy.on('tap', function(evt){
                    if(evt.target === cy){
                        document.getElementById('popup').style.display = 'none';
                    }
                });
            </script>
        </body>
        </html>
        """,
        height=930
    )

    st.markdown("---")
    
    # ─── Methods Reference ────────────────────────────────────────────────────────

    st.subheader("Methods Reference")

    with st.expander("The Heston Model — Mathematical Foundation"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **SDEs**:
            $$ dS_t = \\mu S_t dt + \\sqrt{v_t} S_t dW_t^S $$
            $$ dv_t = \\kappa(\\theta - v_t)dt + \\sigma\\sqrt{v_t} dW_t^v $$
            **Correlation**:
            $$ dW_t^S dW_t^v = \\rho dt $$
            """)
        with col2:
            st.markdown("""
            **Parameters**:
            - $\\kappa$: Mean reversion speed
            - $\\theta$: Long-run variance
            - $\\sigma$: Vol of vol
            - $\\rho$: Correlation
            - $v_0$: Initial variance
            """)
        st.markdown("""
        **Feller condition**: $2\\kappa\\theta > \\sigma^2$ — ensures $v_t > 0$ almost surely.
        The model characteristic function has a closed form (Heston 1993).
        [Read more on Wikipedia](https://en.wikipedia.org/wiki/Heston_model)
        """)

    with st.expander("Total Variance W = IV² × T"):
        st.markdown("""
        **Definition**:
        $$ W(K,T) = \\sigma_{IV}^2(K,T) \\cdot T $$
        
        **Why W instead of raw IV**:
        - Calendar arbitrage condition in W-space: $\\frac{\\partial W}{\\partial T} \\ge 0$ (always valid, even when $v_0 > \\theta$)
        - Raw IV condition $\\frac{\\partial IV}{\\partial T} \\ge 0$ FAILS for inverted term structures ($v_0 > \\theta$)
        - Dupire local vol formula is linear in W → smoother objective
        
        **W → IV recovery**:
        $$ IV = \\sqrt{W/T} $$
        Clamped at `1e-8` for numerical stability.
        [Read more on Wikipedia](https://en.wikipedia.org/wiki/Local_volatility)
        """)

    with st.expander("MLP Surrogate — Architecture and Training"):
        st.markdown("""
        **Architecture**: 
        Input(5) → [Linear(30) → ELU → Dropout(0.1)] × 4 → Linear(88)
        
        **Why ELU**:
        $$ f(x) = \\begin{cases} x & \\text{if } x > 0 \\\\ e^x - 1 & \\text{if } x \\le 0 \\end{cases} $$
        $C^2$ smooth → valid Hessian. Proved: $\\|H_{ELU}\\|_F = 1.039$ vs $\\|H_{ReLU}\\|_F = 0.000$
        
        **Details**:
        - Xavier initialization for all linear layers
        - Training: Adam ($\\eta=1e-3$), ReduceLROnPlateau (patience=15), 200 epochs
        - Scalers: MinMaxScaler for features → $[-1,1]$; StandardScaler for W targets
        - Benchmark: Val RMSE 0.0876 (W-space), calibration 47–135 ms on CPU
        
        [Reference: Horvath, Muguruza & Tomas (2019)](https://ssrn.com/abstract=3322085)
        """)

    with st.expander("L-BFGS-B Calibration — Optimizer and Constraints"):
        st.markdown("""
        **Objective**:
        $$ \\mathcal{L} = \\text{MSE}(N_w(\\theta), \\sigma^*) + \\ell_F + \\ell_C + \\ell_B $$
        
        **Penalties**:
        - **Feller penalty $\\ell_F$**: returns 1e6 (hard barrier) if $2\\kappa\\theta \\le \\sigma^2$
        - **Calendar arb $\\ell_C$**: $\\lambda \\sum [\\text{relu}(-\\frac{\\partial W}{\\partial T})]^2$ , $\\lambda=1e-4$
        - **Butterfly arb $\\ell_B$**: $\\lambda \\sum [\\text{relu}(-\\frac{\\partial^2 IV}{\\partial K^2})]^2$ , $\\lambda=1e-4$
        
        **Optimization**:
        - Gradient: exact Jacobian via PyTorch autograd ($\\frac{\\partial \\mathcal{L}}{\\partial \\theta}$ in one backward pass)
        - Bounds: parameters constrained to the MinMaxScaler's $[-1,1]$ feature domain
        
        [Read more on Wikipedia](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
        """)

    with st.expander("MC Dropout — Epistemic Uncertainty Estimation"):
        st.markdown("""
        **Gal & Ghahramani (2016)**: running N forward passes with dropout active approximates the posterior predictive distribution $p(y^*|x, D)$.
        
        **Formula**:
        $$ p(\\sigma^*|\\theta,D) \\approx \\frac{1}{N} \\sum_i N_{w_i}(\\theta) \\quad \\text{where } w_i \\sim q^*(w) $$
        
        **Implementation**:
        - `model.train()` at inference, 100 passes, compute mean + std
        - The $\\pm 2\\sigma$ bands on the IV surface represent epistemic (model) uncertainty
        - Tight bands (max $\\sigma < 5e-4$) confirm the surrogate has low aleatoric uncertainty in the Heston pricing domain
        
        [Reference: Gal & Ghahramani (2016)](https://arxiv.org/abs/1506.02142) | [Read more on Wikipedia](https://en.wikipedia.org/wiki/Dropout_(neural_networks))
        """)

    with st.expander("Ornstein–Uhlenbeck Parameter Dynamics"):
        st.markdown("""
        **OU SDE**:
        $$ dX_t = \\kappa_{OU}(\\mu - X_t)dt + \\sigma_{OU} dW_t $$
        
        **Euler-Maruyama discretisation**:
        $$ X_{t+1} = X_t + \\kappa_{OU}(\\mu - X_t)\\Delta t + \\sigma_{OU}\\sqrt{\\Delta t} Z_t $$
        where $Z_t \\sim \\mathcal{N}(0,1)$, $\\Delta t = 1/252$
        
        **Details**:
        - 5 correlated OU processes (one per Heston parameter) via Cholesky-factored correlation matrix (empirical SPX dynamics, Cont & Da Fonseca 2002)
        - Hard clamp to empirical bounds after each step
        - Feller repair: if $2\\kappa\\theta \\le \\sigma^2$, reduce $\\sigma$ by 10% iteratively until satisfied
        - OU parameters calibrated to empirical SPX dynamics from Bergomi (2016)
        
        [Read more on Wikipedia](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
        """)

    with st.expander("LSTM — Temporal Parameter Forecasting"):
        st.markdown("""
        **Architecture**: (B, 10, 88) → LSTM(64, 2 layers) → LayerNorm(64) → Linear(64,5)
        
        **LSTM equations**:
        $$ f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f) $$
        
        **Tricks**:
        - **Orthogonal initialization** of recurrent weights (Saxe et al. 2014) — improves gradient flow
        - **Forget-gate bias trick**: $b_f$ initialized to 1 → prevents vanishing gradients at start
        - **LayerNorm** on final hidden state: stabilises long-sequence training
        
        **Training & Inference**:
        - Training: MSE on Z-score normalized labels ($Z = (y-\\mu)/\\sigma$), prevents gradient starvation from scale difference $\\kappa \\approx 3$ vs $v_0 \\approx 0.02$
        - Inference: raw logit → Z-score denorm → physical clamp → physically valid parameters
        - MC Dropout at inference: 50 passes with `model.train()`
        - Test RMSE: 0.336 (normalized), early stopping at epoch 122/200
        
        [Reference: Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf) | [Read more on Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory)
        """)

    with st.expander("Dataset Construction — Sliding Windows"):
        st.markdown("""
        **Structure**:
        - $X[i]$ = W surfaces $[i : i+10]$ — shape `(10, 88)`
        - $y[i]$ = Heston parameters at day $i+10$ — shape `(5,)`
        
        **Details**:
        - 600 trajectories × 170 windows = 102,000 total sequences
        - 80/10/10 split on trajectory boundaries (NOT on window indices) → zero data leakage
        - Augmentation during training: Gaussian noise $\\mathcal{N}(0, 0.001)$ added to input surfaces
        """)
