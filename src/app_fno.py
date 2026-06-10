"""
Deep Rough Heston Calibration — Streamlit Demo UI (FiLM-FNO version).

Model signature change: model(spatial_coords, theta_norm) instead of model(fno_input).
Normalizers are loaded from artifacts/models/ and applied transparently.
"""

import sys, os, time, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_model import MirrorPaddedFNO2d
from calibrate import (calibrate_parameters, compute_confidence_scores,
                       _make_spatial_input, _fno_predict_real_iv, _load_normalizers)
from normalizers import ParameterNormalizer, IVSurfaceNormalizer

# ─── Grid ──────────────────────────────────────────────────────────────────
STRIKES    = np.linspace(-0.5, 0.5, 11)
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
PARAM_NAMES_DISPLAY = ["κ (kappa)", "θ (theta)", "σ (sigma)", "ρ (rho)", "v₀", "H (Hurst)"]
CONF_NAMES = {
    "kappa": "κ (Mean Reversion)",
    "theta": "θ (Long-Run Var)",
    "sigma": "σ (Vol of Vol)",
    "rho":   "ρ (Correlation)",
    "v0":    "v₀ (Initial Var)",
    "H":     "H (Hurst)",
}

@st.cache_resource
def load_model():
    model = MirrorPaddedFNO2d()
    weights_path = "artifacts/models/fno_best.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu",
                                         weights_only=True))
    model.eval()
    # Pre-load normalizers
    _load_normalizers()
    return model

model = load_model()

st.set_page_config(page_title="Deep Rough Heston NN Calibration", layout="wide")
st.title("Deep Learning Volatility — Rough Heston Calibration")
st.caption("FiLM-conditioned FNO Surrogate + L-BFGS Calibration")

# ─── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("True Rough Heston Parameters")
kappa = st.sidebar.slider("κ — Mean Reversion",   0.1,  5.0,  2.5,  step=0.1,  key="kappa")
theta = st.sidebar.slider("θ — Long-Run Variance", 0.01, 0.15, 0.08, step=0.01, key="theta")
sigma = st.sidebar.slider("σ — Vol of Vol",        0.1,  1.0,  0.5,  step=0.01, key="sigma")
rho   = st.sidebar.slider("ρ — Correlation",      -0.9, -0.1, -0.5,  step=0.01, key="rho")
v0    = st.sidebar.slider("v₀ — Initial Variance", 0.01, 0.15, 0.08, step=0.01, key="v0")
H     = st.sidebar.slider("H — Hurst Parameter",  0.02, 0.15, 0.08, step=0.01, key="H")

st.sidebar.divider()
noise_level = st.sidebar.slider(
    "Market Noise Level (Stress Test)", 0.0, 0.10, 0.01, step=0.01, key="noise")

true_params = np.array([kappa, theta, sigma, rho, v0, H])

# ─── Generate surface ───────────────────────────────────────────────────────
if st.sidebar.button("Generate Target Surface", use_container_width=True):
    with torch.no_grad():
        spatial = _make_spatial_input(MATURITIES, STRIKES, device=torch.device("cpu"))
        params_t = torch.tensor(true_params, dtype=torch.float32).unsqueeze(0)
        target_iv = _fno_predict_real_iv(model, params_t, spatial).numpy()

    st.session_state["target_iv"]   = target_iv
    st.session_state["true_params"] = true_params.copy()
    for key in ("calib_results",):
        st.session_state.pop(key, None)
    st.success("Target IV surface generated.")

if "target_iv" not in st.session_state:
    st.info("👈 Set parameters in the sidebar and click **Generate Target Surface** to begin.")
    st.stop()

target_iv   = st.session_state["target_iv"]
stored_true = st.session_state["true_params"]

# ─── Calibrate ──────────────────────────────────────────────────────────────
st.subheader("Calibration")
if st.button("Calibrate", use_container_width=False, key="calibrate_btn"):
    rng   = np.random.default_rng(seed=0)
    noise = rng.normal(0, noise_level * np.abs(target_iv), target_iv.shape)
    market_iv_noisy = np.maximum(target_iv + noise, 1e-4)

    init_params = np.array([1.5, 0.05, 0.4, -0.4, 0.05, 0.05])

    with st.spinner("Running L-BFGS calibration..."):
        calibrated_params, history, elapsed = calibrate_parameters(
            model, market_iv_noisy, init_params, MATURITIES, STRIKES)

    with st.spinner("Computing parameter confidence (Jacobian norms)..."):
        conf_scores = compute_confidence_scores(
            model, calibrated_params, MATURITIES, STRIKES)

    with torch.no_grad():
        spatial   = _make_spatial_input(MATURITIES, STRIKES, device=torch.device("cpu"))
        params_t  = torch.tensor(calibrated_params, dtype=torch.float32).unsqueeze(0)
        calibrated_iv = _fno_predict_real_iv(model, params_t, spatial).numpy()

    st.session_state["calib_results"] = {
        "calibrated_params": calibrated_params,
        "market_iv_noisy":   market_iv_noisy,
        "calibrated_iv":     calibrated_iv,
        "history":           history,
        "elapsed":           elapsed,
        "conf_scores":       conf_scores,
        "noise_level_used":  noise_level,
    }

# ─── Display results ─────────────────────────────────────────────────────────
if "calib_results" in st.session_state:
    res               = st.session_state["calib_results"]
    calibrated_params = res["calibrated_params"]
    market_iv_noisy   = res["market_iv_noisy"]
    calibrated_iv     = res["calibrated_iv"]
    history           = res["history"]
    elapsed           = res["elapsed"]
    conf_scores       = res["conf_scores"]
    noise_used        = res["noise_level_used"]

    st.success(
        f"Calibration took **{elapsed:.4f}s** ({elapsed*1000:.1f} ms) — "
        f"Final Loss: {history[-1]:.6f} — "
        f"Noise level used: {noise_used*100:.1f}%"
    )

    # Parameter table
    df = pd.DataFrame({
        "Parameter":  PARAM_NAMES_DISPLAY,
        "True":       stored_true,
        "Calibrated": calibrated_params,
        "Abs Error":  np.abs(stored_true - calibrated_params),
    })
    st.dataframe(
        df.style.format({"True": "{:.6f}", "Calibrated": "{:.6f}", "Abs Error": "{:.6f}"}),
        use_container_width=True)

    st.download_button(
        label="Download Calibrated Params (JSON)",
        data=json.dumps({n: float(v) for n, v in
                         zip(["kappa","theta","sigma","rho","v0","H"],
                             calibrated_params)}, indent=4),
        file_name="calibrated_params.json", mime="application/json")

    # Confidence scores
    st.subheader("Parameter Confidence (Identifiability)")
    st.caption(
        "Jacobian column Frobenius norm ‖∂IV/∂θᵢ‖_F in real IV space — "
        "measures total sensitivity of the IV surface to each parameter. "
        "FiLM conditioning routes parameters through all Fourier modes (DC-trap fixed)."
    )
    for pname, score in conf_scores.items():
        label = CONF_NAMES.get(pname, pname)
        st.progress(score, text=f"{label}: {score:.2f}")

    if conf_scores.get("kappa", 1.0) < 0.3:
        st.warning(
            "⚠️ **κ is weakly identified** (confidence < 0.3). "
            "In the deep-rough regime (H < 0.1), the IV surface is insensitive "
            "to κ for T < 0.5. Consider fixing κ from historical estimation.")

    # 3D surface plot
    K_grid, T_grid = np.meshgrid(STRIKES, MATURITIES)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=K_grid, y=T_grid, z=market_iv_noisy,
                             colorscale="Blues", opacity=0.7, name="Target (noisy)",
                             showscale=False))
    fig.add_trace(go.Surface(x=K_grid, y=T_grid, z=calibrated_iv,
                             colorscale="Reds", opacity=0.7, name="Calibrated",
                             showscale=False))
    fig.update_layout(
        title=f"Target (Noise={noise_used*100:.1f}%) vs Calibrated FNO Surface",
        scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity",
                   zaxis_title="IV"),
        margin=dict(l=0, r=0, b=0, t=40), height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Greeks
    st.subheader("Autograd Greeks (Hessian)")
    if st.button("Compute Volga & Vanna Heatmaps", use_container_width=False,
                 key="greeks_btn"):
        from fno_greeks import compute_greeks
        import matplotlib.pyplot as plt
        import seaborn as sns

        with st.spinner("Computing Greeks via Autograd..."):
            t_params  = torch.tensor(calibrated_params, dtype=torch.float32)
            t_T       = torch.tensor(MATURITIES, dtype=torch.float32)
            t_K       = torch.tensor(STRIKES,    dtype=torch.float32)
            volga, vanna = compute_greeks(model, t_params, t_T, t_K)

            fig_g, axes = plt.subplots(1, 2, figsize=(14, 5))
            K_labels = [f"{k:.2f}" for k in STRIKES]
            T_labels = [f"{t:.1f}" for t in MATURITIES]
            import seaborn as sns
            sns.heatmap(volga, xticklabels=K_labels, yticklabels=T_labels,
                        cmap="magma", ax=axes[0])
            axes[0].set_title(r"Volga ($\partial^2 IV / \partial \sigma^2$)")
            sns.heatmap(vanna, xticklabels=K_labels, yticklabels=T_labels,
                        cmap="viridis", ax=axes[1])
            axes[1].set_title(r"Vanna ($\partial^2 IV / \partial \sigma \partial \rho$)")
            st.pyplot(fig_g)
            plt.close(fig_g)
