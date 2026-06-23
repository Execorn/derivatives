"""
Deep Volatility Model Zoo Calibration — Streamlit Demo UI.

Supports:
1. Rough Heston (FiLM-FNO v2)
2. Classic Heston (Fourier-COS + Newton)
3. SABR (Hagan Lognormal + Newton)
4. SSVI (Power-law + Newton)
5. Local Volatility (SVI to Dupire LV surface)
6. Rough Bergomi (Bennedsen hybrid MC + Newton)
"""

import sys, os, time, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_model import MirrorPaddedFNO2d
from calibrate import (_make_spatial_input, _fno_predict_real_iv, _load_normalizers,
                       compute_confidence_scores, compute_fim_ellipsoid,
                       compute_confidence_reparameterized, calibrate_reparameterized)
from calibrate_fast import (calibrate_newton, calibrate_heston, calibrate_sabr,
                            calibrate_ssvi, calibrate_rbergomi, compute_local_vol_surface)
from pricing.heston import heston_iv_surface
from pricing.sabr import sabr_iv_surface, ssvi_iv_surface
from pricing.local_vol import check_arbitrage_free

# ─── Grid ──────────────────────────────────────────────────────────────────
STRIKES    = np.linspace(-0.5, 0.5, 11)
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])

@st.cache_resource
def load_model(model_name: str):
    """Load the appropriate FNO surrogate and load corresponding normalizers."""
    if model_name == "Rough Heston":
        model = MirrorPaddedFNO2d(param_dim=6)
        path = "artifacts/weights/fno_v2_final_prod.pth"
        norm_key = "v2"
    elif model_name == "Classic Heston":
        model = MirrorPaddedFNO2d(param_dim=5)
        path = "artifacts/weights/fno_heston_final_prod.pth"
        norm_key = "heston"
    elif model_name == "SABR":
        model = MirrorPaddedFNO2d(param_dim=3)
        path = "artifacts/weights/fno_sabr_final_prod.pth"
        norm_key = "sabr"
    elif model_name == "SSVI":
        model = MirrorPaddedFNO2d(param_dim=11)
        path = "artifacts/weights/fno_ssvi_final_prod.pth"
        norm_key = "ssvi"
    elif model_name == "Local Volatility":
        model = MirrorPaddedFNO2d(param_dim=40)
        path = "artifacts/weights/fno_localvol_final_prod.pth"
        norm_key = "localvol"
    elif model_name == "Rough Bergomi":
        model = MirrorPaddedFNO2d(param_dim=4)
        path = "artifacts/weights/fno_rbergomi_final_prod.pth"
        norm_key = "rbergomi"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    _load_normalizers(norm_key)
    return model

st.set_page_config(page_title="Deep Volatility Model Zoo NN Calibration", layout="wide")
st.title("⚡ Deep Volatility Model Zoo Calibration")

# ─── Sidebar Model Selector ─────────────────────────────────────────────────
model_name = st.sidebar.selectbox(
    "Volatility Model Type",
    ["Rough Heston", "Classic Heston", "SABR", "SSVI", "Local Volatility", "Rough Bergomi"],
    index=0,
    key="model_selector"
)

# Load the corresponding FNO model
model = load_model(model_name)
device = torch.device("cpu")

st.caption(f"FiLM-FNO Surrogate for {model_name} • Optimized Gauss-Newton autograd Jacobians")

# ─── Sidebar Parameters ─────────────────────────────────────────────────────
st.sidebar.header(f"True {model_name} Parameters")

if model_name == "Rough Heston":
    st.sidebar.info("🔒 Ghost parameters fixed: κ=1.0, θ=0.08, H=0.08")
    kappa = 1.0; theta = 0.08; H = 0.08
    sigma = st.sidebar.slider("σ — Vol of Vol", 0.1, 1.0, 0.5, step=0.01)
    rho = st.sidebar.slider("ρ — Correlation", -0.9, -0.1, -0.5, step=0.01)
    v0 = st.sidebar.slider("v₀ — Initial Variance", 0.01, 0.15, 0.08, step=0.01)
    true_params = np.array([kappa, theta, sigma, rho, v0, H])

elif model_name == "Classic Heston":
    kappa = st.sidebar.slider("κ — Mean Reversion", 0.1, 5.0, 2.0, step=0.1)
    theta = st.sidebar.slider("θ — Long-Run Variance", 0.01, 0.15, 0.05, step=0.01)
    sigma = st.sidebar.slider("σ — Vol of Vol", 0.1, 1.0, 0.3, step=0.01)
    rho = st.sidebar.slider("ρ — Correlation", -0.9, -0.1, -0.6, step=0.01)
    v0 = st.sidebar.slider("v₀ — Initial Variance", 0.01, 0.15, 0.05, step=0.01)
    true_params = np.array([kappa, theta, sigma, rho, v0])

elif model_name == "SABR":
    alpha = st.sidebar.slider("α (alpha) — Initial Vol", 0.05, 0.8, 0.20, step=0.01)
    rho = st.sidebar.slider("ρ (rho) — Correlation", -0.9, 0.9, -0.40, step=0.01)
    nu = st.sidebar.slider("ν (nu) — Vol of Vol", 0.1, 1.2, 0.40, step=0.01)
    true_params = np.array([alpha, rho, nu])

elif model_name == "SSVI":
    rho = st.sidebar.slider("ρ (rho) — Correlation", -0.9, 0.9, -0.30, step=0.01)
    eta = st.sidebar.slider("η (eta) — Power Law Vol", 0.05, 4.0, 0.60, step=0.01)
    gamma = st.sidebar.slider("γ (gamma) — Power Law Exp", 0.1, 0.5, 0.30, step=0.01)
    # ATM variance term structure
    st.sidebar.markdown("**ATM Variances (Monotone term structure)**")
    theta_atm = []
    curr = 0.01
    for i, T in enumerate(MATURITIES):
        val = st.sidebar.slider(f"θ_atm(T={T:.1f})", curr, 0.50, curr + 0.02 * i, step=0.005)
        curr = val
        theta_atm.append(val)
    true_params = np.concatenate([np.array(theta_atm), np.array([rho, eta, gamma])])

elif model_name == "Local Volatility":
    st.sidebar.info("Input base SVI parameters. Slices grow with maturity T.")
    a0 = st.sidebar.slider("a0 (ATM variance base)", 0.01, 0.15, 0.04, step=0.005)
    b0 = st.sidebar.slider("b0 (slope base)", 0.05, 0.4, 0.15, step=0.01)
    rho_svi = st.sidebar.slider("ρ (skew base)", -0.85, -0.15, -0.40, step=0.01)
    m_svi = st.sidebar.slider("m (translation base)", -0.15, 0.15, 0.0, step=0.01)
    sigma_svi = st.sidebar.slider("σ (volatility base)", 0.05, 0.35, 0.15, step=0.01)
    
    # Generate 8 slices of SVI parameters
    svi_params = np.zeros((8, 5))
    for j in range(8):
        T = MATURITIES[j]
        scale = T
        svi_params[j, 0] = a0 * scale
        svi_params[j, 1] = b0 * scale
        svi_params[j, 2] = rho_svi
        svi_params[j, 3] = m_svi
        svi_params[j, 4] = sigma_svi
    true_params = svi_params.flatten()

elif model_name == "Rough Bergomi":
    v0 = st.sidebar.slider("v₀ — Initial Variance", 0.01, 0.20, 0.08, step=0.01)
    H = st.sidebar.slider("H — Hurst Parameter", 0.04, 0.15, 0.07, step=0.01)
    eta = st.sidebar.slider("η — Vol of Vol", 0.5, 4.0, 1.5, step=0.1)
    rho = st.sidebar.slider("ρ — Correlation", -0.95, 0.0, -0.7, step=0.01)
    true_params = np.array([v0, H, eta, rho])

noise_level = st.sidebar.slider(
    "Market Noise Level (Stress Test)", 0.0, 0.10, 0.01, step=0.01, key="noise")

# ─── Target Surface Generation ──────────────────────────────────────────────
if st.sidebar.button("Generate Target Surface", use_container_width=True):
    with torch.no_grad():
        if model_name == "Rough Heston":
            spatial = _make_spatial_input(MATURITIES, STRIKES, device=device)
            params_t = torch.tensor(true_params, dtype=torch.float32).unsqueeze(0)
            target_iv = _fno_predict_real_iv(model, params_t, spatial).numpy()
        elif model_name == "Classic Heston":
            p_dict = {
                'kappa': true_params[0], 'theta': true_params[1],
                'sigma': true_params[2], 'rho': true_params[3], 'v0': true_params[4]
            }
            target_iv = heston_iv_surface(p_dict, MATURITIES, STRIKES)
            # Fill NaNs
            for t_idx in range(len(MATURITIES)):
                slice_t = target_iv[t_idx, :]
                valid_vals = slice_t[np.isfinite(slice_t)]
                med = np.median(valid_vals) if len(valid_vals) > 0 else 0.3
                slice_t[~np.isfinite(slice_t)] = med
                target_iv[t_idx, :] = slice_t
        elif model_name == "SABR":
            target_iv = sabr_iv_surface(
                F=1.0, T_grid=MATURITIES, k_grid=STRIKES,
                alpha=true_params[0], beta=1.0, rho=true_params[1], nu=true_params[2],
                iv_type="lognormal"
            )
        elif model_name == "SSVI":
            target_iv = ssvi_iv_surface(
                T_grid=MATURITIES, k_grid=STRIKES,
                theta_grid=true_params[:8], rho=true_params[8], eta=true_params[9], gamma=true_params[10]
            )
        elif model_name == "Local Volatility":
            target_iv = compute_local_vol_surface(true_params, MATURITIES, STRIKES, use_fno=False)
        elif model_name == "Rough Bergomi":
            spatial = _make_spatial_input(MATURITIES, STRIKES, device=device)
            params_t = torch.tensor(true_params, dtype=torch.float32).unsqueeze(0)
            target_iv = _fno_predict_real_iv(model, params_t, spatial).numpy()

    st.session_state["target_iv"] = target_iv
    st.session_state["true_params"] = true_params.copy()
    st.session_state["active_model"] = model_name
    st.session_state.pop("calib_results", None)
    st.success(f"{model_name} target surface generated.")

# Check if model selection has changed
if "active_model" in st.session_state and st.session_state["active_model"] != model_name:
    st.session_state.pop("target_iv", None)
    st.session_state.pop("true_params", None)
    st.session_state.pop("calib_results", None)

if "target_iv" not in st.session_state:
    st.info("👈 Set parameters in the sidebar and click **Generate Target Surface** to begin.")
    st.stop()

target_iv = st.session_state["target_iv"]
stored_true = st.session_state["true_params"]

# ─── Calibration Section ────────────────────────────────────────────────────
st.subheader("Calibration")

if model_name == "Local Volatility":
    st.markdown("**Dupire Local Volatility Surface Mapping (direct pricing vs FNO surrogate)**")
    if st.button("Evaluate Local Vol Surface", use_container_width=True):
        t0 = time.time()
        lv_exact = target_iv
        lv_fno = compute_local_vol_surface(stored_true, MATURITIES, STRIKES, use_fno=True, model=model)
        elapsed = time.time() - t0
        
        st.session_state["calib_results"] = {
            "exact_lv": lv_exact,
            "fno_lv": lv_fno,
            "elapsed": elapsed,
        }
else:
    if st.button("Calibrate parameters using FNO autograd Newton", use_container_width=True):
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(0, noise_level * np.abs(target_iv), target_iv.shape)
        market_iv_noisy = np.maximum(target_iv + noise, 1e-4)

        with st.spinner("Running Gauss-Newton calibration..."):
            t0 = time.time()
            if model_name == "Rough Heston":
                res = calibrate_newton(model, market_iv_noisy, MATURITIES, STRIKES, max_iter=25, verbose=False)
            elif model_name == "Classic Heston":
                res = calibrate_heston(model, market_iv_noisy, MATURITIES, STRIKES, max_iter=25, n_starts=2)
            elif model_name == "SABR":
                res = calibrate_sabr(model, market_iv_noisy, MATURITIES, STRIKES, max_iter=25, n_starts=2)
            elif model_name == "SSVI":
                res = calibrate_ssvi(model, market_iv_noisy, MATURITIES, STRIKES, max_iter=25, n_starts=2)
            elif model_name == "Rough Bergomi":
                res = calibrate_rbergomi(model, market_iv_noisy, MATURITIES, STRIKES, max_iter=25, n_starts=2)
            
            elapsed = time.time() - t0

        st.session_state["calib_results"] = {
            "res": res,
            "market_iv_noisy": market_iv_noisy,
            "elapsed": elapsed,
        }

# ─── Render Results ──────────────────────────────────────────────────────────
if "calib_results" in st.session_state:
    res_dict = st.session_state["calib_results"]
    elapsed = res_dict["elapsed"]
    
    if model_name == "Local Volatility":
        lv_exact = res_dict["exact_lv"]
        lv_fno = res_dict["fno_lv"]
        
        st.success(f"SVI to Dupire Local Volatility evaluation completed in **{elapsed*1000:.1f} ms**.")
        
        # Display comparison plots
        K_grid, T_grid = np.meshgrid(STRIKES, MATURITIES)
        fig = go.Figure()
        fig.add_trace(go.Surface(x=K_grid, y=T_grid, z=lv_exact,
                                 colorscale="Blues", opacity=0.7, name="Exact Dupire", showscale=False))
        fig.add_trace(go.Surface(x=K_grid, y=T_grid, z=lv_fno,
                                 colorscale="Reds", opacity=0.7, name="FNO Surrogate", showscale=False))
        fig.update_layout(
            title="Exact Finite Differences vs FNO Local Volatility Surface",
            scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity", zaxis_title="Local Vol"),
            margin=dict(l=0, r=0, b=0, t=40), height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # SVI arbitrage free check
        is_arb_free = check_arbitrage_free(MATURITIES, STRIKES, stored_true.reshape(8, 5))
        st.markdown(f"**SVI Arbitrage-Free Check**: {'🟢 Calendar & Butterfly Arbitrage Free' if is_arb_free else '🔴 Arbitrage Violation Detected'}")
        
    else:
        res = res_dict["res"]
        market_iv_noisy = res_dict["market_iv_noisy"]
        
        st.success(f"Calibration completed in **{elapsed*1000:.1f} ms** (Iterations: {res.get('n_iter', 'N/A')}).")
        
        # Parameter tables
        if model_name == "Rough Heston":
            st.markdown("**Calibrated Parameters** (κ=1.0, θ=0.08, H=0.08 fixed)")
            p_calib = np.array([1.0, 0.08, res["sigma"], res["rho"], res["v0"], 0.08])
            p_names = ["kappa", "theta", "sigma", "rho", "v0", "H"]
        elif model_name == "Classic Heston":
            p_calib = res["param_vector"]
            p_names = ["kappa", "theta", "sigma", "rho", "v0"]
        elif model_name == "SABR":
            p_calib = np.array([res["alpha"], res["rho"], res["nu"]])
            p_names = ["alpha", "rho", "nu"]
        elif model_name == "SSVI":
            p_calib = np.concatenate([res["theta_atm"], np.array([res["rho"], res["eta"], res["gamma"]])])
            p_names = [f"theta_atm(T={t:.1f})" for t in MATURITIES] + ["rho", "eta", "gamma"]
        elif model_name == "Rough Bergomi":
            p_calib = np.array([res["v0"], res["H"], res["eta"], res["rho"]])
            p_names = ["v0", "H", "eta", "rho"]

        df = pd.DataFrame({
            "Parameter": p_names,
            "True": stored_true,
            "Calibrated": p_calib,
            "Abs Error": np.abs(stored_true - p_calib)
        })
        st.dataframe(df.style.format({"True": "{:.6f}", "Calibrated": "{:.6f}", "Abs Error": "{:.6f}"}), use_container_width=True)
        
        # Surface comparison
        calibrated_iv = res["iv_fitted"]
        K_grid, T_grid = np.meshgrid(STRIKES, MATURITIES)
        fig = go.Figure()
        fig.add_trace(go.Surface(x=K_grid, y=T_grid, z=market_iv_noisy,
                                 colorscale="Blues", opacity=0.7, name="Market (Noisy)", showscale=False))
        fig.add_trace(go.Surface(x=K_grid, y=T_grid, z=calibrated_iv,
                                 colorscale="Reds", opacity=0.7, name="Calibrated FNO", showscale=False))
        fig.update_layout(
            title=f"Market Noisy Surface vs Calibrated FNO Surface (MSE: {res['final_mse']:.2e})",
            scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity", zaxis_title="IV"),
            margin=dict(l=0, r=0, b=0, t=40), height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Convergence plots
        if "loss_history" in res:
            st.subheader("Gauss-Newton Optimization Convergence")
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=np.arange(1, len(res["loss_history"]) + 1), y=res["loss_history"],
                                        mode="lines+markers", line=dict(color="#00d4ff", width=2)))
            fig_lc.update_layout(
                xaxis_title="Gauss-Newton Iteration", yaxis_title="Objective Loss (MSE)", yaxis_type="log",
                height=300, margin=dict(l=0, r=0, b=40, t=40), plot_bgcolor="#111", paper_bgcolor="#111",
                font=dict(color="#eee"), xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333")
            )
            st.plotly_chart(fig_lc, use_container_width=True)
