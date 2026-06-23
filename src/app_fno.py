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

import requests

def render_neural_sde_panel():
    st.header("🔮 Neural SDE Calibration")
    st.markdown("""
    This panel calibrates a **Non-parametric Neural SDE** model to an implied volatility surface.
    The drift $f_\\theta(t, V_t)$ and diffusion $g_\\theta(t, V_t)$ are parameterized as neural networks 
    and calibrated using the SDE Adjoint method.
    """)
    
    # Check if we have an active target IV surface in session state, if not, generate a mock Heston surface
    if "target_iv" in st.session_state:
        target_iv = st.session_state["target_iv"]
        st.info(f"Using active target surface from **{st.session_state.get('active_model')}**")
    else:
        st.warning("No target surface found. Generating a default Classic Heston target surface...")
        # Generate default surface
        p_dict = {'kappa': 2.0, 'theta': 0.05, 'sigma': 0.3, 'rho': -0.6, 'v0': 0.05}
        target_iv = heston_iv_surface(p_dict, MATURITIES, STRIKES)
        # Fill NaNs
        for t_idx in range(len(MATURITIES)):
            slice_t = target_iv[t_idx, :]
            valid_vals = slice_t[np.isfinite(slice_t)]
            med = np.median(valid_vals) if len(valid_vals) > 0 else 0.3
            slice_t[~np.isfinite(slice_t)] = med
            target_iv[t_idx, :] = slice_t
        st.session_state["target_iv"] = target_iv
        st.session_state["active_model"] = "Default Heston"
        st.session_state["true_params"] = np.array([2.0, 0.05, 0.3, -0.6, 0.05])
    
    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        S0 = st.number_input("S₀ — Initial Stock Price", value=100.0, step=5.0)
        epochs = st.slider("Training Epochs", min_value=5, max_value=100, value=30, step=5)
    with col2:
        r = st.number_input("r — Risk-free Rate", value=0.05, step=0.01)
        N_paths = st.slider("Paths for Monte Carlo", min_value=128, max_value=5000, value=1024, step=128)
    with col3:
        q = st.number_input("q — Dividend Yield", value=0.015, step=0.005)
        
    api_url = st.text_input("FastAPI Server URL", value="http://localhost:8000")
    
    if st.button("Run Neural SDE Calibration", use_container_width=True):
        payload = {
            "market_iv": target_iv.tolist(),
            "S0": S0,
            "r": r,
            "q": q,
            "epochs": epochs,
            "N_paths": N_paths
        }
        
        with st.spinner("Calibrating Neural SDE via Adjoint Method on FastAPI server..."):
            try:
                response = requests.post(f"{api_url}/calibrate_neural_sde", json=payload, timeout=600)
                if response.status_code == 200:
                    res_data = response.json()
                    st.success(f"Calibration completed in **{res_data['elapsed_ms']:.1f} ms**!")
                    st.session_state["sde_results"] = res_data
                else:
                    st.error(f"Calibration failed: {response.text}")
            except Exception as e:
                st.error(f"Error calling API at {api_url}: {e}")
                
    if "sde_results" in st.session_state:
        res = st.session_state["sde_results"]
        
        # Display parameters
        st.subheader("Calibrated Parameters")
        p_df = pd.DataFrame({
            "Parameter": ["Initial Variance (v0)", "Correlation (rho)", "Final Option Price RMSE"],
            "Value": [f"{res['v0']:.6f}", f"{res['rho']:.6f}", f"${res['final_rmse']:.6f}"]
        })
        st.dataframe(p_df, use_container_width=True)
        
        # Convergence and surface comparison
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Loss Convergence History")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=np.arange(1, len(res["loss_history"]) + 1), y=res["loss_history"],
                                          mode="lines+markers", line=dict(color="#00d4ff", width=2)))
            fig_loss.update_layout(
                xaxis_title="Epoch", yaxis_title="Loss", yaxis_type="log",
                height=350, margin=dict(l=0, r=0, b=40, t=40)
            )
            st.plotly_chart(fig_loss, use_container_width=True)
            
        with c2:
            st.markdown("### Calibrated SDE Volatility Smile")
            # We can select a maturity slice to plot
            mat_idx = st.selectbox("Select Maturity to Compare", range(8), format_func=lambda idx: f"T = {MATURITIES[idx]:.2f}")
            
            # Target IV for this maturity
            target_slice = target_iv[mat_idx, :]
            
            fig_smile = go.Figure()
            fig_smile.add_trace(go.Scatter(x=STRIKES, y=target_slice, mode="lines+markers", name="Target (Market)", line=dict(color="#00d4ff")))
            
            if "fitted_iv" in res and res["fitted_iv"] is not None:
                fitted_slice = np.array(res["fitted_iv"])[mat_idx, :]
                fig_smile.add_trace(go.Scatter(x=STRIKES, y=fitted_slice, mode="lines+markers", name="Fitted Neural SDE", line=dict(color="#ff3366", dash="dash")))
                
            fig_smile.update_layout(
                xaxis_title="Log-Moneyness", yaxis_title="Implied Volatility",
                height=350, margin=dict(l=0, r=0, b=40, t=40)
            )
            st.plotly_chart(fig_smile, use_container_width=True)

        if "fitted_iv" in res and res["fitted_iv"] is not None:
            # 3D surface comparison
            st.subheader("3D Surface: Target vs Fitted Neural SDE")
            K_grid, T_grid = np.meshgrid(STRIKES, MATURITIES)
            fig_3d = go.Figure()
            fig_3d.add_trace(go.Surface(x=K_grid, y=T_grid, z=target_iv,
                                     colorscale="Blues", opacity=0.7, name="Target", showscale=False))
            fig_3d.add_trace(go.Surface(x=K_grid, y=T_grid, z=np.array(res["fitted_iv"]),
                                     colorscale="Reds", opacity=0.7, name="Neural SDE", showscale=False))
            fig_3d.update_layout(
                scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity", zaxis_title="IV"),
                margin=dict(l=0, r=0, b=0, t=40), height=550
            )
            st.plotly_chart(fig_3d, use_container_width=True)


def render_signature_vol_panel():
    st.header("🖋️ Signature Volatility Smile Forecasting")
    st.markdown("""
    This panel simulates pathwise stock and variance dynamics under the **Signature Volatility** model,
    forecasting option smiles from path signatures of time-extended Brownian motion (up to depth 4).
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        v0 = st.slider("v₀ — Initial Variance", 0.01, 0.15, 0.04, step=0.01)
        rho = st.slider("ρ — Correlation", -1.0, 0.0, -0.5, step=0.05)
    with col2:
        T = st.slider("T — Maturity (Years)", 0.05, 2.0, 0.25, step=0.05)
        S0 = st.number_input("S₀ — Stock Price", value=100.0, step=5.0)
    with col3:
        r = st.number_input("r — Interest Rate", value=0.0, step=0.01)
        N_paths = st.slider("Simulation Path Count", min_value=500, max_value=20000, value=4096, step=500)
        
    st.subheader("Signature Volatility Coefficients (ell)")
    st.markdown("We parameterize a subset of the 30 coefficients for intuitive control:")
    
    cc1, cc2 = st.columns(2)
    with cc1:
        ell_0 = st.slider("Level 1: Time coefficient (l₁)", -0.05, 0.05, 0.01, step=0.001)
        ell_1 = st.slider("Level 1: Brownian motion coefficient (l₂)", -0.1, 0.1, -0.02, step=0.005)
    with cc2:
        ell_7 = st.slider("Level 3: Cross coefficient (l₈)", -0.05, 0.05, 0.0, step=0.001)
        ell_8 = st.slider("Level 3: Volatility clustering coeff (l₉)", -0.05, 0.05, 0.0, step=0.001)
        
    # Construct the full 30-element ell vector
    ell = [0.0] * 30
    ell[0] = ell_0
    ell[1] = ell_1
    ell[7] = ell_7
    ell[8] = ell_8
    
    api_url = st.text_input("FastAPI Server URL", value="http://localhost:8000", key="sig_url")
    
    if st.button("Generate Volatility Smile & Paths", use_container_width=True):
        # Grid of strikes around S0
        strikes = [float(x) for x in np.linspace(0.8 * S0, 1.2 * S0, 11)]
        
        payload = {
            "v0": v0,
            "ell": ell,
            "rho": rho,
            "T": T,
            "S0": S0,
            "r": r,
            "q": 0.0,
            "N_paths": N_paths,
            "strikes": strikes
        }
        
        with st.spinner("Simulating signature paths and calculating IV smile..."):
            try:
                response = requests.post(f"{api_url}/predict/signature_vol", json=payload, timeout=300)
                if response.status_code == 200:
                    st.session_state["sig_results"] = response.json()
                    st.success("Simulation and option pricing completed successfully.")
                else:
                    st.error(f"API call failed: {response.text}")
            except Exception as e:
                st.error(f"Error calling API at {api_url}: {e}")
                
    if "sig_results" in st.session_state:
        res = st.session_state["sig_results"]
        
        # Display 2D Smile Chart
        st.subheader("Forecasted Option Smile")
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(x=res["strikes"], y=res["implied_vols"], mode="lines+markers",
                                      line=dict(color="#00d4ff", width=2), name="Forecasted Smile"))
        fig_smile.update_layout(
            xaxis_title="Strike Price", yaxis_title="Implied Volatility",
            height=350, margin=dict(l=0, r=0, b=40, t=40)
        )
        st.plotly_chart(fig_smile, use_container_width=True)
        
        # Plot 3D Paths if returned
        if "paths_S" in res and res["paths_S"] is not None:
            st.subheader("Sample Simulated 3D Paths")
            st.markdown("Plots joint trajectories of **Stock Price** (X), **Volatility** (Z) and **Time** (Y).")
            
            fig_3d = go.Figure()
            steps = len(res["paths_S"][0])
            t_grid = np.linspace(0.0, T, steps)
            
            for path_idx in range(len(res["paths_S"])):
                S_path = res["paths_S"][path_idx]
                vol_path = np.sqrt(res["paths_vol"][path_idx])
                
                fig_3d.add_trace(go.Scatter3d(
                    x=S_path,
                    y=t_grid,
                    z=vol_path,
                    mode="lines",
                    line=dict(width=3),
                    name=f"Path {path_idx + 1}"
                ))
                
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="Stock Price",
                    yaxis_title="Time (t)",
                    zaxis_title="Volatility (sqrt(V))"
                ),
                margin=dict(l=0, r=0, b=0, t=40), height=550
            )
            st.plotly_chart(fig_3d, use_container_width=True)


def render_deep_hedging_panel():
    st.header("🛡️ Deep Hedging Policy Simulation")
    st.markdown("""
    This panel evaluates **recurrent LSTM-based optimal deep hedging policies** under proportional transaction costs.
    It simulates asset paths and applies the pre-trained neural policy to compute dynamic rebalancing delta decisions.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        option_type = st.selectbox("Option Style", ["european", "barrier", "minimax"])
        S0 = st.number_input("S₀ — Initial Spot Price", value=100.0, step=5.0)
        strike = st.number_input("Strike Price (K)", value=100.0, step=5.0)
    with col2:
        expiry = st.slider("Maturity (T)", 0.05, 0.5, 0.1, step=0.01)
        sigma = st.slider("Asset Volatility (σ)", 0.05, 0.6, 0.2, step=0.01)
        mu = st.number_input("Asset Drift (μ)", value=0.0, step=0.05)
    with col3:
        steps = st.slider("Rebalancing Steps", min_value=5, max_value=100, value=30, step=5)
        N_paths = st.slider("Path Count", min_value=5, max_value=500, value=100, step=5)
        barrier = st.number_input("Barrier level (B)", value=85.0, step=1.0) if option_type == "barrier" else 85.0
        
    st.subheader("Proportional Transaction Costs")
    cost_stock = st.slider("Stock Transaction Cost Coefficient (c_stock)", 0.0, 0.005, 0.0001, step=0.0001, format="%.5f")
    cost_vol = st.slider("Vol Instrument Transaction Cost Coefficient (c_vol)", 0.0, 0.01, 0.0005, step=0.0001, format="%.5f")
    
    api_url = st.text_input("FastAPI Server URL", value="http://localhost:8000", key="hedge_url")
    
    if st.button("Run Deep Hedging Policy Simulation", use_container_width=True):
        payload = {
            "option_type": option_type,
            "S0": S0,
            "strike": strike,
            "barrier": barrier,
            "expiry": expiry,
            "mu": mu,
            "sigma": sigma,
            "steps": steps,
            "N_paths": N_paths,
            "cost_stock": cost_stock,
            "cost_vol": cost_vol
        }
        
        with st.spinner("Simulating paths and optimal delta rebalancing..."):
            try:
                response = requests.post(f"{api_url}/hedge/simulate", json=payload, timeout=300)
                if response.status_code == 200:
                    st.session_state["hedge_results"] = response.json()
                    st.success("Hedging simulation completed.")
                else:
                    st.error(f"API call failed: {response.text}")
            except Exception as e:
                st.error(f"Error calling API at {api_url}: {e}")
                
    if "hedge_results" in st.session_state:
        res = st.session_state["hedge_results"]
        
        # Display summary metrics
        st.subheader("Performance Metrics")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("P&L Standard Deviation (Hedged)", f"{res['std_pnl']:.4f}")
        with m2:
            avg_cost = np.mean(res["costs"])
            st.metric("Average Transaction Cost", f"${avg_cost:.4f}")
        with m3:
            st.metric("Total Entropic/Quadratic Loss", f"{res['final_loss']:.4f}")
            
        # P&L Distribution overlay vs unhedged baseline
        st.subheader("Hedged P&L Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=res["pnl"], name="Hedged P&L", marker_color="#ff3366", opacity=0.75))
        unhedged_pnl = [-p for p in res["payoff"]]
        fig_hist.add_trace(go.Histogram(x=unhedged_pnl, name="Unhedged Baseline", marker_color="#00d4ff", opacity=0.6))
        
        fig_hist.update_layout(
            barmode="overlay",
            xaxis_title="Final P&L",
            yaxis_title="Count",
            height=350,
            margin=dict(l=0, r=0, b=40, t=40)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Hedging corridors
        st.subheader("Optimal Delta Hedging Corridors")
        st.markdown("Scatter plot of LSTM-generated stock hedge ratio (Delta) vs asset spot price across all steps.")
        
        fig_corr = go.Figure()
        spots_flat = []
        deltas_flat = []
        for path_idx in range(len(res["paths_S"])):
            spots_flat.extend(res["paths_S"][path_idx][:-1])
            deltas_flat.extend(res["deltas_stock"][path_idx])
            
        fig_corr.add_trace(go.Scatter(x=spots_flat, y=deltas_flat, mode="markers",
                                      marker=dict(size=4, color="#00ffcc", opacity=0.5),
                                      name="Stock Delta"))
        
        fig_corr.update_layout(
            xaxis_title="Underlying Asset Spot Price",
            yaxis_title="Hedging Ratio (Delta)",
            height=400,
            margin=dict(l=0, r=0, b=40, t=40)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
STRIKES    = np.linspace(-0.5, 0.5, 11)
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])

@st.cache_resource
def load_model(model_name: str):
    """Load the appropriate FNO surrogate and load corresponding normalizers."""
    if model_name in ("Neural SDE", "Signature Volatility", "Deep Hedging"):
        return None
        
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
    ["Rough Heston", "Classic Heston", "SABR", "SSVI", "Local Volatility", "Rough Bergomi", "Neural SDE", "Signature Volatility", "Deep Hedging"],
    index=0,
    key="model_selector"
)

# Load the corresponding FNO model
model = load_model(model_name)
device = torch.device("cpu")

if model is not None:
    st.caption(f"FiLM-FNO Surrogate for {model_name} • Optimized Gauss-Newton autograd Jacobians")
else:
    st.caption(f"Interactive Panel for {model_name}")

if model_name in ("Neural SDE", "Signature Volatility", "Deep Hedging"):
    if "active_model" in st.session_state and st.session_state["active_model"] != model_name:
        st.session_state.pop("target_iv", None)
        st.session_state.pop("true_params", None)
        st.session_state.pop("calib_results", None)
        st.session_state.pop("sde_results", None)
        st.session_state.pop("sig_results", None)
        st.session_state.pop("hedge_results", None)
    st.session_state["active_model"] = model_name

    if model_name == "Neural SDE":
        render_neural_sde_panel()
    elif model_name == "Signature Volatility":
        render_signature_vol_panel()
    elif model_name == "Deep Hedging":
        render_deep_hedging_panel()
    st.stop()

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

if model_name not in ("Neural SDE", "Signature Volatility", "Deep Hedging"):
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
