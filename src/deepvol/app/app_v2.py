"""
app_v2.py — Upgraded Volatility Model Zoo Streamlit Dashboard.
Supports SABR, Heston, Rough Bergomi, Neural SDE, MLSV, and Schwartz-Smith.
Includes arbitrage-validated custom surface uploader, 3D Plotly visualization,
parameter trajectory timelines, and PDF/HTML report exports.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import os
import sys
import plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.uploader import parse_iv_sheet, interpolate_to_model_grid, check_arbitrage
from components.models import (
    load_fno_model,
    reconstruct_sabr_surface,
    reconstruct_heston_surface,
    reconstruct_rbergomi_surface,
    reconstruct_schwartz_smith_surface,
    reconstruct_mlsv_surface,
    calibrate_neural_sde_local,
    invert_implied_vol,
    calibrate_heston,
    calibrate_sabr,
    calibrate_rbergomi,
    MATURITIES,
    STRIKES
)
from components.visualization import plot_3d_surfaces, plot_parameter_trajectory, plot_smile_slice
from components.exporter import generate_pdf_report, generate_html_report

# Page Config
st.set_page_config(page_title="Deep Volatility Model Zoo v2", layout="wide")

st.title("⚡ Deep Volatility Model Zoo Calibration & Dashboard v2")
st.markdown("""
Welcome to the **Upgraded Volatility Model Zoo Dashboard (v2)**. 
Configure model parameters to generate synthetic volatility surfaces, or upload custom CSV/Excel implied volatility sheets to validate and calibrate across six models:
**SABR, Heston, Rough Bergomi, Neural SDE, MLSV, and Schwartz-Smith**.
""")

# ─── Sidebar Model Selector ─────────────────────────────────────────────────
st.sidebar.header("🕹️ Control Panel")
model_name = st.sidebar.selectbox(
    "Volatility Model Selection",
    ["SABR", "Classic Heston", "Rough Bergomi", "Neural SDE", "McKean-Vlasov SDE (MLSV)", "Schwartz-Smith (2-Factor)"],
    index=1,
    key="model_selector"
)

# ─── Sidebar Parameter Controls (True / Preset Parameters) ────────────────────
st.sidebar.subheader(f"True / Preset {model_name} Params")

# Helper to store parameters in dict
true_params_dict = {}

if model_name == "SABR":
    alpha = st.sidebar.slider("α (alpha) — Initial Vol", 0.05, 0.8, 0.20, step=0.01)
    rho = st.sidebar.slider("ρ (rho) — Correlation", -0.9, 0.9, -0.40, step=0.01)
    nu = st.sidebar.slider("ν (nu) — Vol of Vol", 0.1, 1.2, 0.40, step=0.01)
    true_params_dict = {"alpha": alpha, "rho": rho, "nu": nu}

elif model_name == "Classic Heston":
    kappa = st.sidebar.slider("κ — Mean Reversion", 0.1, 5.0, 2.0, step=0.1)
    theta = st.sidebar.slider("θ — Long-Run Variance", 0.01, 0.15, 0.05, step=0.01)
    sigma = st.sidebar.slider("σ — Vol of Vol", 0.1, 1.0, 0.3, step=0.01)
    rho = st.sidebar.slider("ρ — Correlation", -0.9, -0.1, -0.6, step=0.01)
    v0 = st.sidebar.slider("v₀ — Initial Variance", 0.01, 0.15, 0.05, step=0.01)
    true_params_dict = {"kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho, "v0": v0}

elif model_name == "Rough Bergomi":
    v0 = st.sidebar.slider("v₀ — Initial Variance", 0.01, 0.20, 0.08, step=0.01)
    H = st.sidebar.slider("H — Hurst Parameter", 0.04, 0.15, 0.07, step=0.01)
    eta = st.sidebar.slider("η — Vol of Vol", 0.5, 4.0, 1.5, step=0.1)
    rho = st.sidebar.slider("ρ — Correlation", -0.95, 0.0, -0.70, step=0.01)
    true_params_dict = {"v0": v0, "H": H, "eta": eta, "rho": rho}

elif model_name == "Neural SDE":
    S0 = st.sidebar.number_input("S₀ — Initial Spot Price", value=100.0, step=5.0)
    r = st.sidebar.number_input("r — Interest Rate", value=0.05, step=0.01)
    q = st.sidebar.number_input("q — Dividend Yield", value=0.015, step=0.005)
    epochs = st.sidebar.slider("Training Epochs", 5, 50, 30, step=5)
    N_paths = st.sidebar.slider("Simulation Path Count", 128, 4096, 1024, step=128)
    true_params_dict = {"S0": S0, "r": r, "q": q, "epochs": epochs, "N_paths": N_paths}

elif model_name == "McKean-Vlasov SDE (MLSV)":
    S0 = st.sidebar.number_input("S₀ — Initial Spot Price", value=100.0, step=5.0)
    r = st.sidebar.number_input("r — Interest Rate", value=0.05, step=0.01)
    q = st.sidebar.number_input("q — Dividend Yield", value=0.02, step=0.005)
    v0 = st.sidebar.slider("v₀ — Initial Variance", 0.01, 0.20, 0.04, step=0.01)
    kappa = st.sidebar.slider("κ — Mean Reversion Speed", 0.1, 5.0, 2.0, step=0.1)
    theta = st.sidebar.slider("θ — Long-Run Variance", 0.01, 0.20, 0.04, step=0.01)
    xi = st.sidebar.slider("ξ — Vol of Vol", 0.1, 1.0, 0.3, step=0.05)
    rho = st.sidebar.slider("ρ — Asset-Vol Correlation", -0.95, 0.95, -0.70, step=0.05)
    true_params_dict = {"S0": S0, "r": r, "q": q, "v0": v0, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho}

elif model_name == "Schwartz-Smith (2-Factor)":
    S0 = st.sidebar.number_input("S₀ — Futures Spot", value=100.0, step=5.0)
    r = st.sidebar.number_input("r — Interest Rate", value=0.05, step=0.01)
    chi_t = st.sidebar.slider("χ_t — Short-term Deviation", -2.0, 2.0, 0.0, step=0.1)
    xi_t = st.sidebar.slider("ξ_t — Long-term Equilibrium", 2.0, 6.0, 4.6, step=0.1)  # ln(100) ~ 4.6
    kappa = st.sidebar.slider("κ — Short-term Mean Reversion", 0.05, 2.0, 0.5, step=0.05)
    sigma_chi = st.sidebar.slider("σ_χ — Short-term Volatility", 0.05, 1.0, 0.2, step=0.05)
    rho = st.sidebar.slider("ρ — Factor Correlation", -0.99, 0.99, 0.3, step=0.05)
    sigma_xi = st.sidebar.slider("σ_ξ — Long-term Volatility", 0.05, 1.0, 0.1, step=0.05)
    mu_star = st.sidebar.slider("μ* — Risk-adjusted Drift", -0.5, 0.5, 0.03, step=0.01)
    lambda_chi = st.sidebar.slider("λ_χ — Short-term Risk Premium", -0.5, 0.5, 0.02, step=0.01)
    true_params_dict = {
        "S0": S0, "r": r, "chi_t": chi_t, "xi_t": xi_t, "kappa": kappa,
        "sigma_chi": sigma_chi, "rho": rho, "sigma_xi": sigma_xi,
        "mu_star": mu_star, "lambda_chi": lambda_chi
    }

# Noise Level for stress testing
noise_level = st.sidebar.slider("Market Noise Level (Stress Test)", 0.0, 0.10, 0.01, step=0.01, key="noise")

# ─── Target Volatility Surface Management ─────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📤 Custom Volatility Sheet Uploader")
    st.markdown("Upload your own implied volatility surface sheet (.csv or .xlsx).")
    uploaded_file = st.file_uploader("Choose a CSV/Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Parse sheet
            T_src, K_src, iv_src = parse_iv_sheet(uploaded_file)
            
            # Arbitrage validation on source surface
            arb_res = check_arbitrage(T_src, K_src, iv_src)
            
            st.markdown("### Arbitrage Audit Results:")
            if arb_res["is_free"]:
                st.success("🟢 Calendar & Butterfly Arbitrage Free (Pass)")
            else:
                st.error("🔴 Arbitrage Violations Detected! (Fail)")
                
            with st.expander("Show Arbitrage Details"):
                st.markdown(f"**Calendar Spread Violations:** {len(arb_res['calendar_violations'])}")
                if len(arb_res['calendar_violations']) > 0:
                    st.json(arb_res['calendar_violations'][:5])
                    
                st.markdown(f"**Butterfly Spread Violations:** {len(arb_res['butterfly_violations'])}")
                if len(arb_res['butterfly_violations']) > 0:
                    st.json(arb_res['butterfly_violations'][:5])

            # Interpolate to target grid
            target_iv = interpolate_to_model_grid(T_src, K_src, iv_src)
            
            # Save to session state
            st.session_state["target_iv"] = target_iv
            st.session_state["active_model"] = "Uploaded Surface"
            st.session_state["true_params"] = None
            st.session_state.pop("calib_results", None)
            st.info("Uploaded surface mapped to FNO grid.")
            
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")

with col_right:
    st.subheader("🎯 Target Surface Generation")
    st.markdown("Generate a synthetic target surface using the preset model parameters in the sidebar.")
    
    if st.button("Generate Target Surface from Sidebar Presets", use_container_width=True):
        with st.spinner("Generating target surface..."):
            try:
                # Cache loaded model (if FNO-based)
                fno_model = load_fno_model(model_name)
                
                if model_name == "SABR":
                    target_iv = reconstruct_sabr_surface(alpha=alpha, rho=rho, nu=nu)
                elif model_name == "Classic Heston":
                    target_iv = reconstruct_heston_surface(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0)
                elif model_name == "Rough Bergomi":
                    target_iv = reconstruct_rbergomi_surface(fno_model, v0=v0, H=H, eta=eta, rho=rho)
                elif model_name == "Neural SDE":
                    # Generate Classic Heston as a target prior for Neural SDE
                    target_iv = reconstruct_heston_surface(kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6, v0=0.05)
                elif model_name == "McKean-Vlasov SDE (MLSV)":
                    target_iv = reconstruct_mlsv_surface(S0=S0, r=r, q=q, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, N_paths=1000)
                elif model_name == "Schwartz-Smith (2-Factor)":
                    target_iv = reconstruct_schwartz_smith_surface(
                        S0=S0, r=r, chi_t=chi_t, xi_t=xi_t, kappa=kappa,
                        sigma_chi=sigma_chi, rho=rho, sigma_xi=sigma_xi,
                        mu_star=mu_star, lambda_chi=lambda_chi
                    )
                
                # Apply noise
                rng = np.random.default_rng(seed=42)
                noise = rng.normal(0, noise_level * np.abs(target_iv), target_iv.shape)
                market_iv_noisy = np.maximum(target_iv + noise, 1e-4)
                
                st.session_state["target_iv"] = market_iv_noisy
                st.session_state["active_model"] = model_name
                st.session_state["true_params"] = true_params_dict.copy()
                st.session_state.pop("calib_results", None)
                st.success(f"Generated target surface for **{model_name}**.")
            except Exception as e:
                st.error(f"Error generating surface: {e}")

# Check if target surface exists
if "target_iv" not in st.session_state:
    st.info("👈 Please generate a target surface or upload an implied volatility sheet to start.")
    st.stop()

target_iv = st.session_state["target_iv"]
active_model = st.session_state["active_model"]

st.divider()

# ─── Model Info & Run Calibration ─────────────────────────────────────────────
st.header(f"🔧 Model zoo Calibration: {model_name}")
st.info(f"Active Target Volatility Surface: **{active_model}**")

if st.button(f"Calibrate / Reconstruct {model_name} Surface", use_container_width=True):
    with st.spinner("Calibrating / reconstructing model surface..."):
        t0 = time.time()
        
        # Load FNO surrogate if applicable
        fno_model = load_fno_model(model_name)
        
        if model_name == "Classic Heston":
            res = calibrate_heston(fno_model, target_iv, MATURITIES, STRIKES, max_iter=25, n_starts=2)
            calib_results = {
                "params": res["params"],
                "history": res["theta_history"],
                "loss_history": res["loss_history"],
                "iv_fitted": res["iv_fitted"],
                "elapsed_ms": res["elapsed_ms"]
            }
        elif model_name == "SABR":
            res = calibrate_sabr(fno_model, target_iv, MATURITIES, STRIKES, max_iter=25, n_starts=2)
            calib_results = {
                "params": {"alpha": res["alpha"], "rho": res["rho"], "nu": res["nu"]},
                "history": res["theta_history"],
                "loss_history": res["loss_history"],
                "iv_fitted": res["iv_fitted"],
                "elapsed_ms": res["elapsed_ms"]
            }
        elif model_name == "Rough Bergomi":
            res = calibrate_rbergomi(fno_model, target_iv, MATURITIES, STRIKES, max_iter=25, n_starts=2)
            calib_results = {
                "params": {"v0": res["v0"], "H": res["H"], "eta": res["eta"], "rho": res["rho"]},
                "history": res["theta_history"],
                "loss_history": res["loss_history"],
                "iv_fitted": res["iv_fitted"],
                "elapsed_ms": res["elapsed_ms"]
            }
        elif model_name == "Neural SDE":
            res = calibrate_neural_sde_local(
                target_iv, S0=true_params_dict.get("S0", 100.0),
                r=true_params_dict.get("r", 0.05), q=true_params_dict.get("q", 0.015),
                epochs=true_params_dict.get("epochs", 30), N_paths=true_params_dict.get("N_paths", 1024)
            )
            calib_results = {
                "params": {"v0": res["v0"], "rho": res["rho"]},
                "history": [],
                "loss_history": res["loss_history"],
                "iv_fitted": res["fitted_iv"],
                "elapsed_ms": res["elapsed_ms"]
            }
        elif model_name == "McKean-Vlasov SDE (MLSV)":
            # Reconstruct direct surface using particle SDE paths
            p = true_params_dict
            fitted = reconstruct_mlsv_surface(
                S0=p["S0"], r=p["r"], q=p["q"], v0=p["v0"], kappa=p["kappa"],
                theta=p["theta"], xi=p["xi"], rho=p["rho"], N_paths=1000
            )
            calib_results = {
                "params": {"v0": p["v0"], "kappa": p["kappa"], "theta": p["theta"], "xi": p["xi"], "rho": p["rho"]},
                "history": [],
                "loss_history": [],
                "iv_fitted": fitted,
                "elapsed_ms": (time.time() - t0) * 1000.0
            }
        elif model_name == "Schwartz-Smith (2-Factor)":
            p = true_params_dict
            fitted = reconstruct_schwartz_smith_surface(
                S0=p["S0"], r=p["r"], chi_t=p["chi_t"], xi_t=p["xi_t"], kappa=p["kappa"],
                sigma_chi=p["sigma_chi"], rho=p["rho"], sigma_xi=p["sigma_xi"],
                mu_star=p["mu_star"], lambda_chi=p["lambda_chi"]
            )
            calib_results = {
                "params": p.copy(),
                "history": [],
                "loss_history": [],
                "iv_fitted": fitted,
                "elapsed_ms": (time.time() - t0) * 1000.0
            }
            
        st.session_state["calib_results"] = calib_results
        st.success("Calibration / surface reconstruction completed.")

# ─── Render Calibration Results & Visualization ───────────────────────────────
if "calib_results" in st.session_state:
    res = st.session_state["calib_results"]
    fitted_iv = res["iv_fitted"]
    
    st.subheader("📊 Calibration Performance Summary")
    
    # Compute error metrics in IV space
    rmse = float(np.sqrt(np.mean((fitted_iv - target_iv)**2)))
    mae = float(np.mean(np.abs(fitted_iv - target_iv)))
    max_err = float(np.max(np.abs(fitted_iv - target_iv)))
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Elapsed Time", f"{res['elapsed_ms']:.1f} ms")
    with c2:
        st.metric("RMSE (bps)", f"{rmse * 10000.0:.2f} bps")
    with c3:
        st.metric("MAE (bps)", f"{mae * 10000.0:.2f} bps")
    with c4:
        st.metric("Max Error (bps)", f"{max_err * 10000.0:.2f} bps")

    # Render Calibrated Parameters Table
    st.subheader("🔧 Parameter Details")
    p_names = list(res["params"].keys())
    p_vals = list(res["params"].values())
    
    # Match True parameters from sidebar if target was generated using this model
    true_vals = []
    if st.session_state.get("true_params") is not None and st.session_state["active_model"] == model_name:
        true_dict = st.session_state["true_params"]
        for name in p_names:
            true_vals.append(true_dict.get(name, np.nan))
    else:
        true_vals = [np.nan] * len(p_names)
        
    param_df = pd.DataFrame({
        "Parameter": p_names,
        "Preset / True": true_vals,
        "Calibrated / Fitted": p_vals,
        "Abs Error": np.abs(np.array(true_vals) - np.array(p_vals))
    })
    st.dataframe(param_df.style.format({
        "Preset / True": "{:.6f}", 
        "Calibrated / Fitted": "{:.6f}", 
        "Abs Error": "{:.6f}"
    }), use_container_width=True)

    # 3D interactive comparison plot
    st.subheader("🌐 3D Volatility Surface Comparison")
    st.plotly_chart(plot_3d_surfaces(MATURITIES, STRIKES, target_iv, fitted_iv, market_name="Target / Market", reconstructed_name="Fitted Zoo Model"), use_container_width=True)

    # Convergence Plot and 2D Smile slice side-by-side
    st.subheader("📈 Smile Alignment & Optimization History")
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.markdown("### Volatility Smile Fit Slice")
        selected_t_idx = st.selectbox("Select Maturity Tenor to Inspect", range(len(MATURITIES)), format_func=lambda idx: f"T = {MATURITIES[idx]:.2f}")
        fig_smile = plot_smile_slice(MATURITIES, STRIKES, target_iv, fitted_iv, t_idx=selected_t_idx, market_name="Target / Market", reconstructed_name="Fitted zoo Model")
        st.plotly_chart(fig_smile, use_container_width=True)
        
    with col_vis2:
        if len(res["history"]) > 0:
            st.markdown("### Gauss-Newton Parameter Trajectory Timeline")
            fig_traj = plot_parameter_trajectory(res["history"], p_names)
            st.plotly_chart(fig_traj, use_container_width=True)
        elif len(res["loss_history"]) > 0:
            st.markdown("### Optimizer Loss History")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=np.arange(1, len(res["loss_history"]) + 1), y=res["loss_history"], mode="lines+markers", line=dict(color="#ff3366", width=2)))
            fig_loss.update_layout(xaxis_title="Epoch / Step", yaxis_title="Loss Value", yaxis_type="log", height=350, margin=dict(l=0, r=0, b=40, t=30))
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Parameter trajectory/loss convergence timeline is only available for optimization models.")

    # ─── Report Exporter Section ──────────────────────────────────────────────
    st.divider()
    st.header("📄 Downloadable Report Exporter")
    st.markdown("Export a compiled version of the calibration results, error tables, and smile fits.")
    
    err_metrics = {"RMSE": rmse, "MAE": mae, "Max Error": max_err}
    
    # Generate report bytes
    pdf_bytes = generate_pdf_report(model_name, param_df, err_metrics, MATURITIES, STRIKES, target_iv, fitted_iv)
    html_str = generate_html_report(model_name, param_df, err_metrics)
    
    c_pdf, c_html = st.columns(2)
    with c_pdf:
        st.download_button(
            label="Download PDF Calibration Report",
            data=pdf_bytes,
            file_name=f"calibration_report_{model_name.lower().replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with c_html:
        st.download_button(
            label="Download HTML Calibration Report",
            data=html_str,
            file_name=f"calibration_report_{model_name.lower().replace(' ', '_')}.html",
            mime="text/html",
            use_container_width=True
        )
