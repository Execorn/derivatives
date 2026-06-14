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
                       calibrate_reparameterized, compute_confidence_reparameterized,
                       compute_fim_ellipsoid,
                       _make_spatial_input, _fno_predict_real_iv, _load_normalizers)
from calibrate_fast import calibrate_newton
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
    """Load FNO v2 (R²=0.9991, N=40, N_cos=128) with v2 normalizers."""
    model = MirrorPaddedFNO2d()
    weights_path = "artifacts/weights/fno_v2_final_prod.pth"
    if not os.path.exists(weights_path):
        # Fallback to v1 for backward compatibility
        weights_path = "artifacts/models/fno_best.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu",
                                         weights_only=True))
    model.eval()
    _load_normalizers(version="v2" if "v2" in weights_path else "v1")
    return model, weights_path

model, _model_path = load_model()
_using_v2 = "v2" in _model_path

st.set_page_config(page_title="Deep Rough Heston NN Calibration", layout="wide")
st.title("⚡ Deep Rough Heston Calibration")
if _using_v2:
    st.caption("FiLM-FNO v2 Surrogate (R²=0.9991, N=40, Nᵇᵒˢ=128) • Option 2: L-BFGS • Option 3: Newton-Raphson w/ autograd Jacobians")
else:
    st.caption("FiLM-FNO Surrogate + L-BFGS Calibration")

# ─── Calibration mode ────────────────────────────────────────────────────────
st.sidebar.divider()
calib_mode = st.sidebar.radio(
    "Calibration Mode",
    ["Newton-Raphson — Option 3 (recommended)",
     "Reparameterized 3D — Option 2 (L-BFGS)",
     "Full 6D — Option 2 (experimental)"],
    index=0,
    key="calib_mode",
    help="Option 3: Gauss-Newton with autograd Jacobians (3× faster than FD). "
         "Option 2: L-BFGS via scipy. Both use 3D reparameterized space (v₀, ζ=σρ, λ=σ√(1-ρ²)).",
)
_newton_mode  = calib_mode.startswith("Newton")
_reparam_mode = calib_mode.startswith("Reparameterized") or _newton_mode

# ─── Sidebar sliders (grey out ghost params in 3D mode) ──────────────────────
st.sidebar.header("True Rough Heston Parameters")
if _reparam_mode:
    st.sidebar.info("🔒 Ghost parameters fixed: κ=1.0, θ=0.08, H=0.08")
    kappa = 1.0
    theta = 0.08
    H     = 0.08
    st.sidebar.markdown(f"**κ (fixed)** = {kappa}")
    st.sidebar.markdown(f"**θ (fixed)** = {theta}")
    st.sidebar.markdown(f"**H (fixed)** = {H}")
else:
    kappa = st.sidebar.slider("κ — Mean Reversion",   0.1,  5.0,  2.5,  step=0.1,  key="kappa")
    theta = st.sidebar.slider("θ — Long-Run Variance", 0.01, 0.15, 0.08, step=0.01, key="theta")
    H     = st.sidebar.slider("H — Hurst Parameter",  0.02, 0.15, 0.08, step=0.01, key="H")

sigma = st.sidebar.slider("σ — Vol of Vol",        0.1,  1.0,  0.5,  step=0.01, key="sigma")
rho   = st.sidebar.slider("ρ — Correlation",      -0.9, -0.1, -0.5,  step=0.01, key="rho")
v0    = st.sidebar.slider("v₀ — Initial Variance", 0.01, 0.15, 0.08, step=0.01, key="v0")

st.sidebar.divider()
noise_level = st.sidebar.slider(
    "Market Noise Level (Stress Test)", 0.0, 0.10, 0.01, step=0.01, key="noise")

true_params = np.array([kappa, theta, sigma, rho, v0, H])

# FIM info in sidebar for reparameterized mode
if _reparam_mode:
    zeta_true_disp = sigma * rho
    lam_true_disp  = sigma * float(np.sqrt(1.0 - rho**2))
    st.sidebar.markdown(
        f"**ζ = σρ** = {zeta_true_disp:.4f}  \n"
        f"**λ = σ√(1-ρ²)** = {lam_true_disp:.4f}  \n"
        f"---  \n"
        f"*FIM condition number*:  \n"
        f"6D space: **~10⁷–10⁸**  \n"
        f"3D (ζ,λ) space: **~10³–10⁴**"
    )

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

    with st.spinner("Running calibration..."):
        if _newton_mode:
            newton_res = calibrate_newton(
                model, market_iv_noisy, MATURITIES, STRIKES,
                max_iter=20, verbose=False)
            res3d = newton_res
            calibrated_params = np.array([
                1.0, 0.08, res3d["sigma"], res3d["rho"], res3d["v0"], 0.08
            ])
            history = newton_res["history"]
            elapsed = newton_res["elapsed"]
        elif _reparam_mode:
            res3d = calibrate_reparameterized(
                model, market_iv_noisy, MATURITIES, STRIKES)
            newton_res = None
            calibrated_params = np.array([
                1.0, 0.08, res3d["sigma"], res3d["rho"], res3d["v0"], 0.08
            ])
            history = res3d["history"]
            elapsed = res3d["elapsed"]
        else:
            newton_res = None
            res3d      = None
            calibrated_params, history, elapsed = calibrate_parameters(
                model, market_iv_noisy, init_params, MATURITIES, STRIKES)

    with st.spinner("Computing FIM confidence..."):
        if _reparam_mode and res3d:
            fim_res = compute_fim_ellipsoid(
                model, res3d["v0"], res3d["zeta"], res3d["lambda"],
                MATURITIES, STRIKES, sigma_obs=max(noise_level, 0.005))
            conf_scores = compute_confidence_reparameterized(
                model, res3d["v0"], res3d["zeta"], res3d["lambda"],
                MATURITIES, STRIKES)
        else:
            fim_res = None
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
        "res3d":             res3d,
        "newton_res":        newton_res,
        "fim_res":           fim_res,
    }

# ─── Display results ─────────────────────────────────────────────────────────
if "calib_results" in st.session_state:
    res               = st.session_state["calib_results"]
    calibrated_params = res["calibrated_params"]
    market_iv_noisy   = res["market_iv_noisy"]
    calibrated_iv     = res["calibrated_iv"]
    history           = res["history"]
    res3d             = res.get("res3d") or {}
    newton_res        = res.get("newton_res")
    fim_res           = res.get("fim_res")
    elapsed           = res["elapsed"]
    conf_scores       = res["conf_scores"]
    noise_used        = res["noise_level_used"]

    st.success(
        f"Calibration took **{elapsed:.4f}s** ({elapsed*1000:.1f} ms) — "
        f"Final Loss: {history[-1]:.6f} — "
        f"Noise level used: {noise_used*100:.1f}%"
    )

    # Parameter table
    if _reparam_mode:
        # Show 3D reparameterized results
        zeta_t = stored_true[2] * stored_true[3]           # sigma*rho
        lam_t  = stored_true[2] * np.sqrt(1 - stored_true[3]**2)
        zeta_c = res3d["zeta"] if "calib_results" in st.session_state and "res3d" in st.session_state["calib_results"] else float(calibrated_params[2] * calibrated_params[3])
        lam_c  = res3d["lambda"] if "calib_results" in st.session_state and "res3d" in st.session_state["calib_results"] else float(calibrated_params[2] * np.sqrt(1 - calibrated_params[3]**2))
        df3d = pd.DataFrame({
            "Parameter":  ["v₀", "ζ = σρ", "λ = σ√(1-ρ²)", "σ (derived)", "ρ (derived)"],
            "True":       [stored_true[4], zeta_t, lam_t, stored_true[2], stored_true[3]],
            "Calibrated": [calibrated_params[4], res3d["zeta"], res3d["lambda"],
                           res3d["sigma"], res3d["rho"]],
        })
        df3d["Abs Error"] = np.abs(df3d["True"] - df3d["Calibrated"])
        st.dataframe(
            df3d.style.format({"True": "{:.6f}", "Calibrated": "{:.6f}", "Abs Error": "{:.6f}"}),
            use_container_width=True)
        st.caption("Ghost parameters fixed: κ=1.0, θ=0.08, H=0.08  |  "
                   "FIM condition number: 6D ~10⁷–10⁸ → 3D ~10³–10⁴")
    else:
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
    if calib_mode.startswith("Reparameterized"):
        st.caption(
            "Jacobian column Frobenius norm ‖∂IV/∂θᵢ‖_F for (v₀, ζ, λ) — "
            "reparameterization isolates the 3 identifiable degrees of freedom."
        )
        conf_labels = {"v0": "v₀ (Initial Variance)",
                       "zeta": "ζ = σρ (Skew Driver)",
                       "lambda": "λ = σ√(1-ρ²) (Level Driver)"}
        for pname, score in conf_scores.items():
            label = conf_labels.get(pname, pname)
            color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
            st.progress(score, text=f"{color} {label}: {score:.2f}")
    else:
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

    # ── Newton Option 3: Convergence trace + FIM ellipsoid ─────────────────────
    if newton_res is not None:
        st.divider()
        st.subheader("⚡ Newton-Raphson — Option 3 Details")

        col_conv, col_fim = st.columns([1, 1])

        with col_conv:
            st.markdown("**Gauss-Newton convergence trace** (best restart)")
            hist_arr = np.array(newton_res["history"])
            iters    = np.arange(1, len(hist_arr) + 1)
            fig_gn   = go.Figure()
            fig_gn.add_trace(go.Scatter(
                x=iters, y=hist_arr,
                mode="lines+markers",
                line=dict(color="#00d4ff", width=2),
                marker=dict(size=7, color="#00d4ff"),
                name="MSE loss",
            ))
            fig_gn.update_layout(
                title=f"GN Loss — {len(hist_arr)} iters, {newton_res['elapsed']*1000:.0f}ms",
                xaxis_title="Iteration",
                yaxis_title="MSE",
                yaxis_type="log",
                height=300,
                margin=dict(l=0, r=0, b=40, t=40),
                plot_bgcolor="#111",
                paper_bgcolor="#111",
                font=dict(color="#eee"),
                xaxis=dict(gridcolor="#333"),
                yaxis=dict(gridcolor="#333"),
            )
            st.plotly_chart(fig_gn, use_container_width=True)

            # Parameter trajectory
            if newton_res.get("theta_history"):
                th = np.array(newton_res["theta_history"])   # (n_iter, 3)
                fig_tr = go.Figure()
                labels = ["v₀", "ζ", "λ"]
                colors = ["#ff7f50", "#7ecf7e", "#7eaaff"]
                for i, (lbl, col) in enumerate(zip(labels, colors)):
                    fig_tr.add_trace(go.Scatter(
                        x=iters[:len(th)], y=th[:, i],
                        mode="lines+markers",
                        name=lbl, line=dict(color=col, width=2),
                        marker=dict(size=6),
                    ))
                fig_tr.update_layout(
                    title="Parameter trajectory (best restart)",
                    xaxis_title="Iteration", yaxis_title="Value",
                    height=280, margin=dict(l=0, r=0, b=40, t=40),
                    plot_bgcolor="#111", paper_bgcolor="#111",
                    font=dict(color="#eee"),
                    xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
                    legend=dict(bgcolor="#222"),
                )
                st.plotly_chart(fig_tr, use_container_width=True)

        with col_fim:
            st.markdown("**Fisher Information — 95% Confidence Intervals**")
            if fim_res is not None:
                ci   = fim_res["ci_95"]
                stds = fim_res["std_errors"]
                corr = fim_res["corr_matrix"]

                ci_df = pd.DataFrame({
                    "Param": ["v₀", "ζ (zeta)", "λ (lambda)"],
                    "Estimate": [newton_res["v0"], newton_res["zeta"], newton_res["lambda"]],
                    "σ (1-σ)":  [f"±{stds[i]:.4f}" for i in range(3)],
                    "95% CI lo": [f"{ci['v0'][0]:.4f}", f"{ci['zeta'][0]:.4f}", f"{ci['lambda'][0]:.4f}"],
                    "95% CI hi": [f"{ci['v0'][1]:.4f}", f"{ci['zeta'][1]:.4f}", f"{ci['lambda'][1]:.4f}"],
                })
                st.dataframe(ci_df.set_index("Param"), use_container_width=True)

                st.markdown("**Parameter correlation matrix**  \n(ζ–λ anticorrelation = σ=√(ζ²+λ²) manifold degeneracy)")
                corr_df = pd.DataFrame(
                    corr.round(3),
                    index=["v₀", "ζ", "λ"],
                    columns=["v₀", "ζ", "λ"],
                )
                st.dataframe(corr_df.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1),
                             use_container_width=True)

                st.caption(
                    f"FIM computed at σ_obs={fim_res['sigma_obs']*100:.1f}% | "
                    f"95% ellipsoid radius χ²₃=7.815 | "
                    f"Jacobian shape: {fim_res['jacobian'].shape}"
                )
            else:
                st.info("FIM confidence requires Option 3 (Newton) mode.")

        # Method comparison metrics
        st.markdown("---")
        cols_m = st.columns(4)
        cols_m[0].metric("Calibration time", f"{newton_res['elapsed']*1000:.0f} ms")
        cols_m[1].metric("GN iterations", str(newton_res["n_iter"]))
        cols_m[2].metric("Final MSE", f"{newton_res['final_mse']:.2e}")
        cols_m[3].metric("L-BFGS speedup", "~3×",
                          delta="Newton (autograd Jacobian) vs L-BFGS (scipy)",
                          delta_color="off")

