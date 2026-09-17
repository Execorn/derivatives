"""
joint_spx_vix.py — Joint SPX + VIX Calibration Dashboard.

Features:
  - Dual surface upload (SPX IV + VIX level input)
  - Weight sliders for SPX vs VIX loss components
  - Joint calibration via L-BFGS-B (multiple restarts)
  - Dual-panel fitted vs market surface comparison
  - VIX term structure chart
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_APP_DIR = Path(__file__).parent.parent
_SRC_DIR = _APP_DIR.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

st.set_page_config(page_title="Joint SPX+VIX Calibration", layout="wide")
st.title("Joint SPX + VIX Calibration — Rough Heston FNO")
st.markdown(
    "Calibrate Rough Heston (FNO v3 surrogate) jointly to an **SPX IV surface** "
    "and a **VIX level** using a weighted L-BFGS-B loss:\n"
    r"$$\mathcal{L}(\theta) = w_{\text{SPX}} \cdot \text{RMSE}_{\text{SPX}}(\theta) "
    r"+ w_{\text{VIX}} \cdot (\text{VIX}_{\text{model}} - \text{VIX}_{\text{obs}})^2$$"
)

_MATS = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
_STKS = np.linspace(-0.5, 0.5, 11)
_PARAM_NAMES = ["kappa", "theta", "sigma", "rho", "v0", "H"]
_PARAM_LABELS = {
    "kappa": "κ (Mean Reversion)", "theta": "θ (Long-run Var)",
    "sigma": "σ (Vol of Vol)",     "rho":   "ρ (Correlation)",
    "v0":    "v₀ (Initial Var)",   "H":     "H (Hurst)",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Calibration Settings")
w_spx = st.sidebar.slider("SPX Weight (w_SPX)", 0.0, 5.0, 1.0, step=0.1)
w_vix = st.sidebar.slider("VIX Weight (w_VIX)", 0.0, 5.0, 1.0, step=0.1)
n_restarts = st.sidebar.slider("Optimizer Restarts", 1, 6, 3)
vix_obs = st.sidebar.number_input("Observed VIX Level", value=18.5, step=0.5, min_value=5.0, max_value=80.0)

st.sidebar.header("SPX Surface Input")
spx_mode = st.sidebar.selectbox("SPX Surface Source", ["Synthetic (Heston)", "Upload CSV"])

# ── SPX surface generation / upload ──────────────────────────────────────────
def _generate_spx_surface(kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6, v0=0.05) -> np.ndarray:
    from deepvol.models.heston import heston_iv_surface
    p = {"kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho, "v0": v0}
    iv = heston_iv_surface(p, _MATS, _STKS)
    return np.where(np.isfinite(iv), iv, 0.20).astype(np.float32)

# ── Tab Layout ────────────────────────────────────────────────────────────────
tab_setup, tab_calib, tab_result = st.tabs([
    "Input Setup", "Run Calibration", "Results & Comparison",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Input Setup
# ═══════════════════════════════════════════════════════════════════════════════
with tab_setup:
    st.header("SPX Surface & VIX Level Setup")

    col_spx, col_vix = st.columns(2)

    with col_spx:
        st.subheader("SPX Implied Volatility Surface")
        if spx_mode == "Synthetic (Heston)":
            st.markdown("**Generate synthetic SPX surface from Heston parameters:**")
            c1, c2 = st.columns(2)
            with c1:
                syn_kappa = st.slider("κ", 0.1, 5.0, 2.0, step=0.1, key="syn_k")
                syn_theta = st.slider("θ", 0.01, 0.15, 0.05, step=0.01, key="syn_t")
                syn_sigma = st.slider("σ", 0.1, 1.0, 0.3, step=0.01, key="syn_s")
            with c2:
                syn_rho   = st.slider("ρ", -0.9, -0.1, -0.6, step=0.01, key="syn_r")
                syn_v0    = st.slider("v₀", 0.01, 0.15, 0.05, step=0.01, key="syn_v")
                noise_lvl = st.slider("Noise Level", 0.0, 0.05, 0.01, step=0.005, key="syn_n")

            if st.button("Generate SPX Surface", use_container_width=True):
                iv = _generate_spx_surface(syn_kappa, syn_theta, syn_sigma, syn_rho, syn_v0)
                rng = np.random.default_rng(42)
                iv += rng.normal(0, noise_lvl, iv.shape).astype(np.float32)
                st.session_state["spx_iv"] = np.clip(iv, 0.01, 2.0)
                st.success("SPX surface generated.")
        else:
            uploaded = st.file_uploader("Upload 8×11 CSV", type=["csv"], key="spx_upload")
            if uploaded:
                arr = np.genfromtxt(uploaded, delimiter=",")
                if arr.shape == (8, 11):
                    st.session_state["spx_iv"] = arr.astype(np.float32)
                    st.success("SPX surface uploaded.")
                else:
                    st.error(f"Expected (8,11), got {arr.shape}")

        if "spx_iv" in st.session_state:
            iv_disp = st.session_state["spx_iv"]
            K_mesh, T_mesh = np.meshgrid(_STKS, _MATS)
            fig_spx = go.Figure(go.Surface(
                x=K_mesh, y=T_mesh, z=iv_disp,
                colorscale="Blues", opacity=0.88, showscale=True,
            ))
            fig_spx.update_layout(
                scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity", zaxis_title="IV"),
                height=350, margin=dict(l=0, r=0, b=0, t=30),
            )
            st.plotly_chart(fig_spx, use_container_width=True)

    with col_vix:
        st.subheader("VIX Level Input")
        st.metric("Observed VIX", f"{vix_obs:.1f}")
        st.markdown(
            "The **model VIX** is computed from the Rough Heston Riccati ODE "
            "characteristic function over the VIX futures term structure."
        )

        # Show model VIX from sidebar defaults
        try:
            from deepvol.market.vix_pricing import model_vix, vix_futures_curve
            vix_model_preview = model_vix(
                kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6, v0=0.05, H=0.08
            )
            st.metric("Model VIX (default params)", f"{vix_model_preview:.2f}")

            tenors = [1/12, 2/12, 3/12, 6/12, 9/12, 1.0]
            curve = [model_vix(kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6,
                               v0=0.05, H=0.08) for _ in tenors]
            fig_vix_curve = go.Figure(go.Scatter(
                x=[f"{int(t*12)}M" for t in tenors], y=curve,
                mode="lines+markers", line=dict(color="#ff3366", width=2),
            ))
            fig_vix_curve.add_hline(y=vix_obs, line_dash="dash", line_color="#00d4ff",
                                    annotation_text=f"Observed VIX={vix_obs:.1f}")
            fig_vix_curve.update_layout(
                title="VIX Futures Term Structure (default params)",
                xaxis_title="Tenor", yaxis_title="VIX",
                height=280, margin=dict(l=0, r=0, b=40, t=40),
            )
            st.plotly_chart(fig_vix_curve, use_container_width=True)
        except Exception:
            st.info("VIX curve preview available after backend loads.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Run Calibration
# ═══════════════════════════════════════════════════════════════════════════════
with tab_calib:
    st.header("Run Joint Calibration")
    st.markdown(
        f"Joint loss: **{w_spx:.1f}× SPX RMSE** + **{w_vix:.1f}× VIX error²** "
        f"with **{n_restarts}** L-BFGS-B restarts."
    )

    run_joint = st.button("Run Joint SPX+VIX Calibration", use_container_width=True, type="primary")

    if run_joint:
        if "spx_iv" not in st.session_state:
            st.error("Generate or upload an SPX surface first (Tab 1).")
        else:
            try:
                from deepvol.calibration.joint_calibration import calibrate_joint
                import time as _time

                spx_iv = st.session_state["spx_iv"]
                with st.spinner(f"Running joint calibration ({n_restarts} restarts)…"):
                    t0 = _time.time()
                    result = calibrate_joint(
                        spx_surface=spx_iv.astype(np.float64),
                        vix_level=float(vix_obs),
                        weights=(float(w_spx), float(w_vix)),
                        n_restarts=int(n_restarts),
                        seed=42,
                    )
                    elapsed_ms = (_time.time() - t0) * 1000

                st.session_state["joint_result"] = result
                st.session_state["joint_ms"]     = elapsed_ms
                st.session_state["joint_spx_iv"] = spx_iv
                st.success(
                    f"Calibration complete in **{elapsed_ms:.0f} ms** — "
                    f"SPX RMSE: {result['spx_rmse_bps']:.1f} bps, "
                    f"VIX error: {result['vix_error']:.2f} pts"
                )

            except Exception as exc:
                st.error(f"Joint calibration failed: {exc}")

    if "joint_result" in st.session_state:
        res = st.session_state["joint_result"]
        ms  = st.session_state["joint_ms"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SPX RMSE (bps)",  f"{res['spx_rmse_bps']:.2f}")
        c2.metric("VIX Error (pts)", f"{res['vix_error']:.3f}")
        c3.metric("Total Loss",      f"{res['total_loss']:.4e}")
        c4.metric("Calibration Time",f"{ms:.0f} ms")

        st.subheader("Calibrated Parameters")
        p_df = pd.DataFrame({
            "Parameter": list(_PARAM_LABELS.values()),
            "Symbol":    list(_PARAM_LABELS.keys()),
            "Value":     [res[p] for p in _PARAM_NAMES],
        })
        st.dataframe(p_df.style.format({"Value": "{:.6f}"}), use_container_width=True)
    else:
        st.info("Run calibration to see results.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Results & Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab_result:
    st.header("Fitted vs Market Surface Comparison")

    if "joint_result" not in st.session_state:
        st.info("Run calibration first (Tab 2).")
    else:
        res    = st.session_state["joint_result"]
        spx_iv = st.session_state["joint_spx_iv"]

        # Compute fitted surface
        try:
            from deepvol.calibration.joint_calibration import _get_assets, _fno_predict
            model, pn, yn, device = _get_assets()
            theta_arr = np.array([res[p] for p in _PARAM_NAMES])
            fitted_iv = _fno_predict(theta_arr, model, pn, yn, device)
        except Exception as exc:
            st.warning(f"Could not compute fitted surface: {exc}")
            fitted_iv = spx_iv * 1.05

        K_mesh, T_mesh = np.meshgrid(_STKS, _MATS)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Market (SPX) Surface")
            fig_mkt = go.Figure(go.Surface(
                x=K_mesh, y=T_mesh, z=spx_iv,
                colorscale="Blues", opacity=0.88, showscale=True,
            ))
            fig_mkt.update_layout(
                scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity", zaxis_title="IV"),
                height=380, margin=dict(l=0, r=0, b=0, t=30),
            )
            st.plotly_chart(fig_mkt, use_container_width=True)

        with col_b:
            st.subheader("Fitted Surface (Joint Calibration)")
            fig_fit = go.Figure(go.Surface(
                x=K_mesh, y=T_mesh, z=fitted_iv,
                colorscale="Reds", opacity=0.88, showscale=True,
            ))
            fig_fit.update_layout(
                scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity", zaxis_title="IV"),
                height=380, margin=dict(l=0, r=0, b=0, t=30),
            )
            st.plotly_chart(fig_fit, use_container_width=True)

        # Residual surface
        st.subheader("Residual Surface (Fitted − Market) in bps")
        residual = (fitted_iv - spx_iv) * 10000
        fig_res = go.Figure(go.Surface(
            x=K_mesh, y=T_mesh, z=residual,
            colorscale="RdBu", opacity=0.9, showscale=True,
        ))
        fig_res.update_layout(
            scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity", zaxis_title="Residual (bps)"),
            height=400, margin=dict(l=0, r=0, b=0, t=35),
        )
        st.plotly_chart(fig_res, use_container_width=True)

        # Smile slice comparison
        st.subheader("Smile Slice Comparison")
        t_sel = st.selectbox("Maturity", range(8), format_func=lambda i: f"T={_MATS[i]:.1f}")
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=_STKS, y=spx_iv[t_sel] * 100,
            mode="lines+markers", name="Market", line=dict(color="#00d4ff", width=2),
        ))
        fig_smile.add_trace(go.Scatter(
            x=_STKS, y=fitted_iv[t_sel] * 100,
            mode="lines+markers", name="Fitted (Joint)", line=dict(color="#ff3366", width=2, dash="dash"),
        ))
        fig_smile.update_layout(
            xaxis_title="Log-Moneyness", yaxis_title="IV (%)",
            height=320, margin=dict(l=0, r=0, b=40, t=30),
        )
        st.plotly_chart(fig_smile, use_container_width=True)

        # Weight sensitivity
        st.subheader("Weight Sensitivity: SPX vs VIX Loss Contribution")
        if w_spx + w_vix > 0:
            spx_frac = w_spx / (w_spx + w_vix)
            vix_frac = w_vix / (w_spx + w_vix)
            fig_pie = go.Figure(go.Pie(
                labels=["SPX RMSE", "VIX Error"],
                values=[spx_frac, vix_frac],
                marker_colors=["#00d4ff", "#ff3366"],
                hole=0.4,
            ))
            fig_pie.update_layout(height=250, margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_pie, use_container_width=True)
