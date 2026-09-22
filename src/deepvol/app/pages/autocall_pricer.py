"""
autocall_pricer.py — 1-Leg Vanilla Autocallable Note Pricing & Risk Dashboard.

Features:
  - Real-time pricing via Autocall Residual MLP surrogate or Full Monte Carlo engine
  - Live Greeks console (Barrier Delta ΔB, Vega, finite-difference Theta)
  - Interactive scenario analysis with isolated st.fragment rendering
  - Early call probability distribution and survival curves across observation dates
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

_APP_DIR = Path(__file__).parent.parent
_SRC_DIR = _APP_DIR.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from deepvol.models.autocall import (
    make_obs_indices,
    price_autocall_mc,
    autocall_upper_bound,
)
from deepvol.surrogates.autocall_normalizer import (
    AutocallInputNormalizer,
    AutocallOutputNormalizer,
)
from deepvol.surrogates.autocall_mlp import AutocallMLP, compute_greeks
from deepvol.hedging.d_xva import simulate_heston_paths

st.set_page_config(page_title="Autocall Pricer", layout="wide")
st.title("Autocall Pricer — 1-Leg Vanilla Autocallable Note")
st.markdown(
    "Interactive pricing, risk analytics, and scenario sensitivity for capital-protected "
    "**vanilla autocallable notes** under Heston stochastic volatility dynamics."
)


@st.cache_resource
def _load_autocall_model() -> Tuple[
    Optional[AutocallMLP],
    Optional[AutocallInputNormalizer],
    Optional[AutocallOutputNormalizer],
]:
    """Load fitted surrogate model and normalizers from disk."""
    weights_path = _SRC_DIR / "artifacts/weights/autocall_mlp_best.pth"
    norm_in_path = _SRC_DIR / "artifacts/scalers/autocall_input_normalizer.npz"
    norm_out_path = _SRC_DIR / "artifacts/scalers/autocall_output_normalizer.npz"

    if not (weights_path.exists() and norm_in_path.exists() and norm_out_path.exists()):
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        norm_in = AutocallInputNormalizer.load(str(norm_in_path))
        norm_out = AutocallOutputNormalizer.load(str(norm_out_path))

        model = AutocallMLP(in_dim=10, hidden=256, n_layers=5, out_dim=3, dropout=0.1)
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, norm_in, norm_out
    except Exception as exc:
        st.warning(f"Could not load trained surrogate: {exc}")
        return None, None, None


# ── Sidebar Inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Heston Dynamics")
kappa = st.sidebar.slider("Mean Reversion (κ)", 0.5, 5.0, 2.0, step=0.1)
theta = st.sidebar.slider("Long-term Variance (θ)", 0.01, 0.15, 0.04, step=0.005)
sigma = st.sidebar.slider("Vol of Vol (σ)", 0.1, 1.0, 0.3, step=0.05)
rho = st.sidebar.slider("Spot-Vol Correlation (ρ)", -0.9, -0.1, -0.7, step=0.05)
v0 = st.sidebar.slider("Initial Variance (v₀)", 0.01, 0.15, 0.04, step=0.005)

st.sidebar.header("Contract Structure")
B_pct = st.sidebar.slider("Autocall Barrier B (% of S₀)", 85, 115, 100, step=1)
coupon_pct = st.sidebar.slider("Annual Coupon Rate (%)", 3.0, 25.0, 10.0, step=0.5)
T = st.sidebar.slider("Maturity T (Years)", 0.5, 3.0, 1.5, step=0.25)
freq_label = st.sidebar.selectbox(
    "Observation Frequency",
    ["Quarterly (4/yr)", "Semi-Annual / 8 per yr", "Monthly (12/yr)"],
)
freq_map = {"Quarterly (4/yr)": 4, "Semi-Annual / 8 per yr": 8, "Monthly (12/yr)": 12}
n_obs_per_year = freq_map[freq_label]

st.sidebar.header("Market Environment")
r_pct = st.sidebar.slider("Risk-free Rate r (%)", 0.0, 8.0, 3.0, step=0.25)

st.sidebar.header("Execution Engine")
pricing_mode = st.sidebar.radio("Engine Mode", ["MLP Surrogate", "Full MC"])
mc_paths = st.sidebar.selectbox("MC Paths (for MC mode)", [1000, 5000, 50000], index=1)


# ── Parameter Packaging ───────────────────────────────────────────────────────
B_val = B_pct / 100.0
coupon_val = coupon_pct / 100.0
r_val = r_pct / 100.0

raw_params = np.array(
    [kappa, theta, sigma, rho, v0, B_val, coupon_val, T, float(n_obs_per_year), r_val],
    dtype=np.float32,
).reshape(1, 10)

surrogate_model, norm_in, norm_out = _load_autocall_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tab_pricing, tab_scenarios, tab_distribution = st.tabs(
    ["📊 Pricing & Greeks", "📈 Scenario Analysis", "📅 Call Probability by Date"]
)

# ── Tab 1: Pricing ────────────────────────────────────────────────────────────
with tab_pricing:
    col_run, col_badge = st.columns([4, 1])
    use_mc = (pricing_mode == "Full MC") or (surrogate_model is None)

    if surrogate_model is None and pricing_mode == "MLP Surrogate":
        st.warning("Trained MLP weights not found. Falling back to GPU Monte Carlo engine.")

    t0 = time.perf_counter()
    if not use_mc and surrogate_model is not None and norm_in is not None and norm_out is not None:
        with torch.no_grad():
            x_tensor = norm_in.to_tensor(raw_params, device=device)
            pred_norm = surrogate_model(x_tensor)
            pred_real = norm_out.inverse_transform_tensor(pred_norm).cpu().numpy()[0]
            npv_val = float(pred_real[0])
            call_prob_val = float(pred_real[1])
            exp_life_val = float(pred_real[2])

        greeks = compute_greeks(surrogate_model, raw_params, norm_in, norm_out)
        delta_B = greeks["delta_B"]
        vega = greeks["vega"]
        theta_greek = greeks["theta"]
        engine_label = "⚡ Autocall MLP Surrogate"
    else:
        # MC Pricing
        n_obs = max(1, int(round(n_obs_per_year * T)))
        N_steps = max(1, int(round(T * 252)))
        dt = T / N_steps
        obs_indices = make_obs_indices(n_obs, T, N_steps)

        theta_t = torch.tensor([[kappa, theta, sigma, rho, v0]], dtype=torch.float64, device=device)
        B_t = torch.tensor([B_val], dtype=torch.float64, device=device)
        c_t = torch.tensor([coupon_val], dtype=torch.float64, device=device)
        r_t = torch.tensor([r_val], dtype=torch.float64, device=device)

        with torch.no_grad():
            S = simulate_heston_paths(theta_t, 100.0, T, N_steps, mc_paths, r_val, device)
            npv_t, call_p_t, exp_l_t = price_autocall_mc(S, obs_indices, B_t, c_t, r_t, T, dt)
            npv_val = float(npv_t.item())
            call_prob_val = float(call_p_t.item())
            exp_life_val = float(exp_l_t.item())

        delta_B = 0.0
        vega = 0.0
        theta_greek = 0.0
        engine_label = f"🎲 GPU Monte Carlo ({mc_paths:,} paths)"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("NPV (% Par)", f"{npv_val * 100.0:.2f}%")
    m2.metric("Call Probability", f"{call_prob_val * 100.0:.1f}%")
    m3.metric("Expected Life", f"{exp_life_val:.2f} yrs")
    m4.metric("Barrier Delta (ΔB)", f"{delta_B:.4f}" if not use_mc else "N/A")
    m5.metric("Vega (∂NPV/∂v₀)", f"{vega:.4f}" if not use_mc else "N/A")

    st.caption(f"Engine: **{engine_label}** | Inference Latency: **{elapsed_ms:.2f} ms** | Device: **{device.type.upper()}**")

    with st.expander("📋 Contract Terms & Analytical Upper Bound", expanded=True):
        ub = autocall_upper_bound(max(1, int(round(n_obs_per_year * T))), coupon_val, T, r_val)
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f"""
            - **Underlying**: Single-Stock / Index (S₀ = 100)
            - **Autocall Barrier**: {B_pct}% of S₀
            - **Coupon**: {coupon_pct}% p.a.
            - **Frequency**: {n_obs_per_year} observations/year ({max(1, int(round(n_obs_per_year * T)))} total)
            """)
        with c_right:
            st.markdown(f"""
            - **Maturity**: {T:.2f} years
            - **Risk-free Rate**: {r_pct}%
            - **Capital Protection**: 100% at maturity if uncalled
            - **Theoretical Upper Bound**: {ub * 100.0:.2f}%
            """)


# ── Tab 2: Scenario Analysis ──────────────────────────────────────────────────
with tab_scenarios:
    @st.fragment
    def render_scenario_analysis():
        st.subheader("Sensitivity & Scenario Curves")
        st.markdown("Fast multi-parameter sweeps powered by surrogate inference.")

        if surrogate_model is None or norm_in is None or norm_out is None:
            st.info("Train the MLP surrogate to unlock instant scenario curves.")
            return

        # 1. Barrier Sweep
        b_grid = np.linspace(0.85, 1.15, 31)
        grid_params = np.repeat(raw_params, len(b_grid), axis=0)
        grid_params[:, 5] = b_grid

        with torch.no_grad():
            x_b = norm_in.to_tensor(grid_params, device=device)
            preds_b = norm_out.inverse_transform_tensor(surrogate_model(x_b)).cpu().numpy()

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=b_grid * 100, y=preds_b[:, 0] * 100, mode="lines+markers", name="NPV (% Par)", line=dict(color="#1f77b4", width=2)))
        fig1.add_trace(go.Scatter(x=b_grid * 100, y=preds_b[:, 1] * 100, mode="lines", name="Call Prob (%)", yaxis="y2", line=dict(color="#ff7f0e", dash="dash")))
        fig1.update_layout(
            title="Barrier Sensitivity (NPV & Call Probability vs Barrier %)",
            xaxis_title="Autocall Barrier B (% of S₀)",
            yaxis=dict(title="NPV (%)"),
            yaxis2=dict(title="Call Probability (%)", overlaying="y", side="right"),
            template="plotly_dark",
            height=360,
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Coupon Sensitivity
        c_grid = np.linspace(0.03, 0.25, 23)
        grid_c = np.repeat(raw_params, len(c_grid), axis=0)
        grid_c[:, 6] = c_grid

        with torch.no_grad():
            x_c = norm_in.to_tensor(grid_c, device=device)
            preds_c = norm_out.inverse_transform_tensor(surrogate_model(x_c)).cpu().numpy()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=c_grid * 100, y=preds_c[:, 0] * 100, mode="lines+markers", line=dict(color="#2ca02c", width=2)))
        fig2.update_layout(
            title="Coupon Sensitivity (NPV vs Annual Coupon %)",
            xaxis_title="Annual Coupon (%)",
            yaxis_title="NPV (% Par)",
            template="plotly_dark",
            height=340,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Maturity Curve
        t_grid = np.linspace(0.5, 3.0, 26)
        grid_t = np.repeat(raw_params, len(t_grid), axis=0)
        grid_t[:, 7] = t_grid

        with torch.no_grad():
            x_t = norm_in.to_tensor(grid_t, device=device)
            preds_t = norm_out.inverse_transform_tensor(surrogate_model(x_t)).cpu().numpy()

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t_grid, y=preds_t[:, 0] * 100, mode="lines+markers", name="NPV (% Par)", line=dict(color="#9467bd", width=2)))
        fig3.add_trace(go.Scatter(x=t_grid, y=preds_t[:, 1] * 100, mode="lines", name="Call Prob (%)", yaxis="y2", line=dict(color="#8c564b", dash="dot")))
        fig3.update_layout(
            title="Maturity Term Structure (NPV & Call Prob vs Maturity T)",
            xaxis_title="Maturity T (Years)",
            yaxis=dict(title="NPV (%)"),
            yaxis2=dict(title="Call Probability (%)", overlaying="y", side="right"),
            template="plotly_dark",
            height=340,
        )
        st.plotly_chart(fig3, use_container_width=True)

    render_scenario_analysis()


# ── Tab 3: Call Probability by Date ───────────────────────────────────────────
with tab_distribution:
    st.subheader("Observation Date Early Call Distribution")
    st.markdown("Monte Carlo simulation of first-call hitting probability across scheduled observation dates.")

    if st.button("Run Distribution Simulation (5,000 paths)", type="primary"):
        n_obs = max(1, int(round(n_obs_per_year * T)))
        N_steps = max(1, int(round(T * 252)))
        dt = T / N_steps
        obs_indices = make_obs_indices(n_obs, T, N_steps)

        theta_t = torch.tensor([[kappa, theta, sigma, rho, v0]], dtype=torch.float64, device=device)
        with torch.no_grad():
            S = simulate_heston_paths(theta_t, 100.0, T, N_steps, 5000, 0.0, device)

            called = torch.zeros(5000, dtype=torch.bool, device=device)
            call_at_date = []
            obs_dates = []

            for i, idx in enumerate(obs_indices):
                t_i = idx * dt
                obs_dates.append(f"Obs {i+1} ({t_i:.2f}y)")
                trigger = (S[0, :, idx] >= (B_val * 100.0)) & (~called)
                p_first = float(trigger.float().mean().item())
                call_at_date.append(p_first)
                called = called | trigger

            uncalled_p = float((~called).float().mean().item())

        obs_dates.append("Uncalled at Maturity")
        call_at_date.append(uncalled_p)

        cum_called = np.cumsum(call_at_date[:-1])
        survival_curve = [1.0] + list(1.0 - cum_called)

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(x=obs_dates, y=[p * 100 for p in call_at_date], name="P(Call at Date) %"))
        fig_dist.add_trace(go.Scatter(x=obs_dates, y=[s * 100 for s in survival_curve], mode="lines+markers", name="Survival Probability %", line=dict(color="#ff7f0e", width=3)))
        fig_dist.update_layout(
            title="First Early Redemption Probability by Observation Date",
            xaxis_title="Observation Date",
            yaxis_title="Probability (%)",
            template="plotly_dark",
            height=420,
        )
        st.plotly_chart(fig_dist, use_container_width=True)
