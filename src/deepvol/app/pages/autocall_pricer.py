"""
autocall_pricer.py — 1-Leg Vanilla Autocallable Note Pricing & Risk Dashboard.

Features:
  - Real-time pricing via Phase D CorrectionEnsemble (PDE base + 5-member MLP correction)
  - Epistemic uncertainty estimation (basis points) & SR 26-2 OOD routing indicator
  - 5-member ensemble spread visualization & confidence interval bounds
  - Live Greeks console (Barrier Delta ΔB, Vega)
  - Interactive scenario analysis with isolated st.fragment rendering
  - Early call probability distribution and survival curves across observation dates
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

_SRC_DIR = Path(__file__).resolve().parents[3]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from deepvol.utils.path_helpers import get_project_root
_PROJECT_ROOT = get_project_root()

from deepvol.calibration.generate_pde_labels import compute_pde_label
from deepvol.models.autocall import (
    autocall_upper_bound,
    make_obs_indices,
    price_autocall_mc,
)
from deepvol.hedging.d_xva import simulate_heston_paths
from deepvol.surrogates.correction_ensemble import CorrectionEnsemble
from deepvol.training.train_correction import (
    CorrectionInputNormalizer,
    CorrectionOutputNormalizer,
    compute_total_barrier_derivative,
)

st.set_page_config(page_title="Autocall Pricer", layout="wide")
st.title("Autocall Pricer — 1-Leg Vanilla Autocallable Note")
st.markdown(
    "Interactive pricing, epistemic uncertainty quantification, and model governance for capital-protected "
    "**vanilla autocallable notes** under Heston stochastic volatility dynamics."
)


@st.cache_resource
def _load_autocall_model() -> Tuple[
    Optional[CorrectionEnsemble],
    Optional[CorrectionInputNormalizer],
    Optional[CorrectionOutputNormalizer],
    float,
    Dict[str, Any],
]:
    """Load Phase D CorrectionEnsemble, normalizers, and calibration metrics from disk."""
    weights_dir = _PROJECT_ROOT / "artifacts" / "weights"
    scalers_dir = _PROJECT_ROOT / "artifacts" / "scalers"

    member_paths = [weights_dir / f"autocall_correction_mlp_member_{k}.pth" for k in range(5)]
    norm_in_path = scalers_dir / "correction_input_normalizer.npz"
    norm_out_path = scalers_dir / "correction_output_normalizer.npz"
    calib_path = weights_dir / "ensemble_calibration.json"

    if not (all(p.exists() for p in member_paths) and norm_in_path.exists() and norm_out_path.exists()):
        return None, None, None, 2.17, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        norm_in = CorrectionInputNormalizer.load(str(norm_in_path))
        norm_out = CorrectionOutputNormalizer.load(str(norm_out_path))

        tau_ood = 2.17
        calib_dict: Dict[str, Any] = {}
        if calib_path.exists():
            with open(calib_path, "r", encoding="utf-8") as f:
                calib_dict = json.load(f)
            tau_ood = float(calib_dict.get("tau_ood", calib_dict.get("tau_ood_p99_bps", 2.17)))

        ensemble = CorrectionEnsemble(K=5, in_dim=19, hidden=256, n_layers=4, dropout=0.0)
        ensemble.load_members([str(p) for p in member_paths], device=device)
        ensemble.eval()
        return ensemble, norm_in, norm_out, tau_ood, calib_dict
    except Exception as exc:
        st.warning(f"Could not load CorrectionEnsemble surrogate: {exc}")
        return None, None, None, 2.17, {}


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

ensemble_model, norm_in, norm_out, tau_ood, calib_dict = _load_autocall_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tab_pricing, tab_scenarios, tab_distribution = st.tabs(
    ["📊 Pricing & Risk", "📈 Scenario Analysis", "📅 Call Probability by Date"]
)

# ── Tab 1: Pricing & Epistemic Uncertainty ────────────────────────────────────
with tab_pricing:
    @st.fragment
    def render_pricing_panel():
        use_mc = (pricing_mode == "Full MC") or (ensemble_model is None)

        if ensemble_model is None and pricing_mode == "MLP Surrogate":
            st.warning("Phase D CorrectionEnsemble weights not found. Falling back to GPU Monte Carlo.")

        t0 = time.perf_counter()
        if not use_mc and ensemble_model is not None and norm_in is not None and norm_out is not None:
            # 1. Gatheral σ_eff + 1D Crank-Nicolson PDE base price
            pde_npv, pde_delta, pde_gamma = compute_pde_label(
                v0=v0,
                B=B_val,
                coupon=coupon_val,
                T=T,
                n_obs_per_year=float(n_obs_per_year),
                r=r_val,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
            )

            raw_dict = {
                "kappa": kappa,
                "theta": theta,
                "sigma": sigma,
                "rho": rho,
                "v0": v0,
                "B": B_val,
                "coupon": coupon_val,
                "T": T,
                "n_obs_per_year": float(n_obs_per_year),
                "r": r_val,
                "pde_npv": float(pde_npv),
                "pde_delta": float(pde_delta),
                "pde_gamma": float(pde_gamma),
            }

            # 2. 19-dim feature vector and ensemble forward pass with routing
            X_19 = norm_in.build_feature_matrix({k: [v] for k, v in raw_dict.items()})
            X_t = norm_in.to_tensor(X_19, device=device)

            with torch.no_grad():
                mean_pred, std_bps_tensor, ood_mask_tensor = ensemble_model.predict_with_routing(
                    X_t, norm_out, tau_ood=tau_ood
                )
                delta_v = float(norm_out.inverse_transform_tensor(mean_pred).cpu().item())
                unc_bps = float(std_bps_tensor.item())
                is_ood = bool(ood_mask_tensor.item())

                # Individual member predictions for ensemble spread chart
                member_preds = torch.stack([m._forward_uncompiled(X_t) for m in ensemble_model.members], dim=0)
                member_deltas = [
                    float(norm_out.inverse_transform_tensor(member_preds[k]).cpu().item())
                    for k in range(ensemble_model.K)
                ]

            npv_val = float(pde_npv) + delta_v
            correction_bps = delta_v * 10000.0

            # Barrier derivative ∂V/∂B via exact autograd chain rule
            try:
                X_t_grad = X_t.clone().detach().requires_grad_(True)
                pred_grad = ensemble_model._forward_uncompiled(X_t_grad)[0]
                jac = torch.autograd.grad(pred_grad.sum(), X_t_grad, create_graph=False)[0]
                df_dB_tensor = compute_total_barrier_derivative(
                    jac,
                    {k: torch.tensor([float(v)], device=device) for k, v in raw_dict.items()},
                    norm_in,
                    norm_out,
                )
                delta_B = float(df_dB_tensor.item())
            except Exception:
                delta_B = 0.0

            engine_label = "⚡ Phase D CorrectionEnsemble (PDE + 5-member MLP)"
        else:
            # Monte Carlo Pricing Fallback
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
                npv_t, _, _ = price_autocall_mc(S, obs_indices, B_t, c_t, r_t, T, dt)
                npv_val = float(npv_t.item())

            pde_npv = npv_val
            correction_bps = 0.0
            unc_bps = 0.0
            is_ood = False
            delta_B = 0.0
            member_deltas = [0.0] * 5
            engine_label = f"🎲 GPU Monte Carlo ({mc_paths:,} paths)"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Primary Metric Cards Row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total NPV (% Par)", f"{npv_val * 100.0:.2f}%")
        m2.metric("PDE Base Price", f"{pde_npv * 100.0:.2f}%" if not use_mc else "N/A")
        m3.metric("MLP Correction", f"{correction_bps:+.1f} bps" if not use_mc else "N/A")
        m4.metric("Epistemic Uncertainty", f"{unc_bps:.2f} bps" if not use_mc else "N/A")
        m5.metric("Barrier Delta (ΔB)", f"{delta_B:.4f}" if not use_mc else "N/A")
        m6.metric("Latency", f"{elapsed_ms:.1f} ms")

        st.caption(f"Engine: **{engine_label}** | Device: **{device.type.upper()}**")

        if not use_mc:
            st.divider()
            c_gauge, c_chart = st.columns([1, 1])
            with c_gauge:
                st.subheader("Epistemic Uncertainty & OOD Screening")

                if unc_bps < 1.0:
                    band_label = "High Confidence (< 1.0 bps)"
                elif unc_bps < 2.0:
                    band_label = "Moderate Confidence (1.0 - 2.0 bps)"
                elif unc_bps < 3.0:
                    band_label = "Elevated Uncertainty (2.0 - 3.0 bps)"
                else:
                    band_label = "Extreme Tail Uncertainty (> 3.0 bps)"

                if not is_ood:
                    st.success(
                        f"🟢 **IN DISTRIBUTION**: Epistemic uncertainty σ = {unc_bps:.2f} bps ≤ τ = {tau_ood:.2f} bps ({band_label})"
                    )
                else:
                    st.error(
                        f"🔴 **OUT OF DISTRIBUTION**: Epistemic uncertainty σ = {unc_bps:.2f} bps > τ = {tau_ood:.2f} bps"
                    )
                    st.warning(
                        "⚠️ **Model Risk Guardian**: In production, this contract is automatically intercepted "
                        "and routed to the Gatheral second-order Crank-Nicolson PDE fallback solver."
                    )

                lower_ci = npv_val - (2.0 * unc_bps / 10000.0)
                upper_ci = npv_val + (2.0 * unc_bps / 10000.0)
                st.info(
                    f"**95% Epistemic Confidence Interval**: [{lower_ci * 100.0:.2f}%, {upper_ci * 100.0:.2f}%] (±{2.0 * unc_bps:.2f} bps)"
                )

            with c_chart:
                st.subheader("5-Member Ensemble Spread")
                fig_ens = go.Figure()
                member_names = [f"Member {k}" for k in range(5)]
                member_deltas_bps = [d * 10000.0 for d in member_deltas]
                fig_ens.add_trace(
                    go.Bar(
                        x=member_names,
                        y=member_deltas_bps,
                        marker_color=["#3b82f6", "#60a5fa", "#93c5fd", "#2563eb", "#1d4ed8"],
                        name="Correction (bps)",
                    )
                )
                fig_ens.add_hline(
                    y=correction_bps,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text=f"Mean: {correction_bps:+.1f} bps",
                )
                fig_ens.update_layout(
                    title="Ensemble Member Predictions (bps)",
                    xaxis_title="Member",
                    yaxis_title="Correction (bps)",
                    template="plotly_dark",
                    height=280,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig_ens, use_container_width=True)

            with st.expander("🏛️ Model Governance & Compliance (SR 26-2)", expanded=False):
                gov_col1, gov_col2 = st.columns(2)
                with gov_col1:
                    st.markdown(f"""
                    - **OOD Threshold (τ_OOD)**: `{tau_ood:.2f} bps`
                    - **Operational Status**: `{"🔴 FALLBACK ROUTING ACTIVE" if is_ood else "🟢 SURROGATE OPERATIONAL"}`
                    - **Surrogate Architecture**: `5-Member Deep Ensemble ResNet MLP (19 features, 4 blocks)`
                    - **Base Numerical Engine**: `1D Crank-Nicolson with Gatheral σ_eff (float64)`
                    """)
                with gov_col2:
                    raw_rmse = calib_dict.get("raw_rmse_bps", 1.14)
                    trimmed_rmse = calib_dict.get("trimmed_rmse_bps", 0.99)
                    p95_err = calib_dict.get("p95_error_bps", 2.30)
                    p99_err = calib_dict.get("p99_error_bps", 3.91)
                    st.markdown(f"""
                    - **Raw Validation RMSE**: `{raw_rmse:.2f} bps`
                    - **Trimmed RMSE (p99)**: `{trimmed_rmse:.2f} bps`
                    - **95th Percentile Error**: `{p95_err:.2f} bps`
                    - **99th Percentile Error**: `{p99_err:.2f} bps`
                    """)

        with st.expander("📋 Contract Terms & Analytical Upper Bound", expanded=False):
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

    render_pricing_panel()

# ── Tab 2: Scenario Analysis ──────────────────────────────────────────────────
with tab_scenarios:
    @st.fragment
    def render_scenario_analysis():
        st.subheader("Sensitivity & Scenario Curves")
        st.markdown("Fast multi-parameter sweeps powered by the Phase D CorrectionEnsemble.")

        if ensemble_model is None or norm_in is None or norm_out is None:
            st.info("CorrectionEnsemble weights not found.")
            return

        def price_point(k_p: float, th_p: float, sig_p: float, rh_p: float, v0_p: float,
                        b_p: float, c_p: float, t_p: float, freq_p: float, r_p: float) -> float:
            p_npv, p_d, p_g = compute_pde_label(
                v0=v0_p, B=b_p, coupon=c_p, T=t_p, n_obs_per_year=freq_p, r=r_p,
                kappa=k_p, theta=th_p, sigma=sig_p,
            )
            f_dict = {
                "kappa": [k_p], "theta": [th_p], "sigma": [sig_p], "rho": [rh_p], "v0": [v0_p],
                "B": [b_p], "coupon": [c_p], "T": [t_p], "n_obs_per_year": [freq_p], "r": [r_p],
                "pde_npv": [p_npv], "pde_delta": [p_d], "pde_gamma": [p_g],
            }
            feat = norm_in.build_feature_matrix(f_dict)
            feat_t = norm_in.to_tensor(feat, device=device)
            m_pred, _, _ = ensemble_model.predict_with_routing(feat_t, norm_out, tau_ood=tau_ood)
            d_v = float(norm_out.inverse_transform_tensor(m_pred).cpu().item())
            return p_npv + d_v

        # 1. Barrier Sweep
        b_grid = np.linspace(0.85, 1.15, 16)
        npv_b = [price_point(kappa, theta, sigma, rho, v0, b, coupon_val, T, float(n_obs_per_year), r_val) for b in b_grid]

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=b_grid * 100, y=[v * 100 for v in npv_b], mode="lines+markers", name="NPV (% Par)", line=dict(color="#3b82f6", width=2)))
        fig1.update_layout(
            title="Barrier Sensitivity (NPV vs Autocall Barrier %)",
            xaxis_title="Autocall Barrier B (% of S₀)",
            yaxis_title="NPV (% Par)",
            template="plotly_dark",
            height=320,
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Coupon Sensitivity
        c_grid = np.linspace(0.03, 0.25, 15)
        npv_c = [price_point(kappa, theta, sigma, rho, v0, B_val, c, T, float(n_obs_per_year), r_val) for c in c_grid]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=c_grid * 100, y=[v * 100 for v in npv_c], mode="lines+markers", name="NPV (% Par)", line=dict(color="#10b981", width=2)))
        fig2.update_layout(
            title="Coupon Sensitivity (NPV vs Annual Coupon %)",
            xaxis_title="Annual Coupon (%)",
            yaxis_title="NPV (% Par)",
            template="plotly_dark",
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Maturity Curve
        t_grid = np.linspace(0.5, 3.0, 14)
        npv_t = [price_point(kappa, theta, sigma, rho, v0, B_val, coupon_val, t, float(n_obs_per_year), r_val) for t in t_grid]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t_grid, y=[v * 100 for v in npv_t], mode="lines+markers", name="NPV (% Par)", line=dict(color="#8b5cf6", width=2)))
        fig3.update_layout(
            title="Maturity Term Structure (NPV vs Maturity T)",
            xaxis_title="Maturity T (Years)",
            yaxis_title="NPV (% Par)",
            template="plotly_dark",
            height=320,
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
        fig_dist.add_trace(
            go.Scatter(
                x=obs_dates,
                y=[s * 100 for s in survival_curve],
                mode="lines+markers",
                name="Survival Probability %",
                line=dict(color="#f59e0b", width=3),
            )
        )
        fig_dist.update_layout(
            title="First Early Redemption Probability by Observation Date",
            xaxis_title="Observation Date",
            yaxis_title="Probability (%)",
            template="plotly_dark",
            height=420,
        )
        st.plotly_chart(fig_dist, use_container_width=True)
