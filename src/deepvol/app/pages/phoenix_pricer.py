"""
Streamlit Dashboard Panel for Phoenix Two-Barrier Autocallable Notes.
"""

import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from deepvol.models.phoenix import price_phoenix_mc, phoenix_decomposition
from deepvol.models.autocall import price_autocall_mc, make_obs_indices
from deepvol.hedging.d_xva import simulate_heston_paths
from deepvol.surrogates.phoenix_mlp import PhoenixMLP, PhoenixInputNormalizer, PhoenixOutputNormalizer

st.set_page_config(page_title="Phoenix Autocall Pricer", layout="wide")
st.title("🦅 Phoenix Autocall Pricer — Two-Barrier Structure")
st.caption("Two-barrier autocallable note: Autocall Barrier (B_call) + Coupon Corridor Barrier (B_cpn) with optional memory coupon accumulation.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def _load_phoenix_surrogate():
    weights_path = "artifacts/weights/phoenix_mlp_best.pth"
    norm_in_path = "artifacts/scalers/phoenix_input_normalizer.npz"
    norm_out_path = "artifacts/scalers/phoenix_output_normalizer.npz"

    if os.path.exists(weights_path) and os.path.exists(norm_in_path) and os.path.exists(norm_out_path):
        norm_in = PhoenixInputNormalizer.load(norm_in_path)
        norm_out = PhoenixOutputNormalizer.load(norm_out_path)
        model = PhoenixMLP().to(DEVICE)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.eval()
        return model, norm_in, norm_out
    return None, None, None

surrogate_model, norm_in, norm_out = _load_phoenix_surrogate()

with st.sidebar:
    st.header("Heston Volatility Dynamics")
    kappa = st.slider("Mean Reversion (κ)", 0.5, 5.0, 2.0, 0.1)
    theta = st.slider("Long-term Variance (θ)", 0.01, 0.15, 0.04, 0.005)
    sigma = st.slider("Vol of Vol (ξ)", 0.1, 1.0, 0.3, 0.05)
    rho = st.slider("Spot-Vol Correlation (ρ)", -0.9, -0.1, -0.7, 0.05)
    v0 = st.slider("Initial Variance (v₀)", 0.01, 0.15, 0.04, 0.005)

    st.header("Phoenix Contract Terms")
    B_call = st.slider("Autocall Barrier (B_call)", 0.90, 1.15, 1.00, 0.01)
    B_cpn_max = round(B_call - 0.02, 2)
    B_cpn = st.slider("Coupon Barrier (B_cpn)", 0.50, B_cpn_max, min(0.85, B_cpn_max), 0.01)
    coupon = st.slider("Annual Coupon (c)", 0.03, 0.25, 0.08, 0.005)
    T = st.slider("Tenor (T, years)", 0.5, 3.0, 1.0, 0.25)
    obs_freq = st.selectbox("Observation Frequency", ["Quarterly (4/y)", "Semi-Annual (2/y)", "Monthly (12/y)"], index=0)
    n_obs_map = {"Quarterly (4/y)": 4, "Semi-Annual (2/y)": 2, "Monthly (12/y)": 12}
    n_obs_per_year = n_obs_map[obs_freq]
    r = st.slider("Risk-Free Rate (r)", 0.00, 0.08, 0.03, 0.005)
    memory = st.checkbox("Memory Coupons (accumulate missed)", value=False)

    mode = st.radio("Evaluation Mode", ["MLP Surrogate", "GPU Monte Carlo"], index=0 if surrogate_model is not None else 1)
    n_paths_mc = st.selectbox("MC Paths", [5000, 25000, 50000], index=0)

n_obs_total = max(1, int(round(n_obs_per_year * T)))
N_steps = int(round(T * 252))
obs_indices = make_obs_indices(n_obs_total, T, N_steps)

# Pricing calculation
if mode == "MLP Surrogate" and surrogate_model is not None:
    x_raw = np.array([[kappa, theta, sigma, rho, v0, B_call, B_cpn, coupon, T, float(n_obs_per_year), r, 1.0 if memory else 0.0]], dtype=np.float32)
    x_t = norm_in.to_tensor(x_raw, DEVICE)
    with torch.no_grad():
        preds_norm = surrogate_model(x_t).cpu().numpy()
    npv_val, cp_val, cpn_val, el_val = norm_out.inverse_transform(preds_norm)[0]
else:
    try:
        theta_t = torch.tensor([[kappa, theta, sigma, rho, v0]], dtype=torch.float64, device=DEVICE)
        S = simulate_heston_paths(theta_t, S0=100.0, T=T, N_steps=N_steps, N_paths=n_paths_mc, r=r, device=DEVICE)
        B_call_t = torch.tensor([B_call], dtype=torch.float64, device=DEVICE)
        B_cpn_t = torch.tensor([B_cpn], dtype=torch.float64, device=DEVICE)
        c_t = torch.tensor([coupon], dtype=torch.float64, device=DEVICE)
        r_t = torch.tensor([r], dtype=torch.float64, device=DEVICE)
        npv_t, cp_t, cpn_t, el_t = price_phoenix_mc(S, obs_indices, B_call_t, B_cpn_t, c_t, r_t, T, T / N_steps, memory=memory)
        npv_val, cp_val, cpn_val, el_val = float(npv_t.item()), float(cp_t.item()), float(cpn_t.item()), float(el_t.item())
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        st.error("⚠️ GPU out of memory. Reduce MC path count or tenor and retry.")
        st.stop()

# Vanilla benchmark for decomposition
theta_t = torch.tensor([[kappa, theta, sigma, rho, v0]], dtype=torch.float64, device=DEVICE)
S_bench = simulate_heston_paths(theta_t, S0=100.0, T=T, N_steps=N_steps, N_paths=5000, r=r, device=DEVICE)
B_call_t = torch.tensor([B_call], dtype=torch.float64, device=DEVICE)
c_t = torch.tensor([coupon], dtype=torch.float64, device=DEVICE)
r_t = torch.tensor([r], dtype=torch.float64, device=DEVICE)
npv_va, _, _ = price_autocall_mc(S_bench, obs_indices, B_call_t, c_t, r_t, T, T / N_steps)
npv_vanilla = float(npv_va.item())

tab1, tab2, tab3 = st.tabs(["📊 Pricing & Decomposition", "🎯 Barrier Sensitivity", "🧠 Memory Analysis"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Phoenix NPV", f"{npv_val:.4f}", f"{(npv_val - npv_vanilla)*10000:.1f} bps vs Vanilla")
    col2.metric("Early Call Prob", f"{cp_val*100:.1f}%")
    col3.metric("Coupon Prob", f"{cpn_val*100:.1f}%")
    col4.metric("Expected Life", f"{el_val:.2f} yrs")

    corridor_val = max(0.0, npv_val - npv_vanilla)
    fig = go.Figure(data=[
        go.Bar(name="Autocall Leg", x=["Phoenix Note"], y=[npv_vanilla], marker_color="#1f77b4"),
        go.Bar(name="Coupon Corridor Leg", x=["Phoenix Note"], y=[corridor_val], marker_color="#2ca02c")
    ])
    fig.update_layout(barmode="stack", title="Phoenix Payoff Value Decomposition", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Barrier Sensitivity Sweep")
    b_calls = np.linspace(0.92, 1.12, 15)
    spread = B_call - B_cpn
    b_cpns = b_calls - spread
    sweep_npvs = []
    for bc, bp in zip(b_calls, b_cpns):
        x_sw = np.array([[kappa, theta, sigma, rho, v0, bc, bp, coupon, T, float(n_obs_per_year), r, 1.0 if memory else 0.0]], dtype=np.float32)
        if surrogate_model is not None:
            x_t = norm_in.to_tensor(x_sw, DEVICE)
            with torch.no_grad():
                out = norm_out.inverse_transform(surrogate_model(x_t).cpu().numpy())[0]
            sweep_npvs.append(out[0])
        else:
            sweep_npvs.append(npv_val)

    fig_sw = go.Figure()
    fig_sw.add_trace(go.Scatter(x=b_calls, y=sweep_npvs, mode="lines+markers", name="NPV vs B_call"))
    fig_sw.update_layout(xaxis_title="Autocall Barrier (B_call)", yaxis_title="NPV", height=400)
    st.plotly_chart(fig_sw, use_container_width=True)

with tab3:
    st.subheader("Memory Coupon Impact")
    st.info(f"Memory enabled: {memory}. When market drops below coupon barrier B_cpn, unpaid coupons accumulate and are paid in full upon the first subsequent observation meeting the corridor condition.")
