"""
Streamlit Dashboard Panel for Worst-of 2-Asset Autocallable Notes.
"""

import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from deepvol.models.wof_autocall import simulate_correlated_heston_paths, price_wof_autocall_mc
from deepvol.models.autocall import price_autocall_mc, make_obs_indices
from deepvol.surrogates.wof_autocall_egno import WoFAutocallEGNO, WoFInputNormalizer, WoFOutputNormalizer

st.set_page_config(page_title="Worst-of Autocall Pricer", layout="wide")
st.title("📊 Worst-of Autocall Pricer — 2-Asset Correlated Heston")
st.caption("Multi-asset structured product evaluated on the minimum performance: min(S1_t/S1_0, S2_t/S2_0) >= B.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def _load_wof_surrogate():
    weights_path = "artifacts/weights/wof_mlp_best.pth"
    norm_in_path = "artifacts/scalers/wof_input_normalizer.npz"
    norm_out_path = "artifacts/scalers/wof_output_normalizer.npz"

    if os.path.exists(weights_path) and os.path.exists(norm_in_path) and os.path.exists(norm_out_path):
        norm_in = WoFInputNormalizer.load(norm_in_path)
        norm_out = WoFOutputNormalizer.load(norm_out_path)
        model = WoFAutocallEGNO().to(DEVICE)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.eval()
        return model, norm_in, norm_out
    return None, None, None

surrogate_model, norm_in, norm_out = _load_wof_surrogate()

col_a1, col_a2 = st.columns(2)
with col_a1:
    st.subheader("Asset 1 (Heston)")
    k1 = st.slider("Asset 1 κ", 0.5, 5.0, 2.0, key="k1")
    th1 = st.slider("Asset 1 θ", 0.01, 0.15, 0.04, key="th1")
    sig1 = st.slider("Asset 1 ξ", 0.1, 1.0, 0.3, key="sig1")
    rho1 = st.slider("Asset 1 ρ_sv", -0.9, -0.1, -0.7, key="rho1")
    v01 = st.slider("Asset 1 v₀", 0.01, 0.15, 0.04, key="v01")

with col_a2:
    st.subheader("Asset 2 (Heston)")
    k2 = st.slider("Asset 2 κ", 0.5, 5.0, 2.0, key="k2")
    th2 = st.slider("Asset 2 θ", 0.01, 0.15, 0.04, key="th2")
    sig2 = st.slider("Asset 2 ξ", 0.1, 1.0, 0.4, key="sig2")
    rho2 = st.slider("Asset 2 ρ_sv", -0.9, -0.1, -0.6, key="rho2")
    v02 = st.slider("Asset 2 v₀", 0.01, 0.15, 0.04, key="v02")

st.subheader("Coupling & Contract Terms")
col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
    rho_12 = st.slider("Asset Correlation (ρ₁₂)", 0.0, 0.95, 0.5, 0.05)
    B = st.slider("Autocall Barrier (B)", 0.85, 1.10, 1.00, 0.01)
with col_c2:
    coupon = st.slider("Annual Coupon (c)", 0.03, 0.25, 0.10, 0.01)
    T = st.slider("Tenor (T)", 0.5, 3.0, 1.0, 0.25)
with col_c3:
    r = st.slider("Risk-Free Rate (r)", 0.00, 0.08, 0.03, 0.005)
    obs_freq = st.selectbox("Obs Frequency", [4, 8, 12], index=0)

N_steps = int(round(T * 252))
n_obs = max(1, int(round(obs_freq * T)))
obs_indices = make_obs_indices(n_obs, T, N_steps)

# Price calculation
theta1_t = torch.tensor([[k1, th1, sig1, rho1, v01]], dtype=torch.float64, device=DEVICE)
theta2_t = torch.tensor([[k2, th2, sig2, rho2, v02]], dtype=torch.float64, device=DEVICE)
rho_t = torch.tensor([rho_12], dtype=torch.float64, device=DEVICE)
try:
    S1, S2 = simulate_correlated_heston_paths(theta1_t, theta2_t, rho_t, 100.0, 100.0, T, N_steps, 10000, r, DEVICE)
    B_t = torch.tensor([B], dtype=torch.float64, device=DEVICE)
    c_t = torch.tensor([coupon], dtype=torch.float64, device=DEVICE)
    r_t = torch.tensor([r], dtype=torch.float64, device=DEVICE)

    npv_wof, cp_wof, el_wof = price_wof_autocall_mc(S1, S2, obs_indices, B_t, c_t, r_t, T, T / N_steps)
    npv_a1, cp_a1, _ = price_autocall_mc(S1, obs_indices, B_t, c_t, r_t, T, T / N_steps)
    npv_a2, cp_a2, _ = price_autocall_mc(S2, obs_indices, B_t, c_t, r_t, T, T / N_steps)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    st.error("⚠️ GPU out of memory. Reduce MC path count or tenor and retry.")
    st.stop()

st.subheader("Pricing Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Worst-of NPV", f"{float(npv_wof.item()):.4f}", f"{(float(npv_wof.item()) - float(npv_a1.item()))*10000:.0f} bps vs Asset 1")
c2.metric("Call Probability", f"{float(cp_wof.item())*100:.1f}%")
c3.metric("Asset 1 Call Prob", f"{float(cp_a1.item())*100:.1f}%")
c4.metric("Asset 2 Call Prob", f"{float(cp_a2.item())*100:.1f}%")

st.subheader("Worst-of Discount vs Individual Underlyings")
fig = go.Figure(data=[
    go.Bar(name="Asset 1 Only", x=["Note Value"], y=[float(npv_a1.item())], marker_color="#1f77b4"),
    go.Bar(name="Asset 2 Only", x=["Note Value"], y=[float(npv_a2.item())], marker_color="#ff7f0e"),
    go.Bar(name="Worst-of Basket", x=["Note Value"], y=[float(npv_wof.item())], marker_color="#2ca02c"),
])
fig.update_layout(barmode="group", height=380)
st.plotly_chart(fig, use_container_width=True)
