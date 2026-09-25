"""
Streamlit Dashboard Panel for Autocall Model Comparison (Heston vs LV vs SLV vs PDE).
"""

import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from deepvol.models.autocall import price_autocall_mc, make_obs_indices
from deepvol.models.autocall_pde import price_autocall_pde_scalar
from deepvol.models.autocall_slv import lv_vs_heston_comparison

st.set_page_config(page_title="Autocall Model Comparison", layout="wide")
st.title("Autocallable Model Cross-Validation: Heston vs LV vs SLV vs PDE")
st.caption("Cross-model risk and pricing comparison across Stochastic Volatility, Pure Dupire Local Volatility, McKean-Vlasov SLV, and 1D Finite Difference PDE.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with st.sidebar:
    st.header("Contract Terms")
    S0 = st.number_input("Spot S₀", value=100.0, step=1.0)
    B = st.slider("Autocall Barrier (B)", 0.85, 1.15, 1.00, 0.01)
    coupon = st.slider("Annual Coupon", 0.03, 0.25, 0.08, 0.01)
    T = st.slider("Tenor (years)", 0.5, 3.0, 1.0, 0.25)
    obs_freq = st.selectbox("Obs Frequency", [2, 4, 12], index=1)
    r = st.slider("Risk-Free Rate", 0.00, 0.08, 0.03, 0.005)

    st.header("Heston Dynamics")
    kappa = st.slider("κ", 0.5, 5.0, 2.0, 0.1)
    theta = st.slider("θ", 0.01, 0.15, 0.04, 0.005)
    sigma = st.slider("ξ", 0.1, 1.0, 0.3, 0.05)
    rho = st.slider("ρ", -0.9, -0.1, -0.7, 0.05)
    v0 = st.slider("v₀", 0.01, 0.15, 0.04, 0.005)

N_steps = int(round(T * 252))
n_obs = max(1, int(round(obs_freq * T)))
obs_indices = make_obs_indices(n_obs, T, N_steps)

# SVI grid for LV/SLV
T_grid = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float64, device=DEVICE)
K_grid = torch.linspace(-0.5, 0.5, 21, dtype=torch.float64, device=DEVICE)
svi_params = torch.tensor([
    [0.04, 0.1, -0.4, 0.0, 0.1],
    [0.04, 0.1, -0.4, 0.0, 0.1],
    [0.04, 0.1, -0.4, 0.0, 0.1],
    [0.04, 0.1, -0.4, 0.0, 0.1],
], dtype=torch.float64, device=DEVICE)

# Run comparison
try:
    pde_res = price_autocall_pde_scalar(
        S0_val=S0, r=r, T=T, N_S=300, N_T=N_steps, obs_indices=obs_indices,
        B=B, coupon=coupon, sigma_func=flat_sigma_func
    )

    heston_dict = {"kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho, "v0": v0}
    contract_dict = {"S0": S0, "B": B, "coupon": coupon, "T": T, "r": r, "n_obs": obs_freq}

    comp_results = lv_vs_heston_comparison(
        heston_params=heston_dict,
        svi_params=svi_params,
        T_grid=T_grid,
        K_grid=K_grid,
        autocall_contract=contract_dict,
        device=DEVICE,
        n_paths=10000,
    )
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    st.error("GPU memory limit exceeded. Reduce parameters and retry.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
pde_npv = pde_res["npv"]
c1.metric("1D PDE Reference", f"{pde_npv:.4f}", "Exact Reference")
c2.metric("Heston MC", f"{comp_results['heston']['npv']:.4f}", f"{(comp_results['heston']['npv'] - pde_npv)*10000:.1f} bps vs PDE")
c3.metric("Dupire LV MC", f"{comp_results['lv']['npv']:.4f}", f"{(comp_results['lv']['npv'] - pde_npv)*10000:.1f} bps vs PDE")
c4.metric("McKean-Vlasov SLV", f"{comp_results['slv']['npv']:.4f}", f"{(comp_results['slv']['npv'] - pde_npv)*10000:.1f} bps vs PDE")

st.subheader("Model Risk & Valuation Comparison Table")
st.table({
    "Pricing Model": ["1D Crank-Nicolson PDE", "Heston Stochastic Volatility", "Dupire Local Volatility", "Stochastic Local Volatility (SLV)"],
    "NPV Price": [f"{pde_npv:.4f}", f"{comp_results['heston']['npv']:.4f}", f"{comp_results['lv']['npv']:.4f}", f"{comp_results['slv']['npv']:.4f}"],
    "Call Probability": ["—", f"{comp_results['heston']['call_prob']*100:.1f}%", f"{comp_results['lv']['call_prob']*100:.1f}%", f"{comp_results['slv']['call_prob']*100:.1f}%"],
    "Diff from PDE (bps)": ["0.0", f"{(comp_results['heston']['npv'] - pde_npv)*10000:.1f}", f"{(comp_results['lv']['npv'] - pde_npv)*10000:.1f}", f"{(comp_results['slv']['npv'] - pde_npv)*10000:.1f}"],
    "Timing (s)": ["< 0.1s", f"{comp_results['heston']['timing_s']:.2f}s", f"{comp_results['lv']['timing_s']:.2f}s", f"{comp_results['slv']['timing_s']:.2f}s"],
})
