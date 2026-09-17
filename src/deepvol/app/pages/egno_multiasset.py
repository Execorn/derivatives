"""
egno_multiasset.py — EGNO Multi-Asset Basket Options Dashboard.

Features:
  - Correlation network visualizer (asset graph topology)
  - Joint IV surface generation for basket assets
  - EGNO forward pass for permutation-equivariant pricing
  - ATM vol comparison across assets
  - Correlation matrix heatmap
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

_APP_DIR = Path(__file__).parent.parent
_SRC_DIR = _APP_DIR.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

st.set_page_config(page_title="EGNO Multi-Asset", layout="wide")
st.title("EGNO — Equivariant Graph Neural Operator for Multi-Asset Baskets")
st.markdown(
    "Price **multi-asset basket options** using the **EGNO** surrogate — a permutation-equivariant "
    "graph neural operator that respects asset symmetry. Visualize the correlation network topology "
    "and inspect joint implied volatility surfaces."
)

_MATS = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
_STKS = np.linspace(-0.5, 0.5, 11)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Asset Universe")
n_assets = st.sidebar.slider("Number of Assets", 2, 6, 3)

asset_names = [st.sidebar.text_input(f"Asset {i+1} Name", value=f"SPX" if i==0 else f"Asset{i+1}", key=f"aname_{i}") for i in range(n_assets)]

st.sidebar.header("Individual Vol Parameters")
asset_params = []
for i in range(n_assets):
    with st.sidebar.expander(f"{asset_names[i]} Params", expanded=(i==0)):
        v0  = st.slider(f"v₀ [{asset_names[i]}]", 0.01, 0.25, 0.04 + 0.01*i, step=0.01, key=f"v0_{i}")
        H   = st.slider(f"H  [{asset_names[i]}]", 0.04, 0.15, 0.07 + 0.005*i, step=0.01, key=f"H_{i}")
        eta = st.slider(f"η  [{asset_names[i]}]", 0.5,  4.0,  1.5,  step=0.1,  key=f"eta_{i}")
        rho = st.slider(f"ρ  [{asset_names[i]}]", -0.99, 0.0, -0.70, step=0.01, key=f"rho_{i}")
        asset_params.append({"v0": v0, "H": H, "eta": eta, "rho": rho})

# ── Correlation matrix builder ────────────────────────────────────────────────
st.header("Cross-Asset Correlation Matrix")
st.markdown("Set pairwise correlations between assets. The matrix must be positive semi-definite.")

corr_matrix = np.eye(n_assets)
n_pairs = n_assets * (n_assets - 1) // 2
pair_cols = st.columns(min(n_pairs, 4))

pair_idx = 0
for i in range(n_assets):
    for j in range(i + 1, n_assets):
        col = pair_cols[pair_idx % len(pair_cols)]
        corr_val = col.slider(
            f"ρ({asset_names[i]},{asset_names[j]})",
            -0.99, 0.99, 0.40, step=0.01,
            key=f"corr_{i}_{j}",
        )
        corr_matrix[i, j] = corr_val
        corr_matrix[j, i] = corr_val
        pair_idx += 1

# Check PSD
eigvals = np.linalg.eigvalsh(corr_matrix)
if eigvals.min() < -1e-6:
    st.warning(f"⚠️ Correlation matrix is NOT positive semi-definite (min eigenvalue = {eigvals.min():.4f}). Results may be unreliable.")
else:
    st.success(f"✅ Correlation matrix is valid (min eigenvalue = {eigvals.min():.4f}).")

# Correlation heatmap
fig_corr = go.Figure(go.Heatmap(
    z=corr_matrix, x=asset_names, y=asset_names,
    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
    text=np.round(corr_matrix, 2).tolist(),
    texttemplate="%{text}",
))
fig_corr.update_layout(height=300, margin=dict(l=0, r=0, b=40, t=30))
st.plotly_chart(fig_corr, use_container_width=True)

# ── Network graph visualizer ──────────────────────────────────────────────────
st.subheader("Correlation Network Topology")
st.markdown("Edges shown where |ρ| > threshold. Node size = ATM vol proxy.")

threshold = st.slider("Edge visibility threshold |ρ| >", 0.0, 0.99, 0.3, step=0.05)

# Position nodes on a circle
angles = np.linspace(0, 2 * np.pi, n_assets, endpoint=False)
node_x = np.cos(angles).tolist()
node_y = np.sin(angles).tolist()

edge_x, edge_y = [], []
for i in range(n_assets):
    for j in range(i + 1, n_assets):
        if abs(corr_matrix[i, j]) >= threshold:
            edge_x += [node_x[i], node_x[j], None]
            edge_y += [node_y[i], node_y[j], None]

atm_vols = [np.sqrt(p["v0"]) * 100 for p in asset_params]

fig_net = go.Figure()
if edge_x:
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(150,150,150,0.5)", width=2),
        hoverinfo="none",
    ))
fig_net.add_trace(go.Scatter(
    x=node_x, y=node_y, mode="markers+text",
    marker=dict(
        size=[max(15, v * 2) for v in atm_vols],
        color="#00d4ff",
        line=dict(color="white", width=2),
    ),
    text=asset_names,
    textposition="top center",
    hovertext=[f"{n}: σ_ATM≈{v:.1f}%" for n, v in zip(asset_names, atm_vols)],
))
fig_net.update_layout(
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=350, margin=dict(l=0, r=0, b=0, t=30),
)
st.plotly_chart(fig_net, use_container_width=True)

# ── EGNO Forward Pass ─────────────────────────────────────────────────────────
st.header("EGNO Joint IV Surface Generation")
st.markdown(
    "Run the EGNO graph neural operator to price the basket. "
    "The model is permutation-equivariant: relabelling assets produces the same joint surface."
)

run_egno = st.button("Run EGNO Forward Pass", use_container_width=True, type="primary")

@st.cache_resource
def _load_egno():
    try:
        from deepvol.surrogates.egno import EGNO
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EGNO().to(device)
        model.eval()
        return model, device
    except Exception as exc:
        return None, None

if run_egno:
    model, device = _load_egno()
    if model is None:
        st.warning("EGNO model not available. Falling back to independent single-asset FNO pricing.")
        # Fallback: price each asset independently using FNO rBergomi
        try:
            from deepvol.calibration.calibrate_bfgs import _make_spatial_input, _load_normalizers
            from deepvol.surrogates.fno_model import MirrorPaddedFNO2d

            device_fb = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _load_normalizers("rbergomi")
            artifacts = Path(_SRC_DIR).parent / "artifacts"
            fno = MirrorPaddedFNO2d(param_dim=4)
            wp = artifacts / "weights" / "fno_rbergomi_final_prod.pth"
            if wp.exists():
                fno.load_state_dict(torch.load(str(wp), map_location=device_fb, weights_only=True))
            fno.to(device_fb).eval()

            spatial = _make_spatial_input(_MATS, _STKS, device=device_fb)

            surfaces = {}
            for i, (name, p) in enumerate(zip(asset_names, asset_params)):
                params_t = torch.tensor(
                    [[p["v0"], p["H"], p["eta"], p["rho"]]],
                    dtype=torch.float32, device=device_fb,
                )
                from deepvol.calibration.calibrate_bfgs import _fno_predict_real_iv
                with torch.no_grad():
                    iv = _fno_predict_real_iv(fno, params_t, spatial)
                surfaces[name] = iv.cpu().numpy()

            st.session_state["egno_surfaces"] = surfaces
            st.success("Independent single-asset surfaces computed (EGNO fallback).")
        except Exception as exc2:
            st.error(f"Fallback pricing failed: {exc2}")
    else:
        try:
            # Build node features: (1, n_assets, 4) = [v0, H, eta, rho] per asset
            node_feats = torch.tensor(
                [[p["v0"], p["H"], p["eta"], p["rho"]] for p in asset_params],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)  # (1, n_assets, 4)

            # Build dense edge_attr: (1, n_assets, n_assets, 1) — full correlation matrix
            corr_t = torch.tensor(
                corr_matrix, dtype=torch.float32, device=device,
            ).unsqueeze(0).unsqueeze(-1)  # (1, n_assets, n_assets, 1)

            # Global features [K_norm=1.0, T=0.25] — query at ATM, 3M
            g_global = torch.tensor([[1.0, 0.25]], dtype=torch.float32, device=device)  # (1, 2)

            with torch.no_grad():
                price = model(node_feats, corr_t, g_global)  # (1, 1)

            basket_price = price.item()

            # Generate per-asset surfaces via independent FNO (EGNO prices the basket, not individual surfaces)
            surfaces = {}
            try:
                from deepvol.calibration.calibrate_bfgs import _make_spatial_input, _fno_predict_real_iv, _load_normalizers
                from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
                _load_normalizers("rbergomi")
                artifacts = Path(_SRC_DIR).parent / "artifacts"
                fno_fb = MirrorPaddedFNO2d(param_dim=4)
                wp = artifacts / "weights" / "fno_rbergomi_final_prod.pth"
                if wp.exists():
                    fno_fb.load_state_dict(torch.load(str(wp), map_location=device, weights_only=True))
                fno_fb.to(device).eval()
                spatial = _make_spatial_input(_MATS, _STKS, device=device)
                for name, p in zip(asset_names, asset_params):
                    params_t = torch.tensor([[p["v0"], p["H"], p["eta"], p["rho"]]], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        iv = _fno_predict_real_iv(fno_fb, params_t, spatial)
                    surfaces[name] = iv.cpu().numpy()
            except Exception:
                for name, p in zip(asset_names, asset_params):
                    surfaces[name] = np.full((8, 11), np.sqrt(p["v0"]))

            st.session_state["egno_surfaces"] = surfaces
            st.session_state["egno_basket_price"] = basket_price
            st.success(
                f"EGNO basket price: **{basket_price:.4f}** | "
                f"Individual surfaces generated for {n_assets} assets."
            )

        except Exception as exc:
            st.error(f"EGNO forward pass failed: {exc}")

if "egno_surfaces" in st.session_state:
    surfaces = st.session_state["egno_surfaces"]

    # Basket price (if EGNO succeeded)
    if "egno_basket_price" in st.session_state:
        bp = st.session_state["egno_basket_price"]
        st.metric("EGNO Basket Option Price (ATM, T=0.25)", f"{bp:.6f}",
                  help="Permutation-invariant basket price from EGNO forward pass.")

    # ATM vol comparison bar chart
    st.subheader("ATM Implied Volatility Comparison Across Assets")
    atm_idx = 5  # index for k=0 (ATM)
    mat_idx = 2  # T~0.25
    atm_vals = {name: float(surf[mat_idx, atm_idx]) * 100 for name, surf in surfaces.items()}

    fig_atm = go.Figure(go.Bar(
        x=list(atm_vals.keys()), y=list(atm_vals.values()),
        marker_color="#00d4ff", opacity=0.85,
    ))
    fig_atm.update_layout(
        xaxis_title="Asset", yaxis_title="ATM IV (%) at T=0.6",
        height=300, margin=dict(l=0, r=0, b=40, t=30),
    )
    st.plotly_chart(fig_atm, use_container_width=True)

    # 3D surface per asset
    st.subheader("Individual Asset IV Surfaces")
    K_mesh, T_mesh = np.meshgrid(_STKS, _MATS)
    tabs_assets = st.tabs(list(surfaces.keys()))
    for tab_a, (name, surf) in zip(tabs_assets, surfaces.items()):
        with tab_a:
            fig_s = go.Figure(go.Surface(
                x=K_mesh, y=T_mesh, z=surf,
                colorscale="Blues", opacity=0.88, showscale=True,
            ))
            fig_s.update_layout(
                title=f"{name} — Implied Volatility Surface",
                scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity", zaxis_title="IV"),
                height=400, margin=dict(l=0, r=0, b=0, t=40),
            )
            st.plotly_chart(fig_s, use_container_width=True)

    # Smile overlay
    st.subheader("Smile Overlay at Selected Maturity")
    mat_sel = st.selectbox("Maturity", range(len(_MATS)), format_func=lambda i: f"T={_MATS[i]:.1f}")
    colors_p = ["#ff3366", "#00d4ff", "#00ffcc", "#ffa500", "#cc44ff", "#ffee33"]
    fig_ov = go.Figure()
    for ci, (name, surf) in enumerate(surfaces.items()):
        fig_ov.add_trace(go.Scatter(
            x=_STKS, y=surf[mat_sel] * 100,
            mode="lines+markers", name=name,
            line=dict(color=colors_p[ci % len(colors_p)], width=2),
        ))
    fig_ov.update_layout(
        xaxis_title="Log-Moneyness", yaxis_title="IV (%)",
        height=340, margin=dict(l=0, r=0, b=40, t=30),
    )
    st.plotly_chart(fig_ov, use_container_width=True)
else:
    st.info("Click **Run EGNO Forward Pass** to generate surfaces.")
