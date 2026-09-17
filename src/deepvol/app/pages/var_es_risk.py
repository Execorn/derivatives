"""
var_es_risk.py — GPU Monte Carlo VaR / ES Risk Engine Dashboard.

Features:
  - Portfolio builder (add/remove option positions)
  - Heston scenario simulation on GPU
  - VaR and ES computation with confidence level slider
  - P&L loss distribution histogram
  - Spot path fan chart
  - Sensitivity table (Delta, position size)
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

st.set_page_config(page_title="VaR / ES Risk Engine", layout="wide")
st.title("GPU Monte Carlo VaR & Expected Shortfall")
st.markdown(
    "Compute portfolio-level **Value-at-Risk (VaR)** and **Expected Shortfall (ES)** "
    "using GPU-accelerated Heston Monte Carlo with FNO surrogate pricing.\n\n"
    r"$$\text{VaR}_\alpha = Q_{1-\alpha}(\text{Losses}), \quad "
    r"\text{ES}_\alpha = \mathbb{E}[\text{Loss} \mid \text{Loss} \geq \text{VaR}_\alpha]$$"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Market Parameters (Heston)")
S0     = st.sidebar.number_input("S₀ — Current Spot", value=5000.0, step=50.0)
kappa  = st.sidebar.slider("κ — Mean Reversion",   0.1,  5.0, 2.0, step=0.1)
theta  = st.sidebar.slider("θ — Long-run Variance", 0.01, 0.15, 0.05, step=0.01)
sigma  = st.sidebar.slider("σ — Vol of Vol",        0.1,  1.0,  0.3, step=0.01)
rho    = st.sidebar.slider("ρ — Correlation",      -0.9, -0.1, -0.6, step=0.01)
v0     = st.sidebar.slider("v₀ — Initial Variance", 0.01, 0.15, 0.05, step=0.01)
r_rate = st.sidebar.slider("r — Risk-free Rate",    0.0,  0.10, 0.05, step=0.005)
H_par  = st.sidebar.slider("H — Hurst Exponent",   0.04, 0.15, 0.08, step=0.01)

st.sidebar.header("Simulation Settings")
n_paths    = st.sidebar.selectbox("MC Paths",  [1000, 5000, 10000, 20000], index=1)
n_steps    = st.sidebar.selectbox("Steps",     [5, 10, 20], index=1)
alpha_conf = st.sidebar.slider("Confidence Level α", 0.90, 0.99, 0.95, step=0.01)
dt_days    = st.sidebar.slider("Horizon (days)", 1, 10, 1)

# ── Portfolio Builder ─────────────────────────────────────────────────────────
st.header("Portfolio Builder")
st.markdown("Add option positions. Each row: type, strike, maturity, quantity, notional.")

if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = [
        {"type": "call", "K": 5000.0, "T": 0.25, "quantity":  1.0, "notional": 100.0},
        {"type": "put",  "K": 4800.0, "T": 0.25, "quantity": -2.0, "notional": 100.0},
        {"type": "call", "K": 5200.0, "T": 0.50, "quantity":  1.0, "notional": 50.0},
    ]

portfolio = st.session_state["portfolio"]

# Display editable portfolio table
col_add, col_clear = st.columns([1, 1])
with col_add:
    if st.button("＋ Add Position"):
        portfolio.append({"type": "call", "K": float(S0), "T": 0.25, "quantity": 1.0, "notional": 100.0})
        st.session_state["portfolio"] = portfolio
        st.rerun()
with col_clear:
    if st.button("⟳ Reset Portfolio"):
        st.session_state["portfolio"] = [
            {"type": "call", "K": 5000.0, "T": 0.25, "quantity":  1.0, "notional": 100.0},
        ]
        st.rerun()

for idx, pos in enumerate(portfolio):
    cols = st.columns([1, 1.5, 1.5, 1.5, 1.5, 0.6])
    pos["type"]     = cols[0].selectbox("Type",     ["call", "put"],  index=0 if pos["type"]=="call" else 1, key=f"typ_{idx}")
    pos["K"]        = cols[1].number_input("Strike K", value=float(pos["K"]),  step=50.0, key=f"K_{idx}")
    pos["T"]        = cols[2].number_input("Maturity T", value=float(pos["T"]), step=0.05, min_value=0.01, key=f"T_{idx}")
    pos["quantity"] = cols[3].number_input("Quantity", value=float(pos["quantity"]), step=1.0, key=f"qty_{idx}")
    pos["notional"] = cols[4].number_input("Notional", value=float(pos["notional"]), step=10.0, key=f"ntl_{idx}")
    if cols[5].button("✕", key=f"del_{idx}"):
        portfolio.pop(idx)
        st.session_state["portfolio"] = portfolio
        st.rerun()

st.session_state["portfolio"] = portfolio
st.divider()

# ── Run VaR/ES ────────────────────────────────────────────────────────────────
col_run, _ = st.columns([2, 3])
with col_run:
    run_var = st.button("Compute VaR & ES", use_container_width=True, type="primary")

@st.cache_resource
def _load_var_engine():
    from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
    from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer
    from deepvol.risk.var_engine import MonteCarloVaREngine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = Path(_SRC_DIR).parent / "artifacts"

    model = MirrorPaddedFNO2d()
    wp = artifacts / "weights" / "fno_v2_final_prod.pth"
    if wp.exists():
        model.load_state_dict(torch.load(str(wp), map_location=device, weights_only=True))
    model.to(device).eval()

    pn = ParameterNormalizer.load(str(artifacts / "models" / "param_normalizer_v2.npz"))
    yn = IVSurfaceNormalizer.load(str(artifacts / "models" / "iv_normalizer_v2.npz"))
    engine = MonteCarloVaREngine(model=model, pn=pn, yn=yn, device=device)
    return engine, device

if run_var:
    if not portfolio:
        st.error("Add at least one position to the portfolio.")
    else:
        try:
            engine, device = _load_var_engine()
            theta_arr = np.array([kappa, theta, sigma, rho, v0, H_par])
            dt_float  = dt_days / 252.0

            with st.spinner(f"Simulating {n_paths:,} scenarios ({n_steps} steps)…"):
                result = engine.compute_portfolio_var_es(
                    positions=portfolio,
                    S0=float(S0),
                    theta=theta_arr,
                    r=float(r_rate),
                    dt=dt_float,
                    N_paths=int(n_paths),
                    N_steps=int(n_steps),
                    alpha=float(alpha_conf),
                    block_size=4096,
                    seed=42,
                )

            st.session_state["var_result"] = result
            st.success(
                f"VaR/ES computed on **{str(device).upper()}** — "
                f"VaR {int(alpha_conf*100)}%: **{result['var']:.2f}**, "
                f"ES {int(alpha_conf*100)}%: **{result['es']:.2f}**"
            )
        except Exception as exc:
            st.error(f"VaR/ES computation failed: {exc}")

# ── Results ───────────────────────────────────────────────────────────────────
if "var_result" in st.session_state:
    res = st.session_state["var_result"]
    losses = res["losses"]
    spots  = res["spots"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"VaR {int(alpha_conf*100)}%",  f"{res['var']:.2f}")
    c2.metric(f"ES {int(alpha_conf*100)}%",   f"{res['es']:.2f}")
    c3.metric("Mean Loss",   f"{losses.mean():.2f}")
    c4.metric("Std(Loss)",   f"{losses.std():.2f}")

    # Loss distribution histogram
    st.subheader("P&L Loss Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=losses, nbinsx=60,
        name="Scenario Losses", marker_color="#00d4ff", opacity=0.75,
    ))
    fig_hist.add_vline(
        x=res["var"], line_dash="dash", line_color="#ff3366",
        annotation_text=f"VaR {int(alpha_conf*100)}% = {res['var']:.2f}",
        annotation_position="top right",
    )
    fig_hist.add_vline(
        x=res["es"], line_dash="dot", line_color="#ffa500",
        annotation_text=f"ES = {res['es']:.2f}",
        annotation_position="top left",
    )
    fig_hist.update_layout(
        xaxis_title="Loss (portfolio value change)",
        yaxis_title="Scenario Count",
        height=380, margin=dict(l=0, r=0, b=40, t=30),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Spot fan chart
    st.subheader(f"Terminal Spot Distribution ({dt_days}d horizon)")
    fig_fan = go.Figure()
    fig_fan.add_trace(go.Histogram(
        x=spots, nbinsx=60,
        name="Terminal Spot", marker_color="#ff3366", opacity=0.75,
    ))
    fig_fan.add_vline(x=float(S0), line_dash="dash", line_color="#00d4ff",
                      annotation_text=f"S₀={S0:.0f}")
    fig_fan.update_layout(
        xaxis_title="S_T", yaxis_title="Count",
        height=300, margin=dict(l=0, r=0, b=40, t=30),
    )
    st.plotly_chart(fig_fan, use_container_width=True)

    # Risk metrics table
    st.subheader("Risk Summary Table")
    pct_levels = [0.90, 0.95, 0.99]
    rows = []
    for pct in pct_levels:
        v = float(np.quantile(losses, pct))
        tail = losses[losses >= v]
        e = float(tail.mean()) if len(tail) > 0 else v
        rows.append({"Confidence": f"{int(pct*100)}%", "VaR": f"{v:.2f}", "ES": f"{e:.2f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Scenario CSV download
    csv_bytes = pd.DataFrame({"Loss": losses, "Terminal_Spot": spots}).to_csv(index=False).encode()
    st.download_button(
        "Download Scenario Data (CSV)",
        data=csv_bytes,
        file_name="var_es_scenarios.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("Build a portfolio above and click **Compute VaR & ES** to start.")
