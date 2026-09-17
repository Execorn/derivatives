"""
batch_calibration_ui.py — Batch Calibration Panel.

Features:
  - Multi-date calibration of Rough Heston FNO surrogate
  - Progress tracking with per-date results
  - Time-series chart of all 6 Hurst/calibrated parameters (especially H)
  - RMSE heatmap across dates
  - CSV export of calibration history
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path Setup ────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).parent.parent
_SRC_DIR = _APP_DIR.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

st.set_page_config(page_title="Batch Calibration", layout="wide")
st.title("Batch Calibration — Multi-Date Rough Heston")
st.markdown(
    "Calibrate the **FNO v3 surrogate** to market implied volatility surfaces "
    "across multiple dates in parallel. Tracks the time-series of all 6 calibrated "
    "parameters (κ, θ, σ, ρ, v₀, **H**) with emphasis on Hurst exponent dynamics."
)

_PARAM_NAMES = ["kappa", "theta", "sigma", "rho", "v0", "H"]
_PARAM_LABELS = {
    "kappa": "κ (Mean Reversion)",
    "theta": "θ (Long-run Variance)",
    "sigma": "σ (Vol of Vol)",
    "rho":   "ρ (Correlation)",
    "v0":    "v₀ (Initial Variance)",
    "H":     "H (Hurst Exponent)",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Date Range")
start_date = st.sidebar.date_input("Start Date", value=date(2024, 1, 2))
end_date   = st.sidebar.date_input("End Date",   value=date(2024, 1, 12))
currency   = st.sidebar.selectbox("Currency", ["SPX", "BTC", "ETH"])
max_workers = st.sidebar.slider("Parallel Workers", 1, 8, 4)

st.sidebar.header("Surface Mode")
surface_mode = st.sidebar.selectbox(
    "Market Data Source",
    ["Synthetic (Heston prior)", "Live (yfinance/Deribit)"],
)

# ── Generate synthetic surfaces for demo ─────────────────────────────────────
def _synthetic_surfaces(dates: list[str]) -> dict[str, np.ndarray]:
    """Generate synthetic IV surfaces for demo without live market data."""
    from deepvol.models.heston import heston_iv_surface
    rng = np.random.default_rng(seed=42)
    _MATS = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    _STKS = np.linspace(-0.5, 0.5, 11)
    surfaces = {}
    for i, d in enumerate(dates):
        kappa = 2.0 + 0.5 * rng.standard_normal()
        theta = 0.05 + 0.01 * rng.standard_normal()
        sigma = 0.3  + 0.05 * rng.standard_normal()
        rho   = -0.6 + 0.05 * rng.standard_normal()
        v0    = 0.05 + 0.01 * rng.standard_normal()
        p = {
            "kappa": float(np.clip(kappa, 0.5, 5.0)),
            "theta": float(np.clip(theta, 0.01, 0.15)),
            "sigma": float(np.clip(sigma, 0.1, 1.0)),
            "rho":   float(np.clip(rho,  -0.9, -0.1)),
            "v0":    float(np.clip(v0,   0.01, 0.15)),
        }
        iv = heston_iv_surface(p, _MATS, _STKS)
        iv = np.where(np.isfinite(iv), iv, 0.20).astype(np.float32)
        surfaces[d] = iv
    return surfaces


# ── Main Panel ────────────────────────────────────────────────────────────────
dates_all = []
d = start_date
while d <= end_date:
    if d.weekday() < 5:  # business days only
        dates_all.append(d.isoformat())
    d += timedelta(days=1)

st.info(f"**{len(dates_all)} business days** selected from {start_date} to {end_date}.")

col_run, col_status = st.columns([2, 3])
with col_run:
    run_batch = st.button("Run Batch Calibration", use_container_width=True, type="primary")

if run_batch:
    if len(dates_all) == 0:
        st.error("No valid dates in range.")
    else:
        try:
            from deepvol.calibration.batch_calibration import calibrate_batch, results_to_dataframe

            if surface_mode == "Synthetic (Heston prior)":
                surfaces = _synthetic_surfaces(dates_all)
            else:
                surfaces = None  # will fetch from market

            progress_bar = st.progress(0, text="Starting batch calibration…")
            status_placeholder = st.empty()

            t0 = time.time()
            results = calibrate_batch(
                dates=dates_all,
                currency=currency,
                max_workers=max_workers,
                device="auto",
                target_surfaces=surfaces,
                verbose=False,
            )
            elapsed = time.time() - t0

            progress_bar.progress(1.0, text="Done!")
            status_placeholder.success(
                f"Calibrated **{len(results)} surfaces** in **{elapsed:.1f}s** "
                f"({elapsed/len(results)*1000:.0f} ms/surface)."
            )

            df = results_to_dataframe(results)
            st.session_state["batch_df"]      = df
            st.session_state["batch_results"] = results
            st.session_state["batch_elapsed"] = elapsed

        except Exception as exc:
            st.error(f"Batch calibration failed: {exc}")

# ── Results Display ───────────────────────────────────────────────────────────
if "batch_df" in st.session_state:
    df      = st.session_state["batch_df"]
    elapsed = st.session_state["batch_elapsed"]

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dates Calibrated", len(df))
    c2.metric("Avg RMSE (bps)",   f"{df['rmse_bps'].mean():.1f}")
    c3.metric("Avg H",            f"{df['H'].mean():.4f}")
    c4.metric("Total Time (s)",   f"{elapsed:.1f}")

    # Parameter time-series
    st.subheader("Parameter Time-Series")
    param_sel = st.multiselect(
        "Parameters to Plot",
        _PARAM_NAMES,
        default=["H", "sigma", "rho"],
    )

    fig_ts = go.Figure()
    colors = ["#ff3366", "#00d4ff", "#00ffcc", "#ffa500", "#cc44ff", "#ffee33"]
    for ci, pname in enumerate(param_sel):
        if pname in df.columns:
            fig_ts.add_trace(go.Scatter(
                x=df["date"], y=df[pname],
                mode="lines+markers",
                name=_PARAM_LABELS.get(pname, pname),
                line=dict(color=colors[ci % len(colors)], width=2),
            ))
    fig_ts.update_layout(
        xaxis_title="Date", yaxis_title="Parameter Value",
        height=400, margin=dict(l=0, r=0, b=40, t=30),
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Hurst exponent focus
    st.subheader("Hurst Exponent H Dynamics")
    fig_H = go.Figure()
    fig_H.add_trace(go.Scatter(
        x=df["date"], y=df["H"],
        mode="lines+markers+text",
        line=dict(color="#ff3366", width=2),
        fill="tozeroy", fillcolor="rgba(255,51,102,0.15)",
    ))
    fig_H.add_hline(y=0.5,  line_dash="dash", line_color="gray", annotation_text="H=0.5 (BM)")
    fig_H.add_hline(y=df["H"].mean(), line_dash="dot", line_color="#00d4ff",
                    annotation_text=f"Mean H={df['H'].mean():.3f}")
    fig_H.update_layout(
        xaxis_title="Date", yaxis_title="Hurst Exponent H",
        height=320, margin=dict(l=0, r=0, b=40, t=30),
    )
    st.plotly_chart(fig_H, use_container_width=True)

    # RMSE bar chart
    st.subheader("Calibration RMSE per Date (bps)")
    fig_rmse = go.Figure(go.Bar(
        x=df["date"].tolist(), y=df["rmse_bps"].tolist(),
        marker_color="#00d4ff", opacity=0.8,
    ))
    fig_rmse.add_hline(y=50, line_dash="dash", line_color="#ff3366",
                       annotation_text="50 bps threshold")
    fig_rmse.update_layout(
        xaxis_title="Date", yaxis_title="RMSE (bps)",
        height=300, margin=dict(l=0, r=0, b=60, t=30),
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

    # Raw data table
    st.subheader("Calibration Results Table")
    display_cols = ["date", "rmse_bps", "converged"] + _PARAM_NAMES
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[display_cols].style.format({p: "{:.5f}" for p in _PARAM_NAMES if p in df.columns}
                                      | {"rmse_bps": "{:.2f}"}),
        use_container_width=True,
    )

    # CSV download
    csv_bytes = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results as CSV",
        data=csv_bytes,
        file_name=f"batch_calibration_{currency}_{start_date}_{end_date}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("Set the date range and click **Run Batch Calibration** to start.")
