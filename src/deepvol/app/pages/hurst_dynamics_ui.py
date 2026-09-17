"""
hurst_dynamics_ui.py — Hurst Exponent Dynamics Historical Study Dashboard.

Features:
  - Date range selection with currency picker
  - Resume-capable historical calibration (calls run_historical_study)
  - Time-series chart of H (and all parameters)
  - Rolling statistics panel
  - Volatility-of-H and autocorrelation analysis
  - Export to CSV
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_APP_DIR = Path(__file__).parent.parent
_SRC_DIR = _APP_DIR.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

st.set_page_config(page_title="Hurst Dynamics", layout="wide")
st.title("Hurst Exponent H — Historical Dynamics Study")
st.markdown(
    "Calibrate the **Rough Heston FNO surrogate** over a historical date range "
    "and track the time-series of the Hurst exponent H. "
    "The study is **resume-capable** — previously completed dates are loaded from cache."
)

_PARAM_NAMES  = ["kappa", "theta", "sigma", "rho", "v0", "H"]
_PARAM_LABELS = {
    "kappa": "κ (Mean Reversion)",
    "theta": "θ (Long-run Variance)",
    "sigma": "σ (Vol of Vol)",
    "rho":   "ρ (Correlation)",
    "v0":    "v₀ (Initial Variance)",
    "H":     "H (Hurst Exponent)",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Study Configuration")
start_date   = st.sidebar.date_input("Start Date", value=date(2023, 10, 1))
end_date     = st.sidebar.date_input("End Date",   value=date(2024, 3, 31))
currency     = st.sidebar.selectbox("Currency", ["SPX", "BTC", "ETH"])
chunk_size   = st.sidebar.slider("Chunk Size (dates per save)", 2, 10, 5)
max_workers  = st.sidebar.slider("Parallel Workers",            1,  8,  4)

st.sidebar.header("Analysis Options")
roll_window = st.sidebar.slider("Rolling Window (days)", 5, 30, 10)

# ── Buttons ───────────────────────────────────────────────────────────────────
col_run, col_load = st.columns([2, 2])
with col_run:
    run_study = st.button(
        "Run / Resume Historical Study", use_container_width=True, type="primary",
        help="Calibrate FNO day-by-day. Resumes from saved checkpoint.",
    )
with col_load:
    load_cached = st.button(
        "Load Cached Results", use_container_width=True,
        help="Load previously saved results without re-running.",
    )

# ── Run study ─────────────────────────────────────────────────────────────────
if run_study:
    try:
        from deepvol.analysis.hurst_dynamics import run_historical_study
        with st.spinner(
            f"Running historical study: {start_date} → {end_date} for {currency}… "
            "(this may take several minutes; progress auto-saves)"
        ):
            df = run_historical_study(
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                currency=currency,
                chunk_size=chunk_size,
                max_workers=max_workers,
                device="auto",
            )
        if df is not None and not df.empty:
            st.session_state["hurst_df"] = df
            st.success(f"Study complete: {len(df)} dates calibrated.")
        else:
            st.warning("No results returned — check logs.")
    except Exception as exc:
        st.error(f"Historical study failed: {exc}")

if load_cached:
    try:
        import json
        project_root = Path(_SRC_DIR).parent
        path = project_root / "results" / "hurst_dynamics" / f"{currency}_hurst_study.json"
        if path.exists():
            from deepvol.calibration.batch_calibration import CalibrationResult, results_to_dataframe
            with open(path) as f:
                data = json.load(f)
            results = [CalibrationResult.from_dict(d) for d in data]
            df = results_to_dataframe(results)
            st.session_state["hurst_df"] = df
            st.success(f"Loaded {len(df)} cached results from `{path}`.")
        else:
            st.warning(f"No cache file found at `{path}`. Run the study first.")
    except Exception as exc:
        st.error(f"Load failed: {exc}")

# ── Results display ───────────────────────────────────────────────────────────
if "hurst_df" in st.session_state:
    df = st.session_state["hurst_df"].copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dates Calibrated", len(df))
    if "H" in df.columns:
        c2.metric("Mean H",  f"{df['H'].mean():.4f}")
        c3.metric("Std H",   f"{df['H'].std():.4f}")
        c4.metric("Min H",   f"{df['H'].min():.4f}")
        c5.metric("Max H",   f"{df['H'].max():.4f}")

    # ── Hurst Exponent Time-Series ────────────────────────────────────────────
    st.subheader("Hurst Exponent H over Time")
    if "H" in df.columns:
        rolling_H = df["H"].rolling(roll_window, min_periods=1).mean()
        rolling_std = df["H"].rolling(roll_window, min_periods=1).std().fillna(0)

        fig_H = go.Figure()
        # Confidence band
        fig_H.add_trace(go.Scatter(
            x=df["date"].tolist() + df["date"].tolist()[::-1],
            y=(rolling_H + rolling_std).tolist() + (rolling_H - rolling_std).tolist()[::-1],
            fill="toself", fillcolor="rgba(0,212,255,0.15)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        # Raw H
        fig_H.add_trace(go.Scatter(
            x=df["date"], y=df["H"],
            mode="markers", name="H (daily)",
            marker=dict(color="rgba(255,51,102,0.5)", size=5),
        ))
        # Rolling mean
        fig_H.add_trace(go.Scatter(
            x=df["date"], y=rolling_H,
            mode="lines", name=f"H ({roll_window}d rolling mean)",
            line=dict(color="#ff3366", width=2),
        ))
        fig_H.add_hline(y=0.5, line_dash="dash", line_color="gray",
                        annotation_text="H=0.5 (standard BM)")
        fig_H.update_layout(
            xaxis_title="Date", yaxis_title="Hurst Exponent H",
            height=380, margin=dict(l=0, r=0, b=40, t=30),
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig_H, use_container_width=True)

    # ── All Parameters Time-Series ────────────────────────────────────────────
    st.subheader("All Calibrated Parameters Over Time")
    param_sel = st.multiselect(
        "Parameters",
        [p for p in _PARAM_NAMES if p in df.columns],
        default=["H", "sigma", "rho"],
    )
    if param_sel:
        colors = ["#ff3366", "#00d4ff", "#00ffcc", "#ffa500", "#cc44ff", "#ffee33"]
        fig_all = go.Figure()
        for ci, p in enumerate(param_sel):
            fig_all.add_trace(go.Scatter(
                x=df["date"], y=df[p],
                mode="lines", name=_PARAM_LABELS.get(p, p),
                line=dict(color=colors[ci % len(colors)], width=1.5),
            ))
        fig_all.update_layout(
            xaxis_title="Date", height=350, margin=dict(l=0, r=0, b=40, t=30),
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig_all, use_container_width=True)

    # ── Rolling Stats ─────────────────────────────────────────────────────────
    if "H" in df.columns:
        st.subheader(f"Rolling Statistics of H ({roll_window}-day window)")
        col_vol, col_acf = st.columns(2)

        with col_vol:
            roll_std = df["H"].rolling(roll_window, min_periods=1).std()
            fig_vol = go.Figure(go.Scatter(
                x=df["date"], y=roll_std,
                mode="lines", fill="tozeroy",
                line=dict(color="#00ffcc", width=1.5),
                fillcolor="rgba(0,255,204,0.2)",
            ))
            fig_vol.update_layout(
                title=f"Rolling Volatility of H ({roll_window}d)",
                xaxis_title="Date", yaxis_title="σ(H)",
                height=300, margin=dict(l=0, r=0, b=40, t=40),
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        with col_acf:
            H_series = df["H"].dropna().values
            max_lag  = min(30, len(H_series) - 1)
            acf_vals = [
                float(np.corrcoef(H_series[:-lag], H_series[lag:])[0, 1])
                if lag > 0 else 1.0
                for lag in range(max_lag + 1)
            ]
            fig_acf = go.Figure(go.Bar(
                x=list(range(max_lag + 1)), y=acf_vals,
                marker_color="#ffa500", opacity=0.8,
            ))
            fig_acf.add_hline(y=0, line_color="gray")
            conf = 1.96 / np.sqrt(len(H_series))
            fig_acf.add_hline(y=conf,  line_dash="dash", line_color="#00d4ff", annotation_text="+95% CI")
            fig_acf.add_hline(y=-conf, line_dash="dash", line_color="#00d4ff")
            fig_acf.update_layout(
                title="ACF of H Time-Series",
                xaxis_title="Lag (days)", yaxis_title="Autocorrelation",
                height=300, margin=dict(l=0, r=0, b=40, t=40),
            )
            st.plotly_chart(fig_acf, use_container_width=True)

    # ── RMSE track ────────────────────────────────────────────────────────────
    if "rmse_bps" in df.columns:
        st.subheader("Calibration RMSE (bps) Over Time")
        fig_rmse = go.Figure(go.Scatter(
            x=df["date"], y=df["rmse_bps"],
            mode="lines+markers", line=dict(color="#cc44ff", width=1.5),
        ))
        fig_rmse.add_hline(y=50, line_dash="dash", line_color="#ff3366",
                           annotation_text="50 bps threshold")
        fig_rmse.update_layout(
            xaxis_title="Date", yaxis_title="RMSE (bps)",
            height=280, margin=dict(l=0, r=0, b=40, t=30),
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    # ── Data Table & Export ───────────────────────────────────────────────────
    st.subheader("Calibration Results Table")
    display_cols = ["date"] + [p for p in _PARAM_NAMES if p in df.columns]
    if "rmse_bps" in df.columns:
        display_cols.append("rmse_bps")
    num_cols = [p for p in _PARAM_NAMES if p in df.columns]
    st.dataframe(
        df[display_cols].style.format(
            {p: "{:.5f}" for p in num_cols} | ({"rmse_bps": "{:.2f}"} if "rmse_bps" in df.columns else {})
        ),
        use_container_width=True,
    )

    csv_bytes = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Full Study as CSV",
        data=csv_bytes,
        file_name=f"hurst_study_{currency}_{start_date}_{end_date}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info(
        "Click **Run / Resume Historical Study** to start calibrating, "
        "or **Load Cached Results** to display previously completed data."
    )
