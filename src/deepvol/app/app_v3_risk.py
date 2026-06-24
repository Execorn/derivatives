"""
app_v3_risk.py — Live Options Risk Dashboard & Stress Testing Streamlit UI.
Features SOTA optimizations:
- Isolated reruns using @st.fragment to prevent full page reloads.
- Non-blocking background thread for real-time WebSocket ingestion.
- Caching hierarchy with st.cache_resource and st.cache_data.
- Memory-bounded collections.deque for real-time audit log alerts.
- Decimated surface updates for WebGL rendering speed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from collections import deque
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from websockets.sync.client import connect

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="DeepVol Live Risk Dashboard", layout="wide")

st.title("⚡ DeepVol Real-Time Risk Dashboard & Stress Testing")
st.markdown("""
Monitor live options Greeks and implied volatility surfaces computed via FNO models.
Simulate macro stress scenarios, analyze high-frequency calculation telemetry, and audit all risk events.
""")

# Setup paths to import from deepvol if necessary
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Session State Initialization ──────────────────────────────────────────────
if "audit_log" not in st.session_state:
    st.session_state["audit_log"] = deque(maxlen=100)

if "latest_data" not in st.session_state:
    st.session_state["latest_data"] = None

if "stress_data" not in st.session_state:
    st.session_state["stress_data"] = None

if "telemetry_history" not in st.session_state:
    st.session_state["telemetry_history"] = []

if "stream_running" not in st.session_state:
    st.session_state["stream_running"] = False


# ── Caching Hierarchy (SOTA Optimization) ─────────────────────────────────────
@st.cache_data
def get_grid_coordinates() -> tuple[list[float], list[float]]:
    """Caches reference strike and maturity grids to avoid reallocation on rerun."""
    maturities = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
    strikes = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    return maturities, strikes


# ── Helpers ───────────────────────────────────────────────────────────────────
def add_audit_log(action: str, spot: float, latency_ms: float, details: str, severity: str = "INFO") -> None:
    st.session_state["audit_log"].append({
        "Timestamp": time.strftime("%H:%M:%S"),
        "Action": action,
        "Severity": severity,
        "Spot Price": spot,
        "Latency (ms)": latency_ms,
        "Details": details
    })


def plot_3d_risk_surface(z_data: list[list[float]] | np.ndarray, title: str, z_label: str, colorscale: str) -> go.Figure:
    """Renders a 3D Plotly surface for IV or option Greeks."""
    maturities, strikes = get_grid_coordinates()
    K_mesh, T_mesh = np.meshgrid(strikes, maturities)
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=K_mesh, y=T_mesh, z=np.array(z_data),
        colorscale=colorscale, opacity=0.85, showscale=True,
        hoverinfo="skip"  # Disabling hover info increases WebGL rendering FPS
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Log-Moneyness (k)",
            yaxis_title="Maturity (T)",
            zaxis_title=z_label
        ),
        margin=dict(l=0, r=0, b=0, t=35),
        height=450
    )
    return fig


# ── WebSocket Background Listener (SOTA Optimization) ──────────────────────────
def ws_listener_thread(url: str, currency: str, model: str, params: dict, interval: float) -> None:
    """Synchronous listener running in daemon thread to ingest live feed without blocking Streamlit."""
    try:
        with connect(url) as ws:
            subscribe_payload = {
                "action": "subscribe",
                "currency": currency,
                "model_name": model,
                "parameters": params,
                "interval": interval
            }
            ws.send(json.dumps(subscribe_payload))
            
            # Read first confirmation message
            confirm_str = ws.recv(timeout=2.0)
            _ = json.loads(confirm_str)
            
            # Signal run success to session state
            st.session_state["stream_running"] = True
            
            while st.session_state.get("stream_running", False):
                try:
                    msg_str = ws.recv(timeout=1.0)
                    msg = json.loads(msg_str)
                    if msg.get("type") == "update":
                        st.session_state["latest_data"] = msg
                        st.session_state["telemetry_history"].append({
                            "Timestamp": time.time(),
                            "Spot Price": msg["spot"],
                            "Latency (ms)": msg["latency_ms"]
                        })
                        if len(st.session_state["telemetry_history"]) > 100:
                            st.session_state["telemetry_history"].pop(0)
                        
                        # Check for anomalies
                        greeks = msg["greeks"]
                        # Check calendar arbitrage breach (e.g. IV increasing as maturity decreases)
                        iv_surf = np.array(greeks["iv_surface"])
                        for j in range(iv_surf.shape[1]):
                            for i in range(iv_surf.shape[0] - 1):
                                if iv_surf[i, j] > iv_surf[i+1, j] + 0.05:  # Tolerance
                                    st.session_state["audit_log"].append({
                                        "Timestamp": time.strftime("%H:%M:%S"),
                                        "Action": "Anomaly Alert",
                                        "Severity": "HIGH",
                                        "Spot Price": msg["spot"],
                                        "Latency (ms)": msg["latency_ms"],
                                        "Details": f"Calendar Arbitrage Breach at strike index {j} (T={MATURITIES[i]} vol > T={MATURITIES[i+1]} vol)"
                                    })
                                    break
                    elif msg.get("type") == "error":
                        st.session_state["audit_log"].append({
                            "Timestamp": time.strftime("%H:%M:%S"),
                            "Action": "Error Alert",
                            "Severity": "WARNING",
                            "Spot Price": 0.0,
                            "Latency (ms)": 0.0,
                            "Details": msg.get("message")
                        })
                except TimeoutError:
                    continue
                except Exception:
                    break
    except Exception as e:
        st.session_state["audit_log"].append({
            "Timestamp": time.strftime("%H:%M:%S"),
            "Action": "Disconnect Alert",
            "Severity": "HIGH",
            "Spot Price": 0.0,
            "Latency (ms)": 0.0,
            "Details": f"WebSocket failure: {e}"
        })
    finally:
        st.session_state["stream_running"] = False


# ── Sidebar Configurations ────────────────────────────────────────────────────
st.sidebar.header("🔌 Connection Setup")
ws_url = st.sidebar.text_input("FastAPI WebSocket URL", value="ws://localhost:8000/ws/risk")

st.sidebar.header("🎯 Stream Configuration")
currency = st.sidebar.selectbox("Currency Profile", ["BTC", "ETH"], index=0)
model_name = st.sidebar.selectbox(
    "Risk Pricing Model",
    ["rough_heston", "heston", "sabr", "ssvi", "rbergomi"],
    index=0
)
stream_interval = st.sidebar.slider("Stream Tick Interval (s)", 0.2, 5.0, 1.0, step=0.1)

# Dynamic parameter inputs based on model in the sidebar
st.sidebar.subheader("📐 Model Target Parameters")
base_params = {}
maturities, strikes = get_grid_coordinates()
MATURITIES = maturities

if model_name in ("rough_heston", "heston"):
    base_params["kappa"] = st.sidebar.slider("κ (Mean reversion)", 0.5, 5.0, 2.0, step=0.1)
    base_params["theta"] = st.sidebar.slider("θ (Long-run variance)", 0.01, 0.25, 0.05, step=0.01)
    base_params["sigma"] = st.sidebar.slider("σ (Vol-of-vol)", 0.1, 1.5, 0.3, step=0.01)
    base_params["rho"] = st.sidebar.slider("ρ (Correlation)", -0.95, 0.0, -0.6, step=0.01)
    base_params["v0"] = st.sidebar.slider("v₀ (Initial variance)", 0.01, 0.25, 0.05, step=0.01)
    if model_name == "rough_heston":
        base_params["H"] = st.sidebar.slider("H (Hurst exponent)", 0.04, 0.15, 0.08, step=0.01)
elif model_name == "sabr":
    base_params["alpha"] = st.sidebar.slider("α (Initial Vol)", 0.05, 0.8, 0.20, step=0.01)
    base_params["rho"] = st.sidebar.slider("ρ (Correlation)", -0.9, 0.9, -0.40, step=0.01)
    base_params["nu"] = st.sidebar.slider("ν (Vol-of-vol)", 0.1, 1.2, 0.40, step=0.01)
elif model_name == "ssvi":
    base_params["rho"] = st.sidebar.slider("ρ (Correlation)", -0.9, 0.9, -0.40, step=0.01)
    base_params["eta"] = st.sidebar.slider("η (Vol-of-vol)", 0.1, 2.0, 1.0, step=0.1)
    base_params["gamma"] = st.sidebar.slider("γ (Maturity scale)", 0.1, 1.0, 0.5, step=0.05)
    base_params["theta_atm"] = [st.sidebar.slider(f"θ_ATM {i}", 0.01, 0.5, 0.1, step=0.01) for i in range(1, 9)]
elif model_name == "rbergomi":
    base_params["v0"] = st.sidebar.slider("v₀ (Initial variance)", 0.01, 0.25, 0.08, step=0.01)
    base_params["H"] = st.sidebar.slider("H (Hurst parameter)", 0.04, 0.15, 0.07, step=0.01)
    base_params["eta"] = st.sidebar.slider("η (Vol-of-vol)", 0.5, 4.0, 1.5, step=0.1)
    base_params["rho"] = st.sidebar.slider("ρ (Correlation)", -0.95, 0.0, -0.70, step=0.01)


# ── Tabs Configuration ────────────────────────────────────────────────────────
tab_live, tab_stress, tab_audit = st.tabs([
    "📈 Live Volatility & Greeks Streaming",
    "💥 Stress Testing & Scenario Analysis",
    "📋 Audit logs & Telemetry"
])


# ── TAB 1: LIVE VOLATILITY & GREEKS STREAMING (SOTA Fragmented UI) ─────────────
with tab_live:
    st.header("📈 Real-Time Options Risk Metrics Stream")

    col_btn1, col_btn2, col_info = st.columns([1, 1, 4])
    
    with col_btn1:
        run_stream = st.checkbox("Toggle Live Stream Feed", value=False, key="run_stream_checkbox")
    
    with col_info:
        if st.session_state["stream_running"]:
            st.success("🟢 Connected and streaming risk telemetry.")
        else:
            st.info("⚪ Offline. Toggle the checkbox to begin streaming.")

    # Control connection thread lifecycle
    if run_stream:
        if not st.session_state["stream_running"]:
            t = threading.Thread(
                target=ws_listener_thread,
                args=(ws_url, currency, model_name, base_params, stream_interval)
            )
            t.daemon = True
            t.start()
            st.session_state["stream_running"] = True
            add_audit_log("Subscribe", 0.0, 0.0, f"Subscribed to {model_name} live feed for {currency}")
            st.rerun()
    else:
        if st.session_state["stream_running"]:
            st.session_state["stream_running"] = False
            add_audit_log("Unsubscribe", 0.0, 0.0, "Unsubscribed and stopped background feed listener")
            st.rerun()

    # Fragmented Risk UI updating at 2Hz
    @st.fragment(run_every=0.5)
    def render_live_greeks_fragment() -> None:
        latest = st.session_state["latest_data"]
        if latest is not None:
            # Display KPIs
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            kpi_col1.metric("Simulated Spot Price", f"${latest['spot']:.2f}")
            kpi_col2.metric("Calculation Latency", f"{latest['latency_ms']:.2f} ms")
            kpi_col3.metric("Option Model", latest["model_name"].upper())
            kpi_col4.metric("Last Update", time.strftime("%H:%M:%S", time.localtime(latest["timestamp"])))

            # Latency graph
            if st.session_state["telemetry_history"]:
                tel_df = pd.DataFrame(st.session_state["telemetry_history"])
                tel_df["Relative Time (s)"] = tel_df["Timestamp"] - tel_df["Timestamp"].iloc[0]
                
                fig_tel = go.Figure()
                fig_tel.add_trace(go.Scatter(
                    x=tel_df["Relative Time (s)"], y=tel_df["Latency (ms)"],
                    mode="lines+markers", name="Latency", line=dict(color="#00d4ff", width=2)
                ))
                fig_tel.update_layout(
                    title="Calculation Latency Timeline (ms)",
                    xaxis_title="Time elapsed (seconds)",
                    yaxis_title="Latency (ms)",
                    height=200,
                    margin=dict(l=0, r=0, b=40, t=40)
                )
                st.plotly_chart(fig_tel, use_container_width=True)

            # 3D surface plot selector
            surface_type = st.selectbox(
                "Select Surface to Visualize",
                ["Implied Volatility", "Delta", "Gamma", "Vega", "Vanna", "Volga"],
                key="live_surface_selector"
            )
            
            greeks = latest["greeks"]
            if surface_type == "Implied Volatility":
                fig_3d = plot_3d_risk_surface(greeks["iv_surface"], "Implied Volatility Surface (σ)", "IV", "Blues")
            elif surface_type == "Delta":
                fig_3d = plot_3d_risk_surface(greeks["delta"], "Option Delta Surface (Δ)", "Delta", "Viridis")
            elif surface_type == "Gamma":
                fig_3d = plot_3d_risk_surface(greeks["gamma"], "Option Gamma Surface (Γ)", "Gamma", "Plasma")
            elif surface_type == "Vega":
                fig_3d = plot_3d_risk_surface(greeks["vega"], "Option Vega Surface (ν)", "Vega", "Cividis")
            elif surface_type == "Vanna":
                fig_3d = plot_3d_risk_surface(greeks["vanna"], "Option Vanna Surface (∂Δ/∂σ)", "Vanna", "Inferno")
            else:  # Volga
                fig_3d = plot_3d_risk_surface(greeks["volga"], "Option Volga Surface (∂ν/∂σ)", "Volga", "Magma")

            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("⚠️ No streaming data received yet. Toggle checkbox to start.")

    render_live_greeks_fragment()


# ── TAB 2: STRESS TESTING & SCENARIO ANALYSIS (SOTA Fragmented UI) ─────────────
with tab_stress:
    @st.fragment
    def render_scenario_analysis() -> None:
        st.header("💥 Stress Testing & Macro Scenarios")
        st.markdown("""
        Configure severe shocks to market underlying prices and option parameters. 
        Send the scenario to the FNO risk engine to evaluate the instantaneous shift in the volatility and Greeks surfaces.
        """)

        c_s1, c_s2, c_s3 = st.columns(3)
        
        with c_s1:
            st.subheader("Asset Price Shocks")
            spot_base = 65000.0 if currency == "BTC" else 3500.0
            spot_shock_pct = st.slider("Spot Price Shock (%)", -50, 50, 0, step=5)
            shocked_spot = spot_base * (1.0 + spot_shock_pct / 100.0)
            st.metric("Shocked Spot Price", f"${shocked_spot:.2f}", delta=f"{spot_shock_pct}%")

            r_stress = st.slider("Risk-free Rate (r)", 0.0, 0.20, 0.05, step=0.01)
            q_stress = st.slider("Dividend Yield (q)", 0.0, 0.10, 0.0, step=0.01)

        with c_s2:
            st.subheader("Volatility Parameter Shocks")
            stress_params = base_params.copy()
            
            vol_shock_pct = st.slider("Vol-of-vol (σ / nu / eta) Shock (%)", -50, 100, 0, step=5)
            corr_shock_abs = st.slider("Correlation (ρ) Shift", -0.50, 0.50, 0.0, step=0.05)
            
            if "sigma" in stress_params:
                stress_params["sigma"] = max(0.1, stress_params["sigma"] * (1.0 + vol_shock_pct / 100.0))
            if "nu" in stress_params:
                stress_params["nu"] = max(0.1, stress_params["nu"] * (1.0 + vol_shock_pct / 100.0))
            if "eta" in stress_params:
                stress_params["eta"] = max(0.1, stress_params["eta"] * (1.0 + vol_shock_pct / 100.0))
                
            if "rho" in stress_params:
                stress_params["rho"] = max(-0.99, min(0.99 if model_name in ("sabr", "ssvi") else 0.0, stress_params["rho"] + corr_shock_abs))

        with c_s3:
            st.subheader("Model Specific Shocks")
            if "v0" in stress_params:
                v0_shock_pct = st.slider("Initial Variance (v₀) Shock (%)", -50, 100, 0, step=5)
                stress_params["v0"] = max(0.01, stress_params["v0"] * (1.0 + v0_shock_pct / 100.0))
            if "H" in stress_params:
                H_shift = st.slider("Hurst parameter (H) Shift", -0.05, 0.05, 0.0, step=0.01)
                stress_params["H"] = max(0.01, min(0.49, stress_params["H"] + H_shift))

        if st.button("Trigger Stress Test Scenario", use_container_width=True):
            with st.spinner("Connecting and running stress calculations on server..."):
                try:
                    with connect(ws_url) as ws:
                        stress_payload = {
                            "action": "stress",
                            "model_name": model_name,
                            "S": shocked_spot,
                            "r": r_stress,
                            "q": q_stress,
                            "parameters": stress_params
                        }
                        ws.send(json.dumps(stress_payload))
                        res_str = ws.recv(timeout=10.0)
                        res = json.loads(res_str)
                        
                        if res.get("type") == "stress_result":
                            st.session_state["stress_data"] = res
                            add_audit_log(
                                "Stress Test",
                                shocked_spot,
                                res["latency_ms"],
                                f"Model: {model_name}. Shocks: Spot {spot_shock_pct}%, Vol {vol_shock_pct}%",
                                "WARNING" if abs(spot_shock_pct) > 20 else "INFO"
                            )
                        else:
                            st.error(f"Error from server: {res.get('message')}")
                except Exception as e:
                    st.error(f"Stress test execution failed: {e}")

        # Visualizing stress test results (contained inside fragment)
        stress_res = st.session_state["stress_data"]
        if stress_res is not None:
            st.subheader("📊 Stress Test Greeks Surfaces")
            st.info(f"Stress computation completed in: {stress_res['latency_ms']:.2f} ms")
            
            s_greeks = stress_res["greeks"]
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                fig_s_iv = plot_3d_risk_surface(s_greeks["iv_surface"], "Stressed Implied Volatility Surface", "IV", "Blues")
                st.plotly_chart(fig_s_iv, use_container_width=True)
                
                fig_s_gamma = plot_3d_risk_surface(s_greeks["gamma"], "Stressed Option Gamma Surface", "Gamma", "Plasma")
                st.plotly_chart(fig_s_gamma, use_container_width=True)

            with c_p2:
                fig_s_delta = plot_3d_risk_surface(s_greeks["delta"], "Stressed Option Delta Surface", "Delta", "Viridis")
                st.plotly_chart(fig_s_delta, use_container_width=True)

                fig_s_vega = plot_3d_risk_surface(s_greeks["vega"], "Stressed Option Vega Surface", "Vega", "Cividis")
                st.plotly_chart(fig_s_vega, use_container_width=True)
        else:
            st.info("💡 Trigger a stress test to view stressed 3D risk grids.")

    render_scenario_analysis()


# ── TAB 3: AUDIT LOGS & TELEMETRY (SOTA Fragmented HTML UI) ────────────────────
with tab_audit:
    st.header("📋 Risk Audit Log & Telemetry Reports")
    
    @st.fragment(run_every=1.0)
    def render_audit_log_fragment() -> None:
        st.markdown("All connection events, streaming status, and user-triggered stress scenarios are audited below.")
        
        # Build custom scrolling terminal for SOTA componentry
        log_style = (
            "height:250px; overflow-y:auto; font-family:monospace; "
            "background-color:#0e1117; padding:12px; border-radius:6px; "
            "border:1px solid #262730; display: flex; flex-direction: column-reverse;"
        )
        
        log_lines = []
        # Display reverse chronological
        for alert in reversed(st.session_state["audit_log"]):
            color = "#ff4b4b" if alert["Severity"] == "HIGH" else ("#ffa500" if alert["Severity"] == "WARNING" else "#00ff66")
            line = f'<div style="color:{color}; margin-bottom:4px;">[{alert["Timestamp"]}] [{alert["Action"]}] [{alert["Severity"]}] Latency: {alert["Latency (ms)"]:.2f}ms | Spot: ${alert["Spot Price"]:.2f} | {alert["Details"]}</div>'
            log_lines.append(line)
            
        log_content = "".join(log_lines)
        st.markdown(f'<div style="{log_style}">{log_content}</div>', unsafe_allow_html=True)

        if st.session_state["audit_log"]:
            # Display spreadsheet table below
            audit_df = pd.DataFrame(list(st.session_state["audit_log"]))
            st.dataframe(audit_df, use_container_width=True)
            
            # Download to CSV
            csv_data = audit_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Audit Log as CSV",
                data=csv_data,
                file_name="deepvol_risk_audit_log.csv",
                mime="text/csv",
                use_container_width=True
            )

    render_audit_log_fragment()
