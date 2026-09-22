"""
app_main.py — DeepVol Unified Multi-Page Streamlit Application.

Serves as the landing page and navigation hub for all DeepVol dashboards.
Run with:
    streamlit run src/deepvol/app/app_main.py

Pages registered under src/deepvol/app/pages/ are auto-discovered by Streamlit
as additional navigation items when the app is launched from this entrypoint.
"""
import streamlit as st

st.set_page_config(
    page_title="DeepVol — Quantitative Volatility Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Landing Page ──────────────────────────────────────────────────────────────
st.title("DeepVol — Quantitative Volatility Platform")
st.markdown(
    "**DeepVol** is an end-to-end GPU-accelerated implied volatility modeling platform "
    "built on Fourier Neural Operators, Rough Volatility models, and differentiable hedging.\n\n"
    "Use the **sidebar** (left) to navigate between modules."
)

st.divider()

# ── Module Overview ───────────────────────────────────────────────────────────
st.header("Available Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Core Calibration")
    st.markdown("""
| Dashboard | Description |
|-----------|-------------|
| **Calibration Sandbox** (`dashboard.py`) | Gauss-Newton FNO calibration for 9 models; live Greeks console |
| **Model Zoo v2** (`app_v2.py`) | SABR, Heston, rBergomi, MLSV, Schwartz-Smith with PDF export |
| **Batch Calibration** | Multi-date calibration with H time-series tracking |
| **Joint SPX+VIX** | Jointly calibrate to SPX surface + VIX level |
    """)

with col2:
    st.markdown("### 🚀 Advanced Models (P14–P16)")
    st.markdown("""
| Dashboard | Description |
|-----------|-------------|
| **P16 Grey Rough Bergomi** | CUDA Mittag-Leffler path simulation, fractional beta control |
| **P15 D-XVA Hedging** | Heston MC → PIVOT IV → LSTM policy → P&L variance |
| **P14 PI-M-FNO Adaptation** | Reptile / FOMAML online adaptation to crisis surfaces |
| **EGNO Multi-Asset** | Graph neural operator for basket option pricing |
| **Autocall Pricer** | 1-leg vanilla autocall: MLP surrogate + MC pricing, Greeks, scenario analysis |
| **Phoenix Pricer** | 2-barrier autocall: coupon corridor + memory coupons, MLP surrogate |
| **Worst-of Autocall** | 2-asset correlated Heston, EGNO surrogate, correlation sensitivity |
| **Model Comparison** | Heston / LV / SLV / PDE reference pricer, model risk quantification |
    """)

with col3:
    st.markdown("### ⚠️ Risk & Analytics")
    st.markdown("""
| Dashboard | Description |
|-----------|-------------|
| **VaR / ES Risk Engine** | GPU MC VaR & Expected Shortfall with FNO pricing |
| **Live Risk Stream** (`app_v3_risk.py`) | WebSocket streaming, stress testing, audit logs |
| **Hurst Dynamics** | Historical H time-series study with ACF & rolling stats |
    """)

st.divider()

# ── System Status ─────────────────────────────────────────────────────────────
st.header("System Status")

import torch

col_a, col_b, col_c, col_d = st.columns(4)

gpu_available = torch.cuda.is_available()
col_a.metric("GPU", "✅ Available" if gpu_available else "⚠️ CPU Only")
if gpu_available:
    device_name = torch.cuda.get_device_name(0)
    total_mem   = torch.cuda.get_device_properties(0).total_memory / 1e9
    col_b.metric("Device", device_name[:24])
    col_c.metric("VRAM", f"{total_mem:.1f} GB")
    free_mem = (torch.cuda.get_device_properties(0).total_memory -
                torch.cuda.memory_allocated(0)) / 1e9
    col_d.metric("Free VRAM", f"{free_mem:.1f} GB")
else:
    col_b.metric("Device", "CPU")
    col_c.metric("VRAM",   "N/A")
    col_d.metric("Free",   "N/A")

# Torch version
st.caption(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda or 'N/A'}")

# CUDA extension status
try:
    import sys
    from pathlib import Path
    cpp_dir = Path(__file__).parent.parent / "cpp"
    if str(cpp_dir) not in sys.path:
        sys.path.insert(0, str(cpp_dir))
    import deepvol_cuda  # noqa: F401
    st.success("✅ `deepvol_cuda` CUDA extension loaded (Grey Rough Bergomi available)")
except ImportError:
    st.warning(
        "⚠️ `deepvol_cuda` extension not found — "
        "P16 Grey Rough Bergomi panel will be unavailable. "
        "Build with: `python src/deepvol/cpp/setup.py build_ext --inplace`"
    )

st.divider()

# ── Quick Links ───────────────────────────────────────────────────────────────
st.header("Quick Navigation")
st.markdown("""
Use the **sidebar pages** (left panel) to jump directly to any module, or launch each dashboard
independently:

```bash
# Main unified app (this file)
streamlit run src/deepvol/app/app_main.py

# Individual dashboards
streamlit run src/deepvol/app/dashboard.py      # Calibration + Live Greeks
streamlit run src/deepvol/app/app_v2.py         # Model Zoo v2
streamlit run src/deepvol/app/app_v3_risk.py    # Live Risk / WebSocket

# Phase 14–16 panels (standalone)
streamlit run src/deepvol/app/pages/p16_grey_bergomi.py
streamlit run src/deepvol/app/pages/p15_dxva_hedging.py
streamlit run src/deepvol/app/pages/p14_meta_adaptation.py
```
""")

st.divider()
st.caption(
    "DeepVol v3.0 | P1–P16 implementation complete | "
    "GPU-first, float64 pricing layers, SR 26-2 model risk compliance"
)
