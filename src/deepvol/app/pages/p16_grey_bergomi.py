"""
p16_grey_bergomi.py — Grey Rough Bergomi (gRB) Dashboard Panel.

Features:
  - CUDA Monte Carlo simulator with beta (fractional order) control
  - Full IV surface generation via GreyRoughBergomiCalibrator
  - Side-by-side smile comparison: gRB vs Classic rBergomi
  - Active learning uncertainty map (ensemble variance from multiple seeds)
  - 3D Plotly surface visualizer
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

# ── Path Setup ────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).parent.parent
_SRC_DIR = _APP_DIR.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_CPP_DIR = _SRC_DIR / "deepvol" / "cpp"
if str(_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(_CPP_DIR))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="P16 — Grey Rough Bergomi", layout="wide")
st.title("Grey Rough Bergomi (gRB) — C++/CUDA Simulator")
st.markdown(
    "Simulate implied volatility surfaces using the **Grey Rough Bergomi** model "
    "with a fractional Mittag-Leffler memory kernel. Compare against classic "
    "Rough Bergomi and inspect the FNO active-learning uncertainty map."
)

# ── gRB Grid Constants ─────────────────────────────────────────────────────────
_T_GRID_GRB  = [0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
_K_GRID_LOG  = [math.log(x) for x in [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]]
_K_GRID_MONO = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

# ── Model Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def _load_cuda_extension():
    try:
        import deepvol_cuda  # noqa: F401
        return True
    except ImportError:
        return False


@st.cache_resource
def _load_grb_calibrator():
    from deepvol.calibration.grey_calibrator import GreyRoughBergomiCalibrator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cal = GreyRoughBergomiCalibrator(T_grid=_T_GRID_GRB, K_grid=_K_GRID_LOG)
    return cal.to(device), device


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("gRB Model Parameters")
v0   = st.sidebar.slider("v₀ — Initial Variance", 0.01, 0.25, 0.08, step=0.01)
H    = st.sidebar.slider("H — Hurst Parameter",   0.01, 0.49, 0.07, step=0.01)
eta  = st.sidebar.slider("η — Vol of Vol",         0.50, 4.00, 1.50, step=0.10)
rho  = st.sidebar.slider("ρ — Correlation",       -0.99, 0.00,-0.70, step=0.01)
beta = st.sidebar.slider(
    "β — Fractional Order (gRB only)", 0.5, 1.0, 0.9, step=0.01,
    help="Mittag-Leffler kernel order. β=1 recovers standard rBergomi.",
)

st.sidebar.header("Simulation Settings")
n_paths    = st.sidebar.selectbox("Monte Carlo Paths", [1024, 2048, 4096, 8192], index=2)
n_steps    = st.sidebar.selectbox("Simulation Steps",  [50, 100, 200], index=1)
unc_seeds  = st.sidebar.slider("Ensemble Seeds (uncertainty)", 2, 5, 3)

cuda_ok = _load_cuda_extension()
if not cuda_ok:
    st.warning(
        "⚠️ CUDA extension not found. "
        "Run `python cpp/setup.py build_ext --inplace` from the project root."
    )

# ── Helper: rBergomi IV surface (FNO surrogate) ───────────────────────────────
@st.cache_data(show_spinner=False)
def _rbergomi_iv_surface(v0_: float, H_: float, eta_: float, rho_: float) -> np.ndarray:
    """Compute rBergomi IV surface using FNO surrogate (8x11 standard grid)."""
    try:
        from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
        from deepvol.calibration.calibrate_bfgs import _make_spatial_input, _fno_predict_real_iv, _load_normalizers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _load_normalizers("rbergomi")
        model = MirrorPaddedFNO2d(param_dim=4)
        weights_path = Path(_SRC_DIR).parent / "artifacts" / "weights" / "fno_rbergomi_final_prod.pth"
        if weights_path.exists():
            model.load_state_dict(torch.load(str(weights_path), map_location=device, weights_only=True))
        model.to(device).eval()
        _MATS = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
        _STKS = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
        spatial = _make_spatial_input(_MATS, _STKS, device=device)
        params_t = torch.tensor([[v0_, H_, eta_, rho_]], dtype=torch.float32, device=device)
        with torch.no_grad():
            iv = _fno_predict_real_iv(model, params_t, spatial)
        return iv.cpu().numpy()
    except Exception as exc:
        st.warning(f"rBergomi FNO surface unavailable: {exc}")
        return np.full((8, 11), 0.20)


# ── Tab Layout ────────────────────────────────────────────────────────────────
tab_mc, tab_compare, tab_uncertainty = st.tabs([
    "Monte Carlo Surface",
    "gRB vs rBergomi Comparison",
    "Active Learning Uncertainty Map",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Monte Carlo Surface
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mc:
    st.header("gRB Monte Carlo Implied Volatility Surface")
    st.markdown(
        "Generates a full 8×9 IV surface using the C++/CUDA Mittag-Leffler path simulator. "
        "Parameter β controls the fractional memory order of the kernel."
    )

    col_run, col_status = st.columns([2, 3])
    with col_run:
        run_mc = st.button("Run gRB Monte Carlo", use_container_width=True, type="primary")

    if run_mc:
        if not cuda_ok:
            st.error("CUDA extension not available. Cannot run gRB Monte Carlo.")
        else:
            try:
                cal, device = _load_grb_calibrator()
                params = torch.tensor([[v0, H, eta, rho, beta]], dtype=torch.float64, device=device)

                t0 = time.time()
                with st.spinner(f"Simulating {n_paths:,} gRB paths ({n_steps} steps)…"):
                    with torch.no_grad():
                        iv_surface = cal(params).squeeze(0).cpu().numpy()
                elapsed = (time.time() - t0) * 1000

                st.session_state["grb_iv"] = iv_surface
                st.session_state["grb_params"] = dict(v0=v0, H=H, eta=eta, rho=rho, beta=beta)
                st.success(f"gRB Monte Carlo completed in **{elapsed:.1f} ms** ({n_paths:,} paths × {n_steps} steps).")

            except Exception as exc:
                st.error(f"gRB simulation failed: {exc}")

    if "grb_iv" in st.session_state:
        iv = st.session_state["grb_iv"]
        params_used = st.session_state["grb_params"]

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ATM 1M IV", f"{iv[0, 4]*100:.2f}%")
        c2.metric("ATM 3M IV", f"{iv[2, 4]*100:.2f}%")
        c3.metric("Min IV", f"{iv.min()*100:.2f}%")
        c4.metric("Max IV", f"{iv.max()*100:.2f}%")

        # 3D surface
        st.subheader("3D Implied Volatility Surface")
        K_mesh, T_mesh = np.meshgrid(_K_GRID_MONO, _T_GRID_GRB)
        fig3d = go.Figure()
        fig3d.add_trace(go.Surface(
            x=K_mesh, y=T_mesh, z=iv,
            colorscale="Blues", opacity=0.88, showscale=True,
        ))
        fig3d.update_layout(
            scene=dict(
                xaxis_title="Strike (moneyness)",
                yaxis_title="Maturity (T)",
                zaxis_title="Implied Vol (σ)",
            ),
            margin=dict(l=0, r=0, b=0, t=35),
            height=480,
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # Smile slices
        st.subheader("Volatility Smile Slices")
        t_sel = st.selectbox(
            "Select Maturity Slice",
            range(len(_T_GRID_GRB)),
            format_func=lambda i: f"T = {_T_GRID_GRB[i]:.2f}",
        )
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=_K_GRID_MONO, y=iv[t_sel, :] * 100,
            mode="lines+markers", name="gRB",
            line=dict(color="#ff3366", width=2),
        ))
        fig_smile.update_layout(
            xaxis_title="Strike (moneyness)",
            yaxis_title="Implied Volatility (%)",
            height=340,
            margin=dict(l=0, r=0, b=40, t=30),
        )
        st.plotly_chart(fig_smile, use_container_width=True)

        # Heatmap table
        st.subheader("IV Surface Heatmap")
        df_iv = pd.DataFrame(
            iv * 100,
            index=[f"T={t:.2f}" for t in _T_GRID_GRB],
            columns=[f"K={k:.2f}" for k in _K_GRID_MONO],
        )
        st.dataframe(df_iv.style.format("{:.2f}%").background_gradient(cmap="Blues"), use_container_width=True)
    else:
        st.info("Click **Run gRB Monte Carlo** to generate the surface.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: gRB vs rBergomi Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.header("gRB vs Classic Rough Bergomi — Smile Comparison")
    st.markdown(
        "Compare the Grey Rough Bergomi smile (fractional kernel, β<1) "
        "against the standard rBergomi smile (FNO surrogate) "
        "at the same (v₀, H, η, ρ) parameters."
    )

    col_l, col_r = st.columns(2)
    with col_l:
        run_compare = st.button("Run Side-by-Side Comparison", use_container_width=True, type="primary")

    if run_compare:
        if not cuda_ok:
            st.error("CUDA extension required for gRB. rBergomi-only comparison not supported here.")
        else:
            try:
                with st.spinner("Simulating gRB surface…"):
                    cal, device = _load_grb_calibrator()
                    params = torch.tensor([[v0, H, eta, rho, beta]], dtype=torch.float64, device=device)
                    t0 = time.time()
                    with torch.no_grad():
                        grb_iv = cal(params).squeeze(0).cpu().numpy()
                    grb_ms = (time.time() - t0) * 1000

                with st.spinner("Loading rBergomi FNO surrogate…"):
                    rb_iv = _rbergomi_iv_surface(v0, H, eta, rho)

                st.session_state["compare_grb"] = grb_iv
                st.session_state["compare_rb"]  = rb_iv
                st.session_state["compare_ms"]  = grb_ms

            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

    if "compare_grb" in st.session_state:
        grb_iv = st.session_state["compare_grb"]
        rb_iv  = st.session_state["compare_rb"]
        grb_ms = st.session_state["compare_ms"]

        st.success(f"gRB MC completed in **{grb_ms:.1f} ms** (rBergomi via FNO).")

        # Maturity slice comparison
        t_sel_c = st.selectbox(
            "Maturity Slice for Comparison",
            range(len(_T_GRID_GRB)),
            format_func=lambda i: f"T = {_T_GRID_GRB[i]:.2f}",
            key="compare_t_sel",
        )

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=_K_GRID_MONO, y=grb_iv[t_sel_c, :] * 100,
            mode="lines+markers", name=f"gRB (β={beta:.2f})",
            line=dict(color="#ff3366", width=2),
        ))

        # Interpolate rBergomi (8x11 grid) to gRB K grid (9 strikes)
        _STKS_STD = np.linspace(-0.5, 0.5, 11)
        for k_idx, k_mono in enumerate(_K_GRID_MONO):
            k_log = math.log(k_mono)
            rb_iv_at_k = np.interp(k_log, _STKS_STD, rb_iv[min(t_sel_c, 7), :] if rb_iv.shape[0] > t_sel_c else rb_iv[-1, :])

        rb_row = rb_iv[min(t_sel_c, rb_iv.shape[0] - 1), :]
        rb_k_interp = np.array([
            np.interp(math.log(k_mono), _STKS_STD, rb_row) for k_mono in _K_GRID_MONO
        ])

        fig_cmp.add_trace(go.Scatter(
            x=_K_GRID_MONO, y=rb_k_interp * 100,
            mode="lines+markers", name="rBergomi (FNO, β=1)",
            line=dict(color="#00d4ff", width=2, dash="dash"),
        ))
        fig_cmp.update_layout(
            xaxis_title="Strike (moneyness)",
            yaxis_title="Implied Volatility (%)",
            height=370,
            margin=dict(l=0, r=0, b=40, t=30),
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Difference surface
        st.subheader("Difference Surface: gRB − rBergomi")
        K_mesh, T_mesh = np.meshgrid(_K_GRID_MONO, _T_GRID_GRB)
        rb_interp_full = np.array([
            [np.interp(math.log(k), _STKS_STD, rb_iv[min(ti, rb_iv.shape[0]-1), :])
             for k in _K_GRID_MONO]
            for ti in range(len(_T_GRID_GRB))
        ])
        diff = (grb_iv - rb_interp_full) * 100

        fig_diff = go.Figure()
        fig_diff.add_trace(go.Surface(
            x=K_mesh, y=T_mesh, z=diff,
            colorscale="RdBu", opacity=0.9, showscale=True,
        ))
        fig_diff.update_layout(
            title="IV Difference: gRB − rBergomi (%)",
            scene=dict(xaxis_title="Strike", yaxis_title="Maturity", zaxis_title="Δ IV (%)"),
            margin=dict(l=0, r=0, b=0, t=35),
            height=420,
        )
        st.plotly_chart(fig_diff, use_container_width=True)
    else:
        st.info("Click **Run Side-by-Side Comparison** to compute both surfaces.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Active Learning Uncertainty Map
# ═══════════════════════════════════════════════════════════════════════════════
with tab_uncertainty:
    st.header("Active Learning — Ensemble Uncertainty Map")
    st.markdown(
        "Runs the gRB simulator with multiple random seeds and computes the **ensemble variance** "
        "across the IV surface. High-variance cells identify regions where the Sobolev FNO "
        "surrogate would benefit most from new training samples."
    )

    run_unc = st.button("Compute Uncertainty Map", use_container_width=True, type="primary")

    if run_unc:
        if not cuda_ok:
            st.error("CUDA extension required for uncertainty computation.")
        else:
            try:
                cal, device = _load_grb_calibrator()
                params_base = torch.tensor([[v0, H, eta, rho, beta]], dtype=torch.float64, device=device)

                surfaces = []
                with st.spinner(f"Running {unc_seeds} ensemble seeds…"):
                    for seed in range(unc_seeds):
                        torch.manual_seed(seed)
                        with torch.no_grad():
                            iv_s = cal(params_base).squeeze(0).cpu().numpy()
                        surfaces.append(iv_s)

                surfaces = np.stack(surfaces, axis=0)
                mean_iv  = surfaces.mean(axis=0)
                std_iv   = surfaces.std(axis=0)

                st.session_state["unc_mean"] = mean_iv
                st.session_state["unc_std"]  = std_iv
                st.success(f"Ensemble of {unc_seeds} simulations complete.")

            except Exception as exc:
                st.error(f"Uncertainty computation failed: {exc}")

    if "unc_std" in st.session_state:
        mean_iv = st.session_state["unc_mean"]
        std_iv  = st.session_state["unc_std"]

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Mean IV Surface")
            K_mesh, T_mesh = np.meshgrid(_K_GRID_MONO, _T_GRID_GRB)
            fig_mean = go.Figure(go.Surface(
                x=K_mesh, y=T_mesh, z=mean_iv,
                colorscale="Blues", opacity=0.9, showscale=True,
            ))
            fig_mean.update_layout(
                scene=dict(xaxis_title="Strike", yaxis_title="Maturity", zaxis_title="Mean IV"),
                margin=dict(l=0, r=0, b=0, t=30), height=380,
            )
            st.plotly_chart(fig_mean, use_container_width=True)

        with col_b:
            st.subheader("Epistemic Uncertainty (Std Dev)")
            fig_std = go.Figure(go.Surface(
                x=K_mesh, y=T_mesh, z=std_iv * 10000,
                colorscale="Hot", opacity=0.9, showscale=True,
            ))
            fig_std.update_layout(
                title="Std Dev across ensemble (bps)",
                scene=dict(xaxis_title="Strike", yaxis_title="Maturity", zaxis_title="σ_ensemble (bps)"),
                margin=dict(l=0, r=0, b=0, t=35), height=380,
            )
            st.plotly_chart(fig_std, use_container_width=True)

        # Heatmap of uncertainty
        st.subheader("Uncertainty Heatmap (bps)")
        df_std = pd.DataFrame(
            std_iv * 10000,
            index=[f"T={t:.2f}" for t in _T_GRID_GRB],
            columns=[f"K={k:.2f}" for k in _K_GRID_MONO],
        )
        # Identify top-5 highest uncertainty cells
        flat_idx = np.argsort(std_iv.ravel())[-5:][::-1]
        top_cells = [(np.unravel_index(i, std_iv.shape), std_iv.ravel()[i] * 10000) for i in flat_idx]

        st.dataframe(
            df_std.style.format("{:.2f}").background_gradient(cmap="hot"),
            use_container_width=True,
        )

        st.subheader("Top-5 High-Uncertainty Cells (Active Learning Targets)")
        tc_data = [
            {
                "Maturity": f"T={_T_GRID_GRB[r]:.2f}",
                "Strike": f"K={_K_GRID_MONO[c]:.2f}",
                "Uncertainty (bps)": f"{unc:.2f}",
                "Priority": f"#{i+1}",
            }
            for i, ((r, c), unc) in enumerate(top_cells)
        ]
        st.dataframe(pd.DataFrame(tc_data), use_container_width=True)
    else:
        st.info("Click **Compute Uncertainty Map** to run the ensemble.")
