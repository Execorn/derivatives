"""
p14_meta_adaptation.py — PI-M-FNO Online Meta-Adaptation Dashboard.

Features:
  - Upload or generate a stressed/crisis market IV surface
  - Run Reptile or FOMAML inner adaptation steps
  - PDE loss before/after adaptation comparison
  - Surface correction magnitude heatmap
  - Adaptation speed vs inner steps sweep
"""
from __future__ import annotations

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

st.set_page_config(page_title="P14 — PI-M-FNO Adaptation", layout="wide")
st.title("PI-M-FNO — Physics-Informed Online Meta-Adaptation")
st.markdown(
    "Rapidly adapt the FNO surrogate to a **crisis/stressed** implied volatility surface "
    "using Reptile or FOMAML meta-learning. The frozen spectral core is fixed; "
    "only the output MLP head is updated via inner gradient steps."
)

# ── Constants ─────────────────────────────────────────────────────────────────
_MATS = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
_STKS = np.linspace(-0.5, 0.5, 11, dtype=np.float32)

# ── Model Loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def _load_meta_fno():
    from deepvol.surrogates.meta_fno import MetaFNO2d
    from deepvol.surrogates.pde_loss import DupirePDELoss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetaFNO2d(modes1=12, modes2=12, width=64).to(device)
    pde_loss = DupirePDELoss().to(device)
    return model, pde_loss, device


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Adaptation Algorithm")
algo = st.sidebar.selectbox("Algorithm", ["Reptile", "FOMAML"])
inner_steps = st.sidebar.slider("Inner Gradient Steps", 1, 10, 3)
inner_lr    = st.sidebar.number_input("Inner LR", value=1e-3, format="%.5f", step=1e-4)
outer_lr    = st.sidebar.number_input("Outer LR", value=0.1, format="%.4f", step=0.01)

st.sidebar.header("Crisis Surface Generation")
stress_mode = st.sidebar.selectbox(
    "Stress Type",
    ["Vol Spike (+50%)", "Skew Inversion", "Term Structure Flat", "Custom Upload"],
)

# ── Tab Layout ────────────────────────────────────────────────────────────────
tab_setup, tab_adapt, tab_sweep = st.tabs([
    "Setup Crisis Surface",
    "Run Adaptation",
    "Adaptation Speed Sweep",
])

# ── Helper: generate stressed surface ────────────────────────────────────────
def _generate_base_surface() -> np.ndarray:
    """Generate a calm baseline Heston IV surface."""
    from deepvol.models.heston import heston_iv_surface
    p = {"kappa": 2.0, "theta": 0.05, "sigma": 0.3, "rho": -0.6, "v0": 0.05}
    iv = heston_iv_surface(p, _MATS, _STKS)
    iv = np.where(np.isfinite(iv), iv, 0.20)
    return iv.astype(np.float32)


def _apply_stress(base_iv: np.ndarray, mode: str) -> np.ndarray:
    iv = base_iv.copy()
    if mode == "Vol Spike (+50%)":
        iv *= 1.50
    elif mode == "Skew Inversion":
        iv = iv[:, ::-1]
    elif mode == "Term Structure Flat":
        iv[:] = iv.mean(axis=0, keepdims=True)
    return np.clip(iv, 0.01, 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Setup Crisis Surface
# ═══════════════════════════════════════════════════════════════════════════════
with tab_setup:
    st.header("Configure Crisis Implied Volatility Surface")

    if stress_mode == "Custom Upload":
        uploaded = st.file_uploader("Upload 8×11 CSV (maturities × strikes)", type=["csv"])
        if uploaded:
            arr = np.genfromtxt(uploaded, delimiter=",")
            if arr.shape == (8, 11):
                st.session_state["crisis_iv"] = arr.astype(np.float32)
                st.success("Custom surface loaded.")
            else:
                st.error(f"Expected shape (8, 11), got {arr.shape}.")
    else:
        if st.button("Generate Crisis Surface", use_container_width=True, type="primary"):
            base = _generate_base_surface()
            crisis = _apply_stress(base, stress_mode)
            st.session_state["base_iv"]   = base
            st.session_state["crisis_iv"] = crisis
            st.success(f"Crisis surface generated: **{stress_mode}**")

    if "crisis_iv" in st.session_state:
        crisis_iv = st.session_state["crisis_iv"]
        base_iv   = st.session_state.get("base_iv", crisis_iv)

        col_a, col_b = st.columns(2)
        K_mesh, T_mesh = np.meshgrid(_STKS, _MATS)

        with col_a:
            st.subheader("Baseline IV Surface")
            fig_b = go.Figure(go.Surface(
                x=K_mesh, y=T_mesh, z=base_iv,
                colorscale="Blues", opacity=0.88, showscale=True,
            ))
            fig_b.update_layout(
                scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity", zaxis_title="IV"),
                height=380, margin=dict(l=0, r=0, b=0, t=30),
            )
            st.plotly_chart(fig_b, use_container_width=True)

        with col_b:
            st.subheader(f"Crisis IV Surface ({stress_mode})")
            fig_c = go.Figure(go.Surface(
                x=K_mesh, y=T_mesh, z=crisis_iv,
                colorscale="Reds", opacity=0.88, showscale=True,
            ))
            fig_c.update_layout(
                scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity", zaxis_title="IV"),
                height=380, margin=dict(l=0, r=0, b=0, t=30),
            )
            st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("Generate or upload a crisis surface to proceed.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Run Adaptation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_adapt:
    st.header(f"Online Adaptation — {algo}")
    st.markdown(
        f"Runs **{inner_steps}** inner gradient steps of {algo} on the output MLP head, "
        "optimising the DupirePDE loss on the crisis surface. "
        "The spectral convolution core remains frozen."
    )

    run_adapt = st.button(f"Run {algo} Adaptation", use_container_width=True, type="primary")

    if run_adapt:
        if "crisis_iv" not in st.session_state:
            st.error("Generate a crisis surface first (Tab 1).")
        else:
            try:
                model, pde_loss_fn, device = _load_meta_fno()
                crisis_iv = st.session_state["crisis_iv"]

                T_t = torch.tensor(_MATS, dtype=torch.float32, device=device)
                K_t = torch.tensor(_STKS, dtype=torch.float32, device=device)
                T_mesh_t, K_mesh_t = torch.meshgrid(T_t, K_t, indexing="ij")
                crisis_t = torch.tensor(crisis_iv, dtype=torch.float32, device=device)

                # Build grid_inputs: (1, nK, nT, 3) — (K_norm, T_norm, IV)
                K_norm = K_mesh_t / 0.5
                T_norm = (T_mesh_t - T_t.mean()) / (T_t.std() + 1e-8)
                grid_inputs = torch.stack([K_norm, T_norm, crisis_t], dim=-1).unsqueeze(0)

                # Pre-adaptation surface
                with torch.no_grad():
                    core_feats = model.forward_core(grid_inputs)
                    iv_before  = model.forward_mlp(core_feats).squeeze(0)

                # PDE loss before
                with torch.no_grad():
                    loss_before = pde_loss_fn(
                        iv_before, K_mesh_t, T_mesh_t, crisis_t,
                        r=torch.tensor(0.05, device=device),
                        q=torch.tensor(0.0, device=device),
                    ).item()

                # Save original MLP weights
                adaptable = model.get_adaptable_parameters()
                original_weights = [p.data.clone() for p in adaptable]

                # Run inner adaptation
                optimizer = torch.optim.SGD(adaptable, lr=inner_lr)
                loss_history = []

                t0 = time.time()
                with st.spinner(f"Running {inner_steps} {algo} inner steps…"):
                    with torch.no_grad():
                        core_feats_fixed = model.forward_core(grid_inputs)

                    for step in range(inner_steps):
                        optimizer.zero_grad()
                        iv_pred = model.forward_mlp(core_feats_fixed)
                        loss = pde_loss_fn(
                            iv_pred.squeeze(0), K_mesh_t, T_mesh_t, crisis_t,
                            r=torch.tensor(0.05, device=device),
                            q=torch.tensor(0.0, device=device),
                        )
                        loss.backward()
                        optimizer.step()
                        loss_history.append(loss.item())

                elapsed_ms = (time.time() - t0) * 1000

                # Post-adaptation surface
                with torch.no_grad():
                    iv_after = model.forward_mlp(core_feats_fixed).squeeze(0)
                    loss_after = pde_loss_fn(
                        iv_after, K_mesh_t, T_mesh_t, crisis_t,
                        r=torch.tensor(0.05, device=device),
                        q=torch.tensor(0.0, device=device),
                    ).item()

                # Restore original weights
                for p, w in zip(adaptable, original_weights):
                    p.data.copy_(w)

                st.session_state["adapt_result"] = {
                    "iv_before":     iv_before.cpu().numpy(),
                    "iv_after":      iv_after.cpu().numpy(),
                    "crisis_iv":     crisis_iv,
                    "loss_before":   loss_before,
                    "loss_after":    loss_after,
                    "loss_history":  loss_history,
                    "elapsed_ms":    elapsed_ms,
                }
                st.success(
                    f"{algo} adaptation complete in **{elapsed_ms:.1f} ms** — "
                    f"PDE loss: {loss_before:.4e} → {loss_after:.4e}"
                )

            except Exception as exc:
                st.error(f"Adaptation failed: {exc}")

    if "adapt_result" in st.session_state:
        ar = st.session_state["adapt_result"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PDE Loss Before", f"{ar['loss_before']:.4e}")
        c2.metric("PDE Loss After",  f"{ar['loss_after']:.4e}")
        reduction = (ar['loss_before'] - ar['loss_after']) / (ar['loss_before'] + 1e-12)
        c3.metric("Loss Reduction",  f"{reduction*100:.1f}%")
        c4.metric("Adaptation Time", f"{ar['elapsed_ms']:.1f} ms")

        # PDE loss convergence
        st.subheader("PDE Loss Convergence During Inner Steps")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=list(range(1, len(ar["loss_history"]) + 1)),
            y=ar["loss_history"],
            mode="lines+markers",
            line=dict(color="#ff3366", width=2),
        ))
        fig_loss.update_layout(
            xaxis_title="Inner Step", yaxis_title="PDE Loss", yaxis_type="log",
            height=300, margin=dict(l=0, r=0, b=40, t=30),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # Surface correction heatmap
        st.subheader("Surface Correction Magnitude (After − Before) in bps")
        diff = (ar["iv_after"] - ar["iv_before"]) * 10000
        K_mesh, T_mesh = np.meshgrid(_STKS, _MATS)
        fig_corr = go.Figure(go.Surface(
            x=K_mesh, y=T_mesh, z=diff,
            colorscale="RdBu", opacity=0.9, showscale=True,
        ))
        fig_corr.update_layout(
            scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Maturity", zaxis_title="ΔIV (bps)"),
            height=400, margin=dict(l=0, r=0, b=0, t=35),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Heatmap table
        import pandas as pd
        df_diff = pd.DataFrame(
            diff,
            index=[f"T={t:.1f}" for t in _MATS],
            columns=[f"k={k:.2f}" for k in _STKS],
        )
        st.dataframe(
            df_diff.style.format("{:.2f}").background_gradient(cmap="RdBu", vmin=-50, vmax=50),
            use_container_width=True,
        )
    else:
        st.info("Run adaptation to see before/after comparison.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Adaptation Speed Sweep
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sweep:
    st.header("Adaptation Speed vs Inner Steps Sweep")
    st.markdown(
        "Sweeps over 1–8 inner gradient steps and measures wall-clock adaptation time "
        "and final PDE loss to find the optimal trade-off point."
    )

    run_sweep = st.button("Run Speed Sweep (1–8 steps)", use_container_width=True)

    if run_sweep:
        if "crisis_iv" not in st.session_state:
            st.error("Generate a crisis surface first.")
        else:
            try:
                model, pde_loss_fn, device = _load_meta_fno()
                crisis_iv = st.session_state["crisis_iv"]

                T_t = torch.tensor(_MATS, dtype=torch.float32, device=device)
                K_t = torch.tensor(_STKS, dtype=torch.float32, device=device)
                T_mesh_t, K_mesh_t = torch.meshgrid(T_t, K_t, indexing="ij")
                crisis_t = torch.tensor(crisis_iv, dtype=torch.float32, device=device)
                K_norm = K_mesh_t / 0.5
                T_norm = (T_mesh_t - T_t.mean()) / (T_t.std() + 1e-8)
                grid_inputs = torch.stack([K_norm, T_norm, crisis_t], dim=-1).unsqueeze(0)

                adaptable = model.get_adaptable_parameters()
                original_weights = [p.data.clone() for p in adaptable]

                steps_list = list(range(1, 9))
                times_list, losses_list = [], []

                bar = st.progress(0, text="Sweeping inner steps…")
                for si, n_inner in enumerate(steps_list):
                    # Restore original weights for each trial
                    for p, w in zip(adaptable, original_weights):
                        p.data.copy_(w)

                    optimizer = torch.optim.SGD(adaptable, lr=inner_lr)
                    with torch.no_grad():
                        core_feats = model.forward_core(grid_inputs)

                    t0 = time.time()
                    for _ in range(n_inner):
                        optimizer.zero_grad()
                        iv_pred = model.forward_mlp(core_feats)
                        loss = pde_loss_fn(
                            iv_pred.squeeze(0), K_mesh_t, T_mesh_t, crisis_t,
                            r=torch.tensor(0.05, device=device),
                            q=torch.tensor(0.0, device=device),
                        )
                        loss.backward()
                        optimizer.step()

                    elapsed = (time.time() - t0) * 1000
                    times_list.append(elapsed)
                    losses_list.append(loss.item())
                    bar.progress((si + 1) / len(steps_list), text=f"Step {n_inner}/8 — {elapsed:.1f} ms")

                # Restore weights
                for p, w in zip(adaptable, original_weights):
                    p.data.copy_(w)

                st.session_state["sweep_result"] = {
                    "steps": steps_list,
                    "times": times_list,
                    "losses": losses_list,
                }
                st.success("Sweep complete.")
            except Exception as exc:
                st.error(f"Sweep failed: {exc}")

    if "sweep_result" in st.session_state:
        import pandas as pd
        sr = st.session_state["sweep_result"]

        col_a, col_b = st.columns(2)
        with col_a:
            fig_time = go.Figure(go.Scatter(
                x=sr["steps"], y=sr["times"],
                mode="lines+markers", line=dict(color="#00d4ff", width=2),
            ))
            fig_time.update_layout(
                title="Adaptation Time vs Inner Steps",
                xaxis_title="Inner Steps", yaxis_title="Time (ms)",
                height=320, margin=dict(l=0, r=0, b=40, t=40),
            )
            st.plotly_chart(fig_time, use_container_width=True)

        with col_b:
            fig_loss_s = go.Figure(go.Scatter(
                x=sr["steps"], y=sr["losses"],
                mode="lines+markers", line=dict(color="#ff3366", width=2),
            ))
            fig_loss_s.update_layout(
                title="Final PDE Loss vs Inner Steps",
                xaxis_title="Inner Steps", yaxis_title="PDE Loss",
                yaxis_type="log",
                height=320, margin=dict(l=0, r=0, b=40, t=40),
            )
            st.plotly_chart(fig_loss_s, use_container_width=True)

        # Table
        df_sweep = pd.DataFrame({
            "Inner Steps":     sr["steps"],
            "Time (ms)":       [f"{t:.1f}" for t in sr["times"]],
            "Final PDE Loss":  [f"{l:.4e}" for l in sr["losses"]],
        })
        st.dataframe(df_sweep, use_container_width=True)
    else:
        st.info("Run the sweep to see the speed-accuracy trade-off.")
