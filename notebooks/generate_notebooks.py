"""
Generate all project notebooks programmatically.
Run from repo root: python notebooks/generate_notebooks.py
Each notebook is written as a dict and serialised to .ipynb JSON.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")

# ---------------------------------------------------------------------------
# Helper: build a clean notebook skeleton
# ---------------------------------------------------------------------------
def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.0"}
        },
        "cells": cells
    }

def md(src): return {"cell_type": "markdown", "id": "md", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "id": "code", "metadata": {},
                       "source": src, "outputs": [], "execution_count": None}

COMMON_SETUP = """\
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src") if os.path.basename(os.getcwd()) == "notebooks"
                else os.path.join(os.getcwd(), "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "font.family": "serif",
})
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
"""

# ---------------------------------------------------------------------------
# NB 01 — SPX Calibration
# ---------------------------------------------------------------------------
NB_01 = nb([
  md(["# Notebook 01 — SPX Implied-Volatility Surface Calibration\n\n",
      "Fetches a real SPX options chain from yfinance, constructs the 8×11 IV surface,\n",
      "and calibrates the Rough Heston (Lifted) model using the Gauss-Newton surrogate.\n\n",
      "**Runtime estimate:** 2–5 min (network + calibration)"]),
  code(COMMON_SETUP + """
from datetime import date
from market.spx_data import download_spx_chain, clean_chain, to_iv_surface, T_GRID, K_GRID
from calibrate import _load_normalizers
from calibrate_fast import calibrate_newton_h
from fno_model import MirrorPaddedFNO2d
from normalizers import ParameterNormalizer, IVSurfaceNormalizer
"""),
  md(["## 1. Load the FNO v3 Model and Normalizers"]),
  code("""\
model = MirrorPaddedFNO2d(param_dim=6).to(DEVICE)
model.load_state_dict(torch.load(
    "../artifacts/weights/fno_v3_final_prod.pth", map_location=DEVICE))
model.eval()
_load_normalizers("v3")
print("Model loaded — parameters:", sum(p.numel() for p in model.parameters()))
"""),
  md(["## 2. Download and Clean the SPX Options Chain"]),
  code("""\
SNAPSHOT_DATE = date(2024, 1, 2)   # change to any date with cached parquet
S0 = 4742.0   # SPX spot on 2024-01-02
R  = 0.053    # approx risk-free rate (early 2024)
Q  = 0.016    # approx SPX dividend yield
df_raw  = download_spx_chain(SNAPSHOT_DATE, cache=True)
df_clean = clean_chain(df_raw)
iv_surface = to_iv_surface(df_clean, S0, R, Q)   # shape (8, 11)

print(f"Options fetched:  {len(df_raw)}")
print(f"After cleaning:   {len(df_clean)}")
print(f"IV surface shape: {iv_surface.shape}")
print(f"ATM vol (T=0.6):  {iv_surface[2, 5]:.4f}  ({iv_surface[2, 5]*100:.2f}%)")
"""),
  md(["## 3. Plot the Market IV Surface"]),
  code("""\
from mpl_toolkits.mplot3d import Axes3D

K_abs = np.exp(K_GRID)          # strike / forward
T_mesh, K_mesh = np.meshgrid(T_GRID, K_abs, indexing="ij")

fig = plt.figure(figsize=(10, 5))
ax  = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(T_mesh, K_mesh, iv_surface * 100,
                        cmap="RdYlGn_r", alpha=0.9, linewidth=0)
ax.set_xlabel("Maturity T (years)")
ax.set_ylabel("Strike / Forward")
ax.set_zlabel("Implied Vol (%)")
ax.set_title(f"SPX IV Surface — {SNAPSHOT_DATE}")
fig.colorbar(surf, shrink=0.5, label="IV (%)")
plt.tight_layout()
plt.savefig("../tex/thesis/figures/fig_iv_surface.png", dpi=150, bbox_inches="tight")
plt.show()
"""),
  md(["## 4. Calibrate Rough Heston via Gauss-Newton\n\n",
      "Uses the FNO v3 surrogate and exact Jacobians (`torch.func.jacfwd`)."]),
  code("""\
result = calibrate_newton_h(model, iv_surface, T_GRID, K_GRID, max_iter=20, verbose=True)

print("\\nCalibration result:")
rmse_bps = np.sqrt(result['final_mse']) * 1e4
print(f"  RMSE : {rmse_bps:.1f} bps")
print(f"  iters: {result['n_iter']}")
for k in ['v0','zeta','lambda','sigma','rho','H']:
    print(f"  {k:6s} = {result[k]:.4f}")
"""),
  md(["## 5. Compare Market vs Model Surface"]),
  code("""\
pred_surface = result["iv_fitted"]              # (8, 11) in vol units
residuals    = (pred_surface - iv_surface) * 1e4   # in basis points

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, data, title, cmap in zip(
        axes,
        [iv_surface * 100, pred_surface * 100, residuals],
        ["Market IV (%)", "Model IV (%)", "Residuals (bps)"],
        ["RdYlGn_r", "RdYlGn_r", "RdBu_r"]):
    im = ax.imshow(data, aspect="auto", cmap=cmap,
                   extent=[K_GRID[0], K_GRID[-1], T_GRID[-1], T_GRID[0]])
    ax.set_title(title)
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Maturity (years)")
    plt.colorbar(im, ax=ax, fraction=0.04)

plt.suptitle(f"SPX {SNAPSHOT_DATE}  |  RMSE = {np.sqrt(result['final_mse'])*1e4:.1f} bps", fontsize=12)
plt.tight_layout()
plt.show()
print(f"Max |residual|: {np.abs(residuals).max():.1f} bps")
"""),
  md(["## 6. Convergence History"]),
  code("""\
history = result.get("history", [])
if history:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(range(1, len(history)+1), history, "o-", color="#2d6a9f")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE (bps) — log scale")
    ax.set_title("Gauss-Newton convergence")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()
"""),
])


# ---------------------------------------------------------------------------
# NB 02 — IV Surface Completion
# ---------------------------------------------------------------------------
NB_02 = nb([
  md(["# Notebook 02 — Arbitrage-Free IV Surface Completion\n\n",
      "Takes a sparse observed IV surface (missing strikes / maturities),\n",
      "fits SVI slices, enforces calendar-spread and butterfly no-arbitrage constraints,\n",
      "and produces a dense 8×11 surface suitable for calibration.\n\n",
      "**Runtime estimate:** 1–3 min"]),
  code(COMMON_SETUP + """
from market.spx_data import T_GRID, K_GRID, download_spx_chain, clean_chain, to_iv_surface
from arbitrage.surface_completion import (
    complete_surface, check_calendar_spread, check_butterfly,
    make_arbitrage_free, fit_svi_slice,
)
from datetime import date
"""),
  md(["## 1. Build a Realistic Sparse Surface\n\n",
      "We take the 2024-01-02 SPX surface and blank out 40% of quotes at random,\n",
      "simulating a realistic sparse options chain."]),
  code("""\
S0, R, Q = 4742.0, 0.053, 0.016   # SPX spot, risk-free rate, div yield 2024-01-02
rng = np.random.default_rng(0)
df_raw   = download_spx_chain(date(2024, 1, 2), cache=True)
df_clean = clean_chain(df_raw)
full_iv  = to_iv_surface(df_clean, S0, R, Q)   # (8, 11) ground truth

sparse_iv = full_iv.copy().astype(float)
mask = rng.random(sparse_iv.shape) < 0.40   # blank 40% of cells
sparse_iv[mask] = np.nan

n_observed = int((~mask).sum())
print(f"Observed cells : {n_observed} / {sparse_iv.size}")
print(f"Missing cells  : {mask.sum()} / {sparse_iv.size}")
"""),
  md(["## 2. Complete the Surface"]),
  code("""\
dense_iv = complete_surface(sparse_iv, mask, T_GRID, K_GRID)
print("Dense surface shape:", dense_iv.shape)
print("Any NaN remaining?  ", np.isnan(dense_iv).any())

# Enforce arbitrage constraints
af_iv = make_arbitrage_free(dense_iv, T_GRID, K_GRID, S=1.0)
print("Max butterfly violation (after):",
      np.abs(np.diff(np.diff(af_iv, axis=1), axis=1)).max() * 1e4, "bps²")
"""),
  md(["## 3. Check No-Arbitrage Conditions"]),
  code("""\
cal_viol_before = check_calendar_spread(dense_iv, T_GRID)
cal_viol_after  = check_calendar_spread(af_iv,    T_GRID)
but_viol_before = check_butterfly(dense_iv, K_GRID, T_GRID)
but_viol_after  = check_butterfly(af_iv,    K_GRID, T_GRID)

print(f"Calendar violations  — before: {(cal_viol_before > 0).sum()} cells  after: {(cal_viol_after > 0).sum()} cells")
print(f"Butterfly violations — before: {(but_viol_before > 0).sum()} cells  after: {(but_viol_after > 0).sum()} cells")
"""),
  md(["## 4. Visualise: Sparse vs Completed vs Ground Truth"]),
  code("""\
titles = ["Sparse (observed)", "Completed + AF", "Ground Truth (SPX)"]
surfaces = [sparse_iv, af_iv, full_iv]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
vmin, vmax = 0.10, 0.40
for ax, iv, title in zip(axes, surfaces, titles):
    im = ax.imshow(iv * 100, aspect="auto", cmap="RdYlGn_r",
                   vmin=vmin*100, vmax=vmax*100,
                   extent=[K_GRID[0], K_GRID[-1], T_GRID[-1], T_GRID[0]])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Log-moneyness k")
    ax.set_ylabel("Maturity T (yr)")
    plt.colorbar(im, ax=ax, fraction=0.04, label="IV (%)")
plt.tight_layout()
plt.show()
"""),
  md(["## 5. Per-Slice SVI Fit"]),
  code("""\
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, t_idx in zip(axes.flat, range(8)):
    T   = T_GRID[t_idx]
    k   = K_GRID
    obs = sparse_iv[t_idx]
    mask_t = ~np.isnan(obs)
    svi = fit_svi_slice(k[mask_t], obs[mask_t]**2 * T) if mask_t.sum() >= 3 else None

    ax.plot(k, full_iv[t_idx]*100,  "k-",  lw=1.5, label="True")
    ax.plot(k[mask_t], obs[mask_t]*100, "o", ms=4, label="Observed")
    ax.plot(k, af_iv[t_idx]*100,   "r--", lw=1.5, label="Completed")
    if svi:
        w = svi["a"] + svi["b"]*(svi["rho"]*(k-svi["m"]) +
            np.sqrt((k-svi["m"])**2 + svi["sigma"]**2))
        ax.plot(k, np.sqrt(np.maximum(w,0)/T)*100, "g:", lw=1.5, label="SVI")
    ax.set_title(f"T = {T:.1f} yr", fontsize=9)
    ax.set_ylim(8, 45)
    if t_idx == 0:
        ax.legend(fontsize=7)
plt.suptitle("SVI slice fits — Sparse → Arbitrage-Free Surface")
plt.tight_layout()
plt.show()
"""),
])


# ---------------------------------------------------------------------------
# NB 03 — VIX Term Structure Analysis
# ---------------------------------------------------------------------------
NB_03 = nb([
  md(["# Notebook 03 — VIX Term Structure under Rough Heston\n\n",
      "Computes the model VIX, variance swap rates, and VIX futures curve from\n",
      "calibrated Rough Heston parameters. Compares against 4 historical CBOE dates.\n\n",
      "**Runtime estimate:** 5–10 min (ODE integration + yfinance)"]),
  code(COMMON_SETUP + """
from market.vix_pricing import (
    model_vix, vix_futures_curve, model_variance_swap_rate, download_vix_futures
)
from market.variance_swaps import variance_swap_rate, variance_term_structure
from market.vix_futures import fetch_vix_futures
"""),
  md(["## 1. VIX vs Initial Variance v₀"]),
  code("""\
base = dict(kappa=1.5, theta=0.08, sigma=0.8, rho=-0.34, H=0.08)
v0_range = np.linspace(0.01, 0.25, 60)
vix_vals = [model_vix(**base, v0=v0, t=1/12) for v0 in v0_range]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.sqrt(v0_range)*100, vix_vals, color="#2d6a9f", lw=2)
ax.set_xlabel("Spot vol √v₀ (%)")
ax.set_ylabel("Model VIX")
ax.set_title("Model VIX vs initial variance v₀")
plt.tight_layout(); plt.show()
"""),
  md(["## 2. Variance Swap Term Structure"]),
  code("""\
T_range = np.linspace(0.08, 2.0, 50)
for v0, label in [(0.04, "v₀=0.04 (16%)"), (0.10, "v₀=0.10 (32%)"), (0.20, "v₀=0.20 (45%)")]:
    rates = [model_variance_swap_rate(**base, v0=v0, T=T) for T in T_range]
    plt.plot(T_range, np.sqrt(rates)*100, label=label)
plt.xlabel("Maturity T (years)"); plt.ylabel("Fair vol-swap strike (%)")
plt.title("Variance swap term structure"); plt.legend(); plt.tight_layout(); plt.show()
"""),
  md(["## 3. Contango vs Backwardation"]),
  code("""\
maturities = np.linspace(0.05, 2.0, 40)
scenarios = [
    dict(kappa=1.5, theta=0.08, sigma=0.8, rho=-0.34, v0=0.04, H=0.08, label="Low vol (contango)"),
    dict(kappa=1.5, theta=0.08, sigma=0.8, rho=-0.34, v0=0.18, H=0.08, label="High vol (backwardation)"),
    dict(kappa=1.5, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08, label="Flat"),
]
fig, ax = plt.subplots(figsize=(9, 4))
for sc in scenarios:
    label = sc.pop("label")
    curve = vix_futures_curve(**sc, maturities=maturities)
    ax.plot(maturities, curve, lw=2, label=label)
    sc["label"] = label
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("VIX futures price")
ax.set_title("Rough Heston VIX futures curve"); ax.legend(); plt.tight_layout(); plt.show()
"""),
  md(["## 4. Model vs Historical CBOE VIX Futures\n\n",
      "Loads saved calibration results for 4 stress dates and plots model vs market."]),
  code("""\
import json
dates_info = [
    ("2020-03-16", "COVID crash"),
    ("2022-01-24", "Fed hike"),
    ("2024-01-02", "Low vol"),
    ("2024-08-05", "VIX spike"),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (date_str, title) in zip(axes.flat, dates_info):
    vix_path   = f"../results/vix_term_structure/{date_str}.json"
    calib_path = f"../results/spx_calibration/{date_str}.json"
    if not os.path.exists(vix_path):
        ax.set_title(f"{date_str} — no data"); continue
    vix_list = json.load(open(vix_path))
    t_mkt = np.array([r["tenor_months"] for r in vix_list])
    v_mkt = np.array([r["settle_vix"]   for r in vix_list])
    ax.plot(t_mkt, v_mkt, "o--", ms=5, label="CBOE VIX")
    if os.path.exists(calib_path):
        cal = json.load(open(calib_path))
        p = cal.get("params", {})
        if all(k in p for k in ["kappa","theta","sigma","rho","v0","H"]):
            mats = np.linspace(t_mkt.min(), t_mkt.max(), 50)
            curve = vix_futures_curve(**p, maturities=mats)
            ax.plot(mats, curve, "-", lw=2, label="Rough Heston")
    ax.set_title(f"{date_str} — {title}"); ax.set_xlabel("Maturity (months)")
    ax.set_ylabel("VIX"); ax.legend(fontsize=8)
plt.suptitle("Rough Heston VIX Futures Curve — Historical Dates")
plt.tight_layout(); plt.show()
"""),
  md(["## 5. Hurst Exponent Sensitivity"]),
  code("""\
H_vals = [0.05, 0.08, 0.10, 0.12, 0.15]
mats   = np.linspace(0.08, 1.5, 40)
base   = dict(kappa=1.5, theta=0.08, sigma=0.8, rho=-0.34, v0=0.08)
fig, ax = plt.subplots(figsize=(9, 4))
for H in H_vals:
    curve = vix_futures_curve(**base, H=H, maturities=mats)
    ax.plot(mats, curve, lw=2, label=f"H={H:.2f}")
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("VIX futures")
ax.set_title("VIX term structure sensitivity to Hurst exponent H")
ax.legend(); plt.tight_layout(); plt.show()
"""),
])


# ---------------------------------------------------------------------------
# NB 04 — Portfolio Greeks and P&L Attribution
# ---------------------------------------------------------------------------
NB_04 = nb([
  md(["# Notebook 04 — Portfolio Greeks and P&L Attribution\n\n",
      "Computes delta, gamma, vega, vanna, and volga for a sample SPX options\n",
      "portfolio using the FNO surrogate and finite-difference bumps.\n",
      "Then runs a full Taylor-expansion P&L attribution for a market shock.\n\n",
      "**Runtime estimate:** 2–5 min"]),
  code(COMMON_SETUP + """
from greeks.portfolio_greeks import (
    bs_greeks, fno_surface_greeks, portfolio_greeks,
    MATURITIES, STRIKES
)
from greeks.pnl_attribution import pnl_attribution
from fno_model import MirrorPaddedFNO2d
from normalizers import ParameterNormalizer, IVSurfaceNormalizer
from calibrate import _load_normalizers
"""),
  md(["## 1. Load Model"]),
  code("""\
model = MirrorPaddedFNO2d(param_dim=6).to(DEVICE)
model.load_state_dict(torch.load(
    "../artifacts/weights/fno_v3_final_prod.pth", map_location=DEVICE))
model.eval()
_load_normalizers("v3")
import calibrate as _cal
pn = _cal._param_norm
yn = _cal._iv_norm
print("Model ready")
"""),
  md(["## 2. Black-Scholes Greeks Reference"]),
  code("""\
S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
g = bs_greeks(S, K, T, r, sigma)
print("Black-Scholes Greeks (ATM, T=0.5yr, σ=20%)")
for name, val in g.items():
    print(f"  {name:8s} = {val:.6f}")
"""),
  md(["## 3. FNO Greeks Surface"]),
  code("""\
# Calibrated Rough Heston parameters for 2024-01-02
theta = np.array([2.0, 0.04, 0.5, -0.7, 0.04, 0.10], dtype=np.float32)
greeks_surface = fno_surface_greeks(model, theta, pn, yn, S=4800.0)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
greek_names = ["delta", "gamma", "vega", "theta", "vanna", "volga"]
K_abs = np.exp(STRIKES)
T_mesh, K_mesh = np.meshgrid(MATURITIES, K_abs, indexing="ij")

for ax, name in zip(axes.flat, greek_names):
    if name not in greeks_surface:
        ax.set_visible(False); continue
    data = greeks_surface[name]
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r",
                   extent=[STRIKES[0], STRIKES[-1], MATURITIES[-1], MATURITIES[0]])
    ax.set_title(name.capitalize())
    ax.set_xlabel("Log-moneyness"); ax.set_ylabel("Maturity (yr)")
    plt.colorbar(im, ax=ax, fraction=0.04)
plt.suptitle("FNO Greeks Surface — Rough Heston")
plt.tight_layout(); plt.show()
"""),
  md(["## 4. Portfolio Greeks"]),
  code("""\
S_spot = 4800.0
positions = [
    {"K": 4800.0, "T": 0.5,  "type": "call", "quantity":  10.0, "notional": 100.0},
    {"K": 4700.0, "T": 0.5,  "type": "put",  "quantity": -20.0, "notional": 100.0},
    {"K": 4900.0, "T": 1.0,  "type": "call", "quantity":   5.0, "notional": 100.0},
    {"K": 4500.0, "T": 1.5,  "type": "put",  "quantity":  15.0, "notional": 100.0},
    {"K": 5000.0, "T": 0.25, "type": "call", "quantity":  -8.0, "notional": 100.0},
]
port_greeks = portfolio_greeks(positions, model, theta, pn, yn, S=S_spot)

print("Portfolio Greeks")
print(f"  Total Delta : {port_greeks['total_delta']:+.4f}")
print(f"  Total Gamma : {port_greeks['total_gamma']:+.6f}")
vega_total = port_greeks.get('vega_bucket', np.zeros(8)).sum()
# vega_bucket is $/unit-vol (per 1.0 = 100% vol change); divide by 100 for $/1%-vol
print(f"  Total Vega  : {vega_total/100:+.2f}  ($ per 1% vol move)")
print(f"  Total Vanna : {port_greeks['total_vanna']:+.4f}")
print(f"  Total Volga : {port_greeks['total_volga']/1e6:+.3f}M  ($ per unit-vol\u00b2 \u00d7 notional)")
print(f"  Hedge (ES contracts): {port_greeks.get('hedge_contracts', 'N/A')}")
"""),
  md(["## 5. Vega Bucket (by Maturity)"]),
  code("""\
vega_bucket = port_greeks.get("vega_bucket", np.zeros(8))
fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#c0392b" if v < 0 else "#27ae60" for v in vega_bucket]
ax.bar(MATURITIES, vega_bucket, width=0.15, color=colors)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Vega")
ax.set_title("Portfolio Vega Bucket")
plt.tight_layout(); plt.show()
"""),
  md(["## 6. P&L Attribution — Market Shock\n\n",
      "Simulate a -1% spot move and +1 vol point shift."]),
  code("""\
d_iv = np.full((8, 11), 0.01)   # uniform +1% vol shift (d_sigma = 0.01)

# pnl_attribution reads greeks from position dicts.
# Positions above have no greek values → enrich each with BS greeks
# interpolated from the FNO-predicted IV surface.
from greeks.portfolio_greeks import MATURITIES, STRIKES
iv_surf = greeks_surface['iv_surface']   # (8, 11) from fno_surface_greeks

enriched = []
for pos in positions:
    k_pos = np.log(pos['K'] / S_spot)
    t_idx = int(np.argmin(np.abs(np.array(MATURITIES) - pos['T'])))
    k_idx = int(np.argmin(np.abs(np.array(STRIKES) - k_pos)))
    sigma_pos = float(iv_surf[t_idx, k_idx])
    g = bs_greeks(S_spot, pos['K'], pos['T'], 0.05, sigma_pos,
                  option_type=pos['type'])
    enriched.append({**pos,
                     'delta': g['delta'], 'gamma': g['gamma'],
                     'vega':  g['vega'],  'vanna': g['vanna'],
                     'volga': g['volga']})

result = pnl_attribution(
    portfolio=enriched,
    dS=-48.0,                   # -1% of 4800
    d_iv_surface=d_iv,
    S=S_spot,
)
components = result["breakdown"]  # {delta_pnl, gamma_pnl, vega_pnl, vanna_pnl, volga_pnl}
total = result.get("explained_pnl", sum(components.values()))
print("P&L Attribution (dS=-48, dvol=+1%)")
max_abs = max((abs(v) for v in components.values()), default=1) or 1
for name, val in components.items():
    bar = "#" * int(abs(val) / max_abs * 30 + 1)
    print(f"  {name:15s} = {val:+10.2f}  {bar}")
print(f"  {'Total':15s} = {total:+10.2f}")
"""),
  md(["## 7. Waterfall Chart"]),
  code("""\
labels = list(components.keys())
vals   = [components[k] for k in labels]
labels_clean = [l.replace("_pnl","").capitalize() for l in labels]

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#27ae60" if v >= 0 else "#c0392b" for v in vals]
ax.bar(labels_clean, vals, color=colors)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("P&L ($)")
ax.set_title("P&L Attribution — dS=−1%, dVol=+1%")
plt.tight_layout(); plt.show()
"""),
])


# ---------------------------------------------------------------------------
# NB 05 — Crypto Derivatives: Deribit BTC/ETH Calibration (REWRITE)
# ---------------------------------------------------------------------------
NB_05 = nb([
  md(["# Notebook 05 — Crypto Derivatives: Deribit BTC/ETH Calibration\n\n",
      "Fetches a live BTC or ETH option snapshot from Deribit REST API,\n",
      "builds an IV surface, and calibrates the Rough Heston model.\n\n",
      "**Runtime estimate:** 1–3 min (live network required)"]),
  code(COMMON_SETUP + """

from market.deribit_data import fetch_option_snapshot, build_iv_surface, calibrate_crypto
from fno_model import MirrorPaddedFNO2d
from calibrate import _load_normalizers
"""),
  md(["## 1. Fetch Live Option Snapshot from Deribit"]),
  code("""\
CURRENCY = "BTC"   # or "ETH"

# Jupyter already runs an asyncio event loop — use 'await' directly
df = await fetch_option_snapshot(CURRENCY)
print(f"Fetched {len(df)} live {CURRENCY} options from Deribit")
print(f"Columns: {df.columns.tolist()}")
print(df[["instrument_name","expiry","strike","mark_iv","log_moneyness"]].head(10).to_string())
"""),
  md(["## 2. Build the IV Surface"]),
  code("""\
from market.spx_data import T_GRID, K_GRID
iv_surface = build_iv_surface(df, currency=CURRENCY)   # shape (8, 11)
T_nodes, K_nodes = T_GRID, K_GRID
print(f"Maturities (yr): {np.round(T_nodes, 3)}")
print(f"Log-moneyness:   {np.round(K_nodes, 2)}")
print(f"ATM vol range:   {iv_surface[:, 5].min()*100:.1f}% — "
      f"{iv_surface[:, 5].max()*100:.1f}%")
"""),
  md(["## 3. Plot the Live IV Surface"]),
  code("""\
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(iv_surface * 100, aspect="auto", cmap="RdYlGn_r",
               extent=[K_nodes[0], K_nodes[-1], T_nodes[-1], T_nodes[0]])
ax.set_xlabel("Log-moneyness"); ax.set_ylabel("Maturity (yr)")
ax.set_title(f"Live {CURRENCY} IV Surface (Deribit)")
plt.colorbar(im, label="Implied Vol (%)"); plt.tight_layout(); plt.show()
"""),
  md(["## 4. Calibrate Rough Heston to Crypto\n\n",
      "Uses the dedicated `calibrate_crypto()` function which handles the\n",
      "parameter range adjustments needed for BTC/ETH (higher vol-of-vol, wider grid)."]),
  code("""\
# calibrate_crypto handles model loading and normalizer setup internally
result = calibrate_crypto(currency=CURRENCY)
print(f"\\nCalibration RMSE: {result['rmse_bps']:.1f} bps")
print(f"Converged:        {result.get('converged', 'N/A')}")
print(f"Params clipped:   {result.get('params_clipped', False)}")
for name in ['v0', 'sigma', 'rho', 'zeta', 'lambda', 'H']:
    if name in result:
        print(f"  {name:6s} = {result[name]:.4f}")
"""),
  md(["## 5. Market vs Model Overlay"]),
  code("""\
pred = result.get("iv_fitted")   # (8,11) model-predicted surface from calibrate_newton_h
if pred is not None and pred.shape == iv_surface.shape:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title in zip(axes,
            [iv_surface*100, pred*100],
            [f"{CURRENCY} Market IV (%)", "Rough Heston Model IV (%)"]):
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                       extent=[K_nodes[0], K_nodes[-1], T_nodes[-1], T_nodes[0]])
        ax.set_title(title); ax.set_xlabel("Log-moneyness")
        ax.set_ylabel("Maturity (yr)")
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout(); plt.show()
    residuals = (pred - iv_surface) * 1e4
    print(f"Max |residual|: {np.abs(residuals).max():.1f} bps")
else:
    print("Model surface not available — check calibrate_crypto return value")
"""),
])


# ---------------------------------------------------------------------------
# NB 06 — Batch Calibration and Hurst Dynamics (NEW — replaces NB10)
# ---------------------------------------------------------------------------
NB_06 = nb([
  md(["# Notebook 06 — Batch Calibration and Hurst Exponent Dynamics\n\n",
      "Loads real saved calibration results from `results/hurst_dynamics/SPX_hurst_study.json`,\n",
      "plots the evolution of all Rough Heston parameters, and analyses the Hurst exponent H.\n",
      "If no saved results exist, runs a short demo calibration over 5 dates.\n\n",
      "**Runtime estimate:** 30 sec (from saved JSON) or 10–20 min (live calibration)"]),
  code(COMMON_SETUP + """
import json
from pathlib import Path
from calibration.batch_calibration import (
    calibrate_batch, results_to_dataframe, load_results, save_results
)
RESULTS_PATH = "../results/hurst_dynamics/SPX_hurst_study.json"
"""),
  md(["## 1. Load or Generate Calibration Results"]),
  code("""\
p = Path(RESULTS_PATH)
if p.exists():
    results = load_results(RESULTS_PATH)
    df = results_to_dataframe(results)
    print(f"Loaded {len(df)} calibration results from saved file.")
else:
    print("No saved results found. Running short demo (5 dates) ...")
    # Run batch calibration over 5 recent dates
    demo_dates = ["2024-01-02", "2024-01-03", "2024-01-04",
                  "2024-01-05", "2024-01-08"]
    results = calibrate_batch(demo_dates, currency="SPX", device="auto", verbose=True)
    save_results(results, RESULTS_PATH)
    df = results_to_dataframe(results)
    print(f"Calibrated {len(df)} dates, saved to {RESULTS_PATH}")

print(df[["date","kappa","theta","sigma","rho","v0","H","rmse_bps","converged"]].head())
"""),
  md(["## 2. Summary Statistics"]),
  code("""\
conv = df[df["converged"]]
print(f"Converged:     {len(conv)} / {len(df)}  ({100*len(conv)/len(df):.1f}%)")
print(f"Median RMSE:   {conv['rmse_bps'].median():.1f} bps")
print(f"\\nParameter statistics (converged dates):")
for col in ["kappa","theta","sigma","rho","v0","H"]:
    print(f"  {col:6s}  mean={conv[col].mean():.4f}  std={conv[col].std():.4f}"
          f"  min={conv[col].min():.4f}  max={conv[col].max():.4f}")
"""),
  md(["## 3. Parameter Time Series"]),
  code("""\
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    except Exception:
        pass

params = ["kappa","theta","sigma","rho","v0","H"]
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
for ax, param in zip(axes.flat, params):
    ax.plot(range(len(conv)), conv[param], lw=1.5, color="#2d6a9f")
    ax.axhline(conv[param].mean(), color="r", ls="--", lw=0.8, label="Mean")
    ax.set_title(param, fontsize=11); ax.set_ylabel(param)
    ax.set_xlabel("Date index"); ax.legend(fontsize=8)
plt.suptitle("Rough Heston Parameters — Historical Evolution", fontsize=13)
plt.tight_layout(); plt.show()
"""),
  md(["## 4. Hurst Exponent H — Detailed Analysis"]),
  code("""\
H_series = conv["H"].values
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
axes[0].hist(H_series, bins=25, color="#2d6a9f", edgecolor="white")
axes[0].axvline(H_series.mean(), color="r", ls="--", lw=1.5, label=f"Mean={H_series.mean():.3f}")
axes[0].axvline(0.5, color="k", ls=":", lw=1, label="H=0.5 (BM)")
axes[0].set_xlabel("H"); axes[0].set_ylabel("Count")
axes[0].set_title("Hurst Exponent Distribution"); axes[0].legend()

# Time series with rolling mean
axes[1].plot(range(len(H_series)), H_series, lw=1.2, alpha=0.7, color="#2d6a9f", label="H")
if len(H_series) >= 10:
    roll = pd.Series(H_series).rolling(10, min_periods=1).mean()
    axes[1].plot(range(len(H_series)), roll, "r-", lw=2, label="10-day MA")
axes[1].axhline(0.5, color="k", ls=":", lw=0.8)
axes[1].set_xlabel("Date index"); axes[1].set_ylabel("H")
axes[1].set_title("Hurst Exponent Over Time"); axes[1].legend()

# H vs RMSE
sc = axes[2].scatter(H_series, conv["rmse_bps"], alpha=0.5, s=20, c=conv["sigma"], cmap="viridis")
axes[2].set_xlabel("H"); axes[2].set_ylabel("RMSE (bps)")
axes[2].set_title("H vs Calibration RMSE")
plt.colorbar(sc, ax=axes[2], label="sigma (vol-of-vol)")

plt.suptitle(f"Hurst Exponent Analysis — n={len(H_series)} dates")
plt.tight_layout(); plt.show()
print(f"H < 0.5 in {100*(H_series < 0.5).mean():.1f}% of dates (rough regime)")
"""),
  md(["## 5. RMSE Distribution"]),
  code("""\
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(conv["rmse_bps"], bins=30, color="#27ae60", edgecolor="white")
ax.axvline(100, color="r", ls="--", lw=1.5, label="100 bps threshold")
ax.set_xlabel("Calibration RMSE (bps)"); ax.set_ylabel("Count")
ax.set_title("Calibration RMSE Distribution"); ax.legend()
plt.tight_layout(); plt.show()
under100 = (conv["rmse_bps"] < 100).mean() * 100
print(f"{under100:.1f}% of calibrations achieved RMSE < 100 bps")
"""),
])


# ---------------------------------------------------------------------------
# NB 07 — Joint SPX + VIX Calibration (replaces NB09)
# ---------------------------------------------------------------------------
NB_07 = nb([
  md(["# Notebook 07 — Joint SPX + VIX Calibration\n\n",
      "Calibrates Rough Heston parameters to match *both* the SPX IV surface\n",
      "and the VIX futures term structure simultaneously using a combined loss.\n",
      "Analyses weight sensitivity and parameter stability.\n\n",
      "**Runtime estimate:** 10–20 min (grid search + yfinance)"]),
  code(COMMON_SETUP + """
import json
from datetime import date
from market.spx_data import download_spx_chain, clean_chain, to_iv_surface
from market.vix_futures import fetch_vix_futures
from market.vix_pricing import vix_futures_curve
from calibration.joint_calibration import calibrate_joint, calibrate_spx_only
"""),
  md(["## 1. Load Market Data"]),
  code("""\
VAL_DATE = date(2024, 1, 2)

df_spx  = download_spx_chain(VAL_DATE, cache=True)
df_clean = clean_chain(df_spx)
S0, R, Q = 4742.0, 0.053, 0.016   # SPX spot, risk-free rate, div yield
iv_obs   = to_iv_surface(df_clean, S0, R, Q)
print(f"SPX IV surface shape: {iv_obs.shape}")
print(f"ATM vol (6M): {iv_obs[2, 5]*100:.2f}%")

vix_df = fetch_vix_futures(VAL_DATE)
print(f"\\nVIX futures ({len(vix_df)} contracts):")
print(vix_df[["expiry","settle_vix"]].to_string(index=False))
"""),
  md(["## 2. SPX-Only Baseline Calibration"]),
  code("""\
spx_result = calibrate_spx_only(iv_obs)
print("SPX-only calibration:")
rmse_key = next((k for k in ['spx_rmse_bps','rmse_bps','rmse'] if k in spx_result), None)
if rmse_key: print(f"  RMSE: {spx_result[rmse_key]:.1f} bps")
params_raw = spx_result.get('params', spx_result)
pnames = ['kappa','theta','sigma','rho','v0','H']
for k in pnames:
    if k in params_raw: print(f"  {k:6s} = {params_raw[k]:.4f}")
"""),
  md(["## 3. Joint SPX + VIX Calibration"]),
  code("""\
vix_spot = float(vix_df["settle_vix"].iloc[0]) if len(vix_df) else 13.5
joint_result = calibrate_joint(iv_obs, vix_spot, weights=(0.7, 0.3))
print("Joint SPX+VIX calibration:")
for key in ['spx_rmse_bps','rmse_bps','vix_error','total_loss','converged']:
    if key in joint_result: print(f"  {key}: {joint_result[key]}")
params_j = joint_result.get('params', joint_result)
for k in ['kappa','theta','sigma','rho','v0','H']:
    if k in params_j: print(f"  {k:6s} = {params_j[k]:.4f}")
"""),
  md(["## 4. Weight Sensitivity Analysis\n\n",
      "Show how the calibrated parameters shift as we put more/less weight on VIX."]),
  code("""\
weight_schemes = [
    (1.0, 0.0, "SPX only"),
    (1.0, 0.1, "SPX-heavy"),
    (1.0, 0.5, "Balanced"),
    (0.5, 1.0, "VIX-heavy"),
]
records = []
for w_spx, w_vix, label in weight_schemes:
    r = calibrate_joint(iv_obs, vix_spot, weights=(w_spx, w_vix))
    spx_r = r.get('spx_rmse_bps', r.get('rmse_bps', float('nan')))
    vix_r = r.get('vix_error', r.get('vix_rmse', float('nan')))
    rec = {"label": label, "w_vix": w_vix,
           "spx_rmse": spx_r, "vix_rmse": vix_r}
    rec.update(r.get("params", {}))
    records.append(rec)
    print(f"{label:12s}  SPX={spx_r:.1f}bps  VIX={vix_r:.4f}")

df_ws = pd.DataFrame(records)
"""),
  md(["## 5. Pareto Frontier: SPX RMSE vs VIX RMSE"]),
  code("""\
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(df_ws["spx_rmse"], df_ws["vix_rmse"],
                c=df_ws["w_vix"], cmap="coolwarm", s=100, zorder=3)
for _, row in df_ws.iterrows():
    ax.annotate(row["label"], (row["spx_rmse"], row["vix_rmse"]),
                fontsize=8, textcoords="offset points", xytext=(6, 4))
plt.colorbar(sc, label="VIX weight w_vix")
ax.set_xlabel("SPX calibration RMSE (bps)")
ax.set_ylabel("VIX RMSE")
ax.set_title(f"Joint Calibration Pareto Frontier — {VAL_DATE}")
plt.tight_layout(); plt.show()
"""),
  md(["## 6. Model VIX Curve vs Market (Best Joint Fit)"]),
  code("""\
PKEYS = ['kappa','theta','sigma','rho','v0','H']
best_params = {k: joint_result[k] for k in PKEYS}
mats = np.linspace(1/12, 1.5, 50)
model_curve = vix_futures_curve(**best_params, maturities=mats)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(mats, model_curve, lw=2, label="Rough Heston (joint fit)")
if len(vix_df):
    t_mkt = np.arange(1, len(vix_df)+1) / 12
    ax.plot(t_mkt, vix_df["settle_vix"].values, "o--", ms=6, label="CBOE VIX futures")
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("VIX")
ax.set_title(f"Model vs Market VIX Futures — {VAL_DATE}")
ax.legend(); plt.tight_layout(); plt.show()
"""),
  md(["## 7. SPX Residual Heatmap (Joint vs SPX-Only)"]),
  code("""\
# Reuse the model weights — normalizers already loaded by calibrate_joint above
from market.spx_data import T_GRID, K_GRID
from calibrate import _fno_predict_real_iv, _make_spatial_input
import calibrate as _cal
from fno_model import MirrorPaddedFNO2d
model7 = MirrorPaddedFNO2d(param_dim=6).to(DEVICE)
model7.load_state_dict(torch.load("../artifacts/weights/fno_v3_final_prod.pth", map_location=DEVICE))
model7.eval()

PKEYS = ['kappa','theta','sigma','rho','v0','H']

def get_pred(params_dict):
    pn, yn = _cal._param_norm, _cal._iv_norm
    p = np.array([params_dict[k] for k in PKEYS], dtype=np.float32)
    # _IdentityParamNorm has no transform; ParameterNormalizer does
    if hasattr(pn, 'transform'):
        p_norm = pn.transform(p.reshape(1, -1)).flatten().astype(np.float32)
    else:
        p_norm = p
    p_t = torch.tensor(p_norm).unsqueeze(0).to(DEVICE)
    spatial = _make_spatial_input(T_GRID, K_GRID, device=DEVICE)
    with torch.no_grad():
        iv_n = _fno_predict_real_iv(model7, p_t, spatial)
    raw = iv_n.squeeze().cpu().numpy()
    return yn.inverse_transform(raw) if hasattr(yn, 'inverse_transform') else raw

pred_joint = get_pred({k: joint_result[k] for k in PKEYS})
pred_spx   = get_pred({k: spx_result[k]   for k in PKEYS})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, pred, title in zip(axes,
        [pred_spx, pred_joint],
        ["SPX-only residuals (bps)", "Joint residuals (bps)"]):
    resid = (pred - iv_obs) * 1e4
    im = ax.imshow(resid, aspect="auto", cmap="RdBu_r", vmin=-30, vmax=30,
                   extent=[K_GRID[0], K_GRID[-1], T_GRID[-1], T_GRID[0]])
    ax.set_title(title); ax.set_xlabel("Log-moneyness"); ax.set_ylabel("Maturity (yr)")
    plt.colorbar(im, ax=ax, fraction=0.04, label="bps")
plt.suptitle("Calibration Residuals: SPX-only vs Joint")
plt.tight_layout(); plt.show()
"""),
])


# ---------------------------------------------------------------------------
# Save all notebooks
# ---------------------------------------------------------------------------
NOTEBOOKS = {
    "01_spx_calibration.ipynb":       NB_01,
    "02_surface_completion.ipynb":    NB_02,
    "03_vix_analysis.ipynb":          NB_03,
    "04_greeks_portfolio.ipynb":      NB_04,
    "05_crypto_calibration.ipynb":    NB_05,
    "06_batch_calibration.ipynb":     NB_06,
    "07_joint_calibration.ipynb":     NB_07,
}

# Old notebooks to delete
DELETE = [
    "09_vix_joint_calibration.ipynb",
    "10_hurst_dynamics.ipynb",
]

if __name__ == "__main__":
    written = []
    for fname, notebook in NOTEBOOKS.items():
        path = os.path.join(NB_DIR, fname)
        # Give each cell a unique id
        for i, cell in enumerate(notebook["cells"]):
            cell["id"] = f"cell-{i:04d}"
        with open(path, "w") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        size = os.path.getsize(path)
        print(f"  Written: {fname}  ({size:,} bytes)")
        written.append(fname)

    deleted = []
    for fname in DELETE:
        path = os.path.join(NB_DIR, fname)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Deleted: {fname}")
            deleted.append(fname)
        else:
            print(f"  Skip (not found): {fname}")

    print(f"\nDone. Written {len(written)} notebooks, deleted {len(deleted)}.")
    print("Notebooks directory now contains:")
    for f in sorted(os.listdir(NB_DIR)):
        if f.endswith(".ipynb"):
            sz = os.path.getsize(os.path.join(NB_DIR, f))
            print(f"  {f}  ({sz:,} bytes)")
