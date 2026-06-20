# Project Context — Deep Rough Heston Calibration
**Last updated:** 2026-06-19 · **Status:** Thesis complete (51 pp), **Tier 1 COMPLETE** — all 5 market-data extensions implemented, 408 tests passing

---

## 🎯 Project Goal
GPU-accelerated calibration of the **Rough Heston** stochastic volatility model using a
**FiLM-conditioned Fourier Neural Operator (FNO)** surrogate. The FNO maps model parameters
`(κ, θ, σ, ρ, V₀, H)` to a full implied-volatility surface `(8T × 11K)`, replacing slow
Monte-Carlo / Fourier-COS pricing at calibration time.

**Academic context:** Master's thesis, МФТИ ФПМИ, кафедра БИТ (Банковских информационных технологий).
Title: *«Нейронные сети в ценообразовании производных финансовых инструментов»*

**Next phase:** Tier 1 real-market extension — SPX, VIX, Deribit crypto, Greeks at scale.
Deep research prompts sent to Gemini; results pending.

---

## 📁 Directory Structure (post-cleanup 2026-06-18)

```
derivatives/
├── src/                              ← ALL production Python
│   ├── fno_model.py                  ← MirrorPaddedFNO2d (v2/v3), FiLM conditioning
│   ├── normalizers.py                ← ParameterNormalizer, IVSurfaceNormalizer
│   ├── pricing_engine.py             ← GPU Fourier-COS pricer (Bernstein lifting)
│   ├── pricing_engine_gpu.py         ← GPU batch pricer (used by dataset generators)
│   ├── calibrate.py                  ← L-BFGS + FIM + reparameterization (3D)
│   ├── calibrate_fast.py             ← Newton–Gauss calibrator (jacfwd, ~1s)
│   ├── fim_analysis.py               ← Fisher Information Matrix standalone analysis
│   ├── fno_greeks.py                 ← Volga/Vanna via autograd through FNO
│   ├── app_fno.py                    ← Streamlit demo (main entry point)
│   ├── generate_dataset_v4_learnable_h.py  ← 6D training dataset generator
│   ├── train_fno_v3_learnable_h.py         ← FNO v3 training script
│   ├── validate_cuda.py              ← CUDA/COS validation utility
│   ├── market/                       ← [TIER 1 stubs — implement after research]
│   │   ├── spx_data.py               ← §1.1 SPX option chain download + cleaning
│   │   ├── vix_pricing.py            ← §1.3 VIX futures + variance swap pricing
│   │   └── deribit_data.py           ← §1.5 Deribit BTC/ETH option data
│   ├── greeks/                       ← [TIER 1 stubs]
│   │   └── portfolio_greeks.py       ← §1.4 Portfolio Greeks via torch.func
│   └── arbitrage/                    ← [TIER 1 stubs]
│       └── surface_completion.py     ← §1.2 Arbitrage-free IV surface completion
│
├── artifacts/
│   ├── weights/                      ← Production model weights
│   │   ├── fno_v2_final_prod.pth     ← FNO v2 SWA (R²=0.9991, H=0.08 fixed)
│   │   ├── fno_v3_final_prod.pth     ← FNO v3 SWA (R²=0.9981, learnable H)
│   │   └── fno_v3_best.pth           ← FNO v3 best-epoch checkpoint
│   ├── models/                       ← Normalizer stats (npz files)
│   │   ├── param_normalizer_v2.npz / iv_normalizer_v2.npz
│   │   └── param_normalizer_v3.npz / iv_normalizer_v3.npz
│   ├── legacy/                       ← Archived: LSTM weights, FNO v1, diff-FNO
│   └── reports/                      ← Audit reports, comparisons
│
├── data/                             ← Training datasets (gitignored, ~60MB)
│   ├── DeepRoughDataset_v2_fourier.npz  ← v2 COS dataset (32MB)
│   └── DeepRoughDataset_v4_learnable_h.npz ← v4 dataset (29MB, 50k samples)
│
├── benchmarks/                       ← Scripts + result .txt files
├── tests/                            ← pytest (32 tests, all passing)
├── scripts/                          ← Utilities (migrate, stress test)
├── research/                         ← Literature + analysis docs
│   └── deep_research_prompts_tier1.md  ← 5 Gemini Deep Research prompts
├── tex/                              ← LaTeX thesis + Beamer slides
├── articles/                         ← Reference PDFs
├── notebooks/                        ← Jupyter (Tier 1 analysis, empty for now)
├── results/                          ← Calibration outputs (spx/, crypto/, greeks/, vix/)
├── CONTEXT.md / README.md / ROADMAP.md / ROADMAP_ABSOLUTE_MAX.md
└── .venv/                            ← Python 3.14 venv
```


---

## 🧠 Model Architecture

### FNO v2 (production, H fixed at 0.08)
- **Class:** `MirrorPaddedFNO2d` in `src/fno_model.py`
- **Input:** `(B, T=8, K=11, 2)` channels-last spatial grid + 6D parameter vector via FiLM
- **FiLM conditioning:** `γ, β = MLP(θ_norm)`, applied after each spectral conv layer
- **Fourier modes:** 4T × 5K retained, 3-layer spectral conv
- **DC-trap fix:** zero out the (0,0) Fourier mode to remove the global level ambiguity
- **Output:** IV surface `(B, T=8, K=11)` in normalised space → denormalised by IVSurfaceNormalizer
- **Training:** 50k COS-labelled samples, SWA (Stochastic Weight Averaging), R²=0.9991
- **Weights:** `artifacts/weights/fno_v2_final_prod.pth`
- **Normalizers:** `artifacts/models/param_normalizer_v2.npz`, `artifacts/models/iv_normalizer_v2.npz`

### FNO v3 (learnable H)
- **Same architecture** as v2 except H is passed as a learnable input parameter
- **param_dim=6** (same as v2 — κ, θ, σ, ρ, V₀, H all as inputs)
- **R²=0.9981** (slightly lower — harder task, H adds degeneracy)
- **Weights:** `artifacts/weights/fno_v3_final_prod.pth`

### Grids
```python
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])  # 8 points
STRIKES    = np.linspace(-0.5, 0.5, 11)                             # log-moneyness
```
Note: T=0.1 is the hardest point — COS engine shows wing artefacts for deep OTM + short maturity.
Clip IV surface to `[0, 0.80]` before plotting to avoid visual distortion.

### Known Architecture Constraints
- Input spatial grid MUST be `(B, T, K, 2)` (channels-last)
- Both v2 and v3 use `param_dim=6`
- v2: κ=1.0, θ=0.08, H=0.08 are fixed ghost params in app; σ, ρ, V₀ are calibrated
- v3: all 6 parameters are live (H learnable)


---

## ⚡ Calibration Pipeline

### Reparameterization (3D space)
The app calibrates in a lower-dimensional space to improve conditioning:
```
ζ = σρ          (smile skew)
λ = σ√(1−ρ²)   (wing curvature)
```
So calibration finds `(V₀, ζ, λ)` from 3 degrees of freedom instead of `(σ, ρ)`.
Recovery: `σ = √(ζ²+λ²)`, `ρ = ζ/σ`.

### Newton–Gauss (Option 3, recommended)
- **File:** `src/calibrate_fast.py` → `calibrate_newton()`
- **Jacobian:** `torch.func.jacfwd` through FNO (analytic, no finite differences)
- **Speed:** p50=541ms, p95=668ms, ~3× faster than L-BFGS
- **Restarts:** 5 random restarts, best loss selected
- **Convergence:** typically 4–8 iterations

### L-BFGS (Option 2)
- **File:** `src/calibrate.py` → `calibrate_reparameterized()`
- **Optimizer:** `scipy.optimize.minimize(method='L-BFGS-B')`
- **Speed:** ~1.5–2s

### Fisher Information Matrix
- **File:** `src/calibrate.py` → `compute_fim_ellipsoid()`
- **Returns:** `{"fim_matrix", "cov_matrix", "std_errors", "corr_matrix", "ci_95", "jacobian", ...}`
- **Key:** `std_errors` is a **numpy array** `[σ_v0, σ_zeta, σ_lambda]` (NOT a dict)
- **Key:** `ci_95` is a **dict** `{"v0": (lo, hi), "zeta": (lo, hi), "lambda": (lo, hi)}`
- **Confidence display:** uses FIM-based `1 − 2σ/|estimate|` formula (NOT Jacobian norms)
- **Display condition:** `_reparam_mode` flag (True for BOTH Newton and Reparameterized modes)

### Structural Non-Identifiability
- λ confidence is LOW on k∈[−0.5,+0.5] grid (near-ATM only)
- λ = σ√(1−ρ²) controls wing curvature, visible only at |k|>1 (deep OTM)
- This is a structural feature, NOT a calibration failure
- FIM-based confidence bars now correctly show 🟢 even for λ when FIM CIs are tight

---

## 🖥️ Streamlit App (`src/app_fno.py`)

### Entry point
```bash
cd /home/execorn/programming/derivatives
.venv/bin/streamlit run src/app_fno.py
```

### Three calibration modes (sidebar radio):
1. **Newton-Raphson — Option 3 (recommended):** `_newton_mode=True`, `_reparam_mode=True`
2. **Reparameterized 3D — Option 2 (L-BFGS):** `_newton_mode=False`, `_reparam_mode=True`
3. **Full 6D — Option 2 (experimental):** `_newton_mode=False`, `_reparam_mode=False`

### Key UI sections:
- Sidebar: ghost params (κ, θ, H fixed), live params (σ, ρ, V₀), noise slider (step=0.001)
- Main: "Generate Target Surface" → "Calibrate" buttons
- Surface plot: 3D Plotly, clipped to `[0, 0.80]`, z-axis auto-scaled
- Confidence bars: FIM-based when `_reparam_mode=True` and `fim_res` available
- Newton details: GN loss curve, parameter trajectory, FIM table, correlation matrix

### Known issues / notes:
- T=0.1 slice shows COS wing artefacts for deep OTM (clipped in plot, harmless)
- `CONF_NAMES` dict only has 6D keys (kappa/theta/sigma/rho/v0/H) — not zeta/lambda
- FIM display requires `fim_res["std_errors"]` (numpy array) and `fim_res["ci_95"]` (dict of tuples)


---

## 📊 Key Benchmark Results

| Metric | Value | Source |
|---|---|---|
| FNO v2 R² | 0.9991 | `benchmarks/validate_fno_v2.py` |
| FNO v3 R² | 0.9981 | training log |
| COS vs MC speedup | ~1400× | `benchmarks/mc_vs_cos_results.txt` |
| COS vs MC error | <0.3% IV | same |
| Newton p50 latency | 541 ms | `benchmarks/streaming_demo_results.txt` |
| Newton p95 latency | 668 ms | same |
| Newton vs L-BFGS speedup | ~3× | same |
| Hedge P&L improvement | 3× better vs B-S | `benchmarks/greeks_hedge_results.txt` |
| Noise robustness | stable to 2% IV noise | `benchmarks/noise_robustness_results.txt` |
| Test suite | 32/32 pass | `pytest tests/` |

### Parameter Training Ranges (FNO v2/v3)
```python
PARAM_BOUNDS = {
    "kappa": (0.5,  5.0),
    "theta": (0.01, 0.25),
    "sigma": (0.1,  1.5),
    "rho":   (-0.95, 0.0),
    "v0":    (0.01, 0.25),
    "H":     (0.04, 0.15),  # v3 only; v2 fixes H=0.08
}
```
**Crypto note:** BTC/ETH requires extended ranges (V₀ up to 0.6, σ up to 3.0) — FNO retraining needed.

---

## 🔬 Tier 1 Extension Status

| Task | Status | Stub file | Research prompt |
|---|---|---|---|
| §1.1 Real SPX calibration | ✅ **DONE** — rmse=41.5 bps | `src/market/spx_data.py` | commit `3385f29` |
| §1.2 Arbitrage-free completion | ✅ **DONE** — 0 butterfly violations | `src/arbitrage/surface_completion.py` | commit `48d37d4` |
| §1.3 VIX + variance swaps | ✅ **DONE** — VIX=31.3 (kappa=1,θ=0.08) | `src/market/vix_pricing.py` | commit `f2ad787` |
| §1.4 Greeks at scale | ✅ **DONE** — delta/gamma/vega/vanna/volga | `src/greeks/portfolio_greeks.py` | commit `3695410` |
| §1.5 Crypto (Deribit) | ✅ **DONE** — 580 BTC options fetched | `src/market/deribit_data.py` | commit `b7736fd` |

### Live API tests (confirmed 2026-06-18):
- **yfinance 1.4.1:** SPX chain returns 152 strikes, columns: `strike, bid, ask, impliedVolatility, openInterest`
- **Deribit REST API** (no auth needed): 934 BTC options, `mark_iv` in %, name format `BTC-28JUN24-70000-C`
- **py_vollib_vectorized:** installed, for Black-Scholes IV recomputation
- **aiohttp 3.14.1:** installed, for async Deribit batch download
- **fastapi 0.137.1:** installed, for future REST pricing API

### Deribit instrument name format:
```
BTC-28JUN24-70000-C
 ^    ^       ^    ^
coin  expiry  strike type (C/P)
```
Parse: `parts = name.split("-")` → `coin=parts[0]`, `expiry=datetime.strptime(parts[1], "%d%b%y")`,
`strike=int(parts[2])`, `option_type=parts[3]`


---

## 🧹 Cleanup History

### 2026-06-18 Major Cleanup
**Deleted (LSTM-era, never needed again):**
- `src/app.py`, `src/model.py`, `src/seq_model.py`, `src/calibrator.py`
- `src/data_loader.py`, `src/train.py`, `src/train_seq.py`
- `src/benchmark_plots.py`, `src/benchmark_plots_fno.py`
- `src/greeks_autograd.py` (replaced by `src/fno_greeks.py`)
- `src/validation.py`, `src/iv_inverter.py`, `src/test_resolution.py`
- `src/generate_dataset.py`, `src/generate_dataset_v2.py`
- `src/generate_dataset_v3_differential.py`, `src/train_fno.py`
- `src/train_fno_differential.py`
- `tests/test_phase2_uncertainty.py`, `tests/test_phase3_lstm.py`
- `scripts/generate_seq_data.py`
- `data/DeepRoughDataset.npz` (15MB, v1 MC), `data/seq_dataset.npz` (37MB)
- `data/HestonTrainSet.txt.gz` (8MB)
- `build/` (CUDA build artifacts), `build_all_tex.bat`, `.antigravityignore`
- `benchmarks/validate_fno_v2_run.log`, `benchmarks/mc_vs_cos_run.log`
- `benchmarks/fno_v3_eval.npy`

**Archived to `artifacts/legacy/` (not deleted, just moved):**
- `heston_best.pth`, `heston_best_no_dropout.pth`, `heston_lstm_best.pth`
- `fno_final_prod.pth` (FNO v1, MC labels), `fno_diff_final_prod.pth`
- `artifacts/models/fno_best.pth`, `fno_diff_best.pth`
- Old normalizers: `iv_normalizer.npz`, `param_normalizer.npz`, `*_diff.npz`
- Old scalers: `feature_scaler.pkl`, `target_scaler.pkl`, `lstm_label_stats.npz`

**Fixed:**
- `ROADMAP_ABSOLUTE_MAX.md.` → `ROADMAP_ABSOLUTE_MAX.md` (trailing dot typo)

**Net result:** ~100MB freed, 20+ dead files removed, 32/32 tests still passing.

---

## 🛠️ Development Environment

```bash
# Python version
python3.14

# Virtual environment
cd /home/execorn/programming/derivatives
source .venv/bin/activate   # or use .venv/bin/python directly

# Run Streamlit app
.venv/bin/streamlit run src/app_fno.py

# Run tests
.venv/bin/python -m pytest tests/ -q

# Run a benchmark
cd /home/execorn/programming/derivatives
.venv/bin/python benchmarks/streaming_calibration_demo.py

# Compile thesis
cd tex/thesis && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Key Python packages
```
torch          2.x        Neural network + autograd
streamlit      ~1.x       Demo app
plotly         6.7.0      3D surface plots
scipy          1.17.1     L-BFGS, optimization
numpy          2.x        Numerical core
yfinance       1.4.1      SPX option chain download  [NEW]
py_vollib_vectorized       Black-Scholes IV computation  [NEW]
aiohttp        3.14.1     Async HTTP (Deribit)  [NEW]
fastapi        0.137.1    Future REST pricing API  [NEW]
pyarrow        24.0.0     Parquet file I/O
websockets     16.0       WebSocket streams
uvicorn        0.46.0     ASGI server for FastAPI
```

---

## 📝 Next Steps for Incoming Agent

1. **Wait for Gemini Deep Research results** (5 prompts sent, covering §1.1–§1.5)
2. **Implement stubs** in this order:
   - `src/market/spx_data.py` (§1.1) — highest priority, all else builds on it
   - `src/market/deribit_data.py` (§1.5) — easy, API confirmed working
   - `src/market/vix_pricing.py` (§1.3) — needs Rough Heston Laplace transform theory
   - `src/arbitrage/surface_completion.py` (§1.2) — SVI fitting + monotone rearrangement
   - `src/greeks/portfolio_greeks.py` (§1.4) — torch.func.jacfwd through FNO
3. **Store results** in `results/spx_calibration/`, `results/crypto_calibration/`, etc.
4. **Write Jupyter notebooks** in `notebooks/` for analysis and figures
5. **Update ROADMAP_ABSOLUTE_MAX.md** task statuses as Tier 1 items complete

### Critical gotchas for next agent:
- Always activate `.venv` before running anything
- FNO input MUST be `(B, T, K, 2)` channels-last — easy to get wrong
- `fim_res["std_errors"]` is a numpy array (indexed by position 0,1,2)
- `fim_res["ci_95"]` is a dict `{"v0":(lo,hi), "zeta":(lo,hi), "lambda":(lo,hi)}`
- `_reparam_mode = calib_mode.startswith("Reparameterized") or _newton_mode`
  → Newton mode IS reparameterized; check `_reparam_mode`, NOT `calib_mode` string
- Deribit `mark_iv` is in % (e.g., 42.63 means 42.63% = 0.4263 annualised)
- yfinance `impliedVolatility` is often stale — recompute from bid-ask mid via py_vollib
- BTC crypto options need extended param ranges (V₀ up to 0.6) → may need FNO retraining

---

## 🚀 Tier 2 Extension Status (P2) — COMPLETE 2026-06-20

| Module | File | Status | Tests |
|--------|------|--------|-------|
| B1 FastAPI REST API | `src/api/server.py` | ✅ **DONE** | 34 |
| B2 Variance Swap Pricing | `src/market/variance_swaps.py` | ✅ **DONE** | 36 |
| B3 Deribit WebSocket | `src/market/deribit_ws.py` | ✅ **DONE** | 17 |
| B4 Joint SPX+VIX Calibration | `src/calibration/joint_calibration.py` | ✅ **DONE** | 13 |
| B5 GPU Batch Calibration | `src/calibration/batch_calibration.py` | ✅ **DONE** | 15 |

**Total tests as of 2026-06-20:** 511 passing, 0 failing, 28 warnings (expected, py_vollib Below Intrinsic)

### P2 Bug Fixes Applied (from P1 robustness audit)
- `vega_bucket` silent float32 overflow: MATURITIES dtype=float32 (max 3.4e38) → cast to float64
- `datetime.UTC` AttributeError on Python 3.14 → `timezone.utc`
- SPX `all_options.append` indentation error
- `bs_greeks` ZeroDivisionError / OverflowError guards added
- `fno_surface_greeks` non-finite θ clamping + UserWarning

### New Directory Structure Additions (P2)
```
src/
├── api/
│   ├── __init__.py
│   └── server.py          ← FastAPI: /health /iv_surface /greeks /vix /deribit/snapshot
├── calibration/
│   ├── __init__.py
│   ├── joint_calibration.py   ← L-BFGS-B joint SPX+VIX loss
│   └── batch_calibration.py   ← ThreadPoolExecutor + GPU batch FNO inference
└── market/
    ├── variance_swaps.py  ← variance_swap_rate, realized_variance, term_structure
    └── deribit_ws.py      ← DeribitWSClient, stream_realtime_surface (aiohttp WS)

tests/
├── test_api.py              ← 34 tests (FastAPI TestClient, mocked Deribit)
├── test_deribit_ws.py       ← 17 tests (mocked aiohttp WS, no real network)
├── test_joint_calibration.py ← 13 tests (synthetic FNO surface, VIX ODE)
└── test_batch_calibration.py ← 15 tests (dataclass, JSON I/O, calibration pipeline)
```

### Key New APIs
```python
# FastAPI server
from api.server import app    # uvicorn api.server:app --port 8000
# POST /iv_surface {kappa,theta,sigma,rho,v0,H} → {surface: [[float]], T_grid, K_grid}
# POST /greeks     {kappa,...,S} → {delta,gamma,vega,vanna,volga,iv_surface: [[float]]}
# POST /vix        {kappa,...}   → {vix: float}

# Variance swaps
from market.variance_swaps import variance_swap_rate, variance_term_structure
kvar = variance_swap_rate(kappa=2.5, theta=0.08, sigma=0.5, rho=-0.5, v0=0.08, H=0.08, T=1.0)

# WebSocket streaming
from market.deribit_ws import DeribitWSClient, stream_realtime_surface
async with DeribitWSClient() as client:
    async for df in client.stream_iv_surface("BTC"):
        print(df.head())   # live IV surface updates

# Joint calibration
from calibration.joint_calibration import calibrate_joint, calibrate_vix_only
result = calibrate_joint(spx_surface, vix_level=18.5, weights=(1.0, 1.0), n_restarts=3)
# → {kappa, theta, sigma, rho, v0, H, spx_rmse_bps, vix_error, total_loss, converged}

# Batch calibration
from calibration.batch_calibration import calibrate_batch, save_results, results_to_dataframe
results = calibrate_batch(["2024-01-02", "2024-08-05"], currency="SPX", device="auto")
save_results(results, "results/batch/2024.json")
```

### New Packages Installed (P2)
```
httpx          ← FastAPI TestClient (pytest dependency)
pytest-asyncio ← async test support
```

---

## 🔭 Tier 3 Roadmap (P3) — NEXT PHASE

See full plan in `ROADMAP_ABSOLUTE_MAX.md` and `implementation_plan.md`.

### Priority Order
1. **P3-Infrastructure** (30 min) — pyproject.toml asyncio mode, pin httpx in requirements.txt
2. **P3-D: VIX Term Structure** — multi-tenor VIX futures calibration (highest thesis impact)
3. **P3-E: Hurst Dynamics** — batch-calibrate ~1500 SPX dates, novel empirical chapter
4. **P3-F: Greeks Benchmark** — FNO autograd vs finite-difference COS speed/accuracy
5. **P3-A: Rough Bergomi** — Hybrid scheme MC dataset + new FNO training (~3 GPU-days)
6. **P3-B: Neural SDE** — model-free deep learning (most research-intensive, 4–8 weeks)
