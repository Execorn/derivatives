# CONTEXT.md — AI Agent & Developer Orientation Map
# Deep Rough Heston Calibration via FiLM-FNO
# МФТИ ФПМИ Master's Thesis, 2026

---

## 1. Project Identity

**What**: An end-to-end Python framework for real-time calibration of stochastic
volatility (SV) models using neural network surrogates (Fourier Neural Operators).
The FNO replaces a slow analytical/Monte Carlo pricer, enabling Gauss-Newton
calibration in under 1 second instead of minutes.

**Goal of thesis**: A unified framework where practitioners can calibrate *any*
of several SV models (Rough Heston, rBergomi, Classic Heston, SABR, Local Vol,
Neural SDE) to market data using a common FNO-based pipeline. The user selects
a model; the framework handles training data generation, surrogate training,
normalizer creation, and Gauss-Newton calibration automatically.

**Student**: Execorn (МФТИ ФПМИ, Кафедра БИТ, 2026)
**Repo**: `git@github.com:Execorn/derivatives.git`
**Branch**: `master` (single branch, force-pushes allowed)

---

## 2. Repository Map

```
derivatives/
├── src/
│   ├── fno_model.py              MirrorPaddedFNO2d — the core neural architecture
│   ├── normalizers.py            ParameterNormalizer, IVSurfaceNormalizer
│   ├── calibrate.py              Core: _load_normalizers, _fno_predict_real_iv,
│   │                             _make_spatial_input, calibrate_parameters,
│   │                             calibrate_reparameterized
│   ├── calibrate_fast.py         calibrate_newton (3-param, v2),
│   │                             calibrate_newton_h (4-param, v3),
│   │                             fno_jacobian_autograd
│   ├── fim_analysis.py           Fisher Information Matrix, reparameterization
│   ├── fno_greeks.py             Autograd Greeks (delta/gamma/vega/vanna/volga)
│   ├── app_fno.py                Streamlit dashboard
│   ├── cuda_engine.cu            CUDA C++ Lifted Heston kernel
│   ├── pricing_engine.py         CPU Fourier-COS pricer (reference)
│   ├── pricing_engine_gpu.py     GPU Fourier-COS pricer (vectorized)
│   ├── calibration/
│   │   ├── batch_calibration.py  Multi-date Gauss-Newton calibration
│   │   └── joint_calibration.py  Joint SPX + VIX calibration
│   ├── market/
│   │   ├── spx_data.py           SPX options: yfinance download, clean, grid
│   │   ├── vix_futures.py        VIX futures term structure
│   │   ├── vix_pricing.py        Model VIX from Rough Heston parameters
│   │   ├── deribit_data.py       Deribit REST API (BTC/ETH options)
│   │   ├── deribit_ws.py         Deribit WebSocket streaming
│   │   └── variance_swaps.py     Variance/vol swap pricing
│   ├── arbitrage/
│   │   └── surface_completion.py SVI fitting + butterfly/calendar enforcement
│   ├── greeks/
│   │   ├── portfolio_greeks.py   GPU portfolio greeks
│   │   └── pnl_attribution.py    Taylor P&L decomposition
│   ├── analysis/
│   │   ├── hurst_dynamics.py     Historical H study (SPX)
│   │   └── crypto_hurst.py       Historical H study (BTC/ETH)
│   └── api/
│       └── server.py             FastAPI REST server
├── tests/                        535 tests, pytest, no live-network dependencies
├── notebooks/
│   ├── generate_notebooks.py     Source of truth — regenerates all .ipynb
│   └── *.ipynb                   01-07: SPX, surface, VIX, greeks, BTC, batch, joint
├── benchmarks/                   Accuracy and speed studies
├── scripts/                      Utility scripts
├── data/                         GITIGNORED — large training datasets
├── artifacts/
│   ├── weights/                  fno_v2_final_prod.pth, fno_v3_final_prod.pth
│   └── models/                   param_normalizer_v{2,3}.npz, iv_normalizer_v{2,3}.npz
├── results/                      JSON calibration outputs
├── research/                     Research notes + reference PDFs
├── tex/                          LaTeX thesis (main.pdf) + presentation slides
└── README.md                     Public-facing documentation
```

---

## 3. Model Versioning — CRITICAL

This is the most common source of bugs. Always pair model weights with the
correct normalizer version.

| Version | Weights file | Normalizer | Params | H |
|---------|-------------|------------|--------|---|
| v2 | `fno_v2_final_prod.pth` | `param_normalizer_v2.npz` | κ,θ,σ,ρ,V₀ + fixed H=0.08 | Fixed |
| v3 | `fno_v3_final_prod.pth` | `param_normalizer_v3.npz` | κ,θ,σ,ρ,V₀,H (6 params) | Learnable |

**Rules:**
- `calibrate_newton(model, ...)` → uses v2 normalizers (3-param reparameterization,
  H fixed at `_GHOST_H = 0.08`)
- `calibrate_newton_h(model, ...)` → uses v3 normalizers (4-param: v0,ζ,λ,H)
- Never call `_load_normalizers()` without an explicit version string
- `_load_normalizers("v2")` and `_load_normalizers("v3")` are idempotent
  (cached per path — switching version invalidates the cache)
- `_fno_predict_real_iv` is lazy: it does NOT call `_load_normalizers()`
  itself — the caller must pre-load the correct version

---

## 4. Normalizer Architecture

```python
# src/calibrate.py
_NORM_VERSIONS = {
    "v1": ("artifacts/models/param_normalizer_v1.npz",   # legacy identity
            "artifacts/models/iv_normalizer_v1.npz"),
    "v2": ("artifacts/models/param_normalizer_v2.npz",   # 5-param + fixed H
            "artifacts/models/iv_normalizer_v2.npz"),
    "v3": ("artifacts/models/param_normalizer_v3.npz",   # 6-param learnable H
            "artifacts/models/iv_normalizer_v3.npz"),
}
```

`_resolve_norm_path(p)` handles both CWD=project root (scripts) and
CWD=notebooks/ (Jupyter) by falling back to `_ROOT_DIR` which is the
directory two levels above `calibrate.py`.

`spx_data.py:calibrate_to_market()` uses a temporary `_NORM_VERSIONS["v1"]`
redirect so that auto-loading code sees the correct version. This pattern
uses `try/finally` and resets `_param_norm = None` before and after.

---

## 5. FNO Architecture Summary

`MirrorPaddedFNO2d` in `src/fno_model.py`:

```
θ (B×6) ──→ ParameterNormalizer.normalize() ──→ FiLM MLP (hidden=256)
                                                     │ (γᵢ, βᵢ) per layer
(T,K) grid (8×11) ──→ mirror-pad (16×22) ──→ Lifting Conv (1→40 ch)
                        4 × FourierLayer (modes=8×11, FiLM-modulated)
                        Projection (40→1)
                        IVSurfaceNormalizer.denormalize()
                              │
                         IV surface (8×11)   [decimal vol units]
```

- **T_GRID**: 8 maturities (defined in `src/market/spx_data.py`)
- **K_GRID**: 11 log-moneyness strikes
- **param_dim=6** always (v2 passes fixed H internally via `_reparam_to_6d`)
- Mirror padding: enforces put-call parity & calendar spread monotonicity
- FiLM: scale+shift per Fourier layer — better than concatenation for
  parameter conditioning
- ATM-weighted Huber loss during training


---

## 6. Calibration Pipeline (Data Flow)

### Single-date SPX calibration (NB01 / `calibrate_newton_h`)

```
yfinance / parquet cache
       │
       ▼
download_spx_chain(date)  →  pd.DataFrame (raw chain)
       │
clean_chain()             →  filtered DataFrame
       │
to_iv_surface(df, S0, r, q)  →  (8, 11) ndarray in decimal vol
       │
_load_normalizers("v3")   (v3 for learnable H)
       │
calibrate_newton_h(model, iv_surface, T_GRID, K_GRID)
   ┌───────────────────────────────────────────────────────┐
   │  init: 3 starting points (v0_est, ζ, λ, H)           │
   │  loop (max=20 iters):                                 │
   │    p6 = _reparam_to_6d_with_H(v0, ζ, λ, H, device)   │
   │    iv_pred = _fno_predict_real_iv(model, p6, spatial) │
   │    J = fno_jacobian_autograd(model, θ4d, spatial)     │
   │    δθ = -(JᵀJ + ε·diag)⁻¹ Jᵀ r                      │
   │    θ ← clip(θ + α·δθ, bounds)                        │
   │    stop if RMSE < 100 bps                             │
   └───────────────────────────────────────────────────────┘
       │
  result dict: v0, sigma, rho, H, kappa, theta,
               final_mse, history, theta_history,
               n_iter, elapsed, converged
```

### Batch calibration (`calibrate_batch` in batch_calibration.py)

- Iterates over date list with ThreadPoolExecutor (data fetch) +
  sequential GPU calibration
- Resume-capable: already-saved dates are skipped
- kappa=3 starting point added (more robust convergence)
- Convergence threshold: 100 bps (raised from 50 for batch stability)

### Joint SPX + VIX (`calibrate_joint` in joint_calibration.py)

- Minimizes: w_spx·RMSE_spx² + w_vix·RMSE_vix²
- VIX futures curve modeled via `compute_vix_term_structure`
- Uses L-BFGS-B (scipy) on the 6-param space

---

## 7. Key Constants and Grids

```python
# src/market/spx_data.py
T_GRID = [0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # 8 maturities (years)
K_GRID = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0,             # 11 log-moneyness
           0.1,  0.2,  0.3,  0.4,  0.5]                   # 0 = ATM

# src/calibrate_fast.py
_GHOST_KAPPA = 1.0   # fixed for v2 calibration
_GHOST_THETA = 0.08  # fixed for v2 calibration
_GHOST_H     = 0.08  # fixed for v2 calibration (H was fixed during v2 training)

# Rough Heston parameter bounds (v3 / 4D calibration)
_BOUNDS_LOWER_4D = [0.005, -0.55, 0.05, 0.04]   # [v0, ζ, λ, H]
_BOUNDS_UPPER_4D = [0.200,  0.00, 0.80, 0.15]

# 3D bounds (v2 / fix_H calibration)
_BOUNDS_LOWER_3D = [0.005, -0.55, 0.05]
_BOUNDS_UPPER_3D = [0.200,  0.00, 0.80]
```

---

## 8. Training Data Generation

Training datasets are **gitignored** (large .npz files in `data/`).
They must be regenerated if lost.

| Dataset file | Model | N samples | Generation script |
|---|---|---|---|
| `DeepRoughDataset_v2_fourier.npz` | Rough Heston (H=0.08 fixed) | ~100k | `scripts/generate_dataset_v2.py` |
| `DeepRoughDataset_v4_learnable_h.npz` | Rough Heston (H learnable) | ~200k | `scripts/generate_dataset_v4.py` |

Generation pipeline:
1. Sample parameters uniformly from hypercube (see bounds above)
2. Run Lifted Heston (N=40 factors) via `pricing_engine_gpu.py`
3. Compute IV surface via Fourier-COS (N_cos=128 terms)
4. Filter NaN surfaces (exponential midpoint integrator → 2.76% NaN rate)
5. Fit ParameterNormalizer and IVSurfaceNormalizer on the dataset
6. Save: `{params, iv_surfaces, param_mean, param_std, iv_mean, iv_std}`

Training the FNO:
```bash
source .venv/bin/activate
python scripts/train_fno_v3.py \
    --data data/DeepRoughDataset_v4_learnable_h.npz \
    --epochs 300 --lr 1e-3 --batch-size 512 \
    --out artifacts/weights/fno_v3_final_prod.pth
```

---

## 9. Test Suite Structure

535 tests, ~2.5 min runtime. Key test files:

| File | What it tests | Notes |
|------|---------------|-------|
| `test_pricing_engine.py` | Fourier-COS accuracy | CPU vs GPU parity |
| `test_normalizers.py` | Normalizer roundtrips | v2 and v3 |
| `test_calibrate_newton.py` | 3-param Newton (v2) | Self-consistency |
| `test_calibrate_newton_h.py` | 4-param Newton (v3) | Uses v3 fixture! |
| `test_batch_calibration.py` | Multi-date calibration | Resume logic |
| `test_joint_calibration.py` | SPX+VIX joint | L-BFGS-B |
| `test_spx_data.py` | SPX pipeline smoke | RMSE < 5000 bps |
| `test_surface_completion.py` | SVI + arbitrage | Butterfly/calendar |
| `test_portfolio_greeks.py` | Delta/gamma/vega | GPU accuracy |
| `test_pnl_attribution.py` | Taylor P&L | Greeks breakdown |
| `test_api.py` | FastAPI endpoints | `/calibrate`, `/iv_surface` |
| `test_deribit_data.py` | Deribit REST | Mocked responses |
| `test_vix_pricing.py` | VIX term structure | Model vs market |
| `test_hurst_dynamics.py` | Historical H study | 2 skipped (live) |

Run: `pytest tests/ -q` from repo root.

**NEVER change CWD to `tests/` before running** — path resolution in
`_resolve_norm_path` relies on the root being two levels above `calibrate.py`.


---

## 10. Common Pitfalls (Lessons Learned)

### 10.1 Normalizer version mismatch
**Symptom**: RMSE=500-3000 bps on a surface that should calibrate to <50 bps.
**Cause**: Wrong normalizer version paired with model weights.
**Fix**: Always call `_load_normalizers("v2")` before `calibrate_newton`,
         `_load_normalizers("v3")` before `calibrate_newton_h`. Never bare
         `_load_normalizers()`.

### 10.2 CWD-dependent path failures in notebooks
**Symptom**: `FileNotFoundError: artifacts/models/param_normalizer_v3.npz`
  when running from `notebooks/` directory.
**Cause**: Relative path resolution. `os.path.exists(p)` returns False
  because CWD is `notebooks/`, not project root.
**Fix**: `_resolve_norm_path` handles this via `_ROOT_DIR` fallback. If you
  see this error, check that `_ROOT_DIR` is set correctly in `calibrate.py`.
  The `_ROOT_DIR` is `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`.

### 10.3 Notebooks: always use generate_notebooks.py
**Never edit `.ipynb` files directly**. The source of truth is
`notebooks/generate_notebooks.py`. To update a notebook, edit the generator
and run `python generate_notebooks.py`. This prevents JSON merge conflicts.

### 10.4 Batch calibration convergence
The batch calibrator (NB06) uses 3 starting points including `kappa=3`.
The convergence threshold is 100 bps (not 50 bps). Do not tighten this —
it causes failures on real market surfaces with noisy wings.

### 10.5 v2 model on real market data
The v2 model (3-param: v0, ζ, λ) will produce RMSE of 500-2000 bps on real
SPX data. This is expected — v2 has only 3 free parameters and fixes
kappa=1.0, theta=0.08, H=0.08. Use v3 for real-data calibration.

### 10.6 CUDA test skipping
Tests in `TestCalibrateNewtonH` are skipped if CUDA is unavailable (CI env).
This is intentional. Run locally with GPU to execute them.

### 10.7 derivatives-w1 paths
All `derivatives-w1/` hardcoded paths have been removed (commit 1f8448f).
If you see such paths in any file, it's a regression — remove them. The
canonical path is always `/home/execorn/programming/derivatives/`.

---

## 11. How to Add a New SV Model

This is the primary development pattern going forward. Step-by-step:

### Step 1: Simulation / pricing
Create `src/pricing/my_model.py`:
```python
def price_my_model(params: np.ndarray, T_grid, K_grid,
                   N_paths=50_000, device="cuda") -> np.ndarray:
    """Return IV surface (len(T_grid), len(K_grid)) or NaN on failure."""
    ...
```
Use GPU Monte Carlo (torch) for path simulation. Analytical CF + COS if available.

### Step 2: Dataset generation
Create `scripts/generate_dataset_mymodel.py`:
```python
# Sample params from hypercube, call price_my_model, filter NaN, save .npz
# Target: 100k-500k samples. Budget: ~4-6 hours on RTX 3080.
```

### Step 3: Normalizer fitting
```python
from normalizers import ParameterNormalizer, IVSurfaceNormalizer
pn = ParameterNormalizer.fit(param_array)  # (N, D) array
yn = IVSurfaceNormalizer.fit(iv_array)      # (N, 8, 11) array
pn.save("artifacts/models/param_normalizer_mymodel.npz")
yn.save("artifacts/models/iv_normalizer_mymodel.npz")
```

### Step 4: Register version
In `src/calibrate.py`, add to `_NORM_VERSIONS`:
```python
"mymodel": ("artifacts/models/param_normalizer_mymodel.npz",
            "artifacts/models/iv_normalizer_mymodel.npz"),
```

### Step 5: Train FNO surrogate
```bash
python scripts/train_fno.py \
    --data data/MyModelDataset.npz \
    --param-dim D \  # number of parameters
    --out artifacts/weights/fno_mymodel_final_prod.pth
```
`MirrorPaddedFNO2d(param_dim=D)` — no other architecture changes needed.

### Step 6: Write calibrator
In `src/calibrate_fast.py`, add `calibrate_newton_mymodel()`:
- Load v_mymodel normalizers
- Define reparameterization if needed (for numerical stability)
- Run Gauss-Newton loop with `fno_jacobian_autograd`

### Step 7: Tests + notebook
- Add `tests/test_calibrate_mymodel.py` with self-consistency check
- Add notebook `notebooks/generate_notebooks.py` section for NB0X

---

## 12. Environment Setup

```bash
# Tested configuration
python 3.14
torch 2.12 + CUDA 12.6
GPU: RTX 3080 / RTX 4090 / A100 (tested)

# Install
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision torchaudio
pip install pandas numpy scipy scikit-learn matplotlib seaborn \
    streamlit plotly torchdiffeq httpx pytest-asyncio \
    fastapi uvicorn aiohttp py_vollib_vectorized yfinance

# Optional CUDA extension (Lifted Heston direct kernel)
python setup.py build_ext --inplace
```

---

## 13. Git Workflow

```bash
# Single branch (master). No PRs. Force-push is allowed.
git add -A && git commit -m "fix: ..."
git push origin master

# To squash recent commits:
git reset --soft HEAD~N
git commit -m "unified message"
git push origin master --force
```

Commit message prefixes: `fix:` `feat:` `chore:` `docs:` `refactor:` `test:`

---

## 14. Contacts & Context Recovery

If this context is stale or incomplete, the following files always have the
authoritative current state:
- `README.md` — public documentation, updated with each milestone
- `tests/` — executable specification of expected behavior
- `notebooks/generate_notebooks.py` — end-to-end pipeline demonstration
- This file (`CONTEXT.md`) — agent-oriented deep context

