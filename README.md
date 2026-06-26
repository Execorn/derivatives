# Deep Rough Heston Calibration via FiLM-FNO

[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-1008%20passed-brightgreen?style=for-the-badge)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Master's Thesis Project** — МФТИ ФПМИ, Кафедра БИТ, 2026
> *Нейронные сети в ценообразовании производных финансовых инструментов*

An end-to-end GPU-accelerated system for **real-time calibration** of the
**Rough Heston stochastic volatility model** (El Euch, Gatheral & Rosenbaum 2019).
The core idea: replace an expensive Fourier-COS pricer (seconds per surface) with a
**FiLM-conditioned Fourier Neural Operator (FNO)** surrogate that prices a full 8×11
implied-volatility surface in under 1 ms — enabling Gauss-Newton calibration at
interactive speed.

The project covers the full quant stack: mathematical foundations → GPU pricing
→ neural surrogate → real-time calibration → REST API → live market data integration.

---

## Key Results

| Experiment | Result |
|---|---|
| FNO v2 surrogate accuracy | **R² = 0.9991, MAE = 0.058%** (5.8 bp) |
| FNO v3 (learnable H ∈ [0.04, 0.15]) | **R² = 0.9981, MAE = 0.264%** |
| Inference speed | **~4 ms** (batch 1024) vs 5.6 s direct COS — **1400× speedup** |
| Newton calibration — SPX synthetic (NB01) | **8.9 bps RMSE**, converged in 12 iters |
| Batch calibration — 5 dates (NB06) | **12.8 bps median RMSE**, 5/5 converged, H=0.108 |
| Joint SPX + VIX calibration (NB07) | SPX **198.9 bps**, VIX error **0.362**, converged |
| BTC live Deribit calibration (NB05) | **1547 bps RMSE**, v2 model, 548 contracts |
| Streaming p95 latency (20 ticks) | **668 ms** < 1000 ms real-time threshold |
| FIM reparameterization | Condition number **1301× lower** (κ ≈ 770) |
| Delta hedging variance reduction | **+5.1%** vs flat Black-Scholes Δ |
| NaN rate (exponential midpoint integrator) | **2.76%** (was 70.4% with Euler) |
| Neural SDE Adjoint Calibration (NB12) | **8.1 bps final option price RMSE** |
| Signature Vol martingale error (NB13) | **0.11 bps** (well within 10 bps limit) |
| Recurrent Deep Hedging (NB14-15) | **0.4388 European, 0.4349 Barrier P&L std** |
| Minimax GAN market generation (NB16) | **0.0003 final tracking error** |
| FastAPI Deep Hedging throughput | **132.91 RPS (p50 = 6.57 ms)** on GPU |
| FastAPI FNO Calibration throughput | **353.42 RPS (p50 = 80.01 ms)** |
| PI-M-FNO online crisis adaptation speed | **~3.5 ms** (2 inner-gradient steps, PDE loss < 10⁻⁴) |
| D-XVA gradient flow differentiability | **Non-zero, non-NaN** ($\nabla_{\psi} \text{Var}(\Pi_T) \ne \mathbf{0}$) |
| Grey Rough Bergomi MC path simulation | **< 15 ms** on GPU (4096 paths, 100 steps) |
| Sobolev FNO gRB Validation accuracy | **MAE = 0.76 bp, MSE = 7.69e-5** (Active Learning) |

---

## Table of Contents

1. [Mathematical Background](#mathematical-background)
2. [DeepVol Model Zoo](#deepvol-model-zoo)
3. [Monte Carlo Risk Engine (VaR/ES)](#monte-carlo-risk-engine-vares)
4. [Architecture](#architecture)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Notebooks](#notebooks)
8. [REST API](#rest-api)
9. [Benchmarks](#benchmarks)
10. [Test Suite](#test-suite)
11. [Models & Weights](#models--weights)
12. [Project Structure](#project-structure)
13. [Thesis & Publications](#thesis--publications)
14. [License](#license)

---

## Mathematical Background

### The Rough Heston Model

The **Rough Heston** model (El Euch & Rosenbaum 2019) is a stochastic volatility
model where the variance process $v_t$ is driven by a **fractional Brownian motion**
with Hurst exponent $H \in (0, \tfrac{1}{2})$, making it *rough* (non-Markovian):

$$v_t = v_0 + \frac{1}{\Gamma(H + \tfrac{1}{2})} \int_0^t (t-s)^{H-\tfrac{1}{2}} \kappa(\theta - v_s)\,ds + \sigma \int_0^t (t-s)^{H-\tfrac{1}{2}} \sqrt{v_s}\,dW_s$$

The spot price follows:

$$\frac{dS_t}{S_t} = r\,dt + \sqrt{v_t}\,\left(\rho\,dW_t + \sqrt{1-\rho^2}\,dB_t\right)$$

where $W_t, B_t$ are independent Brownian motions and $\rho$ is the correlation.

**Parameters** (6 total):

| Symbol | Name | Range | Role |
|--------|------|--------|------|
| $\kappa$ | Mean reversion speed | $[0.5, 5.0]$ | How fast variance reverts to $\theta$ |
| $\theta$ | Long-run variance | $[0.01, 0.25]$ | Equilibrium variance level |
| $\sigma$ | Vol-of-vol | $[0.1, 1.5]$ | Volatility of the variance process |
| $\rho$ | Correlation | $[-0.95, 0.0]$ | Spot-vol correlation (leverage effect) |
| $V_0$ | Initial variance | $[0.01, 0.25]$ | Variance at $t=0$ |
| $H$ | Hurst exponent | $[0.04, 0.15]$ | Roughness (empirically $H \approx 0.1$) |


### The Lifted Heston Approximation

The fractional kernel makes exact simulation expensive. The **Lifted Heston** model
(Abi Jaber 2019) approximates the fractional integral with $N$ independent Markovian
factors using **Bernstein weights** $c_n$ and **time-scales** $x_n$:

$$v_t \approx \sum_{n=1}^{N} c_n Z_t^{(n)}, \qquad dZ_t^{(n)} = \left[\kappa(\theta - Z_t^{(n)}) - x_n Z_t^{(n)}\right] dt + \sigma \sqrt{v_t}\,dW_t$$

With $N=40$ factors this achieves error $< 1$ bp on the IV surface, recovering the
genuine fractional dynamics without path-dependent memory.

### Characteristic Function and Fourier-COS Pricing

The Rough Heston model has a **semi-analytical characteristic function**:

$$\log \phi(u, t) = \phi_0(u) + \kappa\theta \int_0^t h(u, s)\,ds$$

where $h(u,t)$ satisfies a fractional Riccati ODE solved numerically. Option prices
are computed via the **Fourier-COS method** (Fang & Oosterlee 2008):

$$C(K, T) \approx e^{-rT} \sum_{k=0}^{N_{\text{cos}}-1}{}' \text{Re}\left[\phi\!\left(\frac{k\pi}{b-a}\right) e^{-ik\pi a/(b-a)}\right] V_k$$

with $N_{\text{cos}} = 128$ terms. The GPU implementation prices a full 8×11 IV
surface (88 options) in **< 6 ms** using vectorized CUDA kernels.

### Fisher Information and Identifiability

The 5-parameter Heston system is poorly identified — the FIM condition number exceeds
$10^6$. We apply a **reparameterization** $(v_0, \zeta, \lambda)$ reducing it by
**1301×**, making gradient-based calibration far more stable.

## DeepVol Model Zoo

The DeepVol framework contains a diverse "Zoo" of 13 volatility and derivatives models, implemented with high-performance CPU/GPU solvers and integrated with real-time calibration:

1. **Classic Heston Model**: Traditional stochastic volatility model where variance follows a Cox-Ingersoll-Ross (CIR) process. Option pricing is performed using the exact Fourier-COS series expansion method.
2. **SABR Model (Hagan/Displaced)**: Industry-standard smile model for FX and interest rates. Supports normal (Bachelier) and lognormal (Black) implied volatility via Hagan (2002) asymptotic expansions with displacement shifts.
3. **SSVI (Surface SVI) Model**: Parameterizes the entire volatility surface using Gatheral's SVI slices under strict calendar and butterfly no-arbitrage guarantees.
4. **Local Volatility (Dupire SVI) Model**: Extract state-dependent local volatility $\sigma_{\text{loc}}(T, K)$ by applying Dupire's formula via finite differences to a calibrated SVI surface.
5. **Rough Bergomi (rBergomi) Model**: SOTA rough volatility model where variance is driven by a fractional Brownian motion ($H < 0.5$). Simulated on GPU using the Bennedsen-Lunde-Pakkanen (2017) hybrid convolution scheme.
6. **Neural SDE**: Data-driven generative pricing model where drift and diffusion coefficients of the SDE are parameterized by neural networks and calibrated via adjoint sensitivity.
7. **Signature Volatility**: Model path-dependency using rough path theory, where the asset volatility is represented as a linear function of the signature of historical prices.
8. **McKean-Vlasov SDE (MLSV) Model**: Local stochastic volatility model where the volatility coefficient depends on the probability distribution (marginal laws) of the spot process.
9. **Schwartz-Smith (2-Factor Commodity) Model**: Captures short-term mean-reverting deviations and long-term equilibrium price factors to model commodity futures curves and options.
10. **LMM-SABR Model**: Combines the Libor Market Model (LMM) with SABR stochastic volatility to model the evolution of interest rate forward curves and swaption cubes.
11. **Physics-Informed Meta-Learning FNO (PI-M-FNO)**: Physics-informed neural operator designed to price options under extreme, out-of-distribution market conditions with fast online GPU adaptation (< 10 ms).
12. **End-to-End Differentiable Calibration & Hedging (D-XVA)**: Unifies option pricing, implied volatility inversion (PIVOT), and recurrent deep hedging into a single computational graph trained directly on hedging variance.
13. **Grey Rough Bergomi (gRB) Model**: Generalizes rough Bergomi by replacing fractional Brownian motion with generalized grey Brownian motion, implemented via custom double-precision Mittag-Leffler and Wood-Chan circulant embedding CUDA kernels.

---

## Monte Carlo Risk Engine (VaR/ES)

DeepVol includes a high-performance **GPU-accelerated Monte Carlo Risk Engine** to compute portfolio-level risk metrics:

* **Value-at-Risk (VaR)**: The maximum expected loss at a given confidence level $\alpha$ (e.g., $95\%$ or $99\%$) over a time horizon $\Delta t$.
* **Expected Shortfall (ES)**: Also known as Conditional VaR (CVaR), measuring the average loss in the worst $(1-\alpha)\%$ tail scenarios.

### Methodology
1. **Scenario Generation**: Jointly simulate spot and variance paths under the Heston model using a stable **full-truncation Euler-Maruyama scheme** (Lord, Koekkoek & van Dijk 2010) on GPU to ensure non-negative variance:
   $$dX_t = (r - \tfrac{1}{2} V^+_t) dt + \sqrt{V^+_t} dW_1, \qquad dV_t = \kappa(\theta - V^+_t) dt + \sigma \sqrt{V^+_t} dW_2$$
2. **Accelerated Revaluation**: For each simulated path, the entire option portfolio is revalued using the fast FiLM-FNO pricing surrogate and 2D bilinear interpolation, enabling $10^7$ portfolio valuations per second.
3. **Loss Aggregation**: Compute the portfolio loss distribution, sorting to extract the VaR and Expected Shortfall at the targeted quantiles.

---

## Architecture

### FiLM-Conditioned Fourier Neural Operator

```
θ = (κ, θ, σ, ρ, V₀, H)
         │
         ▼
ParameterNormalizer (z-score to unit hypercube)
         │
         ▼
FiLM MLP: θ → (γ₁,β₁, γ₂,β₂, γ₃,β₃, γ₄,β₄)   [scale+shift per layer]
         │
         ▼
(T, K) grid (8×11) ──→ Mirror-pad (16×22) ──→ Lifting Conv (1→40 channels)
                                                      │
                    ┌─────── 4 × FourierLayer ────────┤
                    │   Spectral truncation (modes=8,11)│
                    │   FiLM(γ_i, β_i) modulation      │
                    └───────────────────────────────────┘
                                                      │
                                               Projection (40→1)
                                                      │
                                             IV surface (8×11)
                                                      │
                                         IVSurfaceNormalizer⁻¹
                                                      │
                                          σ_IV(T, K) in vol units
```

**Key design decisions:**

- **Mirror padding** on the T-axis: enforces put-call parity / calendar spread
  symmetry. Surface is padded from (8×11) to (16×22) before spectral convolution.
- **FiLM conditioning** (Perez et al. 2018): parameters θ control scale/shift of
  every Fourier layer, giving better generalization than concatenation.
- **Channels-last format**: grid is `(B, T, K, C)` — all modules expect this.
- **Martingale prior**: training loss penalizes surfaces that violate
  $E[e^{-rT}S_T] = S_0$, keeping the network in the no-arbitrage subspace.
- **ATM-weighted Huber loss**: down-weights OTM/deep ITM strikes where IV data is
  sparse and noisy.

### Calibration Pipeline

```
Market IV surface (8×11)
         │
         ▼
IVSurfaceNormalizer.normalize()
         │
   ┌─────▼──────────────────────────────────┐
   │  Gauss-Newton loop (max 20 iters)      │
   │  θ_new = θ - (JᵀJ + λI)⁻¹ Jᵀ r       │
   │  J = jacfwd(FNO, θ)  [exact, GPU]     │
   │  r = FNO(θ) - σ_market                │
   └─────┬──────────────────────────────────┘
         │  converged when RMSE < 50 bps
         ▼
ParameterNormalizer.denormalize()
         │
         ▼
θ* = (κ*, θ*, σ*, ρ*, V₀*, H*)
```


---

## Installation

### Requirements

- Python 3.9+
- CUDA 12.8+ and a compatible GPU (tested: RTX 3060/3080/4090, A100)
- PyTorch 2.8+ with CUDA support
- `uv` Python package manager (highly recommended)
- ~4 GB disk space (model weights + datasets)

### Setup

Using `uv` (fastest & recommended):
```bash
# 1. Clone the repository
git clone <repo-url>
cd derivatives

# 2. Sync dependencies and build virtual environment automatically
uv sync

# 3. (Optional) Build the CUDA extension for direct kernel access
uv run python setup.py build_ext --inplace

# 4. Verify installation
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
uv run pytest tests/test_pricing_engine.py -q
```

### One-click setup (Linux)

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
# Creates .venv, installs deps, launches Streamlit on port 8501
```

---

## Quick Start

### 1. Interactive Streamlit Dashboard

* **Main Calibration Dashboard**:
  ```bash
  source .venv/bin/activate
  streamlit run src/deepvol/app/app_v2.py
  # Opens http://localhost:8501
  ```
  The calibration dashboard has three tabs:
  - **IV Surface**: Adjust sliders for all 6 parameters, see the real-time FNO-predicted
    implied-volatility surface update in < 1 ms.
  - **Newton Calibration**: Upload or paste a market IV surface, run Gauss-Newton
    calibration, watch the parameter trajectory converge.
  - **Greeks**: Compute portfolio delta/gamma/vega for a set of positions.

* **Risk Management Dashboard**:
  ```bash
  source .venv/bin/activate
  streamlit run src/deepvol/app/app_v3_risk.py
  # Opens http://localhost:8501
  ```
  The risk dashboard provides:
  - **Live Risk Telemetry**: Streaming Greeks, latency timelines, and spot updates.
  - **Scenario Stress Testing**: Stress testing spot/vol shocks on 3D Greeks grids.
  - **Audit Logs**: Scrolling historical telemetry reports and CSV downloads.

### 2. Single Surface Pricing (Python API)

```python
import sys; sys.path.insert(0, 'src')
import torch
import numpy as np
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.calibration.calibrate_bfgs import _load_normalizers, _fno_predict_real_iv, _make_spatial_input
from deepvol.market.spx_data import T_GRID, K_GRID

# Load model and normalizers
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MirrorPaddedFNO2d(param_dim=6).to(device)
model.load_state_dict(torch.load("artifacts/weights/fno_v3_final_prod.pth", map_location=device, weights_only=True))
model.eval()
_load_normalizers("v3")  # must be called before any forward pass

# Parameters: κ=2.0, θ=0.04, σ=0.5, ρ=-0.7, V₀=0.04, H=0.1
theta = torch.tensor([[2.0, 0.04, 0.5, -0.7, 0.04, 0.1]], dtype=torch.float32, device=device)
spatial = _make_spatial_input(T_GRID, K_GRID, device)

with torch.no_grad():
    iv_surface = _fno_predict_real_iv(model, theta, spatial).squeeze().cpu().numpy()

print(iv_surface.shape)    # (8, 11) — implied vol in decimal (0.20 = 20%)
print(f"ATM 6M: {iv_surface[2, 5]*100:.1f}%")  # roughly 20-25% for typical params
```

### 3. Calibrate to a Market Surface

```python
import sys; sys.path.insert(0, 'src')
import torch
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.calibration.calibrate_newton import calibrate_newton_h
from deepvol.market.spx_data import T_GRID, K_GRID

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MirrorPaddedFNO2d(param_dim=6).to(device)
model.load_state_dict(torch.load("artifacts/weights/fno_v3_final_prod.pth", map_location=device, weights_only=True))
model.eval()

# Market IV surface: 8 maturities × 11 strikes (in decimal vol, e.g. 0.20 = 20%)
from deepvol.market.spx_data import download_spx_chain, clean_chain, to_iv_surface
from datetime import date

# Fetch or load market IV surface
df = download_spx_chain(date(2024, 1, 2))
cleaned = clean_chain(df)

S0 = 4700.0
r = 0.05
q = 0.015
market_iv = to_iv_surface(cleaned, S0, r, q)  # (8, 11) array

# Gauss-Newton calibration in (v₀, ζ=σρ, λ=σ√(1-ρ²), H) space
result = calibrate_newton_h(model, market_iv, T_GRID, K_GRID, max_iter=20, verbose=True)
print(f"RMSE: {result['rmse'] * 1e4:.1f} bps")
print(f"v0={result['v0']:.4f}  σ={result['sigma']:.3f}  ρ={result['rho']:.3f}  H={result['H']:.3f}")
print(f"converged: {result['converged']}")
```

---

## Notebooks

Forty-four self-contained Jupyter notebooks demonstrate the full pipeline end-to-end. They are located in the `notebooks/` directory.

Run them in order:

```bash
source .venv/bin/activate
cd notebooks
jupyter lab   # or: jupyter nbconvert --to notebook --execute --inplace *.ipynb
```

| Notebook | Purpose | Key Output |
|---|---|---|
| `01_spx_calibration.ipynb` | Full SPX calibration pipeline | RMSE=8.9 bps, H=0.113 |
| `02_surface_completion.ipynb` | SVI fit + arbitrage enforcement | Clean IV surface |
| `03_vix_analysis.ipynb` | VIX term structure from Rough Heston | Model vs market curve |
| `04_greeks_portfolio.ipynb` | Delta/gamma/vega/vanna/volga | Hedge portfolio |
| `05_crypto_calibration.ipynb` | Live BTC/ETH from Deribit | RMSE=1547 bps (v2) |
| `06_batch_calibration.ipynb` | Multi-date SPX batch + H dynamics | 12.8 bps median |
| `07_joint_calibration.ipynb` | Joint SPX+VIX calibration | converged=True |
| `08_heston_vs_rheston.ipynb` | Classic Heston vs Rough Heston pricing | Hurst parameter influence |
| `09_sabr_ssvi_calibration.ipynb` | SABR & SSVI calibration | Synthetic smiles |
| `10_local_vol_dupire.ipynb` | SVI-to-Dupire Local Volatility mapping | Clean LV surface |
| `11_rbergomi_calibration.ipynb` | Rough Bergomi HMC calibration | Model vs COS comparison |
| `12_neural_sde_calibration.ipynb` | SDE Adjoint calibration | SDE drift/diffusion prior |
| `13_signature_forecasting.ipynb` | Signature Volatility Forecasting | Out-of-sample smile forecast |
| `14_deep_hedging_european.ipynb` | Recurrent European Deep Hedging | Delta hedging variance reduction |
| `15_barrier_hedging_costs.ipynb` | Recurrent Barrier Deep Hedging | Optimal rebalancing corridors |
| `16_adversarial_market_gen.ipynb` | WGAN-GP and stylized facts alignment | Minimax robust generation |


### Batch Calibration (Multi-Date)

Calibrate Rough Heston to multiple historical dates in parallel. Results are saved
incrementally and the run is resume-capable.

```python
import sys; sys.path.insert(0, 'src')
from deepvol.calibration.batch_calibration import calibrate_batch, results_to_dataframe

# Calibrate SPX on 3 dates (fetches from yfinance, falls back to cached parquet)
results = calibrate_batch(
    dates=["2024-01-02", "2024-01-03", "2024-01-04"],
    currency="SPX",        # or "BTC", "ETH" (via Deribit)
    device="auto",         # "cuda" if available, else "cpu"
    max_workers=4,         # parallel data-fetch threads
    verbose=True,
)
# Output: [1/3] 2024-01-02 — RMSE=18.3 bps (541 ms)

df = results_to_dataframe(results)
print(df[["date", "kappa", "theta", "H", "rmse_bps", "converged"]])
```

### Joint SPX + VIX Calibration

Fit Rough Heston parameters to match *both* the SPX IV surface and the VIX futures
term structure simultaneously:

```python
import sys; sys.path.insert(0, 'src')
from deepvol.calibration.joint_calibration import calibrate_joint
from datetime import date

result = calibrate_joint(
    val_date=date(2024, 1, 2),
    w_spx=0.7,    # weight on SPX RMSE
    w_vix=0.3,    # weight on VIX curve MSE
)
print(f"SPX RMSE: {result['spx_rmse_bps']:.1f} bps")
print(f"VIX RMSE: {result['vix_rmse']:.4f}")
print(f"θ* = {result['params']}")
```

### VIX Futures Term Structure

Model the VIX futures curve from Rough Heston parameters:

```python
import sys; sys.path.insert(0, 'src')
from deepvol.market.vix_futures import fetch_vix_futures
from deepvol.market.vix_pricing import compute_vix_term_structure
from datetime import date
import numpy as np

# Fetch the VIX futures curve for a date (8 contracts)
df = fetch_vix_futures(date(2024, 1, 2))
print(df)
#    expiry  tenor_months  settle_vix
# 0  2024-01-17      1     13.2
# 1  2024-02-14      2     14.1
# ...

# Or compute model VIX from calibrated parameters
theta = np.array([2.0, 0.04, 0.5, -0.7, 0.04, 0.1])  # (κ,θ,σ,ρ,V₀,H)
vix_curve = compute_vix_term_structure(theta)
```

### Arbitrage-Free Surface Completion

Fill gaps in a sparse IV surface and enforce no-arbitrage constraints:

```python
import sys; sys.path.insert(0, 'src')
import numpy as np
from deepvol.arbitrage.surface_completion import (
    complete_sparse_surface,
    make_arbitrage_free,
    fit_svi_slice,
)

# Sparse surface: (T, K) grid with NaN where data is missing
sparse_iv = np.full((8, 11), np.nan)
sparse_iv[2, 4:8] = [0.18, 0.17, 0.19, 0.20]  # some observed strikes at T=0.6

# Complete the surface and enforce butterfly + calendar spread constraints
dense_iv = complete_sparse_surface(sparse_iv)
af_iv = make_arbitrage_free(dense_iv)
print("Max butterfly violation after:", np.max(np.diff(np.diff(af_iv, axis=1), axis=1)))
```

### Portfolio Greeks (GPU)

Compute delta, gamma, vega, vanna, volga for a portfolio of options and the
delta-hedging contracts needed:

```python
import sys; sys.path.insert(0, 'src')
import numpy as np
import torch
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer
from deepvol.greeks.portfolio_greeks import portfolio_greeks

# Load model (same as pricing above)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MirrorPaddedFNO2d(param_dim=6).to(device)
model.load_state_dict(torch.load("artifacts/weights/fno_v3_final_prod.pth", map_location=device))
model.eval()

pn = ParameterNormalizer.load("artifacts/models/param_normalizer_v3.npz")
yn = IVSurfaceNormalizer.load("artifacts/models/iv_normalizer_v3.npz")

# Define a portfolio of options
positions = [
    {"K": 100.0, "T": 0.5, "type": "call", "quantity":  10.0},
    {"K":  95.0, "T": 0.5, "type": "put",  "quantity": -20.0},
    {"K": 105.0, "T": 1.0, "type": "call", "quantity":   5.0},
]

# Calibrated parameters
theta = np.array([2.0, 0.04, 0.5, -0.7, 0.04, 0.1])

greeks = portfolio_greeks(positions, model, theta, pn, yn, S=100.0)
print(f"Portfolio Delta: {greeks['total_delta']:.4f}")
print(f"Portfolio Gamma: {greeks['total_gamma']:.6f}")
print(f"Hedge: sell {greeks['hedge_contracts']} ES futures")
print(f"Vega bucket (by maturity):\n{greeks['vega_bucket']}")
```


### P&L Attribution

Break down realized P&L into Greek components:

```python
import sys; sys.path.insert(0, 'src')
from deepvol.greeks.pnl_attribution import pnl_attribution

# Greeks computed before the move
greeks_before = {
    "total_delta": 0.45,
    "total_gamma": 0.012,
    "total_vanna": -0.003,
    "total_volga": 0.008,
    "vega_bucket": [0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
}

# Market move: spot 100→102, vol 20%→21%
breakdown = pnl_attribution(
    S_before=100.0, S_after=102.0,
    sigma_before=0.20, sigma_after=0.21,
    greeks=greeks_before,
)
print(f"Delta P&L:  {breakdown['delta_pnl']:.4f}")
print(f"Gamma P&L:  {breakdown['gamma_pnl']:.4f}")
print(f"Vega P&L:   {breakdown['vega_pnl']:.4f}")
print(f"Unexplained:{breakdown['residual_pnl']:.4f}")
```

### Hurst Exponent Dynamics Study

Run a historical calibration study to track how H changes over time:

```python
import sys; sys.path.insert(0, 'src')
from deepvol.analysis.hurst_dynamics import run_historical_study

# Resume-capable: already-calibrated dates are skipped
df = run_historical_study(
    start="2024-01-01",
    end="2024-03-31",
    currency="SPX",
    chunk_size=5,     # save every 5 dates
    device="auto",
)
# Results saved to results/hurst_dynamics/SPX_hurst_study.json

print(df[["date", "H", "kappa", "rmse_bps", "converged"]].head(10))
print(f"Mean H: {df['H'].mean():.4f}")  # typically ~0.08-0.10 for SPX
```

### Live Deribit Streaming

Stream real-time BTC/ETH IV surfaces from Deribit via WebSocket:

```python
import asyncio
import sys; sys.path.insert(0, 'src')
from deepvol.market.deribit_ws import DeribitWebSocket

async def main():
    async with DeribitWebSocket(currency="BTC") as ws:
        async for surface in ws.stream_iv_surface():
            print(f"BTC ATM IV (1M): {surface['iv_1m']:.4f}")
            # Calibrate here in real-time...

asyncio.run(main())
```

---

## REST API

Start the FastAPI server and use it to calibrate from any language:

```bash
source .venv/bin/activate
cd /path/to/derivatives
uvicorn deepvol.api.server:app --reload --port 8000 --app-dir src
# OpenAPI docs → http://localhost:8000/docs
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health, model version, device |
| `POST` | `/iv_surface` | FNO inference: θ → IV surface |
| `POST` | `/greeks` | Portfolio Greeks for a set of positions |
| `GET` | `/vix` | VIX futures term structure for a date |
| `GET` | `/deribit/snapshot` | Live BTC/ETH IV surface snapshot |
| `POST` | `/calibrate` | Full calibration: market IV → θ* |
| `POST` | `/calibrate_neural_sde` | Adjoint calibration of non-parametric Neural SDE prior |
| `POST` | `/predict/signature_vol` | Signature Volatility weekly options smile forecasting |
| `POST` | `/hedge/simulate` | Recurrent Deep Hedging delta-hedging simulation |

### Example: POST /iv_surface

```bash
curl -X POST http://localhost:8000/iv_surface \
  -H "Content-Type: application/json" \
  -d '{
    "kappa": 2.0, "theta": 0.04, "sigma": 0.5,
    "rho": -0.7, "v0": 0.04, "H": 0.1
  }'
# Response: {"iv_surface": [[0.195, 0.182, ...], ...], "shape": [8, 11]}
```

### Example: POST /greeks

```bash
curl -X POST http://localhost:8000/greeks \
  -H "Content-Type: application/json" \
  -d '{
    "positions": [{"K": 100.0, "T": 0.5, "type": "call", "quantity": 10}],
    "theta": [2.0, 0.04, 0.5, -0.7, 0.04, 0.1],
    "S": 100.0
  }'
# Response: {"total_delta": 5.42, "total_gamma": 0.12, "hedge_contracts": -5, ...}
```

---

## Benchmarks

All benchmarks are in `benchmarks/`. Run them from the repo root:

```bash
source .venv/bin/activate
# FNO v2 accuracy validation (MAE, R², NaN rate)
python benchmarks/validate_fno_v2.py

# FNO vs CUDA Monte Carlo speed comparison
python benchmarks/vs_cuda_mc.py

# Newton vs L-BFGS noise robustness
python benchmarks/noise_robustness.py

# Delta hedging backtest
python benchmarks/greeks_hedge_backtest.py

# Streaming calibration throughput demo
python benchmarks/streaming_calibration_demo.py
```

### Summary of Results

| Benchmark | Result | File |
|-----------|--------|------|
| FNO v2 accuracy | R²=0.9991, MAE=5.8bp, NaN=0.04% | `validate_fno_v2.py` |
| Inference speed | 4ms/batch vs 5600ms COS | `vs_cuda_mc.py` |
| Newton convergence | 15 iters, 541ms, RMSE < 50bp | `noise_robustness.py` |
| Streaming p95 | 668ms for 20 ticks | `streaming_calibration_demo.py` |
| Delta hedge | +5.1% variance reduction vs B-S | `greeks_hedge_backtest.py` |
| H convergence | N=40 factors → < 1bp surface error | `convergence_N_factors.py` |


---

## Test Suite

**1008 tests passing, 0 failing, 2 skipped (integration only).**

```bash
# Full suite (~8 min)
uv run pytest tests/ -q

# By module
pytest tests/test_pricing_engine.py         -v  # Fourier-COS pricer
pytest tests/test_normalizers.py            -v  # Normalizer roundtrips
pytest tests/test_calibrate_newton.py       -v  # Newton calibrator
pytest tests/test_calibrate_newton_h.py     -v  # Learnable-H calibrator
pytest tests/test_batch_calibration.py      -v  # Multi-date batch
pytest tests/test_joint_calibration.py      -v  # Joint SPX+VIX
pytest tests/test_surface_completion.py     -v  # SVI + arbitrage
pytest tests/test_spx_data.py               -v  # SPX market data
pytest tests/test_vix_pricing.py            -v  # VIX model
pytest tests/test_vix_term_structure.py     -v  # VIX futures curve
pytest tests/test_vix_pricing_stress.py     -v  # VIX adversarial
pytest tests/test_deribit_data.py           -v  # Deribit REST
pytest tests/test_deribit_ws.py             -v  # Deribit WebSocket
pytest tests/test_variance_swaps.py         -v  # Variance/vol swaps
pytest tests/test_portfolio_greeks.py       -v  # Greeks accuracy
pytest tests/test_portfolio_greeks_adversarial.py -v  # Greeks robustness
pytest tests/test_portfolio_greeks_stress.py -v # Greeks large portfolios
pytest tests/test_pnl_attribution.py        -v  # P&L breakdown
pytest tests/test_greeks_benchmark.py       -v  # Greeks timing
pytest tests/test_api.py                    -v  # FastAPI endpoints
pytest tests/test_hurst_dynamics.py         -v  # Hurst study (2 skipped)

# Integration tests (require live yfinance data)
INTEGRATION=1 pytest tests/test_hurst_dynamics.py -v -k "convergence"
```

---

## Models & Weights

| Model | Weights | Params | R² | MAE | Notes |
|-------|---------|--------|-----|-----|-------|
| FNO v2 | `artifacts/weights/fno_v2_final_prod.pth` | 2.2M | 0.9991 | 0.058% | H fixed=0.08 |
| FNO v3 | `artifacts/weights/fno_v3_final_prod.pth` | 2.2M | 0.9981 | 0.264% | H learnable |

Normalizers:
```
artifacts/models/param_normalizer_v2.npz   # 5-parameter (κ,θ,σ,ρ,V₀)
artifacts/models/param_normalizer_v3.npz   # 6-parameter (adds H)
artifacts/models/iv_normalizer_v2.npz
artifacts/models/iv_normalizer_v3.npz
```

Legacy weights (v1, diff-FNO, LSTM) are in `artifacts/legacy/` — not used in
production but kept for reproducibility.

---

## Project Structure

```
derivatives/
├── src/                              Core library
│   └── deepvol/                      Main module package
│       ├── surrogates/               FiLM-FNO model, EGNO, Signature SDE, Greeks, etc.
│       ├── models/                   Stochastic vol models (Classic Heston, SABR, SSVI, Dupire, rBergomi, MLSV, etc.)
│       ├── calibration/              Autograd Newton, Joint, RKHS MLSV calibrators
│       ├── market/                   Data integration (spx_data.py, deribit_ws.py, etc.)
│       ├── arbitrage/                SVI & arbitrage completion (surface_completion.py)
│       ├── greeks/                   Greeks & PnL attribution (portfolio_greeks.py, pnl_attribution.py)
│       ├── analysis/                 Hurst dynamics & research (hurst_dynamics.py, cross_asset_roughness.py)
│       ├── api/                      FastAPI REST server (server.py, websocket.py)
│       ├── app/                      Streamlit dashboards (app_v2.py, app_v3_risk.py, etc.)
│       ├── mrm/                      Model Risk Guardian and particle fallbacks
│       ├── deploy/                   TensorRT compilation and K8s configuration
│       └── utils/                    Common utilities (strikes.py, etc.)
│
├── notebooks/                        Jupyter notebooks (end-to-end demos)
├── tests/                            pytest suite (955+ passing)
├── benchmarks/                       Performance and accuracy studies
├── scripts/                          Utility scripts (batch VIX, plot generation)
├── data/                             Training datasets and market cache (gitignored)
├── artifacts/
│   ├── weights/                      Production model weights (.pth)
│   └── models/                       Normalizer files (.npz)
├── tex/
│   ├── thesis/main.pdf               51-page LaTeX thesis
│   └── presentation/presentation.pdf Defence slides
├── results/                          Saved calibration outputs (JSON)
├── research/                         Research notes, PDFs, SOTA survey
└── articles/                         Reference academic papers (PDF)
```

---

## Thesis & Publications

The full **51-page thesis** is at [`tex/thesis/main.pdf`](tex/thesis/main.pdf).

**Chapters:**
1. Introduction — rough volatility motivation, related work
2. Mathematical foundations — fBm, Rough Heston, Riccati ODE, Lifted Heston
3. GPU Fourier-COS pricing — Bernstein weights, exponential midpoint integrator
4. FiLM-FNO architecture — operator learning, mirror padding, training
5. Experiments — 7 sections: accuracy, speed, calibration, FIM, Newton, streaming,
   delta hedging
6. Conclusion and future work

**Build the thesis:**

```bash
cd tex/thesis
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
# PDF written to tex/thesis/main.pdf
```

**Presentation slides:** `tex/presentation/presentation.pdf`

---

## Completed Phases

| Phase | Status | Key Results |
|-------|--------|-------------|
| **P1**: FNO Surrogate | Complete | FNO v1/v2/v3, FIM reparameterization, Newton calibrator, Streamlit |
| **P2**: Market Extensions | Complete | FastAPI, VIX futures, Deribit streaming, variance swaps, batch calibration |
| **P3**: GPU-Native | Complete | GPU Gauss-Newton, SVI arbitrage enforcement, portfolio Greeks, P&L attribution |
| **P4**: Model Zoo | Complete | Classic Heston, SABR & SSVI, Local Volatility, and Rough Bergomi on GPU |
| **P5**: Neural SDE & Signature | Complete | Lifted Heston factor study, Neural SDE prior + adjoint, and Signature smile forecasting |
| **P6**: Recurrent Deep Hedging | Complete | Recurrent Deep Hedging (European/DOBC Barrier LSTM) and Minimax GAN market generation |
| **P7**: Quant Hardening | Complete | Stability fixes, call-put parity checks, boundary constraints, Feller tests |
| **P8**: Production Orchestration | Complete | High-throughput FastAPI, multi-model dispatch layer, unified configuration |
| **P9**: Differentiable FNO | Complete | JVP pricing loss, calendar/butterfly hard-constrained projection layers |
| **P10**: Model Risk Governance | Complete | Greek Adjoints VRAM optimization, Model Risk Guardian particle fallbacks, PSI drift logs |
| **P11**: Backtesting & Costs | Complete | Frictional env trading cost loops, Whalley-Wilmott delta hedging corridor |
| **P12**: Production Deployment | Complete | TensorRT compilation, Kubernetes Helm deployments, 3D Streamlit visualizer |
| **P13**: Advanced Vol Modeling | Complete | EGNO multi-asset graph surrogates, Joint SPX/VIX signature SDEs, RKHS Landmark MLSV |
| **P14**: Meta-Adaptation | Complete | Physics-Informed Meta-Learning FNO, Reptile/FOMAML online GPU adaptation, Model Risk Guardian |
| **P15**: Differentiable Hedging | Complete | Recurrent Deep Hedging (LSTM/GRU), D-XVA loss, PIVOT implied vol solver, low-vega gating |
| **P16**: Grey Rough Bergomi | Complete | C++/CUDA gRB path simulator, Mittag-Leffler CUDA kernel, uncertainty-guided active learning, Sobolev FNO |


---

## License

Research code — МФТИ ФПМИ Master's thesis project, 2026.
Contact author for usage permissions.

**Key references:**
- El Euch & Rosenbaum (2019) — *The characteristic function of rough Heston models*
- Abi Jaber (2019) — *Lifting the Heston model*
- Horvath, Muguruza & Tomas (2021) — *Deep Learning Volatility*
- Fang & Oosterlee (2008) — *A novel pricing method for European options*
- Perez et al. (2018) — *FiLM: Visual Reasoning with a General Conditioning Layer*
