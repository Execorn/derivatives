# DeepVol: Deep Rough Heston Calibration & Structured Derivatives Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16+-black?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-1550%2B%20passed-brightgreen?style=for-the-badge)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Master's Thesis Project** — МФТИ ФПМИ, Кафедра БИТ, 2026
> *Нейронные сети в ценообразовании производных финансовых инструментов*

An end-to-end GPU-accelerated quantitative platform for **real-time calibration**,
**deep hedging**, and **structured & cross-asset derivatives pricing** under rough stochastic volatility
(El Euch, Gatheral & Rosenbaum 2019), multi-asset jump dynamics, and triangulated FX/commodity smiles.

The system combines:
1. **FiLM-conditioned Fourier Neural Operators (FNO)** pricing full 8×11 implied-volatility surfaces in under 1 ms.
2. **SOTA Multi-Fidelity Autocallable Engines**: Crank-Nicolson Gatheral PDE base combined with a 5-member residual `CorrectionEnsemble` MLP delivering 0.99 bps accuracy with epistemic uncertainty estimation and SR 26-2 compliant model governance.
3. **Triangulated FX/Commodity Cross ML Pricing Engine (`SPEC-DEEPVOL-FX-CROSS-003`)**: $\beta$-split H1 analytical triangulation paired with a 5-member Sobolev DML residual MLP, 2nd-order exact Taylor jets ($s_k, s_{kk}$), CUDA-graph execution ($0.196\text{ ms}$ single-quote full Greeks), and Apache Arrow IPC zero-copy batch streaming ($9.45\text{ ms}$ for 10,000 contracts).
4. **End-to-End Differentiable Risk & Deep Hedging**: PIVOT implied volatility solver, recurrent LSTM/GRU hedging policies under market frictions, and automated model risk fallback routing.
5. **Full-Stack Institutional Delivery**: High-throughput FastAPI REST, Arrow IPC & WebSocket streaming services, Next.js 16+ Turbopack interface with interactive WebGL 3D surfaces, and multi-page Streamlit research dashboards.

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
| Autocall Multi-Fidelity Ensemble (Phase D) | **0.99 bps trimmed RMSE (1.14 bps raw)**, 5-member MLP + PDE base |
| Autocall Epistemic Uncertainty OOD | Calibrated rejection threshold $\tau_{\text{OOD}} = 2.17$ bps |
| Autocall Batch GPU Pricing Latency | **~4.5 ms** (batch 1024) vs ~28 s MC — **6000× speedup** |
| Autocall Sobol-Brownian Bridge QMC | **< 12 ms** (50,000 paths, 252 steps, Milstein) |
| Worst-Of (WoF) Basket Autocalls (EGNO) | **1.87 bps RMSE**, exact permutation equivariance |
| Phoenix Memory Coupon Autocalls | **1.35 bps RMSE**, multi-barrier conditional payout |
| FX/Commodity Cross ML Surrogate (P19) | **0.67 bp Price MAE (0.895 bp stress)**, 5-member Sobolev DML + $\beta$-split H1 |
| FX/Commodity Cross Full-Greeks Latency | **0.196 ms p50 / 0.214 ms p99** (GPU single quote), **9.45 ms p99** (10k Arrow batch) |
| FX/Commodity Cross Arbitrage Audit | **0 violations** across 200,000 valuations ($100\times 100$ grid $\times$ 20 families) |

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
14. [Completed Phases](#completed-phases)
15. [License](#license)

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

The DeepVol framework contains a diverse "Zoo" of 17 volatility and derivatives models, implemented with high-performance CPU/GPU solvers and integrated with real-time calibration:

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
14. **Autocallable Structured Notes (Multi-Fidelity PDE + Ensemble MLP)**: Crank-Nicolson Gatheral PDE base solver combined with a 5-member residual `CorrectionEnsemble` MLP in 19-dimensional barrier-aware feature space, delivering sub-basis-point accuracy (0.99 bps RMSE) with epistemic uncertainty estimation and SR 26-2 model governance.
15. **Phoenix Memory Coupon Autocallables**: Structured notes with conditional memory coupons, knock-out barriers, and European/American protection barriers, accelerated via specialized residual MLP surrogates.
16. **Worst-Of (WoF) Basket Autocallables (EGNO)**: High-dimensional multi-asset basket notes priced via Equivariant Graph Neural Operators (EGNO) enforcing exact permutation equivariance across underlying assets with spectral PSD projection.
17. **Triangulated FX/Commodity Cross ML Engine ($\beta$-Split H1 + Sobolev DML Ensemble + Taylor Jets)**: Prices European options and full smile-inclusive Greeks on synthetic crosses $S^\times = S_1 S_2$ (e.g., XAU/RUB = XAU/USD $\times$ USD/RUB) using a 6-step $\beta$-split H1 analytical base, a 5-member $13\to 64^3\to 1$ SiLU residual ensemble trained via Differential Machine Learning (DML) against a 1601-node Gaussian copula teacher under $\mathbb{Q}^B$, exact 2nd-order Taylor jet propagation ($s_k, s_{kk}$), $C^2$ quintic wing tapering, and SR 11-7 / SR 26-2 Mahalanobis OOD routing.

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

### Autocallable Multi-Fidelity Correction Architecture

Structured autocallable contracts feature discontinuous payoff boundaries across discrete observation dates, creating localized pricing gradients that challenge monolithic neural networks. DeepVol deploys a **hybrid multi-fidelity architecture** combining an analytical PDE prior with an epistemic deep ensemble:

```
Contract Specifications (19-dim: S, K, B_autocall, B_dip, coupon, vol, r, q, tenors)
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
   Crank-Nicolson Gatheral PDE         19D Feature Normalizer
         (Low-Fidelity Base)                    │
                │                               ▼
                │                   5-Member Correction MLP Ensemble
                │                      (Residual Blocks + Dropout)
                │                               │
                │               ┌───────────────┴───────────────┐
                │               ▼                               ▼
                │       Mean Residual ΔV            Epistemic Uncertainty σ_bps
                │               │                               │
                └───────► ( + ) ◄───────────────────────────────┘
                                │
                                ▼
                     AutocallModelGuardian (SR 26-2)
                     - Feature Boundary & OOD Checks
                     - Epistemic Uncertainty Gate (σ_bps > 2.17)
                     - Payoff Monotonicity & Arbitrage Audit
                                │
                   ┌────────────┴────────────┐
                   │ Passed                  │ Breached
                   ▼                         ▼
        Ensemble Valuation         Analytical PDE / GPU-MC
        (NPV ± 2σ, Greeks)                 Fallback
```

**Key components:**
- **Gatheral Crank-Nicolson PDE Base**: Solves the 1D Dupire/Gatheral boundary PDE with discrete jump conditions at coupon dates, capturing the macro structure of early redemption.
- **5-Member `CorrectionEnsemble` MLP**: Each member features 4 ResidualBlocks with 256 hidden units trained under Sobolev loss regularization to fit the residual $\Delta V = V_{\text{QMC}} - V_{\text{PDE}}$.
- **Epistemic Uncertainty Quantification**: Ensemble variance $\sigma_{\text{bps}}$ estimates surrogate uncertainty; queries with $\sigma_{\text{bps}} > \tau_{\text{OOD}} = 2.17$ bps are automatically flagged.
- **SR 26-2 / SR 11-7 Model Guardian**: Enforces population drift monitoring (PSI), bounds checking, and deterministic fallback routing to high-precision GPU Monte Carlo.

### Full-Stack System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Interfaces & Clients                       │
│                                                                        │
│   Next.js 16+ Turbopack App               Streamlit Quantitative Suite │
│   - Interactive 3D WebGL Surface          - app_main.py (Multi-Page)   │
│   - Epistemic Uncertainty Gauge           - autocall_pricer.py         │
│   - Live WebSocket Risk Telemetry         - p16_grey_bergomi.py        │
│   (http://localhost:3000)                 (http://localhost:8501)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / WebSocket
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       FastAPI Orchestration Layer                      │
│                                                                        │
│   /autocall/price           /ws/risk (ConflatedQueue)    /iv_surface   │
│   /autocall/price_batch     /greeks/{model}              /calibrate    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   DeepVol Quantitative Computing Core                  │
│                                                                        │
│   Surrogates & Ensembles          Analytical & PDE       GPU Kernels   │
│   - FiLM-FNO (2D Modes 8x11)      - Gatheral Crank-Nic   - Quasi-MC    │
│   - CorrectionEnsemble (K=5)      - Fourier-COS (N=128)  - Wood-Chan   │
│   - EGNO Permutation Equivariance - SVI Arbitrage Free   - Mittag-L    │
└────────────────────────────────────────────────────────────────────────┘
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

### 1. Web and Research Interfaces

* **Next.js Production Web Application**:
  ```bash
  cd app_frontend
  npm install
  npm run dev
  # Main interface at http://localhost:3000 (3D surface visualizer & streaming telemetry)
  # Structured products pricer at http://localhost:3000/autocall (PDE ensemble & SR 26-2 compliance)
  ```

* **Streamlit Quantitative Research Suite**:
  ```bash
  source .venv/bin/activate
  streamlit run src/deepvol/app/app_main.py
  # Launches the unified multi-page portal at http://localhost:8501
  ```
  Specialized subpages include `autocall_pricer.py` (multi-fidelity PDE ensemble), `phoenix_pricer.py`, `wof_autocall_pricer.py`, `p16_grey_bergomi.py` (fractional path simulation), `p15_dxva_hedging.py`, `p14_meta_adaptation.py`, and `var_es_risk.py`.

### 2. Autocallable Pricing with Epistemic Uncertainty (Python API)

Price structured autocalls using the hybrid Gatheral PDE base and 5-member residual ensemble with SR 26-2 risk validation:

```python
import sys; sys.path.insert(0, 'src')
import torch
from deepvol.surrogates.correction_ensemble import CorrectionEnsemble
from deepvol.mrm.autocall_guardian import AutocallGuardian

# Initialize ensemble (loads 5 members + PDE base + normalizers)
ensemble = CorrectionEnsemble.load_default(device="cuda" if torch.cuda.is_available() else "cpu")
guardian = AutocallGuardian(ensemble)

# Contract parameters (S0, K, B_autocall, B_dip, coupon, vol, r, q, tenors)
contract = {
    "spot": 100.0,
    "strike": 100.0,
    "autocall_barrier": 105.0,
    "dip_barrier": 70.0,
    "coupon_rate": 0.08,
    "volatility": 0.20,
    "rate": 0.03,
    "dividend": 0.01,
    "maturity": 2.0,
    "obs_freq": 0.5,
}

result = guardian.price_contract(contract)
print(f"Price (NPV): {result['price']:.4f}")
print(f"Epistemic Uncertainty: {result['uncertainty_bps']:.2f} bps")
print(f"PDE Base: {result['pde_base']:.4f}, Residual Correction: {result['correction']:.4f}")
print(f"Routing Decision: {result['routing']} (Fallback Active: {result['fallback_active']})")
```

### 3. Single Surface Pricing (Python API)

```python
import sys; sys.path.insert(0, 'src')
import torch
import numpy as np
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.calibration.calibrate_bfgs import _load_normalizers, _fno_predict_real_iv, _make_spatial_input
from deepvol.market.spx_data import T_GRID, K_GRID

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MirrorPaddedFNO2d(param_dim=6).to(device)
model.load_state_dict(torch.load("artifacts/weights/fno_v3_final_prod.pth", map_location=device, weights_only=True))
model.eval()
_load_normalizers("v3")

# Parameters: κ=2.0, θ=0.04, σ=0.5, ρ=-0.7, V₀=0.04, H=0.1
theta = torch.tensor([[2.0, 0.04, 0.5, -0.7, 0.04, 0.1]], dtype=torch.float32, device=device)
spatial = _make_spatial_input(T_GRID, K_GRID, device)

with torch.no_grad():
    iv_surface = _fno_predict_real_iv(model, theta, spatial).squeeze().cpu().numpy()

print(f"IV Surface shape: {iv_surface.shape}")
print(f"ATM 6M Volatility: {iv_surface[2, 5]*100:.2f}%")
```

### 4. Calibrate to a Market Surface

```python
import sys; sys.path.insert(0, 'src')
import torch
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.calibration.calibrate_newton import calibrate_newton_h
from deepvol.market.spx_data import T_GRID, K_GRID, download_spx_chain, clean_chain, to_iv_surface
from datetime import date

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MirrorPaddedFNO2d(param_dim=6).to(device)
model.load_state_dict(torch.load("artifacts/weights/fno_v3_final_prod.pth", map_location=device, weights_only=True))
model.eval()

cleaned = clean_chain(download_spx_chain(date(2024, 1, 2)))
market_iv = to_iv_surface(cleaned, S0=4700.0, r=0.05, q=0.015)

result = calibrate_newton_h(model, market_iv, T_GRID, K_GRID, max_iter=20, verbose=True)
print(f"RMSE: {result['rmse'] * 1e4:.1f} bps, Converged: {result['converged']}")
print(f"Calibrated: v0={result['v0']:.4f}, σ={result['sigma']:.3f}, ρ={result['rho']:.3f}, H={result['H']:.3f}")
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
| `GET` | `/health` | Server health, model versions, and active execution device |
| `POST` | `/iv_surface` | FNO inference: parameter vector θ → implied volatility surface |
| `POST` | `/greeks/{model_name}` | Multi-model per-strike Greeks surface (`rough_heston`, `heston`, `sabr`, `ssvi`, `rbergomi`) |
| `POST` | `/greeks` | Portfolio-level Greeks and hedge contract recommendations |
| `POST` | `/vix` | VIX futures term structure for a given valuation date |
| `GET` | `/deribit/snapshot` | Live BTC/ETH implied volatility surface snapshot |
| `POST` | `/calibrate/{model_name}` | Gauss-Newton / LM model calibration to market surface |
| `POST` | `/calibrate_neural_sde` | Adjoint calibration of non-parametric Neural SDE prior |
| `POST` | `/predict/signature_vol` | Signature Volatility weekly options smile forecasting |
| `POST` | `/hedge/simulate` | Recurrent Deep Hedging delta-hedging trajectory simulation |
| `POST` | `/autocall/price` | Autocall pricing via Crank-Nicolson PDE base + 5-member residual ensemble with SR 26-2 guardian routing |
| `POST` | `/autocall/price_batch` | Vectorized batch autocall pricing (up to 100 contracts per payload) |
| `POST` | `/fx_cross/price` | Triangulated FX/Commodity cross single-option pricing + full smile-inclusive Greeks (CUDA graph B=2 / CPU fallback) |
| `POST` | `/fx_cross/price_batch` | Columnar batch FX/Commodity cross pricing (up to 10,000 contracts & 64 market states; JSON or Apache Arrow IPC) |
| `GET` | `/fx_cross/health` | FX/Commodity cross engine readiness, CUDA graph capture state, active `model_version`, and degraded status |
| `WS` | `/ws/risk` | Streaming WebSocket risk telemetry with `ConflatedQueue` backpressure control |

### Example: POST /autocall/price

```bash
curl -X POST http://localhost:8000/autocall/price \
  -H "Content-Type: application/json" \
  -d '{
    "spot": 100.0,
    "strike": 100.0,
    "autocall_barrier": 105.0,
    "dip_barrier": 70.0,
    "coupon_rate": 0.08,
    "volatility": 0.20,
    "rate": 0.03,
    "dividend": 0.01,
    "maturity": 2.0,
    "obs_freq": 0.5
  }'
# Response:
# {
#   "price": 0.9842,
#   "uncertainty_bps": 1.14,
#   "pde_base": 0.9810,
#   "correction": 0.0032,
#   "delta": 0.4512,
#   "vega": 0.1840,
#   "barrier_delta": -0.0125,
#   "routing": "ensemble",
#   "fallback_active": false
# }
```

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

**1126 tests passing, 0 failing, 2 skipped (integration only).**

```bash
# Run full test suite on CUDA
uv run pytest tests/ -q

# Core pricing and surrogates
pytest tests/test_pricing_engine.py                 -v  # Fourier-COS pricer
pytest tests/test_fno_model.py                      -v  # FiLM-FNO architecture
pytest tests/test_normalizers.py                    -v  # Parameter and IV normalizers
pytest tests/test_calibrate_newton.py               -v  # Gauss-Newton calibration
pytest tests/test_calibrate_newton_h.py             -v  # Learnable Hurst calibration

# Structured derivatives and autocallables
pytest tests/test_autocall_pde.py                   -v  # Gatheral Crank-Nicolson PDE base
pytest tests/test_autocall_mlp.py                   -v  # 5-member residual MLP ensemble
pytest tests/test_autocall_guardian.py              -v  # SR 26-2 OOD gating and fallbacks
pytest tests/test_api_autocall.py                   -v  # Autocall REST and batch endpoints
pytest tests/test_wof_autocall.py                   -v  # Worst-of basket autocalls

# Triangulated FX/Commodity cross engine & forensic audit suite
pytest tests/test_fx_cross_analytic.py              -v  # Beta-split H1 triangulation & closed-form Greeks
pytest tests/test_taylor_jets.py                    -v  # 2nd-order exact Taylor jet propagation
pytest tests/test_fx_cross_surrogate.py             -v  # 5-member Sobolev DML residual ensemble
pytest tests/test_api_fx_cross.py                   -v  # FastAPI JSON & Apache Arrow IPC endpoints
pytest tests/audit/ -m audit                        -v  # 141-test clean-room forensic audit suite

# High-throughput API and streaming
pytest tests/test_api.py                            -v  # FastAPI REST endpoints
pytest tests/test_websocket_server.py               -v  # WebSocket risk telemetry & conflation

# Deep hedging and market dynamics
pytest tests/test_deep_hedging.py                   -v  # Recurrent neural hedging
pytest tests/test_d_xva.py                          -v  # D-XVA differentiable backpropagation
pytest tests/test_grey_bergomi.py                   -v  # Generalized grey Brownian paths
```

---

## Models & Weights

| Category | Model | Weights File | Parameters | Metric | Description |
|---|---|---|---|---|---|
| Surrogates | FNO v2 | `artifacts/weights/fno_v2_final_prod.pth` | 2.2M | MAE 5.8 bps | Fixed H=0.08 Rough Heston |
| Surrogates | FNO v3 | `artifacts/weights/fno_v3_final_prod.pth` | 2.2M | MAE 26.4 bps | Learnable H ∈ [0.04, 0.15] |
| Surrogates | FNO Grey | `artifacts/weights/fno_grey_final_prod.pth` | 2.2M | MAE 0.76 bps | Grey Rough Bergomi |
| Surrogates | FNO Zoo | `artifacts/weights/fno_{heston,sabr,ssvi,rbergomi,localvol}_final_prod.pth` | 2.2M ea. | R² > 0.998 | Model Zoo surrogates |
| Autocalls | Correction Ensemble | `artifacts/weights/autocall_correction_mlp_member_{0..4}.pth` | 2.1M ea. | RMSE 0.99 bps | 5-member residual MLP ensemble |
| Autocalls | Phoenix MLP | `artifacts/weights/phoenix_mlp_best.pth` | 3.2M | RMSE 1.35 bps | Memory coupon structured notes |
| Autocalls | WoF EGNO | `artifacts/weights/wof_mlp_best.pth` | 1.6M | RMSE 1.87 bps | Worst-of basket autocalls |
| FX Cross | ResidualCrossVolEnsemble | `artifacts/fx_cross/checkpoints/member_{0..4}.pt` | 9.2K ea. | MAE 0.67 bps | 5-member Sobolev DML residual cross vol ensemble |
| Hedging | Deep Hedger | `artifacts/weights/deep_hedger_{european,barrier}_prod.pth` | 94K ea. | P&L std 0.43 | Recurrent LSTM policy networks |
| Markets | Minimax GAN | `artifacts/weights/minimax_policy_prod.pth` | 92K | Error 3e-4 | Adversarial stylized facts generator |

Normalizers and Calibration Configurations:
```
artifacts/scalers/correction_input_normalizer.npz    # 19-dim contract feature normalizer
artifacts/scalers/correction_output_normalizer.npz   # Residual target scaler
artifacts/weights/ensemble_calibration.json          # Calibrated uncertainty thresholds (tau_OOD = 2.17 bps)
artifacts/models/param_normalizer_v3.npz             # Rough Heston parameter scaler
artifacts/models/iv_normalizer_v3.npz                # Implied volatility surface normalizer
artifacts/models/{param,iv}_normalizer_{grey,heston,sabr,ssvi,rbergomi,localvol}.npz
```

---

## Project Structure

```
derivatives/
├── app_frontend/                     Next.js 16+ Turbopack institutional web application
│   └── src/app/
│       ├── page.tsx                  Interactive 3D WebGL surface viewer & risk telemetry
│       └── autocall/page.tsx         Autocall pricer with epistemic uncertainty gauge
│
├── src/                              Core Python library
│   └── deepvol/
│       ├── surrogates/               FiLM-FNO, CorrectionEnsemble, EGNO, Signature SDE
│       ├── models/                   Stochastic vol & cross models (Heston, SABR, rBergomi, gRB, Autocalls, FX Cross)
│       ├── calibration/              Gauss-Newton, Joint SPX+VIX, and MLSV particle calibrators
│       ├── market/                   Data feeds, FX delta/SVI conventions, and legacy DAG adapters
│       ├── arbitrage/                SVI and no-arbitrage surface projection
│       ├── greeks/                   GPU Greeks and Greek P&L attribution
│       ├── hedging/                  Deep hedging, frictional trading envs, D-XVA loss
│       ├── mrm/                      Model Risk Guardians (SR 11-7 / SR 26-2) and analytical fallbacks
│       ├── api/                      FastAPI REST server, Arrow IPC codec, CUDA-graph runtime, WebSocket
│       ├── app/                      Streamlit research hub (app_main.py) and 14 specialized pages
│       └── deploy/                   TensorRT compilation and ONNX export
│
├── tests/                            pytest suite (unit, adversarial, and tests/audit/ clean-room suite)
├── notebooks/                        44 Jupyter notebooks covering theory and experiments
├── benchmarks/                       Performance benchmarks and latency profiling
├── artifacts/
│   ├── weights/                      Production neural network weights (.pth)
│   ├── fx_cross/                     FX Cross ensemble checkpoints, model card, and OOD statistics
│   ├── audit/                        Forensic audit report, Wolfram/BrainyCalc logs, oracles, and patches
│   ├── models/                       FNO parameter and surface normalizers (.npz)
│   └── scalers/                      Autocall contract normalizers and calibration JSON
├── data/                             Dataset caches and quotes (gitignored)
└── tex/
    ├── thesis/main.pdf               51-page LaTeX thesis
    └── presentation/presentation.pdf Defence slides
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
| **P17**: Autocall QMC Pipeline | Complete | Sobol-Brownian Bridge QMC path simulation, Milstein stepping, barrier monitoring, Phoenix coupons, Worst-of EGNO |
| **P18 / Phase D**: Multi-Fidelity Ensemble | Complete | Crank-Nicolson Gatheral PDE base + 5-member residual MLP ensemble (0.99 bps RMSE), epistemic uncertainty ($\tau_{\text{OOD}} = 2.17$), SR 26-2 AutocallGuardian, Next.js 16+ & FastAPI UI integration |
| **P19**: Triangulated FX/Commodity Cross ML Engine | Complete | $\beta$-split H1 analytic engine + 5-member Sobolev DML residual MLP (0.67 bp MAE), 2nd-order Taylor jets, CUDA-graph runtime ($0.196\text{ ms}$ p50), Apache Arrow IPC batch streaming ($9.45\text{ ms}$ @ 10k), legacy DAG adapter, and full 17-phase forensic audit certification |


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
