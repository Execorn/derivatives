# ROADMAP.md — Program Maximum: Neural Network Pricing Framework
# Deep Rough Heston → Multi-Model Surrogate Calibration Framework
# Timeline: 6–12 months to thesis completion

---

## Vision

Build a unified, extensible Python framework where any stochastic volatility
model (parametric or data-driven) can be:
1. Simulated / priced via GPU
2. Wrapped in a FNO surrogate trained on a single workstation
3. Calibrated to any option market (equity, crypto, FX, rates, commodity)
   in under 1 second via GPU Gauss-Newton
4. Deployed as a REST API or interactive dashboard

The thesis demonstrates this framework on 4–6 models and 3+ asset classes,
with rigorous mathematical foundations, empirical validation, and a clean
open-source implementation. The academic claim: **FNO-based surrogate
calibration is model-agnostic, training-efficient on consumer hardware,
and production-ready**.

---

## Current State (Completed: P1–P3)

| Phase | Description | Status |
|---|---|---|
| P1 | Rough Heston FNO surrogate (v2: fixed H, v3: learnable H) | Done |
| P1 | FIM reparameterization + Gauss-Newton calibrator | Done |
| P1 | Streamlit dashboard | Done |
| P2 | FastAPI REST server | Done |
| P2 | SPX market data (yfinance + cleaning + grid) | Done |
| P2 | VIX futures term structure + joint calibration | Done |
| P2 | Deribit WebSocket streaming (BTC/ETH) | Done |
| P2 | Variance/vol swap pricing | Done |
| P3 | GPU Gauss-Newton (jacfwd) + batch multi-date calibration | Done |
| P3 | SVI fitting + arbitrage-free surface completion | Done |
| P3 | Portfolio Greeks (delta/gamma/vega/vanna/volga) + P&L attribution | Done |
| P3 | 535 tests passing | Done |

---

## Phase 4: Expanded Model Zoo (Months 1–3)

Goal: Add 4 new SV models to the framework. Each follows the same
pattern: pricer → dataset → FNO → normalizers → calibrator → tests.
All trainable on RTX 3080 / 4090 in under 6 hours per model.

---

### P4.1 — Classic Heston (baseline anchor)

**Why**: Industry baseline; the v2 model (fixed H) approximates this.
A clean Heston implementation strengthens the "rough vs. classical" comparison.

**Mathematical model:**
```
dv_t = κ(θ − v_t) dt + σ√v_t dW_t
dS_t/S_t = √v_t (ρ dW_t + √(1−ρ²) dB_t)
```
Parameters: (κ, θ, σ, ρ, V₀) — 5D.

**Implementation plan:**
- Pricer: Semi-analytical CF + Fourier-COS (already exists via `pricing_engine.py`
  with H→0.5 limit, but implement clean standalone version for clarity)
- Training budget: 200k samples, ~1 hour data gen, ~2 hours FNO training
- Normalizer: `param_normalizer_heston.npz`, `iv_normalizer_heston.npz`
- Calibrator: `calibrate_newton_heston()` — 5D Newton, no reparameterization needed
- Key output: Compare Heston vs Rough Heston RMSE on same market dates
- Deliverable: `tests/test_calibrate_heston.py` + notebook NB08

**Training bounds:**
```
κ ∈ [0.5, 10.0],  θ ∈ [0.01, 0.25],  σ ∈ [0.1, 2.0]
ρ ∈ [−0.95, 0.0],  V₀ ∈ [0.01, 0.25]
```

---

### P4.2 — SABR / SSVI (industry standard, FX + rates)

**Why**: SABR is the dominant model in FX and interest rates markets.
SSVI (Surface SVI) is the arbitrage-free extension used by major dealers.
Essential for multi-asset coverage.

**SABR mathematical model:**
```
dF_t = σ_t F_tᵝ dW_t
dσ_t = ν σ_t dZ_t,   d⟨W,Z⟩_t = ρ dt
```
Parameters: (α, β, ρ, ν) — Hagan et al. (2002) approximate IV formula.
β is often fixed at 0 (normal SABR) or 1 (log-normal SABR).

**SSVI (Gatheral & Jacquier 2014):**
```
w(k, θ_t) = θ_t/2 · [1 + ρ·φ(θ_t)·k + √((φ(θ_t)·k + ρ)² + 1 − ρ²)]
```
SSVI is calendar-arbitrage-free by construction and has only 3 parameters.

**Implementation plan:**
- Pricer: Hagan formula (analytical, microseconds) — no Monte Carlo needed
- FNO input: (α, ρ, ν) for fixed β=0; or (α, β, ρ, ν) for 4D
- Training: 500k samples, ~10 min data gen, ~2 hours training
- SSVI version: 3-param (ρ, η, γ_SSVI) with Hagan-SSVI bridge
- Calibrator: `calibrate_sabr()`, `calibrate_ssvi()`
- Key output: FX EUR/USD calibration (see P7.2)
- Deliverable: `src/pricing/sabr.py`, `tests/test_calibrate_sabr.py`, NB09

---

### P4.3 — Local Volatility (Dupire) — benchmark comparison

**Why**: Local vol is the exact theoretical "perfect fit" model (it always
matches market prices by construction). It serves as an upper bound on
calibration quality and a comparison benchmark.

**Dupire formula:**
```
σ²_LV(T, K) = [∂C/∂T + rK ∂C/∂K] / [½ K² ∂²C/∂K²]
```

**Implementation plan:**
- Two approaches:
  a) Parametric LV: SVI slice fit per maturity → LV surface via Dupire PDE
  b) Neural LV: FNO directly maps (T, K) → σ_LV(T, K) from observed prices
- The neural LV approach (b) is the novel contribution:
  train FNO to learn the mapping from a sparse IV surface to the full LV surface
- Pricer for comparison: finite-difference PDE (Crank-Nicolson) on BS PDE
  with LV diffusion — implemented in `src/pricing/local_vol.py`
- Key use: compare Rough Heston vs Heston vs Local Vol P&L attribution
- Deliverable: `src/pricing/local_vol.py`, notebook NB10 (model comparison)

---

### P4.4 — Rough Bergomi (rBergomi) — most natural extension

**Why**: rBergomi (Bayer, Friz & Gatheral 2016) is the other leading rough
volatility model. It has a simpler structure than Rough Heston (no mean
reversion) but excellent empirical performance. Many papers compare the two.

**Mathematical model:**
```
dS_t/S_t = √V_t dB_t
V_t = ξ_0(t) · exp(η W_t^H − ½ η² t^{2H})
```
where W^H is a fractional Brownian motion with Hurst exponent H.
Parameters: (H, η, ρ) — 3D.

**Key challenge**: No semi-analytical CF. Pure Monte Carlo required.
GPU Monte Carlo with Cholesky-correlated fBm paths, Euler-Maruyama scheme.

**Implementation plan:**
- Pricer: `src/pricing/rbergomi_gpu.py`
  - Hybrid scheme (Bennedsen, Lunde & Pakkanen 2017) — more accurate than Euler
  - 50k paths × 200 time steps on GPU → ~5-10 seconds per surface
  - Batch: 64 parameter sets simultaneously → dataset gen in ~6-8 hours
- Training: 100k samples, ~6 hours data gen, ~3 hours FNO training
- Parameters: (H, η, ρ), with H ∈ [0.04, 0.15], η ∈ [0.5, 4.0], ρ ∈ [−0.95, 0.0]
- Calibrator: `calibrate_rbergomi()` — 3D Newton
- Key comparison: rBergomi vs Rough Heston roughness exponent H on SPX
- Deliverable: `src/pricing/rbergomi_gpu.py`, `tests/test_calibrate_rbergomi.py`, NB11


---

## Phase 5: Neural SDE and Data-Driven Models (Months 2–5)

Goal: Extend the framework beyond parametric SV models to data-driven
approaches where the model itself is learned from market data.

---

### P5.1 — Lifted Heston Accuracy Study (N > 40 factors)

**Why**: The current implementation uses N=40 Bernstein factors. The error
from the rough kernel approximation is bounded by the factor count. Quantify
and document this tradeoff formally for the thesis.

**Plan:**
- Implement configurable N in `pricing_engine_gpu.py` (currently hardcoded)
- Benchmark: N = {5, 10, 20, 40, 80, 160} vs direct rough MC
- Metric: IV surface error (bps) as function of N and H
- Expected: N=40 gives <1 bp error for H≥0.05; N=80 needed for H≈0.04
- Cost: CPU/GPU time grows O(N²) — N=160 on GPU is still fast (~50ms)
- Deliverable: `benchmarks/convergence_N_factors.py` (extended), thesis Section 3.3

---

### P5.2 — Neural SDE as Model Prior (trainable SV model)

**Why**: Instead of choosing a parametric SV model, learn the drift and
diffusion functions from data. This is model-free calibration — the SDE
itself is parameterized by a neural network.

**Mathematical framework (Tzen & Raginsky 2019, Kidger et al. 2021):**
```
dV_t = f_θ(t, V_t) dt + g_θ(t, V_t) dW_t
```
where f_θ, g_θ are small MLPs. Training via:
- **Score matching** (denoising diffusion approach)
- **SDE adjoint method** (backprop through torchdiffeq solver)
- **Variational lower bound** (ELBO on path distributions)

**Feasibility on RTX 3080:**
- Small architecture: f_θ, g_θ are 3-layer MLPs (hidden=32)
- Simulation: Euler-Maruyama, 1000 time steps, 2000 paths
- Training signal: market IV surface (8×11) as observed data
- Approximate training time: 6-12 hours (feasible on single GPU)
- The FNO surrogate wraps the Neural SDE just like any other model

**Key challenge**: Neural SDE → IV surface requires Monte Carlo averaging.
Gradient through this expectation needs:
  a) Re-parameterization trick (differentiable MC), or
  b) Score function estimator (REINFORCE)

**Plan:**
- Implement `src/pricing/neural_sde.py` using `torchdiffeq` + `torchsde`
- Train on SPX historical surfaces (2020-2024) as market data
- Compare: Neural SDE vs Rough Heston at capturing vol smile dynamics
- Deliverable: `src/pricing/neural_sde.py`, NB12 (Neural SDE calibration)

---

### P5.3 — Signature-Based Volatility Model

**Why**: Rough path signatures provide a model-free, non-parametric
representation of path space. Signature-based models (Arribas 2018,
Lacasa et al. 2023) learn volatility surfaces from historical paths.

**Approach:**
- Extract signature features of (log S, log VIX) paths up to depth K=4
- Train an MLP/FNO to map signature features → IV surface
- This is purely data-driven: no SDE, no calibration in the classical sense
- The "parameters" are the signature coefficients of the observed path

**Feasibility:** Signature computation is O(L^K) where L=path length, K=depth.
For K=4, L=100 daily returns: fast (seconds). `iisignature` library.

**Plan:**
- `src/pricing/signature_vol.py`
- Use 252-day rolling window of daily log-returns and VIX as path
- Target: predict the IV surface for the following week
- This is fundamentally a prediction task, not calibration → add a
  "forecast" mode to the framework alongside the "calibrate" mode
- Deliverable: NB13 (Signature-based forecasting)

---

## Phase 6: Deep Hedging — Model-Free Reinforcement Learning (Months 4–7)

**Why**: Deep Hedging (Buehler, Gonon, Teichmann & Wood 2019) is the
leading model-free approach to derivative hedging. Instead of computing
analytical Greeks and hedging based on a model, a neural network learns
the optimal hedge policy directly from market data (or simulated paths).
This is the "missing piece" for a complete framework.

**Mathematical formulation:**
```
Maximize: E[−exp(−λ · PL_T)]  (exponential utility)

PL_T = −V_0 + Σ_{t=0}^{T} δ_t ΔS_t − Σ_{t=0}^{T} c(δ_t)

δ_t = π_θ(information_t)   [hedge policy = neural network]
```
where c(δ_t) is a transaction cost function.

**Feasibility on RTX 3080:**
- Architecture: 3-layer LSTM or Transformer (small, hidden=64)
- Simulation: 100 paths × 252 steps for a European call — fast
- Training: ~2-4 hours for a single instrument
- Scale: portfolio of 5-10 options → ~12-24 hours

---

### P6.1 — Deep Hedging for European Options under Rough Heston

**Plan:**
- Simulate paths from calibrated Rough Heston (use v3 model for fast simulation)
- Train hedge policy π_θ for a delta-neutral portfolio under transaction costs
- Compare: Rough Heston Greeks (from P3) vs Deep Hedging policy
- Metric: hedging error variance reduction vs Black-Scholes delta hedge
- Expected: Deep Hedging reduces hedging error by 10-20% vs model-based Greeks
- Deliverable: `src/hedging/deep_hedging.py`, NB14

---

### P6.2 — Deep Hedging for Exotic Options (Path-Dependent)

**Why**: Deep Hedging is most powerful for exotics where no analytical
Greeks exist. Barrier options, Asian options, variance swaps are natural
targets.

**Plan:**
- Instrument: Down-and-out barrier call (European barrier with rebate)
- The FNO cannot price barrier options (wrong architecture for path-dependency)
- Instead: simulate paths → compute barrier payoff → train hedger
- Transaction costs: proportional (0.01% of notional per rebalance)
- Comparison: delta-hedging with finite-difference barrier greeks vs Deep Hedging
- Deliverable: `src/hedging/barrier_hedging.py`, NB15

---

### P6.3 — Adversarial Market Generation

**Why**: Deep Hedging requires realistic simulated market paths. Train a
GAN or diffusion model to generate market scenarios that fool the hedger,
making it more robust.

**Plan:**
- Generator G: Produces 252-day log-return sequences
- Discriminator D: The hedger's loss — tries to maximize hedging error
- This is a minimax game: the hedger learns to hedge against the worst-case
  generated market
- Implementation: GAN with WGAN-GP loss for stability
- Expected output: A set of "stress scenarios" automatically discovered
  by the adversarial process
- Deliverable: `src/hedging/adversarial_market.py`, NB16 (stretch goal)


---

## Phase 7: Multi-Asset Framework (Months 3–8)

Goal: Extend the calibration framework to FX options, interest rate
swaptions, and commodity options. Each asset class requires a different
market data source, cleaning pipeline, and model choice.

---

### P7.1 — Mixed Local-Stochastic Volatility (MLSV) for Equity

**Why**: MLSV (Lipton 2002, Piterbarg 2007) is the industry standard for
equity exotic desks. It combines local vol (exact fit to vanillas) with
stochastic vol (realistic dynamics for exotics). Adding it to the framework
closes the gap with production systems.

**Mathematical model:**
```
dS_t/S_t = σ_LV(t, S_t) · √V_t · dB_t
dV_t = κ(θ − V_t) dt + ε√V_t dW_t,   d⟨B,W⟩ = ρ dt
```
The local vol component σ_LV(t, S) is calibrated to vanilla prices exactly.
Parameters: (κ, θ, ε, ρ) + the σ_LV surface → high-dimensional.

**Feasibility:** GPU Monte Carlo particle method (McKean-Vlasov SDE).
Dataset generation is expensive (~24 hours). Consider:
  a) Pre-generate a fixed library of σ_LV surfaces and train conditional FNO
  b) Parameterize σ_LV with a low-dim SVI family (3-5 params) → 9-param MLSV

**Plan:**
- Start with simplified parametric MLSV (option b)
- `src/pricing/mlsv_gpu.py` — GPU Monte Carlo
- Training: 50k samples, ~8 hours on RTX 4090
- Compare: MLSV vs Rough Heston on SPX exotics (barrier, variance swap)
- Deliverable: `src/pricing/mlsv_gpu.py`, NB17 (stretch goal)

---

### P7.2 — FX Options: EUR/USD with SABR

**Why**: FX is the second largest option market after equity. SABR is the
universal standard. EUR/USD has liquid vanilla options on Bloomberg.

**Data source:**
- Bloomberg: EUR/USD implied vol surface (25-delta Risk Reversal,
  Butterfly, ATM vol — the "vol triangle" convention)
- Alternative free source: FRED + CBOE historical FX data
- Deribit does not list FX options → need Bloomberg or a broker API

**Market conventions:**
- FX quotes in delta (25δP, 10δP, ATM, 10δC, 25δC) × maturity
- Must convert delta-space quotes to (T, K) grid before FNO input
- Implement: `src/market/fx_data.py` — delta-to-strike conversion

**Model:** SABR (P4.2) calibrated to FX vol triangle
- 3 free params (α, ρ, ν) for fixed β=1 (log-normal SABR for FX)
- Calibration: 3D Newton using SABR FNO
- Compare: SABR vs Rough Bergomi on FX smiles

**Plan:**
- `src/market/fx_data.py` — FRED/Bloomberg data loader
- `src/calibration/fx_calibration.py` — FX-specific pipeline
- NB18 (FX SABR calibration)

---

### P7.3 — Interest Rate Swaptions with LMM-SABR

**Why**: Swaptions are the largest derivatives market by notional. The
industry standard is LIBOR Market Model (LMM) with SABR vol.

**Mathematical model (LMM-SABR):**
```
dL_i = σ_i L_i^β dW_i    (LIBOR rates)
σ_i dynamics follow SABR
```
For a thesis scope, target:
- **Bachelier model** (normal model for rates, 1 param per tenor) — simplest
- **Displaced diffusion SABR** — 4 params, standard practice post-2008
- **FNO surrogate**: maps (α, β, ρ, ν, tenor) → swaption vol cube

**Data source:**
- SOFR swaption vol cube (post-LIBOR transition)
- Free data: ICE, FRED SOFR rates + broker mid vols (limited)
- Feasibility constraint: may need to work with synthetic data / academic datasets

**Plan:**
- Start with Bachelier model + simple SABR as baseline
- `src/market/rates_data.py` — SOFR swaption data
- `src/pricing/bachelier.py`, `src/pricing/sabr_rates.py`
- NB19 (Rates swaption calibration) — thesis bonus chapter

---

### P7.4 — Commodity Options: WTI Crude Oil / Gold

**Why**: Commodity vol surfaces have distinct features (inverted smile,
seasonality, supply shocks) not captured by equity models. Adding commodities
demonstrates the framework's model-agnosticism.

**Model choice:**
- **Schwartz-Smith two-factor** (short + long term factor) — commodity standard
- Or: Heston (equity-like) as baseline, compare with SABR
- Commodities often use Black-76 (log-normal forward) as market convention

**Data source:**
- CME Group (free end-of-day data): WTI crude oil options
- LBMA gold options via Bloomberg / broker
- yfinance: `GC=F` (gold futures), `CL=F` (crude oil futures)

**Plan:**
- `src/market/commodity_data.py`
- `src/pricing/schwartz_smith.py` — two-factor commodity model
- NB20 (Commodity calibration)

---

## Phase 8: Production Framework (Months 6–12)

Goal: Transform the research codebase into a distributable, user-facing
framework. "Users" = quant researchers, students, practitioners.

---

### P8.1 — Clean Python Package (`pip install deepvol`)

**Plan:**
```
deepvol/
├── models/          # SV model zoo (RoughHeston, rBergomi, SABR, ...)
├── surrogates/      # FNO training + inference
├── calibration/     # Newton calibrators, batch, joint
├── market/          # Data adapters (SPX, Deribit, FX, Rates)
├── hedging/         # Greeks, portfolio, deep hedging
└── cli.py           # Command-line interface
```

Package API design:
```python
import deepvol as dv

# One-line calibration
result = dv.calibrate(market_iv, model="rough_heston", device="cuda")
print(f"H = {result.H:.3f}, RMSE = {result.rmse_bps:.1f} bps")

# Train your own surrogate
surrogate = dv.train_surrogate(model="rbergomi", n_samples=100_000)
surrogate.save("my_rbergomi_surrogate.pth")

# Compare models
comparison = dv.compare_models(market_iv,
    models=["rough_heston", "rbergomi", "sabr"],
    metric="rmse_bps")
```

- PyPI release: `pip install deepvol` with pre-trained weights
- License: MIT (research code, attribution required)
- Deliverable: `pyproject.toml`, `deepvol/` package, ReadTheDocs

---

### P8.2 — Enhanced Web Dashboard (Streamlit v2)

**New features beyond current `app_fno.py`:**
- Model selector: choose RH / rBergomi / SABR / Heston from dropdown
- Upload CSV/Excel of market quotes → auto-parse → calibrate → download results
- Multi-date comparison: show H dynamics for last 30 days on a chart
- Model comparison panel: same surface calibrated with 3 models side by side
- Export: PDF report with calibration results and model diagnostics

---

### P8.3 — REST API v2 (FastAPI + Docker)

**New endpoints:**
```
POST /calibrate/model/{model_name}   # any model, not just rough_heston
POST /train_surrogate                 # trigger dataset gen + FNO training job
GET  /models                          # list available models + their accuracy stats
GET  /history/{ticker}/{date}         # historical calibrated parameters
WS   /stream/{ticker}                 # WebSocket live calibration stream
```

**Docker deployment:**
```dockerfile
FROM pytorch/pytorch:2.12-cuda12.6-cudnn9-runtime
COPY . /app
RUN pip install -e /app
CMD ["uvicorn", "deepvol.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### P8.4 — Cloud Deployment (Kubernetes / GCP)

**Architecture:**
```
[Market Data Feed] → Kafka → [Calibration Workers (GPU pods)] → Redis
                                                                    │
[Dashboard] ←── FastAPI ←── Redis ←── [Parameter Store (Postgres)]
```
- GPU pods: 1-2 NVIDIA T4 (Google Cloud, ~$0.35/hr each)
- Auto-scaling: scale up during market hours (9:30–16:00 ET)
- Latency target: calibrate SPX surface within 500ms of each tick
- Cost estimate: ~$50-100/month for continuous operation


---

## Phase 9: Empirical Studies (Months 4–10)

Goal: Produce publishable empirical findings from running the full
framework on real market data. These are the "results" chapters of the thesis.

---

### P9.1 — Cross-Asset Roughness Study

**Research question**: Is H ≈ 0.1 universal, or does it vary significantly
across asset classes and market regimes?

**Methodology:**
- Calibrate Rough Heston (v3) daily for 4 years (2020–2024)
- Assets: SPX, BTC, ETH + FX (EUR/USD via SABR comparison) + WTI
- Sample size: ~1000 data points per asset (4 years × 252 days)
- Compute: mean H, std H, autocorrelation of H, correlation of H across assets
- Regime detection: identify periods where H changes significantly
  (COVID crash 2020, BTC bull 2021, rate hike cycle 2022, AI rally 2023-24)

**Expected findings (hypothesis):**
- SPX: H ≈ 0.08–0.12 (literature: ~0.1)
- BTC: H possibly higher (closer to 0.15) — less rough (more Markovian)
- ETH: higher volatility → more extreme H values
- Cross-asset: H correlates across assets during systemic crises

**Deliverable:** `src/analysis/cross_asset_roughness.py`, NB21, thesis Chapter 5

---

### P9.2 — Model Comparison Study: Which Model Fits Best?

**Research question**: For each asset class, which SV model achieves the
lowest calibration RMSE? Does the best model depend on market conditions?

**Methodology:**
- Calibrate all implemented models (RH, rBergomi, Classic Heston, SABR, LV)
  to the same set of 100+ market dates
- Metrics: calibration RMSE, out-of-sample RMSE (hold-out strikes),
  parameter stability (day-over-day variation), calibration speed
- Statistical tests: Diebold-Mariano test for forecast accuracy comparison

**Expected output:**
| Asset | Best Model | Notes |
|---|---|---|
| SPX | Rough Heston / rBergomi | Both comparable |
| BTC | rBergomi or SABR | Crypto smiles are steeper |
| EUR/USD | SABR | Industry standard for FX |
| Swaptions | LMM-SABR | Only viable option |

**Deliverable:** `src/analysis/model_comparison.py`, NB22, thesis Chapter 5

---

### P9.3 — FNO Surrogate Accuracy vs Training Data Size

**Research question**: How many training samples are needed for the FNO
to achieve X bps accuracy? What is the data-efficiency of the FNO approach?

**Methodology:**
- Train FNO on subsets: {1k, 5k, 10k, 50k, 100k, 200k, 500k} samples
- Measure: MAE (bps), R², NaN rate, calibration RMSE on held-out market dates
- Compare: FNO vs Gaussian Process emulator (GPE) vs Polynomial chaos expansion

**Expected finding:** FNO reaches good accuracy (~5 bps) with ~50k samples
and plateaus — more data gives diminishing returns. GPE is competitive at
low data regimes but fails to scale.

**Deliverable:** `benchmarks/data_efficiency_study.py`, thesis Chapter 4

---

### P9.4 — Real-Time Calibration Latency Profiling

**Research question**: What is the end-to-end latency breakdown for
live market calibration? Where are the bottlenecks?

**Profiling targets:**
- Market data fetch (yfinance / Deribit WebSocket): 50-500ms (network)
- Surface cleaning + grid interpolation: <5ms
- Normalizer forward pass: <0.1ms
- FNO inference (batch=1): ~4ms (GPU) / ~15ms (CPU)
- Jacobian computation (jacfwd, 4 passes): ~16ms (GPU)
- Newton iteration (20 iters × 16ms): ~320ms total
- JSON serialization + HTTP response: <5ms

**Target:** Sub-500ms end-to-end for all 20 Newton iters on GPU
**Current:** 541ms (p50), 668ms (p95) — already meeting target

**Deliverable:** `benchmarks/latency_breakdown.py`, thesis Chapter 4

---

### P9.5 — Hedging Effectiveness Study

**Research question**: Which hedging approach reduces P&L variance most?
Compare:
1. Black-Scholes delta hedge (model-free baseline)
2. Rough Heston Greeks (from `portfolio_greeks.py`) — model-based
3. Deep Hedging policy (P6.1) — RL-based
4. Vega-neutral hedge (model-based, hedge both delta and vega)

**Methodology:**
- Backtest period: 2022-2024 (high volatility regime)
- Portfolio: 10 ATM calls on SPX, daily rebalancing
- Metric: hedging error variance, Sharpe ratio of hedged portfolio
- Transaction costs: 0 and 0.01% per rebalance

**Deliverable:** `benchmarks/hedging_backtest.py`, NB23, thesis Chapter 6

---

## Phase 10: Academic Contributions (Ongoing)

---

### P10.1 — Thesis Completion (Main Goal)

**Thesis structure (target: 80-100 pages):**
1. Introduction — motivation, related work, contributions
2. Mathematical foundations — fBm, Rough Heston, rBergomi, Lifted Heston
3. GPU pricing — Lifted Heston, Fourier-COS, MC for rBergomi
4. FiLM-FNO surrogate — architecture, training, accuracy, data efficiency
5. Calibration framework — Newton, FIM, reparameterization, speed
6. Extended model zoo — Heston, SABR, Local Vol, Neural SDE
7. Empirical results — cross-asset study, model comparison, hedging
8. Production framework — API, dashboard, deployment
9. Conclusion and future work

**Timeline:**
- Months 1-3: P4 (model zoo) — add Heston, SABR, rBergomi, Local Vol
- Months 3-5: P5 (Neural SDE), P9.1-9.2 (empirical)
- Months 4-6: P6 (Deep Hedging), P7 (multi-asset)
- Months 6-9: P8 (production), P9.3-9.5 (remaining studies)
- Months 9-12: P10 (thesis writing, defense prep)

---

### P10.2 — Potential Publications

**Paper 1 (target: Quantitative Finance or JCAM):**
*"FiLM-Conditioned Fourier Neural Operators for Real-Time Stochastic
 Volatility Calibration"*
- Core FNO surrogate + Gauss-Newton result
- Comparison vs L-BFGS, Adam, differential evolution
- Novelty: FiLM conditioning for parameter-conditional operator learning

**Paper 2 (target: Risk Magazine or Applied Mathematical Finance):**
*"A Unified Neural Surrogate Framework for Stochastic Volatility Model
 Calibration Across Asset Classes"*
- Multi-model results (P4)
- Cross-asset roughness study (P9.1)
- Model comparison (P9.2)

**Paper 3 (stretch, target: NeurIPS / ICML Workshop):**
*"Neural SDE Calibration via Score Matching: A Model-Free Approach
 to Implied Volatility Surface Fitting"*
- Neural SDE results (P5.2)
- Comparison with parametric models

---

### P10.3 — Open Source Release

Upon thesis submission:
- Clean `deepvol` package on PyPI with pre-trained weights for all models
- ReadTheDocs documentation with tutorials
- Colab notebooks demonstrating each model
- Pre-computed calibration results for 2020-2024 SPX history
- Companion Hugging Face model card with weights and benchmarks


---

## Summary: Priority Matrix

Prioritized by (academic impact × implementation feasibility) on a single GPU.

| Item | Priority | GPU hours | Academic novelty | Complexity |
|---|---|---|---|---|
| Classic Heston FNO (P4.1) | P0 | 3h | Low (baseline) | Low |
| SABR / SSVI (P4.2) | P0 | 2h | Medium | Low |
| Rough Bergomi FNO (P4.4) | P0 | 10h | High | Medium |
| Local Vol / Dupire (P4.3) | P0 | 3h | Medium | Medium |
| Roughness study P9.1 | P0 | 20h compute | High | Low |
| Model comparison P9.2 | P0 | 10h compute | High | Low |
| Lifted Heston N study (P5.1) | P1 | 4h | Medium | Low |
| Neural SDE (P5.2) | P1 | 12h | High | High |
| Deep Hedging (P6.1) | P1 | 8h | High | High |
| FX SABR (P7.2) | P1 | 2h + data | Medium | Medium |
| Latency profiling P9.4 | P1 | 1h | Low | Low |
| Hedging backtest P9.5 | P1 | 4h | High | Medium |
| MLSV (P7.1) | P2 | 24h | Very high | Very high |
| Rates swaptions (P7.3) | P2 | 5h + data | High | High |
| Commodity options (P7.4) | P2 | 5h + data | Medium | Medium |
| Signature-based model (P5.3) | P2 | 6h | High | High |
| Deep Hedging exotics (P6.2) | P2 | 16h | High | High |
| pip package deepvol (P8.1) | P2 | 0h (eng) | Low | Medium |
| Docker API v2 (P8.3) | P3 | 0h (eng) | Low | Medium |
| Cloud deployment (P8.4) | P3 | ongoing | Low | High |
| Adversarial market GAN (P6.3) | P3 | 24h | Very high | Very high |
| Data efficiency study (P9.3) | P3 | 20h | Medium | Low |

---

## Master Timeline (6–12 Month Plan)

```
Month 1:   P4.1 Classic Heston + P4.2 SABR (fast, establishes pipeline)
Month 2:   P4.4 rBergomi GPU + P5.1 Lifted Heston N study
Month 3:   P4.3 Local Vol + P9.2 Model comparison (SPX)
Month 4:   P5.2 Neural SDE (start) + P7.2 FX SABR + P9.1 Roughness study
Month 5:   P6.1 Deep Hedging (European) + P9.5 Hedging backtest
Month 6:   P9.4 Latency profiling + P8.1 deepvol package (start)
Month 7:   P7.3 Rates swaptions + P7.4 Commodity options
Month 8:   P6.2 Deep Hedging exotics + P5.3 Signature-based
Month 9:   P8.2 Dashboard v2 + P8.3 REST API v2
Month 10:  Thesis writing (Chapters 1-4)
Month 11:  Thesis writing (Chapters 5-9) + P10.2 paper drafts
Month 12:  Defense preparation, slides, mock defense
```

---

## GPU Training Budget Estimate

Assuming RTX 3080 (10GB VRAM) or equivalent:

| Model | Dataset size | Gen time | Train time | Total GPU time |
|---|---|---|---|---|
| Classic Heston | 200k | 1h | 2h | 3h |
| SABR | 500k | 0.2h | 2h | 2.2h |
| Local Vol | 100k | 2h | 2h | 4h |
| Rough Bergomi | 100k | 6h | 3h | 9h |
| Neural SDE | from market | — | 12h | 12h |
| Deep Hedging (European) | sim-based | 1h | 8h | 9h |
| Deep Hedging (barrier) | sim-based | 2h | 16h | 18h |
| MLSV | 50k | 18h | 6h | 24h |
| **Total (P0–P1)** | | | | **~60h** |
| **Total (all)** | | | | **~150h** |

60 hours of GPU compute at 100% utilization ≈ 2.5 days continuous.
Realistically (~8h/day research): 7-10 days total for all P0-P1 items.
MLSV (P2) is the heaviest single item — evaluate feasibility after P0-P1.

---

## Open Questions (Discuss with Supervisor)

1. **Thesis scope**: Is 4 SV models + 3 asset classes sufficient for a
   "complete framework" claim, or do we need all 8 models?

2. **Neural SDE training stability**: Score matching vs adjoint method —
   which gives better gradients for IV surface fitting? Consider a 1-day
   prototype before committing to full implementation.

3. **Deep Hedging**: Is a 20% hedging error reduction publishable as a
   standalone result, or does it need the exotic instruments component?

4. **Data availability for rates/FX**: Without Bloomberg access, SOFR
   swaption data is very sparse. Consider using academic datasets (OptionMetrics
   for rates) or generating synthetic data from calibrated models.

5. **deepvol package**: PyPI release before or after defense? Pre-release
   builds research visibility but requires maintaining the package.

6. **Rough Bergomi vs Rough Heston**: Literature suggests rBergomi fits
   SPX better (lower RMSE) but has no semi-analytical CF. Once rBergomi FNO
   is trained, compare directly and let data guide model choice for the thesis.

