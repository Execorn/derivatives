# Deep Rough Heston Calibration via FiLM-FNO

> Master's Thesis — МФТИ ФПМИ, Кафедра БИТ, 2026
> *Нейронные сети в ценообразовании производных финансовых инструментов*

---

## What This Is

A production-grade Python framework for **real-time calibration of stochastic
volatility models** using neural network surrogates. The core idea: replace an
expensive numerical pricer (Fourier-COS, seconds per surface) with a
**FiLM-conditioned Fourier Neural Operator (FNO)** that prices any implied
volatility surface in under 1 ms, enabling interactive Gauss-Newton calibration.

Currently implemented: **Rough Heston** (El Euch & Rosenbaum 2019), with Hurst
exponent H ∈ [0.04, 0.15] as a free parameter (v3 model). The framework is
designed for extension to additional SV models (Rough Bergomi, SABR, Local Vol,
Neural SDE) with minimal code changes — add a pricer, generate data, train FNO.

---

## Key Results

| Experiment | Result |
|---|---|
| FNO v3 surrogate accuracy | R² = 0.9981, MAE = 0.264% (learnable H) |
| Inference speed | ~4 ms / batch vs 5.6 s Fourier-COS — **1400× speedup** |
| SPX calibration (NB01) | **8.9 bps RMSE**, 12 Newton iters, H = 0.113 |
| Batch calibration — 5 dates (NB06) | **12.8 bps median**, 5/5 converged |
| Joint SPX + VIX (NB07) | Converged; SPX 198.9 bps, VIX error 0.362 |
| BTC live Deribit (NB05) | 1547 bps (v2 model, 548 contracts) |
| Streaming p95 latency | 668 ms < 1 s real-time threshold |
| FIM reparameterization | Condition number 1301× lower (κ ≈ 770) |
| Test suite | 535 passed, 0 failed, 2 skipped (integration only) |

---

## Architecture

```
θ = (κ, θ, σ, ρ, V₀, H)
         │
    ParameterNormalizer
         │
    FiLM MLP → (γᵢ, βᵢ) scale+shift per Fourier layer
         │
(T,K) grid (8×11) → mirror-pad (16×22) → Lifting Conv (1→40 ch)
                     4 × FourierLayer (FiLM-modulated, modes 8×11)
                     Projection (40→1)
                     IVSurfaceNormalizer⁻¹
                           │
                   σ_IV(T, K)   in decimal vol
                           │
             Gauss-Newton calibration loop
             J = jacfwd(FNO, θ)   [3 or 4 forward passes]
             θ* = θ - (JᵀJ + λI)⁻¹ Jᵀ r
```

---

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or see README.md for full install

# Run notebooks (end-to-end demos)
cd notebooks && python generate_notebooks.py
jupyter lab

# Run Streamlit dashboard
streamlit run src/app_fno.py

# Run tests
pytest tests/ -q
```

---

## Structure

| Directory | Purpose |
|---|---|
| `src/` | Core library (model, calibrators, market data, API) |
| `notebooks/` | 7 end-to-end demo notebooks (NB01–NB07) |
| `tests/` | 535-test pytest suite |
| `benchmarks/` | Speed and accuracy studies |
| `artifacts/` | Trained weights + normalizers (.pth, .npz) |
| `research/` | Research notes, SOTA survey, reference PDFs |
| `tex/` | LaTeX thesis (51 pages) + presentation slides |

---

## Documentation

- **README.md** — full public documentation (installation, API, benchmarks)
- **CONTEXT.md** — deep technical context for developers and AI agents
- **ROADMAP.md** — complete future development plan
- **`tex/thesis/main.pdf`** — 51-page academic thesis
