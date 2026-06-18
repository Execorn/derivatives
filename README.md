# Deep Rough Heston Calibration via FiLM-FNO

> **Master's Thesis Project** — МФТИ ФПМИ, Кафедра БИТ, 2026  
> *Нейронные сети в ценообразовании производных финансовых инструментов*

GPU-accelerated calibration of the **Rough Heston** stochastic volatility model using
a **FiLM-conditioned Fourier Neural Operator (FNO)** surrogate that maps model parameters
directly to a full implied-volatility surface — **1400× faster than the reference pricer**.

---

## 🏆 Key Results

| Experiment | Result |
|---|---|
| FNO v2 surrogate accuracy | **R² = 0.9991, MAE = 0.058%** (5.8 bp) |
| Inference speed | **~4 ms** (batch 1024) vs 5.6 s direct COS |
| Newton calibration (no noise) | **541 ms**, 3× faster than L-BFGS |
| Streaming p95 latency (20 ticks) | **668 ms** < 1000 ms real-time threshold ✅ |
| FIM reparameterization | Condition number **1301× lower** (κ ≈ 770) |
| Delta hedging variance reduction | **+5.1%** vs flat Black-Scholes Δ |
| FNO v3 (learnable H ∈ [0.04, 0.15]) | **R² = 0.9981, MAE = 0.264%** (genuine COS) |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install torch numpy scipy streamlit plotly pandas

# 2. Run the interactive demo
streamlit run src/app_fno.py

# 3. Run the test suite
pytest tests/ -v
```

---

## 🏗️ Architecture

```
θ = (κ, θ, σ, ρ, V₀, H) ──→ ParameterNormalizer ──→ FiLM(θ) ──→ γ, β
                                                                      │
(T, K) grid (8×11) ──→ Mirror-pad ──→ Lift Conv ──→ 4× FourierLayer ← FiLM
                                                          │
                                                      Project 40→1
                                                          │
                                                    IV surface (8×11) ──→ IVSurfaceNormalizer⁻¹
```

**Why FNO?**
- Operator learning: learns the *map* `θ → σ_IV(T,K)`, not just point evaluations
- Mirror padding: enforces put-call parity symmetry on the T-axis
- FiLM conditioning: scale-shift modulation gives better generalization than concatenation

---

## 📦 Models & Weights

| Model | Weights | Params | R² | MAE | Notes |
|---|---|---|---|---|---|
| FNO v2 | `artifacts/weights/fno_v2_final_prod.pth` | 2.2M | 0.9991 | 0.058% | H fixed=0.08 |
| FNO v3 | `artifacts/weights/fno_v3_final_prod.pth` | 2.2M | 0.9981 | 0.264% | H learnable |

Normalizers: `artifacts/models/{param,iv}_normalizer_v{2,3}.npz`

---

## 📂 Project Structure

```
src/
  fno_model.py          — MirrorPaddedFNO2d architecture
  normalizers.py        — ParameterNormalizer, IVSurfaceNormalizer
  pricing_engine.py     — GPU Fourier-COS pricer (Bernstein lifting, N=40)
  calibrate.py          — L-BFGS calibrator + FIM ellipsoid
  calibrate_fast.py     — Newton–Gauss calibrator (jacfwd, quadratic convergence)
  calibrate_h.py        — 4D calibrator: (v₀, ζ, λ, H)
  app_fno.py            — Streamlit demo (IV surface + Newton tab + Greeks)

tex/thesis/main.pdf     — 51-page LaTeX thesis (МФТИ ФПМИ БИТ)
data/                   — Datasets (gitignored, reproduce with generate_dataset*)
artifacts/              — Trained model weights and normalizers
benchmarks/             — Benchmark result files
research/               — Deep research notes and SOTA survey
tests/                  — pytest test suite
```

---

## 📐 Model Parameters

```
κ (kappa)  ∈ [0.5,  5.0]   — mean reversion speed
θ (theta)  ∈ [0.01, 0.25]  — long-run variance
σ (sigma)  ∈ [0.1,  1.5]   — vol-of-vol
ρ (rho)    ∈ [-0.95, 0.0]  — spot-vol correlation
V₀         ∈ [0.01, 0.25]  — initial variance
H          ∈ [0.04, 0.15]  — Hurst exponent (rough regime H < 0.5)
```

IV Surface grid: T ∈ {0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0} yr × k ∈ [-0.5, 0.5] (11 points)

---

## 🧪 Tests

```bash
pytest tests/test_pricing_engine.py -v      # COS pricer accuracy
pytest tests/test_calibrate_newton.py -v    # Newton calibrator
pytest tests/test_calibrate_newton_h.py -v  # 4D learnable-H calibrator
```

---

## 📖 Thesis

The full 51-page thesis (`tex/thesis/main.pdf`) covers:
- Mathematical foundations: fBm, Rough Heston, Riccati ODE
- GPU Fourier-COS pricing with Bernstein lifting
- FiLM-FNO architecture and training
- FIM-based identifiability and reparameterization
- Experimental results (7 sections)
- Delta hedging backtest

**Build:**
```bash
cd tex/thesis
pdflatex -interaction=nonstopmode main.tex && biber main
pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

---

## 🔑 Key Technical Notes

1. **Spatial grid format:** Always `(B, T, K, 2)` channels-last (NOT channels-first)
2. **Both v2 and v3 are `param_dim=6`:** v2 fixes H=0.08 at inference, v3 learns H
3. **T=0.1 COS instability:** Small V₀ causes COS NaN — dataset v4 uses median fill
4. **Newton calibrator:** Uses `torch.func.jacfwd` for exact Jacobian → quadratic convergence

---

## 📄 License

Research code — МФТИ Master's thesis project. Contact author for usage permissions.
