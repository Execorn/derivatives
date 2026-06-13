# Deep Learning Calibration of the Lifted Rough Heston Model 📈

[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

A **Master's thesis project** (MIPT) implementing three progressively advanced pipelines for real-time
calibration of the **Lifted Rough Heston** stochastic volatility model (El Euch, Gatheral & Rosenbaum 2019).

The core idea: replace expensive Fourier-COS pricing (seconds per surface) with a
**FiLM-conditioned Fourier Neural Operator (FNO)** surrogate that prices a full 8×11
implied-volatility surface in **< 1 ms** — enabling gradient-based calibration at interactive speed.

---

## Mathematical Foundation

### The Lifted Rough Heston Model

The variance process is driven by a weighted sum of Ornstein–Uhlenbeck factors
that approximate the rough fractional kernel (Hurst exponent H ≈ 0.08):

$$v_t = v_0 + \int_0^t K(t-s)\,\kappa(\theta - v_s)\,ds + \int_0^t K(t-s)\,\sigma\,dW_s^v$$

$$K(t) = \sum_{j=1}^N c_j\,e^{-\gamma_j t} \approx \frac{t^{H-1/2}}{\Gamma(H+1/2)}, \quad H = 0.08$$

The N-factor Bernstein approximation lifts the non-Markovian rough kernel into a
Markovian system of N coupled SDEs — enabling exact Fourier-COS characteristic
function pricing.

| Parameter | Symbol | Range | Role |
|---|---|---|---|
| Mean reversion | κ | [0.1, 5.0] | Speed of variance mean-reversion |
| Long-run variance | θ | [0.01, 0.15] | Steady-state variance level |
| Vol-of-vol | σ | [0.1, 1.0] | Amplitude of variance shocks |
| Correlation | ρ | [−0.9, −0.1] | Asset–variance correlation (skew driver) |
| Initial variance | v₀ | [0.01, 0.15] | Spot variance |
| Hurst exponent | H | 0.08 (fixed) | Roughness; H=0.5 recovers classical Heston |

**Ghost parameters** fixed during 3D calibration: κ = 1.0, θ = 0.08, H = 0.08.

### Reparameterization (Option 1 contribution)

Instead of calibrating in the ill-conditioned 6D space, we use:

$$\zeta = \sigma\rho \in [-0.9,\,-0.01], \quad \lambda = \sigma\sqrt{1-\rho^2} \in [0.01,\,0.99], \quad v_0 \in [0.01,\,0.15]$$

**Result:** Fisher Information Matrix condition number drops from **1.49 × 10⁷** (6D) to
**1.15 × 10⁴** (3D) — a **1301× reduction**, making gradient-based calibration numerically stable.

---

## Three-Option Architecture

```
                        ┌──────────────────────────────────────────┐
                        │         CUDA MC Engine (.so)             │
                        │   Lifted Rough Heston, GPU-accelerated   │
                        │   (Euler-Maruyama, ~15bp bias at T=0.1) │
                        └────────────┬─────────────────────────────┘
                                     │ MC dataset v1 (~10k)
                    ┌────────────────▼────────────────────────────┐
                    │         Fourier-COS Pricer (CPU)            │
                    │  N=40 Bernstein factors, N_cos=64           │
                    │  Exact pricing, NaN at T=0.1 for H=0.08    │
                    └──────────────┬──────────────────────────────┘
                                   │ COS dataset v2 (50k) + diff dataset v3
           ┌───────────────────────┼──────────────────────────────────┐
           │                       │                                  │
    ┌──────▼──────┐       ┌────────▼────────┐              ┌─────────▼─────────┐
    │  OPTION 1   │       │    OPTION 2     │              │     OPTION 3      │
    │             │       │                 │              │                   │
    │ FiLM-FNO v1 │       │  FiLM-FNO v2   │              │ Differential FNO  │
    │ R² = 0.796  │       │  R² ≈ 0.92+    │              │ + Jacobian Head   │
    │             │       │                 │              │ λ_jac = 0.05     │
    │ 3D Reparam  │       │ Convergence     │              │                   │
    │ Calibration │       │ Benchmarks      │              │ Autograd Newton   │
    │ FIM: 1301×  │       │ MC vs COS bias  │              │ Gauss-Newton      │
    │ cond reduce │       │ study           │              │ 1 backward pass   │
    └─────────────┘       └─────────────────┘              └───────────────────┘
```

### Option 1 — FiLM-FNO + Reparameterized Calibration

**Model:** `MirrorPaddedFNO2dWithAttention` — FiLM-conditioned Fourier Neural Operator
with mirror padding (prevents spectral boundary artifacts) and multi-head self-attention.

| Property | Value |
|---|---|
| Architecture | 4 FNO layers, width=32, modes=12; FiLM conditioning; attention |
| Parameters | ~686K (FNO backbone) |
| Input | (8, 11, 2) spatial grid + (6,) parameter vector |
| Output | (8, 11) implied volatility surface |
| Training data | MC dataset v1 (~10k, GPU Euler-Maruyama) |
| Val R² | **0.796** |
| Calibration | 3-start L-BFGS in (v₀, ζ, λ) space; ghost κ=1, θ=0.08, H=0.08 |
| v₀ recovery error | **< 2%** (0% noise) |
| ζ/λ recovery error | ~15% (bounded by R²=0.796) |
| FIM condition (6D) | 1.49 × 10⁷ |
| FIM condition (3D) | 1.15 × 10⁴ (**1301× reduction**) |

Key files: `src/fno_model.py`, `src/train_fno.py`, `src/calibrate.py` (Option 1 version),
`src/fim_analysis.py`, `src/validation.py`, `src/app_fno.py`

### Option 2 — Fourier-COS Exact Pricer + FNO v2

Replaces the MC dataset with **exact Fourier-COS pricing** (no systematic bias), generating
50k samples via Sobol quasi-random sequences.

| Property | Value |
|---|---|
| Pricer | CPU Fourier-COS, N_factors=20, N_cos=64 |
| Dataset | 50k Sobol samples; NaN ~1% at T=0.1 (filled by 2D interpolation) |
| NaN cause | Rough kernel (H=0.08) + short maturity → CF near singularity |
| FNO v2 | Same architecture; trained on exact labels |
| Val R² | **≈ 0.92+** (vs 0.796 for v1 on MC data) |
| Convergence | N=40 Bernstein required for σ=0.5; N=20 gives 29bp error at T=1.0 |

Benchmarks included:
- `benchmarks/convergence_N_factors.py` — N=5,10,20 vs N=40 reference
- `benchmarks/vs_cuda_mc.py` — systematic MC bias per maturity (expected 5–20bp at T=0.1)
- `benchmarks/validate_fno_v2.py` — FNO v1 vs v2 quality comparison

Key files: `src/pricing_engine.py`, `src/generate_dataset_v2.py`, `benchmarks/`

### Option 3 — Differential Machine Learning + Newton-Raphson Calibration

Implements *Huge & Savine (2020)* "Differential Machine Learning" for the FNO surrogate:
the model learns both **IV surfaces** and their **exact Jacobians** ∂IV/∂θ simultaneously.

**DifferentialFNO architecture:**
```
theta (6,) ──► FiLM-FNO backbone ──► IV surface (8, 11)
           └──► JacobianHead MLP ──► dIV/dtheta (8, 11, 5)
                [6→256→256→256→440, dropout=0.20]
```

**Training details:**
- Loss: `L = L_IV + 0.05 · L_Jac` (λ_jac=0.05 prevents Jacobian overfitting)
- T=0.1 excluded from Jacobian loss (noisy FD at short maturities)
- Bug fixed: validation used unmasked MSE vs masked training loss → caused 350× train/val gap
- Final: IV val = 2×10⁻³, Jac val = 0.40 (train/val balanced, no overfitting)

**Autograd calibration** (`src/calibrate_fast.py`):
Instead of using the trained JacobianHead, compute exact ∂IV_FNO/∂θ via `torch.autograd`
in a single backward pass — giving noise-free analytical Jacobians from the smooth FNO.

```python
J = torch.autograd.functional.jacobian(fno_iv_flat, theta)  # 1 backward pass
# vs 5-point FD: 10 forward passes
```

Newton step: `δθ = -(JᵀJ + εI)⁻¹ Jᵀ r` with backtracking line search and LM regularization.

| Property | Value |
|---|---|
| Dataset | 50k samples + (50k, 8, 11, 5) Jacobian tensors; 105MB |
| FD order | 5-point central differences per parameter |
| NaN masking | 4.54% cells interpolated; T=0.1 excluded from Jac loss |
| Autograd speedup | ~5–10× vs 5-point FD (1 backward vs 10 forward passes) |
| DiffFNO parameters | 932,905 (FNO: 686K + JacHead: 246K) |

Key files: `src/train_fno_differential.py`, `src/generate_dataset_v3_differential.py`,
`src/calibrate_fast.py`

---

## Baseline Pipeline (Phases 1–3)

The original thesis baseline using the **classical Heston model** (not rough):

| Phase | Description | Key File |
|---|---|---|
| Phase 1 | MLP surrogate: 5 params → 88-pt Total Variance surface | `src/model.py` |
| Phase 2 | MC Dropout uncertainty: 100 forward passes → ±2σ bounds | `src/calibrator.py` |
| Phase 3 | LSTM temporal dynamics: 10-day surface history → next-day params | `src/seq_model.py` |
| UI | Interactive Streamlit dashboard (Cytoscape.js + KaTeX) | `src/app.py` |

---

## Project Structure

```
derivatives/
├── CONTEXT.md                    # Architectural map for AI agents / onboarding
├── README.md
├── setup.py                      # CUDA extension build script
├── make_thesis_assets.sh         # Generate all thesis figures
├── build_all_tex.sh              # Compile all LaTeX documents
│
├── src/
│   ├── cuda_engine.cu            # GPU Lifted Rough Heston MC (CUDA C++)
│   ├── pricing_engine_gpu.py     # Python/pybind wrapper for CUDA engine
│   ├── pricing_engine.py         # CPU Fourier-COS pricer (N Bernstein factors)
│   │
│   ├── fno_model.py              # FiLM-FNO architectures (v1 + attention variant)
│   ├── normalizers.py            # ParameterNormalizer, IVSurfaceNormalizer
│   │
│   ├── generate_dataset.py       # MC dataset generator (v1, GPU)
│   ├── generate_dataset_v2.py    # Fourier-COS dataset (v2, 50k + nan_mask)
│   ├── generate_dataset_v3_differential.py  # v3 + 5-pt FD Jacobians (105MB)
│   │
│   ├── train_fno.py              # FNO v1/v2 training (ATM-weighted Huber + arb reg)
│   ├── train_fno_differential.py # DifferentialFNO training (IV + Jac loss)
│   │
│   ├── calibrate.py              # Option 2 standard 6D + Option 1 3D calibration
│   ├── calibrate_fast.py         # Option 3 Gauss-Newton + autograd Jacobians
│   ├── fim_analysis.py           # Fisher Information Matrix (6D vs 3D condition)
│   ├── validation.py             # Noise-robustness parameter recovery tests
│   │
│   ├── app_fno.py                # Streamlit: FNO calibration UI (3D/6D toggle)
│   ├── greeks_autograd.py        # Delta / Gamma / Vega via autograd
│   ├── fno_greeks.py             # Greeks for FNO surrogate
│   ├── iv_inverter.py            # Black-Scholes IV inversion (Newton-Raphson)
│   │
│   ├── model.py                  # Baseline MLP surrogate (Phase 1)
│   ├── calibrator.py             # Baseline L-BFGS-B calibrator (Phase 1)
│   ├── seq_model.py              # LSTM temporal dynamics (Phase 3)
│   ├── train.py / train_seq.py   # Baseline training scripts
│   ├── data_loader.py            # Baseline data pipeline
│   └── app.py                    # Baseline Streamlit UI (Phases 1–3)
│
├── benchmarks/
│   ├── convergence_N_factors.py  # Bernstein N-factor study vs N=40 reference
│   ├── vs_cuda_mc.py             # MC vs Fourier-COS systematic bias per maturity
│   ├── validate_fno_v2.py        # FNO v1 vs v2 R², MAE, Jacobian norms
│   └── convergence_results.txt   # Stored results: N=20 → 29bp error at T=1.0
│
├── artifacts/
│   ├── models/                   # Best validation checkpoints + normalizers
│   │   ├── fno_best.pth          # FNO v1 (R²=0.796, MC dataset)
│   │   ├── fno_v2_best.pth       # FNO v2 (R²≈0.92+, COS dataset)
│   │   ├── fno_diff_best.pth     # DifferentialFNO best checkpoint
│   │   └── *_normalizer*.npz     # z-score normalizers (v1, v2, diff)
│   ├── weights/                  # SWA-averaged production models
│   │   ├── fno_v2_final_prod.pth
│   │   ├── fno_diff_final_prod.pth
│   │   ├── heston_best.pth       # Baseline MLP
│   │   └── heston_lstm_best.pth  # Baseline LSTM
│   └── scalers/                  # Baseline MLP scalers (pkl)
│
├── tex/
│   ├── literature_review/        # Master's thesis literature review
│   └── presentation/             # Defense slides (Beamer, Russian)
│
├── data/                         # Datasets — NOT tracked in git (reproduced by scripts)
│   ├── DeepRoughDataset.npz      # MC dataset v1 (~10k samples, 15MB)
│   ├── DeepRoughDataset_v2_fourier.npz    # COS dataset v2 (50k, 32MB)
│   └── DeepRoughDataset_v3_differential.npz  # Diff dataset v3 (50k+Jac, 105MB)
│
└── research/                     # Reference PDFs (papers)
```

---

## Quickstart

### Prerequisites

- **Linux** (tested: Arch Linux)
- **Python 3.14** (exact — CUDA extension ABI is version-specific)
- **CUDA 12.6** + NVIDIA GPU (RTX 3060 or better recommended)

### Setup

```bash
git clone https://github.com/Execorn/derivatives.git
cd derivatives

# Create venv and install dependencies
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt

# Build the CUDA extension (required for dataset generation and GPU pricing)
python setup.py build_ext --inplace
```

### FNO Pipeline (Options 1–3)

```bash
# Step 1: Generate Fourier-COS dataset (GPU, ~10 min for 50k samples)
python src/generate_dataset_v2.py

# Step 2: Train FNO v2 (GPU, ~2 hours for 500 epochs)
python src/train_fno.py

# Step 3 (Option 3): Generate differential dataset with Jacobians (~20 min)
python src/generate_dataset_v3_differential.py

# Step 4 (Option 3): Train DifferentialFNO
python src/train_fno_differential.py --lambda-jac 0.05

# Launch FNO calibration UI
streamlit run src/app_fno.py
```

### Baseline Pipeline (Phases 1–3)

```bash
python src/data_loader.py
python src/train.py --epochs 200
python src/calibrator.py

python scripts/generate_seq_data.py
python src/train_seq.py --epochs 200

streamlit run src/app.py
```

### Skip Training (use pre-trained weights from artifacts/)

```bash
# FNO UI directly
streamlit run src/app_fno.py

# Baseline UI directly
streamlit run src/app.py
```

---

## Key Results

### Option 1 — Reparameterized Calibration

| Metric | Value |
|---|---|
| FIM condition: 6D space | 1.49 × 10⁷ |
| FIM condition: 3D (v₀, ζ, λ) | 1.15 × 10⁴ |
| **FIM reduction factor** | **1301×** |
| v₀ recovery (0% noise) | **< 2% error** |
| ζ, λ recovery (0% noise) | ~15% (bounded by FNO v1 R²) |
| Calibration: 3-start L-BFGS | Stable, converges in < 1s |

### Option 2 — Fourier-COS vs Monte Carlo

| Metric | Value |
|---|---|
| FNO v1 R² (MC labels) | 0.796 |
| FNO v2 R² (COS labels) | **≈ 0.92+** |
| N=20 Bernstein error at T=1.0 | **29 bp** vs N=40 (insufficient) |
| N=40 computation time | 274s per surface (CPU) |
| T=0.1 Fourier-COS | **NaN** for σ > 0.3, H=0.08 (known singularity) |

### Option 3 — Differential FNO

| Metric | Value |
|---|---|
| DiffFNO IV val loss (final) | 2.0 × 10⁻³ |
| DiffFNO Jac val loss (final) | **0.40** (vs 40.9 before bug fix) |
| Jacobian overfitting fix | Masked loss (T=0.1 excluded) + dropout=0.20 + λ=0.05 |
| Autograd vs FD speedup | ~5–10× (1 backward pass vs 10 forward) |
| Dataset generation speed | 68 surfaces/sec (RTX 3060 Laptop) |

---

## Running Benchmarks

```bash
# Bernstein N-factor convergence (CPU, ~10 min)
CUDA_VISIBLE_DEVICES="" python benchmarks/convergence_N_factors.py

# MC vs COS systematic bias (CPU, ~30 min for 200 samples)
CUDA_VISIBLE_DEVICES="" python benchmarks/vs_cuda_mc.py

# FNO v1 vs v2 quality check
python benchmarks/validate_fno_v2.py

# FIM condition number analysis
python src/fim_analysis.py

# Parameter recovery validation (noise robustness)
python src/validation.py

# Newton calibration speed test (autograd vs FD)
python src/calibrate_fast.py
```

---

## References

- El Euch, O., Gatheral, J. and Rosenbaum, M. (2019). *Roughening Heston*. Risk Magazine.
- Horvath, B., Muguruza, A. and Tomas, M. (2019). *Deep Learning Volatility*. SSRN 3322085.
- Huge, B. and Savine, A. (2020). *Differential Machine Learning*. SSRN 3775622.
- Fouque, J.-P. et al. (2011). *Multiscale Stochastic Volatility*. Cambridge University Press.
- Cont, R. and Da Fonseca, J. (2002). *Dynamics of Implied Volatility Surfaces*. Quantitative Finance.
- Gal, Y. and Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. ICML.
- Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility*. RFS.
- Bergomi, L. (2016). *Stochastic Volatility Modeling*. CRC Press.
