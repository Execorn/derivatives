# Deep Learning Calibration of the Heston Stochastic Volatility Model 📈

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end **Neural Network surrogate pipeline** for real-time calibration of the Heston Stochastic Volatility Model.
Implements the methodology of *Horvath, Muguruza & Tomas (2019) "Deep Learning Volatility"* with a modern PyTorch
architecture, C² smooth ELU activations, strict financial no-arbitrage constraints, and an interactive Streamlit dashboard.

---

## Mathematical Foundation

### The Heston Model

$$dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S$$
$$dv_t = \kappa(\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW_t^v, \quad dW_t^S dW_t^v = \rho\,dt$$

| Parameter | Symbol | Meaning |
|---|---|---|
| Mean reversion speed | κ | How fast variance returns to θ |
| Long-run variance | θ | Steady-state variance |
| Vol of vol | σ | Volatility of the variance process |
| Correlation | ρ | Asset–variance correlation |
| Initial variance | v₀ | Variance at t = 0 |

### The Feller Condition

$$2\kappa\theta > \sigma^2$$

Ensures the variance process $v_t$ remains strictly positive. Enforced as a hard penalty (λ = 10³) in the optimizer.

---

## Neural Network Architecture

```
Input(5) → [Linear → ELU] × 4  →  Linear(30, 88)
```

| Property | Value |
|---|---|
| Hidden layers | 4 × 30 neurons |
| Activation | ELU (C² smooth — valid Hessian for 2nd-order calibration) |
| Parameters | 5,698 |
| Input | 5 Heston params, MinMaxScaler → [−1, 1] |
| Output | 88-point IV surface (8 maturities × 11 strikes), StandardScaler |
| Val RMSE | **0.02640** |
| Calibration | **47–135 ms** (L-BFGS-B, CPU) |

---

## Project Structure

```
derivatives/
├── .gitignore
├── README.md
├── setup_and_run.sh          # Linux: one-shot setup + pipeline + UI
├── setup_and_run.bat         # Windows: one-shot setup + pipeline + UI
├── build_all_tex.sh          # Linux: compile all LaTeX documents
├── build_all_tex.bat         # Windows: compile all LaTeX documents
│
├── src/                      # Pure Python source code
│   ├── requirements.txt
│   ├── model.py              # HestonSurrogateMLP (PyTorch nn.Module)
│   ├── data_loader.py        # Data pipeline + scaler persistence
│   ├── train.py              # Training loop (Adam + ReduceLROnPlateau)
│   ├── calibrator.py         # L-BFGS-B + Feller + no-arbitrage penalties
│   ├── benchmark_plots.py    # Publication-quality 3D surface plots
│   ├── greeks_autograd.py    # Jacobian + Hessian proof (C² vs ReLU)
│   └── app.py                # Streamlit interactive demo
│
├── artifacts/                # Generated binary artefacts (git-ignored, .gitkeep)
│   ├── weights/              # heston_best.pth — trained model checkpoint
│   ├── scalers/              # feature_scaler.pkl, target_scaler.pkl
│   └── reports/              # audit_report.md, compile_presentation.sh
│
├── images/
│   └── generated/         # surface_fit.png and other output images
│
├── logs/                     # Runtime logs (git-ignored, .gitkeep)
│
├── tex/
│   ├── .latex_cache/         # Compilation junk (git-ignored, .gitkeep)
│   ├── literature_review/
│   │   ├── main.tex          # Master's thesis literature review
│   │   ├── references.bib
│   │   ├── build.sh          # Linux: pdflatex + biber
│   │   └── build.bat         # Windows: pdflatex + biber
│   └── presentation/
│       ├── presentation.tex  # 8-slide Beamer defense presentation
│       ├── build.sh          # Linux: pdflatex (2 passes)
│       └── build.bat         # Windows: pdflatex (2 passes)
│
├── articles/                 # [READ-ONLY] Reference papers
└── data/                     # Original Horvath et al. dataset (HestonTrainSet.txt.gz)
```

---

## Quickstart

### Linux / macOS

**One command — handles everything:**
```bash
git clone <repo-url>
cd derivatives
bash setup_and_run.sh
```

This will:
1. Create `.venv/` and install all Python dependencies
2. Run the data pipeline (`data_loader.py`)
3. Train the surrogate model for 200 epochs (`train.py`)
4. Run a calibration smoke-test (`calibrator.py`)
5. Launch Streamlit at **http://localhost:8501**

**Skip training** (use existing weights in `artifacts/weights/`):
```bash
bash setup_and_run.sh --skip-train
```

### Windows

```bat
setup_and_run.bat
rem or
setup_and_run.bat --skip-train
```

### Manual Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r src/requirements.txt

# Run pipeline manually
python src/data_loader.py
python src/train.py --epochs 200
python src/calibrator.py
python src/greeks_autograd.py

# Launch UI
streamlit run src/app.py
```

---

## Compiling the LaTeX Documents

### Linux / macOS

```bash
# Install TeX Live (Arch Linux)
sudo pacman -S texlive-most

# Compile everything
bash build_all_tex.sh

# Or compile individually
bash tex/literature_review/build.sh
bash tex/presentation/build.sh
```

### Windows

```bat
REM Install MiKTeX: https://miktex.org/download
build_all_tex.bat

REM Or individually
tex\literature_review\build.bat
tex\presentation\build.bat
```

PDFs are placed next to their `.tex` source files. All compilation junk (`.aux`, `.log`, `.toc`, etc.) goes to `tex/.latex_cache/`.

---

## Benchmark Results

| Metric | Value |
|---|---|
| Calibration time | **47–135 ms** (L-BFGS-B, CPU) |
| Val RMSE | 0.02640 (better than paper's ~0.028) |
| Surface grid | 8 maturities × 11 strikes = 88 points |
| Feller condition | Hard penalty λ = 10³ — always satisfied post-calibration |
| Calendar arbitrage | Soft L2 penalty λ = 10⁻⁴ on ∂IV/∂T violations |
| Butterfly arbitrage | Soft L2 penalty λ = 10⁻⁴ on ∂²IV/∂K² violations |
| ELU Hessian ‖H‖_F | 1.039 (vs. ReLU: 0.000 — proves C² smoothness) |

![Surface fit](images/generated/surface_fit.png)

---

## References

- Horvath, B., Muguruza, A. and Tomas, M. (2019). *Deep Learning Volatility*. SSRN 3322085.
- Itkin, A. (2019). *Deep learning calibration of option pricing models: some pitfalls and solutions*.
- Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility*. RFS.
