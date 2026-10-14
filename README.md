# DeepVol: Neural Operators and Differentiable Pricing for Quantitative Derivatives

> **Master's Thesis Project** — MIPT (ФПМИ, Кафедра БИТ), 2026  
> *Нейронные сети в ценообразовании производных финансовых инструментов*

DeepVol is a GPU-accelerated quantitative library for option surface calibration, structured note valuation, triangulated FX/commodity cross pricing, and frictional deep hedging. Analytic solvers, Riccati integrations, and Black-76 inversions run in `float64`; neural surrogates run in `float32` behind statistical out-of-distribution (OOD) and no-arbitrage guards (SR 11-7 / SR 26-2).

---

## Core Engines & Mathematical Formulation

### 1. Rough Volatility & Fourier Neural Operators (FNO)
Under the Rough Heston model (El Euch & Rosenbaum, 2019), spot variance follows a fractional Volterra process with Hurst exponent $H \in (0, \tfrac{1}{2})$:

$$v_t = v_0 + \frac{1}{\Gamma(H + \tfrac{1}{2})} \int_0^t (t-s)^{H-\tfrac{1}{2}} \kappa(\theta - v_s)\,ds + \sigma \int_0^t (t-s)^{H-\tfrac{1}{2}} \sqrt{v_s}\,dW_s$$

Direct valuation solves the fractional Riccati ODE via an $N=40$ Lifted Heston Markovian approximation (Abi Jaber, 2019) coupled with Fang–Oosterlee Fourier-COS expansion. To accelerate inverse calibration, a FiLM-conditioned 2D Fourier Neural Operator (`MirrorPaddedFNO2d`) maps model parameters $\theta = (\kappa, \theta, \sigma, \rho, v_0, H)$ to the $8 \times 11$ implied volatility surface in a single forward pass. Fisher Information Matrix (FIM) coordinate reparameterization lowers the calibration Hessian condition number by $1300\times$ ($\kappa \approx 770$).

Additional stochastic volatility and rough-path models in `src/deepvol/models/` include SABR, arbitrage-free SSVI, Dupire local volatility, Rough Bergomi, Grey Rough Bergomi (via custom CUDA Mittag-Leffler and Wood–Chan circulant embedding kernels), McKean–Vlasov LSV, and signature SDEs.

### 2. Triangulated FX & Commodity Cross Engine
For synthetic cross pairs $S^\times = S_1 S_2$ (such as `XAU/RUB = XAU/USD × USD/RUB`) lacking liquid direct options markets, the engine prices European vanillas from the two leg smiles and correlation $\rho$:

1. **Canonicalization**: 5-pillar FX market quotes (`ATM`, `25D RR/BF`, `10D RR/BF` under `spot_pna`, `spot_pa`, `forward_pna`, `forward_pa`, and `DNS`/`ATMF` conventions) are inverted to strike space and fitted to arbitrage-free raw SVI slices (with SSVI fallback when the Durrleman density $g(k)$ approaches zero).
2. **$\beta$-Split Analytic Baseline**: A 6-step fixed-point projection under the intermediate measure $\mathbb{Q}^B$ splits cross log-moneyness across both legs ($\beta_1 + \beta_2 = 1$) while enforcing Roger Lee's moment slope bound ($\bar{L} \le 1.9$).
3. **Differential ML Residual**: A 5-member ensemble of $13 \to 64^3 \to 1$ SiLU networks (`ResidualCrossVolNet`), trained with Sobolev differential labels against a 2D Gaussian copula teacher, predicts the copula smile residual $\hat{\varepsilon}$. A $C^2$ quintic taper drives the neural contribution to exact zero for $|z| \ge 7$.
4. **Exact Second-Order Taylor Jets**: Forward propagation of $(x, \dot{x}, \ddot{x})$ through `taylor_jets.py` yields closed-form sticky-moneyness and sticky-strike Greeks (`Delta`, `CashDelta`, `NonAdaptedDelta`, `Gamma`, `CashGamma`, `Vega`, `Vanna`, `Vomma`, `Theta`, `CorrelationSensitivity`, and bucketed `Rho`) without finite differences or second-order autograd.

### 3. Multi-Fidelity Autocallable & Basket Notes
Path-dependent structured notes with discrete observation barriers are priced by decomposing valuation into a 1D Crank–Nicolson Gatheral PDE base solver plus a 5-member residual MLP (`CorrectionEnsemble`) over a 19-dimensional contract feature space:

$$V_{\text{served}} = V_{\text{PDE}} + \Delta \hat{V}_{\text{MLP}}$$

When ensemble epistemic standard deviation exceeds $\tau_{\text{OOD}} = 2.17\text{ bps}$, `AutocallGuardian` routes the contract back to the PDE or Sobol–Brownian Bridge Quasi-Monte Carlo solver. Multi-asset Worst-Of (WoF) basket autocallables use Equivariant Graph Neural Operators (EGNO) to preserve permutation symmetry across underlying assets.

### 4. Differentiable Hedging (D-XVA)
Recurrent policy networks (LSTM/GRU) are trained end-to-end through Jäckel's differentiable PIVOT implied volatility solver to minimize realized P&L variance under proportional and nonlinear transaction costs.

---

## Empirical Results

| Domain / Task | Metric | Measured Result | Reference / Baseline |
| :--- | :--- | :--- | :--- |
| **Rough Heston FiLM-FNO** | Surface MAE / $R^2$ | **5.8 bps** ($R^2 = 0.9991$) | Fourier-COS ($N_{\text{cos}}=128$) |
| **Rough Heston Calibration** | Batch latency (1,024 surfaces) | **4.0 ms** ($1400\times$ speedup) | 5.6 s direct COS |
| **Grey Rough Bergomi FNO** | Surface MAE (Active Learning) | **0.76 bps** | GPU Mittag-Leffler MC |
| **FX/Commodity Cross Engine** | Price RMSE / MAE (in-distribution) | **0.35 bps RMSE / 0.67 bps MAE** | 2D Gaussian Copula ($\mathbb{Q}^B$) |
| **FX/Commodity Cross Engine** | Cross Delta / Gamma / Vega RMSE | **0.055% $\Delta$ / 0.41% $\Gamma_{\text{ATM}}$ / 0.31% $\mathcal{V}$** | Exact copula finite differences |
| **FX/Commodity Cross Engine** | CashDelta conservation error | **$< 10^{-15}$ relative** (`0.00 RUB`) | $\text{CD}_\times = \text{CD}_1 = \text{CD}_2$ |
| **FX/Commodity Cross Latency** | Single quote (GPU CUDA Graph) | **0.196 ms p50 / 0.214 ms p99** | Full price + 13 Greeks |
| **FX/Commodity Cross Latency** | 10,000-contract columnar batch | **9.45 ms p99** (Apache Arrow IPC) | 64 market states |
| **Autocall Residual Ensemble** | Valuation RMSE | **0.99 bps** (trimmed) / **1.14 bps** (raw) | 50,000-path Sobol QMC |
| **Phoenix & Worst-Of Autocalls** | Valuation RMSE | **1.35 bps** (Phoenix) / **1.87 bps** (WoF EGNO) | Multi-asset QMC |

---

## Installation

Requires Python 3.9+, PyTorch 2.8+, and CUDA 12.8+ (CPU fallbacks are supported across all engines).

```bash
uv sync
# Optional: compile C++/CUDA extensions for Grey Rough Bergomi
uv run python setup.py build_ext --inplace
```

---

## Usage & Interfaces

### 1. FastAPI Server (REST, Apache Arrow IPC & WebSocket)
```bash
.venv/bin/uvicorn deepvol.api.server:app --host 0.0.0.0 --port 8000 --app-dir src
# Interactive OpenAPI documentation: http://localhost:8000/docs
```

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `POST` | `/fx_cross/price` | Single triangulated FX/commodity cross option price, full Greeks, and OOD telemetry |
| `POST` | `/fx_cross/price_batch` | Columnar batch pricing up to 10,000 contracts (`application/json` or `application/vnd.apache.arrow.stream`) |
| `GET` | `/fx_cross/health` | CUDA graph capture state, active `model_version`, and execution device |
| `POST` | `/autocall/price` | Multi-fidelity PDE + 5-member residual MLP autocall valuation and Greeks |
| `POST` | `/autocall/price_batch` | Vectorized batch autocall valuation |
| `POST` | `/iv_surface` | FiLM-FNO implied volatility surface evaluation |
| `POST` | `/calibrate/{model_name}` | Inverse surface calibration (`rough_heston`, `heston`, `sabr`, `ssvi`, `rbergomi`) |
| `POST` | `/greeks/{model_name}` | Surface and portfolio Greeks |
| `WS` | `/ws/risk` | Real-time portfolio risk telemetry stream |

### 2. Web & Research Dashboards
```bash
# Next.js 16 web interface (http://localhost:3000 and http://localhost:3000/autocall)
npm --prefix app_frontend run dev

# Streamlit multi-page research suite (http://localhost:8501)
PYTHONPATH=src .venv/bin/streamlit run src/deepvol/app/app_main.py
```

### 3. Python Example: Triangulated FX/Commodity Cross Pricing
```python
from fastapi.testclient import TestClient
from deepvol.api.server import app

payload = {
    "contract": {
        "identifier": "XAURUB",
        "parity": "Call",
        "strike": 382100.0,
        "as_of_date": "2026-09-09T00:00:00Z",
        "expiration_date": "2028-09-08T00:00:00Z",
        "settlement_date": "2028-09-11T00:00:00Z",
        "payment_currency": "RUB",
    },
    "market_data": {
        "leg1": {
            "pair": "XAUUSD",
            "spot": 4414.66,
            "foreign_discount_factor": 0.93091474,
            "domestic_discount_factor": 0.85210345,
            "smile": {"atm_vol": 0.2250, "rr25": 0.0098, "bf25": 0.0092, "rr10": 0.0216, "bf10": 0.0336},
        },
        "leg2": {
            "pair": "USDRUB",
            "spot": 86.551433,
            "foreign_discount_factor": 0.85210345,
            "domestic_discount_factor": 0.68421053,
            "smile": {"atm_vol": 0.1850, "rr25": 0.0120, "bf25": 0.0080, "rr10": 0.0250, "bf10": 0.0210, "delta_convention": "spot_pa"},
        },
        "correlation": 0.0,
        "settlement_discount_factor": 0.68415000,
    },
}

with TestClient(app) as client:
    response = client.post("/fx_cross/price", json=payload).json()
    print(response["product_value"], response["delta"], response["cash_delta"])
```

---

## Verification & Test Suite

```bash
# Full repository test suite
PYTHONPATH=src .venv/bin/pytest -q

# Clean-room mathematical, arbitrage, and runtime audit suite (141 tests)
PYTHONPATH=src .venv/bin/pytest tests/audit -m audit -q
```

---

## Repository Layout

```text
derivatives/
├── src/deepvol/
│   ├── api/            FastAPI server, Pydantic schemas, Arrow IPC codec, CUDA-graph runtime
│   ├── arbitrage/      SVI/SSVI parameterization and no-arbitrage surface completion
│   ├── calibration/    Gauss-Newton FNO, joint SPX+VIX, and MLSV particle calibration
│   ├── greeks/         Portfolio Greeks and P&L attribution
│   ├── hedging/        Recurrent deep hedging, PIVOT IV solver, and D-XVA loss
│   ├── market/         FX delta/ATM conventions, SVI slice canonicalizer, legacy DAG adapter
│   ├── models/         Heston, Rough Heston, SABR, rBergomi, gRB, Autocall PDE, FX Cross, Taylor jets
│   ├── mrm/            SR 11-7 / SR 26-2 Model Risk Guardians, Mahalanobis OOD, PSI drift tracking
│   ├── surrogates/     FiLM-FNO, CorrectionEnsemble, EGNO, ResidualCrossVolNet
│   └── app/            Streamlit research dashboards
├── app_frontend/       Next.js 16 web frontend (WebGL surface viewer & Autocall pricer)
├── artifacts/          Trained checkpoints, normalizers, model cards, and audit verification reports
├── notebooks/          Jupyter notebooks for calibration, hedging, and empirical studies
├── tests/              Unit, integration, and clean-room audit test suites
└── tex/                Master's thesis LaTeX source and presentation slides
```

---

## References

- Abi Jaber, E. (2019). *Lifting the Heston model.* Quantitative Finance, 19(12), 1995–2013.
- Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). *Deep hedging.* Quantitative Finance, 19(8), 1271–1291.
- El Euch, O., & Rosenbaum, M. (2019). *The characteristic function of rough Heston models.* Mathematical Finance, 29(1), 3–38.
- Fang, F., & Oosterlee, C. W. (2008). *A novel pricing method for European options based on Fourier-cosine series expansions.* SIAM Journal on Scientific Computing, 31(2), 826–848.
- Gatheral, J., & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.* Quantitative Finance, 14(1), 59–71.
- Huge, B., & Savine, A. (2020). *Differential Machine Learning.* Risk.
- Li, Z., et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR.
- Reiswich, D., & Wystup, U. (2010). *FX volatility smile construction.* Wilmott Journal, 2(2), 58–70.
