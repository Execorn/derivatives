# DeepVol: Neural Operators & Differentiable Pricing for Quantitative Derivatives

> **Master's Thesis Project** — MIPT (ФПМИ, Кафедра БИТ), 2026  
> *Нейронные сети в ценообразовании производных финансовых инструментов*

**DeepVol** is a GPU-accelerated quantitative library for volatility surface calibration, multi-asset equity autocallable valuation, triangulated FX/commodity cross pricing, and frictional deep hedging.

- **Precision Segregation**: Analytic curves, dividend schedules, Riccati ODEs, and Black-76 inversions run in `float64`; neural surrogates run in `float32` as bounded residual control variates.
- **Model Governance (SR 11-7 / SR 26-2)**: Every neural surrogate is guarded by online Mahalanobis out-of-distribution (OOD) detection, Higham correlation PSD repair, no-arbitrage surface projections, and deterministic fallback solvers.

---

## Core Engines

### 1. Multi-Asset Equity Autocallables (`EQAutocallable`)
Prices seasoned worst-of memory Phoenix autocallables under Dupire local volatility (`D <= 8` equities, `M <= 48` observation periods, 113-field Murex `ConstantMessageMap` DAG schema) via a 3-tier hybrid architecture:

1. **Tier 1 — Exact `float64` Pre/Post-Processing & Analytical Proxy (`< 35 us`)**:
   - Parses `@<40-hex>` Murex DAG payloads, fits clamped cubic zero-rate (`ClampedYieldCurve`) and repo spread (`RepoCurve`) curves, and collapses historical fixings (`collapse_seasoned_schedule`) into Markovian sufficient statistics (`C_mem`, `I_KI_past`, `PV_in_transit`).
   - Computes escrowed cash + proportional dividend forwards with cross-currency quanto adjustment and evaluates a closed-form **Stulz–Margrabe + 32-node Gauss–Hermite Markovian conditional survival proxy** (`compute_analytical_autocall_baseline`).
2. **Tier 2 — Masked $S_D$-Equivariant Graph Neural Operator (`MaskedEQAutocallSurrogate`)**:
   - Canonicalizes basket order ($\pi_{\text{lex}}$) and applies worst-of weight-gated `MaskedEGNOLayer` message passing, smooth `OrderStatisticPooling` (`LogMeanExp`), and a CuDNN-free `ParallelScheduleTransformer` (`SDPBackend.MATH`).
   - Predicts dispersion-attenuated residual stopping hazards $\Delta z_{h,m}$, coupon logits $\Delta z_{q,m}$, and terminal knock-in put loss $\Delta L_{\text{put}}$ anchored on the Tier-1 proxy, guaranteeing exact $S_D$ permutation invariance (`0.00e+00`) and zero non-binding Greek leakage.
3. **Hybrid Chain-Rule Greeks & SR 26-2 Guardian**:
   - Computes all 8 Greeks (`Delta`, `Gamma`, `MultivariateAggregatedCashGamma`, `Vega`, `Vanna`, `Volga`, `Rho`, `DividendsSensitivities`, `CorrelationSensitivities`) by combining `float64` autograd through the Tier-1/Tier-2 graph with analytical surface/curve Jacobians.
   - `EQAutocallModelGuardian` enforces Higham PSD projection, barrier pin-risk checks, 16-D Ledoit–Wolf Mahalanobis OOD gating, and dispersion-scaled residual bounds (`50–200 bps`), routing out-of-distribution or exotic (`OneStar`, `ParticipationCall`) trades to the 32,768-path Sobol Dupire QMC solver (`price_eq_autocall_dupire_qmc`).

### 2. Triangulated FX & Commodity Cross Engine
Prices synthetic cross options $S^\times = S_1 S_2$ (e.g., `XAU/RUB = XAU/USD × USD/RUB`) from two component leg smiles and correlation $\rho$:

1. **Smile Canonicalization**: Inverts 5-pillar FX quotes (`ATM`, `25D RR/BF`, `10D RR/BF` across `spot_pna`, `spot_pa`, `forward_pna`, `forward_pa`, and `DNS`/`ATMF` conventions) to arbitrage-free raw SVI / SSVI slices.
2. **$\beta$-Split Analytic Baseline & DML Residual**: Combines a 6-step fixed-point $\mathbb{Q}^B$ log-moneyness projection with a 5-member `ResidualCrossVolNet` ensemble trained with Sobolev differential labels against a 2D Gaussian copula teacher.
3. **Second-Order Taylor Jets**: Propagates $(x, \dot{x}, \ddot{x})$ through `taylor_jets.py` for closed-form sticky-moneyness and sticky-strike Greeks without finite differences.

### 3. Rough Volatility & Fourier Neural Operators (FNO)
Maps model parameters $\theta = (\kappa, \theta, \sigma, \rho, v_0, H)$ under Rough Heston (El Euch & Rosenbaum, 2019), SABR, SSVI, Rough Bergomi, and Grey Rough Bergomi to implied volatility surfaces via `MirrorPaddedFNO2d`. Fisher Information Matrix (FIM) coordinate reparameterization lowers calibration Hessian conditioning by $1300\times$.

### 4. Differentiable Hedging (D-XVA)
Trains recurrent policy networks end-to-end through Jäckel's differentiable PIVOT implied volatility solver to minimize realized P&L variance under proportional and nonlinear transaction costs.

---

## Empirical Results & Benchmarks

| Engine / Task | Metric | Measured Result | Reference / Baseline |
| :--- | :--- | :--- | :--- |
| **`EQAutocallable` (Reference Trade)** | `ProductValue` / 37-Row `CashFlows` Diff | **`+1.05 bps` (`0.36 SE`) / `0 ULP` on 36 deterministic rows** | Murex `MonteCarloWithAAD` (`32,768` paths) |
| **`EQAutocallable` Surrogate** | Out-of-Sample `ProductValue` MAE | **`2.14 bps`** (`<= 8.0 bps` target) | 32,768-path Sobol Dupire QMC |
| **`EQAutocallable` Latency (`B = 1`)** | `PV` + All 8 Greeks + 37 `CashFlows` | **`1.40 ms` warm / `6.23 ms` cold** (`403x–1,794x` speedup) | `2,509.6 ms` Murex `MonteCarloWithAAD` |
| **`EQAutocallable` Throughput** | Batched GPU valuation + Greeks (`B = 1024`) | **`0.0149 ms/trade`** (`67,114 trades/s`) | Pre-featurized Arrow / CUDA batch |
| **FX/Commodity Cross Engine** | In-Distribution Price RMSE / MAE | **`0.35 bps` RMSE / `0.67 bps` MAE** | 2D Gaussian Copula ($\mathbb{Q}^B$) |
| **FX/Commodity Cross Latency** | Single quote (GPU) / 10k Arrow batch | **`0.196 ms` p50 / `9.45 ms` p99** | Full price + 13 Taylor-jet Greeks |
| **Rough Heston FiLM-FNO** | Surface MAE / Batch calibration (1,024) | **`5.8 bps` ($R^2 = 0.9991$) / `4.0 ms`** ($1400\times$ speedup) | Fourier-COS ($N_{\text{cos}}=128$, `5.6 s`) |
| **Grey Rough Bergomi FNO** | Surface MAE (Active Learning) | **`0.76 bps`** | CUDA Mittag-Leffler + cuFFT MC |
| **1D PDE + Residual MLP Autocall** | Valuation RMSE | **`0.99 bps` (trimmed) / `1.14 bps` (raw)** | 50,000-path Sobol QMC |

---

## Installation

Requires Python 3.9+ and PyTorch 2.0+ (CPU execution is supported out of the box across all engines; CUDA accelerates batched inference and QMC simulation).

```bash
uv sync
# Optional: compile C++/CUDA extensions for Grey Rough Bergomi
uv run python setup.py build_ext --inplace
```

See [`docs/fx_cross_cpu_laptop_guide.md`](docs/fx_cross_cpu_laptop_guide.md) for standalone CPU-only installation and `XAU/RUB` pricing examples.

---

## Usage & API Endpoints

### 1. FastAPI Server (REST, Apache Arrow IPC & WebSocket)
```bash
PYTHONPATH=src .venv/bin/uvicorn deepvol.api.server:app --host 0.0.0.0 --port 8000
# OpenAPI docs: http://localhost:8000/docs
```

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `POST` | `/v1/price/eq_autocall` | Multi-asset `EQAutocallable` valuation, 37-row `CashFlows`, all 8 Greeks, and SR 26-2 Guardian telemetry |
| `POST` | `/v1/price/eq_autocall/batch_arrow` | Zero-copy Apache Arrow IPC batch valuation + first-order Greeks for `EQAutocallable` portfolios |
| `POST` | `/fx_cross/price` | Single triangulated FX/commodity cross option price, 13 Taylor-jet Greeks, and OOD telemetry |
| `POST` | `/fx_cross/price_batch` | Columnar batch pricing up to 10,000 contracts (`application/json` or `application/vnd.apache.arrow.stream`) |
| `POST` | `/autocall/price` | 1D PDE + 5-member residual MLP autocall valuation and Greeks |
| `POST` | `/iv_surface` | FiLM-FNO implied volatility surface evaluation |
| `POST` | `/calibrate/{model_name}` | Inverse surface calibration (`rough_heston`, `heston`, `sabr`, `ssvi`, `rbergomi`) |
| `WS` | `/ws/risk` | Real-time portfolio risk telemetry stream |

### 2. Python Quickstart: Multi-Asset `EQAutocallable` Pricing & All 8 Greeks
```python
from pathlib import Path
from deepvol.api.eq_autocall_runtime import EQAutocallRuntime
from deepvol.api.schemas_eq_autocall import parse_murex_autocall_dag

req, bank_ref = parse_murex_autocall_dag(Path("eq_autocallable_request.txt"))
req_with_greeks = req.model_copy(
    update={
        "Queries": [
            "ProductValue", "CashFlows", "Delta", "Gamma",
            "MultivariateAggregatedCashGamma", "Vega", "Rho",
            "DividendsSensitivities", "Vanna", "Volga", "CorrelationSensitivities",
        ]
    }
)

runtime = EQAutocallRuntime(checkpoint_path="artifacts/checkpoints/eq_autocall_egno.pt")
resp = runtime.price_trade(req_with_greeks)
print(f"ProductValue: {resp.ProductValue:.8f} (Bank MC: {bank_ref['ProductValue']:.8f})")
print(f"VKCO_RUB Delta: {resp.Delta['VKCO_RUB']:.8f}, CashGamma: {resp.MultivariateAggregatedCashGamma:.8f}")
```

### 3. Web & Research Dashboards
```bash
# Next.js web interface (http://localhost:3000)
npm --prefix app_frontend run dev

# Streamlit research suite (http://localhost:8501)
PYTHONPATH=src .venv/bin/streamlit run src/deepvol/app/app_main.py
```

---

## Verification & Benchmarks

```bash
# Multi-Asset EQAutocallable test suite (96 tests) & latency benchmark (G2.1-G2.4)
.venv/bin/pytest tests/test_eq_autocall_*.py tests/test_order_statistic_pooling.py tests/test_api_eq_autocall.py -q
.venv/bin/python benchmarks/eq_autocall_bench.py --suite all

# Triangulated FX/Commodity Cross clean-room audit suite (141 tests)
.venv/bin/pytest tests/audit -m audit -q

# Full repository test suite
.venv/bin/pytest -q
```

---

## Repository Layout

```text
derivatives/
├── src/deepvol/
│   ├── api/            FastAPI server, Murex DAG & FX schemas, Arrow IPC codecs, runtimes
│   ├── arbitrage/      SVI/SSVI parameterization and no-arbitrage surface completion
│   ├── calibration/    Gauss-Newton FNO, Sobol+LKJ generators, and MLSV particle calibration
│   ├── greeks/         Portfolio Greeks and P&L attribution
│   ├── hedging/        Recurrent deep hedging, PIVOT IV solver, and D-XVA loss
│   ├── market/         Clamped yield/repo curves, FX delta conventions, SVI canonicalizer
│   ├── models/
│   │   ├── autocall/   Multi-asset EQAutocallable (collapser, dividends, proxy, Dupire QMC, Greeks)
│   │   └── ...         Heston, Rough Heston, SABR, rBergomi, gRB, FX Cross, Taylor jets
│   ├── mrm/            SR 11-7 / SR 26-2 Guardians (EQAutocall, FX Cross), Mahalanobis OOD, PSI
│   ├── surrogates/     MaskedEQAutocallSurrogate (EGNO), OrderStatisticPooling, FiLM-FNO
│   ├── training/       Spread-attenuated Sobolev trainers & differential ML pipelines
│   └── app/            Streamlit research dashboards
├── app_frontend/       Next.js web frontend (WebGL surface viewer & Autocall pricer)
├── artifacts/          Trained checkpoints (EGNO, FX Cross, FNO), normalizers, and model cards
├── benchmarks/         Latency, throughput, and numerical convergence benchmark suites
├── deploy/             Docker, Kubernetes, Prometheus, and Grafana deployment configs
├── docs/               CPU-only laptop guide and HTML5 technical presentation
├── formal_proofs/      Lean 4 formal verification workspace
├── notebooks/          Jupyter notebooks for calibration, hedging, and empirical studies
├── scripts/            Dataset generation, training, and symbolic verification utilities
├── tests/              Unit, integration, adversarial stress, and clean-room audit test suites
└── tex/                Master's thesis LaTeX source and presentation slides
```

---

## References

- Abi Jaber, E. (2019). *Lifting the Heston model.* Quantitative Finance, 19(12), 1995–2013.
- Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). *Deep hedging.* Quantitative Finance, 19(8), 1271–1291.
- El Euch, O., & Rosenbaum, M. (2019). *The characteristic function of rough Heston models.* Mathematical Finance, 29(1), 3–38.
- Gatheral, J., & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.* Quantitative Finance, 14(1), 59–71.
- Huge, B., & Savine, A. (2020). *Differential Machine Learning.* Risk.
- Li, Z., et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR.
- Stulz, R. M. (1982). *Options on the minimum or the maximum of two risky assets.* Journal of Financial Economics, 10(2), 161–185.
