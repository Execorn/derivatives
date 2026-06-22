# What This Project Does — Plain Language Overview

This document explains the project in simple terms: what problem it solves,
what we built, how all the pieces connect, and how to actually run it end-to-end.

---

## The Problem We Solved

Options traders need to know the **fair price** of financial derivatives — contracts
that give the right to buy or sell an asset at a fixed price in the future.
Pricing these requires a mathematical model of how volatile the asset will be.

The best available model is the **Rough Heston** model. It is mathematically
sophisticated and fits real market data extremely well. The problem: computing
a single set of option prices from it takes **5–6 seconds** on a GPU. Traders
need to re-fit the model every few seconds as markets move. That is too slow by
a factor of ~1000.

---

## What We Built

We replaced the slow pricing calculation with a **neural network surrogate**
that learned to mimic it. After training once (4 hours on a GPU), the network
can reproduce any option surface the model would produce in **under 1 millisecond
— 1400x faster**.

This is not a black-box approximation. The network has mathematical structure
(Fourier Neural Operator) that matches the structure of the pricing problem.
The result is a full working system: from live market data ingestion, through
real-time calibration, to Greeks, P&L, and a REST API.

**Key numbers achieved:**
- Surrogate accuracy: R² = 0.9991, error = 5.8 basis points (0.058%)
- Calibration speed: 541 ms per date (was ~60 s)
- Streaming latency (p95): 668 ms for 20 market ticks
- Delta-hedge improvement: +5.1% variance reduction vs Black-Scholes
- Test suite: 535 tests, 0 failures

---

## The Six Layers — How It All Connects

Think of the project as a stack of six layers, each one feeding into the next:

```
LAYER 1  Mathematical Model
LAYER 2  Training Data Generator
LAYER 3  Neural Network (the surrogate)
LAYER 4  Calibration Engine
LAYER 5  Analytics & Market Data
LAYER 6  Interfaces (dashboard, API, notebooks)
```

### Layer 1 — The Mathematical Model

**File:** `src/pricing_engine_gpu.py`

The Rough Heston model describes how stock prices and their volatility evolve
over time. Given 6 input numbers (parameters), it produces a 2D grid of
implied volatilities — one number for each combination of option expiry (8 dates)
and strike price (11 levels). This 8×11 grid is called the **IV surface**.

Computing it requires solving a differential equation and running a Fourier
integral — expensive, but exact. This layer is the ground truth.

### Layer 2 — Training Data

**Files:** `src/generate_dataset_v4_learnable_h.py`, `data/*.npz`

We ran Layer 1 about 100,000 times with randomly chosen input parameters,
saving each (parameters → surface) pair. This took hours but only needed
to happen once. The saved dataset files (`data/DeepRoughDataset_v4_learnable_h.npz`)
are what the neural network trains on.

### Layer 3 — The Neural Network Surrogate

**Files:** `src/fno_model.py`, `src/normalizers.py`, `artifacts/weights/*.pth`

The **FiLM-conditioned Fourier Neural Operator (FNO)** learned the function
"parameters → IV surface" from the dataset. It has two parts:

- **FNO body**: processes the (T, K) grid using Fourier convolutions — operations
  that naturally capture the global structure of an IV surface.
- **FiLM conditioning**: the 6 input parameters steer the network's behaviour
  at every layer via learned scale-and-shift operations.

After training, the weights are saved to `artifacts/weights/fno_v3_final_prod.pth`.
The normalizers (`artifacts/models/*.npz`) handle converting parameters and
surfaces to/from the standardized range the network expects.

This layer is the core speedup. Once loaded (about 1 second), it answers
any query in < 1 ms.

### Layer 4 — Calibration

**Files:** `src/calibrate.py`, `src/calibration/batch_calibration.py`

Calibration is the inverse problem: given a real market IV surface, find the
6 parameters that make the model reproduce it.

We solve this with **Gauss-Newton iteration**:
1. Start with a guess for the parameters.
2. Ask the FNO: "what surface do these parameters produce?"
3. Compute the difference from the market surface.
4. Use PyTorch's automatic differentiation to compute the gradient.
5. Take a Newton step to reduce the error.
6. Repeat ~15 times until converged (RMSE < 50 basis points).

The whole process takes ~541 ms. `batch_calibration.py` runs this in parallel
for many historical dates, with resume capability (saves progress to JSON).

### Layer 5 — Analytics & Market Data

**Files:** `src/market/`, `src/greeks/`, `src/arbitrage/`, `src/analysis/`

Once we have calibrated parameters θ*, we can compute many things:

| What | File | How |
|------|------|-----|
| VIX futures term structure | `market/vix_futures.py` | Integrates E[v_T] via Riccati ODE |
| Portfolio Greeks (delta, gamma, vega...) | `greeks/portfolio_greeks.py` | FNO + finite differences, GPU-batched |
| P&L attribution | `greeks/pnl_attribution.py` | Taylor expansion in Greeks |
| Arbitrage-free surface completion | `arbitrage/surface_completion.py` | SVI fit + calendar/butterfly enforcement |
| Historical Hurst exponent dynamics | `analysis/hurst_dynamics.py` | Batch-calibrate daily, track H over time |
| Joint SPX+VIX calibration | `calibration/joint_calibration.py` | Combined loss function |
| Live BTC/ETH data | `market/deribit_ws.py` | WebSocket streaming from Deribit exchange |

### Layer 6 — Interfaces

**Files:** `src/app_fno.py`, `src/api/server.py`, `notebooks/`

Three ways to interact with the system:

1. **Streamlit dashboard** (`src/app_fno.py`): Visual interface. Move sliders
   for each parameter and watch the IV surface update in real time. Run Newton
   calibration interactively. Compute Greeks for a portfolio.

2. **FastAPI REST server** (`src/api/server.py`): Exposes the full system as a
   web API. Any program (Python, C++, Excel via HTTP) can send a request and
   get back an IV surface, Greeks, or calibrated parameters.

3. **Jupyter notebooks** (`notebooks/`): Step-by-step analysis of real market
   data — SPX options, VIX term structure, crypto calibration, Greeks.

---

## How to "Put It All Together" — End-to-End Walkthrough

### From scratch: price a surface

```bash
cd /home/execorn/programming/derivatives
source .venv/bin/activate
python - << 'PYTHON'
import sys; sys.path.insert(0, 'src')
import torch
from fno_model import MirrorPaddedFNO2d
from normalizers import ParameterNormalizer, IVSurfaceNormalizer
from calibrate import _load_normalizers, predict_surface
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MirrorPaddedFNO2d(param_dim=6).to(device)
model.load_state_dict(torch.load("artifacts/weights/fno_v3_final_prod.pth",
                                  map_location=device))
model.eval()
_load_normalizers("v3")

# kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, V0=0.04, H=0.10
theta = np.array([2.0, 0.04, 0.5, -0.7, 0.04, 0.10])
surface = predict_surface(theta, model)  # (8, 11) array of implied vols
print("ATM vol at 6 months:", surface[2, 5])
PYTHON
```

### From scratch: calibrate to real SPX market data

```bash
python - << 'PYTHON'
import sys; sys.path.insert(0, 'src')
from calibration.batch_calibration import calibrate_batch, results_to_dataframe

results = calibrate_batch(
    dates=["2024-01-02"],
    currency="SPX",
    device="auto",
    verbose=True,
)
df = results_to_dataframe(results)
print(df[["date", "kappa", "theta", "H", "rmse_bps", "converged"]])
PYTHON
```

### Run the interactive dashboard

```bash
streamlit run src/app_fno.py
# Open http://localhost:8501
```

### Run the REST API

```bash
uvicorn api.server:app --port 8000 --app-dir src
# Open http://localhost:8000/docs for interactive API explorer
curl -X POST http://localhost:8000/iv_surface \
  -H "Content-Type: application/json" \
  -d '{"kappa":2.0,"theta":0.04,"sigma":0.5,"rho":-0.7,"v0":0.04,"H":0.1}'
```

---

## What Was Achieved (Summary)

We took a mathematically rigorous stochastic volatility model that was too slow
for real-time use and built a complete production system around it:

- The slow Fourier-COS pricer (6 s) was replaced by a trained FNO (< 1 ms).
- Calibration went from minutes to under 1 second.
- The system connects to live market data (SPX via yfinance, BTC/ETH via Deribit).
- It computes all standard risk measures (Greeks, P&L attribution, VIX).
- Everything is exposed via REST API and a visual dashboard.
- The work is documented in a 51-page academic thesis and 535 automated tests.

The project demonstrates that modern deep learning (specifically operator learning
with the FNO architecture) can make computationally intractable quantitative
finance problems tractable in real time, without sacrificing mathematical rigor.
