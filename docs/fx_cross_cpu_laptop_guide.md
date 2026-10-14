# FX & Commodity Cross Pricing on a CPU-Only Laptop (`XAU/RUB` Guide)

This guide covers installing and running the DeepVol Triangulated FX/Commodity Cross Pricing Engine (`XAU/RUB = XAU/USD × USD/RUB`) on a **CPU-only laptop** (Linux, macOS, or Windows) without an NVIDIA GPU, CUDA toolkit, or TensorRT installation.

---

## 1. Architecture & CPU Execution Summary

When `torch.cuda.is_available()` is `False`, the FX Cross runtime (`src/deepvol/api/fx_cross_runtime.py`) automatically selects `device="cpu"` and skips CUDA Graph capture:

- **Startup time**: `< 1 second` (no GPU kernel compilation or CUDA Graph capture required).
- **Single-quote latency on CPU**: `~0.31 ms` p99 (`CPUQuotePath`), returning the option price, all 13 analytical + neural-jet Greeks, and SR 26-2 Model Risk telemetry.
- **Precision**: FX market conventions, SVI/SSVI canonicalization, $\beta$-split fixed-point projection, Black-76 valuation, and second-order Taylor jets run in `float64`; the 5-member `13 -> 64 -> 64 -> 64 -> 1` SiLU ensemble runs in `float32`.
- **Pre-trained weights**: Tracked directly in Git under `artifacts/models/fx_cross/` (`228 KB` total):
  - `artifacts/models/fx_cross/residual_vol_v3_m0.pt` … `residual_vol_v3_m4.pt`
  - `artifacts/models/fx_cross/model_card.json` (Ledoit–Wolf Mahalanobis OOD covariance, threshold $q_{0.995}$, and feature normalizer statistics)

---

## 2. Fresh Installation on a CPU-Only Laptop

The default `pyproject.toml` pins GPU-only packages (`tensorrt`, `onnxruntime-gpu`) for server deployments. On a CPU-only laptop, install CPU PyTorch and the core runtime dependencies, then install `deepvol` in editable mode with `--no-deps`. Works with **Python 3.9 – 3.13**.

### Option A: Using `uv` (Recommended — Linux / macOS / WSL2)

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone <YOUR_REPO_URL> derivatives
cd derivatives

# 3. Create a virtual environment (Python 3.10 or any Python 3.9–3.12 on your laptop)
uv venv .venv --python 3.10
source .venv/bin/activate

# 4. Install CPU-only PyTorch (~180 MB download instead of 3.5 GB CUDA wheels)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Install runtime dependencies for the FX Cross engine & FastAPI server
uv pip install "numpy<2.0" scipy pydantic fastapi uvicorn httpx orjson pyarrow scikit-learn

# 6. Install deepvol without pulling GPU-only dependencies (tensorrt / onnxruntime-gpu)
uv pip install --no-deps -e .
```

### Option B: Using `uv` on Native Windows (PowerShell)

```powershell
# 1. Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone and enter the repository
git clone <YOUR_REPO_URL> derivatives
cd derivatives

# 3. Create and activate the virtual environment
uv venv .venv --python 3.10
.venv\Scripts\activate

# 4. Install CPU-only PyTorch and runtime dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install "numpy<2.0" scipy pydantic fastapi uvicorn httpx orjson pyarrow scikit-learn
uv pip install --no-deps -e .
```

### Option C: Using Standard `python3 -m venv` + `pip` (No `uv` Required)

```bash
git clone <YOUR_REPO_URL> derivatives
cd derivatives

python3 -m venv .venv
source .venv/bin/activate   # On Windows PowerShell: .venv\Scripts\activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2.0" scipy pydantic fastapi uvicorn httpx orjson pyarrow scikit-learn
pip install --no-deps -e .
```

### Verify Your CPU Installation

```bash
python -c "import torch, deepvol.models.fx_cross_surrogate; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Status: OK')"
```
Expected output:
```text
PyTorch: 2.8.0+cpu | CUDA: False | Status: OK
```

---

## 3. Running the FastAPI Service & Swagger UI

Start the local pricing server:

```bash
PYTHONPATH=src uvicorn deepvol.api.server:app --host 127.0.0.1 --port 8000
```

Once started (under 1 second on CPU):
- **Interactive Swagger UI**: Open `http://127.0.0.1:8000/docs` in your browser to inspect schemas and test requests interactively.
- **Health Check (`GET /fx_cross/health`)**:

```bash
curl -s http://127.0.0.1:8000/fx_cross/health | python -m json.tool
```
```json
{
  "status": "degraded",
  "device": "cpu",
  "model_version": "1326b33ce012+afc6d3f",
  "graphs_captured": true,
  "quote_bucket_gpu": 2,
  "batch_buckets": [64, 1024, 10240],
  "queue_depth": 0,
  "executor_alive": true
}
```
*(Note: `"status": "degraded"` only reflects that `"device": "cpu"` is active instead of a CUDA GPU; all analytical, neural-jet, and OOD guardian features run normally on CPU.)*

### Example `curl` Request (`XAU/RUB` ATM Call, `K = 382,100.0`)

```bash
curl -s -X POST http://127.0.0.1:8000/fx_cross/price \
  -H "Content-Type: application/json" \
  -d '{
    "contract": {
      "identifier": "XAURUB",
      "parity": "Call",
      "strike": 382100.0,
      "as_of_date": "2026-09-09T00:00:00Z",
      "expiration_date": "2028-09-08T00:00:00Z",
      "settlement_date": "2028-09-11T00:00:00Z",
      "payment_currency": "RUB"
    },
    "market_data": {
      "leg1": {
        "pair": "XAUUSD",
        "spot": 4414.66,
        "foreign_discount_factor": 0.9309147404,
        "domestic_discount_factor": 0.8521034483,
        "smile": {
          "atm_vol": 0.2250,
          "rr25": 0.0098,
          "bf25": 0.0092,
          "rr10": 0.0216,
          "bf10": 0.0336,
          "delta_convention": "spot_pna",
          "atm_convention": "DNS",
          "strangle_convention": "SMILE"
        }
      },
      "leg2": {
        "pair": "USDRUB",
        "spot": 86.551435,
        "foreign_discount_factor": 0.8521034483,
        "domestic_discount_factor": 0.7635754481,
        "smile": {
          "atm_vol": 0.1850,
          "rr25": 0.0120,
          "bf25": 0.0080,
          "rr10": 0.0250,
          "bf10": 0.0210,
          "delta_convention": "spot_pa",
          "atm_convention": "DNS",
          "strangle_convention": "SMILE"
        }
      },
      "correlation": 0.0,
      "settlement_discount_factor": 0.7635754481
    }
  }' | python -m json.tool
```

---

## 4. Complete JSON Schema Reference (`POST /fx_cross/price`)

Defined in [`src/deepvol/api/schemas_fx_cross.py`](../src/deepvol/api/schemas_fx_cross.py) (`TriangulatedFXOptionRequest` and `TriangulatedFXOptionResponse`).

### Request Schema (`TriangulatedFXOptionRequest`)

```jsonc
{
  "contract": {
    "identifier": "XAURUB",                       // [required] string, regex "^[A-Z]{6}$" (CCY_A + CCY_C)
    "parity": "Call",                             // [required] "Call" | "Put"
    "strike": 382100.0,                           // [required] float in [1e-6, 1e9] (cross strike K in payment_currency)
    "as_of_date": "2026-09-09T00:00:00Z",         // [required] timezone-aware ISO-8601 datetime
    "expiration_date": "2028-09-08T00:00:00Z",    // [required] timezone-aware ISO-8601 datetime (> as_of_date)
    "settlement_date": "2028-09-11T00:00:00Z",    // [required] timezone-aware ISO-8601 datetime (>= expiration_date)
    "payment_currency": "RUB",                    // [required] string, regex "^[A-Z]{3}$" (must equal leg2 quote currency C)
    "settlement_type": "CASH",                    // [optional] "CASH" | "PHYSICAL" (default: "CASH")
    "calendar": "NYB,MWB,LNB"                     // [optional] string (default: "NYB,MWB,LNB")
  },
  "market_data": {
    "leg1": {                                     // [required] Leg 1: Asset A in intermediate ccy B (e.g., "XAUUSD")
      "pair": "XAUUSD",                           // [required] string, regex "^[A-Z]{6}$"
      "spot": 4414.66,                            // [required] float > 0.0 (S_1)
      "foreign_discount_factor": 0.9309147404,    // [required] float in [1e-4, 2.5] (D_A(T_exp))
      "domestic_discount_factor": 0.8521034483,   // [required] float in [1e-4, 2.5] (D_B(T_exp); must match leg2.foreign_discount_factor within 1e-6 rel)
      "smile": {                                  // [required] 5-point market smile at option expiry
        "atm_vol": 0.2250,                        // [required] float in [0.01, 3.0] (decimal, e.g., 0.2250 = 22.50%)
        "rr25": 0.0098,                           // [required] float in [-0.50, 0.50] (25D Risk Reversal: sigma_25C - sigma_25P)
        "bf25": 0.0092,                           // [required] float in [-0.05, 0.30] (25D Butterfly: 0.5*(sigma_25C + sigma_25P) - sigma_ATM)
        "rr10": 0.0216,                           // [required] float in [-0.80, 0.80] (10D Risk Reversal: sigma_10C - sigma_10P)
        "bf10": 0.0336,                           // [required] float in [-0.05, 0.50] (10D Butterfly: 0.5*(sigma_10C + sigma_10P) - sigma_ATM)
        "delta_convention": "spot_pna",           // [optional] "spot_pna" | "spot_pa" | "forward_pna" | "forward_pa" (default: "spot_pna")
        "atm_convention": "DNS",                  // [optional] "DNS" | "ATMF" (default: "DNS")
        "strangle_convention": "SMILE"            // [optional] "SMILE" | "MARKET" (default: "SMILE")
      }
    },
    "leg2": {                                     // [required] Leg 2: Intermediate ccy B in quote ccy C (e.g., "USDRUB")
      "pair": "USDRUB",                           // [required] string, regex "^[A-Z]{6}$" (base ccy must equal leg1 quote ccy)
      "spot": 86.551435,                          // [required] float > 0.0 (S_2)
      "foreign_discount_factor": 0.8521034483,    // [required] float in [1e-4, 2.5] (D_B(T_exp); must match leg1.domestic_discount_factor)
      "domestic_discount_factor": 0.7635754481,   // [required] float in [1e-4, 2.5] (D_C(T_exp))
      "smile": {
        "atm_vol": 0.1850,                        // [required] float in [0.01, 3.0]
        "rr25": 0.0120,                           // [required] float in [-0.50, 0.50]
        "bf25": 0.0080,                           // [required] float in [-0.05, 0.30]
        "rr10": 0.0250,                           // [required] float in [-0.80, 0.80]
        "bf10": 0.0210,                           // [required] float in [-0.05, 0.50]
        "delta_convention": "spot_pa",            // [optional] "spot_pna" | "spot_pa" | "forward_pna" | "forward_pa" (default: "spot_pna")
        "atm_convention": "DNS",                  // [optional] "DNS" | "ATMF" (default: "DNS")
        "strangle_convention": "SMILE"            // [optional] "SMILE" | "MARKET" (default: "SMILE")
      }
    },
    "correlation": 0.0,                           // [required] float in [-0.999, 0.999] (instantaneous correlation rho between S_1 and S_2)
    "correlation_type": "HISTORICAL_LOGRETURN",   // [optional] "ATM_TRIANGULATION" | "HISTORICAL_LOGRETURN" (default: "HISTORICAL_LOGRETURN")
    "settlement_discount_factor": 0.7635754481    // [required] float in [1e-4, 2.5] (D_C(T_set))
  },
  "interest_rates": {                             // [optional] object (only needed when requesting bucketed curve Rho)
    "domestic_curve": [                           // [optional] list of CurvePillar (default: [], max 60, strictly increasing tenor_years)
      {
        "pillar_date": "2028-09-08T00:00:00Z",    // [required] timezone-aware ISO-8601 datetime
        "tenor_years": 2.0,                       // [required] float in [0.0, 60.0]
        "zero_rate": 0.1349                       // [required] float in [-0.20, 2.0] (continuously compounded decimal rate)
      }
    ],
    "foreign_curve": []                           // [optional] list of CurvePillar for Asset A curve (default: [])
  },
  "execution_options": {                          // [optional] object (default: all defaults below)
    "engine": "Differential_ML",                  // [optional] "Differential_ML" | "Analytic_Triangulated" | "Analytic_GK" | "Hybrid_Ensemble" | "Copula_Benchmark"
    "precision": "float64",                       // [optional] "float64" | "float32" (default: "float64")
    "use_guardian": true,                         // [optional] bool (default: true; enables Mahalanobis OOD & Durrleman safeguards)
    "smile_dynamics": "STICKY_MONEYNESS",         // [optional] "STICKY_MONEYNESS" | "STICKY_STRIKE" (default: "STICKY_MONEYNESS")
    "requested_queries": [                        // [optional] list of QueryEnum (default: all Greeks except "ModelReserve")
      "ProductValue", "Delta", "CashDelta", "Gamma", "Vega",
      "Theta", "Vanna", "Vomma", "Rho", "CorrelationSensitivity", "SmileVega"
    ]
  }
}
```

### Cross-Field Validation Rules
1. **Currency Triangulation**: `leg1.pair == CCY_A + CCY_B`, `leg2.pair == CCY_B + CCY_C`, `contract.identifier == CCY_A + CCY_C`, and `contract.payment_currency == CCY_C`.
2. **Hub Discount Factor Coherence**: `abs(leg1.domestic_discount_factor - leg2.foreign_discount_factor) <= 1e-6 * max(d1, d2)`.
3. **Wing Positivity**: For both 25D and 10D pillars, `atm_vol + bf - 0.5 * abs(rr) >= 0.01`.
4. **Date Ordering**: `as_of_date < expiration_date <= settlement_date`.

### Response Schema (`TriangulatedFXOptionResponse`)

```jsonc
{
  "status": "done",
  "product_value": 111379.55,                     // Option PV in payment_currency
  "implied_vol_cross": 0.289919,                  // Served effective Black-76 cross implied volatility sigma_x(K, T)
  "forward_cross": 465831.78,                     // Triangulated cross forward F_x = S_1 * S_2 * (D_A / D_C)
  "delta": {
    "cross": 0.659472,                            // dV / dS_cross
    "leg1": 57.0782,                              // dV / dS_1 = S_2 * dV / dS_cross
    "leg2": 2911.34                               // dV / dS_2 = S_1 * dV / dS_cross
  },
  "cash_delta": {
    "cross": 251984.42,                           // S_cross * dV / dS_cross in payment_currency (identical across all 3 keys)
    "leg1": 251984.42,
    "leg2": 251984.42
  },
  "gamma": 0.00000181,                            // Smile-adjusted d^2V / dS_cross^2 (includes carry factor g^2)
  "vega": {
    "cross": 110229.80,                           // dV / d(sigma_cross) per 1.00 (100 vol-pts)
    "leg1": 85612.40,                             // Parallel shift sensitivity to Leg 1 smile
    "leg2": 70318.15,                             // Parallel shift sensitivity to Leg 2 smile
    "smile_matrix_leg1": [85612.4, 1210.5, ...],  // 5-pillar vegas [ATM, RR25, BF25, RR10, BF10] when SmileVega requested
    "smile_matrix_leg2": [70318.1, 980.2, ...]
  },
  "theta": -24.18,                                // Calendar-day model decay: (1/365) * dV/dt in payment_currency/day
  "vanna": 0.0884,                                // d^2V / (dS_cross d(sigma_cross))
  "vomma": 159994.75,                             // d^2V / d(sigma_cross)^2
  "correlation_sensitivity": 8068.93,             // dV / d(rho)
  "rho_domestic": [0.0, 412510.2, ...],           // Bucketed sensitivity to domestic_curve pillars (or null if curve empty)
  "rho_foreign": [0.0, -503968.8, ...],           // Bucketed sensitivity to foreign_curve pillars (or null if curve empty)
  "model_reserve_bps": null,                      // |Copula benchmark - served price| in bps of D_set * F (when ModelReserve requested)
  "warnings": [],                                 // List of WarningCodeEnum ("SMILE_SSVI_FALLBACK", "NN_TAPERED", etc.)
  "latency_ms": 0.31,                             // End-to-end execution latency in milliseconds
  "telemetry": {
    "engine_used": "Differential_ML",
    "standardized_moneyness": -0.485,             // z = ln(K / F) / (sigma_ref * sqrt(T))
    "mahalanobis_distance": 3.047,                // 12-D Ledoit-Wolf Mahalanobis distance
    "ood_threshold": 5.337,                       // Training split 99.5% quantile threshold
    "is_out_of_distribution": false,
    "ensemble_epistemic_std_bps": 0.12,
    "fallback_activated": false,
    "fallback_reason": null,
    "durrleman_min_g": 0.842,
    "smile_dynamics": "STICKY_MONEYNESS",
    "model_version": "residual_vol_v3"
  }
}
```

---

## 5. Pricing Legacy Bank `ConstMessage` Payloads (`legacy_adapter.py`)

If your bank's pricing service emits legacy graph-based `ConstMessage` JSON arrays (`EuropeanVanillaOptionProduct`, `BlackScholesMertonModel`, `MarketDataSet`, `DiscountFactorCurve`, `FXForwardObject`, `FXVolatilitySmileCollection`), use [`src/deepvol/market/legacy_adapter.py`](../src/deepvol/market/legacy_adapter.py) to:
1. Verify DAG integrity and SHA-1 content hashes (`ConstMessageGraph`).
2. Interpolate multi-curve `Act/365F` discount factors and 5-pillar market smiles at expiration (`build_request`).
3. Price the option with the 5-member `ResidualCrossVolNet` ensemble (`PricingEngineEnum.DIFFERENTIAL_ML`).
4. Serialize the output back to the legacy `{"MutableMessage": {...}}` schema (`to_mutable_message`).

### Complete Runnable Example

Run this script directly from the repository root using the included fixture [`tests/fixtures/legacy/xaurub_k10_complete_const_messages.json`](../tests/fixtures/legacy/xaurub_k10_complete_const_messages.json):

```python
import asyncio
import copy
import json
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from deepvol.api.schemas_fx_cross import (
    PricingEngineEnum,
    TriangulatedFXOptionResponse,
)
from deepvol.api.server import _get_fx_cross_state, app, reload_fx_cross_weights
from deepvol.market.legacy_adapter import (
    ConstMessageGraph,
    build_request,
    to_mutable_message,
)
from deepvol.models.fx_cross_surrogate import ResidualCrossVolNet

# 1. Load the legacy ConstMessage array and locate the root CalculatorRequest ID
fixture_path = Path("tests/fixtures/legacy/xaurub_k10_complete_const_messages.json")
raw_messages = json.loads(fixture_path.read_text())
root_id = raw_messages[0]["ConstMessage"]["id"]  # "1b4eef0d8162d4d520fe6bbebfed6798a4600b99"

with TestClient(app) as client:
    # Load the 5-member pre-trained ResidualCrossVolNet ensemble on CPU
    state = _get_fx_cross_state()
    ckpts = sorted(Path("artifacts/models/fx_cross").glob("residual_vol_v3_m*.pt"))
    net_5 = ResidualCrossVolNet(width=64, depth=3, members=len(ckpts))
    ref_ckpt = torch.load(ckpts[0], map_location="cpu", weights_only=True)
    for m, cp in enumerate(ckpts):
        m_ckpt = torch.load(cp, map_location="cpu", weights_only=True)
        m_sd = m_ckpt["state_dict"]
        for l_idx in range(len(net_5.weights)):
            net_5.weights[l_idx].data[m].copy_(m_sd[f"weights.{l_idx}"][0])
            net_5.biases[l_idx].data[m].copy_(m_sd[f"biases.{l_idx}"][0])
    net_5.feature_mean.copy_(ref_ckpt["feature_mean"])
    net_5.feature_std.copy_(ref_ckpt["feature_std"])
    state["engine"].net = net_5
    state["runtime"].cpu_quote_path.engine.net = net_5
    state["model_version"] = "residual_vol_v3_ensemble5"

    # -------------------------------------------------------------------------
    # Case A: Original Deep-ITM Fixture (K = 10.0 RUB, z = -26.4)
    # -------------------------------------------------------------------------
    graph_k10 = ConstMessageGraph(raw_messages, verify_hash=True)
    req_k10, _, ctx_k10 = build_request(graph_k10, root_id)
    req_k10.execution_options.engine = PricingEngineEnum.DIFFERENTIAL_ML

    resp_k10 = TriangulatedFXOptionResponse.model_validate(
        client.post("/fx_cross/price", json=req_k10.model_dump(mode="json")).json()
    )
    mm_k10 = to_mutable_message(
        resp_k10, req_k10, ctx_k10,
        nonce="k10-demo", total_sec=resp_k10.latency_ms / 1000.0, proc_sec=resp_k10.latency_ms / 1000.0,
    )
    res_k10 = mm_k10["MutableMessage"]["value"]["Result"]

    print("=== Case A: Legacy Fixture (K = 10.0 RUB, Deep ITM z = -26.4) ===")
    print(f"ProductValue:     {res_k10['ProductValue']:,.2f} RUB")
    print(f"Delta (XAURUB):   {res_k10['Delta']['XAURUB']:.12f}")
    print(f"CashDelta:        {res_k10['CashDelta']['XAURUB']:,.2f} RUB")
    print(f"NonAdaptedDelta:  {res_k10['NonAdaptedDelta']['XAURUB']:.12f}")
    print(f"Gamma / Vega:     {res_k10['Gamma']['XAURUB']:.6e} / {resp_k10.vega.cross:.6e}")
    print(f"Latency (CPU):    {resp_k10.latency_ms:.3f} ms\n")

    # -------------------------------------------------------------------------
    # Case B: ATM Strike (K = 382,100.0 RUB = 4414.66 XAUUSD * 86.551433 USDRUB)
    # -------------------------------------------------------------------------
    atm_messages = copy.deepcopy(raw_messages)
    for item in atm_messages:
        msg = item.get("ConstMessage", item)
        if msg["value"].get("TypeName") == "EuropeanVanillaOptionProduct":
            msg["value"]["Strike"] = 382100.0

    # Set verify_hash=False because we modified Strike in-memory without re-hashing the DAG IDs
    graph_atm = ConstMessageGraph(atm_messages, verify_hash=False)
    req_atm, _, ctx_atm = build_request(graph_atm, root_id)
    req_atm.execution_options.engine = PricingEngineEnum.DIFFERENTIAL_ML

    resp_atm = TriangulatedFXOptionResponse.model_validate(
        client.post("/fx_cross/price", json=req_atm.model_dump(mode="json")).json()
    )
    mm_atm = to_mutable_message(
        resp_atm, req_atm, ctx_atm,
        nonce="atm-demo", total_sec=resp_atm.latency_ms / 1000.0, proc_sec=resp_atm.latency_ms / 1000.0,
    )
    res_atm = mm_atm["MutableMessage"]["value"]["Result"]

    print("=== Case B: ATM Strike (K = 382,100.0 RUB, Active 5-Member ML Ensemble) ===")
    print(f"ProductValue:            {res_atm['ProductValue']:,.2f} RUB")
    print(f"Implied Cross Vol:       {resp_atm.implied_vol_cross:.6f}")
    print(f"Delta (XAURUB):          {res_atm['Delta']['XAURUB']:.6f}")
    print(f"CashDelta (XAURUB):      {res_atm['CashDelta']['XAURUB']:,.2f} RUB")
    print(f"NonAdaptedDelta:         {res_atm['NonAdaptedDelta']['XAURUB']:.6f}")
    print(f"Gamma (XAURUB):          {res_atm['Gamma']['XAURUB']:.8f}")
    print(f"CashGamma (1% move):     {res_atm['CashGamma']['XAURUB']:,.2f} RUB")
    print(f"Cross Vega:              {resp_atm.vega.cross:,.2f} RUB")
    print(f"Leg1 / Leg2 Vega:        {resp_atm.vega.leg1:,.2f} / {resp_atm.vega.leg2:,.2f} RUB")
    print(f"Theta (1D model decay):  {res_atm['Theta']:,.2f} RUB/day")
    print(f"Vanna / Vomma:           {resp_atm.vanna:.6f} / {resp_atm.vomma:,.2f}")
    print(f"Mahalanobis Distance:    {resp_atm.telemetry.mahalanobis_distance:.3f} (OOD: {resp_atm.telemetry.is_out_of_distribution})")
```

---

## 6. Pricing a Direct Single-Leg FX Option (`USD/RUB` or `XAU/USD`)

Direct FX pairs such as `USD/RUB` or `XAU/USD` have liquid 5-pillar market volatility quotes (`ATM`, `25D RR`, `25D BF`, `10D RR`, `10D BF`). Because the market directly quotes the `USD/RUB` volatility smile, no cross-currency copula triangulation is needed (`ResidualCrossVolNet` is specifically a 2-leg copula residual network for synthetic crosses $S^\times = S_1 S_2$).

You can price a direct `USD/RUB` option in two ways:

### Method A: Using `/fx_cross/price` with a Unit Identity First Leg ($S_1 \equiv 1$)
Pass a degenerate unit leg (`pair: "USDUSD"`, `spot: 1.0`, `foreign_discount_factor == domestic_discount_factor`, `atm_vol: 0.01`, `rr/bf: 0.0`, `correlation: 0.0`, `engine: "Analytic_Triangulated"`). By $\beta$-split property P3, as $\sigma_1 \to 0$, $\beta_1 \to 0$ and $\beta_2 \to 1$, recovering the exact single-leg `USD/RUB` SVI smile and analytical Greeks.

### Method B: Direct Python Call via `SmileCanonicalizer` + `svi_vol_jet` + `black76_otm_parity`

```python
import math
import torch
from deepvol.market.fx_conventions import (
    AtmConvention,
    DeltaConvention,
    LegMarket,
    StrangleConvention,
)
from deepvol.market.smile_canonicalizer import SmileCanonicalizer
from deepvol.models import taylor_jets as tj
from deepvol.models.fx_cross_analytic import (
    Col,
    EngineOutputs,
    N_COLS,
    black76_otm_parity,
    black_partials,
    greeks_from_outputs,
    svi_vol_jet,
)

# 1. Market inputs for a 2Y USD/RUB Call Option (Strike K = 90.0 RUB)
S = 86.551435
K = 90.0
T = 730.0 / 365.0
D_usd = 0.8521034483   # Foreign (USD) discount factor D_f(T)
D_rub = 0.7635754481   # Domestic (RUB) discount factor D_d(T)
D_set = 0.7635754481   # Settlement discount factor D_set(T_set)
g = D_usd / D_rub
F = S * g              # Forward F = 96.5866 RUB

# 2. Canonicalize the 5-pillar USD/RUB smile [ATM, RR25, BF25, RR10, BF10] into SVI parameters
canon = SmileCanonicalizer()
quotes = torch.tensor([[0.1850, 0.0120, 0.0080, 0.0250, 0.0210]], dtype=torch.float64)
convs = [(DeltaConvention.SPOT_PA, AtmConvention.DNS, StrangleConvention.SMILE)]
leg_mkt = LegMarket(
    spot=torch.tensor([S], dtype=torch.float64),
    df_foreign=torch.tensor([D_usd], dtype=torch.float64),
    df_domestic=torch.tensor([D_rub], dtype=torch.float64),
    T=torch.tensor([T], dtype=torch.float64),
)
cs = canon.canonicalize(quotes, convs, leg_mkt, compute_jacobian=False)

# 3. Evaluate the SVI smile jet at log-moneyness k = ln(K / F)
k_val = torch.tensor([math.log(K / F)], dtype=torch.float64)
k_jet = tj.seed_moneyness(k_val, P=3)
T_t = torch.tensor([T], dtype=torch.float64)
delta_zero = tj.from_primal(torch.zeros(1, dtype=torch.float64), P=3)
sigma_jet = svi_vol_jet(k_jet, T_t, cs.svi_params, delta_zero)

# 4. Evaluate Black-76 price and sticky-moneyness Greeks in float64
F_t = torch.tensor([F], dtype=torch.float64)
K_t = torch.tensor([K], dtype=torch.float64)
D_t = torch.tensor([D_set], dtype=torch.float64)
phi_t = torch.tensor([1.0], dtype=torch.float64)  # +1.0 for Call, -1.0 for Put

V = black76_otm_parity(F_t, K_t, sigma_jet.v, T_t, D_t, phi_t)
B_F, B_FF, B_Fs, B_s, B_ss = black_partials(F_t, K_t, sigma_jet.v, T_t, D_t, phi_t)

X = torch.zeros((1, N_COLS), dtype=torch.float64)
X[0, Col.F] = F
X[0, Col.K] = K
X[0, Col.T] = T
X[0, Col.T_SET] = T
X[0, Col.D] = D_set
X[0, Col.G] = g
X[0, Col.PHI] = 1.0
X[0, Col.R_A] = -math.log(D_usd) / T
X[0, Col.R_C] = -math.log(D_rub) / T
X[0, Col.R_SET] = -math.log(D_set) / T

out = EngineOutputs(
    V=V, sigma=sigma_jet.v, s_k=sigma_jet.d, s_kk=sigma_jet.dd,
    s_t=torch.zeros((1, 3), dtype=torch.float64),
    B_F=B_F, B_FF=B_FF, B_Fs=B_Fs, B_s=B_s, B_ss=B_ss, chi=torch.ones_like(V),
)
greeks = greeks_from_outputs(X, out, dynamics="STICKY_MONEYNESS")

print(f"USD/RUB Implied Vol at K={K}: {sigma_jet.v.item():.6f}")
print(f"ProductValue:                 {greeks['price'].item():.4f} RUB")
print(f"Delta (dV/dS):                {greeks['delta'].item():.6f}")
print(f"CashDelta (S * dV/dS):        {greeks['cash_delta'].item():.4f} RUB")
print(f"NonAdaptedDelta (Delta / g):  {(greeks['delta'].item() / g):.6f}")
print(f"Gamma:                        {greeks['gamma'].item():.8f}")
print(f"Vega (per 100 vol-pts):       {greeks['vega'].item():.4f} RUB")
print(f"Theta (per calendar day):     {greeks['theta'].item():.4f} RUB/day")
```
