# Orchestrator Briefing — Tier 1 Real-Market Extension
## For: New Claude agent via /goal + /teamwork-preview
## Project: "Neural Networks in Derivatives Pricing" — МФТИ ФПМИ, кафедра БИТ

---

## ⚠️ MANDATORY OPERATING RULES (read before ANY action)

1. **Git worktrees for parallel work.** If running multiple agents simultaneously on
   different features, use `git worktree add ../derivatives-worker-N branch-name` for
   each worker. NEVER let two agents write to the same branch simultaneously.
   Merge back to master only after each worker's tests pass.

2. **Performance is the absolute priority.** Use GPU acceleration everywhere possible
   (torch CUDA, batched operations). Profile before optimizing. Target: every operation
   that could run on GPU MUST run on GPU.

3. **Hardware constraint: RTX 3060 (6GB VRAM), 16GB RAM.**
   - No task should take >30 minutes wall-clock without user approval
   - No single batch should exceed ~4GB VRAM
   - Prefer `float32` over `float64` (FNO was trained in float32)
   - If a task needs >30 min or >4GB VRAM → STOP, ask user first

4. **If you don't know how to implement something → WRITE A GEMINI DEEP RESEARCH PROMPT.**
   Do NOT guess at math or APIs. Write a specific research prompt, tell the user
   "I need to research X before implementing", and wait for the result.
   This is mandatory for: fractional Riccati ODEs, VIX Laplace transforms,
   SVI no-arbitrage conditions, Deribit websocket protocol.

5. **Never launch heavy tasks (>30 min) without user permission first.**
   Always estimate time before starting: "This will take ~X minutes, proceed?"

6. **Write all large files in small parts** (< 150 lines per write operation).
   Never write a 400-line file in one shot — split into logical sections and
   append sequentially. This prevents prefill/context errors.

---

## 🏗️ Repository

```
Path:    /home/execorn/programming/derivatives
Python:  .venv/bin/python  (3.14)
Tests:   .venv/bin/python -m pytest tests/ -q   ← must stay 32/32 green
App:     .venv/bin/streamlit run src/app_fno.py
GPU:     RTX 3060, 6GB VRAM — float32 only, batches ≤ 4GB
```

**READ THESE FILES FIRST — in order:**
```
CONTEXT.md                                          ← full technical map (337 lines)
ROADMAP_ABSOLUTE_MAX.md                             ← full future roadmap
research/spx_options.md                             ← §1.1 SPX research
research/arbitrage_free_surface.md                  ← §1.2 research
research/rough_heston_calibration.md                ← §1.3 VIX research
research/option_greeks_via_fno_autograd.md          ← §1.4 Greeks research
research/calibrating_rough_heston_to_crypto_options.md  ← §1.5 Crypto research
src/calibrate.py                                    ← FIM key names (CRITICAL)
src/calibrate_fast.py                               ← Newton calibrator
src/fno_model.py                                    ← MirrorPaddedFNO2d
src/app_fno.py                                      ← Streamlit demo
```


---

## 🧠 Existing Infrastructure (DO NOT MODIFY these files)

```
src/fno_model.py       ← MirrorPaddedFNO2d — input MUST be (B,T,K,2) channels-last
src/normalizers.py     ← ParameterNormalizer, IVSurfaceNormalizer
src/pricing_engine.py  ← GPU Fourier-COS (Bernstein lifting)
src/calibrate.py       ← L-BFGS + FIM — fim_res["std_errors"] is np.array [0,1,2]
src/calibrate_fast.py  ← Newton–Gauss — p50=541ms, p95=668ms
src/app_fno.py         ← Streamlit demo — do not modify unless adding new tabs
```

### Load model (copy-paste ready)
```python
import sys; sys.path.insert(0, 'src')
import torch, numpy as np
from fno_model import MirrorPaddedFNO2d
from normalizers import ParameterNormalizer, IVSurfaceNormalizer

model = MirrorPaddedFNO2d()
model.load_state_dict(torch.load(
    'artifacts/weights/fno_v2_final_prod.pth', map_location='cpu'))
model.eval()
pn = ParameterNormalizer.load('artifacts/models/param_normalizer_v2.npz')
yn = IVSurfaceNormalizer.load('artifacts/models/iv_normalizer_v2.npz')
```

### Run Newton calibration on any IV surface
```python
from calibrate_fast import calibrate_newton

# iv_surface: np.ndarray shape (8,11), units = annualised decimal IV (0.20 = 20%)
result = calibrate_newton(model, iv_surface, MATURITIES, STRIKES, pn=pn, yn=yn)
# result keys: v0, zeta, lambda, history, elapsed
# recover: sigma = sqrt(zeta^2 + lambda^2), rho = zeta / sigma
```

### Grids (fixed — FNO was trained on these exactly)
```python
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
STRIKES    = np.linspace(-0.5, 0.5, 11)   # log-moneyness k = log(K_dollar / F)
```

### FNO training parameter ranges (extrapolation = inaccurate)
```python
BOUNDS = {"kappa":(0.5,5), "theta":(0.01,0.25), "sigma":(0.1,1.5),
          "rho":(-0.95,0), "v0":(0.01,0.25), "H":(0.04,0.15)}
# Crypto (BTC): v0 can reach 0.6, sigma up to 3.0 → warn user, may need retraining
```


---

## 📋 Five Tasks — Stubs Already Exist, Research Already Done

### §1.1 — Real SPX Market Data Calibration
**Stub:** `src/market/spx_data.py`  |  **Research:** `research/spx_options.md`

Key implementation facts from research:
- `yfinance.Ticker("^SPX").option_chain(expiry)` → DataFrame with bid/ask/IV/OI
- yfinance IV is STALE — recompute from bid-ask mid via `py_vollib_vectorized`
- Risk-free rate: use 5.0% for 2024 dates (or fetch from FRED via `pandas_datareader`)
- Forward: `F = S * exp((r-q)*T)` — SPX uses dividend-point adjusted forward
- Log-moneyness: `k = log(K_dollar / F)` — only keep k ∈ [-0.5, 0.5]
- Interpolate to (8,11) FNO grid with `scipy.interpolate.RectBivariateSpline`
- Liquidity filter: bid > 0, OI ≥ 10, spread/mid < 20%
- Arbitrage filter: calendar spread (total variance monotone), butterfly (C convex in K)

**Key dates to calibrate:** 2020-03-16, 2022-01-24, 2024-01-02, 2024-08-05
**Output:** `results/spx_calibration/{date}.json`
**Notebook:** `notebooks/01_spx_calibration.ipynb`

---

### §1.2 — Arbitrage-Free IV Surface Completion
**Stub:** `src/arbitrage/surface_completion.py`  |  **Research:** `research/arbitrage_free_surface.md`

Key implementation facts:
- SVI raw: `w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))` where w=IV²*T
- Fit slice-by-slice with `scipy.optimize.minimize(method='SLSQP')` + bounds
- Bounds: `a>0, b>0, |rho|<1, sigma>0`; butterfly check: `d²w/dk² ≥ 0`
- Monotone rearrangement: sort total variance slices to enforce calendar spread
- Test: mask 40% of SPX surface → complete → RMSE on held-out quotes
- Compare: SVI vs cubic_spline vs FNO-completion

**Output:** `results/spx_calibration/completion_test.json`
**Notebook:** `notebooks/02_surface_completion.ipynb`

---

### §1.3 — VIX Futures Pricing
**Stub:** `src/market/vix_pricing.py`  |  **Research:** `research/rough_heston_calibration.md`

Key implementation facts from research:
- VIX² = (1/Δ) * E[∫₀^Δ v_t dt], Δ=30/365
- For stationary case (V₀≈θ): VIX ≈ sqrt(θ) * 100 (rough approximation only)
- EXACT: use modified Riccati ODE — research file has JAX pseudocode
  BUT: we use PyTorch not JAX; adapt using `scipy.integrate.solve_ivp`
- VIX futures CSV: `https://www.cboe.com/tradable_products/vix/vix_historical_data/`
  Fields: `Trade Date, Futures, Open, High, Low, Close, Settle, Change, %Chg, Volume, OI, EFP`
  USE Settle column, not Close
- Align VIX futures with SPX option dates by `pd.merge` on Trade_Date
- DO NOT use JAX — stick to scipy + torch for consistency with existing codebase

**Notebook:** `notebooks/03_vix_analysis.ipynb`
**⚠️ Research gap:** exact Rough Heston Laplace transform for E[∫v dt] is complex.
If unsure → write a Gemini Deep Research prompt for the exact numerical formula.

---

### §1.4 — Portfolio Greeks at Scale
**Stub:** `src/greeks/portfolio_greeks.py`  |  **Research:** `research/option_greeks_via_fno_autograd.md`

Key implementation facts (research provides exact PyTorch code):
```python
# Differentiable Black-Scholes from research file:
from torch.distributions import Normal
normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
d1 = (torch.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*torch.sqrt(T))
d2 = d1 - sigma*torch.sqrt(T)
call = S*torch.exp(-q*T)*normal.cdf(d1) - K*torch.exp(-r*T)*normal.cdf(d2)

# Greeks via autograd:
delta, vega = torch.autograd.grad(call, (S, sigma_iv), create_graph=True)
gamma = torch.autograd.grad(delta, S, retain_graph=True)[0]
vanna = torch.autograd.grad(vega, S, retain_graph=True)[0]
volga = torch.autograd.grad(vega, sigma_iv, retain_graph=True)[0]
```

FNO Jacobian (parameter sensitivities):
```python
# jacfwd preferred: 6 inputs → 88 outputs (more efficient than jacrev)
J = torch.func.jacfwd(fno_forward_wrapper)(theta)  # shape (88, 6)
```

**⚠️ CRITICAL GOTCHA from research:** Spectral convolution kinks → gradient explosion.
If Gamma/Speed diverge: reduce K_max or add L2 regularization on spectral weights.
All normalizers MUST be inside the torch graph (no numpy detachment).

**Notebook:** `notebooks/04_greeks_portfolio.ipynb`

---

### §1.5 — Crypto Derivatives (Deribit)
**Stub:** `src/market/deribit_data.py`  |  **Research:** `research/calibrating_rough_heston_to_crypto_options.md`

Key implementation facts (API CONFIRMED WORKING):
```python
import requests
r = requests.get(
    "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
    params={"currency": "BTC", "kind": "option"}, timeout=10)
data = r.json()["result"]   # 934 instruments currently
# Each: instrument_name, mark_iv (in %), underlying_price, open_interest, bid_iv, ask_iv
```

Parse name: `"BTC-28JUN24-70000-C"` → split on `-`:
```python
from datetime import datetime
parts = name.split("-")
coin   = parts[0]
expiry = datetime.strptime(parts[1], "%d%b%y").date()
strike = int(parts[2])
opt_t  = parts[3]   # "C" or "P"
```

Convert IV: `iv_decimal = mark_iv_pct / 100.0`
Forward from put-call parity: `F = K + (C - P) * exp(r * T)`

**⚠️ BTC PARAM RANGE WARNING:** V₀ for BTC can reach 0.40-0.60, σ up to 2.5.
FNO was trained with V₀≤0.25, σ≤1.5 → extrapolation will be inaccurate.
Options: (a) clip to training range + warn, (b) retrain FNO on extended range.
Retraining on RTX 3060: ~45-60 min for 50k samples → ASK USER before doing this.

**Notebook:** `notebooks/05_crypto_calibration.ipynb`


---

## 🤝 Teamwork Instructions (for /teamwork-preview)

Use `git worktree` to isolate each worker:
```bash
# Orchestrator sets up worktrees before dispatching workers:
git worktree add ../derivatives-w1 -b tier1/spx
git worktree add ../derivatives-w2 -b tier1/arbitrage
git worktree add ../derivatives-w3 -b tier1/vix
git worktree add ../derivatives-w4 -b tier1/greeks
git worktree add ../derivatives-w5 -b tier1/crypto

# After each worker completes and tests pass:
git merge tier1/spx --no-ff -m "feat: §1.1 SPX real market calibration"
git worktree remove ../derivatives-w1
```

**Worker assignments:**

| Worker | Branch | Files to implement | Research file |
|--------|--------|--------------------|---------------|
| W1 | tier1/spx | `src/market/spx_data.py` + `notebooks/01_spx_calibration.ipynb` | `research/spx_options.md` |
| W2 | tier1/arbitrage | `src/arbitrage/surface_completion.py` + `notebooks/02_surface_completion.ipynb` | `research/arbitrage_free_surface.md` |
| W3 | tier1/vix | `src/market/vix_pricing.py` + `notebooks/03_vix_analysis.ipynb` | `research/rough_heston_calibration.md` |
| W4 | tier1/greeks | `src/greeks/portfolio_greeks.py` + `notebooks/04_greeks_portfolio.ipynb` | `research/option_greeks_via_fno_autograd.md` |
| W5 | tier1/crypto | `src/market/deribit_data.py` + `notebooks/05_crypto_calibration.ipynb` | `research/calibrating_rough_heston_to_crypto_options.md` |

**Workers MUST NOT touch:**
`src/fno_model.py`, `src/calibrate.py`, `src/calibrate_fast.py`,
`src/normalizers.py`, `src/pricing_engine.py`, `src/app_fno.py`

**Each worker delivers:**
- Fully implemented module (zero `raise NotImplementedError`)
- Jupyter notebook with plots and analysis
- ≥2 new pytest tests in `tests/test_{module}.py`
- Results saved to `results/{task}/`
- 32/32 existing tests still passing (run before committing)

---

## ✅ Success Criteria

The Tier 1 implementation is DONE when ALL of these work:

```python
# §1.1
from src.market.spx_data import calibrate_to_market
result = calibrate_to_market(date(2024, 1, 2))
assert "params" in result and result["rmse_bps"] < 50

# §1.2
from src.arbitrage.surface_completion import complete_surface, check_butterfly
violations = check_butterfly(completed, K_grid, T_grid)
assert violations.sum() == 0  # no butterfly arbitrage

# §1.3
from src.market.vix_pricing import model_vix
vix = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08)
assert 10 < vix < 40  # reasonable VIX range

# §1.4
from src.greeks.portfolio_greeks import fno_surface_greeks
g = fno_surface_greeks(model, theta, pn, yn, S=5000.0)
assert all(k in g for k in ["delta", "gamma", "vega", "vanna", "volga"])

# §1.5
from src.market.deribit_data import fetch_option_snapshot
import asyncio
df = asyncio.run(fetch_option_snapshot("BTC"))
assert len(df) > 100 and "log_moneyness" in df.columns

# Full test suite
# pytest tests/ -q  →  32+ passed
```

---

## 🚫 Common Pitfalls to Avoid

1. **FNO channels-last:** input MUST be `(B, T, K, 2)` — NOT `(B, 2, T, K)`
2. **yfinance IV is stale** — always recompute from bid-ask mid
3. **Deribit mark_iv is in %** — divide by 100 before using
4. **FIM keys:** `std_errors` = numpy array (index 0,1,2), `ci_95` = dict of `(lo,hi)` tuples
5. **`_reparam_mode`** flag, not `calib_mode.startswith("Reparam")` — Newton is also reparameterized
6. **Spectral kinks → gradient explosion** in Greeks — validate Gamma against COS benchmark
7. **BTC v0 out of range** — FNO trained V₀≤0.25, BTC needs V₀ up to 0.6 → warn + ask before retraining
8. **VIX Laplace transform is complex** — if unsure, write a Gemini Deep Research prompt
9. **No JAX** — use scipy + torch only (no JAX in the codebase)
10. **Write large files in parts** — never write >150 lines in one write_to_file call

