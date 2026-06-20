# Original User Request

## Initial Request — 2026-06-19T01:29:01+03:00

Implement all 5 Tier 1 real-market extensions for a Rough Heston FNO calibration project. The project has an existing GPU-accelerated calibration pipeline using a FiLM-conditioned Fourier Neural Operator (FNO) that maps Rough Heston parameters to implied-volatility surfaces. Stubs, research files, git worktrees, and branches are all pre-created.

Working directory: /home/execorn/programming/derivatives
Integrity mode: development

## Critical Context

Read these files FIRST (in order) before any implementation:
- `CONTEXT.md` — full technical map (338 lines)
- `.agents/TIER1_ORCHESTRATOR_PROMPT.md` — full briefing with code snippets, API details, pitfalls

### Existing Infrastructure (DO NOT MODIFY these files)
- `src/fno_model.py` — MirrorPaddedFNO2d, input MUST be (B,T,K,2) channels-last
- `src/normalizers.py` — ParameterNormalizer, IVSurfaceNormalizer
- `src/pricing_engine.py` — GPU Fourier-COS pricer
- `src/calibrate.py` — L-BFGS + FIM calibration
- `src/calibrate_fast.py` — Newton–Gauss calibrator (p50=541ms)
- `src/app_fno.py` — Streamlit demo

### Load model (copy-paste ready)
```python
import sys; sys.path.insert(0, 'src')
import torch, numpy as np
from fno_model import MirrorPaddedFNO2d
from normalizers import ParameterNormalizer, IVSurfaceNormalizer

model = MirrorPaddedFNO2d()
model.load_state_dict(torch.load('artifacts/weights/fno_v2_final_prod.pth', map_location='cpu'))
model.eval()
pn = ParameterNormalizer.load('artifacts/models/param_normalizer_v2.npz')
yn = IVSurfaceNormalizer.load('artifacts/models/iv_normalizer_v2.npz')
```

### Grids (fixed — FNO trained on these exactly)
```python
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
STRIKES    = np.linspace(-0.5, 0.5, 11)   # log-moneyness k = log(K/F)
```

### FNO parameter ranges
```python
BOUNDS = {"kappa":(0.5,5), "theta":(0.01,0.25), "sigma":(0.1,1.5),
          "rho":(-0.95,0), "v0":(0.01,0.25), "H":(0.04,0.15)}
```

### Newton calibration usage
```python
from calibrate_fast import calibrate_newton
result = calibrate_newton(model, iv_surface, MATURITIES, STRIKES, pn=pn, yn=yn)
# result keys: v0, zeta, lambda, history, elapsed
# recover: sigma = sqrt(zeta^2 + lambda^2), rho = zeta / sigma
```

### Environment
- Python 3.14, venv at `.venv/bin/python`
- RTX 3060 (6GB VRAM), 16GB RAM — float32 only, batches ≤ 4GB
- Tests: `.venv/bin/python -m pytest tests/ -q` → must stay 32+ green
- Key packages: torch 2.x, scipy 1.17.1, yfinance 1.4.1, py_vollib_vectorized, aiohttp 3.14.1

## Requirements

### R1. Real SPX Market Data Calibration (§1.1)
Implement `src/market/spx_data.py` to download SPX option chains via yfinance, recompute IV from bid-ask midpoints using py_vollib_vectorized (yfinance IV is stale), apply liquidity filters (bid>0, OI≥10, spread/mid<20%), apply arbitrage filters (calendar spread + butterfly), construct forward prices F=S*exp((r-q)*T), compute log-moneyness k=log(K/F), interpolate to (8,11) FNO grid via RectBivariateSpline, and calibrate Rough Heston parameters via Newton calibrator. Research file: `research/spx_options.md`. Key dates: 2020-03-16, 2022-01-24, 2024-01-02, 2024-08-05. Use r=5% for 2024 dates. Output to `results/spx_calibration/{date}.json`. Create notebook `notebooks/01_spx_calibration.ipynb`.

**Git worktree already exists:** `/home/execorn/programming/derivatives-w1` on branch `tier1/spx`

### R2. Arbitrage-Free IV Surface Completion (§1.2)
Implement `src/arbitrage/surface_completion.py` with SVI raw parameterization w(k)=a+b*(ρ*(k-m)+√((k-m)²+σ²)) fitted slice-by-slice via scipy SLSQP with bounds (a>0, b>0, |ρ|<1, σ>0), butterfly check d²w/dk²≥0, monotone rearrangement for calendar spread enforcement. Test: mask 40% of SPX surface → complete → RMSE on held-out. Compare SVI vs cubic spline vs FNO-completion. Research: `research/arbitrage_free_surface.md`. Output: `results/spx_calibration/completion_test.json`. Notebook: `notebooks/02_surface_completion.ipynb`.

**Git worktree already exists:** `/home/execorn/programming/derivatives-w2` on branch `tier1/arbitrage`

### R3. VIX Futures Pricing (§1.3)
Implement `src/market/vix_pricing.py`. VIX²=(1/Δ)*E[∫₀^Δ v_t dt], Δ=30/365. For stationary case VIX≈√θ*100 (rough approximation). Exact: use modified Riccati ODE via scipy.integrate.solve_ivp (NOT JAX). Fetch VIX futures CSV from CBOE, use Settle column. Align with SPX dates via pd.merge. Research: `research/rough_heston_calibration.md`. ⚠️ If Rough Heston Laplace transform math is unclear, write a Gemini Deep Research prompt and ask the user. Notebook: `notebooks/03_vix_analysis.ipynb`.

**Git worktree already exists:** `/home/execorn/programming/derivatives-w3` on branch `tier1/vix`

### R4. Portfolio Greeks at Scale (§1.4)
Implement `src/greeks/portfolio_greeks.py`. Differentiable Black-Scholes via torch.distributions.Normal for Δ,Γ,Vega,Vanna,Volga. FNO parameter Jacobian via torch.func.jacfwd (6 inputs → 88 outputs). ⚠️ Spectral convolution kinks → gradient explosion: if Gamma/Speed diverge, reduce K_max or add L2 regularization. All normalizers MUST stay inside the torch graph (no numpy detachment). Research: `research/option_greeks_via_fno_autograd.md`. Notebook: `notebooks/04_greeks_portfolio.ipynb`.

**Git worktree already exists:** `/home/execorn/programming/derivatives-w4` on branch `tier1/greeks`

### R5. Crypto Derivatives — Deribit (§1.5)
Implement `src/market/deribit_data.py`. Deribit REST API (no auth needed): `https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option`. Parse instrument name "BTC-28JUN24-70000-C" → coin/expiry/strike/type. mark_iv is in % → divide by 100. Forward from put-call parity: F=K+(C-P)*exp(r*T). ⚠️ BTC V₀ can reach 0.40-0.60, σ up to 2.5 — FNO trained with V₀≤0.25, σ≤1.5 → clip to training range and warn, do NOT retrain without asking user. Research: `research/calibrating_rough_heston_to_crypto_options.md`. Notebook: `notebooks/05_crypto_calibration.ipynb`.

**Git worktree already exists:** `/home/execorn/programming/derivatives-w5` on branch `tier1/crypto`

## Acceptance Criteria

### Implementation Completeness
- [ ] All 5 stub files have zero `raise NotImplementedError` remaining
- [ ] Each module's public API matches the success criteria in TIER1_ORCHESTRATOR_PROMPT.md
- [ ] Each module has ≥2 new pytest tests in `tests/test_{module}.py`
- [ ] All 32 original tests + new tests pass: `.venv/bin/python -m pytest tests/ -q`

### Functional Correctness
- [ ] §1.1: `from src.market.spx_data import calibrate_to_market; result = calibrate_to_market(date(2024,1,2)); assert "params" in result and result["rmse_bps"] < 50`
- [ ] §1.2: `from src.arbitrage.surface_completion import complete_surface, check_butterfly; violations = check_butterfly(completed, K_grid, T_grid); assert violations.sum() == 0`
- [ ] §1.3: `from src.market.vix_pricing import model_vix; vix = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08); assert 10 < vix < 40`
- [ ] §1.4: `from src.greeks.portfolio_greeks import fno_surface_greeks; g = fno_surface_greeks(model, theta, pn, yn, S=5000.0); assert all(k in g for k in ["delta","gamma","vega","vanna","volga"])`
- [ ] §1.5: `from src.market.deribit_data import fetch_option_snapshot; import asyncio; df = asyncio.run(fetch_option_snapshot("BTC")); assert len(df) > 100 and "log_moneyness" in df.columns`

### Quality
- [ ] No modifications to protected files (fno_model.py, calibrate.py, calibrate_fast.py, normalizers.py, pricing_engine.py, app_fno.py)
- [ ] Results saved to `results/` subdirectories
- [ ] Jupyter notebooks created in `notebooks/` with plots and analysis
- [ ] All work committed to respective tier1/* branches with descriptive commit messages

### Verification
- Verify no protected files were modified: `git diff master -- src/fno_model.py src/calibrate.py src/calibrate_fast.py src/normalizers.py src/pricing_engine.py src/app_fno.py` should be empty

## Follow-up — 2026-06-18T23:01:18Z

RESUME TASK — Rough Heston FNO Calibration, Tier 1 Real-Market Extensions.

Working directory: /home/execorn/programming/derivatives
Integrity mode: development

## Current State (as of 2026-06-19T02:00 — pick up from here)

### Already DONE — do not redo:
- **§1.1 SPX** (`tier1/spx` branch, worktree `/home/execorn/programming/derivatives-w1`): COMMITTED at `3385f29`. `results/spx_calibration/2024-01-02.json` exists. `notebooks/01_spx_calibration.ipynb` exists. 35 tests passing.
- **§1.2 Arbitrage** (`tier1/arbitrage` branch, worktree `/home/execorn/programming/derivatives-w2`): `src/arbitrage/surface_completion.py` is FULLY IMPLEMENTED (576 lines, 0 NotImplementedError). `tests/test_surface_completion.py` exists (150 lines). `results/spx_calibration/completion_test.json` exists. `notebooks/02_surface_completion.ipynb` exists. **BUT NOT YET COMMITTED** — must `git add` and `git commit` in the w2 worktree first.

### Still TODO — implement these:
- **§1.3 VIX** (`tier1/vix` branch, worktree `/home/execorn/programming/derivatives-w3`): `src/market/vix_pricing.py` is still a stub with 5 NotImplementedError.
- **§1.4 Greeks** (`tier1/greeks` branch, worktree `/home/execorn/programming/derivatives-w4`): `src/greeks/portfolio_greeks.py` is still a stub with 6 NotImplementedError.
- **§1.5 Crypto** (`tier1/crypto` branch, worktree `/home/execorn/programming/derivatives-w5`): `src/market/deribit_data.py` is still a stub with 5 NotImplementedError.

---

## Critical Context

Read these files FIRST before implementing:
- `CONTEXT.md` — full technical map
- `.agents/TIER1_ORCHESTRATOR_PROMPT.md` — full briefing with code snippets, API details, pitfalls
- `research/rough_heston_calibration.md` — for §1.3
- `research/option_greeks_via_fno_autograd.md` — for §1.4
- `research/calibrating_rough_heston_to_crypto_options.md` — for §1.5

### Protected files — DO NOT MODIFY:
`src/fno_model.py`, `src/calibrate.py`, `src/calibrate_fast.py`, `src/normalizers.py`, `src/pricing_engine.py`, `src/app_fno.py`

### Load model (copy-paste ready)
```python
import sys; sys.path.insert(0, 'src')
import torch, numpy as np
from fno_model import MirrorPaddedFNO2d
from normalizers import ParameterNormalizer, IVSurfaceNormalizer

model = MirrorPaddedFNO2d()
model.load_state_dict(torch.load('artifacts/weights/fno_v2_final_prod.pth', map_location='cpu'))
model.eval()
pn = ParameterNormalizer.load('artifacts/models/param_normalizer_v2.npz')
yn = IVSurfaceNormalizer.load('artifacts/models/iv_normalizer_v2.npz')
```

### Newton calibration
```python
from calibrate_fast import calibrate_newton
result = calibrate_newton(model, iv_surface, MATURITIES, STRIKES, pn=pn, yn=yn)
# result keys: v0, zeta, lambda, history, elapsed
```

### Grids
```python
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
STRIKES    = np.linspace(-0.5, 0.5, 11)
```

### Environment
- Python 3.14, venv at `/home/execorn/programming/derivatives/.venv/bin/python`
- Run tests from the MAIN repo dir: `/home/execorn/programming/derivatives/.venv/bin/python -m pytest tests/ -q`
- RTX 3060, float32 only

---

## Step 1 — Commit §1.2 (do this first, before anything else)

In worktree `/home/execorn/programming/derivatives-w2` on branch `tier1/arbitrage`:
```bash
cd /home/execorn/programming/derivatives-w2
git add src/arbitrage/surface_completion.py tests/test_surface_completion.py
git commit -m 'feat: implement arbitrage-free IV surface completion (§1.2)'
```
Also copy results and notebook to the main repo if not already there:
- `results/spx_calibration/completion_test.json` — already exists in main repo ✓
- `notebooks/02_surface_completion.ipynb` — already exists in main repo ✓

---

## Step 2 — Implement §1.3 VIX Futures Pricing

**Worktree:** `/home/execorn/programming/derivatives-w3` | **Branch:** `tier1/vix`
**File to implement:** `src/market/vix_pricing.py` (currently 78 lines, 5 NotImplementedError)
**Research:** `research/rough_heston_calibration.md`

Key facts:
- VIX² = (1/Δ) * E[∫₀^Δ v_t dt], Δ = 30/365
- Stationary approximation: VIX ≈ √θ * 100
- Exact: modified Riccati ODE via `scipy.integrate.solve_ivp` (NOT JAX)
- VIX futures CSV from CBOE: https://www.cboe.com/tradable_products/vix/vix_historical_data/ — use Settle column
- Align VIX futures with SPX dates via pd.merge
- DO NOT use JAX

Success check:
```python
from src.market.vix_pricing import model_vix
vix = model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08)
assert 10 < vix < 40
```

Deliverables:
- `src/market/vix_pricing.py` — fully implemented, 0 NotImplementedError
- `tests/test_vix_pricing.py` — ≥2 tests
- `notebooks/03_vix_analysis.ipynb`
- Committed to `tier1/vix`

---

## Step 3 — Implement §1.4 Portfolio Greeks at Scale

**Worktree:** `/home/execorn/programming/derivatives-w4` | **Branch:** `tier1/greeks`
**File to implement:** `src/greeks/portfolio_greeks.py` (currently 109 lines, 6 NotImplementedError)
**Research:** `research/option_greeks_via_fno_autograd.md`

Key facts:
```python
# Differentiable Black-Scholes:
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

# FNO Jacobian (6 inputs → 88 outputs):
J = torch.func.jacfwd(fno_forward_wrapper)(theta)  # shape (88, 6)
```

⚠️ Spectral kinks → gradient explosion. If Gamma diverges, reduce K_max or add L2 regularization on spectral weights. All normalizers MUST stay inside the torch graph.

Success check:
```python
from src.greeks.portfolio_greeks import fno_surface_greeks
g = fno_surface_greeks(model, theta, pn, yn, S=5000.0)
assert all(k in g for k in ["delta", "gamma", "vega", "vanna", "volga"])
```

Deliverables:
- `src/greeks/portfolio_greeks.py` — fully implemented
- `tests/test_portfolio_greeks.py` — ≥2 tests
- `notebooks/04_greeks_portfolio.ipynb`
- Committed to `tier1/greeks`

---

## Step 4 — Implement §1.5 Crypto/Deribit

**Worktree:** `/home/execorn/programming/derivatives-w5` | **Branch:** `tier1/crypto`
**File to implement:** `src/market/deribit_data.py` (currently 122 lines, 5 NotImplementedError)
**Research:** `research/calibrating_rough_heston_to_crypto_options.md`

Key facts:
```python
import requests
r = requests.get(
    "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
    params={"currency": "BTC", "kind": "option"}, timeout=10)
data = r.json()["result"]
# Each: instrument_name, mark_iv (in %), underlying_price, open_interest, bid_iv, ask_iv

# Parse name: "BTC-28JUN24-70000-C"
from datetime import datetime
parts = name.split("-")
coin   = parts[0]
expiry = datetime.strptime(parts[1], "%d%b%y").date()
strike = int(parts[2])
opt_t  = parts[3]

# Convert IV: mark_iv is in percent
iv_decimal = mark_iv_pct / 100.0

# Forward from put-call parity: F = K + (C - P) * exp(r * T)
```

⚠️ BTC V₀ can reach 0.40-0.60, σ up to 2.5 — FNO trained with V₀≤0.25. Clip to training range and WARN the user. Do NOT retrain.

Success check:
```python
from src.market.deribit_data import fetch_option_snapshot
import asyncio
df = asyncio.run(fetch_option_snapshot("BTC"))
assert len(df) > 100 and "log_moneyness" in df.columns
```

Deliverables:
- `src/market/deribit_data.py` — fully implemented
- `tests/test_deribit_data.py` — ≥2 tests
- `notebooks/05_crypto_calibration.ipynb`
- Committed to `tier1/crypto`

---

## Step 5 — Final Integration (after all 5 tasks committed)

1. From main repo (`/home/execorn/programming/derivatives` on `master`):
```bash
git merge tier1/spx --no-ff -m "feat: §1.1 SPX real market calibration"
git merge tier1/arbitrage --no-ff -m "feat: §1.2 arbitrage-free surface completion"
git merge tier1/vix --no-ff -m "feat: §1.3 VIX futures pricing"
git merge tier1/greeks --no-ff -m "feat: §1.4 portfolio Greeks at scale"
git merge tier1/crypto --no-ff -m "feat: §1.5 Deribit crypto calibration"
```
2. Run full test suite: `.venv/bin/python -m pytest tests/ -q` → must show 35+ passed
3. Verify all 5 success checks pass
4. Update `CONTEXT.md` Tier 1 status table — mark all 5 tasks ✅

---

## Acceptance Criteria

### Implementation
- [ ] §1.2: `git log tier1/arbitrage` shows a new commit with surface_completion.py
- [ ] §1.3: `model_vix(kappa=1.0, theta=0.08, sigma=0.8, rho=-0.34, v0=0.10, H=0.08)` returns value between 10 and 40
- [ ] §1.4: `fno_surface_greeks(model, theta, pn, yn, S=5000.0)` returns dict with keys delta, gamma, vega, vanna, volga
- [ ] §1.5: `asyncio.run(fetch_option_snapshot("BTC"))` returns DataFrame with >100 rows and `log_moneyness` column
- [ ] All 5 stubs have 0 NotImplementedError
- [ ] Each has ≥2 pytest tests
- [ ] `notebooks/03_vix_analysis.ipynb`, `notebooks/04_greeks_portfolio.ipynb`, `notebooks/05_crypto_calibration.ipynb` exist

### Tests
- [ ] `.venv/bin/python -m pytest tests/ -q` run from `/home/execorn/programming/derivatives` → 35+ passed
- [ ] No modifications to protected files

### Git
- [ ] All 5 `tier1/*` branches have implementation commits (not just the baseline)
- [ ] All branches merged to master with `--no-ff`
- [ ] `CONTEXT.md` updated with ✅ for all 5 Tier 1 tasks

