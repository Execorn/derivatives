# ROADMAP — Phase P4 and Beyond
# Rough Heston FNO Calibration System — Future Development

*Last updated: 2026-06-21*
*Baseline: Phase P3 complete (537 tests, FNO v3 R²=0.9991, calibration p50=541ms)*

---

## Tier 1 — Real Market Integration (Active / Next 4–8 weeks)

These tasks are already researched (Deep Research reports in `research/`) and
stubbed (`src/market/spx_data.py`, `src/arbitrage/surface_completion.py`, etc.).
Implementation is the current priority.

### T1.1 — SPX Real Market Calibration
**Files:** `src/market/spx_data.py`, `src/calibration/batch_calibration.py`
**Effort:** 1–2 weeks
**Prerequisites:** yfinance ✅, py_vollib_vectorized ✅
**Tasks:**
- Complete `download_spx_chain` with retry logic and rolling cache (parquet)
- Complete `clean_chain`: bid-ask spread filter, OI threshold, remove below-intrinsic
- Complete `to_iv_surface`: map real chain to (8T × 11K) grid via bilinear interpolation
- Backtest calibration on 60 SPX trading days (2024-01-01 to 2024-03-31)
- Compare calibrated Heston vs market Black-Scholes smile

**Success Metrics:**
- Calibration RMSE < 50 bps on 80% of backtest days
- `calibrate_single(date_str, currency="SPX")` returns valid result in < 2s end-to-end
- Zero failures on `pytest tests/test_spx_data.py -v`

---

### T1.2 — Arbitrage-Free Surface Completion
**Files:** `src/arbitrage/surface_completion.py`
**Effort:** 1 week
**Prerequisites:** T1.1 (needs real SPX data for validation)
**Tasks:**
- Complete SVI slice fitting with Durrleman condition check
- Implement `complete_surface`: fill missing strikes via SVI extrapolation
- Validate: zero butterfly violations after completion on SPX data
- GPU batch SVI fitting for all 8 maturities simultaneously

**Success Metrics:**
- `check_butterfly(completed_surface)` returns zero violations for 95% of dates
- `check_calendar_spread(completed_surface)` returns zero violations always
- Completion < 200ms per surface on GPU

---

### T1.3 — VIX Term Structure Joint Calibration
**Files:** `src/market/vix_futures.py`, `src/market/vix_pricing.py`, `src/calibration/joint_calibration.py`
**Effort:** 1–2 weeks
**Prerequisites:** T1.1
**Tasks:**
- Complete VIX futures fetching via yfinance (8 front contracts)
- Implement Rough Heston VIX price via Laplace transform (Gil-Pelaez inversion)
- Joint loss: `L = λ₁·RMSE(SPX_IV) + λ₂·RMSE(VIX_futures)`, λ₁=1.0, λ₂=0.5
- Optimize joint calibration with Newton-Gauss on GPU

**Success Metrics:**
- Joint calibration RMSE_SPX < 60 bps, RMSE_VIX < 1 vol point
- Fit 8 VIX maturities simultaneously in < 1s

---

### T1.4 — Portfolio Greeks at Scale
**Files:** `src/greeks/portfolio_greeks.py`, `src/benchmarks/greeks_benchmark.py`
**Effort:** 1 week
**Prerequisites:** T1.1 (real spot price S)
**Tasks:**
- Batch `portfolio_greeks` for N=10,000 positions without Python loops
- Full `torch.func.jacfwd` Jacobian for FNO parameter risk (∂IV/∂θ)
- Delta-hedging P&L backtest vs Black-Scholes Greeks
- Vega bucketing benchmark: compare to SABR bucket Greeks

**Success Metrics:**
- 10,000-position portfolio Greek aggregation < 100ms on GPU
- Delta-hedge P&L attribution R² > 0.85 (vs Black-Scholes baseline of ~0.60)

---

### T1.5 — Deribit Crypto Options Live Feed
**Files:** `src/market/deribit_data.py`, `src/market/deribit_ws.py`
**Effort:** 1 week
**Prerequisites:** aiohttp ✅
**Tasks:**
- Complete `fetch_option_snapshot` with full error handling and rate limiting
- Build IV surface from Deribit BTC/ETH data (934 options per snapshot)
- Implement `deribit_ws.py` WebSocket streaming for real-time IV updates
- Auto-recalibration on new snapshot (< 2s latency target)

**Success Metrics:**
- `build_iv_surface(df, MATURITIES, STRIKES)` succeeds on live Deribit data
- Real-time recalibration triggered on IV change > 0.5%
- Zero asyncio thread-safety issues (BUG-6 fixed)

---

## Tier 2 — Production Services (6–16 weeks)

### T2.1 — FastAPI REST Calibration Service
**Files:** `src/api/server.py`
**Effort:** 2 weeks
**Prerequisites:** Tier 1 complete, FastAPI ✅
**Tasks:**
- `/calibrate` POST endpoint: accepts currency + date → returns CalibrationResult JSON
- `/greeks` POST endpoint: accepts portfolio JSON → returns aggregated Greeks
- `/surface` GET endpoint: streams IV surface as Plotly JSON
- `/health` endpoint for monitoring
- Background task queue: async calibration jobs with status polling
- Docker container with CUDA 12 base image

**Success Metrics:**
- `/calibrate` p50 < 600ms, p99 < 2s for SPX
- Concurrent: 4 simultaneous calibrations without VRAM OOM
- OpenAPI docs auto-generated

---

### T2.2 — Streaming Calibration (Tick-by-Tick)
**Files:** `src/market/deribit_ws.py`, new `src/streaming/`
**Effort:** 2–3 weeks
**Prerequisites:** T1.5, T2.1
**Tasks:**
- Incremental surface update: only re-fit changed maturities
- Warm-start Newton from previous calibration (reduces iterations 20→5)
- Alert system: notify if H changes by > 0.02 (regime shift signal)
- Persistent WebSocket with exponential backoff reconnect

**Success Metrics:**
- Recalibration latency < 500ms per tick on GPU
- H stability: std(H) < 0.01 over 1h of trading

---

### T2.3 — Deep Hedging RL Environment
**Files:** new `src/hedging/`
**Effort:** 3–4 weeks
**Prerequisites:** T1.4
**Tasks:**
- Gym-compatible environment: FNO as vol surface simulator
- State: (portfolio greeks, current IV surface, H, time-to-expiry)
- Action: delta/vega hedge ratios
- Reward: Sharpe of hedged P&L minus transaction costs
- PPO agent baseline from stable-baselines3
- Compare deep hedge P&L vs Black-Scholes delta hedge

**Success Metrics:**
- Deep hedge Sharpe > 1.5× Black-Scholes delta hedge on held-out test set
- Training: 1M steps in < 4h on RTX 3060

---

### T2.4 — Hurst Dynamics Historical Study UI
**Files:** `src/app_fno.py` (add tab), `src/analysis/hurst_dynamics.py`
**Effort:** 1 week
**Prerequisites:** T1.1, T1.5
**Tasks:**
- Streamlit tab: H(t) time series chart for SPX + BTC + ETH
- Pettitt regime change detection display
- Correlation: H vs VIX, H vs market realized vol
- Export CSV of H time series

**Success Metrics:**
- H time series chart renders for date range 2023-01-01 to present
- Regime change p-value displayed with annotation on chart

---

## Tier 3 — Research Extensions (3–9 months)

### T3.1 — XVA / CVA Module
**Effort:** 4–6 weeks
**Tasks:**
- Credit-adjusted option pricing: CVA = sum_i E[V(τ_i)⁺] × LGD × PD_i
- Rough Heston as underlying dynamics for exposure simulation
- Integration with CDS spread curves (QuantLib or custom)
- Wrong-way risk: correlation between counterparty default and vol

**Success Metrics:**
- CVA computation for vanilla portfolio in < 30s
- Validated against published CVA numbers (Brigo & Masetti benchmark)

---

### T3.2 — Multi-Asset FNO (Correlation Surface)
**Effort:** 6–8 weeks
**Tasks:**
- Extend FNO to 3D input: (T, K₁, K₂) for basket/spread options
- Training data: multi-variate Rough Heston via Bernstein lifting
- Correlation structure via learnable ρ_12 parameter
- Applications: SPX/VIX spread options, BTC/ETH correlation products

**Success Metrics:**
- Basket option pricing error < 1% vs Monte Carlo (10M paths)
- FNO inference < 50ms for 3D surface

---

### T3.3 — Transformer-Based Vol Surface Model
**Effort:** 8–12 weeks
**Tasks:**
- Replace FNO with a transformer encoder over (T, K) grid tokens
- Cross-attention between parameter tokens and surface tokens
- Positional encoding: log-moneyness + log-tenor
- Comparison study: FNO vs Transformer on R², MAE, calibration speed

**Success Metrics:**
- R² ≥ 0.9995 (better than FNO's 0.9991)
- Calibration p50 ≤ 300ms (faster than Newton on FNO)

---

### T3.4 — Rough Bergomi Extension
**Effort:** 4–6 weeks
**Tasks:**
- Implement Rough Bergomi via Cholesky-factored fBM simulation on GPU
- Training data generator alongside existing Rough Heston
- Model selection: AIC/BIC comparison on SPX data
- Joint calibration: let data select between rHeston and rBergomi

**Success Metrics:**
- rBergomi forward variance fit: RMSE_ATM < 0.5%
- GPU simulation of 10,000 paths in < 2s

---

## Tier 4 — Production & Infrastructure (6–18 months)

### T4.1 — Production Kubernetes Deployment
**Effort:** 3–4 weeks
**Tasks:**
- Helm chart for GPU pod (CUDA 12, RTX class)
- Auto-scaling: scale out at calibration queue > 10
- Redis job queue for calibration requests
- Prometheus metrics: calibration latency, RMSE distribution, VRAM usage
- Grafana dashboard

**Success Metrics:**
- 99.9% uptime over 30 days
- < 1s p99 calibration latency under 10 concurrent requests

---

### T4.2 — Regulatory Compliance (FRTB/Basel IV)
**Effort:** 6–8 weeks
**Tasks:**
- Sensitivities-Based Method (SBM): compute GIRR/VEGA buckets per FRTB
- Internal Model Approach: documentation of FNO as approved internal model
- P&L attribution test: automated daily Spearman correlation check
- Model risk framework: confidence intervals via FIM on all calibrated params

**Success Metrics:**
- P&L attribution Spearman ρ > 0.80 over rolling 250-day window
- All FRTB sensitivity buckets computed in < 5s

---

### T4.3 — Real-Time Market Data Integration
**Effort:** 4–6 weeks
**Tasks:**
- Bloomberg BLPAPI or Refinitiv Eikon connector for institutional data
- Real-time SPX options tape (L2 data) processing
- Sub-100ms surface update pipeline
- Circuit breaker: freeze calibration if RMSE > 200 bps (data quality issue)

**Success Metrics:**
- End-to-end latency (quote received → calibration updated) < 200ms
- Zero stale surfaces > 5 minutes during market hours

---

## Priority Summary

```
Now      ████████████████░░░░░░░░░░░░░  Tier 1 (T1.1–T1.5)
4–8 wks  ████████░░░░░░░░░░░░░░░░░░░░░  Tier 2 (T2.1–T2.4)
3–9 mo   ████░░░░░░░░░░░░░░░░░░░░░░░░░  Tier 3 (T3.1–T3.4)
6–18 mo  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░  Tier 4 (T4.1–T4.3)
```

## Key Dependencies

```
T1.1 SPX ──────────────────┐
T1.2 SVI ──────────────────┤──► T2.1 API ──► T4.1 k8s
T1.3 VIX ──────────────────┤
T1.4 Greeks ───────────────┤──► T2.3 Deep Hedging ──► T4.2 FRTB
T1.5 Deribit ──────────────┘──► T2.2 Streaming ──► T4.3 RT Data

T3.2 Multi-Asset FNO (independent)
T3.3 Transformer (independent, can run with T1.1 data)
T3.1 XVA (needs T1.3 VIX)
T3.4 rBergomi (needs T1.1 data)
```
