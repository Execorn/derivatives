# Absolute Maximum Roadmap — Neural Networks in Derivatives Pricing
**Project:** Deep Rough Heston Calibration via FiLM-FNO  
**Institution:** МФТИ ФПМИ, кафедра БИТ  
**Current state:** FNO v2/v3, Newton calibration, 51-page thesis  

---

## TIER 1 — High impact, achievable in 1–2 months

### 1.1 Real Market Data Calibration
The single most important missing piece. Everything so far is synthetic.

- Download SPX option chain from **Yahoo Finance** (`yfinance`) or CBOE DataShop
- Clean data: remove illiquid strikes (bid=0), apply bid-ask midpoint, filter arbitrage violations
- Calibrate FNO v2 to real SPX smiles for multiple dates (2020 crash, 2022 bear, 2024 rally)
- Plot time series of recovered `(κ, θ, σ, ρ, V₀, H)` — do they make economic sense?
- Compare Rough Heston fit quality vs standard Heston (H=0.5) on real data
- Show the "rough volatility premium" in H estimates: empirically H ≈ 0.08–0.12 for SPX

**Key result for thesis:** calibration residuals on real data, parameter time series plot.

---

### 1.2 Arbitrage-Free IV Surface Completion
Real market data has missing quotes (illiquid strikes/maturities). The FNO can fill them.

- Given sparse observed IV (e.g., 3 maturities × 5 strikes), calibrate FNO and interpolate
- Check arbitrage conditions:
  - Calendar spread: `∂C/∂T ≥ 0`
  - Butterfly: `∂²C/∂K² ≥ 0`
  - SVI parametrization as benchmark (Gatheral 2004)
- Apply **monotone rearrangement** (Chernozhukov 2010) as post-processing to enforce convexity
- Compare FNO completion vs cubic spline vs SVI on held-out strikes

**Key result:** FNO completes the surface without arbitrage in 4ms vs minutes for SVI fitting.

---

### 1.3 VIX and Variance Swap Pricing
Natural extension — VIX is a model-free variance swap rate.

- Compute model VIX from calibrated Rough Heston: `VIX² = E[∫₀^(1/12) v_t dt]`
- The Fourier-COS engine already handles this via `E[∫v_t dt]` in characteristic function
- Calibrate jointly to SPX options + VIX futures (joint calibration constraint)
- Show Rough Heston matches both SPX smile and VIX term structure simultaneously
- Standard Heston cannot do this (Gatheral's double calibration problem)

**Key result:** a model that prices both equity options and volatility derivatives jointly.

---

### 1.4 Greeks at Scale via Autograd
Currently Greeks are computed for a single parameter set. Scale to risk management.

- Compute full Greek surface `Δ(T,K)`, `Γ(T,K)`, `Vega(T,K)`, `Vanna(T,K)`, `Volga(T,K)`
- **Theorem:** FNO Greeks are analytic (no finite difference noise) via `torch.autograd`
- Benchmark FNO Greeks vs finite-difference COS Greeks (speed and accuracy)
- Portfolio Greeks: sum over 100 option positions, compute hedge ratios in real time
- Show P&L attribution: delta P&L + vega P&L + higher-order terms

**Key result:** portfolio-level Greeks in <50ms (vs minutes for bump-and-reprice).

---

### 1.5 Crypto Derivatives (Deribit)
BTC/ETH options are freely available and have extreme rough volatility.

- Download BTC/ETH option chain from Deribit API (free, no account needed)
- Hurst exponents for crypto: H ≈ 0.05–0.10 (rougher than equity)
- The FNO v3 (learnable H) should handle this well
- Calibrate daily from 2021–2024 covering multiple crypto cycles
- Show how H changes during market stress (crashes, pumps)

**Key result:** H dynamics for crypto — a novel empirical finding.

---

## TIER 2 — Significant research, 2–4 months

### 2.1 Rough Bergomi Model
The competitor to Rough Heston, arguably better empirical fit.

- Rough Bergomi (Bayer-Friz-Gatheral 2016): `dv_t = ξ₀(t) exp(η W_t^H - η²/2 t^{2H})`
- No closed-form characteristic function → must use Hybrid scheme MC or deep learning
- Train FNO on Rough Bergomi IV surfaces (requires ~10k MC simulations per sample)
- Compare: Rough Bergomi vs Rough Heston on SPX data — which fits better?
- Three parameters only: `(H, η, ρ)` → very low-dimensional calibration

**Key result:** comparative study of two rough volatility models via FNO surrogates.

---

### 2.2 Neural SDE Calibration (Model-Free)
The deep learning frontier: learn the SDE directly from data.

- Neural SDE: `dS = μ dt + σ_θ(t, S, Z) dW` where σ_θ is a neural network
- Fit to market prices without assuming any parametric model
- Tsilifis et al. (2022), Cuchiero et al. (2023) showed Neural SDEs can match any smooth smile
- Challenge: arbitrage-free training requires careful loss design
- Compare: Neural SDE vs Rough Heston FNO on out-of-sample dates

**Key result:** model-free deep calibration as upper bound on fit quality.

---

### 2.3 Lifted Heston / Multi-Factor Rough Volatility
Rough Heston can be approximated by a sum of Ornstein-Uhlenbeck processes.

- Abi Jaber (2022): `v_t = ∑ᵢ cᵢ Vᵢ_t` where each `Vᵢ` is a standard CIR
- Advantage: exact Markovian embedding → fast simulation, no fractional Brownian motion
- The FNO surrogate for lifted Heston would be faster to train (all COS labels are exact)
- Show that 3-factor lifted Heston ≈ Rough Heston with H=0.1 for option pricing
- Calibrate lifted Heston directly from smile in <500ms via Newton

**Key result:** fast Markovian approximation that matches rough vol behavior.

---

### 2.4 Transformer Architecture for IV Surface
Replace FNO with attention mechanism.

- Treat the IV surface as a 2D sequence: each (T,K) point is a token
- Cross-attention between parameter embedding and spatial tokens
- Advantage: handles **irregular grids** (real market quotes are not on a uniform grid)
- Compare: Transformer vs FNO on irregular real market data
- Reference: Vidales et al. (2023) "Rough Transformers for Continuous Recurrent Neural Networks"

**Key result:** direct calibration from irregular market quotes without grid interpolation.

---

### 2.5 Bayesian Calibration with Uncertainty
Replace point estimate with posterior distribution over parameters.

- Amortized variational inference: train an encoder `q(θ|IV_obs)`
- Given observed IV surface, output mean + covariance of θ in one forward pass
- Compare: Bayesian FNO vs FIM ellipsoid (FIM is the Laplace approximation of the posterior)
- Show coverage: 95% credible intervals contain true parameters 95% of the time
- Application: risk-adjusted pricing with parameter uncertainty

**Key result:** calibration uncertainty quantification beyond FIM (captures non-Gaussianity).

---

### 2.6 Exotic Options Pricing
Extend from vanilla to path-dependent payoffs.

- **Barrier options** (up-and-out, down-and-in): need full path simulation under Rough Heston
- **Asian options** (arithmetic average): require moment matching or MC
- **Autocallables** (dominant structured product in Russia/Europe): digital + barrier + coupon
- Train separate FNO for each payoff type, or use a **universal payoff encoder**
- Reference: Cheridito et al. (2021) "Efficient and Accurate Longstaff-Schwartz for High-Dim Exotics"

**Key result:** extend the "1400× speedup" from vanilla to exotic derivatives.

---

### 2.7 Interest Rate Derivatives
Move beyond equity vol — rates markets are equally rough.

- **Hull-White / Vasicek model** via FNO: learn P(0,T) → swaption prices
- **SABR model** for caps/floors: analytical Hagan approximation as ground truth
- **Rough interest rates** (Alfeus-Schlögl 2022): empirical evidence H ≈ 0.4 for rate vol
- Calibrate to EUR swaption cube (available from Bloomberg via Python `blpapi`)
- Joint equity-rates model: Heston + Hull-White correlation structure

**Key result:** multi-asset FNO covering both equity and rates in one model.

---

## TIER 3 — Advanced research, 4–12 months (PhD-level)

### 3.1 Deep Hedging
Model-free hedging via reinforcement learning — Bühler et al. (2019).

- Train a policy `π_θ(t, S_t, v_t) → δ_t` (hedge ratio) to minimize CVaR of replication error
- Environment: simulate Rough Heston paths, place hedges, compute P&L
- Compare deep hedging vs FNO-Δ vs Black-Scholes-Δ: P&L distribution histograms
- Transaction costs: penalize trading frequency (L1 penalty on `Δδ`)
- Show deep hedging is optimal under rough vol + transaction costs where B-S breaks down

**Key result:** end-to-end learned hedging strategy that outperforms classical Greeks.

---

### 3.2 XVA Computation via Machine Learning
Counterparty credit risk — the most computationally expensive problem in banking.

- **CVA** (Credit Valuation Adjustment): `CVA = (1-R) ∫ LGD · EE(t) · λ(t) dt`
- Requires: Expected Exposure (EE) over thousands of Monte Carlo paths × thousands of dates
- Replace nested Monte Carlo with FNO regressor: `EE(t, market_state) ≈ FNO(market_state)`
- Compare: standard nested MC (O(N²)) vs FNO regression (O(N)) — target 100× speedup
- Reference: Huge-Savine (2020) "Differential Machine Learning" (Risk Magazine)

**Key result:** CVA engine running in seconds instead of hours — directly bankable.

---

### 3.3 Physics-Informed Neural Networks for Rough PDE
Price exotics by solving the fractional PDE with a neural network.

- Rough Heston does not satisfy a standard PDE (non-Markovian)
- But the lifted approximation does: `∂V/∂t + L[V] = 0` with `L` depending on all Vᵢ
- PINN: minimize `||∂V/∂t + L[V]||² + ||BC||²` without any simulation
- Challenge: multi-factor PDE has dimension 3-10 → curse of dimensionality
- Use **deep Galerkin method** (Sirignano-Spiliopoulos 2018)

**Key result:** mesh-free PDE solver for high-dimensional rough volatility pricing.

---

### 3.4 Generative Model for Scenario Generation
Replace historical simulation with learned distribution.

- Train a **Conditional Normalizing Flow** or **Score-based Diffusion** on IV surface time series
- Generate risk scenarios: `P(IV_t+1 | IV_t, macro_features)`
- Application: market risk (VaR, ES) without needing historical data going back 10+ years
- Stress test: generate "2008-like" or "COVID-like" scenarios from model
- Reference: Cont-Xu (2022) "Tail-GAN: Tail-Risk Scenario Generation"

**Key result:** data-augmented scenario generator for regulatory stress testing.

---

### 3.5 Online Learning and Streaming Calibration
Real-time recalibration as market ticks arrive.

- Current: recalibrate from scratch at each tick (541ms)
- Online learning: update parameters incrementally using stochastic gradient
- Kalman-filter-style: `θ_t = θ_{t-1} + K_t · (IV_observed - IV_predicted)`
- Where `K_t` is the Kalman gain from FIM: `K = P H^T (H P H^T + R)^{-1}`
- Show: 10× faster convergence starting from previous estimate vs cold start

**Key result:** sub-100ms recalibration that respects parameter continuity over time.

---

### 3.6 Multi-Underlying Joint Calibration
One model for the full equity universe.

- Calibrate SPX, AAPL, MSFT, TSLA simultaneously with shared latent factors
- Factor structure: `θᵢ = Λ fₜ + εᵢ` (parameters = market factor + idiosyncratic)
- FNO with cross-attention between underlyings: learn correlation structure
- Application: basket option pricing, dispersion trading, correlation products
- Reference: Guyon (2022) "Dispersion-Constrained Martingale Schrödinger Bridges"

**Key result:** joint calibration of 50 equity underliers in one FNO pass.

---

### 3.7 Rough Volatility Under Microstructure
Connect market microstructure to rough volatility.

- Empirically: H ≈ 0.1 arises from aggregation of order flow (Gatheral-Jaisson-Rosenbaum 2018)
- Train a **microstructure → macroscopic vol** model using limit order book data
- Input: bid-ask spread, order flow imbalance, trade-by-trade data
- Output: predicted H, σ for the next day
- Data: Deribit tick data (free) or Refinitiv Tick History (university license)

**Key result:** forecasting rough volatility parameters from order book — tradeable alpha.

---

### 3.8 Transfer Learning Across Market Regimes
Adapt calibrated model from normal to stress periods.

- Rough Heston trained on 2015–2019 data: normal conditions
- Fine-tune on 2020 COVID crash with minimal data (few-shot learning)
- Show: fine-tuned model recovers market prices 10× faster than from-scratch calibration
- Meta-learning (MAML): train model to be quickly adaptable to new regimes

**Key result:** regime-robust calibration that doesn't fail in crashes.

---

## TIER 4 — Production / Deployment

### 4.1 REST API Pricing Engine
Production-ready pricing microservice.

```
POST /price
{
  "model": "rough_heston_v3",
  "params": {"kappa": 1.0, "theta": 0.08, ...},
  "instruments": [{"type": "vanilla_call", "K": 100, "T": 0.5}, ...]
}
→ {"prices": [...], "greeks": [...], "latency_ms": 4}
```

- FastAPI + Uvicorn backend, GPU inference via CUDA
- Redis cache for repeated parameter sets
- Rate limiting, authentication, OpenAPI docs
- Target: <10ms p99 latency for single option, <50ms for full surface

---

### 4.2 Real-Time Bloomberg/Reuters Feed Integration
Connect to live market data.

- Receive live SPX option quotes via Bloomberg `blpapi` or Refinitiv Eikon
- Continuously recalibrate Rough Heston every tick (use Kalman update from §3.5)
- Stream calibrated parameters to dashboard (live Streamlit or Grafana)
- Alert when fit quality degrades (market dislocation detection)

---

### 4.3 Regulatory Capital Optimization
Practical application in banking (кафедра БИТ specialty).

- Basel III/IV FRTB: Internal Model Approach requires model validation
- Show FNO passes backtesting requirements: ES breach rate < 5%
- Compute regulatory VaR (99%, 10-day) 100× faster with FNO scenarios vs historical sim
- Reference: Basel Committee on Banking Supervision (2019), FRTB standards

---

### 4.4 MLflow + Model Registry
Production ML engineering.

- Track all training runs: hyperparameters, metrics, model artifacts
- Model versioning: v1 (MC labels), v2 (COS labels), v3 (learnable H), v4 (future)
- A/B testing framework: deploy v2 and v3 simultaneously, route traffic by noise level
- Automated retraining pipeline when model drift is detected

---

## Summary Priority Matrix

| Goal | Impact | Effort | Priority |
|---|---|---|---|
| Real SPX calibration | ⭐⭐⭐⭐⭐ | Low | **DO FIRST** |
| VIX + Variance Swaps | ⭐⭐⭐⭐⭐ | Medium | **DO SECOND** |
| Crypto (Deribit) | ⭐⭐⭐⭐ | Low | High |
| Arbitrage-free completion | ⭐⭐⭐⭐ | Low | High |
| Greeks at scale | ⭐⭐⭐⭐ | Low | High |
| Rough Bergomi FNO | ⭐⭐⭐⭐ | Medium | Medium |
| Transformer architecture | ⭐⭐⭐ | Medium | Medium |
| Bayesian calibration | ⭐⭐⭐ | High | Medium |
| Exotic options | ⭐⭐⭐⭐ | High | Medium |
| Deep Hedging | ⭐⭐⭐⭐⭐ | High | Long-term |
| XVA / CVA | ⭐⭐⭐⭐⭐ | High | Long-term |
| Neural SDE | ⭐⭐⭐ | Very High | Research |
| PINN fractional PDE | ⭐⭐⭐ | Very High | Research |

---

## Recommended Next Sprint (2 weeks)

1. **Day 1–2:** Download SPX option chain for 5 dates via `yfinance`, clean data
2. **Day 3–4:** Run Newton calibration on real SPX smiles, plot fit quality
3. **Day 5–6:** Time series of `(H, σ, ρ)` from 2020–2024 — show H ≈ 0.08–0.12
4. **Day 7–8:** Crypto: pull BTC options from Deribit API, calibrate
5. **Day 9–10:** VIX: compute model VIX from calibrated params, compare to realized VIX
6. **Day 11–14:** Write results into thesis Chapter 5 extension

This alone would make the thesis publishable in a quant finance journal.

---

## Data Sources (all free)

| Data | Source | Access |
|---|---|---|
| SPX options | `yfinance` / CBOE DataShop | Free |
| BTC/ETH options | Deribit REST API | Free, no auth |
| VIX term structure | CBOE website | Free CSV |
| Realized volatility | Oxford-Man Institute | Free |
| Historical H estimates | Rough Vol database (Volatility is Rough paper) | Free |
| Interest rate curves | ECB/Fed API | Free |

