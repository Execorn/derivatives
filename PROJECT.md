# Project: Phase 7 (P7) Multi-Asset Calibration Framework

## Architecture
- `src/pricing/mlsv_gpu.py`: McKean-Vlasov SDE particle solver with kernel density regression on GPU.
- `src/market/fx_data.py` & `src/calibration/fx_calibration.py`: Garman-Kohlhagen delta-to-strike converters, Bloomberg/FRED loaders, and SABR ($\beta=1$) calibration pipeline.
- `src/market/rates_data.py` & `src/pricing/sabr_rates.py` (and `src/pricing/bachelier.py` if needed): SOFR swaption cube loaders, displaced/normal SABR pricing engines, and bilinear parameter interpolation.
- `src/market/commodity_data.py` & `src/pricing/schwartz_smith.py`: CME Crude Oil / Gold loaders, Schwartz-Smith two-factor pricing, and Heston comparison.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | E2E Test Suite | Build opaque-box E2E test suite (Tiers 1-4) for all 4 models and publish `TEST_READY.md`. | None | DONE (fbbbd5ae-0f5c-43d2-99c3-ce6561095e8a) |
| M2 | Equity MLSV (W1) | GPU McKean-Vlasov particle solver, SPX calibration, validation notebook. | M1 | DONE (3a6c1f89-b413-4af8-adcc-4b6d984df51b) |
| M3 | FX SABR (W2) | Delta-to-strike inversion, SABR calibration to EUR/USD smile, validation notebook. | M1 | DONE (d812462c-557f-4e3e-b4dc-b7883df77796) |
| M4 | Rates LMM-SABR (W3) | Displaced SABR pricing, SOFR swaption cube interpolation, validation notebook. | M1 | DONE (849b328c-3b8a-4416-9ebf-82e1b6e8a8cb) |
| M5 | Commodity Schwartz-Smith (W4) | CME futures options cleaning, Schwartz-Smith pricing vs Heston, validation notebook. | M1 | DONE (5b0f0963-bd5a-4133-900f-b3e407d694dc) |
| M6 | Integration & Hardening (Tier 5) | Final E2E pass, adversarial testing and coverage hardening. | M2, M3, M4, M5 | DONE (fa264af5-48a5-4fbe-9a2f-c8ae0af25573) |

## Interface Contracts
### Equity MLSV Engine
- Module: `src/pricing/mlsv_gpu.py`
- Inputs: Spot $S_t$, Volatility $V_t$, parameters $(\kappa, \theta, \epsilon, \rho)$, Dupire local variance function/grid.
- Outputs: Option prices, conditional expectation $\mathbb{E}[V_t \mid S_t = S]$ curves, calibrated local volatility grid.

### FX SABR Engine
- Module: `src/market/fx_data.py`, `src/calibration/fx_calibration.py`
- Inputs: Spot, domestic/foreign rates, RR/BF volatility quotes.
- Outputs: Strike-volatility grid, calibrated SABR parameters $\theta = (\alpha, \rho, \nu)$ for $\beta=1$.

### Rates LMM-SABR Engine
- Module: `src/pricing/sabr_rates.py`
- Inputs: Forward rates, SOFR swap rates, swaption market datasets.
- Outputs: Bachelier/normal SABR prices, interpolated swaption vol cube slices.

### Commodity Schwartz-Smith Engine
- Module: `src/pricing/schwartz_smith.py`
- Inputs: Spot price, futures maturities, short-term/long-term parameters $(\kappa, \mu_y, \sigma_x, \sigma_y, \rho_{xy})$.
- Outputs: Option prices, comparative pricing errors against Heston model.

## Code Layout
- Worktree 1 (feat/mlsv): `/home/execorn/programming/derivatives-w1`
- Worktree 2 (feat/fx-sabr): `/home/execorn/programming/derivatives-w2`
- Worktree 3 (feat/rates-sabr): `/home/execorn/programming/derivatives-w3`
- Worktree 4 (feat/commodity-schwartz-smith): `/home/execorn/programming/derivatives-w4`
