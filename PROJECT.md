# Project: Phase 4 Model Zoo Calibration Framework

## Architecture
The framework is a production-grade pricing and calibration library using conditioning FNO neural surrogates.
- `src/pricing/`: Contains parametric SV model pricers (COS, hybrid fBm path simulator, analytical).
- `src/calibrate_fast.py`: Contains Gauss-Newton calibrators using forward-mode automatic differentiation through FNO surrogates.
- `src/app_fno.py`: Streamlit UI.
- `src/api/server.py`: FastAPI server.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Heston Training Verification | Check if task-994 completed and saved weights | None | DONE (Best val loss: 0.008695 @ epoch 149, saved to artifacts/weights/fno_heston_final_prod.pth) |
| M2 | SABR Training | Train FNO model for SABR model | M1 | DONE (Best val loss: 0.000022 @ epoch 150, saved to artifacts/weights/fno_sabr_final_prod.pth) |
| M3 | SSVI Training | Train FNO model for SSVI model | M2 | DONE (Best val loss: 0.000404 @ epoch 150, saved to artifacts/weights/fno_ssvi_final_prod.pth) |
| M3.5 | Code Audit & Real Data Validation | Run deep code audit and validate models on real SPX/VIX data | M3 | DONE (Conv: 2137a1c4, FNO SPX RMSE: 0.0334, 0 arb violations) |
| M4 | Notebook Generation & Exec | Generate & execute NB08-NB11 | M3.5 | DONE (NB08-NB11 written & executed end-to-end) |
| M5 | Test suite validation | Run pytest on unit and integration tests | M4 | DONE (556 unit/integration tests passed) |
| M6 | UI/API Integration | Verify Streamlit dashboard & FastAPI endpoint | M5 | DONE (Streamlit UI and FastAPI server validated) |

## Interface Contracts
### Model surrogate input shape
- Heston: `(B, 5)` parameters -> `[kappa, theta, sigma, rho, v0]`
- SABR: `(B, 3)` parameters -> `[alpha, rho, nu]`
- SSVI: `(B, 11)` parameters -> `[rho, eta, gamma, theta_1, ..., theta_8]`
- Rough Bergomi: `(B, 4)` parameters -> `[H, eta, rho, v0]`
- Output grid size: `(B, 8, 11)` implied volatility surface (8 maturities, 11 log-moneyness strikes).
