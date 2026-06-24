# Project: Phase 8 (P8) Production Framework

## Architecture
Transitioning the pricing and calibration research codebase into an enterprise-grade package named `deepvol`.
- `deepvol/`: Root package containing:
  - `models/`: Option pricing engines (Heston, SABR, rBergomi, NeuralSDE, MLSV, Schwartz-Smith)
  - `surrogates/`: FNO surrogate architectures & normalizers
  - `calibration/`: Fast calibrator implementations (LM, Gauss-Newton, Joint SPX+VIX)
  - `market/`: Data feeds (yfinance, Deribit WS, Bloomberg/SOFR)
  - `hedging/`: Deep hedging environments and policy networks
- `deepvol/app/`: Streamlit dashboard v2
- `deepvol/api/`: REST API v2 (FastAPI)
- `deploy/`: Cloud configs (Kubernetes manifests + Docker Compose stack)

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|-------------|--------|
| M1 | Python Package & CLI | Set up pyproject.toml, restructure codebase to `deepvol/`, expose clean developer APIs and CLI. | None | DONE (4c4d3009) |
| M2 | Streamlit Dashboard v2 | Implement multi-model selector, CSV/Excel surface uploader, Plotly 3D charts, parameter trajectories, PDF reports. | M1 | DONE (625310a0) |
| M3 | REST API v2 & Docker | Expand FastAPI routes to all models + Greeks, async training trigger, multi-stage CUDA runtime Dockerfile. | M1 | DONE (9fc43d6a) |
| M4 | Cloud Deployment | Design Kafka + Redis + DB + API docker-compose, write Kubernetes manifests for autoscaling GPU pods. | M3 | DONE (317e70f7) |
| M5 | Integration & Validation | Sequentially merge all branches, run tests (regression, latency, memory), perform Forensic Audit. | M1, M2, M3, M4 | IN_PROGRESS (3aa66a0b) |

## Interface Contracts
### Public Developer API
- `deepvol.calibrate(market_iv_surface, model_name, method, device)` -> `CalibrationResult`
- `deepvol.compute_greeks(model_name, parameters, spot, strikes, maturities)` -> `dict`

### REST API v2 Endpoints
- `GET /health` -> `{status: "ok"}`
- `GET /models` -> list of available models and accuracy stats
- `POST /calibrate/{model_name}` -> JSON option grid input -> calibrated params, RMSE, elapsed time
- `POST /greeks/{model_name}` -> model parameters + spot/grid input -> calculated Greeks
- `POST /train` -> trigger asynchronous FNO surrogate training -> `{job_id}`
- `GET /jobs/{job_id}` -> status of training run

## Code Layout
- Worktree 1 (feat/p8-package): `/home/execorn/programming/derivatives-w1`
- Worktree 2 (feat/p8-dashboard): `/home/execorn/programming/derivatives-w2`
- Worktree 3 (feat/p8-api): `/home/execorn/programming/derivatives-w3`
- Worktree 4 (feat/p8-cloud): `/home/execorn/programming/derivatives-w4`
