# Project: Phase 4-6 Stochastic Volatility & Deep Hedging Pipeline Optimization

## Architecture
- `src/pricing/`: Parametric models (Classic Heston, SABR, SSVI, Local Volatility, Rough Bergomi, Neural SDE, Signature Vol).
- `src/hedging/`: Deep Hedging recurrent policy and environments.
- `src/calibrate_fast.py`: Gauss-Newton calibration leveraging FNO surrogates.
- `tests/`: 582 unit and integration tests for validation.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Bottleneck Profiling | Profile and identify bottlenecks in FNO Zoo (P4), Neural SDE/Signature Vol (P5), and Deep Hedging/GAN (P6) pipelines. | None | DONE |
| M2 | FNO Zoo Optimization | Vectorize and GPU-accelerate FNO Zoo data generation & FNO training (Fourier-COS, fBm path simulation). | M1 | DONE |
| M3 | Neural SDE & SigVol Optimization | Optimize on-the-fly path integration, signature extraction, eliminate host-device copies. | M2 | DONE |
| M4 | Deep Hedging & GAN Optimization | Optimize LSTM policy rollouts, state precomputation, minimax training, double-backward. | M3 | DONE |
| M5 | End-to-End Verification & Benchmarking | Run full validation, regression tests, and benchmarking to confirm speedups and zero regressions. | M4 | DONE |

## Interface Contracts
- Ensure existing public APIs in `src/pricing/`, `src/hedging/`, `src/calibrate_fast.py` remain fully backwards compatible.
- All optimizations must preserve output shapes and float/double value ranges.
