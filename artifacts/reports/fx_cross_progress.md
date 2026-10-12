# FX Cross Progress Report

## Milestone 0: Market Conventions and Smile Canonicalization
- **Status**: Tagged `fx-cross-m0` at commit `7eb01de` (Unanimous Gate 3 PASS, 5/5 subagent approvals).
- **Deliverables**:
  - `src/deepvol/market/fx_conventions.py`: Delta conventions (Spot/Forward PNA & PA), bracketed Newton strike solver with adaptive low-$\tau$ machine epsilon scaling and $n_{\text{iter}}=20$ high-vol convergence.
  - `src/deepvol/market/svi_slice.py`: Raw-SVI 5-point slice fitting with fully batched tensor Levenberg-Marquardt optimizer and 401-point Durrleman butterfly certification ($g(x) \ge 0$).
  - `src/deepvol/market/smile_canonicalizer.py`: SSVI fallback, canonical feature extraction $\boldsymbol\xi_i$, exact analytical IFT reverse-mode solve ($R_{\boldsymbol\theta}^\top \lambda = \bar{\boldsymbol\theta}$ via `torch.linalg.solve`), blake2b state caching, and calendar arbitrage checks.
- **Measured Metrics**:
  - Unit/adversarial/stress tests: 171/171 tests pass green in 135s (66 baseline maintained + 105 new tests).
  - Throughput: 64 distinct unmemoized market states canonicalize in **0.671 ms/state** (vs $\le 2.5$ ms SLA budget).
  - Analytical IFT accuracy: Adjoint error $2.87 \times 10^{-7}$ relative error vs central bumps.
  - SVI solver accuracy: Fit RMSE $3.79 \times 10^{-14}$ on exact slices.
  - Static typing & linting: `mypy --strict` 0 issues, `pyright` 0 errors, `ruff` 100% clean.
- **Deviations**: None. All 5 findings from Gate 1 and 2 remediated under strict TDD and formal Wolfram proofs (W0.1–W0.3).
