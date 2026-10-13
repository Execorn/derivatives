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

## Milestone 1: Analytic Base, Taylor Jets and Copula Teacher Dataset
- **Status**: Tagged `fx-cross-m1` at commit `6eb622c` (Unanimous Gate 3 PASS, 5/5 subagent approvals).
- **Deliverables**:
  - `src/deepvol/models/taylor_jets.py`: Hand-written 2nd-order Taylor jets ($s_k, s_{kk}$) and forward tangents with stacked GEMMs in float64/float32.
  - `src/deepvol/models/fx_cross_analytic.py`: Smile-consistent $\beta$-split H1 triangulation with 6 fixed-point iterations, OTM pricing + parity, closed-form Black Greeks, and `AnalyticTriangulatedFallbackEngine` in `fallbacks.py`.
  - `src/deepvol/models/fx_cross_copula.py`: Gaussian copula pricer under $\mathbb{Q}^B$ on 1,601-node quadrature with 1D tensor contraction and PIVOT implied vol solver.
  - `src/deepvol/training/data_generation_cross.py`: 13-D scrambled Sobol sampler with state-level train/val/test/stress holdouts and star discrepancy $\le 0.01$.
  - `src/deepvol/training/generate_cross_shards.py`: Memory-mapped streaming dataset generator with per-strike autograd tape chunking, multivariate chain-rule decomposition, online Welford statistics, and SHA-256 cryptographic manifests.
- **Measured Metrics**:
  - Unit/adversarial/SLA tests: 42/42 M1 tests + 85/85 M0 regression tests pass 100% green.
  - Peak GPU VRAM: Flat at **154.11 MB** across $M \in [1, 64]$ strikes (vs $\le 165$ MB SLA target) and **154.32 MB** during 1,601-node production generation (vs $\le 256$ MB ceiling).
  - Adjoint scaling: Reverse-mode AAD execution ratio **5.56x** forward runtime (vs $\le 10.0$x SLA limit).
  - Gradient fidelity: 117/117 coordinate differentials match decoupled numerical finite difference oracle.
  - PIVOT sensitivity: Analytical rate sensitivities match Black-Scholes to $< 10^{-14}$.
  - Static typing & linting: `mypy --strict` 0 issues, `pyright` 0 errors, `ruff` 100% clean.
- **Deviations**: None. Replaced initial finite-difference facade with authentic reverse-mode vector-Jacobian autograd and zero-copy memmapped disk streaming.
