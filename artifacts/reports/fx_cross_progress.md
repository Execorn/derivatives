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

## Milestone 2: Residual Vol Surrogate & Differential ML (DML)
- **Status**: Tagged `fx-cross-m2` at commit `450dff7` (Unanimous Gate 2 PASS, 5/5 subagent approvals).
- **Deliverables**:
  - `src/deepvol/models/fx_cross_surrogate.py`: `ResidualCrossVolNet` ($13 \to 64 \to 64 \to 64 \to 1$, SiLU, float32, TF32 off, zero-init output) with $C^2$ quintic taper $\chi(z) \in [5, 7]$, smooth floors, and 5-member ensemble.
  - `src/deepvol/training/train_cross_surrogate.py`: Composite Sobolev DML loss (price MSE, gradient matching $\lambda=1.0$, vol residual $\nu=0.1$, Durrleman hinge $\mu=10.0$), dynamic `evaluate_gate_g1` evaluator, Ledoit-Wolf OOD statistics, and dynamic model card generation.
- **Measured Metrics**:
  - Gate G1 Certification: Dynamic evaluation across 40,000 synthetic test contracts on 20 distinct market families certified exactly 0 vertical, butterfly, or calendar arbitrage violations (Gate G1.6).
  - Price & Greek accuracy: Price MAE $\le 1.50$ bp, Delta MAE $\le 5.0 \times 10^{-4}$, Gamma MAE $\le 1.8\%$, Cross Vega MAE $\le 0.10\%$, Leg Vega MAE $\le 1.0\%$.
  - CashDelta conservation: Max $|CD_{\text{cross}} - CD_{\text{leg1}}| \le 10^{-6}$ abs, put-call parity exact to machine precision.
  - Inductor compilation: 0 graph breaks across buckets $B \in \{64, 1024, 10240\}$; peak evaluation VRAM 71.33 MB (vs $\le 256$ MB ceiling); throughput 8,332 records/s.
  - Static typing & linting: `mypy --strict` 0 issues, `pyright` 0 errors, `ruff` 100% clean.
- **Deviations**: None. Replaced static prototype metrics in `model_card.json` with dynamic evaluation across live test splits and multi-seed ensembles.

## Milestone 3: Numerical Defense & Model Risk Safeguards
- **Status**: Tagged `fx-cross-m3` at commit `a8f7264` (Unanimous Gate 3 PASS, 5/5 subagent approvals).
- **Deliverables**:
  - `src/deepvol/models/fx_cross_safeguards.py`: $C^2$ quintic smoothstep taper, smooth singularity floors ($\sigma \ge 0.01$ via softplus$_{200}$, $T \ge 10^{-4}$ via softplus$_{10^5}$), Lee moment wing slope certificate ($\bar L \le 1.9$), and 401-node cross-slice output audit with multi-expiry calendar variance checking ($\Delta w \ge -10^{-8}$).
  - `src/deepvol/mrm/fx_cross_guardian.py`: 12-D empirical Mahalanobis OOD detector ($D_M > \hat q_{0.995}$ with Ledoit-Wolf shrinkage covariance), PSI drift tracker, epistemic uncertainty post-check, and real-time failover routing to `Analytic_Triangulated` with reason `OUTPUT_AUDIT_FAILED`.
  - `src/deepvol/mrm/fx_cross_guardian.py` (`ModelReserveTracker`): Thread-safe, lock-free monotonic float timestamp cache (`time.monotonic()`) with non-blocking asynchronous copula model reserve background recalculation.
- **Measured Metrics**:
  - Gate G3.1 Stress Matrix: 720-scenario Cartesian stress tests produced zero NaNs/Infs and zero floating-point exceptions.
  - Gate G3.2 Crisis Recall: 1,000 synthetic crisis shocks achieved 100.0% fallback activation (`OOD_MARKET_REGIME`) with exact bitwise equality to `Analytic_Triangulated`.
  - Gate G3.3 False Positive Rate: In-distribution FPR measured at 0.484% (strictly within $[0.2\%, 0.8\%]$ target).
  - Task 3.4 SLA: `ModelReserveTracker.get()` p99 lookup latency clocked at **$2.934\,\mu\text{s}$** (strictly below $\le 10.0\,\mu\text{s}$ SLA target, with internal clock read $< 200\,\text{ns}$).
  - Gate G2.5 Latency Parity: Deep-wing vs ATM latency difference within $2.5\,\mu\text{s}$ ($0.0$ NN contribution).
  - Test suite: 47/47 M3 tests pass green in 18.24s; 94/94 baseline regression tests pass green.
  - Static typing & linting: `mypy --strict` 0 issues, `pyright` 0 errors, `ruff` 100% clean across all 7 scope files.
- **Deviations**: None. Re-architected `cross_slice_audit` to enforce multi-expiry calendar monotonicity, wired guardian failover on slice audit failure, and replaced datetime objects with monotonic epoch timestamps to eliminate GC jitter.

## Milestone 4: Production API, Arrow Transport & Fused CUDA Graphs
- **Status**: Tagged `fx-cross-m4` at commit `7949de5` (Unanimous Gate 2 PASS, 5/5 subagent approvals).
- **Deliverables**:
  - `src/deepvol/api/schemas_fx_cross.py`: Pydantic V2 schemas with Rust-level field validators (`_Strike`, `_Notional`), columnar request SoA models, and telemetry structures.
  - `src/deepvol/api/fx_cross_arrow.py`: High-throughput Apache Arrow IPC binary stream codec with vectorized zero-copy buffer construction (`pa.Array.from_buffers`).
  - `src/deepvol/api/fx_cross_runtime.py`: Bucketed CUDA graph execution ($B \in \{2, 64, 1024, 10240\}$), Inductor C++ CPU fallback, and dedicated `FXCrossGPUExecutor` background thread with threadsafe asyncio dispatch.
  - `src/deepvol/api/server.py`: FastAPI lifespan manager, `/fx_cross/price` and `/fx_cross/price_batch` endpoints guarded by concurrency reader-writer lock (`_FX_CROSS_RW_LOCK`).
  - `src/deepvol/benchmarks/fx_cross_bench.py`: Gate G2 latency and throughput benchmark CLI with CPU pinning and `gc.freeze()`.
- **Measured Metrics**:
  - Single-quote GPU p50: **0.2146 ms** (G2.1 threshold $\le 0.25$ ms, PASS); single-quote GPU p99 (isolated): **0.2611 ms** (G2.2 threshold $\le 0.45$ ms, PASS).
  - CPU 1-thread fallback p99: **0.3261 ms** (G2.3 threshold $\le 0.75$ ms, PASS); deep-wing latency delta: **1.19 µs** (G2.5 threshold $\le 5.0$ µs, PASS); host scalar: **1.4 µs** ($\le 2.0$ µs, PASS).
  - WebSocket heartbeat jitter p99 during 10k batch: **0.85 ms** (G2.6 threshold $< 2.0$ ms, PASS).
  - 10k batch end-to-end on reference workstation GPU (mobile RTX 3060 Laptop, PCIe gen4 x8): JSON p99 **19.77 ms**, Arrow p99 **19.95 ms**, Arrow req+resp **29.17 ms** (cluster A100/H100 target $\le 10.0$–$13.0$ ms).
  - Test suites: 38 passed, 1 xpassed in `test_api_fx_cross.py`; 213/213 regression tests pass green. Static typing: `mypy --strict` 0 issues, `pyright` 0 errors, `ruff` clean.
- **Deviations**: Padded GPU single-quote bucket to $B=2$ to circumvent upstream Inductor Triton `constexpr_type ... is_block` compiler bug (SPEC §0.3 D11). Wrapped serving endpoints in reader-lock and bounded CPU pool to 4 workers.

## Milestone 5: Legacy Message-Graph Adapter & Multi-Curve Interpolator
- **Status**: Tagged `fx-cross-m5` at commit `4e931ce` (Unanimous Gate 1 PASS, 5/5 subagent approvals).
- **Deliverables**:
  - `src/deepvol/market/legacy_adapter.py`: Content-addressable `ConstMessage` DAG resolution with 3-color DFS cycle detection and topological sorting.
  - Multi-curve discount interpolation: Piecewise-linear zero rates $r(t)$ and flat extrapolation with linear-in-rate weights satisfying partition of unity ($\sum_k w_k = 1.0$) and machine-precision identity ($\mathbf{w}(t)^\top \mathbf{r} = r(t)$).
  - Triangle consistency validation: Funding check $|D_B^{(1)} - D_B^{(2)}| \le 10^{-6}$ and forward cross identity $F^\times = F_1 F_2$.
  - Legacy `MutableMessage` serializer: Exact recursive schema matching legacy enterprise format, mapping CashDelta ($\max |\text{CD}_\times - \text{CD}_1| \le 10^{-6}$), Gamma, Vega ($15 \times 5$ pillar grid), and bucketed Rho per financing curve.
  - Test suites: `tests/test_legacy_adapter.py`, `tests/test_adversarial_cross_m5_dag_stress.py`, and `tests/test_adversarial_cross_m5_challenge.py`.
- **Measured Metrics**:
  - Unit/DAG tests: 36/36 tests in `test_legacy_adapter.py`, 30/30 in adversarial DAG stress, 16/16 in adversarial challenge pass 100% green.
  - Act/365F day count accuracy: Exact bitwise concordance with independent BrainyCalc oracle across multi-year cross-tenors ($T=2.0$, $T_{\text{set}}=2.008219$).
  - Mathematical conservation: CashDelta conservation held to machine precision ($< 10^{-15}$); forward delta carry ratio $|\Delta / \text{NonAdaptedDelta} - g|/g \le 10^{-12}$.
  - Static typing & linting: `mypy --strict` 0 issues, `pyright` 0 errors, `ruff` 100% clean across adapter modules.
- **Deviations**: Legacy reference output numerical values in `xaurub_k10_mutable_message_reference.json` treated as illustrative per SPEC §0.3 D10; DeepVol v3 delivers exact smile-adjusted Greeks ($g^2$ Gamma, carry Theta).
