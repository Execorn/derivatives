---
name: verifying-symbolic-sdes
description: >-
  Three-stage formal and symbolic verification of quantitative finance models: local Wolfram Engine 15.0
  for infinitesimal generators and Feynman-Kac PDEs, AST-guarded sympy-sandbox for Buckingham Pi
  dimensional analysis and Taylor asymptotics, and Lean 4 (lean-lsp-mcp) for formal proof of martingality and no-arbitrage bounds with zero sorryAx.
---

# Formal & Symbolic SDE Verification (`symbolic-sde-verifier`)

This skill enforces regulatory and mathematical rigor before any SDE, PDE, or financial model is implemented in `deepvol`. It implements a mandatory **Three-Stage Symbolic Verification Pipeline**:

```
[Candidate Model SDE]
        │
        ▼
[Stage A: Wolfram Engine 15.0]  --> Infinitesimal Generator L & Feynman-Kac Backward PDE
        │
        ▼
[Stage B: SymPy AST Sandbox]   --> Buckingham Pi Dimensional Analysis & Taylor Asymptotics
        │
        ▼
[Stage C: Lean 4 Formal Prover] --> Absence of Arbitrage & Martingality Proofs (No sorryAx)
        │
        ▼
[Certified Mathematical Spec for PyTorch / Triton / JAX Engine]
```

---

## Stage A: Local Wolfram Engine 15.0 (`WolframLanguageEvaluator`)

Use the local Wolfram Engine 15.0 to apply multidimensional Itô's Lemma and derive backward Kolmogorov partial differential equations.

### 1. Infinitesimal Generator Derivation
For an $n$-dimensional semimartingale $dX_t = \mu(t, X_t)dt + \Sigma(t, X_t)dW_t^{\mathbb{Q}}$ with covariance matrix $A = \Sigma \Sigma^\top$, the infinitesimal generator $\mathcal{L}$ acting on a test function $V(t, x)$ is:
$$\mathcal{L}V(t, x) = \sum_{i=1}^n \mu_i(t, x) \frac{\partial V}{\partial x_i} + \frac{1}{2} \sum_{i,j=1}^n A_{ij}(t, x) \frac{\partial^2 V}{\partial x_i \partial x_j}$$

### 2. Feynman-Kac Backward PDE
Verify that the discounted option price $V(t, x) = \mathbb{E}^{\mathbb{Q}}\left[e^{-\int_t^T r_s ds} \Phi(X_T) \;\middle\vert\; X_t = x\right]$ solves:
$$\frac{\partial V}{\partial t} + \mathcal{L}V - r(t, x)V = 0, \qquad V(T, x) = \Phi(x)$$

---

## Stage B: AST-Guarded SymPy Sandbox (`scripts/sympy_sandbox_mcp.py`)

Invoked via the local FastMCP server or `sym-mcp`. Guarantees AST safety against arbitrary code execution while automating dimensional and asymptotic consistency checks.

### 1. Buckingham $\Pi$ Dimensional Analysis
Verify that all model quantities obey physical units and scaling laws:
* Time $[T]$: $t, T, \Delta t$.
* Frequency / Rate $[T^{-1}]$: Risk-free rate $r$, dividend yield $q$, variance $v_t$, mean reversion speed $\kappa$.
* Volatility $[T^{-1/2}]$: Volatility of asset $\sigma$.
* Dimensionless $[1]$: Hurst parameter $H \in (0, 1/2)$, log-moneyness $k = \ln(K/F)$, correlation $\rho \in [-1, 1]$.
* Brownian increment: $dW_t \sim [T^{1/2}]$.

### 2. Taylor Asymptotics around At-The-Money ($k \to 0$)
Extract small-time and near-the-money asymptotic expansions of implied volatility:
$$\sigma_{\text{impl}}(k, T) = \sigma_0(T) + \left.\frac{\partial \sigma_{\text{impl}}}{\partial k}\right|_{k=0} k + \frac{1}{2}\left.\frac{\partial^2 \sigma_{\text{impl}}}{\partial k^2}\right|_{k=0} k^2 + \mathcal{O}(k^3)$$
Verify that skew slope $\left.\partial_k \sigma_{\text{impl}}\right|_{k=0} \approx \frac{\rho \eta}{2} T^{H-1/2}$ scales appropriately with Hurst index $H$.

### 3. Vectorized PyTorch Kernel Export
Use `export_pytorch_expr` from `scripts/sympy_sandbox_mcp.py` to translate verified analytical formulas directly into PyTorch tensor evaluators with zero manual transcription error.

---

## Stage C: Lean 4 Formal Prover (`lean-lsp-mcp`)

Before any new volatility surface parametrization or measure transformation enters production, key theorems must be verified in Lean 4 atop Mathlib4.

### 1. Verification Directives
* Invoke `mcp/lean_run_code` and `mcp/lean_goal` to check step-by-step proofs.
* **Absence of Calendar Arbitrage**: Prove total variance monotonicity $\forall k, \; T_1 \le T_2 \implies w(k, T_1) \le w(k, T_2)$.
* **Absence of Butterfly Arbitrage (Durrleman Condition)**:
  $$g(k) = \left(1 - \frac{k w'(k)}{2w(k)}\right)^{\!2} - \frac{(w'(k))^2}{4}\left(\frac{1}{w(k)} + \frac{1}{4}\right) + \frac{w''(k)}{2} \ge 0 \quad \forall k \in \mathbb{R}$$
* **Martingale Drift Invariant**: Verify that the discounted underlying price process $\widetilde{S}_t = e^{-\int_0^t (r_u - q_u)du} S_t$ satisfies the $\mathbb{Q}$-martingale condition:
  $$\mathbb{E}^{\mathbb{Q}}[\widetilde{S}_T \mid \mathcal{F}_t] = \widetilde{S}_t \quad \text{a.s.}$$

### 2. Zero-sorryAx Regulatory Rule
Always run `#print axioms theorem_name`.
If the output contains `sorryAx`, the proof is rejected as incomplete and cannot be used for production model certification.
