---
name: auditing-model-risk-sr26-2
description: >-
  Institutional Model Risk Management (MRM) auditing under Federal Reserve SR 26-2 and SR 11-7 standards:
  dual-layer AI governance, Ledoit-Wolf correlation shrinkage, Population Stability Index (PSI) drift tracking,
  Mahalanobis OOD detection, and automated deterministic fallback circuit breakers for deepvol neural surrogates.
---

# Model Risk Management & Regulatory Governance under Federal Reserve SR 26-2 (`mrm-auditor`)

This skill defines the governance and auditing protocol under the Federal Reserve, OCC, and FDIC **SR 26-2 (Revised Guidance on Model Risk Management)** issued on April 17, 2026, which supersedes the legacy SR 11-7 framework.

---

## 1. Regulatory Transition: SR 11-7 vs. SR 26-2

| Dimension | Legacy SR 11-7 (2011) | Modernized SR 26-2 (2026) |
| :--- | :--- | :--- |
| **Philosophy** | Rigid annual re-validation calendar checklist. | **Principles-based Materiality Framework**: Cadence and depth driven by financial exposure and decision impact. |
| **Generative / Agentic AI** | Unaddressed (treated as generic automation or models). | **Explicitly carved out of quantitative model definition** into a dedicated AI governance track; subject to interim controls. |
| **Quantitative Surrogates** | Narrow traditional statistical models. | **Directly in scope**: Neural operators (FNO, EGNO), Signature SDEs, and particle calibrators producing valuations or Greeks. |
| **Data Quality & Drift** | General IT input quality audit. | **Elevated to primary driver of inherent risk**: Demands continuous production input drift tracking against training data. |

---

## 2. Dual-Layer AI Governance Architecture in Antigravity

To satisfy SR 26-2 compliance inside `deepvol` and Sber CIB PricingService:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Layer 1: Agentic Research Layer                 │
│              (Google Antigravity Subagents: Cmd+E Manager Surface)     │
├────────────────────────────────────────────────────────────────────────┤
│ • Prohibited from directly pushing unverified pricing to production.   │
│ • SDE/PDE derivations gate-checked by local Wolfram Engine 15.0.       │
│ • Proofs certified in Lean 4 (lean-lsp-mcp) with ZERO sorryAx axioms.  │
│ • Code execution guarded by AST whitelists in sympy-sandbox.           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Outputs certified code & models
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Layer 2: Quantitative Pricing Layer             │
│             (deepvol Neural Operators: FNO, EGNO, Signature SDE)       │
├────────────────────────────────────────────────────────────────────────┤
│ • Continuous real-time input drift tracking via PSI.                   │
│ • Out-of-Distribution (OOD) guardian via Mahalanobis distance D_M.     │
│ • Static & Dynamic no-arbitrage projection layers (Durrleman, calendar)│
│ • Sub-millisecond fallback circuit breaker to fp64 Monte Carlo / PDE.  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Circuit Breakers in PricingService

Every request served through `deepvol` must pass the following four deterministic filters:

### A. Ledoit–Wolf Optimal Linear Shrinkage
For empirical correlation matrices $\hat{\boldsymbol{\Sigma}}$ estimated from asset returns $\mathbf{r}_t \in \mathbb{R}^d$:
$$\boldsymbol{\Sigma}_{\text{LW}} = (1 - \alpha^*) \hat{\boldsymbol{\Sigma}} + \alpha^* \mathbf{I}_d$$
$$\alpha^* = \min\left(1, \; \frac{\frac{1}{T_{\text{obs}}^2} \sum_{t=1}^{T_{\text{obs}}} \|\mathbf{r}_t \mathbf{r}_t^\top - \hat{\boldsymbol{\Sigma}}\|_F^2}{\|\hat{\boldsymbol{\Sigma}} - \mathbf{I}_d\|_F^2}\right)$$
Guarantees strict positive definiteness: $\lambda_{\min}(\boldsymbol{\Sigma}_{\text{LW}}) \ge \alpha^* > 0$.

### B. Population Stability Index (PSI) Drift Monitoring
Monitors distribution shift across 10 decile buckets:
$$\text{PSI}_j = \sum_{b=1}^{10} \left(p_{j,b}^{\text{live}} - q_{j,b}^{\text{train}}\right) \ln\left(\frac{p_{j,b}^{\text{live}} + 10^{-8}}{q_{j,b}^{\text{train}} + 10^{-8}}\right)$$
* $\text{PSI} < 0.10$: Green (In-Distribution).
* $0.10 \le \text{PSI} < 0.25$: Yellow (Moderate drift; log warning).
* $\text{PSI} \ge 0.25$: Red (Critical drift; activate automatic fallback).

### C. Multivariate Mahalanobis OOD Distance
In normalized feature / signature space:
$$D_M^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu}_{\text{train}})^\top \boldsymbol{\Sigma}_{\text{train}}^{-1} (\mathbf{x} - \boldsymbol{\mu}_{\text{train}})$$
If $D_M^2(\mathbf{x}) > \chi^2_{p, 0.999}$, flag input as Out-of-Distribution.

### D. No-Arbitrage Verification (Durrleman Condition)
The predicted implied variance surface $w(k, T)$ must satisfy:
$$\min_{k, T} g(k, T) \ge -10^{-6}$$
$$\min_{k, T} \partial_T w(k, T) \ge -10^{-6}$$

---

## 4. Deterministic Fallback Circuit Breaker Protocol

Whenever any filter fails:
```python
if d_m > tau_ood or psi_max >= 0.25 or min_durrleman < -1e-6:
    logger.warning(
        "[SR26-2 CIRCUIT BREAKER] Trade %s triggered failover: DM=%.3f, PSI=%.3f, g=%.6f",
        trade_id, d_m, psi_max, min_durrleman
    )
    # Synchronously route to pre-captured CUDAGraph fp64 Monte Carlo or PDE solver
    result = fallback_engine.price(trade_payload)
    result["circuit_breaker_active"] = True
    result["fallback_engine"] = "CUDAGraph_fp64_MC"
    return result
```
Execution latency of fallback circuit breaker is strictly $< 1\text{ ms}$.
