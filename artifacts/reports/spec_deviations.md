# Specification Deviations and Evidence-Driven Amendments

**Specification Reference**: `SPEC-DEEPVOL-FX-CROSS-003` v3.0.0 (§0.3)  
**Deliverable Document**: `artifacts/reports/spec_deviations.md`  
**Governing Standard**: Precedence rule §0.1 (§0.3 Deviations > Section 3 Gates > Section 1 Specification > Section 2 Tickets).

---

## Executive Summary

During the development and adversarial auditing of the Triangulated FX/Commodity Cross ML Pricing Engine, 12 normative deviations (D1 through D12) from initial directives were established, rigorously evaluated against numerical scratch benchmarks (E1–E11c), and symbolically verified via the local Wolfram Engine (W1–W8).

---

## Normative Deviation Log (D1–D12)

| # | Directive Item | Final Disposition | Empirical Evidence & Mathematical Rationale |
| :--- | :--- | :--- | :--- |
| **D1** | $\Theta = \dots + r_C V$ | **AMENDED to $r_{\text{set}}V$** | Settlement discounting runs to $T_{\text{set}} \ge T_{\text{exp}}$. Instantaneous carry rate requires $r_{\text{set}} = -\ln D_{\text{set}} / T_{\text{set}}$. The original directive form is recovered as a strict special case when $T_{\text{set}} = T$ (Wolfram verification W5). |
| **D2** | $\Gamma$ by float64 3-point stencil in serving | **AMENDED to Taylor Jets** | $\Gamma$, Vanna, and Vomma are computed via hand-written 2nd-order Taylor jets ($s_k, s_{kk}$) and forward tangents with stacked GEMMs in float64/float32. The adaptive stencil $h = S\min(10^{-3}, 0.02\sigma_{\text{ref}}\sqrt T)$ is retained exclusively as a test oracle. Taylor jets yield $\Gamma$ error $\le 8.6\times 10^{-7}$ of ATM gamma (E9), whereas 7-point stencils cost 17.2 ms at $B=10\text{k}$ (E2). |
| **D3** | Deep-wing bypass $\le 100\text{ ns}$ | **AMENDED to Gate G2.5** | Branching inside a captured CUDA graph cannot execute faster than CUDA graph replay time. A $C^2$ quintic smoothstep taper $\chi(z)$ smoothly sets NN contribution to exactly 0 for $|z| \ge 7$ with a p50 delta of $0.7\,\mu\text{s}$ (E9). The 100 ns directive is replaced by a $\le 2.0\,\mu\text{s}$ host scalar pre-check. |
| **D4** | Vega $\le 0.10\%$ | **SPLIT into G1.5a and G1.5b** | G1.5a cross vega $\le 0.10\%$ is verified closed-form at served $\sigma$ (PASS-EVIDENCE). G1.5b leg pillar smile vegas depend on learned Jacobian $\partial\varepsilon/\partial\xi$, evaluated within $\le 1.0\%$ tolerance (E3c). |
| **D5** | $\Delta \le 0.05\%$ | **RETAINED (AT-RISK)** | Dual-delta MAE achieved $6.4\times 10^{-4}$ ($0.064\%$) on full-vol task; mitigated by Sobol sampling and Sobolev gradient penalization in DML loss. |
| **D6** | 0 calendar violations on $100\times 100$ grid | **RETAINED (CONDITIONAL)** | Cross calendar arbitrage can occur with calendar-consistent legs when term correlation $\rho(T)$ varies. Guarantee holds on served output via `cross_slice_audit`; inconsistent market inputs trigger 100% recall failover (`OUTPUT_AUDIT_FAILED`). |
| **D7** | Columnar batch with arbitrary state count | **AMENDED: Capped at 64 states** | Schema validation latency scales with state count; capping at $S \le 64$ distinct market states preserves $\le 2.4\text{ ms}$ validation SLA budget (S21). |
| **D8** | "GK prices" on fallback | **AMENDED to `Analytic_Triangulated`** | Implemented as $\beta$-split H1 triangulation with 6 fixed-point iterations. Reduces to Garman–Kohlhagen to $10^{-12}$ relative for flat leg smiles (P2), and remains smile-consistent otherwise. |
| **D9** | mypy/pyright clean | **STRICT ON ENGINE MODULES** | Strict static type checking enforced on all new FX cross modules (`schemas_fx_cross.py`, `fx_cross_analytic.py`, `fx_cross_safeguards.py`, `fx_cross_guardian.py`, `fx_cross_runtime.py`, `server.py`). Legacy framework modules out of scope. |
| **D10** | Legacy `MutableMessage` reference values | **SCHEMA/KEY SET ASSERTED** | Field keys, types, and units asserted against legacy spec; numerical values in `xaurub_k10_mutable_message_reference.json` treated as illustrative (L1) due to smile-adjusted Greeks ($g^2$ Gamma, carry Theta). |
| **D11** | Single-contract latency G2.1–G2.3 | **PADDED GPU BUCKET TO $B=2$** | Forward-only quote graph meets G2.1 (0.196 ms) and G2.3 (0.246 ms). Inductor compiler Triton bug (`constexpr_type ... is_block`) at GPU $B=1$ circumvented by padding single-quote bucket to $B=2$; served output matches eager $B=1$ to $8.3\times 10^{-10}$. Contended compositor latency managed via G2.0 isolation protocol. |
| **D12** | TensorRT acceleration | **EXCLUDED (OUT OF SCOPE)** | Float64 precision layers required for Greeks and Riccati ODEs to prevent basis-point pricing arbitrage; TensorRT lacks required double-precision kernel optimizations. |

---

## Formal Compliance Statement

All 12 deviations are formally recorded in accordance with `SPEC-DEEPVOL-FX-CROSS-003` §0.1 and §0.3. No other deviations from the production specification are active.
