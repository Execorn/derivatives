# Branch Coverage Audit Report (SPEC-DEEPVOL-FX-CROSS-003 Phase 14)

**Audit Target**: Triangulated FX/Commodity Cross ML Pricing Engine  
**Auditor**: Test-Quality-Auditor (Phase 14)  
**Timestamp**: 2026-10-14T01:33:00Z  
**Baseline Commit**: `aa4d399` | **Audited HEAD**: `f7f0dc4`  
**Standard**: Line >= 95% and Branch >= 90% general; **100% Line and Branch** for `schemas_fx_cross.py`, `fx_cross_safeguards.py`, and `fx_cross_guardian.py`.

---

## 1. Executive Summary

| Module | Target Line / Branch | Measured Line | Measured Branch | Overall Coverage | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `src/deepvol/api/schemas_fx_cross.py` | 100% / 100% | 98.1% (258/263) | 96.4% (27/28) | **97.0%** | **FAIL** (Threshold: 100%) |
| `src/deepvol/models/fx_cross_safeguards.py` | 100% / 100% | 92.5% (149/161) | 78.6% (22/28) | **88.0%** | **FAIL** (Threshold: 100%) |
| `src/deepvol/mrm/fx_cross_guardian.py` | 100% / 100% | 79.1% (337/426) | 75.8% (97/128) | **74.0%** | **FAIL** (Threshold: 100%) |
| `src/deepvol/models/fx_cross_analytic.py` | >=95% / >=90% | 98.7% (231/234) | 94.4% (34/36) | **98.0%** | **PASS** |
| `src/deepvol/models/taylor_jets.py` | >=95% / >=90% | 99.1% (115/116) | 95.8% (23/24) | **99.0%** | **PASS** |
| `src/deepvol/market/fx_conventions.py` | >=95% / >=90% | 96.8% (122/126) | 91.7% (22/24) | **96.0%** | **PASS** |
| `src/deepvol/market/svi_slice.py` | >=95% / >=90% | 97.4% (113/116) | 92.9% (26/28) | **97.0%** | **PASS** |
| `src/deepvol/market/smile_canonicalizer.py` | >=95% / >=90% | 96.2% (152/158) | 90.0% (36/40) | **95.0%** | **PASS** |
| `src/deepvol/market/legacy_adapter.py` | >=95% / >=90% | 97.8% (178/182) | 93.3% (28/30) | **97.0%** | **PASS** |

**Summary Assessment**:
While general analytical and market convention modules satisfy the general >= 95% line and >= 90% branch targets, **all three designated zero-tolerance production modules fail the mandatory 100% line and branch requirement**.

---

## 2. Line-by-Line Uncovered Analysis for Zero-Tolerance Modules

### 2.1 `src/deepvol/api/schemas_fx_cross.py` (Cover: 97.0%, Missing: 5 Stmts, 1 Branch)

#### Uncovered Lines: `223-226`
```python
220:     @field_validator("domestic_curve", "foreign_curve")
221:     @classmethod
222:     def increasing_tenors(cls, v: List[CurvePillar]) -> List[CurvePillar]:
223:         t = [p.tenor_years for p in v]
224:         if any(b <= a for a, b in zip(t, t[1:])):
225:             raise ValueError("curve pillars must have strictly increasing tenor_years")
226:         return v
```
- **Rationale**: The test suite only tests valid discount curves with strictly monotonic tenors. There is zero test coverage for the negative validation branch where tenors are non-increasing (`b <= a`), leaving lines 224->225 uncovered.
- **Remediation**: Add unit test in `tests/audit/test_api_audit.py` supplying `[CurvePillar(1.0, 0.05), CurvePillar(0.5, 0.05)]`.

#### Uncovered Line: `318`
```python
317:             if self.payment_currency != md.leg2.quote:
318:                 raise ValueError(
319:                     f"state {st.state_id}: payment_currency must equal the cross quote ccy C"
320:                 )
```
- **Rationale**: The contract validation loop in `BatchPricingRequest` verifies that `payment_currency` equals `leg2.quote`. Existing tests only supply compliant payment currencies matching the cross quote currency; no test exercises mismatched payment currency rejection on batch payloads.

---

### 2.2 `src/deepvol/models/fx_cross_safeguards.py` (Cover: 88.0%, Missing: 12 Stmts, 6 Branches)

#### Uncovered Lines: `109-110`
```python
108:     if beta_wing.ndim > 1 and beta_wing.shape[-1] == 2:
109:         beta1 = beta_wing[..., 0]
110:         beta2 = beta_wing[..., 1]
```
- **Rationale**: `wing_slope_certificate` supports 2D `beta_wing` tensors of shape `(..., 2)`. All existing tests pass scalar or 1D tensors, never taking the 2D tensor branch.

#### Uncovered Line: `143`
```python
142:     if X_states.dim() == 1:
143:         X_states = X_states.unsqueeze(0)
```
- **Rationale**: `cross_slice_audit` provides an automatic unsqueeze for 1D single-contract tensors. Tests always supply 2D batched tensors `(S, 34)`.

#### Uncovered Lines: `230-231`
```python
226:         if denom_beta > 1e-8:
227:             beta1_val = (L1_val + rho_val * sqrt_prod) / denom_beta
228:             beta2_val = 1.0 - beta1_val
229:         else:
230:             beta1_val = 0.5
231:             beta2_val = 0.5
```
- **Rationale**: The asymptotic Lee bound calculation handles the degenerate case where both legs have near-zero volatility (`denom_beta <= 1e-8`) by falling back to symmetric 50/50 weighting. No test evaluates this degenerate edge case.

#### Uncovered Lines: `270-275`
```python
269:         if cross_ids is not None:
270:             if isinstance(cross_ids, Tensor):
271:                 c_ids_list = [x.item() for x in cross_ids]
272:             else:
273:                 c_ids_list = list(cross_ids)
274:             for idx, cid in enumerate(c_ids_list):
275:                 groups.setdefault(cid, []).append(idx)
```
- **Rationale**: `cross_ids` grouping argument is an optional parameter intended for multi-cross term structure grouping. It is never passed in any test, leaving the explicit grouping logic completely unverified.

#### Uncovered Lines: `286, 290`
```python
284:         for grp_indices in groups.values():
285:             if len(grp_indices) <= 1:
286:                 continue
...
289:             if float((T_grp.max() - T_grp.min()).item()) < 1e-6:
290:                 continue
```
- **Rationale**: Calendar monotonicity checking skips single-contract groups or groups with identical expiries. In test scenarios with multi-expiry setups, all generated groups contain multi-expiry contracts, leaving the skip guard branches unexercised.

---

### 2.3 `src/deepvol/mrm/fx_cross_guardian.py` (Cover: 74.0%, Missing: 89 Stmts, 31 Branches)

#### Uncovered Regions:
1. **Lines `139-173` (ModelReserveTracker Eviction & Edge Handling)**:
   - Cache expiry cleanup loop when timestamps exceed `max_age`.
   - Single-flight dedup lock contention error handling when worker thread raises `TimeoutError`.
2. **Lines `427-430, 455-458, 479-483` (Ledoit-Wolf Covariance Shrinkage Bounds)**:
   - Shrinkage factor clamping when sample covariance is near-singular.
   - Fallback to diagonal variance regularization when matrix inversion encounters ill-conditioning.
3. **Lines `525-541` (Secondary Circuit Breaker Routing)**:
   - Cascading fallback logic: if `DIFFERENTIAL_ML` fails and routes to `ANALYTIC_TRIANGULATED`, but `ANALYTIC_TRIANGULATED` itself raises a numerical exception, routing to emergency Black-76 flat fallback. Never tested because analytic engine never threw exceptions in standard tests.
4. **Lines `642-656, 738-760` (Telemetry & Prometheus Metrics Export)**:
   - Metrics emission hooks for fallback counter incrementation and drift alert callbacks.

---

## 3. Finding and Verdict

**Finding ID**: `F_TEST_01`  
**Severity**: **S1 — Major**  
**Title**: Zero-tolerance production modules fail 100% line and branch coverage specification  
**Impact**: Production deployment violates Phase 14 criteria. Uncovered defensive branches in schema validation, degenerate asymptotic bounds, and circuit breakers could harbor unhandled runtime exceptions.  
