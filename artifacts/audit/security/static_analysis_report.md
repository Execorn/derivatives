# Static Analysis Audit Report (Phase 2)

**Audited Commit**: `f7f0dc4` (tag `fx-cross-v3.0.0`, `fx-cross-m5`)
**Auditor**: API-Security-Auditor
**Date**: 2026-10-13T21:45:00Z
**Scope**: All new and edited files in `src/deepvol/`

---

## 1. Executive Summary

Static analysis and code-pattern review was conducted using:
1. `ruff check` (G4.4)
2. `mypy --strict` (G4.3)
3. `pyright --strict` (G4.3)
4. `bandit -r src/deepvol -ll` (Phase 2 & 16)
5. Comprehensive regex AST grep across Phase 2 prohibited patterns

### Results Summary
- **ruff check (new modules)**: **PASS** (0 findings across 17 new modules). Pre-existing baseline files `server.py` and `fallbacks.py` contain 8 legacy import/variable warnings present prior to FX Cross work.
- **mypy --strict**: **PASS** (0 errors across all 17 new modules with `--follow-imports=silent` per D9).
- **pyright --strict**: **PASS** (0 errors, 0 warnings across all 17 new modules).
- **bandit -r src/deepvol -ll**: **FLAGGED** (13 issues: 4 High, 9 Medium). Specifically, 5 occurrences of unsafe `torch.load` lacking `weights_only=True` in serving, MRM, and training code, plus SHA-1 usage in legacy adapter.
- **Prohibited pattern grep**:
  - `torch.load(` without `weights_only=True`: **5 occurrences detected** (Finding F001).
  - `except Exception:` swallowing errors in `_fx_cross_price_impl`: **2 occurrences detected** (Finding F004).
  - `float32` in analytic code: **0 occurrences** (all analytic layers use `torch.float64` / `np.float64`; `float32` restricted strictly to MLP surrogate and shard generation).
  - `allow_tf32`: Explicitly disabled (`allow_tf32 = False` in `train_cross_surrogate.py:526-527`).
  - `torch.func` / `jacrev` / `vmap` / `create_graph`: **0 in serving code** (only in offline DML training).
  - `.item()` / `.cpu()` in compiled regions: **0 inside compiled graphs** (only used in offline training metric logging, checkpoint serialization, or pre-processing table construction).
  - `@app.on_event`: **0 occurrences** (successfully replaced by FastAPI `lifespan` context manager in `server.py`).
  - `enforce_no_arbitrage_projection`: **0 occurrences on served paths**.
  - `TODO` / `FIXME` / `XXX` / `NotImplementedError`: **0 occurrences in served paths**.
  - Narration comments: **0 occurrences**.

---

## 2. Tool Run Records

### 2.1 ruff check (G4.4)
Command:
```bash
.venv/bin/ruff check \
  src/deepvol/api/fx_cross_arrow.py \
  src/deepvol/api/fx_cross_runtime.py \
  src/deepvol/api/schemas_fx_cross.py \
  src/deepvol/benchmarks/fx_cross_bench.py \
  src/deepvol/market/fx_conventions.py \
  src/deepvol/market/legacy_adapter.py \
  src/deepvol/market/smile_canonicalizer.py \
  src/deepvol/market/svi_slice.py \
  src/deepvol/models/fx_cross_analytic.py \
  src/deepvol/models/fx_cross_copula.py \
  src/deepvol/models/fx_cross_safeguards.py \
  src/deepvol/models/fx_cross_surrogate.py \
  src/deepvol/models/taylor_jets.py \
  src/deepvol/mrm/fx_cross_guardian.py \
  src/deepvol/training/data_generation_cross.py \
  src/deepvol/training/generate_cross_shards.py \
  src/deepvol/training/train_cross_surrogate.py
```
Output:
`All checks passed!`
Artifact: `artifacts/audit/security/ruff_check.log`

### 2.2 mypy --strict (G4.3)
Command:
```bash
.venv/bin/mypy --strict --follow-imports=silent <17 new modules>
```
Output:
`Success: no issues found in 17 source files`
Artifact: `artifacts/audit/security/mypy_strict.log`

### 2.3 pyright --strict (G4.3)
Command:
```bash
.venv/bin/pyright --level error <17 new modules>
```
Output:
`0 errors, 0 warnings, 0 informations`
Artifact: `artifacts/audit/security/pyright_strict.log`

### 2.4 bandit -r src/deepvol -ll (Phase 2 & 16)
Command:
```bash
.venv/bin/bandit -r src/deepvol -ll
```
Artifact: `artifacts/audit/security/bandit.log`
Identified 13 issues:
- `server.py:2208`: B614 (pytorch_load without weights_only=True)
- `fx_cross_guardian.py:162`: B614 (pytorch_load without weights_only=True)
- `train_cross_surrogate.py:507`: B614 (pytorch_load without weights_only=True)
- `train_cross_surrogate.py:1032`: B614 (pytorch_load without weights_only=True)
- `train_cross_surrogate.py:1040`: B614 (pytorch_load without weights_only=True)
- `legacy_adapter.py:148, 849`: B324 (hashlib.sha1 used without usedforsecurity=False)
- `server.py:2752`: B104 (hardcoded bind 0.0.0.0)
- Pre-existing legacy issues in `analysis/cross_asset_roughness.py`, `app/pages/phoenix_pricer.py`, `surrogates/train_fno.py`.

---

## 3. Pattern Review Details & Rationale

| Pattern | Spec Requirement | Auditor Observation | Disposition |
|---|---|---|---|
| `clamp/clip/where/max/min` on differentiable tensors | Smooth softplus floors & C² taper; no hard clamps zeroing gradients | `taper_jet` in `fx_cross_safeguards.py` uses analytic factored derivatives `dS_da`, `d2S_da2` assembled via `tj.unary`. `train_cross_surrogate.py` uses `torch.clamp` in `atanh` inversion guards only. | **PASS** |
| `float32` in analytic code | Strict float64 policy in analytic, H1, Black-76, Riccati, calibration | Analytic code (`fx_cross_analytic.py`, `svi_slice.py`, `legacy_adapter.py`, `fx_cross_copula.py`) is strictly `torch.float64` / `np.float64`. `float32` is strictly isolated to MLP weights/features. | **PASS** |
| `allow_tf32` | TF32 must be off for MLP | Explicitly disabled via `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`. | **PASS** |
| `torch.func` / `vmap` / `create_graph` | Forbidden in serving (§1.8) | Absent from serving code (`fx_cross_runtime.py`, `server.py`, `legacy_adapter.py`). Only used in offline training loss. | **PASS** |
| `.item()` / `.cpu()` in compiled regions | Must not trigger graph breaks (`fullgraph=True`) | No `.item()` or `.cpu()` calls inside `forward` or `forward_single` compiled methods. | **PASS** |
| `torch.load` without `weights_only=True` | Forbidden; causes arbitrary code execution risk | Found in `server.py:2208`, `fx_cross_guardian.py:162`, and `train_cross_surrogate.py:507, 1032, 1040`. | **FAIL (Finding F001)** |
| `@app.on_event` | Replaced by `lifespan` context manager | Verified `lifespan(app: FastAPI)` in `server.py:269-320`. No `@app.on_event` remaining. | **PASS** |
| `enforce_no_arbitrage_projection` | Forbidden on served paths | 0 calls on served paths. | **PASS** |
| `except: pass` / swallowed errors | Forbidden; silent failures | In `server.py:2290` and `2334`, `except Exception:` swallows canonicalization failure in single `/price`, preventing 422 `SMILE_ARBITRAGE`. | **FAIL (Finding F004)** |
| `TODO` / `FIXME` / `NotImplementedError` | Incomplete code forbidden | 0 occurrences in served modules. | **PASS** |
| Narration comments | G4.7 comment discipline | 0 occurrences. Docstrings state contract, shapes, and dtypes. | **PASS** |
