# Multi-Asset `EQAutocallable` CPU Setup & Execution Guide (Linux, Python 3.13, `venv`/`pip`)

This runbook configures and launches the `deepvol` Multi-Asset Equity Autocallable pricing engine (`EQAutocallable`) on a **CPU-only Linux host** (no discrete NVIDIA GPU, no `sudo`/root privileges, and no `uv`).

It assumes you are already inside the root of the cloned `derivatives` repository and that **`python3` (Python 3.13 with `venv` and `pip`)**, **`g++` (`C++23`)**, and **`cmake`** are already installed on the machine.

---

## 1. Pre-Flight Check (From Repository Root)

Verify that the required system toolchain and checked-in model artifacts are present before creating the virtual environment:

```bash
test -f pyproject.toml
test -f artifacts/checkpoints/eq_autocall_egno.pt
test -f artifacts/checkpoints/eq_autocall_guardian_stats.pt
test -f data/autocall/eq_autocallable_request.txt

python3 --version   # Must report Python 3.13.x
g++ --version       # Must support -std=c++23 (GCC >= 13)
cmake --version     # Must be >= 3.20
```

All commands below invoke `.venv/bin/...` directly so they work reliably both interactively and across stateless agent shell calls. For interactive sessions, you may optionally run `source .venv/bin/activate`.

---

## 2. Create `.venv` and Install Python 3.13 CPU Packages

`pyproject.toml` lists CUDA packages (`tensorrt`, `onnxruntime-gpu`) for GPU server deployments. On a CPU-only Python 3.13 host, install **CPU-only PyTorch**, **`cp313`-compatible binary wheels**, and then install `deepvol` in editable mode with `--no-deps`.

> **Python 3.13 Compatibility Note:** `numpy < 2.0` has no prebuilt `cp313` wheels and fails source compilation on Python 3.13. Install **`numpy>=2.1.0,<2.3.0`** together with **`scipy>=1.14.1`** and **`numba>=0.61.0`**.

```bash
# 1. Create virtual environment in .venv strictly via stdlib venv
python3 -m venv .venv

# 2. Upgrade packaging tools inside .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel

# 3. Install CPU-only PyTorch (~180 MB vs ~3.5 GB CUDA runtime)
.venv/bin/python -m pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu

# 4. Install Python 3.13 runtime, FastAPI service, Arrow IPC, and test binary wheels
.venv/bin/python -m pip install \
  "numpy>=2.1.0,<2.3.0" \
  "scipy>=1.14.1" \
  "numba>=0.61.0" \
  "pandas>=2.2.3" \
  "pydantic>=2.10.0" \
  "pyarrow>=18.0.0" \
  "scikit-learn>=1.6.0" \
  "cachetools>=5.3.0" \
  "fastapi>=0.115.0" \
  "uvicorn>=0.32.0" \
  "anyio>=4.4.0" \
  "httpx>=0.27.0" \
  "orjson>=3.10.0" \
  "prometheus-client>=0.20.0" \
  "pyyaml>=6.0.2" \
  "pytest>=8.0" \
  "pytest-asyncio>=0.24" \
  "pytest-randomly>=3.15"

# 5. Install deepvol editable without pulling GPU-only dependencies
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

---

## 3. Build the C++23 Analytical Backend (`cmake` + `g++`)

Compile the standalone C++23 analytical Stulz/Margrabe and discounting library (`src/deepvol/cpp/tier1_autocall_cpu.cpp`) and place the shared library where `EQAutocallRuntime` discovers it automatically, then remove the temporary CMake build tree to keep `git status` clean:

```bash
cmake -S src/deepvol/cpp -B build_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpp -j"$(nproc)"
install -m 755 build_cpp/tier1_autocall_cpu.so src/deepvol/models/autocall/tier1_autocall_cpu.so
rm -rf build_cpp
```

`src/deepvol/cpp/CMakeLists.txt` compiles portable `-mavx2 -mfma -mbmi2` machine code by default (`DEEPVOL_NATIVE_ARCH=OFF`). Pass `-DDEEPVOL_NATIVE_ARCH=ON` to `cmake` if you want `-march=native` tuning for the local host CPU. On the first call to `compute_eq_autocall_greeks`, `src/deepvol/models/autocall/greeks_engine.py` also invokes `g++` automatically to compile and cache its companion OpenMP C++23 Wengert-tape AD shared library into `$HOME/.cache/deepvol/tier1_ad/`.

---

## 4. Configure CPU Environment & Verify Readiness

Set writable user-space cache directories, disable CUDA probing, keep Python assertions active (`PYTHONOPTIMIZE=0`), and pin intra-op CPU threads (adjust `4`–`8` to your physical core count):

```bash
mkdir -p "$HOME/.cache/deepvol" "$HOME/.cache/numba"

export CUDA_VISIBLE_DEVICES=""
export DEEPVOL_DISABLE_CUDA=1
export DEEPVOL_TORCH_THREADS=4
export DEEPVOL_TORCH_INTEROP_THREADS=1
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
export NUMBA_CACHE_DIR="$HOME/.cache/numba"
export DEEPVOL_TIER1_CPP_LIB="$PWD/src/deepvol/models/autocall/tier1_autocall_cpu.so"
export PYTHONPATH="$PWD/src"
export PYTHONOPTIMIZE=0
```

Run the readiness check:

```bash
.venv/bin/python -c '
import sys, torch
from deepvol.api.eq_autocall_runtime import EQAutocallRuntime
from deepvol.models.autocall._tier1_backend import get_tier1_backend

rt = EQAutocallRuntime(device="cpu", n_fallback_paths=4096)
print("Python         :", sys.version.split()[0])
print("PyTorch        :", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("Device         :", rt.device)
print("Native Backend :", get_tier1_backend().name)
print("Checkpoint     :", rt.checkpoint_loaded, "| Guardian Stats:", rt.guardian_stats_loaded)
'
```

Expected output:

```text
Python         : 3.13.x
PyTorch        : 2.x.y+cpu | CUDA available: False
Device         : cpu
Native Backend : cpp23
Checkpoint     : True | Guardian Stats: True
```

`EQAutocallRuntime` initializes the SR 26-2 Guardian with `record_drift_by_default=False` and `psi_hard_circuit_breaker=False`, so one-shot evaluation batches do not accumulate false `PSI_DRIFT_BREACH` circuit-breaker trips.

---

## 5. Launch and Run `EQAutocallable`

### A. Direct Python Valuation (`PV`, `37` CashFlows, All `8` Greeks & `RepoRho`)

Instantiate `EQAutocallRuntime(device="cpu", n_fallback_paths=4096)` (or `16384` when running high-precision PV-only QMC checks on long schedules). In-domain trades run directly on the Masked EGNO surrogate in tens of milliseconds on CPU. When an out-of-distribution (`Tier3_DupireQMC`) fallback triggers:
- **PV + CashFlows only**: runs IEEE-754 `float64` Owen-scrambled Dupire QMC (`n_replicates=8`, `n_substeps_per_period=16`) at `n_fallback_paths`.
- **All-8 Greeks requested on CPU**: routes single-pass CRN QMC Greeks through a CPU-bounded budget (`n_paths <= 1024`, `n_replicates=1`, `n_substeps=2`, `lv_grid_nt=16`, `lv_grid_ny=32`) so CPU Greek sweeps complete quickly without hanging.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from deepvol.api.eq_autocall_runtime import EQAutocallRuntime
from deepvol.api.schemas_eq_autocall import parse_murex_autocall_dag

req, murex_ref = parse_murex_autocall_dag(
    Path("data/autocall/eq_autocallable_request.txt")
)

runtime = EQAutocallRuntime(device="cpu", n_fallback_paths=4096)
# Passing include_all_greeks=True appends all 8 Greek queries automatically
# (Delta, Gamma, MultivariateAggregatedCashGamma, Vega, Rho,
#  DividendsSensitivities, Theta, Cega) alongside RepoRho when RepoSpreadCurves exist.
resp = runtime.price_trade(req, engine="auto", include_all_greeks=True)

cega = resp.Cega or {}
cega_pairs = sum(len(row) for row in cega.values())
repo_keys = sorted((resp.RepoRho or {}).keys())

print(f"Engine Used        : {resp.engine_used}")
print(f"Guardian Fallback  : {resp.fallback_triggered} (reasons={resp.fallback_reasons})")
print(f"ProductValue (ML)  : {resp.ProductValue:.8f}")
print(f"ProductValue (Murex): {murex_ref['ProductValue']:.8f}")
print(f"CashFlow Rows      : {len(resp.CashFlows or [])} (DF_terminal={resp.Notional:.16f})")
print(f"VKCO_RUB Delta     : {resp.Delta['VKCO_RUB']:.8e}")
print(f"Aggregated CashGam : {resp.MultivariateAggregatedCashGamma:.8e}")
print(f"Theta              : {resp.Theta:.8e}")
print(f"Cega Pairs         : {cega_pairs} | RepoRho Keys: {repo_keys}")
print(f"Conformal Half-Wid : {resp.audit_record['conformal_half_width_bps']:.2f} bps")
PY
```

Reference CPU output on canonical trade `45476647/EMTN_CIB_SEC`:
- `Engine Used`: `Tier2_MaskedEGNO` (`Guardian Fallback: False`)
- `ProductValue (ML)`: `0.21819308` (Murex reference: `0.21819305`)
- `CashFlow Rows`: `37` (`DF_terminal = 0.9594909887002673`)
- `VKCO_RUB Delta`: `1.66943654e-03`, `Aggregated CashGam`: `3.69252225e-05`, `Theta`: `-9.10723864e-07`
- `Cega Pairs`: `25`, `RepoRho Keys`: `['MAGN_RUB', 'SIBN_RUB', 'TATN_RUB', 'VKCO_RUB']`
- `Conformal Half-Wid`: `26.76 bps`

### B. Launch & Query the FastAPI Server (`REST`, Raw Murex DAG & Apache Arrow IPC)

Start the HTTP service on an unprivileged loopback port (no root privileges required; OpenAPI docs are served at `http://127.0.0.1:8000/docs`):

```bash
.venv/bin/deepvol-api --host 127.0.0.1 --port 8000
```

From another terminal (with `.venv/bin` or system `curl`), query the `EQAutocallable` endpoints:

1. **Health & Readiness (`GET /eq_autocall/health`)**:

   ```bash
   curl -s http://127.0.0.1:8000/eq_autocall/health | .venv/bin/python -m json.tool
   ```

2. **Price a Raw Murex DAG Payload (`POST /eq_autocall/price_murex_dag`)**:

   ```bash
   curl -s -X POST "http://127.0.0.1:8000/eq_autocall/price_murex_dag?engine=auto&include_all_greeks=true" \
     -H "Content-Type: text/plain" \
     --data-binary @data/autocall/eq_autocallable_request.txt \
     | .venv/bin/python -m json.tool | head -n 40
   ```

3. **Price a Structured JSON Request (`POST /eq_autocall/price` or `/v1/price/eq_autocall`)**:

   ```bash
   .venv/bin/python -c '
   from pathlib import Path
   from deepvol.api.schemas_eq_autocall import parse_murex_autocall_dag

   req, _ = parse_murex_autocall_dag(Path("data/autocall/eq_autocallable_request.txt"))
   Path("/tmp/eq_req.json").write_text(req.model_dump_json())
   '

   curl -s -X POST "http://127.0.0.1:8000/eq_autocall/price?engine=auto&include_all_greeks=true" \
     -H "Content-Type: application/json" \
     --data-binary @/tmp/eq_req.json \
     | .venv/bin/python -m json.tool | head -n 35
   ```

4. **Batch & Apache Arrow IPC Endpoints**:
   - `POST /eq_autocall/price_batch?compute_greeks=true&engine=auto` accepts a JSON list `[req1, req2, ...]`, `{"requests": [...]}`, pre-extracted tensor dicts, or a binary Arrow IPC stream. Because `price_batch` validates the full batch atomically, it returns HTTP `422` if any single trade fails `SMILE_ARBITRAGE` validation; score unscreened Production Shadow batches per trade (Section 6) or pre-filter arbitrage-rejected deals.
   - `POST /eq_autocall/price_arrow?compute_greeks=true` (also `/v1/price/eq_autocall/batch_arrow`) accepts `Content-Type: application/vnd.apache.arrow.stream`.

### C. Run the CPU Test Suite & Latency Benchmark

Execute the full `EQAutocallable` CPU test suite (using `-m "not gpu"` so CUDA-only tests are cleanly skipped):

```bash
.venv/bin/pytest \
  tests/test_eq_autocall_schemas.py \
  tests/test_eq_autocall_curves.py \
  tests/test_eq_autocall_pre_post.py \
  tests/test_eq_autocall_analytical_proxy.py \
  tests/test_eq_autocall_egno.py \
  tests/test_eq_autocall_greeks.py \
  tests/test_eq_autocall_guardian.py \
  tests/test_eq_autocall_dupire_qmc.py \
  tests/test_eq_autocall_qmc_gpu.py \
  tests/test_eq_autocall_extreme_matrix.py \
  tests/test_eq_autocall_cache_concurrency.py \
  tests/test_eq_autocall_training.py \
  tests/test_eq_autocall_defensive_audit.py \
  tests/test_ps_batch_b10316f2_remediation.py \
  tests/test_order_statistic_pooling.py \
  tests/test_api_eq_autocall.py \
  -m "not gpu" -q
```

Run the CPU micro-benchmark (`--suite proxy` passes CPU latency thresholds and exits `0`; adding `pricing full_greeks batch` prints the full CPU table against GPU production SLA targets and exits non-zero on GPU-only latency gates):

```bash
# Tier-1 C++23 analytical proxy CPU gate check (exits 0 on CPU-only host)
.venv/bin/python benchmarks/eq_autocall_bench.py --suite proxy

# Full report across all suites (compares CPU numbers against GPU SLA thresholds)
.venv/bin/python benchmarks/eq_autocall_bench.py --suite proxy pricing full_greeks batch
```

---

## 6. Testing the Full 106-Deal Production Shadow Batch on CPU-Only

Use this workflow to validate a full converted Production Shadow batch (`batch b10316f28c2fdfa9058be6611306735e98fc4fb3`: **106 deals** → `98` priced + `8` expected `SMILE_ARBITRAGE` rejections) or any subset directory (such as a 14-deal sample bundle) on a CPU-only laptop.

### 6.1 Expected Directory Layout

```text
<batch_dir>/
  requests/         # JSON EQAutocallableTradeRequest files (e.g. trade_000_*.json ... trade_105_*.json)
  expected_murex/   # (optional) Murex reference JSONs matched by filename stem (ProductValue, StandardError)
```

### 6.2 Key Rules for CPU-Only Batch Validation

1. **Always strip embedded `Queries` in PV-only mode**: Production Shadow JSON requests carry all Greek query names (`Delta`, `Gamma`, `Vega`, `Rho`, `DividendsSensitivities`, `Theta`, `Cega`, `MultivariateAggregatedCashGamma`) inside `payload["Queries"]`. Passing `include_all_greeks=False` alone does **not** remove queries already present in `payload["Queries"]`; replace `payload["Queries"] = ["ProductValue", "CashFlows"]` explicitly when running PV + CashFlows reconciliation.
2. **Choose the right CPU mode (`pv_only` vs `all_greeks`)**:
   - `--mode pv_only` (recommended for Murex PV reconciliation): OOD trades run IEEE-754 `float64` Dupire QMC (`n_substeps_per_period=16`, `n_replicates=8`) at `--n-paths`. Use `--n-paths 4096` for fast screening across all 106 deals, or `--n-paths 16384` for tighter convergence on long-maturity OOD structures (e.g. `M=60` deals such as `53391858`).
   - `--mode all_greeks`: Passes `include_all_greeks=True` and validates that `ProductValue`, `CashFlows`, all `8` Greeks (`Delta`, `Gamma`, `MultivariateAggregatedCashGamma`, `Vega`, `Rho`, `DividendsSensitivities`, `Theta`, `Cega`), and `RepoRho` (when `RepoSpreadCurves` is present) are finite and free of `NaN`/`Inf`. On `device="cpu"`, OOD single-pass CRN QMC Greeks are internally bounded (`n_paths <= 1024`, `n_replicates=1`, `n_substeps=2`, `lv_grid_nt=16`, `lv_grid_ny=32`) so long-schedule OOD sweeps finish fast without hanging; expect higher Monte Carlo variance on `M=36..60` OOD PV values in `all_greeks` CPU mode than in `pv_only` at `--n-paths 16384`. *(For deep single-trade CPU CRN inspection at higher path counts, build `AutocallPreprocState` via `preprocess_eq_autocall(req)` and call `compute_qmc_crn_greeks(state, queries=..., config=DupireQMCConfig(...), device="cpu")` directly.)*
3. **Catch expected `SMILE_ARBITRAGE` rejections per deal**: Batch `b10316f2` includes `8` deals whose input volatility smiles violate static no-arbitrage bounds (for example, `IRAO_RUB` call-spread slope violations on `51428866` and `64649478`). `EQAutocallRuntime.price_trade` intentionally raises `ValueError: SMILE_ARBITRAGE: ...` naming the offending ticker. Catch `ValueError` per file instead of aborting the batch.
4. **Schema conversion checklist (when regenerating JSON requests from raw Murex messages)**:
   - Keep `"Cega"` as `"Cega"` inside `Queries` (never `"CorrelationSensitivities"`).
   - Route FX volatility matrices with `"TypeName": "FXVolatilitySmileCollection"` inside `MarketDataSet.VolatilitySurfaces`.
   - Legacy Murex Monte Carlo config keys (`CalculationMethod`, `NumberOfPaths`, `BatchSizeForMCPath`, `GreekMetricsSettings`, `StaticDataSet`) are stripped cleanly during parsing into `resp.stripped_legacy_fields`.

### 6.3 Self-Contained CPU Batch Test Runner (`/tmp/run_ps_batch_cpu.py`)

Create the runner script once from the repository root (make sure the Section 4 CPU environment variables are exported first):

```bash
cat <<'PY' > /tmp/run_ps_batch_cpu.py
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from deepvol.api.eq_autocall_runtime import EQAutocallRuntime


def _find_non_finite(obj: Any, prefix: str = "root") -> str | None:
    if obj is None or isinstance(obj, (bool, str, int)):
        return None
    if isinstance(obj, float):
        return None if math.isfinite(obj) else f"{prefix}={obj!r}"
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad = _find_non_finite(v, f"{prefix}.{k}")
            if bad is not None:
                return bad
        return None
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad = _find_non_finite(v, f"{prefix}[{i}]")
            if bad is not None:
                return bad
        return None
    return None


def _load_expected_murex(
    expected_dir: Path | None,
    req_path: Path,
    trade_id: str,
) -> tuple[float | None, float | None]:
    if expected_dir is None or not expected_dir.is_dir():
        return None, None
    cand = expected_dir / req_path.name
    if not cand.is_file():
        matches = sorted(expected_dir.glob(f"*_{trade_id}.json"))
        if matches:
            cand = matches[0]
        elif (expected_dir / f"{trade_id}.json").is_file():
            cand = expected_dir / f"{trade_id}.json"
        else:
            return None, None
    exp = json.loads(cand.read_text(encoding="utf-8"))
    res = exp.get("Result") if isinstance(exp.get("Result"), dict) else exp
    pv = res.get("ProductValue")
    se = res.get("StandardError")
    pv_f = float(pv) if isinstance(pv, (int, float)) and math.isfinite(float(pv)) else None
    se_f = float(se) if isinstance(se, (int, float)) and math.isfinite(float(se)) else None
    return pv_f, se_f


def _validate_greeks(resp: Any, has_repo: bool) -> list[str]:
    missing: list[str] = []
    required_attrs = (
        "Delta",
        "Gamma",
        "MultivariateAggregatedCashGamma",
        "Vega",
        "Rho",
        "DividendsSensitivities",
        "Theta",
        "Cega",
    )
    for attr in required_attrs:
        if getattr(resp, attr, None) is None:
            missing.append(attr)
    if has_repo and getattr(resp, "RepoRho", None) is None:
        missing.append("RepoRho")
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Run PS EQAutocallable batch on CPU-only.")
    ap.add_argument("--requests-dir", type=Path, required=True)
    ap.add_argument("--expected-dir", type=Path, default=None)
    ap.add_argument("--mode", choices=("pv_only", "all_greeks"), default="pv_only")
    ap.add_argument("--n-paths", type=int, default=4096)
    ap.add_argument("--engine", choices=("auto", "surrogate", "proxy", "qmc"), default="auto")
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    req_files = sorted(args.requests_dir.glob("*.json"))
    if not req_files:
        print(f"No *.json request files found in {args.requests_dir}", file=sys.stderr)
        return 2

    t_init0 = time.perf_counter()
    rt = EQAutocallRuntime(device="cpu", n_fallback_paths=int(args.n_paths))
    print(
        f"Runtime initialized in {time.perf_counter() - t_init0:.2f}s | "
        f"device={rt.device} | ckpt={rt.checkpoint_loaded} | stats={rt.guardian_stats_loaded} | "
        f"mode={args.mode} | n_fallback_paths={args.n_paths} | files={len(req_files)}"
    )

    rows: list[dict[str, Any]] = []
    engine_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    priced = 0
    arb_rejected = 0
    unexpected_errors = 0
    non_finite_count = 0
    missing_greeks_count = 0

    t_wall0 = time.perf_counter()
    for idx, req_path in enumerate(req_files, start=1):
        payload = json.loads(req_path.read_text(encoding="utf-8"))
        trade_id = str(payload.get("TradeInfo") or req_path.stem.split("_")[-1])
        has_repo = bool((payload.get("Model") or {}).get("MarketDataSet", {}).get("RepoSpreadCurves"))
        mx_pv, mx_se = _load_expected_murex(args.expected_dir, req_path, trade_id)

        if args.mode == "pv_only":
            payload["Queries"] = ["ProductValue", "CashFlows"]
            inc_all = False
        else:
            inc_all = True

        t0 = time.perf_counter()
        try:
            resp = rt.price_trade(payload, engine=args.engine, include_all_greeks=inc_all)
            dt_ms = (time.perf_counter() - t0) * 1e3
        except ValueError as exc:
            dt_ms = (time.perf_counter() - t0) * 1e3
            msg = str(exc)
            if "SMILE_ARBITRAGE" in msg:
                arb_rejected += 1
                print(f"[{idx:03d}/{len(req_files):03d}] {req_path.name} | REJECTED ({dt_ms:.1f}ms): {msg}")
                rows.append({
                    "file": req_path.name,
                    "trade_id": trade_id,
                    "status": "rejected_arbitrage",
                    "engine": "",
                    "dv_pv": "",
                    "murex_pv": "" if mx_pv is None else f"{mx_pv:.8f}",
                    "diff_bps": "",
                    "z_score": "",
                    "cf_rows": 0,
                    "latency_ms": f"{dt_ms:.2f}",
                    "reasons": msg,
                })
                continue
            unexpected_errors += 1
            print(f"[{idx:03d}/{len(req_files):03d}] {req_path.name} | ERROR ValueError: {msg}", file=sys.stderr)
            continue
        except Exception as exc:
            unexpected_errors += 1
            print(
                f"[{idx:03d}/{len(req_files):03d}] {req_path.name} | ERROR {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue

        priced += 1
        engine_counts[resp.engine_used] += 1
        for rsn in resp.fallback_reasons:
            reason_counts[str(rsn)] += 1

        dumped = resp.model_dump(mode="json")
        non_fin = _find_non_finite(dumped)
        if non_fin is not None:
            non_finite_count += 1
            print(f"  [NON-FINITE] {req_path.name}: {non_fin}", file=sys.stderr)

        cf_count = len(resp.CashFlows or [])
        if args.mode == "all_greeks":
            missing = _validate_greeks(resp, has_repo)
            if missing:
                missing_greeks_count += 1
                print(f"  [MISSING GREEKS] {req_path.name}: {missing}", file=sys.stderr)

        diff_bps: float | None = None
        z_score: float | None = None
        if mx_pv is not None:
            diff_bps = (float(resp.ProductValue) - mx_pv) * 1e4
            if mx_se is not None and mx_se > 0.0:
                z_score = abs(float(resp.ProductValue) - mx_pv) / mx_se

        diff_str = f"{diff_bps:+.2f}bps" if diff_bps is not None else "n/a"
        mx_str = f"{mx_pv:.6f}" if mx_pv is not None else "n/a"
        print(
            f"[{idx:03d}/{len(req_files):03d}] {req_path.name} | {resp.engine_used:18s} | "
            f"PV={resp.ProductValue:.6f} vs {mx_str} ({diff_str}) | cf={cf_count:3d} | "
            f"{dt_ms:6.1f}ms"
        )
        rows.append({
            "file": req_path.name,
            "trade_id": trade_id,
            "status": "priced",
            "engine": resp.engine_used,
            "dv_pv": f"{resp.ProductValue:.8f}",
            "murex_pv": "" if mx_pv is None else f"{mx_pv:.8f}",
            "diff_bps": "" if diff_bps is None else f"{diff_bps:+.2f}",
            "z_score": "" if z_score is None else f"{z_score:.2f}",
            "cf_rows": cf_count,
            "latency_ms": f"{dt_ms:.2f}",
            "reasons": ";".join(str(r) for r in resp.fallback_reasons),
        })

    wall_s = time.perf_counter() - t_wall0
    priced_rows = [r for r in rows if r["status"] == "priced"]
    lats = sorted(float(r["latency_ms"]) for r in priced_rows)
    abs_diffs = sorted(abs(float(r["diff_bps"])) for r in priced_rows if r["diff_bps"] != "")
    z_vals = sorted(float(r["z_score"]) for r in priced_rows if r["z_score"] != "")

    print("\n=== SUMMARY ===")
    print(
        f"Total={len(req_files)} | Priced={priced} | Rejected(SMILE_ARBITRAGE)={arb_rejected} | "
        f"UnexpectedErrors={unexpected_errors} | NonFinite={non_finite_count} | "
        f"MissingGreeks={missing_greeks_count} | Wall={wall_s:.2f}s"
    )
    print(f"Engines          : {dict(engine_counts)}")
    print(f"Fallback Reasons : {dict(reason_counts)}")
    if lats:
        p95_lat = lats[min(len(lats) - 1, int(0.95 * (len(lats) - 1)))]
        print(f"Latency (ms)     : p50={statistics.median(lats):.1f} | p95={p95_lat:.1f} | max={lats[-1]:.1f}")
    if abs_diffs:
        p95_diff = abs_diffs[min(len(abs_diffs) - 1, int(0.95 * (len(abs_diffs) - 1)))]
        print(f"|PV Diff| (bps)  : median={statistics.median(abs_diffs):.2f} | p95={p95_diff:.2f} | max={abs_diffs[-1]:.2f}")
    if z_vals:
        print(f"|PV Diff| / SE   : median={statistics.median(z_vals):.2f} | max={z_vals[-1]:.2f}")

    ranked = [r for r in priced_rows if r["diff_bps"] != ""]
    ranked.sort(key=lambda r: abs(float(r["diff_bps"])), reverse=True)
    if ranked:
        print("\nTop Largest PV Deviations vs Murex:")
        for r in ranked[:10]:
            print(f"  {r['file']:24s} diff={r['diff_bps']:>9s} bps | z={r['z_score']:>6s} | {r['engine']} | reasons={r['reasons']}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else [
            "file", "trade_id", "status", "engine", "dv_pv", "murex_pv",
            "diff_bps", "z_score", "cf_rows", "latency_ms", "reasons",
        ]
        with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved CSV report to {args.out_csv}")

    return 0 if (unexpected_errors == 0 and non_finite_count == 0 and missing_greeks_count == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
PY
```

### 6.4 Run Commands & Pass Criteria

Point `--requests-dir` and `--expected-dir` at your extracted 106-deal Production Shadow directory (or at a 14-deal sample bundle such as `ps_audit_b10316f2/deals/full_ps_requests` and `ps_audit_b10316f2/deals/expected_murex`):

```bash
# 1. Fast CPU PV + CashFlows screening (~1-3 min across all 106 deals)
.venv/bin/python /tmp/run_ps_batch_cpu.py \
  --requests-dir <batch_dir>/requests \
  --expected-dir <batch_dir>/expected_murex \
  --mode pv_only \
  --n-paths 4096 \
  --out-csv /tmp/ps_106_pv4096_cpu.csv

# 2. High-precision CPU PV + CashFlows reconciliation (tight convergence on M=36..60 OOD deals)
.venv/bin/python /tmp/run_ps_batch_cpu.py \
  --requests-dir <batch_dir>/requests \
  --expected-dir <batch_dir>/expected_murex \
  --mode pv_only \
  --n-paths 16384 \
  --out-csv /tmp/ps_106_pv16384_cpu.csv

# 3. Full All-8 Greeks + RepoRho structural & finiteness sweep across all 106 deals
.venv/bin/python /tmp/run_ps_batch_cpu.py \
  --requests-dir <batch_dir>/requests \
  --expected-dir <batch_dir>/expected_murex \
  --mode all_greeks \
  --n-paths 4096 \
  --out-csv /tmp/ps_106_allgreeks_cpu.csv
```

Expected pass criteria on the full 106-deal Production Shadow batch (`b10316f28c2fdfa9058be6611306735e98fc4fb3`):
- `Priced = 98` (exit code `0`) and `Rejected(SMILE_ARBITRAGE) = 8`.
- `UnexpectedErrors = 0`, `NonFinite = 0`, `MissingGreeks = 0`.
- `PSI_DRIFT_BREACH = 0` inside `Fallback Reasons`.
- All `8` Greek fields plus `RepoRho` (when `RepoSpreadCurves` is provided) populated and finite across all `98` priced trades in `--mode all_greeks`.
