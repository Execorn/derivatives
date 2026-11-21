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

On the first call to `compute_eq_autocall_greeks`, `src/deepvol/models/autocall/greeks_engine.py` also invokes `g++` automatically to compile and cache its companion OpenMP C++23 Wengert-tape AD shared library into `$HOME/.cache/deepvol/tier1_ad/`.

---

## 4. Configure CPU Environment & Verify Readiness

Set writable user-space cache directories, disable CUDA probing, and pin intra-op CPU threads (adjust `4`–`8` to your physical core count):

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

---

## 5. Launch and Run `EQAutocallable`

### A. Direct Python Valuation (`PV`, `37` CashFlows & All `8` Greeks)

Instantiate `EQAutocallRuntime(device="cpu", n_fallback_paths=4096)` so any out-of-distribution fallback uses fast CPU Quasi-Monte Carlo (`~0.45 s`), while in-domain trades execute directly on the neural operator surrogate in milliseconds.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from deepvol.api.eq_autocall_runtime import EQAutocallRuntime
from deepvol.api.schemas_eq_autocall import parse_murex_autocall_dag

req, murex_ref = parse_murex_autocall_dag(
    Path("data/autocall/eq_autocallable_request.txt")
)

req_full = req.model_copy(
    update={
        "Queries": [
            "ProductValue",
            "Notional",
            "Coupon",
            "Option",
            "CashFlows",
            "Delta",
            "Gamma",
            "MultivariateAggregatedCashGamma",
            "Vega",
            "Rho",
            "DividendsSensitivities",
            "Theta",
            "Cega",
        ]
    }
)

runtime = EQAutocallRuntime(device="cpu", n_fallback_paths=4096)
resp = runtime.price_trade(req_full, engine="auto")

print(f"Engine Used        : {resp.engine_used}")
print(f"Guardian Fallback  : {resp.fallback_triggered} (reasons={resp.fallback_reasons})")
print(f"ProductValue (ML)  : {resp.ProductValue:.8f}")
print(f"ProductValue (Murex): {murex_ref['ProductValue']:.8f}")
print(f"CashFlow Rows      : {len(resp.CashFlows or [])} (DF_terminal={resp.Notional:.16f})")
print(f"VKCO_RUB Delta     : {resp.Delta['VKCO_RUB']:.8e}")
print(f"Aggregated CashGam : {resp.MultivariateAggregatedCashGamma:.8e}")
print(f"Theta              : {resp.Theta:.8e}")
print(f"Conformal Half-Wid : {resp.audit_record['conformal_half_width_bps']:.2f} bps")
PY
```

Reference CPU output on canonical trade `45476647/EMTN_CIB_SEC`:
- `ProductValue (ML)`: `0.21859666` (Murex reference: `0.21819305`)
- `CashFlow Rows`: `37` (`DF_terminal = 0.9594909887002673`)
- `Conformal Half-Wid`: `38.74 bps`

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
   - `POST /eq_autocall/price_batch?compute_greeks=true&engine=auto` accepts a JSON list `[req1, req2, ...]`, `{"requests": [...]}`, pre-extracted tensor dicts, or a binary Arrow IPC stream.
   - `POST /eq_autocall/price_arrow?compute_greeks=true` (also `/v1/price/eq_autocall/batch_arrow`) accepts `Content-Type: application/vnd.apache.arrow.stream`.

### C. Run the CPU Test Suite & Latency Benchmark

Execute the core `EQAutocallable` CPU tests and surrogate/Greeks micro-benchmarks:

```bash
.venv/bin/pytest \
  tests/test_eq_autocall_schemas.py \
  tests/test_eq_autocall_curves.py \
  tests/test_eq_autocall_pre_post.py \
  tests/test_eq_autocall_analytical_proxy.py \
  tests/test_eq_autocall_egno.py \
  tests/test_eq_autocall_greeks.py \
  tests/test_eq_autocall_guardian.py \
  tests/test_order_statistic_pooling.py \
  tests/test_api_eq_autocall.py \
  -q

.venv/bin/python benchmarks/eq_autocall_bench.py --suite proxy pricing full_greeks batch
```
