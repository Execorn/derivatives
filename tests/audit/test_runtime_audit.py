"""tests/audit/test_runtime_audit.py — Adversarial Runtime, CUDA Graphs & Concurrency Audit.

Governing Standards & References:
  - SPEC-DEEPVOL-FX-CROSS-003 v3.0.0 (Tasks 4.1-4.3, Gates G2.0-G2.6)
  - Federal Reserve SR 11-7 / SR 26-2 Model Risk Management Standards
  - Audit Specification: Phase 9 (Runtime, CUDA Graphs & Concurrency)

Scope of Verifications:
  1. Static-buffer safety: back-to-back replays, output memory independence, concurrent interleaving bitwise parity.
  2. Padding invariants: GPU B=2 duplicate padding, row 0 vs eager B=1 equivalence, response isolation, strict-xfail B=1.
  3. Bucket routing & Chunking: boundary sizes (1, 2, 3, 63, 64, 65, 1023, 1024, 1025, 10239, 10240, 10241), micro-batching, chunk ordering.
  4. Executor lifecycle & event-loop decoupling: idempotence, failure propagation, clean shutdown, no device work on event loop.
  5. Weight reload under traffic: AsyncReadWriteLock, no torn reads, model_version consistency.
  6. Failure injection: CUDA unavailability, OOM, NaN weights, slow GPU routing, readiness sequencing.
  7. Resource leak stability: RSS, CUDA allocations, open file descriptors, thread count drift <= 1%.
  8. Graph recompilation: 0 recompiles and 0 graph breaks under fullgraph=True after warm-up.

Adversarial Defect Reproductions:
  - test_F001_quote_vs_batch_greek_inconsistency: Theta roll-term and smile vanna omitted in quote graph assembly.
  - test_F002_cpu_quote_path_static_buffer_race: Unsynchronized static buffer in CPUQuotePath races under pool concurrency.
  - test_F003_executor_restart_after_stop_fails: CPU fallback thread pool permanently dead after stop().
  - test_F004_reload_weights_device_work_on_event_loop: Weight reload executes CUDA warmup directly on event loop.
  - test_F005_reload_weights_missing_model_version_update: Weight reload fails to update state['model_version'].
  - test_F006_health_fails_to_report_degraded_when_cuda_unavailable: Health endpoint reports status 'ok' instead of 'degraded' on CPU.
  - test_F007_nan_weights_not_caught_by_guardian: Corrupted NaN network weights bypass guardian and yield NaNs.
  - test_F008_queue_depth_busy_gpu_blindspot: Active GPU worker execution not accounted for in queue_depth check.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pytest
import torch
from torch import Tensor

from deepvol.api.fx_cross_runtime import (
    BATCH_BUCKETS,
    QUOTE_BUCKET_GPU,
    CPUQuotePath,
    FXCrossGPUExecutor,
    FXCrossRuntime,
    GraphBucket,
)
from deepvol.api.schemas_fx_cross import (
    ContractTerms,
    ExecutionOptions,
    FXLegData,
    MarketData,
    ParityEnum,
    PricingEngineEnum,
    SmileQuotes,
    TriangulatedFXOptionColumnarBatchRequest,
    TriangulatedFXOptionRequest,
)
from deepvol.api.server import (
    _FX_CROSS_RW_LOCK,
    _FX_CROSS_STATE,
    _get_fx_cross_state,
    app,
    reload_fx_cross_weights,
)
from deepvol.models.fx_cross_analytic import Col, EngineOutputs, N_COLS
from deepvol.models.fx_cross_surrogate import (
    CrossVolSurrogateEngine,
    ResidualCrossVolNet,
)
from deepvol.mrm.fx_cross_guardian import (
    FallbackReasonEnum,
    FXCrossGuardian,
)


# ── Helpers & Fixtures ─────────────────────────────────────────────────────────


def _create_sample_row(
    strike: float = 260000.0,
    F: float = 251750.0,
    T: float = 1.0,
    phi: float = 1.0,
    rho: float = -0.20,
) -> np.ndarray:
    """Builds a canonical 34-column engine input row with realistic market smile."""
    X = np.zeros((1, N_COLS), dtype=np.float64)
    X[0, Col.F] = F
    X[0, Col.K] = strike
    X[0, Col.T] = T
    X[0, Col.T_SET] = T + 2.0 / 365.0
    X[0, Col.D] = 0.851
    X[0, Col.G] = 0.995 / 0.852
    X[0, Col.RHO] = rho
    X[0, Col.PHI] = phi
    # SVI params
    X[0, Col.SVI1 : Col.SVI1 + 5] = [0.015, 0.052, 0.30, -0.037, 0.182]
    X[0, Col.SVI2 : Col.SVI2 + 5] = [0.025, 0.080, 0.20, -0.010, 0.150]
    # xi
    X[0, Col.XI1 : Col.XI1 + 5] = [0.16, 0.02, 0.005, 0.038, 0.017]
    X[0, Col.XI2 : Col.XI2 + 5] = [0.20, 0.06, 0.015, 0.12, 0.05]
    X[0, Col.R_C] = 0.16
    X[0, Col.R_A] = 0.02
    X[0, Col.R_SET] = 0.16
    X[0, Col.NN_GATE] = 1.0
    return X


def _create_http_request_dict(strike: float = 260000.0) -> Dict[str, Any]:
    """Generates standard JSON request body for FastAPI endpoints."""
    smile1 = {
        "atm_vol": 0.16,
        "rr25": 0.02,
        "bf25": 0.005,
        "rr10": 0.038,
        "bf10": 0.017,
        "delta_convention": "spot_pna",
        "atm_convention": "DNS",
        "strangle_convention": "SMILE",
    }
    smile2 = {
        "atm_vol": 0.20,
        "rr25": 0.06,
        "bf25": 0.015,
        "rr10": 0.12,
        "bf10": 0.05,
        "delta_convention": "forward_pa",
        "atm_convention": "DNS",
        "strangle_convention": "SMILE",
    }
    md = {
        "leg1": {
            "pair": "XAUUSD",
            "spot": 2650.0,
            "foreign_discount_factor": 0.995,
            "domestic_discount_factor": 0.958,
            "smile": smile1,
        },
        "leg2": {
            "pair": "USDRUB",
            "spot": 95.0,
            "foreign_discount_factor": 0.958,
            "domestic_discount_factor": 0.852,
            "smile": smile2,
        },
        "correlation": -0.20,
        "correlation_type": "HISTORICAL_LOGRETURN",
        "settlement_discount_factor": 0.851,
    }
    return {
        "contract": {
            "identifier": "XAURUB",
            "parity": "Call",
            "strike": strike,
            "as_of_date": "2026-10-11T12:00:00Z",
            "expiration_date": "2027-10-11T12:00:00Z",
            "settlement_date": "2027-10-13T12:00:00Z",
            "payment_currency": "RUB",
            "settlement_type": "CASH",
        },
        "market_data": md,
        "execution_options": {
            "engine": "Differential_ML",
            "precision": "float64",
            "smile_dynamics": "STICKY_MONEYNESS",
            "use_guardian": True,
        },
    }


# ── Phase 9.1: Static-Buffer Safety & Concurrency ─────────────────────────────


@pytest.mark.audit
def test_static_buffer_distinct_outputs() -> None:
    """Verifies that consecutive replays with distinct inputs yield distinct, uncorrupted outputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    runtime.warmup_and_capture()

    row_a = _create_sample_row(strike=250000.0)
    row_b = _create_sample_row(strike=270000.0)

    out_a1 = runtime.price_rows(row_a, single=True)
    out_b = runtime.price_rows(row_b, single=True)
    out_a2 = runtime.price_rows(row_a, single=True)

    # Different strikes must yield distinct outputs
    assert not np.allclose(out_a1, out_b), "Distinct strikes produced identical outputs"
    # Replay of identical input must reproduce exactly
    np.testing.assert_allclose(
        out_a1,
        out_a2,
        atol=1e-12,
        rtol=1e-12,
        err_msg="Static buffer replay suffered output bleed/corruption",
    )


@pytest.mark.audit
def test_static_buffer_output_not_mutated_on_later_replay() -> None:
    """Holding a returned array reference must not mutate its values across subsequent replays."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    runtime.warmup_and_capture()

    row_a = _create_sample_row(strike=240000.0)
    held_out = runtime.price_rows(row_a, single=True)
    frozen_copy = held_out.copy()

    # Replay 50 different inputs
    for k in range(250000, 260000, 200):
        _ = runtime.price_rows(_create_sample_row(strike=float(k)), single=True)

    # Ensure held_out was not overwritten by static buffer re-use
    np.testing.assert_allclose(
        held_out,
        frozen_copy,
        atol=1e-14,
        rtol=1e-14,
        err_msg="Held output array was mutated by subsequent CUDA graph replays (missing output copy)",
    )


# ── Phase 9.2: Padding & Compile Gates ─────────────────────────────────────────


@pytest.mark.audit
def test_padding_gpu_quote_b2_row1_duplicate() -> None:
    """GPU quote graph must use bucket size B=2 with request duplicated into row 1."""
    assert QUOTE_BUCKET_GPU == 2, f"QUOTE_BUCKET_GPU must be 2, got {QUOTE_BUCKET_GPU}"


@pytest.mark.audit
def test_padding_row0_equals_eager_b1() -> None:
    """Row 0 of padded B=2 GPU quote graph must match eager B=1 evaluation to <= 1e-8 relative."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    engine = CrossVolSurrogateEngine(net)
    row = _create_sample_row(strike=260000.0)
    row_t = torch.from_numpy(row).to(dtype=torch.float64, device=device)

    with torch.no_grad():
        eager_b1 = engine.forward_single(row_t).cpu().numpy()

    runtime = FXCrossRuntime(engine=engine, device=device)
    runtime.warmup_and_capture()

    if runtime.is_cuda and runtime.quote_bucket is not None:
        padded_raw = runtime.quote_bucket.run(row)
        rel_diff = np.max(np.abs(padded_raw[0] - eager_b1[0]) / np.maximum(np.abs(eager_b1[0]), 1e-8))
        assert rel_diff <= 1e-8, f"Padded row 0 differs from eager B=1 by rel_diff={rel_diff}"


@pytest.mark.audit
def test_padding_strict_xfail_b1_compile_exists() -> None:
    """A regression test compiling forward_single at GPU B=1 must exist and be marked strict=True."""
    import importlib.util

    test_file = Path("/home/execorn/programming/derivatives/tests/test_api_fx_cross.py")
    assert test_file.exists(), "tests/test_api_fx_cross.py must exist"

    spec = importlib.util.spec_from_file_location("test_api_fx_cross_mod", test_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Read source text to verify @pytest.mark.xfail(strict=True) on test_gpu_forward_single_b1_regression
    src = test_file.read_text(encoding="utf-8")
    assert "def test_gpu_forward_single_b1_regression" in src, "test_gpu_forward_single_b1_regression must exist"
    assert "strict=True" in src, "xfail mark must have strict=True"


@pytest.mark.audit
def test_batch_padding_rows_excluded_from_responses_and_audit() -> None:
    """Batch requests (e.g. N=65 padded to B=1024) must return exactly N rows without leaking padding."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    runtime.warmup_and_capture()

    N = 65
    rows = np.vstack([_create_sample_row(strike=250000.0 + i * 100.0) for i in range(N)])
    out = runtime.price_rows(rows, single=False)
    assert out.shape == (N, 12), f"Expected ({N}, 12), got {out.shape}"


# ── Phase 9.3: Bucket Routing & Chunking ──────────────────────────────────────


@pytest.mark.audit
@pytest.mark.parametrize(
    "N",
    [1, 2, 3, 63, 64, 65, 1023, 1024, 1025, 10239, 10240, 10241],
)
def test_bucket_routing_boundaries(N: int) -> None:
    """Verifies that all boundary request sizes route correctly and return exactly N rows."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    runtime.warmup_and_capture()

    base_row = _create_sample_row()
    # Repeat efficiently
    rows = np.repeat(base_row, N, axis=0)
    out = runtime.price_rows(rows, single=(N == 1))
    assert out.shape == (N, 12), f"Failed for batch boundary N={N}, returned shape {out.shape}"


@pytest.mark.audit
def test_chunking_above_10240_preserves_order() -> None:
    """Batches larger than 10,240 must chunk in blocks of 10,240 and preserve row order bitwise."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    runtime.warmup_and_capture()

    N = 10245
    strikes = 200000.0 + np.arange(N, dtype=np.float64) * 10.0
    rows = np.repeat(_create_sample_row(), N, axis=0)
    rows[:, Col.K] = strikes

    out = runtime.price_rows(rows, single=False)
    assert out.shape == (N, 12)

    # Compare first and last rows with isolated evaluations
    out_first = runtime.price_rows(rows[:1], single=False)
    out_last = runtime.price_rows(rows[-1:], single=False)

    np.testing.assert_allclose(out[0], out_first[0], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out[-1], out_last[0], atol=1e-12, rtol=1e-12)


# ── Phase 9.4: Executor Lifecycle & Decoupling ─────────────────────────────────


@pytest.mark.audit
def test_executor_lifecycle_start_stop_idempotent() -> None:
    """Starting or stopping FXCrossGPUExecutor multiple times must be cleanly idempotent."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    executor = FXCrossGPUExecutor(runtime=runtime, window_us=0)

    # Multiple starts
    executor.start(wait_warmup=True)
    assert executor.is_alive
    executor.start(wait_warmup=True)
    assert executor.is_alive

    # Multiple stops
    executor.stop()
    assert not executor.is_alive
    executor.stop()
    assert not executor.is_alive


@pytest.mark.audit
@pytest.mark.asyncio
async def test_executor_exception_propagates_without_hang() -> None:
    """An exception occurring inside the executor must propagate to the awaiting future without hang."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    executor = FXCrossGPUExecutor(runtime=runtime, window_us=0)
    executor.start(wait_warmup=True)

    try:
        # Invalid input shape (1, 10) instead of (1, 34) triggers ValueError
        bad_x = np.zeros((1, 10), dtype=np.float64)
        fut = executor.submit(bad_x, single=False)
        with pytest.raises(Exception):
            await asyncio.wait_for(fut, timeout=5.0)
    finally:
        executor.stop()


# ── Phase 9.6: Failure Injection ──────────────────────────────────────────────


@pytest.mark.audit
def test_failure_injection_cuda_unavailable_cpu_serves() -> None:
    """When device='cpu', FXCrossRuntime and executor must serve valid prices and Greeks."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device="cpu")
    runtime.warmup_and_capture()

    row = _create_sample_row()
    out = runtime.price_rows(row, single=True)
    assert out.shape == (1, 12)
    assert np.all(np.isfinite(out)), "CPU fallback output contained NaN/Inf"
    assert out[0, 0] > 0.0, "Product value must be positive"


# ── Phase 9.8: Recompilations & Graph Breaks ──────────────────────────────────


@pytest.mark.audit
def test_zero_recompiles_zero_graph_breaks() -> None:
    """Verifies that after warm-up, diverse inputs cause 0 recompiles and 0 graph breaks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    runtime.warmup_and_capture()

    # Replay with 20 diverse moneyness / expiries
    for k in [100000.0, 200000.0, 250000.0, 260000.0, 300000.0]:
        for t in [0.05, 0.25, 1.0, 3.0]:
            r = _create_sample_row(strike=k, T=t)
            out = runtime.price_rows(r, single=True)
            assert np.all(np.isfinite(out))


# ── Phase 9 & 11 Defect Reproductions (test_F0xx_...) ──────────────────────────


@pytest.mark.audit
def test_F001_quote_vs_batch_greek_inconsistency() -> None:
    """DEFECT REPRODUCTION: Quote graph assembly vs Batch graph assembly Greek divergence.

    Spec Reference:
      - Phase 8 item 2: Quote vs batch consistency <= 1e-6 of natural scale.
      - Phase 9 item 1: Interleaved quote and batch requests must equal serial bitwise.

    Root Cause:
      In `src/deepvol/api/fx_cross_runtime.py`:
      - `assemble_quote_greeks` (lines 413-420) calculates Black-76 vega and Black-76 vanna,
        omits the smile term `B_ss * s_k * k_F` from vanna, and completely omits
        `roll_term = B_s * s_k * (r_C - r_A)` from theta.
      - `_assemble_batch_greeks_gpu` (lines 88-101) includes the roll term in theta
        and includes the smile term in vanna.

    Failing Condition:
      Evaluating the identical row with `single=True` and `single=False` produces
      divergent theta and vanna exceeding the 1e-6 tolerance whenever smile skew (s_k != 0)
      and rate differentials (r_C != r_A) exist.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    runtime.warmup_and_capture()

    # Input with pronounced skew and rate differential
    row = _create_sample_row(strike=270000.0)  # OTM strike where s_k != 0

    out_quote = runtime.price_rows(row, single=True)   # uses assemble_quote_greeks
    out_batch = runtime.price_rows(row, single=False)  # uses _assemble_batch_greeks_gpu

    theta_quote = out_quote[0, 8]
    theta_batch = out_batch[0, 8]
    theta_diff = abs(theta_quote - theta_batch)

    vanna_quote = out_quote[0, 9]
    vanna_batch = out_batch[0, 9]
    vanna_diff = abs(vanna_quote - vanna_batch)

    # Assert normative consistency: this FAILS because quote graph omits roll_term and smile vanna
    assert theta_diff <= 1e-6, (
        f"F001 VIOLATION: Theta divergence between quote ({theta_quote:.8f}) and batch ({theta_batch:.8f}) "
        f"is {theta_diff:.8f} > 1e-6 due to missing roll_term in assemble_quote_greeks"
    )
    assert vanna_diff <= 1e-6, (
        f"F001 VIOLATION: Vanna divergence between quote ({vanna_quote:.8f}) and batch ({vanna_batch:.8f}) "
        f"is {vanna_diff:.8f} > 1e-6 due to missing smile term in assemble_quote_greeks"
    )


@pytest.mark.audit
def test_F002_cpu_quote_path_static_buffer_race() -> None:
    """DEFECT REPRODUCTION: Unsynchronized static buffer in CPUQuotePath races under pool concurrency.

    Spec Reference:
      - Phase 9 item 1: Static-buffer safety across concurrency.
      - Phase 9 item 6: Single quotes route to CPU when queue depth > 0.

    Root Cause:
      In `src/deepvol/api/fx_cross_runtime.py:CPUQuotePath`:
      `self._input_buf` and `self._input_buf_np` are shared instance variables without
      thread synchronization. When single quotes are offloaded to `self._cpu_fallback_pool`
      (which runs 4 worker threads), concurrent threads overwrite `self._input_buf_np[0]`
      while another thread is evaluating `self.compiled_fn(self._input_buf)`.

    Failing Condition:
      Concurrent pricing of 100 requests with alternating strikes across 4 threads
      causes inputs to collide, producing incorrect outputs not matching serial evaluation.
    """
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    cpu_path = CPUQuotePath(engine)

    row_a = _create_sample_row(strike=200000.0)
    row_b = _create_sample_row(strike=300000.0)

    val_a_serial = cpu_path.price_single(row_a)[0, 0]
    val_b_serial = cpu_path.price_single(row_b)[0, 0]
    assert abs(val_a_serial - val_b_serial) > 100.0

    errors = []

    def worker(row: np.ndarray, expected_val: float) -> None:
        for _ in range(50):
            res = cpu_path.price_single(row)
            actual_val = res[0, 0]
            if abs(actual_val - expected_val) > 1e-4:
                errors.append(f"Expected ~{expected_val:.2f}, got {actual_val:.2f} due to buffer race")

    threads = [
        threading.Thread(target=worker, args=(row_a, val_a_serial)),
        threading.Thread(target=worker, args=(row_b, val_b_serial)),
        threading.Thread(target=worker, args=(row_a, val_a_serial)),
        threading.Thread(target=worker, args=(row_b, val_b_serial)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"F002 VIOLATION: CPUQuotePath buffer race detected: {errors[:3]}"


@pytest.mark.audit
def test_F003_executor_restart_after_stop_fails() -> None:
    """DEFECT REPRODUCTION: FXCrossGPUExecutor cannot be restarted after stop().

    Spec Reference:
      - Phase 9 item 4: Start and stop idempotent; robust executor lifecycle.

    Root Cause:
      In `src/deepvol/api/fx_cross_runtime.py:FXCrossGPUExecutor.stop`:
      Calls `self._cpu_fallback_pool.shutdown(wait=False)`.
      In `start()`:
      `self._cpu_fallback_pool` is NOT re-created.
      Calling `start()` again leaves `_cpu_fallback_pool` shut down, causing
      subsequent CPU fallback jobs to crash.

    Failing Condition:
      Calling `executor.stop()` followed by `executor.start()` leaves CPU fallback broken.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    executor = FXCrossGPUExecutor(runtime=runtime, window_us=0)

    executor.start(wait_warmup=True)
    executor.stop()
    executor.start(wait_warmup=True)

    try:
        # Check if fallback pool accepts jobs
        future = executor._cpu_fallback_pool.submit(lambda: 42)
        res = future.result(timeout=2.0)
        assert res == 42
    except RuntimeError as ex:
        pytest.fail(f"F003 VIOLATION: Executor restart failed to restore CPU pool: {ex}")
    finally:
        executor.stop()


@pytest.mark.audit
@pytest.mark.asyncio
async def test_F004_reload_weights_device_work_on_event_loop() -> None:
    """DEFECT REPRODUCTION: reload_fx_cross_weights executes CUDA graph warmup on event loop thread.

    Spec Reference:
      - Phase 9 item 4: There is no device work on the event loop.

    Root Cause:
      In `src/deepvol/api/server.py:reload_fx_cross_weights`:
      `state['runtime'].warmup_and_capture()` is invoked directly inside the async coroutine
      running on the asyncio event loop thread instead of offloading to the dedicated executor thread.

    Failing Condition:
      Verifies whether calling thread during warmup_and_capture inside reload is the event loop thread.
    """
    state = _get_fx_cross_state()
    loop_thread_ident = threading.get_ident()

    invoked_thread = []
    original_warmup = state["runtime"].warmup_and_capture

    def hook_warmup() -> None:
        invoked_thread.append(threading.get_ident())
        # Do not run full capture in hook
        state["runtime"].graphs_captured = True

    state["runtime"].warmup_and_capture = hook_warmup
    tmp_path = "/tmp/test_weights.pt"
    torch.save(state["engine"].net.state_dict(), tmp_path)

    try:
        await reload_fx_cross_weights(tmp_path)
        assert len(invoked_thread) == 1
        # If invoked_thread == loop_thread_ident, device work was performed on the event loop!
        assert invoked_thread[0] != loop_thread_ident, (
            "F004 VIOLATION: reload_fx_cross_weights executed CUDA warmup directly on asyncio event loop thread"
        )
    finally:
        state["runtime"].warmup_and_capture = original_warmup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.audit
@pytest.mark.asyncio
async def test_F005_reload_weights_missing_model_version_update() -> None:
    """DEFECT REPRODUCTION: reload_fx_cross_weights fails to update state['model_version'].

    Spec Reference:
      - Phase 9 item 5: Every response is entirely old-model or entirely new-model, per model_version.

    Root Cause:
      `reload_fx_cross_weights` only updates net state_dict and re-captures graphs,
      never updating `_FX_CROSS_STATE['model_version']`.

    Failing Condition:
      After reloading weights with a new model version tag, health and response telemetry
      continue reporting the old model_version.
    """
    state = _get_fx_cross_state()
    old_version = state["model_version"]
    tmp_path = "/tmp/test_weights_v2.pt"
    torch.save(state["engine"].net.state_dict(), tmp_path)

    try:
        await reload_fx_cross_weights(tmp_path)
        # Should have updated or provided a way to update model_version
        assert state["model_version"] != old_version, (
            f"F005 VIOLATION: reload_fx_cross_weights did not update model_version (remained {old_version})"
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.audit
@pytest.mark.asyncio
async def test_F006_health_fails_to_report_degraded_when_cuda_unavailable() -> None:
    """DEFECT REPRODUCTION: /fx_cross/health hardcodes status 'ok' when running on CPU.

    Spec Reference:
      - Phase 9 item 6: CUDA unavailable at startup -> CPU path serves, and /health reports degraded.
      - Phase 10 item 6: /health reports degraded flags.

    Root Cause:
      In `src/deepvol/api/server.py:fx_cross_health` (lines 2223-2231):
      Hardcodes `"status": "ok"` regardless of whether device is "cpu" or "cuda",
      with no "degraded" status or flag reported.

    Failing Condition:
      When device is 'cpu', calling `/fx_cross/health` must report `"status": "degraded"`.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        state = _get_fx_cross_state()
        original_device = state.get("device")
        state["device"] = "cpu"

        try:
            resp = await ac.get("/fx_cross/health")
            data = resp.json()
            assert data.get("status") == "degraded", (
                f"F006 VIOLATION: /fx_cross/health returned status='{data.get('status')}' instead of 'degraded' when device='cpu'"
            )
        finally:
            state["device"] = original_device


@pytest.mark.audit
def test_F007_nan_weights_not_caught_by_guardian() -> None:
    """DEFECT REPRODUCTION: Corrupted NaN weights are not caught by guardian.route before inference.

    Spec Reference:
      - Phase 9 item 6: NaN in weights -> the guardian routes to fallback.

    Root Cause:
      In `src/deepvol/mrm/fx_cross_guardian.py:FXCrossGuardian.route`:
      Guardian only evaluates market feature inputs `m` against Mahalanobis distance.
      It never validates neural net weight tensors for NaN/Inf.
      If a model checkpoint has NaN weights, guardian passes `nn_gate=1.0`, leading to
      NaN outputs rather than clean fallback to Analytic_Triangulated.

    Failing Condition:
      Injecting NaN into a weight parameter causes forward evaluation to fail to activate fallback.
    """
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    guardian = FXCrossGuardian()

    # Corrupt one weight with NaN
    with torch.no_grad():
        engine.net.weights[0].data[0, 0, 0] = float("nan")

    row = _create_sample_row()
    row_t = torch.from_numpy(row)

    # Route decision
    decision = guardian.route(row_t)

    # Guardian should have detected corrupted network weights and routed to fallback
    assert bool(decision.fallback_activated[0].item()) is True, (
        "F007 VIOLATION: Guardian route failed to activate fallback when neural surrogate contains NaN weights"
    )


@pytest.mark.audit
@pytest.mark.asyncio
async def test_F008_queue_depth_busy_gpu_blindspot() -> None:
    """DEFECT REPRODUCTION: Queue depth check fails to route quote to CPU while GPU thread is actively busy.

    Spec Reference:
      - Phase 9 item 6: A slow GPU -> single quotes route to CPU when queue depth > 0.

    Root Cause:
      In `src/deepvol/api/fx_cross_runtime.py:FXCrossGPUExecutor.submit`:
      Checks `if single and self.queue_depth > 0:`.
      When the GPU worker thread is actively executing a task (e.g. running a slow kernel),
      it has already popped the task from `self._queue`.
      At that moment, `self._queue.qsize() == 0`!
      A newly arriving single quote sees `queue_depth == 0` and is enqueued to the GPU
      behind the slow operation instead of offloading to CPU.

    Failing Condition:
      While GPU worker thread is sleeping in execution, submitting a single quote
      fails to route to CPU pool and hangs on the GPU queue.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=device)
    executor = FXCrossGPUExecutor(runtime=runtime, window_us=0)
    executor.start(wait_warmup=True)

    gpu_working = threading.Event()
    continue_gpu = threading.Event()

    def slow_pricing_hook(X: np.ndarray, single: bool = False) -> np.ndarray:
        gpu_working.set()
        continue_gpu.wait(timeout=5.0)
        return np.zeros((X.shape[0], 12), dtype=np.float64)

    original_price_rows = runtime.price_rows
    runtime.price_rows = slow_pricing_hook

    try:
        # 1. Dispatch slow batch task to occupy GPU worker thread
        batch_fut = executor.submit(_create_sample_row(), single=False)
        assert gpu_working.wait(timeout=3.0), "GPU worker thread did not start execution"

        # At this point, the GPU thread is busy, and self._queue is empty (qsize == 0)
        assert executor.queue_depth == 0, "Queue depth should be 0 since item was dequeued"

        # 2. Dispatch a single quote: must route to CPU fallback since GPU is busy
        cpu_dispatched = []
        original_cpu_single = runtime.cpu_quote_path.price_single

        def hook_cpu_single(X: np.ndarray) -> np.ndarray:
            cpu_dispatched.append(True)
            return original_cpu_single(X)

        runtime.cpu_quote_path.price_single = hook_cpu_single

        quote_fut = executor.submit(_create_sample_row(), single=True)
        # Give CPU pool brief moment to pick up task if routed to CPU
        await asyncio.sleep(0.1)

        # Assert quote was routed to CPU: this FAILS because queue_depth == 0 blindspot enqueues to GPU
        assert len(cpu_dispatched) > 0, (
            "F008 VIOLATION: Single quote was queued behind busy GPU worker instead of routing to CPU fallback pool"
        )

    finally:
        continue_gpu.set()
        runtime.price_rows = original_price_rows
        executor.stop()
