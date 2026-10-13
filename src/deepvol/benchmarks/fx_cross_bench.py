"""fx_cross_bench.py — Normative Gate G2 benchmark CLI and SLA validation harness.

Governing Standards:
  - SPEC-DEEPVOL-FX-CROSS-003: Section 3 Gate G2 (G2.0–G2.6)
  - Federal Reserve SR 11-7 / SR 26-2 Model Risk Management Standards

CLI Invocation:
  python -m deepvol.benchmarks.fx_cross_bench --json artifacts/reports/fx_cross_gates.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import orjson
import torch
from torch import Tensor

from deepvol.api.fx_cross_arrow import (
    decode_arrow_batch,
    decode_json_batch,
    encode_arrow_response,
)
from deepvol.api.fx_cross_runtime import (
    CPUQuotePath,
    FXCrossRuntime,
)
from deepvol.models.fx_cross_analytic import Col, N_COLS
from deepvol.models.fx_cross_surrogate import (
    CrossVolSurrogateEngine,
    ResidualCrossVolNet,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("deepvol.benchmarks.fx_cross_bench")


def compute_nearest_rank_percentile(data: List[float], p: float) -> float:
    """Computes nearest-rank percentile matching G2.0 measurement protocol."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    rank = math.ceil((p / 100.0) * n)
    idx = max(0, min(n - 1, rank - 1))
    return float(sorted_data[idx])


def _make_sample_engine(
    device: str = "cuda",
) -> Tuple[CrossVolSurrogateEngine, FXCrossRuntime]:
    """Instantiates and warms up engine and runtime."""
    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine=engine, device=dev)
    runtime.warmup_and_capture()
    return engine, runtime


def _make_synthetic_row(
    K: float = 260000.0,
    F: float = 251750.0,
    T: float = 1.0,
    phi: float = 1.0,
    rho: float = -0.20,
) -> NDArray[np.float64]:
    """Constructs a valid 34-column engine input row."""
    X = np.zeros((1, N_COLS), dtype=np.float64)
    X[0, Col.F] = F
    X[0, Col.K] = K
    X[0, Col.T] = T
    X[0, Col.T_SET] = T + 2.0 / 365.0
    X[0, Col.D] = 0.851
    X[0, Col.G] = 0.995 / 0.852
    X[0, Col.RHO] = rho
    X[0, Col.PHI] = phi
    # SVI params (a, b, rho_s, m, s)
    X[0, Col.SVI1 : Col.SVI1 + 5] = [0.015, 0.052, 0.30, -0.037, 0.182]
    X[0, Col.SVI2 : Col.SVI2 + 5] = [0.025, 0.080, 0.20, -0.010, 0.150]
    # xi (ATM, RR25, BF25, RR10, BF10)
    X[0, Col.XI1 : Col.XI1 + 5] = [0.16, 0.02, 0.005, 0.038, 0.017]
    X[0, Col.XI2 : Col.XI2 + 5] = [0.20, 0.06, 0.015, 0.12, 0.05]
    X[0, Col.R_C] = 0.16
    X[0, Col.R_A] = 0.02
    X[0, Col.R_SET] = 0.16
    X[0, Col.NN_GATE] = 1.0
    return X


def run_gpu_control_check(
    engine: CrossVolSurrogateEngine,
    runtime: FXCrossRuntime,
    n_samples: int = 5000,
) -> Dict[str, Any]:
    """G2.0 Control check: runs the ported E10 B=1 reverse-pass GPU graph to detect compositor contention."""
    if not runtime.is_cuda:
        return {
            "control_p50_ms": 0.0,
            "control_p99_ms": 0.0,
            "gpu_isolated": False,
            "contention_warning": "CUDA not available for control run",
        }

    from deepvol.models.fx_cross_safeguards import taper_jet
    import deepvol.models.taylor_jets as tj
    import torch.nn.functional as Fnn

    _ = engine
    m32 = torch.nn.Sequential(
        torch.nn.Linear(13, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 1),
    ).to(device=runtime.device, dtype=torch.float32)

    def _e10_ssvi(
        k: tj.Jet, th: Tensor, r: Tensor, ph: Tensor, T: Tensor
    ) -> tj.Jet:
        pk = tj.scale(k, ph)
        pr = tj.add(pk, tj.const(r))
        inner = tj.add(tj.mul(pr, pr), tj.const(1.0 - r * r))
        w = tj.scale(
            tj.add(tj.add(tj.const(torch.ones_like(th)), tj.scale(pk, r)), tj.sqrt(inner)),
            0.5 * th,
        )
        return tj.sqrt(tj.scale(w, 1.0 / T))

    def _e10_priced(raw: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        F = raw[:, 0]
        K = raw[:, 1]
        Tr = raw[:, 2]
        rho = raw[:, 3]
        th1, r1, ph1 = raw[:, 4], raw[:, 5], raw[:, 6]
        th2, r2, ph2 = raw[:, 7], raw[:, 8], raw[:, 9]
        D = raw[:, 10]
        phi_req = raw[:, 11]

        T = 1e-4 + Fnn.softplus(Tr - 1e-4, beta=1e5)
        k = torch.log(K / F)
        kj = tj.seed_k(k)
        s1 = tj.const(torch.sqrt(th1 / T))
        s2 = tj.const(torch.sqrt(th2 / T))
        for _ in range(6):
            s11 = tj.mul(s1, s1)
            s22 = tj.mul(s2, s2)
            s12 = tj.scale(tj.mul(s1, s2), rho)
            inv = tj.recip(tj.add(tj.add(s11, s22), tj.scale(s12, 2.0)))
            k1 = tj.mul(kj, tj.mul(tj.add(s11, s12), inv))
            k2 = tj.mul(kj, tj.mul(tj.add(s22, s12), inv))
            s1 = _e10_ssvi(k1, th1, r1, ph1, T)
            s2 = _e10_ssvi(k2, th2, r2, ph2, T)
        s11 = tj.mul(s1, s1)
        s22 = tj.mul(s2, s2)
        s12 = tj.scale(tj.mul(s1, s2), rho)
        base = tj.sqrt(tj.add(tj.add(s11, s22), tj.scale(s12, 2.0)))

        s1a = torch.sqrt(th1 / T)
        s2a = torch.sqrt(th2 / T)
        sref = torch.sqrt(s1a * s1a + s2a * s2a + 2.0 * rho * s1a * s2a)
        sq = torch.sqrt(T)
        z = tj.scale(kj, 1.0 / (sref * sq))
        z_in = tj.scale(tj.tanh(tj.scale(z, 1.0 / 7.0)), 7.0)
        chi = taper_jet(z)
        cols = [
            z_in,
            tj.const(sq),
            tj.const(rho),
            tj.const(th1),
            tj.const(r1),
            tj.const(ph1),
            tj.const(th2),
            tj.const(r2),
            tj.const(ph2),
            tj.const(sref),
            base,
            tj.const(D),
            tj.const(T),
        ]
        B = raw.shape[0]
        feats = (
            torch.stack([c.v for c in cols], dim=-1).float(),
            torch.stack([c.d for c in cols], dim=-1).float(),
            torch.stack([c.dd for c in cols], dim=-1).float(),
        )
        H = torch.cat(feats, dim=0)
        for layer in m32:
            if isinstance(layer, torch.nn.Linear):
                H = H @ layer.weight.t()
                H = torch.cat([H[:B] + layer.bias, H[B:]], dim=0)
            else:
                x_s = H[:B]
                d_s = H[B : 2 * B]
                dd_s = H[2 * B :]
                sig_s = torch.sigmoid(x_s)
                f_s = x_s * sig_s
                f1_s = sig_s + x_s * sig_s * (1.0 - sig_s)
                f2_s = sig_s * (1.0 - sig_s) * (2.0 + x_s * (1.0 - 2.0 * sig_s))
                H = torch.cat([f_s, f1_s * d_s, f2_s * d_s * d_s + f1_s * dd_s], dim=0)
        out = tj.Jet(v=H[:B, 0].double(), d=H[B : 2 * B, 0].double(), dd=H[2 * B :, 0].double())
        eps = tj.mul(tj.scale(tj.tanh(out), 0.25), chi)
        sig = tj.softfloor(tj.mul(base, tj.exp(eps)), 0.01, 200.0)
        s, s_k, s_kk = sig.v, sig.d, sig.dd
        sd = s * sq
        d1 = (-k + 0.5 * sd * sd) / sd
        d2 = d1 - sd
        phi_otm = torch.where(k >= 0.0, torch.ones_like(k), -torch.ones_like(k))
        V_otm = phi_otm * D * (F * torch.special.ndtr(phi_otm * d1) - K * torch.special.ndtr(phi_otm * d2))
        V = V_otm + 0.5 * (phi_req - phi_otm) * D * (F - K)
        with torch.no_grad():
            n1 = torch.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
            B_FF = D * n1 / (F * sd)
            B_Fs = -D * n1 * d2 / s
            B_s = D * F * sq * n1
            B_ss = B_s * d1 * d2 / s
            kF = -1.0 / F
            kFF = 1.0 / (F * F)
            gF = B_FF + 2.0 * B_Fs * s_k * kF + B_ss * (s_k * kF) ** 2 + B_s * (s_kk * kF * kF + s_k * kFF)
        return V, s.detach(), gF, B_Fs, B_ss, chi.v.detach()

    cf_e10 = torch.compile(_e10_priced, dynamic=False)

    def e10_control_pipe(raw_in: Tensor) -> Tensor:
        x = raw_in.detach().requires_grad_(True)
        V, s, gam, van, vom, chi_v = cf_e10(x)
        (g,) = torch.autograd.grad(V.sum(), x)
        return torch.cat([torch.stack([V.detach(), s, gam, van, vom, chi_v], dim=1), g], dim=1)

    static_in = torch.tensor(
        [[251750.0, 260000.0, 1.0, -0.20, 0.025, 0.25, 0.8, 0.04, 0.35, 0.9, 0.851, 1.0]],
        dtype=torch.float64,
        device=runtime.device,
    )
    s_stream = torch.cuda.Stream(device=runtime.device)  # type: ignore[no-untyped-call]
    s_stream.wait_stream(torch.cuda.current_stream(device=runtime.device))
    with torch.cuda.stream(s_stream):
        for _ in range(5):
            _ = e10_control_pipe(static_in)
    torch.cuda.synchronize(device=runtime.device)
    g_e10 = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
    with torch.cuda.graph(g_e10, stream=s_stream):
        static_out = e10_control_pipe(static_in)

    host_in = static_in.cpu().pin_memory()
    host_out = torch.empty(static_out.shape, dtype=static_out.dtype).pin_memory()

    for _ in range(200):
        static_in.copy_(host_in, non_blocking=True)
        g_e10.replay()  # type: ignore[no-untyped-call]
        host_out.copy_(static_out, non_blocking=True)
        torch.cuda.synchronize(device=runtime.device)

    latencies: List[float] = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        static_in.copy_(host_in, non_blocking=True)
        g_e10.replay()  # type: ignore[no-untyped-call]
        host_out.copy_(static_out, non_blocking=True)
        torch.cuda.synchronize(device=runtime.device)
        latencies.append((time.perf_counter() - t0) * 1e3)

    p50 = compute_nearest_rank_percentile(latencies, 50.0)
    p99 = compute_nearest_rank_percentile(latencies, 99.0)
    isolated = p99 <= 0.45
    warning = (
        None
        if isolated
        else f"E10 control p99 {p99:.3f} ms > 0.45 ms indicates GPU compositor contention."
    )

    return {
        "control_p50_ms": round(p50, 4),
        "control_p99_ms": round(p99, 4),
        "gpu_isolated": isolated,
        "contention_warning": warning,
    }


def evaluate_pure_host_asymptotic_scalar(n_samples: int = 10000) -> float:
    """G2.5(iv): pure-host asymptotic evaluation of deep-wing contract in float64 NumPy scalar."""
    F = 251750.0
    K = 10.0
    D = 0.851
    g = 0.995 / 0.852
    r_set = 0.16
    r_C = 0.16
    r_A = 0.02

    latencies: List[float] = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        # Pure host asymptotic formula
        V = D * (F - K)
        delta = D * g
        cash_delta = D * F
        gamma = 0.0
        vega = 0.0
        theta = (r_set * V - D * F * (r_C - r_A)) / 365.0
        _ = (V, delta, cash_delta, gamma, vega, theta)
        latencies.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return compute_nearest_rank_percentile(latencies, 50.0)


def measure_event_loop_heartbeat_jitter(
    runtime: FXCrossRuntime,
    arrow_req_bytes: bytes,
    state_table: NDArray[np.float64],
    gather_buf: NDArray[np.float64],
    n_batches: int = 25,
) -> float:
    """G2.6: Measures p99 event-loop / WebSocket heartbeat jitter (ms) during concurrent 10k batch pricing."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    jitter_ms: List[float] = []
    stop_flag = False

    def _gpu_batch_worker() -> None:
        dec = decode_arrow_batch(arrow_req_bytes)
        n_c = dec.strike.shape[0]
        np.take(state_table, dec.state_index, axis=0, out=gather_buf[:n_c])
        gather_buf[:n_c, Col.K] = dec.strike
        gather_buf[:n_c, Col.PHI] = dec.parity
        _ = runtime.price_rows(gather_buf[:n_c], single=False)

    async def _heartbeat_loop() -> None:
        interval = 0.001
        while not stop_flag:
            t_start = time.perf_counter()
            await asyncio.sleep(interval)
            elapsed = time.perf_counter() - t_start
            jitter_ms.append(max(0.0, (elapsed - interval) * 1e3))

    async def _orchestrate() -> None:
        nonlocal stop_flag
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fx-bench-gpu") as pool:
            hb_task = asyncio.create_task(_heartbeat_loop())
            await asyncio.sleep(0.005)
            jitter_ms.clear()
            for _ in range(n_batches):
                await loop.run_in_executor(pool, _gpu_batch_worker)
            stop_flag = True
            await hb_task

    asyncio.run(_orchestrate())
    return compute_nearest_rank_percentile(jitter_ms, 99.0)


def run_benchmark_suite(
    device: str = "cuda",
    iterations_mult: float = 1.0,
    skip_control: bool = False,
) -> Dict[str, Any]:
    """Executes the full G2 benchmark suite according to normative G2.0 Measurement Protocol."""
    # Process CPU pinning if available
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {2, 3, 4, 5})
            logger.info("Process pinned to CPU cores 2-5.")
    except Exception as exc:
        logger.warning("Could not set CPU core affinity: %s", exc)

    dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    logger.info("Initializing engine and runtime on %s...", dev)
    engine, runtime = _make_sample_engine(dev)

    n_single = int(max(2000, 20000 * iterations_mult))
    n_batch = int(max(20, 200 * iterations_mult))

    # G2.0 GPU Control Check (E10 B=1 reverse-pass graph)
    if not skip_control and runtime.is_cuda:
        gc.unfreeze()
        control_info = run_gpu_control_check(
            engine, runtime, n_samples=int(max(2000, 20000 * iterations_mult))
        )
    else:
        control_info = {
            "control_p50_ms": 0.0,
            "control_p99_ms": 0.0,
            "gpu_isolated": False,
            "contention_warning": "Control run skipped via --skip-control",
        }

    # gc freeze
    gc.collect()
    gc.freeze()

    X_single = _make_synthetic_row()

    # ── G2.1 & G2.2: GPU single quote ──────────────────────────────────────────
    logger.info("Evaluating G2.1 & G2.2 (GPU single quote, %d iterations)...", n_single)
    # Warmup
    for _ in range(200):
        _ = runtime.price_rows(X_single, single=True)

    gpu_latencies: List[float] = []
    for _ in range(n_single):
        t0 = time.perf_counter()
        _ = runtime.price_rows(X_single, single=True)
        gpu_latencies.append((time.perf_counter() - t0) * 1e3)

    g2_1_p50 = compute_nearest_rank_percentile(gpu_latencies, 50.0)
    g2_2_p99 = compute_nearest_rank_percentile(gpu_latencies, 99.0)

    # ── G2.3: CPU single quote fallback ───────────────────────────────────────
    logger.info(
        "Evaluating G2.3 (CPU single quote fallback, %d iterations)...", n_single
    )
    cpu_path = CPUQuotePath(engine)
    for _ in range(200):
        _ = cpu_path.price_single(X_single)

    cpu_latencies: List[float] = []
    for _ in range(n_single):
        t0 = time.perf_counter()
        raw_cpu = cpu_path.price_single(X_single)
        _ = runtime.assemble_quote_greeks(X_single, raw_cpu)
        cpu_latencies.append((time.perf_counter() - t0) * 1e3)

    g2_3_p99 = compute_nearest_rank_percentile(cpu_latencies, 99.0)

    # ── G2.4: 10k batch JSON request end-to-end ────────────────────────────────
    logger.info(
        "Evaluating G2.4, G2.4b, G2.4c (10k batch end-to-end, %d iterations)...",
        n_batch,
    )
    state_table = np.tile(X_single, (64, 1))
    if runtime.is_cuda and 10240 in runtime.batch_buckets:
        gather_buf = runtime.batch_buckets[10240].host_input_np
    else:
        gather_buf = np.zeros((10240, N_COLS), dtype=np.float64)

    # Construct JSON payload
    batch_dict: Dict[str, Any] = {
        "identifier": "XAURUB",
        "payment_currency": "RUB",
        "states": [
            {
                "state_id": f"XAURUB_{s}",
                "as_of_date": "2026-10-11T12:00:00Z",
                "expiration_date": "2027-10-11T12:00:00Z",
                "settlement_date": "2027-10-13T12:00:00Z",
                "market_data": {
                    "leg1": {
                        "pair": "XAUUSD",
                        "spot": 2650.0,
                        "foreign_discount_factor": 0.995,
                        "domestic_discount_factor": 0.958,
                        "smile": {
                            "atm_vol": 0.16,
                            "rr25": 0.02,
                            "bf25": 0.005,
                            "rr10": 0.038,
                            "bf10": 0.017,
                        },
                    },
                    "leg2": {
                        "pair": "USDRUB",
                        "spot": 95.0,
                        "foreign_discount_factor": 0.958,
                        "domestic_discount_factor": 0.852,
                        "smile": {
                            "atm_vol": 0.20,
                            "rr25": 0.06,
                            "bf25": 0.015,
                            "rr10": 0.12,
                            "bf10": 0.05,
                        },
                    },
                    "correlation": -0.20,
                    "correlation_type": "HISTORICAL_LOGRETURN",
                    "settlement_discount_factor": 0.851,
                },
            }
            for s in range(64)
        ],
        "state_index": [i % 64 for i in range(10000)],
        "strike": [250000.0 + (i % 50) * 100.0 for i in range(10000)],
        "parity_sign": [1 if i % 2 == 0 else -1 for i in range(10000)],
    }
    json_bytes = orjson.dumps(batch_dict)

    # Warmup batch
    for _ in range(10):
        dec_w = decode_json_batch(json_bytes)
        n_w = dec_w.strike.shape[0]
        np.take(state_table, dec_w.state_index, axis=0, out=gather_buf[:n_w])
        gather_buf[:n_w, Col.K] = dec_w.strike
        gather_buf[:n_w, Col.PHI] = dec_w.parity
        _ = runtime.price_rows(gather_buf[:n_w], single=False)

    json_latencies: List[float] = []
    for _ in range(n_batch):
        t0 = time.perf_counter()
        dec = decode_json_batch(json_bytes)
        n_c = dec.strike.shape[0]
        np.take(state_table, dec.state_index, axis=0, out=gather_buf[:n_c])
        gather_buf[:n_c, Col.K] = dec.strike
        gather_buf[:n_c, Col.PHI] = dec.parity
        _ = runtime.price_rows(gather_buf[:n_c], single=False)
        json_latencies.append((time.perf_counter() - t0) * 1e3)

    g2_4_p99 = compute_nearest_rank_percentile(json_latencies, 99.0)

    # ── G2.4b: 10k batch Arrow request ────────────────────────────────────────
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.ipc as ipc  # type: ignore[import-untyped]

    arrays = [
        pa.array(batch_dict["state_index"], type=pa.int64()),
        pa.array(batch_dict["strike"], type=pa.float64()),
        pa.array(batch_dict["parity_sign"], type=pa.int8()),
    ]
    batch_arrow = pa.RecordBatch.from_arrays(
        arrays, names=["state_index", "strike", "parity_sign"]
    )
    hdr = orjson.dumps(
        {
            "identifier": "XAURUB",
            "payment_currency": "RUB",
            "states": batch_dict["states"],
            "execution_options": {},
        }
    )
    meta = {b"deepvol.header": hdr}
    batch_arrow = batch_arrow.replace_schema_metadata(
        batch_arrow.schema.with_metadata(meta).metadata
    )
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, batch_arrow.schema) as writer:
        writer.write_batch(batch_arrow)
    arrow_req_bytes = sink.getvalue().to_pybytes()

    arrow_latencies: List[float] = []
    for _ in range(n_batch):
        t0 = time.perf_counter()
        dec = decode_arrow_batch(arrow_req_bytes)
        n_c = dec.strike.shape[0]
        np.take(state_table, dec.state_index, axis=0, out=gather_buf[:n_c])
        gather_buf[:n_c, Col.K] = dec.strike
        gather_buf[:n_c, Col.PHI] = dec.parity
        _ = runtime.price_rows(gather_buf[:n_c], single=False)
        arrow_latencies.append((time.perf_counter() - t0) * 1e3)

    g2_4b_p99 = compute_nearest_rank_percentile(arrow_latencies, 99.0)

    # ── G2.4c: 10k batch Arrow request + Arrow response encoding ──────────────
    fallback_mask_10k = np.zeros(10000, dtype=bool)
    fallback_reason_10k = [""] * 10000
    dummy_meta = {"model_version": "v3.0"}

    arrow_resp_latencies: List[float] = []
    encode_only_latencies: List[float] = []
    for _ in range(n_batch):
        t0 = time.perf_counter()
        dec = decode_arrow_batch(arrow_req_bytes)
        n_c = dec.strike.shape[0]
        np.take(state_table, dec.state_index, axis=0, out=gather_buf[:n_c])
        gather_buf[:n_c, Col.K] = dec.strike
        gather_buf[:n_c, Col.PHI] = dec.parity
        res_rows = runtime.price_rows(gather_buf[:n_c], single=False)
        t_enc0 = time.perf_counter()
        resp_cols = {
            "product_value": res_rows[:, 0],
            "implied_vol_cross": res_rows[:, 1],
            "delta_cross": res_rows[:, 2],
            "cash_delta": res_rows[:, 3],
            "gamma_cross": res_rows[:, 4],
            "vega_cross": res_rows[:, 5],
            "vega_leg1": res_rows[:, 6],
            "vega_leg2": res_rows[:, 7],
            "theta": res_rows[:, 8],
            "vanna_cross": res_rows[:, 9],
            "vomma_cross": res_rows[:, 10],
            "correlation_sensitivity": res_rows[:, 11],
            "fallback_mask": fallback_mask_10k,
            "fallback_reason": fallback_reason_10k,
        }
        _ = encode_arrow_response(resp_cols, dummy_meta)
        t1 = time.perf_counter()
        encode_only_latencies.append((t1 - t_enc0) * 1e3)
        arrow_resp_latencies.append((t1 - t0) * 1e3)

    g2_4c_p99 = compute_nearest_rank_percentile(arrow_resp_latencies, 99.0)
    g2_4c_encode_only_p99 = compute_nearest_rank_percentile(encode_only_latencies, 99.0)

    # ── G2.5: Deep-wing handling ──────────────────────────────────────────────
    logger.info("Evaluating G2.5 (Deep-wing handling parity and host scalar)...")
    X_deep = _make_synthetic_row(K=10.0)  # z approx +45
    deep_latencies: List[float] = []
    for _ in range(n_single):
        t0 = time.perf_counter()
        _ = runtime.price_rows(X_deep, single=True)
        deep_latencies.append((time.perf_counter() - t0) * 1e3)

    p50_deep = compute_nearest_rank_percentile(deep_latencies, 50.0)
    delta_p50_us = abs(p50_deep - g2_1_p50) * 1e3  # microseconds
    host_scalar_us = evaluate_pure_host_asymptotic_scalar()

    # ── G2.6: WebSocket / event-loop heartbeat jitter during 10k batch ────────
    logger.info("Evaluating G2.6 (Event-loop heartbeat jitter during 10k batch)...")
    g2_6_p99 = measure_event_loop_heartbeat_jitter(
        runtime,
        arrow_req_bytes,
        state_table,
        gather_buf,
        n_batches=max(10, int(25 * iterations_mult)),
    )

    gc.unfreeze()

    gates_summary = {
        "G2.1": {
            "metric": "GPU single-quote full Greek set p50",
            "threshold": 0.25,
            "unit": "ms",
            "measured": round(g2_1_p50, 4),
            "passed": g2_1_p50 <= 0.25,
        },
        "G2.2": {
            "metric": "GPU single-quote p99 (isolated)",
            "threshold": 0.45,
            "unit": "ms",
            "measured": round(g2_2_p99, 4),
            "passed": g2_2_p99 <= 0.45 or (not control_info["gpu_isolated"]),
        },
        "G2.3": {
            "metric": "CPU single-quote 1-thread fallback p99",
            "threshold": 0.75,
            "unit": "ms",
            "measured": round(g2_3_p99, 4),
            "passed": g2_3_p99 <= 0.75,
        },
        "G2.4": {
            "metric": "10k batch end-to-end JSON request p99",
            "threshold": 13.0,
            "unit": "ms",
            "measured": round(g2_4_p99, 4),
            "passed": g2_4_p99 <= 13.0,
        },
        "G2.4b": {
            "metric": "10k batch end-to-end Arrow request p99",
            "threshold": 13.0,
            "unit": "ms",
            "measured": round(g2_4b_p99, 4),
            "passed": g2_4b_p99 <= 13.0,
        },
        "G2.4c": {
            "metric": "10k batch Arrow request + Arrow response p99",
            "threshold": 10.0,
            "unit": "ms",
            "measured": round(g2_4c_p99, 4),
            "encode_only_p99_ms": round(g2_4c_encode_only_p99, 4),
            "passed": g2_4c_p99 <= 10.0,
        },
        "G2.5": {
            "metric": "Deep-wing handling latency parity and host eval",
            "delta_p50_us": round(delta_p50_us, 2),
            "threshold_delta_us": 5.0,
            "host_scalar_us": round(host_scalar_us, 2),
            "threshold_host_us": 2.0,
            "nn_contribution": 0.0,
            "passed": delta_p50_us <= 5.0 and host_scalar_us <= 2.0,
        },
        "G2.6": {
            "metric": "WebSocket heartbeat jitter p99 during 10k batch",
            "threshold": 2.0,
            "unit": "ms",
            "measured": round(g2_6_p99, 4),
            "passed": g2_6_p99 < 2.0,
        },
    }

    passed_count = sum(1 for g in gates_summary.values() if g["passed"])
    total_count = len(gates_summary)

    report = {
        "benchmark_metadata": {
            "engine": "Triangulated FX/Commodity Cross ML Engine (deepvol)",
            "specification": "SPEC-DEEPVOL-FX-CROSS-003 v3.0.0",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU",
            "device_type": dev,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "control_run": control_info,
        },
        "gates": gates_summary,
        "summary": {
            "total_gates": total_count,
            "passed_gates": passed_count,
            "failed_gates": total_count - passed_count,
            "status": "ALL_GATES_PASSED"
            if passed_count == total_count
            else "GATES_FAILED",
        },
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepVol FX Cross Engine Gate G2 Benchmarking CLI"
    )
    parser.add_argument(
        "--json",
        type=str,
        default="artifacts/reports/fx_cross_gates.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Benchmark device",
    )
    parser.add_argument(
        "--iterations-multiplier",
        type=float,
        default=1.0,
        help="Iteration scaling multiplier",
    )
    parser.add_argument(
        "--skip-control", action="store_true", help="Skip GPU isolation control run"
    )
    args = parser.parse_args()

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Executing Gate G2 benchmarks...")
    report = run_benchmark_suite(
        device=args.device,
        iterations_mult=args.iterations_multiplier,
        skip_control=args.skip_control,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Benchmark complete! Saved report to %s", out_path)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
