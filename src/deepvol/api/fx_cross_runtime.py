"""fx_cross_runtime.py — Fused CUDA Graph runtime and dedicated executor for triangulated FX crosses.

Governing Standards:
  - SPEC-DEEPVOL-FX-CROSS-003: Task 4.2 (Gate G2 Latency and Throughput)
  - Federal Reserve SR 11-7 / SR 26-2 Model Governance

Provides:
  - QUOTE_BUCKET_GPU = 2: Padded GPU quote graph avoiding Triton constexpr bug E11.
  - BATCH_BUCKETS = (64, 1024, 10240): Bucketed CUDA graphs with full autograd reverse pass.
  - GraphBucket: Preallocated static device buffers, pinned host buffers, non-blocking transfers,
    and mandatory output cloning.
  - CPUQuotePath: Single-thread compiled C++ Inductor fallback for single quotes.
  - FXCrossRuntime: Coordinates warmup, graph capture, and vectorized Greek assembly.
  - FXCrossGPUExecutor: Dedicated background thread owning CUDA context and event-loop decoupling.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import logging
import math
import queue
import threading
import time
from typing import Any, Callable, Dict, Final, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr
import torch
from torch import Tensor
import torch._inductor.config as inductor_config

from deepvol.models.fx_cross_analytic import Col, N_COLS
from deepvol.models.fx_cross_surrogate import CrossVolSurrogateEngine

# Single-thread inductor compilation prevents subproc pool closed errors in worker threads
inductor_config.compile_threads = 1

logger = logging.getLogger("deepvol.api.fx_cross_runtime")

QUOTE_BUCKET_GPU: Final[int] = 2
BATCH_BUCKETS: Final[Tuple[int, ...]] = (64, 1024, 10240)
_SQ2PI: Final[float] = math.sqrt(2.0 * math.pi)


class GraphBucket:
    """Preallocated static buffers, CUDA Graph capture, and replay engine for a fixed batch size."""

    def __init__(
        self,
        fn: Callable[[Tensor], Tensor],
        B: int,
        n_cols: int,
        device: torch.device,
        pool: Optional[Any] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.B = B
        self.n_cols = n_cols
        self.device = device
        self.fn = fn
        self.pool = pool
        self.stream = stream if stream is not None else torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]

        self.static_input = torch.zeros((B, n_cols), dtype=torch.float64, device=device)
        self.host_input = torch.zeros((B, n_cols), dtype=torch.float64).pin_memory()

        s = self.stream
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            _ = self.fn(self.static_input)
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.empty_cache()

        self.cuda_graph = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
        with torch.cuda.graph(self.cuda_graph, stream=s, pool=self.pool):
            self.static_output = self.fn(self.static_input)
        torch.cuda.empty_cache()

        self.n_out = self.static_output.shape[1]
        self.host_output = torch.zeros(
            (B, self.n_out), dtype=torch.float64
        ).pin_memory()
        self.host_input_np = self.host_input.detach().numpy()
        self.host_output_np = self.host_output.detach().numpy()

    def run(self, host_rows: NDArray[Any]) -> NDArray[Any]:
        """Executes captured graph with host inputs and returns copied valid output rows."""
        N = host_rows.shape[0]
        if N > self.B:
            raise ValueError(f"Requested rows {N} exceeds bucket size {self.B}")

        if N == 1:
            self.host_input_np[0] = host_rows[0]
            if self.B > 1:
                self.host_input_np[1 : self.B] = host_rows[0]
        else:
            self.host_input_np[:N] = host_rows
            if N < self.B:
                self.host_input_np[N : self.B] = host_rows[-1]

        with torch.cuda.stream(self.stream):
            self.static_input.copy_(self.host_input, non_blocking=True)
            self.cuda_graph.replay()  # type: ignore[no-untyped-call]
            self.host_output.copy_(self.static_output, non_blocking=True)

        self.stream.synchronize()

        # Mandatory output cloning prevents static buffer overwrite on subsequent requests
        return self.host_output_np[:N].copy()


class CPUQuotePath:
    """Single-threaded forward-only quote engine compiled via C++ Inductor backend on CPU."""

    def __init__(self, engine: CrossVolSurrogateEngine) -> None:
        self.engine = copy.deepcopy(engine).to("cpu")
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass
        self.compiled_fn = self.engine.forward_single
        try:
            cfn = torch.compile(
                self.engine.forward_single,
                dynamic=False,
                fullgraph=True,
            )
            dummy_X = torch.zeros((1, N_COLS), dtype=torch.float64, device="cpu")
            dummy_X[0, Col.F] = 1.0
            dummy_X[0, Col.K] = 1.0
            dummy_X[0, Col.T] = 1.0
            dummy_X[0, Col.T_SET] = 1.0
            dummy_X[0, Col.D] = 1.0
            dummy_X[0, Col.G] = 1.0
            with torch.no_grad():
                _ = cfn(dummy_X)
            self.compiled_fn = cfn
        except Exception:
            logger.info("CPU Inductor C++ compilation unavailable; using eager CPU forward")
            self.compiled_fn = self.engine.forward_single

    def price_single(self, X: NDArray[Any]) -> NDArray[Any]:
        """Evaluates single-row input (1, 34) on CPU and returns packed output (1, 9)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_t = torch.from_numpy(X).to(dtype=torch.float64, device="cpu")
        with torch.no_grad():
            try:
                out = self.compiled_fn(X_t)
            except Exception:
                out = self.engine.forward_single(X_t)
        return out.detach().numpy().copy()


class FXCrossRuntime:
    """Orchestrates compiled Quote and Batch CUDA Graphs and vectorized Greek assembly."""

    def __init__(self, engine: CrossVolSurrogateEngine, device: str = "cuda") -> None:
        self.engine = engine
        self.device_name = device
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda" and torch.cuda.is_available()

        self.quote_bucket: Optional[GraphBucket] = None
        self.batch_buckets: Dict[int, GraphBucket] = {}
        self.cpu_quote_path = CPUQuotePath(engine)
        self.graphs_captured = False

    def warmup_and_capture(self) -> None:
        """Compiles forward and batch pipelines, captures CUDA Graphs, and freezes memory."""
        if not self.is_cuda:
            self.graphs_captured = True
            return

        self.engine = self.engine.to(self.device)
        torch.cuda.empty_cache()
        shared_stream = torch.cuda.Stream(device=self.device) if self.is_cuda else None  # type: ignore[no-untyped-call]

        # 1. Batch graph: capture largest bucket B=10240 first to create and size shared pool
        cf_batch = torch.compile(self.engine.forward, dynamic=False, fullgraph=True)

        def batch_pipeline(raw_in: Tensor) -> Tensor:
            x = raw_in.detach().requires_grad_(True)
            out = cf_batch(x)
            (grad,) = torch.autograd.grad(out.V.sum(), x)
            eps_std = self.engine.epistemic_std_bps(x)
            packed = torch.cat(
                [
                    out.V.unsqueeze(-1).detach(),
                    grad.detach(),
                    out.sigma.unsqueeze(-1).detach(),
                    out.s_k.unsqueeze(-1).detach(),
                    out.s_kk.unsqueeze(-1).detach(),
                    out.B_F.unsqueeze(-1).detach(),
                    out.B_FF.unsqueeze(-1).detach(),
                    out.B_Fs.unsqueeze(-1).detach(),
                    out.B_s.unsqueeze(-1).detach(),
                    out.B_ss.unsqueeze(-1).detach(),
                    out.chi.unsqueeze(-1).detach(),
                    eps_std.unsqueeze(-1).detach(),
                ],
                dim=1,
            )
            return packed

        mempool: Optional[Any] = None
        for B_sz in sorted(BATCH_BUCKETS, reverse=True):
            gb = GraphBucket(
                batch_pipeline,
                B=B_sz,
                n_cols=N_COLS,
                device=self.device,
                pool=mempool,
                stream=shared_stream,
            )
            self.batch_buckets[B_sz] = gb
            if mempool is None and self.is_cuda:
                mempool = gb.cuda_graph.pool()  # type: ignore[no-untyped-call]

        # 2. Quote graph (B=2 padded) sharing same stream and mempool
        cf_single = torch.compile(
            self.engine.forward_single, dynamic=False, fullgraph=True
        )
        self.quote_bucket = GraphBucket(
            cf_single,
            B=QUOTE_BUCKET_GPU,
            n_cols=N_COLS,
            device=self.device,
            pool=mempool,
            stream=shared_stream,
        )

        torch.cuda.empty_cache()
        gc.collect()
        gc.freeze()
        self.graphs_captured = True

    def price_rows(self, X: NDArray[Any], single: bool = False) -> NDArray[Any]:
        """Selects appropriate bucket, executes graph, and assembles 12 canonical Greek columns."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N = X.shape[0]

        if single and N == 1:
            if self.is_cuda and self.quote_bucket is not None:
                raw_out = self.quote_bucket.run(X)  # (1, 9)
            else:
                raw_out = self.cpu_quote_path.price_single(X)  # (1, 9)
            return self.assemble_quote_greeks(X, raw_out)

        # Batch execution
        if not self.is_cuda or not self.batch_buckets:
            # CPU fallback for batch
            return self._price_rows_cpu(X)

        if N > BATCH_BUCKETS[-1]:
            chunks = []
            max_b = BATCH_BUCKETS[-1]
            for start_idx in range(0, N, max_b):
                sub_X = X[start_idx : start_idx + max_b]
                chunks.append(self.price_rows(sub_X, single=False))
            return np.vstack(chunks)

        bucket_size = next((b for b in BATCH_BUCKETS if b >= N), BATCH_BUCKETS[-1])
        bucket = self.batch_buckets[bucket_size]
        raw_out = bucket.run(X)  # (N, 45)
        return self._assemble_batch_greeks(X, raw_out)

    def _price_rows_cpu(self, X: NDArray[Any]) -> NDArray[Any]:
        """CPU fallback path when CUDA is unavailable."""
        X_t = (
            torch.from_numpy(X)
            .to(dtype=torch.float64, device="cpu")
            .requires_grad_(True)
        )
        out = self.engine.forward(X_t)
        (grad,) = torch.autograd.grad(out.V.sum(), X_t)
        eps_std = self.engine.epistemic_std_bps(X_t)
        packed = torch.cat(
            [
                out.V.unsqueeze(-1).detach(),
                grad.detach(),
                out.sigma.unsqueeze(-1).detach(),
                out.s_k.unsqueeze(-1).detach(),
                out.s_kk.unsqueeze(-1).detach(),
                out.B_F.unsqueeze(-1).detach(),
                out.B_FF.unsqueeze(-1).detach(),
                out.B_Fs.unsqueeze(-1).detach(),
                out.B_s.unsqueeze(-1).detach(),
                out.B_ss.unsqueeze(-1).detach(),
                out.chi.unsqueeze(-1).detach(),
                eps_std.unsqueeze(-1).detach(),
            ],
            dim=1,
        )
        raw_out = packed.detach().numpy().copy()
        return self._assemble_batch_greeks(X, raw_out)

    @staticmethod
    def assemble_quote_greeks(X: NDArray[Any], raw: NDArray[Any]) -> NDArray[Any]:
        """Vectorized assembly of 12 response columns from quote graph output."""
        if X.shape[0] == 1:
            F0 = float(X[0, Col.F])
            K0 = float(X[0, Col.K])
            T0 = max(float(X[0, Col.T]), 1e-4)
            D0 = float(X[0, Col.D])
            g0 = float(X[0, Col.G])
            phi0 = float(X[0, Col.PHI])
            r_C0 = float(X[0, Col.R_C])
            r_A0 = float(X[0, Col.R_A])
            r_set0 = float(X[0, Col.R_SET])

            V0 = float(raw[0, 0])
            sigma0 = float(raw[0, 1])
            gF0 = float(raw[0, 2])
            dVdF0 = float(raw[0, 3])
            B_ss0 = float(raw[0, 5])
            d_rho0 = float(raw[0, 6])
            d_d10 = float(raw[0, 7])
            d_d20 = float(raw[0, 8])

            delta0 = g0 * dVdF0
            cash_delta0 = F0 * dVdF0
            gamma0 = g0 * g0 * gF0

            sd0 = sigma0 * math.sqrt(T0)
            d10 = (-math.log(K0 / F0) + 0.5 * sd0 * sd0) / sd0
            d20 = d10 - sd0
            n10 = math.exp(-0.5 * d10 * d10) / _SQ2PI

            vega0 = D0 * F0 * math.sqrt(T0) * n10
            vanna0 = -D0 * g0 * n10 * d20 / sigma0
            vomma0 = B_ss0

            n_phi_d10 = 0.5 * (1.0 + math.erf(phi0 * d10 / math.sqrt(2.0)))
            carry0 = phi0 * D0 * F0 * n_phi_d10 * (r_C0 - r_A0)
            decay0 = (D0 * F0 * sigma0 * n10) / (2.0 * math.sqrt(T0))
            theta0 = (r_set0 * V0 - carry0 - decay0) / 365.0

            return np.array(
                [
                    [
                        V0,
                        sigma0,
                        delta0,
                        cash_delta0,
                        gamma0,
                        vega0,
                        d_d10,
                        d_d20,
                        theta0,
                        vanna0,
                        vomma0,
                        d_rho0,
                    ]
                ],
                dtype=np.float64,
            )

        F_val = X[:, Col.F]
        K_val = X[:, Col.K]
        T_val = np.maximum(X[:, Col.T], 1e-4)
        D_val = X[:, Col.D]
        g_val = X[:, Col.G]
        phi_val = X[:, Col.PHI]
        r_C = X[:, Col.R_C]
        r_A = X[:, Col.R_A]
        r_set = X[:, Col.R_SET]

        V = raw[:, 0]
        sigma = raw[:, 1]
        gF = raw[:, 2]
        dVdF = raw[:, 3]
        B_ss = raw[:, 5]
        d_rho = raw[:, 6]
        d_delta1 = raw[:, 7]
        d_delta2 = raw[:, 8]

        delta = g_val * dVdF
        cash_delta = F_val * dVdF
        gamma = g_val * g_val * gF

        sd = sigma * np.sqrt(T_val)
        d1 = (-np.log(K_val / F_val) + 0.5 * sd * sd) / sd
        d2 = d1 - sd
        n1 = np.exp(-0.5 * d1 * d1) / _SQ2PI

        vega = D_val * F_val * np.sqrt(T_val) * n1
        vanna = -D_val * g_val * n1 * d2 / sigma
        vomma = B_ss

        n_phi_d1 = ndtr(phi_val * d1)
        carry_term = phi_val * D_val * F_val * n_phi_d1 * (r_C - r_A)
        decay_term = (D_val * F_val * sigma * n1) / (2.0 * np.sqrt(T_val))
        theta = (r_set * V - carry_term - decay_term) / 365.0

        return np.column_stack(
            [
                V,
                sigma,
                delta,
                cash_delta,
                gamma,
                vega,
                d_delta1,
                d_delta2,
                theta,
                vanna,
                vomma,
                d_rho,
            ]
        )

    _assemble_quote_greeks = assemble_quote_greeks

    @staticmethod
    def _assemble_batch_greeks(X: NDArray[Any], raw: NDArray[Any]) -> NDArray[Any]:
        """Vectorized assembly of 12 response columns from batch graph output."""
        F_val = X[:, Col.F]
        K_val = X[:, Col.K]
        T_val = np.maximum(X[:, Col.T], 1e-4)
        D_val = X[:, Col.D]
        g_val = X[:, Col.G]
        phi_val = X[:, Col.PHI]
        r_C = X[:, Col.R_C]
        r_A = X[:, Col.R_A]
        r_set = X[:, Col.R_SET]

        V = raw[:, 0]
        grad = raw[:, 1:35]
        sigma = raw[:, 35]
        s_k = raw[:, 36]
        s_kk = raw[:, 37]
        B_F = raw[:, 38]
        B_FF = raw[:, 39]
        B_Fs = raw[:, 40]
        B_s = raw[:, 41]
        B_ss = raw[:, 42]

        k_F = -1.0 / F_val
        k_FF = 1.0 / (F_val * F_val)
        dVdF = B_F + B_s * s_k * k_F
        d2VdF2 = (
            B_FF
            + 2.0 * B_Fs * s_k * k_F
            + B_ss * (s_k * k_F) ** 2
            + B_s * (s_kk * k_F * k_F + s_k * k_FF)
        )

        delta = g_val * dVdF
        cash_delta = F_val * dVdF
        gamma = g_val * g_val * d2VdF2
        vega = B_s
        vanna = g_val * (B_Fs + B_ss * s_k * k_F)
        vomma = B_ss

        sd = sigma * np.sqrt(T_val)
        d1 = (-np.log(K_val / F_val) + 0.5 * sd * sd) / sd
        n1 = np.exp(-0.5 * d1 * d1) / _SQ2PI

        n_phi_d1 = ndtr(phi_val * d1)
        carry_term = phi_val * D_val * F_val * n_phi_d1 * (r_C - r_A)
        decay_term = (D_val * F_val * sigma * n1) / (2.0 * np.sqrt(T_val))
        roll_term = B_s * s_k * (r_C - r_A)
        theta = (r_set * V - carry_term - decay_term + roll_term) / 365.0

        d_rho = grad[:, Col.RHO]
        d_delta1 = grad[:, Col.DELTA1]
        d_delta2 = grad[:, Col.DELTA2]

        return np.column_stack(
            [
                V,
                sigma,
                delta,
                cash_delta,
                gamma,
                vega,
                d_delta1,
                d_delta2,
                theta,
                vanna,
                vomma,
                d_rho,
            ]
        )


class _Task:
    def __init__(
        self,
        X: NDArray[Any],
        single: bool,
        future: asyncio.Future[NDArray[Any]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.X = X
        self.single = single
        self.future = future
        self.loop = loop


class FXCrossGPUExecutor:
    """Dedicated worker thread managing CUDA context and asynchronous request queuing."""

    def __init__(self, runtime: FXCrossRuntime, window_us: int = 200) -> None:
        self.runtime = runtime
        self.window_us = window_us
        self._queue: queue.Queue[Union[_Task, object]] = queue.Queue()
        self._stop_sentinel = object()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._warmup_error: Optional[Exception] = None
        self._warmup_done: threading.Event = threading.Event()
        self._cpu_fallback_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="deepvol-cpu-fallback"
        )

    def start(self, wait_warmup: bool = True) -> None:
        """Starts the dedicated background executor thread and optionally waits for warmup."""
        if self._running:
            return
        self._running = True
        self._warmup_done.clear()
        self._warmup_error = None
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="FXCrossGPUExecutor",
            daemon=True,
        )
        self._worker_thread.start()
        if wait_warmup:
            self._warmup_done.wait(timeout=120.0)
            if self._warmup_error is not None:
                raise RuntimeError(
                    f"FXCrossGPUExecutor warmup failed: {self._warmup_error}"
                ) from self._warmup_error

    def stop(self) -> None:
        """Stops the worker thread cleanly and joins."""
        if not self._running:
            return
        self._running = False
        self._queue.put(self._stop_sentinel)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
        self._cpu_fallback_pool.shutdown(wait=False)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def is_alive(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def submit(self, X: NDArray[Any], single: bool) -> asyncio.Future[NDArray[Any]]:
        """Dispatches an inference job and returns an awaitable asyncio.Future."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[NDArray[Any]] = loop.create_future()

        # Check worker thread liveness and warmup health
        if self._worker_thread is not None and not self._worker_thread.is_alive():
            err_msg = f"FXCrossGPUExecutor worker thread is not alive. Warmup error: {self._warmup_error}"
            fut.set_exception(RuntimeError(err_msg))
            return fut

        if self._warmup_error is not None:
            fut.set_exception(
                RuntimeError(f"FXCrossGPUExecutor warmup failed: {self._warmup_error}")
            )
            return fut

        # Dynamic fallback routing: bypass busy GPU queue for single quotes
        if single and self.queue_depth > 0:

            def _eval_cpu() -> None:
                try:
                    res = self.runtime.cpu_quote_path.price_single(X)
                    out = self.runtime.assemble_quote_greeks(X, res)
                    loop.call_soon_threadsafe(fut.set_result, out)
                except Exception as ex:
                    loop.call_soon_threadsafe(fut.set_exception, ex)

            self._cpu_fallback_pool.submit(_eval_cpu)
            return fut

        task = _Task(X=X, single=single, future=fut, loop=loop)
        self._queue.put(task)
        return fut

    def _worker_loop(self) -> None:
        """Worker thread loop executing on dedicated CUDA context."""
        try:
            if self.runtime.is_cuda:
                dev_idx = self.runtime.device.index or 0
                torch.cuda.set_device(dev_idx)
            self.runtime.warmup_and_capture()
        except Exception as exc:
            logger.exception("Failed during FXCrossRuntime warmup_and_capture: %s", exc)
            self._warmup_error = exc
        finally:
            self._warmup_done.set()

        if self._warmup_error is not None:
            return

        while self._running:
            try:
                item = self._queue.get()
                if item is self._stop_sentinel:
                    break
                task = item
                assert isinstance(task, _Task)

                if not task.single:
                    try:
                        res = self.runtime.price_rows(task.X, single=False)
                        task.loop.call_soon_threadsafe(task.future.set_result, res)
                    except Exception as ex:
                        task.loop.call_soon_threadsafe(task.future.set_exception, ex)
                    continue

                # Single quote with micro-batching window check
                accumulated = [task]
                if self.window_us > 0 and not self._queue.empty():
                    t_target = time.perf_counter() + self.window_us * 1e-6
                    while time.perf_counter() < t_target and len(accumulated) < 64:
                        try:
                            next_item = self._queue.get_nowait()
                            if next_item is self._stop_sentinel:
                                self._queue.put(self._stop_sentinel)
                                break
                            assert isinstance(next_item, _Task)
                            if next_item.single:
                                accumulated.append(next_item)
                            else:
                                self._queue.put(next_item)
                                break
                        except queue.Empty:
                            break

                if len(accumulated) == 1:
                    try:
                        res = self.runtime.price_rows(task.X, single=True)
                        task.loop.call_soon_threadsafe(task.future.set_result, res)
                    except Exception as ex:
                        task.loop.call_soon_threadsafe(task.future.set_exception, ex)
                else:
                    try:
                        X_cat = np.vstack([t.X for t in accumulated])
                        res_cat = self.runtime.price_rows(X_cat, single=False)
                        for idx, t in enumerate(accumulated):
                            t.loop.call_soon_threadsafe(
                                t.future.set_result, res_cat[idx : idx + 1]
                            )
                    except Exception as ex:
                        for t in accumulated:
                            t.loop.call_soon_threadsafe(t.future.set_exception, ex)

            except Exception as ex:
                logger.error("Executor failure in worker loop: %s", ex, exc_info=True)
