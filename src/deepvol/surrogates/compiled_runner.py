"""Bucketed CUDA Graph and Inductor runner for EGNO autocall surrogate inference and VJP."""

from __future__ import annotations

import collections
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
import hashlib
import itertools
import threading
import warnings
from typing import Any, Callable, Deque, Dict, Optional, Protocol, Set, Tuple, cast

import torch
import torch.nn as nn

from deepvol.models.autocall.gpu_cache_manager import (
    GLOBAL_GPU_CACHE_MANAGER,
    normalize_cuda_device,
)

FWD_BATCH_BUCKETS: Tuple[int, ...] = (1, 16, 64)
VJP_BATCH_BUCKETS: Tuple[int, ...] = (1, 4, 16)
BATCH_BUCKETS: Tuple[int, ...] = FWD_BATCH_BUCKETS
PARTITION_CUDAGRAPH_MAX_BYTES: int = 40 * 1024 * 1024
GEOM_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (1, 4),
    (3, 12),
    (3, 40),
    (5, 1),
    (5, 8),
    (5, 12),
    (5, 24),
    (8, 12),
    (8, 48),
    (8, 64),
    (12, 16),
    (12, 64),
)
POOL_SLOTS_PER_BUCKET: int = 4

_BucketKey = Tuple[int, int, int, torch.dtype, torch.device]
_FullBucketKey = Tuple[int, int, int, torch.dtype, torch.device, int, int, int, int]


class _PoolBudgetExceeded(RuntimeError):
    pass


_POOL_ANCHOR_LOCK = threading.Lock()
_POOL_ANCHORS: Dict[Tuple[Tuple[str, int], Tuple[int, ...]], Tuple[Any, torch.Tensor]] = {}


def _ensure_pool_anchor(device: torch.device, graph_pool: Any, cap_stream: Any) -> None:
    """Keep a minimal anchor graph per shared pool so invalidation never drops PyTorch pool use_count to 0."""
    if graph_pool is None:
        return
    dev_key = normalize_cuda_device(device)
    pool_id = tuple(int(x) for x in graph_pool) if isinstance(graph_pool, tuple) else (id(graph_pool),)
    anchor_key = (dev_key, pool_id)
    with _POOL_ANCHOR_LOCK:
        if anchor_key in _POOL_ANCHORS:
            return
        with torch.inference_mode(False), torch.cuda.stream(cap_stream), torch.no_grad():
            anchor = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
            with torch.cuda.graph(anchor, pool=graph_pool, stream=cap_stream):
                anchor_buf = torch.ones((1,), dtype=torch.float32, device=device)
        cap_stream.synchronize()
        _POOL_ANCHORS[anchor_key] = (anchor, anchor_buf)


class _ResidualsFn(Protocol):
    def __call__(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
        *,
        require_second_order_sdpa: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@dataclass
class _FwdGraphSlot:
    graph: Any
    ev: Any
    nf: torch.Tensor
    ef: torch.Tensor
    am: torch.Tensor
    sf: torch.Tensor
    sm: torch.Tensor
    gf: torch.Tensor
    out_haz: torch.Tensor
    out_cpn: torch.Tensor
    out_put: torch.Tensor


@dataclass
class _VjpGraphSlot:
    graph: Any
    ev: Any
    nf: torch.Tensor
    ef: torch.Tensor
    am: torch.Tensor
    sf: torch.Tensor
    sm: torch.Tensor
    gf: torch.Tensor
    gh: torch.Tensor
    gc: torch.Tensor
    gp: torch.Tensor
    out_haz: torch.Tensor
    out_cpn: torch.Tensor
    out_put: torch.Tensor
    grad_node: torch.Tensor
    grad_edge: torch.Tensor
    grad_sched: torch.Tensor
    grad_global: torch.Tensor


@dataclass
class _SlotPool:
    lock: threading.Lock = field(default_factory=threading.Lock)
    cond: threading.Condition = field(init=False)
    free_slots: Deque[Any] = field(default_factory=collections.deque)
    total_created: int = 0
    invalidated: bool = False

    def __post_init__(self) -> None:
        self.cond = threading.Condition(self.lock)


class CompiledSurrogateRunner:
    """Thread-safe bucketed runner using K=4 event-synced CUDAGraph slots per bucket."""

    def __init__(
        self,
        model: nn.Module,
        *,
        use_tf32: bool = False,
        enable_cudagraphs: bool = True,
        enable_inductor: bool = True,
    ) -> None:
        self.model = model
        self.use_tf32 = bool(use_tf32)
        self.enable_cudagraphs = bool(enable_cudagraphs)
        self.enable_inductor = bool(enable_inductor)

        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._fwd_pools: Dict[_FullBucketKey, _SlotPool] = {}
        self._vjp_pools: Dict[_FullBucketKey, _SlotPool] = {}
        self._active_devices: Set[torch.device] = set()
        self._pool_budget_exceeded_keys: Set[_FullBucketKey] = set()
        self._compiled_cache: Dict[Tuple[str, Tuple[str, int], torch.dtype], _ResidualsFn] = {}
        self._pool_generation: int = 0
        self._watched_tensors: Tuple[torch.Tensor, ...] = tuple(
            itertools.chain(self.model.parameters(), self.model.buffers())
        )
        self._weight_ver: int = self._compute_weight_version()
        self._model_digest: str = self._compute_model_digest()

    def _compute_weight_version(self) -> int:
        total = 0
        for tensor in self._watched_tensors:
            if not tensor.is_inference():
                total += tensor._version
        return total

    def _compute_model_digest(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(type(self.model).__name__.encode("utf-8"))
        for k, v in sorted(self.model.state_dict().items()):
            h.update(k.encode("utf-8"))
            h.update(v.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes())
        return h.hexdigest()

    def _sync_model_version(self) -> str:
        cur_ver = self._compute_weight_version()
        with self._lock:
            if cur_ver == self._weight_ver:
                return self._model_digest
        # Invalidate outside _lock check so invalidate_graphs keeps strict _lock -> _capture_lock -> pool_lock order.
        self.invalidate_graphs()
        new_digest = self._compute_model_digest()
        with self._lock:
            self._weight_ver = self._compute_weight_version()
            self._model_digest = new_digest
            return self._model_digest

    def invalidate_graphs(self) -> None:
        """Clear captured bucket graphs under strict lock order _lock -> _capture_lock -> pool_lock."""
        with self._lock:
            with self._capture_lock:
                devs = sorted(self._active_devices, key=normalize_cuda_device)
                with ExitStack() as stack:
                    for dev in devs:
                        stack.enter_context(GLOBAL_GPU_CACHE_MANAGER.get_shared_graph_pool_lock(dev))
                    self._pool_generation += 1
                    for pool in list(self._fwd_pools.values()) + list(self._vjp_pools.values()):
                        with pool.cond:
                            pool.invalidated = True
                            pool.free_slots.clear()
                            pool.total_created = 0
                            pool.cond.notify_all()
                    self._fwd_pools.clear()
                    self._vjp_pools.clear()
                    self._pool_budget_exceeded_keys.clear()
                    self._compiled_cache.clear()
                    self._active_devices.clear()

    @staticmethod
    def _resolve_bucket(
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        sched_feats: torch.Tensor,
        global_feats: torch.Tensor,
        *,
        is_vjp: bool = False,
    ) -> Optional[_FullBucketKey]:
        b_real = int(node_feats.shape[0])
        d_real = int(node_feats.shape[1])
        m_real = int(sched_feats.shape[1])
        if b_real <= 0 or m_real <= 0:
            return None
        if is_vjp:
            b_caps: Tuple[int, ...] = (1, 4) if d_real >= 9 else VJP_BATCH_BUCKETS
        else:
            b_caps = (1, 16) if d_real >= 9 else FWD_BATCH_BUCKETS
        b_bkt = next((b for b in b_caps if b >= b_real), None)
        geom = next(((d, m) for (d, m) in GEOM_BUCKETS if d >= d_real and m >= m_real), None)
        if b_bkt is None or geom is None:
            return None
        return (
            b_bkt,
            geom[0],
            geom[1],
            node_feats.dtype,
            node_feats.device,
            int(node_feats.shape[-1]),
            int(edge_feats.shape[-1]),
            int(sched_feats.shape[-1]),
            int(global_feats.shape[-1]),
        )

    def _get_capture_model_and_pool(
        self,
        model_digest: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[nn.Module, Any]:
        """Must be called with self._capture_lock held; never acquires self._lock."""
        cap_model = GLOBAL_GPU_CACHE_MANAGER.get_canonical_eval_model(
            model_digest, self.model, device, dtype
        )
        graph_pool = GLOBAL_GPU_CACHE_MANAGER.get_shared_graph_pool(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _z = torch.ones((1, 1), device=device, dtype=dtype)
            _ = torch.matmul(_z, _z)
        return cap_model, graph_pool

    def _acquire_slot(
        self,
        pools_dict: Dict[_FullBucketKey, _SlotPool],
        key: _FullBucketKey,
        capture_fn: Callable[[_FullBucketKey, str], Any],
        model_digest: str,
    ) -> Tuple[_SlotPool, Any, int]:
        while True:
            with self._lock:
                if key in self._pool_budget_exceeded_keys:
                    raise _PoolBudgetExceeded("Partition 2 CUDAGraph pool budget reached.")
                self._active_devices.add(key[4])
                gen = self._pool_generation
                pool = pools_dict.get(key)
                if pool is None:
                    pool = _SlotPool()
                    pools_dict[key] = pool

            should_create = False
            with pool.cond:
                while True:
                    if pool.invalidated:
                        break
                    if pool.free_slots:
                        return pool, pool.free_slots.popleft(), gen
                    if pool.total_created < POOL_SLOTS_PER_BUCKET:
                        pool.total_created += 1
                        should_create = True
                        break
                    pool.cond.wait()

            if pool.invalidated:
                continue

            if should_create:
                try:
                    slot = capture_fn(key, model_digest)
                except _PoolBudgetExceeded:
                    with self._lock:
                        self._pool_budget_exceeded_keys.add(key)
                    with pool.cond:
                        if not pool.invalidated and pool.total_created > 0:
                            pool.total_created -= 1
                        pool.cond.notify_all()
                    raise
                except BaseException:
                    with pool.cond:
                        if not pool.invalidated and pool.total_created > 0:
                            pool.total_created -= 1
                        pool.cond.notify_all()
                    raise
                return pool, slot, gen

    def _release_slot(self, pool: _SlotPool, slot: Any, gen: int) -> None:
        with self._lock:
            active = (gen == self._pool_generation) and not pool.invalidated
        if not active:
            return
        with pool.cond:
            if not pool.invalidated:
                pool.free_slots.append(slot)
                pool.cond.notify()

    @staticmethod
    def _replay_with_pool_fence(device: torch.device, caller_stream: Any, graph: Any) -> None:
        """Serialize shared-pool execution under the global device pool lock with CUDA event fences."""
        pool_lock = GLOBAL_GPU_CACHE_MANAGER.get_shared_graph_pool_lock(device)
        with pool_lock:
            GLOBAL_GPU_CACHE_MANAGER.wait_pool_event(device, caller_stream)
            graph.replay()
            GLOBAL_GPU_CACHE_MANAGER.record_pool_event(device, caller_stream)

    def _capture_fwd_slot(self, key: _FullBucketKey, model_digest: str) -> _FwdGraphSlot:
        b_bkt, d_bkt, m_bkt, dtype, device, f_n, f_e, f_s, f_g = key
        dev_key = normalize_cuda_device(device)
        with self._capture_lock, torch.inference_mode(False):
            with GLOBAL_GPU_CACHE_MANAGER._lock:
                cur_pool_bytes = int(GLOBAL_GPU_CACHE_MANAGER._pool_page_bytes.get(dev_key, 0))
            if cur_pool_bytes >= PARTITION_CUDAGRAPH_MAX_BYTES:
                raise _PoolBudgetExceeded("Partition 2 CUDAGraph 40 MB budget exhausted.")

            cap_model, graph_pool = self._get_capture_model_and_pool(model_digest, device, dtype)
            pool_lock = GLOBAL_GPU_CACHE_MANAGER.get_shared_graph_pool_lock(device)
            nf = torch.zeros((b_bkt, d_bkt, f_n), dtype=dtype, device=device)
            ef = torch.zeros((b_bkt, d_bkt, d_bkt, f_e), dtype=dtype, device=device)
            am = torch.ones((b_bkt, d_bkt), dtype=dtype, device=device)
            sf = torch.zeros((b_bkt, m_bkt, f_s), dtype=dtype, device=device)
            sm = torch.ones((b_bkt, m_bkt), dtype=dtype, device=device)
            gf = torch.zeros((b_bkt, f_g), dtype=dtype, device=device)
            out_haz = torch.empty((b_bkt, m_bkt), dtype=dtype, device=device)
            out_cpn = torch.empty((b_bkt, m_bkt), dtype=dtype, device=device)
            out_put = torch.empty((b_bkt,), dtype=dtype, device=device)

            cap_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            cap_stream.wait_stream(torch.cuda.current_stream(device))
            with pool_lock, GLOBAL_GPU_CACHE_MANAGER.tf32_precision_guard(self.use_tf32):
                GLOBAL_GPU_CACHE_MANAGER.wait_pool_event(device, cap_stream)
                _ensure_pool_anchor(device, graph_pool, cap_stream)
                before_stats = torch.cuda.memory_stats(dev_key[1])
                with torch.cuda.stream(cap_stream), torch.no_grad():
                    for _ in range(2):
                        dh, dc, dp = cap_model.forward_residuals(  # type: ignore[operator]
                            nf, ef, am, sf, sm, gf, require_second_order_sdpa=False
                        )
                        out_haz.copy_(dh)
                        out_cpn.copy_(dc)
                        out_put.copy_(dp)
                    graph = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
                    with torch.cuda.graph(graph, pool=graph_pool, stream=cap_stream):
                        dh, dc, dp = cap_model.forward_residuals(  # type: ignore[operator]
                            nf, ef, am, sf, sm, gf, require_second_order_sdpa=False
                        )
                        out_haz.copy_(dh)
                        out_cpn.copy_(dc)
                        out_put.copy_(dp)
                    del dh, dc, dp
                cap_stream.synchronize()
                GLOBAL_GPU_CACHE_MANAGER.record_pool_event(device, cap_stream)
                total_pool_bytes = GLOBAL_GPU_CACHE_MANAGER.account_pool_capture_bytes(
                    device, before_stats
                )
            if total_pool_bytes > PARTITION_CUDAGRAPH_MAX_BYTES:
                del graph
                raise _PoolBudgetExceeded(
                    f"CUDAGraph pool bytes {total_pool_bytes} exceeded 40 MB budget."
                )
            return _FwdGraphSlot(
                graph=graph,
                ev=torch.cuda.Event(),  # type: ignore[no-untyped-call]
                nf=nf,
                ef=ef,
                am=am,
                sf=sf,
                sm=sm,
                gf=gf,
                out_haz=out_haz,
                out_cpn=out_cpn,
                out_put=out_put,
            )

    def _capture_vjp_slot(self, key: _FullBucketKey, model_digest: str) -> _VjpGraphSlot:
        b_bkt, d_bkt, m_bkt, dtype, device, f_n, f_e, f_s, f_g = key
        dev_key = normalize_cuda_device(device)
        with self._capture_lock, torch.inference_mode(False):
            with GLOBAL_GPU_CACHE_MANAGER._lock:
                cur_pool_bytes = int(GLOBAL_GPU_CACHE_MANAGER._pool_page_bytes.get(dev_key, 0))
            if cur_pool_bytes >= PARTITION_CUDAGRAPH_MAX_BYTES:
                raise _PoolBudgetExceeded("Partition 2 CUDAGraph 40 MB budget exhausted.")

            cap_model, graph_pool = self._get_capture_model_and_pool(model_digest, device, dtype)
            pool_lock = GLOBAL_GPU_CACHE_MANAGER.get_shared_graph_pool_lock(device)
            nf = (torch.randn((b_bkt, d_bkt, f_n), dtype=dtype, device=device) * 0.05).requires_grad_(True)
            ef = (torch.randn((b_bkt, d_bkt, d_bkt, f_e), dtype=dtype, device=device) * 0.05).requires_grad_(True)
            am = torch.ones((b_bkt, d_bkt), dtype=dtype, device=device)
            sf = (torch.randn((b_bkt, m_bkt, f_s), dtype=dtype, device=device) * 0.05).requires_grad_(True)
            sm = torch.ones((b_bkt, m_bkt), dtype=dtype, device=device)
            gf = (torch.randn((b_bkt, f_g), dtype=dtype, device=device) * 0.05).requires_grad_(True)
            gh = torch.ones((b_bkt, m_bkt), dtype=dtype, device=device)
            gc = torch.ones((b_bkt, m_bkt), dtype=dtype, device=device)
            gp = torch.ones((b_bkt,), dtype=dtype, device=device)

            out_haz = torch.empty((b_bkt, m_bkt), dtype=dtype, device=device)
            out_cpn = torch.empty((b_bkt, m_bkt), dtype=dtype, device=device)
            out_put = torch.empty((b_bkt,), dtype=dtype, device=device)
            grad_node = torch.empty_like(nf)
            grad_edge = torch.empty_like(ef)
            grad_sched = torch.empty_like(sf)
            grad_global = torch.empty_like(gf)

            def _run_fwd_bwd() -> None:
                with torch.enable_grad():  # type: ignore[no-untyped-call]
                    dh, dc, dp = cap_model.forward_residuals(  # type: ignore[operator]
                        nf, ef, am, sf, sm, gf, require_second_order_sdpa=False
                    )
                    grads = torch.autograd.grad(
                        (dh, dc, dp),
                        (nf, ef, sf, gf),
                        grad_outputs=(gh, gc, gp),
                        allow_unused=True,
                    )
                with torch.no_grad():
                    out_haz.copy_(dh)
                    out_cpn.copy_(dc)
                    out_put.copy_(dp)
                    if grads[0] is not None:
                        grad_node.copy_(grads[0])
                    else:
                        grad_node.zero_()
                    if grads[1] is not None:
                        grad_edge.copy_(grads[1])
                    else:
                        grad_edge.zero_()
                    if grads[2] is not None:
                        grad_sched.copy_(grads[2])
                    else:
                        grad_sched.zero_()
                    if grads[3] is not None:
                        grad_global.copy_(grads[3])
                    else:
                        grad_global.zero_()

            cap_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            cap_stream.wait_stream(torch.cuda.current_stream(device))
            with pool_lock, GLOBAL_GPU_CACHE_MANAGER.tf32_precision_guard(self.use_tf32):
                GLOBAL_GPU_CACHE_MANAGER.wait_pool_event(device, cap_stream)
                _ensure_pool_anchor(device, graph_pool, cap_stream)
                before_stats = torch.cuda.memory_stats(dev_key[1])
                with torch.cuda.stream(cap_stream):
                    for _ in range(2):
                        _run_fwd_bwd()
                    graph = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
                    with torch.cuda.graph(graph, pool=graph_pool, stream=cap_stream):
                        _run_fwd_bwd()
                cap_stream.synchronize()
                GLOBAL_GPU_CACHE_MANAGER.record_pool_event(device, cap_stream)
                total_pool_bytes = GLOBAL_GPU_CACHE_MANAGER.account_pool_capture_bytes(
                    device, before_stats
                )
            if total_pool_bytes > PARTITION_CUDAGRAPH_MAX_BYTES:
                del graph
                raise _PoolBudgetExceeded(
                    f"CUDAGraph pool bytes {total_pool_bytes} exceeded 40 MB budget."
                )
            return _VjpGraphSlot(
                graph=graph,
                ev=torch.cuda.Event(),  # type: ignore[no-untyped-call]
                nf=nf,
                ef=ef,
                am=am,
                sf=sf,
                sm=sm,
                gf=gf,
                gh=gh,
                gc=gc,
                gp=gp,
                out_haz=out_haz,
                out_cpn=out_cpn,
                out_put=out_put,
                grad_node=grad_node,
                grad_edge=grad_edge,
                grad_sched=grad_sched,
                grad_global=grad_global,
            )

    @staticmethod
    def _copy_inputs_padded(
        slot: Any,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> None:
        b_real, d_real = int(node_feats.shape[0]), int(node_feats.shape[1])
        m_real = int(sched_feats.shape[1])
        b_bkt, d_bkt = int(slot.nf.shape[0]), int(slot.nf.shape[1])
        m_bkt = int(slot.sf.shape[1])
        with torch.no_grad():
            if b_real < b_bkt or d_real < d_bkt:
                slot.nf.zero_()
                slot.ef.zero_()
                slot.am.zero_()
            if b_real < b_bkt or m_real < m_bkt:
                slot.sf.zero_()
                slot.sm.zero_()
            if b_real < b_bkt:
                slot.gf.zero_()
            slot.nf[:b_real, :d_real].copy_(node_feats.detach())
            slot.ef[:b_real, :d_real, :d_real].copy_(edge_feats.detach())
            slot.am[:b_real, :d_real].copy_(asset_mask.detach())
            slot.sf[:b_real, :m_real].copy_(sched_feats.detach())
            slot.sm[:b_real, :m_real].copy_(sched_mask.detach())
            slot.gf[:b_real].copy_(global_feats.detach())

    def _get_eval_residuals_fn(
        self, model_digest: str, device: torch.device, dtype: torch.dtype
    ) -> _ResidualsFn:
        with torch.inference_mode(False):
            eval_model = GLOBAL_GPU_CACHE_MANAGER.get_canonical_eval_model(
                model_digest, self.model, device, dtype
            )
            if not self.enable_inductor:
                return cast(_ResidualsFn, cast(Any, eval_model).forward_residuals)
            c_key = (model_digest, normalize_cuda_device(device), dtype)
            with self._lock:
                fn = self._compiled_cache.get(c_key)
                if fn is None:
                    fn = cast(
                        _ResidualsFn,
                        torch.compile(cast(Any, eval_model).forward_residuals, fullgraph=True),
                    )
                    self._compiled_cache[c_key] = fn
                return fn

    def _run_eager_fwd(
        self,
        model_digest: str,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = node_feats.device
        dtype = node_feats.dtype
        guard = (
            GLOBAL_GPU_CACHE_MANAGER.tf32_precision_guard(self.use_tf32)
            if node_feats.is_cuda
            else nullcontext()
        )
        with guard:
            if torch.is_grad_enabled():
                ref_p: Optional[torch.Tensor] = next(self.model.parameters(), None)
                if ref_p is None:
                    ref_p = next(self.model.buffers(), None)
                if ref_p is None or (ref_p.device == dev and ref_p.dtype == dtype):
                    fn = cast(_ResidualsFn, cast(Any, self.model).forward_residuals)
                else:
                    with torch.inference_mode(False):
                        eval_model = GLOBAL_GPU_CACHE_MANAGER.get_canonical_eval_model(
                            model_digest, self.model, dev, dtype
                        )
                    fn = cast(_ResidualsFn, cast(Any, eval_model).forward_residuals)
            else:
                b_real, d_real = int(node_feats.shape[0]), int(node_feats.shape[1])
                m_real = int(sched_feats.shape[1])
                if self.enable_inductor and b_real > 0 and d_real > 0 and m_real > 0:
                    fn = self._get_eval_residuals_fn(model_digest, dev, dtype)
                else:
                    with torch.inference_mode(False):
                        eval_model = GLOBAL_GPU_CACHE_MANAGER.get_canonical_eval_model(
                            model_digest, self.model, dev, dtype
                        )
                    fn = cast(_ResidualsFn, cast(Any, eval_model).forward_residuals)
            dh, dc, dp = fn(
                node_feats,
                edge_feats,
                asset_mask,
                sched_feats,
                sched_mask,
                global_feats,
                require_second_order_sdpa=False,
            )
            return dh.clone(), dc.clone(), dp.clone()

    def predict_residuals(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward residual prediction with K=4 event-synced bucketed CUDAGraph slots."""
        model_digest = self._sync_model_version()
        b_real, d_real = int(node_feats.shape[0]), int(node_feats.shape[1])
        m_real = int(sched_feats.shape[1])

        if (
            not self.enable_cudagraphs
            or not node_feats.is_cuda
            or torch.is_grad_enabled()
            or b_real <= 0
            or d_real <= 0
            or m_real <= 0
        ):
            return self._run_eager_fwd(
                model_digest,
                node_feats,
                edge_feats,
                asset_mask,
                sched_feats,
                sched_mask,
                global_feats,
            )

        max_fwd_b = 16 if d_real >= 9 else max(FWD_BATCH_BUCKETS)
        if b_real > max_fwd_b and any(d >= d_real and m >= m_real for (d, m) in GEOM_BUCKETS):
            dh_parts, dc_parts, dp_parts = [], [], []
            for start in range(0, b_real, max_fwd_b):
                end = min(b_real, start + max_fwd_b)
                dh_c, dc_c, dp_c = self.predict_residuals(
                    node_feats[start:end],
                    edge_feats[start:end],
                    asset_mask[start:end],
                    sched_feats[start:end],
                    sched_mask[start:end],
                    global_feats[start:end],
                )
                dh_parts.append(dh_c)
                dc_parts.append(dc_c)
                dp_parts.append(dp_c)
            return torch.cat(dh_parts, dim=0), torch.cat(dc_parts, dim=0), torch.cat(dp_parts, dim=0)

        key = self._resolve_bucket(node_feats, edge_feats, sched_feats, global_feats, is_vjp=False)
        if key is None:
            return self._run_eager_fwd(
                model_digest,
                node_feats,
                edge_feats,
                asset_mask,
                sched_feats,
                sched_mask,
                global_feats,
            )

        try:
            pool, slot, gen = self._acquire_slot(
                self._fwd_pools, key, self._capture_fwd_slot, model_digest
            )
        except _PoolBudgetExceeded:
            return self._run_eager_fwd(
                model_digest,
                node_feats,
                edge_feats,
                asset_mask,
                sched_feats,
                sched_mask,
                global_feats,
            )

        caller_stream = torch.cuda.current_stream(node_feats.device)
        try:
            caller_stream.wait_event(slot.ev)
            self._copy_inputs_padded(
                slot,
                node_feats,
                edge_feats,
                asset_mask,
                sched_feats,
                sched_mask,
                global_feats,
            )
            self._replay_with_pool_fence(node_feats.device, caller_stream, slot.graph)
            return (
                slot.out_haz[:b_real, :m_real].clone(),
                slot.out_cpn[:b_real, :m_real].clone(),
                slot.out_put[:b_real].clone(),
            )
        finally:
            slot.ev.record(caller_stream)
            self._release_slot(pool, slot, gen)

    def predict_residuals_and_vjp(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
        g_haz: torch.Tensor,
        g_cpn: torch.Tensor,
        g_put: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run joint forward residuals + first-order VJP using K=4 event-synced CUDAGraph slots."""
        model_digest = self._sync_model_version()
        b_real, d_real = int(node_feats.shape[0]), int(node_feats.shape[1])
        m_real = int(sched_feats.shape[1])

        def _run_fallback() -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            guard = (
                GLOBAL_GPU_CACHE_MANAGER.tf32_precision_guard(self.use_tf32)
                if node_feats.is_cuda
                else nullcontext()
            )
            with guard, torch.inference_mode(False), torch.enable_grad():  # type: ignore[no-untyped-call]
                eval_model = GLOBAL_GPU_CACHE_MANAGER.get_canonical_eval_model(
                    model_digest, self.model, node_feats.device, node_feats.dtype
                )
                nf_r = node_feats.detach().clone().requires_grad_(True)
                ef_r = edge_feats.detach().clone().requires_grad_(True)
                sf_r = sched_feats.detach().clone().requires_grad_(True)
                gf_r = global_feats.detach().clone().requires_grad_(True)
                dh, dc, dp = eval_model.forward_residuals(  # type: ignore[operator]
                    nf_r,
                    ef_r,
                    asset_mask.detach().clone(),
                    sf_r,
                    sched_mask.detach().clone(),
                    gf_r,
                    require_second_order_sdpa=False,
                )
                out_pairs = [
                    (out_t, grad_t)
                    for out_t, grad_t in (
                        (dh, g_haz.detach().clone()),
                        (dc, g_cpn.detach().clone()),
                        (dp, g_put.detach().clone()),
                    )
                    if out_t.requires_grad
                ]
                grads: Tuple[Optional[torch.Tensor], ...]
                if out_pairs:
                    grads = torch.autograd.grad(
                        [p[0] for p in out_pairs],
                        (nf_r, ef_r, sf_r, gf_r),
                        grad_outputs=[p[1] for p in out_pairs],
                        allow_unused=True,
                    )
                else:
                    grads = (None, None, None, None)
            gn = grads[0].detach().clone() if grads[0] is not None else torch.zeros_like(nf_r)
            ge = grads[1].detach().clone() if grads[1] is not None else torch.zeros_like(ef_r)
            gs = grads[2].detach().clone() if grads[2] is not None else torch.zeros_like(sf_r)
            gg = grads[3].detach().clone() if grads[3] is not None else torch.zeros_like(gf_r)
            return dh.detach().clone(), dc.detach().clone(), dp.detach().clone(), gn, ge, gs, gg

        if (
            not self.enable_cudagraphs
            or not node_feats.is_cuda
            or b_real <= 0
            or d_real <= 0
            or m_real <= 0
        ):
            return _run_fallback()

        max_vjp_b = 4 if d_real >= 9 else max(VJP_BATCH_BUCKETS)
        if b_real > max_vjp_b and any(d >= d_real and m >= m_real for (d, m) in GEOM_BUCKETS):
            chunks = [
                self.predict_residuals_and_vjp(
                    node_feats[s:e],
                    edge_feats[s:e],
                    asset_mask[s:e],
                    sched_feats[s:e],
                    sched_mask[s:e],
                    global_feats[s:e],
                    g_haz[s:e],
                    g_cpn[s:e],
                    g_put[s:e],
                )
                for s in range(0, b_real, max_vjp_b)
                for e in (min(b_real, s + max_vjp_b),)
            ]
            return (
                torch.cat([c[0] for c in chunks], dim=0),
                torch.cat([c[1] for c in chunks], dim=0),
                torch.cat([c[2] for c in chunks], dim=0),
                torch.cat([c[3] for c in chunks], dim=0),
                torch.cat([c[4] for c in chunks], dim=0),
                torch.cat([c[5] for c in chunks], dim=0),
                torch.cat([c[6] for c in chunks], dim=0),
            )

        key = self._resolve_bucket(node_feats, edge_feats, sched_feats, global_feats, is_vjp=True)
        if key is None:
            return _run_fallback()

        try:
            pool, slot, gen = self._acquire_slot(
                self._vjp_pools, key, self._capture_vjp_slot, model_digest
            )
        except _PoolBudgetExceeded:
            return _run_fallback()

        caller_stream = torch.cuda.current_stream(node_feats.device)
        b_bkt = int(slot.nf.shape[0])
        m_bkt = int(slot.sf.shape[1])
        try:
            caller_stream.wait_event(slot.ev)
            self._copy_inputs_padded(
                slot,
                node_feats,
                edge_feats,
                asset_mask,
                sched_feats,
                sched_mask,
                global_feats,
            )
            with torch.no_grad():
                if b_real < b_bkt or m_real < m_bkt:
                    slot.gh.zero_()
                    slot.gc.zero_()
                if b_real < b_bkt:
                    slot.gp.zero_()
                slot.gh[:b_real, :m_real].copy_(g_haz.detach())
                slot.gc[:b_real, :m_real].copy_(g_cpn.detach())
                slot.gp[:b_real].copy_(g_put.detach())
            self._replay_with_pool_fence(node_feats.device, caller_stream, slot.graph)
            return (
                slot.out_haz[:b_real, :m_real].clone(),
                slot.out_cpn[:b_real, :m_real].clone(),
                slot.out_put[:b_real].clone(),
                slot.grad_node[:b_real, :d_real].clone(),
                slot.grad_edge[:b_real, :d_real, :d_real].clone(),
                slot.grad_sched[:b_real, :m_real].clone(),
                slot.grad_global[:b_real].clone(),
            )
        finally:
            slot.ev.record(caller_stream)
            self._release_slot(pool, slot, gen)


__all__ = [
    "BATCH_BUCKETS",
    "FWD_BATCH_BUCKETS",
    "VJP_BATCH_BUCKETS",
    "PARTITION_CUDAGRAPH_MAX_BYTES",
    "GEOM_BUCKETS",
    "POOL_SLOTS_PER_BUCKET",
    "CompiledSurrogateRunner",
]
