from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import copy
import gc
import threading
from typing import Any, Callable, Dict, Generic, Iterator, List, Mapping, Optional, Set, Tuple, TypeVar

import torch
import torch.nn as nn

_V = TypeVar("_V")
_PRIMITIVE_TYPES = (int, float, str, bytes, bool, type(None))


def normalize_cuda_device(device: torch.device | str) -> Tuple[str, int]:
    """Map 'cuda' and explicit 'cuda:idx' to the identical normalized (type, index) tuple."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return (dev.type, -1)
    if dev.index is not None:
        return ("cuda", int(dev.index))
    idx = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    return ("cuda", idx)


def _walk_unique_tensors(obj: Any, seen: Set[int], tensors_out: List[torch.Tensor]) -> None:
    """Collect unique tensors reachable through nested containers, mappings, and dataclasses."""
    stack: List[Any] = [obj]
    while stack:
        cur = stack.pop()
        if type(cur) in _PRIMITIVE_TYPES:
            continue
        if isinstance(cur, torch.Tensor):
            oid = id(cur)
            if oid not in seen:
                seen.add(oid)
                tensors_out.append(cur)
            continue
        oid = id(cur)
        if oid in seen:
            continue
        if isinstance(cur, (tuple, list)):
            seen.add(oid)
            stack.extend(cur)
        elif isinstance(cur, dict):
            seen.add(oid)
            stack.extend(cur.values())
        else:
            dc_fields = getattr(cur, "__dataclass_fields__", None)
            if dc_fields is not None and not isinstance(cur, type):
                seen.add(oid)
                for name in dc_fields:
                    stack.append(getattr(cur, name, None))
            elif hasattr(cur, "values") and isinstance(cur, Mapping):
                seen.add(oid)
                stack.extend(cur.values())


def measure_tensor_nbytes(obj: Any, _seen: Optional[Set[int]] = None) -> int:
    """Recursively count unique GPU/CPU tensor bytes across nested containers and dataclasses."""
    if type(obj) in _PRIMITIVE_TYPES:
        return 0
    if isinstance(obj, torch.Tensor):
        if _seen is not None:
            oid = id(obj)
            if oid in _seen:
                return 0
            _seen.add(oid)
        return int(obj.numel() * obj.element_size())
    seen = set() if _seen is None else _seen
    tensors: List[torch.Tensor] = []
    _walk_unique_tensors(obj, seen, tensors)
    return sum(int(t.numel() * t.element_size()) for t in tensors)


def _extract_cuda_tensors(obj: Any) -> Tuple[torch.Tensor, ...]:
    """Extract unique CUDA tensors once at cache insertion to avoid hit-time tree walks."""
    if type(obj) in _PRIMITIVE_TYPES:
        return ()
    if isinstance(obj, torch.Tensor):
        return (obj,) if obj.is_cuda else ()
    tensors: List[torch.Tensor] = []
    _walk_unique_tensors(obj, set(), tensors)
    return tuple(t for t in tensors if t.is_cuda)


def _record_cuda_tensors_stream(
    cuda_tensors: Tuple[torch.Tensor, ...],
    stream: Any = None,
) -> None:
    """Record active CUDA streams on cached CUDA tensors with one current_stream lookup per device."""
    if not cuda_tensors:
        return
    if stream is not None:
        for t in cuda_tensors:
            t.record_stream(stream)
        return
    dev0 = cuda_tensors[0].device
    stream0 = torch.cuda.current_stream(dev0)
    if len(cuda_tensors) == 1:
        cuda_tensors[0].record_stream(stream0)
        return
    extra_streams: Optional[Dict[torch.device, Any]] = None
    for t in cuda_tensors:
        t_dev = t.device
        if t_dev == dev0:
            t.record_stream(stream0)
        else:
            if extra_streams is None:
                extra_streams = {}
            st_dev = extra_streams.get(t_dev)
            if st_dev is None:
                st_dev = torch.cuda.current_stream(t_dev)
                extra_streams[t_dev] = st_dev
            t.record_stream(st_dev)


def record_obj_cuda_stream(obj: Any, stream: Any = None, _seen: Optional[Set[int]] = None) -> None:
    """Record active CUDA stream on cached tensors before reuse or LRU eviction."""
    if type(obj) in _PRIMITIVE_TYPES:
        return
    if isinstance(obj, torch.Tensor):
        if not obj.is_cuda:
            return
        if _seen is not None:
            oid = id(obj)
            if oid in _seen:
                return
            _seen.add(oid)
        target = torch.cuda.current_stream(obj.device) if stream is None else stream
        obj.record_stream(target)
        return
    seen = set() if _seen is None else _seen
    tensors: List[torch.Tensor] = []
    _walk_unique_tensors(obj, seen, tensors)
    cuda_tensors = tuple(t for t in tensors if t.is_cuda)
    _record_cuda_tensors_stream(cuda_tensors, stream=stream)


class PartitionedGpuLRU(Generic[_V]):
    """Thread-safe byte-budgeted and entry-capped LRU partition backed by GlobalGpuCacheManager."""

    def __init__(
        self,
        name: str,
        *,
        max_entries: int,
        max_bytes: int,
        on_evict: Optional[Callable[[Any, _V], None]] = None,
    ) -> None:
        self.name = str(name)
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self._on_evict = on_evict
        self._lock = threading.RLock()
        self._store: OrderedDict[Any, Tuple[_V, int, Tuple[torch.Tensor, ...]]] = OrderedDict()
        self._resident_bytes: int = 0

    def get(self, key: Any) -> Optional[_V]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            self._store.move_to_end(key)
            val, _, cuda_tensors = entry
            if cuda_tensors:
                _record_cuda_tensors_stream(cuda_tensors)
            return val

    def put(self, key: Any, val: _V, *, explicit_bytes: Optional[int] = None) -> _V:
        nb = int(explicit_bytes) if explicit_bytes is not None else measure_tensor_nbytes(val)
        cuda_tensors = _extract_cuda_tensors(val)
        with self._lock:
            if key in self._store:
                old_val, old_nb, old_cuda = self._store.pop(key)
                self._resident_bytes -= old_nb
                if old_cuda:
                    _record_cuda_tensors_stream(old_cuda)
                if self._on_evict is not None and old_val is not val:
                    self._on_evict(key, old_val)
            if nb > self.max_bytes:
                # Reject oversized entries so a single tensor never exceeds the partition budget.
                return val
            while self._store and (
                len(self._store) >= self.max_entries or (self._resident_bytes + nb) > self.max_bytes
            ):
                ev_k, (ev_v, ev_nb, ev_cuda) = self._store.popitem(last=False)
                self._resident_bytes -= ev_nb
                if ev_cuda:
                    _record_cuda_tensors_stream(ev_cuda)
                if self._on_evict is not None:
                    self._on_evict(ev_k, ev_v)
            self._store[key] = (val, nb, cuda_tensors)
            self._resident_bytes += nb
            if cuda_tensors:
                _record_cuda_tensors_stream(cuda_tensors)
            return val

    def remove(self, key: Any) -> bool:
        with self._lock:
            entry = self._store.pop(key, None)
            if entry is None:
                return False
            ev_v, ev_nb, ev_cuda = entry
            self._resident_bytes -= ev_nb
            if ev_cuda:
                _record_cuda_tensors_stream(ev_cuda)
            if self._on_evict is not None:
                self._on_evict(key, ev_v)
            return True

    def clear(self) -> None:
        with self._lock:
            for k, (v, _, v_cuda) in list(self._store.items()):
                if v_cuda:
                    _record_cuda_tensors_stream(v_cuda)
                if self._on_evict is not None:
                    self._on_evict(k, v)
            self._store.clear()
            self._resident_bytes = 0

    @property
    def resident_bytes(self) -> int:
        with self._lock:
            return int(self._resident_bytes)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class GlobalGpuCacheManager:
    """Global <= 192 MB GPU cache registry, shared CUDAGraph pool, and canonical eval model store."""

    TOTAL_BUDGET_BYTES: int = 192 * 1024 * 1024
    PARTITION_BUDGETS: Mapping[str, int] = {
        "static_constants": 16 * 1024 * 1024,
        "cudagraph_pool": 40 * 1024 * 1024,
        "triton_kernels": 24 * 1024 * 1024,
        "sobol_normals": 24 * 1024 * 1024,
        "deal_prep_lv_vega": 88 * 1024 * 1024,
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tf32_lock = threading.Lock()
        self._partitions: Dict[str, PartitionedGpuLRU[Any]] = {}
        self._graph_pools: Dict[Tuple[str, int], Any] = {}
        self._pool_locks: Dict[Tuple[str, int], threading.Lock] = {}
        self._pool_events: Dict[Tuple[str, int], Optional[Any]] = {}
        self._pool_page_bytes: Dict[Tuple[str, int], int] = {}
        self._canonical_models: Dict[Tuple[str, Tuple[str, int], torch.dtype], Tuple[int, nn.Module]] = {}

    def register_partition(
        self,
        name: str,
        *,
        max_entries: int,
        max_bytes: int,
        on_evict: Optional[Callable[[Any, Any], None]] = None,
    ) -> PartitionedGpuLRU[Any]:
        with self._lock:
            existing = self._partitions.get(name)
            if existing is not None:
                return existing
            part: PartitionedGpuLRU[Any] = PartitionedGpuLRU(
                name,
                max_entries=max_entries,
                max_bytes=max_bytes,
                on_evict=on_evict,
            )
            self._partitions[name] = part
            return part

    def get_shared_graph_pool(self, device: torch.device | str) -> Any:
        dev_key = normalize_cuda_device(device)
        with self._lock:
            if dev_key not in self._graph_pools:
                if dev_key[0] == "cuda":
                    with torch.cuda.device(dev_key[1]):
                        pool = torch.cuda.graph_pool_handle()  # type: ignore[no-untyped-call]
                else:
                    pool = None
                self._graph_pools[dev_key] = pool
                self._pool_locks[dev_key] = threading.Lock()
                self._pool_events[dev_key] = None
                self._pool_page_bytes[dev_key] = 0
            return self._graph_pools[dev_key]

    def get_shared_graph_pool_lock(self, device: torch.device | str) -> threading.Lock:
        dev_key = normalize_cuda_device(device)
        self.get_shared_graph_pool(device)
        with self._lock:
            return self._pool_locks[dev_key]

    def record_pool_event(self, device: torch.device | str, stream: Any) -> Any:
        dev_key = normalize_cuda_device(device)
        if dev_key[0] != "cuda" or stream is None:
            return None
        ev = torch.cuda.Event(enable_timing=False)  # type: ignore[no-untyped-call]
        ev.record(stream)  # type: ignore[no-untyped-call]
        with self._lock:
            self._pool_events[dev_key] = ev
        return ev

    def wait_pool_event(self, device: torch.device | str, stream: Any) -> None:
        if stream is None:
            return
        dev_key = normalize_cuda_device(device)
        with self._lock:
            ev = self._pool_events.get(dev_key)
        if ev is not None:
            stream.wait_event(ev)

    def account_pool_capture_bytes(self, device: torch.device | str, before_stats: Mapping[str, int]) -> int:
        """Measure actual CUDAGraph allocator pool page growth from memory_stats()."""
        dev_key = normalize_cuda_device(device)
        if dev_key[0] != "cuda" or not torch.cuda.is_available():
            return 0
        after_stats = torch.cuda.memory_stats(dev_key[1])
        delta = max(
            0,
            int(after_stats.get("allocated_bytes.pool.current", 0))
            - int(before_stats.get("allocated_bytes.pool.current", 0)),
        )
        with self._lock:
            self._pool_page_bytes[dev_key] = self._pool_page_bytes.get(dev_key, 0) + delta
            return self._pool_page_bytes[dev_key]

    def get_canonical_eval_model(
        self,
        model_digest: str,
        surrogate: nn.Module,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> nn.Module:
        """Return one shared parameterless-grad eval mirror across Runner and Greeks."""
        dev_key = normalize_cuda_device(device)
        m_key = (str(model_digest), dev_key, dtype)
        ver = sum(
            (0 if p.is_inference() else int(p._version)) for p in surrogate.parameters()
        ) + sum((0 if b.is_inference() else int(b._version)) for b in surrogate.buffers())
        with self._lock:
            cached = self._canonical_models.get(m_key)
            if cached is not None and cached[0] == ver:
                return cached[1]
            target_dev = torch.device(dev_key[0], dev_key[1]) if dev_key[0] == "cuda" else torch.device("cpu")
            with torch.inference_mode(False), torch.no_grad():
                mirror = copy.deepcopy(surrogate)
                # Ensure inference tensors cloned under inference_mode never break .eval() buffer freezes.
                for mod in mirror.modules():
                    for p_name, p_val in list(mod._parameters.items()):
                        if p_val is not None:
                            t_p = p_val.detach().clone() if p_val.is_inference() else p_val.detach()
                            mod._parameters[p_name] = nn.Parameter(t_p, requires_grad=False)
                    for b_name, b_val in list(mod._buffers.items()):
                        if b_val is not None and b_val.is_inference():
                            mod._buffers[b_name] = b_val.detach().clone()
                mirror = mirror.to(device=target_dev, dtype=dtype).eval()
                for p in mirror.parameters():
                    p.requires_grad_(False)
            self._canonical_models[m_key] = (ver, mirror)
            return mirror

    @contextmanager
    def tf32_precision_guard(self, enabled: bool = False) -> Iterator[None]:
        """Serialize global matmul and cuDNN TF32 flags and restore prior state on exit."""
        with self._tf32_lock:
            prev_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
            prev_cudnn = bool(torch.backends.cudnn.allow_tf32)
            prev_prec = str(torch.get_float32_matmul_precision())
            try:
                torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
                torch.backends.cudnn.allow_tf32 = bool(enabled)
                torch.set_float32_matmul_precision("high" if enabled else "highest")
                yield
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev_matmul
                torch.backends.cudnn.allow_tf32 = prev_cudnn
                torch.set_float32_matmul_precision(prev_prec)

    def total_resident_bytes(self) -> int:
        with self._lock:
            part_bytes = sum(p.resident_bytes for p in self._partitions.values())
            pool_bytes = sum(self._pool_page_bytes.values())
            return int(part_bytes + pool_bytes)

    def partition_stats(self) -> Dict[str, Tuple[int, int]]:
        with self._lock:
            return {name: (len(p), p.resident_bytes) for name, p in self._partitions.items()}

    def clear_all(self, *, clear_canonical_models: bool = False) -> None:
        """Clear all registered GPU LRU partitions, optional canonical models, and release CUDA allocator cache."""
        cuda_ok = bool(torch.cuda.is_available() and torch.cuda.is_initialized())  # type: ignore[no-untyped-call]
        if cuda_ok:
            torch.cuda.synchronize()
        with self._lock:
            for part in list(self._partitions.values()):
                part.clear()
            for dev_key in list(self._pool_events.keys()):
                self._pool_events[dev_key] = None
            if clear_canonical_models:
                self._canonical_models.clear()
        gc.collect()
        if cuda_ok:
            torch.cuda.empty_cache()
        with self._lock:
            for dev_key in list(self._pool_page_bytes.keys()):
                if dev_key[0] == "cuda" and cuda_ok:
                    stats = torch.cuda.memory_stats(dev_key[1])
                    self._pool_page_bytes[dev_key] = int(stats.get("allocated_bytes.pool.current", 0))
                else:
                    self._pool_page_bytes[dev_key] = 0


GLOBAL_GPU_CACHE_MANAGER = GlobalGpuCacheManager()
