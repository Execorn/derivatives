"""Tier 1 deterministic float64 pre- and post-processor for EQAutocallable."""

from __future__ import annotations

from array import array
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass, field, fields as dc_fields, replace as dc_replace
import datetime
from enum import Enum
import hashlib
import inspect
import math
from pathlib import Path
import re
import struct
import threading
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
import torch

from deepvol.api.schemas_eq_autocall import (
    DiscountFactorCurveSpec,
    DividendsSpec,
    EQAutocallableTradeRequest,
    EquityVolSmileCollectionSpec,
    FixingsSpec,
    FXVolatilitySmileCollectionSpec,
    MarketDataSetSpec,
    RepoSpreadCurveSpec,
    parse_murex_autocall_dag,
)
from deepvol.market.eq_curves import (
    ClampedYieldCurve,
    RepoCurve,
    year_fraction,
    year_fractions,
)
from deepvol.models.autocall.analytical_baseline import (
    AutocallBaselineOutput,
    compute_analytical_autocall_baseline,
)
from deepvol.models.autocall.dupire_qmc_solver import RegularizedVolSurface
from deepvol.models.autocall.escrowed_dividend import (
    EscrowedDividendOutput,
    build_discount_rate_evaluator,
    compute_escrowed_dividend_forwards,
    equity_forward_curve_priority as _curve_ois_priority_rank,
    infer_asset_currency_from_context,
)
from deepvol.models.autocall.feature_layout import (
    D_MAX_DEFAULT,
    FEATURE_SCHEMA_VERSION,
    M_MAX_DEFAULT,
    PAIR_ORDER_DEPENDENT_INDICES,
    AutocallEdgeCol,
    AutocallGlobalCol,
    AutocallNodeCol,
    AutocallSchedCol,
    MonitoringIndexCode,
    monitoring_index_to_code,
)
from deepvol.models.autocall.seasoned_collapser import (
    CashFlowItem,
    SeasonedScheduleState,
    _to_utc_datetime,
    collapse_seasoned_schedule,
)

_BASELINE_SIG_PARAMS: frozenset[str] = frozenset(
    inspect.signature(compute_analytical_autocall_baseline).parameters.keys()
)
_VOL_SURF_SIG_PARAMS: frozenset[str] = frozenset(
    inspect.signature(RegularizedVolSurface.__init__).parameters.keys()
)
_HAS_IV_BATCH: bool = hasattr(RegularizedVolSurface, "iv_batch") and callable(
    getattr(RegularizedVolSurface, "iv_batch")
)


class _LazySurfaceJacobianDict(dict[str, npt.NDArray[np.float64]]):
    """Lazily computes and caches surface quote Jacobians only if accessed."""

    def __init__(
        self,
        builder: Callable[[str], npt.NDArray[np.float64]],
        keys: Sequence[str],
        *,
        _shared_cache: dict[str, npt.NDArray[np.float64]] | None = None,
        _lock: threading.RLock | None = None,
    ) -> None:
        super().__init__()
        self._builder = builder
        self._keys = list(keys)
        self._shared_cache: dict[str, npt.NDArray[np.float64]] = (
            _shared_cache if _shared_cache is not None else {}
        )
        self._lock: threading.RLock = _lock if _lock is not None else threading.RLock()

    def _get_or_build(self, key: str) -> npt.NDArray[np.float64]:
        with self._lock:
            cached = self._shared_cache.get(key)
            if cached is None:
                built = np.asarray(self._builder(key), dtype=np.float64).copy()
                built.setflags(write=False)
                self._shared_cache[key] = built
                cached = built
            return cached.copy()

    def _populate_all(self) -> None:
        with self._lock:
            for k in self._keys:
                if not super().__contains__(k):
                    super().__setitem__(k, self._get_or_build(k))

    def __getitem__(self, key: str) -> npt.NDArray[np.float64]:
        with self._lock:
            if not super().__contains__(key) and key in self._keys:
                super().__setitem__(key, self._get_or_build(key))
            return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._keys:
                return self[key]
            return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._keys or super().__contains__(key)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Any:
        with self._lock:
            self._populate_all()
            return iter(list(super().__iter__()))

    def keys(self) -> Any:
        with self._lock:
            self._populate_all()
            return dict(super().items()).keys()

    def values(self) -> Any:
        with self._lock:
            self._populate_all()
            return dict(super().items()).values()

    def items(self) -> Any:
        with self._lock:
            self._populate_all()
            return dict(super().items()).items()

    def clone(self) -> _LazySurfaceJacobianDict:
        cloned = _LazySurfaceJacobianDict(
            self._builder,
            self._keys,
            _shared_cache=self._shared_cache,
            _lock=self._lock,
        )
        with self._lock:
            for k, v in dict.items(self):
                dict.__setitem__(cloned, k, v.copy())
        return cloned

    def __deepcopy__(self, memo: Any) -> _LazySurfaceJacobianDict:
        _ = memo
        return self.clone()


@dataclass
class AutocallPreprocState:
    """Complete Tier 1 pre-processed state in IEEE-754 float64."""

    # Padded feature tensors in torch.float64: (B, D_max, 16), (B, D_max, D_max, 4), (B, M_max, 12), (B, 16)
    node_feats: torch.Tensor
    edge_feats: torch.Tensor
    asset_mask: torch.Tensor
    sched_feats: torch.Tensor
    sched_mask: torch.Tensor
    global_feats: torch.Tensor

    # Dimensions & canonical ordering
    d_assets: int
    m_rem: int
    d_max: int
    m_max: int
    original_identifiers: list[str]
    sorted_identifiers: list[str]
    lex_perm: npt.NDArray[np.int64]
    inv_lex_perm: npt.NDArray[np.int64]

    # Active unpadded float64 tensors for analytical baseline, Greeks, and W1 post-processor
    spots_t0: torch.Tensor
    ref_spots: torch.Tensor
    moneyness_t0: torch.Tensor
    corr_matrix: torch.Tensor
    tau_obs: torch.Tensor
    delta_tau: torch.Tensor
    tau_pay: torch.Tensor
    df_obs: torch.Tensor
    df_pay: torch.Tensor
    ko_barriers: torch.Tensor
    is_callable: torch.Tensor
    cpn_barriers: torch.Tensor
    is_guaranteed_cpn: torch.Tensor
    coupon_amounts: torch.Tensor
    forward_rel: torch.Tensor
    forward_perf: torch.Tensor
    cash_div_leverage: torch.Tensor
    iv_at_fwd: torch.Tensor
    v_perf: torch.Tensor

    # Sub-engine states & Jacobians
    seasoned_state: SeasonedScheduleState
    dividend_schedule: EscrowedDividendOutput
    vol_surfaces: dict[str, RegularizedVolSurface]
    surface_feature_jacobians: dict[str, npt.NDArray[np.float64]]
    iv_at_fwd_jacobians: dict[str, npt.NDArray[np.float64]]
    discount_curve: Any
    baseline: AutocallBaselineOutput

    # Product payoff & contract terms
    strike: float = 0.90
    option_gearing: float = 1.11111
    lower_strike: float | None = None
    ki_barrier: float | None = None
    i_ki_past: float = 0.0
    c_mem_0: float = 0.0
    is_memory: bool = True
    ko_spread: float = -0.15
    cpn_spread: float = -0.15
    ki_spread: float = -0.15
    no_notional_repayment: bool = False
    autocall_coupon: float = 0.0
    force_coupons_on_autocall: bool = False
    allow_autocall_on_expiry: bool = False
    participation_call_strike: float | None = None
    participation_call_rate: float = 0.0
    one_star_strike: float | None = None
    one_star_coupon: float = 0.0
    is_long_basket: bool = False
    payment_currency: str = "RUB"
    callability_index: str = "WorstOf"
    coupon_index: str | None = None
    option_index: str | None = None
    funding_leg: dict[str, Any] | None = None
    pv_in_transit: float = 0.0
    already_autocalled: bool = False
    trade_info: str | None = None
    queries: list[str] = field(default_factory=lambda: ["ProductValue", "CashFlows"])
    coupon_barrier_type: str = "In"
    coupon_barrier_direction: str = "Up"
    cash_to_settle_today: float = 0.0
    quanto_meta: dict[str, Any] = field(default_factory=dict)
    asset_financing_curves: dict[str, Any] = field(default_factory=dict)
    discount_curve_id: str | None = None
    r_native_obs: torch.Tensor | None = None
    warnings: list[str] = field(default_factory=list)
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    repo_curves_map: dict[str, Any] = field(default_factory=dict)
    quanto_fx_specs: dict[str, Any] = field(default_factory=dict)
    state_digest: str = ""
    surface_digests: dict[str, str] = field(default_factory=dict)

    def to_tier2_float32_tensors(self) -> dict[str, torch.Tensor]:
        """Precision boundary segregation casting only the 6 padded neural input tensors."""
        return {
            "node_feats": self.node_feats.to(dtype=torch.float32),
            "edge_feats": self.edge_feats.to(dtype=torch.float32),
            "asset_mask": self.asset_mask.to(dtype=torch.float32),
            "sched_feats": self.sched_feats.to(dtype=torch.float32),
            "sched_mask": self.sched_mask.to(dtype=torch.float32),
            "global_feats": self.global_feats.to(dtype=torch.float32),
        }

    @property
    def canonical_codes(self) -> list[str]:
        return self.sorted_identifiers

    @property
    def schedule_state(self) -> SeasonedScheduleState:
        return self.seasoned_state

    @property
    def escrowed_dividends(self) -> EscrowedDividendOutput:
        return self.dividend_schedule

    @property
    def node_features(self) -> torch.Tensor:
        return self.node_feats

    @property
    def edge_features(self) -> torch.Tensor:
        return self.edge_feats

    @property
    def schedule_features(self) -> torch.Tensor:
        return self.sched_feats

    @property
    def global_context(self) -> torch.Tensor:
        return self.global_feats

    def unpermute_assets(
        self, tensor_or_array: torch.Tensor | npt.NDArray[np.float64]
    ) -> torch.Tensor | npt.NDArray[np.float64]:
        """Applies the inverse lexicographical permutation pi_lex^{-1} along dim 0."""
        if isinstance(tensor_or_array, torch.Tensor):
            inv_idx = torch.as_tensor(self.inv_lex_perm, dtype=torch.long, device=tensor_or_array.device)
            return tensor_or_array.index_select(0, inv_idx)
        return tensor_or_array[self.inv_lex_perm]


@dataclass
class AutocallPostprocOutput:
    """Output of the W1 affine stopping-time cash-flow and ProductValue assembler in float64."""

    product_value: torch.Tensor
    notional_pv: torch.Tensor
    coupon_pv: torch.Tensor
    option_pv: torch.Tensor
    funding_pv: torch.Tensor
    pv_in_transit: torch.Tensor
    survival_probs: torch.Tensor
    p_stop: torch.Tensor
    q_cpn: torch.Tensor
    l_put: torch.Tensor
    l_call: torch.Tensor
    cashflows: list[CashFlowItem]
    standard_error: float = 0.0
    reporting_currency: str = "RUB"
    cash_to_settle_today: float = 0.0
    l_onestar: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0, dtype=torch.float64))
    p_onestar: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0, dtype=torch.float64))

    @property
    def cash_flows(self) -> list[CashFlowItem]:
        return self.cashflows


# ── Thread-Safe Content-Addressed Caching ─────────────────────────────────────


@dataclass(frozen=True)
class ContentDigestKey:
    """Collision-resistant SHA-256 content-addressed cache key."""

    namespace: str
    digest_hex: str
    payload_len: int


_V = TypeVar("_V")


class ThreadSafeLRUCache:
    """Thread-safe bounded LRU cache with optional defensive value cloning."""

    def __init__(
        self,
        maxsize: int = 512,
        copy_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self._maxsize = max(int(maxsize), 1)
        self._copy_fn = copy_fn
        self._lock = threading.RLock()
        self._data: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any, default: Any = None, *, clone_return: bool = True) -> Any:
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            val = self._data[key]
            return self._copy_fn(val) if (clone_return and self._copy_fn is not None) else val

    def put(self, key: Any, value: Any, *, clone_return: bool = True) -> Any:
        with self._lock:
            if clone_return and self._copy_fn is not None:
                stored = self._copy_fn(value)
                self._data[key] = stored
                self._data.move_to_end(key)
                while len(self._data) > self._maxsize:
                    self._data.popitem(last=False)
                return self._copy_fn(stored)
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)
            return value

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            if key not in self._data:
                raise KeyError(key)
            self._data.move_to_end(key)
            val = self._data[key]
            return self._copy_fn(val) if self._copy_fn is not None else val

    def __setitem__(self, key: Any, value: Any) -> None:
        self.put(key, value)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_UTC = datetime.timezone.utc
_RAW_HEX_RE = re.compile(r"^[0-9a-fA-F]{32,128}$")
_PACK_U32 = struct.Struct("!BI").pack
_PACK_U64 = struct.Struct("!Q").pack
_PACK_F64 = struct.Struct("!Bd").pack
_PACK_I64 = struct.Struct("!Bq").pack
_PACK_DT = struct.Struct("!BH5BI").pack
_PACK_DA = struct.Struct("!BH2B").pack
_PACK_CASH_DIV = struct.Struct("!H5BIH5BId").pack
_PACK_ND_HDR = {
    0: struct.Struct("!II").pack,
    1: struct.Struct("!IQI").pack,
    2: struct.Struct("!I2QI").pack,
    3: struct.Struct("!I3QI").pack,
}
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_MODEL_SORTED_FIELDS_CACHE: dict[type, tuple[tuple[str, bytes], ...]] = {}


def _get_model_sorted_fields(cls: type) -> tuple[tuple[str, bytes], ...]:
    """Caches pre-sorted `(field_name, framed_key_bytes)` pairs once per Pydantic class."""
    cached = _MODEL_SORTED_FIELDS_CACHE.get(cls)
    if cached is not None:
        return cached
    fields = getattr(cls, "model_fields", None)
    if fields is None:
        return ()
    pack_u32 = _PACK_U32
    pairs = tuple(
        (k, pack_u32(0x53, len(kb)) + kb)
        for k in sorted(fields)
        for kb in (k.encode("utf-8"),)
    )
    _MODEL_SORTED_FIELDS_CACHE[cls] = pairs
    return pairs


def _flush_canonical_buf(
    buf: bytearray,
    upd: Callable[[Any], Any],
    flushed: list[int],
    extra: Any = None,
) -> None:
    if buf:
        flushed[0] += len(buf)
        upd(buf)
        buf.clear()
    if extra is not None:
        flushed[0] += len(extra)
        upd(extra)


def _append_raw_mv(
    buf: bytearray,
    upd: Callable[[Any], Any],
    flushed: list[int],
    mv: memoryview,
) -> None:
    if len(mv) <= 4096 and len(buf) + len(mv) <= 65536:
        buf.extend(mv)
    else:
        _flush_canonical_buf(buf, upd, flushed, mv)


def _append_str_val(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    buf.extend(_PACK_U32(0x53, len(b)))
    buf.extend(b)


def _append_opt_str_val(buf: bytearray, s: str | None) -> None:
    if s is None:
        buf.append(0x4E)
    else:
        b = s.encode("utf-8")
        buf.extend(_PACK_U32(0x53, len(b)))
        buf.extend(b)


def _append_dt_bytes(buf: bytearray, dt_obj: datetime.datetime) -> None:
    dt_u = dt_obj if dt_obj.tzinfo is _UTC else _to_utc_datetime(dt_obj)
    buf.extend(
        _PACK_DT(
            0x54,
            dt_u.year,
            dt_u.month,
            dt_u.day,
            dt_u.hour,
            dt_u.minute,
            dt_u.second,
            dt_u.microsecond,
        )
    )


def _append_dt_seq_fast(
    buf: bytearray,
    upd: Callable[[Any], Any],
    flushed: list[int],
    seq: Any,
) -> bool:
    if seq is None:
        buf.append(0x4E)
        return True
    if not isinstance(seq, (list, tuple)):
        return _stream_canonical_val(buf, upd, flushed, seq)
    buf.extend(_PACK_U32(0x4C if isinstance(seq, list) else 0x55, len(seq)))
    dt_cls = datetime.datetime
    utc_tz = _UTC
    pack_dt = _PACK_DT
    for item in seq:
        if type(item) is dt_cls and item.tzinfo is utc_tz:
            buf.extend(
                pack_dt(
                    0x54,
                    item.year,
                    item.month,
                    item.day,
                    item.hour,
                    item.minute,
                    item.second,
                    item.microsecond,
                )
            )
        elif not _stream_canonical_val(buf, upd, flushed, item):
            return False
    if len(buf) > 65536:
        _flush_canonical_buf(buf, upd, flushed)
    return True


def _append_float_seq_typed(
    buf: bytearray,
    upd: Callable[[Any], Any],
    flushed: list[int],
    seq: Any,
) -> bool:
    if seq is None:
        buf.append(0x4E)
        return True
    if isinstance(seq, (list, tuple)):
        n_seq = len(seq)
        if n_seq == 0:
            buf.extend(_PACK_U32(0x64, 0))
            return True
        try:
            src = [0.0 if float(x) == 0.0 else float(x) for x in seq] if (0.0 in seq) else seq
            arr_d = array("d", src)
            buf.extend(_PACK_U32(0x64, n_seq))
            _append_raw_mv(buf, upd, flushed, memoryview(arr_d).cast("B"))
            return True
        except (TypeError, OverflowError, ValueError):
            pass
    return _stream_canonical_val(buf, upd, flushed, seq)


def _append_mapping_bytes(
    buf: bytearray,
    upd: Callable[[Any], Any],
    flushed: list[int],
    obj: Mapping[Any, Any],
) -> bool:
    buf.extend(_PACK_U32(0x4D, len(obj)))
    if not obj:
        return True
    try:
        keys = sorted(obj)
    except TypeError:
        keys = sorted(obj, key=lambda x: (str(type(x)), str(x)))
    for k in keys:
        if type(k) is str:
            _append_str_val(buf, k)
        elif not _stream_canonical_val(buf, upd, flushed, k):
            return False
        if not _stream_canonical_val(buf, upd, flushed, obj[k]):
            return False
    if len(buf) > 65536:
        _flush_canonical_buf(buf, upd, flushed)
    return True


def _stream_canonical_val(
    buf: bytearray,
    upd: Callable[[Any], Any],
    flushed: list[int],
    obj: Any,
) -> bool:
    """Appends canonical collision-free binary bytes into `buf` (flushing large payloads)."""
    if obj is None:
        buf.append(0x4E)
        return True

    tp = type(obj)
    if tp is str:
        _append_str_val(buf, obj)
        return True
    if tp is float:
        buf.extend(_PACK_F64(0x46, 0.0 if obj == 0.0 else obj))
        return True
    if tp is int:
        if _INT64_MIN <= obj <= _INT64_MAX:
            buf.extend(_PACK_I64(0x71, obj))
        else:
            b = str(obj).encode("ascii")
            buf.extend(_PACK_U32(0x49, len(b)))
            buf.extend(b)
        return True
    if tp is bool:
        buf.append(0x31 if obj else 0x30)
        return True
    if tp is datetime.datetime:
        _append_dt_bytes(buf, obj)
        return True
    if tp is datetime.date:
        buf.extend(_PACK_DA(0x44, obj.year, obj.month, obj.day))
        return True
    if tp is list:
        n_items = len(obj)
        if n_items > 0 and type(obj[0]) is float and all(type(x) is float for x in obj):
            src = [0.0 if x == 0.0 else x for x in obj] if (0.0 in obj) else obj
            arr_d = array("d", src)
            buf.extend(_PACK_U32(0x64, n_items))
            _append_raw_mv(buf, upd, flushed, memoryview(arr_d).cast("B"))
            return True
        buf.extend(_PACK_U32(0x4C, n_items))
        for item in obj:
            if not _stream_canonical_val(buf, upd, flushed, item):
                return False
        if len(buf) > 65536:
            _flush_canonical_buf(buf, upd, flushed)
        return True
    if tp is tuple:
        buf.extend(_PACK_U32(0x55, len(obj)))
        for item in obj:
            if not _stream_canonical_val(buf, upd, flushed, item):
                return False
        if len(buf) > 65536:
            _flush_canonical_buf(buf, upd, flushed)
        return True
    if tp is dict:
        return _append_mapping_bytes(buf, upd, flushed, obj)
    if tp is bytes:
        buf.extend(_PACK_U32(0x59, len(obj)))
        _append_raw_mv(buf, upd, flushed, memoryview(obj))
        return True

    if isinstance(obj, Enum):
        return _stream_canonical_val(buf, upd, flushed, obj.value)
    if isinstance(obj, (bool, np.bool_)):
        buf.append(0x31 if bool(obj) else 0x30)
        return True
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        buf.extend(_PACK_F64(0x46, 0.0 if fv == 0.0 else fv))
        return True
    if isinstance(obj, (int, np.integer)):
        iv = int(obj)
        if _INT64_MIN <= iv <= _INT64_MAX:
            buf.extend(_PACK_I64(0x71, iv))
        else:
            b = str(iv).encode("ascii")
            buf.extend(_PACK_U32(0x49, len(b)))
            buf.extend(b)
        return True
    if isinstance(obj, datetime.datetime):
        _append_dt_bytes(buf, obj)
        return True
    if isinstance(obj, datetime.date):
        buf.extend(_PACK_DA(0x44, obj.year, obj.month, obj.day))
        return True
    if isinstance(obj, (bytearray, memoryview)):
        mv_raw = obj if isinstance(obj, memoryview) else memoryview(obj)
        n_b = mv_raw.nbytes
        buf.extend(_PACK_U32(0x59, n_b))
        if n_b > 0:
            try:
                mv = mv_raw.cast("B")
            except TypeError:
                mv = memoryview(bytes(obj))
            _append_raw_mv(buf, upd, flushed, mv)
        return True
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            return _stream_canonical_val(
                buf, upd, flushed, ("__np_obj__", obj.shape, obj.dtype.str, obj.tolist())
            )
        arr = obj if obj.flags.c_contiguous else np.ascontiguousarray(obj)
        ndim = arr.ndim
        dtype_b = arr.dtype.str.encode("ascii")
        buf.append(0x4B)
        hdr_packer = _PACK_ND_HDR.get(ndim)
        if hdr_packer is not None:
            buf.extend(hdr_packer(ndim, *arr.shape, len(dtype_b)))
        else:
            buf.extend(struct.pack(f"!I{ndim}QI", ndim, *arr.shape, len(dtype_b)))
        buf.extend(dtype_b)
        n_b = arr.nbytes
        buf.extend(_PACK_U64(n_b))
        if n_b > 0:
            try:
                mv = memoryview(cast(Any, arr)).cast("B")
            except TypeError:
                mv = memoryview(arr.tobytes())
            _append_raw_mv(buf, upd, flushed, mv)
        return True
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad:
            return False
        return _stream_canonical_val(buf, upd, flushed, obj.detach().cpu().numpy())
    if isinstance(obj, (ClampedYieldCurve, RepoCurve, RegularizedVolSurface)):
        return _stream_canonical_val(buf, upd, flushed, _canonical_market_obj_repr(obj))
    if isinstance(obj, BaseModel):
        field_pairs = _get_model_sorted_fields(tp)
        cls_b = tp.__name__.encode("utf-8")
        buf.extend(_PACK_U32(0x50, len(cls_b)))
        buf.extend(cls_b)
        obj_d = obj.__dict__
        for k, k_hdr in field_pairs:
            if k in obj_d:
                buf.append(0x01)
                buf.extend(k_hdr)
                if not _stream_canonical_val(buf, upd, flushed, obj_d[k]):
                    return False
        buf.append(0x00)
        extra = getattr(obj, "__pydantic_extra__", None)
        if extra:
            buf.append(0x01)
            if not _append_mapping_bytes(buf, upd, flushed, extra):
                return False
        else:
            buf.append(0x00)
        if len(buf) > 65536:
            _flush_canonical_buf(buf, upd, flushed)
        return True
    if isinstance(obj, Mapping):
        return _append_mapping_bytes(buf, upd, flushed, obj)
    if isinstance(obj, (list, tuple)):
        buf.extend(_PACK_U32(0x4C if isinstance(obj, list) else 0x55, len(obj)))
        for item in obj:
            if not _stream_canonical_val(buf, upd, flushed, item):
                return False
        return True
    if isinstance(obj, (set, frozenset)):
        items = sorted(obj, key=lambda x: (str(type(x)), str(x)))
        buf.extend(_PACK_U32(0x5A if isinstance(obj, set) else 0x7A, len(items)))
        for item in items:
            if not _stream_canonical_val(buf, upd, flushed, item):
                return False
        return True
    if hasattr(obj, "__dict__"):
        cls_b = tp.__name__.encode("utf-8")
        buf.extend(_PACK_U32(0x4F, len(cls_b)))
        buf.extend(cls_b)
        return _append_mapping_bytes(buf, upd, flushed, vars(obj))
    return False


def _feed_canonical_bytes(hasher: Any, obj: Any, counter: list[int]) -> bool:
    """Feeds deterministic canonical byte representation into a hasher."""
    buf = bytearray()
    flushed = [0]
    upd = hasher.update
    ok = _stream_canonical_val(buf, upd, flushed, obj)
    if buf:
        flushed[0] += len(buf)
        upd(buf)
    counter[0] += flushed[0]
    return ok


def _canonical_market_obj_repr(obj: Any) -> Any:
    """Returns deterministic canonical tuple excluding runtime locks and spline caches."""
    if isinstance(obj, ClampedYieldCurve):
        return (
            "ClampedYieldCurve",
            obj.identifier,
            obj.currency,
            obj.as_of_date,
            obj.dcc,
            obj.interpolation_method,
            obj.space_coords,
            obj.extrapolation_method,
            obj.quote_expression,
            getattr(obj, "_input_mode", None),
            obj.tenors,
            obj.rates,
        )
    if isinstance(obj, RepoCurve):
        inner = getattr(obj, "_inner_curve", None)
        return (
            "RepoCurve",
            obj.identifier,
            obj.as_of_date,
            obj.dcc,
            obj.interpolation_method,
            obj.extrapolation_method,
            obj.quote_expression,
            obj.is_flat_zero,
            _canonical_market_obj_repr(inner) if inner is not None else None,
        )
    if isinstance(obj, RegularizedVolSurface):
        return (
            "RegularizedVolSurface",
            getattr(obj, "surface_id", None),
            getattr(obj, "as_of_date", None),
            getattr(obj, "dcc", None),
            getattr(obj, "smile_interp", None),
            getattr(obj, "smile_extrap", None),
            getattr(obj, "bc_type", None),
            float(getattr(obj, "min_vol", 0.0)),
            float(getattr(obj, "max_local_vol", 0.0)),
            getattr(obj, "tenors", None),
            getattr(obj, "tenors_act365", None),
            getattr(obj, "strikes", None),
            getattr(obj, "quotes", None),
            getattr(obj, "calendar_proj_digest", None),
        )
    return obj


class _UncacheablePayloadError(Exception):
    """Raised when a payload contains an autograd-tracked tensor or unhashable object."""


def _append_vol_surface_to_buf(
    buf: bytearray,
    upd: Callable[[Any], Any],
    flushed: list[int],
    surf_spec: Any,
    as_of_dt: datetime.datetime,
    surface_id: str | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> bool:
    """Streams volatility surface metadata and numeric arrays directly into `buf`."""
    if isinstance(surf_spec, EquityVolSmileCollectionSpec):
        t_name: Any = surf_spec.TypeName
        eff_id: Any = surface_id or surf_spec.Identifier
        dcc_raw: Any = surf_spec.DayCountConvention
        interp_raw: Any = surf_spec.SmileInterpolationMethod
        extrap_raw: Any = surf_spec.SmileExtrapolationMethod
        ccy_raw: Any = surf_spec.Currency
        exp_dates: Any = surf_spec.ExpirationDates
        tenors_raw: Any = None
        deltas_raw: Any = None
        as_of_surf_raw: Any = None
        fwd_obj_raw: Any = None
        fin_curve_raw: Any = None
        cal_raw: Any = None
        min_vol_raw: Any = None
        max_lv_raw: Any = None
        bc_type_raw: Any = None
        strikes: Any = surf_spec.Strikes
        quotes: Any = surf_spec.VolatilityQuotes
    elif isinstance(surf_spec, FXVolatilitySmileCollectionSpec):
        t_name = surf_spec.TypeName
        eff_id = surface_id or surf_spec.Identifier
        dcc_raw = surf_spec.DayCountConvention
        interp_raw = surf_spec.SmileInterpolationMethod
        extrap_raw = surf_spec.SmileExtrapolationMethod
        ccy_raw = None
        exp_dates = None
        tenors_raw = surf_spec.Tenors
        deltas_raw = surf_spec.Deltas
        as_of_surf_raw = surf_spec.AsOfDate
        fwd_obj_raw = surf_spec.ForwardObject
        fin_curve_raw = surf_spec.FinancingCurve
        cal_raw = surf_spec.Calendar
        min_vol_raw = None
        max_lv_raw = None
        bc_type_raw = None
        strikes = None
        quotes = surf_spec.VolatilityQuotes
    else:
        eff_id = surface_id or _extract_attr(surf_spec, "Identifier", "identifier", default="")
        t_name = _extract_attr(surf_spec, "TypeName", "type_name", default="")
        dcc_raw = _extract_attr(surf_spec, "DayCountConvention", "day_count_convention", "dcc", default="Act365")
        interp_raw = _extract_attr(
            surf_spec, "SmileInterpolationMethod", "smile_interpolation_method", "smile_interp", default="CubicSpline"
        )
        extrap_raw = _extract_attr(
            surf_spec, "SmileExtrapolationMethod", "smile_extrapolation_method", "smile_extrap", default="Linear"
        )
        ccy_raw = _extract_attr(surf_spec, "Currency", "currency", default=None)
        exp_dates = _extract_attr(surf_spec, "ExpirationDates", "expiration_dates", "exp_dates")
        tenors_raw = _extract_attr(surf_spec, "tenors", "Tenors")
        deltas_raw = _extract_attr(surf_spec, "Deltas", "deltas")
        as_of_surf_raw = _extract_attr(surf_spec, "AsOfDate", "as_of_date")
        fwd_obj_raw = _extract_attr(surf_spec, "ForwardObject", "forward_object")
        fin_curve_raw = _extract_attr(surf_spec, "FinancingCurve", "financing_curve")
        cal_raw = _extract_attr(surf_spec, "Calendar", "calendar", "calendar_proj_digest")
        min_vol_raw = _extract_attr(surf_spec, "min_vol", "MinVol")
        max_lv_raw = _extract_attr(surf_spec, "max_local_vol", "MaxLocalVol")
        bc_type_raw = _extract_attr(surf_spec, "bc_type", "BcType")
        strikes = _extract_attr(surf_spec, "Strikes", "strikes")
        quotes = _extract_attr(surf_spec, "VolatilityQuotes", "volatility_quotes", "quotes")

    buf.extend(b"VSF")
    _append_str_val(buf, str(t_name))
    _append_str_val(buf, str(eff_id))
    _append_str_val(buf, str(getattr(dcc_raw, "value", dcc_raw)))
    _append_str_val(buf, str(getattr(interp_raw, "value", interp_raw)))
    _append_str_val(buf, str(getattr(extrap_raw, "value", extrap_raw)))
    _append_opt_str_val(buf, str(ccy_raw) if ccy_raw is not None else None)
    _append_dt_bytes(buf, as_of_dt)
    if not _stream_canonical_val(buf, upd, flushed, as_of_surf_raw):
        return False
    if not _append_dt_seq_fast(buf, upd, flushed, exp_dates):
        return False
    for meta_field in (tenors_raw, deltas_raw, fwd_obj_raw, fin_curve_raw, cal_raw):
        if not _stream_canonical_val(buf, upd, flushed, meta_field):
            return False
    if min_vol_raw is not None:
        mv_f = float(min_vol_raw)
        buf.extend(_PACK_F64(0x46, 0.0 if mv_f == 0.0 else mv_f))
    else:
        buf.append(0x4E)
    if max_lv_raw is not None:
        mlv_f = float(max_lv_raw)
        buf.extend(_PACK_F64(0x46, 0.0 if mlv_f == 0.0 else mlv_f))
    else:
        buf.append(0x4E)
    _append_opt_str_val(buf, str(bc_type_raw) if bc_type_raw is not None else None)
    if kwargs is not None:
        if not _append_mapping_bytes(buf, upd, flushed, kwargs):
            return False
    else:
        buf.append(0x4E)

    if strikes is None:
        buf.append(0x4E)
    elif isinstance(strikes, np.ndarray):
        if not _stream_canonical_val(buf, upd, flushed, strikes):
            return False
    elif not _append_float_seq_typed(buf, upd, flushed, strikes):
        return False

    if quotes is None:
        buf.append(0x4E)
    elif isinstance(quotes, np.ndarray):
        if not _stream_canonical_val(buf, upd, flushed, quotes):
            return False
    elif isinstance(quotes, (list, tuple)):
        try:
            row_lens: list[int] = []
            q_arr = array("d")
            for row in quotes:
                if type(row) is list:
                    row_lens.append(len(row))
                    if 0.0 in row:
                        q_arr.extend(0.0 if float(x) == 0.0 else float(x) for x in row)
                    else:
                        q_arr.fromlist(row)
                elif isinstance(row, (list, tuple)):
                    row_lens.append(len(row))
                    if 0.0 in row:
                        q_arr.extend(0.0 if float(x) == 0.0 else float(x) for x in row)
                    else:
                        q_arr.extend(row)
                else:
                    raise TypeError
            buf.extend(_PACK_U32(0x51, len(row_lens)))
            if row_lens:
                len_arr = array("Q", row_lens)
                _append_raw_mv(buf, upd, flushed, memoryview(len_arr).cast("B"))
                if q_arr:
                    _append_raw_mv(buf, upd, flushed, memoryview(q_arr).cast("B"))
        except (TypeError, ValueError, OverflowError):
            if not _stream_canonical_val(buf, upd, flushed, quotes):
                return False
    else:
        if not _stream_canonical_val(buf, upd, flushed, quotes):
            return False
    return True


def _stream_vol_surface_bytes(
    upd: Callable[[bytes], None],
    surf_spec: Any,
    as_of_dt: datetime.datetime,
    surface_id: str | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> int:
    """Streams live volatility surface fields into a hasher; returns streamed byte count."""
    buf = bytearray()
    flushed = [0]

    def _emit_bytes(chunk: Any) -> None:
        upd(bytes(chunk))

    ok = _append_vol_surface_to_buf(
        buf,
        _emit_bytes,
        flushed,
        surf_spec,
        as_of_dt,
        surface_id=surface_id,
        kwargs=kwargs,
    )
    if not ok:
        raise _UncacheablePayloadError()
    if buf:
        flushed[0] += len(buf)
        upd(bytes(buf))
    return flushed[0]


def build_vol_surface_cache_key(
    surf_spec: Any,
    as_of_dt: datetime.datetime,
    surface_id: str | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> ContentDigestKey | None:
    """Builds a blake2b ContentDigestKey for a volatility surface spec without full model_dump."""
    try:
        h = hashlib.blake2b(digest_size=16)
        buf = bytearray()
        flushed = [0]
        ok = _append_vol_surface_to_buf(
            buf,
            h.update,
            flushed,
            surf_spec,
            as_of_dt,
            surface_id=surface_id,
            kwargs=kwargs,
        )
        if not ok:
            return None
        if buf:
            flushed[0] += len(buf)
            h.update(buf)
        return ContentDigestKey(
            namespace="vol_surface",
            digest_hex=h.hexdigest(),
            payload_len=flushed[0],
        )
    except (TypeError, ValueError, AttributeError, _UncacheablePayloadError):
        return digest_market_bytes(
            "vol_surface",
            (surf_spec, as_of_dt, dict(kwargs) if kwargs is not None else {"surface_id": surface_id}),
        )


def _append_market_dataset_to_buf(
    buf: bytearray,
    upd: Callable[[Any], Any],
    flushed: list[int],
    mkt: Any,
    as_of_dt: datetime.datetime,
) -> bool:
    """Streams complete MarketDataSet interior state directly into `buf`."""
    _append_dt_bytes(buf, as_of_dt)

    spots = mkt.Spots
    if spots:
        s_keys = sorted(spots)
        buf.extend(_PACK_U32(0x4D, len(s_keys)))
        for k in s_keys:
            _append_str_val(buf, k)
            v = float(spots[k])
            buf.extend(_PACK_F64(0x46, 0.0 if v == 0.0 else v))
    else:
        buf.extend(_PACK_U32(0x4D, 0))

    corrs = mkt.Correlations
    if corrs:
        c_keys = sorted(corrs)
        buf.extend(_PACK_U32(0x4D, len(c_keys)))
        for k in c_keys:
            _append_str_val(buf, k)
            v = float(corrs[k])
            buf.extend(_PACK_F64(0x46, 0.0 if v == 0.0 else v))
    else:
        buf.extend(_PACK_U32(0x4D, 0))

    bundles = mkt.RatesCurvesBundles
    buf.extend(_PACK_U32(0x52, len(bundles)))
    for cid in sorted(bundles):
        _append_str_val(buf, cid)
        c = bundles[cid]
        if isinstance(c, DiscountFactorCurveSpec):
            _append_str_val(buf, c.Identifier)
            _append_str_val(buf, c.Currency)
            _append_dt_bytes(buf, c.AsOfDate)
            _append_str_val(buf, c.InterpolationDayCountConvention.value)
            _append_str_val(buf, c.InterpolationMethod.value)
            _append_str_val(buf, c.InterpolationSpaceCoordinates.value)
            _append_str_val(buf, c.ExtrapolationMethod.value)
            _append_str_val(buf, c.RateQuoteExpression.value)
            rs = c.RatesSchedule
            if not _append_dt_seq_fast(buf, upd, flushed, rs.Dates):
                return False
            if not _append_float_seq_typed(buf, upd, flushed, rs.Rates):
                return False
            if not _append_float_seq_typed(buf, upd, flushed, rs.Spreads):
                return False
        elif not _stream_canonical_val(buf, upd, flushed, c):
            return False

    repos = mkt.RepoSpreadCurves
    buf.extend(_PACK_U32(0x50, len(repos)))
    for cid in sorted(repos):
        _append_str_val(buf, cid)
        c_repo = repos[cid]
        if isinstance(c_repo, RepoSpreadCurveSpec):
            _append_str_val(buf, c_repo.Identifier)
            _append_str_val(buf, c_repo.InterpolationDayCountConvention.value)
            _append_str_val(buf, c_repo.InterpolationMethod.value)
            _append_str_val(buf, c_repo.ExtrapolationMethod.value)
            _append_str_val(buf, c_repo.InterpolationSpaceCoordinates.value)
            _append_str_val(buf, c_repo.RateQuoteExpression.value)
            rs_r = c_repo.RatesSchedule
            buf.extend(_PACK_U32(0x4C, len(rs_r.Periods)))
            for p_str in rs_r.Periods:
                _append_str_val(buf, p_str)
            if not _append_float_seq_typed(buf, upd, flushed, rs_r.Rates):
                return False
        elif not _stream_canonical_val(buf, upd, flushed, c_repo):
            return False

    surfs = mkt.VolatilitySurfaces
    buf.extend(_PACK_U32(0x56, len(surfs)))
    for sid in sorted(surfs):
        _append_str_val(buf, sid)
        if not _append_vol_surface_to_buf(buf, upd, flushed, surfs[sid], as_of_dt, surface_id=sid):
            return False

    divs = mkt.Dividends
    buf.extend(_PACK_U32(0x44, len(divs)))
    utc_tz = _UTC
    pack_div = _PACK_CASH_DIV
    for tid in sorted(divs):
        _append_str_val(buf, tid)
        d_sp = divs[tid]
        if isinstance(d_sp, DividendsSpec):
            _append_str_val(buf, d_sp.Identifier)
            _append_opt_str_val(buf, d_sp.Currency)
            cdl = d_sp.CashDividendList
            buf.extend(_PACK_U32(0x4C, len(cdl)))
            for cd in cdl:
                ex_dt = cd.ExDate
                if ex_dt.tzinfo is not utc_tz:
                    ex_dt = _to_utc_datetime(ex_dt)
                pay_dt = cd.PaymentDate
                if pay_dt.tzinfo is not utc_tz:
                    pay_dt = _to_utc_datetime(pay_dt)
                amt = float(cd.Amount)
                buf.extend(
                    pack_div(
                        ex_dt.year,
                        ex_dt.month,
                        ex_dt.day,
                        ex_dt.hour,
                        ex_dt.minute,
                        ex_dt.second,
                        ex_dt.microsecond,
                        pay_dt.year,
                        pay_dt.month,
                        pay_dt.day,
                        pay_dt.hour,
                        pay_dt.minute,
                        pay_dt.second,
                        pay_dt.microsecond,
                        0.0 if amt == 0.0 else amt,
                    )
                )
        elif not _stream_canonical_val(buf, upd, flushed, d_sp):
            return False

    fixings = mkt.Fixings
    buf.extend(_PACK_U32(0x58, len(fixings)))
    for tid in sorted(fixings):
        _append_str_val(buf, tid)
        f_sp = fixings[tid]
        if isinstance(f_sp, FixingsSpec):
            _append_str_val(buf, f_sp.Identifier)
            if not _append_dt_seq_fast(buf, upd, flushed, f_sp.Dates):
                return False
            if not _append_float_seq_typed(buf, upd, flushed, f_sp.Values):
                return False
        elif not _stream_canonical_val(buf, upd, flushed, f_sp):
            return False

    return True


def _stream_eq_request_live(
    req_obj: Any,
    *,
    d_max: int,
    m_max: int,
    as_of_dt: datetime.datetime,
) -> ContentDigestKey | None:
    """Streams live EQAutocallableTradeRequest market and contract fields via struct+blake2b."""
    try:
        prod = req_obj.Product
        mdl = req_obj.Model
        mkt = mdl.MarketDataSet
        h = hashlib.blake2b(digest_size=16)
        buf = bytearray()
        flushed = [0]
        upd = h.update

        ver_b = FEATURE_SCHEMA_VERSION.encode("ascii")
        buf.extend(struct.pack("!2II", int(d_max), int(m_max), len(ver_b)))
        buf.extend(ver_b)

        if not _stream_canonical_val(buf, upd, flushed, prod):
            return None

        fin_ids = mdl.FinancingCurveIdentifiers
        if fin_ids:
            buf.extend(_PACK_U32(0x46, len(fin_ids)))
            for x in fin_ids:
                _append_str_val(buf, str(x))
        else:
            buf.extend(_PACK_U32(0x46, 0))

        if not _append_market_dataset_to_buf(buf, upd, flushed, mkt, as_of_dt):
            return None

        if buf:
            flushed[0] += len(buf)
            upd(buf)
        return ContentDigestKey(
            namespace="preproc_state",
            digest_hex=h.hexdigest(),
            payload_len=flushed[0],
        )
    except (AttributeError, TypeError, ValueError, KeyError, _UncacheablePayloadError):
        return None


def digest_market_bytes(namespace: str, payload: Any) -> ContentDigestKey | None:
    """Builds a blake2b ContentDigestKey covering 100% of interior values."""
    if isinstance(payload, EQAutocallableTradeRequest) or (
        hasattr(payload, "Product")
        and hasattr(payload, "Model")
        and hasattr(getattr(payload, "Model"), "MarketDataSet")
    ):
        as_of_raw = getattr(payload.Model.MarketDataSet, "AsOfDate", None)
        if as_of_raw is not None:
            key_live = _stream_eq_request_live(
                payload,
                d_max=D_MAX_DEFAULT,
                m_max=M_MAX_DEFAULT,
                as_of_dt=_to_utc_datetime(as_of_raw),
            )
            if key_live is not None:
                return ContentDigestKey(
                    namespace=namespace,
                    digest_hex=key_live.digest_hex,
                    payload_len=key_live.payload_len,
                )

    h = hashlib.blake2b(digest_size=16)
    buf = bytearray()
    flushed = [0]
    upd = h.update
    if isinstance(payload, MarketDataSetSpec):
        try:
            ok = _append_market_dataset_to_buf(buf, upd, flushed, payload, _to_utc_datetime(payload.AsOfDate))
        except (AttributeError, TypeError, ValueError, KeyError):
            ok = False
    else:
        ok = _stream_canonical_val(buf, upd, flushed, payload)
    if not ok:
        return None
    if buf:
        flushed[0] += len(buf)
        upd(buf)
    return ContentDigestKey(
        namespace=namespace,
        digest_hex=h.hexdigest(),
        payload_len=flushed[0],
    )


def _clone_dataclass_tensors(obj: Any) -> Any:
    """Clones dataclass fields containing tensors, ndarrays, lists, or dicts."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, list):
        return [_clone_dataclass_tensors(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_dataclass_tensors(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _clone_dataclass_tensors(v) for k, v in obj.items()}
    return obj


def _clone_dataclass_fields(obj: Any) -> Any:
    field_names = {f.name for f in dc_fields(obj)}
    updates = {
        name: _clone_dataclass_tensors(getattr(obj, name))
        for name in field_names
    }
    cloned = dc_replace(obj, **updates)
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if k not in field_names:
                setattr(cloned, k, _clone_dataclass_tensors(v))
    return cloned


def compute_surface_digest(surf: RegularizedVolSurface) -> str:
    """Computes a 64-hex SHA-256 content digest for a regularized volatility surface."""
    h = hashlib.sha256()
    counter = [0]
    _feed_canonical_bytes(h, _canonical_market_obj_repr(surf), counter)
    return h.hexdigest()


def _detach_for_digest(val: Any) -> Any:
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().contiguous().numpy()
    if isinstance(val, (ClampedYieldCurve, RepoCurve, RegularizedVolSurface)):
        return _canonical_market_obj_repr(val)
    if isinstance(val, Mapping):
        return {str(k): _detach_for_digest(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_detach_for_digest(x) for x in val]
    return val


def compute_preproc_state_digest(state: AutocallPreprocState) -> str:
    """Computes an immutable 64-hex SHA-256 digest covering all market and contract state inputs."""
    h = hashlib.sha256()
    counter = [0]
    surf_digests = (
        dict(state.surface_digests)
        if state.surface_digests
        else {k: compute_surface_digest(v) for k, v in state.vol_surfaces.items()}
    )
    payload = (
        state.feature_schema_version,
        int(state.d_assets),
        int(state.m_rem),
        int(state.d_max),
        int(state.m_max),
        tuple(state.original_identifiers),
        tuple(state.sorted_identifiers),
        state.lex_perm,
        state.inv_lex_perm,
        _detach_for_digest(state.spots_t0),
        _detach_for_digest(state.ref_spots),
        _detach_for_digest(state.corr_matrix),
        _detach_for_digest(state.tau_obs),
        _detach_for_digest(state.delta_tau),
        _detach_for_digest(state.tau_pay),
        _detach_for_digest(state.df_obs),
        _detach_for_digest(state.df_pay),
        _detach_for_digest(state.ko_barriers),
        _detach_for_digest(state.is_callable),
        _detach_for_digest(state.cpn_barriers),
        _detach_for_digest(state.is_guaranteed_cpn),
        _detach_for_digest(state.coupon_amounts),
        _detach_for_digest(state.forward_rel),
        _detach_for_digest(state.forward_perf),
        _detach_for_digest(state.cash_div_leverage),
        _detach_for_digest(state.iv_at_fwd),
        _detach_for_digest(state.v_perf),
        _detach_for_digest(state.r_native_obs),
        surf_digests,
        _detach_for_digest(state.discount_curve),
        state.discount_curve_id,
        _detach_for_digest(state.asset_financing_curves),
        _detach_for_digest(state.repo_curves_map),
        _detach_for_digest(state.quanto_fx_specs),
        _detach_for_digest(state.quanto_meta),
        _detach_for_digest(state.funding_leg),
        state.seasoned_state.as_of_date,
        state.seasoned_state.start_date,
        tuple(state.seasoned_state.future_obs_dates),
        tuple(state.seasoned_state.future_pay_dates),
        float(state.c_mem_0),
        float(state.i_ki_past),
        bool(state.already_autocalled),
        float(state.pv_in_transit),
        float(state.cash_to_settle_today),
        float(state.strike),
        float(state.option_gearing),
        float(state.lower_strike) if state.lower_strike is not None else None,
        float(state.ki_barrier) if state.ki_barrier is not None else None,
        bool(state.is_memory),
        float(state.ko_spread),
        float(state.cpn_spread),
        float(state.ki_spread),
        bool(state.no_notional_repayment),
        float(state.autocall_coupon),
        bool(state.force_coupons_on_autocall),
        bool(state.allow_autocall_on_expiry),
        float(state.participation_call_strike) if state.participation_call_strike is not None else None,
        float(state.participation_call_rate),
        float(state.one_star_strike) if state.one_star_strike is not None else None,
        float(state.one_star_coupon),
        bool(state.is_long_basket),
        state.payment_currency,
        state.callability_index,
        state.coupon_index,
        state.option_index,
        state.coupon_barrier_type,
        state.coupon_barrier_direction,
        {
            str(k): [(ev.ex_date, ev.payment_date, float(ev.amount)) for ev in evs]
            for k, evs in state.dividend_schedule.dividend_events.items()
        },
        _detach_for_digest(state.dividend_schedule.dividend_amount_tensors),
    )
    _feed_canonical_bytes(h, payload, counter)
    return h.hexdigest()


def _clone_preproc_state(state: AutocallPreprocState) -> AutocallPreprocState:
    """Deep-clones mutable containers and tensors so cached templates cannot be poisoned."""
    surf_jacs: dict[str, npt.NDArray[np.float64]] = (
        state.surface_feature_jacobians.clone()
        if isinstance(state.surface_feature_jacobians, _LazySurfaceJacobianDict)
        else {k: v.copy() for k, v in state.surface_feature_jacobians.items()}
    )
    iv_jacs: dict[str, npt.NDArray[np.float64]] = (
        state.iv_at_fwd_jacobians.clone()
        if isinstance(state.iv_at_fwd_jacobians, _LazySurfaceJacobianDict)
        else {k: v.copy() for k, v in state.iv_at_fwd_jacobians.items()}
    )
    funding_clone: dict[str, Any] | None = None
    if state.funding_leg is not None:
        funding_clone = {
            k: (list(v) if isinstance(v, list) else v)
            for k, v in state.funding_leg.items()
        }
    return dc_replace(
        state,
        node_feats=state.node_feats.clone(),
        edge_feats=state.edge_feats.clone(),
        asset_mask=state.asset_mask.clone(),
        sched_feats=state.sched_feats.clone(),
        sched_mask=state.sched_mask.clone(),
        global_feats=state.global_feats.clone(),
        original_identifiers=list(state.original_identifiers),
        sorted_identifiers=list(state.sorted_identifiers),
        lex_perm=state.lex_perm.copy(),
        inv_lex_perm=state.inv_lex_perm.copy(),
        spots_t0=state.spots_t0.clone(),
        ref_spots=state.ref_spots.clone(),
        moneyness_t0=state.moneyness_t0.clone(),
        corr_matrix=state.corr_matrix.clone(),
        tau_obs=state.tau_obs.clone(),
        delta_tau=state.delta_tau.clone(),
        tau_pay=state.tau_pay.clone(),
        df_obs=state.df_obs.clone(),
        df_pay=state.df_pay.clone(),
        ko_barriers=state.ko_barriers.clone(),
        is_callable=state.is_callable.clone(),
        cpn_barriers=state.cpn_barriers.clone(),
        is_guaranteed_cpn=state.is_guaranteed_cpn.clone(),
        coupon_amounts=state.coupon_amounts.clone(),
        forward_rel=state.forward_rel.clone(),
        forward_perf=state.forward_perf.clone(),
        cash_div_leverage=state.cash_div_leverage.clone(),
        iv_at_fwd=state.iv_at_fwd.clone(),
        v_perf=state.v_perf.clone(),
        seasoned_state=_clone_dataclass_fields(state.seasoned_state),
        dividend_schedule=_clone_dataclass_fields(state.dividend_schedule),
        vol_surfaces=dict(state.vol_surfaces),
        surface_feature_jacobians=surf_jacs,
        iv_at_fwd_jacobians=iv_jacs,
        baseline=_clone_dataclass_fields(state.baseline),
        funding_leg=funding_clone,
        queries=list(state.queries),
        quanto_meta={k: dict(v) for k, v in state.quanto_meta.items()},
        asset_financing_curves=dict(state.asset_financing_curves),
        r_native_obs=state.r_native_obs.clone() if state.r_native_obs is not None else None,
        warnings=list(state.warnings),
        repo_curves_map=dict(state.repo_curves_map),
        quanto_fx_specs=dict(state.quanto_fx_specs),
        surface_digests=dict(state.surface_digests),
    )


def _normalize_repo_curves_map(
    raw_map: Any,
    as_of_dt: Any,
) -> dict[str, Any]:
    """Converts raw `RepoSpreadCurves` entries into `RepoCurve` instances for AD/QMC Greeks."""
    if not isinstance(raw_map, Mapping) or not raw_map:
        return {}
    eff_dt = _to_utc_datetime(as_of_dt) if as_of_dt is not None else None
    out: dict[str, Any] = {}
    for cid, spec in raw_map.items():
        if spec is None:
            continue
        if callable(getattr(spec, "repo_df", None)) or callable(getattr(spec, "df", None)):
            out[str(cid)] = spec
        else:
            out[str(cid)] = RepoCurve(spec, as_of_date=eff_dt)
    return out


def _freeze_obj_tensors(obj: Any) -> None:
    if obj is None:
        return
    if hasattr(obj, "__dict__"):
        for val in vars(obj).values():
            if isinstance(val, torch.Tensor):
                if val.requires_grad:
                    val.requires_grad_(False)
            elif isinstance(val, dict):
                for inner in val.values():
                    if isinstance(inner, torch.Tensor) and inner.requires_grad:
                        inner.requires_grad_(False)


def _freeze_preproc_state_tensors(state: AutocallPreprocState) -> None:
    """Detaches tensor fields before storing shared read-only entries in _PREPROC_STATE_CACHE."""
    _freeze_obj_tensors(state)
    _freeze_obj_tensors(state.seasoned_state)
    _freeze_obj_tensors(state.dividend_schedule)
    _freeze_obj_tensors(state.baseline)


def _shallow_clone_preproc_state(state: AutocallPreprocState) -> AutocallPreprocState:
    """Clones mutable tensors and Python containers while sharing immutable market objects."""
    cloned = copy.copy(state)
    cloned.node_feats = state.node_feats.clone()
    cloned.edge_feats = state.edge_feats.clone()
    cloned.asset_mask = state.asset_mask.clone()
    cloned.sched_feats = state.sched_feats.clone()
    cloned.sched_mask = state.sched_mask.clone()
    cloned.global_feats = state.global_feats.clone()
    cloned.lex_perm = state.lex_perm.copy()
    cloned.inv_lex_perm = state.inv_lex_perm.copy()
    cloned.spots_t0 = state.spots_t0.clone()
    cloned.ref_spots = state.ref_spots.clone()
    cloned.moneyness_t0 = state.moneyness_t0.clone()
    cloned.corr_matrix = state.corr_matrix.clone()
    cloned.tau_obs = state.tau_obs.clone()
    cloned.delta_tau = state.delta_tau.clone()
    cloned.tau_pay = state.tau_pay.clone()
    cloned.df_obs = state.df_obs.clone()
    cloned.df_pay = state.df_pay.clone()
    cloned.ko_barriers = state.ko_barriers.clone()
    cloned.is_callable = state.is_callable.clone()
    cloned.cpn_barriers = state.cpn_barriers.clone()
    cloned.is_guaranteed_cpn = state.is_guaranteed_cpn.clone()
    cloned.coupon_amounts = state.coupon_amounts.clone()
    cloned.forward_rel = state.forward_rel.clone()
    cloned.forward_perf = state.forward_perf.clone()
    cloned.cash_div_leverage = state.cash_div_leverage.clone()
    cloned.iv_at_fwd = state.iv_at_fwd.clone()
    cloned.v_perf = state.v_perf.clone()
    cloned.r_native_obs = state.r_native_obs.clone() if state.r_native_obs is not None else None
    cloned.original_identifiers = list(state.original_identifiers)
    cloned.sorted_identifiers = list(state.sorted_identifiers)
    cloned.vol_surfaces = dict(state.vol_surfaces)
    cloned.surface_feature_jacobians = (
        state.surface_feature_jacobians.clone()
        if isinstance(state.surface_feature_jacobians, _LazySurfaceJacobianDict)
        else {k: v.copy() for k, v in state.surface_feature_jacobians.items()}
    )
    cloned.iv_at_fwd_jacobians = (
        state.iv_at_fwd_jacobians.clone()
        if isinstance(state.iv_at_fwd_jacobians, _LazySurfaceJacobianDict)
        else {k: v.copy() for k, v in state.iv_at_fwd_jacobians.items()}
    )
    cloned.funding_leg = (
        {k: (list(v) if isinstance(v, list) else v) for k, v in state.funding_leg.items()}
        if state.funding_leg is not None
        else None
    )
    cloned.queries = list(state.queries)
    cloned.quanto_meta = {k: dict(v) for k, v in state.quanto_meta.items()}
    cloned.asset_financing_curves = dict(state.asset_financing_curves)
    cloned.warnings = list(state.warnings)
    cloned.repo_curves_map = dict(state.repo_curves_map)
    cloned.quanto_fx_specs = dict(state.quanto_fx_specs)
    cloned.surface_digests = dict(state.surface_digests)

    seasoned_copy = copy.copy(state.seasoned_state)
    seasoned_copy.original_identifiers = list(state.seasoned_state.original_identifiers)
    seasoned_copy.sorted_identifiers = list(state.seasoned_state.sorted_identifiers)
    seasoned_copy.lex_perm = state.seasoned_state.lex_perm.copy()
    seasoned_copy.inv_lex_perm = state.seasoned_state.inv_lex_perm.copy()
    seasoned_copy.spots_t0 = state.seasoned_state.spots_t0.clone()
    seasoned_copy.ref_spots = state.seasoned_state.ref_spots.clone()
    seasoned_copy.moneyness_t0 = state.seasoned_state.moneyness_t0.clone()
    seasoned_copy.tau_obs = state.seasoned_state.tau_obs.copy()
    seasoned_copy.delta_tau = state.seasoned_state.delta_tau.copy()
    seasoned_copy.tau_pay = state.seasoned_state.tau_pay.copy()
    seasoned_copy.future_coupon_rates = state.seasoned_state.future_coupon_rates.copy()
    seasoned_copy.historical_records = list(state.seasoned_state.historical_records)
    seasoned_copy.past_cashflows = list(state.seasoned_state.past_cashflows)
    seasoned_copy.future_obs_dates = list(state.seasoned_state.future_obs_dates)
    seasoned_copy.future_pay_dates = list(state.seasoned_state.future_pay_dates)
    seasoned_copy.warnings = list(state.seasoned_state.warnings)
    cloned.seasoned_state = seasoned_copy
    return cloned


_MARKET_OBJECT_CACHE = ThreadSafeLRUCache(maxsize=512, copy_fn=None)
_PREPROC_STATE_CACHE = ThreadSafeLRUCache(maxsize=512, copy_fn=_clone_preproc_state)
_FUNDING_WEIGHTS_CACHE = ThreadSafeLRUCache(maxsize=256, copy_fn=None)


def _extract_attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Extracts the first matching key or attribute from a dict or Pydantic model."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        for name in names:
            val = obj.get(name)
            if val is not None:
                return val
        return default
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
    return default


def _normalize_coupon_param(val: Any) -> float | list[float] | None:
    if val is None:
        return None
    if isinstance(val, (str, bytes)):
        return float(val)
    if isinstance(val, np.ndarray):
        return [float(x) for x in val.tolist()] if val.ndim > 0 else float(val.item())
    if isinstance(val, Sequence):
        return [float(x) for x in val]
    return float(val)


def _get_cached_yield_curve(spec: Any, as_of_dt: datetime.datetime) -> ClampedYieldCurve:
    if isinstance(spec, ClampedYieldCurve):
        return spec
    key = digest_market_bytes("yield_curve", (spec, as_of_dt))
    if key is not None:
        cached = _MARKET_OBJECT_CACHE.get(key, clone_return=False)
        if cached is not None:
            return cast(ClampedYieldCurve, cached)
    curve = ClampedYieldCurve(spec, as_of_date=as_of_dt)
    if key is not None:
        return cast(ClampedYieldCurve, _MARKET_OBJECT_CACHE.put(key, curve, clone_return=False))
    return curve


def _infer_asset_currency_5tier(
    ticker: str,
    payment_ccy: str,
    *,
    div_src: Any = None,
    vol_surfaces_map: Mapping[str, Any] | None = None,
    correlations_map: Mapping[str, float] | None = None,
    rates_bundles: Mapping[str, Any] | None = None,
) -> str:
    """Delegates asset currency resolution to the canonical 5-tier escrowed_dividend resolver."""
    return infer_asset_currency_from_context(
        ticker,
        payment_ccy,
        div_spec=div_src,
        fx_smiles=vol_surfaces_map,
        vol_surfaces=vol_surfaces_map,
        correlations=correlations_map,
        rates_bundles=rates_bundles,
    )


def _get_cached_vol_surface(
    surf_spec: Any,
    as_of_dt: datetime.datetime,
    surface_id: str | None = None,
) -> RegularizedVolSurface:
    if (
        isinstance(surf_spec, Mapping)
        and surface_id is not None
        and "Identifier" not in surf_spec
        and "identifier" not in surf_spec
    ):
        surf_spec = {**surf_spec, "Identifier": str(surface_id)}
    kwargs: dict[str, Any] = {"bc_type": "clamped"}
    if "smile_interp" in _VOL_SURF_SIG_PARAMS:
        s_interp = _extract_attr(surf_spec, "SmileInterpolationMethod", "smile_interpolation_method")
        if s_interp is not None:
            kwargs["smile_interp"] = str(getattr(s_interp, "value", s_interp))
    if "smile_extrap" in _VOL_SURF_SIG_PARAMS:
        s_extrap = _extract_attr(surf_spec, "SmileExtrapolationMethod", "smile_extrapolation_method")
        if s_extrap is not None:
            kwargs["smile_extrap"] = str(getattr(s_extrap, "value", s_extrap))
    if "surface_id" in _VOL_SURF_SIG_PARAMS:
        eff_id = surface_id or _extract_attr(surf_spec, "Identifier", "identifier")
        if eff_id is not None:
            kwargs["surface_id"] = str(eff_id)

    key = build_vol_surface_cache_key(
        surf_spec,
        as_of_dt,
        surface_id=surface_id,
        kwargs=kwargs,
    )
    if key is not None:
        cached = _MARKET_OBJECT_CACHE.get(key, clone_return=False)
        if cached is not None:
            return cast(RegularizedVolSurface, cached)
    surf = RegularizedVolSurface(surf_spec, as_of_date=as_of_dt, **kwargs)
    if key is not None:
        return cast(RegularizedVolSurface, _MARKET_OBJECT_CACHE.put(key, surf, clone_return=False))
    return surf


def _resolve_curves_and_financing(
    mkt: Any,
    model: Any,
    sorted_ids: Sequence[str],
    payment_ccy: str,
    dividends_map: Mapping[str, Any],
    predef_divs_map: Mapping[str, Any],
    as_of_dt: datetime.datetime,
) -> tuple[Any, str | None, dict[str, Any]]:
    """Resolves the payment-currency discount curve and per-asset OIS forward growth curves."""
    rates_bundles = _extract_attr(mkt, "RatesCurvesBundles", "rates_curves_bundles")
    vol_surfaces_map = _extract_attr(mkt, "VolatilitySurfaces", "volatility_surfaces", default={})
    correlations_map = _extract_attr(mkt, "Correlations", "correlations", default={})
    fin_ids_raw = _extract_attr(model, "FinancingCurveIdentifiers", "financing_curve_identifiers")
    fin_ids: list[str] = [str(x) for x in fin_ids_raw] if fin_ids_raw else []

    disc_curve_spec = _extract_attr(mkt, "discount_curve")
    disc_curve_id: str | None = None

    if disc_curve_spec is not None:
        disc_curve_id = str(_extract_attr(disc_curve_spec, "Identifier", "identifier", default="")) or None
    elif isinstance(rates_bundles, Mapping) and len(rates_bundles) > 0:
        pay_prefix = f"{payment_ccy.upper()}_"
        # Prefer a curve in FinancingCurveIdentifiers matching payment_ccy for note discounting.
        for cid in fin_ids:
            if cid in rates_bundles:
                spec_c = rates_bundles[cid]
                spec_ccy = str(_extract_attr(spec_c, "Currency", "currency", default="")).upper()
                if spec_ccy == payment_ccy.upper() or cid.upper().startswith(pay_prefix):
                    disc_curve_id = cid
                    disc_curve_spec = spec_c
                    break
        if disc_curve_spec is None:
            for cid in fin_ids:
                if cid in rates_bundles:
                    disc_curve_id = cid
                    disc_curve_spec = rates_bundles[cid]
                    break
        if disc_curve_spec is None:
            same_ccy_cands: list[tuple[tuple[int, str], int, str, Any]] = []
            fallback_cands: list[tuple[tuple[int, str], int, str, Any]] = []
            for idx, (cid, spec_c) in enumerate(rates_bundles.items()):
                cid_str = str(cid)
                spec_ccy = str(_extract_attr(spec_c, "Currency", "currency", default="")).strip().upper()
                prio = _curve_ois_priority_rank(cid_str, fin_ids)
                entry = (prio, idx, cid_str, spec_c)
                if spec_ccy == payment_ccy.upper() or cid_str.upper().startswith(pay_prefix) or cid_str.upper() == payment_ccy.upper():
                    same_ccy_cands.append(entry)
                fallback_cands.append(entry)
            chosen_list = same_ccy_cands if same_ccy_cands else fallback_cands
            chosen_list.sort(key=lambda item: (item[0], item[1]))
            disc_curve_id = chosen_list[0][2]
            disc_curve_spec = chosen_list[0][3]

    disc_curve_obj = _get_cached_yield_curve(disc_curve_spec, as_of_dt) if disc_curve_spec is not None else None

    asset_fin_curves: dict[str, Any] = {}
    if isinstance(rates_bundles, Mapping) and len(rates_bundles) > 0:
        for ticker in sorted_ids:
            div_src = predef_divs_map.get(ticker) if isinstance(predef_divs_map, Mapping) else None
            if div_src is None and isinstance(dividends_map, Mapping):
                div_src = dividends_map.get(ticker)
            inferred_ccy = _infer_asset_currency_5tier(
                ticker,
                payment_ccy,
                div_src=div_src,
                vol_surfaces_map=vol_surfaces_map if isinstance(vol_surfaces_map, Mapping) else None,
                correlations_map=correlations_map if isinstance(correlations_map, Mapping) else None,
                rates_bundles=rates_bundles,
            )

            matched_spec: Any = None
            if ticker in rates_bundles:
                matched_spec = rates_bundles[ticker]
            else:
                ccy_prefix = f"{inferred_ccy}_"
                candidates: list[tuple[int, int, Any]] = []
                for idx, (cid, spec_c) in enumerate(rates_bundles.items()):
                    cid_str = str(cid)
                    cid_u = cid_str.strip().upper()
                    spec_ccy = str(_extract_attr(spec_c, "Currency", "currency", default="")).strip().upper()
                    if spec_ccy == inferred_ccy or cid_u.startswith(ccy_prefix) or cid_u == inferred_ccy:
                        rank, _ = _curve_ois_priority_rank(cid_str, fin_ids)
                        candidates.append((rank, idx, spec_c))
                if candidates:
                    candidates.sort(key=lambda item: (item[0], item[1]))
                    matched_spec = candidates[0][2]

            if matched_spec is not None:
                asset_fin_curves[ticker] = (
                    disc_curve_obj
                    if (disc_curve_spec is not None and matched_spec is disc_curve_spec)
                    else _get_cached_yield_curve(matched_spec, as_of_dt)
                )
            elif inferred_ccy == payment_ccy.upper() and disc_curve_obj is not None:
                asset_fin_curves[ticker] = disc_curve_obj
    elif disc_curve_obj is not None:
        for ticker in sorted_ids:
            inferred_ccy = _infer_asset_currency_5tier(
                ticker,
                payment_ccy,
                vol_surfaces_map=vol_surfaces_map if isinstance(vol_surfaces_map, Mapping) else None,
                correlations_map=correlations_map if isinstance(correlations_map, Mapping) else None,
            )
            if inferred_ccy == payment_ccy.upper():
                asset_fin_curves[ticker] = disc_curve_obj

    return disc_curve_obj, disc_curve_id, asset_fin_curves


def _build_correlation_matrix(
    sorted_ids: Sequence[str],
    corr_input: Mapping[str, float] | Sequence[Sequence[float]] | torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Constructs the (D, D) correlation matrix in canonical sorted asset order in torch.float64."""
    d_assets = len(sorted_ids)
    if corr_input is None:
        if d_assets <= 1:
            return torch.eye(d_assets, dtype=torch.float64, device=device)
        raise ValueError("Missing Correlations map for multi-asset basket.")
    if isinstance(corr_input, torch.Tensor):
        t_out = corr_input.to(dtype=torch.float64, device=device)
        if t_out.shape != (d_assets, d_assets) or not bool(torch.all(torch.isfinite(t_out)).item()):
            raise ValueError(f"Correlation tensor must be finite with shape ({d_assets}, {d_assets}).")
        return t_out
    if not isinstance(corr_input, Mapping):
        arr = np.asarray(corr_input, dtype=np.float64)
        if arr.shape != (d_assets, d_assets) or not np.all(np.isfinite(arr)):
            raise ValueError(f"Correlation matrix must be finite with shape ({d_assets}, {d_assets}).")
        return torch.as_tensor(arr, dtype=torch.float64, device=device)
    if len(corr_input) == 0:
        if d_assets <= 1:
            return torch.eye(d_assets, dtype=torch.float64, device=device)
        raise ValueError("Missing Correlations map for multi-asset basket.")

    mat = np.eye(d_assets, dtype=np.float64)
    for i in range(d_assets):
        id_i = sorted_ids[i]
        diag_key = f"{id_i}&{id_i}"
        if diag_key in corr_input:
            diag_val = float(corr_input[diag_key])
            if not math.isfinite(diag_val) or abs(diag_val - 1.0) > 1e-6:
                raise ValueError(f"Invalid self-correlation {diag_key!r}={diag_val!r}; expected 1.0.")
        for j in range(i + 1, d_assets):
            id_j = sorted_ids[j]
            k1 = f"{id_i}&{id_j}"
            k2 = f"{id_j}&{id_i}"
            in_1 = k1 in corr_input
            in_2 = k2 in corr_input
            if in_1 and in_2:
                v1 = float(corr_input[k1])
                v2 = float(corr_input[k2])
                if not math.isfinite(v1) or not math.isfinite(v2):
                    raise ValueError(f"Non-finite correlation for pair {k1!r}.")
                if abs(v1 - v2) > 1e-6:
                    raise ValueError(f"Contradictory correlations {k1!r}={v1} vs {k2!r}={v2}.")
                c_val = v1
            elif in_1:
                c_val = float(corr_input[k1])
            elif in_2:
                c_val = float(corr_input[k2])
            else:
                raise ValueError(f"Missing pairwise correlation for {k1!r}.")
            if not math.isfinite(c_val) or c_val < -1.0 - 1e-9 or c_val > 1.0 + 1e-9:
                raise ValueError(f"Correlation for {k1!r}={c_val} is outside [-1, 1].")
            c_clamped = max(-1.0, min(1.0, c_val))
            mat[i, j] = c_clamped
            mat[j, i] = c_clamped
    return torch.as_tensor(mat, dtype=torch.float64, device=device)


def _extract_funding_fixings(
    fixings_map: Any,
    rate_identifier: str | None,
) -> tuple[tuple[datetime.datetime, float], ...]:
    """Extracts sorted (date, rate) pairs for a floating funding rate identifier."""
    if not rate_identifier or not isinstance(fixings_map, Mapping):
        return ()
    f_obj = fixings_map.get(rate_identifier)
    if f_obj is None:
        return ()
    if isinstance(f_obj, Mapping) and not any(
        k in f_obj for k in ("Dates", "dates", "Values", "values")
    ):
        raw_dates: Any = list(f_obj.keys())
        raw_vals: Any = list(f_obj.values())
    else:
        raw_dates = _extract_attr(f_obj, "Dates", "dates")
        raw_vals = _extract_attr(f_obj, "Values", "values")
        if raw_dates is None or raw_vals is None:
            return ()
        raw_dates = list(raw_dates)
        raw_vals = list(raw_vals)
    if len(raw_dates) != len(raw_vals):
        raise ValueError(
            f"Funding fixings Dates ({len(raw_dates)}) and Values ({len(raw_vals)}) length mismatch."
        )
    if len(raw_dates) == 0:
        return ()
    pairs: list[tuple[datetime.datetime, float]] = []
    for d, v in zip(raw_dates, raw_vals):
        f_val = float(v)
        if not math.isfinite(f_val):
            raise ValueError(f"Non-finite funding fixing value {v!r} on {d!r}.")
        pairs.append((_to_utc_datetime(d), f_val))
    pairs.sort(key=lambda item: item[0])
    return tuple(pairs)


def _lookup_funding_fixing(
    fixings_pairs: Sequence[tuple[datetime.datetime, float]],
    target_dt: datetime.datetime,
) -> float | None:
    """Finds an exact or up to 7-day prior UTC fixing for a funding reset date."""
    if not fixings_pairs:
        return None
    target_date = target_dt.date()
    best_prior_date: datetime.date | None = None
    best_prior_val: float | None = None
    for f_dt, f_val in fixings_pairs:
        if f_dt == target_dt:
            return float(f_val)
        f_date = f_dt.date()
        if f_date == target_date:
            return float(f_val)
        delta_days = (target_date - f_date).days
        if 0 < delta_days <= 7 and (best_prior_date is None or f_date >= best_prior_date):
            best_prior_date = f_date
            best_prior_val = float(f_val)
    return best_prior_val


def _apply_cached_request_metadata(
    state: AutocallPreprocState,
    queries: list[str],
    trade_info: Any,
) -> AutocallPreprocState:
    state.queries = list(queries)
    state.trade_info = str(trade_info) if trade_info is not None else None
    return state


def _lookup_preproc_cache(
    cache_key: ContentDigestKey,
    *,
    fast_shallow_cache: bool,
    queries: list[str],
    trade_info: Any,
) -> AutocallPreprocState | None:
    if fast_shallow_cache:
        cached_raw = _PREPROC_STATE_CACHE.get(cache_key, clone_return=False)
        if cached_raw is None:
            return None
        cached_clone = _shallow_clone_preproc_state(cast(AutocallPreprocState, cached_raw))
    else:
        cached_opt = cast("AutocallPreprocState | None", _PREPROC_STATE_CACHE.get(cache_key))
        if cached_opt is None:
            return None
        cached_clone = cached_opt
    return _apply_cached_request_metadata(cached_clone, queries, trade_info)


def preprocess_eq_autocall(
    req_or_raw: Any,
    *,
    spots_override: torch.Tensor | None = None,
    corr_override: torch.Tensor | None = None,
    df_pay_override: torch.Tensor | None = None,
    dividend_amount_overrides: Mapping[str, torch.Tensor] | None = None,
    as_of_date_override: datetime.datetime | datetime.date | str | None = None,
    d_max: int = D_MAX_DEFAULT,
    m_max: int = M_MAX_DEFAULT,
    _fast_shallow_cache: bool = False,
) -> AutocallPreprocState:
    """Executes the Tier 1 deterministic `torch.float64` pre-processor on an Autocallable trade."""
    if isinstance(req_or_raw, (str, Path)):
        req_obj, _ = parse_murex_autocall_dag(req_or_raw)
    else:
        req_obj = req_or_raw

    if isinstance(req_obj, EQAutocallableTradeRequest):
        product: Any = req_obj.Product
        model: Any = req_obj.Model
        mkt: Any = model.MarketDataSet
        trade_info = req_obj.TradeInfo
        queries_raw = req_obj.Queries
        as_of_raw = as_of_date_override if as_of_date_override is not None else mkt.AsOfDate
    else:
        product = _extract_attr(req_obj, "Product", "product")
        model = _extract_attr(req_obj, "Model", "model")
        mkt = _extract_attr(model, "MarketDataSet", "market_data_set") if model is not None else None
        if mkt is None:
            mkt = _extract_attr(req_obj, "MarketDataSet", "market_data")
        if product is None or mkt is None:
            raise ValueError("Request must contain Product and Model.MarketDataSet (or market_data).")
        trade_info = _extract_attr(req_obj, "TradeInfo", "trade_info", "trade_id")
        queries_raw = _extract_attr(req_obj, "Queries", "queries", default=["ProductValue", "CashFlows"])
        as_of_raw = (
            as_of_date_override
            if as_of_date_override is not None
            else _extract_attr(mkt, "AsOfDate", "as_of_date")
        )

    queries = [str(getattr(q, "value", q)) for q in queries_raw]
    as_of_dt = _to_utc_datetime(as_of_raw)

    cache_key: ContentDigestKey | None = None
    can_cache = (
        spots_override is None
        and corr_override is None
        and df_pay_override is None
        and dividend_amount_overrides is None
        and as_of_date_override is None
    )
    if can_cache and model is not None and isinstance(req_obj, BaseModel):
        if _fast_shallow_cache:
            req_d = getattr(req_obj, "__dict__", None)
            raw_hex = req_d.get("_raw_payload_digest_hex") if isinstance(req_d, dict) else None
            is_imm_raw = req_d.get("_immutable_cache_source") is not False if isinstance(req_d, dict) else True
            if (
                is_imm_raw
                and isinstance(raw_hex, str)
                and 32 <= len(raw_hex) <= 128
                and (len(raw_hex) & 1) == 0
                and _RAW_HEX_RE.match(raw_hex) is not None
            ):
                raw_key = ContentDigestKey(
                    namespace="preproc_state_raw",
                    digest_hex=f"{raw_hex}:{int(d_max)}:{int(m_max)}:{FEATURE_SCHEMA_VERSION}",
                    payload_len=len(raw_hex),
                )
                cached_hit = _lookup_preproc_cache(
                    raw_key,
                    fast_shallow_cache=True,
                    queries=queries,
                    trade_info=trade_info,
                )
                if cached_hit is not None:
                    return cached_hit
                live_key = _stream_eq_request_live(
                    req_obj,
                    d_max=int(d_max),
                    m_max=int(m_max),
                    as_of_dt=as_of_dt,
                )
                if live_key is not None:
                    live_raw = _PREPROC_STATE_CACHE.get(live_key, clone_return=False)
                    if live_raw is not None:
                        _PREPROC_STATE_CACHE.put(raw_key, live_raw, clone_return=False)
                        return _apply_cached_request_metadata(
                            _shallow_clone_preproc_state(cast(AutocallPreprocState, live_raw)),
                            queries,
                            trade_info,
                        )
                cache_key = raw_key
        if cache_key is None:
            cache_key = _stream_eq_request_live(
                req_obj,
                d_max=int(d_max),
                m_max=int(m_max),
                as_of_dt=as_of_dt,
            )
            if cache_key is not None:
                cached_hit = _lookup_preproc_cache(
                    cache_key,
                    fast_shallow_cache=_fast_shallow_cache,
                    queries=queries,
                    trade_info=trade_info,
                )
                if cached_hit is not None:
                    return cached_hit

    raw_start = _extract_attr(product, "StartDate", "start_date")
    if raw_start is None:
        raise ValueError("Product missing StartDate.")
    start_dt = _to_utc_datetime(raw_start)
    payment_ccy = str(_extract_attr(product, "PaymentCurrency", "payment_currency", default="RUB"))
    dcc_raw = _extract_attr(product, "DayCountConvention", "day_count_convention", default="Act360")
    dcc = str(getattr(dcc_raw, "value", dcc_raw))

    raw_uids = _extract_attr(product, "UnderlyingIdentifiers", "underlying_identifiers")
    if not raw_uids:
        raise ValueError("Product missing UnderlyingIdentifiers.")
    underlying_ids = [str(x) for x in raw_uids]
    raw_ref_spots = _extract_attr(product, "UnderlyingReferenceSpots", "underlying_reference_spots")
    if raw_ref_spots is None:
        raise ValueError("Product missing UnderlyingReferenceSpots.")
    ref_spots = [float(x) for x in raw_ref_spots]
    if len(ref_spots) != len(underlying_ids):
        raise ValueError("UnderlyingReferenceSpots length must match UnderlyingIdentifiers.")
    if any((not math.isfinite(r) or r <= 0.0) for r in ref_spots):
        raise ValueError("UnderlyingReferenceSpots must be strictly positive and finite.")

    periods_obj = _extract_attr(product, "periods")
    cpn_levels_for_collapser: list[float] | None
    if periods_obj is not None:
        obs_dates = [_extract_attr(p, "observation_date") for p in periods_obj]
        pay_dates = [_extract_attr(p, "payment_date") for p in periods_obj]
        ko_levels = [
            float(_extract_attr(p, "autocall_barrier", default=500.0) or 500.0) for p in periods_obj
        ]
        cpn_levels = [float(_extract_attr(p, "coupon_barrier", default=1e-5)) for p in periods_obj]
        cpn_levels_for_collapser = cpn_levels
        cpn_amounts: list[float] | None = [
            float(_extract_attr(p, "coupon_amount", default=0.0)) for p in periods_obj
        ]
        fixed_cpn_amt: float | list[float] | None = None
        fixed_rate: float | list[float] | None = None
    else:
        obs_dates = list(_extract_attr(product, "ObservationDates", "observation_dates") or ())
        pay_dates = list(_extract_attr(product, "PaymentDates", "payment_dates") or ())
        ko_levels = [
            float(x)
            for x in (_extract_attr(product, "AutoCallTriggerLevels", "autocall_trigger_levels") or ())
        ]
        raw_cpn_levels = _extract_attr(product, "CouponBarrierLevels", "coupon_barrier_levels")
        if raw_cpn_levels is not None and len(raw_cpn_levels) > 0:
            cpn_levels = [float(x) for x in raw_cpn_levels]
            cpn_levels_for_collapser = cpn_levels
        else:
            cpn_levels = [1e-5] * len(obs_dates)
            cpn_levels_for_collapser = None
        cpn_amounts = None
        fixed_cpn_amt = _normalize_coupon_param(
            _extract_attr(product, "FixedCouponAmount", "fixed_coupon_amount")
        )
        fixed_rate = _normalize_coupon_param(
            _extract_attr(product, "FixedRate", "fixed_rate")
        )

    if any(not math.isfinite(x) for x in ko_levels) or any(not math.isfinite(x) for x in cpn_levels):
        raise ValueError("Barrier levels must be finite.")
    if cpn_amounts is not None and any(not math.isfinite(x) for x in cpn_amounts):
        raise ValueError("Coupon amounts must be finite.")

    is_memory = bool(_extract_attr(product, "IsMemory", "is_memory", default=False))
    call_idx_raw = _extract_attr(
        product, "CallabilityMonitoringIndex", "callability_index", default="WorstOf"
    )
    callability_index = str(getattr(call_idx_raw, "value", call_idx_raw))

    cpn_idx_raw = _extract_attr(product, "CouponBarrierMonitoringIndex", "coupon_index")
    coupon_index = str(getattr(cpn_idx_raw, "value", cpn_idx_raw)) if cpn_idx_raw is not None else None

    opt_idx_raw = _extract_attr(product, "OptionMonitoringIndex", "option_index")
    option_index = str(getattr(opt_idx_raw, "value", opt_idx_raw)) if opt_idx_raw is not None else None

    downside_put = _extract_attr(product, "downside_put")
    modifiers = _extract_attr(product, "modifiers")

    dp_strike = _extract_attr(downside_put, "strike") if downside_put is not None else None
    strike = float(
        dp_strike if dp_strike is not None else _extract_attr(product, "Strike", "strike", default=0.90)
    )
    dp_gearing = _extract_attr(downside_put, "gearing") if downside_put is not None else None
    option_gearing = float(
        dp_gearing
        if dp_gearing is not None
        else _extract_attr(product, "OptionGearing", "option_gearing", default=1.11111)
    )
    dp_lower = _extract_attr(downside_put, "lower_strike") if downside_put is not None else None
    lower_strike_raw = (
        dp_lower if dp_lower is not None else _extract_attr(product, "LowerStrike", "lower_strike")
    )
    lower_strike = float(lower_strike_raw) if lower_strike_raw is not None else None

    dp_ki = _extract_attr(downside_put, "knock_in_barrier") if downside_put is not None else None
    ki_barrier_raw = (
        dp_ki
        if dp_ki is not None
        else _extract_attr(product, "OptionDownAndInBarrierLevel", "knock_in_barrier")
    )
    ki_barrier = float(ki_barrier_raw) if ki_barrier_raw is not None else None

    ki_obs_type_raw = _extract_attr(
        product, "OptionDownAndInBarrierObservationType", "ki_observation_type"
    )
    ki_obs_type = str(getattr(ki_obs_type_raw, "value", ki_obs_type_raw)) if ki_obs_type_raw else None
    ki_monitored_past_override = _extract_attr(product, "ki_monitored_past")
    if ki_monitored_past_override is not None:
        ki_monitored_past = bool(ki_monitored_past_override)
    elif ki_barrier is None or ki_barrier <= 0.0:
        ki_monitored_past = False
    elif ki_obs_type is None or ki_obs_type.strip().lower() == "european":
        ki_monitored_past = False
    else:
        ki_monitored_past = True

    ko_spread = float(
        _extract_attr(
            product,
            "AutoCallTriggerAbsoluteDigitalSpread",
            "autocall_digital_spread",
            default=-0.15,
        )
    )
    cpn_spread_raw = _extract_attr(
        product,
        "CouponBarrierAbsoluteDigitalSpread",
        "coupon_digital_spread",
    )
    cpn_spread = float(cpn_spread_raw) if cpn_spread_raw is not None else ko_spread

    dp_ki_spread = (
        _extract_attr(downside_put, "knock_in_digital_spread") if downside_put is not None else None
    )
    ki_spread_raw = (
        dp_ki_spread
        if dp_ki_spread is not None
        else _extract_attr(product, "OptionDownAndInBarrierAbsoluteDigitalSpread")
    )
    ki_spread = float(ki_spread_raw) if ki_spread_raw is not None else ko_spread

    allow_autocall_on_expiry = bool(
        _extract_attr(modifiers, "allow_autocall_on_expiry")
        if modifiers is not None
        else _extract_attr(product, "AllowAutocallOnExpiry", "allow_autocall_on_expiry", default=False)
    )
    force_coupons_on_autocall = bool(
        _extract_attr(modifiers, "force_coupons_on_autocall")
        if modifiers is not None
        else _extract_attr(product, "ForceCouponsOnAutocall", "force_coupons_on_autocall", default=False)
    )
    ac_cpn_raw = (
        _extract_attr(modifiers, "autocall_redemption_coupon") if modifiers is not None else None
    )
    autocall_coupon = float(
        ac_cpn_raw
        if ac_cpn_raw is not None
        else _extract_attr(product, "AutoCallCoupon", "autocall_coupon", default=0.0)
    )
    no_notional_repayment = bool(
        _extract_attr(modifiers, "no_notional_repayment")
        if modifiers is not None
        else _extract_attr(product, "NoNotionalRepayment", "no_notional_repayment", default=False)
    )
    is_long_basket = bool(
        _extract_attr(modifiers, "is_long_basket")
        if modifiers is not None
        else _extract_attr(product, "IsLongBasket", "is_long_basket", default=False)
    )
    if is_long_basket and (
        option_index is None or option_index.strip() == "" or option_index.strip() == "WorstOf"
    ):
        option_index = "AverageOf"
    part_strike_raw = (
        _extract_attr(modifiers, "participation_call_strike")
        if modifiers is not None
        else _extract_attr(product, "ParticipationCallStrike", "participation_call_strike")
    )
    part_strike = float(part_strike_raw) if part_strike_raw is not None else None
    part_rate_raw = (
        _extract_attr(modifiers, "participation_call_rate") if modifiers is not None else None
    )
    part_rate = float(
        part_rate_raw
        if part_rate_raw is not None
        else _extract_attr(product, "ParticipationCallRate", "participation_call_rate", default=0.0)
    )
    one_star_strike_raw = (
        _extract_attr(modifiers, "one_star_strike")
        if modifiers is not None
        else _extract_attr(product, "OneStarStrike", "one_star_strike")
    )
    one_star_strike = float(one_star_strike_raw) if one_star_strike_raw is not None else None
    os_cpn_raw = _extract_attr(modifiers, "one_star_coupon") if modifiers is not None else None
    one_star_coupon = float(
        os_cpn_raw
        if os_cpn_raw is not None
        else _extract_attr(product, "OneStarCoupon", "one_star_coupon", default=0.0)
    )

    for scalar_name, scalar_val in (
        ("strike", strike),
        ("option_gearing", option_gearing),
        ("lower_strike", lower_strike),
        ("ki_barrier", ki_barrier),
        ("ko_spread", ko_spread),
        ("cpn_spread", cpn_spread),
        ("ki_spread", ki_spread),
        ("autocall_coupon", autocall_coupon),
        ("part_strike", part_strike),
        ("part_rate", part_rate),
        ("one_star_strike", one_star_strike),
        ("one_star_coupon", one_star_coupon),
    ):
        if scalar_val is not None and not math.isfinite(float(scalar_val)):
            raise ValueError(f"Non-finite contract parameter {scalar_name}={scalar_val!r}.")

    cpn_barrier_type_raw = _extract_attr(
        product, "CouponBarrierType", "coupon_barrier_type", default="In"
    )
    cpn_barrier_type = str(getattr(cpn_barrier_type_raw, "value", cpn_barrier_type_raw) or "In")
    cpn_barrier_dir_raw = _extract_attr(
        product, "CouponBarrierDirection", "coupon_barrier_direction", default="Up"
    )
    cpn_barrier_direction = str(getattr(cpn_barrier_dir_raw, "value", cpn_barrier_dir_raw) or "Up")

    spots_map = _extract_attr(mkt, "Spots", "spots")
    fixings_map = _extract_attr(mkt, "Fixings", "fixings")
    repo_curves_map = _extract_attr(mkt, "RepoSpreadCurves", "repo_spread_curves", "repo_curves", default={})
    dividends_map = _extract_attr(mkt, "Dividends", "dividends", default={})
    predef_divs_map = _extract_attr(
        product,
        "PredefinedDividends",
        "predefined_dividends",
        default=_extract_attr(mkt, "predefined_dividends", default={}),
    )
    vol_surfaces_map = _extract_attr(mkt, "VolatilitySurfaces", "volatility_surfaces", default={})
    correlations_map = _extract_attr(mkt, "Correlations", "correlations")
    rates_bundles = _extract_attr(mkt, "RatesCurvesBundles", "rates_curves_bundles")
    fin_ids_raw = _extract_attr(model, "FinancingCurveIdentifiers", "financing_curve_identifiers")

    # FundingLeg extraction supporting explicit empty FundingLegPaymentDates=[] suppression
    funding_leg_obj = _extract_attr(product, "funding_leg")
    rate_id_raw = _extract_attr(product, "RateIdentifier", "rate_identifier")
    rate_id = str(rate_id_raw).strip() if rate_id_raw is not None else ""
    rate_id_clean: str | None = rate_id or None
    funding_leg_dates = _extract_attr(product, "FundingLegPaymentDates", "funding_leg_payment_dates")
    raw_fs = _extract_attr(product, "FundingSpread", "funding_spread", default=0.0)
    funding_spread = float(raw_fs if raw_fs is not None else 0.0)
    if not math.isfinite(funding_spread):
        raise ValueError(f"Non-finite FundingSpread {raw_fs!r}.")
    funding_leg_dict: dict[str, Any] | None = None
    if funding_leg_obj is not None:
        raw_f_dates = _extract_attr(funding_leg_obj, "payment_dates", default=[])
        if raw_f_dates:
            raw_rid = _extract_attr(funding_leg_obj, "rate_identifier")
            f_rid_s = str(raw_rid).strip() if raw_rid is not None else ""
            f_rid_str: str | None = f_rid_s or None
            raw_obj_fs = _extract_attr(funding_leg_obj, "funding_spread", default=0.0)
            obj_fs = float(raw_obj_fs if raw_obj_fs is not None else 0.0)
            if not math.isfinite(obj_fs):
                raise ValueError(f"Non-finite funding_leg spread {raw_obj_fs!r}.")
            funding_leg_dict = {
                "rate_identifier": f_rid_str,
                "funding_spread": obj_fs,
                "payment_dates": [_to_utc_datetime(d) for d in raw_f_dates],
                "funding_fixings": _extract_funding_fixings(fixings_map, f_rid_str),
            }
    elif funding_leg_dates is not None:
        if len(funding_leg_dates) > 0:
            funding_leg_dict = {
                "rate_identifier": rate_id_clean,
                "funding_spread": funding_spread,
                "payment_dates": [_to_utc_datetime(d) for d in funding_leg_dates],
                "funding_fixings": _extract_funding_fixings(fixings_map, rate_id_clean),
            }
    elif (rate_id_clean is not None or funding_spread != 0.0) and pay_dates:
        funding_leg_dict = {
            "rate_identifier": rate_id_clean,
            "funding_spread": funding_spread,
            "payment_dates": [_to_utc_datetime(d) for d in pay_dates],
            "funding_fixings": _extract_funding_fixings(fixings_map, rate_id_clean),
        }

    # Content-addressed cache lookup hashing 100% of market and contract inputs (for dict/custom payloads)
    if cache_key is None and can_cache:
        cache_payload = {
            "schema_ver": FEATURE_SCHEMA_VERSION,
            "d_max": int(d_max),
            "m_max": int(m_max),
            "as_of_dt": as_of_dt,
            "start_dt": start_dt,
            "payment_ccy": payment_ccy,
            "dcc": dcc,
            "underlying_ids": underlying_ids,
            "ref_spots": ref_spots,
            "obs_dates": [_to_utc_datetime(d) for d in obs_dates],
            "pay_dates": [_to_utc_datetime(d) for d in pay_dates],
            "ko_levels": ko_levels,
            "cpn_levels": cpn_levels,
            "cpn_amounts": cpn_amounts,
            "fixed_cpn_amt": fixed_cpn_amt,
            "fixed_rate": fixed_rate,
            "is_memory": is_memory,
            "callability_index": callability_index,
            "coupon_index": coupon_index,
            "option_index": option_index,
            "strike": strike,
            "option_gearing": option_gearing,
            "lower_strike": lower_strike,
            "ki_barrier": ki_barrier,
            "ki_monitored_past": ki_monitored_past,
            "ki_obs_type": ki_obs_type,
            "ko_spread": ko_spread,
            "cpn_spread": cpn_spread,
            "ki_spread": ki_spread,
            "no_notional_repayment": no_notional_repayment,
            "autocall_coupon": autocall_coupon,
            "force_coupons_on_autocall": force_coupons_on_autocall,
            "allow_autocall_on_expiry": allow_autocall_on_expiry,
            "is_long_basket": is_long_basket,
            "part_strike": part_strike,
            "part_rate": part_rate,
            "one_star_strike": one_star_strike,
            "one_star_coupon": one_star_coupon,
            "cpn_barrier_type": cpn_barrier_type,
            "cpn_barrier_direction": cpn_barrier_direction,
            "funding_leg": funding_leg_dict,
            "fin_ids": [str(x) for x in fin_ids_raw] if fin_ids_raw else None,
            "disc_curve_spec": _extract_attr(mkt, "discount_curve"),
            "rates_bundles": rates_bundles,
            "spots_map": spots_map,
            "fixings_map": fixings_map,
            "repo_curves_map": repo_curves_map,
            "dividends_map": dividends_map,
            "predef_divs_map": predef_divs_map,
            "vol_surfaces_map": vol_surfaces_map,
            "correlations_map": correlations_map,
        }
        cache_key = digest_market_bytes("preproc_state", cache_payload)
        if cache_key is not None:
            cached_hit = _lookup_preproc_cache(
                cache_key,
                fast_shallow_cache=_fast_shallow_cache,
                queries=queries,
                trade_info=trade_info,
            )
            if cached_hit is not None:
                return cached_hit

    # Canonical lexicographical sorting order needed to resolve asset financing curves
    sorted_preview_ids = sorted(underlying_ids)
    disc_curve_obj, disc_curve_id, asset_fin_curves = _resolve_curves_and_financing(
        mkt=mkt,
        model=model,
        sorted_ids=sorted_preview_ids,
        payment_ccy=payment_ccy,
        dividends_map=dividends_map if isinstance(dividends_map, Mapping) else {},
        predef_divs_map=predef_divs_map if isinstance(predef_divs_map, Mapping) else {},
        as_of_dt=as_of_dt,
    )

    if funding_leg_dict is not None:
        f_rid = funding_leg_dict.get("rate_identifier")
        f_rid_clean = str(f_rid).strip() if f_rid is not None else ""
        f_curve_obj: Any = None
        f_curve_id: str | None = disc_curve_id
        if f_rid_clean:
            f_curve_id = f_rid_clean
            if isinstance(rates_bundles, Mapping) and f_rid_clean in rates_bundles:
                f_curve_obj = (
                    disc_curve_obj
                    if (disc_curve_id is not None and f_rid_clean == disc_curve_id)
                    else _get_cached_yield_curve(rates_bundles[f_rid_clean], as_of_dt)
                )
            else:
                f_curve_obj = disc_curve_obj
                f_curve_id = disc_curve_id
        else:
            f_curve_obj = disc_curve_obj
        funding_leg_dict["funding_curve"] = f_curve_obj
        funding_leg_dict["funding_curve_id"] = f_curve_id

    _, df_fin_fn = build_discount_rate_evaluator(disc_curve_obj, as_of_dt)

    if spots_override is None and not isinstance(spots_map, Mapping):
        raise ValueError("Missing MarketDataSet.Spots mapping.")

    # 1. Seasoned Schedule Partitioning & Markovian Collapse
    seasoned = collapse_seasoned_schedule(
        as_of_date=as_of_dt,
        start_date=start_dt,
        observation_dates=obs_dates,
        payment_dates=pay_dates,
        underlying_identifiers=underlying_ids,
        underlying_reference_spots=ref_spots,
        spots_t0=spots_override if spots_override is not None else spots_map,
        ko_barriers=ko_levels,
        cpn_barriers=cpn_levels_for_collapser,
        fixed_coupon_amount=fixed_cpn_amt,
        fixed_rate=fixed_rate,
        coupon_amounts=cpn_amounts,
        day_count_convention=dcc,
        time_day_count_convention="Act365",
        is_memory=is_memory,
        callability_index=callability_index,
        coupon_index=coupon_index,
        option_index=option_index,
        fixings=fixings_map,
        ki_barrier=ki_barrier,
        ki_monitored_past=ki_monitored_past,
        allow_autocall_on_expiry=allow_autocall_on_expiry,
        force_coupons_on_autocall=force_coupons_on_autocall,
        autocall_coupon=autocall_coupon,
        no_notional_repayment=no_notional_repayment,
        payment_currency=payment_ccy,
        df_pay_fn=df_fin_fn,
        coupon_barrier_type=cpn_barrier_type,
        coupon_barrier_direction=cpn_barrier_direction,
        strike=strike,
        option_gearing=option_gearing,
        lower_strike=lower_strike,
        is_long_basket=is_long_basket,
        participation_call_strike=part_strike,
        participation_call_rate=part_rate,
        one_star_strike=one_star_strike,
        one_star_coupon=one_star_coupon,
        funding_leg=funding_leg_dict,
    )
    if not hasattr(seasoned, "warnings") or getattr(seasoned, "warnings", None) is None:
        setattr(seasoned, "warnings", [])
    seasoned_warnings: list[str] = getattr(seasoned, "warnings")
    if (
        ki_obs_type_raw is None
        and ki_monitored_past_override is None
        and ki_barrier is not None
        and float(ki_barrier) > 0.0
    ):
        if "KI_OBS_TYPE_DEFAULTED_EUROPEAN" not in seasoned_warnings:
            seasoned_warnings.append("KI_OBS_TYPE_DEFAULTED_EUROPEAN")

    sorted_ids = seasoned.sorted_identifiers
    d_assets = len(sorted_ids)
    m_rem = len(seasoned.future_indices)
    device = seasoned.spots_t0.device

    # 2. Escrowed/Affine Cash Dividend Forward & Quanto Engine
    div_out = compute_escrowed_dividend_forwards(
        identifiers=sorted_ids,
        as_of_date=as_of_dt,
        observation_dates=seasoned.future_obs_dates,
        spots_t0=seasoned.spots_t0,
        ref_spots=seasoned.ref_spots,
        discount_curve=disc_curve_obj,
        asset_financing_curves=asset_fin_curves,
        repo_curves=repo_curves_map,
        market_dividends=dividends_map,
        predefined_dividends=predef_divs_map,
        dividend_amount_overrides=dividend_amount_overrides,
        payment_currency=payment_ccy,
        vol_surfaces=vol_surfaces_map,
        correlations=correlations_map,
    )

    # 3. Regularized Volatility Surfaces & Schedule Implied Volatilities
    reg_surfaces: dict[str, RegularizedVolSurface] = {}
    surface_digests: dict[str, str] = {}
    iv_at_fwd_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    d_iv_dk_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    node_surf_feats_np = np.zeros((d_assets, 7), dtype=np.float64)
    surf_tau_obs_map: dict[str, npt.NDArray[np.float64]] = {}
    surf_tau_obs_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    fwd_price_np = div_out.forward_price.detach().cpu().numpy().astype(np.float64)
    fwd_req_grad = bool(div_out.forward_price.requires_grad)

    if m_rem > 0 and not seasoned.already_autocalled:
        if not isinstance(vol_surfaces_map, Mapping):
            raise ValueError("Missing MarketDataSet.VolatilitySurfaces mapping.")
        for i, ticker in enumerate(sorted_ids):
            if ticker not in vol_surfaces_map:
                raise ValueError(f"Missing EquityVolatilitySmileCollection for underlying {ticker!r}.")
            surf_spec = vol_surfaces_map[ticker]
            surf_i = _get_cached_vol_surface(surf_spec, as_of_dt, surface_id=ticker)
            reg_surfaces[ticker] = surf_i
            surface_digests[ticker] = compute_surface_digest(surf_i)
            surf_warns = getattr(surf_i, "warnings", None)
            if isinstance(surf_warns, Sequence) and not isinstance(surf_warns, (str, bytes)):
                for w_msg in surf_warns:
                    if str(w_msg) not in seasoned_warnings:
                        seasoned_warnings.append(str(w_msg))

            raw_surf_dcc = getattr(surf_i, "dcc", None) or _extract_attr(
                surf_spec, "DayCountConvention", "day_count_convention", default="Act365"
            )
            surf_dcc = str(getattr(raw_surf_dcc, "value", raw_surf_dcc))
            tau_surf_obs = (
                np.asarray(seasoned.tau_obs, dtype=np.float64)
                if surf_dcc == "Act365"
                else year_fractions(as_of_dt, seasoned.future_obs_dates, surf_dcc)
            )
            surf_tau_obs_map[ticker] = tau_surf_obs
            surf_tau_obs_np[i, :] = tau_surf_obs

            f_row = fwd_price_np[i, :]
            if _HAS_IV_BATCH:
                iv_batch_fn = getattr(surf_i, "iv_batch")
                iv_at_fwd_np[i, :] = np.asarray(
                    iv_batch_fn(tau_surf_obs, f_row), dtype=np.float64
                )
                if fwd_req_grad:
                    eps_k = np.maximum(1e-5 * f_row, 1e-6)
                    f_hi = f_row + eps_k
                    f_lo = np.maximum(f_row - eps_k, 1e-6)
                    iv_hi = np.asarray(iv_batch_fn(tau_surf_obs, f_hi), dtype=np.float64)
                    iv_lo = np.asarray(iv_batch_fn(tau_surf_obs, f_lo), dtype=np.float64)
                    d_iv_dk_np[i, :] = (iv_hi - iv_lo) / (f_hi - f_lo)
            else:
                for m in range(m_rem):
                    t_m = float(tau_surf_obs[m])
                    f_im = float(f_row[m])
                    iv_at_fwd_np[i, m] = float(surf_i.iv(t_m, f_im))
                    if fwd_req_grad:
                        eps_k_s = max(1e-5 * f_im, 1e-6)
                        f_hi_s = f_im + eps_k_s
                        f_lo_s = max(f_im - eps_k_s, 1e-6)
                        iv_hi_s = float(surf_i.iv(t_m, f_hi_s))
                        iv_lo_s = float(surf_i.iv(t_m, f_lo_s))
                        d_iv_dk_np[i, m] = (iv_hi_s - iv_lo_s) / (f_hi_s - f_lo_s)

            t_short = float(tau_surf_obs[0])
            f_short = float(f_row[0])
            t_term = float(tau_surf_obs[-1])
            f_term = float(f_row[-1])
            node_surf_feats_np[i, :] = surf_i.extract_node_surface_features_values_only(
                t_short,
                f_short,
                t_term,
                f_term,
                iv_short_val=float(iv_at_fwd_np[i, 0]),
                iv_term_val=float(iv_at_fwd_np[i, -1]),
            )

    id_to_idx = {ticker: idx for idx, ticker in enumerate(sorted_ids)}

    def _build_surf_feat_jac(ticker: str) -> npt.NDArray[np.float64]:
        if ticker not in reg_surfaces or m_rem == 0:
            return np.zeros((7, 1, 1), dtype=np.float64)
        surf_i = reg_surfaces[ticker]
        idx = id_to_idx[ticker]
        tau_s = surf_tau_obs_map[ticker]
        _, jac_7d = surf_i.extract_node_surface_features(
            float(tau_s[0]),
            float(fwd_price_np[idx, 0]),
            float(tau_s[-1]),
            float(fwd_price_np[idx, -1]),
        )
        return jac_7d

    def _build_iv_fwd_jac(ticker: str) -> npt.NDArray[np.float64]:
        if ticker not in reg_surfaces or m_rem == 0:
            return np.zeros((0, 1, 1), dtype=np.float64)
        surf_i = reg_surfaces[ticker]
        idx = id_to_idx[ticker]
        tau_s = surf_tau_obs_map[ticker]
        return np.stack(
            [
                surf_i.iv_and_d_iv_d_quotes(float(tau_s[m]), float(fwd_price_np[idx, m]))[1]
                for m in range(m_rem)
            ],
            axis=0,
        )

    jac_keys = list(reg_surfaces.keys())
    surf_feat_jacs: dict[str, npt.NDArray[np.float64]] = _LazySurfaceJacobianDict(
        _build_surf_feat_jac, jac_keys
    )
    iv_fwd_jacs: dict[str, npt.NDArray[np.float64]] = _LazySurfaceJacobianDict(
        _build_iv_fwd_jac, jac_keys
    )

    iv_at_fwd_t = torch.as_tensor(iv_at_fwd_np, dtype=torch.float64, device=device)
    if div_out.forward_price.requires_grad and m_rem > 0 and not seasoned.already_autocalled:
        d_iv_dk_t = torch.as_tensor(d_iv_dk_np, dtype=torch.float64, device=device)
        iv_at_fwd_t = iv_at_fwd_t + d_iv_dk_t * (
            div_out.forward_price - div_out.forward_price.detach()
        )

    tau_obs_t = torch.as_tensor(seasoned.tau_obs, dtype=torch.float64, device=device)
    delta_tau_t = torch.as_tensor(seasoned.delta_tau, dtype=torch.float64, device=device)
    tau_pay_t = torch.as_tensor(seasoned.tau_pay, dtype=torch.float64, device=device)

    all_use_disc_curve = d_assets > 0 and all(
        asset_fin_curves.get(t, disc_curve_obj) is disc_curve_obj for t in sorted_ids
    )
    if all_use_disc_curve:
        df_obs_t = div_out.df_fin_obs[0]
    else:
        df_obs_np = np.array([df_fin_fn(d) for d in seasoned.future_obs_dates], dtype=np.float64)
        df_obs_t = torch.as_tensor(df_obs_np, dtype=torch.float64, device=device)

    if df_pay_override is not None:
        df_pay_t = df_pay_override.to(dtype=torch.float64, device=device)
    elif m_rem > 0:
        df_pay_np = np.array([df_fin_fn(d) for d in seasoned.future_pay_dates], dtype=np.float64)
        df_pay_t = torch.as_tensor(df_pay_np, dtype=torch.float64, device=device)
    else:
        df_pay_t = torch.empty(0, dtype=torch.float64, device=device)

    surf_tau_obs_t = torch.as_tensor(surf_tau_obs_np, dtype=torch.float64, device=device)
    v_perf_t = iv_at_fwd_t * torch.sqrt(surf_tau_obs_t.clamp(min=1e-12))

    if corr_override is not None:
        corr_t = corr_override.to(dtype=torch.float64, device=device)
    else:
        corr_t = _build_correlation_matrix(sorted_ids, correlations_map, device)

    assert seasoned.sentinel_barriers is not None
    sb = seasoned.sentinel_barriers
    ko_barriers_t = torch.as_tensor(sb.ko_barriers_raw, dtype=torch.float64, device=device)
    is_callable_t = torch.as_tensor(sb.is_callable, dtype=torch.float64, device=device)
    cpn_barriers_t = torch.as_tensor(sb.cpn_barriers_raw, dtype=torch.float64, device=device)
    is_guaranteed_t = torch.as_tensor(sb.is_guaranteed_cpn, dtype=torch.float64, device=device)
    coupon_amounts_t = torch.as_tensor(seasoned.future_coupon_rates, dtype=torch.float64, device=device)

    # 4. Analytical Stulz-Margrabe Moment-Matched Basket Proxy
    proxy_inputs: dict[str, Any] = {
        "forward_perf": div_out.forward_perf,
        "v_perf": v_perf_t,
        "corr_matrix": corr_t,
        "df_pay": df_pay_t,
        "ko_barriers": ko_barriers_t,
        "is_callable": is_callable_t,
        "cpn_barriers": cpn_barriers_t,
        "is_guaranteed_cpn": is_guaranteed_t,
        "coupon_amounts": coupon_amounts_t,
        "strike": strike,
        "option_gearing": option_gearing,
        "lower_strike": lower_strike,
        "ki_barrier": ki_barrier,
        "i_ki_past": seasoned.i_ki_past,
        "c_mem_0": seasoned.c_mem_0,
        "is_memory": is_memory,
        "ko_spread": ko_spread,
        "cpn_spread": cpn_spread,
        "ki_spread": ki_spread,
        "no_notional_repayment": no_notional_repayment,
        "autocall_coupon": autocall_coupon,
        "force_coupons_on_autocall": force_coupons_on_autocall,
        "allow_autocall_on_expiry": allow_autocall_on_expiry,
        "coupon_barrier_type": cpn_barrier_type,
        "coupon_barrier_direction": cpn_barrier_direction,
        "is_long_basket": is_long_basket,
        "pv_in_transit": seasoned.pv_in_transit,
        "already_autocalled": seasoned.already_autocalled,
        "callability_index": callability_index,
        "coupon_index": coupon_index,
        "option_index": option_index,
        "inv_lex_perm": seasoned.inv_lex_perm,
        "tau_obs": tau_obs_t,
        "participation_call_strike": part_strike,
        "participation_call_rate": part_rate,
        "one_star_strike": one_star_strike,
        "one_star_coupon": one_star_coupon,
    }
    baseline_kwargs: dict[str, Any] = {}
    if "use_fast_cpu" in _BASELINE_SIG_PARAMS:
        baseline_kwargs["use_fast_cpu"] = None
    baseline = compute_analytical_autocall_baseline(proxy_inputs, **baseline_kwargs)

    # 5. Fixed-Padding Feature Tensors in torch.float64 (preserving autograd graph)
    d_pad = max(d_max, d_assets)
    m_pad = max(m_max, m_rem)
    n_node_cols = len(AutocallNodeCol)
    n_edge_cols = len(AutocallEdgeCol)
    n_sched_cols = len(AutocallSchedCol)
    n_global_cols = len(AutocallGlobalCol)

    # 5a. Node Features: (1, d_pad, 16)
    log_m0 = torch.log(seasoned.moneyness_t0.clamp(min=1e-12))
    if m_rem > 0:
        tau_term = tau_obs_t[-1].clamp(min=1e-6)
        log_fwd_carry_1y = torch.log(div_out.forward_rel[:, -1].clamp(min=1e-12)) / tau_term
        log_div_lev = torch.log(div_out.cash_div_leverage[:, -1].clamp(min=1.0))
        f_p_term = div_out.forward_perf[:, -1].clamp(min=1e-12)
        v_term = v_perf_t[:, -1].clamp(min=1e-6)
        k_eff = max(strike, 1e-4)
        dist_to_put = torch.log(f_p_term / k_eff) / v_term
    else:
        log_fwd_carry_1y = torch.zeros(d_assets, dtype=torch.float64, device=device)
        log_div_lev = torch.zeros(d_assets, dtype=torch.float64, device=device)
        dist_to_put = torch.zeros(d_assets, dtype=torch.float64, device=device)

    surf_feats_t = torch.as_tensor(node_surf_feats_np, dtype=torch.float64, device=device)
    active_ones = torch.ones(d_assets, dtype=torch.float64, device=device)

    # Set pair role bits only when an active monitoring index depends on asset declaration order.
    role_leg_a = torch.zeros(d_assets, dtype=torch.float64, device=device)
    role_leg_b = torch.zeros(d_assets, dtype=torch.float64, device=device)
    active_indices = (
        callability_index,
        coupon_index or callability_index,
        option_index or callability_index,
    )
    if d_assets >= 2 and any(idx in PAIR_ORDER_DEPENDENT_INDICES for idx in active_indices):
        role_leg_a[int(seasoned.inv_lex_perm[0])] = 1.0
        role_leg_b[int(seasoned.inv_lex_perm[1])] = 1.0

    node_cols: list[torch.Tensor] = [torch.zeros(d_assets, dtype=torch.float64, device=device)] * n_node_cols
    node_cols[AutocallNodeCol.LOG_MONEYNESS] = log_m0
    node_cols[AutocallNodeCol.LOG_FWD_CARRY_1Y] = log_fwd_carry_1y
    node_cols[AutocallNodeCol.CASH_DIV_LEVERAGE] = log_div_lev
    node_cols[AutocallNodeCol.REPO_SPREAD_AVG] = div_out.repo_spread_avg
    node_cols[AutocallNodeCol.IV_SHORT_ATM] = surf_feats_t[:, 0]
    node_cols[AutocallNodeCol.IV_TERM_ATM] = surf_feats_t[:, 1]
    node_cols[AutocallNodeCol.IV_TERM_SLOPE] = surf_feats_t[:, 2]
    node_cols[AutocallNodeCol.SSVI_SKEW_RHO] = surf_feats_t[:, 3]
    node_cols[AutocallNodeCol.SSVI_CURV_ETA] = surf_feats_t[:, 4]
    node_cols[AutocallNodeCol.SMILE_RESIDUAL_PC1] = surf_feats_t[:, 5]
    node_cols[AutocallNodeCol.SMILE_RESIDUAL_PC2] = surf_feats_t[:, 6]
    node_cols[AutocallNodeCol.DIST_TO_PUT_STRIKE] = dist_to_put
    node_cols[AutocallNodeCol.PROXY_DELTA_WEIGHT] = baseline.proxy_delta_weights
    node_cols[AutocallNodeCol.ASSET_ACTIVE_MASK] = active_ones
    node_cols[AutocallNodeCol.ROLE_LEG_A] = role_leg_a
    node_cols[AutocallNodeCol.ROLE_LEG_B] = role_leg_b
    node_active = torch.stack(node_cols, dim=1)

    if d_pad > d_assets:
        node_pad = torch.zeros((d_pad - d_assets, n_node_cols), dtype=torch.float64, device=device)
        node_feats = torch.cat([node_active, node_pad], dim=0).unsqueeze(0)
        asset_mask = torch.cat(
            [active_ones, torch.zeros(d_pad - d_assets, dtype=torch.float64, device=device)],
            dim=0,
        ).unsqueeze(0)
    else:
        node_feats = node_active.unsqueeze(0)
        asset_mask = active_ones.unsqueeze(0)

    # 5b. Edge Features: (1, d_pad, d_pad, 4)
    if m_rem > 0:
        v_term_vec = v_perf_t[:, -1]
        log_fp_term = torch.log(div_out.forward_perf[:, -1].clamp(min=1e-12))
    else:
        v_term_vec = torch.zeros(d_assets, dtype=torch.float64, device=device)
        log_fp_term = log_m0

    cov_active = corr_t * v_term_vec.unsqueeze(1) * v_term_vec.unsqueeze(0)
    log_perf_diff_active = torch.abs(log_fp_term.unsqueeze(1) - log_fp_term.unsqueeze(0))
    off_diag_active = 1.0 - torch.eye(d_assets, dtype=torch.float64, device=device)

    edge_cols: list[torch.Tensor] = [
        torch.zeros((d_assets, d_assets), dtype=torch.float64, device=device)
    ] * n_edge_cols
    edge_cols[AutocallEdgeCol.CORR_RHO] = corr_t
    edge_cols[AutocallEdgeCol.COVARIANCE_TERM] = cov_active
    edge_cols[AutocallEdgeCol.LOG_PERF_DIFF] = log_perf_diff_active
    edge_cols[AutocallEdgeCol.OFF_DIAG_MASK] = off_diag_active
    edge_active = torch.stack(edge_cols, dim=-1)

    if d_pad > d_assets:
        edge_full = torch.zeros((d_pad, d_pad, n_edge_cols), dtype=torch.float64, device=device)
        edge_full[:d_assets, :d_assets, :] = edge_active
        edge_feats = edge_full.unsqueeze(0)
    else:
        edge_feats = edge_active.unsqueeze(0)

    # 5c. Schedule Features: (1, m_pad, 12)
    if m_rem > 0:
        log_ko_feat_t = torch.as_tensor(sb.log_ko_barrier_feat, dtype=torch.float64, device=device)
        log_cpn_feat_t = torch.as_tensor(sb.log_cpn_barrier_feat, dtype=torch.float64, device=device)
        log_fw_t = torch.log(baseline.f_w.clamp(min=1e-12))
        sched_cols: list[torch.Tensor] = [
            torch.zeros(m_rem, dtype=torch.float64, device=device)
        ] * n_sched_cols
        sched_cols[AutocallSchedCol.TAU_OBS] = tau_obs_t
        sched_cols[AutocallSchedCol.DELTA_TAU] = delta_tau_t
        sched_cols[AutocallSchedCol.DF_PAY] = df_pay_t
        sched_cols[AutocallSchedCol.LOG_KO_BARRIER] = log_ko_feat_t
        sched_cols[AutocallSchedCol.IS_CALLABLE_FLAG] = is_callable_t
        sched_cols[AutocallSchedCol.LOG_CPN_BARRIER] = log_cpn_feat_t
        sched_cols[AutocallSchedCol.IS_GUARANTEED_CPN] = is_guaranteed_t
        sched_cols[AutocallSchedCol.COUPON_RATE] = coupon_amounts_t
        sched_cols[AutocallSchedCol.STULZ_FWD_W] = log_fw_t
        sched_cols[AutocallSchedCol.STULZ_VOL_W] = baseline.v_w
        sched_cols[AutocallSchedCol.PROXY_KO_PROB] = baseline.p_proxy
        sched_cols[AutocallSchedCol.PROXY_CPN_PROB] = baseline.q_proxy
        sched_active = torch.stack(sched_cols, dim=1)
        sched_ones = torch.ones(m_rem, dtype=torch.float64, device=device)
        if m_pad > m_rem:
            sched_pad = torch.zeros((m_pad - m_rem, n_sched_cols), dtype=torch.float64, device=device)
            sched_feats = torch.cat([sched_active, sched_pad], dim=0).unsqueeze(0)
            sched_mask = torch.cat(
                [sched_ones, torch.zeros(m_pad - m_rem, dtype=torch.float64, device=device)],
                dim=0,
            ).unsqueeze(0)
        else:
            sched_feats = sched_active.unsqueeze(0)
            sched_mask = sched_ones.unsqueeze(0)
    else:
        sched_feats = torch.zeros((1, m_pad, n_sched_cols), dtype=torch.float64, device=device)
        sched_mask = torch.zeros((1, m_pad), dtype=torch.float64, device=device)

    # 5d. Global Context Vector: (1, 16)
    has_ki = 1.0 if (ki_barrier is not None and ki_barrier > 0.0) else 0.0
    b_ki_val = float(ki_barrier) if (ki_barrier is not None and ki_barrier > 0.0) else float(strike)
    lower_k_val = float(lower_strike) if lower_strike is not None else 0.0
    call_code = float(monitoring_index_to_code(callability_index))
    cpn_code = float(monitoring_index_to_code(coupon_index or callability_index))
    opt_code = float(monitoring_index_to_code(option_index or callability_index))

    global_cols: list[torch.Tensor] = [torch.tensor(0.0, dtype=torch.float64, device=device)] * n_global_cols
    global_cols[AutocallGlobalCol.K_PUT] = torch.tensor(float(strike), dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.B_KI] = torch.tensor(b_ki_val, dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.HAS_KI_FLAG] = torch.tensor(has_ki, dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.I_KI_PAST] = torch.tensor(float(seasoned.i_ki_past), dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.G_PUT] = torch.tensor(float(option_gearing), dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.C_MEM_0] = torch.tensor(float(seasoned.c_mem_0), dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.IS_MEMORY_FLAG] = torch.tensor(1.0 if is_memory else 0.0, dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.ABS_KO_SPREAD] = torch.tensor(abs(float(ko_spread)), dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.ABS_CPN_SPREAD] = torch.tensor(abs(float(cpn_spread)), dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.L_PUT_PROXY] = baseline.l_put_proxy
    global_cols[AutocallGlobalCol.LOWER_STRIKE] = torch.tensor(lower_k_val, dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.CALL_INDEX_ID] = torch.tensor(call_code, dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.CPN_INDEX_ID] = torch.tensor(cpn_code, dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.OPT_INDEX_ID] = torch.tensor(opt_code, dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.IS_LONG_BASKET] = torch.tensor(1.0 if is_long_basket else 0.0, dtype=torch.float64, device=device)
    global_cols[AutocallGlobalCol.ABS_KI_SPREAD] = torch.tensor(abs(float(ki_spread)), dtype=torch.float64, device=device)
    global_vec = torch.stack(global_cols, dim=0).unsqueeze(0)

    state_out = AutocallPreprocState(
        node_feats=node_feats,
        edge_feats=edge_feats,
        asset_mask=asset_mask,
        sched_feats=sched_feats,
        sched_mask=sched_mask,
        global_feats=global_vec,
        d_assets=d_assets,
        m_rem=m_rem,
        d_max=d_pad,
        m_max=m_pad,
        original_identifiers=seasoned.original_identifiers,
        sorted_identifiers=sorted_ids,
        lex_perm=seasoned.lex_perm,
        inv_lex_perm=seasoned.inv_lex_perm,
        spots_t0=seasoned.spots_t0,
        ref_spots=seasoned.ref_spots,
        moneyness_t0=seasoned.moneyness_t0,
        corr_matrix=corr_t,
        tau_obs=tau_obs_t,
        delta_tau=delta_tau_t,
        tau_pay=tau_pay_t,
        df_obs=df_obs_t,
        df_pay=df_pay_t,
        ko_barriers=ko_barriers_t,
        is_callable=is_callable_t,
        cpn_barriers=cpn_barriers_t,
        is_guaranteed_cpn=is_guaranteed_t,
        coupon_amounts=coupon_amounts_t,
        forward_rel=div_out.forward_rel,
        forward_perf=div_out.forward_perf,
        cash_div_leverage=div_out.cash_div_leverage,
        iv_at_fwd=iv_at_fwd_t,
        v_perf=v_perf_t,
        seasoned_state=seasoned,
        dividend_schedule=div_out,
        vol_surfaces=reg_surfaces,
        surface_feature_jacobians=surf_feat_jacs,
        iv_at_fwd_jacobians=iv_fwd_jacs,
        discount_curve=disc_curve_obj,
        baseline=baseline,
        strike=strike,
        option_gearing=option_gearing,
        lower_strike=lower_strike,
        ki_barrier=ki_barrier,
        i_ki_past=seasoned.i_ki_past,
        c_mem_0=seasoned.c_mem_0,
        is_memory=is_memory,
        ko_spread=ko_spread,
        cpn_spread=cpn_spread,
        ki_spread=ki_spread,
        no_notional_repayment=no_notional_repayment,
        autocall_coupon=autocall_coupon,
        force_coupons_on_autocall=force_coupons_on_autocall,
        allow_autocall_on_expiry=allow_autocall_on_expiry,
        participation_call_strike=part_strike,
        participation_call_rate=part_rate,
        one_star_strike=one_star_strike,
        one_star_coupon=one_star_coupon,
        is_long_basket=is_long_basket,
        payment_currency=payment_ccy,
        callability_index=callability_index,
        coupon_index=coupon_index,
        option_index=option_index,
        funding_leg=funding_leg_dict,
        pv_in_transit=seasoned.pv_in_transit,
        already_autocalled=seasoned.already_autocalled,
        trade_info=str(trade_info) if trade_info is not None else None,
        queries=queries,
        coupon_barrier_type=cpn_barrier_type,
        coupon_barrier_direction=cpn_barrier_direction,
        cash_to_settle_today=seasoned.cash_to_settle_today,
        quanto_meta=div_out.quanto_meta,
        asset_financing_curves=asset_fin_curves,
        discount_curve_id=disc_curve_id,
        r_native_obs=(
            getattr(div_out, "r_native_obs").to(dtype=torch.float64, device=device)
            if isinstance(getattr(div_out, "r_native_obs", None), torch.Tensor)
            else (
                torch.as_tensor(getattr(div_out, "r_native_obs"), dtype=torch.float64, device=device)
                if getattr(div_out, "r_native_obs", None) is not None
                else None
            )
        ),
        warnings=list(seasoned_warnings),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        repo_curves_map=_normalize_repo_curves_map(repo_curves_map, as_of_dt),
        surface_digests=surface_digests,
    )
    state_out.state_digest = compute_preproc_state_digest(state_out)
    if cache_key is not None:
        _freeze_preproc_state_tensors(state_out)
        _PREPROC_STATE_CACHE.put(cache_key, state_out, clone_return=False)
        if _fast_shallow_cache:
            return _shallow_clone_preproc_state(state_out)
        return _clone_preproc_state(state_out)
    return state_out


def _batch_eval_curve_dfs(
    curve_obj: Any,
    df_fn: Callable[[datetime.datetime], Any],
    as_of_dt: datetime.datetime,
    target_dts: Sequence[datetime.datetime],
) -> npt.NDArray[np.float64]:
    """Evaluates discount factors for multiple future dates in a single vectorized curve call."""
    n_dts = len(target_dts)
    if n_dts == 0:
        return np.zeros(0, dtype=np.float64)
    if curve_obj is None:
        return np.ones(n_dts, dtype=np.float64)
    if isinstance(curve_obj, ClampedYieldCurve):
        taus = (
            year_fractions(as_of_dt, list(target_dts), curve_obj.dcc)
            if curve_obj.as_of_date is None
            else None
        )
        raw = curve_obj.df(list(target_dts) if taus is None else taus)
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        return np.where(np.asarray([dt <= as_of_dt for dt in target_dts]), 1.0, arr)
    eval_batch = getattr(df_fn, "_eval_tau_batch", None)
    if callable(eval_batch):
        raw_dcc = getattr(curve_obj, "dcc", "Act365")
        taus = year_fractions(as_of_dt, list(target_dts), str(raw_dcc))
        arr = np.asarray(eval_batch(taus), dtype=np.float64).reshape(-1)
        return np.where(taus <= 0.0, 1.0, arr)
    return np.asarray([float(df_fn(dt)) for dt in target_dts], dtype=np.float64)


def _map_future_funding_intervals(
    start_dt: datetime.datetime,
    as_of_dt: datetime.datetime,
    max_past_pay_dt: datetime.datetime | None,
    f_dates: Sequence[datetime.datetime],
    future_pay_dates: Sequence[datetime.datetime],
) -> list[tuple[datetime.datetime, datetime.datetime, int]]:
    """Map future funding accrual periods onto indices of `future_pay_dates`."""
    m_pay = len(future_pay_dates)
    if m_pay == 0:
        return []
    max_future_pay_dt = future_pay_dates[-1]
    is_pay_sorted = all(
        future_pay_dates[idx] <= future_pay_dates[idx + 1] for idx in range(m_pay - 1)
    )
    intervals: list[tuple[datetime.datetime, datetime.datetime, int]] = []
    prev_f_dt = start_dt
    for f_dt in f_dates:
        if f_dt <= as_of_dt:
            prev_f_dt = f_dt
            continue
        if max_past_pay_dt is not None and f_dt <= max_past_pay_dt:
            prev_f_dt = f_dt
            continue
        if f_dt > max_future_pay_dt:
            prev_f_dt = f_dt
            continue
        if is_pay_sorted:
            m_idx = min(bisect_left(future_pay_dates, f_dt), m_pay - 1)
        else:
            m_idx = m_pay - 1
            for m_cand in range(m_pay):
                if future_pay_dates[m_cand] >= f_dt:
                    m_idx = m_cand
                    break
        intervals.append((prev_f_dt, f_dt, m_idx))
        prev_f_dt = f_dt
    return intervals


def _compute_funding_period_weights_np(
    preproc_state: AutocallPreprocState,
    m_rem: int,
) -> npt.NDArray[np.float64]:
    """Computes deterministic per-period FundingLeg weights (df_fin * (r_fwd + spread) * alpha)."""
    if preproc_state.funding_leg is None or m_rem == 0:
        return np.zeros(m_rem, dtype=np.float64)

    f_leg = preproc_state.funding_leg
    ss = preproc_state.seasoned_state
    future_pay_dates = ss.future_pay_dates[:m_rem]
    if not future_pay_dates:
        return np.zeros(m_rem, dtype=np.float64)

    raw_f_rid = f_leg.get("rate_identifier")
    f_rid = str(raw_f_rid).strip() if raw_f_rid is not None else ""
    f_curve = f_leg.get("funding_curve")
    f_curve_id = f_leg.get("funding_curve_id")
    disc_curve = preproc_state.discount_curve
    disc_id = preproc_state.discount_curve_id
    as_of_dt = ss.as_of_date
    start_dt = ss.start_date
    f_dcc = ss.day_count_convention
    f_spread = float(f_leg.get("funding_spread") or 0.0)
    f_dates = f_leg.get("payment_dates") or ()
    f_fixings = f_leg.get("funding_fixings") or ()

    can_cache_fw = (
        isinstance(disc_curve, (ClampedYieldCurve, type(None)))
        and isinstance(f_curve, (ClampedYieldCurve, type(None)))
    )
    fw_key: Any = None
    max_past_pay_dt: datetime.datetime | None = None
    has_max_past = False
    if can_cache_fw:
        f_fix_key = (
            f_fixings
            if isinstance(f_fixings, tuple)
            else (tuple(f_fixings) if isinstance(f_fixings, list) else ())
        )
        state_dig = preproc_state.state_digest
        if state_dig:
            fw_key = (
                state_dig,
                disc_curve,
                f_curve,
                f_curve_id,
                disc_id,
                f_spread,
                f_rid,
                len(f_dates),
                f_dates[0] if f_dates else None,
                f_dates[-1] if f_dates else None,
                f_fix_key,
                m_rem,
            )
        else:
            max_past_pay_dt = max((r.payment_date for r in ss.historical_records), default=None)
            has_max_past = True
            fw_key = (
                disc_curve,
                f_curve,
                f_curve_id,
                disc_id,
                as_of_dt,
                start_dt,
                f_dcc,
                f_spread,
                f_rid,
                tuple(f_dates),
                f_fix_key,
                tuple(future_pay_dates),
                max_past_pay_dt,
            )
        cached_w = _FUNDING_WEIGHTS_CACHE.get(fw_key, clone_return=False)
        if cached_w is not None:
            return cast("npt.NDArray[np.float64]", cached_w)

    if not has_max_past:
        max_past_pay_dt = max((r.payment_date for r in ss.historical_records), default=None)

    intervals = _map_future_funding_intervals(
        start_dt, as_of_dt, max_past_pay_dt, f_dates, future_pay_dates
    )

    if not intervals:
        out_zero = np.zeros(m_rem, dtype=np.float64)
        if fw_key is not None:
            _FUNDING_WEIGHTS_CACHE.put(fw_key, out_zero, clone_return=False)
        return out_zero

    n_iv = len(intervals)
    same_curve = (
        f_curve is None
        or f_curve is preproc_state.discount_curve
        or (f_curve_id is not None and disc_id is not None and f_curve_id == disc_id)
    )

    dcc_norm = (
        str(getattr(f_dcc, "value", f_dcc))
        .strip()
        .upper()
        .replace("/", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )
    is_additive_dcc = dcc_norm not in ("THIRTY360", "30360", "30360US", "30U360")
    end_dts = [iv[1] for iv in intervals]
    is_contig = n_iv > 1 and all(intervals[j][0] == intervals[j - 1][1] for j in range(1, n_iv))
    if n_iv == 1 or (is_contig and is_additive_dcc):
        alphas = year_fractions(intervals[0][0], end_dts, f_dcc)
        if n_iv > 1:
            alphas[1:] = np.diff(alphas)
    else:
        alphas = np.array([year_fraction(p_dt, e_dt, f_dcc) for p_dt, e_dt, _ in intervals], dtype=np.float64)

    df_fin_arr = (
        preproc_state.df_pay.detach().cpu().numpy().reshape(-1)[: len(future_pay_dates)]
        if end_dts == future_pay_dates
        else _batch_eval_curve_dfs(
            preproc_state.discount_curve,
            build_discount_rate_evaluator(preproc_state.discount_curve, as_of_dt)[1],
            as_of_dt,
            end_dts,
        )
    )

    r_fwds = np.zeros(n_iv, dtype=np.float64)
    if f_rid:
        fwd_curve_obj = preproc_state.discount_curve if same_curve else f_curve
        if same_curve:
            df_fwd_end = df_fin_arr
            df_fwd_fn = build_discount_rate_evaluator(fwd_curve_obj, as_of_dt)[1]
        else:
            _, df_fwd_fn = build_discount_rate_evaluator(fwd_curve_obj, as_of_dt)
            df_fwd_end = _batch_eval_curve_dfs(fwd_curve_obj, df_fwd_fn, as_of_dt, end_dts)

        end_df_by_dt = {e_dt: float(df_fwd_end[j]) for j, e_dt in enumerate(end_dts)}
        prev_needed_idx: list[int] = []
        prev_needed_dts: list[datetime.datetime] = []
        fixed_vals: list[float | None] = [None] * n_iv
        prev_cached_vals: dict[int, float] = {}
        stub_needed = False
        for j, (p_dt, _e_dt, _m) in enumerate(intervals):
            if p_dt <= as_of_dt:
                fv = _lookup_funding_fixing(f_fixings, p_dt)
                fixed_vals[j] = fv
                if fv is None:
                    stub_needed = True
            elif p_dt in end_df_by_dt:
                prev_cached_vals[j] = end_df_by_dt[p_dt]
            else:
                prev_needed_idx.append(j)
                prev_needed_dts.append(p_dt)

        if prev_needed_dts:
            df_fwd_prev_vals = _batch_eval_curve_dfs(
                fwd_curve_obj, df_fwd_fn, as_of_dt, prev_needed_dts
            )
            for k, j in enumerate(prev_needed_idx):
                prev_cached_vals[j] = float(df_fwd_prev_vals[k])

        for j, (p_dt, e_dt, _m) in enumerate(intervals):
            fix_val = fixed_vals[j]
            if fix_val is not None:
                r_fwds[j] = fix_val
            elif p_dt <= as_of_dt:
                alpha_rem = max(year_fraction(as_of_dt, e_dt, f_dcc), 1e-8) if stub_needed else 1.0
                r_fwds[j] = (1.0 / max(float(df_fwd_end[j]), 1e-12) - 1.0) / alpha_rem
            else:
                df_p = prev_cached_vals[j]
                r_fwds[j] = (df_p / max(float(df_fwd_end[j]), 1e-12) - 1.0) / max(float(alphas[j]), 1e-8)

    weights = df_fin_arr * (r_fwds + f_spread) * alphas
    period_undisc_np = np.zeros(m_rem, dtype=np.float64)
    m_indices = np.asarray([iv[2] for iv in intervals], dtype=np.int64)
    if n_iv == m_rem and np.array_equal(m_indices, np.arange(m_rem, dtype=np.int64)):
        period_undisc_np = weights
    else:
        np.add.at(period_undisc_np, m_indices, weights)
    if fw_key is not None:
        _FUNDING_WEIGHTS_CACHE.put(fw_key, period_undisc_np, clone_return=False)
    return period_undisc_np


def _evaluate_future_funding_cf(
    preproc_state: AutocallPreprocState,
    surv_prev: torch.Tensor,
    m_rem: int,
    device: torch.device,
    df_fin_fn_override: Callable[[datetime.datetime], torch.Tensor | float] | None,
    df_fwd_fn_override: Callable[[datetime.datetime], torch.Tensor | float] | None,
) -> torch.Tensor:
    """Computes stateless future period FundingLeg PVs without touching settled or in-transit dates."""
    if preproc_state.funding_leg is None or m_rem == 0:
        return torch.zeros(m_rem, dtype=torch.float64, device=device)

    if df_fin_fn_override is None and df_fwd_fn_override is None:
        w_np = _compute_funding_period_weights_np(preproc_state, m_rem)
        w_t = torch.as_tensor(w_np, dtype=torch.float64, device=device)
        return surv_prev * w_t

    f_leg = preproc_state.funding_leg
    raw_f_rid = f_leg.get("rate_identifier")
    f_rid = str(raw_f_rid).strip() if raw_f_rid is not None else ""
    f_curve = f_leg.get("funding_curve")
    f_curve_id = f_leg.get("funding_curve_id")
    disc_id = preproc_state.discount_curve_id
    as_of_dt = preproc_state.seasoned_state.as_of_date
    start_dt = preproc_state.seasoned_state.start_date
    f_dcc = preproc_state.seasoned_state.day_count_convention
    f_spread = float(f_leg.get("funding_spread") or 0.0)
    f_dates = f_leg.get("payment_dates") or ()
    f_fixings = f_leg.get("funding_fixings") or ()

    future_pay_dates = preproc_state.seasoned_state.future_pay_dates[:m_rem]
    if not future_pay_dates:
        return torch.zeros(m_rem, dtype=torch.float64, device=device)
    hist_recs = preproc_state.seasoned_state.historical_records
    max_past_pay_dt = max((r.payment_date for r in hist_recs), default=None)
    intervals = _map_future_funding_intervals(
        start_dt, as_of_dt, max_past_pay_dt, f_dates, future_pay_dates
    )

    if not intervals:
        return torch.zeros(m_rem, dtype=torch.float64, device=device)

    same_curve = (
        f_curve is None
        or f_curve is preproc_state.discount_curve
        or (f_curve_id is not None and disc_id is not None and f_curve_id == disc_id)
    )

    eval_fin_fn: Callable[[datetime.datetime], torch.Tensor | float] = (
        df_fin_fn_override
        if df_fin_fn_override is not None
        else build_discount_rate_evaluator(preproc_state.discount_curve, as_of_dt)[1]
    )
    eval_fwd_fn: Callable[[datetime.datetime], torch.Tensor | float]
    if df_fwd_fn_override is not None:
        eval_fwd_fn = df_fwd_fn_override
    elif f_rid and not same_curve:
        eval_fwd_fn = build_discount_rate_evaluator(f_curve, as_of_dt)[1]
    else:
        eval_fwd_fn = eval_fin_fn

    period_tensors = [torch.tensor(0.0, dtype=torch.float64, device=device) for _ in range(m_rem)]
    for p_dt, e_dt, m_idx in intervals:
        alpha_j = year_fraction(p_dt, e_dt, f_dcc)
        df_curr = eval_fin_fn(e_dt)
        if not f_rid:
            r_fwd_j: Any = 0.0
        else:
            fixed_reset = _lookup_funding_fixing(f_fixings, p_dt) if p_dt <= as_of_dt else None
            if fixed_reset is not None:
                r_fwd_j = fixed_reset
            elif p_dt <= as_of_dt:
                alpha_rem = max(year_fraction(as_of_dt, e_dt, f_dcc), 1e-8)
                df_curr_fwd = eval_fwd_fn(e_dt)
                if isinstance(df_curr_fwd, torch.Tensor):
                    r_fwd_j = (1.0 / df_curr_fwd.clamp(min=1e-12) - 1.0) / alpha_rem
                else:
                    r_fwd_j = (1.0 / max(float(df_curr_fwd), 1e-12) - 1.0) / alpha_rem
            else:
                df_prev_fwd = eval_fwd_fn(p_dt)
                df_curr_fwd = eval_fwd_fn(e_dt)
                if isinstance(df_curr_fwd, torch.Tensor) or isinstance(df_prev_fwd, torch.Tensor):
                    df_prev_t = (
                        df_prev_fwd
                        if isinstance(df_prev_fwd, torch.Tensor)
                        else torch.tensor(float(df_prev_fwd), dtype=torch.float64, device=device)
                    )
                    df_curr_t = (
                        df_curr_fwd
                        if isinstance(df_curr_fwd, torch.Tensor)
                        else torch.tensor(float(df_curr_fwd), dtype=torch.float64, device=device)
                    )
                    r_fwd_j = (df_prev_t / df_curr_t.clamp(min=1e-12) - 1.0) / max(alpha_j, 1e-8)
                else:
                    r_fwd_j = (float(df_prev_fwd) / max(float(df_curr_fwd), 1e-12) - 1.0) / max(alpha_j, 1e-8)
        cf_j = surv_prev[m_idx] * df_curr * (r_fwd_j + f_spread) * alpha_j
        period_tensors[m_idx] = period_tensors[m_idx] + cf_j
    return torch.stack(period_tensors)


def _append_future_cashflow_items(
    cashflows: list[CashFlowItem],
    *,
    future_pay_dates: Sequence[datetime.datetime],
    ccy: str,
    fund_arr: npt.NDArray[np.float64],
    cpn_arr: npt.NDArray[np.float64],
    notional_arr: npt.NDArray[np.float64],
    df_last: float,
    p_last: float,
    surv_last: float,
    opt_amt: float,
    opt_only_amt: float,
    l_onestar_val: float,
    p_star_val: float,
    option_se: float,
    preproc_state: AutocallPreprocState,
) -> None:
    """Appends future CashFlowItem rows from host float64 arrays without per-period D2H calls."""
    m_eff = min(len(future_pay_dates), len(fund_arr), len(cpn_arr), len(notional_arr))
    if m_eff <= 0:
        return
    pay_slice = future_pay_dates[:m_eff]
    cf_new = object.__new__
    cf_setattr = object.__setattr__
    cf_cls = CashFlowItem

    def _fast_cf(desc: Any, dt_val: datetime.datetime, amt: float, se: float) -> CashFlowItem:
        obj = cf_new(cf_cls)
        cf_setattr(
            obj,
            "__dict__",
            {
                "Description": desc,
                "Date": dt_val,
                "Amount": amt,
                "StandardError": se,
                "Currency": ccy,
                "Leg": 0,
                "StatusText": "Estimate",
            },
        )
        return obj

    add_cf = cashflows.append
    fund_l = fund_arr[:m_eff].tolist()
    cpn_l = cpn_arr[:m_eff].tolist()
    not_l = notional_arr[:m_eff].tolist()
    se_opt = float(option_se)
    if m_eff > 1:
        dt_0 = pay_slice[0]
        cp_f = {"Description": "FundingLeg", "Date": dt_0, "Amount": 0.0, "StandardError": 0.0, "Currency": ccy, "Leg": 0, "StatusText": "Estimate"}.copy
        cp_c = {"Description": "Coupon", "Date": dt_0, "Amount": 0.0, "StandardError": 0.0, "Currency": ccy, "Leg": 0, "StatusText": "Estimate"}.copy
        cp_n = {"Description": "Notional", "Date": dt_0, "Amount": 0.0, "StandardError": 0.0, "Currency": ccy, "Leg": 0, "StatusText": "Estimate"}.copy
        for idx in range(m_eff - 1):
            pay_dt = pay_slice[idx]
            fund_amt = fund_l[idx]
            o1 = cf_new(cf_cls)
            d1 = cp_f()
            d1["Date"] = pay_dt
            d1["Amount"] = -fund_amt if fund_amt != 0.0 else 0.0
            cf_setattr(o1, "__dict__", d1)
            add_cf(o1)

            o2 = cf_new(cf_cls)
            d2 = cp_c()
            d2["Date"] = pay_dt
            d2["Amount"] = cpn_l[idx]
            cf_setattr(o2, "__dict__", d2)
            add_cf(o2)

            o3 = cf_new(cf_cls)
            d3 = cp_n()
            d3["Date"] = pay_dt
            d3["Amount"] = not_l[idx]
            cf_setattr(o3, "__dict__", d3)
            add_cf(o3)

    m_last = m_eff - 1
    pay_dt = pay_slice[m_last]
    fund_amt = fund_l[m_last]
    cpn_amt = cpn_l[m_last]
    notional_amt = not_l[m_last]
    no_notional = preproc_state.no_notional_repayment
    add_cf(_fast_cf("FundingLeg", pay_dt, -fund_amt if fund_amt != 0.0 else 0.0, 0.0))
    add_cf(_fast_cf("Coupon", pay_dt, cpn_amt, 0.0))
    if preproc_state.is_long_basket:
        stop_notional_amt = 0.0 if no_notional else df_last * p_last
        surv_notional_amt = 0.0 if no_notional else df_last * surv_last
        add_cf(_fast_cf("IsLongBasket", pay_dt, surv_notional_amt + opt_amt, se_opt))
        add_cf(_fast_cf("Option", pay_dt, 0.0, 0.0))
        add_cf(_fast_cf("Notional", pay_dt, stop_notional_amt, 0.0))
        return
    os_strike = preproc_state.one_star_strike
    if os_strike is not None and float(os_strike) > 0.0:
        onestar_principal_amt = 0.0 if no_notional else df_last * p_star_val
        onestar_bonus_amt = df_last * l_onestar_val
        onestar_total_amt = onestar_principal_amt + onestar_bonus_amt
        notional_rem_amt = max(notional_amt - onestar_principal_amt, 0.0)
        add_cf(_fast_cf("OneStarCoupon", pay_dt, onestar_total_amt, 0.0))
        add_cf(_fast_cf("Option", pay_dt, opt_only_amt, se_opt))
        add_cf(_fast_cf("Notional", pay_dt, notional_rem_amt, 0.0))
        return
    add_cf(_fast_cf("Option", pay_dt, opt_amt, se_opt))
    add_cf(_fast_cf("Notional", pay_dt, notional_amt, 0.0))


def _to_host_f64_1d(tensor: torch.Tensor, m_rem: int) -> npt.NDArray[np.float64]:
    arr = tensor.detach().cpu().numpy()
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64, copy=False)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr if arr.shape[0] == m_rem else arr[:m_rem]


def assemble_autocall_cashflows_and_pv(
    preproc_state: AutocallPreprocState,
    *,
    p_stop: torch.Tensor | None = None,
    q_cpn: torch.Tensor | None = None,
    l_put: torch.Tensor | None = None,
    l_call: torch.Tensor | None = None,
    l_onestar: torch.Tensor | float | None = None,
    p_onestar: torch.Tensor | float | None = None,
    h_prob: torch.Tensor | None = None,
    q_prob: torch.Tensor | None = None,
    coupon_cf_undisc: torch.Tensor | npt.NDArray[np.float64] | None = None,
    option_se: float = 0.0,
    total_se: float | None = None,
    df_fin_fn_override: Callable[[datetime.datetime], torch.Tensor | float] | None = None,
    df_fwd_fn_override: Callable[[datetime.datetime], torch.Tensor | float] | None = None,
    build_cashflow_items: bool = True,
) -> AutocallPostprocOutput:
    """Executes the W1 affine stopping-time cash-flow and ProductValue post-processor in float64."""
    m_rem = preproc_state.m_rem
    device = preproc_state.spots_t0.device
    ccy = preproc_state.payment_currency
    ss = preproc_state.seasoned_state
    base = preproc_state.baseline
    cash_today = float(preproc_state.cash_to_settle_today)

    if q_cpn is None and q_prob is not None:
        q_cpn = q_prob

    raw_l_put: Any = base.l_put_proxy if l_put is None else l_put
    if l_call is None:
        try:
            raw_l_call_cand: Any = base.l_call_proxy
            raw_l_call: Any = 0.0 if raw_l_call_cand is None else raw_l_call_cand
        except AttributeError:
            raw_l_call = 0.0
    else:
        raw_l_call = l_call
    if l_onestar is None:
        try:
            raw_l_os_cand: Any = base.l_onestar_proxy
            raw_l_onestar: Any = 0.0 if raw_l_os_cand is None else raw_l_os_cand
        except AttributeError:
            raw_l_onestar = 0.0
    else:
        raw_l_onestar = l_onestar
    if p_onestar is not None:
        raw_p_onestar: Any = p_onestar
    else:
        try:
            raw_p_onestar = base.p_onestar_proxy
        except AttributeError:
            raw_p_onestar = None

    tensor_cls = torch.Tensor
    p_src_check = base.p_proxy if p_stop is None else p_stop
    q_src_check = base.q_proxy if q_cpn is None else q_cpn
    is_callable_t = preproc_state.is_callable
    df_pay_t = preproc_state.df_pay
    cpn_amt_t = preproc_state.coupon_amounts
    any_ad_input = (
        df_fin_fn_override is not None
        or df_fwd_fn_override is not None
        or (p_stop is None and isinstance(h_prob, tensor_cls) and h_prob.requires_grad)
        or (isinstance(p_src_check, tensor_cls) and p_src_check.requires_grad)
        or (isinstance(q_src_check, tensor_cls) and q_src_check.requires_grad)
        or (isinstance(raw_l_put, tensor_cls) and raw_l_put.requires_grad)
        or (isinstance(raw_l_call, tensor_cls) and raw_l_call.requires_grad)
        or (isinstance(raw_l_onestar, tensor_cls) and raw_l_onestar.requires_grad)
        or (isinstance(raw_p_onestar, tensor_cls) and raw_p_onestar.requires_grad)
        or (isinstance(coupon_cf_undisc, tensor_cls) and coupon_cf_undisc.requires_grad)
        or df_pay_t.requires_grad
        or cpn_amt_t.requires_grad
        or is_callable_t.requires_grad
    )

    if not any_ad_input and m_rem > 0 and not preproc_state.already_autocalled:
        # Pure host NumPy vectorized fast path without intermediate PyTorch op dispatches.
        is_cpu = device.type == "cpu"
        is_call_np = _to_host_f64_1d(
            is_callable_t if is_callable_t.device.type == "cpu" else is_callable_t.cpu(),
            m_rem,
        )
        if is_cpu:
            if p_stop is None and h_prob is not None:
                h_src = _to_host_f64_1d(h_prob, m_rem)
                h_np = np.clip(h_src, 0.0, 1.0) * is_call_np
                surv_h_np = np.cumprod(1.0 - h_np)
                surv_prev_h_np = np.empty(m_rem, dtype=np.float64)
                surv_prev_h_np[0] = 1.0
                if m_rem > 1:
                    surv_prev_h_np[1:] = surv_h_np[:-1]
                p_np = surv_prev_h_np * h_np
            else:
                p_src = _to_host_f64_1d(p_src_check, m_rem)
                p_np = np.maximum(p_src, 0.0) * is_call_np

            q_np = np.clip(_to_host_f64_1d(q_src_check, m_rem), 0.0, 1.0)
            df_pay_np = _to_host_f64_1d(df_pay_t, m_rem)
            cpn_np = _to_host_f64_1d(cpn_amt_t, m_rem)
            if isinstance(coupon_cf_undisc, tensor_cls):
                undisc_np: npt.NDArray[np.float64] | None = _to_host_f64_1d(coupon_cf_undisc, m_rem)
            elif coupon_cf_undisc is not None:
                undisc_raw = np.asarray(coupon_cf_undisc, dtype=np.float64)
                if undisc_raw.ndim != 1:
                    undisc_raw = undisc_raw.reshape(-1)
                undisc_np = undisc_raw if undisc_raw.shape[0] == m_rem else undisc_raw[:m_rem]
            else:
                undisc_np = None
            l_put_val = float(raw_l_put)
            l_call_val = float(raw_l_call)
            l_os_val = float(raw_l_onestar)
            if raw_p_onestar is not None:
                p_star_raw = float(raw_p_onestar)
            elif preproc_state.one_star_coupon != 0.0:
                p_star_raw = l_os_val / float(preproc_state.one_star_coupon)
            else:
                p_star_raw = 0.0
        else:
            # One batched D2H transfer for GPU inference tensors when AD is inactive.
            p_raw_t = (
                h_prob.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
                if (p_stop is None and h_prob is not None)
                else (
                    base.p_proxy.reshape(-1)[:m_rem]
                    if p_stop is None
                    else p_stop.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
                )
            )
            q_raw_t = (
                base.q_proxy.reshape(-1)[:m_rem]
                if q_cpn is None
                else q_cpn.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
            )
            vecs = [
                p_raw_t,
                q_raw_t,
                df_pay_t.reshape(-1)[:m_rem],
                cpn_amt_t.reshape(-1)[:m_rem],
            ]
            has_undisc_t = isinstance(coupon_cf_undisc, tensor_cls)
            if has_undisc_t:
                vecs.append(
                    cast(torch.Tensor, coupon_cf_undisc).to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
                )
            sc_vals = [
                torch.as_tensor(raw_l_put, dtype=torch.float64, device=device).reshape(()),
                torch.as_tensor(raw_l_call, dtype=torch.float64, device=device).reshape(()),
                torch.as_tensor(raw_l_onestar, dtype=torch.float64, device=device).reshape(()),
                (
                    torch.as_tensor(raw_p_onestar, dtype=torch.float64, device=device).reshape(())
                    if raw_p_onestar is not None
                    else torch.tensor(float("nan"), dtype=torch.float64, device=device)
                ),
            ]
            packed_in = torch.cat([torch.stack(vecs).reshape(-1), torch.stack(sc_vals)], dim=0).detach().cpu().numpy()
            v_block = packed_in[: len(vecs) * m_rem].reshape(len(vecs), m_rem)
            p_src_gpu, q_src_gpu, df_pay_np, cpn_np = v_block[:4]
            if has_undisc_t:
                undisc_np = v_block[4]
            elif coupon_cf_undisc is not None:
                undisc_np = np.asarray(coupon_cf_undisc, dtype=np.float64).reshape(-1)[:m_rem]
            else:
                undisc_np = None
            l_put_val, l_call_val, l_os_val, p_star_maybe = (float(x) for x in packed_in[len(vecs) * m_rem :])
            if not math.isnan(p_star_maybe):
                p_star_raw = p_star_maybe
            elif preproc_state.one_star_coupon != 0.0:
                p_star_raw = l_os_val / float(preproc_state.one_star_coupon)
            else:
                p_star_raw = 0.0
            if p_stop is None and h_prob is not None:
                h_np = np.clip(p_src_gpu, 0.0, 1.0) * is_call_np
                surv_h_np = np.cumprod(1.0 - h_np)
                surv_prev_h_np = np.empty(m_rem, dtype=np.float64)
                surv_prev_h_np[0] = 1.0
                if m_rem > 1:
                    surv_prev_h_np[1:] = surv_h_np[:-1]
                p_np = surv_prev_h_np * h_np
            else:
                p_np = np.maximum(p_src_gpu, 0.0) * is_call_np
            q_np = np.clip(q_src_gpu, 0.0, 1.0)

        surv_np = 1.0 - np.cumsum(p_np)
        np.clip(surv_np, 0.0, None, out=surv_np)
        surv_prev_np = np.empty(m_rem, dtype=np.float64)
        surv_prev_np[0] = 1.0
        if m_rem > 1:
            surv_prev_np[1:] = surv_np[:-1]
        surv_last_f = float(surv_np[-1])
        p_star_val = 0.0 if p_star_raw <= 0.0 else (p_star_raw if p_star_raw <= surv_last_f else surv_last_f)

        period_w_np = _compute_funding_period_weights_np(preproc_state, m_rem)
        fund_np = surv_prev_np * period_w_np

        if preproc_state.no_notional_repayment:
            notional_cf_np = np.zeros(m_rem, dtype=np.float64)
        else:
            stop_plus_surv = p_np.copy()
            stop_plus_surv[-1] += surv_last_f
            notional_cf_np = df_pay_np * stop_plus_surv

        ac_coupon = float(preproc_state.autocall_coupon)
        if undisc_np is not None:
            coupon_cf_np = df_pay_np * undisc_np
        else:
            if preproc_state.force_coupons_on_autocall:
                eff_q_np = np.clip(
                    np.maximum(q_np, p_np / np.maximum(surv_prev_np, 1e-12)),
                    0.0,
                    1.0,
                )
            else:
                eff_q_np = q_np
            if preproc_state.is_memory:
                cpn_out = [0.0] * m_rem
                mem_f = float(preproc_state.c_mem_0)
                df_l = df_pay_np.tolist()
                sp_l = surv_prev_np.tolist()
                eq_l = eff_q_np.tolist()
                cpn_l = cpn_np.tolist()
                p_l = p_np.tolist()
                for m in range(m_rem):
                    eff_c_m = cpn_l[m] + mem_f
                    eq_m = eq_l[m]
                    cpn_out[m] = df_l[m] * (sp_l[m] * eq_m * eff_c_m + p_l[m] * ac_coupon)
                    mem_f = eff_c_m * (1.0 - eq_m)
                coupon_cf_np = np.array(cpn_out, dtype=np.float64)
            else:
                coupon_cf_np = df_pay_np * (
                    surv_prev_np * eff_q_np * cpn_np + p_np * ac_coupon
                )

        df_last = float(df_pay_np[-1])
        gearing = float(preproc_state.option_gearing)
        part_rate = float(preproc_state.participation_call_rate)
        opt_only_amt = df_last * (-gearing * l_put_val + part_rate * l_call_val)
        opt_amt = opt_only_amt + df_last * l_os_val

        notional_pv_s = float(notional_cf_np.sum())
        coupon_pv_s = float(coupon_cf_np.sum())
        funding_pv_s = float(fund_np.sum())
        pv_it_s = float(preproc_state.pv_in_transit)
        pv_total_s = pv_it_s + notional_pv_s + coupon_pv_s + opt_amt - funding_pv_s

        cashflows_fast: list[CashFlowItem] = list(ss.past_cashflows) if build_cashflow_items else []
        if build_cashflow_items:
            _append_future_cashflow_items(
                cashflows_fast,
                future_pay_dates=ss.future_pay_dates[:m_rem],
                ccy=ccy,
                fund_arr=fund_np,
                cpn_arr=coupon_cf_np,
                notional_arr=notional_cf_np,
                df_last=df_last,
                p_last=float(p_np[-1]),
                surv_last=surv_last_f,
                opt_amt=opt_amt,
                opt_only_amt=opt_only_amt,
                l_onestar_val=l_os_val,
                p_star_val=p_star_val,
                option_se=option_se,
                preproc_state=preproc_state,
            )

        out_sc = torch.tensor(
            [pv_total_s, notional_pv_s, coupon_pv_s, opt_amt, funding_pv_s, pv_it_s, l_put_val, l_call_val, l_os_val, p_star_val],
            dtype=torch.float64,
            device=device,
        )
        pv_t, not_pv_t, cpn_pv_t, opt_pv_t, fund_pv_t, it_pv_t, lp_t, lc_t, los_t, pos_t = out_sc.unbind(0)
        eff_total_se = float(total_se) if total_se is not None else float(option_se)
        return AutocallPostprocOutput(
            product_value=pv_t,
            notional_pv=not_pv_t,
            coupon_pv=cpn_pv_t,
            option_pv=opt_pv_t,
            funding_pv=fund_pv_t,
            pv_in_transit=it_pv_t,
            survival_probs=torch.from_numpy(surv_np) if is_cpu else torch.as_tensor(surv_np, dtype=torch.float64, device=device),
            p_stop=torch.from_numpy(p_np) if is_cpu else torch.as_tensor(p_np, dtype=torch.float64, device=device),
            q_cpn=torch.from_numpy(q_np) if is_cpu else torch.as_tensor(q_np, dtype=torch.float64, device=device),
            l_put=lp_t,
            l_call=lc_t,
            cashflows=cashflows_fast,
            standard_error=eff_total_se,
            reporting_currency=ccy,
            cash_to_settle_today=cash_today,
            l_onestar=los_t,
            p_onestar=pos_t,
        )

    is_call_t = is_callable_t.reshape(-1)[:m_rem]
    if p_stop is None and h_prob is not None:
        h_t = (
            h_prob.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem].clamp(0.0, 1.0)
            * is_call_t
        )
        if m_rem > 0:
            one_minus_h = 1.0 - h_t
            surv_h = torch.cumprod(one_minus_h, dim=0)
            surv_prev_h = torch.cat(
                [torch.ones(1, dtype=torch.float64, device=device), surv_h[:-1]],
                dim=0,
            )
            p_stop = surv_prev_h * h_t
        else:
            p_stop = h_t
    if q_cpn is None and q_prob is not None:
        q_cpn = q_prob

    p_t = (
        (
            base.p_proxy.reshape(-1)[:m_rem]
            if p_stop is None
            else p_stop.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
        ).clamp(min=0.0)
        * is_call_t
    )
    q_t = (
        base.q_proxy.reshape(-1)[:m_rem]
        if q_cpn is None
        else q_cpn.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
    ).clamp(0.0, 1.0)
    l_put_t = (
        base.l_put_proxy
        if l_put is None
        else l_put.to(dtype=torch.float64, device=device).reshape(())
    )
    if l_call is None:
        raw_lc_ad = getattr(base, "l_call_proxy", None)
        l_call_t = (
            raw_lc_ad
            if raw_lc_ad is not None
            else torch.tensor(0.0, dtype=torch.float64, device=device)
        )
    else:
        l_call_t = l_call.to(dtype=torch.float64, device=device).reshape(())
    if l_onestar is None:
        raw_los_ad = getattr(base, "l_onestar_proxy", None)
        l_onestar_t = (
            raw_los_ad
            if raw_los_ad is not None
            else torch.tensor(0.0, dtype=torch.float64, device=device)
        )
    else:
        l_onestar_t = torch.as_tensor(l_onestar, dtype=torch.float64, device=device).reshape(())

    if p_onestar is not None:
        p_onestar_raw = torch.as_tensor(p_onestar, dtype=torch.float64, device=device).reshape(())
    else:
        raw_pos_ad = getattr(base, "p_onestar_proxy", None)
        if raw_pos_ad is not None:
            p_onestar_raw = torch.as_tensor(raw_pos_ad, dtype=torch.float64, device=device).reshape(())
        elif float(preproc_state.one_star_coupon) != 0.0:
            p_onestar_raw = l_onestar_t / float(preproc_state.one_star_coupon)
        else:
            p_onestar_raw = torch.tensor(0.0, dtype=torch.float64, device=device)

    cashflows: list[CashFlowItem] = list(ss.past_cashflows) if build_cashflow_items else []
    pv_in_transit_t = torch.tensor(preproc_state.pv_in_transit, dtype=torch.float64, device=device)

    if m_rem == 0 or preproc_state.already_autocalled:
        zero_link = (
            0.0 * p_t.sum()
            + 0.0 * q_t.sum()
            + 0.0 * l_put_t
            + 0.0 * l_call_t
            + 0.0 * l_onestar_t
            + 0.0 * p_onestar_raw
        )
        need_grad_empty = bool(zero_link.requires_grad or pv_in_transit_t.requires_grad)
        z_s = zero_link if need_grad_empty else torch.tensor(0.0, dtype=torch.float64, device=device)
        pv_out_empty = pv_in_transit_t + zero_link if need_grad_empty else pv_in_transit_t
        p_zero = 0.0 * p_t if need_grad_empty else torch.zeros(m_rem, dtype=torch.float64, device=device)
        q_zero = 0.0 * q_t if need_grad_empty else torch.zeros(m_rem, dtype=torch.float64, device=device)
        l_put_zero = 0.0 * l_put_t if need_grad_empty else torch.tensor(0.0, dtype=torch.float64, device=device)
        l_call_zero = 0.0 * l_call_t if need_grad_empty else torch.tensor(0.0, dtype=torch.float64, device=device)
        l_os_zero = 0.0 * l_onestar_t if need_grad_empty else torch.tensor(0.0, dtype=torch.float64, device=device)
        if build_cashflow_items and m_rem > 0:
            zeros_m = np.zeros(m_rem, dtype=np.float64)
            _append_future_cashflow_items(
                cashflows,
                future_pay_dates=preproc_state.seasoned_state.future_pay_dates[:m_rem],
                ccy=ccy,
                fund_arr=zeros_m,
                cpn_arr=zeros_m,
                notional_arr=zeros_m,
                df_last=0.0,
                p_last=0.0,
                surv_last=0.0,
                opt_amt=0.0,
                opt_only_amt=0.0,
                l_onestar_val=0.0,
                p_star_val=0.0,
                option_se=0.0,
                preproc_state=preproc_state,
            )
        return AutocallPostprocOutput(
            product_value=pv_out_empty,
            notional_pv=z_s,
            coupon_pv=z_s,
            option_pv=z_s,
            funding_pv=z_s,
            pv_in_transit=pv_in_transit_t,
            survival_probs=torch.zeros(m_rem, dtype=torch.float64, device=device),
            p_stop=p_zero,
            q_cpn=q_zero,
            l_put=l_put_zero,
            l_call=l_call_zero,
            cashflows=cashflows,
            standard_error=0.0,
            reporting_currency=ccy,
            cash_to_settle_today=cash_today,
            l_onestar=l_os_zero,
            p_onestar=z_s,
        )

    cum_p = torch.cumsum(p_t, dim=0)
    survival_probs = (1.0 - cum_p).clamp(min=0.0)
    surv_prev = torch.cat(
        [torch.ones(1, dtype=torch.float64, device=device), survival_probs[:-1]],
        dim=0,
    )
    p_onestar_t = torch.minimum(p_onestar_raw.clamp(min=0.0), survival_probs[-1])

    df_pay = preproc_state.df_pay.reshape(-1)[:m_rem]
    cpn_rates = preproc_state.coupon_amounts.reshape(-1)[:m_rem]
    funding_cf_t = _evaluate_future_funding_cf(
        preproc_state, surv_prev, m_rem, device, df_fin_fn_override, df_fwd_fn_override
    )

    coupon_cf_undisc_t = (
        torch.as_tensor(coupon_cf_undisc, dtype=torch.float64, device=device).reshape(-1)[:m_rem]
        if coupon_cf_undisc is not None
        else None
    )

    put_sign = -1.0
    gearing = float(preproc_state.option_gearing)
    part_rate = float(preproc_state.participation_call_rate)
    ac_coupon = float(preproc_state.autocall_coupon)

    need_grad = bool(
        p_t.requires_grad
        or q_t.requires_grad
        or l_put_t.requires_grad
        or l_call_t.requires_grad
        or l_onestar_t.requires_grad
        or df_pay.requires_grad
        or cpn_rates.requires_grad
        or (coupon_cf_undisc_t is not None and coupon_cf_undisc_t.requires_grad)
        or funding_cf_t.requires_grad
        or p_onestar_t.requires_grad
        or pv_in_transit_t.requires_grad
    )

    if not need_grad:
        if device.type == "cpu":
            p_np = p_t.detach().numpy()
            q_np = q_t.detach().numpy()
            surv_np = survival_probs.detach().numpy()
            surv_prev_np = surv_prev.detach().numpy()
            df_pay_np = df_pay.detach().numpy()
            cpn_np = cpn_rates.detach().numpy()
            fund_np = funding_cf_t.detach().numpy()
            undisc_np = coupon_cf_undisc_t.detach().numpy() if coupon_cf_undisc_t is not None else None
            l_put_val = float(l_put_t)
            l_call_val = float(l_call_t)
            l_os_val = float(l_onestar_t)
            p_star_val = float(p_onestar_t)
        else:
            # Single batched D2H transfer for fast host NumPy cashflow assembly when AD is inactive.
            scalars_t = torch.stack([l_put_t, l_call_t, l_onestar_t, p_onestar_t])
            vec_list = [p_t, q_t, survival_probs, surv_prev, df_pay, cpn_rates, funding_cf_t]
            if coupon_cf_undisc_t is not None:
                vec_list.append(coupon_cf_undisc_t)
            packed = torch.cat([torch.stack(vec_list).reshape(-1), scalars_t], dim=0).detach().cpu().numpy()
            n_vecs = len(vec_list)
            vecs_np = packed[: n_vecs * m_rem].reshape(n_vecs, m_rem)
            p_np, q_np, surv_np, surv_prev_np, df_pay_np, cpn_np, fund_np = vecs_np[:7]
            undisc_np = vecs_np[7] if coupon_cf_undisc_t is not None else None
            l_put_val, l_call_val, l_os_val, p_star_val = (float(x) for x in packed[n_vecs * m_rem :])

        if preproc_state.no_notional_repayment:
            notional_cf_np = np.zeros(m_rem, dtype=np.float64)
        else:
            stop_plus_surv = p_np.copy()
            stop_plus_surv[-1] += surv_np[-1]
            notional_cf_np = df_pay_np * stop_plus_surv

        if undisc_np is not None:
            coupon_cf_np = df_pay_np * undisc_np
        else:
            if preproc_state.force_coupons_on_autocall:
                eff_q_np = np.clip(
                    np.maximum(q_np, p_np / np.maximum(surv_prev_np, 1e-12)),
                    0.0,
                    1.0,
                )
            else:
                eff_q_np = q_np
            if preproc_state.is_memory:
                coupon_cf_np = np.empty(m_rem, dtype=np.float64)
                mem_f = float(preproc_state.c_mem_0)
                for m in range(m_rem):
                    eff_c_m = float(cpn_np[m]) + mem_f
                    eq_m = float(eff_q_np[m])
                    coupon_cf_np[m] = float(df_pay_np[m]) * (
                        float(surv_prev_np[m]) * eq_m * eff_c_m + float(p_np[m]) * ac_coupon
                    )
                    mem_f = eff_c_m * (1.0 - eq_m)
            else:
                coupon_cf_np = df_pay_np * (
                    surv_prev_np * eff_q_np * cpn_np + p_np * ac_coupon
                )

        df_last = float(df_pay_np[-1])
        opt_only_amt = df_last * (put_sign * gearing * l_put_val + part_rate * l_call_val)
        opt_amt = opt_only_amt + df_last * l_os_val

        notional_pv_s = float(notional_cf_np.sum())
        coupon_pv_s = float(coupon_cf_np.sum())
        funding_pv_s = float(fund_np.sum())
        pv_total_s = float(preproc_state.pv_in_transit) + notional_pv_s + coupon_pv_s + opt_amt - funding_pv_s

        out_scalars = torch.tensor(
            [notional_pv_s, coupon_pv_s, opt_amt, funding_pv_s, pv_total_s],
            dtype=torch.float64,
            device=device,
        )
        notional_pv, coupon_pv, option_pv, funding_pv, product_value = (
            out_scalars[0],
            out_scalars[1],
            out_scalars[2],
            out_scalars[3],
            out_scalars[4],
        )

        if build_cashflow_items:
            _append_future_cashflow_items(
                cashflows,
                future_pay_dates=preproc_state.seasoned_state.future_pay_dates[:m_rem],
                ccy=ccy,
                fund_arr=fund_np,
                cpn_arr=coupon_cf_np,
                notional_arr=notional_cf_np,
                df_last=df_last,
                p_last=float(p_np[-1]),
                surv_last=float(surv_np[-1]),
                opt_amt=opt_amt,
                opt_only_amt=opt_only_amt,
                l_onestar_val=l_os_val,
                p_star_val=p_star_val,
                option_se=option_se,
                preproc_state=preproc_state,
            )
    else:
        if preproc_state.no_notional_repayment:
            notional_cf_t = torch.zeros(m_rem, dtype=torch.float64, device=device)
        else:
            last_extra = torch.zeros(m_rem, dtype=torch.float64, device=device)
            last_extra[-1] = survival_probs[-1]
            notional_cf_t = df_pay * (p_t + last_extra)

        if coupon_cf_undisc_t is not None:
            coupon_cf_t = df_pay * coupon_cf_undisc_t
        else:
            if preproc_state.force_coupons_on_autocall:
                eff_q_all = torch.maximum(q_t, p_t / surv_prev.clamp(min=1e-12)).clamp(0.0, 1.0)
            else:
                eff_q_all = q_t
            if preproc_state.is_memory:
                coupon_cf_list: list[torch.Tensor] = []
                mem_acc = torch.tensor(preproc_state.c_mem_0, dtype=torch.float64, device=device)
                for m in range(m_rem):
                    eff_c_m_t = cpn_rates[m] + mem_acc
                    eff_q_m_t = eff_q_all[m]
                    coupon_cf_list.append(
                        df_pay[m] * (surv_prev[m] * eff_q_m_t * eff_c_m_t + p_t[m] * ac_coupon)
                    )
                    mem_acc = eff_c_m_t * (1.0 - eff_q_m_t)
                coupon_cf_t = torch.stack(coupon_cf_list)
            else:
                coupon_cf_t = df_pay * (surv_prev * eff_q_all * cpn_rates + p_t * ac_coupon)

        opt_only_term_t = df_pay[-1] * (put_sign * gearing * l_put_t + part_rate * l_call_t)
        cf_option_term = opt_only_term_t + df_pay[-1] * l_onestar_t

        notional_pv = notional_cf_t.sum()
        coupon_pv = coupon_cf_t.sum()
        funding_pv = funding_cf_t.sum()
        option_pv = cf_option_term
        product_value = pv_in_transit_t + notional_pv + coupon_pv + option_pv - funding_pv

        if build_cashflow_items:
            scalars_ad = torch.stack(
                [
                    df_pay[-1],
                    p_t[-1],
                    survival_probs[-1],
                    cf_option_term,
                    opt_only_term_t,
                    l_onestar_t,
                    p_onestar_t,
                ]
            )
            packed_ad = (
                torch.cat(
                    [torch.stack([funding_cf_t, coupon_cf_t, notional_cf_t]).reshape(-1), scalars_ad],
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            vecs_ad = packed_ad[: 3 * m_rem].reshape(3, m_rem)
            df_last, p_last, surv_last, opt_amt, opt_only_amt, l_os_val, p_star_val = (
                float(x) for x in packed_ad[3 * m_rem :]
            )
            _append_future_cashflow_items(
                cashflows,
                future_pay_dates=preproc_state.seasoned_state.future_pay_dates[:m_rem],
                ccy=ccy,
                fund_arr=vecs_ad[0],
                cpn_arr=vecs_ad[1],
                notional_arr=vecs_ad[2],
                df_last=df_last,
                p_last=p_last,
                surv_last=surv_last,
                opt_amt=opt_amt,
                opt_only_amt=opt_only_amt,
                l_onestar_val=l_os_val,
                p_star_val=p_star_val,
                option_se=option_se,
                preproc_state=preproc_state,
            )

    eff_total_se = float(total_se) if total_se is not None else float(option_se)
    return AutocallPostprocOutput(
        product_value=product_value,
        notional_pv=notional_pv,
        coupon_pv=coupon_pv,
        option_pv=option_pv,
        funding_pv=funding_pv,
        pv_in_transit=pv_in_transit_t,
        survival_probs=survival_probs,
        p_stop=p_t,
        q_cpn=q_t,
        l_put=l_put_t,
        l_call=l_call_t,
        cashflows=cashflows,
        standard_error=eff_total_se,
        reporting_currency=ccy,
        cash_to_settle_today=cash_today,
        l_onestar=l_onestar_t,
        p_onestar=p_onestar_t,
    )


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "D_MAX_DEFAULT",
    "M_MAX_DEFAULT",
    "MonitoringIndexCode",
    "monitoring_index_to_code",
    "PAIR_ORDER_DEPENDENT_INDICES",
    "AutocallNodeCol",
    "AutocallEdgeCol",
    "AutocallSchedCol",
    "AutocallGlobalCol",
    "ContentDigestKey",
    "ThreadSafeLRUCache",
    "build_vol_surface_cache_key",
    "digest_market_bytes",
    "compute_surface_digest",
    "compute_preproc_state_digest",
    "AutocallPreprocState",
    "AutocallPostprocOutput",
    "preprocess_eq_autocall",
    "assemble_autocall_cashflows_and_pv",
]
