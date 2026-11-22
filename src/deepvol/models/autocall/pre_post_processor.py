"""Tier 1 deterministic float64 pre- and post-processor for EQAutocallable."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, fields as dc_fields, replace as dc_replace
import datetime
from enum import Enum
import hashlib
import inspect
from pathlib import Path
import struct
import threading
from typing import Any, Callable, Mapping, Sequence, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch

from deepvol.api.schemas_eq_autocall import parse_murex_autocall_dag
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
    _infer_asset_currency,
    build_discount_rate_evaluator,
    compute_escrowed_dividend_forwards,
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

    def __init__(self, builder: Callable[[str], npt.NDArray[np.float64]], keys: Sequence[str]) -> None:
        super().__init__()
        self._builder = builder
        self._keys = list(keys)

    def _populate_all(self) -> None:
        for k in self._keys:
            if not super().__contains__(k):
                super().__setitem__(k, self._builder(k))

    def __getitem__(self, key: str) -> npt.NDArray[np.float64]:
        if not super().__contains__(key) and key in self._keys:
            super().__setitem__(key, self._builder(key))
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._keys:
            return self[key]
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._keys or super().__contains__(key)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Any:
        self._populate_all()
        return super().__iter__()

    def keys(self) -> Any:
        self._populate_all()
        return super().keys()

    def values(self) -> Any:
        self._populate_all()
        return super().values()

    def items(self) -> Any:
        self._populate_all()
        return super().items()

    def clone(self) -> _LazySurfaceJacobianDict:
        cloned = _LazySurfaceJacobianDict(self._builder, self._keys)
        for k, v in dict.items(self):
            dict.__setitem__(cloned, k, v.copy())
        return cloned


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

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            val = self._data[key]
            return self._copy_fn(val) if self._copy_fn is not None else val

    def put(self, key: Any, value: Any) -> Any:
        with self._lock:
            stored = self._copy_fn(value) if self._copy_fn is not None else value
            self._data[key] = stored
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)
            return self._copy_fn(stored) if self._copy_fn is not None else stored

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def __getitem__(self, key: Any) -> Any:
        res = self.get(key)
        if res is None and key not in self._data:
            raise KeyError(key)
        return res

    def __setitem__(self, key: Any, value: Any) -> None:
        self.put(key, value)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def _feed_canonical_bytes(hasher: Any, obj: Any, counter: list[int]) -> bool:
    """Feeds deterministic canonical byte representation into a SHA-256 hasher.

    Returns False if `obj` contains an active autograd tensor or unhashable transient
    that must bypass caching.
    """
    if obj is None:
        hasher.update(b"N;")
        counter[0] += 2
        return True
    if isinstance(obj, bool):
        hasher.update(b"B1;" if obj else b"B0;")
        counter[0] += 3
        return True
    if isinstance(obj, int):
        b = str(obj).encode("ascii")
        hasher.update(b"I" + struct.pack("!I", len(b)) + b)
        counter[0] += 5 + len(b)
        return True
    if isinstance(obj, float):
        # Canonicalize -0.0 to +0.0 while preserving NaN bit patterns explicitly.
        val = 0.0 if obj == 0.0 else obj
        packed = struct.pack("!d", val)
        hasher.update(b"F" + packed)
        counter[0] += 9
        return True
    if isinstance(obj, Enum):
        return _feed_canonical_bytes(hasher, obj.value, counter)
    if isinstance(obj, str):
        b = obj.encode("utf-8")
        hasher.update(b"S" + struct.pack("!I", len(b)) + b)
        counter[0] += 5 + len(b)
        return True
    if isinstance(obj, (bytes, bytearray)):
        hasher.update(b"Y" + struct.pack("!I", len(obj)) + bytes(obj))
        counter[0] += 5 + len(obj)
        return True
    if isinstance(obj, datetime.datetime):
        dt_utc = _to_utc_datetime(obj)
        b = dt_utc.isoformat().encode("ascii")
        hasher.update(b"DT" + struct.pack("!I", len(b)) + b)
        counter[0] += 6 + len(b)
        return True
    if isinstance(obj, datetime.date):
        b = obj.isoformat().encode("ascii")
        hasher.update(b"DA" + struct.pack("!I", len(b)) + b)
        counter[0] += 6 + len(b)
        return True
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        shape_b = repr(arr.shape).encode("ascii")
        dtype_b = arr.dtype.str.encode("ascii")
        raw = arr.tobytes()
        hasher.update(
            b"NP"
            + struct.pack("!II", len(shape_b), len(dtype_b))
            + shape_b
            + dtype_b
            + struct.pack("!Q", len(raw))
            + raw
        )
        counter[0] += 18 + len(shape_b) + len(dtype_b) + len(raw)
        return True
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad:
            return False
        arr = obj.detach().cpu().contiguous().numpy()
        return _feed_canonical_bytes(hasher, arr, counter)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        dumped = obj.model_dump(mode="python")
        hasher.update(b"PM")
        counter[0] += 2
        return _feed_canonical_bytes(hasher, dumped, counter)
    if isinstance(obj, Mapping):
        hasher.update(b"M" + struct.pack("!I", len(obj)))
        counter[0] += 5
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            if not _feed_canonical_bytes(hasher, str(k), counter):
                return False
            if not _feed_canonical_bytes(hasher, obj[k], counter):
                return False
        return True
    if isinstance(obj, (list, tuple)):
        hasher.update(b"L" + struct.pack("!I", len(obj)))
        counter[0] += 5
        for item in obj:
            if not _feed_canonical_bytes(hasher, item, counter):
                return False
        return True
    if isinstance(obj, (set, frozenset)):
        items = sorted(obj, key=lambda x: str(x))
        hasher.update(b"T" + struct.pack("!I", len(items)))
        counter[0] += 5
        for item in items:
            if not _feed_canonical_bytes(hasher, item, counter):
                return False
        return True
    if isinstance(obj, (ClampedYieldCurve, RepoCurve, RegularizedVolSurface)):
        return _feed_canonical_bytes(hasher, obj.__dict__, counter)
    if hasattr(obj, "__dict__"):
        hasher.update(b"O")
        counter[0] += 1
        return _feed_canonical_bytes(hasher, vars(obj), counter)
    return False


def digest_market_bytes(namespace: str, payload: Any) -> ContentDigestKey | None:
    """Builds a SHA-256 ContentDigestKey covering 100% of interior values."""
    h = hashlib.sha256()
    counter = [0]
    if not _feed_canonical_bytes(h, payload, counter):
        return None
    return ContentDigestKey(namespace=namespace, digest_hex=h.hexdigest(), payload_len=counter[0])


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
    )


_MARKET_OBJECT_CACHE = ThreadSafeLRUCache(maxsize=512, copy_fn=None)
_PREPROC_STATE_CACHE = ThreadSafeLRUCache(maxsize=512, copy_fn=_clone_preproc_state)


def _extract_attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Extracts the first matching key or attribute from a dict or Pydantic model."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
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
        cached = _MARKET_OBJECT_CACHE.get(key)
        if cached is not None:
            return cast(ClampedYieldCurve, cached)
    curve = ClampedYieldCurve(spec, as_of_date=as_of_dt)
    if key is not None:
        return cast(ClampedYieldCurve, _MARKET_OBJECT_CACHE.put(key, curve))
    return curve


_VERSION_STATUS_TAGS: frozenset[str] = frozenset(
    {
        "OLD",
        "NEW",
        "REG",
        "PREF",
        "P",
        "DR",
        "ADR",
        "GDR",
        "CLS",
        "LOCAL",
        "SHS",
        "ORD",
        "A",
        "B",
        "C",
        "1",
        "2",
    }
)
_KNOWN_ISO_CURRENCIES: frozenset[str] = frozenset(
    {
        "RUB",
        "USD",
        "CNH",
        "CNY",
        "EUR",
        "GBP",
        "HKD",
        "CHF",
        "JPY",
        "AED",
        "KZT",
        "BYN",
        "TRY",
        "INR",
        "BRL",
        "ZAR",
        "CAD",
        "AUD",
        "SGD",
        "NZD",
        "SEK",
        "NOK",
        "DKK",
        "PLN",
        "MXN",
        "KRW",
        "TWD",
    }
)
_COUNTRY_SUFFIX_TO_CCY: dict[str, str] = {
    "US": "USD",
    "HK": "HKD",
    "CN": "CNH",
    "EU": "EUR",
    "LN": "GBP",
    "GR": "EUR",
    "FP": "EUR",
    "GY": "EUR",
    "SW": "CHF",
    "JP": "JPY",
    "KZ": "KZT",
}
_OIS_CURVE_TOKENS: tuple[str, ...] = (
    "OIS",
    "SOFR",
    "RUONIA",
    "ESTR",
    "SONIA",
    "TONAR",
    "SARON",
    "SHIBOR",
)
_FUNDING_CURVE_TOKENS: tuple[str, ...] = ("FUNDING", "FUND", "SECD")


def _curve_ois_priority_rank(cid: str, fin_ids: Sequence[str] = ()) -> tuple[int, str]:
    """Ranks candidate same-currency curves for equity forward growth (F-12).

    Rank 0: OIS curves ('OIS', 'SOFR', 'RUONIA', 'ESTR', 'SONIA') without 'XCCY', 'FUNDING', 'SECD'.
    Rank 1: Standard yield / FX curves not in fin_ids and without 'XCCY', 'FUNDING', 'SECD'.
    Rank 2: 'XCCY' cross-currency basis curves.
    Rank 3: Curves in fin_ids or containing 'FUNDING' / 'SECD'.
    """
    cid_u = str(cid).strip().upper()
    fin_set = {str(f).strip().upper() for f in fin_ids}
    has_xccy = "XCCY" in cid_u
    has_funding = any(tok in cid_u for tok in _FUNDING_CURVE_TOKENS)
    has_ois = any(tok in cid_u for tok in _OIS_CURVE_TOKENS)
    if has_ois and not has_xccy and not has_funding:
        return (0, cid_u)
    if not has_xccy and not has_funding and cid_u not in fin_set:
        return (1, cid_u)
    if has_xccy:
        return (2, cid_u)
    return (3, cid_u)


def _infer_asset_currency_5tier(
    ticker: str,
    payment_ccy: str,
    *,
    div_src: Any = None,
    vol_surfaces_map: Mapping[str, Any] | None = None,
    correlations_map: Mapping[str, float] | None = None,
    rates_bundles: Mapping[str, Any] | None = None,
) -> str:
    """5-tier deterministic asset currency inference (I-5a)."""
    pay_ccy_upper = str(payment_ccy).strip().upper()

    # Tier 1: Explicit Currency metadata on div_src or vol_surfaces_map[ticker]
    for src in (
        div_src,
        vol_surfaces_map.get(ticker) if isinstance(vol_surfaces_map, Mapping) else None,
    ):
        raw_ccy = _extract_attr(src, "Currency", "currency") if src is not None else None
        if isinstance(raw_ccy, str):
            c_up = raw_ccy.strip().upper()
            if len(c_up) == 3 and c_up.isalpha() and c_up not in _VERSION_STATUS_TAGS:
                return c_up

    # Tier 2: Tokenized '_' scan after stripping version/status tags
    tokens = [tok.strip().upper() for tok in str(ticker).split("_") if tok.strip()]
    while len(tokens) > 1 and tokens[-1] in _VERSION_STATUS_TAGS:
        tokens.pop()
    if len(tokens) > 2 and tokens[-1] == "RX" and len(tokens[-2]) == 3 and tokens[-2].isalpha():
        tokens.pop()
    for tok in reversed(tokens[1:]):
        if tok in _KNOWN_ISO_CURRENCIES:
            return tok
    if len(tokens) >= 2 and len(tokens[-1]) == 3 and tokens[-1].isalpha() and tokens[-1] not in _VERSION_STATUS_TAGS:
        return tokens[-1]

    # Tier 3: Correlations & FXVolatilitySmileCollection pair graph ('AAPLUS&USDRUB', 'MSFT&USDRUB' -> 'USD')
    if isinstance(correlations_map, Mapping):
        fx_candidates: list[str] = []
        if isinstance(vol_surfaces_map, Mapping):
            for sid, s_obj in vol_surfaces_map.items():
                t_name = _extract_attr(s_obj, "TypeName", "type_name")
                sid_clean = str(sid).strip().upper().replace("_", "").replace("/", "").replace("-", "")
                if t_name == "FXVolatilitySmileCollection" or (
                    len(sid_clean) == 6
                    and sid_clean[:3] in _KNOWN_ISO_CURRENCIES
                    and sid_clean[3:] in _KNOWN_ISO_CURRENCIES
                ):
                    fx_candidates.append(str(sid))
        # Also scan correlation keys directly for 6-letter FX pairs paired with ticker
        for corr_key in correlations_map:
            if "&" in str(corr_key):
                lhs, rhs = str(corr_key).split("&", 1)
                other = rhs if lhs == ticker else (lhs if rhs == ticker else None)
                if other is not None and other not in fx_candidates:
                    other_clean = other.strip().upper().replace("_", "").replace("/", "").replace("-", "")
                    if (
                        len(other_clean) == 6
                        and other_clean[:3] in _KNOWN_ISO_CURRENCIES
                        and other_clean[3:] in _KNOWN_ISO_CURRENCIES
                    ):
                        fx_candidates.append(other)

        for fx_id in fx_candidates:
            if f"{ticker}&{fx_id}" in correlations_map or f"{fx_id}&{ticker}" in correlations_map:
                fx_clean = fx_id.strip().upper().replace("_", "").replace("/", "").replace("-", "")
                if len(fx_clean) == 6 and fx_clean[:3].isalpha() and fx_clean[3:].isalpha():
                    c1, c2 = fx_clean[:3], fx_clean[3:]
                    if c1 == pay_ccy_upper and c2 != pay_ccy_upper:
                        return c2
                    if c2 == pay_ccy_upper and c1 != pay_ccy_upper:
                        return c1

    # Tier 4: 2-letter country suffix ('AAPLUS', 'GOOGLUS' -> 'USD' when foreign curve/FX exists)
    base_tok = tokens[0] if tokens else str(ticker).strip().upper()
    if len(base_tok) >= 5 and base_tok[-2:] in _COUNTRY_SUFFIX_TO_CCY:
        cand_ccy = _COUNTRY_SUFFIX_TO_CCY[base_tok[-2:]]
        has_ccy_curve = isinstance(rates_bundles, Mapping) and any(
            str(cid).strip().upper().startswith(cand_ccy)
            or str(_extract_attr(spec_c, "Currency", "currency", default="")).strip().upper() == cand_ccy
            for cid, spec_c in rates_bundles.items()
        )
        has_ccy_fx = isinstance(vol_surfaces_map, Mapping) and any(
            cand_ccy in str(sid).strip().upper() for sid in vol_surfaces_map
        )
        if has_ccy_curve or has_ccy_fx or (not rates_bundles and not vol_surfaces_map):
            return cand_ccy

    # Tier 5: Default to payment_currency
    return pay_ccy_upper


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
        s_extrap = _extract_attr(surf_spec, "SmileExtrapolationMethod", "smile_interpolation_method")
        s_extrap = _extract_attr(surf_spec, "SmileExtrapolationMethod", "smile_extrapolation_method")
        if s_extrap is not None:
            kwargs["smile_extrap"] = str(getattr(s_extrap, "value", s_extrap))
    if "surface_id" in _VOL_SURF_SIG_PARAMS:
        eff_id = surface_id or _extract_attr(surf_spec, "Identifier", "identifier")
        if eff_id is not None:
            kwargs["surface_id"] = str(eff_id)

    key = digest_market_bytes("vol_surface", (surf_spec, as_of_dt, kwargs))
    if key is not None:
        cached = _MARKET_OBJECT_CACHE.get(key)
        if cached is not None:
            return cast(RegularizedVolSurface, cached)
    surf = RegularizedVolSurface(surf_spec, as_of_date=as_of_dt, **kwargs)
    if key is not None:
        return cast(RegularizedVolSurface, _MARKET_OBJECT_CACHE.put(key, surf))
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
            for cid, spec_c in rates_bundles.items():
                spec_ccy = str(_extract_attr(spec_c, "Currency", "currency", default="")).upper()
                if spec_ccy == payment_ccy.upper() or str(cid).upper().startswith(pay_prefix):
                    disc_curve_id = str(cid)
                    disc_curve_spec = spec_c
                    break
        if disc_curve_spec is None:
            disc_curve_id = str(next(iter(rates_bundles.keys())))
            disc_curve_spec = rates_bundles[disc_curve_id]

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
        raise KeyError("Missing Correlations map for multi-asset basket.")
    if isinstance(corr_input, torch.Tensor):
        return corr_input.to(dtype=torch.float64, device=device)
    if not isinstance(corr_input, Mapping):
        return torch.as_tensor(np.asarray(corr_input, dtype=np.float64), dtype=torch.float64, device=device)

    mat = np.eye(d_assets, dtype=np.float64)
    for i in range(d_assets):
        for j in range(i + 1, d_assets):
            id_i = sorted_ids[i]
            id_j = sorted_ids[j]
            k1 = f"{id_i}&{id_j}"
            k2 = f"{id_j}&{id_i}"
            if k1 in corr_input:
                c_val = float(corr_input[k1])
            elif k2 in corr_input:
                c_val = float(corr_input[k2])
            else:
                raise KeyError(f"Missing pairwise correlation for {k1!r}.")
            mat[i, j] = c_val
            mat[j, i] = c_val
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
    raw_dates = _extract_attr(f_obj, "Dates", "dates", default=())
    raw_vals = _extract_attr(f_obj, "Values", "values", default=())
    if not raw_dates or not raw_vals:
        return ()
    pairs = [(_to_utc_datetime(d), float(v)) for d, v in zip(raw_dates, raw_vals)]
    pairs.sort(key=lambda item: item[0])
    return tuple(pairs)


def _lookup_funding_fixing(
    fixings_pairs: Sequence[tuple[datetime.datetime, float]],
    target_dt: datetime.datetime,
) -> float | None:
    """Finds an exact or same-calendar-day UTC fixing for a funding reset date."""
    if not fixings_pairs:
        return None
    target_date = target_dt.date()
    exact: float | None = None
    for f_dt, f_val in fixings_pairs:
        if f_dt == target_dt:
            return float(f_val)
        if f_dt.date() == target_date:
            exact = float(f_val)
    return exact


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
) -> AutocallPreprocState:
    """Executes the Tier 1 deterministic `torch.float64` pre-processor on an Autocallable trade."""
    if isinstance(req_or_raw, (str, Path)):
        req_obj, _ = parse_murex_autocall_dag(req_or_raw)
    else:
        req_obj = req_or_raw

    product = _extract_attr(req_obj, "Product", "product")
    model = _extract_attr(req_obj, "Model", "model")
    mkt = _extract_attr(model, "MarketDataSet", "market_data_set") if model is not None else None
    if mkt is None:
        mkt = _extract_attr(req_obj, "MarketDataSet", "market_data")

    if product is None or mkt is None:
        raise ValueError("Request must contain Product and Model.MarketDataSet (or market_data).")

    trade_info = _extract_attr(req_obj, "TradeInfo", "trade_info", "trade_id")
    queries_raw = _extract_attr(req_obj, "Queries", "queries", default=["ProductValue", "CashFlows"])
    queries = [str(getattr(q, "value", q)) for q in queries_raw]

    as_of_raw = (
        as_of_date_override
        if as_of_date_override is not None
        else _extract_attr(mkt, "AsOfDate", "as_of_date")
    )
    as_of_dt = _to_utc_datetime(as_of_raw)

    start_dt = _to_utc_datetime(_extract_attr(product, "StartDate", "start_date"))
    payment_ccy = str(_extract_attr(product, "PaymentCurrency", "payment_currency", default="RUB"))
    dcc_raw = _extract_attr(product, "DayCountConvention", "day_count_convention", default="Act360")
    dcc = str(getattr(dcc_raw, "value", dcc_raw))

    underlying_ids = [
        str(x) for x in _extract_attr(product, "UnderlyingIdentifiers", "underlying_identifiers")
    ]
    ref_spots = [
        float(x) for x in _extract_attr(product, "UnderlyingReferenceSpots", "underlying_reference_spots")
    ]

    periods_obj = _extract_attr(product, "periods")
    if periods_obj is not None:
        obs_dates = [_extract_attr(p, "observation_date") for p in periods_obj]
        pay_dates = [_extract_attr(p, "payment_date") for p in periods_obj]
        ko_levels = [
            float(_extract_attr(p, "autocall_barrier", default=500.0) or 500.0) for p in periods_obj
        ]
        cpn_levels = [float(_extract_attr(p, "coupon_barrier", default=1e-5)) for p in periods_obj]
        cpn_amounts: list[float] | None = [
            float(_extract_attr(p, "coupon_amount", default=0.0)) for p in periods_obj
        ]
        fixed_cpn_amt: float | list[float] | None = None
        fixed_rate: float | list[float] | None = None
    else:
        obs_dates = list(_extract_attr(product, "ObservationDates", "observation_dates"))
        pay_dates = list(_extract_attr(product, "PaymentDates", "payment_dates"))
        ko_levels = [float(x) for x in _extract_attr(product, "AutoCallTriggerLevels", "autocall_trigger_levels")]
        raw_cpn_levels = _extract_attr(product, "CouponBarrierLevels", "coupon_barrier_levels")
        cpn_levels = [float(x) for x in raw_cpn_levels] if raw_cpn_levels is not None else [1e-5] * len(obs_dates)
        cpn_amounts = None
        fixed_cpn_amt = _normalize_coupon_param(
            _extract_attr(product, "FixedCouponAmount", "fixed_coupon_amount")
        )
        fixed_rate = _normalize_coupon_param(
            _extract_attr(product, "FixedRate", "fixed_rate")
        )

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

    strike = float(
        _extract_attr(downside_put, "strike")
        if downside_put is not None
        else _extract_attr(product, "Strike", "strike", default=0.90)
    )
    option_gearing = float(
        _extract_attr(downside_put, "gearing")
        if downside_put is not None
        else _extract_attr(product, "OptionGearing", "option_gearing", default=1.11111)
    )
    lower_strike_raw = (
        _extract_attr(downside_put, "lower_strike")
        if downside_put is not None
        else _extract_attr(product, "LowerStrike", "lower_strike")
    )
    lower_strike = float(lower_strike_raw) if lower_strike_raw is not None else None

    ki_barrier_raw = (
        _extract_attr(downside_put, "knock_in_barrier")
        if downside_put is not None
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
    elif ki_obs_type is None or ki_obs_type == "European":
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

    ki_spread_raw = (
        _extract_attr(downside_put, "knock_in_digital_spread")
        if downside_put is not None
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
    autocall_coupon = float(
        _extract_attr(modifiers, "autocall_redemption_coupon", default=0.0)
        if modifiers is not None
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
    part_strike_raw = (
        _extract_attr(modifiers, "participation_call_strike")
        if modifiers is not None
        else _extract_attr(product, "ParticipationCallStrike", "participation_call_strike")
    )
    part_strike = float(part_strike_raw) if part_strike_raw is not None else None
    part_rate = float(
        _extract_attr(modifiers, "participation_call_rate", default=0.0)
        if modifiers is not None
        else _extract_attr(product, "ParticipationCallRate", "participation_call_rate", default=0.0)
    )
    one_star_strike_raw = (
        _extract_attr(modifiers, "one_star_strike")
        if modifiers is not None
        else _extract_attr(product, "OneStarStrike", "one_star_strike")
    )
    one_star_strike = float(one_star_strike_raw) if one_star_strike_raw is not None else None
    one_star_coupon = float(
        _extract_attr(modifiers, "one_star_coupon", default=0.0)
        if modifiers is not None
        else _extract_attr(product, "OneStarCoupon", "one_star_coupon", default=0.0)
    )

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

    # FundingLeg extraction supporting explicit FundingLeg dates or Warrant PaymentDates fallback
    funding_leg_obj = _extract_attr(product, "funding_leg")
    rate_id = _extract_attr(product, "RateIdentifier", "rate_identifier")
    funding_leg_dates = _extract_attr(product, "FundingLegPaymentDates", "funding_leg_payment_dates")
    funding_spread = float(_extract_attr(product, "FundingSpread", "funding_spread", default=0.0))
    funding_leg_dict: dict[str, Any] | None = None
    if funding_leg_obj is not None:
        raw_f_dates = _extract_attr(funding_leg_obj, "payment_dates", default=[])
        if raw_f_dates:
            raw_rid = _extract_attr(funding_leg_obj, "rate_identifier")
            f_rid_str = str(raw_rid) if raw_rid is not None else None
            funding_leg_dict = {
                "rate_identifier": f_rid_str,
                "funding_spread": float(_extract_attr(funding_leg_obj, "funding_spread", default=0.0)),
                "payment_dates": [_to_utc_datetime(d) for d in raw_f_dates],
                "funding_fixings": _extract_funding_fixings(fixings_map, f_rid_str),
            }
    elif funding_leg_dates:
        f_rid_str = str(rate_id) if rate_id is not None else None
        funding_leg_dict = {
            "rate_identifier": f_rid_str,
            "funding_spread": funding_spread,
            "payment_dates": [_to_utc_datetime(d) for d in funding_leg_dates],
            "funding_fixings": _extract_funding_fixings(fixings_map, f_rid_str),
        }
    elif (rate_id is not None or funding_spread != 0.0) and pay_dates:
        f_rid_str = str(rate_id) if rate_id is not None else None
        funding_leg_dict = {
            "rate_identifier": f_rid_str,
            "funding_spread": funding_spread,
            "payment_dates": [_to_utc_datetime(d) for d in pay_dates],
            "funding_fixings": _extract_funding_fixings(fixings_map, f_rid_str),
        }

    # Content-addressed cache lookup hashing 100% of market and contract inputs
    cache_key: ContentDigestKey | None = None
    if (
        spots_override is None
        and corr_override is None
        and df_pay_override is None
        and dividend_amount_overrides is None
        and as_of_date_override is None
    ):
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
            cached_clone = cast("AutocallPreprocState | None", _PREPROC_STATE_CACHE.get(cache_key))
            if cached_clone is not None:
                cached_clone.queries = list(queries)
                cached_clone.trade_info = str(trade_info) if trade_info is not None else None
                return cached_clone

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
        f_curve_obj: Any = None
        f_curve_id: str | None = disc_curve_id
        if f_rid is not None:
            f_curve_id = str(f_rid)
            if isinstance(rates_bundles, Mapping) and f_rid in rates_bundles:
                f_curve_obj = (
                    disc_curve_obj
                    if (disc_curve_id is not None and f_rid == disc_curve_id)
                    else _get_cached_yield_curve(rates_bundles[f_rid], as_of_dt)
                )
            else:
                f_curve_obj = disc_curve_obj
                f_curve_id = disc_curve_id
        else:
            f_curve_obj = disc_curve_obj
        funding_leg_dict["funding_curve"] = f_curve_obj
        funding_leg_dict["funding_curve_id"] = f_curve_id

    _, df_fin_fn = build_discount_rate_evaluator(disc_curve_obj, as_of_dt)

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
        cpn_barriers=cpn_levels,
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
    iv_at_fwd_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    d_iv_dk_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    node_surf_feats_np = np.zeros((d_assets, 7), dtype=np.float64)
    surf_tau_obs_map: dict[str, npt.NDArray[np.float64]] = {}
    fwd_price_np = div_out.forward_price.detach().cpu().numpy().astype(np.float64)
    fwd_req_grad = bool(div_out.forward_price.requires_grad)

    if m_rem > 0 and not seasoned.already_autocalled:
        for i, ticker in enumerate(sorted_ids):
            if ticker not in vol_surfaces_map:
                raise KeyError(f"Missing EquityVolatilitySmileCollection for underlying {ticker!r}.")
            surf_spec = vol_surfaces_map[ticker]
            surf_i = _get_cached_vol_surface(surf_spec, as_of_dt, surface_id=ticker)
            reg_surfaces[ticker] = surf_i
            surf_warns = getattr(surf_i, "warnings", None)
            if isinstance(surf_warns, Sequence) and not isinstance(surf_warns, (str, bytes)):
                for w_msg in surf_warns:
                    if str(w_msg) not in seasoned_warnings:
                        seasoned_warnings.append(str(w_msg))

            raw_surf_dcc = _extract_attr(surf_spec, "DayCountConvention", "day_count_convention", default="Act365")
            surf_dcc = str(getattr(raw_surf_dcc, "value", raw_surf_dcc))
            tau_surf_obs = (
                np.asarray(seasoned.tau_obs, dtype=np.float64)
                if surf_dcc == "Act365"
                else year_fractions(as_of_dt, seasoned.future_obs_dates, surf_dcc)
            )
            surf_tau_obs_map[ticker] = tau_surf_obs

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

    v_perf_t = iv_at_fwd_t * torch.sqrt(tau_obs_t.clamp(min=1e-12)).unsqueeze(0)

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
    )
    if cache_key is not None:
        return cast(AutocallPreprocState, _PREPROC_STATE_CACHE.put(cache_key, state_out))
    return state_out


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
    cash_today = float(
        getattr(
            preproc_state,
            "cash_to_settle_today",
            getattr(preproc_state.seasoned_state, "cash_to_settle_today", 0.0),
        )
    )

    if p_stop is None and h_prob is not None:
        h_t = (
            h_prob.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
            * preproc_state.is_callable
        )
        one_minus_h = 1.0 - h_t.clamp(0.0, 1.0)
        surv_h = torch.cumprod(one_minus_h, dim=0)
        surv_prev_h = torch.cat(
            [torch.ones(1, dtype=torch.float64, device=device), surv_h[:-1]],
            dim=0,
        )
        p_stop = surv_prev_h * h_t
    if q_cpn is None and q_prob is not None:
        q_cpn = q_prob

    p_t = (
        preproc_state.baseline.p_proxy
        if p_stop is None
        else p_stop.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
    )
    q_t = (
        preproc_state.baseline.q_proxy
        if q_cpn is None
        else q_cpn.to(dtype=torch.float64, device=device).reshape(-1)[:m_rem]
    )
    l_put_t = (
        preproc_state.baseline.l_put_proxy
        if l_put is None
        else l_put.to(dtype=torch.float64, device=device).reshape(())
    )
    l_call_t = (
        getattr(preproc_state.baseline, "l_call_proxy", torch.tensor(0.0, dtype=torch.float64, device=device))
        if l_call is None
        else l_call.to(dtype=torch.float64, device=device).reshape(())
    )
    l_onestar_t = (
        getattr(preproc_state.baseline, "l_onestar_proxy", torch.tensor(0.0, dtype=torch.float64, device=device))
        if l_onestar is None
        else torch.as_tensor(l_onestar, dtype=torch.float64, device=device).reshape(())
    )

    if p_onestar is not None:
        p_onestar_raw = torch.as_tensor(p_onestar, dtype=torch.float64, device=device).reshape(())
    elif hasattr(preproc_state.baseline, "p_onestar_proxy") and getattr(preproc_state.baseline, "p_onestar_proxy") is not None:
        p_onestar_raw = torch.as_tensor(
            getattr(preproc_state.baseline, "p_onestar_proxy"), dtype=torch.float64, device=device
        ).reshape(())
    elif float(preproc_state.one_star_coupon) > 0.0:
        p_onestar_raw = l_onestar_t / float(preproc_state.one_star_coupon)
    else:
        p_onestar_raw = torch.tensor(0.0, dtype=torch.float64, device=device)

    p_t = p_t * preproc_state.is_callable

    cashflows: list[CashFlowItem] = (
        list(preproc_state.seasoned_state.past_cashflows) if build_cashflow_items else []
    )
    pv_in_transit_t = torch.tensor(preproc_state.pv_in_transit, dtype=torch.float64, device=device)

    if m_rem == 0 or preproc_state.already_autocalled:
        z_s = torch.tensor(0.0, dtype=torch.float64, device=device)
        if build_cashflow_items:
            for m in range(m_rem):
                pay_dt = preproc_state.seasoned_state.future_pay_dates[m]
                cashflows.append(CashFlowItem("FundingLeg", pay_dt, 0.0, 0.0, ccy, 0, "Estimate"))
                cashflows.append(CashFlowItem("Coupon", pay_dt, 0.0, 0.0, ccy, 0, "Estimate"))
                if m == m_rem - 1:
                    cashflows.append(CashFlowItem("Option", pay_dt, 0.0, 0.0, ccy, 0, "Estimate"))
                cashflows.append(CashFlowItem("Notional", pay_dt, 0.0, 0.0, ccy, 0, "Estimate"))
        return AutocallPostprocOutput(
            product_value=pv_in_transit_t,
            notional_pv=z_s,
            coupon_pv=z_s,
            option_pv=z_s,
            funding_pv=z_s,
            pv_in_transit=pv_in_transit_t,
            survival_probs=torch.zeros(m_rem, dtype=torch.float64, device=device),
            p_stop=p_t,
            q_cpn=q_t,
            l_put=l_put_t,
            l_call=l_call_t,
            cashflows=cashflows,
            standard_error=0.0,
            reporting_currency=ccy,
            cash_to_settle_today=cash_today,
            l_onestar=l_onestar_t,
            p_onestar=z_s,
        )

    cum_p = torch.cumsum(p_t, dim=0)
    survival_probs = (1.0 - cum_p).clamp(min=0.0)
    surv_prev = torch.cat(
        [torch.ones(1, dtype=torch.float64, device=device), survival_probs[:-1]],
        dim=0,
    )
    p_onestar_t = torch.minimum(p_onestar_raw.clamp(min=0.0), survival_probs[-1])

    df_pay = preproc_state.df_pay
    cpn_rates = preproc_state.coupon_amounts

    # Pure stateless FundingLeg evaluation without mutating preproc_state.funding_leg
    funding_cf_per_period: list[torch.Tensor] = [
        torch.tensor(0.0, dtype=torch.float64, device=device) for _ in range(m_rem)
    ]
    if preproc_state.funding_leg is not None:
        f_leg = preproc_state.funding_leg
        f_rid = f_leg.get("rate_identifier")
        f_curve = f_leg.get("funding_curve")
        f_curve_id = f_leg.get("funding_curve_id")
        disc_id = preproc_state.discount_curve_id
        as_of_dt = preproc_state.seasoned_state.as_of_date
        start_dt = preproc_state.seasoned_state.start_date
        f_dcc = preproc_state.seasoned_state.day_count_convention

        df_fin_fn: Callable[[datetime.datetime], torch.Tensor | float]
        if df_fin_fn_override is not None:
            df_fin_fn = df_fin_fn_override
        else:
            _, df_fin_fn = build_discount_rate_evaluator(preproc_state.discount_curve, as_of_dt)

        same_curve = (
            f_curve is None
            or f_curve is preproc_state.discount_curve
            or (f_curve_id is not None and disc_id is not None and f_curve_id == disc_id)
        )
        df_fwd_fn: Callable[[datetime.datetime], torch.Tensor | float]
        if df_fwd_fn_override is not None:
            df_fwd_fn = df_fwd_fn_override
        elif f_rid is not None and not same_curve:
            _, df_fwd_fn = build_discount_rate_evaluator(f_curve, as_of_dt)
        else:
            df_fwd_fn = df_fin_fn

        f_spread = float(f_leg.get("funding_spread", 0.0))
        f_dates = f_leg.get("payment_dates", ())
        f_fixings = f_leg.get("funding_fixings", ())

        prev_f_dt = start_dt
        for f_dt in f_dates:
            if f_dt <= as_of_dt:
                prev_f_dt = f_dt
                continue
            alpha_j = year_fraction(prev_f_dt, f_dt, f_dcc)
            df_curr = df_fin_fn(f_dt)
            if f_rid is None:
                r_fwd_j: Any = 0.0
            else:
                fixed_reset = _lookup_funding_fixing(f_fixings, prev_f_dt) if prev_f_dt < as_of_dt else None
                if fixed_reset is not None:
                    r_fwd_j = fixed_reset
                elif prev_f_dt < as_of_dt:
                    # Stub forward rate from valuation date when historical reset fixing is absent.
                    alpha_rem = max(year_fraction(as_of_dt, f_dt, f_dcc), 1e-8)
                    df_curr_fwd = df_fwd_fn(f_dt)
                    if isinstance(df_curr_fwd, torch.Tensor):
                        r_fwd_j = (1.0 / df_curr_fwd.clamp(min=1e-12) - 1.0) / alpha_rem
                    else:
                        r_fwd_j = (1.0 / max(float(df_curr_fwd), 1e-12) - 1.0) / alpha_rem
                else:
                    df_prev_fwd = df_fwd_fn(prev_f_dt)
                    df_curr_fwd = df_fwd_fn(f_dt)
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
                        r_fwd_j = (
                            float(df_prev_fwd) / max(float(df_curr_fwd), 1e-12) - 1.0
                        ) / max(alpha_j, 1e-8)

            m_idx = m_rem - 1
            for m in range(m_rem):
                if preproc_state.seasoned_state.future_pay_dates[m] >= f_dt:
                    m_idx = m
                    break
            cf_fund_j = surv_prev[m_idx] * df_curr * (r_fwd_j + f_spread) * alpha_j
            funding_cf_per_period[m_idx] = funding_cf_per_period[m_idx] + cf_fund_j
            prev_f_dt = f_dt

    coupon_cf_undisc_t = (
        torch.as_tensor(coupon_cf_undisc, dtype=torch.float64, device=device).reshape(-1)[:m_rem]
        if coupon_cf_undisc is not None
        else None
    )

    notional_cf_list: list[torch.Tensor] = []
    coupon_cf_list: list[torch.Tensor] = []
    mem_acc = torch.tensor(preproc_state.c_mem_0, dtype=torch.float64, device=device)

    for m in range(m_rem):
        if preproc_state.no_notional_repayment:
            cf_notional_m = torch.tensor(0.0, dtype=torch.float64, device=device)
        elif m < m_rem - 1:
            cf_notional_m = df_pay[m] * p_t[m]
        else:
            cf_notional_m = df_pay[m] * (p_t[m] + survival_probs[m])
        notional_cf_list.append(cf_notional_m)

        if coupon_cf_undisc_t is not None:
            cf_cpn_m = df_pay[m] * coupon_cf_undisc_t[m]
        else:
            c_m = cpn_rates[m]
            eff_c_m = (c_m + mem_acc) if preproc_state.is_memory else c_m
            if preproc_state.force_coupons_on_autocall:
                eff_q_m = torch.maximum(q_t[m], p_t[m] / surv_prev[m].clamp(min=1e-12))
            else:
                eff_q_m = q_t[m]
            cf_cpn_m = df_pay[m] * (
                surv_prev[m] * eff_q_m * eff_c_m + p_t[m] * float(preproc_state.autocall_coupon)
            )
            if preproc_state.is_memory:
                mem_acc = eff_c_m * (1.0 - eff_q_m)
        coupon_cf_list.append(cf_cpn_m)

    put_sign = -1.0
    cf_option_term = df_pay[-1] * (
        put_sign * float(preproc_state.option_gearing) * l_put_t
        + float(preproc_state.participation_call_rate) * l_call_t
        + l_onestar_t
    )

    notional_pv = torch.stack(notional_cf_list).sum()
    coupon_pv = torch.stack(coupon_cf_list).sum()
    funding_pv = torch.stack(funding_cf_per_period).sum()
    option_pv = cf_option_term

    product_value = pv_in_transit_t + notional_pv + coupon_pv + option_pv - funding_pv

    if build_cashflow_items:
        for m in range(m_rem):
            pay_dt = preproc_state.seasoned_state.future_pay_dates[m]
            fund_amt = float(funding_cf_per_period[m].detach().cpu().item())
            cpn_amt = float(coupon_cf_list[m].detach().cpu().item())
            notional_amt = float(notional_cf_list[m].detach().cpu().item())

            cashflows.append(
                CashFlowItem(
                    Description="FundingLeg",
                    Date=pay_dt,
                    Amount=-fund_amt if fund_amt != 0.0 else 0.0,
                    StandardError=0.0,
                    Currency=ccy,
                    Leg=0,
                    StatusText="Estimate",
                )
            )
            cashflows.append(
                CashFlowItem(
                    Description="Coupon",
                    Date=pay_dt,
                    Amount=cpn_amt,
                    StandardError=0.0,
                    Currency=ccy,
                    Leg=0,
                    StatusText="Estimate",
                )
            )
            if m == m_rem - 1:
                opt_amt = float(cf_option_term.detach().cpu().item())
                if preproc_state.is_long_basket:
                    stop_notional_amt = (
                        0.0
                        if preproc_state.no_notional_repayment
                        else float((df_pay[-1] * p_t[-1]).detach().cpu().item())
                    )
                    surv_notional_amt = (
                        0.0
                        if preproc_state.no_notional_repayment
                        else float((df_pay[-1] * survival_probs[-1]).detach().cpu().item())
                    )
                    cashflows.append(
                        CashFlowItem(
                            Description="IsLongBasket",  # type: ignore[arg-type]
                            Date=pay_dt,
                            Amount=surv_notional_amt + opt_amt,
                            StandardError=float(option_se),
                            Currency=ccy,
                            Leg=0,
                            StatusText="Estimate",
                        )
                    )
                    cashflows.append(
                        CashFlowItem(
                            Description="Option",
                            Date=pay_dt,
                            Amount=0.0,
                            StandardError=0.0,
                            Currency=ccy,
                            Leg=0,
                            StatusText="Estimate",
                        )
                    )
                    cashflows.append(
                        CashFlowItem(
                            Description="Notional",
                            Date=pay_dt,
                            Amount=stop_notional_amt,
                            StandardError=0.0,
                            Currency=ccy,
                            Leg=0,
                            StatusText="Estimate",
                        )
                    )
                    continue

                if preproc_state.one_star_strike is not None and float(preproc_state.one_star_strike) > 0.0:
                    p_star_val = float(p_onestar_t.detach().cpu().item())
                    if p_star_val > 0.0 or float(preproc_state.one_star_strike) > 0.0:
                        df_last = float(df_pay[-1].detach().cpu().item())
                        onestar_principal_amt = (
                            0.0 if preproc_state.no_notional_repayment else df_last * p_star_val
                        )
                        onestar_bonus_amt = df_last * float(l_onestar_t.detach().cpu().item())
                        onestar_total_amt = onestar_principal_amt + onestar_bonus_amt
                        notional_rem_amt = max(notional_amt - onestar_principal_amt, 0.0)
                        opt_only_amt = float(
                            (
                                df_pay[-1]
                                * (
                                    put_sign * float(preproc_state.option_gearing) * l_put_t
                                    + float(preproc_state.participation_call_rate) * l_call_t
                                )
                            )
                            .detach()
                            .cpu()
                            .item()
                        )
                        cashflows.append(
                            CashFlowItem(
                                Description="OneStarCoupon",  # type: ignore[arg-type]
                                Date=pay_dt,
                                Amount=onestar_total_amt,
                                StandardError=0.0,
                                Currency=ccy,
                                Leg=0,
                                StatusText="Estimate",
                            )
                        )
                        cashflows.append(
                            CashFlowItem(
                                Description="Option",
                                Date=pay_dt,
                                Amount=opt_only_amt,
                                StandardError=float(option_se),
                                Currency=ccy,
                                Leg=0,
                                StatusText="Estimate",
                            )
                        )
                        cashflows.append(
                            CashFlowItem(
                                Description="Notional",
                                Date=pay_dt,
                                Amount=notional_rem_amt,
                                StandardError=0.0,
                                Currency=ccy,
                                Leg=0,
                                StatusText="Estimate",
                            )
                        )
                        continue

                cashflows.append(
                    CashFlowItem(
                        Description="Option",
                        Date=pay_dt,
                        Amount=opt_amt,
                        StandardError=float(option_se),
                        Currency=ccy,
                        Leg=0,
                        StatusText="Estimate",
                    )
                )
            cashflows.append(
                CashFlowItem(
                    Description="Notional",
                    Date=pay_dt,
                    Amount=notional_amt,
                    StandardError=0.0,
                    Currency=ccy,
                    Leg=0,
                    StatusText="Estimate",
                )
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
    "digest_market_bytes",
    "AutocallPreprocState",
    "AutocallPostprocOutput",
    "preprocess_eq_autocall",
    "assemble_autocall_cashflows_and_pv",
]
