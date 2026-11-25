from __future__ import annotations

from array import array
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Union, cast

import numpy as np
import numpy.typing as npt
import orjson
from pydantic import ConfigDict, Field
from scipy.special import ndtr
import torch

from deepvol.api.eq_autocall_arrow import (
    MAX_BATCH_TRADES as ARROW_MAX_BATCH_TRADES,
    DecodedEQAutocallArrowBatch,
    decode_eq_autocall_arrow_request,
    encode_eq_autocall_arrow_response,
)
from deepvol.api.schemas_eq_autocall import (
    _DESCRIPTION_TO_SHORT_CODE,
    AutocallQueryMetric,
    AutoCallableProductSpec,
    CashFlowRow,
    DupireModelSpec,
    EQAutocallableTradeRequest,
    EQAutocallableTradeResponse,
    parse_murex_autocall_dag,
    to_decimal_merged_rho,
)
from deepvol.models.autocall._tier1_backend import get_tier1_backend
from deepvol.models.autocall.analytical_baseline import (
    compute_analytical_autocall_baseline,
)
from deepvol.market.eq_curves import year_fraction
from deepvol.models.autocall.escrowed_dividend import build_discount_rate_evaluator
from deepvol.models.autocall.feature_layout import (
    AutocallGlobalCol,
    AutocallNodeCol,
    AutocallSchedCol,
    CanonicalTier3Config,
    resolve_canonical_tier3_config,
)
import deepvol.models.autocall.greeks_engine as _greeks_mod
from deepvol.models.autocall.greeks_engine import (
    AutocallGreeksResult,
    BatchedSurrogateGreeksOutput,
    compute_batched_surrogate_greeks,
    compute_eq_autocall_greeks,
)
from deepvol.models.autocall.pre_post_processor import (
    D_MAX_DEFAULT,
    M_MAX_DEFAULT,
    AutocallPreprocState,
    _get_cached_vol_surface,
    _lookup_funding_fixing,
    _normalize_repo_curves_map,
    _resolve_vol_surface_kwargs,
    assemble_autocall_cashflows_and_pv,
    build_vol_surface_cache_key,
    digest_market_bytes,
    preprocess_eq_autocall,
)
from deepvol.models.autocall.seasoned_collapser import _to_utc_datetime
from deepvol.mrm.eq_autocall_guardian import (
    DEFAULT_GUARDIAN_STATS_PATH,
    AutocallFallbackReason,
    EQAutocallModelGuardian,
    GuardianPricingResult,
    SR262AuditRecord,
)
from deepvol.surrogates.eq_autocall_egno import (
    AutocallSurrogateOutput,
    MaskedEQAutocallSurrogate,
    assemble_surrogate_product_value,
)

DEFAULT_CHECKPOINT_PATH: Path = (
    Path(__file__).resolve().parents[3] / "artifacts" / "checkpoints" / "eq_autocall_egno.pt"
)
_ = ndtr(0.0)


def _load_surrogate_from_ckpt(
    checkpoint_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MaskedEQAutocallSurrogate, dict[str, Any]]:
    """Loads `MaskedEQAutocallSurrogate` using M4 `load_eq_autocall_surrogate` or direct `weights_only=True`."""
    try:
        train_mod = importlib.import_module("deepvol.training.train_eq_autocall")
        loader = getattr(train_mod, "load_eq_autocall_surrogate")
        return loader(checkpoint_path, device=device, dtype=dtype)  # type: ignore[no-any-return]
    except ImportError:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        model_kwargs: dict[str, Any] = {
            "d_max": int(cfg.get("d_max", D_MAX_DEFAULT)),
            "m_max": int(cfg.get("m_max", M_MAX_DEFAULT)),
            "d_model": int(cfg.get("d_model", 128)),
            "n_egno_layers": int(cfg.get("n_egno_layers", 3)),
            "n_sched_layers": int(cfg.get("n_sched_layers", 2)),
            "n_heads": int(cfg.get("n_heads", 4)),
        }
        for extra_k in ("d_node", "d_edge", "d_sched", "d_global", "n_func_codes", "func_emb_dim"):
            if extra_k in cfg and extra_k in _SURR_INIT_PARAMS:
                model_kwargs[extra_k] = int(cfg[extra_k])
        model = MaskedEQAutocallSurrogate(**model_kwargs).to(device=device, dtype=dtype)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(sd, strict=True)
        metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
        return model.eval(), dict(metrics)

try:
    _compiled_mod = importlib.import_module("deepvol.surrogates.compiled_runner")
    _CompiledSurrogateRunner: Any = getattr(_compiled_mod, "CompiledSurrogateRunner", None)
except ImportError:
    _CompiledSurrogateRunner = None

logger = logging.getLogger("deepvol.api.eq_autocall_runtime")

EngineMode = Literal["auto", "surrogate", "proxy", "qmc"]
_VALID_ENGINES: frozenset[str] = frozenset({"auto", "surrogate", "proxy", "qmc"})
MAX_BATCH_TRADES: int = int(ARROW_MAX_BATCH_TRADES)
CUDA_MICROBATCH_LIMIT_M48: int = 1024
_TINY_CUDA_BATCH_THRESHOLD: int = 128

_GREEKS_BATCH_PARAMS = frozenset(inspect.signature(compute_batched_surrogate_greeks).parameters.keys())
_GREEKS_SINGLE_PARAMS = frozenset(inspect.signature(compute_eq_autocall_greeks).parameters.keys())
_SUPPORTS_STABILIZE_F32_SURROGATE: bool = "stabilize_f32_surrogate" in _GREEKS_SINGLE_PARAMS
_SUPPORTS_GREEKS_REQUEST_PARAM: bool = "request" in _GREEKS_SINGLE_PARAMS
_SUPPORTS_GREEKS_THETA_PARAM: bool = "theta_roll_days" in _GREEKS_SINGLE_PARAMS
_SUPPORTS_GREEKS_THETA_CONV_PARAM: bool = "theta_convention" in _GREEKS_SINGLE_PARAMS
_QMC_CRN_FN = getattr(_greeks_mod, "compute_qmc_crn_greeks", None)
_QMC_CRN_SIG = inspect.signature(_QMC_CRN_FN) if callable(_QMC_CRN_FN) else None
_QMC_CRN_PARAMS = (
    frozenset(_QMC_CRN_SIG.parameters.keys())
    if _QMC_CRN_SIG is not None
    else frozenset()
)
_QMC_CRN_ACCEPTS_VARKW: bool = (
    any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _QMC_CRN_SIG.parameters.values())
    if _QMC_CRN_SIG is not None
    else False
)
_ASSEMBLE_SURR_PARAMS = frozenset(inspect.signature(assemble_surrogate_product_value).parameters.keys())
_SURR_INIT_PARAMS = tuple(inspect.signature(MaskedEQAutocallSurrogate.__init__).parameters.keys())
_GUARDIAN_PRICE_PARAMS = frozenset(
    inspect.signature(EQAutocallModelGuardian.price_with_guardian).parameters.keys()
)

_SURFACE_ARB_CACHE_LOCK = threading.RLock()
_SURFACE_ARB_CACHE: OrderedDict[tuple[str, str, int], tuple[bool, str | None]] = OrderedDict()
_SURFACE_ARB_FAST_CACHE: OrderedDict[tuple[Any, ...], tuple[bool, str | None]] = OrderedDict()
_SURFACE_ARB_CACHE_MAXSIZE: int = 512

_FX_TENOR_CACHE_LOCK = threading.Lock()
_FX_TENOR_YEARS_CACHE: OrderedDict[Any, float] = OrderedDict()
_FX_TENOR_CACHE_MAXSIZE: int = 512

_PARSE_REQUEST_CACHE_LOCK = threading.RLock()
_PARSE_REQUEST_CACHE: OrderedDict[
    tuple[Any, ...],
    tuple[EQAutocallableTradeRequest, bool, tuple[tuple[str | int, ...], ...]],
] = OrderedDict()
_PARSE_REQUEST_CACHE_MAXSIZE: int = 256


def _clone_cached_request(cached_req: EQAutocallableTradeRequest) -> EQAutocallableTradeRequest:
    return cached_req.model_copy(
        update={
            "Queries": list(cached_req.Queries),
            "stripped_legacy_fields": list(cached_req.stripped_legacy_fields),
        },
        deep=False,
    )


def _collect_null_paths(root: Any) -> tuple[tuple[str | int, ...], ...] | None:
    """Collects paths holding `None` in a validated dict (returns None if any float is non-finite)."""
    isfinite = math.isfinite
    null_paths: list[tuple[str | int, ...]] = []
    stack: list[tuple[Any, tuple[str | int, ...]]] = [(root, ())]
    while stack:
        cur, prefix = stack.pop()
        tc = type(cur)
        if tc is dict:
            for k, v in cur.items():
                if v is None:
                    null_paths.append((*prefix, k))
                elif type(v) is float:
                    if not isfinite(v):
                        return None
                elif type(v) is dict or type(v) is list or type(v) is tuple:
                    stack.append((v, (*prefix, k)))
        elif tc is list or tc is tuple:
            if cur:
                elem_types = set(map(type, cur))
                if elem_types == {str}:
                    continue
                if elem_types <= {float, int}:
                    try:
                        if all(map(isfinite, cur)):
                            continue
                    except TypeError:
                        pass
                    return None
                for idx_el, item in enumerate(cur):
                    if item is None:
                        null_paths.append((*prefix, idx_el))
                    elif type(item) is float:
                        if not isfinite(item):
                            return None
                    elif type(item) is dict or type(item) is list or type(item) is tuple:
                        stack.append((item, (*prefix, idx_el)))
    return tuple(null_paths)


def _check_no_nan_at_null_paths(root: Any, null_paths: tuple[tuple[str | int, ...], ...]) -> bool:
    """Verifies cached `None` positions are not masqueraded by non-finite floats in orjson output."""
    isfinite = math.isfinite
    for path in null_paths:
        cur = root
        try:
            for step in path:
                cur = cur[step]
        except (KeyError, IndexError, TypeError):
            return False
        if type(cur) is float and not isfinite(cur):
            return False
    return True


_CPU_GREEKS_CACHE_LOCK = threading.Lock()
_CPU_GREEKS_MICROBATCH_CACHE: OrderedDict[tuple[Any, ...], BatchedSurrogateGreeksOutput] = OrderedDict()
_CPU_GREEKS_MICROBATCH_MAXSIZE: int = 256
_SURROGATE_DIGEST_CACHE: dict[
    int, tuple[MaskedEQAutocallSurrogate, tuple[tuple[tuple[int, ...], int], ...], bytes]
] = {}


def _get_surrogate_content_digest(surrogate: MaskedEQAutocallSurrogate) -> bytes:
    """Computes a cached SHA-256 content digest over non-zero surrogate parameters."""
    if surrogate.is_zero_residual():
        return b"zero_res"
    params = list(surrogate.parameters())
    ver_sig = tuple((tuple(p.shape), 0 if p.is_inference() else int(p._version)) for p in params)
    sid = id(surrogate)
    cached = _SURROGATE_DIGEST_CACHE.get(sid)
    if cached is not None and cached[0] is surrogate and cached[1] == ver_sig:
        return cached[2]
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    digest = h.digest()
    if len(_SURROGATE_DIGEST_CACHE) >= 64:
        _SURROGATE_DIGEST_CACHE.clear()
    _SURROGATE_DIGEST_CACHE[sid] = (surrogate, ver_sig, digest)
    return digest


def _cpu_greeks_cache_key(
    surrogate: MaskedEQAutocallSurrogate,
    node_d: torch.Tensor,
    edge_d: torch.Tensor,
    amask_d: torch.Tensor,
    sched_d: torch.Tensor,
    smask_d: torch.Tensor,
    glob_d: torch.Tensor,
    compute_cash_gamma: bool,
    no_notional_repayment: bool,
    autocall_coupon: float,
    force_coupons_on_autocall: bool,
    is_long_basket: torch.Tensor | bool | None,
    w_fund_per_period: torch.Tensor | float | None,
    lower_strike: torch.Tensor | float | None,
) -> tuple[Any, ...]:
    """Builds a deterministic content digest key for CPU `compute_batched_surrogate_greeks` microbatches."""
    is_zero = surrogate.is_zero_residual()
    surr_tag: Any = (
        ("zero_res", int(surrogate.d_max), int(surrogate.m_max))
        if is_zero
        else (
            int(surrogate.d_max),
            int(surrogate.m_max),
            _get_surrogate_content_digest(surrogate),
        )
    )
    h = hashlib.blake2b(digest_size=16)
    for t in (node_d, edge_d, amask_d, sched_d, smask_d, glob_d):
        arr = t.detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    for opt_t in (is_long_basket, w_fund_per_period, lower_strike):
        if opt_t is None:
            h.update(b"N")
        elif isinstance(opt_t, torch.Tensor):
            h.update(b"T")
            h.update(opt_t.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(b"S")
            h.update(repr(opt_t).encode("utf-8"))
    return (
        surr_tag,
        node_d.shape,
        sched_d.shape,
        bool(compute_cash_gamma),
        bool(no_notional_repayment),
        float(autocall_coupon),
        bool(force_coupons_on_autocall),
        h.digest(),
    )

ALL_8_GREEK_QUERIES: tuple[str, ...] = (
    "Delta",
    "Gamma",
    "MultivariateAggregatedCashGamma",
    "Vega",
    "Rho",
    "DividendsSensitivities",
    "Theta",
    "Cega",
)

_MUREX_DAG_MARKER_RE = re.compile(r"(?m)(?:^@[0-9a-fA-F]{40}\s*$|class\s*:\s*ConstantMessageMap)")


class EQAutocallRuntimeResponse(EQAutocallableTradeResponse):
    """Canonical runtime trade response extending `EQAutocallableTradeResponse` with full Greek and MRM fields."""

    model_config = ConfigDict(extra="allow")

    Gamma: Optional[Union[Dict[str, Dict[str, float]], Dict[str, float], List[List[float]]]] = None
    Cega: Optional[Union[Dict[str, Dict[str, float]], Dict[str, float], List[List[float]]]] = None
    fallback_reasons: List[str] = Field(default_factory=list)
    stripped_legacy_fields: List[str] = Field(default_factory=list)
    Option: Optional[float] = None
    Notional: Optional[float] = None
    Coupon: Optional[float] = None
    FundingLeg: Optional[float] = None
    audit_record: Optional[Dict[str, Any]] = None
    RepoRho: Optional[Dict[str, List[float]]] = None
    rho_unit_convention: Literal["quote", "decimal"] = "quote"


def _put_surface_arb_cache(
    cache: OrderedDict[Any, tuple[bool, str | None]],
    key: Any,
    verdict: tuple[bool, str | None],
) -> None:
    cache[key] = verdict
    cache.move_to_end(key)
    while len(cache) > _SURFACE_ARB_CACHE_MAXSIZE:
        cache.popitem(last=False)


def _validate_fx_surface_quotes(ticker: str, surf: Any, quotes: npt.NDArray[np.float64]) -> None:
    if quotes.size == 0 or quotes.ndim != 2:
        raise ValueError(
            f"SMILE_ARBITRAGE: FX volatility quotes on surface={ticker} must be a non-empty 2D grid."
        )
    if not bool(np.all(np.isfinite(quotes))):
        bad_q = np.argwhere(~np.isfinite(quotes))
        m_b, j_b = int(bad_q[0, 0]), int(bad_q[0, 1])
        raise ValueError(
            f"SMILE_ARBITRAGE: Non-finite volatility quote on surface={ticker}, "
            f"slice={m_b}, col={j_b}, quote={float(quotes[m_b, j_b])!r}."
        )
    tenors_attr = getattr(surf, "Tenors", None)
    if tenors_attr is not None and (len(tenors_attr) == 0 or quotes.shape[0] != len(tenors_attr)):
        raise ValueError(
            f"SMILE_ARBITRAGE: FX Tenors count ({len(tenors_attr)}) != VolatilityQuotes rows ({quotes.shape[0]}) on surface={ticker}."
        )
    deltas_attr = getattr(surf, "Deltas", None)
    if deltas_attr is not None:
        if len(deltas_attr) == 0 or quotes.shape[1] != len(deltas_attr):
            raise ValueError(
                f"SMILE_ARBITRAGE: FX Deltas count ({len(deltas_attr)}) != VolatilityQuotes cols ({quotes.shape[1]}) on surface={ticker}."
            )
        for col_idx, delta_tok in enumerate(deltas_attr):
            tok_u = str(delta_tok).strip().upper()
            col = quotes[:, col_idx]
            if "RR" in tok_u or "BF" in tok_u:
                if bool(np.any(np.abs(col) > 5.00)):
                    m_b = int(np.argmax(np.abs(col)))
                    raise ValueError(
                        f"SMILE_ARBITRAGE: FX RR/BF quote out of bounds on surface={ticker}, "
                        f"slice={m_b}, col={delta_tok!r}, quote={float(col[m_b]):.5f} (exceeds |5.00|)."
                    )
            elif bool(np.any(col < 1e-4)) or bool(np.any(col > 5.00)):
                bad_col = np.flatnonzero((col < 1e-4) | (col > 5.00))
                m_b = int(bad_col[0])
                raise ValueError(
                    f"SMILE_ARBITRAGE: FX volatility quote out of certifiable bounds [1e-4, 5.00] on surface={ticker}, "
                    f"slice={m_b}, col={delta_tok!r}, quote={float(col[m_b]):.5f}."
                )
    elif bool(np.any(quotes < 1e-4)) or bool(np.any(quotes > 5.00)):
        bad_q = np.argwhere((quotes < 1e-4) | (quotes > 5.00))
        m_b, j_b = int(bad_q[0, 0]), int(bad_q[0, 1])
        raise ValueError(
            f"SMILE_ARBITRAGE: FX volatility quote out of certifiable bounds [1e-4, 5.00] on surface={ticker}, "
            f"slice={m_b}, col={j_b}, quote={float(quotes[m_b, j_b]):.5f}."
        )


def _validate_surfaces_for_arbitrage(req: EQAutocallableTradeRequest) -> None:
    """Screens input `VolatilitySurfaces` for corrupt or arbitrageable quotes before spline caching."""
    as_of_dt = req.Model.MarketDataSet.AsOfDate
    utc_tz = timezone.utc
    as_of_utc = (
        as_of_dt
        if (isinstance(as_of_dt, datetime) and as_of_dt.tzinfo is utc_tz)
        else _to_utc_datetime(as_of_dt)
    )
    vol_surfs = req.Model.MarketDataSet.VolatilitySurfaces

    prod_obj = getattr(req, "Product", None)
    prod_uids = list(getattr(prod_obj, "UnderlyingIdentifiers", ()) or ())
    seen_tickers: set[str] = set()
    ordered_surfs: list[tuple[str, Any]] = []
    for uid in prod_uids:
        if uid in vol_surfs and uid not in seen_tickers:
            seen_tickers.add(uid)
            ordered_surfs.append((uid, vol_surfs[uid]))
    for ticker, surf in vol_surfs.items():
        if ticker not in seen_tickers:
            seen_tickers.add(ticker)
            ordered_surfs.append((ticker, surf))

    deferred_spline_checks: list[tuple[str, Any, tuple[Any, ...]]] = []

    # Fast vectorized Black-76 discrete convexity pass catches severe smile arbitrage in <=0.4 ms cold.
    for ticker, surf in ordered_surfs:
        type_name = str(getattr(surf, "TypeName", "") or "")
        strikes_attr = getattr(surf, "Strikes", None)
        exp_dates_attr = getattr(surf, "ExpirationDates", None)
        if (
            type_name == "FXVolatilitySmileCollection"
            or getattr(surf, "Deltas", None) is not None
            or getattr(surf, "Tenors", None) is not None
            or strikes_attr is None
            or exp_dates_attr is None
        ):
            quotes_fx = np.asarray(surf.VolatilityQuotes, dtype=np.float64)
            _validate_fx_surface_quotes(str(ticker), surf, quotes_fx)
            continue

        n_exp = len(exp_dates_attr)
        if n_exp == 0:
            raise ValueError(f"SMILE_ARBITRAGE: Missing ExpirationDates on surface={ticker}.")

        dt_cls = datetime
        exp_utc_list = [
            d if (type(d) is dt_cls and d.tzinfo is utc_tz) else _to_utc_datetime(d)
            for d in exp_dates_attr
        ]
        dcc_attr = getattr(surf, "DayCountConvention", "Act365")
        dcc_str = str(getattr(dcc_attr, "value", dcc_attr) or "Act365")
        interp_attr = getattr(surf, "SmileInterpolationMethod", "CubicSpline")
        interp_str = str(getattr(interp_attr, "value", interp_attr) or "CubicSpline")
        extrap_attr = getattr(surf, "SmileExtrapolationMethod", "Linear")
        extrap_str = str(getattr(extrap_attr, "value", extrap_attr) or "Linear")
        surf_id = str(getattr(surf, "Identifier", ticker))

        raw_q = surf.VolatilityQuotes
        fast_arb_key: tuple[Any, ...] | None = None
        if (
            _SURFACE_ARB_FAST_CACHE
            and isinstance(strikes_attr, (list, tuple))
            and isinstance(raw_q, (list, tuple))
            and len(strikes_attr) > 0
            and len(raw_q) == n_exp
        ):
            n_s = len(strikes_attr)
            try:
                s_bytes = array("d", strikes_attr).tobytes()
                q_arr_raw = array("d")
                all_rows_match = True
                for q_row in raw_q:
                    if not isinstance(q_row, (list, tuple)) or len(q_row) != n_s:
                        all_rows_match = False
                        break
                    q_arr_raw.fromlist(q_row) if type(q_row) is list else q_arr_raw.extend(q_row)
                if all_rows_match:
                    fast_arb_key = (
                        str(ticker),
                        surf_id,
                        as_of_utc,
                        dcc_str,
                        interp_str,
                        extrap_str,
                        tuple(exp_utc_list),
                        s_bytes,
                        q_arr_raw.tobytes(),
                    )
                    with _SURFACE_ARB_CACHE_LOCK:
                        fast_verdict = _SURFACE_ARB_FAST_CACHE.get(fast_arb_key)
                        if fast_verdict is not None:
                            _SURFACE_ARB_FAST_CACHE.move_to_end(fast_arb_key)
                    if fast_verdict is not None:
                        is_valid_f, cached_err_f = fast_verdict
                        if not is_valid_f:
                            raise ValueError(
                                cached_err_f
                                or f"SMILE_ARBITRAGE: Invalid volatility surface on surface={ticker}."
                            )
                        continue
            except (TypeError, ValueError, OverflowError):
                fast_arb_key = None

        quotes = np.asarray(raw_q, dtype=np.float64)
        strikes = np.asarray(strikes_attr, dtype=np.float64)
        if quotes.size == 0 or quotes.ndim != 2:
            raise ValueError(
                f"SMILE_ARBITRAGE: VolatilityQuotes on surface={ticker} must be non-empty 2D."
            )
        q_min = float(quotes.min())
        q_max = float(quotes.max())
        if not (math.isfinite(q_min) and math.isfinite(q_max)):
            bad_q = np.argwhere(~np.isfinite(quotes))
            m_b, j_b = int(bad_q[0, 0]), int(bad_q[0, 1])
            k_b = float(strikes[j_b]) if (strikes.ndim == 1 and j_b < strikes.size) else float("nan")
            raise ValueError(
                f"SMILE_ARBITRAGE: Non-finite volatility surface quote on surface={ticker}, "
                f"slice={m_b}, K={k_b:.4f}, quote={float(quotes[m_b, j_b])!r}."
            )
        if strikes.size == 0:
            raise ValueError(f"SMILE_ARBITRAGE: Non-finite volatility surface strikes on surface={ticker}.")
        s_min = float(strikes.min())
        s_max = float(strikes.max())
        if not (math.isfinite(s_min) and math.isfinite(s_max)):
            bad_s = np.flatnonzero(~np.isfinite(strikes))
            j_b = int(bad_s[0]) if bad_s.size > 0 else 0
            raise ValueError(
                f"SMILE_ARBITRAGE: Non-finite volatility surface strike on surface={ticker}, "
                f"strike_idx={j_b}, K={float(strikes.flat[j_b])!r}."
            )

        if fast_arb_key is None:
            fast_arb_key = (
                str(ticker),
                surf_id,
                as_of_utc,
                dcc_str,
                interp_str,
                extrap_str,
                tuple(exp_utc_list),
                strikes.tobytes(),
                quotes.tobytes(),
            )
        fast_verdict = None
        if _SURFACE_ARB_FAST_CACHE:
            with _SURFACE_ARB_CACHE_LOCK:
                fast_verdict = _SURFACE_ARB_FAST_CACHE.get(fast_arb_key)
                if fast_verdict is not None:
                    _SURFACE_ARB_FAST_CACHE.move_to_end(fast_arb_key)
        if fast_verdict is not None:
            is_valid_f, cached_err_f = fast_verdict
            if not is_valid_f:
                raise ValueError(
                    cached_err_f or f"SMILE_ARBITRAGE: Invalid volatility surface on surface={ticker}."
                )
            continue

        try:
            if q_min < 1e-4 or q_max > 5.00:
                bad_q = np.argwhere((quotes < 1e-4) | (quotes > 5.00))
                m_b, j_b = int(bad_q[0, 0]), int(bad_q[0, 1])
                k_b = float(strikes[j_b]) if (strikes.ndim == 1 and j_b < strikes.size) else float("nan")
                raise ValueError(
                    f"SMILE_ARBITRAGE: Volatility quote out of certifiable bounds [1e-4, 5.00] on surface={ticker}, "
                    f"slice={m_b}, K={k_b:.4f}, quote={float(quotes[m_b, j_b]):.5f}."
                )
            if s_min <= 0.0:
                bad_k = np.flatnonzero(strikes <= 0.0)
                j_b = int(bad_k[0]) if bad_k.size > 0 else 0
                raise ValueError(
                    f"SMILE_ARBITRAGE: Non-positive strike on surface={ticker}, K={float(strikes.flat[j_b]):.4f}."
                )
            if strikes.size >= 2 and bool(np.all((strikes[1:] - strikes[:-1]) > 1e-9)):
                u_s = strikes
                already_sorted = True
            else:
                order = np.argsort(strikes)
                s_sorted = strikes[order]
                s_round = np.round(s_sorted, 9)
                keep = np.concatenate(([True], (s_round[1:] - s_round[:-1]) > 0.0))
                u_s = s_sorted[keep]
                already_sorted = False
            if len(u_s) < 2:
                raise ValueError(
                    f"SMILE_ARBITRAGE: Fewer than 2 distinct strikes on surface={ticker}."
                )
            prev_exp = as_of_utc
            t_vals: list[float] = []
            inv_sec_yr = 1.0 / 31536000.0
            for idx, exp_utc in enumerate(exp_utc_list):
                if idx == 0 and exp_utc <= as_of_utc:
                    raise ValueError(
                        f"SMILE_ARBITRAGE: ExpirationDate {exp_utc.isoformat()} <= AsOfDate on surface={ticker}, slice=0."
                    )
                if idx > 0 and exp_utc <= prev_exp:
                    raise ValueError(
                        f"SMILE_ARBITRAGE: Non-increasing ExpirationDates on surface={ticker}, slice={idx}."
                    )
                t_vals.append(max((exp_utc - as_of_utc).total_seconds() * inv_sec_yr, 1e-8))
                prev_exp = exp_utc

            if quotes.shape != (n_exp, strikes.size):
                raise ValueError(
                    f"SMILE_ARBITRAGE: Quotes shape {quotes.shape} != ({n_exp}, {strikes.size}) on surface={ticker}."
                )

            # Constant-volatility Black-76 smiles are strictly decreasing and convex across positive strikes.
            if q_max > q_min:
                q_u = quotes if already_sorted else quotes[:, order][:, keep]
                t_vec = np.asarray(t_vals, dtype=np.float64)[:, None]
                k_ref = float(u_s[len(u_s) >> 1]) if already_sorted else float(np.median(u_s))
                k_norm = u_s / k_ref
                y = np.log(k_norm)[None, :]
                sqrt_w = np.sqrt(np.maximum((q_u * q_u) * t_vec, 1e-12))
                d1 = -y / sqrt_w + 0.5 * sqrt_w
                d2 = d1 - sqrt_w
                calls = ndtr(d1) - k_norm[None, :] * ndtr(d2)
                dk = k_norm[1:] - k_norm[:-1]
                slopes = (calls[:, 1:] - calls[:, :-1]) / dk[None, :]
                max_slope = float(slopes.max())
                min_slope = float(slopes.min())
                d2c = slopes[:, 1:] - slopes[:, :-1] if len(u_s) >= 3 else None
                min_d2c = float(d2c.min()) if d2c is not None else 0.0
                slice_slope_bad = (slopes.max(axis=1) > 1e-3) | (slopes.min(axis=1) < -1.0 - 1e-3)
                slice_d2c_bad = (
                    d2c.min(axis=1) < -1e-3
                    if d2c is not None
                    else np.zeros(n_exp, dtype=bool)
                )
                violating_slices = [int(m) for m in np.flatnonzero(slice_slope_bad | slice_d2c_bad)]
                if max_slope > 1e-3 or min_slope < -1.0 - 1e-3:
                    if max_slope > 1e-3 and (min_slope >= -1.0 - 1e-3 or max_slope >= -1.0 - min_slope):
                        m_off, j_seg = map(int, np.unravel_index(int(np.argmax(slopes)), slopes.shape))
                        off_slope = max_slope
                    else:
                        m_off, j_seg = map(int, np.unravel_index(int(np.argmin(slopes)), slopes.shape))
                        off_slope = min_slope
                    j_prev = j_seg
                    j_curr = min(j_seg + 1, len(u_s) - 1)
                    raise ValueError(
                        f"SMILE_ARBITRAGE: Call-spread slope violation on surface={ticker}, "
                        f"slices={violating_slices}, worst_slice={m_off} (T={t_vals[m_off]:.4f}Y), "
                        f"K={float(u_s[j_curr]):.4f}, prev_K={float(u_s[j_prev]):.4f}, "
                        f"vol_jump={float(q_u[m_off, j_prev]):.5f}->{float(q_u[m_off, j_curr]):.5f}, "
                        f"slope={off_slope:.6f} (range [{min_slope:.4f}, {max_slope:.4f}]), "
                        f"min_d2c={min_d2c:.6f}."
                    )
                if d2c is not None and min_d2c < -1e-3:
                    m_off, j_d2 = map(int, np.unravel_index(int(np.argmin(d2c)), d2c.shape))
                    left_jump = abs(float(q_u[m_off, j_d2 + 1] - q_u[m_off, j_d2]))
                    right_jump = (
                        abs(float(q_u[m_off, j_d2 + 2] - q_u[m_off, j_d2 + 1]))
                        if j_d2 + 2 < len(u_s)
                        else -1.0
                    )
                    if j_d2 + 2 < len(u_s) and right_jump > left_jump:
                        j_prev = j_d2 + 1
                        j_curr = j_d2 + 2
                        off_slope = float(slopes[m_off, j_d2 + 1])
                    else:
                        j_prev = j_d2
                        j_curr = min(j_d2 + 1, len(u_s) - 1)
                        off_slope = float(slopes[m_off, j_d2])
                    raise ValueError(
                        f"SMILE_ARBITRAGE: Discrete call convexity violation on surface={ticker}, "
                        f"slices={violating_slices}, worst_slice={m_off} (T={t_vals[m_off]:.4f}Y), "
                        f"K={float(u_s[j_curr]):.4f}, prev_K={float(u_s[j_prev]):.4f}, "
                        f"vol_jump={float(q_u[m_off, j_prev]):.5f}->{float(q_u[m_off, j_curr]):.5f}, "
                        f"slope={off_slope:.6f}, min_d2c={min_d2c:.6f}, "
                        f"slope_range=[{min_slope:.4f}, {max_slope:.4f}]."
                    )
        except ValueError as exc:
            err_msg = str(exc)
            if "SMILE_ARBITRAGE" in err_msg:
                with _SURFACE_ARB_CACHE_LOCK:
                    _put_surface_arb_cache(_SURFACE_ARB_FAST_CACHE, fast_arb_key, (False, err_msg))
            raise

        deferred_spline_checks.append((str(ticker), surf, fast_arb_key))

    for ticker, surf, fast_arb_key in deferred_spline_checks:
        surf_kwargs = _resolve_vol_surface_kwargs(surf, surface_id=ticker)
        surf_key_obj = build_vol_surface_cache_key(
            surf,
            as_of_utc,
            surface_id=ticker,
            kwargs=surf_kwargs,
        )
        arb_key: tuple[str, str, int] | None = (
            (ticker, surf_key_obj.digest_hex, surf_key_obj.payload_len)
            if surf_key_obj is not None
            else None
        )
        if arb_key is not None:
            with _SURFACE_ARB_CACHE_LOCK:
                cached_verdict = _SURFACE_ARB_CACHE.get(arb_key)
                if cached_verdict is not None:
                    _SURFACE_ARB_CACHE.move_to_end(arb_key)
                    _put_surface_arb_cache(_SURFACE_ARB_FAST_CACHE, fast_arb_key, cached_verdict)
            if cached_verdict is not None:
                is_valid, cached_err = cached_verdict
                if not is_valid:
                    raise ValueError(cached_err or f"SMILE_ARBITRAGE: Invalid volatility surface for {ticker}.")
                continue

        try:
            _get_cached_vol_surface(surf, as_of_utc, surface_id=ticker, _cache_key=surf_key_obj)
        except ValueError as exc:
            err_msg = str(exc)
            if "SMILE_ARBITRAGE" in err_msg:
                with _SURFACE_ARB_CACHE_LOCK:
                    if arb_key is not None:
                        _put_surface_arb_cache(_SURFACE_ARB_CACHE, arb_key, (False, err_msg))
                    _put_surface_arb_cache(_SURFACE_ARB_FAST_CACHE, fast_arb_key, (False, err_msg))
            raise

        with _SURFACE_ARB_CACHE_LOCK:
            if arb_key is not None:
                _put_surface_arb_cache(_SURFACE_ARB_CACHE, arb_key, (True, None))
            _put_surface_arb_cache(_SURFACE_ARB_FAST_CACHE, fast_arb_key, (True, None))


def _repair_request_surfaces(req: EQAutocallableTradeRequest) -> EQAutocallableTradeRequest:
    """Clones `req` and projects arbitrageable equity smiles via `repair_arbitrage_quotes` before pre-gating."""
    try:
        solver_mod = importlib.import_module("deepvol.models.autocall.dupire_qmc_solver")
        repair_fn = getattr(solver_mod, "repair_arbitrage_quotes", None)
    except ImportError:
        repair_fn = None
    if not callable(repair_fn):
        raise RuntimeError(
            "repair_arbitrage=True requires dupire_qmc_solver.repair_arbitrage_quotes, which is unavailable."
        )

    mds = req.Model.MarketDataSet
    as_of_dt = mds.AsOfDate
    as_of_utc = (
        as_of_dt
        if (isinstance(as_of_dt, datetime) and as_of_dt.tzinfo is timezone.utc)
        else _to_utc_datetime(as_of_dt)
    )
    spots_map = getattr(mds, "Spots", None) or {}
    vol_surfs = getattr(mds, "VolatilitySurfaces", None) or {}
    repaired_surfs: dict[str, Any] = dict(vol_surfs)
    any_changed = False
    inv_sec_yr = 1.0 / 31536000.0

    for ticker, surf in vol_surfs.items():
        type_name = str(getattr(surf, "TypeName", "") or "")
        strikes_attr = getattr(surf, "Strikes", None)
        exp_dates_attr = getattr(surf, "ExpirationDates", None)
        if (
            type_name == "FXVolatilitySmileCollection"
            or getattr(surf, "Deltas", None) is not None
            or getattr(surf, "Tenors", None) is not None
            or strikes_attr is None
            or exp_dates_attr is None
        ):
            continue
        strikes = np.asarray(strikes_attr, dtype=np.float64)
        quotes = np.asarray(surf.VolatilityQuotes, dtype=np.float64)
        if (
            strikes.ndim != 1
            or strikes.size < 2
            or quotes.ndim != 2
            or quotes.shape != (len(exp_dates_attr), strikes.size)
            or not bool(np.all(np.isfinite(strikes)))
            or not bool(np.all(np.isfinite(quotes)))
            or float(strikes.min()) <= 0.0
        ):
            continue
        t_exp = np.array(
            [max((_to_utc_datetime(exp_d) - as_of_utc).total_seconds() * inv_sec_yr, 1e-8) for exp_d in exp_dates_attr],
            dtype=np.float64,
        )
        raw_spot = spots_map.get(ticker)
        k_ref = (
            float(raw_spot)
            if (raw_spot is not None and math.isfinite(float(raw_spot)) and float(raw_spot) > 0.0)
            else float(np.median(strikes))
        )
        repaired_q_raw, repair_info = repair_fn(strikes, quotes, t_exp, k_ref)
        repaired_q = np.asarray(repaired_q_raw, dtype=np.float64)
        was_repaired = (
            bool(repair_info.get("repaired", False))
            if isinstance(repair_info, Mapping)
            else not np.array_equal(repaired_q, quotes)
        )
        if was_repaired or not np.array_equal(repaired_q, quotes):
            any_changed = True
            if hasattr(surf, "model_copy"):
                repaired_surfs[ticker] = surf.model_copy(
                    update={"VolatilityQuotes": repaired_q.tolist()},
                    deep=False,
                )
            else:
                repaired_surfs[ticker] = surf

    if not any_changed:
        return req

    new_mds = mds.model_copy(update={"VolatilitySurfaces": repaired_surfs}, deep=False)
    for attr in ("_cached_mds_digest_hex", "_batch_arbitrage_checked"):
        new_mds.__dict__.pop(attr, None)
    new_model = req.Model.model_copy(update={"MarketDataSet": new_mds}, deep=False)
    repaired_req = req.model_copy(update={"Model": new_model}, deep=False)
    return _strip_untrusted_digests(repaired_req)


def _validate_engine(engine: str) -> str:
    """Strictly validates and normalizes `engine` against `EngineMode` (`auto`, `surrogate`, `proxy`, `qmc`)."""
    eng_norm = str(engine).strip().lower()
    if eng_norm not in _VALID_ENGINES:
        raise ValueError(
            f"Unsupported engine {engine!r}; expected one of {sorted(_VALID_ENGINES)}."
        )
    return eng_norm


def _build_cpu_surrogate_mirror(surrogate: MaskedEQAutocallSurrogate) -> MaskedEQAutocallSurrogate:
    """Constructs an exact CPU float32 mirror matching `surrogate` constructor attributes."""
    candidate_attrs = (
        "d_max",
        "m_max",
        "d_model",
        "n_egno_layers",
        "n_sched_layers",
        "n_heads",
        "d_node",
        "d_edge",
        "d_sched",
        "d_global",
        "init_tau",
        "n_func_codes",
        "func_emb_dim",
    )
    kwargs: dict[str, Any] = {}
    allowed = set(_SURR_INIT_PARAMS)
    for attr in candidate_attrs:
        if attr in allowed and hasattr(surrogate, attr):
            kwargs[attr] = int(getattr(surrogate, attr)) if attr != "init_tau" else float(getattr(surrogate, attr))
    cpu_surr = MaskedEQAutocallSurrogate(**kwargs).to(device="cpu", dtype=torch.float32)
    cpu_surr.load_state_dict(
        {k: v.detach().cpu() for k, v in surrogate.state_dict().items()},
        strict=True,
    )
    cpu_surr.eval()
    return cpu_surr


def _decode_residuals_to_surrogate_output(
    delta_hazard: torch.Tensor,
    delta_coupon: torch.Tensor,
    delta_put_loss: torch.Tensor,
    sched_feats: torch.Tensor,
    sched_mask: torch.Tensor,
    global_feats: torch.Tensor,
) -> AutocallSurrogateOutput:
    """Reconstructs `AutocallSurrogateOutput` probabilities from raw residual predictions in float64 put space."""
    active_sched = sched_mask > 0.5
    m_sched = active_sched.to(dtype=sched_feats.dtype)
    sched_clean = torch.where(active_sched.unsqueeze(-1), sched_feats, torch.zeros_like(sched_feats))

    c_call = sched_clean[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)].detach().clamp(0.0, 1.0) * m_sched
    g_cpn = sched_clean[:, :, int(AutocallSchedCol.IS_GUARANTEED_CPN)].detach().clamp(0.0, 1.0) * m_sched
    p_proxy = sched_clean[:, :, int(AutocallSchedCol.PROXY_KO_PROB)].clamp(0.0, 1.0) * c_call
    q_proxy = sched_clean[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)].clamp(0.0, 1.0) * m_sched

    cum_p_proxy = torch.cumsum(p_proxy, dim=1)
    surv_prev_proxy = (1.0 - (cum_p_proxy - p_proxy)).clamp(min=1e-6)
    h_proxy = (p_proxy / surv_prev_proxy).clamp(min=1e-6, max=1.0 - 1e-6)
    logit_h_proxy = torch.log(h_proxy / (1.0 - h_proxy))

    hazard = c_call * m_sched * torch.sigmoid(logit_h_proxy + delta_hazard)
    survival = torch.cumprod(1.0 - hazard, dim=1)
    surv_prev = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)
    stopping = surv_prev * hazard

    q_proxy_clamped = q_proxy.clamp(min=1e-6, max=1.0 - 1e-6)
    logit_q_proxy = torch.log(q_proxy_clamped / (1.0 - q_proxy_clamped))
    q_gate = (q_proxy * q_proxy) / (q_proxy * q_proxy + 1e-10)
    q_pred = q_gate * torch.sigmoid(logit_q_proxy + delta_coupon)
    coupon = m_sched * (g_cpn + (1.0 - g_cpn) * q_pred)

    k_put_64 = global_feats[:, int(AutocallGlobalCol.K_PUT)].to(torch.float64).clamp(min=0.0)
    if global_feats.shape[-1] > int(AutocallGlobalCol.LOWER_STRIKE):
        k_low_64 = global_feats[:, int(AutocallGlobalCol.LOWER_STRIKE)].to(torch.float64).clamp(min=0.0)
    else:
        k_low_64 = torch.zeros_like(k_put_64)
    k_eff_64 = torch.clamp(k_put_64 - k_low_64, min=0.0)
    l_put_proxy_64 = global_feats[:, int(AutocallGlobalCol.L_PUT_PROXY)].to(torch.float64).clamp(min=0.0)
    surv_proxy_term_64 = (1.0 - p_proxy.sum(dim=1)).to(torch.float64).clamp(min=0.0)
    denom_64 = torch.clamp(k_eff_64 * surv_proxy_term_64, min=1e-12)
    eps_r = 1e-12
    r_proxy_64 = torch.clamp(l_put_proxy_64 / denom_64, min=eps_r, max=1.0 - eps_r)
    logit_r_64 = torch.log(r_proxy_64 / (1.0 - r_proxy_64))
    u_put_64 = k_eff_64 * survival[:, -1].to(torch.float64)
    put_loss = (u_put_64 * torch.sigmoid(logit_r_64 + delta_put_loss.to(torch.float64))).to(
        dtype=sched_feats.dtype
    )

    return AutocallSurrogateOutput(
        stopping_probs=stopping,
        survival_probs=survival,
        hazard_rates=hazard,
        coupon_probs=coupon,
        put_loss=put_loss,
        delta_hazard_logits=delta_hazard,
        delta_coupon_logits=delta_coupon,
        delta_put_loss=delta_put_loss,
    )


def _call_with_supported_kwargs(
    fn: Callable[..., Any],
    allowed_params: frozenset[str],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Forwards only keyword arguments accepted by `fn` (preserves pre-/post-merge shard compatibility)."""
    filtered = {k: v for k, v in kwargs.items() if k in allowed_params and v is not None}
    return fn(*args, **filtered)


def _trim_active_dimensions(
    node_feats: torch.Tensor,
    edge_feats: torch.Tensor,
    asset_mask: torch.Tensor,
    sched_feats: torch.Tensor,
    sched_mask: torch.Tensor,
) -> tuple[int, int]:
    """Computes minimal active `(d_act, m_act)` across a batch using masks."""
    amask_b = asset_mask > 0.5
    smask_b = sched_mask > 0.5
    any_a = amask_b.any(dim=0)
    any_s = smask_b.any(dim=0)
    d_act = int(any_a.nonzero()[-1].item()) + 1 if bool(any_a.any()) else 1
    m_act = int(any_s.nonzero()[-1].item()) + 1 if bool(any_s.any()) else 1
    return min(d_act, int(node_feats.shape[1])), min(m_act, int(sched_feats.shape[1]))


def _concat_batched_greeks(
    chunks: list[BatchedSurrogateGreeksOutput],
) -> BatchedSurrogateGreeksOutput:
    """Concatenates chunked `BatchedSurrogateGreeksOutput` slices along dimension 0."""
    if len(chunks) == 1:
        return chunks[0]
    first = chunks[0]
    kwargs: dict[str, Any] = {}
    for field_name in (
        "product_value",
        "stopping_probs",
        "survival_probs",
        "coupon_probs",
        "put_loss",
        "delta",
        "vega_atm",
        "vega_surface_feats",
        "rho_df",
        "cega",
        "cash_gamma_euler",
        "div_sensitivities",
        "theta",
    ):
        if not hasattr(first, field_name):
            continue
        vals = [getattr(c, field_name) for c in chunks]
        if vals[0] is None:
            kwargs[field_name] = None
        else:
            kwargs[field_name] = torch.cat(vals, dim=0)
    return BatchedSurrogateGreeksOutput(**kwargs)


def _parse_fx_tenor_to_years(tenor_val: Any) -> float:
    """Converts an FX surface tenor token (e.g., '1W', '1M', '3M', '1Y' or float) to Act/365 years."""
    if isinstance(tenor_val, bool):
        raise ValueError(f"Invalid boolean FX tenor: {tenor_val!r}")
    if isinstance(tenor_val, (int, float, np.number)):
        val_f = float(tenor_val)
        if not math.isfinite(val_f):
            raise ValueError(f"Non-finite FX tenor: {tenor_val!r}")
        return max(val_f, 1e-6)
    with _FX_TENOR_CACHE_LOCK:
        cached_y = _FX_TENOR_YEARS_CACHE.get(tenor_val)
        if cached_y is not None:
            _FX_TENOR_YEARS_CACHE.move_to_end(tenor_val)
            return cached_y
    s = str(tenor_val).strip().upper()
    res_y = 1.0
    if s:
        if s in ("ON", "O/N", "TN", "T/N", "SN", "S/N"):
            res_y = 1.0 / 365.0
        else:
            m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([DWMY])$", s)
            if m:
                num = float(m.group(1))
                unit = m.group(2)
                if unit == "D":
                    res_y = max(num / 365.0, 1e-6)
                elif unit == "W":
                    res_y = max((num * 7.0) / 365.0, 1e-6)
                elif unit == "M":
                    res_y = max(num / 12.0, 1e-6)
                elif unit == "Y":
                    res_y = max(num, 1e-6)
            else:
                try:
                    num_f = float(s)
                    if not math.isfinite(num_f):
                        raise ValueError(f"Non-finite FX tenor: {tenor_val!r}")
                    res_y = max(num_f, 1e-6)
                except ValueError:
                    res_y = 1.0
    if isinstance(tenor_val, str) and math.isfinite(res_y):
        with _FX_TENOR_CACHE_LOCK:
            _FX_TENOR_YEARS_CACHE[tenor_val] = res_y
            _FX_TENOR_YEARS_CACHE.move_to_end(tenor_val)
            while len(_FX_TENOR_YEARS_CACHE) > _FX_TENOR_CACHE_MAXSIZE:
                _FX_TENOR_YEARS_CACHE.popitem(last=False)
    return res_y


class _CegaDict(dict):  # type: ignore[type-arg]
    """Dictionary subclass for Result.Cega supporting both nested [A][B] and pair 'A&B' key lookups."""

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str) and "&" in key:
            a, b = key.split("&", 1)
            row_a = super().get(a)
            if isinstance(row_a, Mapping) and b in row_a:
                return True
            row_b = super().get(b)
            if isinstance(row_b, Mapping) and a in row_b:
                return True
        return False

    def __getitem__(self, key: Any) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str) and "&" in key:
            a, b = key.split("&", 1)
            row_a = super().get(a)
            if isinstance(row_a, Mapping) and b in row_a:
                return row_a[b]
            row_b = super().get(b)
            if isinstance(row_b, Mapping) and a in row_b:
                return row_b[a]
        raise KeyError(key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _strip_untrusted_digests(req: EQAutocallableTradeRequest) -> EQAutocallableTradeRequest:
    """Returns a shallow copy without cached raw-payload digests for caller-supplied request objects."""
    if (
        "_raw_payload_digest_hex" not in req.__dict__
        and "_cached_mds_digest_hex" not in req.__dict__
        and "_immutable_cache_source" not in req.__dict__
        and "_arb_validated" not in req.__dict__
    ):
        return req
    req_clean = req.model_copy(deep=False)
    for attr in (
        "_raw_payload_digest_hex",
        "_cached_mds_digest_hex",
        "_immutable_cache_source",
        "_arb_validated",
    ):
        req_clean.__dict__.pop(attr, None)
    return req_clean


def _resolve_request_mds_digest_hex(
    req_model: EQAutocallableTradeRequest,
    *,
    allow_cached: bool = True,
    known_digest: str | None = None,
) -> str:
    """Returns a per-invocation or immutable-cached MarketDataSet blake2b hex digest."""
    if known_digest is not None:
        return known_digest
    has_raw_digest = isinstance(getattr(req_model, "_raw_payload_digest_hex", None), str)
    if allow_cached and has_raw_digest:
        cached_hex = getattr(req_model, "_cached_mds_digest_hex", None)
        if isinstance(cached_hex, str) and len(cached_hex) > 0:
            return cached_hex
    mds_hex = _mds_digest_hex(req_model.Model.MarketDataSet)
    if allow_cached and has_raw_digest:
        try:
            object.__setattr__(req_model, "_cached_mds_digest_hex", mds_hex)
        except (AttributeError, TypeError):
            pass
        src_req = getattr(req_model, "_immutable_cache_source", None)
        if src_req is not None and src_req is not req_model:
            try:
                object.__setattr__(src_req, "_cached_mds_digest_hex", mds_hex)
            except (AttributeError, TypeError):
                pass
    return mds_hex


def _attach_runtime_market_metadata(
    state: AutocallPreprocState,
    req_model: EQAutocallableTradeRequest,
    *,
    mds_digest_hex: str | None = None,
    allow_cached: bool = True,
) -> str:
    """Attaches raw market specs (RepoSpreadCurves, FX VolatilitySurfaces) and warning flags onto state."""
    mds = req_model.Model.MarketDataSet
    ki_lvl = getattr(req_model.Product, "OptionDownAndInBarrierLevel", None)
    ki_obs_type = getattr(req_model.Product, "OptionDownAndInBarrierObservationType", None)
    eff_mds_hex = _resolve_request_mds_digest_hex(
        req_model,
        allow_cached=allow_cached,
        known_digest=mds_digest_hex,
    )
    meta_tag = (
        getattr(state, "state_digest", None) or id(state),
        eff_mds_hex,
        ki_lvl,
        ki_obs_type,
    )
    if (
        getattr(state, "_runtime_meta_tag", None) == meta_tag
        and (not getattr(mds, "RepoSpreadCurves", None) or getattr(state, "repo_curves_map", None))
        and (not getattr(state, "quanto_meta", None) or getattr(state, "quanto_fx_specs", None))
    ):
        return eff_mds_hex
    if getattr(mds, "RepoSpreadCurves", None):
        existing_rc = getattr(state, "repo_curves_map", None)
        if not existing_rc:
            setattr(
                state,
                "repo_curves_map",
                _normalize_repo_curves_map(mds.RepoSpreadCurves, mds.AsOfDate),
            )
        elif any(
            not (callable(getattr(v, "repo_df", None)) or callable(getattr(v, "df", None)))
            for v in existing_rc.values()
        ):
            setattr(
                state,
                "repo_curves_map",
                _normalize_repo_curves_map(existing_rc, mds.AsOfDate),
            )
    fx_specs: dict[str, Any] = {}
    quanto_meta = getattr(state, "quanto_meta", None) or {}
    for ticker, qm in quanto_meta.items():
        if not isinstance(qm, Mapping):
            continue
        fx_id = str(qm.get("fx_id", ""))
        if fx_id and fx_id in mds.VolatilitySurfaces:
            fx_specs[fx_id] = mds.VolatilitySurfaces[fx_id]
        if "fx_sign" not in qm and isinstance(qm, dict):
            pay_ccy = str(state.payment_currency).strip().upper()
            fx_clean = fx_id.strip().upper().replace("_", "").replace("/", "")
            qm["fx_sign"] = (
                -1.0
                if (pay_ccy and len(fx_clean) >= len(pay_ccy) and fx_clean.startswith(pay_ccy) and not fx_clean.endswith(pay_ccy))
                else 1.0
            )
    if fx_specs:
        setattr(state, "quanto_fx_specs", fx_specs)

    if ki_lvl is not None and float(ki_lvl) > 0.0 and ki_obs_type is None:
        st_warns = list(getattr(state, "warnings", None) or [])
        if "KI_OBS_TYPE_DEFAULTED_EUROPEAN" not in st_warns:
            st_warns.append("KI_OBS_TYPE_DEFAULTED_EUROPEAN")
        setattr(state, "warnings", st_warns)
        s_warns = getattr(state.seasoned_state, "warnings", None)
        if isinstance(s_warns, list):
            if "KI_OBS_TYPE_DEFAULTED_EUROPEAN" not in s_warns:
                s_warns.append("KI_OBS_TYPE_DEFAULTED_EUROPEAN")
        elif isinstance(s_warns, tuple):
            if "KI_OBS_TYPE_DEFAULTED_EUROPEAN" not in s_warns:
                try:
                    object.__setattr__(
                        state.seasoned_state,
                        "warnings",
                        (*s_warns, "KI_OBS_TYPE_DEFAULTED_EUROPEAN"),
                    )
                except (AttributeError, TypeError):
                    pass
        elif s_warns is None:
            try:
                object.__setattr__(
                    state.seasoned_state,
                    "warnings",
                    ("KI_OBS_TYPE_DEFAULTED_EUROPEAN",),
                )
            except (AttributeError, TypeError):
                pass
    try:
        setattr(state, "_runtime_meta_tag", meta_tag)
    except (AttributeError, TypeError):
        pass
    return eff_mds_hex


def _collect_state_and_surface_warnings(state: AutocallPreprocState) -> list[str]:
    """Collects deduplicated warnings from AutocallPreprocState, SeasonedScheduleState, and RegularizedVolSurface."""
    out: list[str] = []
    for src in (
        getattr(state, "warnings", None),
        getattr(state.seasoned_state, "warnings", None),
    ):
        if src:
            for w in src:
                w_s = str(w)
                if w_s not in out:
                    out.append(w_s)
    for surf in (state.vol_surfaces or {}).values():
        surf_warns = getattr(surf, "warnings", None)
        if surf_warns:
            for w in surf_warns:
                w_s = str(w)
                if w_s not in out:
                    out.append(w_s)
    return out


def _collect_quanto_fx_ids(state: AutocallPreprocState) -> list[str]:
    """Returns ordered quanto FX surface identifiers (e.g. `CNHRUB`, `USDRUB`) present on state."""
    fx_ids: list[str] = []
    seen: set[str] = set()
    fx_specs = getattr(state, "quanto_fx_specs", None)
    if isinstance(fx_specs, Mapping):
        for fx_id in fx_specs.keys():
            fx_s = str(fx_id).strip()
            if fx_s and fx_s not in seen:
                seen.add(fx_s)
                fx_ids.append(fx_s)
    quanto_meta = getattr(state, "quanto_meta", None) or {}
    if isinstance(quanto_meta, Mapping):
        for qm in quanto_meta.values():
            if isinstance(qm, Mapping):
                fx_s = str(qm.get("fx_id", "")).strip()
                if fx_s and fx_s not in seen:
                    seen.add(fx_s)
                    fx_ids.append(fx_s)
    return fx_ids


def _merge_fx_delta_and_gamma(
    greeks_out: AutocallGreeksResult | None,
    state: AutocallPreprocState,
) -> tuple[
    Optional[Dict[str, float]],
    Optional[Union[Dict[str, Dict[str, float]], Dict[str, float], List[List[float]]]],
]:
    """Emits `0.0` quanto FX Delta/Gamma keys (`CNHRUB`, `USDRUB`, etc.) when absent."""
    if greeks_out is None:
        return None, None
    fx_ids = _collect_quanto_fx_ids(state)
    delta_out: Optional[Dict[str, float]] = (
        dict(greeks_out.delta) if greeks_out.delta is not None else None
    )
    if delta_out is not None and fx_ids:
        for fx_id in fx_ids:
            delta_out.setdefault(fx_id, 0.0)

    gamma_raw = greeks_out.gamma
    gamma_out: Optional[Union[Dict[str, Dict[str, float]], Dict[str, float], List[List[float]]]]
    if gamma_raw is None:
        gamma_out = None
    elif isinstance(gamma_raw, Mapping):
        is_nested = any(isinstance(v, Mapping) for v in gamma_raw.values())
        if is_nested:
            nested_gamma: dict[str, dict[str, float]] = {
                str(k): (
                    {str(ik): float(iv) for ik, iv in v.items()}
                    if isinstance(v, Mapping)
                    else {}
                )
                for k, v in gamma_raw.items()
            }
            for fx_id in fx_ids:
                if fx_id not in nested_gamma:
                    for row in nested_gamma.values():
                        row.setdefault(fx_id, 0.0)
                    nested_gamma[fx_id] = {k: 0.0 for k in nested_gamma}
                    nested_gamma[fx_id][fx_id] = 0.0
            gamma_out = nested_gamma
        else:
            flat_gamma: dict[str, float] = {
                str(k): float(cast(Any, v))
                for k, v in gamma_raw.items()
                if not isinstance(v, Mapping)
            }
            for fx_id in fx_ids:
                flat_gamma.setdefault(fx_id, 0.0)
            gamma_out = flat_gamma
    else:
        gamma_out = gamma_raw
    return delta_out, gamma_out


def _merge_fx_vega_and_cega(
    greeks_out: AutocallGreeksResult | None,
    state: AutocallPreprocState,
    req_model: EQAutocallableTradeRequest,
) -> tuple[dict[str, list[list[float]]] | None, dict[str, dict[str, float]] | None]:
    """Merges FX Vega (projected onto 2D surface grid) and Equity-FX Cega into Result.Vega and Result.Cega."""
    if greeks_out is None:
        return None, None

    vega_merged: dict[str, list[list[float]]] | None = (
        dict(greeks_out.vega) if greeks_out.vega is not None else None
    )
    if vega_merged is not None and greeks_out.fx_vega:
        quanto_meta = getattr(state, "quanto_meta", None) or {}
        fx_specs_map = getattr(state, "quanto_fx_specs", None) or req_model.Model.MarketDataSet.VolatilitySurfaces
        tau_t = state.tau_obs.detach()
        if tau_t.device.type != "cpu":
            tau_t = tau_t.cpu()
        tau_obs_np = np.asarray(tau_t.numpy(), dtype=np.float64)
        as_of_dt = req_model.Model.MarketDataSet.AsOfDate
        fx_accum: dict[str, npt.NDArray[np.float64]] = {}
        fx_grid_memo: dict[str, tuple[int, int, int, npt.NDArray[np.float64]] | None] = {}

        for ticker, vec_1d in greeks_out.fx_vega.items():
            qm = quanto_meta.get(ticker) or {}
            fx_id = str(qm.get("fx_id", ""))
            if not fx_id or fx_id in vega_merged:
                continue
            vec_arr = np.asarray(vec_1d, dtype=np.float64).reshape(-1)
            if fx_id not in fx_grid_memo:
                fx_spec = fx_specs_map.get(fx_id) if isinstance(fx_specs_map, Mapping) else None
                grid_info: tuple[int, int, int, npt.NDArray[np.float64]] | None = None
                if fx_spec is not None and getattr(fx_spec, "VolatilityQuotes", None) is not None:
                    q_arr = np.asarray(fx_spec.VolatilityQuotes, dtype=np.float64)
                    if q_arr.ndim == 2 and q_arr.shape[0] > 0 and q_arr.shape[1] > 0:
                        n_ten, n_cols = q_arr.shape
                        deltas_raw = getattr(fx_spec, "Deltas", None) or ()
                        atm_idx = n_cols // 2
                        for d_i, d_tok in enumerate(deltas_raw):
                            if str(d_tok).strip().upper() in ("ATM", "ATMF", "DNS", "0.5", "50D"):
                                atm_idx = d_i
                                break
                        tenors_raw = getattr(fx_spec, "Tenors", None)
                        exp_raw = getattr(fx_spec, "ExpirationDates", None)
                        if tenors_raw is not None and len(tenors_raw) == n_ten:
                            t_grid = np.array([_parse_fx_tenor_to_years(t) for t in tenors_raw], dtype=np.float64)
                        elif exp_raw is not None and len(exp_raw) == n_ten:
                            t_grid = np.array(
                                [max(year_fraction(as_of_dt, e_dt, "Act365"), 1e-6) for e_dt in exp_raw],
                                dtype=np.float64,
                            )
                        else:
                            t_grid = np.linspace(0.25, max(float(tau_obs_np[-1]) if tau_obs_np.size else 1.0, 0.25), n_ten)
                        grid_info = (n_ten, n_cols, atm_idx, t_grid)
                fx_grid_memo[fx_id] = grid_info

            g_info = fx_grid_memo[fx_id]
            if g_info is not None:
                n_ten, n_cols, atm_idx, t_grid = g_info
                sort_order = np.argsort(t_grid)
                t_sorted = t_grid[sort_order]
                grid_2d = fx_accum.get(fx_id)
                if grid_2d is None:
                    grid_2d = np.zeros((n_ten, n_cols), dtype=np.float64)
                    fx_accum[fx_id] = grid_2d

                for m_i in range(min(len(vec_arr), len(tau_obs_np))):
                    g_m = float(vec_arr[m_i])
                    t_m = float(tau_obs_np[m_i])
                    if n_ten == 1 or t_m <= float(t_sorted[0]):
                        grid_2d[int(sort_order[0]), atm_idx] += g_m
                    elif t_m >= float(t_sorted[-1]):
                        grid_2d[int(sort_order[-1]), atm_idx] += g_m
                    else:
                        idx_hi = int(np.searchsorted(t_sorted, t_m, side="right"))
                        idx_lo = max(idx_hi - 1, 0)
                        t_lo = float(t_sorted[idx_lo])
                        t_hi = float(t_sorted[idx_hi])
                        w_hi = (t_m - t_lo) / max(t_hi - t_lo, 1e-12)
                        grid_2d[int(sort_order[idx_lo]), atm_idx] += (1.0 - w_hi) * g_m
                        grid_2d[int(sort_order[idx_hi]), atm_idx] += w_hi * g_m
                continue
            fx_accum[fx_id] = fx_accum.get(fx_id, np.zeros((1, len(vec_arr)), dtype=np.float64)) + vec_arr[None, :]

        for fx_id, mat_2d in fx_accum.items():
            vega_merged[fx_id] = mat_2d.tolist()

    cega_merged: dict[str, dict[str, float]] | None = None
    if greeks_out.cega is not None:
        base_cega: dict[str, dict[str, float]] = {
            k: dict(v) for k, v in greeks_out.cega.items()
        }
        if greeks_out.fx_cega:
            for ticker, fx_map in greeks_out.fx_cega.items():
                if not isinstance(fx_map, Mapping):
                    continue
                base_cega.setdefault(ticker, {})
                for fx_id, c_val in fx_map.items():
                    val_f = float(c_val)
                    base_cega[ticker][fx_id] = val_f
                    fx_row = base_cega.setdefault(fx_id, {})
                    fx_row[ticker] = val_f
                    fx_row.setdefault(fx_id, 0.0)
        cega_merged = _CegaDict(base_cega)

    return vega_merged, cega_merged


def _compute_qmc_crn_if_available(
    state: AutocallPreprocState,
    *,
    queries: Sequence[str],
    n_paths: int | None = None,
    request: Any = None,
    device: torch.device | None = None,
    theta_roll_days: int = 1,
    theta_convention: str = "murex_bizday_forward_fixed",
) -> AutocallGreeksResult | None:
    """Computes Dupire QMC CRN Greeks under `CanonicalTier3Config` (0.00 bps drift); returns None on failure."""
    crn_fn = getattr(_greeks_mod, "compute_qmc_crn_greeks", None)
    if callable(crn_fn):
        try:
            if crn_fn is _QMC_CRN_FN:
                params = _QMC_CRN_PARAMS
                accepts_varkw = _QMC_CRN_ACCEPTS_VARKW
            else:
                crn_sig = inspect.signature(crn_fn)
                params = frozenset(crn_sig.parameters.keys())
                accepts_varkw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in crn_sig.parameters.values()
                )
            eff_device = (
                device
                if device is not None
                else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            )
            canon_cfg: CanonicalTier3Config = resolve_canonical_tier3_config(
                int(getattr(state, "d_assets", 1)),
                int(getattr(state, "m_rem", 1)),
                device_type=("cuda" if eff_device.type == "cuda" else "cpu"),
                n_fallback_paths=n_paths,
            )
            crn_kwargs: dict[str, Any] = {"queries": queries}
            cfg_obj: Any = None
            try:
                aad_mod = importlib.import_module("deepvol.models.autocall.qmc_aad_greeks")
                aad_cfg_cls = getattr(aad_mod, "QmcAadGreeksConfig", None)
                if aad_cfg_cls is not None:
                    cfg_obj = aad_cfg_cls(
                        n_paths=canon_cfg.n_paths,
                        n_replicates=canon_cfg.n_replicates,
                        n_substeps=canon_cfg.n_substeps,
                        lv_grid_nt=canon_cfg.lv_grid_nt,
                        lv_grid_ny=canon_cfg.lv_grid_ny,
                        cv_mode="multifidelity_gbm",
                        n_cheap_mf=canon_cfg.n_cheap_mf,
                        seed=canon_cfg.seed,
                    )
            except ImportError:
                cfg_obj = None
            if cfg_obj is None:
                for mod_name in (
                    "deepvol.models.autocall.dupire_qmc_gpu",
                    "deepvol.models.autocall.dupire_qmc_solver",
                ):
                    try:
                        qmc_mod = importlib.import_module(mod_name)
                        cfg_cls = getattr(qmc_mod, "DupireQMCConfig", None)
                        if cfg_cls is not None:
                            cfg_obj = cfg_cls(
                                n_paths=canon_cfg.n_paths,
                                n_replicates=canon_cfg.n_replicates,
                                n_substeps=canon_cfg.n_substeps,
                                cv_mode=canon_cfg.cv_mode,
                                seed=canon_cfg.seed,
                                lv_grid_nt=canon_cfg.lv_grid_nt,
                                lv_grid_ny=canon_cfg.lv_grid_ny,
                                adaptive_substeps=canon_cfg.adaptive_substeps,
                            )
                            break
                    except ImportError:
                        continue
            crn_kwargs["config"] = cfg_obj if cfg_obj is not None else canon_cfg
            if request is not None and (accepts_varkw or "request" in params):
                crn_kwargs["request"] = request
            if accepts_varkw or "device" in params:
                crn_kwargs["device"] = eff_device
            if accepts_varkw or "theta_roll_days" in params:
                crn_kwargs["theta_roll_days"] = int(theta_roll_days)
            if accepts_varkw or "theta_convention" in params:
                crn_kwargs["theta_convention"] = str(theta_convention)
            return crn_fn(state, **crn_kwargs)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning(
                "QMC CRN Greeks failed (%s); falling back to Tier-1 Analytical Proxy AD.",
                exc,
            )
            return None
    return None


def _mds_digest_hex(mds: Any) -> str:
    """Computes a live 16-byte blake2b hex digest for `MarketDataSetSpec`."""
    key_obj = digest_market_bytes("mds", mds)
    if key_obj is not None:
        return key_obj.digest_hex
    raw = orjson.dumps(
        mds.model_dump(mode="json", by_alias=True),
        option=orjson.OPT_SORT_KEYS,
    )
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


_CASHFLOW_ROW_FIELDS_SET: frozenset[str] = frozenset(CashFlowRow.model_fields.keys())


def _build_cashflows_fast(cf_items: Sequence[Any]) -> list[CashFlowRow]:
    # Bypass per-row Pydantic model_construct dict-pop/set-add loop on long schedules (e.g. 180+ rows).
    fields_copy = _CASHFLOW_ROW_FIELDS_SET.copy
    desc_get = _DESCRIPTION_TO_SHORT_CODE.get
    isfinite = math.isfinite
    row_cls = CashFlowRow
    new_fn = row_cls.__new__
    setattr_fn = object.__setattr__
    out: list[CashFlowRow] = []
    append_fn = out.append
    for cf in cf_items:
        raw_desc = cf.Description
        desc_s = raw_desc if type(raw_desc) is str else str(raw_desc)
        raw_amt = cf.Amount
        amt_f = raw_amt if type(raw_amt) is float else float(raw_amt)
        raw_se = cf.StandardError
        se_f = raw_se if type(raw_se) is float else float(raw_se)
        if not (isfinite(amt_f) and isfinite(se_f)):
            raise ValueError(
                f"Non-finite CashFlowRow output: Amount={amt_f!r}, StandardError={se_f!r}"
            )
        raw_ccy = cf.Currency
        raw_leg = cf.Leg
        raw_stat = cf.StatusText
        row = new_fn(row_cls)
        setattr_fn(
            row,
            "__dict__",
            {
                "Description": desc_s,
                "Date": cf.Date,
                "Amount": amt_f,
                "StandardError": se_f,
                "Currency": raw_ccy if type(raw_ccy) is str else str(raw_ccy),
                "Leg": raw_leg if type(raw_leg) is int else int(raw_leg),
                "StatusText": raw_stat if type(raw_stat) is str else str(raw_stat),
                "ShortCode": desc_get(desc_s),
            },
        )
        setattr_fn(row, "__pydantic_fields_set__", fields_copy())
        setattr_fn(row, "__pydantic_extra__", None)
        setattr_fn(row, "__pydantic_private__", None)
        append_fn(row)
    return out


def _assert_finite_nested_floats(val: Any, label: str) -> None:
    if val is None or isinstance(val, (str, bool)):
        return
    t_val = type(val)
    isfinite = math.isfinite
    if t_val is float:
        if not isfinite(val):
            raise ValueError(f"Non-finite runtime output in {label}: {val!r}")
        return
    if t_val is int:
        return
    if isinstance(val, (int, float, np.number)):
        if not isfinite(float(val)):
            raise ValueError(f"Non-finite runtime output in {label}: {val!r}")
        return
    if isinstance(val, np.ndarray):
        if val.dtype.kind in ("f", "c", "i", "u") and not bool(np.all(np.isfinite(val))):
            raise ValueError(f"Non-finite ndarray runtime output in {label}")
        return
    if isinstance(val, (list, tuple)):
        if not val:
            return
        if type(val[0]) in (float, int):
            try:
                if isfinite(sum(val)):
                    return
            except (TypeError, OverflowError):
                pass
        for idx, elem in enumerate(val):
            _assert_finite_nested_floats(elem, f"{label}[{idx}]")
        return
    if isinstance(val, dict):
        if not val:
            return
        v0 = next(iter(val.values()))
        if type(v0) in (float, int):
            try:
                if isfinite(sum(val.values())):
                    return
            except (TypeError, OverflowError):
                pass
        for k, v in val.items():
            _assert_finite_nested_floats(v, f"{label}.{k}")
        return
    if isinstance(val, Mapping):
        for k, v in val.items():
            _assert_finite_nested_floats(v, f"{label}.{k}")


def _map_greeks_engine_label(raw_label: str | None) -> str:
    """Maps internal Greek engine tags to canonical runtime telemetry names."""
    tok = str(raw_label or "").strip().lower()
    if tok == "tier1_ad":
        return "Tier1_AnalyticalProxy_AD"
    if tok.startswith("qmc"):
        return "Tier4_DupireQMC_CRN"
    return "Tier2_SingleVJP_AD"


def _assemble_runtime_response(
    *,
    product_value: float,
    reporting_currency: str,
    standard_error: float,
    cash_to_settle_today: float,
    cash_flows: list[CashFlowRow],
    delta: Optional[Dict[str, float]],
    gamma: Optional[Union[Dict[str, Dict[str, float]], Dict[str, float], List[List[float]]]],
    cash_gamma: Optional[Union[float, List[List[float]]]],
    vega: Optional[Dict[str, List[List[float]]]],
    rho: Optional[Dict[str, List[float]]],
    repo_rho: Optional[Dict[str, List[float]]],
    rho_unit_convention: Literal["quote", "decimal"] = "quote",
    dividends_sensitivities: Optional[Dict[str, List[float]]],
    theta: Optional[float],
    cega: Optional[Union[Dict[str, Dict[str, float]], Dict[str, float], List[List[float]]]],
    trade_info: Optional[str],
    engine_used: str,
    fallback_triggered: bool,
    fallback_reason: Optional[str],
    fallback_reasons: List[str],
    stripped_legacy_fields: List[str],
    option: Optional[float],
    notional: Optional[float],
    coupon: Optional[float],
    funding_leg: Optional[float],
    audit_record: Optional[Dict[str, Any]],
    latency_ms: float,
) -> EQAutocallRuntimeResponse:
    """Constructs `EQAutocallRuntimeResponse` via `model_construct` with all 27 fields set."""
    for s_name, s_val in (
        ("ProductValue", product_value),
        ("StandardError", standard_error),
        ("CashToSettleToday", cash_to_settle_today),
        ("Option", option),
        ("Notional", notional),
        ("Coupon", coupon),
        ("FundingLeg", funding_leg),
        ("Theta", theta),
    ):
        if s_val is not None and not math.isfinite(float(s_val)):
            raise ValueError(f"Non-finite scalar runtime output {s_name}={s_val!r}")
    for row_idx, cf_row in enumerate(cash_flows):
        if not (math.isfinite(float(cf_row.Amount)) and math.isfinite(float(cf_row.StandardError))):
            raise ValueError(f"Non-finite CashFlowRow[{row_idx}] output")
    for g_name, g_val in (
        ("Delta", delta),
        ("Gamma", gamma),
        ("MultivariateAggregatedCashGamma", cash_gamma),
        ("Vega", vega),
        ("Rho", rho),
        ("RepoRho", repo_rho),
        ("DividendsSensitivities", dividends_sensitivities),
        ("Cega", cega),
        ("audit_record", audit_record),
    ):
        if g_val is not None:
            _assert_finite_nested_floats(g_val, g_name)
    cega_val = (
        _CegaDict(cega)
        if (isinstance(cega, Mapping) and not isinstance(cega, _CegaDict))
        else cega
    )
    return EQAutocallRuntimeResponse.model_construct(
        ProductValue=product_value,
        ReportingCurrency=reporting_currency,
        StandardError=standard_error,
        CashToSettleToday=cash_to_settle_today,
        CashFlows=cash_flows,
        Delta=delta,
        Gamma=gamma,
        MultivariateAggregatedCashGamma=cash_gamma,
        Vega=vega,
        Rho=rho,
        RepoRho=repo_rho,
        rho_unit_convention=rho_unit_convention,
        DividendsSensitivities=dividends_sensitivities,
        Theta=theta,
        Cega=cega_val,
        TradeInfo=trade_info,
        engine_used=engine_used,
        fallback_triggered=fallback_triggered,
        fallback_reason=fallback_reason,
        fallback_reasons=fallback_reasons,
        stripped_legacy_fields=stripped_legacy_fields,
        Option=option,
        Notional=notional,
        Coupon=coupon,
        FundingLeg=funding_leg,
        audit_record=audit_record,
        latency_ms=latency_ms,
    )


def _market_dedup_key(
    req: EQAutocallableTradeRequest,
    *,
    mds_digest_hex: str | None = None,
) -> str:
    """Computes a stable digest for `Model` (`MarketDataSet` + `FinancingCurveIdentifiers` + legacy fields)."""
    model = req.Model
    mds_hex = mds_digest_hex if mds_digest_hex is not None else _mds_digest_hex(model.MarketDataSet)
    fin_ids = model.FinancingCurveIdentifiers
    fin_part = ",".join(str(x) for x in fin_ids) if fin_ids is not None else "<NONE>"
    leg_fields = getattr(model, "stripped_legacy_fields", None)
    leg_part = ",".join(str(x) for x in leg_fields) if leg_fields else ""
    raw_key = f"{mds_hex}|fin:{fin_part}|leg:{leg_part}".encode("utf-8")
    return hashlib.blake2b(raw_key, digest_size=16).hexdigest()


def _compute_w_fund_per_period(state: AutocallPreprocState, m_max: int) -> torch.Tensor:
    """Computes survival-weighted funding-leg cashflow coefficients per period `[0..m_max-1]`."""
    out = torch.zeros(m_max, dtype=torch.float64)
    m_rem = int(state.m_rem)
    if state.funding_leg is None or m_rem <= 0 or state.already_autocalled:
        return out
    f_leg = state.funding_leg
    cached_w_pair = f_leg.get("_cached_w_fund_per_period")
    if cached_w_pair is not None:
        w_cached = torch.as_tensor(cached_w_pair[1], dtype=torch.float64).reshape(-1)
        n_copy = min(m_rem, int(w_cached.numel()), m_max)
        out[:n_copy] = w_cached[:n_copy]
        return out

    f_rid = f_leg.get("rate_identifier")
    f_curve = f_leg.get("funding_curve")
    f_curve_id = f_leg.get("funding_curve_id")
    disc_id = state.discount_curve_id
    as_of_dt = state.seasoned_state.as_of_date
    start_dt = state.seasoned_state.start_date
    f_dcc = state.seasoned_state.day_count_convention
    _, df_fin_fn = build_discount_rate_evaluator(state.discount_curve, as_of_dt)
    same_curve = (
        f_curve is None
        or f_curve is state.discount_curve
        or (f_curve_id is not None and disc_id is not None and f_curve_id == disc_id)
    )
    if f_rid is not None and not same_curve:
        _, df_fwd_fn = build_discount_rate_evaluator(f_curve, as_of_dt)
    else:
        df_fwd_fn = df_fin_fn

    raw_f_spread = f_leg.get("funding_spread", 0.0)
    f_spread = 0.0 if raw_f_spread is None else float(raw_f_spread)
    f_dates = f_leg.get("payment_dates", ())
    f_fixings = f_leg.get("funding_fixings", ())
    prev_f_dt = start_dt
    for f_dt in f_dates:
        if f_dt <= as_of_dt:
            prev_f_dt = f_dt
            continue
        alpha_j = year_fraction(prev_f_dt, f_dt, f_dcc)
        df_curr = float(df_fin_fn(f_dt))
        if f_rid is None:
            r_fwd_j = 0.0
        else:
            fixed_reset = _lookup_funding_fixing(f_fixings, prev_f_dt) if prev_f_dt < as_of_dt else None
            if fixed_reset is not None:
                r_fwd_j = float(fixed_reset)
            elif prev_f_dt < as_of_dt:
                alpha_rem = max(year_fraction(as_of_dt, f_dt, f_dcc), 1e-8)
                df_curr_fwd = max(float(df_fwd_fn(f_dt)), 1e-12)
                r_fwd_j = (1.0 / df_curr_fwd - 1.0) / alpha_rem
            else:
                df_prev_fwd = float(df_fwd_fn(prev_f_dt))
                df_curr_fwd = max(float(df_fwd_fn(f_dt)), 1e-12)
                r_fwd_j = (df_prev_fwd / df_curr_fwd - 1.0) / max(alpha_j, 1e-8)
        m_idx = m_rem - 1
        for m in range(m_rem):
            if state.seasoned_state.future_pay_dates[m] >= f_dt:
                m_idx = m
                break
        if m_idx < m_max:
            out[m_idx] = out[m_idx] + df_curr * (r_fwd_j + f_spread) * alpha_j
        prev_f_dt = f_dt
    return out


class EQAutocallRuntime:
    """Unified 4-Tier pricing, Greek sensitivity, Murex DAG, and Apache Arrow IPC runtime for Equity Autocallables."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
        stats_path: str | Path | None = DEFAULT_GUARDIAN_STATS_PATH,
        device: str | torch.device | None = None,
        d_max: int = D_MAX_DEFAULT,
        m_max: int = M_MAX_DEFAULT,
        guardian: EQAutocallModelGuardian | None = None,
        surrogate: MaskedEQAutocallSurrogate | None = None,
        n_fallback_paths: int | None = None,
        theta_convention: str = "murex_bizday_forward_fixed",
        repair_arbitrage: bool = False,
        strict_checkpoint: bool = False,
        enable_compiled_runner: bool = True,
    ) -> None:
        if os.environ.get("DEEPVOL_TIER1_CPP_LIB", "").strip().lower() != "numpy":
            backend = get_tier1_backend()
            if getattr(backend, "name", None) != "cpp23":
                raise RuntimeError(
                    f"Expected Tier-1 native backend 'cpp23' at runtime startup, got {getattr(backend, 'name', None)!r}. "
                    "Set DEEPVOL_TIER1_CPP_LIB=numpy only when explicitly running without C++23."
                )
        self._lock = threading.RLock()
        self.d_max = int(d_max)
        self.m_max = int(m_max)
        self.n_fallback_paths: int | None = (
            int(n_fallback_paths) if n_fallback_paths is not None else None
        )
        self.theta_convention: str = str(theta_convention)
        self.repair_arbitrage: bool = bool(repair_arbitrage)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.stats_path = Path(stats_path) if stats_path is not None else None
        self.checkpoint_metrics: dict[str, Any] = {}
        self.checkpoint_loaded: bool = False
        self.checkpoint_error: str | None = None
        self.guardian_stats_loaded: bool = False
        self.guardian_stats_error: str | None = None
        self.model_version: str = "eq-autocall-egno-zero-init"

        if surrogate is not None:
            self.surrogate = surrogate.to(device=self.device, dtype=torch.float32).eval()
            self.model_version = "eq-autocall-egno-custom"
            self.checkpoint_loaded = True
        elif self.checkpoint_path is not None and self.checkpoint_path.is_file():
            try:
                loaded_model, loaded_metrics = _load_surrogate_from_ckpt(
                    self.checkpoint_path,
                    device=self.device,
                    dtype=torch.float32,
                )
                self.surrogate = loaded_model
                self.checkpoint_metrics = loaded_metrics
                self.model_version = f"eq-autocall-egno-{self.checkpoint_path.stem}"
                self.checkpoint_loaded = True
            except Exception as exc:
                if strict_checkpoint:
                    raise RuntimeError(
                        f"Failed to load strict checkpoint {self.checkpoint_path}: {exc}"
                    ) from exc
                logger.warning(
                    "Failed to load checkpoint %s (%s); falling back to zero_init_residual_heads.",
                    self.checkpoint_path,
                    exc,
                )
                self.checkpoint_error = str(exc)
                self.surrogate = MaskedEQAutocallSurrogate(
                    d_max=self.d_max,
                    m_max=self.m_max,
                    d_node=16,
                    d_global=16,
                ).to(device=self.device, dtype=torch.float32)
                self.surrogate.zero_init_residual_heads(std=0.0)
                self.surrogate.eval()
        else:
            missing_msg = f"Checkpoint file not found: {self.checkpoint_path}"
            if strict_checkpoint:
                raise FileNotFoundError(missing_msg)
            self.checkpoint_error = missing_msg
            self.surrogate = MaskedEQAutocallSurrogate(
                d_max=self.d_max,
                m_max=self.m_max,
                d_node=16,
                d_global=16,
            ).to(device=self.device, dtype=torch.float32)
            self.surrogate.zero_init_residual_heads(std=0.0)
            self.surrogate.eval()

        # Maintain a CPU mirror of the surrogate for zero-transfer single-trade CPU preprocessing & Greeks
        if self.device.type == "cuda":
            self._cpu_surrogate = _build_cpu_surrogate_mirror(self.surrogate)
        else:
            self._cpu_surrogate = self.surrogate
        self._proxy_surrogate: MaskedEQAutocallSurrogate | None = (
            self.surrogate if self.surrogate.is_zero_residual() else None
        )

        self._compiled_runner: Any = None
        if (
            enable_compiled_runner
            and self.device.type == "cuda"
            and _CompiledSurrogateRunner is not None
        ):
            try:
                self._compiled_runner = _CompiledSurrogateRunner(
                    self.surrogate,
                    use_tf32=False,
                    enable_cudagraphs=True,
                    enable_inductor=False,
                )
            except Exception as exc:
                logger.warning("CompiledSurrogateRunner disabled due to init error: %s", exc)
                self._compiled_runner = None

        self.guardian = (
            guardian
            if guardian is not None
            else EQAutocallModelGuardian(
                d_max=self.d_max,
                m_max=self.m_max,
                record_drift_by_default=False,
                psi_hard_circuit_breaker=False,
            )
        )
        if self.stats_path is not None and self.stats_path.is_file():
            try:
                self.guardian.load_calibrated_stats(self.stats_path)
                self.guardian_stats_loaded = True
            except Exception as exc:
                self.guardian_stats_error = str(exc)
                logger.warning(
                    "Failed to load Guardian calibrated stats %s: %s",
                    self.stats_path,
                    exc,
                )
        elif self.stats_path is not None:
            self.guardian_stats_error = f"Guardian stats file not found: {self.stats_path}"

    def _parse_request(
        self,
        req: EQAutocallableTradeRequest | str | bytes | Path | Mapping[str, Any],
    ) -> tuple[EQAutocallableTradeRequest, bool]:
        """Normalizes `req` into a validated `EQAutocallableTradeRequest` and returns `(req_model, is_murex_dag)`."""
        if isinstance(req, EQAutocallableTradeRequest):
            return _strip_untrusted_digests(req), False

        if isinstance(req, Path):
            try:
                raw_bytes_in = req.read_bytes()
            except OSError as exc:
                raise ValueError(f"Failed to read request file {req!r}: {exc}") from exc
            req = raw_bytes_in

        if isinstance(req, bytes):
            try:
                req = req.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Failed to decode UTF-8 request bytes: {exc}") from exc

        if isinstance(req, str):
            s = req.strip()
            raw_bytes_s = s.encode("utf-8")
            raw_digest_bytes = hashlib.blake2b(raw_bytes_s, digest_size=16).digest()
            raw_digest_hex = raw_digest_bytes.hex()
            p_key: tuple[Any, ...] = ("str", raw_digest_bytes, len(s))
            with _PARSE_REQUEST_CACHE_LOCK:
                hit_p = _PARSE_REQUEST_CACHE.get(p_key)
                if hit_p is not None:
                    _PARSE_REQUEST_CACHE.move_to_end(p_key)
                    cached_req, cached_is_dag, _ = hit_p
                    return _clone_cached_request(cached_req), cached_is_dag
            if s.startswith("{") and "\nclass:" not in s and not _MUREX_DAG_MARKER_RE.search(s):
                json_obj: Any = None
                parsed_json_ok = False
                try:
                    json_obj = orjson.loads(s)
                    parsed_json_ok = True
                except (orjson.JSONDecodeError, ValueError):
                    parsed_json_ok = False
                if parsed_json_ok and isinstance(json_obj, dict) and "Product" in json_obj and "Model" in json_obj:
                    json_obj.pop("Result", None)
                    validated = EQAutocallableTradeRequest.model_validate(json_obj)
                    object.__setattr__(validated, "_raw_payload_digest_hex", raw_digest_hex)
                    object.__setattr__(validated, "_immutable_cache_source", validated)
                    with _PARSE_REQUEST_CACHE_LOCK:
                        _PARSE_REQUEST_CACHE[p_key] = (validated, False, ())
                        _PARSE_REQUEST_CACHE.move_to_end(p_key)
                        while len(_PARSE_REQUEST_CACHE) > _PARSE_REQUEST_CACHE_MAXSIZE:
                            _PARSE_REQUEST_CACHE.popitem(last=False)
                    return _clone_cached_request(validated), False
            parsed_req, _ = parse_murex_autocall_dag(s, allow_file_path=False)
            object.__setattr__(parsed_req, "_raw_payload_digest_hex", raw_digest_hex)
            object.__setattr__(parsed_req, "_immutable_cache_source", parsed_req)
            with _PARSE_REQUEST_CACHE_LOCK:
                _PARSE_REQUEST_CACHE[p_key] = (parsed_req, True, ())
                _PARSE_REQUEST_CACHE.move_to_end(p_key)
                while len(_PARSE_REQUEST_CACHE) > _PARSE_REQUEST_CACHE_MAXSIZE:
                    _PARSE_REQUEST_CACHE.popitem(last=False)
            return _clone_cached_request(parsed_req), True

        if isinstance(req, (dict, Mapping)):
            if "dag_text" in req:
                raw_dag = req["dag_text"]
                if not isinstance(raw_dag, str):
                    raise ValueError("'dag_text' must be a string")
                parsed_req, _ = self._parse_request(raw_dag)
                return parsed_req, True
            is_direct_json = "Product" in req and "Model" in req
            if is_direct_json and "Result" in req:
                payload_obj: dict[str, Any] = {k: v for k, v in req.items() if k != "Result"}
            elif type(req) is dict:
                payload_obj = req
            else:
                payload_obj = dict(req)

            d_key: tuple[Any, ...] | None = None
            d_digest_hex: str | None = None
            d_bytes = b""
            try:
                d_bytes = orjson.dumps(payload_obj, option=orjson.OPT_SORT_KEYS)
                h_dict = hashlib.blake2b(
                    b"json_dict\x00" if is_direct_json else b"dag_dict\x00",
                    digest_size=16,
                )
                h_dict.update(d_bytes)
                d_digest_bytes = h_dict.digest()
                d_digest_hex = d_digest_bytes.hex()
                d_key = (
                    "json_dict" if is_direct_json else "dag_dict",
                    d_digest_bytes,
                    len(d_bytes),
                )
            except (TypeError, ValueError, OverflowError):
                d_key = None
                d_digest_hex = None

            if d_key is not None:
                with _PARSE_REQUEST_CACHE_LOCK:
                    hit_d = _PARSE_REQUEST_CACHE.get(d_key)
                    if hit_d is not None:
                        cached_req, cached_is_dag, null_paths = hit_d
                        if not null_paths or _check_no_nan_at_null_paths(payload_obj, null_paths):
                            _PARSE_REQUEST_CACHE.move_to_end(d_key)
                            return _clone_cached_request(cached_req), cached_is_dag

            if is_direct_json:
                validated = EQAutocallableTradeRequest.model_validate(payload_obj)
                is_dag_res = False
            else:
                validated, _ = parse_murex_autocall_dag(payload_obj, allow_file_path=False)
                is_dag_res = True

            if d_key is not None:
                entry_null_paths = _collect_null_paths(payload_obj) if b"null" in d_bytes else ()
                if entry_null_paths is not None:
                    if d_digest_hex is not None:
                        object.__setattr__(validated, "_raw_payload_digest_hex", d_digest_hex)
                        object.__setattr__(validated, "_immutable_cache_source", validated)
                    with _PARSE_REQUEST_CACHE_LOCK:
                        _PARSE_REQUEST_CACHE[d_key] = (validated, is_dag_res, entry_null_paths)
                        _PARSE_REQUEST_CACHE.move_to_end(d_key)
                        while len(_PARSE_REQUEST_CACHE) > _PARSE_REQUEST_CACHE_MAXSIZE:
                            _PARSE_REQUEST_CACHE.popitem(last=False)
            return _clone_cached_request(validated), is_dag_res

        raise TypeError(f"Unsupported request type: {type(req).__name__}")

    def _predict_surrogate_compiled(self, state: AutocallPreprocState) -> AutocallSurrogateOutput:
        """Runs CUDA `CompiledSurrogateRunner` residual inference and decodes to CPU."""
        assert self._compiled_runner is not None
        expected_d_node = int(getattr(self.surrogate, "d_node", 16))
        expected_d_global = int(getattr(self.surrogate, "d_global", 16))
        nf_raw = state.node_feats
        ef_raw = state.edge_feats
        am_raw = state.asset_mask
        sf_raw = state.sched_feats
        sm_raw = state.sched_mask
        gf_raw = state.global_feats
        d_max = int(nf_raw.shape[1])
        m_max = int(sf_raw.shape[1])
        d_act = min(max(int(state.d_assets), 1), d_max)
        m_act = min(max(int(state.m_rem), 1), m_max)
        tensor_sig = tuple(
            (
                t.data_ptr(),
                -1 if t.is_inference() else int(t._version),
                tuple(t.shape),
                str(t.dtype),
                str(t.device),
            )
            for t in (nf_raw, ef_raw, am_raw, sf_raw, sm_raw, gf_raw)
        )
        cache_sig = (tensor_sig, expected_d_node, expected_d_global, d_act, m_act)
        cached_pack = getattr(state, "_compiled_packed_cpu_cache", None)
        if isinstance(cached_pack, tuple) and len(cached_pack) == 14 and cached_pack[0] == cache_sig:
            (
                _,
                packed_cpu_in,
                n1,
                n2,
                n3,
                n4,
                n5,
                e_dim,
                s_dim,
                sched_feats_t2,
                sched_mask_t2,
                gf_t_t2,
                b_in,
                m_max,
            ) = cached_pack
        else:
            t2 = state.to_tier2_float32_tensors()
            nf_t = t2["node_feats"]
            gf_t_t2 = t2["global_feats"]
            if int(nf_t.shape[2]) < expected_d_node:
                nf_t = torch.nn.functional.pad(nf_t, (0, expected_d_node - int(nf_t.shape[2])))
            elif int(nf_t.shape[2]) > expected_d_node:
                nf_t = nf_t[:, :, :expected_d_node]
            if int(gf_t_t2.shape[1]) < expected_d_global:
                gf_t_t2 = torch.nn.functional.pad(gf_t_t2, (0, expected_d_global - int(gf_t_t2.shape[1])))
            elif int(gf_t_t2.shape[1]) > expected_d_global:
                gf_t_t2 = gf_t_t2[:, :expected_d_global]
            sched_feats_t2 = t2["sched_feats"]
            sched_mask_t2 = t2["sched_mask"]
            e_dim = int(t2["edge_feats"].shape[3])
            s_dim = int(sched_feats_t2.shape[2])
            b_in = int(nf_t.shape[0])
            m_max = int(sched_feats_t2.shape[1])
            with torch.no_grad():
                nf_c = nf_t[:, :d_act, :].reshape(-1)
                ef_c = t2["edge_feats"][:, :d_act, :d_act, :].reshape(-1)
                am_c = t2["asset_mask"][:, :d_act].reshape(-1)
                sf_c = sched_feats_t2[:, :m_act, :].reshape(-1)
                sm_c = sched_mask_t2[:, :m_act].reshape(-1)
                gf_c = gf_t_t2.reshape(-1)
                n1 = int(nf_c.numel())
                n2 = int(ef_c.numel())
                n3 = int(am_c.numel())
                n4 = int(sf_c.numel())
                n5 = int(sm_c.numel())
                packed_cpu_in = torch.cat([nf_c, ef_c, am_c, sf_c, sm_c, gf_c], dim=0).cpu().contiguous()
            try:
                object.__setattr__(
                    state,
                    "_compiled_packed_cpu_cache",
                    (
                        cache_sig,
                        packed_cpu_in,
                        n1,
                        n2,
                        n3,
                        n4,
                        n5,
                        e_dim,
                        s_dim,
                        sched_feats_t2,
                        sched_mask_t2,
                        gf_t_t2,
                        b_in,
                        m_max,
                    ),
                )
            except (AttributeError, TypeError):
                pass
        dev = self.device
        with torch.no_grad():
            packed_in = (
                packed_cpu_in.to(device=dev)
                if packed_cpu_in.device != dev
                else packed_cpu_in.clone()
            )
            node_f = packed_in[:n1].view(b_in, d_act, expected_d_node)
            edge_f = packed_in[n1 : n1 + n2].view(b_in, d_act, d_act, e_dim)
            amask_f = packed_in[n1 + n2 : n1 + n2 + n3].view(b_in, d_act)
            o4 = n1 + n2 + n3
            sched_f = packed_in[o4 : o4 + n4].view(b_in, m_act, s_dim)
            smask_f = packed_in[o4 + n4 : o4 + n4 + n5].view(b_in, m_act)
            glob_f = packed_in[o4 + n4 + n5 :].view(b_in, expected_d_global)
            dhaz_act, dcpn_act, dput = self._compiled_runner.predict_residuals(
                node_f, edge_f, amask_f, sched_f, smask_f, glob_f
            )
            if m_act < m_max:
                dhaz_dev = torch.nn.functional.pad(dhaz_act, (0, m_max - m_act))
                dcpn_dev = torch.nn.functional.pad(dcpn_act, (0, m_max - m_act))
            else:
                dhaz_dev = dhaz_act
                dcpn_dev = dcpn_act
            packed_cpu = torch.cat(
                [dhaz_dev, dcpn_dev, dput.reshape(dhaz_dev.shape[0], 1)],
                dim=-1,
            ).cpu()
        dhaz_cpu = packed_cpu[:, :m_max]
        dcpn_cpu = packed_cpu[:, m_max : 2 * m_max]
        dput_cpu = packed_cpu[:, 2 * m_max]
        return _decode_residuals_to_surrogate_output(
            dhaz_cpu,
            dcpn_cpu,
            dput_cpu,
            sched_feats_t2,
            sched_mask_t2,
            gf_t_t2,
        )

    def price_trade(
        self,
        req: EQAutocallableTradeRequest | str | bytes | Path | Mapping[str, Any],
        engine: str = "auto",
        *,
        include_all_greeks: bool | None = None,
        theta_roll_days: int = 1,
        theta_convention: str | None = None,
        repair_arbitrage: bool | None = None,
    ) -> EQAutocallRuntimeResponse:
        """Prices a single Multi-Asset Equity Autocallable trade and computes requested Greeks in `float64`."""
        t0 = time.perf_counter()
        eng_norm = _validate_engine(engine)
        req_model, is_dag = self._parse_request(req)
        eff_repair = self.repair_arbitrage if repair_arbitrage is None else bool(repair_arbitrage)
        eff_theta_conv = (
            self.theta_convention if theta_convention is None else str(theta_convention)
        )
        allow_cached_mds_digest = isinstance(getattr(req_model, "_raw_payload_digest_hex", None), str)
        if eff_repair:
            req_model = _repair_request_surfaces(req_model)
            allow_cached_mds_digest = False
        return self._price_parsed_trade(
            req_model,
            is_dag=is_dag,
            eng_norm=eng_norm,
            t0=t0,
            include_all_greeks=include_all_greeks,
            theta_roll_days=theta_roll_days,
            theta_convention=eff_theta_conv,
            allow_cached_mds_digest=allow_cached_mds_digest,
        )

    def _price_parsed_trade(
        self,
        req_model: EQAutocallableTradeRequest,
        *,
        is_dag: bool,
        eng_norm: str,
        t0: float,
        include_all_greeks: bool | None,
        theta_roll_days: int,
        theta_convention: str,
        allow_cached_mds_digest: bool,
    ) -> EQAutocallRuntimeResponse:
        src_req = (
            getattr(req_model, "_immutable_cache_source", None)
            if allow_cached_mds_digest
            else None
        )
        if not (
            allow_cached_mds_digest
            and src_req is not None
            and getattr(src_req, "_arb_validated", False) is True
        ):
            _validate_surfaces_for_arbitrage(req_model)
            if allow_cached_mds_digest and src_req is not None:
                try:
                    object.__setattr__(src_req, "_arb_validated", True)
                except (AttributeError, TypeError):
                    pass

        stripped_fields = list(req_model.stripped_legacy_fields)
        state: AutocallPreprocState = preprocess_eq_autocall(
            req_model,
            d_max=self.d_max,
            m_max=self.m_max,
            _fast_shallow_cache=True,
        )
        mds_digest_hex = _attach_runtime_market_metadata(
            state,
            req_model,
            mds_digest_hex=getattr(state, "_mds_digest_hex", None),
            allow_cached=allow_cached_mds_digest,
        )
        if state.baseline is None:
            state.baseline = compute_analytical_autocall_baseline(state)

        force_qmc = eng_norm == "qmc"
        use_proxy_only = eng_norm == "proxy"
        force_surrogate = eng_norm == "surrogate"

        requested_queries = [str(getattr(q, "value", q)) for q in req_model.Queries]
        compute_all_8 = is_dag if include_all_greeks is None else bool(include_all_greeks)
        if compute_all_8:
            for gq in ALL_8_GREEK_QUERIES:
                if gq not in requested_queries:
                    requested_queries.append(gq)

        greek_queries_present = [
            q for q in requested_queries if q in ALL_8_GREEK_QUERIES or q == "ConvexCega"
        ]
        greeks_extra: dict[str, Any] = {}
        if _SUPPORTS_STABILIZE_F32_SURROGATE:
            greeks_extra["stabilize_f32_surrogate"] = not force_surrogate
        if _SUPPORTS_GREEKS_REQUEST_PARAM:
            greeks_extra["request"] = req_model
        if _SUPPORTS_GREEKS_THETA_PARAM:
            greeks_extra["theta_roll_days"] = int(theta_roll_days)
        if _SUPPORTS_GREEKS_THETA_CONV_PARAM:
            greeks_extra["theta_convention"] = str(theta_convention)
        greeks_out: AutocallGreeksResult | None = None
        greeks_engine_used = "None"
        persisted_audit_rec: SR262AuditRecord | None = None
        surr_for_greeks = self.surrogate if self.device.type == "cuda" else self._cpu_surrogate

        if use_proxy_only:
            post_proxy = assemble_autocall_cashflows_and_pv(state)
            pv_val = float(post_proxy.product_value.detach().cpu().item())
            se_val = 0.0
            notional_val = float(post_proxy.notional_pv.detach().cpu().item())
            coupon_val = float(post_proxy.coupon_pv.detach().cpu().item())
            option_val = float(post_proxy.option_pv.detach().cpu().item())
            funding_val = float(-post_proxy.funding_pv.detach().cpu().item())
            ccy_val = post_proxy.reporting_currency
            cash_today_val = float(post_proxy.cash_to_settle_today)
            cf_items = post_proxy.cashflows
            engine_used_str = "Tier1_AnalyticalProxy"
            fb_triggered = False
            fb_reasons_str: list[str] = []
            audit_dict: dict[str, Any] | None = None
            active_state = state
            surrogate_for_greeks: MaskedEQAutocallSurrogate | None = None
            if len(greek_queries_present) > 0:
                greeks_engine_used = "Tier1_AnalyticalProxy_AD"
        elif force_surrogate and state.m_rem > 0 and not state.seasoned_state.already_autocalled:
            pre_passed, pre_reasons, r_psd, diag = self.guardian.evaluate_pre_inference(
                state,
                state.baseline,
                record_drift=False,
            )
            if bool(diag.get("correlation_shrunk_applied", diag.get("correlation_shrunk", False))) or (
                AutocallFallbackReason.CORRELATION_NOT_PSD in pre_reasons
            ):
                _ = self.guardian._apply_shrunk_correlation_to_state(state, r_psd)
                _attach_runtime_market_metadata(state, req_model, mds_digest_hex=mds_digest_hex)
            post_surr = None
            s_out: AutocallSurrogateOutput | None = None
            if len(greek_queries_present) > 0:
                greeks_out = compute_eq_autocall_greeks(
                    state,
                    queries=requested_queries,
                    surrogate=surr_for_greeks,
                    **greeks_extra,
                )
                post_surr = greeks_out.postproc_output
                if post_surr is None:
                    raise ValueError("Surrogate Greeks missing postproc_output under engine='surrogate'.")
                greeks_engine_used = _map_greeks_engine_label(
                    getattr(greeks_out, "greeks_engine_used", None)
                )
                s_dev = state.spots_t0.device
                p_stop_t = post_surr.p_stop.detach().to(dtype=torch.float64, device=s_dev)
                surv_t = post_surr.survival_probs.detach().to(dtype=torch.float64, device=s_dev)
                q_cpn_t = post_surr.q_cpn.detach().to(dtype=torch.float64, device=s_dev)
                l_put_t = post_surr.l_put.detach().to(dtype=torch.float64, device=s_dev)
                z_haz = torch.zeros_like(p_stop_t)
                s_out = AutocallSurrogateOutput(
                    stopping_probs=p_stop_t.to(dtype=torch.float32).unsqueeze(0),
                    survival_probs=surv_t.to(dtype=torch.float32).unsqueeze(0),
                    hazard_rates=z_haz.to(dtype=torch.float32).unsqueeze(0),
                    coupon_probs=q_cpn_t.to(dtype=torch.float32).unsqueeze(0),
                    put_loss=l_put_t.to(dtype=torch.float32).reshape(1),
                    delta_hazard_logits=z_haz.to(dtype=torch.float32).unsqueeze(0),
                    delta_coupon_logits=z_haz.to(dtype=torch.float32).unsqueeze(0),
                    delta_put_loss=torch.zeros((1,), dtype=torch.float32, device=s_dev),
                )
            else:
                if self._compiled_runner is not None and self.device.type == "cuda":
                    s_out = self._predict_surrogate_compiled(state)
                else:
                    t2_inputs = state.to_tier2_float32_tensors()
                    with torch.no_grad():
                        s_out = self._cpu_surrogate(
                            node_feats=t2_inputs["node_feats"],
                            edge_feats=t2_inputs["edge_feats"],
                            asset_mask=t2_inputs["asset_mask"],
                            sched_feats=t2_inputs["sched_feats"],
                            sched_mask=t2_inputs["sched_mask"],
                            global_feats=t2_inputs["global_feats"],
                        )
                m_rem_s = int(state.m_rem)
                s_dev = state.spots_t0.device
                p_stop_t = (
                    s_out.stopping_probs[0, :m_rem_s].to(dtype=torch.float64, device=s_dev)
                    * state.is_callable
                )
                q_cpn_t = s_out.coupon_probs[0, :m_rem_s].to(dtype=torch.float64, device=s_dev)
                l_put_t = s_out.put_loss[0].to(dtype=torch.float64, device=s_dev)
                post_surr = assemble_autocall_cashflows_and_pv(
                    state,
                    p_stop=p_stop_t,
                    q_cpn=q_cpn_t,
                    l_put=l_put_t,
                )
            post_passed, post_reasons = self.guardian.evaluate_post_inference(
                s_out,
                post_surr,
                state,
                state.baseline,
            )
            if not post_passed:
                return self._build_qmc_fallback_response(
                    req_model,
                    state,
                    stripped_fields,
                    requested_queries,
                    compute_greeks=(len(greek_queries_present) > 0),
                    t0=t0,
                    pre_reasons=pre_reasons,
                    post_reasons=post_reasons,
                    pre_diag=diag,
                    theta_roll_days=theta_roll_days,
                    theta_convention=theta_convention,
                    mds_digest_hex=mds_digest_hex,
                )
            fb_reasons_str = [r.value for r in (*pre_reasons, *post_reasons)]
            audit_rec = SR262AuditRecord(
                trade_id=req_model.TradeInfo if req_model.TradeInfo is not None else (state.trade_info if state.trade_info is not None else "UNSPECIFIED_TRADE"),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                pre_gate_passed=bool(pre_passed),
                post_gate_passed=bool(post_passed),
                fallback_triggered=False,
                fallback_reasons=fb_reasons_str,
                warnings=list(diag.get("warnings", [])),
                min_corr_eigenvalue_raw=float(diag.get("min_corr_eigenvalue_raw", 1.0)),
                min_corr_eigenvalue_shrunk=float(diag.get("min_corr_eigenvalue_shrunk", 1.0)),
                correlation_shrunk_applied=bool(diag.get("correlation_shrunk_applied", False)),
                mahalanobis_d2=float(diag.get("mahalanobis_d2", 0.0)),
                max_psi_score=float(diag.get("max_psi_score", 0.0)),
                min_pin_risk_z=float(diag.get("min_pin_risk_z", 100.0)),
                engine_selected="MaskedEGNO_Surrogate",
                conformal_half_width_bps=float(diag.get("conformal_half_width_bps", 0.0)),
                greeks_engine_used=greeks_engine_used,
            )
            with self.guardian._lock:
                self.guardian.audit_log.append(audit_rec)
            persisted_audit_rec = audit_rec
            pv_val = float(post_surr.product_value.detach().cpu().item())
            se_val = 0.0
            notional_val = float(post_surr.notional_pv.detach().cpu().item())
            coupon_val = float(post_surr.coupon_pv.detach().cpu().item())
            option_val = float(post_surr.option_pv.detach().cpu().item())
            funding_val = float(-post_surr.funding_pv.detach().cpu().item())
            ccy_val = post_surr.reporting_currency
            cash_today_val = float(post_surr.cash_to_settle_today)
            cf_items = post_surr.cashflows
            engine_used_str = "MaskedEGNO_Surrogate"
            fb_triggered = False
            audit_dict = audit_rec.to_dict()
            active_state = state
            surrogate_for_greeks = surr_for_greeks
        else:
            handled_single_pass_qmc = False
            pre_gate_sp_res: tuple[
                bool,
                list[AutocallFallbackReason],
                npt.NDArray[np.float64],
                dict[str, Any],
            ] | None = None
            qmc_crn_attempted = False
            if (
                state.m_rem > 0
                and not state.seasoned_state.already_autocalled
            ):
                pre_passed_sp, pre_reasons_sp, r_psd_sp, diag_sp = self.guardian.evaluate_pre_inference(
                    state,
                    state.baseline,
                    record_drift=False,
                )
                pre_gate_sp_res = (pre_passed_sp, pre_reasons_sp, r_psd_sp, diag_sp)
                if bool(diag_sp.get("correlation_shrunk_applied", False)) or (
                    AutocallFallbackReason.CORRELATION_NOT_PSD in pre_reasons_sp
                ):
                    _ = self.guardian._apply_shrunk_correlation_to_state(state, r_psd_sp)
                    _attach_runtime_market_metadata(state, req_model, mds_digest_hex=mds_digest_hex)
                if force_qmc or not pre_passed_sp:
                    sp_reasons = list(pre_reasons_sp)
                    if force_qmc and AutocallFallbackReason.ENGINE_REQUESTED_QMC not in sp_reasons:
                        sp_reasons.append(AutocallFallbackReason.ENGINE_REQUESTED_QMC)
                    qmc_crn_attempted = True
                    qmc_stage1_out = _compute_qmc_crn_if_available(
                        state,
                        queries=requested_queries,
                        n_paths=self.n_fallback_paths,
                        request=req_model,
                        device=self.device,
                        theta_roll_days=theta_roll_days,
                        theta_convention=theta_convention,
                    )
                    if qmc_stage1_out is not None and qmc_stage1_out.postproc_output is not None:
                        post_qmc = qmc_stage1_out.postproc_output
                        pv_val = float(post_qmc.product_value.detach().cpu().item())
                        se_val = float(
                            getattr(post_qmc, "standard_error", qmc_stage1_out.standard_error)
                        )
                        notional_val = float(post_qmc.notional_pv.detach().cpu().item())
                        coupon_val = float(post_qmc.coupon_pv.detach().cpu().item())
                        option_val = float(post_qmc.option_pv.detach().cpu().item())
                        funding_val = float(-post_qmc.funding_pv.detach().cpu().item())
                        ccy_val = post_qmc.reporting_currency
                        cash_today_val = float(post_qmc.cash_to_settle_today)
                        cf_items = post_qmc.cashflows
                        engine_used_str = "Tier3_DupireQMC"
                        fb_triggered = True
                        fb_reasons_str = [r.value for r in sp_reasons]
                        active_state = state
                        surrogate_for_greeks = None
                        if len(greek_queries_present) > 0:
                            greeks_out = qmc_stage1_out
                            greeks_engine_used = _map_greeks_engine_label(
                                getattr(qmc_stage1_out, "greeks_engine_used", "qmc_crn")
                            )
                        else:
                            greeks_out = None
                            greeks_engine_used = "None"
                        sp_warns = list(diag_sp.get("warnings", []))
                        audit_rec = SR262AuditRecord(
                            trade_id=req_model.TradeInfo if req_model.TradeInfo is not None else (state.trade_info if state.trade_info is not None else "UNSPECIFIED_TRADE"),
                            timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            pre_gate_passed=bool(pre_passed_sp),
                            post_gate_passed=True,
                            fallback_triggered=True,
                            fallback_reasons=fb_reasons_str,
                            warnings=sp_warns,
                            min_corr_eigenvalue_raw=float(diag_sp.get("min_corr_eigenvalue_raw", 1.0)),
                            min_corr_eigenvalue_shrunk=float(diag_sp.get("min_corr_eigenvalue_shrunk", 1.0)),
                            correlation_shrunk_applied=bool(diag_sp.get("correlation_shrunk_applied", False)),
                            mahalanobis_d2=float(diag_sp.get("mahalanobis_d2", 0.0)),
                            max_psi_score=float(diag_sp.get("max_psi_score", 0.0)),
                            min_pin_risk_z=float(diag_sp.get("min_pin_risk_z", 100.0)),
                            engine_selected="Tier3_DupireQMC",
                            conformal_half_width_bps=float(diag_sp.get("conformal_half_width_bps", 0.0)),
                            greeks_engine_used=greeks_engine_used,
                        )
                        with self.guardian._lock:
                            self.guardian.audit_log.append(audit_rec)
                        persisted_audit_rec = audit_rec
                        audit_dict = audit_rec.to_dict()
                        handled_single_pass_qmc = True

            if not handled_single_pass_qmc:
                def _greeks_fn_runtime(st: AutocallPreprocState, _surr: Any) -> AutocallGreeksResult:
                    if (
                        (not getattr(st, "repo_curves_map", None) and getattr(req_model.Model.MarketDataSet, "RepoSpreadCurves", None))
                        or (getattr(st, "quanto_meta", None) and not getattr(st, "quanto_fx_specs", None))
                    ):
                        _attach_runtime_market_metadata(st, req_model, mds_digest_hex=mds_digest_hex)
                    return compute_eq_autocall_greeks(
                        st,
                        queries=greek_queries_present,
                        surrogate=surr_for_greeks,
                        **greeks_extra,
                    )

                greeks_fn = (
                    _greeks_fn_runtime
                    if (len(greek_queries_present) > 0 and not force_qmc)
                    else None
                )
                pv_surr_fn: Any = None
                if self.device.type == "cuda" and not force_qmc and self._compiled_runner is not None:
                    def _compiled_surr_with_memo(st: AutocallPreprocState) -> AutocallSurrogateOutput:
                        s_res = self._predict_surrogate_compiled(st)
                        try:
                            setattr(st, "_precomputed_surr_out", s_res)
                        except (AttributeError, TypeError):
                            pass
                        return s_res

                    pv_surr_fn = _compiled_surr_with_memo
                guardian_kwargs: dict[str, Any] = {}
                if pv_surr_fn is not None:
                    guardian_kwargs["surrogate_forward_fn"] = pv_surr_fn
                pg_fn = self.guardian.price_with_guardian
                guardian_params = (
                    _GUARDIAN_PRICE_PARAMS
                    if getattr(pg_fn, "__func__", None) is EQAutocallModelGuardian.price_with_guardian
                    else frozenset(inspect.signature(pg_fn).parameters.keys())
                )
                if pre_gate_sp_res is not None and "pre_gate_result" in guardian_params:
                    guardian_kwargs["pre_gate_result"] = pre_gate_sp_res
                if (
                    state.m_rem > 0
                    and not state.seasoned_state.already_autocalled
                    and not qmc_crn_attempted
                    and "qmc_greeks_fn" in guardian_params
                ):
                    def _qmc_greeks_fn_runtime(st: AutocallPreprocState) -> AutocallGreeksResult | None:
                        if (
                            (not getattr(st, "repo_curves_map", None) and getattr(req_model.Model.MarketDataSet, "RepoSpreadCurves", None))
                            or (getattr(st, "quanto_meta", None) and not getattr(st, "quanto_fx_specs", None))
                        ):
                            _attach_runtime_market_metadata(st, req_model, mds_digest_hex=mds_digest_hex)
                        return _compute_qmc_crn_if_available(
                            st,
                            queries=requested_queries,
                            n_paths=self.n_fallback_paths,
                            request=req_model,
                            device=self.device,
                            theta_roll_days=theta_roll_days,
                            theta_convention=theta_convention,
                        )

                    guardian_kwargs["qmc_greeks_fn"] = _qmc_greeks_fn_runtime
                eff_fb_paths = (
                    self.n_fallback_paths
                    if self.n_fallback_paths is not None
                    else (8192 if self.device.type == "cuda" else 4096)
                )
                guardian_res: GuardianPricingResult = self.guardian.price_with_guardian(
                    state,
                    surrogate=surr_for_greeks,
                    n_fallback_paths=eff_fb_paths,
                    force_qmc=force_qmc,
                    record_drift=False,
                    greeks_fn=greeks_fn,
                    **guardian_kwargs,
                )
                _attach_runtime_market_metadata(
                    guardian_res.preproc_state,
                    req_model,
                    mds_digest_hex=mds_digest_hex,
                )
                pv_val = float(guardian_res.product_value)
                se_val = float(guardian_res.standard_error)
                notional_val = float(guardian_res.notional_pv)
                coupon_val = float(guardian_res.coupon_pv)
                option_val = float(guardian_res.option_pv)
                funding_val = float(-guardian_res.funding_pv)
                ccy_val = guardian_res.reporting_currency
                cash_today_val = float(guardian_res.cash_to_settle_today)
                cf_items = guardian_res.cashflows
                engine_used_str = guardian_res.engine_selected
                fb_triggered = bool(guardian_res.fallback_triggered)
                fb_reasons_str = [r.value for r in guardian_res.fallback_reasons]
                active_state = guardian_res.preproc_state
                surrogate_for_greeks = (
                    surr_for_greeks
                    if (not fb_triggered and not force_qmc and engine_used_str != "Tier1_Deterministic")
                    else None
                )
                if guardian_res.greeks_out is not None:
                    greeks_out = guardian_res.greeks_out
                    greeks_engine_used = _map_greeks_engine_label(
                        getattr(greeks_out, "greeks_engine_used", None)
                    )
                elif len(greek_queries_present) > 0:
                    if engine_used_str == "Tier3_DupireQMC":
                        greeks_engine_used = "Tier4_DupireQMC_CRN"
                    else:
                        greeks_engine_used = "Tier1_AnalyticalProxy_AD"
                persisted_audit_rec = guardian_res.audit_record
                with self.guardian._lock:
                    persisted_audit_rec.greeks_engine_used = greeks_engine_used
                audit_dict = persisted_audit_rec.to_dict()

        if len(greek_queries_present) > 0 and greeks_out is None:
            if (
                not use_proxy_only
                and engine_used_str == "Tier3_DupireQMC"
                and active_state.m_rem > 0
                and not active_state.seasoned_state.already_autocalled
            ):
                greeks_out = _compute_qmc_crn_if_available(
                    active_state,
                    queries=greek_queries_present,
                    n_paths=self.n_fallback_paths,
                    request=req_model,
                    device=self.device,
                    theta_roll_days=theta_roll_days,
                    theta_convention=theta_convention,
                )
                if greeks_out is None:
                    greeks_engine_used = "Tier1_AnalyticalProxy_AD"
                    st_warns = list(getattr(active_state, "warnings", None) or [])
                    if "QMC_CRN_GREEKS_FALLBACK_TO_PROXY" not in st_warns:
                        st_warns.append("QMC_CRN_GREEKS_FALLBACK_TO_PROXY")
                    setattr(active_state, "warnings", st_warns)
                    greeks_out = compute_eq_autocall_greeks(
                        active_state,
                        queries=requested_queries,
                        surrogate=None,
                        **greeks_extra,
                    )
                else:
                    greeks_engine_used = _map_greeks_engine_label(
                        getattr(greeks_out, "greeks_engine_used", "qmc_crn")
                    )
            else:
                greeks_out = compute_eq_autocall_greeks(
                    active_state,
                    queries=requested_queries,
                    surrogate=surrogate_for_greeks,
                    **greeks_extra,
                )
                greeks_engine_used = _map_greeks_engine_label(
                    getattr(greeks_out, "greeks_engine_used", None)
                )
            if persisted_audit_rec is not None:
                with self.guardian._lock:
                    persisted_audit_rec.greeks_engine_used = greeks_engine_used
            if audit_dict is not None:
                audit_dict["greeks_engine_used"] = greeks_engine_used

        if not math.isfinite(pv_val) or not math.isfinite(se_val):
            raise ValueError(f"Non-finite pricing output: ProductValue={pv_val!r}, StandardError={se_val!r}")

        cf_rows = _build_cashflows_fast(cf_items)

        if audit_dict is not None:
            extra_warns = _collect_state_and_surface_warnings(active_state)
            cur_warns = list(audit_dict.get("warnings") or [])
            for w_s in extra_warns:
                if w_s not in cur_warns:
                    cur_warns.append(w_s)
            audit_dict["warnings"] = cur_warns
            if persisted_audit_rec is not None:
                with self.guardian._lock:
                    persisted_audit_rec.warnings = list(cur_warns)

        delta_out, gamma_out = _merge_fx_delta_and_gamma(greeks_out, active_state)
        vega_out, cega_out = _merge_fx_vega_and_cega(greeks_out, active_state, req_model)
        rho_conv: Literal["quote", "decimal"] = (
            "decimal"
            if (greeks_out is not None and getattr(greeks_out, "rho_unit_convention", "quote") == "decimal")
            else "quote"
        )
        latency_ms = (time.perf_counter() - t0) * 1e3
        primary_fb_reason = fb_reasons_str[0] if len(fb_reasons_str) > 0 else None

        return _assemble_runtime_response(
            product_value=pv_val,
            reporting_currency=ccy_val,
            standard_error=se_val,
            cash_to_settle_today=cash_today_val,
            cash_flows=cf_rows,
            delta=delta_out,
            gamma=gamma_out,
            cash_gamma=(
                greeks_out.multivariate_aggregated_cash_gamma
                if greeks_out is not None
                else None
            ),
            vega=vega_out,
            rho=greeks_out.rho if greeks_out is not None else None,
            repo_rho=getattr(greeks_out, "repo_rho", None) if greeks_out is not None else None,
            rho_unit_convention=rho_conv,
            dividends_sensitivities=(
                greeks_out.dividends_sensitivities if greeks_out is not None else None
            ),
            theta=greeks_out.theta if greeks_out is not None else None,
            cega=cega_out,
            trade_info=req_model.TradeInfo,
            engine_used=engine_used_str,
            fallback_triggered=fb_triggered,
            fallback_reason=primary_fb_reason,
            fallback_reasons=fb_reasons_str,
            stripped_legacy_fields=stripped_fields,
            option=option_val,
            notional=notional_val,
            coupon=coupon_val,
            funding_leg=funding_val,
            audit_record=audit_dict,
            latency_ms=round(latency_ms, 4),
        )

    def price_murex_dag(
        self,
        raw_dag_text: str | bytes | Path | Mapping[str, Any],
        engine: str = "auto",
        *,
        include_all_greeks: bool = True,
        theta_roll_days: int = 1,
        theta_convention: str | None = None,
        repair_arbitrage: bool | None = None,
    ) -> EQAutocallRuntimeResponse:
        """Parses and prices a raw Murex `ConstantMessageMap` DAG payload (such as `eq_autocallable_request.txt`)."""
        t0 = time.perf_counter()
        eng_norm = _validate_engine(engine)
        req_model, parsed_is_dag = self._parse_request(raw_dag_text)
        eff_repair = self.repair_arbitrage if repair_arbitrage is None else bool(repair_arbitrage)
        eff_theta_conv = (
            self.theta_convention if theta_convention is None else str(theta_convention)
        )
        allow_cached_mds_digest = isinstance(getattr(req_model, "_raw_payload_digest_hex", None), str)
        if eff_repair:
            req_model = _repair_request_surfaces(req_model)
            allow_cached_mds_digest = False
        return self._price_parsed_trade(
            req_model,
            is_dag=parsed_is_dag,
            eng_norm=eng_norm,
            t0=t0,
            include_all_greeks=include_all_greeks,
            theta_roll_days=theta_roll_days,
            theta_convention=eff_theta_conv,
            allow_cached_mds_digest=allow_cached_mds_digest,
        )

    def _run_tensor_microbatch(
        self,
        node_d: torch.Tensor,
        edge_d: torch.Tensor,
        amask_d: torch.Tensor,
        sched_d: torch.Tensor,
        smask_d: torch.Tensor,
        glob_d: torch.Tensor,
        *,
        spots_t0: torch.Tensor | None,
        ref_spots: torch.Tensor | None,
        pv_in_transit: float | torch.Tensor,
        compute_greeks: bool,
        compute_cash_gamma: bool,
        no_notional_repayment: bool,
        autocall_coupon: float,
        force_coupons_on_autocall: bool,
        is_long_basket: torch.Tensor | None,
        w_fund_per_period: torch.Tensor | None,
        lower_strike: torch.Tensor | None,
        funding_pv: float | torch.Tensor,
        proxy_only: bool = False,
    ) -> BatchedSurrogateGreeksOutput:
        """Executes one validated tensor microbatch on `self.device`."""
        dev = node_d.device
        batch_size, d_max, _ = node_d.shape
        _, m_max, _ = sched_d.shape
        eff_surrogate = self._get_proxy_surrogate() if proxy_only else self.surrogate

        if compute_greeks:
            if dev.type == "cpu":
                ckey = _cpu_greeks_cache_key(
                    eff_surrogate,
                    node_d,
                    edge_d,
                    amask_d,
                    sched_d,
                    smask_d,
                    glob_d,
                    compute_cash_gamma=compute_cash_gamma,
                    no_notional_repayment=no_notional_repayment,
                    autocall_coupon=autocall_coupon,
                    force_coupons_on_autocall=force_coupons_on_autocall,
                    is_long_basket=is_long_basket,
                    w_fund_per_period=w_fund_per_period,
                    lower_strike=lower_strike,
                )
                with _CPU_GREEKS_CACHE_LOCK:
                    unit_out = _CPU_GREEKS_MICROBATCH_CACHE.get(ckey)
                    if unit_out is not None:
                        _CPU_GREEKS_MICROBATCH_CACHE.move_to_end(ckey)
                if unit_out is None:
                    with torch.set_grad_enabled(True):
                        unit_out = _call_with_supported_kwargs(
                            compute_batched_surrogate_greeks,
                            _GREEKS_BATCH_PARAMS,
                            eff_surrogate,
                            node_d,
                            edge_d,
                            amask_d,
                            sched_d,
                            smask_d,
                            glob_d,
                            spots_t0=None,
                            ref_spots=None,
                            pv_in_transit=0.0,
                            compute_cash_gamma=compute_cash_gamma,
                            no_notional_repayment=no_notional_repayment,
                            autocall_coupon=autocall_coupon,
                            force_coupons_on_autocall=force_coupons_on_autocall,
                            is_long_basket=is_long_basket,
                            w_fund_per_period=w_fund_per_period,
                            lower_strike=lower_strike,
                        )
                    with _CPU_GREEKS_CACHE_LOCK:
                        _CPU_GREEKS_MICROBATCH_CACHE[ckey] = unit_out
                        _CPU_GREEKS_MICROBATCH_CACHE.move_to_end(ckey)
                        while len(_CPU_GREEKS_MICROBATCH_CACHE) > _CPU_GREEKS_MICROBATCH_MAXSIZE:
                            _CPU_GREEKS_MICROBATCH_CACHE.popitem(last=False)
                assert unit_out is not None
                pv_adj = unit_out.product_value.clone()
                if pv_in_transit is not None:
                    pv_adj = pv_adj + torch.as_tensor(pv_in_transit, dtype=torch.float64, device=dev)
                if funding_pv is not None:
                    pv_adj = pv_adj - torch.as_tensor(funding_pv, dtype=torch.float64, device=dev)
                delta_adj = unit_out.delta.clone()
                if spots_t0 is not None or ref_spots is not None:
                    amask_bool = amask_d > 0.5
                    amask_64 = amask_bool.to(dtype=torch.float64)
                    log_m0_64 = torch.where(
                        amask_bool,
                        node_d[:, :, int(AutocallNodeCol.LOG_MONEYNESS)],
                        torch.zeros_like(amask_64),
                    )
                    sp_unit = torch.exp(log_m0_64).clamp(min=1e-12)
                    g_log_s = delta_adj * sp_unit
                    if spots_t0 is not None:
                        sp_eff = spots_t0.to(dtype=torch.float64, device=dev).clamp(min=1e-12)
                    else:
                        assert ref_spots is not None
                        sp_eff = sp_unit * ref_spots.to(dtype=torch.float64, device=dev).clamp(min=1e-12)
                    delta_adj = (g_log_s / sp_eff) * amask_64
                return BatchedSurrogateGreeksOutput(
                    product_value=pv_adj,
                    stopping_probs=unit_out.stopping_probs.clone(),
                    survival_probs=unit_out.survival_probs.clone(),
                    coupon_probs=unit_out.coupon_probs.clone(),
                    put_loss=unit_out.put_loss.clone(),
                    delta=delta_adj,
                    vega_atm=unit_out.vega_atm.clone(),
                    vega_surface_feats=unit_out.vega_surface_feats.clone(),
                    rho_df=unit_out.rho_df.clone(),
                    cega=unit_out.cega.clone(),
                    cash_gamma_euler=(
                        unit_out.cash_gamma_euler.clone()
                        if unit_out.cash_gamma_euler is not None
                        else None
                    ),
                )
            with torch.set_grad_enabled(True):
                cuda_out: BatchedSurrogateGreeksOutput = _call_with_supported_kwargs(
                    compute_batched_surrogate_greeks,
                    _GREEKS_BATCH_PARAMS,
                    eff_surrogate,
                    node_d,
                    edge_d,
                    amask_d,
                    sched_d,
                    smask_d,
                    glob_d,
                    spots_t0=spots_t0,
                    ref_spots=ref_spots,
                    pv_in_transit=pv_in_transit,
                    compute_cash_gamma=compute_cash_gamma,
                    no_notional_repayment=no_notional_repayment,
                    autocall_coupon=autocall_coupon,
                    force_coupons_on_autocall=force_coupons_on_autocall,
                    is_long_basket=is_long_basket,
                    w_fund_per_period=w_fund_per_period,
                    lower_strike=lower_strike,
                )
            if funding_pv is not None:
                fund_sub = torch.as_tensor(funding_pv, dtype=torch.float64, device=dev)
                if bool((fund_sub != 0.0).any()):
                    return BatchedSurrogateGreeksOutput(
                        product_value=cuda_out.product_value - fund_sub,
                        stopping_probs=cuda_out.stopping_probs,
                        survival_probs=cuda_out.survival_probs,
                        coupon_probs=cuda_out.coupon_probs,
                        put_loss=cuda_out.put_loss,
                        delta=cuda_out.delta,
                        vega_atm=cuda_out.vega_atm,
                        vega_surface_feats=cuda_out.vega_surface_feats,
                        rho_df=cuda_out.rho_df,
                        cega=cuda_out.cega,
                        cash_gamma_euler=cuda_out.cash_gamma_euler,
                    )
            return cuda_out

        with torch.no_grad():
            d_act, m_act = _trim_active_dimensions(node_d, edge_d, amask_d, sched_d, smask_d)
            node_f32 = node_d[:, :d_act, :].to(dtype=torch.float32)
            edge_f32 = edge_d[:, :d_act, :d_act, :].to(dtype=torch.float32)
            amask_f32 = amask_d[:, :d_act].to(dtype=torch.float32)
            sched_f32 = sched_d[:, :m_act, :].to(dtype=torch.float32)
            smask_f32 = smask_d[:, :m_act].to(dtype=torch.float32)
            glob_f32 = glob_d.to(dtype=torch.float32)

            if proxy_only:
                z_res = torch.zeros((batch_size, m_max), dtype=torch.float32, device=dev)
                z_put = torch.zeros(batch_size, dtype=torch.float32, device=dev)
                out = _decode_residuals_to_surrogate_output(
                    z_res,
                    z_res,
                    z_put,
                    sched_d.to(dtype=torch.float32),
                    smask_d.to(dtype=torch.float32),
                    glob_f32,
                )
            elif self._compiled_runner is not None and dev.type == "cuda":
                dhaz_act, dcpn_act, dput = self._compiled_runner.predict_residuals(
                    node_f32, edge_f32, amask_f32, sched_f32, smask_f32, glob_f32
                )
                if m_act < m_max:
                    dhaz = torch.nn.functional.pad(dhaz_act, (0, m_max - m_act))
                    dcpn = torch.nn.functional.pad(dcpn_act, (0, m_max - m_act))
                else:
                    dhaz = dhaz_act
                    dcpn = dcpn_act
                out = _decode_residuals_to_surrogate_output(
                    dhaz,
                    dcpn,
                    dput,
                    sched_d.to(dtype=torch.float32),
                    smask_d.to(dtype=torch.float32),
                    glob_f32,
                )
            else:
                out_act = self.surrogate(
                    node_f32, edge_f32, amask_f32, sched_f32, smask_f32, glob_f32
                )
                if m_act < m_max:
                    pad_len = m_max - m_act
                    out = AutocallSurrogateOutput(
                        stopping_probs=torch.nn.functional.pad(out_act.stopping_probs, (0, pad_len)),
                        survival_probs=torch.cat(
                            [
                                out_act.survival_probs,
                                out_act.survival_probs[:, -1:].expand(-1, pad_len),
                            ],
                            dim=1,
                        ),
                        hazard_rates=torch.nn.functional.pad(out_act.hazard_rates, (0, pad_len)),
                        coupon_probs=torch.nn.functional.pad(out_act.coupon_probs, (0, pad_len)),
                        put_loss=out_act.put_loss,
                        delta_hazard_logits=torch.nn.functional.pad(out_act.delta_hazard_logits, (0, pad_len)),
                        delta_coupon_logits=torch.nn.functional.pad(out_act.delta_coupon_logits, (0, pad_len)),
                        delta_put_loss=out_act.delta_put_loss,
                    )
                else:
                    out = out_act
            pv_f64 = _call_with_supported_kwargs(
                assemble_surrogate_product_value,
                _ASSEMBLE_SURR_PARAMS,
                out,
                sched_d,
                smask_d,
                glob_d,
                no_notional_repayment=no_notional_repayment,
                autocall_coupon=autocall_coupon,
                force_coupons_on_autocall=force_coupons_on_autocall,
                is_long_basket=is_long_basket,
                pv_in_transit=pv_in_transit,
                funding_pv=funding_pv,
                w_fund_per_period=w_fund_per_period,
                lower_strike=lower_strike,
            )
        return BatchedSurrogateGreeksOutput(
            product_value=pv_f64.detach().to(dtype=torch.float64),
            stopping_probs=out.stopping_probs.detach().to(dtype=torch.float64),
            survival_probs=out.survival_probs.detach().to(dtype=torch.float64),
            coupon_probs=out.coupon_probs.detach().to(dtype=torch.float64),
            put_loss=out.put_loss.detach().to(dtype=torch.float64),
            delta=torch.zeros((batch_size, d_max), dtype=torch.float64, device=dev),
            vega_atm=torch.zeros((batch_size, d_max, 2), dtype=torch.float64, device=dev),
            vega_surface_feats=torch.zeros((batch_size, d_max, 7), dtype=torch.float64, device=dev),
            rho_df=torch.zeros((batch_size, m_max), dtype=torch.float64, device=dev),
            cega=torch.zeros((batch_size, d_max, d_max), dtype=torch.float64, device=dev),
            cash_gamma_euler=torch.zeros(batch_size, dtype=torch.float64, device=dev),
        )

    def _get_proxy_surrogate(self) -> MaskedEQAutocallSurrogate:
        if self._proxy_surrogate is not None:
            return self._proxy_surrogate
        with self._lock:
            if self._proxy_surrogate is None:
                p_surr = MaskedEQAutocallSurrogate(
                    d_max=self.d_max,
                    m_max=self.m_max,
                    d_node=int(getattr(self.surrogate, "d_node", 16)),
                    d_global=int(getattr(self.surrogate, "d_global", 16)),
                ).to(device=self.device, dtype=torch.float32)
                p_surr.zero_init_residual_heads(std=0.0)
                p_surr.eval()
                self._proxy_surrogate = p_surr
            return self._proxy_surrogate

    def price_tensor_batch(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
        *,
        spots_t0: torch.Tensor | None = None,
        ref_spots: torch.Tensor | None = None,
        pv_in_transit: float | torch.Tensor = 0.0,
        compute_greeks: bool = True,
        compute_cash_gamma: bool = True,
        no_notional_repayment: bool = False,
        autocall_coupon: float = 0.0,
        force_coupons_on_autocall: bool = False,
        is_long_basket: torch.Tensor | None = None,
        w_fund_per_period: torch.Tensor | None = None,
        lower_strike: torch.Tensor | None = None,
        funding_pv: float | torch.Tensor = 0.0,
        run_guardian: bool = True,
        proxy_only: bool = False,
    ) -> BatchedSurrogateGreeksOutput:
        """Vectorized GPU/CPU tensor batch execution with microbatch chunking and Guardian telemetry."""
        if (
            node_feats.ndim != 3
            or edge_feats.ndim != 4
            or asset_mask.ndim != 2
            or sched_feats.ndim != 3
            or sched_mask.ndim != 2
            or global_feats.ndim != 2
        ):
            raise ValueError("Invalid tensor rank in price_tensor_batch.")
        batch_size, d_dim, n_col = node_feats.shape
        sched_batch, m_dim, s_col = sched_feats.shape
        g_col = int(global_feats.shape[-1])

        if batch_size <= 0 or batch_size > MAX_BATCH_TRADES:
            raise ValueError(f"batch_size={batch_size} out of bounds [1, {MAX_BATCH_TRADES}].")
        if d_dim < 1 or m_dim < 1:
            raise ValueError(f"Tensor dimensions (D={d_dim}, M={m_dim}) must be >= 1.")
        if d_dim > self.surrogate.d_max or m_dim > self.surrogate.m_max:
            raise ValueError(
                f"Tensor batch dimensions (D={d_dim}, M={m_dim}) exceed surrogate capacity "
                f"({self.surrogate.d_max}, {self.surrogate.m_max})."
            )
        if sched_batch != batch_size or s_col != len(AutocallSchedCol):
            raise ValueError(
                f"Expected sched_feats shape ({batch_size}, {m_dim}, {len(AutocallSchedCol)}), "
                f"got {tuple(sched_feats.shape)}."
            )
        expected_d_node = int(getattr(self.surrogate, "d_node", 16))
        expected_d_global = int(getattr(self.surrogate, "d_global", 16))
        allowed_d_node = {14, 16, 18, expected_d_node}
        allowed_d_global = {10, 16, expected_d_global}
        if (
            n_col not in allowed_d_node
            or g_col not in allowed_d_global
            or n_col > max(expected_d_node, 18)
            or g_col > max(expected_d_global, 16)
        ):
            raise ValueError(
                f"Feature schema width mismatch: got (d_node={n_col}, d_global={g_col}), "
                f"expected ({expected_d_node}, {expected_d_global})."
            )
        if n_col < expected_d_node:
            node_feats = torch.nn.functional.pad(node_feats, (0, expected_d_node - n_col))
            n_col = expected_d_node
        elif n_col > expected_d_node:
            node_feats = node_feats[:, :, :expected_d_node]
            n_col = expected_d_node
        if g_col < expected_d_global:
            global_feats = torch.nn.functional.pad(global_feats, (0, expected_d_global - g_col))
            g_col = expected_d_global
        elif g_col > expected_d_global:
            global_feats = global_feats[:, :expected_d_global]
            g_col = expected_d_global
        if (
            edge_feats.shape[:3] != (batch_size, d_dim, d_dim)
            or int(edge_feats.shape[3]) < 4
            or asset_mask.shape != (batch_size, d_dim)
        ):
            raise ValueError("edge_feats or asset_mask shape mismatch with node_feats.")
        if sched_mask.shape != (batch_size, m_dim) or global_feats.shape[0] != batch_size:
            raise ValueError("sched_mask or global_feats batch dimension mismatch.")

        d_act, m_act = _trim_active_dimensions(node_feats, edge_feats, asset_mask, sched_feats, sched_mask)

        pre_diag: dict[str, Any] | None = None
        if run_guardian and not proxy_only:
            pre_diag = self.guardian.evaluate_tensor_batch_pre(
                node_feats,
                edge_feats,
                asset_mask,
                sched_feats,
                sched_mask,
                global_feats,
                record_drift=False,
            )

        dev = self.device
        node_d = node_feats.to(device=dev, dtype=torch.float64)
        edge_d = edge_feats.to(device=dev, dtype=torch.float64)
        amask_d = asset_mask.to(device=dev, dtype=torch.float64)
        sched_d = sched_feats.to(device=dev, dtype=torch.float64)
        smask_d = sched_mask.to(device=dev, dtype=torch.float64)
        glob_d = global_feats.to(device=dev, dtype=torch.float64)
        sp_d = spots_t0.to(device=dev, dtype=torch.float64) if spots_t0 is not None else None
        ref_d = ref_spots.to(device=dev, dtype=torch.float64) if ref_spots is not None else None
        pvit_t = pv_in_transit.to(device=dev, dtype=torch.float64) if isinstance(pv_in_transit, torch.Tensor) else pv_in_transit
        fund_t = funding_pv.to(device=dev, dtype=torch.float64) if isinstance(funding_pv, torch.Tensor) else funding_pv
        ilb_d = is_long_basket.to(device=dev, dtype=torch.float64) if is_long_basket is not None else None
        wf_d = w_fund_per_period.to(device=dev, dtype=torch.float64) if w_fund_per_period is not None else None
        lk_d = lower_strike.to(device=dev, dtype=torch.float64) if lower_strike is not None else None

        need_chunking = (
            dev.type == "cuda"
            and (
                (m_act > 24 and batch_size > CUDA_MICROBATCH_LIMIT_M48)
                or batch_size > CUDA_MICROBATCH_LIMIT_M48
            )
        )
        if not need_chunking:
            out = self._run_tensor_microbatch(
                node_d,
                edge_d,
                amask_d,
                sched_d,
                smask_d,
                glob_d,
                spots_t0=sp_d,
                ref_spots=ref_d,
                pv_in_transit=pvit_t,
                compute_greeks=compute_greeks,
                compute_cash_gamma=compute_cash_gamma,
                no_notional_repayment=no_notional_repayment,
                autocall_coupon=autocall_coupon,
                force_coupons_on_autocall=force_coupons_on_autocall,
                is_long_basket=ilb_d,
                w_fund_per_period=wf_d,
                lower_strike=lk_d,
                funding_pv=fund_t,
                proxy_only=proxy_only,
            )
        else:
            chunk_size = CUDA_MICROBATCH_LIMIT_M48
            chunks: list[BatchedSurrogateGreeksOutput] = []
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                sl = slice(start, end)
                pvit_sl = pvit_t[sl] if isinstance(pvit_t, torch.Tensor) and pvit_t.ndim > 0 else pvit_t
                fund_sl = fund_t[sl] if isinstance(fund_t, torch.Tensor) and fund_t.ndim > 0 else fund_t
                chunks.append(
                    self._run_tensor_microbatch(
                        node_d[sl],
                        edge_d[sl],
                        amask_d[sl],
                        sched_d[sl],
                        smask_d[sl],
                        glob_d[sl],
                        spots_t0=sp_d[sl] if sp_d is not None else None,
                        ref_spots=ref_d[sl] if ref_d is not None else None,
                        pv_in_transit=pvit_sl,
                        compute_greeks=compute_greeks,
                        compute_cash_gamma=compute_cash_gamma,
                        no_notional_repayment=no_notional_repayment,
                        autocall_coupon=autocall_coupon,
                        force_coupons_on_autocall=force_coupons_on_autocall,
                        is_long_basket=ilb_d[sl] if ilb_d is not None else None,
                        w_fund_per_period=wf_d[sl] if wf_d is not None else None,
                        lower_strike=lk_d[sl] if lk_d is not None else None,
                        funding_pv=fund_sl,
                        proxy_only=proxy_only,
                    )
                )
            out = _concat_batched_greeks(chunks)

        resolved_engine_tag = "Tier1_AnalyticalProxy" if proxy_only else "MaskedEGNO_Surrogate"
        if not proxy_only and run_guardian and pre_diag is not None:
            post_diag = self.guardian.evaluate_tensor_batch_post(
                out.stopping_probs,
                out.coupon_probs,
                out.put_loss,
                out.product_value,
                smask_d,
                glob_d,
            )
            batch_engine_selected = "MaskedEGNO_Surrogate"
            batch_greeks_selected = "Tier2_BatchedAD" if compute_greeks else "None"
            if not bool(post_diag.get("all_passed", True)):
                # Fail over non-finite/simplex/put violations cleanly to zero-residual Tier-1 baseline.
                with torch.no_grad():
                    z_res = torch.zeros((batch_size, m_dim), dtype=torch.float32, device=dev)
                    z_put = torch.zeros(batch_size, dtype=torch.float32, device=dev)
                    base_out = _decode_residuals_to_surrogate_output(
                        z_res,
                        z_res,
                        z_put,
                        sched_d.to(dtype=torch.float32),
                        smask_d.to(dtype=torch.float32),
                        glob_d.to(dtype=torch.float32),
                    )
                    base_pv = _call_with_supported_kwargs(
                        assemble_surrogate_product_value,
                        _ASSEMBLE_SURR_PARAMS,
                        base_out,
                        sched_d,
                        smask_d,
                        glob_d,
                        no_notional_repayment=no_notional_repayment,
                        autocall_coupon=autocall_coupon,
                        force_coupons_on_autocall=force_coupons_on_autocall,
                        is_long_basket=ilb_d,
                        pv_in_transit=pvit_t,
                        funding_pv=fund_t,
                        w_fund_per_period=wf_d,
                        lower_strike=lk_d,
                    )
                out = BatchedSurrogateGreeksOutput(
                    product_value=base_pv.detach().to(dtype=torch.float64),
                    stopping_probs=base_out.stopping_probs.detach().to(dtype=torch.float64),
                    survival_probs=base_out.survival_probs.detach().to(dtype=torch.float64),
                    coupon_probs=base_out.coupon_probs.detach().to(dtype=torch.float64),
                    put_loss=base_out.put_loss.detach().to(dtype=torch.float64),
                    delta=torch.where(torch.isfinite(out.delta), out.delta, 0.0),
                    vega_atm=torch.where(torch.isfinite(out.vega_atm), out.vega_atm, 0.0),
                    vega_surface_feats=torch.where(
                        torch.isfinite(out.vega_surface_feats), out.vega_surface_feats, 0.0
                    ),
                    rho_df=torch.where(torch.isfinite(out.rho_df), out.rho_df, 0.0),
                    cega=torch.where(torch.isfinite(out.cega), out.cega, 0.0),
                    cash_gamma_euler=(
                        torch.where(torch.isfinite(out.cash_gamma_euler), out.cash_gamma_euler, 0.0)
                        if out.cash_gamma_euler is not None
                        else None
                    ),
                )
                batch_engine_selected = "Tier1_AnalyticalProxy_Fallback"
                batch_greeks_selected = "Tier1_AnalyticalProxy_AD" if compute_greeks else "None"
            resolved_engine_tag = batch_engine_selected
            self.guardian.record_batch_audit(
                trade_id=f"TENSOR_BATCH_{batch_size}",
                pre_diag=pre_diag,
                post_diag=post_diag,
                engine_selected=batch_engine_selected,
                greeks_engine_used=batch_greeks_selected,
            )
        try:
            object.__setattr__(out, "_engine_selected", resolved_engine_tag)
        except (AttributeError, TypeError):
            pass
        return out

    def _build_qmc_fallback_response(
        self,
        req_m: EQAutocallableTradeRequest,
        state_k: AutocallPreprocState,
        stripped_f: list[str],
        requested_queries: Sequence[str],
        compute_greeks: bool,
        t0: float,
        *,
        pre_reasons: Sequence[AutocallFallbackReason] = (),
        post_reasons: Sequence[AutocallFallbackReason] = (),
        pre_diag: Mapping[str, Any] | None = None,
        theta_roll_days: int = 1,
        theta_convention: str = "murex_bizday_forward_fixed",
        mds_digest_hex: str | None = None,
    ) -> EQAutocallRuntimeResponse:
        """Executes single-pass post-gate Tier-3/4 Dupire QMC fallback directly on an already preprocessed state."""
        greek_queries_present = [
            q for q in requested_queries if q in ALL_8_GREEK_QUERIES or q == "ConvexCega"
        ]
        eff_mds_hex = _attach_runtime_market_metadata(
            state_k,
            req_m,
            mds_digest_hex=mds_digest_hex,
        )
        combined_reasons: list[AutocallFallbackReason] = []
        for r in (*pre_reasons, *post_reasons):
            if r not in combined_reasons:
                combined_reasons.append(r)

        greeks_out: AutocallGreeksResult | None = None
        greeks_engine_used = "None"
        handled_single_pass = False
        pv_val = 0.0
        se_val = 0.0
        notional_val = 0.0
        coupon_val = 0.0
        option_val = 0.0
        funding_val = 0.0
        ccy_val = state_k.payment_currency
        cash_today_val = float(getattr(state_k.seasoned_state, "cash_to_settle_today", 0.0))
        cf_items: Sequence[Any] = ()
        engine_used_str = "Tier3_DupireQMC"
        fb_triggered = True
        active_state = state_k
        want_greeks_resp = bool(compute_greeks and len(greek_queries_present) > 0)

        if state_k.m_rem > 0 and not state_k.seasoned_state.already_autocalled:
            qmc_stage1_out = _compute_qmc_crn_if_available(
                state_k,
                queries=requested_queries,
                n_paths=self.n_fallback_paths,
                request=req_m,
                device=self.device,
                theta_roll_days=theta_roll_days,
                theta_convention=theta_convention,
            )
            if qmc_stage1_out is not None and qmc_stage1_out.postproc_output is not None:
                post_qmc = qmc_stage1_out.postproc_output
                pv_val = float(post_qmc.product_value.detach().cpu().item())
                se_val = float(getattr(post_qmc, "standard_error", qmc_stage1_out.standard_error))
                notional_val = float(post_qmc.notional_pv.detach().cpu().item())
                coupon_val = float(post_qmc.coupon_pv.detach().cpu().item())
                option_val = float(post_qmc.option_pv.detach().cpu().item())
                funding_val = float(-post_qmc.funding_pv.detach().cpu().item())
                ccy_val = post_qmc.reporting_currency
                cash_today_val = float(post_qmc.cash_to_settle_today)
                cf_items = post_qmc.cashflows
                if want_greeks_resp:
                    greeks_out = qmc_stage1_out
                    greeks_engine_used = _map_greeks_engine_label(
                        getattr(qmc_stage1_out, "greeks_engine_used", "qmc_crn")
                    )
                else:
                    greeks_out = None
                    greeks_engine_used = "None"
                diag_m = dict(pre_diag or {})
                fb_reasons_str = [r.value for r in combined_reasons]
                audit_rec = SR262AuditRecord(
                    trade_id=req_m.TradeInfo if req_m.TradeInfo is not None else (state_k.trade_info if state_k.trade_info is not None else "UNSPECIFIED_TRADE"),
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    pre_gate_passed=(len(pre_reasons) == 0),
                    post_gate_passed=(len(post_reasons) == 0),
                    fallback_triggered=True,
                    fallback_reasons=fb_reasons_str,
                    warnings=list(diag_m.get("warnings", [])),
                    min_corr_eigenvalue_raw=float(diag_m.get("min_corr_eigenvalue_raw", 1.0)),
                    min_corr_eigenvalue_shrunk=float(diag_m.get("min_corr_eigenvalue_shrunk", 1.0)),
                    correlation_shrunk_applied=bool(diag_m.get("correlation_shrunk_applied", False)),
                    mahalanobis_d2=float(diag_m.get("mahalanobis_d2", 0.0)),
                    max_psi_score=float(diag_m.get("max_psi_score", 0.0)),
                    min_pin_risk_z=float(diag_m.get("min_pin_risk_z", 100.0)),
                    engine_selected="Tier3_DupireQMC",
                    conformal_half_width_bps=float(diag_m.get("conformal_half_width_bps", 0.0)),
                    greeks_engine_used=greeks_engine_used,
                )
                with self.guardian._lock:
                    self.guardian.audit_log.append(audit_rec)
                audit_dict = audit_rec.to_dict()
                handled_single_pass = True

        if not handled_single_pass:
            eff_fb_paths = (
                self.n_fallback_paths
                if self.n_fallback_paths is not None
                else (8192 if self.device.type == "cuda" else 4096)
            )
            guardian_res = self.guardian.price_with_guardian(
                state_k,
                surrogate=None,
                n_fallback_paths=eff_fb_paths,
                force_qmc=True,
                record_drift=False,
                greeks_fn=None,
            )
            _attach_runtime_market_metadata(
                guardian_res.preproc_state,
                req_m,
                mds_digest_hex=eff_mds_hex,
            )
            active_state = guardian_res.preproc_state
            if want_greeks_resp:
                greeks_extra: dict[str, Any] = {}
                if _SUPPORTS_STABILIZE_F32_SURROGATE:
                    greeks_extra["stabilize_f32_surrogate"] = True
                if _SUPPORTS_GREEKS_REQUEST_PARAM:
                    greeks_extra["request"] = req_m
                if _SUPPORTS_GREEKS_THETA_PARAM:
                    greeks_extra["theta_roll_days"] = int(theta_roll_days)
                if _SUPPORTS_GREEKS_THETA_CONV_PARAM:
                    greeks_extra["theta_convention"] = str(theta_convention)
                greeks_out = _compute_qmc_crn_if_available(
                    active_state,
                    queries=requested_queries,
                    n_paths=self.n_fallback_paths,
                    request=req_m,
                    device=self.device,
                    theta_roll_days=theta_roll_days,
                    theta_convention=theta_convention,
                )
                if greeks_out is None:
                    greeks_engine_used = "Tier1_AnalyticalProxy_AD"
                    st_warns = list(getattr(active_state, "warnings", None) or [])
                    if "QMC_CRN_GREEKS_FALLBACK_TO_PROXY" not in st_warns:
                        st_warns.append("QMC_CRN_GREEKS_FALLBACK_TO_PROXY")
                    setattr(active_state, "warnings", st_warns)
                    greeks_out = compute_eq_autocall_greeks(
                        active_state,
                        queries=requested_queries,
                        surrogate=None,
                        **greeks_extra,
                    )
                else:
                    greeks_engine_used = _map_greeks_engine_label(
                        getattr(greeks_out, "greeks_engine_used", "qmc_crn")
                    )

            merged_reasons = [r for r in combined_reasons]
            for gr in guardian_res.fallback_reasons:
                if gr not in merged_reasons:
                    merged_reasons.append(gr)
            fb_reasons_str = [r.value for r in merged_reasons]
            with self.guardian._lock:
                if req_m.TradeInfo is not None:
                    guardian_res.audit_record.trade_id = req_m.TradeInfo
                guardian_res.audit_record.post_gate_passed = (len(post_reasons) == 0) if post_reasons else guardian_res.audit_record.post_gate_passed
                guardian_res.audit_record.fallback_reasons = list(fb_reasons_str)
                guardian_res.audit_record.greeks_engine_used = greeks_engine_used
            audit_rec = guardian_res.audit_record
            audit_dict = audit_rec.to_dict()
            pv_val = float(guardian_res.product_value)
            se_val = float(guardian_res.standard_error)
            notional_val = float(guardian_res.notional_pv)
            coupon_val = float(guardian_res.coupon_pv)
            option_val = float(guardian_res.option_pv)
            funding_val = float(-guardian_res.funding_pv)
            ccy_val = guardian_res.reporting_currency
            cash_today_val = float(guardian_res.cash_to_settle_today)
            cf_items = guardian_res.cashflows
            engine_used_str = guardian_res.engine_selected
            fb_triggered = bool(guardian_res.fallback_triggered)

        extra_warns = _collect_state_and_surface_warnings(active_state)
        cur_warns = list(audit_dict.get("warnings") or [])
        for w_s in extra_warns:
            if w_s not in cur_warns:
                cur_warns.append(w_s)
        audit_dict["warnings"] = cur_warns
        with self.guardian._lock:
            audit_rec.warnings = list(cur_warns)
        delta_out, gamma_out = _merge_fx_delta_and_gamma(greeks_out, active_state)
        vega_out, cega_out = _merge_fx_vega_and_cega(greeks_out, active_state, req_m)
        rho_conv: Literal["quote", "decimal"] = (
            "decimal"
            if (greeks_out is not None and getattr(greeks_out, "rho_unit_convention", "quote") == "decimal")
            else "quote"
        )
        cf_rows = _build_cashflows_fast(cf_items)
        return _assemble_runtime_response(
            product_value=pv_val,
            reporting_currency=ccy_val,
            standard_error=se_val,
            cash_to_settle_today=cash_today_val,
            cash_flows=cf_rows,
            delta=delta_out,
            gamma=gamma_out,
            cash_gamma=(
                greeks_out.multivariate_aggregated_cash_gamma
                if greeks_out is not None
                else None
            ),
            vega=vega_out,
            rho=greeks_out.rho if greeks_out is not None else None,
            repo_rho=getattr(greeks_out, "repo_rho", None) if greeks_out is not None else None,
            rho_unit_convention=rho_conv,
            dividends_sensitivities=(
                greeks_out.dividends_sensitivities if greeks_out is not None else None
            ),
            theta=greeks_out.theta if greeks_out is not None else None,
            cega=cega_out,
            trade_info=req_m.TradeInfo,
            engine_used=engine_used_str,
            fallback_triggered=fb_triggered,
            fallback_reason=fb_reasons_str[0] if len(fb_reasons_str) > 0 else None,
            fallback_reasons=fb_reasons_str,
            stripped_legacy_fields=stripped_f,
            option=option_val,
            notional=notional_val,
            coupon=coupon_val,
            funding_leg=funding_val,
            audit_record=audit_dict,
            latency_ms=round((time.perf_counter() - t0) * 1e3, 4),
        )

    def price_batch_json(
        self,
        requests: Sequence[EQAutocallableTradeRequest | dict[str, Any] | str],
        compute_greeks: bool = True,
        engine: str = "auto",
        *,
        theta_convention: str | None = None,
        repair_arbitrage: bool | None = None,
    ) -> list[EQAutocallRuntimeResponse]:
        """Prices a sequence of JSON/Pydantic requests via collated batching with market dedup & SR 26-2 audit."""
        if len(requests) == 0 or len(requests) > MAX_BATCH_TRADES:
            raise ValueError(
                f"Batch requests length {len(requests)} out of bounds [1, {MAX_BATCH_TRADES}]."
            )

        engine_norm = _validate_engine(engine)
        eff_repair = self.repair_arbitrage if repair_arbitrage is None else bool(repair_arbitrage)
        eff_theta_conv = (
            self.theta_convention if theta_convention is None else str(theta_convention)
        )
        full_jacobian_queries = {"Gamma", "Vega", "Rho", "DividendsSensitivities", "Theta"}

        model_cache: dict[str, DupireModelSpec] = {}
        raw_model_cache: dict[
            str,
            tuple[DupireModelSpec, tuple[tuple[str | int, ...], ...]],
        ] = {}
        mds_cache: dict[
            str,
            tuple[Any, tuple[tuple[str | int, ...], ...]],
        ] = {}
        mds_only_cache: dict[str, Any] = {}
        mds_id_cache: dict[int, tuple[Mapping[str, Any], str, bytes]] = {}
        parsed_items: list[tuple[EQAutocallableTradeRequest, list[str], list[str], set[str], str]] = []
        _standard_keys = {
            "Product",
            "Model",
            "Queries",
            "TradeInfo",
            "Result",
            "stripped_legacy_fields",
        }
        for item in requests:
            req_m: EQAutocallableTradeRequest
            mkt_key: str
            if (
                not eff_repair
                and isinstance(item, Mapping)
                and "Product" in item
                and "Model" in item
                and set(item.keys()) <= _standard_keys
            ):
                raw_model_obj = item["Model"]
                if isinstance(raw_model_obj, Mapping):
                    raw_mds = raw_model_obj.get("MarketDataSet")
                    raw_mds_dict: dict[str, Any] | None = None
                    mds_key_opt: str | None = None
                    mds_bytes = b""
                    if isinstance(raw_mds, Mapping):
                        raw_mds_dict = raw_mds if type(raw_mds) is dict else dict(raw_mds)
                        hit_mds = mds_id_cache.get(id(raw_mds))
                        if hit_mds is not None and hit_mds[0] is raw_mds:
                            mds_key_opt = hit_mds[1]
                            mds_bytes = hit_mds[2]
                        if mds_key_opt is None:
                            try:
                                mds_bytes = orjson.dumps(
                                    raw_mds_dict,
                                    option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SORT_KEYS,
                                )
                                mds_key_opt = hashlib.blake2b(mds_bytes, digest_size=16).hexdigest()
                                mds_id_cache[id(raw_mds)] = (raw_mds, mds_key_opt, mds_bytes)
                            except (TypeError, ValueError, OverflowError):
                                mds_key_opt = None
                                mds_bytes = b""

                    mds_hit_obj: Any = None
                    if mds_key_opt is not None and mds_key_opt in mds_cache and raw_mds_dict is not None:
                        cand_mds, cand_null_paths = mds_cache[mds_key_opt]
                        if not cand_null_paths or _check_no_nan_at_null_paths(raw_mds_dict, cand_null_paths):
                            mds_hit_obj = cand_mds

                    if (
                        mds_hit_obj is not None
                        and set(raw_model_obj.keys())
                        <= {"TypeName", "MarketDataSet", "FinancingCurveIdentifiers", "stripped_legacy_fields"}
                    ):
                        cached_mds = mds_hit_obj
                        raw_type = raw_model_obj.get("TypeName")
                        if raw_type is not None and raw_type != "DupireLocalVolatilityModel":
                            raise ValueError(f"Unsupported Model.TypeName={raw_type!r}")
                        fin_raw = raw_model_obj.get("FinancingCurveIdentifiers")
                        if fin_raw is None:
                            fin_list: list[str] = []
                        elif isinstance(fin_raw, (list, tuple)) and all(isinstance(x, str) for x in fin_raw):
                            fin_list = [str(x) for x in fin_raw]
                        else:
                            raise ValueError("Model.FinancingCurveIdentifiers must be a list of strings")
                        raw_model_leg = raw_model_obj.get("stripped_legacy_fields")
                        if raw_model_leg is None:
                            model_leg: list[str] = []
                        elif isinstance(raw_model_leg, (list, tuple)):
                            model_leg = [str(x) for x in raw_model_leg]
                        else:
                            raise ValueError("Model.stripped_legacy_fields must be a list of strings")
                        for curve in cached_mds.RatesCurvesBundles.values():
                            for f_s in curve.stripped_legacy_fields:
                                if f_s not in model_leg:
                                    model_leg.append(f_s)
                        fin_part = ",".join(fin_list)
                        leg_part = ",".join(model_leg)
                        mkt_key = hashlib.blake2b(
                            f"{mds_key_opt}|fin:{fin_part}|leg:{leg_part}".encode("utf-8"),
                            digest_size=16,
                        ).hexdigest()
                        cached_model = model_cache.get(mkt_key)
                        if cached_model is None:
                            cached_model = DupireModelSpec.model_construct(
                                TypeName="DupireLocalVolatilityModel",
                                MarketDataSet=cached_mds,
                                FinancingCurveIdentifiers=fin_list,
                                stripped_legacy_fields=model_leg,
                            )
                            model_cache[mkt_key] = cached_model
                    else:
                        raw_model_dict = raw_model_obj if type(raw_model_obj) is dict else dict(raw_model_obj)
                        raw_mkt_key: str | None = None
                        m_bytes = b""
                        try:
                            m_bytes = orjson.dumps(
                                raw_model_dict,
                                option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SORT_KEYS,
                            )
                            raw_mkt_key = hashlib.blake2b(m_bytes, digest_size=16).hexdigest()
                        except (TypeError, ValueError, OverflowError):
                            raw_mkt_key = None
                            m_bytes = b""

                        cached_model = None
                        if raw_mkt_key is not None and raw_mkt_key in raw_model_cache:
                            cand_model, cand_m_nulls = raw_model_cache[raw_mkt_key]
                            if not cand_m_nulls or _check_no_nan_at_null_paths(raw_model_dict, cand_m_nulls):
                                cached_model = cand_model

                        if cached_model is None:
                            if mds_hit_obj is not None:
                                model_copy_in = dict(raw_model_obj)
                                model_copy_in["MarketDataSet"] = mds_hit_obj
                                cached_model = DupireModelSpec.model_validate(model_copy_in)
                            else:
                                cached_model = DupireModelSpec.model_validate(raw_model_obj)
                                eff_mds_key = mds_key_opt or _mds_digest_hex(cached_model.MarketDataSet)
                                try:
                                    object.__setattr__(
                                        cached_model.MarketDataSet,
                                        "_cached_mds_digest_hex",
                                        eff_mds_key,
                                    )
                                except (AttributeError, TypeError):
                                    pass
                                mds_nulls = (
                                    _collect_null_paths(raw_mds_dict)
                                    if (raw_mds_dict is not None and b"null" in mds_bytes)
                                    else ()
                                )
                                if mds_nulls is not None:
                                    mds_cache[eff_mds_key] = (cached_model.MarketDataSet, mds_nulls)
                                mds_only_cache[eff_mds_key] = cached_model.MarketDataSet
                            if raw_mkt_key is not None:
                                m_nulls = _collect_null_paths(raw_model_dict) if b"null" in m_bytes else ()
                                if m_nulls is not None:
                                    raw_model_cache[raw_mkt_key] = (cached_model, m_nulls)
                        mkt_key = raw_mkt_key or _market_dedup_key(
                            EQAutocallableTradeRequest.model_construct(
                                Product=AutoCallableProductSpec.model_construct(
                                    **{}
                                ),
                                Model=cached_model,
                                Queries=[],
                            )
                        )

                    prod_spec = AutoCallableProductSpec.model_validate(item["Product"])
                    raw_queries = (
                        item["Queries"]
                        if "Queries" in item
                        else ("ProductValue", "CashFlows")
                    )
                    if not isinstance(raw_queries, (list, tuple)) or len(raw_queries) == 0:
                        raise ValueError("Queries must not be empty")
                    queries_list = [AutocallQueryMetric(q) for q in raw_queries]
                    raw_trade_info = item.get("TradeInfo")
                    if raw_trade_info is not None and type(raw_trade_info) is not str:
                        raise ValueError(
                            f"TradeInfo must be a string or null, got {type(raw_trade_info).__name__}"
                        )
                    raw_top_stripped = item.get("stripped_legacy_fields")
                    if raw_top_stripped is None:
                        top_stripped: list[str] = []
                    elif isinstance(raw_top_stripped, (list, tuple)):
                        top_stripped = [str(x) for x in raw_top_stripped]
                    else:
                        raise ValueError("stripped_legacy_fields must be a list of strings")
                    merged_stripped = list(
                        dict.fromkeys(
                            [
                                *top_stripped,
                                *cached_model.stripped_legacy_fields,
                            ]
                        )
                    )
                    req_m = EQAutocallableTradeRequest.model_construct(
                        Product=prod_spec,
                        Model=cached_model,
                        Queries=queries_list,
                        TradeInfo=raw_trade_info,
                        stripped_legacy_fields=merged_stripped,
                    )
                    collect_stripped = getattr(req_m, "_collect_nested_stripped_fields", None)
                    if callable(collect_stripped):
                        collect_stripped()
                    if not getattr(cached_model.MarketDataSet, "_batch_arbitrage_checked", False):
                        _validate_surfaces_for_arbitrage(req_m)
                        try:
                            object.__setattr__(cached_model.MarketDataSet, "_batch_arbitrage_checked", True)
                        except (AttributeError, TypeError):
                            pass
                    mkt_key = _market_dedup_key(req_m)
                    model_cache[mkt_key] = cached_model
                else:
                    req_m, _is_dag = self._parse_request(item)
                    mds_hex = _resolve_request_mds_digest_hex(
                        req_m,
                        allow_cached=isinstance(getattr(req_m, "_raw_payload_digest_hex", None), str),
                    )
                    if mds_hex not in mds_only_cache:
                        _validate_surfaces_for_arbitrage(req_m)
                        mds_only_cache[mds_hex] = req_m.Model.MarketDataSet
                    elif req_m.Model.MarketDataSet is not mds_only_cache[mds_hex]:
                        req_m = req_m.model_copy(
                            update={
                                "Model": req_m.Model.model_copy(
                                    update={"MarketDataSet": mds_only_cache[mds_hex]}
                                )
                            }
                        )
                    mkt_key = _market_dedup_key(req_m, mds_digest_hex=mds_hex)
                    if mkt_key not in model_cache:
                        model_cache[mkt_key] = req_m.Model
                    else:
                        req_m = req_m.model_copy(update={"Model": model_cache[mkt_key]})
            else:
                req_m, _is_dag = self._parse_request(item)
                if eff_repair:
                    req_m = _repair_request_surfaces(req_m)
                mds_hex = _resolve_request_mds_digest_hex(
                    req_m,
                    allow_cached=(not eff_repair)
                    and isinstance(getattr(req_m, "_raw_payload_digest_hex", None), str),
                )
                if mds_hex not in mds_only_cache:
                    _validate_surfaces_for_arbitrage(req_m)
                    mds_only_cache[mds_hex] = req_m.Model.MarketDataSet
                elif req_m.Model.MarketDataSet is not mds_only_cache[mds_hex]:
                    req_m = req_m.model_copy(
                        update={
                            "Model": req_m.Model.model_copy(
                                update={"MarketDataSet": mds_only_cache[mds_hex]}
                            )
                        }
                    )
                mkt_key = _market_dedup_key(req_m, mds_digest_hex=mds_hex)
                if mkt_key not in model_cache:
                    model_cache[mkt_key] = req_m.Model
                else:
                    req_m = req_m.model_copy(update={"Model": model_cache[mkt_key]})

            stripped_f = list(req_m.stripped_legacy_fields)
            q_list = [str(getattr(q, "value", q)) for q in req_m.Queries]
            if not compute_greeks:
                q_list = [q for q in q_list if q in ("ProductValue", "CashFlows")]
            parsed_items.append((req_m, stripped_f, q_list, set(q_list), mkt_key))

        results: list[EQAutocallRuntimeResponse | None] = [None] * len(requests)
        buckets: dict[tuple[bool, float, bool], list[tuple[int, AutocallPreprocState, list[AutocallFallbackReason], dict[str, Any]]]] = {}
        pre_state_cache: dict[tuple[Any, ...], tuple[AutocallPreprocState, bool, list[AutocallFallbackReason], Any, dict[str, Any]]] = {}

        if engine_norm in ("auto", "surrogate"):
            surr_d_cap = min(self.d_max, int(getattr(self.surrogate, "d_max", self.d_max)))
            surr_m_cap = min(self.m_max, int(getattr(self.surrogate, "m_max", self.m_max)))
            for idx, (req_m, _stripped, _q_list, q_set, mkt_key) in enumerate(parsed_items):
                if compute_greeks and bool(q_set & full_jacobian_queries):
                    continue
                raw_item = requests[idx]
                if isinstance(raw_item, Mapping) and "Product" in raw_item and isinstance(raw_item["Product"], Mapping):
                    try:
                        prod_bytes = orjson.dumps(
                            dict(raw_item["Product"]),
                            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
                        )
                    except (TypeError, ValueError, OverflowError):
                        prod_bytes = orjson.dumps(
                            req_m.Product.model_dump(mode="json", by_alias=True),
                            option=orjson.OPT_SORT_KEYS,
                        )
                else:
                    prod_bytes = orjson.dumps(
                        req_m.Product.model_dump(mode="json", by_alias=True),
                        option=orjson.OPT_SORT_KEYS,
                    )
                req_cache_key: tuple[Any, ...] = (
                    mkt_key,
                    req_m.TradeInfo is not None,
                    req_m.TradeInfo,
                    hashlib.blake2b(prod_bytes, digest_size=16).digest(),
                )
                cached_pre = pre_state_cache.get(req_cache_key)
                if cached_pre is None:
                    state_i = preprocess_eq_autocall(
                        req_m,
                        d_max=self.d_max,
                        m_max=self.m_max,
                        _fast_shallow_cache=True,
                    )
                    batch_mds_hex = _attach_runtime_market_metadata(
                        state_i,
                        req_m,
                        allow_cached=isinstance(getattr(req_m, "_raw_payload_digest_hex", None), str),
                    )
                    if state_i.baseline is None:
                        state_i.baseline = compute_analytical_autocall_baseline(state_i)
                    if state_i.m_rem == 0 or state_i.seasoned_state.already_autocalled:
                        continue
                    if state_i.d_assets > surr_d_cap or state_i.m_rem > surr_m_cap:
                        continue
                    pre_passed, pre_reasons, r_psd, diag = self.guardian.evaluate_pre_inference(
                        state_i,
                        state_i.baseline,
                        record_drift=False,
                    )
                    if bool(diag.get("correlation_shrunk_applied", diag.get("correlation_shrunk", False))) or (
                        AutocallFallbackReason.CORRELATION_NOT_PSD in pre_reasons
                    ):
                        _ = self.guardian._apply_shrunk_correlation_to_state(state_i, r_psd)
                        _attach_runtime_market_metadata(
                            state_i,
                            req_m,
                            mds_digest_hex=batch_mds_hex,
                        )
                    pre_state_cache[req_cache_key] = (state_i, pre_passed, pre_reasons, r_psd, diag)
                else:
                    state_i, pre_passed, pre_reasons, r_psd, diag = cached_pre
                    if state_i.m_rem == 0 or state_i.seasoned_state.already_autocalled:
                        continue
                    if state_i.d_assets > surr_d_cap or state_i.m_rem > surr_m_cap:
                        continue
                if (not pre_passed) and engine_norm != "surrogate":
                    continue
                mod_key = (
                    bool(state_i.no_notional_repayment),
                    round(float(state_i.autocall_coupon), 12),
                    bool(state_i.force_coupons_on_autocall),
                )
                buckets.setdefault(mod_key, []).append(
                    (idx, state_i, pre_reasons, diag)
                )

        for (no_notional_flag, ac_cpn, force_cpn_flag), entries in buckets.items():
            t0_batch = time.perf_counter()
            b_indices = [e[0] for e in entries]
            b_states = [e[1] for e in entries]
            b_pre_reasons = [e[2] for e in entries]
            b_pre_diags = [e[3] for e in entries]

            need_batched_greeks = bool(
                compute_greeks
                and any(
                    bool(
                        parsed_items[i][3]
                        & {"Delta", "MultivariateAggregatedCashGamma", "Cega", "ConvexCega"}
                    )
                    for i in b_indices
                )
            )
            need_cg = bool(
                need_batched_greeks
                and any("MultivariateAggregatedCashGamma" in parsed_items[i][3] for i in b_indices)
            )
            b_surr = len(b_states)
            d_max = self.d_max
            m_max = self.m_max
            node_b = torch.cat([s.node_feats for s in b_states], dim=0)
            edge_b = torch.cat([s.edge_feats for s in b_states], dim=0)
            amask_b = torch.cat([s.asset_mask for s in b_states], dim=0)
            sched_b = torch.cat([s.sched_feats for s in b_states], dim=0)
            smask_b = torch.cat([s.sched_mask for s in b_states], dim=0)
            glob_b = torch.cat([s.global_feats for s in b_states], dim=0)

            spots_b = torch.ones((b_surr, d_max), dtype=torch.float64)
            refs_b = torch.ones((b_surr, d_max), dtype=torch.float64)
            pvit_b = torch.zeros(b_surr, dtype=torch.float64)
            ilb_b = torch.zeros(b_surr, dtype=torch.float64)
            lstrike_b = torch.zeros(b_surr, dtype=torch.float64)
            has_any_funding = any(s.funding_leg is not None for s in b_states)
            w_fund_b = torch.zeros((b_surr, m_max), dtype=torch.float64) if has_any_funding else None
            w_fund_cache: dict[str, torch.Tensor] = {}
            for k, s in enumerate(b_states):
                d_act = int(s.d_assets)
                spots_b[k, :d_act] = s.spots_t0.detach().cpu().to(dtype=torch.float64)
                refs_b[k, :d_act] = s.ref_spots.detach().cpu().to(dtype=torch.float64)
                pvit_b[k] = float(s.seasoned_state.pv_in_transit)
                ilb_b[k] = 1.0 if bool(s.is_long_basket) else 0.0
                lstrike_b[k] = float(s.lower_strike) if s.lower_strike is not None else 0.0
                if w_fund_b is not None and s.funding_leg is not None:
                    s_dig = getattr(s, "state_digest", None)
                    if isinstance(s_dig, str) and s_dig:
                        w_vec = w_fund_cache.get(s_dig)
                        if w_vec is None:
                            w_vec = _compute_w_fund_per_period(s, m_max)
                            w_fund_cache[s_dig] = w_vec
                    else:
                        w_vec = _compute_w_fund_per_period(s, m_max)
                    w_fund_b[k, :] = w_vec

            batch_out = self.price_tensor_batch(
                node_b,
                edge_b,
                amask_b,
                sched_b,
                smask_b,
                glob_b,
                spots_t0=spots_b,
                ref_spots=refs_b,
                pv_in_transit=pvit_b,
                compute_greeks=need_batched_greeks,
                compute_cash_gamma=need_cg,
                no_notional_repayment=no_notional_flag,
                autocall_coupon=ac_cpn,
                force_coupons_on_autocall=force_cpn_flag,
                is_long_basket=ilb_b,
                w_fund_per_period=w_fund_b,
                lower_strike=lstrike_b,
                run_guardian=False,
            )
            per_trade_ms = ((time.perf_counter() - t0_batch) * 1e3) / max(b_surr, 1)

            stop_np_b = batch_out.stopping_probs.detach().cpu().numpy()
            cpn_np_b = batch_out.coupon_probs.detach().cpu().numpy()
            put_np_b = batch_out.put_loss.detach().cpu().numpy()
            stop_t = batch_out.stopping_probs.detach().cpu()
            cpn_t = batch_out.coupon_probs.detach().cpu()
            put_t = batch_out.put_loss.detach().cpu()
            delta_np = batch_out.delta.detach().cpu().numpy()
            cega_np = batch_out.cega.detach().cpu().numpy()
            cg_np = (
                batch_out.cash_gamma_euler.detach().cpu().numpy()
                if batch_out.cash_gamma_euler is not None
                else np.zeros(b_surr, dtype=np.float64)
            )
            g_engine_label = "Tier2_BatchedAD" if need_batched_greeks else "None"
            post_cache: dict[
                tuple[Any, bytes, bytes, float, bool],
                tuple[Any, bool, list[AutocallFallbackReason], list[CashFlowRow]],
            ] = {}
            ts_now = datetime.now(timezone.utc).isoformat()
            audit_batch_append: list[SR262AuditRecord] = []

            for k, orig_idx in enumerate(b_indices):
                req_m, stripped_f, q_list, q_set, _mkt_k = parsed_items[orig_idx]
                state_k = b_states[k]
                pre_reasons_k = b_pre_reasons[k]
                pre_diag_k = b_pre_diags[k]
                m_rem = int(state_k.m_rem)
                d_act = int(state_k.d_assets)
                orig_ids = state_k.original_identifiers
                inv_perm = state_k.inv_lex_perm
                want_cfs = "CashFlows" in q_set

                st_key = getattr(state_k, "state_digest", None) or k
                post_key = (
                    st_key,
                    stop_np_b[k, :m_rem].tobytes(),
                    cpn_np_b[k, :m_rem].tobytes(),
                    float(put_np_b[k]),
                    want_cfs,
                )
                cached_post = post_cache.get(post_key)
                if cached_post is None:
                    post_res = assemble_autocall_cashflows_and_pv(
                        state_k,
                        p_stop=stop_t[k, :m_rem],
                        q_cpn=cpn_t[k, :m_rem],
                        l_put=put_t[k],
                        build_cashflow_items=want_cfs,
                    )
                    post_passed, post_reasons_k = self.guardian.evaluate_post_inference(
                        {
                            "stopping_probs": stop_t[k, :m_rem],
                            "coupon_probs": cpn_t[k, :m_rem],
                            "put_loss": put_t[k],
                        },
                        post_res,
                        state_k,
                    )
                    cf_rows_k = (
                        [
                            CashFlowRow.model_construct(
                                Description=str(cf.Description),
                                Date=cf.Date,
                                Amount=float(cf.Amount),
                                StandardError=float(cf.StandardError),
                                Currency=str(cf.Currency),
                                Leg=int(cf.Leg),
                                StatusText=str(cf.StatusText),
                                ShortCode=_DESCRIPTION_TO_SHORT_CODE.get(str(cf.Description)),
                            )
                            for cf in post_res.cashflows
                        ]
                        if want_cfs
                        else []
                    )
                    post_cache[post_key] = (post_res, post_passed, post_reasons_k, cf_rows_k)
                else:
                    post_res, post_passed, post_reasons_k, cf_rows_k = cached_post

                if not post_passed:
                    results[orig_idx] = self._build_qmc_fallback_response(
                        req_m,
                        state_k,
                        stripped_f,
                        q_list,
                        compute_greeks=compute_greeks,
                        t0=t0_batch,
                        pre_reasons=pre_reasons_k,
                        post_reasons=post_reasons_k,
                        pre_diag=pre_diag_k,
                        theta_convention=eff_theta_conv,
                    )
                    continue

                fb_reasons_enum = [*pre_reasons_k, *post_reasons_k]
                fb_reasons = [r.value for r in fb_reasons_enum]
                fb_triggered_k = False if engine_norm == "surrogate" else bool(len(fb_reasons) > 0)
                min_pin_z = float(pre_diag_k.get("min_pin_risk_z", 100.0))
                conf_hw = float(pre_diag_k.get("conformal_half_width_bps", 0.0))
                audit_rec = SR262AuditRecord(
                    trade_id=req_m.TradeInfo if req_m.TradeInfo is not None else (state_k.trade_info if state_k.trade_info is not None else f"BATCH_{orig_idx}"),
                    timestamp_utc=ts_now,
                    pre_gate_passed=(len(pre_reasons_k) == 0),
                    post_gate_passed=bool(post_passed),
                    fallback_triggered=fb_triggered_k,
                    fallback_reasons=fb_reasons,
                    warnings=list(pre_diag_k.get("warnings", [])),
                    min_corr_eigenvalue_raw=float(pre_diag_k.get("min_corr_eigenvalue_raw", 1.0)),
                    min_corr_eigenvalue_shrunk=float(pre_diag_k.get("min_corr_eigenvalue_shrunk", 1.0)),
                    correlation_shrunk_applied=bool(
                        pre_diag_k.get("correlation_shrunk_applied", pre_diag_k.get("correlation_shrunk", False))
                    ),
                    mahalanobis_d2=float(pre_diag_k.get("mahalanobis_d2", 0.0)),
                    max_psi_score=float(pre_diag_k.get("max_psi_score", 0.0)),
                    min_pin_risk_z=min_pin_z,
                    engine_selected="MaskedEGNO_Surrogate",
                    conformal_half_width_bps=conf_hw,
                    greeks_engine_used=g_engine_label,
                )
                audit_batch_append.append(audit_rec)

                primary_fb_reason = fb_reasons[0] if len(fb_reasons) > 0 else None
                delta_out_k: dict[str, float] | None = None
                if "Delta" in q_set:
                    delta_orig = delta_np[k, :d_act][inv_perm]
                    delta_out_k = {
                        orig_ids[i]: float(delta_orig[i]) for i in range(d_act)
                    }
                    for fx_id in _collect_quanto_fx_ids(state_k):
                        delta_out_k.setdefault(fx_id, 0.0)
                cg_out_k = float(cg_np[k]) if "MultivariateAggregatedCashGamma" in q_set else None
                cega_out_k: _CegaDict | None = None
                if "Cega" in q_set or "ConvexCega" in q_set:
                    cega_orig = cega_np[k, :d_act, :d_act][np.ix_(inv_perm, inv_perm)]
                    cega_out_k = _CegaDict(
                        {
                            orig_ids[i]: {
                                orig_ids[j]: float(cega_orig[i, j]) for j in range(d_act)
                            }
                            for i in range(d_act)
                        }
                    )
                results[orig_idx] = _assemble_runtime_response(
                    product_value=float(post_res.product_value.detach().cpu().item()),
                    reporting_currency=post_res.reporting_currency,
                    standard_error=0.0,
                    cash_to_settle_today=float(post_res.cash_to_settle_today),
                    cash_flows=list(cf_rows_k),
                    delta=delta_out_k,
                    gamma=None,
                    cash_gamma=cg_out_k,
                    vega=None,
                    rho=None,
                    repo_rho=None,
                    dividends_sensitivities=None,
                    theta=None,
                    cega=cega_out_k,
                    trade_info=req_m.TradeInfo,
                    engine_used="MaskedEGNO_Surrogate",
                    fallback_triggered=fb_triggered_k,
                    fallback_reason=primary_fb_reason,
                    fallback_reasons=fb_reasons,
                    stripped_legacy_fields=stripped_f,
                    option=float(post_res.option_pv.detach().cpu().item()),
                    notional=float(post_res.notional_pv.detach().cpu().item()),
                    coupon=float(post_res.coupon_pv.detach().cpu().item()),
                    funding_leg=float(-post_res.funding_pv.detach().cpu().item()),
                    audit_record=audit_rec.to_dict(),
                    latency_ms=round(float(per_trade_ms), 4),
                )

            if audit_batch_append:
                with self.guardian._lock:
                    self.guardian.audit_log.extend(audit_batch_append)

        for idx, (req_m, stripped_f, q_list, _q_set, _mkt_k) in enumerate(parsed_items):
            if results[idx] is None:
                req_call = req_m
                if not compute_greeks:
                    filt_q = [AutocallQueryMetric(q) for q in q_list] or [
                        AutocallQueryMetric("ProductValue")
                    ]
                    req_call = req_m.model_copy(update={"Queries": filt_q})
                res_i = self.price_trade(
                    req_call,
                    engine=engine_norm,
                    include_all_greeks=False,
                    theta_convention=eff_theta_conv,
                    repair_arbitrage=False,
                )
                if stripped_f and not res_i.stripped_legacy_fields:
                    res_i = res_i.model_copy(update={"stripped_legacy_fields": stripped_f})
                results[idx] = res_i

        return [r for r in results if r is not None]

    @staticmethod
    def _parse_arrow_meta_float(meta: Mapping[str, Any], key: str, default: float) -> float:
        raw = meta.get(key)
        if raw is None:
            return float(default)
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"Arrow metadata field '{key}' must be finite float, got bool.")
        try:
            val = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Arrow metadata field '{key}' must be numeric, got {raw!r}.") from exc
        if not math.isfinite(val):
            raise ValueError(f"Arrow metadata field '{key}' must be finite, got {raw!r}.")
        return val

    @staticmethod
    def _parse_arrow_meta_bool(meta: Mapping[str, Any], key: str, default: bool) -> bool:
        raw = meta.get(key)
        if raw is None:
            return bool(default)
        if isinstance(raw, (bool, np.bool_)):
            return bool(raw)
        if isinstance(raw, str):
            norm = raw.strip().lower()
            if norm in ("true", "1", "yes"):
                return True
            if norm in ("false", "0", "no"):
                return False
            raise ValueError(f"Arrow metadata field '{key}' must be boolean, got {raw!r}.")
        if isinstance(raw, int) and raw in (0, 1):
            return bool(raw)
        raise ValueError(f"Arrow metadata field '{key}' must be boolean, got {raw!r}.")

    def price_arrow_ipc(
        self,
        body: bytes,
        compute_greeks: bool = True,
    ) -> bytes:
        """Decodes an Apache Arrow IPC request, enforces schema bounds, runs `price_tensor_batch`, and encodes Arrow IPC."""
        t0 = time.perf_counter()
        decoded: DecodedEQAutocallArrowBatch = decode_eq_autocall_arrow_request(body)
        if decoded.d_max > self.d_max or decoded.m_max > self.m_max:
            raise ValueError(
                f"Arrow batch dimensions (d_max={decoded.d_max}, m_max={decoded.m_max}) "
                f"exceed runtime capacity ({self.d_max}, {self.m_max})."
            )
        pvit: float | torch.Tensor = (
            decoded.pv_in_transit
            if decoded.pv_in_transit is not None
            else self._parse_arrow_meta_float(decoded.metadata, "pv_in_transit", 0.0)
        )
        no_notional = self._parse_arrow_meta_bool(
            decoded.metadata, "no_notional_repayment", False
        )
        ac_coupon = self._parse_arrow_meta_float(decoded.metadata, "autocall_coupon", 0.0)
        force_cpn = self._parse_arrow_meta_bool(
            decoded.metadata, "force_coupons_on_autocall", False
        )
        fund_pv = self._parse_arrow_meta_float(decoded.metadata, "funding_pv", 0.0)
        batch_out = self.price_tensor_batch(
            decoded.node_feats,
            decoded.edge_feats,
            decoded.asset_mask,
            decoded.sched_feats,
            decoded.sched_mask,
            decoded.global_feats,
            spots_t0=decoded.spots_t0,
            ref_spots=decoded.ref_spots,
            pv_in_transit=pvit,
            compute_greeks=compute_greeks,
            compute_cash_gamma=compute_greeks,
            no_notional_repayment=no_notional,
            autocall_coupon=ac_coupon,
            force_coupons_on_autocall=force_cpn,
            funding_pv=fund_pv,
            run_guardian=True,
        )
        for tensor_name in (
            "product_value",
            "stopping_probs",
            "survival_probs",
            "coupon_probs",
            "put_loss",
            "delta",
            "vega_atm",
            "vega_surface_feats",
            "rho_df",
            "cega",
            "cash_gamma_euler",
        ):
            t_val = getattr(batch_out, tensor_name, None)
            if t_val is not None and not bool(torch.isfinite(t_val).all()):
                raise ValueError(f"Non-finite {tensor_name} produced for Arrow IPC batch.")
        resolved_engine = str(getattr(batch_out, "_engine_selected", "MaskedEGNO_Surrogate"))
        latency_ms = (time.perf_counter() - t0) * 1e3
        return encode_eq_autocall_arrow_response(
            batch_out,
            latency_ms=latency_ms,
            engine_used=resolved_engine,
            metadata_dict=decoded.metadata,
        )


__all__ = [
    "ALL_8_GREEK_QUERIES",
    "CUDA_MICROBATCH_LIMIT_M48",
    "EngineMode",
    "EQAutocallRuntimeResponse",
    "EQAutocallRuntime",
    "MAX_BATCH_TRADES",
    "to_decimal_merged_rho",
]
