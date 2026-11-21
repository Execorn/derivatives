"""greeks_engine.py -- End-to-End Differentiable Chain-Rule Greeks Assembler.

Implements decoupled float64/float32 single-VJP surrogate AD across native `.to()` cast
boundaries, Forward-over-Reverse `jvp(grad)` exact D x D Gamma Hessians, single-tangent
Euler CashGamma, multi-curve Funding + Repo Rho, 1-day calendar-roll Theta, live quanto
and FX sensitivities, external bounded LRU caches, extended batched surrogate Greeks, and
Common Random Numbers (CRN) QMC Greeks (`compute_qmc_crn_greeks`).
"""

from __future__ import annotations

from collections import OrderedDict
import contextlib
import copy
import ctypes
from dataclasses import dataclass, field, fields, replace
import datetime as dt
import hashlib
import inspect
import math
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Callable, Mapping, Sequence, cast
import warnings
import zlib

import numpy as np
import numpy.typing as npt
import torch

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
    from triton.language.extra.cuda import libdevice as _triton_libdevice  # type: ignore[import-untyped]

    _HAS_TRITON_TIER1 = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _triton_libdevice = None
    _HAS_TRITON_TIER1 = False

from deepvol.market.eq_curves import ClampedYieldCurve
from deepvol.models.autocall.analytical_baseline import (
    _BALANCED_TREES_D5,
    AutocallBaselineOutput,
    _black76_put_torch,
    _clark_symmetrized_torch,
    _digital_call_spread_prob_torch,
    _euler_next_corr_torch,
    _eval_coupon_prob_torch,
    _exact_piecewise_terminal_put_torch,
    _ramp_survival_post_moments_lognormal_torch,
    _truncated_upper_std_moments_torch,
    compute_analytical_autocall_baseline,
    stulz_margrabe_moments_with_weights,
)
from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCConfig, DupireQMCGpuSolver
from deepvol.models.autocall.dupire_qmc_solver import RegularizedVolSurface
from deepvol.models.autocall.feature_layout import (
    AutocallEdgeCol,
    AutocallGlobalCol,
    AutocallNodeCol,
    AutocallSchedCol,
    PAIR_ORDER_DEPENDENT_INDICES,
    monitoring_index_to_code,
)
from deepvol.models.autocall.pre_post_processor import (
    AutocallPostprocOutput,
    AutocallPreprocState,
    ThreadSafeLRUCache,
    assemble_autocall_cashflows_and_pv,
    digest_market_bytes,
    preprocess_eq_autocall,
)
from deepvol.models.autocall.seasoned_collapser import CashFlowItem, _year_fraction
from deepvol.surrogates.eq_autocall_egno import (
    AutocallSurrogateOutput,
    MaskedEQAutocallSurrogate,
    assemble_surrogate_product_value,
)


def cast_to_surrogate_boundary(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Casts a Tier-1 IEEE-754 float64 tensor across the surrogate precision boundary."""
    return tensor.to(device=device, dtype=dtype)


def cast_from_surrogate_boundary(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Casts Tier-2 surrogate residual outputs back to Tier-1 float64 precision."""
    return tensor.to(device=device, dtype=dtype)


@dataclass
class AutocallGreeksResult:
    """Complete valuation, cash-flow schedule, and 8-Greek sensitivity bundle in float64."""

    product_value: float
    cash_flows: list[Any]
    delta: dict[str, float] | None = None
    gamma: dict[str, dict[str, float]] | None = None
    multivariate_aggregated_cash_gamma: float | None = None
    vega: dict[str, list[list[float]]] | None = None
    rho: dict[str, list[float]] | None = None
    dividends_sensitivities: dict[str, list[float]] | None = None
    theta: float | None = None
    cega: dict[str, dict[str, float]] | None = None

    # Tensor & NumPy float64 representations in original caller asset order (pi_lex^{-1})
    delta_tensor: torch.Tensor | None = None
    gamma_tensor: torch.Tensor | None = None
    vega_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    rho_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    dividends_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    cega_tensor: torch.Tensor | None = None

    reporting_currency: str = "RUB"
    standard_error: float = 0.0
    cash_to_settle_today: float = 0.0
    postproc_output: AutocallPostprocOutput | None = field(default=None, repr=False)
    repo_rho: dict[str, list[float]] | None = None
    fx_vega: dict[str, list[float]] | None = None
    fx_cega: dict[str, dict[str, float]] | None = None
    greeks_engine_used: str = "tier3_ad"

    @property
    def cashflows(self) -> list[Any]:
        """Alias for `cash_flows`."""
        return self.cash_flows


@dataclass
class BatchedSurrogateGreeksOutput:
    """Vectorized batched valuation and first-order Greek sensitivities in torch.float64."""

    product_value: torch.Tensor
    stopping_probs: torch.Tensor
    survival_probs: torch.Tensor
    coupon_probs: torch.Tensor
    put_loss: torch.Tensor
    delta: torch.Tensor
    vega_atm: torch.Tensor
    vega_surface_feats: torch.Tensor
    rho_df: torch.Tensor
    cega: torch.Tensor
    cash_gamma_euler: torch.Tensor | None = None


@dataclass(frozen=True)
class _SurfaceGeomCache:
    """Cached grid geometry and spline basis tensors (invariant across quote changes)."""

    strikes: torch.Tensor
    tenors: torch.Tensor
    basis_c: torch.Tensor
    slope_left: torch.Tensor
    slope_right: torch.Tensor
    e_first: torch.Tensor
    e_last: torch.Tensor
    dedup_t: torch.Tensor
    cal_proj_jac: torch.Tensor | None
    k_min: float
    k_max: float
    n_unique: int
    n_exp: int
    strikes_np: npt.NDArray[np.float64] | None = None
    tenors_np: npt.NDArray[np.float64] | None = None
    basis_c_np: npt.NDArray[np.float64] | None = None
    slope_left_np: npt.NDArray[np.float64] | None = None
    slope_right_np: npt.NDArray[np.float64] | None = None
    dedup_t_np: npt.NDArray[np.float64] | None = None
    cal_proj_jac_np: npt.NDArray[np.float64] | None = None
    n_raw: int = 0
    strikes_ptr: int = 0
    tenors_ptr: int = 0
    basis_c_ptr: int = 0
    slope_left_ptr: int = 0
    slope_right_ptr: int = 0
    dedup_t_ptr: int = 0
    cal_proj_jac_ptr: int = 0


@dataclass(frozen=True)
class _SurfaceQuoteCache:
    """Cached quote values and per-slice polynomial coefficients."""

    quotes: torch.Tensor
    slice_c: torch.Tensor | None = None
    slice_slope_left: torch.Tensor | None = None
    slice_slope_right: torch.Tensor | None = None
    quotes_np: npt.NDArray[np.float64] | None = None
    slice_c_np: npt.NDArray[np.float64] | None = None
    slope_left_np: npt.NDArray[np.float64] | None = None
    slope_right_np: npt.NDArray[np.float64] | None = None
    quotes_ptr: int = 0
    slice_c_ptr: int = 0
    slope_left_ptr: int = 0
    slope_right_ptr: int = 0


# External bounded registries keep cached tensors off shared market objects (thread-safe).
_SURFACE_GEOM_CACHE = ThreadSafeLRUCache(maxsize=512)
_SURFACE_QUOTE_CACHE = ThreadSafeLRUCache(maxsize=512)
_CURVE_DELTA_CACHE = ThreadSafeLRUCache(maxsize=512)
_CURVE_JAC_CACHE = ThreadSafeLRUCache(maxsize=512)
_THETA_CURVE_CACHE = ThreadSafeLRUCache(maxsize=512)
_RAW_CURVE_CACHE = ThreadSafeLRUCache(maxsize=256)
_SURROGATE_CUDAGRAPH_REGISTRY = ThreadSafeLRUCache(maxsize=64)
_SURR_MASK_REGISTRY = ThreadSafeLRUCache(maxsize=128)
_COMPILE_LOCK = threading.Lock()
_CUDA_GRAPH_LOCK = threading.RLock()


def _in_functorch_transform() -> bool:
    """Returns True when executing inside a functorch vmap/jvp/grad dynamic interpreter."""
    fc_mod = getattr(torch._C, "_functorch", None)
    peek_fn = getattr(fc_mod, "peek_interpreter_stack", None)
    if callable(peek_fn):
        try:
            return peek_fn() is not None
        except Exception:
            return False
    return False

_ASSEMBLE_SURR_PARAMS: frozenset[str] = frozenset(
    inspect.signature(assemble_surrogate_product_value).parameters.keys()
)
_SURR_FWD_PARAMS: frozenset[str] = frozenset(
    inspect.signature(MaskedEQAutocallSurrogate.forward).parameters.keys()
)


def _normalize_curve_obj(
    curve: Any,
    *,
    as_of_date: dt.datetime | dt.date | None = None,
) -> Any:
    """Converts a raw dict curve specification into a cached ClampedYieldCurve if needed."""
    if curve is None or not isinstance(curve, Mapping):
        return curve
    raw_digest = digest_market_bytes("raw_curve", curve)
    as_of_key = as_of_date.isoformat() if isinstance(as_of_date, (dt.datetime, dt.date)) else str(as_of_date)
    cache_key = (raw_digest, as_of_key)
    if raw_digest is not None:
        cached = _RAW_CURVE_CACHE.get(cache_key)
        if cached is not None:
            return cached
    built = ClampedYieldCurve(curve, as_of_date=as_of_date)
    if raw_digest is not None:
        _RAW_CURVE_CACHE.put(cache_key, built)
    return built


def _fast_array_digest(*parts: Any) -> str:
    """Fast C-level 64-bit CRC32 + shape/scalar mix over NumPy buffers and metadata."""
    c1 = 0
    c2 = 0x9E3779B9
    crc = zlib.crc32
    for part in parts:
        if part is None:
            c2 = (c2 * 1664525 + 1013904223) & 0xFFFFFFFF
        elif isinstance(part, np.ndarray):
            arr_c = part if part.flags.c_contiguous else np.ascontiguousarray(part)
            c1 = crc(cast(Any, arr_c), c1)
            c2 = (c2 * 1664525 + c1 + arr_c.size + arr_c.ndim * 131071) & 0xFFFFFFFF
        elif isinstance(part, (tuple, list)):
            for sub in part:
                if isinstance(sub, np.ndarray):
                    arr_c = sub if sub.flags.c_contiguous else np.ascontiguousarray(sub)
                    c1 = crc(cast(Any, arr_c), c1)
                    c2 = (c2 * 1664525 + c1 + arr_c.size) & 0xFFFFFFFF
                else:
                    c2 = (c2 * 1664525 + hash(sub)) & 0xFFFFFFFF
        else:
            c2 = (c2 * 1664525 + hash(part)) & 0xFFFFFFFF
    return f"{c1 & 0xFFFFFFFF:08x}{c2 & 0xFFFFFFFF:08x}"


def _get_surface_torch_caches(
    surf: RegularizedVolSurface,
    device: torch.device,
) -> tuple[_SurfaceGeomCache, _SurfaceQuoteCache]:
    """Retrieves split geometry and quote torch.float64 caches from external bounded registries."""
    dev_key = (device.type, device.index)
    cal_digest = str(getattr(surf, "calendar_proj_digest", "") or "")
    was_proj = bool(getattr(surf, "was_projected", False))
    geom_digest = _fast_array_digest(
        getattr(surf, "raw_strikes", surf.strikes),
        surf.strikes,
        surf.tenors,
        surf.dedup_matrix,
        str(getattr(surf, "bc_type", "")),
        str(getattr(surf, "interp_method", "")),
        str(getattr(surf, "extrap_method", "")),
    )
    geom_key = (geom_digest, cal_digest, was_proj, dev_key)

    geom = _SURFACE_GEOM_CACHE.get(geom_key)
    if geom is None:
        surf._ensure_basis_spline()
        if (
            surf._fast_basis_coeffs is None
            or surf._basis_slope_left is None
            or surf._basis_slope_right is None
        ):
            raise RuntimeError("RegularizedVolSurface basis spline failed to initialize.")

        n_u = int(surf.n_unique_strikes)
        e_first = torch.zeros(n_u, dtype=torch.float64, device=device)
        e_first[0] = 1.0
        e_last = torch.zeros(n_u, dtype=torch.float64, device=device)
        e_last[-1] = 1.0

        cal_jac_np = getattr(surf, "calendar_proj_jacobian", None)
        cal_jac_t = (
            torch.tensor(np.array(cal_jac_np, copy=True), dtype=torch.float64, device=device)
            if cal_jac_np is not None
            else None
        )

        strikes_np = np.ascontiguousarray(surf.strikes, dtype=np.float64)
        tenors_np = np.ascontiguousarray(surf.tenors, dtype=np.float64)
        basis_c_np = np.ascontiguousarray(surf._fast_basis_coeffs, dtype=np.float64)
        slope_left_np = np.ascontiguousarray(surf._basis_slope_left, dtype=np.float64)
        slope_right_np = np.ascontiguousarray(surf._basis_slope_right, dtype=np.float64)
        dedup_t_np = np.ascontiguousarray(surf.dedup_matrix.T, dtype=np.float64)
        cal_proj_jac_np = (
            np.ascontiguousarray(cal_jac_np, dtype=np.float64) if cal_jac_np is not None else None
        )
        # Copy frozen NumPy buffers explicitly so PyTorch emits zero non-writable warnings.
        geom = _SurfaceGeomCache(
            strikes=torch.tensor(strikes_np.copy(), dtype=torch.float64, device=device),
            tenors=torch.tensor(tenors_np.copy(), dtype=torch.float64, device=device),
            basis_c=torch.tensor(basis_c_np.copy(), dtype=torch.float64, device=device),
            slope_left=torch.tensor(slope_left_np.copy(), dtype=torch.float64, device=device),
            slope_right=torch.tensor(slope_right_np.copy(), dtype=torch.float64, device=device),
            e_first=e_first,
            e_last=e_last,
            dedup_t=torch.tensor(dedup_t_np.copy(), dtype=torch.float64, device=device),
            cal_proj_jac=cal_jac_t,
            k_min=float(surf.strikes[0]),
            k_max=float(surf.strikes[-1]),
            n_unique=n_u,
            n_exp=int(surf.n_exp),
            strikes_np=strikes_np,
            tenors_np=tenors_np,
            basis_c_np=basis_c_np,
            slope_left_np=slope_left_np,
            slope_right_np=slope_right_np,
            dedup_t_np=dedup_t_np,
            cal_proj_jac_np=cal_proj_jac_np,
            n_raw=int(surf.n_raw_strikes),
            strikes_ptr=int(strikes_np.__array_interface__["data"][0]),
            tenors_ptr=int(tenors_np.__array_interface__["data"][0]),
            basis_c_ptr=int(basis_c_np.__array_interface__["data"][0]),
            slope_left_ptr=int(slope_left_np.__array_interface__["data"][0]),
            slope_right_ptr=int(slope_right_np.__array_interface__["data"][0]),
            dedup_t_ptr=int(dedup_t_np.__array_interface__["data"][0]),
            cal_proj_jac_ptr=int(cal_proj_jac_np.__array_interface__["data"][0]) if cal_proj_jac_np is not None else 0,
        )
        if geom_digest is not None and not _in_functorch_transform():
            _SURFACE_GEOM_CACHE.put(geom_key, geom)

    quote_digest = _fast_array_digest(
        surf.raw_quotes,
        surf.quotes,
        float(surf.min_vol),
        float(getattr(surf, "max_local_vol", 2.5)),
    )
    quote_key = (geom_digest, quote_digest, cal_digest, was_proj, dev_key)
    qcache = _SURFACE_QUOTE_CACHE.get(quote_key)
    if qcache is None:
        surf._ensure_basis_spline()
        slice_c_np_raw = getattr(surf, "_fast_slice_coeffs", None)
        slice_sl_np_raw = getattr(surf, "_slopes_left", None)
        slice_sr_np_raw = getattr(surf, "_slopes_right", None)
        quotes_np = np.ascontiguousarray(surf.quotes, dtype=np.float64)
        slice_c_np = np.ascontiguousarray(slice_c_np_raw, dtype=np.float64) if slice_c_np_raw is not None else None
        slope_l_np = np.ascontiguousarray(slice_sl_np_raw, dtype=np.float64) if slice_sl_np_raw is not None else None
        slope_r_np = np.ascontiguousarray(slice_sr_np_raw, dtype=np.float64) if slice_sr_np_raw is not None else None
        qcache = _SurfaceQuoteCache(
            quotes=torch.tensor(quotes_np.copy(), dtype=torch.float64, device=device),
            slice_c=(
                torch.tensor(slice_c_np.copy(), dtype=torch.float64, device=device)
                if slice_c_np is not None
                else None
            ),
            slice_slope_left=(
                torch.tensor(slope_l_np.copy(), dtype=torch.float64, device=device)
                if slope_l_np is not None
                else None
            ),
            slice_slope_right=(
                torch.tensor(slope_r_np.copy(), dtype=torch.float64, device=device)
                if slope_r_np is not None
                else None
            ),
            quotes_np=quotes_np,
            slice_c_np=slice_c_np,
            slope_left_np=slope_l_np,
            slope_right_np=slope_r_np,
            quotes_ptr=int(quotes_np.__array_interface__["data"][0]),
            slice_c_ptr=int(slice_c_np.__array_interface__["data"][0]) if slice_c_np is not None else 0,
            slope_left_ptr=int(slope_l_np.__array_interface__["data"][0]) if slope_l_np is not None else 0,
            slope_right_ptr=int(slope_r_np.__array_interface__["data"][0]) if slope_r_np is not None else 0,
        )
        if geom_digest is not None and quote_digest is not None and not _in_functorch_transform():
            _SURFACE_QUOTE_CACHE.put(quote_key, qcache)

    return geom, qcache


def _bucket_idx_torch(grid: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Fast O(log N) bucket index via torch.searchsorted (requires contiguous 1D sorted grid)."""
    n_grid = int(grid.shape[0])
    idx = torch.searchsorted(grid, queries.contiguous(), right=True) - 1
    return torch.clamp(idx, min=0, max=max(n_grid - 2, 0))


def _strike_basis_u_batch_torch(
    geom: _SurfaceGeomCache,
    k_queries: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch vectorized strike spline basis w_u(K) in R^{K x N_unique} (functorch-safe)."""
    n_u = geom.n_unique
    if n_u < 2:
        return torch.ones(
            (k_queries.shape[0], max(n_u, 1)), dtype=torch.float64, device=k_queries.device
        )

    seg_idx = _bucket_idx_torch(geom.strikes, k_queries)
    k_left = geom.strikes[seg_idx]
    dx = (k_queries - k_left).unsqueeze(-1)
    c_seg = geom.basis_c[:, seg_idx, :]  # (4, K, N_unique)
    w_in = ((c_seg[0] * dx + c_seg[1]) * dx + c_seg[2]) * dx + c_seg[3]

    w_left = geom.e_first.unsqueeze(0) + (k_queries - geom.k_min).unsqueeze(-1) * geom.slope_left.unsqueeze(0)
    w_right = geom.e_last.unsqueeze(0) + (k_queries - geom.k_max).unsqueeze(-1) * geom.slope_right.unsqueeze(0)

    w_in = torch.where((k_queries < geom.k_min).unsqueeze(-1), w_left, w_in)
    w_in = torch.where((k_queries > geom.k_max).unsqueeze(-1), w_right, w_in)
    return w_in


def _project_surface_quote_delta(
    quote_delta: torch.Tensor,
    geom: _SurfaceGeomCache,
) -> torch.Tensor:
    """Projects raw surface quote deltas through deduplication and calendar projection Jacobians."""
    # dedup_matrix has shape (N_unique, N_raw); dedup_t is (N_raw, N_unique).
    delta_dedup = torch.matmul(quote_delta, geom.dedup_t)
    if geom.cal_proj_jac is not None:
        return torch.einsum("mnu,nu->mu", geom.cal_proj_jac, delta_dedup)
    return delta_dedup


def _eval_reg_surface_iv_batch_torch(
    surf: RegularizedVolSurface,
    t_queries: npt.NDArray[np.float64] | torch.Tensor,
    k_queries: torch.Tensor,
    quote_delta: torch.Tensor | None = None,
    *,
    cached_pair: tuple[_SurfaceGeomCache, _SurfaceQuoteCache] | None = None,
    delta_prefix_len: int | None = None,
    allow_cpp: bool = True,
) -> torch.Tensor:
    """Pure-PyTorch torch.float64 evaluation of sigma_IV(t_j, K_j) across K query pairs."""
    device = k_queries.device
    geom, qcache = cached_pair if cached_pair is not None else _get_surface_torch_caches(surf, device)
    t_vals = (
        t_queries.to(dtype=torch.float64, device=device)
        if isinstance(t_queries, torch.Tensor)
        else torch.tensor(np.asarray(t_queries, dtype=np.float64), dtype=torch.float64, device=device)
    ).clamp(min=1e-8)

    proj_q_delta = (
        _project_surface_quote_delta(quote_delta, geom) if quote_delta is not None else None
    )
    n_q = int(k_queries.shape[0])
    d_len = n_q if delta_prefix_len is None else min(max(int(delta_prefix_len), 0), n_q)

    def _add_delta_prefix(base_t: torch.Tensor, shift_prefix: torch.Tensor) -> torch.Tensor:
        if d_len >= n_q:
            return base_t + shift_prefix
        return torch.cat([base_t[:d_len] + shift_prefix, base_t[d_len:]], dim=0)

    if (
        allow_cpp
        and quote_delta is None
        and qcache.slice_c_np is not None
        and geom.n_unique >= 2
        and geom.strikes_np is not None
        and geom.tenors_np is not None
        and _load_tier1_cpp_ad_fn() is not None
        and _SURFACE_CPP_AD_FN is not None
    ):
        t_vals_np = np.ascontiguousarray(t_vals.detach().cpu().numpy(), dtype=np.float64)
        if not torch.is_grad_enabled() or not k_queries.requires_grad:
            s_fn = _SURFACE_CPP_AD_FN
            q_cnt = int(k_queries.shape[0])
            k_np = np.ascontiguousarray(k_queries.detach().cpu().numpy(), dtype=np.float64)
            iv_np = np.empty((q_cnt,), dtype=np.float64)
            dk_np = np.empty((q_cnt,), dtype=np.float64)
            assert qcache.quotes_np is not None and qcache.slope_left_np is not None and qcache.slope_right_np is not None
            s_fn(
                q_cnt, geom.n_unique, geom.n_exp,
                geom.k_min, geom.k_max, float(surf.min_vol),
                geom.strikes_np.ctypes.data, geom.tenors_np.ctypes.data, qcache.quotes_np.ctypes.data,
                qcache.slice_c_np.ctypes.data, qcache.slope_left_np.ctypes.data, qcache.slope_right_np.ctypes.data,
                t_vals_np.ctypes.data, k_np.ctypes.data,
                iv_np.ctypes.data, dk_np.ctypes.data,
            )
            return torch.from_numpy(iv_np).to(device=device)
        iv_cpp, _ = _SurfaceEvalCppFn.apply(  # type: ignore[no-untyped-call]
            k_queries, t_vals_np, geom, qcache, float(surf.min_vol), surf, t_vals
        )
        return cast(torch.Tensor, iv_cpp)

    if (
        qcache.slice_c is not None
        and qcache.slice_slope_left is not None
        and qcache.slice_slope_right is not None
        and geom.n_unique >= 2
    ):
        seg_idx = _bucket_idx_torch(geom.strikes, k_queries)
        k_left = geom.strikes[seg_idx]
        dx = k_queries - k_left
        mask_left = k_queries < geom.k_min
        mask_right = k_queries > geom.k_max

        # Preserve full differentiation wrt k_queries even when quote_delta is active (Vanna).
        w_u_delta = (
            _strike_basis_u_batch_torch(geom, k_queries[:d_len])
            if (proj_q_delta is not None and d_len > 0)
            else None
        )

        if geom.n_exp == 1:
            c0 = qcache.slice_c[:, seg_idx, 0]
            val = ((c0[0] * dx + c0[1]) * dx + c0[2]) * dx + c0[3]
            val_left = qcache.quotes[0, 0] + qcache.slice_slope_left[0] * (k_queries - geom.k_min)
            val_right = qcache.quotes[0, -1] + qcache.slice_slope_right[0] * (k_queries - geom.k_max)
            val = torch.where(mask_left, val_left, val)
            val = torch.where(mask_right, val_right, val)
            if proj_q_delta is not None and w_u_delta is not None:
                val = _add_delta_prefix(val, (proj_q_delta[0].unsqueeze(0) * w_u_delta).sum(dim=-1))
            return torch.clamp(val, min=surf.min_vol)

        idx0 = _bucket_idx_torch(geom.tenors, t_vals)
        idx1 = idx0 + 1
        c0 = qcache.slice_c[:, seg_idx, idx0]
        c1 = qcache.slice_c[:, seg_idx, idx1]
        v0 = ((c0[0] * dx + c0[1]) * dx + c0[2]) * dx + c0[3]
        v1 = ((c1[0] * dx + c1[1]) * dx + c1[2]) * dx + c1[3]

        v0_left = qcache.quotes[idx0, 0] + qcache.slice_slope_left[idx0] * (k_queries - geom.k_min)
        v1_left = qcache.quotes[idx1, 0] + qcache.slice_slope_left[idx1] * (k_queries - geom.k_min)
        v0 = torch.where(mask_left, v0_left, v0)
        v1 = torch.where(mask_left, v1_left, v1)

        v0_right = qcache.quotes[idx0, -1] + qcache.slice_slope_right[idx0] * (k_queries - geom.k_max)
        v1_right = qcache.quotes[idx1, -1] + qcache.slice_slope_right[idx1] * (k_queries - geom.k_max)
        v0 = torch.where(mask_right, v0_right, v0)
        v1 = torch.where(mask_right, v1_right, v1)

        if proj_q_delta is not None and w_u_delta is not None:
            v0 = _add_delta_prefix(v0, (proj_q_delta[idx0[:d_len]] * w_u_delta).sum(dim=-1))
            v1 = _add_delta_prefix(v1, (proj_q_delta[idx1[:d_len]] * w_u_delta).sum(dim=-1))

        v0 = torch.clamp(v0, min=surf.min_vol)
        v1 = torch.clamp(v1, min=surf.min_vol)

        t0_t = geom.tenors[idx0]
        t1_t = geom.tenors[idx1]
        alpha_t = (t_vals - t0_t) / (t1_t - t0_t).clamp(min=1e-12)
        w = (1.0 - alpha_t) * ((v0 * v0) * t0_t) + alpha_t * ((v1 * v1) * t1_t)
        v_interp = torch.sqrt(torch.clamp(w / t_vals, min=surf.min_vol * surf.min_vol))

        v_interp = torch.where(t_vals <= geom.tenors[0], v0, v_interp)
        v_interp = torch.where(t_vals >= geom.tenors[-1], v1, v_interp)
        return v_interp

    w_u = _strike_basis_u_batch_torch(geom, k_queries)  # (K, N_unique)
    if geom.n_exp == 1:
        val = (qcache.quotes[0].unsqueeze(0) * w_u).sum(dim=-1)
        if proj_q_delta is not None and d_len > 0:
            val = _add_delta_prefix(val, (proj_q_delta[0].unsqueeze(0) * w_u[:d_len]).sum(dim=-1))
        return torch.clamp(val, min=surf.min_vol)

    idx0 = _bucket_idx_torch(geom.tenors, t_vals)
    idx1 = idx0 + 1

    v0 = (qcache.quotes[idx0] * w_u).sum(dim=-1)
    v1 = (qcache.quotes[idx1] * w_u).sum(dim=-1)
    if proj_q_delta is not None and d_len > 0:
        v0 = _add_delta_prefix(v0, (proj_q_delta[idx0[:d_len]] * w_u[:d_len]).sum(dim=-1))
        v1 = _add_delta_prefix(v1, (proj_q_delta[idx1[:d_len]] * w_u[:d_len]).sum(dim=-1))
    v0 = torch.clamp(v0, min=surf.min_vol)
    v1 = torch.clamp(v1, min=surf.min_vol)

    t0_t = geom.tenors[idx0]
    t1_t = geom.tenors[idx1]
    alpha_t = (t_vals - t0_t) / (t1_t - t0_t).clamp(min=1e-12)
    w = (1.0 - alpha_t) * ((v0 * v0) * t0_t) + alpha_t * ((v1 * v1) * t1_t)
    v_interp = torch.sqrt(torch.clamp(w / t_vals, min=surf.min_vol * surf.min_vol))

    v_interp = torch.where(t_vals <= geom.tenors[0], v0, v_interp)
    v_interp = torch.where(t_vals >= geom.tenors[-1], v1, v_interp)
    return v_interp


def _eval_reg_surface_iv_torch(
    surf: RegularizedVolSurface,
    t: float | torch.Tensor,
    k_query: torch.Tensor,
    quote_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable torch.float64 evaluation of sigma_IV(t, K) wrt both K and raw surface quotes."""
    t_tensor = (
        t.reshape(1).to(dtype=torch.float64, device=k_query.device)
        if isinstance(t, torch.Tensor)
        else torch.tensor([float(t)], dtype=torch.float64, device=k_query.device)
    )
    return _eval_reg_surface_iv_batch_torch(surf, t_tensor, k_query.reshape(1), quote_delta).squeeze(0)


def _surface_quote_vjp(
    surf: RegularizedVolSurface,
    t_vals: torch.Tensor,
    k_queries: torch.Tensor,
    g_iv: torch.Tensor,
    *,
    cached_pair: tuple[_SurfaceGeomCache, _SurfaceQuoteCache] | None = None,
) -> torch.Tensor:
    """Computes exact VJP dL/d(raw_quotes) from upstream cotangent g_iv = dL/d(sigma_IV)."""
    device = k_queries.device
    geom, qcache = cached_pair if cached_pair is not None else _get_surface_torch_caches(surf, device)
    with torch.no_grad():
        if (
            geom.n_unique >= 2
            and qcache.slice_c_np is not None
            and qcache.quotes_np is not None
            and qcache.slope_left_np is not None
            and qcache.slope_right_np is not None
            and geom.strikes_np is not None
            and geom.tenors_np is not None
            and geom.basis_c_np is not None
            and geom.slope_left_np is not None
            and geom.slope_right_np is not None
            and geom.dedup_t_np is not None
            and _load_tier1_cpp_ad_fn() is not None
            and _SURFACE_VJP_CPP_FN is not None
        ):
            t_np = np.ascontiguousarray(t_vals.detach().cpu().numpy(), dtype=np.float64)
            k_np = np.ascontiguousarray(k_queries.detach().cpu().numpy(), dtype=np.float64)
            g_np = np.ascontiguousarray(g_iv.detach().cpu().numpy(), dtype=np.float64)
            out_np = np.empty((geom.n_exp, geom.n_raw), dtype=np.float64)
            cal_ptr = geom.cal_proj_jac_ptr
            _SURFACE_VJP_CPP_FN(
                int(k_np.shape[0]),
                geom.n_unique,
                geom.n_exp,
                geom.n_raw,
                1 if cal_ptr != 0 else 0,
                geom.k_min,
                geom.k_max,
                float(surf.min_vol),
                geom.strikes_ptr,
                geom.tenors_ptr,
                qcache.quotes_ptr,
                qcache.slice_c_ptr,
                qcache.slope_left_ptr,
                qcache.slope_right_ptr,
                geom.basis_c_ptr,
                geom.slope_left_ptr,
                geom.slope_right_ptr,
                geom.dedup_t_ptr,
                cal_ptr,
                t_np.__array_interface__["data"][0],
                k_np.__array_interface__["data"][0],
                g_np.__array_interface__["data"][0],
                out_np.__array_interface__["data"][0],
            )
            return torch.from_numpy(out_np).to(device=device)

        t_c = t_vals.to(dtype=torch.float64, device=device).clamp(min=1e-8)
        k_c = k_queries.to(dtype=torch.float64, device=device)
        g_c = g_iv.to(dtype=torch.float64, device=device)
        w_u = _strike_basis_u_batch_torch(geom, k_c)  # (Q, N_unique)
        proj_q_grad = torch.zeros((geom.n_exp, max(geom.n_unique, 1)), dtype=torch.float64, device=device)

        def _slice_vals(exp_idx: torch.Tensor) -> torch.Tensor:
            if (
                qcache.slice_c is not None
                and qcache.slice_slope_left is not None
                and qcache.slice_slope_right is not None
                and geom.n_unique >= 2
            ):
                seg_idx = _bucket_idx_torch(geom.strikes, k_c)
                dx = k_c - geom.strikes[seg_idx]
                c_s = qcache.slice_c[:, seg_idx, exp_idx]
                v_raw = ((c_s[0] * dx + c_s[1]) * dx + c_s[2]) * dx + c_s[3]
                v_l = qcache.quotes[exp_idx, 0] + qcache.slice_slope_left[exp_idx] * (k_c - geom.k_min)
                v_r = qcache.quotes[exp_idx, -1] + qcache.slice_slope_right[exp_idx] * (k_c - geom.k_max)
                v_raw = torch.where(k_c < geom.k_min, v_l, v_raw)
                return torch.where(k_c > geom.k_max, v_r, v_raw)
            return (qcache.quotes[exp_idx] * w_u).sum(dim=-1)

        if geom.n_exp == 1:
            z0 = torch.zeros((int(k_c.shape[0]),), dtype=torch.long, device=device)
            v0_raw = _slice_vals(z0)
            g_eff = torch.where(v0_raw >= surf.min_vol, g_c, torch.zeros_like(g_c))
            proj_q_grad[0] = (g_eff.unsqueeze(-1) * w_u).sum(dim=0)
        else:
            idx0 = _bucket_idx_torch(geom.tenors, t_c)
            idx1 = idx0 + 1
            v0_raw = _slice_vals(idx0)
            v1_raw = _slice_vals(idx1)
            m0 = v0_raw >= surf.min_vol
            m1 = v1_raw >= surf.min_vol
            v0 = torch.clamp(v0_raw, min=surf.min_vol)
            v1 = torch.clamp(v1_raw, min=surf.min_vol)

            t0_t = geom.tenors[idx0]
            t1_t = geom.tenors[idx1]
            alpha_t = (t_c - t0_t) / (t1_t - t0_t).clamp(min=1e-12)
            w_var = (1.0 - alpha_t) * ((v0 * v0) * t0_t) + alpha_t * ((v1 * v1) * t1_t)
            wt = w_var / t_c
            mv2 = surf.min_vol * surf.min_vol
            v_interp = torch.sqrt(torch.clamp(wt, min=mv2))
            in_lo = t_c <= geom.tenors[0]
            in_hi = (~in_lo) & (t_c >= geom.tenors[-1])
            in_mid = (~in_lo) & (~in_hi) & (wt > mv2)
            dv_dv0 = torch.where(
                in_lo,
                torch.ones_like(v0),
                torch.where(in_mid, (1.0 - alpha_t) * t0_t * v0 / (t_c * v_interp), torch.zeros_like(v0)),
            )
            dv_dv1 = torch.where(
                in_hi,
                torch.ones_like(v1),
                torch.where(in_mid, alpha_t * t1_t * v1 / (t_c * v_interp), torch.zeros_like(v1)),
            )
            g0 = torch.where(m0, g_c * dv_dv0, torch.zeros_like(g_c))
            g1 = torch.where(m1, g_c * dv_dv1, torch.zeros_like(g_c))
            proj_q_grad.index_add_(0, idx0, g0.unsqueeze(-1) * w_u)
            proj_q_grad.index_add_(0, idx1, g1.unsqueeze(-1) * w_u)

        g_dedup = (
            torch.einsum("mnu,mu->nu", geom.cal_proj_jac, proj_q_grad)
            if geom.cal_proj_jac is not None
            else proj_q_grad
        )
        return torch.matmul(g_dedup, geom.dedup_t.transpose(0, 1))


def _curve_df_delta_multiplier(
    curve: Any,
    dates: Sequence[Any],
    pillar_deltas: torch.Tensor,
    *,
    as_of_date: dt.datetime | dt.date | None = None,
) -> torch.Tensor:
    """Computes exp(-Delta_r(d) * tau) multiplier from curve pillar quote deltas."""
    device = pillar_deltas.device
    if len(dates) == 0:
        return torch.ones(0, dtype=torch.float64, device=device)

    curve_obj = _normalize_curve_obj(curve, as_of_date=as_of_date)
    dev_key = (device.type, device.index)
    curve_digest = _fast_array_digest(
        getattr(curve_obj, "tenors", None),
        getattr(curve_obj, "_raw_quotes", getattr(curve_obj, "quotes", None)),
        getattr(curve_obj, "dcc", ""),
        getattr(curve_obj, "space_coords", ""),
        getattr(curve_obj, "_input_mode", ""),
    )
    dates_key = tuple(dates)
    cache_key = (curve_digest, dates_key, int(pillar_deltas.numel()), dev_key)

    cached = _CURVE_DELTA_CACHE.get(cache_key) if curve_digest is not None else None
    if cached is not None:
        w_t, tau_t = cached
    else:
        as_of = getattr(curve_obj, "as_of_date", as_of_date)
        dcc_curve = getattr(curve_obj, "dcc", "Act365")
        if hasattr(curve_obj, "spline_weights") and callable(curve_obj.spline_weights):
            w_np = np.asarray(curve_obj.spline_weights(dates), dtype=np.float64)
        else:
            w_np = np.zeros((len(dates), pillar_deltas.numel()), dtype=np.float64)

        if as_of is not None and isinstance(dates[0], (dt.datetime, dt.date)):
            tau_np = np.array([_year_fraction(as_of, d, dcc_curve) for d in dates], dtype=np.float64)
        else:
            tau_np = np.asarray(dates, dtype=np.float64)

        w_t = torch.tensor(w_np, dtype=torch.float64, device=device)
        tau_t = torch.tensor(tau_np, dtype=torch.float64, device=device)
        if curve_digest is not None and not _in_functorch_transform():
            _CURVE_DELTA_CACHE.put(cache_key, (w_t, tau_t))

    scale = float(getattr(curve_obj, "quote_scale", 0.01))
    space_coords = getattr(curve_obj, "space_coords", "Rates")
    input_mode = getattr(curve_obj, "_input_mode", None)

    if input_mode == "rates" or (input_mode is None and space_coords == "Rates"):
        delta_r = torch.mv(w_t, pillar_deltas) * scale
        return torch.exp(-delta_r * tau_t.clamp(min=0.0))
    if input_mode == "log_df":
        tenors_t = torch.tensor(
            np.asarray(curve_obj.tenors, dtype=np.float64), dtype=torch.float64, device=device
        ).clamp(min=1e-12)
        delta_r_pil = -(pillar_deltas * scale) / tenors_t
        delta_r = torch.mv(w_t, delta_r_pil)
        return torch.exp(-delta_r * tau_t.clamp(min=0.0))
    if input_mode is None and space_coords == "LogarithmOfDiscountFactors":
        tenors_t = torch.tensor(
            np.asarray(curve_obj.tenors, dtype=np.float64), dtype=torch.float64, device=device
        )
        delta_y = -tenors_t * pillar_deltas * scale
        return torch.exp(torch.mv(w_t, delta_y))

    # DiscountFactors space
    df0_np = np.asarray(curve_obj.df(dates), dtype=np.float64).reshape(-1)
    _, d_df_np = curve_obj.df_and_d_df_d_pillar(dates, in_quote_units=True)
    df0_t = torch.tensor(df0_np, dtype=torch.float64, device=device).clamp(min=1e-15)
    d_df_t = torch.tensor(np.asarray(d_df_np, dtype=np.float64), dtype=torch.float64, device=device)
    df_new = (df0_t + torch.mv(d_df_t, pillar_deltas)).clamp(min=1e-15)
    df_ratio: torch.Tensor = df_new / df0_t
    return df_ratio


def _curve_df_delta_jacobian(
    curve: Any,
    dates: Sequence[Any],
    n_pillars: int,
    device: torch.device,
    *,
    as_of_date: dt.datetime | dt.date | None = None,
) -> torch.Tensor:
    """Returns cached (len(dates), n_pillars) float64 Jacobian d(mult)/d(pillar_deltas) at delta=0."""
    if len(dates) == 0 or n_pillars <= 0:
        return torch.zeros((len(dates), max(n_pillars, 0)), dtype=torch.float64, device=device)

    curve_obj = _normalize_curve_obj(curve, as_of_date=as_of_date)
    dev_key = (device.type, device.index)
    curve_digest = _fast_array_digest(
        getattr(curve_obj, "tenors", None),
        getattr(curve_obj, "_raw_quotes", getattr(curve_obj, "quotes", None)),
        getattr(curve_obj, "dcc", ""),
        getattr(curve_obj, "space_coords", ""),
        getattr(curve_obj, "_input_mode", ""),
    )
    dates_key = tuple(dates)
    jac_key = (curve_digest, dates_key, int(n_pillars), dev_key)
    cached_j = _CURVE_JAC_CACHE.get(jac_key)
    if cached_j is not None:
        return cast(torch.Tensor, cached_j)

    delta_key = (curve_digest, dates_key, int(n_pillars), dev_key)
    cached_d = _CURVE_DELTA_CACHE.get(delta_key)
    if cached_d is not None:
        w_t, tau_t = cached_d
    else:
        as_of = getattr(curve_obj, "as_of_date", as_of_date)
        dcc_curve = getattr(curve_obj, "dcc", "Act365")
        if hasattr(curve_obj, "spline_weights") and callable(curve_obj.spline_weights):
            w_np = np.asarray(curve_obj.spline_weights(dates), dtype=np.float64)
        else:
            w_np = np.zeros((len(dates), n_pillars), dtype=np.float64)
        if as_of is not None and isinstance(dates[0], (dt.datetime, dt.date)):
            tau_np = np.array([_year_fraction(as_of, d, dcc_curve) for d in dates], dtype=np.float64)
        else:
            tau_np = np.asarray(dates, dtype=np.float64)
        w_t = torch.tensor(w_np, dtype=torch.float64, device=device)
        tau_t = torch.tensor(tau_np, dtype=torch.float64, device=device)
        _CURVE_DELTA_CACHE.put(delta_key, (w_t, tau_t))

    scale = float(getattr(curve_obj, "quote_scale", 0.01))
    space_coords = getattr(curve_obj, "space_coords", "Rates")
    input_mode = getattr(curve_obj, "_input_mode", None)

    if input_mode == "rates" or (input_mode is None and space_coords == "Rates"):
        jac = (-scale) * tau_t.clamp(min=0.0).unsqueeze(-1) * w_t
    elif input_mode == "log_df":
        tenors_t = torch.tensor(
            np.asarray(curve_obj.tenors, dtype=np.float64), dtype=torch.float64, device=device
        ).clamp(min=1e-12)
        jac = (scale * tau_t.clamp(min=0.0).unsqueeze(-1)) * (w_t / tenors_t.unsqueeze(0))
    elif input_mode is None and space_coords == "LogarithmOfDiscountFactors":
        tenors_t = torch.tensor(
            np.asarray(curve_obj.tenors, dtype=np.float64), dtype=torch.float64, device=device
        )
        jac = (-scale) * w_t * tenors_t.unsqueeze(0)
    else:
        df0_np = np.asarray(curve_obj.df(dates), dtype=np.float64).reshape(-1)
        _, d_df_np = curve_obj.df_and_d_df_d_pillar(dates, in_quote_units=True)
        df0_t = torch.tensor(df0_np, dtype=torch.float64, device=device).clamp(min=1e-15)
        d_df_t = torch.tensor(np.asarray(d_df_np, dtype=np.float64), dtype=torch.float64, device=device)
        jac = d_df_t / df0_t.unsqueeze(-1)

    _CURVE_JAC_CACHE.put(jac_key, jac)
    return cast(torch.Tensor, jac)


def _curve_df_cached_np(curve_obj: Any, dates: Sequence[Any]) -> npt.NDArray[np.float64]:
    """Bounded-LRU cached curve.df(dates) as contiguous float64 NumPy array."""
    c_digest = _fast_array_digest(
        getattr(curve_obj, "tenors", None),
        getattr(curve_obj, "_raw_quotes", getattr(curve_obj, "quotes", None)),
        getattr(curve_obj, "dcc", ""),
        getattr(curve_obj, "space_coords", ""),
        getattr(curve_obj, "_input_mode", ""),
        getattr(curve_obj, "as_of_date", None),
    )
    dates_key = tuple(dates)
    c_key = (c_digest, dates_key)
    cached = _THETA_CURVE_CACHE.get(c_key)
    if cached is not None:
        return cast(npt.NDArray[np.float64], cached)
    arr = np.ascontiguousarray(np.asarray(curve_obj.df(dates), dtype=np.float64).reshape(-1))
    _THETA_CURVE_CACHE.put(c_key, arr)
    return arr


def _smooth_corr_guard(corr_sym: torch.Tensor) -> torch.Tensor:
    """Applies a C-infinity saturation guard near |rho| -> 1 without perturbing |rho| <= 0.999."""
    d_assets = corr_sym.shape[-1]
    eye_d = torch.eye(d_assets, dtype=corr_sym.dtype, device=corr_sym.device)
    abs_c = torch.abs(corr_sym)
    thr = 0.999
    span = 0.0009
    guarded = torch.sign(corr_sym) * (thr + span * torch.tanh((abs_c - thr) / span))
    off_diag = torch.where(abs_c <= thr, corr_sym, guarded)
    corr_out: torch.Tensor = off_diag * (1.0 - eye_d) + eye_d
    return corr_out


def _escrowed_spot_softplus(spots_col: torch.Tensor, div_pv_t: torch.Tensor) -> torch.Tensor:
    """Smooth softplus escrowed-spot floor matching escrowed_dividend.py (0-ULP away from 99%)."""
    floor_spot = 0.01 * spots_col
    scale_sp = floor_spot.clamp(min=1e-12)
    return floor_spot + scale_sp * torch.nn.functional.softplus(
        (spots_col - div_pv_t - floor_spot) / scale_sp, beta=1.0, threshold=40.0
    )


def _eval_index_scalar(
    perf_m: npt.NDArray[np.float64],
    idx_name: str,
    inv_lex_perm: npt.NDArray[np.int64],
) -> float:
    """Evaluates a deterministic Murex monitoring index on 1D sorted asset performances."""
    if idx_name == "WorstOf":
        return float(np.min(perf_m))
    if idx_name == "BestOf":
        return float(np.max(perf_m))
    if idx_name in ("Average", "AverageOf", "MultiPerf"):
        return float(np.mean(perf_m))
    if idx_name == "Dispersion":
        return float(np.std(perf_m))
    if idx_name in ("RatioAB", "Outperformance", "Ratio"):
        a = float(perf_m[int(inv_lex_perm[0])])
        b = max(float(perf_m[int(inv_lex_perm[1])]), 1e-12) if len(perf_m) >= 2 else 1.0
        return a / b
    if idx_name == "RatioBA":
        a = max(float(perf_m[int(inv_lex_perm[0])]), 1e-12)
        b = float(perf_m[int(inv_lex_perm[1])]) if len(perf_m) >= 2 else a
        return b / a
    if idx_name in ("DiffAB", "Spread"):
        a = float(perf_m[int(inv_lex_perm[0])])
        b = float(perf_m[int(inv_lex_perm[1])]) if len(perf_m) >= 2 else 0.0
        return a - b
    if idx_name == "DiffBA":
        a = float(perf_m[int(inv_lex_perm[0])])
        b = float(perf_m[int(inv_lex_perm[1])]) if len(perf_m) >= 2 else 0.0
        return b - a
    if idx_name == "RatioBestWorst":
        return float(np.max(perf_m)) / max(float(np.min(perf_m)), 1e-12)
    if idx_name == "RatioBestAverage":
        return float(np.max(perf_m)) / max(float(np.mean(perf_m)), 1e-12)
    if idx_name == "RatioAverageWorst":
        return float(np.mean(perf_m)) / max(float(np.min(perf_m)), 1e-12)
    if idx_name == "DiffBestWorst":
        return float(np.max(perf_m) - np.min(perf_m))
    if idx_name == "DiffBestAverage":
        return float(np.max(perf_m) - np.mean(perf_m))
    if idx_name == "DiffAverageWorst":
        return float(np.mean(perf_m) - np.min(perf_m))
    return float(np.min(perf_m))


def _lookup_funding_fixing_val(
    fixings_pairs: Sequence[tuple[dt.datetime, float]],
    target_dt: dt.datetime,
) -> float | None:
    """Finds an exact or same-calendar-day fixing for a funding reset date."""
    if not fixings_pairs:
        return None
    target_date = target_dt.date()
    fallback: float | None = None
    for f_dt, f_val in fixings_pairs:
        if f_dt == target_dt:
            return float(f_val)
        if f_dt.date() == target_date:
            fallback = float(f_val)
    return fallback


def _compute_funding_weights_per_period(
    base_state: AutocallPreprocState,
    *,
    as_of_override: dt.datetime | None = None,
    df_disc_fn: Callable[[dt.datetime], torch.Tensor | float] | None = None,
    df_fwd_fn: Callable[[dt.datetime], torch.Tensor | float] | None = None,
    period_start_idx: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Pure stateless FundingLeg PV weight per surviving observation period (W_fund in R^{M_rem}).

    Replaces fragile mutable `_cached_w_fund_per_period` lookups so Gamma + Rho and rolled
    Theta share identical deterministic funding PV weights without aliasing or drop-outs.
    """
    m_total = base_state.m_rem
    m_out = max(m_total - period_start_idx, 0)
    eff_dev = device if device is not None else base_state.spots_t0.device
    if m_out == 0 or base_state.funding_leg is None or base_state.already_autocalled:
        return torch.zeros(m_out, dtype=torch.float64, device=eff_dev)

    f_leg = base_state.funding_leg
    f_rid = f_leg.get("rate_identifier")
    as_of_orig = base_state.seasoned_state.as_of_date
    eff_as_of = as_of_override if as_of_override is not None else as_of_orig
    start_dt = base_state.seasoned_state.start_date
    f_dcc = base_state.seasoned_state.day_count_convention

    disc_curve = _normalize_curve_obj(base_state.discount_curve, as_of_date=as_of_orig)
    f_curve = _normalize_curve_obj(f_leg.get("funding_curve"), as_of_date=as_of_orig)
    f_curve_id = f_leg.get("funding_curve_id")
    disc_id = base_state.discount_curve_id
    same_curve = (
        f_curve is None
        or f_curve is base_state.discount_curve
        or (f_curve_id is not None and disc_id is not None and f_curve_id == disc_id)
    )

    if df_disc_fn is None:
        disc_anchor = (
            float(np.asarray(disc_curve.df(eff_as_of), dtype=np.float64).reshape(-1)[0])
            if (as_of_override is not None and disc_curve is not None and hasattr(disc_curve, "df"))
            else 1.0
        )
        disc_anchor_safe = max(disc_anchor, 1e-15)

        def _default_disc_fn(d: dt.datetime) -> float:
            if disc_curve is None or not hasattr(disc_curve, "df"):
                return 1.0
            val = float(np.asarray(disc_curve.df(d), dtype=np.float64).reshape(-1)[0])
            return val / disc_anchor_safe

        eff_disc_fn: Callable[[dt.datetime], torch.Tensor | float] = _default_disc_fn
    else:
        eff_disc_fn = df_disc_fn

    if df_fwd_fn is None:
        if same_curve:
            eff_fwd_fn = eff_disc_fn
        else:
            fwd_anchor = (
                float(np.asarray(f_curve.df(eff_as_of), dtype=np.float64).reshape(-1)[0])
                if (as_of_override is not None and f_curve is not None and hasattr(f_curve, "df"))
                else 1.0
            )
            fwd_anchor_safe = max(fwd_anchor, 1e-15)

            def _default_fwd_fn(d: dt.datetime) -> float:
                if f_curve is None or not hasattr(f_curve, "df"):
                    return 1.0
                val = float(np.asarray(f_curve.df(d), dtype=np.float64).reshape(-1)[0])
                return val / fwd_anchor_safe

            eff_fwd_fn = _default_fwd_fn
    else:
        eff_fwd_fn = df_fwd_fn

    f_spread = float(f_leg.get("funding_spread", 0.0))
    f_dates = f_leg.get("payment_dates", ())
    f_fixings = f_leg.get("funding_fixings", ())
    future_pay_dts = base_state.seasoned_state.future_pay_dates[period_start_idx:]

    weights: list[torch.Tensor] = [
        torch.tensor(0.0, dtype=torch.float64, device=eff_dev) for _ in range(m_out)
    ]
    prev_f_dt = start_dt
    for f_dt in f_dates:
        if f_dt <= eff_as_of:
            prev_f_dt = f_dt
            continue
        alpha_j = _year_fraction(prev_f_dt, f_dt, f_dcc)
        df_curr = eff_disc_fn(f_dt)
        if f_rid is None:
            r_fwd_j: torch.Tensor | float = 0.0
        else:
            fixed_reset = _lookup_funding_fixing_val(f_fixings, prev_f_dt) if prev_f_dt < eff_as_of else None
            if fixed_reset is not None:
                r_fwd_j = fixed_reset
            elif prev_f_dt < eff_as_of:
                alpha_rem = max(_year_fraction(eff_as_of, f_dt, f_dcc), 1e-8)
                df_curr_fwd = eff_fwd_fn(f_dt)
                if isinstance(df_curr_fwd, torch.Tensor):
                    r_fwd_j = (1.0 / df_curr_fwd.clamp(min=1e-12) - 1.0) / alpha_rem
                else:
                    r_fwd_j = (1.0 / max(float(df_curr_fwd), 1e-12) - 1.0) / alpha_rem
            else:
                df_prev_fwd = eff_fwd_fn(prev_f_dt)
                df_curr_fwd = eff_fwd_fn(f_dt)
                if isinstance(df_curr_fwd, torch.Tensor):
                    r_fwd_j = (df_prev_fwd / df_curr_fwd.clamp(min=1e-12) - 1.0) / max(alpha_j, 1e-8)
                else:
                    r_fwd_j = (
                        float(df_prev_fwd) / max(float(df_curr_fwd), 1e-12) - 1.0
                    ) / max(alpha_j, 1e-8)

        m_idx = m_out - 1
        for m in range(m_out):
            if future_pay_dts[m] >= f_dt:
                m_idx = m
                break
        contrib = df_curr * (r_fwd_j + f_spread) * alpha_j
        if isinstance(contrib, torch.Tensor):
            weights[m_idx] = weights[m_idx] + contrib.to(dtype=torch.float64, device=eff_dev)
        else:
            weights[m_idx] = weights[m_idx] + float(contrib)
        prev_f_dt = f_dt

    return torch.stack(weights, dim=0)


def _rolled_in_transit_pv(
    base_state: AutocallPreprocState,
    disc_curve: Any,
    eff_as_of: dt.datetime,
    curve_pillar_deltas: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Re-discounts surviving In-Transit cash flows to `eff_as_of`, dropping settled items."""
    as_of_orig = base_state.seasoned_state.as_of_date
    if eff_as_of == as_of_orig and curve_pillar_deltas is None:
        return torch.tensor(base_state.pv_in_transit, dtype=torch.float64, device=device)

    disc_anchor = 1.0
    if eff_as_of != as_of_orig and disc_curve is not None and hasattr(disc_curve, "df"):
        disc_anchor = max(
            float(np.asarray(disc_curve.df(eff_as_of), dtype=np.float64).reshape(-1)[0]),
            1e-15,
        )

    pv_t = torch.tensor(0.0, dtype=torch.float64, device=device)
    past_cfs = base_state.seasoned_state.past_cashflows
    estimate_cfs = [cf for cf in past_cfs if getattr(cf, "StatusText", "") == "Estimate"]
    if estimate_cfs:
        for cf in estimate_cfs:
            pay_dt = cf.Date
            if eff_as_of != as_of_orig and pay_dt <= eff_as_of:
                continue
            amt = float(cf.Amount) / disc_anchor
            if curve_pillar_deltas is not None and abs(amt) > 0.0:
                mult = _curve_df_delta_multiplier(
                    disc_curve, [pay_dt], curve_pillar_deltas, as_of_date=as_of_orig
                )[0]
                if eff_as_of != as_of_orig:
                    anchor_mult = _curve_df_delta_multiplier(
                        disc_curve, [eff_as_of], curve_pillar_deltas, as_of_date=as_of_orig
                    )[0]
                    mult = mult / anchor_mult.clamp(min=1e-15)
                pv_t = pv_t + amt * mult
            else:
                pv_t = pv_t + amt
        return pv_t

    # Fallback when past_cashflows was omitted: use HistoricalPeriodRecord rows.
    for rec in base_state.seasoned_state.historical_records:
        if rec.status == "In-Transit" and (rec.coupon_paid != 0.0 or rec.notional_paid != 0.0):
            if eff_as_of != as_of_orig and rec.payment_date <= eff_as_of:
                continue
            amt = float(rec.discount_factor * (rec.coupon_paid + rec.notional_paid)) / disc_anchor
            if curve_pillar_deltas is not None:
                mult = _curve_df_delta_multiplier(
                    disc_curve, [rec.payment_date], curve_pillar_deltas, as_of_date=as_of_orig
                )[0]
                pv_t = pv_t + amt * mult
            else:
                pv_t = pv_t + amt
    return pv_t


def _collapse_crossed_periods_for_roll(
    base_state: AutocallPreprocState,
    disc_curve: Any,
    eff_as_of: dt.datetime,
    spots_sorted_np: npt.NDArray[np.float64],
    ref_spots_np: npt.NDArray[np.float64],
) -> tuple[int, bool, float, float, float]:
    """Deterministically settles any observation dates crossed during a calendar roll."""
    obs_dts = base_state.seasoned_state.future_obs_dates
    pay_dts = base_state.seasoned_state.future_pay_dates
    m_rem = base_state.m_rem
    if m_rem == 0 or not obs_dts or obs_dts[0] > eff_as_of:
        return 0, bool(base_state.already_autocalled), float(base_state.i_ki_past), float(base_state.c_mem_0), 0.0

    perf_m = spots_sorted_np / np.maximum(ref_spots_np, 1e-12)
    inv_perm = base_state.inv_lex_perm
    call_idx = base_state.callability_index
    cpn_idx = base_state.coupon_index or call_idx
    opt_idx = base_state.option_index or call_idx
    w_call = _eval_index_scalar(perf_m, call_idx, inv_perm)
    w_cpn = _eval_index_scalar(perf_m, cpn_idx, inv_perm)
    w_opt = _eval_index_scalar(perf_m, opt_idx, inv_perm)

    ko_arr = base_state.ko_barriers.detach().cpu().numpy()
    cpn_arr = base_state.cpn_barriers.detach().cpu().numpy()
    is_call_arr = base_state.is_callable.detach().cpu().numpy()
    cpn_rates = base_state.coupon_amounts.detach().cpu().numpy()

    type_out = base_state.coupon_barrier_type.strip().lower() == "out"
    dir_down = base_state.coupon_barrier_direction.strip().lower() == "down"
    eff_i_ki = float(base_state.i_ki_past)
    eff_c_mem = float(base_state.c_mem_0)
    eff_autocalled = bool(base_state.already_autocalled)
    crossed_pv = 0.0

    disc_anchor = (
        max(float(np.asarray(disc_curve.df(eff_as_of), dtype=np.float64).reshape(-1)[0]), 1e-15)
        if disc_curve is not None and hasattr(disc_curve, "df")
        else 1.0
    )

    # Funding weights per period if already contracted on crossed observation dates.
    k_cross = 0
    for m in range(m_rem):
        if obs_dts[m] > eff_as_of:
            break
        k_cross = m + 1
        if eff_autocalled:
            continue
        if base_state.ki_barrier is not None and base_state.ki_barrier > 0.0 and w_opt < float(base_state.ki_barrier):
            eff_i_ki = 1.0

        b_ko = float(ko_arr[m])
        b_cpn = float(cpn_arr[m])
        c_m = float(cpn_rates[m])
        ko_trig = bool(is_call_arr[m] > 0.5 and w_call >= b_ko)
        hit_dir = (w_cpn <= b_cpn) if dir_down else (w_cpn >= b_cpn)
        base_cpn = (not hit_dir) if type_out else hit_dir
        is_guar = bool(base_state.is_guaranteed_cpn[m].item() > 0.5)
        cpn_trig = bool(is_guar or base_cpn or (ko_trig and base_state.force_coupons_on_autocall))

        if cpn_trig:
            cpn_paid = (c_m + eff_c_mem) if base_state.is_memory else c_m
            eff_c_mem = 0.0
        else:
            cpn_paid = 0.0
            if base_state.is_memory:
                eff_c_mem += c_m

        opt_paid = 0.0
        if ko_trig:
            cpn_paid += float(base_state.autocall_coupon)
            notional_paid = 0.0 if base_state.no_notional_repayment else 1.0
            eff_autocalled = True
        elif m == m_rem - 1:
            notional_paid = 0.0 if base_state.no_notional_repayment else 1.0
            k_put = float(base_state.strike)
            k_low = float(base_state.lower_strike) if base_state.lower_strike is not None else 0.0
            ki_active = (base_state.ki_barrier is None or base_state.ki_barrier <= 0.0) or (eff_i_ki > 0.5)
            put_loss_val = 0.0
            if ki_active and k_put > max(k_low, 0.0):
                put_loss_val = max(k_put - w_opt, 0.0) - max(max(k_low, 0.0) - w_opt, 0.0)
            put_sign = 1.0 if base_state.is_long_basket else -1.0
            opt_paid += put_sign * float(base_state.option_gearing) * put_loss_val
            if base_state.participation_call_strike is not None and base_state.participation_call_rate != 0.0:
                opt_paid += float(base_state.participation_call_rate) * max(
                    w_opt - float(base_state.participation_call_strike), 0.0
                )
            if base_state.one_star_strike is not None and base_state.one_star_coupon != 0.0:
                if float(np.max(perf_m)) >= float(base_state.one_star_strike):
                    opt_paid += float(base_state.one_star_coupon)
            eff_autocalled = True
        else:
            notional_paid = 0.0

        pay_dt_m = pay_dts[m]
        if pay_dt_m > eff_as_of:
            df_pay_m = (
                float(np.asarray(disc_curve.df(pay_dt_m), dtype=np.float64).reshape(-1)[0])
                / disc_anchor
                if (disc_curve is not None and hasattr(disc_curve, "df"))
                else 1.0
            )
            crossed_pv += df_pay_m * (cpn_paid + notional_paid + opt_paid)

    # Subtract surviving funding payments ONLY when autocall/expiry terminates the future loop;
    # otherwise `_compute_funding_weights_per_period(..., period_start_idx=k_cross)` already prices them.
    if eff_autocalled and k_cross > 0 and base_state.funding_leg is not None:
        w_fund_all = _compute_funding_weights_per_period(
            base_state,
            as_of_override=eff_as_of,
            device=base_state.spots_t0.device,
        )
        crossed_pv -= float(w_fund_all[:k_cross].sum().detach().cpu().item())

    return k_cross, eff_autocalled, eff_i_ki, eff_c_mem, crossed_pv


_INV_SQRT2_PURE: float = 0.70710678118654752440084436210485
_INV_SQRT2PI_PURE: float = 0.39894228040143267793994605993438
_SQRT_HALF_PI_PURE: float = 1.2533141373155002512078826424055


def _ramp_survival_post_moments_lognormal_pure(
    f_cond: torch.Tensor,
    lam: torch.Tensor,
    b_ko: torch.Tensor,
    ko_spread: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph-break-free closed-form truncated-lognormal posterior (mu_u, var_u) for torch.compile."""
    f_c = f_cond.clamp(min=1e-16)
    l_c = lam.clamp(min=1e-12)
    b_c = b_ko.clamp(min=1e-8)
    m_u = torch.log(f_c) - 0.5 * l_c * l_c

    if abs(ko_spread) <= 1e-6:
        u_star = (torch.log(b_c) - m_u) / l_c
        return _truncated_upper_std_moments_torch(u_star)

    b_lo = torch.minimum(b_c, b_c + ko_spread)
    b_hi = torch.maximum(b_c, b_c + ko_spread)
    b_hi_pos = b_hi > 0.0
    b_hi_safe = b_hi.clamp(min=1e-16)
    db = (b_hi_safe - b_lo).clamp(min=1e-12)
    u_hi = (torch.log(b_hi_safe) - m_u) / l_c
    has_lo = b_lo > 0.0
    b_lo_pos = torch.where(has_lo, b_lo, torch.ones_like(b_lo))
    u_lo = torch.where(has_lo, (torch.log(b_lo_pos) - m_u) / l_c, torch.full_like(u_hi, -100.0))

    use_scaled = u_hi < -4.0

    f_cond_d = torch.where(use_scaled, b_hi_safe, f_c)
    u_hi_d = torch.where(use_scaled, torch.zeros_like(u_hi), u_hi)
    u_lo_d = torch.where(use_scaled | (~has_lo), torch.full_like(u_hi, -100.0), u_lo)
    Phi_hi = torch.special.ndtr(u_hi_d)
    Phi_hi_m = torch.special.ndtr(u_hi_d - l_c)
    phi_hi = torch.exp(-0.5 * u_hi_d * u_hi_d) * _INV_SQRT2PI_PURE
    Phi_lo = torch.where(has_lo, torch.special.ndtr(u_lo_d), torch.zeros_like(u_hi))
    Phi_lo_m = torch.where(has_lo, torch.special.ndtr(u_lo_d - l_c), torch.zeros_like(u_hi))
    phi_lo = torch.where(has_lo, torch.exp(-0.5 * u_lo_d * u_lo_d) * _INV_SQRT2PI_PURE, torch.zeros_like(u_hi))

    M0_d = (b_hi_safe * Phi_hi - b_lo * Phi_lo - f_cond_d * (Phi_hi_m - Phi_lo_m)) / db
    M1_d = -(l_c * f_cond_d / db) * (Phi_hi_m - Phi_lo_m)
    M2_d = M0_d + l_c * M1_d + (l_c / db) * (b_hi_safe * phi_hi - b_lo * phi_lo)
    d_tiny = M0_d <= 1e-300
    M0_d_safe = torch.where(use_scaled | d_tiny, torch.ones_like(M0_d), M0_d)
    mu_d = M1_d / M0_d_safe
    var_d = (M2_d / M0_d_safe - mu_d * mu_d).clamp(min=1e-12)

    u_hi_s = torch.where(use_scaled, u_hi, torch.full_like(u_hi, -4.0))
    u_lo_s = torch.where(use_scaled & has_lo, u_lo, torch.full_like(u_hi, -1e9))
    exp_lo = torch.where(has_lo, torch.exp(-0.5 * (u_lo_s - u_hi_s) * (u_lo_s + u_hi_s)), torch.zeros_like(u_hi))
    r_hi = _SQRT_HALF_PI_PURE * torch.special.erfcx(-u_hi_s * _INV_SQRT2_PURE)
    r_hi_m = _SQRT_HALF_PI_PURE * torch.special.erfcx(-(u_hi_s - l_c) * _INV_SQRT2_PURE)
    r_lo = torch.where(has_lo, _SQRT_HALF_PI_PURE * torch.special.erfcx(-u_lo_s * _INV_SQRT2_PURE), torch.zeros_like(u_hi))
    r_lo_m = torch.where(
        has_lo, _SQRT_HALF_PI_PURE * torch.special.erfcx(-(u_lo_s - l_c) * _INV_SQRT2_PURE), torch.zeros_like(u_hi)
    )

    M0_s = (b_hi_safe * (r_hi - r_hi_m) - b_lo * exp_lo * (r_lo - r_lo_m)) / db
    M1_s = -(l_c / db) * (b_hi_safe * r_hi_m - b_lo * exp_lo * r_lo_m)
    M2_s = M0_s + l_c * M1_s + (l_c / db) * (b_hi_safe - b_lo * exp_lo)
    s_tiny = M0_s <= 1e-300
    M0_s_safe = torch.where((~use_scaled) | s_tiny, torch.ones_like(M0_s), M0_s)
    mu_s = M1_s / M0_s_safe
    var_s = (M2_s / M0_s_safe - mu_s * mu_s).clamp(min=1e-12)

    mu = torch.where(use_scaled, mu_s, mu_d)
    var = torch.where(use_scaled, var_s, var_d)
    tiny_mask = torch.where(use_scaled, s_tiny, d_tiny)
    mu_fb, var_fb = _truncated_upper_std_moments_torch(u_hi)
    mu = torch.where(tiny_mask, mu_fb, mu)
    var = torch.where(tiny_mask, var_fb, var)
    mu = torch.where(b_hi_pos, mu, torch.zeros_like(mu))
    var = torch.where(b_hi_pos, var, torch.ones_like(var))
    return mu, var


def _clark_extremum_pure(
    f_in: torch.Tensor,
    v_in: torch.Tensor,
    c_in: torch.Tensor,
    ord_f: torch.Tensor,
    is_best_of: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetrized Clark extrema reduction matching analytical_baseline & C++23 Tape AD."""
    del ord_f
    return _clark_symmetrized_torch(f_in, v_in, c_in, None, is_best_of)


def _tier1_baseline_pure_torch(
    fp: torch.Tensor,
    vp: torch.Tensor,
    corr: torch.Tensor,
    ko_t: torch.Tensor,
    cb_t: torch.Tensor,
    ig_t: torch.Tensor,
    ord_call: torch.Tensor,
    ord_cpn: torch.Tensor,
    ord_opt: torch.Tensor,
    *,
    is_call_flags: tuple[bool, ...],
    call_code: int,
    cpn_code: int,
    opt_code: int,
    cpn_b_type: str,
    cpn_b_dir: str,
    force_cpn: bool,
    allow_exp: bool,
    ko_spread: float,
    cpn_spread: float,
    ki_spread: float,
    strike: float,
    option_gearing: float,
    lower_strike_val: float,
    has_active_ki: bool,
    ki_barrier_val: float,
    ko_b_term: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Graph-break-free batched float64 Tier-1 baseline matching analytical_baseline.py."""
    b_sz, d_assets, m_rem = fp.shape
    device = fp.device
    call_is_best = call_code == 1
    cpn_is_best = cpn_code == 1
    opt_is_best = opt_code == 1
    cpn_differs = cpn_code != call_code
    opt_differs = opt_code != call_code
    cpn_dir_down = cpn_b_dir.strip().lower() == "down"
    cpn_type_out = cpn_b_type.strip().lower() == "out"

    f_all = fp.transpose(1, 2).reshape(b_sz * m_rem, d_assets)
    v_all = vp.transpose(1, 2).reshape(b_sz * m_rem, d_assets)
    c_all = corr.unsqueeze(1).expand(b_sz, m_rem, d_assets, d_assets).reshape(b_sz * m_rem, d_assets, d_assets)
    fw_flat, vw_flat, w_flat = _clark_extremum_pure(f_all, v_all, c_all, ord_call, call_is_best)
    f_w = fw_flat.view(b_sz, m_rem)
    v_w = vw_flat.view(b_sz, m_rem)
    w_call_all = w_flat.view(b_sz, m_rem, d_assets)
    proxy_delta_weights = w_call_all[:, -1, :]

    if cpn_differs:
        fc_flat, vc_flat, wc_flat = _clark_extremum_pure(f_all, v_all, c_all, ord_cpn, cpn_is_best)
        f_cpn_raw = fc_flat.view(b_sz, m_rem)
        v_cpn_raw = vc_flat.view(b_sz, m_rem)
        w_cpn_all = wc_flat.view(b_sz, m_rem, d_assets)
        # Cross-index correlation between callability and coupon extrema
        a_cc = w_call_all * vp.transpose(1, 2)
        b_cc = w_cpn_all * vp.transpose(1, 2)
        cov_cc = torch.einsum("bmi,bij,bmj->bm", a_cc, corr, b_cc)
        va_cc = torch.einsum("bmi,bij,bmj->bm", a_cc, corr, a_cc).clamp(min=1e-30)
        vb_cc = torch.einsum("bmi,bij,bmj->bm", b_cc, corr, b_cc).clamp(min=1e-30)
        rho_cc_all = (cov_cc / torch.sqrt(va_cc * vb_cc)).clamp(-0.9999, 0.9999)
    else:
        f_cpn_raw = f_w
        v_cpn_raw = v_w
        rho_cc_all = torch.ones((b_sz, m_rem), dtype=torch.float64, device=device)

    z_mean = torch.zeros(b_sz, dtype=torch.float64, device=device)
    z_std = torch.ones(b_sz, dtype=torch.float64, device=device)
    surv_cur = torch.ones(b_sz, dtype=torch.float64, device=device)

    p_list: list[torch.Tensor] = []
    surv_list: list[torch.Tensor] = []
    surv_prev_list: list[torch.Tensor] = []
    q_list: list[torch.Tensor] = []
    f_cond_term = f_w[:, -1]
    v_cond_term = v_w[:, -1]
    z_mean_term = z_mean
    z_std_term = z_std

    for m in range(m_rem):
        fw_m = f_w[:, m].clamp(min=1e-16)
        vw_m = v_w[:, m].clamp(min=1e-12)
        if m == 0:
            f_cond_m = fw_m
            v_cond_m = vw_m
        else:
            f_cond_m = fw_m * torch.exp(vw_m * z_mean - 0.5 * (vw_m * vw_m) * (1.0 - z_std * z_std))
            v_cond_m = (vw_m * z_std).clamp(min=1e-12)

        if m == m_rem - 1:
            f_cond_term = f_cond_m
            v_cond_term = v_cond_m
            z_mean_term = z_mean
            z_std_term = z_std

        ko_b_m = ko_t[m]
        is_call_m = is_call_flags[m]
        if is_call_m:
            h_m = _digital_call_spread_prob_torch(f_cond_m, v_cond_m, ko_b_m, ko_spread).clamp(0.0, 1.0 - 1e-7)
        else:
            h_m = torch.zeros(b_sz, dtype=torch.float64, device=device)

        p_m = surv_cur * h_m
        surv_next = surv_cur * (1.0 - h_m)

        if cpn_differs:
            fc_raw = f_cpn_raw[:, m].clamp(min=1e-16)
            vc_raw = v_cpn_raw[:, m].clamp(min=1e-12)
            rho_cc = rho_cc_all[:, m]
            z_mc = rho_cc * z_mean
            z_vc = rho_cc * rho_cc * (z_std * z_std) + (1.0 - rho_cc * rho_cc)
            f_cpn_cond = fc_raw * torch.exp(vc_raw * z_mc - 0.5 * (vc_raw * vc_raw) * (1.0 - z_vc))
            v_cpn_cond = (vc_raw * torch.sqrt(z_vc.clamp(min=1e-12))).clamp(min=1e-12)
        else:
            f_cpn_cond = f_cond_m
            v_cpn_cond = v_cond_m

        eff_cpn_b_m = (
            torch.minimum(cb_t[m], ko_b_m)
            if (not cpn_differs and force_cpn and is_call_m and not cpn_dir_down and not cpn_type_out)
            else cb_t[m]
        )
        q_raw_m = _eval_coupon_prob_torch(
            f_cpn_cond,
            v_cpn_cond,
            eff_cpn_b_m,
            cpn_spread,
            direction=cpn_b_dir,
            b_type=cpn_b_type,
            is_normal=False,
            index_code=cpn_code,
        )
        if force_cpn and is_call_m:
            q_raw_m = torch.maximum(q_raw_m, h_m)
        q_m = torch.where(ig_t[m] > 0.5, torch.ones_like(q_raw_m), q_raw_m)

        p_list.append(p_m)
        surv_prev_list.append(surv_cur)
        surv_list.append(surv_next)
        q_list.append(q_m)

        if m < m_rem - 1:
            if is_call_m:
                mu_u, var_u = _ramp_survival_post_moments_lognormal_pure(f_cond_m, v_cond_m, ko_b_m, ko_spread)
                z_post_mean = z_mean + z_std * mu_u
                z_post_var = (z_std * z_std * var_u).clamp(min=1e-12)
            else:
                z_post_mean = z_mean
                z_post_var = z_std * z_std
            if is_call_m or m > 0:
                r_next = _euler_next_corr_torch(
                    fp[:, :, m],
                    vp[:, :, m].clamp(min=1e-12),
                    w_call_all[:, m, :],
                    fp[:, :, m + 1],
                    vp[:, :, m + 1].clamp(min=1e-12),
                    w_call_all[:, m + 1, :],
                    corr,
                    False,
                )
                z_mean = r_next * z_post_mean
                z_std = torch.sqrt((r_next * r_next * z_post_var + (1.0 - r_next * r_next)).clamp(min=1e-12))

        surv_cur = surv_next

    p_proxy = torch.stack(p_list, dim=1)
    q_proxy = torch.stack(q_list, dim=1)
    surv_term = surv_list[-1]
    surv_prev_term = surv_prev_list[-1]

    if opt_differs:
        f_ot_t = fp[:, :, -1]
        v_ot_t = vp[:, :, -1]
        f_opt_raw, v_opt_raw, w_opt_t = _clark_extremum_pure(f_ot_t, v_ot_t, corr, ord_opt, opt_is_best)
        a_co = w_call_all[:, -1, :] * v_ot_t
        b_co = w_opt_t * v_ot_t
        cov_co = torch.einsum("bi,bij,bj->b", a_co, corr, b_co)
        va_co = torch.einsum("bi,bij,bj->b", a_co, corr, a_co).clamp(min=1e-30)
        vb_co = torch.einsum("bi,bij,bj->b", b_co, corr, b_co).clamp(min=1e-30)
        rho_co = (cov_co / torch.sqrt(va_co * vb_co)).clamp(-0.9999, 0.9999)
        z_mo = rho_co * z_mean_term
        z_vo = rho_co * rho_co * (z_std_term * z_std_term) + (1.0 - rho_co * rho_co)
        v_opt_r = v_opt_raw.clamp(min=1e-12)
        f_opt_cond = f_opt_raw.clamp(min=1e-16) * torch.exp(
            v_opt_r * z_mo - 0.5 * (v_opt_r * v_opt_r) * (1.0 - z_vo)
        )
        v_opt_cond = (v_opt_r * torch.sqrt(z_vo.clamp(min=1e-12))).clamp(min=1e-12)
    else:
        f_opt_cond = f_cond_term
        v_opt_cond = v_cond_term

    if strike <= 1e-6 or option_gearing <= 0.0:
        l_put_proxy = torch.zeros(b_sz, dtype=torch.float64, device=device)
    else:
        use_direct_ko = allow_exp and is_call_flags[-1] and not opt_differs
        surv_mult = surv_prev_term if use_direct_ko else surv_term
        put_cond = _exact_piecewise_terminal_put_torch(
            f_opt_cond,
            v_opt_cond,
            False,
            strike,
            lower_strike_val,
            has_active_ki,
            ki_barrier_val,
            ki_spread,
            use_direct_ko,
            ko_b_term,
            ko_spread,
        )
        l_put_proxy = surv_mult * put_cond

    return f_w, v_w, p_proxy, q_proxy, l_put_proxy, proxy_delta_weights


_TIER1_CPP_AD_SRC = r"""
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kMaxD = 32;
constexpr double kInvSqrt2 = 0.70710678118654752440084436210485;
constexpr double kInvSqrt2Pi = 0.39894228040143267793994605993438;
constexpr double kSqrtTwoOverPi = 0.79788456080286535587989211986876;
constexpr double kSqrtHalfPi = 1.2533141373155002512078826424055;

inline double ndtr_d(double x) noexcept {
    return 0.5 * std::erfc(-x * kInvSqrt2);
}

inline double exact_erfcx_d(double x) noexcept {
    if (std::isnan(x)) return x;
    double y = std::abs(x);
    if (y <= 0.46875) {
        constexpr double a[5] = {3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02, 3.20937758913846947e03, 1.85777706184603153e-01};
        constexpr double b[4] = {2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03, 2.84423683343917062e03};
        double ysq = y * y;
        double xnum = a[4] * ysq, xden = ysq;
        for (int i = 0; i < 3; ++i) {
            xnum = (xnum + a[i]) * ysq;
            xden = (xden + b[i]) * ysq;
        }
        double erf_v = y * (xnum + a[3]) / (xden + b[3]);
        double erfc_v = 1.0 - (x < 0.0 ? -erf_v : erf_v);
        return std::exp(ysq) * erfc_v;
    }
    if (y <= 4.0) {
        constexpr double c[9] = {5.64188496988670089e-01, 8.88314979438837594e00, 6.61191906371416295e01, 2.98635138197400131e02, 8.81952221241769090e02, 1.71204761263407058e03, 2.05107837782607147e03, 1.23033935479799725e03, 2.15311535474403846e-08};
        constexpr double d[8] = {1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02, 1.62138957456669019e03, 3.29079923573345963e03, 4.36261909014324716e03, 3.43936767414372164e03, 1.23033935480374942e03};
        double xnum = c[8] * y, xden = y;
        for (int i = 0; i < 7; ++i) {
            xnum = (xnum + c[i]) * y;
            xden = (xden + d[i]) * y;
        }
        double res = (xnum + c[7]) / (xden + d[7]);
        if (x < 0.0) res = 2.0 * std::exp(y * y) - res;
        return res;
    }
    if (x < 0.0) {
        double ysq = y * y;
        if (ysq > 709.0) return std::numeric_limits<double>::infinity();
        return 2.0 * std::exp(ysq);
    }
    double ysq = y * y;
    if (ysq > 1e300) return 0.0;
    constexpr double p[6] = {3.05326634961232344e-01, 3.60344899949804439e-01, 1.25781726111229246e-01, 1.60837851487422766e-02, 6.58749161529837803e-04, 1.63153871373020978e-02};
    constexpr double q[5] = {2.56852019228982235e00, 1.87295284992346047e00, 5.27905102951428412e-01, 6.05183413124413191e-02, 2.33520497626869185e-03};
    double z = 1.0 / ysq;
    double xnum = p[5] * z, xden = z;
    for (int i = 0; i < 4; ++i) {
        xnum = (xnum + p[i]) * z;
        xden = (xden + q[i]) * z;
    }
    return (0.56418958354775628694807945156077 - z * (xnum + p[4]) / (xden + q[4])) / x;
}

struct Node {
    int p1;
    int p2;
    double w1;
    double w2;
};

struct AD {
    double v;
    int id;
    struct Tape* t;
    AD() : v(0.0), id(-1), t(nullptr) {}
    AD(double val) : v(val), id(-1), t(nullptr) {}
    AD(double val, int idx, Tape* tp) : v(val), id(idx), t(tp) {}
};

struct Tape {
    std::vector<Node> nodes;
    std::vector<double> adj;
    std::vector<AD> fp, vp, corr, f_w, v_w, w_call, f_cpn, v_cpn, rho_cc, p_proxy, q_proxy;

    Tape() { nodes.reserve(16384); }
    void reset() { nodes.clear(); }
    int leaf() {
        int id = static_cast<int>(nodes.size());
        nodes.push_back({-1, -1, 0.0, 0.0});
        return id;
    }
    int unary(int p, double w) {
        if (p < 0 || w == 0.0) return -1;
        int id = static_cast<int>(nodes.size());
        nodes.push_back({p, -1, w, 0.0});
        return id;
    }
    int binary(int p1, int p2, double w1, double w2) {
        if (p1 < 0 && p2 < 0) return -1;
        if (p1 < 0) return unary(p2, w2);
        if (p2 < 0) return unary(p1, w1);
        int id = static_cast<int>(nodes.size());
        nodes.push_back({p1, p2, w1, w2});
        return id;
    }
    void zero_grad() { adj.assign(nodes.size(), 0.0); }
    void backward_sweep() {
        for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
            double g = adj[i];
            if (g == 0.0) continue;
            const Node& n = nodes[i];
            if (n.p1 >= 0) adj[n.p1] += g * n.w1;
            if (n.p2 >= 0) adj[n.p2] += g * n.w2;
        }
    }
};

inline AD make_leaf(Tape& tp, double v) { return AD(v, tp.leaf(), &tp); }

inline AD operator+(const AD& a, const AD& b) {
    Tape* t = a.t ? a.t : b.t;
    return t ? AD(a.v + b.v, t->binary(a.id, b.id, 1.0, 1.0), t) : AD(a.v + b.v);
}
inline AD operator+(const AD& a, double b) { return AD(a.v + b, a.id, a.t); }
inline AD operator+(double a, const AD& b) { return AD(a + b.v, b.id, b.t); }
inline AD operator-(const AD& a, const AD& b) {
    Tape* t = a.t ? a.t : b.t;
    return t ? AD(a.v - b.v, t->binary(a.id, b.id, 1.0, -1.0), t) : AD(a.v - b.v);
}
inline AD operator-(const AD& a, double b) { return AD(a.v - b, a.id, a.t); }
inline AD operator-(double a, const AD& b) {
    return (!b.t || b.id < 0) ? AD(a - b.v) : AD(a - b.v, b.t->unary(b.id, -1.0), b.t);
}
inline AD operator-(const AD& a) {
    return (!a.t || a.id < 0) ? AD(-a.v) : AD(-a.v, a.t->unary(a.id, -1.0), a.t);
}
inline AD operator*(const AD& a, const AD& b) {
    Tape* t = a.t ? a.t : b.t;
    return t ? AD(a.v * b.v, t->binary(a.id, b.id, b.v, a.v), t) : AD(a.v * b.v);
}
inline AD operator*(const AD& a, double b) {
    return (!a.t || a.id < 0) ? AD(a.v * b) : AD(a.v * b, a.t->unary(a.id, b), a.t);
}
inline AD operator*(double a, const AD& b) { return b * a; }
inline AD operator/(const AD& a, const AD& b) {
    Tape* t = a.t ? a.t : b.t;
    if (!t) return AD(a.v / b.v);
    double inv = 1.0 / b.v;
    return AD(a.v * inv, t->binary(a.id, b.id, inv, -a.v * inv * inv), t);
}
inline AD operator/(const AD& a, double b) { return a * (1.0 / b); }
inline AD operator/(double a, const AD& b) {
    if (!b.t || b.id < 0) return AD(a / b.v);
    double inv = 1.0 / b.v;
    return AD(a * inv, b.t->unary(b.id, -a * inv * inv), b.t);
}
inline AD ad_log(const AD& a) {
    return (!a.t || a.id < 0) ? AD(std::log(a.v)) : AD(std::log(a.v), a.t->unary(a.id, 1.0 / a.v), a.t);
}
inline AD ad_exp(const AD& a) {
    double e = std::exp(a.v);
    return (!a.t || a.id < 0) ? AD(e) : AD(e, a.t->unary(a.id, e), a.t);
}
inline AD ad_sqrt(const AD& a) {
    double s = std::sqrt(a.v);
    if (!a.t || a.id < 0) return AD(s);
    return AD(s, a.t->unary(a.id, s > 0.0 ? 0.5 / s : 0.0), a.t);
}
inline AD ad_ndtr(const AD& a) {
    double p = ndtr_d(a.v);
    if (!a.t || a.id < 0) return AD(p);
    return AD(p, a.t->unary(a.id, kInvSqrt2Pi * std::exp(-0.5 * a.v * a.v)), a.t);
}
inline AD ad_erfcx(const AD& a) {
    double r = exact_erfcx_d(a.v);
    if (!a.t || a.id < 0) return AD(r);
    return AD(r, a.t->unary(a.id, 2.0 * a.v * r - 1.1283791670955125738961589031215), a.t);
}
inline AD ad_clamp_min(const AD& a, double lo) { return (a.v < lo) ? AD(lo) : a; }
inline AD ad_clamp(const AD& a, double lo, double hi) {
    if (a.v < lo) return AD(lo);
    if (a.v > hi) return AD(hi);
    return a;
}
inline AD ad_min(const AD& a, const AD& b) {
    if (a.v < b.v) return a;
    if (b.v < a.v) return b;
    Tape* t = a.t ? a.t : b.t;
    return t ? AD(a.v, t->binary(a.id, b.id, 0.5, 0.5), t) : AD(a.v);
}
inline AD ad_max(const AD& a, const AD& b) { return (a.v >= b.v) ? a : b; }
inline double ad_log(double a) noexcept { return std::log(a); }
inline double ad_exp(double a) noexcept { return std::exp(a); }
inline double ad_sqrt(double a) noexcept { return std::sqrt(a); }
inline double ad_ndtr(double a) noexcept { return ndtr_d(a); }
inline double ad_clamp_min(double a, double lo) noexcept { return (a < lo) ? lo : a; }
inline double ad_clamp(double a, double lo, double hi) noexcept {
    if (a < lo) return lo;
    if (a > hi) return hi;
    return a;
}

inline AD black76_call(const AD& f, double k, const AD& v) {
    AD fc = ad_clamp_min(f, 1e-16), vc = ad_clamp_min(v, 1e-12);
    double kc = k > 1e-16 ? k : 1e-16;
    AD d1 = (ad_log(fc / kc) + 0.5 * vc * vc) / vc;
    return ad_clamp_min(fc * ad_ndtr(d1) - kc * ad_ndtr(d1 - vc), 0.0);
}

inline AD digital_call_spread(const AD& f, const AD& v, double barrier, double spread) {
    AD fc = ad_clamp_min(f, 1e-16), vc = ad_clamp_min(v, 1e-12);
    double kc = barrier > 1e-16 ? barrier : 1e-16;
    if (std::abs(spread) <= 1e-12) return ad_ndtr((ad_log(fc / kc) - 0.5 * vc * vc) / vc);
    double k_lo = spread < 0.0 ? std::max(kc + spread, 1e-16) : kc;
    double k_hi = spread < 0.0 ? kc : std::max(kc + spread, 1e-16);
    double denom = std::max(k_hi - k_lo, 1e-12);
    return ad_clamp((black76_call(fc, k_lo, vc) - black76_call(fc, k_hi, vc)) / denom, 0.0, 1.0);
}

inline AD eval_coupon_prob(const AD& f, const AD& v, double barrier, double spread, int dir_down, int type_out) {
    double eff_sp = dir_down ? -spread : spread;
    AD p_up = digital_call_spread(f, v, barrier, eff_sp);
    AD p_ev = dir_down ? (1.0 - p_up) : p_up;
    return type_out ? (1.0 - p_ev) : p_ev;
}

inline void trunc_upper_moments(const AD& alpha, AD& mu_out, AD& var_out) {
    if (alpha.v <= -4.0) {
        AD x_cf = -alpha;
        AD v(0.0);
        for (int k = 80; k >= 3; --k) v = static_cast<double>(k) / (x_cf + v);
        AD r2 = 2.0 / (x_cf + v);
        AD psi = 1.0 / (x_cf + r2);
        mu_out = -x_cf - psi;
        var_out = ad_clamp_min(psi * (r2 - psi), 1e-12);
        return;
    }
    if (alpha.v >= 26.0) { mu_out = AD(0.0); var_out = AD(1.0); return; }
    AD lam = kSqrtTwoOverPi / ad_erfcx(-alpha * kInvSqrt2);
    mu_out = -lam;
    var_out = ad_clamp_min(1.0 - alpha * lam - lam * lam, 1e-12);
}

inline void ramp_survival_post_moments(const AD& f_cond, const AD& lam, double b_ko, double ko_spread, AD& mu_out, AD& var_out) {
    AD fc = ad_clamp_min(f_cond, 1e-16), lc = ad_clamp_min(lam, 1e-12);
    double bc = b_ko > 1e-8 ? b_ko : 1e-8;
    AD m_u = ad_log(fc) - 0.5 * lc * lc;
    if (std::abs(ko_spread) <= 1e-6) {
        trunc_upper_moments((std::log(bc) - m_u) / lc, mu_out, var_out);
        return;
    }
    double b_lo = std::min(bc, bc + ko_spread), b_hi = std::max(bc, bc + ko_spread);
    if (b_hi <= 0.0) { mu_out = AD(0.0); var_out = AD(1.0); return; }
    double b_hi_safe = std::max(b_hi, 1e-16);
    double db = std::max(b_hi_safe - b_lo, 1e-12);
    AD u_hi = (std::log(b_hi_safe) - m_u) / lc;
    bool has_lo = b_lo > 0.0;
    AD u_lo = has_lo ? (std::log(b_lo) - m_u) / lc : AD(-100.0);
    if (u_hi.v >= -4.0) {
        AD Phi_hi = ad_ndtr(u_hi), Phi_hi_m = ad_ndtr(u_hi - lc), phi_hi = ad_exp(-0.5 * u_hi * u_hi) * kInvSqrt2Pi;
        AD Phi_lo = has_lo ? ad_ndtr(u_lo) : AD(0.0);
        AD Phi_lo_m = has_lo ? ad_ndtr(u_lo - lc) : AD(0.0);
        AD phi_lo = has_lo ? ad_exp(-0.5 * u_lo * u_lo) * kInvSqrt2Pi : AD(0.0);
        AD M0 = (b_hi_safe * Phi_hi - b_lo * Phi_lo - fc * (Phi_hi_m - Phi_lo_m)) / db;
        AD M1 = -(lc * fc / db) * (Phi_hi_m - Phi_lo_m);
        AD M2 = M0 + lc * M1 + (lc / db) * (b_hi_safe * phi_hi - b_lo * phi_lo);
        if (M0.v > 1e-300) {
            mu_out = M1 / M0;
            var_out = ad_clamp_min(M2 / M0 - mu_out * mu_out, 1e-12);
            return;
        }
        trunc_upper_moments(u_hi, mu_out, var_out);
        return;
    }
    AD exp_lo = has_lo ? ad_exp(-0.5 * (u_lo - u_hi) * (u_lo + u_hi)) : AD(0.0);
    AD r_hi = kSqrtHalfPi * ad_erfcx(-u_hi * kInvSqrt2), r_hi_m = kSqrtHalfPi * ad_erfcx(-(u_hi - lc) * kInvSqrt2);
    AD r_lo = has_lo ? kSqrtHalfPi * ad_erfcx(-u_lo * kInvSqrt2) : AD(0.0);
    AD r_lo_m = has_lo ? kSqrtHalfPi * ad_erfcx(-(u_lo - lc) * kInvSqrt2) : AD(0.0);
    AD M0 = (b_hi_safe * (r_hi - r_hi_m) - b_lo * exp_lo * (r_lo - r_lo_m)) / db;
    AD M1 = -(lc / db) * (b_hi_safe * r_hi_m - b_lo * exp_lo * r_lo_m);
    AD M2 = M0 + lc * M1 + (lc / db) * (b_hi_safe - b_lo * exp_lo);
    if (M0.v > 1e-300) {
        mu_out = M1 / M0;
        var_out = ad_clamp_min(M2 / M0 - mu_out * mu_out, 1e-12);
        return;
    }
    trunc_upper_moments(u_hi, mu_out, var_out);
}

template <typename Num>
inline void clark_one_order_t(
    int d, const Num* f_in, const Num* v_in, const Num* c_in, const int64_t* order,
    bool is_best_of, Num& f_out, Num& v_out, Num* w_out
) {
    if (d == 1) { f_out = ad_clamp_min(f_in[0], 1e-16); v_out = ad_clamp_min(v_in[0], 1e-12); w_out[0] = Num(1.0); return; }
    Num f_ord[kMaxD], v_ord[kMaxD], c_ord[kMaxD][kMaxD], rho_rem[kMaxD], w1_steps[kMaxD], w2_steps[kMaxD];
    int inv_ord[kMaxD];
    for (int i = 0; i < d; ++i) {
        int oi = static_cast<int>(order[i]);
        inv_ord[oi] = i;
        f_ord[i] = ad_clamp_min(f_in[oi], 1e-16);
        v_ord[i] = ad_clamp_min(v_in[oi], 1e-12);
    }
    for (int i = 0; i < d; ++i) {
        int oi = static_cast<int>(order[i]);
        for (int j = 0; j < d; ++j) c_ord[i][j] = c_in[oi * d + static_cast<int>(order[j])];
    }
    Num f_cur = f_ord[0], v_cur = v_ord[0];
    for (int k = 1; k < d; ++k) rho_rem[k] = c_ord[0][k];
    for (int j = 1; j < d; ++j) {
        const Num& f2 = f_ord[j], &v2 = v_ord[j];
        Num rho12 = ad_clamp(rho_rem[j], -0.999999, 0.999999);
        Num v1s = v_cur * v_cur, v2s = v2 * v2;
        Num var_m = ad_clamp_min(v1s + v2s - 2.0 * rho12 * v_cur * v2, 1e-16);
        Num sig_m = ad_sqrt(var_m);
        Num lfr = ad_log(f_cur / f2);
        Num d1 = (lfr + 0.5 * var_m) / sig_m, d2 = d1 - sig_m;
        Num w1 = is_best_of ? ad_ndtr(d1) : ad_ndtr(-d1);
        Num w2 = is_best_of ? ad_ndtr(-d2) : ad_ndtr(d2);
        Num e_m1 = ad_clamp_min(f_cur * w1 + f2 * w2, 1e-16);
        w1_steps[j - 1] = w1; w2_steps[j - 1] = w2;
        Num f1_sq = ad_clamp_min((f_cur * f_cur) * ad_exp(v1s), 1e-30);
        Num f2_sq = ad_clamp_min((f2 * f2) * ad_exp(v2s), 1e-30);
        Num sig_m2 = 2.0 * sig_m;
        Num d1_sq = (ad_log(f1_sq / f2_sq) + 0.5 * sig_m2 * sig_m2) / sig_m2, d2_sq = d1_sq - sig_m2;
        Num e_m2 = is_best_of ? (f1_sq * ad_ndtr(d1_sq) + f2_sq * ad_ndtr(-d2_sq))
                              : (f1_sq * ad_ndtr(-d1_sq) + f2_sq * ad_ndtr(d2_sq));
        Num v_new = ad_sqrt(ad_clamp_min(ad_log(ad_clamp_min(e_m2 / ad_clamp_min(e_m1 * e_m1, 1e-30), 1.0)), 1e-16));
        for (int k = j + 1; k < d; ++k) {
            const Num& vk = v_ord[k];
            Num ea1 = rho_rem[k] * v_cur * vk, ea2 = c_ord[j][k] * v2 * vk;
            Num f1k = f_cur * ad_exp(ea1), f2k = f2 * ad_exp(ea2);
            Num d1k = (lfr + (ea1 - ea2) + 0.5 * var_m) / sig_m, d2k = d1k - sig_m;
            Num empk = is_best_of ? (f1k * ad_ndtr(d1k) + f2k * ad_ndtr(-d2k))
                                  : (f1k * ad_ndtr(-d1k) + f2k * ad_ndtr(d2k));
            Num cov_log = ad_log(ad_clamp_min(empk / e_m1, 1e-16));
            rho_rem[k] = ad_clamp(cov_log / ad_clamp_min(v_new * vk, 1e-16), -0.999999, 0.999999);
        }
        f_cur = e_m1; v_cur = v_new;
    }
    Num suff[kMaxD], w_ord[kMaxD];
    suff[d - 1] = Num(1.0);
    for (int k = d - 2; k >= 0; --k) suff[k] = suff[k + 1] * w1_steps[k];
    w_ord[0] = suff[0];
    for (int k = 1; k < d; ++k) w_ord[k] = w2_steps[k - 1] * suff[k];
    for (int i = 0; i < d; ++i) w_out[i] = w_ord[inv_ord[i]];
    f_out = f_cur; v_out = v_cur;
}

template <typename Num>
inline void clark_pair_merge_t(
    const Num& f1, const Num& v1, const Num& f2, const Num& v2, const Num& rho12,
    int n_rem, const Num* v_rem, const Num* r1_rem, const Num* r2_rem,
    bool best_of, Num& e_m1, Num& v_new, Num& w1, Num& w2, Num* rho_m_rem
) {
    Num rho_c = ad_clamp(rho12, -0.999999, 0.999999);
    Num v1s = v1 * v1, v2s = v2 * v2;
    Num var_m = ad_clamp_min(v1s + v2s - 2.0 * rho_c * v1 * v2, 1e-16);
    Num sig_m = ad_sqrt(var_m);
    Num lfr = ad_log(f1 / f2);
    Num d1 = (lfr + 0.5 * var_m) / sig_m;
    Num d2 = d1 - sig_m;
    w1 = best_of ? ad_ndtr(d1) : ad_ndtr(-d1);
    w2 = best_of ? ad_ndtr(-d2) : ad_ndtr(d2);
    e_m1 = ad_clamp_min(f1 * w1 + f2 * w2, 1e-16);
    Num f1_sq = ad_clamp_min((f1 * f1) * ad_exp(v1s), 1e-30);
    Num f2_sq = ad_clamp_min((f2 * f2) * ad_exp(v2s), 1e-30);
    Num sig_m2 = 2.0 * sig_m, var_m2 = sig_m2 * sig_m2;
    Num d1_sq = (ad_log(f1_sq / f2_sq) + 0.5 * var_m2) / sig_m2;
    Num d2_sq = d1_sq - sig_m2;
    Num e_m2 = best_of ? (f1_sq * ad_ndtr(d1_sq) + f2_sq * ad_ndtr(-d2_sq))
                       : (f1_sq * ad_ndtr(-d1_sq) + f2_sq * ad_ndtr(d2_sq));
    v_new = ad_sqrt(ad_clamp_min(ad_log(ad_clamp_min(e_m2 / ad_clamp_min(e_m1 * e_m1, 1e-30), 1.0)), 1e-16));
    for (int k = 0; k < n_rem; ++k) {
        const Num& vk = v_rem[k];
        Num ea1 = r1_rem[k] * v1 * vk;
        Num ea2 = r2_rem[k] * v2 * vk;
        Num f1k = f1 * ad_exp(ea1), f2k = f2 * ad_exp(ea2);
        Num d1k = (lfr + (ea1 - ea2) + 0.5 * var_m) / sig_m;
        Num d2k = d1k - sig_m;
        Num empk = best_of ? (f1k * ad_ndtr(d1k) + f2k * ad_ndtr(-d2k))
                           : (f1k * ad_ndtr(-d1k) + f2k * ad_ndtr(d2k));
        Num cov_log = ad_log(ad_clamp_min(empk / e_m1, 1e-16));
        rho_m_rem[k] = ad_clamp(cov_log / ad_clamp_min(v_new * vk, 1e-16), -0.999999, 0.999999);
    }
}

template <typename Num>
inline void eval_d4_tree_t(
    const Num* f_safe, const Num* v_safe, const Num* c_in,
    int a, int b, int c, int e_d, bool is_best_of,
    Num& f_root, Num& v_root, Num& wa, Num& wb, Num& wc, Num& wd
) {
    Num v_rem1[2] = {v_safe[c], v_safe[e_d]};
    Num r1_1[2] = {c_in[a * 4 + c], c_in[a * 4 + e_d]};
    Num r2_1[2] = {c_in[b * 4 + c], c_in[b * 4 + e_d]};
    Num f_ab, v_ab, w_a, w_b, rho_ab_cd[2];
    clark_pair_merge_t(
        f_safe[a], v_safe[a], f_safe[b], v_safe[b], c_in[a * 4 + b],
        2, v_rem1, r1_1, r2_1, is_best_of, f_ab, v_ab, w_a, w_b, rho_ab_cd
    );
    Num f_cd, v_cd, w_c, w_d, rho_cd_ab[1];
    clark_pair_merge_t(
        f_safe[c], v_safe[c], f_safe[e_d], v_safe[e_d], c_in[c * 4 + e_d],
        1, &v_ab, &rho_ab_cd[0], &rho_ab_cd[1], is_best_of, f_cd, v_cd, w_c, w_d, rho_cd_ab
    );
    Num w_ab, w_cd;
    clark_pair_merge_t<Num>(
        f_ab, v_ab, f_cd, v_cd, rho_cd_ab[0],
        0, nullptr, nullptr, nullptr, is_best_of, f_root, v_root, w_ab, w_cd, nullptr
    );
    wa = w_ab * w_a; wb = w_ab * w_b; wc = w_cd * w_c; wd = w_cd * w_d;
}

template <typename Num>
inline void eval_d5_tree_t(
    const Num* f_safe, const Num* v_safe, const Num* c_in,
    int a, int b, int c, int d_i, int e, bool is_best_of,
    Num& f_root, Num& v_root, Num& wa, Num& wb, Num& wc, Num& wd, Num& we
) {
    Num v_rem1[3] = {v_safe[c], v_safe[d_i], v_safe[e]};
    Num r1_1[3] = {c_in[a * 5 + c], c_in[a * 5 + d_i], c_in[a * 5 + e]};
    Num r2_1[3] = {c_in[b * 5 + c], c_in[b * 5 + d_i], c_in[b * 5 + e]};
    Num f_ab, v_ab, w_a, w_b, rho_ab_cde[3];
    clark_pair_merge_t(
        f_safe[a], v_safe[a], f_safe[b], v_safe[b], c_in[a * 5 + b],
        3, v_rem1, r1_1, r2_1, is_best_of, f_ab, v_ab, w_a, w_b, rho_ab_cde
    );
    Num v_rem2[2] = {v_ab, v_safe[e]};
    Num r1_2[2] = {rho_ab_cde[0], c_in[c * 5 + e]};
    Num r2_2[2] = {rho_ab_cde[1], c_in[d_i * 5 + e]};
    Num f_cd, v_cd, w_c, w_d, rho_cd_abe[2];
    clark_pair_merge_t(
        f_safe[c], v_safe[c], f_safe[d_i], v_safe[d_i], c_in[c * 5 + d_i],
        2, v_rem2, r1_2, r2_2, is_best_of, f_cd, v_cd, w_c, w_d, rho_cd_abe
    );
    Num f_abcd, v_abcd, w_ab, w_cd, rho_abcd_e[1];
    clark_pair_merge_t(
        f_ab, v_ab, f_cd, v_cd, rho_cd_abe[0],
        1, &v_safe[e], &rho_ab_cde[2], &rho_cd_abe[1], is_best_of, f_abcd, v_abcd, w_ab, w_cd, rho_abcd_e
    );
    Num w_abcd;
    clark_pair_merge_t<Num>(
        f_abcd, v_abcd, f_safe[e], v_safe[e], rho_abcd_e[0],
        0, nullptr, nullptr, nullptr, is_best_of, f_root, v_root, w_abcd, we, nullptr
    );
    Num w_ab_t = w_abcd * w_ab, w_cd_t = w_abcd * w_cd;
    wa = w_ab_t * w_a; wb = w_ab_t * w_b; wc = w_cd_t * w_c; wd = w_cd_t * w_d;
}

struct TiedClarkItem {
    int m;
    bool is_best_of;
    int f_out_id;
    int v_out_id;
    int w_out_ids[kMaxD];
    const int64_t* order;
    double g_sep_v;
    double f_sym_v;
    double m2_sym_v;
    double w_sym_v[kMaxD];
    double f_f_v;
    double v_f_v;
    double w_f_v[kMaxD];
};

inline double compute_tie_gsep_d(int d, const double* f_safe) noexcept {
    constexpr double kInvTwoTau2 = 1.0 / (2.0 * 0.008 * 0.008);
    double g_sep = 1.0;
    for (int i = 0; i < d; ++i) {
        for (int j = i + 1; j < d; ++j) {
            double dl = std::log(f_safe[i] / f_safe[j]);
            double p_ij = std::exp(-(dl * dl) * kInvTwoTau2);
            double one_m = 1.0 - p_ij;
            g_sep *= (one_m * one_m);
        }
    }
    return g_sep;
}

inline void clark_symmetrized_outer(
    Tape& tp, int d, int m_idx, const AD* f_in, const AD* v_in, const AD* c_in,
    const int64_t* order, bool is_best_of, AD& f_out, AD& v_out, AD* w_out,
    std::vector<TiedClarkItem>& tied_items
) {
    if (d == 1) {
        f_out = ad_clamp_min(f_in[0], 1e-16);
        v_out = ad_clamp_min(v_in[0], 1e-12);
        w_out[0] = AD(1.0);
        return;
    }
    double f_safe[kMaxD], v_safe[kMaxD], c_val[kMaxD * kMaxD];
    for (int i = 0; i < d; ++i) {
        f_safe[i] = f_in[i].v > 1e-16 ? f_in[i].v : 1e-16;
        v_safe[i] = v_in[i].v > 1e-12 ? v_in[i].v : 1e-12;
    }
    double g_sep = compute_tie_gsep_d(d, f_safe);
    double w_tie = 1.0 - g_sep;
    if (w_tie < 1e-12) {
        clark_one_order_t(d, f_in, v_in, c_in, order, is_best_of, f_out, v_out, w_out);
        return;
    }
    for (int i = 0; i < d * d; ++i) c_val[i] = c_in[i].v;

    double f_sym = 0.0, m2_sym = 0.0, w_sym[kMaxD] = {0.0};
    if (d == 2) {
        double v_m = 0.0, w1 = 0.0, w2 = 0.0;
        clark_pair_merge_t<double>(
            f_safe[0], v_safe[0], f_safe[1], v_safe[1], c_val[1],
            0, nullptr, nullptr, nullptr, is_best_of, f_sym, v_m, w1, w2, nullptr
        );
        m2_sym = f_sym * f_sym * std::exp(v_m * v_m);
        w_sym[0] = w1; w_sym[1] = w2;
    } else if (d == 3) {
        static constexpr int64_t kPerms3[3][3] = {{0, 1, 2}, {0, 2, 1}, {1, 2, 0}};
        double ft = 0.0, vt = 0.0, wt[3];
        for (int t = 0; t < 3; ++t) {
            clark_one_order_t(3, f_safe, v_safe, c_val, kPerms3[t], is_best_of, ft, vt, wt);
            f_sym += ft; m2_sym += ft * ft * std::exp(vt * vt);
            for (int i = 0; i < 3; ++i) w_sym[i] += wt[i];
        }
        constexpr double kInv3 = 1.0 / 3.0;
        f_sym *= kInv3; m2_sym *= kInv3;
        for (int i = 0; i < 3; ++i) w_sym[i] *= kInv3;
    } else if (d == 4) {
        static constexpr int kTrees4[6][4] = {
            {0, 1, 2, 3}, {2, 3, 0, 1}, {0, 2, 1, 3}, {1, 3, 0, 2}, {0, 3, 1, 2}, {1, 2, 0, 3}
        };
        for (int t = 0; t < 6; ++t) {
            int a = kTrees4[t][0], b = kTrees4[t][1], c = kTrees4[t][2], e_d = kTrees4[t][3];
            double fr = 0.0, vr = 0.0, wa = 0.0, wb = 0.0, wc = 0.0, wd = 0.0;
            eval_d4_tree_t(f_safe, v_safe, c_val, a, b, c, e_d, is_best_of, fr, vr, wa, wb, wc, wd);
            f_sym += fr; m2_sym += fr * fr * std::exp(vr * vr);
            w_sym[a] += wa; w_sym[b] += wb; w_sym[c] += wc; w_sym[e_d] += wd;
        }
        constexpr double kInv6 = 1.0 / 6.0;
        f_sym *= kInv6; m2_sym *= kInv6;
        for (int i = 0; i < 4; ++i) w_sym[i] *= kInv6;
    } else if (d == 5) {
        static constexpr int kTrees4[6][4] = {
            {0, 1, 2, 3}, {2, 3, 0, 1}, {0, 2, 1, 3}, {1, 3, 0, 2}, {0, 3, 1, 2}, {1, 2, 0, 3}
        };
        for (int e = 0; e < 5; ++e) {
            int rem4[4], ri = 0;
            for (int x = 0; x < 5; ++x) if (x != e) rem4[ri++] = x;
            for (int t = 0; t < 6; ++t) {
                int a = rem4[kTrees4[t][0]], b = rem4[kTrees4[t][1]], c = rem4[kTrees4[t][2]], d_i = rem4[kTrees4[t][3]];
                double fr = 0.0, vr = 0.0, wa = 0.0, wb = 0.0, wc = 0.0, wd = 0.0, we = 0.0;
                eval_d5_tree_t(f_safe, v_safe, c_val, a, b, c, d_i, e, is_best_of, fr, vr, wa, wb, wc, wd, we);
                f_sym += fr; m2_sym += fr * fr * std::exp(vr * vr);
                w_sym[a] += wa; w_sym[b] += wb; w_sym[c] += wc; w_sym[d_i] += wd; w_sym[e] += we;
            }
        }
        constexpr double kInv30 = 1.0 / 30.0;
        f_sym *= kInv30; m2_sym *= kInv30;
        for (int i = 0; i < 5; ++i) w_sym[i] *= kInv30;
    } else {
        int64_t perm_f[kMaxD], perm_r[kMaxD];
        double ft = 0.0, vt = 0.0, wt[kMaxD];
        for (int s = 0; s < d; ++s) {
            for (int k = 0; k < d; ++k) perm_f[k] = (k + s) % d;
            for (int k = 0; k < d; ++k) perm_r[k] = perm_f[d - 1 - k];
            clark_one_order_t(d, f_safe, v_safe, c_val, perm_f, is_best_of, ft, vt, wt);
            f_sym += ft; m2_sym += ft * ft * std::exp(vt * vt);
            for (int i = 0; i < d; ++i) w_sym[i] += wt[i];
            clark_one_order_t(d, f_safe, v_safe, c_val, perm_r, is_best_of, ft, vt, wt);
            f_sym += ft; m2_sym += ft * ft * std::exp(vt * vt);
            for (int i = 0; i < d; ++i) w_sym[i] += wt[i];
        }
        double inv_2d = 1.0 / (2.0 * static_cast<double>(d));
        f_sym *= inv_2d; m2_sym *= inv_2d;
        for (int i = 0; i < d; ++i) w_sym[i] *= inv_2d;
    }

    double f_out_v = f_sym, v_out_v = std::sqrt(ad_clamp_min(std::log(ad_clamp_min(m2_sym / ad_clamp_min(f_sym * f_sym, 1e-30), 1.0)), 1e-16));
    double w_out_v[kMaxD];
    for (int i = 0; i < d; ++i) w_out_v[i] = w_sym[i];

    TiedClarkItem item;
    item.m = m_idx;
    item.is_best_of = is_best_of;
    item.order = order;
    item.g_sep_v = g_sep;
    item.f_sym_v = f_sym;
    item.m2_sym_v = m2_sym;
    for (int i = 0; i < d; ++i) item.w_sym_v[i] = w_sym[i];
    item.f_f_v = 0.0; item.v_f_v = 0.0;
    for (int i = 0; i < d; ++i) item.w_f_v[i] = 0.0;

    if (g_sep >= 1e-15) {
        clark_one_order_t(d, f_safe, v_safe, c_val, order, is_best_of, item.f_f_v, item.v_f_v, item.w_f_v);
        f_out_v = g_sep * item.f_f_v + w_tie * f_sym;
        double m2_out = g_sep * (item.f_f_v * item.f_f_v * std::exp(item.v_f_v * item.v_f_v)) + w_tie * m2_sym;
        v_out_v = std::sqrt(ad_clamp_min(std::log(ad_clamp_min(m2_out / ad_clamp_min(f_out_v * f_out_v, 1e-30), 1.0)), 1e-16));
        for (int i = 0; i < d; ++i) w_out_v[i] = g_sep * item.w_f_v[i] + w_tie * w_sym[i];
    }

    f_out = make_leaf(tp, f_out_v);
    v_out = make_leaf(tp, v_out_v);
    item.f_out_id = f_out.id;
    item.v_out_id = v_out.id;
    for (int i = 0; i < d; ++i) {
        w_out[i] = make_leaf(tp, w_out_v[i]);
        item.w_out_ids[i] = w_out[i].id;
    }
    tied_items.push_back(item);
}

inline void accum_sub_tape_grads(
    Tape& tp, const Tape& sub_tp, int d, int M, int m_idx,
    bool need_fp_grad, bool need_vp_grad, bool need_corr_grad,
    const double* fp_r, const double* vp_r,
    const AD* f_leaves, const AD* v_leaves, const AD* c_leaves
) {
    if (need_fp_grad) {
        for (int i = 0; i < d; ++i) {
            if (fp_r[i * M + m_idx] >= 1e-16 && tp.fp[i * M + m_idx].id >= 0 && f_leaves[i].id >= 0) {
                tp.adj[tp.fp[i * M + m_idx].id] += sub_tp.adj[f_leaves[i].id];
            }
        }
    }
    if (need_vp_grad) {
        for (int i = 0; i < d; ++i) {
            if (vp_r[i * M + m_idx] >= 1e-12 && tp.vp[i * M + m_idx].id >= 0 && v_leaves[i].id >= 0) {
                tp.adj[tp.vp[i * M + m_idx].id] += sub_tp.adj[v_leaves[i].id];
            }
        }
    }
    if (need_corr_grad) {
        for (int i = 0; i < d * d; ++i) {
            if (tp.corr[i].id >= 0 && c_leaves[i].id >= 0) {
                tp.adj[tp.corr[i].id] += sub_tp.adj[c_leaves[i].id];
            }
        }
    }
}

inline void clark_symmetrized_bwd_item(
    Tape& tp, Tape& sub_tp, int d, int M,
    bool need_fp_grad, bool need_vp_grad, bool need_corr_grad,
    const double* fp_r, const double* vp_r, const double* corr_r,
    const TiedClarkItem& item
) {
    double gf_out = tp.adj[item.f_out_id];
    double gv_out = tp.adj[item.v_out_id];
    double gw_out[kMaxD];
    bool any_gw = false;
    for (int i = 0; i < d; ++i) {
        gw_out[i] = tp.adj[item.w_out_ids[i]];
        if (gw_out[i] != 0.0) any_gw = true;
    }
    if (gf_out == 0.0 && gv_out == 0.0 && !any_gw) return;

    int m_idx = item.m;
    double f_safe[kMaxD], v_safe[kMaxD];
    for (int i = 0; i < d; ++i) {
        f_safe[i] = fp_r[i * M + m_idx] > 1e-16 ? fp_r[i * M + m_idx] : 1e-16;
        v_safe[i] = vp_r[i * M + m_idx] > 1e-12 ? vp_r[i * M + m_idx] : 1e-12;
    }

    // Step A: Scalar blend adjoints on tiny sub_tp
    double g_gsep = 0.0, g_fsym = 0.0, g_m2sym = 0.0, g_wsym[kMaxD] = {0.0};
    double g_ff = 0.0, g_vf = 0.0, g_wf[kMaxD] = {0.0};
    {
        sub_tp.reset();
        AD gs_l = make_leaf(sub_tp, item.g_sep_v);
        AD wt_l = 1.0 - gs_l;
        AD fs_l = make_leaf(sub_tp, item.f_sym_v);
        AD m2s_l = make_leaf(sub_tp, item.m2_sym_v);
        AD ws_l[kMaxD], ff_l, vf_l, wf_l[kMaxD];
        for (int i = 0; i < d; ++i) ws_l[i] = make_leaf(sub_tp, item.w_sym_v[i]);
        AD fo_l, vo_l, wo_l[kMaxD];
        if (item.g_sep_v < 1e-15) {
            fo_l = fs_l;
            vo_l = ad_sqrt(ad_clamp_min(ad_log(ad_clamp_min(m2s_l / ad_clamp_min(fs_l * fs_l, 1e-30), 1.0)), 1e-16));
            for (int i = 0; i < d; ++i) wo_l[i] = ws_l[i];
        } else {
            ff_l = make_leaf(sub_tp, item.f_f_v);
            vf_l = make_leaf(sub_tp, item.v_f_v);
            for (int i = 0; i < d; ++i) wf_l[i] = make_leaf(sub_tp, item.w_f_v[i]);
            fo_l = gs_l * ff_l + wt_l * fs_l;
            AD m2o_l = gs_l * (ff_l * ff_l * ad_exp(vf_l * vf_l)) + wt_l * m2s_l;
            vo_l = ad_sqrt(ad_clamp_min(ad_log(ad_clamp_min(m2o_l / ad_clamp_min(fo_l * fo_l, 1e-30), 1.0)), 1e-16));
            for (int i = 0; i < d; ++i) wo_l[i] = gs_l * wf_l[i] + wt_l * ws_l[i];
        }
        sub_tp.zero_grad();
        if (fo_l.id >= 0) sub_tp.adj[fo_l.id] += gf_out;
        if (vo_l.id >= 0) sub_tp.adj[vo_l.id] += gv_out;
        for (int i = 0; i < d; ++i) if (wo_l[i].id >= 0) sub_tp.adj[wo_l[i].id] += gw_out[i];
        sub_tp.backward_sweep();
        g_gsep = sub_tp.adj[gs_l.id];
        g_fsym = sub_tp.adj[fs_l.id];
        g_m2sym = sub_tp.adj[m2s_l.id];
        for (int i = 0; i < d; ++i) g_wsym[i] = sub_tp.adj[ws_l[i].id];
        if (item.g_sep_v >= 1e-15) {
            g_ff = sub_tp.adj[ff_l.id];
            g_vf = sub_tp.adj[vf_l.id];
            for (int i = 0; i < d; ++i) g_wf[i] = sub_tp.adj[wf_l[i].id];
        }
    }

    // Step B: g_sep wrt fp
    if (need_fp_grad && g_gsep != 0.0) {
        constexpr double kInvTwoTau2 = 1.0 / (2.0 * 0.008 * 0.008);
        sub_tp.reset();
        AD f_l[kMaxD];
        for (int i = 0; i < d; ++i) f_l[i] = make_leaf(sub_tp, f_safe[i]);
        AD gs(1.0);
        for (int i = 0; i < d; ++i) {
            for (int j = i + 1; j < d; ++j) {
                AD dl = ad_log(f_l[i] / f_l[j]);
                AD one_m = 1.0 - ad_exp(-(dl * dl) * kInvTwoTau2);
                gs = gs * (one_m * one_m);
            }
        }
        sub_tp.zero_grad();
        if (gs.id >= 0) {
            sub_tp.adj[gs.id] += g_gsep;
            sub_tp.backward_sweep();
            for (int i = 0; i < d; ++i) {
                if (fp_r[i * M + m_idx] >= 1e-16 && tp.fp[i * M + m_idx].id >= 0) {
                    tp.adj[tp.fp[i * M + m_idx].id] += sub_tp.adj[f_l[i].id];
                }
            }
        }
    }

    auto init_leaves = [&](AD* f_l, AD* v_l, AD* c_l) {
        sub_tp.reset();
        for (int i = 0; i < d; ++i) {
            f_l[i] = need_fp_grad ? make_leaf(sub_tp, f_safe[i]) : AD(f_safe[i]);
            v_l[i] = need_vp_grad ? make_leaf(sub_tp, v_safe[i]) : AD(v_safe[i]);
        }
        for (int i = 0; i < d * d; ++i) {
            c_l[i] = need_corr_grad ? make_leaf(sub_tp, corr_r[i]) : AD(corr_r[i]);
        }
    };

    // Step C: separated order branch if g_sep >= 1e-15
    if (item.g_sep_v >= 1e-15) {
        bool any_wf = (g_ff != 0.0 || g_vf != 0.0);
        for (int i = 0; i < d; ++i) if (g_wf[i] != 0.0) any_wf = true;
        if (any_wf) {
            AD f_l[kMaxD], v_l[kMaxD], c_l[kMaxD * kMaxD], ft, vt, wt[kMaxD];
            init_leaves(f_l, v_l, c_l);
            clark_one_order_t(d, f_l, v_l, c_l, item.order, item.is_best_of, ft, vt, wt);
            sub_tp.zero_grad();
            if (ft.id >= 0) sub_tp.adj[ft.id] += g_ff;
            if (vt.id >= 0) sub_tp.adj[vt.id] += g_vf;
            for (int i = 0; i < d; ++i) if (wt[i].id >= 0) sub_tp.adj[wt[i].id] += g_wf[i];
            sub_tp.backward_sweep();
            accum_sub_tape_grads(tp, sub_tp, d, M, m_idx, need_fp_grad, need_vp_grad, need_corr_grad, fp_r, vp_r, f_l, v_l, c_l);
        }
    }

    // Step D: L1-resident per-tree backward sweep
    bool any_sym_w = (g_fsym != 0.0 || g_m2sym != 0.0);
    for (int i = 0; i < d; ++i) if (g_wsym[i] != 0.0) any_sym_w = true;
    if (!any_sym_w) return;

    auto finish_tree = [&](double inv_n, AD& ft, AD& vt, AD* wt, AD* f_l, AD* v_l, AD* c_l) {
        AD m2t = ft * ft * ad_exp(vt * vt);
        sub_tp.zero_grad();
        if (ft.id >= 0) sub_tp.adj[ft.id] += g_fsym * inv_n;
        if (m2t.id >= 0) sub_tp.adj[m2t.id] += g_m2sym * inv_n;
        for (int i = 0; i < d; ++i) if (wt[i].id >= 0) sub_tp.adj[wt[i].id] += g_wsym[i] * inv_n;
        sub_tp.backward_sweep();
        accum_sub_tape_grads(tp, sub_tp, d, M, m_idx, need_fp_grad, need_vp_grad, need_corr_grad, fp_r, vp_r, f_l, v_l, c_l);
    };

    AD f_l[kMaxD], v_l[kMaxD], c_l[kMaxD * kMaxD];
    if (d == 2) {
        init_leaves(f_l, v_l, c_l);
        AD ft, vt, wt[2];
        clark_pair_merge_t<AD>(
            f_l[0], v_l[0], f_l[1], v_l[1], c_l[1],
            0, nullptr, nullptr, nullptr, item.is_best_of, ft, vt, wt[0], wt[1], nullptr
        );
        finish_tree(1.0, ft, vt, wt, f_l, v_l, c_l);
    } else if (d == 3) {
        static constexpr int64_t kPerms3[3][3] = {{0, 1, 2}, {0, 2, 1}, {1, 2, 0}};
        constexpr double kInv3 = 1.0 / 3.0;
        for (int t = 0; t < 3; ++t) {
            init_leaves(f_l, v_l, c_l);
            AD ft, vt, wt[3];
            clark_one_order_t(3, f_l, v_l, c_l, kPerms3[t], item.is_best_of, ft, vt, wt);
            finish_tree(kInv3, ft, vt, wt, f_l, v_l, c_l);
        }
    } else if (d == 4) {
        static constexpr int kTrees4[6][4] = {
            {0, 1, 2, 3}, {2, 3, 0, 1}, {0, 2, 1, 3}, {1, 3, 0, 2}, {0, 3, 1, 2}, {1, 2, 0, 3}
        };
        constexpr double kInv6 = 1.0 / 6.0;
        for (int t = 0; t < 6; ++t) {
            int a = kTrees4[t][0], b = kTrees4[t][1], c = kTrees4[t][2], e_d = kTrees4[t][3];
            init_leaves(f_l, v_l, c_l);
            AD ft, vt, wt[4];
            eval_d4_tree_t(f_l, v_l, c_l, a, b, c, e_d, item.is_best_of, ft, vt, wt[a], wt[b], wt[c], wt[e_d]);
            finish_tree(kInv6, ft, vt, wt, f_l, v_l, c_l);
        }
    } else if (d == 5) {
        static constexpr int kTrees4[6][4] = {
            {0, 1, 2, 3}, {2, 3, 0, 1}, {0, 2, 1, 3}, {1, 3, 0, 2}, {0, 3, 1, 2}, {1, 2, 0, 3}
        };
        constexpr double kInv30 = 1.0 / 30.0;
        for (int e = 0; e < 5; ++e) {
            int rem4[4], ri = 0;
            for (int x = 0; x < 5; ++x) if (x != e) rem4[ri++] = x;
            for (int t = 0; t < 6; ++t) {
                int a = rem4[kTrees4[t][0]], b = rem4[kTrees4[t][1]], c = rem4[kTrees4[t][2]], d_i = rem4[kTrees4[t][3]];
                init_leaves(f_l, v_l, c_l);
                AD ft, vt, wt[5];
                eval_d5_tree_t(f_l, v_l, c_l, a, b, c, d_i, e, item.is_best_of, ft, vt, wt[a], wt[b], wt[c], wt[d_i], wt[e]);
                finish_tree(kInv30, ft, vt, wt, f_l, v_l, c_l);
            }
        }
    } else {
        int64_t perm_f[kMaxD], perm_r[kMaxD];
        double inv_2d = 1.0 / (2.0 * static_cast<double>(d));
        for (int s = 0; s < d; ++s) {
            for (int k = 0; k < d; ++k) perm_f[k] = (k + s) % d;
            for (int k = 0; k < d; ++k) perm_r[k] = perm_f[d - 1 - k];
            AD ft, vt, wt[kMaxD];
            init_leaves(f_l, v_l, c_l);
            clark_one_order_t(d, f_l, v_l, c_l, perm_f, item.is_best_of, ft, vt, wt);
            finish_tree(inv_2d, ft, vt, wt, f_l, v_l, c_l);
            init_leaves(f_l, v_l, c_l);
            clark_one_order_t(d, f_l, v_l, c_l, perm_r, item.is_best_of, ft, vt, wt);
            finish_tree(inv_2d, ft, vt, wt, f_l, v_l, c_l);
        }
    }
}

inline AD exact_piecewise_terminal_put(
    const AD& f, const AD& v, double strike, double k_low_in,
    int has_active_ki, double ki_barrier, double ki_spread,
    int has_expiry_ko, double ko_barrier, double ko_spread
) {
    if (strike <= 1e-8) return AD(0.0);
    double k_low = std::max(k_low_in, 0.0);
    if (k_low >= strike) return AD(0.0);
    double pts[10]; int n_pts = 0;
    pts[n_pts++] = 0.0; pts[n_pts++] = strike;
    if (k_low > 0.0) pts[n_pts++] = k_low;
    if (has_active_ki) {
        if (std::abs(ki_spread) <= 1e-6) pts[n_pts++] = ki_barrier;
        else { pts[n_pts++] = std::min(ki_barrier, ki_barrier - ki_spread); pts[n_pts++] = std::max(ki_barrier, ki_barrier - ki_spread); }
    }
    if (has_expiry_ko) {
        if (std::abs(ko_spread) <= 1e-6) pts[n_pts++] = ko_barrier;
        else { pts[n_pts++] = std::min(ko_barrier, ko_barrier + ko_spread); pts[n_pts++] = std::max(ko_barrier, ko_barrier + ko_spread); }
    }
    int n_f = 0;
    for (int i = 0; i < n_pts; ++i) if (pts[i] >= 0.0 && pts[i] <= strike) pts[n_f++] = pts[i];
    std::sort(pts, pts + n_f);
    AD fc = ad_clamp_min(f, 1e-16), vc = ad_clamp_min(v, 1e-12), m_u = ad_log(fc) - 0.5 * vc * vc, total(0.0);
    for (int idx = 0; idx < n_f - 1; ++idx) {
        double a = pts[idx], b = pts[idx + 1];
        if (b - a <= 1e-14 || b <= 0.0) continue;
        double mid = 0.5 * (a + b);
        double c0 = (mid <= k_low) ? (strike - k_low) : strike, c1 = (mid <= k_low) ? 0.0 : -1.0;
        double d0 = 1.0, d1 = 0.0;
        if (has_active_ki) {
            if (std::abs(ki_spread) <= 1e-6) { if (mid >= ki_barrier) continue; }
            else {
                double blo = std::min(ki_barrier, ki_barrier - ki_spread), bhi = std::max(ki_barrier, ki_barrier - ki_spread);
                if (mid >= bhi) continue;
                if (mid > blo) { double db = bhi - blo; d0 = bhi / db; d1 = -1.0 / db; }
            }
        }
        double e0 = 1.0, e1 = 0.0;
        if (has_expiry_ko) {
            if (std::abs(ko_spread) <= 1e-6) { if (mid >= ko_barrier) continue; }
            else {
                double blo = std::min(ko_barrier, ko_barrier + ko_spread), bhi = std::max(ko_barrier, ko_barrier + ko_spread);
                if (mid >= bhi) continue;
                if (mid > blo) { double db = bhi - blo; e0 = bhi / db; e1 = -1.0 / db; }
            }
        }
        double cd0 = c0 * d0, cd1 = c0 * d1 + c1 * d0, cd2 = c1 * d1;
        double q0 = cd0 * e0, q1 = cd0 * e1 + cd1 * e0, q2 = cd1 * e1 + cd2 * e0, q3 = cd2 * e1;
        int max_k = (q3 != 0.0) ? 3 : ((q2 != 0.0) ? 2 : 1);
        AD u_b = (std::log(b) - m_u) / vc;
        bool has_a = a > 0.0;
        AD u_a = has_a ? (std::log(a) - m_u) / vc : AD(0.0);
        AD fp_k(1.0), moms[4];
        for (int k = 0; k <= max_k; ++k) {
            AD scale = fp_k * ad_exp(0.5 * static_cast<double>(k * (k - 1)) * (vc * vc));
            AD pb = ad_ndtr(u_b - static_cast<double>(k) * vc);
            AD pa = has_a ? ad_ndtr(u_a - static_cast<double>(k) * vc) : AD(0.0);
            moms[k] = scale * (pb - pa);
            fp_k = fp_k * fc;
        }
        total = total + q0 * moms[0] + q1 * moms[1];
        if (max_k >= 2) total = total + q2 * moms[2];
        if (max_k >= 3) total = total + q3 * moms[3];
    }
    return ad_clamp_min(total, 0.0);
}

void run_single_row(
    Tape& tp, int D, int M, bool need_fp_grad, bool need_vp_grad, bool need_corr_grad,
    const double* fp_r, const double* vp_r, const double* corr_r,
    const double* ko_t, const int* is_call_flags, const double* cb_t, const double* ig_t,
    const int64_t* ord_call_r, const int64_t* ord_cpn_r, const int64_t* ord_opt_r,
    int call_code, int cpn_code, int opt_code, int cpn_dir_down, int cpn_type_out,
    int force_cpn, int allow_exp, double ko_spread, double cpn_spread, double ki_spread,
    double strike, double option_gearing, double lower_strike, int has_active_ki, double ki_barrier,
    double* fw_o, double* vw_o, double* p_o, double* q_o, double* lput_o, double* dw_o,
    const double* g_fw, const double* g_vw, const double* g_p, const double* g_q, const double* g_lput, const double* g_dw,
    double* g_fp, double* g_vp, double* g_corr
) {
    tp.reset();
    tp.fp.resize(D * M); tp.vp.resize(D * M); tp.corr.resize(D * D);
    for (int i = 0; i < D * M; ++i) {
        tp.fp[i] = need_fp_grad ? make_leaf(tp, fp_r[i]) : AD(fp_r[i]);
        tp.vp[i] = need_vp_grad ? make_leaf(tp, vp_r[i]) : AD(vp_r[i]);
    }
    for (int i = 0; i < D * D; ++i) {
        tp.corr[i] = need_corr_grad ? make_leaf(tp, corr_r[i]) : AD(corr_r[i]);
    }
    bool call_best = (call_code == 1), cpn_best = (cpn_code == 1), opt_best = (opt_code == 1);
    bool cpn_diff = (cpn_code != call_code), opt_diff = (opt_code != call_code);
    tp.f_w.resize(M); tp.v_w.resize(M); tp.w_call.resize(M * D);
    tp.f_cpn.resize(M); tp.v_cpn.resize(M); tp.rho_cc.resize(M);
    tp.p_proxy.resize(M); tp.q_proxy.resize(M);
    AD fm_f[kMaxD], fm_v[kMaxD], wm_c[kMaxD], wm_p[kMaxD];
    std::vector<TiedClarkItem> tied_items;
    tied_items.reserve(static_cast<size_t>(M * 2 + 2));

    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < D; ++i) { fm_f[i] = tp.fp[i * M + m]; fm_v[i] = tp.vp[i * M + m]; }
        clark_symmetrized_outer(tp, D, m, fm_f, fm_v, tp.corr.data(), ord_call_r + m * D, call_best, tp.f_w[m], tp.v_w[m], wm_c, tied_items);
        for (int i = 0; i < D; ++i) tp.w_call[m * D + i] = wm_c[i];
        if (cpn_diff) {
            clark_symmetrized_outer(tp, D, m, fm_f, fm_v, tp.corr.data(), ord_cpn_r + m * D, cpn_best, tp.f_cpn[m], tp.v_cpn[m], wm_p, tied_items);
            AD cov(0.0), va(0.0), vb(0.0);
            for (int i = 0; i < D; ++i) {
                AD ai = wm_c[i] * fm_v[i], bi = wm_p[i] * fm_v[i];
                for (int j = 0; j < D; ++j) {
                    AD aj = wm_c[j] * fm_v[j], bj = wm_p[j] * fm_v[j];
                    const AD& cij = tp.corr[i * D + j];
                    cov = cov + ai * cij * bj; va = va + ai * cij * aj; vb = vb + bi * cij * bj;
                }
            }
            tp.rho_cc[m] = ad_clamp(cov / ad_sqrt(ad_clamp_min(va, 1e-30) * ad_clamp_min(vb, 1e-30)), -0.9999, 0.9999);
        }
    }

    AD z_mean(0.0), z_std(1.0), surv_cur(1.0);
    AD f_cond_term(0.0), v_cond_term(0.0), z_mean_term(0.0), z_std_term(1.0);
    AD surv_prev_term(1.0), surv_term(1.0);
    for (int m = 0; m < M; ++m) {
        AD fwm = ad_clamp_min(tp.f_w[m], 1e-16), vwm = ad_clamp_min(tp.v_w[m], 1e-12);
        AD fc = (m == 0) ? fwm : (fwm * ad_exp(vwm * z_mean - 0.5 * (vwm * vwm) * (1.0 - z_std * z_std)));
        AD vc = (m == 0) ? vwm : ad_clamp_min(vwm * z_std, 1e-12);
        if (m == M - 1) { f_cond_term = fc; v_cond_term = vc; z_mean_term = z_mean; z_std_term = z_std; }
        bool is_c = (is_call_flags[m] != 0);
        AD hm = is_c ? ad_clamp(digital_call_spread(fc, vc, ko_t[m], ko_spread), 0.0, 1.0 - 1e-7) : AD(0.0);
        tp.p_proxy[m] = surv_cur * hm;
        AD surv_next = surv_cur * (1.0 - hm);
        if (m == M - 1) { surv_prev_term = surv_cur; surv_term = surv_next; }
        AD f_cc = fc, v_cc = vc;
        if (cpn_diff) {
            AD fcr = ad_clamp_min(tp.f_cpn[m], 1e-16), vcr = ad_clamp_min(tp.v_cpn[m], 1e-12);
            const AD& rc = tp.rho_cc[m];
            AD z_mc = rc * z_mean, z_vc = rc * rc * (z_std * z_std) + (1.0 - rc * rc);
            f_cc = fcr * ad_exp(vcr * z_mc - 0.5 * (vcr * vcr) * (1.0 - z_vc));
            v_cc = ad_clamp_min(vcr * ad_sqrt(ad_clamp_min(z_vc, 1e-12)), 1e-12);
        }
        double eff_cb = (!cpn_diff && force_cpn && is_c && !cpn_dir_down && !cpn_type_out) ? std::min(cb_t[m], ko_t[m]) : cb_t[m];
        AD q_raw = eval_coupon_prob(f_cc, v_cc, eff_cb, cpn_spread, cpn_dir_down, cpn_type_out);
        if (force_cpn && is_c) q_raw = ad_max(q_raw, hm);
        tp.q_proxy[m] = (ig_t[m] > 0.5) ? AD(1.0) : q_raw;
        if (m < M - 1) {
            AD z_pm = z_mean, z_pv = z_std * z_std;
            if (is_c) {
                AD mu_u, var_u;
                ramp_survival_post_moments(fc, vc, ko_t[m], ko_spread, mu_u, var_u);
                z_pm = z_mean + z_std * mu_u;
                z_pv = ad_clamp_min(z_std * z_std * var_u, 1e-12);
            }
            if (is_c || m > 0) {
                AD num(0.0), dc(0.0), dn(0.0);
                for (int i = 0; i < D; ++i) {
                    AD vc_i = ad_clamp_min(tp.vp[i * M + m], 1e-12);
                    AD vn_i = ad_clamp_min(tp.vp[i * M + m + 1], 1e-12);
                    const AD& wi = tp.w_call[m * D + i], &win = tp.w_call[(m + 1) * D + i];
                    for (int j = 0; j < D; ++j) {
                        AD vc_j = ad_clamp_min(tp.vp[j * M + m], 1e-12);
                        AD vn_j = ad_clamp_min(tp.vp[j * M + m + 1], 1e-12);
                        const AD& wj_c = tp.w_call[m * D + j], &wj_n = tp.w_call[(m + 1) * D + j];
                        const AD& cij = tp.corr[i * D + j];
                        num = num + wi * wj_n * cij * vc_i * ad_min(vc_j, vn_j);
                        dc = dc + wi * wj_c * cij * vc_i * vc_j;
                        dn = dn + win * wj_n * cij * vn_i * vn_j;
                    }
                }
                AD denom = ad_clamp_min(ad_sqrt(ad_clamp_min(dc, 1e-30)) * ad_sqrt(ad_clamp_min(dn, 1e-30)), 1e-30);
                AD r_next = ad_clamp(num / denom, -0.9999, 0.9999);
                z_mean = r_next * z_pm;
                z_std = ad_sqrt(ad_clamp_min(r_next * r_next * z_pv + (1.0 - r_next * r_next), 1e-12));
            }
        }
        surv_cur = surv_next;
    }

    AD f_oc = f_cond_term, v_oc = v_cond_term;
    if (opt_diff) {
        AD fo_f[kMaxD], fo_v[kMaxD], wo[kMaxD], fo_r, vo_r;
        for (int i = 0; i < D; ++i) { fo_f[i] = tp.fp[i * M + M - 1]; fo_v[i] = tp.vp[i * M + M - 1]; }
        clark_symmetrized_outer(tp, D, M - 1, fo_f, fo_v, tp.corr.data(), ord_opt_r, opt_best, fo_r, vo_r, wo, tied_items);
        AD cov(0.0), va(0.0), vb(0.0);
        for (int i = 0; i < D; ++i) {
            AD ai = tp.w_call[(M - 1) * D + i] * fo_v[i], bi = wo[i] * fo_v[i];
            for (int j = 0; j < D; ++j) {
                AD aj = tp.w_call[(M - 1) * D + j] * fo_v[j], bj = wo[j] * fo_v[j];
                const AD& cij = tp.corr[i * D + j];
                cov = cov + ai * cij * bj; va = va + ai * cij * aj; vb = vb + bi * cij * bj;
            }
        }
        AD rco = ad_clamp(cov / ad_sqrt(ad_clamp_min(va, 1e-30) * ad_clamp_min(vb, 1e-30)), -0.9999, 0.9999);
        AD z_mo = rco * z_mean_term, z_vo = rco * rco * (z_std_term * z_std_term) + (1.0 - rco * rco);
        AD vor = ad_clamp_min(vo_r, 1e-12);
        f_oc = ad_clamp_min(fo_r, 1e-16) * ad_exp(vor * z_mo - 0.5 * (vor * vor) * (1.0 - z_vo));
        v_oc = ad_clamp_min(vor * ad_sqrt(ad_clamp_min(z_vo, 1e-12)), 1e-12);
    }

    AD l_put(0.0);
    if (strike > 1e-6 && option_gearing > 0.0) {
        int use_dir_ko = (allow_exp && is_call_flags[M - 1] && !opt_diff) ? 1 : 0;
        const AD& smult = use_dir_ko ? surv_prev_term : surv_term;
        l_put = smult * exact_piecewise_terminal_put(
            f_oc, v_oc, strike, lower_strike, has_active_ki, ki_barrier, ki_spread,
            use_dir_ko, ko_t[M - 1], ko_spread
        );
    }

    if (fw_o) for (int m = 0; m < M; ++m) fw_o[m] = tp.f_w[m].v;
    if (vw_o) for (int m = 0; m < M; ++m) vw_o[m] = tp.v_w[m].v;
    if (p_o) for (int m = 0; m < M; ++m) p_o[m] = tp.p_proxy[m].v;
    if (q_o) for (int m = 0; m < M; ++m) q_o[m] = tp.q_proxy[m].v;
    if (lput_o) *lput_o = l_put.v;
    if (dw_o) for (int i = 0; i < D; ++i) dw_o[i] = tp.w_call[(M - 1) * D + i].v;

    if (need_fp_grad || need_vp_grad || need_corr_grad) {
        tp.zero_grad();
        if (g_fw) for (int m = 0; m < M; ++m) if (tp.f_w[m].id >= 0) tp.adj[tp.f_w[m].id] += g_fw[m];
        if (g_vw) for (int m = 0; m < M; ++m) if (tp.v_w[m].id >= 0) tp.adj[tp.v_w[m].id] += g_vw[m];
        if (g_p) for (int m = 0; m < M; ++m) if (tp.p_proxy[m].id >= 0) tp.adj[tp.p_proxy[m].id] += g_p[m];
        if (g_q) for (int m = 0; m < M; ++m) if (tp.q_proxy[m].id >= 0) tp.adj[tp.q_proxy[m].id] += g_q[m];
        if (g_lput && l_put.id >= 0) tp.adj[l_put.id] += *g_lput;
        if (g_dw) for (int i = 0; i < D; ++i) if (tp.w_call[(M - 1) * D + i].id >= 0) tp.adj[tp.w_call[(M - 1) * D + i].id] += g_dw[i];
        tp.backward_sweep();
        if (!tied_items.empty()) {
            thread_local Tape sub_tp;
            for (const auto& ti : tied_items) {
                clark_symmetrized_bwd_item(tp, sub_tp, D, M, need_fp_grad, need_vp_grad, need_corr_grad, fp_r, vp_r, corr_r, ti);
            }
        }
        if (need_fp_grad) for (int i = 0; i < D * M; ++i) g_fp[i] = (tp.fp[i].id >= 0) ? tp.adj[tp.fp[i].id] : 0.0;
        if (need_vp_grad) for (int i = 0; i < D * M; ++i) g_vp[i] = (tp.vp[i].id >= 0) ? tp.adj[tp.vp[i].id] : 0.0;
        if (need_corr_grad) for (int i = 0; i < D * D; ++i) g_corr[i] = (tp.corr[i].id >= 0) ? tp.adj[tp.corr[i].id] : 0.0;
    }
}

}  // namespace

extern "C" {
void tier1_pure_fwd_bwd(
    int B, int D, int M, int need_fp_grad, int need_vp_grad, int need_corr_grad,
    const double* fp_all, const double* vp_all, const double* corr_all,
    const double* ko_t, const int* is_call_flags, const double* cb_t, const double* ig_t,
    const int64_t* ord_call, const int64_t* ord_cpn, const int64_t* ord_opt,
    int call_code, int cpn_code, int opt_code, int cpn_dir_down, int cpn_type_out,
    int force_cpn, int allow_exp, double ko_spread, double cpn_spread, double ki_spread,
    double strike, double option_gearing, double lower_strike, int has_active_ki, double ki_barrier,
    double* fw_out, double* vw_out, double* p_out, double* q_out, double* lput_out, double* dw_out,
    const double* g_fw, const double* g_vw, const double* g_p, const double* g_q, const double* g_lput, const double* g_dw,
    double* g_fp, double* g_vp, double* g_corr
) {
    struct ActiveGuard {
        std::atomic<int>& cnt;
        ActiveGuard(std::atomic<int>& c) noexcept : cnt(c) { cnt.fetch_add(1, std::memory_order_relaxed); }
        ~ActiveGuard() noexcept { cnt.fetch_sub(1, std::memory_order_relaxed); }
    };
    static std::atomic<int> g_active_calls{0};
    ActiveGuard ag(g_active_calls);
    const bool allow_par = (B >= 4) && (M * D >= 25) && (g_active_calls.load(std::memory_order_relaxed) <= 2);
#pragma omp parallel for if(allow_par) schedule(dynamic, 1)
    for (int b = 0; b < B; ++b) {
        thread_local Tape tp;
        run_single_row(
            tp, D, M, need_fp_grad != 0, need_vp_grad != 0, (need_corr_grad != 0 && b == 0) || need_corr_grad == 2,
            fp_all + b * D * M, vp_all + b * D * M, corr_all + b * D * D,
            ko_t, is_call_flags, cb_t, ig_t,
            ord_call + b * M * D, ord_cpn + b * M * D, ord_opt + b * D,
            call_code, cpn_code, opt_code, cpn_dir_down, cpn_type_out,
            force_cpn, allow_exp, ko_spread, cpn_spread, ki_spread,
            strike, option_gearing, lower_strike, has_active_ki, ki_barrier,
            fw_out ? fw_out + b * M : nullptr,
            vw_out ? vw_out + b * M : nullptr,
            p_out ? p_out + b * M : nullptr,
            q_out ? q_out + b * M : nullptr,
            lput_out ? lput_out + b : nullptr,
            dw_out ? dw_out + b * D : nullptr,
            g_fw ? g_fw + b * M : nullptr,
            g_vw ? g_vw + b * M : nullptr,
            g_p ? g_p + b * M : nullptr,
            g_q ? g_q + b * M : nullptr,
            g_lput ? g_lput + b : nullptr,
            g_dw ? g_dw + b * D : nullptr,
            g_fp ? g_fp + b * D * M : nullptr,
            g_vp ? g_vp + b * D * M : nullptr,
            g_corr ? g_corr + b * D * D : nullptr
        );
    }
}

void surface_eval_fwd_dk(
    int Q, int n_u, int n_exp,
    double k_min, double k_max, double min_vol,
    const double* strikes, const double* tenors, const double* quotes,
    const double* slice_c, const double* slope_l, const double* slope_r,
    const double* t_q, const double* k_q,
    double* iv_out, double* div_dk_out
) {
    int n_seg = n_u - 1;
    int stride_c = n_seg * n_exp;
    double mv2 = min_vol * min_vol;
    auto eval_slice = [&](int e, int seg, double k, double& val, double& dval) {
        if (k < k_min) {
            double sl = slope_l[e];
            val = quotes[e * n_u] + sl * (k - k_min);
            dval = sl;
        } else if (k > k_max) {
            double sr = slope_r[e];
            val = quotes[e * n_u + n_u - 1] + sr * (k - k_max);
            dval = sr;
        } else {
            double dx = k - strikes[seg];
            double c0 = slice_c[seg * n_exp + e];
            double c1 = slice_c[stride_c + seg * n_exp + e];
            double c2 = slice_c[2 * stride_c + seg * n_exp + e];
            double c3 = slice_c[3 * stride_c + seg * n_exp + e];
            val = ((c0 * dx + c1) * dx + c2) * dx + c3;
            dval = (3.0 * c0 * dx + 2.0 * c1) * dx + c2;
        }
        if (val < min_vol) { val = min_vol; dval = 0.0; }
    };
    for (int q = 0; q < Q; ++q) {
        double t = t_q[q] > 1e-8 ? t_q[q] : 1e-8;
        double k = k_q[q];
        auto it_s = std::upper_bound(strikes, strikes + n_u, k);
        int seg = static_cast<int>(it_s - strikes) - 1;
        if (seg < 0) seg = 0;
        if (seg > n_seg - 1) seg = n_seg - 1;
        if (n_exp == 1) {
            double v0, dv0;
            eval_slice(0, seg, k, v0, dv0);
            iv_out[q] = v0;
            div_dk_out[q] = dv0;
            continue;
        }
        auto it_t = std::upper_bound(tenors, tenors + n_exp, t);
        int idx0 = static_cast<int>(it_t - tenors) - 1;
        if (idx0 < 0) idx0 = 0;
        if (idx0 > n_exp - 2) idx0 = n_exp - 2;
        int idx1 = idx0 + 1;
        double v0, dv0, v1, dv1;
        eval_slice(idx0, seg, k, v0, dv0);
        eval_slice(idx1, seg, k, v1, dv1);
        if (t <= tenors[0]) {
            iv_out[q] = v0; div_dk_out[q] = dv0;
        } else if (t >= tenors[n_exp - 1]) {
            iv_out[q] = v1; div_dk_out[q] = dv1;
        } else {
            double t0 = tenors[idx0], t1 = tenors[idx1];
            double dt_span = (t1 - t0) > 1e-12 ? (t1 - t0) : 1e-12;
            double alpha = (t - t0) / dt_span;
            double w = (1.0 - alpha) * (v0 * v0 * t0) + alpha * (v1 * v1 * t1);
            double wt = w / t;
            if (wt <= mv2) {
                iv_out[q] = min_vol; div_dk_out[q] = 0.0;
            } else {
                double iv = std::sqrt(wt);
                iv_out[q] = iv;
                div_dk_out[q] = ((1.0 - alpha) * t0 * v0 * dv0 + alpha * t1 * v1 * dv1) / (t * iv);
            }
        }
    }
}

void surface_quote_vjp_cpp(
    int Q, int n_u, int n_exp, int n_raw, int has_cal_jac,
    double k_min, double k_max, double min_vol,
    const double* strikes, const double* tenors, const double* quotes,
    const double* slice_c, const double* slice_sl, const double* slice_sr,
    const double* basis_c, const double* basis_sl, const double* basis_sr,
    const double* dedup_t, const double* cal_proj_jac,
    const double* t_q, const double* k_q, const double* g_iv,
    double* out_vjp
) {
    const int n_seg = std::max(n_u - 1, 1);
    const int stride_c = n_seg * n_exp;
    const int stride_b = n_seg * n_u;
    const double mv2 = min_vol * min_vol;
    std::vector<double> proj_q_grad(static_cast<size_t>(n_exp * n_u), 0.0);

    auto eval_raw = [&](int e, int seg, double k) -> double {
        if (k < k_min) return quotes[e * n_u] + slice_sl[e] * (k - k_min);
        if (k > k_max) return quotes[e * n_u + n_u - 1] + slice_sr[e] * (k - k_max);
        double dx = k - strikes[seg];
        double c0 = slice_c[seg * n_exp + e];
        double c1 = slice_c[stride_c + seg * n_exp + e];
        double c2 = slice_c[2 * stride_c + seg * n_exp + e];
        double c3 = slice_c[3 * stride_c + seg * n_exp + e];
        return ((c0 * dx + c1) * dx + c2) * dx + c3;
    };

    auto add_basis = [&](int e0, double g0, int e1, double g1, int seg, double k) {
        if (g0 == 0.0 && g1 == 0.0) return;
        double* p0 = &proj_q_grad[e0 * n_u];
        double* p1 = &proj_q_grad[e1 * n_u];
        if (k < k_min) {
            double d = k - k_min;
            if (g0 != 0.0) { p0[0] += g0; for (int u = 0; u < n_u; ++u) p0[u] += g0 * d * basis_sl[u]; }
            if (g1 != 0.0) { p1[0] += g1; for (int u = 0; u < n_u; ++u) p1[u] += g1 * d * basis_sl[u]; }
            return;
        }
        if (k > k_max) {
            double d = k - k_max;
            if (g0 != 0.0) { p0[n_u - 1] += g0; for (int u = 0; u < n_u; ++u) p0[u] += g0 * d * basis_sr[u]; }
            if (g1 != 0.0) { p1[n_u - 1] += g1; for (int u = 0; u < n_u; ++u) p1[u] += g1 * d * basis_sr[u]; }
            return;
        }
        double dx = k - strikes[seg];
        const double* b0 = &basis_c[seg * n_u];
        const double* b1 = &basis_c[stride_b + seg * n_u];
        const double* b2 = &basis_c[2 * stride_b + seg * n_u];
        const double* b3 = &basis_c[3 * stride_b + seg * n_u];
        for (int u = 0; u < n_u; ++u) {
            double wu = ((b0[u] * dx + b1[u]) * dx + b2[u]) * dx + b3[u];
            if (g0 != 0.0) p0[u] += g0 * wu;
            if (g1 != 0.0) p1[u] += g1 * wu;
        }
    };

    for (int q = 0; q < Q; ++q) {
        double t = t_q[q] > 1e-8 ? t_q[q] : 1e-8;
        double k = k_q[q];
        double gc = g_iv[q];
        auto it_s = std::upper_bound(strikes, strikes + n_u, k);
        int seg = static_cast<int>(it_s - strikes) - 1;
        if (seg < 0) seg = 0;
        if (seg > n_seg - 1) seg = n_seg - 1;
        if (n_exp == 1) {
            double v0_raw = eval_raw(0, seg, k);
            if (v0_raw >= min_vol) add_basis(0, gc, 0, 0.0, seg, k);
            continue;
        }
        auto it_t = std::upper_bound(tenors, tenors + n_exp, t);
        int idx0 = static_cast<int>(it_t - tenors) - 1;
        if (idx0 < 0) idx0 = 0;
        if (idx0 > n_exp - 2) idx0 = n_exp - 2;
        int idx1 = idx0 + 1;
        double v0_raw = eval_raw(idx0, seg, k);
        double v1_raw = eval_raw(idx1, seg, k);
        double v0 = v0_raw >= min_vol ? v0_raw : min_vol;
        double v1 = v1_raw >= min_vol ? v1_raw : min_vol;
        double t0 = tenors[idx0], t1 = tenors[idx1];
        double dt_span = (t1 - t0) > 1e-12 ? (t1 - t0) : 1e-12;
        double alpha = (t - t0) / dt_span;
        double wt = ((1.0 - alpha) * (v0 * v0 * t0) + alpha * (v1 * v1 * t1)) / t;
        double v_interp = std::sqrt(wt > mv2 ? wt : mv2);
        bool in_lo = (t <= tenors[0]);
        bool in_hi = (!in_lo) && (t >= tenors[n_exp - 1]);
        bool in_mid = (!in_lo) && (!in_hi) && (wt > mv2);
        double dv_dv0 = in_lo ? 1.0 : (in_mid ? (1.0 - alpha) * t0 * v0 / (t * v_interp) : 0.0);
        double dv_dv1 = in_hi ? 1.0 : (in_mid ? alpha * t1 * v1 / (t * v_interp) : 0.0);
        double g0 = (v0_raw >= min_vol) ? (gc * dv_dv0) : 0.0;
        double g1 = (v1_raw >= min_vol) ? (gc * dv_dv1) : 0.0;
        add_basis(idx0, g0, idx1, g1, seg, k);
    }

    std::vector<double> g_dedup_buf;
    const double* g_dedup = proj_q_grad.data();
    if (has_cal_jac) {
        g_dedup_buf.assign(static_cast<size_t>(n_exp * n_u), 0.0);
        for (int m = 0; m < n_exp; ++m) {
            for (int n = 0; n < n_exp; ++n) {
                const double* jptr = &cal_proj_jac[(m * n_exp + n) * n_u];
                const double* pptr = &proj_q_grad[m * n_u];
                double* dptr = &g_dedup_buf[n * n_u];
                for (int u = 0; u < n_u; ++u) dptr[u] += jptr[u] * pptr[u];
            }
        }
        g_dedup = g_dedup_buf.data();
    }
    for (int m = 0; m < n_exp; ++m) {
        const double* dptr = &g_dedup[m * n_u];
        double* optr = &out_vjp[m * n_raw];
        for (int r = 0; r < n_raw; ++r) {
            const double* dtptr = &dedup_t[r * n_u];
            double acc = 0.0;
            for (int u = 0; u < n_u; ++u) acc += dptr[u] * dtptr[u];
            optr[r] = acc;
        }
    }
}

void surr_features_fwd_bwd(
    int B, int D, int M, int q_len, int d_node, int d_global, int do_bwd,
    double k_eff_n,
    const double* m_b, const double* fr_b, const double* fp_b, const double* cd_b,
    const double* vp_b, const double* iv_b, const double* dw_in, const double* cm_in,
    const double* dfp_b, const double* fw_in, const double* vw_in, const double* p_in,
    const double* q_in, const double* lp_in, const double* r_spr,
    const double* tau_vec, const double* dtau_vec, const double* log_ko,
    const double* ic_vec, const double* log_cpn, const double* ig_vec,
    const double* cpn_vec, const double* role_a, const double* role_b,
    const double* g_static,
    double* nf_out, double* ef_out, double* sf_out, double* gf_out,
    const double* g_nf, const double* g_ef, const double* g_sf, const double* g_gf,
    double* g_m, double* g_fr, double* g_fp, double* g_cd,
    double* g_vp, double* g_iv, double* g_dw, double* g_cm,
    double* g_dfp, double* g_fw, double* g_vw, double* g_p,
    double* g_q, double* g_lp, double* g_rspr
) {
    const double tau_term = tau_vec[M - 1] > 1e-6 ? tau_vec[M - 1] : 1e-6;
    if (!do_bwd) {
        for (int b = 0; b < B; ++b) {
            const double* mb = &m_b[b * D];
            const double* frb = &fr_b[b * D * M];
            const double* fpb = &fp_b[b * D * M];
            const double* cdb = &cd_b[b * D * M];
            const double* vpb = &vp_b[b * D * M];
            const double* ivb = &iv_b[b * D * q_len];
            const double* dwb = &dw_in[b * D];
            const double* cmb = &cm_in[b * D * D];
            double* nfo = &nf_out[b * D * d_node];
            double* efo = &ef_out[b * D * D * 4];
            std::vector<double> vt(D), logfpt(D);
            for (int i = 0; i < D; ++i) {
                double mv = mb[i] > 1e-12 ? mb[i] : 1e-12;
                double frv = frb[i * M + M - 1] > 1e-12 ? frb[i * M + M - 1] : 1e-12;
                double cdv = cdb[i * M + M - 1] > 1.0 ? cdb[i * M + M - 1] : 1.0;
                double fpt = fpb[i * M + M - 1] > 1e-12 ? fpb[i * M + M - 1] : 1e-12;
                double vti = vpb[i * M + M - 1] > 1e-6 ? vpb[i * M + M - 1] : 1e-6;
                vt[i] = vti;
                logfpt[i] = std::log(fpt);
                const double* ivi = &ivb[i * q_len];
                double ivs = ivi[0], ivt = ivi[M - 1];
                double icv = ivi[M], im10 = ivi[M + 1], ip10 = ivi[M + 2], im20 = ivi[M + 3], ip20 = ivi[M + 4];
                double dr = (0.20 * icv > 1e-6) ? (0.20 * icv) : 1e-6;
                double de = (0.01 * icv > 1e-6) ? (0.01 * icv) : 1e-6;
                double eta = (ip10 - 2.0 * icv + im10) / de;
                if (eta < 1e-4) eta = 1e-4;
                double row[16] = {
                    std::log(mv), std::log(frv) / tau_term, std::log(cdv), r_spr[i],
                    ivs, ivt, ivt - ivs, std::tanh((ip10 - im10) / dr), eta,
                    im20 - icv, ip20 - icv, std::log(fpt / k_eff_n) / vti,
                    dwb[i], 1.0, role_a[i], role_b[i]
                };
                for (int c = 0; c < d_node; ++c) nfo[i * d_node + c] = row[c];
            }
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    int idx = (i * D + j) * 4;
                    double c = cmb[i * D + j];
                    efo[idx + 0] = c;
                    efo[idx + 1] = c * vt[i] * vt[j];
                    efo[idx + 2] = std::abs(logfpt[i] - logfpt[j]);
                    efo[idx + 3] = (i == j) ? 0.0 : 1.0;
                }
            }
            const double* dfpb = &dfp_b[b * M];
            const double* fwb = &fw_in[b * M];
            const double* vwb = &vw_in[b * M];
            const double* pb = &p_in[b * M];
            const double* qb = &q_in[b * M];
            double* sfo = &sf_out[b * M * 12];
            for (int m = 0; m < M; ++m) {
                int idx = m * 12;
                double fwv = fwb[m] > 1e-12 ? fwb[m] : 1e-12;
                sfo[idx + 0] = tau_vec[m];
                sfo[idx + 1] = dtau_vec[m];
                sfo[idx + 2] = dfpb[m];
                sfo[idx + 3] = log_ko[m];
                sfo[idx + 4] = ic_vec[m];
                sfo[idx + 5] = log_cpn[m];
                sfo[idx + 6] = ig_vec[m];
                sfo[idx + 7] = cpn_vec[m];
                sfo[idx + 8] = std::log(fwv);
                sfo[idx + 9] = vwb[m];
                sfo[idx + 10] = pb[m];
                sfo[idx + 11] = qb[m];
            }
            double* gfo = &gf_out[b * d_global];
            for (int c = 0; c < d_global; ++c) gfo[c] = (c == 9) ? lp_in[b] : g_static[c];
        }
        return;
    }

    for (int b = 0; b < B; ++b) {
        const double* mb = &m_b[b * D];
        const double* frb = &fr_b[b * D * M];
        const double* fpb = &fp_b[b * D * M];
        const double* cdb = &cd_b[b * D * M];
        const double* vpb = &vp_b[b * D * M];
        const double* ivb = &iv_b[b * D * q_len];
        const double* cmb = &cm_in[b * D * D];
        const double* gn = &g_nf[b * D * d_node];
        const double* ge = &g_ef[b * D * D * 4];
        const double* gs = &g_sf[b * M * 12];
        const double* gg = &g_gf[b * d_global];
        double* gmb = &g_m[b * D];
        double* gfrb = &g_fr[b * D * M];
        double* gfpb = &g_fp[b * D * M];
        double* gcdb = &g_cd[b * D * M];
        double* gvpb = &g_vp[b * D * M];
        double* givb = &g_iv[b * D * q_len];
        double* gdwb = &g_dw[b * D];
        double* gcmb = &g_cm[b * D * D];

        std::vector<double> vt(D), fpt(D), logfpt(D);
        for (int i = 0; i < D; ++i) {
            double fpti = fpb[i * M + M - 1] > 1e-12 ? fpb[i * M + M - 1] : 1e-12;
            double vti = vpb[i * M + M - 1] > 1e-6 ? vpb[i * M + M - 1] : 1e-6;
            fpt[i] = fpti;
            vt[i] = vti;
            logfpt[i] = std::log(fpti);
            const double* gn_i = &gn[i * d_node];
            double g_row[16] = {0.0};
            for (int c = 0; c < d_node; ++c) g_row[c] = gn_i[c];

            if (mb[i] > 1e-12) gmb[i] += g_row[0] / mb[i];
            if (frb[i * M + M - 1] > 1e-12) gfrb[i * M + M - 1] += g_row[1] / (frb[i * M + M - 1] * tau_term);
            if (cdb[i * M + M - 1] >= 1.0) gcdb[i * M + M - 1] += g_row[2] / cdb[i * M + M - 1];
            g_rspr[i] += g_row[3];

            const double* ivi = &ivb[i * q_len];
            double* givi = &givb[i * q_len];
            double icv = ivi[M], im10 = ivi[M + 1], ip10 = ivi[M + 2];
            givi[0] += g_row[4] - g_row[6];
            givi[M - 1] += g_row[5] + g_row[6];

            double dr = (0.20 * icv > 1e-6) ? (0.20 * icv) : 1e-6;
            double z_r = (ip10 - im10) / dr;
            double th_r = std::tanh(z_r);
            double gz = g_row[7] * (1.0 - th_r * th_r);
            givi[M + 2] += gz / dr;
            givi[M + 1] -= gz / dr;
            if (0.20 * icv > 1e-6) givi[M] -= gz * z_r / icv;

            double de = (0.01 * icv > 1e-6) ? (0.01 * icv) : 1e-6;
            double eta_raw = (ip10 - 2.0 * icv + im10) / de;
            if (eta_raw > 1e-4) {
                double g_e = g_row[8] / de;
                givi[M + 2] += g_e;
                givi[M + 1] += g_e;
                givi[M] -= 2.0 * g_e;
                if (0.01 * icv > 1e-6) givi[M] -= g_row[8] * eta_raw / icv;
            }
            givi[M + 3] += g_row[9];
            givi[M + 4] += g_row[10];
            givi[M] -= (g_row[9] + g_row[10]);

            double dtp = std::log(fpti / k_eff_n) / vti;
            if (fpb[i * M + M - 1] > 1e-12) gfpb[i * M + M - 1] += g_row[11] / (fpti * vti);
            if (vpb[i * M + M - 1] > 1e-6) gvpb[i * M + M - 1] -= g_row[11] * dtp / vti;
            gdwb[i] += g_row[12];
        }

        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                int idx = (i * D + j) * 4;
                double c = cmb[i * D + j];
                gcmb[i * D + j] += ge[idx + 0] + ge[idx + 1] * vt[i] * vt[j];
                if (vpb[i * M + M - 1] > 1e-6) gvpb[i * M + M - 1] += ge[idx + 1] * c * vt[j];
                if (vpb[j * M + M - 1] > 1e-6) gvpb[j * M + M - 1] += ge[idx + 1] * c * vt[i];
                double d_ij = logfpt[i] - logfpt[j];
                double sgn = (d_ij > 0.0) ? 1.0 : ((d_ij < 0.0) ? -1.0 : 0.0);
                double gd = ge[idx + 2] * sgn;
                if (fpb[i * M + M - 1] > 1e-12) gfpb[i * M + M - 1] += gd / fpt[i];
                if (fpb[j * M + M - 1] > 1e-12) gfpb[j * M + M - 1] -= gd / fpt[j];
            }
        }

        const double* fwb = &fw_in[b * M];
        double* gdfpb = &g_dfp[b * M];
        double* gfwb = &g_fw[b * M];
        double* gvwb = &g_vw[b * M];
        double* gpb = &g_p[b * M];
        double* gqb = &g_q[b * M];
        for (int m = 0; m < M; ++m) {
            int idx = m * 12;
            gdfpb[m] += gs[idx + 2];
            if (fwb[m] > 1e-12) gfwb[m] += gs[idx + 8] / fwb[m];
            gvwb[m] += gs[idx + 9];
            gpb[m] += gs[idx + 10];
            gqb[m] += gs[idx + 11];
        }
        if (d_global > 9) g_lp[b] += gg[9];
    }
}

void surr_post_pv_fwd_bwd(
    int B, int M, int has_surr, int has_w_fund,
    int no_notional_repay, int force_cpn_ac, int is_memory, int is_long_basket, int do_bwd,
    double c_mem_0, double ac_cpn, double opt_gear, double part_call_rate, double k_eff_put,
    const double* d_haz, const double* d_cpn, const double* d_put,
    const double* p_px, const double* q_px, const double* lp_px,
    const double* l_call, const double* l_onestar,
    const double* df_pay, const double* w_fund, const double* pv_in_tr,
    const double* ic, const double* ig, const double* cpn_rates,
    double* pv_out, double* p_stop_0, double* q_cpn_0, double* l_put_0,
    const double* g_pv,
    double* g_dhaz, double* g_dcpn, double* g_dput,
    double* g_ppx, double* g_qpx, double* g_lppx,
    double* g_lcall, double* g_lonestar,
    double* g_dfpay, double* g_wfund, double* g_pvintr
) {
    const double put_sign = is_long_basket ? 1.0 : -1.0;
    std::vector<double> p_px_c(M), cum_p_px(M), surv_prev_px(M), h_px_c(M), haz(M), surv1(M), surv1_prev(M), p_stop(M);
    std::vector<double> q_px_c(M), q_gate(M), sig_q(M), q_cpn(M);
    std::vector<double> p_t(M), surv2(M), surv2_prev(M), eff_q(M), mem_arr(M + 1), eff_c_arr(M);
    std::vector<double> g_p_stop(M), g_q_cpn(M), g_p_t(M), g_surv2(M), g_surv2_prev(M), g_eff_q(M), g_haz(M), g_surv1(M), g_surv1_prev(M);

    for (int b = 0; b < B; ++b) {
        const double* ppx = &p_px[b * M];
        const double* qpx = &q_px[b * M];
        const double* dfp = &df_pay[b * M];
        const double* wf = has_w_fund ? &w_fund[b * M] : nullptr;
        double lpp = lp_px[b];
        double l_put_v = 0.0;
        double surv_px_term = 1.0, sig_put = 0.0, u_put = 0.0;
        double denom_r = 1e-12, r_px = 1e-12, lp_c = 0.0;

        if (has_surr) {
            const double* dh = &d_haz[b * M];
            const double* dc = &d_cpn[b * M];
            double dp = d_put[b];
            double csum = 0.0;
            for (int m = 0; m < M; ++m) {
                double pc = ppx[m] < 0.0 ? 0.0 : (ppx[m] > 1.0 ? 1.0 : ppx[m]);
                pc *= ic[m];
                p_px_c[m] = pc;
                double prev_px = 1.0 - csum;
                csum += pc;
                cum_p_px[m] = csum;
                surv_prev_px[m] = prev_px > 1e-6 ? prev_px : 1e-6;
                double hp = pc / surv_prev_px[m];
                if (hp < 1e-6) hp = 1e-6;
                if (hp > 1.0 - 1e-6) hp = 1.0 - 1e-6;
                h_px_c[m] = hp;
                double lg = std::log(hp / (1.0 - hp)) + dh[m];
                double sg = 1.0 / (1.0 + std::exp(-lg));
                haz[m] = ic[m] * sg;
            }
            double s_run = 1.0;
            for (int m = 0; m < M; ++m) {
                surv1_prev[m] = s_run;
                s_run *= (1.0 - haz[m]);
                surv1[m] = s_run;
                p_stop[m] = surv1_prev[m] * haz[m];
            }
            for (int m = 0; m < M; ++m) {
                double qv = qpx[m];
                double qc = qv < 1e-6 ? 1e-6 : (qv > 1.0 - 1e-6 ? 1.0 - 1e-6 : qv);
                q_px_c[m] = qc;
                double q2 = qv * qv;
                double qg = q2 / (q2 + 1e-10);
                q_gate[m] = qg;
                double lg = std::log(qc / (1.0 - qc)) + dc[m];
                double sg = 1.0 / (1.0 + std::exp(-lg));
                sig_q[m] = sg;
                q_cpn[m] = ig[m] + (1.0 - ig[m]) * qg * sg;
            }
            surv_px_term = (1.0 - csum) > 0.0 ? (1.0 - csum) : 0.0;
            double ksp = k_eff_put * surv_px_term;
            denom_r = ksp > 1e-12 ? ksp : 1e-12;
            lp_c = lpp > 0.0 ? lpp : 0.0;
            double r_raw = lp_c / denom_r;
            r_px = r_raw < 1e-12 ? 1e-12 : (r_raw > 1.0 - 1e-12 ? 1.0 - 1e-12 : r_raw);
            double lg_r = std::log(r_px / (1.0 - r_px)) + dp;
            sig_put = 1.0 / (1.0 + std::exp(-lg_r));
            u_put = k_eff_put * surv1[M - 1];
            l_put_v = u_put * sig_put;
        } else {
            for (int m = 0; m < M; ++m) { p_stop[m] = ppx[m]; q_cpn[m] = qpx[m]; }
            l_put_v = lpp;
        }

        if (b == 0) {
            for (int m = 0; m < M; ++m) { p_stop_0[m] = p_stop[m]; q_cpn_0[m] = q_cpn[m]; }
            *l_put_0 = l_put_v;
        }

        double cp2 = 0.0;
        for (int m = 0; m < M; ++m) {
            double pt = p_stop[m] * ic[m];
            p_t[m] = pt;
            surv2_prev[m] = (m == 0) ? 1.0 : surv2[m - 1];
            cp2 += pt;
            double sv = 1.0 - cp2;
            surv2[m] = sv > 0.0 ? sv : 0.0;
        }
        double not_pv = 0.0;
        if (!no_notional_repay) {
            for (int m = 0; m < M; ++m) not_pv += dfp[m] * p_t[m];
            not_pv += dfp[M - 1] * surv2[M - 1];
        }
        for (int m = 0; m < M; ++m) {
            if (force_cpn_ac) {
                double sp_c = surv2_prev[m] > 1e-12 ? surv2_prev[m] : 1e-12;
                double rat = p_t[m] / sp_c;
                eff_q[m] = q_cpn[m] >= rat ? q_cpn[m] : rat;
            } else {
                eff_q[m] = q_cpn[m];
            }
        }
        double cpn_pv = 0.0;
        if (is_memory) {
            mem_arr[0] = c_mem_0;
            for (int m = 0; m < M; ++m) {
                double ec = cpn_rates[m] + mem_arr[m];
                eff_c_arr[m] = ec;
                cpn_pv += dfp[m] * (surv2_prev[m] * eff_q[m] * ec + p_t[m] * ac_cpn);
                mem_arr[m + 1] = ec * (1.0 - eff_q[m]);
            }
        } else {
            for (int m = 0; m < M; ++m) {
                cpn_pv += dfp[m] * (surv2_prev[m] * eff_q[m] * cpn_rates[m] + p_t[m] * ac_cpn);
            }
        }
        double opt_undisc = put_sign * opt_gear * l_put_v + part_call_rate * l_call[b] + l_onestar[b];
        double opt_pv = dfp[M - 1] * opt_undisc;
        double fund_pv = 0.0;
        if (has_w_fund) {
            for (int m = 0; m < M; ++m) fund_pv += surv2_prev[m] * wf[m];
        }
        pv_out[b] = not_pv + cpn_pv + opt_pv - fund_pv + pv_in_tr[b];

        if (!do_bwd) continue;
        double gp = g_pv[b];
        if (gp == 0.0) continue;
        g_pvintr[b] += gp;

        std::fill(g_p_t.begin(), g_p_t.end(), 0.0);
        std::fill(g_surv2.begin(), g_surv2.end(), 0.0);
        std::fill(g_surv2_prev.begin(), g_surv2_prev.end(), 0.0);
        std::fill(g_eff_q.begin(), g_eff_q.end(), 0.0);
        std::fill(g_q_cpn.begin(), g_q_cpn.end(), 0.0);
        std::fill(g_p_stop.begin(), g_p_stop.end(), 0.0);

        double* gdfpb = &g_dfpay[b * M];
        if (has_w_fund) {
            double* gwfb = &g_wfund[b * M];
            for (int m = 0; m < M; ++m) {
                gwfb[m] -= gp * surv2_prev[m];
                g_surv2_prev[m] -= gp * wf[m];
            }
        }
        gdfpb[M - 1] += gp * opt_undisc;
        double g_opt_u = gp * dfp[M - 1];
        double g_l_put = g_opt_u * put_sign * opt_gear;
        g_lcall[b] += g_opt_u * part_call_rate;
        g_lonestar[b] += g_opt_u;

        if (is_memory) {
            double g_mem = 0.0;
            for (int m = M - 1; m >= 0; --m) {
                double ec = eff_c_arr[m];
                double eqm = eff_q[m];
                double g_ec = g_mem * (1.0 - eqm);
                g_eff_q[m] -= g_mem * ec;
                double term_c = gp * dfp[m];
                gdfpb[m] += gp * (surv2_prev[m] * eqm * ec + p_t[m] * ac_cpn);
                g_surv2_prev[m] += term_c * eqm * ec;
                g_eff_q[m] += term_c * surv2_prev[m] * ec;
                g_ec += term_c * surv2_prev[m] * eqm;
                g_p_t[m] += term_c * ac_cpn;
                g_mem = g_ec;
            }
        } else {
            for (int m = 0; m < M; ++m) {
                double term_c = gp * dfp[m];
                gdfpb[m] += gp * (surv2_prev[m] * eff_q[m] * cpn_rates[m] + p_t[m] * ac_cpn);
                g_surv2_prev[m] += term_c * eff_q[m] * cpn_rates[m];
                g_eff_q[m] += term_c * surv2_prev[m] * cpn_rates[m];
                g_p_t[m] += term_c * ac_cpn;
            }
        }
        for (int m = 0; m < M; ++m) {
            if (force_cpn_ac) {
                double sp_c = surv2_prev[m] > 1e-12 ? surv2_prev[m] : 1e-12;
                double rat = p_t[m] / sp_c;
                if (q_cpn[m] >= rat) {
                    g_q_cpn[m] += g_eff_q[m];
                } else {
                    g_p_t[m] += g_eff_q[m] / sp_c;
                    if (surv2_prev[m] > 1e-12) g_surv2_prev[m] -= g_eff_q[m] * rat / surv2_prev[m];
                }
            } else {
                g_q_cpn[m] += g_eff_q[m];
            }
        }
        if (!no_notional_repay) {
            for (int m = 0; m < M; ++m) {
                gdfpb[m] += gp * p_t[m];
                g_p_t[m] += gp * dfp[m];
            }
            gdfpb[M - 1] += gp * surv2[M - 1];
            g_surv2[M - 1] += gp * dfp[M - 1];
        }
        double cp2_acc = 0.0;
        double cum_g_sv = 0.0;
        for (int m = M - 1; m >= 0; --m) {
            if (m < M - 1) g_surv2[m] += g_surv2_prev[m + 1];
            double sv_raw = 0.0;
            double csum2 = 0.0;
            for (int k = 0; k <= m; ++k) csum2 += p_t[k];
            sv_raw = 1.0 - csum2;
            double g_sv_u = (sv_raw > 0.0) ? -g_surv2[m] : 0.0;
            cum_g_sv += g_sv_u;
            g_p_t[m] += cum_g_sv;
            g_p_stop[m] += g_p_t[m] * ic[m];
        }
        (void)cp2_acc;

        double* gppxb = &g_ppx[b * M];
        double* gqpxb = &g_qpx[b * M];
        if (!has_surr) {
            for (int m = 0; m < M; ++m) { gppxb[m] += g_p_stop[m]; gqpxb[m] += g_q_cpn[m]; }
            g_lppx[b] += g_l_put;
            continue;
        }

        double* gdhb = &g_dhaz[b * M];
        double* gdcb = &g_dcpn[b * M];
        std::fill(g_haz.begin(), g_haz.end(), 0.0);
        std::fill(g_surv1.begin(), g_surv1.end(), 0.0);
        std::fill(g_surv1_prev.begin(), g_surv1_prev.end(), 0.0);

        // Put head backward
        g_surv1[M - 1] += g_l_put * k_eff_put * sig_put;
        double g_sig_put = g_l_put * u_put;
        double g_lgr = g_sig_put * sig_put * (1.0 - sig_put);
        g_dput[b] += g_lgr;
        if (lp_c / denom_r > 1e-12 && lp_c / denom_r < 1.0 - 1e-12) {
            double g_rpx = g_lgr / (r_px * (1.0 - r_px));
            if (lpp > 0.0) g_lppx[b] += g_rpx / denom_r;
            if (k_eff_put * surv_px_term > 1e-12) {
                double g_spxt = -g_rpx * (lp_c * k_eff_put) / (denom_r * denom_r);
                if (1.0 - cum_p_px[M - 1] > 0.0) {
                    for (int m = 0; m < M; ++m) {
                        if (ppx[m] > 0.0 && ppx[m] < 1.0) gppxb[m] -= g_spxt * ic[m];
                    }
                }
            }
        }

        // Coupon head backward
        for (int m = 0; m < M; ++m) {
            double gqp = g_q_cpn[m] * (1.0 - ig[m]);
            double gsgq = gqp * q_gate[m];
            double gqg = gqp * sig_q[m];
            double glgq = gsgq * sig_q[m] * (1.0 - sig_q[m]);
            gdcb[m] += glgq;
            double qv = qpx[m];
            double q2 = qv * qv;
            double dqg = gqg * (2.0 * qv * 1e-10) / ((q2 + 1e-10) * (q2 + 1e-10));
            gqpxb[m] += dqg;
            if (qv > 1e-6 && qv < 1.0 - 1e-6) {
                gqpxb[m] += glgq / (q_px_c[m] * (1.0 - q_px_c[m]));
            }
        }

        // Hazard stopping backward
        for (int m = M - 1; m >= 0; --m) {
            g_haz[m] += g_p_stop[m] * surv1_prev[m];
            g_surv1_prev[m] += g_p_stop[m] * haz[m];
            double g_s1 = g_surv1[m];
            g_haz[m] -= g_s1 * surv1_prev[m];
            g_surv1_prev[m] += g_s1 * (1.0 - haz[m]);
            if (m > 0) g_surv1[m - 1] += g_surv1_prev[m];
        }
        double cum_g_prevpx = 0.0;
        for (int m = M - 1; m >= 0; --m) {
            double g_sg = g_haz[m] * ic[m];
            double sg = (ic[m] > 0.5) ? (haz[m]) : 0.0;
            // Recompute sigmoid(logit_h_proxy + dh) accurately when ic == 0 too
            double lg_h = std::log(h_px_c[m] / (1.0 - h_px_c[m])) + d_haz[b * M + m];
            sg = 1.0 / (1.0 + std::exp(-lg_h));
            double g_lg = g_sg * sg * (1.0 - sg);
            gdhb[m] += g_lg;
            double pc = p_px_c[m];
            double spx = surv_prev_px[m];
            double hp_raw = pc / spx;
            double g_hp = (hp_raw > 1e-6 && hp_raw < 1.0 - 1e-6) ? (g_lg / (h_px_c[m] * (1.0 - h_px_c[m]))) : 0.0;
            double g_pc = g_hp / spx;
            double g_spx = -g_hp * pc / (spx * spx);
            double prev_raw = 1.0 - (cum_p_px[m] - pc);
            double g_prev_sum = (prev_raw > 1e-6) ? -g_spx : 0.0;
            g_pc += cum_g_prevpx;
            cum_g_prevpx += g_prev_sum;
            if (ppx[m] > 0.0 && ppx[m] < 1.0) gppxb[m] += g_pc * ic[m];
        }
    }
}
}
"""

_TIER1_CPP_AD_LOCK = threading.Lock()
_TIER1_CPP_AD_FN: Any | None = None
_SURFACE_CPP_AD_FN: Any | None = None
_SURFACE_VJP_CPP_FN: Any | None = None
_SURR_FEAT_CPP_FN: Any | None = None
_SURR_POST_CPP_FN: Any | None = None
_TIER1_CPP_AD_FAILED = False


def _load_tier1_cpp_ad_fn() -> Any | None:
    """Lazily builds/loads the cached C++23 float64 Wengert-tape Tier-1 & Surface AD shared library."""
    global _TIER1_CPP_AD_FN, _SURFACE_CPP_AD_FN, _SURFACE_VJP_CPP_FN, _SURR_FEAT_CPP_FN, _SURR_POST_CPP_FN, _TIER1_CPP_AD_FAILED
    if _TIER1_CPP_AD_FN is not None:
        return _TIER1_CPP_AD_FN
    if _TIER1_CPP_AD_FAILED:
        return None
    with _TIER1_CPP_AD_LOCK:
        if _TIER1_CPP_AD_FN is not None:
            return _TIER1_CPP_AD_FN
        if _TIER1_CPP_AD_FAILED:
            return None
        try:
            src_hash = hashlib.sha256(_TIER1_CPP_AD_SRC.encode("utf-8")).hexdigest()[:16]
            cache_dir = Path.home() / ".cache" / "deepvol" / "tier1_ad"
            cache_dir.mkdir(parents=True, exist_ok=True)
            so_path = cache_dir / f"tier1_ad_{src_hash}.so"
            if not so_path.is_file():
                cpp_path = cache_dir / f"tier1_ad_{src_hash}.{os.getpid()}.cpp"
                tmp_so = cache_dir / f"tier1_ad_{src_hash}.{os.getpid()}.so"
                cpp_path.write_text(_TIER1_CPP_AD_SRC, encoding="utf-8")
                subprocess.run(
                    ["g++", "-O3", "-std=c++23", "-fno-fast-math", "-fopenmp", "-shared", "-fPIC", str(cpp_path), "-o", str(tmp_so)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                os.replace(tmp_so, so_path)
                cpp_path.unlink(missing_ok=True)
            lib = ctypes.CDLL(str(so_path))
            c_fn = lib.tier1_pure_fwd_bwd
            c_fn.argtypes = (
                [ctypes.c_int] * 6
                + [ctypes.c_void_p] * 3
                + [ctypes.c_void_p] * 4
                + [ctypes.c_void_p] * 3
                + [ctypes.c_int] * 7
                + [ctypes.c_double] * 6
                + [ctypes.c_int, ctypes.c_double]
                + [ctypes.c_void_p] * 15
            )
            c_fn.restype = None
            s_fn = lib.surface_eval_fwd_dk
            s_fn.argtypes = (
                [ctypes.c_int] * 3
                + [ctypes.c_double] * 3
                + [ctypes.c_void_p] * 10
            )
            s_fn.restype = None
            sv_fn = lib.surface_quote_vjp_cpp
            sv_fn.argtypes = (
                [ctypes.c_int] * 5
                + [ctypes.c_double] * 3
                + [ctypes.c_void_p] * 15
            )
            sv_fn.restype = None
            sf_fn = lib.surr_features_fwd_bwd
            sf_fn.argtypes = (
                [ctypes.c_int] * 7
                + [ctypes.c_double]
                + [ctypes.c_void_p] * 48
            )
            sf_fn.restype = None
            sp_fn = lib.surr_post_pv_fwd_bwd
            sp_fn.argtypes = (
                [ctypes.c_int] * 9
                + [ctypes.c_double] * 5
                + [ctypes.c_void_p] * 30
            )
            sp_fn.restype = None
            _SURFACE_CPP_AD_FN = s_fn
            _SURFACE_VJP_CPP_FN = sv_fn
            _SURR_FEAT_CPP_FN = sf_fn
            _SURR_POST_CPP_FN = sp_fn
            _TIER1_CPP_AD_FN = c_fn
            return c_fn
        except Exception:
            _TIER1_CPP_AD_FAILED = True
            return None


class _SurfaceEvalCppFn(torch.autograd.Function):
    """Fast C++23 regularized cubic-spline VolSurface evaluator with exact backward & jvp."""

    @staticmethod
    def forward(
        k_queries: torch.Tensor,
        t_vals_np: npt.NDArray[np.float64],
        geom: _SurfaceGeomCache,
        qcache: _SurfaceQuoteCache,
        min_vol: float,
        surf: RegularizedVolSurface,
        t_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del surf, t_vals
        s_fn = _SURFACE_CPP_AD_FN
        assert s_fn is not None
        assert geom.strikes_np is not None and geom.tenors_np is not None
        assert qcache.quotes_np is not None and qcache.slice_c_np is not None
        assert qcache.slope_left_np is not None and qcache.slope_right_np is not None
        dev = k_queries.device
        q_cnt = int(k_queries.shape[0])
        k_np = np.ascontiguousarray(k_queries.detach().cpu().numpy(), dtype=np.float64)
        iv_np = np.empty((q_cnt,), dtype=np.float64)
        dk_np = np.empty((q_cnt,), dtype=np.float64)
        s_fn(
            q_cnt, geom.n_unique, geom.n_exp,
            geom.k_min, geom.k_max, float(min_vol),
            geom.strikes_ptr, geom.tenors_ptr, qcache.quotes_ptr,
            qcache.slice_c_ptr, qcache.slope_left_ptr, qcache.slope_right_ptr,
            t_vals_np.__array_interface__["data"][0], k_np.__array_interface__["data"][0],
            iv_np.__array_interface__["data"][0], dk_np.__array_interface__["data"][0],
        )
        return torch.from_numpy(iv_np).to(device=dev), torch.from_numpy(dk_np).to(device=dev)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        k_queries, _, geom, qcache, _, surf, t_vals = inputs
        ctx.save_for_backward(k_queries, t_vals, output[1])
        ctx.surf_meta = (surf, geom, qcache)

    @staticmethod
    def backward(ctx: Any, g_iv: torch.Tensor, g_dk: torch.Tensor) -> tuple[Any, ...]:
        del g_dk
        k_queries, t_vals, dk_t = ctx.saved_tensors
        if g_iv is not None and (torch.is_grad_enabled() or g_iv.requires_grad) and bool(ctx.needs_input_grad[0]):
            surf, geom, qcache = ctx.surf_meta
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                k_g = k_queries if k_queries.requires_grad else k_queries.detach().clone().requires_grad_(True)
                iv_t = _eval_reg_surface_iv_batch_torch(
                    surf, t_vals, k_g, None, cached_pair=(geom, qcache), allow_cpp=False
                )
                g_k = torch.autograd.grad(
                    iv_t, k_g, grad_outputs=g_iv, create_graph=True, allow_unused=True
                )[0]
                return g_k, None, None, None, None, None, None
        return g_iv * dk_t, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx: Any, d_k: torch.Tensor, *_: Any) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, dk_t = ctx.saved_tensors
        return (d_k * dk_t if d_k is not None else torch.zeros_like(dk_t)), torch.zeros_like(dk_t)


class _MultiSurfaceEvalCppFn(torch.autograd.Function):
    """Single-dispatch C++23 spline evaluation across all D assets (eliminates D-1 Function.apply calls)."""

    @staticmethod
    def forward(
        k_all: torch.Tensor,
        t_vals_np: npt.NDArray[np.float64],
        asset_metas: tuple[tuple[RegularizedVolSurface, _SurfaceGeomCache, _SurfaceQuoteCache], ...],
        t_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del t_vals
        s_fn = _SURFACE_CPP_AD_FN
        assert s_fn is not None
        d_cnt, q_cnt = int(k_all.shape[0]), int(k_all.shape[1])
        dev = k_all.device
        k_np = np.ascontiguousarray(k_all.detach().cpu().numpy(), dtype=np.float64)
        iv_np = np.empty((d_cnt, q_cnt), dtype=np.float64)
        dk_np = np.empty((d_cnt, q_cnt), dtype=np.float64)
        t_ptr = int(t_vals_np.__array_interface__["data"][0])
        k_base_ptr = int(k_np.__array_interface__["data"][0])
        iv_base_ptr = int(iv_np.__array_interface__["data"][0])
        dk_base_ptr = int(dk_np.__array_interface__["data"][0])
        row_bytes = q_cnt * 8
        for i in range(d_cnt):
            surf_i, geom, qcache = asset_metas[i]
            off = i * row_bytes
            s_fn(
                q_cnt, geom.n_unique, geom.n_exp,
                geom.k_min, geom.k_max, float(surf_i.min_vol),
                geom.strikes_ptr, geom.tenors_ptr, qcache.quotes_ptr,
                qcache.slice_c_ptr, qcache.slope_left_ptr, qcache.slope_right_ptr,
                t_ptr, k_base_ptr + off,
                iv_base_ptr + off, dk_base_ptr + off,
            )
        return torch.from_numpy(iv_np).to(device=dev), torch.from_numpy(dk_np).to(device=dev)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        k_all, _, asset_metas, t_vals = inputs
        ctx.save_for_backward(k_all, t_vals, output[1])
        ctx.asset_metas = asset_metas

    @staticmethod
    def backward(ctx: Any, g_iv: torch.Tensor, g_dk: torch.Tensor) -> tuple[Any, ...]:
        del g_dk
        k_all, t_vals, dk_t = ctx.saved_tensors
        if g_iv is not None and (torch.is_grad_enabled() or g_iv.requires_grad) and bool(ctx.needs_input_grad[0]):
            metas = ctx.asset_metas
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                k_g = k_all if k_all.requires_grad else k_all.detach().clone().requires_grad_(True)
                rows_iv = [
                    _eval_reg_surface_iv_batch_torch(
                        metas[i][0], t_vals, k_g[i], None, cached_pair=(metas[i][1], metas[i][2]), allow_cpp=False
                    )
                    for i in range(int(k_all.shape[0]))
                ]
                iv_stack = torch.stack(rows_iv, dim=0)
                g_k = torch.autograd.grad(
                    iv_stack, k_g, grad_outputs=g_iv, create_graph=True, allow_unused=True
                )[0]
                return g_k, None, None, None
        return g_iv * dk_t, None, None, None

    @staticmethod
    def jvp(ctx: Any, d_k: torch.Tensor, *_: Any) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, dk_t = ctx.saved_tensors
        return (d_k * dk_t if d_k is not None else torch.zeros_like(dk_t)), torch.zeros_like(dk_t)


def _run_tier1_cpp_forward_numpy(
    fp_np: npt.NDArray[np.float64],
    vp_np: npt.NDArray[np.float64],
    corr_np: npt.NDArray[np.float64],
    ko_np: npt.NDArray[np.float64],
    ic_np: npt.NDArray[np.int32],
    cb_np: npt.NDArray[np.float64],
    ig_np: npt.NDArray[np.float64],
    oc_np: npt.NDArray[np.int64],
    ocp_np: npt.NDArray[np.int64],
    oop_np: npt.NDArray[np.int64],
    meta: tuple[Any, ...],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    c_fn = _load_tier1_cpp_ad_fn()
    assert c_fn is not None
    b_sz, d_p, m_p = fp_np.shape
    fw_out = np.empty((b_sz, m_p), dtype=np.float64)
    vw_out = np.empty((b_sz, m_p), dtype=np.float64)
    p_out = np.empty((b_sz, m_p), dtype=np.float64)
    q_out = np.empty((b_sz, m_p), dtype=np.float64)
    lput_out = np.empty((b_sz,), dtype=np.float64)
    dw_out = np.empty((b_sz, d_p), dtype=np.float64)
    c_fn(
        b_sz, d_p, m_p, 0, 0, 0,
        fp_np.ctypes.data, vp_np.ctypes.data, corr_np.ctypes.data,
        ko_np.ctypes.data, ic_np.ctypes.data, cb_np.ctypes.data, ig_np.ctypes.data,
        oc_np.ctypes.data, ocp_np.ctypes.data, oop_np.ctypes.data,
        *meta,
        fw_out.ctypes.data, vw_out.ctypes.data, p_out.ctypes.data,
        q_out.ctypes.data, lput_out.ctypes.data, dw_out.ctypes.data,
        None, None, None, None, None, None, None, None, None,
    )
    return fw_out, vw_out, p_out, q_out, lput_out, dw_out


class _Tier1CppAdFn(torch.autograd.Function):
    """Thread-safe IEEE-754 float64 Tier-1 AD bridge with setup_context, backward, and jvp."""

    @staticmethod
    def forward(
        fp: torch.Tensor,
        vp: torch.Tensor,
        corr: torch.Tensor,
        ko_t: torch.Tensor,
        cb_t: torch.Tensor,
        ig_t: torch.Tensor,
        ord_call: torch.Tensor,
        ord_cpn: torch.Tensor,
        ord_opt: torch.Tensor,
        is_call_flags: tuple[bool, ...],
        meta: tuple[Any, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del is_call_flags
        dev = fp.device
        fp_np = np.ascontiguousarray(fp.detach().cpu().numpy(), dtype=np.float64)
        vp_np = np.ascontiguousarray(vp.detach().cpu().numpy(), dtype=np.float64)
        corr_np = np.ascontiguousarray(corr.detach().cpu().numpy(), dtype=np.float64)
        ko_np = np.ascontiguousarray(ko_t.detach().cpu().numpy(), dtype=np.float64)
        cb_np = np.ascontiguousarray(cb_t.detach().cpu().numpy(), dtype=np.float64)
        ig_np = np.ascontiguousarray(ig_t.detach().cpu().numpy(), dtype=np.float64)
        ic_np = meta[-1]
        oc_np = np.ascontiguousarray(ord_call.detach().cpu().numpy(), dtype=np.int64)
        ocp_np = np.ascontiguousarray(ord_cpn.detach().cpu().numpy(), dtype=np.int64)
        oop_np = np.ascontiguousarray(ord_opt.detach().cpu().numpy(), dtype=np.int64)
        fw_n, vw_n, p_n, q_n, lp_n, dw_n = _run_tier1_cpp_forward_numpy(
            fp_np, vp_np, corr_np, ko_np, ic_np, cb_np, ig_np, oc_np, ocp_np, oop_np, meta[:-1]
        )
        return (
            torch.from_numpy(fw_n).to(device=dev),
            torch.from_numpy(vw_n).to(device=dev),
            torch.from_numpy(p_n).to(device=dev),
            torch.from_numpy(q_n).to(device=dev),
            torch.from_numpy(lp_n).to(device=dev),
            torch.from_numpy(dw_n).to(device=dev),
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        fp, vp, corr, ko_t, cb_t, ig_t, ord_call, ord_cpn, ord_opt, is_call_flags, meta = inputs
        ctx.save_for_backward(fp, vp, corr, ko_t, cb_t, ig_t, ord_call, ord_cpn, ord_opt)
        ctx.is_call_flags = is_call_flags
        ctx.meta = meta

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        g_fw, g_vw, g_p, g_q, g_lp, g_dw = grad_outputs
        fp, vp, corr, ko_t, cb_t, ig_t, ord_call, ord_cpn, ord_opt = ctx.saved_tensors
        need_fp = bool(ctx.needs_input_grad[0])
        need_vp = bool(ctx.needs_input_grad[1])
        need_corr = bool(ctx.needs_input_grad[2])
        if not (need_fp or need_vp or need_corr):
            return (None,) * 11
        if torch.is_grad_enabled() or any(g is not None and g.requires_grad for g in grad_outputs):
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                fp_g = fp if fp.requires_grad else fp.detach().clone().requires_grad_(need_fp)
                vp_g = vp if vp.requires_grad else vp.detach().clone().requires_grad_(need_vp)
                c_g = corr if corr.requires_grad else corr.detach().clone().requires_grad_(need_corr)
                meta = ctx.meta
                out_t = _tier1_baseline_pure_torch(
                    fp_g, vp_g, c_g, ko_t, cb_t, ig_t, ord_call, ord_cpn, ord_opt,
                    is_call_flags=ctx.is_call_flags,
                    call_code=meta[0], cpn_code=meta[1], opt_code=meta[2],
                    cpn_b_type="out" if meta[4] else "in",
                    cpn_b_dir="down" if meta[3] else "up",
                    force_cpn=bool(meta[5]), allow_exp=bool(meta[6]),
                    ko_spread=float(meta[7]), cpn_spread=float(meta[8]), ki_spread=float(meta[9]),
                    strike=float(meta[10]), option_gearing=float(meta[11]),
                    lower_strike_val=float(meta[12]), has_active_ki=bool(meta[13]),
                    ki_barrier_val=float(meta[14]), ko_b_term=float(ko_t[-1].detach().cpu().item()),
                )
                pairs = [(o, g) for o, g in zip(out_t, grad_outputs) if g is not None and o.requires_grad]
                if not pairs:
                    return (None,) * 11
                targets = [o for o, _ in pairs]
                grads_w = [g for _, g in pairs]
                req_in = [t for t, flg in ((fp_g, need_fp), (vp_g, need_vp), (c_g, need_corr)) if flg]
                g_res = torch.autograd.grad(targets, req_in, grad_outputs=grads_w, create_graph=True, allow_unused=True)
                it = iter(g_res)
                return (
                    next(it) if need_fp else None,
                    next(it) if need_vp else None,
                    next(it) if need_corr else None,
                    None, None, None, None, None, None, None, None,
                )

        c_fn = _load_tier1_cpp_ad_fn()
        assert c_fn is not None
        dev = fp.device
        b_sz, d_p, m_p = fp.shape
        fp_np = np.ascontiguousarray(fp.detach().cpu().numpy(), dtype=np.float64)
        vp_np = np.ascontiguousarray(vp.detach().cpu().numpy(), dtype=np.float64)
        corr_np = np.ascontiguousarray(corr.detach().cpu().numpy(), dtype=np.float64)
        ko_np = np.ascontiguousarray(ko_t.detach().cpu().numpy(), dtype=np.float64)
        cb_np = np.ascontiguousarray(cb_t.detach().cpu().numpy(), dtype=np.float64)
        ig_np = np.ascontiguousarray(ig_t.detach().cpu().numpy(), dtype=np.float64)
        meta = ctx.meta
        ic_np = meta[-1]
        oc_np = np.ascontiguousarray(ord_call.detach().cpu().numpy(), dtype=np.int64)
        ocp_np = np.ascontiguousarray(ord_cpn.detach().cpu().numpy(), dtype=np.int64)
        oop_np = np.ascontiguousarray(ord_opt.detach().cpu().numpy(), dtype=np.int64)

        def _cot_ptr(g_t: torch.Tensor | None) -> tuple[npt.NDArray[np.float64] | None, Any]:
            if g_t is None:
                return None, None
            arr = np.ascontiguousarray(g_t.detach().cpu().numpy(), dtype=np.float64)
            return arr, arr.ctypes.data

        gfw_a, gfw_p = _cot_ptr(g_fw)
        gvw_a, gvw_p = _cot_ptr(g_vw)
        gp_a, gp_p = _cot_ptr(g_p)
        gq_a, gq_p = _cot_ptr(g_q)
        glp_a, glp_p = _cot_ptr(g_lp)
        gdw_a, gdw_p = _cot_ptr(g_dw)
        _keep = (gfw_a, gvw_a, gp_a, gq_a, glp_a, gdw_a)
        del _keep

        g_fp_np = np.zeros((b_sz, d_p, m_p), dtype=np.float64) if need_fp else None
        g_vp_np = np.zeros((b_sz, d_p, m_p), dtype=np.float64) if need_vp else None
        g_corr_np = np.zeros((b_sz, d_p, d_p), dtype=np.float64) if need_corr else None
        c_fn(
            b_sz, d_p, m_p, int(need_fp), int(need_vp), 2 if need_corr else 0,
            fp_np.ctypes.data, vp_np.ctypes.data, corr_np.ctypes.data,
            ko_np.ctypes.data, ic_np.ctypes.data, cb_np.ctypes.data, ig_np.ctypes.data,
            oc_np.ctypes.data, ocp_np.ctypes.data, oop_np.ctypes.data,
            *meta[:-1],
            None, None, None, None, None, None,
            gfw_p, gvw_p, gp_p, gq_p, glp_p, gdw_p,
            g_fp_np.ctypes.data if g_fp_np is not None else None,
            g_vp_np.ctypes.data if g_vp_np is not None else None,
            g_corr_np.ctypes.data if g_corr_np is not None else None,
        )
        return (
            torch.from_numpy(g_fp_np).to(device=dev) if g_fp_np is not None else None,
            torch.from_numpy(g_vp_np).to(device=dev) if g_vp_np is not None else None,
            torch.from_numpy(g_corr_np).to(device=dev) if g_corr_np is not None else None,
            None, None, None, None, None, None, None, None,
        )

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> tuple[torch.Tensor, ...]:
        d_fp, d_vp, d_corr = grad_inputs[0], grad_inputs[1], grad_inputs[2]
        fp, vp, corr, ko_t, cb_t, ig_t, ord_call, ord_cpn, ord_opt = ctx.saved_tensors
        dev = fp.device
        fp_n = np.ascontiguousarray(fp.detach().cpu().numpy(), dtype=np.float64)
        vp_n = np.ascontiguousarray(vp.detach().cpu().numpy(), dtype=np.float64)
        c_n = np.ascontiguousarray(corr.detach().cpu().numpy(), dtype=np.float64)
        df_n = np.ascontiguousarray(d_fp.detach().cpu().numpy(), dtype=np.float64) if d_fp is not None else np.zeros_like(fp_n)
        dv_n = np.ascontiguousarray(d_vp.detach().cpu().numpy(), dtype=np.float64) if d_vp is not None else np.zeros_like(vp_n)
        dc_n = np.ascontiguousarray(d_corr.detach().cpu().numpy(), dtype=np.float64) if d_corr is not None else np.zeros_like(c_n)
        ko_n = np.ascontiguousarray(ko_t.detach().cpu().numpy(), dtype=np.float64)
        cb_n = np.ascontiguousarray(cb_t.detach().cpu().numpy(), dtype=np.float64)
        ig_n = np.ascontiguousarray(ig_t.detach().cpu().numpy(), dtype=np.float64)
        meta = ctx.meta
        ic_n = meta[-1]
        oc_n = np.ascontiguousarray(ord_call.detach().cpu().numpy(), dtype=np.int64)
        ocp_n = np.ascontiguousarray(ord_cpn.detach().cpu().numpy(), dtype=np.int64)
        oop_n = np.ascontiguousarray(ord_opt.detach().cpu().numpy(), dtype=np.int64)
        h = 1.0e-7
        out_p = _run_tier1_cpp_forward_numpy(
            fp_n + h * df_n, vp_n + h * dv_n, c_n + h * dc_n,
            ko_n, ic_n, cb_n, ig_n, oc_n, ocp_n, oop_n, meta[:-1],
        )
        out_m = _run_tier1_cpp_forward_numpy(
            fp_n - h * df_n, vp_n - h * dv_n, c_n - h * dc_n,
            ko_n, ic_n, cb_n, ig_n, oc_n, ocp_n, oop_n, meta[:-1],
        )
        return tuple(torch.from_numpy((op - om) / (2.0 * h)).to(device=dev) for op, om in zip(out_p, out_m))


def _run_tier1_pure_cached(
    fp: torch.Tensor,
    vp: torch.Tensor,
    corr: torch.Tensor,
    ko_t: torch.Tensor,
    cb_t: torch.Tensor,
    ig_t: torch.Tensor,
    *,
    is_call_flags: tuple[bool, ...],
    call_code: int,
    cpn_code: int,
    opt_code: int,
    cpn_b_type: str,
    cpn_b_dir: str,
    force_cpn: bool,
    allow_exp: bool,
    ko_spread: float,
    cpn_spread: float,
    ki_spread: float,
    strike: float,
    option_gearing: float,
    lower_strike_val: float,
    has_active_ki: bool,
    ki_barrier_val: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatches Tier-1 baseline to cached C++23 Wengert-tape AD when available, else pure PyTorch."""
    b_sz_p, d_p, m_p = fp.shape
    f_safe_p = fp.detach().transpose(1, 2).reshape(b_sz_p * m_p, d_p).clamp(min=1e-16)
    ord_asc = (
        torch.argsort(f_safe_p, dim=1, stable=True)
        if d_p > 1
        else torch.zeros((b_sz_p * m_p, 1), dtype=torch.long, device=fp.device)
    )
    ord_desc = ord_asc.flip(dims=[1]) if d_p > 1 else ord_asc
    ord_call = ord_desc if call_code == 1 else ord_asc
    ord_cpn = ord_desc if cpn_code == 1 else ord_asc
    ord_opt_asc = ord_asc.view(b_sz_p, m_p, d_p)[:, -1, :]
    ord_opt = ord_opt_asc.flip(dims=[1]) if (opt_code == 1 and d_p > 1) else ord_opt_asc

    if d_p <= 32 and _load_tier1_cpp_ad_fn() is not None:
        ic_arr = np.array([1 if flg else 0 for flg in is_call_flags], dtype=np.int32)
        meta: tuple[Any, ...] = (
            int(call_code),
            int(cpn_code),
            int(opt_code),
            1 if cpn_b_dir.strip().lower() == "down" else 0,
            1 if cpn_b_type.strip().lower() == "out" else 0,
            1 if force_cpn else 0,
            1 if allow_exp else 0,
            float(ko_spread),
            float(cpn_spread),
            float(ki_spread),
            float(strike),
            float(option_gearing),
            float(lower_strike_val),
            1 if has_active_ki else 0,
            float(ki_barrier_val),
            ic_arr,
        )
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            _Tier1CppAdFn.apply(fp, vp, corr, ko_t, cb_t, ig_t, ord_call, ord_cpn, ord_opt, is_call_flags, meta),  # type: ignore[no-untyped-call]
        )

    ko_b_term = float(ko_t[-1].detach().cpu().item())
    return _tier1_baseline_pure_torch(
        fp, vp, corr, ko_t, cb_t, ig_t, ord_call, ord_cpn, ord_opt,
        is_call_flags=is_call_flags, call_code=call_code, cpn_code=cpn_code, opt_code=opt_code,
        cpn_b_type=cpn_b_type, cpn_b_dir=cpn_b_dir, force_cpn=force_cpn, allow_exp=allow_exp,
        ko_spread=ko_spread, cpn_spread=cpn_spread, ki_spread=ki_spread, strike=strike,
        option_gearing=option_gearing, lower_strike_val=lower_strike_val,
        has_active_ki=has_active_ki, ki_barrier_val=ki_barrier_val, ko_b_term=ko_b_term,
    )


def _call_surrogate_residuals_fast(
    surrogate: MaskedEQAutocallSurrogate,
    nf_t2: torch.Tensor,
    ef_t2: torch.Tensor,
    am_t2: torch.Tensor,
    sf_t2: torch.Tensor,
    sm_t2: torch.Tensor,
    gf_t2: torch.Tensor,
    *,
    require_second_order_sdpa: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invokes surrogate residual forward consistently across cold and warm calls."""
    fwd_res_fn = getattr(surrogate, "forward_residuals", None)
    if callable(fwd_res_fn):
        out_r = fwd_res_fn(
            nf_t2, ef_t2, am_t2, sf_t2, sm_t2, gf_t2, require_second_order_sdpa=require_second_order_sdpa
        )
        return out_r[0], out_r[1], out_r[2]
    surr_out = surrogate(
        node_feats=nf_t2,
        edge_feats=ef_t2,
        asset_mask=am_t2,
        sched_feats=sf_t2,
        sched_mask=sm_t2,
        global_feats=gf_t2,
        return_residuals_only=True,
        all_active=True,
    )
    return surr_out.delta_hazard_logits, surr_out.delta_coupon_logits, surr_out.delta_put_loss


def _get_surrogate_cuda_graph_runner(
    surrogate: MaskedEQAutocallSurrogate,
    nf_shape: tuple[int, ...],
    ef_shape: tuple[int, ...],
    sf_shape: tuple[int, ...],
    gf_shape: tuple[int, ...],
    surr_dev: torch.device,
    surr_dtype: torch.dtype,
    m_rem: int,
) -> tuple[Any, ...] | None:
    """Returns per-thread split (fwd, bwd) CUDAGraph runner for static surrogate shapes."""
    if not hasattr(surrogate, "forward_residuals"):
        return None
    b_tot = int(nf_shape[0])
    key = (
        threading.get_ident(),
        id(surrogate),
        str(surr_dev),
        str(surr_dtype),
        nf_shape,
        ef_shape,
        sf_shape,
        gf_shape,
        int(m_rem),
    )
    cached = _SURROGATE_CUDAGRAPH_REGISTRY.get(key)
    if cached is not None:
        return cast(tuple[Any, ...], cached)
    try:
        with _COMPILE_LOCK:
            cached = _SURROGATE_CUDAGRAPH_REGISTRY.get(key)
            if cached is not None:
                return cast(tuple[Any, ...], cached)
            m_copy = copy.deepcopy(surrogate).to(device=surr_dev, dtype=surr_dtype).eval()
            for param in m_copy.parameters():
                param.requires_grad_(False)
            # Warm primary cuBLAS context before graph capture to prevent first-capture handle init warnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _z = torch.ones((1, 1), device=surr_dev, dtype=surr_dtype)
                _ = torch.matmul(_z, _z)

            n1 = int(nf_shape[0] * nf_shape[1] * nf_shape[2])
            n2 = int(ef_shape[0] * ef_shape[1] * ef_shape[2] * ef_shape[3])
            n3 = int(sf_shape[0] * sf_shape[1] * sf_shape[2])
            n4 = int(gf_shape[0] * gf_shape[1])
            tot_in = n1 + n2 + n3 + n4
            nh = b_tot * m_rem
            tot_out = 2 * nh + b_tot

            stream = torch.cuda.Stream(device=surr_dev)  # type: ignore[no-untyped-call]
            s_packed_g = torch.randn(tot_in, device=surr_dev, dtype=surr_dtype) * 0.05
            s_am_g = torch.ones((b_tot, int(nf_shape[1])), device=surr_dev, dtype=surr_dtype)
            s_sm_g = torch.ones((b_tot, int(sf_shape[1])), device=surr_dev, dtype=surr_dtype)
            s_out_g = torch.empty(tot_out, device=surr_dev, dtype=surr_dtype)
            s_gin_g = torch.ones(tot_out, device=surr_dev, dtype=surr_dtype)
            s_gout_g = torch.empty(tot_in, device=surr_dev, dtype=surr_dtype)
            s_nf = s_packed_g[:n1].view(nf_shape).detach().requires_grad_(True)
            s_ef = s_packed_g[n1 : n1 + n2].view(ef_shape).detach().requires_grad_(True)
            s_sf = s_packed_g[n1 + n2 : n1 + n2 + n3].view(sf_shape).detach().requires_grad_(True)
            s_gf = s_packed_g[n1 + n2 + n3 :].view(gf_shape).detach().requires_grad_(True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with torch.cuda.stream(stream):
                    with torch.enable_grad():  # type: ignore[no-untyped-call]
                        for _ in range(3):
                            wdh, wdc, wdp = m_copy.forward_residuals(
                                s_nf, s_ef, s_am_g, s_sf, s_sm_g, s_gf, require_second_order_sdpa=False
                            )
                            w_hs, w_cs, w_ps = wdh[:b_tot, :m_rem], wdc[:b_tot, :m_rem], wdp[:b_tot]
                            torch.autograd.grad(
                                (w_hs, w_cs, w_ps),
                                (s_nf, s_ef, s_sf, s_gf),
                                grad_outputs=(
                                    s_gin_g[:nh].view(b_tot, m_rem),
                                    s_gin_g[nh : 2 * nh].view(b_tot, m_rem),
                                    s_gin_g[2 * nh :].view(b_tot),
                                ),
                                allow_unused=True,
                            )
                stream.synchronize()

                g_fwd = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
                with torch.cuda.graph(g_fwd, stream=stream):
                    with torch.enable_grad():  # type: ignore[no-untyped-call]
                        fdh, fdc, fdp = m_copy.forward_residuals(
                            s_nf, s_ef, s_am_g, s_sf, s_sm_g, s_gf, require_second_order_sdpa=False
                        )
                        f_hs = fdh[:b_tot, :m_rem]
                        f_cs = fdc[:b_tot, :m_rem]
                        f_ps = fdp[:b_tot]
                        s_out_g[:nh].copy_(f_hs.detach().reshape(-1))
                        s_out_g[nh : 2 * nh].copy_(f_cs.detach().reshape(-1))
                        s_out_g[2 * nh :].copy_(f_ps.detach().reshape(-1))

                g_bwd = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
                with torch.cuda.graph(g_bwd, pool=g_fwd.pool(), stream=stream):  # type: ignore[no-untyped-call]
                    with torch.enable_grad():  # type: ignore[no-untyped-call]
                        b_gs = torch.autograd.grad(
                            (f_hs, f_cs, f_ps),
                            (s_nf, s_ef, s_sf, s_gf),
                            grad_outputs=(
                                s_gin_g[:nh].view(b_tot, m_rem),
                                s_gin_g[nh : 2 * nh].view(b_tot, m_rem),
                                s_gin_g[2 * nh :].view(b_tot),
                            ),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        g_parts = [
                            (g.reshape(-1) if g is not None else torch.zeros_like(inp).reshape(-1))
                            for g, inp in zip(b_gs, (s_nf, s_ef, s_sf, s_gf))
                        ]
                        s_gout_g.copy_(torch.cat(g_parts, dim=0))
                stream.synchronize()
            keep_alive = (m_copy, s_am_g, s_sm_g, s_nf, s_ef, s_sf, s_gf, f_hs, f_cs, f_ps)
            runner = (stream, g_fwd, g_bwd, s_packed_g, s_out_g, s_gin_g, s_gout_g, nh, b_tot, m_rem, keep_alive)
            _SURROGATE_CUDAGRAPH_REGISTRY.put(key, runner)
            return runner
    except Exception:
        return None


class _PackedCudaSurrogateFn(torch.autograd.Function):
    """Crosses CPU(float64) -> CUDA(float32) -> CPU(float64) in ONE contiguous transfer pair."""

    @staticmethod
    def forward(
        packed_cpu_f64: torch.Tensor,
        surrogate: MaskedEQAutocallSurrogate,
        nf_shape: tuple[int, ...],
        ef_shape: tuple[int, ...],
        sf_shape: tuple[int, ...],
        gf_shape: tuple[int, ...],
        am_g: torch.Tensor,
        sm_g: torch.Tensor,
        surr_dev: torch.device,
        surr_dtype: torch.dtype,
        m_rem: int,
        box: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b_tot = int(nf_shape[0])
        need_g = bool(packed_cpu_f64.requires_grad)
        cg_runner = _get_surrogate_cuda_graph_runner(
            surrogate, nf_shape, ef_shape, sf_shape, gf_shape, surr_dev, surr_dtype, m_rem
        )
        if cg_runner is not None:
            stream, g_fwd, g_bwd, s_packed_g, s_out_g, s_gin_g, s_gout_g, nh, _, _, _ = cg_runner
            with torch.cuda.stream(stream):
                s_packed_g.copy_(packed_cpu_f64.detach().to(dtype=surr_dtype), non_blocking=False)
                g_fwd.replay()
                out_cpu = s_out_g.to(device=packed_cpu_f64.device, dtype=torch.float64)
            box.append(("cg", cg_runner) if need_g else None)
            return (
                out_cpu[:nh].view(b_tot, m_rem),
                out_cpu[nh : 2 * nh].view(b_tot, m_rem),
                out_cpu[2 * nh :].view(b_tot),
            )

        n1 = int(nf_shape[0] * nf_shape[1] * nf_shape[2])
        n2 = int(ef_shape[0] * ef_shape[1] * ef_shape[2] * ef_shape[3])
        n3 = int(sf_shape[0] * sf_shape[1] * sf_shape[2])
        packed_g = packed_cpu_f64.detach().to(dtype=surr_dtype).to(device=surr_dev)
        if need_g:
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                nf_g = packed_g[:n1].view(nf_shape).detach().requires_grad_(True)
                ef_g = packed_g[n1 : n1 + n2].view(ef_shape).detach().requires_grad_(True)
                sf_g = packed_g[n1 + n2 : n1 + n2 + n3].view(sf_shape).detach().requires_grad_(True)
                gf_g = packed_g[n1 + n2 + n3 :].view(gf_shape).detach().requires_grad_(True)
                d_haz_g, d_cpn_g, d_put_g = _call_surrogate_residuals_fast(
                    surrogate, nf_g, ef_g, am_g, sf_g, sm_g, gf_g, require_second_order_sdpa=False
                )
                haz_s = d_haz_g[:b_tot, :m_rem]
                cpn_s = d_cpn_g[:b_tot, :m_rem]
                put_s = d_put_g[:b_tot]
            box.append(("eager", nf_g, ef_g, sf_g, gf_g, haz_s, cpn_s, put_s))
        else:
            nf_g = packed_g[:n1].view(nf_shape)
            ef_g = packed_g[n1 : n1 + n2].view(ef_shape)
            sf_g = packed_g[n1 + n2 : n1 + n2 + n3].view(sf_shape)
            gf_g = packed_g[n1 + n2 + n3 :].view(gf_shape)
            d_haz_g, d_cpn_g, d_put_g = _call_surrogate_residuals_fast(
                surrogate, nf_g, ef_g, am_g, sf_g, sm_g, gf_g, require_second_order_sdpa=False
            )
            haz_s = d_haz_g[:b_tot, :m_rem]
            cpn_s = d_cpn_g[:b_tot, :m_rem]
            put_s = d_put_g[:b_tot]
            box.append(None)
        out_g = torch.cat(
            [haz_s.detach().reshape(-1), cpn_s.detach().reshape(-1), put_s.detach().reshape(-1)], dim=0
        )
        out_cpu = out_g.to(device=packed_cpu_f64.device, dtype=torch.float64)
        nh = int(haz_s.numel())
        return (
            out_cpu[:nh].view(b_tot, m_rem),
            out_cpu[nh : 2 * nh].view(b_tot, m_rem),
            out_cpu[2 * nh :].view(b_tot),
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        packed_cpu_f64, surrogate, nf_shape, ef_shape, sf_shape, gf_shape, am_g, sm_g, surr_dev, surr_dtype, m_rem, box = inputs
        ctx.gpu_state = box.pop()
        ctx.meta = (
            packed_cpu_f64.detach(),
            surrogate,
            nf_shape,
            ef_shape,
            sf_shape,
            gf_shape,
            am_g,
            sm_g,
            surr_dev,
            surr_dtype,
            m_rem,
            packed_cpu_f64.device,
        )

    @staticmethod
    def backward(ctx: Any, g_haz: torch.Tensor, g_cpn: torch.Tensor, g_put: torch.Tensor) -> tuple[Any, ...]:
        gpu_state = ctx.gpu_state
        ctx.gpu_state = None
        if gpu_state is None or not bool(ctx.needs_input_grad[0]):
            return (None,) * 12
        cpu_dev = ctx.meta[11]
        surr_dtype = ctx.meta[9]
        if gpu_state[0] == "cg":
            stream, _, g_bwd, _, _, s_gin_g, s_gout_g, _, _, _, _ = gpu_state[1]
            g_cat_cpu = torch.cat(
                (g_haz.reshape(-1), g_cpn.reshape(-1), g_put.reshape(-1)), dim=0
            ).to(dtype=surr_dtype)
            with torch.cuda.stream(stream):
                s_gin_g.copy_(g_cat_cpu, non_blocking=False)
                g_bwd.replay()
                g_packed_cpu = s_gout_g.to(device=cpu_dev, dtype=torch.float64)
            return (g_packed_cpu,) + (None,) * 11
        _, nf_g, ef_g, sf_g, gf_g, haz_s, cpn_s, put_s = gpu_state
        surr_dev = nf_g.device
        surr_dtype = nf_g.dtype
        g_cat_g = torch.cat(
            [g_haz.reshape(-1), g_cpn.reshape(-1), g_put.reshape(-1)], dim=0
        ).to(dtype=surr_dtype).to(device=surr_dev)
        nh = int(haz_s.numel())
        gh_g = g_cat_g[:nh].view(haz_s.shape)
        gc_g = g_cat_g[nh : 2 * nh].view(cpn_s.shape)
        gp_g = g_cat_g[2 * nh :].view(put_s.shape)
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            grads_g = torch.autograd.grad(
                (haz_s, cpn_s, put_s),
                (nf_g, ef_g, sf_g, gf_g),
                grad_outputs=(gh_g, gc_g, gp_g),
                allow_unused=True,
            )
        g_parts = [
            (g.reshape(-1) if g is not None else torch.zeros_like(inp).reshape(-1))
            for g, inp in zip(grads_g, (nf_g, ef_g, sf_g, gf_g))
        ]
        g_packed_cpu = torch.cat(g_parts, dim=0).to(device=cpu_dev, dtype=torch.float64)
        return (g_packed_cpu,) + (None,) * 11

    @staticmethod
    def jvp(ctx: Any, d_packed: torch.Tensor, *_: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed_cpu_f64, surrogate, nf_shape, ef_shape, sf_shape, gf_shape, am_g, sm_g, surr_dev, surr_dtype, m_rem, cpu_dev = ctx.meta
        b_tot = int(nf_shape[0])
        if d_packed is None:
            return (
                torch.zeros((b_tot, m_rem), dtype=torch.float64, device=cpu_dev),
                torch.zeros((b_tot, m_rem), dtype=torch.float64, device=cpu_dev),
                torch.zeros((b_tot,), dtype=torch.float64, device=cpu_dev),
            )
        h = 1.0e-4
        n1 = int(nf_shape[0] * nf_shape[1] * nf_shape[2])
        n2 = int(ef_shape[0] * ef_shape[1] * ef_shape[2] * ef_shape[3])
        n3 = int(sf_shape[0] * sf_shape[1] * sf_shape[2])

        def _eval_at(p_cpu: torch.Tensor) -> torch.Tensor:
            pg = p_cpu.detach().to(dtype=surr_dtype).to(device=surr_dev)
            dh, dc, dp = _call_surrogate_residuals_fast(
                surrogate,
                pg[:n1].view(nf_shape),
                pg[n1 : n1 + n2].view(ef_shape),
                am_g,
                pg[n1 + n2 : n1 + n2 + n3].view(sf_shape),
                sm_g,
                pg[n1 + n2 + n3 :].view(gf_shape),
                require_second_order_sdpa=False,
            )
            return torch.cat(
                [dh[:b_tot, :m_rem].reshape(-1), dc[:b_tot, :m_rem].reshape(-1), dp[:b_tot].reshape(-1)], dim=0
            ).to(device=cpu_dev, dtype=torch.float64)

        diff = (_eval_at(packed_cpu_f64 + h * d_packed) - _eval_at(packed_cpu_f64 - h * d_packed)) / (2.0 * h)
        nh = b_tot * m_rem
        return diff[:nh].view(b_tot, m_rem), diff[nh : 2 * nh].view(b_tot, m_rem), diff[2 * nh :].view(b_tot)


class _SurrFeaturesCppFn(torch.autograd.Function):
    """Fused C++23 float64 surrogate feature builder with exact analytical reverse-mode AD & JVP."""

    @staticmethod
    def forward(
        m_b: torch.Tensor,
        fr_b: torch.Tensor,
        fp_b: torch.Tensor,
        cd_b: torch.Tensor,
        vp_b: torch.Tensor,
        iv_b: torch.Tensor,
        dw_in: torch.Tensor,
        cm_in: torch.Tensor,
        dfp_b: torch.Tensor,
        fw_in: torch.Tensor,
        vw_in: torch.Tensor,
        p_in: torch.Tensor,
        q_in: torch.Tensor,
        lp_in: torch.Tensor,
        r_spr: torch.Tensor,
        static_np: tuple[Any, ...],
        np_box: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sf_fn = _SURR_FEAT_CPP_FN
        assert sf_fn is not None
        tau_np, dtau_np, lko_np, ic_np, lcp_np, ig_np, cpn_np, ra_np, rb_np, gst_np, d_node, d_global, k_eff_n = static_np
        dev = m_b.device
        b_sz, d_cnt = int(m_b.shape[0]), int(m_b.shape[1])
        m_cnt = int(fr_b.shape[2])
        q_len = int(iv_b.shape[2])

        in_nps = tuple(
            np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float64)
            for t in (m_b, fr_b, fp_b, cd_b, vp_b, iv_b, dw_in, cm_in, dfp_b, fw_in, vw_in, p_in, q_in, lp_in, r_spr)
        )
        nf_np = np.empty((b_sz, d_cnt, d_node), dtype=np.float64)
        ef_np = np.empty((b_sz, d_cnt, d_cnt, 4), dtype=np.float64)
        sf_np = np.empty((b_sz, m_cnt, 12), dtype=np.float64)
        gf_np = np.empty((b_sz, d_global), dtype=np.float64)

        sf_fn(
            b_sz, d_cnt, m_cnt, q_len, d_node, d_global, 0, float(k_eff_n),
            *(a.ctypes.data for a in in_nps),
            tau_np.ctypes.data, dtau_np.ctypes.data, lko_np.ctypes.data, ic_np.ctypes.data,
            lcp_np.ctypes.data, ig_np.ctypes.data, cpn_np.ctypes.data, ra_np.ctypes.data,
            rb_np.ctypes.data, gst_np.ctypes.data,
            nf_np.ctypes.data, ef_np.ctypes.data, sf_np.ctypes.data, gf_np.ctypes.data,
            0, 0, 0, 0,
            *([0] * 15),
        )
        np_box.append((in_nps, static_np, b_sz, d_cnt, m_cnt, q_len, dev))
        return (
            torch.from_numpy(nf_np).to(device=dev),
            torch.from_numpy(ef_np).to(device=dev),
            torch.from_numpy(sf_np).to(device=dev),
            torch.from_numpy(gf_np).to(device=dev),
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        ctx.saved_meta = inputs[16].pop()

    @staticmethod
    def backward(
        ctx: Any, g_nf: torch.Tensor, g_ef: torch.Tensor, g_sf: torch.Tensor, g_gf: torch.Tensor
    ) -> tuple[Any, ...]:
        sf_fn = _SURR_FEAT_CPP_FN
        assert sf_fn is not None
        in_nps, static_np, b_sz, d_cnt, m_cnt, q_len, dev = ctx.saved_meta
        tau_np, dtau_np, lko_np, ic_np, lcp_np, ig_np, cpn_np, ra_np, rb_np, gst_np, d_node, d_global, k_eff_n = static_np
        gn = np.ascontiguousarray(g_nf.detach().cpu().numpy(), dtype=np.float64) if g_nf is not None else np.zeros((b_sz, d_cnt, d_node), dtype=np.float64)
        ge = np.ascontiguousarray(g_ef.detach().cpu().numpy(), dtype=np.float64) if g_ef is not None else np.zeros((b_sz, d_cnt, d_cnt, 4), dtype=np.float64)
        gs = np.ascontiguousarray(g_sf.detach().cpu().numpy(), dtype=np.float64) if g_sf is not None else np.zeros((b_sz, m_cnt, 12), dtype=np.float64)
        gg = np.ascontiguousarray(g_gf.detach().cpu().numpy(), dtype=np.float64) if g_gf is not None else np.zeros((b_sz, d_global), dtype=np.float64)
        g_outs = tuple(np.zeros_like(a) for a in in_nps)
        sf_fn(
            b_sz, d_cnt, m_cnt, q_len, d_node, d_global, 1, float(k_eff_n),
            *(a.ctypes.data for a in in_nps),
            tau_np.ctypes.data, dtau_np.ctypes.data, lko_np.ctypes.data, ic_np.ctypes.data,
            lcp_np.ctypes.data, ig_np.ctypes.data, cpn_np.ctypes.data, ra_np.ctypes.data,
            rb_np.ctypes.data, gst_np.ctypes.data,
            0, 0, 0, 0,
            gn.ctypes.data, ge.ctypes.data, gs.ctypes.data, gg.ctypes.data,
            *(g.ctypes.data for g in g_outs),
        )
        return tuple(
            torch.from_numpy(g).to(device=dev) if need else None
            for g, need in zip(g_outs, ctx.needs_input_grad[:15])
        ) + (None, None)

    @staticmethod
    def jvp(ctx: Any, *d_inputs: Any) -> tuple[torch.Tensor, ...]:
        sf_fn = _SURR_FEAT_CPP_FN
        assert sf_fn is not None
        in_nps, static_np, b_sz, d_cnt, m_cnt, q_len, dev = ctx.saved_meta
        tau_np, dtau_np, lko_np, ic_np, lcp_np, ig_np, cpn_np, ra_np, rb_np, gst_np, d_node, d_global, k_eff_n = static_np
        h = 1.0e-7
        d_nps = tuple(
            np.ascontiguousarray(dt.detach().cpu().numpy(), dtype=np.float64) if dt is not None else np.zeros_like(base)
            for dt, base in zip(d_inputs[:15], in_nps)
        )

        def _eval(sign: float) -> tuple[npt.NDArray[np.float64], ...]:
            nf_np = np.empty((b_sz, d_cnt, d_node), dtype=np.float64)
            ef_np = np.empty((b_sz, d_cnt, d_cnt, 4), dtype=np.float64)
            sf_np = np.empty((b_sz, m_cnt, 12), dtype=np.float64)
            gf_np = np.empty((b_sz, d_global), dtype=np.float64)
            pts = tuple(a + (sign * h) * da for a, da in zip(in_nps, d_nps))
            sf_fn(
                b_sz, d_cnt, m_cnt, q_len, d_node, d_global, 0, float(k_eff_n),
                *(a.ctypes.data for a in pts),
                tau_np.ctypes.data, dtau_np.ctypes.data, lko_np.ctypes.data, ic_np.ctypes.data,
                lcp_np.ctypes.data, ig_np.ctypes.data, cpn_np.ctypes.data, ra_np.ctypes.data,
                rb_np.ctypes.data, gst_np.ctypes.data,
                nf_np.ctypes.data, ef_np.ctypes.data, sf_np.ctypes.data, gf_np.ctypes.data,
                0, 0, 0, 0,
                *([0] * 15),
            )
            return nf_np, ef_np, sf_np, gf_np

        op, om = _eval(1.0), _eval(-1.0)
        return tuple(torch.from_numpy((a - b) / (2.0 * h)).to(device=dev) for a, b in zip(op, om))


class _SurrPostPvCppFn(torch.autograd.Function):
    """Fused C++23 float64 post-surrogate stopping/coupon/put + PV assembler with exact reverse AD."""

    @staticmethod
    def forward(
        d_haz: torch.Tensor,
        d_cpn: torch.Tensor,
        d_put: torch.Tensor,
        p_px: torch.Tensor,
        q_px: torch.Tensor,
        lp_px: torch.Tensor,
        l_call: torch.Tensor,
        l_onestar: torch.Tensor,
        df_pay: torch.Tensor,
        w_fund: torch.Tensor | None,
        pv_in_tr: torch.Tensor,
        static_meta: tuple[Any, ...],
        np_box: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sp_fn = _SURR_POST_CPP_FN
        assert sp_fn is not None
        (
            ic_np, ig_np, cpn_np, has_surr, no_notional, force_cpn, is_mem, is_long,
            c_mem_0, ac_cpn, opt_gear, part_rate, k_eff_put,
        ) = static_meta
        dev = p_px.device
        b_sz, m_cnt = int(p_px.shape[0]), int(p_px.shape[1])
        has_wf = 1 if w_fund is not None else 0
        wf_in = w_fund if w_fund is not None else torch.zeros((b_sz, m_cnt), dtype=torch.float64, device=dev)
        in_nps = tuple(
            np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float64)
            for t in (d_haz, d_cpn, d_put, p_px, q_px, lp_px, l_call, l_onestar, df_pay, wf_in, pv_in_tr)
        )
        pv_np = np.empty((b_sz,), dtype=np.float64)
        ps0_np = np.empty((m_cnt,), dtype=np.float64)
        qc0_np = np.empty((m_cnt,), dtype=np.float64)
        lp0_np = np.empty((1,), dtype=np.float64)
        sp_fn(
            b_sz, m_cnt, has_surr, has_wf, no_notional, force_cpn, is_mem, is_long, 0,
            float(c_mem_0), float(ac_cpn), float(opt_gear), float(part_rate), float(k_eff_put),
            *(a.ctypes.data for a in in_nps),
            ic_np.ctypes.data, ig_np.ctypes.data, cpn_np.ctypes.data,
            pv_np.ctypes.data, ps0_np.ctypes.data, qc0_np.ctypes.data, lp0_np.ctypes.data,
            0, *([0] * 11),
        )
        np_box.append((in_nps, static_meta, b_sz, m_cnt, has_wf, dev))
        return (
            torch.from_numpy(pv_np).to(device=dev),
            torch.from_numpy(ps0_np).to(device=dev),
            torch.from_numpy(qc0_np).to(device=dev),
            torch.tensor(float(lp0_np[0]), dtype=torch.float64, device=dev),
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        ctx.saved_meta = inputs[12].pop()

    @staticmethod
    def backward(ctx: Any, g_pv: torch.Tensor, *_: Any) -> tuple[Any, ...]:
        sp_fn = _SURR_POST_CPP_FN
        assert sp_fn is not None
        in_nps, static_meta, b_sz, m_cnt, has_wf, dev = ctx.saved_meta
        (
            ic_np, ig_np, cpn_np, has_surr, no_notional, force_cpn, is_mem, is_long,
            c_mem_0, ac_cpn, opt_gear, part_rate, k_eff_put,
        ) = static_meta
        gpv_np = np.ascontiguousarray(g_pv.detach().cpu().numpy(), dtype=np.float64) if g_pv is not None else np.zeros((b_sz,), dtype=np.float64)
        g_outs = tuple(np.zeros_like(a) for a in in_nps)
        pv_dummy = np.empty((b_sz,), dtype=np.float64)
        ps0_d = np.empty((m_cnt,), dtype=np.float64)
        qc0_d = np.empty((m_cnt,), dtype=np.float64)
        lp0_d = np.empty((1,), dtype=np.float64)
        sp_fn(
            b_sz, m_cnt, has_surr, has_wf, no_notional, force_cpn, is_mem, is_long, 1,
            float(c_mem_0), float(ac_cpn), float(opt_gear), float(part_rate), float(k_eff_put),
            *(a.ctypes.data for a in in_nps),
            ic_np.ctypes.data, ig_np.ctypes.data, cpn_np.ctypes.data,
            pv_dummy.ctypes.data, ps0_d.ctypes.data, qc0_d.ctypes.data, lp0_d.ctypes.data,
            gpv_np.ctypes.data,
            *(g.ctypes.data for g in g_outs),
        )
        res = [
            (torch.from_numpy(g).to(device=dev) if need else None)
            for g, need in zip(g_outs, ctx.needs_input_grad[:11])
        ]
        if not has_wf:
            res[9] = None
        return tuple(res) + (None, None)

    @staticmethod
    def jvp(ctx: Any, *d_inputs: Any) -> tuple[torch.Tensor, ...]:
        sp_fn = _SURR_POST_CPP_FN
        assert sp_fn is not None
        in_nps, static_meta, b_sz, m_cnt, has_wf, dev = ctx.saved_meta
        (
            ic_np, ig_np, cpn_np, has_surr, no_notional, force_cpn, is_mem, is_long,
            c_mem_0, ac_cpn, opt_gear, part_rate, k_eff_put,
        ) = static_meta
        h = 1.0e-7
        d_nps = tuple(
            np.ascontiguousarray(dt.detach().cpu().numpy(), dtype=np.float64) if dt is not None else np.zeros_like(base)
            for dt, base in zip(d_inputs[:11], in_nps)
        )

        def _eval(sign: float) -> tuple[npt.NDArray[np.float64], ...]:
            pv_np = np.empty((b_sz,), dtype=np.float64)
            ps0_np = np.empty((m_cnt,), dtype=np.float64)
            qc0_np = np.empty((m_cnt,), dtype=np.float64)
            lp0_np = np.empty((1,), dtype=np.float64)
            pts = tuple(a + (sign * h) * da for a, da in zip(in_nps, d_nps))
            sp_fn(
                b_sz, m_cnt, has_surr, has_wf, no_notional, force_cpn, is_mem, is_long, 0,
                float(c_mem_0), float(ac_cpn), float(opt_gear), float(part_rate), float(k_eff_put),
                *(a.ctypes.data for a in pts),
                ic_np.ctypes.data, ig_np.ctypes.data, cpn_np.ctypes.data,
                pv_np.ctypes.data, ps0_np.ctypes.data, qc0_np.ctypes.data, lp0_np.ctypes.data,
                0, *([0] * 11),
            )
            return pv_np, ps0_np, qc0_np, lp0_np

        op, om = _eval(1.0), _eval(-1.0)
        return (
            torch.from_numpy((op[0] - om[0]) / (2.0 * h)).to(device=dev),
            torch.from_numpy((op[1] - om[1]) / (2.0 * h)).to(device=dev),
            torch.from_numpy((op[2] - om[2]) / (2.0 * h)).to(device=dev),
            torch.tensor(float((op[3][0] - om[3][0]) / (2.0 * h)), dtype=torch.float64, device=dev),
        )


class _PreSurrFusedFn(torch.autograd.Function):
    """Fused C++23 + NumPy pre-surrogate pipeline (escrowed fwd -> surfaces -> Tier1 -> surr features)."""

    @staticmethod
    def _eval_np(
        sp0_np: npt.NDArray[np.float64],
        spg_np: npt.NDArray[np.float64],
        corr_np: npt.NDArray[np.float64],
        iv0_np: npt.NDArray[np.float64] | None,
        div_del_np: npt.NDArray[np.float64] | None,
        df_pay_m_np: npt.NDArray[np.float64] | None,
        df_fin_m_np: npt.NDArray[np.float64] | None,
        df_repo_m_np: npt.NDArray[np.float64] | None,
        static_ctx: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        (
            ref_spots_np, div_pv_base_np, r_tot_base_np, df_fin_base_np, df_pay_base_np,
            repo_spr_base_np, tau_obs_np, sqrt_tau_np, t_1_q_np, t_1_q_ptr,
            asset_metas_b, tier1_meta, ko_np, cb_np, ig_np, feat_static_np, theta_pack,
        ) = static_ctx
        s_fn = _SURFACE_CPP_AD_FN
        c_fn = _TIER1_CPP_AD_FN
        sf_fn = _SURR_FEAT_CPP_FN
        assert s_fn is not None and c_fn is not None and sf_fn is not None

        b_g = int(spg_np.shape[0])
        b_diff = 1 + b_g
        d_cnt = int(sp0_np.shape[0])
        m_cnt = int(tau_obs_np.shape[0])
        q_len = int(t_1_q_np.shape[0])

        spots_all = np.empty((b_diff, d_cnt), dtype=np.float64)
        spots_all[0] = sp0_np
        if b_g > 0:
            spots_all[1:] = spg_np

        div_pv_0 = div_pv_base_np + div_del_np if div_del_np is not None else div_pv_base_np
        sp_3 = spots_all[:, :, None]
        floor_sp = 0.01 * sp_3
        scale_sp = np.maximum(floor_sp, 1e-12)
        z_sp = (sp_3 - div_pv_0[None, :, :] - floor_sp) / scale_sp
        z_sp_c = np.minimum(z_sp, 40.0)
        sp_z = np.where(z_sp > 40.0, z_sp, np.log1p(np.exp(np.maximum(z_sp_c, -80.0))))
        sig_sp = np.where(z_sp > 40.0, 1.0, 1.0 / (1.0 + np.exp(-np.maximum(z_sp_c, -80.0))))
        escrowed = floor_sp + scale_sp * sp_z

        df_fin_obs = np.maximum(df_fin_base_np * df_fin_m_np if df_fin_m_np is not None else df_fin_base_np, 1e-15)
        df_repo_q = np.exp(r_tot_base_np) * df_fin_base_np
        if df_repo_m_np is not None:
            df_repo_q = df_repo_q * df_repo_m_np
            tau_term = max(float(tau_obs_np[-1]), 1e-6)
            repo_spr = repo_spr_base_np - np.log(np.maximum(df_repo_m_np[:, -1], 1e-15)) / tau_term
        else:
            repo_spr = repo_spr_base_np.copy()

        gf_0 = df_repo_q / df_fin_obs
        fwd_price = escrowed * gf_0[None, :, :]
        fwd_rel = fwd_price / sp_3
        fwd_perf = fwd_price / ref_spots_np[None, :, None]
        cd_lev = sp_3 / escrowed
        moneyness = spots_all / ref_spots_np[None, :]

        df_pay_0 = df_pay_base_np * df_pay_m_np if df_pay_m_np is not None else df_pay_base_np
        df_pay_diff = np.broadcast_to(df_pay_0, (b_diff, m_cnt)).copy()

        f_st = fwd_price[:, :, -1:]
        k_all = np.concatenate([fwd_price, f_st, 0.90 * f_st, 1.10 * f_st, 0.80 * f_st, 1.20 * f_st], axis=2)
        k_flat = np.ascontiguousarray(k_all.transpose(1, 0, 2).reshape(d_cnt, b_diff * q_len), dtype=np.float64)
        iv_flat = np.empty((d_cnt, b_diff * q_len), dtype=np.float64)
        dk_flat = np.empty((d_cnt, b_diff * q_len), dtype=np.float64)
        k_base_ptr = int(k_flat.__array_interface__["data"][0])
        iv_base_ptr = int(iv_flat.__array_interface__["data"][0])
        dk_base_ptr = int(dk_flat.__array_interface__["data"][0])
        t_tiled = np.ascontiguousarray(np.broadcast_to(t_1_q_np, (b_diff, q_len)), dtype=np.float64)
        t_tiled_ptr = int(t_tiled.__array_interface__["data"][0])
        row_bytes = b_diff * q_len * 8
        for i in range(d_cnt):
            surf_i, geom, qcache = asset_metas_b[i]
            off = i * row_bytes
            s_fn(
                b_diff * q_len, geom.n_unique, geom.n_exp,
                geom.k_min, geom.k_max, float(surf_i.min_vol),
                geom.strikes_ptr, geom.tenors_ptr, qcache.quotes_ptr,
                qcache.slice_c_ptr, qcache.slope_left_ptr, qcache.slope_right_ptr,
                t_tiled_ptr, k_base_ptr + off,
                iv_base_ptr + off, dk_base_ptr + off,
            )
        iv_all = np.ascontiguousarray(iv_flat.reshape(d_cnt, b_diff, q_len).transpose(1, 0, 2), dtype=np.float64)
        dk_all = np.ascontiguousarray(dk_flat.reshape(d_cnt, b_diff, q_len).transpose(1, 0, 2), dtype=np.float64)
        if iv0_np is not None:
            iv_all[0] += iv0_np
        v_perf = np.ascontiguousarray(iv_all[:, :, :m_cnt] * sqrt_tau_np[None, None, :], dtype=np.float64)

        eye_d = np.eye(d_cnt, dtype=np.float64)
        corr_raw_sym = 0.5 * (corr_np + corr_np.T) * (1.0 - eye_d) + eye_d
        abs_c = np.abs(corr_raw_sym)
        thr, span = 0.999, 0.0009
        th_c = np.tanh((abs_c - thr) / span)
        guarded = np.sign(corr_raw_sym) * (thr + span * th_c)
        corr_sym = np.where(abs_c <= thr, corr_raw_sym, guarded) * (1.0 - eye_d) + eye_d
        corr_b = np.ascontiguousarray(np.broadcast_to(corr_sym, (b_diff, d_cnt, d_cnt)), dtype=np.float64)

        call_code = int(tier1_meta[0])
        cpn_code = int(tier1_meta[1])
        opt_code = int(tier1_meta[2])
        f_safe = np.maximum(fwd_perf.transpose(0, 2, 1).reshape(b_diff * m_cnt, d_cnt), 1e-16)
        ord_asc = np.argsort(f_safe, axis=1, kind="mergesort") if d_cnt > 1 else np.zeros((b_diff * m_cnt, 1), dtype=np.int64)
        ord_desc = np.ascontiguousarray(ord_asc[:, ::-1], dtype=np.int64) if d_cnt > 1 else ord_asc
        ord_call = ord_desc if call_code == 1 else ord_asc
        ord_cpn = ord_desc if cpn_code == 1 else ord_asc
        ord_opt_asc = ord_asc.reshape(b_diff, m_cnt, d_cnt)[:, -1, :]
        ord_opt = np.ascontiguousarray(ord_opt_asc[:, ::-1], dtype=np.int64) if (opt_code == 1 and d_cnt > 1) else np.ascontiguousarray(ord_opt_asc, dtype=np.int64)

        fw_o = np.empty((b_diff, m_cnt), dtype=np.float64)
        vw_o = np.empty((b_diff, m_cnt), dtype=np.float64)
        p_o = np.empty((b_diff, m_cnt), dtype=np.float64)
        q_o = np.empty((b_diff, m_cnt), dtype=np.float64)
        lp_o = np.empty((b_diff,), dtype=np.float64)
        dw_o = np.empty((b_diff, d_cnt), dtype=np.float64)
        ic_arr = tier1_meta[-1]
        c_fn(
            b_diff, d_cnt, m_cnt, 0, 0, 0,
            fwd_perf.__array_interface__["data"][0], v_perf.__array_interface__["data"][0], corr_b.__array_interface__["data"][0],
            ko_np.__array_interface__["data"][0], ic_arr.__array_interface__["data"][0],
            cb_np.__array_interface__["data"][0], ig_np.__array_interface__["data"][0],
            ord_call.__array_interface__["data"][0], ord_cpn.__array_interface__["data"][0], ord_opt.__array_interface__["data"][0],
            *tier1_meta[:-1],
            fw_o.__array_interface__["data"][0], vw_o.__array_interface__["data"][0], p_o.__array_interface__["data"][0],
            q_o.__array_interface__["data"][0], lp_o.__array_interface__["data"][0], dw_o.__array_interface__["data"][0],
            0, 0, 0, 0, 0, 0, 0, 0, 0,
        )

        tau_np_s, dtau_np_s, lko_np_s, ic_np_s, lcp_np_s, ig_np_s, cpn_np_s, ra_np, rb_np, gst_np, d_node, d_global, k_eff_n = feat_static_np
        has_th = 1 if theta_pack is not None else 0
        b_tot = b_diff + has_th
        nf_np = np.empty((b_tot, d_cnt, d_node), dtype=np.float64)
        ef_np = np.empty((b_tot, d_cnt, d_cnt, 4), dtype=np.float64)
        sf_np = np.empty((b_tot, m_cnt, 12), dtype=np.float64)
        gf_np = np.empty((b_tot, d_global), dtype=np.float64)
        sf_fn(
            b_diff, d_cnt, m_cnt, q_len, d_node, d_global, 0, float(k_eff_n),
            moneyness.__array_interface__["data"][0], fwd_rel.__array_interface__["data"][0],
            fwd_perf.__array_interface__["data"][0], cd_lev.__array_interface__["data"][0],
            v_perf.__array_interface__["data"][0], iv_all.__array_interface__["data"][0],
            dw_o.__array_interface__["data"][0], corr_b.__array_interface__["data"][0],
            df_pay_diff.__array_interface__["data"][0], fw_o.__array_interface__["data"][0],
            vw_o.__array_interface__["data"][0], p_o.__array_interface__["data"][0],
            q_o.__array_interface__["data"][0], lp_o.__array_interface__["data"][0],
            repo_spr.__array_interface__["data"][0],
            tau_np_s.__array_interface__["data"][0], dtau_np_s.__array_interface__["data"][0],
            lko_np_s.__array_interface__["data"][0], ic_np_s.__array_interface__["data"][0],
            lcp_np_s.__array_interface__["data"][0], ig_np_s.__array_interface__["data"][0],
            cpn_np_s.__array_interface__["data"][0], ra_np.__array_interface__["data"][0],
            rb_np.__array_interface__["data"][0], gst_np.__array_interface__["data"][0],
            nf_np.__array_interface__["data"][0], ef_np.__array_interface__["data"][0],
            sf_np.__array_interface__["data"][0], gf_np.__array_interface__["data"][0],
            0, 0, 0, 0,
            *([0] * 15),
        )
        if theta_pack is not None:
            (
                m0_th, fpr_th, fpp_th, cdl_th, vp_th, iv_th, dw_th,
                dfp_th, fw_th, vw_th, p_th, q_th, lp_th, tau_th, dtau_th,
            ) = theta_pack
            sf_fn(
                1, d_cnt, m_cnt, q_len, d_node, d_global, 0, float(k_eff_n),
                m0_th.__array_interface__["data"][0], fpr_th.__array_interface__["data"][0],
                fpp_th.__array_interface__["data"][0], cdl_th.__array_interface__["data"][0],
                vp_th.__array_interface__["data"][0], iv_th.__array_interface__["data"][0],
                dw_th.__array_interface__["data"][0], corr_sym.__array_interface__["data"][0],
                dfp_th.__array_interface__["data"][0], fw_th.__array_interface__["data"][0],
                vw_th.__array_interface__["data"][0], p_th.__array_interface__["data"][0],
                q_th.__array_interface__["data"][0], lp_th.__array_interface__["data"][0],
                repo_spr_base_np.__array_interface__["data"][0],
                tau_th.__array_interface__["data"][0], dtau_th.__array_interface__["data"][0],
                lko_np_s.__array_interface__["data"][0], ic_np_s.__array_interface__["data"][0],
                lcp_np_s.__array_interface__["data"][0], ig_np_s.__array_interface__["data"][0],
                cpn_np_s.__array_interface__["data"][0], ra_np.__array_interface__["data"][0],
                rb_np.__array_interface__["data"][0], gst_np.__array_interface__["data"][0],
                nf_np[b_diff].__array_interface__["data"][0], ef_np[b_diff].__array_interface__["data"][0],
                sf_np[b_diff].__array_interface__["data"][0], gf_np[b_diff].__array_interface__["data"][0],
                0, 0, 0, 0,
                *([0] * 15),
            )

        saved = (
            b_diff, d_cnt, m_cnt, q_len, d_node, d_global, has_th,
            spots_all, div_pv_0, escrowed, sp_z, sig_sp, df_fin_obs, gf_0, df_pay_base_np, df_fin_base_np,
            df_fin_m_np, df_repo_m_np, moneyness, fwd_rel, fwd_perf, fwd_price, cd_lev, v_perf, iv_all, dk_all,
            dw_o, corr_b, abs_c, th_c, eye_d, df_pay_diff, fw_o, vw_o, p_o, q_o, lp_o, repo_spr,
            ord_call, ord_cpn, ord_opt, k_all[0],
        )
        return nf_np, ef_np, sf_np, gf_np, p_o, q_o, lp_o, df_pay_diff, saved

    @staticmethod
    def forward(
        spots_var: torch.Tensor,
        spots_gamma: torch.Tensor,
        corr_var: torch.Tensor,
        iv_0_del: torch.Tensor | None,
        div_del: torch.Tensor | None,
        df_pay_m: torch.Tensor | None,
        df_fin_m: torch.Tensor | None,
        df_repo_m: torch.Tensor | None,
        static_ctx: tuple[Any, ...],
        box: list[Any],
        k0_box: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = spots_var.device
        sp0_np = np.ascontiguousarray(spots_var.detach().cpu().numpy(), dtype=np.float64)
        spg_np = np.ascontiguousarray(spots_gamma.detach().cpu().numpy(), dtype=np.float64)
        corr_np = np.ascontiguousarray(corr_var.detach().cpu().numpy(), dtype=np.float64)
        iv0_np = np.ascontiguousarray(iv_0_del.detach().cpu().numpy(), dtype=np.float64) if iv_0_del is not None else None
        div_np = np.ascontiguousarray(div_del.detach().cpu().numpy(), dtype=np.float64) if div_del is not None else None
        dfp_np = np.ascontiguousarray(df_pay_m.detach().cpu().numpy(), dtype=np.float64) if df_pay_m is not None else None
        dff_np = np.ascontiguousarray(df_fin_m.detach().cpu().numpy(), dtype=np.float64) if df_fin_m is not None else None
        dfr_np = np.ascontiguousarray(df_repo_m.detach().cpu().numpy(), dtype=np.float64) if df_repo_m is not None else None
        nf_np, ef_np, sf_np, gf_np, p_o, q_o, lp_o, dfp_diff, saved = _PreSurrFusedFn._eval_np(
            sp0_np, spg_np, corr_np, iv0_np, div_np, dfp_np, dff_np, dfr_np, static_ctx
        )
        box.append((saved, static_ctx, dev, sp0_np, spg_np, corr_np, iv0_np, div_np, dfp_np, dff_np, dfr_np))
        k0_box.append(saved[-1])
        return (
            torch.from_numpy(nf_np).to(device=dev),
            torch.from_numpy(ef_np).to(device=dev),
            torch.from_numpy(sf_np).to(device=dev),
            torch.from_numpy(gf_np).to(device=dev),
            torch.from_numpy(p_o).to(device=dev),
            torch.from_numpy(q_o).to(device=dev),
            torch.from_numpy(lp_o).to(device=dev),
            torch.from_numpy(dfp_diff).to(device=dev),
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del output
        ctx.saved_meta = inputs[9].pop()

    @staticmethod
    def backward(
        ctx: Any,
        g_nf: torch.Tensor | None,
        g_ef: torch.Tensor | None,
        g_sf: torch.Tensor | None,
        g_gf: torch.Tensor | None,
        g_p_in: torch.Tensor | None,
        g_q_in: torch.Tensor | None,
        g_lp_in: torch.Tensor | None,
        g_dfp_in: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        saved, static_ctx, dev, *_ = ctx.saved_meta
        (
            b_diff, d_cnt, m_cnt, q_len, d_node, d_global, has_th,
            spots_all, div_pv_0, escrowed, sp_z, sig_sp, df_fin_obs, gf_0, df_pay_base_np, df_fin_base_np,
            df_fin_m_np, df_repo_m_np, moneyness, fwd_rel, fwd_perf, fwd_price, cd_lev, v_perf, iv_all, dk_all,
            dw_o, corr_b, abs_c, th_c, eye_d, df_pay_diff, fw_o, vw_o, p_o, q_o, lp_o, repo_spr,
            ord_call, ord_cpn, ord_opt, _,
        ) = saved
        (
            ref_spots_np, _, _, _, _, _, tau_obs_np, sqrt_tau_np, _, _,
            _, tier1_meta, ko_np, cb_np, ig_np, feat_static_np, _,
        ) = static_ctx
        tau_np_s, dtau_np_s, lko_np_s, ic_np_s, lcp_np_s, ig_np_s, cpn_np_s, ra_np, rb_np, gst_np, _, _, k_eff_n = feat_static_np

        c_fn = _TIER1_CPP_AD_FN
        sf_fn = _SURR_FEAT_CPP_FN
        assert c_fn is not None and sf_fn is not None

        gn = np.ascontiguousarray(g_nf[:b_diff].detach().cpu().numpy(), dtype=np.float64) if g_nf is not None else np.zeros((b_diff, d_cnt, d_node), dtype=np.float64)
        ge_n = np.ascontiguousarray(g_ef[:b_diff].detach().cpu().numpy(), dtype=np.float64) if g_ef is not None else np.zeros((b_diff, d_cnt, d_cnt, 4), dtype=np.float64)
        gs = np.ascontiguousarray(g_sf[:b_diff].detach().cpu().numpy(), dtype=np.float64) if g_sf is not None else np.zeros((b_diff, m_cnt, 12), dtype=np.float64)
        gg = np.ascontiguousarray(g_gf[:b_diff].detach().cpu().numpy(), dtype=np.float64) if g_gf is not None else np.zeros((b_diff, d_global), dtype=np.float64)

        g_m = np.zeros_like(moneyness)
        g_fr = np.zeros_like(fwd_rel)
        g_fp = np.zeros_like(fwd_perf)
        g_cd = np.zeros_like(cd_lev)
        g_vp = np.zeros_like(v_perf)
        g_iv = np.zeros_like(iv_all)
        g_dw = np.zeros_like(dw_o)
        g_cm = np.zeros_like(corr_b)
        g_dfp = np.zeros_like(df_pay_diff)
        g_fw = np.zeros_like(fw_o)
        g_vw = np.zeros_like(vw_o)
        g_p = np.zeros_like(p_o)
        g_q = np.zeros_like(q_o)
        g_lp = np.zeros_like(lp_o)
        g_rspr = np.zeros_like(repo_spr)

        sf_fn(
            b_diff, d_cnt, m_cnt, q_len, d_node, d_global, 1, float(k_eff_n),
            moneyness.__array_interface__["data"][0], fwd_rel.__array_interface__["data"][0],
            fwd_perf.__array_interface__["data"][0], cd_lev.__array_interface__["data"][0],
            v_perf.__array_interface__["data"][0], iv_all.__array_interface__["data"][0],
            dw_o.__array_interface__["data"][0], corr_b.__array_interface__["data"][0],
            df_pay_diff.__array_interface__["data"][0], fw_o.__array_interface__["data"][0],
            vw_o.__array_interface__["data"][0], p_o.__array_interface__["data"][0],
            q_o.__array_interface__["data"][0], lp_o.__array_interface__["data"][0],
            repo_spr.__array_interface__["data"][0],
            tau_np_s.__array_interface__["data"][0], dtau_np_s.__array_interface__["data"][0],
            lko_np_s.__array_interface__["data"][0], ic_np_s.__array_interface__["data"][0],
            lcp_np_s.__array_interface__["data"][0], ig_np_s.__array_interface__["data"][0],
            cpn_np_s.__array_interface__["data"][0], ra_np.__array_interface__["data"][0],
            rb_np.__array_interface__["data"][0], gst_np.__array_interface__["data"][0],
            0, 0, 0, 0,
            gn.__array_interface__["data"][0], ge_n.__array_interface__["data"][0],
            gs.__array_interface__["data"][0], gg.__array_interface__["data"][0],
            g_m.__array_interface__["data"][0], g_fr.__array_interface__["data"][0],
            g_fp.__array_interface__["data"][0], g_cd.__array_interface__["data"][0],
            g_vp.__array_interface__["data"][0], g_iv.__array_interface__["data"][0],
            g_dw.__array_interface__["data"][0], g_cm.__array_interface__["data"][0],
            g_dfp.__array_interface__["data"][0], g_fw.__array_interface__["data"][0],
            g_vw.__array_interface__["data"][0], g_p.__array_interface__["data"][0],
            g_q.__array_interface__["data"][0], g_lp.__array_interface__["data"][0],
            g_rspr.__array_interface__["data"][0],
        )

        if g_p_in is not None:
            g_p += np.ascontiguousarray(g_p_in.detach().cpu().numpy(), dtype=np.float64)
        if g_q_in is not None:
            g_q += np.ascontiguousarray(g_q_in.detach().cpu().numpy(), dtype=np.float64)
        if g_lp_in is not None:
            g_lp += np.ascontiguousarray(g_lp_in.detach().cpu().numpy(), dtype=np.float64)

        need_corr = bool(ctx.needs_input_grad[2])
        g_fp_t1 = np.empty_like(fwd_perf)
        g_vp_t1 = np.empty_like(v_perf)
        g_corr_t1 = np.zeros_like(corr_b) if need_corr else None
        ic_arr = tier1_meta[-1]
        c_fn(
            b_diff, d_cnt, m_cnt, 1, 1, 1 if need_corr else 0,
            fwd_perf.__array_interface__["data"][0], v_perf.__array_interface__["data"][0], corr_b.__array_interface__["data"][0],
            ko_np.__array_interface__["data"][0], ic_arr.__array_interface__["data"][0],
            cb_np.__array_interface__["data"][0], ig_np.__array_interface__["data"][0],
            ord_call.__array_interface__["data"][0], ord_cpn.__array_interface__["data"][0], ord_opt.__array_interface__["data"][0],
            *tier1_meta[:-1],
            0, 0, 0, 0, 0, 0,
            g_fw.__array_interface__["data"][0], g_vw.__array_interface__["data"][0],
            g_p.__array_interface__["data"][0], g_q.__array_interface__["data"][0],
            g_lp.__array_interface__["data"][0], g_dw.__array_interface__["data"][0],
            g_fp_t1.__array_interface__["data"][0], g_vp_t1.__array_interface__["data"][0],
            g_corr_t1.__array_interface__["data"][0] if g_corr_t1 is not None else 0,
        )

        g_fp += g_fp_t1
        g_vp += g_vp_t1
        g_iv[:, :, :m_cnt] += g_vp * sqrt_tau_np[None, None, :]
        g_iv0_out = g_iv[0].copy() if bool(ctx.needs_input_grad[3]) else None

        g_k_all = g_iv * dk_all
        g_fwd_price = g_k_all[:, :, :m_cnt].copy()
        g_fwd_price[:, :, -1] += (
            g_k_all[:, :, m_cnt]
            + 0.90 * g_k_all[:, :, m_cnt + 1]
            + 1.10 * g_k_all[:, :, m_cnt + 2]
            + 0.80 * g_k_all[:, :, m_cnt + 3]
            + 1.20 * g_k_all[:, :, m_cnt + 4]
        )
        sp_3 = spots_all[:, :, None]
        g_fwd_price += g_fr / sp_3 + g_fp / ref_spots_np[None, :, None]
        g_spots_all = (
            g_m / ref_spots_np[None, :]
            - (g_fr * fwd_price / (sp_3 * sp_3)).sum(axis=2)
            + (g_cd / escrowed).sum(axis=2)
        )
        g_escrowed = g_fwd_price * gf_0[None, :, :] - g_cd * sp_3 / (escrowed * escrowed)
        dE_ds = np.where(
            0.01 * sp_3 > 1e-12,
            0.01 + 0.01 * sp_z + sig_sp * (div_pv_0[None, :, :] / sp_3),
            0.01 + 0.99 * sig_sp,
        )
        g_spots_all += (g_escrowed * dE_ds).sum(axis=2)

        g_sp0_out = torch.from_numpy(g_spots_all[0].copy()).to(device=dev) if bool(ctx.needs_input_grad[0]) else None
        g_spg_out = torch.from_numpy(g_spots_all[1:].copy()).to(device=dev) if bool(ctx.needs_input_grad[1]) else None

        g_corr_out = None
        if need_corr and g_corr_t1 is not None:
            g_corr_sym = g_cm[0] + g_corr_t1[0]
            g_off = g_corr_sym * (1.0 - eye_d)
            g_raw_sym = np.where(abs_c <= 0.999, g_off, g_off * (1.0 - th_c * th_c))
            g_corr_in = 0.5 * (g_raw_sym + g_raw_sym.T) * (1.0 - eye_d)
            g_corr_out = torch.from_numpy(g_corr_in).to(device=dev)

        g_iv0_t = torch.from_numpy(g_iv0_out).to(device=dev) if g_iv0_out is not None else None
        g_div_out = torch.from_numpy(-(g_escrowed[0] * sig_sp[0])).to(device=dev) if bool(ctx.needs_input_grad[4]) else None

        g_dfp_out = None
        if bool(ctx.needs_input_grad[5]):
            g_dfp_tot = g_dfp[0].copy()
            if g_dfp_in is not None:
                g_dfp_tot += np.ascontiguousarray(g_dfp_in[0].detach().cpu().numpy(), dtype=np.float64)
            g_dfp_out = torch.from_numpy(g_dfp_tot * df_pay_base_np).to(device=dev)

        g_growth0 = g_fwd_price[0] * escrowed[0]
        g_dff_out = None
        if bool(ctx.needs_input_grad[6]):
            raw_dff = df_fin_base_np * df_fin_m_np if df_fin_m_np is not None else df_fin_base_np
            g_dff = np.where(raw_dff > 1e-15, -(g_growth0 * gf_0 / df_fin_obs) * df_fin_base_np, 0.0)
            g_dff_out = torch.from_numpy(g_dff).to(device=dev)

        g_dfr_out = None
        if bool(ctx.needs_input_grad[7]) and df_repo_m_np is not None:
            g_dfr = g_growth0 * gf_0 / df_repo_m_np
            tau_term = max(float(tau_obs_np[-1]), 1e-6)
            g_dfr[:, -1] -= np.where(df_repo_m_np[:, -1] > 1e-15, g_rspr / (tau_term * df_repo_m_np[:, -1]), 0.0)
            g_dfr_out = torch.from_numpy(g_dfr).to(device=dev)

        return (g_sp0_out, g_spg_out, g_corr_out, g_iv0_t, g_div_out, g_dfp_out, g_dff_out, g_dfr_out, None, None, None)

    @staticmethod
    def jvp(ctx: Any, *d_inputs: Any) -> tuple[torch.Tensor, ...]:
        _, static_ctx, dev, sp0_np, spg_np, corr_np, iv0_np, div_np, dfp_np, dff_np, dfr_np = ctx.saved_meta
        h = 1.0e-7
        bases = (sp0_np, spg_np, corr_np, iv0_np, div_np, dfp_np, dff_np, dfr_np)
        dirs = tuple(
            np.ascontiguousarray(dt.detach().cpu().numpy(), dtype=np.float64) if (dt is not None and b is not None) else None
            for dt, b in zip(d_inputs[:8], bases)
        )

        def _ev(s: float) -> tuple[npt.NDArray[np.float64], ...]:
            pts = [(b + (s * h) * d) if (b is not None and d is not None) else b for b, d in zip(bases, dirs)]
            return cast(
                tuple[npt.NDArray[np.float64], ...],
                _PreSurrFusedFn._eval_np(
                    pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7], static_ctx
                )[:8],
            )

        op, om = _ev(1.0), _ev(-1.0)
        return tuple(torch.from_numpy((a - b) / (2.0 * h)).to(device=dev) for a, b in zip(op, om))


def evaluate_differentiable_pv(
    state_or_request: AutocallPreprocState | Any,
    *,
    spots_t0: torch.Tensor | None = None,
    corr_matrix: torch.Tensor | None = None,
    df_pay: torch.Tensor | None = None,
    df_obs: torch.Tensor | None = None,
    div_amounts: Mapping[str, torch.Tensor] | None = None,
    surface_quote_overrides: Mapping[str, torch.Tensor] | None = None,
    vol_quote_deltas: Mapping[str, torch.Tensor] | None = None,
    curve_pillar_deltas: torch.Tensor | None = None,
    funding_pillar_deltas: torch.Tensor | None = None,
    asset_curve_pillar_deltas: Mapping[str, torch.Tensor] | None = None,
    repo_pillar_deltas: Mapping[str, torch.Tensor] | None = None,
    repo_curves: Mapping[str, Any] | None = None,
    fx_vol_deltas: Mapping[str, torch.Tensor] | None = None,
    fx_corr_deltas: Mapping[str, torch.Tensor] | None = None,
    surrogate: MaskedEQAutocallSurrogate | None = None,
    spots_in_original_order: bool = False,
    tau_shift: float = 0.0,
    as_of_date_override: dt.datetime | None = None,
    require_second_order_sdpa: bool = False,
    use_fast_cpu_baseline: bool | None = None,
    return_postproc: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, AutocallPostprocOutput]:
    """Evaluates `ProductValue` with a connected IEEE-754 float64 outer AD graph and native `.to()` surrogate casts."""
    if isinstance(state_or_request, AutocallPreprocState):
        base_state = state_or_request
    else:
        base_state = preprocess_eq_autocall(state_or_request)

    eff_surrogate = (
        None
        if (
            surrogate is not None
            and hasattr(surrogate, "is_zero_residual")
            and surrogate.is_zero_residual()
        )
        else surrogate
    )

    d_assets = base_state.d_assets
    m_rem = base_state.m_rem
    d_pad = base_state.d_max
    m_pad = base_state.m_max
    sorted_ids = base_state.sorted_identifiers

    device = base_state.spots_t0.device
    for cand in (
        spots_t0,
        corr_matrix,
        df_pay,
        df_obs,
        curve_pillar_deltas,
        funding_pillar_deltas,
    ):
        if isinstance(cand, torch.Tensor):
            device = cand.device
            break

    if spots_t0 is not None:
        sp_in = spots_t0.to(dtype=torch.float64, device=device)
        if spots_in_original_order:
            lex_idx = torch.tensor(base_state.lex_perm, dtype=torch.long, device=device)
            spots_sorted = sp_in.index_select(0, lex_idx)
        else:
            spots_sorted = sp_in
    else:
        spots_sorted = base_state.spots_t0.to(dtype=torch.float64, device=device)

    ref_spots = base_state.ref_spots.to(dtype=torch.float64, device=device)
    moneyness_t0 = spots_sorted / ref_spots

    as_of_orig = base_state.seasoned_state.as_of_date
    eff_as_of = as_of_date_override
    if eff_as_of is None and tau_shift != 0.0 and isinstance(as_of_orig, dt.datetime):
        day_delta = round(float(tau_shift) * 365.0)
        if abs(float(tau_shift) - day_delta / 365.0) < 1e-6 and day_delta != 0:
            eff_as_of = as_of_orig + dt.timedelta(days=day_delta)

    disc_curve_obj = _normalize_curve_obj(base_state.discount_curve, as_of_date=as_of_orig)
    disc_id = str(getattr(disc_curve_obj, "identifier", getattr(disc_curve_obj, "Identifier", base_state.discount_curve_id or "DEFAULT")))

    # Handle seasoned trades or calendar rolls that cross all remaining observation dates
    if m_rem == 0 or base_state.already_autocalled:
        pv_transit = _rolled_in_transit_pv(
            base_state,
            disc_curve_obj,
            eff_as_of if eff_as_of is not None else as_of_orig,
            curve_pillar_deltas,
            device,
        )
        if return_postproc:
            post_zero = assemble_autocall_cashflows_and_pv(
                base_state,
                p_stop=torch.zeros(m_rem, dtype=torch.float64, device=device),
                q_cpn=torch.zeros(m_rem, dtype=torch.float64, device=device),
                l_put=torch.tensor(0.0, dtype=torch.float64, device=device),
            )
            post_zero.product_value = pv_transit
            return pv_transit, post_zero
        return pv_transit

    k_cross = 0
    eff_autocalled = bool(base_state.already_autocalled)
    eff_i_ki = float(base_state.i_ki_past)
    eff_c_mem = float(base_state.c_mem_0)
    crossed_pv = 0.0
    if eff_as_of is not None:
        k_cross, eff_autocalled, eff_i_ki, eff_c_mem, crossed_pv = _collapse_crossed_periods_for_roll(
            base_state,
            disc_curve_obj,
            eff_as_of,
            spots_sorted.detach().cpu().numpy(),
            ref_spots.detach().cpu().numpy(),
        )
        if eff_autocalled or k_cross >= m_rem:
            pv_done = _rolled_in_transit_pv(
                base_state, disc_curve_obj, eff_as_of, curve_pillar_deltas, device
            ) + crossed_pv
            return pv_done

    m_slice = slice(k_cross, m_rem)
    m_active = m_rem - k_cross
    future_obs_dts = base_state.seasoned_state.future_obs_dates[m_slice]
    future_pay_dts = base_state.seasoned_state.future_pay_dates[m_slice]

    # 2. Observation & payment tenors
    tau_obs_orig = base_state.tau_obs.to(dtype=torch.float64, device=device)[m_slice]
    tau_pay_orig = base_state.tau_pay.to(dtype=torch.float64, device=device)[m_slice]
    time_dcc = base_state.seasoned_state.day_count_convention
    if eff_as_of is not None:
        tau_obs_np = np.array(
            [max(_year_fraction(eff_as_of, d, time_dcc), 1e-6) for d in future_obs_dts],
            dtype=np.float64,
        )
        tau_pay_np = np.array(
            [max(_year_fraction(eff_as_of, d, time_dcc), 1e-6) for d in future_pay_dts],
            dtype=np.float64,
        )
        tau_obs_t = torch.tensor(tau_obs_np, dtype=torch.float64, device=device)
        tau_pay_t = torch.tensor(tau_pay_np, dtype=torch.float64, device=device)
        delta_tau_t = (
            torch.cat([tau_obs_t[:1], (tau_obs_t[1:] - tau_obs_t[:-1]).clamp(min=1e-8)], dim=0)
            if m_active > 1
            else tau_obs_t[:1]
        )
    elif tau_shift != 0.0:
        tau_obs_t = (tau_obs_orig - float(tau_shift)).clamp(min=1e-6)
        tau_pay_t = (tau_pay_orig - float(tau_shift)).clamp(min=1e-6)
        delta_tau_t = (
            torch.cat([tau_obs_t[:1], (tau_obs_t[1:] - tau_obs_t[:-1]).clamp(min=1e-8)], dim=0)
            if m_active > 1
            else tau_obs_t[:1]
        )
    else:
        tau_obs_t = tau_obs_orig
        tau_pay_t = tau_pay_orig
        delta_tau_t = base_state.delta_tau.to(dtype=torch.float64, device=device)[m_slice]

    # 3. Discount factors at observation and payment dates using rolled datetime dates
    disc_anchor_val = (
        max(float(np.asarray(disc_curve_obj.df(eff_as_of), dtype=np.float64).reshape(-1)[0]), 1e-15)
        if (eff_as_of is not None and disc_curve_obj is not None and hasattr(disc_curve_obj, "df"))
        else 1.0
    )
    if df_obs is not None:
        df_obs_t = df_obs.to(dtype=torch.float64, device=device)[m_slice]
    else:
        if eff_as_of is not None and hasattr(disc_curve_obj, "df"):
            df_obs_np = (
                np.asarray(disc_curve_obj.df(future_obs_dts), dtype=np.float64).reshape(-1)
                / disc_anchor_val
            )
            df_obs_t = torch.tensor(df_obs_np, dtype=torch.float64, device=device)
        elif tau_shift != 0.0 and hasattr(disc_curve_obj, "df"):
            df_obs_np = np.asarray(
                disc_curve_obj.df(tau_obs_t.detach().cpu().numpy()),
                dtype=np.float64,
            ).reshape(-1)
            df_obs_t = torch.tensor(df_obs_np, dtype=torch.float64, device=device)
        else:
            df_obs_t = base_state.df_obs.to(dtype=torch.float64, device=device)[m_slice]
        if curve_pillar_deltas is not None:
            obs_mult = _curve_df_delta_multiplier(
                disc_curve_obj, future_obs_dts, curve_pillar_deltas, as_of_date=as_of_orig
            )
            if eff_as_of is not None:
                obs_mult = obs_mult / _curve_df_delta_multiplier(
                    disc_curve_obj, [eff_as_of], curve_pillar_deltas, as_of_date=as_of_orig
                )[0].clamp(min=1e-15)
            df_obs_t = df_obs_t * obs_mult

    if df_pay is not None:
        df_pay_t = df_pay.to(dtype=torch.float64, device=device)[m_slice]
    else:
        if eff_as_of is not None and hasattr(disc_curve_obj, "df"):
            df_pay_np = (
                np.asarray(disc_curve_obj.df(future_pay_dts), dtype=np.float64).reshape(-1)
                / disc_anchor_val
            )
            df_pay_t = torch.tensor(df_pay_np, dtype=torch.float64, device=device)
        elif tau_shift != 0.0 and hasattr(disc_curve_obj, "df"):
            df_pay_np = np.asarray(
                disc_curve_obj.df(tau_pay_t.detach().cpu().numpy()),
                dtype=np.float64,
            ).reshape(-1)
            df_pay_t = torch.tensor(df_pay_np, dtype=torch.float64, device=device)
        else:
            df_pay_t = base_state.df_pay.to(dtype=torch.float64, device=device)[m_slice]
        if curve_pillar_deltas is not None:
            pay_mult = _curve_df_delta_multiplier(
                disc_curve_obj, future_pay_dts, curve_pillar_deltas, as_of_date=as_of_orig
            )
            if eff_as_of is not None:
                pay_mult = pay_mult / _curve_df_delta_multiplier(
                    disc_curve_obj, [eff_as_of], curve_pillar_deltas, as_of_date=as_of_orig
                )[0].clamp(min=1e-15)
            df_pay_t = df_pay_t * pay_mult

    # Resolve per-asset financing curves and deltas (supporting multi-curve & asset-ccy curves)
    f_leg_meta = base_state.funding_leg or {}
    f_leg_curve_id = f_leg_meta.get("funding_curve_id")
    asset_fin_deltas: list[torch.Tensor | None] = []
    asset_fin_curves: list[Any] = []
    for ticker in sorted_ids:
        c_i = _normalize_curve_obj(
            base_state.asset_financing_curves.get(ticker, disc_curve_obj),
            as_of_date=as_of_orig,
        )
        asset_fin_curves.append(c_i)
        c_i_id = str(getattr(c_i, "identifier", getattr(c_i, "Identifier", disc_id)))
        if asset_curve_pillar_deltas is not None and ticker in asset_curve_pillar_deltas:
            asset_fin_deltas.append(asset_curve_pillar_deltas[ticker])
        elif c_i is disc_curve_obj or c_i_id == disc_id:
            asset_fin_deltas.append(curve_pillar_deltas)
        elif funding_pillar_deltas is not None and f_leg_curve_id is not None and c_i_id == str(f_leg_curve_id):
            asset_fin_deltas.append(funding_pillar_deltas)
        else:
            asset_fin_deltas.append(None)

    # 4. Buehler Escrowed Dividend PVs, Smooth Softplus Floor, Forwards, & Repo/Quanto Live Drift
    div_sched = base_state.dividend_schedule
    eff_as_of_cmp = eff_as_of if eff_as_of is not None else as_of_orig
    need_div_recomp = (
        div_amounts is not None
        or curve_pillar_deltas is not None
        or funding_pillar_deltas is not None
        or asset_curve_pillar_deltas is not None
        or repo_pillar_deltas is not None
        or eff_as_of is not None
        or tau_shift != 0.0
    )
    if not need_div_recomp:
        div_pv_t = div_sched.div_pv.to(dtype=torch.float64, device=device)[:, m_slice]
    else:
        div_pv_rows: list[torch.Tensor] = []
        for i, ticker in enumerate(sorted_ids):
            events_i = div_sched.dividend_events.get(ticker, [])
            if len(events_i) > 0:
                amt_t = (
                    div_amounts[ticker].to(dtype=torch.float64, device=device)
                    if (div_amounts is not None and ticker in div_amounts)
                    else div_sched.dividend_amount_tensors[ticker].to(dtype=torch.float64, device=device)
                )
                w_base_t = div_sched.dividend_pv_weights[ticker].to(dtype=torch.float64, device=device)
                inc_t = div_sched.dividend_obs_incidence[ticker].to(dtype=torch.float64, device=device)[:, m_slice]
                w_t = w_base_t

                c_i = asset_fin_curves[i]
                fin_delta_i = asset_fin_deltas[i]
                if eff_as_of is not None and c_i is not None and hasattr(c_i, "df"):
                    c_anchor = max(
                        float(np.asarray(c_i.df(eff_as_of), dtype=np.float64).reshape(-1)[0]),
                        1e-15,
                    )
                    valid_np = np.array([1.0 if ev.ex_date > eff_as_of else 0.0 for ev in events_i], dtype=np.float64)
                    # Note: incidence already zeroed events <= as_of_orig; re-scaling by 1 / c_i.df(eff_as_of) rolls PV.
                    w_t = w_t * (torch.tensor(valid_np, dtype=torch.float64, device=device) / c_anchor)
                elif tau_shift != 0.0 and c_i is not None and hasattr(c_i, "df"):
                    t_ex_arr = np.array([ev.tau_ex_act365 for ev in events_i], dtype=np.float64)
                    t_pay_arr = np.array([ev.tau_pay_act365 for ev in events_i], dtype=np.float64)
                    valid_mask = t_ex_arr > float(tau_shift)
                    t_pay_shift_arr = np.maximum(t_pay_arr - float(tau_shift), 1e-6)
                    df_new_arr = np.asarray(c_i.df(t_pay_shift_arr), dtype=np.float64).reshape(-1)
                    df_old_arr = np.asarray(c_i.df(t_pay_arr), dtype=np.float64).reshape(-1)
                    ratio_arr = np.where(valid_mask, df_new_arr / np.maximum(df_old_arr, 1e-15), 0.0)
                    w_t = w_t * torch.tensor(ratio_arr, dtype=torch.float64, device=device)

                if fin_delta_i is not None:
                    active_pay_dates = [ev.payment_date for ev in events_i]
                    div_df_mult = _curve_df_delta_multiplier(
                        c_i, active_pay_dates, fin_delta_i, as_of_date=as_of_orig
                    )
                    active_ex_mask = torch.tensor(
                        [1.0 if ev.ex_date > eff_as_of_cmp else 0.0 for ev in events_i],
                        dtype=torch.float64,
                        device=device,
                    )
                    w_t = w_t * torch.where(active_ex_mask > 0.5, div_df_mult, torch.ones_like(div_df_mult))

                if repo_pillar_deltas is not None and ticker in repo_pillar_deltas and repo_curves is not None and ticker in repo_curves:
                    r_curve_i = _normalize_curve_obj(repo_curves[ticker], as_of_date=as_of_orig)
                    ex_dts = [ev.ex_date for ev in events_i]
                    repo_ex_mult = _curve_df_delta_multiplier(
                        r_curve_i, ex_dts, repo_pillar_deltas[ticker], as_of_date=as_of_orig
                    )
                    active_ex_mask = torch.tensor(
                        [1.0 if ev.ex_date > eff_as_of_cmp else 0.0 for ev in events_i],
                        dtype=torch.float64,
                        device=device,
                    )
                    w_t = w_t / torch.where(active_ex_mask > 0.5, repo_ex_mult.clamp(min=1e-15), torch.ones_like(repo_ex_mult))

                pv_each = amt_t * w_t
                div_pv_row = torch.matmul(pv_each.unsqueeze(0), inc_t).squeeze(0)
            else:
                div_pv_row = torch.zeros(m_active, dtype=torch.float64, device=device)
            div_pv_rows.append(div_pv_row)
        div_pv_t = torch.stack(div_pv_rows, dim=0)

    spots_col = spots_sorted.unsqueeze(1)
    ref_col = ref_spots.unsqueeze(1)
    escrowed_spot = _escrowed_spot_softplus(spots_col, div_pv_t)

    # 5. Live Quanto Drift & FX Sensitivities + Sticky-Strike IVs + 7D Node Surface Features
    r_tot_base = div_sched.r_tot_obs.to(dtype=torch.float64, device=device)[:, m_slice]
    df_fin_base = div_sched.df_fin_obs.to(dtype=torch.float64, device=device)[:, m_slice].clamp(min=1e-15)
    repo_spread_avg = div_sched.repo_spread_avg.to(dtype=torch.float64, device=device).clone()

    # Build per-asset financing curve DFs at observation dates
    df_fin_obs_rows: list[torch.Tensor] = []
    for i, ticker in enumerate(sorted_ids):
        c_i = asset_fin_curves[i]
        fin_delta_i = asset_fin_deltas[i]
        c_i_id = str(getattr(c_i, "identifier", getattr(c_i, "Identifier", disc_id)))
        if eff_as_of is None and tau_shift == 0.0 and df_obs is None:
            df_fin_i = df_fin_base[i]
        elif (c_i is disc_curve_obj or c_i_id == disc_id) and df_obs is not None:
            df_fin_i = df_obs_t
        elif eff_as_of is not None and c_i is not None and hasattr(c_i, "df"):
            c_anchor = max(float(np.asarray(c_i.df(eff_as_of), dtype=np.float64).reshape(-1)[0]), 1e-15)
            df_fin_i = torch.tensor(
                np.asarray(c_i.df(future_obs_dts), dtype=np.float64).reshape(-1) / c_anchor,
                dtype=torch.float64,
                device=device,
            )
        elif tau_shift != 0.0 and c_i is not None and hasattr(c_i, "df"):
            df_fin_i = torch.tensor(
                np.asarray(c_i.df(tau_obs_t.detach().cpu().numpy()), dtype=np.float64).reshape(-1),
                dtype=torch.float64,
                device=device,
            )
        else:
            df_fin_i = df_fin_base[i]

        if fin_delta_i is not None and not ((c_i is disc_curve_obj or c_i_id == disc_id) and df_obs is not None):
            f_mult = _curve_df_delta_multiplier(
                c_i, future_obs_dts, fin_delta_i, as_of_date=as_of_orig
            )
            if eff_as_of is not None:
                f_mult = f_mult / _curve_df_delta_multiplier(
                    c_i, [eff_as_of], fin_delta_i, as_of_date=as_of_orig
                )[0].clamp(min=1e-15)
            df_fin_i = df_fin_i * f_mult
        df_fin_obs_rows.append(df_fin_i.clamp(min=1e-15))
    df_fin_obs_t = torch.stack(df_fin_obs_rows, dim=0)

    # Repo + Quanto component without base financing DF
    df_repo_quanto_base = torch.exp(r_tot_base) * df_fin_base
    if eff_as_of is not None or tau_shift != 0.0:
        tau_ratio = (tau_obs_t / tau_obs_orig.clamp(min=1e-12)).unsqueeze(0)
        df_repo_quanto_base = torch.pow(df_repo_quanto_base.clamp(min=1e-15), tau_ratio)

    # Optional repo pillar deltas adjust df_repo_obs and node column REPO_SPREAD_AVG
    if repo_pillar_deltas is not None and repo_curves is not None:
        rq_rows: list[torch.Tensor] = []
        for i, ticker in enumerate(sorted_ids):
            row_i = df_repo_quanto_base[i]
            if ticker in repo_pillar_deltas and ticker in repo_curves:
                r_curve_i = _normalize_curve_obj(repo_curves[ticker], as_of_date=as_of_orig)
                r_mult = _curve_df_delta_multiplier(
                    r_curve_i, future_obs_dts, repo_pillar_deltas[ticker], as_of_date=as_of_orig
                )
                row_i = row_i * r_mult
                tau_term_repo = max(float(tau_obs_t[-1].detach().cpu().item()), 1e-6)
                repo_spread_avg[i] = repo_spread_avg[i] - torch.log(r_mult[-1].clamp(min=1e-15)) / tau_term_repo
            rq_rows.append(row_i)
        df_repo_quanto_base = torch.stack(rq_rows, dim=0)

    # First pass of growth factor (prior to live quanto drift correction) gives forward anchors,
    # then live quanto drift adjustment updates growth factor consistently.
    quanto_meta = getattr(base_state, "quanto_meta", None) or {}
    if quanto_meta:
        q_corr_rows: list[torch.Tensor] = []
        tau_obs_365_orig = tau_obs_orig.clamp(min=1e-12)
        base_spots_det = base_state.spots_t0.detach().to(dtype=torch.float64, device=device)
        for i, ticker in enumerate(sorted_ids):
            qm = quanto_meta.get(ticker)
            if not qm:
                q_corr_rows.append(torch.zeros(m_active, dtype=torch.float64, device=device))
                continue
            rho_base_f = float(qm.get("rho_s_fx", 0.0))
            sig_fx_arr = np.asarray(qm.get("sig_fx_obs", [0.0] * m_rem), dtype=np.float64)[m_slice]
            sig_fx_base_t = torch.tensor(sig_fx_arr, dtype=torch.float64, device=device)
            sig_fx_live = (
                sig_fx_base_t + fx_vol_deltas[ticker].to(dtype=torch.float64, device=device)
                if (fx_vol_deltas is not None and ticker in fx_vol_deltas)
                else sig_fx_base_t
            )
            rho_live = (
                rho_base_f + fx_corr_deltas[ticker].to(dtype=torch.float64, device=device).reshape(())
                if (fx_corr_deltas is not None and ticker in fx_corr_deltas)
                else rho_base_f
            )
            surf_i = base_state.vol_surfaces[ticker]
            q_delta_i_q: torch.Tensor | None = None
            if vol_quote_deltas is not None and ticker in vol_quote_deltas:
                q_delta_i_q = vol_quote_deltas[ticker].to(dtype=torch.float64, device=device)
            elif surface_quote_overrides is not None and ticker in surface_quote_overrides:
                raw_q_base = torch.tensor(
                    np.array(surf_i.raw_quotes, copy=True), dtype=torch.float64, device=device
                )
                q_delta_i_q = surface_quote_overrides[ticker].to(dtype=torch.float64, device=device) - raw_q_base
            sig_eq_live = _eval_reg_surface_iv_batch_torch(
                surf_i, tau_obs_t, spots_sorted[i].expand(m_active), q_delta_i_q
            )
            with torch.no_grad():
                sig_eq_base = _eval_reg_surface_iv_batch_torch(
                    surf_i, tau_obs_orig, base_spots_det[i].expand(m_active), None
                )
            drift_live = -rho_live * sig_eq_live * sig_fx_live * tau_obs_t
            drift_base = -rho_base_f * sig_eq_base * sig_fx_base_t * tau_obs_365_orig
            q_corr_rows.append(drift_live - drift_base)
        quanto_delta_t = torch.stack(q_corr_rows, dim=0)
        growth_factor = (df_repo_quanto_base / df_fin_obs_t) * torch.exp(quanto_delta_t)
    else:
        growth_factor = df_repo_quanto_base / df_fin_obs_t

    forward_price = escrowed_spot * growth_factor
    forward_rel = forward_price / spots_col
    forward_perf = forward_price / ref_col
    cash_div_leverage = spots_col / escrowed_spot

    # Evaluate sticky-strike IVs and full differentiable 7D smile features for every asset
    t_term_t = tau_obs_t[-1]
    iv_rows: list[torch.Tensor] = []
    surf_feat_rows: list[torch.Tensor] = []
    t_all_queries = (
        torch.cat([tau_obs_t, t_term_t.expand(5)], dim=0)
        if eff_surrogate is not None
        else tau_obs_t
    )

    for i, ticker in enumerate(sorted_ids):
        surf_i = base_state.vol_surfaces[ticker]
        q_delta_i: torch.Tensor | None = None
        if vol_quote_deltas is not None and ticker in vol_quote_deltas:
            q_delta_i = vol_quote_deltas[ticker].to(dtype=torch.float64, device=device)
        elif surface_quote_overrides is not None and ticker in surface_quote_overrides:
            raw_q_base = torch.tensor(
                np.array(surf_i.raw_quotes, copy=True), dtype=torch.float64, device=device
            )
            q_delta_i = surface_quote_overrides[ticker].to(dtype=torch.float64, device=device) - raw_q_base

        if eff_surrogate is not None:
            # Live forward stencil preserves full derivative flow through smile slope/skew/curvature (GRK-016).
            f_stencil = forward_price[i, -1]
            k_queries = torch.cat(
                [
                    forward_price[i],
                    torch.stack(
                        [
                            f_stencil,
                            0.90 * f_stencil,
                            1.10 * f_stencil,
                            0.80 * f_stencil,
                            1.20 * f_stencil,
                        ],
                        dim=0,
                    ),
                ],
                dim=0,
            )
            iv_all = _eval_reg_surface_iv_batch_torch(surf_i, t_all_queries, k_queries, q_delta_i)
            iv_m_vec = iv_all[:m_active]
            iv_rows.append(iv_m_vec)

            iv_short = iv_m_vec[0]
            iv_term = iv_m_vec[-1]
            iv_c = iv_all[m_active]
            iv_m10 = iv_all[m_active + 1]
            iv_p10 = iv_all[m_active + 2]
            iv_m20 = iv_all[m_active + 3]
            iv_p20 = iv_all[m_active + 4]

            iv_slope = iv_term - iv_short
            denom_skew = torch.clamp(0.20 * iv_c, min=1e-6)
            ssvi_rho = torch.tanh((iv_p10 - iv_m10) / denom_skew)
            denom_curv = torch.clamp(0.01 * iv_c, min=1e-6)
            ssvi_eta = torch.clamp((iv_p10 - 2.0 * iv_c + iv_m10) / denom_curv, min=1e-4)
            pc1 = iv_m20 - iv_c
            pc2 = iv_p20 - iv_c
            surf_feat_rows.append(
                torch.stack([iv_short, iv_term, iv_slope, ssvi_rho, ssvi_eta, pc1, pc2], dim=0)
            )
        else:
            iv_m_vec = _eval_reg_surface_iv_batch_torch(
                surf_i, tau_obs_t, forward_price[i], q_delta_i
            )
            iv_rows.append(iv_m_vec)

    iv_at_fwd_t = torch.stack(iv_rows, dim=0)
    v_perf_t = iv_at_fwd_t * torch.sqrt(tau_obs_t.clamp(min=1e-12)).unsqueeze(0)

    # 6. Symmetrized Correlation Matrix with smooth |rho| -> 1 guard
    if corr_matrix is not None:
        corr_in = corr_matrix.to(dtype=torch.float64, device=device)
        eye_d = torch.eye(d_assets, dtype=torch.float64, device=device)
        corr_raw_sym = 0.5 * (corr_in + corr_in.transpose(0, 1)) * (1.0 - eye_d) + eye_d
    else:
        corr_raw_sym = base_state.corr_matrix.to(dtype=torch.float64, device=device)
    corr_sym = _smooth_corr_guard(corr_raw_sym)

    ko_barriers_t = base_state.ko_barriers.to(dtype=torch.float64, device=device)[m_slice]
    is_callable_t = base_state.is_callable.to(dtype=torch.float64, device=device)[m_slice]
    cpn_barriers_t = base_state.cpn_barriers.to(dtype=torch.float64, device=device)[m_slice]
    is_guaranteed_t = base_state.is_guaranteed_cpn.to(dtype=torch.float64, device=device)[m_slice]
    coupon_amounts_t = base_state.coupon_amounts.to(dtype=torch.float64, device=device)[m_slice]

    # 7. Analytical Stulz-Margrabe Moment-Matched Basket Proxy in torch.float64
    proxy_inputs = {
        "forward_perf": forward_perf,
        "v_perf": v_perf_t,
        "corr_matrix": corr_sym,
        "df_pay": df_pay_t,
        "ko_barriers": ko_barriers_t,
        "is_callable": is_callable_t,
        "cpn_barriers": cpn_barriers_t,
        "is_guaranteed_cpn": is_guaranteed_t,
        "coupon_amounts": coupon_amounts_t,
        "strike": base_state.strike,
        "option_gearing": base_state.option_gearing,
        "lower_strike": base_state.lower_strike,
        "ki_barrier": base_state.ki_barrier,
        "i_ki_past": eff_i_ki,
        "c_mem_0": eff_c_mem,
        "is_memory": base_state.is_memory,
        "ko_spread": base_state.ko_spread,
        "cpn_spread": base_state.cpn_spread,
        "ki_spread": base_state.ki_spread,
        "no_notional_repayment": base_state.no_notional_repayment,
        "autocall_coupon": base_state.autocall_coupon,
        "force_coupons_on_autocall": base_state.force_coupons_on_autocall,
        "coupon_barrier_type": base_state.coupon_barrier_type,
        "coupon_barrier_direction": base_state.coupon_barrier_direction,
        "is_long_basket": base_state.is_long_basket,
        "participation_call_strike": base_state.participation_call_strike,
        "participation_call_rate": base_state.participation_call_rate,
        "one_star_strike": base_state.one_star_strike,
        "one_star_coupon": base_state.one_star_coupon,
        "pv_in_transit": base_state.pv_in_transit if eff_as_of is None else 0.0,
        "already_autocalled": eff_autocalled,
        "callability_index": base_state.callability_index,
        "coupon_index": base_state.coupon_index,
        "option_index": base_state.option_index,
        "allow_autocall_on_expiry": base_state.allow_autocall_on_expiry,
        "inv_lex_perm": base_state.inv_lex_perm,
    }
    # Decouple float64 C++23 proxy (when no Tier-1 AD is required) and pure-Tensor compiled Tier-1 AD.
    t1_needs_grad = torch.is_grad_enabled() and any(
        isinstance(t, torch.Tensor) and t.requires_grad
        for t in (forward_perf, v_perf_t, corr_sym, df_pay_t, ko_barriers_t, cpn_barriers_t, coupon_amounts_t)
    )
    want_fast_cpu = (use_fast_cpu_baseline is True) or (
        use_fast_cpu_baseline is None and device.type == "cpu" and not t1_needs_grad
    )
    call_code = monitoring_index_to_code(base_state.callability_index)
    cpn_code = monitoring_index_to_code(base_state.coupon_index or base_state.callability_index)
    opt_code = monitoring_index_to_code(base_state.option_index or base_state.callability_index)
    can_pure_tier1 = (
        not want_fast_cpu
        and m_active >= 1
        and not eff_autocalled
        and call_code in (0, 1)
        and cpn_code in (0, 1)
        and opt_code in (0, 1)
        and (base_state.participation_call_strike is None or float(base_state.participation_call_strike) <= 0.0 or float(base_state.participation_call_rate) == 0.0)
        and (base_state.one_star_strike is None or float(base_state.one_star_strike) <= 0.0 or float(base_state.one_star_coupon) == 0.0)
        and not _in_functorch_transform()
    )
    if want_fast_cpu:
        baseline = compute_analytical_autocall_baseline(proxy_inputs, use_fast_cpu=True)
    elif can_pure_tier1:
        is_call_flags_t1 = tuple(bool(x) for x in (is_callable_t.detach().cpu() > 0.5).tolist())
        has_ki_t1 = (
            base_state.ki_barrier is not None
            and float(base_state.ki_barrier) > 0.0
            and eff_i_ki < 0.5
        )
        ki_b_t1 = float(base_state.ki_barrier) if base_state.ki_barrier is not None else 0.0
        fw_b, vw_b, p_b, q_b, lp_b, dw_b = _run_tier1_pure_cached(
            forward_perf.unsqueeze(0),
            v_perf_t.unsqueeze(0),
            corr_sym.unsqueeze(0),
            ko_barriers_t,
            cpn_barriers_t,
            is_guaranteed_t,
            is_call_flags=is_call_flags_t1,
            call_code=call_code,
            cpn_code=cpn_code,
            opt_code=opt_code,
            cpn_b_type=str(base_state.coupon_barrier_type),
            cpn_b_dir=str(base_state.coupon_barrier_direction),
            force_cpn=bool(base_state.force_coupons_on_autocall),
            allow_exp=bool(base_state.allow_autocall_on_expiry),
            ko_spread=float(base_state.ko_spread),
            cpn_spread=float(base_state.cpn_spread),
            ki_spread=float(base_state.ki_spread),
            strike=float(base_state.strike),
            option_gearing=float(base_state.option_gearing),
            lower_strike_val=float(base_state.lower_strike) if base_state.lower_strike is not None else 0.0,
            has_active_ki=has_ki_t1,
            ki_barrier_val=ki_b_t1,
        )
        z_m0 = torch.zeros(m_active, dtype=torch.float64, device=device)
        z_s0 = torch.tensor(0.0, dtype=torch.float64, device=device)
        baseline = AutocallBaselineOutput(
            f_w=fw_b[0],
            v_w=vw_b[0],
            h_proxy=z_m0,
            p_proxy=p_b[0],
            survival_proxy=z_m0,
            q_proxy=q_b[0],
            l_put_proxy=lp_b[0],
            proxy_delta_weights=dw_b[0],
            option_pv_proxy=z_s0,
            notional_pv_proxy=z_s0,
            coupon_pv_proxy=z_s0,
            product_value_proxy=z_s0,
            l_call_proxy=z_s0,
            l_onestar_proxy=z_s0,
            has_unsupported_exotic_index=False,
            unsupported_indices=(),
        )
    else:
        baseline = compute_analytical_autocall_baseline(proxy_inputs, use_fast_cpu=False)

    # 8. Decoupled Float64/Float32 Single-VJP Surrogate AD via plain `.to()` boundaries (GRK-001)
    if eff_surrogate is not None:
        first_param = next(eff_surrogate.parameters(), None)
        surr_dtype = first_param.dtype if first_param is not None else torch.float32
        surr_dev = first_param.device if first_param is not None else device
        surr_d_node = int(getattr(eff_surrogate, "d_node", len(AutocallNodeCol)))
        surr_d_global = int(getattr(eff_surrogate, "d_global", len(AutocallGlobalCol)))

        surf_feats_t = torch.stack(surf_feat_rows, dim=0)
        log_m0 = torch.log(moneyness_t0.clamp(min=1e-12))
        tau_term = tau_obs_t[-1].clamp(min=1e-6)
        log_fwd_carry_1y = torch.log(forward_rel[:, -1].clamp(min=1e-12)) / tau_term
        log_div_lev = torch.log(cash_div_leverage[:, -1].clamp(min=1.0))
        f_p_term = forward_perf[:, -1].clamp(min=1e-12)
        v_term = v_perf_t[:, -1].clamp(min=1e-6)
        k_eff = max(float(base_state.strike), 1e-4)
        dist_to_put = torch.log(f_p_term / k_eff) / v_term
        active_ones = torch.ones(d_assets, dtype=torch.float64, device=device)

        role_leg_a = torch.zeros(d_assets, dtype=torch.float64, device=device)
        role_leg_b = torch.zeros(d_assets, dtype=torch.float64, device=device)
        active_idx_names = (
            base_state.callability_index,
            base_state.coupon_index or base_state.callability_index,
            base_state.option_index or base_state.callability_index,
        )
        if d_assets >= 2 and any(nm in PAIR_ORDER_DEPENDENT_INDICES for nm in active_idx_names):
            role_leg_a[int(base_state.inv_lex_perm[0])] = 1.0
            role_leg_b[int(base_state.inv_lex_perm[1])] = 1.0

        node_cols: list[torch.Tensor] = [torch.zeros(d_assets, dtype=torch.float64, device=device)] * len(AutocallNodeCol)
        node_cols[AutocallNodeCol.LOG_MONEYNESS] = log_m0
        node_cols[AutocallNodeCol.LOG_FWD_CARRY_1Y] = log_fwd_carry_1y
        node_cols[AutocallNodeCol.CASH_DIV_LEVERAGE] = log_div_lev
        node_cols[AutocallNodeCol.REPO_SPREAD_AVG] = repo_spread_avg
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
        node_feats = torch.stack(node_cols, dim=1)[:, :surr_d_node].unsqueeze(0)
        asset_mask = active_ones.unsqueeze(0)

        v_term_vec = v_perf_t[:, -1]
        log_fp_term = torch.log(f_p_term)
        cov_active = corr_sym * v_term_vec.unsqueeze(1) * v_term_vec.unsqueeze(0)
        log_perf_diff_active = torch.abs(log_fp_term.unsqueeze(1) - log_fp_term.unsqueeze(0))
        off_diag_active = 1.0 - torch.eye(d_assets, dtype=torch.float64, device=device)
        edge_feats = torch.stack(
            [corr_sym, cov_active, log_perf_diff_active, off_diag_active], dim=-1
        ).unsqueeze(0)

        sb = base_state.seasoned_state.sentinel_barriers
        if sb is None:
            raise RuntimeError("Seasoned state sentinel_barriers must be initialized before evaluating PV.")
        log_ko_feat_t = torch.tensor(
            np.array(sb.log_ko_barrier_feat[m_slice], copy=True), dtype=torch.float64, device=device
        )
        log_cpn_feat_t = torch.tensor(
            np.array(sb.log_cpn_barrier_feat[m_slice], copy=True), dtype=torch.float64, device=device
        )
        log_fw_t = torch.log(baseline.f_w.clamp(min=1e-12))
        sched_feats = torch.stack(
            [
                tau_obs_t,
                delta_tau_t,
                df_pay_t,
                log_ko_feat_t,
                is_callable_t,
                log_cpn_feat_t,
                is_guaranteed_t,
                coupon_amounts_t,
                log_fw_t,
                baseline.v_w,
                baseline.p_proxy,
                baseline.q_proxy,
            ],
            dim=1,
        ).unsqueeze(0)
        sched_mask = torch.ones((1, m_active), dtype=torch.float64, device=device)

        has_ki = 1.0 if (base_state.ki_barrier is not None and base_state.ki_barrier > 0.0) else 0.0
        b_ki_val = (
            float(base_state.ki_barrier)
            if (base_state.ki_barrier is not None and base_state.ki_barrier > 0.0)
            else float(base_state.strike)
        )
        lower_k_val = float(base_state.lower_strike) if base_state.lower_strike is not None else 0.0
        g_cols: list[torch.Tensor] = [torch.tensor(0.0, dtype=torch.float64, device=device)] * len(AutocallGlobalCol)
        g_cols[AutocallGlobalCol.K_PUT] = torch.tensor(float(base_state.strike), dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.B_KI] = torch.tensor(b_ki_val, dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.HAS_KI_FLAG] = torch.tensor(has_ki, dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.I_KI_PAST] = torch.tensor(eff_i_ki, dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.G_PUT] = torch.tensor(float(base_state.option_gearing), dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.C_MEM_0] = torch.tensor(eff_c_mem, dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.IS_MEMORY_FLAG] = torch.tensor(1.0 if base_state.is_memory else 0.0, dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.ABS_KO_SPREAD] = torch.tensor(abs(float(base_state.ko_spread)), dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.ABS_CPN_SPREAD] = torch.tensor(abs(float(base_state.cpn_spread)), dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.L_PUT_PROXY] = baseline.l_put_proxy
        g_cols[AutocallGlobalCol.LOWER_STRIKE] = torch.tensor(lower_k_val, dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.CALL_INDEX_ID] = torch.tensor(float(monitoring_index_to_code(base_state.callability_index)), dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.CPN_INDEX_ID] = torch.tensor(float(monitoring_index_to_code(base_state.coupon_index or base_state.callability_index)), dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.OPT_INDEX_ID] = torch.tensor(float(monitoring_index_to_code(base_state.option_index or base_state.callability_index)), dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.IS_LONG_BASKET] = torch.tensor(1.0 if base_state.is_long_basket else 0.0, dtype=torch.float64, device=device)
        g_cols[AutocallGlobalCol.ABS_KI_SPREAD] = torch.tensor(abs(float(base_state.ki_spread)), dtype=torch.float64, device=device)
        global_feats = torch.stack(g_cols, dim=0)[:surr_d_global].unsqueeze(0)

        nf_t2 = cast_to_surrogate_boundary(node_feats, device=surr_dev, dtype=surr_dtype)
        ef_t2 = cast_to_surrogate_boundary(edge_feats, device=surr_dev, dtype=surr_dtype)
        am_t2 = cast_to_surrogate_boundary(asset_mask, device=surr_dev, dtype=surr_dtype)
        sf_t2 = cast_to_surrogate_boundary(sched_feats, device=surr_dev, dtype=surr_dtype)
        sm_t2 = cast_to_surrogate_boundary(sched_mask, device=surr_dev, dtype=surr_dtype)
        gf_t2 = cast_to_surrogate_boundary(global_feats, device=surr_dev, dtype=surr_dtype)

        d_haz_t2, d_cpn_t2, d_put_t2 = _call_surrogate_residuals_fast(
            eff_surrogate,
            nf_t2,
            ef_t2,
            am_t2,
            sf_t2,
            sm_t2,
            gf_t2,
            require_second_order_sdpa=require_second_order_sdpa,
        )
        d_haz = cast_from_surrogate_boundary(d_haz_t2[0, :m_active], device=device, dtype=torch.float64)
        d_cpn = cast_from_surrogate_boundary(d_cpn_t2[0, :m_active], device=device, dtype=torch.float64)
        d_put = cast_from_surrogate_boundary(d_put_t2[0], device=device, dtype=torch.float64)

        p_proxy_c = baseline.p_proxy.clamp(0.0, 1.0) * is_callable_t
        cum_p_proxy = torch.cumsum(p_proxy_c, dim=0)
        surv_prev_proxy = (1.0 - (cum_p_proxy - p_proxy_c)).clamp(min=1e-6)
        h_proxy_c = (p_proxy_c / surv_prev_proxy).clamp(min=1e-6, max=1.0 - 1e-6)
        logit_h_proxy = torch.log(h_proxy_c / (1.0 - h_proxy_c))

        hazard_rates = is_callable_t * torch.sigmoid(logit_h_proxy + d_haz)
        survival_probs = torch.cumprod(1.0 - hazard_rates, dim=0)
        surv_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64, device=device), survival_probs[:-1]],
            dim=0,
        )
        p_stop = surv_prev * hazard_rates

        q_proxy_c = baseline.q_proxy.clamp(min=1e-6, max=1.0 - 1e-6)
        logit_q_proxy = torch.log(q_proxy_c / (1.0 - q_proxy_c))
        q_gate = (baseline.q_proxy * baseline.q_proxy) / (baseline.q_proxy * baseline.q_proxy + 1e-10)
        q_pred = q_gate * torch.sigmoid(logit_q_proxy + d_cpn)
        q_cpn = is_guaranteed_t + (1.0 - is_guaranteed_t) * q_pred

        # C-infinity logit-space put-spread head in float64 (SCH-010)
        surv_term = survival_probs[-1]
        k_put_val = max(float(base_state.strike), 0.0)
        k_low_val = max(float(base_state.lower_strike), 0.0) if base_state.lower_strike is not None else 0.0
        k_eff_put = max(k_put_val - k_low_val, 0.0)
        l_put_proxy_c = baseline.l_put_proxy.clamp(min=0.0)
        surv_proxy_term = (1.0 - p_proxy_c.sum()).clamp(min=0.0)
        denom_r = torch.clamp(k_eff_put * surv_proxy_term, min=1e-12)
        eps_r = 1e-12
        r_proxy = torch.clamp(l_put_proxy_c / denom_r, min=eps_r, max=1.0 - eps_r)
        logit_r = torch.log(r_proxy / (1.0 - r_proxy))
        u_put = k_eff_put * surv_term
        l_put = u_put * torch.sigmoid(logit_r + d_put)

        node_feats_post = node_feats
        edge_feats_post = edge_feats
        asset_mask_post = asset_mask
        sched_feats_post = sched_feats
        sched_mask_post = sched_mask
        global_feats_post = global_feats
    else:
        node_feats_post = base_state.node_feats
        edge_feats_post = base_state.edge_feats
        asset_mask_post = base_state.asset_mask
        sched_feats_post = base_state.sched_feats
        sched_mask_post = base_state.sched_mask
        global_feats_post = base_state.global_feats
        p_stop = baseline.p_proxy
        q_cpn = baseline.q_proxy
        l_put = baseline.l_put_proxy
        cum_p_b = torch.cumsum(p_stop * is_callable_t, dim=0)
        survival_probs = (1.0 - cum_p_b).clamp(min=0.0)
        surv_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64, device=device), survival_probs[:-1]], dim=0
        )

    # 9. Assemble W1 cash flows and ProductValue in torch.float64
    seasoned_for_post = base_state.seasoned_state
    if k_cross > 0:
        from dataclasses import replace as _dc_replace

        seasoned_for_post = _dc_replace(
            base_state.seasoned_state,
            future_obs_dates=future_obs_dts,
            future_pay_dates=future_pay_dts,
            as_of_date=eff_as_of if eff_as_of is not None else as_of_orig,
        )

    state_for_post = AutocallPreprocState(
        node_feats=node_feats_post,
        edge_feats=edge_feats_post,
        asset_mask=asset_mask_post,
        sched_feats=sched_feats_post,
        sched_mask=sched_mask_post,
        global_feats=global_feats_post,
        d_assets=d_assets,
        m_rem=m_active,
        d_max=d_pad,
        m_max=m_pad,
        original_identifiers=base_state.original_identifiers,
        sorted_identifiers=sorted_ids,
        lex_perm=base_state.lex_perm,
        inv_lex_perm=base_state.inv_lex_perm,
        spots_t0=spots_sorted,
        ref_spots=ref_spots,
        moneyness_t0=moneyness_t0,
        corr_matrix=corr_sym,
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
        forward_rel=forward_rel,
        forward_perf=forward_perf,
        cash_div_leverage=cash_div_leverage,
        iv_at_fwd=iv_at_fwd_t,
        v_perf=v_perf_t,
        seasoned_state=seasoned_for_post,
        dividend_schedule=div_sched,
        vol_surfaces=base_state.vol_surfaces,
        surface_feature_jacobians=base_state.surface_feature_jacobians,
        iv_at_fwd_jacobians=base_state.iv_at_fwd_jacobians,
        discount_curve=disc_curve_obj,
        baseline=baseline,
        strike=base_state.strike,
        option_gearing=base_state.option_gearing,
        lower_strike=base_state.lower_strike,
        ki_barrier=base_state.ki_barrier,
        i_ki_past=eff_i_ki,
        c_mem_0=eff_c_mem,
        is_memory=base_state.is_memory,
        ko_spread=base_state.ko_spread,
        cpn_spread=base_state.cpn_spread,
        ki_spread=base_state.ki_spread,
        no_notional_repayment=base_state.no_notional_repayment,
        autocall_coupon=base_state.autocall_coupon,
        force_coupons_on_autocall=base_state.force_coupons_on_autocall,
        coupon_barrier_type=base_state.coupon_barrier_type,
        coupon_barrier_direction=base_state.coupon_barrier_direction,
        allow_autocall_on_expiry=base_state.allow_autocall_on_expiry,
        participation_call_strike=base_state.participation_call_strike,
        participation_call_rate=base_state.participation_call_rate,
        one_star_strike=base_state.one_star_strike,
        one_star_coupon=base_state.one_star_coupon,
        is_long_basket=base_state.is_long_basket,
        payment_currency=base_state.payment_currency,
        callability_index=base_state.callability_index,
        coupon_index=base_state.coupon_index,
        option_index=base_state.option_index,
        funding_leg=base_state.funding_leg if (eff_as_of is None and funding_pillar_deltas is None) else None,
        pv_in_transit=base_state.pv_in_transit if eff_as_of is None else 0.0,
        already_autocalled=eff_autocalled,
        trade_info=base_state.trade_info,
        queries=base_state.queries,
        quanto_meta=base_state.quanto_meta,
        asset_financing_curves=base_state.asset_financing_curves,
        discount_curve_id=disc_id,
    )

    df_fin_fn_override: Callable[[Any], torch.Tensor | float] | None = None
    if curve_pillar_deltas is not None or eff_as_of is not None:
        df_fin_memo: dict[Any, torch.Tensor] = {}
        eff_as_of_mult: torch.Tensor | None = None
        if curve_pillar_deltas is not None and eff_as_of is not None:
            eff_as_of_mult = _curve_df_delta_multiplier(
                disc_curve_obj, [eff_as_of], curve_pillar_deltas, as_of_date=as_of_orig
            )[0].clamp(min=1e-15)

        def _df_fin_override(fin_dates: Any) -> torch.Tensor | float:
            is_scalar = not isinstance(fin_dates, (list, tuple, np.ndarray))
            if is_scalar and fin_dates in df_fin_memo:
                return df_fin_memo[fin_dates]
            date_list = [fin_dates] if is_scalar else list(fin_dates)
            base_dfs_np = (
                np.asarray(disc_curve_obj.df(date_list), dtype=np.float64).reshape(-1)
                / disc_anchor_val
            )
            res: torch.Tensor = torch.tensor(base_dfs_np, dtype=torch.float64, device=device)
            if curve_pillar_deltas is not None:
                mult = _curve_df_delta_multiplier(
                    disc_curve_obj, date_list, curve_pillar_deltas, as_of_date=as_of_orig
                )
                if eff_as_of_mult is not None:
                    mult = mult / eff_as_of_mult
                res = res * mult
            if is_scalar:
                val = res[0]
                df_fin_memo[fin_dates] = val
                return val
            for d_item, v_item in zip(date_list, res):
                df_fin_memo[d_item] = v_item
            return res

        df_fin_fn_override = _df_fin_override

    post_out = assemble_autocall_cashflows_and_pv(
        state_for_post,
        p_stop=p_stop,
        q_cpn=q_cpn,
        l_put=l_put,
        l_call=baseline.l_call_proxy,
        l_onestar=baseline.l_onestar_proxy,
        df_fin_fn_override=df_fin_fn_override,
        build_cashflow_items=return_postproc,
    )
    pv_total = post_out.product_value

    # When calendar-rolled or funding_pillar_deltas is active, evaluate pure stateless funding PV.
    if (eff_as_of is not None or funding_pillar_deltas is not None) and base_state.funding_leg is not None:
        f_curve_leg = _normalize_curve_obj(base_state.funding_leg.get("funding_curve"), as_of_date=as_of_orig)
        f_leg_cid = base_state.funding_leg.get("funding_curve_id")
        same_curve_leg = (
            f_curve_leg is None
            or f_curve_leg is base_state.discount_curve
            or (f_leg_cid is not None and disc_id == str(f_leg_cid))
        )
        df_fwd_fn_ov: Callable[[Any], torch.Tensor | float] | None = None
        if not same_curve_leg and f_curve_leg is not None and (funding_pillar_deltas is not None or eff_as_of is not None):
            fwd_anchor_val = (
                max(float(np.asarray(f_curve_leg.df(eff_as_of), dtype=np.float64).reshape(-1)[0]), 1e-15)
                if eff_as_of is not None
                else 1.0
            )

            def _df_fwd_override(fwd_dates: Any) -> torch.Tensor:
                is_scalar = not isinstance(fwd_dates, (list, tuple, np.ndarray))
                date_list = [fwd_dates] if is_scalar else list(fwd_dates)
                base_dfs_np = (
                    np.asarray(f_curve_leg.df(date_list), dtype=np.float64).reshape(-1)
                    / fwd_anchor_val
                )
                res = torch.tensor(base_dfs_np, dtype=torch.float64, device=device)
                if funding_pillar_deltas is not None:
                    mult = _curve_df_delta_multiplier(
                        f_curve_leg, date_list, funding_pillar_deltas, as_of_date=as_of_orig
                    )
                    if eff_as_of is not None:
                        mult = mult / _curve_df_delta_multiplier(
                            f_curve_leg, [eff_as_of], funding_pillar_deltas, as_of_date=as_of_orig
                        )[0].clamp(min=1e-15)
                    res = res * mult
                return res[0] if is_scalar else res

            df_fwd_fn_ov = _df_fwd_override

        w_fund_t = _compute_funding_weights_per_period(
            base_state,
            as_of_override=eff_as_of,
            df_disc_fn=df_fin_fn_override,
            df_fwd_fn=df_fwd_fn_ov,
            period_start_idx=k_cross,
            device=device,
        )
        funding_pv_diff = (surv_prev * w_fund_t).sum()
        pv_total = pv_total - funding_pv_diff
        post_out.funding_pv = funding_pv_diff
        post_out.product_value = pv_total

    # Include rolled or curve-bumped In-Transit PV
    if eff_as_of is not None:
        in_transit_rolled = _rolled_in_transit_pv(
            base_state, disc_curve_obj, eff_as_of, curve_pillar_deltas, device
        )
        pv_total = pv_total + in_transit_rolled + crossed_pv
        post_out.product_value = pv_total
    elif curve_pillar_deltas is not None and base_state.seasoned_state.in_transit_indices:
        in_transit_bumped = _rolled_in_transit_pv(
            base_state, disc_curve_obj, as_of_orig, curve_pillar_deltas, device
        )
        pv_total = pv_total - base_state.pv_in_transit + in_transit_bumped
        post_out.product_value = pv_total

    if return_postproc:
        return pv_total, post_out
    return pv_total


def compute_euler_cash_gamma(
    pv_of_eps_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Evaluates scalar directional Euler CashGamma d^2/d_eps^2 PV(S_0 * (1 + eps))|_{eps=0} in O(1) calls."""
    eps = torch.tensor(0.0, dtype=torch.float64, device=device, requires_grad=True)
    pv = pv_of_eps_fn(eps)
    grad_eps = torch.autograd.grad(pv, eps, create_graph=True)[0]
    d2_pv = torch.autograd.grad(grad_eps, eps, retain_graph=False)[0]
    return d2_pv.to(dtype=torch.float64)


def _extract_repo_curves_map(
    state_or_request: Any,
    sorted_ids: Sequence[str],
    as_of_dt: dt.datetime,
) -> dict[str, Any]:
    """Extracts per-asset RepoCurve objects when a raw request dict/JSON is supplied."""
    out: dict[str, Any] = {}
    if isinstance(state_or_request, AutocallPreprocState):
        existing = getattr(state_or_request.dividend_schedule, "repo_curves", None)
        if isinstance(existing, Mapping):
            for ticker in sorted_ids:
                if ticker in existing and existing[ticker] is not None:
                    out[ticker] = _normalize_curve_obj(existing[ticker], as_of_date=as_of_dt)
        return out

    req_dict: Mapping[str, Any] | None = None
    if isinstance(state_or_request, Mapping):
        req_dict = state_or_request
    elif isinstance(state_or_request, str):
        import json

        try:
            parsed = json.loads(state_or_request)
            if isinstance(parsed, Mapping):
                req_dict = parsed
        except Exception:
            req_dict = None
    if req_dict is None:
        return out

    mkt = req_dict.get("MarketData") or req_dict.get("market_data") or {}
    if not isinstance(mkt, Mapping):
        return out
    repos = mkt.get("RepoCurves") or mkt.get("repo_curves") or mkt.get("Repos") or {}
    if not isinstance(repos, Mapping):
        return out
    from deepvol.market.eq_curves import RepoCurve

    for ticker in sorted_ids:
        spec = repos.get(ticker)
        if spec is None:
            continue
        if hasattr(spec, "df") and callable(spec.df):
            out[ticker] = spec
        elif isinstance(spec, Mapping):
            try:
                out[ticker] = RepoCurve(spec, as_of_date=as_of_dt)
            except Exception:
                out[ticker] = _normalize_curve_obj(spec, as_of_date=as_of_dt)
    return out


def _assemble_pv_batch_torch(
    state: AutocallPreprocState,
    *,
    p_stop: torch.Tensor,
    q_cpn: torch.Tensor,
    l_put: torch.Tensor,
    l_call: torch.Tensor,
    l_onestar: torch.Tensor,
    df_pay_b: torch.Tensor,
    w_fund_b: torch.Tensor | None,
    pv_in_transit_b: torch.Tensor,
) -> torch.Tensor:
    """Vectorized batched W1 affine stopping-time PV assembly matching assemble_autocall_cashflows_and_pv."""
    b_sz, m_rem = p_stop.shape
    device = p_stop.device
    is_call = state.is_callable.to(dtype=torch.float64, device=device).unsqueeze(0)
    p_t = p_stop * is_call
    cum_p = torch.cumsum(p_t, dim=1)
    surv = (1.0 - cum_p).clamp(min=0.0)
    surv_prev = torch.cat(
        [torch.ones((b_sz, 1), dtype=torch.float64, device=device), surv[:, :-1]],
        dim=1,
    )

    if state.no_notional_repayment:
        notional_pv = torch.zeros(b_sz, dtype=torch.float64, device=device)
    else:
        notional_pv = (df_pay_b * p_t).sum(dim=1) + df_pay_b[:, -1] * surv[:, -1]

    cpn_rates = state.coupon_amounts.to(dtype=torch.float64, device=device)
    if state.force_coupons_on_autocall:
        eff_q = torch.maximum(q_cpn, p_t / surv_prev.clamp(min=1e-12))
    else:
        eff_q = q_cpn

    if state.is_memory:
        mem_acc = torch.full((b_sz,), float(state.c_mem_0), dtype=torch.float64, device=device)
        cpn_cfs: list[torch.Tensor] = []
        ac_cpn = float(state.autocall_coupon)
        for m in range(m_rem):
            eff_c_m = cpn_rates[m] + mem_acc
            eq_m = eff_q[:, m]
            cpn_cfs.append(df_pay_b[:, m] * (surv_prev[:, m] * eq_m * eff_c_m + p_t[:, m] * ac_cpn))
            mem_acc = eff_c_m * (1.0 - eq_m)
        coupon_pv = torch.stack(cpn_cfs, dim=1).sum(dim=1)
    else:
        cpn_undisc = surv_prev * eff_q * cpn_rates.unsqueeze(0) + p_t * float(state.autocall_coupon)
        coupon_pv = (df_pay_b * cpn_undisc).sum(dim=1)

    put_sign = 1.0 if state.is_long_basket else -1.0
    option_pv = df_pay_b[:, -1] * (
        put_sign * float(state.option_gearing) * l_put
        + float(state.participation_call_rate) * l_call
        + l_onestar
    )
    funding_pv = (
        (surv_prev * w_fund_b).sum(dim=1)
        if w_fund_b is not None
        else torch.zeros(b_sz, dtype=torch.float64, device=device)
    )
    return notional_pv + coupon_pv + option_pv - funding_pv + pv_in_transit_b


def _escrowed_spot_softplus_np(
    spots_col: npt.NDArray[np.float64], div_pv: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """NumPy float64 match of _escrowed_spot_softplus."""
    floor_sp = 0.01 * spots_col
    scale_sp = np.maximum(floor_sp, 1e-12)
    z = (spots_col - div_pv - floor_sp) / scale_sp
    sp = np.where(z > 40.0, z, np.log1p(np.exp(np.minimum(z, 40.0))))
    return np.asarray(floor_sp + scale_sp * sp, dtype=np.float64)


def _try_eval_fused_pre_surr(
    state: AutocallPreprocState,
    *,
    spots_var: torch.Tensor,
    spots_gamma: torch.Tensor,
    corr_var: torch.Tensor,
    div_vars: Mapping[str, torch.Tensor] | None,
    vol_deltas: Mapping[str, torch.Tensor] | None,
    pillar_deltas: torch.Tensor | None,
    funding_pillar_deltas: torch.Tensor | None,
    asset_curve_deltas: Mapping[str, torch.Tensor] | None,
    repo_deltas: Mapping[str, torch.Tensor] | None,
    eff_surrogate: MaskedEQAutocallSurrogate | None,
    include_theta: bool,
    iv_0_delta_leaf: torch.Tensor | None,
    div_pv_delta_leaf: torch.Tensor | None,
    df_pay_mult_leaf: torch.Tensor | None,
    df_fin_mult_leaf: torch.Tensor | None,
    df_repo_mult_leaf: torch.Tensor | None,
    w_fund_delta_leaf: torch.Tensor | None,
    in_transit_delta_leaf: torch.Tensor | None,
    out_aux: dict[str, Any] | None,
) -> tuple[Any, ...] | None:
    """Fast fused C++23 + NumPy pre-surrogate pipeline for standard pure Tier-1 surrogate calls."""
    if (
        eff_surrogate is None
        or vol_deltas is not None
        or pillar_deltas is not None
        or (div_vars is not None and len(div_vars) > 0)
        or funding_pillar_deltas is not None
        or (asset_curve_deltas is not None and len(asset_curve_deltas) > 0)
        or (repo_deltas is not None and len(repo_deltas) > 0)
        or state.d_assets <= 0
        or state.d_assets > 32
        or bool(getattr(state, "quanto_meta", None))
    ):
        return None
    if (
        (state.participation_call_strike is not None and float(state.participation_call_strike) > 0.0 and float(state.participation_call_rate) != 0.0)
        or (state.one_star_strike is not None and float(state.one_star_strike) > 0.0 and float(state.one_star_coupon) != 0.0)
    ):
        return None

    call_code = monitoring_index_to_code(state.callability_index)
    cpn_code = monitoring_index_to_code(state.coupon_index or state.callability_index)
    opt_code = monitoring_index_to_code(state.option_index or state.callability_index)
    if call_code not in (0, 1) or cpn_code not in (0, 1) or opt_code not in (0, 1):
        return None

    if (
        _load_tier1_cpp_ad_fn() is None
        or _SURFACE_CPP_AD_FN is None
        or _SURR_FEAT_CPP_FN is None
        or _SURR_POST_CPP_FN is None
    ):
        return None

    device = spots_var.device
    d_assets = state.d_assets
    m_rem = state.m_rem
    sorted_ids = state.sorted_identifiers
    surf_caches_b: dict[str, tuple[_SurfaceGeomCache, _SurfaceQuoteCache]] = {
        ticker: _get_surface_torch_caches(state.vol_surfaces[ticker], device) for ticker in sorted_ids
    }
    if not all(
        surf_caches_b[t][1].slice_c_np is not None
        and surf_caches_b[t][0].n_unique >= 2
        and surf_caches_b[t][0].strikes_np is not None
        and surf_caches_b[t][0].tenors_np is not None
        for t in sorted_ids
    ):
        return None

    div_sched = state.dividend_schedule
    ref_spots_np = np.ascontiguousarray(state.ref_spots.detach().cpu().numpy(), dtype=np.float64)
    div_pv_base_np = np.ascontiguousarray(div_sched.div_pv.detach().cpu().numpy(), dtype=np.float64)
    r_tot_base_np = np.ascontiguousarray(div_sched.r_tot_obs.detach().cpu().numpy(), dtype=np.float64)
    df_fin_base_np = np.ascontiguousarray(div_sched.df_fin_obs.detach().cpu().numpy(), dtype=np.float64)
    df_pay_base_np = np.ascontiguousarray(state.df_pay.detach().cpu().numpy(), dtype=np.float64)
    repo_spr_base_np = np.ascontiguousarray(div_sched.repo_spread_avg.detach().cpu().numpy(), dtype=np.float64)
    tau_obs_t = state.tau_obs.to(dtype=torch.float64, device=device)
    delta_tau_t = state.delta_tau.to(dtype=torch.float64, device=device)
    tau_obs_np = np.ascontiguousarray(tau_obs_t.detach().cpu().numpy(), dtype=np.float64)
    dtau_np_s = np.ascontiguousarray(delta_tau_t.detach().cpu().numpy(), dtype=np.float64)

    as_of_orig = state.seasoned_state.as_of_date
    disc_curve_obj = _normalize_curve_obj(state.discount_curve, as_of_date=as_of_orig)
    asset_fin_curves = [
        _normalize_curve_obj(state.asset_financing_curves.get(ticker, disc_curve_obj), as_of_date=as_of_orig)
        for ticker in sorted_ids
    ]

    sqrt_tau_np = np.sqrt(np.maximum(tau_obs_np, 1e-12))
    t_1_q_np = np.ascontiguousarray(
        np.concatenate([tau_obs_np, np.full(5, tau_obs_np[-1], dtype=np.float64)]), dtype=np.float64
    )
    t_1_q_ptr = int(t_1_q_np.__array_interface__["data"][0])
    asset_metas_b = tuple((state.vol_surfaces[t], surf_caches_b[t][0], surf_caches_b[t][1]) for t in sorted_ids)

    ko_t = state.ko_barriers.to(dtype=torch.float64, device=device)
    ic_t = state.is_callable.to(dtype=torch.float64, device=device)
    cb_t = state.cpn_barriers.to(dtype=torch.float64, device=device)
    ig_t = state.is_guaranteed_cpn.to(dtype=torch.float64, device=device)
    cpn_t = state.coupon_amounts.to(dtype=torch.float64, device=device)
    ko_np_s = np.ascontiguousarray(ko_t.detach().cpu().numpy(), dtype=np.float64)
    ic_np_s = np.ascontiguousarray(ic_t.detach().cpu().numpy(), dtype=np.float64)
    cb_np_s = np.ascontiguousarray(cb_t.detach().cpu().numpy(), dtype=np.float64)
    ig_np_s = np.ascontiguousarray(ig_t.detach().cpu().numpy(), dtype=np.float64)
    cpn_np_s = np.ascontiguousarray(cpn_t.detach().cpu().numpy(), dtype=np.float64)
    ic_arr_i32 = np.array([1 if x > 0.5 else 0 for x in ic_np_s.tolist()], dtype=np.int32)

    has_ki_b = state.ki_barrier is not None and float(state.ki_barrier) > 0.0 and float(state.i_ki_past) < 0.5
    ki_b_val = float(state.ki_barrier) if state.ki_barrier is not None else 0.0
    tier1_meta: tuple[Any, ...] = (
        int(call_code),
        int(cpn_code),
        int(opt_code),
        1 if str(state.coupon_barrier_direction).strip().lower() == "down" else 0,
        1 if str(state.coupon_barrier_type).strip().lower() == "out" else 0,
        1 if bool(state.force_coupons_on_autocall) else 0,
        1 if bool(state.allow_autocall_on_expiry) else 0,
        float(state.ko_spread),
        float(state.cpn_spread),
        float(state.ki_spread),
        float(state.strike),
        float(state.option_gearing),
        float(state.lower_strike) if state.lower_strike is not None else 0.0,
        1 if has_ki_b else 0,
        ki_b_val,
        ic_arr_i32,
    )

    sb = state.seasoned_state.sentinel_barriers
    if sb is None:
        raise RuntimeError("Seasoned state sentinel_barriers must be initialized before evaluating PV.")
    surr_d_node = int(getattr(eff_surrogate, "d_node", len(AutocallNodeCol)))
    surr_d_global = int(getattr(eff_surrogate, "d_global", len(AutocallGlobalCol)))

    role_a_np = np.zeros((d_assets,), dtype=np.float64)
    role_b_np = np.zeros((d_assets,), dtype=np.float64)
    active_idx_names = (
        state.callability_index,
        state.coupon_index or state.callability_index,
        state.option_index or state.callability_index,
    )
    if d_assets >= 2 and any(nm in PAIR_ORDER_DEPENDENT_INDICES for nm in active_idx_names):
        role_a_np[int(state.inv_lex_perm[0])] = 1.0
        role_b_np[int(state.inv_lex_perm[1])] = 1.0
    has_ki_f = 1.0 if (state.ki_barrier is not None and state.ki_barrier > 0.0) else 0.0
    b_ki_v = float(state.ki_barrier) if (state.ki_barrier is not None and state.ki_barrier > 0.0) else float(state.strike)
    lk_v = float(state.lower_strike) if state.lower_strike is not None else 0.0
    g_static_np = np.zeros((len(AutocallGlobalCol),), dtype=np.float64)
    g_static_np[AutocallGlobalCol.K_PUT] = float(state.strike)
    g_static_np[AutocallGlobalCol.B_KI] = b_ki_v
    g_static_np[AutocallGlobalCol.HAS_KI_FLAG] = has_ki_f
    g_static_np[AutocallGlobalCol.I_KI_PAST] = float(state.i_ki_past)
    g_static_np[AutocallGlobalCol.G_PUT] = float(state.option_gearing)
    g_static_np[AutocallGlobalCol.C_MEM_0] = float(state.c_mem_0)
    g_static_np[AutocallGlobalCol.IS_MEMORY_FLAG] = 1.0 if state.is_memory else 0.0
    g_static_np[AutocallGlobalCol.ABS_KO_SPREAD] = abs(float(state.ko_spread))
    g_static_np[AutocallGlobalCol.ABS_CPN_SPREAD] = abs(float(state.cpn_spread))
    g_static_np[AutocallGlobalCol.LOWER_STRIKE] = lk_v
    g_static_np[AutocallGlobalCol.CALL_INDEX_ID] = float(call_code)
    g_static_np[AutocallGlobalCol.CPN_INDEX_ID] = float(cpn_code)
    g_static_np[AutocallGlobalCol.OPT_INDEX_ID] = float(opt_code)
    g_static_np[AutocallGlobalCol.IS_LONG_BASKET] = 1.0 if state.is_long_basket else 0.0
    g_static_np[AutocallGlobalCol.ABS_KI_SPREAD] = abs(float(state.ki_spread))
    lko_np_s = np.ascontiguousarray(sb.log_ko_barrier_feat, dtype=np.float64)
    lcp_np_s = np.ascontiguousarray(sb.log_cpn_barrier_feat, dtype=np.float64)
    feat_static_np = (
        tau_obs_np, dtau_np_s, lko_np_s, ic_np_s, lcp_np_s, ig_np_s, cpn_np_s,
        role_a_np, role_b_np, g_static_np, surr_d_node, surr_d_global, max(float(state.strike), 1e-4),
    )

    b_g = int(spots_gamma.shape[0])
    b_diff = 1 + b_g

    # Optional fast Theta row with static rolled curve/div caching
    theta_row_data: dict[str, Any] | None = None
    theta_pack: tuple[Any, ...] | None = None
    if include_theta and isinstance(as_of_orig, dt.datetime):
        as_of_rolled = as_of_orig + dt.timedelta(days=1)
        k_cross_th, eff_ac_th, _, _, _ = _collapse_crossed_periods_for_roll(
            state, disc_curve_obj, as_of_rolled, state.spots_t0.detach().cpu().numpy(), ref_spots_np
        )
        if k_cross_th == 0 and not eff_ac_th:
            future_obs_dts = state.seasoned_state.future_obs_dates
            future_pay_dts = state.seasoned_state.future_pay_dates
            time_dcc = state.seasoned_state.day_count_convention

            def _c_dig(c_obj: Any) -> str | None:
                if c_obj is None or not hasattr(c_obj, "df"):
                    return None
                return _fast_array_digest(
                    getattr(c_obj, "tenors", None),
                    getattr(c_obj, "_raw_quotes", getattr(c_obj, "quotes", None)),
                    getattr(c_obj, "dcc", ""),
                    getattr(c_obj, "space_coords", ""),
                    getattr(c_obj, "_input_mode", ""),
                )

            c_dig_memo: dict[int, str | None] = {}
            fin_digs: list[str | None] = []
            for c_i in asset_fin_curves:
                ci_id = id(c_i)
                if ci_id not in c_dig_memo:
                    c_dig_memo[ci_id] = _c_dig(c_i)
                fin_digs.append(c_dig_memo[ci_id])
            th_key = (
                "th_roll",
                as_of_rolled,
                time_dcc,
                tuple(future_obs_dts),
                tuple(future_pay_dts),
                _c_dig(disc_curve_obj),
                tuple(fin_digs),
                _fast_array_digest(div_pv_base_np, df_fin_base_np, df_pay_base_np, r_tot_base_np),
                tuple((t, len(div_sched.dividend_events.get(t, ()))) for t in sorted_ids),
            )
            th_cached = _THETA_CURVE_CACHE.get(th_key)
            if th_cached is not None:
                tau_obs_th_np, delta_tau_th_np, df_pay_th_np, div_pv_th_np, df_fin_th_np = th_cached
            else:
                tau_obs_th_np = np.maximum(
                    np.array([_year_fraction(as_of_rolled, d, time_dcc) for d in future_obs_dts], dtype=np.float64),
                    1e-6,
                )
                delta_tau_th_np = (
                    np.concatenate([tau_obs_th_np[:1], np.maximum(tau_obs_th_np[1:] - tau_obs_th_np[:-1], 1e-8)])
                    if m_rem > 1
                    else tau_obs_th_np[:1].copy()
                )
                disc_anchor_th = (
                    max(float(_curve_df_cached_np(disc_curve_obj, [as_of_rolled])[0]), 1e-15)
                    if (disc_curve_obj is not None and hasattr(disc_curve_obj, "df"))
                    else 1.0
                )
                df_pay_th_np = (
                    np.ascontiguousarray(_curve_df_cached_np(disc_curve_obj, future_pay_dts) / disc_anchor_th, dtype=np.float64)
                    if (disc_curve_obj is not None and hasattr(disc_curve_obj, "df"))
                    else df_pay_base_np.copy()
                )
                div_rows_th: list[npt.NDArray[np.float64]] = []
                df_fin_rows_th: list[npt.NDArray[np.float64]] = []
                anc_memo_th: dict[int, float] = {}
                obs_df_memo_th: dict[int, npt.NDArray[np.float64]] = {}
                for i, ticker in enumerate(sorted_ids):
                    c_i = asset_fin_curves[i]
                    has_df_i = c_i is not None and hasattr(c_i, "df")
                    c_id_i = id(c_i)
                    if has_df_i and c_id_i not in anc_memo_th:
                        anc_memo_th[c_id_i] = max(float(_curve_df_cached_np(c_i, [as_of_rolled])[0]), 1e-15)
                    c_anchor_i = anc_memo_th.get(c_id_i, 1.0)
                    events_i = div_sched.dividend_events.get(ticker, [])
                    if len(events_i) > 0:
                        amt_np = div_sched.dividend_amount_tensors[ticker].detach().cpu().numpy()
                        w_np = div_sched.dividend_pv_weights[ticker].detach().cpu().numpy()
                        inc_np = div_sched.dividend_obs_incidence[ticker].detach().cpu().numpy()
                        val_np = np.array([1.0 if ev.ex_date > as_of_rolled else 0.0 for ev in events_i], dtype=np.float64)
                        div_rows_th.append(np.matmul(amt_np * w_np * (val_np / c_anchor_i), inc_np))
                    else:
                        div_rows_th.append(np.zeros(m_rem, dtype=np.float64))
                    if has_df_i:
                        if c_id_i not in obs_df_memo_th:
                            obs_df_memo_th[c_id_i] = _curve_df_cached_np(c_i, future_obs_dts)
                        df_fin_rows_th.append(np.maximum(obs_df_memo_th[c_id_i] / c_anchor_i, 1e-15))
                    else:
                        df_fin_rows_th.append(np.maximum(df_fin_base_np[i], 1e-15))
                div_pv_th_np = np.ascontiguousarray(np.stack(div_rows_th, axis=0), dtype=np.float64)
                df_fin_th_np = np.ascontiguousarray(np.stack(df_fin_rows_th, axis=0), dtype=np.float64)
                _THETA_CURVE_CACHE.put(th_key, (tau_obs_th_np, delta_tau_th_np, df_pay_th_np, div_pv_th_np, df_fin_th_np))

            sp0_state_np = np.ascontiguousarray(state.spots_t0.detach().cpu().numpy(), dtype=np.float64)
            tau_ratio_th = (tau_obs_th_np / np.maximum(tau_obs_np, 1e-12))[None, :]
            growth_th = np.power(np.maximum(np.exp(r_tot_base_np) * df_fin_base_np, 1e-15), tau_ratio_th) / df_fin_th_np
            es_th = _escrowed_spot_softplus_np(sp0_state_np[:, None], div_pv_th_np)
            fp_price_th = es_th * growth_th
            fp_rel_th_np = np.ascontiguousarray((fp_price_th / sp0_state_np[:, None])[None, :, :], dtype=np.float64)
            fp_perf_th_2d = fp_price_th / ref_spots_np[:, None]
            fp_perf_th_np = np.ascontiguousarray(fp_perf_th_2d[None, :, :], dtype=np.float64)
            cd_lev_th_np = np.ascontiguousarray((sp0_state_np[:, None] / es_th)[None, :, :], dtype=np.float64)
            m0_th_np = np.ascontiguousarray((sp0_state_np / ref_spots_np)[None, :], dtype=np.float64)

            t_1_th_np = np.ascontiguousarray(
                np.concatenate([tau_obs_th_np, np.full(5, tau_obs_th_np[-1], dtype=np.float64)]), dtype=np.float64
            )
            q_len_th = int(t_1_th_np.shape[0])
            fst_th = fp_price_th[:, -1:]
            k_th_np = np.ascontiguousarray(
                np.concatenate([fp_price_th, fst_th, 0.90 * fst_th, 1.10 * fst_th, 0.80 * fst_th, 1.20 * fst_th], axis=1),
                dtype=np.float64,
            )
            iv_th_np = np.empty((d_assets, q_len_th), dtype=np.float64)
            dk_th_np = np.empty((d_assets, q_len_th), dtype=np.float64)
            s_fn_th = _SURFACE_CPP_AD_FN
            assert s_fn_th is not None
            t_th_ptr = int(t_1_th_np.__array_interface__["data"][0])
            k_th_ptr = int(k_th_np.__array_interface__["data"][0])
            iv_th_ptr = int(iv_th_np.__array_interface__["data"][0])
            dk_th_ptr = int(dk_th_np.__array_interface__["data"][0])
            row_th_b = q_len_th * 8
            for i in range(d_assets):
                surf_i, geom_i, qc_i = asset_metas_b[i]
                off_i = i * row_th_b
                s_fn_th(
                    q_len_th, geom_i.n_unique, geom_i.n_exp,
                    geom_i.k_min, geom_i.k_max, float(surf_i.min_vol),
                    geom_i.strikes_ptr, geom_i.tenors_ptr, qc_i.quotes_ptr,
                    qc_i.slice_c_ptr, qc_i.slope_left_ptr, qc_i.slope_right_ptr,
                    t_th_ptr, k_th_ptr + off_i, iv_th_ptr + off_i, dk_th_ptr + off_i,
                )
            vp_th_np = np.ascontiguousarray((iv_th_np[:, :m_rem] * np.sqrt(np.maximum(tau_obs_th_np, 1e-12))[None, :])[None, :, :], dtype=np.float64)

            eye_d_th = np.eye(d_assets, dtype=np.float64)
            c_in_th = np.ascontiguousarray(corr_var.detach().cpu().numpy(), dtype=np.float64)
            c_raw_th = 0.5 * (c_in_th + c_in_th.T) * (1.0 - eye_d_th) + eye_d_th
            abs_ct = np.abs(c_raw_th)
            c_sym_th = np.ascontiguousarray(
                (np.where(abs_ct <= 0.999, c_raw_th, np.sign(c_raw_th) * (0.999 + 0.0009 * np.tanh((abs_ct - 0.999) / 0.0009))) * (1.0 - eye_d_th) + eye_d_th)[None, :, :],
                dtype=np.float64,
            )
            fs_th = np.maximum(fp_perf_th_2d.T, 1e-16)
            oa_th = np.argsort(fs_th, axis=1, kind="mergesort") if d_assets > 1 else np.zeros((m_rem, 1), dtype=np.int64)
            od_th = np.ascontiguousarray(oa_th[:, ::-1], dtype=np.int64) if d_assets > 1 else oa_th
            oc_th = od_th if call_code == 1 else oa_th
            ocp_th = od_th if cpn_code == 1 else oa_th
            oo_th = np.ascontiguousarray(oa_th[-1:, ::-1], dtype=np.int64) if (opt_code == 1 and d_assets > 1) else np.ascontiguousarray(oa_th[-1:, :], dtype=np.int64)
            fw_th, vw_th, p_th, q_th, lp_th, dw_th = _run_tier1_cpp_forward_numpy(
                fp_perf_th_np, vp_th_np, c_sym_th, ko_np_s, ic_arr_i32, cb_np_s, ig_np_s,
                oc_th, ocp_th, oo_th, tier1_meta[:-1],
            )
            w_fund_th_np: npt.NDArray[np.float64] | None = None
            if state.funding_leg is not None:
                df_fin_th_ov: Callable[[Any], torch.Tensor | float] | None = None
                if disc_curve_obj is not None and hasattr(disc_curve_obj, "df"):
                    disc_anc_f = max(float(_curve_df_cached_np(disc_curve_obj, [as_of_rolled])[0]), 1e-15)
                    memo_th_f: dict[Any, float] = {}

                    def _df_th_f(f_dts: Any) -> torch.Tensor | float:
                        if not isinstance(f_dts, (list, tuple, np.ndarray)) and f_dts in memo_th_f:
                            return memo_th_f[f_dts]
                        dl_f = [f_dts] if not isinstance(f_dts, (list, tuple, np.ndarray)) else list(f_dts)
                        val_f = _curve_df_cached_np(disc_curve_obj, dl_f) / disc_anc_f
                        if not isinstance(f_dts, (list, tuple, np.ndarray)):
                            fv = float(val_f[0])
                            memo_th_f[f_dts] = fv
                            return fv
                        return torch.tensor(val_f, dtype=torch.float64, device=device)

                    df_fin_th_ov = _df_th_f
                wf_t_th = _compute_funding_weights_per_period(
                    state, as_of_override=as_of_rolled, df_disc_fn=df_fin_th_ov, df_fwd_fn=None, period_start_idx=0, device=device
                )
                w_fund_th_np = np.ascontiguousarray(wf_t_th.detach().cpu().numpy(), dtype=np.float64)
            if state.seasoned_state.in_transit_indices or state.seasoned_state.past_cashflows:
                in_transit_th_f = float(_rolled_in_transit_pv(state, disc_curve_obj, as_of_rolled, None, device).detach().cpu().item())
            else:
                in_transit_th_f = 0.0

            theta_pack = (
                m0_th_np, fp_rel_th_np, fp_perf_th_np, cd_lev_th_np, vp_th_np,
                np.ascontiguousarray(iv_th_np[None, :, :], dtype=np.float64),
                dw_th,
                np.ascontiguousarray(df_pay_th_np[None, :], dtype=np.float64),
                fw_th, vw_th, p_th, q_th, lp_th, tau_obs_th_np, delta_tau_th_np,
            )
            theta_row_data = {
                "cpp_fast": True,
                "p_np": p_th[0],
                "q_np": q_th[0],
                "lp_f": float(lp_th[0]),
                "df_pay_np": df_pay_th_np,
                "w_fund_np": w_fund_th_np,
                "in_transit_f": in_transit_th_f,
            }

    static_ctx = (
        ref_spots_np, div_pv_base_np, r_tot_base_np, df_fin_base_np, df_pay_base_np,
        repo_spr_base_np, tau_obs_np, sqrt_tau_np, t_1_q_np, t_1_q_ptr,
        asset_metas_b, tier1_meta, ko_np_s, cb_np_s, ig_np_s, feat_static_np, theta_pack,
    )
    box: list[Any] = []
    k0_box: list[Any] = []
    nf_all, ef_all, sf_all, gf_all, p_proxy_b, q_proxy_b, l_put_proxy_b, df_pay_diff_b = _PreSurrFusedFn.apply(  # type: ignore[no-untyped-call]
        spots_var,
        spots_gamma,
        corr_var,
        iv_0_delta_leaf,
        div_pv_delta_leaf,
        df_pay_mult_leaf,
        df_fin_mult_leaf,
        df_repo_mult_leaf,
        static_ctx,
        box,
        k0_box,
    )
    b_tot = int(nf_all.shape[0])
    am_all = torch.ones((b_tot, d_assets), dtype=torch.float64, device=device)
    sm_all = torch.ones((b_tot, m_rem), dtype=torch.float64, device=device)
    l_call_b = torch.zeros(b_diff, dtype=torch.float64, device=device)
    l_onestar_b = torch.zeros(b_diff, dtype=torch.float64, device=device)

    w_fund_b: torch.Tensor | None = None
    if state.funding_leg is not None:
        w_fund_0 = _compute_funding_weights_per_period(
            state, df_disc_fn=None, df_fwd_fn=None, period_start_idx=0, device=device
        )
        if w_fund_delta_leaf is not None:
            w_fund_0 = w_fund_0 + w_fund_delta_leaf
        w_fund_b = torch.cat([w_fund_0.unsqueeze(0), w_fund_0.detach().unsqueeze(0).expand(b_g, m_rem)], dim=0)

    in_transit_0 = torch.tensor(float(state.pv_in_transit), dtype=torch.float64, device=device)
    if in_transit_delta_leaf is not None:
        in_transit_0 = in_transit_0 + in_transit_delta_leaf
    pv_in_transit_b = torch.cat([in_transit_0.unsqueeze(0), in_transit_0.detach().expand(b_g)], dim=0)

    if out_aux is not None:
        k0_arr = k0_box.pop()
        out_aux["k_0_surf"] = [torch.from_numpy(k0_arr[i].copy()).to(device=device) for i in range(d_assets)]
        out_aux["k_0_q"] = [None] * d_assets
        out_aux["t_1_q"] = torch.from_numpy(t_1_q_np.copy()).to(device=device)
        out_aux["tau_obs_t"] = tau_obs_t
        out_aux["surf_caches_b"] = surf_caches_b
        out_aux["asset_fin_curves"] = asset_fin_curves
        out_aux["disc_curve_obj"] = disc_curve_obj
        out_aux["as_of_orig"] = as_of_orig

    k_eff_put_v = max(
        max(float(state.strike), 0.0) - (max(float(state.lower_strike), 0.0) if state.lower_strike is not None else 0.0),
        0.0,
    )
    return (
        nf_all, ef_all, am_all, sf_all, sm_all, gf_all,
        p_proxy_b, q_proxy_b, l_put_proxy_b, l_call_b, l_onestar_b,
        df_pay_diff_b, w_fund_b, pv_in_transit_b, theta_row_data,
        ic_t, ig_t, cpn_t, ic_np_s, ig_np_s, cpn_np_s, k_eff_put_v, b_diff,
    )


def _evaluate_greeks_single_batch(
    state: AutocallPreprocState,
    *,
    spots_var: torch.Tensor,
    spots_gamma: torch.Tensor,
    corr_var: torch.Tensor,
    div_vars: Mapping[str, torch.Tensor] | None,
    vol_deltas: Mapping[str, torch.Tensor] | None,
    pillar_deltas: torch.Tensor | None,
    funding_pillar_deltas: torch.Tensor | None,
    asset_curve_deltas: Mapping[str, torch.Tensor] | None,
    repo_deltas: Mapping[str, torch.Tensor] | None,
    repo_curves: Mapping[str, Any] | None,
    fx_vol_deltas: Mapping[str, torch.Tensor] | None,
    fx_corr_deltas: Mapping[str, torch.Tensor] | None,
    eff_surrogate: MaskedEQAutocallSurrogate | None,
    include_theta: bool,
    iv_0_delta_leaf: torch.Tensor | None = None,
    iv_q0_delta_leaf: torch.Tensor | None = None,
    div_pv_delta_leaf: torch.Tensor | None = None,
    df_pay_mult_leaf: torch.Tensor | None = None,
    df_fin_mult_leaf: torch.Tensor | None = None,
    df_repo_mult_leaf: torch.Tensor | None = None,
    w_fund_delta_leaf: torch.Tensor | None = None,
    in_transit_delta_leaf: torch.Tensor | None = None,
    out_aux: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, AutocallPostprocOutput, torch.Tensor | None]:
    """Evaluates Row 0 (full 1st-order AD), Rows 1..B_g (spot bumps), and optional Theta row in one batch."""
    fused_res = _try_eval_fused_pre_surr(
        state,
        spots_var=spots_var,
        spots_gamma=spots_gamma,
        corr_var=corr_var,
        div_vars=div_vars,
        vol_deltas=vol_deltas,
        pillar_deltas=pillar_deltas,
        funding_pillar_deltas=funding_pillar_deltas,
        asset_curve_deltas=asset_curve_deltas,
        repo_deltas=repo_deltas,
        eff_surrogate=eff_surrogate,
        include_theta=include_theta,
        iv_0_delta_leaf=iv_0_delta_leaf,
        div_pv_delta_leaf=div_pv_delta_leaf,
        df_pay_mult_leaf=df_pay_mult_leaf,
        df_fin_mult_leaf=df_fin_mult_leaf,
        df_repo_mult_leaf=df_repo_mult_leaf,
        w_fund_delta_leaf=w_fund_delta_leaf,
        in_transit_delta_leaf=in_transit_delta_leaf,
        out_aux=out_aux,
    )
    if fused_res is not None:
        (
            f_nf, f_ef, f_am, f_sf, f_sm, f_gf,
            f_ppx, f_qpx, f_lppx, f_lc, f_lo,
            f_dfp, f_wf, f_pvit, f_th_data,
            f_ic, f_ig, f_cpn, f_ic_np, f_ig_np, f_cpn_np, f_keff, f_bdiff,
        ) = fused_res
        return _eval_post_surr_tail(
            state=state,
            device=spots_var.device,
            m_rem=state.m_rem,
            b_diff=f_bdiff,
            eff_surrogate=eff_surrogate,
            use_cpp_glue=True,
            nf_all=f_nf,
            ef_all=f_ef,
            am_all=f_am,
            sf_all=f_sf,
            sm_all=f_sm,
            gf_all=f_gf,
            p_proxy_b=f_ppx,
            q_proxy_b=f_qpx,
            l_put_proxy_b=f_lppx,
            l_call_b=f_lc,
            l_onestar_b=f_lo,
            df_pay_diff_b=f_dfp,
            w_fund_b=f_wf,
            pv_in_transit_b=f_pvit,
            theta_row_data=f_th_data,
            ic_t=f_ic,
            ig_t=f_ig,
            cpn_t=f_cpn,
            ic_np_s=f_ic_np,
            ig_np_s=f_ig_np,
            cpn_np_s=f_cpn_np,
            k_eff_put_v=f_keff,
        )
    device = spots_var.device
    d_assets = state.d_assets
    m_rem = state.m_rem
    sorted_ids = state.sorted_identifiers
    as_of_orig = state.seasoned_state.as_of_date
    disc_curve_obj = _normalize_curve_obj(state.discount_curve, as_of_date=as_of_orig)
    disc_id = str(getattr(disc_curve_obj, "identifier", getattr(disc_curve_obj, "Identifier", state.discount_curve_id or "DEFAULT")))

    b_g = int(spots_gamma.shape[0])
    b_diff = 1 + b_g
    spots_all = torch.cat([spots_var.unsqueeze(0), spots_gamma], dim=0)  # (b_diff, D)
    ref_spots = state.ref_spots.to(dtype=torch.float64, device=device)
    moneyness_all = spots_all / ref_spots.unsqueeze(0)

    tau_obs_t = state.tau_obs.to(dtype=torch.float64, device=device)
    delta_tau_t = state.delta_tau.to(dtype=torch.float64, device=device)
    df_pay_0 = state.df_pay.to(dtype=torch.float64, device=device)
    df_obs_0 = state.df_obs.to(dtype=torch.float64, device=device)
    future_obs_dts = state.seasoned_state.future_obs_dates
    future_pay_dts = state.seasoned_state.future_pay_dates
    if pillar_deltas is not None:
        df_obs_0 = df_obs_0 * _curve_df_delta_multiplier(
            disc_curve_obj, future_obs_dts, pillar_deltas, as_of_date=as_of_orig
        )
        df_pay_0 = df_pay_0 * _curve_df_delta_multiplier(
            disc_curve_obj, future_pay_dts, pillar_deltas, as_of_date=as_of_orig
        )
    elif df_pay_mult_leaf is not None:
        df_pay_0 = df_pay_0 * df_pay_mult_leaf

    f_leg_meta = state.funding_leg or {}
    f_leg_curve_id = f_leg_meta.get("funding_curve_id")
    asset_fin_deltas: list[torch.Tensor | None] = []
    asset_fin_curves: list[Any] = []
    for ticker in sorted_ids:
        c_i = _normalize_curve_obj(
            state.asset_financing_curves.get(ticker, disc_curve_obj), as_of_date=as_of_orig
        )
        asset_fin_curves.append(c_i)
        c_i_id = str(getattr(c_i, "identifier", getattr(c_i, "Identifier", disc_id)))
        if asset_curve_deltas is not None and ticker in asset_curve_deltas:
            asset_fin_deltas.append(asset_curve_deltas[ticker])
        elif c_i is disc_curve_obj or c_i_id == disc_id:
            asset_fin_deltas.append(pillar_deltas)
        elif funding_pillar_deltas is not None and f_leg_curve_id is not None and c_i_id == str(f_leg_curve_id):
            asset_fin_deltas.append(funding_pillar_deltas)
        else:
            asset_fin_deltas.append(None)

    div_sched = state.dividend_schedule
    need_div_recomp = bool(
        (div_vars is not None and len(div_vars) > 0)
        or pillar_deltas is not None
        or funding_pillar_deltas is not None
        or (asset_curve_deltas is not None and len(asset_curve_deltas) > 0)
        or (repo_deltas is not None and len(repo_deltas) > 0)
    )
    if not need_div_recomp:
        div_pv_0 = div_sched.div_pv.to(dtype=torch.float64, device=device)
        if div_pv_delta_leaf is not None:
            div_pv_0 = div_pv_0 + div_pv_delta_leaf
    else:
        div_pv_rows: list[torch.Tensor] = []
        for i, ticker in enumerate(sorted_ids):
            events_i = div_sched.dividend_events.get(ticker, [])
            if len(events_i) > 0:
                amt_t = (
                    div_vars[ticker].to(dtype=torch.float64, device=device)
                    if (div_vars is not None and ticker in div_vars)
                    else div_sched.dividend_amount_tensors[ticker].to(dtype=torch.float64, device=device)
                )
                w_t = div_sched.dividend_pv_weights[ticker].to(dtype=torch.float64, device=device)
                inc_t = div_sched.dividend_obs_incidence[ticker].to(dtype=torch.float64, device=device)
                c_i = asset_fin_curves[i]
                fin_delta_i = asset_fin_deltas[i]
                if fin_delta_i is not None:
                    active_pay_dates = [ev.payment_date for ev in events_i]
                    div_df_mult = _curve_df_delta_multiplier(
                        c_i, active_pay_dates, fin_delta_i, as_of_date=as_of_orig
                    )
                    active_ex = torch.tensor(
                        [1.0 if ev.ex_date > as_of_orig else 0.0 for ev in events_i],
                        dtype=torch.float64,
                        device=device,
                    )
                    w_t = w_t * torch.where(active_ex > 0.5, div_df_mult, torch.ones_like(div_df_mult))
                if repo_deltas is not None and ticker in repo_deltas and repo_curves is not None and ticker in repo_curves:
                    r_curve_i = _normalize_curve_obj(repo_curves[ticker], as_of_date=as_of_orig)
                    repo_ex_mult = _curve_df_delta_multiplier(
                        r_curve_i, [ev.ex_date for ev in events_i], repo_deltas[ticker], as_of_date=as_of_orig
                    )
                    active_ex = torch.tensor(
                        [1.0 if ev.ex_date > as_of_orig else 0.0 for ev in events_i],
                        dtype=torch.float64,
                        device=device,
                    )
                    w_t = w_t / torch.where(active_ex > 0.5, repo_ex_mult.clamp(min=1e-15), torch.ones_like(repo_ex_mult))
                div_pv_row = torch.matmul((amt_t * w_t).unsqueeze(0), inc_t).squeeze(0)
            else:
                div_pv_row = torch.zeros(m_rem, dtype=torch.float64, device=device)
            div_pv_rows.append(div_pv_row)
        div_pv_0 = torch.stack(div_pv_rows, dim=0)

    div_pv_b = torch.cat(
        [div_pv_0.unsqueeze(0), div_pv_0.detach().unsqueeze(0).expand(b_g, d_assets, m_rem)],
        dim=0,
    )
    escrowed_all = _escrowed_spot_softplus(spots_all.unsqueeze(-1), div_pv_b)

    r_tot_base = div_sched.r_tot_obs.to(dtype=torch.float64, device=device)
    df_fin_base = div_sched.df_fin_obs.to(dtype=torch.float64, device=device).clamp(min=1e-15)
    repo_spread_avg = div_sched.repo_spread_avg.to(dtype=torch.float64, device=device).clone()

    if df_fin_mult_leaf is not None:
        df_fin_obs_t = (df_fin_base * df_fin_mult_leaf).clamp(min=1e-15)
    else:
        df_fin_obs_rows: list[torch.Tensor] = []
        for i, ticker in enumerate(sorted_ids):
            c_i = asset_fin_curves[i]
            fin_delta_i = asset_fin_deltas[i]
            c_i_id = str(getattr(c_i, "identifier", getattr(c_i, "Identifier", disc_id)))
            if (c_i is disc_curve_obj or c_i_id == disc_id) and pillar_deltas is not None:
                df_fin_i = df_obs_0
            elif fin_delta_i is not None:
                df_fin_i = df_fin_base[i] * _curve_df_delta_multiplier(
                    c_i, future_obs_dts, fin_delta_i, as_of_date=as_of_orig
                )
            else:
                df_fin_i = df_fin_base[i]
            df_fin_obs_rows.append(df_fin_i.clamp(min=1e-15))
        df_fin_obs_t = torch.stack(df_fin_obs_rows, dim=0)
    df_repo_quanto_base = torch.exp(r_tot_base) * df_fin_base
    if df_repo_mult_leaf is not None:
        df_repo_quanto_base = df_repo_quanto_base * df_repo_mult_leaf
        tau_term_repo = max(float(tau_obs_t[-1].detach().cpu().item()), 1e-6)
        repo_spread_avg = repo_spread_avg - torch.log(df_repo_mult_leaf[:, -1].clamp(min=1e-15)) / tau_term_repo
    elif repo_deltas is not None and repo_curves is not None:
        rq_rows: list[torch.Tensor] = []
        for i, ticker in enumerate(sorted_ids):
            row_i = df_repo_quanto_base[i]
            if ticker in repo_deltas and ticker in repo_curves:
                r_curve_i = _normalize_curve_obj(repo_curves[ticker], as_of_date=as_of_orig)
                r_mult = _curve_df_delta_multiplier(
                    r_curve_i, future_obs_dts, repo_deltas[ticker], as_of_date=as_of_orig
                )
                row_i = row_i * r_mult
                tau_term_repo = max(float(tau_obs_t[-1].detach().cpu().item()), 1e-6)
                repo_spread_avg[i] = repo_spread_avg[i] - torch.log(r_mult[-1].clamp(min=1e-15)) / tau_term_repo
            rq_rows.append(row_i)
        df_repo_quanto_base = torch.stack(rq_rows, dim=0)

    quanto_meta = getattr(state, "quanto_meta", None) or {}
    k_0_q_list: list[torch.Tensor | None] = []
    if quanto_meta:
        q_corr_rows: list[torch.Tensor] = []
        tau_obs_365 = tau_obs_t.clamp(min=1e-12)
        base_sp_det = state.spots_t0.detach().to(dtype=torch.float64, device=device)
        for i, ticker in enumerate(sorted_ids):
            qm = quanto_meta.get(ticker)
            if not qm:
                q_corr_rows.append(torch.zeros((b_diff, m_rem), dtype=torch.float64, device=device))
                k_0_q_list.append(None)
                continue
            rho_base_f = float(qm.get("rho_s_fx", 0.0))
            sig_fx_base_t = torch.tensor(
                np.asarray(qm.get("sig_fx_obs", [0.0] * m_rem), dtype=np.float64),
                dtype=torch.float64,
                device=device,
            )
            sig_fx_live = (
                sig_fx_base_t + fx_vol_deltas[ticker].to(dtype=torch.float64, device=device)
                if (fx_vol_deltas is not None and ticker in fx_vol_deltas)
                else sig_fx_base_t
            )
            rho_live = (
                rho_base_f + fx_corr_deltas[ticker].to(dtype=torch.float64, device=device).reshape(())
                if (fx_corr_deltas is not None and ticker in fx_corr_deltas)
                else rho_base_f
            )
            surf_i = state.vol_surfaces[ticker]
            q_del_i = vol_deltas[ticker].to(dtype=torch.float64, device=device) if (vol_deltas is not None and ticker in vol_deltas) else None
            t_q_rep = tau_obs_t.unsqueeze(0).expand(b_diff, m_rem).reshape(-1)
            k_sp_rep = spots_all[:, i : i + 1].expand(b_diff, m_rem).reshape(-1)
            if q_del_i is not None:
                sig_eq_0 = _eval_reg_surface_iv_batch_torch(surf_i, tau_obs_t, spots_all[0, i].expand(m_rem), q_del_i)
                sig_eq_rest = _eval_reg_surface_iv_batch_torch(
                    surf_i, tau_obs_t.unsqueeze(0).expand(b_g, m_rem).reshape(-1), spots_all[1:, i : i + 1].expand(b_g, m_rem).reshape(-1), None
                ).view(b_g, m_rem)
                sig_eq_b = torch.cat([sig_eq_0.unsqueeze(0), sig_eq_rest], dim=0)
                k_0_q_list.append(None)
            else:
                sig_eq_b = _eval_reg_surface_iv_batch_torch(surf_i, t_q_rep, k_sp_rep, None).view(b_diff, m_rem)
                if iv_q0_delta_leaf is not None:
                    k_0_q_list.append(spots_all[0, i].detach().expand(m_rem))
                    sig_eq_b = torch.cat([(sig_eq_b[0] + iv_q0_delta_leaf[i]).unsqueeze(0), sig_eq_b[1:]], dim=0)
                else:
                    k_0_q_list.append(None)
            with torch.no_grad():
                sig_eq_base = _eval_reg_surface_iv_batch_torch(surf_i, tau_obs_t, base_sp_det[i].expand(m_rem), None)
            drift_base = (-rho_base_f * sig_eq_base * sig_fx_base_t * tau_obs_365).unsqueeze(0)
            drift_0 = (-rho_live * sig_eq_b[0] * sig_fx_live * tau_obs_t).unsqueeze(0)
            drift_rest = -rho_base_f * sig_eq_b[1:] * sig_fx_base_t.unsqueeze(0) * tau_obs_t.unsqueeze(0)
            q_corr_rows.append(torch.cat([drift_0, drift_rest], dim=0) - drift_base)
        quanto_delta_b = torch.stack(q_corr_rows, dim=1)  # (b_diff, D, M)
        gf_0 = (df_repo_quanto_base / df_fin_obs_t).unsqueeze(0)
        growth_all = torch.cat([gf_0, gf_0.detach().expand(b_g, d_assets, m_rem)], dim=0) * torch.exp(quanto_delta_b)
    else:
        gf_0 = df_repo_quanto_base / df_fin_obs_t
        growth_all = torch.cat([gf_0.unsqueeze(0), gf_0.detach().unsqueeze(0).expand(b_g, d_assets, m_rem)], dim=0)

    fwd_price_all = escrowed_all * growth_all
    fwd_rel_all = fwd_price_all / spots_all.unsqueeze(-1)
    fwd_perf_all = fwd_price_all / ref_spots.view(1, d_assets, 1)
    cash_div_lev_all = spots_all.unsqueeze(-1) / escrowed_all

    # Vectorized surface evaluation per asset across all b_diff rows
    t_term_t = tau_obs_t[-1]
    t_1_q = torch.cat([tau_obs_t, t_term_t.expand(5)], dim=0) if eff_surrogate is not None else tau_obs_t
    q_len = int(t_1_q.shape[0])
    iv_asset_b: list[torch.Tensor] = []
    sf_asset_b: list[torch.Tensor] = []
    sqrt_tau = torch.sqrt(tau_obs_t.clamp(min=1e-12))

    surf_caches_b: dict[str, tuple[_SurfaceGeomCache, _SurfaceQuoteCache]] = {
        ticker: _get_surface_torch_caches(state.vol_surfaces[ticker], device) for ticker in sorted_ids
    }
    k_0_surf_list: list[torch.Tensor] = []

    use_batch_surf_cpp = (
        vol_deltas is None
        and d_assets > 0
        and _load_tier1_cpp_ad_fn() is not None
        and _SURFACE_CPP_AD_FN is not None
        and all(
            surf_caches_b[t][1].slice_c_np is not None
            and surf_caches_b[t][0].n_unique >= 2
            and surf_caches_b[t][0].strikes_np is not None
            and surf_caches_b[t][0].tenors_np is not None
            for t in sorted_ids
        )
    )
    use_cpp_glue = (
        _load_tier1_cpp_ad_fn() is not None
        and _SURR_FEAT_CPP_FN is not None
        and _SURR_POST_CPP_FN is not None
    )
    if use_batch_surf_cpp:
        if eff_surrogate is not None:
            f_st_all = fwd_price_all[:, :, -1:]
            k_all_b = torch.cat(
                [fwd_price_all, f_st_all, 0.90 * f_st_all, 1.10 * f_st_all, 0.80 * f_st_all, 1.20 * f_st_all],
                dim=2,
            )  # (b_diff, D, q_len)
        else:
            k_all_b = fwd_price_all
        k_0_surf_list = [k_all_b[0, i].detach() for i in range(d_assets)]
        t_all_b = t_1_q.unsqueeze(0).expand(b_diff, q_len).reshape(-1)
        t_all_b_np = np.ascontiguousarray(t_all_b.detach().cpu().numpy(), dtype=np.float64)
        asset_metas_b = tuple(
            (state.vol_surfaces[t], surf_caches_b[t][0], surf_caches_b[t][1]) for t in sorted_ids
        )
        k_flat_d = k_all_b.transpose(0, 1).reshape(d_assets, b_diff * q_len)
        iv_flat_d, _ = _MultiSurfaceEvalCppFn.apply(  # type: ignore[no-untyped-call]
            k_flat_d, t_all_b_np, asset_metas_b, t_all_b
        )
        iv_all_mat = iv_flat_d.view(d_assets, b_diff, q_len).transpose(0, 1)
        if iv_0_delta_leaf is not None:
            iv_all_mat = torch.cat([(iv_all_mat[0] + iv_0_delta_leaf).unsqueeze(0), iv_all_mat[1:]], dim=0)
        iv_at_fwd_b = iv_all_mat[:, :, :m_rem]
        if eff_surrogate is not None and not use_cpp_glue:
            iv_short_a = iv_at_fwd_b[:, :, 0]
            iv_term_a = iv_at_fwd_b[:, :, -1]
            iv_c_a = iv_all_mat[:, :, m_rem]
            iv_m10_a = iv_all_mat[:, :, m_rem + 1]
            iv_p10_a = iv_all_mat[:, :, m_rem + 2]
            iv_m20_a = iv_all_mat[:, :, m_rem + 3]
            iv_p20_a = iv_all_mat[:, :, m_rem + 4]
            sf_asset_b = [
                torch.stack(
                    [
                        iv_short_a[:, i],
                        iv_term_a[:, i],
                        iv_term_a[:, i] - iv_short_a[:, i],
                        torch.tanh((iv_p10_a[:, i] - iv_m10_a[:, i]) / torch.clamp(0.20 * iv_c_a[:, i], min=1e-6)),
                        torch.clamp(
                            (iv_p10_a[:, i] - 2.0 * iv_c_a[:, i] + iv_m10_a[:, i]) / torch.clamp(0.01 * iv_c_a[:, i], min=1e-6),
                            min=1e-4,
                        ),
                        iv_m20_a[:, i] - iv_c_a[:, i],
                        iv_p20_a[:, i] - iv_c_a[:, i],
                    ],
                    dim=1,
                )
                for i in range(d_assets)
            ]
    else:
        iv_mat_rows: list[torch.Tensor] = []
        for i, ticker in enumerate(sorted_ids):
            surf_i = state.vol_surfaces[ticker]
            sc_pair = surf_caches_b[ticker]
            q_del_i = vol_deltas[ticker].to(dtype=torch.float64, device=device) if (vol_deltas is not None and ticker in vol_deltas) else None
            if eff_surrogate is not None:
                f_st = fwd_price_all[:, i, -1:]  # (b_diff, 1)
                k_mat = torch.cat(
                    [fwd_price_all[:, i, :], f_st, 0.90 * f_st, 1.10 * f_st, 0.80 * f_st, 1.20 * f_st],
                    dim=1,
                )  # (b_diff, m_rem + 5)
            else:
                k_mat = fwd_price_all[:, i, :]

            k_0_surf_list.append(k_mat[0].detach())
            t_all_b = t_1_q.unsqueeze(0).expand(b_diff, q_len).reshape(-1)
            iv_mat = _eval_reg_surface_iv_batch_torch(
                surf_i,
                t_all_b,
                k_mat.reshape(-1),
                q_del_i,
                cached_pair=sc_pair,
                delta_prefix_len=q_len if q_del_i is not None else None,
            ).view(b_diff, q_len)
            if iv_0_delta_leaf is not None:
                iv_mat = torch.cat([(iv_mat[0] + iv_0_delta_leaf[i]).unsqueeze(0), iv_mat[1:]], dim=0)

            iv_mat_rows.append(iv_mat)
            iv_m = iv_mat[:, :m_rem]
            iv_asset_b.append(iv_m)
            if eff_surrogate is not None and not use_cpp_glue:
                iv_short = iv_m[:, 0]
                iv_term = iv_m[:, -1]
                iv_c = iv_mat[:, m_rem]
                iv_m10 = iv_mat[:, m_rem + 1]
                iv_p10 = iv_mat[:, m_rem + 2]
                iv_m20 = iv_mat[:, m_rem + 3]
                iv_p20 = iv_mat[:, m_rem + 4]
                iv_slope = iv_term - iv_short
                ssvi_rho = torch.tanh((iv_p10 - iv_m10) / torch.clamp(0.20 * iv_c, min=1e-6))
                ssvi_eta = torch.clamp((iv_p10 - 2.0 * iv_c + iv_m10) / torch.clamp(0.01 * iv_c, min=1e-6), min=1e-4)
                pc1 = iv_m20 - iv_c
                pc2 = iv_p20 - iv_c
                sf_asset_b.append(torch.stack([iv_short, iv_term, iv_slope, ssvi_rho, ssvi_eta, pc1, pc2], dim=1))

        iv_all_mat = torch.stack(iv_mat_rows, dim=1) if iv_mat_rows else torch.empty((b_diff, 0, q_len), dtype=torch.float64, device=device)
        iv_at_fwd_b = torch.stack(iv_asset_b, dim=1)  # (b_diff, D, M)
    v_perf_b = iv_at_fwd_b * sqrt_tau.view(1, 1, m_rem)

    eye_d = torch.eye(d_assets, dtype=torch.float64, device=device)
    corr_in = corr_var.to(dtype=torch.float64, device=device)
    corr_raw_sym = 0.5 * (corr_in + corr_in.transpose(0, 1)) * (1.0 - eye_d) + eye_d
    corr_sym_0 = _smooth_corr_guard(corr_raw_sym)
    corr_b = torch.cat([corr_sym_0.unsqueeze(0), corr_sym_0.detach().unsqueeze(0).expand(b_g, d_assets, d_assets)], dim=0)

    ko_t = state.ko_barriers.to(dtype=torch.float64, device=device)
    ic_t = state.is_callable.to(dtype=torch.float64, device=device)
    cb_t = state.cpn_barriers.to(dtype=torch.float64, device=device)
    ig_t = state.is_guaranteed_cpn.to(dtype=torch.float64, device=device)
    cpn_t = state.coupon_amounts.to(dtype=torch.float64, device=device)

    call_code = monitoring_index_to_code(state.callability_index)
    cpn_code = monitoring_index_to_code(state.coupon_index or state.callability_index)
    opt_code = monitoring_index_to_code(state.option_index or state.callability_index)
    can_pure_tier1 = (
        call_code in (0, 1)
        and cpn_code in (0, 1)
        and opt_code in (0, 1)
        and (state.participation_call_strike is None or float(state.participation_call_strike) <= 0.0 or float(state.participation_call_rate) == 0.0)
        and (state.one_star_strike is None or float(state.one_star_strike) <= 0.0 or float(state.one_star_coupon) == 0.0)
    )
    if can_pure_tier1:
        is_call_flags = tuple(bool(x) for x in (ic_t.detach().cpu() > 0.5).tolist())
        has_ki_b = state.ki_barrier is not None and float(state.ki_barrier) > 0.0 and float(state.i_ki_past) < 0.5
        ki_b_val = float(state.ki_barrier) if state.ki_barrier is not None else 0.0
        f_w_b, v_w_b, p_proxy_b, q_proxy_b, l_put_proxy_b, dw_b = _run_tier1_pure_cached(
            fwd_perf_all,
            v_perf_b,
            corr_b,
            ko_t,
            cb_t,
            ig_t,
            is_call_flags=is_call_flags,
            call_code=call_code,
            cpn_code=cpn_code,
            opt_code=opt_code,
            cpn_b_type=str(state.coupon_barrier_type),
            cpn_b_dir=str(state.coupon_barrier_direction),
            force_cpn=bool(state.force_coupons_on_autocall),
            allow_exp=bool(state.allow_autocall_on_expiry),
            ko_spread=float(state.ko_spread),
            cpn_spread=float(state.cpn_spread),
            ki_spread=float(state.ki_spread),
            strike=float(state.strike),
            option_gearing=float(state.option_gearing),
            lower_strike_val=float(state.lower_strike) if state.lower_strike is not None else 0.0,
            has_active_ki=has_ki_b,
            ki_barrier_val=ki_b_val,
        )
        l_call_b = torch.zeros(b_diff, dtype=torch.float64, device=device)
        l_onestar_b = torch.zeros(b_diff, dtype=torch.float64, device=device)
    else:
        base_fields_common = {
            "df_pay": df_pay_0,
            "ko_barriers": ko_t,
            "is_callable": ic_t,
            "cpn_barriers": cb_t,
            "is_guaranteed_cpn": ig_t,
            "coupon_amounts": cpn_t,
            "strike": state.strike,
            "option_gearing": state.option_gearing,
            "lower_strike": state.lower_strike,
            "ki_barrier": state.ki_barrier,
            "i_ki_past": state.i_ki_past,
            "c_mem_0": state.c_mem_0,
            "is_memory": state.is_memory,
            "ko_spread": state.ko_spread,
            "cpn_spread": state.cpn_spread,
            "ki_spread": state.ki_spread,
            "no_notional_repayment": state.no_notional_repayment,
            "autocall_coupon": state.autocall_coupon,
            "force_coupons_on_autocall": state.force_coupons_on_autocall,
            "coupon_barrier_type": state.coupon_barrier_type,
            "coupon_barrier_direction": state.coupon_barrier_direction,
            "is_long_basket": state.is_long_basket,
            "participation_call_strike": state.participation_call_strike,
            "participation_call_rate": state.participation_call_rate,
            "one_star_strike": state.one_star_strike,
            "one_star_coupon": state.one_star_coupon,
            "pv_in_transit": 0.0,
            "already_autocalled": False,
            "callability_index": state.callability_index,
            "coupon_index": state.coupon_index,
            "option_index": state.option_index,
            "allow_autocall_on_expiry": state.allow_autocall_on_expiry,
            "inv_lex_perm": state.inv_lex_perm,
        }
        if b_g > 0 and corr_sym_0.requires_grad:
            # Rows 1..b_g belong to spot-Gamma stencils; detach corr_matrix so Cega only differentiates Row 0.
            b0 = compute_analytical_autocall_baseline(
                {**base_fields_common, "forward_perf": fwd_perf_all[:1], "v_perf": v_perf_b[:1], "corr_matrix": corr_sym_0},
                use_fast_cpu=False,
            )
            bg = compute_analytical_autocall_baseline(
                {**base_fields_common, "forward_perf": fwd_perf_all[1:], "v_perf": v_perf_b[1:], "corr_matrix": corr_sym_0.detach()},
                use_fast_cpu=False,
            )
            f_w_b = torch.cat([b0.f_w, bg.f_w], dim=0)
            v_w_b = torch.cat([b0.v_w, bg.v_w], dim=0)
            p_proxy_b = torch.cat([b0.p_proxy, bg.p_proxy], dim=0)
            q_proxy_b = torch.cat([b0.q_proxy, bg.q_proxy], dim=0)
            l_put_proxy_b = torch.cat([b0.l_put_proxy, bg.l_put_proxy], dim=0)
            dw_b = torch.cat([b0.proxy_delta_weights, bg.proxy_delta_weights], dim=0)
            l_call_b = torch.cat([b0.l_call_proxy, bg.l_call_proxy], dim=0)
            l_onestar_b = torch.cat([b0.l_onestar_proxy, bg.l_onestar_proxy], dim=0)
        else:
            base_b = compute_analytical_autocall_baseline(
                {**base_fields_common, "forward_perf": fwd_perf_all, "v_perf": v_perf_b, "corr_matrix": corr_sym_0},
                use_fast_cpu=False,
            )
            f_w_b = base_b.f_w
            v_w_b = base_b.v_w
            p_proxy_b = base_b.p_proxy
            q_proxy_b = base_b.q_proxy
            l_put_proxy_b = base_b.l_put_proxy
            dw_b = base_b.proxy_delta_weights
            l_call_b = base_b.l_call_proxy
            l_onestar_b = base_b.l_onestar_proxy

    # Optional fast Theta row appended to the same surrogate batch when 1-day roll does not cross an obs date.
    theta_row_data: dict[str, Any] | None = None
    if include_theta and isinstance(as_of_orig, dt.datetime):
        as_of_rolled = as_of_orig + dt.timedelta(days=1)
        k_cross_th, eff_ac_th, _, _, _ = _collapse_crossed_periods_for_roll(
            state,
            disc_curve_obj,
            as_of_rolled,
            state.spots_t0.detach().cpu().numpy(),
            ref_spots.detach().cpu().numpy(),
        )
        if k_cross_th == 0 and not eff_ac_th:
            with torch.no_grad():
                time_dcc = state.seasoned_state.day_count_convention
                tau_obs_th = torch.tensor(
                    [_year_fraction(as_of_rolled, d, time_dcc) for d in future_obs_dts],
                    dtype=torch.float64,
                    device=device,
                ).clamp(min=1e-6)
                disc_anchor_th = (
                    max(float(_curve_df_cached_np(disc_curve_obj, [as_of_rolled])[0]), 1e-15)
                    if (disc_curve_obj is not None and hasattr(disc_curve_obj, "df"))
                    else 1.0
                )
                if disc_curve_obj is not None and hasattr(disc_curve_obj, "df"):
                    df_pay_th = torch.tensor(
                        _curve_df_cached_np(disc_curve_obj, future_pay_dts) / disc_anchor_th,
                        dtype=torch.float64,
                        device=device,
                    )
                else:
                    df_pay_th = state.df_pay.to(dtype=torch.float64, device=device)

                div_rows_th: list[torch.Tensor] = []
                df_fin_rows_th: list[torch.Tensor] = []
                anc_memo_th: dict[int, float] = {}
                obs_df_memo_th: dict[int, npt.NDArray[np.float64]] = {}
                for i, ticker in enumerate(sorted_ids):
                    c_i = asset_fin_curves[i]
                    has_df_i = c_i is not None and hasattr(c_i, "df")
                    c_id_i = id(c_i)
                    if has_df_i and c_id_i not in anc_memo_th:
                        anc_memo_th[c_id_i] = max(float(_curve_df_cached_np(c_i, [as_of_rolled])[0]), 1e-15)
                    c_anchor_i = anc_memo_th.get(c_id_i, 1.0)
                    events_i = div_sched.dividend_events.get(ticker, [])
                    if len(events_i) > 0:
                        amt_i = div_sched.dividend_amount_tensors[ticker].to(dtype=torch.float64, device=device)
                        w_i = div_sched.dividend_pv_weights[ticker].to(dtype=torch.float64, device=device)
                        inc_i = div_sched.dividend_obs_incidence[ticker].to(dtype=torch.float64, device=device)
                        valid_i = torch.tensor(
                            [1.0 if ev.ex_date > as_of_rolled else 0.0 for ev in events_i],
                            dtype=torch.float64,
                            device=device,
                        )
                        div_rows_th.append(torch.matmul((amt_i * w_i * (valid_i / c_anchor_i)).unsqueeze(0), inc_i).squeeze(0))
                    else:
                        div_rows_th.append(torch.zeros(m_rem, dtype=torch.float64, device=device))

                    if has_df_i:
                        if c_id_i not in obs_df_memo_th:
                            obs_df_memo_th[c_id_i] = _curve_df_cached_np(c_i, future_obs_dts)
                        df_fin_rows_th.append(
                            torch.tensor(
                                obs_df_memo_th[c_id_i] / c_anchor_i,
                                dtype=torch.float64,
                                device=device,
                            ).clamp(min=1e-15)
                        )
                    else:
                        df_fin_rows_th.append(df_fin_base[i].clamp(min=1e-15))

                div_pv_th = torch.stack(div_rows_th, dim=0)
                df_fin_th = torch.stack(df_fin_rows_th, dim=0)
                tau_ratio_th = (tau_obs_th / tau_obs_t.clamp(min=1e-12)).unsqueeze(0)
                df_rq_th = torch.pow((torch.exp(r_tot_base) * df_fin_base).clamp(min=1e-15), tau_ratio_th)
                growth_th = df_rq_th / df_fin_th
                if quanto_meta:
                    q_th_rows: list[torch.Tensor] = []
                    base_sp_det = state.spots_t0.detach().to(dtype=torch.float64, device=device)
                    for i, ticker in enumerate(sorted_ids):
                        qm = quanto_meta.get(ticker)
                        if not qm:
                            q_th_rows.append(torch.zeros(m_rem, dtype=torch.float64, device=device))
                            continue
                        rho_f = float(qm.get("rho_s_fx", 0.0))
                        sig_fx_t = torch.tensor(np.asarray(qm.get("sig_fx_obs", [0.0] * m_rem), dtype=np.float64), dtype=torch.float64, device=device)
                        surf_i = state.vol_surfaces[ticker]
                        sig_new = _eval_reg_surface_iv_batch_torch(surf_i, tau_obs_th, base_sp_det[i].expand(m_rem), None)
                        sig_old = _eval_reg_surface_iv_batch_torch(surf_i, tau_obs_t, base_sp_det[i].expand(m_rem), None)
                        q_th_rows.append(-rho_f * sig_fx_t * (sig_new * tau_obs_th - sig_old * tau_obs_t.clamp(min=1e-12)))
                    growth_th = growth_th * torch.exp(torch.stack(q_th_rows, dim=0))

                sp0_det = state.spots_t0.detach().to(dtype=torch.float64, device=device)
                es_th = _escrowed_spot_softplus(sp0_det.unsqueeze(1), div_pv_th)
                fp_price_th = es_th * growth_th
                fp_rel_th = fp_price_th / sp0_det.unsqueeze(1)
                fp_perf_th = fp_price_th / ref_spots.unsqueeze(1)
                cd_lev_th = sp0_det.unsqueeze(1) / es_th
                t_1_th = torch.cat([tau_obs_th, tau_obs_th[-1].expand(5)], dim=0) if eff_surrogate is not None else tau_obs_th
                iv_th_rows: list[torch.Tensor] = []
                sf_th_rows: list[torch.Tensor] = []
                iv_th_all: torch.Tensor
                if use_batch_surf_cpp:
                    s_fn_th = _SURFACE_CPP_AD_FN
                    assert s_fn_th is not None
                    q_len_th = int(t_1_th.shape[0])
                    if eff_surrogate is not None:
                        fst_a = fp_price_th[:, -1:]
                        k_th_all = torch.cat(
                            [fp_price_th, fst_a, 0.90 * fst_a, 1.10 * fst_a, 0.80 * fst_a, 1.20 * fst_a],
                            dim=1,
                        )
                    else:
                        k_th_all = fp_price_th
                    t_th_np = np.ascontiguousarray(t_1_th.cpu().numpy(), dtype=np.float64)
                    k_th_np = np.ascontiguousarray(k_th_all.cpu().numpy(), dtype=np.float64)
                    iv_th_np = np.empty((d_assets, q_len_th), dtype=np.float64)
                    dk_th_np = np.empty((d_assets, q_len_th), dtype=np.float64)
                    for i, ticker in enumerate(sorted_ids):
                        surf_i = state.vol_surfaces[ticker]
                        g_i, q_i = surf_caches_b[ticker]
                        assert g_i.strikes_np is not None and g_i.tenors_np is not None
                        assert q_i.quotes_np is not None and q_i.slice_c_np is not None
                        assert q_i.slope_left_np is not None and q_i.slope_right_np is not None
                        s_fn_th(
                            q_len_th, g_i.n_unique, g_i.n_exp,
                            g_i.k_min, g_i.k_max, float(surf_i.min_vol),
                            g_i.strikes_np.ctypes.data, g_i.tenors_np.ctypes.data, q_i.quotes_np.ctypes.data,
                            q_i.slice_c_np.ctypes.data, q_i.slope_left_np.ctypes.data, q_i.slope_right_np.ctypes.data,
                            t_th_np.ctypes.data, k_th_np[i].ctypes.data,
                            iv_th_np[i].ctypes.data, dk_th_np[i].ctypes.data,
                        )
                    iv_th_all = torch.from_numpy(iv_th_np).to(device=device)
                    vp_th = iv_th_all[:, :m_rem] * torch.sqrt(tau_obs_th.clamp(min=1e-12)).unsqueeze(0)
                    if eff_surrogate is not None and not use_cpp_glue:
                        iv_s = iv_th_all[:, 0]
                        iv_t = iv_th_all[:, m_rem - 1]
                        ic_v = iv_th_all[:, m_rem]
                        im10 = iv_th_all[:, m_rem + 1]
                        ip10 = iv_th_all[:, m_rem + 2]
                        im20 = iv_th_all[:, m_rem + 3]
                        ip20 = iv_th_all[:, m_rem + 4]
                        sf_th_rows = [
                            torch.stack(
                                [
                                    iv_s,
                                    iv_t,
                                    iv_t - iv_s,
                                    torch.tanh((ip10 - im10) / torch.clamp(0.20 * ic_v, min=1e-6)),
                                    torch.clamp((ip10 - 2.0 * ic_v + im10) / torch.clamp(0.01 * ic_v, min=1e-6), min=1e-4),
                                    im20 - ic_v,
                                    ip20 - ic_v,
                                ],
                                dim=1,
                            )
                        ]
                else:
                    iv_th_full: list[torch.Tensor] = []
                    for i, ticker in enumerate(sorted_ids):
                        surf_i = state.vol_surfaces[ticker]
                        if eff_surrogate is not None:
                            fst = fp_price_th[i, -1]
                            kq = torch.cat([fp_price_th[i], torch.stack([fst, 0.90 * fst, 1.10 * fst, 0.80 * fst, 1.20 * fst])], dim=0)
                            iv_a = _eval_reg_surface_iv_batch_torch(surf_i, t_1_th, kq, None, cached_pair=surf_caches_b[ticker])
                            iv_th_full.append(iv_a)
                            iv_m_i = iv_a[:m_rem]
                            iv_th_rows.append(iv_m_i)
                            if not use_cpp_glue:
                                ic_v, im10, ip10, im20, ip20 = iv_a[m_rem], iv_a[m_rem + 1], iv_a[m_rem + 2], iv_a[m_rem + 3], iv_a[m_rem + 4]
                                sf_th_rows.append(
                                    torch.stack(
                                        [
                                            iv_m_i[0],
                                            iv_m_i[-1],
                                            iv_m_i[-1] - iv_m_i[0],
                                            torch.tanh((ip10 - im10) / torch.clamp(0.20 * ic_v, min=1e-6)),
                                            torch.clamp((ip10 - 2.0 * ic_v + im10) / torch.clamp(0.01 * ic_v, min=1e-6), min=1e-4),
                                            im20 - ic_v,
                                            ip20 - ic_v,
                                        ],
                                        dim=0,
                                    )
                                )
                        else:
                            iv_a = _eval_reg_surface_iv_batch_torch(surf_i, tau_obs_th, fp_price_th[i], None, cached_pair=surf_caches_b[ticker])
                            iv_th_full.append(iv_a)
                            iv_th_rows.append(iv_a)
                    iv_th_all = torch.stack(iv_th_full, dim=0)
                    vp_th = torch.stack(iv_th_rows, dim=0) * torch.sqrt(tau_obs_th.clamp(min=1e-12)).unsqueeze(0)
                if can_pure_tier1 and d_assets <= 32 and _load_tier1_cpp_ad_fn() is not None:
                    fp_1 = fp_perf_th.unsqueeze(0)
                    vp_1 = vp_th.unsqueeze(0)
                    c_1 = corr_sym_0.detach().unsqueeze(0)
                    f_s1 = fp_1.detach().transpose(1, 2).reshape(m_rem, d_assets).clamp(min=1e-16)
                    ord_a1 = (
                        torch.argsort(f_s1, dim=1, stable=True)
                        if d_assets > 1
                        else torch.zeros((m_rem, 1), dtype=torch.long, device=device)
                    )
                    ord_d1 = ord_a1.flip(dims=[1]) if d_assets > 1 else ord_a1
                    ord_c1 = ord_d1 if call_code == 1 else ord_a1
                    ord_cp1 = ord_d1 if cpn_code == 1 else ord_a1
                    ord_oa1 = ord_a1[-1:, :]
                    ord_o1 = ord_oa1.flip(dims=[1]) if (opt_code == 1 and d_assets > 1) else ord_oa1
                    ic_1 = np.array([1 if bool(x) else 0 for x in (ic_t.detach().cpu() > 0.5).tolist()], dtype=np.int32)
                    has_ki_th = state.ki_barrier is not None and float(state.ki_barrier) > 0.0 and float(state.i_ki_past) < 0.5
                    meta_th: tuple[Any, ...] = (
                        int(call_code),
                        int(cpn_code),
                        int(opt_code),
                        1 if str(state.coupon_barrier_direction).strip().lower() == "down" else 0,
                        1 if str(state.coupon_barrier_type).strip().lower() == "out" else 0,
                        1 if bool(state.force_coupons_on_autocall) else 0,
                        1 if bool(state.allow_autocall_on_expiry) else 0,
                        float(state.ko_spread),
                        float(state.cpn_spread),
                        float(state.ki_spread),
                        float(state.strike),
                        float(state.option_gearing),
                        float(state.lower_strike) if state.lower_strike is not None else 0.0,
                        1 if has_ki_th else 0,
                        float(state.ki_barrier) if state.ki_barrier is not None else 0.0,
                    )
                    fw_1, vw_1, p_1, q_1, lp_1, dw_1 = _run_tier1_cpp_forward_numpy(
                        np.ascontiguousarray(fp_1.cpu().numpy(), dtype=np.float64),
                        np.ascontiguousarray(vp_1.cpu().numpy(), dtype=np.float64),
                        np.ascontiguousarray(c_1.cpu().numpy(), dtype=np.float64),
                        np.ascontiguousarray(ko_t.detach().cpu().numpy(), dtype=np.float64),
                        ic_1,
                        np.ascontiguousarray(cb_t.detach().cpu().numpy(), dtype=np.float64),
                        np.ascontiguousarray(ig_t.detach().cpu().numpy(), dtype=np.float64),
                        np.ascontiguousarray(ord_c1.cpu().numpy(), dtype=np.int64),
                        np.ascontiguousarray(ord_cp1.cpu().numpy(), dtype=np.int64),
                        np.ascontiguousarray(ord_o1.cpu().numpy(), dtype=np.int64),
                        meta_th,
                    )
                    z0 = torch.zeros((), dtype=torch.float64, device=device)
                    p_1_t = torch.from_numpy(p_1[0]).to(device=device)
                    base_th = AutocallBaselineOutput(
                        f_w=torch.from_numpy(fw_1[0]).to(device=device),
                        v_w=torch.from_numpy(vw_1[0]).to(device=device),
                        h_proxy=p_1_t,
                        p_proxy=p_1_t,
                        survival_proxy=torch.cumprod(1.0 - p_1_t, dim=0),
                        q_proxy=torch.from_numpy(q_1[0]).to(device=device),
                        l_put_proxy=torch.tensor(float(lp_1[0]), dtype=torch.float64, device=device),
                        proxy_delta_weights=torch.from_numpy(dw_1[0]).to(device=device),
                        option_pv_proxy=z0,
                        notional_pv_proxy=z0,
                        coupon_pv_proxy=z0,
                        product_value_proxy=z0,
                        l_call_proxy=z0,
                        l_onestar_proxy=z0,
                        has_unsupported_exotic_index=False,
                        unsupported_indices=(),
                    )
                else:
                    base_th_raw = compute_analytical_autocall_baseline(
                        {
                            "forward_perf": fp_perf_th.detach().cpu(),
                            "v_perf": vp_th.detach().cpu(),
                            "corr_matrix": corr_sym_0.detach().cpu(),
                            "df_pay": df_pay_th.detach().cpu(),
                            "tau_obs": tau_obs_th.detach().cpu(),
                            "ko_barriers": ko_t.detach().cpu(),
                            "is_callable": ic_t.detach().cpu(),
                            "cpn_barriers": cb_t.detach().cpu(),
                            "is_guaranteed_cpn": ig_t.detach().cpu(),
                            "coupon_amounts": cpn_t.detach().cpu(),
                            "strike": state.strike,
                            "option_gearing": state.option_gearing,
                            "lower_strike": state.lower_strike,
                            "ki_barrier": state.ki_barrier,
                            "i_ki_past": state.i_ki_past,
                            "c_mem_0": state.c_mem_0,
                            "is_memory": state.is_memory,
                            "ko_spread": state.ko_spread,
                            "cpn_spread": state.cpn_spread,
                            "ki_spread": state.ki_spread,
                            "no_notional_repayment": state.no_notional_repayment,
                            "autocall_coupon": state.autocall_coupon,
                            "force_coupons_on_autocall": state.force_coupons_on_autocall,
                            "coupon_barrier_type": state.coupon_barrier_type,
                            "coupon_barrier_direction": state.coupon_barrier_direction,
                            "is_long_basket": state.is_long_basket,
                            "participation_call_strike": state.participation_call_strike,
                            "participation_call_rate": state.participation_call_rate,
                            "one_star_strike": state.one_star_strike,
                            "one_star_coupon": state.one_star_coupon,
                            "pv_in_transit": 0.0,
                            "already_autocalled": False,
                            "callability_index": state.callability_index,
                            "coupon_index": state.coupon_index,
                            "option_index": state.option_index,
                            "allow_autocall_on_expiry": state.allow_autocall_on_expiry,
                            "inv_lex_perm": state.inv_lex_perm,
                        },
                        use_fast_cpu=True,
                    )
                    base_th = (
                        AutocallBaselineOutput(
                            f_w=base_th_raw.f_w.to(device=device),
                            v_w=base_th_raw.v_w.to(device=device),
                            h_proxy=base_th_raw.h_proxy.to(device=device),
                            p_proxy=base_th_raw.p_proxy.to(device=device),
                            survival_proxy=base_th_raw.survival_proxy.to(device=device),
                            q_proxy=base_th_raw.q_proxy.to(device=device),
                            l_put_proxy=base_th_raw.l_put_proxy.to(device=device),
                            proxy_delta_weights=base_th_raw.proxy_delta_weights.to(device=device),
                            option_pv_proxy=base_th_raw.option_pv_proxy.to(device=device),
                            notional_pv_proxy=base_th_raw.notional_pv_proxy.to(device=device),
                            coupon_pv_proxy=base_th_raw.coupon_pv_proxy.to(device=device),
                            product_value_proxy=base_th_raw.product_value_proxy.to(device=device),
                            l_call_proxy=base_th_raw.l_call_proxy.to(device=device),
                            l_onestar_proxy=base_th_raw.l_onestar_proxy.to(device=device),
                            has_unsupported_exotic_index=base_th_raw.has_unsupported_exotic_index,
                            unsupported_indices=base_th_raw.unsupported_indices,
                        )
                        if device.type != "cpu"
                        else base_th_raw
                    )
                delta_tau_th = (
                    torch.cat([tau_obs_th[:1], (tau_obs_th[1:] - tau_obs_th[:-1]).clamp(min=1e-8)], dim=0)
                    if m_rem > 1
                    else tau_obs_th[:1]
                )
                df_fin_th_ov: Callable[[Any], torch.Tensor | float] | None = None
                if disc_curve_obj is not None and hasattr(disc_curve_obj, "df"):
                    memo_th: dict[Any, float] = {}

                    def _df_th_ov(f_dts: Any) -> torch.Tensor | float:
                        if not isinstance(f_dts, (list, tuple, np.ndarray)) and f_dts in memo_th:
                            return memo_th[f_dts]
                        val = np.asarray(disc_curve_obj.df(f_dts), dtype=np.float64) / disc_anchor_th
                        if not isinstance(f_dts, (list, tuple, np.ndarray)):
                            fv = float(val.reshape(-1)[0])
                            memo_th[f_dts] = fv
                            return fv
                        return torch.tensor(val, dtype=torch.float64, device=device)

                    df_fin_th_ov = _df_th_ov
                w_fund_th = (
                    _compute_funding_weights_per_period(
                        state,
                        as_of_override=as_of_rolled,
                        df_disc_fn=df_fin_th_ov,
                        df_fwd_fn=None,
                        period_start_idx=0,
                        device=device,
                    )
                    if state.funding_leg is not None
                    else None
                )
                in_transit_th = _rolled_in_transit_pv(state, disc_curve_obj, as_of_rolled, None, device)
                sf_th_val: torch.Tensor | None = None
                if sf_th_rows:
                    sf_th_val = sf_th_rows[0] if (use_batch_surf_cpp and len(sf_th_rows) == 1) else torch.stack(sf_th_rows, dim=0)
                theta_row_data = {
                    "tau_obs": tau_obs_th,
                    "delta_tau": delta_tau_th,
                    "df_pay": df_pay_th,
                    "fp_rel": fp_rel_th,
                    "fp_perf": fp_perf_th,
                    "cd_lev": cd_lev_th,
                    "v_perf": vp_th,
                    "iv_all": iv_th_all,
                    "sf_feats": sf_th_val,
                    "base": base_th,
                    "w_fund": w_fund_th,
                    "in_transit": in_transit_th,
                }

    # Funding weights & In-Transit PV on Row 0 (and detached across Gamma rows)
    w_fund_0: torch.Tensor | None = None
    if state.funding_leg is not None:
        df_fin_ov: Callable[[Any], torch.Tensor | float] | None = None
        df_fwd_ov: Callable[[Any], torch.Tensor | float] | None = None
        if pillar_deltas is not None:
            df_fin_memo: dict[Any, torch.Tensor] = {}

            def _df_fin_ov_b(fin_dts: Any) -> torch.Tensor | float:
                is_sc = not isinstance(fin_dts, (list, tuple, np.ndarray))
                if is_sc and fin_dts in df_fin_memo:
                    return df_fin_memo[fin_dts]
                dl = [fin_dts] if is_sc else list(fin_dts)
                b_dfs = torch.tensor(np.asarray(disc_curve_obj.df(dl), dtype=np.float64).reshape(-1), dtype=torch.float64, device=device)
                res_f = b_dfs * _curve_df_delta_multiplier(disc_curve_obj, dl, pillar_deltas, as_of_date=as_of_orig)
                if is_sc:
                    df_fin_memo[fin_dts] = res_f[0]
                    return res_f[0]
                for d_it, v_it in zip(dl, res_f):
                    df_fin_memo[d_it] = v_it
                return res_f

            df_fin_ov = _df_fin_ov_b
        if funding_pillar_deltas is not None:
            f_curve_leg = _normalize_curve_obj(state.funding_leg.get("funding_curve"), as_of_date=as_of_orig)
            if f_curve_leg is not None:
                def _df_fwd_ov_b(fwd_dts: Any) -> torch.Tensor:
                    is_sc = not isinstance(fwd_dts, (list, tuple, np.ndarray))
                    dl = [fwd_dts] if is_sc else list(fwd_dts)
                    b_dfs = torch.tensor(np.asarray(f_curve_leg.df(dl), dtype=np.float64).reshape(-1), dtype=torch.float64, device=device)
                    res_w = b_dfs * _curve_df_delta_multiplier(f_curve_leg, dl, funding_pillar_deltas, as_of_date=as_of_orig)
                    return res_w[0] if is_sc else res_w

                df_fwd_ov = _df_fwd_ov_b
        w_fund_0 = _compute_funding_weights_per_period(
            state, df_disc_fn=df_fin_ov, df_fwd_fn=df_fwd_ov, period_start_idx=0, device=device
        )
        if w_fund_delta_leaf is not None:
            w_fund_0 = w_fund_0 + w_fund_delta_leaf

    w_fund_b = (
        torch.cat([w_fund_0.unsqueeze(0), w_fund_0.detach().unsqueeze(0).expand(b_g, m_rem)], dim=0)
        if w_fund_0 is not None
        else None
    )
    if pillar_deltas is not None and state.seasoned_state.in_transit_indices:
        in_transit_0 = _rolled_in_transit_pv(state, disc_curve_obj, as_of_orig, pillar_deltas, device)
    else:
        in_transit_0 = torch.tensor(float(state.pv_in_transit), dtype=torch.float64, device=device)
        if in_transit_delta_leaf is not None:
            in_transit_0 = in_transit_0 + in_transit_delta_leaf
    if out_aux is not None:
        out_aux["k_0_surf"] = k_0_surf_list
        out_aux["k_0_q"] = k_0_q_list
        out_aux["t_1_q"] = t_1_q
        out_aux["tau_obs_t"] = tau_obs_t
        out_aux["surf_caches_b"] = surf_caches_b
        out_aux["asset_fin_curves"] = asset_fin_curves
        out_aux["disc_curve_obj"] = disc_curve_obj
        out_aux["as_of_orig"] = as_of_orig
    pv_in_transit_b = torch.cat(
        [in_transit_0.unsqueeze(0), in_transit_0.detach().expand(b_g)], dim=0
    )
    df_pay_diff_b = torch.cat(
        [df_pay_0.unsqueeze(0), df_pay_0.detach().unsqueeze(0).expand(b_g, m_rem)], dim=0
    )

    ic_np_s = np.ascontiguousarray(ic_t.detach().cpu().numpy(), dtype=np.float64)
    ig_np_s = np.ascontiguousarray(ig_t.detach().cpu().numpy(), dtype=np.float64)
    cpn_np_s = np.ascontiguousarray(cpn_t.detach().cpu().numpy(), dtype=np.float64)
    k_eff_put_v = max(
        max(float(state.strike), 0.0) - (max(float(state.lower_strike), 0.0) if state.lower_strike is not None else 0.0),
        0.0,
    )

    d_haz_t2: torch.Tensor | None = None
    d_cpn_t2: torch.Tensor | None = None
    d_put_t2: torch.Tensor | None = None

    if eff_surrogate is not None:
        first_param = next(eff_surrogate.parameters(), None)
        surr_dtype = first_param.dtype if first_param is not None else torch.float32
        surr_dev = first_param.device if first_param is not None else device
        surr_d_node = int(getattr(eff_surrogate, "d_node", len(AutocallNodeCol)))
        surr_d_global = int(getattr(eff_surrogate, "d_global", len(AutocallGlobalCol)))

        sb = state.seasoned_state.sentinel_barriers
        if sb is None:
            raise RuntimeError("Seasoned state sentinel_barriers must be initialized before evaluating PV.")

        if use_cpp_glue:
            role_a_np = np.zeros((d_assets,), dtype=np.float64)
            role_b_np = np.zeros((d_assets,), dtype=np.float64)
            active_idx_names = (
                state.callability_index,
                state.coupon_index or state.callability_index,
                state.option_index or state.callability_index,
            )
            if d_assets >= 2 and any(nm in PAIR_ORDER_DEPENDENT_INDICES for nm in active_idx_names):
                role_a_np[int(state.inv_lex_perm[0])] = 1.0
                role_b_np[int(state.inv_lex_perm[1])] = 1.0
            has_ki_f = 1.0 if (state.ki_barrier is not None and state.ki_barrier > 0.0) else 0.0
            b_ki_v = float(state.ki_barrier) if (state.ki_barrier is not None and state.ki_barrier > 0.0) else float(state.strike)
            lk_v = float(state.lower_strike) if state.lower_strike is not None else 0.0
            g_static_np = np.zeros((len(AutocallGlobalCol),), dtype=np.float64)
            g_static_np[AutocallGlobalCol.K_PUT] = float(state.strike)
            g_static_np[AutocallGlobalCol.B_KI] = b_ki_v
            g_static_np[AutocallGlobalCol.HAS_KI_FLAG] = has_ki_f
            g_static_np[AutocallGlobalCol.I_KI_PAST] = float(state.i_ki_past)
            g_static_np[AutocallGlobalCol.G_PUT] = float(state.option_gearing)
            g_static_np[AutocallGlobalCol.C_MEM_0] = float(state.c_mem_0)
            g_static_np[AutocallGlobalCol.IS_MEMORY_FLAG] = 1.0 if state.is_memory else 0.0
            g_static_np[AutocallGlobalCol.ABS_KO_SPREAD] = abs(float(state.ko_spread))
            g_static_np[AutocallGlobalCol.ABS_CPN_SPREAD] = abs(float(state.cpn_spread))
            g_static_np[AutocallGlobalCol.LOWER_STRIKE] = lk_v
            g_static_np[AutocallGlobalCol.CALL_INDEX_ID] = float(call_code)
            g_static_np[AutocallGlobalCol.CPN_INDEX_ID] = float(cpn_code)
            g_static_np[AutocallGlobalCol.OPT_INDEX_ID] = float(opt_code)
            g_static_np[AutocallGlobalCol.IS_LONG_BASKET] = 1.0 if state.is_long_basket else 0.0
            g_static_np[AutocallGlobalCol.ABS_KI_SPREAD] = abs(float(state.ki_spread))
            tau_np_s = np.ascontiguousarray(tau_obs_t.detach().cpu().numpy(), dtype=np.float64)
            dtau_np_s = np.ascontiguousarray(delta_tau_t.detach().cpu().numpy(), dtype=np.float64)
            lko_np_s = np.ascontiguousarray(sb.log_ko_barrier_feat, dtype=np.float64)
            lcp_np_s = np.ascontiguousarray(sb.log_cpn_barrier_feat, dtype=np.float64)
            feat_static_np = (
                tau_np_s, dtau_np_s, lko_np_s, ic_np_s, lcp_np_s, ig_np_s, cpn_np_s,
                role_a_np, role_b_np, g_static_np, surr_d_node, surr_d_global, max(float(state.strike), 1e-4),
            )
            feat_box: list[Any] = []
            nf_all, ef_all, sf_all, gf_all = _SurrFeaturesCppFn.apply(  # type: ignore[no-untyped-call]
                moneyness_all, fwd_rel_all, fwd_perf_all, cash_div_lev_all, v_perf_b, iv_all_mat,
                dw_b, corr_b, df_pay_diff_b, f_w_b, v_w_b, p_proxy_b, q_proxy_b, l_put_proxy_b,
                repo_spread_avg, feat_static_np, feat_box,
            )
            am_all = torch.ones((b_diff, d_assets), dtype=torch.float64, device=device)
            sm_all = torch.ones((b_diff, m_rem), dtype=torch.float64, device=device)
            if theta_row_data is not None:
                with torch.no_grad():
                    b_th_info = theta_row_data["base"]
                    sp0_d = state.spots_t0.detach().to(dtype=torch.float64, device=device)
                    m0_th = (sp0_d / ref_spots).unsqueeze(0)
                    th_static_np = (
                        np.ascontiguousarray(theta_row_data["tau_obs"].cpu().numpy(), dtype=np.float64),
                        np.ascontiguousarray(theta_row_data["delta_tau"].cpu().numpy(), dtype=np.float64),
                        lko_np_s, ic_np_s, lcp_np_s, ig_np_s, cpn_np_s,
                        role_a_np, role_b_np, g_static_np, surr_d_node, surr_d_global, max(float(state.strike), 1e-4),
                    )
                    th_in_nps = tuple(
                        np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float64)
                        for t in (
                            m0_th,
                            theta_row_data["fp_rel"].unsqueeze(0),
                            theta_row_data["fp_perf"].unsqueeze(0),
                            theta_row_data["cd_lev"].unsqueeze(0),
                            theta_row_data["v_perf"].unsqueeze(0),
                            theta_row_data["iv_all"].unsqueeze(0),
                            b_th_info.proxy_delta_weights.unsqueeze(0),
                            corr_sym_0.detach().unsqueeze(0),
                            theta_row_data["df_pay"].unsqueeze(0),
                            b_th_info.f_w.unsqueeze(0),
                            b_th_info.v_w.unsqueeze(0),
                            b_th_info.p_proxy.unsqueeze(0),
                            b_th_info.q_proxy.unsqueeze(0),
                            b_th_info.l_put_proxy.unsqueeze(0),
                            div_sched.repo_spread_avg,
                        )
                    )
                    nf_th_np = np.empty((1, d_assets, surr_d_node), dtype=np.float64)
                    ef_th_np = np.empty((1, d_assets, d_assets, 4), dtype=np.float64)
                    sf_th_np = np.empty((1, m_rem, 12), dtype=np.float64)
                    gf_th_np = np.empty((1, surr_d_global), dtype=np.float64)
                    sf_fn_th = _SURR_FEAT_CPP_FN
                    assert sf_fn_th is not None
                    sf_fn_th(
                        1, d_assets, m_rem, int(theta_row_data["iv_all"].shape[1]), surr_d_node, surr_d_global, 0,
                        max(float(state.strike), 1e-4),
                        *(a.ctypes.data for a in th_in_nps),
                        *(a.ctypes.data for a in th_static_np[:10]),
                        nf_th_np.ctypes.data, ef_th_np.ctypes.data, sf_th_np.ctypes.data, gf_th_np.ctypes.data,
                        0, 0, 0, 0,
                        *([0] * 15),
                    )
                nf_all = torch.cat([nf_all, torch.from_numpy(nf_th_np).to(device=device)], dim=0)
                ef_all = torch.cat([ef_all, torch.from_numpy(ef_th_np).to(device=device)], dim=0)
                am_all = torch.ones((b_diff + 1, d_assets), dtype=torch.float64, device=device)
                sf_all = torch.cat([sf_all, torch.from_numpy(sf_th_np).to(device=device)], dim=0)
                sm_all = torch.ones((b_diff + 1, m_rem), dtype=torch.float64, device=device)
                gf_all = torch.cat([gf_all, torch.from_numpy(gf_th_np).to(device=device)], dim=0)
        else:
            surf_feats_b = torch.stack(sf_asset_b, dim=1)  # (b_diff, D, 7)
            log_ko_feat_t = torch.tensor(np.array(sb.log_ko_barrier_feat, copy=True), dtype=torch.float64, device=device)
            log_cpn_feat_t = torch.tensor(np.array(sb.log_cpn_barrier_feat, copy=True), dtype=torch.float64, device=device)

            role_leg_a = torch.zeros(d_assets, dtype=torch.float64, device=device)
            role_leg_b = torch.zeros(d_assets, dtype=torch.float64, device=device)
            active_idx_names = (
                state.callability_index,
                state.coupon_index or state.callability_index,
                state.option_index or state.callability_index,
            )
            if d_assets >= 2 and any(nm in PAIR_ORDER_DEPENDENT_INDICES for nm in active_idx_names):
                role_leg_a[int(state.inv_lex_perm[0])] = 1.0
                role_leg_b[int(state.inv_lex_perm[1])] = 1.0
            active_ones = torch.ones(d_assets, dtype=torch.float64, device=device)

            def _build_surr_tensors(
                b_n: int,
                m_b: torch.Tensor,
                fr_b: torch.Tensor,
                fp_b: torch.Tensor,
                cd_b: torch.Tensor,
                vp_b: torch.Tensor,
                sf_b: torch.Tensor,
                dw_in: torch.Tensor,
                cm_in: torch.Tensor,
                tau_vec: torch.Tensor,
                dtau_vec: torch.Tensor,
                dfp_b: torch.Tensor,
                fw_in: torch.Tensor,
                vw_in: torch.Tensor,
                p_in: torch.Tensor,
                q_in: torch.Tensor,
                lp_in: torch.Tensor,
                r_spr: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                log_m0_n = torch.log(m_b.clamp(min=1e-12))
                tau_term_n = tau_vec[-1].clamp(min=1e-6)
                log_fc_n = torch.log(fr_b[:, :, -1].clamp(min=1e-12)) / tau_term_n
                log_dl_n = torch.log(cd_b[:, :, -1].clamp(min=1.0))
                fp_t_n = fp_b[:, :, -1].clamp(min=1e-12)
                v_t_n = vp_b[:, :, -1].clamp(min=1e-6)
                k_eff_n = max(float(state.strike), 1e-4)
                dtp_n = torch.log(fp_t_n / k_eff_n) / v_t_n
                a_1 = active_ones.unsqueeze(0).expand(b_n, d_assets)
                ra = role_leg_a.unsqueeze(0).expand(b_n, d_assets)
                rb = role_leg_b.unsqueeze(0).expand(b_n, d_assets)
                rs = r_spr.unsqueeze(0).expand(b_n, d_assets)
                z_nd = torch.zeros((b_n, d_assets), dtype=torch.float64, device=device)
                n_cols: list[torch.Tensor] = [z_nd] * len(AutocallNodeCol)
                n_cols[AutocallNodeCol.LOG_MONEYNESS] = log_m0_n
                n_cols[AutocallNodeCol.LOG_FWD_CARRY_1Y] = log_fc_n
                n_cols[AutocallNodeCol.CASH_DIV_LEVERAGE] = log_dl_n
                n_cols[AutocallNodeCol.REPO_SPREAD_AVG] = rs
                n_cols[AutocallNodeCol.IV_SHORT_ATM] = sf_b[:, :, 0]
                n_cols[AutocallNodeCol.IV_TERM_ATM] = sf_b[:, :, 1]
                n_cols[AutocallNodeCol.IV_TERM_SLOPE] = sf_b[:, :, 2]
                n_cols[AutocallNodeCol.SSVI_SKEW_RHO] = sf_b[:, :, 3]
                n_cols[AutocallNodeCol.SSVI_CURV_ETA] = sf_b[:, :, 4]
                n_cols[AutocallNodeCol.SMILE_RESIDUAL_PC1] = sf_b[:, :, 5]
                n_cols[AutocallNodeCol.SMILE_RESIDUAL_PC2] = sf_b[:, :, 6]
                n_cols[AutocallNodeCol.DIST_TO_PUT_STRIKE] = dtp_n
                n_cols[AutocallNodeCol.PROXY_DELTA_WEIGHT] = dw_in
                n_cols[AutocallNodeCol.ASSET_ACTIVE_MASK] = a_1
                n_cols[AutocallNodeCol.ROLE_LEG_A] = ra
                n_cols[AutocallNodeCol.ROLE_LEG_B] = rb
                nf_out = torch.stack(n_cols, dim=2)[:, :, :surr_d_node]

                log_fp_t = torch.log(fp_t_n)
                cov_act = cm_in * v_t_n.unsqueeze(2) * v_t_n.unsqueeze(1)
                lpd_act = torch.abs(log_fp_t.unsqueeze(2) - log_fp_t.unsqueeze(1))
                off_d = (1.0 - eye_d).unsqueeze(0).expand(b_n, d_assets, d_assets)
                ef_out = torch.stack([cm_in, cov_act, lpd_act, off_d], dim=-1)

                sf_out = torch.stack(
                    [
                        tau_vec.unsqueeze(0).expand(b_n, m_rem),
                        dtau_vec.unsqueeze(0).expand(b_n, m_rem),
                        dfp_b,
                        log_ko_feat_t.unsqueeze(0).expand(b_n, m_rem),
                        ic_t.unsqueeze(0).expand(b_n, m_rem),
                        log_cpn_feat_t.unsqueeze(0).expand(b_n, m_rem),
                        ig_t.unsqueeze(0).expand(b_n, m_rem),
                        cpn_t.unsqueeze(0).expand(b_n, m_rem),
                        torch.log(fw_in.clamp(min=1e-12)),
                        vw_in,
                        p_in,
                        q_in,
                    ],
                    dim=2,
                )
                has_ki_f = 1.0 if (state.ki_barrier is not None and state.ki_barrier > 0.0) else 0.0
                b_ki_v = float(state.ki_barrier) if (state.ki_barrier is not None and state.ki_barrier > 0.0) else float(state.strike)
                lk_v = float(state.lower_strike) if state.lower_strike is not None else 0.0
                g_static_vals = [0.0] * len(AutocallGlobalCol)
                g_static_vals[AutocallGlobalCol.K_PUT] = float(state.strike)
                g_static_vals[AutocallGlobalCol.B_KI] = b_ki_v
                g_static_vals[AutocallGlobalCol.HAS_KI_FLAG] = has_ki_f
                g_static_vals[AutocallGlobalCol.I_KI_PAST] = float(state.i_ki_past)
                g_static_vals[AutocallGlobalCol.G_PUT] = float(state.option_gearing)
                g_static_vals[AutocallGlobalCol.C_MEM_0] = float(state.c_mem_0)
                g_static_vals[AutocallGlobalCol.IS_MEMORY_FLAG] = 1.0 if state.is_memory else 0.0
                g_static_vals[AutocallGlobalCol.ABS_KO_SPREAD] = abs(float(state.ko_spread))
                g_static_vals[AutocallGlobalCol.ABS_CPN_SPREAD] = abs(float(state.cpn_spread))
                g_static_vals[AutocallGlobalCol.LOWER_STRIKE] = lk_v
                g_static_vals[AutocallGlobalCol.CALL_INDEX_ID] = float(call_code)
                g_static_vals[AutocallGlobalCol.CPN_INDEX_ID] = float(cpn_code)
                g_static_vals[AutocallGlobalCol.OPT_INDEX_ID] = float(opt_code)
                g_static_vals[AutocallGlobalCol.IS_LONG_BASKET] = 1.0 if state.is_long_basket else 0.0
                g_static_vals[AutocallGlobalCol.ABS_KI_SPREAD] = abs(float(state.ki_spread))
                g_base = torch.tensor(g_static_vals, dtype=torch.float64, device=device).unsqueeze(0).expand(b_n, -1).clone()
                g_base[:, AutocallGlobalCol.L_PUT_PROXY] = lp_in
                gf_out = g_base[:, :surr_d_global]
                return nf_out, ef_out, a_1, sf_out, torch.ones((b_n, m_rem), dtype=torch.float64, device=device), gf_out

            nf_all, ef_all, am_all, sf_all, sm_all, gf_all = _build_surr_tensors(
                b_diff,
                moneyness_all,
                fwd_rel_all,
                fwd_perf_all,
                cash_div_lev_all,
                v_perf_b,
                surf_feats_b,
                dw_b,
                corr_b,
                tau_obs_t,
                delta_tau_t,
                df_pay_diff_b,
                f_w_b,
                v_w_b,
                p_proxy_b,
                q_proxy_b,
                l_put_proxy_b,
                repo_spread_avg,
            )
            if theta_row_data is not None:
                with torch.no_grad():
                    b_th_info = theta_row_data["base"]
                    sp0_d = state.spots_t0.detach().to(dtype=torch.float64, device=device)
                    m0_th = (sp0_d / ref_spots).unsqueeze(0)
                    nf_th, ef_th, am_th, sf_th, sm_th, gf_th = _build_surr_tensors(
                        1,
                        m0_th,
                        theta_row_data["fp_rel"].unsqueeze(0),
                        theta_row_data["fp_perf"].unsqueeze(0),
                        theta_row_data["cd_lev"].unsqueeze(0),
                        theta_row_data["v_perf"].unsqueeze(0),
                        theta_row_data["sf_feats"].unsqueeze(0),
                        b_th_info.proxy_delta_weights.unsqueeze(0),
                        corr_sym_0.detach().unsqueeze(0),
                        theta_row_data["tau_obs"],
                        theta_row_data["delta_tau"],
                        theta_row_data["df_pay"].unsqueeze(0),
                        b_th_info.f_w.unsqueeze(0),
                        b_th_info.v_w.unsqueeze(0),
                        b_th_info.p_proxy.unsqueeze(0),
                        b_th_info.q_proxy.unsqueeze(0),
                        b_th_info.l_put_proxy.unsqueeze(0),
                        div_sched.repo_spread_avg.to(dtype=torch.float64, device=device),
                    )
                nf_all = torch.cat([nf_all, nf_th], dim=0)
                ef_all = torch.cat([ef_all, ef_th], dim=0)
                am_all = torch.cat([am_all, am_th], dim=0)
                sf_all = torch.cat([sf_all, sf_th], dim=0)
                sm_all = torch.cat([sm_all, sm_th], dim=0)
                gf_all = torch.cat([gf_all, gf_th], dim=0)

        if device.type == "cpu" and surr_dev.type == "cuda":
            packed_cpu = torch.cat(
                [nf_all.reshape(-1), ef_all.reshape(-1), sf_all.reshape(-1), gf_all.reshape(-1)], dim=0
            )
            am_key = ("am", str(surr_dev), str(surr_dtype), tuple(am_all.shape))
            am_g = _SURR_MASK_REGISTRY.get(am_key)
            if am_g is None:
                am_g = torch.ones(am_all.shape, device=surr_dev, dtype=surr_dtype)
                _SURR_MASK_REGISTRY.put(am_key, am_g)
            sm_key = ("sm", str(surr_dev), str(surr_dtype), tuple(sm_all.shape))
            sm_g = _SURR_MASK_REGISTRY.get(sm_key)
            if sm_g is None:
                sm_g = torch.ones(sm_all.shape, device=surr_dev, dtype=surr_dtype)
                _SURR_MASK_REGISTRY.put(sm_key, sm_g)
            box_cuda: list[Any] = []
            d_haz_t2, d_cpn_t2, d_put_t2 = _PackedCudaSurrogateFn.apply(  # type: ignore[no-untyped-call]
                packed_cpu,
                eff_surrogate,
                tuple(nf_all.shape),
                tuple(ef_all.shape),
                tuple(sf_all.shape),
                tuple(gf_all.shape),
                am_g,
                sm_g,
                surr_dev,
                surr_dtype,
                m_rem,
                box_cuda,
            )
            d_haz_diff = d_haz_t2[:b_diff, :m_rem]
            d_cpn_diff = d_cpn_t2[:b_diff, :m_rem]
            d_put_diff = d_put_t2[:b_diff]
        else:
            d_haz_t2, d_cpn_t2, d_put_t2 = _call_surrogate_residuals_fast(
                eff_surrogate,
                cast_to_surrogate_boundary(nf_all, device=surr_dev, dtype=surr_dtype),
                cast_to_surrogate_boundary(ef_all, device=surr_dev, dtype=surr_dtype),
                cast_to_surrogate_boundary(am_all, device=surr_dev, dtype=surr_dtype),
                cast_to_surrogate_boundary(sf_all, device=surr_dev, dtype=surr_dtype),
                cast_to_surrogate_boundary(sm_all, device=surr_dev, dtype=surr_dtype),
                cast_to_surrogate_boundary(gf_all, device=surr_dev, dtype=surr_dtype),
                require_second_order_sdpa=False,
            )
            d_haz_diff = cast_from_surrogate_boundary(d_haz_t2[:b_diff, :m_rem], device=device, dtype=torch.float64)
            d_cpn_diff = cast_from_surrogate_boundary(d_cpn_t2[:b_diff, :m_rem], device=device, dtype=torch.float64)
            d_put_diff = cast_from_surrogate_boundary(d_put_t2[:b_diff], device=device, dtype=torch.float64)
    else:
        d_haz_diff = torch.zeros((b_diff, m_rem), dtype=torch.float64, device=device)
        d_cpn_diff = torch.zeros((b_diff, m_rem), dtype=torch.float64, device=device)
        d_put_diff = torch.zeros((b_diff,), dtype=torch.float64, device=device)

    nf_empty = torch.empty((0,), dtype=torch.float64, device=device)
    if eff_surrogate is None:
        nf_all = ef_all = am_all = sf_all = sm_all = gf_all = nf_empty
    return _eval_post_surr_tail(
        state=state,
        device=device,
        m_rem=m_rem,
        b_diff=b_diff,
        eff_surrogate=eff_surrogate,
        use_cpp_glue=use_cpp_glue,
        nf_all=nf_all,
        ef_all=ef_all,
        am_all=am_all,
        sf_all=sf_all,
        sm_all=sm_all,
        gf_all=gf_all,
        p_proxy_b=p_proxy_b,
        q_proxy_b=q_proxy_b,
        l_put_proxy_b=l_put_proxy_b,
        l_call_b=l_call_b,
        l_onestar_b=l_onestar_b,
        df_pay_diff_b=df_pay_diff_b,
        w_fund_b=w_fund_b,
        pv_in_transit_b=pv_in_transit_b,
        theta_row_data=theta_row_data,
        ic_t=ic_t,
        ig_t=ig_t,
        cpn_t=cpn_t,
        ic_np_s=ic_np_s,
        ig_np_s=ig_np_s,
        cpn_np_s=cpn_np_s,
        k_eff_put_v=k_eff_put_v,
        precomputed_residuals=(d_haz_diff, d_cpn_diff, d_put_diff, d_haz_t2, d_cpn_t2, d_put_t2),
    )


def _eval_post_surr_tail(
    *,
    state: AutocallPreprocState,
    device: torch.device,
    m_rem: int,
    b_diff: int,
    eff_surrogate: MaskedEQAutocallSurrogate | None,
    use_cpp_glue: bool,
    nf_all: torch.Tensor,
    ef_all: torch.Tensor,
    am_all: torch.Tensor,
    sf_all: torch.Tensor,
    sm_all: torch.Tensor,
    gf_all: torch.Tensor,
    p_proxy_b: torch.Tensor,
    q_proxy_b: torch.Tensor,
    l_put_proxy_b: torch.Tensor,
    l_call_b: torch.Tensor,
    l_onestar_b: torch.Tensor,
    df_pay_diff_b: torch.Tensor,
    w_fund_b: torch.Tensor | None,
    pv_in_transit_b: torch.Tensor,
    theta_row_data: dict[str, Any] | None,
    ic_t: torch.Tensor,
    ig_t: torch.Tensor,
    cpn_t: torch.Tensor,
    ic_np_s: npt.NDArray[np.float64],
    ig_np_s: npt.NDArray[np.float64],
    cpn_np_s: npt.NDArray[np.float64],
    k_eff_put_v: float,
    precomputed_residuals: tuple[Any, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, AutocallPostprocOutput, torch.Tensor | None]:
    """Runs surrogate residuals (if not precomputed), post-surrogate PV assembly, and optional Theta PV."""
    d_haz_t2: torch.Tensor | None = None
    d_cpn_t2: torch.Tensor | None = None
    d_put_t2: torch.Tensor | None = None
    if precomputed_residuals is not None:
        d_haz_diff, d_cpn_diff, d_put_diff, d_haz_t2, d_cpn_t2, d_put_t2 = precomputed_residuals
    elif eff_surrogate is not None:
        first_param = next(eff_surrogate.parameters(), None)
        surr_dtype = first_param.dtype if first_param is not None else torch.float32
        surr_dev = first_param.device if first_param is not None else device
        if device.type == "cpu" and surr_dev.type == "cuda":
            packed_cpu = torch.cat(
                [nf_all.reshape(-1), ef_all.reshape(-1), sf_all.reshape(-1), gf_all.reshape(-1)], dim=0
            )
            am_key = ("am", str(surr_dev), str(surr_dtype), tuple(am_all.shape))
            am_g = _SURR_MASK_REGISTRY.get(am_key)
            if am_g is None:
                am_g = torch.ones(am_all.shape, device=surr_dev, dtype=surr_dtype)
                _SURR_MASK_REGISTRY.put(am_key, am_g)
            sm_key = ("sm", str(surr_dev), str(surr_dtype), tuple(sm_all.shape))
            sm_g = _SURR_MASK_REGISTRY.get(sm_key)
            if sm_g is None:
                sm_g = torch.ones(sm_all.shape, device=surr_dev, dtype=surr_dtype)
                _SURR_MASK_REGISTRY.put(sm_key, sm_g)
            box_cuda: list[Any] = []
            d_haz_t2, d_cpn_t2, d_put_t2 = _PackedCudaSurrogateFn.apply(  # type: ignore[no-untyped-call]
                packed_cpu,
                eff_surrogate,
                tuple(nf_all.shape),
                tuple(ef_all.shape),
                tuple(sf_all.shape),
                tuple(gf_all.shape),
                am_g,
                sm_g,
                surr_dev,
                surr_dtype,
                m_rem,
                box_cuda,
            )
            d_haz_diff = d_haz_t2[:b_diff, :m_rem]
            d_cpn_diff = d_cpn_t2[:b_diff, :m_rem]
            d_put_diff = d_put_t2[:b_diff]
        else:
            d_haz_t2, d_cpn_t2, d_put_t2 = _call_surrogate_residuals_fast(
                eff_surrogate,
                cast_to_surrogate_boundary(nf_all, device=surr_dev, dtype=surr_dtype),
                cast_to_surrogate_boundary(ef_all, device=surr_dev, dtype=surr_dtype),
                cast_to_surrogate_boundary(am_all, device=surr_dev, dtype=surr_dtype),
                cast_to_surrogate_boundary(sf_all, device=surr_dev, dtype=surr_dtype),
                cast_to_surrogate_boundary(sm_all, device=surr_dev, dtype=surr_dtype),
                cast_to_surrogate_boundary(gf_all, device=surr_dev, dtype=surr_dtype),
                require_second_order_sdpa=False,
            )
            d_haz_diff = cast_from_surrogate_boundary(d_haz_t2[:b_diff, :m_rem], device=device, dtype=torch.float64)
            d_cpn_diff = cast_from_surrogate_boundary(d_cpn_t2[:b_diff, :m_rem], device=device, dtype=torch.float64)
            d_put_diff = cast_from_surrogate_boundary(d_put_t2[:b_diff], device=device, dtype=torch.float64)
    else:
        d_haz_t2 = d_cpn_t2 = d_put_t2 = None
        d_haz_diff = torch.zeros((b_diff, m_rem), dtype=torch.float64, device=device)
        d_cpn_diff = torch.zeros((b_diff, m_rem), dtype=torch.float64, device=device)
        d_put_diff = torch.zeros((b_diff,), dtype=torch.float64, device=device)

    if use_cpp_glue:
        post_static_meta = (
            ic_np_s,
            ig_np_s,
            cpn_np_s,
            1 if eff_surrogate is not None else 0,
            1 if state.no_notional_repayment else 0,
            1 if state.force_coupons_on_autocall else 0,
            1 if state.is_memory else 0,
            1 if state.is_long_basket else 0,
            float(state.c_mem_0),
            float(state.autocall_coupon),
            float(state.option_gearing),
            float(state.participation_call_rate),
            k_eff_put_v,
        )
        post_box: list[Any] = []
        pv_batch, p_stop_0_t, q_cpn_0_t, l_put_0_t = _SurrPostPvCppFn.apply(  # type: ignore[no-untyped-call]
            d_haz_diff,
            d_cpn_diff,
            d_put_diff,
            p_proxy_b,
            q_proxy_b,
            l_put_proxy_b,
            l_call_b,
            l_onestar_b,
            df_pay_diff_b,
            w_fund_b,
            pv_in_transit_b,
            post_static_meta,
            post_box,
        )
    else:
        if eff_surrogate is not None:
            p_px_c = p_proxy_b.clamp(0.0, 1.0) * ic_t.unsqueeze(0)
            surv_prev_px = (1.0 - (torch.cumsum(p_px_c, dim=1) - p_px_c)).clamp(min=1e-6)
            h_px_c = (p_px_c / surv_prev_px).clamp(min=1e-6, max=1.0 - 1e-6)
            haz_b = ic_t.unsqueeze(0) * torch.sigmoid(torch.log(h_px_c / (1.0 - h_px_c)) + d_haz_diff)
            surv_b = torch.cumprod(1.0 - haz_b, dim=1)
            surv_prev_b = torch.cat([torch.ones((b_diff, 1), dtype=torch.float64, device=device), surv_b[:, :-1]], dim=1)
            p_stop_b = surv_prev_b * haz_b

            q_px_c = q_proxy_b.clamp(min=1e-6, max=1.0 - 1e-6)
            q_gate = (q_proxy_b * q_proxy_b) / (q_proxy_b * q_proxy_b + 1e-10)
            q_pred_b = q_gate * torch.sigmoid(torch.log(q_px_c / (1.0 - q_px_c)) + d_cpn_diff)
            q_cpn_b = ig_t.unsqueeze(0) + (1.0 - ig_t.unsqueeze(0)) * q_pred_b

            surv_px_term = (1.0 - p_px_c.sum(dim=1)).clamp(min=0.0)
            r_px = torch.clamp(l_put_proxy_b.clamp(min=0.0) / torch.clamp(k_eff_put_v * surv_px_term, min=1e-12), min=1e-12, max=1.0 - 1e-12)
            l_put_b = (k_eff_put_v * surv_b[:, -1]) * torch.sigmoid(torch.log(r_px / (1.0 - r_px)) + d_put_diff)
        else:
            p_stop_b = p_proxy_b
            q_cpn_b = q_proxy_b
            l_put_b = l_put_proxy_b
        pv_batch = _assemble_pv_batch_torch(
            state,
            p_stop=p_stop_b,
            q_cpn=q_cpn_b,
            l_put=l_put_b,
            l_call=l_call_b,
            l_onestar=l_onestar_b,
            df_pay_b=df_pay_diff_b,
            w_fund_b=w_fund_b,
            pv_in_transit_b=pv_in_transit_b,
        )
        p_stop_0_t = p_stop_b[0].detach()
        q_cpn_0_t = q_cpn_b[0].detach()
        l_put_0_t = l_put_b[0].detach()

    if use_cpp_glue and state.funding_leg is None and m_rem > 0 and not state.already_autocalled:
        p0_np = p_stop_0_t.detach().cpu().numpy() * ic_np_s
        q0_np = q_cpn_0_t.detach().cpu().numpy()
        lp0_f = float(l_put_0_t.detach().cpu().item())
        lc0_f = float(l_call_b[0].detach().cpu().item())
        lo0_f = float(l_onestar_b[0].detach().cpu().item())
        dfp0_np = df_pay_diff_b[0].detach().cpu().numpy()
        surv_np = np.maximum(1.0 - np.cumsum(p0_np), 0.0)
        surv_prev_np = np.concatenate([np.ones(1, dtype=np.float64), surv_np[:-1]])
        if state.no_notional_repayment:
            not_cf_np = np.zeros(m_rem, dtype=np.float64)
        else:
            not_cf_np = dfp0_np * p0_np
            not_cf_np[-1] += dfp0_np[-1] * surv_np[-1]
        eff_q0 = np.maximum(q0_np, p0_np / np.maximum(surv_prev_np, 1e-12)) if state.force_coupons_on_autocall else q0_np
        if state.is_memory:
            cpn_cf_np = np.empty(m_rem, dtype=np.float64)
            mem_v = float(state.c_mem_0)
            ac_v = float(state.autocall_coupon)
            for m_i in range(m_rem):
                ec_i = float(cpn_np_s[m_i]) + mem_v
                eq_i = float(eff_q0[m_i])
                cpn_cf_np[m_i] = float(dfp0_np[m_i]) * (float(surv_prev_np[m_i]) * eq_i * ec_i + float(p0_np[m_i]) * ac_v)
                mem_v = ec_i * (1.0 - eq_i)
        else:
            cpn_cf_np = dfp0_np * (surv_prev_np * eff_q0 * cpn_np_s + p0_np * float(state.autocall_coupon))
        put_sign_f = 1.0 if state.is_long_basket else -1.0
        opt_pv_f = float(dfp0_np[-1]) * (put_sign_f * float(state.option_gearing) * lp0_f + float(state.participation_call_rate) * lc0_f + lo0_f)
        ccy_s = state.payment_currency
        cash_today_f = float(getattr(state, "cash_to_settle_today", getattr(state.seasoned_state, "cash_to_settle_today", 0.0)))
        cfs: list[CashFlowItem] = list(state.seasoned_state.past_cashflows)
        pay_dts = state.seasoned_state.future_pay_dates
        for m_i in range(m_rem):
            pay_dt_i = pay_dts[m_i]
            cfs.append(CashFlowItem("FundingLeg", pay_dt_i, 0.0, 0.0, ccy_s, 0, "Estimate"))
            cfs.append(CashFlowItem("Coupon", pay_dt_i, float(cpn_cf_np[m_i]), 0.0, ccy_s, 0, "Estimate"))
            if m_i == m_rem - 1:
                cfs.append(CashFlowItem("Option", pay_dt_i, opt_pv_f, 0.0, ccy_s, 0, "Estimate"))
            cfs.append(CashFlowItem("Notional", pay_dt_i, float(not_cf_np[m_i]), 0.0, ccy_s, 0, "Estimate"))
        post_out = AutocallPostprocOutput(
            product_value=pv_batch[0].detach(),
            notional_pv=torch.tensor(float(not_cf_np.sum()), dtype=torch.float64, device=device),
            coupon_pv=torch.tensor(float(cpn_cf_np.sum()), dtype=torch.float64, device=device),
            option_pv=torch.tensor(opt_pv_f, dtype=torch.float64, device=device),
            funding_pv=torch.tensor(0.0, dtype=torch.float64, device=device),
            pv_in_transit=torch.tensor(float(state.pv_in_transit), dtype=torch.float64, device=device),
            survival_probs=torch.from_numpy(surv_np).to(device=device),
            p_stop=p_stop_0_t.detach(),
            q_cpn=q_cpn_0_t.detach(),
            l_put=l_put_0_t.detach(),
            l_call=l_call_b[0].detach(),
            cashflows=cfs,
            standard_error=0.0,
            reporting_currency=ccy_s,
            cash_to_settle_today=cash_today_f,
            l_onestar=l_onestar_b[0].detach(),
        )
    else:
        post_out = assemble_autocall_cashflows_and_pv(
            state,
            p_stop=p_stop_0_t.detach(),
            q_cpn=q_cpn_0_t.detach(),
            l_put=l_put_0_t.detach(),
            l_call=l_call_b[0].detach(),
            l_onestar=l_onestar_b[0].detach(),
            build_cashflow_items=True,
        )
        post_out.product_value = pv_batch[0].detach()

    pv_theta_t: torch.Tensor | None = None
    if theta_row_data is not None:
        with torch.no_grad():
            if use_cpp_glue and theta_row_data.get("cpp_fast"):
                sp_fn_th = _SURR_POST_CPP_FN
                assert sp_fn_th is not None
                wf_th_np = theta_row_data["w_fund_np"]
                if eff_surrogate is not None:
                    assert d_haz_t2 is not None and d_cpn_t2 is not None and d_put_t2 is not None
                    dh_th_np = np.ascontiguousarray(d_haz_t2[-1, :m_rem].detach().cpu().numpy(), dtype=np.float64)
                    dc_th_np = np.ascontiguousarray(d_cpn_t2[-1, :m_rem].detach().cpu().numpy(), dtype=np.float64)
                    dp_th_np = np.ascontiguousarray(d_put_t2[-1].detach().cpu().numpy().reshape(1), dtype=np.float64)
                else:
                    dh_th_np = np.zeros((1, m_rem), dtype=np.float64)
                    dc_th_np = np.zeros((1, m_rem), dtype=np.float64)
                    dp_th_np = np.zeros((1,), dtype=np.float64)
                wf_th_arr = wf_th_np if wf_th_np is not None else np.zeros((1, m_rem), dtype=np.float64)
                z1_np = np.zeros((1,), dtype=np.float64)
                it_np = np.array([float(theta_row_data["in_transit_f"])], dtype=np.float64)
                lp_np_th = np.array([float(theta_row_data["lp_f"])], dtype=np.float64)
                th_p_nps = (
                    dh_th_np, dc_th_np, dp_th_np,
                    theta_row_data["p_np"], theta_row_data["q_np"], lp_np_th,
                    z1_np, z1_np, theta_row_data["df_pay_np"], wf_th_arr, it_np,
                )
                pv_th_out = np.empty((1,), dtype=np.float64)
                ps0_th = np.empty((m_rem,), dtype=np.float64)
                qc0_th = np.empty((m_rem,), dtype=np.float64)
                lp0_th = np.empty((1,), dtype=np.float64)
                sp_fn_th(
                    1, m_rem,
                    1 if eff_surrogate is not None else 0,
                    1 if wf_th_np is not None else 0,
                    1 if state.no_notional_repayment else 0,
                    1 if state.force_coupons_on_autocall else 0,
                    1 if state.is_memory else 0,
                    1 if state.is_long_basket else 0,
                    0,
                    float(state.c_mem_0), float(state.autocall_coupon),
                    float(state.option_gearing), float(state.participation_call_rate), k_eff_put_v,
                    *(a.__array_interface__["data"][0] for a in th_p_nps),
                    ic_np_s.__array_interface__["data"][0],
                    ig_np_s.__array_interface__["data"][0],
                    cpn_np_s.__array_interface__["data"][0],
                    pv_th_out.__array_interface__["data"][0],
                    ps0_th.__array_interface__["data"][0],
                    qc0_th.__array_interface__["data"][0],
                    lp0_th.__array_interface__["data"][0],
                    0, *([0] * 11),
                )
                pv_theta_t = torch.tensor(float(pv_th_out[0]), dtype=torch.float64, device=device)
            elif use_cpp_glue:
                b_th = theta_row_data["base"]
                sp_fn_th = _SURR_POST_CPP_FN
                assert sp_fn_th is not None
                wf_th = theta_row_data["w_fund"]
                if eff_surrogate is not None:
                    assert d_haz_t2 is not None and d_cpn_t2 is not None and d_put_t2 is not None
                    dh_th_np = np.ascontiguousarray(d_haz_t2[-1, :m_rem].detach().cpu().numpy(), dtype=np.float64)
                    dc_th_np = np.ascontiguousarray(d_cpn_t2[-1, :m_rem].detach().cpu().numpy(), dtype=np.float64)
                    dp_th_np = np.ascontiguousarray(d_put_t2[-1].detach().cpu().numpy().reshape(1), dtype=np.float64)
                else:
                    dh_th_np = np.zeros((1, m_rem), dtype=np.float64)
                    dc_th_np = np.zeros((1, m_rem), dtype=np.float64)
                    dp_th_np = np.zeros((1,), dtype=np.float64)
                wf_th_np = np.ascontiguousarray(wf_th.detach().cpu().numpy(), dtype=np.float64) if wf_th is not None else np.zeros((1, m_rem), dtype=np.float64)
                th_p_nps = (
                    dh_th_np,
                    dc_th_np,
                    dp_th_np,
                    np.ascontiguousarray(b_th.p_proxy.detach().cpu().numpy(), dtype=np.float64),
                    np.ascontiguousarray(b_th.q_proxy.detach().cpu().numpy(), dtype=np.float64),
                    np.ascontiguousarray(b_th.l_put_proxy.detach().cpu().numpy().reshape(1), dtype=np.float64),
                    np.ascontiguousarray(b_th.l_call_proxy.detach().cpu().numpy().reshape(1), dtype=np.float64),
                    np.ascontiguousarray(b_th.l_onestar_proxy.detach().cpu().numpy().reshape(1), dtype=np.float64),
                    np.ascontiguousarray(theta_row_data["df_pay"].detach().cpu().numpy(), dtype=np.float64),
                    wf_th_np,
                    np.ascontiguousarray(theta_row_data["in_transit"].detach().cpu().numpy().reshape(1), dtype=np.float64),
                )
                pv_th_out = np.empty((1,), dtype=np.float64)
                ps0_th = np.empty((m_rem,), dtype=np.float64)
                qc0_th = np.empty((m_rem,), dtype=np.float64)
                lp0_th = np.empty((1,), dtype=np.float64)
                sp_fn_th(
                    1, m_rem,
                    1 if eff_surrogate is not None else 0,
                    1 if wf_th is not None else 0,
                    1 if state.no_notional_repayment else 0,
                    1 if state.force_coupons_on_autocall else 0,
                    1 if state.is_memory else 0,
                    1 if state.is_long_basket else 0,
                    0,
                    float(state.c_mem_0), float(state.autocall_coupon),
                    float(state.option_gearing), float(state.participation_call_rate), k_eff_put_v,
                    *(a.ctypes.data for a in th_p_nps),
                    ic_np_s.ctypes.data, ig_np_s.ctypes.data, cpn_np_s.ctypes.data,
                    pv_th_out.ctypes.data, ps0_th.ctypes.data, qc0_th.ctypes.data, lp0_th.ctypes.data,
                    0, *([0] * 11),
                )
                pv_theta_t = torch.tensor(float(pv_th_out[0]), dtype=torch.float64, device=device)
            else:
                b_th = theta_row_data["base"]
                if eff_surrogate is not None:
                    assert d_haz_t2 is not None and d_cpn_t2 is not None and d_put_t2 is not None
                    dh_th = cast_from_surrogate_boundary(d_haz_t2[-1, :m_rem], device=device, dtype=torch.float64)
                    dc_th = cast_from_surrogate_boundary(d_cpn_t2[-1, :m_rem], device=device, dtype=torch.float64)
                    dp_th = cast_from_surrogate_boundary(d_put_t2[-1], device=device, dtype=torch.float64)
                    p_c_th = b_th.p_proxy.clamp(0.0, 1.0) * ic_t
                    sp_px_th = (1.0 - (torch.cumsum(p_c_th, dim=0) - p_c_th)).clamp(min=1e-6)
                    hp_th = (p_c_th / sp_px_th).clamp(min=1e-6, max=1.0 - 1e-6)
                    haz_th = ic_t * torch.sigmoid(torch.log(hp_th / (1.0 - hp_th)) + dh_th)
                    sv_th = torch.cumprod(1.0 - haz_th, dim=0)
                    sp_th = torch.cat([torch.ones(1, dtype=torch.float64, device=device), sv_th[:-1]], dim=0)
                    p_th = sp_th * haz_th
                    qp_th = b_th.q_proxy.clamp(min=1e-6, max=1.0 - 1e-6)
                    qg_th = (b_th.q_proxy * b_th.q_proxy) / (b_th.q_proxy * b_th.q_proxy + 1e-10)
                    q_th = ig_t + (1.0 - ig_t) * qg_th * torch.sigmoid(torch.log(qp_th / (1.0 - qp_th)) + dc_th)
                    rp_th = torch.clamp(b_th.l_put_proxy.clamp(min=0.0) / torch.clamp(k_eff_put_v * (1.0 - p_c_th.sum()).clamp(min=0.0), min=1e-12), min=1e-12, max=1.0 - 1e-12)
                    lp_th = (k_eff_put_v * sv_th[-1]) * torch.sigmoid(torch.log(rp_th / (1.0 - rp_th)) + dp_th)
                else:
                    p_th, q_th, lp_th = b_th.p_proxy, b_th.q_proxy, b_th.l_put_proxy
                wf_th = theta_row_data["w_fund"]
                pv_theta_t = _assemble_pv_batch_torch(
                    state,
                    p_stop=p_th.unsqueeze(0),
                    q_cpn=q_th.unsqueeze(0),
                    l_put=lp_th.unsqueeze(0),
                    l_call=b_th.l_call_proxy.unsqueeze(0),
                    l_onestar=b_th.l_onestar_proxy.unsqueeze(0),
                    df_pay_b=theta_row_data["df_pay"].unsqueeze(0),
                    w_fund_b=wf_th.unsqueeze(0) if wf_th is not None else None,
                    pv_in_transit_b=theta_row_data["in_transit"].unsqueeze(0),
                )[0]

    return pv_batch[0], pv_batch, post_out, pv_theta_t


_STATE_ALL_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(AutocallPreprocState))
_STATE_ACTIVE_CPU_FIELDS: tuple[str, ...] = (
    "spots_t0",
    "ref_spots",
    "corr_matrix",
    "tau_obs",
    "delta_tau",
    "df_obs",
    "df_pay",
    "ko_barriers",
    "is_callable",
    "cpn_barriers",
    "is_guaranteed_cpn",
    "coupon_amounts",
)


def _state_to_cpu(state: AutocallPreprocState) -> AutocallPreprocState:
    """Returns a shallow copy of AutocallPreprocState with active tensor fields placed on CPU."""
    field_names = _STATE_ACTIVE_CPU_FIELDS if (state.m_rem > 0 and not state.already_autocalled) else _STATE_ALL_FIELD_NAMES
    updates: dict[str, Any] = {}
    for fname in field_names:
        val = getattr(state, fname)
        if isinstance(val, torch.Tensor) and val.device.type != "cpu":
            updates[fname] = val.detach().cpu()
    if not updates:
        return state
    return replace(state, **updates)


def compute_eq_autocall_greeks(
    state_or_request: AutocallPreprocState | Any,
    *,
    queries: Sequence[str] | None = None,
    surrogate: MaskedEQAutocallSurrogate | None = None,
    theta_day_fraction: float = 1.0 / 365.0,
) -> AutocallGreeksResult:
    """Computes `ProductValue`, `CashFlows`, and all requested Greek metrics in IEEE-754 float64.

    Uses decoupled float64/float32 single-VJP surrogate AD across native `.to()` casts,
    Forward-over-Reverse `jvp(grad)` exact D x D Gamma with `require_second_order_sdpa=True`
    (never mutating global `sdpa_kernel` flags), single-tangent Euler JVP for `CashGamma`-only,
    multi-curve Funding + Asset + Repo `Rho`, true 1-day calendar-roll `Theta`, and live
    differentiable quanto + FX Vega/Cega. Zero result caching or shared-object aliasing.
    """
    if isinstance(state_or_request, AutocallPreprocState):
        state = state_or_request
    else:
        state = preprocess_eq_autocall(state_or_request)

    orig_device = state.spots_t0.device
    if orig_device.type == "cuda":
        state = _state_to_cpu(state)

    raw_queries = queries if queries is not None else state.queries
    q_set = {str(getattr(q, "value", q)) for q in raw_queries}

    want_delta = "Delta" in q_set
    want_gamma = "Gamma" in q_set
    want_cash_gamma = "MultivariateAggregatedCashGamma" in q_set
    want_vega = "Vega" in q_set
    want_rho = "Rho" in q_set
    want_divs = "DividendsSensitivities" in q_set
    want_theta = "Theta" in q_set
    want_cega = "Cega" in q_set or "ConvexCega" in q_set

    d_assets = state.d_assets
    m_rem = state.m_rem
    orig_ids = state.original_identifiers
    sorted_ids = state.sorted_identifiers
    inv_perm = state.inv_lex_perm
    device = state.spots_t0.device
    as_of_orig = state.seasoned_state.as_of_date

    disc_curve = _normalize_curve_obj(state.discount_curve, as_of_date=as_of_orig)
    disc_id = str(
        getattr(disc_curve, "identifier", getattr(disc_curve, "Identifier", state.discount_curve_id or "DEFAULT"))
    )

    def _curve_n_pillars(c_obj: Any) -> int:
        if c_obj is None:
            return 0
        if hasattr(c_obj, "tenors"):
            return len(c_obj.tenors)
        if isinstance(c_obj, Mapping) and "RatesSchedule" in c_obj:
            return len(c_obj["RatesSchedule"].get("Rates", []))
        return 0

    k_disc_pillars = _curve_n_pillars(disc_curve)

    eff_surrogate = (
        None
        if (
            surrogate is not None
            and hasattr(surrogate, "is_zero_residual")
            and surrogate.is_zero_residual()
        )
        else surrogate
    )

    need_spot_grad = want_delta or want_gamma
    spots_var = state.spots_t0.detach().clone().to(dtype=torch.float64, device=device)
    spots_var.requires_grad_(need_spot_grad)

    corr_var = state.corr_matrix.detach().clone().to(dtype=torch.float64, device=device)
    corr_var.requires_grad_(want_cega and d_assets > 1)

    pillar_deltas: torch.Tensor | None = None
    if want_rho and k_disc_pillars > 0:
        pillar_deltas = torch.zeros(k_disc_pillars, dtype=torch.float64, device=device, requires_grad=True)

    # Distinct funding curve pillar deltas when FundingLeg forward curve != discount curve
    funding_pillar_deltas: torch.Tensor | None = None
    fin_curve_id: str | None = None
    k_fin_pillars = 0
    if want_rho and state.funding_leg is not None:
        f_curve_leg = _normalize_curve_obj(state.funding_leg.get("funding_curve"), as_of_date=as_of_orig)
        f_cid_raw = state.funding_leg.get("funding_curve_id")
        fin_curve_id = str(getattr(f_curve_leg, "identifier", f_cid_raw or disc_id))
        same_disc_fin = (
            f_curve_leg is None
            or f_curve_leg is state.discount_curve
            or (fin_curve_id == disc_id)
        )
        if not same_disc_fin and f_curve_leg is not None:
            k_fin_pillars = _curve_n_pillars(f_curve_leg)
            if k_fin_pillars > 0:
                funding_pillar_deltas = torch.zeros(
                    k_fin_pillars, dtype=torch.float64, device=device, requires_grad=True
                )

    # Distinct asset financing curves (multi-currency baskets)
    extra_curve_vars: dict[str, torch.Tensor] = {}
    asset_curve_deltas: dict[str, torch.Tensor] = {}
    if want_rho and m_rem > 0 and state.asset_financing_curves:
        for ticker in sorted_ids:
            c_i = _normalize_curve_obj(state.asset_financing_curves.get(ticker), as_of_date=as_of_orig)
            if c_i is None:
                continue
            cid_i = str(getattr(c_i, "identifier", getattr(c_i, "Identifier", "")))
            if not cid_i or cid_i == disc_id or (fin_curve_id is not None and cid_i == fin_curve_id):
                continue
            if cid_i not in extra_curve_vars:
                k_ci = _curve_n_pillars(c_i)
                if k_ci > 0:
                    extra_curve_vars[cid_i] = torch.zeros(
                        k_ci, dtype=torch.float64, device=device, requires_grad=True
                    )
            if cid_i in extra_curve_vars:
                asset_curve_deltas[ticker] = extra_curve_vars[cid_i]

    # Optional RepoCurve sensitivities
    repo_curves_map = _extract_repo_curves_map(state_or_request, sorted_ids, as_of_orig)
    repo_deltas: dict[str, torch.Tensor] = {}
    if want_rho and m_rem > 0 and repo_curves_map:
        for ticker, r_curve_i in repo_curves_map.items():
            k_ri = _curve_n_pillars(r_curve_i)
            if k_ri > 0:
                repo_deltas[ticker] = torch.zeros(k_ri, dtype=torch.float64, device=device, requires_grad=True)

    surr_first_p = next(eff_surrogate.parameters(), None) if eff_surrogate is not None else None
    is_f32_surr = surr_first_p is not None and surr_first_p.dtype == torch.float32
    use_batched_gamma = (
        (want_gamma or want_cash_gamma)
        and m_rem > 0
        and not state.already_autocalled
        and is_f32_surr
    )

    div_vars: dict[str, torch.Tensor] = {}
    div_grad_tickers: list[str] = []
    if want_divs:
        for ticker in sorted_ids:
            base_amt = state.dividend_schedule.dividend_amount_tensors.get(ticker)
            if base_amt is not None and base_amt.numel() > 0:
                if not use_batched_gamma:
                    t_div = base_amt.detach().clone().to(dtype=torch.float64, device=device).requires_grad_(True)
                    div_vars[ticker] = t_div
                div_grad_tickers.append(ticker)
    div_grad_set = set(div_grad_tickers)

    vol_deltas: dict[str, torch.Tensor] = {}
    vol_grad_tickers: list[str] = []
    if want_vega and m_rem > 0 and not state.already_autocalled:
        for ticker in sorted_ids:
            surf = state.vol_surfaces.get(ticker)
            if surf is None:
                continue
            if not use_batched_gamma:
                v_del = torch.zeros(
                    (surf.n_exp, surf.n_raw_strikes),
                    dtype=torch.float64,
                    device=device,
                    requires_grad=True,
                )
                vol_deltas[ticker] = v_del
            vol_grad_tickers.append(ticker)
    vol_grad_set = set(vol_grad_tickers)

    # Optional FX Vega and Equity-FX Cega when quanto metadata is present
    quanto_meta = getattr(state, "quanto_meta", None) or {}
    fx_vol_deltas: dict[str, torch.Tensor] = {}
    fx_corr_deltas: dict[str, torch.Tensor] = {}
    if m_rem > 0 and quanto_meta:
        for ticker in sorted_ids:
            qm = quanto_meta.get(ticker)
            if not qm:
                continue
            if want_vega:
                fx_vol_deltas[ticker] = torch.zeros(m_rem, dtype=torch.float64, device=device, requires_grad=True)
            if want_cega:
                fx_corr_deltas[ticker] = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    spots_gamma_leaf: torch.Tensor | None = None
    h_gamma_vec: torch.Tensor | None = None
    h_euler_rel: float = 1.0e-3
    pv_theta_batched: torch.Tensor | None = None

    cuda_graph_ctx: Any = contextlib.nullcontext()
    grad_map: dict[tuple[str, str], torch.Tensor] = {}
    iv_0_delta_leaf: torch.Tensor | None = None
    iv_q0_delta_leaf: torch.Tensor | None = None
    div_pv_delta_leaf: torch.Tensor | None = None
    df_pay_mult_leaf: torch.Tensor | None = None
    df_fin_mult_leaf: torch.Tensor | None = None
    df_repo_mult_leaf: torch.Tensor | None = None
    w_fund_delta_leaf: torch.Tensor | None = None
    in_transit_delta_leaf: torch.Tensor | None = None
    batch_aux: dict[str, Any] = {}
    with cuda_graph_ctx:
        if use_batched_gamma:
            base_spots_b = state.spots_t0.detach().to(dtype=torch.float64, device=device)
            h_gamma_vec = torch.clamp(base_spots_b * h_euler_rel, min=1e-6)
            eye_g = torch.eye(d_assets, dtype=torch.float64, device=device)
            sp_rows: list[torch.Tensor] = []
            for gi in range(d_assets):
                sp_rows.append(base_spots_b + h_gamma_vec[gi] * eye_g[gi])
                sp_rows.append(base_spots_b - h_gamma_vec[gi] * eye_g[gi])
            spots_gamma_leaf = torch.stack(sp_rows, dim=0).detach().requires_grad_(True)

            if want_vega:
                iv_0_delta_leaf = torch.zeros(
                    (d_assets, m_rem + 5), dtype=torch.float64, device=device, requires_grad=True
                )
                if quanto_meta:
                    iv_q0_delta_leaf = torch.zeros(
                        (d_assets, m_rem), dtype=torch.float64, device=device, requires_grad=True
                    )
            if want_divs or want_rho:
                div_pv_delta_leaf = torch.zeros(
                    (d_assets, m_rem), dtype=torch.float64, device=device, requires_grad=True
                )
            if want_rho:
                df_pay_mult_leaf = torch.ones(m_rem, dtype=torch.float64, device=device, requires_grad=True)
                df_fin_mult_leaf = torch.ones((d_assets, m_rem), dtype=torch.float64, device=device, requires_grad=True)
                if repo_deltas:
                    df_repo_mult_leaf = torch.ones(
                        (d_assets, m_rem), dtype=torch.float64, device=device, requires_grad=True
                    )
                if state.funding_leg is not None:
                    w_fund_delta_leaf = torch.zeros(m_rem, dtype=torch.float64, device=device, requires_grad=True)
                if state.seasoned_state.in_transit_indices:
                    in_transit_delta_leaf = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)

            can_batch_theta = (
                want_theta
                and abs(float(theta_day_fraction) - 1.0 / 365.0) < 1e-6
                and isinstance(as_of_orig, dt.datetime)
            )
            pv_t, pv_loss_target, post_out, pv_theta_batched = _evaluate_greeks_single_batch(
                state,
                spots_var=spots_var,
                spots_gamma=spots_gamma_leaf,
                corr_var=corr_var,
                div_vars=None,
                vol_deltas=None,
                pillar_deltas=None,
                funding_pillar_deltas=None,
                asset_curve_deltas=None,
                repo_deltas=None,
                repo_curves=repo_curves_map if repo_curves_map else None,
                fx_vol_deltas=fx_vol_deltas if fx_vol_deltas else None,
                fx_corr_deltas=fx_corr_deltas if fx_corr_deltas else None,
                eff_surrogate=eff_surrogate,
                include_theta=can_batch_theta,
                iv_0_delta_leaf=iv_0_delta_leaf,
                iv_q0_delta_leaf=iv_q0_delta_leaf,
                div_pv_delta_leaf=div_pv_delta_leaf,
                df_pay_mult_leaf=df_pay_mult_leaf,
                df_fin_mult_leaf=df_fin_mult_leaf,
                df_repo_mult_leaf=df_repo_mult_leaf,
                w_fund_delta_leaf=w_fund_delta_leaf,
                in_transit_delta_leaf=in_transit_delta_leaf,
                out_aux=batch_aux,
            )
            pv_sum_target = pv_loss_target.sum()
        else:
            pv_res = evaluate_differentiable_pv(
                state,
                spots_t0=spots_var,
                corr_matrix=corr_var,
                div_amounts=div_vars if want_divs else None,
                vol_quote_deltas=vol_deltas if want_vega else None,
                curve_pillar_deltas=pillar_deltas,
                funding_pillar_deltas=funding_pillar_deltas,
                asset_curve_pillar_deltas=asset_curve_deltas if asset_curve_deltas else None,
                repo_pillar_deltas=repo_deltas if repo_deltas else None,
                repo_curves=repo_curves_map if repo_curves_map else None,
                fx_vol_deltas=fx_vol_deltas if fx_vol_deltas else None,
                fx_corr_deltas=fx_corr_deltas if fx_corr_deltas else None,
                surrogate=eff_surrogate,
                require_second_order_sdpa=False,
                return_postproc=True,
            )
            if not isinstance(pv_res, tuple):
                raise RuntimeError("Expected tuple (pv_t, post_out) when return_postproc=True.")
            pv_t, post_out = pv_res
            pv_sum_target = pv_t

        grad_inputs: list[torch.Tensor] = []
        input_tags: list[tuple[str, str]] = []
        if need_spot_grad and pv_t.requires_grad:
            grad_inputs.append(spots_var)
            input_tags.append(("spots", ""))
        if spots_gamma_leaf is not None and pv_sum_target.requires_grad:
            grad_inputs.append(spots_gamma_leaf)
            input_tags.append(("spots_gamma", ""))
        if want_cega and d_assets > 1 and pv_t.requires_grad:
            grad_inputs.append(corr_var)
            input_tags.append(("corr", ""))
        if use_batched_gamma:
            if iv_0_delta_leaf is not None and pv_t.requires_grad:
                grad_inputs.append(iv_0_delta_leaf)
                input_tags.append(("aux_iv_0", ""))
            if iv_q0_delta_leaf is not None and pv_t.requires_grad:
                grad_inputs.append(iv_q0_delta_leaf)
                input_tags.append(("aux_iv_q0", ""))
            if div_pv_delta_leaf is not None and pv_t.requires_grad:
                grad_inputs.append(div_pv_delta_leaf)
                input_tags.append(("aux_div_pv", ""))
            if df_pay_mult_leaf is not None and pv_t.requires_grad:
                grad_inputs.append(df_pay_mult_leaf)
                input_tags.append(("aux_df_pay", ""))
            if df_fin_mult_leaf is not None and pv_t.requires_grad:
                grad_inputs.append(df_fin_mult_leaf)
                input_tags.append(("aux_df_fin", ""))
            if df_repo_mult_leaf is not None and pv_t.requires_grad:
                grad_inputs.append(df_repo_mult_leaf)
                input_tags.append(("aux_df_repo", ""))
            if w_fund_delta_leaf is not None and pv_t.requires_grad:
                grad_inputs.append(w_fund_delta_leaf)
                input_tags.append(("aux_w_fund", ""))
            if in_transit_delta_leaf is not None and pv_t.requires_grad:
                grad_inputs.append(in_transit_delta_leaf)
                input_tags.append(("aux_in_transit", ""))
        else:
            if want_rho and pillar_deltas is not None and pv_t.requires_grad:
                grad_inputs.append(pillar_deltas)
                input_tags.append(("rho", disc_id))
            if want_rho and funding_pillar_deltas is not None and fin_curve_id is not None and pv_t.requires_grad:
                grad_inputs.append(funding_pillar_deltas)
                input_tags.append(("rho", fin_curve_id))
            if want_rho and pv_t.requires_grad:
                for cid_extra, c_tensor in extra_curve_vars.items():
                    grad_inputs.append(c_tensor)
                    input_tags.append(("rho", cid_extra))
                for ticker, r_tensor in repo_deltas.items():
                    grad_inputs.append(r_tensor)
                    input_tags.append(("repo_rho", ticker))
            if want_divs and pv_t.requires_grad:
                for ticker in div_grad_tickers:
                    grad_inputs.append(div_vars[ticker])
                    input_tags.append(("div", ticker))
            if want_vega and pv_t.requires_grad:
                for ticker in vol_grad_tickers:
                    grad_inputs.append(vol_deltas[ticker])
                    input_tags.append(("vega", ticker))
        if want_vega and pv_t.requires_grad:
            for ticker, fx_v in fx_vol_deltas.items():
                grad_inputs.append(fx_v)
                input_tags.append(("fx_vega", ticker))
        if want_cega and pv_t.requires_grad:
            for ticker, fx_c in fx_corr_deltas.items():
                grad_inputs.append(fx_c)
                input_tags.append(("fx_cega", ticker))

        if grad_inputs:
            grads = torch.autograd.grad(
                pv_sum_target,
                grad_inputs,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            for tag, g_val in zip(input_tags, grads):
                if g_val is not None:
                    grad_map[tag] = g_val

        if use_batched_gamma and pv_t.requires_grad:
            div_sched_b = state.dividend_schedule
            if want_vega:
                g_iv_0_all = grad_map.get(("aux_iv_0", ""))
                g_iv_q0_all = grad_map.get(("aux_iv_q0", ""))
                for i, ticker in enumerate(sorted_ids):
                    if ticker not in vol_grad_set:
                        continue
                    surf_i = state.vol_surfaces[ticker]
                    sc_pair = batch_aux["surf_caches_b"][ticker]
                    g_v: torch.Tensor | None = None
                    if g_iv_0_all is not None:
                        g_v = _surface_quote_vjp(
                            surf_i,
                            batch_aux["t_1_q"],
                            batch_aux["k_0_surf"][i],
                            g_iv_0_all[i],
                            cached_pair=sc_pair,
                        )
                    if g_iv_q0_all is not None and batch_aux["k_0_q"][i] is not None:
                        k_q_i = batch_aux["k_0_q"][i]
                        if k_q_i is not None:
                            g_vq = _surface_quote_vjp(
                                surf_i,
                                batch_aux["tau_obs_t"],
                                k_q_i,
                                g_iv_q0_all[i],
                                cached_pair=sc_pair,
                            )
                            g_v = g_vq if g_v is None else (g_v + g_vq)
                    if g_v is not None:
                        grad_map[("vega", ticker)] = g_v

            if want_divs:
                g_div_pv_all = grad_map.get(("aux_div_pv", ""))
                if g_div_pv_all is not None:
                    for i, ticker in enumerate(sorted_ids):
                        if ticker not in div_grad_set:
                            continue
                        w_t = div_sched_b.dividend_pv_weights[ticker].to(dtype=torch.float64, device=device)
                        inc_t = div_sched_b.dividend_obs_incidence[ticker].to(dtype=torch.float64, device=device)
                        grad_map[("div", ticker)] = torch.matmul(inc_t, g_div_pv_all[i].detach()) * w_t

            if want_rho:
                disc_c_b = batch_aux["disc_curve_obj"]
                as_of_b = batch_aux["as_of_orig"]
                asset_fin_c_b = batch_aux["asset_fin_curves"]
                fut_obs_dts = state.seasoned_state.future_obs_dates
                fut_pay_dts = state.seasoned_state.future_pay_dates
                f_leg_cid = (state.funding_leg or {}).get("funding_curve_id")

                rho_acc: dict[tuple[str, str], torch.Tensor] = {}

                def _acc_rho(tag: tuple[str, str], vec: torch.Tensor) -> None:
                    prev = rho_acc.get(tag)
                    rho_acc[tag] = vec if prev is None else (prev + vec)

                g_df_pay = grad_map.get(("aux_df_pay", ""))
                if g_df_pay is not None and pillar_deltas is not None:
                    jac_pay = _curve_df_delta_jacobian(
                        disc_c_b, fut_pay_dts, k_disc_pillars, device, as_of_date=as_of_b
                    )
                    _acc_rho(("rho", disc_id), torch.mv(jac_pay.T, g_df_pay.detach()))

                g_df_fin = grad_map.get(("aux_df_fin", ""))
                g_div_pv_r = grad_map.get(("aux_div_pv", ""))
                g_df_repo = grad_map.get(("aux_df_repo", ""))

                # Precompute tag mapping for each curve_tensor identity to avoid redundant curve work.
                fin_tag_map: dict[int, tuple[tuple[str, str], int]] = {}
                if pillar_deltas is not None:
                    fin_tag_map[id(pillar_deltas)] = (("rho", disc_id), k_disc_pillars)
                if funding_pillar_deltas is not None and fin_curve_id is not None:
                    fin_tag_map[id(funding_pillar_deltas)] = (("rho", fin_curve_id), k_fin_pillars)
                for cid_extra, c_tensor in extra_curve_vars.items():
                    fin_tag_map[id(c_tensor)] = (("rho", cid_extra), int(c_tensor.numel()))

                jac_obs_memo: dict[tuple[int, str], torch.Tensor] = {}
                for i, ticker in enumerate(sorted_ids):
                    c_i = asset_fin_c_b[i]
                    c_i_id = str(getattr(c_i, "identifier", getattr(c_i, "Identifier", disc_id)))
                    if ticker in asset_curve_deltas:
                        fin_delta_i: torch.Tensor | None = asset_curve_deltas[ticker]
                    elif c_i is disc_c_b or c_i_id == disc_id:
                        fin_delta_i = pillar_deltas
                    elif (
                        funding_pillar_deltas is not None
                        and f_leg_cid is not None
                        and c_i_id == str(f_leg_cid)
                    ):
                        fin_delta_i = funding_pillar_deltas
                    else:
                        fin_delta_i = None

                    fin_info = fin_tag_map.get(id(fin_delta_i)) if fin_delta_i is not None else None
                    if g_df_fin is not None and fin_info is not None:
                        tag_fin, n_pil_fin = fin_info
                        m_key = (id(c_i), tag_fin[1])
                        if m_key not in jac_obs_memo:
                            jac_obs_memo[m_key] = _curve_df_delta_jacobian(
                                c_i, fut_obs_dts, n_pil_fin, device, as_of_date=as_of_b
                            )
                        _acc_rho(tag_fin, torch.mv(jac_obs_memo[m_key].T, g_df_fin[i].detach()))

                    if g_df_repo is not None and ticker in repo_deltas and ticker in repo_curves_map:
                        r_curve_i = _normalize_curve_obj(repo_curves_map[ticker], as_of_date=as_of_b)
                        jac_r = _curve_df_delta_jacobian(
                            r_curve_i, fut_obs_dts, int(repo_deltas[ticker].numel()), device, as_of_date=as_of_b
                        )
                        _acc_rho(("repo_rho", ticker), torch.mv(jac_r.T, g_df_repo[i].detach()))

                    if g_div_pv_r is not None:
                        events_i = div_sched_b.dividend_events.get(ticker, [])
                        if len(events_i) > 0 and (fin_info is not None or ticker in repo_deltas):
                            amt_t = div_sched_b.dividend_amount_tensors[ticker].to(dtype=torch.float64, device=device)
                            w_t = div_sched_b.dividend_pv_weights[ticker].to(dtype=torch.float64, device=device)
                            inc_t = div_sched_b.dividend_obs_incidence[ticker].to(dtype=torch.float64, device=device)
                            active_ex = torch.tensor(
                                [1.0 if ev.ex_date > as_of_b else 0.0 for ev in events_i],
                                dtype=torch.float64,
                                device=device,
                            )
                            g_w_div = torch.matmul(inc_t, g_div_pv_r[i].detach()) * amt_t * w_t * active_ex
                            if fin_info is not None:
                                tag_fin, n_pil_fin = fin_info
                                jac_div_fin = _curve_df_delta_jacobian(
                                    c_i, [ev.payment_date for ev in events_i], n_pil_fin, device, as_of_date=as_of_b
                                )
                                _acc_rho(tag_fin, torch.mv(jac_div_fin.T, g_w_div))
                            if ticker in repo_deltas and ticker in repo_curves_map:
                                r_curve_i = _normalize_curve_obj(repo_curves_map[ticker], as_of_date=as_of_b)
                                jac_div_repo = _curve_df_delta_jacobian(
                                    r_curve_i,
                                    [ev.ex_date for ev in events_i],
                                    int(repo_deltas[ticker].numel()),
                                    device,
                                    as_of_date=as_of_b,
                                )
                                _acc_rho(("repo_rho", ticker), -torch.mv(jac_div_repo.T, g_w_div))

                g_w_fund = grad_map.get(("aux_w_fund", ""))
                g_in_tr = grad_map.get(("aux_in_transit", ""))
                need_autow_rho = (
                    (g_w_fund is not None and state.funding_leg is not None)
                    or (g_in_tr is not None and pillar_deltas is not None and bool(state.seasoned_state.in_transit_indices))
                )
                if need_autow_rho:
                    auto_inputs: list[torch.Tensor] = []
                    auto_tags: list[tuple[str, str]] = []
                    if pillar_deltas is not None:
                        auto_inputs.append(pillar_deltas)
                        auto_tags.append(("rho", disc_id))
                    if funding_pillar_deltas is not None and fin_curve_id is not None:
                        auto_inputs.append(funding_pillar_deltas)
                        auto_tags.append(("rho", fin_curve_id))
                    if auto_inputs:
                        with torch.enable_grad():  # type: ignore[no-untyped-call]
                            rho_loss = torch.zeros((), dtype=torch.float64, device=device)
                            if g_w_fund is not None and state.funding_leg is not None:
                                df_fin_ov_r: Callable[[Any], torch.Tensor | float] | None = None
                                df_fwd_ov_r: Callable[[Any], torch.Tensor | float] | None = None
                                if pillar_deltas is not None:
                                    df_fin_memo_r: dict[Any, torch.Tensor] = {}

                                    def _df_fin_ov_r(fin_dts: Any) -> torch.Tensor | float:
                                        is_sc = not isinstance(fin_dts, (list, tuple, np.ndarray))
                                        if is_sc and fin_dts in df_fin_memo_r:
                                            return df_fin_memo_r[fin_dts]
                                        dl = [fin_dts] if is_sc else list(fin_dts)
                                        b_dfs = torch.tensor(
                                            _curve_df_cached_np(disc_c_b, dl),
                                            dtype=torch.float64,
                                            device=device,
                                        )
                                        res_f = b_dfs * _curve_df_delta_multiplier(
                                            disc_c_b, dl, pillar_deltas, as_of_date=as_of_b
                                        )
                                        if is_sc:
                                            df_fin_memo_r[fin_dts] = res_f[0]
                                            return res_f[0]
                                        for d_it, v_it in zip(dl, res_f):
                                            df_fin_memo_r[d_it] = v_it
                                        return res_f

                                    df_fin_ov_r = _df_fin_ov_r
                                if funding_pillar_deltas is not None:
                                    f_curve_leg_r = _normalize_curve_obj(
                                        state.funding_leg.get("funding_curve"), as_of_date=as_of_b
                                    )
                                    if f_curve_leg_r is not None:
                                        def _df_fwd_ov_r(fwd_dts: Any) -> torch.Tensor:
                                            is_sc = not isinstance(fwd_dts, (list, tuple, np.ndarray))
                                            dl = [fwd_dts] if is_sc else list(fwd_dts)
                                            b_dfs = torch.tensor(
                                                _curve_df_cached_np(f_curve_leg_r, dl),
                                                dtype=torch.float64,
                                                device=device,
                                            )
                                            res_w = b_dfs * _curve_df_delta_multiplier(
                                                f_curve_leg_r, dl, funding_pillar_deltas, as_of_date=as_of_b
                                            )
                                            return res_w[0] if is_sc else res_w

                                        df_fwd_ov_r = _df_fwd_ov_r
                                if df_fin_ov_r is not None or df_fwd_ov_r is not None:
                                    w_fund_r = _compute_funding_weights_per_period(
                                        state, df_disc_fn=df_fin_ov_r, df_fwd_fn=df_fwd_ov_r, period_start_idx=0, device=device
                                    )
                                    rho_loss = rho_loss + (g_w_fund.detach() * w_fund_r).sum()

                            if g_in_tr is not None and pillar_deltas is not None and state.seasoned_state.in_transit_indices:
                                in_tr_r = _rolled_in_transit_pv(state, disc_c_b, as_of_b, pillar_deltas, device)
                                rho_loss = rho_loss + g_in_tr.detach() * in_tr_r

                            if rho_loss.requires_grad:
                                r_grads = torch.autograd.grad(
                                    rho_loss,
                                    auto_inputs,
                                    create_graph=False,
                                    retain_graph=False,
                                    allow_unused=True,
                                )
                                for tag, g_val in zip(auto_tags, r_grads):
                                    if g_val is not None:
                                        _acc_rho(tag, g_val)

                for tag, g_acc in rho_acc.items():
                    grad_map[tag] = g_acc

    inv_idx_t = torch.tensor(inv_perm, dtype=torch.long, device=device)
    spots_orig_t = state.spots_t0.detach().index_select(0, inv_idx_t)

    # 1. Delta (in original asset order pi_lex^{-1})
    delta_dict: dict[str, float] | None = None
    delta_tensor_orig: torch.Tensor | None = None
    grad_spots_sorted = grad_map.get(("spots", ""))
    if want_delta:
        if grad_spots_sorted is not None:
            delta_tensor_orig = grad_spots_sorted.detach().index_select(0, inv_idx_t).to(dtype=torch.float64)
        else:
            delta_tensor_orig = torch.zeros(d_assets, dtype=torch.float64, device=device)
        delta_list = delta_tensor_orig.detach().cpu().numpy().tolist()
        delta_dict = {orig_ids[i]: delta_list[i] for i in range(d_assets)}

    # 2. Gamma & 3. MultivariateAggregatedCashGamma via thread-safe batched AD Delta FD (zero torch.func)
    gamma_dict: dict[str, dict[str, float]] | None = None
    gamma_tensor_orig: torch.Tensor | None = None
    cash_gamma_agg: float | None = None

    if want_gamma or want_cash_gamma:
        if m_rem == 0 or state.already_autocalled:
            if want_gamma:
                gamma_tensor_orig = torch.zeros((d_assets, d_assets), dtype=torch.float64, device=device)
            if want_cash_gamma:
                cash_gamma_agg = 0.0
        else:
            g_gamma_batch = grad_map.get(("spots_gamma", ""))
            if use_batched_gamma:
                if g_gamma_batch is not None and h_gamma_vec is not None:
                    hess_cols = [
                        (g_gamma_batch[2 * gi] - g_gamma_batch[2 * gi + 1]) / (2.0 * h_gamma_vec[gi])
                        for gi in range(d_assets)
                    ]
                    hess_sorted = torch.stack(hess_cols, dim=0).detach()
                else:
                    hess_sorted = torch.zeros((d_assets, d_assets), dtype=torch.float64, device=device)
                hess_sym = 0.5 * (hess_sorted + hess_sorted.transpose(0, 1))
                gamma_orig_tmp = hess_sym.index_select(0, inv_idx_t).index_select(1, inv_idx_t)
                if want_gamma:
                    gamma_tensor_orig = gamma_orig_tmp
                if want_cash_gamma:
                    s_outer = spots_orig_t.unsqueeze(1) * spots_orig_t.unsqueeze(0)
                    cash_gamma_agg = float((s_outer * gamma_orig_tmp).sum().cpu().item())
            else:
                # Pure float64 / None surrogate path: exact 2nd-order PyTorch AD without torch.func.
                base_spots = state.spots_t0.detach().to(dtype=torch.float64, device=device)
                if want_gamma:
                    sp_rev = base_spots.clone().requires_grad_(True)
                    pv_rev = evaluate_differentiable_pv(
                        state,
                        spots_t0=sp_rev,
                        surrogate=eff_surrogate,
                        require_second_order_sdpa=True,
                        return_postproc=False,
                    )
                    if not isinstance(pv_rev, torch.Tensor):
                        raise RuntimeError("Expected scalar torch.Tensor from evaluate_differentiable_pv.")
                    g_rev = torch.autograd.grad(
                        pv_rev, sp_rev, create_graph=True, retain_graph=True, allow_unused=True
                    )[0]
                    if g_rev is not None and g_rev.requires_grad:
                        cols: list[torch.Tensor] = []
                        for col_i in range(d_assets):
                            row_i = torch.autograd.grad(
                                g_rev[col_i],
                                sp_rev,
                                retain_graph=(col_i < d_assets - 1),
                                allow_unused=True,
                            )[0]
                            cols.append(
                                row_i.detach()
                                if row_i is not None
                                else torch.zeros(d_assets, dtype=torch.float64, device=device)
                            )
                        hess_sorted = torch.stack(cols, dim=0)
                    else:
                        hess_sorted = torch.zeros((d_assets, d_assets), dtype=torch.float64, device=device)
                    hess_sym = 0.5 * (hess_sorted + hess_sorted.transpose(0, 1))
                    gamma_tensor_orig = hess_sym.index_select(0, inv_idx_t).index_select(1, inv_idx_t)
                    if want_cash_gamma:
                        s_outer = spots_orig_t.unsqueeze(1) * spots_orig_t.unsqueeze(0)
                        cash_gamma_agg = float((s_outer * gamma_tensor_orig).sum().cpu().item())
                elif want_cash_gamma:
                    def _pv_of_eps_exact(eps_scalar: torch.Tensor) -> torch.Tensor:
                        res_e = evaluate_differentiable_pv(
                            state,
                            spots_t0=base_spots * (1.0 + eps_scalar),
                            surrogate=eff_surrogate,
                            require_second_order_sdpa=True,
                            return_postproc=False,
                        )
                        if not isinstance(res_e, torch.Tensor):
                            raise RuntimeError("Expected scalar torch.Tensor from evaluate_differentiable_pv.")
                        return res_e

                    cash_gamma_agg = float(
                        compute_euler_cash_gamma(_pv_of_eps_exact, device=device).detach().cpu().item()
                    )

        if want_gamma and gamma_tensor_orig is not None:
            gamma_list = gamma_tensor_orig.detach().cpu().numpy().tolist()
            gamma_dict = {
                orig_ids[i]: dict(zip(orig_ids, gamma_list[i]))
                for i in range(d_assets)
            }

    # 4. Vega (in original asset order pi_lex^{-1})
    vega_dict: dict[str, list[list[float]]] | None = None
    vega_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    fx_vega_dict: dict[str, list[float]] | None = None
    if want_vega:
        vega_dict = {}
        vega_arrays = {}
        for ticker in orig_ids:
            surf = state.vol_surfaces.get(ticker)
            g_v = grad_map.get(("vega", ticker))
            if g_v is not None:
                arr_v = g_v.detach().cpu().numpy().astype(np.float64, copy=True)
            elif surf is not None:
                arr_v = np.zeros((surf.n_exp, surf.n_raw_strikes), dtype=np.float64)
            else:
                arr_v = np.zeros((0, 0), dtype=np.float64)
            vega_arrays[ticker] = arr_v
            vega_dict[ticker] = arr_v.tolist()
        if fx_vol_deltas:
            fx_vega_dict = {}
            for ticker in orig_ids:
                g_fxv = grad_map.get(("fx_vega", ticker))
                if g_fxv is not None:
                    fx_vega_dict[ticker] = g_fxv.detach().cpu().numpy().astype(np.float64, copy=False).tolist()

    # 5. Rho (discount curve + distinct funding curve + distinct asset financing curves + repo)
    rho_dict: dict[str, list[float]] | None = None
    rho_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    repo_rho_dict: dict[str, list[float]] | None = None
    if want_rho:
        rho_arrays = {}
        rho_dict = {}

        def _store_rho(cid: str, n_pil: int) -> None:
            g_r = grad_map.get(("rho", cid))
            if g_r is not None:
                arr_r = g_r.detach().cpu().numpy().astype(np.float64, copy=True)
            else:
                arr_r = np.zeros(n_pil, dtype=np.float64)
            rho_arrays[cid] = arr_r
            rho_dict[cid] = arr_r.tolist()

        _store_rho(disc_id, k_disc_pillars)
        if funding_pillar_deltas is not None and fin_curve_id is not None and fin_curve_id != disc_id:
            _store_rho(fin_curve_id, k_fin_pillars)
        for cid_extra, c_tensor in extra_curve_vars.items():
            if cid_extra not in rho_arrays:
                _store_rho(cid_extra, int(c_tensor.numel()))

        if repo_deltas:
            repo_rho_dict = {}
            for ticker in orig_ids:
                if ticker in repo_deltas:
                    g_rr = grad_map.get(("repo_rho", ticker))
                    if g_rr is not None:
                        arr_rr = g_rr.detach().cpu().numpy().astype(np.float64, copy=True)
                    else:
                        arr_rr = np.zeros(int(repo_deltas[ticker].numel()), dtype=np.float64)
                    repo_rho_dict[ticker] = arr_rr.tolist()

    # 6. DividendsSensitivities (in original asset order pi_lex^{-1})
    divs_dict: dict[str, list[float]] | None = None
    divs_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    if want_divs:
        divs_dict = {}
        divs_arrays = {}
        for ticker in orig_ids:
            g_d = grad_map.get(("div", ticker))
            if g_d is not None:
                arr_d = g_d.detach().cpu().numpy().astype(np.float64, copy=True)
            else:
                n_ev = len(state.dividend_schedule.dividend_events.get(ticker, []))
                arr_d = np.zeros(n_ev, dtype=np.float64)
            divs_arrays[ticker] = arr_d
            divs_dict[ticker] = arr_d.tolist()

    # 7. Cega (symmetrized off-diagonal dPV/d_rho_{ij} + dPV/d_rho_{ji} in original asset order)
    cega_dict: dict[str, dict[str, float]] | None = None
    cega_tensor_orig: torch.Tensor | None = None
    fx_cega_dict: dict[str, dict[str, float]] | None = None
    if want_cega:
        g_corr = grad_map.get(("corr", ""))
        if g_corr is not None and d_assets > 1:
            eye_d = torch.eye(d_assets, dtype=torch.float64, device=device)
            cega_sorted = (g_corr.detach() + g_corr.detach().transpose(0, 1)) * (1.0 - eye_d)
            cega_tensor_orig = cega_sorted.index_select(0, inv_idx_t).index_select(1, inv_idx_t)
        else:
            cega_tensor_orig = torch.zeros((d_assets, d_assets), dtype=torch.float64, device=device)
        cega_list = cega_tensor_orig.detach().cpu().numpy().tolist()
        cega_dict = {
            orig_ids[i]: dict(zip(orig_ids, cega_list[i]))
            for i in range(d_assets)
        }
        if fx_corr_deltas:
            fx_cega_dict = {}
            for ticker in orig_ids:
                g_fxc = grad_map.get(("fx_cega", ticker))
                if g_fxc is not None:
                    fx_id = str(quanto_meta.get(ticker, {}).get("fx_id", "FX"))
                    fx_cega_dict[ticker] = {fx_id: float(g_fxc.detach().cpu().item())}

    # 8. Theta (true 1-calendar-day roll PV(t_0 + 1d) - PV(t_0))
    theta_val: float | None = None
    if want_theta:
        if pv_theta_batched is not None:
            theta_val = float((pv_theta_batched - pv_t.detach()).cpu().item())
        else:
            with torch.no_grad():
                if abs(float(theta_day_fraction) - 1.0 / 365.0) < 1e-6 and isinstance(as_of_orig, dt.datetime):
                    as_of_rolled = as_of_orig + dt.timedelta(days=1)
                    pv_rolled = evaluate_differentiable_pv(
                        state,
                        surrogate=eff_surrogate,
                        as_of_date_override=as_of_rolled,
                        use_fast_cpu_baseline=True,
                        return_postproc=False,
                    )
                else:
                    pv_rolled = evaluate_differentiable_pv(
                        state,
                        surrogate=eff_surrogate,
                        tau_shift=float(theta_day_fraction),
                        use_fast_cpu_baseline=True,
                        return_postproc=False,
                    )
                if not isinstance(pv_rolled, torch.Tensor):
                    raise RuntimeError("Expected torch.Tensor from rolled Theta evaluation.")
                theta_val = float((pv_rolled - pv_t.detach()).cpu().item())

    if orig_device.type != "cpu":
        if delta_tensor_orig is not None:
            delta_tensor_orig = delta_tensor_orig.to(device=orig_device)
        if gamma_tensor_orig is not None:
            gamma_tensor_orig = gamma_tensor_orig.to(device=orig_device)
        if cega_tensor_orig is not None:
            cega_tensor_orig = cega_tensor_orig.to(device=orig_device)

    return AutocallGreeksResult(
        product_value=float(pv_t.detach().cpu().item()),
        cash_flows=list(post_out.cashflows),
        delta=delta_dict,
        gamma=gamma_dict,
        multivariate_aggregated_cash_gamma=cash_gamma_agg,
        vega=vega_dict,
        rho=rho_dict,
        dividends_sensitivities=divs_dict,
        theta=theta_val,
        cega=cega_dict,
        delta_tensor=delta_tensor_orig,
        gamma_tensor=gamma_tensor_orig,
        vega_arrays=vega_arrays,
        rho_arrays=rho_arrays,
        dividends_arrays=divs_arrays,
        cega_tensor=cega_tensor_orig,
        reporting_currency=post_out.reporting_currency,
        standard_error=post_out.standard_error,
        cash_to_settle_today=post_out.cash_to_settle_today,
        postproc_output=post_out,
        repo_rho=repo_rho_dict,
        fx_vega=fx_vega_dict,
        fx_cega=fx_cega_dict,
        greeks_engine_used="tier3_ad" if eff_surrogate is not None else "tier1_ad",
    )


def _batched_put_spread_loss_torch(
    f_w: torch.Tensor,
    v_w: torch.Tensor,
    k_strike: torch.Tensor,
    k_low: torch.Tensor,
    active_ki: torch.Tensor,
    b_ki: torch.Tensor,
    ko_spread_mag: torch.Tensor,
) -> torch.Tensor:
    """Evaluates vectorized terminal put-spread expected loss [K_low, K_strike] with optional KI."""
    k_eff_hi = torch.where(active_ki, torch.minimum(k_strike, b_ki), k_strike)
    put_hi = _black76_put_torch(f_w, k_eff_hi, v_w)
    put_lo = torch.where(k_low > 1e-8, _black76_put_torch(f_w, torch.minimum(k_low, k_eff_hi), v_w), torch.zeros_like(put_hi))
    put_spread_part = (put_hi - put_lo).clamp(min=0.0)

    ki_up = _digital_call_spread_prob_torch(f_w, v_w, b_ki, -ko_spread_mag)
    gap_eff = torch.clamp(k_strike - torch.maximum(b_ki, k_low), min=0.0)
    gap_part = torch.where(
        active_ki & (k_strike > k_low),
        gap_eff * (1.0 - ki_up).clamp(0.0, 1.0),
        torch.zeros_like(put_spread_part),
    )
    return torch.where(
        k_strike > k_low + 1e-8,
        put_spread_part + gap_part,
        torch.zeros_like(put_spread_part),
    )


_INV_SQRT2_GE: float = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI_GE: float = 1.0 / math.sqrt(2.0 * math.pi)
_TIER1_CG_LOCK = threading.Lock()
_TIER1_CG_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_TIER1_CG_SEEN: dict[tuple[Any, ...], int] = {}
_TIER1_CG_MAXSIZE = 8


def _ndtr_f32(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(-x * _INV_SQRT2_GE)


def _clark_f32_masked(
    f_in: torch.Tensor,
    v_in: torch.Tensor,
    c_in: torch.Tensor,
    am_in: torch.Tensor,
    *,
    is_best_of: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-host-sync masked Clark reduction in float32 for GPU batched Tier-1 VJPs."""
    n_rows, d_max = f_in.shape
    if d_max == 1:
        f_0 = f_in[:, 0].clamp(min=1e-12)
        v_0 = v_in[:, 0].clamp(min=1e-6)
        return f_0, v_0, am_in

    active_b = am_in > 0.5
    f_safe = torch.where(active_b, f_in.clamp(min=1e-12), torch.ones_like(f_in))
    v_safe = torch.where(active_b, v_in.clamp(min=1e-6), torch.full_like(v_in, 0.1))
    pair_active = active_b.unsqueeze(2) & active_b.unsqueeze(1)
    eye_d = torch.eye(d_max, dtype=f_in.dtype, device=f_in.device).unsqueeze(0)
    c_safe = torch.where(pair_active, c_in, eye_d)

    base_key = -f_safe.detach() if is_best_of else f_safe.detach()
    sort_key = torch.where(active_b, base_key, torch.full_like(f_safe, 1e12))
    ord_f = torch.argsort(sort_key, dim=1, stable=True)
    inv_o = torch.argsort(ord_f, dim=1)
    f_o = torch.gather(f_safe, 1, ord_f)
    v_o = torch.gather(v_safe, 1, ord_f)
    m_o = torch.gather(am_in, 1, ord_f)
    ord_row = ord_f.unsqueeze(2).expand(n_rows, d_max, d_max)
    ord_col = ord_f.unsqueeze(1).expand(n_rows, d_max, d_max)
    c_o = torch.gather(torch.gather(c_safe, 1, ord_row), 2, ord_col)

    f_cur = f_o[:, 0]
    v_cur = v_o[:, 0]
    rho_cur = c_o[:, 0, :]
    w_list: list[torch.Tensor] = [torch.ones(n_rows, dtype=f_in.dtype, device=f_in.device)]
    for _ in range(1, d_max):
        w_list.append(torch.zeros(n_rows, dtype=f_in.dtype, device=f_in.device))

    for j in range(1, d_max):
        active_j = m_o[:, j] > 0.5
        f_2 = f_o[:, j]
        v_2 = v_o[:, j]
        rho_12 = rho_cur[:, j].clamp(-0.99999, 0.99999)
        var_m = (v_cur * v_cur + v_2 * v_2 - 2.0 * rho_12 * v_cur * v_2).clamp(min=1e-12)
        sig_m = torch.sqrt(var_m)
        d1 = (torch.log(f_cur / f_2) + 0.5 * var_m) / sig_m
        d2 = d1 - sig_m
        if is_best_of:
            w1 = _ndtr_f32(d1)
            w2 = _ndtr_f32(-d2)
        else:
            w1 = _ndtr_f32(-d1)
            w2 = _ndtr_f32(d2)
        e_m1 = (f_cur * w1 + f_2 * w2).clamp(min=1e-12)
        f1_sq = (f_cur * f_cur) * torch.exp(v_cur * v_cur)
        f2_sq = (f_2 * f_2) * torch.exp(v_2 * v_2)
        sig_m2 = 2.0 * sig_m
        var_m2 = sig_m2 * sig_m2
        d1_sq = (torch.log(f1_sq.clamp(min=1e-24) / f2_sq.clamp(min=1e-24)) + 0.5 * var_m2) / sig_m2
        d2_sq = d1_sq - sig_m2
        if is_best_of:
            e_m2 = f1_sq * _ndtr_f32(d1_sq) + f2_sq * _ndtr_f32(-d2_sq)
        else:
            e_m2 = f1_sq * _ndtr_f32(-d1_sq) + f2_sq * _ndtr_f32(d2_sq)
        v_new = torch.sqrt(torch.clamp(torch.log(torch.clamp(e_m2 / (e_m1 * e_m1).clamp(min=1e-24), min=1.0)), min=1e-12))
        for prev_idx in range(j):
            w_list[prev_idx] = torch.where(active_j, w_list[prev_idx] * w1, w_list[prev_idx])
        w_list[j] = torch.where(active_j, w2, torch.zeros_like(w2))
        if j + 1 < d_max:
            v_rem = v_o[:, j + 1 :]
            rho_1_rem = rho_cur[:, j + 1 :]
            rho_2_rem = c_o[:, j, j + 1 :]
            f1_k = f_cur.unsqueeze(1) * torch.exp(rho_1_rem * v_cur.unsqueeze(1) * v_rem)
            f2_k = f_2.unsqueeze(1) * torch.exp(rho_2_rem * v_2.unsqueeze(1) * v_rem)
            var_m_col = var_m.unsqueeze(1)
            sig_m_col = sig_m.unsqueeze(1)
            d1_k = (torch.log(f1_k / f2_k) + 0.5 * var_m_col) / sig_m_col
            d2_k = d1_k - sig_m_col
            if is_best_of:
                e_m_pk = f1_k * _ndtr_f32(d1_k) + f2_k * _ndtr_f32(-d2_k)
            else:
                e_m_pk = f1_k * _ndtr_f32(-d1_k) + f2_k * _ndtr_f32(d2_k)
            cov_log = torch.log(torch.clamp(e_m_pk / e_m1.unsqueeze(1), min=1e-12))
            rho_new_rem = torch.clamp(cov_log / (v_new.unsqueeze(1) * v_rem).clamp(min=1e-12), -0.99999, 0.99999)
            updated_rem = torch.where(active_j.unsqueeze(1), rho_new_rem, rho_1_rem)
            rho_cur = torch.cat([rho_cur[:, : j + 1], updated_rem], dim=1)
        f_cur = torch.where(active_j, e_m1, f_cur)
        v_cur = torch.where(active_j, v_new, v_cur)
    w_o = torch.stack(w_list, dim=1) * m_o
    return f_cur, v_cur, torch.gather(w_o, 1, inv_o)


def _ko_and_ramp_merged_f32(
    fc: torch.Tensor,
    vc: torch.Tensor,
    b_ko: torch.Tensor,
    ko_sp_mag: torch.Tensor,
    *,
    need_ramp: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Evaluates KO call-spread probability, gradients, and survival-conditioned moments in one pass."""
    f_c = fc.clamp(min=1e-12)
    l_c = vc.clamp(min=1e-6)
    b_c = b_ko.clamp(min=1e-6)
    is_sp = ko_sp_mag > 1e-5
    b_lo = torch.minimum(b_c, b_c - ko_sp_mag).clamp(min=1e-6)
    b_hi = torch.maximum(b_c, b_c - ko_sp_mag).clamp(min=1e-6)
    db = (b_hi - b_lo).clamp(min=1e-7)
    inv_l = 1.0 / l_c
    ln_fc = torch.log(f_c)
    m_u = ln_fc - 0.5 * l_c * l_c
    u_hi = (torch.log(b_hi) - m_u) * inv_l
    u_lo = (torch.log(b_lo) - m_u) * inv_l
    phi_cdf_hi = _ndtr_f32(u_hi)
    phi_cdf_hi_m = _ndtr_f32(u_hi - l_c)
    phi_cdf_lo = _ndtr_f32(u_lo)
    phi_cdf_lo_m = _ndtr_f32(u_lo - l_c)
    phi_hi = torch.exp(-0.5 * u_hi * u_hi) * _INV_SQRT2PI_GE
    phi_lo = torch.exp(-0.5 * u_lo * u_lo) * _INV_SQRT2PI_GE
    dphi_m_tail = _ndtr_f32(-(u_lo - l_c)) - _ndtr_f32(-(u_hi - l_c))
    dphi_m = torch.where(u_hi + u_lo > 0.0, dphi_m_tail, phi_cdf_hi_m - phi_cdf_lo_m)

    m0_sp = (b_hi * phi_cdf_hi - b_lo * phi_cdf_lo - f_c * dphi_m) / db
    ko_sp_tail = (f_c * dphi_m_tail + b_hi * _ndtr_f32(-u_hi) - b_lo * _ndtr_f32(-u_lo)) / db
    ko_sp_raw = torch.where(u_hi + u_lo > 0.0, ko_sp_tail, 1.0 - m0_sp)
    in_01 = (ko_sp_raw > 0.0) & (ko_sp_raw < 1.0)
    dko_df_sp = torch.where(in_01, dphi_m / db, torch.zeros_like(f_c))
    dko_dv_sp = torch.where(in_01, (b_lo * phi_lo - b_hi * phi_hi) / db, torch.zeros_like(l_c))

    u_pt = (torch.log(b_c) - m_u) * inv_l
    ko_pt_raw = _ndtr_f32(-u_pt).clamp(0.0, 1.0)
    phi_pt = torch.exp(-0.5 * u_pt * u_pt) * _INV_SQRT2PI_GE
    dko_df_pt = phi_pt * inv_l / f_c
    dko_dv_pt = phi_pt * (-torch.log(f_c / b_c) * inv_l * inv_l - 0.5)

    ko_raw = torch.where(is_sp, ko_sp_raw.clamp(0.0, 1.0), ko_pt_raw)
    dko_df = torch.where(is_sp, dko_df_sp, dko_df_pt)
    dko_dv = torch.where(is_sp, dko_dv_sp, dko_dv_pt)

    if not need_ramp:
        return ko_raw, dko_df, dko_dv, None, None, None, None, None, None

    m1_sp = -(l_c * f_c / db) * dphi_m
    m2_sp = m0_sp + l_c * m1_sp + (l_c / db) * (b_hi * phi_hi - b_lo * phi_lo)
    m0_s = m0_sp.clamp(min=1e-20)
    inv_m0 = 1.0 / m0_s
    mu_sp = m1_sp * inv_m0
    var_sp = (m2_sp * inv_m0 - mu_sp * mu_sp).clamp(min=1e-8)

    dm0_df = -dphi_m / db
    dm0_dl = (b_hi * phi_hi - b_lo * phi_lo) / db
    phi_hi_m = (b_hi / f_c) * phi_hi
    phi_lo_m = (b_lo / f_c) * phi_lo
    dphi_m_df = -(phi_hi_m - phi_lo_m) * inv_l / f_c
    dm1_df = -(l_c / db) * (dphi_m + f_c * dphi_m_df)
    dphi_m_dl = (-u_hi * phi_hi_m + u_lo * phi_lo_m) * inv_l
    dm1_dl = -(f_c / db) * (dphi_m + l_c * dphi_m_dl)
    b_uphi = b_hi * u_hi * phi_hi - b_lo * u_lo * phi_lo
    dm2_df = dm0_df + l_c * dm1_df + (l_c / db) * (b_uphi * inv_l / f_c)
    dm2_dl = dm0_dl + m1_sp + l_c * dm1_dl + (1.0 / db) * (
        (b_hi * phi_hi - b_lo * phi_lo)
        - l_c * (b_hi * u_hi * phi_hi * (-u_hi * inv_l + 1.0) - b_lo * u_lo * phi_lo * (-u_lo * inv_l + 1.0))
    )
    dmu_df_sp = (dm1_df - mu_sp * dm0_df) * inv_m0
    dmu_dl_sp = (dm1_dl - mu_sp * dm0_dl) * inv_m0
    active_v = (m2_sp * inv_m0 - mu_sp * mu_sp) > 1e-8
    dvar_df_sp = torch.where(active_v, (dm2_df - (m2_sp * inv_m0) * dm0_df) * inv_m0 - 2.0 * mu_sp * dmu_df_sp, torch.zeros_like(f_c))
    dvar_dl_sp = torch.where(active_v, (dm2_dl - (m2_sp * inv_m0) * dm0_dl) * inv_m0 - 2.0 * mu_sp * dmu_dl_sp, torch.zeros_like(l_c))

    # Point-digital truncated upper normal fallback when |ko_sp_mag| <= 1e-5
    cdf_pt = _ndtr_f32(u_pt).clamp(min=1e-20)
    lam_pt = phi_pt / cdf_pt
    mu_pt = -lam_pt
    var_pt = (1.0 + u_pt * lam_pt - lam_pt * lam_pt).clamp(min=1e-8)
    dlam_du = -lam_pt * (u_pt + lam_pt)
    dmu_df_pt = -dlam_du * (-inv_l / f_c)
    dmu_dl_pt = -dlam_du * (-u_pt * inv_l + 1.0)
    dvar_du_pt = lam_pt + (u_pt - 2.0 * lam_pt) * dlam_du
    active_vpt = (1.0 + u_pt * lam_pt - lam_pt * lam_pt) > 1e-8
    dvar_df_pt = torch.where(active_vpt, dvar_du_pt * (-inv_l / f_c), torch.zeros_like(f_c))
    dvar_dl_pt = torch.where(active_vpt, dvar_du_pt * (-u_pt * inv_l + 1.0), torch.zeros_like(l_c))

    return (
        ko_raw,
        dko_df,
        dko_dv,
        torch.where(is_sp, mu_sp, mu_pt),
        torch.where(is_sp, var_sp, var_pt),
        torch.where(is_sp, dmu_df_sp, dmu_df_pt),
        torch.where(is_sp, dmu_dl_sp, dmu_dl_pt),
        torch.where(is_sp, dvar_df_sp, dvar_df_pt),
        torch.where(is_sp, dvar_dl_sp, dvar_dl_pt),
    )


def _exact_piecewise_terminal_put_tensor_f32(
    f: torch.Tensor,
    v: torch.Tensor,
    k_strike: torch.Tensor,
    k_low: torch.Tensor,
    active_ki: torch.Tensor,
    b_ki: torch.Tensor,
    ki_sp_mag: torch.Tensor,
    is_call_term: torch.Tensor,
    b_ko_term: torch.Tensor,
    ko_sp_mag: torch.Tensor,
) -> torch.Tensor:
    """Zero-branch tensorized piecewise terminal put in float32 matching exact Tier-1 integrals."""
    k_s = k_strike.clamp(min=0.0)
    k_l = k_low.clamp(min=0.0)
    has_put = (k_s > 1e-8) & (k_s > k_l)
    k_s_safe = k_s.clamp(min=1e-6)
    k_l_c = torch.minimum(k_l, k_s_safe)

    b_ki_c = b_ki.clamp(min=0.0)
    ki_sp_c = ki_sp_mag.clamp(min=0.0)
    b_ki_lo = torch.where(active_ki, b_ki_c.clamp(max=1e9), torch.zeros_like(k_s))
    b_ki_hi = torch.where(active_ki, b_ki_c + ki_sp_c, torch.zeros_like(k_s))
    is_ki_sp = active_ki & (ki_sp_c > 1e-6)

    has_ko = is_call_term > 0.5
    ko_sp_c = ko_sp_mag.clamp(min=0.0)
    b_ko_hi = torch.where(has_ko, b_ko_term.clamp(min=0.0), k_s_safe)
    b_ko_lo = torch.where(has_ko, (b_ko_hi - ko_sp_c).clamp(min=0.0), k_s_safe)
    is_ko_sp = has_ko & (ko_sp_c > 1e-6)

    raw_pts = torch.stack(
        [
            torch.zeros_like(k_s_safe),
            k_l_c,
            torch.clamp(b_ki_lo, min=0.0).minimum(k_s_safe),
            torch.clamp(b_ki_hi, min=0.0).minimum(k_s_safe),
            torch.clamp(b_ko_lo, min=0.0).minimum(k_s_safe),
            torch.clamp(b_ko_hi, min=0.0).minimum(k_s_safe),
            k_s_safe,
        ],
        dim=1,
    )
    pts, _ = torch.sort(raw_pts, dim=1)

    f_c = f.clamp(min=1e-12)
    v_c = v.clamp(min=1e-6)
    inv_v = 1.0 / v_c
    ln_f = torch.log(f_c)
    s2 = v_c * v_c
    exp_s2 = torch.exp(s2)
    exp_3s2 = torch.exp(3.0 * s2)

    total = torch.zeros_like(f_c)
    db_ki = (b_ki_hi - b_ki_lo).clamp(min=1e-7)
    db_ko = (b_ko_hi - b_ko_lo).clamp(min=1e-7)

    for k in range(6):
        a = pts[:, k]
        b = pts[:, k + 1]
        active_int = (b - a) > 1e-9
        mid = 0.5 * (a + b)

        below_low = mid <= k_l_c
        c0 = torch.where(below_low, k_s_safe - k_l_c, k_s_safe)
        c1 = torch.where(below_low, torch.zeros_like(k_s_safe), -torch.ones_like(k_s_safe))

        ki_dead = active_ki & (mid >= b_ki_hi)
        ki_ramp = is_ki_sp & (mid > b_ki_lo) & (mid < b_ki_hi)
        d0 = torch.where(ki_dead, torch.zeros_like(k_s_safe), torch.where(ki_ramp, b_ki_hi / db_ki, torch.ones_like(k_s_safe)))
        d1 = torch.where(ki_ramp, -1.0 / db_ki, torch.zeros_like(k_s_safe))

        ko_dead = has_ko & (mid >= b_ko_hi)
        ko_ramp = is_ko_sp & (mid > b_ko_lo) & (mid < b_ko_hi)
        e0 = torch.where(ko_dead, torch.zeros_like(k_s_safe), torch.where(ko_ramp, b_ko_hi / db_ko, torch.ones_like(k_s_safe)))
        e1 = torch.where(ko_ramp, -1.0 / db_ko, torch.zeros_like(k_s_safe))

        cd0 = c0 * d0
        cd1 = c0 * d1 + c1 * d0
        cd2 = c1 * d1
        q0 = cd0 * e0
        q1 = cd0 * e1 + cd1 * e0
        q2 = cd1 * e1 + cd2 * e0
        q3 = cd2 * e1

        has_a = a > 1e-12
        b_s = b.clamp(min=1e-12)
        a_s = a.clamp(min=1e-12)
        ub = (torch.log(b_s) - ln_f + 0.5 * s2) * inv_v
        ua = (torch.log(a_s) - ln_f + 0.5 * s2) * inv_v

        m0 = torch.where(has_a, _ndtr_f32(ub) - _ndtr_f32(ua), _ndtr_f32(ub))
        m1 = f_c * torch.where(has_a, _ndtr_f32(ub - v_c) - _ndtr_f32(ua - v_c), _ndtr_f32(ub - v_c))
        m2 = (f_c * f_c) * exp_s2 * torch.where(has_a, _ndtr_f32(ub - 2.0 * v_c) - _ndtr_f32(ua - 2.0 * v_c), _ndtr_f32(ub - 2.0 * v_c))
        m3 = (f_c * f_c * f_c) * exp_3s2 * torch.where(has_a, _ndtr_f32(ub - 3.0 * v_c) - _ndtr_f32(ua - 3.0 * v_c), _ndtr_f32(ub - 3.0 * v_c))
        contrib = q0 * m0 + q1 * m1 + q2 * m2 + q3 * m3
        total = total + torch.where(active_int, contrib, torch.zeros_like(contrib))

    return torch.where(has_put, total, torch.zeros_like(total))


def _markov_merged_and_adjoint_f32(
    f_w: torch.Tensor,
    v_w: torch.Tensor,
    r_next_all: torch.Tensor | None,
    sm: torch.Tensor,
    b_ko: torch.Tensor,
    b_cpn: torch.Tensor,
    is_c: torch.Tensor,
    is_g: torch.Tensor,
    ko_sp_mag: torch.Tensor,
    cpn_sp_mag: torch.Tensor,
    ki_sp_mag: torch.Tensor,
    k_strike: torch.Tensor,
    k_low: torch.Tensor,
    active_ki: torch.Tensor,
    b_ki: torch.Tensor,
    force_cpn_vec: torch.Tensor,
    g_p: torch.Tensor,
    g_q: torch.Tensor,
    g_lp: torch.Tensor,
    *,
    all_sched_act: bool,
    idx_last: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Forward + analytical reverse-mode adjoint of the 1D Markov KO/coupon/terminal-put chain."""
    b_sz, m_act = f_w.shape
    device = f_w.device
    z_mean = torch.zeros(b_sz, dtype=torch.float32, device=device)
    z_std = torch.ones(b_sz, dtype=torch.float32, device=device)
    surv = torch.ones(b_sz, dtype=torch.float32, device=device)

    cache: list[tuple[Any, ...]] = []
    fc_list: list[torch.Tensor] = []
    vc_list: list[torch.Tensor] = []
    h_list: list[torch.Tensor] = []
    dh_df_list: list[torch.Tensor] = []
    dh_dv_list: list[torch.Tensor] = []
    p_list: list[torch.Tensor] = []
    surv_prev_list: list[torch.Tensor] = []
    surv_list: list[torch.Tensor] = []

    for m in range(m_act):
        fw_m = f_w[:, m]
        vw_m = v_w[:, m]
        zm_in, zs_in, surv_in = z_mean, z_std, surv
        if m == 0:
            fc, vc = fw_m, vw_m
        else:
            fc = fw_m * torch.exp(vw_m * zm_in - 0.5 * vw_m * vw_m * (1.0 - zs_in * zs_in))
            vc = (vw_m * zs_in).clamp(min=1e-6)
        fc_list.append(fc)
        vc_list.append(vc)
        is_c_m = is_c[:, m] * sm[:, m]
        ko_raw, dko_df, dko_dv, mu_u, var_u, dmu_df, dmu_dv, dvar_df, dvar_dv = _ko_and_ramp_merged_f32(
            fc, vc, b_ko[:, m], ko_sp_mag, need_ramp=(m < m_act - 1)
        )
        clamped_hi = (ko_raw >= 0.0) & (ko_raw <= (1.0 - 1e-7))
        h_m = is_c_m * ko_raw.clamp(0.0, 1.0 - 1e-7)
        dh_df = torch.where(clamped_hi, is_c_m * dko_df, torch.zeros_like(dko_df))
        dh_dv = torch.where(clamped_hi, is_c_m * dko_dv, torch.zeros_like(dko_dv))
        h_list.append(h_m)
        dh_df_list.append(dh_df)
        dh_dv_list.append(dh_dv)
        p_list.append(surv_in * h_m)
        surv_prev_list.append(surv_in)
        surv_next = surv_in * (1.0 - h_m)
        surv_list.append(surv_next)

        if m < m_act - 1:
            assert r_next_all is not None
            assert mu_u is not None and var_u is not None
            c_on = is_c_m > 0.5
            z_pm = torch.where(c_on, zm_in + zs_in * mu_u, zm_in)
            z_pv = torch.where(c_on, (zs_in * zs_in * var_u).clamp(min=1e-8), zs_in * zs_in)
            rn = r_next_all[:, m]
            z_mean = rn * z_pm
            z_std_sq = (rn * rn * z_pv + (1.0 - rn * rn)).clamp(min=1e-8)
            z_std = torch.sqrt(z_std_sq)
            cache.append((zm_in, zs_in, surv_in, fc, vc, h_m, dh_df, dh_dv, mu_u, var_u, dmu_df, dmu_dv, dvar_df, dvar_dv, z_pm, z_pv, z_std, c_on))
        else:
            cache.append((zm_in, zs_in, surv_in, fc, vc, h_m, dh_df, dh_dv, None, None, None, None, None, None, None, None, None, None))
        surv = surv_next

    fc_all = torch.stack(fc_list, dim=1)
    vc_all = torch.stack(vc_list, dim=1)
    h_all = torch.stack(h_list, dim=1)
    dh_df_all = torch.stack(dh_df_list, dim=1)
    dh_dv_all = torch.stack(dh_dv_list, dim=1)

    force_2d = (force_cpn_vec > 0.5).unsqueeze(1) & (is_c * sm > 0.5)
    eff_b_cpn = torch.where(force_2d, torch.minimum(b_cpn, b_ko), b_cpn).clamp(min=1e-6)
    cpn_sp_v = cpn_sp_mag.unsqueeze(1)
    is_q_sp = cpn_sp_v > 1e-5
    bc_lo = torch.minimum(eff_b_cpn, eff_b_cpn - cpn_sp_v).clamp(min=1e-6)
    bc_hi = torch.maximum(eff_b_cpn, eff_b_cpn - cpn_sp_v).clamp(min=1e-6)
    dbc = (bc_hi - bc_lo).clamp(min=1e-7)
    inv_vc = 1.0 / vc_all.clamp(min=1e-6)
    ln_fc_all = torch.log(fc_all.clamp(min=1e-12))
    uc_hi = (torch.log(bc_hi) - ln_fc_all + 0.5 * vc_all * vc_all) * inv_vc
    uc_lo = (torch.log(bc_lo) - ln_fc_all + 0.5 * vc_all * vc_all) * inv_vc
    phic_cdf_hi = _ndtr_f32(uc_hi)
    phic_cdf_hi_m = _ndtr_f32(uc_hi - vc_all)
    phic_cdf_lo = _ndtr_f32(uc_lo)
    phic_cdf_lo_m = _ndtr_f32(uc_lo - vc_all)
    dphic_m_tail = _ndtr_f32(-(uc_lo - vc_all)) - _ndtr_f32(-(uc_hi - vc_all))
    dphic_m = torch.where(uc_hi + uc_lo > 0.0, dphic_m_tail, phic_cdf_hi_m - phic_cdf_lo_m)
    q_sp_tail = (fc_all * dphic_m_tail + bc_hi * _ndtr_f32(-uc_hi) - bc_lo * _ndtr_f32(-uc_lo)) / dbc
    q_sp_main = 1.0 - (bc_hi * phic_cdf_hi - bc_lo * phic_cdf_lo - fc_all * dphic_m) / dbc
    q_sp_raw = torch.where(uc_hi + uc_lo > 0.0, q_sp_tail, q_sp_main)
    in_q01 = (q_sp_raw > 0.0) & (q_sp_raw < 1.0)
    dq_df_sp = torch.where(in_q01, dphic_m / dbc, torch.zeros_like(fc_all))
    phic_hi = torch.exp(-0.5 * uc_hi * uc_hi) * _INV_SQRT2PI_GE
    phic_lo = torch.exp(-0.5 * uc_lo * uc_lo) * _INV_SQRT2PI_GE
    dq_dv_sp = torch.where(in_q01, (bc_lo * phic_lo - bc_hi * phic_hi) / dbc, torch.zeros_like(vc_all))

    uc_pt = (torch.log(eff_b_cpn) - ln_fc_all + 0.5 * vc_all * vc_all) * inv_vc
    q_pt_raw = _ndtr_f32(-uc_pt).clamp(0.0, 1.0)
    phic_pt = torch.exp(-0.5 * uc_pt * uc_pt) * _INV_SQRT2PI_GE
    dq_df_pt = phic_pt * inv_vc / fc_all.clamp(min=1e-12)
    dq_dv_pt = phic_pt * (-torch.log(fc_all.clamp(min=1e-12) / eff_b_cpn) * inv_vc * inv_vc - 0.5)

    q_base = torch.where(is_q_sp, q_sp_raw.clamp(0.0, 1.0), q_pt_raw)
    dq_df_base = torch.where(is_q_sp, dq_df_sp, dq_df_pt)
    dq_dv_base = torch.where(is_q_sp, dq_dv_sp, dq_dv_pt)

    use_forced_h = force_2d & (h_all > q_base)
    q_raw_eff = torch.where(use_forced_h, h_all, q_base)
    dq_df_eff = torch.where(use_forced_h, dh_df_all, dq_df_base)
    dq_dv_eff = torch.where(use_forced_h, dh_dv_all, dq_dv_base)

    not_guar = (is_g <= 0.5) & (sm > 0.5)
    q_px = torch.where(is_g > 0.5, torch.ones_like(q_raw_eff), q_raw_eff) * sm
    dq_df_all = torch.where(not_guar, dq_df_eff, torch.zeros_like(dq_df_eff))
    dq_dv_all = torch.where(not_guar, dq_dv_eff, torch.zeros_like(dq_dv_eff))

    if all_sched_act:
        fc_term = fc_list[-1]
        vc_term = vc_list[-1]
        surv_term = surv_list[-1]
        surv_prev_term = surv_prev_list[-1]
        is_call_term = is_c[:, -1]
        b_ko_term = b_ko[:, -1]
    else:
        assert idx_last is not None
        idx_1 = idx_last.unsqueeze(1)
        fc_term = fc_all.gather(1, idx_1).squeeze(1)
        vc_term = vc_all.gather(1, idx_1).squeeze(1)
        surv_term = torch.stack(surv_list, dim=1).gather(1, idx_1).squeeze(1)
        surv_prev_term = torch.stack(surv_prev_list, dim=1).gather(1, idx_1).squeeze(1)
        is_call_term = is_c.gather(1, idx_1).squeeze(1)
        b_ko_term = b_ko.gather(1, idx_1).squeeze(1)

    with torch.enable_grad():  # type: ignore[no-untyped-call]
        fc_t = fc_term.detach().requires_grad_(True)
        vc_t = vc_term.detach().requires_grad_(True)
        use_prev = is_call_term > 0.5
        sm_t = torch.where(use_prev, surv_prev_term, surv_term).detach().requires_grad_(True)
        put_val = _exact_piecewise_terminal_put_tensor_f32(
            fc_t, vc_t, k_strike, k_low, active_ki, b_ki, ki_sp_mag, is_call_term, b_ko_term, ko_sp_mag
        )
        l_put = sm_t * put_val
        a_fc_term, a_vc_term, a_sm_term = torch.autograd.grad((g_lp * l_put).sum(), (fc_t, vc_t, sm_t))

    a_surv_t = torch.where(use_prev, torch.zeros_like(a_sm_term), a_sm_term)
    a_surv_p = torch.where(use_prev, a_sm_term, torch.zeros_like(a_sm_term))

    a_surv = torch.zeros(b_sz, dtype=torch.float32, device=device)
    a_zm = torch.zeros(b_sz, dtype=torch.float32, device=device)
    a_zs = torch.zeros(b_sz, dtype=torch.float32, device=device)
    g_fw_list: list[torch.Tensor | None] = [None] * m_act
    g_vw_list: list[torch.Tensor | None] = [None] * m_act
    g_rn_list: list[torch.Tensor | None] = [None] * max(m_act - 1, 0)

    for m in range(m_act - 1, -1, -1):
        zm_in, zs_in, surv_in, fc, vc, h_m, dh_df, dh_dv, mu_u, var_u, dmu_df, dmu_dv, dvar_df, dvar_dv, z_pm, z_pv, z_std_out, c_on = cache[m]
        if all_sched_act:
            if m == m_act - 1:
                a_fc = a_fc_term
                a_vc = a_vc_term
                a_surv = a_surv + a_surv_t
            else:
                a_fc = torch.zeros(b_sz, dtype=torch.float32, device=device)
                a_vc = torch.zeros(b_sz, dtype=torch.float32, device=device)
        else:
            assert idx_last is not None
            is_last_m = idx_last == m
            a_fc = torch.where(is_last_m, a_fc_term, torch.zeros_like(a_fc_term))
            a_vc = torch.where(is_last_m, a_vc_term, torch.zeros_like(a_vc_term))
            a_surv = torch.where(is_last_m, a_surv + a_surv_t, a_surv)

        if m < m_act - 1:
            assert r_next_all is not None
            rn = r_next_all[:, m]
            a_z_std_sq = a_zs / (2.0 * z_std_out.clamp(min=1e-8))
            g_rn_list[m] = a_zm * z_pm + a_z_std_sq * (2.0 * rn * (z_pv - 1.0))
            a_z_pm = a_zm * rn
            a_z_pv = a_z_std_sq * (rn * rn)
            a_z_pv = torch.where(c_on & (zs_in * zs_in * var_u <= 1e-8), torch.zeros_like(a_z_pv), a_z_pv)
            a_mu = torch.where(c_on, a_z_pm * zs_in, torch.zeros_like(a_z_pm))
            a_var = torch.where(c_on, a_z_pv * (zs_in * zs_in), torch.zeros_like(a_z_pv))
            a_zm_next = a_z_pm
            a_zs_next = torch.where(c_on, a_z_pm * mu_u + a_z_pv * 2.0 * zs_in * var_u, a_z_pv * 2.0 * zs_in)
            a_fc = a_fc + a_mu * dmu_df + a_var * dvar_df
            a_vc = a_vc + a_mu * dmu_dv + a_var * dvar_dv
        else:
            a_zm_next = torch.zeros(b_sz, dtype=torch.float32, device=device)
            a_zs_next = torch.zeros(b_sz, dtype=torch.float32, device=device)

        a_h = g_p[:, m] * surv_in - a_surv * surv_in
        a_surv_in = g_p[:, m] * h_m + a_surv * (1.0 - h_m)
        if all_sched_act:
            if m == m_act - 1:
                a_surv_in = a_surv_in + a_surv_p
        else:
            assert idx_last is not None
            a_surv_in = torch.where(idx_last == m, a_surv_in + a_surv_p, a_surv_in)

        a_fc = a_fc + a_h * dh_df + g_q[:, m] * dq_df_all[:, m]
        a_vc = a_vc + a_h * dh_dv + g_q[:, m] * dq_dv_all[:, m]
        if m == 0:
            g_fw_list[m] = a_fc
            g_vw_list[m] = a_vc
        else:
            vw_m = v_w[:, m]
            g_fw_list[m] = a_fc * (fc / f_w[:, m])
            dfc_dvw = fc * (zm_in - vw_m * (1.0 - zs_in * zs_in))
            g_vw_list[m] = a_fc * dfc_dvw + a_vc * zs_in
            a_zm = a_zm_next + a_fc * fc * vw_m
            a_zs = a_zs_next + a_fc * fc * (vw_m * vw_m * zs_in) + a_vc * vw_m
        a_surv = a_surv_in

    g_fw_t = torch.stack([t for t in g_fw_list if t is not None], dim=1)
    g_vw_t = torch.stack([t for t in g_vw_list if t is not None], dim=1)
    g_rn_t = torch.stack([t for t in g_rn_list if t is not None], dim=1) if m_act > 1 else None
    return torch.stack(p_list, dim=1), q_px, l_put.detach(), g_fw_t, g_vw_t, g_rn_t


def _eval_tier1_vjp_f32_core(
    inputs: tuple[torch.Tensor, ...],
    *,
    is_best_of: bool,
    all_sched_act: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        fp_in, v_in, corr_in, am_in, sm_in, b_ko_in, b_cpn_in, is_c_in, is_g_in,
        ko_sp_in, cpn_sp_in, ki_sp_in, k_strike_in, k_low_in, active_ki_in, b_ki_in,
        force_cpn_in, idx_last_in,
        w_lfw_in, w_vw_in, w_p_in, w_q_in, w_lp_in, w_wt_in,
    ) = inputs
    b_sz, d_act, m_act = fp_in.shape
    fp_req = fp_in.detach().requires_grad_(True)
    vp_req = v_in.detach().requires_grad_(True)
    corr_req = corr_in.detach().requires_grad_(True)

    with torch.enable_grad():  # type: ignore[no-untyped-call]
        fp_flat = fp_req.transpose(1, 2).reshape(b_sz * m_act, d_act)
        vp_flat = vp_req.transpose(1, 2).reshape(b_sz * m_act, d_act)
        c_flat = corr_req.unsqueeze(1).expand(b_sz, m_act, d_act, d_act).reshape(b_sz * m_act, d_act, d_act)
        am_flat = am_in.unsqueeze(1).expand(b_sz, m_act, d_act).reshape(b_sz * m_act, d_act)
        fw_flat, vw_flat, w_flat = _clark_f32_masked(fp_flat, vp_flat, c_flat, am_flat, is_best_of=is_best_of)
        f_w = fw_flat.view(b_sz, m_act).clamp(min=1e-12)
        v_w = vw_flat.view(b_sz, m_act).clamp(min=1e-6)
        w_all = w_flat.view(b_sz, m_act, d_act) * am_in.unsqueeze(1)
        if m_act > 1:
            w_cur = w_all[:, :-1, :]
            w_nxt = w_all[:, 1:, :]
            v_cur = vp_req.transpose(1, 2)[:, :-1, :].clamp(min=1e-6)
            v_nxt = vp_req.transpose(1, 2)[:, 1:, :].clamp(min=1e-6)
            v_min = torch.minimum(v_cur.unsqueeze(-2), v_nxt.unsqueeze(-2))
            c_exp = corr_req.unsqueeze(1)
            num = (w_cur.unsqueeze(-1) * w_nxt.unsqueeze(-2) * c_exp * v_cur.unsqueeze(-1) * v_min).sum(dim=(-2, -1))
            cv_cur = w_cur * v_cur
            cv_nxt = w_nxt * v_nxt
            den_cur = (cv_cur.unsqueeze(-1) * cv_cur.unsqueeze(-2) * c_exp).sum(dim=(-2, -1)).clamp(min=1e-24)
            den_nxt = (cv_nxt.unsqueeze(-1) * cv_nxt.unsqueeze(-2) * c_exp).sum(dim=(-2, -1)).clamp(min=1e-24)
            r_next_all: torch.Tensor | None = (num / (torch.sqrt(den_cur) * torch.sqrt(den_nxt)).clamp(min=1e-24)).clamp(-0.9999, 0.9999)
        else:
            r_next_all = None

    idx_last_arg = None if all_sched_act else idx_last_in
    with torch.no_grad():
        _, _, _, g_fw_m, g_vw_m, g_rn_m = _markov_merged_and_adjoint_f32(
            f_w.detach(),
            v_w.detach(),
            r_next_all.detach() if r_next_all is not None else None,
            sm_in,
            b_ko_in,
            b_cpn_in,
            is_c_in,
            is_g_in,
            ko_sp_in,
            cpn_sp_in,
            ki_sp_in,
            k_strike_in,
            k_low_in,
            active_ki_in,
            b_ki_in,
            force_cpn_in,
            w_p_in,
            w_q_in,
            w_lp_in,
            all_sched_act=all_sched_act,
            idx_last=idx_last_arg,
        )
        g_fw_tot = g_fw_m + w_lfw_in / f_w.detach()
        g_vw_tot = g_vw_m + w_vw_in

    with torch.enable_grad():  # type: ignore[no-untyped-call]
        if all_sched_act:
            w_term_node = w_all[:, -1, :]
        else:
            assert idx_last_arg is not None
            w_term_node = w_all.gather(1, idx_last_arg.view(b_sz, 1, 1).expand(b_sz, 1, d_act)).squeeze(1)

        out_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
            (f_w, g_fw_tot),
            (v_w, g_vw_tot),
        ]
        if r_next_all is not None and g_rn_m is not None:
            out_pairs.append((r_next_all, g_rn_m))
        out_pairs.append((w_term_node, w_wt_in))
        active_pairs = [(o, g) for o, g in out_pairs if o.requires_grad]
        if active_pairs:
            g_fp, g_vp, g_c = torch.autograd.grad(
                [o for o, _ in active_pairs],
                (fp_req, vp_req, corr_req),
                grad_outputs=[g for _, g in active_pairs],
                allow_unused=True,
            )
        else:
            g_fp, g_vp, g_c = None, None, None
    return (
        g_fp.detach() if g_fp is not None else torch.zeros_like(fp_in),
        g_vp.detach() if g_vp is not None else torch.zeros_like(v_in),
        g_c.detach() if g_c is not None else torch.zeros_like(corr_in),
    )


if _HAS_TRITON_TIER1:

    @triton.jit
    def _triton_normcdf64(x):  # type: ignore[no-untyped-def]
        return _triton_libdevice.normcdf(x)

    @triton.jit
    def _triton_normpdf64(x):  # type: ignore[no-untyped-def]
        return 0.39894228040143267794 * tl.exp(-0.5 * x * x)

    @triton.jit  # type: ignore[untyped-decorator]
    def _markov_fwd_kernel64(  # type: ignore[no-untyped-def]
        fw_ptr, vw_ptr, rn_ptr, sm_ptr, bko_ptr, bcpn_ptr, isc_ptr, isg_ptr,
        kosp_ptr, cpnsp_ptr, kisp_ptr, kstr_ptr, klow_ptr, actki_ptr, bki_ptr,
        fcpn_ptr, idxlast_ptr,
        pout_ptr, qout_ptr, lputout_ptr,
        scratch_ptr, term_ptr,
        B, M: tl.constexpr, ALL_SCHED_ACT: tl.constexpr, SAVE_SCRATCH: tl.constexpr, BLOCK_B: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b_offs < B

        ko_sp_mag = tl.load(kosp_ptr + b_offs, mask=b_mask, other=0.0)
        cpn_sp_mag = tl.load(cpnsp_ptr + b_offs, mask=b_mask, other=0.0)
        force_cpn = tl.load(fcpn_ptr + b_offs, mask=b_mask, other=0.0)
        if ALL_SCHED_ACT:
            idx_last = tl.full([BLOCK_B], M - 1, dtype=tl.int32)
        else:
            idx_last = tl.load(idxlast_ptr + b_offs, mask=b_mask, other=M - 1).to(tl.int32)

        z_mean = tl.zeros([BLOCK_B], dtype=tl.float64)
        z_std = tl.full([BLOCK_B], 1.0, dtype=tl.float64)
        surv = tl.full([BLOCK_B], 1.0, dtype=tl.float64)

        fc_term = tl.zeros([BLOCK_B], dtype=tl.float64)
        vc_term = tl.zeros([BLOCK_B], dtype=tl.float64)
        surv_term = tl.zeros([BLOCK_B], dtype=tl.float64)
        surv_prev_term = tl.zeros([BLOCK_B], dtype=tl.float64)
        is_call_term = tl.zeros([BLOCK_B], dtype=tl.float64)
        b_ko_term = tl.zeros([BLOCK_B], dtype=tl.float64)

        is_ko_sp = ko_sp_mag > 1e-5
        is_q_sp = cpn_sp_mag > 1e-5
        MB = M * B

        for m in range(M):
            bm_offs = b_offs * M + m
            fw_m = tl.maximum(tl.load(fw_ptr + bm_offs, mask=b_mask, other=1.0), 1e-12)
            vw_m = tl.maximum(tl.load(vw_ptr + bm_offs, mask=b_mask, other=0.1), 1e-6)
            sm_m = tl.load(sm_ptr + bm_offs, mask=b_mask, other=0.0)
            b_ko_m = tl.load(bko_ptr + bm_offs, mask=b_mask, other=1.0)
            b_cpn_m = tl.load(bcpn_ptr + bm_offs, mask=b_mask, other=1.0)
            is_c_raw = tl.load(isc_ptr + bm_offs, mask=b_mask, other=0.0)
            is_g_raw = tl.load(isg_ptr + bm_offs, mask=b_mask, other=0.0)

            zm_in = z_mean
            zs_in = z_std
            surv_in = surv
            if m == 0:
                fc = fw_m
                vc = vw_m
            else:
                fc = tl.maximum(fw_m * tl.exp(vw_m * zm_in - 0.5 * vw_m * vw_m * (1.0 - zs_in * zs_in)), 1e-12)
                vc = tl.maximum(vw_m * zs_in, 1e-6)

            is_c_m = is_c_raw * sm_m

            b_hi = tl.maximum(b_ko_m, 1e-6)
            b_lo = tl.maximum(b_hi - ko_sp_mag, 1e-6)
            db = tl.maximum(b_hi - b_lo, 1e-7)
            inv_l = 1.0 / vc
            ln_fc = tl.log(fc)
            m_u = ln_fc - 0.5 * vc * vc
            u_hi = (tl.log(b_hi) - m_u) * inv_l
            u_lo = (tl.log(b_lo) - m_u) * inv_l
            phi_cdf_hi = _triton_normcdf64(u_hi)
            phi_cdf_hi_m = _triton_normcdf64(u_hi - vc)
            phi_cdf_lo = _triton_normcdf64(u_lo)
            phi_cdf_lo_m = _triton_normcdf64(u_lo - vc)
            phi_hi = _triton_normpdf64(u_hi)
            phi_lo = _triton_normpdf64(u_lo)
            dphi_m_tail = _triton_normcdf64(-(u_lo - vc)) - _triton_normcdf64(-(u_hi - vc))
            dphi_m = tl.where(u_hi + u_lo > 0.0, dphi_m_tail, phi_cdf_hi_m - phi_cdf_lo_m)
            m0_sp = (b_hi * phi_cdf_hi - b_lo * phi_cdf_lo - fc * dphi_m) / db
            ko_sp_tail = (fc * dphi_m_tail + b_hi * _triton_normcdf64(-u_hi) - b_lo * _triton_normcdf64(-u_lo)) / db
            ko_sp_raw = tl.where(u_hi + u_lo > 0.0, ko_sp_tail, 1.0 - m0_sp)
            in_01 = (ko_sp_raw > 0.0) & (ko_sp_raw < 1.0)
            dko_df_sp = tl.where(in_01, dphi_m / db, 0.0)
            dko_dv_sp = tl.where(in_01, (b_lo * phi_lo - b_hi * phi_hi) / db, 0.0)

            u_pt = (tl.log(b_hi) - m_u) * inv_l
            ko_pt_raw = tl.minimum(tl.maximum(_triton_normcdf64(-u_pt), 0.0), 1.0)
            phi_pt = _triton_normpdf64(u_pt)
            dko_df_pt = phi_pt * inv_l / fc
            dko_dv_pt = phi_pt * (-tl.log(fc / b_hi) * inv_l * inv_l - 0.5)

            ko_raw = tl.where(is_ko_sp, tl.minimum(tl.maximum(ko_sp_raw, 0.0), 1.0), ko_pt_raw)
            dko_df_r = tl.where(is_ko_sp, dko_df_sp, dko_df_pt)
            dko_dv_r = tl.where(is_ko_sp, dko_dv_sp, dko_dv_pt)

            clamped_hi = (ko_raw >= 0.0) & (ko_raw <= (1.0 - 1e-7))
            h_m = is_c_m * tl.minimum(tl.maximum(ko_raw, 0.0), 1.0 - 1e-7)
            dh_df = tl.where(clamped_hi, is_c_m * dko_df_r, 0.0)
            dh_dv = tl.where(clamped_hi, is_c_m * dko_dv_r, 0.0)
            p_m = surv_in * h_m
            surv_next = surv_in * (1.0 - h_m)
            tl.store(pout_ptr + bm_offs, p_m, mask=b_mask)

            force_m = (force_cpn > 0.5) & (is_c_m > 0.5)
            eff_bc = tl.maximum(tl.where(force_m, tl.minimum(b_cpn_m, b_ko_m), b_cpn_m), 1e-6)
            bc_lo = tl.maximum(tl.minimum(eff_bc, eff_bc - cpn_sp_mag), 1e-6)
            bc_hi = tl.maximum(tl.maximum(eff_bc, eff_bc - cpn_sp_mag), 1e-6)
            dbc = tl.maximum(bc_hi - bc_lo, 1e-7)
            uc_hi = (tl.log(bc_hi) - ln_fc + 0.5 * vc * vc) * inv_l
            uc_lo = (tl.log(bc_lo) - ln_fc + 0.5 * vc * vc) * inv_l
            dphic_m_tail = _triton_normcdf64(-(uc_lo - vc)) - _triton_normcdf64(-(uc_hi - vc))
            dphic_m = tl.where(uc_hi + uc_lo > 0.0, dphic_m_tail, _triton_normcdf64(uc_hi - vc) - _triton_normcdf64(uc_lo - vc))
            q_sp_tail = (fc * dphic_m_tail + bc_hi * _triton_normcdf64(-uc_hi) - bc_lo * _triton_normcdf64(-uc_lo)) / dbc
            q_sp_main = 1.0 - (bc_hi * _triton_normcdf64(uc_hi) - bc_lo * _triton_normcdf64(uc_lo) - fc * dphic_m) / dbc
            q_sp_raw = tl.where(uc_hi + uc_lo > 0.0, q_sp_tail, q_sp_main)
            in_q01 = (q_sp_raw > 0.0) & (q_sp_raw < 1.0)
            dq_df_sp = tl.where(in_q01, dphic_m / dbc, 0.0)
            dq_dv_sp = tl.where(in_q01, (bc_lo * _triton_normpdf64(uc_lo) - bc_hi * _triton_normpdf64(uc_hi)) / dbc, 0.0)

            uc_pt = (tl.log(eff_bc) - ln_fc + 0.5 * vc * vc) * inv_l
            q_pt_raw = tl.minimum(tl.maximum(_triton_normcdf64(-uc_pt), 0.0), 1.0)
            phic_pt = _triton_normpdf64(uc_pt)
            dq_df_pt = phic_pt * inv_l / fc
            dq_dv_pt = phic_pt * (-tl.log(fc / eff_bc) * inv_l * inv_l - 0.5)

            q_base = tl.where(is_q_sp, tl.minimum(tl.maximum(q_sp_raw, 0.0), 1.0), q_pt_raw)
            dq_df_b = tl.where(is_q_sp, dq_df_sp, dq_df_pt)
            dq_dv_b = tl.where(is_q_sp, dq_dv_sp, dq_dv_pt)

            use_fh = force_m & (h_m > q_base)
            q_raw_eff = tl.where(use_fh, h_m, q_base)
            dq_df_eff = tl.where(use_fh, dh_df, dq_df_b)
            dq_dv_eff = tl.where(use_fh, dh_dv, dq_dv_b)
            not_guar = (is_g_raw <= 0.5) & (sm_m > 0.5)
            q_m = tl.where(is_g_raw > 0.5, 1.0, q_raw_eff) * sm_m
            dq_df_m = tl.where(not_guar, dq_df_eff, 0.0)
            dq_dv_m = tl.where(not_guar, dq_dv_eff, 0.0)
            tl.store(qout_ptr + bm_offs, q_m, mask=b_mask)

            is_last_m = idx_last == m
            fc_term = tl.where(is_last_m, fc, fc_term)
            vc_term = tl.where(is_last_m, vc, vc_term)
            surv_term = tl.where(is_last_m, surv_next, surv_term)
            surv_prev_term = tl.where(is_last_m, surv_in, surv_prev_term)
            is_call_term = tl.where(is_last_m, is_c_raw, is_call_term)
            b_ko_term = tl.where(is_last_m, b_ko_m, b_ko_term)

            if SAVE_SCRATCH:
                sb = m * B + b_offs
                tl.store(scratch_ptr + 0 * MB + sb, zm_in, mask=b_mask)
                tl.store(scratch_ptr + 1 * MB + sb, zs_in, mask=b_mask)
                tl.store(scratch_ptr + 2 * MB + sb, surv_in, mask=b_mask)
                tl.store(scratch_ptr + 3 * MB + sb, fc, mask=b_mask)
                tl.store(scratch_ptr + 4 * MB + sb, h_m, mask=b_mask)
                tl.store(scratch_ptr + 5 * MB + sb, dh_df, mask=b_mask)
                tl.store(scratch_ptr + 6 * MB + sb, dh_dv, mask=b_mask)
                tl.store(scratch_ptr + 7 * MB + sb, dq_df_m, mask=b_mask)
                tl.store(scratch_ptr + 8 * MB + sb, dq_dv_m, mask=b_mask)

            if m < M - 1:
                m1_sp = -(vc * fc / db) * dphi_m
                m2_sp = m0_sp + vc * m1_sp + (vc / db) * (b_hi * phi_hi - b_lo * phi_lo)
                m0_s = tl.maximum(m0_sp, 1e-20)
                inv_m0 = 1.0 / m0_s
                mu_sp = m1_sp * inv_m0
                var_sp = tl.maximum(m2_sp * inv_m0 - mu_sp * mu_sp, 1e-8)

                cdf_pt = tl.maximum(_triton_normcdf64(u_pt), 1e-20)
                lam_pt = phi_pt / cdf_pt
                mu_pt = -lam_pt
                var_pt = tl.maximum(1.0 + u_pt * lam_pt - lam_pt * lam_pt, 1e-8)

                mu_u = tl.where(is_ko_sp, mu_sp, mu_pt)
                var_u = tl.where(is_ko_sp, var_sp, var_pt)
                c_on = is_c_m > 0.5
                z_pm = tl.where(c_on, zm_in + zs_in * mu_u, zm_in)
                z_pv = tl.where(c_on, tl.maximum(zs_in * zs_in * var_u, 1e-8), zs_in * zs_in)
                rn = tl.load(rn_ptr + b_offs * (M - 1) + m, mask=b_mask, other=0.0)
                z_mean = rn * z_pm
                z_std = tl.sqrt(tl.maximum(rn * rn * z_pv + (1.0 - rn * rn), 1e-8))

                if SAVE_SCRATCH:
                    sb = m * B + b_offs
                    dm0_df = -dphi_m / db
                    dm0_dl = (b_hi * phi_hi - b_lo * phi_lo) / db
                    phi_hi_m = (b_hi / fc) * phi_hi
                    phi_lo_m = (b_lo / fc) * phi_lo
                    dphi_m_df = -(phi_hi_m - phi_lo_m) * inv_l / fc
                    dm1_df = -(vc / db) * (dphi_m + fc * dphi_m_df)
                    dphi_m_dl = (-u_hi * phi_hi_m + u_lo * phi_lo_m) * inv_l
                    dm1_dl = -(fc / db) * (dphi_m + vc * dphi_m_dl)
                    b_uphi = b_hi * u_hi * phi_hi - b_lo * u_lo * phi_lo
                    dm2_df = dm0_df + vc * dm1_df + (vc / db) * (b_uphi * inv_l / fc)
                    dm2_dl = dm0_dl + m1_sp + vc * dm1_dl + (1.0 / db) * (
                        (b_hi * phi_hi - b_lo * phi_lo)
                        - vc * (b_hi * u_hi * phi_hi * (-u_hi * inv_l + 1.0) - b_lo * u_lo * phi_lo * (-u_lo * inv_l + 1.0))
                    )
                    dmu_df_sp = (dm1_df - mu_sp * dm0_df) * inv_m0
                    dmu_dl_sp = (dm1_dl - mu_sp * dm0_dl) * inv_m0
                    active_v = (m2_sp * inv_m0 - mu_sp * mu_sp) > 1e-8
                    dvar_df_sp = tl.where(active_v, (dm2_df - (m2_sp * inv_m0) * dm0_df) * inv_m0 - 2.0 * mu_sp * dmu_df_sp, 0.0)
                    dvar_dl_sp = tl.where(active_v, (dm2_dl - (m2_sp * inv_m0) * dm0_dl) * inv_m0 - 2.0 * mu_sp * dmu_dl_sp, 0.0)

                    dlam_du = -lam_pt * (u_pt + lam_pt)
                    dmu_df_pt = -dlam_du * (-inv_l / fc)
                    dmu_dl_pt = -dlam_du * (-u_pt * inv_l + 1.0)
                    dvar_du_pt = lam_pt + (u_pt - 2.0 * lam_pt) * dlam_du
                    active_vpt = (1.0 + u_pt * lam_pt - lam_pt * lam_pt) > 1e-8
                    dvar_df_pt = tl.where(active_vpt, dvar_du_pt * (-inv_l / fc), 0.0)
                    dvar_dl_pt = tl.where(active_vpt, dvar_du_pt * (-u_pt * inv_l + 1.0), 0.0)

                    dmu_df = tl.where(is_ko_sp, dmu_df_sp, dmu_df_pt)
                    dmu_dv = tl.where(is_ko_sp, dmu_dl_sp, dmu_dl_pt)
                    dvar_df = tl.where(is_ko_sp, dvar_df_sp, dvar_df_pt)
                    dvar_dv = tl.where(is_ko_sp, dvar_dl_sp, dvar_dl_pt)

                    tl.store(scratch_ptr + 9 * MB + sb, mu_u, mask=b_mask)
                    tl.store(scratch_ptr + 10 * MB + sb, var_u, mask=b_mask)
                    tl.store(scratch_ptr + 11 * MB + sb, dmu_df, mask=b_mask)
                    tl.store(scratch_ptr + 12 * MB + sb, dmu_dv, mask=b_mask)
                    tl.store(scratch_ptr + 13 * MB + sb, dvar_df, mask=b_mask)
                    tl.store(scratch_ptr + 14 * MB + sb, dvar_dv, mask=b_mask)
                    tl.store(scratch_ptr + 15 * MB + sb, z_pm, mask=b_mask)
                    tl.store(scratch_ptr + 16 * MB + sb, z_pv, mask=b_mask)
                    tl.store(scratch_ptr + 17 * MB + sb, z_std, mask=b_mask)
            surv = surv_next

        if SAVE_SCRATCH:
            tl.store(term_ptr + 0 * B + b_offs, fc_term, mask=b_mask)
            tl.store(term_ptr + 1 * B + b_offs, vc_term, mask=b_mask)
            tl.store(term_ptr + 2 * B + b_offs, surv_term, mask=b_mask)
            tl.store(term_ptr + 3 * B + b_offs, surv_prev_term, mask=b_mask)
            tl.store(term_ptr + 4 * B + b_offs, is_call_term, mask=b_mask)
            tl.store(term_ptr + 5 * B + b_offs, b_ko_term, mask=b_mask)
        else:
            k_strike = tl.load(kstr_ptr + b_offs, mask=b_mask, other=1.0)
            k_low = tl.load(klow_ptr + b_offs, mask=b_mask, other=0.0)
            active_ki = tl.load(actki_ptr + b_offs, mask=b_mask, other=0.0)
            b_ki = tl.load(bki_ptr + b_offs, mask=b_mask, other=0.0)
            ki_sp_mag = tl.load(kisp_ptr + b_offs, mask=b_mask, other=0.0)
            k_s_c = tl.maximum(k_strike, 0.0)
            k_l_c = tl.minimum(tl.maximum(k_low, 0.0), tl.maximum(k_s_c, 1e-6))
            has_put = (k_s_c > 1e-8) & (k_s_c > k_l_c)
            k_s_safe = tl.maximum(k_s_c, 1e-6)
            has_ki = active_ki > 0.5
            b_ki_c = tl.maximum(b_ki, 0.0)
            ki_sp_c = tl.maximum(ki_sp_mag, 0.0)
            b_ki_lo = tl.where(has_ki, tl.minimum(b_ki_c, 1e9), 0.0)
            b_ki_hi = tl.where(has_ki, b_ki_c + ki_sp_c, 0.0)
            is_ki_sp = has_ki & (ki_sp_c > 1e-6)
            has_ko = is_call_term > 0.5
            ko_sp_c = tl.maximum(ko_sp_mag, 0.0)
            b_ko_hi = tl.where(has_ko, tl.maximum(b_ko_term, 0.0), k_s_safe)
            b_ko_lo = tl.where(has_ko, tl.maximum(b_ko_hi - ko_sp_c, 0.0), k_s_safe)
            is_ko_sp_p = has_ko & (ko_sp_c > 1e-6)
            db_ki = tl.maximum(b_ki_hi - b_ki_lo, 1e-7)
            db_ko = tl.maximum(b_ko_hi - b_ko_lo, 1e-7)
            p0 = 0.0 * k_s_safe
            p1 = k_l_c
            p2 = tl.minimum(tl.maximum(b_ki_lo, 0.0), k_s_safe)
            p3 = tl.minimum(tl.maximum(b_ki_hi, 0.0), k_s_safe)
            p4 = tl.minimum(tl.maximum(b_ko_lo, 0.0), k_s_safe)
            p5 = tl.minimum(tl.maximum(b_ko_hi, 0.0), k_s_safe)
            p6 = k_s_safe
            for _ in tl.static_range(6):
                p0, p1 = tl.minimum(p0, p1), tl.maximum(p0, p1)
                p2, p3 = tl.minimum(p2, p3), tl.maximum(p2, p3)
                p4, p5 = tl.minimum(p4, p5), tl.maximum(p4, p5)
                p1, p2 = tl.minimum(p1, p2), tl.maximum(p1, p2)
                p3, p4 = tl.minimum(p3, p4), tl.maximum(p3, p4)
                p5, p6 = tl.minimum(p5, p6), tl.maximum(p5, p6)
            inv_vp = 1.0 / vc_term
            ln_fp = tl.log(fc_term)
            s2 = vc_term * vc_term
            exp_s2 = tl.exp(s2)
            exp_3s2 = tl.exp(3.0 * s2)
            f2 = fc_term * fc_term
            f3 = f2 * fc_term
            total = 0.0 * fc_term
            for k in tl.static_range(6):
                a = tl.where(k == 0, p0, tl.where(k == 1, p1, tl.where(k == 2, p2, tl.where(k == 3, p3, tl.where(k == 4, p4, p5)))))
                b = tl.where(k == 0, p1, tl.where(k == 1, p2, tl.where(k == 2, p3, tl.where(k == 3, p4, tl.where(k == 4, p5, p6)))))
                active_int = (b - a) > 1e-9
                mid = 0.5 * (a + b)
                below_low = mid <= k_l_c
                c0 = tl.where(below_low, k_s_safe - k_l_c, k_s_safe)
                c1 = tl.where(below_low, 0.0 * k_s_safe, -1.0 + 0.0 * k_s_safe)
                ki_dead = has_ki & (mid >= b_ki_hi)
                ki_ramp = is_ki_sp & (mid > b_ki_lo) & (mid < b_ki_hi)
                d0 = tl.where(ki_dead, 0.0, tl.where(ki_ramp, b_ki_hi / db_ki, 1.0))
                d1 = tl.where(ki_ramp, -1.0 / db_ki, 0.0)
                ko_dead = has_ko & (mid >= b_ko_hi)
                ko_ramp = is_ko_sp_p & (mid > b_ko_lo) & (mid < b_ko_hi)
                e0 = tl.where(ko_dead, 0.0, tl.where(ko_ramp, b_ko_hi / db_ko, 1.0))
                e1 = tl.where(ko_ramp, -1.0 / db_ko, 0.0)
                cd0 = c0 * d0
                cd1 = c0 * d1 + c1 * d0
                cd2 = c1 * d1
                q0 = cd0 * e0
                q1 = cd0 * e1 + cd1 * e0
                q2 = cd1 * e1 + cd2 * e0
                q3 = cd2 * e1
                has_a = a > 1e-12
                ub = (tl.log(tl.maximum(b, 1e-12)) - ln_fp + 0.5 * s2) * inv_vp
                ua = (tl.log(tl.maximum(a, 1e-12)) - ln_fp + 0.5 * s2) * inv_vp
                m0 = _triton_normcdf64(ub) - tl.where(has_a, _triton_normcdf64(ua), 0.0)
                m1 = fc_term * (_triton_normcdf64(ub - vc_term) - tl.where(has_a, _triton_normcdf64(ua - vc_term), 0.0))
                m2 = f2 * exp_s2 * (_triton_normcdf64(ub - 2.0 * vc_term) - tl.where(has_a, _triton_normcdf64(ua - 2.0 * vc_term), 0.0))
                m3 = f3 * exp_3s2 * (_triton_normcdf64(ub - 3.0 * vc_term) - tl.where(has_a, _triton_normcdf64(ua - 3.0 * vc_term), 0.0))
                total = total + tl.where(active_int, q0 * m0 + q1 * m1 + q2 * m2 + q3 * m3, 0.0)
            sm_t = tl.where(is_call_term > 0.5, surv_prev_term, surv_term)
            tl.store(lputout_ptr + b_offs, sm_t * tl.where(has_put, total, 0.0), mask=b_mask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _markov_bwd_kernel64(  # type: ignore[no-untyped-def]
        fw_ptr, vw_ptr, rn_ptr, sm_ptr, isc_ptr,
        kosp_ptr, kisp_ptr, kstr_ptr, klow_ptr, actki_ptr, bki_ptr,
        idxlast_ptr, gp_ptr, gq_ptr, glp_ptr,
        lputout_ptr, gfwout_ptr, gvwout_ptr, grnout_ptr,
        scratch_ptr, term_ptr,
        B, M: tl.constexpr, ALL_SCHED_ACT: tl.constexpr, BLOCK_B: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b_offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        b_mask = b_offs < B

        ko_sp_mag = tl.load(kosp_ptr + b_offs, mask=b_mask, other=0.0)
        ki_sp_mag = tl.load(kisp_ptr + b_offs, mask=b_mask, other=0.0)
        k_strike = tl.load(kstr_ptr + b_offs, mask=b_mask, other=1.0)
        k_low = tl.load(klow_ptr + b_offs, mask=b_mask, other=0.0)
        active_ki = tl.load(actki_ptr + b_offs, mask=b_mask, other=0.0)
        b_ki = tl.load(bki_ptr + b_offs, mask=b_mask, other=0.0)
        g_lp = tl.load(glp_ptr + b_offs, mask=b_mask, other=0.0)
        if ALL_SCHED_ACT:
            idx_last = tl.full([BLOCK_B], M - 1, dtype=tl.int32)
        else:
            idx_last = tl.load(idxlast_ptr + b_offs, mask=b_mask, other=M - 1).to(tl.int32)

        fc_term = tl.load(term_ptr + 0 * B + b_offs, mask=b_mask, other=1.0)
        vc_term = tl.load(term_ptr + 1 * B + b_offs, mask=b_mask, other=0.1)
        surv_term = tl.load(term_ptr + 2 * B + b_offs, mask=b_mask, other=1.0)
        surv_prev_term = tl.load(term_ptr + 3 * B + b_offs, mask=b_mask, other=1.0)
        is_call_term = tl.load(term_ptr + 4 * B + b_offs, mask=b_mask, other=0.0)
        b_ko_term = tl.load(term_ptr + 5 * B + b_offs, mask=b_mask, other=1.0)

        k_s_c = tl.maximum(k_strike, 0.0)
        k_l_c = tl.minimum(tl.maximum(k_low, 0.0), tl.maximum(k_s_c, 1e-6))
        has_put = (k_s_c > 1e-8) & (k_s_c > k_l_c)
        k_s_safe = tl.maximum(k_s_c, 1e-6)
        has_ki = active_ki > 0.5
        b_ki_c = tl.maximum(b_ki, 0.0)
        ki_sp_c = tl.maximum(ki_sp_mag, 0.0)
        b_ki_lo = tl.where(has_ki, tl.minimum(b_ki_c, 1e9), 0.0)
        b_ki_hi = tl.where(has_ki, b_ki_c + ki_sp_c, 0.0)
        is_ki_sp = has_ki & (ki_sp_c > 1e-6)
        has_ko = is_call_term > 0.5
        ko_sp_c = tl.maximum(ko_sp_mag, 0.0)
        b_ko_hi = tl.where(has_ko, tl.maximum(b_ko_term, 0.0), k_s_safe)
        b_ko_lo = tl.where(has_ko, tl.maximum(b_ko_hi - ko_sp_c, 0.0), k_s_safe)
        is_ko_sp_p = has_ko & (ko_sp_c > 1e-6)
        db_ki = tl.maximum(b_ki_hi - b_ki_lo, 1e-7)
        db_ko = tl.maximum(b_ko_hi - b_ko_lo, 1e-7)

        p0 = 0.0 * k_s_safe
        p1 = k_l_c
        p2 = tl.minimum(tl.maximum(b_ki_lo, 0.0), k_s_safe)
        p3 = tl.minimum(tl.maximum(b_ki_hi, 0.0), k_s_safe)
        p4 = tl.minimum(tl.maximum(b_ko_lo, 0.0), k_s_safe)
        p5 = tl.minimum(tl.maximum(b_ko_hi, 0.0), k_s_safe)
        p6 = k_s_safe
        for _ in tl.static_range(6):
            p0, p1 = tl.minimum(p0, p1), tl.maximum(p0, p1)
            p2, p3 = tl.minimum(p2, p3), tl.maximum(p2, p3)
            p4, p5 = tl.minimum(p4, p5), tl.maximum(p4, p5)
            p1, p2 = tl.minimum(p1, p2), tl.maximum(p1, p2)
            p3, p4 = tl.minimum(p3, p4), tl.maximum(p3, p4)
            p5, p6 = tl.minimum(p5, p6), tl.maximum(p5, p6)

        inv_v = 1.0 / vc_term
        ln_f = tl.log(fc_term)
        s2 = vc_term * vc_term
        exp_s2 = tl.exp(s2)
        exp_3s2 = tl.exp(3.0 * s2)
        f2 = fc_term * fc_term
        f3 = f2 * fc_term
        total = 0.0 * fc_term
        df_tot = 0.0 * fc_term
        dv_tot = 0.0 * fc_term

        for k in tl.static_range(6):
            a = tl.where(k == 0, p0, tl.where(k == 1, p1, tl.where(k == 2, p2, tl.where(k == 3, p3, tl.where(k == 4, p4, p5)))))
            b = tl.where(k == 0, p1, tl.where(k == 1, p2, tl.where(k == 2, p3, tl.where(k == 3, p4, tl.where(k == 4, p5, p6)))))
            active_int = (b - a) > 1e-9
            mid = 0.5 * (a + b)
            below_low = mid <= k_l_c
            c0 = tl.where(below_low, k_s_safe - k_l_c, k_s_safe)
            c1 = tl.where(below_low, 0.0 * k_s_safe, -1.0 + 0.0 * k_s_safe)
            ki_dead = has_ki & (mid >= b_ki_hi)
            ki_ramp = is_ki_sp & (mid > b_ki_lo) & (mid < b_ki_hi)
            d0 = tl.where(ki_dead, 0.0, tl.where(ki_ramp, b_ki_hi / db_ki, 1.0))
            d1 = tl.where(ki_ramp, -1.0 / db_ki, 0.0)
            ko_dead = has_ko & (mid >= b_ko_hi)
            ko_ramp = is_ko_sp_p & (mid > b_ko_lo) & (mid < b_ko_hi)
            e0 = tl.where(ko_dead, 0.0, tl.where(ko_ramp, b_ko_hi / db_ko, 1.0))
            e1 = tl.where(ko_ramp, -1.0 / db_ko, 0.0)
            cd0 = c0 * d0
            cd1 = c0 * d1 + c1 * d0
            cd2 = c1 * d1
            q0 = cd0 * e0
            q1 = cd0 * e1 + cd1 * e0
            q2 = cd1 * e1 + cd2 * e0
            q3 = cd2 * e1
            has_a = a > 1e-12
            ub = (tl.log(tl.maximum(b, 1e-12)) - ln_f + 0.5 * s2) * inv_v
            ua = (tl.log(tl.maximum(a, 1e-12)) - ln_f + 0.5 * s2) * inv_v
            dub_df = -inv_v / fc_term
            dub_dv = 1.0 - ub * inv_v
            dua_dv = 1.0 - ua * inv_v

            m0 = _triton_normcdf64(ub) - tl.where(has_a, _triton_normcdf64(ua), 0.0)
            p0b = _triton_normpdf64(ub)
            p0a = tl.where(has_a, _triton_normpdf64(ua), 0.0)
            dm0_df = (p0b - p0a) * dub_df
            dm0_dv = p0b * dub_dv - p0a * dua_dv

            i1 = _triton_normcdf64(ub - vc_term) - tl.where(has_a, _triton_normcdf64(ua - vc_term), 0.0)
            p1b = _triton_normpdf64(ub - vc_term)
            p1a = tl.where(has_a, _triton_normpdf64(ua - vc_term), 0.0)
            m1 = fc_term * i1
            dm1_df = i1 + fc_term * (p1b - p1a) * dub_df
            dm1_dv = fc_term * (p1b * (dub_dv - 1.0) - p1a * (dua_dv - 1.0))

            i2 = _triton_normcdf64(ub - 2.0 * vc_term) - tl.where(has_a, _triton_normcdf64(ua - 2.0 * vc_term), 0.0)
            p2b = _triton_normpdf64(ub - 2.0 * vc_term)
            p2a = tl.where(has_a, _triton_normpdf64(ua - 2.0 * vc_term), 0.0)
            s2_sc = f2 * exp_s2
            m2 = s2_sc * i2
            dm2_df = 2.0 * fc_term * exp_s2 * i2 + s2_sc * (p2b - p2a) * dub_df
            dm2_dv = 2.0 * vc_term * m2 + s2_sc * (p2b * (dub_dv - 2.0) - p2a * (dua_dv - 2.0))

            i3 = _triton_normcdf64(ub - 3.0 * vc_term) - tl.where(has_a, _triton_normcdf64(ua - 3.0 * vc_term), 0.0)
            p3b = _triton_normpdf64(ub - 3.0 * vc_term)
            p3a = tl.where(has_a, _triton_normpdf64(ua - 3.0 * vc_term), 0.0)
            s3_sc = f3 * exp_3s2
            m3 = s3_sc * i3
            dm3_df = 3.0 * f2 * exp_3s2 * i3 + s3_sc * (p3b - p3a) * dub_df
            dm3_dv = 6.0 * vc_term * m3 + s3_sc * (p3b * (dub_dv - 3.0) - p3a * (dua_dv - 3.0))

            total = total + tl.where(active_int, q0 * m0 + q1 * m1 + q2 * m2 + q3 * m3, 0.0)
            df_tot = df_tot + tl.where(active_int, q0 * dm0_df + q1 * dm1_df + q2 * dm2_df + q3 * dm3_df, 0.0)
            dv_tot = dv_tot + tl.where(active_int, q0 * dm0_dv + q1 * dm1_dv + q2 * dm2_dv + q3 * dm3_dv, 0.0)

        put_val = tl.where(has_put, total, 0.0)
        dput_df = tl.where(has_put, df_tot, 0.0)
        dput_dv = tl.where(has_put, dv_tot, 0.0)
        use_prev = is_call_term > 0.5
        sm_t = tl.where(use_prev, surv_prev_term, surv_term)
        tl.store(lputout_ptr + b_offs, sm_t * put_val, mask=b_mask)

        a_fc_term = g_lp * sm_t * dput_df
        a_vc_term = g_lp * sm_t * dput_dv
        a_sm_term = g_lp * put_val
        a_surv_t = tl.where(use_prev, 0.0, a_sm_term)
        a_surv_p = tl.where(use_prev, a_sm_term, 0.0)

        a_surv = tl.zeros([BLOCK_B], dtype=tl.float64)
        a_zm = tl.zeros([BLOCK_B], dtype=tl.float64)
        a_zs = tl.zeros([BLOCK_B], dtype=tl.float64)
        MB = M * B

        for m_rev in range(M):
            m = M - 1 - m_rev
            bm_offs = b_offs * M + m
            sb = m * B + b_offs
            zm_in = tl.load(scratch_ptr + 0 * MB + sb, mask=b_mask, other=0.0)
            zs_in = tl.load(scratch_ptr + 1 * MB + sb, mask=b_mask, other=1.0)
            surv_in = tl.load(scratch_ptr + 2 * MB + sb, mask=b_mask, other=1.0)
            fc = tl.load(scratch_ptr + 3 * MB + sb, mask=b_mask, other=1.0)
            h_m = tl.load(scratch_ptr + 4 * MB + sb, mask=b_mask, other=0.0)
            dh_df = tl.load(scratch_ptr + 5 * MB + sb, mask=b_mask, other=0.0)
            dh_dv = tl.load(scratch_ptr + 6 * MB + sb, mask=b_mask, other=0.0)
            dq_df_m = tl.load(scratch_ptr + 7 * MB + sb, mask=b_mask, other=0.0)
            dq_dv_m = tl.load(scratch_ptr + 8 * MB + sb, mask=b_mask, other=0.0)

            is_last_m = idx_last == m
            a_fc = tl.where(is_last_m, a_fc_term, 0.0)
            a_vc = tl.where(is_last_m, a_vc_term, 0.0)
            a_surv = tl.where(is_last_m, a_surv + a_surv_t, a_surv)

            if m < M - 1:
                mu_u = tl.load(scratch_ptr + 9 * MB + sb, mask=b_mask, other=0.0)
                var_u = tl.load(scratch_ptr + 10 * MB + sb, mask=b_mask, other=1.0)
                dmu_df = tl.load(scratch_ptr + 11 * MB + sb, mask=b_mask, other=0.0)
                dmu_dv = tl.load(scratch_ptr + 12 * MB + sb, mask=b_mask, other=0.0)
                dvar_df = tl.load(scratch_ptr + 13 * MB + sb, mask=b_mask, other=0.0)
                dvar_dv = tl.load(scratch_ptr + 14 * MB + sb, mask=b_mask, other=0.0)
                z_pm = tl.load(scratch_ptr + 15 * MB + sb, mask=b_mask, other=0.0)
                z_pv = tl.load(scratch_ptr + 16 * MB + sb, mask=b_mask, other=1.0)
                z_std_out = tl.load(scratch_ptr + 17 * MB + sb, mask=b_mask, other=1.0)

                is_c_m = tl.load(isc_ptr + bm_offs, mask=b_mask, other=0.0) * tl.load(sm_ptr + bm_offs, mask=b_mask, other=0.0)
                c_on = is_c_m > 0.5
                rn = tl.load(rn_ptr + b_offs * (M - 1) + m, mask=b_mask, other=0.0)
                a_z_std_sq = a_zs / (2.0 * tl.maximum(z_std_out, 1e-8))
                g_rn_m = a_zm * z_pm + a_z_std_sq * (2.0 * rn * (z_pv - 1.0))
                tl.store(grnout_ptr + b_offs * (M - 1) + m, g_rn_m, mask=b_mask)
                a_z_pm = a_zm * rn
                a_z_pv_raw = a_z_std_sq * (rn * rn)
                a_z_pv = tl.where(c_on & (zs_in * zs_in * var_u <= 1e-8), 0.0, a_z_pv_raw)
                a_mu = tl.where(c_on, a_z_pm * zs_in, 0.0)
                a_var = tl.where(c_on, a_z_pv * (zs_in * zs_in), 0.0)
                a_zm_next = a_z_pm
                a_zs_next = tl.where(c_on, a_z_pm * mu_u + a_z_pv * 2.0 * zs_in * var_u, a_z_pv * 2.0 * zs_in)
                a_fc = a_fc + a_mu * dmu_df + a_var * dvar_df
                a_vc = a_vc + a_mu * dmu_dv + a_var * dvar_dv
            else:
                a_zm_next = tl.zeros([BLOCK_B], dtype=tl.float64)
                a_zs_next = tl.zeros([BLOCK_B], dtype=tl.float64)

            g_p_m = tl.load(gp_ptr + bm_offs, mask=b_mask, other=0.0)
            g_q_m = tl.load(gq_ptr + bm_offs, mask=b_mask, other=0.0)
            a_h = g_p_m * surv_in - a_surv * surv_in
            a_surv_in = g_p_m * h_m + a_surv * (1.0 - h_m)
            a_surv_in = tl.where(is_last_m, a_surv_in + a_surv_p, a_surv_in)
            a_fc = a_fc + a_h * dh_df + g_q_m * dq_df_m
            a_vc = a_vc + a_h * dh_dv + g_q_m * dq_dv_m

            fw_m = tl.maximum(tl.load(fw_ptr + bm_offs, mask=b_mask, other=1.0), 1e-12)
            vw_m = tl.maximum(tl.load(vw_ptr + bm_offs, mask=b_mask, other=0.1), 1e-6)
            if m == 0:
                tl.store(gfwout_ptr + bm_offs, a_fc, mask=b_mask)
                tl.store(gvwout_ptr + bm_offs, a_vc, mask=b_mask)
            else:
                tl.store(gfwout_ptr + bm_offs, a_fc * (fc / fw_m), mask=b_mask)
                dfc_dvw = fc * (zm_in - vw_m * (1.0 - zs_in * zs_in))
                tl.store(gvwout_ptr + bm_offs, a_fc * dfc_dvw + a_vc * zs_in, mask=b_mask)
                a_zm = a_zm_next + a_fc * fc * vw_m
                a_zs = a_zs_next + a_fc * fc * (vw_m * vw_m * zs_in) + a_vc * vw_m
            a_surv = a_surv_in

    @triton.jit  # type: ignore[untyped-decorator]
    def _clark_fwd_kernel64(  # type: ignore[no-untyped-def]
        f_ptr, v_ptr, c_ptr, m_ptr,
        fwout_ptr, vwout_ptr, wout_ptr,
        sc_s_ptr, sc_w_ptr, sc_rho_ptr,
        BM, D: tl.constexpr, IS_BEST: tl.constexpr, SAVE_SC: tl.constexpr, BLOCK_BM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        r = pid * BLOCK_BM + tl.arange(0, BLOCK_BM)
        rmask = r < BM

        f0 = tl.load(f_ptr + r * D + 0, mask=rmask, other=1.0)
        v0 = tl.load(v_ptr + r * D + 0, mask=rmask, other=0.1)
        m0 = tl.load(m_ptr + r * D + 0, mask=rmask, other=0.0)

        if D == 1:
            tl.store(fwout_ptr + r, tl.maximum(f0, 1e-12), mask=rmask)
            tl.store(vwout_ptr + r, tl.maximum(v0, 1e-6), mask=rmask)
            tl.store(wout_ptr + r, m0, mask=rmask)
            return

        tl.store(wout_ptr + r * D + 0, tl.full([BLOCK_BM], 1.0, dtype=tl.float64), mask=rmask)
        for p in tl.static_range(1, D):
            tl.store(wout_ptr + r * D + p, tl.zeros([BLOCK_BM], dtype=tl.float64), mask=rmask)
            c0p = tl.load(c_ptr + (r * D + 0) * D + p, mask=rmask, other=0.0)
            tl.store(sc_rho_ptr + p * BM + r, c0p, mask=rmask)

        f_cur = f0
        v_cur = v0
        DB = D * BM
        DDB = D * D * BM

        for j in tl.static_range(1, D):
            act_j = tl.load(m_ptr + r * D + j, mask=rmask, other=0.0) > 0.5
            f_2 = tl.load(f_ptr + r * D + j, mask=rmask, other=1.0)
            v_2 = tl.load(v_ptr + r * D + j, mask=rmask, other=0.1)
            rho_12_raw = tl.load(sc_rho_ptr + j * BM + r, mask=rmask, other=0.0)
            rho_12 = tl.minimum(tl.maximum(rho_12_raw, -0.99999), 0.99999)
            var_m = tl.maximum(v_cur * v_cur + v_2 * v_2 - 2.0 * rho_12 * v_cur * v_2, 1e-12)
            sig_m = tl.sqrt(var_m)
            d1 = (tl.log(f_cur / f_2) + 0.5 * var_m) / sig_m
            d2 = d1 - sig_m
            w1 = _triton_normcdf64(d1) if IS_BEST else _triton_normcdf64(-d1)
            w2 = _triton_normcdf64(-d2) if IS_BEST else _triton_normcdf64(d2)
            e_m1 = tl.maximum(f_cur * w1 + f_2 * w2, 1e-12)

            f1_sq = (f_cur * f_cur) * tl.exp(v_cur * v_cur)
            f2_sq = (f_2 * f_2) * tl.exp(v_2 * v_2)
            sig_m2 = 2.0 * sig_m
            var_m2 = sig_m2 * sig_m2
            d1_sq = (tl.log(tl.maximum(f1_sq, 1e-24) / tl.maximum(f2_sq, 1e-24)) + 0.5 * var_m2) / sig_m2
            d2_sq = d1_sq - sig_m2
            w1_sq = _triton_normcdf64(d1_sq) if IS_BEST else _triton_normcdf64(-d1_sq)
            w2_sq = _triton_normcdf64(-d2_sq) if IS_BEST else _triton_normcdf64(d2_sq)
            e_m2 = f1_sq * w1_sq + f2_sq * w2_sq
            v_new = tl.sqrt(tl.maximum(tl.log(tl.maximum(e_m2 / tl.maximum(e_m1 * e_m1, 1e-24), 1.0)), 1e-12))

            if SAVE_SC:
                jb = j * BM + r
                tl.store(sc_s_ptr + 0 * DB + jb, f_cur, mask=rmask)
                tl.store(sc_s_ptr + 1 * DB + jb, v_cur, mask=rmask)
                tl.store(sc_s_ptr + 2 * DB + jb, rho_12_raw, mask=rmask)
                tl.store(sc_s_ptr + 3 * DB + jb, sig_m, mask=rmask)
                tl.store(sc_s_ptr + 4 * DB + jb, d1, mask=rmask)
                tl.store(sc_s_ptr + 5 * DB + jb, d2, mask=rmask)
                tl.store(sc_s_ptr + 6 * DB + jb, w1, mask=rmask)
                tl.store(sc_s_ptr + 7 * DB + jb, w2, mask=rmask)
                tl.store(sc_s_ptr + 8 * DB + jb, e_m1, mask=rmask)
                tl.store(sc_s_ptr + 9 * DB + jb, e_m2, mask=rmask)
                tl.store(sc_s_ptr + 10 * DB + jb, v_new, mask=rmask)
                tl.store(sc_s_ptr + 11 * DB + jb, d1_sq, mask=rmask)
                tl.store(sc_s_ptr + 12 * DB + jb, w1_sq, mask=rmask)
                tl.store(sc_s_ptr + 13 * DB + jb, w2_sq, mask=rmask)

            for p in range(j):
                wp_old = tl.load(wout_ptr + r * D + p, mask=rmask, other=0.0)
                if SAVE_SC:
                    tl.store(sc_w_ptr + (j * D + p) * BM + r, wp_old, mask=rmask)
                tl.store(wout_ptr + r * D + p, tl.where(act_j, wp_old * w1, wp_old), mask=rmask)
            tl.store(wout_ptr + r * D + j, tl.where(act_j, w2, 0.0), mask=rmask)

            for k in range(j + 1, D):
                v_rem = tl.load(v_ptr + r * D + k, mask=rmask, other=0.1)
                rho_1_k = tl.load(sc_rho_ptr + k * BM + r, mask=rmask, other=0.0)
                rho_2_k = tl.load(c_ptr + (r * D + j) * D + k, mask=rmask, other=0.0)
                f1_k = f_cur * tl.exp(rho_1_k * v_cur * v_rem)
                f2_k = f_2 * tl.exp(rho_2_k * v_2 * v_rem)
                d1_k = (tl.log(f1_k / f2_k) + 0.5 * var_m) / sig_m
                d2_k = d1_k - sig_m
                w1_k = _triton_normcdf64(d1_k) if IS_BEST else _triton_normcdf64(-d1_k)
                w2_k = _triton_normcdf64(-d2_k) if IS_BEST else _triton_normcdf64(d2_k)
                e_m_pk = f1_k * w1_k + f2_k * w2_k
                cov_log = tl.log(tl.maximum(e_m_pk / e_m1, 1e-12))
                rho_raw_k = cov_log / tl.maximum(v_new * v_rem, 1e-12)
                rho_new_k = tl.minimum(tl.maximum(rho_raw_k, -0.99999), 0.99999)
                if SAVE_SC:
                    tl.store(sc_rho_ptr + (j * D + k) * BM + r, rho_1_k, mask=rmask)
                    tl.store(sc_rho_ptr + DDB + (j * D + k) * BM + r, rho_raw_k, mask=rmask)
                tl.store(sc_rho_ptr + k * BM + r, tl.where(act_j, rho_new_k, rho_1_k), mask=rmask)

            f_cur = tl.where(act_j, e_m1, f_cur)
            v_cur = tl.where(act_j, v_new, v_cur)

        tl.store(fwout_ptr + r, f_cur, mask=rmask)
        tl.store(vwout_ptr + r, v_cur, mask=rmask)
        for i in tl.static_range(D):
            wi = tl.load(wout_ptr + r * D + i, mask=rmask, other=0.0)
            mi = tl.load(m_ptr + r * D + i, mask=rmask, other=0.0)
            tl.store(wout_ptr + r * D + i, wi * mi, mask=rmask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _clark_bwd_kernel64(  # type: ignore[no-untyped-def]
        f_ptr, v_ptr, c_ptr, m_ptr,
        gfw_ptr, gvw_ptr, gw_ptr,
        gfout_ptr, gvout_ptr, gcout_ptr,
        sc_s_ptr, sc_w_ptr, sc_rho_ptr,
        BM, D: tl.constexpr, IS_BEST: tl.constexpr, BLOCK_BM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        r = pid * BLOCK_BM + tl.arange(0, BLOCK_BM)
        rmask = r < BM
        sign = 1.0 if IS_BEST else -1.0

        if D == 1:
            f0 = tl.load(f_ptr + r, mask=rmask, other=1.0)
            v0 = tl.load(v_ptr + r, mask=rmask, other=0.1)
            gfw = tl.load(gfw_ptr + r, mask=rmask, other=0.0)
            gvw = tl.load(gvw_ptr + r, mask=rmask, other=0.0)
            tl.store(gfout_ptr + r, tl.where(f0 > 1e-12, gfw, 0.0), mask=rmask)
            tl.store(gvout_ptr + r, tl.where(v0 > 1e-6, gvw, 0.0), mask=rmask)
            tl.store(gcout_ptr + r, tl.zeros([BLOCK_BM], dtype=tl.float64), mask=rmask)
            return

        DB = D * BM
        DDB = D * D * BM

        for i in tl.static_range(D):
            mi = tl.load(m_ptr + r * D + i, mask=rmask, other=0.0)
            awi = tl.load(gw_ptr + r * D + i, mask=rmask, other=0.0) * mi
            tl.store(sc_w_ptr + i * BM + r, awi, mask=rmask)
            tl.store(sc_rho_ptr + i * BM + r, tl.zeros([BLOCK_BM], dtype=tl.float64), mask=rmask)
            tl.store(gfout_ptr + r * D + i, tl.zeros([BLOCK_BM], dtype=tl.float64), mask=rmask)
            tl.store(gvout_ptr + r * D + i, tl.zeros([BLOCK_BM], dtype=tl.float64), mask=rmask)
            for k in tl.static_range(D):
                tl.store(gcout_ptr + (r * D + i) * D + k, tl.zeros([BLOCK_BM], dtype=tl.float64), mask=rmask)

        a_fcur = tl.load(gfw_ptr + r, mask=rmask, other=0.0)
        a_vcur = tl.load(gvw_ptr + r, mask=rmask, other=0.0)

        for j_rev in tl.static_range(D - 1):
            j = D - 1 - j_rev
            jb = j * BM + r
            act_j = tl.load(m_ptr + r * D + j, mask=rmask, other=0.0) > 0.5
            fc = tl.load(sc_s_ptr + 0 * DB + jb, mask=rmask, other=1.0)
            vc = tl.load(sc_s_ptr + 1 * DB + jb, mask=rmask, other=0.1)
            r12_raw = tl.load(sc_s_ptr + 2 * DB + jb, mask=rmask, other=0.0)
            sigm = tl.load(sc_s_ptr + 3 * DB + jb, mask=rmask, other=0.1)
            d1 = tl.load(sc_s_ptr + 4 * DB + jb, mask=rmask, other=0.0)
            d2 = tl.load(sc_s_ptr + 5 * DB + jb, mask=rmask, other=0.0)
            w1 = tl.load(sc_s_ptr + 6 * DB + jb, mask=rmask, other=0.5)
            w2 = tl.load(sc_s_ptr + 7 * DB + jb, mask=rmask, other=0.5)
            em1 = tl.load(sc_s_ptr + 8 * DB + jb, mask=rmask, other=1.0)
            em2 = tl.load(sc_s_ptr + 9 * DB + jb, mask=rmask, other=1.0)
            vnew = tl.load(sc_s_ptr + 10 * DB + jb, mask=rmask, other=0.1)
            d1sq = tl.load(sc_s_ptr + 11 * DB + jb, mask=rmask, other=0.0)
            w1sq = tl.load(sc_s_ptr + 12 * DB + jb, mask=rmask, other=0.5)
            w2sq = tl.load(sc_s_ptr + 13 * DB + jb, mask=rmask, other=0.5)

            f2 = tl.load(f_ptr + r * D + j, mask=rmask, other=1.0)
            v2 = tl.load(v_ptr + r * D + j, mask=rmask, other=0.1)
            r12 = tl.minimum(tl.maximum(r12_raw, -0.99999), 0.99999)
            var_m = sigm * sigm

            a_em1 = tl.where(act_j, a_fcur, 0.0)
            a_fcur_in = tl.where(act_j, 0.0, a_fcur)
            a_vnew = tl.where(act_j, a_vcur, 0.0)
            a_vcur_in = tl.where(act_j, 0.0, a_vcur)
            a_f2 = tl.zeros([BLOCK_BM], dtype=tl.float64)
            a_v2 = tl.zeros([BLOCK_BM], dtype=tl.float64)
            a_sigm = tl.zeros([BLOCK_BM], dtype=tl.float64)

            for k in range(j + 1, D):
                a_rk_out = tl.load(sc_rho_ptr + k * BM + r, mask=rmask, other=0.0)
                a_rnew = tl.where(act_j, a_rk_out, 0.0)
                a_r1_pass = tl.where(act_j, 0.0, a_rk_out)

                v_rem = tl.load(v_ptr + r * D + k, mask=rmask, other=0.1)
                rho_1_k = tl.load(sc_rho_ptr + (j * D + k) * BM + r, mask=rmask, other=0.0)
                rho_raw_k = tl.load(sc_rho_ptr + DDB + (j * D + k) * BM + r, mask=rmask, other=0.0)
                rho_2_k = tl.load(c_ptr + (r * D + j) * D + k, mask=rmask, other=0.0)

                f1_k = fc * tl.exp(rho_1_k * vc * v_rem)
                f2_k = f2 * tl.exp(rho_2_k * v2 * v_rem)
                d1_k = (tl.log(f1_k / f2_k) + 0.5 * var_m) / sigm
                d2_k = d1_k - sigm
                w1_k = _triton_normcdf64(d1_k) if IS_BEST else _triton_normcdf64(-d1_k)
                w2_k = _triton_normcdf64(-d2_k) if IS_BEST else _triton_normcdf64(d2_k)
                epk = f1_k * w1_k + f2_k * w2_k

                in_rho = (rho_raw_k > -0.99999) & (rho_raw_k < 0.99999)
                a_rraw = tl.where(in_rho, a_rnew, 0.0)
                vn_vr = vnew * v_rem
                vn_vr_c = tl.maximum(vn_vr, 1e-12)
                a_covlog = a_rraw / vn_vr_c
                a_vnvr = tl.where(vn_vr > 1e-12, -a_rraw * rho_raw_k / vn_vr_c, 0.0)
                a_vnew = a_vnew + a_vnvr * v_rem

                avk = tl.load(gvout_ptr + r * D + k, mask=rmask, other=0.0)
                avk = avk + a_vnvr * vnew

                in_cov = (epk / em1) > 1e-12
                a_epk = tl.where(in_cov, a_covlog / epk, 0.0)
                a_em1 = a_em1 - tl.where(in_cov, a_covlog / em1, 0.0)

                a_f1k = a_epk * w1_k
                a_f2k = a_epk * w2_k
                a_sigm = a_sigm + a_epk * (sign * f1_k * _triton_normpdf64(d1_k))

                a_fcur_in = a_fcur_in + a_f1k * (f1_k / fc)
                a_exp1 = a_f1k * f1_k
                a_r1_pass = a_r1_pass + a_exp1 * (vc * v_rem)
                a_vcur_in = a_vcur_in + a_exp1 * (rho_1_k * v_rem)
                avk = avk + a_exp1 * (rho_1_k * vc)

                a_f2 = a_f2 + a_f2k * (f2_k / f2)
                a_exp2 = a_f2k * f2_k
                acjk = tl.load(gcout_ptr + (r * D + j) * D + k, mask=rmask, other=0.0)
                tl.store(gcout_ptr + (r * D + j) * D + k, acjk + a_exp2 * (v2 * v_rem), mask=rmask)
                a_v2 = a_v2 + a_exp2 * (rho_2_k * v_rem)
                avk = avk + a_exp2 * (rho_2_k * v2)

                tl.store(gvout_ptr + r * D + k, avk, mask=rmask)
                tl.store(sc_rho_ptr + k * BM + r, a_r1_pass, mask=rmask)

            a_w1 = tl.zeros([BLOCK_BM], dtype=tl.float64)
            for p in range(j):
                a_wp_out = tl.load(sc_w_ptr + p * BM + r, mask=rmask, other=0.0)
                wp_prev = tl.load(sc_w_ptr + (j * D + p) * BM + r, mask=rmask, other=0.0)
                a_w1 = a_w1 + tl.where(act_j, a_wp_out * wp_prev, 0.0)
                tl.store(sc_w_ptr + p * BM + r, tl.where(act_j, a_wp_out * w1, a_wp_out), mask=rmask)
            a_w2 = tl.where(act_j, tl.load(sc_w_ptr + j * BM + r, mask=rmask, other=0.0), 0.0)

            pw1 = sign * _triton_normpdf64(d1)
            pw2 = -sign * _triton_normpdf64(d2)
            a_d1_tot = a_w1 * pw1 + a_w2 * pw2
            a_sigm = a_sigm - a_w2 * pw2
            inv_sig = 1.0 / sigm
            a_fcur_in = a_fcur_in + a_d1_tot * inv_sig / fc
            a_f2 = a_f2 - a_d1_tot * inv_sig / f2
            a_sigm = a_sigm + a_d1_tot * (0.5 - (tl.log(fc / f2)) * inv_sig * inv_sig)

            em1_sq = em1 * em1
            em1_sq_c = tl.maximum(em1_sq, 1e-24)
            ratio_sq = em2 / em1_sq_c
            v_sq_raw = tl.log(tl.maximum(ratio_sq, 1.0))
            in_vn = (v_sq_raw > 1e-12) & (ratio_sq > 1.0)
            a_vsq = tl.where(in_vn, a_vnew / (2.0 * vnew), 0.0)
            a_em2 = a_vsq / em2
            a_em1 = a_em1 + tl.where((em1_sq > 1e-24) & in_vn, -2.0 * a_vsq / em1, 0.0)

            f1sq = (fc * fc) * tl.exp(vc * vc)
            f2sq = (f2 * f2) * tl.exp(v2 * v2)
            a_f1sq = a_em2 * w1sq
            a_f2sq = a_em2 * w2sq
            a_sigm = a_sigm + a_em2 * (2.0 * sign * f1sq * _triton_normpdf64(d1sq))
            a_fcur_in = a_fcur_in + a_f1sq * (2.0 * f1sq / fc)
            a_vcur_in = a_vcur_in + a_f1sq * (2.0 * vc * f1sq)
            a_f2 = a_f2 + a_f2sq * (2.0 * f2sq / f2)
            a_v2 = a_v2 + a_f2sq * (2.0 * v2 * f2sq)

            in_em1 = (fc * w1 + f2 * w2) > 1e-12
            a_em1_raw = tl.where(in_em1, a_em1, 0.0)
            a_fcur_in = a_fcur_in + a_em1_raw * w1
            a_f2 = a_f2 + a_em1_raw * w2
            a_sigm = a_sigm + a_em1_raw * (sign * fc * _triton_normpdf64(d1))

            var_m_raw = vc * vc + v2 * v2 - 2.0 * r12 * vc * v2
            a_varm = tl.where(var_m_raw > 1e-12, a_sigm / (2.0 * sigm), 0.0)
            a_vcur_in = a_vcur_in + a_varm * (2.0 * vc - 2.0 * r12 * v2)
            a_v2 = a_v2 + a_varm * (2.0 * v2 - 2.0 * r12 * vc)
            in_r12 = (r12_raw > -0.99999) & (r12_raw < 0.99999)
            tl.store(sc_rho_ptr + j * BM + r, tl.where(in_r12, a_varm * (-2.0 * vc * v2), 0.0), mask=rmask)

            gf_j = tl.load(gfout_ptr + r * D + j, mask=rmask, other=0.0)
            gv_j = tl.load(gvout_ptr + r * D + j, mask=rmask, other=0.0)
            tl.store(gfout_ptr + r * D + j, gf_j + a_f2, mask=rmask)
            tl.store(gvout_ptr + r * D + j, gv_j + a_v2, mask=rmask)

            a_fcur = a_fcur_in
            a_vcur = a_vcur_in

        gf0 = tl.load(gfout_ptr + r * D + 0, mask=rmask, other=0.0)
        gv0 = tl.load(gvout_ptr + r * D + 0, mask=rmask, other=0.0)
        tl.store(gfout_ptr + r * D + 0, gf0 + a_fcur, mask=rmask)
        tl.store(gvout_ptr + r * D + 0, gv0 + a_vcur, mask=rmask)
        for k in tl.static_range(1, D):
            ar0k = tl.load(sc_rho_ptr + k * BM + r, mask=rmask, other=0.0)
            tl.store(gcout_ptr + (r * D + 0) * D + k, ar0k, mask=rmask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _rnext_fwd_kernel64(  # type: ignore[no-untyped-def]
        w_ptr, v_ptr, c_ptr, rn_ptr, sc_ptr,
        B, M: tl.constexpr, D: tl.constexpr, SAVE_SC: tl.constexpr, BLOCK_B: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        bmask = b < B
        MB = (M - 1) * B
        for m in range(M - 1):
            num = tl.zeros([BLOCK_B], dtype=tl.float64)
            dc = tl.zeros([BLOCK_B], dtype=tl.float64)
            dn = tl.zeros([BLOCK_B], dtype=tl.float64)
            base_wc = (b * M + m) * D
            base_wn = (b * M + m + 1) * D
            for i in range(D):
                wci = tl.load(w_ptr + base_wc + i, mask=bmask, other=0.0)
                wni = tl.load(w_ptr + base_wn + i, mask=bmask, other=0.0)
                vci = tl.maximum(tl.load(v_ptr + (b * D + i) * M + m, mask=bmask, other=1e-6), 1e-6)
                vni = tl.maximum(tl.load(v_ptr + (b * D + i) * M + m + 1, mask=bmask, other=1e-6), 1e-6)
                cvci = wci * vci
                cvni = wni * vni
                for k in range(D):
                    wck = tl.load(w_ptr + base_wc + k, mask=bmask, other=0.0)
                    wnk = tl.load(w_ptr + base_wn + k, mask=bmask, other=0.0)
                    vck = tl.maximum(tl.load(v_ptr + (b * D + k) * M + m, mask=bmask, other=1e-6), 1e-6)
                    vnk = tl.maximum(tl.load(v_ptr + (b * D + k) * M + m + 1, mask=bmask, other=1e-6), 1e-6)
                    cik = tl.load(c_ptr + (b * D + i) * D + k, mask=bmask, other=0.0)
                    num = num + cvci * wnk * tl.minimum(vck, vnk) * cik
                    dc = dc + cvci * (wck * vck) * cik
                    dn = dn + cvni * (wnk * vnk) * cik
            dc_c = tl.maximum(dc, 1e-24)
            dn_c = tl.maximum(dn, 1e-24)
            den = tl.maximum(tl.sqrt(dc_c) * tl.sqrt(dn_c), 1e-24)
            rn_raw = num / den
            rn = tl.minimum(tl.maximum(rn_raw, -0.9999), 0.9999)
            tl.store(rn_ptr + b * (M - 1) + m, rn, mask=bmask)
            if SAVE_SC:
                sb = m * B + b
                tl.store(sc_ptr + 0 * MB + sb, num, mask=bmask)
                tl.store(sc_ptr + 1 * MB + sb, dc, mask=bmask)
                tl.store(sc_ptr + 2 * MB + sb, dn, mask=bmask)
                tl.store(sc_ptr + 3 * MB + sb, den, mask=bmask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _rnext_bwd_kernel64(  # type: ignore[no-untyped-def]
        w_ptr, v_ptr, c_ptr, grn_ptr, sc_ptr,
        gw_ptr, gv_ptr, gc_ptr,
        B, M: tl.constexpr, D: tl.constexpr, BLOCK_B: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        bmask = b < B
        MB = (M - 1) * B

        for i in range(D):
            for m in range(M):
                tl.store(gv_ptr + (b * D + i) * M + m, tl.zeros([BLOCK_B], dtype=tl.float64), mask=bmask)
            for k in range(D):
                tl.store(gc_ptr + (b * D + i) * D + k, tl.zeros([BLOCK_B], dtype=tl.float64), mask=bmask)

        for m in range(M - 1):
            sb = m * B + b
            num = tl.load(sc_ptr + 0 * MB + sb, mask=bmask, other=0.0)
            dc = tl.load(sc_ptr + 1 * MB + sb, mask=bmask, other=1.0)
            dn = tl.load(sc_ptr + 2 * MB + sb, mask=bmask, other=1.0)
            den = tl.load(sc_ptr + 3 * MB + sb, mask=bmask, other=1.0)
            grn = tl.load(grn_ptr + b * (M - 1) + m, mask=bmask, other=0.0)

            dc_c = tl.maximum(dc, 1e-24)
            dn_c = tl.maximum(dn, 1e-24)
            sc = tl.sqrt(dc_c)
            sn = tl.sqrt(dn_c)
            raw = num / den
            in_rn = (raw > -0.9999) & (raw < 0.9999)
            a_raw = tl.where(in_rn, grn, 0.0)
            a_num = a_raw / den
            a_den = tl.where((sc * sn) > 1e-24, -a_raw * num / (den * den), 0.0)
            a_dc = tl.where(dc > 1e-24, 0.5 * a_den * sn / sc, 0.0)
            a_dn = tl.where(dn > 1e-24, 0.5 * a_den * sc / sn, 0.0)

            base_wc = (b * M + m) * D
            base_wn = (b * M + m + 1) * D

            for i in range(D):
                wci = tl.load(w_ptr + base_wc + i, mask=bmask, other=0.0)
                wni = tl.load(w_ptr + base_wn + i, mask=bmask, other=0.0)
                vci_raw = tl.load(v_ptr + (b * D + i) * M + m, mask=bmask, other=1e-6)
                vni_raw = tl.load(v_ptr + (b * D + i) * M + m + 1, mask=bmask, other=1e-6)
                vci = tl.maximum(vci_raw, 1e-6)
                vni = tl.maximum(vni_raw, 1e-6)
                cvci = wci * vci
                cvni = wni * vni

                dwci = tl.zeros([BLOCK_B], dtype=tl.float64)
                dwni = tl.zeros([BLOCK_B], dtype=tl.float64)
                dvci = tl.zeros([BLOCK_B], dtype=tl.float64)
                dvni = tl.zeros([BLOCK_B], dtype=tl.float64)

                for k in range(D):
                    wck = tl.load(w_ptr + base_wc + k, mask=bmask, other=0.0)
                    wnk = tl.load(w_ptr + base_wn + k, mask=bmask, other=0.0)
                    vck_raw = tl.load(v_ptr + (b * D + k) * M + m, mask=bmask, other=1e-6)
                    vnk_raw = tl.load(v_ptr + (b * D + k) * M + m + 1, mask=bmask, other=1e-6)
                    vck = tl.maximum(vck_raw, 1e-6)
                    vnk = tl.maximum(vnk_raw, 1e-6)
                    cvck = wck * vck
                    cvnk = wnk * vnk
                    cik = tl.load(c_ptr + (b * D + i) * D + k, mask=bmask, other=0.0)
                    cki = tl.load(c_ptr + (b * D + k) * D + i, mask=bmask, other=0.0)

                    vmin_kk = tl.minimum(vck, vnk)
                    vmin_ii = tl.minimum(vci, vni)
                    dwci = dwci + a_num * cik * wnk * vmin_kk * vci
                    dvci = dvci + a_num * cik * wci * wnk * vmin_kk + a_num * cki * cvck * wni * tl.where(vci <= vni, 1.0, 0.0)
                    dwni = dwni + a_num * cki * cvck * vmin_ii
                    dvni = dvni + a_num * cki * cvck * wni * tl.where(vni < vci, 1.0, 0.0)

                    g_cvci = a_dc * (cik + cki) * cvck
                    dwci = dwci + g_cvci * vci
                    dvci = dvci + g_cvci * wci

                    g_cvni = a_dn * (cik + cki) * cvnk
                    dwni = dwni + g_cvni * vni
                    dvni = dvni + g_cvni * wni

                    gcik_old = tl.load(gc_ptr + (b * D + i) * D + k, mask=bmask, other=0.0)
                    gcik_add = a_num * cvci * wnk * vmin_kk + a_dc * cvci * cvck + a_dn * cvni * cvnk
                    tl.store(gc_ptr + (b * D + i) * D + k, gcik_old + gcik_add, mask=bmask)

                tl.store(gw_ptr + base_wc + i, tl.load(gw_ptr + base_wc + i, mask=bmask, other=0.0) + dwci, mask=bmask)
                tl.store(gw_ptr + base_wn + i, tl.load(gw_ptr + base_wn + i, mask=bmask, other=0.0) + dwni, mask=bmask)
                gv_c_old = tl.load(gv_ptr + (b * D + i) * M + m, mask=bmask, other=0.0)
                gv_n_old = tl.load(gv_ptr + (b * D + i) * M + m + 1, mask=bmask, other=0.0)
                tl.store(gv_ptr + (b * D + i) * M + m, gv_c_old + tl.where(vci_raw >= 1e-6, dvci, 0.0), mask=bmask)
                tl.store(gv_ptr + (b * D + i) * M + m + 1, gv_n_old + tl.where(vni_raw >= 1e-6, dvni, 0.0), mask=bmask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _clark_fwd_kernel32_dyn(  # type: ignore[no-untyped-def]
        f_ptr, v_ptr, c_ptr, m_ptr,
        fwout_ptr, vwout_ptr, wout_ptr,
        sc_s_ptr, sc_w_ptr, sc_rho_ptr,
        perm_ptr, N_ROWS, BM, D: tl.constexpr, IS_BEST: tl.constexpr, SAVE_SC: tl.constexpr,
        BLOCK_BM: tl.constexpr, HAS_PERM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        r = pid * BLOCK_BM + tl.arange(0, BLOCK_BM)
        rmask = r < BM
        p_id = r // N_ROWS if HAS_PERM else r
        row_id = r % N_ROWS if HAS_PERM else r
        src_0 = tl.load(perm_ptr + p_id * D + 0, mask=rmask, other=0) if HAS_PERM else 0
        f0 = tl.load(f_ptr + row_id * D + src_0, mask=rmask, other=1.0)
        v0 = tl.load(v_ptr + row_id * D + src_0, mask=rmask, other=0.1)
        m0 = tl.load(m_ptr + row_id * D + src_0, mask=rmask, other=0.0)
        if D == 1:
            tl.store(fwout_ptr + r, tl.maximum(f0, 1e-12), mask=rmask)
            tl.store(vwout_ptr + r, tl.maximum(v0, 1e-6), mask=rmask)
            tl.store(wout_ptr + r, m0, mask=rmask)
            return
        tl.store(wout_ptr + r * D + 0, tl.full([BLOCK_BM], 1.0, dtype=tl.float32), mask=rmask)
        for p in range(1, D):
            tl.store(wout_ptr + r * D + p, tl.zeros([BLOCK_BM], dtype=tl.float32), mask=rmask)
            src_p = tl.load(perm_ptr + p_id * D + p, mask=rmask, other=0) if HAS_PERM else p
            c0p = tl.load(c_ptr + (row_id * D + src_0) * D + src_p, mask=rmask, other=0.0)
            tl.store(sc_rho_ptr + p * BM + r, c0p, mask=rmask)
        f_cur = f0
        v_cur = v0
        DB = D * BM
        DDB = D * D * BM
        for j in range(1, D):
            src_j = tl.load(perm_ptr + p_id * D + j, mask=rmask, other=0) if HAS_PERM else j
            act_j = tl.load(m_ptr + row_id * D + src_j, mask=rmask, other=0.0) > 0.5
            f_2 = tl.load(f_ptr + row_id * D + src_j, mask=rmask, other=1.0)
            v_2 = tl.load(v_ptr + row_id * D + src_j, mask=rmask, other=0.1)
            rho_12_raw = tl.load(sc_rho_ptr + j * BM + r, mask=rmask, other=0.0)
            rho_12 = tl.minimum(tl.maximum(rho_12_raw, -0.99999), 0.99999)
            var_m = tl.maximum(v_cur * v_cur + v_2 * v_2 - 2.0 * rho_12 * v_cur * v_2, 1e-12)
            sig_m = tl.sqrt(var_m)
            d1 = (tl.log(f_cur / f_2) + 0.5 * var_m) / sig_m
            d2 = d1 - sig_m
            w1 = _triton_libdevice.normcdf(d1) if IS_BEST else _triton_libdevice.normcdf(-d1)
            w2 = _triton_libdevice.normcdf(-d2) if IS_BEST else _triton_libdevice.normcdf(d2)
            e_m1 = tl.maximum(f_cur * w1 + f_2 * w2, 1e-12)
            f1_sq = (f_cur * f_cur) * tl.exp(v_cur * v_cur)
            f2_sq = (f_2 * f_2) * tl.exp(v_2 * v_2)
            sig_m2 = 2.0 * sig_m
            var_m2 = sig_m2 * sig_m2
            d1_sq = (tl.log(tl.maximum(f1_sq, 1e-24) / tl.maximum(f2_sq, 1e-24)) + 0.5 * var_m2) / sig_m2
            d2_sq = d1_sq - sig_m2
            w1_sq = _triton_libdevice.normcdf(d1_sq) if IS_BEST else _triton_libdevice.normcdf(-d1_sq)
            w2_sq = _triton_libdevice.normcdf(-d2_sq) if IS_BEST else _triton_libdevice.normcdf(d2_sq)
            e_m2 = f1_sq * w1_sq + f2_sq * w2_sq
            v_new = tl.sqrt(tl.maximum(tl.log(tl.maximum(e_m2 / tl.maximum(e_m1 * e_m1, 1e-24), 1.0)), 1e-12))
            if SAVE_SC:
                jb = j * BM + r
                tl.store(sc_s_ptr + 0 * DB + jb, f_cur, mask=rmask)
                tl.store(sc_s_ptr + 1 * DB + jb, v_cur, mask=rmask)
                tl.store(sc_s_ptr + 2 * DB + jb, rho_12_raw, mask=rmask)
                tl.store(sc_s_ptr + 3 * DB + jb, sig_m, mask=rmask)
                tl.store(sc_s_ptr + 4 * DB + jb, d1, mask=rmask)
                tl.store(sc_s_ptr + 5 * DB + jb, d2, mask=rmask)
                tl.store(sc_s_ptr + 6 * DB + jb, w1, mask=rmask)
                tl.store(sc_s_ptr + 7 * DB + jb, w2, mask=rmask)
                tl.store(sc_s_ptr + 8 * DB + jb, e_m1, mask=rmask)
                tl.store(sc_s_ptr + 9 * DB + jb, e_m2, mask=rmask)
                tl.store(sc_s_ptr + 10 * DB + jb, v_new, mask=rmask)
                tl.store(sc_s_ptr + 11 * DB + jb, d1_sq, mask=rmask)
                tl.store(sc_s_ptr + 12 * DB + jb, w1_sq, mask=rmask)
                tl.store(sc_s_ptr + 13 * DB + jb, w2_sq, mask=rmask)
            for p in range(j):
                wp_old = tl.load(wout_ptr + r * D + p, mask=rmask, other=0.0)
                if SAVE_SC:
                    tl.store(sc_w_ptr + (j * D + p) * BM + r, wp_old, mask=rmask)
                tl.store(wout_ptr + r * D + p, tl.where(act_j, wp_old * w1, wp_old), mask=rmask)
            tl.store(wout_ptr + r * D + j, tl.where(act_j, w2, 0.0), mask=rmask)
            for k in range(j + 1, D):
                src_k = tl.load(perm_ptr + p_id * D + k, mask=rmask, other=0) if HAS_PERM else k
                v_rem = tl.load(v_ptr + row_id * D + src_k, mask=rmask, other=0.1)
                rho_1_k = tl.load(sc_rho_ptr + k * BM + r, mask=rmask, other=0.0)
                rho_2_k = tl.load(c_ptr + (row_id * D + src_j) * D + src_k, mask=rmask, other=0.0)
                f1_k = f_cur * tl.exp(rho_1_k * v_cur * v_rem)
                f2_k = f_2 * tl.exp(rho_2_k * v_2 * v_rem)
                d1_k = (tl.log(f1_k / f2_k) + 0.5 * var_m) / sig_m
                d2_k = d1_k - sig_m
                w1_k = _triton_libdevice.normcdf(d1_k) if IS_BEST else _triton_libdevice.normcdf(-d1_k)
                w2_k = _triton_libdevice.normcdf(-d2_k) if IS_BEST else _triton_libdevice.normcdf(d2_k)
                e_m_pk = f1_k * w1_k + f2_k * w2_k
                cov_log = tl.log(tl.maximum(e_m_pk / e_m1, 1e-12))
                rho_raw_k = cov_log / tl.maximum(v_new * v_rem, 1e-12)
                rho_new_k = tl.minimum(tl.maximum(rho_raw_k, -0.99999), 0.99999)
                if SAVE_SC:
                    tl.store(sc_rho_ptr + (j * D + k) * BM + r, rho_1_k, mask=rmask)
                    tl.store(sc_rho_ptr + DDB + (j * D + k) * BM + r, rho_raw_k, mask=rmask)
                tl.store(sc_rho_ptr + k * BM + r, tl.where(act_j, rho_new_k, rho_1_k), mask=rmask)
            f_cur = tl.where(act_j, e_m1, f_cur)
            v_cur = tl.where(act_j, v_new, v_cur)
        tl.store(fwout_ptr + r, f_cur, mask=rmask)
        tl.store(vwout_ptr + r, v_cur, mask=rmask)
        for i in range(D):
            src_i = tl.load(perm_ptr + p_id * D + i, mask=rmask, other=0) if HAS_PERM else i
            wi = tl.load(wout_ptr + r * D + i, mask=rmask, other=0.0)
            mi = tl.load(m_ptr + row_id * D + src_i, mask=rmask, other=0.0)
            tl.store(wout_ptr + r * D + i, wi * mi, mask=rmask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _clark_bwd_kernel32_dyn(  # type: ignore[no-untyped-def]
        f_ptr, v_ptr, c_ptr, m_ptr,
        gfw_ptr, gvw_ptr, gw_ptr,
        gfout_ptr, gvout_ptr, gcout_ptr,
        sc_s_ptr, sc_w_ptr, sc_rho_ptr,
        perm_ptr, N_ROWS, BM, D: tl.constexpr, IS_BEST: tl.constexpr,
        BLOCK_BM: tl.constexpr, HAS_PERM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        r = pid * BLOCK_BM + tl.arange(0, BLOCK_BM)
        rmask = r < BM
        p_id = r // N_ROWS if HAS_PERM else r
        row_id = r % N_ROWS if HAS_PERM else r
        sign = 1.0 if IS_BEST else -1.0
        if D == 1:
            f0 = tl.load(f_ptr + r, mask=rmask, other=1.0)
            v0 = tl.load(v_ptr + r, mask=rmask, other=0.1)
            gfw = tl.load(gfw_ptr + r, mask=rmask, other=0.0)
            gvw = tl.load(gvw_ptr + r, mask=rmask, other=0.0)
            tl.store(gfout_ptr + r, tl.where(f0 > 1e-12, gfw, 0.0), mask=rmask)
            tl.store(gvout_ptr + r, tl.where(v0 > 1e-6, gvw, 0.0), mask=rmask)
            if not HAS_PERM:
                tl.store(gcout_ptr + r, tl.zeros([BLOCK_BM], dtype=tl.float32), mask=rmask)
            return
        DB = D * BM
        DDB = D * D * BM
        for i in range(D):
            src_i = tl.load(perm_ptr + p_id * D + i, mask=rmask, other=0) if HAS_PERM else i
            mi = tl.load(m_ptr + row_id * D + src_i, mask=rmask, other=0.0)
            awi = tl.load(gw_ptr + r * D + i, mask=rmask, other=0.0) * mi
            tl.store(sc_w_ptr + i * BM + r, awi, mask=rmask)
            tl.store(sc_rho_ptr + i * BM + r, tl.zeros([BLOCK_BM], dtype=tl.float32), mask=rmask)
            tl.store(gfout_ptr + r * D + i, tl.zeros([BLOCK_BM], dtype=tl.float32), mask=rmask)
            tl.store(gvout_ptr + r * D + i, tl.zeros([BLOCK_BM], dtype=tl.float32), mask=rmask)
            if not HAS_PERM:
                for k in range(D):
                    tl.store(gcout_ptr + (r * D + i) * D + k, tl.zeros([BLOCK_BM], dtype=tl.float32), mask=rmask)
        a_fcur = tl.load(gfw_ptr + r, mask=rmask, other=0.0)
        a_vcur = tl.load(gvw_ptr + r, mask=rmask, other=0.0)
        for j_rev in range(D - 1):
            j = D - 1 - j_rev
            jb = j * BM + r
            src_j = tl.load(perm_ptr + p_id * D + j, mask=rmask, other=0) if HAS_PERM else j
            act_j = tl.load(m_ptr + row_id * D + src_j, mask=rmask, other=0.0) > 0.5
            fc = tl.load(sc_s_ptr + 0 * DB + jb, mask=rmask, other=1.0)
            vc = tl.load(sc_s_ptr + 1 * DB + jb, mask=rmask, other=0.1)
            r12_raw = tl.load(sc_s_ptr + 2 * DB + jb, mask=rmask, other=0.0)
            sigm = tl.load(sc_s_ptr + 3 * DB + jb, mask=rmask, other=0.1)
            d1 = tl.load(sc_s_ptr + 4 * DB + jb, mask=rmask, other=0.0)
            d2 = tl.load(sc_s_ptr + 5 * DB + jb, mask=rmask, other=0.0)
            w1 = tl.load(sc_s_ptr + 6 * DB + jb, mask=rmask, other=0.5)
            w2 = tl.load(sc_s_ptr + 7 * DB + jb, mask=rmask, other=0.5)
            em1 = tl.load(sc_s_ptr + 8 * DB + jb, mask=rmask, other=1.0)
            em2 = tl.load(sc_s_ptr + 9 * DB + jb, mask=rmask, other=1.0)
            vnew = tl.load(sc_s_ptr + 10 * DB + jb, mask=rmask, other=0.1)
            d1sq = tl.load(sc_s_ptr + 11 * DB + jb, mask=rmask, other=0.0)
            w1sq = tl.load(sc_s_ptr + 12 * DB + jb, mask=rmask, other=0.5)
            w2sq = tl.load(sc_s_ptr + 13 * DB + jb, mask=rmask, other=0.5)
            f2 = tl.load(f_ptr + row_id * D + src_j, mask=rmask, other=1.0)
            v2 = tl.load(v_ptr + row_id * D + src_j, mask=rmask, other=0.1)
            r12 = tl.minimum(tl.maximum(r12_raw, -0.99999), 0.99999)
            var_m = sigm * sigm
            a_em1 = tl.where(act_j, a_fcur, 0.0)
            a_fcur_in = tl.where(act_j, 0.0, a_fcur)
            a_vnew = tl.where(act_j, a_vcur, 0.0)
            a_vcur_in = tl.where(act_j, 0.0, a_vcur)
            a_f2 = tl.zeros([BLOCK_BM], dtype=tl.float32)
            a_v2 = tl.zeros([BLOCK_BM], dtype=tl.float32)
            a_sigm = tl.zeros([BLOCK_BM], dtype=tl.float32)
            for k in range(j + 1, D):
                src_k = tl.load(perm_ptr + p_id * D + k, mask=rmask, other=0) if HAS_PERM else k
                a_rk_out = tl.load(sc_rho_ptr + k * BM + r, mask=rmask, other=0.0)
                a_rnew = tl.where(act_j, a_rk_out, 0.0)
                a_r1_pass = tl.where(act_j, 0.0, a_rk_out)
                v_rem = tl.load(v_ptr + row_id * D + src_k, mask=rmask, other=0.1)
                rho_1_k = tl.load(sc_rho_ptr + (j * D + k) * BM + r, mask=rmask, other=0.0)
                rho_raw_k = tl.load(sc_rho_ptr + DDB + (j * D + k) * BM + r, mask=rmask, other=0.0)
                rho_2_k = tl.load(c_ptr + (row_id * D + src_j) * D + src_k, mask=rmask, other=0.0)
                f1_k = fc * tl.exp(rho_1_k * vc * v_rem)
                f2_k = f2 * tl.exp(rho_2_k * v2 * v_rem)
                d1_k = (tl.log(f1_k / f2_k) + 0.5 * var_m) / sigm
                d2_k = d1_k - sigm
                w1_k = _triton_libdevice.normcdf(d1_k) if IS_BEST else _triton_libdevice.normcdf(-d1_k)
                w2_k = _triton_libdevice.normcdf(-d2_k) if IS_BEST else _triton_libdevice.normcdf(d2_k)
                epk = f1_k * w1_k + f2_k * w2_k
                in_rho = (rho_raw_k > -0.99999) & (rho_raw_k < 0.99999)
                a_rraw = tl.where(in_rho, a_rnew, 0.0)
                vn_vr = vnew * v_rem
                vn_vr_c = tl.maximum(vn_vr, 1e-12)
                a_covlog = a_rraw / vn_vr_c
                a_vnvr = tl.where(vn_vr > 1e-12, -a_rraw * rho_raw_k / vn_vr_c, 0.0)
                a_vnew = a_vnew + a_vnvr * v_rem
                avk = tl.load(gvout_ptr + r * D + k, mask=rmask, other=0.0)
                avk = avk + a_vnvr * vnew
                in_cov = (epk / em1) > 1e-12
                a_epk = tl.where(in_cov, a_covlog / epk, 0.0)
                a_em1 = a_em1 - tl.where(in_cov, a_covlog / em1, 0.0)
                a_f1k = a_epk * w1_k
                a_f2k = a_epk * w2_k
                a_sigm = a_sigm + a_epk * (sign * f1_k * (0.3989422804014327 * tl.exp(-0.5 * d1_k * d1_k)))
                a_fcur_in = a_fcur_in + a_f1k * (f1_k / fc)
                a_exp1 = a_f1k * f1_k
                a_r1_pass = a_r1_pass + a_exp1 * (vc * v_rem)
                a_vcur_in = a_vcur_in + a_exp1 * (rho_1_k * v_rem)
                avk = avk + a_exp1 * (rho_1_k * vc)
                a_f2 = a_f2 + a_f2k * (f2_k / f2)
                a_exp2 = a_f2k * f2_k
                dc_jk = a_exp2 * (v2 * v_rem)
                if HAS_PERM:
                    tl.atomic_add(gcout_ptr + (row_id * D + src_j) * D + src_k, dc_jk, mask=rmask)
                else:
                    acjk = tl.load(gcout_ptr + (r * D + j) * D + k, mask=rmask, other=0.0)
                    tl.store(gcout_ptr + (r * D + j) * D + k, acjk + dc_jk, mask=rmask)
                a_v2 = a_v2 + a_exp2 * (rho_2_k * v_rem)
                avk = avk + a_exp2 * (rho_2_k * v2)
                tl.store(gvout_ptr + r * D + k, avk, mask=rmask)
                tl.store(sc_rho_ptr + k * BM + r, a_r1_pass, mask=rmask)
            a_w1 = tl.zeros([BLOCK_BM], dtype=tl.float32)
            for p in range(j):
                a_wp_out = tl.load(sc_w_ptr + p * BM + r, mask=rmask, other=0.0)
                wp_prev = tl.load(sc_w_ptr + (j * D + p) * BM + r, mask=rmask, other=0.0)
                a_w1 = a_w1 + tl.where(act_j, a_wp_out * wp_prev, 0.0)
                tl.store(sc_w_ptr + p * BM + r, tl.where(act_j, a_wp_out * w1, a_wp_out), mask=rmask)
            a_w2 = tl.where(act_j, tl.load(sc_w_ptr + j * BM + r, mask=rmask, other=0.0), 0.0)
            pw1 = sign * (0.3989422804014327 * tl.exp(-0.5 * d1 * d1))
            pw2 = -sign * (0.3989422804014327 * tl.exp(-0.5 * d2 * d2))
            a_d1_tot = a_w1 * pw1 + a_w2 * pw2
            a_sigm = a_sigm - a_w2 * pw2
            inv_sig = 1.0 / sigm
            a_fcur_in = a_fcur_in + a_d1_tot * inv_sig / fc
            a_f2 = a_f2 - a_d1_tot * inv_sig / f2
            a_sigm = a_sigm + a_d1_tot * (0.5 - (tl.log(fc / f2)) * inv_sig * inv_sig)
            em1_sq = em1 * em1
            em1_sq_c = tl.maximum(em1_sq, 1e-24)
            ratio_sq = em2 / em1_sq_c
            v_sq_raw = tl.log(tl.maximum(ratio_sq, 1.0))
            in_vn = (v_sq_raw > 1e-12) & (ratio_sq > 1.0)
            a_vsq = tl.where(in_vn, a_vnew / (2.0 * vnew), 0.0)
            a_em2 = a_vsq / em2
            a_em1 = a_em1 + tl.where((em1_sq > 1e-24) & in_vn, -2.0 * a_vsq / em1, 0.0)
            f1sq = (fc * fc) * tl.exp(vc * vc)
            f2sq = (f2 * f2) * tl.exp(v2 * v2)
            a_f1sq = a_em2 * w1sq
            a_f2sq = a_em2 * w2sq
            a_sigm = a_sigm + a_em2 * (2.0 * sign * f1sq * (0.3989422804014327 * tl.exp(-0.5 * d1sq * d1sq)))
            a_fcur_in = a_fcur_in + a_f1sq * (2.0 * f1sq / fc)
            a_vcur_in = a_vcur_in + a_f1sq * (2.0 * vc * f1sq)
            a_f2 = a_f2 + a_f2sq * (2.0 * f2sq / f2)
            a_v2 = a_v2 + a_f2sq * (2.0 * v2 * f2sq)
            in_em1 = (fc * w1 + f2 * w2) > 1e-12
            a_em1_raw = tl.where(in_em1, a_em1, 0.0)
            a_fcur_in = a_fcur_in + a_em1_raw * w1
            a_f2 = a_f2 + a_em1_raw * w2
            a_sigm = a_sigm + a_em1_raw * (sign * fc * (0.3989422804014327 * tl.exp(-0.5 * d1 * d1)))
            var_m_raw = vc * vc + v2 * v2 - 2.0 * r12 * vc * v2
            a_varm = tl.where(var_m_raw > 1e-12, a_sigm / (2.0 * sigm), 0.0)
            a_vcur_in = a_vcur_in + a_varm * (2.0 * vc - 2.0 * r12 * v2)
            a_v2 = a_v2 + a_varm * (2.0 * v2 - 2.0 * r12 * vc)
            in_r12 = (r12_raw > -0.99999) & (r12_raw < 0.99999)
            tl.store(sc_rho_ptr + j * BM + r, tl.where(in_r12, a_varm * (-2.0 * vc * v2), 0.0), mask=rmask)
            gf_j = tl.load(gfout_ptr + r * D + j, mask=rmask, other=0.0)
            gv_j = tl.load(gvout_ptr + r * D + j, mask=rmask, other=0.0)
            tl.store(gfout_ptr + r * D + j, gf_j + a_f2, mask=rmask)
            tl.store(gvout_ptr + r * D + j, gv_j + a_v2, mask=rmask)
            a_fcur = a_fcur_in
            a_vcur = a_vcur_in
        gf0 = tl.load(gfout_ptr + r * D + 0, mask=rmask, other=0.0)
        gv0 = tl.load(gvout_ptr + r * D + 0, mask=rmask, other=0.0)
        tl.store(gfout_ptr + r * D + 0, gf0 + a_fcur, mask=rmask)
        tl.store(gvout_ptr + r * D + 0, gv0 + a_vcur, mask=rmask)
        src_0 = tl.load(perm_ptr + p_id * D + 0, mask=rmask, other=0) if HAS_PERM else 0
        for k in range(1, D):
            ar0k = tl.load(sc_rho_ptr + k * BM + r, mask=rmask, other=0.0)
            if HAS_PERM:
                src_k = tl.load(perm_ptr + p_id * D + k, mask=rmask, other=0)
                tl.atomic_add(gcout_ptr + (row_id * D + src_0) * D + src_k, ar0k, mask=rmask)
            else:
                ac0k = tl.load(gcout_ptr + (r * D + 0) * D + k, mask=rmask, other=0.0)
                tl.store(gcout_ptr + (r * D + 0) * D + k, ac0k + ar0k, mask=rmask)

    def _eval_clark32_fwd_triton(
        f_o: torch.Tensor,
        v_o: torch.Tensor,
        c_o: torch.Tensor,
        m_o: torch.Tensor,
        *,
        is_best_of: bool,
        save_sc: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None]:
        n_rows, d_act = f_o.shape
        dev = f_o.device
        act_cnt = m_o.sum(dim=1, keepdim=True).clamp(min=1.0)
        f_scale = ((f_o * m_o).sum(dim=1, keepdim=True) / act_cnt).clamp(min=1e-16)
        u32 = torch.where(m_o > 0.5, f_o / f_scale, 1.0).float().contiguous()
        v32 = v_o.float().contiguous()
        c32 = c_o.float().contiguous()
        m32 = m_o.float().contiguous()
        fw32 = torch.empty(n_rows, dtype=torch.float32, device=dev)
        vw32 = torch.empty(n_rows, dtype=torch.float32, device=dev)
        wo32 = torch.empty((n_rows, d_act), dtype=torch.float32, device=dev)
        sc_rho = torch.empty((2 if save_sc else 1, d_act, d_act, n_rows) if save_sc else (d_act, n_rows), dtype=torch.float32, device=dev)
        sc_s = torch.empty((14, d_act, n_rows), dtype=torch.float32, device=dev) if save_sc else u32
        sc_w = torch.empty((d_act, d_act, n_rows), dtype=torch.float32, device=dev) if save_sc else u32
        _clark_fwd_kernel32_dyn[(triton.cdiv(n_rows, 64),)](
            u32, v32, c32, m32, fw32, vw32, wo32, sc_s, sc_w, sc_rho,
            u32, n_rows, n_rows, D=d_act, IS_BEST=is_best_of, SAVE_SC=save_sc, BLOCK_BM=64, HAS_PERM=False, num_warps=2,
        )
        fw64 = f_scale.squeeze(1) * fw32.double()
        vw64 = vw32.double()
        w64 = wo32.double()
        pack = (f_scale, u32, v32, c32, m32, fw64, act_cnt, sc_s, sc_w, sc_rho) if save_sc else None
        return fw64, vw64, w64, pack

    def _eval_clark32_bwd_triton(
        pack: tuple[torch.Tensor, ...],
        g_fw: torch.Tensor,
        g_vw: torch.Tensor,
        g_wo: torch.Tensor,
        *,
        is_best_of: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_scale, u32, v32, c32, m32, fw64, act_cnt, sc_s, sc_w, sc_rho = pack
        n_rows, d_act = u32.shape
        f_scale_1d = f_scale.squeeze(1)
        af32 = (g_fw * f_scale_1d).float().contiguous()
        av32 = g_vw.float().contiguous()
        aw32 = g_wo.float().contiguous()
        gu32 = torch.empty_like(u32)
        gv32 = torch.empty_like(v32)
        gc32 = torch.empty_like(c32)
        _clark_bwd_kernel32_dyn[(triton.cdiv(n_rows, 64),)](
            u32, v32, c32, m32, af32, av32, aw32, gu32, gv32, gc32, sc_s, sc_w, sc_rho,
            u32, n_rows, n_rows, D=d_act, IS_BEST=is_best_of, BLOCK_BM=64, HAS_PERM=False, num_warps=2,
        )
        gu64 = gu32.double()
        u64 = u32.double()
        m64 = m32.double()
        # Preserve degree-1 homogeneity across the row scale normalization.
        hom_corr = ((g_fw * fw64 - (gu64 * u64 * m64).sum(dim=1)) / act_cnt.squeeze(1)).unsqueeze(1) * m64
        return (gu64 + hom_corr) / f_scale, gv32.double(), gc32.double()

    def _sym_gsep_dyn(f_t: torch.Tensor, am_t: torch.Tensor | None = None) -> torch.Tensor:
        d_act = int(f_t.shape[1])
        ii, jj = torch.triu_indices(d_act, d_act, offset=1, device=f_t.device)
        dl = torch.log(f_t[:, ii].clamp(min=1e-16) / f_t[:, jj].clamp(min=1e-16))
        fac = (1.0 - torch.exp(-7812.5 * (dl * dl))).square()
        if am_t is not None:
            pair_on = (am_t[:, ii] > 0.5) & (am_t[:, jj] > 0.5)
            fac = torch.where(pair_on, fac, torch.ones_like(fac))
        out_gsep: torch.Tensor = fac.prod(dim=1)
        return out_gsep

    def _canon_sym_perms(d_act: int, device: torch.device) -> torch.Tensor:
        if d_act == 2:
            return torch.tensor([[0, 1]], dtype=torch.long, device=device)
        if d_act == 3:
            return torch.tensor([[0, 1, 2], [0, 2, 1], [1, 2, 0]], dtype=torch.long, device=device)
        base = torch.arange(d_act, dtype=torch.long, device=device)
        rolls = [torch.roll(base, shifts=-s, dims=0) for s in range(d_act)]
        return torch.stack(rolls + [torch.flip(p, dims=[0]) for p in rolls], dim=0)

    def _eval_sym_dyn32_fwd(
        f_t: torch.Tensor,
        v_t: torch.Tensor,
        c_t: torch.Tensor,
        am_t: torch.Tensor,
        *,
        is_best_of: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_rows, d_act = f_t.shape
        dev = f_t.device
        perms = _canon_sym_perms(d_act, dev)
        perms_i32 = perms.to(torch.int32).contiguous()
        n_p = int(perms.shape[0])
        chunk_n = max(1, 65_536 // n_p)
        act_cnt_row = am_t.sum(dim=1, keepdim=True).clamp(min=1.0)
        f_scale_row = ((f_t * am_t).sum(dim=1, keepdim=True) / act_cnt_row).clamp(min=1e-16)
        u32_base = torch.where(am_t > 0.5, f_t / f_scale_row, 1.0).float()
        v32_base = v_t.float()
        c32_base = c_t.float()
        m32_base = am_t.float()
        fw_out = torch.empty(n_rows, dtype=torch.float64, device=dev)
        vw_out = torch.empty(n_rows, dtype=torch.float64, device=dev)
        w_out = torch.empty((n_rows, d_act), dtype=torch.float64, device=dev)
        for s_c in range(0, n_rows, chunk_n):
            e_c = min(s_c + chunk_n, n_rows)
            cn = e_c - s_c
            n_tot = n_p * cn
            uc_b = u32_base[s_c:e_c].contiguous()
            vc_b = v32_base[s_c:e_c].contiguous()
            cc_b = c32_base[s_c:e_c].contiguous()
            mc_b = m32_base[s_c:e_c].contiguous()
            fw32 = torch.empty(n_tot, dtype=torch.float32, device=dev)
            vw32 = torch.empty(n_tot, dtype=torch.float32, device=dev)
            wo32 = torch.empty((n_tot, d_act), dtype=torch.float32, device=dev)
            sc_rho = torch.empty((d_act, n_tot), dtype=torch.float32, device=dev)
            _clark_fwd_kernel32_dyn[(triton.cdiv(n_tot, 64),)](
                uc_b, vc_b, cc_b, mc_b, fw32, vw32, wo32, uc_b, uc_b, sc_rho,
                perms_i32, cn, n_tot, D=d_act, IS_BEST=is_best_of, SAVE_SC=False, BLOCK_BM=64, HAS_PERM=True, num_warps=2,
            )
            f_sc_c = f_scale_row[s_c:e_c].squeeze(1).unsqueeze(0)
            ft_v = f_sc_c * fw32.view(n_p, cn).double()
            vt_v = vw32.view(n_p, cn).double()
            f_sym = ft_v.mean(dim=0)
            m2_sym = (ft_v * ft_v * torch.exp(vt_v * vt_v)).mean(dim=0)
            v_sym = torch.sqrt(torch.clamp(torch.log(torch.clamp(m2_sym / (f_sym * f_sym).clamp(min=1e-30), min=1.0)), min=1e-16))
            p_exp = perms.unsqueeze(1).expand(n_p, cn, d_act).reshape(n_tot, d_act)
            wt_uns = torch.zeros((n_tot, d_act), dtype=torch.float32, device=dev).scatter(1, p_exp, wo32).view(n_p, cn, d_act).mean(dim=0).double()
            fw_out[s_c:e_c] = f_sym
            vw_out[s_c:e_c] = v_sym
            w_out[s_c:e_c] = wt_uns
        return fw_out, vw_out, w_out

    def _eval_sym_dyn32_bwd(
        f_t: torch.Tensor,
        v_t: torch.Tensor,
        c_t: torch.Tensor,
        am_t: torch.Tensor,
        af: torch.Tensor,
        av: torch.Tensor,
        aw: torch.Tensor,
        *,
        is_best_of: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_rows, d_act = f_t.shape
        dev = f_t.device
        perms = _canon_sym_perms(d_act, dev)
        perms_i32 = perms.to(torch.int32).contiguous()
        n_p = int(perms.shape[0])
        chunk_n = max(1, 65_536 // n_p)
        act_cnt_row = am_t.sum(dim=1, keepdim=True).clamp(min=1.0)
        f_scale_row = ((f_t * am_t).sum(dim=1, keepdim=True) / act_cnt_row).clamp(min=1e-16)
        u32_base = torch.where(am_t > 0.5, f_t / f_scale_row, 1.0).float()
        v32_base = v_t.float()
        c32_base = c_t.float()
        m32_base = am_t.float()
        fw_out = torch.empty(n_rows, dtype=torch.float64, device=dev)
        vw_out = torch.empty(n_rows, dtype=torch.float64, device=dev)
        w_out = torch.empty((n_rows, d_act), dtype=torch.float64, device=dev)
        gf_out = torch.empty_like(f_t)
        gv_out = torch.empty_like(v_t)
        gc_out = torch.empty_like(c_t)
        inv_np = 1.0 / float(n_p)
        for s_c in range(0, n_rows, chunk_n):
            e_c = min(s_c + chunk_n, n_rows)
            cn = e_c - s_c
            n_tot = n_p * cn
            p_3d = perms.unsqueeze(1).expand(n_p, cn, d_act)
            p_exp = p_3d.reshape(n_tot, d_act)
            uc_b = u32_base[s_c:e_c].contiguous()
            vc_b = v32_base[s_c:e_c].contiguous()
            cc_b = c32_base[s_c:e_c].contiguous()
            mc_b = m32_base[s_c:e_c].contiguous()
            fw32 = torch.empty(n_tot, dtype=torch.float32, device=dev)
            vw32 = torch.empty(n_tot, dtype=torch.float32, device=dev)
            wo32 = torch.empty((n_tot, d_act), dtype=torch.float32, device=dev)
            sc_rho = torch.empty((2, d_act, d_act, n_tot), dtype=torch.float32, device=dev)
            sc_s = torch.empty((14, d_act, n_tot), dtype=torch.float32, device=dev)
            sc_w = torch.empty((d_act, d_act, n_tot), dtype=torch.float32, device=dev)
            _clark_fwd_kernel32_dyn[(triton.cdiv(n_tot, 64),)](
                uc_b, vc_b, cc_b, mc_b, fw32, vw32, wo32, sc_s, sc_w, sc_rho,
                perms_i32, cn, n_tot, D=d_act, IS_BEST=is_best_of, SAVE_SC=True, BLOCK_BM=64, HAS_PERM=True, num_warps=2,
            )
            f_sc_c = f_scale_row[s_c:e_c]
            f_sc_1d = f_sc_c.squeeze(1)
            ft_v = f_sc_1d.unsqueeze(0) * fw32.view(n_p, cn).double()
            vt_v = vw32.view(n_p, cn).double()
            m2_v = ft_v * ft_v * torch.exp(vt_v * vt_v)
            f_sym = ft_v.mean(dim=0)
            m2_sym = m2_v.mean(dim=0)
            fs2 = (f_sym * f_sym).clamp(min=1e-30)
            ratio = (m2_sym / fs2).clamp(min=1.0)
            lnr = torch.log(ratio)
            v_sym = torch.sqrt(torch.clamp(lnr, min=1e-16))
            wt_uns = torch.zeros((n_tot, d_act), dtype=torch.float32, device=dev).scatter(1, p_exp, wo32).view(n_p, cn, d_act).mean(dim=0).double()
            fw_out[s_c:e_c] = f_sym
            vw_out[s_c:e_c] = v_sym
            w_out[s_c:e_c] = wt_uns

            af_c = af[s_c:e_c]
            av_c = av[s_c:e_c]
            aw_c = aw[s_c:e_c]
            in_v = (lnr > 1e-16) & (m2_sym > fs2)
            a_lnr = torch.where(in_v, av_c / (2.0 * v_sym), 0.0)
            a_m2 = torch.where(m2_sym > fs2, a_lnr / m2_sym, 0.0)
            a_fsym_tot = af_c - torch.where((f_sym * f_sym > 1e-30) & (m2_sym > fs2), 2.0 * a_lnr / f_sym.clamp(min=1e-15), 0.0)
            a_ft_v = (a_fsym_tot.unsqueeze(0) + a_m2.unsqueeze(0) * (2.0 * m2_v / ft_v.clamp(min=1e-15))) * inv_np
            a_vt_p = ((a_m2.unsqueeze(0) * (2.0 * vt_v * m2_v)) * inv_np).reshape(n_tot)
            a_wt_p = (aw_c.unsqueeze(0).expand(n_p, cn, d_act).gather(2, p_3d) * inv_np).reshape(n_tot, d_act)

            f_sc_p = f_sc_c.unsqueeze(0).expand(n_p, cn, 1).reshape(n_tot, 1)
            f_sc_p1 = f_sc_p.squeeze(1)
            act_p = act_cnt_row[s_c:e_c].unsqueeze(0).expand(n_p, cn, 1).reshape(n_tot)
            af32 = (a_ft_v.reshape(n_tot) * f_sc_p1).float().contiguous()
            av32 = a_vt_p.float().contiguous()
            aw32 = a_wt_p.float().contiguous()
            gu32 = torch.empty((n_tot, d_act), dtype=torch.float32, device=dev)
            gv32 = torch.empty((n_tot, d_act), dtype=torch.float32, device=dev)
            gc32 = torch.zeros((cn, d_act, d_act), dtype=torch.float32, device=dev)
            _clark_bwd_kernel32_dyn[(triton.cdiv(n_tot, 64),)](
                uc_b, vc_b, cc_b, mc_b, af32, av32, aw32, gu32, gv32, gc32, sc_s, sc_w, sc_rho,
                perms_i32, cn, n_tot, D=d_act, IS_BEST=is_best_of, BLOCK_BM=64, HAS_PERM=True, num_warps=2,
            )
            uc_p = uc_b.unsqueeze(0).expand(n_p, cn, d_act).gather(2, p_3d).reshape(n_tot, d_act).double()
            mc_p = mc_b.unsqueeze(0).expand(n_p, cn, d_act).gather(2, p_3d).reshape(n_tot, d_act).double()
            gu64 = gu32.double()
            hom_corr = ((a_ft_v.reshape(n_tot) * ft_v.reshape(n_tot) - (gu64 * uc_p * mc_p).sum(dim=1)) / act_p).unsqueeze(1) * mc_p
            gfp = (gu64 + hom_corr) / f_sc_p
            gf_uns = torch.zeros((n_p, cn, d_act), dtype=torch.float64, device=dev).scatter_add(
                2, p_3d, gfp.view(n_p, cn, d_act)
            ).sum(dim=0)
            gv_uns = torch.zeros((n_p, cn, d_act), dtype=torch.float32, device=dev).scatter_add(
                2, p_3d, gv32.view(n_p, cn, d_act)
            ).sum(dim=0).double()
            gf_out[s_c:e_c] = gf_uns
            gv_out[s_c:e_c] = gv_uns
            gc_out[s_c:e_c] = gc32.double()
        return fw_out, vw_out, w_out, gf_out, gv_out, gc_out

    def _rnext_fwd_torch(
        w_all: torch.Tensor,
        v_in: torch.Tensor,
        corr_in: torch.Tensor,
    ) -> torch.Tensor:
        w_cur = w_all[:, :-1, :]
        w_nxt = w_all[:, 1:, :]
        v_t = v_in.transpose(1, 2).clamp(min=1e-6)
        v_cur = v_t[:, :-1, :]
        v_nxt = v_t[:, 1:, :]
        v_min = torch.minimum(v_cur.unsqueeze(-2), v_nxt.unsqueeze(-2))
        c_exp = corr_in.unsqueeze(1)
        num = (w_cur.unsqueeze(-1) * w_nxt.unsqueeze(-2) * c_exp * v_cur.unsqueeze(-1) * v_min).sum(dim=(-2, -1))
        cv_cur = w_cur * v_cur
        cv_nxt = w_nxt * v_nxt
        den_cur = (cv_cur.unsqueeze(-1) * cv_cur.unsqueeze(-2) * c_exp).sum(dim=(-2, -1)).clamp(min=1e-24)
        den_nxt = (cv_nxt.unsqueeze(-1) * cv_nxt.unsqueeze(-2) * c_exp).sum(dim=(-2, -1)).clamp(min=1e-24)
        return (num / (torch.sqrt(den_cur) * torch.sqrt(den_nxt)).clamp(min=1e-24)).clamp(-0.9999, 0.9999)

    def _rnext_fwd_bwd_torch(
        w_all: torch.Tensor,
        v_in: torch.Tensor,
        corr_in: torch.Tensor,
        g_rn_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            w_r = w_all.detach().requires_grad_(True)
            v_r = v_in.detach().requires_grad_(True)
            c_r = corr_in.detach().requires_grad_(True)
            rn = _rnext_fwd_torch(w_r, v_r, c_r)
            gw, gv, gc = torch.autograd.grad(rn, (w_r, v_r, c_r), grad_outputs=g_rn_m, allow_unused=True)
        return (
            gw.detach() if gw is not None else torch.zeros_like(w_all),
            gv.detach() if gv is not None else torch.zeros_like(v_in),
            gc.detach() if gc is not None else torch.zeros_like(corr_in),
        )

    _KCOL_DF_PAY = tl.constexpr(int(AutocallSchedCol.DF_PAY))
    _KCOL_CPN_RATE = tl.constexpr(int(AutocallSchedCol.COUPON_RATE))
    _KCOL_IS_CALL = tl.constexpr(int(AutocallSchedCol.IS_CALLABLE_FLAG))
    _KCOL_IS_GUAR = tl.constexpr(int(AutocallSchedCol.IS_GUARANTEED_CPN))
    _KCOL_P_PX = tl.constexpr(int(AutocallSchedCol.PROXY_KO_PROB))
    _KCOL_Q_PX = tl.constexpr(int(AutocallSchedCol.PROXY_CPN_PROB))
    _KCOL_K_PUT = tl.constexpr(int(AutocallGlobalCol.K_PUT))
    _KCOL_G_PUT = tl.constexpr(int(AutocallGlobalCol.G_PUT))
    _KCOL_C_MEM0 = tl.constexpr(int(AutocallGlobalCol.C_MEM_0))
    _KCOL_IS_MEM = tl.constexpr(int(AutocallGlobalCol.IS_MEMORY_FLAG))
    _KCOL_L_PUT = tl.constexpr(int(AutocallGlobalCol.L_PUT_PROXY))

    @triton.jit  # type: ignore[untyped-decorator]
    def _w1_fwd_bwd_kernel64(  # type: ignore[no-untyped-def]
        dh_ptr, dc_ptr, dp_ptr, sf_ptr, sm_ptr, gf_ptr,
        nonot_ptr, accpn_ptr, fcpn_ptr, islong_ptr, wfund_ptr, lowk_ptr, onestar_ptr, partcall_ptr,
        pv_out, pstop_out, surv_out, qcpn_out, putloss_out,
        gdh_out, gdc_out, gdp_out, gsf_out, ggf_out,
        sc_ptr,
        B, M: tl.constexpr, D_SCHED: tl.constexpr, D_GLOB: tl.constexpr,
        HAS_LOW_K: tl.constexpr, W_LEN: tl.constexpr, BLOCK_B: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        bmask = b < B
        MB = M * B

        dp = tl.load(dp_ptr + b, mask=bmask, other=0.0)
        no_not = tl.load(nonot_ptr + b, mask=bmask, other=0.0)
        ac_cpn = tl.load(accpn_ptr + b, mask=bmask, other=0.0)
        force_cpn = tl.load(fcpn_ptr + b, mask=bmask, other=0.0)
        is_long = tl.load(islong_ptr + b, mask=bmask, other=0.0)
        low_k = tl.load(lowk_ptr + b, mask=bmask, other=0.0)
        one_star = tl.load(onestar_ptr + b, mask=bmask, other=0.0)
        part_call = tl.load(partcall_ptr + b, mask=bmask, other=0.0)

        k_put_raw = tl.load(gf_ptr + b * D_GLOB + _KCOL_K_PUT, mask=bmask, other=0.0)
        k_put = tl.maximum(k_put_raw, 0.0)
        g_put = tl.load(gf_ptr + b * D_GLOB + _KCOL_G_PUT, mask=bmask, other=0.0)
        c_mem_0 = tl.load(gf_ptr + b * D_GLOB + _KCOL_C_MEM0, mask=bmask, other=0.0)
        is_mem_raw = tl.load(gf_ptr + b * D_GLOB + _KCOL_IS_MEM, mask=bmask, other=0.0)
        is_mem = tl.minimum(tl.maximum(is_mem_raw, 0.0), 1.0)
        l_put_raw = tl.load(gf_ptr + b * D_GLOB + _KCOL_L_PUT, mask=bmask, other=0.0)
        l_put_px = tl.maximum(l_put_raw, 0.0)

        last_idx = tl.zeros([BLOCK_B], dtype=tl.int32)
        has_any_i = tl.zeros([BLOCK_B], dtype=tl.int32)
        df_term = tl.zeros([BLOCK_B], dtype=tl.float64)
        surv_term = tl.zeros([BLOCK_B], dtype=tl.float64)

        cum_p = tl.zeros([BLOCK_B], dtype=tl.float64)
        s_cur = tl.full([BLOCK_B], 1.0, dtype=tl.float64)
        mem_buf = c_mem_0 * is_mem
        pv_not_sum = tl.zeros([BLOCK_B], dtype=tl.float64)
        pv_cpn = tl.zeros([BLOCK_B], dtype=tl.float64)
        fund_pv = tl.zeros([BLOCK_B], dtype=tl.float64)

        for m in range(M):
            bm = b * M + m
            sb = m * B + b
            act_m = tl.load(sm_ptr + bm, mask=bmask, other=0.0) > 0.5
            m_sched = tl.where(act_m, 1.0, 0.0)
            has_any_i = tl.where(act_m, 1, has_any_i)
            last_idx = tl.where(act_m, m, last_idx)

            sf_base = bm * D_SCHED
            df_m = tl.where(act_m, tl.load(sf_ptr + sf_base + _KCOL_DF_PAY, mask=bmask, other=0.0), 0.0)
            cr_m = tl.where(act_m, tl.load(sf_ptr + sf_base + _KCOL_CPN_RATE, mask=bmask, other=0.0), 0.0)
            cc_raw = tl.load(sf_ptr + sf_base + _KCOL_IS_CALL, mask=bmask, other=0.0)
            gc_raw = tl.load(sf_ptr + sf_base + _KCOL_IS_GUAR, mask=bmask, other=0.0)
            ppx_raw = tl.load(sf_ptr + sf_base + _KCOL_P_PX, mask=bmask, other=0.0)
            qpx_raw = tl.load(sf_ptr + sf_base + _KCOL_Q_PX, mask=bmask, other=0.0)

            c_call = tl.where(act_m, tl.minimum(tl.maximum(cc_raw, 0.0), 1.0), 0.0)
            g_cpn = tl.where(act_m, tl.minimum(tl.maximum(gc_raw, 0.0), 1.0), 0.0)
            p_px = tl.where(act_m, tl.minimum(tl.maximum(ppx_raw, 0.0), 1.0), 0.0) * c_call
            q_px = tl.where(act_m, tl.minimum(tl.maximum(qpx_raw, 0.0), 1.0), 0.0)

            sp_prev = tl.maximum(1.0 - cum_p, 1e-6)
            h_px = tl.minimum(tl.maximum(p_px / sp_prev, 1e-6), 1.0 - 1e-6)
            cum_p = cum_p + p_px

            dh_m = tl.load(dh_ptr + bm, mask=bmask, other=0.0)
            dc_m = tl.load(dc_ptr + bm, mask=bmask, other=0.0)
            sig_h = tl.sigmoid(tl.log(h_px / (1.0 - h_px)) + dh_m)
            haz_m = c_call * m_sched * sig_h

            surv_prev_m = s_cur
            s_cur = s_cur * (1.0 - haz_m)
            p_stop_m = surv_prev_m * haz_m
            tl.store(pstop_out + bm, p_stop_m, mask=bmask)
            tl.store(surv_out + bm, s_cur, mask=bmask)

            q_px_c = tl.minimum(tl.maximum(q_px, 1e-6), 1.0 - 1e-6)
            q_den = q_px * q_px + 1e-10
            q_gate = (q_px * q_px) / q_den
            sig_q = tl.sigmoid(tl.log(q_px_c / (1.0 - q_px_c)) + dc_m)
            q_cpn_m = m_sched * (g_cpn + (1.0 - g_cpn) * q_gate * sig_q)
            tl.store(qcpn_out + bm, q_cpn_m, mask=bmask)

            df_term = tl.where(act_m, df_m, df_term)
            surv_term = tl.where(act_m, s_cur, surv_term)
            pv_not_sum = pv_not_sum + df_m * p_stop_m * m_sched

            c_eff_m = cr_m + mem_buf
            sp_c = tl.maximum(surv_prev_m, 1e-12)
            q_forced = tl.maximum(q_cpn_m, p_stop_m / sp_c)
            eff_q_m = q_cpn_m + force_cpn * (q_forced - q_cpn_m)
            pv_cpn = pv_cpn + df_m * (surv_prev_m * eff_q_m * c_eff_m + p_stop_m * ac_cpn) * m_sched
            mem_next = tl.where(act_m, is_mem * (1.0 - eff_q_m) * c_eff_m, mem_buf)

            if m < W_LEN:
                wf_m = tl.load(wfund_ptr + b * W_LEN + m, mask=bmask, other=0.0)
                fund_pv = fund_pv + surv_prev_m * wf_m * m_sched

            tl.store(sc_ptr + 0 * MB + sb, p_px, mask=bmask)
            tl.store(sc_ptr + 1 * MB + sb, sp_prev, mask=bmask)
            tl.store(sc_ptr + 2 * MB + sb, haz_m, mask=bmask)
            tl.store(sc_ptr + 3 * MB + sb, sig_h, mask=bmask)
            tl.store(sc_ptr + 4 * MB + sb, surv_prev_m, mask=bmask)
            tl.store(sc_ptr + 5 * MB + sb, sig_q, mask=bmask)
            tl.store(sc_ptr + 6 * MB + sb, q_cpn_m, mask=bmask)
            tl.store(sc_ptr + 7 * MB + sb, c_eff_m, mask=bmask)
            tl.store(sc_ptr + 8 * MB + sb, eff_q_m, mask=bmask)
            tl.store(sc_ptr + 9 * MB + sb, mem_buf, mask=bmask)
            tl.store(sc_ptr + 10 * MB + sb, df_m, mask=bmask)
            tl.store(sc_ptr + 11 * MB + sb, c_call, mask=bmask)
            tl.store(sc_ptr + 12 * MB + sb, g_cpn, mask=bmask)

            mem_buf = mem_next

        has_any = has_any_i > 0
        df_term = tl.where(has_any, df_term, 0.0)
        pv_notional = (1.0 - no_not) * (pv_not_sum + df_term * surv_term)

        if HAS_LOW_K:
            k_eff = tl.maximum(k_put - tl.maximum(low_k, 0.0), 0.0)
        else:
            k_eff = k_put
        sp_term = tl.maximum(1.0 - cum_p, 0.0)
        denom_r = tl.maximum(k_eff * sp_term, 1e-12)
        r_px_raw = l_put_px / denom_r
        r_px = tl.minimum(tl.maximum(r_px_raw, 1e-12), 1.0 - 1e-12)
        sig_p = tl.sigmoid(tl.log(r_px / (1.0 - r_px)) + dp)
        surv_last = s_cur
        put_loss = k_eff * surv_last * sig_p
        tl.store(putloss_out + b, put_loss, mask=bmask)

        if HAS_LOW_K:
            cap_pl = k_eff * surv_term
            use_pl = put_loss <= cap_pl
            eff_put_loss = tl.where(use_pl, put_loss, cap_pl)
        else:
            use_pl = bmask
            eff_put_loss = put_loss

        put_sign = 2.0 * is_long - 1.0
        pv_option = put_sign * df_term * g_put * eff_put_loss + one_star + part_call
        pv = tl.where(has_any, pv_notional + pv_cpn + pv_option - fund_pv, 0.0)
        tl.store(pv_out + b, pv, mask=bmask)

        a_pv = tl.where(has_any, 1.0, 0.0)
        a_pv_not = a_pv
        a_pv_cpn = a_pv
        a_pv_opt = a_pv
        a_fund = -a_pv

        a_df_term = a_pv_not * (1.0 - no_not) * surv_term + a_pv_opt * put_sign * g_put * eff_put_loss
        a_surv_term = a_pv_not * (1.0 - no_not) * df_term
        a_g_put = a_pv_opt * put_sign * df_term * eff_put_loss
        a_eff_pl = a_pv_opt * put_sign * df_term * g_put
        if HAS_LOW_K:
            a_pl = tl.where(use_pl, a_eff_pl, 0.0)
            a_cap_pl = tl.where(use_pl, 0.0, a_eff_pl)
            a_surv_term = a_surv_term + a_cap_pl * k_eff
            a_k_eff = a_cap_pl * surv_term
        else:
            a_pl = a_eff_pl
            a_k_eff = tl.zeros([BLOCK_B], dtype=tl.float64)

        a_k_eff = a_k_eff + a_pl * surv_last * sig_p
        a_surv_last = a_pl * k_eff * sig_p
        a_sig_p = a_pl * k_eff * surv_last
        a_dp = a_sig_p * sig_p * (1.0 - sig_p)
        tl.store(gdp_out + b, a_dp, mask=bmask)

        a_r_px = a_dp / (r_px * (1.0 - r_px))
        in_rpx = (r_px_raw > 1e-12) & (r_px_raw < (1.0 - 1e-12))
        a_rpx_raw = tl.where(in_rpx, a_r_px, 0.0)
        a_l_put_px = a_rpx_raw / denom_r
        a_denom_r = tl.where((k_eff * sp_term) > 1e-12, -a_rpx_raw * r_px_raw / denom_r, 0.0)
        a_k_eff = a_k_eff + a_denom_r * sp_term
        a_sp_term = a_denom_r * k_eff
        a_cum_p_term = tl.where((1.0 - cum_p) > 0.0, -a_sp_term, 0.0)

        if HAS_LOW_K:
            a_k_put = tl.where((k_put - tl.maximum(low_k, 0.0) > 0.0) & (k_put_raw >= 0.0), a_k_eff, 0.0)
        else:
            a_k_put = tl.where(k_put_raw >= 0.0, a_k_eff, 0.0)
        a_l_put_raw = tl.where(l_put_raw >= 0.0, a_l_put_px, 0.0)

        for c in tl.static_range(D_GLOB):
            tl.store(ggf_out + b * D_GLOB + c, tl.zeros([BLOCK_B], dtype=tl.float64), mask=bmask)
        tl.store(ggf_out + b * D_GLOB + _KCOL_K_PUT, a_k_put, mask=bmask)
        tl.store(ggf_out + b * D_GLOB + _KCOL_G_PUT, a_g_put, mask=bmask)
        tl.store(ggf_out + b * D_GLOB + _KCOL_L_PUT, a_l_put_raw, mask=bmask)

        a_s_cur = a_surv_last
        a_mem_next = tl.zeros([BLOCK_B], dtype=tl.float64)
        a_cum_p_running = a_cum_p_term

        for m_rev in range(M):
            m = M - 1 - m_rev
            bm = b * M + m
            sb = m * B + b
            act_m = tl.load(sm_ptr + bm, mask=bmask, other=0.0) > 0.5
            m_sched = tl.where(act_m, 1.0, 0.0)
            is_last = has_any & (last_idx == m)

            p_px = tl.load(sc_ptr + 0 * MB + sb, mask=bmask, other=0.0)
            sp_prev = tl.load(sc_ptr + 1 * MB + sb, mask=bmask, other=1.0)
            haz_m = tl.load(sc_ptr + 2 * MB + sb, mask=bmask, other=0.0)
            sig_h = tl.load(sc_ptr + 3 * MB + sb, mask=bmask, other=0.5)
            surv_prev_m = tl.load(sc_ptr + 4 * MB + sb, mask=bmask, other=1.0)
            sig_q = tl.load(sc_ptr + 5 * MB + sb, mask=bmask, other=0.5)
            q_cpn_m = tl.load(sc_ptr + 6 * MB + sb, mask=bmask, other=0.0)
            c_eff_m = tl.load(sc_ptr + 7 * MB + sb, mask=bmask, other=0.0)
            eff_q_m = tl.load(sc_ptr + 8 * MB + sb, mask=bmask, other=0.0)
            df_m = tl.load(sc_ptr + 10 * MB + sb, mask=bmask, other=0.0)
            c_call = tl.load(sc_ptr + 11 * MB + sb, mask=bmask, other=0.0)
            g_cpn = tl.load(sc_ptr + 12 * MB + sb, mask=bmask, other=0.0)
            p_stop_m = surv_prev_m * haz_m

            a_s_out = a_s_cur + tl.where(is_last, a_surv_term, 0.0)
            a_df_m = tl.where(is_last, a_df_term, 0.0)

            a_mem_buf_in = tl.where(act_m, 0.0, a_mem_next)
            a_is_mem_step = tl.where(act_m, a_mem_next * (1.0 - eff_q_m) * c_eff_m, 0.0)
            a_eff_q_m = tl.where(act_m, -a_mem_next * is_mem * c_eff_m, 0.0)
            a_c_eff_m = tl.where(act_m, a_mem_next * is_mem * (1.0 - eff_q_m), 0.0)

            a_surv_prev = tl.zeros([BLOCK_B], dtype=tl.float64)
            if m < W_LEN:
                wf_m = tl.load(wfund_ptr + b * W_LEN + m, mask=bmask, other=0.0)
                a_surv_prev = a_surv_prev + a_fund * wf_m * m_sched

            cpn_term_m = surv_prev_m * eff_q_m * c_eff_m + p_stop_m * ac_cpn
            a_df_m = a_df_m + a_pv_cpn * cpn_term_m * m_sched
            a_surv_prev = a_surv_prev + a_pv_cpn * df_m * eff_q_m * c_eff_m * m_sched
            a_eff_q_m = a_eff_q_m + a_pv_cpn * df_m * surv_prev_m * c_eff_m * m_sched
            a_c_eff_m = a_c_eff_m + a_pv_cpn * df_m * surv_prev_m * eff_q_m * m_sched
            a_p_stop_m = a_pv_cpn * df_m * ac_cpn * m_sched

            a_cr_m = a_c_eff_m
            a_mem_buf_in = a_mem_buf_in + a_c_eff_m

            sp_c = tl.maximum(surv_prev_m, 1e-12)
            q_ratio = p_stop_m / sp_c
            use_q_forced = q_cpn_m >= q_ratio
            a_q_forced = a_eff_q_m * force_cpn
            a_q_cpn_m = a_eff_q_m * (1.0 - force_cpn) + tl.where(use_q_forced, a_q_forced, 0.0)
            a_q_ratio = tl.where(use_q_forced, 0.0, a_q_forced)
            a_p_stop_m = a_p_stop_m + a_q_ratio / sp_c
            a_surv_prev = a_surv_prev + tl.where(surv_prev_m > 1e-12, -a_q_ratio * p_stop_m / (sp_c * sp_c), 0.0)

            a_df_m = a_df_m + a_pv_not * (1.0 - no_not) * p_stop_m * m_sched
            a_p_stop_m = a_p_stop_m + a_pv_not * (1.0 - no_not) * df_m * m_sched

            sf_base = bm * D_SCHED
            qpx_raw = tl.load(sf_ptr + sf_base + _KCOL_Q_PX, mask=bmask, other=0.0)
            q_px = tl.where(act_m, tl.minimum(tl.maximum(qpx_raw, 0.0), 1.0), 0.0)
            q_px_c = tl.minimum(tl.maximum(q_px, 1e-6), 1.0 - 1e-6)
            q_den = q_px * q_px + 1e-10
            q_gate = (q_px * q_px) / q_den

            a_q_inner = a_q_cpn_m * m_sched
            a_g_cpn = a_q_inner * (1.0 - q_gate * sig_q)
            a_q_gate = a_q_inner * (1.0 - g_cpn) * sig_q
            a_sig_q = a_q_inner * (1.0 - g_cpn) * q_gate
            a_dc_m = a_sig_q * sig_q * (1.0 - sig_q)
            tl.store(gdc_out + bm, a_dc_m, mask=bmask)

            a_q_px = a_q_gate * (2.0 * q_px * 1e-10 / (q_den * q_den))
            in_qpxc = (q_px > 1e-6) & (q_px < (1.0 - 1e-6))
            a_q_px = a_q_px + tl.where(in_qpxc, a_dc_m / (q_px_c * (1.0 - q_px_c)), 0.0)

            a_surv_prev = a_surv_prev + a_p_stop_m * haz_m + a_s_out * (1.0 - haz_m)
            a_haz_m = a_p_stop_m * surv_prev_m - a_s_out * surv_prev_m

            a_c_call = a_haz_m * m_sched * sig_h
            a_sig_h = a_haz_m * c_call * m_sched
            a_dh_m = a_sig_h * sig_h * (1.0 - sig_h)
            tl.store(gdh_out + bm, a_dh_m, mask=bmask)

            h_raw = p_px / sp_prev
            h_px = tl.minimum(tl.maximum(h_raw, 1e-6), 1.0 - 1e-6)
            in_hpx = (h_raw > 1e-6) & (h_raw < (1.0 - 1e-6))
            a_h_raw = tl.where(in_hpx, a_dh_m / (h_px * (1.0 - h_px)), 0.0)
            a_p_px_step = a_h_raw / sp_prev + a_cum_p_running
            a_sp_prev_step = tl.where(sp_prev > 1e-6, -a_h_raw * p_px / (sp_prev * sp_prev), 0.0)
            a_cum_p_running = a_cum_p_running - a_sp_prev_step

            ppx_raw = tl.load(sf_ptr + sf_base + _KCOL_P_PX, mask=bmask, other=0.0)
            ppx_c = tl.where(act_m, tl.minimum(tl.maximum(ppx_raw, 0.0), 1.0), 0.0)
            a_c_call = a_c_call + a_p_px_step * ppx_c
            a_ppx_c = a_p_px_step * c_call

            cc_raw = tl.load(sf_ptr + sf_base + _KCOL_IS_CALL, mask=bmask, other=0.0)
            gc_raw = tl.load(sf_ptr + sf_base + _KCOL_IS_GUAR, mask=bmask, other=0.0)
            # Match PyTorch clamp(0, 1) subgradient at exact boundary values
            in_cc = act_m & (cc_raw >= 0.0) & (cc_raw <= 1.0)
            in_gc = act_m & (gc_raw >= 0.0) & (gc_raw <= 1.0)
            in_ppx = act_m & (ppx_raw >= 0.0) & (ppx_raw <= 1.0)
            in_qpx = act_m & (qpx_raw >= 0.0) & (qpx_raw <= 1.0)

            for cs in tl.static_range(D_SCHED):
                tl.store(gsf_out + sf_base + cs, tl.zeros([BLOCK_B], dtype=tl.float64), mask=bmask)
            tl.store(gsf_out + sf_base + _KCOL_DF_PAY, tl.where(act_m, a_df_m, 0.0), mask=bmask)
            tl.store(gsf_out + sf_base + _KCOL_CPN_RATE, tl.where(act_m, a_cr_m, 0.0), mask=bmask)
            tl.store(gsf_out + sf_base + _KCOL_IS_CALL, tl.where(in_cc, a_c_call, 0.0), mask=bmask)
            tl.store(gsf_out + sf_base + _KCOL_IS_GUAR, tl.where(in_gc, a_g_cpn, 0.0), mask=bmask)
            tl.store(gsf_out + sf_base + _KCOL_P_PX, tl.where(in_ppx, a_ppx_c, 0.0), mask=bmask)
            tl.store(gsf_out + sf_base + _KCOL_Q_PX, tl.where(in_qpx, a_q_px, 0.0), mask=bmask)

            g_ismem_cur = tl.load(ggf_out + b * D_GLOB + _KCOL_IS_MEM, mask=bmask, other=0.0)
            tl.store(ggf_out + b * D_GLOB + _KCOL_IS_MEM, g_ismem_cur + a_is_mem_step, mask=bmask)

            a_s_cur = a_surv_prev
            a_mem_next = a_mem_buf_in

        a_cmem0 = a_mem_next * is_mem
        a_ismem_init = a_mem_next * c_mem_0
        in_ismem = (is_mem_raw >= 0.0) & (is_mem_raw <= 1.0)
        g_ismem_tot = tl.load(ggf_out + b * D_GLOB + _KCOL_IS_MEM, mask=bmask, other=0.0) + a_ismem_init
        tl.store(ggf_out + b * D_GLOB + _KCOL_C_MEM0, a_cmem0, mask=bmask)
        tl.store(ggf_out + b * D_GLOB + _KCOL_IS_MEM, tl.where(in_ismem, g_ismem_tot, 0.0), mask=bmask)

    _SYM5_PAIR_LUT: tuple[tuple[int, int], ...] = (
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
        (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
    )
    _SYM5_PAIR_IDX: dict[tuple[int, int], int] = {p: i for i, p in enumerate(_SYM5_PAIR_LUT)}
    _SYM5_PAIR_REM: tuple[tuple[int, ...], ...] = tuple(
        tuple(x for x in range(5) if x not in p) for p in _SYM5_PAIR_LUT
    )
    _SYM5_TREES_META: list[tuple[int, ...]] = []
    for _sa, _sb, _sc, _sd, _se in _BALANCED_TREES_D5:
        _p_ab = _SYM5_PAIR_IDX[(_sa, _sb)]
        _p_cd = _SYM5_PAIR_IDX[(_sc, _sd)]
        _ra = _SYM5_PAIR_REM[_p_ab]
        _rc = _SYM5_PAIR_REM[_p_cd]
        _SYM5_TREES_META.append(
            (_p_ab, _p_cd, _se, _ra.index(_sc), _ra.index(_sd), _ra.index(_se), _rc.index(_se))
        )
    _SYM5_TREES_META.sort(key=lambda t: t[0])
    _SYM5_LUT_LOCK = threading.Lock()
    _SYM5_LUT_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_sym5_luts(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev_idx = int(device.index or 0)
        with _SYM5_LUT_LOCK:
            cached = _SYM5_LUT_CACHE.get(dev_idx)
            if cached is not None:
                return cached
            plut = torch.tensor(_SYM5_PAIR_LUT, dtype=torch.int32, device=device)
            prem = torch.tensor(_SYM5_PAIR_REM, dtype=torch.int32, device=device)
            tmeta = torch.tensor(_SYM5_TREES_META, dtype=torch.int32, device=device)
            _SYM5_LUT_CACHE[dev_idx] = (plut, prem, tmeta)
            return plut, prem, tmeta

    @triton.jit
    def _triton_normpdf32(x):  # type: ignore[no-untyped-def]
        return 0.3989422804014327 * tl.exp(-0.5 * x * x)

    @triton.jit  # type: ignore[untyped-decorator]
    def _sym5_pm_fwd0_32(f1, v1, f2, v2, rho, IS_BEST: tl.constexpr):  # type: ignore[no-untyped-def]
        rc = tl.minimum(tl.maximum(rho, -0.999999), 0.999999)
        var = tl.maximum(v1 * v1 + v2 * v2 - 2.0 * rc * v1 * v2, 1e-12)
        sig = tl.sqrt(var)
        d1 = (tl.log(f1 / f2) + 0.5 * var) / sig
        d2 = d1 - sig
        s_sign = 1.0 if IS_BEST else -1.0
        w1 = _triton_normcdf64(s_sign * d1)
        w2 = _triton_normcdf64(-s_sign * d2)
        em1 = tl.maximum(f1 * w1 + f2 * w2, 1e-12)
        f1s = tl.maximum(f1 * f1 * tl.exp(v1 * v1), 1e-24)
        f2s = tl.maximum(f2 * f2 * tl.exp(v2 * v2), 1e-24)
        sig2 = 2.0 * sig
        var2 = sig2 * sig2
        d1s = (tl.log(f1s / f2s) + 0.5 * var2) / sig2
        d2s = d1s - sig2
        em2 = f1s * _triton_normcdf64(s_sign * d1s) + f2s * _triton_normcdf64(-s_sign * d2s)
        r2 = tl.maximum(em2 / tl.maximum(em1 * em1, 1e-24), 1.0)
        vn = tl.sqrt(tl.maximum(tl.log(r2), 1e-12))
        return em1, vn, w1, w2, var, sig

    @triton.jit  # type: ignore[untyped-decorator]
    def _sym5_pm_rem_32(f1, v1, f2, v2, var, sig, em1, vn, vk, r1k, r2k, IS_BEST: tl.constexpr):  # type: ignore[no-untyped-def]
        f1k = tl.maximum(f1 * tl.exp(r1k * v1 * vk), 1e-12)
        f2k = tl.maximum(f2 * tl.exp(r2k * v2 * vk), 1e-12)
        d1k = (tl.log(f1k / f2k) + 0.5 * var) / sig
        d2k = d1k - sig
        s_sign = 1.0 if IS_BEST else -1.0
        empk = f1k * _triton_normcdf64(s_sign * d1k) + f2k * _triton_normcdf64(-s_sign * d2k)
        cov = tl.log(tl.maximum(empk / em1, 1e-12))
        return tl.minimum(tl.maximum(cov / tl.maximum(vn * vk, 1e-12), -0.999999), 0.999999)

    @triton.jit  # type: ignore[untyped-decorator]
    def _sym5_pm_bwd_core_32(  # type: ignore[no-untyped-def]
        f1, v1, f2, v2, rho,
        a_em1, a_vn, a_w1, a_w2,
        IS_BEST: tl.constexpr,
    ):
        rc = tl.minimum(tl.maximum(rho, -0.999999), 0.999999)
        var_raw = v1 * v1 + v2 * v2 - 2.0 * rc * v1 * v2
        var = tl.maximum(var_raw, 1e-12)
        sig = tl.sqrt(var)
        d1 = (tl.log(f1 / f2) + 0.5 * var) / sig
        d2 = d1 - sig
        s_sign = 1.0 if IS_BEST else -1.0
        w1 = _triton_normcdf64(s_sign * d1)
        w2 = _triton_normcdf64(-s_sign * d2)
        em1_raw = f1 * w1 + f2 * w2
        em1 = tl.maximum(em1_raw, 1e-12)
        f1s_raw = f1 * f1 * tl.exp(v1 * v1)
        f2s_raw = f2 * f2 * tl.exp(v2 * v2)
        f1s = tl.maximum(f1s_raw, 1e-24)
        f2s = tl.maximum(f2s_raw, 1e-24)
        sig2 = 2.0 * sig
        var2 = sig2 * sig2
        d1s = (tl.log(f1s / f2s) + 0.5 * var2) / sig2
        d2s = d1s - sig2
        w1s = _triton_normcdf64(s_sign * d1s)
        w2s = _triton_normcdf64(-s_sign * d2s)
        em2 = f1s * w1s + f2s * w2s
        em1_sq_raw = em1 * em1
        em1_sq = tl.maximum(em1_sq_raw, 1e-24)
        r2_raw = em2 / em1_sq
        r2 = tl.maximum(r2_raw, 1.0)
        lv = tl.log(r2)
        vn = tl.sqrt(tl.maximum(lv, 1e-12))

        a_em1_c = tl.where(em1_raw >= 1e-12, a_em1, 0.0)
        a_r2 = tl.where((lv >= 1e-12) & (r2_raw >= 1.0), a_vn / (2.0 * vn * r2), 0.0)
        a_em2 = a_r2 / em1_sq
        a_em1_c = a_em1_c - tl.where(em1_sq_raw >= 1e-24, 2.0 * a_r2 * r2_raw / em1, 0.0)

        a_f1s = tl.where(f1s_raw >= 1e-24, a_em2 * w1s, 0.0)
        a_f2s = tl.where(f2s_raw >= 1e-24, a_em2 * w2s, 0.0)
        p1s = _triton_normpdf32(d1s)
        p2s = _triton_normpdf32(d2s)
        g_d1s = s_sign * a_em2 * f1s * p1s
        g_d2s = -s_sign * a_em2 * f2s * p2s
        g_sig2 = g_d1s * (1.0 - d1s / sig2) - g_d2s * (d1s / sig2)
        g_lfr2 = (g_d1s + g_d2s) / sig2
        a_f1s = a_f1s + tl.where(f1s_raw >= 1e-24, g_lfr2 / f1s, 0.0)
        a_f2s = a_f2s - tl.where(f2s_raw >= 1e-24, g_lfr2 / f2s, 0.0)

        df1 = a_em1_c * w1 + 2.0 * a_f1s * f1s_raw / f1
        df2 = a_em1_c * w2 + 2.0 * a_f2s * f2s_raw / f2
        dv1 = 2.0 * a_f1s * f1s_raw * v1
        dv2 = 2.0 * a_f2s * f2s_raw * v2

        p1 = _triton_normpdf32(d1)
        p2 = _triton_normpdf32(d2)
        g_d1 = s_sign * (a_em1_c * f1 + a_w1) * p1
        g_d2 = -s_sign * (a_em1_c * f2 + a_w2) * p2
        g_sig = 2.0 * g_sig2 + g_d1 * (1.0 - d1 / sig) - g_d2 * (d1 / sig)
        g_lfr = (g_d1 + g_d2) / sig

        df1 = df1 + g_lfr / f1
        df2 = df2 - g_lfr / f2
        g_var = tl.where(var_raw >= 1e-12, g_sig / (2.0 * sig), 0.0)
        dv1 = dv1 + g_var * 2.0 * (v1 - rc * v2)
        dv2 = dv2 + g_var * 2.0 * (v2 - rc * v1)
        drho = tl.where((rho >= -0.999999) & (rho <= 0.999999), -2.0 * g_var * v1 * v2, 0.0)
        return df1, dv1, df2, dv2, drho

    @triton.jit  # type: ignore[untyped-decorator]
    def _sym5_pm_rem_bwd_32(  # type: ignore[no-untyped-def]
        f1, v1, f2, v2, rho, vk, r1k, r2k,
        var, sig, em1, vn, rnew_raw, a_rnew_c,
        IS_BEST: tl.constexpr,
    ):
        a_rnew = tl.where((rnew_raw > -0.999999) & (rnew_raw < 0.999999), a_rnew_c, 0.0)
        f1k_raw = f1 * tl.exp(r1k * v1 * vk)
        f2k_raw = f2 * tl.exp(r2k * v2 * vk)
        f1k = tl.maximum(f1k_raw, 1e-12)
        f2k = tl.maximum(f2k_raw, 1e-12)
        d1k = (tl.log(f1k / f2k) + 0.5 * var) / sig
        d2k = d1k - sig
        s_sign = 1.0 if IS_BEST else -1.0
        w1k = _triton_normcdf64(s_sign * d1k)
        w2k = _triton_normcdf64(-s_sign * d2k)
        empk = f1k * w1k + f2k * w2k
        ratio_k = empk / em1
        cov_k = tl.log(tl.maximum(ratio_k, 1e-12))
        vvk = vn * vk
        vvk_c = tl.maximum(vvk, 1e-12)

        a_cov = a_rnew / vvk_c
        a_vvk = tl.where(vvk >= 1e-12, -a_rnew * cov_k / (vvk_c * vvk_c), 0.0)
        a_vn = a_vvk * vk
        a_vk = a_vvk * vn
        a_empk = tl.where(ratio_k >= 1e-12, a_cov / empk, 0.0)
        a_em1 = -tl.where(ratio_k >= 1e-12, a_cov / em1, 0.0)

        p1k = _triton_normpdf32(d1k)
        p2k = _triton_normpdf32(d2k)
        g_d1k = s_sign * a_empk * f1k * p1k
        g_d2k = -s_sign * a_empk * f2k * p2k
        g_vark = (g_d1k - g_d2k) / (2.0 * sig) - (g_d1k * d1k + g_d2k * d2k) / (2.0 * var)
        g_lfrk = (g_d1k + g_d2k) / sig
        a_f1k = tl.where(f1k_raw >= 1e-12, a_empk * w1k + g_lfrk / f1k, 0.0)
        a_f2k = tl.where(f2k_raw >= 1e-12, a_empk * w2k - g_lfrk / f2k, 0.0)

        df1 = a_f1k * f1k_raw / f1
        df2 = a_f2k * f2k_raw / f2
        dv1 = a_f1k * f1k_raw * r1k * vk
        dv2 = a_f2k * f2k_raw * r2k * vk
        dr1k = a_f1k * f1k_raw * v1 * vk
        dr2k = a_f2k * f2k_raw * v2 * vk
        a_vk = a_vk + a_f1k * f1k_raw * r1k * v1 + a_f2k * f2k_raw * r2k * v2

        rc = tl.minimum(tl.maximum(rho, -0.999999), 0.999999)
        var_raw = v1 * v1 + v2 * v2 - 2.0 * rc * v1 * v2
        g_vk = tl.where(var_raw >= 1e-12, g_vark, 0.0)
        dv1 = dv1 + g_vk * 2.0 * (v1 - rc * v2)
        dv2 = dv2 + g_vk * 2.0 * (v2 - rc * v1)
        drho = tl.where((rho >= -0.999999) & (rho <= 0.999999), -2.0 * g_vk * v1 * v2, 0.0)
        return df1, dv1, df2, dv2, drho, a_em1, a_vn, a_vk, dr1k, dr2k

    @triton.jit  # type: ignore[untyped-decorator]
    def _sym5_pairs_fwd_kernel32(  # type: ignore[no-untyped-def]
        f_ptr, v_ptr, c_ptr, plut_ptr, prem_ptr, pbuf_ptr,
        N, IS_BEST: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = n < N
        SN = 10 * N
        for p in range(10):
            ia = tl.load(plut_ptr + p * 2 + 0)
            ib = tl.load(plut_ptr + p * 2 + 1)
            ik0 = tl.load(prem_ptr + p * 3 + 0)
            ik1 = tl.load(prem_ptr + p * 3 + 1)
            ik2 = tl.load(prem_ptr + p * 3 + 2)
            fa = tl.load(f_ptr + n * 5 + ia, mask=nmask, other=1.0)
            fb = tl.load(f_ptr + n * 5 + ib, mask=nmask, other=1.0)
            va = tl.load(v_ptr + n * 5 + ia, mask=nmask, other=0.1)
            vb = tl.load(v_ptr + n * 5 + ib, mask=nmask, other=0.1)
            cab = tl.load(c_ptr + (n * 5 + ia) * 5 + ib, mask=nmask, other=0.0)
            fab, vab, wa, wb, var, sig = _sym5_pm_fwd0_32(fa, va, fb, vb, cab, IS_BEST)
            vk0 = tl.load(v_ptr + n * 5 + ik0, mask=nmask, other=0.1)
            vk1 = tl.load(v_ptr + n * 5 + ik1, mask=nmask, other=0.1)
            vk2 = tl.load(v_ptr + n * 5 + ik2, mask=nmask, other=0.1)
            rak0 = tl.load(c_ptr + (n * 5 + ia) * 5 + ik0, mask=nmask, other=0.0)
            rbk0 = tl.load(c_ptr + (n * 5 + ib) * 5 + ik0, mask=nmask, other=0.0)
            rak1 = tl.load(c_ptr + (n * 5 + ia) * 5 + ik1, mask=nmask, other=0.0)
            rbk1 = tl.load(c_ptr + (n * 5 + ib) * 5 + ik1, mask=nmask, other=0.0)
            rak2 = tl.load(c_ptr + (n * 5 + ia) * 5 + ik2, mask=nmask, other=0.0)
            rbk2 = tl.load(c_ptr + (n * 5 + ib) * 5 + ik2, mask=nmask, other=0.0)
            rk0 = _sym5_pm_rem_32(fa, va, fb, vb, var, sig, fab, vab, vk0, rak0, rbk0, IS_BEST)
            rk1 = _sym5_pm_rem_32(fa, va, fb, vb, var, sig, fab, vab, vk1, rak1, rbk1, IS_BEST)
            rk2 = _sym5_pm_rem_32(fa, va, fb, vb, var, sig, fab, vab, vk2, rak2, rbk2, IS_BEST)
            base = p * N + n
            tl.store(pbuf_ptr + 0 * SN + base, fab, mask=nmask)
            tl.store(pbuf_ptr + 1 * SN + base, vab, mask=nmask)
            tl.store(pbuf_ptr + 2 * SN + base, wa, mask=nmask)
            tl.store(pbuf_ptr + 3 * SN + base, wb, mask=nmask)
            tl.store(pbuf_ptr + 4 * SN + base, rk0, mask=nmask)
            tl.store(pbuf_ptr + 5 * SN + base, rk1, mask=nmask)
            tl.store(pbuf_ptr + 6 * SN + base, rk2, mask=nmask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _sym5_trees_fwd_kernel32(  # type: ignore[no-untyped-def]
        f_ptr, v_ptr, c_ptr, pbuf_ptr, plut_ptr, tmeta_ptr,
        fsym_out, vsym_out, wsym_out, m2sym_out,
        N, IS_BEST: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = n < N
        f_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        m2_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        w0 = tl.zeros([BLOCK_N], dtype=tl.float32)
        w1 = tl.zeros([BLOCK_N], dtype=tl.float32)
        w2 = tl.zeros([BLOCK_N], dtype=tl.float32)
        w3 = tl.zeros([BLOCK_N], dtype=tl.float32)
        w4 = tl.zeros([BLOCK_N], dtype=tl.float32)
        SN = 10 * N

        for tr in range(30):
            p_ab = tl.load(tmeta_ptr + tr * 7 + 0)
            p_cd = tl.load(tmeta_ptr + tr * 7 + 1)
            ie = tl.load(tmeta_ptr + tr * 7 + 2)
            kc = tl.load(tmeta_ptr + tr * 7 + 3)
            kd = tl.load(tmeta_ptr + tr * 7 + 4)
            ke_ab = tl.load(tmeta_ptr + tr * 7 + 5)
            ke_cd = tl.load(tmeta_ptr + tr * 7 + 6)
            ia = tl.load(plut_ptr + p_ab * 2 + 0)
            ib = tl.load(plut_ptr + p_ab * 2 + 1)
            ic = tl.load(plut_ptr + p_cd * 2 + 0)
            id_ = tl.load(plut_ptr + p_cd * 2 + 1)

            fab = tl.load(pbuf_ptr + 0 * SN + p_ab * N + n, mask=nmask, other=1.0)
            vab = tl.load(pbuf_ptr + 1 * SN + p_ab * N + n, mask=nmask, other=0.1)
            wa = tl.load(pbuf_ptr + 2 * SN + p_ab * N + n, mask=nmask, other=0.5)
            wb = tl.load(pbuf_ptr + 3 * SN + p_ab * N + n, mask=nmask, other=0.5)
            rab_c = tl.load(pbuf_ptr + (4 + kc) * SN + p_ab * N + n, mask=nmask, other=0.0)
            rab_d = tl.load(pbuf_ptr + (4 + kd) * SN + p_ab * N + n, mask=nmask, other=0.0)
            rab_e = tl.load(pbuf_ptr + (4 + ke_ab) * SN + p_ab * N + n, mask=nmask, other=0.0)

            fc = tl.load(f_ptr + n * 5 + ic, mask=nmask, other=1.0)
            fd = tl.load(f_ptr + n * 5 + id_, mask=nmask, other=1.0)
            vc = tl.load(v_ptr + n * 5 + ic, mask=nmask, other=0.1)
            vd = tl.load(v_ptr + n * 5 + id_, mask=nmask, other=0.1)
            ccd = tl.load(c_ptr + (n * 5 + ic) * 5 + id_, mask=nmask, other=0.0)

            fcd = tl.load(pbuf_ptr + 0 * SN + p_cd * N + n, mask=nmask, other=1.0)
            vcd = tl.load(pbuf_ptr + 1 * SN + p_cd * N + n, mask=nmask, other=0.1)
            wc = tl.load(pbuf_ptr + 2 * SN + p_cd * N + n, mask=nmask, other=0.5)
            wd = tl.load(pbuf_ptr + 3 * SN + p_cd * N + n, mask=nmask, other=0.5)
            rcd_e = tl.load(pbuf_ptr + (4 + ke_cd) * SN + p_cd * N + n, mask=nmask, other=0.0)

            fe = tl.load(f_ptr + n * 5 + ie, mask=nmask, other=1.0)
            ve = tl.load(v_ptr + n * 5 + ie, mask=nmask, other=0.1)

            rc_cd = tl.minimum(tl.maximum(ccd, -0.999999), 0.999999)
            var_cd = tl.maximum(vc * vc + vd * vd - 2.0 * rc_cd * vc * vd, 1e-12)
            sig_cd = tl.sqrt(var_cd)
            rcd_ab = _sym5_pm_rem_32(fc, vc, fd, vd, var_cd, sig_cd, fcd, vcd, vab, rab_c, rab_d, IS_BEST)

            fabcd, vabcd, wab, wcd, var3, sig3 = _sym5_pm_fwd0_32(fab, vab, fcd, vcd, rcd_ab, IS_BEST)
            rabcd_e = _sym5_pm_rem_32(fab, vab, fcd, vcd, var3, sig3, fabcd, vabcd, ve, rab_e, rcd_e, IS_BEST)
            froot, vroot, wabcd, we, _, _ = _sym5_pm_fwd0_32(fabcd, vabcd, fe, ve, rabcd_e, IS_BEST)

            f_acc = f_acc + froot
            m2_acc = m2_acc + froot * froot * tl.exp(vroot * vroot)

            wa_f = wabcd * wab * wa
            wb_f = wabcd * wab * wb
            wc_f = wabcd * wcd * wc
            wd_f = wabcd * wcd * wd
            w0 = w0 + tl.where(ia == 0, wa_f, tl.where(ib == 0, wb_f, tl.where(ic == 0, wc_f, tl.where(id_ == 0, wd_f, we))))
            w1 = w1 + tl.where(ia == 1, wa_f, tl.where(ib == 1, wb_f, tl.where(ic == 1, wc_f, tl.where(id_ == 1, wd_f, we))))
            w2 = w2 + tl.where(ia == 2, wa_f, tl.where(ib == 2, wb_f, tl.where(ic == 2, wc_f, tl.where(id_ == 2, wd_f, we))))
            w3 = w3 + tl.where(ia == 3, wa_f, tl.where(ib == 3, wb_f, tl.where(ic == 3, wc_f, tl.where(id_ == 3, wd_f, we))))
            w4 = w4 + tl.where(ia == 4, wa_f, tl.where(ib == 4, wb_f, tl.where(ic == 4, wc_f, tl.where(id_ == 4, wd_f, we))))

        fsym = f_acc / 30.0
        m2sym = m2_acc / 30.0
        vsym = tl.sqrt(tl.maximum(tl.log(tl.maximum(m2sym / tl.maximum(fsym * fsym, 1e-24), 1.0)), 1e-12))
        tl.store(fsym_out + n, fsym, mask=nmask)
        tl.store(vsym_out + n, vsym, mask=nmask)
        tl.store(m2sym_out + n, m2sym, mask=nmask)
        tl.store(wsym_out + n * 5 + 0, w0 / 30.0, mask=nmask)
        tl.store(wsym_out + n * 5 + 1, w1 / 30.0, mask=nmask)
        tl.store(wsym_out + n * 5 + 2, w2 / 30.0, mask=nmask)
        tl.store(wsym_out + n * 5 + 3, w3 / 30.0, mask=nmask)
        tl.store(wsym_out + n * 5 + 4, w4 / 30.0, mask=nmask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _sym5_trees_bwd_kernel32(  # type: ignore[no-untyped-def]
        f_ptr, v_ptr, c_ptr, pbuf_ptr, plut_ptr, tmeta_ptr,
        fsym_ptr, vsym_ptr, m2sym_ptr,
        af_in, av_in, aw_in,
        gpbuf_out, gf_out, gv_out, gc_out,
        N, IS_BEST: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = n < N
        SN = 10 * N

        fsym = tl.load(fsym_ptr + n, mask=nmask, other=1.0)
        vsym = tl.load(vsym_ptr + n, mask=nmask, other=0.1)
        m2sym = tl.load(m2sym_ptr + n, mask=nmask, other=1.0)
        a_fsym = tl.load(af_in + n, mask=nmask, other=0.0)
        a_vsym = tl.load(av_in + n, mask=nmask, other=0.0)
        aw0 = tl.load(aw_in + n * 5 + 0, mask=nmask, other=0.0) / 30.0
        aw1 = tl.load(aw_in + n * 5 + 1, mask=nmask, other=0.0) / 30.0
        aw2 = tl.load(aw_in + n * 5 + 2, mask=nmask, other=0.0) / 30.0
        aw3 = tl.load(aw_in + n * 5 + 3, mask=nmask, other=0.0) / 30.0
        aw4 = tl.load(aw_in + n * 5 + 4, mask=nmask, other=0.0) / 30.0

        fs2_raw = fsym * fsym
        fs2 = tl.maximum(fs2_raw, 1e-24)
        r_sym_raw = m2sym / fs2
        r_sym = tl.maximum(r_sym_raw, 1.0)
        lv_sym = tl.log(r_sym)
        a_rsym = tl.where((lv_sym >= 1e-12) & (r_sym_raw >= 1.0), a_vsym / (2.0 * vsym * r_sym), 0.0)
        a_m2sym = (a_rsym / fs2) / 30.0
        a_fsym_tot = (a_fsym - tl.where(fs2_raw >= 1e-24, 2.0 * a_rsym * r_sym_raw / fsym, 0.0)) / 30.0

        for tr in range(30):
            p_ab = tl.load(tmeta_ptr + tr * 7 + 0)
            p_cd = tl.load(tmeta_ptr + tr * 7 + 1)
            ie = tl.load(tmeta_ptr + tr * 7 + 2)
            kc = tl.load(tmeta_ptr + tr * 7 + 3)
            kd = tl.load(tmeta_ptr + tr * 7 + 4)
            ke_ab = tl.load(tmeta_ptr + tr * 7 + 5)
            ke_cd = tl.load(tmeta_ptr + tr * 7 + 6)
            ia = tl.load(plut_ptr + p_ab * 2 + 0)
            ib = tl.load(plut_ptr + p_ab * 2 + 1)
            ic = tl.load(plut_ptr + p_cd * 2 + 0)
            id_ = tl.load(plut_ptr + p_cd * 2 + 1)

            fab = tl.load(pbuf_ptr + 0 * SN + p_ab * N + n, mask=nmask, other=1.0)
            vab = tl.load(pbuf_ptr + 1 * SN + p_ab * N + n, mask=nmask, other=0.1)
            wa = tl.load(pbuf_ptr + 2 * SN + p_ab * N + n, mask=nmask, other=0.5)
            wb = tl.load(pbuf_ptr + 3 * SN + p_ab * N + n, mask=nmask, other=0.5)
            rab_c = tl.load(pbuf_ptr + (4 + kc) * SN + p_ab * N + n, mask=nmask, other=0.0)
            rab_d = tl.load(pbuf_ptr + (4 + kd) * SN + p_ab * N + n, mask=nmask, other=0.0)
            rab_e = tl.load(pbuf_ptr + (4 + ke_ab) * SN + p_ab * N + n, mask=nmask, other=0.0)

            fc = tl.load(f_ptr + n * 5 + ic, mask=nmask, other=1.0)
            fd = tl.load(f_ptr + n * 5 + id_, mask=nmask, other=1.0)
            vc = tl.load(v_ptr + n * 5 + ic, mask=nmask, other=0.1)
            vd = tl.load(v_ptr + n * 5 + id_, mask=nmask, other=0.1)
            ccd = tl.load(c_ptr + (n * 5 + ic) * 5 + id_, mask=nmask, other=0.0)

            fcd = tl.load(pbuf_ptr + 0 * SN + p_cd * N + n, mask=nmask, other=1.0)
            vcd = tl.load(pbuf_ptr + 1 * SN + p_cd * N + n, mask=nmask, other=0.1)
            wc = tl.load(pbuf_ptr + 2 * SN + p_cd * N + n, mask=nmask, other=0.5)
            wd = tl.load(pbuf_ptr + 3 * SN + p_cd * N + n, mask=nmask, other=0.5)
            rcd_e = tl.load(pbuf_ptr + (4 + ke_cd) * SN + p_cd * N + n, mask=nmask, other=0.0)

            fe = tl.load(f_ptr + n * 5 + ie, mask=nmask, other=1.0)
            ve = tl.load(v_ptr + n * 5 + ie, mask=nmask, other=0.1)

            rc_cd = tl.minimum(tl.maximum(ccd, -0.999999), 0.999999)
            var_cd = tl.maximum(vc * vc + vd * vd - 2.0 * rc_cd * vc * vd, 1e-12)
            sig_cd = tl.sqrt(var_cd)
            rcd_ab = _sym5_pm_rem_32(fc, vc, fd, vd, var_cd, sig_cd, fcd, vcd, vab, rab_c, rab_d, IS_BEST)

            fabcd, vabcd, wab, wcd, var3, sig3 = _sym5_pm_fwd0_32(fab, vab, fcd, vcd, rcd_ab, IS_BEST)
            rabcd_e = _sym5_pm_rem_32(fab, vab, fcd, vcd, var3, sig3, fabcd, vabcd, ve, rab_e, rcd_e, IS_BEST)
            froot, vroot, wabcd, we, _, _ = _sym5_pm_fwd0_32(fabcd, vabcd, fe, ve, rabcd_e, IS_BEST)

            aw_a_l = tl.where(ia == 0, aw0, tl.where(ia == 1, aw1, tl.where(ia == 2, aw2, tl.where(ia == 3, aw3, aw4))))
            aw_b_l = tl.where(ib == 0, aw0, tl.where(ib == 1, aw1, tl.where(ib == 2, aw2, tl.where(ib == 3, aw3, aw4))))
            aw_c_l = tl.where(ic == 0, aw0, tl.where(ic == 1, aw1, tl.where(ic == 2, aw2, tl.where(ic == 3, aw3, aw4))))
            aw_d_l = tl.where(id_ == 0, aw0, tl.where(id_ == 1, aw1, tl.where(id_ == 2, aw2, tl.where(id_ == 3, aw3, aw4))))
            aw_e_l = tl.where(ie == 0, aw0, tl.where(ie == 1, aw1, tl.where(ie == 2, aw2, tl.where(ie == 3, aw3, aw4))))

            a_wabcd = aw_a_l * wab * wa + aw_b_l * wab * wb + aw_c_l * wcd * wc + aw_d_l * wcd * wd
            a_we = aw_e_l
            a_wab = wabcd * (aw_a_l * wa + aw_b_l * wb)
            a_wcd = wabcd * (aw_c_l * wc + aw_d_l * wd)
            g_wa_add = wabcd * wab * aw_a_l
            g_wb_add = wabcd * wab * aw_b_l
            g_wc_add = wabcd * wcd * aw_c_l
            g_wd_add = wabcd * wcd * aw_d_l

            ev2 = tl.exp(vroot * vroot)
            a_froot = a_fsym_tot + a_m2sym * 2.0 * froot * ev2
            a_vroot = a_m2sym * froot * froot * ev2 * 2.0 * vroot

            a_fabcd, a_vabcd, a_fe, a_ve, a_rabcd_e = _sym5_pm_bwd_core_32(
                fabcd, vabcd, fe, ve, rabcd_e, a_froot, a_vroot, a_wabcd, a_we, IS_BEST
            )
            df_1r, dv_1r, df_2r, dv_2r, dr_3r, a_fabcd_r, a_vabcd_r, ave_r, a_rabe, a_rcde = _sym5_pm_rem_bwd_32(
                fab, vab, fcd, vcd, rcd_ab, ve, rab_e, rcd_e, var3, sig3, fabcd, vabcd, rabcd_e, a_rabcd_e, IS_BEST
            )
            a_fabcd = a_fabcd + a_fabcd_r
            a_vabcd = a_vabcd + a_vabcd_r
            a_ve = a_ve + ave_r

            a_fab3, a_vab3, a_fcd3, a_vcd3, a_rcd_ab = _sym5_pm_bwd_core_32(
                fab, vab, fcd, vcd, rcd_ab, a_fabcd, a_vabcd, a_wab, a_wcd, IS_BEST
            )
            a_fab = df_1r + a_fab3
            a_vab = dv_1r + a_vab3
            a_fcd = df_2r + a_fcd3
            a_vcd = dv_2r + a_vcd3
            a_rcd_ab = a_rcd_ab + dr_3r

            df_c2, dv_c2, df_d2, dv_d2, d_ccd, a_fcd_r, a_vcd_r, a_vab_r, a_rabc, a_rabd = _sym5_pm_rem_bwd_32(
                fc, vc, fd, vd, ccd, vab, rab_c, rab_d, var_cd, sig_cd, fcd, vcd, rcd_ab, a_rcd_ab, IS_BEST
            )
            a_fcd = a_fcd + a_fcd_r
            a_vcd = a_vcd + a_vcd_r
            a_vab = a_vab + a_vab_r

            tl.store(gpbuf_out + (0 * 10 + p_ab) * N + n, tl.load(gpbuf_out + (0 * 10 + p_ab) * N + n, mask=nmask) + a_fab, mask=nmask)
            tl.store(gpbuf_out + (1 * 10 + p_ab) * N + n, tl.load(gpbuf_out + (1 * 10 + p_ab) * N + n, mask=nmask) + a_vab, mask=nmask)
            tl.store(gpbuf_out + (2 * 10 + p_ab) * N + n, tl.load(gpbuf_out + (2 * 10 + p_ab) * N + n, mask=nmask) + g_wa_add, mask=nmask)
            tl.store(gpbuf_out + (3 * 10 + p_ab) * N + n, tl.load(gpbuf_out + (3 * 10 + p_ab) * N + n, mask=nmask) + g_wb_add, mask=nmask)
            tl.store(gpbuf_out + ((4 + kc) * 10 + p_ab) * N + n, tl.load(gpbuf_out + ((4 + kc) * 10 + p_ab) * N + n, mask=nmask) + a_rabc, mask=nmask)
            tl.store(gpbuf_out + ((4 + kd) * 10 + p_ab) * N + n, tl.load(gpbuf_out + ((4 + kd) * 10 + p_ab) * N + n, mask=nmask) + a_rabd, mask=nmask)
            tl.store(gpbuf_out + ((4 + ke_ab) * 10 + p_ab) * N + n, tl.load(gpbuf_out + ((4 + ke_ab) * 10 + p_ab) * N + n, mask=nmask) + a_rabe, mask=nmask)

            tl.store(gpbuf_out + (0 * 10 + p_cd) * N + n, tl.load(gpbuf_out + (0 * 10 + p_cd) * N + n, mask=nmask) + a_fcd, mask=nmask)
            tl.store(gpbuf_out + (1 * 10 + p_cd) * N + n, tl.load(gpbuf_out + (1 * 10 + p_cd) * N + n, mask=nmask) + a_vcd, mask=nmask)
            tl.store(gpbuf_out + (2 * 10 + p_cd) * N + n, tl.load(gpbuf_out + (2 * 10 + p_cd) * N + n, mask=nmask) + g_wc_add, mask=nmask)
            tl.store(gpbuf_out + (3 * 10 + p_cd) * N + n, tl.load(gpbuf_out + (3 * 10 + p_cd) * N + n, mask=nmask) + g_wd_add, mask=nmask)
            tl.store(gpbuf_out + ((4 + ke_cd) * 10 + p_cd) * N + n, tl.load(gpbuf_out + ((4 + ke_cd) * 10 + p_cd) * N + n, mask=nmask) + a_rcde, mask=nmask)

            tl.store(gf_out + n * 5 + ic, tl.load(gf_out + n * 5 + ic, mask=nmask) + df_c2, mask=nmask)
            tl.store(gf_out + n * 5 + id_, tl.load(gf_out + n * 5 + id_, mask=nmask) + df_d2, mask=nmask)
            tl.store(gf_out + n * 5 + ie, tl.load(gf_out + n * 5 + ie, mask=nmask) + a_fe, mask=nmask)
            tl.store(gv_out + n * 5 + ic, tl.load(gv_out + n * 5 + ic, mask=nmask) + dv_c2, mask=nmask)
            tl.store(gv_out + n * 5 + id_, tl.load(gv_out + n * 5 + id_, mask=nmask) + dv_d2, mask=nmask)
            tl.store(gv_out + n * 5 + ie, tl.load(gv_out + n * 5 + ie, mask=nmask) + a_ve, mask=nmask)
            tl.store(gc_out + (n * 5 + ic) * 5 + id_, tl.load(gc_out + (n * 5 + ic) * 5 + id_, mask=nmask) + d_ccd, mask=nmask)

    @triton.jit  # type: ignore[untyped-decorator]
    def _sym5_pairs_bwd_kernel32(  # type: ignore[no-untyped-def]
        f_ptr, v_ptr, c_ptr, plut_ptr, prem_ptr, pbuf_ptr, gpbuf_ptr,
        gf_out, gv_out, gc_out,
        N, IS_BEST: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = n < N
        SN = 10 * N
        for p in range(10):
            ia = tl.load(plut_ptr + p * 2 + 0)
            ib = tl.load(plut_ptr + p * 2 + 1)
            ik0 = tl.load(prem_ptr + p * 3 + 0)
            ik1 = tl.load(prem_ptr + p * 3 + 1)
            ik2 = tl.load(prem_ptr + p * 3 + 2)
            fa = tl.load(f_ptr + n * 5 + ia, mask=nmask, other=1.0)
            fb = tl.load(f_ptr + n * 5 + ib, mask=nmask, other=1.0)
            va = tl.load(v_ptr + n * 5 + ia, mask=nmask, other=0.1)
            vb = tl.load(v_ptr + n * 5 + ib, mask=nmask, other=0.1)
            cab = tl.load(c_ptr + (n * 5 + ia) * 5 + ib, mask=nmask, other=0.0)
            rc_ab = tl.minimum(tl.maximum(cab, -0.999999), 0.999999)
            var = tl.maximum(va * va + vb * vb - 2.0 * rc_ab * va * vb, 1e-12)
            sig = tl.sqrt(var)

            base = p * N + n
            fab = tl.load(pbuf_ptr + 0 * SN + base, mask=nmask, other=1.0)
            vab = tl.load(pbuf_ptr + 1 * SN + base, mask=nmask, other=0.1)
            rk0 = tl.load(pbuf_ptr + 4 * SN + base, mask=nmask, other=0.0)
            rk1 = tl.load(pbuf_ptr + 5 * SN + base, mask=nmask, other=0.0)
            rk2 = tl.load(pbuf_ptr + 6 * SN + base, mask=nmask, other=0.0)

            a_fab = tl.load(gpbuf_ptr + 0 * SN + base, mask=nmask, other=0.0)
            a_vab = tl.load(gpbuf_ptr + 1 * SN + base, mask=nmask, other=0.0)
            a_wa = tl.load(gpbuf_ptr + 2 * SN + base, mask=nmask, other=0.0)
            a_wb = tl.load(gpbuf_ptr + 3 * SN + base, mask=nmask, other=0.0)
            a_rk0 = tl.load(gpbuf_ptr + 4 * SN + base, mask=nmask, other=0.0)
            a_rk1 = tl.load(gpbuf_ptr + 5 * SN + base, mask=nmask, other=0.0)
            a_rk2 = tl.load(gpbuf_ptr + 6 * SN + base, mask=nmask, other=0.0)

            dfa = tl.zeros([BLOCK_N], dtype=tl.float32)
            dfb = tl.zeros([BLOCK_N], dtype=tl.float32)
            dva = tl.zeros([BLOCK_N], dtype=tl.float32)
            dvb = tl.zeros([BLOCK_N], dtype=tl.float32)
            dcab = tl.zeros([BLOCK_N], dtype=tl.float32)

            vk0 = tl.load(v_ptr + n * 5 + ik0, mask=nmask, other=0.1)
            rak0 = tl.load(c_ptr + (n * 5 + ia) * 5 + ik0, mask=nmask, other=0.0)
            rbk0 = tl.load(c_ptr + (n * 5 + ib) * 5 + ik0, mask=nmask, other=0.0)
            d1, dv1, d2, dv2, drh, aem, avn, avk0, drak0, drbk0 = _sym5_pm_rem_bwd_32(
                fa, va, fb, vb, cab, vk0, rak0, rbk0, var, sig, fab, vab, rk0, a_rk0, IS_BEST
            )
            dfa += d1
            dfb += d2
            dva += dv1
            dvb += dv2
            dcab += drh
            a_fab += aem
            a_vab += avn
            tl.store(gv_out + n * 5 + ik0, tl.load(gv_out + n * 5 + ik0, mask=nmask) + avk0, mask=nmask)
            tl.store(gc_out + (n * 5 + ia) * 5 + ik0, tl.load(gc_out + (n * 5 + ia) * 5 + ik0, mask=nmask) + drak0, mask=nmask)
            tl.store(gc_out + (n * 5 + ib) * 5 + ik0, tl.load(gc_out + (n * 5 + ib) * 5 + ik0, mask=nmask) + drbk0, mask=nmask)

            vk1 = tl.load(v_ptr + n * 5 + ik1, mask=nmask, other=0.1)
            rak1 = tl.load(c_ptr + (n * 5 + ia) * 5 + ik1, mask=nmask, other=0.0)
            rbk1 = tl.load(c_ptr + (n * 5 + ib) * 5 + ik1, mask=nmask, other=0.0)
            d1, dv1, d2, dv2, drh, aem, avn, avk1, drak1, drbk1 = _sym5_pm_rem_bwd_32(
                fa, va, fb, vb, cab, vk1, rak1, rbk1, var, sig, fab, vab, rk1, a_rk1, IS_BEST
            )
            dfa += d1
            dfb += d2
            dva += dv1
            dvb += dv2
            dcab += drh
            a_fab += aem
            a_vab += avn
            tl.store(gv_out + n * 5 + ik1, tl.load(gv_out + n * 5 + ik1, mask=nmask) + avk1, mask=nmask)
            tl.store(gc_out + (n * 5 + ia) * 5 + ik1, tl.load(gc_out + (n * 5 + ia) * 5 + ik1, mask=nmask) + drak1, mask=nmask)
            tl.store(gc_out + (n * 5 + ib) * 5 + ik1, tl.load(gc_out + (n * 5 + ib) * 5 + ik1, mask=nmask) + drbk1, mask=nmask)

            vk2 = tl.load(v_ptr + n * 5 + ik2, mask=nmask, other=0.1)
            rak2 = tl.load(c_ptr + (n * 5 + ia) * 5 + ik2, mask=nmask, other=0.0)
            rbk2 = tl.load(c_ptr + (n * 5 + ib) * 5 + ik2, mask=nmask, other=0.0)
            d1, dv1, d2, dv2, drh, aem, avn, avk2, drak2, drbk2 = _sym5_pm_rem_bwd_32(
                fa, va, fb, vb, cab, vk2, rak2, rbk2, var, sig, fab, vab, rk2, a_rk2, IS_BEST
            )
            dfa += d1
            dfb += d2
            dva += dv1
            dvb += dv2
            dcab += drh
            a_fab += aem
            a_vab += avn
            tl.store(gv_out + n * 5 + ik2, tl.load(gv_out + n * 5 + ik2, mask=nmask) + avk2, mask=nmask)
            tl.store(gc_out + (n * 5 + ia) * 5 + ik2, tl.load(gc_out + (n * 5 + ia) * 5 + ik2, mask=nmask) + drak2, mask=nmask)
            tl.store(gc_out + (n * 5 + ib) * 5 + ik2, tl.load(gc_out + (n * 5 + ib) * 5 + ik2, mask=nmask) + drbk2, mask=nmask)

            df1, dv1, df2, dv2, drh = _sym5_pm_bwd_core_32(fa, va, fb, vb, cab, a_fab, a_vab, a_wa, a_wb, IS_BEST)
            tl.store(gf_out + n * 5 + ia, tl.load(gf_out + n * 5 + ia, mask=nmask) + dfa + df1, mask=nmask)
            tl.store(gf_out + n * 5 + ib, tl.load(gf_out + n * 5 + ib, mask=nmask) + dfb + df2, mask=nmask)
            tl.store(gv_out + n * 5 + ia, tl.load(gv_out + n * 5 + ia, mask=nmask) + dva + dv1, mask=nmask)
            tl.store(gv_out + n * 5 + ib, tl.load(gv_out + n * 5 + ib, mask=nmask) + dvb + dv2, mask=nmask)
            tl.store(gc_out + (n * 5 + ia) * 5 + ib, tl.load(gc_out + (n * 5 + ia) * 5 + ib, mask=nmask) + dcab + drh, mask=nmask)

    def _sym5_gsep(f_t: torch.Tensor) -> torch.Tensor:
        f_log = torch.log(f_t.clamp(min=1e-16))
        dl = torch.cat(
            [
                f_log[:, 0:1] - f_log[:, 1:5],
                f_log[:, 1:2] - f_log[:, 2:5],
                f_log[:, 2:3] - f_log[:, 3:5],
                f_log[:, 3:4] - f_log[:, 4:5],
            ],
            dim=1,
        )
        out: torch.Tensor = (1.0 - torch.exp(-7812.5 * (dl * dl))).square().prod(dim=1)
        return out

    def _sym5_gsep_vjp(f_t: torch.Tensor, a_gsep: torch.Tensor) -> torch.Tensor:
        in_c = f_t >= 1e-16
        f_log = torch.log(f_t.clamp(min=1e-16))
        dl = torch.cat(
            [
                f_log[:, 0:1] - f_log[:, 1:5],
                f_log[:, 1:2] - f_log[:, 2:5],
                f_log[:, 2:3] - f_log[:, 3:5],
                f_log[:, 3:4] - f_log[:, 4:5],
            ],
            dim=1,
        )
        u = torch.exp(-7812.5 * (dl * dl))
        one_m_u = 1.0 - u
        phi = one_m_u * one_m_u
        eye_10 = torch.eye(10, dtype=f_t.dtype, device=f_t.device).unsqueeze(0)
        prod_excl = torch.where(eye_10 > 0.5, 1.0, phi.unsqueeze(1)).prod(dim=2)
        d_dl = 31250.0 * dl * u * one_m_u * prod_excl * a_gsep.unsqueeze(1)
        g_log = torch.stack(
            [
                d_dl[:, 0:4].sum(dim=1),
                -d_dl[:, 0] + d_dl[:, 4:7].sum(dim=1),
                -d_dl[:, 1] - d_dl[:, 4] + d_dl[:, 7:9].sum(dim=1),
                -d_dl[:, 2] - d_dl[:, 5] - d_dl[:, 7] + d_dl[:, 9],
                -d_dl[:, 3] - d_dl[:, 6] - d_dl[:, 8] - d_dl[:, 9],
            ],
            dim=1,
        )
        return torch.where(in_c, g_log / f_t.clamp(min=1e-16), 0.0)

    def _eval_sym5_fwd_triton(
        f_t: torch.Tensor,
        v_t: torch.Tensor,
        c_t: torch.Tensor,
        *,
        is_best_of: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_rows = int(f_t.shape[0])
        dev = f_t.device
        plut, prem, tmeta = _get_sym5_luts(dev)
        # Scale forwards by row mean before float32 execution to prevent 1:64 FP64 slowdown while preserving CashGamma differences
        f_scale = f_t.mean(dim=1, keepdim=True)
        u32 = (f_t / f_scale).float().contiguous()
        v32 = v_t.float().contiguous()
        c32 = c_t.float().contiguous()
        pbuf = torch.empty((7, 10, n_rows), dtype=torch.float32, device=dev)
        fsym32 = torch.empty((n_rows,), dtype=torch.float32, device=dev)
        vsym32 = torch.empty((n_rows,), dtype=torch.float32, device=dev)
        m2sym32 = torch.empty((n_rows,), dtype=torch.float32, device=dev)
        wsym32 = torch.empty((n_rows, 5), dtype=torch.float32, device=dev)
        grid = (triton.cdiv(n_rows, 64),)
        _sym5_pairs_fwd_kernel32[grid](u32, v32, c32, plut, prem, pbuf, n_rows, IS_BEST=is_best_of, BLOCK_N=64, num_warps=2)
        _sym5_trees_fwd_kernel32[grid](u32, v32, c32, pbuf, plut, tmeta, fsym32, vsym32, wsym32, m2sym32, n_rows, IS_BEST=is_best_of, BLOCK_N=64, num_warps=2)
        return f_scale.squeeze(1) * fsym32.double(), vsym32.double(), wsym32.double(), pbuf, m2sym32

    def _eval_sym5_bwd_triton(
        f_t: torch.Tensor,
        v_t: torch.Tensor,
        c_t: torch.Tensor,
        pbuf: torch.Tensor,
        fsym: torch.Tensor,
        vsym: torch.Tensor,
        m2sym: torch.Tensor,
        af: torch.Tensor,
        av: torch.Tensor,
        aw: torch.Tensor,
        *,
        is_best_of: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_rows = int(f_t.shape[0])
        dev = f_t.device
        plut, prem, tmeta = _get_sym5_luts(dev)
        f_scale = f_t.mean(dim=1, keepdim=True)
        f_scale_1d = f_scale.squeeze(1)
        u64 = f_t / f_scale
        u32 = u64.float().contiguous()
        v32 = v_t.float().contiguous()
        c32 = c_t.float().contiguous()
        fsym32 = (fsym / f_scale_1d).float().contiguous()
        vsym32 = vsym.float().contiguous()
        af32 = (af * f_scale_1d).float().contiguous()
        av32 = av.float().contiguous()
        aw32 = aw.float().contiguous()
        gpbuf = torch.zeros((7, 10, n_rows), dtype=torch.float32, device=dev)
        gf32 = torch.zeros_like(u32)
        gv32 = torch.zeros_like(v32)
        gc32 = torch.zeros_like(c32)
        grid = (triton.cdiv(n_rows, 64),)
        _sym5_trees_bwd_kernel32[grid](
            u32, v32, c32, pbuf, plut, tmeta, fsym32, vsym32, m2sym,
            af32, av32, aw32, gpbuf, gf32, gv32, gc32,
            n_rows, IS_BEST=is_best_of, BLOCK_N=64, num_warps=2,
        )
        _sym5_pairs_bwd_kernel32[grid](
            u32, v32, c32, plut, prem, pbuf, gpbuf, gf32, gv32, gc32,
            n_rows, IS_BEST=is_best_of, BLOCK_N=64, num_warps=2,
        )
        gu64 = gf32.double()
        gf_out = (gu64 + (af * fsym - (gu64 * u64).sum(dim=1)).unsqueeze(1) / 5.0) / f_scale
        return gf_out, gv32.double(), gc32.double()


def _clark_tie_mask_bm(fp_bm: torch.Tensor, amask_64: torch.Tensor) -> torch.Tensor | None:
    """Returns per-(b,m) tie mask where pairwise |ln(f_i/f_j)| triggers _TIE_TAU symmetrization."""
    b_sz, d_act, m_act = fp_bm.shape
    if d_act <= 1:
        return None
    fp_t = fp_bm.detach().transpose(1, 2).reshape(b_sz * m_act, d_act).clamp(min=1e-16)
    am_t = (amask_64 > 0.5).unsqueeze(1).expand(b_sz, m_act, d_act).reshape(b_sz * m_act, d_act)
    pair_off = (am_t.unsqueeze(2) & am_t.unsqueeze(1)) & (~torch.eye(d_act, dtype=torch.bool, device=fp_bm.device).unsqueeze(0))
    dl_sq = (torch.log(fp_t.unsqueeze(2) / fp_t.unsqueeze(1))) ** 2
    # Threshold matches w_tie > 1e-12 for _TIE_TAU=0.008: dl^2 < -2 * 0.008^2 * ln(1e-12)
    tie_bm = ((dl_sq < 0.0035367) & pair_off).any(dim=(-2, -1))
    return tie_bm if bool(tie_bm.any().item()) else None


def _eval_tier1_fwd_triton_f64(
    inputs: tuple[torch.Tensor, ...],
    *,
    is_best_of: bool,
    all_sched_act: bool,
    tie_bm: torch.Tensor | None = None,
    sym5_cache: dict[str, Any] | None = None,
    force_sym5_only: bool = False,
    force_sym5_blend_all: bool = False,
    use_clark32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-only Triton Tier-1 proxies (float64 for D <= 5, f_scale float32 for 6 <= D <= 8)."""
    (
        fp_in, v_in, corr_in, am_in, sm_in, b_ko_in, b_cpn_in, is_c_in, is_g_in,
        ko_sp_in, cpn_sp_in, ki_sp_in, k_strike_in, k_low_in, active_ki_in, b_ki_in,
        force_cpn_in, idx_last_in,
    ) = inputs
    b_sz, d_act, m_act = fp_in.shape
    dev = fp_in.device
    bm = b_sz * m_act
    use_dyn32 = (d_act > 5) or (use_clark32 and d_act == 5)

    fp_flat = fp_in.transpose(1, 2).reshape(bm, d_act)
    vp_flat = v_in.transpose(1, 2).reshape(bm, d_act)
    am_flat = am_in.unsqueeze(1).expand(b_sz, m_act, d_act).reshape(bm, d_act)
    act_b = am_flat > 0.5
    f_safe = torch.where(act_b, fp_flat.clamp(min=1e-12), 1.0)
    v_safe = torch.where(act_b, vp_flat.clamp(min=1e-6), 0.1)

    c_safe: torch.Tensor | None = None
    tie5_idx: torch.Tensor | None = None
    tie_dyn_idx: torch.Tensor | None = None
    tie_rem_idx: torch.Tensor | None = None
    g_sep5: torch.Tensor | None = None
    g_sep5_max = 1.0
    g_sep_dyn: torch.Tensor | None = None
    g_sep_dyn_max = 1.0
    use_sym5_only = False
    use_sym_dyn_only = False

    if d_act > 1:
        pair_act = (am_in > 0.5).unsqueeze(2) & (am_in > 0.5).unsqueeze(1)
        eye_d = torch.eye(d_act, dtype=torch.float64, device=dev).unsqueeze(0)
        c_safe_b = torch.where(pair_act, corr_in, eye_d)
        c_safe = c_safe_b.unsqueeze(1).expand(b_sz, m_act, d_act, d_act).reshape(bm, d_act, d_act)
        if force_sym5_only and d_act == 5:
            use_sym5_only = True
        elif force_sym5_blend_all and d_act == 5:
            g_sep5 = _sym5_gsep(f_safe.contiguous())
        elif tie_bm is not None:
            tie_idx = tie_bm.nonzero(as_tuple=True)[0]
            if d_act == 5:
                is_act5 = act_b.index_select(0, tie_idx).all(dim=1)
                if bool(is_act5.all().item()):
                    tie5_idx = tie_idx
                else:
                    tie5_idx = tie_idx[is_act5]
                    tie_rem_idx = tie_idx[~is_act5]
                    if int(tie5_idx.numel()) == 0:
                        tie5_idx = None
                    if int(tie_rem_idx.numel()) == 0:
                        tie_rem_idx = None
            elif d_act in (2, 3, 6, 7, 8):
                is_full = act_b.index_select(0, tie_idx).all(dim=1)
                if bool(is_full.all().item()):
                    tie_dyn_idx = tie_idx
                else:
                    tie_dyn_idx = tie_idx[is_full]
                    tie_rem_idx = tie_idx[~is_full]
                    if int(tie_dyn_idx.numel()) == 0:
                        tie_dyn_idx = None
                    if int(tie_rem_idx.numel()) == 0:
                        tie_rem_idx = None
            else:
                tie_rem_idx = tie_idx
            if tie5_idx is not None:
                f_t5 = f_safe.index_select(0, tie5_idx)
                g_sep5 = _sym5_gsep(f_t5)
                g_sep5_max = float(g_sep5.max().item())
                if int(tie5_idx.numel()) == bm and g_sep5_max < 1e-15:
                    use_sym5_only = True
            if tie_dyn_idx is not None:
                f_td = f_safe.index_select(0, tie_dyn_idx)
                g_sep_dyn = _sym_gsep_dyn(f_td, act_b.index_select(0, tie_dyn_idx).double())
                g_sep_dyn_max = float(g_sep_dyn.max().item())
                if int(tie_dyn_idx.numel()) == bm and g_sep_dyn_max < 1e-15:
                    use_sym_dyn_only = True

    if use_sym5_only and c_safe is not None:
        v_t5 = v_safe.contiguous()
        f_sym5, v_sym5, w_sym5, pbuf5, m2sym5 = _eval_sym5_fwd_triton(
            f_safe.contiguous(), v_t5, c_safe.contiguous(), is_best_of=is_best_of
        )
        fw_flat = f_sym5
        vw_flat = v_sym5
        w_unsort = w_sym5
        if sym5_cache is not None:
            sym5_cache["use_sym5_only"] = True
            sym5_cache["tie5_idx"] = tie5_idx
            sym5_cache["fwd"] = (f_safe.contiguous(), v_t5, c_safe.contiguous(), f_sym5, v_sym5, w_sym5, pbuf5, m2sym5, g_sep5, g_sep5_max)
    elif use_sym_dyn_only and c_safe is not None:
        fw_flat, vw_flat, w_unsort = _eval_sym_dyn32_fwd(
            f_safe.contiguous(), v_safe.contiguous(), c_safe.contiguous(), am_flat.contiguous(), is_best_of=is_best_of
        )
    else:
        if d_act > 1:
            assert c_safe is not None
            sort_key = torch.where(act_b, -f_safe if is_best_of else f_safe, 1e12)
            ord_f = torch.argsort(sort_key, dim=1, stable=True)
            inv_o = torch.argsort(ord_f, dim=1)
            f_o = torch.gather(f_safe, 1, ord_f).contiguous()
            v_o = torch.gather(v_safe, 1, ord_f).contiguous()
            m_o = torch.gather(am_flat, 1, ord_f).contiguous()
            c_o = torch.gather(
                torch.gather(c_safe, 1, ord_f.unsqueeze(2).expand(bm, d_act, d_act)),
                2,
                ord_f.unsqueeze(1).expand(bm, d_act, d_act),
            ).contiguous()
        else:
            f_o = f_safe.contiguous()
            v_o = v_safe.contiguous()
            m_o = am_flat.contiguous()
            c_o = f_o
            inv_o = None

        if use_dyn32:
            fw_flat, vw_flat, w_o_flat, _ = _eval_clark32_fwd_triton(
                f_o, v_o, c_o, m_o, is_best_of=is_best_of, save_sc=False
            )
        else:
            sc_rho = torch.empty((d_act, bm), dtype=torch.float64, device=dev) if d_act > 1 else f_o
            sc_dummy = f_o
            fw_flat = torch.empty(bm, dtype=torch.float64, device=dev)
            vw_flat = torch.empty(bm, dtype=torch.float64, device=dev)
            w_o_flat = torch.empty((bm, d_act), dtype=torch.float64, device=dev)
            grid_bm = (triton.cdiv(bm, 32),)
            _clark_fwd_kernel64[grid_bm](
                f_o, v_o, c_o, m_o, fw_flat, vw_flat, w_o_flat, sc_dummy, sc_dummy, sc_rho,
                bm, D=d_act, IS_BEST=is_best_of, SAVE_SC=False, BLOCK_BM=32, num_warps=1,
            )
        w_unsort = torch.gather(w_o_flat, 1, inv_o) if inv_o is not None else w_o_flat

        if force_sym5_blend_all and c_safe is not None and g_sep5 is not None:
            f_t5 = f_safe.contiguous()
            v_t5 = v_safe.contiguous()
            c_t5 = c_safe.contiguous()
            f_sym5, v_sym5, w_sym5, _, _ = _eval_sym5_fwd_triton(
                f_t5, v_t5, c_t5, is_best_of=is_best_of
            )
            w_tie5 = 1.0 - g_sep5
            fw_flat = g_sep5 * fw_flat + w_tie5 * f_sym5
            vw_flat = g_sep5 * vw_flat + w_tie5 * v_sym5
            w_unsort = g_sep5.unsqueeze(1) * w_unsort + w_tie5.unsqueeze(1) * w_sym5
        elif tie5_idx is not None and c_safe is not None:
            assert g_sep5 is not None
            f_t5 = f_safe.index_select(0, tie5_idx).contiguous()
            v_t5 = v_safe.index_select(0, tie5_idx).contiguous()
            c_t5 = c_safe.index_select(0, tie5_idx).contiguous()
            f_sym5, v_sym5, w_sym5, pbuf5, m2sym5 = _eval_sym5_fwd_triton(
                f_t5, v_t5, c_t5, is_best_of=is_best_of
            )
            fw_flat = fw_flat.clone()
            vw_flat = vw_flat.clone()
            w_unsort = w_unsort.clone()
            if g_sep5_max < 1e-15:
                fw_flat.index_copy_(0, tie5_idx, f_sym5)
                vw_flat.index_copy_(0, tie5_idx, v_sym5)
                w_unsort.index_copy_(0, tie5_idx, w_sym5)
            else:
                w_tie5 = 1.0 - g_sep5
                f_blend = g_sep5 * fw_flat.index_select(0, tie5_idx) + w_tie5 * f_sym5
                v_blend = g_sep5 * vw_flat.index_select(0, tie5_idx) + w_tie5 * v_sym5
                w_blend = g_sep5.unsqueeze(1) * w_unsort.index_select(0, tie5_idx) + w_tie5.unsqueeze(1) * w_sym5
                fw_flat.index_copy_(0, tie5_idx, f_blend)
                vw_flat.index_copy_(0, tie5_idx, v_blend)
                w_unsort.index_copy_(0, tie5_idx, w_blend)
            if sym5_cache is not None:
                sym5_cache["use_sym5_only"] = False
                sym5_cache["tie5_idx"] = tie5_idx
                sym5_cache["fwd"] = (f_t5, v_t5, c_t5, f_sym5, v_sym5, w_sym5, pbuf5, m2sym5, g_sep5, g_sep5_max)

        if tie_dyn_idx is not None and c_safe is not None:
            assert g_sep_dyn is not None
            f_td = f_safe.index_select(0, tie_dyn_idx).contiguous()
            v_td = v_safe.index_select(0, tie_dyn_idx).contiguous()
            c_td = c_safe.index_select(0, tie_dyn_idx).contiguous()
            m_td = am_flat.index_select(0, tie_dyn_idx).contiguous()
            f_sd, v_sd, w_sd = _eval_sym_dyn32_fwd(f_td, v_td, c_td, m_td, is_best_of=is_best_of)
            fw_flat = fw_flat.clone()
            vw_flat = vw_flat.clone()
            w_unsort = w_unsort.clone()
            if g_sep_dyn_max < 1e-15:
                fw_flat.index_copy_(0, tie_dyn_idx, f_sd)
                vw_flat.index_copy_(0, tie_dyn_idx, v_sd)
                w_unsort.index_copy_(0, tie_dyn_idx, w_sd)
            else:
                w_td = 1.0 - g_sep_dyn
                f_f = fw_flat.index_select(0, tie_dyn_idx)
                v_f = vw_flat.index_select(0, tie_dyn_idx)
                w_f = w_unsort.index_select(0, tie_dyn_idx)
                f_b = g_sep_dyn * f_f + w_td * f_sd
                m2_b = g_sep_dyn * (f_f * f_f * torch.exp(v_f * v_f)) + w_td * (f_sd * f_sd * torch.exp(v_sd * v_sd))
                v_b = torch.sqrt(torch.clamp(torch.log(torch.clamp(m2_b / (f_b * f_b).clamp(min=1e-30), min=1.0)), min=1e-16))
                w_b = g_sep_dyn.unsqueeze(1) * w_f + w_td.unsqueeze(1) * w_sd
                fw_flat.index_copy_(0, tie_dyn_idx, f_b)
                vw_flat.index_copy_(0, tie_dyn_idx, v_b)
                w_unsort.index_copy_(0, tie_dyn_idx, w_b)

        if tie_rem_idx is not None and c_safe is not None:
            fw_flat = fw_flat.clone()
            vw_flat = vw_flat.clone()
            w_unsort = w_unsort.clone()
            with torch.no_grad():
                for s_t in range(0, int(tie_rem_idx.numel()), 512):
                    sub_idx = tie_rem_idx[s_t : s_t + 512]
                    fw_tie, vw_tie, w_tie = stulz_margrabe_moments_with_weights(
                        fp_flat.index_select(0, sub_idx),
                        vp_flat.index_select(0, sub_idx),
                        c_safe.index_select(0, sub_idx),
                        am_flat.index_select(0, sub_idx),
                        monitoring_index="BestOf" if is_best_of else "WorstOf",
                    )
                    fw_flat.index_copy_(0, sub_idx, fw_tie)
                    vw_flat.index_copy_(0, sub_idx, vw_tie)
                    w_unsort.index_copy_(0, sub_idx, w_tie)

    f_w = fw_flat.view(b_sz, m_act).clamp(min=1e-12)
    v_w = vw_flat.view(b_sz, m_act).clamp(min=1e-6)
    w_all = (w_unsort.view(b_sz, m_act, d_act) * am_in.unsqueeze(1)).contiguous()

    grid_b = (triton.cdiv(b_sz, 32),)
    if m_act > 1:
        r_next = torch.empty((b_sz, m_act - 1), dtype=torch.float64, device=dev)
        _rnext_fwd_kernel64[grid_b](
            w_all, v_in, corr_in, r_next, f_w,
            b_sz, M=m_act, D=d_act, SAVE_SC=False, BLOCK_B=32, num_warps=1,
        )
    else:
        r_next = f_w

    p_out = torch.empty((b_sz, m_act), dtype=torch.float64, device=dev)
    q_out = torch.empty((b_sz, m_act), dtype=torch.float64, device=dev)
    lp_out = torch.empty((b_sz,), dtype=torch.float64, device=dev)
    _markov_fwd_kernel64[grid_b](
        f_w, v_w, r_next, sm_in, b_ko_in, b_cpn_in, is_c_in, is_g_in,
        ko_sp_in, cpn_sp_in, ki_sp_in, k_strike_in, k_low_in, active_ki_in, b_ki_in,
        force_cpn_in, idx_last_in,
        p_out, q_out, lp_out, f_w, f_w,
        b_sz, M=m_act, ALL_SCHED_ACT=all_sched_act, SAVE_SCRATCH=False, BLOCK_B=32, num_warps=1,
    )
    w_term = (
        w_all[:, -1, :]
        if all_sched_act
        else w_all.gather(1, idx_last_in.view(b_sz, 1, 1).expand(b_sz, 1, d_act)).squeeze(1)
    )
    return torch.log(f_w), v_w, p_out, q_out, lp_out, w_term


def _eval_tier1_vjp_triton_f64(
    inputs: tuple[torch.Tensor, ...],
    *,
    is_best_of: bool,
    all_sched_act: bool,
    tie_bm: torch.Tensor | None = None,
    sym5_cache: dict[str, Any] | None = None,
    force_sym5_only: bool = False,
    force_sym5_blend_all: bool = False,
    use_clark32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Triton Tier-1 VJP eliminating autograd graph overhead (D <= 8)."""
    (
        fp_in, v_in, corr_in, am_in, sm_in, b_ko_in, b_cpn_in, is_c_in, is_g_in,
        ko_sp_in, cpn_sp_in, ki_sp_in, k_strike_in, k_low_in, active_ki_in, b_ki_in,
        force_cpn_in, idx_last_in,
        w_lfw_in, w_vw_in, w_p_in, w_q_in, w_lp_in, w_wt_in,
    ) = inputs
    b_sz, d_act, m_act = fp_in.shape
    dev = fp_in.device
    bm = b_sz * m_act
    use_dyn32 = (d_act > 5) or (use_clark32 and d_act == 5)

    fp_flat = fp_in.transpose(1, 2).reshape(bm, d_act)
    vp_flat = v_in.transpose(1, 2).reshape(bm, d_act)
    am_flat = am_in.unsqueeze(1).expand(b_sz, m_act, d_act).reshape(bm, d_act)
    act_b = am_flat > 0.5
    f_safe = torch.where(act_b, fp_flat.clamp(min=1e-12), 1.0)
    v_safe = torch.where(act_b, vp_flat.clamp(min=1e-6), 0.1)

    pair_act: torch.Tensor | None = None
    c_safe: torch.Tensor | None = None
    ord_f: torch.Tensor | None = None
    inv_o: torch.Tensor | None = None
    tie5_idx: torch.Tensor | None = None
    tie_dyn_idx: torch.Tensor | None = None
    tie_rem_idx: torch.Tensor | None = None
    sym5_fwd_data: tuple[Any, ...] | None = None
    use_sym5_only = False
    use_sym_dyn_only = False
    g_sep_dyn: torch.Tensor | None = None
    g_sep_dyn_max = 1.0
    sym_dyn_blend_data: tuple[torch.Tensor, ...] | None = None
    dyn32_pack: tuple[torch.Tensor, ...] | None = None

    if d_act > 1:
        pair_act_t = (am_in > 0.5).unsqueeze(2) & (am_in > 0.5).unsqueeze(1)
        eye_d = torch.eye(d_act, dtype=torch.float64, device=dev).unsqueeze(0)
        c_safe_t = torch.where(pair_act_t, corr_in, eye_d).unsqueeze(1).expand(b_sz, m_act, d_act, d_act).reshape(bm, d_act, d_act)
        pair_act = pair_act_t
        c_safe = c_safe_t
        if force_sym5_only and d_act == 5:
            f_t5 = f_safe.contiguous()
            v_t5 = v_safe.contiguous()
            c_t5 = c_safe.contiguous()
            f_sym5, v_sym5, w_sym5, pbuf5, m2sym5 = _eval_sym5_fwd_triton(
                f_t5, v_t5, c_t5, is_best_of=is_best_of
            )
            sym5_fwd_data = (f_t5, v_t5, c_t5, f_sym5, v_sym5, w_sym5, pbuf5, m2sym5, None, 0.0)
            use_sym5_only = True
        elif force_sym5_blend_all and d_act == 5:
            f_t5 = f_safe.contiguous()
            v_t5 = v_safe.contiguous()
            c_t5 = c_safe.contiguous()
            g_sep5 = _sym5_gsep(f_t5)
            f_sym5, v_sym5, w_sym5, pbuf5, m2sym5 = _eval_sym5_fwd_triton(
                f_t5, v_t5, c_t5, is_best_of=is_best_of
            )
            sym5_fwd_data = (f_t5, v_t5, c_t5, f_sym5, v_sym5, w_sym5, pbuf5, m2sym5, g_sep5, 1.0)
        elif sym5_cache is not None and "fwd" in sym5_cache:
            tie5_idx = sym5_cache["tie5_idx"]
            sym5_fwd_data = sym5_cache["fwd"]
            use_sym5_only = bool(sym5_cache.get("use_sym5_only", False))
            if tie_bm is not None and d_act == 5 and tie5_idx is not None:
                tie_idx = tie_bm.nonzero(as_tuple=True)[0]
                if int(tie5_idx.numel()) < int(tie_idx.numel()):
                    is_act5 = act_b.index_select(0, tie_idx).all(dim=1)
                    tie_rem_idx = tie_idx[~is_act5]
                    if int(tie_rem_idx.numel()) == 0:
                        tie_rem_idx = None
        elif tie_bm is not None:
            tie_idx = tie_bm.nonzero(as_tuple=True)[0]
            if d_act == 5:
                is_act5 = act_b.index_select(0, tie_idx).all(dim=1)
                if bool(is_act5.all().item()):
                    tie5_idx = tie_idx
                else:
                    tie5_idx = tie_idx[is_act5]
                    tie_rem_idx = tie_idx[~is_act5]
                    if int(tie5_idx.numel()) == 0:
                        tie5_idx = None
                    if int(tie_rem_idx.numel()) == 0:
                        tie_rem_idx = None
            elif d_act in (2, 3, 6, 7, 8):
                is_full = act_b.index_select(0, tie_idx).all(dim=1)
                if bool(is_full.all().item()):
                    tie_dyn_idx = tie_idx
                else:
                    tie_dyn_idx = tie_idx[is_full]
                    tie_rem_idx = tie_idx[~is_full]
                    if int(tie_dyn_idx.numel()) == 0:
                        tie_dyn_idx = None
                    if int(tie_rem_idx.numel()) == 0:
                        tie_rem_idx = None
            else:
                tie_rem_idx = tie_idx
            if tie5_idx is not None:
                f_t5 = f_safe.index_select(0, tie5_idx).contiguous()
                v_t5 = v_safe.index_select(0, tie5_idx).contiguous()
                c_t5 = c_safe.index_select(0, tie5_idx).contiguous()
                g_sep5 = _sym5_gsep(f_t5)
                g_sep5_max = float(g_sep5.max().item())
                f_sym5, v_sym5, w_sym5, pbuf5, m2sym5 = _eval_sym5_fwd_triton(
                    f_t5, v_t5, c_t5, is_best_of=is_best_of
                )
                sym5_fwd_data = (f_t5, v_t5, c_t5, f_sym5, v_sym5, w_sym5, pbuf5, m2sym5, g_sep5, g_sep5_max)
                if int(tie5_idx.numel()) == bm and g_sep5_max < 1e-15:
                    use_sym5_only = True
            if tie_dyn_idx is not None:
                f_td = f_safe.index_select(0, tie_dyn_idx).contiguous()
                g_sep_dyn = _sym_gsep_dyn(f_td, act_b.index_select(0, tie_dyn_idx).double())
                g_sep_dyn_max = float(g_sep_dyn.max().item())
                if int(tie_dyn_idx.numel()) == bm and g_sep_dyn_max < 1e-15:
                    use_sym_dyn_only = True

    if use_sym5_only and sym5_fwd_data is not None:
        _, _, _, f_sym5, v_sym5, w_sym5, _, _, _, _ = sym5_fwd_data
        fw_flat = f_sym5
        vw_flat = v_sym5
        w_unsort = w_sym5
        grid_bm = None
    elif use_sym_dyn_only and c_safe is not None:
        fw_flat, vw_flat, w_unsort = _eval_sym_dyn32_fwd(
            f_safe.contiguous(), v_safe.contiguous(), c_safe.contiguous(), am_flat.contiguous(), is_best_of=is_best_of
        )
        grid_bm = None
    else:
        if d_act > 1:
            assert c_safe is not None
            sort_key = torch.where(act_b, -f_safe if is_best_of else f_safe, 1e12)
            ord_f_t = torch.argsort(sort_key, dim=1, stable=True)
            inv_o_t = torch.argsort(ord_f_t, dim=1)
            f_o = torch.gather(f_safe, 1, ord_f_t).contiguous()
            v_o = torch.gather(v_safe, 1, ord_f_t).contiguous()
            m_o = torch.gather(am_flat, 1, ord_f_t).contiguous()
            c_o = torch.gather(
                torch.gather(c_safe, 1, ord_f_t.unsqueeze(2).expand(bm, d_act, d_act)),
                2,
                ord_f_t.unsqueeze(1).expand(bm, d_act, d_act),
            ).contiguous()
            ord_f = ord_f_t
            inv_o = inv_o_t
        else:
            f_o = f_safe.contiguous()
            v_o = v_safe.contiguous()
            m_o = am_flat.contiguous()
            c_o = f_o

        if use_dyn32:
            fw_flat, vw_flat, w_o_flat, dyn32_pack = _eval_clark32_fwd_triton(
                f_o, v_o, c_o, m_o, is_best_of=is_best_of, save_sc=True
            )
            grid_bm = None
        else:
            sc_s = torch.empty((14, d_act, bm), dtype=torch.float64, device=dev) if d_act > 1 else f_o
            sc_w = torch.empty((d_act, d_act, bm), dtype=torch.float64, device=dev) if d_act > 1 else f_o
            sc_rho = torch.empty((2, d_act, d_act, bm), dtype=torch.float64, device=dev) if d_act > 1 else f_o
            fw_flat = torch.empty(bm, dtype=torch.float64, device=dev)
            vw_flat = torch.empty(bm, dtype=torch.float64, device=dev)
            w_o_flat = torch.empty((bm, d_act), dtype=torch.float64, device=dev)
            grid_bm = (triton.cdiv(bm, 32),)
            _clark_fwd_kernel64[grid_bm](
                f_o, v_o, c_o, m_o, fw_flat, vw_flat, w_o_flat, sc_s, sc_w, sc_rho,
                bm, D=d_act, IS_BEST=is_best_of, SAVE_SC=True, BLOCK_BM=32, num_warps=1,
            )
        w_unsort = torch.gather(w_o_flat, 1, inv_o) if inv_o is not None else w_o_flat

    f_fast_5: torch.Tensor | None = None
    v_fast_5: torch.Tensor | None = None
    w_fast_5: torch.Tensor | None = None
    if force_sym5_blend_all and sym5_fwd_data is not None:
        _, _, _, f_sym5, v_sym5, w_sym5, _, _, g_sep5, _ = sym5_fwd_data
        f_fast_5 = fw_flat
        v_fast_5 = vw_flat
        w_fast_5 = w_unsort
        w_tie5 = 1.0 - g_sep5
        fw_flat = g_sep5 * f_fast_5 + w_tie5 * f_sym5
        vw_flat = g_sep5 * v_fast_5 + w_tie5 * v_sym5
        w_unsort = g_sep5.unsqueeze(1) * w_fast_5 + w_tie5.unsqueeze(1) * w_sym5
    elif not use_sym5_only and sym5_fwd_data is not None and tie5_idx is not None:
        _, _, _, f_sym5, v_sym5, w_sym5, _, _, g_sep5, g_sep5_max = sym5_fwd_data
        fw_flat = fw_flat.clone()
        vw_flat = vw_flat.clone()
        w_unsort = w_unsort.clone()
        if g_sep5_max < 1e-15:
            fw_flat.index_copy_(0, tie5_idx, f_sym5)
            vw_flat.index_copy_(0, tie5_idx, v_sym5)
            w_unsort.index_copy_(0, tie5_idx, w_sym5)
        else:
            f_fast_5 = fw_flat.index_select(0, tie5_idx)
            v_fast_5 = vw_flat.index_select(0, tie5_idx)
            w_fast_5 = w_unsort.index_select(0, tie5_idx)
            w_tie5 = 1.0 - g_sep5
            fw_flat.index_copy_(0, tie5_idx, g_sep5 * f_fast_5 + w_tie5 * f_sym5)
            vw_flat.index_copy_(0, tie5_idx, g_sep5 * v_fast_5 + w_tie5 * v_sym5)
            w_unsort.index_copy_(0, tie5_idx, g_sep5.unsqueeze(1) * w_fast_5 + w_tie5.unsqueeze(1) * w_sym5)

    if not use_sym_dyn_only and tie_dyn_idx is not None and c_safe is not None:
        assert g_sep_dyn is not None
        f_td = f_safe.index_select(0, tie_dyn_idx).contiguous()
        v_td = v_safe.index_select(0, tie_dyn_idx).contiguous()
        c_td = c_safe.index_select(0, tie_dyn_idx).contiguous()
        m_td = am_flat.index_select(0, tie_dyn_idx).contiguous()
        f_sd, v_sd, w_sd = _eval_sym_dyn32_fwd(f_td, v_td, c_td, m_td, is_best_of=is_best_of)
        fw_flat = fw_flat.clone()
        vw_flat = vw_flat.clone()
        w_unsort = w_unsort.clone()
        if g_sep_dyn_max < 1e-15:
            fw_flat.index_copy_(0, tie_dyn_idx, f_sd)
            vw_flat.index_copy_(0, tie_dyn_idx, v_sd)
            w_unsort.index_copy_(0, tie_dyn_idx, w_sd)
        else:
            f_fd = fw_flat.index_select(0, tie_dyn_idx)
            v_fd = vw_flat.index_select(0, tie_dyn_idx)
            w_fd = w_unsort.index_select(0, tie_dyn_idx)
            w_td = 1.0 - g_sep_dyn
            f_b = g_sep_dyn * f_fd + w_td * f_sd
            m2_b = g_sep_dyn * (f_fd * f_fd * torch.exp(v_fd * v_fd)) + w_td * (f_sd * f_sd * torch.exp(v_sd * v_sd))
            v_b = torch.sqrt(torch.clamp(torch.log(torch.clamp(m2_b / (f_b * f_b).clamp(min=1e-30), min=1.0)), min=1e-16))
            w_b = g_sep_dyn.unsqueeze(1) * w_fd + w_td.unsqueeze(1) * w_sd
            fw_flat.index_copy_(0, tie_dyn_idx, f_b)
            vw_flat.index_copy_(0, tie_dyn_idx, v_b)
            w_unsort.index_copy_(0, tie_dyn_idx, w_b)
            sym_dyn_blend_data = (f_td, v_td, c_td, m_td, f_fd, v_fd, w_fd, f_sd, v_sd, w_sd)

    if tie_rem_idx is not None and c_safe is not None:
        fw_flat = fw_flat.clone()
        vw_flat = vw_flat.clone()
        w_unsort = w_unsort.clone()
        with torch.no_grad():
            for s_t in range(0, int(tie_rem_idx.numel()), 512):
                sub_idx = tie_rem_idx[s_t : s_t + 512]
                fw_tie, vw_tie, w_tie = stulz_margrabe_moments_with_weights(
                    fp_flat.index_select(0, sub_idx),
                    vp_flat.index_select(0, sub_idx),
                    c_safe.index_select(0, sub_idx),
                    am_flat.index_select(0, sub_idx),
                    monitoring_index="BestOf" if is_best_of else "WorstOf",
                )
                fw_flat.index_copy_(0, sub_idx, fw_tie)
                vw_flat.index_copy_(0, sub_idx, vw_tie)
                w_unsort.index_copy_(0, sub_idx, w_tie)

    f_w_raw = fw_flat.view(b_sz, m_act)
    v_w_raw = vw_flat.view(b_sz, m_act)
    f_w = f_w_raw.clamp(min=1e-12)
    v_w = v_w_raw.clamp(min=1e-6)
    w_all = (w_unsort.view(b_sz, m_act, d_act) * am_in.unsqueeze(1)).contiguous()

    grid_b = (triton.cdiv(b_sz, 32),)
    if m_act > 1:
        r_next = torch.empty((b_sz, m_act - 1), dtype=torch.float64, device=dev)
        sc_rn = torch.empty((4, m_act - 1, b_sz), dtype=torch.float64, device=dev)
        _rnext_fwd_kernel64[grid_b](
            w_all, v_in, corr_in, r_next, sc_rn,
            b_sz, M=m_act, D=d_act, SAVE_SC=True, BLOCK_B=32, num_warps=1,
        )
    else:
        r_next = f_w
        sc_rn = f_w

    p_out = torch.empty((b_sz, m_act), dtype=torch.float64, device=dev)
    q_out = torch.empty((b_sz, m_act), dtype=torch.float64, device=dev)
    lp_out = torch.empty((b_sz,), dtype=torch.float64, device=dev)
    g_fw_m = torch.empty((b_sz, m_act), dtype=torch.float64, device=dev)
    g_vw_m = torch.empty((b_sz, m_act), dtype=torch.float64, device=dev)
    g_rn_m = torch.empty((b_sz, max(m_act - 1, 1)), dtype=torch.float64, device=dev)
    sc_mk = torch.empty((18, m_act, b_sz), dtype=torch.float64, device=dev)
    term_mk = torch.empty((6, b_sz), dtype=torch.float64, device=dev)

    _markov_fwd_kernel64[grid_b](
        f_w, v_w, r_next, sm_in, b_ko_in, b_cpn_in, is_c_in, is_g_in,
        ko_sp_in, cpn_sp_in, ki_sp_in, k_strike_in, k_low_in, active_ki_in, b_ki_in,
        force_cpn_in, idx_last_in,
        p_out, q_out, lp_out, sc_mk, term_mk,
        b_sz, M=m_act, ALL_SCHED_ACT=all_sched_act, SAVE_SCRATCH=True, BLOCK_B=32, num_warps=1,
    )
    _markov_bwd_kernel64[grid_b](
        f_w, v_w, r_next, sm_in, is_c_in,
        ko_sp_in, ki_sp_in, k_strike_in, k_low_in, active_ki_in, b_ki_in,
        idx_last_in, w_p_in, w_q_in, w_lp_in,
        lp_out, g_fw_m, g_vw_m, g_rn_m, sc_mk, term_mk,
        b_sz, M=m_act, ALL_SCHED_ACT=all_sched_act, BLOCK_B=32, num_warps=1,
    )

    g_w_tot = torch.zeros((b_sz, m_act, d_act), dtype=torch.float64, device=dev)
    wt_masked = w_wt_in * am_in
    if all_sched_act:
        g_w_tot[:, -1, :] = wt_masked
    else:
        g_w_tot.scatter_add_(1, idx_last_in.view(b_sz, 1, 1).expand(b_sz, 1, d_act), wt_masked.unsqueeze(1))

    if m_act > 1:
        g_v_rn = torch.empty_like(v_in)
        g_c_rn = torch.empty_like(corr_in)
        _rnext_bwd_kernel64[grid_b](
            w_all, v_in, corr_in, g_rn_m, sc_rn,
            g_w_tot, g_v_rn, g_c_rn,
            b_sz, M=m_act, D=d_act, BLOCK_B=32, num_warps=1,
        )
    else:
        g_v_rn = None
        g_c_rn = None

    g_w_tot = g_w_tot * am_in.unsqueeze(1)
    g_fw_tot = torch.where(f_w_raw >= 1e-12, g_fw_m + w_lfw_in / f_w, 0.0).reshape(bm)
    g_vw_tot = torch.where(v_w_raw >= 1e-6, g_vw_m + w_vw_in, 0.0).reshape(bm)
    gw_flat = g_w_tot.reshape(bm, d_act)

    if use_sym5_only and sym5_fwd_data is not None:
        f_t5, v_t5, c_t5, f_sym5, v_sym5, _, pbuf5, m2sym5, _, _ = sym5_fwd_data
        gf_uns, gv_uns, gc_uns = _eval_sym5_bwd_triton(
            f_t5, v_t5, c_t5, pbuf5, f_sym5, v_sym5, m2sym5,
            g_fw_tot, g_vw_tot, gw_flat, is_best_of=is_best_of,
        )
        assert pair_act is not None
        g_c_clark = torch.where(pair_act, gc_uns.view(b_sz, m_act, d_act, d_act).sum(dim=1), 0.0)
        g_fp_unmasked = torch.where(act_b & (fp_flat >= 1e-12), gf_uns, 0.0)
        g_vp_unmasked = torch.where(act_b & (vp_flat >= 1e-6), gv_uns, 0.0)
        g_fp = g_fp_unmasked.view(b_sz, m_act, d_act).transpose(1, 2).contiguous()
        g_vp_c = g_vp_unmasked.view(b_sz, m_act, d_act).transpose(1, 2).contiguous()
        g_vp = g_vp_c + (g_v_rn if g_v_rn is not None else 0.0)
        g_c = g_c_clark + (g_c_rn if g_c_rn is not None else 0.0)
        return g_fp, g_vp, g_c

    if use_sym_dyn_only and c_safe is not None:
        _, _, _, gf_uns, gv_uns, gc_uns = _eval_sym_dyn32_bwd(
            f_safe.contiguous(), v_safe.contiguous(), c_safe.contiguous(), am_flat.contiguous(),
            g_fw_tot, g_vw_tot, gw_flat, is_best_of=is_best_of,
        )
        assert pair_act is not None
        g_c_clark = torch.where(pair_act, gc_uns.view(b_sz, m_act, d_act, d_act).sum(dim=1), 0.0)
        g_fp_unmasked = torch.where(act_b & (fp_flat >= 1e-12), gf_uns, 0.0)
        g_vp_unmasked = torch.where(act_b & (vp_flat >= 1e-6), gv_uns, 0.0)
        g_fp = g_fp_unmasked.view(b_sz, m_act, d_act).transpose(1, 2).contiguous()
        g_vp_c = g_vp_unmasked.view(b_sz, m_act, d_act).transpose(1, 2).contiguous()
        g_vp = g_vp_c + (g_v_rn if g_v_rn is not None else 0.0)
        g_c = g_c_clark + (g_c_rn if g_c_rn is not None else 0.0)
        return g_fp, g_vp, g_c

    g_fp_5: torch.Tensor | None = None
    g_vp_5: torch.Tensor | None = None
    g_c_5: torch.Tensor | None = None
    if force_sym5_blend_all and sym5_fwd_data is not None:
        f_t5, v_t5, c_t5, f_sym5, v_sym5, w_sym5, pbuf5, m2sym5, g_sep5, _ = sym5_fwd_data
        af_5 = g_fw_tot
        av_5 = g_vw_tot
        aw_5 = gw_flat
        w_tie5 = 1.0 - g_sep5
        g_fp_5, g_vp_5, g_c_5 = _eval_sym5_bwd_triton(
            f_t5, v_t5, c_t5, pbuf5, f_sym5, v_sym5, m2sym5,
            af_5 * w_tie5, av_5 * w_tie5, aw_5 * w_tie5.unsqueeze(1),
            is_best_of=is_best_of,
        )
        assert f_fast_5 is not None and v_fast_5 is not None and w_fast_5 is not None
        a_gsep5 = (
            af_5 * (f_fast_5 - f_sym5)
            + av_5 * (v_fast_5 - v_sym5)
            + (aw_5 * (w_fast_5 - w_sym5)).sum(dim=-1)
        )
        grad_gsep = _sym5_gsep_vjp(fp_flat, a_gsep5)
        g_fp_5 = torch.where(fp_flat >= 1e-12, g_fp_5, 0.0) + grad_gsep
        g_fw_tot = af_5 * g_sep5
        g_vw_tot = av_5 * g_sep5
        gw_flat = aw_5 * g_sep5.unsqueeze(1)
    elif sym5_fwd_data is not None and tie5_idx is not None:
        f_t5, v_t5, c_t5, f_sym5, v_sym5, w_sym5, pbuf5, m2sym5, g_sep5, g_sep5_max = sym5_fwd_data
        af_5 = g_fw_tot.index_select(0, tie5_idx)
        av_5 = g_vw_tot.index_select(0, tie5_idx)
        aw_5 = gw_flat.index_select(0, tie5_idx)
        if g_sep5_max < 1e-15:
            g_fp_5, g_vp_5, g_c_5 = _eval_sym5_bwd_triton(
                f_t5, v_t5, c_t5, pbuf5, f_sym5, v_sym5, m2sym5,
                af_5, av_5, aw_5, is_best_of=is_best_of,
            )
            g_fw_tot = g_fw_tot.clone()
            g_vw_tot = g_vw_tot.clone()
            gw_flat = gw_flat.clone()
            g_fw_tot.index_fill_(0, tie5_idx, 0.0)
            g_vw_tot.index_fill_(0, tie5_idx, 0.0)
            gw_flat.index_fill_(0, tie5_idx, 0.0)
        else:
            w_tie5 = 1.0 - g_sep5
            g_fp_5, g_vp_5, g_c_5 = _eval_sym5_bwd_triton(
                f_t5, v_t5, c_t5, pbuf5, f_sym5, v_sym5, m2sym5,
                af_5 * w_tie5, av_5 * w_tie5, aw_5 * w_tie5.unsqueeze(1),
                is_best_of=is_best_of,
            )
            assert f_fast_5 is not None and v_fast_5 is not None and w_fast_5 is not None
            a_gsep5 = (
                af_5 * (f_fast_5 - f_sym5)
                + av_5 * (v_fast_5 - v_sym5)
                + (aw_5 * (w_fast_5 - w_sym5)).sum(dim=-1)
            )
            f_t5_r = fp_flat.index_select(0, tie5_idx)
            grad_gsep = _sym5_gsep_vjp(f_t5_r, a_gsep5)
            g_fp_5 = torch.where(f_t5_r >= 1e-12, g_fp_5, 0.0) + grad_gsep
            g_fw_tot = g_fw_tot.clone()
            g_vw_tot = g_vw_tot.clone()
            gw_flat = gw_flat.clone()
            g_fw_tot.index_copy_(0, tie5_idx, af_5 * g_sep5)
            g_vw_tot.index_copy_(0, tie5_idx, av_5 * g_sep5)
            gw_flat.index_copy_(0, tie5_idx, aw_5 * g_sep5.unsqueeze(1))

    g_fp_dyn_tie: torch.Tensor | None = None
    g_vp_dyn_tie: torch.Tensor | None = None
    g_c_dyn_tie: torch.Tensor | None = None
    if tie_dyn_idx is not None and c_safe is not None:
        af_d = g_fw_tot.index_select(0, tie_dyn_idx)
        av_d = g_vw_tot.index_select(0, tie_dyn_idx)
        aw_d = gw_flat.index_select(0, tie_dyn_idx)
        if g_sep_dyn_max < 1e-15:
            f_td = f_safe.index_select(0, tie_dyn_idx).contiguous()
            v_td = v_safe.index_select(0, tie_dyn_idx).contiguous()
            c_td = c_safe.index_select(0, tie_dyn_idx).contiguous()
            m_td = am_flat.index_select(0, tie_dyn_idx).contiguous()
            _, _, _, g_fp_dyn_tie, g_vp_dyn_tie, g_c_dyn_tie = _eval_sym_dyn32_bwd(
                f_td, v_td, c_td, m_td, af_d, av_d, aw_d, is_best_of=is_best_of,
            )
            g_fw_tot = g_fw_tot.clone()
            g_vw_tot = g_vw_tot.clone()
            gw_flat = gw_flat.clone()
            g_fw_tot.index_fill_(0, tie_dyn_idx, 0.0)
            g_vw_tot.index_fill_(0, tie_dyn_idx, 0.0)
            gw_flat.index_fill_(0, tie_dyn_idx, 0.0)
        else:
            assert sym_dyn_blend_data is not None
            f_td, v_td, c_td, m_td, f_fd, v_fd, w_fd, f_sd, v_sd, w_sd = sym_dyn_blend_data
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                f_td_r = fp_flat.index_select(0, tie_dyn_idx).detach().requires_grad_(True)
                f_fd_r = f_fd.detach().requires_grad_(True)
                v_fd_r = v_fd.detach().requires_grad_(True)
                w_fd_r = w_fd.detach().requires_grad_(True)
                f_sd_r = f_sd.detach().requires_grad_(True)
                v_sd_r = v_sd.detach().requires_grad_(True)
                w_sd_r = w_sd.detach().requires_grad_(True)
                gs_r = _sym_gsep_dyn(f_td_r.clamp(min=1e-12), m_td)
                wt_r = 1.0 - gs_r
                fb_r = gs_r * f_fd_r + wt_r * f_sd_r
                m2_r = gs_r * (f_fd_r * f_fd_r * torch.exp(v_fd_r * v_fd_r)) + wt_r * (f_sd_r * f_sd_r * torch.exp(v_sd_r * v_sd_r))
                vb_r = torch.sqrt(torch.clamp(torch.log(torch.clamp(m2_r / (fb_r * fb_r).clamp(min=1e-30), min=1.0)), min=1e-16))
                wb_r = gs_r.unsqueeze(1) * w_fd_r + wt_r.unsqueeze(1) * w_sd_r
                b_obj = (fb_r * af_d + vb_r * av_d).sum() + (wb_r * aw_d).sum()
                g_gs_f, a_ff, a_vf, a_wf, a_fs, a_vs, a_ws = torch.autograd.grad(
                    b_obj, (f_td_r, f_fd_r, v_fd_r, w_fd_r, f_sd_r, v_sd_r, w_sd_r)
                )
            _, _, _, gf_s, g_vp_dyn_tie, g_c_dyn_tie = _eval_sym_dyn32_bwd(
                f_td, v_td, c_td, m_td, a_fs, a_vs, a_ws, is_best_of=is_best_of,
            )
            g_fp_dyn_tie = torch.where(f_td_r >= 1e-12, gf_s, 0.0) + g_gs_f
            g_fw_tot = g_fw_tot.clone()
            g_vw_tot = g_vw_tot.clone()
            gw_flat = gw_flat.clone()
            g_fw_tot.index_copy_(0, tie_dyn_idx, a_ff)
            g_vw_tot.index_copy_(0, tie_dyn_idx, a_vf)
            gw_flat.index_copy_(0, tie_dyn_idx, a_wf)

    g_fp_tie: torch.Tensor | None = None
    g_vp_tie: torch.Tensor | None = None
    g_c_tie: torch.Tensor | None = None
    if tie_rem_idx is not None:
        tie_n = int(tie_rem_idx.numel())
        c_bm_raw = corr_in.unsqueeze(1).expand(b_sz, m_act, d_act, d_act).reshape(bm, d_act, d_act)
        fp_tie_chunks: list[torch.Tensor] = []
        vp_tie_chunks: list[torch.Tensor] = []
        c_tie_chunks: list[torch.Tensor] = []
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            for s_t in range(0, tie_n, 512):
                sub_idx = tie_rem_idx[s_t : s_t + 512]
                fp_t_r = fp_flat.index_select(0, sub_idx).detach().requires_grad_(True)
                vp_t_r = vp_flat.index_select(0, sub_idx).detach().requires_grad_(True)
                c_t_r = c_bm_raw.index_select(0, sub_idx).detach().requires_grad_(True)
                fw_tie, vw_tie, w_tie = stulz_margrabe_moments_with_weights(
                    fp_t_r,
                    vp_t_r,
                    c_t_r,
                    am_flat.index_select(0, sub_idx),
                    monitoring_index="BestOf" if is_best_of else "WorstOf",
                )
                tie_obj = (
                    fw_tie * g_fw_tot.index_select(0, sub_idx)
                    + vw_tie * g_vw_tot.index_select(0, sub_idx)
                ).sum() + (w_tie * gw_flat.index_select(0, sub_idx)).sum()
                g_fp_t, g_vp_t, g_c_t = torch.autograd.grad(tie_obj, (fp_t_r, vp_t_r, c_t_r), allow_unused=True)
                fp_tie_chunks.append(g_fp_t.detach() if g_fp_t is not None else torch.zeros_like(fp_t_r))
                vp_tie_chunks.append(g_vp_t.detach() if g_vp_t is not None else torch.zeros_like(vp_t_r))
                c_tie_chunks.append(g_c_t.detach() if g_c_t is not None else torch.zeros_like(c_t_r))
        g_fp_tie = torch.cat(fp_tie_chunks, dim=0)
        g_vp_tie = torch.cat(vp_tie_chunks, dim=0)
        g_c_tie = torch.cat(c_tie_chunks, dim=0)
        g_fw_tot = g_fw_tot.clone()
        g_vw_tot = g_vw_tot.clone()
        gw_flat = gw_flat.clone()
        g_fw_tot.index_fill_(0, tie_rem_idx, 0.0)
        g_vw_tot.index_fill_(0, tie_rem_idx, 0.0)
        gw_flat.index_fill_(0, tie_rem_idx, 0.0)

    gw_o = torch.gather(gw_flat, 1, ord_f).contiguous() if ord_f is not None else gw_flat.contiguous()

    if use_dyn32:
        assert dyn32_pack is not None
        gf_o, gv_o, gc_o = _eval_clark32_bwd_triton(
            dyn32_pack, g_fw_tot, g_vw_tot, gw_o, is_best_of=is_best_of
        )
    else:
        gf_o = torch.empty((bm, d_act), dtype=torch.float64, device=dev)
        gv_o = torch.empty((bm, d_act), dtype=torch.float64, device=dev)
        gc_o = torch.empty((bm, d_act, d_act), dtype=torch.float64, device=dev)
        assert grid_bm is not None
        _clark_bwd_kernel64[grid_bm](
            f_o, v_o, c_o, m_o, g_fw_tot, g_vw_tot, gw_o, gf_o, gv_o, gc_o, sc_s, sc_w, sc_rho,
            bm, D=d_act, IS_BEST=is_best_of, BLOCK_BM=32, num_warps=1,
        )

    if inv_o is not None:
        gf_uns = torch.gather(gf_o, 1, inv_o)
        gv_uns = torch.gather(gv_o, 1, inv_o)
        gc_uns = torch.gather(
            torch.gather(gc_o, 1, inv_o.unsqueeze(2).expand(bm, d_act, d_act)),
            2,
            inv_o.unsqueeze(1).expand(bm, d_act, d_act),
        )
        if force_sym5_blend_all and g_fp_5 is not None and g_vp_5 is not None and g_c_5 is not None:
            gf_uns = torch.where(act_b & (fp_flat >= 1e-12), gf_uns, 0.0) + g_fp_5
            gv_uns = torch.where(act_b & (vp_flat >= 1e-6), gv_uns, 0.0) + g_vp_5
            gc_uns = gc_uns + g_c_5
        elif tie5_idx is not None and g_fp_5 is not None and g_vp_5 is not None and g_c_5 is not None:
            assert sym5_fwd_data is not None
            g_sep5_max = sym5_fwd_data[9]
            gf_uns = gf_uns.clone()
            gv_uns = gv_uns.clone()
            gc_uns = gc_uns.clone()
            if g_sep5_max < 1e-15:
                gf_uns.index_copy_(0, tie5_idx, g_fp_5)
                gv_uns.index_copy_(0, tie5_idx, g_vp_5)
                gc_uns.index_copy_(0, tie5_idx, g_c_5)
            else:
                gf_uns.index_copy_(0, tie5_idx, gf_uns.index_select(0, tie5_idx) + g_fp_5)
                gv_uns.index_copy_(0, tie5_idx, gv_uns.index_select(0, tie5_idx) + g_vp_5)
                gc_uns.index_copy_(0, tie5_idx, gc_uns.index_select(0, tie5_idx) + g_c_5)
        if tie_dyn_idx is not None and g_fp_dyn_tie is not None and g_vp_dyn_tie is not None and g_c_dyn_tie is not None:
            gf_uns = gf_uns.clone()
            gv_uns = gv_uns.clone()
            gc_uns = gc_uns.clone()
            if g_sep_dyn_max < 1e-15:
                gf_uns.index_copy_(0, tie_dyn_idx, g_fp_dyn_tie)
                gv_uns.index_copy_(0, tie_dyn_idx, g_vp_dyn_tie)
                gc_uns.index_copy_(0, tie_dyn_idx, g_c_dyn_tie)
            else:
                gf_uns.index_copy_(0, tie_dyn_idx, gf_uns.index_select(0, tie_dyn_idx) + g_fp_dyn_tie)
                gv_uns.index_copy_(0, tie_dyn_idx, gv_uns.index_select(0, tie_dyn_idx) + g_vp_dyn_tie)
                gc_uns.index_copy_(0, tie_dyn_idx, gc_uns.index_select(0, tie_dyn_idx) + g_c_dyn_tie)
        if tie_rem_idx is not None and g_fp_tie is not None and g_vp_tie is not None and g_c_tie is not None:
            gf_uns = gf_uns.clone()
            gv_uns = gv_uns.clone()
            gc_uns = gc_uns.clone()
            gf_uns.index_copy_(0, tie_rem_idx, g_fp_tie)
            gv_uns.index_copy_(0, tie_rem_idx, g_vp_tie)
            gc_uns.index_copy_(0, tie_rem_idx, g_c_tie)
        assert pair_act is not None
        g_c_clark = torch.where(pair_act, gc_uns.view(b_sz, m_act, d_act, d_act).sum(dim=1), 0.0)
    else:
        gf_uns = gf_o
        gv_uns = gv_o
        g_c_clark = torch.zeros_like(corr_in)

    no_tie = tie5_idx is None and tie_dyn_idx is None and tie_rem_idx is None and not force_sym5_blend_all
    g_fp_unmasked = torch.where(act_b & (fp_flat >= 1e-12), gf_uns, 0.0) if no_tie else gf_uns
    g_vp_unmasked = torch.where(act_b & (vp_flat >= 1e-6), gv_uns, 0.0) if no_tie else gv_uns
    g_fp = g_fp_unmasked.view(b_sz, m_act, d_act).transpose(1, 2).contiguous()
    g_vp_c = g_vp_unmasked.view(b_sz, m_act, d_act).transpose(1, 2).contiguous()
    g_vp = g_vp_c + (g_v_rn if g_v_rn is not None else 0.0)
    g_c = g_c_clark + (g_c_rn if g_c_rn is not None else 0.0)
    return g_fp, g_vp, g_c


def _batched_tier1_vjp_cuda_f32(
    fp_bm: torch.Tensor,
    v_bm: torch.Tensor,
    corr_sym: torch.Tensor,
    amask_64: torch.Tensor,
    smask_64: torch.Tensor,
    b_ko: torch.Tensor,
    b_cpn: torch.Tensor,
    is_call_base: torch.Tensor,
    is_guar_base: torch.Tensor,
    ko_spread_mag: torch.Tensor,
    cpn_spread_mag: torch.Tensor,
    ki_spread_mag: torch.Tensor,
    k_strike: torch.Tensor,
    k_low: torch.Tensor,
    active_ki: torch.Tensor,
    b_ki: torch.Tensor,
    force_cpn_vec: torch.Tensor,
    idx_last: torch.Tensor | None,
    w_lfw: torch.Tensor,
    w_vw: torch.Tensor,
    w_p: torch.Tensor,
    w_q: torch.Tensor,
    w_lp: torch.Tensor,
    w_wt: torch.Tensor,
    *,
    is_best_of: bool,
    all_sched_act: bool,
    sym5_cache: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDAGraph-cached GPU Tier-1 VJP (Triton float64 D <= 5, Triton f32 f_scale 6 <= D <= 8, f32 fallback)."""
    b_sz, d_act, m_act = fp_bm.shape
    device = fp_bm.device
    use_triton_f64 = bool(_HAS_TRITON_TIER1 and d_act <= 8)
    t_dtype = torch.float64 if use_triton_f64 else torch.float32
    idx_t = torch.zeros(b_sz, dtype=torch.long, device=device) if idx_last is None else idx_last.to(device=device, dtype=torch.long)
    live_inputs = (
        fp_bm.detach().to(t_dtype).contiguous(),
        v_bm.detach().to(t_dtype).contiguous(),
        corr_sym.detach().to(t_dtype).contiguous(),
        amask_64.detach().to(t_dtype).contiguous(),
        smask_64.detach().to(t_dtype).contiguous(),
        b_ko.detach().to(t_dtype).contiguous(),
        b_cpn.detach().to(t_dtype).contiguous(),
        is_call_base.detach().to(t_dtype).contiguous(),
        is_guar_base.detach().to(t_dtype).contiguous(),
        ko_spread_mag.squeeze(1).detach().to(t_dtype).contiguous(),
        cpn_spread_mag.squeeze(1).detach().to(t_dtype).contiguous(),
        ki_spread_mag.squeeze(1).detach().to(t_dtype).contiguous(),
        k_strike.detach().to(t_dtype).contiguous(),
        k_low.detach().to(t_dtype).contiguous(),
        (active_ki.detach().to(torch.float64) if use_triton_f64 else active_ki.detach()).contiguous(),
        b_ki.detach().to(t_dtype).contiguous(),
        force_cpn_vec.detach().to(t_dtype).contiguous(),
        idx_t.contiguous(),
        w_lfw.detach().to(t_dtype).contiguous(),
        w_vw.detach().to(t_dtype).contiguous(),
        w_p.detach().to(t_dtype).contiguous(),
        w_q.detach().to(t_dtype).contiguous(),
        w_lp.detach().to(t_dtype).contiguous(),
        w_wt.detach().to(t_dtype).contiguous(),
    )

    can_cg = bool(
        use_triton_f64
        and d_act <= 5
        and (b_sz * d_act * m_act) <= 150_000
        and not torch.cuda.is_current_stream_capturing()  # type: ignore[no-untyped-call]
    )
    use_cg_sym5 = False
    use_cg_sym5_blend = False
    use_clark32_5 = False
    tie_bm: torch.Tensor | None = None
    if can_cg and d_act > 1:
        if d_act == 5:
            fp_fl_c = fp_bm.detach().transpose(1, 2).reshape(b_sz * m_act, 5).clamp(min=1e-16)
            lf_c = torch.log(fp_fl_c)
            s_lf_c = torch.sort(lf_c, dim=1).values
            min_gap_c = (s_lf_c[:, 1:] - s_lf_c[:, :-1]).min()
            f_rmax_c = fp_fl_c.max(dim=1).values / fp_fl_c.min(dim=1).values
            flags = torch.stack([
                (amask_64 > 0.5).all(),
                (f_rmax_c < 1.002).all(),
                min_gap_c > 0.062,
            ]).tolist()
            if bool(flags[0]):
                if bool(flags[1]):
                    use_cg_sym5 = True
                elif bool(flags[2]):
                    use_clark32_5 = True
                else:
                    use_cg_sym5_blend = True
        else:
            fp_flat_c = fp_bm.detach().transpose(1, 2).reshape(b_sz * m_act, d_act).clamp(min=1e-16)
            am_flat_c = (amask_64 > 0.5).unsqueeze(1).expand(b_sz, m_act, d_act).reshape(b_sz * m_act, d_act)
            pair_off_c = (am_flat_c.unsqueeze(2) & am_flat_c.unsqueeze(1)) & (
                ~torch.eye(d_act, dtype=torch.bool, device=device).unsqueeze(0)
            )
            dl_sq_c = torch.log(fp_flat_c.unsqueeze(2) / fp_flat_c.unsqueeze(1)) ** 2
            tie_bm_raw = ((dl_sq_c < 0.0035367) & pair_off_c).any(dim=(-2, -1))
            has_tie_c = bool(tie_bm_raw.any().item())
            if has_tie_c:
                tie_bm = tie_bm_raw
    elif use_triton_f64:
        tie_bm = _clark_tie_mask_bm(fp_bm, amask_64)

    can_cg_now = bool(can_cg and (use_cg_sym5 or use_cg_sym5_blend or use_clark32_5 or tie_bm is None))
    cg_tag = (
        "sym5_vjp64"
        if use_cg_sym5
        else (
            "sym5_blend_vjp64"
            if use_cg_sym5_blend
            else ("vjp32c" if use_clark32_5 else ("vjp64" if use_triton_f64 else "vjp32"))
        )
    )
    cache_key = (cg_tag, int(device.index or 0), b_sz, d_act, m_act, is_best_of, all_sched_act)
    do_capture = False
    if can_cg_now:
        with _TIER1_CG_LOCK:
            entry_chk = _TIER1_CG_CACHE.get(cache_key)
            if entry_chk is not None:
                do_capture = True
            else:
                seen = _TIER1_CG_SEEN.get(cache_key, 0)
                if seen >= 1:
                    do_capture = True
                else:
                    if len(_TIER1_CG_SEEN) >= 256:
                        _TIER1_CG_SEEN.clear()
                    _TIER1_CG_SEEN[cache_key] = 1

    if not do_capture:
        if use_triton_f64:
            g_fp, g_vp, g_c = _eval_tier1_vjp_triton_f64(
                live_inputs,
                is_best_of=is_best_of,
                all_sched_act=all_sched_act,
                tie_bm=tie_bm,
                sym5_cache=sym5_cache,
                force_sym5_only=use_cg_sym5,
                force_sym5_blend_all=use_cg_sym5_blend,
                use_clark32=use_clark32_5,
            )
        else:
            g_fp, g_vp, g_c = _eval_tier1_vjp_f32_core(
                live_inputs, is_best_of=is_best_of, all_sched_act=all_sched_act
            )
        return g_fp.to(torch.float64), g_vp.to(torch.float64), g_c.to(torch.float64)

    def _cg_core_vjp(
        inputs: tuple[torch.Tensor, ...], *, is_best_of: bool, all_sched_act: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _eval_tier1_vjp_triton_f64(
            inputs,
            is_best_of=is_best_of,
            all_sched_act=all_sched_act,
            force_sym5_only=use_cg_sym5,
            force_sym5_blend_all=use_cg_sym5_blend,
            use_clark32=use_clark32_5,
        )

    core_fn = _cg_core_vjp
    with _TIER1_CG_LOCK:
        entry = _TIER1_CG_CACHE.get(cache_key)
        if entry is not None:
            _TIER1_CG_CACHE.move_to_end(cache_key)
        else:
            static_in = tuple(t.clone() for t in live_inputs)
            cap_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            cap_stream.wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(cap_stream):
                for _ in range(2):
                    _ = core_fn(
                        static_in, is_best_of=is_best_of, all_sched_act=all_sched_act
                    )
            torch.cuda.current_stream(device=device).wait_stream(cap_stream)
            cg = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
            with torch.cuda.graph(cg):
                static_out = core_fn(
                    static_in, is_best_of=is_best_of, all_sched_act=all_sched_act
                )
            entry = {
                "lock": threading.Lock(),
                "graph": cg,
                "static_in": static_in,
                "static_out": static_out,
            }
            _TIER1_CG_CACHE[cache_key] = entry
            evicted = False
            while len(_TIER1_CG_CACHE) > _TIER1_CG_MAXSIZE:
                _, old_entry = _TIER1_CG_CACHE.popitem(last=False)
                del old_entry
                evicted = True
            if evicted:
                torch.cuda.empty_cache()

    entry_lock: threading.Lock = entry["lock"]
    with entry_lock:
        for s_t, l_t in zip(entry["static_in"], live_inputs):
            s_t.copy_(l_t)
        entry["graph"].replay()
        g_fp, g_vp, g_c = (o.to(torch.float64).clone() for o in entry["static_out"])
    return g_fp, g_vp, g_c


def _ko_and_ramp_fwd_only_f32(
    fc: torch.Tensor,
    vc: torch.Tensor,
    b_ko: torch.Tensor,
    ko_sp_mag: torch.Tensor,
    *,
    need_ramp: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Forward-only KO call-spread probability and truncated-normal moments without derivative work."""
    f_c = fc.clamp(min=1e-12)
    l_c = vc.clamp(min=1e-6)
    b_c = b_ko.clamp(min=1e-6)
    is_sp = ko_sp_mag > 1e-5
    b_lo = torch.minimum(b_c, b_c - ko_sp_mag).clamp(min=1e-6)
    b_hi = torch.maximum(b_c, b_c - ko_sp_mag).clamp(min=1e-6)
    db = (b_hi - b_lo).clamp(min=1e-7)
    inv_l = 1.0 / l_c
    m_u = torch.log(f_c) - 0.5 * l_c * l_c
    u_hi = (torch.log(b_hi) - m_u) * inv_l
    u_lo = (torch.log(b_lo) - m_u) * inv_l
    phi_cdf_hi = _ndtr_f32(u_hi)
    phi_cdf_hi_m = _ndtr_f32(u_hi - l_c)
    phi_cdf_lo = _ndtr_f32(u_lo)
    phi_cdf_lo_m = _ndtr_f32(u_lo - l_c)
    dphi_m_tail = _ndtr_f32(-(u_lo - l_c)) - _ndtr_f32(-(u_hi - l_c))
    dphi_m = torch.where(u_hi + u_lo > 0.0, dphi_m_tail, phi_cdf_hi_m - phi_cdf_lo_m)
    m0_sp = (b_hi * phi_cdf_hi - b_lo * phi_cdf_lo - f_c * dphi_m) / db
    ko_sp_tail = (f_c * dphi_m_tail + b_hi * _ndtr_f32(-u_hi) - b_lo * _ndtr_f32(-u_lo)) / db
    ko_sp_raw = torch.where(u_hi + u_lo > 0.0, ko_sp_tail, 1.0 - m0_sp)
    u_pt = (torch.log(b_c) - m_u) * inv_l
    ko_pt_raw = _ndtr_f32(-u_pt).clamp(0.0, 1.0)
    ko_raw = torch.where(is_sp, ko_sp_raw.clamp(0.0, 1.0), ko_pt_raw)
    if not need_ramp:
        return ko_raw, None, None
    phi_hi = torch.exp(-0.5 * u_hi * u_hi) * _INV_SQRT2PI_GE
    phi_lo = torch.exp(-0.5 * u_lo * u_lo) * _INV_SQRT2PI_GE
    m1_sp = -(l_c * f_c / db) * dphi_m
    m2_sp = m0_sp + l_c * m1_sp + (l_c / db) * (b_hi * phi_hi - b_lo * phi_lo)
    inv_m0 = 1.0 / m0_sp.clamp(min=1e-20)
    mu_sp = m1_sp * inv_m0
    var_sp = (m2_sp * inv_m0 - mu_sp * mu_sp).clamp(min=1e-8)
    phi_pt = torch.exp(-0.5 * u_pt * u_pt) * _INV_SQRT2PI_GE
    lam_pt = phi_pt / _ndtr_f32(u_pt).clamp(min=1e-20)
    mu_pt = -lam_pt
    var_pt = (1.0 + u_pt * lam_pt - lam_pt * lam_pt).clamp(min=1e-8)
    return ko_raw, torch.where(is_sp, mu_sp, mu_pt), torch.where(is_sp, var_sp, var_pt)


def _eval_tier1_fwd_f32_core(
    inputs: tuple[torch.Tensor, ...],
    *,
    is_best_of: bool,
    all_sched_act: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-only float32 Tier-1 proxy kernel for CUDAGraph capture."""
    (
        fp_in, v_in, corr_in, am_in, sm_in, b_ko_in, b_cpn_in, is_c_in, is_g_in,
        ko_sp_in, cpn_sp_in, ki_sp_in, k_strike_in, k_low_in, active_ki_in, b_ki_in,
        force_cpn_in, idx_last_in,
    ) = inputs
    b_sz, d_act, m_act = fp_in.shape
    device = fp_in.device
    with torch.no_grad():
        fp_flat = fp_in.transpose(1, 2).reshape(b_sz * m_act, d_act)
        v_flat = v_in.transpose(1, 2).reshape(b_sz * m_act, d_act)
        c_flat = corr_in.unsqueeze(1).expand(b_sz, m_act, d_act, d_act).reshape(b_sz * m_act, d_act, d_act)
        am_flat = am_in.unsqueeze(1).expand(b_sz, m_act, d_act).reshape(b_sz * m_act, d_act)
        fw_flat, vw_flat, w_flat = _clark_f32_masked(fp_flat, v_flat, c_flat, am_flat, is_best_of=is_best_of)
        f_w = fw_flat.view(b_sz, m_act).clamp(min=1e-12)
        v_w = vw_flat.view(b_sz, m_act).clamp(min=1e-6)
        w_all = w_flat.view(b_sz, m_act, d_act) * am_in.unsqueeze(1)
        if m_act > 1:
            w_cur, w_nxt = w_all[:, :-1, :], w_all[:, 1:, :]
            v_cur = v_in.transpose(1, 2)[:, :-1, :].clamp(min=1e-6)
            v_nxt = v_in.transpose(1, 2)[:, 1:, :].clamp(min=1e-6)
            v_min = torch.minimum(v_cur.unsqueeze(-2), v_nxt.unsqueeze(-2))
            c_exp = corr_in.unsqueeze(1)
            num = (w_cur.unsqueeze(-1) * w_nxt.unsqueeze(-2) * c_exp * v_cur.unsqueeze(-1) * v_min).sum(dim=(-2, -1))
            cv_cur, cv_nxt = w_cur * v_cur, w_nxt * v_nxt
            den_cur = (cv_cur.unsqueeze(-1) * cv_cur.unsqueeze(-2) * c_exp).sum(dim=(-2, -1)).clamp(min=1e-24)
            den_nxt = (cv_nxt.unsqueeze(-1) * cv_nxt.unsqueeze(-2) * c_exp).sum(dim=(-2, -1)).clamp(min=1e-24)
            r_next_all: torch.Tensor | None = (num / (torch.sqrt(den_cur) * torch.sqrt(den_nxt)).clamp(min=1e-24)).clamp(-0.9999, 0.9999)
        else:
            r_next_all = None

        z_mean = torch.zeros(b_sz, dtype=torch.float32, device=device)
        z_std = torch.ones(b_sz, dtype=torch.float32, device=device)
        surv = torch.ones(b_sz, dtype=torch.float32, device=device)
        fc_list: list[torch.Tensor] = []
        vc_list: list[torch.Tensor] = []
        h_list: list[torch.Tensor] = []
        p_list: list[torch.Tensor] = []
        surv_list: list[torch.Tensor] = []
        surv_prev_list: list[torch.Tensor] = []

        for m in range(m_act):
            fw_m = f_w[:, m]
            vw_m = v_w[:, m]
            if m == 0:
                fc = fw_m
                vc = vw_m
            else:
                fc = fw_m * torch.exp(vw_m * z_mean - 0.5 * (vw_m * vw_m) * (1.0 - z_std * z_std))
                vc = (vw_m * z_std).clamp(min=1e-6)
            fc_list.append(fc)
            vc_list.append(vc)
            is_c_m = is_c_in[:, m] * sm_in[:, m]
            ko_raw, mu_u, var_u = _ko_and_ramp_fwd_only_f32(
                fc, vc, b_ko_in[:, m], ko_sp_in, need_ramp=(m < m_act - 1)
            )
            h_m = is_c_m * ko_raw.clamp(0.0, 1.0 - 1e-7)
            h_list.append(h_m)
            p_list.append(surv * h_m)
            surv_prev_list.append(surv)
            surv_next = surv * (1.0 - h_m)
            surv_list.append(surv_next)
            if m < m_act - 1:
                assert r_next_all is not None and mu_u is not None and var_u is not None
                c_on = is_c_m > 0.5
                z_pm = torch.where(c_on, z_mean + z_std * mu_u, z_mean)
                z_pv = torch.where(c_on, (z_std * z_std * var_u).clamp(min=1e-8), z_std * z_std)
                rn = r_next_all[:, m]
                z_mean = rn * z_pm
                z_std = torch.sqrt((rn * rn * z_pv + (1.0 - rn * rn)).clamp(min=1e-8))
            surv = surv_next

        fc_all = torch.stack(fc_list, dim=1)
        vc_all = torch.stack(vc_list, dim=1)
        h_all = torch.stack(h_list, dim=1)
        force_2d = (force_cpn_in > 0.5).unsqueeze(1) & (is_c_in * sm_in > 0.5)
        eff_b_cpn = torch.where(force_2d, torch.minimum(b_cpn_in, b_ko_in), b_cpn_in).clamp(min=1e-6)
        cpn_sp_v = cpn_sp_in.unsqueeze(1)
        is_q_sp = cpn_sp_v > 1e-5
        bc_lo = torch.minimum(eff_b_cpn, eff_b_cpn - cpn_sp_v).clamp(min=1e-6)
        bc_hi = torch.maximum(eff_b_cpn, eff_b_cpn - cpn_sp_v).clamp(min=1e-6)
        dbc = (bc_hi - bc_lo).clamp(min=1e-7)
        inv_vc = 1.0 / vc_all.clamp(min=1e-6)
        ln_fc_all = torch.log(fc_all.clamp(min=1e-12))
        uc_hi = (torch.log(bc_hi) - ln_fc_all + 0.5 * vc_all * vc_all) * inv_vc
        uc_lo = (torch.log(bc_lo) - ln_fc_all + 0.5 * vc_all * vc_all) * inv_vc
        dphic_tail = _ndtr_f32(-(uc_lo - vc_all)) - _ndtr_f32(-(uc_hi - vc_all))
        dphic_m = torch.where(uc_hi + uc_lo > 0.0, dphic_tail, _ndtr_f32(uc_hi - vc_all) - _ndtr_f32(uc_lo - vc_all))
        q_sp_tail = (fc_all * dphic_tail + bc_hi * _ndtr_f32(-uc_hi) - bc_lo * _ndtr_f32(-uc_lo)) / dbc
        q_sp_main = 1.0 - (bc_hi * _ndtr_f32(uc_hi) - bc_lo * _ndtr_f32(uc_lo) - fc_all * dphic_m) / dbc
        q_sp_raw = torch.where(uc_hi + uc_lo > 0.0, q_sp_tail, q_sp_main)
        uc_pt = (torch.log(eff_b_cpn) - ln_fc_all + 0.5 * vc_all * vc_all) * inv_vc
        q_pt_raw = _ndtr_f32(-uc_pt).clamp(0.0, 1.0)
        q_base = torch.where(is_q_sp, q_sp_raw.clamp(0.0, 1.0), q_pt_raw)
        q_raw_eff = torch.where(force_2d & (h_all > q_base), h_all, q_base)
        q_px = torch.where(is_g_in > 0.5, torch.ones_like(q_raw_eff), q_raw_eff) * sm_in

        if all_sched_act:
            fc_term = fc_list[-1]
            vc_term = vc_list[-1]
            surv_term = surv_list[-1]
            surv_prev_term = surv_prev_list[-1]
            is_c_term = is_c_in[:, -1]
            b_ko_term = b_ko_in[:, -1]
            w_term = w_all[:, -1, :]
        else:
            idx_1 = idx_last_in.unsqueeze(1)
            fc_term = fc_all.gather(1, idx_1).squeeze(1)
            vc_term = vc_all.gather(1, idx_1).squeeze(1)
            surv_term = torch.stack(surv_list, dim=1).gather(1, idx_1).squeeze(1)
            surv_prev_term = torch.stack(surv_prev_list, dim=1).gather(1, idx_1).squeeze(1)
            is_c_term = is_c_in.gather(1, idx_1).squeeze(1)
            b_ko_term = b_ko_in.gather(1, idx_1).squeeze(1)
            w_term = w_all.gather(1, idx_1.unsqueeze(2).expand(b_sz, 1, d_act)).squeeze(1)

        has_ko_term = is_c_term > 0.5
        surv_mult = torch.where(has_ko_term, surv_prev_term, surv_term)
        l_put = surv_mult * _exact_piecewise_terminal_put_tensor_f32(
            fc_term, vc_term, k_strike_in, k_low_in, active_ki_in, b_ki_in, ki_sp_in, is_c_term, b_ko_term, ko_sp_in
        )
        return torch.log(f_w), v_w, torch.stack(p_list, dim=1), q_px, l_put, w_term


def _batched_tier1_fwd_cuda_f32(
    fp_bm: torch.Tensor,
    v_bm: torch.Tensor,
    corr_sym: torch.Tensor,
    amask_64: torch.Tensor,
    smask_64: torch.Tensor,
    b_ko: torch.Tensor,
    b_cpn: torch.Tensor,
    is_call_base: torch.Tensor,
    is_guar_base: torch.Tensor,
    ko_spread_mag: torch.Tensor,
    cpn_spread_mag: torch.Tensor,
    ki_spread_mag: torch.Tensor,
    k_strike: torch.Tensor,
    k_low: torch.Tensor,
    active_ki: torch.Tensor,
    b_ki: torch.Tensor,
    force_cpn_vec: torch.Tensor,
    idx_last: torch.Tensor | None,
    *,
    is_best_of: bool,
    all_sched_act: bool,
    sym5_cache: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDAGraph-cached forward-only Tier-1 proxy kernel for fast CashGamma shifted batches."""
    b_sz, d_act, m_act = fp_bm.shape
    device = fp_bm.device
    use_triton_f64 = bool(_HAS_TRITON_TIER1 and d_act <= 8)
    t_dtype = torch.float64 if use_triton_f64 else torch.float32
    idx_t = torch.zeros(b_sz, dtype=torch.long, device=device) if idx_last is None else idx_last.to(device=device, dtype=torch.long)
    live_inputs = (
        fp_bm.detach().to(t_dtype).contiguous(),
        v_bm.detach().to(t_dtype).contiguous(),
        corr_sym.detach().to(t_dtype).contiguous(),
        amask_64.detach().to(t_dtype).contiguous(),
        smask_64.detach().to(t_dtype).contiguous(),
        b_ko.detach().to(t_dtype).contiguous(),
        b_cpn.detach().to(t_dtype).contiguous(),
        is_call_base.detach().to(t_dtype).contiguous(),
        is_guar_base.detach().to(t_dtype).contiguous(),
        ko_spread_mag.squeeze(1).detach().to(t_dtype).contiguous(),
        cpn_spread_mag.squeeze(1).detach().to(t_dtype).contiguous(),
        ki_spread_mag.squeeze(1).detach().to(t_dtype).contiguous(),
        k_strike.detach().to(t_dtype).contiguous(),
        k_low.detach().to(t_dtype).contiguous(),
        (active_ki.detach().to(torch.float64) if use_triton_f64 else active_ki.detach()).contiguous(),
        b_ki.detach().to(t_dtype).contiguous(),
        force_cpn_vec.detach().to(t_dtype).contiguous(),
        idx_t.contiguous(),
    )
    can_cg = bool(
        use_triton_f64
        and d_act <= 5
        and (b_sz * d_act * m_act) <= 150_000
        and not torch.cuda.is_current_stream_capturing()  # type: ignore[no-untyped-call]
    )
    use_cg_sym5 = False
    use_cg_sym5_blend = False
    use_clark32_5 = False
    tie_bm: torch.Tensor | None = None
    if can_cg and d_act > 1:
        if d_act == 5:
            fp_fl_c = fp_bm.detach().transpose(1, 2).reshape(b_sz * m_act, 5).clamp(min=1e-16)
            lf_c = torch.log(fp_fl_c)
            s_lf_c = torch.sort(lf_c, dim=1).values
            min_gap_c = (s_lf_c[:, 1:] - s_lf_c[:, :-1]).min()
            f_rmax_c = fp_fl_c.max(dim=1).values / fp_fl_c.min(dim=1).values
            flags = torch.stack([
                (amask_64 > 0.5).all(),
                (f_rmax_c < 1.002).all(),
                min_gap_c > 0.062,
            ]).tolist()
            if bool(flags[0]):
                if bool(flags[1]):
                    use_cg_sym5 = True
                elif bool(flags[2]):
                    use_clark32_5 = True
                else:
                    use_cg_sym5_blend = True
        else:
            fp_flat_c = fp_bm.detach().transpose(1, 2).reshape(b_sz * m_act, d_act).clamp(min=1e-16)
            am_flat_c = (amask_64 > 0.5).unsqueeze(1).expand(b_sz, m_act, d_act).reshape(b_sz * m_act, d_act)
            pair_off_c = (am_flat_c.unsqueeze(2) & am_flat_c.unsqueeze(1)) & (
                ~torch.eye(d_act, dtype=torch.bool, device=device).unsqueeze(0)
            )
            dl_sq_c = torch.log(fp_flat_c.unsqueeze(2) / fp_flat_c.unsqueeze(1)) ** 2
            tie_bm_raw = ((dl_sq_c < 0.0035367) & pair_off_c).any(dim=(-2, -1))
            has_tie_c = bool(tie_bm_raw.any().item())
            if has_tie_c:
                tie_bm = tie_bm_raw
    elif use_triton_f64:
        tie_bm = _clark_tie_mask_bm(fp_bm, amask_64)

    can_cg_now = bool(can_cg and (use_cg_sym5 or use_cg_sym5_blend or use_clark32_5 or tie_bm is None))
    cg_tag = (
        "sym5_fwd64"
        if use_cg_sym5
        else (
            "sym5_blend_fwd64"
            if use_cg_sym5_blend
            else ("fwd32c" if use_clark32_5 else ("fwd64" if use_triton_f64 else "fwd32"))
        )
    )
    cache_key = (cg_tag, int(device.index or 0), b_sz, d_act, m_act, is_best_of, all_sched_act)
    do_capture = False
    if can_cg_now:
        with _TIER1_CG_LOCK:
            entry_chk = _TIER1_CG_CACHE.get(cache_key)
            if entry_chk is not None:
                do_capture = True
            else:
                seen = _TIER1_CG_SEEN.get(cache_key, 0)
                if seen >= 1:
                    do_capture = True
                else:
                    if len(_TIER1_CG_SEEN) >= 256:
                        _TIER1_CG_SEEN.clear()
                    _TIER1_CG_SEEN[cache_key] = 1

    if not do_capture:
        if use_triton_f64:
            outs = _eval_tier1_fwd_triton_f64(
                live_inputs,
                is_best_of=is_best_of,
                all_sched_act=all_sched_act,
                tie_bm=tie_bm,
                sym5_cache=sym5_cache,
                force_sym5_only=use_cg_sym5,
                force_sym5_blend_all=use_cg_sym5_blend,
                use_clark32=use_clark32_5,
            )
        else:
            outs = _eval_tier1_fwd_f32_core(
                live_inputs, is_best_of=is_best_of, all_sched_act=all_sched_act
            )
        return tuple(o.to(torch.float64) for o in outs)  # type: ignore[return-value]

    def _cg_core_fwd(
        inputs: tuple[torch.Tensor, ...], *, is_best_of: bool, all_sched_act: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _eval_tier1_fwd_triton_f64(
            inputs,
            is_best_of=is_best_of,
            all_sched_act=all_sched_act,
            force_sym5_only=use_cg_sym5,
            force_sym5_blend_all=use_cg_sym5_blend,
            use_clark32=use_clark32_5,
        )

    core_fn = _cg_core_fwd
    with _TIER1_CG_LOCK:
        entry = _TIER1_CG_CACHE.get(cache_key)
        if entry is not None:
            _TIER1_CG_CACHE.move_to_end(cache_key)
        else:
            static_in = tuple(t.clone() for t in live_inputs)
            cap_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            cap_stream.wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(cap_stream):
                for _ in range(2):
                    _ = core_fn(static_in, is_best_of=is_best_of, all_sched_act=all_sched_act)
            torch.cuda.current_stream(device=device).wait_stream(cap_stream)
            cg = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
            with torch.cuda.graph(cg):
                static_out = core_fn(static_in, is_best_of=is_best_of, all_sched_act=all_sched_act)
            entry = {"lock": threading.Lock(), "graph": cg, "static_in": static_in, "static_out": static_out}
            _TIER1_CG_CACHE[cache_key] = entry
            evicted = False
            while len(_TIER1_CG_CACHE) > _TIER1_CG_MAXSIZE:
                _, old_entry = _TIER1_CG_CACHE.popitem(last=False)
                del old_entry
                evicted = True
            if evicted:
                torch.cuda.empty_cache()

    entry_lock: threading.Lock = entry["lock"]
    with entry_lock:
        for s_t, l_t in zip(entry["static_in"], live_inputs):
            s_t.copy_(l_t)
        entry["graph"].replay()
        o0, o1, o2, o3, o4, o5 = (o.to(torch.float64).clone() for o in entry["static_out"])
        return o0, o1, o2, o3, o4, o5


def _batched_w1_head_and_vjp(
    d_haz_f64: torch.Tensor,
    d_cpn_f64: torch.Tensor,
    d_put_f64: torch.Tensor,
    sched_64: torch.Tensor,
    smask_64: torch.Tensor,
    glob_64: torch.Tensor,
    *,
    no_notional_repayment: torch.Tensor | bool,
    autocall_coupon: torch.Tensor | float,
    force_coupons_on_autocall: torch.Tensor | bool,
    is_long_basket: torch.Tensor | bool | None,
    w_fund_act: torch.Tensor | None,
    lower_strike: torch.Tensor | float | None,
    one_star_pv: torch.Tensor | float,
    part_call_pv: torch.Tensor | float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Evaluates W1 head and exact float64 VJP wrt (d_haz, d_cpn, d_put, sched_64, glob_64)."""
    with torch.enable_grad():  # type: ignore[no-untyped-call]
        dh_t = d_haz_f64.detach().requires_grad_(True)
        dc_t = d_cpn_f64.detach().requires_grad_(True)
        dp_t = d_put_f64.detach().requires_grad_(True)
        sf_t = sched_64.detach().requires_grad_(True)
        gf_t = glob_64.detach().requires_grad_(True)

        active_sched = smask_64 > 0.5
        c_call = torch.where(active_sched, sf_t[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)], torch.zeros_like(smask_64)).clamp(0.0, 1.0) * smask_64
        g_cpn = torch.where(active_sched, sf_t[:, :, int(AutocallSchedCol.IS_GUARANTEED_CPN)], torch.zeros_like(smask_64)).clamp(0.0, 1.0) * smask_64
        p_px = torch.where(active_sched, sf_t[:, :, int(AutocallSchedCol.PROXY_KO_PROB)], torch.zeros_like(smask_64)).clamp(0.0, 1.0) * c_call
        q_px = torch.where(active_sched, sf_t[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)], torch.zeros_like(smask_64)).clamp(0.0, 1.0) * smask_64

        cum_p = torch.cumsum(p_px, dim=1)
        sp_prev = (1.0 - (cum_p - p_px)).clamp(min=1e-6)
        h_px = (p_px / sp_prev).clamp(min=1e-6, max=1.0 - 1e-6)
        logit_h = torch.log(h_px / (1.0 - h_px))
        haz = c_call * smask_64 * torch.sigmoid(logit_h + dh_t)
        surv = torch.cumprod(1.0 - haz, dim=1)
        surv_prev = torch.cat([torch.ones_like(surv[:, :1]), surv[:, :-1]], dim=1)
        p_stop = surv_prev * haz

        q_px_c = q_px.clamp(min=1e-6, max=1.0 - 1e-6)
        logit_q = torch.log(q_px_c / (1.0 - q_px_c))
        q_gate = (q_px * q_px) / (q_px * q_px + 1e-10)
        q_pred = q_gate * torch.sigmoid(logit_q + dc_t)
        q_cpn = smask_64 * (g_cpn + (1.0 - g_cpn) * q_pred)

        k_put = gf_t[:, int(AutocallGlobalCol.K_PUT)].clamp(min=0.0)
        if gf_t.shape[-1] > int(AutocallGlobalCol.LOWER_STRIKE):
            k_low = gf_t[:, int(AutocallGlobalCol.LOWER_STRIKE)].clamp(min=0.0)
        else:
            k_low = torch.zeros_like(k_put)
        k_eff = torch.clamp(k_put - k_low, min=0.0)
        l_put_px = gf_t[:, int(AutocallGlobalCol.L_PUT_PROXY)].clamp(min=0.0)
        sp_term = (1.0 - p_px.sum(dim=1)).clamp(min=0.0)
        denom = torch.clamp(k_eff * sp_term, min=1e-12)
        r_px = torch.clamp(l_put_px / denom, min=1e-12, max=1.0 - 1e-12)
        logit_r = torch.log(r_px / (1.0 - r_px))
        u_put = k_eff * surv[:, -1]
        put_loss = u_put * torch.sigmoid(logit_r + dp_t)

        out_64 = AutocallSurrogateOutput(
            stopping_probs=p_stop,
            survival_probs=surv,
            hazard_rates=haz,
            coupon_probs=q_cpn,
            put_loss=put_loss,
            delta_hazard_logits=dh_t,
            delta_coupon_logits=dc_t,
            delta_put_loss=dp_t,
        )
        asm_opt_kwargs: dict[str, Any] = {}
        if "is_long_basket" in _ASSEMBLE_SURR_PARAMS and is_long_basket is not None:
            asm_opt_kwargs["is_long_basket"] = is_long_basket
        if "w_fund_per_period" in _ASSEMBLE_SURR_PARAMS and w_fund_act is not None:
            asm_opt_kwargs["w_fund_per_period"] = w_fund_act
        if "lower_strike" in _ASSEMBLE_SURR_PARAMS and lower_strike is not None:
            asm_opt_kwargs["lower_strike"] = lower_strike
        if "one_star_pv" in _ASSEMBLE_SURR_PARAMS:
            asm_opt_kwargs["one_star_pv"] = one_star_pv
        if "part_call_pv" in _ASSEMBLE_SURR_PARAMS:
            asm_opt_kwargs["part_call_pv"] = part_call_pv
        pv_64 = assemble_surrogate_product_value(
            out_64,
            sf_t,
            smask_64,
            gf_t,
            no_notional_repayment=no_notional_repayment,
            autocall_coupon=autocall_coupon,
            force_coupons_on_autocall=force_coupons_on_autocall,
            pv_in_transit=0.0,
            **asm_opt_kwargs,
        )
        g_dh, g_dc, g_dp, g_sf, g_gf = torch.autograd.grad(
            pv_64.sum(),
            (dh_t, dc_t, dp_t, sf_t, gf_t),
            allow_unused=True,
        )
    return (
        pv_64.detach(),
        p_stop.detach(),
        surv.detach(),
        q_cpn.detach(),
        put_loss.detach(),
        g_dh.detach() if g_dh is not None else torch.zeros_like(d_haz_f64),
        g_dc.detach() if g_dc is not None else torch.zeros_like(d_cpn_f64),
        g_dp.detach() if g_dp is not None else torch.zeros_like(d_put_f64),
        g_sf.detach() if g_sf is not None else torch.zeros_like(sched_64),
        g_gf.detach() if g_gf is not None else torch.zeros_like(glob_64),
    )


_SURR_CG_LOCK = threading.Lock()
_SURR_CG_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_SURR_CG_SEEN: dict[tuple[Any, ...], int] = {}
_SURR_CG_MAXSIZE = 4


def _eval_w1_f64_core(
    d_haz_f64: torch.Tensor,
    d_cpn_f64: torch.Tensor,
    d_put_f64: torch.Tensor,
    sched_64: torch.Tensor,
    smask_64: torch.Tensor,
    glob_64: torch.Tensor,
    no_notional_vec: torch.Tensor,
    ac_cpn_vec: torch.Tensor,
    force_cpn_vec: torch.Tensor,
    is_long_vec: torch.Tensor,
    w_fund_2d: torch.Tensor,
    lower_k_vec: torch.Tensor,
    one_star_vec: torch.Tensor,
    part_call_vec: torch.Tensor,
    *,
    has_low_k: bool,
    w_len: int,
) -> tuple[torch.Tensor, ...]:
    """Zero-host-sync, CUDAGraph-safe float64 W1 stopping-time head + VJP."""
    b_sz, m_act, _ = sched_64.shape
    device = sched_64.device
    if _HAS_TRITON_TIER1 and sched_64.is_cuda and b_sz > 0 and m_act > 0:
        d_sched = int(sched_64.shape[2])
        d_glob = int(glob_64.shape[1])
        pv_o = torch.empty(b_sz, dtype=torch.float64, device=device)
        pstop_o = torch.empty((b_sz, m_act), dtype=torch.float64, device=device)
        surv_o = torch.empty((b_sz, m_act), dtype=torch.float64, device=device)
        qcpn_o = torch.empty((b_sz, m_act), dtype=torch.float64, device=device)
        pl_o = torch.empty(b_sz, dtype=torch.float64, device=device)
        gdh_o = torch.empty((b_sz, m_act), dtype=torch.float64, device=device)
        gdc_o = torch.empty((b_sz, m_act), dtype=torch.float64, device=device)
        gdp_o = torch.empty(b_sz, dtype=torch.float64, device=device)
        gsf_o = torch.empty((b_sz, m_act, d_sched), dtype=torch.float64, device=device)
        ggf_o = torch.empty((b_sz, d_glob), dtype=torch.float64, device=device)
        sc = torch.empty((13, m_act, b_sz), dtype=torch.float64, device=device)
        nonot_c = no_notional_vec.contiguous()
        wf_c = w_fund_2d[:, :w_len].contiguous() if w_len > 0 else nonot_c
        grid = (triton.cdiv(b_sz, 32),)
        _w1_fwd_bwd_kernel64[grid](
            d_haz_f64.contiguous(),
            d_cpn_f64.contiguous(),
            d_put_f64.contiguous(),
            sched_64.contiguous(),
            smask_64.contiguous(),
            glob_64.contiguous(),
            nonot_c,
            ac_cpn_vec.contiguous(),
            force_cpn_vec.contiguous(),
            is_long_vec.contiguous(),
            wf_c,
            lower_k_vec.contiguous(),
            one_star_vec.contiguous(),
            part_call_vec.contiguous(),
            pv_o, pstop_o, surv_o, qcpn_o, pl_o,
            gdh_o, gdc_o, gdp_o, gsf_o, ggf_o, sc,
            b_sz, M=m_act, D_SCHED=d_sched, D_GLOB=d_glob,
            HAS_LOW_K=has_low_k, W_LEN=w_len, BLOCK_B=32, num_warps=1,
        )
        return pv_o, pstop_o, surv_o, qcpn_o, pl_o, gdh_o, gdc_o, gdp_o, gsf_o, ggf_o
    with torch.enable_grad():  # type: ignore[no-untyped-call]
        dh_t = d_haz_f64.detach().requires_grad_(True)
        dc_t = d_cpn_f64.detach().requires_grad_(True)
        dp_t = d_put_f64.detach().requires_grad_(True)
        sf_t = sched_64.detach().requires_grad_(True)
        gf_t = glob_64.detach().requires_grad_(True)

        active_sched = smask_64 > 0.5
        m_sched = active_sched.to(dtype=torch.float64)
        has_any = active_sched.any(dim=1)
        sched_clean = torch.where(active_sched.unsqueeze(-1), sf_t, torch.zeros_like(sf_t))
        df_pay = sched_clean[:, :, int(AutocallSchedCol.DF_PAY)]
        cpn_rate = sched_clean[:, :, int(AutocallSchedCol.COUPON_RATE)]

        c_call = torch.where(active_sched, sf_t[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)], torch.zeros_like(m_sched)).clamp(0.0, 1.0) * m_sched
        g_cpn = torch.where(active_sched, sf_t[:, :, int(AutocallSchedCol.IS_GUARANTEED_CPN)], torch.zeros_like(m_sched)).clamp(0.0, 1.0) * m_sched
        p_px = torch.where(active_sched, sf_t[:, :, int(AutocallSchedCol.PROXY_KO_PROB)], torch.zeros_like(m_sched)).clamp(0.0, 1.0) * c_call
        q_px = torch.where(active_sched, sf_t[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)], torch.zeros_like(m_sched)).clamp(0.0, 1.0) * m_sched

        cum_p = torch.cumsum(p_px, dim=1)
        sp_prev = (1.0 - (cum_p - p_px)).clamp(min=1e-6)
        h_px = (p_px / sp_prev).clamp(min=1e-6, max=1.0 - 1e-6)
        haz = c_call * m_sched * torch.sigmoid(torch.log(h_px / (1.0 - h_px)) + dh_t)

        # Product loop avoids torch.cumprod CUDAGraph capture errors on CUDA
        surv_list: list[torch.Tensor] = []
        surv_prev_list: list[torch.Tensor] = []
        s_cur = torch.ones(b_sz, dtype=torch.float64, device=device)
        for m in range(m_act):
            surv_prev_list.append(s_cur)
            s_cur = s_cur * (1.0 - haz[:, m])
            surv_list.append(s_cur)
        surv = torch.stack(surv_list, dim=1)
        surv_prev = torch.stack(surv_prev_list, dim=1)
        p_stop = surv_prev * haz

        q_px_c = q_px.clamp(min=1e-6, max=1.0 - 1e-6)
        q_gate = (q_px * q_px) / (q_px * q_px + 1e-10)
        q_cpn = m_sched * (g_cpn + (1.0 - g_cpn) * q_gate * torch.sigmoid(torch.log(q_px_c / (1.0 - q_px_c)) + dc_t))

        idx_grid = torch.arange(m_act, device=device).unsqueeze(0)
        last_active_idx = torch.where(active_sched, idx_grid, torch.zeros_like(idx_grid)).argmax(dim=1, keepdim=True)
        df_term = torch.where(has_any, df_pay.gather(1, last_active_idx).squeeze(1), torch.zeros_like(no_notional_vec))
        surv_term = surv.gather(1, last_active_idx).squeeze(1)

        pv_notional = (1.0 - no_notional_vec) * ((df_pay * p_stop * m_sched).sum(dim=1) + df_term * surv_term)

        k_put = gf_t[:, int(AutocallGlobalCol.K_PUT)].clamp(min=0.0)
        g_put = gf_t[:, int(AutocallGlobalCol.G_PUT)]
        c_mem_0 = gf_t[:, int(AutocallGlobalCol.C_MEM_0)]
        is_memory = gf_t[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)].clamp(0.0, 1.0)
        mem_buf = c_mem_0 * is_memory
        pv_coupon = torch.zeros(b_sz, dtype=torch.float64, device=device)
        for m in range(m_act):
            c_eff_m = cpn_rate[:, m] + mem_buf
            q_m = q_cpn[:, m]
            q_forced = torch.maximum(q_m, p_stop[:, m] / surv_prev[:, m].clamp(min=1e-12))
            eff_q_m = q_m + force_cpn_vec * (q_forced - q_m)
            pv_coupon = pv_coupon + df_pay[:, m] * (surv_prev[:, m] * eff_q_m * c_eff_m + p_stop[:, m] * ac_cpn_vec) * m_sched[:, m]
            mem_buf = torch.where(active_sched[:, m], is_memory * (1.0 - eff_q_m) * c_eff_m, mem_buf)

        if has_low_k:
            k_eff = torch.clamp(k_put - torch.clamp(lower_k_vec, min=0.0), min=0.0)
        else:
            k_eff = k_put
        l_put_px = gf_t[:, int(AutocallGlobalCol.L_PUT_PROXY)].clamp(min=0.0)
        sp_term = (1.0 - p_px.sum(dim=1)).clamp(min=0.0)
        r_px = torch.clamp(l_put_px / torch.clamp(k_eff * sp_term, min=1e-12), min=1e-12, max=1.0 - 1e-12)
        put_loss = k_eff * surv[:, -1] * torch.sigmoid(torch.log(r_px / (1.0 - r_px)) + dp_t)
        eff_put_loss = torch.minimum(put_loss, k_eff * surv_term) if has_low_k else put_loss

        pv_option = (2.0 * is_long_vec - 1.0) * df_term * g_put * eff_put_loss + one_star_vec + part_call_vec
        fund_pv = (surv_prev[:, :w_len] * w_fund_2d[:, :w_len] * m_sched[:, :w_len]).sum(dim=1) if w_len > 0 else torch.zeros_like(pv_option)
        pv_64 = torch.where(has_any, pv_notional + pv_coupon + pv_option - fund_pv, torch.zeros_like(pv_notional))

        g_dh, g_dc, g_dp, g_sf, g_gf = torch.autograd.grad(
            pv_64.sum(),
            (dh_t, dc_t, dp_t, sf_t, gf_t),
            allow_unused=True,
        )
    return (
        pv_64.detach(),
        p_stop.detach(),
        surv.detach(),
        q_cpn.detach(),
        put_loss.detach(),
        g_dh.detach() if g_dh is not None else torch.zeros_like(d_haz_f64),
        g_dc.detach() if g_dc is not None else torch.zeros_like(d_cpn_f64),
        g_dp.detach() if g_dp is not None else torch.zeros_like(d_put_f64),
        g_sf.detach() if g_sf is not None else torch.zeros_like(sched_64),
        g_gf.detach() if g_gf is not None else torch.zeros_like(glob_64),
    )


def _expand_mod_vec(val: Any, b_sz: int, dtype: torch.dtype, device: torch.device, default: float = 0.0) -> torch.Tensor:
    if val is None:
        return torch.full((b_sz,), default, dtype=dtype, device=device)
    if isinstance(val, (bool, int, float)):
        return torch.full((b_sz,), float(val), dtype=dtype, device=device)
    t = torch.as_tensor(val, dtype=dtype, device=device)
    if t.ndim == 0:
        return t.expand(b_sz).contiguous()
    return t.reshape(-1).expand(b_sz).contiguous()


def _eval_surr_w1_fwdbwd_core(
    surrogate: torch.nn.Module,
    live_inputs: tuple[torch.Tensor, ...],
    *,
    has_low_k: bool,
    w_len: int,
    fwd_res_fn: Any = None,
) -> tuple[torch.Tensor, ...]:
    """Single-graph surrogate forward_residuals + float64 W1 head + surrogate VJP."""
    (
        node_64, edge_64, sched_64, glob_64, amask_64, smask_64,
        no_notional_vec, ac_cpn_vec, force_cpn_vec, is_long_vec,
        w_fund_2d, lower_k_vec, one_star_vec, part_call_vec,
    ) = live_inputs
    surr_d_node = int(getattr(surrogate, "d_node", node_64.shape[-1]))
    surr_d_global = int(getattr(surrogate, "d_global", glob_64.shape[-1]))
    fn_res = fwd_res_fn if callable(fwd_res_fn) else surrogate.forward_residuals
    with torch.enable_grad():  # type: ignore[no-untyped-call]
        node_in = node_64[:, :, :surr_d_node].detach().float().requires_grad_(True)
        edge_in = edge_64.detach().float().requires_grad_(True)
        sched_in = sched_64.detach().float().requires_grad_(True)
        glob_in = glob_64[:, :surr_d_global].detach().float().requires_grad_(True)
        d_haz_t, d_cpn_t, d_put_t = fn_res(  # type: ignore[operator]
            node_in,
            edge_in,
            amask_64.float(),
            sched_in,
            smask_64.float(),
            glob_in,
            require_second_order_sdpa=True,
        )
        pv_64, p_stop_64, surv_64, q_cpn_64, put_loss_64, g_dh_64, g_dc_64, g_dp_64, g_sf_w1, g_gf_w1 = _eval_w1_f64_core(
            d_haz_t.double(),
            d_cpn_t.double(),
            d_put_t.double().reshape(node_64.shape[0]),
            sched_64,
            smask_64,
            glob_64[:, :surr_d_global],
            no_notional_vec,
            ac_cpn_vec,
            force_cpn_vec,
            is_long_vec,
            w_fund_2d,
            lower_k_vec,
            one_star_vec,
            part_call_vec,
            has_low_k=has_low_k,
            w_len=w_len,
        )
        g_node_act, g_edge_act, g_sched_act, g_glob_act = torch.autograd.grad(
            (d_haz_t, d_cpn_t, d_put_t),
            (node_in, edge_in, sched_in, glob_in),
            grad_outputs=(
                g_dh_64.to(dtype=d_haz_t.dtype),
                g_dc_64.to(dtype=d_cpn_t.dtype),
                g_dp_64.to(dtype=d_put_t.dtype),
            ),
            allow_unused=True,
        )
    return (
        pv_64,
        p_stop_64,
        surv_64,
        q_cpn_64,
        put_loss_64,
        g_sf_w1,
        g_gf_w1,
        g_node_act.detach() if g_node_act is not None else torch.zeros_like(node_in),
        g_edge_act.detach() if g_edge_act is not None else torch.zeros_like(edge_in),
        g_sched_act.detach() if g_sched_act is not None else torch.zeros_like(sched_in),
        g_glob_act.detach() if g_glob_act is not None else torch.zeros_like(glob_in),
    )


def _batched_surr_w1_fwdbwd_cuda(
    surrogate: torch.nn.Module,
    node_64: torch.Tensor,
    edge_64: torch.Tensor,
    sched_64: torch.Tensor,
    glob_64: torch.Tensor,
    amask_64: torch.Tensor,
    smask_64: torch.Tensor,
    *,
    no_notional_repayment: Any,
    autocall_coupon: Any,
    force_coupons_on_autocall: Any,
    is_long_basket: Any,
    w_fund_act: torch.Tensor | None,
    lower_strike: Any,
    one_star_pv: Any,
    part_call_pv: Any,
    use_tf32: bool,
) -> tuple[torch.Tensor, ...]:
    """Combined surrogate forward_residuals + float64 W1 + backward VJP."""
    b_sz = node_64.shape[0]
    m_act = sched_64.shape[1]
    device = node_64.device
    surr_d_global = int(getattr(surrogate, "d_global", glob_64.shape[-1]))
    g_sub = glob_64[:, :surr_d_global]

    if is_long_basket is None:
        if g_sub.shape[-1] > int(AutocallGlobalCol.IS_LONG_BASKET):
            is_long_vec = (g_sub[:, int(AutocallGlobalCol.IS_LONG_BASKET)] > 0.5).to(dtype=torch.float64)
        else:
            is_long_vec = torch.zeros(b_sz, dtype=torch.float64, device=device)
    else:
        is_long_vec = _expand_mod_vec(is_long_basket, b_sz, torch.float64, device, 0.0)

    if lower_strike is None:
        has_low_k = g_sub.shape[-1] > int(AutocallGlobalCol.LOWER_STRIKE)
        lower_k_vec = g_sub[:, int(AutocallGlobalCol.LOWER_STRIKE)].detach() if has_low_k else torch.zeros(b_sz, dtype=torch.float64, device=device)
    else:
        has_low_k = True
        lower_k_vec = _expand_mod_vec(lower_strike, b_sz, torch.float64, device, 0.0)

    if w_fund_act is not None:
        w_f = torch.as_tensor(w_fund_act, dtype=torch.float64, device=device)
        if w_f.ndim == 1:
            w_f = w_f.unsqueeze(0)
        w_len = min(m_act, int(w_f.shape[1]))
        w_fund_2d = w_f[:, :w_len].expand(b_sz, w_len).contiguous()
    else:
        w_len = 0
        w_fund_2d = torch.zeros((b_sz, 1), dtype=torch.float64, device=device)

    live_inputs = (
        node_64.detach(),
        edge_64.detach(),
        sched_64.detach(),
        glob_64.detach(),
        amask_64.detach(),
        smask_64.detach(),
        _expand_mod_vec(no_notional_repayment, b_sz, torch.float64, device, 0.0),
        _expand_mod_vec(autocall_coupon, b_sz, torch.float64, device, 0.0),
        _expand_mod_vec(force_coupons_on_autocall, b_sz, torch.float64, device, 0.0),
        is_long_vec.detach(),
        w_fund_2d.detach(),
        lower_k_vec.detach(),
        _expand_mod_vec(one_star_pv, b_sz, torch.float64, device, 0.0),
        _expand_mod_vec(part_call_pv, b_sz, torch.float64, device, 0.0),
    )

    d_act = int(node_64.shape[1])
    surr_d_node = int(getattr(surrogate, "d_node", node_64.shape[-1]))
    p_ver = int(getattr(next(surrogate.parameters(), None), "_version", 0))
    use_cg = bool(
        device.type == "cuda"
        and (b_sz * d_act * m_act) <= 150_000
        and not torch.cuda.is_current_stream_capturing()  # type: ignore[no-untyped-call]
    )
    if not use_cg:
        return _eval_surr_w1_fwdbwd_core(surrogate, live_inputs, has_low_k=has_low_k, w_len=w_len)
    cache_key = (
        id(surrogate), p_ver, int(device.index or 0), b_sz, d_act, m_act,
        surr_d_node, surr_d_global, bool(use_tf32), bool(has_low_k), int(w_len),
    )
    do_capture = False
    with _SURR_CG_LOCK:
        entry = _SURR_CG_CACHE.get(cache_key)
        if entry is not None:
            _SURR_CG_CACHE.move_to_end(cache_key)
            do_capture = True
        else:
            seen = _SURR_CG_SEEN.get(cache_key, 0)
            if seen >= 1:
                do_capture = True
            else:
                if len(_SURR_CG_SEEN) >= 256:
                    _SURR_CG_SEEN.clear()
                _SURR_CG_SEEN[cache_key] = 1
    if not do_capture:
        return _eval_surr_w1_fwdbwd_core(surrogate, live_inputs, has_low_k=has_low_k, w_len=w_len)
    with _SURR_CG_LOCK:
        entry = _SURR_CG_CACHE.get(cache_key)
        if entry is not None:
            _SURR_CG_CACHE.move_to_end(cache_key)
        else:
            static_in = tuple(t.clone() for t in live_inputs)
            cap_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            cap_stream.wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(cap_stream):
                for _ in range(2):
                    _ = _eval_surr_w1_fwdbwd_core(surrogate, static_in, has_low_k=has_low_k, w_len=w_len)
            torch.cuda.current_stream(device=device).wait_stream(cap_stream)
            cg = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
            with torch.cuda.graph(cg):
                static_out = _eval_surr_w1_fwdbwd_core(surrogate, static_in, has_low_k=has_low_k, w_len=w_len)
            entry = {"lock": threading.Lock(), "graph": cg, "static_in": static_in, "static_out": static_out}
            _SURR_CG_CACHE[cache_key] = entry
            evicted = False
            while len(_SURR_CG_CACHE) > _SURR_CG_MAXSIZE:
                _, old_entry = _SURR_CG_CACHE.popitem(last=False)
                del old_entry
                evicted = True
            if evicted:
                torch.cuda.empty_cache()
    entry_lock: threading.Lock = entry["lock"]
    with entry_lock:
        for s_t, l_t in zip(entry["static_in"], live_inputs):
            s_t.copy_(l_t)
        entry["graph"].replay()
        return tuple(o.clone() for o in entry["static_out"])


_RECON_CG_LOCK = threading.Lock()
_RECON_CG_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_RECON_CG_SEEN: dict[tuple[Any, ...], int] = {}
_RECON_CG_MAXSIZE = 8


def _reconstruct_d5_cg_core(
    node_64: torch.Tensor,
    v_m_raw: torch.Tensor,
    corr_sym_64: torch.Tensor,
    amask_64: torch.Tensor,
    tau_all: torch.Tensor,
    tau_term: torch.Tensor,
    log_fw_base: torch.Tensor,
    vw_base_in: torch.Tensor,
    *,
    is_best: bool,
    has_div: bool,
    has_vw: bool,
    use_sym5_only: bool = False,
    no_ties: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-host-sync CUDAGraph-safe forward_perf + cdl reconstruction for D=5 all-active."""
    b_sz, d_act, m_act = v_m_raw.shape
    bm = b_sz * m_act
    log_m0 = node_64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)]
    log_carry = node_64[:, :, int(AutocallNodeCol.LOG_FWD_CARRY_1Y)]
    log_dlev = node_64[:, :, int(AutocallNodeCol.CASH_DIV_LEVERAGE)].clamp(-10.0, 10.0)
    rq = log_carry + log_dlev / tau_term
    fp_nodiv = torch.exp((log_m0.unsqueeze(2) + rq.unsqueeze(2) * tau_all.unsqueeze(1)).clamp(-20.0, 20.0)) * amask_64.unsqueeze(2)
    fw_base = torch.exp(log_fw_base.clamp(-20.0, 20.0))
    c_flat = corr_sym_64.unsqueeze(1).expand(b_sz, m_act, d_act, d_act).reshape(bm, d_act, d_act)
    v_flat = v_m_raw.transpose(1, 2).reshape(bm, d_act).clamp(min=1e-6)
    am_flat = amask_64.unsqueeze(1).expand(b_sz, m_act, d_act).reshape(bm, d_act)

    def _run_m_d5(fp_m_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_safe = fp_m_3d.transpose(1, 2).reshape(bm, d_act).clamp(min=1e-12)
        if use_sym5_only:
            f_sym5, v_sym5, _, _, _ = _eval_sym5_fwd_triton(
                f_safe.contiguous(), v_flat.contiguous(), c_flat.contiguous(), is_best_of=is_best
            )
            return f_sym5.view(b_sz, m_act).clamp(min=1e-16), v_sym5.view(b_sz, m_act).clamp(min=1e-12)
        ord_f = torch.argsort(-f_safe if is_best else f_safe, dim=1, stable=True)
        f_o = torch.gather(f_safe, 1, ord_f).contiguous()
        v_o = torch.gather(v_flat, 1, ord_f).contiguous()
        m_o = torch.gather(am_flat, 1, ord_f).contiguous()
        c_o = torch.gather(
            torch.gather(c_flat, 1, ord_f.unsqueeze(2).expand(bm, d_act, d_act)),
            2,
            ord_f.unsqueeze(1).expand(bm, d_act, d_act),
        ).contiguous()
        fw_fl, vw_fl, _, _ = _eval_clark32_fwd_triton(
            f_o, v_o, c_o, m_o, is_best_of=is_best, save_sc=False
        )
        if no_ties:
            return fw_fl.view(b_sz, m_act).clamp(min=1e-16), vw_fl.view(b_sz, m_act).clamp(min=1e-12)
        f_sym5, v_sym5, _, _, _ = _eval_sym5_fwd_triton(
            f_safe.contiguous(), v_flat.contiguous(), c_flat.contiguous(), is_best_of=is_best
        )
        g_sep5 = _sym5_gsep(f_safe)
        w_tie5 = 1.0 - g_sep5
        fw_bl = g_sep5 * fw_fl + w_tie5 * f_sym5
        vw_bl = g_sep5 * vw_fl + w_tie5 * v_sym5
        return fw_bl.view(b_sz, m_act).clamp(min=1e-16), vw_bl.view(b_sz, m_act).clamp(min=1e-12)

    if has_div:
        fp_div = fp_nodiv * torch.exp(-log_dlev).unsqueeze(2)
        fw_nd, vw_nd = _run_m_d5(fp_nodiv)
        fw_d, vw_d = _run_m_d5(fp_div)
        denom_fw = fw_nd - fw_d
        denom_fw_safe = torch.where(denom_fw.abs() > 1e-10, denom_fw, torch.ones_like(denom_fw))
        alpha_fw = torch.where(
            denom_fw.abs() > 1e-10,
            ((fw_nd - fw_base) / denom_fw_safe).clamp(0.0, 1.0),
            torch.zeros_like(fw_base),
        )
        if has_vw:
            vw_span = vw_nd - vw_d
            use_vw = vw_span.abs() > 1e-6
            lo = torch.zeros_like(vw_base_in)
            hi = torch.ones_like(vw_base_in)
            f_lo = vw_nd - vw_base_in
            f_hi = vw_d - vw_base_in
            al = ((vw_nd - vw_base_in) / torch.where(use_vw, vw_span, torch.ones_like(vw_span))).clamp(0.0, 1.0)
            for _ in range(5):
                fp_c = torch.exp(torch.log(fp_nodiv.clamp(min=1e-16)) - al.unsqueeze(1) * log_dlev.unsqueeze(2)) * amask_64.unsqueeze(2)
                _, vw_c = _run_m_d5(fp_c)
                f_c = vw_c - vw_base_in
                same_lo = (f_c * f_lo) >= 0.0
                lo = torch.where(same_lo, al, lo)
                f_lo = torch.where(same_lo, f_c, f_lo)
                hi = torch.where(~same_lo, al, hi)
                f_hi = torch.where(~same_lo, f_c, f_hi)
                f_diff = f_hi - f_lo
                al_sec = torch.where(
                    f_diff.abs() > 1e-12,
                    lo - f_lo * (hi - lo) / torch.where(f_diff.abs() > 1e-12, f_diff, torch.ones_like(f_diff)),
                    0.5 * (lo + hi),
                )
                al = al_sec.clamp(0.0, 1.0)
            alpha = torch.where(use_vw, al, alpha_fw)
        else:
            alpha = alpha_fw
        alpha[:, -1] = 1.0
        fp_est = torch.exp(torch.log(fp_nodiv.clamp(min=1e-16)) - alpha.unsqueeze(1) * log_dlev.unsqueeze(2)) * amask_64.unsqueeze(2)
        fw_est, _ = _run_m_d5(fp_est)
        fp_est = fp_est * (fw_base / fw_est).unsqueeze(1)
        cdl_est = torch.exp(alpha.unsqueeze(1) * log_dlev.unsqueeze(2)) * amask_64.unsqueeze(2)
        return fp_est, cdl_est

    fw_nd, _ = _run_m_d5(fp_nodiv)
    fp_scaled = fp_nodiv * (fw_base / fw_nd).unsqueeze(1)
    cdl_ones = torch.ones_like(fp_scaled) * amask_64.unsqueeze(2)
    return fp_scaled, cdl_ones


def _reconstruct_period_forwards_and_cdl(
    node_64: torch.Tensor,
    v_m_raw: torch.Tensor,
    corr_sym_64: torch.Tensor,
    amask_64: torch.Tensor,
    tau_all: torch.Tensor,
    tau_term: torch.Tensor,
    log_fw_base: torch.Tensor,
    call_idx_str: str,
    vw_base: torch.Tensor | None = None,
    *,
    term_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstructs per-period forward_perf and cash_div_leverage from STULZ_FWD_W when omitted."""
    b_sz, d_act, m_act = v_m_raw.shape
    log_m0 = node_64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)]
    log_carry = node_64[:, :, int(AutocallNodeCol.LOG_FWD_CARRY_1Y)]
    log_dlev = node_64[:, :, int(AutocallNodeCol.CASH_DIV_LEVERAGE)].clamp(-10.0, 10.0)
    if m_act == 1 and not term_only:
        fp_1 = torch.exp((log_m0 + log_carry * tau_term).clamp(-20.0, 20.0)).unsqueeze(2) * amask_64.unsqueeze(2)
        cdl_1 = torch.exp(log_dlev).unsqueeze(2) * amask_64.unsqueeze(2)
        return fp_1, cdl_1

    use_triton_f64 = bool(
        _HAS_TRITON_TIER1 and v_m_raw.is_cuda and d_act <= 8 and call_idx_str in ("WorstOf", "BestOf")
    )
    is_best = call_idx_str == "BestOf"
    if (
        not term_only
        and use_triton_f64
        and d_act == 5
        and (b_sz * d_act * m_act) <= 150_000
        and not torch.cuda.is_current_stream_capturing()  # type: ignore[no-untyped-call]
    ):
        rq_chk = log_carry + log_dlev / tau_term
        lf_nd = (log_m0.unsqueeze(2) + rq_chk.unsqueeze(2) * tau_all.unsqueeze(1)).clamp(-20.0, 20.0)
        lf_d = lf_nd - log_dlev.unsqueeze(2)
        sym5_only_chk = (
            ((lf_nd.max(dim=1).values - lf_nd.min(dim=1).values) < 0.001).all()
            & ((lf_d.max(dim=1).values - lf_d.min(dim=1).values) < 0.001).all()
        )
        s_nd = torch.sort(lf_nd, dim=1).values
        s_d = torch.sort(lf_d, dim=1).values
        no_ties_chk = (
            ((s_nd[:, 1:] - s_nd[:, :-1]).min() > 0.062)
            & ((s_d[:, 1:] - s_d[:, :-1]).min() > 0.062)
        )
        recon_fl = torch.stack([
            (log_dlev.abs() > 1e-10).any(),
            (amask_64 > 0.5).all(),
            sym5_only_chk,
            no_ties_chk,
        ]).tolist()
        has_div = bool(recon_fl[0])
        all_act = bool(recon_fl[1])
        use_sym5_only_rc = bool(recon_fl[2])
        no_ties_rc = bool(recon_fl[3])
    else:
        has_div = bool((log_dlev.abs() > 1e-10).any().item())
        all_act = bool((amask_64 > 0.5).all().item())
        use_sym5_only_rc = False
        no_ties_rc = False
    if (
        not term_only
        and use_triton_f64
        and d_act == 5
        and all_act
        and (b_sz * d_act * m_act) <= 150_000
        and not torch.cuda.is_current_stream_capturing()  # type: ignore[no-untyped-call]
    ):
        device = v_m_raw.device
        has_vw = vw_base is not None
        vw_in = vw_base.detach().contiguous() if vw_base is not None else torch.empty((b_sz, m_act), dtype=torch.float64, device=device)
        live_in = (
            node_64.detach().contiguous(),
            v_m_raw.detach().contiguous(),
            corr_sym_64.detach().contiguous(),
            amask_64.detach().contiguous(),
            tau_all.detach().contiguous(),
            tau_term.detach().contiguous(),
            log_fw_base.detach().contiguous(),
            vw_in,
        )
        r_key = (int(device.index or 0), b_sz, d_act, m_act, is_best, has_div, has_vw, use_sym5_only_rc, no_ties_rc)
        do_cap = False
        with _RECON_CG_LOCK:
            r_entry = _RECON_CG_CACHE.get(r_key)
            if r_entry is not None:
                _RECON_CG_CACHE.move_to_end(r_key)
                do_cap = True
            else:
                seen = _RECON_CG_SEEN.get(r_key, 0)
                if seen >= 1:
                    do_cap = True
                else:
                    if len(_RECON_CG_SEEN) >= 256:
                        _RECON_CG_SEEN.clear()
                    _RECON_CG_SEEN[r_key] = 1
        if not do_cap:
            return _reconstruct_d5_cg_core(
                *live_in, is_best=is_best, has_div=has_div, has_vw=has_vw, use_sym5_only=use_sym5_only_rc, no_ties=no_ties_rc
            )
        with _RECON_CG_LOCK:
            r_entry = _RECON_CG_CACHE.get(r_key)
            if r_entry is not None:
                _RECON_CG_CACHE.move_to_end(r_key)
            else:
                static_in = tuple(t.clone() for t in live_in)
                cap_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
                cap_stream.wait_stream(torch.cuda.current_stream(device=device))
                with torch.cuda.stream(cap_stream):
                    for _ in range(2):
                        _ = _reconstruct_d5_cg_core(
                            *static_in, is_best=is_best, has_div=has_div, has_vw=has_vw, use_sym5_only=use_sym5_only_rc, no_ties=no_ties_rc
                        )
                torch.cuda.current_stream(device=device).wait_stream(cap_stream)
                cg = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
                with torch.cuda.graph(cg):
                    static_out = _reconstruct_d5_cg_core(
                        *static_in, is_best=is_best, has_div=has_div, has_vw=has_vw, use_sym5_only=use_sym5_only_rc, no_ties=no_ties_rc
                    )
                r_entry = {"lock": threading.Lock(), "graph": cg, "static_in": static_in, "static_out": static_out}
                _RECON_CG_CACHE[r_key] = r_entry
                evicted = False
                while len(_RECON_CG_CACHE) > _RECON_CG_MAXSIZE:
                    _, old_e = _RECON_CG_CACHE.popitem(last=False)
                    del old_e
                    evicted = True
                if evicted:
                    torch.cuda.empty_cache()
        r_lock: threading.Lock = r_entry["lock"]
        with r_lock:
            for s_t, l_t in zip(r_entry["static_in"], live_in):
                s_t.copy_(l_t)
            r_entry["graph"].replay()
            return r_entry["static_out"][0].clone(), r_entry["static_out"][1].clone()

    rq = log_carry + log_dlev / tau_term
    fp_nodiv = torch.exp((log_m0.unsqueeze(2) + rq.unsqueeze(2) * tau_all.unsqueeze(1)).clamp(-20.0, 20.0)) * amask_64.unsqueeze(2)
    fw_base = torch.exp(log_fw_base.clamp(-20.0, 20.0))

    use_dyn32 = d_act > 5
    use_clark_f32_cuda = bool(
        not use_triton_f64 and v_m_raw.is_cuda and call_idx_str in ("WorstOf", "BestOf")
    )
    c_flat = corr_sym_64.unsqueeze(1).expand(b_sz, m_act, d_act, d_act).reshape(b_sz * m_act, d_act, d_act)
    am_flat = amask_64.unsqueeze(1).expand(b_sz, m_act, d_act).reshape(b_sz * m_act, d_act)
    v_flat = v_m_raw.transpose(1, 2).reshape(b_sz * m_act, d_act)

    def _run_moments(fp_m_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fp_fl = fp_m_3d.transpose(1, 2).reshape(b_sz * m_act, d_act)
        if use_triton_f64:
            bm = b_sz * m_act
            act_b = am_flat > 0.5
            f_safe = torch.where(act_b, fp_fl.clamp(min=1e-12), 1.0)
            v_safe = torch.where(act_b, v_flat.clamp(min=1e-6), 0.1)
            if use_dyn32 and all_act and d_act in (6, 7, 8):
                g_sd = _sym_gsep_dyn(f_safe, am_flat)
                if float(g_sd.max().item()) < 1e-15:
                    f_sd, v_sd, _ = _eval_sym_dyn32_fwd(
                        f_safe.contiguous(), v_safe.contiguous(), c_flat.contiguous(), am_flat.contiguous(), is_best_of=is_best
                    )
                    return f_sd.view(b_sz, m_act).clamp(min=1e-16), v_sd.view(b_sz, m_act).clamp(min=1e-12)
            if d_act > 1:
                sort_key = torch.where(act_b, -f_safe if is_best else f_safe, 1e12)
                ord_f = torch.argsort(sort_key, dim=1, stable=True)
                f_o = torch.gather(f_safe, 1, ord_f).contiguous()
                v_o = torch.gather(v_safe, 1, ord_f).contiguous()
                m_o = torch.gather(am_flat, 1, ord_f).contiguous()
                c_o = torch.gather(
                    torch.gather(c_flat, 1, ord_f.unsqueeze(2).expand(bm, d_act, d_act)),
                    2,
                    ord_f.unsqueeze(1).expand(bm, d_act, d_act),
                ).contiguous()
            else:
                f_o = f_safe.contiguous()
                v_o = v_safe.contiguous()
                m_o = am_flat.contiguous()
                c_o = f_o
            if use_dyn32:
                fw_fl, vw_fl, _, _ = _eval_clark32_fwd_triton(
                    f_o, v_o, c_o, m_o, is_best_of=is_best, save_sc=False
                )
            else:
                sc_rho = torch.empty((d_act, bm), dtype=torch.float64, device=v_m_raw.device)
                fw_fl = torch.empty(bm, dtype=torch.float64, device=v_m_raw.device)
                vw_fl = torch.empty(bm, dtype=torch.float64, device=v_m_raw.device)
                wo_fl = torch.empty((bm, d_act), dtype=torch.float64, device=v_m_raw.device)
                _clark_fwd_kernel64[(triton.cdiv(bm, 32),)](
                    f_o, v_o, c_o, m_o, fw_fl, vw_fl, wo_fl, f_o, f_o, sc_rho,
                    bm, D=d_act, IS_BEST=is_best, SAVE_SC=False, BLOCK_BM=32, num_warps=1,
                )
            if d_act == 5 and all_act:
                f_sym5, v_sym5, _, _, _ = _eval_sym5_fwd_triton(
                    f_safe.contiguous(), v_safe.contiguous(), c_flat.contiguous(), is_best_of=is_best
                )
                g_sep5 = _sym5_gsep(f_safe)
                w_tie5 = 1.0 - g_sep5
                fw_bl = g_sep5 * fw_fl + w_tie5 * f_sym5
                vw_bl = g_sep5 * vw_fl + w_tie5 * v_sym5
                return fw_bl.view(b_sz, m_act).clamp(min=1e-16), vw_bl.view(b_sz, m_act).clamp(min=1e-12)
            tie_bm = _clark_tie_mask_bm(fp_m_3d, amask_64)
            if tie_bm is not None:
                tie_idx = tie_bm.nonzero(as_tuple=True)[0]
                tie5_idx: torch.Tensor | None = None
                tie_dyn_idx: torch.Tensor | None = None
                tie_rem_idx: torch.Tensor | None = tie_idx
                if d_act == 5:
                    if all_act:
                        tie5_idx = tie_idx
                        tie_rem_idx = None
                    else:
                        is_act5 = act_b.index_select(0, tie_idx).all(dim=1)
                        if bool(is_act5.any().item()):
                            tie5_idx = tie_idx[is_act5]
                            tie_rem_idx = tie_idx[~is_act5]
                elif d_act in (2, 3, 6, 7, 8):
                    if all_act:
                        tie_dyn_idx = tie_idx
                        tie_rem_idx = None
                    else:
                        is_full = act_b.index_select(0, tie_idx).all(dim=1)
                        if bool(is_full.any().item()):
                            tie_dyn_idx = tie_idx[is_full]
                            tie_rem_idx = tie_idx[~is_full]
                if tie5_idx is not None:
                    f_t5 = f_safe.index_select(0, tie5_idx).contiguous()
                    v_t5 = v_safe.index_select(0, tie5_idx).contiguous()
                    c_t5 = c_flat.index_select(0, tie5_idx).contiguous()
                    f_sym5, v_sym5, _, _, _ = _eval_sym5_fwd_triton(f_t5, v_t5, c_t5, is_best_of=is_best)
                    g_sep5 = _sym5_gsep(f_t5)
                    w_tie5 = 1.0 - g_sep5
                    fw_fl = fw_fl.clone()
                    vw_fl = vw_fl.clone()
                    fw_fl.index_copy_(0, tie5_idx, g_sep5 * fw_fl.index_select(0, tie5_idx) + w_tie5 * f_sym5)
                    vw_fl.index_copy_(0, tie5_idx, g_sep5 * vw_fl.index_select(0, tie5_idx) + w_tie5 * v_sym5)
                if tie_dyn_idx is not None:
                    f_td = f_safe.index_select(0, tie_dyn_idx).contiguous()
                    v_td = v_safe.index_select(0, tie_dyn_idx).contiguous()
                    c_td = c_flat.index_select(0, tie_dyn_idx).contiguous()
                    m_td = am_flat.index_select(0, tie_dyn_idx).contiguous()
                    f_sd, v_sd, _ = _eval_sym_dyn32_fwd(f_td, v_td, c_td, m_td, is_best_of=is_best)
                    g_sd = _sym_gsep_dyn(f_td, m_td)
                    w_sd = 1.0 - g_sd
                    f_fd = fw_fl.index_select(0, tie_dyn_idx)
                    v_fd = vw_fl.index_select(0, tie_dyn_idx)
                    f_b = g_sd * f_fd + w_sd * f_sd
                    m2_b = g_sd * (f_fd * f_fd * torch.exp(v_fd * v_fd)) + w_sd * (f_sd * f_sd * torch.exp(v_sd * v_sd))
                    v_b = torch.sqrt(torch.clamp(torch.log(torch.clamp(m2_b / (f_b * f_b).clamp(min=1e-30), min=1.0)), min=1e-16))
                    fw_fl = fw_fl.clone()
                    vw_fl = vw_fl.clone()
                    fw_fl.index_copy_(0, tie_dyn_idx, f_b)
                    vw_fl.index_copy_(0, tie_dyn_idx, v_b)
                if tie_rem_idx is not None and int(tie_rem_idx.numel()) > 0:
                    fw_fl = fw_fl.clone()
                    vw_fl = vw_fl.clone()
                    for s_t in range(0, int(tie_rem_idx.numel()), 512):
                        sub_idx = tie_rem_idx[s_t : s_t + 512]
                        fw_tie, vw_tie, _ = stulz_margrabe_moments_with_weights(
                            fp_fl.index_select(0, sub_idx),
                            v_flat.index_select(0, sub_idx),
                            c_flat.index_select(0, sub_idx),
                            am_flat.index_select(0, sub_idx),
                            monitoring_index=call_idx_str,
                        )
                        fw_fl.index_copy_(0, sub_idx, fw_tie)
                        vw_fl.index_copy_(0, sub_idx, vw_tie)
            return fw_fl.view(b_sz, m_act).clamp(min=1e-16), vw_fl.view(b_sz, m_act).clamp(min=1e-12)
        if use_clark_f32_cuda:
            fw_fl, vw_fl, _ = _clark_f32_masked(
                fp_fl.float(), v_flat.float(), c_flat.float(), am_flat.float(), is_best_of=is_best
            )
            return fw_fl.double().view(b_sz, m_act).clamp(min=1e-16), vw_fl.double().view(b_sz, m_act).clamp(min=1e-12)
        fw_fl, vw_fl, _ = stulz_margrabe_moments_with_weights(
            fp_fl, v_flat, c_flat, am_flat, monitoring_index=call_idx_str
        )
        return fw_fl.view(b_sz, m_act).clamp(min=1e-16), vw_fl.view(b_sz, m_act).clamp(min=1e-12)

    if term_only:
        fp_t = torch.exp((log_m0 + log_carry * tau_term).clamp(-20.0, 20.0)).unsqueeze(2) * amask_64.unsqueeze(2)
        fw_t, _ = _run_moments(fp_t)
        fp_t = fp_t * (fw_base / fw_t).unsqueeze(1)
        cdl_t = (torch.exp(log_dlev) if has_div else torch.ones_like(log_dlev)).unsqueeze(2) * amask_64.unsqueeze(2)
        return fp_t, cdl_t

    if has_div:
        fp_div = fp_nodiv * torch.exp(-log_dlev).unsqueeze(2)
        fw_nd, vw_nd = _run_moments(fp_nodiv)
        fw_d, vw_d = _run_moments(fp_div)
        denom_fw = fw_nd - fw_d
        denom_fw_safe = torch.where(denom_fw.abs() > 1e-10, denom_fw, torch.ones_like(denom_fw))
        alpha_fw = torch.where(
            denom_fw.abs() > 1e-10,
            ((fw_nd - fw_base) / denom_fw_safe).clamp(0.0, 1.0),
            torch.zeros_like(fw_base),
        )
        if vw_base is not None:
            # STULZ_VOL_W is scale-invariant, isolating non-uniform cash dividends from rate-curve slope.
            vw_span = vw_nd - vw_d
            use_vw = vw_span.abs() > 1e-6
            if bool(use_vw.any().item()):
                lo = torch.zeros_like(vw_base)
                hi = torch.ones_like(vw_base)
                f_lo = vw_nd - vw_base
                f_hi = vw_d - vw_base
                al = ((vw_nd - vw_base) / torch.where(use_vw, vw_span, torch.ones_like(vw_span))).clamp(0.0, 1.0)
                for _ in range(5):
                    fp_c = torch.exp(torch.log(fp_nodiv.clamp(min=1e-16)) - al.unsqueeze(1) * log_dlev.unsqueeze(2)) * amask_64.unsqueeze(2)
                    _, vw_c = _run_moments(fp_c)
                    f_c = vw_c - vw_base
                    same_lo = (f_c * f_lo) >= 0.0
                    lo = torch.where(same_lo, al, lo)
                    f_lo = torch.where(same_lo, f_c, f_lo)
                    hi = torch.where(~same_lo, al, hi)
                    f_hi = torch.where(~same_lo, f_c, f_hi)
                    f_diff = f_hi - f_lo
                    al_sec = torch.where(f_diff.abs() > 1e-12, lo - f_lo * (hi - lo) / torch.where(f_diff.abs() > 1e-12, f_diff, torch.ones_like(f_diff)), 0.5 * (lo + hi))
                    al = al_sec.clamp(0.0, 1.0)
                alpha = torch.where(use_vw, al, alpha_fw)
            else:
                alpha = alpha_fw
        else:
            alpha = alpha_fw
        alpha[:, -1] = 1.0
        fp_est = torch.exp(torch.log(fp_nodiv.clamp(min=1e-16)) - alpha.unsqueeze(1) * log_dlev.unsqueeze(2)) * amask_64.unsqueeze(2)
        fw_est, _ = _run_moments(fp_est)
        fp_est = fp_est * (fw_base / fw_est).unsqueeze(1)
        cdl_est = torch.exp(alpha.unsqueeze(1) * log_dlev.unsqueeze(2)) * amask_64.unsqueeze(2)
        return fp_est, cdl_est

    fw_nd, _ = _run_moments(fp_nodiv)
    fp_scaled = fp_nodiv * (fw_base / fw_nd).unsqueeze(1)
    cdl_ones = torch.ones_like(fp_scaled) * amask_64.unsqueeze(2)
    return fp_scaled, cdl_ones


def _batched_tier1_proxies(
    fp_bm: torch.Tensor,
    v_bm: torch.Tensor,
    corr_sym: torch.Tensor,
    amask_64: torch.Tensor,
    smask_64: torch.Tensor,
    b_ko: torch.Tensor,
    b_cpn: torch.Tensor,
    is_call_base: torch.Tensor,
    is_guar_base: torch.Tensor,
    ko_spread_mag: torch.Tensor,
    cpn_spread_mag: torch.Tensor,
    ki_spread_mag: torch.Tensor,
    k_strike: torch.Tensor,
    k_low: torch.Tensor,
    active_ki: torch.Tensor,
    b_ki: torch.Tensor,
    *,
    force_coupons: bool | torch.Tensor,
    call_idx_str: str,
    all_sched_act: bool,
    idx_last: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes batched Markovian Tier-1 proxies (log_fw, vw, p, q, l_put, w_term)."""
    b_sz, d_act, m_act = fp_bm.shape
    device = fp_bm.device

    fp_flat = fp_bm.transpose(1, 2).reshape(b_sz * m_act, d_act)
    v_flat = v_bm.transpose(1, 2).reshape(b_sz * m_act, d_act)
    c_flat = corr_sym.unsqueeze(1).expand(b_sz, m_act, d_act, d_act).reshape(b_sz * m_act, d_act, d_act)
    am_flat = amask_64.unsqueeze(1).expand(b_sz, m_act, d_act).reshape(b_sz * m_act, d_act)
    fw_flat, vw_flat, w_flat = stulz_margrabe_moments_with_weights(
        fp_flat, v_flat, c_flat, am_flat, monitoring_index=call_idx_str,
    )
    f_w = fw_flat.view(b_sz, m_act).clamp(min=1e-16)
    v_w = vw_flat.view(b_sz, m_act).clamp(min=1e-12)
    w_call_all = w_flat.view(b_sz, m_act, d_act) * amask_64.unsqueeze(1)

    ko_sp_vec = -ko_spread_mag.squeeze(1)
    cpn_sp_vec = -cpn_spread_mag.squeeze(1)
    ko_sp_scalar = float(ko_sp_vec[0].item())
    if isinstance(force_coupons, torch.Tensor):
        force_2d = (force_coupons.to(device=device, dtype=torch.float64).reshape(-1, 1) > 0.5)
        has_any_force = bool(force_2d.any().item())
    else:
        force_2d = None
        has_any_force = bool(force_coupons)

    z_mean = torch.zeros(b_sz, dtype=torch.float64, device=device)
    z_std = torch.ones(b_sz, dtype=torch.float64, device=device)
    surv_cur = torch.ones(b_sz, dtype=torch.float64, device=device)

    p_list: list[torch.Tensor] = []
    q_list: list[torch.Tensor] = []
    surv_list: list[torch.Tensor] = []
    surv_prev_list: list[torch.Tensor] = []
    f_cond_list: list[torch.Tensor] = []
    v_cond_list: list[torch.Tensor] = []

    for m in range(m_act):
        fw_m = f_w[:, m]
        vw_m = v_w[:, m]
        if m == 0:
            f_cond_m = fw_m
            v_cond_m = vw_m
        else:
            f_cond_m = fw_m * torch.exp(vw_m * z_mean - 0.5 * (vw_m * vw_m) * (1.0 - z_std * z_std))
            v_cond_m = (vw_m * z_std).clamp(min=1e-12)

        is_c_m = is_call_base[:, m] * smask_64[:, m]
        b_ko_m = b_ko[:, m]
        any_call_m = bool((is_c_m > 0.5).any())
        if any_call_m:
            ko_raw_m = _digital_call_spread_prob_torch(f_cond_m, v_cond_m, b_ko_m, ko_sp_vec)
            h_m = is_c_m * ko_raw_m.clamp(0.0, 1.0 - 1e-7)
        else:
            h_m = torch.zeros(b_sz, dtype=torch.float64, device=device)

        p_m = surv_cur * h_m
        surv_next = surv_cur * (1.0 - h_m)

        if has_any_force:
            f_mask_m = (force_2d.squeeze(1) & (is_c_m > 0.5)) if force_2d is not None else (is_c_m > 0.5)
            eff_cb_m = torch.where(f_mask_m, torch.minimum(b_cpn[:, m], b_ko_m), b_cpn[:, m])
        else:
            f_mask_m = None
            eff_cb_m = b_cpn[:, m]
        q_raw_m = _digital_call_spread_prob_torch(f_cond_m, v_cond_m, eff_cb_m, cpn_sp_vec)
        if has_any_force and f_mask_m is not None:
            q_raw_m = torch.where(f_mask_m, torch.maximum(q_raw_m, h_m), q_raw_m)
        q_m = torch.where(is_guar_base[:, m] > 0.5, torch.ones_like(q_raw_m), q_raw_m) * smask_64[:, m]

        p_list.append(p_m)
        q_list.append(q_m)
        surv_prev_list.append(surv_cur)
        surv_list.append(surv_next)
        f_cond_list.append(f_cond_m)
        v_cond_list.append(v_cond_m)

        if m < m_act - 1:
            if any_call_m:
                mu_u, var_u = _ramp_survival_post_moments_lognormal_torch(f_cond_m, v_cond_m, b_ko_m, ko_sp_scalar)
                z_post_mean = torch.where(is_c_m > 0.5, z_mean + z_std * mu_u, z_mean)
                z_post_var = torch.where(is_c_m > 0.5, (z_std * z_std * var_u).clamp(min=1e-12), z_std * z_std)
            else:
                z_post_mean = z_mean
                z_post_var = z_std * z_std
            r_next = _euler_next_corr_torch(
                fp_bm[:, :, m],
                v_bm[:, :, m].clamp(min=1e-12),
                w_call_all[:, m, :],
                fp_bm[:, :, m + 1],
                v_bm[:, :, m + 1].clamp(min=1e-12),
                w_call_all[:, m + 1, :],
                corr_sym,
                False,
            )
            z_mean = r_next * z_post_mean
            z_std = torch.sqrt((r_next * r_next * z_post_var + (1.0 - r_next * r_next)).clamp(min=1e-12))

        surv_cur = surv_next

    p_proxy = torch.stack(p_list, dim=1)
    q_proxy = torch.stack(q_list, dim=1)
    if all_sched_act:
        f_cond_term = f_cond_list[-1]
        v_cond_term = v_cond_list[-1]
        surv_term = surv_list[-1]
        surv_prev_term = surv_prev_list[-1]
        is_call_term = is_call_base[:, -1]
        b_ko_term = b_ko[:, -1]
        w_term = w_call_all[:, -1, :]
    else:
        assert idx_last is not None
        idx_1 = idx_last.unsqueeze(1)
        f_cond_term = torch.stack(f_cond_list, dim=1).gather(1, idx_1).squeeze(1)
        v_cond_term = torch.stack(v_cond_list, dim=1).gather(1, idx_1).squeeze(1)
        surv_term = torch.stack(surv_list, dim=1).gather(1, idx_1).squeeze(1)
        surv_prev_term = torch.stack(surv_prev_list, dim=1).gather(1, idx_1).squeeze(1)
        is_call_term = is_call_base.gather(1, idx_1).squeeze(1)
        b_ko_term = b_ko.gather(1, idx_1).squeeze(1)
        w_term = w_call_all.gather(1, idx_1.unsqueeze(2).expand(b_sz, 1, d_act)).squeeze(1)

    uniform_contract = (
        float((k_strike.max() - k_strike.min()).item()) < 1e-12
        and float((k_low.max() - k_low.min()).item()) < 1e-12
        and float((b_ki.max() - b_ki.min()).item()) < 1e-12
        and bool((active_ki == active_ki[0]).all().item())
        and bool(((is_call_term > 0.5) == (is_call_term[0] > 0.5)).all().item())
        and float((b_ko_term.max() - b_ko_term.min()).item()) < 1e-12
    )
    if uniform_contract:
        use_direct_ko = bool(float(is_call_term[0].item()) > 0.5)
        surv_mult = surv_prev_term if use_direct_ko else surv_term
        l_put_proxy = surv_mult * _exact_piecewise_terminal_put_torch(
            f_cond_term,
            v_cond_term,
            False,
            float(k_strike[0].item()),
            float(k_low[0].item()),
            bool(active_ki[0].item()),
            float(b_ki[0].item()),
            -float(ki_spread_mag[0].item()),
            use_direct_ko,
            float(b_ko_term[0].item()),
            ko_sp_scalar,
        )
    else:
        l_put_proxy = surv_term * _batched_put_spread_loss_torch(
            f_cond_term,
            v_cond_term,
            k_strike,
            k_low,
            active_ki,
            b_ki,
            ko_spread_mag.squeeze(1),
        )
    return torch.log(f_w.clamp(min=1e-12)), v_w, p_proxy, q_proxy, l_put_proxy, w_term


_E2E_CG_LOCK = threading.Lock()
_E2E_CG_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_E2E_CG_SEEN: dict[tuple[Any, ...], int] = {}
_E2E_CG_MAXSIZE = 8


def _compute_e2e_d5_cg_core(
    surrogate: MaskedEQAutocallSurrogate,
    live_in: tuple[torch.Tensor, ...],
    *,
    d_max: int,
    d_node: int,
    m_max: int,
    is_best: bool,
    has_div: bool,
    force_cpn_f: float,
    has_spots: bool,
    has_ref_spots: bool,
    use_sym5_only: bool = False,
    no_ties: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Single-graph end-to-end CUDA surrogate Greeks core for small D=5 Markov batches."""
    node_64, edge_64, sched_64, glob_64, amask_64, smask_64, sp_in_t, ref_in_t = live_in
    batch_size, d_act, m_act = int(node_64.shape[0]), 5, int(sched_64.shape[1])
    device = node_64.device
    surr_d_global = int(getattr(surrogate, "d_global", glob_64.shape[-1]))
    b_eff = 2 * batch_size
    h_cg = 7.4e-4

    iv_short_node = node_64[:, :, int(AutocallNodeCol.IV_SHORT_ATM)].clamp(min=1e-4)
    iv_term_node = node_64[:, :, int(AutocallNodeCol.IV_TERM_ATM)].clamp(min=1e-4)
    tau_all = sched_64[:, :, int(AutocallSchedCol.TAU_OBS)].clamp(min=1e-8)
    tau_short = tau_all[:, :1]
    tau_term = tau_all[:, -1:].clamp(min=1e-6)
    v_i_term = (iv_term_node * torch.sqrt(tau_term)).clamp(min=1e-6)

    log_fw_base = sched_64[:, :, int(AutocallSchedCol.STULZ_FWD_W)].clamp(-20.0, 20.0)
    v_w_base = sched_64[:, :, int(AutocallSchedCol.STULZ_VOL_W)].clamp(min=1e-6)
    b_ko = torch.exp(sched_64[:, :, int(AutocallSchedCol.LOG_KO_BARRIER)].clamp(-20.0, 20.0))
    b_cpn = torch.exp(sched_64[:, :, int(AutocallSchedCol.LOG_CPN_BARRIER)].clamp(-20.0, 20.0))
    is_call_base = sched_64[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)] * smask_64
    is_guar_base = sched_64[:, :, int(AutocallSchedCol.IS_GUARANTEED_CPN)]
    ko_spread_mag = glob_64[:, int(AutocallGlobalCol.ABS_KO_SPREAD) : int(AutocallGlobalCol.ABS_KO_SPREAD) + 1]
    cpn_spread_mag = glob_64[:, int(AutocallGlobalCol.ABS_CPN_SPREAD) : int(AutocallGlobalCol.ABS_CPN_SPREAD) + 1]
    ki_spread_mag = (
        glob_64[:, int(AutocallGlobalCol.ABS_KI_SPREAD) : int(AutocallGlobalCol.ABS_KI_SPREAD) + 1]
        if glob_64.shape[-1] > int(AutocallGlobalCol.ABS_KI_SPREAD)
        else ko_spread_mag
    )
    k_strike = glob_64[:, int(AutocallGlobalCol.K_PUT)]
    b_ki = glob_64[:, int(AutocallGlobalCol.B_KI)]
    has_ki = glob_64[:, int(AutocallGlobalCol.HAS_KI_FLAG)]
    i_ki_past = glob_64[:, int(AutocallGlobalCol.I_KI_PAST)]
    active_ki = (has_ki > 0.5) & (i_ki_past < 0.5) & (b_ki > 0.0)
    has_low_k = glob_64.shape[-1] > int(AutocallGlobalCol.LOWER_STRIKE)
    k_low = glob_64[:, int(AutocallGlobalCol.LOWER_STRIKE)].clamp(min=0.0) if has_low_k else torch.zeros_like(k_strike)

    eye_d_64 = torch.eye(d_act, dtype=torch.float64, device=device).unsqueeze(0)
    pair_mask_64 = amask_64.unsqueeze(1) * amask_64.unsqueeze(2) * (1.0 - eye_d_64)
    corr_sym_64 = (
        0.5 * (edge_64[:, :, :, int(AutocallEdgeCol.CORR_RHO)] + edge_64[:, :, :, int(AutocallEdgeCol.CORR_RHO)].transpose(1, 2))
        * (1.0 - eye_d_64)
        + eye_d_64
    )
    sqrt_tau_m = torch.sqrt(tau_all)
    span_t = (tau_term - tau_short).clamp(min=1e-8)
    alpha_m = ((tau_all - tau_short) / span_t).clamp(0.0, 1.0)
    v_m_raw = (
        (1.0 - alpha_m).unsqueeze(1) * iv_short_node.unsqueeze(2)
        + alpha_m.unsqueeze(1) * iv_term_node.unsqueeze(2)
    ) * sqrt_tau_m.unsqueeze(1)

    if m_act == 1:
        log_m0_1 = node_64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)]
        log_carry_1 = node_64[:, :, int(AutocallNodeCol.LOG_FWD_CARRY_1Y)]
        log_dlev_1 = node_64[:, :, int(AutocallNodeCol.CASH_DIV_LEVERAGE)].clamp(-10.0, 10.0)
        fp_bm_raw = torch.exp((log_m0_1 + log_carry_1 * tau_term).clamp(-20.0, 20.0)).unsqueeze(2) * amask_64.unsqueeze(2)
        cdl_bm = torch.exp(log_dlev_1).unsqueeze(2) * amask_64.unsqueeze(2)
    else:
        fp_bm_raw, cdl_bm = _reconstruct_d5_cg_core(
            node_64, v_m_raw, corr_sym_64, amask_64, tau_all, tau_term, log_fw_base, v_w_base,
            is_best=is_best, has_div=has_div, has_vw=True, use_sym5_only=use_sym5_only, no_ties=no_ties,
        )

    eps_shifts = torch.cat(
        [
            torch.full((batch_size,), h_cg, dtype=torch.float64, device=device),
            torch.full((batch_size,), -h_cg, dtype=torch.float64, device=device),
        ],
        dim=0,
    )
    one_plus_u = (1.0 + eps_shifts).view(b_eff, 1)
    cdl_bm_2 = torch.cat([cdl_bm, cdl_bm], dim=0)
    inv_lev_bm = 1.0 / cdl_bm_2.clamp(min=1.0)
    escrow_frac_bm = (one_plus_u.unsqueeze(2) - (1.0 - inv_lev_bm)).clamp(min=1e-6)
    log_fp_shift_bm = torch.log(escrow_frac_bm / inv_lev_bm.clamp(min=1e-6))
    log_fp_shift_term = log_fp_shift_bm[:, :, -1]

    node_2 = torch.cat([node_64, node_64], dim=0)
    edge_2 = torch.cat([edge_64, edge_64], dim=0)
    sched_2 = torch.cat([sched_64, sched_64], dim=0)
    glob_2 = torch.cat([glob_64, glob_64], dim=0)
    amask_2 = torch.cat([amask_64, amask_64], dim=0)
    smask_2 = torch.cat([smask_64, smask_64], dim=0)
    tau_term_2 = torch.cat([tau_term, tau_term], dim=0)
    v_i_term_2 = torch.cat([v_i_term, v_i_term], dim=0)
    sqrt_tau_m_2 = torch.cat([sqrt_tau_m, sqrt_tau_m], dim=0)
    alpha_m_2 = torch.cat([alpha_m, alpha_m], dim=0)
    iv_term_node_2 = torch.cat([iv_term_node, iv_term_node], dim=0)
    v_m_raw_2 = torch.cat([v_m_raw, v_m_raw], dim=0)
    corr_sym_2 = torch.cat([corr_sym_64, corr_sym_64], dim=0)
    pair_mask_2 = torch.cat([pair_mask_64, pair_mask_64], dim=0)
    b_ko_2 = torch.cat([b_ko, b_ko], dim=0)
    b_cpn_2 = torch.cat([b_cpn, b_cpn], dim=0)
    is_call_2 = torch.cat([is_call_base, is_call_base], dim=0)
    is_guar_2 = torch.cat([is_guar_base, is_guar_base], dim=0)
    ko_sp_2 = torch.cat([ko_spread_mag, ko_spread_mag], dim=0)
    cpn_sp_2 = torch.cat([cpn_spread_mag, cpn_spread_mag], dim=0)
    ki_sp_2 = torch.cat([ki_spread_mag, ki_spread_mag], dim=0)
    k_strike_2 = torch.cat([k_strike, k_strike], dim=0)
    k_low_2 = torch.cat([k_low, k_low], dim=0)
    active_ki_2 = torch.cat([active_ki, active_ki], dim=0)
    b_ki_2 = torch.cat([b_ki, b_ki], dim=0)

    log_m0_u = node_2[:, :, int(AutocallNodeCol.LOG_MONEYNESS)] + torch.log(one_plus_u)
    log_div_lev_u = torch.log((one_plus_u / escrow_frac_bm[:, :, -1]).clamp(min=1.0))
    log_carry_u = (
        node_2[:, :, int(AutocallNodeCol.LOG_FWD_CARRY_1Y)]
        + (log_fp_shift_term - torch.log(one_plus_u)) / tau_term_2
    )
    dist_put_u = node_2[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)] + log_fp_shift_term / v_i_term_2
    node_2[:, :, int(AutocallNodeCol.LOG_MONEYNESS)] = log_m0_u
    node_2[:, :, int(AutocallNodeCol.CASH_DIV_LEVERAGE)] = log_div_lev_u
    node_2[:, :, int(AutocallNodeCol.LOG_FWD_CARRY_1Y)] = log_carry_u
    node_2[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)] = dist_put_u
    log_fp_term_2 = log_m0_u + log_carry_u * tau_term_2
    edge_2[:, :, :, int(AutocallEdgeCol.LOG_PERF_DIFF)] = torch.sqrt(
        (log_fp_term_2.unsqueeze(2) - log_fp_term_2.unsqueeze(1)) ** 2 + 1e-18
    ) * pair_mask_2

    fp_bm_eff = torch.cat([fp_bm_raw, fp_bm_raw], dim=0) * torch.exp(log_fp_shift_bm)
    fcpn_64 = torch.full((b_eff,), force_cpn_f, dtype=torch.float64, device=device)
    idx_dummy = torch.zeros(b_eff, dtype=torch.long, device=device)
    t1_fwd_in = (
        fp_bm_eff, v_m_raw_2, corr_sym_2, amask_2, smask_2, b_ko_2, b_cpn_2,
        is_call_2, is_guar_2, ko_sp_2.squeeze(1), cpn_sp_2.squeeze(1), ki_sp_2.squeeze(1),
        k_strike_2, k_low_2, active_ki_2.to(torch.float64), b_ki_2, fcpn_64, idx_dummy,
    )
    lfw_u, vw_u, p_u, q_u, lput_u, w_term_u = _eval_tier1_fwd_triton_f64(
        t1_fwd_in,
        is_best_of=is_best,
        all_sched_act=True,
        force_sym5_only=use_sym5_only,
        force_sym5_blend_all=(not use_sym5_only) and (not no_ties),
        use_clark32=no_ties,
    )
    if int(getattr(surrogate, "d_node", d_node)) > int(AutocallNodeCol.PROXY_DELTA_WEIGHT):
        node_2[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)] = w_term_u
    sched_2[:, :, int(AutocallSchedCol.STULZ_FWD_W)] = lfw_u
    sched_2[:, :, int(AutocallSchedCol.STULZ_VOL_W)] = vw_u
    sched_2[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] = p_u
    sched_2[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)] = q_u
    glob_2[:, int(AutocallGlobalCol.L_PUT_PROXY)] = lput_u

    cdl_bm_u = (one_plus_u.unsqueeze(2) / escrow_frac_bm.clamp(min=1e-6)).clamp(min=1.0) * amask_2.unsqueeze(2)
    cdl_term_eff = cdl_bm_u[:, :, -1]

    zeros_b = torch.zeros(b_eff, dtype=torch.float64, device=device)
    is_long_vec = (
        (glob_2[:, int(AutocallGlobalCol.IS_LONG_BASKET)] > 0.5).to(torch.float64)
        if surr_d_global > int(AutocallGlobalCol.IS_LONG_BASKET)
        else zeros_b
    )
    w_fund_2d = torch.zeros((b_eff, 1), dtype=torch.float64, device=device)
    w1_in = (
        node_2.detach(), edge_2.detach(), sched_2.detach(), glob_2.detach(),
        amask_2.detach(), smask_2.detach(),
        zeros_b, zeros_b, fcpn_64, is_long_vec, w_fund_2d, k_low_2.detach(), zeros_b, zeros_b,
    )
    (
        pv_64, p_stop_64, surv_64, q_cpn_64, put_loss_64,
        g_sched_w1, g_glob_w1, g_node_act, g_edge_act, g_sched_act, g_glob_act,
    ) = _eval_surr_w1_fwdbwd_core(surrogate, w1_in, has_low_k=has_low_k, w_len=0)

    g_node_64 = g_node_act.to(torch.float64)
    if g_node_64.shape[-1] < d_node:
        g_node_64 = torch.nn.functional.pad(g_node_64, (0, d_node - g_node_64.shape[-1]))
    g_edge_64 = g_edge_act.to(torch.float64)
    g_sched_64 = g_sched_act.to(torch.float64) + g_sched_w1
    g_glob_64 = g_glob_act.to(torch.float64) + g_glob_w1

    t1_vjp_in = t1_fwd_in + (
        g_sched_64[:, :, int(AutocallSchedCol.STULZ_FWD_W)] * smask_2,
        g_sched_64[:, :, int(AutocallSchedCol.STULZ_VOL_W)] * smask_2,
        g_sched_64[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] * smask_2,
        g_sched_64[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)] * smask_2,
        g_glob_64[:, int(AutocallGlobalCol.L_PUT_PROXY)],
        g_node_64[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)] * amask_2,
    )
    g_fp_bm, g_v_bm, g_corr_bm = _eval_tier1_vjp_triton_f64(
        t1_vjp_in,
        is_best_of=is_best,
        all_sched_act=True,
        force_sym5_only=use_sym5_only,
        force_sym5_blend_all=(not use_sym5_only) and (not no_ties),
        use_clark32=no_ties,
    )
    t1_d_log_s = (g_fp_bm * fp_bm_eff * cdl_bm_u).sum(dim=2)
    d_pv_d_sig_m = torch.where(
        (amask_2 > 0.5).unsqueeze(2) & (smask_2 > 0.5).unsqueeze(1),
        g_v_bm * sqrt_tau_m_2.unsqueeze(1),
        0.0,
    )
    vega_tier1_short = ((1.0 - alpha_m_2).unsqueeze(1) * d_pv_d_sig_m).sum(dim=2)
    vega_tier1_term = (alpha_m_2.unsqueeze(1) * d_pv_d_sig_m).sum(dim=2)
    w_v_term_ij = g_corr_bm + g_corr_bm.transpose(1, 2)

    g_lpd = g_edge_64[:, :, :, int(AutocallEdgeCol.LOG_PERF_DIFF)] * pair_mask_2
    sign_fp = torch.sign(log_fp_term_2.unsqueeze(2) - log_fp_term_2.unsqueeze(1))
    d_edge_lpd_d_log_s = ((g_lpd + g_lpd.transpose(1, 2)) * sign_fp).sum(dim=2) * cdl_term_eff
    g_dist_put = g_node_64[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)] * (cdl_term_eff / v_i_term_2)
    g_carry = g_node_64[:, :, int(AutocallNodeCol.LOG_FWD_CARRY_1Y)] * ((cdl_term_eff - 1.0) / tau_term_2)
    g_div_lev = g_node_64[:, :, int(AutocallNodeCol.CASH_DIV_LEVERAGE)] * (1.0 - cdl_term_eff)
    g_log_s_total = (
        g_node_64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)]
        + g_dist_put + g_carry + g_div_lev + d_edge_lpd_d_log_s + t1_d_log_s
    )
    g_cov_64 = g_edge_64[:, :, :, int(AutocallEdgeCol.COVARIANCE_TERM)] * pair_mask_2
    vega_cov_term = (
        (g_cov_64 + g_cov_64.transpose(1, 2)) * corr_sym_2 * v_i_term_2.unsqueeze(1) * torch.sqrt(tau_term_2).unsqueeze(-1)
    ).sum(dim=2) + (g_edge_64[:, :, :, int(AutocallEdgeCol.COVARIANCE_TERM)] * eye_d_64).diagonal(dim1=1, dim2=2) * (
        2.0 * iv_term_node_2 * tau_term_2
    )
    vega_dist_term = -g_node_64[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)] * node_2[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)] / iv_term_node_2
    g_node_64[:, :, int(AutocallNodeCol.IV_SHORT_ATM)] = g_node_64[:, :, int(AutocallNodeCol.IV_SHORT_ATM)] + vega_tier1_short
    g_node_64[:, :, int(AutocallNodeCol.IV_TERM_ATM)] = g_node_64[:, :, int(AutocallNodeCol.IV_TERM_ATM)] + vega_tier1_term + vega_cov_term + vega_dist_term
    g_corr_direct = g_edge_64[:, :, :, int(AutocallEdgeCol.CORR_RHO)]
    cega_cov_term = (g_cov_64 + g_cov_64.transpose(1, 2)) * (v_i_term_2.unsqueeze(2) * v_i_term_2.unsqueeze(1))
    cega_act_64 = ((g_corr_direct + g_corr_direct.transpose(1, 2)) + cega_cov_term + w_v_term_ij) * pair_mask_2

    g_eps_p = g_log_s_total[:batch_size].sum(dim=1) / (1.0 + h_cg)
    g_eps_m = g_log_s_total[batch_size:].sum(dim=1) / (1.0 - h_cg)
    cash_gamma_t = (g_eps_p - g_eps_m) / (2.0 * h_cg)
    g_log_s_base = 0.5 * (g_log_s_total[:batch_size] + g_log_s_total[batch_size:])
    g_node_base = 0.5 * (g_node_64[:batch_size] + g_node_64[batch_size:])
    g_sched_base = 0.5 * (g_sched_64[:batch_size] + g_sched_64[batch_size:])
    cega_base = 0.5 * (cega_act_64[:batch_size] + cega_act_64[batch_size:])
    pv_base = 0.5 * (pv_64[:batch_size] + pv_64[batch_size:])
    p_stop_base = 0.5 * (p_stop_64[:batch_size] + p_stop_64[batch_size:])
    surv_base = 0.5 * (surv_64[:batch_size] + surv_64[batch_size:])
    q_cpn_base = 0.5 * (q_cpn_64[:batch_size] + q_cpn_64[batch_size:])
    put_loss_base = 0.5 * (put_loss_64[:batch_size] + put_loss_64[batch_size:])

    g_node_f64 = torch.zeros((batch_size, d_max, d_node), dtype=torch.float64, device=device)
    g_node_f64[:, :d_act, :] = g_node_base
    rho_df_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
    rho_df_f64[:, :m_act] = g_sched_base[:, :, int(AutocallSchedCol.DF_PAY)] * smask_64
    stop_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
    stop_f64[:, :m_act] = p_stop_base
    surv_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
    surv_f64[:, :m_act] = surv_base
    if m_act < m_max:
        surv_f64[:, m_act:] = surv_f64[:, m_act - 1 : m_act]
    cpn_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
    cpn_f64[:, :m_act] = q_cpn_base

    if has_spots:
        sp_f64 = sp_in_t[:, :d_act].clamp(min=1e-12)
    else:
        sp_f64 = torch.exp(node_64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)]).clamp(min=1e-12)
        if has_ref_spots:
            sp_f64 = sp_f64 * ref_in_t[:, :d_act].clamp(min=1e-12)
    delta_f64 = torch.zeros((batch_size, d_max), dtype=torch.float64, device=device)
    delta_f64[:, :d_act] = (g_log_s_base / sp_f64) * amask_64
    amask_f_pad = torch.zeros((batch_size, d_max), dtype=torch.float64, device=device)
    amask_f_pad[:, :d_act] = amask_64
    vega_surface_f64 = g_node_f64[:, :, 4:11] * amask_f_pad.unsqueeze(-1)
    vega_atm_f64 = g_node_f64[:, :, 4:6] * amask_f_pad.unsqueeze(-1)
    cega_f64 = torch.zeros((batch_size, d_max, d_max), dtype=torch.float64, device=device)
    cega_f64[:, :d_act, :d_act] = cega_base
    return (
        pv_base, stop_f64, surv_f64, cpn_f64, put_loss_base,
        delta_f64, vega_atm_f64, vega_surface_f64, rho_df_f64, cega_f64, cash_gamma_t,
    )


def compute_batched_surrogate_greeks(
    surrogate: MaskedEQAutocallSurrogate,
    node_feats: torch.Tensor,
    edge_feats: torch.Tensor,
    asset_mask: torch.Tensor,
    sched_feats: torch.Tensor,
    sched_mask: torch.Tensor,
    global_feats: torch.Tensor,
    *,
    spots_t0: torch.Tensor | None = None,
    ref_spots: torch.Tensor | None = None,
    pv_in_transit: torch.Tensor | float | None = None,
    no_notional_repayment: torch.Tensor | bool = False,
    autocall_coupon: torch.Tensor | float = 0.0,
    force_coupons_on_autocall: torch.Tensor | bool = False,
    is_long_basket: torch.Tensor | bool | None = None,
    w_fund_per_period: torch.Tensor | None = None,
    lower_strike: torch.Tensor | float | None = None,
    one_star_pv: torch.Tensor | float = 0.0,
    part_call_pv: torch.Tensor | float = 0.0,
    forward_perf: torch.Tensor | None = None,
    v_perf: torch.Tensor | None = None,
    cash_div_leverage: torch.Tensor | None = None,
    compute_cash_gamma: bool = False,
    require_second_order_sdpa: bool = False,
) -> BatchedSurrogateGreeksOutput:
    """Vectorized GPU/CPU batched valuation and Greeks matching single-trade Tier-1/Tier-2 parity.

    Uses thread-safe `inputs=(node_in, edge_in, sched_in, glob_in)` gradient filtering (zero
    in-place `p.requires_grad_` mutation), decoupled W1 stopping-time float64 adjoint, fast
    CUDAGraph-cached float32 Tier-1 VJPs on CUDA batches, and central FD of first-order Euler
    derivatives for CashGamma without second-order SDPA create_graph overhead.
    """
    first_param = next(surrogate.parameters(), None)
    surr_dtype = first_param.dtype if first_param is not None else torch.float32
    device = node_feats.device

    batch_size, d_max, d_node = node_feats.shape
    _, m_max, d_sched = sched_feats.shape

    amask_bool = asset_mask > 0.5
    smask_bool = sched_mask > 0.5
    amask_f64 = amask_bool.to(dtype=torch.float64, device=device)
    smask_f64 = smask_bool.to(dtype=torch.float64, device=device)

    if d_max > 1 and not bool(amask_bool[:, -1].all()):
        any_asset = amask_bool.any(dim=0)
        d_act = int(any_asset.nonzero()[-1].item()) + 1 if any_asset.any() else 1
    else:
        d_act = d_max
    if m_max > 1 and not bool(smask_bool[:, -1].all()):
        any_sched = smask_bool.any(dim=0)
        m_act = int(any_sched.nonzero()[-1].item()) + 1 if any_sched.any() else 1
    else:
        m_act = m_max

    if (
        device.type == "cuda"
        and batch_size <= 8
        and surr_dtype == torch.float32
        and _HAS_TRITON_TIER1
        and d_act == 5
        and compute_cash_gamma
        and forward_perf is None
        and v_perf is None
        and cash_div_leverage is None
        and w_fund_per_period is None
        and pv_in_transit is None
        and isinstance(no_notional_repayment, bool)
        and not no_notional_repayment
        and isinstance(autocall_coupon, (int, float))
        and float(autocall_coupon) == 0.0
        and isinstance(one_star_pv, (int, float))
        and float(one_star_pv) == 0.0
        and isinstance(part_call_pv, (int, float))
        and float(part_call_pv) == 0.0
        and lower_strike is None
        and is_long_basket is None
        and isinstance(force_coupons_on_autocall, (bool, int, float))
        and callable(getattr(surrogate, "forward_residuals", None))
        and not torch.cuda.is_current_stream_capturing()  # type: ignore[no-untyped-call]
    ):
        has_cid = global_feats.shape[-1] > int(AutocallGlobalCol.CALL_INDEX_ID)
        c0_t = global_feats[0, int(AutocallGlobalCol.CALL_INDEX_ID)].round() if has_cid else torch.zeros((), dtype=torch.float64, device=device)
        same_cid_t = (global_feats[:, int(AutocallGlobalCol.CALL_INDEX_ID)].round() == c0_t).all() if has_cid else torch.ones((), dtype=torch.bool, device=device)
        div_t = (node_feats[:, :d_act, int(AutocallNodeCol.CASH_DIV_LEVERAGE)].abs() > 1e-10).any()
        lm_0 = node_feats[:, :d_act, int(AutocallNodeCol.LOG_MONEYNESS)]
        lc_0 = node_feats[:, :d_act, int(AutocallNodeCol.LOG_FWD_CARRY_1Y)]
        ld_0 = node_feats[:, :d_act, int(AutocallNodeCol.CASH_DIV_LEVERAGE)].clamp(-10.0, 10.0)
        ta_0 = sched_feats[:, :m_act, int(AutocallSchedCol.TAU_OBS)].clamp(min=1e-8)
        tt_0 = ta_0[:, -1:].clamp(min=1e-6)
        lf_nd0 = (lm_0.unsqueeze(2) + (lc_0 + ld_0 / tt_0).unsqueeze(2) * ta_0.unsqueeze(1)).clamp(-20.0, 20.0)
        lf_d0 = lf_nd0 - ld_0.unsqueeze(2)
        sym5_only_t = (
            ((lf_nd0.max(dim=1).values - lf_nd0.min(dim=1).values) < 0.001).all()
            & ((lf_d0.max(dim=1).values - lf_d0.min(dim=1).values) < 0.001).all()
        )
        s_nd0 = torch.sort(lf_nd0, dim=1).values
        s_d0 = torch.sort(lf_d0, dim=1).values
        no_ties_t = (
            ((s_nd0[:, 1:] - s_nd0[:, :-1]).min() > 0.062)
            & ((s_d0[:, 1:] - s_d0[:, :-1]).min() > 0.062)
        )
        fl = torch.stack([
            amask_bool[:, :d_act].all().to(torch.float64),
            smask_bool[:, :m_act].all().to(torch.float64),
            same_cid_t.to(torch.float64),
            c0_t.to(torch.float64),
            div_t.to(torch.float64),
            sym5_only_t.to(torch.float64),
            no_ties_t.to(torch.float64),
        ]).tolist()
        all_act_e2e = fl[0] > 0.5
        all_sched_e2e = fl[1] > 0.5
        same_cid_e2e = fl[2] > 0.5
        cid_0_e2e = int(round(fl[3]))
        has_div_e2e = fl[4] > 0.5
        use_sym5_e2e = fl[5] > 0.5
        no_ties_e2e = fl[6] > 0.5
        if all_act_e2e and all_sched_e2e and same_cid_e2e and cid_0_e2e in (0, 1):
            is_best_e2e = cid_0_e2e == 1
            force_cpn_f = float(force_coupons_on_autocall)
            sp_in_t = (
                spots_t0.to(dtype=torch.float64, device=device).reshape(-1, d_max).expand(batch_size, d_max).contiguous()
                if spots_t0 is not None
                else torch.empty((batch_size, d_max), dtype=torch.float64, device=device)
            )
            ref_in_t = (
                ref_spots.to(dtype=torch.float64, device=device).reshape(-1, d_max).expand(batch_size, d_max).contiguous()
                if ref_spots is not None
                else torch.empty((batch_size, d_max), dtype=torch.float64, device=device)
            )
            live_e2e = (
                node_feats[:, :d_act, :].detach().to(dtype=torch.float64, device=device).contiguous(),
                edge_feats[:, :d_act, :d_act, :].detach().to(dtype=torch.float64, device=device).contiguous(),
                sched_feats[:, :m_act, :].detach().to(dtype=torch.float64, device=device).contiguous(),
                global_feats.detach().to(dtype=torch.float64, device=device).contiguous(),
                amask_f64[:, :d_act].detach().contiguous(),
                smask_f64[:, :m_act].detach().contiguous(),
                sp_in_t,
                ref_in_t,
            )
            p_ver = int(getattr(first_param, "_version", 0)) if first_param is not None else 0
            surr_d_node = int(getattr(surrogate, "d_node", d_node))
            surr_d_global = int(getattr(surrogate, "d_global", global_feats.shape[-1]))
            e2e_key = (
                id(surrogate), p_ver, int(device.index or 0), batch_size, d_max, d_node, m_max, m_act,
                surr_d_node, surr_d_global, is_best_e2e, has_div_e2e, force_cpn_f,
                spots_t0 is not None, ref_spots is not None, use_sym5_e2e, no_ties_e2e,
            )
            do_cap_e2e = False
            with _E2E_CG_LOCK:
                e2e_entry = _E2E_CG_CACHE.get(e2e_key)
                if e2e_entry is not None:
                    _E2E_CG_CACHE.move_to_end(e2e_key)
                    do_cap_e2e = True
                else:
                    seen_e2e = _E2E_CG_SEEN.get(e2e_key, 0)
                    if seen_e2e >= 1:
                        do_cap_e2e = True
                    else:
                        if len(_E2E_CG_SEEN) >= 256:
                            _E2E_CG_SEEN.clear()
                        _E2E_CG_SEEN[e2e_key] = 1
            if not do_cap_e2e:
                e2e_outs = _compute_e2e_d5_cg_core(
                    surrogate, live_e2e, d_max=d_max, d_node=d_node, m_max=m_max,
                    is_best=is_best_e2e, has_div=has_div_e2e, force_cpn_f=force_cpn_f,
                    has_spots=(spots_t0 is not None), has_ref_spots=(ref_spots is not None),
                    use_sym5_only=use_sym5_e2e, no_ties=no_ties_e2e,
                )
            else:
                with _E2E_CG_LOCK:
                    e2e_entry = _E2E_CG_CACHE.get(e2e_key)
                    if e2e_entry is not None:
                        _E2E_CG_CACHE.move_to_end(e2e_key)
                    else:
                        static_in = tuple(t.clone() for t in live_e2e)
                        cap_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
                        cap_stream.wait_stream(torch.cuda.current_stream(device=device))
                        with torch.cuda.stream(cap_stream):
                            for _ in range(2):
                                _ = _compute_e2e_d5_cg_core(
                                    surrogate, static_in, d_max=d_max, d_node=d_node, m_max=m_max,
                                    is_best=is_best_e2e, has_div=has_div_e2e, force_cpn_f=force_cpn_f,
                                    has_spots=(spots_t0 is not None), has_ref_spots=(ref_spots is not None),
                                    use_sym5_only=use_sym5_e2e, no_ties=no_ties_e2e,
                                )
                        torch.cuda.current_stream(device=device).wait_stream(cap_stream)
                        cg = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
                        with torch.cuda.graph(cg):
                            static_out = _compute_e2e_d5_cg_core(
                                surrogate, static_in, d_max=d_max, d_node=d_node, m_max=m_max,
                                is_best=is_best_e2e, has_div=has_div_e2e, force_cpn_f=force_cpn_f,
                                has_spots=(spots_t0 is not None), has_ref_spots=(ref_spots is not None),
                                use_sym5_only=use_sym5_e2e, no_ties=no_ties_e2e,
                            )
                        e2e_entry = {"lock": threading.Lock(), "graph": cg, "static_in": static_in, "static_out": static_out}
                        _E2E_CG_CACHE[e2e_key] = e2e_entry
                        evicted = False
                        while len(_E2E_CG_CACHE) > _E2E_CG_MAXSIZE:
                            _, old_e = _E2E_CG_CACHE.popitem(last=False)
                            del old_e
                            evicted = True
                        if evicted:
                            torch.cuda.empty_cache()
                e2e_lock: threading.Lock = e2e_entry["lock"]
                with e2e_lock:
                    for s_t, l_t in zip(e2e_entry["static_in"], live_e2e):
                        s_t.copy_(l_t)
                    e2e_entry["graph"].replay()
                    e2e_outs = tuple(o.clone() for o in e2e_entry["static_out"])
            return BatchedSurrogateGreeksOutput(
                product_value=e2e_outs[0],
                stopping_probs=e2e_outs[1],
                survival_probs=e2e_outs[2],
                coupon_probs=e2e_outs[3],
                put_loss=e2e_outs[4],
                delta=e2e_outs[5],
                vega_atm=e2e_outs[6],
                vega_surface_feats=e2e_outs[7],
                rho_df=e2e_outs[8],
                cega=e2e_outs[9],
                cash_gamma_euler=e2e_outs[10],
            )

    if device.type == "cuda" and batch_size > 1:
        work_per_trade = max(m_act, 1) * max(d_act, 1) * (2 if compute_cash_gamma else 1)
        max_work_budget = 100_000 if batch_size > 1024 else 400_000
        if batch_size * work_per_trade > max_work_budget:
            chunk_b = max(1, min(batch_size, max_work_budget // max(work_per_trade, 1)))
            if chunk_b < batch_size:
                def _sl(val: Any, s_idx: int, e_idx: int, min_ndim: int = 1) -> Any:
                    if isinstance(val, torch.Tensor) and val.ndim >= min_ndim and int(val.shape[0]) == batch_size:
                        return val[s_idx:e_idx]
                    return val

                chunk_outs: list[BatchedSurrogateGreeksOutput] = []
                for s_c in range(0, batch_size, chunk_b):
                    e_c = min(s_c + chunk_b, batch_size)
                    chunk_outs.append(
                        compute_batched_surrogate_greeks(
                            surrogate,
                            node_feats[s_c:e_c],
                            edge_feats[s_c:e_c],
                            asset_mask[s_c:e_c],
                            sched_feats[s_c:e_c],
                            sched_mask[s_c:e_c],
                            global_feats[s_c:e_c],
                            spots_t0=_sl(spots_t0, s_c, e_c, 2),
                            ref_spots=_sl(ref_spots, s_c, e_c, 2),
                            pv_in_transit=_sl(pv_in_transit, s_c, e_c, 1),
                            no_notional_repayment=_sl(no_notional_repayment, s_c, e_c, 1),
                            autocall_coupon=_sl(autocall_coupon, s_c, e_c, 1),
                            force_coupons_on_autocall=_sl(force_coupons_on_autocall, s_c, e_c, 1),
                            is_long_basket=_sl(is_long_basket, s_c, e_c, 1),
                            w_fund_per_period=_sl(w_fund_per_period, s_c, e_c, 2),
                            lower_strike=_sl(lower_strike, s_c, e_c, 1),
                            one_star_pv=_sl(one_star_pv, s_c, e_c, 1),
                            part_call_pv=_sl(part_call_pv, s_c, e_c, 1),
                            forward_perf=_sl(forward_perf, s_c, e_c, 3),
                            v_perf=_sl(v_perf, s_c, e_c, 3),
                            cash_div_leverage=_sl(cash_div_leverage, s_c, e_c, 3),
                            compute_cash_gamma=compute_cash_gamma,
                            require_second_order_sdpa=require_second_order_sdpa,
                        )
                    )
                return BatchedSurrogateGreeksOutput(
                    product_value=torch.cat([o.product_value for o in chunk_outs], dim=0),
                    stopping_probs=torch.cat([o.stopping_probs for o in chunk_outs], dim=0),
                    survival_probs=torch.cat([o.survival_probs for o in chunk_outs], dim=0),
                    coupon_probs=torch.cat([o.coupon_probs for o in chunk_outs], dim=0),
                    put_loss=torch.cat([o.put_loss for o in chunk_outs], dim=0),
                    delta=torch.cat([o.delta for o in chunk_outs], dim=0),
                    vega_atm=torch.cat([o.vega_atm for o in chunk_outs], dim=0),
                    vega_surface_feats=torch.cat([o.vega_surface_feats for o in chunk_outs], dim=0),
                    rho_df=torch.cat([o.rho_df for o in chunk_outs], dim=0),
                    cega=torch.cat([o.cega for o in chunk_outs], dim=0),
                    cash_gamma_euler=(
                        torch.cat([o.cash_gamma_euler for o in chunk_outs if o.cash_gamma_euler is not None], dim=0)
                        if compute_cash_gamma
                        else None
                    ),
                )

    surr_d_node = int(getattr(surrogate, "d_node", d_node))
    surr_d_global = int(getattr(surrogate, "d_global", global_feats.shape[-1]))
    amask_64 = amask_f64[:, :d_act]
    smask_64 = smask_f64[:, :m_act]
    all_sched_act = bool(smask_bool[:, :m_act].all())
    idx_last: torch.Tensor | None = (
        None if all_sched_act else (smask_64.sum(dim=1).long() - 1).clamp(min=0)
    )

    node_64 = node_feats[:, :d_act, :].detach().to(dtype=torch.float64, device=device)
    edge_64 = edge_feats[:, :d_act, :d_act, :].detach().to(dtype=torch.float64, device=device)
    sched_64 = sched_feats[:, :m_act, :].detach().to(dtype=torch.float64, device=device)
    glob_64 = global_feats.detach().to(dtype=torch.float64, device=device)

    w_fund_act = (
        w_fund_per_period[:, :m_act].to(device=device, dtype=torch.float64)
        if w_fund_per_period is not None and w_fund_per_period.ndim >= 2
        else w_fund_per_period
    )

    with torch.no_grad():
        iv_short_node = node_64[:, :, int(AutocallNodeCol.IV_SHORT_ATM)].clamp(min=1e-4)
        iv_term_node = node_64[:, :, int(AutocallNodeCol.IV_TERM_ATM)].clamp(min=1e-4)
        tau_all = sched_64[:, :, int(AutocallSchedCol.TAU_OBS)].clamp(min=1e-8)
        tau_short = tau_all[:, :1]
        tau_term = (
            tau_all[:, -1:].clamp(min=1e-6)
            if idx_last is None
            else tau_all.gather(1, idx_last.unsqueeze(1)).clamp(min=1e-6)
        )
        v_i_term = (iv_term_node * torch.sqrt(tau_term)).clamp(min=1e-6)

        w_proxy = node_64[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)].clamp(min=0.0) * amask_64
        log_fp_term_node = (
            node_64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)]
            + node_64[:, :, int(AutocallNodeCol.LOG_FWD_CARRY_1Y)] * tau_term
        )
        log_fw_base = sched_64[:, :, int(AutocallSchedCol.STULZ_FWD_W)].clamp(-20.0, 20.0)
        f_w_term = torch.exp(
            (log_fw_base.gather(1, idx_last.unsqueeze(1)) if idx_last is not None else log_fw_base[:, -1:]).clamp(
                -20.0, 20.0
            )
        ).clamp(min=1e-12)
        v_w_base = sched_64[:, :, int(AutocallSchedCol.STULZ_VOL_W)].clamp(min=1e-6)
        b_ko = torch.exp(sched_64[:, :, int(AutocallSchedCol.LOG_KO_BARRIER)].clamp(-20.0, 20.0))
        b_cpn = torch.exp(sched_64[:, :, int(AutocallSchedCol.LOG_CPN_BARRIER)].clamp(-20.0, 20.0))
        is_call_base = sched_64[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)] * smask_64
        is_guar_base = sched_64[:, :, int(AutocallSchedCol.IS_GUARANTEED_CPN)]
        ko_spread_mag = glob_64[:, int(AutocallGlobalCol.ABS_KO_SPREAD) : int(AutocallGlobalCol.ABS_KO_SPREAD) + 1]
        cpn_spread_mag = glob_64[:, int(AutocallGlobalCol.ABS_CPN_SPREAD) : int(AutocallGlobalCol.ABS_CPN_SPREAD) + 1]
        ki_spread_mag = (
            glob_64[:, int(AutocallGlobalCol.ABS_KI_SPREAD) : int(AutocallGlobalCol.ABS_KI_SPREAD) + 1]
            if glob_64.shape[-1] > int(AutocallGlobalCol.ABS_KI_SPREAD)
            else ko_spread_mag
        )
        k_strike = glob_64[:, int(AutocallGlobalCol.K_PUT)]
        b_ki = glob_64[:, int(AutocallGlobalCol.B_KI)]
        has_ki = glob_64[:, int(AutocallGlobalCol.HAS_KI_FLAG)]
        i_ki_past = glob_64[:, int(AutocallGlobalCol.I_KI_PAST)]
        active_ki = (has_ki > 0.5) & (i_ki_past < 0.5) & (b_ki > 0.0)

        if lower_strike is not None:
            k_low = torch.as_tensor(lower_strike, dtype=torch.float64, device=device).clamp(min=0.0)
            if k_low.ndim == 0:
                k_low = k_low.expand_as(k_strike)
        elif glob_64.shape[-1] > int(AutocallGlobalCol.LOWER_STRIKE):
            k_low = glob_64[:, int(AutocallGlobalCol.LOWER_STRIKE)].clamp(min=0.0)
        else:
            k_low = torch.zeros_like(k_strike)

        eye_d_64 = torch.eye(d_act, dtype=torch.float64, device=device).unsqueeze(0)
        pair_mask_64 = amask_64.unsqueeze(1) * amask_64.unsqueeze(2) * (1.0 - eye_d_64)
        corr_sym_64 = (
            0.5 * (edge_64[:, :, :, int(AutocallEdgeCol.CORR_RHO)] + edge_64[:, :, :, int(AutocallEdgeCol.CORR_RHO)].transpose(1, 2))
            * (1.0 - eye_d_64)
            + eye_d_64
        )
        sqrt_tau_m = torch.sqrt(tau_all)
        span_t = (tau_term - tau_short).clamp(min=1e-8)
        alpha_m = ((tau_all - tau_short) / span_t).clamp(0.0, 1.0)
        if v_perf is not None:
            v_m_raw = v_perf[:, :d_act, :m_act].detach().to(dtype=torch.float64, device=device) * amask_64.unsqueeze(2)
        else:
            iv_m_bm = (
                (1.0 - alpha_m).unsqueeze(1) * iv_short_node.unsqueeze(2)
                + alpha_m.unsqueeze(1) * iv_term_node.unsqueeze(2)
            )
            v_m_raw = iv_m_bm * sqrt_tau_m.unsqueeze(1)

        call_code_0 = (
            int(round(float(glob_64[0, int(AutocallGlobalCol.CALL_INDEX_ID)].item())))
            if glob_64.shape[-1] > int(AutocallGlobalCol.CALL_INDEX_ID)
            else 0
        )
        same_call_code = (
            bool((glob_64[:, int(AutocallGlobalCol.CALL_INDEX_ID)].round() == call_code_0).all().item())
            if glob_64.shape[-1] > int(AutocallGlobalCol.CALL_INDEX_ID)
            else True
        )
        use_markov_tier1 = same_call_code and call_code_0 in (0, 1)
        call_idx_str = "BestOf" if call_code_0 == 1 else "WorstOf"
        sym5_call_cache: dict[str, Any] = {}

        term_rec_only = bool(not use_markov_tier1 and all_sched_act and m_act > 1)
        if forward_perf is not None and cash_div_leverage is not None:
            fp_bm_raw = forward_perf[:, :d_act, :m_act].detach().to(dtype=torch.float64, device=device) * amask_64.unsqueeze(2)
            cdl_bm = cash_div_leverage[:, :d_act, :m_act].detach().to(dtype=torch.float64, device=device) * amask_64.unsqueeze(2)
        elif forward_perf is not None:
            fp_bm_raw = forward_perf[:, :d_act, :m_act].detach().to(dtype=torch.float64, device=device) * amask_64.unsqueeze(2)
            if term_rec_only:
                _, cdl_bm = _reconstruct_period_forwards_and_cdl(
                    node_64, v_m_raw[:, :, -1:], corr_sym_64, amask_64, tau_all[:, -1:], tau_term,
                    log_fw_base[:, -1:], call_idx_str, vw_base=v_w_base[:, -1:], term_only=True,
                )
            else:
                _, cdl_bm = _reconstruct_period_forwards_and_cdl(
                    node_64, v_m_raw, corr_sym_64, amask_64, tau_all, tau_term, log_fw_base, call_idx_str, vw_base=v_w_base
                )
        elif cash_div_leverage is not None:
            if term_rec_only:
                fp_bm_raw, _ = _reconstruct_period_forwards_and_cdl(
                    node_64, v_m_raw[:, :, -1:], corr_sym_64, amask_64, tau_all[:, -1:], tau_term,
                    log_fw_base[:, -1:], call_idx_str, vw_base=v_w_base[:, -1:], term_only=True,
                )
            else:
                fp_bm_raw, _ = _reconstruct_period_forwards_and_cdl(
                    node_64, v_m_raw, corr_sym_64, amask_64, tau_all, tau_term, log_fw_base, call_idx_str, vw_base=v_w_base
                )
            cdl_bm = cash_div_leverage[:, :d_act, :m_act].detach().to(dtype=torch.float64, device=device) * amask_64.unsqueeze(2)
        elif term_rec_only:
            fp_bm_raw, cdl_bm = _reconstruct_period_forwards_and_cdl(
                node_64, v_m_raw[:, :, -1:], corr_sym_64, amask_64, tau_all[:, -1:], tau_term,
                log_fw_base[:, -1:], call_idx_str, vw_base=v_w_base[:, -1:], term_only=True,
            )
        else:
            fp_bm_raw, cdl_bm = _reconstruct_period_forwards_and_cdl(
                node_64, v_m_raw, corr_sym_64, amask_64, tau_all, tau_term, log_fw_base, call_idx_str, vw_base=v_w_base
            )

        cdl_term = (
            cdl_bm[:, :, -1]
            if idx_last is None
            else cdl_bm.gather(2, idx_last.view(batch_size, 1, 1).expand(batch_size, d_act, 1)).squeeze(2)
        )
        f_p_term = (
            fp_bm_raw[:, :, -1]
            if idx_last is None
            else fp_bm_raw.gather(2, idx_last.view(batch_size, 1, 1).expand(batch_size, d_act, 1)).squeeze(2)
        ) * amask_64
        w_log_fw = w_proxy * (f_p_term / f_w_term)
        w_lev = w_log_fw * cdl_term

    # Match h_euler_rel=1e-3 order while bounding 0.5*(+h,-h) midpoint curvature on Delta/Cega.
    h_cg = 7.4e-4
    if compute_cash_gamma:
        b_eff = 2 * batch_size
        eps_shifts = torch.cat(
            [
                torch.full((batch_size,), h_cg, dtype=torch.float64, device=device),
                torch.full((batch_size,), -h_cg, dtype=torch.float64, device=device),
            ],
            dim=0,
        )
        one_plus_u = (1.0 + eps_shifts).view(b_eff, 1)
        cdl_bm_2 = torch.cat([cdl_bm, cdl_bm], dim=0)
        inv_lev_bm = 1.0 / cdl_bm_2.clamp(min=1.0)
        escrow_frac_bm = (one_plus_u.unsqueeze(2) - (1.0 - inv_lev_bm)).clamp(min=1e-6)
        log_fp_shift_bm = torch.log(escrow_frac_bm / inv_lev_bm.clamp(min=1e-6))
        if idx_last is None:
            log_fp_shift_term = log_fp_shift_bm[:, :, -1]
            idx_last_eff: torch.Tensor | None = None
        else:
            idx_last_eff = torch.cat([idx_last, idx_last], dim=0)
            log_fp_shift_term = log_fp_shift_bm.gather(
                2, idx_last_eff.view(b_eff, 1, 1).expand(b_eff, d_act, 1)
            ).squeeze(2)

        node_2 = torch.cat([node_64, node_64], dim=0)
        edge_2 = torch.cat([edge_64, edge_64], dim=0)
        sched_2 = torch.cat([sched_64, sched_64], dim=0)
        glob_2 = torch.cat([glob_64, glob_64], dim=0)
        amask_2 = torch.cat([amask_64, amask_64], dim=0)
        smask_2 = torch.cat([smask_64, smask_64], dim=0)
        tau_term_2 = torch.cat([tau_term, tau_term], dim=0)
        v_i_term_2 = torch.cat([v_i_term, v_i_term], dim=0)
        sqrt_tau_m_2 = torch.cat([sqrt_tau_m, sqrt_tau_m], dim=0)
        alpha_m_2 = torch.cat([alpha_m, alpha_m], dim=0)
        iv_term_node_2 = torch.cat([iv_term_node, iv_term_node], dim=0)
        v_m_raw_2 = torch.cat([v_m_raw, v_m_raw], dim=0)
        corr_sym_2 = torch.cat([corr_sym_64, corr_sym_64], dim=0)
        pair_mask_2 = torch.cat([pair_mask_64, pair_mask_64], dim=0)
        b_ko_2 = torch.cat([b_ko, b_ko], dim=0)
        b_cpn_2 = torch.cat([b_cpn, b_cpn], dim=0)
        is_call_2 = torch.cat([is_call_base, is_call_base], dim=0)
        is_guar_2 = torch.cat([is_guar_base, is_guar_base], dim=0)
        ko_sp_2 = torch.cat([ko_spread_mag, ko_spread_mag], dim=0)
        cpn_sp_2 = torch.cat([cpn_spread_mag, cpn_spread_mag], dim=0)
        ki_sp_2 = torch.cat([ki_spread_mag, ki_spread_mag], dim=0)
        k_strike_2 = torch.cat([k_strike, k_strike], dim=0)
        k_low_2 = torch.cat([k_low, k_low], dim=0)
        active_ki_2 = torch.cat([active_ki, active_ki], dim=0)
        b_ki_2 = torch.cat([b_ki, b_ki], dim=0)

        log_m0_u = node_2[:, :, int(AutocallNodeCol.LOG_MONEYNESS)] + torch.log(one_plus_u)
        log_div_lev_u = torch.log((one_plus_u / escrow_frac_bm[:, :, -1]).clamp(min=1.0)) if idx_last_eff is None else torch.log(
            (one_plus_u / escrow_frac_bm.gather(2, idx_last_eff.view(b_eff, 1, 1).expand(b_eff, d_act, 1)).squeeze(2)).clamp(min=1.0)
        )
        log_carry_u = (
            node_2[:, :, int(AutocallNodeCol.LOG_FWD_CARRY_1Y)]
            + (log_fp_shift_term - torch.log(one_plus_u)) / tau_term_2
        )
        dist_put_u = (
            node_2[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)]
            + log_fp_shift_term / v_i_term_2
        )
        node_2[:, :, int(AutocallNodeCol.LOG_MONEYNESS)] = log_m0_u
        node_2[:, :, int(AutocallNodeCol.CASH_DIV_LEVERAGE)] = log_div_lev_u
        node_2[:, :, int(AutocallNodeCol.LOG_FWD_CARRY_1Y)] = log_carry_u
        node_2[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)] = dist_put_u

        log_fp_term_2 = log_m0_u + log_carry_u * tau_term_2
        edge_2[:, :, :, int(AutocallEdgeCol.LOG_PERF_DIFF)] = torch.sqrt(
            (log_fp_term_2.unsqueeze(2) - log_fp_term_2.unsqueeze(1)) ** 2 + 1e-18
        ) * pair_mask_2

        fp_bm_eff = torch.cat([fp_bm_raw, fp_bm_raw], dim=0) * torch.exp(log_fp_shift_bm)
        # Recompute shifted Tier-1 proxies for +h and -h cleanly
        if use_markov_tier1:
            if device.type != "cuda":
                lfw_u, vw_u, p_u, q_u, lput_u, w_term_u = _batched_tier1_proxies(
                    fp_bm_eff,
                    v_m_raw_2,
                    corr_sym_2,
                    amask_2,
                    smask_2,
                    b_ko_2,
                    b_cpn_2,
                    is_call_2,
                    is_guar_2,
                    ko_sp_2,
                    cpn_sp_2,
                    ki_sp_2,
                    k_strike_2,
                    k_low_2,
                    active_ki_2,
                    b_ki_2,
                    force_coupons=torch.cat([force_coupons_on_autocall, force_coupons_on_autocall], dim=0)
                    if isinstance(force_coupons_on_autocall, torch.Tensor) and force_coupons_on_autocall.ndim >= 1
                    else force_coupons_on_autocall,
                    call_idx_str=call_idx_str,
                    all_sched_act=all_sched_act,
                    idx_last=idx_last_eff,
                )
            else:
                # Single b_eff CUDAGraph forward call (saves ~4 ms vs 2x chunked launch)
                fcpn_2 = (
                    torch.cat([force_coupons_on_autocall, force_coupons_on_autocall], dim=0).to(device=device, dtype=torch.float32)
                    if isinstance(force_coupons_on_autocall, torch.Tensor) and force_coupons_on_autocall.ndim >= 1
                    else torch.full((b_eff,), 1.0 if bool(force_coupons_on_autocall) else 0.0, dtype=torch.float32, device=device)
                )
                lfw_u, vw_u, p_u, q_u, lput_u, w_term_u = _batched_tier1_fwd_cuda_f32(
                    fp_bm_eff, v_m_raw_2, corr_sym_2, amask_2, smask_2, b_ko_2, b_cpn_2,
                    is_call_2, is_guar_2, ko_sp_2, cpn_sp_2, ki_sp_2, k_strike_2, k_low_2,
                    active_ki_2, b_ki_2, fcpn_2, idx_last_eff,
                    is_best_of=(call_code_0 == 1), all_sched_act=all_sched_act,
                    sym5_cache=sym5_call_cache,
                )
            if surr_d_node > int(AutocallNodeCol.PROXY_DELTA_WEIGHT):
                node_2[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)] = w_term_u
            sched_2[:, :, int(AutocallSchedCol.STULZ_FWD_W)] = lfw_u
            sched_2[:, :, int(AutocallSchedCol.STULZ_VOL_W)] = vw_u
            sched_2[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] = p_u
            sched_2[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)] = q_u
            glob_2[:, int(AutocallGlobalCol.L_PUT_PROXY)] = lput_u
        else:
            log_fw_2 = torch.cat([log_fw_base, log_fw_base], dim=0)
            v_w_2 = torch.cat([v_w_base, v_w_base], dim=0)
            w_log_fw_2 = torch.cat([w_log_fw, w_log_fw], dim=0)
            dlog_fw_u = (w_log_fw_2 * log_fp_shift_term).sum(dim=1, keepdim=True)
            f_w_cg = torch.exp(log_fw_2 + dlog_fw_u)
            h_raw_cg = is_call_2 * _digital_call_spread_prob_torch(f_w_cg, v_w_2, b_ko_2, -ko_sp_2)
            h_c_cg = h_raw_cg.clamp(0.0, 1.0 - 1e-6)
            if m_act == 1:
                p_proxy_cg = h_c_cg * smask_2
            else:
                s_c_cg = torch.cumprod(1.0 - h_c_cg, dim=-1)
                sp_c_cg = torch.cat([torch.ones((b_eff, 1), dtype=torch.float64, device=device), s_c_cg[:, :-1]], dim=-1)
                p_proxy_cg = (sp_c_cg * h_c_cg) * smask_2
            surv_term_cg = (1.0 - p_proxy_cg.sum(dim=-1)).clamp(min=0.0)
            q_raw_cg = _digital_call_spread_prob_torch(f_w_cg, v_w_2, b_cpn_2, -cpn_sp_2)
            q_proxy_cg = (is_guar_2 + (1.0 - is_guar_2) * q_raw_cg) * smask_2
            if idx_last_eff is None:
                f_w_term_cg, v_w_term_cg = f_w_cg[:, -1], v_w_2[:, -1]
            else:
                f_w_term_cg = f_w_cg.gather(1, idx_last_eff.unsqueeze(1)).squeeze(1)
                v_w_term_cg = v_w_2.gather(1, idx_last_eff.unsqueeze(1)).squeeze(1)
            l_put_proxy_cg = surv_term_cg * _batched_put_spread_loss_torch(
                f_w_term_cg, v_w_term_cg, k_strike_2, k_low_2, active_ki_2, b_ki_2, ko_sp_2.squeeze(1)
            )
            sched_2[:, :, int(AutocallSchedCol.STULZ_FWD_W)] = log_fw_2 + dlog_fw_u
            sched_2[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] = p_proxy_cg
            sched_2[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)] = q_proxy_cg
            glob_2[:, int(AutocallGlobalCol.L_PUT_PROXY)] = l_put_proxy_cg

        node_eff_64 = node_2
        edge_eff_64 = edge_2
        sched_eff_64 = sched_2
        glob_eff_64 = glob_2
        amask_eff_64 = amask_2
        smask_eff_64 = smask_2
        fp_bm_in_eff = fp_bm_eff
        v_m_raw_eff = v_m_raw_2
        corr_sym_eff = corr_sym_2
        pair_mask_eff = pair_mask_2
        tau_term_eff = tau_term_2
        sqrt_tau_m_eff = sqrt_tau_m_2
        alpha_m_eff = alpha_m_2
        iv_term_node_eff = iv_term_node_2
        v_i_term_eff = v_i_term_2
        b_ko_eff = b_ko_2
        b_cpn_eff = b_cpn_2
        is_call_eff = is_call_2
        is_guar_eff = is_guar_2
        ko_sp_eff = ko_sp_2
        cpn_sp_eff = cpn_sp_2
        ki_sp_eff = ki_sp_2
        k_strike_eff = k_strike_2
        k_low_eff = k_low_2
        active_ki_eff = active_ki_2
        b_ki_eff = b_ki_2
        cdl_bm_u = (one_plus_u.unsqueeze(2) / escrow_frac_bm.clamp(min=1e-6)).clamp(min=1.0) * amask_2.unsqueeze(2)
        cdl_bm_eff = cdl_bm_u
        cdl_term_eff = (
            cdl_bm_u[:, :, -1]
            if idx_last_eff is None
            else cdl_bm_u.gather(2, idx_last_eff.view(b_eff, 1, 1).expand(b_eff, d_act, 1)).squeeze(2)
        )
        log_fp_eff = log_fp_term_2
        w_lev_eff = torch.cat([w_lev, w_lev], dim=0)
        log_fw_eff = sched_2[:, :, int(AutocallSchedCol.STULZ_FWD_W)]
        v_w_eff = sched_2[:, :, int(AutocallSchedCol.STULZ_VOL_W)].clamp(min=1e-6)

        def _double_opt(val: Any) -> Any:
            if isinstance(val, torch.Tensor) and val.ndim >= 1 and int(val.shape[0]) == batch_size:
                return torch.cat([val, val], dim=0)
            return val

        w_fund_eff = _double_opt(w_fund_act)
        no_notional_eff = _double_opt(no_notional_repayment)
        autocall_cpn_eff = _double_opt(autocall_coupon)
        force_cpn_eff = _double_opt(force_coupons_on_autocall)
        is_long_eff = _double_opt(is_long_basket)
        lower_k_eff = _double_opt(lower_strike)
        one_star_eff = _double_opt(one_star_pv)
        part_call_eff = _double_opt(part_call_pv)
    else:
        b_eff = batch_size
        idx_last_eff = idx_last
        node_eff_64 = node_64
        edge_eff_64 = edge_64
        sched_eff_64 = sched_64
        glob_eff_64 = glob_64
        amask_eff_64 = amask_64
        smask_eff_64 = smask_64
        fp_bm_in_eff = fp_bm_raw
        v_m_raw_eff = v_m_raw
        corr_sym_eff = corr_sym_64
        pair_mask_eff = pair_mask_64
        tau_term_eff = tau_term
        sqrt_tau_m_eff = sqrt_tau_m
        alpha_m_eff = alpha_m
        iv_term_node_eff = iv_term_node
        v_i_term_eff = v_i_term
        b_ko_eff = b_ko
        b_cpn_eff = b_cpn
        is_call_eff = is_call_base
        is_guar_eff = is_guar_base
        ko_sp_eff = ko_spread_mag
        cpn_sp_eff = cpn_spread_mag
        ki_sp_eff = ki_spread_mag
        k_strike_eff = k_strike
        k_low_eff = k_low
        active_ki_eff = active_ki
        b_ki_eff = b_ki
        cdl_bm_eff = cdl_bm
        cdl_term_eff = cdl_term
        log_fp_eff = log_fp_term_node
        w_lev_eff = w_lev
        log_fw_eff = log_fw_base
        v_w_eff = v_w_base
        w_fund_eff = w_fund_act
        no_notional_eff = no_notional_repayment
        autocall_cpn_eff = autocall_coupon
        force_cpn_eff = force_coupons_on_autocall
        is_long_eff = is_long_basket
        lower_k_eff = lower_strike
        one_star_eff = one_star_pv
        part_call_eff = part_call_pv

    fwd_res_fn = getattr(surrogate, "forward_residuals", None)
    use_cg_surr = bool(
        device.type == "cuda"
        and callable(fwd_res_fn)
        and surr_dtype == torch.float32
    )
    if use_cg_surr:
        (
            pv_64,
            p_stop_64,
            surv_64,
            q_cpn_64,
            put_loss_64,
            g_sched_w1,
            g_glob_w1,
            g_node_act,
            g_edge_act,
            g_sched_act,
            g_glob_act,
        ) = _batched_surr_w1_fwdbwd_cuda(
            surrogate,
            node_eff_64,
            edge_eff_64,
            sched_eff_64,
            glob_eff_64,
            amask_eff_64,
            smask_eff_64,
            no_notional_repayment=no_notional_eff,
            autocall_coupon=autocall_cpn_eff,
            force_coupons_on_autocall=force_cpn_eff,
            is_long_basket=is_long_eff,
            w_fund_act=w_fund_eff,
            lower_strike=lower_k_eff,
            one_star_pv=one_star_eff,
            part_call_pv=part_call_eff,
            use_tf32=False,
        )
    else:
        node_in = node_eff_64[:, :, :surr_d_node].to(device=device, dtype=surr_dtype).requires_grad_(True)
        edge_in = edge_eff_64.to(device=device, dtype=surr_dtype).requires_grad_(True)
        sched_in = sched_eff_64.to(device=device, dtype=surr_dtype).requires_grad_(True)
        glob_in = glob_eff_64[:, :surr_d_global].to(device=device, dtype=surr_dtype).requires_grad_(True)
        amask_act = amask_eff_64.to(device=device, dtype=surr_dtype)
        smask_act = smask_eff_64.to(device=device, dtype=surr_dtype)
        if callable(fwd_res_fn):
            d_haz_t, d_cpn_t, d_put_t = fwd_res_fn(
                node_in,
                edge_in,
                amask_act,
                sched_in,
                smask_act,
                glob_in,
                require_second_order_sdpa=True if device.type == "cuda" else require_second_order_sdpa,
            )
        else:
            surr_out_raw = surrogate(
                node_in,
                edge_in,
                amask_act,
                sched_in,
                smask_act,
                glob_in,
                detach_gate_weights=False,
                return_residuals_only=True,
                all_active=True,
            )
            d_haz_t = surr_out_raw.delta_hazard_logits[:, :m_act]
            d_cpn_t = surr_out_raw.delta_coupon_logits[:, :m_act]
            d_put_t = surr_out_raw.delta_put_loss

        (
            pv_64,
            p_stop_64,
            surv_64,
            q_cpn_64,
            put_loss_64,
            g_dh_64,
            g_dc_64,
            g_dp_64,
            g_sched_w1,
            g_glob_w1,
        ) = _batched_w1_head_and_vjp(
            d_haz_t.detach().to(dtype=torch.float64),
            d_cpn_t.detach().to(dtype=torch.float64),
            d_put_t.detach().to(dtype=torch.float64),
            sched_eff_64,
            smask_eff_64,
            glob_eff_64[:, :surr_d_global],
            no_notional_repayment=no_notional_eff,
            autocall_coupon=autocall_cpn_eff,
            force_coupons_on_autocall=force_cpn_eff,
            is_long_basket=is_long_eff,
            w_fund_act=w_fund_eff,
            lower_strike=lower_k_eff,
            one_star_pv=one_star_eff,
            part_call_pv=part_call_eff,
        )

        # Thread-safe: VJP from exact float64 W1 head adjoints through float32 surrogate residuals.
        g_node_act, g_edge_act, g_sched_act, g_glob_act = torch.autograd.grad(
            (d_haz_t, d_cpn_t, d_put_t),
            (node_in, edge_in, sched_in, glob_in),
            grad_outputs=(
                g_dh_64.to(dtype=d_haz_t.dtype),
                g_dc_64.to(dtype=d_cpn_t.dtype),
                g_dp_64.to(dtype=d_put_t.dtype),
            ),
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )

    with torch.no_grad():
        g_node_64 = (
            g_node_act.detach().to(dtype=torch.float64)
            if g_node_act is not None
            else torch.zeros_like(node_eff_64[:, :, :surr_d_node])
        )
        if g_node_64.shape[-1] < d_node:
            g_node_64 = torch.nn.functional.pad(g_node_64, (0, d_node - g_node_64.shape[-1]))
        g_edge_64 = (
            g_edge_act.detach().to(dtype=torch.float64)
            if g_edge_act is not None
            else torch.zeros_like(edge_eff_64)
        )
        g_sched_64 = (
            g_sched_act.detach().to(dtype=torch.float64)
            if g_sched_act is not None
            else torch.zeros_like(sched_eff_64)
        ) + g_sched_w1
        g_glob_64 = (
            g_glob_act.detach().to(dtype=torch.float64)
            if g_glob_act is not None
            else torch.zeros_like(glob_eff_64[:, :surr_d_global])
        ) + g_glob_w1

        if use_markov_tier1:
            if device.type == "cuda":
                fcpn_vec = (
                    force_cpn_eff.to(device=device, dtype=torch.float32).reshape(-1)
                    if isinstance(force_cpn_eff, torch.Tensor) and force_cpn_eff.ndim >= 1
                    else torch.full((b_eff,), 1.0 if bool(force_cpn_eff) else 0.0, dtype=torch.float32, device=device)
                )
                g_fp_bm, g_v_bm, g_corr_bm = _batched_tier1_vjp_cuda_f32(
                    fp_bm_in_eff,
                    v_m_raw_eff,
                    corr_sym_eff,
                    amask_eff_64,
                    smask_eff_64,
                    b_ko_eff,
                    b_cpn_eff,
                    is_call_eff,
                    is_guar_eff,
                    ko_sp_eff,
                    cpn_sp_eff,
                    ki_sp_eff,
                    k_strike_eff,
                    k_low_eff,
                    active_ki_eff,
                    b_ki_eff,
                    fcpn_vec,
                    idx_last_eff,
                    g_sched_64[:, :, int(AutocallSchedCol.STULZ_FWD_W)] * smask_eff_64,
                    g_sched_64[:, :, int(AutocallSchedCol.STULZ_VOL_W)] * smask_eff_64,
                    g_sched_64[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] * smask_eff_64,
                    g_sched_64[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)] * smask_eff_64,
                    g_glob_64[:, int(AutocallGlobalCol.L_PUT_PROXY)],
                    g_node_64[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)] * amask_eff_64,
                    is_best_of=(call_code_0 == 1),
                    all_sched_act=all_sched_act,
                    sym5_cache=sym5_call_cache,
                )
            else:
                with torch.enable_grad():  # type: ignore[no-untyped-call]
                    fp_bm_in = fp_bm_in_eff.detach().requires_grad_(True)
                    v_bm_in = v_m_raw_eff.detach().requires_grad_(True)
                    corr_in = corr_sym_eff.detach().requires_grad_(True)
                    log_fw_t1, vw_t1, p_t1, q_t1, lput_t1, w_term_t1 = _batched_tier1_proxies(
                        fp_bm_in,
                        v_bm_in,
                        corr_in,
                        amask_eff_64,
                        smask_eff_64,
                        b_ko_eff,
                        b_cpn_eff,
                        is_call_eff,
                        is_guar_eff,
                        ko_sp_eff,
                        cpn_sp_eff,
                        ki_sp_eff,
                        k_strike_eff,
                        k_low_eff,
                        active_ki_eff,
                        b_ki_eff,
                        force_coupons=force_cpn_eff,
                        call_idx_str=call_idx_str,
                        all_sched_act=all_sched_act,
                        idx_last=idx_last_eff,
                    )
                    tier1_obj = (
                        (
                            g_sched_64[:, :, int(AutocallSchedCol.STULZ_FWD_W)] * smask_eff_64 * log_fw_t1
                            + g_sched_64[:, :, int(AutocallSchedCol.STULZ_VOL_W)] * smask_eff_64 * vw_t1
                            + g_sched_64[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] * smask_eff_64 * p_t1
                            + g_sched_64[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)] * smask_eff_64 * q_t1
                        ).sum()
                        + (g_glob_64[:, int(AutocallGlobalCol.L_PUT_PROXY)] * lput_t1).sum()
                        + (g_node_64[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)] * amask_eff_64 * w_term_t1).sum()
                    )
                    g_fp_bm, g_v_bm, g_corr_bm = torch.autograd.grad(
                        tier1_obj,
                        (fp_bm_in, v_bm_in, corr_in),
                        allow_unused=True,
                    )
            t1_d_log_s = (
                (g_fp_bm.detach() * fp_bm_in_eff * cdl_bm_eff).sum(dim=2)
                if g_fp_bm is not None
                else torch.zeros((b_eff, d_act), dtype=torch.float64, device=device)
            )
            if g_v_bm is not None:
                d_pv_d_sig_m = torch.where(
                    (amask_eff_64 > 0.5).unsqueeze(2) & (smask_eff_64 > 0.5).unsqueeze(1),
                    g_v_bm.detach() * sqrt_tau_m_eff.unsqueeze(1),
                    0.0,
                )
                vega_tier1_short = ((1.0 - alpha_m_eff).unsqueeze(1) * d_pv_d_sig_m).sum(dim=2)
                vega_tier1_term = (alpha_m_eff.unsqueeze(1) * d_pv_d_sig_m).sum(dim=2)
            else:
                vega_tier1_short = torch.zeros((b_eff, d_act), dtype=torch.float64, device=device)
                vega_tier1_term = torch.zeros((b_eff, d_act), dtype=torch.float64, device=device)
            w_v_term_ij = (
                (g_corr_bm.detach() + g_corr_bm.detach().transpose(1, 2))
                if g_corr_bm is not None
                else torch.zeros_like(corr_sym_eff)
            )
        else:
            # Fallback unconditioned Tier-1 FD for exotic / mixed monitoring indices
            h_fd = 1e-4
            shifts_fw = torch.tensor([-h_fd, h_fd, 0.0, 0.0], dtype=torch.float64, device=device).view(4, 1, 1)
            shifts_vw = torch.tensor([0.0, 0.0, -h_fd, h_fd], dtype=torch.float64, device=device).view(4, 1, 1)
            f_w_pts = torch.exp(log_fw_eff.unsqueeze(0) + shifts_fw)
            v_w_pts = (v_w_eff.unsqueeze(0) + shifts_vw).clamp(min=1e-6)
            h_raw_pts = is_call_eff.unsqueeze(0) * _digital_call_spread_prob_torch(
                f_w_pts, v_w_pts, b_ko_eff.unsqueeze(0), -ko_sp_eff.unsqueeze(0)
            )
            h_mod_pts = h_raw_pts.clamp(0.0, 1.0 - 1e-6)
            if m_act == 1:
                surv_mod_pts = 1.0 - h_mod_pts
                p_mod_pts = h_mod_pts * smask_eff_64.unsqueeze(0)
            else:
                surv_mod_pts = torch.cumprod(1.0 - h_mod_pts, dim=-1)
                surv_prev_pts = torch.cat(
                    [torch.ones((4, b_eff, 1), dtype=torch.float64, device=device), surv_mod_pts[:, :, :-1]],
                    dim=-1,
                )
                p_mod_pts = surv_prev_pts * h_mod_pts * smask_eff_64.unsqueeze(0)
            q_raw_pts = _digital_call_spread_prob_torch(
                f_w_pts, v_w_pts, b_cpn_eff.unsqueeze(0), -cpn_sp_eff.unsqueeze(0)
            )
            q_mod_pts = (is_guar_eff.unsqueeze(0) + (1.0 - is_guar_eff.unsqueeze(0)) * q_raw_pts) * smask_eff_64.unsqueeze(0)
            if all_sched_act:
                f_w_term_pts = f_w_pts[:, :, -1]
                v_w_term_pts = v_w_pts[:, :, -1]
                surv_term_pts = surv_mod_pts[:, :, -1]
            else:
                assert idx_last_eff is not None
                idx_last_3d = idx_last_eff.view(1, b_eff, 1).expand(4, b_eff, 1)
                f_w_term_pts = f_w_pts.gather(2, idx_last_3d).squeeze(2)
                v_w_term_pts = v_w_pts.gather(2, idx_last_3d).squeeze(2)
                surv_term_pts = surv_mod_pts.gather(2, idx_last_3d).squeeze(2)
            l_put_pts = surv_term_pts * _batched_put_spread_loss_torch(
                f_w_term_pts, v_w_term_pts, k_strike_eff.unsqueeze(0), k_low_eff.unsqueeze(0),
                active_ki_eff.unsqueeze(0), b_ki_eff.unsqueeze(0), ko_sp_eff.squeeze(1).unsqueeze(0),
            )
            dp_dfw = (p_mod_pts[1] - p_mod_pts[0]) / (2.0 * h_fd)
            dq_dfw = (q_mod_pts[1] - q_mod_pts[0]) / (2.0 * h_fd)
            dlput_dfw = (l_put_pts[1] - l_put_pts[0]) / (2.0 * h_fd)
            g_lput_dfw = g_glob_64[:, int(AutocallGlobalCol.L_PUT_PROXY)] * dlput_dfw
            d_pv_d_log_fw_m = (
                g_sched_64[:, :, int(AutocallSchedCol.STULZ_FWD_W)] * smask_eff_64
                + g_sched_64[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] * dp_dfw
                + g_sched_64[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)] * dq_dfw
            )
            if all_sched_act:
                d_pv_d_log_fw_m[:, -1] = d_pv_d_log_fw_m[:, -1] + g_lput_dfw
            else:
                assert idx_last_eff is not None
                d_pv_d_log_fw_m.scatter_add_(1, idx_last_eff.unsqueeze(1), g_lput_dfw.unsqueeze(1))
            t1_d_log_s = d_pv_d_log_fw_m.sum(dim=1).unsqueeze(1) * w_lev_eff
            dp_dvw = (p_mod_pts[3] - p_mod_pts[2]) / (2.0 * h_fd)
            dq_dvw = (q_mod_pts[3] - q_mod_pts[2]) / (2.0 * h_fd)
            dlput_dvw = (l_put_pts[3] - l_put_pts[2]) / (2.0 * h_fd)
            g_lput_dvw = g_glob_64[:, int(AutocallGlobalCol.L_PUT_PROXY)] * dlput_dvw
            d_pv_d_vw_m = (
                g_sched_64[:, :, int(AutocallSchedCol.STULZ_VOL_W)] * smask_eff_64
                + g_sched_64[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] * dp_dvw
                + g_sched_64[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)] * dq_dvw
            )
            if all_sched_act:
                d_pv_d_vw_m[:, -1] = d_pv_d_vw_m[:, -1] + g_lput_dvw
            else:
                assert idx_last_eff is not None
                d_pv_d_vw_m.scatter_add_(1, idx_last_eff.unsqueeze(1), g_lput_dvw.unsqueeze(1))
            w_proxy_eff = node_eff_64[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)].clamp(min=0.0) * amask_eff_64
            wv_raw_eff = w_proxy_eff.unsqueeze(2) * v_m_raw_eff
            cov_w_m = (wv_raw_eff * torch.bmm(corr_sym_eff, wv_raw_eff)).sum(dim=1).clamp(min=1e-18)
            v_w_scale_m = v_w_eff / torch.sqrt(cov_w_m)
            v_m_eff = v_m_raw_eff * v_w_scale_m.unsqueeze(1)
            w_v_m = w_proxy_eff.unsqueeze(2) * v_m_eff
            cov_w_v_sum = torch.bmm(corr_sym_eff, w_v_m)
            d_vw_d_sig_m = (
                w_proxy_eff.unsqueeze(2) * sqrt_tau_m_eff.unsqueeze(1) * v_w_scale_m.unsqueeze(1) * cov_w_v_sum
            ) / v_w_eff.unsqueeze(1)
            smask_eff_bool = smask_eff_64 > 0.5
            d_pv_d_sig_m = torch.where(
                smask_eff_bool.unsqueeze(1),
                d_pv_d_vw_m.unsqueeze(1) * d_vw_d_sig_m,
                0.0,
            )
            vega_tier1_short = ((1.0 - alpha_m_eff).unsqueeze(1) * d_pv_d_sig_m).sum(dim=2)
            vega_tier1_term = (alpha_m_eff.unsqueeze(1) * d_pv_d_sig_m).sum(dim=2)
            w_bm_term = torch.where(smask_eff_bool, d_pv_d_vw_m / v_w_eff, 0.0).unsqueeze(1) * 2.0
            w_v_term_ij = torch.bmm(w_v_m * w_bm_term, w_v_m.transpose(1, 2))

        # Edge LOG_PERF_DIFF and DIST_TO_PUT / LOG_FWD_CARRY / CASH_DIV_LEVERAGE spot chain rules
        g_lpd = g_edge_64[:, :, :, int(AutocallEdgeCol.LOG_PERF_DIFF)] * pair_mask_eff
        sign_fp = torch.sign(log_fp_eff.unsqueeze(2) - log_fp_eff.unsqueeze(1))
        d_edge_lpd_d_log_s = ((g_lpd + g_lpd.transpose(1, 2)) * sign_fp).sum(dim=2) * cdl_term_eff

        g_dist_put = g_node_64[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)] * (cdl_term_eff / v_i_term_eff)
        g_carry = g_node_64[:, :, int(AutocallNodeCol.LOG_FWD_CARRY_1Y)] * ((cdl_term_eff - 1.0) / tau_term_eff)
        g_div_lev = g_node_64[:, :, int(AutocallNodeCol.CASH_DIV_LEVERAGE)] * (1.0 - cdl_term_eff)

        g_log_s_total = (
            g_node_64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)]
            + g_dist_put
            + g_carry
            + g_div_lev
            + d_edge_lpd_d_log_s
            + t1_d_log_s
        )

        g_cov_64 = g_edge_64[:, :, :, int(AutocallEdgeCol.COVARIANCE_TERM)] * pair_mask_eff
        vega_cov_term = (
            (g_cov_64 + g_cov_64.transpose(1, 2))
            * corr_sym_eff
            * v_i_term_eff.unsqueeze(1)
            * torch.sqrt(tau_term_eff).unsqueeze(-1)
        ).sum(dim=2) + (
            g_edge_64[:, :, :, int(AutocallEdgeCol.COVARIANCE_TERM)] * (eye_d_64)
        ).diagonal(dim1=1, dim2=2) * (2.0 * iv_term_node_eff * tau_term_eff)
        vega_dist_term = (
            -g_node_64[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)]
            * node_eff_64[:, :, int(AutocallNodeCol.DIST_TO_PUT_STRIKE)]
            / iv_term_node_eff
        )
        g_node_64[:, :, int(AutocallNodeCol.IV_SHORT_ATM)] = (
            g_node_64[:, :, int(AutocallNodeCol.IV_SHORT_ATM)] + vega_tier1_short
        )
        g_node_64[:, :, int(AutocallNodeCol.IV_TERM_ATM)] = (
            g_node_64[:, :, int(AutocallNodeCol.IV_TERM_ATM)]
            + vega_tier1_term
            + vega_cov_term
            + vega_dist_term
        )

        g_corr_direct = g_edge_64[:, :, :, int(AutocallEdgeCol.CORR_RHO)]
        cega_cov_term = (g_cov_64 + g_cov_64.transpose(1, 2)) * (
            v_i_term_eff.unsqueeze(2) * v_i_term_eff.unsqueeze(1)
        )
        cega_act_64 = ((g_corr_direct + g_corr_direct.transpose(1, 2)) + cega_cov_term + w_v_term_ij) * pair_mask_eff

        cash_gamma_t: torch.Tensor | None = None
        if compute_cash_gamma:
            g_eps_p = g_log_s_total[:batch_size].sum(dim=1) / (1.0 + h_cg)
            g_eps_m = g_log_s_total[batch_size:].sum(dim=1) / (1.0 - h_cg)
            cash_gamma_t = (g_eps_p - g_eps_m) / (2.0 * h_cg)

            g_log_s_base = 0.5 * (g_log_s_total[:batch_size] + g_log_s_total[batch_size:])
            g_node_base = 0.5 * (g_node_64[:batch_size] + g_node_64[batch_size:])
            g_sched_base = 0.5 * (g_sched_64[:batch_size] + g_sched_64[batch_size:])
            cega_base = 0.5 * (cega_act_64[:batch_size] + cega_act_64[batch_size:])
            pv_base = 0.5 * (pv_64[:batch_size] + pv_64[batch_size:])
            p_stop_base = 0.5 * (p_stop_64[:batch_size] + p_stop_64[batch_size:])
            surv_base = 0.5 * (surv_64[:batch_size] + surv_64[batch_size:])
            q_cpn_base = 0.5 * (q_cpn_64[:batch_size] + q_cpn_64[batch_size:])
            put_loss_base = 0.5 * (put_loss_64[:batch_size] + put_loss_64[batch_size:])
        else:
            g_log_s_base = g_log_s_total
            g_node_base = g_node_64
            g_sched_base = g_sched_64
            cega_base = cega_act_64
            pv_base = pv_64
            p_stop_base = p_stop_64
            surv_base = surv_64
            q_cpn_base = q_cpn_64
            put_loss_base = put_loss_64

        g_node_f64 = torch.zeros((batch_size, d_max, d_node), dtype=torch.float64, device=device)
        g_node_f64[:, :d_act, :] = g_node_base

        rho_df_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
        rho_df_f64[:, :m_act] = g_sched_base[:, :, int(AutocallSchedCol.DF_PAY)] * smask_64

        stop_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
        stop_f64[:, :m_act] = p_stop_base

        surv_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
        surv_f64[:, :m_act] = surv_base
        if m_act < m_max:
            surv_f64[:, m_act:] = surv_f64[:, m_act - 1 : m_act]

        cpn_f64 = torch.zeros((batch_size, m_max), dtype=torch.float64, device=device)
        cpn_f64[:, :m_act] = q_cpn_base
        put_loss_f64 = put_loss_base

        if spots_t0 is not None:
            sp_f64 = spots_t0.to(dtype=torch.float64, device=device)[:, :d_act].clamp(min=1e-12)
        else:
            log_m0_f64 = node_64[:, :, int(AutocallNodeCol.LOG_MONEYNESS)]
            sp_f64 = torch.exp(log_m0_f64).clamp(min=1e-12)
            if ref_spots is not None:
                sp_f64 = sp_f64 * ref_spots.to(dtype=torch.float64, device=device)[:, :d_act].clamp(min=1e-12)

        delta_f64 = torch.zeros((batch_size, d_max), dtype=torch.float64, device=device)
        delta_f64[:, :d_act] = (g_log_s_base / sp_f64) * amask_64

        vega_surface_f64 = g_node_f64[:, :, 4:11] * amask_f64.unsqueeze(-1)
        vega_atm_f64 = g_node_f64[:, :, 4:6] * amask_f64.unsqueeze(-1)

        cega_f64 = torch.zeros((batch_size, d_max, d_max), dtype=torch.float64, device=device)
        cega_f64[:, :d_act, :d_act] = cega_base

        pv_f64 = pv_base
        if pv_in_transit is not None:
            pv_f64 = pv_f64 + torch.as_tensor(pv_in_transit, dtype=torch.float64, device=device)

    return BatchedSurrogateGreeksOutput(
        product_value=pv_f64,
        stopping_probs=stop_f64,
        survival_probs=surv_f64,
        coupon_probs=cpn_f64,
        put_loss=put_loss_f64.detach(),
        delta=delta_f64,
        vega_atm=vega_atm_f64,
        vega_surface_feats=vega_surface_f64,
        rho_df=rho_df_f64,
        cega=cega_f64,
        cash_gamma_euler=cash_gamma_t,
    )


def _clone_state_with_overrides(
    state: AutocallPreprocState,
    *,
    spots_t0: torch.Tensor | None = None,
    corr_matrix: torch.Tensor | None = None,
    df_pay: torch.Tensor | None = None,
    df_obs: torch.Tensor | None = None,
    vol_surfaces: dict[str, RegularizedVolSurface] | None = None,
    div_pv: torch.Tensor | None = None,
) -> AutocallPreprocState:
    """Shallow-clones an AutocallPreprocState with bumped Tier-1 fields for CRN QMC Greeks."""
    from dataclasses import replace as _dc_replace

    new_spots = spots_t0 if spots_t0 is not None else state.spots_t0
    new_moneyness = new_spots / state.ref_spots.clamp(min=1e-12)
    div_sched = state.dividend_schedule
    new_div_pv = div_pv if div_pv is not None else div_sched.div_pv
    new_df_obs = df_obs if df_obs is not None else state.df_obs
    new_df_pay = df_pay if df_pay is not None else state.df_pay

    spots_col = new_spots.unsqueeze(1)
    ref_col = state.ref_spots.unsqueeze(1)
    new_escrowed = _escrowed_spot_softplus(spots_col, new_div_pv)
    df_repo_quanto = torch.exp(div_sched.r_tot_obs) * div_sched.df_fin_obs.clamp(min=1e-15)
    new_growth = df_repo_quanto / new_df_obs.clamp(min=1e-15).unsqueeze(0)
    new_fwd = new_escrowed * new_growth
    new_fwd_rel = new_fwd / spots_col.clamp(min=1e-12)
    new_fwd_perf = new_fwd / ref_col.clamp(min=1e-12)
    new_lev = spots_col / new_escrowed.clamp(min=1e-12)

    new_div_sched = _dc_replace(
        div_sched,
        div_pv=new_div_pv,
        escrowed_spot=new_escrowed,
        forward_price=new_fwd,
        forward_rel=new_fwd_rel,
        forward_perf=new_fwd_perf,
        cash_div_leverage=new_lev,
    )
    return _dc_replace(
        state,
        spots_t0=new_spots,
        moneyness_t0=new_moneyness,
        corr_matrix=corr_matrix if corr_matrix is not None else state.corr_matrix,
        df_obs=new_df_obs,
        df_pay=new_df_pay,
        forward_rel=new_fwd_rel,
        forward_perf=new_fwd_perf,
        cash_div_leverage=new_lev,
        dividend_schedule=new_div_sched,
        vol_surfaces=vol_surfaces if vol_surfaces is not None else state.vol_surfaces,
    )


def compute_qmc_crn_greeks(
    state: AutocallPreprocState,
    *,
    queries: Sequence[str] | None = None,
    config: DupireQMCConfig | None = None,
) -> AutocallGreeksResult:
    """Computes Common Random Numbers (CRN) QMC Greeks for Tier-4 QMC-routed exotic trades.

    Reuses the exact RQMC Sobol stream across bumped states (`DupireQMCGpuSolver`) so pathwise
    tangent Delta and CRN central-difference Gamma/CashGamma/Vega/Rho/Cega/Theta remain low-variance.
    """
    eff_cfg = config if config is not None else DupireQMCConfig()
    qmc_device = (
        torch.device("cuda")
        if (state.spots_t0.device.type == "cpu" and torch.cuda.is_available())
        else state.spots_t0.device
    )
    solver = DupireQMCGpuSolver(device=qmc_device, backend="auto")
    base_res = solver.simulate_and_price(state, config=eff_cfg)

    raw_queries = queries if queries is not None else state.queries
    q_set = {str(getattr(q, "value", q)) for q in raw_queries}

    want_delta = "Delta" in q_set
    want_gamma = "Gamma" in q_set
    want_cash_gamma = "MultivariateAggregatedCashGamma" in q_set
    want_vega = "Vega" in q_set
    want_rho = "Rho" in q_set
    want_divs = "DividendsSensitivities" in q_set
    want_theta = "Theta" in q_set
    want_cega = "Cega" in q_set or "ConvexCega" in q_set

    d_assets = state.d_assets
    m_rem = state.m_rem
    orig_ids = state.original_identifiers
    sorted_ids = state.sorted_identifiers
    inv_perm = state.inv_lex_perm
    device = state.spots_t0.device
    inv_idx_t = torch.tensor(inv_perm, dtype=torch.long, device=device)

    bump_states: list[AutocallPreprocState] = []
    bump_meta: list[tuple[str, Any]] = []

    base_spots = state.spots_t0.detach().to(dtype=torch.float64, device=device)
    spots_orig = base_spots.index_select(0, inv_idx_t)

    # 1. Spot bumps for Delta / Gamma (2*D + cross pairs) or Euler CashGamma (2)
    h_spot_rel = 5.0e-3
    h_spots = torch.clamp(base_spots * h_spot_rel, min=1e-4)
    if (want_delta or want_gamma or want_cash_gamma) and m_rem > 0 and not state.already_autocalled:
        if want_delta or want_gamma:
            for i in range(d_assets):
                sp_p = base_spots.clone()
                sp_p[i] = sp_p[i] + h_spots[i]
                sp_m = base_spots.clone()
                sp_m[i] = sp_m[i] - h_spots[i]
                bump_states.append(_clone_state_with_overrides(state, spots_t0=sp_p))
                bump_meta.append(("sp_p", i))
                bump_states.append(_clone_state_with_overrides(state, spots_t0=sp_m))
                bump_meta.append(("sp_m", i))
        if want_gamma:
            for i in range(d_assets):
                for j in range(i + 1, d_assets):
                    sp_pp = base_spots.clone()
                    sp_pp[i] = sp_pp[i] + h_spots[i]
                    sp_pp[j] = sp_pp[j] + h_spots[j]
                    sp_mm = base_spots.clone()
                    sp_mm[i] = sp_mm[i] - h_spots[i]
                    sp_mm[j] = sp_mm[j] - h_spots[j]
                    bump_states.append(_clone_state_with_overrides(state, spots_t0=sp_pp))
                    bump_meta.append(("sp_pp", (i, j)))
                    bump_states.append(_clone_state_with_overrides(state, spots_t0=sp_mm))
                    bump_meta.append(("sp_mm", (i, j)))
        elif want_cash_gamma:
            bump_states.append(_clone_state_with_overrides(state, spots_t0=base_spots * (1.0 + h_spot_rel)))
            bump_meta.append(("euler_p", 0))
            bump_states.append(_clone_state_with_overrides(state, spots_t0=base_spots * (1.0 - h_spot_rel)))
            bump_meta.append(("euler_m", 0))

    # 2. Convexity-safe vol surface quote bumps for Vega on active tenors up to tau_max
    as_of_orig = state.seasoned_state.as_of_date
    if want_vega and m_rem > 0 and not state.already_autocalled:
        tau_max = float(state.tau_obs.max().item()) if state.tau_obs.numel() > 0 else 0.0
        for ticker in sorted_ids:
            surf = state.vol_surfaces.get(ticker)
            if surf is None:
                continue
            surf_dcc = getattr(surf, "dcc", "Act365")
            surf_interp = getattr(surf, "interp_method", "CubicSpline")
            surf_extrap = getattr(surf, "extrap_method", "Flat")
            surf_max_lv = getattr(surf, "max_local_vol", 2.50)
            for e_idx in range(surf.n_exp):
                # Local vol piecewise-linear variance in tenor never reads slices strictly beyond tau_max's bracket
                if e_idx > 0 and float(surf.tenors[e_idx - 1]) > tau_max + 1e-8:
                    continue
                for k_idx in range(surf.n_raw_strikes):
                    s_up = None
                    eff_h = 0.0
                    for h_try in (1.0e-3, 2.5e-4, 1.0e-4, 2.5e-5):
                        rq_up = np.array(surf.raw_quotes, copy=True)
                        rq_up[e_idx, k_idx] += h_try
                        try:
                            s_up = RegularizedVolSurface(
                                {
                                    "identifier": ticker,
                                    "tenors": surf.tenors,
                                    "strikes": surf.raw_strikes,
                                    "quotes": rq_up,
                                    "DayCountConvention": surf_dcc,
                                    "SmileInterpolationMethod": surf_interp,
                                    "SmileExtrapolationMethod": surf_extrap,
                                },
                                as_of_date=as_of_orig,
                                min_vol=surf.min_vol,
                                max_local_vol=surf_max_lv,
                            )
                            eff_h = h_try
                            break
                        except ValueError:
                            continue
                    if s_up is None:
                        # Local bell stencil preserves unit integrated bump while eliminating single-strike curvature spikes
                        raw_w = [(k_idx - 1, 0.25), (k_idx, 0.50), (k_idx + 1, 0.25)]
                        valid_w = [(kk, ww) for kk, ww in raw_w if 0 <= kk < surf.n_raw_strikes]
                        w_sum = sum(ww for _, ww in valid_w)
                        for h_try in (1.0e-3, 2.5e-4, 1.0e-4, 2.5e-5):
                            rq_up = np.array(surf.raw_quotes, copy=True)
                            for kk, ww in valid_w:
                                rq_up[e_idx, kk] += h_try * (ww / w_sum)
                            try:
                                s_up = RegularizedVolSurface(
                                    {
                                        "identifier": ticker,
                                        "tenors": surf.tenors,
                                        "strikes": surf.raw_strikes,
                                        "quotes": rq_up,
                                        "DayCountConvention": surf_dcc,
                                        "SmileInterpolationMethod": surf_interp,
                                        "SmileExtrapolationMethod": surf_extrap,
                                    },
                                    as_of_date=as_of_orig,
                                    min_vol=surf.min_vol,
                                    max_local_vol=surf_max_lv,
                                )
                                eff_h = h_try
                                break
                            except ValueError:
                                continue
                    if s_up is not None:
                        new_surfs = dict(state.vol_surfaces)
                        new_surfs[ticker] = s_up
                        bump_states.append(_clone_state_with_overrides(state, vol_surfaces=new_surfs))
                        bump_meta.append(("vega", (ticker, e_idx, k_idx, eff_h)))

    # 3. Rho pillar bumps on discount curve (1 bp = 1e-4)
    disc_curve = _normalize_curve_obj(state.discount_curve, as_of_date=as_of_orig)
    disc_id = str(getattr(disc_curve, "identifier", state.discount_curve_id or "DEFAULT"))
    k_pil = len(disc_curve.tenors) if hasattr(disc_curve, "tenors") else 0
    h_rho = 1.0e-4
    if want_rho and k_pil > 0:
        for p_idx in range(k_pil):
            p_vec = torch.zeros(k_pil, dtype=torch.float64, device=device)
            p_vec[p_idx] = h_rho
            m_obs = _curve_df_delta_multiplier(
                disc_curve, state.seasoned_state.future_obs_dates, p_vec, as_of_date=as_of_orig
            )
            m_pay = _curve_df_delta_multiplier(
                disc_curve, state.seasoned_state.future_pay_dates, p_vec, as_of_date=as_of_orig
            )
            bump_states.append(
                _clone_state_with_overrides(
                    state,
                    df_obs=state.df_obs.to(device=device, dtype=torch.float64) * m_obs,
                    df_pay=state.df_pay.to(device=device, dtype=torch.float64) * m_pay,
                )
            )
            bump_meta.append(("rho", p_idx))

    # 4. Cega off-diagonal correlation bumps (+h_corr)
    h_corr = 1.0e-2
    if want_cega and d_assets > 1 and m_rem > 0 and not state.already_autocalled:
        for i in range(d_assets):
            for j in range(i + 1, d_assets):
                c_up = state.corr_matrix.detach().clone().to(dtype=torch.float64, device=device)
                c_up[i, j] = (c_up[i, j] + h_corr).clamp(-0.999, 0.999)
                c_up[j, i] = c_up[i, j]
                bump_states.append(_clone_state_with_overrides(state, corr_matrix=c_up))
                bump_meta.append(("cega", (i, j)))

    # Execute CRN QMC simulation across base + all bumped states together
    all_states = [state] + bump_states
    pv_batch_t, _, _, _, _ = solver.simulate_batch_for_training(
        all_states, config=eff_cfg, compute_pathwise_delta=False
    )
    pv_arr = pv_batch_t.detach().cpu().numpy().astype(np.float64)
    pv0 = float(pv_arr[0])

    tag_to_pv: dict[tuple[str, Any], float] = {}
    vega_meta_lookup: dict[tuple[str, int, int], tuple[float, float]] = {}
    for idx_m, meta in enumerate(bump_meta):
        val = float(pv_arr[idx_m + 1])
        tag_to_pv[meta] = val
        if meta[0] == "vega":
            tk, e_i, k_i, h_used = meta[1]
            vega_meta_lookup[(tk, e_i, k_i)] = (val, h_used)

    delta_dict: dict[str, float] | None = None
    delta_tensor_orig: torch.Tensor | None = None
    if want_delta:
        if m_rem > 0 and not state.already_autocalled:
            # CRN central difference avoids pathwise digital-spread spike artifacts on tight spreads
            d_sorted = torch.tensor(
                [(tag_to_pv[("sp_p", i)] - tag_to_pv[("sp_m", i)]) / (2.0 * float(h_spots[i].item())) for i in range(d_assets)],
                dtype=torch.float64,
                device=device,
            )
        else:
            d_sorted = torch.zeros(d_assets, dtype=torch.float64, device=device)
        delta_tensor_orig = d_sorted.index_select(0, inv_idx_t)
        delta_dict = {orig_ids[i]: float(delta_tensor_orig[i].cpu().item()) for i in range(d_assets)}

    gamma_dict: dict[str, dict[str, float]] | None = None
    gamma_tensor_orig: torch.Tensor | None = None
    cash_gamma_agg: float | None = None
    if want_gamma:
        hess_sorted = torch.zeros((d_assets, d_assets), dtype=torch.float64, device=device)
        if m_rem > 0 and not state.already_autocalled:
            for i in range(d_assets):
                hi = float(h_spots[i].item())
                hess_sorted[i, i] = (tag_to_pv[("sp_p", i)] - 2.0 * pv0 + tag_to_pv[("sp_m", i)]) / (hi * hi)
            for i in range(d_assets):
                for j in range(i + 1, d_assets):
                    hi = float(h_spots[i].item())
                    hj = float(h_spots[j].item())
                    d2 = tag_to_pv[("sp_pp", (i, j))] - 2.0 * pv0 + tag_to_pv[("sp_mm", (i, j))]
                    gij = (d2 - hi * hi * float(hess_sorted[i, i]) - hj * hj * float(hess_sorted[j, j])) / (2.0 * hi * hj)
                    hess_sorted[i, j] = gij
                    hess_sorted[j, i] = gij
        gamma_tensor_orig = hess_sorted.index_select(0, inv_idx_t).index_select(1, inv_idx_t)
        gamma_dict = {
            orig_ids[i]: {orig_ids[j]: float(gamma_tensor_orig[i, j].cpu().item()) for j in range(d_assets)}
            for i in range(d_assets)
        }
        if want_cash_gamma:
            s_out = spots_orig.unsqueeze(1) * spots_orig.unsqueeze(0)
            cash_gamma_agg = float((s_out * gamma_tensor_orig).sum().cpu().item())
    elif want_cash_gamma:
        if m_rem > 0 and not state.already_autocalled:
            if ("euler_p", 0) in tag_to_pv:
                cash_gamma_agg = (tag_to_pv[("euler_p", 0)] - 2.0 * pv0 + tag_to_pv[("euler_m", 0)]) / (h_spot_rel * h_spot_rel)
            else:
                state_ep = _clone_state_with_overrides(state, spots_t0=base_spots * (1.0 + h_spot_rel))
                state_em = _clone_state_with_overrides(state, spots_t0=base_spots * (1.0 - h_spot_rel))
                pv_e_t, _, _, _, _ = solver.simulate_batch_for_training(
                    [state_ep, state_em], config=eff_cfg, compute_pathwise_delta=False
                )
                cash_gamma_agg = (float(pv_e_t[0].item()) - 2.0 * pv0 + float(pv_e_t[1].item())) / (h_spot_rel * h_spot_rel)
        else:
            cash_gamma_agg = 0.0

    vega_dict: dict[str, list[list[float]]] | None = None
    vega_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    if want_vega:
        vega_dict = {}
        vega_arrays = {}
        for ticker in orig_ids:
            surf = state.vol_surfaces.get(ticker)
            if surf is None:
                arr_v = np.zeros((0, 0), dtype=np.float64)
                vega_arrays[ticker] = arr_v
                vega_dict[ticker] = []
                continue
            arr_v = np.zeros((surf.n_exp, surf.n_raw_strikes), dtype=np.float64)
            if m_rem > 0 and not state.already_autocalled:
                for e_idx in range(surf.n_exp):
                    for k_idx in range(surf.n_raw_strikes):
                        entry = vega_meta_lookup.get((ticker, e_idx, k_idx))
                        if entry is not None:
                            arr_v[e_idx, k_idx] = (entry[0] - pv0) / entry[1]
            vega_arrays[ticker] = arr_v
            vega_dict[ticker] = [[float(x) for x in row] for row in arr_v]

    rho_dict: dict[str, list[float]] | None = None
    rho_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    if want_rho:
        arr_r = np.array(
            [(tag_to_pv[("rho", p_idx)] - pv0) / h_rho for p_idx in range(k_pil)],
            dtype=np.float64,
        )
        rho_arrays = {disc_id: arr_r}
        rho_dict = {disc_id: [float(x) for x in arr_r]}

    divs_dict: dict[str, list[float]] | None = None
    divs_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    if want_divs:
        divs_dict = {}
        divs_arrays = {}
        h_div = 1.0e-3
        for i_s, ticker in enumerate(sorted_ids):
            events_i = state.dividend_schedule.dividend_events.get(ticker, [])
            arr_d = np.zeros(len(events_i), dtype=np.float64)
            if len(events_i) > 0 and m_rem > 0 and not state.already_autocalled:
                w_base = state.dividend_schedule.dividend_pv_weights[ticker].to(device=device, dtype=torch.float64)
                inc = state.dividend_schedule.dividend_obs_incidence[ticker].to(device=device, dtype=torch.float64)
                div_bumped_states: list[AutocallPreprocState] = []
                for k_ev in range(len(events_i)):
                    dpv = state.dividend_schedule.div_pv.clone().to(device=device, dtype=torch.float64)
                    dpv[i_s] = dpv[i_s] + h_div * w_base[k_ev] * inc[k_ev]
                    div_bumped_states.append(_clone_state_with_overrides(state, div_pv=dpv))
                pv_div_t, _, _, _, _ = solver.simulate_batch_for_training(
                    div_bumped_states, config=eff_cfg, compute_pathwise_delta=False
                )
                pv_div_arr = pv_div_t.detach().cpu().numpy().astype(np.float64)
                arr_d = (pv_div_arr - pv0) / h_div
            divs_arrays[ticker] = arr_d
            divs_dict[ticker] = [float(x) for x in arr_d]

    cega_dict: dict[str, dict[str, float]] | None = None
    cega_tensor_orig: torch.Tensor | None = None
    if want_cega:
        cega_sorted = torch.zeros((d_assets, d_assets), dtype=torch.float64, device=device)
        if m_rem > 0 and not state.already_autocalled:
            for i in range(d_assets):
                for j in range(i + 1, d_assets):
                    c_val = 2.0 * (tag_to_pv[("cega", (i, j))] - pv0) / h_corr
                    cega_sorted[i, j] = c_val
                    cega_sorted[j, i] = c_val
        cega_tensor_orig = cega_sorted.index_select(0, inv_idx_t).index_select(1, inv_idx_t)
        cega_dict = {
            orig_ids[i]: {orig_ids[j]: float(cega_tensor_orig[i, j].cpu().item()) for j in range(d_assets)}
            for i in range(d_assets)
        }

    theta_val: float | None = None
    if want_theta:
        with torch.no_grad():
            as_of_rolled = as_of_orig + dt.timedelta(days=1) if isinstance(as_of_orig, dt.datetime) else None
            pv_r = evaluate_differentiable_pv(
                state,
                surrogate=None,
                as_of_date_override=as_of_rolled,
                tau_shift=0.0 if as_of_rolled is not None else 1.0 / 365.0,
                return_postproc=False,
            )
            if not isinstance(pv_r, torch.Tensor):
                raise RuntimeError("Expected torch.Tensor from rolled Theta evaluation.")
            theta_val = float(pv_r.item() - pv0)

    return AutocallGreeksResult(
        product_value=pv0,
        cash_flows=list(base_res.cashflows),
        delta=delta_dict,
        gamma=gamma_dict,
        multivariate_aggregated_cash_gamma=cash_gamma_agg,
        vega=vega_dict,
        rho=rho_dict,
        dividends_sensitivities=divs_dict,
        theta=theta_val,
        cega=cega_dict,
        delta_tensor=delta_tensor_orig,
        gamma_tensor=gamma_tensor_orig,
        vega_arrays=vega_arrays,
        rho_arrays=rho_arrays,
        dividends_arrays=divs_arrays,
        cega_tensor=cega_tensor_orig,
        reporting_currency=state.payment_currency,
        standard_error=float(base_res.standard_error),
        cash_to_settle_today=float(state.cash_to_settle_today),
        greeks_engine_used=f"qmc_crn_{solver.backend_name}",
    )


__all__ = [
    "AutocallGreeksResult",
    "BatchedSurrogateGreeksOutput",
    "cast_to_surrogate_boundary",
    "cast_from_surrogate_boundary",
    "evaluate_differentiable_pv",
    "compute_euler_cash_gamma",
    "compute_eq_autocall_greeks",
    "compute_batched_surrogate_greeks",
    "compute_qmc_crn_greeks",
]
