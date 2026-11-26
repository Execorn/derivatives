"""dupire_qmc_gpu.py: GPU-vectorized Owen-scrambled RQMC Dupire local-vol solver and PEMC.

Implements SPEC-DEEPVOL-EQ-AUTOCALL-002 §5.5.2 for NVIDIA Ampere (`sm_86`) and CPU fallback:
  - `DupireQMCConfig` & `DupireQMCBackend` protocol unifying:
      * `GpuTritonDupireQMC`: Adaptive (D, 64, 128) GPU 2D Dupire LV table, on-the-fly
        GPU Joe-Kuo Gray-code Sobol (`<= 21,201` dims) + GPU Philox antithetic normal
        fallback (`dim >= 21,201`), Triton-fused substep kernel (FP32 diffusion + FP64
        martingale renormalization & pathwise memory-coupon/funding accumulation), and
        strictly unbiased control variates (`bs1d_exact` / `multifidelity_gbm`).
      * `GpuTorchDupireQMC`: Pure PyTorch CUDA backend (zero Triton dependency fallback).
      * `CpuNumpyDupireQMC`: IEEE-754 float64 CPU fallback delegating to `dupire_qmc_solver`.
      * `DupireQMCGpuSolver`: Primary facade with multi-contract `(D, M_rem)`-bucketed
        `simulate_batch_for_training(..., compute_pathwise_delta=True)`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
import threading
from typing import Any, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
from scipy.stats import qmc
import torch
import torch.nn.functional as F

from deepvol.models.autocall.dupire_qmc_solver import (
    _DIFF_INDEX_CODES,
    _PAIR_ORDER_INDEX_CODES,
    DupireQMCResult,
    _assemble_qmc_postproc,
    _black76_put_np,
    _compute_funding_weights_np,
    _next_power_of_two,
    _resolve_r_native_obs_tensor,
    price_eq_autocall_dupire_qmc,
    resolve_monitoring_index_code,
)
from deepvol.models.autocall.feature_layout import MonitoringIndexCode
from deepvol.models.autocall.gpu_cache_manager import (
    GLOBAL_GPU_CACHE_MANAGER,
    GlobalGpuCacheManager,
    PartitionedGpuLRU,
    normalize_cuda_device,
)
from deepvol.models.autocall.pre_post_processor import (
    AutocallPreprocState,
    assemble_autocall_cashflows_and_pv,
)

try:
    import importlib

    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
    from triton.runtime import driver as _triton_driver  # type: ignore[import-untyped]

    _triton_knobs: Any = getattr(triton, "knobs", None)
    if _triton_knobs is None:
        try:
            _triton_knobs = importlib.import_module("triton.runtime.knobs")
        except ImportError:
            _triton_knobs = None

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _triton_driver = None
    _triton_knobs = None
    _HAS_TRITON = False

_TRITON_PLAN_FAST: Dict[int, Tuple[Any, Tuple[Tuple[str, bool, Any], ...], int, bool, Tuple[Tuple[str, Any], ...]]] = {}
_TRITON_COMPILED_FAST: Dict[Any, Tuple[Any, Any, Any]] = {}
_MISSING_ARG = object()
_GET_RAW_CUDA_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _arr_bytes_equal(a: Any, snap: npt.NDArray[Any]) -> bool:
    """Fast exact content check via C-contiguous uint8 memoryview before array_equal fallback."""
    if (
        isinstance(a, np.ndarray)
        and a.dtype == snap.dtype
        and a.shape == snap.shape
        and a.flags.c_contiguous
        and snap.flags.c_contiguous
    ):
        return bool(
            memoryview(cast(Any, a.reshape(-1).view(np.uint8)))
            == memoryview(cast(Any, snap.reshape(-1).view(np.uint8)))
        )
    return bool(np.array_equal(a, snap))


def _append_triton_arg_sig(sig_parts: List[Any], v: Any, tensor_cls: type) -> None:
    tv = type(v)
    if tv is tensor_cls:
        sig_parts.append(v.dtype)
        sig_parts.append((v.data_ptr() & 15) == 0)
    elif tv is int:
        sig_parts.append(1 if v == 1 else (2 if (v & 15) == 0 else 0))
        sig_parts.append(-2147483648 <= v <= 2147483647)
    elif v is None:
        sig_parts.append(None)
    elif tv is float or tv is bool:
        sig_parts.append(tv)
    elif hasattr(v, "data_ptr"):
        sig_parts.append(v.dtype)
        sig_parts.append((int(v.data_ptr()) & 15) == 0)
    else:
        sig_parts.append(tv)


def _fast_triton_launch(jit_fn: Any, grid: Tuple[int, ...], *pos_args: Any, **kw_args: Any) -> None:
    """Bypass Triton JIT binder/specializer overhead on warm hits while preserving exact specialization."""
    knobs = _triton_knobs
    drv = _triton_driver
    if (
        drv is None
        or (
            knobs is not None
            and hasattr(knobs, "runtime")
            and (
                knobs.runtime.launch_enter_hook is not None
                or knobs.runtime.launch_exit_hook is not None
                or knobs.runtime.debug
            )
        )
        or getattr(jit_fn, "debug", False)
    ):
        jit_fn[grid](*pos_args, **kw_args)
        return

    jid = id(jit_fn)
    p_hit = _TRITON_PLAN_FAST.get(jid)
    if p_hit is not None and p_hit[0] is jit_fn:
        _, param_plan, n_lead_pos, ce_only_tail, tail_ce = p_hit
    else:
        param_plan = tuple((kp.name, bool(kp.is_constexpr), kp.default) for kp in jit_fn.params)
        n_lead_pos = 0
        for _, is_ce_p, _ in param_plan:
            if is_ce_p:
                break
            n_lead_pos += 1
        ce_only_tail = all(is_ce_p for _, is_ce_p, _ in param_plan[n_lead_pos:])
        tail_ce = tuple((nm_p, def_p) for nm_p, _, def_p in param_plan[n_lead_pos:]) if ce_only_tail else ()
        if len(_TRITON_PLAN_FAST) >= 256:
            _TRITON_PLAN_FAST.clear()
        _TRITON_PLAN_FAST[jid] = (jit_fn, param_plan, n_lead_pos, ce_only_tail, tail_ce)

    active_drv = drv.active
    dev_idx = active_drv.get_current_device()
    kw_get = kw_args.get
    missing = _MISSING_ARG
    tensor_cls = torch.Tensor
    n_pos = len(pos_args)
    sig_parts: List[Any] = [jid, dev_idx]

    if ce_only_tail and n_pos == n_lead_pos:
        for v in pos_args:
            _append_triton_arg_sig(sig_parts, v, tensor_cls)
        ce_vals: List[Any] = []
        if kw_args:
            for name, default in tail_ce:
                v = kw_get(name, default)
                ce_vals.append(v)
                sig_parts.append(v)
        else:
            for _, default in tail_ce:
                ce_vals.append(default)
                sig_parts.append(default)
        launch_args: Sequence[Any] = (*pos_args, *ce_vals)
    else:
        run_args: List[Any] = []
        pos_i = 0
        for name, is_ce, default in param_plan:
            v = kw_get(name, missing)
            if v is missing:
                if (not is_ce) and pos_i < n_pos:
                    v = pos_args[pos_i]
                    pos_i += 1
                else:
                    v = default
            run_args.append(v)
            if is_ce:
                sig_parts.append(v)
            else:
                _append_triton_arg_sig(sig_parts, v, tensor_cls)
        if pos_i != n_pos:
            jit_fn[grid](*pos_args, **kw_args)
            return
        launch_args = run_args

    if kw_args:
        for kw_tag, kw_key in (("nw", "num_warps"), ("ns", "num_stages"), ("nc", "num_ctas")):
            kw_val = kw_get(kw_key, missing)
            if kw_val is not missing:
                sig_parts.append((kw_tag, kw_val))

    k_key = tuple(sig_parts)
    c_hit = _TRITON_COMPILED_FAST.get(k_key)
    if c_hit is None:
        compiled = jit_fn[grid](*pos_args, **kw_args)
        compiled._init_handles()
        if len(_TRITON_COMPILED_FAST) >= 1024:
            _TRITON_COMPILED_FAST.clear()
        _TRITON_COMPILED_FAST[k_key] = (compiled.run, compiled.function, compiled.packed_metadata)
        return

    run_fn, cu_fn, packed_meta = c_hit
    g0 = grid[0]
    g1 = grid[1] if len(grid) > 1 else 1
    g2 = grid[2] if len(grid) > 2 else 1
    stream = (
        _GET_RAW_CUDA_STREAM(dev_idx)
        if _GET_RAW_CUDA_STREAM is not None
        else active_drv.get_current_stream(dev_idx)
    )
    run_fn(g0, g1, g2, stream, cu_fn, packed_meta, None, None, None, *launch_args)


def _haar_global_dims_for_period(m_rem: int, d_assets: int, n_sub: int, m: int) -> npt.NDArray[np.int64]:
    """Returns global Sobol dimension indices (D * (n_sub - 1),) for Haar Brownian bridge levels."""
    md_tot = m_rem * d_assets
    dims = np.empty((d_assets, max(n_sub - 1, 0)), dtype=np.int64)
    haar_levels = int(round(math.log2(n_sub)))
    d_idx = np.arange(d_assets, dtype=np.int64)[:, None]
    for lvl in range(haar_levels):
        k_pow = 1 << lvl
        k_idx = np.arange(k_pow, dtype=np.int64)[None, :]
        dims[:, k_pow - 1 : (k_pow - 1) + k_pow] = k_pow * md_tot + (m * d_assets + d_idx) * k_pow + k_idx
    return dims.reshape(-1)


def _haar_reconstruct_residuals_torch(b_modes: torch.Tensor, n_sub: int) -> torch.Tensor:
    """Transforms Haar bridge modes `(..., n_sub - 1, n_rep)` into zero-mean substep residuals `(..., n_sub, n_rep)`."""
    haar_levels = int(round(math.log2(n_sub)))
    out_shape = list(b_modes.shape)
    out_shape[-2] = n_sub
    acc = torch.zeros(out_shape, dtype=b_modes.dtype, device=b_modes.device)
    for lvl in range(haar_levels):
        k_pow = 1 << lvl
        span = n_sub // k_pow
        half = span // 2
        w_lvl = math.sqrt(float(k_pow))
        for j in range(k_pow):
            mode_t = w_lvl * b_modes[..., (k_pow - 1) + j : (k_pow - 1) + j + 1, :]
            s0 = j * span
            acc[..., s0 : s0 + half, :] += mode_t
            acc[..., s0 + half : s0 + span, :] -= mode_t
    return acc / math.sqrt(float(n_sub))


_SOBOL_MAX_DIM: int = 21201
_SOBOL_BITS: int = 30
_SOBOL_INV_SCALE: float = 1.0 / float(1 << _SOBOL_BITS)

_SOBOL_HOST_SV: Optional[npt.NDArray[np.int32]] = None
_SOBOL_HOST_LOCK = threading.RLock()

_STATIC_CONSTANTS_LRU: PartitionedGpuLRU[Any] = GLOBAL_GPU_CACHE_MANAGER.register_partition(
    "static_constants",
    max_entries=256,
    max_bytes=GlobalGpuCacheManager.PARTITION_BUDGETS["static_constants"],
)
_DEAL_PREP_LRU: PartitionedGpuLRU[Any] = GLOBAL_GPU_CACHE_MANAGER.register_partition(
    "deal_prep_lv_vega",
    max_entries=512,
    max_bytes=GlobalGpuCacheManager.PARTITION_BUDGETS["deal_prep_lv_vega"],
)
_LV_TABLE_LRU: PartitionedGpuLRU[Any] = GLOBAL_GPU_CACHE_MANAGER.register_partition(
    "lv_table_cache",
    max_entries=512,
    max_bytes=GlobalGpuCacheManager.PARTITION_BUDGETS["lv_table_cache"],
)
_SCHED_SUBSTEP_LOCK = threading.RLock()


@dataclass(frozen=True)
class GpuQmcStaticSchedule:
    """Immutable cached GPU device buffers for a preprocessed autocall schedule."""

    digest_hex: str
    lv_t_grid: torch.Tensor
    r_nat_obs: torch.Tensor
    r_tot_obs: torch.Tensor
    df_fin_obs: torch.Tensor
    df_pay: torch.Tensor
    fwd_perf: torch.Tensor
    ref_spots: torch.Tensor
    ko_barriers: torch.Tensor
    cpn_barriers: torch.Tensor
    is_callable: torch.Tensor
    is_guar: torch.Tensor
    coupon_amounts: torch.Tensor
    w_fund: torch.Tensor


@dataclass(frozen=True)
class _GpuQmcScheduleBundle:
    """Cached device schedule plus precomputed Cholesky, BB/PCA, quanto substeps, and CV metadata."""

    schedule: GpuQmcStaticSchedule
    lv_digest: str
    spots_t0: torch.Tensor
    v_term: torch.Tensor
    chol_f32: torch.Tensor
    a_pca_f32: torch.Tensor
    m_bb_f32: torch.Tensor
    q_shift_bnd_f32: torch.Tensor
    t_prev_f32: torch.Tensor
    tau_obs_f32: torch.Tensor
    inv_lex_perm_gpu: Optional[torch.Tensor]
    sqrt_dvar_dom_f64: torch.Tensor
    dom_idx: int
    v_dom_term: float
    tau_term: float
    f_dom: float
    k_cv: float
    k_low: float
    df_term: float
    mu_cv: float
    substep_tensors: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    tau_obs_np: npt.NDArray[np.float64]
    df_pay_np: npt.NDArray[np.float64]
    var_cum_gbm_np: npt.NDArray[np.float64]
    d_var_gbm_np: npt.NDArray[np.float64]
    max_rel_skew: float
    max_dt: float
    t_prev_tuple: Tuple[float, ...]
    dt_arr_tuple: Tuple[float, ...]
    sqrt_dvar_dom_tuple: Tuple[float, ...]
    substep_cpu_meta: Dict[int, Tuple[Tuple[float, ...], Tuple[float, ...]]]


@dataclass(frozen=True)
class DupireQMCConfig:
    """Configuration for GPU/CPU Owen-scrambled RQMC Dupire LV simulation."""

    n_paths: int = 8192
    n_replicates: int = 8
    n_substeps: int = 4
    mode: Literal["dupire_lv", "terminal_iv"] = "dupire_lv"
    precision: Literal["mixed_fp32_fp64", "fp64_audit"] = "mixed_fp32_fp64"
    cv_mode: Literal["bs1d_exact", "multifidelity_gbm", "none"] = "multifidelity_gbm"
    seed: int = 42
    lv_grid_nt: int = 32
    lv_grid_ny: int = 64
    adaptive_substeps: bool = False
    n_cheap_mf: Optional[int] = None

    @classmethod
    def from_canonical(
        cls,
        canonical: Any,
        *,
        mode: Literal["dupire_lv", "terminal_iv"] = "dupire_lv",
        precision: Literal["mixed_fp32_fp64", "fp64_audit"] = "mixed_fp32_fp64",
    ) -> "DupireQMCConfig":
        """Convert a `CanonicalTier3Config` (or compatible object) into `DupireQMCConfig`."""
        if isinstance(canonical, cls):
            return canonical
        return cls(
            n_paths=int(getattr(canonical, "n_paths", 8192)),
            n_replicates=int(getattr(canonical, "n_replicates", 8)),
            n_substeps=int(getattr(canonical, "n_substeps", 4)),
            mode=mode,
            precision=precision,
            cv_mode=cast(
                Literal["bs1d_exact", "multifidelity_gbm", "none"],
                getattr(canonical, "cv_mode", "multifidelity_gbm"),
            ),
            seed=int(getattr(canonical, "seed", 42)),
            lv_grid_nt=int(getattr(canonical, "lv_grid_nt", 32)),
            lv_grid_ny=int(getattr(canonical, "lv_grid_ny", 64)),
            adaptive_substeps=bool(getattr(canonical, "adaptive_substeps", False)),
            n_cheap_mf=(
                int(getattr(canonical, "n_cheap_mf"))
                if getattr(canonical, "n_cheap_mf", None) is not None
                else None
            ),
        )


def resolve_dupire_qmc_config(
    d_assets: int,
    m_rem: int,
    *,
    device_type: Literal["cuda", "cpu"],
    n_fallback_paths: Optional[int] = None,
    seed: int = 42,
    mode: Literal["dupire_lv", "terminal_iv"] = "dupire_lv",
    precision: Literal["mixed_fp32_fp64", "fp64_audit"] = "mixed_fp32_fp64",
) -> DupireQMCConfig:
    """Resolve deterministic `DupireQMCConfig` aligned with `CanonicalTier3Config`."""
    try:
        import deepvol.models.autocall.feature_layout as _fl_mod

        resolver = getattr(_fl_mod, "resolve_canonical_tier3_config", None)
        if callable(resolver):
            canonical = resolver(
                d_assets,
                m_rem,
                device_type=device_type,
                n_fallback_paths=n_fallback_paths,
                seed=seed,
            )
            return DupireQMCConfig.from_canonical(canonical, mode=mode, precision=precision)
    except ImportError:
        pass

    _ = (int(d_assets), int(m_rem))
    dev_tok = "cuda" if str(device_type).strip().lower().startswith("cuda") else "cpu"
    eff_paths = (
        max(2, int(n_fallback_paths))
        if n_fallback_paths is not None
        else (8192 if dev_tok == "cuda" else 4096)
    )
    eff_r = min(max(1, eff_paths // 2), 8 if (dev_tok == "cuda" or eff_paths >= 8192) else 4)
    n_sub = 4
    n_cheap = max(262144, 8 * eff_paths) if dev_tok == "cuda" else max(32768, 8 * eff_paths)
    return DupireQMCConfig(
        n_paths=eff_paths,
        n_replicates=eff_r,
        n_substeps=n_sub,
        mode=mode,
        precision=precision,
        cv_mode="multifidelity_gbm",
        seed=int(seed),
        lv_grid_nt=32,
        lv_grid_ny=64,
        adaptive_substeps=False,
        n_cheap_mf=n_cheap,
    )


def _resolve_ki_obs_mode(state: Any) -> str:
    """Resolve normalized KI observation mode (`None`/empty defaults strictly to `'European'`)."""
    ki_obs_raw = getattr(state, "ki_obs_type", None)
    if ki_obs_raw is None:
        ss = getattr(state, "seasoned_state", None)
        ki_obs_raw = getattr(ss, "ki_obs_type", None) if ss is not None else None
    if ki_obs_raw is None:
        return "European"
    ki_str = str(getattr(ki_obs_raw, "value", ki_obs_raw) or "").strip()
    if not ki_str:
        return "European"
    ki_norm = ki_str.lower()
    if ki_norm in ("european", "terminal", "atmaturity", "maturity"):
        return "European"
    if ki_norm in ("discrete", "periodic", "observation"):
        return "Discrete"
    if ki_norm == "daily":
        return "Daily"
    if ki_norm in ("continuous", "american"):
        return "Continuous"
    return "European"


def _resolve_onestar_effective_params(
    one_star_strike: float,
    ki_barrier: Optional[float],
    ko_spread: float,
) -> Tuple[float, float]:
    """Return `(k_os_eff, eps_eff)` implementing the OneStar left-spread rule."""
    if ki_barrier is not None and float(ko_spread) < -1e-9:
        sp = float(ko_spread)
        return float(one_star_strike) + 0.5 * sp, abs(sp)
    return float(one_star_strike), 0.02


def _resolve_eff_and_ki_opt_indices(state: Any) -> Tuple[str, str, bool]:
    """Return `(eff_opt_idx, ki_opt_idx, is_long)` aligned with `dupire_qmc_solver`."""
    is_long = bool(getattr(state, "is_long_basket", False))
    raw_opt_idx = getattr(state, "option_index", None)
    fallback = str(raw_opt_idx if raw_opt_idx else state.callability_index)
    eff_opt_idx = (
        "AverageOf"
        if (is_long and (raw_opt_idx is None or str(raw_opt_idx).strip() in ("", "WorstOf")))
        else fallback
    )
    ki_opt_idx = "WorstOf" if is_long else fallback
    return eff_opt_idx, ki_opt_idx, is_long


class DupireQMCBackend(Protocol):
    """Protocol seam unifying GpuTritonDupireQMC, GpuTorchDupireQMC, and CpuNumpyDupireQMC."""

    @property
    def backend_name(self) -> str:
        ...

    def simulate_and_price(
        self,
        state: AutocallPreprocState,
        config: Optional[DupireQMCConfig] = None,
    ) -> DupireQMCResult:
        """Simulate multi-asset Dupire LV RQMC paths and return float64 PV, RQMC SE, and CFs."""
        ...

    def simulate_batch_for_training(
        self,
        states: Sequence[AutocallPreprocState],
        config: Optional[DupireQMCConfig] = None,
        compute_pathwise_delta: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Multi-contract (D, M_rem)-bucketed GPU QMC teacher for batched dataset generation.

        Returns:
            Tuple of (pv_mc, p_stop_mc, q_cpn_mc, l_put_mc, delta_mc_raw_or_none) in torch.float64.
        """
        ...


def _cpu_f64_array(val: Any) -> npt.NDArray[np.float64]:
    """Extracts a contiguous float64 NumPy array without CUDA D2H sync when already on CPU."""
    if isinstance(val, torch.Tensor):
        t_cpu = val.detach() if val.device.type == "cpu" else val.detach().cpu()
        return np.ascontiguousarray(t_cpu.numpy(), dtype=np.float64)
    return np.ascontiguousarray(val, dtype=np.float64)


def _hash_update_chunk(h: Any, data: bytes) -> None:
    """Feeds length-prefixed bytes into a blake2b hasher to prevent boundary collisions."""
    h.update(len(data).to_bytes(8, "little"))
    h.update(data)


def _hash_update_arr(h: Any, arr: npt.NDArray[Any]) -> None:
    """Feeds shape and contiguous bytes of a NumPy array into a blake2b hasher."""
    c_arr = np.ascontiguousarray(arr)
    _hash_update_chunk(h, repr(c_arr.shape).encode("ascii"))
    _hash_update_chunk(h, c_arr.tobytes())


def _hash_update_str(h: Any, text: str) -> None:
    _hash_update_chunk(h, text.encode("utf-8"))


def _surface_content_digest(surf: Any) -> str:
    """Full content blake2b digest covering strikes, tenors, quotes, interpolator, and bounds."""
    pre_digest = getattr(surf, "content_digest", None)
    if isinstance(pre_digest, str) and pre_digest:
        return f"{pre_digest}:{float(surf.max_local_vol):.17g}"
    h = hashlib.blake2b(digest_size=20)
    _hash_update_arr(h, np.asarray(surf.strikes, dtype=np.float64))
    _hash_update_arr(h, np.asarray(surf.tenors, dtype=np.float64))
    _hash_update_arr(h, np.asarray(getattr(surf, "tenors_act365", surf.tenors), dtype=np.float64))
    _hash_update_arr(h, np.asarray(surf.quotes, dtype=np.float64))
    _hash_update_str(h, str(getattr(surf, "calendar_proj_digest", "")))
    _hash_update_str(h, str(getattr(surf, "smile_interp", "CubicSpline")))
    _hash_update_str(h, str(getattr(surf, "smile_extrap", "Flat")))
    _hash_update_str(h, str(getattr(surf, "bc_type", "clamped")))
    _hash_update_str(h, str(getattr(surf, "dcc", "Act365")))
    _hash_update_chunk(h, repr((float(surf.min_vol), float(surf.max_local_vol))).encode("ascii"))
    return h.hexdigest()


def _div_events_tuple(evs: Sequence[Any]) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(
        (
            float(getattr(ev, "tau_ex_act365", 0.0)),
            float(getattr(ev, "amount", 0.0)),
            float(getattr(ev, "pv_discount_weight", 0.0)),
        )
        for ev in evs
    )


def _compute_lv_batch_digest(
    surfs: Sequence[Any],
    spots_np: npt.NDArray[np.float64],
    r_tot_obs_np: npt.NDArray[np.float64],
    df_fin_obs_np: npt.NDArray[np.float64],
    tau_obs_np: npt.NDArray[np.float64],
    div_events_list: Sequence[Sequence[Any]],
    v_term_np: npt.NDArray[np.float64],
    n_t: int,
    n_y: int,
    device: torch.device,
    use_triton: bool,
    fwd_prices_np: Optional[npt.NDArray[np.float64]] = None,
) -> str:
    """Full content blake2b digest for cached LV tables."""
    h = hashlib.blake2b(digest_size=24)
    for s in surfs:
        _hash_update_str(h, _surface_content_digest(s))
    _hash_update_arr(h, spots_np)
    _hash_update_arr(h, r_tot_obs_np)
    _hash_update_arr(h, df_fin_obs_np)
    _hash_update_arr(h, tau_obs_np)
    _hash_update_arr(h, v_term_np)
    if fwd_prices_np is not None:
        _hash_update_arr(h, fwd_prices_np)
    div_tup = tuple(_div_events_tuple(evs) for evs in div_events_list)
    _hash_update_str(
        h, repr((div_tup, int(n_t), int(n_y), normalize_cuda_device(device), bool(use_triton)))
    )
    return h.hexdigest()


def _hash_curve_obj(h: Any, curve: Any) -> None:
    if curve is None:
        _hash_update_str(h, "None")
        return
    if isinstance(curve, Mapping):
        _hash_update_str(h, repr(sorted((str(k), repr(v)) for k, v in curve.items())))
        return
    _hash_update_str(h, str(getattr(curve, "identifier", "")))
    _hash_update_str(h, str(getattr(curve, "interpolation_method", "")))
    _hash_update_str(h, str(getattr(curve, "extrapolation_method", "")))
    _hash_update_str(h, str(getattr(curve, "dcc", "")))
    _hash_update_str(h, str(getattr(curve, "as_of_date", "")))
    _hash_update_str(h, str(getattr(curve, "_input_mode", "")))
    for attr in (
        "tenors",
        "rates",
        "spreads",
        "_tenors",
        "_y_pillars",
        "space_coords",
        "_raw_quotes",
    ):
        v = getattr(curve, attr, None)
        if v is not None:
            if isinstance(v, (str, bytes)):
                _hash_update_str(h, repr(v))
            else:
                try:
                    _hash_update_arr(h, np.asarray(v, dtype=np.float64))
                except (TypeError, ValueError):
                    _hash_update_str(h, repr(v))
    inner = getattr(curve, "_inner_curve", None)
    if inner is not None and inner is not curve:
        _hash_curve_obj(h, inner)


def _build_bb_orthogonal_matrix(tau_obs_np: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Constructs exact orthogonal (M, M) temporal Brownian-bridge increment transform."""
    m_rem = int(len(tau_obs_np))
    if m_rem <= 1:
        return np.ones((max(m_rem, 1), max(m_rem, 1)), dtype=np.float64)
    dt_arr = np.maximum(
        np.diff(np.concatenate([[0.0], tau_obs_np.astype(np.float64)])),
        1e-8,
    )
    t_cum = np.concatenate([[0.0], np.cumsum(dt_arr)])
    c_path = np.zeros((m_rem + 1, m_rem), dtype=np.float64)
    c_path[m_rem, 0] = math.sqrt(float(t_cum[m_rem]))
    queue: deque[Tuple[int, int]] = deque([(0, m_rem)])
    col_idx = 1
    while queue:
        l_idx, r_idx = queue.popleft()
        if r_idx - l_idx <= 1:
            continue
        mid = (l_idx + r_idx) // 2
        tl, tm, tr = float(t_cum[l_idx]), float(t_cum[mid]), float(t_cum[r_idx])
        denom = max(tr - tl, 1e-12)
        wl = (tr - tm) / denom
        wr = (tm - tl) / denom
        c_path[mid, :] = wl * c_path[l_idx, :] + wr * c_path[r_idx, :]
        c_path[mid, col_idx] = math.sqrt(max((tm - tl) * (tr - tm) / denom, 0.0))
        col_idx += 1
        queue.append((l_idx, mid))
        queue.append((mid, r_idx))
    inv_sqrt_dt = 1.0 / np.sqrt(dt_arr)[:, None]
    return np.ascontiguousarray((c_path[1:, :] - c_path[:-1, :]) * inv_sqrt_dt, dtype=np.float64)


def _get_sobol_direction_numbers(device: torch.device) -> torch.Tensor:
    """Returns cached (21201, 30) Joe-Kuo Sobol direction numbers on `device`."""
    global _SOBOL_HOST_SV
    cache_key = ("sobol_sv", normalize_cuda_device(device))
    hit = _STATIC_CONSTANTS_LRU.get(cache_key)
    if hit is not None:
        return hit  # type: ignore[no-any-return]
    with _SOBOL_HOST_LOCK:
        if _SOBOL_HOST_SV is None:
            sob = qmc.Sobol(d=_SOBOL_MAX_DIM, scramble=False)
            _SOBOL_HOST_SV = np.ascontiguousarray(sob._sv.astype(np.int32))
        t_gpu = torch.from_numpy(_SOBOL_HOST_SV).to(device=device, non_blocking=False)
    return _STATIC_CONSTANTS_LRU.put(cache_key, t_gpu)  # type: ignore[no-any-return]


def _splitmix64_30bit_shifts(keys_u64: npt.NDArray[np.uint64], n_rep: int) -> npt.NDArray[np.int32]:
    """Deterministic vectorized 30-bit Sobol digital shifts keyed strictly by `(global_dim, replicate)`."""
    r_idx = np.arange(n_rep, dtype=np.uint64)[None, :]
    z = (keys_u64[:, None] * np.uint64(0x9E3779B97F4A7C15)) ^ (
        (r_idx + np.uint64(1)) * np.uint64(0xBF58476D1CE4E5B9)
    )
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.asarray((z >> np.uint64(34)) & np.uint64((1 << _SOBOL_BITS) - 1), dtype=np.int32)


def _get_sobol_period_shifts(
    *,
    n_replicates: int,
    d_assets: int,
    m_rem: int,
    n_sub: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns cached compact int32 Sobol scramble shifts `(M, R, D)` and `(M, R, D*n_sub)`."""
    cache_key = (
        "sobol_period_shifts",
        int(n_replicates),
        int(d_assets),
        int(m_rem),
        int(n_sub),
        int(seed),
        normalize_cuda_device(device),
    )
    hit = _STATIC_CONSTANTS_LRU.get(cache_key)
    if hit is not None:
        return hit  # type: ignore[no-any-return]

    shifts_c_cpu = torch.empty((m_rem, n_replicates, d_assets), dtype=torch.int32, device="cpu")
    is_haar = n_sub > 1 and ((n_sub & (n_sub - 1)) == 0)
    shifts_s_cpu = (
        torch.zeros((m_rem, n_replicates, d_assets * n_sub), dtype=torch.int32, device="cpu")
        if n_sub > 1
        else torch.empty((m_rem, n_replicates, 0), dtype=torch.int32, device="cpu")
    )
    gen_s = torch.Generator(device="cpu")
    sub_dim_base = m_rem * d_assets
    seed_base_s = int(seed) + 31337
    for m in range(m_rem):
        gen_s.manual_seed(int(seed) + (m * d_assets) * 131 + 17)
        torch.randint(0, 1 << _SOBOL_BITS, (n_replicates, d_assets), out=shifts_c_cpu[m], generator=gen_s)
        if is_haar:
            # Key each Haar bridge mode strictly by its invariant global dimension so n_sub=2,4,8,16 nest.
            g_dims = _haar_global_dims_for_period(m_rem, d_assets, n_sub, m)
            keys = np.uint64(seed_base_s) + g_dims.astype(np.uint64) * np.uint64(131) + np.uint64(17)
            sh_modes = _splitmix64_30bit_shifts(keys, n_replicates).reshape(d_assets, n_sub - 1, n_replicates)
            sh_t = torch.from_numpy(np.ascontiguousarray(sh_modes.transpose(2, 0, 1)))
            shifts_s_cpu[m].view(n_replicates, d_assets, n_sub)[:, :, : n_sub - 1].copy_(sh_t)
        elif n_sub > 1:
            gen_s.manual_seed(seed_base_s + (sub_dim_base + m * d_assets * n_sub) * 131 + 17)
            torch.randint(
                0,
                1 << _SOBOL_BITS,
                (n_replicates, d_assets * n_sub),
                out=shifts_s_cpu[m],
                generator=gen_s,
            )

    shifts_c_gpu = shifts_c_cpu.to(device=device, non_blocking=False)
    shifts_s_gpu = shifts_s_cpu.to(device=device, non_blocking=False)
    return _STATIC_CONSTANTS_LRU.put(cache_key, (shifts_c_gpu, shifts_s_gpu))  # type: ignore[no-any-return]


def _get_surface_gpu_tensors(surf: Any, device: torch.device) -> Dict[str, Any]:
    """External thread-safe LRU cache of GPU spline tensors for a `RegularizedVolSurface`."""
    key = ("surf_gpu", _surface_content_digest(surf), normalize_cuda_device(device))
    hit = _STATIC_CONSTANTS_LRU.get(key)
    if hit is not None:
        return hit  # type: ignore[no-any-return]

    slice_c_np = getattr(surf, "_fast_slice_coeffs", None)
    if slice_c_np is None:
        raise RuntimeError("RegularizedVolSurface missing _fast_slice_coeffs.")
    strikes_np = np.asarray(surf.strikes, dtype=np.float64).ravel()
    tenors_np = np.asarray(surf.tenors, dtype=np.float64).ravel()
    tenors_365_np = np.asarray(getattr(surf, "tenors_act365", surf.tenors), dtype=np.float64).ravel()
    quotes_np = np.asarray(surf.quotes, dtype=np.float64)
    slice_c_arr = np.asarray(slice_c_np, dtype=np.float64)
    slice_c_tr_np = np.ascontiguousarray(np.transpose(slice_c_arr, (0, 2, 1)))
    sl_left_np = np.asarray(surf._slopes_left, dtype=np.float64).ravel()
    sl_right_np = np.asarray(surf._slopes_right, dtype=np.float64).ravel()

    q_shape = quotes_np.shape
    sc_shape = slice_c_arr.shape
    sctr_shape = slice_c_tr_np.shape
    packed_np = np.concatenate([
        strikes_np,
        tenors_np,
        tenors_365_np,
        quotes_np.ravel(),
        slice_c_arr.ravel(),
        slice_c_tr_np.ravel(),
        sl_left_np,
        sl_right_np,
    ])
    packed_gpu = torch.as_tensor(packed_np, dtype=torch.float64, device=device)
    off = 0
    n_k = strikes_np.size
    strikes_t = packed_gpu[off : off + n_k]
    off += n_k
    n_e = tenors_np.size
    tenors_t = packed_gpu[off : off + n_e]
    off += n_e
    n_e365 = tenors_365_np.size
    tenors_365_t = packed_gpu[off : off + n_e365]
    off += n_e365
    n_q = quotes_np.size
    quotes_t = packed_gpu[off : off + n_q].view(q_shape)
    off += n_q
    n_sc = slice_c_arr.size
    slice_c_t = packed_gpu[off : off + n_sc].view(sc_shape)
    off += n_sc
    n_sctr = slice_c_tr_np.size
    slice_c_tr_t = packed_gpu[off : off + n_sctr].view(sctr_shape)
    off += n_sctr
    n_sl = sl_left_np.size
    slopes_left_t = packed_gpu[off : off + n_sl]
    off += n_sl
    slopes_right_t = packed_gpu[off : off + sl_right_np.size]

    entry = {
        "strikes": strikes_t,
        "tenors": tenors_t,
        "tenors_365": tenors_365_t,
        "quotes": quotes_t,
        "slice_c": slice_c_t,
        "slice_c_tr": slice_c_tr_t,
        "slopes_left": slopes_left_t,
        "slopes_right": slopes_right_t,
    }
    return _STATIC_CONSTANTS_LRU.put(key, entry)  # type: ignore[no-any-return]


def _ensure_bundle_substeps(
    bundle: _GpuQmcScheduleBundle,
    n_sub: int,
    device: torch.device,
) -> None:
    """Lazily populates substep tensors on `bundle` from its compact device boundary shifts."""
    with _SCHED_SUBSTEP_LOCK:
        if n_sub in bundle.substep_tensors:
            return
        dt_per_np = np.asarray(bundle.dt_arr_tuple, dtype=np.float64)
        dt_sub_np = (dt_per_np / float(n_sub)).astype(np.float32)
        sqrt_dt_np = np.sqrt(dt_sub_np).astype(np.float32)
        alphas_gpu = (
            (torch.arange(n_sub, dtype=torch.float32, device=device) + 0.5) / float(n_sub)
        ).view(1, 1, n_sub)
        q0 = bundle.q_shift_bnd_f32[:, :, 0:1]
        q1 = bundle.q_shift_bnd_f32[:, :, 1:2]
        q_shift_gpu = (q0 + alphas_gpu * (q1 - q0)).contiguous()
        dt_sub_gpu = torch.as_tensor(dt_sub_np, dtype=torch.float32, device=device)
        sqrt_dt_gpu = torch.as_tensor(sqrt_dt_np, dtype=torch.float32, device=device)
        cpu_meta = (
            tuple(float(x) for x in dt_sub_np.tolist()),
            tuple(float(x) for x in sqrt_dt_np.tolist()),
        )
        bundle.substep_tensors[n_sub] = (dt_sub_gpu, sqrt_dt_gpu, q_shift_gpu)
        bundle.substep_cpu_meta[n_sub] = cpu_meta


def _get_or_build_static_schedule(
    state: AutocallPreprocState,
    device: torch.device,
    *,
    n_sub: int,
    lv_grid_nt: int,
    lv_grid_ny: int,
    use_triton: bool,
) -> _GpuQmcScheduleBundle:
    """Lazily builds and caches immutable GPU schedule tensors for `state` without `.cpu()` stalls."""
    dev_norm = normalize_cuda_device(device)
    spots_np = _cpu_f64_array(state.spots_t0)
    ref_np = _cpu_f64_array(state.ref_spots)
    if not (np.all(np.isfinite(spots_np)) and np.all(spots_np > 0.0)):
        raise ValueError("Spots must be strictly positive and finite.")
    if not (np.all(np.isfinite(ref_np)) and np.all(ref_np > 0.0)):
        raise ValueError("UnderlyingReferenceSpots must be strictly positive and finite.")

    d_assets = int(state.d_assets)
    m_rem = int(state.m_rem)
    corr_np = _cpu_f64_array(state.corr_matrix)
    if not np.all(np.isfinite(corr_np)):
        raise ValueError("Correlation matrix contains non-finite values.")
    tau_obs_np = _cpu_f64_array(state.tau_obs)
    df_pay_np = _cpu_f64_array(state.df_pay)
    ko_barriers_np = _cpu_f64_array(state.ko_barriers)
    is_callable_np = _cpu_f64_array(state.is_callable)
    cpn_barriers_np = _cpu_f64_array(state.cpn_barriers)
    is_guar_np = _cpu_f64_array(state.is_guaranteed_cpn)
    coupon_amounts_np = _cpu_f64_array(state.coupon_amounts)
    fwd_prices_np = _cpu_f64_array(state.dividend_schedule.forward_price)
    iv_at_fwd_np = _cpu_f64_array(state.iv_at_fwd)
    df_fin_obs_np = _cpu_f64_array(state.dividend_schedule.df_fin_obs)
    r_tot_obs_np = _cpu_f64_array(state.dividend_schedule.r_tot_obs)
    r_nat_obs_np = _cpu_f64_array(_resolve_r_native_obs_tensor(state))
    v_perf_np = _cpu_f64_array(state.v_perf)
    v_term_np = np.ascontiguousarray(v_perf_np[:, -1], dtype=np.float64)

    surfs_ordered = [state.vol_surfaces[t] for t in state.sorted_identifiers]
    div_events_map = getattr(state.dividend_schedule, "dividend_events", {})
    div_events_ordered = [div_events_map.get(t, []) for t in state.sorted_identifiers]
    inv_perm_raw = getattr(state, "inv_lex_perm", None)
    inv_perm_np = np.ascontiguousarray(inv_perm_raw, dtype=np.int64) if inv_perm_raw is not None else None
    future_obs_dates = getattr(state.seasoned_state, "future_obs_dates", ())

    dom_idx_opt: Optional[int] = None
    try:
        dw_np = _cpu_f64_array(state.baseline.proxy_delta_weights)
        dom_idx_opt = int(np.argmax(np.abs(dw_np))) if np.max(np.abs(dw_np)) > 0 else 0
    except (AttributeError, TypeError, ValueError):
        dom_idx_opt = None

    h = hashlib.blake2b(digest_size=24)
    for arr in (
        spots_np,
        ref_np,
        corr_np,
        tau_obs_np,
        df_pay_np,
        ko_barriers_np,
        is_callable_np,
        cpn_barriers_np,
        is_guar_np,
        coupon_amounts_np,
        fwd_prices_np,
        iv_at_fwd_np,
        df_fin_obs_np,
        r_tot_obs_np,
        r_nat_obs_np,
        v_term_np,
    ):
        _hash_update_arr(h, arr)
    if inv_perm_np is not None:
        _hash_update_arr(h, inv_perm_np)
    lk_val = float(state.lower_strike) if state.lower_strike is not None else -1.0
    _hash_update_str(
        h,
        repr(
            (
                d_assets,
                m_rem,
                dom_idx_opt,
                float(state.strike),
                lk_val,
                int(lv_grid_nt),
                int(lv_grid_ny),
                bool(use_triton),
                tuple(str(d) for d in future_obs_dates),
            )
        ),
    )
    lv_digest = _compute_lv_batch_digest(
        surfs_ordered,
        spots_np,
        r_nat_obs_np,
        df_fin_obs_np,
        tau_obs_np,
        div_events_ordered,
        v_term_np,
        lv_grid_nt,
        lv_grid_ny,
        device,
        use_triton,
        fwd_prices_np=fwd_prices_np,
    )
    _hash_update_str(h, lv_digest)
    f_leg = getattr(state, "funding_leg", None)
    if f_leg is not None:
        f_fix_raw = f_leg.get("funding_fixings") or ()
        f_fix_tup = (
            tuple(sorted((str(dt_r), float(val_r)) for dt_r, val_r in f_fix_raw.items()))
            if isinstance(f_fix_raw, Mapping)
            else tuple((str(dt_r), float(val_r)) for dt_r, val_r in f_fix_raw)
        )
        hist_recs = getattr(state.seasoned_state, "historical_records", ()) or ()
        max_past_str = max((str(getattr(r, "payment_date", "")) for r in hist_recs), default="")
        _hash_update_str(
            h,
            repr(
                (
                    f_leg.get("rate_identifier"),
                    f_leg.get("funding_curve_id"),
                    f_leg.get("day_count", f_leg.get("day_count_convention")),
                    0.0 if f_leg.get("funding_spread") is None else float(f_leg.get("funding_spread", 0.0)),
                    tuple(str(d) for d in (f_leg.get("payment_dates") or ())),
                    f_fix_tup,
                    str(getattr(state.seasoned_state, "as_of_date", "")),
                    str(getattr(state.seasoned_state, "start_date", "")),
                    str(getattr(state.seasoned_state, "day_count_convention", "")),
                    tuple(str(d) for d in getattr(state.seasoned_state, "future_pay_dates", ())),
                    max_past_str,
                )
            ),
        )
        _hash_curve_obj(h, getattr(state, "discount_curve", None))
        _hash_curve_obj(h, f_leg.get("funding_curve"))

    digest_hex = h.hexdigest()
    content_key = ("static_sched", digest_hex, dev_norm)
    cached = _DEAL_PREP_LRU.get(content_key)
    if cached is not None:
        bundle_hit: _GpuQmcScheduleBundle = cached
        _ensure_bundle_substeps(bundle_hit, n_sub, device)
        return bundle_hit

    # Precompute surface DCC cumulative variances so terminal_iv and MF_CV honor Act360/Bus252.
    var_cum_gbm_np = np.empty((d_assets, m_rem), dtype=np.float64)
    max_rel_skew = 0.0
    for i, surf_i in enumerate(surfs_ordered):
        dcc_i = str(getattr(surf_i, "dcc", "Act365")).strip().lower()
        non_365 = dcc_i not in ("act365", "act365f", "act_365", "act_365f")
        if hasattr(surf_i, "strikes") and hasattr(surf_i, "quotes") and len(surf_i.strikes) >= 2:
            q_s = np.asarray(surf_i.quotes, dtype=np.float64)
            k_s = np.asarray(surf_i.strikes, dtype=np.float64)
            sp_i = max(float(spots_np[i]), 1e-12)
            log_k = np.log(np.maximum(k_s, 1e-12) / sp_i)
            atm_mid = max(float(np.median(q_s)), 1e-3)
            log_span = max(float(log_k[-1] - log_k[0]), 0.10)
            wing_rel = float(np.max(q_s.max(axis=1) - q_s.min(axis=1)) / (log_span * atm_mid))
            # Capture steep skew/curvature inside [0.4*S, 2.0*S] where autocall barriers reside.
            dlog_k = np.maximum(np.diff(log_k), 1e-3)
            loc_slopes = np.abs(np.diff(q_s, axis=1)) / (dlog_k[None, :] * atm_mid)
            rel_s = max(wing_rel, float(np.max(loc_slopes)))
            if rel_s > max_rel_skew:
                max_rel_skew = rel_s
        if non_365:
            if getattr(surf_i, "as_of_date", None) is not None and len(future_obs_dates) == m_rem:
                t_surf_vec = [surf_i.tenor_from_date(dt) for dt in future_obs_dates]
            else:
                t_surf_vec = [surf_i.map_act365_to_surface_tenor(float(tm)) for tm in tau_obs_np]
            for m in range(m_rem):
                iv_m = float(surf_i.iv(float(t_surf_vec[m]), float(fwd_prices_np[i, m])))
                var_cum_gbm_np[i, m] = (iv_m * iv_m) * float(t_surf_vec[m])
        else:
            var_cum_gbm_np[i, :] = (iv_at_fwd_np[i, :] ** 2) * tau_obs_np

    d_var_gbm_np = np.empty((d_assets, m_rem), dtype=np.float64)
    var_run = np.zeros(d_assets, dtype=np.float64)
    for m in range(m_rem):
        dv_col = np.maximum(var_cum_gbm_np[:, m] - var_run, 1e-12)
        d_var_gbm_np[:, m] = dv_col
        var_run = var_run + dv_col

    dom_idx = dom_idx_opt if dom_idx_opt is not None else int(np.argmax(var_cum_gbm_np[:, -1]))

    corr_sym = 0.5 * (corr_np + corr_np.T)
    eigvals, eigvecs = np.linalg.eigh(corr_sym)
    if np.min(eigvals) < 1e-8:
        eigvals = np.maximum(eigvals, 1e-6)
        corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d_inv = 1.0 / np.sqrt(np.diag(corr_psd))
        corr_sym = corr_psd * np.outer(d_inv, d_inv)
        eigvals, eigvecs = np.linalg.eigh(corr_sym)
    chol_np = np.linalg.cholesky(corr_sym).astype(np.float32)

    order_desc = np.argsort(eigvals)[::-1]
    ev_desc = np.maximum(eigvals[order_desc], 0.0)
    a_pca_np = np.ascontiguousarray(
        eigvecs[:, order_desc] * np.sqrt(ev_desc)[None, :], dtype=np.float32
    )
    m_bb_np = _build_bb_orthogonal_matrix(tau_obs_np).astype(np.float32)

    w_fund_np = (
        _compute_funding_weights_np(state)
        if f_leg is not None
        else np.zeros(m_rem, dtype=np.float64)
    )
    fwd_perf_np = np.ascontiguousarray(fwd_prices_np / ref_np[:, None], dtype=np.float64)
    tau_max = max(float(tau_obs_np[-1]), 1e-4)
    lv_t_grid_np = np.linspace(0.0, tau_max, lv_grid_nt, dtype=np.float64)

    t_prev_np = np.concatenate([[0.0], tau_obs_np[:-1]]).astype(np.float32)
    tau_obs_32_np = tau_obs_np.astype(np.float32)
    v_dom_term = math.sqrt(max(float(var_cum_gbm_np[dom_idx, -1]), 1e-12))
    tau_term = max(float(tau_obs_np[-1]), 1e-8)
    dt_arr = np.maximum(tau_obs_np - np.concatenate([[0.0], tau_obs_np[:-1]]), 1e-8)
    sqrt_dvar_dom_np = np.sqrt(d_var_gbm_np[dom_idx, :])

    # Single packed float64 H2D upload for all schedule tensors
    f64_parts = [
        lv_t_grid_np.ravel(),
        r_nat_obs_np.ravel(),
        r_tot_obs_np.ravel(),
        df_fin_obs_np.ravel(),
        df_pay_np.ravel(),
        fwd_perf_np.ravel(),
        ref_np.ravel(),
        ko_barriers_np.ravel(),
        cpn_barriers_np.ravel(),
        is_callable_np.ravel(),
        is_guar_np.ravel(),
        coupon_amounts_np.ravel(),
        w_fund_np.ravel(),
        spots_np.ravel(),
        v_term_np.ravel(),
        sqrt_dvar_dom_np.ravel(),
        tau_obs_np.ravel(),
    ]
    packed_64_gpu = torch.as_tensor(np.ascontiguousarray(np.concatenate(f64_parts)), dtype=torch.float64, device=device)
    o64 = 0
    lv_t_grid_t = packed_64_gpu[o64 : o64 + lv_grid_nt]
    o64 += lv_grid_nt
    r_nat_t = packed_64_gpu[o64 : o64 + d_assets * m_rem].view(d_assets, m_rem)
    o64 += d_assets * m_rem
    r_tot_t = packed_64_gpu[o64 : o64 + d_assets * m_rem].view(d_assets, m_rem)
    o64 += d_assets * m_rem
    df_fin_t = packed_64_gpu[o64 : o64 + d_assets * m_rem].view(d_assets, m_rem)
    o64 += d_assets * m_rem
    df_pay_t = packed_64_gpu[o64 : o64 + m_rem]
    o64 += m_rem
    fwd_perf_t = packed_64_gpu[o64 : o64 + d_assets * m_rem].view(d_assets, m_rem)
    o64 += d_assets * m_rem
    ref_spots_t = packed_64_gpu[o64 : o64 + d_assets]
    o64 += d_assets
    ko_t = packed_64_gpu[o64 : o64 + m_rem]
    o64 += m_rem
    cpn_t = packed_64_gpu[o64 : o64 + m_rem]
    o64 += m_rem
    is_call_t = packed_64_gpu[o64 : o64 + m_rem]
    o64 += m_rem
    is_guar_t = packed_64_gpu[o64 : o64 + m_rem]
    o64 += m_rem
    cpn_amt_t = packed_64_gpu[o64 : o64 + m_rem]
    o64 += m_rem
    w_fund_t = packed_64_gpu[o64 : o64 + m_rem]
    o64 += m_rem
    spots_t = packed_64_gpu[o64 : o64 + d_assets]
    o64 += d_assets
    v_term_t = packed_64_gpu[o64 : o64 + d_assets]
    o64 += d_assets
    sqrt_dvar_dom_t = packed_64_gpu[o64 : o64 + m_rem]
    o64 += m_rem
    tau_obs_64_t = packed_64_gpu[o64 : o64 + m_rem]

    dt_sub_np = (dt_arr / float(n_sub)).astype(np.float32)
    sqrt_dt_np = np.sqrt(dt_sub_np).astype(np.float32)
    r_tot_prev = np.concatenate([np.zeros((d_assets, 1), dtype=np.float64), r_tot_obs_np[:, :-1]], axis=1)
    r_nat_prev = np.concatenate([np.zeros((d_assets, 1), dtype=np.float64), r_nat_obs_np[:, :-1]], axis=1)
    q_bnd_start = (r_tot_prev - r_nat_prev).T[:, :, None]  # (M, D, 1)
    q_bnd_end = (r_tot_obs_np - r_nat_obs_np).T[:, :, None]  # (M, D, 1)
    q_shift_bnd_np = np.concatenate([q_bnd_start, q_bnd_end], axis=2).astype(np.float32)  # (M, D, 2)
    alphas = ((np.arange(n_sub, dtype=np.float64) + 0.5) / float(n_sub)).reshape(1, 1, n_sub)
    q_shift_np = (
        (r_tot_prev[:, :, None] + alphas * (r_tot_obs_np - r_tot_prev)[:, :, None])
        - (r_nat_prev[:, :, None] + alphas * (r_nat_obs_np - r_nat_prev)[:, :, None])
    ).transpose(1, 0, 2).astype(np.float32)

    packed_32 = np.concatenate([
        chol_np.ravel(),
        a_pca_np.ravel(),
        m_bb_np.ravel(),
        q_shift_bnd_np.ravel(),
        t_prev_np.ravel(),
        tau_obs_32_np.ravel(),
        dt_sub_np.ravel(),
        sqrt_dt_np.ravel(),
        q_shift_np.ravel(),
    ])
    packed_32_gpu = torch.as_tensor(np.ascontiguousarray(packed_32), dtype=torch.float32, device=device)
    o32 = 0
    chol_f32 = packed_32_gpu[o32 : o32 + d_assets * d_assets].view(d_assets, d_assets)
    o32 += d_assets * d_assets
    a_pca_f32 = packed_32_gpu[o32 : o32 + d_assets * d_assets].view(d_assets, d_assets)
    o32 += d_assets * d_assets
    m_bb_f32 = packed_32_gpu[o32 : o32 + m_rem * m_rem].view(m_rem, m_rem)
    o32 += m_rem * m_rem
    q_shift_bnd_f32 = packed_32_gpu[o32 : o32 + m_rem * d_assets * 2].view(m_rem, d_assets, 2)
    o32 += m_rem * d_assets * 2
    t_prev_f32 = packed_32_gpu[o32 : o32 + m_rem]
    o32 += m_rem
    tau_obs_f32 = packed_32_gpu[o32 : o32 + m_rem]
    o32 += m_rem
    dt_sub_f32 = packed_32_gpu[o32 : o32 + m_rem]
    o32 += m_rem
    sqrt_dt_f32 = packed_32_gpu[o32 : o32 + m_rem]
    o32 += m_rem
    q_shift_f32 = packed_32_gpu[o32:].view(m_rem, d_assets, n_sub)

    inv_perm_gpu = (
        torch.as_tensor(inv_perm_np, dtype=torch.long, device=device)
        if inv_perm_np is not None
        else None
    )

    f_dom = float(fwd_perf_np[dom_idx, -1])
    k_cv = float(state.strike) if state.strike > 1e-6 else 1.0
    k_low = (
        float(state.lower_strike)
        if (state.lower_strike is not None and float(state.lower_strike) > 0.0 and state.strike > 1e-6)
        else -1.0
    )
    df_term = float(df_pay_np[-1])
    mu_cv = df_term * _black76_put_np(f_dom, k_cv, v_dom_term)
    if k_low > 0.0:
        mu_cv -= df_term * _black76_put_np(f_dom, k_low, v_dom_term)

    sched = GpuQmcStaticSchedule(
        digest_hex=digest_hex,
        lv_t_grid=lv_t_grid_t,
        r_nat_obs=r_nat_t,
        r_tot_obs=r_tot_t,
        df_fin_obs=df_fin_t,
        df_pay=df_pay_t,
        fwd_perf=fwd_perf_t,
        ref_spots=ref_spots_t,
        ko_barriers=ko_t,
        cpn_barriers=cpn_t,
        is_callable=is_call_t,
        is_guar=is_guar_t,
        coupon_amounts=cpn_amt_t,
        w_fund=w_fund_t,
    )
    bundle = _GpuQmcScheduleBundle(
        schedule=sched,
        lv_digest=lv_digest,
        spots_t0=spots_t,
        v_term=v_term_t,
        chol_f32=chol_f32,
        a_pca_f32=a_pca_f32,
        m_bb_f32=m_bb_f32,
        q_shift_bnd_f32=q_shift_bnd_f32,
        t_prev_f32=t_prev_f32,
        tau_obs_f32=tau_obs_f32,
        inv_lex_perm_gpu=inv_perm_gpu,
        sqrt_dvar_dom_f64=sqrt_dvar_dom_t,
        dom_idx=dom_idx,
        v_dom_term=v_dom_term,
        tau_term=tau_term,
        f_dom=f_dom,
        k_cv=k_cv,
        k_low=k_low,
        df_term=df_term,
        mu_cv=mu_cv,
        substep_tensors={n_sub: (dt_sub_f32, sqrt_dt_f32, q_shift_f32)},
        tau_obs_np=tau_obs_np,
        df_pay_np=df_pay_np,
        var_cum_gbm_np=var_cum_gbm_np,
        d_var_gbm_np=d_var_gbm_np,
        max_rel_skew=max_rel_skew,
        max_dt=float(np.max(dt_arr)),
        t_prev_tuple=tuple(float(x) for x in t_prev_np.tolist()),
        dt_arr_tuple=tuple(float(x) for x in dt_arr.tolist()),
        sqrt_dvar_dom_tuple=tuple(float(x) for x in sqrt_dvar_dom_np.tolist()),
        substep_cpu_meta={
            n_sub: (
                tuple(float(x) for x in dt_sub_np.tolist()),
                tuple(float(x) for x in sqrt_dt_np.tolist()),
            )
        },
    )
    object.__setattr__(bundle, "_tau_obs_64", tau_obs_64_t)
    _DEAL_PREP_LRU.put(content_key, bundle)
    return bundle


if _HAS_TRITON:

    @triton.jit  # type: ignore[untyped-decorator]
    def _sobol_normals_triton_kernel(  # type: ignore[no-untyped-def]
        sv_ptr,
        shifts_ptr,
        out_ptr,
        dim_offset,
        n_dims,
        n_rep,
        m_pow,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_r = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep

        idx = offs_n.to(tl.uint32)
        gray = idx ^ (idx >> 1)

        dim_idx = dim_offset + pid_d
        acc = tl.zeros((BLOCK_N,), dtype=tl.uint32)
        z_vec = tl.zeros((BLOCK_N,), dtype=tl.uint32)
        for b in range(m_pow):
            bit_on = (gray >> b) & 1
            dir_num = tl.load(sv_ptr + dim_idx * 30 + b).to(tl.uint32)
            acc = acc ^ tl.where(bit_on != 0, dir_num, z_vec)

        shift = tl.load(shifts_ptr + pid_r * n_dims + pid_d).to(tl.uint32)
        acc = acc ^ shift
        u = (acc.to(tl.float32) + 0.5) * (1.0 / 1073741824.0)
        u = tl.minimum(tl.maximum(u, 1e-6), 1.0 - 1e-6)
        z = tl.extra.cuda.libdevice.normcdfinv(u)

        out_off = (pid_r * n_dims + pid_d) * n_rep + offs_n
        tl.store(out_ptr + out_off, z, mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _dupire_substep_triton_kernel(  # type: ignore[no-untyped-def]
        log_x_ptr,          # (R, D, N_rep) float32
        z_corr_ptr,         # (R, D, n_sub, N_rep) float32
        lv_table_ptr,       # (D, N_t, N_y) float32
        y_min_ptr,          # (D,) float32
        inv_dy_ptr,         # (D,) float32
        q_shift_bnd_m_ptr,  # (D, 2) float32
        t_start,            # float32
        dt_sub,             # float32
        sqrt_dt_sub,        # float32
        inv_dt_grid,        # float32 = (N_t - 1) / tau_max
        d_assets,           # int32
        n_rep: tl.constexpr,
        n_sub: tl.constexpr,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_r = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep

        lx_off = (pid_r * d_assets + pid_d) * n_rep + offs_n
        lx = tl.load(log_x_ptr + lx_off, mask=mask_n, other=0.0)

        y_min = tl.load(y_min_ptr + pid_d)
        inv_dy = tl.load(inv_dy_ptr + pid_d)
        q0 = tl.load(q_shift_bnd_m_ptr + pid_d * 2 + 0)
        q1 = tl.load(q_shift_bnd_m_ptr + pid_d * 2 + 1)
        lv_base = pid_d * (N_t * N_y)
        z_base = (pid_r * d_assets + pid_d) * (n_sub * n_rep)

        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4
        inv_n_sub = 1.0 / float(n_sub)

        for s in range(n_sub):
            s_f = tl.zeros((), dtype=tl.float32) + s
            t_s0 = t_start + s_f * dt_sub
            t_s1 = t_start + (s_f + 1.0) * dt_sub
            q_sh0 = q0 + (s_f * inv_n_sub) * (q1 - q0)
            q_sh1 = q0 + ((s_f + 1.0) * inv_n_sub) * (q1 - q0)

            ut0 = tl.minimum(tl.maximum(t_s0 * inv_dt_grid, 0.0), max_t_f)
            it0_0 = ut0.to(tl.int32)
            ft0 = ut0 - it0_0.to(tl.float32)
            it1_0 = tl.minimum(it0_0 + 1, N_t - 1)
            uy0 = tl.minimum(tl.maximum((lx + q_sh0 - y_min) * inv_dy, 0.0), max_y_f)
            iy0_0 = uy0.to(tl.int32)
            fy0 = uy0 - iy0_0.to(tl.float32)
            iy1_0 = tl.minimum(iy0_0 + 1, N_y - 1)
            r00 = lv_base + it0_0 * N_y
            r10 = lv_base + it1_0 * N_y
            a0 = tl.load(lv_table_ptr + r00 + iy0_0, mask=mask_n, other=0.2)
            a1 = tl.load(lv_table_ptr + r00 + iy1_0, mask=mask_n, other=0.2)
            b0 = tl.load(lv_table_ptr + r10 + iy0_0, mask=mask_n, other=0.2)
            b1 = tl.load(lv_table_ptr + r10 + iy1_0, mask=mask_n, other=0.2)
            lv0 = (1.0 - ft0) * (a0 + fy0 * (a1 - a0)) + ft0 * (b0 + fy0 * (b1 - b0))

            mu0 = -0.5 * lv0 * lv0
            y_h = lx + mu0 * dt_sub
            sig_step = lv0 * sqrt_dt_sub

            ut1 = tl.minimum(tl.maximum(t_s1 * inv_dt_grid, 0.0), max_t_f)
            it0_1 = ut1.to(tl.int32)
            ft1 = ut1 - it0_1.to(tl.float32)
            it1_1 = tl.minimum(it0_1 + 1, N_t - 1)
            r01 = lv_base + it0_1 * N_y
            r11 = lv_base + it1_1 * N_y

            uyp = tl.minimum(tl.maximum((y_h + sig_step + q_sh1 - y_min) * inv_dy, 0.0), max_y_f)
            iyp0 = uyp.to(tl.int32)
            fyp = uyp - iyp0.to(tl.float32)
            iyp1 = tl.minimum(iyp0 + 1, N_y - 1)
            p00 = tl.load(lv_table_ptr + r01 + iyp0, mask=mask_n, other=0.2)
            p01 = tl.load(lv_table_ptr + r01 + iyp1, mask=mask_n, other=0.2)
            p10 = tl.load(lv_table_ptr + r11 + iyp0, mask=mask_n, other=0.2)
            p11 = tl.load(lv_table_ptr + r11 + iyp1, mask=mask_n, other=0.2)
            lv_p = (1.0 - ft1) * (p00 + fyp * (p01 - p00)) + ft1 * (p10 + fyp * (p11 - p10))

            uym = tl.minimum(tl.maximum((y_h - sig_step + q_sh1 - y_min) * inv_dy, 0.0), max_y_f)
            iym0 = uym.to(tl.int32)
            fym = uym - iym0.to(tl.float32)
            iym1 = tl.minimum(iym0 + 1, N_y - 1)
            m00 = tl.load(lv_table_ptr + r01 + iym0, mask=mask_n, other=0.2)
            m01 = tl.load(lv_table_ptr + r01 + iym1, mask=mask_n, other=0.2)
            m10 = tl.load(lv_table_ptr + r11 + iym0, mask=mask_n, other=0.2)
            m11 = tl.load(lv_table_ptr + r11 + iym1, mask=mask_n, other=0.2)
            lv_m = (1.0 - ft1) * (m00 + fym * (m01 - m00)) + ft1 * (m10 + fym * (m11 - m10))

            mu_p = -0.5 * lv_p * lv_p
            mu_m = -0.5 * lv_m * lv_m
            z = tl.load(z_corr_ptr + z_base + s * n_rep + offs_n, mask=mask_n, other=0.0)
            lx = lx + (
                0.5 * (mu0 + 0.5 * (mu_p + mu_m)) * dt_sub
                + 0.25 * (mu_p - mu_m) * dt_sub * z
                + 0.25 * (lv_p + lv_m + 2.0 * lv0) * sqrt_dt_sub * z
                + 0.25 * (lv_p - lv_m) * (z * z - 1.0) * sqrt_dt_sub
            )

        tl.store(log_x_ptr + lx_off, lx, mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _dupire_horizon_triton_kernel(  # type: ignore[no-untyped-def]
        z_steps_ptr,      # (R, D, M, n_sub, N_rep) float32
        lv_table_ptr,     # (D, N_t, N_y) float32
        y_min_ptr,        # (D,) float32
        inv_dy_ptr,       # (D,) float32
        q_shift_ptr,      # (M, D, n_sub) float32
        t_prev_ptr,       # (M,) float32
        dt_sub_ptr,       # (M,) float32
        sqrt_dt_sub_ptr,  # (M,) float32
        fwd_rel_ptr,      # (D, M) float64
        perf_out_ptr,     # (R, M, D, N_rep) float64
        inv_dt_grid,
        d_assets,
        m_rem,
        n_rep: tl.constexpr,
        n_sub: tl.constexpr,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_d = tl.program_id(0)
        pid_r = tl.program_id(1)
        offs_n = tl.arange(0, BLOCK_N)

        lx = tl.zeros((BLOCK_N,), dtype=tl.float32)
        y_min = tl.load(y_min_ptr + pid_d)
        inv_dy = tl.load(inv_dy_ptr + pid_d)
        lv_base = pid_d * (N_t * N_y)
        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4
        inv_n_rep = 1.0 / float(n_rep)

        for m in range(m_rem):
            t_start = tl.load(t_prev_ptr + m)
            dt_sub = tl.load(dt_sub_ptr + m)
            sqrt_dt = tl.load(sqrt_dt_sub_ptr + m)

            for s in range(n_sub):
                s_f = tl.zeros((), dtype=tl.float32) + s
                t_step = t_start + (s_f + 0.5) * dt_sub
                ut = tl.minimum(tl.maximum(t_step * inv_dt_grid, 0.0), max_t_f)
                it0 = ut.to(tl.int32)
                ft = ut - it0.to(tl.float32)
                it1 = tl.minimum(it0 + 1, N_t - 1)

                q_sh = tl.load(q_shift_ptr + (m * d_assets + pid_d) * n_sub + s)
                uy = tl.minimum(tl.maximum((lx + q_sh - y_min) * inv_dy, 0.0), max_y_f)
                iy0 = uy.to(tl.int32)
                fy = uy - iy0.to(tl.float32)
                iy1 = tl.minimum(iy0 + 1, N_y - 1)

                row0 = lv_base + it0 * N_y
                row1 = lv_base + it1 * N_y
                v00 = tl.load(lv_table_ptr + row0 + iy0)
                v01 = tl.load(lv_table_ptr + row0 + iy1)
                v10 = tl.load(lv_table_ptr + row1 + iy0)
                v11 = tl.load(lv_table_ptr + row1 + iy1)

                v0 = v00 + fy * (v01 - v00)
                v1 = v10 + fy * (v11 - v10)
                lv = v0 + ft * (v1 - v0)

                z_off = (((pid_r * d_assets + pid_d) * m_rem + m) * n_sub + s) * n_rep + offs_n
                z = tl.load(z_steps_ptr + z_off)
                lx = lx + (-0.5 * lv * lv * dt_sub + lv * sqrt_dt * z)

            lx_f64 = lx.to(tl.float64)
            ex_f64 = tl.exp(lx_f64)
            mean_x = tl.sum(ex_f64, axis=0) * inv_n_rep
            log_shift = tl.log(tl.maximum(mean_x, 1e-12))
            lx_f64 = lx_f64 - log_shift
            lx = lx_f64.to(tl.float32)

            fwd_rel_m = tl.load(fwd_rel_ptr + pid_d * m_rem + m)
            p_off = ((pid_r * m_rem + m) * d_assets + pid_d) * n_rep + offs_n
            tl.store(perf_out_ptr + p_off, fwd_rel_m * tl.exp(lx_f64))

    @triton.jit  # type: ignore[untyped-decorator]
    def _payoff_horizon_triton_kernel(  # type: ignore[no-untyped-def]
        ko_all_ptr,       # (R, M, N_rep) float64
        cpn_all_ptr,      # (R, M, N_rep) float64
        df_pay_ptr,       # (M,) float64
        cpn_amt_ptr,      # (M,) float64
        w_fund_ptr,       # (M,) float64
        stop_out_ptr,     # (R, M, N_rep) float64
        cpn_al_out_ptr,   # (R, M, N_rep) float64
        surv_b_out_ptr,   # (R, M, N_rep) float64
        cpn_cf_out_ptr,   # (R, M, N_rep) float64
        alive_out_ptr,    # (R, N_rep) float64
        tpv_out_ptr,      # (R, N_rep) float64
        c_mem_0,
        ac_cpn,
        m_rem,
        n_rep: tl.constexpr,
        IS_MEM: tl.constexpr,
        NO_NOTIONAL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_r = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep

        al = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
        unp = tl.full((BLOCK_N,), c_mem_0, dtype=tl.float64)
        tpv = tl.zeros((BLOCK_N,), dtype=tl.float64)

        for m in range(m_rem):
            off_m = (pid_r * m_rem + m) * n_rep + offs_n
            ko_m = tl.load(ko_all_ptr + off_m, mask=mask_n, other=0.0)
            cpn_m = tl.load(cpn_all_ptr + off_m, mask=mask_n, other=0.0)
            df_m = tl.load(df_pay_ptr + m)
            c_m = tl.load(cpn_amt_ptr + m)
            wf_m = tl.load(w_fund_ptr + m)

            stop_m = al * ko_m
            cpn_al_m = al * cpn_m
            eff_c = (c_m + unp) if IS_MEM else c_m
            cpn_undisc_m = cpn_al_m * eff_c + stop_m * ac_cpn

            tl.store(stop_out_ptr + off_m, stop_m, mask=mask_n)
            tl.store(cpn_al_out_ptr + off_m, cpn_al_m, mask=mask_n)
            tl.store(surv_b_out_ptr + off_m, al, mask=mask_n)
            tl.store(cpn_cf_out_ptr + off_m, cpn_undisc_m, mask=mask_n)

            tpv = tpv - al * wf_m + df_m * cpn_undisc_m
            if IS_MEM:
                unp = al * (unp + c_m) * (1.0 - cpn_m)
            al = al * (1.0 - ko_m)
            if not NO_NOTIONAL:
                tpv = tpv + df_m * tl.where(m == m_rem - 1, stop_m + al, stop_m)

        out_rn = pid_r * n_rep + offs_n
        tl.store(alive_out_ptr + out_rn, al, mask=mask_n)
        tl.store(tpv_out_ptr + out_rn, tpv, mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _sample_lv_grid_triton(  # type: ignore[no-untyped-def]
        lv_table_ptr,
        lv_base,
        t_step,
        log_y,
        y_min,
        inv_dy,
        inv_dt_grid,
        max_t_f,
        max_y_f,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        mask_n,
    ):
        ut = tl.minimum(tl.maximum(t_step * inv_dt_grid, 0.0), max_t_f)
        it0 = ut.to(tl.int32)
        ft = ut - it0.to(tl.float32)
        it1 = tl.minimum(it0 + 1, N_t - 1)
        uy = tl.minimum(tl.maximum((log_y - y_min) * inv_dy, 0.0), max_y_f)
        iy0 = uy.to(tl.int32)
        fy = uy - iy0.to(tl.float32)
        iy1 = tl.minimum(iy0 + 1, N_y - 1)
        row0 = lv_base + it0 * N_y
        row1 = lv_base + it1 * N_y
        v00 = tl.load(lv_table_ptr + row0 + iy0, mask=mask_n, other=0.2)
        v01 = tl.load(lv_table_ptr + row0 + iy1, mask=mask_n, other=0.2)
        v10 = tl.load(lv_table_ptr + row1 + iy0, mask=mask_n, other=0.2)
        v11 = tl.load(lv_table_ptr + row1 + iy1, mask=mask_n, other=0.2)
        v0 = v00 + fy * (v01 - v00)
        v1 = v10 + fy * (v11 - v10)
        return v0 + ft * (v1 - v0)

    @triton.jit  # type: ignore[untyped-decorator]
    def _dupire_batch_substep_triton_kernel(  # type: ignore[no-untyped-def]
        log_x_ptr,         # (B_c, R, D, N_rep) float32
        dlog_x_ptr,        # (B_c, R, D, N_rep) float32
        z_indep_ptr,       # (R, D, n_sub, N_rep) float32
        chol_ptr,          # (B_c, D, D) float32
        lv_table_ptr,      # (B_c, D, N_t, N_y) float32
        y_min_ptr,         # (B_c, D) float32
        inv_dy_ptr,        # (B_c, D) float32
        q_shift_bnd_ptr,   # (B_c, D, 2) float32
        t_start_ptr,       # (B_c,) float32
        dt_sub_ptr,        # (B_c,) float32
        sqrt_dt_ptr,       # (B_c,) float32
        inv_dt_grid_ptr,   # (B_c,) float32
        r_reps: tl.constexpr,
        d_assets,
        n_rep: tl.constexpr,
        n_sub: tl.constexpr,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        BLOCK_N: tl.constexpr,
        COMPUTE_DELTA: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bd = tl.program_id(1)
        pid_r = tl.program_id(2)

        pid_b = pid_bd // d_assets
        pid_d = pid_bd - pid_b * d_assets

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep

        lx_off = ((pid_b * d_assets + pid_d) * r_reps + pid_r) * n_rep + offs_n
        lx = tl.load(log_x_ptr + lx_off, mask=mask_n, other=0.0)
        dlx = tl.load(dlog_x_ptr + lx_off, mask=mask_n, other=0.0) if COMPUTE_DELTA else tl.zeros((BLOCK_N,), dtype=tl.float32)

        y_min = tl.load(y_min_ptr + pid_b * d_assets + pid_d)
        inv_dy = tl.load(inv_dy_ptr + pid_b * d_assets + pid_d)
        q0 = tl.load(q_shift_bnd_ptr + (pid_b * d_assets + pid_d) * 2 + 0)
        q1 = tl.load(q_shift_bnd_ptr + (pid_b * d_assets + pid_d) * 2 + 1)
        t_start = tl.load(t_start_ptr + pid_b)
        dt_sub = tl.load(dt_sub_ptr + pid_b)
        sqrt_dt = tl.load(sqrt_dt_ptr + pid_b)
        inv_dt_grid = tl.load(inv_dt_grid_ptr + pid_b)

        lv_base = (pid_b * d_assets + pid_d) * (N_t * N_y)
        chol_row_base = (pid_b * d_assets + pid_d) * d_assets

        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4
        inv_n_sub = 1.0 / float(n_sub)

        for s in tl.static_range(n_sub):
            t_s0 = t_start + float(s) * dt_sub
            t_s1 = t_start + float(s + 1) * dt_sub
            q_sh0 = q0 + (float(s) * inv_n_sub) * (q1 - q0)
            q_sh1 = q0 + (float(s + 1) * inv_n_sub) * (q1 - q0)

            lv0 = _sample_lv_grid_triton(
                lv_table_ptr, lv_base, t_s0, lx + q_sh0, y_min, inv_dy, inv_dt_grid, max_t_f, max_y_f, N_t, N_y, mask_n
            )
            mu0 = -0.5 * lv0 * lv0
            y_h = lx + mu0 * dt_sub
            sig_step = lv0 * sqrt_dt
            lv_p = _sample_lv_grid_triton(
                lv_table_ptr, lv_base, t_s1, y_h + sig_step + q_sh1, y_min, inv_dy, inv_dt_grid, max_t_f, max_y_f, N_t, N_y, mask_n
            )
            lv_m = _sample_lv_grid_triton(
                lv_table_ptr, lv_base, t_s1, y_h - sig_step + q_sh1, y_min, inv_dy, inv_dt_grid, max_t_f, max_y_f, N_t, N_y, mask_n
            )
            mu_p = -0.5 * lv_p * lv_p
            mu_m = -0.5 * lv_m * lv_m

            z_c = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k in range(d_assets):
                l_dk = tl.load(chol_ptr + chol_row_base + k)
                z_k_off = ((pid_r * d_assets + k) * n_sub + s) * n_rep + offs_n
                z_k = tl.load(z_indep_ptr + z_k_off, mask=mask_n, other=0.0)
                z_c = z_c + l_dk * z_k

            lx = lx + (
                0.5 * (mu0 + 0.5 * (mu_p + mu_m)) * dt_sub
                + 0.25 * (mu_p - mu_m) * dt_sub * z_c
                + 0.25 * (lv_p + lv_m + 2.0 * lv0) * sqrt_dt * z_c
                + 0.25 * (lv_p - lv_m) * (z_c * z_c - 1.0) * sqrt_dt
            )
            if COMPUTE_DELTA:
                dlv_dy = (lv_p - lv_m) / tl.maximum(2.0 * sig_step, 1e-6)
                dlx = dlx + (-lv0 * dlv_dy * dt_sub + dlv_dy * sqrt_dt * z_c) * (1.0 + dlx)

        tl.store(log_x_ptr + lx_off, lx, mask=mask_n)
        if COMPUTE_DELTA:
            tl.store(dlog_x_ptr + lx_off, dlx, mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _build_lv_table_triton_kernel(  # type: ignore[no-untyped-def]
        strikes_ptr,      # (A, max_u) f64
        n_u_ptr,          # (A,) i32
        tenors_ptr,       # (A, max_exp) f64
        tenors_365_ptr,   # (A, max_exp) f64
        n_exp_ptr,        # (A,) i32
        quotes_ptr,       # (A, max_exp, max_u) f64
        slice_c_ptr,      # (A, 4, max_exp, max_u - 1) f64
        sl_left_ptr,      # (A, max_exp) f64
        sl_right_ptr,     # (A, max_exp) f64
        is_act365_ptr,    # (A,) i32
        min_vol_ptr,      # (A,) f64
        max_lv_ptr,       # (A,) f64
        f_grid_ptr,       # (A, N_t) f64
        r_zero_ptr,       # (A, N_t) f64
        q_repo_ptr,       # (A, N_t) f64
        t_eff_ptr,        # (A, N_t) f64
        y_min_ptr,        # (A,) f64
        dy_ptr,           # (A,) f64
        lv_out_ptr,       # (A, N_t, N_y) f32
        max_exp: tl.constexpr,
        max_u: tl.constexpr,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_a = tl.program_id(1)

        n_u = tl.load(n_u_ptr + pid_a)
        n_exp = tl.load(n_exp_ptr + pid_a)
        is_365 = tl.load(is_act365_ptr + pid_a)
        min_vol = tl.load(min_vol_ptr + pid_a)
        max_lv = tl.load(max_lv_ptr + pid_a)

        at_off = pid_a * N_t + pid_t
        f_t = tl.load(f_grid_ptr + at_off)
        r_z = tl.load(r_zero_ptr + at_off)
        q_r = tl.load(q_repo_ptr + at_off)
        t_eff = tl.load(t_eff_ptr + at_off)

        t_surf = t_eff
        dt_ratio = t_eff * 0.0 + 1.0
        if is_365 == 0:
            t365_base = pid_a * max_exp
            last_365 = tl.load(tenors_365_ptr + t365_base + n_exp - 1)
            last_dcc = tl.load(tenors_ptr + t365_base + n_exp - 1)
            if t_eff > last_365:
                ratio_ext = last_dcc / tl.maximum(last_365, 1e-12)
                t_surf = last_dcc + ratio_ext * (t_eff - last_365)
                dt_ratio = tl.maximum(ratio_ext, 1e-12)
            else:
                prev_365 = t_eff * 0.0
                prev_dcc = t_eff * 0.0
                found = 0
                for m in range(max_exp):
                    if m < n_exp and found == 0:
                        curr_365 = tl.load(tenors_365_ptr + t365_base + m)
                        curr_dcc = tl.load(tenors_ptr + t365_base + m)
                        if t_eff <= curr_365 or m == n_exp - 1:
                            denom_d = tl.maximum(curr_365 - prev_365, 1e-12)
                            slope = (curr_dcc - prev_dcc) / denom_d
                            t_surf = prev_dcc + slope * (t_eff - prev_365)
                            dt_ratio = tl.maximum(slope, 1e-12)
                            found = 1
                        prev_365 = curr_365
                        prev_dcc = curr_dcc

        t_base = pid_a * max_exp
        t_first = tl.load(tenors_ptr + t_base)
        t_last = tl.load(tenors_ptr + t_base + n_exp - 1)
        idx0 = 0
        idx1 = 0
        alpha = t_eff * 0.0
        t0 = t_first
        t1 = t_first
        mode_t = 0
        if n_exp == 1 or t_surf <= t_first:
            mode_t = 0
            idx0 = 0
            idx1 = 0
        elif t_surf >= t_last:
            mode_t = 2
            idx0 = n_exp - 1
            idx1 = n_exp - 1
        else:
            mode_t = 1
            for m in range(max_exp - 1):
                if m < n_exp - 1:
                    tm0 = tl.load(tenors_ptr + t_base + m)
                    tm1 = tl.load(tenors_ptr + t_base + m + 1)
                    if t_surf >= tm0 and t_surf <= tm1:
                        idx0 = m
                        idx1 = m + 1
                        t0 = tm0
                        t1 = tm1
            alpha = (t_surf - t0) / tl.maximum(t1 - t0, 1e-12)

        y_min = tl.load(y_min_ptr + pid_a)
        dy = tl.load(dy_ptr + pid_a)
        offs_y = tl.arange(0, N_y)
        y_vec = y_min + offs_y.to(tl.float64) * dy
        s_vec = tl.maximum(f_t * tl.exp(y_vec), 1e-4)

        s_base = pid_a * max_u
        k_min = tl.load(strikes_ptr + s_base)
        k_max = tl.load(strikes_ptr + s_base + n_u - 1)
        k_clip = tl.minimum(tl.maximum(s_vec, k_min), k_max)

        lo = tl.zeros((N_y,), dtype=tl.int32)
        hi = tl.full((N_y,), n_u - 1, dtype=tl.int32)
        for _ in tl.static_range(10):
            mid = (lo + hi) >> 1
            k_mid = tl.load(strikes_ptr + s_base + mid)
            go_right = k_clip >= k_mid
            lo = tl.where(go_right, mid, lo)
            hi = tl.where(go_right, hi, mid)
        seg_idx = tl.minimum(tl.maximum(lo, 0), n_u - 2)
        k_left = tl.load(strikes_ptr + s_base + seg_idx)
        dx = k_clip - k_left

        max_um1 = max_u - 1
        sc_stride_a = 4 * max_exp * max_um1
        sc_stride_c = max_exp * max_um1
        sc_stride_e = max_um1

        sc0_base = pid_a * sc_stride_a + idx0 * sc_stride_e + seg_idx
        c0_0 = tl.load(slice_c_ptr + sc0_base)
        c1_0 = tl.load(slice_c_ptr + sc0_base + sc_stride_c)
        c2_0 = tl.load(slice_c_ptr + sc0_base + 2 * sc_stride_c)
        c3_0 = tl.load(slice_c_ptr + sc0_base + 3 * sc_stride_c)
        v0 = ((c0_0 * dx + c1_0) * dx + c2_0) * dx + c3_0
        dv0 = (3.0 * c0_0 * dx + 2.0 * c1_0) * dx + c2_0
        d2v0 = 6.0 * c0_0 * dx + 2.0 * c1_0

        left_mask = s_vec < k_min
        right_mask = s_vec > k_max
        q0_l = tl.load(quotes_ptr + (pid_a * max_exp + idx0) * max_u)
        q0_r = tl.load(quotes_ptr + (pid_a * max_exp + idx0) * max_u + n_u - 1)
        sl0_l = tl.load(sl_left_ptr + t_base + idx0)
        sl0_r = tl.load(sl_right_ptr + t_base + idx0)
        v0 = tl.where(left_mask, q0_l + sl0_l * (s_vec - k_min), tl.where(right_mask, q0_r + sl0_r * (s_vec - k_max), v0))
        dv0 = tl.where(left_mask, sl0_l, tl.where(right_mask, sl0_r, dv0))
        d2v0 = tl.where(left_mask | right_mask, 0.0, d2v0)
        cl0 = v0 < min_vol
        v0 = tl.maximum(v0, min_vol)
        dv0 = tl.where(cl0, 0.0, dv0)
        d2v0 = tl.where(cl0, 0.0, d2v0)

        if mode_t != 1:
            w = (v0 * v0) * t_surf
            dw_dt = v0 * v0
            dw_dk = 2.0 * t_surf * v0 * dv0
            d2w_dk2 = 2.0 * t_surf * (dv0 * dv0 + v0 * d2v0)
        else:
            sc1_base = pid_a * sc_stride_a + idx1 * sc_stride_e + seg_idx
            c0_1 = tl.load(slice_c_ptr + sc1_base)
            c1_1 = tl.load(slice_c_ptr + sc1_base + sc_stride_c)
            c2_1 = tl.load(slice_c_ptr + sc1_base + 2 * sc_stride_c)
            c3_1 = tl.load(slice_c_ptr + sc1_base + 3 * sc_stride_c)
            v1 = ((c0_1 * dx + c1_1) * dx + c2_1) * dx + c3_1
            dv1 = (3.0 * c0_1 * dx + 2.0 * c1_1) * dx + c2_1
            d2v1 = 6.0 * c0_1 * dx + 2.0 * c1_1

            q1_l = tl.load(quotes_ptr + (pid_a * max_exp + idx1) * max_u)
            q1_r = tl.load(quotes_ptr + (pid_a * max_exp + idx1) * max_u + n_u - 1)
            sl1_l = tl.load(sl_left_ptr + t_base + idx1)
            sl1_r = tl.load(sl_right_ptr + t_base + idx1)
            v1 = tl.where(left_mask, q1_l + sl1_l * (s_vec - k_min), tl.where(right_mask, q1_r + sl1_r * (s_vec - k_max), v1))
            dv1 = tl.where(left_mask, sl1_l, tl.where(right_mask, sl1_r, dv1))
            d2v1 = tl.where(left_mask | right_mask, 0.0, d2v1)
            cl1 = v1 < min_vol
            v1 = tl.maximum(v1, min_vol)
            dv1 = tl.where(cl1, 0.0, dv1)
            d2v1 = tl.where(cl1, 0.0, d2v1)

            w0 = (v0 * v0) * t0
            w1 = (v1 * v1) * t1
            w = (1.0 - alpha) * w0 + alpha * w1
            dw_dt = (w1 - w0) / tl.maximum(t1 - t0, 1e-12)
            dw_dk = (1.0 - alpha) * (2.0 * t0 * v0 * dv0) + alpha * (2.0 * t1 * v1 * dv1)
            d2w_dk2 = (1.0 - alpha) * (2.0 * t0 * (dv0 * dv0 + v0 * d2v0)) + alpha * (2.0 * t1 * (dv1 * dv1 + v1 * d2v1))

        w_safe = tl.maximum(w, 1e-8)
        dw_dy = s_vec * dw_dk
        dw_dt_y = dw_dt + (r_z - q_r) * dw_dy
        d2w_dy2 = (s_vec * s_vec) * d2w_dk2 + dw_dy
        denom = (
            1.0
            - (y_vec / w_safe) * dw_dy
            + 0.25 * (-0.25 - 1.0 / w_safe + (y_vec * y_vec) / (w_safe * w_safe)) * (dw_dy * dw_dy)
            + 0.5 * d2w_dy2
        )
        denom_safe = tl.maximum(denom, 0.05)
        lv_dcc = tl.sqrt(tl.maximum(dw_dt_y / denom_safe, min_vol * min_vol))
        lv_365 = tl.minimum(tl.maximum(lv_dcc * tl.sqrt(dt_ratio), min_vol), max_lv)

        out_off = (pid_a * N_t + pid_t) * N_y + offs_y
        tl.store(lv_out_ptr + out_off, lv_365.to(tl.float32))

    @triton.jit  # type: ignore[untyped-decorator]
    def _training_period_step_triton_kernel(  # type: ignore[no-untyped-def]
        alive_ptr,      # (B_c, N_tot) f64
        unpaid_ptr,     # (B_c, N_tot) f64
        total_pv_ptr,   # (B_c, N_tot) f64
        ko_ptr,         # (B_c, N_tot) f64
        cpn_ptr,        # (B_c, N_tot) f64 (raw cpn_ind)
        dko_dw_ptr,     # (B_c, N_tot) f64
        dcpn_dw_ptr,    # (B_c, N_tot) f64
        d_alive_ptr,    # (B_c, D, N_tot) f64
        d_unpaid_ptr,   # (B_c, D, N_tot) f64
        d_total_pv_ptr, # (B_c, D, N_tot) f64
        dw_call_ptr,    # (B_c, D, N_tot) f64
        dw_cpn_ptr,     # (B_c, D, N_tot) f64
        df_pay_m_ptr,   # (B_c,) f64
        cpn_amt_m_ptr,  # (B_c,) f64
        w_fund_m_ptr,   # (B_c,) f64
        ac_cpn_ptr,     # (B_c,) f64
        is_mem_ptr,     # (B_c,) f64
        force_cpn_ptr,  # (B_c,) f64
        no_not_ptr,     # (B_c,) f64
        n_tot,
        d_assets,
        is_last,
        COMPUTE_DELTA: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_tot
        bn_off = pid_b * n_tot + offs_n

        al = tl.load(alive_ptr + bn_off, mask=mask_n, other=0.0)
        unp = tl.load(unpaid_ptr + bn_off, mask=mask_n, other=0.0)
        tpv = tl.load(total_pv_ptr + bn_off, mask=mask_n, other=0.0)
        ko = tl.load(ko_ptr + bn_off, mask=mask_n, other=0.0)
        cpn_i = tl.load(cpn_ptr + bn_off, mask=mask_n, other=0.0)

        df_m = tl.load(df_pay_m_ptr + pid_b)
        c_m = tl.load(cpn_amt_m_ptr + pid_b)
        wf_m = tl.load(w_fund_m_ptr + pid_b)
        ac_c = tl.load(ac_cpn_ptr + pid_b)
        is_mem = tl.load(is_mem_ptr + pid_b)
        force_cpn = tl.load(force_cpn_ptr + pid_b)
        no_not = tl.load(no_not_ptr + pid_b)

        force_cond = (force_cpn > 0.5) & (ko > cpn_i)
        cpn_e = tl.where(force_cond, ko, cpn_i)
        eff_c = tl.where(is_mem > 0.5, c_m + unp, c_m)
        stop_m = al * ko

        tpv_next = tpv - al * wf_m + df_m * (al * cpn_e * eff_c + stop_m * ac_c)
        al_next = al * (1.0 - ko)
        unp_next = tl.where(is_mem > 0.5, al * eff_c * (1.0 - cpn_e), unp)
        not_mult = (1.0 - no_not) * df_m
        tpv_next = tpv_next + not_mult * tl.where(is_last != 0, stop_m + al_next, stop_m)

        tl.store(alive_ptr + bn_off, al_next, mask=mask_n)
        tl.store(unpaid_ptr + bn_off, unp_next, mask=mask_n)
        tl.store(total_pv_ptr + bn_off, tpv_next, mask=mask_n)

        if COMPUTE_DELTA:
            dko_w = tl.load(dko_dw_ptr + bn_off, mask=mask_n, other=0.0)
            dcpn_w = tl.load(dcpn_dw_ptr + bn_off, mask=mask_n, other=0.0)
            for d in range(d_assets):
                off_3d = (pid_b * d_assets + d) * n_tot + offs_n
                d_al = tl.load(d_alive_ptr + off_3d, mask=mask_n, other=0.0)
                d_unp = tl.load(d_unpaid_ptr + off_3d, mask=mask_n, other=0.0)
                d_tpv = tl.load(d_total_pv_ptr + off_3d, mask=mask_n, other=0.0)
                dwc = tl.load(dw_call_ptr + off_3d, mask=mask_n, other=0.0)
                dwq = tl.load(dw_cpn_ptr + off_3d, mask=mask_n, other=0.0)

                d_ko = dko_w * dwc
                d_cpn = tl.where(force_cond, d_ko, dcpn_w * dwq)
                d_stop = d_al * ko + al * d_ko
                d_eff_c = tl.where(is_mem > 0.5, d_unp, d_unp * 0.0)
                d_cpn_pay = (
                    d_al * (cpn_e * eff_c)
                    + al * (d_cpn * eff_c + cpn_e * d_eff_c)
                    + d_stop * ac_c
                )
                d_al_next = d_al * (1.0 - ko) - al * d_ko
                d_unp_next = tl.where(
                    is_mem > 0.5,
                    d_al * (eff_c * (1.0 - cpn_e)) + al * (d_eff_c * (1.0 - cpn_e) - eff_c * d_cpn),
                    d_unp,
                )
                d_tpv_next = d_tpv - d_al * wf_m + df_m * d_cpn_pay + not_mult * tl.where(is_last != 0, d_stop + d_al_next, d_stop)

                tl.store(d_alive_ptr + off_3d, d_al_next, mask=mask_n)
                tl.store(d_unpaid_ptr + off_3d, d_unp_next, mask=mask_n)
                tl.store(d_total_pv_ptr + off_3d, d_tpv_next, mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _sobol_vec(  # type: ignore[no-untyped-def]
        sv_ptr,
        dim_idx,
        shift_u32,
        pid_u32,
        offs_u32,
        m_pow,
        LOW_BITS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Returns BLOCK_N float32 standard normal variates via Gray-code Sobol + uint32 XOR shift."""
        acc = tl.full((BLOCK_N,), shift_u32, dtype=tl.uint32)
        gray = offs_u32 ^ (offs_u32 >> 1)
        z_vec = tl.zeros((BLOCK_N,), dtype=tl.uint32)
        for b in tl.static_range(LOW_BITS):
            bit_on = (gray >> b) & 1
            dir_num = tl.load(sv_ptr + dim_idx * 30 + b).to(tl.uint32)
            acc = acc ^ tl.where(bit_on != 0, dir_num, z_vec)
        if m_pow > LOW_BITS:
            pid_gray = pid_u32 ^ (pid_u32 >> 1)
            hi_acc = tl.zeros((), dtype=tl.uint32)
            z_sca = tl.zeros((), dtype=tl.uint32)
            for b_hi in range(LOW_BITS, m_pow):
                hi_bit = (pid_gray >> (b_hi - LOW_BITS)) & 1
                dir_num = tl.load(sv_ptr + dim_idx * 30 + b_hi).to(tl.uint32)
                hi_acc = hi_acc ^ tl.where(hi_bit != 0, dir_num, z_sca)
            acc = acc ^ hi_acc
        u = (acc.to(tl.float32) + 0.5) * (1.0 / 1073741824.0)
        u = tl.minimum(tl.maximum(u, 1e-6), 1.0 - 1e-6)
        return tl.extra.cuda.libdevice.normcdfinv(u)

    @triton.jit  # type: ignore[untyped-decorator]
    def _sobol_vec_flat(  # type: ignore[no-untyped-def]
        sv_ptr,
        dim_idx,
        shift_u32,
        offs_u32,
        m_pow,
        BLOCK_N: tl.constexpr,
    ):
        """Single-CTA Sobol normal generator using dynamic loop over m_pow bits (no n_rep recompile)."""
        acc = tl.full((BLOCK_N,), shift_u32, dtype=tl.uint32)
        gray = offs_u32 ^ (offs_u32 >> 1)
        z_vec = tl.zeros((BLOCK_N,), dtype=tl.uint32)
        for b in range(m_pow):
            bit_on = (gray >> b) & 1
            dir_num = tl.load(sv_ptr + dim_idx * 30 + b).to(tl.uint32)
            acc = acc ^ tl.where(bit_on != 0, dir_num, z_vec)
        u = (acc.to(tl.float32) + 0.5) * (1.0 / 1073741824.0)
        u = tl.minimum(tl.maximum(u, 1e-6), 1.0 - 1e-6)
        return tl.extra.cuda.libdevice.normcdfinv(u)

    @triton.jit  # type: ignore[untyped-decorator]
    def _sobol_bridge_period_chunk_kernel(  # type: ignore[no-untyped-def]
        sv_ptr,          # (21201, 30) int32
        shifts_c_ptr,    # (R, D) int32 for period m
        shifts_s_ptr,    # (R, D * n_sub) int32 for period m
        z_coarse_ptr,    # (R, D, N_rep) float32 out
        z_sub_ptr,       # (R, D, n_sub, N_rep) float32 out
        z_haar_ptr,      # (R, D, n_sub, N_rep) float32 workspace when IS_HAAR else stub
        coarse_dim_base,
        sub_dim_base,
        period_m,
        md_total,
        d_assets,
        n_sub,
        haar_levels,
        inv_sqrt_sub,    # float32 = 1 / sqrt(n_sub)
        inv_n_sub,       # float32 = 1 / n_sub
        n_rep,
        m_pow,
        HAS_SUB: tl.constexpr,
        IS_CORR_COARSE: tl.constexpr,
        IS_HAAR: tl.constexpr,
        LOW_BITS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Generates single-period Sobol coarse and nested Haar Brownian-bridge substeps on chip."""
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_r = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep
        offs_u32 = offs_n.to(tl.uint32)
        pid_u32 = pid_n.to(tl.uint32)

        if IS_CORR_COARSE:
            zc_base = tl.zeros((BLOCK_N,), dtype=tl.float32)
        else:
            sh_c = tl.load(shifts_c_ptr + pid_r * d_assets + pid_d).to(tl.uint32)
            zc_base = _sobol_vec(
                sv_ptr, coarse_dim_base + pid_d, sh_c, pid_u32, offs_u32, m_pow, LOW_BITS, BLOCK_N
            )
            c_off = (pid_r * d_assets + pid_d) * n_rep + offs_n
            tl.store(z_coarse_ptr + c_off, zc_base, mask=mask_n)

        if not HAS_SUB:
            if not IS_CORR_COARSE:
                tl.store(
                    z_sub_ptr + ((pid_r * d_assets + pid_d) * n_sub) * n_rep + offs_n,
                    zc_base,
                    mask=mask_n,
                )
            return

        rd_sub_base = (pid_r * d_assets + pid_d) * n_sub
        sub_sh_base = (pid_r * d_assets + pid_d) * n_sub

        if IS_HAAR:
            one_i32 = tl.zeros((), dtype=tl.int32) + 1
            for lvl in range(haar_levels):
                k_pow = one_i32 << lvl
                dim_blk = k_pow * md_total + (period_m * d_assets + pid_d) * k_pow
                mode_base = k_pow - 1
                for k in range(k_pow):
                    mode = mode_base + k
                    sh_s = tl.load(shifts_s_ptr + sub_sh_base + mode).to(tl.uint32)
                    b_mode = _sobol_vec(
                        sv_ptr, dim_blk + k, sh_s, pid_u32, offs_u32, m_pow, LOW_BITS, BLOCK_N
                    )
                    tl.store(z_haar_ptr + (rd_sub_base + mode) * n_rep + offs_n, b_mode, mask=mask_n)

            for s in range(n_sub):
                s_i32 = tl.zeros((), dtype=tl.int32) + s
                acc = tl.zeros((BLOCK_N,), dtype=tl.float32) if IS_CORR_COARSE else zc_base
                for lvl in range(haar_levels):
                    k_pow = one_i32 << lvl
                    shift_b = haar_levels - 1 - lvl
                    bit = (s_i32 >> shift_b) & 1
                    mode = (k_pow - 1) + (s_i32 >> (shift_b + 1))
                    b_mode = tl.load(z_haar_ptr + (rd_sub_base + mode) * n_rep + offs_n, mask=mask_n, other=0.0)
                    w_lvl = tl.sqrt(k_pow.to(tl.float32))
                    acc = acc + tl.where(bit == 0, w_lvl, -w_lvl) * b_mode
                tl.store(z_sub_ptr + (rd_sub_base + s) * n_rep + offs_n, acc * inv_sqrt_sub, mask=mask_n)
            return

        dim_s_base = sub_dim_base + pid_d * n_sub
        sum_z = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for s in range(n_sub):
            sh_s = tl.load(shifts_s_ptr + sub_sh_base + s).to(tl.uint32)
            zs = _sobol_vec(sv_ptr, dim_s_base + s, sh_s, pid_u32, offs_u32, m_pow, LOW_BITS, BLOCK_N)
            sum_z = sum_z + zs
            tl.store(z_sub_ptr + (rd_sub_base + s) * n_rep + offs_n, zs, mask=mask_n)

        adj = (
            (-sum_z * inv_n_sub)
            if IS_CORR_COARSE
            else (zc_base * inv_sqrt_sub - sum_z * inv_n_sub)
        )
        for s in range(n_sub):
            s_off = (rd_sub_base + s) * n_rep + offs_n
            zs = tl.load(z_sub_ptr + s_off, mask=mask_n, other=0.0)
            tl.store(z_sub_ptr + s_off, zs + adj, mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _dupire_step_period_tile_kernel(  # type: ignore[no-untyped-def]
        z_coarse_ptr,       # (R, M, D, N_rep) if IS_CORR_COARSE else (R, D, N_rep) float32
        z_sub_ptr,          # (R, D, n_sub, N_rep) float32
        chol_ptr,           # (D, D) float32 lower triangular
        lv_table_ptr,       # (D, N_t, N_y) float32
        y_min_ptr,          # (D,) float32
        inv_dy_ptr,         # (D,) float32
        q_shift_bnd_m_ptr,  # (D, 2) float32
        lx_buf_ptr,         # (R, D, N_rep) float32 in/out
        ex_buf_ptr,         # (R, D, N_rep) float64 out
        tile_ex_sum_ptr,    # (R, D, n_tiles) float64 in/out
        z_flat_cum_ptr,     # (R, N_rep) float64 in/out
        period_idx,
        m_rem,
        t_start,            # float32
        dt_sub,             # float32
        sqrt_dt_sub,        # float32
        inv_dt_grid,        # float32
        inv_n_sub,          # float32
        inv_sqrt_sub,       # float32
        inv_n_rep,          # float64 = 1 / n_rep
        sqrt_dvar_dom_m,    # float64
        dom_idx,
        d_assets,
        n_rep,
        n_sub,
        n_tiles,
        HAS_CV: tl.constexpr,
        HAS_SUB: tl.constexpr,
        IS_CORR_COARSE: tl.constexpr,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Steps one period across BLOCK_N paths with Cholesky, q_shift, Platen Weak-2.0, and fp64 EMS sums."""
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_r = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep
        rd_off = (pid_r * d_assets + pid_d) * n_rep + offs_n

        if period_idx == 0:
            lx = tl.zeros((BLOCK_N,), dtype=tl.float32)
        else:
            lx = tl.load(lx_buf_ptr + rd_off, mask=mask_n, other=0.0)
            ex_sum = tl.zeros((), dtype=tl.float64)
            tes_base = (pid_r * d_assets + pid_d) * n_tiles
            for t in range(n_tiles):
                ex_sum = ex_sum + tl.load(tile_ex_sum_ptr + tes_base + t)
            log_shift = tl.log(tl.maximum(ex_sum * inv_n_rep, 1e-12))
            lx = (lx.to(tl.float64) - log_shift).to(tl.float32)

        if IS_CORR_COARSE:
            zc_d = tl.load(
                z_coarse_ptr + ((pid_r * m_rem + period_idx) * d_assets + pid_d) * n_rep + offs_n,
                mask=mask_n,
                other=0.0,
            )
            if HAS_CV and (pid_d == dom_idx):
                cv_off = pid_r * n_rep + offs_n
                zf = (
                    tl.zeros((BLOCK_N,), dtype=tl.float64)
                    if period_idx == 0
                    else tl.load(z_flat_cum_ptr + cv_off, mask=mask_n, other=0.0)
                )
                zf = zf + sqrt_dvar_dom_m * zc_d.to(tl.float64)
                tl.store(z_flat_cum_ptr + cv_off, zf, mask=mask_n)
        else:
            zc_d = tl.zeros((BLOCK_N,), dtype=tl.float32)
            if HAS_CV and (pid_d == dom_idx):
                zc_dom = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for j in range(pid_d + 1):
                    l_dj = tl.load(chol_ptr + pid_d * d_assets + j)
                    zj = tl.load(z_coarse_ptr + (pid_r * d_assets + j) * n_rep + offs_n, mask=mask_n, other=0.0)
                    zc_dom = zc_dom + l_dj * zj
                cv_off = pid_r * n_rep + offs_n
                zf = (
                    tl.zeros((BLOCK_N,), dtype=tl.float64)
                    if period_idx == 0
                    else tl.load(z_flat_cum_ptr + cv_off, mask=mask_n, other=0.0)
                )
                zf = zf + sqrt_dvar_dom_m * zc_dom.to(tl.float64)
                tl.store(z_flat_cum_ptr + cv_off, zf, mask=mask_n)

        y_min = tl.load(y_min_ptr + pid_d)
        inv_dy = tl.load(inv_dy_ptr + pid_d)
        q0 = tl.load(q_shift_bnd_m_ptr + pid_d * 2 + 0)
        q1 = tl.load(q_shift_bnd_m_ptr + pid_d * 2 + 1)
        lv_base = pid_d * (N_t * N_y)
        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4

        for s in range(n_sub):
            s_f = tl.zeros((), dtype=tl.float32) + s
            t_s0 = t_start + s_f * dt_sub
            t_s1 = t_start + (s_f + 1.0) * dt_sub
            q_sh0 = q0 + (s_f * inv_n_sub) * (q1 - q0)
            q_sh1 = q0 + ((s_f + 1.0) * inv_n_sub) * (q1 - q0)

            lv0 = _sample_lv_grid_triton(
                lv_table_ptr, lv_base, t_s0, lx + q_sh0, y_min, inv_dy, inv_dt_grid, max_t_f, max_y_f, N_t, N_y, mask_n
            )
            mu0 = -0.5 * lv0 * lv0
            y_h = lx + mu0 * dt_sub
            sig_step = lv0 * sqrt_dt_sub
            lv_p = _sample_lv_grid_triton(
                lv_table_ptr, lv_base, t_s1, y_h + sig_step + q_sh1, y_min, inv_dy, inv_dt_grid, max_t_f, max_y_f, N_t, N_y, mask_n
            )
            lv_m = _sample_lv_grid_triton(
                lv_table_ptr, lv_base, t_s1, y_h - sig_step + q_sh1, y_min, inv_dy, inv_dt_grid, max_t_f, max_y_f, N_t, N_y, mask_n
            )
            mu_p = -0.5 * lv_p * lv_p
            mu_m = -0.5 * lv_m * lv_m

            if IS_CORR_COARSE and (not HAS_SUB):
                z_corr = zc_d
            else:
                z_corr = (zc_d * inv_sqrt_sub) if IS_CORR_COARSE else tl.zeros((BLOCK_N,), dtype=tl.float32)
                for j in range(pid_d + 1):
                    l_dj = tl.load(chol_ptr + pid_d * d_assets + j)
                    zj = tl.load(
                        z_sub_ptr + ((pid_r * d_assets + j) * n_sub + s) * n_rep + offs_n,
                        mask=mask_n,
                        other=0.0,
                    )
                    z_corr = z_corr + l_dj * zj
            lx = lx + (
                0.5 * (mu0 + 0.5 * (mu_p + mu_m)) * dt_sub
                + 0.25 * (mu_p - mu_m) * dt_sub * z_corr
                + 0.25 * (lv_p + lv_m + 2.0 * lv0) * sqrt_dt_sub * z_corr
                + 0.25 * (lv_p - lv_m) * (z_corr * z_corr - 1.0) * sqrt_dt_sub
            )

        tl.store(lx_buf_ptr + rd_off, lx, mask=mask_n)
        ex64 = tl.where(mask_n, tl.exp(lx.to(tl.float64)), 0.0)
        tl.store(ex_buf_ptr + rd_off, ex64, mask=mask_n)
        tl.store(tile_ex_sum_ptr + ((pid_r * d_assets + pid_d) * n_tiles + pid_n), tl.sum(ex64, axis=0))

    @triton.jit  # type: ignore[untyped-decorator]
    def _eval_index_from_stats_triton(  # type: ignore[no-untyped-def]
        CODE: tl.constexpr,
        D_ASSETS: tl.constexpr,
        w_min,
        w_max,
        w_sum,
        w_disp,
        p_lex0,
        p_lex1,
        w_min2,
        w_min3,
        w_max2,
    ):
        """Evaluates any of the 18 canonical Murex monitoring functionals from asset stats."""
        inv_d = 1.0 / float(D_ASSETS)
        w_avg = w_sum * inv_d
        zero = w_min * 0.0
        if CODE == 0:
            res = w_min
        elif CODE == 1:
            res = w_max
        elif CODE == 2:
            res = w_avg
        elif CODE == 3:
            res = zero if D_ASSETS <= 1 else tl.sqrt(tl.maximum(w_disp * inv_d, 0.0))
        elif CODE == 4:
            res = p_lex0 if D_ASSETS < 2 else (p_lex0 / tl.maximum(p_lex1, 1e-12))
        elif CODE == 5:
            res = p_lex0 if D_ASSETS < 2 else (p_lex1 / tl.maximum(p_lex0, 1e-12))
        elif CODE == 6:
            res = w_max / tl.maximum(w_min, 1e-12)
        elif CODE == 7:
            res = w_max / tl.maximum(w_avg, 1e-12)
        elif CODE == 8:
            res = w_avg / tl.maximum(w_min, 1e-12)
        elif CODE == 9:
            res = zero if D_ASSETS < 2 else (p_lex0 - p_lex1)
        elif CODE == 10:
            res = zero if D_ASSETS < 2 else (p_lex1 - p_lex0)
        elif CODE == 11:
            res = w_max - w_min
        elif CODE == 12:
            res = w_max - w_avg
        elif CODE == 13:
            res = w_avg - w_min
        elif CODE == 15:
            res = w_min if D_ASSETS < 2 else w_min2
        elif CODE == 16:
            res = w_min if D_ASSETS < 2 else (w_min2 if D_ASSETS == 2 else w_min3)
        elif CODE == 17:
            res = w_max if D_ASSETS < 2 else w_max2
        else:
            res = tl.abs(w_min - 1.0) + 1.0
        return res

    @triton.jit  # type: ignore[untyped-decorator]
    def _mf_period_step_triton(  # type: ignore[no-untyped-def]
        w_call,
        w_cpn,
        w_opt,
        al,
        unp,
        tpv,
        not_ki,
        is_call,
        b_ko_m,
        is_guar,
        b_cpn_m,
        df_m,
        c_m,
        wf_m,
        period_idx,
        m_rem,
        ac_cpn,
        ko_spread,
        cpn_spread,
        ki_barrier,
        ki_spread,
        IS_MEM: tl.constexpr,
        NO_NOTIONAL: tl.constexpr,
        FORCE_CPN_AC: tl.constexpr,
        CPN_DIR_DOWN: tl.constexpr,
        CPN_TYPE_OUT: tl.constexpr,
        HAS_KO_SPREAD: tl.constexpr,
        HAS_CPN_SPREAD: tl.constexpr,
        HAS_KI: tl.constexpr,
        HAS_KI_SPREAD: tl.constexpr,
        IS_DISCRETE_KI: tl.constexpr,
    ):
        """Advances pathwise autocall survival, memory coupons, funding, and notional over one period."""
        if HAS_KO_SPREAD:
            blo_ko = tl.minimum(b_ko_m, b_ko_m + ko_spread)
            bhi_ko = tl.maximum(b_ko_m, b_ko_m + ko_spread)
            ko_raw = tl.minimum(
                tl.maximum((w_call - blo_ko) / tl.maximum(bhi_ko - blo_ko, 1e-9), 0.0), 1.0
            )
        else:
            ko_raw = tl.where(w_call >= b_ko_m, 1.0, 0.0)
        ko_ind = tl.where(is_call, ko_raw, 0.0)

        eff_s_cpn = -cpn_spread if CPN_DIR_DOWN else cpn_spread
        if HAS_CPN_SPREAD:
            blo_cp = tl.minimum(b_cpn_m, b_cpn_m + eff_s_cpn)
            bhi_cp = tl.maximum(b_cpn_m, b_cpn_m + eff_s_cpn)
            cpn_ramp = tl.minimum(
                tl.maximum((w_cpn - blo_cp) / tl.maximum(bhi_cp - blo_cp, 1e-9), 0.0), 1.0
            )
        else:
            cpn_ramp = tl.where(w_cpn >= b_cpn_m, 1.0, 0.0)
        base_cpn = (1.0 - cpn_ramp) if CPN_DIR_DOWN else cpn_ramp
        cpn_raw = (1.0 - base_cpn) if CPN_TYPE_OUT else base_cpn
        cpn_ind = tl.where(is_guar, 1.0, cpn_raw)
        if FORCE_CPN_AC:
            cpn_ind = tl.maximum(cpn_ind, ko_ind)

        if HAS_KI and IS_DISCRETE_KI:
            if HAS_KI_SPREAD:
                blo_ki = tl.minimum(ki_barrier, ki_barrier + ki_spread)
                bhi_ki = tl.maximum(ki_barrier, ki_barrier + ki_spread)
                ki_up_m = tl.minimum(
                    tl.maximum((w_opt - blo_ki) / tl.maximum(bhi_ki - blo_ki, 1e-9), 0.0), 1.0
                )
            else:
                ki_up_m = tl.where(w_opt >= ki_barrier, 1.0, 0.0)
            not_ki_next = not_ki * ki_up_m
        else:
            not_ki_next = not_ki

        tpv_next = tpv - al * wf_m
        stop_m = al * ko_ind
        cpn_al_m = al * cpn_ind
        eff_c = (c_m + unp) if IS_MEM else c_m
        cpn_undisc_m = cpn_al_m * eff_c + stop_m * ac_cpn
        tpv_next = tpv_next + df_m * cpn_undisc_m

        if IS_MEM:
            unp_next = al * (unp + c_m) * (1.0 - cpn_ind)
        else:
            unp_next = unp
        al_next = al * (1.0 - ko_ind)

        if not NO_NOTIONAL:
            if period_idx < m_rem - 1:
                tpv_next = tpv_next + df_m * stop_m
            else:
                tpv_next = tpv_next + df_m * (stop_m + al_next)

        return al_next, unp_next, tpv_next, not_ki_next, stop_m, cpn_al_m, cpn_undisc_m

    @triton.jit  # type: ignore[untyped-decorator]
    def _mf_terminal_opt_triton(  # type: ignore[no-untyped-def]
        w_min,
        w_max,
        w_avg,
        w_opt,
        al_next,
        not_ki,
        df_term,
        strike,
        max_spread_put,
        option_gearing,
        ki_barrier,
        ki_spread,
        one_star_strike,
        one_star_eps,
        one_star_coupon,
        pcall_strike,
        pcall_rate,
        HAS_KI: tl.constexpr,
        HAS_KI_SPREAD: tl.constexpr,
        IS_DISCRETE_KI: tl.constexpr,
        HAS_PUT: tl.constexpr,
        HAS_LOW_K: tl.constexpr,
        HAS_ONESTAR: tl.constexpr,
        HAS_OS_CPN: tl.constexpr,
        HAS_PCALL: tl.constexpr,
        IS_LONG_BASKET: tl.constexpr,
    ):
        """Evaluates terminal put/call/KI/OneStar/IsLongBasket payoff on surviving paths."""
        if HAS_ONESTAR:
            star_ind = tl.minimum(
                tl.maximum((w_max - (one_star_strike - 0.5 * one_star_eps)) / one_star_eps, 0.0), 1.0
            )
        else:
            star_ind = w_max * 0.0

        if IS_LONG_BASKET:
            if HAS_KI and IS_DISCRETE_KI:
                ki_wof_k = 1.0 - not_ki
            else:
                if HAS_KI_SPREAD:
                    blo_lb = tl.minimum(strike, strike + ki_spread)
                    bhi_lb = tl.maximum(strike, strike + ki_spread)
                    ki_up_lb = tl.minimum(
                        tl.maximum((w_min - blo_lb) / tl.maximum(bhi_lb - blo_lb, 1e-9), 0.0), 1.0
                    )
                else:
                    ki_up_lb = tl.where(w_min >= strike, 1.0, 0.0)
                ki_wof_k = 1.0 - ki_up_lb
            eff_gearing = tl.where(tl.abs(option_gearing) > 1e-12, option_gearing, 1.0)
            put_int = ((1.0 - w_avg) * ki_wof_k) / eff_gearing
            if HAS_ONESTAR:
                put_int = put_int * (1.0 - star_ind)
            unit_put = al_next * put_int
        elif HAS_PUT:
            put_int = tl.maximum(strike - w_opt, 0.0)
            if HAS_LOW_K:
                put_int = tl.minimum(put_int, max_spread_put)
            if HAS_KI:
                if IS_DISCRETE_KI:
                    ki_ind = 1.0 - not_ki
                else:
                    if HAS_KI_SPREAD:
                        blo_ki = tl.minimum(ki_barrier, ki_barrier + ki_spread)
                        bhi_ki = tl.maximum(ki_barrier, ki_barrier + ki_spread)
                        ki_up = tl.minimum(
                            tl.maximum((w_opt - blo_ki) / tl.maximum(bhi_ki - blo_ki, 1e-9), 0.0), 1.0
                        )
                    else:
                        ki_up = tl.where(w_opt >= ki_barrier, 1.0, 0.0)
                    ki_ind = 1.0 - ki_up
                put_int = put_int * ki_ind
            if HAS_ONESTAR:
                put_int = put_int * (1.0 - star_ind)
            unit_put = al_next * put_int
        else:
            unit_put = w_max * 0.0

        if HAS_PCALL:
            unit_call = al_next * tl.maximum(w_opt - pcall_strike, 0.0)
        else:
            unit_call = w_max * 0.0

        opt_payoff = df_term * (-option_gearing * unit_put + pcall_rate * unit_call)
        if HAS_OS_CPN:
            onestar_paths = al_next * star_ind * one_star_coupon
            opt_payoff = opt_payoff + df_term * onestar_paths
        else:
            onestar_paths = w_max * 0.0
        pstar_paths = (al_next * star_ind) if HAS_ONESTAR else (w_max * 0.0)
        return opt_payoff, unit_put, unit_call, onestar_paths, pstar_paths

    @triton.jit  # type: ignore[untyped-decorator]
    def _autocall_payoff_period_tile_kernel(  # type: ignore[no-untyped-def]
        ex_buf_ptr,             # (R, D, N_rep) float64
        tile_ex_sum_ptr,        # (R, D, n_tiles) float64
        fwd_perf_ptr,           # (D, M) float64
        df_pay_ptr,             # (M,) float64
        ko_barriers_ptr,        # (M,) float64
        is_callable_ptr,        # (M,) float64
        cpn_barriers_ptr,       # (M,) float64
        is_guar_ptr,            # (M,) float64
        cpn_amt_ptr,            # (M,) float64
        w_fund_ptr,             # (M,) float64
        inv_lex_perm_ptr,       # (D,) int64
        alive_ptr,              # (R, N_rep) float64
        unpaid_ptr,             # (R, N_rep) float64
        total_pv_ptr,           # (R, N_rep) float64
        not_ki_disc_ptr,        # (R, N_rep) float64
        z_flat_cum_ptr,         # (R, N_rep) float64
        z_coarse_corr_ptr,      # (R, M, D, N_rep) float32 when HAS_MF_CV
        d_var_gbm_ptr,          # (D, M) float64 when HAS_MF_CV
        lx_g_ptr,               # (R, D, N_rep) float64 when HAS_MF_CV
        perf_g_ptr,             # (R, D, N_rep) float64 when HAS_MF_CV & NEED_DISP
        alive_g_ptr,            # (R, N_rep) float64 when HAS_MF_CV
        unpaid_g_ptr,           # (R, N_rep) float64 when HAS_MF_CV
        pv_g_ptr,               # (R, N_rep) float64 when HAS_MF_CV
        not_ki_g_ptr,           # (R, N_rep) float64 when HAS_MF_CV
        period_tile_stats_ptr,  # (M, R, n_tiles, 4) float64
        term_tile_stats_ptr,    # (R, n_tiles, 11) float64
        period_idx,
        m_rem,
        eff_r,
        n_rep,
        n_tiles,
        inv_n_rep,              # float64
        c_mem_0,                # float64
        ac_cpn,                 # float64
        ko_spread,              # float64
        cpn_spread,             # float64
        strike,                 # float64
        max_spread_put,         # float64
        option_gearing,         # float64
        ki_barrier,             # float64
        ki_spread,              # float64
        one_star_strike,        # float64
        one_star_eps,           # float64
        one_star_coupon,        # float64
        pcall_strike,           # float64
        pcall_rate,             # float64
        f_dom,                  # float64
        v_dom_term,             # float64
        k_cv,                   # float64
        k_low,                  # float64
        D_ASSETS: tl.constexpr,
        CALL_CODE: tl.constexpr,
        CPN_CODE: tl.constexpr,
        OPT_CODE: tl.constexpr,
        IS_MEM: tl.constexpr,
        NO_NOTIONAL: tl.constexpr,
        FORCE_CPN_AC: tl.constexpr,
        CPN_DIR_DOWN: tl.constexpr,
        CPN_TYPE_OUT: tl.constexpr,
        HAS_KO_SPREAD: tl.constexpr,
        HAS_CPN_SPREAD: tl.constexpr,
        HAS_KI: tl.constexpr,
        HAS_KI_SPREAD: tl.constexpr,
        IS_DISCRETE_KI: tl.constexpr,
        HAS_PUT: tl.constexpr,
        HAS_LOW_K: tl.constexpr,
        HAS_ONESTAR: tl.constexpr,
        HAS_OS_CPN: tl.constexpr,
        HAS_PCALL: tl.constexpr,
        IS_LONG_BASKET: tl.constexpr,
        HAS_CV: tl.constexpr,
        HAS_CV_LOW: tl.constexpr,
        HAS_MF_CV: tl.constexpr,
        HAS_INV_PERM: tl.constexpr,
        NEED_DISP: tl.constexpr,
        NEED_ORDER: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused monitoring index + contractual autocall period + terminal option & CV tile."""
        pid_n = tl.program_id(0)
        pid_r = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep

        if HAS_INV_PERM:
            lex0 = tl.load(inv_lex_perm_ptr)
            lex1 = tl.load(inv_lex_perm_ptr + 1) if D_ASSETS >= 2 else 0
        else:
            lex0 = 0
            lex1 = 1 if D_ASSETS >= 2 else 0

        w_min = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        w_max = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)
        w_sum = tl.zeros((BLOCK_N,), dtype=tl.float64)
        p_lex0 = tl.zeros((BLOCK_N,), dtype=tl.float64)
        p_lex1 = tl.zeros((BLOCK_N,), dtype=tl.float64)
        min2 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        min3 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        max2 = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)

        if HAS_MF_CV:
            wg_min = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            wg_max = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)
            wg_sum = tl.zeros((BLOCK_N,), dtype=tl.float64)
            pg_lex0 = tl.zeros((BLOCK_N,), dtype=tl.float64)
            pg_lex1 = tl.zeros((BLOCK_N,), dtype=tl.float64)
            ming2 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            ming3 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            maxg2 = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)

        for d in tl.static_range(D_ASSETS):
            ex_sum = tl.zeros((), dtype=tl.float64)
            tes_base = (pid_r * D_ASSETS + d) * n_tiles
            for t in range(n_tiles):
                ex_sum = ex_sum + tl.load(tile_ex_sum_ptr + tes_base + t)
            fwd_d = tl.load(fwd_perf_ptr + d * m_rem + period_idx)
            scale_d = fwd_d / tl.maximum(ex_sum * inv_n_rep, 1e-12)
            rd_off = (pid_r * D_ASSETS + d) * n_rep + offs_n
            ex_d = tl.load(ex_buf_ptr + rd_off, mask=mask_n, other=0.0)
            p_d = ex_d * scale_d
            if NEED_ORDER:
                c1 = p_d < w_min
                c2 = (~c1) & (p_d < min2)
                c3 = (~c1) & (~c2) & (p_d < min3)
                min3 = tl.where(c1, min2, tl.where(c2, min2, tl.where(c3, p_d, min3)))
                min2 = tl.where(c1, w_min, tl.where(c2, p_d, min2))
                g1 = p_d > w_max
                g2 = (~g1) & (p_d > max2)
                max2 = tl.where(g1, w_max, tl.where(g2, p_d, max2))
            w_min = tl.minimum(w_min, p_d)
            w_max = tl.maximum(w_max, p_d)
            w_sum = w_sum + p_d
            p_lex0 = tl.where(d == lex0, p_d, p_lex0)
            p_lex1 = tl.where(d == lex1, p_d, p_lex1)

            if HAS_MF_CV:
                if period_idx == 0:
                    lxg_prev = tl.zeros((BLOCK_N,), dtype=tl.float64)
                else:
                    lxg_prev = tl.load(lx_g_ptr + rd_off, mask=mask_n, other=0.0)
                zc_d = tl.load(
                    z_coarse_corr_ptr + ((pid_r * m_rem + period_idx) * D_ASSETS + d) * n_rep + offs_n,
                    mask=mask_n,
                    other=0.0,
                )
                dv_g = tl.load(d_var_gbm_ptr + d * m_rem + period_idx)
                lxg_curr = lxg_prev - 0.5 * dv_g + tl.sqrt(dv_g) * zc_d.to(tl.float64)
                tl.store(lx_g_ptr + rd_off, lxg_curr, mask=mask_n)
                pg_d = fwd_d * tl.exp(lxg_curr)
                if NEED_DISP and D_ASSETS > 1:
                    tl.store(perf_g_ptr + rd_off, pg_d, mask=mask_n)
                if NEED_ORDER:
                    cg1 = pg_d < wg_min
                    cg2 = (~cg1) & (pg_d < ming2)
                    cg3 = (~cg1) & (~cg2) & (pg_d < ming3)
                    ming3 = tl.where(cg1, ming2, tl.where(cg2, ming2, tl.where(cg3, pg_d, ming3)))
                    ming2 = tl.where(cg1, wg_min, tl.where(cg2, pg_d, ming2))
                    gg1 = pg_d > wg_max
                    gg2 = (~gg1) & (pg_d > maxg2)
                    maxg2 = tl.where(gg1, wg_max, tl.where(gg2, pg_d, maxg2))
                wg_min = tl.minimum(wg_min, pg_d)
                wg_max = tl.maximum(wg_max, pg_d)
                wg_sum = wg_sum + pg_d
                pg_lex0 = tl.where(d == lex0, pg_d, pg_lex0)
                pg_lex1 = tl.where(d == lex1, pg_d, pg_lex1)

        w_disp = tl.zeros((BLOCK_N,), dtype=tl.float64)
        wg_disp = tl.zeros((BLOCK_N,), dtype=tl.float64)
        if NEED_DISP and D_ASSETS > 1:
            w_mean = w_sum * (1.0 / float(D_ASSETS))
            for d in tl.static_range(D_ASSETS):
                ex_sum = tl.zeros((), dtype=tl.float64)
                tes_base = (pid_r * D_ASSETS + d) * n_tiles
                for t in range(n_tiles):
                    ex_sum = ex_sum + tl.load(tile_ex_sum_ptr + tes_base + t)
                fwd_d = tl.load(fwd_perf_ptr + d * m_rem + period_idx)
                scale_d = fwd_d / tl.maximum(ex_sum * inv_n_rep, 1e-12)
                ex_d = tl.load(ex_buf_ptr + (pid_r * D_ASSETS + d) * n_rep + offs_n, mask=mask_n, other=0.0)
                diff = ex_d * scale_d - w_mean
                w_disp = w_disp + diff * diff
            if HAS_MF_CV:
                wg_mean = wg_sum * (1.0 / float(D_ASSETS))
                for d in tl.static_range(D_ASSETS):
                    pg_d = tl.load(perf_g_ptr + (pid_r * D_ASSETS + d) * n_rep + offs_n, mask=mask_n, other=0.0)
                    diffg = pg_d - wg_mean
                    wg_disp = wg_disp + diffg * diffg

        w_call = _eval_index_from_stats_triton(
            CALL_CODE, D_ASSETS, w_min, w_max, w_sum, w_disp, p_lex0, p_lex1, min2, min3, max2
        )
        w_cpn = (
            w_call
            if CPN_CODE == CALL_CODE
            else _eval_index_from_stats_triton(
                CPN_CODE, D_ASSETS, w_min, w_max, w_sum, w_disp, p_lex0, p_lex1, min2, min3, max2
            )
        )
        w_opt = (
            w_call
            if OPT_CODE == CALL_CODE
            else (
                w_cpn
                if OPT_CODE == CPN_CODE
                else _eval_index_from_stats_triton(
                    OPT_CODE, D_ASSETS, w_min, w_max, w_sum, w_disp, p_lex0, p_lex1, min2, min3, max2
                )
            )
        )

        is_call = tl.load(is_callable_ptr + period_idx) > 0.5
        b_ko_m = tl.load(ko_barriers_ptr + period_idx)
        is_guar = tl.load(is_guar_ptr + period_idx) > 0.5
        b_cpn_m = tl.load(cpn_barriers_ptr + period_idx)
        df_m = tl.load(df_pay_ptr + period_idx)
        c_m = tl.load(cpn_amt_ptr + period_idx)
        wf_m = tl.load(w_fund_ptr + period_idx)

        rn_off = pid_r * n_rep + offs_n
        if period_idx == 0:
            al = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
            unp = tl.full((BLOCK_N,), c_mem_0, dtype=tl.float64)
            tpv = tl.zeros((BLOCK_N,), dtype=tl.float64)
            not_ki = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
        else:
            al = tl.load(alive_ptr + rn_off, mask=mask_n, other=0.0)
            unp = tl.load(unpaid_ptr + rn_off, mask=mask_n, other=0.0)
            tpv = tl.load(total_pv_ptr + rn_off, mask=mask_n, other=0.0)
            not_ki = (
                tl.load(not_ki_disc_ptr + rn_off, mask=mask_n, other=1.0)
                if IS_DISCRETE_KI
                else tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
            )

        w_ki = w_min if IS_LONG_BASKET else w_opt
        al_next, unp_next, tpv, not_ki, stop_m, cpn_al_m, cpn_undisc_m = _mf_period_step_triton(
            w_call,
            w_cpn,
            w_ki,
            al,
            unp,
            tpv,
            not_ki,
            is_call,
            b_ko_m,
            is_guar,
            b_cpn_m,
            df_m,
            c_m,
            wf_m,
            period_idx,
            m_rem,
            ac_cpn,
            ko_spread,
            cpn_spread,
            ki_barrier,
            ki_spread,
            IS_MEM,
            NO_NOTIONAL,
            FORCE_CPN_AC,
            CPN_DIR_DOWN,
            CPN_TYPE_OUT,
            HAS_KO_SPREAD,
            HAS_CPN_SPREAD,
            HAS_KI,
            HAS_KI_SPREAD,
            IS_DISCRETE_KI,
        )

        if HAS_MF_CV:
            wg_call = _eval_index_from_stats_triton(
                CALL_CODE, D_ASSETS, wg_min, wg_max, wg_sum, wg_disp, pg_lex0, pg_lex1, ming2, ming3, maxg2
            )
            wg_cpn = (
                wg_call
                if CPN_CODE == CALL_CODE
                else _eval_index_from_stats_triton(
                    CPN_CODE, D_ASSETS, wg_min, wg_max, wg_sum, wg_disp, pg_lex0, pg_lex1, ming2, ming3, maxg2
                )
            )
            wg_opt = (
                wg_call
                if OPT_CODE == CALL_CODE
                else (
                    wg_cpn
                    if OPT_CODE == CPN_CODE
                    else _eval_index_from_stats_triton(
                        OPT_CODE, D_ASSETS, wg_min, wg_max, wg_sum, wg_disp, pg_lex0, pg_lex1, ming2, ming3, maxg2
                    )
                )
            )
            if period_idx == 0:
                al_g = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
                unp_g = tl.full((BLOCK_N,), c_mem_0, dtype=tl.float64)
                tpv_g = tl.zeros((BLOCK_N,), dtype=tl.float64)
                not_ki_g = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
            else:
                al_g = tl.load(alive_g_ptr + rn_off, mask=mask_n, other=0.0)
                unp_g = tl.load(unpaid_g_ptr + rn_off, mask=mask_n, other=0.0)
                tpv_g = tl.load(pv_g_ptr + rn_off, mask=mask_n, other=0.0)
                not_ki_g = (
                    tl.load(not_ki_g_ptr + rn_off, mask=mask_n, other=1.0)
                    if IS_DISCRETE_KI
                    else tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
                )
            wg_ki = wg_min if IS_LONG_BASKET else wg_opt
            al_g_next, unp_g_next, tpv_g, not_ki_g, _, _, _ = _mf_period_step_triton(
                wg_call,
                wg_cpn,
                wg_ki,
                al_g,
                unp_g,
                tpv_g,
                not_ki_g,
                is_call,
                b_ko_m,
                is_guar,
                b_cpn_m,
                df_m,
                c_m,
                wf_m,
                period_idx,
                m_rem,
                ac_cpn,
                ko_spread,
                cpn_spread,
                ki_barrier,
                ki_spread,
                IS_MEM,
                NO_NOTIONAL,
                FORCE_CPN_AC,
                CPN_DIR_DOWN,
                CPN_TYPE_OUT,
                HAS_KO_SPREAD,
                HAS_CPN_SPREAD,
                HAS_KI,
                HAS_KI_SPREAD,
                IS_DISCRETE_KI,
            )

        p_base = ((period_idx * eff_r + pid_r) * n_tiles + pid_n) * 4
        tl.store(period_tile_stats_ptr + p_base + 0, tl.sum(tl.where(mask_n, stop_m, 0.0), axis=0))
        tl.store(period_tile_stats_ptr + p_base + 1, tl.sum(tl.where(mask_n, cpn_al_m, 0.0), axis=0))
        tl.store(period_tile_stats_ptr + p_base + 2, tl.sum(tl.where(mask_n, al, 0.0), axis=0))
        tl.store(period_tile_stats_ptr + p_base + 3, tl.sum(tl.where(mask_n, cpn_undisc_m, 0.0), axis=0))

        if period_idx < m_rem - 1:
            tl.store(alive_ptr + rn_off, al_next, mask=mask_n)
            if IS_MEM:
                tl.store(unpaid_ptr + rn_off, unp_next, mask=mask_n)
            tl.store(total_pv_ptr + rn_off, tpv, mask=mask_n)
            if IS_DISCRETE_KI:
                tl.store(not_ki_disc_ptr + rn_off, not_ki, mask=mask_n)
            if HAS_MF_CV:
                tl.store(alive_g_ptr + rn_off, al_g_next, mask=mask_n)
                if IS_MEM:
                    tl.store(unpaid_g_ptr + rn_off, unp_g_next, mask=mask_n)
                tl.store(pv_g_ptr + rn_off, tpv_g, mask=mask_n)
                if IS_DISCRETE_KI:
                    tl.store(not_ki_g_ptr + rn_off, not_ki_g, mask=mask_n)
        else:
            w_avg_term = w_sum * (1.0 / float(D_ASSETS))
            opt_payoff, unit_put, unit_call, onestar_paths, pstar_paths = _mf_terminal_opt_triton(
                w_min,
                w_max,
                w_avg_term,
                w_opt,
                al_next,
                not_ki,
                df_m,
                strike,
                max_spread_put,
                option_gearing,
                ki_barrier,
                ki_spread,
                one_star_strike,
                one_star_eps,
                one_star_coupon,
                pcall_strike,
                pcall_rate,
                HAS_KI,
                HAS_KI_SPREAD,
                IS_DISCRETE_KI,
                HAS_PUT,
                HAS_LOW_K,
                HAS_ONESTAR,
                HAS_OS_CPN,
                HAS_PCALL,
                IS_LONG_BASKET,
            )
            y = tpv + opt_payoff

            if HAS_MF_CV:
                wg_avg_term = wg_sum * (1.0 / float(D_ASSETS))
                opt_g, _, _, _, _ = _mf_terminal_opt_triton(
                    wg_min,
                    wg_max,
                    wg_avg_term,
                    wg_opt,
                    al_g_next,
                    not_ki_g,
                    df_m,
                    strike,
                    max_spread_put,
                    option_gearing,
                    ki_barrier,
                    ki_spread,
                    one_star_strike,
                    one_star_eps,
                    one_star_coupon,
                    pcall_strike,
                    pcall_rate,
                    HAS_KI,
                    HAS_KI_SPREAD,
                    IS_DISCRETE_KI,
                    HAS_PUT,
                    HAS_LOW_K,
                    HAS_ONESTAR,
                    HAS_OS_CPN,
                    HAS_PCALL,
                    IS_LONG_BASKET,
                )
                x = tpv_g + opt_g
            elif HAS_CV:
                zf = tl.load(z_flat_cum_ptr + rn_off, mask=mask_n, other=0.0)
                p_dom = f_dom * tl.exp(-0.5 * v_dom_term * v_dom_term + zf)
                x = df_m * tl.maximum(k_cv - p_dom, 0.0)
                if HAS_CV_LOW:
                    x = x - df_m * tl.maximum(k_low - p_dom, 0.0)
            else:
                x = tl.zeros((BLOCK_N,), dtype=tl.float64)

            y_m = tl.where(mask_n, y, 0.0)
            o_m = tl.where(mask_n, opt_payoff, 0.0)
            x_m = tl.where(mask_n, x, 0.0)

            t_base = (pid_r * n_tiles + pid_n) * 11
            tl.store(term_tile_stats_ptr + t_base + 0, tl.sum(tl.where(mask_n, unit_put, 0.0), axis=0))
            tl.store(term_tile_stats_ptr + t_base + 1, tl.sum(tl.where(mask_n, unit_call, 0.0), axis=0))
            tl.store(
                term_tile_stats_ptr + t_base + 2, tl.sum(tl.where(mask_n, onestar_paths, 0.0), axis=0)
            )
            tl.store(term_tile_stats_ptr + t_base + 3, tl.sum(tl.where(mask_n, pstar_paths, 0.0), axis=0))
            tl.store(term_tile_stats_ptr + t_base + 4, tl.sum(y_m, axis=0))
            tl.store(term_tile_stats_ptr + t_base + 5, tl.sum(y_m * y_m, axis=0))
            tl.store(term_tile_stats_ptr + t_base + 6, tl.sum(o_m, axis=0))
            tl.store(term_tile_stats_ptr + t_base + 7, tl.sum(o_m * o_m, axis=0))
            tl.store(term_tile_stats_ptr + t_base + 8, tl.sum(x_m, axis=0))
            tl.store(term_tile_stats_ptr + t_base + 9, tl.sum(x_m * x_m, axis=0))
            tl.store(term_tile_stats_ptr + t_base + 10, tl.sum(x_m * y_m, axis=0))

    @triton.jit  # type: ignore[untyped-decorator]
    def _sobol_pca_triton_kernel(  # type: ignore[no-untyped-def]
        sv_ptr,        # (21201, 30) int32
        shifts_ptr,    # (M, R, D) int32
        a_pca_ptr,     # (D, D) float32
        out_ptr,       # (R, M, D, N_rep) float32
        m_rem,
        eff_r,
        n_rep,
        m_pow,
        D_ASSETS: tl.constexpr,
        LOW_BITS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Generates PCA-correlated coarse Sobol normals (R, M, D, N_rep) at full SM occupancy."""
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_r = tl.program_id(2)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep
        offs_u32 = offs_n.to(tl.uint32)
        pid_u32 = pid_n.to(tl.uint32)

        for d in range(D_ASSETS):
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k in range(D_ASSETS):
                sh_k = tl.load(shifts_ptr + (pid_m * eff_r + pid_r) * D_ASSETS + k).to(tl.uint32)
                xi_k = _sobol_vec(
                    sv_ptr, pid_m * D_ASSETS + k, sh_k, pid_u32, offs_u32, m_pow, LOW_BITS, BLOCK_N
                )
                acc = acc + tl.load(a_pca_ptr + d * D_ASSETS + k) * xi_k
            out_off = ((pid_r * m_rem + pid_m) * D_ASSETS + d) * n_rep + offs_n
            tl.store(out_ptr + out_off, acc, mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _dupire_persistent_qmc_kernel(  # type: ignore[no-untyped-def]
        sv_ptr,               # (21201, 30) int32
        shifts_s_ptr,         # (M, R, D * n_sub) int32
        z_coarse_corr_ptr,    # (R, M, D, N_rep) float32
        chol_ptr,             # (D, D) float32
        lv_table_ptr,         # (D, N_t, N_y) float32
        y_min_ptr,            # (D,) float32
        inv_dy_ptr,           # (D,) float32
        q_shift_bnd_ptr,      # (M, D, 2) float32
        t_prev_ptr,           # (M,) float32
        dt_sub_ptr,           # (M,) float32
        sqrt_dt_sub_ptr,      # (M,) float32
        fwd_perf_ptr,         # (D, M) float64
        df_pay_ptr,           # (M,) float64
        ko_barriers_ptr,      # (M,) float64
        is_callable_ptr,      # (M,) float64
        cpn_barriers_ptr,     # (M,) float64
        is_guar_ptr,          # (M,) float64
        cpn_amt_ptr,          # (M,) float64
        w_fund_ptr,           # (M,) float64
        d_var_gbm_ptr,        # (D, M) float64
        sdvar_dom_ptr,        # (M,) float64
        inv_lex_perm_ptr,     # (D,) int64
        ws_lx_ptr,            # (R, D, N_rep) float32
        ws_lx_g_ptr,          # (R, D, N_rep) float64
        ws_zsub_ptr,          # (R, D, n_sub, N_rep) float32
        ws_haar_ptr,          # (R, n_sub, N_rep) float32 when IS_HAAR else stub
        ws_perf_ptr,          # (R, D, N_rep) float64
        ws_perf_g_ptr,        # (R, D, N_rep) float64
        period_stats_ptr,     # (M, R, 1, 4) float64
        term_stats_ptr,       # (R, 1, 11) float64
        m_rem,
        eff_r,
        n_rep,
        n_sub,
        haar_levels,
        m_pow,
        sub_dim_base,
        dom_idx,
        inv_dt_grid,          # float32
        inv_n_rep,            # float64
        inv_sqrt_sub,         # float32
        inv_n_sub,            # float32
        c_mem_0,              # float64
        ac_cpn,               # float64
        ko_spread,            # float64
        cpn_spread,           # float64
        strike,               # float64
        max_spread_put,       # float64
        option_gearing,       # float64
        ki_barrier,           # float64
        ki_spread,            # float64
        one_star_strike,      # float64
        one_star_eps,         # float64
        one_star_coupon,      # float64
        pcall_strike,         # float64
        pcall_rate,           # float64
        f_dom,                # float64
        v_dom_term,           # float64
        k_cv,                 # float64
        k_low,                # float64
        D_ASSETS: tl.constexpr,
        CALL_CODE: tl.constexpr,
        CPN_CODE: tl.constexpr,
        OPT_CODE: tl.constexpr,
        IS_MEM: tl.constexpr,
        NO_NOTIONAL: tl.constexpr,
        FORCE_CPN_AC: tl.constexpr,
        CPN_DIR_DOWN: tl.constexpr,
        CPN_TYPE_OUT: tl.constexpr,
        HAS_KO_SPREAD: tl.constexpr,
        HAS_CPN_SPREAD: tl.constexpr,
        HAS_KI: tl.constexpr,
        HAS_KI_SPREAD: tl.constexpr,
        IS_DISCRETE_KI: tl.constexpr,
        HAS_PUT: tl.constexpr,
        HAS_LOW_K: tl.constexpr,
        HAS_ONESTAR: tl.constexpr,
        HAS_OS_CPN: tl.constexpr,
        HAS_PCALL: tl.constexpr,
        IS_LONG_BASKET: tl.constexpr,
        HAS_CV: tl.constexpr,
        HAS_CV_LOW: tl.constexpr,
        HAS_MF_CV: tl.constexpr,
        HAS_SUB: tl.constexpr,
        IS_HAAR: tl.constexpr,
        HAS_INV_PERM: tl.constexpr,
        NEED_DISP: tl.constexpr,
        NEED_ORDER: tl.constexpr,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Persistent single-kernel multi-period Dupire Platen Weak-2.0 + EMS + MF_CV solver."""
        pid_r = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep
        offs_u32 = offs_n.to(tl.uint32)

        if HAS_INV_PERM:
            lex0 = tl.load(inv_lex_perm_ptr)
            lex1 = tl.load(inv_lex_perm_ptr + 1) if D_ASSETS >= 2 else 0
        else:
            lex0 = 0
            lex1 = 1 if D_ASSETS >= 2 else 0

        al = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
        unp = tl.full((BLOCK_N,), c_mem_0, dtype=tl.float64)
        tpv = tl.zeros((BLOCK_N,), dtype=tl.float64)
        not_ki = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
        zf = tl.zeros((BLOCK_N,), dtype=tl.float64)

        al_g = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
        unp_g = tl.full((BLOCK_N,), c_mem_0, dtype=tl.float64)
        tpv_g = tl.zeros((BLOCK_N,), dtype=tl.float64)
        not_ki_g = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)

        w_min = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        w_max = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)
        w_sum = tl.zeros((BLOCK_N,), dtype=tl.float64)
        w_disp = tl.zeros((BLOCK_N,), dtype=tl.float64)
        p_lex0 = tl.zeros((BLOCK_N,), dtype=tl.float64)
        p_lex1 = tl.zeros((BLOCK_N,), dtype=tl.float64)
        min2 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        min3 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        max2 = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)

        wg_min = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        wg_max = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)
        wg_sum = tl.zeros((BLOCK_N,), dtype=tl.float64)
        wg_disp = tl.zeros((BLOCK_N,), dtype=tl.float64)
        pg_lex0 = tl.zeros((BLOCK_N,), dtype=tl.float64)
        pg_lex1 = tl.zeros((BLOCK_N,), dtype=tl.float64)
        ming2 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        ming3 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        maxg2 = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)

        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4

        for m in range(m_rem):
            t_start = tl.load(t_prev_ptr + m)
            dt_sub = tl.load(dt_sub_ptr + m)
            sqrt_dt = tl.load(sqrt_dt_sub_ptr + m)

            if HAS_SUB:
                sh_m_base = (m * eff_r + pid_r) * (D_ASSETS * n_sub)
                if IS_HAAR:
                    one_i32 = tl.zeros((), dtype=tl.int32) + 1
                    haar_r_base = pid_r * n_sub
                    for d_src in range(D_ASSETS):
                        rd_sub = (pid_r * D_ASSETS + d_src) * n_sub
                        d_sh_base = sh_m_base + d_src * n_sub
                        for lvl in range(haar_levels):
                            k_pow = one_i32 << lvl
                            dim_blk = k_pow * sub_dim_base + (m * D_ASSETS + d_src) * k_pow
                            mode_base = k_pow - 1
                            for k in range(k_pow):
                                mode = mode_base + k
                                sh_s = tl.load(shifts_s_ptr + d_sh_base + mode).to(tl.uint32)
                                b_mode = _sobol_vec_flat(sv_ptr, dim_blk + k, sh_s, offs_u32, m_pow, BLOCK_N)
                                tl.store(
                                    ws_haar_ptr + (haar_r_base + mode) * n_rep + offs_n,
                                    b_mode,
                                    mask=mask_n,
                                )
                        for s in range(n_sub):
                            s_i32 = tl.zeros((), dtype=tl.int32) + s
                            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                            for lvl in range(haar_levels):
                                k_pow = one_i32 << lvl
                                shift_b = haar_levels - 1 - lvl
                                bit = (s_i32 >> shift_b) & 1
                                mode = (k_pow - 1) + (s_i32 >> (shift_b + 1))
                                b_mode = tl.load(
                                    ws_haar_ptr + (haar_r_base + mode) * n_rep + offs_n,
                                    mask=mask_n,
                                    other=0.0,
                                )
                                w_lvl = tl.sqrt(k_pow.to(tl.float32))
                                acc = acc + tl.where(bit == 0, w_lvl, -w_lvl) * b_mode
                            tl.store(ws_zsub_ptr + (rd_sub + s) * n_rep + offs_n, acc * inv_sqrt_sub, mask=mask_n)
                else:
                    dim_m_base = sub_dim_base + m * (D_ASSETS * n_sub)
                    for d_src in range(D_ASSETS):
                        rd_sub = (pid_r * D_ASSETS + d_src) * n_sub
                        sum_z = tl.zeros((BLOCK_N,), dtype=tl.float32)
                        for s in range(n_sub):
                            sh_s = tl.load(shifts_s_ptr + sh_m_base + d_src * n_sub + s).to(tl.uint32)
                            zs = _sobol_vec_flat(
                                sv_ptr, dim_m_base + d_src * n_sub + s, sh_s, offs_u32, m_pow, BLOCK_N
                            )
                            sum_z = sum_z + zs
                            tl.store(ws_zsub_ptr + (rd_sub + s) * n_rep + offs_n, zs, mask=mask_n)
                        z_mean_s = sum_z * inv_n_sub
                        for s in range(n_sub):
                            s_off = (rd_sub + s) * n_rep + offs_n
                            zs = tl.load(ws_zsub_ptr + s_off, mask=mask_n, other=0.0)
                            tl.store(ws_zsub_ptr + s_off, zs - z_mean_s, mask=mask_n)

            w_min = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            w_max = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)
            w_sum = tl.zeros((BLOCK_N,), dtype=tl.float64)
            w_disp = tl.zeros((BLOCK_N,), dtype=tl.float64)
            p_lex0 = tl.zeros((BLOCK_N,), dtype=tl.float64)
            p_lex1 = tl.zeros((BLOCK_N,), dtype=tl.float64)
            min2 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            min3 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            max2 = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)

            if HAS_MF_CV:
                wg_min = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
                wg_max = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)
                wg_sum = tl.zeros((BLOCK_N,), dtype=tl.float64)
                wg_disp = tl.zeros((BLOCK_N,), dtype=tl.float64)
                pg_lex0 = tl.zeros((BLOCK_N,), dtype=tl.float64)
                pg_lex1 = tl.zeros((BLOCK_N,), dtype=tl.float64)
                ming2 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
                ming3 = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
                maxg2 = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)

            for d in tl.static_range(D_ASSETS):
                rd_off = (pid_r * D_ASSETS + d) * n_rep + offs_n
                if m == 0:
                    lx = tl.zeros((BLOCK_N,), dtype=tl.float32)
                else:
                    lx = tl.load(ws_lx_ptr + rd_off, mask=mask_n, other=0.0)

                zc_off = ((pid_r * m_rem + m) * D_ASSETS + d) * n_rep + offs_n
                zc_d = tl.load(z_coarse_corr_ptr + zc_off, mask=mask_n, other=0.0)

                if HAS_CV and (d == dom_idx):
                    sdvar_m = tl.load(sdvar_dom_ptr + m)
                    zf = zf + sdvar_m * zc_d.to(tl.float64)

                y_min = tl.load(y_min_ptr + d)
                inv_dy = tl.load(inv_dy_ptr + d)
                q0 = tl.load(q_shift_bnd_ptr + (m * D_ASSETS + d) * 2 + 0)
                q1 = tl.load(q_shift_bnd_ptr + (m * D_ASSETS + d) * 2 + 1)
                lv_base = d * (N_t * N_y)

                for s in range(n_sub):
                    s_f = tl.zeros((), dtype=tl.float32) + s
                    t_s0 = t_start + s_f * dt_sub
                    t_s1 = t_start + (s_f + 1.0) * dt_sub
                    q_sh0 = q0 + (s_f * inv_n_sub) * (q1 - q0)
                    q_sh1 = q0 + ((s_f + 1.0) * inv_n_sub) * (q1 - q0)

                    lv0 = _sample_lv_grid_triton(
                        lv_table_ptr, lv_base, t_s0, lx + q_sh0, y_min, inv_dy, inv_dt_grid, max_t_f, max_y_f, N_t, N_y, mask_n
                    )
                    mu0 = -0.5 * lv0 * lv0
                    y_h = lx + mu0 * dt_sub
                    sig_step = lv0 * sqrt_dt
                    lv_p = _sample_lv_grid_triton(
                        lv_table_ptr, lv_base, t_s1, y_h + sig_step + q_sh1, y_min, inv_dy, inv_dt_grid, max_t_f, max_y_f, N_t, N_y, mask_n
                    )
                    lv_m = _sample_lv_grid_triton(
                        lv_table_ptr, lv_base, t_s1, y_h - sig_step + q_sh1, y_min, inv_dy, inv_dt_grid, max_t_f, max_y_f, N_t, N_y, mask_n
                    )
                    mu_p = -0.5 * lv_p * lv_p
                    mu_m = -0.5 * lv_m * lv_m

                    if HAS_SUB:
                        z_corr = zc_d * inv_sqrt_sub
                        for j in range(d + 1):
                            l_dj = tl.load(chol_ptr + d * D_ASSETS + j)
                            zj = tl.load(
                                ws_zsub_ptr + ((pid_r * D_ASSETS + j) * n_sub + s) * n_rep + offs_n,
                                mask=mask_n,
                                other=0.0,
                            )
                            z_corr = z_corr + l_dj * zj
                    else:
                        z_corr = zc_d

                    lx = lx + (
                        0.5 * (mu0 + 0.5 * (mu_p + mu_m)) * dt_sub
                        + 0.25 * (mu_p - mu_m) * dt_sub * z_corr
                        + 0.25 * (lv_p + lv_m + 2.0 * lv0) * sqrt_dt * z_corr
                        + 0.25 * (lv_p - lv_m) * (z_corr * z_corr - 1.0) * sqrt_dt
                    )

                lx_f64 = lx.to(tl.float64)
                ex64 = tl.where(mask_n, tl.exp(lx_f64), 0.0)
                mean_x = tl.sum(ex64, axis=0) * inv_n_rep
                scale_x = tl.maximum(mean_x, 1e-12)
                lx_norm = (lx_f64 - tl.log(scale_x)).to(tl.float32)
                tl.store(ws_lx_ptr + rd_off, lx_norm, mask=mask_n)

                fwd_d = tl.load(fwd_perf_ptr + d * m_rem + m)
                p_d = fwd_d * (ex64 / scale_x)
                if NEED_DISP and D_ASSETS > 1:
                    tl.store(ws_perf_ptr + rd_off, p_d, mask=mask_n)
                if NEED_ORDER:
                    c1 = p_d < w_min
                    c2 = (~c1) & (p_d < min2)
                    c3 = (~c1) & (~c2) & (p_d < min3)
                    min3 = tl.where(c1, min2, tl.where(c2, min2, tl.where(c3, p_d, min3)))
                    min2 = tl.where(c1, w_min, tl.where(c2, p_d, min2))
                    g1 = p_d > w_max
                    g2 = (~g1) & (p_d > max2)
                    max2 = tl.where(g1, w_max, tl.where(g2, p_d, max2))
                w_min = tl.minimum(w_min, p_d)
                w_max = tl.maximum(w_max, p_d)
                w_sum = w_sum + p_d
                p_lex0 = tl.where(d == lex0, p_d, p_lex0)
                p_lex1 = tl.where(d == lex1, p_d, p_lex1)

                if HAS_MF_CV:
                    if m == 0:
                        lxg_prev = tl.zeros((BLOCK_N,), dtype=tl.float64)
                    else:
                        lxg_prev = tl.load(ws_lx_g_ptr + rd_off, mask=mask_n, other=0.0)
                    dv_g = tl.load(d_var_gbm_ptr + d * m_rem + m)
                    lxg_curr = lxg_prev - 0.5 * dv_g + tl.sqrt(dv_g) * zc_d.to(tl.float64)
                    tl.store(ws_lx_g_ptr + rd_off, lxg_curr, mask=mask_n)
                    pg_d = fwd_d * tl.exp(lxg_curr)
                    if NEED_DISP and D_ASSETS > 1:
                        tl.store(ws_perf_g_ptr + rd_off, pg_d, mask=mask_n)
                    if NEED_ORDER:
                        cg1 = pg_d < wg_min
                        cg2 = (~cg1) & (pg_d < ming2)
                        cg3 = (~cg1) & (~cg2) & (pg_d < ming3)
                        ming3 = tl.where(cg1, ming2, tl.where(cg2, ming2, tl.where(cg3, pg_d, ming3)))
                        ming2 = tl.where(cg1, wg_min, tl.where(cg2, pg_d, ming2))
                        gg1 = pg_d > wg_max
                        gg2 = (~gg1) & (pg_d > maxg2)
                        maxg2 = tl.where(gg1, wg_max, tl.where(gg2, pg_d, maxg2))
                    wg_min = tl.minimum(wg_min, pg_d)
                    wg_max = tl.maximum(wg_max, pg_d)
                    wg_sum = wg_sum + pg_d
                    pg_lex0 = tl.where(d == lex0, pg_d, pg_lex0)
                    pg_lex1 = tl.where(d == lex1, pg_d, pg_lex1)

            if NEED_DISP and D_ASSETS > 1:
                w_mean = w_sum * (1.0 / float(D_ASSETS))
                for d in tl.static_range(D_ASSETS):
                    p_d = tl.load(ws_perf_ptr + (pid_r * D_ASSETS + d) * n_rep + offs_n, mask=mask_n, other=0.0)
                    diff = p_d - w_mean
                    w_disp = w_disp + diff * diff
                if HAS_MF_CV:
                    wg_mean = wg_sum * (1.0 / float(D_ASSETS))
                    for d in tl.static_range(D_ASSETS):
                        pg_d = tl.load(ws_perf_g_ptr + (pid_r * D_ASSETS + d) * n_rep + offs_n, mask=mask_n, other=0.0)
                        diffg = pg_d - wg_mean
                        wg_disp = wg_disp + diffg * diffg

            w_call = _eval_index_from_stats_triton(
                CALL_CODE, D_ASSETS, w_min, w_max, w_sum, w_disp, p_lex0, p_lex1, min2, min3, max2
            )
            w_cpn = (
                w_call
                if CPN_CODE == CALL_CODE
                else _eval_index_from_stats_triton(
                    CPN_CODE, D_ASSETS, w_min, w_max, w_sum, w_disp, p_lex0, p_lex1, min2, min3, max2
                )
            )
            w_opt = (
                w_call
                if OPT_CODE == CALL_CODE
                else (
                    w_cpn
                    if OPT_CODE == CPN_CODE
                    else _eval_index_from_stats_triton(
                        OPT_CODE, D_ASSETS, w_min, w_max, w_sum, w_disp, p_lex0, p_lex1, min2, min3, max2
                    )
                )
            )

            is_call = tl.load(is_callable_ptr + m) > 0.5
            b_ko_m = tl.load(ko_barriers_ptr + m)
            if HAS_KO_SPREAD:
                blo_ko = tl.minimum(b_ko_m, b_ko_m + ko_spread)
                bhi_ko = tl.maximum(b_ko_m, b_ko_m + ko_spread)
                ko_raw = tl.minimum(
                    tl.maximum((w_call - blo_ko) / tl.maximum(bhi_ko - blo_ko, 1e-9), 0.0), 1.0
                )
            else:
                ko_raw = tl.where(w_call >= b_ko_m, 1.0, 0.0)
            ko_ind = tl.where(is_call, ko_raw, 0.0)

            is_guar = tl.load(is_guar_ptr + m) > 0.5
            b_cpn_m = tl.load(cpn_barriers_ptr + m)
            eff_s_cpn = -cpn_spread if CPN_DIR_DOWN else cpn_spread
            if HAS_CPN_SPREAD:
                blo_cp = tl.minimum(b_cpn_m, b_cpn_m + eff_s_cpn)
                bhi_cp = tl.maximum(b_cpn_m, b_cpn_m + eff_s_cpn)
                cpn_ramp = tl.minimum(
                    tl.maximum((w_cpn - blo_cp) / tl.maximum(bhi_cp - blo_cp, 1e-9), 0.0), 1.0
                )
            else:
                cpn_ramp = tl.where(w_cpn >= b_cpn_m, 1.0, 0.0)
            base_cpn = (1.0 - cpn_ramp) if CPN_DIR_DOWN else cpn_ramp
            cpn_raw = (1.0 - base_cpn) if CPN_TYPE_OUT else base_cpn
            cpn_ind = tl.where(is_guar, 1.0, cpn_raw)
            if FORCE_CPN_AC:
                cpn_ind = tl.maximum(cpn_ind, ko_ind)

            if HAS_KI and IS_DISCRETE_KI:
                w_ki_m = w_min if IS_LONG_BASKET else w_opt
                if HAS_KI_SPREAD:
                    blo_ki = tl.minimum(ki_barrier, ki_barrier + ki_spread)
                    bhi_ki = tl.maximum(ki_barrier, ki_barrier + ki_spread)
                    ki_up_m = tl.minimum(
                        tl.maximum((w_ki_m - blo_ki) / tl.maximum(bhi_ki - blo_ki, 1e-9), 0.0), 1.0
                    )
                else:
                    ki_up_m = tl.where(w_ki_m >= ki_barrier, 1.0, 0.0)
                not_ki = not_ki * ki_up_m

            df_m = tl.load(df_pay_ptr + m)
            c_m = tl.load(cpn_amt_ptr + m)
            wf_m = tl.load(w_fund_ptr + m)

            tpv = tpv - al * wf_m
            stop_m = al * ko_ind
            cpn_al_m = al * cpn_ind
            eff_c = (c_m + unp) if IS_MEM else c_m
            cpn_undisc_m = cpn_al_m * eff_c + stop_m * ac_cpn
            tpv = tpv + df_m * cpn_undisc_m

            if IS_MEM:
                unp = al * (unp + c_m) * (1.0 - cpn_ind)
            al_next = al * (1.0 - ko_ind)

            if not NO_NOTIONAL:
                tpv = tpv + df_m * tl.where(m < m_rem - 1, stop_m, stop_m + al_next)

            p_base = (m * eff_r + pid_r) * 4
            tl.store(period_stats_ptr + p_base + 0, tl.sum(tl.where(mask_n, stop_m, 0.0), axis=0))
            tl.store(period_stats_ptr + p_base + 1, tl.sum(tl.where(mask_n, cpn_al_m, 0.0), axis=0))
            tl.store(period_stats_ptr + p_base + 2, tl.sum(tl.where(mask_n, al, 0.0), axis=0))
            tl.store(period_stats_ptr + p_base + 3, tl.sum(tl.where(mask_n, cpn_undisc_m, 0.0), axis=0))
            al = al_next

            if HAS_MF_CV:
                wg_call = _eval_index_from_stats_triton(
                    CALL_CODE, D_ASSETS, wg_min, wg_max, wg_sum, wg_disp, pg_lex0, pg_lex1, ming2, ming3, maxg2
                )
                wg_cpn = (
                    wg_call
                    if CPN_CODE == CALL_CODE
                    else _eval_index_from_stats_triton(
                        CPN_CODE, D_ASSETS, wg_min, wg_max, wg_sum, wg_disp, pg_lex0, pg_lex1, ming2, ming3, maxg2
                    )
                )
                wg_opt_m = (
                    wg_call
                    if OPT_CODE == CALL_CODE
                    else (
                        wg_cpn
                        if OPT_CODE == CPN_CODE
                        else _eval_index_from_stats_triton(
                            OPT_CODE, D_ASSETS, wg_min, wg_max, wg_sum, wg_disp, pg_lex0, pg_lex1, ming2, ming3, maxg2
                        )
                    )
                )
                if HAS_KO_SPREAD:
                    ko_g_raw = tl.minimum(
                        tl.maximum((wg_call - blo_ko) / tl.maximum(bhi_ko - blo_ko, 1e-9), 0.0), 1.0
                    )
                else:
                    ko_g_raw = tl.where(wg_call >= b_ko_m, 1.0, 0.0)
                ko_g = tl.where(is_call, ko_g_raw, 0.0)
                if HAS_CPN_SPREAD:
                    cpn_g_ramp = tl.minimum(
                        tl.maximum((wg_cpn - blo_cp) / tl.maximum(bhi_cp - blo_cp, 1e-9), 0.0), 1.0
                    )
                else:
                    cpn_g_ramp = tl.where(wg_cpn >= b_cpn_m, 1.0, 0.0)
                base_cpn_g = (1.0 - cpn_g_ramp) if CPN_DIR_DOWN else cpn_g_ramp
                cpn_g_raw = (1.0 - base_cpn_g) if CPN_TYPE_OUT else base_cpn_g
                cpn_g = tl.where(is_guar, 1.0, cpn_g_raw)
                if FORCE_CPN_AC:
                    cpn_g = tl.maximum(cpn_g, ko_g)
                if HAS_KI and IS_DISCRETE_KI:
                    wg_ki_m = wg_min if IS_LONG_BASKET else wg_opt_m
                    if HAS_KI_SPREAD:
                        ki_ug = tl.minimum(
                            tl.maximum((wg_ki_m - blo_ki) / tl.maximum(bhi_ki - blo_ki, 1e-9), 0.0), 1.0
                        )
                    else:
                        ki_ug = tl.where(wg_ki_m >= ki_barrier, 1.0, 0.0)
                    not_ki_g = not_ki_g * ki_ug
                tpv_g = tpv_g - al_g * wf_m
                st_g = al_g * ko_g
                eff_cg = (c_m + unp_g) if IS_MEM else c_m
                tpv_g = tpv_g + df_m * (al_g * cpn_g * eff_cg + st_g * ac_cpn)
                if IS_MEM:
                    unp_g = al_g * (unp_g + c_m) * (1.0 - cpn_g)
                al_g_next = al_g * (1.0 - ko_g)
                if not NO_NOTIONAL:
                    tpv_g = tpv_g + df_m * tl.where(m < m_rem - 1, st_g, st_g + al_g_next)
                al_g = al_g_next

        df_term = tl.load(df_pay_ptr + m_rem - 1)
        w_opt_term = _eval_index_from_stats_triton(
            OPT_CODE, D_ASSETS, w_min, w_max, w_sum, w_disp, p_lex0, p_lex1, min2, min3, max2
        )
        if HAS_ONESTAR:
            star_ind = tl.minimum(
                tl.maximum((w_max - (one_star_strike - 0.5 * one_star_eps)) / one_star_eps, 0.0), 1.0
            )
        else:
            star_ind = tl.zeros((BLOCK_N,), dtype=tl.float64)

        if IS_LONG_BASKET:
            w_avg_term = w_sum * (1.0 / float(D_ASSETS))
            if HAS_KI and IS_DISCRETE_KI:
                ki_wof_k = 1.0 - not_ki
            else:
                if HAS_KI_SPREAD:
                    blo_lb = tl.minimum(strike, strike + ki_spread)
                    bhi_lb = tl.maximum(strike, strike + ki_spread)
                    ki_up_lb = tl.minimum(
                        tl.maximum((w_min - blo_lb) / tl.maximum(bhi_lb - blo_lb, 1e-9), 0.0), 1.0
                    )
                else:
                    ki_up_lb = tl.where(w_min >= strike, 1.0, 0.0)
                ki_wof_k = 1.0 - ki_up_lb
            eff_gearing = tl.where(tl.abs(option_gearing) > 1e-12, option_gearing, 1.0)
            put_int = ((1.0 - w_avg_term) * ki_wof_k) / eff_gearing
            if HAS_ONESTAR:
                put_int = put_int * (1.0 - star_ind)
            unit_put = al * put_int
        elif HAS_PUT:
            put_int = tl.maximum(strike - w_opt_term, 0.0)
            if HAS_LOW_K:
                put_int = tl.minimum(put_int, max_spread_put)
            if HAS_KI:
                if IS_DISCRETE_KI:
                    ki_ind = 1.0 - not_ki
                else:
                    if HAS_KI_SPREAD:
                        blo_ki = tl.minimum(ki_barrier, ki_barrier + ki_spread)
                        bhi_ki = tl.maximum(ki_barrier, ki_barrier + ki_spread)
                        ki_up = tl.minimum(
                            tl.maximum((w_opt_term - blo_ki) / tl.maximum(bhi_ki - blo_ki, 1e-9), 0.0), 1.0
                        )
                    else:
                        ki_up = tl.where(w_opt_term >= ki_barrier, 1.0, 0.0)
                    ki_ind = 1.0 - ki_up
                put_int = put_int * ki_ind
            if HAS_ONESTAR:
                put_int = put_int * (1.0 - star_ind)
            unit_put = al * put_int
        else:
            unit_put = tl.zeros((BLOCK_N,), dtype=tl.float64)

        if HAS_PCALL:
            unit_call = al * tl.maximum(w_opt_term - pcall_strike, 0.0)
        else:
            unit_call = tl.zeros((BLOCK_N,), dtype=tl.float64)

        opt_payoff = df_term * (-option_gearing * unit_put + pcall_rate * unit_call)
        if HAS_OS_CPN:
            onestar_paths = al * star_ind * one_star_coupon
            opt_payoff = opt_payoff + df_term * onestar_paths
        else:
            onestar_paths = tl.zeros((BLOCK_N,), dtype=tl.float64)
        pstar_paths = (al * star_ind) if HAS_ONESTAR else tl.zeros((BLOCK_N,), dtype=tl.float64)
        y = tpv + opt_payoff

        if HAS_MF_CV:
            wg_opt_term = _eval_index_from_stats_triton(
                OPT_CODE, D_ASSETS, wg_min, wg_max, wg_sum, wg_disp, pg_lex0, pg_lex1, ming2, ming3, maxg2
            )
            if HAS_ONESTAR:
                star_g = tl.minimum(
                    tl.maximum((wg_max - (one_star_strike - 0.5 * one_star_eps)) / one_star_eps, 0.0), 1.0
                )
            else:
                star_g = tl.zeros((BLOCK_N,), dtype=tl.float64)
            if IS_LONG_BASKET:
                wg_avg_term = wg_sum * (1.0 / float(D_ASSETS))
                if HAS_KI and IS_DISCRETE_KI:
                    ki_wof_kg = 1.0 - not_ki_g
                else:
                    if HAS_KI_SPREAD:
                        blo_lb_g = tl.minimum(strike, strike + ki_spread)
                        bhi_lb_g = tl.maximum(strike, strike + ki_spread)
                        ki_up_lb_g = tl.minimum(
                            tl.maximum((wg_min - blo_lb_g) / tl.maximum(bhi_lb_g - blo_lb_g, 1e-9), 0.0), 1.0
                        )
                    else:
                        ki_up_lb_g = tl.where(wg_min >= strike, 1.0, 0.0)
                    ki_wof_kg = 1.0 - ki_up_lb_g
                eff_gearing_g = tl.where(tl.abs(option_gearing) > 1e-12, option_gearing, 1.0)
                put_g = ((1.0 - wg_avg_term) * ki_wof_kg) / eff_gearing_g
                if HAS_ONESTAR:
                    put_g = put_g * (1.0 - star_g)
                uput_g = al_g * put_g
            elif HAS_PUT:
                put_g = tl.maximum(strike - wg_opt_term, 0.0)
                if HAS_LOW_K:
                    put_g = tl.minimum(put_g, max_spread_put)
                if HAS_KI:
                    if IS_DISCRETE_KI:
                        ki_g = 1.0 - not_ki_g
                    else:
                        if HAS_KI_SPREAD:
                            blo_ki = tl.minimum(ki_barrier, ki_barrier + ki_spread)
                            bhi_ki = tl.maximum(ki_barrier, ki_barrier + ki_spread)
                            ki_upg = tl.minimum(
                                tl.maximum((wg_opt_term - blo_ki) / tl.maximum(bhi_ki - blo_ki, 1e-9), 0.0), 1.0
                            )
                        else:
                            ki_upg = tl.where(wg_opt_term >= ki_barrier, 1.0, 0.0)
                        ki_g = 1.0 - ki_upg
                    put_g = put_g * ki_g
                if HAS_ONESTAR:
                    put_g = put_g * (1.0 - star_g)
                uput_g = al_g * put_g
            else:
                uput_g = tl.zeros((BLOCK_N,), dtype=tl.float64)
            if HAS_PCALL:
                ucall_g = al_g * tl.maximum(wg_opt_term - pcall_strike, 0.0)
            else:
                ucall_g = tl.zeros((BLOCK_N,), dtype=tl.float64)
            opt_g = df_term * (-option_gearing * uput_g + pcall_rate * ucall_g)
            if HAS_OS_CPN:
                opt_g = opt_g + df_term * al_g * star_g * one_star_coupon
            x = tpv_g + opt_g
        elif HAS_CV:
            p_dom = f_dom * tl.exp(-0.5 * v_dom_term * v_dom_term + zf)
            x = df_term * tl.maximum(k_cv - p_dom, 0.0)
            if HAS_CV_LOW:
                x = x - df_term * tl.maximum(k_low - p_dom, 0.0)
        else:
            x = tl.zeros((BLOCK_N,), dtype=tl.float64)

        y_m = tl.where(mask_n, y, 0.0)
        o_m = tl.where(mask_n, opt_payoff, 0.0)
        x_m = tl.where(mask_n, x, 0.0)
        t_base = pid_r * 11
        tl.store(term_stats_ptr + t_base + 0, tl.sum(tl.where(mask_n, unit_put, 0.0), axis=0))
        tl.store(term_stats_ptr + t_base + 1, tl.sum(tl.where(mask_n, unit_call, 0.0), axis=0))
        tl.store(term_stats_ptr + t_base + 2, tl.sum(tl.where(mask_n, onestar_paths, 0.0), axis=0))
        tl.store(term_stats_ptr + t_base + 3, tl.sum(tl.where(mask_n, pstar_paths, 0.0), axis=0))
        tl.store(term_stats_ptr + t_base + 4, tl.sum(y_m, axis=0))
        tl.store(term_stats_ptr + t_base + 5, tl.sum(y_m * y_m, axis=0))
        tl.store(term_stats_ptr + t_base + 6, tl.sum(o_m, axis=0))
        tl.store(term_stats_ptr + t_base + 7, tl.sum(o_m * o_m, axis=0))
        tl.store(term_stats_ptr + t_base + 8, tl.sum(x_m, axis=0))
        tl.store(term_stats_ptr + t_base + 9, tl.sum(x_m * x_m, axis=0))
        tl.store(term_stats_ptr + t_base + 10, tl.sum(x_m * y_m, axis=0))

    @triton.jit  # type: ignore[untyped-decorator]
    def _cheap_mf_gbm_triton_kernel(  # type: ignore[no-untyped-def]
        z_corr_ptr,         # (rc, M, D, n_pairs_rep) float32
        d_var_gbm_ptr,      # (D, M) float64
        fwd_perf_ptr,       # (D, M) float64
        df_pay_ptr,         # (M,) float64
        ko_barriers_ptr,    # (M,) float64
        is_callable_ptr,    # (M,) float64
        cpn_barriers_ptr,   # (M,) float64
        is_guar_ptr,        # (M,) float64
        cpn_amt_ptr,        # (M,) float64
        w_fund_ptr,         # (M,) float64
        inv_lex_perm_ptr,   # (D,) int64
        ws_lx_ptr,          # (2, rc, D, n_pairs_rep) float64
        ws_perf_ptr,        # (2, rc, D, n_pairs_rep) float64
        tile_pv_sum_ptr,    # (rc, n_tiles_mf) float64
        m_rem,
        rc,
        n_pairs_rep,
        n_tiles_mf,
        c_mem_0,
        ac_cpn,
        ko_spread,
        cpn_spread,
        strike,
        max_spread_put,
        option_gearing,
        ki_barrier,
        ki_spread,
        one_star_strike,
        one_star_eps,
        one_star_coupon,
        pcall_strike,
        pcall_rate,
        D_ASSETS: tl.constexpr,
        CALL_CODE: tl.constexpr,
        CPN_CODE: tl.constexpr,
        OPT_CODE: tl.constexpr,
        IS_MEM: tl.constexpr,
        NO_NOTIONAL: tl.constexpr,
        FORCE_CPN_AC: tl.constexpr,
        CPN_DIR_DOWN: tl.constexpr,
        CPN_TYPE_OUT: tl.constexpr,
        HAS_KO_SPREAD: tl.constexpr,
        HAS_CPN_SPREAD: tl.constexpr,
        HAS_KI: tl.constexpr,
        HAS_KI_SPREAD: tl.constexpr,
        IS_DISCRETE_KI: tl.constexpr,
        HAS_PUT: tl.constexpr,
        HAS_LOW_K: tl.constexpr,
        HAS_ONESTAR: tl.constexpr,
        HAS_OS_CPN: tl.constexpr,
        HAS_PCALL: tl.constexpr,
        IS_LONG_BASKET: tl.constexpr,
        HAS_INV_PERM: tl.constexpr,
        NEED_DISP: tl.constexpr,
        NEED_ORDER: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Evaluates antithetic (+Z, -Z) BB+PCA flat-ATM GBM control variate in float64."""
        pid_n = tl.program_id(0)
        pid_r = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_pairs_rep

        if HAS_INV_PERM:
            lex0 = tl.load(inv_lex_perm_ptr)
            lex1 = tl.load(inv_lex_perm_ptr + 1) if D_ASSETS >= 2 else 0
        else:
            lex0 = 0
            lex1 = 1 if D_ASSETS >= 2 else 0

        al_p = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
        unp_p = tl.full((BLOCK_N,), c_mem_0, dtype=tl.float64)
        tpv_p = tl.zeros((BLOCK_N,), dtype=tl.float64)
        not_ki_p = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)

        al_m = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
        unp_m = tl.full((BLOCK_N,), c_mem_0, dtype=tl.float64)
        tpv_m = tl.zeros((BLOCK_N,), dtype=tl.float64)
        not_ki_m = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)

        w_min_p = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        w_max_p = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)
        w_sum_p = tl.zeros((BLOCK_N,), dtype=tl.float64)
        w_disp_p = tl.zeros((BLOCK_N,), dtype=tl.float64)
        p_lex0_p = tl.zeros((BLOCK_N,), dtype=tl.float64)
        p_lex1_p = tl.zeros((BLOCK_N,), dtype=tl.float64)
        min2_p = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        min3_p = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        max2_p = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)

        w_min_m = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        w_max_m = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)
        w_sum_m = tl.zeros((BLOCK_N,), dtype=tl.float64)
        w_disp_m = tl.zeros((BLOCK_N,), dtype=tl.float64)
        p_lex0_m = tl.zeros((BLOCK_N,), dtype=tl.float64)
        p_lex1_m = tl.zeros((BLOCK_N,), dtype=tl.float64)
        min2_m = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        min3_m = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
        max2_m = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)

        side0_base = pid_r * D_ASSETS
        side1_base = (rc + pid_r) * D_ASSETS

        for m in range(m_rem):
            w_min_p = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            w_max_p = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)
            w_sum_p = tl.zeros((BLOCK_N,), dtype=tl.float64)
            w_disp_p = tl.zeros((BLOCK_N,), dtype=tl.float64)
            p_lex0_p = tl.zeros((BLOCK_N,), dtype=tl.float64)
            p_lex1_p = tl.zeros((BLOCK_N,), dtype=tl.float64)
            min2_p = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            min3_p = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            max2_p = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)

            w_min_m = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            w_max_m = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)
            w_sum_m = tl.zeros((BLOCK_N,), dtype=tl.float64)
            w_disp_m = tl.zeros((BLOCK_N,), dtype=tl.float64)
            p_lex0_m = tl.zeros((BLOCK_N,), dtype=tl.float64)
            p_lex1_m = tl.zeros((BLOCK_N,), dtype=tl.float64)
            min2_m = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            min3_m = tl.full((BLOCK_N,), 1e30, dtype=tl.float64)
            max2_m = tl.full((BLOCK_N,), -1e30, dtype=tl.float64)

            for d in tl.static_range(D_ASSETS):
                off0 = (side0_base + d) * n_pairs_rep + offs_n
                off1 = (side1_base + d) * n_pairs_rep + offs_n
                if m == 0:
                    lx_prev_p = tl.zeros((BLOCK_N,), dtype=tl.float64)
                    lx_prev_m = tl.zeros((BLOCK_N,), dtype=tl.float64)
                else:
                    lx_prev_p = tl.load(ws_lx_ptr + off0, mask=mask_n, other=0.0)
                    lx_prev_m = tl.load(ws_lx_ptr + off1, mask=mask_n, other=0.0)

                zc = tl.load(
                    z_corr_ptr + ((pid_r * m_rem + m) * D_ASSETS + d) * n_pairs_rep + offs_n,
                    mask=mask_n,
                    other=0.0,
                )
                dv = tl.load(d_var_gbm_ptr + d * m_rem + m)
                dmu = -0.5 * dv
                step = tl.sqrt(dv) * zc.to(tl.float64)
                lx_curr_p = lx_prev_p + dmu + step
                lx_curr_m = lx_prev_m + dmu - step
                tl.store(ws_lx_ptr + off0, lx_curr_p, mask=mask_n)
                tl.store(ws_lx_ptr + off1, lx_curr_m, mask=mask_n)

                fwd_d = tl.load(fwd_perf_ptr + d * m_rem + m)
                p_p = fwd_d * tl.exp(lx_curr_p)
                p_m = fwd_d * tl.exp(lx_curr_m)
                if NEED_DISP and D_ASSETS > 1:
                    tl.store(ws_perf_ptr + off0, p_p, mask=mask_n)
                    tl.store(ws_perf_ptr + off1, p_m, mask=mask_n)
                if NEED_ORDER:
                    c1 = p_p < w_min_p
                    c2 = (~c1) & (p_p < min2_p)
                    c3 = (~c1) & (~c2) & (p_p < min3_p)
                    min3_p = tl.where(c1, min2_p, tl.where(c2, min2_p, tl.where(c3, p_p, min3_p)))
                    min2_p = tl.where(c1, w_min_p, tl.where(c2, p_p, min2_p))
                    g1 = p_p > w_max_p
                    g2 = (~g1) & (p_p > max2_p)
                    max2_p = tl.where(g1, w_max_p, tl.where(g2, p_p, max2_p))

                    c1m = p_m < w_min_m
                    c2m = (~c1m) & (p_m < min2_m)
                    c3m = (~c1m) & (~c2m) & (p_m < min3_m)
                    min3_m = tl.where(c1m, min2_m, tl.where(c2m, min2_m, tl.where(c3m, p_m, min3_m)))
                    min2_m = tl.where(c1m, w_min_m, tl.where(c2m, p_m, min2_m))
                    g1m = p_m > w_max_m
                    g2m = (~g1m) & (p_m > max2_m)
                    max2_m = tl.where(g1m, w_max_m, tl.where(g2m, p_m, max2_m))

                w_min_p = tl.minimum(w_min_p, p_p)
                w_max_p = tl.maximum(w_max_p, p_p)
                w_sum_p = w_sum_p + p_p
                p_lex0_p = tl.where(d == lex0, p_p, p_lex0_p)
                p_lex1_p = tl.where(d == lex1, p_p, p_lex1_p)

                w_min_m = tl.minimum(w_min_m, p_m)
                w_max_m = tl.maximum(w_max_m, p_m)
                w_sum_m = w_sum_m + p_m
                p_lex0_m = tl.where(d == lex0, p_m, p_lex0_m)
                p_lex1_m = tl.where(d == lex1, p_m, p_lex1_m)

            if NEED_DISP and D_ASSETS > 1:
                w_mean_p = w_sum_p * (1.0 / float(D_ASSETS))
                w_mean_m = w_sum_m * (1.0 / float(D_ASSETS))
                for d in tl.static_range(D_ASSETS):
                    p_p = tl.load(ws_perf_ptr + (side0_base + d) * n_pairs_rep + offs_n, mask=mask_n, other=0.0)
                    p_m = tl.load(ws_perf_ptr + (side1_base + d) * n_pairs_rep + offs_n, mask=mask_n, other=0.0)
                    dp = p_p - w_mean_p
                    dm = p_m - w_mean_m
                    w_disp_p = w_disp_p + dp * dp
                    w_disp_m = w_disp_m + dm * dm

            w_call_p = _eval_index_from_stats_triton(
                CALL_CODE, D_ASSETS, w_min_p, w_max_p, w_sum_p, w_disp_p, p_lex0_p, p_lex1_p, min2_p, min3_p, max2_p
            )
            w_cpn_p = (
                w_call_p
                if CPN_CODE == CALL_CODE
                else _eval_index_from_stats_triton(
                    CPN_CODE, D_ASSETS, w_min_p, w_max_p, w_sum_p, w_disp_p, p_lex0_p, p_lex1_p, min2_p, min3_p, max2_p
                )
            )
            w_opt_p = (
                w_call_p
                if OPT_CODE == CALL_CODE
                else (
                    w_cpn_p
                    if OPT_CODE == CPN_CODE
                    else _eval_index_from_stats_triton(
                        OPT_CODE, D_ASSETS, w_min_p, w_max_p, w_sum_p, w_disp_p, p_lex0_p, p_lex1_p, min2_p, min3_p, max2_p
                    )
                )
            )

            w_call_m = _eval_index_from_stats_triton(
                CALL_CODE, D_ASSETS, w_min_m, w_max_m, w_sum_m, w_disp_m, p_lex0_m, p_lex1_m, min2_m, min3_m, max2_m
            )
            w_cpn_m = (
                w_call_m
                if CPN_CODE == CALL_CODE
                else _eval_index_from_stats_triton(
                    CPN_CODE, D_ASSETS, w_min_m, w_max_m, w_sum_m, w_disp_m, p_lex0_m, p_lex1_m, min2_m, min3_m, max2_m
                )
            )
            w_opt_m = (
                w_call_m
                if OPT_CODE == CALL_CODE
                else (
                    w_cpn_m
                    if OPT_CODE == CPN_CODE
                    else _eval_index_from_stats_triton(
                        OPT_CODE, D_ASSETS, w_min_m, w_max_m, w_sum_m, w_disp_m, p_lex0_m, p_lex1_m, min2_m, min3_m, max2_m
                    )
                )
            )

            is_call = tl.load(is_callable_ptr + m) > 0.5
            b_ko_m = tl.load(ko_barriers_ptr + m)
            is_guar = tl.load(is_guar_ptr + m) > 0.5
            b_cpn_m = tl.load(cpn_barriers_ptr + m)
            df_m = tl.load(df_pay_ptr + m)
            c_m = tl.load(cpn_amt_ptr + m)
            wf_m = tl.load(w_fund_ptr + m)

            w_ki_p = w_min_p if IS_LONG_BASKET else w_opt_p
            al_p, unp_p, tpv_p, not_ki_p, _, _, _ = _mf_period_step_triton(
                w_call_p,
                w_cpn_p,
                w_ki_p,
                al_p,
                unp_p,
                tpv_p,
                not_ki_p,
                is_call,
                b_ko_m,
                is_guar,
                b_cpn_m,
                df_m,
                c_m,
                wf_m,
                m,
                m_rem,
                ac_cpn,
                ko_spread,
                cpn_spread,
                ki_barrier,
                ki_spread,
                IS_MEM,
                NO_NOTIONAL,
                FORCE_CPN_AC,
                CPN_DIR_DOWN,
                CPN_TYPE_OUT,
                HAS_KO_SPREAD,
                HAS_CPN_SPREAD,
                HAS_KI,
                HAS_KI_SPREAD,
                IS_DISCRETE_KI,
            )
            w_ki_m = w_min_m if IS_LONG_BASKET else w_opt_m
            al_m, unp_m, tpv_m, not_ki_m, _, _, _ = _mf_period_step_triton(
                w_call_m,
                w_cpn_m,
                w_ki_m,
                al_m,
                unp_m,
                tpv_m,
                not_ki_m,
                is_call,
                b_ko_m,
                is_guar,
                b_cpn_m,
                df_m,
                c_m,
                wf_m,
                m,
                m_rem,
                ac_cpn,
                ko_spread,
                cpn_spread,
                ki_barrier,
                ki_spread,
                IS_MEM,
                NO_NOTIONAL,
                FORCE_CPN_AC,
                CPN_DIR_DOWN,
                CPN_TYPE_OUT,
                HAS_KO_SPREAD,
                HAS_CPN_SPREAD,
                HAS_KI,
                HAS_KI_SPREAD,
                IS_DISCRETE_KI,
            )

        df_term = tl.load(df_pay_ptr + m_rem - 1)
        w_opt_term_p = _eval_index_from_stats_triton(
            OPT_CODE, D_ASSETS, w_min_p, w_max_p, w_sum_p, w_disp_p, p_lex0_p, p_lex1_p, min2_p, min3_p, max2_p
        )
        w_opt_term_m = _eval_index_from_stats_triton(
            OPT_CODE, D_ASSETS, w_min_m, w_max_m, w_sum_m, w_disp_m, p_lex0_m, p_lex1_m, min2_m, min3_m, max2_m
        )
        inv_d_mf = 1.0 / float(D_ASSETS)
        opt_p, _, _, _, _ = _mf_terminal_opt_triton(
            w_min_p,
            w_max_p,
            w_sum_p * inv_d_mf,
            w_opt_term_p,
            al_p,
            not_ki_p,
            df_term,
            strike,
            max_spread_put,
            option_gearing,
            ki_barrier,
            ki_spread,
            one_star_strike,
            one_star_eps,
            one_star_coupon,
            pcall_strike,
            pcall_rate,
            HAS_KI,
            HAS_KI_SPREAD,
            IS_DISCRETE_KI,
            HAS_PUT,
            HAS_LOW_K,
            HAS_ONESTAR,
            HAS_OS_CPN,
            HAS_PCALL,
            IS_LONG_BASKET,
        )
        opt_m, _, _, _, _ = _mf_terminal_opt_triton(
            w_min_m,
            w_max_m,
            w_sum_m * inv_d_mf,
            w_opt_term_m,
            al_m,
            not_ki_m,
            df_term,
            strike,
            max_spread_put,
            option_gearing,
            ki_barrier,
            ki_spread,
            one_star_strike,
            one_star_eps,
            one_star_coupon,
            pcall_strike,
            pcall_rate,
            HAS_KI,
            HAS_KI_SPREAD,
            IS_DISCRETE_KI,
            HAS_PUT,
            HAS_LOW_K,
            HAS_ONESTAR,
            HAS_OS_CPN,
            HAS_PCALL,
            IS_LONG_BASKET,
        )

        pv_pair = tl.where(mask_n, 0.5 * ((tpv_p + opt_p) + (tpv_m + opt_m)), 0.0)
        tl.store(tile_pv_sum_ptr + pid_r * n_tiles_mf + pid_n, tl.sum(pv_pair, axis=0))

    @triton.jit  # type: ignore[untyped-decorator]
    def _finalize_qmc_pv_stats_kernel(  # type: ignore[no-untyped-def]
        period_tile_stats_ptr,  # (M, R, n_tiles, 4) float64
        term_tile_stats_ptr,    # (R, n_tiles, 11) float64
        packed_out_ptr,         # (8 + 3 * M,) float64
        m_rem,
        eff_r,
        n_tiles,
        inv_n_rep,              # float64
        inv_tot,                # float64
        inv_r,                  # float64
        inv_rr1,                # float64
        denom_se1,              # float64
        mu_cv,                  # float64
        se_mu_cv,               # float64
        HAS_CV: tl.constexpr,
        EFF_R_GT_1: tl.constexpr,
        R_PAD: tl.constexpr,
    ):
        """Deterministically sums per-tile statistics in 0..n_tiles-1 order into 1 D2H buffer."""
        offs_r = tl.arange(0, R_PAD)
        mask_r = offs_r < eff_r

        for m in range(m_rem):
            s_stop = tl.zeros((), dtype=tl.float64)
            s_cpnal = tl.zeros((), dtype=tl.float64)
            s_survb = tl.zeros((), dtype=tl.float64)
            s_cpncf = tl.zeros((), dtype=tl.float64)
            for r in range(eff_r):
                r_off = (m * eff_r + r) * n_tiles
                for t in range(n_tiles):
                    b_off = (r_off + t) * 4
                    s_stop = s_stop + tl.load(period_tile_stats_ptr + b_off + 0)
                    s_cpnal = s_cpnal + tl.load(period_tile_stats_ptr + b_off + 1)
                    s_survb = s_survb + tl.load(period_tile_stats_ptr + b_off + 2)
                    s_cpncf = s_cpncf + tl.load(period_tile_stats_ptr + b_off + 3)
            p_stop_m = s_stop * inv_tot
            cpn_al_m = s_cpnal * inv_tot
            surv_b_m = s_survb * inv_tot
            q_cpn_m = tl.where(
                surv_b_m > 1e-12, cpn_al_m / tl.maximum(surv_b_m, 1e-12), 0.0
            )
            cpn_cf_m = s_cpncf * inv_tot
            tl.store(packed_out_ptr + 8 + m, p_stop_m)
            tl.store(packed_out_ptr + 8 + m_rem + m, q_cpn_m)
            tl.store(packed_out_ptr + 8 + 2 * m_rem + m, cpn_cf_m)

        s_put = tl.zeros((R_PAD,), dtype=tl.float64)
        s_call = tl.zeros((R_PAD,), dtype=tl.float64)
        s_star = tl.zeros((R_PAD,), dtype=tl.float64)
        s_pstar = tl.zeros((R_PAD,), dtype=tl.float64)
        s_y = tl.zeros((R_PAD,), dtype=tl.float64)
        s_yy = tl.zeros((R_PAD,), dtype=tl.float64)
        s_o = tl.zeros((R_PAD,), dtype=tl.float64)
        s_oo = tl.zeros((R_PAD,), dtype=tl.float64)
        s_x = tl.zeros((R_PAD,), dtype=tl.float64)
        s_xx = tl.zeros((R_PAD,), dtype=tl.float64)
        s_xy = tl.zeros((R_PAD,), dtype=tl.float64)

        for t in range(n_tiles):
            t_off = (offs_r * n_tiles + t) * 11
            s_put = s_put + tl.load(term_tile_stats_ptr + t_off + 0, mask=mask_r, other=0.0)
            s_call = s_call + tl.load(term_tile_stats_ptr + t_off + 1, mask=mask_r, other=0.0)
            s_star = s_star + tl.load(term_tile_stats_ptr + t_off + 2, mask=mask_r, other=0.0)
            s_pstar = s_pstar + tl.load(term_tile_stats_ptr + t_off + 3, mask=mask_r, other=0.0)
            s_y = s_y + tl.load(term_tile_stats_ptr + t_off + 4, mask=mask_r, other=0.0)
            s_yy = s_yy + tl.load(term_tile_stats_ptr + t_off + 5, mask=mask_r, other=0.0)
            s_o = s_o + tl.load(term_tile_stats_ptr + t_off + 6, mask=mask_r, other=0.0)
            s_oo = s_oo + tl.load(term_tile_stats_ptr + t_off + 7, mask=mask_r, other=0.0)
            s_x = s_x + tl.load(term_tile_stats_ptr + t_off + 8, mask=mask_r, other=0.0)
            s_xx = s_xx + tl.load(term_tile_stats_ptr + t_off + 9, mask=mask_r, other=0.0)
            s_xy = s_xy + tl.load(term_tile_stats_ptr + t_off + 10, mask=mask_r, other=0.0)

        l_put = tl.sum(s_put, axis=0) * inv_tot
        l_call = tl.sum(s_call, axis=0) * inv_tot
        l_onestar = tl.sum(s_star, axis=0) * inv_tot
        p_onestar = tl.sum(s_pstar, axis=0) * inv_tot
        raw_pv_mean = tl.sum(s_y, axis=0) * inv_tot

        y_means = s_y * inv_n_rep
        o_means = s_o * inv_n_rep

        if not EFF_R_GT_1:
            y0 = tl.sum(s_y, axis=0)
            yy0 = tl.sum(s_yy, axis=0)
            o0 = tl.sum(s_o, axis=0)
            oo0 = tl.sum(s_oo, axis=0)
            var_y = tl.maximum((yy0 - y0 * y0 * inv_n_rep) / denom_se1, 0.0)
            var_o = tl.maximum((oo0 - o0 * o0 * inv_n_rep) / denom_se1, 0.0)
            cv_pv_est = raw_pv_mean
            total_se = tl.sqrt(var_y)
            option_se = tl.sqrt(var_o)
        else:
            o_bar = tl.sum(o_means, axis=0) * inv_r
            do = tl.where(mask_r, o_means - o_bar, 0.0)
            option_se = tl.sqrt(tl.maximum(tl.sum(do * do, axis=0) * inv_rr1, 0.0))

            if HAS_CV:
                x_means = s_x * inv_n_rep
                cov_r = tl.where(mask_r, s_xy - s_x * s_y * inv_n_rep, 0.0)
                varx_r = tl.where(mask_r, s_xx - s_x * s_x * inv_n_rep, 0.0)
                tot_cov = tl.sum(cov_r, axis=0)
                tot_var = tl.sum(varx_r, axis=0)
                loo_cov = tot_cov - cov_r
                loo_var = tot_var - varx_r
                beta_loo = tl.where(
                    loo_var > 1e-18,
                    loo_cov / tl.maximum(loo_var, 1e-18),
                    0.0,
                )
                v_cv_r = tl.where(mask_r, y_means - beta_loo * (x_means - mu_cv), 0.0)
                cv_pv_est = tl.sum(v_cv_r, axis=0) * inv_r
                mean_beta = tl.sum(tl.where(mask_r, beta_loo, 0.0), axis=0) * inv_r
                dv = tl.where(mask_r, v_cv_r - cv_pv_est, 0.0)
                total_se = tl.sqrt(
                    tl.maximum(
                        tl.sum(dv * dv, axis=0) * inv_rr1 + (mean_beta * se_mu_cv) * (mean_beta * se_mu_cv),
                        0.0,
                    )
                )
            else:
                cv_pv_est = raw_pv_mean
                dy = tl.where(mask_r, y_means - raw_pv_mean, 0.0)
                total_se = tl.sqrt(tl.maximum(tl.sum(dy * dy, axis=0) * inv_rr1, 0.0))

        tl.store(packed_out_ptr + 0, l_put)
        tl.store(packed_out_ptr + 1, l_call)
        tl.store(packed_out_ptr + 2, l_onestar)
        tl.store(packed_out_ptr + 3, p_onestar)
        tl.store(packed_out_ptr + 4, raw_pv_mean)
        tl.store(packed_out_ptr + 5, cv_pv_est)
        tl.store(packed_out_ptr + 6, total_se)
        tl.store(packed_out_ptr + 7, option_se)
else:
    _sobol_vec_flat = None


def _sobol_from_shifts_torch(
    *,
    sv_gpu: torch.Tensor,
    shifts_gpu: torch.Tensor,
    dim_offset: int,
    n_rep: int,
    device: torch.device,
    dim_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes scrambled-Sobol standard normals (R, n_sobol, N_rep) in pure PyTorch from shifts."""
    n_replicates, n_sobol = int(shifts_gpu.shape[0]), int(shifts_gpu.shape[1])
    m_pow = int(round(math.log2(n_rep)))
    idx = torch.arange(n_rep, dtype=torch.int64, device=device)
    gray = idx ^ (idx >> 1)
    if dim_indices is not None:
        sv_slice = sv_gpu[dim_indices, :m_pow].to(torch.int64)
    else:
        sv_slice = sv_gpu[dim_offset : dim_offset + n_sobol, :m_pow].to(torch.int64)
    bits = (gray[None, :, None] >> torch.arange(m_pow, device=device)[None, None, :]) & 1
    acc = torch.zeros((n_replicates, n_sobol, n_rep), dtype=torch.int64, device=device)
    for b in range(m_pow):
        mask_b = bits[0, :, b]
        acc ^= mask_b[None, None, :] * sv_slice[None, :, b : b + 1]
    acc ^= shifts_gpu.to(torch.int64).unsqueeze(-1)
    u = ((acc.to(torch.float32) + 0.5) * _SOBOL_INV_SCALE).clamp(1e-6, 1.0 - 1e-6)
    z_out: torch.Tensor = torch.special.ndtri(u)
    return z_out


def _generate_gpu_sobol_tile(
    *,
    n_rep: int,
    n_replicates: int,
    n_dims: int,
    dim_offset: int,
    seed: int,
    device: torch.device,
    use_triton: bool = True,
) -> torch.Tensor:
    """Generates (R, n_dims, N_rep) float32 standard normal tile via GPU Sobol + Philox >= 21,201."""
    if n_dims <= 0:
        return torch.empty((n_replicates, 0, n_rep), dtype=torch.float32, device=device)

    n_sobol = max(0, min(n_dims, _SOBOL_MAX_DIM - dim_offset))
    n_philox = n_dims - n_sobol

    parts: List[torch.Tensor] = []
    if n_sobol > 0:
        sv_gpu = _get_sobol_direction_numbers(device)
        m_pow = int(round(math.log2(n_rep)))
        gen_s = torch.Generator(device="cpu")
        gen_s.manual_seed(int(seed) + dim_offset * 131 + 17)
        shifts = torch.randint(
            0, 1 << _SOBOL_BITS, (n_replicates, n_sobol), dtype=torch.int32, device="cpu", generator=gen_s
        ).to(device=device)
        if use_triton and _HAS_TRITON and device.type == "cuda":
            z_sobol = torch.empty((n_replicates, n_sobol, n_rep), dtype=torch.float32, device=device)
            block_n = min(256, max(16, _next_power_of_two(n_rep)))
            grid = (triton.cdiv(n_rep, block_n), n_sobol, n_replicates)
            _fast_triton_launch(
                _sobol_normals_triton_kernel,
                grid,
                sv_gpu,
                shifts,
                z_sobol,
                dim_offset,
                n_sobol,
                n_rep,
                m_pow,
                BLOCK_N=block_n,
            )
        else:
            z_sobol = _sobol_from_shifts_torch(
                sv_gpu=sv_gpu,
                shifts_gpu=shifts,
                dim_offset=dim_offset,
                n_rep=n_rep,
                device=device,
            )
        parts.append(z_sobol)

    if n_philox > 0:
        gen_p = torch.Generator(device="cpu")
        gen_p.manual_seed(int(seed) + dim_offset * 977 + 99991)
        half = (n_rep + 1) // 2
        z_half = torch.randn(
            (n_replicates, n_philox, half), dtype=torch.float32, device="cpu", generator=gen_p
        ).to(device=device)
        z_philox = torch.cat([z_half, -z_half], dim=-1)[:, :, :n_rep]
        parts.append(z_philox)

    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def _compute_escrowed_fwd_and_drift_batch(
    spots_t0: torch.Tensor,
    r_tot_obs: torch.Tensor,
    df_fin_obs: torch.Tensor,
    tau_2d: torch.Tensor,
    t_eff: torch.Tensor,
    div_events_list: Sequence[Sequence[Any]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes batched (A, N_t) escrowed forward, r_zero, and q_repo grids matching QAG."""
    a_cnt = int(spots_t0.shape[0])
    m_len = int(tau_2d.shape[1])
    device = spots_t0.device

    zeros_a = torch.zeros((a_cnt, 1), dtype=torch.float64, device=device)
    ones_a = torch.ones((a_cnt, 1), dtype=torch.float64, device=device)
    tau_knots = torch.cat([zeros_a, tau_2d], dim=1).contiguous()
    r_tot_knots = torch.cat([zeros_a, r_tot_obs], dim=1)
    log_df_knots = torch.log(torch.clamp(torch.cat([ones_a, df_fin_obs], dim=1), min=1e-15))

    max_div = max((len(evs) for evs in div_events_list), default=0)
    if max_div > 0:
        ex_taus_np = np.zeros((a_cnt, max_div), dtype=np.float64)
        ex_pvs_np = np.zeros((a_cnt, max_div), dtype=np.float64)
        for a_i, evs in enumerate(div_events_list):
            for j, ev in enumerate(evs):
                ex_taus_np[a_i, j] = float(getattr(ev, "tau_ex_act365", 0.0))
                ex_pvs_np[a_i, j] = float(getattr(ev, "amount", 0.0)) * float(
                    getattr(ev, "pv_discount_weight", 0.0)
                )
        ex_taus = torch.tensor(ex_taus_np, dtype=torch.float64, device=device)
        ex_pvs = torch.tensor(ex_pvs_np, dtype=torch.float64, device=device)
        mask_ex_k = (ex_taus[:, None, :] > 0.0) & (ex_taus[:, None, :] <= tau_knots[:, :, None] + 1e-12)
        d_pv_knots = (mask_ex_k.to(torch.float64) * ex_pvs[:, None, :]).sum(dim=2)
    else:
        d_pv_knots = torch.zeros((a_cnt, m_len + 1), dtype=torch.float64, device=device)

    floor_sp = 0.01 * spots_t0[:, None]
    scale_sp = torch.clamp(floor_sp, min=1e-12)
    escrowed_knots = floor_sp + scale_sp * F.softplus(
        (spots_t0[:, None] - d_pv_knots - floor_sp) / scale_sp, beta=1.0, threshold=40.0
    )

    k_idx = torch.clamp(torch.searchsorted(tau_knots, t_eff, right=True) - 1, 0, m_len - 1)
    tau0 = torch.gather(tau_knots, 1, k_idx)
    tau1 = torch.gather(tau_knots, 1, k_idx + 1)
    r0 = torch.gather(r_tot_knots, 1, k_idx)
    r1 = torch.gather(r_tot_knots, 1, k_idx + 1)
    ldf0 = torch.gather(log_df_knots, 1, k_idx)
    ldf1 = torch.gather(log_df_knots, 1, k_idx + 1)

    dt_k = torch.clamp(tau1 - tau0, min=1e-8)
    alpha_k = torch.clamp((t_eff - tau0) / dt_k, 0.0, 1.0)
    r_tot_grid = r0 + alpha_k * (r1 - r0)
    r_zero_grid = -(ldf1 - ldf0) / dt_k

    # Smooth exponential interpolation of escrowed spot across tau_knots matching qmc_aad_greeks.
    esc0 = torch.gather(escrowed_knots, 1, k_idx)
    esc1 = torch.gather(escrowed_knots, 1, k_idx + 1)
    q_div_grid = -torch.log(torch.clamp(esc1 / torch.clamp(esc0, min=1e-12), min=1e-12)) / dt_k
    escrowed_grid = esc0 * torch.exp(-q_div_grid * torch.clamp(t_eff - tau0, min=0.0))
    f_grid = torch.clamp(escrowed_grid * torch.exp(r_tot_grid), min=1e-4)

    f_nat_knots = torch.clamp(escrowed_knots * torch.exp(r_tot_knots), min=1e-12)
    fn0 = torch.gather(f_nat_knots, 1, k_idx)
    fn1 = torch.gather(f_nat_knots, 1, k_idx + 1)
    buehler_eff_drift = torch.log(fn1 / fn0) / dt_k
    q_repo_grid = r_zero_grid - buehler_eff_drift
    return f_grid, r_zero_grid, q_repo_grid


def _build_lv_tables_batch(
    surfs: Sequence[Any],
    spots_t0: torch.Tensor,
    r_tot_obs: torch.Tensor,
    df_fin_obs: torch.Tensor,
    tau_obs: torch.Tensor,
    div_events_list: Sequence[Sequence[Any]],
    v_term: torch.Tensor,
    n_t: int,
    n_y: int,
    device: torch.device,
    *,
    use_triton: bool = True,
    lv_digest: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds batched (A, N_t, N_y) float32 Dupire LV tables via fused Triton kernel."""
    a_cnt = len(surfs)
    if a_cnt == 0:
        return (
            torch.empty((0, n_t, n_y), dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    if lv_digest is None:
        lv_digest = _compute_lv_batch_digest(
            surfs,
            _cpu_f64_array(spots_t0),
            _cpu_f64_array(r_tot_obs),
            _cpu_f64_array(df_fin_obs),
            _cpu_f64_array(tau_obs),
            div_events_list,
            _cpu_f64_array(v_term),
            n_t,
            n_y,
            device,
            use_triton,
        )
    lv_cache_key = ("lv_table_v2", lv_digest)
    cached_lv = _LV_TABLE_LRU.get(lv_cache_key)
    if cached_lv is not None:
        return cached_lv  # type: ignore[no-any-return]

    can_triton = bool(use_triton and _HAS_TRITON and device.type == "cuda" and (n_y & (n_y - 1)) == 0)
    g_list = [_get_surface_gpu_tensors(s, device) for s in surfs]
    n_u_list = [int(g["strikes"].shape[0]) for g in g_list]
    n_exp_list = [int(g["tenors"].shape[0]) for g in g_list]
    max_u = max(n_u_list)
    max_exp = max(n_exp_list)
    if max_u > 1024 or not can_triton:
        tau_2d_cpu = (
            tau_obs.unsqueeze(0).expand(a_cnt, -1) if tau_obs.ndim == 1 else tau_obs
        ).detach().cpu().numpy()
        spots_cpu = spots_t0.detach().cpu().numpy()
        r_cpu = r_tot_obs.detach().cpu().numpy()
        df_cpu = df_fin_obs.detach().cpu().numpy()
        vt_cpu = v_term.detach().cpu().numpy()
        lv_l: List[torch.Tensor] = []
        ym_l: List[float] = []
        idy_l: List[float] = []
        for a_i in range(a_cnt):
            lv_i, ym_i, idy_i = _eval_gpu_surface_local_vol_table(
                surfs[a_i],
                float(spots_cpu[a_i]),
                r_cpu[a_i],
                df_cpu[a_i],
                tau_2d_cpu[a_i],
                div_events_list[a_i],
                float(vt_cpu[a_i]),
                n_t,
                n_y,
                device,
            )
            lv_l.append(lv_i)
            ym_l.append(ym_i)
            idy_l.append(idy_i)
        out_fb = (
            torch.stack(lv_l, dim=0),
            torch.tensor(ym_l, dtype=torch.float32, device=device),
            torch.tensor(idy_l, dtype=torch.float32, device=device),
        )
        return _LV_TABLE_LRU.put(lv_cache_key, out_fb)  # type: ignore[no-any-return]

    tau_2d = tau_obs.unsqueeze(0).expand(a_cnt, -1) if tau_obs.ndim == 1 else tau_obs
    tau_max = torch.clamp(tau_2d[:, -1], min=1e-4)
    steps = torch.linspace(0.0, 1.0, n_t, dtype=torch.float64, device=device)
    t_eff = torch.clamp(tau_max[:, None] * steps[None, :], min=1e-4)

    f_grid, r_zero_grid, q_repo_grid = _compute_escrowed_fwd_and_drift_batch(
        spots_t0, r_tot_obs, df_fin_obs, tau_2d, t_eff, div_events_list
    )

    half_w = torch.clamp(5.0 * v_term, min=0.50, max=6.0)
    y_min_64 = -half_w
    dy_64 = (2.0 * half_w) / float(max(n_y - 1, 1))
    inv_dy_32 = (float(max(n_y - 1, 1)) / (2.0 * half_w)).to(torch.float32)

    uniform_shape = all(nu == max_u for nu in n_u_list) and all(ne == max_exp for ne in n_exp_list)
    if uniform_shape:
        strikes_b = torch.stack([g["strikes"] for g in g_list], dim=0)
        tenors_b = torch.stack([g["tenors"] for g in g_list], dim=0)
        tenors_365_b = torch.stack([g["tenors_365"] for g in g_list], dim=0)
        quotes_b = torch.stack([g["quotes"] for g in g_list], dim=0)
        slice_c_b = torch.stack([g["slice_c_tr"] for g in g_list], dim=0)
        sl_left_b = torch.stack([g["slopes_left"] for g in g_list], dim=0)
        sl_right_b = torch.stack([g["slopes_right"] for g in g_list], dim=0)
    else:
        strikes_b = torch.zeros((a_cnt, max_u), dtype=torch.float64, device=device)
        tenors_b = torch.zeros((a_cnt, max_exp), dtype=torch.float64, device=device)
        tenors_365_b = torch.zeros((a_cnt, max_exp), dtype=torch.float64, device=device)
        quotes_b = torch.zeros((a_cnt, max_exp, max_u), dtype=torch.float64, device=device)
        slice_c_b = torch.zeros((a_cnt, 4, max_exp, max_u - 1), dtype=torch.float64, device=device)
        sl_left_b = torch.zeros((a_cnt, max_exp), dtype=torch.float64, device=device)
        sl_right_b = torch.zeros((a_cnt, max_exp), dtype=torch.float64, device=device)
        for a_i, g in enumerate(g_list):
            nu = n_u_list[a_i]
            ne = n_exp_list[a_i]
            strikes_b[a_i, :nu] = g["strikes"]
            tenors_b[a_i, :ne] = g["tenors"]
            tenors_365_b[a_i, :ne] = g["tenors_365"]
            quotes_b[a_i, :ne, :nu] = g["quotes"]
            slice_c_b[a_i, :, :ne, : nu - 1] = g["slice_c_tr"]
            sl_left_b[a_i, :ne] = g["slopes_left"]
            sl_right_b[a_i, :ne] = g["slopes_right"]

    n_u_t = torch.tensor(n_u_list, dtype=torch.int32, device=device)
    n_exp_t = torch.tensor(n_exp_list, dtype=torch.int32, device=device)
    is_365_t = torch.tensor(
        [
            1
            if str(getattr(s, "dcc", "Act365")).strip().lower()
            in ("act365", "act365f", "act_365", "act_365f")
            else 0
            for s in surfs
        ],
        dtype=torch.int32,
        device=device,
    )
    min_vol_t = torch.tensor([float(s.min_vol) for s in surfs], dtype=torch.float64, device=device)
    max_lv_t = torch.tensor([float(s.max_local_vol) for s in surfs], dtype=torch.float64, device=device)

    lv_out = torch.empty((a_cnt, n_t, n_y), dtype=torch.float32, device=device)
    _fast_triton_launch(
        _build_lv_table_triton_kernel,
        (n_t, a_cnt),
        strikes_b,
        n_u_t,
        tenors_b,
        tenors_365_b,
        n_exp_t,
        quotes_b,
        slice_c_b,
        sl_left_b,
        sl_right_b,
        is_365_t,
        min_vol_t,
        max_lv_t,
        f_grid,
        r_zero_grid,
        q_repo_grid,
        t_eff,
        y_min_64,
        dy_64,
        lv_out,
        max_exp=max_exp,
        max_u=max_u,
        N_t=n_t,
        N_y=n_y,
    )
    out_lv_tuple = (lv_out, y_min_64.to(torch.float32), inv_dy_32)
    return _LV_TABLE_LRU.put(lv_cache_key, out_lv_tuple)  # type: ignore[no-any-return]


def _eval_gpu_surface_local_vol_table(
    surf: Any,
    spots_t0_i: float,
    r_tot_obs_i: npt.NDArray[np.float64],
    df_fin_obs_i: npt.NDArray[np.float64],
    tau_obs: npt.NDArray[np.float64],
    div_events_i: Sequence[Any],
    v_term_i: float,
    n_t: int,
    n_y: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float, float]:
    """Build adaptive (N_t, N_y) float32 Dupire local-volatility table on GPU."""
    g = _get_surface_gpu_tensors(surf, device)
    strikes = g["strikes"]
    tenors = g["tenors"]
    tenors_365 = g["tenors_365"]
    quotes = g["quotes"]
    slice_c = g["slice_c"]
    slopes_left = g["slopes_left"]
    slopes_right = g["slopes_right"]
    n_exp = int(tenors.shape[0])
    n_u = int(strikes.shape[0])

    tau_max = max(float(tau_obs[-1]), 1e-4)
    t_grid_365 = torch.linspace(0.0, tau_max, n_t, dtype=torch.float64, device=device)
    t_eff_365 = torch.clamp(t_grid_365, min=1e-4)

    f_grid_2d, r_zero_2d, q_repo_2d = _compute_escrowed_fwd_and_drift_batch(
        torch.tensor([spots_t0_i], dtype=torch.float64, device=device),
        torch.as_tensor(r_tot_obs_i, dtype=torch.float64, device=device).unsqueeze(0),
        torch.as_tensor(df_fin_obs_i, dtype=torch.float64, device=device).unsqueeze(0),
        torch.as_tensor(tau_obs, dtype=torch.float64, device=device).unsqueeze(0),
        t_eff_365.unsqueeze(0),
        [div_events_i],
    )
    f_grid = f_grid_2d[0]
    r_zero_grid = r_zero_2d[0]
    q_repo_grid = q_repo_2d[0]

    half_w = min(max(5.0 * float(v_term_i), 0.50), 6.0)
    y_min_val = -half_w
    inv_dy_val = float(n_y - 1) / (2.0 * half_w)
    y_grid = torch.linspace(-half_w, half_w, n_y, dtype=torch.float64, device=device)
    s_grid = torch.clamp(f_grid[:, None] * torch.exp(y_grid[None, :]), min=1e-4)  # (N_t, N_y)

    # Map Act365 t_eff_365 to surface DCC coordinate matching map_act365_to_surface_tenor
    is_act365 = str(getattr(surf, "dcc", "Act365")).strip().lower() in (
        "act365",
        "act365f",
        "act_365",
        "act_365f",
    )
    if is_act365:
        t_surf = t_eff_365
        dt_surf_ratio = torch.ones_like(t_eff_365)
    else:
        t365_0 = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), tenors_365])
        tdcc_0 = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), tenors])
        m_i = torch.clamp(torch.searchsorted(t365_0, t_eff_365, right=True) - 1, 0, len(tenors) - 1)
        slope_dcc = (tdcc_0[m_i + 1] - tdcc_0[m_i]) / torch.clamp(t365_0[m_i + 1] - t365_0[m_i], min=1e-12)
        t_surf_in = tdcc_0[m_i] + slope_dcc * (t_eff_365 - t365_0[m_i])
        ratio_ext = tenors[-1] / torch.clamp(tenors_365[-1], min=1e-12)
        t_surf_ext = tenors[-1] + ratio_ext * (t_eff_365 - tenors_365[-1])
        is_ext = t_eff_365 > tenors_365[-1]
        t_surf = torch.where(is_ext, t_surf_ext, t_surf_in)
        dt_surf_ratio = torch.clamp(torch.where(is_ext, ratio_ext, slope_dcc), min=1e-12)

    # Evaluate spline value + 1st/2nd K-derivatives on GPU in 1 searchsorted pass
    k_min = strikes[0]
    k_max = strikes[-1]
    k_clip = torch.clamp(s_grid, min=k_min, max=k_max)
    seg_idx = torch.clamp(torch.searchsorted(strikes, k_clip, right=True) - 1, 0, n_u - 2)  # (N_t, N_y)
    dx = k_clip - strikes[seg_idx]  # (N_t, N_y)

    # slice_c: (4, N_u-1, N_exp) -> gather for all N_exp slices at seg_idx -> (4, N_t, N_y, N_exp)
    c_all = slice_c[:, seg_idx, :]
    dx_e = dx.unsqueeze(-1)
    v_all = ((c_all[0] * dx_e + c_all[1]) * dx_e + c_all[2]) * dx_e + c_all[3]
    dv_all = (3.0 * c_all[0] * dx_e + 2.0 * c_all[1]) * dx_e + c_all[2]
    d2v_all = 6.0 * c_all[0] * dx_e + 2.0 * c_all[1]

    left_mask = (s_grid < k_min).unsqueeze(-1)
    right_mask = (s_grid > k_max).unsqueeze(-1)
    s_e = s_grid.unsqueeze(-1)
    v_left = quotes[:, 0].view(1, 1, n_exp) + slopes_left.view(1, 1, n_exp) * (s_e - k_min)
    v_right = quotes[:, -1].view(1, 1, n_exp) + slopes_right.view(1, 1, n_exp) * (s_e - k_max)
    v_all = torch.where(left_mask, v_left, torch.where(right_mask, v_right, v_all))
    dv_all = torch.where(
        left_mask,
        slopes_left.view(1, 1, n_exp),
        torch.where(right_mask, slopes_right.view(1, 1, n_exp), dv_all),
    )
    d2v_all = torch.where(left_mask | right_mask, torch.zeros_like(d2v_all), d2v_all)

    min_vol = float(surf.min_vol)
    max_lv = float(surf.max_local_vol)
    clamped_low = v_all < min_vol
    v_all = torch.clamp(v_all, min=min_vol)
    dv_all = torch.where(clamped_low, torch.zeros_like(dv_all), dv_all)
    d2v_all = torch.where(clamped_low, torch.zeros_like(d2v_all), d2v_all)

    if n_exp == 1:
        sig = v_all[:, :, 0]
        dsig = dv_all[:, :, 0]
        d2sig = d2v_all[:, :, 0]
        w = (sig * sig) * t_surf[:, None]
        dw_dt = sig * sig
        dw_dk = 2.0 * t_surf[:, None] * sig * dsig
        d2w_dk2 = 2.0 * t_surf[:, None] * (dsig * dsig + sig * d2sig)
    else:
        idx0 = torch.clamp(torch.searchsorted(tenors, t_surf, right=True) - 1, 0, n_exp - 2)
        idx1 = idx0 + 1
        rows = torch.arange(n_t, device=device)
        v0 = v_all[rows, :, idx0]
        dv0 = dv_all[rows, :, idx0]
        d2v0 = d2v_all[rows, :, idx0]
        v1 = v_all[rows, :, idx1]
        dv1 = dv_all[rows, :, idx1]
        d2v1 = d2v_all[rows, :, idx1]

        t0 = tenors[idx0][:, None]
        t1 = tenors[idx1][:, None]
        dt_inv = 1.0 / torch.clamp(t1 - t0, min=1e-12)
        alpha = (t_surf[:, None] - t0) * dt_inv

        w0 = (v0 * v0) * t0
        w1 = (v1 * v1) * t1
        w = (1.0 - alpha) * w0 + alpha * w1
        dw_dt = (w1 - w0) * dt_inv

        dw0_dk = 2.0 * t0 * v0 * dv0
        dw1_dk = 2.0 * t1 * v1 * dv1
        dw_dk = (1.0 - alpha) * dw0_dk + alpha * dw1_dk

        d2w0_dk2 = 2.0 * t0 * (dv0 * dv0 + v0 * d2v0)
        d2w1_dk2 = 2.0 * t1 * (dv1 * dv1 + v1 * d2v1)
        d2w_dk2 = (1.0 - alpha) * d2w0_dk2 + alpha * d2w1_dk2

        low_t = (t_surf <= tenors[0])[:, None]
        high_t = (t_surf >= tenors[-1])[:, None]
        sig_0, dsig_0, d2sig_0 = v_all[:, :, 0], dv_all[:, :, 0], d2v_all[:, :, 0]
        sig_e, dsig_e, d2sig_e = v_all[:, :, -1], dv_all[:, :, -1], d2v_all[:, :, -1]
        w = torch.where(low_t, (sig_0 * sig_0) * t_surf[:, None], torch.where(high_t, (sig_e * sig_e) * t_surf[:, None], w))
        dw_dt = torch.where(low_t, sig_0 * sig_0, torch.where(high_t, sig_e * sig_e, dw_dt))
        dw_dk = torch.where(
            low_t,
            2.0 * t_surf[:, None] * sig_0 * dsig_0,
            torch.where(high_t, 2.0 * t_surf[:, None] * sig_e * dsig_e, dw_dk),
        )
        d2w_dk2 = torch.where(
            low_t,
            2.0 * t_surf[:, None] * (dsig_0 * dsig_0 + sig_0 * d2sig_0),
            torch.where(high_t, 2.0 * t_surf[:, None] * (dsig_e * dsig_e + sig_e * d2sig_e), d2w_dk2),
        )

    w_safe = torch.clamp(w, min=1e-8)
    dw_dy = s_grid * dw_dk
    dw_dt_y = dw_dt + (r_zero_grid - q_repo_grid)[:, None] * dw_dy
    d2w_dy2 = (s_grid * s_grid) * d2w_dk2 + dw_dy
    y_mat = y_grid[None, :]
    denom = (
        1.0
        - (y_mat / w_safe) * dw_dy
        + 0.25 * (-0.25 - 1.0 / w_safe + (y_mat * y_mat) / (w_safe * w_safe)) * (dw_dy * dw_dy)
        + 0.5 * d2w_dy2
    )
    denom_safe = torch.clamp(denom, min=0.05)
    lv_dcc = torch.sqrt(torch.clamp(dw_dt_y / denom_safe, min=min_vol * min_vol))
    lv_365 = torch.clamp(lv_dcc * torch.sqrt(dt_surf_ratio)[:, None], min=min_vol, max=max_lv)
    return lv_365.to(torch.float32), y_min_val, inv_dy_val


def _qmc_index_code(index_name: str) -> int:
    """Maps any MonitoringIndex name or alias to canonical integer code 0..17."""
    return resolve_monitoring_index_code(index_name)


def _evaluate_monitoring_index_gpu(
    perf: torch.Tensor,
    index_name: str,
    inv_lex_perm_t: Optional[torch.Tensor] = None,
    *,
    asset_dim: int = -1,
) -> torch.Tensor:
    """Evaluates any of the 18 canonical Murex monitoring functionals along `asset_dim`."""
    code = resolve_monitoring_index_code(index_name)
    if code in _PAIR_ORDER_INDEX_CODES and inv_lex_perm_t is not None:
        p = perf.index_select(asset_dim, inv_lex_perm_t)
    else:
        p = perf

    d_len = int(p.shape[asset_dim])
    if code == int(MonitoringIndexCode.WORST_OF):
        return p.min(dim=asset_dim).values
    if code == int(MonitoringIndexCode.BEST_OF):
        return p.max(dim=asset_dim).values
    if code == int(MonitoringIndexCode.AVERAGE_OF):
        return p.mean(dim=asset_dim)
    if code == int(MonitoringIndexCode.DISPERSION):
        if d_len <= 1:
            return torch.zeros_like(p.select(asset_dim, 0))
        mean_p = p.mean(dim=asset_dim, keepdim=True)
        return torch.sqrt(((p - mean_p) ** 2).mean(dim=asset_dim).clamp(min=0.0))
    if code == int(MonitoringIndexCode.RATIO_BEST_WORST):
        return p.max(dim=asset_dim).values / p.min(dim=asset_dim).values.clamp(min=1e-12)
    if code == int(MonitoringIndexCode.RATIO_BEST_AVERAGE):
        return p.max(dim=asset_dim).values / p.mean(dim=asset_dim).clamp(min=1e-12)
    if code == int(MonitoringIndexCode.RATIO_AVERAGE_WORST):
        return p.mean(dim=asset_dim) / p.min(dim=asset_dim).values.clamp(min=1e-12)
    if code == int(MonitoringIndexCode.DIFF_BEST_WORST):
        return p.max(dim=asset_dim).values - p.min(dim=asset_dim).values
    if code == int(MonitoringIndexCode.DIFF_BEST_AVERAGE):
        return p.max(dim=asset_dim).values - p.mean(dim=asset_dim)
    if code == int(MonitoringIndexCode.DIFF_AVERAGE_WORST):
        return p.mean(dim=asset_dim) - p.min(dim=asset_dim).values
    if code == int(MonitoringIndexCode.RATIO_AB):
        p0 = p.select(asset_dim, 0)
        return p0 if d_len < 2 else (p0 / p.select(asset_dim, 1).clamp(min=1e-12))
    if code == int(MonitoringIndexCode.RATIO_BA):
        p0 = p.select(asset_dim, 0)
        return p0 if d_len < 2 else (p.select(asset_dim, 1) / p0.clamp(min=1e-12))
    if code == int(MonitoringIndexCode.DIFF_AB):
        if d_len < 2:
            return torch.zeros_like(p.select(asset_dim, 0))
        return p.select(asset_dim, 0) - p.select(asset_dim, 1)
    if code == int(MonitoringIndexCode.DIFF_BA):
        if d_len < 2:
            return torch.zeros_like(p.select(asset_dim, 0))
        return p.select(asset_dim, 1) - p.select(asset_dim, 0)
    if code == int(MonitoringIndexCode.TWIN_WIN):
        w_min = p.min(dim=asset_dim).values
        return torch.abs(w_min - 1.0) + 1.0
    if code == int(MonitoringIndexCode.SECOND_WORST_OF):
        k = min(1, max(d_len - 1, 0))
        return torch.kthvalue(p, k + 1, dim=asset_dim).values
    if code == int(MonitoringIndexCode.THIRD_WORST_OF):
        k = min(2, max(d_len - 1, 0))
        return torch.kthvalue(p, k + 1, dim=asset_dim).values
    if code == int(MonitoringIndexCode.SECOND_BEST_OF):
        k = max(0, d_len - 2)
        return torch.kthvalue(p, k + 1, dim=asset_dim).values
    return p.min(dim=asset_dim).values


def _digital_ramp_torch(
    w_vals: torch.Tensor,
    barrier: float,
    spread: float,
) -> torch.Tensor:
    """Call-spread digital ramp indicator in torch.float64."""
    if abs(spread) < 1e-9:
        return (w_vals >= barrier).to(w_vals.dtype)
    b_low = min(barrier, barrier + spread)
    b_high = max(barrier, barrier + spread)
    return torch.clamp((w_vals - b_low) / (b_high - b_low), 0.0, 1.0)


def _apply_loo_control_variate(
    y_rep: torch.Tensor,   # (R, N_rep) float64
    x_rep: torch.Tensor,   # (R, N_rep) float64
    mu_x: Union[float, torch.Tensor],
    se_mu_x: float = 0.0,
) -> Tuple[float, float]:
    """Unbiased leave-one-replicate-out control variate estimator across R replicates."""
    r_count = int(y_rep.shape[0])
    y_means = y_rep.mean(dim=1)  # (R,)
    x_means = x_rep.mean(dim=1)  # (R,)
    if r_count < 2:
        val = float(y_means.mean().item())
        se = float(y_rep.reshape(-1).std(unbiased=True).item() / math.sqrt(float(y_rep.numel())))
        return val, se

    # Precompute within-replicate centered sums of squares & cross-products for O(1) LOO beta_{-r}
    dy = y_rep - y_means.unsqueeze(1)
    dx = x_rep - x_means.unsqueeze(1)
    cov_xy_r = (dy * dx).sum(dim=1)  # (R,)
    var_xx_r = (dx * dx).sum(dim=1)  # (R,)
    tot_cov = cov_xy_r.sum()
    tot_var = var_xx_r.sum()

    mu_x_t = torch.as_tensor(mu_x, dtype=torch.float64, device=y_rep.device)
    loo_cov = tot_cov - cov_xy_r
    loo_var = tot_var - var_xx_r
    beta_loo = torch.where(loo_var > 1e-18, loo_cov / torch.clamp(loo_var, min=1e-18), torch.zeros_like(loo_cov))
    # Independent replicate estimate: E[y_means_r - beta_{-r} * (x_means_r - mu_x)] == E[y]
    v_cv_r = y_means - beta_loo * (x_means - mu_x_t)
    est = float(v_cv_r.mean().item())
    mean_beta = float(beta_loo.mean().item())
    se_cv_sq = float(v_cv_r.var(unbiased=True).item() / float(r_count))
    se = math.sqrt(max(se_cv_sq + (mean_beta * float(se_mu_x)) ** 2, 0.0))
    return est, se


def _black76_put_exact_f64(fwd: float, strike: float, vol_sqrt_t: float) -> float:
    """Exact Black-76 undiscounted European put price E[(K - P)^+]."""
    if strike <= 0.0:
        return 0.0
    if vol_sqrt_t <= 1e-12:
        return max(strike - fwd, 0.0)
    d1 = (math.log(max(fwd, 1e-12) / strike) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    inv_sqrt2 = 0.7071067811865475
    return float(strike * (0.5 * math.erfc(float(d2) * inv_sqrt2)) - fwd * (0.5 * math.erfc(float(d1) * inv_sqrt2)))


class CpuNumpyDupireQMC:
    """CPU IEEE-754 float64 Dupire QMC backend delegating to `dupire_qmc_solver`."""

    @property
    def backend_name(self) -> str:
        return "CpuNumpyDupireQMC"

    def simulate_and_price(
        self,
        state: AutocallPreprocState,
        config: Optional[DupireQMCConfig] = None,
    ) -> DupireQMCResult:
        if config is None:
            cfg = resolve_dupire_qmc_config(
                int(state.d_assets),
                int(state.m_rem),
                device_type="cpu",
            )
        else:
            cfg = DupireQMCConfig.from_canonical(config)
        return price_eq_autocall_dupire_qmc(
            state,
            n_paths=cfg.n_paths,
            n_replicates=cfg.n_replicates,
            mode=cfg.mode,
            n_substeps_per_period=cfg.n_substeps,
            scramble=True,
            seed=cfg.seed,
            device="cpu",
            use_gpu=False,
            precision="fp64_audit",
            cv_mode=cfg.cv_mode,
        )

    def simulate_batch_for_training(
        self,
        states: Sequence[AutocallPreprocState],
        config: Optional[DupireQMCConfig] = None,
        compute_pathwise_delta: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if len(states) == 0:
            z0 = torch.empty(0, dtype=torch.float64)
            return (
                z0,
                torch.empty((0, 48), dtype=torch.float64),
                torch.empty((0, 48), dtype=torch.float64),
                z0,
                z0.unsqueeze(-1) if compute_pathwise_delta else None,
            )
        if config is None:
            max_d = max(int(st.d_assets) for st in states)
            max_m = max(int(st.m_rem) for st in states)
            cfg = resolve_dupire_qmc_config(max_d, max_m, device_type="cpu")
        else:
            cfg = DupireQMCConfig.from_canonical(config)
        cpu_cfg = DupireQMCConfig(
            n_paths=int(cfg.n_paths),
            n_replicates=int(cfg.n_replicates),
            n_substeps=int(cfg.n_substeps),
            mode=cfg.mode,
            precision=cfg.precision,
            cv_mode=cfg.cv_mode,
            lv_grid_nt=min(int(cfg.lv_grid_nt), 16),
            lv_grid_ny=min(int(cfg.lv_grid_ny), 32),
            seed=int(cfg.seed),
            adaptive_substeps=bool(getattr(cfg, "adaptive_substeps", False)),
            n_cheap_mf=cfg.n_cheap_mf,
        )
        return GpuTorchDupireQMC(device="cpu", use_triton=False).simulate_batch_for_training(
            states,
            config=cpu_cfg,
            compute_pathwise_delta=compute_pathwise_delta,
        )


def _onestar_ramp_torch(
    w_best: torch.Tensor,
    k_os: Union[float, torch.Tensor],
    eps_os: Union[float, torch.Tensor] = 0.02,
) -> torch.Tensor:
    """Smooth step over `[K_os - 0.5*eps_os, K_os + 0.5*eps_os]`."""
    return torch.clamp((w_best - (k_os - 0.5 * eps_os)) / eps_os, 0.0, 1.0)


def _finalize_dupire_qmc_result(
    state: AutocallPreprocState,
    cfg: DupireQMCConfig,
    *,
    n_total_paths: int,
    p_stop_arr: npt.NDArray[np.float64],
    q_cpn_arr: npt.NDArray[np.float64],
    cpn_cf_expected: npt.NDArray[np.float64],
    df_pay_np: npt.NDArray[np.float64],
    l_put_val: float,
    l_call_val: float,
    l_onestar_val: float,
    p_onestar_val: float,
    raw_pv_mean: float,
    cv_pv_est: float,
    total_se: float,
    option_se: float,
) -> DupireQMCResult:
    asm = _assemble_qmc_postproc(
        state,
        p_stop_arr=p_stop_arr,
        q_cpn_arr=q_cpn_arr,
        cpn_cf_expected=cpn_cf_expected,
        df_pay=df_pay_np,
        l_put_val=l_put_val,
        l_call_val=l_call_val,
        l_onestar_val=l_onestar_val,
        p_onestar_val=p_onestar_val,
        option_se=option_se,
        total_se=total_se,
        assemble_fn=assemble_autocall_cashflows_and_pv,
    )

    cv_shift = cv_pv_est - raw_pv_mean
    adj_prod = float(asm.product_value.item()) + cv_shift
    is_long_flag = bool(getattr(state, "is_long_basket", False))
    has_onestar_bonus = (
        getattr(state, "one_star_strike", None) is not None
        and float(getattr(state, "one_star_strike", 0.0) or 0.0) > 0.0
        and float(getattr(state, "one_star_coupon", 0.0) or 0.0) != 0.0
    )
    has_zero_opt = (
        float(getattr(state, "option_gearing", 1.0)) == 0.0
        and float(getattr(state, "participation_call_rate", 0.0)) == 0.0
        and not is_long_flag
        and not has_onestar_bonus
    )
    no_notional = bool(getattr(state, "no_notional_repayment", False))
    shift_to_notional = has_zero_opt and not no_notional
    adj_opt = float(asm.option_pv.item()) if shift_to_notional else (float(asm.option_pv.item()) + cv_shift)
    adj_notional = (float(asm.notional_pv.item()) + cv_shift) if shift_to_notional else float(asm.notional_pv.item())
    if cv_shift != 0.0 and asm.cashflows:
        target_order: Tuple[str, ...]
        if (
            is_long_flag
            and any(cf.Description == "IsLongBasket" and cf.StatusText == "Estimate" for cf in asm.cashflows)
        ):
            target_order = ("IsLongBasket", "Option", "Notional")
        elif shift_to_notional:
            target_order = ("Notional", "Option", "Coupon")
        else:
            target_order = ("Option", "OneStarCoupon", "Coupon", "Notional")
        for target_desc in target_order:
            replaced = False
            for idx_cf in range(len(asm.cashflows) - 1, -1, -1):
                cf = asm.cashflows[idx_cf]
                if cf.Description == target_desc and cf.StatusText == "Estimate":
                    asm.cashflows[idx_cf] = type(cf)(
                        Description=cf.Description,
                        Date=cf.Date,
                        Amount=float(cf.Amount + cv_shift),
                        StandardError=float(option_se),
                        Currency=cf.Currency,
                        Leg=cf.Leg,
                        StatusText=cf.StatusText,
                    )
                    replaced = True
                    break
            if replaced:
                break

    return DupireQMCResult(
        product_value=adj_prod,
        standard_error=total_se,
        option_pv=adj_opt,
        option_se=option_se,
        notional_pv=adj_notional,
        coupon_pv=float(asm.coupon_pv.item()),
        funding_pv=float(asm.funding_pv.item()),
        p_stop=p_stop_arr,
        q_cpn=q_cpn_arr,
        l_put=l_put_val,
        l_call=l_call_val,
        cashflows=asm.cashflows,
        n_paths=n_total_paths,
        mode=cfg.mode,
        l_onestar=l_onestar_val,
        p_onestar=p_onestar_val,
    )


def _eval_continuous_ki_step_survival_torch(
    perf_prev: torch.Tensor,
    perf_curr: torch.Tensor,
    corr_64: torch.Tensor,
    lv_step: torch.Tensor,
    dt_step: float,
    eff_opt_idx: str,
    opt_code: int,
    ki_barrier: float,
    inv_perm_gpu: Optional[torch.Tensor],
) -> torch.Tensor:
    """Brownian-bridge non-knock-in survival probability over one step [t0, t1] in float64."""
    w_prev = _evaluate_monitoring_index_gpu(perf_prev, eff_opt_idx, inv_perm_gpu, asset_dim=-1)
    w_curr = _evaluate_monitoring_index_gpu(perf_curr, eff_opt_idx, inv_perm_gpu, asset_dim=-1)
    alive_pair = (w_prev > ki_barrier) & (w_curr > ki_barrier)
    if not bool(torch.any(alive_pair).item()):
        return torch.zeros_like(w_curr, dtype=torch.float64)
    perf_mid = 0.5 * (perf_prev + perf_curr)
    w_mid, dw_dp = _eval_index_and_grad_gpu(
        perf_mid, eff_opt_idx, inv_perm_gpu, compute_grad=True, asset_dim=-1
    )
    assert dw_dp is not None
    weighted = dw_dp * perf_mid * lv_step.to(torch.float64)
    var_abs = (weighted * torch.matmul(weighted, corr_64)).sum(dim=-1)
    if opt_code in _DIFF_INDEX_CODES or ki_barrier <= 0.0:
        eta = (
            2.0
            * torch.clamp(w_prev - ki_barrier, min=0.0)
            * torch.clamp(w_curr - ki_barrier, min=0.0)
            / torch.clamp(var_abs * max(dt_step, 1e-12), min=1e-12)
        )
    else:
        w_mid_s = torch.clamp(w_mid, min=1e-12)
        var_log = var_abs / (w_mid_s * w_mid_s)
        w_p_s = torch.where(
            alive_pair,
            torch.clamp(w_prev, min=ki_barrier + 1e-12),
            torch.full_like(w_prev, ki_barrier + 0.1),
        )
        w_c_s = torch.where(
            alive_pair,
            torch.clamp(w_curr, min=ki_barrier + 1e-12),
            torch.full_like(w_curr, ki_barrier + 0.1),
        )
        eta = (
            2.0
            * torch.log(w_p_s / ki_barrier)
            * torch.log(w_c_s / ki_barrier)
            / torch.clamp(var_log * max(dt_step, 1e-12), min=1e-12)
        )
    return torch.where(
        alive_pair, 1.0 - torch.exp(-torch.clamp(eta, min=0.0)), torch.zeros_like(eta)
    )


class GpuTorchDupireQMC:
    """GPU PyTorch CUDA Dupire RQMC backend (Triton-optional fallback)."""

    def __init__(self, device: Union[str, torch.device] = "cuda", *, use_triton: bool = False) -> None:
        self.device = torch.device(device)
        self._use_triton = bool(use_triton and _HAS_TRITON and self.device.type == "cuda")

    @property
    def backend_name(self) -> str:
        return "GpuTritonDupireQMC" if self._use_triton else "GpuTorchDupireQMC"

    def simulate_and_price(
        self,
        state: AutocallPreprocState,
        config: Optional[DupireQMCConfig] = None,
    ) -> DupireQMCResult:
        dev = self.device
        d_assets = int(state.d_assets)
        m_rem = int(state.m_rem)
        if config is None:
            cfg = resolve_dupire_qmc_config(
                d_assets,
                m_rem,
                device_type="cuda" if dev.type == "cuda" else "cpu",
            )
        else:
            cfg = DupireQMCConfig.from_canonical(config)

        if m_rem == 0 or bool(getattr(state, "already_autocalled", False)) or state.seasoned_state.already_autocalled:
            z_m = np.zeros(m_rem, dtype=np.float64)
            asm = assemble_autocall_cashflows_and_pv(
                state,
                p_stop=torch.zeros(m_rem, dtype=torch.float64),
                q_cpn=torch.zeros(m_rem, dtype=torch.float64),
                l_put=torch.tensor(0.0, dtype=torch.float64),
                option_se=0.0,
            )
            return DupireQMCResult(
                product_value=float(asm.product_value.item()),
                standard_error=0.0,
                option_pv=0.0,
                option_se=0.0,
                notional_pv=float(asm.notional_pv.item()),
                coupon_pv=float(asm.coupon_pv.item()),
                funding_pv=float(asm.funding_pv.item()),
                p_stop=z_m,
                q_cpn=z_m,
                l_put=0.0,
                l_call=0.0,
                cashflows=asm.cashflows,
                n_paths=cfg.n_paths,
                mode=cfg.mode,
                l_onestar=0.0,
            )

        eff_r = max(1, int(cfg.n_replicates))
        n_rep = _next_power_of_two(max(2, int(cfg.n_paths) // eff_r))
        n_total_paths = eff_r * n_rep
        eff_n_cheap = int(cfg.n_cheap_mf) if cfg.n_cheap_mf is not None else (4 * n_total_paths)
        n_sub_req = max(1, int(cfg.n_substeps))

        bundle = _get_or_build_static_schedule(
            state,
            dev,
            n_sub=n_sub_req,
            lv_grid_nt=cfg.lv_grid_nt,
            lv_grid_ny=cfg.lv_grid_ny,
            use_triton=self._use_triton,
        )
        n_sub = n_sub_req
        if cfg.n_substeps == 4 and bool(getattr(cfg, "adaptive_substeps", False)):
            if bundle.max_dt <= 0.15 and bundle.max_rel_skew < 0.25:
                n_sub = 2
            elif bundle.max_rel_skew >= 0.45 or bundle.max_dt >= 0.50:
                n_sub = 8
            else:
                n_sub = 4
            if n_sub != n_sub_req:
                _ensure_bundle_substeps(bundle, n_sub, dev)

        sched = bundle.schedule
        tau_obs_np = bundle.tau_obs_np
        df_pay_np = bundle.df_pay_np
        chol_f32 = bundle.chol_f32
        inv_perm_gpu = bundle.inv_lex_perm_gpu
        dt_sub_gpu, sqrt_dt_gpu, _ = bundle.substep_tensors[n_sub]
        dt_sub_tuple, sqrt_dt_tuple = bundle.substep_cpu_meta[n_sub]

        lv_table: Optional[torch.Tensor] = None
        y_min_gpu: Optional[torch.Tensor] = None
        inv_dy_gpu: Optional[torch.Tensor] = None
        tau_max = max(float(tau_obs_np[-1]), 1e-4)
        inv_dt_grid = float(cfg.lv_grid_nt - 1) / tau_max
        surfs_ordered = [state.vol_surfaces[t] for t in state.sorted_identifiers]
        div_events_map = getattr(state.dividend_schedule, "dividend_events", {})

        if cfg.mode == "dupire_lv":
            tau_obs_64_t = getattr(bundle, "_tau_obs_64", None)
            tau_arg = (
                tau_obs_64_t
                if tau_obs_64_t is not None
                else torch.as_tensor(tau_obs_np, dtype=torch.float64, device=dev)
            )
            lv_table, y_min_gpu, inv_dy_gpu = _build_lv_tables_batch(
                surfs_ordered,
                bundle.spots_t0,
                sched.r_nat_obs,
                sched.df_fin_obs,
                tau_arg,
                [div_events_map.get(ticker, []) for ticker in state.sorted_identifiers],
                bundle.v_term,
                cfg.lv_grid_nt,
                cfg.lv_grid_ny,
                dev,
                use_triton=self._use_triton,
                lv_digest=bundle.lv_digest,
            )

        eff_cpn_idx = state.coupon_index if state.coupon_index else state.callability_index
        eff_opt_idx, ki_opt_idx, is_long = _resolve_eff_and_ki_opt_indices(state)
        cpn_dir = str(getattr(state, "coupon_barrier_direction", "Up") or "Up").strip().lower()
        cpn_type = str(getattr(state, "coupon_barrier_type", "In") or "In").strip().lower()

        call_code = _qmc_index_code(state.callability_index)
        cpn_code = _qmc_index_code(eff_cpn_idx)
        opt_code = _qmc_index_code(eff_opt_idx)
        ki_opt_code = _qmc_index_code(ki_opt_idx)
        is_diff_opt = opt_code in _DIFF_INDEX_CODES

        ki_mode = _resolve_ki_obs_mode(state)
        has_ki = (
            state.ki_barrier is not None
            and (float(state.ki_barrier) > 1e-6 or (is_diff_opt and float(state.ki_barrier) < -1e-6))
            and float(state.seasoned_state.i_ki_past) < 0.5
        )
        is_discrete_ki = bool(has_ki and ki_mode in ("Discrete", "Daily"))
        cont_ki_active = bool(has_ki and ki_mode == "Continuous")

        has_ko_spread = abs(float(state.ko_spread)) >= 1e-9
        has_cpn_spread = abs(float(state.cpn_spread)) >= 1e-9
        has_ki_spread = (has_ki or is_long) and abs(float(state.ki_spread)) >= 1e-9
        has_put = ((float(state.strike) > 1e-6) or (is_diff_opt and float(state.strike) < -1e-6)) and float(state.option_gearing) != 0.0
        has_low_k = (
            has_put
            and state.lower_strike is not None
            and (float(state.lower_strike) > 1e-6 or (is_diff_opt and float(state.lower_strike) < -1e-6))
        )
        max_spread_put = (
            max(float(state.strike) - float(state.lower_strike), 0.0)
            if (has_low_k and state.lower_strike is not None)
            else 0.0
        )
        has_onestar = state.one_star_strike is not None and float(state.one_star_strike) > 0.0
        one_star_strike_raw = float(state.one_star_strike) if (has_onestar and state.one_star_strike is not None) else 0.0
        one_star_strike, one_star_eps = (
            _resolve_onestar_effective_params(
                one_star_strike_raw,
                float(state.ki_barrier) if state.ki_barrier is not None else None,
                float(state.ko_spread),
            )
            if has_onestar
            else (0.0, 0.02)
        )
        one_star_coupon = float(state.one_star_coupon)
        has_os_cpn = has_onestar and one_star_coupon != 0.0
        has_pcall = state.participation_call_strike is not None and float(state.participation_call_rate) != 0.0
        pcall_strike = float(state.participation_call_strike) if (has_pcall and state.participation_call_strike is not None) else 0.0
        pcall_rate = float(state.participation_call_rate) if has_pcall else 0.0
        need_disp = (call_code == 3) or (cpn_code == 3) or (opt_code == 3)
        need_order = (
            call_code in (15, 16, 17) or cpn_code in (15, 16, 17) or opt_code in (15, 16, 17)
        )

        ko_barriers_np = _cpu_f64_array(state.ko_barriers)
        is_callable_np = _cpu_f64_array(state.is_callable)
        cpn_barriers_np = _cpu_f64_array(state.cpn_barriers)
        is_guar_np = _cpu_f64_array(state.is_guaranteed_cpn)
        coupon_amounts_np = _cpu_f64_array(state.coupon_amounts)
        ref_spots_gpu = sched.ref_spots
        fwd_prices_gpu = sched.fwd_perf * ref_spots_gpu[:, None]
        w_fund_np = (
            _cpu_f64_array(sched.w_fund)
            if dev.type == "cpu"
            else (
                _compute_funding_weights_np(state)
                if getattr(state, "funding_leg", None) is not None
                else np.zeros(m_rem, dtype=np.float64)
            )
        )

        has_bs_cv = cfg.cv_mode == "bs1d_exact"
        has_mf_cv = cfg.cv_mode == "multifidelity_gbm"
        is_haar = bool(n_sub > 1 and ((n_sub & (n_sub - 1)) == 0))
        haar_levels = int(round(math.log2(n_sub))) if is_haar else 0
        sobol_dim_req = m_rem * d_assets * (n_sub if is_haar else (1 + n_sub))
        use_persistent_triton = bool(
            self._use_triton
            and _HAS_TRITON
            and dev.type == "cuda"
            and cfg.mode == "dupire_lv"
            and cfg.precision != "fp64_audit"
            and cfg.cv_mode in ("bs1d_exact", "multifidelity_gbm", "none")
            and not cont_ki_active
            and n_rep >= 16
            and n_rep <= 1024
            and (n_rep & (n_rep - 1)) == 0
            and (sobol_dim_req <= _SOBOL_MAX_DIM)
            and (eff_r * m_rem * d_assets * n_rep <= 8 * 1024 * 1024)
        )
        use_bb_pca_tiled = (eff_r * m_rem * d_assets * n_rep) <= (16 * 1024 * 1024)
        can_tiled_triton = bool(
            self._use_triton
            and _HAS_TRITON
            and dev.type == "cuda"
            and cfg.mode == "dupire_lv"
            and cfg.precision != "fp64_audit"
            and cfg.cv_mode in ("bs1d_exact", "multifidelity_gbm", "none")
            and not (has_mf_cv and not use_bb_pca_tiled)
            and not cont_ki_active
            and n_rep >= 16
            and (n_rep & (n_rep - 1)) == 0
            and (sobol_dim_req <= _SOBOL_MAX_DIM)
        )

        if use_persistent_triton:
            assert lv_table is not None and y_min_gpu is not None and inv_dy_gpu is not None
            sv_gpu = _get_sobol_direction_numbers(dev)
            shifts_c, shifts_s = _get_sobol_period_shifts(
                n_replicates=eff_r,
                d_assets=d_assets,
                m_rem=m_rem,
                n_sub=n_sub,
                seed=cfg.seed,
                device=dev,
            )
            m_pow = int(round(math.log2(n_rep)))
            block_nc = min(128, max(16, n_rep))
            n_tiles_c = int(triton.cdiv(n_rep, block_nc))
            low_bits_c = int(round(math.log2(block_nc)))

            xi_pca = torch.empty((eff_r, m_rem, d_assets, n_rep), dtype=torch.float32, device=dev)
            _fast_triton_launch(
                _sobol_pca_triton_kernel,
                (n_tiles_c, m_rem, eff_r),
                sv_gpu,
                shifts_c,
                bundle.a_pca_f32,
                xi_pca,
                m_rem,
                eff_r,
                n_rep,
                m_pow,
                D_ASSETS=d_assets,
                LOW_BITS=low_bits_c,
                BLOCK_N=block_nc,
                num_warps=4,
            )
            z_coarse_corr = (
                torch.matmul(bundle.m_bb_f32, xi_pca.view(eff_r, m_rem, d_assets * n_rep))
                .view(eff_r, m_rem, d_assets, n_rep)
                .contiguous()
            )
            del xi_pca

            inv_perm_arg = (
                inv_perm_gpu
                if inv_perm_gpu is not None
                else torch.empty(2, dtype=torch.int64, device=dev)
            )
            ws_lx = torch.empty((eff_r, d_assets, n_rep), dtype=torch.float32, device=dev)
            ws_lx_g = (
                torch.empty((eff_r, d_assets, n_rep), dtype=torch.float64, device=dev)
                if has_mf_cv
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            ws_zsub = (
                torch.empty((eff_r, d_assets, n_sub, n_rep), dtype=torch.float32, device=dev)
                if n_sub > 1
                else torch.empty(1, dtype=torch.float32, device=dev)
            )
            ws_haar = (
                torch.empty((eff_r, n_sub, n_rep), dtype=torch.float32, device=dev)
                if is_haar
                else torch.empty(1, dtype=torch.float32, device=dev)
            )
            ws_perf = (
                torch.empty((eff_r, d_assets, n_rep), dtype=torch.float64, device=dev)
                if (need_disp and d_assets > 1)
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            ws_perf_g = (
                torch.empty((eff_r, d_assets, n_rep), dtype=torch.float64, device=dev)
                if (has_mf_cv and need_disp and d_assets > 1)
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            d_var_gbm_gpu = (
                torch.as_tensor(bundle.d_var_gbm_np, dtype=torch.float64, device=dev)
                if has_mf_cv
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            period_tile_stats = torch.empty((m_rem, eff_r, 1, 4), dtype=torch.float64, device=dev)
            term_tile_stats = torch.empty((eff_r, 1, 11), dtype=torch.float64, device=dev)
            packed_out = torch.empty(8 + 3 * m_rem, dtype=torch.float64, device=dev)

            has_cv_arg = has_bs_cv or has_mf_cv
            has_cv_low = has_bs_cv and bundle.k_low > 0.0
            inv_n_rep = 1.0 / float(n_rep)
            inv_sqrt_sub = 1.0 / math.sqrt(float(n_sub))
            inv_n_sub = 1.0 / float(n_sub)
            _fast_triton_launch(
                _dupire_persistent_qmc_kernel,
                (eff_r,),
                sv_gpu,
                shifts_s,
                z_coarse_corr,
                chol_f32,
                lv_table,
                y_min_gpu,
                inv_dy_gpu,
                bundle.q_shift_bnd_f32,
                bundle.t_prev_f32,
                dt_sub_gpu,
                sqrt_dt_gpu,
                sched.fwd_perf,
                sched.df_pay,
                sched.ko_barriers,
                sched.is_callable,
                sched.cpn_barriers,
                sched.is_guar,
                sched.coupon_amounts,
                sched.w_fund,
                d_var_gbm_gpu,
                bundle.sqrt_dvar_dom_f64,
                inv_perm_arg,
                ws_lx,
                ws_lx_g,
                ws_zsub,
                ws_haar,
                ws_perf,
                ws_perf_g,
                period_tile_stats,
                term_tile_stats,
                m_rem,
                eff_r,
                n_rep,
                n_sub,
                haar_levels,
                m_pow,
                m_rem * d_assets,
                bundle.dom_idx,
                inv_dt_grid,
                inv_n_rep,
                inv_sqrt_sub,
                inv_n_sub,
                float(state.seasoned_state.c_mem_0),
                float(state.autocall_coupon),
                float(state.ko_spread),
                float(state.cpn_spread),
                float(state.strike),
                max_spread_put,
                float(state.option_gearing),
                float(state.ki_barrier) if (has_ki and state.ki_barrier is not None) else 0.0,
                float(state.ki_spread) if (has_ki or is_long) else 0.0,
                one_star_strike,
                one_star_eps,
                one_star_coupon,
                pcall_strike,
                pcall_rate,
                bundle.f_dom,
                bundle.v_dom_term,
                bundle.k_cv,
                bundle.k_low,
                D_ASSETS=d_assets,
                CALL_CODE=call_code,
                CPN_CODE=cpn_code,
                OPT_CODE=opt_code,
                IS_MEM=bool(state.is_memory),
                NO_NOTIONAL=bool(state.no_notional_repayment),
                FORCE_CPN_AC=bool(state.force_coupons_on_autocall),
                CPN_DIR_DOWN=cpn_dir == "down",
                CPN_TYPE_OUT=cpn_type == "out",
                HAS_KO_SPREAD=has_ko_spread,
                HAS_CPN_SPREAD=has_cpn_spread,
                HAS_KI=has_ki,
                HAS_KI_SPREAD=has_ki_spread,
                IS_DISCRETE_KI=is_discrete_ki,
                HAS_PUT=has_put,
                HAS_LOW_K=has_low_k,
                HAS_ONESTAR=has_onestar,
                HAS_OS_CPN=has_os_cpn,
                HAS_PCALL=has_pcall,
                IS_LONG_BASKET=is_long,
                HAS_CV=has_bs_cv,
                HAS_CV_LOW=has_cv_low,
                HAS_MF_CV=has_mf_cv,
                HAS_SUB=n_sub > 1,
                IS_HAAR=is_haar,
                HAS_INV_PERM=inv_perm_gpu is not None,
                NEED_DISP=need_disp,
                NEED_ORDER=need_order,
                N_t=cfg.lv_grid_nt,
                N_y=cfg.lv_grid_ny,
                BLOCK_N=n_rep,
                num_warps=8 if n_rep >= 512 else 4,
            )
            del z_coarse_corr, ws_lx, ws_lx_g, ws_zsub, ws_haar, ws_perf, ws_perf_g

            if has_mf_cv:
                eff_mu_cv, eff_se_mu_cv = self._eval_cheap_gbm_expectation(
                    state=state,
                    chol_f32=chol_f32,
                    tau_obs_np=tau_obs_np,
                    d_var_gbm_np=bundle.d_var_gbm_np,
                    df_pay_np=df_pay_np,
                    ko_barriers_np=ko_barriers_np,
                    is_callable_np=is_callable_np,
                    cpn_barriers_np=cpn_barriers_np,
                    is_guar_np=is_guar_np,
                    coupon_amounts_np=coupon_amounts_np,
                    fwd_prices_gpu=fwd_prices_gpu,
                    ref_spots_gpu=ref_spots_gpu,
                    w_fund_np=w_fund_np,
                    eff_cpn_idx=eff_cpn_idx,
                    eff_opt_idx=eff_opt_idx,
                    cpn_dir=cpn_dir,
                    cpn_type=cpn_type,
                    inv_perm_gpu=inv_perm_gpu,
                    n_cheap=eff_n_cheap,
                    seed=cfg.seed + 88888,
                    a_pca_f32=bundle.a_pca_f32,
                    m_bb_f32=bundle.m_bb_f32,
                    schedule_digest_hex=bundle.schedule.digest_hex,
                    sched_gpu_tuple=(
                        sched.df_pay,
                        sched.ko_barriers,
                        sched.is_callable,
                        sched.cpn_barriers,
                        sched.is_guar,
                        sched.coupon_amounts,
                        sched.w_fund,
                    ),
                )
            else:
                eff_mu_cv = bundle.mu_cv
                eff_se_mu_cv = 0.0

            r_pad = max(2, _next_power_of_two(eff_r))
            inv_tot = 1.0 / float(eff_r * n_rep)
            inv_r = 1.0 / float(eff_r)
            inv_rr1 = 1.0 / float(eff_r * max(eff_r - 1, 1))
            denom_se1 = float(max(n_rep - 1, 1) * n_rep)
            _fast_triton_launch(
                _finalize_qmc_pv_stats_kernel,
                (1,),
                period_tile_stats,
                term_tile_stats,
                packed_out,
                m_rem,
                eff_r,
                1,
                inv_n_rep,
                inv_tot,
                inv_r,
                inv_rr1,
                denom_se1,
                eff_mu_cv,
                eff_se_mu_cv,
                HAS_CV=has_cv_arg,
                EFF_R_GT_1=eff_r > 1,
                R_PAD=r_pad,
                num_warps=4,
            )
            packed_np = packed_out.cpu().numpy()
            return _finalize_dupire_qmc_result(
                state,
                cfg,
                n_total_paths=n_total_paths,
                p_stop_arr=np.ascontiguousarray(packed_np[8 : 8 + m_rem], dtype=np.float64),
                q_cpn_arr=np.ascontiguousarray(packed_np[8 + m_rem : 8 + 2 * m_rem], dtype=np.float64),
                cpn_cf_expected=np.ascontiguousarray(packed_np[8 + 2 * m_rem : 8 + 3 * m_rem], dtype=np.float64),
                df_pay_np=df_pay_np,
                l_put_val=float(packed_np[0]),
                l_call_val=float(packed_np[1]),
                l_onestar_val=float(packed_np[2]),
                p_onestar_val=float(packed_np[3]),
                raw_pv_mean=float(packed_np[4]),
                cv_pv_est=float(packed_np[5]),
                total_se=float(packed_np[6]),
                option_se=float(packed_np[7]),
            )

        if can_tiled_triton:
            assert lv_table is not None and y_min_gpu is not None and inv_dy_gpu is not None
            block_n = min(256, max(16, n_rep))
            n_tiles = int(triton.cdiv(n_rep, block_n))
            m_pow = int(round(math.log2(n_rep)))
            low_bits = int(round(math.log2(block_n)))
            r_pad = max(2, _next_power_of_two(eff_r))
            has_sub = n_sub > 1
            has_cv = has_bs_cv
            has_cv_arg = has_bs_cv or has_mf_cv
            has_cv_low = has_bs_cv and bundle.k_low > 0.0
            has_inv_perm = inv_perm_gpu is not None
            inv_perm_arg = (
                inv_perm_gpu
                if inv_perm_gpu is not None
                else torch.empty(2, dtype=torch.int64, device=dev)
            )

            sv_gpu = _get_sobol_direction_numbers(dev)
            shifts_c, shifts_s = _get_sobol_period_shifts(
                n_replicates=eff_r,
                d_assets=d_assets,
                m_rem=m_rem,
                n_sub=n_sub,
                seed=cfg.seed,
                device=dev,
            )

            if use_bb_pca_tiled:
                block_nc = min(128, max(16, n_rep))
                n_tiles_c = int(triton.cdiv(n_rep, block_nc))
                low_bits_c = int(round(math.log2(block_nc)))
                xi_pca = torch.empty((eff_r, m_rem, d_assets, n_rep), dtype=torch.float32, device=dev)
                _fast_triton_launch(
                    _sobol_pca_triton_kernel,
                    (n_tiles_c, m_rem, eff_r),
                    sv_gpu,
                    shifts_c,
                    bundle.a_pca_f32,
                    xi_pca,
                    m_rem,
                    eff_r,
                    n_rep,
                    m_pow,
                    D_ASSETS=d_assets,
                    LOW_BITS=low_bits_c,
                    BLOCK_N=block_nc,
                    num_warps=4,
                )
                z_coarse_buf = (
                    torch.matmul(bundle.m_bb_f32, xi_pca.view(eff_r, m_rem, d_assets * n_rep))
                    .view(eff_r, m_rem, d_assets, n_rep)
                    .contiguous()
                )
                del xi_pca
                z_sub_buf = (
                    torch.empty((eff_r, d_assets, n_sub, n_rep), dtype=torch.float32, device=dev)
                    if has_sub
                    else torch.empty(1, dtype=torch.float32, device=dev)
                )
            else:
                z_coarse_buf = torch.empty((eff_r, d_assets, n_rep), dtype=torch.float32, device=dev)
                z_sub_buf = torch.empty((eff_r, d_assets, n_sub, n_rep), dtype=torch.float32, device=dev)
            z_haar_buf = (
                torch.empty((eff_r, d_assets, n_sub, n_rep), dtype=torch.float32, device=dev)
                if (is_haar and has_sub)
                else torch.empty(1, dtype=torch.float32, device=dev)
            )

            lx_buf = torch.empty((eff_r, d_assets, n_rep), dtype=torch.float32, device=dev)
            ex_buf = torch.empty((eff_r, d_assets, n_rep), dtype=torch.float64, device=dev)
            tile_ex_sum = torch.empty((eff_r, d_assets, n_tiles), dtype=torch.float64, device=dev)
            alive_buf = torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
            unpaid_buf = (
                torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
                if state.is_memory
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            not_ki_buf = (
                torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
                if is_discrete_ki
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            total_pv_buf = torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
            z_flat_cum_buf = (
                torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
                if has_cv
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            d_var_gbm_gpu = (
                torch.as_tensor(bundle.d_var_gbm_np, dtype=torch.float64, device=dev)
                if has_mf_cv
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            lx_g_buf = (
                torch.empty((eff_r, d_assets, n_rep), dtype=torch.float64, device=dev)
                if has_mf_cv
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            perf_g_buf = (
                torch.empty((eff_r, d_assets, n_rep), dtype=torch.float64, device=dev)
                if (has_mf_cv and need_disp and d_assets > 1)
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            alive_g_buf = (
                torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
                if has_mf_cv
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            unpaid_g_buf = (
                torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
                if (has_mf_cv and state.is_memory)
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            pv_g_buf = (
                torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
                if has_mf_cv
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            not_ki_g_buf = (
                torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
                if (has_mf_cv and is_discrete_ki)
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            period_tile_stats = torch.empty((m_rem, eff_r, n_tiles, 4), dtype=torch.float64, device=dev)
            term_tile_stats = torch.empty((eff_r, n_tiles, 11), dtype=torch.float64, device=dev)
            packed_out = torch.empty(8 + 3 * m_rem, dtype=torch.float64, device=dev)

            inv_n_rep = 1.0 / float(n_rep)
            inv_sqrt_sub = 1.0 / math.sqrt(float(n_sub))
            inv_n_sub = 1.0 / float(n_sub)
            sub_dim_base = m_rem * d_assets
            t_prev_tuple = bundle.t_prev_tuple
            sdvar_tuple = bundle.sqrt_dvar_dom_tuple
            grid_tile_3d = (n_tiles, d_assets, eff_r)
            grid_tile_2d = (n_tiles, eff_r)

            for m in range(m_rem):
                if has_sub or (not use_bb_pca_tiled):
                    _fast_triton_launch(
                        _sobol_bridge_period_chunk_kernel,
                        grid_tile_3d,
                        sv_gpu,
                        shifts_c[m],
                        shifts_s[m],
                        z_coarse_buf,
                        z_sub_buf,
                        z_haar_buf,
                        m * d_assets,
                        sub_dim_base + m * (d_assets * n_sub),
                        m,
                        sub_dim_base,
                        d_assets,
                        n_sub,
                        haar_levels,
                        inv_sqrt_sub,
                        inv_n_sub,
                        n_rep,
                        m_pow,
                        HAS_SUB=has_sub,
                        IS_CORR_COARSE=use_bb_pca_tiled,
                        IS_HAAR=is_haar,
                        LOW_BITS=low_bits,
                        BLOCK_N=block_n,
                        num_warps=4,
                    )
                _fast_triton_launch(
                    _dupire_step_period_tile_kernel,
                    grid_tile_3d,
                    z_coarse_buf,
                    z_sub_buf,
                    chol_f32,
                    lv_table,
                    y_min_gpu,
                    inv_dy_gpu,
                    bundle.q_shift_bnd_f32[m],
                    lx_buf,
                    ex_buf,
                    tile_ex_sum,
                    z_flat_cum_buf,
                    m,
                    m_rem,
                    t_prev_tuple[m],
                    dt_sub_tuple[m],
                    sqrt_dt_tuple[m],
                    inv_dt_grid,
                    inv_n_sub,
                    inv_sqrt_sub,
                    inv_n_rep,
                    sdvar_tuple[m],
                    bundle.dom_idx,
                    d_assets,
                    n_rep,
                    n_sub,
                    n_tiles,
                    HAS_CV=has_cv,
                    HAS_SUB=has_sub,
                    IS_CORR_COARSE=use_bb_pca_tiled,
                    N_t=cfg.lv_grid_nt,
                    N_y=cfg.lv_grid_ny,
                    BLOCK_N=block_n,
                    num_warps=4,
                )
                _fast_triton_launch(
                    _autocall_payoff_period_tile_kernel,
                    grid_tile_2d,
                    ex_buf,
                    tile_ex_sum,
                    sched.fwd_perf,
                    sched.df_pay,
                    sched.ko_barriers,
                    sched.is_callable,
                    sched.cpn_barriers,
                    sched.is_guar,
                    sched.coupon_amounts,
                    sched.w_fund,
                    inv_perm_arg,
                    alive_buf,
                    unpaid_buf,
                    total_pv_buf,
                    not_ki_buf,
                    z_flat_cum_buf,
                    z_coarse_buf,
                    d_var_gbm_gpu,
                    lx_g_buf,
                    perf_g_buf,
                    alive_g_buf,
                    unpaid_g_buf,
                    pv_g_buf,
                    not_ki_g_buf,
                    period_tile_stats,
                    term_tile_stats,
                    m,
                    m_rem,
                    eff_r,
                    n_rep,
                    n_tiles,
                    inv_n_rep,
                    float(state.seasoned_state.c_mem_0),
                    float(state.autocall_coupon),
                    float(state.ko_spread),
                    float(state.cpn_spread),
                    float(state.strike),
                    max_spread_put,
                    float(state.option_gearing),
                    float(state.ki_barrier) if (has_ki and state.ki_barrier is not None) else 0.0,
                    float(state.ki_spread) if (has_ki or is_long) else 0.0,
                    one_star_strike,
                    one_star_eps,
                    one_star_coupon,
                    pcall_strike,
                    pcall_rate,
                    bundle.f_dom,
                    bundle.v_dom_term,
                    bundle.k_cv,
                    bundle.k_low,
                    D_ASSETS=d_assets,
                    CALL_CODE=call_code,
                    CPN_CODE=cpn_code,
                    OPT_CODE=opt_code,
                    IS_MEM=bool(state.is_memory),
                    NO_NOTIONAL=bool(state.no_notional_repayment),
                    FORCE_CPN_AC=bool(state.force_coupons_on_autocall),
                    CPN_DIR_DOWN=cpn_dir == "down",
                    CPN_TYPE_OUT=cpn_type == "out",
                    HAS_KO_SPREAD=has_ko_spread,
                    HAS_CPN_SPREAD=has_cpn_spread,
                    HAS_KI=has_ki,
                    HAS_KI_SPREAD=has_ki_spread,
                    IS_DISCRETE_KI=is_discrete_ki,
                    HAS_PUT=has_put,
                    HAS_LOW_K=has_low_k,
                    HAS_ONESTAR=has_onestar,
                    HAS_OS_CPN=has_os_cpn,
                    HAS_PCALL=has_pcall,
                    IS_LONG_BASKET=is_long,
                    HAS_CV=has_cv,
                    HAS_CV_LOW=has_cv_low,
                    HAS_MF_CV=has_mf_cv,
                    HAS_INV_PERM=has_inv_perm,
                    NEED_DISP=need_disp,
                    NEED_ORDER=need_order,
                    BLOCK_N=block_n,
                    num_warps=4,
                )

            del z_coarse_buf, z_sub_buf, z_haar_buf, lx_buf, ex_buf, tile_ex_sum, alive_buf, unpaid_buf, not_ki_buf, total_pv_buf
            del lx_g_buf, perf_g_buf, alive_g_buf, unpaid_g_buf, pv_g_buf, not_ki_g_buf

            if has_mf_cv:
                eff_mu_cv, eff_se_mu_cv = self._eval_cheap_gbm_expectation(
                    state=state,
                    chol_f32=chol_f32,
                    tau_obs_np=tau_obs_np,
                    d_var_gbm_np=bundle.d_var_gbm_np,
                    df_pay_np=df_pay_np,
                    ko_barriers_np=ko_barriers_np,
                    is_callable_np=is_callable_np,
                    cpn_barriers_np=cpn_barriers_np,
                    is_guar_np=is_guar_np,
                    coupon_amounts_np=coupon_amounts_np,
                    fwd_prices_gpu=fwd_prices_gpu,
                    ref_spots_gpu=ref_spots_gpu,
                    w_fund_np=w_fund_np,
                    eff_cpn_idx=eff_cpn_idx,
                    eff_opt_idx=eff_opt_idx,
                    cpn_dir=cpn_dir,
                    cpn_type=cpn_type,
                    inv_perm_gpu=inv_perm_gpu,
                    n_cheap=eff_n_cheap,
                    seed=cfg.seed + 88888,
                    a_pca_f32=bundle.a_pca_f32,
                    m_bb_f32=bundle.m_bb_f32,
                    schedule_digest_hex=bundle.schedule.digest_hex,
                    sched_gpu_tuple=(
                        sched.df_pay,
                        sched.ko_barriers,
                        sched.is_callable,
                        sched.cpn_barriers,
                        sched.is_guar,
                        sched.coupon_amounts,
                        sched.w_fund,
                    ),
                )
            else:
                eff_mu_cv = bundle.mu_cv
                eff_se_mu_cv = 0.0

            inv_tot = 1.0 / float(eff_r * n_rep)
            inv_r = 1.0 / float(eff_r)
            inv_rr1 = 1.0 / float(eff_r * max(eff_r - 1, 1))
            denom_se1 = float(max(n_rep - 1, 1) * n_rep)
            _fast_triton_launch(
                _finalize_qmc_pv_stats_kernel,
                (1,),
                period_tile_stats,
                term_tile_stats,
                packed_out,
                m_rem,
                eff_r,
                n_tiles,
                inv_n_rep,
                inv_tot,
                inv_r,
                inv_rr1,
                denom_se1,
                eff_mu_cv,
                eff_se_mu_cv,
                HAS_CV=has_cv_arg,
                EFF_R_GT_1=eff_r > 1,
                R_PAD=r_pad,
                num_warps=4,
            )

            packed_np = packed_out.cpu().numpy()
            return _finalize_dupire_qmc_result(
                state,
                cfg,
                n_total_paths=n_total_paths,
                p_stop_arr=np.ascontiguousarray(packed_np[8 : 8 + m_rem], dtype=np.float64),
                q_cpn_arr=np.ascontiguousarray(packed_np[8 + m_rem : 8 + 2 * m_rem], dtype=np.float64),
                cpn_cf_expected=np.ascontiguousarray(packed_np[8 + 2 * m_rem : 8 + 3 * m_rem], dtype=np.float64),
                df_pay_np=df_pay_np,
                l_put_val=float(packed_np[0]),
                l_call_val=float(packed_np[1]),
                l_onestar_val=float(packed_np[2]),
                p_onestar_val=float(packed_np[3]),
                raw_pv_mean=float(packed_np[4]),
                cv_pv_est=float(packed_np[5]),
                total_se=float(packed_np[6]),
                option_se=float(packed_np[7]),
            )

        # Fallback path for terminal_iv, fp64_audit, Continuous KI, high-dim Sobol, or CPU torch
        dom_idx = bundle.dom_idx
        v_dom_term = bundle.v_dom_term
        same_cpn_call = eff_cpn_idx == state.callability_index
        is_fp64_audit = cfg.precision == "fp64_audit"
        step_dtype = torch.float64 if is_fp64_audit else torch.float32
        lv_table_step = (
            lv_table.to(torch.float64)
            if (is_fp64_audit and lv_table is not None)
            else lv_table
        )
        corr_64 = torch.as_tensor(state.corr_matrix, dtype=torch.float64, device=dev) if cont_ki_active else None
        sub_dim_base = m_rem * d_assets
        use_bb_pca_fb = bool(
            (sub_dim_base <= _SOBOL_MAX_DIM) and (n_rep >= 16) and ((n_rep & (n_rep - 1)) == 0)
        )
        sub_in_sobol_fb = bool(
            use_bb_pca_fb and (m_rem * d_assets * (n_sub if is_haar else (1 + n_sub)) <= _SOBOL_MAX_DIM)
        )
        if use_bb_pca_fb:
            sv_gpu_fb = _get_sobol_direction_numbers(dev)
            shifts_c_fb, shifts_s_fb = _get_sobol_period_shifts(
                n_replicates=eff_r,
                d_assets=d_assets,
                m_rem=m_rem,
                n_sub=n_sub if sub_in_sobol_fb else 1,
                seed=cfg.seed,
                device=dev,
            )
            r_chunk_fb = max(1, min(eff_r, (8 * 1024 * 1024) // max(m_rem * d_assets * n_rep, 1)))
            if is_haar and (d_assets * m_rem * n_sub >= 128):
                r_chunk_fb = min(r_chunk_fb, 4)
        else:
            sv_gpu_fb = None
            shifts_c_fb = None
            shifts_s_fb = None
            r_chunk_fb = eff_r

        total_pv_paths = torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
        opt_payoff_paths = torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
        x_cv_all = (
            torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
            if has_bs_cv
            else None
        )
        pv_gbm_all = (
            torch.empty((eff_r, n_rep), dtype=torch.float64, device=dev)
            if has_mf_cv
            else None
        )
        p_stop_sum = torch.zeros(m_rem, dtype=torch.float64, device=dev)
        cpn_alive_sum = torch.zeros(m_rem, dtype=torch.float64, device=dev)
        surv_before_sum = torch.zeros(m_rem, dtype=torch.float64, device=dev)
        cpn_cf_sum = torch.zeros(m_rem, dtype=torch.float64, device=dev)
        unit_put_sum = torch.zeros((), dtype=torch.float64, device=dev)
        unit_call_sum = torch.zeros((), dtype=torch.float64, device=dev)
        onestar_sum = torch.zeros((), dtype=torch.float64, device=dev)
        p_onestar_sum = torch.zeros((), dtype=torch.float64, device=dev)

        ki_bar_val = float(state.ki_barrier) if state.ki_barrier is not None else 0.0
        ex_taus_fb: Optional[torch.Tensor] = None
        ex_pvs_fb: Optional[torch.Tensor] = None
        if cont_ki_active and cfg.mode != "terminal_iv":
            div_lists_fb = [div_events_map.get(ticker, []) for ticker in state.sorted_identifiers]
            max_div_fb = max((len(evs) for evs in div_lists_fb), default=0)
            if max_div_fb > 0:
                ex_t_np = np.zeros((d_assets, max_div_fb), dtype=np.float64)
                ex_p_np = np.zeros((d_assets, max_div_fb), dtype=np.float64)
                for a_i, evs in enumerate(div_lists_fb):
                    for j, ev in enumerate(evs):
                        ex_t_np[a_i, j] = float(getattr(ev, "tau_ex_act365", 0.0))
                        ex_p_np[a_i, j] = float(getattr(ev, "amount", 0.0)) * float(
                            getattr(ev, "pv_discount_weight", 0.0)
                        )
                ex_taus_fb = torch.tensor(ex_t_np, dtype=torch.float64, device=dev)
                ex_pvs_fb = torch.tensor(ex_p_np, dtype=torch.float64, device=dev)
        coarse_dim_base = 0
        anti_tiv_fb = bool(use_bb_pca_fb and (cfg.mode == "terminal_iv") and (n_rep >= 32) and ((n_rep % 2) == 0))
        n_gen_fb = (n_rep // 2) if anti_tiv_fb else n_rep
        m_pow_fb = int(round(math.log2(n_gen_fb))) if use_bb_pca_fb else 0
        block_nc_fb = min(128, max(16, n_gen_fb)) if use_bb_pca_fb else 16
        n_tiles_c_fb = int((n_gen_fb + block_nc_fb - 1) // block_nc_fb) if use_bb_pca_fb else 1
        low_bits_c_fb = int(round(math.log2(block_nc_fb))) if use_bb_pca_fb else 4

        for r0_fb in range(0, eff_r, r_chunk_fb):
            rc_fb = min(r_chunk_fb, eff_r - r0_fb)
            z_coarse_corr_blk: Optional[torch.Tensor] = None
            if use_bb_pca_fb:
                assert sv_gpu_fb is not None and shifts_c_fb is not None
                sh_c_blk = shifts_c_fb[:, r0_fb : r0_fb + rc_fb, :].contiguous()
                if self._use_triton and _HAS_TRITON and dev.type == "cuda":
                    xi_pca_fb = torch.empty((rc_fb, m_rem, d_assets, n_gen_fb), dtype=torch.float32, device=dev)
                    _fast_triton_launch(
                        _sobol_pca_triton_kernel,
                        (n_tiles_c_fb, m_rem, rc_fb),
                        sv_gpu_fb,
                        sh_c_blk,
                        bundle.a_pca_f32,
                        xi_pca_fb,
                        m_rem,
                        rc_fb,
                        n_gen_fb,
                        m_pow_fb,
                        D_ASSETS=d_assets,
                        LOW_BITS=low_bits_c_fb,
                        BLOCK_N=block_nc_fb,
                        num_warps=4,
                    )
                else:
                    sh_c_flat = sh_c_blk.permute(1, 0, 2).reshape(rc_fb, m_rem * d_assets)
                    xi_raw_fb = _sobol_from_shifts_torch(
                        sv_gpu=sv_gpu_fb,
                        shifts_gpu=sh_c_flat,
                        dim_offset=0,
                        n_rep=n_gen_fb,
                        device=dev,
                    ).view(rc_fb, m_rem, d_assets, n_gen_fb)
                    xi_pca_fb = torch.matmul(bundle.a_pca_f32, xi_raw_fb)
                    del xi_raw_fb
                z_half_fb = torch.matmul(
                    bundle.m_bb_f32, xi_pca_fb.view(rc_fb, m_rem, d_assets * n_gen_fb)
                ).view(rc_fb, m_rem, d_assets, n_gen_fb)
                z_coarse_corr_blk = (
                    torch.cat([z_half_fb, -z_half_fb], dim=-1)
                    if anti_tiv_fb
                    else z_half_fb
                )
                del xi_pca_fb, z_half_fb, sh_c_blk

            log_x = torch.zeros((rc_fb, d_assets, n_rep), dtype=step_dtype, device=dev)
            alive = torch.ones((rc_fb, n_rep), dtype=torch.float64, device=dev)
            unpaid_mem = torch.full((rc_fb, n_rep), float(state.seasoned_state.c_mem_0), dtype=torch.float64, device=dev)
            pv_blk = torch.zeros((rc_fb, n_rep), dtype=torch.float64, device=dev)
            z_flat_cum = torch.zeros((rc_fb, n_rep), dtype=torch.float64, device=dev)
            not_ki_disc = torch.ones((rc_fb, n_rep), dtype=torch.float64, device=dev)
            p_noki_cont = torch.ones((rc_fb, n_rep), dtype=torch.float64, device=dev) if cont_ki_active else None
            perf_sub_prev = (
                (bundle.spots_t0 / ref_spots_gpu).view(1, 1, d_assets).expand(rc_fb, n_rep, d_assets).to(torch.float64)
                if cont_ki_active
                else None
            )

            log_x_gbm: Optional[torch.Tensor] = None
            alive_gbm: Optional[torch.Tensor] = None
            unpaid_gbm: Optional[torch.Tensor] = None
            pv_gbm_fine: Optional[torch.Tensor] = None
            not_ki_gbm = torch.ones((rc_fb, n_rep), dtype=torch.float64, device=dev)
            if has_mf_cv:
                log_x_gbm = torch.zeros((rc_fb, d_assets, n_rep), dtype=torch.float64, device=dev)
                alive_gbm = torch.ones((rc_fb, n_rep), dtype=torch.float64, device=dev)
                unpaid_gbm = torch.full((rc_fb, n_rep), float(state.seasoned_state.c_mem_0), dtype=torch.float64, device=dev)
                pv_gbm_fine = torch.zeros((rc_fb, n_rep), dtype=torch.float64, device=dev)

            t_prev = 0.0
            perf_term = torch.empty((rc_fb, n_rep, d_assets), dtype=torch.float64, device=dev)
            perf_tg = perf_term

            for m in range(m_rem):
                t_curr = float(tau_obs_np[m])
                if z_coarse_corr_blk is not None:
                    z_coarse_corr = z_coarse_corr_blk[:, m]
                    z_coarse_indep: Optional[torch.Tensor] = None
                else:
                    z_coarse_indep = _generate_gpu_sobol_tile(
                        n_rep=n_rep,
                        n_replicates=rc_fb,
                        n_dims=d_assets,
                        dim_offset=coarse_dim_base + m * d_assets,
                        seed=cfg.seed,
                        device=dev,
                        use_triton=self._use_triton,
                    )
                    z_coarse_corr = torch.matmul(chol_f32, z_coarse_indep)

                if has_bs_cv:
                    z_flat_cum = z_flat_cum + bundle.sqrt_dvar_dom_tuple[m] * z_coarse_corr[:, dom_idx, :].to(torch.float64)

                if cfg.mode == "terminal_iv":
                    dt_m_ti = max(t_curr - t_prev, 1e-8)
                    d_var = bundle.d_var_gbm_np[:, m]
                    dv_t = torch.as_tensor(d_var, dtype=torch.float64, device=dev).view(1, d_assets, 1)
                    log_x_f64 = log_x.to(torch.float64) - 0.5 * dv_t + torch.sqrt(dv_t) * z_coarse_corr.to(torch.float64)
                    log_x = log_x_f64.to(step_dtype)
                    if p_noki_cont is not None and perf_sub_prev is not None and corr_64 is not None:
                        perf_curr_ti = (fwd_prices_gpu[:, m] / ref_spots_gpu).view(1, d_assets, 1) * torch.exp(log_x_f64)
                        perf_curr_ti = perf_curr_ti.permute(0, 2, 1)
                        lv_ti = torch.sqrt(dv_t / dt_m_ti).permute(0, 2, 1).expand(rc_fb, n_rep, d_assets)
                        p_noki_cont = p_noki_cont * _eval_continuous_ki_step_survival_torch(
                            perf_sub_prev, perf_curr_ti, corr_64, lv_ti, dt_m_ti, ki_opt_idx, ki_opt_code, ki_bar_val, inv_perm_gpu
                        )
                        perf_sub_prev = perf_curr_ti
                else:
                    if n_sub == 1:
                        z_steps_corr = z_coarse_corr.unsqueeze(2).contiguous()
                    else:
                        sub_off_m = sub_dim_base + m * (d_assets * n_sub)
                        if sub_in_sobol_fb and is_haar and sv_gpu_fb is not None and shifts_s_fb is not None:
                            g_dims_m = _haar_global_dims_for_period(m_rem, d_assets, n_sub, m)
                            sh_s_m = shifts_s_fb[m, r0_fb : r0_fb + rc_fb, : d_assets * (n_sub - 1)].contiguous()
                            dim_idx_t = torch.as_tensor(g_dims_m, dtype=torch.long, device=dev)
                            b_modes_flat = _sobol_from_shifts_torch(
                                sv_gpu=sv_gpu_fb,
                                shifts_gpu=sh_s_m,
                                dim_offset=0,
                                n_rep=n_rep,
                                device=dev,
                                dim_indices=dim_idx_t,
                            )
                            z_sub_centered = _haar_reconstruct_residuals_torch(
                                b_modes_flat.view(rc_fb, d_assets, n_sub - 1, n_rep), n_sub
                            )
                            del sh_s_m, dim_idx_t, b_modes_flat
                        else:
                            if sub_in_sobol_fb and sv_gpu_fb is not None and shifts_s_fb is not None:
                                sh_s_m = shifts_s_fb[m, r0_fb : r0_fb + rc_fb, :].contiguous()
                                if self._use_triton and _HAS_TRITON and dev.type == "cuda":
                                    z_sub_flat = torch.empty((rc_fb, d_assets * n_sub, n_rep), dtype=torch.float32, device=dev)
                                    block_ns = min(256, max(16, _next_power_of_two(n_rep)))
                                    _fast_triton_launch(
                                        _sobol_normals_triton_kernel,
                                        (triton.cdiv(n_rep, block_ns), d_assets * n_sub, rc_fb),
                                        sv_gpu_fb,
                                        sh_s_m,
                                        z_sub_flat,
                                        sub_off_m,
                                        d_assets * n_sub,
                                        n_rep,
                                        m_pow_fb,
                                        BLOCK_N=block_ns,
                                    )
                                else:
                                    z_sub_flat = _sobol_from_shifts_torch(
                                        sv_gpu=sv_gpu_fb,
                                        shifts_gpu=sh_s_m,
                                        dim_offset=sub_off_m,
                                        n_rep=n_rep,
                                        device=dev,
                                    )
                                z_sub_raw = z_sub_flat.view(rc_fb, d_assets, n_sub, n_rep)
                                del sh_s_m, z_sub_flat
                            else:
                                z_sub_raw = _generate_gpu_sobol_tile(
                                    n_rep=n_rep,
                                    n_replicates=rc_fb,
                                    n_dims=d_assets * n_sub,
                                    dim_offset=sub_off_m,
                                    seed=cfg.seed + 31337 + r0_fb * 997,
                                    device=dev,
                                    use_triton=self._use_triton,
                                ).view(rc_fb, d_assets, n_sub, n_rep)
                            z_sub_centered = z_sub_raw - z_sub_raw.mean(dim=2, keepdim=True)
                            del z_sub_raw
                        z_steps_corr = torch.matmul(
                            chol_f32, z_sub_centered.reshape(rc_fb, d_assets, n_sub * n_rep)
                        ).view(rc_fb, d_assets, n_sub, n_rep)
                        if z_coarse_corr_blk is not None:
                            z_steps_corr = z_steps_corr + z_coarse_corr.unsqueeze(2) / math.sqrt(float(n_sub))
                        else:
                            assert z_coarse_indep is not None
                            z_steps_corr = z_steps_corr + torch.matmul(
                                chol_f32, z_coarse_indep
                            ).unsqueeze(2) / math.sqrt(float(n_sub))
                        del z_sub_centered

                    dt_sub = dt_sub_tuple[m]
                    sqrt_dt_sub = sqrt_dt_tuple[m]
                    q_bnd_m = bundle.q_shift_bnd_f32[m]
                    assert lv_table_step is not None and y_min_gpu is not None and inv_dy_gpu is not None

                    if self._use_triton and not is_fp64_audit and not cont_ki_active:
                        block_n = min(256, max(16, _next_power_of_two(n_rep)))
                        grid_sub = (triton.cdiv(n_rep, block_n), d_assets, rc_fb)
                        _fast_triton_launch(
                            _dupire_substep_triton_kernel,
                            grid_sub,
                            log_x,
                            z_steps_corr,
                            lv_table_step,
                            y_min_gpu,
                            inv_dy_gpu,
                            q_bnd_m,
                            float(t_prev),
                            float(dt_sub),
                            float(sqrt_dt_sub),
                            float(inv_dt_grid),
                            d_assets,
                            n_rep=n_rep,
                            n_sub=n_sub,
                            N_t=cfg.lv_grid_nt,
                            N_y=cfg.lv_grid_ny,
                            BLOCK_N=block_n,
                        )
                    else:
                        max_t_f = float(cfg.lv_grid_nt - 1) - 1e-4
                        max_y_f = float(cfg.lv_grid_ny - 1) - 1e-4
                        d_idx = torch.arange(d_assets, device=dev).view(1, d_assets, 1)
                        ym_v = y_min_gpu.to(step_dtype).view(1, d_assets, 1)
                        idy_v = inv_dy_gpu.to(step_dtype).view(1, d_assets, 1)
                        q0_v = q_bnd_m[:, 0].to(step_dtype).view(1, d_assets, 1)
                        q1_v = q_bnd_m[:, 1].to(step_dtype).view(1, d_assets, 1)
                        f_curr_s = fwd_prices_gpu[:, m].to(torch.float64)
                        r_tot_0_fb = (
                            torch.zeros(d_assets, dtype=torch.float64, device=dev)
                            if m == 0
                            else sched.r_tot_obs[:, m - 1]
                        )
                        r_tot_1_fb = sched.r_tot_obs[:, m]
                        spots_fb_64 = bundle.spots_t0.to(torch.float64)
                        floor_fb_64 = 0.01 * spots_fb_64
                        scale_fb_64 = torch.clamp(floor_fb_64, min=1e-12)

                        def _eval_lv_pt(t_q: float, lx_q: torch.Tensor) -> torch.Tensor:
                            ut = min(max(t_q * inv_dt_grid, 0.0), max_t_f)
                            it0 = int(ut)
                            ft = ut - float(it0)
                            it1 = min(it0 + 1, cfg.lv_grid_nt - 1)
                            uy = torch.clamp((lx_q - ym_v) * idy_v, 0.0, max_y_f)
                            iy0 = uy.to(torch.long)
                            fy = uy - iy0.to(step_dtype)
                            iy1 = torch.clamp(iy0 + 1, max=cfg.lv_grid_ny - 1)
                            v00 = lv_table_step[d_idx, it0, iy0]
                            v01 = lv_table_step[d_idx, it0, iy1]
                            v10 = lv_table_step[d_idx, it1, iy0]
                            v11 = lv_table_step[d_idx, it1, iy1]
                            return (1.0 - ft) * (v00 + fy * (v01 - v00)) + ft * (v10 + fy * (v11 - v10))

                        for s in range(n_sub):
                            a0 = float(s) / float(n_sub)
                            a1 = float(s + 1) / float(n_sub)
                            t_s0 = t_prev + float(s) * dt_sub
                            t_s1 = t_prev + float(s + 1) * dt_sub
                            q_sh0 = q0_v + a0 * (q1_v - q0_v)
                            q_sh1 = q0_v + a1 * (q1_v - q0_v)

                            lv0 = _eval_lv_pt(t_s0, log_x + q_sh0)
                            mu0 = -0.5 * (lv0 * lv0)
                            y_h = log_x + mu0 * dt_sub
                            sig_step = lv0 * sqrt_dt_sub
                            lv_p = _eval_lv_pt(t_s1, y_h + sig_step + q_sh1)
                            lv_m = _eval_lv_pt(t_s1, y_h - sig_step + q_sh1)
                            mu_p = -0.5 * (lv_p * lv_p)
                            mu_m = -0.5 * (lv_m * lv_m)
                            z_s = z_steps_corr[:, :, s, :].to(step_dtype)
                            log_x = log_x + (
                                0.5 * (mu0 + 0.5 * (mu_p + mu_m)) * dt_sub
                                + 0.25 * (mu_p - mu_m) * dt_sub * z_s
                                + 0.25 * (lv_p + lv_m + 2.0 * lv0) * sqrt_dt_sub * z_s
                                + 0.25 * (lv_p - lv_m) * (z_s * z_s - 1.0) * sqrt_dt_sub
                            )
                            if p_noki_cont is not None and perf_sub_prev is not None and corr_64 is not None:
                                if s == n_sub - 1:
                                    f_s1 = f_curr_s
                                else:
                                    r_tot_s1 = r_tot_0_fb + a1 * (r_tot_1_fb - r_tot_0_fb)
                                    if ex_taus_fb is not None and ex_pvs_fb is not None:
                                        mask_ex_s1 = (ex_taus_fb > 0.0) & (ex_taus_fb <= t_s1 + 1e-12)
                                        d_pv_s1 = (mask_ex_s1.to(torch.float64) * ex_pvs_fb).sum(dim=1)
                                    else:
                                        d_pv_s1 = torch.zeros(d_assets, dtype=torch.float64, device=dev)
                                    esc_s1 = floor_fb_64 + scale_fb_64 * F.softplus(
                                        (spots_fb_64 - d_pv_s1 - floor_fb_64) / scale_fb_64,
                                        beta=1.0,
                                        threshold=40.0,
                                    )
                                    f_s1 = torch.clamp(esc_s1 * torch.exp(r_tot_s1), min=1e-12)
                                perf_sub_curr = (f_s1 / ref_spots_gpu.to(torch.float64)).view(1, d_assets, 1) * torch.exp(log_x.to(torch.float64))
                                perf_sub_curr = perf_sub_curr.permute(0, 2, 1)
                                p_noki_cont = p_noki_cont * _eval_continuous_ki_step_survival_torch(
                                    perf_sub_prev,
                                    perf_sub_curr,
                                    corr_64,
                                    lv0.to(torch.float64).permute(0, 2, 1),
                                    dt_sub,
                                    ki_opt_idx,
                                    ki_opt_code,
                                    ki_bar_val,
                                    inv_perm_gpu,
                                )
                                perf_sub_prev = perf_sub_curr
                    del z_steps_corr

                if z_coarse_indep is not None:
                    del z_coarse_indep
                log_x_f64 = log_x.to(torch.float64)
                mean_x = torch.exp(log_x_f64).mean(dim=-1, keepdim=True)
                log_shift = torch.log(torch.clamp(mean_x, min=1e-12))
                log_x_f64 = log_x_f64 - log_shift
                log_x = log_x_f64.to(step_dtype)

                perf_m = (sched.fwd_perf[:, m].view(1, d_assets, 1) * torch.exp(log_x_f64)).permute(0, 2, 1)
                w_call = _evaluate_monitoring_index_gpu(perf_m, state.callability_index, inv_perm_gpu)
                w_cpn = w_call if same_cpn_call else _evaluate_monitoring_index_gpu(perf_m, eff_cpn_idx, inv_perm_gpu)
                if is_discrete_ki and state.ki_barrier is not None:
                    w_opt_m = _evaluate_monitoring_index_gpu(perf_m, ki_opt_idx, inv_perm_gpu)
                    not_ki_disc = not_ki_disc * _digital_ramp_torch(w_opt_m, float(state.ki_barrier), float(state.ki_spread))

                if is_callable_np[m] > 0.5:
                    ko_ind = _digital_ramp_torch(w_call, float(ko_barriers_np[m]), float(state.ko_spread))
                else:
                    ko_ind = torch.zeros((rc_fb, n_rep), dtype=torch.float64, device=dev)

                if is_guar_np[m] > 0.5:
                    cpn_ind = torch.ones((rc_fb, n_rep), dtype=torch.float64, device=dev)
                else:
                    b_cpn_m = float(cpn_barriers_np[m])
                    s_cpn = float(state.cpn_spread)
                    if cpn_dir == "down":
                        base_cpn = 1.0 - _digital_ramp_torch(w_cpn, b_cpn_m, -s_cpn)
                    else:
                        base_cpn = _digital_ramp_torch(w_cpn, b_cpn_m, s_cpn)
                    cpn_ind = (1.0 - base_cpn) if cpn_type == "out" else base_cpn

                if state.force_coupons_on_autocall:
                    cpn_ind = torch.maximum(cpn_ind, ko_ind)

                if w_fund_np[m] != 0.0:
                    pv_blk = pv_blk - alive * float(w_fund_np[m])

                stop_m = alive * ko_ind
                p_stop_sum[m] = p_stop_sum[m] + stop_m.sum()
                surv_before_sum[m] = surv_before_sum[m] + alive.sum()
                cpn_alive_sum[m] = cpn_alive_sum[m] + (alive * cpn_ind).sum()

                c_m = float(coupon_amounts_np[m])
                eff_c_paths = (c_m + unpaid_mem) if state.is_memory else c_m
                cpn_undisc_m = alive * cpn_ind * eff_c_paths + stop_m * float(state.autocall_coupon)
                cpn_cf_sum[m] = cpn_cf_sum[m] + cpn_undisc_m.sum()
                pv_blk = pv_blk + float(df_pay_np[m]) * cpn_undisc_m

                if state.is_memory:
                    unpaid_mem = alive * (unpaid_mem + c_m) * (1.0 - cpn_ind)

                alive = alive * (1.0 - ko_ind)
                if not state.no_notional_repayment:
                    if m < m_rem - 1:
                        pv_blk = pv_blk + float(df_pay_np[m]) * stop_m
                    else:
                        pv_blk = pv_blk + float(df_pay_np[m]) * (stop_m + alive)

                if (
                    has_mf_cv
                    and log_x_gbm is not None
                    and alive_gbm is not None
                    and unpaid_gbm is not None
                    and pv_gbm_fine is not None
                ):
                    dv_g = bundle.d_var_gbm_np[:, m]
                    dv_gt = torch.as_tensor(dv_g, dtype=torch.float64, device=dev).view(1, d_assets, 1)
                    log_x_gbm = log_x_gbm - 0.5 * dv_gt + torch.sqrt(dv_gt) * z_coarse_corr.to(torch.float64)
                    perf_mg = (sched.fwd_perf[:, m].view(1, d_assets, 1) * torch.exp(log_x_gbm)).permute(0, 2, 1)
                    wc_g = _evaluate_monitoring_index_gpu(perf_mg, state.callability_index, inv_perm_gpu)
                    wq_g = wc_g if same_cpn_call else _evaluate_monitoring_index_gpu(perf_mg, eff_cpn_idx, inv_perm_gpu)
                    if is_discrete_ki and state.ki_barrier is not None:
                        w_og_m = _evaluate_monitoring_index_gpu(perf_mg, ki_opt_idx, inv_perm_gpu)
                        not_ki_gbm = not_ki_gbm * _digital_ramp_torch(w_og_m, float(state.ki_barrier), float(state.ki_spread))
                    ko_g = _digital_ramp_torch(wc_g, float(ko_barriers_np[m]), float(state.ko_spread)) if is_callable_np[m] > 0.5 else torch.zeros_like(alive_gbm)
                    if is_guar_np[m] > 0.5:
                        cp_g = torch.ones_like(alive_gbm)
                    else:
                        b_cpn_m = float(cpn_barriers_np[m])
                        s_cpn = float(state.cpn_spread)
                        bc_g = (1.0 - _digital_ramp_torch(wq_g, b_cpn_m, -s_cpn)) if cpn_dir == "down" else _digital_ramp_torch(wq_g, b_cpn_m, s_cpn)
                        cp_g = (1.0 - bc_g) if cpn_type == "out" else bc_g
                    if state.force_coupons_on_autocall:
                        cp_g = torch.maximum(cp_g, ko_g)
                    if w_fund_np[m] != 0.0:
                        pv_gbm_fine = pv_gbm_fine - alive_gbm * float(w_fund_np[m])
                    st_g = alive_gbm * ko_g
                    ec_g = (c_m + unpaid_gbm) if state.is_memory else c_m
                    pv_gbm_fine = pv_gbm_fine + float(df_pay_np[m]) * (alive_gbm * cp_g * ec_g + st_g * float(state.autocall_coupon))
                    if state.is_memory:
                        unpaid_gbm = alive_gbm * (unpaid_gbm + c_m) * (1.0 - cp_g)
                    alive_gbm = alive_gbm * (1.0 - ko_g)
                    if not state.no_notional_repayment:
                        pv_gbm_fine = pv_gbm_fine + float(df_pay_np[m]) * (st_g if m < m_rem - 1 else (st_g + alive_gbm))
                    if m == m_rem - 1:
                        perf_tg = perf_mg

                if m == m_rem - 1:
                    perf_term = perf_m

                t_prev = t_curr

            if z_coarse_corr_blk is not None:
                del z_coarse_corr_blk

            w_opt = _evaluate_monitoring_index_gpu(perf_term, eff_opt_idx, inv_perm_gpu)
            star_ind: Optional[torch.Tensor] = None
            if has_onestar:
                w_best = _evaluate_monitoring_index_gpu(perf_term, "BestOf", inv_perm_gpu)
                star_ind = _onestar_ramp_torch(w_best, one_star_strike, one_star_eps)

            if is_long:
                avg_t = _evaluate_monitoring_index_gpu(perf_term, "AverageOf", inv_perm_gpu)
                wof_t = _evaluate_monitoring_index_gpu(perf_term, "WorstOf", inv_perm_gpu)
                if has_ki and state.ki_barrier is not None:
                    if p_noki_cont is not None:
                        ki_wof_k = 1.0 - p_noki_cont
                    elif is_discrete_ki:
                        ki_wof_k = 1.0 - not_ki_disc
                    else:
                        ki_wof_k = 1.0 - _digital_ramp_torch(wof_t, float(state.strike), float(state.ki_spread))
                else:
                    ki_wof_k = 1.0 - _digital_ramp_torch(wof_t, float(state.strike), float(state.ki_spread))
                eff_gearing = float(state.option_gearing) if abs(float(state.option_gearing)) > 1e-12 else 1.0
                put_int = ((1.0 - avg_t) * ki_wof_k) / eff_gearing
                if star_ind is not None:
                    put_int = put_int * (1.0 - star_ind)
                unit_put_blk = alive * put_int
            elif has_put:
                put_int = torch.clamp(float(state.strike) - w_opt, min=0.0)
                if has_low_k:
                    put_int = torch.clamp(put_int, max=max_spread_put)
                if has_ki and state.ki_barrier is not None:
                    if p_noki_cont is not None:
                        ki_ind = 1.0 - p_noki_cont
                    elif is_discrete_ki:
                        ki_ind = 1.0 - not_ki_disc
                    else:
                        ki_ind = 1.0 - _digital_ramp_torch(w_opt, float(state.ki_barrier), float(state.ki_spread))
                    put_int = put_int * ki_ind
                if star_ind is not None:
                    put_int = put_int * (1.0 - star_ind)
                unit_put_blk = alive * put_int
            else:
                unit_put_blk = torch.zeros((rc_fb, n_rep), dtype=torch.float64, device=dev)

            if has_pcall:
                unit_call_blk = alive * torch.clamp(w_opt - pcall_strike, min=0.0)
            else:
                unit_call_blk = torch.zeros((rc_fb, n_rep), dtype=torch.float64, device=dev)

            unit_put_sum = unit_put_sum + unit_put_blk.sum()
            unit_call_sum = unit_call_sum + unit_call_blk.sum()
            if star_ind is not None:
                p_onestar_sum = p_onestar_sum + (alive * star_ind).sum()

            put_sign = -1.0
            opt_blk = float(df_pay_np[-1]) * (
                put_sign * float(state.option_gearing) * unit_put_blk
                + float(state.participation_call_rate) * unit_call_blk
            )
            if star_ind is not None and one_star_coupon != 0.0:
                onestar_blk = alive * star_ind * one_star_coupon
                opt_blk = opt_blk + float(df_pay_np[-1]) * onestar_blk
                onestar_sum = onestar_sum + onestar_blk.sum()

            opt_payoff_paths[r0_fb : r0_fb + rc_fb] = opt_blk
            total_pv_paths[r0_fb : r0_fb + rc_fb] = pv_blk + opt_blk

            if x_cv_all is not None:
                p_dom = bundle.f_dom * torch.exp(-0.5 * (v_dom_term * v_dom_term) + z_flat_cum)
                x_blk = float(df_pay_np[-1]) * torch.clamp(bundle.k_cv - p_dom, min=0.0)
                if bundle.k_low > 0.0:
                    x_blk = x_blk - float(df_pay_np[-1]) * torch.clamp(bundle.k_low - p_dom, min=0.0)
                x_cv_all[r0_fb : r0_fb + rc_fb] = x_blk
            elif pv_gbm_all is not None and pv_gbm_fine is not None and alive_gbm is not None:
                w_og = _evaluate_monitoring_index_gpu(perf_tg, eff_opt_idx, inv_perm_gpu)
                st_g_term: Optional[torch.Tensor] = None
                if has_onestar:
                    st_g_term = _onestar_ramp_torch(
                        _evaluate_monitoring_index_gpu(perf_tg, "BestOf", inv_perm_gpu),
                        one_star_strike,
                        one_star_eps,
                    )
                if is_long:
                    avg_tg = _evaluate_monitoring_index_gpu(perf_tg, "AverageOf", inv_perm_gpu)
                    wof_tg = _evaluate_monitoring_index_gpu(perf_tg, "WorstOf", inv_perm_gpu)
                    if has_ki and state.ki_barrier is not None and is_discrete_ki:
                        ki_wof_kg = 1.0 - not_ki_gbm
                    else:
                        ki_wof_kg = 1.0 - _digital_ramp_torch(wof_tg, float(state.strike), float(state.ki_spread))
                    eff_gearing_g = float(state.option_gearing) if abs(float(state.option_gearing)) > 1e-12 else 1.0
                    pi_g = ((1.0 - avg_tg) * ki_wof_kg) / eff_gearing_g
                    if st_g_term is not None:
                        pi_g = pi_g * (1.0 - st_g_term)
                    up_g = alive_gbm * pi_g
                elif has_put:
                    pi_g = torch.clamp(float(state.strike) - w_og, min=0.0)
                    if has_low_k:
                        pi_g = torch.clamp(pi_g, max=max_spread_put)
                    if has_ki and state.ki_barrier is not None:
                        ki_g_ind = (
                            1.0 - not_ki_gbm
                            if is_discrete_ki
                            else (1.0 - _digital_ramp_torch(w_og, float(state.ki_barrier), float(state.ki_spread)))
                        )
                        pi_g = pi_g * ki_g_ind
                    if st_g_term is not None:
                        pi_g = pi_g * (1.0 - st_g_term)
                    up_g = alive_gbm * pi_g
                else:
                    up_g = torch.zeros_like(alive_gbm)
                uc_g = (
                    alive_gbm * torch.clamp(w_og - pcall_strike, min=0.0)
                    if has_pcall
                    else torch.zeros_like(alive_gbm)
                )
                pv_gbm_fine = pv_gbm_fine + float(df_pay_np[-1]) * (
                    put_sign * float(state.option_gearing) * up_g + float(state.participation_call_rate) * uc_g
                )
                if st_g_term is not None and one_star_coupon != 0.0:
                    pv_gbm_fine = pv_gbm_fine + float(df_pay_np[-1]) * alive_gbm * st_g_term * one_star_coupon
                pv_gbm_all[r0_fb : r0_fb + rc_fb] = pv_gbm_fine

        inv_tot_fb = 1.0 / float(n_total_paths)
        stats_cpu = torch.stack(
            [
                p_stop_sum * inv_tot_fb,
                cpn_alive_sum * inv_tot_fb,
                surv_before_sum * inv_tot_fb,
                cpn_cf_sum * inv_tot_fb,
            ],
            dim=0,
        ).cpu().numpy()
        p_stop_arr = stats_cpu[0]
        surv_before_arr = stats_cpu[2]
        q_cpn_arr = np.where(
            surv_before_arr > 1e-12,
            stats_cpu[1] / np.maximum(surv_before_arr, 1e-12),
            0.0,
        )
        cpn_cf_expected = stats_cpu[3]

        if eff_r > 1:
            rep_opt = opt_payoff_paths.mean(dim=1)
            opt_se_t = rep_opt.std(unbiased=True) / math.sqrt(float(eff_r))
        else:
            opt_se_t = opt_payoff_paths.reshape(-1).std(unbiased=True) / math.sqrt(float(n_total_paths))

        scalars_cpu = torch.stack(
            [
                unit_put_sum * inv_tot_fb,
                unit_call_sum * inv_tot_fb,
                onestar_sum * inv_tot_fb,
                p_onestar_sum * inv_tot_fb,
                total_pv_paths.mean(),
                opt_se_t,
            ]
        ).cpu().tolist()
        l_put_val = float(scalars_cpu[0])
        l_call_val = float(scalars_cpu[1])
        l_onestar_val = float(scalars_cpu[2])
        p_onestar_val = float(scalars_cpu[3])
        raw_pv_mean = float(scalars_cpu[4])
        option_se = float(scalars_cpu[5])

        if x_cv_all is not None:
            cv_pv_est, total_se = _apply_loo_control_variate(total_pv_paths, x_cv_all, bundle.mu_cv)
        elif pv_gbm_all is not None:
            mu_mf, se_mu_mf = self._eval_cheap_gbm_expectation(
                state=state,
                chol_f32=chol_f32,
                tau_obs_np=tau_obs_np,
                d_var_gbm_np=bundle.d_var_gbm_np,
                df_pay_np=df_pay_np,
                ko_barriers_np=ko_barriers_np,
                is_callable_np=is_callable_np,
                cpn_barriers_np=cpn_barriers_np,
                is_guar_np=is_guar_np,
                coupon_amounts_np=coupon_amounts_np,
                fwd_prices_gpu=fwd_prices_gpu,
                ref_spots_gpu=ref_spots_gpu,
                w_fund_np=w_fund_np,
                eff_cpn_idx=eff_cpn_idx,
                eff_opt_idx=eff_opt_idx,
                cpn_dir=cpn_dir,
                cpn_type=cpn_type,
                inv_perm_gpu=inv_perm_gpu,
                n_cheap=eff_n_cheap,
                seed=cfg.seed + 88888,
                a_pca_f32=bundle.a_pca_f32,
                m_bb_f32=bundle.m_bb_f32,
                schedule_digest_hex=bundle.schedule.digest_hex,
            )
            cv_pv_est, total_se = _apply_loo_control_variate(
                total_pv_paths, pv_gbm_all, mu_mf, se_mu_x=se_mu_mf
            )
        else:
            cv_pv_est = raw_pv_mean
            if eff_r > 1:
                rep_pv = total_pv_paths.mean(dim=1)
                total_se = float(rep_pv.std(unbiased=True).item() / math.sqrt(float(eff_r)))
            else:
                total_se = float(total_pv_paths.reshape(-1).std(unbiased=True).item() / math.sqrt(float(n_total_paths)))

        return _finalize_dupire_qmc_result(
            state,
            cfg,
            n_total_paths=n_total_paths,
            p_stop_arr=p_stop_arr,
            q_cpn_arr=q_cpn_arr,
            cpn_cf_expected=cpn_cf_expected,
            df_pay_np=df_pay_np,
            l_put_val=l_put_val,
            l_call_val=l_call_val,
            l_onestar_val=l_onestar_val,
            p_onestar_val=p_onestar_val,
            raw_pv_mean=raw_pv_mean,
            cv_pv_est=cv_pv_est,
            total_se=total_se,
            option_se=option_se,
        )

    def _eval_cheap_gbm_expectation(
        self,
        *,
        state: AutocallPreprocState,
        chol_f32: torch.Tensor,
        tau_obs_np: npt.NDArray[np.float64],
        d_var_gbm_np: npt.NDArray[np.float64],
        df_pay_np: npt.NDArray[np.float64],
        ko_barriers_np: npt.NDArray[np.float64],
        is_callable_np: npt.NDArray[np.float64],
        cpn_barriers_np: npt.NDArray[np.float64],
        is_guar_np: npt.NDArray[np.float64],
        coupon_amounts_np: npt.NDArray[np.float64],
        fwd_prices_gpu: torch.Tensor,
        ref_spots_gpu: torch.Tensor,
        w_fund_np: npt.NDArray[np.float64],
        eff_cpn_idx: str,
        eff_opt_idx: str,
        cpn_dir: str,
        cpn_type: str,
        inv_perm_gpu: Optional[torch.Tensor],
        n_cheap: int,
        seed: int,
        a_pca_f32: Optional[torch.Tensor] = None,
        m_bb_f32: Optional[torch.Tensor] = None,
        schedule_digest_hex: Optional[str] = None,
        sched_gpu_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[float, float]:
        """Evaluates antithetic BB+PCA scrambled-Sobol flat-ATM GBM (mean, SE)."""
        dev = self.device
        dev_norm = normalize_cuda_device(dev)
        d_assets = int(state.d_assets)
        m_rem = int(state.m_rem)

        call_code = _qmc_index_code(state.callability_index)
        cpn_code = _qmc_index_code(eff_cpn_idx)
        opt_code = _qmc_index_code(eff_opt_idx)
        is_diff_opt = opt_code in _DIFF_INDEX_CODES
        is_long = bool(getattr(state, "is_long_basket", False))
        ki_opt_idx = "WorstOf" if is_long else eff_opt_idx
        ki_mode = _resolve_ki_obs_mode(state)
        has_ki = (
            state.ki_barrier is not None
            and (float(state.ki_barrier) > 1e-6 or (is_diff_opt and float(state.ki_barrier) < -1e-6))
            and float(state.seasoned_state.i_ki_past) < 0.5
        )
        is_discrete_ki = bool(has_ki and ki_mode in ("Discrete", "Daily"))

        has_put = ((float(state.strike) > 1e-6) or (is_diff_opt and float(state.strike) < -1e-6)) and float(state.option_gearing) != 0.0
        has_low_k = (
            has_put
            and state.lower_strike is not None
            and (float(state.lower_strike) > 1e-6 or (is_diff_opt and float(state.lower_strike) < -1e-6))
        )
        max_spread_put = (
            max(float(state.strike) - float(state.lower_strike), 0.0)
            if (has_low_k and state.lower_strike is not None)
            else 0.0
        )
        has_onestar = state.one_star_strike is not None and float(state.one_star_strike) > 0.0
        one_star_strike_raw = float(state.one_star_strike) if (has_onestar and state.one_star_strike is not None) else 0.0
        one_star_strike, one_star_eps = (
            _resolve_onestar_effective_params(
                one_star_strike_raw,
                float(state.ki_barrier) if state.ki_barrier is not None else None,
                float(state.ko_spread),
            )
            if has_onestar
            else (0.0, 0.02)
        )
        one_star_coupon = float(state.one_star_coupon)
        has_os_cpn = has_onestar and one_star_coupon != 0.0
        has_pcall = state.participation_call_strike is not None and float(state.participation_call_rate) != 0.0
        pcall_strike = float(state.participation_call_strike) if (has_pcall and state.participation_call_strike is not None) else 0.0
        pcall_rate = float(state.participation_call_rate) if has_pcall else 0.0

        use_triton_mf = bool(
            self._use_triton
            and _HAS_TRITON
            and dev.type == "cuda"
            and (m_rem * d_assets <= _SOBOL_MAX_DIM)
        )
        max_pairs_vram = max(2048, (8 * 1024 * 1024) // max(m_rem * d_assets, 1))
        max_pairs_pow2 = 1 << (max_pairs_vram.bit_length() - 1)
        if use_triton_mf:
            target_paths = max(int(n_cheap), 2_097_152)
            target_pairs = (target_paths + 1) // 2
            n_pairs_rep = _next_power_of_two(min(max(target_pairs // 16, 2048), 65536, max_pairs_pow2))
            r_mf = max(8, (target_pairs + n_pairs_rep - 1) // n_pairs_rep)
        else:
            min_paths_pt = 16_384 if dev.type == "cpu" else 262_144
            target_paths = max(int(n_cheap), min_paths_pt)
            target_pairs = (target_paths + 1) // 2
            n_pairs_rep = _next_power_of_two(
                min(max(target_pairs // 8, 512 if dev.type == "cpu" else 2048), 16384, max_pairs_pow2)
            )
            r_mf = max(8 if n_pairs_rep >= 1024 else 4, (target_pairs + n_pairs_rep - 1) // n_pairs_rep)

        h_mf = hashlib.blake2b(digest_size=24)
        if schedule_digest_hex:
            _hash_update_str(h_mf, schedule_digest_hex)
        else:
            for arr in (
                d_var_gbm_np,
                df_pay_np,
                ko_barriers_np,
                is_callable_np,
                cpn_barriers_np,
                is_guar_np,
                coupon_amounts_np,
                w_fund_np,
            ):
                _hash_update_arr(h_mf, arr)
            _hash_update_arr(h_mf, chol_f32.detach().cpu().numpy())
            _hash_update_arr(h_mf, fwd_prices_gpu.detach().cpu().numpy())
            _hash_update_arr(h_mf, ref_spots_gpu.detach().cpu().numpy())
        _hash_update_arr(h_mf, d_var_gbm_np)
        _hash_update_arr(h_mf, w_fund_np)
        payoff_sig = (
            d_assets,
            m_rem,
            call_code,
            cpn_code,
            opt_code,
            cpn_dir,
            cpn_type,
            float(state.seasoned_state.c_mem_0),
            float(state.autocall_coupon),
            float(state.ko_spread),
            float(state.cpn_spread),
            float(state.strike),
            max_spread_put,
            float(state.option_gearing),
            float(state.ki_barrier) if (has_ki and state.ki_barrier is not None) else 0.0,
            float(state.ki_spread) if (has_ki or is_long) else 0.0,
            has_ki,
            is_discrete_ki,
            ki_mode,
            is_long,
            has_low_k,
            has_onestar,
            one_star_strike,
            one_star_eps,
            one_star_coupon,
            has_os_cpn,
            has_pcall,
            pcall_strike,
            pcall_rate,
            bool(state.is_memory),
            bool(state.no_notional_repayment),
            bool(state.force_coupons_on_autocall),
            r_mf,
            n_pairs_rep,
            int(seed),
            use_triton_mf,
        )
        _hash_update_str(h_mf, repr(payoff_sig))
        mf_cache_key = ("mf_gbm_expectation", h_mf.hexdigest(), dev_norm)
        cached_mf = _DEAL_PREP_LRU.get(mf_cache_key)
        if cached_mf is not None:
            return float(cached_mf[0]), float(cached_mf[1])

        if m_bb_f32 is None:
            m_bb_t = torch.as_tensor(
                _build_bb_orthogonal_matrix(tau_obs_np).astype(np.float32),
                dtype=torch.float32,
                device=dev,
            )
        else:
            m_bb_t = m_bb_f32
        if a_pca_f32 is None:
            corr_np = _cpu_f64_array(state.corr_matrix)
            corr_sym = 0.5 * (corr_np + corr_np.T)
            eigvals, eigvecs = np.linalg.eigh(corr_sym)
            order_desc = np.argsort(eigvals)[::-1]
            ev_desc = np.maximum(eigvals[order_desc], 0.0)
            a_pca_np = np.ascontiguousarray(
                eigvecs[:, order_desc] * np.sqrt(ev_desc)[None, :], dtype=np.float32
            )
            a_pca_t = torch.as_tensor(a_pca_np, dtype=torch.float32, device=dev)
        else:
            a_pca_t = a_pca_f32

        if use_triton_mf and n_pairs_rep >= 128:
            sv_gpu = _get_sobol_direction_numbers(dev)
            need_disp = (call_code == 3) or (cpn_code == 3) or (opt_code == 3)
            need_order = (
                call_code in (15, 16, 17) or cpn_code in (15, 16, 17) or opt_code in (15, 16, 17)
            )
            inv_perm_arg = (
                inv_perm_gpu
                if inv_perm_gpu is not None
                else torch.empty(2, dtype=torch.int64, device=dev)
            )
            elems_z = max(m_rem * d_assets * n_pairs_rep, 1)
            elems_ws = max(2 * d_assets * n_pairs_rep, 1)
            r_chunk_cap = max(
                1,
                min(
                    r_mf,
                    (7 * 1024 * 1024) // elems_z,
                    (3 * 1024 * 1024) // elems_ws,
                ),
            )
            block_n_mf = min(128, n_pairs_rep)
            n_tiles_mf = int(triton.cdiv(n_pairs_rep, block_n_mf))
            m_pow_mf = int(round(math.log2(n_pairs_rep)))
            low_bits_mf = int(round(math.log2(block_n_mf)))

            fwd_perf_64 = (fwd_prices_gpu / ref_spots_gpu[:, None]).to(torch.float64).contiguous()
            d_var_gpu = torch.as_tensor(np.ascontiguousarray(d_var_gbm_np), dtype=torch.float64, device=dev)
            if sched_gpu_tuple is not None:
                df_pay_t, ko_bar_t, is_call_t, cpn_bar_t, is_guar_t, cpn_amt_t, w_fund_t = sched_gpu_tuple
            else:
                f64_pack = np.concatenate([
                    df_pay_np.ravel(),
                    ko_barriers_np.ravel(),
                    is_callable_np.ravel(),
                    cpn_barriers_np.ravel(),
                    is_guar_np.ravel(),
                    coupon_amounts_np.ravel(),
                    w_fund_np.ravel(),
                ])
                pack_t = torch.as_tensor(np.ascontiguousarray(f64_pack), dtype=torch.float64, device=dev)
                df_pay_t = pack_t[0:m_rem]
                ko_bar_t = pack_t[m_rem : 2 * m_rem]
                is_call_t = pack_t[2 * m_rem : 3 * m_rem]
                cpn_bar_t = pack_t[3 * m_rem : 4 * m_rem]
                is_guar_t = pack_t[4 * m_rem : 5 * m_rem]
                cpn_amt_t = pack_t[5 * m_rem : 6 * m_rem]
                w_fund_t = pack_t[6 * m_rem : 7 * m_rem]

            sh_all_key = ("mf_pca_shifts", dev_norm, r_mf, m_rem, d_assets, int(seed))
            shifts_all_gpu = _STATIC_CONSTANTS_LRU.get(sh_all_key)
            if shifts_all_gpu is None:
                gen_mf = torch.Generator(device="cpu")
                shifts_all_cpu = torch.empty((m_rem, r_mf, d_assets), dtype=torch.int32, device="cpu")
                for m in range(m_rem):
                    gen_mf.manual_seed(int(seed) + (m * d_assets) * 131 + 17)
                    torch.randint(
                        0,
                        1 << _SOBOL_BITS,
                        (r_mf, d_assets),
                        out=shifts_all_cpu[m],
                        generator=gen_mf,
                    )
                shifts_all_gpu = _STATIC_CONSTANTS_LRU.put(
                    sh_all_key, shifts_all_cpu.to(device=dev, non_blocking=False)
                )

            ws_lx = torch.empty((2, r_chunk_cap, d_assets, n_pairs_rep), dtype=torch.float64, device=dev)
            ws_perf = (
                torch.empty((2, r_chunk_cap, d_assets, n_pairs_rep), dtype=torch.float64, device=dev)
                if (need_disp and d_assets > 1)
                else torch.empty(1, dtype=torch.float64, device=dev)
            )
            tile_pv_sum = torch.empty((r_mf, n_tiles_mf), dtype=torch.float64, device=dev)

            for r0 in range(0, r_mf, r_chunk_cap):
                rc = min(r_chunk_cap, r_mf - r0)
                sh_slice = shifts_all_gpu[:, r0 : r0 + rc, :].contiguous()
                xi_pca = torch.empty((rc, m_rem, d_assets, n_pairs_rep), dtype=torch.float32, device=dev)
                _fast_triton_launch(
                    _sobol_pca_triton_kernel,
                    (n_tiles_mf, m_rem, rc),
                    sv_gpu,
                    sh_slice,
                    a_pca_t,
                    xi_pca,
                    m_rem,
                    rc,
                    n_pairs_rep,
                    m_pow_mf,
                    D_ASSETS=d_assets,
                    LOW_BITS=low_bits_mf,
                    BLOCK_N=block_n_mf,
                    num_warps=4,
                )
                z_corr_blk = (
                    torch.matmul(m_bb_t, xi_pca.view(rc, m_rem, d_assets * n_pairs_rep))
                    .view(rc, m_rem, d_assets, n_pairs_rep)
                    .contiguous()
                )
                del xi_pca, sh_slice

                _fast_triton_launch(
                    _cheap_mf_gbm_triton_kernel,
                    (n_tiles_mf, rc),
                    z_corr_blk,
                    d_var_gpu,
                    fwd_perf_64,
                    df_pay_t,
                    ko_bar_t,
                    is_call_t,
                    cpn_bar_t,
                    is_guar_t,
                    cpn_amt_t,
                    w_fund_t,
                    inv_perm_arg,
                    ws_lx,
                    ws_perf,
                    tile_pv_sum[r0 : r0 + rc],
                    m_rem,
                    rc,
                    n_pairs_rep,
                    n_tiles_mf,
                    float(state.seasoned_state.c_mem_0),
                    float(state.autocall_coupon),
                    float(state.ko_spread),
                    float(state.cpn_spread),
                    float(state.strike),
                    max_spread_put,
                    float(state.option_gearing),
                    float(state.ki_barrier) if (has_ki and state.ki_barrier is not None) else 0.0,
                    float(state.ki_spread) if (has_ki or is_long) else 0.0,
                    one_star_strike,
                    one_star_eps,
                    one_star_coupon,
                    pcall_strike,
                    pcall_rate,
                    D_ASSETS=d_assets,
                    CALL_CODE=call_code,
                    CPN_CODE=cpn_code,
                    OPT_CODE=opt_code,
                    IS_MEM=bool(state.is_memory),
                    NO_NOTIONAL=bool(state.no_notional_repayment),
                    FORCE_CPN_AC=bool(state.force_coupons_on_autocall),
                    CPN_DIR_DOWN=cpn_dir == "down",
                    CPN_TYPE_OUT=cpn_type == "out",
                    HAS_KO_SPREAD=abs(float(state.ko_spread)) >= 1e-9,
                    HAS_CPN_SPREAD=abs(float(state.cpn_spread)) >= 1e-9,
                    HAS_KI=has_ki,
                    HAS_KI_SPREAD=(has_ki or is_long) and abs(float(state.ki_spread)) >= 1e-9,
                    IS_DISCRETE_KI=is_discrete_ki,
                    HAS_PUT=has_put,
                    HAS_LOW_K=has_low_k,
                    HAS_ONESTAR=has_onestar,
                    HAS_OS_CPN=has_os_cpn,
                    HAS_PCALL=has_pcall,
                    IS_LONG_BASKET=is_long,
                    HAS_INV_PERM=inv_perm_gpu is not None,
                    NEED_DISP=need_disp,
                    NEED_ORDER=need_order,
                    BLOCK_N=block_n_mf,
                    num_warps=4,
                )
                del z_corr_blk

            rep_means = tile_pv_sum.sum(dim=1) / float(n_pairs_rep)
            mu_mf = float(rep_means.mean().item())
            se_mu_mf = float(rep_means.std(unbiased=True).item() / math.sqrt(float(r_mf))) if r_mf > 1 else 0.0
            del ws_lx, ws_perf, tile_pv_sum
            _DEAL_PREP_LRU.put(mf_cache_key, (mu_mf, se_mu_mf))
            return mu_mf, se_mu_mf

        dv_all_f64 = torch.tensor(d_var_gbm_np.T.copy(), dtype=torch.float64, device=dev)  # (M_rem, D)
        sqrt_dv_f64 = torch.sqrt(dv_all_f64)
        fwd_rel_64 = (fwd_prices_gpu / ref_spots_gpu[:, None]).T.to(torch.float64).contiguous()

        use_bb_pca_pt = (m_rem * d_assets <= _SOBOL_MAX_DIM) and (n_pairs_rep >= 16) and ((n_pairs_rep & (n_pairs_rep - 1)) == 0)
        r_chunk = max(1, min(r_mf, (4 * 1024 * 1024) // max(m_rem * d_assets * n_pairs_rep, 1)))
        rep_means_list: List[torch.Tensor] = []
        same_cpn_call = eff_cpn_idx == state.callability_index
        sv_gpu_pt = _get_sobol_direction_numbers(dev) if use_bb_pca_pt else None
        for r0 in range(0, r_mf, r_chunk):
            rc = min(r_chunk, r_mf - r0)
            z_corr_pt: Optional[torch.Tensor] = None
            if use_bb_pca_pt and sv_gpu_pt is not None:
                gen_pt = torch.Generator(device="cpu")
                sh_pt_cpu = torch.empty((rc, m_rem * d_assets), dtype=torch.int32, device="cpu")
                for m in range(m_rem):
                    gen_pt.manual_seed(int(seed + r0 * 997) + (m * d_assets) * 131 + 17)
                    torch.randint(
                        0,
                        1 << _SOBOL_BITS,
                        (rc, d_assets),
                        out=sh_pt_cpu[:, m * d_assets : (m + 1) * d_assets],
                        generator=gen_pt,
                    )
                xi_raw_pt = _sobol_from_shifts_torch(
                    sv_gpu=sv_gpu_pt,
                    shifts_gpu=sh_pt_cpu.to(device=dev, non_blocking=False),
                    dim_offset=0,
                    n_rep=n_pairs_rep,
                    device=dev,
                ).view(rc, m_rem, d_assets, n_pairs_rep)
                xi_pca_pt = torch.matmul(a_pca_t, xi_raw_pt)
                z_corr_pt = torch.matmul(
                    m_bb_t, xi_pca_pt.view(rc, m_rem, d_assets * n_pairs_rep)
                ).view(rc, m_rem, d_assets, n_pairs_rep)
                del xi_raw_pt, xi_pca_pt

            batch_2r = 2 * rc
            log_x_c = torch.zeros((batch_2r, d_assets, n_pairs_rep), dtype=torch.float64, device=dev)
            alive = torch.ones((batch_2r, n_pairs_rep), dtype=torch.float64, device=dev)
            unpaid = torch.full((batch_2r, n_pairs_rep), float(state.seasoned_state.c_mem_0), dtype=torch.float64, device=dev)
            not_ki_c = torch.ones((batch_2r, n_pairs_rep), dtype=torch.float64, device=dev)
            pv = torch.zeros((batch_2r, n_pairs_rep), dtype=torch.float64, device=dev)
            perf_last = torch.empty((batch_2r, n_pairs_rep, d_assets), dtype=torch.float64, device=dev)

            for m in range(m_rem):
                if z_corr_pt is not None:
                    z_c_pos = z_corr_pt[:, m].to(torch.float64)
                else:
                    z_m = _generate_gpu_sobol_tile(
                        n_rep=n_pairs_rep,
                        n_replicates=rc,
                        n_dims=d_assets,
                        dim_offset=m * d_assets,
                        seed=seed + r0 * 997,
                        device=dev,
                        use_triton=self._use_triton,
                    )
                    z_c_pos = torch.matmul(chol_f32, z_m).to(torch.float64)
                z_c = torch.cat([z_c_pos, -z_c_pos], dim=0)
                log_x_c = log_x_c - 0.5 * dv_all_f64[m].view(1, d_assets, 1) + sqrt_dv_f64[m].view(1, d_assets, 1) * z_c
                perf_m = (fwd_rel_64[m].view(1, d_assets, 1) * torch.exp(log_x_c)).permute(0, 2, 1)
                w_call_m = _evaluate_monitoring_index_gpu(perf_m, state.callability_index, inv_perm_gpu)
                w_cpn_m = w_call_m if same_cpn_call else _evaluate_monitoring_index_gpu(perf_m, eff_cpn_idx, inv_perm_gpu)
                if is_discrete_ki and state.ki_barrier is not None:
                    w_opt_m = _evaluate_monitoring_index_gpu(perf_m, ki_opt_idx, inv_perm_gpu)
                    not_ki_c = not_ki_c * _digital_ramp_torch(w_opt_m, float(state.ki_barrier), float(state.ki_spread))
                ko_i = (
                    _digital_ramp_torch(w_call_m, float(ko_barriers_np[m]), float(state.ko_spread))
                    if is_callable_np[m] > 0.5
                    else torch.zeros_like(alive)
                )
                if is_guar_np[m] > 0.5:
                    cp_i = torch.ones_like(alive)
                else:
                    b_cpn_m = float(cpn_barriers_np[m])
                    s_cpn = float(state.cpn_spread)
                    bc_i = (
                        1.0 - _digital_ramp_torch(w_cpn_m, b_cpn_m, -s_cpn)
                        if cpn_dir == "down"
                        else _digital_ramp_torch(w_cpn_m, b_cpn_m, s_cpn)
                    )
                    cp_i = (1.0 - bc_i) if cpn_type == "out" else bc_i
                if state.force_coupons_on_autocall:
                    cp_i = torch.maximum(cp_i, ko_i)
                if w_fund_np[m] != 0.0:
                    pv = pv - alive * float(w_fund_np[m])
                st_m = alive * ko_i
                c_m = float(coupon_amounts_np[m])
                ec = (c_m + unpaid) if state.is_memory else c_m
                pv = pv + float(df_pay_np[m]) * (alive * cp_i * ec + st_m * float(state.autocall_coupon))
                if state.is_memory:
                    unpaid = alive * (unpaid + c_m) * (1.0 - cp_i)
                alive = alive * (1.0 - ko_i)
                if not state.no_notional_repayment:
                    pv = pv + float(df_pay_np[m]) * (st_m if m < m_rem - 1 else (st_m + alive))
                if m == m_rem - 1:
                    perf_last = perf_m

            if z_corr_pt is not None:
                del z_corr_pt

            w_opt = _evaluate_monitoring_index_gpu(perf_last, eff_opt_idx, inv_perm_gpu)
            st_term: Optional[torch.Tensor] = None
            if has_onestar:
                st_term = _onestar_ramp_torch(
                    _evaluate_monitoring_index_gpu(perf_last, "BestOf", inv_perm_gpu),
                    one_star_strike,
                    one_star_eps,
                )
            if is_long:
                avg_t = _evaluate_monitoring_index_gpu(perf_last, "AverageOf", inv_perm_gpu)
                wof_t = _evaluate_monitoring_index_gpu(perf_last, "WorstOf", inv_perm_gpu)
                if has_ki and state.ki_barrier is not None and is_discrete_ki:
                    ki_wof_k = 1.0 - not_ki_c
                else:
                    ki_wof_k = 1.0 - _digital_ramp_torch(wof_t, float(state.strike), float(state.ki_spread))
                eff_gearing = float(state.option_gearing) if abs(float(state.option_gearing)) > 1e-12 else 1.0
                pi = ((1.0 - avg_t) * ki_wof_k) / eff_gearing
                if st_term is not None:
                    pi = pi * (1.0 - st_term)
                up = alive * pi
            elif has_put:
                pi = torch.clamp(float(state.strike) - w_opt, min=0.0)
                if has_low_k:
                    pi = torch.clamp(pi, max=max_spread_put)
                if has_ki and state.ki_barrier is not None:
                    ki_c = (
                        1.0 - not_ki_c
                        if is_discrete_ki
                        else (1.0 - _digital_ramp_torch(w_opt, float(state.ki_barrier), float(state.ki_spread)))
                    )
                    pi = pi * ki_c
                if st_term is not None:
                    pi = pi * (1.0 - st_term)
                up = alive * pi
            else:
                up = torch.zeros_like(alive)
            uc = (
                alive * torch.clamp(w_opt - pcall_strike, min=0.0)
                if has_pcall
                else torch.zeros_like(alive)
            )
            put_sign = -1.0
            pv = pv + float(df_pay_np[-1]) * (
                put_sign * float(state.option_gearing) * up + float(state.participation_call_rate) * uc
            )
            if st_term is not None and one_star_coupon != 0.0:
                pv = pv + float(df_pay_np[-1]) * alive * st_term * one_star_coupon
            pv_pairs = 0.5 * (pv[:rc] + pv[rc:])
            rep_means_list.append(pv_pairs.mean(dim=1))

        rep_means_pt = torch.cat(rep_means_list, dim=0)
        mu_mf = float(rep_means_pt.mean().item())
        se_mu_mf = float(rep_means_pt.std(unbiased=True).item() / math.sqrt(float(r_mf))) if r_mf > 1 else 0.0
        _DEAL_PREP_LRU.put(mf_cache_key, (mu_mf, se_mu_mf))
        return mu_mf, se_mu_mf

    def simulate_batch_for_training(
        self,
        states: Sequence[AutocallPreprocState],
        config: Optional[DupireQMCConfig] = None,
        compute_pathwise_delta: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Multi-contract (D, M_rem)-bucketed GPU QMC teacher with single-pass pathwise tangent Delta."""
        b_total = len(states)
        if b_total == 0:
            z0 = torch.empty(0, dtype=torch.float64)
            return (
                z0,
                torch.empty((0, 48), dtype=torch.float64),
                torch.empty((0, 48), dtype=torch.float64),
                z0,
                torch.empty((0, 8), dtype=torch.float64) if compute_pathwise_delta else None,
            )
        if config is None:
            max_d = max(int(s.d_assets) for s in states)
            max_m = max(int(s.m_rem) for s in states)
            base_cfg = resolve_dupire_qmc_config(
                max_d,
                max_m,
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                n_fallback_paths=8192,
            )
            cfg = DupireQMCConfig(
                n_paths=base_cfg.n_paths,
                n_replicates=base_cfg.n_replicates,
                n_substeps=max(base_cfg.n_substeps, 16),
                mode=base_cfg.mode,
                precision=base_cfg.precision,
                cv_mode="none",
                seed=base_cfg.seed,
                lv_grid_nt=base_cfg.lv_grid_nt,
                lv_grid_ny=base_cfg.lv_grid_ny,
                adaptive_substeps=base_cfg.adaptive_substeps,
                n_cheap_mf=base_cfg.n_cheap_mf,
            )
        else:
            cfg = DupireQMCConfig.from_canonical(config)

        d_max = max(int(s.d_max) for s in states)
        m_max = max(int(s.m_max) for s in states)
        pv_out = torch.zeros(b_total, dtype=torch.float64)
        p_out = torch.zeros((b_total, m_max), dtype=torch.float64)
        q_out = torch.zeros((b_total, m_max), dtype=torch.float64)
        l_out = torch.zeros(b_total, dtype=torch.float64)
        delta_out = torch.zeros((b_total, d_max), dtype=torch.float64) if compute_pathwise_delta else None

        # Bucket contracts by exact (d_assets, m_rem) for coalesced GPU batch execution
        buckets: Dict[Tuple[int, int], List[int]] = {}
        for idx, st in enumerate(states):
            key = (int(st.d_assets), int(st.m_rem))
            buckets.setdefault(key, []).append(idx)

        dev = self.device
        eff_r_bkt = max(1, int(cfg.n_replicates))
        n_rep_bkt = _next_power_of_two(max(2, int(cfg.n_paths) // eff_r_bkt))
        n_sub_bkt = max(1, int(cfg.n_substeps))

        for (d_act, m_act), indices in buckets.items():
            if m_act == 0:
                for idx in indices:
                    st = states[idx]
                    pv_out[idx] = float(st.pv_in_transit) + float(
                        getattr(st, "cash_to_settle_today", 0.0) or 0.0
                    )
                continue
            active_indices: List[int] = []
            for idx in indices:
                st = states[idx]
                if st.seasoned_state.already_autocalled or bool(getattr(st, "already_autocalled", False)):
                    pv_out[idx] = float(st.pv_in_transit) + float(
                        getattr(st, "cash_to_settle_today", 0.0) or 0.0
                    )
                else:
                    active_indices.append(idx)
            if not active_indices:
                continue
            indices = active_indices
            chunk_cap = max(
                8,
                min(32, (10 * 1024 * 1024) // max(d_act * max(int(cfg.n_paths), 1024) * 8, 1)),
            )
            # Precompute bucket-shared coarse Sobol tiles once across chunks; reuse z_steps when <= 16 MB
            z_coarse_raw_bkt = _generate_gpu_sobol_tile(
                n_rep=n_rep_bkt,
                n_replicates=eff_r_bkt,
                n_dims=m_act * d_act,
                dim_offset=0,
                seed=cfg.seed,
                device=dev,
                use_triton=self._use_triton,
            )
            z_coarse_bkt = (
                z_coarse_raw_bkt.view(eff_r_bkt, m_act, d_act, n_rep_bkt)
                .permute(1, 0, 2, 3)
                .contiguous()
            )
            del z_coarse_raw_bkt
            z_steps_bkt: Optional[torch.Tensor] = None
            steps_elems = m_act * eff_r_bkt * d_act * n_sub_bkt * n_rep_bkt
            if len(indices) > chunk_cap and n_sub_bkt > 1 and steps_elems <= 4 * 1024 * 1024:
                z_sub_bkt = (
                    _generate_gpu_sobol_tile(
                        n_rep=n_rep_bkt,
                        n_replicates=eff_r_bkt,
                        n_dims=m_act * d_act * n_sub_bkt,
                        dim_offset=m_act * d_act,
                        seed=cfg.seed + 31337,
                        device=dev,
                        use_triton=self._use_triton,
                    )
                    .view(eff_r_bkt, m_act, d_act, n_sub_bkt, n_rep_bkt)
                    .permute(1, 0, 2, 3, 4)
                    .contiguous()
                )
                z_sub_bkt.sub_(z_sub_bkt.mean(dim=3, keepdim=True))
                z_sub_bkt.add_(z_coarse_bkt.unsqueeze(3) * (1.0 / math.sqrt(float(n_sub_bkt))))
                z_steps_bkt = z_sub_bkt

            for start in range(0, len(indices), chunk_cap):
                chunk_idx = indices[start : start + chunk_cap]
                self._run_training_bucket(
                    states=states,
                    chunk_idx=chunk_idx,
                    d_act=d_act,
                    m_act=m_act,
                    cfg=cfg,
                    compute_pathwise_delta=compute_pathwise_delta,
                    pv_out=pv_out,
                    p_out=p_out,
                    q_out=q_out,
                    l_out=l_out,
                    delta_out=delta_out,
                    z_coarse_all=z_coarse_bkt,
                    z_steps_cached=z_steps_bkt,
                )
            del z_coarse_bkt, z_steps_bkt

        return pv_out, p_out, q_out, l_out, delta_out

    def _run_training_bucket(
        self,
        *,
        states: Sequence[AutocallPreprocState],
        chunk_idx: Sequence[int],
        d_act: int,
        m_act: int,
        cfg: DupireQMCConfig,
        compute_pathwise_delta: bool,
        pv_out: torch.Tensor,
        p_out: torch.Tensor,
        q_out: torch.Tensor,
        l_out: torch.Tensor,
        delta_out: Optional[torch.Tensor],
        z_coarse_all: Optional[torch.Tensor] = None,
        z_steps_cached: Optional[torch.Tensor] = None,
    ) -> None:
        dev = self.device
        b_c = len(chunk_idx)
        eff_r = max(1, int(cfg.n_replicates))
        n_rep = _next_power_of_two(max(2, int(cfg.n_paths) // eff_r))
        n_tot = eff_r * n_rep
        n_sub = max(1, int(cfg.n_substeps))

        # Bulk-pack contract tensors on CPU so H2D uses 4 transfers instead of ~35
        st_chunk = [states[i] for i in chunk_idx]
        corr_stack_host = torch.stack(
            [st.corr_matrix.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0
        )
        if not bool(torch.all(torch.isfinite(corr_stack_host)).item()):
            raise ValueError("Correlation matrix contains non-finite values.")
        corr_stack_64 = corr_stack_host.to(dev)
        m2d_host = torch.stack(
            [
                torch.stack([st.tau_obs.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.df_pay.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.ko_barriers.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.is_callable.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.cpn_barriers.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.is_guaranteed_cpn.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.coupon_amounts.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack(
                    [
                        torch.as_tensor(_compute_funding_weights_np(st), dtype=torch.float64)
                        if getattr(st, "funding_leg", None) is not None
                        else torch.zeros(m_act, dtype=torch.float64)
                        for st in st_chunk
                    ],
                    dim=0,
                ),
            ],
            dim=0,
        )  # (8, B_c, M)
        tau_t, df_pay_t, ko_t, is_call_t, cpn_t, is_guar_t, cpn_amt_t, w_fund_t = m2d_host.to(dev).unbind(dim=0)

        m3d_host = torch.stack(
            [
                torch.stack([st.dividend_schedule.forward_price.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.cash_div_leverage.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([_resolve_r_native_obs_tensor(st).detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.dividend_schedule.df_fin_obs.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
            ],
            dim=0,
        )  # (4, B_c, D, M)
        fwd_t, cash_lev_t, r_tot_t, df_fin_t = m3d_host.to(dev).unbind(dim=0)
        r_tot_q_host = torch.stack(
            [st.dividend_schedule.r_tot_obs.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0
        )
        q_full_host = r_tot_q_host - m3d_host[2]
        q_prev_host = torch.cat([torch.zeros((b_c, d_act, 1), dtype=torch.float64), q_full_host[:, :, :-1]], dim=2)
        q_shift_bnd_m_t = (
            torch.stack([q_prev_host, q_full_host], dim=3)
            .permute(2, 0, 1, 3)
            .to(device=dev, dtype=torch.float32)
            .contiguous()
        )  # (M, B_c, D, 2)

        d1d_host = torch.stack(
            [
                torch.stack([st.ref_spots.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.spots_t0.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.v_perf[:, -1].detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
            ],
            dim=0,
        )  # (3, B_c, D)
        ref_t, spots_t, v_term_t = d1d_host.to(dev).unbind(dim=0)

        scal_np = np.empty((25, b_c), dtype=np.float64)
        surfs_flat: List[Any] = []
        div_events_flat: List[Sequence[Any]] = []
        for bi, st in enumerate(st_chunk):
            e_opt_i, _, is_long_i = _resolve_eff_and_ki_opt_indices(st)
            opt_c_i = _qmc_index_code(e_opt_i)
            is_diff_i = opt_c_i in _DIFF_INDEX_CODES
            scal_np[0, bi] = float(st.seasoned_state.c_mem_0)
            scal_np[1, bi] = 1.0 if st.is_memory else 0.0
            scal_np[2, bi] = 1.0 if st.force_coupons_on_autocall else 0.0
            scal_np[3, bi] = 1.0 if st.no_notional_repayment else 0.0
            scal_np[4, bi] = float(st.autocall_coupon)
            scal_np[5, bi] = float(st.ko_spread)
            scal_np[6, bi] = float(st.cpn_spread)
            scal_np[7, bi] = float(st.pv_in_transit) + float(getattr(st, "cash_to_settle_today", 0.0) or 0.0)
            scal_np[8, bi] = float(st.strike)
            scal_np[9, bi] = float(st.option_gearing)
            has_put_i = (
                (float(st.strike) > 1e-6 or (is_diff_i and float(st.strike) < -1e-6))
                and float(st.option_gearing) != 0.0
            )
            lk = st.lower_strike
            has_low_i = has_put_i and lk is not None and (float(lk) > 1e-6 or (is_diff_i and float(lk) < -1e-6))
            scal_np[10, bi] = float(lk) if (has_low_i and lk is not None) else 0.0
            kb = st.ki_barrier
            has_ki_i = (
                kb is not None
                and (float(kb) > 1e-6 or (is_diff_i and float(kb) < -1e-6))
                and float(st.seasoned_state.i_ki_past) < 0.5
            )
            scal_np[11, bi] = float(kb) if (has_ki_i and kb is not None) else 0.0
            scal_np[12, bi] = float(st.ki_spread)
            sk = st.one_star_strike
            has_star_i = sk is not None and float(sk) > 0.0
            if has_star_i and sk is not None:
                sk_eff_i, eps_os_i = _resolve_onestar_effective_params(
                    float(sk),
                    float(kb) if kb is not None else None,
                    float(st.ko_spread),
                )
            else:
                sk_eff_i, eps_os_i = 0.0, 0.02
            scal_np[13, bi] = sk_eff_i
            scal_np[14, bi] = float(st.one_star_coupon)
            pk = st.participation_call_strike
            pr = st.participation_call_rate
            has_pc_i = pk is not None and float(pr) != 0.0
            scal_np[15, bi] = float(pk) if (has_pc_i and pk is not None) else 0.0
            scal_np[16, bi] = float(pr)
            scal_np[17, bi] = -1.0
            scal_np[18, bi] = 1.0 if has_low_i else 0.0
            scal_np[19, bi] = 1.0 if has_ki_i else 0.0
            scal_np[20, bi] = 1.0 if has_put_i else 0.0
            ki_m_i = _resolve_ki_obs_mode(st)
            scal_np[21, bi] = 1.0 if (has_ki_i and ki_m_i in ("Discrete", "Daily")) else 0.0
            scal_np[22, bi] = 1.0 if has_pc_i else 0.0
            scal_np[23, bi] = eps_os_i
            scal_np[24, bi] = 1.0 if is_long_i else 0.0
            div_ev_map = getattr(st.dividend_schedule, "dividend_events", {})
            for ticker in st.sorted_identifiers:
                surfs_flat.append(st.vol_surfaces[ticker])
                div_events_flat.append(div_ev_map.get(ticker, []))

        scal_gpu = torch.as_tensor(scal_np, dtype=torch.float64, device=dev)

        corr_sym_t = 0.5 * (corr_stack_64 + corr_stack_64.transpose(-1, -2))
        chol_64, chol_info = torch.linalg.cholesky_ex(corr_sym_t)
        if torch.any(chol_info != 0):
            bad_rows = torch.nonzero(chol_info != 0, as_tuple=False).reshape(-1)
            for r_i in bad_rows.tolist():
                ev, evec = torch.linalg.eigh(corr_sym_t[r_i])
                ev = torch.clamp(ev, min=1e-6)
                c_psd = (evec * ev.unsqueeze(0)) @ evec.transpose(0, 1)
                di = torch.rsqrt(torch.clamp(torch.diag(c_psd), min=1e-12))
                c_psd = c_psd * torch.outer(di, di)
                chol_64[r_i] = torch.linalg.cholesky(c_psd)
        chol_t = chol_64.to(torch.float32).contiguous()  # (B_c, D, D)

        tau_rep_gpu = tau_t.unsqueeze(1).expand(b_c, d_act, m_act).reshape(b_c * d_act, m_act)
        lv_digest_bkt = _compute_lv_batch_digest(
            surfs_flat,
            d1d_host[1].reshape(-1).numpy(),
            m3d_host[2].reshape(-1, m_act).numpy(),
            m3d_host[3].reshape(-1, m_act).numpy(),
            m2d_host[0].unsqueeze(1).expand(b_c, d_act, m_act).reshape(-1, m_act).numpy(),
            div_events_flat,
            d1d_host[2].reshape(-1).numpy(),
            cfg.lv_grid_nt,
            cfg.lv_grid_ny,
            dev,
            self._use_triton,
        )
        lv_flat, ymin_flat, invdy_flat = _build_lv_tables_batch(
            surfs_flat,
            spots_t.reshape(-1),
            r_tot_t.reshape(-1, m_act),
            df_fin_t.reshape(-1, m_act),
            tau_rep_gpu,
            div_events_flat,
            v_term_t.reshape(-1),
            cfg.lv_grid_nt,
            cfg.lv_grid_ny,
            dev,
            use_triton=self._use_triton,
            lv_digest=lv_digest_bkt,
        )
        lv_table_b = lv_flat.view(b_c, d_act, cfg.lv_grid_nt, cfg.lv_grid_ny).contiguous()
        ymin_t = ymin_flat.view(b_c, d_act).contiguous()
        invdy_t = invdy_flat.view(b_c, d_act).contiguous()
        inv_dt_grid_t = (float(cfg.lv_grid_nt - 1) / torch.clamp(tau_t[:, -1], min=1e-4)).to(torch.float32).contiguous()

        # Precompute grouped index descriptors to avoid per-contract loops in time steps
        perm_cache: Dict[Tuple[int, ...], torch.Tensor] = {}
        call_groups: Dict[Tuple[str, Optional[Tuple[int, ...]]], List[int]] = {}
        cpn_groups: Dict[Tuple[str, Optional[Tuple[int, ...]]], List[int]] = {}
        opt_groups: Dict[Tuple[str, Optional[Tuple[int, ...]]], List[int]] = {}
        ki_opt_groups: Dict[Tuple[str, Optional[Tuple[int, ...]]], List[int]] = {}
        cpn_diff_list: List[int] = []
        cpn_down_flags: List[bool] = []
        cpn_out_flags: List[bool] = []
        for bi, st in enumerate(st_chunk):
            inv_raw = getattr(st, "inv_lex_perm", None)
            p_key = tuple(int(x) for x in inv_raw) if inv_raw is not None else None
            if p_key is not None and p_key not in perm_cache:
                perm_cache[p_key] = torch.tensor(p_key, dtype=torch.long, device=dev)
            c_idx = str(st.callability_index)
            call_groups.setdefault((c_idx, p_key), []).append(bi)
            e_cpn = str(st.coupon_index if st.coupon_index else st.callability_index)
            if e_cpn != c_idx:
                cpn_diff_list.append(bi)
                cpn_groups.setdefault((e_cpn, p_key), []).append(bi)
            e_opt, ki_opt_i, _ = _resolve_eff_and_ki_opt_indices(st)
            opt_groups.setdefault((e_opt, p_key), []).append(bi)
            ki_opt_groups.setdefault((ki_opt_i, p_key), []).append(bi)
            cpn_down_flags.append(str(getattr(st, "coupon_barrier_direction", "Up") or "Up").strip().lower() == "down")
            cpn_out_flags.append(str(getattr(st, "coupon_barrier_type", "In") or "In").strip().lower() == "out")
        call_groups_t = [
            (k, torch.tensor(v, dtype=torch.long, device=dev)) for k, v in call_groups.items()
        ]
        cpn_groups_t = [
            (k, torch.tensor(v, dtype=torch.long, device=dev)) for k, v in cpn_groups.items()
        ]
        opt_groups_t = [
            (k, torch.tensor(v, dtype=torch.long, device=dev)) for k, v in opt_groups.items()
        ]
        ki_opt_groups_t = (
            opt_groups_t
            if ki_opt_groups == opt_groups
            else [(k, torch.tensor(v, dtype=torch.long, device=dev)) for k, v in ki_opt_groups.items()]
        )
        cpn_down_mask_t = torch.tensor(cpn_down_flags, dtype=torch.bool, device=dev).view(b_c, 1)
        cpn_out_mask_t = torch.tensor(cpn_out_flags, dtype=torch.bool, device=dev).view(b_c, 1)

        tau_prev_t = torch.cat([torch.zeros((b_c, 1), dtype=torch.float64, device=dev), tau_t[:, :-1]], dim=1)
        dt_per_t = torch.clamp(tau_t - tau_prev_t, min=1e-8)
        dt_sub_t_all = dt_per_t / float(n_sub)
        # (m_act, b_c) contiguous layout so slice [m] is stride-1 contiguous for Triton
        t_start_all_t = tau_prev_t.transpose(0, 1).to(torch.float32).contiguous()
        dt_sub_all_t = dt_sub_t_all.transpose(0, 1).to(torch.float32).contiguous()
        sqrt_dt_all_t = torch.sqrt(dt_sub_t_all).transpose(0, 1).to(torch.float32).contiguous()

        df_pay_m_t = df_pay_t.transpose(0, 1).contiguous()
        cpn_amt_m_t = cpn_amt_t.transpose(0, 1).contiguous()
        w_fund_m_t = w_fund_t.transpose(0, 1).contiguous()

        c_mem0_t = scal_gpu[0].view(b_c, 1)
        is_mem_t = scal_gpu[1].view(b_c, 1)
        force_cpn_t = scal_gpu[2].view(b_c, 1)
        no_notional_t = scal_gpu[3].view(b_c, 1)
        ac_cpn_t = scal_gpu[4].view(b_c, 1)
        ko_spread_t = scal_gpu[5].view(b_c, 1)
        cpn_spread_t = scal_gpu[6].view(b_c, 1)
        pv_in_trans_t = scal_gpu[7]

        is_mem_1d = scal_gpu[1].contiguous()
        force_cpn_1d = scal_gpu[2].contiguous()
        no_not_1d = scal_gpu[3].contiguous()
        ac_cpn_1d = scal_gpu[4].contiguous()

        # SoA layout: (B_c, D, R, N_rep) and (B_c, D, N_tot) for coalesced GPU reads/writes
        log_x = torch.zeros((b_c, d_act, eff_r, n_rep), dtype=torch.float32, device=dev)
        empty_f32_stub = torch.empty(1, dtype=torch.float32, device=dev)
        alive = torch.ones((b_c, n_tot), dtype=torch.float64, device=dev)
        unpaid_mem = c_mem0_t.expand(b_c, n_tot).clone()
        total_pv = torch.zeros((b_c, n_tot), dtype=torch.float64, device=dev)

        # Forward-mode pathwise tangent state wrt log(S_0) in coalesced SoA (B_c, D, N_tot)
        d_alive = torch.zeros((b_c, d_act, n_tot), dtype=torch.float64, device=dev) if compute_pathwise_delta else None
        d_unpaid = torch.zeros((b_c, d_act, n_tot), dtype=torch.float64, device=dev) if compute_pathwise_delta else None
        d_total_pv = torch.zeros((b_c, d_act, n_tot), dtype=torch.float64, device=dev) if compute_pathwise_delta else None
        empty_3d_stub = torch.empty(1, dtype=torch.float64, device=dev)

        has_any_disc_ki = bool(np.any(scal_np[21] > 0.5))
        is_disc_ki_b = scal_gpu[21].view(b_c, 1) > 0.5
        not_ki_disc_b = torch.ones((b_c, n_tot), dtype=torch.float64, device=dev) if has_any_disc_ki else None
        d_not_ki_disc_b = (
            torch.zeros((b_c, d_act, n_tot), dtype=torch.float64, device=dev)
            if (has_any_disc_ki and compute_pathwise_delta)
            else None
        )
        ki_bar_b = scal_gpu[11].view(b_c, 1)
        ki_spr_b = scal_gpu[12].view(b_c, 1)

        p_stop_b = torch.zeros((b_c, m_act), dtype=torch.float64, device=dev)
        q_cpn_b = torch.zeros((b_c, m_act), dtype=torch.float64, device=dev)

        sub_dim_base = m_act * d_act
        if z_coarse_all is None:
            z_coarse_all = (
                _generate_gpu_sobol_tile(
                    n_rep=n_rep,
                    n_replicates=eff_r,
                    n_dims=m_act * d_act,
                    dim_offset=0,
                    seed=cfg.seed,
                    device=dev,
                    use_triton=self._use_triton,
                )
                .view(eff_r, m_act, d_act, n_rep)
                .permute(1, 0, 2, 3)
                .contiguous()
            )
        m_blk_cap = max(1, min(m_act, (4 * 1024 * 1024) // max(eff_r * d_act * n_sub * n_rep, 1)))
        z_steps_blk: Optional[torch.Tensor] = z_steps_cached
        blk_end = m_act if z_steps_cached is not None else 0
        blk_start = 0

        for m in range(m_act):
            if n_sub == 1:
                z_steps_indep = z_coarse_all[m].unsqueeze(2).contiguous()
            else:
                if z_steps_blk is None or m >= blk_end:
                    blk_start = m
                    blk_end = min(m_act, m + m_blk_cap)
                    m_len = blk_end - blk_start
                    z_sub_raw = (
                        _generate_gpu_sobol_tile(
                            n_rep=n_rep,
                            n_replicates=eff_r,
                            n_dims=m_len * d_act * n_sub,
                            dim_offset=sub_dim_base + blk_start * (d_act * n_sub),
                            seed=cfg.seed + 31337,
                            device=dev,
                            use_triton=self._use_triton,
                        )
                        .view(eff_r, m_len, d_act, n_sub, n_rep)
                        .permute(1, 0, 2, 3, 4)
                        .contiguous()
                    )
                    z_sub_raw.sub_(z_sub_raw.mean(dim=3, keepdim=True))
                    z_sub_raw.add_(z_coarse_all[blk_start:blk_end].unsqueeze(3) * (1.0 / math.sqrt(float(n_sub))))
                    z_steps_blk = z_sub_raw
                z_steps_indep = z_steps_blk[m - blk_start]

            if self._use_triton:
                t_start_t = t_start_all_t[m]
                dt_sub_t = dt_sub_all_t[m]
                sqrt_dt_t = sqrt_dt_all_t[m]
                block_n = min(256, max(16, _next_power_of_two(n_rep)))
                grid_b = (triton.cdiv(n_rep, block_n), b_c * d_act, eff_r)
                _fast_triton_launch(
                    _dupire_batch_substep_triton_kernel,
                    grid_b,
                    log_x,
                    empty_f32_stub,
                    z_steps_indep,
                    chol_t,
                    lv_table_b,
                    ymin_t,
                    invdy_t,
                    q_shift_bnd_m_t[m],
                    t_start_t,
                    dt_sub_t,
                    sqrt_dt_t,
                    inv_dt_grid_t,
                    r_reps=eff_r,
                    d_assets=d_act,
                    n_rep=n_rep,
                    n_sub=n_sub,
                    N_t=cfg.lv_grid_nt,
                    N_y=cfg.lv_grid_ny,
                    BLOCK_N=block_n,
                    COMPUTE_DELTA=0,
                )
            else:
                z_steps_corr = torch.einsum("bij,rjsn->brisn", chol_t, z_steps_indep).permute(0, 2, 1, 3, 4)
                max_t_f = float(cfg.lv_grid_nt - 1) - 1e-4
                max_y_f = float(cfg.lv_grid_ny - 1) - 1e-4
                dt_sub_t = dt_sub_all_t[m].view(b_c, 1, 1, 1)
                sqrt_dt_t = sqrt_dt_all_t[m].view(b_c, 1, 1, 1)
                t_start_t = t_start_all_t[m].view(b_c, 1, 1, 1)
                b_idx = torch.arange(b_c, device=dev).view(b_c, 1, 1, 1)
                d_idx = torch.arange(d_act, device=dev).view(1, d_act, 1, 1)
                ym_v = ymin_t.view(b_c, d_act, 1, 1)
                idy_v = invdy_t.view(b_c, d_act, 1, 1)
                idg_v = inv_dt_grid_t.view(b_c, 1, 1, 1)
                q0_v = q_shift_bnd_m_t[m, :, :, 0].view(b_c, d_act, 1, 1)
                q1_v = q_shift_bnd_m_t[m, :, :, 1].view(b_c, d_act, 1, 1)

                def _eval_lv_b(t_q: torch.Tensor, lx_q: torch.Tensor) -> torch.Tensor:
                    ut = torch.clamp(t_q * idg_v, 0.0, max_t_f)
                    it0 = ut.to(torch.long)
                    ft = ut - it0.to(torch.float32)
                    it1 = torch.clamp(it0 + 1, max=cfg.lv_grid_nt - 1)
                    uy = torch.clamp((lx_q - ym_v) * idy_v, 0.0, max_y_f)
                    iy0 = uy.to(torch.long)
                    fy = uy - iy0.to(torch.float32)
                    iy1 = torch.clamp(iy0 + 1, max=cfg.lv_grid_ny - 1)
                    v00 = lv_table_b[b_idx, d_idx, it0, iy0]
                    v01 = lv_table_b[b_idx, d_idx, it0, iy1]
                    v10 = lv_table_b[b_idx, d_idx, it1, iy0]
                    v11 = lv_table_b[b_idx, d_idx, it1, iy1]
                    return torch.lerp(torch.lerp(v00, v01, fy), torch.lerp(v10, v11, fy), ft)

                for s in range(n_sub):
                    a0 = float(s) / float(n_sub)
                    a1 = float(s + 1) / float(n_sub)
                    t_s0 = t_start_t + float(s) * dt_sub_t
                    t_s1 = t_start_t + float(s + 1) * dt_sub_t
                    q_sh0 = q0_v + a0 * (q1_v - q0_v)
                    q_sh1 = q0_v + a1 * (q1_v - q0_v)
                    lv0 = _eval_lv_b(t_s0, log_x + q_sh0)
                    mu0 = -0.5 * lv0 * lv0
                    y_h = log_x + mu0 * dt_sub_t
                    sig_step = lv0 * sqrt_dt_t
                    lv_p = _eval_lv_b(t_s1, y_h + sig_step + q_sh1)
                    lv_m = _eval_lv_b(t_s1, y_h - sig_step + q_sh1)
                    mu_p = -0.5 * lv_p * lv_p
                    mu_m = -0.5 * lv_m * lv_m
                    z_s = z_steps_corr[:, :, :, s, :]
                    log_x = log_x + (
                        0.5 * (mu0 + 0.5 * (mu_p + mu_m)) * dt_sub_t
                        + 0.25 * (mu_p - mu_m) * dt_sub_t * z_s
                        + 0.25 * (lv_p + lv_m + 2.0 * lv0) * sqrt_dt_t * z_s
                        + 0.25 * (lv_p - lv_m) * (z_s * z_s - 1.0) * sqrt_dt_t
                    )

            # Martingale renormalization per (b, d, r) with zero-copy SoA perf_m (B_c, D, N_tot)
            lx_f64 = log_x.to(torch.float64)
            exp_lx = torch.exp(lx_f64)
            mean_x = torch.clamp(exp_lx.mean(dim=-1, keepdim=True), min=1e-12)
            lx_f64 = lx_f64 - torch.log(mean_x)
            log_x = lx_f64.to(torch.float32)

            s_m = fwd_t[:, :, m].view(b_c, d_act, 1, 1) * (exp_lx / mean_x)
            perf_m = (s_m / ref_t.view(b_c, d_act, 1, 1)).reshape(b_c, d_act, n_tot)
            del lx_f64, exp_lx, mean_x, s_m

            if compute_pathwise_delta:
                dperf_diag: Optional[torch.Tensor] = perf_m * cash_lev_t[:, :, m].unsqueeze(-1)
            else:
                dperf_diag = None

            if len(call_groups_t) == 1:
                (c_idx_only, p_key_only), _ = call_groups_t[0]
                inv_p_only = perm_cache.get(p_key_only) if p_key_only is not None else None
                w_call, gwc = _eval_index_and_grad_gpu(
                    perf_m, c_idx_only, inv_p_only, compute_grad=compute_pathwise_delta, asset_dim=1
                )
                if compute_pathwise_delta and gwc is not None and dperf_diag is not None:
                    dw_call: Optional[torch.Tensor] = gwc * dperf_diag
                    del gwc
                else:
                    dw_call = None
            else:
                w_call = torch.empty((b_c, n_tot), dtype=torch.float64, device=dev)
                dw_call = torch.empty((b_c, d_act, n_tot), dtype=torch.float64, device=dev) if compute_pathwise_delta else None
                for (c_idx_g, p_key_g), idx_t in call_groups_t:
                    inv_p_g = perm_cache.get(p_key_g) if p_key_g is not None else None
                    wc_g, gwc_g = _eval_index_and_grad_gpu(
                        perf_m[idx_t], c_idx_g, inv_p_g, compute_grad=compute_pathwise_delta, asset_dim=1
                    )
                    w_call[idx_t] = wc_g
                    if dw_call is not None and gwc_g is not None and dperf_diag is not None:
                        dw_call[idx_t] = gwc_g * dperf_diag[idx_t]

            if not cpn_diff_list:
                w_cpn = w_call
                dw_cpn = dw_call
            else:
                w_cpn = w_call.clone()
                dw_cpn = dw_call.clone() if dw_call is not None else None
                for (e_cpn_g, p_key_g), idx_t in cpn_groups_t:
                    inv_p_g = perm_cache.get(p_key_g) if p_key_g is not None else None
                    wq_g, gwq_g = _eval_index_and_grad_gpu(
                        perf_m[idx_t], e_cpn_g, inv_p_g, compute_grad=compute_pathwise_delta, asset_dim=1
                    )
                    w_cpn[idx_t] = wq_g
                    if dw_cpn is not None and gwq_g is not None and dperf_diag is not None:
                        dw_cpn[idx_t] = gwq_g * dperf_diag[idx_t]

            if has_any_disc_ki and not_ki_disc_b is not None:
                if len(ki_opt_groups_t) == 1:
                    (o_idx_only, p_key_only), _ = ki_opt_groups_t[0]
                    inv_p_only = perm_cache.get(p_key_only) if p_key_only is not None else None
                    w_opt_m, gwo_m = _eval_index_and_grad_gpu(
                        perf_m, o_idx_only, inv_p_only, compute_grad=compute_pathwise_delta, asset_dim=1
                    )
                    dw_opt_m = (
                        (gwo_m * dperf_diag)
                        if (compute_pathwise_delta and gwo_m is not None and dperf_diag is not None)
                        else None
                    )
                else:
                    w_opt_m = torch.empty((b_c, n_tot), dtype=torch.float64, device=dev)
                    dw_opt_m = (
                        torch.empty((b_c, d_act, n_tot), dtype=torch.float64, device=dev)
                        if compute_pathwise_delta
                        else None
                    )
                    for (o_idx_g, p_key_g), idx_t in ki_opt_groups_t:
                        inv_p_g = perm_cache.get(p_key_g) if p_key_g is not None else None
                        wo_g, gwo_g = _eval_index_and_grad_gpu(
                            perf_m[idx_t], o_idx_g, inv_p_g, compute_grad=compute_pathwise_delta, asset_dim=1
                        )
                        w_opt_m[idx_t] = wo_g
                        if dw_opt_m is not None and gwo_g is not None and dperf_diag is not None:
                            dw_opt_m[idx_t] = gwo_g * dperf_diag[idx_t]
                ki_up_m, dki_up_m = _digital_ramp_and_deriv_batch(
                    w_opt_m, ki_bar_b, ki_spr_b, compute_deriv=compute_pathwise_delta
                )
                if (
                    compute_pathwise_delta
                    and d_not_ki_disc_b is not None
                    and dki_up_m is not None
                    and dw_opt_m is not None
                ):
                    d_not_ki_disc_b = torch.where(
                        is_disc_ki_b.unsqueeze(1),
                        d_not_ki_disc_b * ki_up_m.unsqueeze(1)
                        + not_ki_disc_b.unsqueeze(1) * (dki_up_m.unsqueeze(1) * dw_opt_m),
                        d_not_ki_disc_b,
                    )
                not_ki_disc_b = torch.where(is_disc_ki_b, not_ki_disc_b * ki_up_m, not_ki_disc_b)

            if m < m_act - 1:
                del perf_m, dperf_diag

            ko_ind, dko_dw = _digital_ramp_and_deriv_batch(
                w_call, ko_t[:, m : m + 1], ko_spread_t, compute_deriv=compute_pathwise_delta
            )
            ko_ind = ko_ind * is_call_t[:, m : m + 1]
            if dko_dw is not None:
                dko_dw = dko_dw * is_call_t[:, m : m + 1]

            eff_sc = torch.where(cpn_down_mask_t, -cpn_spread_t, cpn_spread_t)
            base_i, dbase = _digital_ramp_and_deriv_batch(
                w_cpn, cpn_t[:, m : m + 1], eff_sc, compute_deriv=compute_pathwise_delta
            )
            base_i = torch.where(cpn_down_mask_t, 1.0 - base_i, base_i)
            cpn_ind = torch.where(cpn_out_mask_t, 1.0 - base_i, base_i)
            guar_m = is_guar_t[:, m : m + 1] > 0.5
            cpn_ind = torch.where(guar_m, torch.ones_like(cpn_ind), cpn_ind)
            dcpn_dw: Optional[torch.Tensor] = None
            if compute_pathwise_delta and dbase is not None:
                dbase = torch.where(cpn_down_mask_t, -dbase, dbase)
                dcpn_raw = torch.where(cpn_out_mask_t, -dbase, dbase)
                dcpn_dw = torch.where(guar_m, torch.zeros_like(dcpn_raw), dcpn_raw)

            force_mask = (force_cpn_t > 0.5) & (ko_ind > cpn_ind)
            cpn_ind_eff = torch.where(force_mask, ko_ind, cpn_ind)
            p_stop_b[:, m] = (alive * ko_ind).mean(dim=1)
            surv_bm = alive.mean(dim=1)
            q_cpn_b[:, m] = torch.where(
                surv_bm > 1e-12,
                (alive * cpn_ind_eff).mean(dim=1) / torch.clamp(surv_bm, min=1e-12),
                torch.zeros_like(surv_bm),
            )

            if self._use_triton:
                block_step = min(256, max(16, _next_power_of_two(n_tot)))
                grid_step = (triton.cdiv(n_tot, block_step), b_c)
                has_pw = bool(
                    compute_pathwise_delta
                    and dw_call is not None
                    and dw_cpn is not None
                    and dko_dw is not None
                    and dcpn_dw is not None
                    and d_alive is not None
                    and d_unpaid is not None
                    and d_total_pv is not None
                )
                _fast_triton_launch(
                    _training_period_step_triton_kernel,
                    grid_step,
                    alive,
                    unpaid_mem,
                    total_pv,
                    ko_ind,
                    cpn_ind,
                    dko_dw if dko_dw is not None else empty_3d_stub,
                    dcpn_dw if dcpn_dw is not None else empty_3d_stub,
                    d_alive if d_alive is not None else empty_3d_stub,
                    d_unpaid if d_unpaid is not None else empty_3d_stub,
                    d_total_pv if d_total_pv is not None else empty_3d_stub,
                    dw_call if dw_call is not None else empty_3d_stub,
                    dw_cpn if dw_cpn is not None else empty_3d_stub,
                    df_pay_m_t[m],
                    cpn_amt_m_t[m],
                    w_fund_m_t[m],
                    ac_cpn_1d,
                    is_mem_1d,
                    force_cpn_1d,
                    no_not_1d,
                    n_tot,
                    d_act,
                    1 if m == m_act - 1 else 0,
                    COMPUTE_DELTA=1 if has_pw else 0,
                    BLOCK_N=block_step,
                )
            else:
                if compute_pathwise_delta and dw_call is not None and dw_cpn is not None and dko_dw is not None and dcpn_dw is not None and d_alive is not None and d_unpaid is not None and d_total_pv is not None:
                    d_ko = dko_dw.unsqueeze(1) * dw_call
                    d_cpn_raw = dcpn_dw.unsqueeze(1) * dw_cpn
                    d_cpn = torch.where(force_mask.unsqueeze(1), d_ko, d_cpn_raw)

                    wf_m = w_fund_t[:, m : m + 1].unsqueeze(1)
                    d_total_pv = d_total_pv - d_alive * wf_m

                    d_stop_m = d_alive * ko_ind.unsqueeze(1) + alive.unsqueeze(1) * d_ko
                    c_m_v = cpn_amt_t[:, m : m + 1]
                    eff_c = torch.where(is_mem_t > 0.5, c_m_v + unpaid_mem, c_m_v)
                    d_eff_c = torch.where(is_mem_t.unsqueeze(1) > 0.5, d_unpaid, torch.zeros_like(d_unpaid))
                    d_cpn_pay = (
                        d_alive * (cpn_ind_eff * eff_c).unsqueeze(1)
                        + alive.unsqueeze(1) * (d_cpn * eff_c.unsqueeze(1) + cpn_ind_eff.unsqueeze(1) * d_eff_c)
                        + d_stop_m * ac_cpn_t.unsqueeze(1)
                    )
                    d_total_pv = d_total_pv + df_pay_t[:, m : m + 1].unsqueeze(1) * d_cpn_pay

                    if torch.any(is_mem_t > 0.5):
                        d_unpaid_next = (
                            d_alive * (eff_c * (1.0 - cpn_ind_eff)).unsqueeze(1)
                            + alive.unsqueeze(1)
                            * (d_eff_c * (1.0 - cpn_ind_eff).unsqueeze(1) - eff_c.unsqueeze(1) * d_cpn)
                        )
                        d_unpaid = torch.where(is_mem_t.unsqueeze(1) > 0.5, d_unpaid_next, d_unpaid)

                    d_alive_next = d_alive * (1.0 - ko_ind).unsqueeze(1) - alive.unsqueeze(1) * d_ko
                    not_mask = (1.0 - no_notional_t).unsqueeze(1) * df_pay_t[:, m : m + 1].unsqueeze(1)
                    if m < m_act - 1:
                        d_total_pv = d_total_pv + not_mask * d_stop_m
                    else:
                        d_total_pv = d_total_pv + not_mask * (d_stop_m + d_alive_next)
                    d_alive = d_alive_next

                total_pv = total_pv - alive * w_fund_t[:, m : m + 1]
                stop_m = alive * ko_ind
                c_m_v = cpn_amt_t[:, m : m + 1]
                eff_c = torch.where(is_mem_t > 0.5, c_m_v + unpaid_mem, c_m_v)
                total_pv = total_pv + df_pay_t[:, m : m + 1] * (alive * cpn_ind_eff * eff_c + stop_m * ac_cpn_t)
                unpaid_mem = torch.where(is_mem_t > 0.5, alive * eff_c * (1.0 - cpn_ind_eff), unpaid_mem)
                alive = alive * (1.0 - ko_ind)
                not_mult = (1.0 - no_notional_t) * df_pay_t[:, m : m + 1]
                if m < m_act - 1:
                    total_pv = total_pv + not_mult * stop_m
                else:
                    total_pv = total_pv + not_mult * (stop_m + alive)

            del dw_call, dw_cpn
            if m == m_act - 1:
                perf_term = perf_m
                if compute_pathwise_delta:
                    dperf_term_diag = dperf_diag
                del perf_m, dperf_diag

        # Vectorized terminal option & tangent across bucket in SoA (B_c, D, N_tot)
        if len(opt_groups_t) == 1:
            (o_idx_only, p_key_only), _ = opt_groups_t[0]
            inv_p_only = perm_cache.get(p_key_only) if p_key_only is not None else None
            w_opt_b, gw_opt_b = _eval_index_and_grad_gpu(
                perf_term, o_idx_only, inv_p_only, compute_grad=compute_pathwise_delta, asset_dim=1
            )
            dw_opt_b = (
                (gw_opt_b * dperf_term_diag)
                if (compute_pathwise_delta and gw_opt_b is not None and dperf_term_diag is not None)
                else None
            )
        else:
            w_opt_b = torch.empty((b_c, n_tot), dtype=torch.float64, device=dev)
            dw_opt_b = (
                torch.empty((b_c, d_act, n_tot), dtype=torch.float64, device=dev)
                if compute_pathwise_delta
                else None
            )
            for (o_idx_g, p_key_g), idx_t in opt_groups_t:
                inv_p_g = perm_cache.get(p_key_g) if p_key_g is not None else None
                wo_g, gwo_g = _eval_index_and_grad_gpu(
                    perf_term[idx_t], o_idx_g, inv_p_g, compute_grad=compute_pathwise_delta, asset_dim=1
                )
                w_opt_b[idx_t] = wo_g
                if dw_opt_b is not None and gwo_g is not None and dperf_term_diag is not None:
                    dw_opt_b[idx_t] = gwo_g * dperf_term_diag[idx_t]

        strike_b = scal_gpu[8].view(b_c, 1)
        gearing_b = scal_gpu[9].view(b_c, 1)
        has_put_b = scal_gpu[20].view(b_c, 1) > 0.5
        low_k_b = scal_gpu[10].view(b_c, 1)
        has_low_b = scal_gpu[18].view(b_c, 1) > 0.5
        has_ki_b = scal_gpu[19].view(b_c, 1) > 0.5
        star_k_b = scal_gpu[13].view(b_c, 1)
        star_eps_b = scal_gpu[23].view(b_c, 1)
        is_long_b = scal_gpu[24].view(b_c, 1) > 0.5
        star_cpn_b = scal_gpu[14].view(b_c, 1)
        has_star_b = star_k_b > 0.0
        pcall_k_b = scal_gpu[15].view(b_c, 1)
        pcall_rate_b = scal_gpu[16].view(b_c, 1)
        has_pcall_b = scal_gpu[22].view(b_c, 1) > 0.5
        put_sign_b = scal_gpu[17].view(b_c, 1)

        has_any_low = bool(np.any(scal_np[18] > 0.5))
        has_any_ki = bool(np.any(scal_np[19] > 0.5))
        has_any_star = bool(np.any(scal_np[13] > 0.0))
        has_any_star_cpn = bool(np.any((scal_np[13] > 0.0) & (scal_np[14] != 0.0)))
        has_any_long = bool(np.any(scal_np[24] > 0.5))

        raw_put_b = torch.where(has_put_b, torch.clamp(strike_b - w_opt_b, min=0.0), torch.zeros_like(w_opt_b))
        d_put_dw_b = torch.where(has_put_b & (w_opt_b < strike_b), -torch.ones_like(w_opt_b), torch.zeros_like(w_opt_b))

        if has_any_low:
            cap_b = torch.clamp(strike_b - low_k_b, min=0.0)
            under_cap = raw_put_b < cap_b
            raw_put_b = torch.where(has_low_b, torch.minimum(raw_put_b, cap_b), raw_put_b)
            d_put_dw_b = torch.where(has_low_b & (~under_cap), torch.zeros_like(d_put_dw_b), d_put_dw_b)

        d_raw_put_b: Optional[torch.Tensor] = (
            (d_put_dw_b.unsqueeze(1) * dw_opt_b)
            if (compute_pathwise_delta and dw_opt_b is not None)
            else None
        )

        if has_any_ki:
            ki_up_b, dki_up_b = _digital_ramp_and_deriv_batch(
                w_opt_b, ki_bar_b, ki_spr_b, compute_deriv=compute_pathwise_delta
            )
            ki_ind_eu = 1.0 - ki_up_b
            ki_ind_disc = (
                (1.0 - not_ki_disc_b)
                if (has_any_disc_ki and not_ki_disc_b is not None)
                else ki_ind_eu
            )
            ki_nonlong_b = has_ki_b & (~is_long_b)
            ki_ind_b = torch.where(
                ki_nonlong_b,
                torch.where(is_disc_ki_b, ki_ind_disc, ki_ind_eu),
                torch.ones_like(ki_up_b),
            )
            raw_put_before_ki = raw_put_b
            if compute_pathwise_delta and d_raw_put_b is not None:
                d_ki_eu = (
                    (-dki_up_b).unsqueeze(1) * dw_opt_b
                    if (dki_up_b is not None and dw_opt_b is not None)
                    else torch.zeros_like(d_raw_put_b)
                )
                d_ki_disc = (
                    (-d_not_ki_disc_b)
                    if (has_any_disc_ki and d_not_ki_disc_b is not None)
                    else d_ki_eu
                )
                d_ki_b = torch.where(
                    ki_nonlong_b.unsqueeze(1),
                    torch.where(is_disc_ki_b.unsqueeze(1), d_ki_disc, d_ki_eu),
                    torch.zeros_like(d_raw_put_b),
                )
                d_raw_put_b = d_raw_put_b * ki_ind_b.unsqueeze(1) + raw_put_before_ki.unsqueeze(1) * d_ki_b
            raw_put_b = raw_put_before_ki * ki_ind_b

        if has_any_long:
            w_avg_b, gw_avg_b = _eval_index_and_grad_gpu(
                perf_term, "AverageOf", None, compute_grad=compute_pathwise_delta, asset_dim=1
            )
            w_wof_b, gw_wof_b = _eval_index_and_grad_gpu(
                perf_term, "WorstOf", None, compute_grad=compute_pathwise_delta, asset_dim=1
            )
            ki_wof_up, dki_wof_up = _digital_ramp_and_deriv_batch(
                w_wof_b, strike_b, ki_spr_b, compute_deriv=compute_pathwise_delta
            )
            ki_wof_eu = 1.0 - ki_wof_up
            ki_wof_k = (
                torch.where(is_disc_ki_b, 1.0 - not_ki_disc_b, ki_wof_eu)
                if (has_any_disc_ki and not_ki_disc_b is not None)
                else ki_wof_eu
            )
            gearing_safe = torch.where(gearing_b.abs() > 1e-12, gearing_b, torch.ones_like(gearing_b))
            raw_put_lb = ((1.0 - w_avg_b) * ki_wof_k) / gearing_safe
            raw_put_b = torch.where(is_long_b, raw_put_lb, raw_put_b)
            if (
                compute_pathwise_delta
                and d_raw_put_b is not None
                and gw_avg_b is not None
                and gw_wof_b is not None
                and dperf_term_diag is not None
            ):
                dw_avg_b = gw_avg_b * dperf_term_diag
                dw_wof_b = gw_wof_b * dperf_term_diag
                d_ki_wof_eu = (
                    (-dki_wof_up).unsqueeze(1) * dw_wof_b
                    if dki_wof_up is not None
                    else torch.zeros_like(d_raw_put_b)
                )
                d_ki_wof_k = (
                    torch.where(is_disc_ki_b.unsqueeze(1), -d_not_ki_disc_b, d_ki_wof_eu)
                    if (has_any_disc_ki and d_not_ki_disc_b is not None)
                    else d_ki_wof_eu
                )
                d_raw_put_lb = (
                    -dw_avg_b * ki_wof_k.unsqueeze(1) + (1.0 - w_avg_b).unsqueeze(1) * d_ki_wof_k
                ) / gearing_safe.unsqueeze(1)
                d_raw_put_b = torch.where(is_long_b.unsqueeze(1), d_raw_put_lb, d_raw_put_b)

        star_ind_b = torch.zeros_like(w_opt_b)
        dstar_b: Optional[torch.Tensor] = None
        if has_any_star:
            w_best_b, gw_best_b = _eval_index_and_grad_gpu(
                perf_term, "BestOf", None, compute_grad=compute_pathwise_delta, asset_dim=1
            )
            star_ind_b = torch.where(has_star_b, _onestar_ramp_torch(w_best_b, star_k_b, star_eps_b), star_ind_b)
            if compute_pathwise_delta and gw_best_b is not None and dperf_term_diag is not None:
                in_os = (w_best_b >= (star_k_b - 0.5 * star_eps_b)) & (w_best_b <= (star_k_b + 0.5 * star_eps_b))
                dstar_dwbest = torch.where(
                    has_star_b & in_os,
                    1.0 / star_eps_b,
                    torch.zeros_like(w_best_b),
                )
                dstar_b = dstar_dwbest.unsqueeze(1) * (gw_best_b * dperf_term_diag)
            raw_put_before_star = raw_put_b
            raw_put_b = raw_put_before_star * (1.0 - star_ind_b)
            if compute_pathwise_delta and d_raw_put_b is not None:
                d_raw_put_b = d_raw_put_b * (1.0 - star_ind_b).unsqueeze(1)
                if dstar_b is not None:
                    d_raw_put_b = d_raw_put_b - raw_put_before_star.unsqueeze(1) * dstar_b

        unit_put_b = alive * raw_put_b
        l_put_b = unit_put_b.mean(dim=1)
        unit_call_b = torch.where(
            has_pcall_b, alive * torch.clamp(w_opt_b - pcall_k_b, min=0.0), torch.zeros_like(alive)
        )

        df_term_b = df_pay_t[:, -1:]
        opt_pv_b = df_term_b * (put_sign_b * gearing_b * unit_put_b + pcall_rate_b * unit_call_b)
        if has_any_star_cpn:
            opt_pv_b = opt_pv_b + df_term_b * alive * star_ind_b * star_cpn_b
        total_pv = total_pv + opt_pv_b

        if (
            compute_pathwise_delta
            and dw_opt_b is not None
            and d_alive is not None
            and d_total_pv is not None
            and d_raw_put_b is not None
        ):
            d_unit_put_b = d_alive * raw_put_b.unsqueeze(1) + alive.unsqueeze(1) * d_raw_put_b
            dc_dw_b = torch.where(has_pcall_b & (w_opt_b > pcall_k_b), torch.ones_like(w_opt_b), torch.zeros_like(w_opt_b))
            c_int_b = torch.where(has_pcall_b, torch.clamp(w_opt_b - pcall_k_b, min=0.0), torch.zeros_like(w_opt_b))
            d_unit_call_b = (
                d_alive * c_int_b.unsqueeze(1)
                + (alive * dc_dw_b).unsqueeze(1) * dw_opt_b
            )
            d_opt_b = df_term_b.unsqueeze(1) * (
                (put_sign_b * gearing_b).unsqueeze(1) * d_unit_put_b
                + pcall_rate_b.unsqueeze(1) * d_unit_call_b
            )
            if has_any_star_cpn:
                d_star_term = d_alive * star_ind_b.unsqueeze(1)
                if dstar_b is not None:
                    d_star_term = d_star_term + alive.unsqueeze(1) * dstar_b
                d_opt_b = d_opt_b + df_term_b.unsqueeze(1) * (d_star_term * star_cpn_b.unsqueeze(1))
            d_total_pv = d_total_pv + d_opt_b

        pv_means = total_pv.mean(dim=1) + pv_in_trans_t
        pv_cpu = pv_means.detach().cpu()
        p_cpu = p_stop_b.detach().cpu()
        q_cpu = q_cpn_b.detach().cpu()
        l_cpu = l_put_b.detach().cpu()
        if compute_pathwise_delta and d_total_pv is not None and delta_out is not None:
            d_cpu = d_total_pv.mean(dim=2).detach().cpu()
        else:
            d_cpu = None

        for bi, orig_idx in enumerate(chunk_idx):
            pv_out[orig_idx] = pv_cpu[bi]
            p_out[orig_idx, :m_act] = p_cpu[bi]
            q_out[orig_idx, :m_act] = q_cpu[bi]
            l_out[orig_idx] = l_cpu[bi]
            if d_cpu is not None and delta_out is not None:
                delta_out[orig_idx, :d_act] = d_cpu[bi]


def _eval_index_and_grad_gpu(
    perf: torch.Tensor,
    index_name: str,
    inv_lex_perm_t: Optional[torch.Tensor],
    *,
    compute_grad: bool = False,
    asset_dim: int = -1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Evaluates monitoring functional and its exact pathwise subgradient dW/dP_sorted."""
    code = resolve_monitoring_index_code(index_name)
    is_asym = code in _PAIR_ORDER_INDEX_CODES
    p = perf.index_select(asset_dim, inv_lex_perm_t) if (is_asym and inv_lex_perm_t is not None) else perf
    d = int(p.shape[asset_dim])

    def _tie_weights(target_keepdim: torch.Tensor) -> torch.Tensor:
        ties = (p == target_keepdim).to(p.dtype)
        return ties / torch.clamp(ties.sum(dim=asset_dim, keepdim=True), min=1.0)

    if code == int(MonitoringIndexCode.WORST_OF):
        w_val, _ = p.min(dim=asset_dim)
        if not compute_grad:
            return w_val, None
        return w_val, _tie_weights(w_val.unsqueeze(asset_dim))
    if code == int(MonitoringIndexCode.BEST_OF):
        w_val, _ = p.max(dim=asset_dim)
        if not compute_grad:
            return w_val, None
        return w_val, _tie_weights(w_val.unsqueeze(asset_dim))
    if code == int(MonitoringIndexCode.AVERAGE_OF):
        w_val = p.mean(dim=asset_dim)
        if not compute_grad:
            return w_val, None
        return w_val, torch.full_like(p, 1.0 / float(max(d, 1)))
    if code == int(MonitoringIndexCode.DISPERSION):
        if d <= 1:
            w_val = torch.zeros_like(p.select(asset_dim, 0))
            return (w_val, torch.zeros_like(p)) if compute_grad else (w_val, None)
        diff = p - p.mean(dim=asset_dim, keepdim=True)
        w_val = torch.sqrt((diff * diff).mean(dim=asset_dim).clamp(min=0.0))
        if not compute_grad:
            return w_val, None
        w_unsq = w_val.unsqueeze(asset_dim)
        grad = torch.where(
            w_unsq > 1e-12,
            diff / (float(d) * w_unsq.clamp(min=1e-12)),
            torch.zeros_like(p),
        )
        return w_val, grad

    w_val = _evaluate_monitoring_index_gpu(perf, index_name, inv_lex_perm_t, asset_dim=asset_dim)
    if not compute_grad:
        return w_val, None

    grad = torch.zeros_like(p)
    if code == int(MonitoringIndexCode.RATIO_BEST_WORST):
        w_max = p.max(dim=asset_dim, keepdim=True).values
        w_min_raw = p.min(dim=asset_dim, keepdim=True).values
        w_min = w_min_raw.clamp(min=1e-12)
        grad = _tie_weights(w_max) / w_min - (w_max / (w_min * w_min)) * _tie_weights(w_min_raw)
    elif code == int(MonitoringIndexCode.RATIO_BEST_AVERAGE):
        w_max = p.max(dim=asset_dim, keepdim=True).values
        w_avg = p.mean(dim=asset_dim, keepdim=True).clamp(min=1e-12)
        grad = _tie_weights(w_max) / w_avg - (w_max / (w_avg * w_avg)) / float(max(d, 1))
    elif code == int(MonitoringIndexCode.RATIO_AVERAGE_WORST):
        w_avg = p.mean(dim=asset_dim, keepdim=True)
        w_min_raw = p.min(dim=asset_dim, keepdim=True).values
        w_min = w_min_raw.clamp(min=1e-12)
        grad = (1.0 / float(max(d, 1))) / w_min - (w_avg / (w_min * w_min)) * _tie_weights(w_min_raw)
    elif code == int(MonitoringIndexCode.DIFF_BEST_WORST):
        w_max = p.max(dim=asset_dim, keepdim=True).values
        w_min = p.min(dim=asset_dim, keepdim=True).values
        grad = _tie_weights(w_max) - _tie_weights(w_min)
    elif code == int(MonitoringIndexCode.DIFF_BEST_AVERAGE):
        w_max = p.max(dim=asset_dim, keepdim=True).values
        grad = _tie_weights(w_max) - 1.0 / float(max(d, 1))
    elif code == int(MonitoringIndexCode.DIFF_AVERAGE_WORST):
        w_min = p.min(dim=asset_dim, keepdim=True).values
        grad = 1.0 / float(max(d, 1)) - _tie_weights(w_min)
    elif code == int(MonitoringIndexCode.RATIO_AB):
        if d >= 2:
            b_safe = p.select(asset_dim, 1).clamp(min=1e-12)
            grad.select(asset_dim, 0).copy_(1.0 / b_safe)
            grad.select(asset_dim, 1).copy_(-p.select(asset_dim, 0) / (b_safe * b_safe))
        else:
            grad.select(asset_dim, 0).fill_(1.0)
    elif code == int(MonitoringIndexCode.RATIO_BA):
        if d >= 2:
            a_safe = p.select(asset_dim, 0).clamp(min=1e-12)
            grad.select(asset_dim, 1).copy_(1.0 / a_safe)
            grad.select(asset_dim, 0).copy_(-p.select(asset_dim, 1) / (a_safe * a_safe))
        else:
            grad.select(asset_dim, 0).fill_(1.0)
    elif code == int(MonitoringIndexCode.DIFF_AB):
        if d >= 2:
            grad.select(asset_dim, 0).fill_(1.0)
            grad.select(asset_dim, 1).fill_(-1.0)
    elif code == int(MonitoringIndexCode.DIFF_BA):
        if d >= 2:
            grad.select(asset_dim, 1).fill_(1.0)
            grad.select(asset_dim, 0).fill_(-1.0)
    elif code == int(MonitoringIndexCode.TWIN_WIN):
        w_min = p.min(dim=asset_dim, keepdim=True).values
        sign_tw = torch.where(w_min < 1.0, -1.0, 1.0).to(p.dtype)
        grad = sign_tw * _tie_weights(w_min)
    else:
        grad = _tie_weights(w_val.unsqueeze(asset_dim))

    if is_asym and inv_lex_perm_t is not None:
        grad_sorted = torch.zeros_like(grad)
        perm_shape = [1] * grad.ndim
        perm_shape[asset_dim] = d
        grad_sorted.scatter_(asset_dim, inv_lex_perm_t.view(*perm_shape).expand_as(grad), grad)
        return w_val, grad_sorted
    return w_val, grad


def _digital_ramp_and_deriv_batch(
    w_vals: torch.Tensor,     # (B_c, N)
    barrier: torch.Tensor,    # (B_c, 1)
    spread: torch.Tensor,     # (B_c, 1)
    *,
    compute_deriv: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Batched call-spread digital ramp and smoothed pathwise derivative wrt W."""
    abs_s = torch.abs(spread)
    hard_mask = abs_s < 1e-9
    b_low = torch.minimum(barrier, barrier + spread)
    b_high = torch.maximum(barrier, barrier + spread)
    width = torch.clamp(b_high - b_low, min=1e-9)
    ramp = torch.clamp((w_vals - b_low) / width, 0.0, 1.0)
    hard_ind = (w_vals >= barrier).to(w_vals.dtype)
    val = torch.where(hard_mask, hard_ind, ramp)

    if not compute_deriv:
        return val, None

    # Pathwise derivative: exact 1/width on [b_low, b_high] when spread != 0, else narrow 0.02 Gaussian kernel
    in_ramp = (w_vals >= b_low) & (w_vals <= b_high)
    deriv_ramp = torch.where(in_ramp, 1.0 / width, torch.zeros_like(w_vals))
    eps_pw = 0.02
    z_pw = (w_vals - barrier) / eps_pw
    deriv_hard = torch.exp(-0.5 * z_pw * z_pw) / (math.sqrt(2.0 * math.pi) * eps_pw)
    deriv = torch.where(hard_mask, deriv_hard, deriv_ramp)
    return val, deriv


class GpuTritonDupireQMC(GpuTorchDupireQMC):
    """GPU Triton-fused Sobol + Dupire Local-Vol RQMC backend."""

    def __init__(self, device: Union[str, torch.device] = "cuda") -> None:
        super().__init__(device=device, use_triton=True)


class DupireQMCGpuSolver:
    """Primary GPU/CPU Dupire RQMC facade auto-selecting Triton CUDA when available."""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        *,
        backend: Optional[Literal["auto", "triton", "torch", "cpu"]] = "auto",
    ) -> None:
        if device is None:
            eff_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            eff_dev = torch.device(device)
        self.device = eff_dev

        self._backend: DupireQMCBackend
        if backend == "cpu" or eff_dev.type != "cuda" or not torch.cuda.is_available():
            self._backend = CpuNumpyDupireQMC()
        elif backend == "torch" or not _HAS_TRITON:
            self._backend = GpuTorchDupireQMC(device=eff_dev, use_triton=False)
        else:
            self._backend = GpuTritonDupireQMC(device=eff_dev)

    @property
    def backend_name(self) -> str:
        return self._backend.backend_name

    def simulate_and_price(
        self,
        state: AutocallPreprocState,
        config: Optional[DupireQMCConfig] = None,
    ) -> DupireQMCResult:
        return self._backend.simulate_and_price(state, config=config)

    def simulate_batch_for_training(
        self,
        states: Sequence[AutocallPreprocState],
        config: Optional[DupireQMCConfig] = None,
        compute_pathwise_delta: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self._backend.simulate_batch_for_training(
            states, config=config, compute_pathwise_delta=compute_pathwise_delta
        )


__all__ = [
    "_arr_bytes_equal",
    "_fast_triton_launch",
    "_splitmix64_30bit_shifts",
    "_sobol_vec_flat",
    "resolve_dupire_qmc_config",
    "DupireQMCConfig",
    "DupireQMCBackend",
    "GpuQmcStaticSchedule",
    "GpuTritonDupireQMC",
    "GpuTorchDupireQMC",
    "CpuNumpyDupireQMC",
    "DupireQMCGpuSolver",
]
