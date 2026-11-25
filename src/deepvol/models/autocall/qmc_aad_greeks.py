"""Tier-4 GPU/CPU Dupire QMC Pathwise-AAD and Top-3 LOO/L2O CRN Greeks engine."""

from __future__ import annotations

import ctypes
import datetime as dt
import hashlib
import logging
import math
import os
import threading
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Literal, Optional, Sequence, Tuple, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch

try:
    import numba  # type: ignore[import-untyped]

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional CPU JIT dependency
    numba = None
    _HAS_NUMBA = False

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
    from triton import knobs as _triton_knobs
    from triton.runtime import driver as _triton_driver  # type: ignore[import-untyped]

    _HAS_TRITON = True
except Exception:  # pragma: no cover - optional GPU Triton dependency
    triton = None
    tl = None
    _triton_knobs = None
    _triton_driver = None
    _HAS_TRITON = False

from deepvol.models.autocall.dupire_qmc_gpu import _get_sobol_direction_numbers
from deepvol.models.autocall.feature_layout import MonitoringIndexCode, monitoring_index_to_code
from deepvol.models.autocall.gpu_cache_manager import (
    GLOBAL_GPU_CACHE_MANAGER,
    PartitionedGpuLRU,
    normalize_cuda_device,
)
from deepvol.models.autocall.pre_post_processor import assemble_autocall_cashflows_and_pv

_T = TypeVar("_T")
logger = logging.getLogger(__name__)

_INDEX_STRING_TO_CODE: Dict[str, int] = {
    "worstof": int(MonitoringIndexCode.WORST_OF),
    "lookback": int(MonitoringIndexCode.WORST_OF),
    "bestof": int(MonitoringIndexCode.BEST_OF),
    "average": int(MonitoringIndexCode.AVERAGE_OF),
    "averageof": int(MonitoringIndexCode.AVERAGE_OF),
    "multiperf": int(MonitoringIndexCode.AVERAGE_OF),
    "equallyweightedbasket": int(MonitoringIndexCode.AVERAGE_OF),
    "dispersion": int(MonitoringIndexCode.DISPERSION),
    "ratioab": int(MonitoringIndexCode.RATIO_AB),
    "ratio": int(MonitoringIndexCode.RATIO_AB),
    "ratioba": int(MonitoringIndexCode.RATIO_BA),
    "ratiobestworst": int(MonitoringIndexCode.RATIO_BEST_WORST),
    "ratiobestaverage": int(MonitoringIndexCode.RATIO_BEST_AVERAGE),
    "ratioaverageworst": int(MonitoringIndexCode.RATIO_AVERAGE_WORST),
    "diffab": int(MonitoringIndexCode.DIFF_AB),
    "outperformance": int(MonitoringIndexCode.DIFF_AB),
    "spread": int(MonitoringIndexCode.DIFF_AB),
    "diffba": int(MonitoringIndexCode.DIFF_BA),
    "diffbestworst": int(MonitoringIndexCode.DIFF_BEST_WORST),
    "rainbow": int(MonitoringIndexCode.DIFF_BEST_WORST),
    "spreadbestworst": int(MonitoringIndexCode.DIFF_BEST_WORST),
    "diffbestaverage": int(MonitoringIndexCode.DIFF_BEST_AVERAGE),
    "spreadbestaverage": int(MonitoringIndexCode.DIFF_BEST_AVERAGE),
    "diffaverageworst": int(MonitoringIndexCode.DIFF_AVERAGE_WORST),
    "spreadaverageworst": int(MonitoringIndexCode.DIFF_AVERAGE_WORST),
    "twinwin": int(MonitoringIndexCode.TWIN_WIN),
    "secondworstof": int(MonitoringIndexCode.SECOND_WORST_OF),
    "thirdworstof": int(MonitoringIndexCode.THIRD_WORST_OF),
    "secondbestof": int(MonitoringIndexCode.SECOND_BEST_OF),
}


def _resolve_internal_index_code(index_name: str | None, fallback_code: int = 0) -> int:
    """Maps a monitoring index string cleanly to canonical code 0..17."""
    if not index_name or not index_name.strip():
        return fallback_code
    key = index_name.strip().lower()
    if key in _INDEX_STRING_TO_CODE:
        return _INDEX_STRING_TO_CODE[key]
    code = monitoring_index_to_code(index_name)
    if code == int(MonitoringIndexCode.TWIN_WIN) and key != "twinwin":
        raise ValueError(f"Unsupported monitoring index {index_name!r}.")
    return int(code)


class _ThreadSafeLRUCache(Generic[_T]):
    """Bounded thread-safe OrderedDict cache guarded by RLock."""

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = maxsize
        self._lock = threading.RLock()
        self._data: OrderedDict[Any, _T] = OrderedDict()

    def get(self, key: Any) -> Optional[_T]:
        with self._lock:
            val = self._data.get(key)
            if val is not None:
                self._data.move_to_end(key)
            return val

    def put(self, key: Any, val: _T) -> None:
        with self._lock:
            self._data[key] = val
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_CRN_SOBOL_CACHE: PartitionedGpuLRU[Any] = GLOBAL_GPU_CACHE_MANAGER.register_partition(
    "sobol_normals", max_entries=256, max_bytes=20 * 1024 * 1024
)
_CRN_ZCORR_CACHE: PartitionedGpuLRU[Any] = GLOBAL_GPU_CACHE_MANAGER.register_partition(
    "sobol_normals", max_entries=16, max_bytes=20 * 1024 * 1024
)
_CRN_LV_CACHE: PartitionedGpuLRU[Any] = GLOBAL_GPU_CACHE_MANAGER.register_partition(
    "lv_table_cache", max_entries=512, max_bytes=16 * 1024 * 1024
)
_CRN_VEGA_JAC_CACHE: _ThreadSafeLRUCache[Any] = _ThreadSafeLRUCache(maxsize=4096)
_CRN_CURVE_JAC_CACHE: _ThreadSafeLRUCache[Any] = _ThreadSafeLRUCache(maxsize=2048)
_CRN_CPU_PREP_CACHE: _ThreadSafeLRUCache[Any] = _ThreadSafeLRUCache(maxsize=512)
_CRN_MF_CHEAP_CACHE: _ThreadSafeLRUCache[Any] = _ThreadSafeLRUCache(maxsize=512)
_CRN_DEAL_PREP_CACHE: PartitionedGpuLRU[Any] = GLOBAL_GPU_CACHE_MANAGER.register_partition(
    "deal_prep_lv_vega", max_entries=512, max_bytes=32 * 1024 * 1024
)
_CRN_STATIC_CACHE: PartitionedGpuLRU[Any] = GLOBAL_GPU_CACHE_MANAGER.register_partition(
    "static_constants", max_entries=256, max_bytes=8 * 1024 * 1024
)
_CRN_CUDAGRAPH_POOL_PARTITION: PartitionedGpuLRU[Any] = GLOBAL_GPU_CACHE_MANAGER.register_partition(
    "cudagraph_pool", max_entries=8, max_bytes=24 * 1024 * 1024
)
_CRN_TRITON_DIRECT_CACHE: _ThreadSafeLRUCache[Any] = _ThreadSafeLRUCache(maxsize=256)
_TRITON_PLAN_CACHE: _ThreadSafeLRUCache[Tuple[Tuple[str, bool, Any], ...]] = _ThreadSafeLRUCache(maxsize=512)


class _CrnGpuViews:
    """Non-dataclass holder for sliced GPU buffers backed by a compact tensor tuple in cache."""

    __slots__ = (
        "gpu_meta",
        "lv_f32_g",
        "lv_sp_pm_g",
        "lv_th_g",
        "chol_f32_g",
        "chol_f64_g",
        "tau_step_g",
        "rn_step_g",
        "rt_step_g",
        "fwd_perf_g",
        "fwd_th_g",
        "y_min_base_g",
        "inv_dy_base_g",
        "inv_dt_base_g",
        "id_d_g",
        "sp_pm_meta_g",
        "sched_f32_g",
        "eps_aad_g",
        "u_rot_gpu",
        "v_rot_gpu",
        "d_var_f32_g",
        "fwd_perf_f32_g",
        "df_pay_f32_g",
        "w_fund_f32_g",
        "ko_f32_g",
        "is_call_f32_g",
        "cpn_f32_g",
        "is_guar_f32_g",
        "cpn_rates_f32_g",
        "eps_f32_g",
        "payoff_f32_g",
        "chol_scaled_f32_g",
        "l_inv_t_g",
        "c_inv_g",
        "cdl_m_g",
        "cdl2_last_g",
        "fwd_sp_pm_g",
        "pre_pack",
        "stat_pack",
        "stat_pos_g",
        "stat_itr_g",
        "cv_eig_f32_g",
        "dyn_entries",
        "th_pack",
        "ko_th_g",
        "ic_th_g",
        "cp_th_g",
        "ig_th_g",
        "cr_th_g",
        "payoff_th_g",
        "eps_th_g",
        "df_th_g",
        "wf_th_g",
        "need_sp_cg_alt_g",
        "cross_pairs_vib",
        "w_t_bwd_g",
        "dummy_f32_1",
        "fwd_perf_f32_step_g",
        "inv_dt_f32_val",
        "cg_bucket_pack",
        "cg_meta_cache",
    )

    gpu_meta: Dict[str, Any]
    lv_f32_g: Any
    lv_sp_pm_g: Any
    lv_th_g: Any
    chol_f32_g: Any
    chol_f64_g: Any
    tau_step_g: Any
    rn_step_g: Any
    rt_step_g: Any
    fwd_perf_g: Any
    fwd_th_g: Any
    y_min_base_g: Any
    inv_dy_base_g: Any
    inv_dt_base_g: Any
    id_d_g: Any
    sp_pm_meta_g: Any
    sched_f32_g: Any
    eps_aad_g: Any
    u_rot_gpu: Any
    v_rot_gpu: Any
    d_var_f32_g: Any
    fwd_perf_f32_g: Any
    df_pay_f32_g: Any
    w_fund_f32_g: Any
    ko_f32_g: Any
    is_call_f32_g: Any
    cpn_f32_g: Any
    is_guar_f32_g: Any
    cpn_rates_f32_g: Any
    eps_f32_g: Any
    payoff_f32_g: Any
    chol_scaled_f32_g: Any
    l_inv_t_g: Any
    c_inv_g: Any
    cdl_m_g: Any
    cdl2_last_g: Any
    fwd_sp_pm_g: Any
    pre_pack: Any
    stat_pack: Any
    stat_pos_g: Any
    stat_itr_g: Any
    cv_eig_f32_g: Any
    dyn_entries: Any
    th_pack: Any
    ko_th_g: Any
    ic_th_g: Any
    cp_th_g: Any
    ig_th_g: Any
    cr_th_g: Any
    payoff_th_g: Any
    eps_th_g: Any
    df_th_g: Any
    wf_th_g: Any
    need_sp_cg_alt_g: bool
    cross_pairs_vib: Tuple[Tuple[int, int], ...]
    w_t_bwd_g: Any
    dummy_f32_1: Any
    fwd_perf_f32_step_g: Any
    inv_dt_f32_val: float
    cg_bucket_pack: Any
    cg_meta_cache: Dict[Any, Any]


class _CrnCpuPrep:
    """Non-dataclass holder for cached CPU deal-prep state across warm GPU evaluations."""

    __slots__ = (
        "lv_y_min_vec",
        "lv_y_max_vec",
        "t_max",
        "lv_t_grid",
        "u_time_np",
        "chol_base",
        "v_eig_np",
        "sqrt_ev_np",
        "r_native_obs_np",
        "esc_base",
        "growth_base",
        "w_fund_base_np",
        "base_lv_tables",
        "base_lv_caches",
        "vol_atm_obs",
        "th_prep",
        "th_chunk_in_step1",
        "d_var_gbm_np",
        "mf_cheap_digest",
        "use_mf_cv",
        "apply_ema_mf",
        "ema_group_mf",
        "n_rep_mf",
        "n_rep_gen_mf",
        "eff_r_mf",
        "sp_p_vec",
        "sp_m_vec",
        "cdl_m_np",
        "cdl2_last_np",
        "fwd_sp_p_np",
        "fwd_sp_m_np",
        "sp_task_meta",
        "adj_rho_tasks",
        "rho_tasks",
        "adj_repo_tasks",
        "repo_task_meta",
        "adj_div_task",
        "div_task_meta",
        "adj_fxc_tasks",
        "fxc_tasks",
        "adj_fxv_tasks",
        "fxv_tasks",
        "rho_dict_tmpl",
        "rho_arrays_tmpl",
        "repo_rho_has_map",
        "divs_dict_tmpl",
        "divs_arrays_tmpl",
        "fx_vega_has_map",
        "fx_cega_has_map",
        "need_pv0_eval",
        "tok_base",
        "gpu_eval_rows_total",
        "n_stat_rows",
        "n_pre_rows",
        "vib_tau_prev",
        "vib_d_tau",
        "vib_period_level",
        "vib_period_idx",
        "vib_sub_idx",
        "vib_step_start",
        "vib_n_steps",
        "h_vib_val",
        "df_last_val",
        "k_cross_g",
        "m_rem_th_g",
        "ko_arr_np",
        "cpn_arr_np",
        "is_call_np",
        "is_guar_np",
        "cpn_rates_np",
        "can_triton_sobol",
        "dim_coarse",
        "dim_fine",
        "sobol_key",
        "rot_sobol_key",
        "need_stage2_aad",
        "use_triton_aad_s2",
        "use_pathwise_rate_adj",
        "skip_step1_loo",
        "use_vibrato_gamma",
        "need_spot_bumps",
        "tau_obs_dig_base",
        "surf_dig_map",
        "stat_itr_np",
    )

    lv_y_min_vec: Any
    lv_y_max_vec: Any
    t_max: float
    lv_t_grid: Any
    u_time_np: Any
    chol_base: Any
    v_eig_np: Any
    sqrt_ev_np: Any
    r_native_obs_np: Any
    esc_base: Any
    growth_base: Any
    w_fund_base_np: Any
    base_lv_tables: Any
    base_lv_caches: Any
    vol_atm_obs: Any
    th_prep: Any
    th_chunk_in_step1: bool
    d_var_gbm_np: Any
    mf_cheap_digest: str
    use_mf_cv: bool
    apply_ema_mf: bool
    ema_group_mf: int
    n_rep_mf: int
    n_rep_gen_mf: int
    eff_r_mf: int
    sp_p_vec: Any
    sp_m_vec: Any
    cdl_m_np: Any
    cdl2_last_np: Any
    fwd_sp_p_np: Any
    fwd_sp_m_np: Any
    sp_task_meta: Any
    adj_rho_tasks: Any
    rho_tasks: Any
    adj_repo_tasks: Any
    repo_task_meta: Any
    adj_div_task: Any
    div_task_meta: Any
    adj_fxc_tasks: Any
    fxc_tasks: Any
    adj_fxv_tasks: Any
    fxv_tasks: Any
    rho_dict_tmpl: Any
    rho_arrays_tmpl: Any
    repo_rho_has_map: bool
    divs_dict_tmpl: Any
    divs_arrays_tmpl: Any
    fx_vega_has_map: bool
    fx_cega_has_map: bool
    need_pv0_eval: bool
    tok_base: Any
    gpu_eval_rows_total: int
    n_stat_rows: int
    n_pre_rows: int
    vib_tau_prev: Any
    vib_d_tau: Any
    vib_period_level: bool
    vib_period_idx: int
    vib_sub_idx: int
    vib_step_start: int
    vib_n_steps: int
    h_vib_val: float
    df_last_val: float
    k_cross_g: int
    m_rem_th_g: int
    ko_arr_np: Any
    cpn_arr_np: Any
    is_call_np: Any
    is_guar_np: Any
    cpn_rates_np: Any
    can_triton_sobol: bool
    dim_coarse: int
    dim_fine: int
    sobol_key: Any
    rot_sobol_key: Any
    need_stage2_aad: bool
    use_triton_aad_s2: bool
    use_pathwise_rate_adj: bool
    skip_step1_loo: bool
    use_vibrato_gamma: bool
    need_spot_bumps: bool
    tau_obs_dig_base: Any
    surf_dig_map: Any
    stat_itr_np: Any


def _seq_fast_key(v: Any) -> Any:
    if v is None:
        return None
    tv = type(v)
    if tv is tuple:
        return v
    if tv is list:
        return tuple(v)
    if isinstance(v, np.ndarray):
        return (v.shape, v.dtype.str, v.tobytes())
    return repr(v)


def _curve_fast_sig(c_any: Any, sig_memo: Optional[Dict[int, Any]] = None) -> Any:
    """Fast deterministic content signature for discount/financing/repo curves without full serialization."""
    if c_any is None:
        return None
    if sig_memo is not None:
        hit_m = sig_memo.get(id(c_any))
        if hit_m is not None:
            return hit_m
    if isinstance(c_any, Mapping):
        rs = c_any.get("RatesSchedule")
        rs_m = rs if isinstance(rs, Mapping) else {}
        sig_out: Any = (
            "dict",
            str(c_any.get("Identifier", c_any.get("identifier", ""))),
            str(c_any.get("DayCountConvention", "")),
            str(c_any.get("InterpolationDayCountConvention", "")),
            str(c_any.get("InterpolationMethod", "")),
            str(c_any.get("InterpolationSpaceCoordinates", "")),
            str(c_any.get("ExtrapolationMethod", "")),
            str(c_any.get("RateQuoteExpression", "")),
            repr(c_any.get("AsOfDate", c_any.get("as_of_date", ""))),
            _seq_fast_key(c_any.get("Dates")),
            _seq_fast_key(c_any.get("Periods")),
            _seq_fast_key(c_any.get("DiscountFactors")),
            _seq_fast_key(c_any.get("Spreads")),
            _seq_fast_key(c_any.get("Rates")),
            _seq_fast_key(rs_m.get("Dates")),
            _seq_fast_key(rs_m.get("Periods")),
            _seq_fast_key(rs_m.get("Rates")),
            _seq_fast_key(rs_m.get("Spreads")),
            _seq_fast_key(rs_m.get("DiscountFactors")),
        )
        if sig_memo is not None:
            sig_memo[id(c_any)] = sig_out
        return sig_out
    d_any = getattr(c_any, "__dict__", None)
    if d_any is not None and hasattr(type(c_any), "model_fields"):
        cn = d_any.get("_inner_curve") or c_any
    else:
        cn = getattr(c_any, "_inner_curve", None) or c_any
    if hasattr(type(cn), "model_fields"):
        d_cn = getattr(cn, "__dict__", None)
        if d_cn is not None:
            extra_cn = getattr(cn, "__pydantic_extra__", None)
            if extra_cn:
                def _cn_get(key: str, default: Any = None) -> Any:
                    if key in d_cn:
                        return d_cn[key]
                    return extra_cn.get(key, default)
            else:
                _cn_get = d_cn.get
            rs_attr = _cn_get("RatesSchedule")
            if (
                rs_attr is not None
                or ("DiscountFactors" in d_cn)
                or ("Periods" in d_cn)
                or (extra_cn is not None and ("DiscountFactors" in extra_cn or "Periods" in extra_cn))
            ):
                snap_pm = getattr(c_any, "_crn_curve_pm_snap", None)
                if snap_pm is not None and snap_pm[0] is cn:
                    _, sig_snap_pm, cn_seq_cur_keys, rs_attr_snap, rs_seq_snap, id_snap, asof_snap = snap_pm
                    cur_id_pm = _cn_get("Identifier")
                    if cur_id_pm is None:
                        cur_id_pm = _cn_get("identifier")
                    cur_asof_pm = _cn_get("AsOfDate")
                    if cur_asof_pm is None:
                        cur_asof_pm = _cn_get("as_of_date", "")
                    if cur_id_pm == id_snap and cur_asof_pm == asof_snap and rs_attr is rs_attr_snap:
                        cn_ok_pm = (
                            _cn_get("Dates") == cn_seq_cur_keys[0]
                            and _cn_get("Periods") == cn_seq_cur_keys[1]
                            and _cn_get("DiscountFactors") == cn_seq_cur_keys[2]
                            and _cn_get("Spreads") == cn_seq_cur_keys[3]
                        )
                        if cn_ok_pm:
                            if rs_attr is None:
                                if sig_memo is not None:
                                    sig_memo[id(c_any)] = sig_snap_pm
                                return sig_snap_pm
                            d_rs_c = getattr(rs_attr, "__dict__", None) if not isinstance(rs_attr, Mapping) else None
                            if d_rs_c is not None and not getattr(rs_attr, "__pydantic_extra__", None):
                                if (
                                    d_rs_c.get("Dates") == rs_seq_snap[0]
                                    and d_rs_c.get("Periods") == rs_seq_snap[1]
                                    and d_rs_c.get("Rates") == rs_seq_snap[2]
                                    and d_rs_c.get("Spreads") == rs_seq_snap[3]
                                    and d_rs_c.get("DiscountFactors") == rs_seq_snap[4]
                                ):
                                    if sig_memo is not None:
                                        sig_memo[id(c_any)] = sig_snap_pm
                                    return sig_snap_pm
                id_cn = _cn_get("Identifier")
                if id_cn is None:
                    id_cn = _cn_get("identifier")
                if id_cn is None and d_any is not None:
                    id_cn = d_any.get("Identifier", d_any.get("identifier", ""))
                if id_cn is None:
                    id_cn = ""
                asof_cn = _cn_get("AsOfDate")
                if asof_cn is None:
                    asof_cn = _cn_get("as_of_date", "")
                if rs_attr is not None:
                    if isinstance(rs_attr, Mapping):
                        rs_dates = _seq_fast_key(rs_attr.get("Dates"))
                        rs_periods = _seq_fast_key(rs_attr.get("Periods"))
                        rs_rates = _seq_fast_key(rs_attr.get("Rates"))
                        rs_spreads = _seq_fast_key(rs_attr.get("Spreads"))
                        rs_dfs = _seq_fast_key(rs_attr.get("DiscountFactors"))
                    elif hasattr(type(rs_attr), "model_fields") and getattr(rs_attr, "__dict__", None) is not None:
                        d_rs = rs_attr.__dict__
                        extra_rs = getattr(rs_attr, "__pydantic_extra__", None)
                        if extra_rs:
                            rs_dates = _seq_fast_key(d_rs["Dates"] if "Dates" in d_rs else extra_rs.get("Dates"))
                            rs_periods = _seq_fast_key(d_rs["Periods"] if "Periods" in d_rs else extra_rs.get("Periods"))
                            rs_rates = _seq_fast_key(d_rs["Rates"] if "Rates" in d_rs else extra_rs.get("Rates"))
                            rs_spreads = _seq_fast_key(d_rs["Spreads"] if "Spreads" in d_rs else extra_rs.get("Spreads"))
                            rs_dfs = _seq_fast_key(d_rs["DiscountFactors"] if "DiscountFactors" in d_rs else extra_rs.get("DiscountFactors"))
                        else:
                            rs_dates = _seq_fast_key(d_rs.get("Dates"))
                            rs_periods = _seq_fast_key(d_rs.get("Periods"))
                            rs_rates = _seq_fast_key(d_rs.get("Rates"))
                            rs_spreads = _seq_fast_key(d_rs.get("Spreads"))
                            rs_dfs = _seq_fast_key(d_rs.get("DiscountFactors"))
                    else:
                        rs_dates = _seq_fast_key(getattr(rs_attr, "Dates", None))
                        rs_periods = _seq_fast_key(getattr(rs_attr, "Periods", None))
                        rs_rates = _seq_fast_key(getattr(rs_attr, "Rates", None))
                        rs_spreads = _seq_fast_key(getattr(rs_attr, "Spreads", None))
                        rs_dfs = _seq_fast_key(getattr(rs_attr, "DiscountFactors", None))
                else:
                    rs_dates = rs_periods = rs_rates = rs_spreads = rs_dfs = None
                sig_rs: Any = (
                    "model",
                    type(c_any).__name__,
                    type(cn).__name__,
                    str(id_cn),
                    str(_cn_get("DayCountConvention", "")),
                    str(_cn_get("InterpolationDayCountConvention", "")),
                    str(_cn_get("InterpolationMethod", "")),
                    str(_cn_get("InterpolationSpaceCoordinates", "")),
                    str(_cn_get("ExtrapolationMethod", "")),
                    str(_cn_get("RateQuoteExpression", "")),
                    repr(asof_cn),
                    _seq_fast_key(_cn_get("Dates")),
                    _seq_fast_key(_cn_get("Periods")),
                    _seq_fast_key(_cn_get("DiscountFactors")),
                    _seq_fast_key(_cn_get("Spreads")),
                    rs_dates,
                    rs_periods,
                    rs_rates,
                    rs_spreads,
                    rs_dfs,
                )
                try:
                    def _copy_seq_snap(v: Any) -> Any:
                        return list(v) if isinstance(v, list) else v

                    d_rs_snap = (
                        getattr(rs_attr, "__dict__", None)
                        if (rs_attr is not None and not isinstance(rs_attr, Mapping) and not getattr(rs_attr, "__pydantic_extra__", None))
                        else None
                    )
                    if (
                        all(isinstance(x, (type(None), tuple, list)) for x in (_cn_get("Dates"), _cn_get("Periods"), _cn_get("DiscountFactors"), _cn_get("Spreads")))
                        and (
                            rs_attr is None
                            or (
                                d_rs_snap is not None
                                and all(isinstance(d_rs_snap.get(k_rs), (type(None), tuple, list)) for k_rs in ("Dates", "Periods", "Rates", "Spreads", "DiscountFactors"))
                            )
                        )
                    ):
                        cn_snap_t = (
                            _copy_seq_snap(_cn_get("Dates")),
                            _copy_seq_snap(_cn_get("Periods")),
                            _copy_seq_snap(_cn_get("DiscountFactors")),
                            _copy_seq_snap(_cn_get("Spreads")),
                        )
                        rs_snap_t = (
                            (
                                _copy_seq_snap(d_rs_snap.get("Dates")),
                                _copy_seq_snap(d_rs_snap.get("Periods")),
                                _copy_seq_snap(d_rs_snap.get("Rates")),
                                _copy_seq_snap(d_rs_snap.get("Spreads")),
                                _copy_seq_snap(d_rs_snap.get("DiscountFactors")),
                            )
                            if d_rs_snap is not None
                            else ()
                        )
                        raw_id_snap = _cn_get("Identifier")
                        if raw_id_snap is None:
                            raw_id_snap = _cn_get("identifier")
                        raw_asof_snap = _cn_get("AsOfDate")
                        if raw_asof_snap is None:
                            raw_asof_snap = _cn_get("as_of_date", "")
                        setattr(c_any, "_crn_curve_pm_snap", (cn, sig_rs, cn_snap_t, rs_attr, rs_snap_t, raw_id_snap, raw_asof_snap))
                except (AttributeError, TypeError):
                    pass
                if sig_memo is not None:
                    sig_memo[id(c_any)] = sig_rs
                return sig_rs
    rs_attr = getattr(cn, "RatesSchedule", None)
    if rs_attr is not None or hasattr(cn, "DiscountFactors") or hasattr(cn, "Periods"):
        return (
            "model",
            type(c_any).__name__,
            type(cn).__name__,
            str(getattr(cn, "Identifier", getattr(cn, "identifier", getattr(c_any, "Identifier", getattr(c_any, "identifier", ""))))),
            str(getattr(cn, "DayCountConvention", "")),
            str(getattr(cn, "InterpolationDayCountConvention", "")),
            str(getattr(cn, "InterpolationMethod", "")),
            str(getattr(cn, "InterpolationSpaceCoordinates", "")),
            str(getattr(cn, "ExtrapolationMethod", "")),
            str(getattr(cn, "RateQuoteExpression", "")),
            repr(getattr(cn, "AsOfDate", getattr(cn, "as_of_date", ""))),
            _seq_fast_key(getattr(cn, "Dates", None)),
            _seq_fast_key(getattr(cn, "Periods", None)),
            _seq_fast_key(getattr(cn, "DiscountFactors", None)),
            _seq_fast_key(getattr(cn, "Spreads", None)),
            _seq_fast_key(getattr(rs_attr, "Dates", None) if rs_attr is not None else None),
            _seq_fast_key(getattr(rs_attr, "Periods", None) if rs_attr is not None else None),
            _seq_fast_key(getattr(rs_attr, "Rates", None) if rs_attr is not None else None),
            _seq_fast_key(getattr(rs_attr, "Spreads", None) if rs_attr is not None else None),
            _seq_fast_key(getattr(rs_attr, "DiscountFactors", None) if rs_attr is not None else None),
        )
    qa = getattr(cn, "_raw_quotes", getattr(cn, "quotes", getattr(cn, "values", getattr(cn, "rates", None))))
    ta = getattr(cn, "tenors", None)
    snap = d_any.get("_crn_curve_sig_snap") if d_any is not None else getattr(c_any, "_crn_curve_sig_snap", None)
    if snap is not None and len(snap) == 6 and snap[4] is cn:
        s_sig, q_snap, t_snap, raw_meta_snap, _, _ = snap
        raw_meta_cur = (
            getattr(cn, "dcc", getattr(c_any, "dcc", "")),
            getattr(cn, "space_coords", getattr(c_any, "space_coords", "")),
            getattr(cn, "_input_mode", getattr(c_any, "_input_mode", "")),
            getattr(cn, "quote_scale", getattr(c_any, "quote_scale", 0.01)),
            getattr(cn, "interp_method", getattr(cn, "interpolation_method", getattr(c_any, "interp_method", ""))),
            getattr(cn, "extrap_method", getattr(cn, "extrapolation_method", getattr(c_any, "extrap_method", ""))),
            getattr(cn, "as_of_date", getattr(c_any, "as_of_date", None)),
        )
        if raw_meta_cur == raw_meta_snap:
            if qa is None and q_snap is None:
                q_ok = True
            elif isinstance(qa, np.ndarray) and isinstance(q_snap, np.ndarray):
                q_ok = _arr_bytes_equal(qa, q_snap)
            elif qa is not None and q_snap is not None:
                q_ok = bool(np.array_equal(qa, q_snap))
            else:
                q_ok = False
            if q_ok:
                t_ok = False
                if ta is None and t_snap is None:
                    t_ok = True
                elif isinstance(ta, np.ndarray) and isinstance(t_snap, np.ndarray):
                    t_ok = _arr_bytes_equal(ta, t_snap)
                elif (not isinstance(ta, np.ndarray)) and (not isinstance(t_snap, np.ndarray)):
                    t_ok = bool(ta == t_snap)
                if t_ok:
                    if sig_memo is not None:
                        sig_memo[id(c_any)] = s_sig
                    return s_sig
    raw_meta_cur = (
        getattr(cn, "dcc", getattr(c_any, "dcc", "")),
        getattr(cn, "space_coords", getattr(c_any, "space_coords", "")),
        getattr(cn, "_input_mode", getattr(c_any, "_input_mode", "")),
        getattr(cn, "quote_scale", getattr(c_any, "quote_scale", 0.01)),
        getattr(cn, "interp_method", getattr(cn, "interpolation_method", getattr(c_any, "interp_method", ""))),
        getattr(cn, "extrap_method", getattr(cn, "extrapolation_method", getattr(c_any, "extrap_method", ""))),
        getattr(cn, "as_of_date", getattr(c_any, "as_of_date", None)),
    )
    dcc_v = str(raw_meta_cur[0])
    sc_v = str(raw_meta_cur[1])
    im_v = str(raw_meta_cur[2])
    qs_v = float(raw_meta_cur[3])
    interp_v = str(raw_meta_cur[4])
    extrap_v = str(raw_meta_cur[5])
    asof_v = repr(raw_meta_cur[6])
    meta_v = (dcc_v, sc_v, im_v, qs_v, interp_v, extrap_v, asof_v)
    if snap is not None and len(snap) == 4:
        s_sig, q_snap, t_snap, meta_snap = snap
        if meta_snap == meta_v:
            if qa is None and q_snap is None:
                q_ok = True
            elif qa is not None and q_snap is not None:
                q_ok = bool(np.array_equal(qa, q_snap))
            else:
                q_ok = False
            if q_ok:
                if ta is None and t_snap is None:
                    if sig_memo is not None:
                        sig_memo[id(c_any)] = s_sig
                    return s_sig
                if isinstance(ta, np.ndarray) and isinstance(t_snap, np.ndarray):
                    if np.array_equal(ta, t_snap):
                        if sig_memo is not None:
                            sig_memo[id(c_any)] = s_sig
                        return s_sig
                elif (not isinstance(ta, np.ndarray)) and (not isinstance(t_snap, np.ndarray)):
                    if ta == t_snap:
                        if sig_memo is not None:
                            sig_memo[id(c_any)] = s_sig
                        return s_sig
    q_arr = np.ascontiguousarray(qa, dtype=np.float64) if qa is not None else None
    qb = q_arr.tobytes() if q_arr is not None else b""
    tb = np.ascontiguousarray(ta).tobytes() if isinstance(ta, np.ndarray) else repr(ta).encode("utf-8")
    sig = (
        type(c_any).__name__,
        type(cn).__name__,
        str(getattr(c_any, "identifier", getattr(c_any, "Identifier", getattr(cn, "identifier", "")))),
        dcc_v,
        sc_v,
        im_v,
        qs_v,
        interp_v,
        extrap_v,
        asof_v,
        tb,
        qb,
    )
    try:
        setattr(
            c_any,
            "_crn_curve_sig_snap",
            (sig, q_arr.copy() if q_arr is not None else None, ta.copy() if isinstance(ta, np.ndarray) else ta, raw_meta_cur, cn, meta_v),
        )
    except (AttributeError, TypeError):
        return sig
    return sig


def clear_qmc_aad_gpu_caches() -> None:
    """Releases GPU-resident QMC-AAD cache entries to bound peak CUDA allocations."""
    _CRN_SOBOL_CACHE.clear()
    _CRN_ZCORR_CACHE.clear()
    _CRN_DEAL_PREP_CACHE.clear()
    _CRN_LV_CACHE.clear()
    _CRN_STATIC_CACHE.clear()
    _CRN_CPU_PREP_CACHE.clear()
    _CRN_MF_CHEAP_CACHE.clear()
    _CRN_QMC_CG_CACHE.clear()
    _QMC_CG_SCRATCH_ARENA.clear()
    _QMC_CG_ARENA_KEEPALIVE.clear()
    _QMC_CG_STREAMS.clear()


_QMC_CG_MAX_ENTRIES: int = 8


def _resolve_qmc_cg_bucket(d_assets: int, m_rem: int, n_sub: int) -> Optional[Tuple[int, int]]:
    """Maps heterogeneous `(d_assets, m_rem, n_sub)` deals into <= 7 canonical CUDAGraph buckets."""
    if d_assets <= 0 or d_assets > 6 or m_rem <= 0 or m_rem > 60:
        return None
    if m_rem <= 10 and n_sub == 8:
        return (6, 10)
    if n_sub != 4:
        return None
    d_pad = 3 if d_assets <= 3 else 6
    if m_rem <= 20:
        return (d_pad, 20)
    if m_rem <= 40:
        return (d_pad, 40)
    return (d_pad, 60)


class _QmcCudagraphCache:
    """Bounded (<=8 entry) shape-keyed CUDAGraph LRU backed by GLOBAL_GPU_CACHE_MANAGER cudagraph_pool."""

    def __init__(self, max_entries: int = _QMC_CG_MAX_ENTRIES) -> None:
        self._part = GLOBAL_GPU_CACHE_MANAGER.register_cudagraph_pool(max_entries=max_entries)

    def get(self, key: Any) -> Optional[Dict[str, Any]]:
        return cast(Optional[Dict[str, Any]], self._part.get(key))

    def put(self, key: Any, entry: Dict[str, Any]) -> None:
        out_t = entry.get("out_flat_g")
        exp_bytes = int(out_t.numel() * out_t.element_size()) if isinstance(out_t, torch.Tensor) else 0
        self._part.put(key, entry, explicit_bytes=exp_bytes)

    def clear(self) -> None:
        self._part.clear()


_CRN_QMC_CG_CACHE = _QmcCudagraphCache(max_entries=_QMC_CG_MAX_ENTRIES)
_QMC_CG_SCRATCH_ARENA: Dict[Tuple[int, str], torch.Tensor] = {}
_QMC_CG_ARENA_KEEPALIVE: List[torch.Tensor] = []
_QMC_CG_STREAMS: Dict[int, torch.cuda.Stream] = {}
GLOBAL_GPU_CACHE_MANAGER.register_on_clear_callback(clear_qmc_aad_gpu_caches)


def _get_qmc_cg_stream(device: torch.device) -> torch.cuda.Stream:
    dev_idx = int(device.index or 0)
    s = _QMC_CG_STREAMS.get(dev_idx)
    if s is None or s.device != device:
        s = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        _QMC_CG_STREAMS[dev_idx] = s
    return s


def _alloc_cg_buf(
    use_arena: bool,
    device: torch.device,
    slot: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    *,
    zero: bool = False,
) -> torch.Tensor:
    """Slices a pointer-stable pre-allocated CG scratch buffer when large enough, else falls back to torch alloc."""
    if use_arena:
        n_el = 1
        for d_s in shape:
            n_el *= int(d_s)
        elem_sz = 8 if dtype == torch.float64 else 4
        n_bytes = n_el * elem_sz
        buf = _QMC_CG_SCRATCH_ARENA.get((int(device.index or 0), slot))
        if buf is not None and int(buf.numel()) >= n_bytes and buf.device == device:
            view = buf[:n_bytes].view(dtype).view(shape)
            if zero:
                view.zero_()
            return view
    return torch.zeros(shape, dtype=dtype, device=device) if zero else torch.empty(shape, dtype=dtype, device=device)


def _ensure_qmc_cg_scratch_arena(
    device: torch.device,
    max_d: int,
    max_m: int,
    n_sub: int,
    max_r_chunk: int,
    n_rep: int,
    lv_nt: int,
    lv_ny: int,
    n_lv_shards: int,
    max_nb: int,
    n_tiles_eval: int,
    max_m_th: int,
    max_d_mf: int,
) -> None:
    """Pre-allocates fixed pointer-stable scratch buffers outside CUDAGraph capture so all graphs share one workspace."""
    dev_idx = int(device.index or 0)
    # Unified canonical high-water budget (c_m_steps * c_d * c_n_chunk <= 6*60*4*2*1024) fits (6,60,r=2) and (6,40,r=2)/(6,20,r=4).
    c_d = max(int(max_d), 6)
    c_m = max(int(max_m), 60)
    c_dm_steps = max(int(max_d) * int(max_m) * int(n_sub) * int(max_r_chunk) * int(n_rep), 6 * 60 * 4 * 2 * 1024)
    c_dm_per = max(int(max_d) * int(max_m) * int(max_r_chunk) * int(n_rep), 6 * 60 * 2 * 1024)
    c_dm_th = max(int(max_d) * int(max_m_th) * int(max_r_chunk) * int(n_rep), c_dm_per)
    c_n_chunk = max(int(max_r_chunk) * int(n_rep), 8192)
    c_shards = max(int(n_lv_shards), 128)
    c_grid_s2 = max(1, int(max_r_chunk), 8) * max(1, (int(n_rep) + 127) // 128)
    c_grid_nv = (c_n_chunk + 31) // 32
    slots_bytes = {
        # z_raw (inside _get_z_indep_chunk_gpu) and bar_zc (stage-2 bwd) have disjoint lifetimes.
        "z_raw_or_bzc": c_dm_steps * 4,
        "z_out": c_dm_steps * 4,
        "z_corr": c_dm_steps * 4,
        "lx_period": c_dm_per * 4,
        "perf_base": c_dm_per * 4,
        "vib_lx": c_d * c_n_chunk * 4,
        "vib_j": c_d * c_n_chunk * 4,
        "vib_sig": c_d * c_n_chunk * 4,
        "bar_lx_post": c_dm_per * 4,
        "out_cf_wf": 2 * c_grid_s2 * c_m * 8,
        "bar_q": max(c_d * c_m * int(n_sub), 6 * 60 * 4) * 4,
        "bar_lv": c_shards * c_d * max(int(lv_nt), 32) * max(int(lv_ny), 64) * 4,
        "d_ln_surv": (2 * c_d) * c_n_chunk * 4,
        "h_vib_out": c_grid_nv * c_d * c_d * 8,
        "frozen_term": 4 * c_n_chunk * 8,
        "ema_shift": 2 * max(int(max_r_chunk), 8) * c_m * max(int(max_d_mf), c_d) * 4,
        # z_mean_aux (get_z), ws_s2 (stage-2 AAD), perf_sp_pm, and perf_th have disjoint lifetimes within each chunk.
        "perf_aux": max(2 * c_dm_per * 4, c_dm_th * 4, 7 * c_m * max(int(max_r_chunk) * int(n_rep), 4096) * 4),
        "pv_tiles": max(int(max_nb), 64) * max(int(n_tiles_eval), 32) * 8,
    }
    cleared_graphs = False
    for s_name, s_bytes in slots_bytes.items():
        k_ar = (dev_idx, s_name)
        cur = _QMC_CG_SCRATCH_ARENA.get(k_ar)
        if cur is None or cur.device != device or int(cur.numel()) < int(s_bytes):
            if cur is not None and cur.device == device:
                if not cleared_graphs:
                    _get_qmc_cg_stream(device).synchronize()
                    _CRN_QMC_CG_CACHE.clear()
                    cleared_graphs = True
                _QMC_CG_ARENA_KEEPALIVE.append(cur)
            _QMC_CG_SCRATCH_ARENA[k_ar] = torch.empty((max(256, int(s_bytes)),), dtype=torch.uint8, device=device)


class _PlatenSdeAadArgsC(ctypes.Structure):
    """C++23 ABI v2 arguments for crn_platen_sde_aad_cpp (sizeof == 272)."""

    _fields_ = [
        ("d_assets", ctypes.c_int64),
        ("m_rem", ctypes.c_int64),
        ("n_sub", ctypes.c_int64),
        ("eff_r", ctypes.c_int64),
        ("n_rep", ctypes.c_int64),
        ("lv_nt", ctypes.c_int64),
        ("lv_ny", ctypes.c_int64),
        ("lv_tables", ctypes.c_void_p),
        ("z_indep", ctypes.c_void_p),
        ("chol_base", ctypes.c_void_p),
        ("tau_obs", ctypes.c_void_p),
        ("r_native_obs", ctypes.c_void_p),
        ("r_tot_obs", ctypes.c_void_p),
        ("lv_t_grid", ctypes.c_void_p),
        ("lv_y_min", ctypes.c_void_p),
        ("lv_y_max", ctypes.c_void_p),
        ("bar_lx_post", ctypes.c_void_p),
        ("lx_norm_in", ctypes.c_void_p),
        ("q_shift_sub", ctypes.c_void_p),
        ("lx_period_out", ctypes.c_void_p),
        ("norm_x_out", ctypes.c_void_p),
        ("bar_lv_out", ctypes.c_void_p),
        ("bar_chol_out", ctypes.c_void_p),
        ("bar_q_shift_out", ctypes.c_void_p),
        ("run_forward", ctypes.c_int32),
        ("run_backward", ctypes.c_int32),
        ("compute_lv_aad", ctypes.c_int32),
        ("compute_chol_aad", ctypes.c_int32),
        ("compute_q_aad", ctypes.c_int32),
        ("has_q_shift_sub", ctypes.c_int32),
        ("out_lx_pre", ctypes.c_void_p),
        ("out_j_pre", ctypes.c_void_p),
        ("out_sig_pre", ctypes.c_void_p),
        ("vib_period_idx", ctypes.c_int32),
        ("vib_sub_idx", ctypes.c_int32),
        ("save_vib_pre", ctypes.c_int32),
        ("_pad_vib", ctypes.c_int32),
        ("vib_j_pre_max", ctypes.c_double),
        ("vib_dsig_dy_max", ctypes.c_double),
    ]


class _SobolGenArgsC(ctypes.Structure):
    """C++23 ABI v2 arguments for crn_generate_sobol_normals_cpp (sizeof == 128)."""

    _fields_ = [
        ("d_assets", ctypes.c_int64),
        ("m_rem", ctypes.c_int64),
        ("n_sub", ctypes.c_int64),
        ("eff_r", ctypes.c_int64),
        ("n_rep", ctypes.c_int64),
        ("sobol_dim", ctypes.c_int64),
        ("n_bits", ctypes.c_int64),
        ("dir_numbers", ctypes.c_void_p),
        ("digital_shifts", ctypes.c_void_p),
        ("bb_ortho_mat", ctypes.c_void_p),
        ("chol_mat", ctypes.c_void_p),
        ("z_indep_out", ctypes.c_void_p),
        ("z_corr_out", ctypes.c_void_p),
        ("use_gray_code", ctypes.c_int32),
        ("apply_bb_time", ctypes.c_int32),
        ("apply_corr", ctypes.c_int32),
        ("antithetic", ctypes.c_int32),
        ("_reserved", ctypes.c_int32 * 2),
    ]


class _VibratoGammaArgsC(ctypes.Structure):
    """C++23 ABI v2 arguments for crn_eval_vibrato_gamma_cpp (sizeof == 392)."""

    _fields_ = [
        ("d_assets", ctypes.c_int64),
        ("m_rem", ctypes.c_int64),
        ("n_paths_total", ctypes.c_int64),
        ("vib_step_start", ctypes.c_int64),
        ("vib_n_steps", ctypes.c_int64),
        ("perf_mat", ctypes.c_void_p),
        ("lx_pre", ctypes.c_void_p),
        ("j_pre", ctypes.c_void_p),
        ("sig_pre", ctypes.c_void_p),
        ("sv_vib", ctypes.c_void_p),
        ("chol_mat", ctypes.c_void_p),
        ("l_inv_t", ctypes.c_void_p),
        ("c_inv", ctypes.c_void_p),
        ("fwd_term", ctypes.c_void_p),
        ("cdl_m", ctypes.c_void_p),
        ("cdl2_last", ctypes.c_void_p),
        ("ko_barriers", ctypes.c_void_p),
        ("is_callable", ctypes.c_void_p),
        ("cpn_barriers", ctypes.c_void_p),
        ("is_guaranteed_cpn", ctypes.c_void_p),
        ("coupon_rates", ctypes.c_void_p),
        ("eps_vec", ctypes.c_void_p),
        ("inv_lex_perm", ctypes.c_void_p),
        ("frozen_term_out", ctypes.c_void_p),
        ("h_vib_out", ctypes.c_void_p),
        ("df_last", ctypes.c_double),
        ("h_vib", ctypes.c_double),
        ("ko_spread", ctypes.c_double),
        ("cpn_spread", ctypes.c_double),
        ("ki_spread", ctypes.c_double),
        ("one_star_strike", ctypes.c_double),
        ("one_star_eps", ctypes.c_double),
        ("strike", ctypes.c_double),
        ("lower_strike", ctypes.c_double),
        ("ki_barrier", ctypes.c_double),
        ("option_gearing", ctypes.c_double),
        ("autocall_coupon", ctypes.c_double),
        ("c_mem_0", ctypes.c_double),
        ("i_ki_past", ctypes.c_double),
        ("part_call_strike", ctypes.c_double),
        ("part_call_rate", ctypes.c_double),
        ("one_star_coupon", ctypes.c_double),
        ("call_code", ctypes.c_int32),
        ("cpn_code", ctypes.c_int32),
        ("opt_code", ctypes.c_int32),
        ("has_ki", ctypes.c_int32),
        ("ki_active", ctypes.c_int32),
        ("has_one_star", ctypes.c_int32),
        ("is_memory", ctypes.c_int32),
        ("is_long_basket", ctypes.c_int32),
        ("no_notional", ctypes.c_int32),
        ("force_cpn_ac", ctypes.c_int32),
        ("cpn_dir_down", ctypes.c_int32),
        ("cpn_type_out", ctypes.c_int32),
        ("has_lower_strike", ctypes.c_int32),
        ("has_part_call", ctypes.c_int32),
    ]


if (
    ctypes.sizeof(_PlatenSdeAadArgsC) != 272
    or ctypes.sizeof(_SobolGenArgsC) != 128
    or ctypes.sizeof(_VibratoGammaArgsC) != 392
):
    raise AssertionError("C++23 ABI v2 ctypes struct size mismatch")


_CPP_CRN_V2_FNS: Optional[Tuple[Any, Any, Any]] = None
_CPP_STEP_SPOT_BUMPS_CPU_FN: Any = None
_CPP_STEP_SPOT_BUMPS_CHECKED = False
_CPP_CRN_V2_INIT_LOCK = threading.Lock()


def _get_cpp_step_spot_bumps_fn() -> Any:
    """Returns optional C++23 batched spot-bump trajectory stepper (`crn_step_spot_bumps_cpu`)."""
    global _CPP_STEP_SPOT_BUMPS_CPU_FN, _CPP_STEP_SPOT_BUMPS_CHECKED
    if _CPP_STEP_SPOT_BUMPS_CHECKED:
        return _CPP_STEP_SPOT_BUMPS_CPU_FN
    with _CPP_CRN_V2_INIT_LOCK:
        if _CPP_STEP_SPOT_BUMPS_CHECKED:
            return _CPP_STEP_SPOT_BUMPS_CPU_FN
        try:
            from deepvol.models.autocall._tier1_backend import _require_cpp23_backend

            backend = _require_cpp23_backend()
            _CPP_STEP_SPOT_BUMPS_CPU_FN = getattr(backend, "crn_step_spot_bumps_cpu", None)
        except (OSError, ImportError, AttributeError, RuntimeError):
            _CPP_STEP_SPOT_BUMPS_CPU_FN = None
        _CPP_STEP_SPOT_BUMPS_CHECKED = True
        return _CPP_STEP_SPOT_BUMPS_CPU_FN


def _get_cpp_crn_v2_fns() -> Tuple[Any, Any, Any]:
    """Loads C++23 ABI v2 CRN functions (`platen_sde_aad`, `generate_sobol_normals`, `eval_vibrato_gamma`)."""
    global _CPP_CRN_V2_FNS
    if _CPP_CRN_V2_FNS is not None:
        return _CPP_CRN_V2_FNS
    with _CPP_CRN_V2_INIT_LOCK:
        if _CPP_CRN_V2_FNS is not None:
            return _CPP_CRN_V2_FNS
        sde_fn: Any = None
        sob_fn: Any = None
        vib_fn: Any = None
        try:
            from deepvol.models.autocall._tier1_backend import _locate_cpp23_shared_lib

            lib_path = _locate_cpp23_shared_lib()
            if lib_path is not None:
                lib = ctypes.CDLL(str(lib_path))
                raw_sde = getattr(lib, "crn_platen_sde_aad_cpp", None)
                if raw_sde is not None:
                    raw_sde.argtypes = [ctypes.POINTER(_PlatenSdeAadArgsC)]
                    raw_sde.restype = ctypes.c_int32
                    sde_fn = raw_sde
                raw_sob = getattr(lib, "crn_generate_sobol_normals_cpp", None)
                if raw_sob is not None:
                    raw_sob.argtypes = [ctypes.POINTER(_SobolGenArgsC)]
                    raw_sob.restype = ctypes.c_int32
                    sob_fn = raw_sob
                raw_vib = getattr(lib, "crn_eval_vibrato_gamma_cpp", None)
                if raw_vib is not None:
                    raw_vib.argtypes = [ctypes.POINTER(_VibratoGammaArgsC)]
                    raw_vib.restype = ctypes.c_int32
                    vib_fn = raw_vib
        except (OSError, ImportError, AttributeError):
            sde_fn, sob_fn, vib_fn = None, None, None
        _CPP_CRN_V2_FNS = (sde_fn, sob_fn, vib_fn)
        return _CPP_CRN_V2_FNS


def _arr_bytes_equal(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> bool:
    """Fast exact byte comparison for contiguous NumPy arrays without allocating temporary bytes objects."""
    if a.shape != b.shape or a.dtype != b.dtype or not (a.flags.c_contiguous and b.flags.c_contiguous):
        return False
    return bool(memoryview(cast(Any, a.ravel().view(np.uint8))) == memoryview(cast(Any, b.ravel().view(np.uint8))))


def _np_f64(t: torch.Tensor) -> npt.NDArray[np.float64]:
    """Zero-copy NumPy float64 view for CPU float64 tensors, copying only across device or dtype."""
    if t.device.type == "cpu" and t.dtype == torch.float64:
        return np.asarray(t.detach().numpy(), dtype=np.float64)
    return np.asarray(t.detach().cpu().numpy(), dtype=np.float64)


_TRITON_PLAN_FAST: Dict[int, Tuple[Any, Tuple[Tuple[str, bool, Any], ...], int, bool, Tuple[Tuple[str, Any], ...]]] = {}
_TRITON_COMPILED_FAST: Dict[Any, Tuple[Any, Any, Any]] = {}
_MISSING_ARG = object()


def _fast_triton_launch(jit_fn: Any, grid: Tuple[int, ...], *pos_args: Any, **kw_args: Any) -> None:
    """Bypass Triton JIT binder/specializer overhead on warm hits while preserving exact specialization."""
    knobs = _triton_knobs
    drv = _triton_driver
    if (
        knobs is None
        or drv is None
        or knobs.runtime.launch_enter_hook is not None
        or knobs.runtime.launch_exit_hook is not None
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
        if len(_TRITON_PLAN_FAST) >= 128:
            _TRITON_PLAN_FAST.clear()
        _TRITON_PLAN_FAST[jid] = (jit_fn, param_plan, n_lead_pos, ce_only_tail, tail_ce)

    active_drv = drv.active
    dev_idx = active_drv.get_current_device()
    kw_get = kw_args.get
    missing = _MISSING_ARG
    tensor_cls = torch.Tensor
    n_pos = len(pos_args)
    sig_parts: List[Any] = [jid, dev_idx]
    sig_append = sig_parts.append

    if ce_only_tail and n_pos == n_lead_pos:
        for v in pos_args:
            tv = type(v)
            if tv is tensor_cls:
                sig_append(v.dtype)
                sig_append((v.data_ptr() & 15) == 0)
            elif tv is int:
                sig_append(1 if v == 1 else (2 if (v & 15) == 0 else 0))
                sig_append(-2147483648 <= v <= 2147483647)
            elif v is None:
                sig_append(None)
            elif tv is float or tv is bool:
                sig_append(tv)
            elif hasattr(v, "data_ptr"):
                sig_append(v.dtype)
                sig_append((int(v.data_ptr()) & 15) == 0)
            else:
                sig_append(tv)
        ce_vals: List[Any] = []
        ce_append = ce_vals.append
        for name, default in tail_ce:
            v = kw_get(name, default)
            ce_append(v)
            sig_append(v)
        nw_v = kw_get("num_warps", missing)
        if nw_v is not missing:
            sig_append(("nw", nw_v))
        ns_v = kw_get("num_stages", missing)
        if ns_v is not missing:
            sig_append(("ns", ns_v))
        nc_v = kw_get("num_ctas", missing)
        if nc_v is not missing:
            sig_append(("nc", nc_v))
        k_key = tuple(sig_parts)
        c_hit = _TRITON_COMPILED_FAST.get(k_key)
        if c_hit is None:
            compiled = jit_fn[grid](*pos_args, **kw_args)
            compiled._init_handles()
            if len(_TRITON_COMPILED_FAST) >= 512:
                _TRITON_COMPILED_FAST.clear()
            _TRITON_COMPILED_FAST[k_key] = (compiled.run, compiled.function, compiled.packed_metadata)
            return
        run_fn, cu_fn, packed_meta = c_hit
        g0 = grid[0]
        g1 = grid[1] if len(grid) > 1 else 1
        g2 = grid[2] if len(grid) > 2 else 1
        stream = active_drv.get_current_stream(dev_idx)
        run_fn(g0, g1, g2, stream, cu_fn, packed_meta, None, None, None, *pos_args, *ce_vals)
        return

    run_args: List[Any] = []
    run_append = run_args.append
    pos_i = 0
    for name, is_ce, default in param_plan:
        v = kw_get(name, missing)
        if v is missing:
            if (not is_ce) and pos_i < n_pos:
                v = pos_args[pos_i]
                pos_i += 1
            else:
                v = default
        run_append(v)
        if is_ce:
            sig_append(v)
        else:
            tv = type(v)
            if tv is tensor_cls:
                sig_append(v.dtype)
                sig_append((v.data_ptr() & 15) == 0)
            elif tv is int:
                sig_append(1 if v == 1 else (2 if (v & 15) == 0 else 0))
                sig_append(-2147483648 <= v <= 2147483647)
            elif v is None:
                sig_append(None)
            elif tv is float or tv is bool:
                sig_append(tv)
            elif hasattr(v, "data_ptr"):
                sig_append(v.dtype)
                sig_append((int(v.data_ptr()) & 15) == 0)
            else:
                sig_append(tv)

    if pos_i != n_pos:
        jit_fn[grid](*pos_args, **kw_args)
        return

    nw_v = kw_get("num_warps", missing)
    if nw_v is not missing:
        sig_append(("nw", nw_v))
    ns_v = kw_get("num_stages", missing)
    if ns_v is not missing:
        sig_append(("ns", ns_v))
    nc_v = kw_get("num_ctas", missing)
    if nc_v is not missing:
        sig_append(("nc", nc_v))
    k_key = tuple(sig_parts)

    c_hit = _TRITON_COMPILED_FAST.get(k_key)
    if c_hit is None:
        compiled = jit_fn[grid](*pos_args, **kw_args)
        compiled._init_handles()
        if len(_TRITON_COMPILED_FAST) >= 512:
            _TRITON_COMPILED_FAST.clear()
        _TRITON_COMPILED_FAST[k_key] = (compiled.run, compiled.function, compiled.packed_metadata)
        return

    run_fn, cu_fn, packed_meta = c_hit
    g0 = grid[0]
    g1 = grid[1] if len(grid) > 1 else 1
    g2 = grid[2] if len(grid) > 2 else 1
    stream = active_drv.get_current_stream(dev_idx)
    run_fn(
        g0,
        g1,
        g2,
        stream,
        cu_fn,
        packed_meta,
        None,
        None,
        None,
        *run_args,
    )


def _content_digest(*items: Any) -> str:
    """Computes collision-resistant 128-bit blake2b digest across array payloads and scalars."""
    h = hashlib.blake2b(digest_size=16)
    for item in items:
        if item is None:
            h.update(b"<None>")
        elif isinstance(item, np.ndarray):
            arr = np.ascontiguousarray(item)
            h.update(repr(arr.shape).encode("ascii"))
            h.update(arr.dtype.str.encode("ascii"))
            h.update(memoryview(cast(Any, arr.view(np.uint8))))
        elif isinstance(item, torch.Tensor):
            arr = np.ascontiguousarray(item.detach().cpu().numpy())
            h.update(repr(arr.shape).encode("ascii"))
            h.update(arr.dtype.str.encode("ascii"))
            h.update(memoryview(cast(Any, arr.view(np.uint8))))
        else:
            h.update(repr(item).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def _to_gpu_tensor(arr: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Converts array-like input to contiguous writable tensor on target device without read-only warnings."""
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype=dtype, device=device).contiguous()
    np_arr = np.asarray(arr)
    if not np_arr.flags.writeable or not np_arr.flags.c_contiguous:
        np_arr = np.array(np_arr, copy=True, order="C")
    t = torch.as_tensor(np_arr, dtype=dtype)
    return t.to(device=device) if t.device != device else t


def _surface_content_digest(surf: Any) -> str:
    """Returns immutable blake2b digest for a RegularizedVolSurface without relying on id() or pointers."""
    q_raw = surf.quotes
    q_cur = q_raw if (isinstance(q_raw, np.ndarray) and q_raw.dtype == np.float64) else np.asarray(q_raw, dtype=np.float64)
    k_raw = surf.strikes
    k_cur = k_raw if (isinstance(k_raw, np.ndarray) and k_raw.dtype == np.float64) else np.asarray(k_raw, dtype=np.float64)
    t_raw = surf.tenors
    t_cur = t_raw if (isinstance(t_raw, np.ndarray) and t_raw.dtype == np.float64) else np.asarray(t_raw, dtype=np.float64)
    t365_raw = getattr(surf, "tenors_act365", t_cur)
    t365_cur = (
        t365_raw
        if (isinstance(t365_raw, np.ndarray) and t365_raw.dtype == np.float64)
        else np.asarray(t365_raw, dtype=np.float64)
    )
    cal_d = str(getattr(surf, "calendar_proj_digest", ""))
    meta_t = (
        cal_d,
        str(getattr(surf, "identifier", getattr(surf, "Identifier", ""))),
        str(getattr(surf, "dcc", "")),
        str(getattr(surf, "smile_interp", "")),
        str(getattr(surf, "smile_extrap", "")),
        str(getattr(surf, "bc_type", "")),
        float(surf.min_vol),
        float(surf.max_local_vol),
    )
    snap = getattr(surf, "_crn_surf_sig_snap", None)
    if snap is not None:
        s_dig, q_snap, k_snap, t_snap, t365_snap, m_snap = snap
        if (
            m_snap == meta_t
            and q_cur.shape == q_snap.shape
            and k_cur.shape == k_snap.shape
            and t_cur.shape == t_snap.shape
            and t365_cur.shape == t365_snap.shape
            and _arr_bytes_equal(q_cur, q_snap)
            and _arr_bytes_equal(k_cur, k_snap)
            and _arr_bytes_equal(t_cur, t_snap)
            and _arr_bytes_equal(t365_cur, t365_snap)
        ):
            return cast(str, s_dig)
    dig = _content_digest(k_cur, t_cur, t365_cur, q_cur, *meta_t)
    try:
        setattr(
            surf,
            "_crn_surf_sig_snap",
            (dig, q_cur.copy(), k_cur.copy(), t_cur.copy(), t365_cur.copy(), meta_t),
        )
    except (AttributeError, TypeError):
        return dig
    return dig


@dataclass(frozen=True)
class QmcAadGreeksConfig:
    """Configuration for Tier-4 GPU/CPU QMC Pathwise-AAD and Top-3 LOO/L2O CRN Greeks."""

    n_paths: int = 8192
    n_replicates: int = 8
    n_substeps: int = 4
    lv_grid_nt: int = 32
    lv_grid_ny: int = 64
    h_spot_rel: float = 0.01
    h_corr: float = 0.01
    h_rho: float = 1.0e-4
    one_star_eps: float = 0.02
    seed: int = 42
    barrier_eps_min: float = 0.005
    barrier_eps_max: float = 0.02
    ki_monitoring_type: Optional[Literal["European", "Discrete", "Daily", "Continuous"]] = None
    cv_mode: Literal["multifidelity_gbm", "none"] = "multifidelity_gbm"
    n_cheap_mf: int = 65536
    corr_aad_mode: Literal["cholesky", "sym_sqrt"] = "cholesky"


class _CrnLooL2oBatchArgsC(ctypes.Structure):
    """Exact 320B C ABI descriptor matching CrnLooL2oBatchArgs in tier1_autocall_cpu.cpp."""

    _fields_ = [
        ("n_bumps", ctypes.c_int64),
        ("d_assets", ctypes.c_int64),
        ("m_rem", ctypes.c_int64),
        ("n_paths_total", ctypes.c_int64),
        ("perf_base", ctypes.c_void_p),
        ("bump_perf_a", ctypes.c_void_p),
        ("bump_perf_b", ctypes.c_void_p),
        ("bump_asset_a", ctypes.c_void_p),
        ("bump_asset_b", ctypes.c_void_p),
        ("df_pay_mat", ctypes.c_void_p),
        ("w_fund_mat", ctypes.c_void_p),
        ("pv_in_transit", ctypes.c_void_p),
        ("pv_out", ctypes.c_void_p),
        ("ko_barriers", ctypes.c_void_p),
        ("is_callable", ctypes.c_void_p),
        ("cpn_barriers", ctypes.c_void_p),
        ("is_guaranteed_cpn", ctypes.c_void_p),
        ("coupon_rates", ctypes.c_void_p),
        ("ko_spread", ctypes.c_double),
        ("cpn_spread", ctypes.c_double),
        ("ki_spread", ctypes.c_double),
        ("one_star_strike", ctypes.c_double),
        ("one_star_eps", ctypes.c_double),
        ("strike", ctypes.c_double),
        ("lower_strike", ctypes.c_double),
        ("ki_barrier", ctypes.c_double),
        ("option_gearing", ctypes.c_double),
        ("autocall_coupon", ctypes.c_double),
        ("c_mem_0", ctypes.c_double),
        ("i_ki_past", ctypes.c_double),
        ("part_call_strike", ctypes.c_double),
        ("part_call_rate", ctypes.c_double),
        ("one_star_coupon", ctypes.c_double),
        ("call_code", ctypes.c_int32),
        ("cpn_code", ctypes.c_int32),
        ("opt_code", ctypes.c_int32),
        ("has_ki", ctypes.c_int32),
        ("is_memory", ctypes.c_int32),
        ("is_long_basket", ctypes.c_int32),
        ("no_notional", ctypes.c_int32),
        ("force_cpn_ac", ctypes.c_int32),
        ("cpn_dir_down", ctypes.c_int32),
        ("cpn_type_out", ctypes.c_int32),
        ("_reserved", ctypes.c_int32 * 4),
    ]


_CPP_CRN_LOCK = threading.Lock()
_CPP_CRN_LOADED = False
_CPP_STEP_1D_FN: Any = None
_CPP_LOO_L2O_FN: Any = None


def _get_cpp_crn_fns() -> Tuple[Any, Any]:
    """Lazily loads C++23 crn_step_1d_trajectories_cpp and crn_eval_payoff_loo_l2o_cpp."""
    global _CPP_CRN_LOADED, _CPP_STEP_1D_FN, _CPP_LOO_L2O_FN
    if _CPP_CRN_LOADED:
        return _CPP_STEP_1D_FN, _CPP_LOO_L2O_FN
    with _CPP_CRN_LOCK:
        if _CPP_CRN_LOADED:
            return _CPP_STEP_1D_FN, _CPP_LOO_L2O_FN
        try:
            from deepvol.models.autocall._tier1_backend import _locate_cpp23_shared_lib

            lib_path = _locate_cpp23_shared_lib()
            if lib_path is not None:
                lib = ctypes.CDLL(str(lib_path))
                step_fn = getattr(lib, "crn_step_1d_trajectories_cpp", None)
                loo_fn = getattr(lib, "crn_eval_payoff_loo_l2o_cpp", None)
                if step_fn is not None and loo_fn is not None:
                    step_fn.argtypes = [
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                    ]
                    step_fn.restype = None
                    loo_fn.argtypes = [ctypes.POINTER(_CrnLooL2oBatchArgsC)]
                    loo_fn.restype = None
                    _CPP_STEP_1D_FN = step_fn
                    _CPP_LOO_L2O_FN = loo_fn
        except Exception as exc:
            logger.warning("Failed to load C++ CRN kernel library; falling back to PyTorch/Python: %s", exc)
            _CPP_STEP_1D_FN = None
            _CPP_LOO_L2O_FN = None
        _CPP_CRN_LOADED = True
        return _CPP_STEP_1D_FN, _CPP_LOO_L2O_FN


def _next_power_of_two_int(n: int) -> int:
    v = 1
    while v < n:
        v <<= 1
    return v


def _next_pow2_clamped(n: int, lo: int = 1, hi: int = 32) -> int:
    v = max(lo, _next_power_of_two_int(n))
    return min(v, hi)


def cholesky_corr_adjoint_np(
    chol_lower: npt.NDArray[np.float64],
    bar_chol: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """O(D^3) lower-Cholesky adjoint C_tilde = L^{-T} phi(L^T bar_L) L^{-1}."""
    d = chol_lower.shape[0]
    if d <= 1:
        return np.zeros((d, d), dtype=np.float64)
    l_tril = np.tril(chol_lower)
    g_tril = np.tril(bar_chol)
    m_mat = l_tril.T @ g_tril
    phi_m = np.tril(m_mat, k=-1) + 0.5 * np.diag(np.diag(m_mat))
    l_inv = np.linalg.inv(l_tril)
    return np.asarray(l_inv.T @ phi_m @ l_inv, dtype=np.float64)


def sym_sqrt_corr_adjoint_np(
    v_eig: npt.NDArray[np.float64],
    sqrt_ev: npt.NDArray[np.float64],
    bar_q_sym: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Frechet adjoint of Sigma^{1/2} = V diag(sqrt_ev) V^T via Sylvester eigen-division."""
    d = v_eig.shape[0]
    if d <= 1:
        return np.zeros((d, d), dtype=np.float64)
    b_sym = 0.5 * (bar_q_sym + bar_q_sym.T)
    denom_raw = sqrt_ev[:, None] + sqrt_ev[None, :]
    inv_denom = denom_raw / (denom_raw * denom_raw + 1e-24)
    m_eig = (v_eig.T @ b_sym @ v_eig) * inv_denom
    return np.asarray(v_eig @ m_eig @ v_eig.T, dtype=np.float64)


def _build_bb_orthogonal_np(tau_obs_np: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Constructs exact orthogonal (M, M) dyadic Brownian-bridge time basis."""
    m_rem = int(len(tau_obs_np))
    if m_rem <= 1:
        return np.ones((max(m_rem, 1), max(m_rem, 1)), dtype=np.float64)
    dt_arr = np.maximum(np.diff(np.concatenate([[0.0], tau_obs_np.astype(np.float64)])), 1e-12)
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
        c_path[mid, :] = ((tr - tm) / denom) * c_path[l_idx, :] + ((tm - tl) / denom) * c_path[r_idx, :]
        c_path[mid, col_idx] = math.sqrt(max((tm - tl) * (tr - tm) / denom, 0.0))
        col_idx += 1
        queue.append((l_idx, mid))
        queue.append((mid, r_idx))
    inv_sqrt_dt = 1.0 / np.sqrt(dt_arr)[:, None]
    return np.ascontiguousarray((c_path[1:, :] - c_path[:-1, :]) * inv_sqrt_dt, dtype=np.float64)


def _sym_sqrt_and_kl_pca_np(
    tau_obs_np: npt.NDArray[np.float64],
    corr_np: npt.NDArray[np.float64],
    corr_aad_mode: str = "cholesky",
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Builds dyadic Brownian-bridge time basis U_time, correlation factor (Cholesky or Q_sym), and KL-PCA eigenvectors."""
    key = ("bb_corr_fac", str(corr_aad_mode), tau_obs_np.tobytes(), corr_np.tobytes())
    cached = _CRN_VEGA_JAC_CACHE.get(key)
    if cached is not None:
        return cast(
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
            cached,
        )
    u_time = _build_bb_orthogonal_np(tau_obs_np)

    d = int(corr_np.shape[0])
    if d > 1:
        corr_sym = 0.5 * (corr_np + corr_np.T)
        ev_raw, v_raw = np.linalg.eigh(corr_sym)
        idx_desc = np.argsort(ev_raw)[::-1]
        ev = np.maximum(ev_raw[idx_desc], 1e-12)
        v_eig = np.ascontiguousarray(v_raw[:, idx_desc], dtype=np.float64)
        sqrt_ev = np.ascontiguousarray(np.sqrt(ev), dtype=np.float64)
        if corr_aad_mode == "cholesky":
            chol_factor: Optional[npt.NDArray[np.float64]] = None
            for jitter in (0.0, 1e-12, 1e-9, 1e-6):
                try:
                    c_try = corr_sym if jitter == 0.0 else (corr_sym + jitter * np.eye(d, dtype=np.float64))
                    chol_factor = np.ascontiguousarray(np.linalg.cholesky(c_try), dtype=np.float64)
                    break
                except np.linalg.LinAlgError:
                    continue
            if chol_factor is None:
                c_psd = (v_eig * ev[None, :]) @ v_eig.T
                c_psd = 0.5 * (c_psd + c_psd.T) + 1e-10 * np.eye(d, dtype=np.float64)
                chol_factor = np.ascontiguousarray(np.linalg.cholesky(c_psd), dtype=np.float64)
        else:
            chol_factor = np.ascontiguousarray((v_eig * sqrt_ev[None, :]) @ v_eig.T, dtype=np.float64)
    else:
        v_eig = np.ones((1, 1), dtype=np.float64)
        sqrt_ev = np.ones((1,), dtype=np.float64)
        chol_factor = np.ones((1, 1), dtype=np.float64)
    out = (u_time, chol_factor, v_eig, sqrt_ev)
    _CRN_VEGA_JAC_CACHE.put(key, out)
    return out


if _HAS_TRITON:

    @triton.jit  # type: ignore[untyped-decorator]
    def _sobol_bridge_normals_kernel(  # type: ignore[no-untyped-def]
        sv_ptr,
        shifts_ptr,
        out_ptr,
        m_rem,
        d_assets,
        cg_flags_ptr=None,
        n_rep: tl.constexpr = 1024,
        n_sub: tl.constexpr = 4,
        m_pow: tl.constexpr = 10,
        BLOCK_N: tl.constexpr = 256,
        USE_RT_FLAGS: tl.constexpr = False,
        D_PAD: tl.constexpr = 0,
    ):
        pid_n = tl.program_id(0)
        pid_mr = tl.program_id(1)
        n_replicates = tl.num_programs(1) // m_rem
        pid_r = pid_mr // m_rem
        m_idx = pid_mr % m_rem

        if USE_RT_FLAGS:
            d_act = tl.load(cg_flags_ptr + 0)
            m_act = tl.load(cg_flags_ptr + 1)
            if m_idx >= m_act:
                return
        else:
            d_act = d_assets
            m_act = m_rem

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep
        idx = offs_n.to(tl.uint32)
        gray = idx ^ (idx >> 1)

        inv_sqrt_sub = 1.0 / tl.sqrt(float(n_sub))
        n_dims_total = m_rem * (n_sub + 1) * d_assets
        coarse_base = m_idx * d_act
        fine_base = (m_act + m_idx * n_sub) * d_act
        n_total = n_replicates * n_rep

        n_d_loop: tl.constexpr = D_PAD if USE_RT_FLAGS else d_assets
        for d in range(n_d_loop):
            d_ok = (d < d_act) if USE_RT_FLAGS else True
            if d_ok:
                c_dim = coarse_base + d
                c_acc = tl.zeros((BLOCK_N,), dtype=tl.uint32)
                for b in range(m_pow):
                    bit_on = (gray >> b) & 1
                    dir_num = tl.load(sv_ptr + c_dim * 30 + b).to(tl.uint32)
                    c_acc = c_acc ^ tl.where(bit_on != 0, dir_num, tl.zeros((BLOCK_N,), dtype=tl.uint32))
                c_shift = tl.load(shifts_ptr + pid_r * n_dims_total + c_dim).to(tl.uint32)
                c_acc = c_acc ^ c_shift
                cu = (c_acc.to(tl.float32) + 0.5) * (1.0 / 1073741824.0)
                cu = tl.minimum(tl.maximum(cu, 1e-6), 1.0 - 1e-6)
                zc = tl.extra.cuda.libdevice.normcdfinv(cu)

                if n_sub == 1:
                    out_off = (m_idx * d_assets + d) * n_total + pid_r * n_rep + offs_n
                    tl.store(out_ptr + out_off, zc, mask=mask_n)
                else:
                    z_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
                    for s in range(n_sub):
                        f_dim = fine_base + s * d_act + d
                        f_acc = tl.zeros((BLOCK_N,), dtype=tl.uint32)
                        for b in range(m_pow):
                            bit_on = (gray >> b) & 1
                            dir_num = tl.load(sv_ptr + f_dim * 30 + b).to(tl.uint32)
                            f_acc = f_acc ^ tl.where(bit_on != 0, dir_num, tl.zeros((BLOCK_N,), dtype=tl.uint32))
                        f_shift = tl.load(shifts_ptr + pid_r * n_dims_total + f_dim).to(tl.uint32)
                        f_acc = f_acc ^ f_shift
                        fu = (f_acc.to(tl.float32) + 0.5) * (1.0 / 1073741824.0)
                        fu = tl.minimum(tl.maximum(fu, 1e-6), 1.0 - 1e-6)
                        zs = tl.extra.cuda.libdevice.normcdfinv(fu)
                        z_sum = z_sum + zs
                        step_idx = m_idx * n_sub + s
                        out_off = (step_idx * d_assets + d) * n_total + pid_r * n_rep + offs_n
                        tl.store(out_ptr + out_off, zs, mask=mask_n)
                    z_mean = z_sum * (1.0 / float(n_sub))
                    delta_z = zc * inv_sqrt_sub - z_mean
                    for s in range(n_sub):
                        step_idx = m_idx * n_sub + s
                        out_off = (step_idx * d_assets + d) * n_total + pid_r * n_rep + offs_n
                        cur = tl.load(out_ptr + out_off, mask=mask_n, other=0.0)
                        tl.store(out_ptr + out_off, cur + delta_z, mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _triton_lv_bilinear_f32(  # type: ignore[no-untyped-def]
        lv_table_ptr,
        lv_base,
        t_val,
        y_val,
        inv_dt_grid,
        y_min_d,
        inv_dy_d,
        max_t_f,
        max_y_f,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        mask_n,
    ):
        u_t = tl.minimum(tl.maximum(t_val * inv_dt_grid, 0.0), max_t_f)
        it0 = u_t.to(tl.int32)
        wt = u_t - it0.to(tl.float32)
        it1 = tl.minimum(it0 + 1, N_t - 1)

        u_y_raw = (y_val - y_min_d) * inv_dy_d
        in_y = (u_y_raw > 0.0) & (u_y_raw < max_y_f)
        u_y = tl.minimum(tl.maximum(u_y_raw, 0.0), max_y_f)
        iy0 = u_y.to(tl.int32)
        wy = u_y - iy0.to(tl.float32)
        iy1 = tl.minimum(iy0 + 1, N_y - 1)

        s00 = tl.load(lv_table_ptr + lv_base + it0 * N_y + iy0, mask=mask_n, other=0.0)
        s01 = tl.load(lv_table_ptr + lv_base + it0 * N_y + iy1, mask=mask_n, other=0.0)
        s10 = tl.load(lv_table_ptr + lv_base + it1 * N_y + iy0, mask=mask_n, other=0.0)
        s11 = tl.load(lv_table_ptr + lv_base + it1 * N_y + iy1, mask=mask_n, other=0.0)
        v0 = (1.0 - wt) * s00 + wt * s10
        v1 = (1.0 - wt) * s01 + wt * s11
        sig = (1.0 - wy) * v0 + wy * v1
        dsig_dy_raw = tl.where(in_y, (v1 - v0) * inv_dy_d, 0.0)
        dsig_dy = tl.minimum(tl.maximum(dsig_dy_raw, -4.0), 4.0)
        return sig, dsig_dy, it0, it1, wt, iy0, iy1, wy

    @triton.jit  # type: ignore[untyped-decorator]
    def _triton_platen_fwd_substep_f32(  # type: ignore[no-untyped-def]
        lv_table_ptr,
        lv_base,
        lx,
        t_start,
        dt,
        sqrt_dt,
        q_s,
        zc,
        inv_dt_grid,
        y_min_d,
        inv_dy_d,
        max_t_f,
        max_y_f,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        mask_n,
    ):
        sig0, d0, _, _, _, _, _, _ = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_start, lx + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )
        t_end = t_start + dt
        dw = sqrt_dt * zc
        a0 = -0.5 * sig0 * sig0
        lx_hat = lx + a0 * dt + sig0 * dw
        lx_p = lx + a0 * dt + sig0 * sqrt_dt
        lx_m = lx + a0 * dt - sig0 * sqrt_dt

        sigh, dh, _, _, _, _, _, _ = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_end, lx_hat + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )
        sigp, dp, _, _, _, _, _, _ = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_end, lx_p + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )
        sigm, dm, _, _, _, _, _, _ = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_end, lx_m + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )
        ah = -0.5 * sigh * sigh
        zc2m1 = zc * zc - 1.0
        lx_next = (
            lx
            + 0.5 * (a0 + ah) * dt
            + 0.25 * (sigp + sigm + 2.0 * sig0) * dw
            + 0.25 * (sigp - sigm) * sqrt_dt * zc2m1
        )
        d_a0_dx = -sig0 * d0
        d_hat_dx = 1.0 + d_a0_dx * dt + d0 * dw
        d_p_dx = 1.0 + d_a0_dx * dt + d0 * sqrt_dt
        d_m_dx = 1.0 + d_a0_dx * dt - d0 * sqrt_dt
        step_j_raw = (
            1.0
            + 0.5 * (d_a0_dx - sigh * dh * d_hat_dx) * dt
            + 0.25 * (dp * d_p_dx + dm * d_m_dx + 2.0 * d0) * dw
            + 0.25 * (dp * d_p_dx - dm * d_m_dx) * sqrt_dt * zc2m1
        )
        step_j = tl.minimum(tl.maximum(step_j_raw, 0.25), 2.5)
        return lx_next, step_j

    @triton.jit  # type: ignore[untyped-decorator]
    def _triton_lv_cta_scatter_row_f32(  # type: ignore[no-untyped-def]
        bar_lv_d_ptr,
        it0,
        it1,
        wt,
        iy0,
        iy1,
        wy,
        bar_val,
        N_y: tl.constexpr,
        mask_act,
        BLOCK_Y: tl.constexpr = 16,
    ):
        if tl.max(tl.where(mask_act, 1, 0), axis=0) == 0:
            return
        iy_lo = tl.min(tl.where(mask_act, iy0, N_y), axis=0)
        iy_hi = tl.max(tl.where(mask_act, iy1, -1), axis=0)
        w0 = tl.where(mask_act, bar_val * (1.0 - wy), 0.0)
        w1 = tl.where(mask_act, bar_val * wy, 0.0)
        for y_base in range(0, N_y, BLOCK_Y):
            if (y_base + BLOCK_Y > iy_lo) & (y_base <= iy_hi):
                ys = y_base + tl.arange(0, BLOCK_Y)
                m0 = iy0[:, None] == ys[None, :]
                m1 = iy1[:, None] == ys[None, :]
                acc_y = tl.sum(tl.where(m0, w0[:, None], 0.0) + tl.where(m1, w1[:, None], 0.0), axis=0)
                ymask = (ys < N_y) & (tl.abs(acc_y) > 1e-24)
                tl.atomic_add(bar_lv_d_ptr + it0 * N_y + ys, acc_y * (1.0 - wt), mask=ymask, sem="relaxed")
                tl.atomic_add(bar_lv_d_ptr + it1 * N_y + ys, acc_y * wt, mask=ymask, sem="relaxed")

    @triton.jit  # type: ignore[untyped-decorator]
    def _triton_lv_cta_scatter_row3_f32(  # type: ignore[no-untyped-def]
        bar_lv_d_ptr,
        it0,
        it1,
        wt,
        iy0_a,
        iy1_a,
        wy_a,
        bar_a,
        iy0_b,
        iy1_b,
        wy_b,
        bar_b,
        iy0_c,
        iy1_c,
        wy_c,
        bar_c,
        N_y: tl.constexpr,
        mask_act,
        BLOCK_Y: tl.constexpr = 16,
    ):
        if tl.max(tl.where(mask_act, 1, 0), axis=0) == 0:
            return
        iy_lo = tl.min(
            tl.where(mask_act, tl.minimum(iy0_a, tl.minimum(iy0_b, iy0_c)), N_y),
            axis=0,
        )
        iy_hi = tl.max(
            tl.where(mask_act, tl.maximum(iy1_a, tl.maximum(iy1_b, iy1_c)), -1),
            axis=0,
        )
        wa0 = tl.where(mask_act, bar_a * (1.0 - wy_a), 0.0)
        wa1 = tl.where(mask_act, bar_a * wy_a, 0.0)
        wb0 = tl.where(mask_act, bar_b * (1.0 - wy_b), 0.0)
        wb1 = tl.where(mask_act, bar_b * wy_b, 0.0)
        wc0 = tl.where(mask_act, bar_c * (1.0 - wy_c), 0.0)
        wc1 = tl.where(mask_act, bar_c * wy_c, 0.0)
        for y_base in range(0, N_y, BLOCK_Y):
            if (y_base + BLOCK_Y > iy_lo) & (y_base <= iy_hi):
                ys = y_base + tl.arange(0, BLOCK_Y)
                acc_y = tl.sum(
                    tl.where(iy0_a[:, None] == ys[None, :], wa0[:, None], 0.0)
                    + tl.where(iy1_a[:, None] == ys[None, :], wa1[:, None], 0.0)
                    + tl.where(iy0_b[:, None] == ys[None, :], wb0[:, None], 0.0)
                    + tl.where(iy1_b[:, None] == ys[None, :], wb1[:, None], 0.0)
                    + tl.where(iy0_c[:, None] == ys[None, :], wc0[:, None], 0.0)
                    + tl.where(iy1_c[:, None] == ys[None, :], wc1[:, None], 0.0),
                    axis=0,
                )
                ymask = (ys < N_y) & (tl.abs(acc_y) > 1e-24)
                tl.atomic_add(bar_lv_d_ptr + it0 * N_y + ys, acc_y * (1.0 - wt), mask=ymask, sem="relaxed")
                tl.atomic_add(bar_lv_d_ptr + it1 * N_y + ys, acc_y * wt, mask=ymask, sem="relaxed")

    @triton.jit  # type: ignore[untyped-decorator]
    def _triton_lv_direct_scatter_f32(  # type: ignore[no-untyped-def]
        bar_lv_d_ptr,
        it0,
        it1,
        wt,
        iy0,
        iy1,
        wy,
        bar_val,
        N_y: tl.constexpr,
        mask_act,
    ):
        m = mask_act & (tl.abs(bar_val) > 1e-24)
        w0t = 1.0 - wt
        row0 = it0 * N_y
        row1 = it1 * N_y
        tl.atomic_add(bar_lv_d_ptr + row0 + iy0, bar_val * (w0t * (1.0 - wy)), mask=m, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row0 + iy1, bar_val * (w0t * wy), mask=m, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row1 + iy0, bar_val * (wt * (1.0 - wy)), mask=m, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row1 + iy1, bar_val * (wt * wy), mask=m, sem="relaxed")

    @triton.jit  # type: ignore[untyped-decorator]
    def _triton_lv_direct_scatter3_f32(  # type: ignore[no-untyped-def]
        bar_lv_d_ptr,
        it0,
        it1,
        wt,
        iy0_a,
        iy1_a,
        wy_a,
        bar_a,
        iy0_b,
        iy1_b,
        wy_b,
        bar_b,
        iy0_c,
        iy1_c,
        wy_c,
        bar_c,
        N_y: tl.constexpr,
        mask_act,
    ):
        w0t = 1.0 - wt
        row0 = it0 * N_y
        row1 = it1 * N_y
        ma = mask_act & (tl.abs(bar_a) > 1e-24)
        tl.atomic_add(bar_lv_d_ptr + row0 + iy0_a, bar_a * (w0t * (1.0 - wy_a)), mask=ma, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row0 + iy1_a, bar_a * (w0t * wy_a), mask=ma, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row1 + iy0_a, bar_a * (wt * (1.0 - wy_a)), mask=ma, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row1 + iy1_a, bar_a * (wt * wy_a), mask=ma, sem="relaxed")
        mb = mask_act & (tl.abs(bar_b) > 1e-24)
        tl.atomic_add(bar_lv_d_ptr + row0 + iy0_b, bar_b * (w0t * (1.0 - wy_b)), mask=mb, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row0 + iy1_b, bar_b * (w0t * wy_b), mask=mb, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row1 + iy0_b, bar_b * (wt * (1.0 - wy_b)), mask=mb, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row1 + iy1_b, bar_b * (wt * wy_b), mask=mb, sem="relaxed")
        mc = mask_act & (tl.abs(bar_c) > 1e-24)
        tl.atomic_add(bar_lv_d_ptr + row0 + iy0_c, bar_c * (w0t * (1.0 - wy_c)), mask=mc, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row0 + iy1_c, bar_c * (w0t * wy_c), mask=mc, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row1 + iy0_c, bar_c * (wt * (1.0 - wy_c)), mask=mc, sem="relaxed")
        tl.atomic_add(bar_lv_d_ptr + row1 + iy1_c, bar_c * (wt * wy_c), mask=mc, sem="relaxed")

    @triton.jit  # type: ignore[untyped-decorator]
    def _qmc_fwd_checkpoint_kernel(  # type: ignore[no-untyped-def]
        sv_ptr,
        shifts_ptr,
        chol_ptr,
        lv_table_ptr,
        y_min_ptr,
        inv_dy_ptr,
        t_prev_ptr,
        dt_sub_ptr,
        sqrt_dt_ptr,
        q_shift_mid_ptr,
        r_nat_mid_ptr,
        tile_max_ptr,
        tile_ex_sum_ptr,
        lx_period_ptr,
        vib_lx_pre_ptr,
        vib_j_pre_ptr,
        vib_sig_pre_ptr,
        vib_j_state_ptr,
        fwd_perf_f32_ptr,
        perf_out_ptr,
        inv_dt_grid,
        d_assets,
        m_rem,
        n_tiles,
        vib_period_idx,
        vib_sub_idx,
        cg_flags_ptr=None,
        payoff_f64_ptr=None,
        n_rep: tl.constexpr = 1024,
        n_sub: tl.constexpr = 4,
        N_t: tl.constexpr = 32,
        N_y: tl.constexpr = 64,
        BLOCK_N: tl.constexpr = 256,
        period_idx: int = 0,
        FUSE_M: tl.constexpr = False,
        SAVE_VIBRATO: tl.constexpr = False,
        VIB_PERIOD_LEVEL: tl.constexpr = False,
        USE_Z_CORR: tl.constexpr = False,
        EMIT_PERF: tl.constexpr = False,
        USE_RT_FLAGS: tl.constexpr = False,
    ):
        _u_sh = shifts_ptr
        _u_rn = r_nat_mid_ptr
        pid_n = tl.program_id(0)
        pid_dr = tl.program_id(1)
        pid_r = pid_dr // d_assets
        d = pid_dr % d_assets
        if USE_RT_FLAGS:
            d_act = tl.load(cg_flags_ptr + 0)
            m_act = tl.load(cg_flags_ptr + 1)
            if d >= d_act:
                return
            eff_inv_dt = tl.load(payoff_f64_ptr + 17).to(tl.float32)
            eff_vp_idx = tl.load(cg_flags_ptr + 18)
            eff_vs_idx = tl.load(cg_flags_ptr + 19)
            eff_vp_lvl = tl.load(cg_flags_ptr + 22) != 0
        else:
            m_act = m_rem
            eff_inv_dt = inv_dt_grid
            eff_vp_idx = vib_period_idx
            eff_vs_idx = vib_sub_idx
            eff_vp_lvl = VIB_PERIOD_LEVEL
        n_replicates = tl.num_programs(1) // d_assets
        n_total = n_replicates * n_rep
        n_steps_tot = m_rem * n_sub

        offs_rep = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_rep < n_rep
        gid_n = pid_r * n_rep + offs_rep

        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4
        y_min_d = tl.load(y_min_ptr + d)
        inv_dy_d = tl.load(inv_dy_ptr + d)
        lv_base = d * (N_t * N_y)

        if FUSE_M:
            lx = tl.zeros((BLOCK_N,), dtype=tl.float32)
            j_lx = tl.full((BLOCK_N,), 1.0, dtype=tl.float32)
            for m_cur in range(m_rem):
                if m_cur < m_act:
                    if SAVE_VIBRATO:
                        if eff_vp_lvl & (m_cur == eff_vp_idx):
                            v_off = d * n_total + gid_n
                            tl.store(vib_lx_pre_ptr + v_off, lx, mask=mask_n)
                            tl.store(vib_j_pre_ptr + v_off, j_lx, mask=mask_n)
                            sig_sq_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                            for sv_k in tl.static_range(n_sub):
                                s_idx_k = m_cur * n_sub + sv_k
                                t_mid_k = tl.load(t_prev_ptr + s_idx_k)
                                q_k = tl.load(q_shift_mid_ptr + d * n_steps_tot + s_idx_k)
                                sig_k, _dk, _i0k, _i1k, _wtk, _j0k, _j1k, _wyk = _triton_lv_bilinear_f32(
                                    lv_table_ptr, lv_base, t_mid_k, lx + q_k, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
                                )
                                sig_sq_acc = sig_sq_acc + sig_k * sig_k
                            tl.store(vib_sig_pre_ptr + v_off, tl.sqrt(sig_sq_acc * (1.0 / float(n_sub))), mask=mask_n)

                    for s in tl.static_range(n_sub):
                        step_idx = m_cur * n_sub + s
                        t_mid = tl.load(t_prev_ptr + step_idx)
                        dt = tl.load(dt_sub_ptr + step_idx)
                        sqrt_dt = tl.load(sqrt_dt_ptr + step_idx)
                        t_start = tl.maximum(t_mid - 0.5 * dt, 0.0)
                        q_s = tl.load(q_shift_mid_ptr + d * n_steps_tot + step_idx)

                        if SAVE_VIBRATO:
                            if (~eff_vp_lvl) & (m_cur == eff_vp_idx) & (s == eff_vs_idx):
                                v_off = d * n_total + gid_n
                                tl.store(vib_lx_pre_ptr + v_off, lx, mask=mask_n)
                                tl.store(vib_j_pre_ptr + v_off, j_lx, mask=mask_n)
                                sig_vib, _dv, _i0v, _i1v, _wtv, _j0v, _j1v, _wyv = _triton_lv_bilinear_f32(
                                    lv_table_ptr, lv_base, t_mid, lx + q_s, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
                                )
                                tl.store(vib_sig_pre_ptr + v_off, sig_vib, mask=mask_n)

                        if USE_Z_CORR:
                            zc = tl.load(sv_ptr + (step_idx * d_assets + d) * n_total + gid_n, mask=mask_n, other=0.0)
                        else:
                            zc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                            for j in range(d_assets):
                                l_dj = tl.load(chol_ptr + d * d_assets + j)
                                zj = tl.load(sv_ptr + (step_idx * d_assets + j) * n_total + gid_n, mask=mask_n, other=0.0)
                                zc = zc + l_dj * zj

                        lx, step_j = _triton_platen_fwd_substep_f32(
                            lv_table_ptr, lv_base, lx, t_start, dt, sqrt_dt, q_s, zc,
                            eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                        )
                        if SAVE_VIBRATO:
                            j_lx = j_lx * step_j

                    max_lx = tl.max(tl.where(mask_n, lx, -1.0e30), axis=0)
                    ex_v = tl.where(mask_n, tl.exp(lx - max_lx), 0.0)
                    mean_rep = tl.maximum(tl.sum(ex_v, axis=0) / float(n_rep), 1.0e-30)
                    omega = ex_v / (float(n_rep) * mean_rep)
                    lx = lx - (max_lx + tl.log(mean_rep))
                    if SAVE_VIBRATO:
                        j_dev = tl.minimum(tl.maximum(j_lx, -6.0), 6.0) - 1.0
                        j_lx = tl.minimum(tl.maximum(1.0 + (j_dev - tl.sum(tl.where(mask_n, omega * j_dev, 0.0), axis=0)), -6.0), 6.0)
                    per_off = (m_cur * d_assets + d) * n_total + gid_n
                    tl.store(lx_period_ptr + per_off, lx, mask=mask_n)
                    if EMIT_PERF:
                        fp_dm = tl.load(fwd_perf_f32_ptr + d * m_rem + m_cur)
                        tl.store(perf_out_ptr + per_off, fp_dm * (ex_v / mean_rep), mask=mask_n)
        else:
            has_prev = period_idx > 0
            prev_period = tl.maximum(period_idx - 1, 0)
            prev_off = (prev_period * d_assets + d) * n_total + gid_n
            lx = tl.load(lx_period_ptr + prev_off, mask=mask_n & has_prev, other=0.0)
            if SAVE_VIBRATO:
                j_lx = tl.load(vib_j_state_ptr + d * n_total + gid_n, mask=mask_n & has_prev, other=1.0)
            else:
                j_lx = tl.full((BLOCK_N,), 1.0, dtype=tl.float32)

            if SAVE_VIBRATO:
                if VIB_PERIOD_LEVEL:
                    if period_idx == vib_period_idx:
                        v_off = d * n_total + gid_n
                        tl.store(vib_lx_pre_ptr + v_off, lx, mask=mask_n)
                        tl.store(vib_j_pre_ptr + v_off, j_lx, mask=mask_n)
                        sig_sq_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                        for sv_k in tl.static_range(n_sub):
                            s_idx_k = period_idx * n_sub + sv_k
                            t_mid_k = tl.load(t_prev_ptr + s_idx_k)
                            q_k = tl.load(q_shift_mid_ptr + d * n_steps_tot + s_idx_k)
                            sig_k, _dk2, _i0k2, _i1k2, _wtk2, _j0k2, _j1k2, _wyk2 = _triton_lv_bilinear_f32(
                                lv_table_ptr, lv_base, t_mid_k, lx + q_k, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
                            )
                            sig_sq_acc = sig_sq_acc + sig_k * sig_k
                        tl.store(vib_sig_pre_ptr + v_off, tl.sqrt(sig_sq_acc * (1.0 / float(n_sub))), mask=mask_n)

            for s in tl.static_range(n_sub):
                step_idx = period_idx * n_sub + s
                t_mid = tl.load(t_prev_ptr + step_idx)
                dt = tl.load(dt_sub_ptr + step_idx)
                sqrt_dt = tl.load(sqrt_dt_ptr + step_idx)
                t_start = tl.maximum(t_mid - 0.5 * dt, 0.0)
                q_s = tl.load(q_shift_mid_ptr + d * n_steps_tot + step_idx)

                if SAVE_VIBRATO:
                    if not VIB_PERIOD_LEVEL:
                        if (period_idx == vib_period_idx) & (s == vib_sub_idx):
                            v_off = d * n_total + gid_n
                            tl.store(vib_lx_pre_ptr + v_off, lx, mask=mask_n)
                            tl.store(vib_j_pre_ptr + v_off, j_lx, mask=mask_n)
                            sig_vib, _dv2, _i0v2, _i1v2, _wtv2, _j0v2, _j1v2, _wyv2 = _triton_lv_bilinear_f32(
                                lv_table_ptr, lv_base, t_mid, lx + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
                            )
                            tl.store(vib_sig_pre_ptr + v_off, sig_vib, mask=mask_n)

                if USE_Z_CORR:
                    zc = tl.load(sv_ptr + (step_idx * d_assets + d) * n_total + gid_n, mask=mask_n, other=0.0)
                else:
                    zc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                    for j in range(d_assets):
                        l_dj = tl.load(chol_ptr + d * d_assets + j)
                        zj = tl.load(sv_ptr + (step_idx * d_assets + j) * n_total + gid_n, mask=mask_n, other=0.0)
                        zc = zc + l_dj * zj

                lx, step_j = _triton_platen_fwd_substep_f32(
                    lv_table_ptr, lv_base, lx, t_start, dt, sqrt_dt, q_s, zc,
                    inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                )
                if SAVE_VIBRATO:
                    j_lx = tl.minimum(tl.maximum(j_lx * step_j, -6.0), 6.0)

            per_off = (period_idx * d_assets + d) * n_total + gid_n
            tl.store(lx_period_ptr + per_off, lx, mask=mask_n)
            if SAVE_VIBRATO:
                tl.store(vib_j_state_ptr + d * n_total + gid_n, j_lx, mask=mask_n)
            max_lx = tl.max(tl.where(mask_n, lx, -1.0e30), axis=0)
            ex_v = tl.where(mask_n, tl.exp(lx - max_lx), 0.0)
            t_idx = (pid_r * n_tiles + pid_n) * d_assets + d
            tl.store(tile_max_ptr + t_idx, max_lx)
            tl.store(tile_ex_sum_ptr + t_idx, tl.sum(ex_v, axis=0))

    @triton.jit  # type: ignore[untyped-decorator]
    def _triton_platen_bwd_substep_f32(  # type: ignore[no-untyped-def]
        lv_table_ptr,
        lv_base,
        bar_lv_cta_base_ptr,
        bar_chol_ptr,
        bar_q_shift_ptr,
        sv_ptr,
        chol_ptr,
        lx_pre,
        bar_x,
        t_start,
        dt,
        sqrt_dt,
        q_s,
        step_idx,
        d,
        d_assets,
        n_total,
        gid_n,
        inv_dt_grid,
        y_min_d,
        inv_dy_d,
        max_t_f,
        max_y_f,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        BLOCK_N: tl.constexpr,
        mask_n,
        COMPUTE_LV_AAD: tl.constexpr,
        COMPUTE_CHOL_AAD: tl.constexpr,
        COMPUTE_Q_AAD: tl.constexpr,
        shard_off=0,
        USE_Z_CORR: tl.constexpr = False,
        z_corr_ptr=None,
    ):
        if USE_Z_CORR:
            zc = tl.load(z_corr_ptr + (step_idx * d_assets + d) * n_total + gid_n, mask=mask_n, other=0.0)
        else:
            zc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for j in range(d_assets):
                l_dj = tl.load(chol_ptr + d * d_assets + j)
                zj = tl.load(sv_ptr + (step_idx * d_assets + j) * n_total + gid_n, mask=mask_n, other=0.0)
                zc = zc + l_dj * zj

        sig0, d0, it0_0, it1_0, wt_0, iy0_0, iy1_0, wy_0 = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_start, lx_pre + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )
        t_end = t_start + dt
        dw = sqrt_dt * zc
        a0 = -0.5 * sig0 * sig0
        lx_hat = lx_pre + a0 * dt + sig0 * dw
        lx_p = lx_pre + a0 * dt + sig0 * sqrt_dt
        lx_m = lx_pre + a0 * dt - sig0 * sqrt_dt

        sigh, dh, it0_h, it1_h, wt_h, iy0_h, iy1_h, wy_h = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_end, lx_hat + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )
        sigp, dp, it0_p, it1_p, wt_p, iy0_p, iy1_p, wy_p = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_end, lx_p + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )
        sigm, dm, it0_m, it1_m, wt_m, iy0_m, iy1_m, wy_m = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_end, lx_m + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )

        zc2m1 = zc * zc - 1.0
        bar_xm = tl.where(mask_n, bar_x, 0.0)
        bar_lvh = bar_xm * (-0.5 * dt * sigh)
        bar_lvp = bar_xm * (0.25 * (dw + sqrt_dt * zc2m1))
        bar_lvm = bar_xm * (0.25 * (dw - sqrt_dt * zc2m1))
        bar_lv0_dir = bar_xm * (-0.5 * dt * sig0 + 0.5 * dw)

        bar_lx_hat = bar_lvh * dh
        bar_lx_p = bar_lvp * dp
        bar_lx_m = bar_lvm * dm

        if COMPUTE_CHOL_AAD:
            bar_zc_tot = (
                bar_xm * (0.25 * (sigp + sigm + 2.0 * sig0) * sqrt_dt + 0.5 * (sigp - sigm) * sqrt_dt * zc)
                + bar_lx_hat * (sig0 * sqrt_dt)
            )
            for j in range(d_assets):
                zj = tl.load(sv_ptr + (step_idx * d_assets + j) * n_total + gid_n, mask=mask_n, other=0.0)
                bar_l_dj = tl.sum(tl.where(mask_n, bar_zc_tot * zj, 0.0), axis=0)
                tl.atomic_add(bar_chol_ptr + d * d_assets + j, bar_l_dj)

        bar_a0 = dt * (bar_lx_hat + bar_lx_p + bar_lx_m)
        bar_lv0 = bar_lv0_dir + bar_a0 * (-sig0) + bar_lx_hat * dw + (bar_lx_p - bar_lx_m) * sqrt_dt

        if COMPUTE_LV_AAD:
            m_act = mask_n & (tl.abs(bar_xm) > 1e-24)
            _triton_lv_cta_scatter_row_f32(
                bar_lv_cta_base_ptr, it0_0, it1_0, wt_0, iy0_0, iy1_0, wy_0, bar_lv0, N_y, m_act, 16
            )
            _triton_lv_cta_scatter_row3_f32(
                bar_lv_cta_base_ptr,
                it0_h, it1_h, wt_h,
                iy0_h, iy1_h, wy_h, bar_lvh,
                iy0_p, iy1_p, wy_p, bar_lvp,
                iy0_m, iy1_m, wy_m, bar_lvm,
                N_y, m_act, 16,
            )

        bar_y = bar_lv0 * d0 + bar_lx_hat + bar_lx_p + bar_lx_m
        _u_bq = bar_q_shift_ptr
        _u_cq = COMPUTE_Q_AAD
        _u_so = shard_off
        return bar_xm + bar_y, bar_y

    @triton.jit  # type: ignore[untyped-decorator]
    def _triton_platen_bwd_substep_step_f32(  # type: ignore[no-untyped-def]
        lv_table_ptr,
        lv_base,
        bar_lv_d_ptr,
        bar_zc_ptr,
        sv_ptr,
        z_corr_ptr,
        chol_ptr,
        lx_pre,
        bar_x,
        t_start,
        dt,
        sqrt_dt,
        q_s,
        step_idx,
        d,
        d_assets,
        n_total,
        gid_n,
        inv_dt_grid,
        y_min_d,
        inv_dy_d,
        max_t_f,
        max_y_f,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        BLOCK_N: tl.constexpr,
        mask_n,
        COMPUTE_LV_AAD: tl.constexpr,
        COMPUTE_CHOL_AAD: tl.constexpr,
        USE_Z_CORR: tl.constexpr = False,
        zc_in=0.0,
        HAS_ZC_IN: tl.constexpr = False,
    ):
        if HAS_ZC_IN:
            zc = zc_in
        elif USE_Z_CORR:
            zc = tl.load(z_corr_ptr + (step_idx * d_assets + d) * n_total + gid_n, mask=mask_n, other=0.0)
        else:
            zc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for j in range(d_assets):
                l_dj = tl.load(chol_ptr + d * d_assets + j)
                zj = tl.load(sv_ptr + (step_idx * d_assets + j) * n_total + gid_n, mask=mask_n, other=0.0)
                zc = zc + l_dj * zj

        sig0, d0, it0_0, it1_0, wt_0, iy0_0, iy1_0, wy_0 = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_start, lx_pre + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )
        t_end = t_start + dt
        dw = sqrt_dt * zc
        a0 = -0.5 * sig0 * sig0
        lx_hat = lx_pre + a0 * dt + sig0 * dw
        lx_p = lx_pre + a0 * dt + sig0 * sqrt_dt
        lx_m = lx_pre + a0 * dt - sig0 * sqrt_dt

        sigh, dh, it0_h, it1_h, wt_h, iy0_h, iy1_h, wy_h = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_end, lx_hat + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )
        sigp, dp, _i0p, _i1p, _wtp, iy0_p, iy1_p, wy_p = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_end, lx_p + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )
        sigm, dm, _i0m, _i1m, _wtm, iy0_m, iy1_m, wy_m = _triton_lv_bilinear_f32(
            lv_table_ptr, lv_base, t_end, lx_m + q_s, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n
        )

        zc2m1 = zc * zc - 1.0
        bar_xm = tl.where(mask_n, bar_x, 0.0)
        bar_lvh = bar_xm * (-0.5 * dt * sigh)
        bar_lvp = bar_xm * (0.25 * (dw + sqrt_dt * zc2m1))
        bar_lvm = bar_xm * (0.25 * (dw - sqrt_dt * zc2m1))
        bar_lv0_dir = bar_xm * (-0.5 * dt * sig0 + 0.5 * dw)

        bar_lx_hat = bar_lvh * dh
        bar_lx_p = bar_lvp * dp
        bar_lx_m = bar_lvm * dm

        if COMPUTE_CHOL_AAD:
            bar_zc_tot = (
                bar_xm * (0.25 * (sigp + sigm + 2.0 * sig0) * sqrt_dt + 0.5 * (sigp - sigm) * sqrt_dt * zc)
                + bar_lx_hat * (sig0 * sqrt_dt)
            )
            tl.store(bar_zc_ptr + (step_idx * d_assets + d) * n_total + gid_n, bar_zc_tot, mask=mask_n)

        bar_a0 = dt * (bar_lx_hat + bar_lx_p + bar_lx_m)
        bar_lv0 = bar_lv0_dir + bar_a0 * (-sig0) + bar_lx_hat * dw + (bar_lx_p - bar_lx_m) * sqrt_dt

        if COMPUTE_LV_AAD:
            m_act = mask_n & (tl.abs(bar_xm) > 1e-24)
            _triton_lv_direct_scatter_f32(
                bar_lv_d_ptr, it0_0, it1_0, wt_0, iy0_0, iy1_0, wy_0, bar_lv0, N_y, m_act
            )
            _triton_lv_direct_scatter3_f32(
                bar_lv_d_ptr,
                it0_h, it1_h, wt_h,
                iy0_h, iy1_h, wy_h, bar_lvh,
                iy0_p, iy1_p, wy_p, bar_lvp,
                iy0_m, iy1_m, wy_m, bar_lvm,
                N_y, m_act,
            )

        bar_y = bar_lv0 * d0 + bar_lx_hat + bar_lx_p + bar_lx_m
        return bar_xm + bar_y, bar_y

    @triton.jit  # type: ignore[untyped-decorator]
    def _triton_load_step_params_f32(  # type: ignore[no-untyped-def]
        sv_ptr,
        chol_ptr,
        t_prev_ptr,
        dt_sub_ptr,
        sqrt_dt_ptr,
        q_shift_mid_ptr,
        step_idx,
        d,
        d_assets,
        n_steps_tot,
        n_total,
        gid_n,
        BLOCK_N: tl.constexpr,
        mask_n,
        USE_Z_CORR: tl.constexpr = False,
    ):
        t_mid = tl.load(t_prev_ptr + step_idx)
        dt = tl.load(dt_sub_ptr + step_idx)
        sqrt_dt = tl.load(sqrt_dt_ptr + step_idx)
        t_start = tl.maximum(t_mid - 0.5 * dt, 0.0)
        q_s = tl.load(q_shift_mid_ptr + d * n_steps_tot + step_idx)
        if USE_Z_CORR:
            zc = tl.load(sv_ptr + (step_idx * d_assets + d) * n_total + gid_n, mask=mask_n, other=0.0)
        else:
            zc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for j in range(d_assets):
                l_dj = tl.load(chol_ptr + d * d_assets + j)
                zj = tl.load(sv_ptr + (step_idx * d_assets + j) * n_total + gid_n, mask=mask_n, other=0.0)
                zc = zc + l_dj * zj
        return t_start, dt, sqrt_dt, q_s, zc

    @triton.jit  # type: ignore[untyped-decorator]
    def _qmc_bwd_sde_aad_kernel(  # type: ignore[no-untyped-def]
        sv_ptr,
        shifts_ptr,
        chol_ptr,
        lv_table_ptr,
        y_min_ptr,
        inv_dy_ptr,
        t_prev_ptr,
        dt_sub_ptr,
        sqrt_dt_ptr,
        q_shift_mid_ptr,
        bar_x_post_ptr,
        bar_lx_post_ptr,
        lx_period_ptr,
        bar_lv_cta_ptr,
        bar_chol_ptr,
        bar_q_shift_ptr,
        inv_dt_grid,
        d_assets,
        m_rem,
        n_rep: tl.constexpr,
        n_sub: tl.constexpr,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        BLOCK_N: tl.constexpr,
        period_idx: int = 0,
        FUSE_M: tl.constexpr = False,
        COMPUTE_LV_AAD: tl.constexpr = True,
        COMPUTE_CHOL_AAD: tl.constexpr = True,
        COMPUTE_Q_AAD: tl.constexpr = True,
        ATOM_SHARDS: tl.constexpr = 1,
    ):
        _u_sh = shifts_ptr
        pid_n = tl.program_id(0)
        pid_dr = tl.program_id(1)
        pid_r = pid_dr // d_assets
        d = pid_dr % d_assets
        n_replicates = tl.num_programs(1) // d_assets
        n_total = n_replicates * n_rep
        n_steps_tot = m_rem * n_sub

        offs_rep = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_rep < n_rep
        gid_n = pid_r * n_rep + offs_rep

        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4
        y_min_d = tl.load(y_min_ptr + d)
        inv_dy_d = tl.load(inv_dy_ptr + d)
        lv_base = d * (N_t * N_y)
        _u_nr = n_replicates
        _u_as = ATOM_SHARDS
        bar_lv_cta_base_ptr = bar_lv_cta_ptr + d * (N_t * N_y)
        shard_off = 0

        if FUSE_M:
            bar_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for rev_m in range(m_rem):
                p_cur = m_rem - 1 - rev_m
                p_off = (p_cur * d_assets + d) * n_total + gid_n
                bar_tot = tl.where(
                    mask_n, bar_x + tl.load(bar_lx_post_ptr + p_off, mask=mask_n, other=0.0), 0.0
                )
                lx_p = tl.load(lx_period_ptr + p_off, mask=mask_n, other=0.0)
                omega = tl.where(mask_n, tl.exp(lx_p) / float(n_rep), 0.0)
                omega_sum = tl.sum(bar_tot, axis=0)
                bar_x = tl.where(mask_n, bar_tot - omega * omega_sum, 0.0)

                has_prev = p_cur > 0
                prev_m = tl.maximum(p_cur - 1, 0)
                lx_start = tl.load(
                    lx_period_ptr + (prev_m * d_assets + d) * n_total + gid_n,
                    mask=mask_n & has_prev,
                    other=0.0,
                )

                if n_sub == 4:
                    b_s = p_cur * 4
                    ts0, dt0, sd0, qs0, zc0 = _triton_load_step_params_f32(
                        sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                        b_s + 0, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n,
                    )
                    lx_1, _sj0 = _triton_platen_fwd_substep_f32(
                        lv_table_ptr, lv_base, lx_start, ts0, dt0, sd0, qs0, zc0,
                        inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                    )
                    ts1, dt1, sd1, qs1, zc1 = _triton_load_step_params_f32(
                        sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                        b_s + 1, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n,
                    )
                    lx_2, _sj1 = _triton_platen_fwd_substep_f32(
                        lv_table_ptr, lv_base, lx_1, ts1, dt1, sd1, qs1, zc1,
                        inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                    )
                    ts2, dt2, sd2, qs2, zc2 = _triton_load_step_params_f32(
                        sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                        b_s + 2, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n,
                    )
                    lx_3, _sj2 = _triton_platen_fwd_substep_f32(
                        lv_table_ptr, lv_base, lx_2, ts2, dt2, sd2, qs2, zc2,
                        inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                    )
                    tm3 = tl.load(t_prev_ptr + b_s + 3)
                    dt3 = tl.load(dt_sub_ptr + b_s + 3)
                    sd3 = tl.load(sqrt_dt_ptr + b_s + 3)
                    ts3 = tl.maximum(tm3 - 0.5 * dt3, 0.0)
                    qs3 = tl.load(q_shift_mid_ptr + d * n_steps_tot + b_s + 3)
                    bar_x, bar_y3 = _triton_platen_bwd_substep_f32(
                        lv_table_ptr, lv_base, bar_lv_cta_base_ptr, bar_chol_ptr, bar_q_shift_ptr,
                        sv_ptr, chol_ptr, lx_3, bar_x, ts3, dt3, sd3, qs3, b_s + 3,
                        d, d_assets, n_total, gid_n, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f,
                        N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, False, shard_off,
                    )
                    if COMPUTE_Q_AAD:
                        tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 3, tl.sum(bar_y3, axis=0))
                    bar_x, bar_y2 = _triton_platen_bwd_substep_f32(
                        lv_table_ptr, lv_base, bar_lv_cta_base_ptr, bar_chol_ptr, bar_q_shift_ptr,
                        sv_ptr, chol_ptr, lx_2, bar_x, ts2, dt2, sd2, qs2, b_s + 2,
                        d, d_assets, n_total, gid_n, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f,
                        N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, False, shard_off,
                    )
                    if COMPUTE_Q_AAD:
                        tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 2, tl.sum(bar_y2, axis=0))
                    bar_x, bar_y1 = _triton_platen_bwd_substep_f32(
                        lv_table_ptr, lv_base, bar_lv_cta_base_ptr, bar_chol_ptr, bar_q_shift_ptr,
                        sv_ptr, chol_ptr, lx_1, bar_x, ts1, dt1, sd1, qs1, b_s + 1,
                        d, d_assets, n_total, gid_n, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f,
                        N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, False, shard_off,
                    )
                    if COMPUTE_Q_AAD:
                        tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 1, tl.sum(bar_y1, axis=0))
                    bar_x, bar_y0 = _triton_platen_bwd_substep_f32(
                        lv_table_ptr, lv_base, bar_lv_cta_base_ptr, bar_chol_ptr, bar_q_shift_ptr,
                        sv_ptr, chol_ptr, lx_start, bar_x, ts0, dt0, sd0, qs0, b_s + 0,
                        d, d_assets, n_total, gid_n, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f,
                        N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, False, shard_off,
                    )
                    if COMPUTE_Q_AAD:
                        tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 0, tl.sum(bar_y0, axis=0))
                else:
                    for rev_s in tl.static_range(n_sub):
                        s = n_sub - 1 - rev_s
                        lx_pre = lx_start
                        for k in range(n_sub - 1):
                            if k < s:
                                k_idx = p_cur * n_sub + k
                                tsk, dtk, sdk, qk, zck = _triton_load_step_params_f32(
                                    sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                                    k_idx, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n,
                                )
                                lx_pre, _step_jk = _triton_platen_fwd_substep_f32(
                                    lv_table_ptr, lv_base, lx_pre, tsk, dtk, sdk, qk, zck,
                                    inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                                )

                        step_idx = p_cur * n_sub + s
                        t_mid = tl.load(t_prev_ptr + step_idx)
                        dt = tl.load(dt_sub_ptr + step_idx)
                        sqrt_dt = tl.load(sqrt_dt_ptr + step_idx)
                        t_start = tl.maximum(t_mid - 0.5 * dt, 0.0)
                        q_s = tl.load(q_shift_mid_ptr + d * n_steps_tot + step_idx)

                        bar_next, bar_y = _triton_platen_bwd_substep_f32(
                            lv_table_ptr, lv_base, bar_lv_cta_base_ptr, bar_chol_ptr, bar_q_shift_ptr,
                            sv_ptr, chol_ptr, lx_pre, bar_x, t_start, dt, sqrt_dt, q_s, step_idx,
                            d, d_assets, n_total, gid_n, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f,
                            N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, False, shard_off,
                        )
                        if COMPUTE_Q_AAD:
                            tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + step_idx, tl.sum(bar_y, axis=0))
                        bar_x = bar_next
        else:
            bar_x = tl.load(bar_x_post_ptr + d * n_total + gid_n, mask=mask_n, other=0.0)
            has_prev = period_idx > 0
            prev_m = tl.maximum(period_idx - 1, 0)
            lx_start = tl.load(
                lx_period_ptr + (prev_m * d_assets + d) * n_total + gid_n,
                mask=mask_n & has_prev,
                other=0.0,
            )

            if n_sub == 4:
                b_s = period_idx * 4
                ts0, dt0, sd0, qs0, zc0 = _triton_load_step_params_f32(
                    sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                    b_s + 0, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n,
                )
                lx_1, _sj0_2 = _triton_platen_fwd_substep_f32(
                    lv_table_ptr, lv_base, lx_start, ts0, dt0, sd0, qs0, zc0,
                    inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                )
                ts1, dt1, sd1, qs1, zc1 = _triton_load_step_params_f32(
                    sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                    b_s + 1, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n,
                )
                lx_2, _sj1_2 = _triton_platen_fwd_substep_f32(
                    lv_table_ptr, lv_base, lx_1, ts1, dt1, sd1, qs1, zc1,
                    inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                )
                ts2, dt2, sd2, qs2, zc2 = _triton_load_step_params_f32(
                    sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                    b_s + 2, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n,
                )
                lx_3, _sj2_2 = _triton_platen_fwd_substep_f32(
                    lv_table_ptr, lv_base, lx_2, ts2, dt2, sd2, qs2, zc2,
                    inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                )
                tm3 = tl.load(t_prev_ptr + b_s + 3)
                dt3 = tl.load(dt_sub_ptr + b_s + 3)
                sd3 = tl.load(sqrt_dt_ptr + b_s + 3)
                ts3 = tl.maximum(tm3 - 0.5 * dt3, 0.0)
                qs3 = tl.load(q_shift_mid_ptr + d * n_steps_tot + b_s + 3)
                bar_x, bar_y3 = _triton_platen_bwd_substep_f32(
                    lv_table_ptr, lv_base, bar_lv_cta_base_ptr, bar_chol_ptr, bar_q_shift_ptr,
                    sv_ptr, chol_ptr, lx_3, bar_x, ts3, dt3, sd3, qs3, b_s + 3,
                    d, d_assets, n_total, gid_n, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f,
                    N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, False, shard_off,
                )
                if COMPUTE_Q_AAD:
                    tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 3, tl.sum(bar_y3, axis=0))
                bar_x, bar_y2 = _triton_platen_bwd_substep_f32(
                    lv_table_ptr, lv_base, bar_lv_cta_base_ptr, bar_chol_ptr, bar_q_shift_ptr,
                    sv_ptr, chol_ptr, lx_2, bar_x, ts2, dt2, sd2, qs2, b_s + 2,
                    d, d_assets, n_total, gid_n, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f,
                    N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, False, shard_off,
                )
                if COMPUTE_Q_AAD:
                    tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 2, tl.sum(bar_y2, axis=0))
                bar_x, bar_y1 = _triton_platen_bwd_substep_f32(
                    lv_table_ptr, lv_base, bar_lv_cta_base_ptr, bar_chol_ptr, bar_q_shift_ptr,
                    sv_ptr, chol_ptr, lx_1, bar_x, ts1, dt1, sd1, qs1, b_s + 1,
                    d, d_assets, n_total, gid_n, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f,
                    N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, False, shard_off,
                )
                if COMPUTE_Q_AAD:
                    tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 1, tl.sum(bar_y1, axis=0))
                bar_x, bar_y0 = _triton_platen_bwd_substep_f32(
                    lv_table_ptr, lv_base, bar_lv_cta_base_ptr, bar_chol_ptr, bar_q_shift_ptr,
                    sv_ptr, chol_ptr, lx_start, bar_x, ts0, dt0, sd0, qs0, b_s + 0,
                    d, d_assets, n_total, gid_n, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f,
                    N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, False, shard_off,
                )
                if COMPUTE_Q_AAD:
                    tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 0, tl.sum(bar_y0, axis=0))
            else:
                for rev_s in tl.static_range(n_sub):
                    s = n_sub - 1 - rev_s
                    lx_pre = lx_start
                    for k in range(n_sub - 1):
                        if k < s:
                            k_idx = period_idx * n_sub + k
                            tsk, dtk, sdk, qk, zck = _triton_load_step_params_f32(
                                sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                                k_idx, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n,
                            )
                            lx_pre, _step_jk2 = _triton_platen_fwd_substep_f32(
                                lv_table_ptr, lv_base, lx_pre, tsk, dtk, sdk, qk, zck,
                                inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                            )

                    step_idx = period_idx * n_sub + s
                    t_mid = tl.load(t_prev_ptr + step_idx)
                    dt = tl.load(dt_sub_ptr + step_idx)
                    sqrt_dt = tl.load(sqrt_dt_ptr + step_idx)
                    t_start = tl.maximum(t_mid - 0.5 * dt, 0.0)
                    q_s = tl.load(q_shift_mid_ptr + d * n_steps_tot + step_idx)

                    bar_next, bar_y = _triton_platen_bwd_substep_f32(
                        lv_table_ptr, lv_base, bar_lv_cta_base_ptr, bar_chol_ptr, bar_q_shift_ptr,
                        sv_ptr, chol_ptr, lx_pre, bar_x, t_start, dt, sqrt_dt, q_s, step_idx,
                        d, d_assets, n_total, gid_n, inv_dt_grid, y_min_d, inv_dy_d, max_t_f, max_y_f,
                        N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, False, shard_off,
                    )
                    if COMPUTE_Q_AAD:
                        tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + step_idx, tl.sum(bar_y, axis=0))
                    bar_x = bar_next

            tl.store(bar_x_post_ptr + d * n_total + gid_n, bar_x, mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _qmc_bwd_state_prop_kernel(  # type: ignore[no-untyped-def]
        sv_ptr,
        chol_ptr,
        lv_table_ptr,
        y_min_ptr,
        inv_dy_ptr,
        t_prev_ptr,
        dt_sub_ptr,
        sqrt_dt_ptr,
        q_shift_mid_ptr,
        bar_lx_post_ptr,
        lx_period_ptr,
        dummy_ptr,
        inv_dt_grid,
        d_assets,
        m_rem,
        cg_flags_ptr=None,
        payoff_f64_ptr=None,
        n_rep: tl.constexpr = 1024,
        n_sub: tl.constexpr = 4,
        N_t: tl.constexpr = 32,
        N_y: tl.constexpr = 64,
        BLOCK_N: tl.constexpr = 256,
        USE_Z_CORR: tl.constexpr = False,
        USE_RT_FLAGS: tl.constexpr = False,
    ):
        pid_dr = tl.program_id(1)
        pid_r = pid_dr // d_assets
        d = pid_dr % d_assets
        if USE_RT_FLAGS:
            d_act = tl.load(cg_flags_ptr + 0)
            m_act = tl.load(cg_flags_ptr + 1)
            if d >= d_act:
                return
            eff_inv_dt = tl.load(payoff_f64_ptr + 17).to(tl.float32)
        else:
            m_act = m_rem
            eff_inv_dt = inv_dt_grid
        n_replicates = tl.num_programs(1) // d_assets
        n_total = n_replicates * n_rep
        n_steps_tot = m_rem * n_sub

        offs_rep = tl.arange(0, BLOCK_N)
        mask_n = offs_rep < n_rep
        gid_n = pid_r * n_rep + offs_rep

        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4
        y_min_d = tl.load(y_min_ptr + d)
        inv_dy_d = tl.load(inv_dy_ptr + d)
        lv_base = d * (N_t * N_y)

        bar_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for rev_m in range(m_rem):
            p_cur = m_rem - 1 - rev_m
            if p_cur < m_act:
                p_off = (p_cur * d_assets + d) * n_total + gid_n
                bar_tot = tl.where(
                    mask_n, bar_x + tl.load(bar_lx_post_ptr + p_off, mask=mask_n, other=0.0), 0.0
                )
                lx_p = tl.load(lx_period_ptr + p_off, mask=mask_n, other=0.0)
                omega = tl.where(mask_n, tl.exp(lx_p) / float(n_rep), 0.0)
                omega_sum = tl.sum(bar_tot, axis=0)
                bar_x = tl.where(mask_n, bar_tot - omega * omega_sum, 0.0)
                tl.store(bar_lx_post_ptr + p_off, bar_x, mask=mask_n)

                if p_cur > 0:
                    prev_m = p_cur - 1
                    lx_start = tl.load(
                        lx_period_ptr + (prev_m * d_assets + d) * n_total + gid_n,
                        mask=mask_n,
                        other=0.0,
                    )
                    if n_sub == 4:
                        b_s = p_cur * 4
                        ts0, dt0, sd0, qs0, zc0 = _triton_load_step_params_f32(
                            sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                            b_s + 0, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n, USE_Z_CORR,
                        )
                        lx_1, _sj0_p = _triton_platen_fwd_substep_f32(
                            lv_table_ptr, lv_base, lx_start, ts0, dt0, sd0, qs0, zc0,
                            eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                        )
                        ts1, dt1, sd1, qs1, zc1 = _triton_load_step_params_f32(
                            sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                            b_s + 1, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n, USE_Z_CORR,
                        )
                        lx_2, _sj1_p = _triton_platen_fwd_substep_f32(
                            lv_table_ptr, lv_base, lx_1, ts1, dt1, sd1, qs1, zc1,
                            eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                        )
                        ts2, dt2, sd2, qs2, zc2 = _triton_load_step_params_f32(
                            sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                            b_s + 2, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n, USE_Z_CORR,
                        )
                        lx_3, _sj2_p = _triton_platen_fwd_substep_f32(
                            lv_table_ptr, lv_base, lx_2, ts2, dt2, sd2, qs2, zc2,
                            eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                        )
                        tm3 = tl.load(t_prev_ptr + b_s + 3)
                        dt3 = tl.load(dt_sub_ptr + b_s + 3)
                        sd3 = tl.load(sqrt_dt_ptr + b_s + 3)
                        ts3 = tl.maximum(tm3 - 0.5 * dt3, 0.0)
                        qs3 = tl.load(q_shift_mid_ptr + d * n_steps_tot + b_s + 3)
                        bar_x, _by3 = _triton_platen_bwd_substep_f32(
                            lv_table_ptr, lv_base, dummy_ptr, dummy_ptr, dummy_ptr,
                            sv_ptr, chol_ptr, lx_3, bar_x, ts3, dt3, sd3, qs3, b_s + 3,
                            d, d_assets, n_total, gid_n, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f,
                            N_t, N_y, BLOCK_N, mask_n, False, False, False, 0, USE_Z_CORR, sv_ptr,
                        )
                        bar_x, _by2 = _triton_platen_bwd_substep_f32(
                            lv_table_ptr, lv_base, dummy_ptr, dummy_ptr, dummy_ptr,
                            sv_ptr, chol_ptr, lx_2, bar_x, ts2, dt2, sd2, qs2, b_s + 2,
                            d, d_assets, n_total, gid_n, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f,
                            N_t, N_y, BLOCK_N, mask_n, False, False, False, 0, USE_Z_CORR, sv_ptr,
                        )
                        bar_x, _by1 = _triton_platen_bwd_substep_f32(
                            lv_table_ptr, lv_base, dummy_ptr, dummy_ptr, dummy_ptr,
                            sv_ptr, chol_ptr, lx_1, bar_x, ts1, dt1, sd1, qs1, b_s + 1,
                            d, d_assets, n_total, gid_n, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f,
                            N_t, N_y, BLOCK_N, mask_n, False, False, False, 0, USE_Z_CORR, sv_ptr,
                        )
                        bar_x, _by0 = _triton_platen_bwd_substep_f32(
                            lv_table_ptr, lv_base, dummy_ptr, dummy_ptr, dummy_ptr,
                            sv_ptr, chol_ptr, lx_start, bar_x, ts0, dt0, sd0, qs0, b_s + 0,
                            d, d_assets, n_total, gid_n, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f,
                            N_t, N_y, BLOCK_N, mask_n, False, False, False, 0, USE_Z_CORR, sv_ptr,
                        )
                    else:
                        for rev_s in tl.static_range(n_sub):
                            s = n_sub - 1 - rev_s
                            lx_pre = lx_start
                            for k in range(n_sub - 1):
                                if k < s:
                                    k_idx = p_cur * n_sub + k
                                    tsk, dtk, sdk, qk, zck = _triton_load_step_params_f32(
                                        sv_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                                        k_idx, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n, USE_Z_CORR,
                                    )
                                    lx_pre, _sjk = _triton_platen_fwd_substep_f32(
                                        lv_table_ptr, lv_base, lx_pre, tsk, dtk, sdk, qk, zck,
                                        eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                                    )
                            step_idx = p_cur * n_sub + s
                            t_mid = tl.load(t_prev_ptr + step_idx)
                            dt = tl.load(dt_sub_ptr + step_idx)
                            sqrt_dt = tl.load(sqrt_dt_ptr + step_idx)
                            t_start = tl.maximum(t_mid - 0.5 * dt, 0.0)
                            q_s = tl.load(q_shift_mid_ptr + d * n_steps_tot + step_idx)
                            bar_x, _bys = _triton_platen_bwd_substep_f32(
                                lv_table_ptr, lv_base, dummy_ptr, dummy_ptr, dummy_ptr,
                                sv_ptr, chol_ptr, lx_pre, bar_x, t_start, dt, sqrt_dt, q_s, step_idx,
                                d, d_assets, n_total, gid_n, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f,
                                N_t, N_y, BLOCK_N, mask_n, False, False, False, 0, USE_Z_CORR, sv_ptr,
                            )

    @triton.jit  # type: ignore[untyped-decorator]
    def _qmc_bwd_period_scatter_kernel(  # type: ignore[no-untyped-def]
        sv_ptr,
        z_corr_ptr,
        chol_ptr,
        lv_table_ptr,
        y_min_ptr,
        inv_dy_ptr,
        t_prev_ptr,
        dt_sub_ptr,
        sqrt_dt_ptr,
        q_shift_mid_ptr,
        bar_lx_post_ptr,
        lx_period_ptr,
        bar_step_shard_ptr,
        bar_zc_ptr,
        bar_q_shift_ptr,
        inv_dt_grid,
        d_assets,
        m_rem,
        n_replicates,
        cg_flags_ptr=None,
        payoff_f64_ptr=None,
        n_rep: tl.constexpr = 1024,
        n_sub: tl.constexpr = 4,
        N_t: tl.constexpr = 32,
        N_y: tl.constexpr = 64,
        BLOCK_N: tl.constexpr = 128,
        N_LV_SHARDS: tl.constexpr = 1,
        COMPUTE_LV_AAD: tl.constexpr = True,
        COMPUTE_CHOL_AAD: tl.constexpr = True,
        COMPUTE_Q_AAD: tl.constexpr = True,
        USE_Z_CORR: tl.constexpr = False,
        USE_RT_FLAGS: tl.constexpr = False,
    ):
        pid_n = tl.program_id(0)
        pid_mrd = tl.program_id(1)
        n_tiles_p = tl.num_programs(0)
        rd_tot = n_replicates * d_assets
        p_cur = pid_mrd // rd_tot
        rem = pid_mrd % rd_tot
        pid_r = rem // d_assets
        d = rem % d_assets
        if USE_RT_FLAGS:
            d_act = tl.load(cg_flags_ptr + 0)
            m_act = tl.load(cg_flags_ptr + 1)
            if (d >= d_act) | (p_cur >= m_act):
                return
            eff_inv_dt = tl.load(payoff_f64_ptr + 17).to(tl.float32)
        else:
            eff_inv_dt = inv_dt_grid

        offs_rep = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_rep < n_rep
        n_total = n_replicates * n_rep
        gid_n = pid_r * n_rep + offs_rep
        n_steps_tot = m_rem * n_sub

        p_off = (p_cur * d_assets + d) * n_total + gid_n
        bar_x = tl.where(mask_n, tl.load(bar_lx_post_ptr + p_off, mask=mask_n, other=0.0), 0.0)
        if tl.max(tl.abs(bar_x), axis=0) <= 1e-24:
            if COMPUTE_CHOL_AAD:
                z0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for s_z in tl.static_range(n_sub):
                    tl.store(bar_zc_ptr + ((p_cur * n_sub + s_z) * d_assets + d) * n_total + gid_n, z0, mask=mask_n)
            return

        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4
        y_min_d = tl.load(y_min_ptr + d)
        inv_dy_d = tl.load(inv_dy_ptr + d)
        lv_base = d * (N_t * N_y)

        _u_tp = n_tiles_p
        n_mr = m_rem * n_replicates
        mr_idx = pid_mrd // d_assets
        shard_id = ((pid_n * n_mr + mr_idx) % N_LV_SHARDS) if N_LV_SHARDS > 1 else 0
        bar_step_base_ptr = bar_step_shard_ptr + (shard_id * d_assets + d) * (N_t * N_y)
        z_load_ptr = z_corr_ptr if USE_Z_CORR else sv_ptr

        has_prev = p_cur > 0
        prev_m = tl.maximum(p_cur - 1, 0)
        lx_start = tl.load(
            lx_period_ptr + (prev_m * d_assets + d) * n_total + gid_n,
            mask=mask_n & has_prev,
            other=0.0,
        )

        if n_sub == 4:
            b_s = p_cur * 4
            ts0, dt0, sd0, qs0, zc0 = _triton_load_step_params_f32(
                z_load_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                b_s + 0, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n, USE_Z_CORR,
            )
            lx_1, _sj0_sc = _triton_platen_fwd_substep_f32(
                lv_table_ptr, lv_base, lx_start, ts0, dt0, sd0, qs0, zc0,
                eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
            )
            ts1, dt1, sd1, qs1, zc1 = _triton_load_step_params_f32(
                z_load_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                b_s + 1, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n, USE_Z_CORR,
            )
            lx_2, _sj1_sc = _triton_platen_fwd_substep_f32(
                lv_table_ptr, lv_base, lx_1, ts1, dt1, sd1, qs1, zc1,
                eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
            )
            ts2, dt2, sd2, qs2, zc2 = _triton_load_step_params_f32(
                z_load_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                b_s + 2, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n, USE_Z_CORR,
            )
            lx_3, _sj2_sc = _triton_platen_fwd_substep_f32(
                lv_table_ptr, lv_base, lx_2, ts2, dt2, sd2, qs2, zc2,
                eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
            )
            tm3 = tl.load(t_prev_ptr + b_s + 3)
            dt3 = tl.load(dt_sub_ptr + b_s + 3)
            sd3 = tl.load(sqrt_dt_ptr + b_s + 3)
            ts3 = tl.maximum(tm3 - 0.5 * dt3, 0.0)
            qs3 = tl.load(q_shift_mid_ptr + d * n_steps_tot + b_s + 3)
            bar_x, bar_y3 = _triton_platen_bwd_substep_step_f32(
                lv_table_ptr, lv_base, bar_step_base_ptr, bar_zc_ptr,
                sv_ptr, z_corr_ptr, chol_ptr, lx_3, bar_x, ts3, dt3, sd3, qs3, b_s + 3,
                d, d_assets, n_total, gid_n, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f,
                N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, USE_Z_CORR,
            )
            if COMPUTE_Q_AAD:
                tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 3, tl.sum(bar_y3, axis=0), sem="relaxed")
            bar_x, bar_y2 = _triton_platen_bwd_substep_step_f32(
                lv_table_ptr, lv_base, bar_step_base_ptr, bar_zc_ptr,
                sv_ptr, z_corr_ptr, chol_ptr, lx_2, bar_x, ts2, dt2, sd2, qs2, b_s + 2,
                d, d_assets, n_total, gid_n, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f,
                N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, USE_Z_CORR, zc2, True,
            )
            if COMPUTE_Q_AAD:
                tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 2, tl.sum(bar_y2, axis=0), sem="relaxed")
            bar_x, bar_y1 = _triton_platen_bwd_substep_step_f32(
                lv_table_ptr, lv_base, bar_step_base_ptr, bar_zc_ptr,
                sv_ptr, z_corr_ptr, chol_ptr, lx_1, bar_x, ts1, dt1, sd1, qs1, b_s + 1,
                d, d_assets, n_total, gid_n, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f,
                N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, USE_Z_CORR, zc1, True,
            )
            if COMPUTE_Q_AAD:
                tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 1, tl.sum(bar_y1, axis=0), sem="relaxed")
            bar_x, bar_y0 = _triton_platen_bwd_substep_step_f32(
                lv_table_ptr, lv_base, bar_step_base_ptr, bar_zc_ptr,
                sv_ptr, z_corr_ptr, chol_ptr, lx_start, bar_x, ts0, dt0, sd0, qs0, b_s + 0,
                d, d_assets, n_total, gid_n, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f,
                N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, USE_Z_CORR, zc0, True,
            )
            if COMPUTE_Q_AAD:
                tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + b_s + 0, tl.sum(bar_y0, axis=0), sem="relaxed")
        else:
            for rev_s in tl.static_range(n_sub):
                s = n_sub - 1 - rev_s
                lx_pre = lx_start
                for k in range(n_sub - 1):
                    if k < s:
                        k_idx = p_cur * n_sub + k
                        tsk, dtk, sdk, qk, zck = _triton_load_step_params_f32(
                            z_load_ptr, chol_ptr, t_prev_ptr, dt_sub_ptr, sqrt_dt_ptr, q_shift_mid_ptr,
                            k_idx, d, d_assets, n_steps_tot, n_total, gid_n, BLOCK_N, mask_n, USE_Z_CORR,
                        )
                        lx_pre, _sjk_sc = _triton_platen_fwd_substep_f32(
                            lv_table_ptr, lv_base, lx_pre, tsk, dtk, sdk, qk, zck,
                            eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f, N_t, N_y, mask_n,
                        )
                step_idx = p_cur * n_sub + s
                t_mid = tl.load(t_prev_ptr + step_idx)
                dt = tl.load(dt_sub_ptr + step_idx)
                sqrt_dt = tl.load(sqrt_dt_ptr + step_idx)
                t_start = tl.maximum(t_mid - 0.5 * dt, 0.0)
                q_s = tl.load(q_shift_mid_ptr + d * n_steps_tot + step_idx)
                bar_x, bar_y = _triton_platen_bwd_substep_step_f32(
                    lv_table_ptr, lv_base, bar_step_base_ptr, bar_zc_ptr,
                    sv_ptr, z_corr_ptr, chol_ptr, lx_pre, bar_x, t_start, dt, sqrt_dt, q_s, step_idx,
                    d, d_assets, n_total, gid_n, eff_inv_dt, y_min_d, inv_dy_d, max_t_f, max_y_f,
                    N_t, N_y, BLOCK_N, mask_n, COMPUTE_LV_AAD, COMPUTE_CHOL_AAD, USE_Z_CORR,
                )
                if COMPUTE_Q_AAD:
                    tl.atomic_add(bar_q_shift_ptr + d * n_steps_tot + step_idx, tl.sum(bar_y, axis=0), sem="relaxed")

    @triton.jit  # type: ignore[untyped-decorator]
    def _triton_lv_bilinear_f64(  # type: ignore[no-untyped-def]
        lv_tables_ptr,
        lv_base,
        t_val,
        y_val,
        inv_dt_grid,
        y_min,
        inv_dy_grid,
        max_t_f,
        max_y_f,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        mask_n,
    ):
        u_t = tl.minimum(tl.maximum(t_val * inv_dt_grid, 0.0), max_t_f)
        it0 = u_t.to(tl.int32)
        wt = u_t - it0.to(tl.float64)
        it1 = tl.minimum(it0 + 1, N_t - 1)
        u_y = tl.minimum(tl.maximum((y_val - y_min) * inv_dy_grid, 0.0), max_y_f)
        iy0 = u_y.to(tl.int32)
        wy = u_y - iy0.to(tl.float64)
        iy1 = tl.minimum(iy0 + 1, N_y - 1)
        s00 = tl.load(lv_tables_ptr + lv_base + it0 * N_y + iy0, mask=mask_n, other=0.0)
        s01 = tl.load(lv_tables_ptr + lv_base + it0 * N_y + iy1, mask=mask_n, other=0.0)
        s10 = tl.load(lv_tables_ptr + lv_base + it1 * N_y + iy0, mask=mask_n, other=0.0)
        s11 = tl.load(lv_tables_ptr + lv_base + it1 * N_y + iy1, mask=mask_n, other=0.0)
        v0 = (1.0 - wt) * s00 + wt * s10
        v1 = (1.0 - wt) * s01 + wt * s11
        return (1.0 - wy) * v0 + wy * v1

    @triton.jit  # type: ignore[untyped-decorator]
    def _crn_step_1d_f64_kernel(  # type: ignore[no-untyped-def]
        lv_tables_ptr,
        z_corr_ptr,
        tau_obs_ptr,
        r_native_obs_ptr,
        r_tot_obs_ptr,
        meta_lv_ptr,
        meta_r_ptr,
        meta_z_ptr,
        meta_tset_ptr,
        y_min_vec_ptr,
        inv_dy_vec_ptr,
        inv_dt_vec_ptr,
        out_norm_x_ptr,
        tile_max_ptr,
        tile_ex_sum_ptr,
        b_sz,
        m_idx,
        m_total,
        n_tiles,
        z_mode: tl.constexpr,
        n_rep: tl.constexpr,
        n_sub: tl.constexpr,
        N_t: tl.constexpr,
        N_y: tl.constexpr,
        BLOCK_N: tl.constexpr,
        FUSE_M: tl.constexpr = False,
    ):
        pid_n = tl.program_id(0)
        pid_br = tl.program_id(1)
        n_replicates = tl.num_programs(1) // b_sz
        b = pid_br // n_replicates
        r = pid_br % n_replicates
        n_total = n_replicates * n_rep

        offs_rep = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_rep < n_rep
        gid_n = r * n_rep + offs_rep

        max_t_f = float(N_t - 1) - 1e-6
        max_y_f = float(N_y - 1) - 1e-6
        if z_mode == 2:
            lv_idx = tl.load(meta_lv_ptr + b)
            r_idx = tl.load(meta_r_ptr + b)
            z_idx = tl.load(meta_z_ptr + b)
        else:
            lv_idx = 0 if z_mode == 1 else b
            r_idx = 0
            z_idx = b if z_mode == 1 else 0
        t_set = tl.load(meta_tset_ptr + b) if meta_tset_ptr is not None else 0
        y_idx = r_idx if z_mode == 2 else lv_idx
        y_min = tl.load(y_min_vec_ptr + y_idx)
        inv_dy_grid = tl.load(inv_dy_vec_ptr + y_idx)
        inv_dt_grid = tl.load(inv_dt_vec_ptr + t_set)
        lv_base = lv_idx * (N_t * N_y)
        r_off = r_idx * m_total
        tau_off = t_set * m_total

        if FUSE_M:
            lx = tl.zeros((BLOCK_N,), dtype=tl.float64)
            t_prev = tl.zeros((), dtype=tl.float64)
            rt_prev = tl.zeros((), dtype=tl.float64)
            rn_prev = tl.zeros((), dtype=tl.float64)
            for m_cur in range(m_total):
                t_cur = tl.load(tau_obs_ptr + tau_off + m_cur)
                rt_cur = tl.load(r_tot_obs_ptr + r_off + m_cur)
                rn_cur = tl.load(r_native_obs_ptr + r_off + m_cur)
                d_tau = tl.maximum(t_cur - t_prev, 1e-10)
                dt = d_tau / float(n_sub)
                sqrt_dt = tl.sqrt(dt)
                drt = rt_cur - rt_prev
                drn = rn_cur - rn_prev

                for s in tl.static_range(n_sub):
                    alpha_0 = float(s) / float(n_sub)
                    alpha_1 = float(s + 1) / float(n_sub)
                    alpha_s = 0.5 * (alpha_0 + alpha_1)
                    t_start = t_prev + alpha_0 * d_tau
                    t_end = t_prev + alpha_1 * d_tau
                    q_s = (rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn)

                    if z_mode == 0:
                        z_off = (m_cur * n_sub + s) * n_total + gid_n
                    else:
                        z_off = ((z_idx * m_total + m_cur) * n_sub + s) * n_total + gid_n
                    zc = tl.load(z_corr_ptr + z_off, mask=mask_n, other=0.0)

                    sig0 = _triton_lv_bilinear_f64(
                        lv_tables_ptr, lv_base, t_start, lx + q_s, inv_dt_grid, y_min, inv_dy_grid, max_t_f, max_y_f, N_t, N_y, mask_n
                    )
                    a0 = -0.5 * sig0 * sig0
                    dw = sqrt_dt * zc
                    sigh = _triton_lv_bilinear_f64(
                        lv_tables_ptr, lv_base, t_end, lx + a0 * dt + sig0 * dw + q_s, inv_dt_grid, y_min, inv_dy_grid, max_t_f, max_y_f, N_t, N_y, mask_n
                    )
                    sigp = _triton_lv_bilinear_f64(
                        lv_tables_ptr, lv_base, t_end, lx + a0 * dt + sig0 * sqrt_dt + q_s, inv_dt_grid, y_min, inv_dy_grid, max_t_f, max_y_f, N_t, N_y, mask_n
                    )
                    sigm = _triton_lv_bilinear_f64(
                        lv_tables_ptr, lv_base, t_end, lx + a0 * dt - sig0 * sqrt_dt + q_s, inv_dt_grid, y_min, inv_dy_grid, max_t_f, max_y_f, N_t, N_y, mask_n
                    )
                    ah = -0.5 * sigh * sigh
                    lx = (
                        lx
                        + 0.5 * (a0 + ah) * dt
                        + 0.25 * (sigp + sigm + 2.0 * sig0) * dw
                        + 0.25 * (sigp - sigm) * sqrt_dt * (zc * zc - 1.0)
                    )

                max_lx = tl.max(tl.where(mask_n, lx, -1.0e300), axis=0)
                ex_v = tl.where(mask_n, tl.exp(lx - max_lx), 0.0)
                mean_rep = tl.maximum(tl.sum(ex_v, axis=0) / float(n_rep), 1.0e-300)
                lx = lx - (max_lx + tl.log(mean_rep))
                cur_off = (b * m_total + m_cur) * n_total + gid_n
                tl.store(out_norm_x_ptr + cur_off, tl.exp(lx), mask=mask_n)
                t_prev = t_cur
                rt_prev = rt_cur
                rn_prev = rn_cur
        else:
            has_prev = m_idx > 0
            prev_m = tl.maximum(m_idx - 1, 0)
            prev_off = (b * m_total + prev_m) * n_total + gid_n
            lx = tl.load(out_norm_x_ptr + prev_off, mask=mask_n & has_prev, other=0.0)

            t_prev = tl.where(has_prev, tl.load(tau_obs_ptr + tau_off + prev_m, mask=has_prev, other=0.0), 0.0)
            t_cur = tl.load(tau_obs_ptr + tau_off + m_idx)
            rt_prev = tl.where(has_prev, tl.load(r_tot_obs_ptr + r_off + prev_m, mask=has_prev, other=0.0), 0.0)
            rt_cur = tl.load(r_tot_obs_ptr + r_off + m_idx)
            rn_prev = tl.where(has_prev, tl.load(r_native_obs_ptr + r_off + prev_m, mask=has_prev, other=0.0), 0.0)
            rn_cur = tl.load(r_native_obs_ptr + r_off + m_idx)

            d_tau = tl.maximum(t_cur - t_prev, 1e-10)
            dt = d_tau / float(n_sub)
            sqrt_dt = tl.sqrt(dt)
            drt = rt_cur - rt_prev
            drn = rn_cur - rn_prev

            for s in tl.static_range(n_sub):
                alpha_0 = float(s) / float(n_sub)
                alpha_1 = float(s + 1) / float(n_sub)
                alpha_s = 0.5 * (alpha_0 + alpha_1)
                t_start = t_prev + alpha_0 * d_tau
                t_end = t_prev + alpha_1 * d_tau
                q_s = (rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn)

                if z_mode == 0:
                    z_off = (m_idx * n_sub + s) * n_total + gid_n
                else:
                    z_off = ((z_idx * m_total + m_idx) * n_sub + s) * n_total + gid_n
                zc = tl.load(z_corr_ptr + z_off, mask=mask_n, other=0.0)

                sig0 = _triton_lv_bilinear_f64(
                    lv_tables_ptr, lv_base, t_start, lx + q_s, inv_dt_grid, y_min, inv_dy_grid, max_t_f, max_y_f, N_t, N_y, mask_n
                )
                a0 = -0.5 * sig0 * sig0
                dw = sqrt_dt * zc
                sigh = _triton_lv_bilinear_f64(
                    lv_tables_ptr, lv_base, t_end, lx + a0 * dt + sig0 * dw + q_s, inv_dt_grid, y_min, inv_dy_grid, max_t_f, max_y_f, N_t, N_y, mask_n
                )
                sigp = _triton_lv_bilinear_f64(
                    lv_tables_ptr, lv_base, t_end, lx + a0 * dt + sig0 * sqrt_dt + q_s, inv_dt_grid, y_min, inv_dy_grid, max_t_f, max_y_f, N_t, N_y, mask_n
                )
                sigm = _triton_lv_bilinear_f64(
                    lv_tables_ptr, lv_base, t_end, lx + a0 * dt - sig0 * sqrt_dt + q_s, inv_dt_grid, y_min, inv_dy_grid, max_t_f, max_y_f, N_t, N_y, mask_n
                )
                ah = -0.5 * sigh * sigh
                lx = (
                    lx
                    + 0.5 * (a0 + ah) * dt
                    + 0.25 * (sigp + sigm + 2.0 * sig0) * dw
                    + 0.25 * (sigp - sigm) * sqrt_dt * (zc * zc - 1.0)
                )

            cur_off = (b * m_total + m_idx) * n_total + gid_n
            tl.store(out_norm_x_ptr + cur_off, lx, mask=mask_n)
            max_lx = tl.max(tl.where(mask_n, lx, -1.0e300), axis=0)
            ex_v = tl.where(mask_n, tl.exp(lx - max_lx), 0.0)
            tl.store(tile_max_ptr + (pid_br * n_tiles + pid_n), max_lx)
            tl.store(tile_ex_sum_ptr + (pid_br * n_tiles + pid_n), tl.sum(ex_v, axis=0))

    @triton.jit  # type: ignore[untyped-decorator]
    def _crn_step_1d_f32_kernel(  # type: ignore[no-untyped-def]
        lv_tables_ptr,
        z_corr_ptr,
        tau_obs_ptr,
        r_native_obs_ptr,
        r_tot_obs_ptr,
        meta_lv_ptr,
        meta_r_ptr,
        meta_z_ptr,
        meta_tset_ptr,
        y_min_vec_ptr,
        inv_dy_vec_ptr,
        inv_dt_vec_ptr,
        fwd_perf_ptr,
        out_norm_x_ptr,
        b_sz,
        m_total,
        d_z_stride,
        cg_th_flags_ptr=None,
        z_mode: tl.constexpr = 2,
        n_rep: tl.constexpr = 1024,
        n_sub: tl.constexpr = 4,
        N_t: tl.constexpr = 32,
        N_y: tl.constexpr = 64,
        BLOCK_N: tl.constexpr = 1024,
        Z_LAYOUT_MDN: tl.constexpr = False,
        HAS_FWD_SCALE: tl.constexpr = False,
        OUT_MDN: tl.constexpr = False,
        OUT_F32: tl.constexpr = False,
        USE_TH_RT_FLAGS: tl.constexpr = False,
    ):
        pid_br = tl.program_id(1)
        n_replicates = tl.num_programs(1) // b_sz
        b = pid_br // n_replicates
        r = pid_br % n_replicates
        n_total = n_replicates * n_rep

        offs_rep = tl.arange(0, BLOCK_N)
        mask_n = offs_rep < n_rep
        gid_n = r * n_rep + offs_rep

        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4
        if z_mode == 2:
            lv_idx = tl.load(meta_lv_ptr + b)
            r_idx = tl.load(meta_r_ptr + b)
            z_idx = tl.load(meta_z_ptr + b)
        else:
            lv_idx = 0 if z_mode == 1 else b
            r_idx = 0
            z_idx = b if z_mode == 1 else 0
        if USE_TH_RT_FLAGS:
            if tl.load(cg_th_flags_ptr + 16) == 0:
                return
            if lv_idx >= tl.load(cg_th_flags_ptr + 0):
                return
            m_loop_n = tl.load(cg_th_flags_ptr + 1)
            k_cross_rt = tl.load(cg_th_flags_ptr + 23)
        else:
            m_loop_n = m_total
            k_cross_rt = 0
        t_set = tl.load(meta_tset_ptr + b) if meta_tset_ptr is not None else 0
        y_idx = r_idx if z_mode == 2 else lv_idx
        y_min = tl.load(y_min_vec_ptr + y_idx).to(tl.float32)
        inv_dy_grid = tl.load(inv_dy_vec_ptr + y_idx).to(tl.float32)
        inv_dt_grid = tl.load(inv_dt_vec_ptr + t_set).to(tl.float32)
        lv_base = lv_idx * (N_t * N_y)
        r_off = r_idx * m_total
        tau_off = t_set * m_total

        lx = tl.zeros((BLOCK_N,), dtype=tl.float32)
        t_prev = tl.zeros((), dtype=tl.float32)
        rt_prev = tl.zeros((), dtype=tl.float32)
        rn_prev = tl.zeros((), dtype=tl.float32)
        for m_cur in range(m_loop_n):
            t_cur = tl.load(tau_obs_ptr + tau_off + m_cur).to(tl.float32)
            rt_cur = tl.load(r_tot_obs_ptr + r_off + m_cur).to(tl.float32)
            rn_cur = tl.load(r_native_obs_ptr + r_off + m_cur).to(tl.float32)
            d_tau = tl.maximum(t_cur - t_prev, 1e-7)
            dt = d_tau / float(n_sub)
            sqrt_dt = tl.sqrt(dt)
            drt = rt_cur - rt_prev
            drn = rn_cur - rn_prev

            for s in tl.static_range(n_sub):
                alpha_0 = float(s) / float(n_sub)
                alpha_s = (float(s) + 0.5) / float(n_sub)
                t_start = t_prev + alpha_0 * d_tau
                q_s = (rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn)

                m_z = m_cur + k_cross_rt
                if z_mode == 0:
                    z_off = (m_z * n_sub + s) * n_total + gid_n
                elif Z_LAYOUT_MDN:
                    z_off = ((m_z * n_sub + s) * d_z_stride + z_idx) * n_total + gid_n
                else:
                    z_off = ((z_idx * m_total + m_z) * n_sub + s) * n_total + gid_n
                zc = tl.load(z_corr_ptr + z_off, mask=mask_n, other=0.0)

                lx, _ = _triton_platen_fwd_substep_f32(
                    lv_tables_ptr, lv_base, lx, t_start, dt, sqrt_dt, q_s, zc,
                    inv_dt_grid, y_min, inv_dy_grid, max_t_f, max_y_f, N_t, N_y, mask_n,
                )

            max_lx = tl.max(tl.where(mask_n, lx, -1.0e30), axis=0)
            ex_v = tl.where(mask_n, tl.exp(lx - max_lx), 0.0)
            mean_rep = tl.maximum(tl.sum(ex_v, axis=0) / float(n_rep), 1.0e-30)
            lx = lx - (max_lx + tl.log(mean_rep))
            ex_norm = ex_v / mean_rep
            if HAS_FWD_SCALE:
                fp_bm = tl.load(fwd_perf_ptr + b * m_total + m_cur).to(tl.float32)
                ex_norm = ex_norm * fp_bm
            if OUT_MDN:
                cur_off = (m_cur * b_sz + b) * n_total + gid_n
            else:
                cur_off = (b * m_total + m_cur) * n_total + gid_n
            if OUT_F32:
                tl.store(out_norm_x_ptr + cur_off, ex_norm, mask=mask_n)
            else:
                tl.store(out_norm_x_ptr + cur_off, ex_norm.to(tl.float64), mask=mask_n)
            t_prev = t_cur
            rt_prev = rt_cur
            rn_prev = rn_cur

    @triton.jit  # type: ignore[untyped-decorator]
    def _eval_index_tile_helper(  # type: ignore[no-untyped-def]
        perf_base_ptr,
        m_off,
        pa,
        pb,
        aa,
        bb,
        has_a,
        has_b,
        scale_row_ptr,
        has_sc,
        ia,
        ib,
        offs_n,
        mask_n,
        d_assets,
        n_total,
        CODE: tl.constexpr,
        DTYPE: tl.constexpr = tl.float64,
    ):
        tiny_d = tl.zeros((), dtype=DTYPE) + 1e-12
        if CODE == 4 or CODE == 5:
            base_ia = tl.load(perf_base_ptr + (m_off + ia) * n_total + offs_n, mask=mask_n, other=1.0).to(DTYPE)
            va = tl.where(has_a & (aa == ia), pa.to(DTYPE), tl.where(has_b & (bb == ia), pb.to(DTYPE), base_ia))
            if scale_row_ptr is not None:
                sca = tl.load(scale_row_ptr + m_off + ia, mask=has_sc, other=1.0).to(DTYPE)
                va = tl.where(has_sc, va * sca, va)
            if d_assets < 2:
                return va
            base_ib = tl.load(perf_base_ptr + (m_off + ib) * n_total + offs_n, mask=mask_n, other=1.0).to(DTYPE)
            vb = tl.where(has_a & (aa == ib), pa.to(DTYPE), tl.where(has_b & (bb == ib), pb.to(DTYPE), base_ib))
            if scale_row_ptr is not None:
                scb = tl.load(scale_row_ptr + m_off + ib, mask=has_sc, other=1.0).to(DTYPE)
                vb = tl.where(has_sc, vb * scb, vb)
            return va / tl.maximum(vb, tiny_d) if CODE == 4 else vb / tl.maximum(va, tiny_d)
        if CODE == 9 or CODE == 10:
            if d_assets < 2:
                return tl.zeros(offs_n.shape, dtype=DTYPE)
            base_ia = tl.load(perf_base_ptr + (m_off + ia) * n_total + offs_n, mask=mask_n, other=1.0).to(DTYPE)
            base_ib = tl.load(perf_base_ptr + (m_off + ib) * n_total + offs_n, mask=mask_n, other=1.0).to(DTYPE)
            va = tl.where(has_a & (aa == ia), pa.to(DTYPE), tl.where(has_b & (bb == ia), pb.to(DTYPE), base_ia))
            vb = tl.where(has_a & (aa == ib), pa.to(DTYPE), tl.where(has_b & (bb == ib), pb.to(DTYPE), base_ib))
            if scale_row_ptr is not None:
                sca = tl.load(scale_row_ptr + m_off + ia, mask=has_sc, other=1.0).to(DTYPE)
                scb = tl.load(scale_row_ptr + m_off + ib, mask=has_sc, other=1.0).to(DTYPE)
                va = tl.where(has_sc, va * sca, va)
                vb = tl.where(has_sc, vb * scb, vb)
            return (va - vb) if CODE == 9 else (vb - va)

        need_min1_v: tl.constexpr = (CODE == 0 or CODE == 6 or CODE == 8 or CODE == 11 or CODE == 13 or CODE == 14)
        need_min2_v: tl.constexpr = (CODE == 15)
        need_min3_v: tl.constexpr = (CODE == 16)
        need_max1_v: tl.constexpr = (CODE == 1 or CODE == 6 or CODE == 7 or CODE == 11 or CODE == 12)
        need_max2_v: tl.constexpr = (CODE == 17)
        need_sum1_v: tl.constexpr = (CODE == 2 or CODE == 3 or CODE == 7 or CODE == 8 or CODE == 12 or CODE == 13)
        need_sum2_v: tl.constexpr = (CODE == 3)

        big_val: tl.constexpr = 1e30 if DTYPE == tl.float32 else 1e300
        min1 = tl.full(offs_n.shape, big_val, dtype=DTYPE)
        min2 = tl.full(offs_n.shape, big_val, dtype=DTYPE)
        min3 = tl.full(offs_n.shape, big_val, dtype=DTYPE)
        max1 = tl.full(offs_n.shape, -big_val, dtype=DTYPE)
        max2 = tl.full(offs_n.shape, -big_val, dtype=DTYPE)
        s1 = tl.zeros(offs_n.shape, dtype=DTYPE)
        s2 = tl.zeros(offs_n.shape, dtype=DTYPE)

        for d in range(d_assets):
            p_raw = tl.load(perf_base_ptr + (m_off + d) * n_total + offs_n, mask=mask_n, other=1.0).to(DTYPE)
            p_d = tl.where(has_a & (d == aa), pa.to(DTYPE), tl.where(has_b & (d == bb), pb.to(DTYPE), p_raw))
            if scale_row_ptr is not None:
                sc_d = tl.load(scale_row_ptr + m_off + d, mask=has_sc, other=1.0).to(DTYPE)
                p_d = tl.where(has_sc, p_d * sc_d, p_d)
            if need_min1_v:
                min1 = tl.minimum(min1, p_d)
            elif need_min2_v:
                lt1 = p_d < min1
                lt2 = (~lt1) & (p_d < min2)
                min2 = tl.where(lt1, min1, tl.where(lt2, p_d, min2))
                min1 = tl.where(lt1, p_d, min1)
            elif need_min3_v:
                lt1 = p_d < min1
                lt2 = (~lt1) & (p_d < min2)
                lt3 = (~lt1) & (~lt2) & (p_d < min3)
                min3 = tl.where(lt1, min2, tl.where(lt2, min2, tl.where(lt3, p_d, min3)))
                min2 = tl.where(lt1, min1, tl.where(lt2, p_d, min2))
                min1 = tl.where(lt1, p_d, min1)
            if need_max1_v:
                max1 = tl.maximum(max1, p_d)
            elif need_max2_v:
                gt1 = p_d > max1
                gt2 = (~gt1) & (p_d > max2)
                max2 = tl.where(gt1, max1, tl.where(gt2, p_d, max2))
                max1 = tl.where(gt1, p_d, max1)
            if need_sum1_v:
                s1 = s1 + p_d
            if need_sum2_v:
                s2 = s2 + p_d * p_d

        inv_d = 1.0 / (tl.zeros((), dtype=DTYPE) + d_assets)
        w_avg = s1 * inv_d
        if CODE == 0:
            return min1
        if CODE == 1:
            return max1
        if CODE == 2:
            return w_avg
        if CODE == 3:
            return tl.where(d_assets <= 1, tl.zeros(offs_n.shape, dtype=DTYPE), tl.sqrt(tl.maximum(s2 * inv_d - w_avg * w_avg, 0.0)))
        if CODE == 6:
            return max1 / tl.maximum(min1, tiny_d)
        if CODE == 7:
            return max1 / tl.maximum(w_avg, tiny_d)
        if CODE == 8:
            return w_avg / tl.maximum(min1, tiny_d)
        if CODE == 11:
            return max1 - min1
        if CODE == 12:
            return max1 - w_avg
        if CODE == 13:
            return w_avg - min1
        if CODE == 14:
            return tl.where(min1 < 1.0, 2.0 - min1, min1)
        if CODE == 15:
            return tl.where(d_assets > 1, min2, min1)
        if CODE == 16:
            return tl.where(d_assets >= 3, min3, tl.where(d_assets == 2, min2, min1))
        if CODE == 17:
            return tl.where(d_assets > 1, max2, max1)
        return min1

    @triton.jit  # type: ignore[untyped-decorator]
    def _eval_index_tile_012_helper(  # type: ignore[no-untyped-def]
        perf_base_ptr,
        m_off,
        pa,
        pb,
        aa,
        bb,
        has_a,
        has_b,
        scale_row_ptr,
        has_sc,
        offs_n,
        mask_n,
        d_act,
        n_total,
        D_PAD: tl.constexpr = 8,
        DTYPE: tl.constexpr = tl.float64,
    ):
        big_v: tl.constexpr = 1e30 if DTYPE == tl.float32 else 1e300
        min1 = tl.full(offs_n.shape, big_v, dtype=DTYPE)
        max1 = tl.full(offs_n.shape, -big_v, dtype=DTYPE)
        s1 = tl.zeros(offs_n.shape, dtype=DTYPE)
        for d in tl.static_range(D_PAD):
            if d < d_act:
                p_raw = tl.load(perf_base_ptr + (m_off + d) * n_total + offs_n, mask=mask_n, other=1.0).to(DTYPE)
                p_d = tl.where(has_a & (d == aa), pa.to(DTYPE), tl.where(has_b & (d == bb), pb.to(DTYPE), p_raw))
                if scale_row_ptr is not None:
                    sc_d = tl.load(scale_row_ptr + m_off + d, mask=has_sc, other=1.0).to(DTYPE)
                    p_d = tl.where(has_sc, p_d * sc_d, p_d)
                min1 = tl.minimum(min1, p_d)
                max1 = tl.maximum(max1, p_d)
                s1 = s1 + p_d
        inv_d = 1.0 / tl.maximum(tl.zeros((), dtype=DTYPE) + d_act, 1.0)
        return min1, max1, s1 * inv_d

    @triton.jit
    def _triton_c2_s5(t):  # type: ignore[no-untyped-def]
        u = tl.minimum(tl.maximum(t, 0.0), 1.0)
        return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))

    @triton.jit
    def _triton_c2_ds5(t):  # type: ignore[no-untyped-def]
        u = tl.minimum(tl.maximum(t, 0.0), 1.0)
        in_rng = (t > 0.0) & (t < 1.0)
        return tl.where(in_rng, 30.0 * u * u * (1.0 - u) * (1.0 - u), 0.0)

    @triton.jit  # type: ignore[untyped-decorator]
    def _crn_loo_l2o_payoff_f64_kernel(  # type: ignore[no-untyped-def]
        perf_base_ptr,
        alt_perf_ptr,
        bump_perf_a_ptr,
        bump_perf_b_ptr,
        scale_mat_ptr,
        bump_asset_a_ptr,
        bump_asset_b_ptr,
        a_slot_ptr,
        b_slot_ptr,
        s_slot_ptr,
        df_pay_mat_ptr,
        w_fund_mat_ptr,
        df_row_idx_ptr,
        wf_row_idx_ptr,
        pv_out_ptr,
        ko_barriers_ptr,
        is_callable_ptr,
        cpn_barriers_ptr,
        is_guar_ptr,
        cpn_rates_ptr,
        inv_lex_perm_ptr,
        eps_vec_ptr,
        frozen_term_ptr,
        i_ki_ext_ptr,
        scalar_f64_ptr,
        n_bumps,
        d_assets,
        m_rem,
        n_total,
        tile_start,
        cg_flags_ptr=None,
        CALL_CODE: tl.constexpr = 0,
        CPN_CODE: tl.constexpr = 0,
        OPT_CODE: tl.constexpr = 0,
        HAS_KI: tl.constexpr = False,
        IS_MEM: tl.constexpr = False,
        IS_LONG: tl.constexpr = False,
        NO_NOTIONAL: tl.constexpr = False,
        FORCE_CPN_AC: tl.constexpr = False,
        CPN_DIR_DOWN: tl.constexpr = False,
        CPN_TYPE_OUT: tl.constexpr = False,
        N_TILE: tl.constexpr = 128,
        BLOCK_N: tl.constexpr = 128,
        IS_FULL_BATCH: tl.constexpr = False,
        HAS_ONE_STAR: tl.constexpr = False,
        HAS_LOWER_STRIKE: tl.constexpr = False,
        HAS_PART_CALL: tl.constexpr = False,
        USE_C2_PRETERM: tl.constexpr = False,
        USE_FROZEN_TERM: tl.constexpr = False,
        HAS_EXT_KI: tl.constexpr = False,
        USE_F32_EVAL: tl.constexpr = False,
        USE_RT_FLAGS: tl.constexpr = False,
        D_PAD: tl.constexpr = 0,
        CG_BUMPS_IDX: tl.constexpr = -1,
    ):
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)
        n_t_tiles = tl.num_programs(1)
        if USE_RT_FLAGS:
            eff_nb = tl.load(cg_flags_ptr + CG_BUMPS_IDX) if CG_BUMPS_IDX >= 0 else n_bumps
            if pid_b >= eff_nb:
                return
            d_act = tl.load(cg_flags_ptr + 0)
            m_act = tl.load(cg_flags_ptr + 1)
            rt_call = tl.load(cg_flags_ptr + 2)
            rt_cpn = tl.load(cg_flags_ptr + 3)
            rt_opt = tl.load(cg_flags_ptr + 4)
            f_has_ki = tl.load(cg_flags_ptr + 5) != 0
            f_has_os = tl.load(cg_flags_ptr + 7) != 0
            f_is_mem = tl.load(cg_flags_ptr + 8) != 0
            f_is_long = tl.load(cg_flags_ptr + 9) != 0
            f_no_not = tl.load(cg_flags_ptr + 10) != 0
            f_frc_cpn = tl.load(cg_flags_ptr + 11) != 0
            f_cpn_dn = tl.load(cg_flags_ptr + 12) != 0
            f_cpn_out = tl.load(cg_flags_ptr + 13) != 0
            f_has_low = tl.load(cg_flags_ptr + 14) != 0
            f_has_pc = tl.load(cg_flags_ptr + 15) != 0
        else:
            if pid_b >= n_bumps:
                return
            d_act = d_assets
            m_act = m_rem
            rt_call = CALL_CODE
            rt_cpn = CPN_CODE
            rt_opt = OPT_CODE
            f_has_ki = HAS_KI
            f_has_os = HAS_ONE_STAR
            f_is_mem = IS_MEM
            f_is_long = IS_LONG
            f_no_not = NO_NOTIONAL
            f_frc_cpn = FORCE_CPN_AC
            f_cpn_dn = CPN_DIR_DOWN
            f_cpn_out = CPN_TYPE_OUT
            f_has_low = HAS_LOWER_STRIKE
            f_has_pc = HAS_PART_CALL
        EVAL_DTYPE: tl.constexpr = tl.float32 if USE_F32_EVAL else tl.float64
        tiny_e = tl.zeros((), dtype=EVAL_DTYPE) + 1e-12
        ko_spread = tl.load(scalar_f64_ptr + 0).to(EVAL_DTYPE)
        cpn_spread = tl.load(scalar_f64_ptr + 1).to(EVAL_DTYPE)
        ki_spread = tl.load(scalar_f64_ptr + 2).to(EVAL_DTYPE)
        one_star_strike = tl.load(scalar_f64_ptr + 3).to(EVAL_DTYPE)
        one_star_eps = tl.load(scalar_f64_ptr + 4).to(EVAL_DTYPE)
        strike = tl.load(scalar_f64_ptr + 5).to(EVAL_DTYPE)
        lower_strike = tl.load(scalar_f64_ptr + 6).to(EVAL_DTYPE)
        ki_barrier = tl.load(scalar_f64_ptr + 7).to(EVAL_DTYPE)
        option_gearing = tl.load(scalar_f64_ptr + 8).to(EVAL_DTYPE)
        ac_cpn = tl.load(scalar_f64_ptr + 9).to(EVAL_DTYPE)
        c_mem_0 = tl.load(scalar_f64_ptr + 10).to(EVAL_DTYPE)
        i_ki_past = tl.load(scalar_f64_ptr + 11).to(EVAL_DTYPE)
        part_call_strike = tl.load(scalar_f64_ptr + 12).to(EVAL_DTYPE)
        part_call_rate = tl.load(scalar_f64_ptr + 13).to(EVAL_DTYPE)
        one_star_coupon = tl.load(scalar_f64_ptr + 14).to(EVAL_DTYPE)
        eff_tile_start = tile_start + pid_t * N_TILE
        df_row = tl.load(df_row_idx_ptr + pid_b)
        wf_row = tl.load(wf_row_idx_ptr + pid_b)
        if IS_FULL_BATCH:
            p_src_ptr = bump_perf_a_ptr + pid_b * m_rem * d_assets * n_total
            aa = -1
            bb = -1
            a_slot = 0
            b_slot = 0
            has_a = tl.full((), False, tl.int1)
            has_b = tl.full((), False, tl.int1)
            scale_row_ptr = scale_mat_ptr
            has_sc = tl.full((), False, tl.int1)
        else:
            aa_raw = tl.load(bump_asset_a_ptr + pid_b)
            bb_raw = tl.load(bump_asset_b_ptr + pid_b)
            a_slot = tl.load(a_slot_ptr + pid_b)
            b_slot = tl.load(b_slot_ptr + pid_b)
            is_alt = aa_raw == -2
            p_src_ptr = tl.where(
                is_alt,
                alt_perf_ptr + a_slot.to(tl.int64) * (m_rem * d_assets * n_total),
                perf_base_ptr,
            )
            aa = tl.where(is_alt, -1, aa_raw)
            bb = tl.where(is_alt | (bb_raw == aa_raw), -1, bb_raw)
            has_a = aa >= 0
            has_b = bb >= 0
            if scale_mat_ptr is not None:
                s_slot = tl.load(s_slot_ptr + pid_b)
                has_sc = s_slot >= 0
                scale_row_ptr = tl.where(has_sc, scale_mat_ptr + s_slot.to(tl.int64) * (m_rem * d_assets), scale_mat_ptr)
            else:
                has_sc = tl.full((), False, tl.int1)
                scale_row_ptr = scale_mat_ptr

        ia = tl.load(inv_lex_perm_ptr + 0).to(tl.int32)
        ib_idx = tl.where(d_act > 1, 1, 0)
        ib = tl.load(inv_lex_perm_ptr + ib_idx).to(tl.int32)
        eff_opt_rt = tl.where(f_is_long, 2, rt_opt) if USE_RT_FLAGS else (2 if IS_LONG else OPT_CODE)

        hard_ko = tl.abs(ko_spread) <= 1e-6
        eff_cpn_sp = tl.where(f_cpn_dn, -cpn_spread, cpn_spread) if USE_RT_FLAGS else (-cpn_spread if CPN_DIR_DOWN else cpn_spread)
        hard_cpn = tl.abs(eff_cpn_sp) <= 1e-6
        hard_ki = tl.abs(ki_spread) <= 1e-6
        blo_ki = tl.minimum(ki_barrier, ki_barrier + ki_spread)
        bhi_ki = tl.maximum(ki_barrier, ki_barrier + ki_spread)
        inv_w_ki = 1.0 / tl.maximum(bhi_ki - blo_ki, tiny_e)
        inv_w_ko = 1.0 / tl.maximum(tl.abs(ko_spread), tiny_e)
        inv_w_cp = 1.0 / tl.maximum(tl.abs(eff_cpn_sp), tiny_e)
        inv_os_eps = 1.0 / tl.maximum(one_star_eps, tiny_e)
        has_one_star = (f_has_os & (one_star_strike > 0.0)) if USE_RT_FLAGS else (one_star_strike > 0.0)
        has_part_call = (f_has_pc & (part_call_rate != 0.0)) if USE_RT_FLAGS else ((part_call_rate != 0.0) if HAS_PART_CALL else False)
        inv_N = 1.0 / tl.maximum(tl.zeros((), dtype=tl.float64) + n_total, 1.0)

        n_subtiles: tl.constexpr = N_TILE // BLOCK_N
        tile_pv_acc = tl.zeros((), dtype=tl.float64)
        zero_vec = tl.zeros((BLOCK_N,), dtype=EVAL_DTYPE)
        one_vec = tl.full((BLOCK_N,), 1.0, dtype=EVAL_DTYPE)
        zero_vec_f64 = tl.zeros((BLOCK_N,), dtype=tl.float64)
        need_tail_opt: tl.constexpr = ((2 if IS_LONG else OPT_CODE) != CALL_CODE) and ((2 if IS_LONG else OPT_CODE) != CPN_CODE)
        need_tail_best: tl.constexpr = HAS_ONE_STAR and ((2 if IS_LONG else OPT_CODE) != 1) and (CALL_CODE != 1)
        m_pre_limit_rt = (m_act - 1) if USE_FROZEN_TERM else m_act
        m_pre_loop: tl.constexpr = m_rem if USE_RT_FLAGS else (m_rem - 1 if USE_FROZEN_TERM else m_rem)

        for st in tl.static_range(n_subtiles):
            sub_start = eff_tile_start + st * BLOCK_N
            offs_n = sub_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_total

            surv_prev = tl.where(mask_n, one_vec, zero_vec)
            c_mem = tl.full((BLOCK_N,), c_mem_0, dtype=EVAL_DTYPE)
            pv_path = zero_vec
            last_wc = zero_vec
            last_wcp = zero_vec
            last_m1_rt = zero_vec
            last_x1_rt = zero_vec
            last_wa_rt = zero_vec

            for m in range(m_pre_loop):
                m_ok = (m < m_pre_limit_rt) if USE_RT_FLAGS else True
                if m_ok & (tl.max(surv_prev, axis=0) > 1e-24):
                    m_off = m * d_assets
                    if (not IS_FULL_BATCH) and (bump_perf_a_ptr is not None):
                        pa = tl.load(
                            bump_perf_a_ptr + (a_slot * m_rem + m) * n_total + offs_n,
                            mask=mask_n & has_a,
                            other=1.0,
                        ).to(EVAL_DTYPE)
                    else:
                        pa = zero_vec
                    if (not IS_FULL_BATCH) and (bump_perf_b_ptr is not None):
                        pb = tl.load(
                            bump_perf_b_ptr + (b_slot * m_rem + m) * n_total + offs_n,
                            mask=mask_n & has_b,
                            other=1.0,
                        ).to(EVAL_DTYPE)
                    else:
                        pb = zero_vec

                    is_call_m = tl.load(is_callable_ptr + m) > 0.5
                    is_guar_m = tl.load(is_guar_ptr + m) > 0.5

                    if USE_RT_FLAGS:
                        m1_t, x1_t, wa_t = _eval_index_tile_012_helper(
                            p_src_ptr, m_off, pa, pb, aa, bb, has_a, has_b, scale_row_ptr, has_sc, offs_n, mask_n, d_act, n_total, D_PAD, EVAL_DTYPE
                        )
                        w_call = tl.where(rt_call == 0, m1_t, tl.where(rt_call == 1, x1_t, wa_t))
                        w_cpn = tl.where(rt_cpn == 0, m1_t, tl.where(rt_cpn == 1, x1_t, wa_t))
                        is_last_rt = m == (m_act - 1)
                        last_m1_rt = tl.where(is_last_rt, m1_t, last_m1_rt)
                        last_x1_rt = tl.where(is_last_rt, x1_t, last_x1_rt)
                        last_wa_rt = tl.where(is_last_rt, wa_t, last_wa_rt)
                    else:
                        w_call = _eval_index_tile_helper(
                            p_src_ptr, m_off, pa, pb, aa, bb, has_a, has_b, scale_row_ptr, has_sc, ia, ib, offs_n, mask_n, d_assets, n_total, CALL_CODE, EVAL_DTYPE
                        )
                        if CPN_CODE == CALL_CODE:
                            w_cpn = w_call
                        else:
                            w_cpn = _eval_index_tile_helper(
                                p_src_ptr, m_off, pa, pb, aa, bb, has_a, has_b, scale_row_ptr, has_sc, ia, ib, offs_n, mask_n, d_assets, n_total, CPN_CODE, EVAL_DTYPE
                            )

                    b_ko = tl.load(ko_barriers_ptr + m).to(EVAL_DTYPE)
                    b_cp = tl.load(cpn_barriers_ptr + m).to(EVAL_DTYPE)
                    if USE_C2_PRETERM and USE_FROZEN_TERM:
                        eps_m = tl.load(eps_vec_ptr + m).to(EVAL_DTYPE) if eps_vec_ptr is not None else (tl.zeros((), dtype=EVAL_DTYPE) + 0.02)
                        lo_ko_e = tl.where(hard_ko, b_ko - 0.5 * eps_m, tl.minimum(b_ko, b_ko + ko_spread))
                        hi_ko_e = tl.where(hard_ko, b_ko + 0.5 * eps_m, tl.maximum(b_ko, b_ko + ko_spread))
                        inv_ko_e = 1.0 / tl.maximum(hi_ko_e - lo_ko_e, tiny_e)
                        g_ko_raw = _triton_c2_s5((w_call - lo_ko_e) * inv_ko_e)
                        g_ko = tl.where(is_call_m, g_ko_raw, zero_vec)

                        lo_cp_e = tl.where(hard_cpn, b_cp - 0.5 * eps_m, tl.minimum(b_cp, b_cp + eff_cpn_sp))
                        hi_cp_e = tl.where(hard_cpn, b_cp + 0.5 * eps_m, tl.maximum(b_cp, b_cp + eff_cpn_sp))
                        inv_cp_e = 1.0 / tl.maximum(hi_cp_e - lo_cp_e, tiny_e)
                        ramp_c2 = _triton_c2_s5((w_cpn - lo_cp_e) * inv_cp_e)
                        if USE_RT_FLAGS:
                            hit_c2 = tl.where(f_cpn_dn, 1.0 - ramp_c2, ramp_c2)
                            ind_c2 = tl.where(f_cpn_out, 1.0 - hit_c2, hit_c2)
                        else:
                            hit_c2 = (1.0 - ramp_c2) if CPN_DIR_DOWN else ramp_c2
                            ind_c2 = (1.0 - hit_c2) if CPN_TYPE_OUT else hit_c2
                        g_cpn = tl.where(is_guar_m, one_vec, ind_c2)
                    elif USE_C2_PRETERM:
                        if m < m_act - 1:
                            eps_m = tl.load(eps_vec_ptr + m).to(EVAL_DTYPE) if eps_vec_ptr is not None else (tl.zeros((), dtype=EVAL_DTYPE) + 0.02)
                            lo_ko_e = tl.where(hard_ko, b_ko - 0.5 * eps_m, tl.minimum(b_ko, b_ko + ko_spread))
                            hi_ko_e = tl.where(hard_ko, b_ko + 0.5 * eps_m, tl.maximum(b_ko, b_ko + ko_spread))
                            inv_ko_e = 1.0 / tl.maximum(hi_ko_e - lo_ko_e, tiny_e)
                            g_ko_raw = _triton_c2_s5((w_call - lo_ko_e) * inv_ko_e)
                            g_ko = tl.where(is_call_m, g_ko_raw, zero_vec)

                            lo_cp_e = tl.where(hard_cpn, b_cp - 0.5 * eps_m, tl.minimum(b_cp, b_cp + eff_cpn_sp))
                            hi_cp_e = tl.where(hard_cpn, b_cp + 0.5 * eps_m, tl.maximum(b_cp, b_cp + eff_cpn_sp))
                            inv_cp_e = 1.0 / tl.maximum(hi_cp_e - lo_cp_e, tiny_e)
                            ramp_c2 = _triton_c2_s5((w_cpn - lo_cp_e) * inv_cp_e)
                            if USE_RT_FLAGS:
                                hit_c2 = tl.where(f_cpn_dn, 1.0 - ramp_c2, ramp_c2)
                                ind_c2 = tl.where(f_cpn_out, 1.0 - hit_c2, hit_c2)
                            else:
                                hit_c2 = (1.0 - ramp_c2) if CPN_DIR_DOWN else ramp_c2
                                ind_c2 = (1.0 - hit_c2) if CPN_TYPE_OUT else hit_c2
                            g_cpn = tl.where(is_guar_m, one_vec, ind_c2)
                        else:
                            blo_ko = tl.minimum(b_ko, b_ko + ko_spread)
                            g_ko_ramp = tl.minimum(tl.maximum((w_call - blo_ko) * inv_w_ko, zero_vec), one_vec)
                            g_ko_raw = tl.where(hard_ko, tl.where(w_call >= b_ko, one_vec, zero_vec), g_ko_ramp)
                            g_ko = tl.where(is_call_m, g_ko_raw, zero_vec)

                            if USE_RT_FLAGS:
                                hit_dir = tl.where(f_cpn_dn, w_cpn <= b_cp, w_cpn >= b_cp)
                                g_cpn_hard = tl.where(f_cpn_out, tl.where(hit_dir, zero_vec, one_vec), tl.where(hit_dir, one_vec, zero_vec))
                            else:
                                hit_dir = (w_cpn <= b_cp) if CPN_DIR_DOWN else (w_cpn >= b_cp)
                                g_cpn_hard = tl.where(hit_dir, zero_vec if CPN_TYPE_OUT else one_vec, one_vec if CPN_TYPE_OUT else zero_vec)
                            blo_cp = tl.minimum(b_cp, b_cp + eff_cpn_sp)
                            ramp_up = tl.minimum(tl.maximum((w_cpn - blo_cp) * inv_w_cp, zero_vec), one_vec)
                            if USE_RT_FLAGS:
                                hit_g = tl.where(f_cpn_dn, 1.0 - ramp_up, ramp_up)
                                g_cpn_soft = tl.where(f_cpn_out, 1.0 - hit_g, hit_g)
                            else:
                                hit_g = (1.0 - ramp_up) if CPN_DIR_DOWN else ramp_up
                                g_cpn_soft = (1.0 - hit_g) if CPN_TYPE_OUT else hit_g
                            g_cpn_raw = tl.where(hard_cpn, g_cpn_hard, g_cpn_soft)
                            g_cpn = tl.where(is_guar_m, one_vec, g_cpn_raw)
                    else:
                        blo_ko = tl.minimum(b_ko, b_ko + ko_spread)
                        g_ko_ramp = tl.minimum(tl.maximum((w_call - blo_ko) * inv_w_ko, zero_vec), one_vec)
                        g_ko_raw = tl.where(hard_ko, tl.where(w_call >= b_ko, one_vec, zero_vec), g_ko_ramp)
                        g_ko = tl.where(is_call_m, g_ko_raw, zero_vec)

                        if USE_RT_FLAGS:
                            hit_dir = tl.where(f_cpn_dn, w_cpn <= b_cp, w_cpn >= b_cp)
                            g_cpn_hard = tl.where(f_cpn_out, tl.where(hit_dir, zero_vec, one_vec), tl.where(hit_dir, one_vec, zero_vec))
                        else:
                            hit_dir = (w_cpn <= b_cp) if CPN_DIR_DOWN else (w_cpn >= b_cp)
                            g_cpn_hard = tl.where(hit_dir, zero_vec if CPN_TYPE_OUT else one_vec, one_vec if CPN_TYPE_OUT else zero_vec)
                        blo_cp = tl.minimum(b_cp, b_cp + eff_cpn_sp)
                        ramp_up = tl.minimum(tl.maximum((w_cpn - blo_cp) * inv_w_cp, zero_vec), one_vec)
                        if USE_RT_FLAGS:
                            hit_g = tl.where(f_cpn_dn, 1.0 - ramp_up, ramp_up)
                            g_cpn_soft = tl.where(f_cpn_out, 1.0 - hit_g, hit_g)
                        else:
                            hit_g = (1.0 - ramp_up) if CPN_DIR_DOWN else ramp_up
                            g_cpn_soft = (1.0 - hit_g) if CPN_TYPE_OUT else hit_g
                        g_cpn_raw = tl.where(hard_cpn, g_cpn_hard, g_cpn_soft)
                        g_cpn = tl.where(is_guar_m, one_vec, g_cpn_raw)

                    eff_g_cpn = (tl.where(f_frc_cpn, tl.maximum(g_cpn, g_ko), g_cpn) if USE_RT_FLAGS else (tl.maximum(g_cpn, g_ko) if FORCE_CPN_AC else g_cpn))
                    c_rate = tl.load(cpn_rates_ptr + m).to(EVAL_DTYPE)
                    p_m = surv_prev * g_ko
                    if USE_RT_FLAGS:
                        eff_c = tl.where(f_is_mem, c_rate + c_mem, zero_vec + c_rate)
                        cpn_path = surv_prev * eff_g_cpn * eff_c + p_m * ac_cpn
                        c_mem = tl.where(f_is_mem, surv_prev * eff_c * (1.0 - eff_g_cpn), c_mem)
                    elif IS_MEM:
                        eff_c = c_rate + c_mem
                        cpn_path = surv_prev * eff_g_cpn * eff_c + p_m * ac_cpn
                        c_mem = surv_prev * eff_c * (1.0 - eff_g_cpn)
                    else:
                        cpn_path = surv_prev * eff_g_cpn * c_rate + p_m * ac_cpn

                    df_m = tl.load(df_pay_mat_ptr + df_row * m_rem + m).to(EVAL_DTYPE)
                    if USE_RT_FLAGS:
                        pv_path = pv_path + df_m * tl.where(f_no_not, cpn_path, p_m + cpn_path)
                    elif NO_NOTIONAL:
                        pv_path = pv_path + df_m * cpn_path
                    else:
                        pv_path = pv_path + df_m * (p_m + cpn_path)
                    if w_fund_mat_ptr is not None:
                        wf_m = tl.load(w_fund_mat_ptr + wf_row * m_rem + m).to(EVAL_DTYPE)
                        pv_path = pv_path - wf_m * surv_prev

                    surv_prev = tl.maximum(surv_prev - p_m, zero_vec)
                    last_wc = w_call
                    last_wcp = w_cpn

            m_last = m_act - 1
            df_last = tl.load(df_pay_mat_ptr + df_row * m_rem + m_last).to(EVAL_DTYPE)
            if USE_FROZEN_TERM:
                if tl.max(surv_prev, axis=0) > 1e-24:
                    gk_f = tl.load(frozen_term_ptr + 0 * n_total + offs_n, mask=mask_n, other=0.0).to(EVAL_DTYPE)
                    egc_f = tl.load(frozen_term_ptr + 1 * n_total + offs_n, mask=mask_n, other=0.0).to(EVAL_DTYPE)
                    opt_f = tl.load(frozen_term_ptr + 2 * n_total + offs_n, mask=mask_n, other=0.0).to(EVAL_DTYPE)
                    c_last = tl.load(cpn_rates_ptr + m_last).to(EVAL_DTYPE)
                    eff_c_last = (tl.where(f_is_mem, c_last + c_mem, zero_vec + c_last) if USE_RT_FLAGS else ((c_last + c_mem) if IS_MEM else (zero_vec + c_last)))
                    pm_last = surv_prev * gk_f
                    cpn_last = surv_prev * egc_f * eff_c_last + pm_last * ac_cpn
                    surv_end = tl.maximum(surv_prev - pm_last, zero_vec)
                    if w_fund_mat_ptr is not None:
                        wf_last = tl.load(w_fund_mat_ptr + wf_row * m_rem + m_last).to(EVAL_DTYPE)
                        pv_path = pv_path - wf_last * surv_prev
                    term_pay = (tl.where(f_no_not, cpn_last + surv_end * opt_f, pm_last + cpn_last + surv_end * (1.0 + opt_f)) if USE_RT_FLAGS else ((pm_last + cpn_last + surv_end * (1.0 + opt_f)) if not NO_NOTIONAL else (cpn_last + surv_end * opt_f)))
                    pv_path = pv_path + df_last * term_pay
            else:
                if USE_RT_FLAGS:
                    last_avg = last_wa_rt
                    last_wo = tl.where(f_is_long, last_m1_rt, tl.where(eff_opt_rt == 0, last_m1_rt, tl.where(eff_opt_rt == 1, last_x1_rt, last_wa_rt)))
                    last_wb = tl.where(f_has_os, last_x1_rt, zero_vec)
                else:
                    if need_tail_opt or need_tail_best:
                        m_off_last = m_last * d_assets
                        if (not IS_FULL_BATCH) and (bump_perf_a_ptr is not None):
                            pa_last = tl.load(
                                bump_perf_a_ptr + (a_slot * m_rem + m_last) * n_total + offs_n,
                                mask=mask_n & has_a,
                                other=1.0,
                            ).to(EVAL_DTYPE)
                        else:
                            pa_last = zero_vec
                        if (not IS_FULL_BATCH) and (bump_perf_b_ptr is not None):
                            pb_last = tl.load(
                                bump_perf_b_ptr + (b_slot * m_rem + m_last) * n_total + offs_n,
                                mask=mask_n & has_b,
                                other=1.0,
                            ).to(EVAL_DTYPE)
                        else:
                            pb_last = zero_vec
                    else:
                        m_off_last = 0
                        pa_last = zero_vec
                        pb_last = zero_vec

                    if IS_LONG:
                        last_avg = last_wc if CALL_CODE == 2 else (
                            last_wcp if CPN_CODE == 2 else _eval_index_tile_helper(
                                p_src_ptr, m_off_last, pa_last, pb_last, aa, bb, has_a, has_b, scale_row_ptr, has_sc, ia, ib, offs_n, mask_n, d_assets, n_total, 2, EVAL_DTYPE
                            )
                        )
                        last_wo = last_wc if CALL_CODE == 0 else (
                            last_wcp if CPN_CODE == 0 else _eval_index_tile_helper(
                                p_src_ptr, m_off_last, pa_last, pb_last, aa, bb, has_a, has_b, scale_row_ptr, has_sc, ia, ib, offs_n, mask_n, d_assets, n_total, 0, EVAL_DTYPE
                            )
                        )
                    else:
                        last_avg = zero_vec
                        if OPT_CODE == CALL_CODE:
                            last_wo = last_wc
                        elif OPT_CODE == CPN_CODE:
                            last_wo = last_wcp
                        else:
                            last_wo = _eval_index_tile_helper(
                                p_src_ptr, m_off_last, pa_last, pb_last, aa, bb, has_a, has_b, scale_row_ptr, has_sc, ia, ib, offs_n, mask_n, d_assets, n_total, OPT_CODE, EVAL_DTYPE
                            )
                    if not HAS_ONE_STAR:
                        last_wb = zero_vec
                    elif (not IS_LONG) and (OPT_CODE == 1):
                        last_wb = last_wo
                    elif CALL_CODE == 1:
                        last_wb = last_wc
                    else:
                        last_wb = _eval_index_tile_helper(
                            p_src_ptr, m_off_last, pa_last, pb_last, aa, bb, has_a, has_b, scale_row_ptr, has_sc, ia, ib, offs_n, mask_n, d_assets, n_total, 1, EVAL_DTYPE
                        )

                if HAS_EXT_KI and (i_ki_ext_ptr is not None):
                    ki_gate = tl.load(i_ki_ext_ptr + offs_n, mask=mask_n, other=i_ki_past).to(EVAL_DTYPE)
                elif USE_RT_FLAGS:
                    b_ki_eff = tl.where(f_is_long, strike, ki_barrier)
                    blo_klb = tl.minimum(b_ki_eff, b_ki_eff + ki_spread)
                    bhi_klb = tl.maximum(b_ki_eff, b_ki_eff + ki_spread)
                    inv_wklb = 1.0 / tl.maximum(bhi_klb - blo_klb, tiny_e)
                    g_ki_hard = tl.where(last_wo < b_ki_eff, one_vec, zero_vec)
                    g_ki_soft = tl.minimum(tl.maximum((bhi_klb - last_wo) * inv_wklb, zero_vec), one_vec)
                    g_ki = tl.where(hard_ki, g_ki_hard, g_ki_soft)
                    ki_gate = tl.where(f_has_ki, tl.maximum(i_ki_past, g_ki), one_vec)
                elif HAS_KI:
                    if IS_LONG:
                        blo_klb = tl.minimum(strike, strike + ki_spread)
                        bhi_klb = tl.maximum(strike, strike + ki_spread)
                        inv_wklb = 1.0 / tl.maximum(bhi_klb - blo_klb, tiny_e)
                        g_ki_hard = tl.where(last_wo < strike, one_vec, zero_vec)
                        g_ki_soft = tl.minimum(tl.maximum((bhi_klb - last_wo) * inv_wklb, zero_vec), one_vec)
                    else:
                        g_ki_hard = tl.where(last_wo < ki_barrier, one_vec, zero_vec)
                        g_ki_soft = tl.minimum(tl.maximum((bhi_ki - last_wo) * inv_w_ki, zero_vec), one_vec)
                    g_ki = tl.where(hard_ki, g_ki_hard, g_ki_soft)
                    ki_gate = tl.maximum(i_ki_past, g_ki)
                else:
                    ki_gate = one_vec

                half_os = 0.5 * one_star_eps
                star_soft = tl.minimum(tl.maximum((last_wb - (one_star_strike - half_os)) * inv_os_eps, zero_vec), one_vec)
                star_hard = tl.where(last_wb >= one_star_strike, one_vec, zero_vec)
                star_ind = tl.where(has_one_star, tl.where(one_star_eps <= 1e-6, star_hard, star_soft), zero_vec)

                opt_val = zero_vec
                if USE_RT_FLAGS:
                    opt_long = opt_val - (1.0 - last_avg) * ki_gate * (1.0 - star_ind)
                    put_intr_low = tl.maximum(strike - last_wo, zero_vec) - tl.maximum(lower_strike - last_wo, zero_vec)
                    put_intr_plain = tl.maximum(strike - last_wo, zero_vec)
                    put_intr = tl.where(f_has_low, put_intr_low, put_intr_plain)
                    put_mask = tl.where(f_has_low, (strike > lower_strike) & (option_gearing > 0.0), option_gearing > 0.0)
                    opt_short = tl.where(put_mask, opt_val - option_gearing * ki_gate * (1.0 - star_ind) * put_intr, opt_val)
                    opt_val = tl.where(f_is_long, opt_long, opt_short)
                elif IS_LONG:
                    opt_val = opt_val - (1.0 - last_avg) * ki_gate * (1.0 - star_ind)
                else:
                    if HAS_LOWER_STRIKE:
                        put_intr = tl.maximum(strike - last_wo, zero_vec) - tl.maximum(lower_strike - last_wo, zero_vec)
                        put_mask = (strike > lower_strike) & (option_gearing > 0.0)
                    else:
                        put_intr = tl.maximum(strike - last_wo, zero_vec)
                        put_mask = option_gearing > 0.0
                    opt_val = tl.where(put_mask, opt_val - option_gearing * ki_gate * (1.0 - star_ind) * put_intr, opt_val)
                call_intr = tl.maximum(last_wo - part_call_strike, zero_vec)
                opt_val = tl.where(has_part_call, opt_val + part_call_rate * call_intr, opt_val)
                os_mask = has_one_star & (one_star_coupon != 0.0)
                opt_val = tl.where(os_mask, opt_val + star_ind * one_star_coupon, opt_val)

                if USE_RT_FLAGS:
                    pv_path = pv_path + df_last * tl.where(f_no_not, surv_prev * opt_val, surv_prev * (1.0 + opt_val))
                elif NO_NOTIONAL:
                    pv_path = pv_path + df_last * (surv_prev * opt_val)
                else:
                    pv_path = pv_path + df_last * (surv_prev * (1.0 + opt_val))
            tile_pv_acc += tl.sum(tl.where(mask_n, pv_path.to(tl.float64), zero_vec_f64), axis=0) * inv_N

        tl.store(pv_out_ptr + pid_b * n_t_tiles + pid_t, tile_pv_acc)

    @triton.jit  # type: ignore[untyped-decorator]
    def _eval_idx_base_stats_f64(  # type: ignore[no-untyped-def]
        perf_ptr,
        m_off,
        ia,
        ib,
        offs_n,
        mask_n,
        d_assets,
        n_total,
        NEED_MIN1: tl.constexpr,
        NEED_MIN2: tl.constexpr,
        NEED_MIN3: tl.constexpr,
        NEED_MAX1: tl.constexpr,
        NEED_MAX2: tl.constexpr,
        NEED_SUM1: tl.constexpr,
        NEED_SUM2: tl.constexpr,
        NEED_RATIO: tl.constexpr,
        DTYPE: tl.constexpr = tl.float64,
    ):
        big_v: tl.constexpr = 1e30 if DTYPE == tl.float32 else 1e300
        min1 = tl.full(offs_n.shape, big_v, dtype=DTYPE)
        min2 = tl.full(offs_n.shape, big_v, dtype=DTYPE)
        min3 = tl.full(offs_n.shape, big_v, dtype=DTYPE)
        idx_min1 = tl.zeros(offs_n.shape, dtype=tl.int32)
        idx_min2 = tl.zeros(offs_n.shape, dtype=tl.int32)
        idx_min3 = tl.zeros(offs_n.shape, dtype=tl.int32)
        max1 = tl.full(offs_n.shape, -big_v, dtype=DTYPE)
        max2 = tl.full(offs_n.shape, -big_v, dtype=DTYPE)
        idx_max1 = tl.zeros(offs_n.shape, dtype=tl.int32)
        idx_max2 = tl.zeros(offs_n.shape, dtype=tl.int32)
        s1 = tl.zeros(offs_n.shape, dtype=DTYPE)
        s2 = tl.zeros(offs_n.shape, dtype=DTYPE)

        for d in range(d_assets):
            pd = tl.load(perf_ptr + (m_off + d) * n_total + offs_n, mask=mask_n, other=1.0).to(DTYPE)
            if NEED_MIN3:
                lt1 = pd < min1
                lt2 = (~lt1) & (pd < min2)
                lt3 = (~lt1) & (~lt2) & (pd < min3)
                min3 = tl.where(lt1, min2, tl.where(lt2, min2, tl.where(lt3, pd, min3)))
                idx_min3 = tl.where(lt1, idx_min2, tl.where(lt2, idx_min2, tl.where(lt3, d, idx_min3)))
                min2 = tl.where(lt1, min1, tl.where(lt2, pd, min2))
                idx_min2 = tl.where(lt1, idx_min1, tl.where(lt2, d, idx_min2))
                min1 = tl.where(lt1, pd, min1)
                idx_min1 = tl.where(lt1, d, idx_min1)
            elif NEED_MIN2:
                lt1 = pd < min1
                lt2 = (~lt1) & (pd < min2)
                min2 = tl.where(lt1, min1, tl.where(lt2, pd, min2))
                idx_min2 = tl.where(lt1, idx_min1, tl.where(lt2, d, idx_min2))
                min1 = tl.where(lt1, pd, min1)
                idx_min1 = tl.where(lt1, d, idx_min1)
            elif NEED_MIN1:
                lt1 = pd < min1
                min1 = tl.where(lt1, pd, min1)
                idx_min1 = tl.where(lt1, d, idx_min1)

            if NEED_MAX2:
                gt1 = pd > max1
                gt2 = (~gt1) & (pd > max2)
                max2 = tl.where(gt1, max1, tl.where(gt2, pd, max2))
                idx_max2 = tl.where(gt1, idx_max1, tl.where(gt2, d, idx_max2))
                max1 = tl.where(gt1, pd, max1)
                idx_max1 = tl.where(gt1, d, idx_max1)
            elif NEED_MAX1:
                gt1 = pd > max1
                max1 = tl.where(gt1, pd, max1)
                idx_max1 = tl.where(gt1, d, idx_max1)

            if NEED_SUM1:
                s1 = s1 + pd
            if NEED_SUM2:
                s2 = s2 + pd * pd

        if NEED_RATIO:
            pa = tl.load(perf_ptr + (m_off + ia) * n_total + offs_n, mask=mask_n, other=1.0).to(DTYPE)
            pb = tl.load(perf_ptr + (m_off + ib) * n_total + offs_n, mask=mask_n, other=1.0).to(DTYPE)
        else:
            pa = tl.zeros(offs_n.shape, dtype=DTYPE)
            pb = tl.full(offs_n.shape, 1.0, dtype=DTYPE)
        return min1, min2, min3, idx_min1, idx_min2, idx_min3, max1, max2, idx_max1, idx_max2, s1, s2, pa, pb

    @triton.jit  # type: ignore[untyped-decorator]
    def _select_idx_from_stats_f64(  # type: ignore[no-untyped-def]
        min1, min2, min3, max1, max2, s1, s2, pa, pb, inv_d, d_assets, CODE: tl.constexpr
    ):
        w_avg = s1 * inv_d
        if CODE == 0:
            return min1
        if CODE == 1:
            return max1
        if CODE == 2:
            return w_avg
        if CODE == 3:
            return tl.where(d_assets <= 1, s1 * 0.0, tl.sqrt(tl.maximum(s2 * inv_d - w_avg * w_avg, 0.0)))
        if CODE == 4:
            return tl.where(d_assets < 2, pa, pa / tl.maximum(pb, 1e-12))
        if CODE == 5:
            return tl.where(d_assets < 2, pa, pb / tl.maximum(pa, 1e-12))
        if CODE == 6:
            return max1 / tl.maximum(min1, 1e-12)
        if CODE == 7:
            return max1 / tl.maximum(w_avg, 1e-12)
        if CODE == 8:
            return w_avg / tl.maximum(min1, 1e-12)
        if CODE == 9:
            return tl.where(d_assets < 2, s1 * 0.0, pa - pb)
        if CODE == 10:
            return tl.where(d_assets < 2, s1 * 0.0, pb - pa)
        if CODE == 11:
            return max1 - min1
        if CODE == 12:
            return max1 - w_avg
        if CODE == 13:
            return w_avg - min1
        if CODE == 14:
            return tl.where(min1 < 1.0, 2.0 - min1, min1)
        if CODE == 15:
            return tl.where(d_assets > 1, min2, min1)
        if CODE == 16:
            return tl.where(d_assets >= 3, min3, tl.where(d_assets == 2, min2, min1))
        if CODE == 17:
            return tl.where(d_assets > 1, max2, max1)
        return min1

    @triton.jit  # type: ignore[untyped-decorator]
    def _accum_idx_grad_f64(  # type: ignore[no-untyped-def]
        bp, bar_w, pd, d, d_assets, inv_d, ia, ib,
        min1, max1, w_avg, w_disp, idx_min1, idx_min2, idx_min3, idx_max1, idx_max2, pa, pb,
        CODE: tl.constexpr,
    ):
        zero_s = tl.zeros((), dtype=pd.dtype)
        one_s = zero_s + 1.0
        tiny_g = zero_s + 1e-12
        if CODE == 0:
            return tl.where(d == idx_min1, bp + bar_w, bp)
        if CODE == 1:
            return tl.where(d == idx_max1, bp + bar_w, bp)
        if CODE == 2:
            return bp + bar_w * inv_d
        if CODE == 3:
            g_d = tl.where((d_assets > 1) & (w_disp > tiny_g), bar_w * (pd - w_avg) * inv_d / tl.maximum(w_disp, tiny_g), zero_s)
            return bp + g_d
        if CODE == 4:
            if d_assets < 2:
                return tl.where(d == ia, bp + bar_w, bp)
            vb_c = tl.maximum(pb, tiny_g)
            bp = tl.where(d == ia, bp + bar_w / vb_c, bp)
            return tl.where((d == ib) & (pb > tiny_g), bp - bar_w * pa / (vb_c * vb_c), bp)
        if CODE == 5:
            if d_assets < 2:
                return tl.where(d == ia, bp + bar_w, bp)
            va_c = tl.maximum(pa, tiny_g)
            bp = tl.where(d == ib, bp + bar_w / va_c, bp)
            return tl.where((d == ia) & (pa > tiny_g), bp - bar_w * pb / (va_c * va_c), bp)
        if CODE == 6:
            m_c = tl.maximum(min1, tiny_g)
            bp = tl.where(d == idx_max1, bp + bar_w / m_c, bp)
            return tl.where((d == idx_min1) & (min1 > tiny_g), bp - bar_w * max1 / (m_c * m_c), bp)
        if CODE == 7:
            a_c = tl.maximum(w_avg, tiny_g)
            bp = tl.where(w_avg > tiny_g, bp - bar_w * max1 * inv_d / (a_c * a_c), bp)
            return tl.where(d == idx_max1, bp + bar_w / a_c, bp)
        if CODE == 8:
            m_c = tl.maximum(min1, tiny_g)
            bp = bp + bar_w * inv_d / m_c
            return tl.where((d == idx_min1) & (min1 > tiny_g), bp - bar_w * w_avg / (m_c * m_c), bp)
        if CODE == 9:
            if d_assets < 2:
                return bp
            return tl.where(d == ia, bp + bar_w, tl.where(d == ib, bp - bar_w, bp))
        if CODE == 10:
            if d_assets < 2:
                return bp
            return tl.where(d == ib, bp + bar_w, tl.where(d == ia, bp - bar_w, bp))
        if CODE == 11:
            return tl.where(d == idx_max1, bp + bar_w, bp) - tl.where(d == idx_min1, bar_w, zero_s)
        if CODE == 12:
            return tl.where(d == idx_max1, bp + bar_w, bp) - bar_w * inv_d
        if CODE == 13:
            return bp + bar_w * inv_d - tl.where(d == idx_min1, bar_w, zero_s)
        if CODE == 14:
            s_tw = tl.where(min1 < 1.0, -one_s, one_s)
            return tl.where(d == idx_min1, bp + bar_w * s_tw, bp)
        if CODE == 15:
            t_idx = tl.where(d_assets > 1, idx_min2, idx_min1)
            return tl.where(d == t_idx, bp + bar_w, bp)
        if CODE == 16:
            t_idx = tl.where(d_assets >= 3, idx_min3, tl.where(d_assets == 2, idx_min2, idx_min1))
            return tl.where(d == t_idx, bp + bar_w, bp)
        if CODE == 17:
            t_idx = tl.where(d_assets > 1, idx_max2, idx_max1)
            return tl.where(d == t_idx, bp + bar_w, bp)
        return bp

    @triton.jit  # type: ignore[untyped-decorator]
    def _eval_idx_base_stats_012_f64(  # type: ignore[no-untyped-def]
        perf_ptr,
        m_off,
        offs_n,
        mask_n,
        d_act,
        n_total,
        D_PAD: tl.constexpr,
        DTYPE: tl.constexpr = tl.float64,
    ):
        big_v: tl.constexpr = 1e30 if DTYPE == tl.float32 else 1e300
        min1 = tl.full(offs_n.shape, big_v, dtype=DTYPE)
        idx_min1 = tl.zeros(offs_n.shape, dtype=tl.int32)
        max1 = tl.full(offs_n.shape, -big_v, dtype=DTYPE)
        idx_max1 = tl.zeros(offs_n.shape, dtype=tl.int32)
        s1 = tl.zeros(offs_n.shape, dtype=DTYPE)
        for d in range(D_PAD):
            if d < d_act:
                pd = tl.load(perf_ptr + (m_off + d) * n_total + offs_n, mask=mask_n, other=1.0).to(DTYPE)
                lt1 = pd < min1
                min1 = tl.where(lt1, pd, min1)
                idx_min1 = tl.where(lt1, d, idx_min1)
                gt1 = pd > max1
                max1 = tl.where(gt1, pd, max1)
                idx_max1 = tl.where(gt1, d, idx_max1)
                s1 = s1 + pd
        inv_d = 1.0 / tl.maximum(tl.zeros((), dtype=DTYPE) + d_act, 1.0)
        return min1, idx_min1, max1, idx_max1, s1 * inv_d, inv_d

    @triton.jit  # type: ignore[untyped-decorator]
    def _payoff_aad_state_adjoints_f64_kernel(  # type: ignore[no-untyped-def]
        perf_ptr,
        ws_ptr,
        bar_lx_post_ptr,
        out_cf_wf_ptr,
        out_stage1_rep_ptr,
        i_ki_ext_ptr,
        df_pay_ptr,
        w_fund_ptr,
        ko_bar_ptr,
        is_call_ptr,
        cpn_bar_ptr,
        is_guar_ptr,
        cpn_rates_ptr,
        eps_vec_ptr,
        inv_perm_ptr,
        scalar_f64_ptr,
        d_assets,
        m_rem,
        n_total,
        cg_flags_ptr=None,
        n_rep: tl.constexpr = 1024,
        blocks_per_rep: tl.constexpr = 1,
        CALL_CODE: tl.constexpr = 0,
        CPN_CODE: tl.constexpr = 0,
        EFF_OPT_CODE: tl.constexpr = 0,
        HAS_KI: tl.constexpr = False,
        KI_ACTIVE: tl.constexpr = False,
        HAS_ONE_STAR: tl.constexpr = False,
        IS_MEM: tl.constexpr = False,
        NO_NOTIONAL: tl.constexpr = False,
        FORCE_CPN_AC: tl.constexpr = False,
        CPN_DIR_DOWN: tl.constexpr = False,
        CPN_TYPE_OUT: tl.constexpr = False,
        BLOCK_N: tl.constexpr = 256,
        HAS_LOWER_STRIKE: tl.constexpr = True,
        HAS_PART_CALL: tl.constexpr = False,
        EMIT_STAGE1: tl.constexpr = False,
        HAS_EXT_KI: tl.constexpr = False,
        USE_F32_INNER: tl.constexpr = False,
        IS_LONG: tl.constexpr = False,
        USE_RT_FLAGS: tl.constexpr = False,
        D_PAD: tl.constexpr = 0,
    ):
        pid_n = tl.program_id(0)
        rep_id = pid_n // blocks_per_rep
        tile_in_rep = pid_n % blocks_per_rep
        local_n = tile_in_rep * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n = rep_id * n_rep + local_n
        mask_n = (local_n < n_rep) & (offs_n < n_total)
        if USE_RT_FLAGS:
            d_act = tl.load(cg_flags_ptr + 0)
            m_act = tl.load(cg_flags_ptr + 1)
            rt_call = tl.load(cg_flags_ptr + 2)
            rt_cpn = tl.load(cg_flags_ptr + 3)
            rt_opt = tl.load(cg_flags_ptr + 4)
            f_has_ki = tl.load(cg_flags_ptr + 5) != 0
            f_ki_act = tl.load(cg_flags_ptr + 6) != 0
            f_has_os = tl.load(cg_flags_ptr + 7) != 0
            f_is_mem = tl.load(cg_flags_ptr + 8) != 0
            f_is_long = tl.load(cg_flags_ptr + 9) != 0
            f_no_not = tl.load(cg_flags_ptr + 10) != 0
            f_frc_cpn = tl.load(cg_flags_ptr + 11) != 0
            f_cpn_dn = tl.load(cg_flags_ptr + 12) != 0
            f_cpn_out = tl.load(cg_flags_ptr + 13) != 0
            f_has_low = tl.load(cg_flags_ptr + 14) != 0
            f_has_pc = tl.load(cg_flags_ptr + 15) != 0
        else:
            d_act = d_assets
            m_act = m_rem
            rt_call = CALL_CODE
            rt_cpn = CPN_CODE
            rt_opt = EFF_OPT_CODE
            f_has_ki = HAS_KI
            f_ki_act = KI_ACTIVE
            f_has_os = HAS_ONE_STAR
            f_is_mem = IS_MEM
            f_is_long = IS_LONG
            f_no_not = NO_NOTIONAL
            f_frc_cpn = FORCE_CPN_AC
            f_cpn_dn = CPN_DIR_DOWN
            f_cpn_out = CPN_TYPE_OUT
            f_has_low = HAS_LOWER_STRIKE
            f_has_pc = HAS_PART_CALL

        INNER_DTYPE: tl.constexpr = tl.float32 if USE_F32_INNER else tl.float64
        tiny_i = tl.zeros((), dtype=INNER_DTYPE) + 1e-12
        ko_spread = tl.load(scalar_f64_ptr + 0).to(INNER_DTYPE)
        cpn_spread = tl.load(scalar_f64_ptr + 1).to(INNER_DTYPE)
        ki_spread = tl.load(scalar_f64_ptr + 2).to(INNER_DTYPE)
        one_star_strike = tl.load(scalar_f64_ptr + 3).to(INNER_DTYPE)
        one_star_eps = tl.load(scalar_f64_ptr + 4).to(INNER_DTYPE)
        strike = tl.load(scalar_f64_ptr + 5).to(INNER_DTYPE)
        lower_strike = tl.load(scalar_f64_ptr + 6).to(INNER_DTYPE)
        ki_barrier = tl.load(scalar_f64_ptr + 7).to(INNER_DTYPE)
        option_gearing = tl.load(scalar_f64_ptr + 8).to(INNER_DTYPE)
        ac_cpn = tl.load(scalar_f64_ptr + 9).to(INNER_DTYPE)
        c_mem_0 = tl.load(scalar_f64_ptr + 10).to(INNER_DTYPE)
        i_ki_past = tl.load(scalar_f64_ptr + 11).to(INNER_DTYPE)
        part_call_strike = tl.load(scalar_f64_ptr + 12).to(INNER_DTYPE)
        part_call_rate = tl.load(scalar_f64_ptr + 13).to(INNER_DTYPE)
        one_star_coupon = tl.load(scalar_f64_ptr + 14).to(INNER_DTYPE)

        ia = tl.load(inv_perm_ptr + 0).to(tl.int32)
        ib_idx = tl.where(d_act > 1, 1, 0)
        ib = tl.load(inv_perm_ptr + ib_idx).to(tl.int32)

        zero_vec = tl.zeros((BLOCK_N,), dtype=INNER_DTYPE)
        one_vec = tl.full((BLOCK_N,), 1.0, dtype=INNER_DTYPE)
        scale_f64 = 1.0 / (tl.zeros((), dtype=tl.float64) + n_total)
        inv_rep_f64 = 1.0 / float(n_rep)
        scale = 1.0 / (tl.zeros((), dtype=INNER_DTYPE) + n_total)
        stride_mn = m_rem * n_total
        inv_d_f = 1.0 / tl.maximum(tl.zeros((), dtype=INNER_DTYPE) + d_act, 1.0)

        need_min1_k: tl.constexpr = (
            IS_LONG
            or (CALL_CODE == 0) or (CALL_CODE == 6) or (CALL_CODE == 8) or (CALL_CODE == 11) or (CALL_CODE == 13) or (CALL_CODE == 14)
            or (CPN_CODE == 0) or (CPN_CODE == 6) or (CPN_CODE == 8) or (CPN_CODE == 11) or (CPN_CODE == 13) or (CPN_CODE == 14)
            or (EFF_OPT_CODE == 0) or (EFF_OPT_CODE == 6) or (EFF_OPT_CODE == 8) or (EFF_OPT_CODE == 11) or (EFF_OPT_CODE == 13) or (EFF_OPT_CODE == 14)
        )
        need_min2_k: tl.constexpr = (CALL_CODE == 15) or (CPN_CODE == 15) or (EFF_OPT_CODE == 15)
        need_min3_k: tl.constexpr = (CALL_CODE == 16) or (CPN_CODE == 16) or (EFF_OPT_CODE == 16)
        need_max1_k: tl.constexpr = (
            HAS_ONE_STAR
            or (CALL_CODE == 1) or (CALL_CODE == 6) or (CALL_CODE == 7) or (CALL_CODE == 11) or (CALL_CODE == 12)
            or (CPN_CODE == 1) or (CPN_CODE == 6) or (CPN_CODE == 7) or (CPN_CODE == 11) or (CPN_CODE == 12)
            or (EFF_OPT_CODE == 1) or (EFF_OPT_CODE == 6) or (EFF_OPT_CODE == 7) or (EFF_OPT_CODE == 11) or (EFF_OPT_CODE == 12)
        )
        need_max2_k: tl.constexpr = (CALL_CODE == 17) or (CPN_CODE == 17) or (EFF_OPT_CODE == 17)
        need_sum1_k: tl.constexpr = (
            IS_LONG
            or (CALL_CODE == 2) or (CALL_CODE == 3) or (CALL_CODE == 7) or (CALL_CODE == 8) or (CALL_CODE == 12) or (CALL_CODE == 13)
            or (CPN_CODE == 2) or (CPN_CODE == 3) or (CPN_CODE == 7) or (CPN_CODE == 8) or (CPN_CODE == 12) or (CPN_CODE == 13)
            or (EFF_OPT_CODE == 2) or (EFF_OPT_CODE == 3) or (EFF_OPT_CODE == 7) or (EFF_OPT_CODE == 8) or (EFF_OPT_CODE == 12) or (EFF_OPT_CODE == 13)
        )
        need_sum2_k: tl.constexpr = (CALL_CODE == 3) or (CPN_CODE == 3) or (EFF_OPT_CODE == 3)
        need_ratio_k: tl.constexpr = (
            (CALL_CODE == 4) or (CALL_CODE == 5) or (CALL_CODE == 9) or (CALL_CODE == 10)
            or (CPN_CODE == 4) or (CPN_CODE == 5) or (CPN_CODE == 9) or (CPN_CODE == 10)
            or (EFF_OPT_CODE == 4) or (EFF_OPT_CODE == 5) or (EFF_OPT_CODE == 9) or (EFF_OPT_CODE == 10)
        )

        hard_ko_sp = tl.abs(ko_spread) > 1e-6
        eff_csp = tl.where(f_cpn_dn, -cpn_spread, cpn_spread) if USE_RT_FLAGS else (-cpn_spread if CPN_DIR_DOWN else cpn_spread)
        hard_cpn_sp = tl.abs(eff_csp) > 1e-6
        if USE_RT_FLAGS:
            sign_cpn_rt = tl.where(f_cpn_dn, -1.0, 1.0) * tl.where(f_cpn_out, -1.0, 1.0)
        else:
            sign_cpn: tl.constexpr = (-1.0 if CPN_DIR_DOWN else 1.0) * (-1.0 if CPN_TYPE_OUT else 1.0)
            sign_cpn_rt = tl.zeros((), dtype=INNER_DTYPE) + sign_cpn
        inv_w_ko_s1 = 1.0 / tl.maximum(tl.abs(ko_spread), tiny_i)
        inv_w_cp_s1 = 1.0 / tl.maximum(tl.abs(eff_csp), tiny_i)

        surv = one_vec
        c_mem = tl.full((BLOCK_N,), c_mem_0, dtype=INNER_DTYPE)
        surv_s1 = one_vec
        c_mem_s1 = tl.full((BLOCK_N,), c_mem_0, dtype=INNER_DTYPE)
        pv_s1 = zero_vec
        last_wo_pre = zero_vec
        last_avg_pre = zero_vec
        last_wb_pre = zero_vec

        n_s1_cols = 3 * m_rem + 6

        for m in range(m_rem):
            m_ok_f = (m < m_act) if USE_RT_FLAGS else True
            if m_ok_f:
                m_off = m * d_assets
                if USE_RT_FLAGS:
                    min1, _, max1, _, w_avg_0, _ = _eval_idx_base_stats_012_f64(
                        perf_ptr, m_off, offs_n, mask_n, d_act, n_total, D_PAD, INNER_DTYPE
                    )
                    w_call = tl.where(rt_call == 0, min1, tl.where(rt_call == 1, max1, w_avg_0))
                    w_cpn = tl.where(rt_cpn == 0, min1, tl.where(rt_cpn == 1, max1, w_avg_0))
                    is_last_f = m == (m_act - 1)
                    wo_cand = tl.where(f_is_long, min1, tl.where(rt_opt == 0, min1, tl.where(rt_opt == 1, max1, w_avg_0)))
                    last_wo_pre = tl.where(is_last_f, wo_cand, last_wo_pre)
                    last_avg_pre = tl.where(is_last_f & f_is_long, w_avg_0, last_avg_pre)
                    last_wb_pre = tl.where(is_last_f & f_has_os, max1, last_wb_pre)
                else:
                    min1, min2, min3, _, _, _, max1, max2, _, _, s1, s2, pa_r, pb_r = _eval_idx_base_stats_f64(
                        perf_ptr, m_off, ia, ib, offs_n, mask_n, d_assets, n_total,
                        need_min1_k or need_min2_k or need_min3_k,
                        need_min2_k,
                        need_min3_k,
                        need_max1_k or need_max2_k,
                        need_max2_k,
                        need_sum1_k or need_sum2_k,
                        need_sum2_k,
                        need_ratio_k,
                        INNER_DTYPE,
                    )
                    w_call = _select_idx_from_stats_f64(min1, min2, min3, max1, max2, s1, s2, pa_r, pb_r, inv_d_f, d_assets, CALL_CODE)
                    if CPN_CODE == CALL_CODE:
                        w_cpn = w_call
                    else:
                        w_cpn = _select_idx_from_stats_f64(min1, min2, min3, max1, max2, s1, s2, pa_r, pb_r, inv_d_f, d_assets, CPN_CODE)
                    if m == m_rem - 1:
                        last_wo_pre = min1 if IS_LONG else _select_idx_from_stats_f64(min1, min2, min3, max1, max2, s1, s2, pa_r, pb_r, inv_d_f, d_assets, EFF_OPT_CODE)
                        last_avg_pre = (s1 * inv_d_f) if IS_LONG else zero_vec
                        last_wb_pre = max1 if HAS_ONE_STAR else zero_vec

                eps_m = tl.load(eps_vec_ptr + m).to(INNER_DTYPE)
                half_eps = 0.5 * eps_m

                b_ko = tl.load(ko_bar_ptr + m).to(INNER_DTYPE)
                lo_ko = tl.where(hard_ko_sp, tl.minimum(b_ko, b_ko + ko_spread), b_ko - half_eps)
                hi_ko = tl.where(hard_ko_sp, tl.maximum(b_ko, b_ko + ko_spread), b_ko + half_eps)
                inv_w_ko = 1.0 / tl.maximum(hi_ko - lo_ko, tiny_i)
                u_ko = (w_call - lo_ko) * inv_w_ko
                g_ko_raw = tl.minimum(tl.maximum(u_ko, zero_vec), one_vec)
                dgk_raw = tl.where((u_ko > 0.0) & (u_ko < 1.0), inv_w_ko, zero_vec)
                is_call_m = tl.load(is_call_ptr + m) > 0.5
                g_k = tl.where(is_call_m, g_ko_raw, zero_vec)
                dgk = tl.where(is_call_m, dgk_raw, zero_vec)

                b_cp = tl.load(cpn_bar_ptr + m).to(INNER_DTYPE)
                lo_cp = tl.where(hard_cpn_sp, tl.minimum(b_cp, b_cp + eff_csp), b_cp - half_eps)
                hi_cp = tl.where(hard_cpn_sp, tl.maximum(b_cp, b_cp + eff_csp), b_cp + half_eps)
                inv_w_cp = 1.0 / tl.maximum(hi_cp - lo_cp, tiny_i)
                u_cp = (w_cpn - lo_cp) * inv_w_cp
                ramp_cp = tl.minimum(tl.maximum(u_cp, zero_vec), one_vec)
                dramp_cp = tl.where((u_cp > 0.0) & (u_cp < 1.0), inv_w_cp, zero_vec)
                if USE_RT_FLAGS:
                    hit_cp = tl.where(f_cpn_dn, 1.0 - ramp_cp, ramp_cp)
                    ind_cp = tl.where(f_cpn_out, 1.0 - hit_cp, hit_cp)
                else:
                    hit_cp = (1.0 - ramp_cp) if CPN_DIR_DOWN else ramp_cp
                    ind_cp = (1.0 - hit_cp) if CPN_TYPE_OUT else hit_cp
                is_guar_m = tl.load(is_guar_ptr + m) > 0.5
                g_c_raw = tl.where(is_guar_m, one_vec, ind_cp)
                dgc_raw = tl.where(is_guar_m, zero_vec, dramp_cp * sign_cpn_rt)

                if USE_RT_FLAGS:
                    w_c_max = tl.where(f_frc_cpn, tl.where(g_c_raw > g_k, one_vec, tl.where(g_c_raw < g_k, zero_vec, 0.5 * one_vec)), one_vec)
                    w_k_max = 1.0 - w_c_max
                    eff_g_c = tl.where(f_frc_cpn, tl.maximum(g_c_raw, g_k), g_c_raw)
                elif FORCE_CPN_AC:
                    eff_g_c = tl.maximum(g_c_raw, g_k)
                    w_c_max = tl.where(g_c_raw > g_k, one_vec, tl.where(g_c_raw < g_k, zero_vec, 0.5 * one_vec))
                    w_k_max = 1.0 - w_c_max
                else:
                    eff_g_c = g_c_raw
                    w_c_max = one_vec
                    w_k_max = zero_vec

                c_rate = tl.load(cpn_rates_ptr + m).to(INNER_DTYPE)
                eff_c = (tl.where(f_is_mem, c_rate + c_mem, zero_vec + c_rate) if USE_RT_FLAGS else ((c_rate + c_mem) if IS_MEM else (zero_vec + c_rate)))

                mn_off = m * n_total + offs_n
                tl.store(ws_ptr + 0 * stride_mn + mn_off, surv, mask=mask_n)
                tl.store(ws_ptr + 1 * stride_mn + mn_off, g_k, mask=mask_n)
                tl.store(ws_ptr + 2 * stride_mn + mn_off, eff_g_c, mask=mask_n)
                tl.store(ws_ptr + 3 * stride_mn + mn_off, eff_c, mask=mask_n)
                tl.store(ws_ptr + 4 * stride_mn + mn_off, dgk, mask=mask_n)
                tl.store(ws_ptr + 5 * stride_mn + mn_off, w_c_max * dgc_raw, mask=mask_n)
                tl.store(ws_ptr + 6 * stride_mn + mn_off, w_k_max * dgk, mask=mask_n)

                if USE_RT_FLAGS:
                    c_mem = tl.where(f_is_mem, surv * eff_c * (1.0 - eff_g_c), c_mem)
                elif IS_MEM:
                    c_mem = surv * eff_c * (1.0 - eff_g_c)
                surv = tl.maximum(surv - surv * g_k, zero_vec)

                if EMIT_STAGE1 and (out_stage1_rep_ptr is not None):
                    gk_s1 = tl.where(
                        is_call_m,
                        tl.where(
                            hard_ko_sp,
                            tl.minimum(tl.maximum((w_call - tl.minimum(b_ko, b_ko + ko_spread)) * inv_w_ko_s1, zero_vec), one_vec),
                            tl.where(w_call >= b_ko, one_vec, zero_vec),
                        ),
                        zero_vec,
                    )
                    if USE_RT_FLAGS:
                        hit_h_s1 = tl.where(f_cpn_dn, w_cpn <= b_cp, w_cpn >= b_cp)
                        gc_hard_s1 = tl.where(f_cpn_out, tl.where(hit_h_s1, zero_vec, one_vec), tl.where(hit_h_s1, one_vec, zero_vec))
                    else:
                        hit_h_s1 = (w_cpn <= b_cp) if CPN_DIR_DOWN else (w_cpn >= b_cp)
                        gc_hard_s1 = tl.where(hit_h_s1, zero_vec if CPN_TYPE_OUT else one_vec, one_vec if CPN_TYPE_OUT else zero_vec)
                    blo_cp_s1 = tl.minimum(b_cp, b_cp + eff_csp)
                    r_up_s1 = tl.minimum(tl.maximum((w_cpn - blo_cp_s1) * inv_w_cp_s1, zero_vec), one_vec)
                    if USE_RT_FLAGS:
                        hit_s1 = tl.where(f_cpn_dn, 1.0 - r_up_s1, r_up_s1)
                        gc_soft_s1 = tl.where(f_cpn_out, 1.0 - hit_s1, hit_s1)
                        gc_s1 = tl.where(is_guar_m, one_vec, tl.where(hard_cpn_sp, gc_soft_s1, gc_hard_s1))
                        eff_gc_s1 = tl.where(f_frc_cpn, tl.maximum(gc_s1, gk_s1), gc_s1)
                    else:
                        hit_s1 = (1.0 - r_up_s1) if CPN_DIR_DOWN else r_up_s1
                        gc_soft_s1 = (1.0 - hit_s1) if CPN_TYPE_OUT else hit_s1
                        gc_s1 = tl.where(is_guar_m, one_vec, tl.where(hard_cpn_sp, gc_soft_s1, gc_hard_s1))
                        eff_gc_s1 = tl.maximum(gc_s1, gk_s1) if FORCE_CPN_AC else gc_s1
                    pm_s1 = surv_s1 * gk_s1
                    eff_c_s1 = (tl.where(f_is_mem, c_rate + c_mem_s1, zero_vec + c_rate) if USE_RT_FLAGS else ((c_rate + c_mem_s1) if IS_MEM else (zero_vec + c_rate)))
                    cpn_cf_s1 = surv_s1 * eff_gc_s1 * eff_c_s1 + pm_s1 * ac_cpn
                    if USE_RT_FLAGS:
                        c_mem_s1 = tl.where(f_is_mem, surv_s1 * eff_c_s1 * (1.0 - eff_gc_s1), c_mem_s1)
                    elif IS_MEM:
                        c_mem_s1 = surv_s1 * eff_c_s1 * (1.0 - eff_gc_s1)
                    df_m_s1 = tl.load(df_pay_ptr + m).to(INNER_DTYPE)
                    wf_m_s1 = tl.load(w_fund_ptr + m).to(INNER_DTYPE) if w_fund_ptr is not None else (tl.zeros((), dtype=INNER_DTYPE))
                    if USE_RT_FLAGS:
                        pv_s1 = pv_s1 + df_m_s1 * tl.where(f_no_not, cpn_cf_s1, pm_s1 + cpn_cf_s1) - wf_m_s1 * surv_s1
                    elif NO_NOTIONAL:
                        pv_s1 = pv_s1 + df_m_s1 * cpn_cf_s1 - wf_m_s1 * surv_s1
                    else:
                        pv_s1 = pv_s1 + df_m_s1 * (pm_s1 + cpn_cf_s1) - wf_m_s1 * surv_s1
                    p_stop_sum_r = tl.sum(tl.where(mask_n, pm_s1, zero_vec).to(tl.float64), axis=0) * inv_rep_f64
                    q_cpn_sum_r = tl.sum(tl.where(mask_n, surv_s1 * eff_gc_s1, zero_vec).to(tl.float64), axis=0) * inv_rep_f64
                    cpn_cf_sum_r = tl.sum(tl.where(mask_n, cpn_cf_s1, zero_vec).to(tl.float64), axis=0) * inv_rep_f64
                    tl.atomic_add(out_stage1_rep_ptr + rep_id * n_s1_cols + m, p_stop_sum_r)
                    tl.atomic_add(out_stage1_rep_ptr + rep_id * n_s1_cols + m_rem + m, q_cpn_sum_r)
                    tl.atomic_add(out_stage1_rep_ptr + rep_id * n_s1_cols + 2 * m_rem + m, cpn_cf_sum_r)
                    surv_s1 = tl.maximum(surv_s1 - pm_s1, zero_vec)

        m_last = m_act - 1
        last_wo = last_wo_pre
        last_wb = last_wb_pre
        eps_term = tl.load(eps_vec_ptr + m_last).to(INNER_DTYPE)
        if HAS_EXT_KI and (i_ki_ext_ptr is not None):
            ki_gate = tl.load(i_ki_ext_ptr + offs_n, mask=mask_n, other=i_ki_past).to(INNER_DTYPE)
            d_iki_dwo = zero_vec
            ki_gate_s1 = ki_gate
        elif USE_RT_FLAGS:
            sp_ki_pos = tl.abs(ki_spread) > 1e-6
            blo_k = tl.minimum(ki_barrier, ki_barrier + ki_spread)
            bhi_k = tl.where(sp_ki_pos, tl.maximum(ki_barrier, ki_barrier + ki_spread), ki_barrier + 0.5 * eps_term)
            inv_wk = 1.0 / tl.maximum(tl.where(sp_ki_pos, bhi_k - blo_k, eps_term), tiny_i)
            u_ki = (bhi_k - last_wo) * inv_wk
            i_ki = tl.minimum(tl.maximum(u_ki, zero_vec), one_vec)
            d_iki_dwo = tl.where(f_ki_act & (u_ki > 0.0) & (u_ki < 1.0), -inv_wk, zero_vec)
            ki_gate_off = tl.where(f_has_ki, zero_vec + i_ki_past, one_vec)
            ki_gate = tl.where(f_ki_act, i_ki, ki_gate_off)
            ki_gate_s1 = tl.where(f_ki_act, tl.where(sp_ki_pos, i_ki, tl.where(last_wo < ki_barrier, one_vec, zero_vec)), ki_gate_off)
        elif KI_ACTIVE:
            sp_ki_pos = tl.abs(ki_spread) > 1e-6
            blo_k = tl.minimum(ki_barrier, ki_barrier + ki_spread)
            bhi_k = tl.where(sp_ki_pos, tl.maximum(ki_barrier, ki_barrier + ki_spread), ki_barrier + 0.5 * eps_term)
            inv_wk = 1.0 / tl.maximum(tl.where(sp_ki_pos, bhi_k - blo_k, eps_term), tiny_i)
            u_ki = (bhi_k - last_wo) * inv_wk
            i_ki = tl.minimum(tl.maximum(u_ki, zero_vec), one_vec)
            d_iki_dwo = tl.where((u_ki > 0.0) & (u_ki < 1.0), -inv_wk, zero_vec)
            ki_gate = i_ki
            if sp_ki_pos:
                ki_gate_s1 = i_ki
            else:
                ki_gate_s1 = tl.where(last_wo < ki_barrier, one_vec, zero_vec)
        else:
            d_iki_dwo = zero_vec
            ki_gate = (zero_vec + i_ki_past) if HAS_KI else one_vec
            ki_gate_s1 = ki_gate

        if USE_RT_FLAGS:
            os_soft = one_star_eps > 1e-6
            inv_wos = 1.0 / tl.maximum(one_star_eps, tiny_i)
            u_os = (last_wb - (one_star_strike - 0.5 * one_star_eps)) * inv_wos
            star_ind = tl.where(f_has_os, tl.where(os_soft, tl.minimum(tl.maximum(u_os, zero_vec), one_vec), tl.where(last_wb >= one_star_strike, one_vec, zero_vec)), zero_vec)
            d_star = tl.where(f_has_os & os_soft & (u_os > 0.0) & (u_os < 1.0), inv_wos, zero_vec)
        elif HAS_ONE_STAR:
            os_soft = one_star_eps > 1e-6
            inv_wos = 1.0 / tl.maximum(one_star_eps, tiny_i)
            u_os = (last_wb - (one_star_strike - 0.5 * one_star_eps)) * inv_wos
            star_ind = tl.where(os_soft, tl.minimum(tl.maximum(u_os, zero_vec), one_vec), tl.where(last_wb >= one_star_strike, one_vec, zero_vec))
            d_star = tl.where(os_soft & (u_os > 0.0) & (u_os < 1.0), inv_wos, zero_vec)
        else:
            star_ind = zero_vec
            d_star = zero_vec

        opt_pay = zero_vec
        d_opt_d_wo = zero_vec
        d_opt_d_avg = zero_vec
        d_opt_d_wb = zero_vec
        if USE_RT_FLAGS:
            sp_ki_ilb = tl.where(tl.abs(ki_spread) > 1e-6, ki_spread, -eps_term)
            lo_k_ilb = tl.minimum(strike, strike + sp_ki_ilb)
            hi_k_ilb = tl.maximum(strike, strike + sp_ki_ilb)
            inv_wk_ilb = 1.0 / tl.maximum(hi_k_ilb - lo_k_ilb, tiny_i)
            u_ki_ilb = (hi_k_ilb - last_wo) * inv_wk_ilb
            g_ki_ilb = tl.minimum(tl.maximum(u_ki_ilb, zero_vec), one_vec)
            ki_gate_ilb_act = tl.maximum(i_ki_past, g_ki_ilb)
            ki_gate_ilb = tl.where(f_ki_act, ki_gate_ilb_act, one_vec)
            d_iki_dwo_ilb = tl.where(f_ki_act & (u_ki_ilb > 0.0) & (u_ki_ilb < 1.0) & (g_ki_ilb >= i_ki_past), -inv_wk_ilb, zero_vec)
            ki_gate_ilb_s1 = tl.where(
                f_ki_act,
                tl.where(tl.abs(ki_spread) > 1e-6, ki_gate_ilb_act, tl.maximum(i_ki_past, tl.where(last_wo < strike, one_vec, zero_vec))),
                one_vec,
            )
            opt_l_val = -(1.0 - last_avg_pre) * ki_gate_ilb * (1.0 - star_ind)
            put_intr_l = tl.maximum(strike - last_wo, zero_vec) - tl.maximum(lower_strike - last_wo, zero_vec)
            put_intr_p = tl.maximum(strike - last_wo, zero_vec)
            put_intr = tl.where(f_has_low, put_intr_l, put_intr_p)
            d_put_l = tl.where(last_wo < strike, -one_vec, zero_vec) - tl.where(last_wo < lower_strike, -one_vec, zero_vec)
            d_put_p = tl.where(last_wo < strike, -one_vec, zero_vec)
            d_put_dwo = tl.where(f_has_low, d_put_l, d_put_p)
            put_mask = tl.where(f_has_low, (strike > lower_strike) & (option_gearing > 0.0), option_gearing > 0.0)
            opt_s_val = tl.where(put_mask, -option_gearing * ki_gate * (1.0 - star_ind) * put_intr, zero_vec)
            opt_pay = tl.where(f_is_long, opt_l_val, opt_s_val)
            d_opt_d_avg = tl.where(f_is_long, ki_gate_ilb * (1.0 - star_ind), zero_vec)
            d_wo_l = -(1.0 - last_avg_pre) * (1.0 - star_ind) * d_iki_dwo_ilb
            d_wo_s = tl.where(put_mask, -option_gearing * (1.0 - star_ind) * (d_iki_dwo * put_intr + ki_gate * d_put_dwo), zero_vec)
            d_opt_d_wo = tl.where(f_is_long, d_wo_l, d_wo_s)
            d_wb_l = (1.0 - last_avg_pre) * ki_gate_ilb * d_star
            d_wb_s = tl.where(put_mask, option_gearing * ki_gate * put_intr * d_star, zero_vec)
            d_opt_d_wb = tl.where(f_is_long, d_wb_l, d_wb_s)
            ki_gate_ilb_s1 = tl.where(f_is_long, ki_gate_ilb_s1, ki_gate_s1)
            put_mask = tl.where(f_is_long, zero_vec > 1.0, put_mask)
        elif IS_LONG:
            if HAS_EXT_KI and (i_ki_ext_ptr is not None):
                ki_gate_ilb = ki_gate
                ki_gate_ilb_s1 = ki_gate_s1
                d_iki_dwo_ilb = zero_vec
            elif KI_ACTIVE:
                sp_ki_ilb = tl.where(tl.abs(ki_spread) > 1e-6, ki_spread, -eps_term)
                lo_k_ilb = tl.minimum(strike, strike + sp_ki_ilb)
                hi_k_ilb = tl.maximum(strike, strike + sp_ki_ilb)
                inv_wk_ilb = 1.0 / tl.maximum(hi_k_ilb - lo_k_ilb, tiny_i)
                u_ki_ilb = (hi_k_ilb - last_wo) * inv_wk_ilb
                g_ki_ilb = tl.minimum(tl.maximum(u_ki_ilb, zero_vec), one_vec)
                ki_gate_ilb = tl.maximum(i_ki_past, g_ki_ilb)
                d_iki_dwo_ilb = tl.where((u_ki_ilb > 0.0) & (u_ki_ilb < 1.0) & (g_ki_ilb >= i_ki_past), -inv_wk_ilb, zero_vec)
                ki_gate_ilb_s1 = tl.where(
                    tl.abs(ki_spread) > 1e-6,
                    ki_gate_ilb,
                    tl.maximum(i_ki_past, tl.where(last_wo < strike, one_vec, zero_vec)),
                )
            else:
                ki_gate_ilb = one_vec
                ki_gate_ilb_s1 = one_vec
                d_iki_dwo_ilb = zero_vec
            opt_pay = opt_pay - (1.0 - last_avg_pre) * ki_gate_ilb * (1.0 - star_ind)
            d_opt_d_avg = ki_gate_ilb * (1.0 - star_ind)
            d_opt_d_wo = d_opt_d_wo - (1.0 - last_avg_pre) * (1.0 - star_ind) * d_iki_dwo_ilb
            d_opt_d_wb = d_opt_d_wb + (1.0 - last_avg_pre) * ki_gate_ilb * d_star
            put_mask = zero_vec > 1.0
            put_intr = zero_vec
        else:
            if HAS_LOWER_STRIKE:
                put_mask = (strike > lower_strike) & (option_gearing > 0.0)
                put_intr = tl.maximum(strike - last_wo, zero_vec) - tl.maximum(lower_strike - last_wo, zero_vec)
                d_put_dwo = tl.where(last_wo < strike, -one_vec, zero_vec) - tl.where(last_wo < lower_strike, -one_vec, zero_vec)
            else:
                put_mask = option_gearing > 0.0
                put_intr = tl.maximum(strike - last_wo, zero_vec)
                d_put_dwo = tl.where(last_wo < strike, -one_vec, zero_vec)

            opt_pay = tl.where(put_mask, opt_pay - option_gearing * ki_gate * (1.0 - star_ind) * put_intr, opt_pay)
            d_opt_d_wo = tl.where(
                put_mask,
                d_opt_d_wo - option_gearing * (1.0 - star_ind) * (d_iki_dwo * put_intr + ki_gate * d_put_dwo),
                d_opt_d_wo,
            )
            d_opt_d_wb = tl.where(put_mask, d_opt_d_wb + option_gearing * ki_gate * put_intr * d_star, d_opt_d_wb)
            ki_gate_ilb_s1 = ki_gate_s1

        has_pc = (f_has_pc & (part_call_rate != 0.0)) if USE_RT_FLAGS else ((part_call_rate != 0.0) if HAS_PART_CALL else False)
        call_intr = tl.maximum(last_wo - part_call_strike, zero_vec)
        opt_pay = tl.where(has_pc, opt_pay + part_call_rate * call_intr, opt_pay)
        d_opt_d_wo = tl.where(has_pc & (last_wo > part_call_strike), d_opt_d_wo + part_call_rate, d_opt_d_wo)

        os_cpn_mask = (f_has_os & (one_star_coupon != 0.0)) if USE_RT_FLAGS else ((one_star_coupon != 0.0) if HAS_ONE_STAR else False)
        opt_pay = tl.where(os_cpn_mask, opt_pay + star_ind * one_star_coupon, opt_pay)
        d_opt_d_wb = tl.where(os_cpn_mask, d_opt_d_wb + d_star * one_star_coupon, d_opt_d_wb)

        df_last = tl.load(df_pay_ptr + m_last).to(INNER_DTYPE)
        if EMIT_STAGE1 and (out_stage1_rep_ptr is not None):
            if USE_RT_FLAGS:
                u_put_s1 = tl.where(
                    f_is_long,
                    surv_s1 * ki_gate_ilb_s1 * (1.0 - star_ind) * (1.0 - last_avg_pre),
                    tl.where(put_mask, surv_s1 * ki_gate_s1 * (1.0 - star_ind) * put_intr, zero_vec),
                )
                put_term_s1 = tl.where(f_is_long, -u_put_s1, -option_gearing * u_put_s1)
                u_call_s1 = tl.where(has_pc, surv_s1 * call_intr, zero_vec)
                p_os_s1 = tl.where(f_has_os, surv_s1 * star_ind, zero_vec)
                l_os_s1 = tl.where(os_cpn_mask, p_os_s1 * one_star_coupon, zero_vec)
                opt_s1 = put_term_s1 + tl.where(has_pc, part_call_rate * u_call_s1, zero_vec) + l_os_s1
                term_s1 = (surv_s1 * tl.where(f_no_not, 0.0, 1.0)) + opt_s1
            else:
                if IS_LONG:
                    u_put_s1 = surv_s1 * ki_gate_ilb_s1 * (1.0 - star_ind) * (1.0 - last_avg_pre)
                    put_term_s1 = -u_put_s1
                else:
                    u_put_s1 = tl.where(put_mask, surv_s1 * ki_gate_s1 * (1.0 - star_ind) * put_intr, zero_vec)
                    put_term_s1 = -option_gearing * u_put_s1
                u_call_s1 = tl.where(has_pc, surv_s1 * call_intr, zero_vec)
                p_os_s1 = tl.where(HAS_ONE_STAR, surv_s1 * star_ind, zero_vec)
                l_os_s1 = tl.where(os_cpn_mask, p_os_s1 * one_star_coupon, zero_vec)
                opt_s1 = (
                    put_term_s1
                    + (part_call_rate * u_call_s1 if has_pc else zero_vec)
                    + l_os_s1
                )
                term_s1 = (surv_s1 * (0.0 if NO_NOTIONAL else 1.0)) + opt_s1
            pv_s1 = pv_s1 + df_last * term_s1
            pv_s1_f64 = pv_s1.to(tl.float64)
            base_c = rep_id * n_s1_cols + 3 * m_rem
            tl.atomic_add(out_stage1_rep_ptr + base_c + 0, tl.sum(tl.where(mask_n, u_put_s1, zero_vec).to(tl.float64), axis=0) * inv_rep_f64)
            tl.atomic_add(out_stage1_rep_ptr + base_c + 1, tl.sum(tl.where(mask_n, u_call_s1, zero_vec).to(tl.float64), axis=0) * inv_rep_f64)
            tl.atomic_add(out_stage1_rep_ptr + base_c + 2, tl.sum(tl.where(mask_n, l_os_s1, zero_vec).to(tl.float64), axis=0) * inv_rep_f64)
            tl.atomic_add(out_stage1_rep_ptr + base_c + 3, tl.sum(tl.where(mask_n, p_os_s1, zero_vec).to(tl.float64), axis=0) * inv_rep_f64)
            tl.atomic_add(out_stage1_rep_ptr + base_c + 4, tl.sum(tl.where(mask_n, pv_s1_f64, 0.0), axis=0) * inv_rep_f64)
            tl.atomic_add(out_stage1_rep_ptr + base_c + 5, tl.sum(tl.where(mask_n, pv_s1_f64 * pv_s1_f64, 0.0), axis=0) * inv_rep_f64)

        term_not = (tl.where(f_no_not, 0.0, 1.0).to(INNER_DTYPE) if USE_RT_FLAGS else (tl.zeros((), dtype=INNER_DTYPE) + (0.0 if NO_NOTIONAL else 1.0)))
        bar_surv = scale * df_last * (term_not + opt_pay)
        bar_last_wo = scale * df_last * surv * d_opt_d_wo
        bar_last_avg = scale * df_last * surv * d_opt_d_avg
        bar_last_wb = scale * df_last * surv * d_opt_d_wb
        bar_c_mem = zero_vec
        cf_tail_vec = surv * (tl.where(f_no_not, opt_pay, 1.0 + opt_pay) if USE_RT_FLAGS else (opt_pay if NO_NOTIONAL else (1.0 + opt_pay)))
        n_blk_n = tl.num_programs(0)
        opt_bwd_rt = tl.where(f_is_long, 0, rt_opt) if USE_RT_FLAGS else (0 if IS_LONG else EFF_OPT_CODE)

        for rev_m in range(m_rem):
            m = (m_last - rev_m) if not USE_RT_FLAGS else (m_rem - 1 - rev_m)
            m_ok_b = (m < m_act) if USE_RT_FLAGS else True
            if m_ok_b:
                mn_off = m * n_total + offs_n
                surv_pre = tl.load(ws_ptr + 0 * stride_mn + mn_off, mask=mask_n, other=0.0).to(INNER_DTYPE)
                g_k = tl.load(ws_ptr + 1 * stride_mn + mn_off, mask=mask_n, other=0.0).to(INNER_DTYPE)
                eff_g_c = tl.load(ws_ptr + 2 * stride_mn + mn_off, mask=mask_n, other=0.0).to(INNER_DTYPE)
                eff_c = tl.load(ws_ptr + 3 * stride_mn + mn_off, mask=mask_n, other=0.0).to(INNER_DTYPE)
                dgk = tl.load(ws_ptr + 4 * stride_mn + mn_off, mask=mask_n, other=0.0).to(INNER_DTYPE)
                dgc_from_cpn = tl.load(ws_ptr + 5 * stride_mn + mn_off, mask=mask_n, other=0.0).to(INNER_DTYPE)
                dgc_from_ko = tl.load(ws_ptr + 6 * stride_mn + mn_off, mask=mask_n, other=0.0).to(INNER_DTYPE)

                df_m = tl.load(df_pay_ptr + m).to(INNER_DTYPE)
                is_last = m == m_last
                if out_cf_wf_ptr is not None:
                    p_m_val = surv_pre * g_k
                    cpn_val = surv_pre * eff_g_c * eff_c + p_m_val * ac_cpn
                    cf_path = (tl.where(f_no_not, cpn_val, p_m_val + cpn_val) if USE_RT_FLAGS else (cpn_val if NO_NOTIONAL else (p_m_val + cpn_val)))
                    cf_path = tl.where(is_last, cf_path + cf_tail_vec, cf_path)
                    cf_m_sum = tl.sum(tl.where(mask_n, cf_path, zero_vec).to(tl.float64), axis=0) * scale_f64
                    surv_m_sum = tl.sum(tl.where(mask_n, surv_pre, zero_vec).to(tl.float64), axis=0) * scale_f64
                    tl.store(out_cf_wf_ptr + pid_n * m_rem + m, cf_m_sum)
                    tl.store(out_cf_wf_ptr + (n_blk_n + pid_n) * m_rem + m, surv_m_sum)
                surv_next_pos = (surv_pre - surv_pre * g_k) > 0.0
                bar_surv_eff = tl.where(surv_next_pos, bar_surv, zero_vec)
                bar_p_m = scale * df_m * (term_not + ac_cpn) - bar_surv_eff
                bar_eff_g_c = scale * df_m * surv_pre * eff_c - (tl.where(f_is_mem, bar_c_mem * surv_pre * eff_c, zero_vec) if USE_RT_FLAGS else (bar_c_mem * surv_pre * eff_c if IS_MEM else zero_vec))

                wf_m = tl.load(w_fund_ptr + m).to(INNER_DTYPE) if w_fund_ptr is not None else (tl.zeros((), dtype=INNER_DTYPE))
                bar_surv = (
                    bar_surv_eff
                    + bar_p_m * g_k
                    + scale * df_m * eff_g_c * eff_c
                    - scale * wf_m
                    + (tl.where(f_is_mem, bar_c_mem * eff_c * (1.0 - eff_g_c), zero_vec) if USE_RT_FLAGS else (bar_c_mem * eff_c * (1.0 - eff_g_c) if IS_MEM else zero_vec))
                )
                if USE_RT_FLAGS:
                    bar_c_mem = tl.where(f_is_mem, scale * df_m * surv_pre * eff_g_c + bar_c_mem * surv_pre * (1.0 - eff_g_c), bar_c_mem)
                elif IS_MEM:
                    bar_c_mem = scale * df_m * surv_pre * eff_g_c + bar_c_mem * surv_pre * (1.0 - eff_g_c)

                bar_g_k = bar_p_m * surv_pre
                bar_w_call = bar_g_k * dgk + bar_eff_g_c * dgc_from_ko
                bar_w_cpn = bar_eff_g_c * dgc_from_cpn

                b_wo = tl.where(is_last, bar_last_wo, zero_vec)
                b_avg = (tl.where(is_last & f_is_long, bar_last_avg, zero_vec) if USE_RT_FLAGS else (tl.where(is_last, bar_last_avg, zero_vec) if IS_LONG else zero_vec))
                b_wb = tl.where(is_last, bar_last_wb, zero_vec)
                m_off = m * d_assets
                if USE_RT_FLAGS:
                    _, idx_min1, _, idx_max1, _, inv_d_rt = _eval_idx_base_stats_012_f64(
                        perf_ptr, m_off, offs_n, mask_n, d_act, n_total, D_PAD, INNER_DTYPE
                    )
                    for d in range(D_PAD):
                        if d < d_act:
                            pd = tl.load(perf_ptr + (m_off + d) * n_total + offs_n, mask=mask_n, other=1.0).to(INNER_DTYPE)
                            g_call = tl.where(rt_call == 0, tl.where(d == idx_min1, bar_w_call, zero_vec), tl.where(rt_call == 1, tl.where(d == idx_max1, bar_w_call, zero_vec), bar_w_call * inv_d_rt))
                            g_cpn = tl.where(rt_cpn == 0, tl.where(d == idx_min1, bar_w_cpn, zero_vec), tl.where(rt_cpn == 1, tl.where(d == idx_max1, bar_w_cpn, zero_vec), bar_w_cpn * inv_d_rt))
                            g_opt = tl.where(opt_bwd_rt == 0, tl.where(d == idx_min1, b_wo, zero_vec), tl.where(opt_bwd_rt == 1, tl.where(d == idx_max1, b_wo, zero_vec), b_wo * inv_d_rt))
                            bp = g_call + g_cpn + g_opt + b_avg * inv_d_rt + tl.where(d == idx_max1, b_wb, zero_vec)
                            tl.store(bar_lx_post_ptr + (m_off + d) * n_total + offs_n, (bp * pd).to(tl.float32), mask=mask_n)
                else:
                    OPT_BWD_K: tl.constexpr = 0 if IS_LONG else EFF_OPT_CODE
                    if OPT_BWD_K == CALL_CODE:
                        bar_w_call = bar_w_call + b_wo
                        b_wo = zero_vec
                    elif OPT_BWD_K == CPN_CODE:
                        bar_w_cpn = bar_w_cpn + b_wo
                        b_wo = zero_vec
                    if IS_LONG:
                        if CALL_CODE == 2:
                            bar_w_call = bar_w_call + b_avg
                            b_avg = zero_vec
                        elif CPN_CODE == 2:
                            bar_w_cpn = bar_w_cpn + b_avg
                            b_avg = zero_vec
                    if CALL_CODE == 1:
                        bar_w_call = bar_w_call + b_wb
                        b_wb = zero_vec

                    min1_v, _, _, idx_min1, idx_min2, idx_min3, max1_v, _, idx_max1, idx_max2, s1_v, s2_v, pa_r, pb_r = _eval_idx_base_stats_f64(
                        perf_ptr, m_off, ia, ib, offs_n, mask_n, d_assets, n_total,
                        need_min1_k or need_min2_k or need_min3_k,
                        need_min2_k,
                        need_min3_k,
                        need_max1_k or need_max2_k,
                        need_max2_k,
                        need_sum1_k or need_sum2_k,
                        need_sum2_k,
                        need_ratio_k,
                        INNER_DTYPE,
                    )
                    w_avg_v = s1_v * inv_d_f
                    w_disp_v = tl.sqrt(tl.maximum(s2_v * inv_d_f - w_avg_v * w_avg_v, 0.0))
                    eff_wc_tot = bar_w_call + (bar_w_cpn if CPN_CODE == CALL_CODE else zero_vec)

                    for d in range(d_assets):
                        pd = tl.load(perf_ptr + (m_off + d) * n_total + offs_n, mask=mask_n, other=1.0).to(INNER_DTYPE)
                        bp = zero_vec
                        bp = _accum_idx_grad_f64(
                            bp, eff_wc_tot, pd, d, d_assets, inv_d_f, ia, ib,
                            min1_v, max1_v, w_avg_v, w_disp_v, idx_min1, idx_min2, idx_min3, idx_max1, idx_max2, pa_r, pb_r,
                            CALL_CODE,
                        )
                        if CPN_CODE != CALL_CODE:
                            bp = _accum_idx_grad_f64(
                                bp, bar_w_cpn, pd, d, d_assets, inv_d_f, ia, ib,
                                min1_v, max1_v, w_avg_v, w_disp_v, idx_min1, idx_min2, idx_min3, idx_max1, idx_max2, pa_r, pb_r,
                                CPN_CODE,
                            )
                        if (OPT_BWD_K != CALL_CODE) and (OPT_BWD_K != CPN_CODE):
                            bp = _accum_idx_grad_f64(
                                bp, b_wo, pd, d, d_assets, inv_d_f, ia, ib,
                                min1_v, max1_v, w_avg_v, w_disp_v, idx_min1, idx_min2, idx_min3, idx_max1, idx_max2, pa_r, pb_r,
                                OPT_BWD_K,
                            )
                        if IS_LONG and (CALL_CODE != 2) and (CPN_CODE != 2):
                            bp = bp + b_avg * inv_d_f
                        if HAS_ONE_STAR and (CALL_CODE != 1):
                            bp = tl.where(d == idx_max1, bp + b_wb, bp)

                        tl.store(bar_lx_post_ptr + (m_off + d) * n_total + offs_n, (bp * pd).to(tl.float32), mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _mf_gbm_ema_shift_f32_kernel(  # type: ignore[no-untyped-def]
        z_coarse_corr_ptr,  # (R, M, D, n_rep_in) or (M, D, R*n_rep_in) float32
        d_var_f32_ptr,      # (D, M) float32
        ema_shift_ptr,      # (n_sgn, R, M, D) float32
        n_r,
        d_assets,
        m_rem,
        n_rep_in,
        z_scale,
        cg_flags_ptr=None,
        ANTITHETIC: tl.constexpr = False,
        BLOCK_N: tl.constexpr = 256,
        LAYOUT_MDN: tl.constexpr = False,
        USE_RT_FLAGS: tl.constexpr = False,
    ):
        pid_rs = tl.program_id(0)
        d = tl.program_id(1)
        if USE_RT_FLAGS:
            d_act = tl.load(cg_flags_ptr + 0)
            m_act = tl.load(cg_flags_ptr + 1)
            if d >= d_act:
                return
        else:
            m_act = m_rem
        n_sgn: tl.constexpr = 2 if ANTITHETIC else 1
        sgn_id = pid_rs % n_sgn
        r_idx = pid_rs // n_sgn
        z_sign = (tl.where(sgn_id == 0, 1.0, -1.0) if ANTITHETIC else 1.0) * z_scale
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep_in
        inv_rep = 1.0 / tl.maximum(tl.zeros((), dtype=tl.float32) + n_rep_in, 1.0)
        n_tot_z = n_r * n_rep_in
        lx = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for m in range(m_rem):
            if m < m_act:
                dv = tl.load(d_var_f32_ptr + d * m_rem + m)
                sdv = tl.sqrt(tl.maximum(dv, 0.0))
                if LAYOUT_MDN:
                    z_off_e = (m * d_assets + d) * n_tot_z + r_idx * n_rep_in + offs_n
                else:
                    z_off_e = ((r_idx * m_rem + m) * d_assets + d) * n_rep_in + offs_n
                z_val = tl.load(z_coarse_corr_ptr + z_off_e, mask=mask_n, other=0.0) * z_sign
                lx_raw = lx - 0.5 * dv + sdv * z_val
                mx = tl.max(tl.where(mask_n, lx_raw, -1.0e30), axis=0)
                ex_sum = tl.sum(tl.where(mask_n, tl.exp(lx_raw - mx), 0.0), axis=0)
                lse = mx + tl.log(tl.maximum(ex_sum * inv_rep, 1.0e-30))
                tl.store(ema_shift_ptr + ((sgn_id * n_r + r_idx) * m_rem + m) * d_assets + d, lse)
                lx = lx_raw - lse

    @triton.jit  # type: ignore[untyped-decorator]
    def _mf_gbm_fused_f32_kernel(  # type: ignore[no-untyped-def]
        z_coarse_corr_ptr,  # (R, M, D, n_rep_in) or (M, D, R*n_rep_in) float32
        d_var_f32_ptr,      # (D, M) float32
        fwd_perf_f32_ptr,   # (D, M) float32
        df_pay_f32_ptr,     # (M,) float32
        w_fund_f32_ptr,     # (M,) float32 or None
        ko_bar_f32_ptr,     # (M,) float32
        is_call_f32_ptr,    # (M,) float32
        cpn_bar_f32_ptr,    # (M,) float32
        is_guar_f32_ptr,    # (M,) float32
        cpn_rates_f32_ptr,  # (M,) float32
        eps_vec_f32_ptr,    # (M,) float32
        inv_perm_ptr,       # (D,) int64
        scalar_f32_ptr,     # (15,) float32
        out_s1_ptr,         # (R, 3*M + 5) float64 pre-zeroed
        out_cf_surv_ptr,    # (R, 2, M) float64 pre-zeroed or None
        ema_shift_ptr,      # (n_sgn, R, M, D) float32 or None
        d_assets,
        m_rem,
        n_rep_in,
        z_scale,
        cg_flags_ptr=None,
        D_PAD: tl.constexpr = 8,
        CALL_CODE: tl.constexpr = 0,
        CPN_CODE: tl.constexpr = 0,
        EFF_OPT_CODE: tl.constexpr = 0,
        HAS_KI: tl.constexpr = False,
        KI_ACTIVE: tl.constexpr = False,
        HAS_ONE_STAR: tl.constexpr = False,
        IS_MEM: tl.constexpr = False,
        NO_NOTIONAL: tl.constexpr = False,
        FORCE_CPN_AC: tl.constexpr = False,
        CPN_DIR_DOWN: tl.constexpr = False,
        CPN_TYPE_OUT: tl.constexpr = False,
        BLOCK_N: tl.constexpr = 256,
        HAS_LOWER_STRIKE: tl.constexpr = True,
        HAS_PART_CALL: tl.constexpr = False,
        COMPUTE_CF_SURV: tl.constexpr = False,
        Z_SIGN: tl.constexpr = 1.0,
        OUT_SCALE: tl.constexpr = 1.0,
        ANTITHETIC: tl.constexpr = False,
        APPLY_EMA: tl.constexpr = False,
        LAYOUT_MDN: tl.constexpr = False,
        IS_LONG: tl.constexpr = False,
        USE_RT_FLAGS: tl.constexpr = False,
    ):
        pid_r = tl.program_id(0)
        n_r = tl.num_programs(0)
        pid_t = tl.program_id(1)
        offs_n = pid_t * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_rep_in
        n_tot_z = n_r * n_rep_in

        if USE_RT_FLAGS:
            d_act = tl.load(cg_flags_ptr + 0)
            m_act = tl.load(cg_flags_ptr + 1)
            rt_call = tl.load(cg_flags_ptr + 2)
            rt_cpn = tl.load(cg_flags_ptr + 3)
            rt_opt = tl.load(cg_flags_ptr + 4)
            f_has_ki = tl.load(cg_flags_ptr + 5) != 0
            f_ki_act = tl.load(cg_flags_ptr + 6) != 0
            f_has_os = tl.load(cg_flags_ptr + 7) != 0
            f_is_mem = tl.load(cg_flags_ptr + 8) != 0
            f_is_long = tl.load(cg_flags_ptr + 9) != 0
            f_no_not = tl.load(cg_flags_ptr + 10) != 0
            f_frc_cpn = tl.load(cg_flags_ptr + 11) != 0
            f_cpn_dn = tl.load(cg_flags_ptr + 12) != 0
            f_cpn_out = tl.load(cg_flags_ptr + 13) != 0
            f_has_low = tl.load(cg_flags_ptr + 14) != 0
            f_has_pc = tl.load(cg_flags_ptr + 15) != 0
        else:
            d_act = d_assets
            m_act = m_rem
            rt_call = CALL_CODE
            rt_cpn = CPN_CODE
            rt_opt = EFF_OPT_CODE
            f_has_ki = HAS_KI
            f_ki_act = KI_ACTIVE
            f_has_os = HAS_ONE_STAR
            f_is_mem = IS_MEM
            f_is_long = IS_LONG
            f_no_not = NO_NOTIONAL
            f_frc_cpn = FORCE_CPN_AC
            f_cpn_dn = CPN_DIR_DOWN
            f_cpn_out = CPN_TYPE_OUT
            f_has_low = HAS_LOWER_STRIKE
            f_has_pc = HAS_PART_CALL

        offs_d = tl.arange(0, D_PAD)[:, None]
        mask_d = offs_d < d_act

        ko_spread = tl.load(scalar_f32_ptr + 0)
        cpn_spread = tl.load(scalar_f32_ptr + 1)
        ki_spread = tl.load(scalar_f32_ptr + 2)
        one_star_strike = tl.load(scalar_f32_ptr + 3)
        one_star_eps = tl.load(scalar_f32_ptr + 4)
        strike = tl.load(scalar_f32_ptr + 5)
        lower_strike = tl.load(scalar_f32_ptr + 6)
        ki_barrier = tl.load(scalar_f32_ptr + 7)
        option_gearing = tl.load(scalar_f32_ptr + 8)
        ac_cpn = tl.load(scalar_f32_ptr + 9)
        c_mem_0 = tl.load(scalar_f32_ptr + 10)
        i_ki_past = tl.load(scalar_f32_ptr + 11)
        part_call_strike = tl.load(scalar_f32_ptr + 12)
        part_call_rate = tl.load(scalar_f32_ptr + 13)
        one_star_coupon = tl.load(scalar_f32_ptr + 14)

        ia = tl.load(inv_perm_ptr + 0).to(tl.int32)
        ib_idx = tl.where(d_act > 1, 1, 0)
        ib = tl.load(inv_perm_ptr + ib_idx).to(tl.int32)

        zero_vec = tl.zeros((BLOCK_N,), dtype=tl.float32)
        one_vec = tl.full((BLOCK_N,), 1.0, dtype=tl.float32)
        inv_d_f = 1.0 / tl.maximum(tl.zeros((), dtype=tl.float32) + d_act, 1.0)

        need_min2_k: tl.constexpr = False if USE_RT_FLAGS else ((CALL_CODE == 15) or (CPN_CODE == 15) or (EFF_OPT_CODE == 15))
        need_min3_k: tl.constexpr = False if USE_RT_FLAGS else ((CALL_CODE == 16) or (CPN_CODE == 16) or (EFF_OPT_CODE == 16))
        need_max2_k: tl.constexpr = False if USE_RT_FLAGS else ((CALL_CODE == 17) or (CPN_CODE == 17) or (EFF_OPT_CODE == 17))
        need_sum2_k: tl.constexpr = False if USE_RT_FLAGS else ((CALL_CODE == 3) or (CPN_CODE == 3) or (EFF_OPT_CODE == 3))
        need_ratio_k: tl.constexpr = False if USE_RT_FLAGS else (
            (CALL_CODE == 4) or (CALL_CODE == 5) or (CALL_CODE == 9) or (CALL_CODE == 10)
            or (CPN_CODE == 4) or (CPN_CODE == 5) or (CPN_CODE == 9) or (CPN_CODE == 10)
            or (EFF_OPT_CODE == 4) or (EFF_OPT_CODE == 5) or (EFF_OPT_CODE == 9) or (EFF_OPT_CODE == 10)
        )

        hard_ko_sp = tl.abs(ko_spread) > 1e-6
        eff_csp = tl.where(f_cpn_dn, -cpn_spread, cpn_spread) if USE_RT_FLAGS else (-cpn_spread if CPN_DIR_DOWN else cpn_spread)
        hard_cpn_sp = tl.abs(eff_csp) > 1e-6
        inv_w_ko_s1 = 1.0 / tl.maximum(tl.abs(ko_spread), 1e-12)
        inv_w_cp_s1 = 1.0 / tl.maximum(tl.abs(eff_csp), 1e-12)
        n_s1_cols = 3 * m_rem + 5
        m_last = m_act - 1
        df_last = tl.load(df_pay_f32_ptr + m_last)
        eps_term = tl.load(eps_vec_f32_ptr + m_last)
        n_sgn_f: tl.constexpr = 2 if ANTITHETIC else 1

        for sgn_idx in tl.static_range(n_sgn_f):
            z_sign_cur = ((1.0 if sgn_idx == 0 else -1.0) if ANTITHETIC else Z_SIGN) * z_scale
            eff_scale_cur = (0.5 if ANTITHETIC else 1.0) * OUT_SCALE
            inv_rep_f64 = eff_scale_cur / tl.maximum(tl.zeros((), dtype=tl.float64) + n_rep_in, 1.0)

            lx = tl.zeros((D_PAD, BLOCK_N), dtype=tl.float32)
            surv_s1 = one_vec
            c_mem_s1 = tl.full((BLOCK_N,), c_mem_0, dtype=tl.float32)
            pv_s1 = zero_vec
            surv_sm = one_vec
            c_mem_sm = tl.full((BLOCK_N,), c_mem_0, dtype=tl.float32)
            last_wo = zero_vec
            last_avg = zero_vec
            last_wb = zero_vec
            last_cf_sm = zero_vec

            for m in range(m_rem):
                m_ok_f = (m < m_act) if USE_RT_FLAGS else True
                tile_alive = tl.max(tl.where(mask_n, surv_s1, zero_vec), axis=0) > 1e-24
                if COMPUTE_CF_SURV:
                    tile_alive = tile_alive or (tl.max(tl.where(mask_n, surv_sm, zero_vec), axis=0) > 1e-24)
                if m_ok_f & tile_alive:
                    dv = tl.load(d_var_f32_ptr + offs_d * m_rem + m, mask=mask_d, other=0.0)
                    sdv = tl.sqrt(tl.maximum(dv, 0.0))
                    if LAYOUT_MDN:
                        z_off = (m * d_assets + offs_d) * n_tot_z + pid_r * n_rep_in + offs_n[None, :]
                    else:
                        z_off = ((pid_r * m_rem + m) * d_assets + offs_d) * n_rep_in + offs_n[None, :]
                    z_val = tl.load(z_coarse_corr_ptr + z_off, mask=mask_d & mask_n[None, :], other=0.0) * z_sign_cur
                    if APPLY_EMA and (ema_shift_ptr is not None):
                        sh_ema = tl.load(
                            ema_shift_ptr + ((sgn_idx * n_r + pid_r) * m_rem + m) * d_assets + offs_d,
                            mask=mask_d,
                            other=0.0,
                        )
                        lx = lx - 0.5 * dv + sdv * z_val - sh_ema
                    else:
                        lx = lx - 0.5 * dv + sdv * z_val
                    fp = tl.load(fwd_perf_f32_ptr + offs_d * m_rem + m, mask=mask_d, other=1.0)
                    p_mat = fp * tl.exp(lx)

                    p_min_src = tl.where(mask_d, p_mat, 1.0e30)
                    min1 = tl.min(p_min_src, axis=0)
                    min2 = zero_vec
                    min3 = zero_vec
                    if need_min2_k or need_min3_k:
                        idx_m1 = tl.argmin(p_min_src, axis=0)
                        p_min2_src = tl.where(mask_d & (offs_d != idx_m1[None, :]), p_mat, 1.0e30)
                        min2 = tl.min(p_min2_src, axis=0)
                        if need_min3_k:
                            idx_m2 = tl.argmin(p_min2_src, axis=0)
                            min3 = tl.min(
                                tl.where(mask_d & (offs_d != idx_m1[None, :]) & (offs_d != idx_m2[None, :]), p_mat, 1.0e30),
                                axis=0,
                            )

                    p_max_src = tl.where(mask_d, p_mat, -1.0e30)
                    max1 = tl.max(p_max_src, axis=0)
                    max2 = zero_vec
                    if need_max2_k:
                        idx_x1 = tl.argmax(p_max_src, axis=0)
                        max2 = tl.max(tl.where(mask_d & (offs_d != idx_x1[None, :]), p_mat, -1.0e30), axis=0)

                    s1 = tl.sum(tl.where(mask_d, p_mat, 0.0), axis=0)
                    s2 = tl.sum(tl.where(mask_d, p_mat * p_mat, 0.0), axis=0) if need_sum2_k else zero_vec
                    pa = tl.sum(tl.where(mask_d & (offs_d == ia), p_mat, 0.0), axis=0) if need_ratio_k else zero_vec
                    pb = tl.sum(tl.where(mask_d & (offs_d == ib), p_mat, 0.0), axis=0) if need_ratio_k else one_vec

                    if USE_RT_FLAGS:
                        w_avg_mf = s1 * inv_d_f
                        w_call = tl.where(rt_call == 0, min1, tl.where(rt_call == 1, max1, w_avg_mf))
                        w_cpn = tl.where(rt_cpn == 0, min1, tl.where(rt_cpn == 1, max1, w_avg_mf))
                        is_last_mf = m == m_last
                        wo_cand_mf = tl.where(f_is_long, min1, tl.where(rt_opt == 0, min1, tl.where(rt_opt == 1, max1, w_avg_mf)))
                        last_wo = tl.where(is_last_mf, wo_cand_mf, last_wo)
                        last_avg = tl.where(is_last_mf & f_is_long, w_avg_mf, last_avg)
                        last_wb = tl.where(is_last_mf & f_has_os, max1, last_wb)
                    else:
                        w_call = _select_idx_from_stats_f64(min1, min2, min3, max1, max2, s1, s2, pa, pb, inv_d_f, d_assets, CALL_CODE)
                        w_cpn = w_call if CPN_CODE == CALL_CODE else _select_idx_from_stats_f64(
                            min1, min2, min3, max1, max2, s1, s2, pa, pb, inv_d_f, d_assets, CPN_CODE
                        )
                        if m == m_rem - 1:
                            last_wo = min1 if IS_LONG else _select_idx_from_stats_f64(
                                min1, min2, min3, max1, max2, s1, s2, pa, pb, inv_d_f, d_assets, EFF_OPT_CODE
                            )
                            last_avg = (s1 * inv_d_f) if IS_LONG else zero_vec
                            last_wb = max1 if HAS_ONE_STAR else zero_vec

                    b_ko = tl.load(ko_bar_f32_ptr + m)
                    is_call_m = tl.load(is_call_f32_ptr + m) > 0.5
                    gk_s1 = tl.where(
                        is_call_m,
                        tl.where(
                            hard_ko_sp,
                            tl.minimum(tl.maximum((w_call - tl.minimum(b_ko, b_ko + ko_spread)) * inv_w_ko_s1, zero_vec), one_vec),
                            tl.where(w_call >= b_ko, one_vec, zero_vec),
                        ),
                        zero_vec,
                    )
                    b_cp = tl.load(cpn_bar_f32_ptr + m)
                    is_guar_m = tl.load(is_guar_f32_ptr + m) > 0.5
                    if USE_RT_FLAGS:
                        hit_h_s1 = tl.where(f_cpn_dn, w_cpn <= b_cp, w_cpn >= b_cp)
                        gc_hard_s1 = tl.where(f_cpn_out, tl.where(hit_h_s1, zero_vec, one_vec), tl.where(hit_h_s1, one_vec, zero_vec))
                    else:
                        hit_h_s1 = (w_cpn <= b_cp) if CPN_DIR_DOWN else (w_cpn >= b_cp)
                        gc_hard_s1 = tl.where(hit_h_s1, zero_vec if CPN_TYPE_OUT else one_vec, one_vec if CPN_TYPE_OUT else zero_vec)
                    blo_cp_s1 = tl.minimum(b_cp, b_cp + eff_csp)
                    r_up_s1 = tl.minimum(tl.maximum((w_cpn - blo_cp_s1) * inv_w_cp_s1, zero_vec), one_vec)
                    if USE_RT_FLAGS:
                        hit_s1 = tl.where(f_cpn_dn, 1.0 - r_up_s1, r_up_s1)
                        gc_soft_s1 = tl.where(f_cpn_out, 1.0 - hit_s1, hit_s1)
                        gc_s1 = tl.where(is_guar_m, one_vec, tl.where(hard_cpn_sp, gc_soft_s1, gc_hard_s1))
                        eff_gc_s1 = tl.where(f_frc_cpn, tl.maximum(gc_s1, gk_s1), gc_s1)
                    else:
                        hit_s1 = (1.0 - r_up_s1) if CPN_DIR_DOWN else r_up_s1
                        gc_soft_s1 = (1.0 - hit_s1) if CPN_TYPE_OUT else hit_s1
                        gc_s1 = tl.where(is_guar_m, one_vec, tl.where(hard_cpn_sp, gc_soft_s1, gc_hard_s1))
                        eff_gc_s1 = tl.maximum(gc_s1, gk_s1) if FORCE_CPN_AC else gc_s1

                    c_rate = tl.load(cpn_rates_f32_ptr + m)
                    pm_s1 = surv_s1 * gk_s1
                    eff_c_s1 = (tl.where(f_is_mem, c_rate + c_mem_s1, zero_vec + c_rate) if USE_RT_FLAGS else ((c_rate + c_mem_s1) if IS_MEM else (zero_vec + c_rate)))
                    cpn_cf_s1 = surv_s1 * eff_gc_s1 * eff_c_s1 + pm_s1 * ac_cpn
                    if USE_RT_FLAGS:
                        c_mem_s1 = tl.where(f_is_mem, surv_s1 * eff_c_s1 * (1.0 - eff_gc_s1), c_mem_s1)
                    elif IS_MEM:
                        c_mem_s1 = surv_s1 * eff_c_s1 * (1.0 - eff_gc_s1)
                    df_m = tl.load(df_pay_f32_ptr + m)
                    wf_m = tl.load(w_fund_f32_ptr + m) if w_fund_f32_ptr is not None else 0.0
                    if USE_RT_FLAGS:
                        pv_s1 = pv_s1 + df_m * tl.where(f_no_not, cpn_cf_s1, pm_s1 + cpn_cf_s1) - wf_m * surv_s1
                    elif NO_NOTIONAL:
                        pv_s1 = pv_s1 + df_m * cpn_cf_s1 - wf_m * surv_s1
                    else:
                        pv_s1 = pv_s1 + df_m * (pm_s1 + cpn_cf_s1) - wf_m * surv_s1
                    tl.atomic_add(
                        out_s1_ptr + pid_r * n_s1_cols + m,
                        tl.sum(tl.where(mask_n, pm_s1, zero_vec), axis=0).to(tl.float64) * inv_rep_f64,
                    )
                    tl.atomic_add(
                        out_s1_ptr + pid_r * n_s1_cols + m_rem + m,
                        tl.sum(tl.where(mask_n, surv_s1 * eff_gc_s1, zero_vec), axis=0).to(tl.float64) * inv_rep_f64,
                    )
                    tl.atomic_add(
                        out_s1_ptr + pid_r * n_s1_cols + 2 * m_rem + m,
                        tl.sum(tl.where(mask_n, cpn_cf_s1, zero_vec), axis=0).to(tl.float64) * inv_rep_f64,
                    )
                    surv_s1 = tl.maximum(surv_s1 - pm_s1, zero_vec)

                    if COMPUTE_CF_SURV and (out_cf_surv_ptr is not None):
                        eps_m = tl.load(eps_vec_f32_ptr + m)
                        half_eps = 0.5 * eps_m
                        lo_ko = tl.where(hard_ko_sp, tl.minimum(b_ko, b_ko + ko_spread), b_ko - half_eps)
                        hi_ko = tl.where(hard_ko_sp, tl.maximum(b_ko, b_ko + ko_spread), b_ko + half_eps)
                        u_ko = (w_call - lo_ko) / tl.maximum(hi_ko - lo_ko, 1e-12)
                        g_k_sm = tl.where(is_call_m, tl.minimum(tl.maximum(u_ko, zero_vec), one_vec), zero_vec)
                        lo_cp = tl.where(hard_cpn_sp, tl.minimum(b_cp, b_cp + eff_csp), b_cp - half_eps)
                        hi_cp = tl.where(hard_cpn_sp, tl.maximum(b_cp, b_cp + eff_csp), b_cp + half_eps)
                        ramp_cp = tl.minimum(tl.maximum((w_cpn - lo_cp) / tl.maximum(hi_cp - lo_cp, 1e-12), zero_vec), one_vec)
                        if USE_RT_FLAGS:
                            hit_cp = tl.where(f_cpn_dn, 1.0 - ramp_cp, ramp_cp)
                            ind_cp = tl.where(f_cpn_out, 1.0 - hit_cp, hit_cp)
                            g_c_sm = tl.where(is_guar_m, one_vec, ind_cp)
                            eff_gc_sm = tl.where(f_frc_cpn, tl.maximum(g_c_sm, g_k_sm), g_c_sm)
                        else:
                            hit_cp = (1.0 - ramp_cp) if CPN_DIR_DOWN else ramp_cp
                            ind_cp = (1.0 - hit_cp) if CPN_TYPE_OUT else hit_cp
                            g_c_sm = tl.where(is_guar_m, one_vec, ind_cp)
                            eff_gc_sm = tl.maximum(g_c_sm, g_k_sm) if FORCE_CPN_AC else g_c_sm
                        p_m_sm = surv_sm * g_k_sm
                        eff_c_sm = (tl.where(f_is_mem, c_rate + c_mem_sm, zero_vec + c_rate) if USE_RT_FLAGS else ((c_rate + c_mem_sm) if IS_MEM else (zero_vec + c_rate)))
                        cpn_sm = surv_sm * eff_gc_sm * eff_c_sm + p_m_sm * ac_cpn
                        cf_sm = (tl.where(f_no_not, cpn_sm, p_m_sm + cpn_sm) if USE_RT_FLAGS else (cpn_sm if NO_NOTIONAL else (p_m_sm + cpn_sm)))
                        tl.atomic_add(
                            out_cf_surv_ptr + (pid_r * 2 + 1) * m_rem + m,
                            tl.sum(tl.where(mask_n, surv_sm, zero_vec), axis=0).to(tl.float64) * inv_rep_f64,
                        )
                        if m == m_last:
                            last_cf_sm = cf_sm
                        else:
                            tl.atomic_add(
                                out_cf_surv_ptr + (pid_r * 2 + 0) * m_rem + m,
                                tl.sum(tl.where(mask_n, cf_sm, zero_vec), axis=0).to(tl.float64) * inv_rep_f64,
                            )
                        if USE_RT_FLAGS:
                            c_mem_sm = tl.where(f_is_mem, surv_sm * eff_c_sm * (1.0 - eff_gc_sm), c_mem_sm)
                        elif IS_MEM:
                            c_mem_sm = surv_sm * eff_c_sm * (1.0 - eff_gc_sm)
                        surv_sm = tl.maximum(surv_sm - p_m_sm, zero_vec)

            if USE_RT_FLAGS:
                sp_ki_pos = tl.abs(ki_spread) > 1e-6
                b_ki_mf = tl.where(f_is_long, strike, ki_barrier)
                blo_k = tl.minimum(b_ki_mf, b_ki_mf + ki_spread)
                bhi_k = tl.where(sp_ki_pos, tl.maximum(b_ki_mf, b_ki_mf + ki_spread), b_ki_mf + 0.5 * eps_term)
                inv_wk = 1.0 / tl.maximum(tl.where(sp_ki_pos, bhi_k - blo_k, eps_term), 1e-12)
                i_ki_act = tl.minimum(tl.maximum((bhi_k - last_wo) * inv_wk, zero_vec), one_vec)
                ki_off = tl.where(f_has_ki, zero_vec + i_ki_past, one_vec)
                i_ki = tl.where(f_ki_act, i_ki_act, ki_off)
                ki_gate_s1 = tl.where(f_ki_act, tl.where(sp_ki_pos, i_ki_act, tl.where(last_wo < b_ki_mf, one_vec, zero_vec)), ki_off)
            elif KI_ACTIVE:
                sp_ki_pos = tl.abs(ki_spread) > 1e-6
                if IS_LONG:
                    blo_k = tl.minimum(strike, strike + ki_spread)
                    bhi_k = tl.where(sp_ki_pos, tl.maximum(strike, strike + ki_spread), strike + 0.5 * eps_term)
                    inv_wk = 1.0 / tl.maximum(tl.where(sp_ki_pos, bhi_k - blo_k, eps_term), 1e-12)
                    i_ki = tl.minimum(tl.maximum((bhi_k - last_wo) * inv_wk, zero_vec), one_vec)
                    ki_gate_s1 = i_ki if sp_ki_pos else tl.where(last_wo < strike, one_vec, zero_vec)
                else:
                    blo_k = tl.minimum(ki_barrier, ki_barrier + ki_spread)
                    bhi_k = tl.where(sp_ki_pos, tl.maximum(ki_barrier, ki_barrier + ki_spread), ki_barrier + 0.5 * eps_term)
                    inv_wk = 1.0 / tl.maximum(tl.where(sp_ki_pos, bhi_k - blo_k, eps_term), 1e-12)
                    i_ki = tl.minimum(tl.maximum((bhi_k - last_wo) * inv_wk, zero_vec), one_vec)
                    ki_gate_s1 = i_ki if sp_ki_pos else tl.where(last_wo < ki_barrier, one_vec, zero_vec)
            else:
                i_ki = (zero_vec + i_ki_past) if HAS_KI else one_vec
                ki_gate_s1 = i_ki

            if USE_RT_FLAGS:
                os_soft = one_star_eps > 1e-6
                inv_wos = 1.0 / tl.maximum(one_star_eps, 1e-12)
                u_os = (last_wb - (one_star_strike - 0.5 * one_star_eps)) * inv_wos
                star_ind = tl.where(f_has_os, tl.where(os_soft, tl.minimum(tl.maximum(u_os, zero_vec), one_vec), tl.where(last_wb >= one_star_strike, one_vec, zero_vec)), zero_vec)
            elif HAS_ONE_STAR:
                os_soft = one_star_eps > 1e-6
                inv_wos = 1.0 / tl.maximum(one_star_eps, 1e-12)
                u_os = (last_wb - (one_star_strike - 0.5 * one_star_eps)) * inv_wos
                star_ind = tl.where(
                    os_soft,
                    tl.minimum(tl.maximum(u_os, zero_vec), one_vec),
                    tl.where(last_wb >= one_star_strike, one_vec, zero_vec),
                )
            else:
                star_ind = zero_vec

            has_pc = (f_has_pc & (part_call_rate != 0.0)) if USE_RT_FLAGS else ((part_call_rate != 0.0) if HAS_PART_CALL else False)
            call_intr = tl.maximum(last_wo - part_call_strike, zero_vec)
            os_cpn_mask = (f_has_os & (one_star_coupon != 0.0)) if USE_RT_FLAGS else ((one_star_coupon != 0.0) if HAS_ONE_STAR else False)

            if USE_RT_FLAGS:
                u_put_long = surv_s1 * ki_gate_s1 * (1.0 - star_ind) * (1.0 - last_avg)
                put_intr_s = tl.where(f_has_low, tl.maximum(strike - last_wo, zero_vec) - tl.maximum(lower_strike - last_wo, zero_vec), tl.maximum(strike - last_wo, zero_vec))
                put_mask_s = tl.where(f_has_low, (strike > lower_strike) & (option_gearing > 0.0), option_gearing > 0.0)
                u_put_short = tl.where(put_mask_s, surv_s1 * ki_gate_s1 * (1.0 - star_ind) * put_intr_s, zero_vec)
                u_put_s1 = tl.where(f_is_long, u_put_long, u_put_short)
                put_term_s1 = tl.where(f_is_long, -u_put_long, -option_gearing * u_put_short)
                put_mask = tl.where(f_is_long, zero_vec > 1.0, put_mask_s)
                put_intr = tl.where(f_is_long, zero_vec, put_intr_s)
                u_call_s1 = tl.where(has_pc, surv_s1 * call_intr, zero_vec)
                p_os_s1 = tl.where(f_has_os, surv_s1 * star_ind, zero_vec)
                l_os_s1 = tl.where(os_cpn_mask, p_os_s1 * one_star_coupon, zero_vec)
                opt_s1 = put_term_s1 + tl.where(has_pc, part_call_rate * u_call_s1, zero_vec) + l_os_s1
                term_s1 = (surv_s1 * tl.where(f_no_not, 0.0, 1.0)) + opt_s1
            else:
                if IS_LONG:
                    u_put_s1 = surv_s1 * ki_gate_s1 * (1.0 - star_ind) * (1.0 - last_avg)
                    put_term_s1 = -u_put_s1
                    put_mask = zero_vec > 1.0
                    put_intr = zero_vec
                else:
                    if HAS_LOWER_STRIKE:
                        put_mask = (strike > lower_strike) & (option_gearing > 0.0)
                        put_intr = tl.maximum(strike - last_wo, zero_vec) - tl.maximum(lower_strike - last_wo, zero_vec)
                    else:
                        put_mask = option_gearing > 0.0
                        put_intr = tl.maximum(strike - last_wo, zero_vec)
                    u_put_s1 = tl.where(put_mask, surv_s1 * ki_gate_s1 * (1.0 - star_ind) * put_intr, zero_vec)
                    put_term_s1 = -option_gearing * u_put_s1

                u_call_s1 = tl.where(has_pc, surv_s1 * call_intr, zero_vec)
                p_os_s1 = tl.where(HAS_ONE_STAR, surv_s1 * star_ind, zero_vec)
                l_os_s1 = tl.where(os_cpn_mask, p_os_s1 * one_star_coupon, zero_vec)
                opt_s1 = put_term_s1 + (part_call_rate * u_call_s1 if has_pc else zero_vec) + l_os_s1
                term_s1 = (surv_s1 * (0.0 if NO_NOTIONAL else 1.0)) + opt_s1
            pv_s1 = pv_s1 + df_last * term_s1

            base_c = pid_r * n_s1_cols + 3 * m_rem
            tl.atomic_add(out_s1_ptr + base_c + 0, tl.sum(tl.where(mask_n, u_put_s1, zero_vec), axis=0).to(tl.float64) * inv_rep_f64)
            tl.atomic_add(out_s1_ptr + base_c + 1, tl.sum(tl.where(mask_n, u_call_s1, zero_vec), axis=0).to(tl.float64) * inv_rep_f64)
            tl.atomic_add(out_s1_ptr + base_c + 2, tl.sum(tl.where(mask_n, l_os_s1, zero_vec), axis=0).to(tl.float64) * inv_rep_f64)
            tl.atomic_add(out_s1_ptr + base_c + 3, tl.sum(tl.where(mask_n, p_os_s1, zero_vec), axis=0).to(tl.float64) * inv_rep_f64)
            tl.atomic_add(out_s1_ptr + base_c + 4, tl.sum(tl.where(mask_n, pv_s1, zero_vec), axis=0).to(tl.float64) * inv_rep_f64)

            if COMPUTE_CF_SURV and (out_cf_surv_ptr is not None):
                opt_sm = zero_vec
                if USE_RT_FLAGS:
                    opt_sm = tl.where(f_is_long, opt_sm - (1.0 - last_avg) * i_ki * (1.0 - star_ind), tl.where(put_mask, opt_sm - option_gearing * i_ki * (1.0 - star_ind) * put_intr, opt_sm))
                elif IS_LONG:
                    opt_sm = opt_sm - (1.0 - last_avg) * i_ki * (1.0 - star_ind)
                else:
                    opt_sm = tl.where(put_mask, opt_sm - option_gearing * i_ki * (1.0 - star_ind) * put_intr, opt_sm)
                opt_sm = tl.where(has_pc, opt_sm + part_call_rate * call_intr, opt_sm)
                opt_sm = tl.where(os_cpn_mask, opt_sm + star_ind * one_star_coupon, opt_sm)
                cf_tail = surv_sm * (tl.where(f_no_not, opt_sm, 1.0 + opt_sm) if USE_RT_FLAGS else (opt_sm if NO_NOTIONAL else (1.0 + opt_sm)))
                tl.atomic_add(
                    out_cf_surv_ptr + (pid_r * 2 + 0) * m_rem + m_last,
                    tl.sum(tl.where(mask_n, last_cf_sm + cf_tail, zero_vec), axis=0).to(tl.float64) * inv_rep_f64,
                )

    @triton.jit  # type: ignore[untyped-decorator]
    def _vibrato_gamma_f64_kernel(  # type: ignore[no-untyped-def]
        perf_ptr,
        lx_pre_ptr,
        j_pre_ptr,
        sig_pre_ptr,
        sv_ptr,
        chol_ptr,
        l_inv_t_ptr,
        c_inv_ptr,
        fwd_term_ptr,
        cdl_m_ptr,
        cdl2_last_ptr,
        ko_bar_ptr,
        is_call_ptr,
        cpn_bar_ptr,
        is_guar_ptr,
        cpn_rates_ptr,
        eps_vec_ptr,
        inv_perm_ptr,
        scalar_f64_ptr,
        d_ln_surv_tmp_ptr,
        frozen_term_out_ptr,
        h_vib_out_ptr,
        df_last,
        h_vib,
        vib_step_start,
        vib_n_steps,
        d_assets,
        m_rem,
        n_total,
        cg_flags_ptr=None,
        CALL_CODE: tl.constexpr = 0,
        CPN_CODE: tl.constexpr = 0,
        EFF_OPT_CODE: tl.constexpr = 0,
        HAS_KI: tl.constexpr = False,
        KI_ACTIVE: tl.constexpr = False,
        HAS_ONE_STAR: tl.constexpr = False,
        IS_MEM: tl.constexpr = False,
        NO_NOTIONAL: tl.constexpr = False,
        FORCE_CPN_AC: tl.constexpr = False,
        CPN_DIR_DOWN: tl.constexpr = False,
        CPN_TYPE_OUT: tl.constexpr = False,
        HAS_LOWER_STRIKE: tl.constexpr = True,
        HAS_PART_CALL: tl.constexpr = False,
        BLOCK_N: tl.constexpr = 128,
        USE_F32_PRE: tl.constexpr = False,
        IS_LONG: tl.constexpr = False,
        USE_RT_FLAGS: tl.constexpr = False,
        D_PAD: tl.constexpr = 0,
        N_SUB_PAD: tl.constexpr = 8,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_total
        zero_vec = tl.zeros((BLOCK_N,), dtype=tl.float64)
        one_vec = tl.full((BLOCK_N,), 1.0, dtype=tl.float64)
        scale = 1.0 / (tl.zeros((), dtype=tl.float64) + n_total)
        if USE_RT_FLAGS:
            d_act = tl.load(cg_flags_ptr + 0)
            m_act = tl.load(cg_flags_ptr + 1)
            rt_call = tl.load(cg_flags_ptr + 2)
            rt_cpn = tl.load(cg_flags_ptr + 3)
            rt_opt = tl.load(cg_flags_ptr + 4)
            f_has_ki = tl.load(cg_flags_ptr + 5) != 0
            f_ki_act = tl.load(cg_flags_ptr + 6) != 0
            f_has_os = tl.load(cg_flags_ptr + 7) != 0
            f_is_mem = tl.load(cg_flags_ptr + 8) != 0
            f_is_long = tl.load(cg_flags_ptr + 9) != 0
            f_no_not = tl.load(cg_flags_ptr + 10) != 0
            f_frc_cpn = tl.load(cg_flags_ptr + 11) != 0
            f_cpn_dn = tl.load(cg_flags_ptr + 12) != 0
            f_cpn_out = tl.load(cg_flags_ptr + 13) != 0
            f_has_low = tl.load(cg_flags_ptr + 14) != 0
            f_has_pc = tl.load(cg_flags_ptr + 15) != 0
            eff_vib_start = tl.load(cg_flags_ptr + 20)
            eff_vib_n = tl.load(cg_flags_ptr + 21)
            eff_df_last = tl.load(scalar_f64_ptr + 15)
            eff_h_vib = tl.load(scalar_f64_ptr + 16)
        else:
            d_act = d_assets
            m_act = m_rem
            rt_call = CALL_CODE
            rt_cpn = CPN_CODE
            rt_opt = EFF_OPT_CODE
            f_has_ki = HAS_KI
            f_ki_act = KI_ACTIVE
            f_has_os = HAS_ONE_STAR
            f_is_mem = IS_MEM
            f_is_long = IS_LONG
            f_no_not = NO_NOTIONAL
            f_frc_cpn = FORCE_CPN_AC
            f_cpn_dn = CPN_DIR_DOWN
            f_cpn_out = CPN_TYPE_OUT
            f_has_low = HAS_LOWER_STRIKE
            f_has_pc = HAS_PART_CALL
            eff_vib_start = vib_step_start
            eff_vib_n = vib_n_steps
            eff_df_last = df_last
            eff_h_vib = h_vib
        inv_d_f = 1.0 / tl.maximum(tl.zeros((), dtype=tl.float64) + d_act, 1.0)

        ko_spread = tl.load(scalar_f64_ptr + 0)
        cpn_spread = tl.load(scalar_f64_ptr + 1)
        ki_spread = tl.load(scalar_f64_ptr + 2)
        one_star_strike = tl.load(scalar_f64_ptr + 3)
        one_star_eps = tl.load(scalar_f64_ptr + 4)
        strike = tl.load(scalar_f64_ptr + 5)
        lower_strike = tl.load(scalar_f64_ptr + 6)
        ki_barrier = tl.load(scalar_f64_ptr + 7)
        option_gearing = tl.load(scalar_f64_ptr + 8)
        ac_cpn = tl.load(scalar_f64_ptr + 9)
        c_mem_0 = tl.load(scalar_f64_ptr + 10)
        i_ki_past = tl.load(scalar_f64_ptr + 11)
        part_call_strike = tl.load(scalar_f64_ptr + 12)
        part_call_rate = tl.load(scalar_f64_ptr + 13)
        one_star_coupon = tl.load(scalar_f64_ptr + 14)

        ia = tl.load(inv_perm_ptr + 0).to(tl.int32)
        ib_idx = tl.where(d_act > 1, 1, 0)
        ib = tl.load(inv_perm_ptr + ib_idx).to(tl.int32)

        hard_ko_sp = tl.abs(ko_spread) > 1e-6
        eff_csp = tl.where(f_cpn_dn, -cpn_spread, cpn_spread) if USE_RT_FLAGS else (-cpn_spread if CPN_DIR_DOWN else cpn_spread)
        hard_cpn_sp = tl.abs(eff_csp) > 1e-6
        hard_ki_sp = tl.abs(ki_spread) > 1e-6
        blo_ki_k = tl.minimum(ki_barrier, ki_barrier + ki_spread)
        bhi_ki_k = tl.maximum(ki_barrier, ki_barrier + ki_spread)
        inv_sp_ko = 1.0 / tl.maximum(tl.abs(ko_spread), 1e-12)
        inv_sp_cp = 1.0 / tl.maximum(tl.abs(eff_csp), 1e-12)
        inv_sp_ki = 1.0 / tl.maximum(bhi_ki_k - blo_ki_k, 1e-12)
        inv_sp_os = 1.0 / tl.maximum(one_star_eps, 1e-12)
        need_min1_k: tl.constexpr = (
            IS_LONG
            or (CALL_CODE == 0) or (CALL_CODE == 6) or (CALL_CODE == 8) or (CALL_CODE == 11) or (CALL_CODE == 13) or (CALL_CODE == 14)
            or (CPN_CODE == 0) or (CPN_CODE == 6) or (CPN_CODE == 8) or (CPN_CODE == 11) or (CPN_CODE == 13) or (CPN_CODE == 14)
            or (EFF_OPT_CODE == 0) or (EFF_OPT_CODE == 6) or (EFF_OPT_CODE == 8) or (EFF_OPT_CODE == 11) or (EFF_OPT_CODE == 13) or (EFF_OPT_CODE == 14)
        )
        need_min2_k: tl.constexpr = (CALL_CODE == 15) or (CPN_CODE == 15) or (EFF_OPT_CODE == 15)
        need_min3_k: tl.constexpr = (CALL_CODE == 16) or (CPN_CODE == 16) or (EFF_OPT_CODE == 16)
        need_max1_k: tl.constexpr = (
            HAS_ONE_STAR
            or (CALL_CODE == 1) or (CALL_CODE == 6) or (CALL_CODE == 7) or (CALL_CODE == 11) or (CALL_CODE == 12)
            or (CPN_CODE == 1) or (CPN_CODE == 6) or (CPN_CODE == 7) or (CPN_CODE == 11) or (CPN_CODE == 12)
            or (EFF_OPT_CODE == 1) or (EFF_OPT_CODE == 6) or (EFF_OPT_CODE == 7) or (EFF_OPT_CODE == 11) or (EFF_OPT_CODE == 12)
        )
        need_max2_k: tl.constexpr = (CALL_CODE == 17) or (CPN_CODE == 17) or (EFF_OPT_CODE == 17)
        need_sum1_k: tl.constexpr = (
            IS_LONG
            or (CALL_CODE == 2) or (CALL_CODE == 3) or (CALL_CODE == 7) or (CALL_CODE == 8) or (CALL_CODE == 12) or (CALL_CODE == 13)
            or (CPN_CODE == 2) or (CPN_CODE == 3) or (CPN_CODE == 7) or (CPN_CODE == 8) or (CPN_CODE == 12) or (CPN_CODE == 13)
            or (EFF_OPT_CODE == 2) or (EFF_OPT_CODE == 3) or (EFF_OPT_CODE == 7) or (EFF_OPT_CODE == 8) or (EFF_OPT_CODE == 12) or (EFF_OPT_CODE == 13)
        )
        need_sum2_k: tl.constexpr = (CALL_CODE == 3) or (CPN_CODE == 3) or (EFF_OPT_CODE == 3)
        need_ratio_k: tl.constexpr = (
            (CALL_CODE == 4) or (CALL_CODE == 5) or (CALL_CODE == 9) or (CALL_CODE == 10)
            or (CPN_CODE == 4) or (CPN_CODE == 5) or (CPN_CODE == 9) or (CPN_CODE == 10)
            or (EFF_OPT_CODE == 4) or (EFF_OPT_CODE == 5) or (EFF_OPT_CODE == 9) or (EFF_OPT_CODE == 10)
        )

        PRE_DTYPE: tl.constexpr = tl.float32 if USE_F32_PRE else tl.float64
        zero_p = tl.zeros((BLOCK_N,), dtype=PRE_DTYPE)
        one_p = tl.full((BLOCK_N,), 1.0, dtype=PRE_DTYPE)
        tiny_p = tl.zeros((), dtype=PRE_DTYPE) + 1e-12
        inv_d_p = 1.0 / tl.maximum(tl.zeros((), dtype=PRE_DTYPE) + d_act, 1.0)
        ko_sp_p = ko_spread.to(PRE_DTYPE)
        eff_csp_p = eff_csp.to(PRE_DTYPE)
        surv_pre_p = one_p
        c_mem_p = tl.full((BLOCK_N,), c_mem_0.to(PRE_DTYPE), dtype=PRE_DTYPE)
        m_last = m_act - 1
        n_init_d: tl.constexpr = (2 * D_PAD) if USE_RT_FLAGS else (2 * d_assets if IS_MEM else d_assets)
        for d_init in range(n_init_d):
            tl.store(d_ln_surv_tmp_ptr + d_init * n_total + offs_n, zero_p, mask=mask_n)

        for m in range(m_rem - 1):
            m_ok_p = (m < m_last) if USE_RT_FLAGS else True
            if m_ok_p & (tl.max(surv_pre_p, axis=0) > 1e-24):
                m_off = m * d_assets
                if USE_RT_FLAGS:
                    min1, idx_min1, max1, idx_max1, w_avg_p0, _ = _eval_idx_base_stats_012_f64(
                        perf_ptr, m_off, offs_n, mask_n, d_act, n_total, D_PAD, PRE_DTYPE
                    )
                    w_call = tl.where(rt_call == 0, min1, tl.where(rt_call == 1, max1, w_avg_p0))
                    w_cpn = tl.where(rt_cpn == 0, min1, tl.where(rt_cpn == 1, max1, w_avg_p0))
                else:
                    min1, min2, min3, idx_min1, idx_min2, idx_min3, max1, max2, idx_max1, idx_max2, s1, s2, pa_r, pb_r = _eval_idx_base_stats_f64(
                        perf_ptr, m_off, ia, ib, offs_n, mask_n, d_assets, n_total,
                        need_min1_k or need_min2_k or need_min3_k,
                        need_min2_k, need_min3_k,
                        need_max1_k or need_max2_k,
                        need_max2_k,
                        need_sum1_k or need_sum2_k,
                        need_sum2_k,
                        need_ratio_k,
                        PRE_DTYPE,
                    )
                    w_call = _select_idx_from_stats_f64(min1, min2, min3, max1, max2, s1, s2, pa_r, pb_r, inv_d_p, d_assets, CALL_CODE)
                    w_cpn = w_call if CPN_CODE == CALL_CODE else _select_idx_from_stats_f64(min1, min2, min3, max1, max2, s1, s2, pa_r, pb_r, inv_d_p, d_assets, CPN_CODE)

                eps_m = tl.load(eps_vec_ptr + m).to(PRE_DTYPE)
                b_ko = tl.load(ko_bar_ptr + m).to(PRE_DTYPE)
                lo_ko = tl.where(hard_ko_sp, tl.minimum(b_ko, b_ko + ko_sp_p), b_ko - 0.5 * eps_m)
                hi_ko = tl.where(hard_ko_sp, tl.maximum(b_ko, b_ko + ko_sp_p), b_ko + 0.5 * eps_m)
                inv_w_ko = 1.0 / tl.maximum(hi_ko - lo_ko, tiny_p)
                t_ko = (w_call - lo_ko) * inv_w_ko
                is_call_m = tl.load(is_call_ptr + m) > 0.5
                g_k = tl.where(is_call_m, _triton_c2_s5(t_ko), zero_p)
                dg_k_dw = tl.where(is_call_m, _triton_c2_ds5(t_ko) * inv_w_ko, zero_p)
                coef_surv_m = -dg_k_dw / tl.maximum(1.0 - g_k, tl.zeros((), dtype=PRE_DTYPE) + 1e-8)

                b_cp = tl.load(cpn_bar_ptr + m).to(PRE_DTYPE)
                lo_cp = tl.where(hard_cpn_sp, tl.minimum(b_cp, b_cp + eff_csp_p), b_cp - 0.5 * eps_m)
                hi_cp = tl.where(hard_cpn_sp, tl.maximum(b_cp, b_cp + eff_csp_p), b_cp + 0.5 * eps_m)
                inv_w_cp = 1.0 / tl.maximum(hi_cp - lo_cp, tiny_p)
                t_cp = (w_cpn - lo_cp) * inv_w_cp
                ramp_c2 = _triton_c2_s5(t_cp)
                dramp_c2_dw = _triton_c2_ds5(t_cp) * inv_w_cp
                if USE_RT_FLAGS:
                    hit_c2 = tl.where(f_cpn_dn, 1.0 - ramp_c2, ramp_c2)
                    ind_c2 = tl.where(f_cpn_out, 1.0 - hit_c2, hit_c2)
                    sign_cpn_k_rt = (tl.where(f_cpn_dn, -1.0, 1.0) * tl.where(f_cpn_out, -1.0, 1.0)).to(PRE_DTYPE)
                else:
                    hit_c2 = (1.0 - ramp_c2) if CPN_DIR_DOWN else ramp_c2
                    ind_c2 = (1.0 - hit_c2) if CPN_TYPE_OUT else hit_c2
                    sign_cpn_k: tl.constexpr = (-1.0 if CPN_DIR_DOWN else 1.0) * (-1.0 if CPN_TYPE_OUT else 1.0)
                    sign_cpn_k_rt = tl.zeros((), dtype=PRE_DTYPE) + sign_cpn_k
                is_guar_m = tl.load(is_guar_ptr + m) > 0.5
                g_c = tl.where(is_guar_m, one_p, ind_c2)
                dg_c_dw = tl.where(is_guar_m, zero_p, dramp_c2_dw * sign_cpn_k_rt)
                if USE_RT_FLAGS:
                    w_c_max_m = tl.where(f_frc_cpn, tl.where(g_c > g_k, one_p, tl.where(g_c < g_k, zero_p, 0.5 * one_p)), one_p)
                    w_k_max_m = 1.0 - w_c_max_m
                    eff_gc = tl.where(f_frc_cpn, tl.maximum(g_c, g_k), g_c)
                elif FORCE_CPN_AC:
                    eff_gc = tl.maximum(g_c, g_k)
                    w_c_max_m = tl.where(g_c > g_k, one_p, tl.where(g_c < g_k, zero_p, 0.5 * one_p))
                    w_k_max_m = 1.0 - w_c_max_m
                else:
                    eff_gc = g_c
                    w_c_max_m = one_p
                    w_k_max_m = zero_p
                c_rate = tl.load(cpn_rates_ptr + m).to(PRE_DTYPE)
                eff_c_m = (tl.where(f_is_mem, c_rate + c_mem_p, zero_p) if USE_RT_FLAGS else ((c_rate + c_mem_p) if IS_MEM else zero_p))
                if USE_RT_FLAGS:
                    c_mem_p = tl.where(f_is_mem, eff_c_m * (1.0 - eff_gc), c_mem_p)
                elif IS_MEM:
                    c_mem_p = eff_c_m * (1.0 - eff_gc)
                surv_pre_p = tl.maximum(surv_pre_p - surv_pre_p * g_k, zero_p)

                if USE_RT_FLAGS:
                    for d in range(D_PAD):
                        if d < d_act:
                            pd = tl.load(perf_ptr + (m_off + d) * n_total + offs_n, mask=mask_n, other=1.0).to(PRE_DTYPE)
                            cdl_md = tl.load(cdl_m_ptr + m * d_assets + d).to(PRE_DTYPE)
                            bp_d = tl.where(rt_call == 0, tl.where(d == idx_min1, coef_surv_m, zero_p), tl.where(rt_call == 1, tl.where(d == idx_max1, coef_surv_m, zero_p), coef_surv_m * inv_d_p))
                            prev_dlns = tl.load(d_ln_surv_tmp_ptr + d * n_total + offs_n, mask=mask_n, other=0.0).to(PRE_DTYPE)
                            tl.store(d_ln_surv_tmp_ptr + d * n_total + offs_n, prev_dlns + bp_d * pd * cdl_md, mask=mask_n)
                            if f_is_mem:
                                w_c_cm = w_c_max_m * dg_c_dw
                                w_k_cm = w_k_max_m * dg_k_dw
                                bp_cm = (
                                    tl.where(rt_cpn == 0, tl.where(d == idx_min1, w_c_cm, zero_p), tl.where(rt_cpn == 1, tl.where(d == idx_max1, w_c_cm, zero_p), w_c_cm * inv_d_p))
                                    + tl.where(rt_call == 0, tl.where(d == idx_min1, w_k_cm, zero_p), tl.where(rt_call == 1, tl.where(d == idx_max1, w_k_cm, zero_p), w_k_cm * inv_d_p))
                                )
                                d_eff_gc_d = bp_cm * pd * cdl_md
                                prev_dcm = tl.load(d_ln_surv_tmp_ptr + (d_assets + d) * n_total + offs_n, mask=mask_n, other=0.0).to(PRE_DTYPE)
                                tl.store(
                                    d_ln_surv_tmp_ptr + (d_assets + d) * n_total + offs_n,
                                    prev_dcm * (1.0 - eff_gc) - eff_c_m * d_eff_gc_d,
                                    mask=mask_n,
                                )
                else:
                    w_avg_v = s1 * inv_d_p
                    w_disp_v = tl.sqrt(tl.maximum(s2 * inv_d_p - w_avg_v * w_avg_v, 0.0))
                    for d in range(d_assets):
                        pd = tl.load(perf_ptr + (m_off + d) * n_total + offs_n, mask=mask_n, other=1.0).to(PRE_DTYPE)
                        cdl_md = tl.load(cdl_m_ptr + m * d_assets + d).to(PRE_DTYPE)
                        bp_d = _accum_idx_grad_f64(
                            zero_p, coef_surv_m, pd, d, d_assets, inv_d_p, ia, ib,
                            min1, max1, w_avg_v, w_disp_v, idx_min1, idx_min2, idx_min3, idx_max1, idx_max2, pa_r, pb_r,
                            CALL_CODE,
                        )
                        prev_dlns = tl.load(d_ln_surv_tmp_ptr + d * n_total + offs_n, mask=mask_n, other=0.0).to(PRE_DTYPE)
                        tl.store(d_ln_surv_tmp_ptr + d * n_total + offs_n, prev_dlns + bp_d * pd * cdl_md, mask=mask_n)
                        if IS_MEM:
                            if CPN_CODE == CALL_CODE:
                                bar_wc_cm = (w_c_max_m * dg_c_dw + w_k_max_m * dg_k_dw) if FORCE_CPN_AC else dg_c_dw
                                bp_cm = _accum_idx_grad_f64(
                                    zero_p, bar_wc_cm, pd, d, d_assets, inv_d_p, ia, ib,
                                    min1, max1, w_avg_v, w_disp_v, idx_min1, idx_min2, idx_min3, idx_max1, idx_max2, pa_r, pb_r,
                                    CALL_CODE,
                                )
                            else:
                                bp_cm = _accum_idx_grad_f64(
                                    zero_p, w_c_max_m * dg_c_dw, pd, d, d_assets, inv_d_p, ia, ib,
                                    min1, max1, w_avg_v, w_disp_v, idx_min1, idx_min2, idx_min3, idx_max1, idx_max2, pa_r, pb_r,
                                    CPN_CODE,
                                )
                                if FORCE_CPN_AC:
                                    bp_cm = _accum_idx_grad_f64(
                                        bp_cm, w_k_max_m * dg_k_dw, pd, d, d_assets, inv_d_p, ia, ib,
                                        min1, max1, w_avg_v, w_disp_v, idx_min1, idx_min2, idx_min3, idx_max1, idx_max2, pa_r, pb_r,
                                        CALL_CODE,
                                    )
                            d_eff_gc_d = bp_cm * pd * cdl_md
                            prev_dcm = tl.load(d_ln_surv_tmp_ptr + (d_assets + d) * n_total + offs_n, mask=mask_n, other=0.0).to(PRE_DTYPE)
                            tl.store(
                                d_ln_surv_tmp_ptr + (d_assets + d) * n_total + offs_n,
                                prev_dcm * (1.0 - eff_gc) - eff_c_m * d_eff_gc_d,
                                mask=mask_n,
                            )
        surv_pre = surv_pre_p.to(tl.float64)
        c_mem = c_mem_p.to(tl.float64)

        if frozen_term_out_ptr is not None:
            if m_act > 1:
                m_off_last = m_last * d_assets
                if USE_RT_FLAGS:
                    min1_b, _, max1_b, _, w_avg_b, _ = _eval_idx_base_stats_012_f64(
                        perf_ptr, m_off_last, offs_n, mask_n, d_act, n_total, D_PAD, tl.float64
                    )
                    wc_b = tl.where(rt_call == 0, min1_b, tl.where(rt_call == 1, max1_b, w_avg_b))
                    wcp_b = tl.where(rt_cpn == 0, min1_b, tl.where(rt_cpn == 1, max1_b, w_avg_b))
                    wo_b = tl.where(f_is_long, min1_b, tl.where(rt_opt == 0, min1_b, tl.where(rt_opt == 1, max1_b, w_avg_b)))
                    avg_b = tl.where(f_is_long, w_avg_b, zero_vec)
                    wb_b = tl.where(f_has_os, max1_b, zero_vec)
                else:
                    min1_b, min2_b, min3_b, _, _, _, max1_b, max2_b, _, _, s1_b, s2_b, pa_b, pb_b = _eval_idx_base_stats_f64(
                        perf_ptr, m_off_last, ia, ib, offs_n, mask_n, d_assets, n_total,
                        need_min1_k or need_min2_k or need_min3_k,
                        need_min2_k, need_min3_k,
                        need_max1_k or need_max2_k, need_max2_k,
                        need_sum1_k or need_sum2_k, need_sum2_k, need_ratio_k,
                    )
                    wc_b = _select_idx_from_stats_f64(min1_b, min2_b, min3_b, max1_b, max2_b, s1_b, s2_b, pa_b, pb_b, inv_d_f, d_assets, CALL_CODE)
                    wcp_b = wc_b if CPN_CODE == CALL_CODE else _select_idx_from_stats_f64(min1_b, min2_b, min3_b, max1_b, max2_b, s1_b, s2_b, pa_b, pb_b, inv_d_f, d_assets, CPN_CODE)
                    wo_b = min1_b if IS_LONG else (wc_b if EFF_OPT_CODE == CALL_CODE else (wcp_b if EFF_OPT_CODE == CPN_CODE else _select_idx_from_stats_f64(min1_b, min2_b, min3_b, max1_b, max2_b, s1_b, s2_b, pa_b, pb_b, inv_d_f, d_assets, EFF_OPT_CODE)))
                    avg_b = (s1_b * inv_d_f) if IS_LONG else zero_vec
                    wb_b = max1_b if HAS_ONE_STAR else zero_vec

                b_ko_l = tl.load(ko_bar_ptr + m_last)
                is_c_l = tl.load(is_call_ptr + m_last) > 0.5
                gk_l = tl.where(
                    is_c_l,
                    tl.where(hard_ko_sp, tl.minimum(tl.maximum((wc_b - tl.minimum(b_ko_l, b_ko_l + ko_spread)) * inv_sp_ko, zero_vec), one_vec), tl.where(wc_b >= b_ko_l, one_vec, zero_vec)),
                    zero_vec,
                )
                b_cp_l = tl.load(cpn_bar_ptr + m_last)
                is_g_l = tl.load(is_guar_ptr + m_last) > 0.5
                if USE_RT_FLAGS:
                    hit_hl = tl.where(f_cpn_dn, wcp_b <= b_cp_l, wcp_b >= b_cp_l)
                    gc_hl = tl.where(f_cpn_out, tl.where(hit_hl, zero_vec, one_vec), tl.where(hit_hl, one_vec, zero_vec))
                    blo_cpl = tl.minimum(b_cp_l, b_cp_l + eff_csp)
                    r_upl = tl.minimum(tl.maximum((wcp_b - blo_cpl) * inv_sp_cp, zero_vec), one_vec)
                    hit_sl = tl.where(f_cpn_dn, 1.0 - r_upl, r_upl)
                    gc_sl = tl.where(f_cpn_out, 1.0 - hit_sl, hit_sl)
                    gc_l = tl.where(is_g_l, one_vec, tl.where(hard_cpn_sp, gc_sl, gc_hl))
                    egc_l = tl.where(f_frc_cpn, tl.maximum(gc_l, gk_l), gc_l)

                    b_kil = tl.where(f_is_long, strike, ki_barrier)
                    blo_kl = tl.minimum(b_kil, b_kil + ki_spread)
                    bhi_kl = tl.maximum(b_kil, b_kil + ki_spread)
                    inv_wkl = 1.0 / tl.maximum(bhi_kl - blo_kl, 1e-12)
                    ki_act_l = tl.where(hard_ki_sp, tl.minimum(tl.maximum((bhi_kl - wo_b) * inv_wkl, zero_vec), one_vec), tl.where(wo_b < b_kil, one_vec, zero_vec))
                    ki_g_l = tl.where(f_ki_act, ki_act_l, tl.where(f_has_ki, zero_vec + i_ki_past, one_vec))
                else:
                    hit_hl = (wcp_b <= b_cp_l) if CPN_DIR_DOWN else (wcp_b >= b_cp_l)
                    gc_hl = tl.where(hit_hl, zero_vec if CPN_TYPE_OUT else one_vec, one_vec if CPN_TYPE_OUT else zero_vec)
                    blo_cpl = tl.minimum(b_cp_l, b_cp_l + eff_csp)
                    r_upl = tl.minimum(tl.maximum((wcp_b - blo_cpl) * inv_sp_cp, zero_vec), one_vec)
                    hit_sl = (1.0 - r_upl) if CPN_DIR_DOWN else r_upl
                    gc_sl = (1.0 - hit_sl) if CPN_TYPE_OUT else hit_sl
                    gc_l = tl.where(is_g_l, one_vec, tl.where(hard_cpn_sp, gc_sl, gc_hl))
                    egc_l = tl.maximum(gc_l, gk_l) if FORCE_CPN_AC else gc_l

                    if IS_LONG and KI_ACTIVE:
                        blo_kl = tl.minimum(strike, strike + ki_spread)
                        bhi_kl = tl.maximum(strike, strike + ki_spread)
                        inv_wkl = 1.0 / tl.maximum(bhi_kl - blo_kl, 1e-12)
                        ki_g_l = tl.where(hard_ki_sp, tl.minimum(tl.maximum((bhi_kl - wo_b) * inv_wkl, zero_vec), one_vec), tl.where(wo_b < strike, one_vec, zero_vec))
                    else:
                        ki_g_l = (
                            (tl.where(hard_ki_sp, tl.minimum(tl.maximum((bhi_ki_k - wo_b) * inv_sp_ki, zero_vec), one_vec), tl.where(wo_b < ki_barrier, one_vec, zero_vec)))
                            if KI_ACTIVE else ((zero_vec + i_ki_past) if HAS_KI else one_vec)
                        )
                half_os_l = 0.5 * one_star_eps
                os_l = tl.where(
                    f_has_os,
                    tl.where(one_star_eps <= 1e-6, tl.where(wb_b >= one_star_strike, one_vec, zero_vec), tl.minimum(tl.maximum((wb_b - (one_star_strike - half_os_l)) * inv_sp_os, zero_vec), one_vec)),
                    zero_vec,
                )
                opt_l = zero_vec
                if USE_RT_FLAGS:
                    opt_l_long = -(1.0 - avg_b) * ki_g_l * (1.0 - os_l)
                    put_intr_l = tl.where(f_has_low, tl.maximum(strike - wo_b, zero_vec) - tl.maximum(lower_strike - wo_b, zero_vec), tl.maximum(strike - wo_b, zero_vec))
                    put_m_l = tl.where(f_has_low, (strike > lower_strike) & (option_gearing > 0.0), option_gearing > 0.0)
                    opt_l_short = tl.where(put_m_l, -option_gearing * ki_g_l * (1.0 - os_l) * put_intr_l, zero_vec)
                    opt_l = tl.where(f_is_long, opt_l_long, opt_l_short)
                    opt_l = tl.where(f_has_pc & (part_call_rate != 0.0), opt_l + part_call_rate * tl.maximum(wo_b - part_call_strike, zero_vec), opt_l)
                    opt_l = tl.where(f_has_os & (one_star_coupon != 0.0), opt_l + os_l * one_star_coupon, opt_l)
                else:
                    if IS_LONG:
                        opt_l = opt_l - (1.0 - avg_b) * ki_g_l * (1.0 - os_l)
                    else:
                        put_intr_l = (tl.maximum(strike - wo_b, zero_vec) - tl.maximum(lower_strike - wo_b, zero_vec)) if HAS_LOWER_STRIKE else tl.maximum(strike - wo_b, zero_vec)
                        put_m_l = ((strike > lower_strike) & (option_gearing > 0.0)) if HAS_LOWER_STRIKE else (option_gearing > 0.0)
                        opt_l = tl.where(put_m_l, opt_l - option_gearing * ki_g_l * (1.0 - os_l) * put_intr_l, opt_l)
                    if HAS_PART_CALL:
                        if part_call_rate != 0.0:
                            opt_l = opt_l + part_call_rate * tl.maximum(wo_b - part_call_strike, zero_vec)
                    if HAS_ONE_STAR:
                        if one_star_coupon != 0.0:
                            opt_l = opt_l + os_l * one_star_coupon
                tl.store(frozen_term_out_ptr + 0 * n_total + offs_n, gk_l, mask=mask_n)
                tl.store(frozen_term_out_ptr + 1 * n_total + offs_n, egc_l, mask=mask_n)
                tl.store(frozen_term_out_ptr + 2 * n_total + offs_n, opt_l, mask=mask_n)
                tl.store(frozen_term_out_ptr + 3 * n_total + offs_n, os_l, mask=mask_n)

        n_d_vib: tl.constexpr = D_PAD if USE_RT_FLAGS else d_assets
        if tl.max(surv_pre, axis=0) <= 1e-24:
            for i in range(n_d_vib):
                if (i < d_act) if USE_RT_FLAGS else True:
                    for j in range(n_d_vib):
                        if ((j >= i) & (j < d_act)) if USE_RT_FLAGS else (j >= i):
                            tl.store(h_vib_out_ptr + (pid_n * d_assets + i) * d_assets + j, 0.0)
                            if i != j:
                                tl.store(h_vib_out_ptr + (pid_n * d_assets + j) * d_assets + i, 0.0)
        else:
            sqrt_h_vib = tl.sqrt(eff_h_vib)
            inv_sqrt_n = 1.0 / tl.sqrt(tl.zeros((), dtype=tl.float64) + eff_vib_n)
            f0 = zero_vec
            fp = zero_vec
            fm = zero_vec
            egc_p = zero_vec
            egc_m = zero_vec
            b_ko_l = tl.load(ko_bar_ptr + m_last)
            is_c_l = tl.load(is_call_ptr + m_last) > 0.5
            b_cp_l = tl.load(cpn_bar_ptr + m_last)
            is_g_l = tl.load(is_guar_ptr + m_last) > 0.5
            c_rate_l = tl.load(cpn_rates_ptr + m_last)
            eff_c_l = (tl.where(f_is_mem, c_rate_l + c_mem, zero_vec + c_rate_l) if USE_RT_FLAGS else ((c_rate_l + c_mem) if IS_MEM else (zero_vec + c_rate_l)))
            term_not = (tl.where(f_no_not, 0.0, 1.0) if USE_RT_FLAGS else (0.0 if NO_NOTIONAL else 1.0))

            for sgn in tl.static_range(3):
                mult = 0.0 if sgn == 0 else (1.0 if sgn == 1 else -1.0)
                min1_s = tl.full((BLOCK_N,), 1e300, dtype=tl.float64)
                min2_s = tl.full((BLOCK_N,), 1e300, dtype=tl.float64)
                min3_s = tl.full((BLOCK_N,), 1e300, dtype=tl.float64)
                max1_s = tl.full((BLOCK_N,), -1e300, dtype=tl.float64)
                max2_s = tl.full((BLOCK_N,), -1e300, dtype=tl.float64)
                s1_s = zero_vec
                s2_s = zero_vec
                pa_s = zero_vec
                pb_s = one_vec
                for d in range(n_d_vib):
                    if (d < d_act) if USE_RT_FLAGS else True:
                        lxp = tl.load(lx_pre_ptr + d * n_total + offs_n, mask=mask_n, other=0.0).to(tl.float64)
                        sigp = tl.maximum(tl.load(sig_pre_ptr + d * n_total + offs_n, mask=mask_n, other=0.2).to(tl.float64), 1e-4)
                        if sgn == 0:
                            xd = lxp - 0.5 * sigp * sigp * eff_h_vib
                        else:
                            zc_d = zero_vec
                            for k in range(n_d_vib):
                                if (k < d_act) if USE_RT_FLAGS else True:
                                    l_dk = tl.load(chol_ptr + d * d_assets + k)
                                    zk_sum = zero_vec
                                    n_sv_loop: tl.constexpr = N_SUB_PAD if USE_RT_FLAGS else vib_n_steps
                                    for s_v in range(n_sv_loop):
                                        if (s_v < eff_vib_n) if USE_RT_FLAGS else True:
                                            st_i = eff_vib_start + s_v
                                            zk_sum = zk_sum + tl.load(sv_ptr + (st_i * d_assets + k) * n_total + offs_n, mask=mask_n, other=0.0).to(tl.float64)
                                    zc_d = zc_d + l_dk * (zk_sum * inv_sqrt_n)
                            xd = lxp - 0.5 * sigp * sigp * eff_h_vib + mult * sigp * sqrt_h_vib * zc_d
                        fwd_d = tl.load(fwd_term_ptr + (d * m_rem + m_last if USE_RT_FLAGS else d))
                        pd = fwd_d * tl.exp(tl.minimum(tl.maximum(xd, -30.0), 30.0))
                        if USE_RT_FLAGS:
                            min1_s = tl.minimum(min1_s, pd)
                            max1_s = tl.maximum(max1_s, pd)
                            s1_s = s1_s + pd
                        else:
                            if need_min3_k:
                                lt1 = pd < min1_s
                                lt2 = (~lt1) & (pd < min2_s)
                                lt3 = (~lt1) & (~lt2) & (pd < min3_s)
                                min3_s = tl.where(lt1, min2_s, tl.where(lt2, min2_s, tl.where(lt3, pd, min3_s)))
                                min2_s = tl.where(lt1, min1_s, tl.where(lt2, pd, min2_s))
                                min1_s = tl.where(lt1, pd, min1_s)
                            elif need_min2_k:
                                lt1 = pd < min1_s
                                lt2 = (~lt1) & (pd < min2_s)
                                min2_s = tl.where(lt1, min1_s, tl.where(lt2, pd, min2_s))
                                min1_s = tl.where(lt1, pd, min1_s)
                            elif need_min1_k:
                                min1_s = tl.minimum(min1_s, pd)
                            if need_max2_k:
                                gt1 = pd > max1_s
                                gt2 = (~gt1) & (pd > max2_s)
                                max2_s = tl.where(gt1, max1_s, tl.where(gt2, pd, max2_s))
                                max1_s = tl.where(gt1, pd, max1_s)
                            elif need_max1_k:
                                max1_s = tl.maximum(max1_s, pd)
                            if need_sum1_k:
                                s1_s = s1_s + pd
                            if need_sum2_k:
                                s2_s = s2_s + pd * pd
                            if d == ia:
                                pa_s = pd
                            if d == ib:
                                pb_s = pd

                if USE_RT_FLAGS:
                    w_avg_s = s1_s * inv_d_f
                    wc_s = tl.where(rt_call == 0, min1_s, tl.where(rt_call == 1, max1_s, w_avg_s))
                    wcp_s = tl.where(rt_cpn == 0, min1_s, tl.where(rt_cpn == 1, max1_s, w_avg_s))
                    wo_s = tl.where(f_is_long, min1_s, tl.where(rt_opt == 0, min1_s, tl.where(rt_opt == 1, max1_s, w_avg_s)))
                    avg_s = tl.where(f_is_long, w_avg_s, zero_vec)
                    wb_s = tl.where(f_has_os, max1_s, zero_vec)
                else:
                    wc_s = _select_idx_from_stats_f64(min1_s, min2_s, min3_s, max1_s, max2_s, s1_s, s2_s, pa_s, pb_s, inv_d_f, d_assets, CALL_CODE)
                    wcp_s = wc_s if CPN_CODE == CALL_CODE else _select_idx_from_stats_f64(min1_s, min2_s, min3_s, max1_s, max2_s, s1_s, s2_s, pa_s, pb_s, inv_d_f, d_assets, CPN_CODE)
                    wo_s = min1_s if IS_LONG else (wc_s if EFF_OPT_CODE == CALL_CODE else (wcp_s if EFF_OPT_CODE == CPN_CODE else _select_idx_from_stats_f64(min1_s, min2_s, min3_s, max1_s, max2_s, s1_s, s2_s, pa_s, pb_s, inv_d_f, d_assets, EFF_OPT_CODE)))
                    avg_s = (s1_s * inv_d_f) if IS_LONG else zero_vec
                    wb_s = max1_s if HAS_ONE_STAR else zero_vec

                gk_s = tl.where(
                    is_c_l,
                    tl.where(hard_ko_sp, tl.minimum(tl.maximum((wc_s - tl.minimum(b_ko_l, b_ko_l + ko_spread)) * inv_sp_ko, zero_vec), one_vec), tl.where(wc_s >= b_ko_l, one_vec, zero_vec)),
                    zero_vec,
                )
                if USE_RT_FLAGS:
                    hit_hs = tl.where(f_cpn_dn, wcp_s <= b_cp_l, wcp_s >= b_cp_l)
                    gc_hs = tl.where(f_cpn_out, tl.where(hit_hs, zero_vec, one_vec), tl.where(hit_hs, one_vec, zero_vec))
                    blo_cps = tl.minimum(b_cp_l, b_cp_l + eff_csp)
                    r_ups = tl.minimum(tl.maximum((wcp_s - blo_cps) * inv_sp_cp, zero_vec), one_vec)
                    hit_ss = tl.where(f_cpn_dn, 1.0 - r_ups, r_ups)
                    gc_ss = tl.where(f_cpn_out, 1.0 - hit_ss, hit_ss)
                    gc_s = tl.where(is_g_l, one_vec, tl.where(hard_cpn_sp, gc_ss, gc_hs))
                    egc_s = tl.where(f_frc_cpn, tl.maximum(gc_s, gk_s), gc_s)

                    b_kis = tl.where(f_is_long, strike, ki_barrier)
                    blo_ks = tl.minimum(b_kis, b_kis + ki_spread)
                    bhi_ks = tl.maximum(b_kis, b_kis + ki_spread)
                    inv_wks = 1.0 / tl.maximum(bhi_ks - blo_ks, 1e-12)
                    ki_act_s = tl.where(hard_ki_sp, tl.minimum(tl.maximum((bhi_ks - wo_s) * inv_wks, zero_vec), one_vec), tl.where(wo_s < b_kis, one_vec, zero_vec))
                    ki_g_s = tl.where(f_ki_act, ki_act_s, tl.where(f_has_ki, zero_vec + i_ki_past, one_vec))
                else:
                    hit_hs = (wcp_s <= b_cp_l) if CPN_DIR_DOWN else (wcp_s >= b_cp_l)
                    gc_hs = tl.where(hit_hs, zero_vec if CPN_TYPE_OUT else one_vec, one_vec if CPN_TYPE_OUT else zero_vec)
                    blo_cps = tl.minimum(b_cp_l, b_cp_l + eff_csp)
                    r_ups = tl.minimum(tl.maximum((wcp_s - blo_cps) * inv_sp_cp, zero_vec), one_vec)
                    hit_ss = (1.0 - r_ups) if CPN_DIR_DOWN else r_ups
                    gc_ss = (1.0 - hit_ss) if CPN_TYPE_OUT else hit_ss
                    gc_s = tl.where(is_g_l, one_vec, tl.where(hard_cpn_sp, gc_ss, gc_hs))
                    egc_s = tl.maximum(gc_s, gk_s) if FORCE_CPN_AC else gc_s

                    if IS_LONG and KI_ACTIVE:
                        blo_ks = tl.minimum(strike, strike + ki_spread)
                        bhi_ks = tl.maximum(strike, strike + ki_spread)
                        inv_wks = 1.0 / tl.maximum(bhi_ks - blo_ks, 1e-12)
                        ki_g_s = tl.where(hard_ki_sp, tl.minimum(tl.maximum((bhi_ks - wo_s) * inv_wks, zero_vec), one_vec), tl.where(wo_s < strike, one_vec, zero_vec))
                    else:
                        ki_g_s = (
                            (tl.where(hard_ki_sp, tl.minimum(tl.maximum((bhi_ki_k - wo_s) * inv_sp_ki, zero_vec), one_vec), tl.where(wo_s < ki_barrier, one_vec, zero_vec)))
                            if KI_ACTIVE else ((zero_vec + i_ki_past) if HAS_KI else one_vec)
                        )
                half_oss = 0.5 * one_star_eps
                os_s = tl.where(
                    f_has_os,
                    tl.where(one_star_eps <= 1e-6, tl.where(wb_s >= one_star_strike, one_vec, zero_vec), tl.minimum(tl.maximum((wb_s - (one_star_strike - half_oss)) * inv_sp_os, zero_vec), one_vec)),
                    zero_vec,
                )
                opt_s = zero_vec
                if USE_RT_FLAGS:
                    opt_s_long = -(1.0 - avg_s) * ki_g_s * (1.0 - os_s)
                    put_intr_s = tl.where(f_has_low, tl.maximum(strike - wo_s, zero_vec) - tl.maximum(lower_strike - wo_s, zero_vec), tl.maximum(strike - wo_s, zero_vec))
                    put_m_s = tl.where(f_has_low, (strike > lower_strike) & (option_gearing > 0.0), option_gearing > 0.0)
                    opt_s_short = tl.where(put_m_s, -option_gearing * ki_g_s * (1.0 - os_s) * put_intr_s, zero_vec)
                    opt_s = tl.where(f_is_long, opt_s_long, opt_s_short)
                    opt_s = tl.where(f_has_pc & (part_call_rate != 0.0), opt_s + part_call_rate * tl.maximum(wo_s - part_call_strike, zero_vec), opt_s)
                    opt_s = tl.where(f_has_os & (one_star_coupon != 0.0), opt_s + os_s * one_star_coupon, opt_s)
                else:
                    if IS_LONG:
                        opt_s = opt_s - (1.0 - avg_s) * ki_g_s * (1.0 - os_s)
                    else:
                        put_intr_s = (tl.maximum(strike - wo_s, zero_vec) - tl.maximum(lower_strike - wo_s, zero_vec)) if HAS_LOWER_STRIKE else tl.maximum(strike - wo_s, zero_vec)
                        put_m_s = ((strike > lower_strike) & (option_gearing > 0.0)) if HAS_LOWER_STRIKE else (option_gearing > 0.0)
                        opt_s = tl.where(put_m_s, opt_s - option_gearing * ki_g_s * (1.0 - os_s) * put_intr_s, opt_s)
                    if HAS_PART_CALL:
                        if part_call_rate != 0.0:
                            opt_s = opt_s + part_call_rate * tl.maximum(wo_s - part_call_strike, zero_vec)
                    if HAS_ONE_STAR:
                        if one_star_coupon != 0.0:
                            opt_s = opt_s + os_s * one_star_coupon

                f_val = gk_s * (term_not + ac_cpn) + egc_s * eff_c_l + (1.0 - gk_s) * (term_not + opt_s)
                if sgn == 0:
                    f0 = f_val
                elif sgn == 1:
                    fp = f_val
                    egc_p = egc_s
                else:
                    fm = f_val
                    egc_m = egc_s

            df_diff = 0.5 * (fp - fm)
            df_hess = 0.5 * (fp + fm) - f0
            degc_diff = (tl.where(f_is_mem, 0.5 * (egc_p - egc_m), zero_vec) if USE_RT_FLAGS else (0.5 * (egc_p - egc_m) if IS_MEM else zero_vec))
            for i in range(n_d_vib):
                if (i < d_act) if USE_RT_FLAGS else True:
                    sig_i = tl.maximum(tl.load(sig_pre_ptr + i * n_total + offs_n, mask=mask_n, other=0.2).to(tl.float64), 1e-4)
                    inv_si = 1.0 / tl.maximum(sig_i * sqrt_h_vib, 1e-6)
                    j_pre_i = tl.minimum(
                        tl.maximum(tl.load(j_pre_ptr + i * n_total + offs_n, mask=mask_n, other=1.0).to(tl.float64), -8.0),
                        8.0,
                    )
                    j_i = j_pre_i * tl.load(cdl_m_ptr + m_last * d_assets + i)
                    cdl2_i = tl.load(cdl2_last_ptr + i)
                    dlns_i = tl.load(d_ln_surv_tmp_ptr + i * n_total + offs_n, mask=mask_n, other=0.0).to(tl.float64)
                    dcm_i = (tl.where(f_is_mem, tl.load(d_ln_surv_tmp_ptr + (d_assets + i) * n_total + offs_n, mask=mask_n, other=0.0).to(tl.float64), zero_vec) if USE_RT_FLAGS else (tl.load(d_ln_surv_tmp_ptr + (d_assets + i) * n_total + offs_n, mask=mask_n, other=0.0).to(tl.float64) if IS_MEM else zero_vec))
                    xi_i = zero_vec
                    for k in range(n_d_vib):
                        if (k < d_act) if USE_RT_FLAGS else True:
                            lit_ik = tl.load(l_inv_t_ptr + i * d_assets + k)
                            zk_s = zero_vec
                            n_sv_i: tl.constexpr = N_SUB_PAD if USE_RT_FLAGS else vib_n_steps
                            for s_v in range(n_sv_i):
                                if (s_v < eff_vib_n) if USE_RT_FLAGS else True:
                                    st_i = eff_vib_start + s_v
                                    zk_s = zk_s + tl.load(sv_ptr + (st_i * d_assets + k) * n_total + offs_n, mask=mask_n, other=0.0).to(tl.float64)
                            xi_i = xi_i + lit_ik * (zk_s * inv_sqrt_n)
                    w1_raw_i = df_diff * xi_i * inv_si * j_pre_i
                    w1_i = df_diff * xi_i * inv_si * j_i
                    w1_egc_i = degc_diff * xi_i * inv_si * j_i

                    for j in range(n_d_vib):
                        if ((j >= i) & (j < d_act)) if USE_RT_FLAGS else (j >= i):
                            sig_j = tl.maximum(tl.load(sig_pre_ptr + j * n_total + offs_n, mask=mask_n, other=0.2).to(tl.float64), 1e-4)
                            inv_sj = 1.0 / tl.maximum(sig_j * sqrt_h_vib, 1e-6)
                            j_pre_j = tl.minimum(
                                tl.maximum(tl.load(j_pre_ptr + j * n_total + offs_n, mask=mask_n, other=1.0).to(tl.float64), -8.0),
                                8.0,
                            )
                            j_j = j_pre_j * tl.load(cdl_m_ptr + m_last * d_assets + j)
                            dlns_j = tl.load(d_ln_surv_tmp_ptr + j * n_total + offs_n, mask=mask_n, other=0.0).to(tl.float64)
                            dcm_j = (tl.where(f_is_mem, tl.load(d_ln_surv_tmp_ptr + (d_assets + j) * n_total + offs_n, mask=mask_n, other=0.0).to(tl.float64), zero_vec) if USE_RT_FLAGS else (tl.load(d_ln_surv_tmp_ptr + (d_assets + j) * n_total + offs_n, mask=mask_n, other=0.0).to(tl.float64) if IS_MEM else zero_vec))
                            xi_j = zero_vec
                            for k in range(n_d_vib):
                                if (k < d_act) if USE_RT_FLAGS else True:
                                    lit_jk = tl.load(l_inv_t_ptr + j * d_assets + k)
                                    zk_s = zero_vec
                                    n_sv_j: tl.constexpr = N_SUB_PAD if USE_RT_FLAGS else vib_n_steps
                                    for s_v in range(n_sv_j):
                                        if (s_v < eff_vib_n) if USE_RT_FLAGS else True:
                                            st_i = eff_vib_start + s_v
                                            zk_s = zk_s + tl.load(sv_ptr + (st_i * d_assets + k) * n_total + offs_n, mask=mask_n, other=0.0).to(tl.float64)
                                    xi_j = xi_j + lit_jk * (zk_s * inv_sqrt_n)
                            w1_j = df_diff * xi_j * inv_sj * j_j
                            w1_egc_j = degc_diff * xi_j * inv_sj * j_j
                            cinv_ij = tl.load(c_inv_ptr + i * d_assets + j)
                            w2_ij = df_hess * (xi_i * xi_j - cinv_ij) * inv_si * inv_sj * j_i * j_j
                            cmem_cross = dcm_i * w1_egc_j + dcm_j * w1_egc_i
                            diag_corr = tl.where(i == j, w1_raw_i * cdl2_i, zero_vec)
                            h_ij = eff_df_last * surv_pre * (w2_ij + dlns_i * w1_j + dlns_j * w1_i + cmem_cross + diag_corr)
                            h_sum = tl.sum(tl.where(mask_n, h_ij, zero_vec), axis=0) * scale
                            tl.store(h_vib_out_ptr + (pid_n * d_assets + i) * d_assets + j, h_sum)
                            if i != j:
                                tl.store(h_vib_out_ptr + (pid_n * d_assets + j) * d_assets + i, h_sum)


def _eval_index_batch_torch(
    perf_m_d_n: torch.Tensor,
    index_code: int,
    inv_perm: Sequence[int] | npt.NDArray[np.int64],
) -> torch.Tensor:
    """Vectorized monitoring index evaluation on GPU/CPU (M, D, N) tensors."""
    d = perf_m_d_n.shape[1]
    if index_code == 0:
        return perf_m_d_n[:, 0, :] if d == 1 else torch.min(perf_m_d_n, dim=1).values
    if index_code == 1:
        return perf_m_d_n[:, 0, :] if d == 1 else torch.max(perf_m_d_n, dim=1).values
    if index_code == 2:
        return perf_m_d_n[:, 0, :] if d == 1 else torch.mean(perf_m_d_n, dim=1)
    if index_code == 14:
        w_min = perf_m_d_n[:, 0, :] if d == 1 else torch.min(perf_m_d_n, dim=1).values
        return torch.where(w_min < 1.0, 2.0 - w_min, w_min)
    if index_code == 3:
        if d <= 1:
            return torch.zeros((perf_m_d_n.shape[0], perf_m_d_n.shape[2]), dtype=perf_m_d_n.dtype, device=perf_m_d_n.device)
        return torch.std(perf_m_d_n, dim=1, unbiased=False)
    if index_code == 6:
        return torch.max(perf_m_d_n, dim=1).values / torch.min(perf_m_d_n, dim=1).values.clamp(min=1e-12)
    if index_code == 7:
        return torch.max(perf_m_d_n, dim=1).values / torch.mean(perf_m_d_n, dim=1).clamp(min=1e-12)
    if index_code == 8:
        return torch.mean(perf_m_d_n, dim=1) / torch.min(perf_m_d_n, dim=1).values.clamp(min=1e-12)
    if index_code == 11:
        if d == 1:
            return torch.zeros((perf_m_d_n.shape[0], perf_m_d_n.shape[2]), dtype=perf_m_d_n.dtype, device=perf_m_d_n.device)
        return torch.max(perf_m_d_n, dim=1).values - torch.min(perf_m_d_n, dim=1).values
    if index_code == 12:
        if d == 1:
            return torch.zeros((perf_m_d_n.shape[0], perf_m_d_n.shape[2]), dtype=perf_m_d_n.dtype, device=perf_m_d_n.device)
        return torch.max(perf_m_d_n, dim=1).values - torch.mean(perf_m_d_n, dim=1)
    if index_code == 13:
        if d == 1:
            return torch.zeros((perf_m_d_n.shape[0], perf_m_d_n.shape[2]), dtype=perf_m_d_n.dtype, device=perf_m_d_n.device)
        return torch.mean(perf_m_d_n, dim=1) - torch.min(perf_m_d_n, dim=1).values
    if index_code in (15, 16, 17):
        k_pos = min(1, d - 1) if index_code == 15 else (min(2, d - 1) if index_code == 16 else max(0, d - 2))
        vals, _ = torch.sort(perf_m_d_n, dim=1, stable=True)
        return vals[:, k_pos, :]
    ia = int(inv_perm[0])
    if d < 2:
        if index_code in (9, 10):
            return torch.zeros_like(perf_m_d_n[:, ia, :])
        return perf_m_d_n[:, ia, :]
    ib = int(inv_perm[1])
    if index_code == 4:
        return perf_m_d_n[:, ia, :] / perf_m_d_n[:, ib, :].clamp(min=1e-12)
    if index_code == 5:
        return perf_m_d_n[:, ib, :] / perf_m_d_n[:, ia, :].clamp(min=1e-12)
    if index_code == 9:
        return perf_m_d_n[:, ia, :] - perf_m_d_n[:, ib, :]
    if index_code == 10:
        return perf_m_d_n[:, ib, :] - perf_m_d_n[:, ia, :]
    raise ValueError(f"Unsupported monitoring index code {index_code}.")


def _eval_index_and_weights_torch(
    perf_m_d_n: torch.Tensor,
    index_code: int,
    inv_perm: Sequence[int] | npt.NDArray[np.int64],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (W (M, N), dW/dP (M, D, N)) for pathwise payoff AAD."""
    m_rem, d, n_tot = perf_m_d_n.shape
    dtype = perf_m_d_n.dtype
    dev = perf_m_d_n.device
    grad_p = torch.zeros_like(perf_m_d_n)
    ia = int(inv_perm[0])
    ib = int(inv_perm[1]) if d >= 2 else ia

    if index_code == 0:
        w, idx = torch.min(perf_m_d_n, dim=1)
        grad_p.scatter_(1, idx.unsqueeze(1), 1.0)
        return w, grad_p
    if index_code == 1:
        w, idx = torch.max(perf_m_d_n, dim=1)
        grad_p.scatter_(1, idx.unsqueeze(1), 1.0)
        return w, grad_p
    if index_code == 2:
        w = torch.mean(perf_m_d_n, dim=1)
        grad_p.fill_(1.0 / float(d))
        return w, grad_p
    if index_code == 14:
        w_min, idx = torch.min(perf_m_d_n, dim=1)
        sgn = torch.where(w_min < 1.0, -1.0, 1.0).to(dtype=dtype)
        grad_p.scatter_(1, idx.unsqueeze(1), sgn.unsqueeze(1))
        return torch.where(w_min < 1.0, 2.0 - w_min, w_min), grad_p
    if index_code == 3:
        if d <= 1:
            return torch.zeros((m_rem, n_tot), dtype=dtype, device=dev), grad_p
        w_avg = torch.mean(perf_m_d_n, dim=1, keepdim=True)
        diff = perf_m_d_n - w_avg
        var = torch.mean(diff * diff, dim=1)
        std = torch.sqrt(var.clamp(min=0.0))
        inv_d_std = torch.where(std > 1e-12, 1.0 / (float(d) * std.clamp(min=1e-12)), torch.zeros_like(std))
        grad_p = diff * inv_d_std.unsqueeze(1)
        return std, grad_p
    if index_code in (4, 5):
        if d < 2:
            grad_p[:, ia, :] = 1.0
            return perf_m_d_n[:, ia, :], grad_p
        pa = perf_m_d_n[:, ia, :]
        pb = perf_m_d_n[:, ib, :]
        if index_code == 4:
            pb_s = pb.clamp(min=1e-12)
            w = pa / pb_s
            grad_p[:, ia, :] = 1.0 / pb_s
            grad_p[:, ib, :] = torch.where(pb > 1e-12, -pa / (pb_s * pb_s), torch.zeros_like(pa))
        else:
            pa_s = pa.clamp(min=1e-12)
            w = pb / pa_s
            grad_p[:, ib, :] = 1.0 / pa_s
            grad_p[:, ia, :] = torch.where(pa > 1e-12, -pb / (pa_s * pa_s), torch.zeros_like(pb))
        return w, grad_p
    if index_code in (9, 10):
        if d < 2:
            return torch.zeros((m_rem, n_tot), dtype=dtype, device=dev), grad_p
        if index_code == 9:
            grad_p[:, ia, :] = 1.0
            grad_p[:, ib, :] = -1.0
            return perf_m_d_n[:, ia, :] - perf_m_d_n[:, ib, :], grad_p
        grad_p[:, ib, :] = 1.0
        grad_p[:, ia, :] = -1.0
        return perf_m_d_n[:, ib, :] - perf_m_d_n[:, ia, :], grad_p
    if index_code in (6, 7, 8, 11, 12, 13):
        w_max, i_max = torch.max(perf_m_d_n, dim=1)
        w_min, i_min = torch.min(perf_m_d_n, dim=1)
        w_avg = torch.mean(perf_m_d_n, dim=1)
        if index_code == 6:
            den = w_min.clamp(min=1e-12)
            w = w_max / den
            grad_p.scatter_add_(1, i_max.unsqueeze(1), (1.0 / den).unsqueeze(1))
            g_den = torch.where(w_min > 1e-12, -w_max / (den * den), torch.zeros_like(den))
            grad_p.scatter_add_(1, i_min.unsqueeze(1), g_den.unsqueeze(1))
            return w, grad_p
        if index_code == 7:
            den = w_avg.clamp(min=1e-12)
            w = w_max / den
            g_avg = torch.where(w_avg > 1e-12, -w_max / (float(d) * den * den), torch.zeros_like(den))
            grad_p = g_avg.unsqueeze(1).expand_as(perf_m_d_n).clone()
            grad_p.scatter_add_(1, i_max.unsqueeze(1), (1.0 / den).unsqueeze(1))
            return w, grad_p
        if index_code == 8:
            den = w_min.clamp(min=1e-12)
            w = w_avg / den
            grad_p.fill_(0.0)
            grad_p = (1.0 / (float(d) * den)).unsqueeze(1).expand_as(perf_m_d_n).clone()
            g_den = torch.where(w_min > 1e-12, -w_avg / (den * den), torch.zeros_like(den))
            grad_p.scatter_add_(1, i_min.unsqueeze(1), g_den.unsqueeze(1))
            return w, grad_p
        if index_code == 11:
            grad_p.scatter_add_(1, i_max.unsqueeze(1), torch.ones((m_rem, 1, n_tot), dtype=dtype, device=dev))
            grad_p.scatter_add_(1, i_min.unsqueeze(1), -torch.ones((m_rem, 1, n_tot), dtype=dtype, device=dev))
            return w_max - w_min, grad_p
        if index_code == 12:
            grad_p = torch.full_like(perf_m_d_n, -1.0 / float(d))
            grad_p.scatter_add_(1, i_max.unsqueeze(1), torch.ones((m_rem, 1, n_tot), dtype=dtype, device=dev))
            return w_max - w_avg, grad_p
        grad_p = torch.full_like(perf_m_d_n, 1.0 / float(d))
        grad_p.scatter_add_(1, i_min.unsqueeze(1), -torch.ones((m_rem, 1, n_tot), dtype=dtype, device=dev))
        return w_avg - w_min, grad_p
    if index_code in (15, 16, 17):
        k_pos = min(1, d - 1) if index_code == 15 else (min(2, d - 1) if index_code == 16 else max(0, d - 2))
        vals, ord_idx = torch.sort(perf_m_d_n, dim=1, stable=True)
        grad_p.scatter_(1, ord_idx[:, k_pos : k_pos + 1, :], 1.0)
        return vals[:, k_pos, :], grad_p
    raise ValueError(f"Unsupported monitoring index code {index_code}.")


_TRITON_PAYOFF_AAD_CODES = set(range(18))


def _compute_payoff_aad_state_adjoints_torch(
    perf_m_d_n: torch.Tensor,
    *,
    ko_barriers: torch.Tensor,
    is_callable: torch.Tensor,
    cpn_barriers: torch.Tensor,
    is_guar: torch.Tensor,
    cpn_rates: torch.Tensor,
    df_pay: torch.Tensor,
    w_fund: torch.Tensor | None,
    inv_perm: Sequence[int] | npt.NDArray[np.int64],
    call_code: int,
    cpn_code: int,
    opt_code: int,
    ko_spread: float,
    cpn_spread: float,
    ki_spread: float,
    cpn_dir_down: bool,
    cpn_type_out: bool,
    is_memory: bool,
    force_cpn_ac: bool,
    no_notional: bool,
    is_long_basket: bool,
    has_ki: bool,
    ki_barrier: float,
    i_ki_past: float,
    ki_monitoring_type: str,
    tau_obs: npt.NDArray[np.float64],
    vol_atm_obs: npt.NDArray[np.float64] | None,
    strike: float,
    lower_strike: float | None,
    option_gearing: float,
    ac_cpn: float,
    c_mem_0: float,
    part_call_strike: float | None,
    part_call_rate: float,
    one_star_strike: float | None,
    one_star_coupon: float,
    one_star_eps: float,
    barrier_eps_min: float,
    barrier_eps_max: float,
    lx_ckpt: torch.Tensor | None = None,
    lv_tables_t: torch.Tensor | None = None,
    corr_t: torch.Tensor | None = None,
    fwd_perf_t: torch.Tensor | None = None,
    r_nat_mid_t: torch.Tensor | None = None,
    y_min: float | Sequence[float] | npt.NDArray[np.float64] = -1.25,
    y_max: float | Sequence[float] | npt.NDArray[np.float64] = 1.25,
    n_sub: int = 4,
    corr_grad_out: Optional[List[torch.Tensor]] = None,
    n_rep: int = 0,
    emit_stage1: bool = False,
    out_stage1_rep: Optional[List[torch.Tensor]] = None,
    compute_grads: bool = True,
    out_i_ki_ext: Optional[List[torch.Tensor]] = None,
    out_cf_wf: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Computes exact smoothed pathwise payoff adjoints w.r.t. log_perf (M, D, N) and fwd_perf."""
    _ = (lx_ckpt, r_nat_mid_t, n_sub)
    m_rem, d_assets, n_tot = perf_m_d_n.shape
    dev = perf_m_d_n.device
    p64 = perf_m_d_n.to(dtype=torch.float64).detach().requires_grad_(compute_grads)

    eff_opt_code = int(opt_code)
    w_call, _ = _eval_index_and_weights_torch(p64, call_code, inv_perm)
    w_cpn = w_call if cpn_code == call_code else _eval_index_and_weights_torch(p64, cpn_code, inv_perm)[0]
    if is_long_basket:
        w_opt = w_call if call_code == 0 else (w_cpn if cpn_code == 0 else _eval_index_and_weights_torch(p64, 0, inv_perm)[0])
        w_avg_opt = w_call if call_code == 2 else (w_cpn if cpn_code == 2 else _eval_index_and_weights_torch(p64, 2, inv_perm)[0])
    elif eff_opt_code == call_code:
        w_opt = w_call
        w_avg_opt = None
    elif eff_opt_code == cpn_code:
        w_opt = w_cpn
        w_avg_opt = None
    else:
        w_opt = _eval_index_and_weights_torch(p64, eff_opt_code, inv_perm)[0]
        w_avg_opt = None

    tau_prev = np.concatenate([[0.0], tau_obs[:-1]])
    dt_period = np.maximum(tau_obs - tau_prev, 1e-8)

    if vol_atm_obs is not None:
        sig_est = np.maximum(np.mean(vol_atm_obs, axis=0), 0.05)
    else:
        sig_est = np.full(m_rem, 0.25, dtype=np.float64)
    eps_vec_np = np.clip(
        0.65 * sig_est * np.sqrt(dt_period),
        max(float(barrier_eps_min), 0.015),
        max(float(barrier_eps_max), 0.08),
    )
    eps_vec = torch.tensor(eps_vec_np, dtype=torch.float64, device=dev).view(m_rem, 1)

    def _ramp_up_vec(x: torch.Tensor, bar: torch.Tensor, sp: float) -> torch.Tensor:
        if abs(sp) > 1e-6:
            lo = torch.minimum(bar, bar + sp)
            hi = torch.maximum(bar, bar + sp)
            return ((x - lo) / (hi - lo).clamp(min=1e-12)).clamp(0.0, 1.0)
        half = 0.5 * eps_vec
        return ((x - (bar - half)) / eps_vec).clamp(0.0, 1.0)

    ko_raw = _ramp_up_vec(w_call, ko_barriers.view(m_rem, 1), ko_spread)
    g_ko_all = torch.where(is_callable.view(m_rem, 1) > 0.5, ko_raw, torch.zeros_like(ko_raw))

    eff_cpn_sp = -cpn_spread if cpn_dir_down else cpn_spread
    cpn_ramp = _ramp_up_vec(w_cpn, cpn_barriers.view(m_rem, 1), eff_cpn_sp)
    cpn_hit = (1.0 - cpn_ramp) if cpn_dir_down else cpn_ramp
    cpn_ind = (1.0 - cpn_hit) if cpn_type_out else cpn_hit
    g_cpn_raw = torch.where(is_guar.view(m_rem, 1) > 0.5, torch.ones_like(cpn_ind), cpn_ind)
    eff_g_cpn_all = torch.maximum(g_cpn_raw, g_ko_all) if force_cpn_ac else g_cpn_raw

    surv = torch.ones(n_tot, dtype=torch.float64, device=dev)
    c_mem = torch.full((n_tot,), float(c_mem_0), dtype=torch.float64, device=dev)
    pv_total = torch.zeros(n_tot, dtype=torch.float64, device=dev)

    for m in range(m_rem):
        g_k = g_ko_all[m]
        g_c = eff_g_cpn_all[m]
        p_m = surv * g_k
        c_m = cpn_rates[m]
        if is_memory:
            eff_c = c_m + c_mem
            cpn_cf = surv * g_c * eff_c + p_m * float(ac_cpn)
            c_mem = surv * eff_c * (1.0 - g_c)
        else:
            cpn_cf = surv * g_c * c_m + p_m * float(ac_cpn)
        pv_total = pv_total + df_pay[m] * cpn_cf
        if not no_notional:
            pv_total = pv_total + df_pay[m] * p_m
        if w_fund is not None:
            pv_total = pv_total - w_fund[m] * surv
        surv = (surv - p_m).clamp(min=0.0)

    if not no_notional:
        pv_total = pv_total + df_pay[m_rem - 1] * surv

    bar_lv_cont: torch.Tensor | None = None
    lv_in: torch.Tensor | None = None
    corr_in: torch.Tensor | None = None
    ki_mode_norm = str(ki_monitoring_type or "Discrete").strip().lower()
    if has_ki and ki_barrier > 0.0 and i_ki_past < 0.5:
        if (
            ki_mode_norm in ("continuous", "daily")
            and lv_tables_t is not None
            and corr_t is not None
            and fwd_perf_t is not None
        ):
            lv_in = lv_tables_t.to(dtype=torch.float64).detach().requires_grad_(compute_grads)
            corr_in = corr_t.to(dtype=torch.float64).detach().requires_grad_(compute_grads)
            p_noki = torch.ones(n_tot, dtype=torch.float64, device=dev)
            w_0 = _eval_index_and_weights_torch(
                torch.ones((1, d_assets, 1), dtype=torch.float64, device=dev),
                0 if is_long_basket else eff_opt_code,
                inv_perm,
            )[0][0, 0]
            w_prev_k = w_0.expand(n_tot)
            n_t_lv, n_y_lv = lv_in.shape[1], lv_in.shape[2]
            y_min_arr = np.broadcast_to(np.asarray(y_min, dtype=np.float64), (d_assets,))
            y_max_arr = np.broadcast_to(np.asarray(y_max, dtype=np.float64), (d_assets,))
            y_min_t = torch.tensor(y_min_arr, dtype=torch.float64, device=dev).view(d_assets, 1)
            inv_dy_t = torch.tensor(
                float(n_y_lv - 1) / np.maximum(y_max_arr - y_min_arr, 1e-12),
                dtype=torch.float64,
                device=dev,
            ).view(d_assets, 1)
            ki_eval_idx = 0 if is_long_basket else eff_opt_code
            b_ki_base = float(strike) if is_long_basket else float(ki_barrier)
            is_daily_bgk = ki_mode_norm == "daily"
            bgk_coef = 0.5826 * math.sqrt(1.0 / 252.0)
            for m in range(m_rem):
                w_m, dw_m = _eval_index_and_weights_torch(p64[m : m + 1], ki_eval_idx, inv_perm)
                w_cur = w_m[0]
                dw_p = (dw_m[0] * p64[m]) / w_cur.clamp(min=1e-12)
                dt_m = float(dt_period[m])
                t_m = float(0.5 * (tau_prev[m] + tau_obs[m]))
                ut = min(max(t_m * float(n_t_lv - 1) / max(float(tau_obs[-1]), 1e-4), 0.0), float(n_t_lv - 1) - 1e-4)
                it0 = int(ut)
                it1 = min(it0 + 1, n_t_lv - 1)
                ft = ut - float(it0)
                lx_m = torch.log((p64[m] / fwd_perf_t[:, m : m + 1].clamp(min=1e-12)).clamp(min=1e-12))
                uy = ((lx_m - y_min_t) * inv_dy_t).clamp(0.0, float(n_y_lv - 1) - 1e-4)
                iy0 = uy.to(torch.long)
                iy1 = (iy0 + 1).clamp(max=n_y_lv - 1)
                fy = uy - iy0.to(torch.float64)
                s00 = torch.gather(lv_in[:, it0, :], 1, iy0)
                s01 = torch.gather(lv_in[:, it0, :], 1, iy1)
                s10 = torch.gather(lv_in[:, it1, :], 1, iy0)
                s11 = torch.gather(lv_in[:, it1, :], 1, iy1)
                sig_d = (1.0 - fy) * ((1.0 - ft) * s00 + ft * s10) + fy * ((1.0 - ft) * s01 + ft * s11)
                dw_sig = dw_p * sig_d
                var_w = (dw_sig * (corr_in @ dw_sig)).sum(dim=0).clamp(min=1e-10)
                b_ki_step: torch.Tensor | float = (
                    (b_ki_base * torch.exp(-bgk_coef * torch.sqrt(var_w))).clamp(min=1e-6)
                    if is_daily_bgk
                    else b_ki_base
                )
                alive_pair = (w_prev_k > b_ki_step) & (w_cur > b_ki_step)
                w_p_s = torch.where(alive_pair, w_prev_k, b_ki_step + 0.1)
                w_c_s = torch.where(alive_pair, w_cur, b_ki_step + 0.1)
                eta = (2.0 * torch.log(w_p_s / b_ki_step) * torch.log(w_c_s / b_ki_step)) / (var_w * dt_m)
                p_step = torch.where(alive_pair, 1.0 - torch.exp(-eta.clamp(min=0.0)), torch.zeros_like(eta))
                p_noki = p_noki * p_step
                w_prev_k = w_cur
            i_ki_t = 1.0 - p_noki
        elif ki_mode_norm == "discrete":
            lv_in = None
            b_ki_base = float(strike) if is_long_basket else float(ki_barrier)
            p_noki_d = torch.ones(n_tot, dtype=torch.float64, device=dev)
            for m_k in range(m_rem):
                if abs(ki_spread) > 1e-6:
                    blo_m = min(b_ki_base, b_ki_base + float(ki_spread))
                    bhi_m = max(b_ki_base, b_ki_base + float(ki_spread))
                    g_k_m = ((bhi_m - w_opt[m_k]) / max(bhi_m - blo_m, 1e-12)).clamp(0.0, 1.0)
                else:
                    eps_m = float(eps_vec_np[m_k])
                    bhi_m = b_ki_base + 0.5 * eps_m
                    g_k_m = ((bhi_m - w_opt[m_k]) / max(eps_m, 1e-12)).clamp(0.0, 1.0)
                p_noki_d = p_noki_d * (1.0 - g_k_m)
            i_ki_t = 1.0 - p_noki_d
        else:
            b_ki_base = float(strike) if is_long_basket else float(ki_barrier)
            if abs(ki_spread) > 1e-6:
                blo_k = min(b_ki_base, b_ki_base + float(ki_spread))
                bhi_k = max(b_ki_base, b_ki_base + float(ki_spread))
                g_ki = ((bhi_k - w_opt[m_rem - 1]) / max(bhi_k - blo_k, 1e-12)).clamp(0.0, 1.0)
            else:
                eps_term = float(eps_vec_np[-1])
                bhi_k = b_ki_base + 0.5 * eps_term
                g_ki = ((bhi_k - w_opt[m_rem - 1]) / max(eps_term, 1e-12)).clamp(0.0, 1.0)
            i_ki_t = g_ki
    else:
        i_ki_t = torch.full((n_tot,), float(i_ki_past), dtype=torch.float64, device=dev)
        lv_in = None
    if out_i_ki_ext is not None:
        out_i_ki_ext.append(i_ki_t.detach().to(dtype=torch.float64))

    has_os = one_star_strike is not None and float(one_star_strike) > 0.0
    if has_os and one_star_strike is not None:
        w_best = w_opt if ((not is_long_basket) and eff_opt_code == 1) else _eval_index_and_weights_torch(p64, 1, inv_perm)[0]
        k_os = float(one_star_strike)
        if one_star_eps <= 1e-6:
            star_ind = (w_best[m_rem - 1] >= k_os).to(dtype=torch.float64)
        else:
            half_os = 0.5 * one_star_eps
            star_ind = ((w_best[m_rem - 1] - (k_os - half_os)) / one_star_eps).clamp(0.0, 1.0)
    else:
        star_ind = torch.zeros(n_tot, dtype=torch.float64, device=dev)

    opt_pay = torch.zeros(n_tot, dtype=torch.float64, device=dev)
    if is_long_basket:
        assert w_avg_opt is not None
        ki_gate = i_ki_t if has_ki else 1.0
        opt_pay = opt_pay - (1.0 - w_avg_opt[m_rem - 1]) * ki_gate * (1.0 - star_ind)
        put_active = False
        put_intr = torch.zeros(n_tot, dtype=torch.float64, device=dev)
    else:
        has_low_k = lower_strike is not None
        if has_low_k:
            assert lower_strike is not None
            k_low = float(lower_strike)
            put_active = float(strike) > k_low and float(option_gearing) > 0.0
            put_intr = (float(strike) - w_opt[m_rem - 1]).clamp(min=0.0) - (k_low - w_opt[m_rem - 1]).clamp(min=0.0)
        else:
            put_active = float(option_gearing) > 0.0
            put_intr = (float(strike) - w_opt[m_rem - 1]).clamp(min=0.0)
        if put_active:
            ki_gate = i_ki_t if has_ki else 1.0
            opt_pay = opt_pay - float(option_gearing) * ki_gate * (1.0 - star_ind) * put_intr
    if part_call_strike is not None and float(part_call_strike) >= 0.0 and float(part_call_rate) != 0.0:
        opt_pay = opt_pay + float(part_call_rate) * (w_opt[m_rem - 1] - float(part_call_strike)).clamp(min=0.0)
    if has_os and float(one_star_coupon) != 0.0:
        opt_pay = opt_pay + star_ind * float(one_star_coupon)

    if not compute_grads:
        grad_p_res = torch.zeros_like(p64)
        bar_lv_cont = None
    else:
        loss = (pv_total + df_pay[m_rem - 1] * surv * opt_pay).mean()
        if not loss.requires_grad:
            grad_p_res = torch.zeros_like(p64)
            bar_lv_cont = torch.zeros_like(lv_in) if lv_in is not None else None
            if corr_grad_out is not None and corr_in is not None:
                corr_grad_out.append(torch.zeros_like(corr_in))
        elif lv_in is not None and corr_in is not None and lv_in.requires_grad and corr_in.requires_grad:
            grad_p_raw, grad_lv_raw, grad_corr_raw = torch.autograd.grad(
                loss, (p64, lv_in, corr_in), allow_unused=True
            )
            grad_p_res = grad_p_raw if grad_p_raw is not None else torch.zeros_like(p64)
            bar_lv_cont = grad_lv_raw if grad_lv_raw is not None else torch.zeros_like(lv_in)
            if corr_grad_out is not None:
                corr_grad_out.append(
                    grad_corr_raw.detach() if grad_corr_raw is not None else torch.zeros_like(corr_in)
                )
        elif lv_in is not None and lv_in.requires_grad:
            grad_p_raw, grad_lv_raw = torch.autograd.grad(loss, (p64, lv_in), allow_unused=True)
            grad_p_res = grad_p_raw if grad_p_raw is not None else torch.zeros_like(p64)
            bar_lv_cont = grad_lv_raw if grad_lv_raw is not None else torch.zeros_like(lv_in)
        else:
            grad_p_raw = torch.autograd.grad(loss, p64, allow_unused=True)[0]
            grad_p_res = grad_p_raw if grad_p_raw is not None else torch.zeros_like(p64)

    if emit_stage1 and n_rep > 0 and out_stage1_rep is not None:
        with torch.no_grad():
            n_r_blk = max(n_tot // n_rep, 1)
            wc_d = w_call.detach()
            wcp_d = w_cpn.detach()
            wo_d = w_opt.detach()
            s1_mat = torch.zeros((n_r_blk, 3 * m_rem + 6), dtype=torch.float64, device=dev)
            cf_wf_mat = torch.zeros((2, m_rem), dtype=torch.float64, device=dev) if out_cf_wf is not None else None
            if abs(ko_spread) > 1e-6:
                gk_s1_all = g_ko_all.detach()
            else:
                gk_s1_all = torch.where(
                    is_callable.view(m_rem, 1) > 0.5,
                    (wc_d >= ko_barriers.view(m_rem, 1)).to(dtype=torch.float64),
                    torch.zeros_like(wc_d),
                )
            if abs(cpn_spread) > 1e-6:
                egc_s1_all = eff_g_cpn_all.detach()
            else:
                h_cpn = (wcp_d <= cpn_barriers.view(m_rem, 1)) if cpn_dir_down else (wcp_d >= cpn_barriers.view(m_rem, 1))
                ind_h = (1.0 - h_cpn.to(dtype=torch.float64)) if cpn_type_out else h_cpn.to(dtype=torch.float64)
                gc_s1 = torch.where(is_guar.view(m_rem, 1) > 0.5, torch.ones_like(ind_h), ind_h)
                egc_s1_all = torch.maximum(gc_s1, gk_s1_all) if force_cpn_ac else gc_s1
            surv_s1 = torch.ones(n_tot, dtype=torch.float64, device=dev)
            cmem_s1 = torch.full((n_tot,), float(c_mem_0), dtype=torch.float64, device=dev)
            pv_s1 = torch.zeros(n_tot, dtype=torch.float64, device=dev)
            for m in range(m_rem):
                pm_s1 = surv_s1 * gk_s1_all[m]
                qc_s1 = surv_s1 * egc_s1_all[m]
                if is_memory:
                    ec_s1 = cpn_rates[m] + cmem_s1
                    ccf_s1 = qc_s1 * ec_s1 + pm_s1 * float(ac_cpn)
                    cmem_s1 = surv_s1 * ec_s1 * (1.0 - egc_s1_all[m])
                else:
                    ccf_s1 = qc_s1 * cpn_rates[m] + pm_s1 * float(ac_cpn)
                wf_m_s1 = w_fund[m] if w_fund is not None else 0.0
                cf_m_s1 = ccf_s1 if no_notional else (pm_s1 + ccf_s1)
                pv_s1 = pv_s1 + df_pay[m] * cf_m_s1 - wf_m_s1 * surv_s1
                if cf_wf_mat is not None:
                    cf_wf_mat[0, m] = cf_m_s1.mean()
                    cf_wf_mat[1, m] = surv_s1.mean()
                s1_mat[:, m] = pm_s1.view(n_r_blk, n_rep).mean(dim=1)
                s1_mat[:, m_rem + m] = qc_s1.view(n_r_blk, n_rep).mean(dim=1)
                s1_mat[:, 2 * m_rem + m] = ccf_s1.view(n_r_blk, n_rep).mean(dim=1)
                surv_s1 = (surv_s1 - pm_s1).clamp(min=0.0)
            if has_ki and ki_barrier > 0.0 and i_ki_past < 0.5:
                if ki_mode_norm in ("continuous", "daily", "discrete") or abs(ki_spread) > 1e-6:
                    ki_s1 = i_ki_t.detach()
                else:
                    b_ki_s1 = float(strike) if is_long_basket else float(ki_barrier)
                    ki_s1 = (wo_d[m_rem - 1] < b_ki_s1).to(dtype=torch.float64)
            else:
                ki_s1 = (
                    torch.ones((n_tot,), dtype=torch.float64, device=dev)
                    if not has_ki
                    else torch.full((n_tot,), float(i_ki_past), dtype=torch.float64, device=dev)
                )
            star_s1 = star_ind.detach()
            if is_long_basket:
                assert w_avg_opt is not None
                u_put_s1 = surv_s1 * ki_s1 * (1.0 - star_s1) * (1.0 - w_avg_opt.detach()[m_rem - 1])
                put_term_s1 = -u_put_s1
            else:
                u_put_s1 = (
                    surv_s1 * ki_s1 * (1.0 - star_s1) * put_intr.detach()
                    if put_active
                    else torch.zeros(n_tot, dtype=torch.float64, device=dev)
                )
                put_term_s1 = -float(option_gearing) * u_put_s1
            has_pc = part_call_strike is not None and float(part_call_strike) >= 0.0 and float(part_call_rate) != 0.0
            u_call_s1 = (
                surv_s1 * (wo_d[m_rem - 1] - float(part_call_strike)).clamp(min=0.0)
                if has_pc and part_call_strike is not None
                else torch.zeros(n_tot, dtype=torch.float64, device=dev)
            )
            p_os_s1 = surv_s1 * star_s1 if has_os else torch.zeros(n_tot, dtype=torch.float64, device=dev)
            l_os_s1 = p_os_s1 * float(one_star_coupon) if (has_os and float(one_star_coupon) != 0.0) else torch.zeros(n_tot, dtype=torch.float64, device=dev)
            opt_s1 = put_term_s1 + (float(part_call_rate) * u_call_s1 if has_pc else 0.0) + l_os_s1
            term_s1 = opt_s1 if no_notional else (surv_s1 + opt_s1)
            pv_s1 = pv_s1 + df_pay[m_rem - 1] * term_s1
            c0 = 3 * m_rem
            s1_mat[:, c0 + 0] = u_put_s1.view(n_r_blk, n_rep).mean(dim=1)
            s1_mat[:, c0 + 1] = u_call_s1.view(n_r_blk, n_rep).mean(dim=1)
            s1_mat[:, c0 + 2] = l_os_s1.view(n_r_blk, n_rep).mean(dim=1)
            s1_mat[:, c0 + 3] = p_os_s1.view(n_r_blk, n_rep).mean(dim=1)
            s1_mat[:, c0 + 4] = pv_s1.view(n_r_blk, n_rep).mean(dim=1)
            s1_mat[:, c0 + 5] = (pv_s1 * pv_s1).view(n_r_blk, n_rep).mean(dim=1)
            out_stage1_rep.append(s1_mat)
            if cf_wf_mat is not None and out_cf_wf is not None:
                cf_wf_mat[0, m_rem - 1] += term_s1.mean()
                out_cf_wf.append(cf_wf_mat)

    bar_lx_post = grad_p_res * p64.detach()
    bar_ln_fwd = bar_lx_post.sum(dim=2).transpose(0, 1).contiguous()
    return bar_lx_post.to(dtype=torch.float32), bar_ln_fwd.to(dtype=torch.float64), bar_lv_cont


if _HAS_NUMBA:
    @numba.njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
    def _nb_eval_index_and_weights(
        perf_m_d_n: npt.NDArray[np.float64],
        code: int,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        m_rem, d_assets, n_tot = perf_m_d_n.shape
        w_out = np.empty((m_rem, n_tot), dtype=np.float64)
        idx_out = np.empty((m_rem, n_tot), dtype=np.int32)
        inv_d = 1.0 / float(d_assets)
        for mn in numba.prange(m_rem * n_tot):
            m = mn // n_tot
            n = mn - m * n_tot
            if code == 0:
                best_v = perf_m_d_n[m, 0, n]
                best_k = 0
                for d in range(1, d_assets):
                    v = perf_m_d_n[m, d, n]
                    if v < best_v:
                        best_v = v
                        best_k = d
                w_out[m, n] = best_v
                idx_out[m, n] = best_k
            elif code == 1:
                best_v = perf_m_d_n[m, 0, n]
                best_k = 0
                for d in range(1, d_assets):
                    v = perf_m_d_n[m, d, n]
                    if v > best_v:
                        best_v = v
                        best_k = d
                w_out[m, n] = best_v
                idx_out[m, n] = best_k
            else:
                acc = 0.0
                for d in range(d_assets):
                    acc += perf_m_d_n[m, d, n]
                w_out[m, n] = acc * inv_d
                idx_out[m, n] = 0
        return w_out, idx_out


def _eval_index_and_weights_np(
    perf_m_d_n: npt.NDArray[np.float64],
    code: int,
) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.int32]]]:
    """Vectorized NumPy/Numba evaluator for codes {0: WorstOf, 1: BestOf, 2: AverageOf}."""
    if _HAS_NUMBA and code in (0, 1, 2):
        w_out, idx_out = _nb_eval_index_and_weights(perf_m_d_n, int(code))
        return w_out, (idx_out if code in (0, 1) else None)
    if code == 0:
        idx = np.argmin(perf_m_d_n, axis=1)
        return np.take_along_axis(perf_m_d_n, idx[:, None, :], axis=1)[:, 0, :], idx.astype(np.int32, copy=False)
    if code == 1:
        idx = np.argmax(perf_m_d_n, axis=1)
        return np.take_along_axis(perf_m_d_n, idx[:, None, :], axis=1)[:, 0, :], idx.astype(np.int32, copy=False)
    return np.mean(perf_m_d_n, axis=1), None


if _HAS_NUMBA:
    @numba.njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
    def _nb_payoff_aad_bwd_and_s1(
        perf_m_d_n: npt.NDArray[np.float64],
        g_ko_all: npt.NDArray[np.float64],
        dg_ko_all: npt.NDArray[np.float64],
        eff_g_cpn_all: npt.NDArray[np.float64],
        dgc_from_cpn: npt.NDArray[np.float64],
        dgc_from_ko: npt.NDArray[np.float64],
        surv_pre_arr: npt.NDArray[np.float64],
        eff_c_arr: npt.NDArray[np.float64],
        surv_end: npt.NDArray[np.float64],
        opt_pay: npt.NDArray[np.float64],
        d_opt_d_wo: npt.NDArray[np.float64],
        d_opt_d_avg: npt.NDArray[np.float64],
        d_opt_d_wb: npt.NDArray[np.float64],
        d_opt_d_iki: npt.NDArray[np.float64],
        d_iki_dwo_last: npt.NDArray[np.float64],
        d_iki_dwo_arr: npt.NDArray[np.float64],
        has_d_iki_arr: bool,
        df_pay: npt.NDArray[np.float64],
        w_fund_arr: npt.NDArray[np.float64],
        has_w_fund: bool,
        idx_call: npt.NDArray[np.int32],
        has_idx_call: bool,
        idx_cpn: npt.NDArray[np.int32],
        has_idx_cpn: bool,
        idx_opt: npt.NDArray[np.int32],
        has_idx_opt: bool,
        idx_best_last: npt.NDArray[np.int32],
        has_idx_best: bool,
        call_code: int,
        cpn_code: int,
        eff_opt_code: int,
        is_long_basket: bool,
        is_memory: bool,
        no_notional: bool,
        has_os: bool,
        ac_c: float,
        bar_lx_post_np: npt.NDArray[np.float64],
        bar_ln_fwd_np: npt.NDArray[np.float64],
    ) -> None:
        m_rem, d_assets, n_tot = perf_m_d_n.shape
        m_last = m_rem - 1
        scale = 1.0 / float(n_tot)
        inv_d = 1.0 / float(d_assets)
        df_last = float(df_pay[m_last])
        same_cpn_call = cpn_code == call_code
        tile_n = 128
        n_tiles = (n_tot + tile_n - 1) // tile_n
        bar_ln_tiles = np.zeros((n_tiles, d_assets, m_rem), dtype=np.float64)

        for t_i in numba.prange(n_tiles):
            n_start = t_i * tile_n
            n_end = min(n_start + tile_n, n_tot)
            n_len = n_end - n_start
            bar_surv = np.empty(n_len, dtype=np.float64)
            bar_c_mem = np.zeros(n_len, dtype=np.float64)
            bar_last_wo = np.empty(n_len, dtype=np.float64)
            bar_last_avg = np.empty(n_len, dtype=np.float64)
            bar_last_wb = np.empty(n_len, dtype=np.float64)
            bar_iki_disc = np.empty(n_len, dtype=np.float64)
            for k in range(n_len):
                n = n_start + k
                op = opt_pay[n]
                se = surv_end[n]
                term_mult = op if no_notional else (1.0 + op)
                bar_surv[k] = scale * df_last * term_mult
                d_iki_term = 0.0 if has_d_iki_arr else (d_opt_d_iki[n] * d_iki_dwo_last[n])
                s_df_se = scale * df_last * se
                bar_last_wo[k] = s_df_se * (d_opt_d_wo[n] + d_iki_term)
                bar_last_avg[k] = s_df_se * d_opt_d_avg[n]
                bar_last_wb[k] = s_df_se * d_opt_d_wb[n]
                bar_iki_disc[k] = s_df_se * d_opt_d_iki[n]

            for m in range(m_rem - 1, -1, -1):
                df_m = float(df_pay[m])
                wf_m = float(w_fund_arr[m]) if has_w_fund else 0.0
                s_df_m = scale * df_m
                s_wf_m = scale * wf_m
                pm_off = (0.0 if no_notional else s_df_m) + s_df_m * ac_c
                is_last = m == m_last
                for k in range(n_len):
                    n = n_start + k
                    surv_pre = surv_pre_arr[m, n]
                    g_k = g_ko_all[m, n]
                    eff_g_c = eff_g_cpn_all[m, n]
                    eff_c = eff_c_arr[m, n]
                    bs = bar_surv[k]
                    bcm = bar_c_mem[k]

                    bar_pm = pm_off - bs
                    bar_eff_g_c = s_df_m * surv_pre * eff_c
                    if is_memory:
                        bar_eff_g_c -= bcm * surv_pre * eff_c

                    bs_next = bs + bar_pm * g_k + s_df_m * eff_g_c * eff_c - s_wf_m
                    if is_memory:
                        bs_next += bcm * eff_c * (1.0 - eff_g_c)
                        bar_c_mem[k] = s_df_m * surv_pre * eff_g_c + bcm * surv_pre * (1.0 - eff_g_c)
                    bar_surv[k] = bs_next

                    bw_call = (bar_pm * surv_pre) * dg_ko_all[m, n] + bar_eff_g_c * dgc_from_ko[m, n]
                    bw_cpn = bar_eff_g_c * dgc_from_cpn[m, n]
                    bw_opt = (bar_iki_disc[k] * d_iki_dwo_arr[m, n]) if has_d_iki_arr else 0.0
                    bw_avg = 0.0
                    bw_wb = 0.0
                    if is_last:
                        bw_opt += bar_last_wo[k]
                        if is_long_basket:
                            bw_avg = bar_last_avg[k]
                        if has_os:
                            bw_wb = bar_last_wb[k]

                    eff_wc_tot = bw_call + (bw_cpn if same_cpn_call else 0.0)
                    uni_w = 0.0
                    if call_code == 2:
                        uni_w += eff_wc_tot * inv_d
                    if (not same_cpn_call) and cpn_code == 2:
                        uni_w += bw_cpn * inv_d
                    if is_long_basket:
                        uni_w += bw_avg * inv_d
                    elif eff_opt_code == 2:
                        uni_w += bw_opt * inv_d

                    for d in range(d_assets):
                        bp = uni_w
                        if has_idx_call and idx_call[m, n] == d:
                            bp += eff_wc_tot
                        if (not same_cpn_call) and has_idx_cpn and idx_cpn[m, n] == d:
                            bp += bw_cpn
                        if has_idx_opt and idx_opt[m, n] == d:
                            bp += bw_opt
                        if is_last and has_os and has_idx_best and idx_best_last[n] == d:
                            bp += bw_wb
                        lx_v = bp * perf_m_d_n[m, d, n]
                        bar_lx_post_np[m, d, n] = lx_v
                        bar_ln_tiles[t_i, d, m] += lx_v

        for t_i in range(n_tiles):
            for d in range(d_assets):
                for m in range(m_rem):
                    bar_ln_fwd_np[d, m] += bar_ln_tiles[t_i, d, m]

    @numba.njit(cache=True)  # type: ignore[untyped-decorator]
    def _nb_payoff_aad_s1_loop(
        gk_s1_all: npt.NDArray[np.float64],
        egc_s1_all: npt.NDArray[np.float64],
        cpn_rates: npt.NDArray[np.float64],
        df_pay: npt.NDArray[np.float64],
        w_fund_arr: npt.NDArray[np.float64],
        has_w_fund: bool,
        ki_s1: npt.NDArray[np.float64],
        star_ind: npt.NDArray[np.float64],
        put_intr: npt.NDArray[np.float64],
        call_intr: npt.NDArray[np.float64],
        w_avg_last: npt.NDArray[np.float64],
        s1_mat: npt.NDArray[np.float64],
        cf_wf_out: npt.NDArray[np.float64],
        want_cf_wf: bool,
        n_rep: int,
        c_mem_0: float,
        ac_cpn: float,
        option_gearing: float,
        part_call_rate: float,
        one_star_coupon: float,
        is_memory: bool,
        no_notional: bool,
        is_long_basket: bool,
        put_active: bool,
        has_pc: bool,
        has_os: bool,
    ) -> None:
        m_rem, n_tot = gk_s1_all.shape
        n_r_blk = max(n_tot // n_rep, 1)
        m_last = m_rem - 1
        inv_n_tot = 1.0 / float(n_tot)
        inv_rep = 1.0 / float(n_rep)
        surv_s1 = np.ones(n_tot, dtype=np.float64)
        cmem_s1 = np.full(n_tot, c_mem_0, dtype=np.float64)
        pv_s1 = np.zeros(n_tot, dtype=np.float64)

        for m in range(m_rem):
            c_r = float(cpn_rates[m])
            df_m = float(df_pay[m])
            wf_m = float(w_fund_arr[m]) if has_w_fund else 0.0
            cf_sum = 0.0
            sv_sum = 0.0
            for r_i in range(n_r_blk):
                base_n = r_i * n_rep
                pm_sum = 0.0
                qc_sum = 0.0
                ccf_sum = 0.0
                for k in range(n_rep):
                    n = base_n + k
                    sv = surv_s1[n]
                    gk = gk_s1_all[m, n]
                    egc = egc_s1_all[m, n]
                    pm = sv * gk
                    qc = sv * egc
                    if is_memory:
                        ec = c_r + cmem_s1[n]
                        ccf = qc * ec + pm * ac_cpn
                        cmem_s1[n] = sv * ec * (1.0 - egc)
                    else:
                        ccf = qc * c_r + pm * ac_cpn
                    cf_m = ccf if no_notional else (pm + ccf)
                    pv_s1[n] += df_m * cf_m - wf_m * sv
                    rem_s = sv - pm
                    surv_s1[n] = rem_s if rem_s > 0.0 else 0.0
                    pm_sum += pm
                    qc_sum += qc
                    ccf_sum += ccf
                    cf_sum += cf_m
                    sv_sum += sv
                s1_mat[r_i, m] = pm_sum * inv_rep
                s1_mat[r_i, m_rem + m] = qc_sum * inv_rep
                s1_mat[r_i, 2 * m_rem + m] = ccf_sum * inv_rep
            if want_cf_wf:
                cf_wf_out[0, m] = cf_sum * inv_n_tot
                cf_wf_out[1, m] = sv_sum * inv_n_tot

        df_last = float(df_pay[m_last])
        c0 = 3 * m_rem
        term_cf_sum = 0.0
        for r_i in range(n_r_blk):
            base_n = r_i * n_rep
            s_uput = 0.0
            s_ucall = 0.0
            s_los = 0.0
            s_pos = 0.0
            s_pv = 0.0
            s_pv2 = 0.0
            for k in range(n_rep):
                n = base_n + k
                sv = surv_s1[n]
                st_n = star_ind[n]
                if is_long_basket:
                    u_put = sv * ki_s1[n] * (1.0 - st_n) * (1.0 - w_avg_last[n])
                    put_t = -u_put
                else:
                    u_put = (sv * ki_s1[n] * (1.0 - st_n) * put_intr[n]) if put_active else 0.0
                    put_t = -option_gearing * u_put
                u_call = (sv * call_intr[n]) if has_pc else 0.0
                p_os = (sv * st_n) if has_os else 0.0
                l_os = (p_os * one_star_coupon) if (has_os and one_star_coupon != 0.0) else 0.0
                opt_t = put_t + (part_call_rate * u_call if has_pc else 0.0) + l_os
                term_t = opt_t if no_notional else (sv + opt_t)
                pv_n = pv_s1[n] + df_last * term_t
                s_uput += u_put
                s_ucall += u_call
                s_los += l_os
                s_pos += p_os
                s_pv += pv_n
                s_pv2 += pv_n * pv_n
                term_cf_sum += term_t
            s1_mat[r_i, c0 + 0] = s_uput * inv_rep
            s1_mat[r_i, c0 + 1] = s_ucall * inv_rep
            s1_mat[r_i, c0 + 2] = s_los * inv_rep
            s1_mat[r_i, c0 + 3] = s_pos * inv_rep
            s1_mat[r_i, c0 + 4] = s_pv * inv_rep
            s1_mat[r_i, c0 + 5] = s_pv2 * inv_rep
        if want_cf_wf:
            cf_wf_out[0, m_last] += term_cf_sum * inv_n_tot


def _compute_payoff_aad_state_adjoints_cpu_np(
    perf_m_d_n: npt.NDArray[np.float64],
    *,
    ko_barriers: npt.NDArray[np.float64],
    is_callable: npt.NDArray[np.float64],
    cpn_barriers: npt.NDArray[np.float64],
    is_guar: npt.NDArray[np.float64],
    cpn_rates: npt.NDArray[np.float64],
    df_pay: npt.NDArray[np.float64],
    w_fund: Optional[npt.NDArray[np.float64]],
    call_code: int,
    cpn_code: int,
    opt_code: int,
    ko_spread: float,
    cpn_spread: float,
    ki_spread: float,
    cpn_dir_down: bool,
    cpn_type_out: bool,
    is_memory: bool,
    force_cpn_ac: bool,
    no_notional: bool,
    is_long_basket: bool,
    has_ki: bool,
    ki_barrier: float,
    i_ki_past: float,
    ki_monitoring_type: str,
    tau_obs: npt.NDArray[np.float64],
    vol_atm_obs: Optional[npt.NDArray[np.float64]],
    strike: float,
    lower_strike: Optional[float],
    option_gearing: float,
    ac_cpn: float,
    c_mem_0: float,
    part_call_strike: Optional[float],
    part_call_rate: float,
    one_star_strike: Optional[float],
    one_star_coupon: float,
    one_star_eps: float,
    barrier_eps_min: float,
    barrier_eps_max: float,
    n_rep: int,
    emit_stage1: bool = False,
    compute_grads: bool = True,
    want_cf_wf: bool = False,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]]:
    """Closed-form float64 analytical payoff AAD & Stage-1 evaluation in pure NumPy for codes {0,1,2}."""
    m_rem, d_assets, n_tot = perf_m_d_n.shape
    eff_opt_code = int(opt_code)
    w_call, idx_call = _eval_index_and_weights_np(perf_m_d_n, call_code)
    if cpn_code == call_code:
        w_cpn, idx_cpn = w_call, idx_call
    else:
        w_cpn, idx_cpn = _eval_index_and_weights_np(perf_m_d_n, cpn_code)

    w_avg_opt: Optional[npt.NDArray[np.float64]] = None
    if is_long_basket:
        w_opt, idx_opt = (w_call, idx_call) if call_code == 0 else ((w_cpn, idx_cpn) if cpn_code == 0 else _eval_index_and_weights_np(perf_m_d_n, 0))
        w_avg_opt = w_call if call_code == 2 else (w_cpn if cpn_code == 2 else _eval_index_and_weights_np(perf_m_d_n, 2)[0])
    elif eff_opt_code == call_code:
        w_opt, idx_opt = w_call, idx_call
    elif eff_opt_code == cpn_code:
        w_opt, idx_opt = w_cpn, idx_cpn
    else:
        w_opt, idx_opt = _eval_index_and_weights_np(perf_m_d_n, eff_opt_code)

    tau_prev = np.concatenate([[0.0], tau_obs[:-1]])
    dt_period = np.maximum(tau_obs - tau_prev, 1e-8)
    sig_est = np.maximum(np.mean(vol_atm_obs, axis=0), 0.05) if vol_atm_obs is not None else np.full(m_rem, 0.25, dtype=np.float64)
    eps_vec_np = np.clip(
        0.65 * sig_est * np.sqrt(dt_period),
        max(float(barrier_eps_min), 0.015),
        max(float(barrier_eps_max), 0.08),
    )[:, None]

    hard_ko = abs(ko_spread) <= 1e-6
    eff_csp = -cpn_spread if cpn_dir_down else cpn_spread
    hard_cpn = abs(eff_csp) <= 1e-6

    lo_ko = np.minimum(ko_barriers, ko_barriers + ko_spread)[:, None] if not hard_ko else (ko_barriers[:, None] - 0.5 * eps_vec_np)
    hi_ko = np.maximum(ko_barriers, ko_barriers + ko_spread)[:, None] if not hard_ko else (ko_barriers[:, None] + 0.5 * eps_vec_np)
    inv_w_ko = 1.0 / np.maximum(hi_ko - lo_ko, 1e-12)
    u_ko = (w_call - lo_ko) * inv_w_ko
    ko_raw = np.clip(u_ko, 0.0, 1.0)
    call_mask_m = (is_callable > 0.5)[:, None]
    g_ko_all = np.where(call_mask_m, ko_raw, 0.0)
    dg_ko_all = np.where(call_mask_m & (u_ko > 0.0) & (u_ko < 1.0), inv_w_ko, 0.0) if compute_grads else np.empty(0, dtype=np.float64)

    lo_cp = np.minimum(cpn_barriers, cpn_barriers + eff_csp)[:, None] if not hard_cpn else (cpn_barriers[:, None] - 0.5 * eps_vec_np)
    hi_cp = np.maximum(cpn_barriers, cpn_barriers + eff_csp)[:, None] if not hard_cpn else (cpn_barriers[:, None] + 0.5 * eps_vec_np)
    inv_w_cp = 1.0 / np.maximum(hi_cp - lo_cp, 1e-12)
    u_cp = (w_cpn - lo_cp) * inv_w_cp
    ramp_cp = np.clip(u_cp, 0.0, 1.0)
    hit_cp = (1.0 - ramp_cp) if cpn_dir_down else ramp_cp
    ind_cp = (1.0 - hit_cp) if cpn_type_out else hit_cp
    guar_mask_m = (is_guar > 0.5)[:, None]
    g_cpn_raw = np.where(guar_mask_m, 1.0, ind_cp)
    sign_cpn = (-1.0 if cpn_dir_down else 1.0) * (-1.0 if cpn_type_out else 1.0)
    dg_cpn_raw = np.where((~guar_mask_m) & (u_cp > 0.0) & (u_cp < 1.0), inv_w_cp * sign_cpn, 0.0) if compute_grads else np.empty(0, dtype=np.float64)

    if force_cpn_ac:
        eff_g_cpn_all = np.maximum(g_cpn_raw, g_ko_all)
        if compute_grads:
            w_c_max = np.where(g_cpn_raw > g_ko_all, 1.0, np.where(g_cpn_raw < g_ko_all, 0.0, 0.5))
            dgc_from_cpn = w_c_max * dg_cpn_raw
            dgc_from_ko = (1.0 - w_c_max) * dg_ko_all
        else:
            dgc_from_cpn = np.empty(0, dtype=np.float64)
            dgc_from_ko = np.empty(0, dtype=np.float64)
    else:
        eff_g_cpn_all = g_cpn_raw
        dgc_from_cpn = dg_cpn_raw
        dgc_from_ko = np.zeros_like(dg_ko_all) if compute_grads else np.empty(0, dtype=np.float64)

    surv_pre_arr = np.empty((m_rem, n_tot), dtype=np.float64)
    eff_c_arr = np.empty((m_rem, n_tot), dtype=np.float64)
    surv = np.ones(n_tot, dtype=np.float64)
    c_mem = np.full(n_tot, float(c_mem_0), dtype=np.float64)
    for m in range(m_rem):
        surv_pre_arr[m] = surv
        if is_memory:
            eff_c = cpn_rates[m] + c_mem
            eff_c_arr[m] = eff_c
            c_mem = surv * eff_c * (1.0 - eff_g_cpn_all[m])
        else:
            eff_c_arr[m] = cpn_rates[m]
        surv = np.maximum(surv - surv * g_ko_all[m], 0.0)
    surv_end = surv

    ki_mode_norm = str(ki_monitoring_type or "Discrete").strip().lower()
    m_last = m_rem - 1
    d_iki_dwo_arr: Optional[npt.NDArray[np.float64]] = None
    d_iki_dwo_last = np.zeros(n_tot, dtype=np.float64)
    if has_ki and ki_barrier > 0.0 and i_ki_past < 0.5:
        b_ki_base = float(strike) if is_long_basket else float(ki_barrier)
        if ki_mode_norm == "discrete":
            if abs(ki_spread) > 1e-6:
                blo_k = min(b_ki_base, b_ki_base + float(ki_spread))
                bhi_k = max(b_ki_base, b_ki_base + float(ki_spread))
                inv_wk = np.full((m_rem, 1), 1.0 / max(bhi_k - blo_k, 1e-12), dtype=np.float64)
            else:
                bhi_k = b_ki_base + 0.5 * eps_vec_np
                inv_wk = 1.0 / np.maximum(eps_vec_np, 1e-12)
            u_ki_m = (bhi_k - w_opt) * inv_wk
            g_k_m = np.clip(u_ki_m, 0.0, 1.0)
            one_m_g = 1.0 - g_k_m
            p_noki_d = np.prod(one_m_g, axis=0)
            i_ki_t = 1.0 - p_noki_d
            if compute_grads:
                dg_k_m = np.where((u_ki_m > 0.0) & (u_ki_m < 1.0), -inv_wk, 0.0)
                pref = np.ones((m_rem, n_tot), dtype=np.float64)
                suff = np.ones((m_rem, n_tot), dtype=np.float64)
                for m_k in range(1, m_rem):
                    pref[m_k] = pref[m_k - 1] * one_m_g[m_k - 1]
                for m_k in range(m_rem - 2, -1, -1):
                    suff[m_k] = suff[m_k + 1] * one_m_g[m_k + 1]
                d_iki_dwo_arr = -pref * suff * dg_k_m
        else:
            if abs(ki_spread) > 1e-6:
                blo_k = min(b_ki_base, b_ki_base + float(ki_spread))
                bhi_k = max(b_ki_base, b_ki_base + float(ki_spread))
                inv_wk_f = 1.0 / max(bhi_k - blo_k, 1e-12)
            else:
                eps_term = float(eps_vec_np[-1, 0])
                bhi_k = b_ki_base + 0.5 * eps_term
                inv_wk_f = 1.0 / max(eps_term, 1e-12)
            u_ki = (bhi_k - w_opt[m_last]) * inv_wk_f
            i_ki_t = np.clip(u_ki, 0.0, 1.0)
            if compute_grads:
                d_iki_dwo_last = np.where((u_ki > 0.0) & (u_ki < 1.0), -inv_wk_f, 0.0)
    else:
        i_ki_t = np.full(n_tot, float(i_ki_past), dtype=np.float64)

    has_os = one_star_strike is not None and float(one_star_strike) > 0.0
    if has_os and one_star_strike is not None:
        w_best_last, idx_best_last = (
            (w_opt[m_last], idx_opt[m_last] if idx_opt is not None else None)
            if ((not is_long_basket) and eff_opt_code == 1)
            else (
                np.max(perf_m_d_n[m_last], axis=0),
                np.argmax(perf_m_d_n[m_last], axis=0),
            )
        )
        k_os = float(one_star_strike)
        if one_star_eps <= 1e-6:
            star_ind = (w_best_last >= k_os).astype(np.float64)
            d_star = np.zeros(n_tot, dtype=np.float64)
        else:
            half_os = 0.5 * one_star_eps
            u_os = (w_best_last - (k_os - half_os)) / one_star_eps
            star_ind = np.clip(u_os, 0.0, 1.0)
            d_star = np.where((u_os > 0.0) & (u_os < 1.0), 1.0 / one_star_eps, 0.0)
    else:
        idx_best_last = None
        star_ind = np.zeros(n_tot, dtype=np.float64)
        d_star = np.zeros(n_tot, dtype=np.float64)

    opt_pay = np.zeros(n_tot, dtype=np.float64)
    d_opt_d_wo = np.zeros(n_tot, dtype=np.float64)
    d_opt_d_avg = np.zeros(n_tot, dtype=np.float64)
    d_opt_d_wb = np.zeros(n_tot, dtype=np.float64)
    d_opt_d_iki = np.zeros(n_tot, dtype=np.float64)
    ki_gate = i_ki_t if has_ki else 1.0
    if is_long_basket:
        assert w_avg_opt is not None
        opt_pay = opt_pay - (1.0 - w_avg_opt[m_last]) * ki_gate * (1.0 - star_ind)
        put_active = False
        put_intr = np.zeros(n_tot, dtype=np.float64)
        if compute_grads:
            d_opt_d_avg = (ki_gate if has_ki else 1.0) * (1.0 - star_ind)
            if has_ki:
                d_opt_d_iki = -(1.0 - w_avg_opt[m_last]) * (1.0 - star_ind)
            d_opt_d_wb = (1.0 - w_avg_opt[m_last]) * ki_gate * d_star
    else:
        if lower_strike is not None:
            k_low = float(lower_strike)
            put_active = float(strike) > k_low and float(option_gearing) > 0.0
            put_intr = np.maximum(float(strike) - w_opt[m_last], 0.0) - np.maximum(k_low - w_opt[m_last], 0.0)
            if compute_grads and put_active:
                d_put = np.where(w_opt[m_last] < float(strike), -1.0, 0.0) - np.where(w_opt[m_last] < k_low, -1.0, 0.0)
            else:
                d_put = np.zeros(n_tot, dtype=np.float64)
        else:
            put_active = float(option_gearing) > 0.0
            put_intr = np.maximum(float(strike) - w_opt[m_last], 0.0)
            if compute_grads and put_active:
                d_put = np.where(w_opt[m_last] < float(strike), -1.0, 0.0)
            else:
                d_put = np.zeros(n_tot, dtype=np.float64)
        if put_active:
            opt_pay = opt_pay - float(option_gearing) * ki_gate * (1.0 - star_ind) * put_intr
            if compute_grads:
                if has_ki:
                    d_opt_d_iki = -float(option_gearing) * (1.0 - star_ind) * put_intr
                d_opt_d_wo = d_opt_d_wo - float(option_gearing) * ki_gate * (1.0 - star_ind) * d_put
                d_opt_d_wb = float(option_gearing) * ki_gate * put_intr * d_star

    has_pc = part_call_strike is not None and float(part_call_strike) >= 0.0 and float(part_call_rate) != 0.0
    call_intr = np.maximum(w_opt[m_last] - float(part_call_strike), 0.0) if (has_pc and part_call_strike is not None) else np.zeros(n_tot, dtype=np.float64)
    if has_pc and part_call_strike is not None:
        opt_pay = opt_pay + float(part_call_rate) * call_intr
        if compute_grads:
            d_opt_d_wo = d_opt_d_wo + np.where(w_opt[m_last] > float(part_call_strike), float(part_call_rate), 0.0)
    if has_os and float(one_star_coupon) != 0.0:
        opt_pay = opt_pay + star_ind * float(one_star_coupon)
        if compute_grads:
            d_opt_d_wb = d_opt_d_wb + d_star * float(one_star_coupon)

    bar_lx_post_np = np.zeros((m_rem, d_assets, n_tot), dtype=np.float64)
    bar_ln_fwd_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    use_nb_pa = bool(_HAS_NUMBA and call_code in (0, 1, 2) and cpn_code in (0, 1, 2) and eff_opt_code in (0, 1, 2))
    wf_nb_pa = np.ascontiguousarray(w_fund, dtype=np.float64) if w_fund is not None else np.zeros(m_rem, dtype=np.float64)
    if compute_grads and use_nb_pa:
        dummy_i2d = np.empty((1, 1), dtype=np.int32)
        dummy_i1d = np.empty(1, dtype=np.int32)
        dummy_f2d = np.empty((1, 1), dtype=np.float64)
        _nb_payoff_aad_bwd_and_s1(
            np.ascontiguousarray(perf_m_d_n, dtype=np.float64),
            np.ascontiguousarray(g_ko_all, dtype=np.float64),
            np.ascontiguousarray(dg_ko_all, dtype=np.float64),
            np.ascontiguousarray(eff_g_cpn_all, dtype=np.float64),
            np.ascontiguousarray(dgc_from_cpn, dtype=np.float64),
            np.ascontiguousarray(dgc_from_ko, dtype=np.float64),
            np.ascontiguousarray(surv_pre_arr, dtype=np.float64),
            np.ascontiguousarray(eff_c_arr, dtype=np.float64),
            np.ascontiguousarray(surv_end, dtype=np.float64),
            np.ascontiguousarray(opt_pay, dtype=np.float64),
            np.ascontiguousarray(d_opt_d_wo, dtype=np.float64),
            np.ascontiguousarray(d_opt_d_avg, dtype=np.float64),
            np.ascontiguousarray(d_opt_d_wb, dtype=np.float64),
            np.ascontiguousarray(d_opt_d_iki, dtype=np.float64),
            np.ascontiguousarray(d_iki_dwo_last, dtype=np.float64),
            np.ascontiguousarray(d_iki_dwo_arr, dtype=np.float64) if d_iki_dwo_arr is not None else dummy_f2d,
            d_iki_dwo_arr is not None,
            np.ascontiguousarray(df_pay, dtype=np.float64),
            wf_nb_pa,
            w_fund is not None,
            np.ascontiguousarray(idx_call, dtype=np.int32) if idx_call is not None else dummy_i2d,
            idx_call is not None,
            np.ascontiguousarray(idx_cpn, dtype=np.int32) if idx_cpn is not None else dummy_i2d,
            idx_cpn is not None,
            np.ascontiguousarray(idx_opt, dtype=np.int32) if idx_opt is not None else dummy_i2d,
            idx_opt is not None,
            np.ascontiguousarray(idx_best_last, dtype=np.int32) if idx_best_last is not None else dummy_i1d,
            idx_best_last is not None,
            int(call_code),
            int(cpn_code),
            int(eff_opt_code),
            bool(is_long_basket),
            bool(is_memory),
            bool(no_notional),
            bool(has_os),
            float(ac_cpn),
            bar_lx_post_np,
            bar_ln_fwd_np,
        )
    elif compute_grads:
        scale = 1.0 / float(n_tot)
        df_last = float(df_pay[m_last])
        term_mult = opt_pay if no_notional else (1.0 + opt_pay)
        bar_surv = scale * df_last * term_mult
        bar_c_mem = np.zeros(n_tot, dtype=np.float64)
        bar_last_wo = scale * df_last * surv_end * (d_opt_d_wo + (d_opt_d_iki * d_iki_dwo_last if d_iki_dwo_arr is None else 0.0))
        bar_last_avg = scale * df_last * surv_end * d_opt_d_avg
        bar_last_wb = scale * df_last * surv_end * d_opt_d_wb
        bar_iki_disc = (scale * df_last * surv_end * d_opt_d_iki) if d_iki_dwo_arr is not None else None
        inv_d = 1.0 / float(d_assets)
        ac_c = float(ac_cpn)

        for m in range(m_rem - 1, -1, -1):
            surv_pre = surv_pre_arr[m]
            g_k = g_ko_all[m]
            eff_g_c = eff_g_cpn_all[m]
            eff_c = eff_c_arr[m]
            df_m = float(df_pay[m])
            wf_m = float(w_fund[m]) if w_fund is not None else 0.0

            bar_pm = -bar_surv
            if not no_notional:
                bar_pm = bar_pm + scale * df_m
            bar_pm = bar_pm + scale * df_m * ac_c
            bar_eff_g_c = scale * df_m * surv_pre * eff_c
            if is_memory:
                bar_eff_g_c = bar_eff_g_c - bar_c_mem * surv_pre * eff_c

            bar_surv_next = bar_surv + bar_pm * g_k + scale * df_m * eff_g_c * eff_c - scale * wf_m
            if is_memory:
                bar_surv_next = bar_surv_next + bar_c_mem * eff_c * (1.0 - eff_g_c)
            bar_surv = bar_surv_next
            if is_memory:
                bar_c_mem = scale * df_m * surv_pre * eff_g_c + bar_c_mem * surv_pre * (1.0 - eff_g_c)

            bar_g_k = bar_pm * surv_pre
            bar_w_call = bar_g_k * dg_ko_all[m] + bar_eff_g_c * dgc_from_ko[m]
            bar_w_cpn = bar_eff_g_c * dgc_from_cpn[m]

            bar_w_opt = np.zeros(n_tot, dtype=np.float64)
            bar_w_avg = np.zeros(n_tot, dtype=np.float64)
            bar_w_wb = np.zeros(n_tot, dtype=np.float64)
            if bar_iki_disc is not None and d_iki_dwo_arr is not None:
                bar_w_opt = bar_iki_disc * d_iki_dwo_arr[m]
            if m == m_last:
                bar_w_opt = bar_w_opt + bar_last_wo
                if is_long_basket:
                    bar_w_avg = bar_last_avg
                if has_os:
                    bar_w_wb = bar_last_wb

            # Merge index sensitivities with identical codes before scattering across assets.
            bp_m = np.zeros((d_assets, n_tot), dtype=np.float64)
            eff_wc_tot = bar_w_call + (bar_w_cpn if cpn_code == call_code else 0.0)
            if call_code == 2:
                bp_m += eff_wc_tot * inv_d
            elif idx_call is not None:
                np.add.at(bp_m, (idx_call[m], np.arange(n_tot)), eff_wc_tot)
            if cpn_code != call_code:
                if cpn_code == 2:
                    bp_m += bar_w_cpn * inv_d
                elif idx_cpn is not None:
                    np.add.at(bp_m, (idx_cpn[m], np.arange(n_tot)), bar_w_cpn)
            if is_long_basket:
                if idx_opt is not None:
                    np.add.at(bp_m, (idx_opt[m], np.arange(n_tot)), bar_w_opt)
                bp_m += bar_w_avg * inv_d
            else:
                if eff_opt_code == 2:
                    bp_m += bar_w_opt * inv_d
                elif idx_opt is not None:
                    np.add.at(bp_m, (idx_opt[m], np.arange(n_tot)), bar_w_opt)
            if has_os and m == m_last and idx_best_last is not None:
                np.add.at(bp_m, (idx_best_last, np.arange(n_tot)), bar_w_wb)

            lx_m = bp_m * perf_m_d_n[m]
            bar_lx_post_np[m] = lx_m
            bar_ln_fwd_np[:, m] = lx_m.sum(axis=1)

    stage1_rep_out: Optional[npt.NDArray[np.float64]] = None
    cf_wf_out: Optional[npt.NDArray[np.float64]] = None
    if emit_stage1 and n_rep > 0:
        n_r_blk = max(n_tot // n_rep, 1)
        s1_mat = np.zeros((n_r_blk, 3 * m_rem + 6), dtype=np.float64)
        cf_wf_out = np.zeros((2, m_rem), dtype=np.float64) if want_cf_wf else None
        gk_s1_all = g_ko_all if not hard_ko else np.where(call_mask_m, (w_call >= ko_barriers[:, None]).astype(np.float64), 0.0)
        if not hard_cpn:
            egc_s1_all = eff_g_cpn_all
        else:
            h_cpn = (w_cpn <= cpn_barriers[:, None]) if cpn_dir_down else (w_cpn >= cpn_barriers[:, None])
            ind_h = (1.0 - h_cpn.astype(np.float64)) if cpn_type_out else h_cpn.astype(np.float64)
            gc_s1 = np.where(guar_mask_m, 1.0, ind_h)
            egc_s1_all = np.maximum(gc_s1, gk_s1_all) if force_cpn_ac else gc_s1
        if has_ki and ki_barrier > 0.0 and i_ki_past < 0.5:
            if ki_mode_norm == "discrete" or abs(ki_spread) > 1e-6:
                ki_s1 = i_ki_t
            else:
                b_ki_s1 = float(strike) if is_long_basket else float(ki_barrier)
                ki_s1 = (w_opt[m_last] < b_ki_s1).astype(np.float64)
        else:
            ki_s1 = np.ones(n_tot, dtype=np.float64) if not has_ki else np.full(n_tot, float(i_ki_past), dtype=np.float64)

        if use_nb_pa:
            cf_nb_pa = cf_wf_out if cf_wf_out is not None else np.zeros((2, m_rem), dtype=np.float64)
            w_avg_l_nb = np.ascontiguousarray(w_avg_opt[m_last], dtype=np.float64) if (is_long_basket and w_avg_opt is not None) else np.empty(1, dtype=np.float64)
            _nb_payoff_aad_s1_loop(
                np.ascontiguousarray(gk_s1_all, dtype=np.float64),
                np.ascontiguousarray(egc_s1_all, dtype=np.float64),
                np.ascontiguousarray(cpn_rates, dtype=np.float64),
                np.ascontiguousarray(df_pay, dtype=np.float64),
                wf_nb_pa,
                w_fund is not None,
                np.ascontiguousarray(ki_s1, dtype=np.float64),
                np.ascontiguousarray(star_ind, dtype=np.float64),
                np.ascontiguousarray(put_intr, dtype=np.float64),
                np.ascontiguousarray(call_intr, dtype=np.float64),
                w_avg_l_nb,
                s1_mat,
                cf_nb_pa,
                bool(want_cf_wf),
                int(n_rep),
                float(c_mem_0),
                float(ac_cpn),
                float(option_gearing),
                float(part_call_rate),
                float(one_star_coupon),
                bool(is_memory),
                bool(no_notional),
                bool(is_long_basket),
                bool(put_active),
                bool(has_pc and part_call_strike is not None),
                bool(has_os),
            )
            stage1_rep_out = s1_mat
        else:
            surv_s1 = np.ones(n_tot, dtype=np.float64)
            cmem_s1 = np.full(n_tot, float(c_mem_0), dtype=np.float64)
            pv_s1 = np.zeros(n_tot, dtype=np.float64)
            inv_n_tot = 1.0 / float(n_tot)
            for m in range(m_rem):
                pm_s1 = surv_s1 * gk_s1_all[m]
                qc_s1 = surv_s1 * egc_s1_all[m]
                if is_memory:
                    ec_s1 = cpn_rates[m] + cmem_s1
                    ccf_s1 = qc_s1 * ec_s1 + pm_s1 * float(ac_cpn)
                    cmem_s1 = surv_s1 * ec_s1 * (1.0 - egc_s1_all[m])
                else:
                    ccf_s1 = qc_s1 * cpn_rates[m] + pm_s1 * float(ac_cpn)
                wf_m_s1 = float(w_fund[m]) if w_fund is not None else 0.0
                cf_m_s1 = ccf_s1 if no_notional else (pm_s1 + ccf_s1)
                pv_s1 = pv_s1 + float(df_pay[m]) * cf_m_s1 - wf_m_s1 * surv_s1
                if cf_wf_out is not None:
                    cf_wf_out[0, m] = float(cf_m_s1.sum() * inv_n_tot)
                    cf_wf_out[1, m] = float(surv_s1.sum() * inv_n_tot)
                s1_mat[:, m] = pm_s1.reshape(n_r_blk, n_rep).mean(axis=1)
                s1_mat[:, m_rem + m] = qc_s1.reshape(n_r_blk, n_rep).mean(axis=1)
                s1_mat[:, 2 * m_rem + m] = ccf_s1.reshape(n_r_blk, n_rep).mean(axis=1)
                surv_s1 = np.maximum(surv_s1 - pm_s1, 0.0)

            if is_long_basket:
                assert w_avg_opt is not None
                u_put_s1 = surv_s1 * ki_s1 * (1.0 - star_ind) * (1.0 - w_avg_opt[m_last])
                put_term_s1 = -u_put_s1
            else:
                u_put_s1 = (surv_s1 * ki_s1 * (1.0 - star_ind) * put_intr) if put_active else np.zeros(n_tot, dtype=np.float64)
                put_term_s1 = -float(option_gearing) * u_put_s1
            u_call_s1 = (surv_s1 * call_intr) if (has_pc and part_call_strike is not None) else np.zeros(n_tot, dtype=np.float64)
            p_os_s1 = (surv_s1 * star_ind) if has_os else np.zeros(n_tot, dtype=np.float64)
            l_os_s1 = (p_os_s1 * float(one_star_coupon)) if (has_os and float(one_star_coupon) != 0.0) else np.zeros(n_tot, dtype=np.float64)
            opt_s1 = put_term_s1 + (float(part_call_rate) * u_call_s1 if has_pc else 0.0) + l_os_s1
            term_s1 = opt_s1 if no_notional else (surv_s1 + opt_s1)
            pv_s1 = pv_s1 + float(df_pay[m_last]) * term_s1
            c0 = 3 * m_rem
            s1_mat[:, c0 + 0] = u_put_s1.reshape(n_r_blk, n_rep).mean(axis=1)
            s1_mat[:, c0 + 1] = u_call_s1.reshape(n_r_blk, n_rep).mean(axis=1)
            s1_mat[:, c0 + 2] = l_os_s1.reshape(n_r_blk, n_rep).mean(axis=1)
            s1_mat[:, c0 + 3] = p_os_s1.reshape(n_r_blk, n_rep).mean(axis=1)
            s1_mat[:, c0 + 4] = pv_s1.reshape(n_r_blk, n_rep).mean(axis=1)
            s1_mat[:, c0 + 5] = (pv_s1 * pv_s1).reshape(n_r_blk, n_rep).mean(axis=1)
            stage1_rep_out = s1_mat
            if cf_wf_out is not None:
                cf_wf_out[0, m_last] += float(term_s1.sum() * inv_n_tot)

    return bar_lx_post_np, bar_ln_fwd_np, stage1_rep_out, cf_wf_out


if _HAS_NUMBA:
    @numba.njit(inline="always")  # type: ignore[untyped-decorator]
    def _nb_c2_s5(x: float, lo_v: float, inv_w: float) -> float:
        u = (x - lo_v) * inv_w
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        return ((6.0 * u - 15.0) * u + 10.0) * u * u * u

    @numba.njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
    def _nb_c2_precompute_stats(
        p_pre: npt.NDArray[np.float64],
        pm_sp: npt.NDArray[np.float64],
        need_min: bool,
        need_max: bool,
        need_sum: bool,
    ) -> Tuple[
        npt.NDArray[np.int32],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
        npt.NDArray[np.float64],
        npt.NDArray[np.int32],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        m_pre, d_assets, n_tot = p_pre.shape
        mn_shape = (m_pre, n_tot) if need_min else (1, 1)
        mx_shape = (m_pre, n_tot) if need_max else (1, 1)
        sm_shape = (m_pre, n_tot) if need_sum else (1, 1)
        mk1 = np.empty(mn_shape, dtype=np.int32)
        mv1 = np.empty(mn_shape, dtype=np.float64)
        mk2 = np.empty(mn_shape, dtype=np.int32)
        mv2 = np.empty(mn_shape, dtype=np.float64)
        mk3 = np.empty(mn_shape, dtype=np.int32)
        mv3 = np.empty(mn_shape, dtype=np.float64)
        min_p = np.empty(mn_shape, dtype=np.float64)
        min_m = np.empty(mn_shape, dtype=np.float64)
        xk1 = np.empty(mx_shape, dtype=np.int32)
        xv1 = np.empty(mx_shape, dtype=np.float64)
        xk2 = np.empty(mx_shape, dtype=np.int32)
        xv2 = np.empty(mx_shape, dtype=np.float64)
        xk3 = np.empty(mx_shape, dtype=np.int32)
        xv3 = np.empty(mx_shape, dtype=np.float64)
        max_p = np.empty(mx_shape, dtype=np.float64)
        max_m = np.empty(mx_shape, dtype=np.float64)
        sum_b = np.empty(sm_shape, dtype=np.float64)
        sum_sp_p = np.empty(sm_shape, dtype=np.float64)
        sum_sp_m = np.empty(sm_shape, dtype=np.float64)
        inv_d = 1.0 / float(d_assets)

        for mn in numba.prange(m_pre * n_tot):
            m = mn // n_tot
            n = mn - m * n_tot
            if need_min:
                k1, k2, k3 = 0, 0, 0
                v1, v2, v3 = 1e300, 1e300, 1e300
                mp = 1e300
                mm = 1e300
                for d in range(d_assets):
                    v = p_pre[m, d, n]
                    if v < v1:
                        k3, v3 = k2, v2
                        k2, v2 = k1, v1
                        k1, v1 = d, v
                    elif v < v2:
                        k3, v3 = k2, v2
                        k2, v2 = d, v
                    elif v < v3:
                        k3, v3 = d, v
                    vp = pm_sp[d, m, n]
                    vm = pm_sp[d_assets + d, m, n]
                    if vp < mp:
                        mp = vp
                    if vm < mm:
                        mm = vm
                # Identity sentinel (1e300) ensures d_assets==1/2 leave-one/two-out excludes all bumped assets.
                mk1[m, n], mv1[m, n] = k1, v1
                mk2[m, n], mv2[m, n] = k2, v2
                mk3[m, n], mv3[m, n] = k3, v3
                min_p[m, n], min_m[m, n] = mp, mm
            if need_max:
                k1, k2, k3 = 0, 0, 0
                v1, v2, v3 = -1e300, -1e300, -1e300
                xp = -1e300
                xm = -1e300
                for d in range(d_assets):
                    v = p_pre[m, d, n]
                    if v > v1:
                        k3, v3 = k2, v2
                        k2, v2 = k1, v1
                        k1, v1 = d, v
                    elif v > v2:
                        k3, v3 = k2, v2
                        k2, v2 = d, v
                    elif v > v3:
                        k3, v3 = d, v
                    vp = pm_sp[d, m, n]
                    vm = pm_sp[d_assets + d, m, n]
                    if vp > xp:
                        xp = vp
                    if vm > xm:
                        xm = vm
                # Identity sentinel (-1e300) ensures d_assets==1/2 leave-one/two-out excludes all bumped assets.
                xk1[m, n], xv1[m, n] = k1, v1
                xk2[m, n], xv2[m, n] = k2, v2
                xk3[m, n], xv3[m, n] = k3, v3
                max_p[m, n], max_m[m, n] = xp, xm
            if need_sum:
                sb, spp, spm = 0.0, 0.0, 0.0
                for d in range(d_assets):
                    sb += p_pre[m, d, n]
                    spp += pm_sp[d, m, n]
                    spm += pm_sp[d_assets + d, m, n]
                sum_b[m, n] = sb
                sum_sp_p[m, n] = spp * inv_d
                sum_sp_m[m, n] = spm * inv_d

        return (
            mk1, mv1, mk2, mv2, mk3, mv3, min_p, min_m,
            xk1, xv1, xk2, xv2, xk3, xv3, max_p, max_m,
            sum_b, sum_sp_p, sum_sp_m,
        )

    @numba.njit(inline="always")  # type: ignore[untyped-decorator]
    def _nb_fill_idx_slice(
        code_idx: int,
        aa_i: int,
        ab_i: int,
        asl_i: int,
        bsl_i: int,
        m: int,
        n_start: int,
        n_len: int,
        inv_d: float,
        p_pre: npt.NDArray[np.float64],
        pm_sp: npt.NDArray[np.float64],
        mk1: npt.NDArray[np.int32],
        mv1: npt.NDArray[np.float64],
        mk2: npt.NDArray[np.int32],
        mv2: npt.NDArray[np.float64],
        mv3: npt.NDArray[np.float64],
        min_p: npt.NDArray[np.float64],
        min_m: npt.NDArray[np.float64],
        xk1: npt.NDArray[np.int32],
        xv1: npt.NDArray[np.float64],
        xk2: npt.NDArray[np.int32],
        xv2: npt.NDArray[np.float64],
        xv3: npt.NDArray[np.float64],
        max_p: npt.NDArray[np.float64],
        max_m: npt.NDArray[np.float64],
        sum_b: npt.NDArray[np.float64],
        sum_sp_p: npt.NDArray[np.float64],
        sum_sp_m: npt.NDArray[np.float64],
        out_w: npt.NDArray[np.float64],
    ) -> None:
        n_end = n_start + n_len
        if code_idx == 2:
            if aa_i == -2:
                src_s = sum_sp_p[m, n_start:n_end] if asl_i == 0 else sum_sp_m[m, n_start:n_end]
                for k in range(n_len):
                    out_w[k] = src_s[k]
            elif aa_i == -1:
                sb_m = sum_b[m, n_start:n_end]
                for k in range(n_len):
                    out_w[k] = sb_m[k] * inv_d
            elif ab_i < 0:
                sb_m = sum_b[m, n_start:n_end]
                pa_m = p_pre[m, aa_i, n_start:n_end]
                sp_a = pm_sp[asl_i, m, n_start:n_end]
                for k in range(n_len):
                    out_w[k] = (sb_m[k] - pa_m[k] + sp_a[k]) * inv_d
            else:
                sb_m = sum_b[m, n_start:n_end]
                pa_m = p_pre[m, aa_i, n_start:n_end]
                pb_m = p_pre[m, ab_i, n_start:n_end]
                sp_a = pm_sp[asl_i, m, n_start:n_end]
                sp_b = pm_sp[bsl_i, m, n_start:n_end]
                for k in range(n_len):
                    out_w[k] = (sb_m[k] - pa_m[k] - pb_m[k] + sp_a[k] + sp_b[k]) * inv_d
        elif code_idx == 0:
            if aa_i == -2:
                src_s = min_p[m, n_start:n_end] if asl_i == 0 else min_m[m, n_start:n_end]
                for k in range(n_len):
                    out_w[k] = src_s[k]
            elif aa_i == -1:
                v1_m = mv1[m, n_start:n_end]
                for k in range(n_len):
                    out_w[k] = v1_m[k]
            elif ab_i < 0:
                k1_m = mk1[m, n_start:n_end]
                v1_m = mv1[m, n_start:n_end]
                v2_m = mv2[m, n_start:n_end]
                sp_a = pm_sp[asl_i, m, n_start:n_end]
                for k in range(n_len):
                    bv = v2_m[k] if k1_m[k] == aa_i else v1_m[k]
                    sv = sp_a[k]
                    out_w[k] = bv if bv < sv else sv
            else:
                k1_m = mk1[m, n_start:n_end]
                k2_m = mk2[m, n_start:n_end]
                v1_m = mv1[m, n_start:n_end]
                v2_m = mv2[m, n_start:n_end]
                v3_m = mv3[m, n_start:n_end]
                sp_a = pm_sp[asl_i, m, n_start:n_end]
                sp_b = pm_sp[bsl_i, m, n_start:n_end]
                for k in range(n_len):
                    k1n = k1_m[k]
                    k2n = k2_m[k]
                    m1 = (k1n == aa_i) or (k1n == ab_i)
                    m2 = (k2n == aa_i) or (k2n == ab_i)
                    bv = v3_m[k] if (m1 and m2) else (v2_m[k] if m1 else v1_m[k])
                    s1v = sp_a[k]
                    s2v = sp_b[k]
                    sv = s1v if s1v < s2v else s2v
                    out_w[k] = bv if bv < sv else sv
        else:
            if aa_i == -2:
                src_s = max_p[m, n_start:n_end] if asl_i == 0 else max_m[m, n_start:n_end]
                for k in range(n_len):
                    out_w[k] = src_s[k]
            elif aa_i == -1:
                v1_m = xv1[m, n_start:n_end]
                for k in range(n_len):
                    out_w[k] = v1_m[k]
            elif ab_i < 0:
                k1_m = xk1[m, n_start:n_end]
                v1_m = xv1[m, n_start:n_end]
                v2_m = xv2[m, n_start:n_end]
                sp_a = pm_sp[asl_i, m, n_start:n_end]
                for k in range(n_len):
                    bv = v2_m[k] if k1_m[k] == aa_i else v1_m[k]
                    sv = sp_a[k]
                    out_w[k] = bv if bv > sv else sv
            else:
                k1_m = xk1[m, n_start:n_end]
                k2_m = xk2[m, n_start:n_end]
                v1_m = xv1[m, n_start:n_end]
                v2_m = xv2[m, n_start:n_end]
                v3_m = xv3[m, n_start:n_end]
                sp_a = pm_sp[asl_i, m, n_start:n_end]
                sp_b = pm_sp[bsl_i, m, n_start:n_end]
                for k in range(n_len):
                    k1n = k1_m[k]
                    k2n = k2_m[k]
                    m1 = (k1n == aa_i) or (k1n == ab_i)
                    m2 = (k2n == aa_i) or (k2n == ab_i)
                    bv = v3_m[k] if (m1 and m2) else (v2_m[k] if m1 else v1_m[k])
                    s1v = sp_a[k]
                    s2v = sp_b[k]
                    sv = s1v if s1v > s2v else s2v
                    out_w[k] = bv if bv > sv else sv

    @numba.njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
    def _nb_c2_preterm_eval(
        p_pre: npt.NDArray[np.float64],
        pm_sp: npt.NDArray[np.float64],
        frozen_term: npt.NDArray[np.float64],
        aa_pre: npt.NDArray[np.int32],
        ab_pre: npt.NDArray[np.int32],
        asl_pre: npt.NDArray[np.int32],
        bsl_pre: npt.NDArray[np.int32],
        ko_barriers: npt.NDArray[np.float64],
        is_callable: npt.NDArray[np.float64],
        cpn_barriers: npt.NDArray[np.float64],
        is_guar: npt.NDArray[np.float64],
        cpn_rates: npt.NDArray[np.float64],
        df_pay: npt.NDArray[np.float64],
        w_fund_arr: npt.NDArray[np.float64],
        has_w_fund: bool,
        eps_vec: npt.NDArray[np.float64],
        mk1: npt.NDArray[np.int32],
        mv1: npt.NDArray[np.float64],
        mk2: npt.NDArray[np.int32],
        mv2: npt.NDArray[np.float64],
        mk3: npt.NDArray[np.int32],
        mv3: npt.NDArray[np.float64],
        min_p: npt.NDArray[np.float64],
        min_m: npt.NDArray[np.float64],
        xk1: npt.NDArray[np.int32],
        xv1: npt.NDArray[np.float64],
        xk2: npt.NDArray[np.int32],
        xv2: npt.NDArray[np.float64],
        xk3: npt.NDArray[np.int32],
        xv3: npt.NDArray[np.float64],
        max_p: npt.NDArray[np.float64],
        max_m: npt.NDArray[np.float64],
        sum_b: npt.NDArray[np.float64],
        sum_sp_p: npt.NDArray[np.float64],
        sum_sp_m: npt.NDArray[np.float64],
        call_code: int,
        cpn_code: int,
        hard_ko: bool,
        hard_cpn: bool,
        ko_spread: float,
        eff_csp: float,
        cpn_dir_down: bool,
        cpn_type_out: bool,
        is_memory: bool,
        force_cpn_ac: bool,
        no_notional: bool,
        ac_cpn: float,
        c_mem_0: float,
    ) -> npt.NDArray[np.float64]:
        m_pre, d_assets, n_tot = p_pre.shape
        n_rows = len(aa_pre)
        inv_d = 1.0 / float(d_assets)
        m_last = m_pre
        c_last = float(cpn_rates[m_last])
        df_last = float(df_pay[m_last])
        wf_last = float(w_fund_arr[m_last]) if has_w_fund else 0.0
        gk_f = frozen_term[0]
        egc_f = frozen_term[1]
        opt_f = frozen_term[2]
        out_pv = np.empty(n_rows, dtype=np.float64)
        flip_cpn = cpn_dir_down != cpn_type_out
        tile_n = 512
        n_tiles = max(1, (n_tot + tile_n - 1) // tile_n)
        tile_pv = np.zeros((n_rows, n_tiles), dtype=np.float64)

        for task_i in numba.prange(n_rows * n_tiles):
            row_i = task_i // n_tiles
            tile_i = task_i - row_i * n_tiles
            n_start = tile_i * tile_n
            n_end = min(n_start + tile_n, n_tot)
            n_len = n_end - n_start
            aa_i = int(aa_pre[row_i])
            ab_i = int(ab_pre[row_i])
            asl_i = int(asl_pre[row_i])
            bsl_i = int(bsl_pre[row_i])
            surv_arr = np.ones(n_len, dtype=np.float64)
            cmem_arr = np.full(n_len, c_mem_0, dtype=np.float64)
            pv_arr = np.zeros(n_len, dtype=np.float64)
            wc_buf = np.empty(n_len, dtype=np.float64)
            wcp_buf = np.empty(n_len, dtype=np.float64)

            for m in range(m_pre):
                eps_m = float(eps_vec[m])
                callable_m = is_callable[m] > 0.5
                guar_m = is_guar[m] > 0.5
                bk = float(ko_barriers[m])
                lo_k = (bk - 0.5 * eps_m) if hard_ko else min(bk, bk + ko_spread)
                hi_k = (bk + 0.5 * eps_m) if hard_ko else max(bk, bk + ko_spread)
                inv_wk = 1.0 / max(hi_k - lo_k, 1e-12)
                bc = float(cpn_barriers[m])
                lo_c = (bc - 0.5 * eps_m) if hard_cpn else min(bc, bc + eff_csp)
                hi_c = (bc + 0.5 * eps_m) if hard_cpn else max(bc, bc + eff_csp)
                inv_wc = 1.0 / max(hi_c - lo_c, 1e-12)
                c_rate = float(cpn_rates[m])
                df_m = float(df_pay[m])
                wf_m = float(w_fund_arr[m]) if has_w_fund else 0.0

                if callable_m:
                    _nb_fill_idx_slice(
                        call_code, aa_i, ab_i, asl_i, bsl_i, m, n_start, n_len, inv_d,
                        p_pre, pm_sp, mk1, mv1, mk2, mv2, mv3, min_p, min_m,
                        xk1, xv1, xk2, xv2, xv3, max_p, max_m, sum_b, sum_sp_p, sum_sp_m, wc_buf,
                    )
                if (not guar_m) and not (callable_m and cpn_code == call_code):
                    _nb_fill_idx_slice(
                        cpn_code, aa_i, ab_i, asl_i, bsl_i, m, n_start, n_len, inv_d,
                        p_pre, pm_sp, mk1, mv1, mk2, mv2, mv3, min_p, min_m,
                        xk1, xv1, xk2, xv2, xv3, max_p, max_m, sum_b, sum_sp_p, sum_sp_m, wcp_buf,
                    )
                    wcp_src = wcp_buf
                else:
                    wcp_src = wc_buf

                for k in range(n_len):
                    gk = _nb_c2_s5(wc_buf[k], lo_k, inv_wk) if callable_m else 0.0
                    if guar_m:
                        gc = 1.0
                    else:
                        gc = _nb_c2_s5(wcp_src[k], lo_c, inv_wc)
                        if flip_cpn:
                            gc = 1.0 - gc
                        if callable_m and force_cpn_ac and gk > gc:
                            gc = gk
                    surv = surv_arr[k]
                    pm = surv * gk if callable_m else 0.0
                    if is_memory:
                        u_m = cmem_arr[k] + c_rate
                        tmp = surv * gc * u_m + (pm * ac_cpn if callable_m else 0.0)
                        cmem_arr[k] = (1.0 - gc) * surv * u_m
                    else:
                        tmp = surv * gc * c_rate + (pm * ac_cpn if callable_m else 0.0)
                    if callable_m and (not no_notional):
                        tmp += pm
                    pv_cur = pv_arr[k] + df_m * tmp
                    if has_w_fund:
                        pv_cur -= wf_m * surv
                    pv_arr[k] = pv_cur
                    if callable_m:
                        rem_s = surv - pm
                        surv_arr[k] = rem_s if rem_s > 0.0 else 0.0

            pv_sum = 0.0
            for k in range(n_len):
                n = n_start + k
                surv = surv_arr[k]
                eff_c_last = (c_last + cmem_arr[k]) if is_memory else c_last
                pm = surv * gk_f[n]
                cpn_last = surv * egc_f[n] * eff_c_last + pm * ac_cpn
                rem_s = surv - pm
                if rem_s < 0.0:
                    rem_s = 0.0
                pv_cur = pv_arr[k] - (wf_last * surv if has_w_fund else 0.0)
                term_pay = (
                    (cpn_last + rem_s * opt_f[n])
                    if no_notional
                    else (pm + cpn_last + rem_s * (1.0 + opt_f[n]))
                )
                pv_sum += pv_cur + df_last * term_pay
            tile_pv[row_i, tile_i] = pv_sum

        inv_n = 1.0 / float(n_tot)
        for row_i in range(n_rows):
            s_r = 0.0
            for tile_i in range(n_tiles):
                s_r += tile_pv[row_i, tile_i]
            out_pv[row_i] = s_r * inv_n

        return out_pv


def _eval_c2_preterm_cpu_np(
    perf_base_m_d_n: npt.NDArray[np.float64],
    perf_sp_p: npt.NDArray[np.float64],
    perf_sp_m: npt.NDArray[np.float64],
    frozen_term: npt.NDArray[np.float64],
    *,
    aa_pre: npt.NDArray[np.int32],
    ab_pre: npt.NDArray[np.int32],
    asl_pre: npt.NDArray[np.int32],
    bsl_pre: npt.NDArray[np.int32],
    ko_barriers: npt.NDArray[np.float64],
    is_callable: npt.NDArray[np.float64],
    cpn_barriers: npt.NDArray[np.float64],
    is_guar: npt.NDArray[np.float64],
    cpn_rates: npt.NDArray[np.float64],
    df_pay: npt.NDArray[np.float64],
    w_fund: Optional[npt.NDArray[np.float64]],
    eps_vec: npt.NDArray[np.float64],
    call_code: int,
    cpn_code: int,
    ko_spread: float,
    cpn_spread: float,
    cpn_dir_down: bool,
    cpn_type_out: bool,
    is_memory: bool,
    force_cpn_ac: bool,
    no_notional: bool,
    ac_cpn: float,
    c_mem_0: float,
    want_gamma: bool,
) -> npt.NDArray[np.float64]:
    """L2-cached in-place C^2 quintic-ramp LOO/L2O PreTerm evaluator for CPU Vibrato Gamma (`M > 1`)."""
    m_rem, d_assets, n_tot = perf_base_m_d_n.shape
    m_pre = m_rem - 1
    n_rows = int(len(aa_pre))
    p_pre = np.ascontiguousarray(perf_base_m_d_n[:m_pre], dtype=np.float64)
    pm_sp = np.ascontiguousarray(
        np.concatenate([perf_sp_p[:, :m_pre, :], perf_sp_m[:, :m_pre, :]], axis=0),
        dtype=np.float64,
    )

    need_min = (call_code == 0) or (cpn_code == 0)
    need_max = (call_code == 1) or (cpn_code == 1)
    need_sum = (call_code == 2) or (cpn_code == 2)
    hard_ko = abs(ko_spread) <= 1e-6
    eff_csp = -cpn_spread if cpn_dir_down else cpn_spread
    hard_cpn = abs(eff_csp) <= 1e-6
    if _HAS_NUMBA and call_code in (0, 1, 2) and cpn_code in (0, 1, 2):
        _ = want_gamma
        mk1, mv1, mk2, mv2, mk3, mv3, min_p, min_m, xk1, xv1, xk2, xv2, xk3, xv3, max_p, max_m, sum_b, sum_sp_p, sum_sp_m = (
            _nb_c2_precompute_stats(p_pre, pm_sp, need_min, need_max, need_sum)
        )
        wf_arr = (
            np.ascontiguousarray(w_fund, dtype=np.float64)
            if w_fund is not None
            else np.zeros(m_rem, dtype=np.float64)
        )
        return cast(
            npt.NDArray[np.float64],
            _nb_c2_preterm_eval(
                p_pre,
                pm_sp,
                np.ascontiguousarray(frozen_term, dtype=np.float64),
                np.ascontiguousarray(aa_pre, dtype=np.int32),
                np.ascontiguousarray(ab_pre, dtype=np.int32),
                np.ascontiguousarray(asl_pre, dtype=np.int32),
                np.ascontiguousarray(bsl_pre, dtype=np.int32),
                np.ascontiguousarray(ko_barriers, dtype=np.float64),
                np.ascontiguousarray(is_callable, dtype=np.float64),
                np.ascontiguousarray(cpn_barriers, dtype=np.float64),
                np.ascontiguousarray(is_guar, dtype=np.float64),
                np.ascontiguousarray(cpn_rates, dtype=np.float64),
                np.ascontiguousarray(df_pay, dtype=np.float64),
                wf_arr,
                w_fund is not None,
                np.ascontiguousarray(eps_vec, dtype=np.float64),
                mk1, mv1, mk2, mv2, mk3, mv3, min_p, min_m,
                xk1, xv1, xk2, xv2, xk3, xv3, max_p, max_m,
                sum_b, sum_sp_p, sum_sp_m,
                int(call_code),
                int(cpn_code),
                bool(hard_ko),
                bool(hard_cpn),
                float(ko_spread),
                float(eff_csp),
                bool(cpn_dir_down),
                bool(cpn_type_out),
                bool(is_memory),
                bool(force_cpn_ac),
                bool(no_notional),
                float(ac_cpn),
                float(c_mem_0),
            ),
        )
    need_sort = need_min or need_max
    order = np.argsort(p_pre, axis=1) if need_sort else np.empty(0, dtype=np.intp)
    sum_base = p_pre.sum(axis=1) if (call_code == 2 or cpn_code == 2) else np.empty(0, dtype=np.float64)
    inv_d = 1.0 / float(d_assets)
    ar_m = np.arange(m_pre)[:, None]
    ar_n = np.arange(n_tot)[None, :]

    min_pack: Optional[Tuple[Any, ...]] = None
    max_pack: Optional[Tuple[Any, ...]] = None
    if need_min:
        mk1 = order[:, 0, :]
        mv1 = p_pre[ar_m, mk1, ar_n]
        mk2 = order[:, 1, :] if d_assets > 1 else mk1
        mv2 = p_pre[ar_m, mk2, ar_n] if d_assets > 1 else np.full_like(mv1, 1e300)
        mk3 = order[:, 2, :] if d_assets > 2 else mk2
        mv3 = p_pre[ar_m, mk3, ar_n] if d_assets > 2 else np.full_like(mv1, 1e300)
        min_pack = (mk1, mv1, mk2, mv2, mk3, mv3, pm_sp[:d_assets].min(axis=0), pm_sp[d_assets:].min(axis=0))
    if need_max:
        xk1 = order[:, d_assets - 1, :]
        xv1 = p_pre[ar_m, xk1, ar_n]
        xk2 = order[:, d_assets - 2, :] if d_assets > 1 else xk1
        xv2 = p_pre[ar_m, xk2, ar_n] if d_assets > 1 else np.full_like(xv1, -1e300)
        xk3 = order[:, d_assets - 3, :] if d_assets > 2 else xk2
        xv3 = p_pre[ar_m, xk3, ar_n] if d_assets > 2 else np.full_like(xv1, -1e300)
        max_pack = (xk1, xv1, xk2, xv2, xk3, xv3, pm_sp[:d_assets].max(axis=0), pm_sp[d_assets:].max(axis=0))
    sum_sp_p = pm_sp[:d_assets].sum(axis=0) * inv_d if (call_code == 2 or cpn_code == 2) else np.empty(0, dtype=np.float64)
    sum_sp_m = pm_sp[d_assets:].sum(axis=0) * inv_d if (call_code == 2 or cpn_code == 2) else np.empty(0, dtype=np.float64)

    def _fill_idx_slice(code_idx: int, b_s: int, b_e: int, m: int, out_buf: npt.NDArray[np.float64], tmp_b: npt.NDArray[np.float64]) -> None:
        if code_idx == 2:
            sb_m = sum_base[m]
            pm_m = p_pre[m]
            for bi, row_i in enumerate(range(b_s, b_e)):
                aa_i = int(aa_pre[row_i])
                ab_i = int(ab_pre[row_i])
                if aa_i == -2:
                    out_buf[bi] = sum_sp_p[m] if asl_pre[row_i] == 0 else sum_sp_m[m]
                elif aa_i == -1:
                    np.multiply(sb_m, inv_d, out=out_buf[bi])
                elif ab_i < 0:
                    np.subtract(sb_m, pm_m[aa_i], out=tmp_b[bi])
                    tmp_b[bi] += pm_sp[int(asl_pre[row_i]), m]
                    np.multiply(tmp_b[bi], inv_d, out=out_buf[bi])
                else:
                    np.subtract(sb_m, pm_m[aa_i], out=tmp_b[bi])
                    tmp_b[bi] -= pm_m[ab_i]
                    tmp_b[bi] += pm_sp[int(asl_pre[row_i]), m]
                    tmp_b[bi] += pm_sp[int(bsl_pre[row_i]), m]
                    np.multiply(tmp_b[bi], inv_d, out=out_buf[bi])
            return
        is_min = code_idx == 0
        pk = min_pack if is_min else max_pack
        assert pk is not None
        k1, v1, k2, v2, k3, v3, all_p, all_m = pk
        k1_m, v1_m, k2_m, v2_m, v3_m = k1[m], v1[m], k2[m], v2[m], v3[m]
        comb_fn = np.minimum if is_min else np.maximum
        for bi, row_i in enumerate(range(b_s, b_e)):
            aa_i = int(aa_pre[row_i])
            ab_i = int(ab_pre[row_i])
            if aa_i == -2:
                out_buf[bi] = all_p[m] if asl_pre[row_i] == 0 else all_m[m]
            elif aa_i == -1:
                out_buf[bi] = v1_m
            elif ab_i < 0:
                np.copyto(tmp_b[bi], v2_m, where=(k1_m == aa_i))
                np.copyto(tmp_b[bi], v1_m, where=(k1_m != aa_i))
                comb_fn(tmp_b[bi], pm_sp[int(asl_pre[row_i]), m], out=out_buf[bi])
            else:
                m1 = (k1_m == aa_i) | (k1_m == ab_i)
                m2 = (k2_m == aa_i) | (k2_m == ab_i)
                np.copyto(tmp_b[bi], v1_m)
                np.copyto(tmp_b[bi], v2_m, where=m1)
                np.copyto(tmp_b[bi], v3_m, where=m1 & m2)
                comb_fn(pm_sp[int(asl_pre[row_i]), m], pm_sp[int(bsl_pre[row_i]), m], out=out_buf[bi])
                comb_fn(tmp_b[bi], out_buf[bi], out=out_buf[bi])

    def _c2_s5_inplace(x_in: npt.NDArray[np.float64], lo_v: float, inv_w: float, out_buf: npt.NDArray[np.float64], u_buf: npt.NDArray[np.float64]) -> None:
        np.subtract(x_in, lo_v, out=u_buf)
        u_buf *= inv_w
        np.clip(u_buf, 0.0, 1.0, out=u_buf)
        np.multiply(6.0, u_buf, out=out_buf)
        out_buf -= 15.0
        out_buf *= u_buf
        out_buf += 10.0
        out_buf *= u_buf
        out_buf *= u_buf
        out_buf *= u_buf

    hard_ko = abs(ko_spread) <= 1e-6
    eff_csp = -cpn_spread if cpn_dir_down else cpn_spread
    hard_cpn = abs(eff_csp) <= 1e-6
    m_last = m_rem - 1
    c_last = float(cpn_rates[m_last])
    df_last = float(df_pay[m_last])
    wf_last = float(w_fund[m_last]) if w_fund is not None else 0.0
    gk_f = frozen_term[0][None, :]
    egc_f = frozen_term[1][None, :]
    opt_f = frozen_term[2][None, :]
    out_pv = np.empty(n_rows, dtype=np.float64)

    r_blk = 8
    for b_s in range(0, n_rows, r_blk):
        b_e = min(n_rows, b_s + r_blk)
        nb = b_e - b_s
        surv_prev = np.ones((nb, n_tot), dtype=np.float64)
        c_mem = np.full((nb, n_tot), float(c_mem_0), dtype=np.float64) if is_memory else np.empty(0, dtype=np.float64)
        pv_path = np.zeros((nb, n_tot), dtype=np.float64)
        w_buf = np.empty((nb, n_tot), dtype=np.float64)
        w_cpn_buf = np.empty((nb, n_tot), dtype=np.float64)
        gk_buf = np.empty((nb, n_tot), dtype=np.float64)
        gc_buf = np.empty((nb, n_tot), dtype=np.float64)
        u_buf = np.empty((nb, n_tot), dtype=np.float64)
        pm_buf = np.empty((nb, n_tot), dtype=np.float64)
        tmp_buf = np.empty((nb, n_tot), dtype=np.float64)

        for m in range(m_pre):
            eps_m = float(eps_vec[m])
            callable_m = is_callable[m] > 0.5
            guar_m = is_guar[m] > 0.5
            if callable_m:
                _fill_idx_slice(call_code, b_s, b_e, m, w_buf, tmp_buf)
                bk = float(ko_barriers[m])
                lo_k = (bk - 0.5 * eps_m) if hard_ko else min(bk, bk + ko_spread)
                hi_k = (bk + 0.5 * eps_m) if hard_ko else max(bk, bk + ko_spread)
                _c2_s5_inplace(w_buf, lo_k, 1.0 / max(hi_k - lo_k, 1e-12), gk_buf, u_buf)
            if not guar_m:
                if callable_m and cpn_code == call_code:
                    w_c_src = w_buf
                else:
                    _fill_idx_slice(cpn_code, b_s, b_e, m, w_cpn_buf, tmp_buf)
                    w_c_src = w_cpn_buf
                bc = float(cpn_barriers[m])
                lo_c = (bc - 0.5 * eps_m) if hard_cpn else min(bc, bc + eff_csp)
                hi_c = (bc + 0.5 * eps_m) if hard_cpn else max(bc, bc + eff_csp)
                _c2_s5_inplace(w_c_src, lo_c, 1.0 / max(hi_c - lo_c, 1e-12), gc_buf, u_buf)
                if cpn_dir_down != cpn_type_out:
                    np.subtract(1.0, gc_buf, out=gc_buf)
                if callable_m and force_cpn_ac:
                    np.maximum(gc_buf, gk_buf, out=gc_buf)
            else:
                gc_buf.fill(1.0)

            c_rate = float(cpn_rates[m])
            df_m = float(df_pay[m])
            if callable_m:
                np.multiply(surv_prev, gk_buf, out=pm_buf)
            else:
                pm_buf.fill(0.0)
            if is_memory:
                np.add(c_mem, c_rate, out=u_buf)
                np.multiply(surv_prev, gc_buf, out=tmp_buf)
                tmp_buf *= u_buf
                if callable_m:
                    tmp_buf += pm_buf * float(ac_cpn)
                np.subtract(1.0, gc_buf, out=c_mem)
                c_mem *= surv_prev
                c_mem *= u_buf
            else:
                np.multiply(surv_prev, gc_buf, out=tmp_buf)
                tmp_buf *= c_rate
                if callable_m:
                    tmp_buf += pm_buf * float(ac_cpn)
            if callable_m and (not no_notional):
                tmp_buf += pm_buf
            tmp_buf *= df_m
            pv_path += tmp_buf
            if w_fund is not None:
                pv_path -= float(w_fund[m]) * surv_prev
            if callable_m:
                surv_prev -= pm_buf
                np.maximum(surv_prev, 0.0, out=surv_prev)

        eff_c_last = (c_last + c_mem) if is_memory else c_last
        np.multiply(surv_prev, gk_f, out=pm_buf)
        cpn_last = surv_prev * egc_f * eff_c_last + pm_buf * float(ac_cpn)
        np.subtract(surv_prev, pm_buf, out=tmp_buf)
        np.maximum(tmp_buf, 0.0, out=tmp_buf)
        if w_fund is not None:
            pv_path -= wf_last * surv_prev
        term_pay = (cpn_last + tmp_buf * opt_f) if no_notional else (pm_buf + cpn_last + tmp_buf * (1.0 + opt_f))
        pv_path += df_last * term_pay
        out_pv[b_s:b_e] = pv_path.mean(axis=1)

    _ = want_gamma
    return out_pv


def _compute_payoff_aad_state_adjoints_triton(
    perf_m_d_n: torch.Tensor,
    *,
    ko_barriers: torch.Tensor,
    is_callable: torch.Tensor,
    cpn_barriers: torch.Tensor,
    is_guar: torch.Tensor,
    cpn_rates: torch.Tensor,
    df_pay: torch.Tensor,
    w_fund: torch.Tensor | None,
    inv_perm_g: torch.Tensor,
    payoff_f64_g: torch.Tensor,
    call_code: int,
    cpn_code: int,
    eff_opt_code: int,
    cpn_dir_down: bool,
    cpn_type_out: bool,
    is_memory: bool,
    force_cpn_ac: bool,
    no_notional: bool,
    has_ki: bool,
    ki_barrier: float,
    i_ki_past: float,
    has_one_star: bool,
    tau_obs: npt.NDArray[np.float64],
    vol_atm_obs: npt.NDArray[np.float64] | None,
    barrier_eps_min: float,
    barrier_eps_max: float,
    compute_cf_wf: bool = False,
    eps_g_cached: Optional[torch.Tensor] = None,
    n_rep: int = 0,
    has_lower_strike: bool = True,
    has_part_call: bool = False,
    emit_stage1: bool = False,
    i_ki_ext_g: Optional[torch.Tensor] = None,
    out_stage1_rep_g: Optional[torch.Tensor] = None,
    is_long_basket: bool = False,
    cg_flags_g: Optional[torch.Tensor] = None,
    use_rt_flags: bool = False,
    d_pad_k: int = 0,
    use_cg_arena: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Triton forward+reverse pathwise payoff adjoints w.r.t. log_perf (M, D, N) and fwd_perf (D, M)."""
    m_rem, d_assets, n_tot = perf_m_d_n.shape
    dev = perf_m_d_n.device
    if eps_g_cached is not None:
        eps_g = eps_g_cached
    else:
        tau_prev = np.concatenate([[0.0], tau_obs[:-1]])
        dt_period = np.maximum(tau_obs - tau_prev, 1e-8)
        sig_est = np.maximum(np.mean(vol_atm_obs, axis=0), 0.05) if vol_atm_obs is not None else np.full(m_rem, 0.25, dtype=np.float64)
        eps_vec_np = np.clip(
            0.65 * sig_est * np.sqrt(dt_period),
            max(float(barrier_eps_min), 0.015),
            max(float(barrier_eps_max), 0.08),
        )
        eps_g = torch.as_tensor(eps_vec_np, dtype=torch.float64, device=dev)
    use_f32_aad = bool(n_tot <= 16384 and os.environ.get("DEEPVOL_AAD_F32", "1") == "1")
    target_p_dt = torch.float32 if use_f32_aad else torch.float64
    p_kern = (
        perf_m_d_n
        if (perf_m_d_n.dtype == target_p_dt and perf_m_d_n.is_contiguous())
        else perf_m_d_n.to(dtype=target_p_dt).contiguous()
    )
    eff_rt_flags = bool(use_rt_flags and cg_flags_g is not None and d_pad_k > 0)
    ws_g = _alloc_cg_buf(
        use_cg_arena,
        dev,
        "perf_aux",
        (7, m_rem, n_tot),
        torch.float32 if use_f32_aad else torch.float64,
        zero=eff_rt_flags,
    )
    bar_lx_post_g = _alloc_cg_buf(
        use_cg_arena,
        dev,
        "bar_lx_post",
        (m_rem, d_assets, n_tot),
        torch.float32,
        zero=eff_rt_flags,
    )
    eff_n_rep = n_rep if n_rep > 0 else n_tot
    eff_r_cnt = max(n_tot // eff_n_rep, 1)
    block_n = min(128, max(32, _next_power_of_two_int(eff_n_rep)))
    blocks_per_rep = (eff_n_rep + block_n - 1) // block_n
    grid_n = eff_r_cnt * blocks_per_rep
    out_cf_wf_g = (
        _alloc_cg_buf(
            use_cg_arena,
            dev,
            "out_cf_wf",
            (2, grid_n, m_rem),
            torch.float64,
            zero=eff_rt_flags,
        )
        if compute_cf_wf
        else None
    )
    if emit_stage1 and out_stage1_rep_g is not None:
        out_stage1_rep_g.zero_()
    _fast_triton_launch(
        _payoff_aad_state_adjoints_f64_kernel,
        (grid_n,),
        p_kern,
        ws_g,
        bar_lx_post_g,
        out_cf_wf_g,
        out_stage1_rep_g if emit_stage1 else None,
        i_ki_ext_g,
        df_pay,
        w_fund,
        ko_barriers,
        is_callable,
        cpn_barriers,
        is_guar,
        cpn_rates,
        eps_g,
        inv_perm_g,
        payoff_f64_g,
        d_assets,
        m_rem,
        n_tot,
        cg_flags_ptr=cg_flags_g if eff_rt_flags else None,
        n_rep=eff_n_rep,
        blocks_per_rep=blocks_per_rep,
        CALL_CODE=0 if eff_rt_flags else call_code,
        CPN_CODE=0 if eff_rt_flags else cpn_code,
        EFF_OPT_CODE=0 if eff_rt_flags else eff_opt_code,
        HAS_KI=False if eff_rt_flags else has_ki,
        KI_ACTIVE=False if eff_rt_flags else bool(has_ki and ki_barrier > 0.0 and i_ki_past < 0.5),
        HAS_ONE_STAR=False if eff_rt_flags else has_one_star,
        IS_MEM=False if eff_rt_flags else is_memory,
        NO_NOTIONAL=False if eff_rt_flags else no_notional,
        FORCE_CPN_AC=False if eff_rt_flags else force_cpn_ac,
        CPN_DIR_DOWN=False if eff_rt_flags else cpn_dir_down,
        CPN_TYPE_OUT=False if eff_rt_flags else cpn_type_out,
        BLOCK_N=block_n,
        HAS_LOWER_STRIKE=True if eff_rt_flags else has_lower_strike,
        HAS_PART_CALL=False if eff_rt_flags else has_part_call,
        EMIT_STAGE1=bool(emit_stage1 and out_stage1_rep_g is not None),
        HAS_EXT_KI=bool(i_ki_ext_g is not None),
        USE_F32_INNER=use_f32_aad,
        IS_LONG=False if eff_rt_flags else is_long_basket,
        USE_RT_FLAGS=eff_rt_flags,
        D_PAD=d_pad_k if eff_rt_flags else 0,
        num_warps=4,
    )
    bar_ln_fwd = bar_lx_post_g.sum(dim=2, dtype=torch.float64).T.contiguous()
    bar_cf_wf = out_cf_wf_g.sum(dim=1) if out_cf_wf_g is not None else None
    return bar_lx_post_g, bar_ln_fwd, bar_cf_wf


def _map_t_eff_to_surf_np(
    surf: Any,
    t_eff: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Maps Act365 simulation times to surface DCC coordinates and local variance scaling ratios."""
    dcc_str = str(getattr(surf, "dcc", "Act365")).strip().lower()
    t_dcc = np.asarray(surf.tenors, dtype=np.float64).ravel()
    t_365 = np.asarray(getattr(surf, "tenors_act365", surf.tenors), dtype=np.float64).ravel()
    if (
        dcc_str in ("act365", "act365f", "act_365", "act_365f")
        or t_dcc.size == 0
        or t_365.size != t_dcc.size
    ):
        return t_eff.copy(), np.ones_like(t_eff)
    k365 = np.concatenate([[0.0], t_365])
    kdcc = np.concatenate([[0.0], t_dcc])
    seg_t = np.clip(np.searchsorted(k365, t_eff, side="right") - 1, 0, t_365.size - 1)
    dt365 = np.maximum(k365[seg_t + 1] - k365[seg_t], 1e-12)
    slope_seg = (kdcc[seg_t + 1] - kdcc[seg_t]) / dt365
    t_in = kdcc[seg_t] + slope_seg * (t_eff - k365[seg_t])
    slope_tail = float(kdcc[-1] / max(float(k365[-1]), 1e-12))
    t_tail = kdcc[-1] + slope_tail * (t_eff - k365[-1])
    past_last = t_eff > k365[-1]
    t_surf = np.maximum(np.where(past_last, t_tail, t_in), 1e-6)
    dt_ratio = np.maximum(np.where(past_last, slope_tail, slope_seg), 1e-12)
    return t_surf, dt_ratio


def _eval_basis_matrices_for_slice(
    surf: Any,
    s_mat: npt.NDArray[np.float64],
    k_clip: npt.NDArray[np.float64],
    seg_idx: npt.NDArray[np.int64],
    dx: npt.NDArray[np.float64],
    left_mask: npt.NDArray[np.bool_],
    right_mask: npt.NDArray[np.bool_],
    slice_idx_1d: Optional[npt.NDArray[np.int64]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluates (B, dB/dK, d2B/dK2) spline weights cleanly for both CubicSpline and PCHIP."""
    n_u = surf.n_unique_strikes
    lv_nt, lv_ny = s_mat.shape
    k_min = float(surf.strikes[0])
    k_max = float(surf.strikes[-1])
    per_slice = getattr(surf, "_fast_basis_coeffs_per_slice", None)
    fast_basis = getattr(surf, "_fast_basis_coeffs", None)

    if per_slice is not None and slice_idx_1d is not None:
        cb = per_slice.swapaxes(0, 1)[:, slice_idx_1d[:, None], seg_idx, :]
        dx_e = dx[:, :, None]
        b_mat = ((cb[0] * dx_e + cb[1]) * dx_e + cb[2]) * dx_e + cb[3]
        bp_mat = (3.0 * cb[0] * dx_e + 2.0 * cb[1]) * dx_e + cb[2]
        bpp_mat = 6.0 * cb[0] * dx_e + 2.0 * cb[1]
        sl_ps = getattr(surf, "_basis_slope_left_per_slice", None)
        sr_ps = getattr(surf, "_basis_slope_right_per_slice", None)
        sl_b = sl_ps[slice_idx_1d][:, None, :] if sl_ps is not None else getattr(surf, "_basis_slope_left", None)
        sr_b = sr_ps[slice_idx_1d][:, None, :] if sr_ps is not None else getattr(surf, "_basis_slope_right", None)
    elif fast_basis is not None:
        cb = fast_basis[:, seg_idx, :]
        dx_e = dx[:, :, None]
        b_mat = ((cb[0] * dx_e + cb[1]) * dx_e + cb[2]) * dx_e + cb[3]
        bp_mat = (3.0 * cb[0] * dx_e + 2.0 * cb[1]) * dx_e + cb[2]
        bpp_mat = 6.0 * cb[0] * dx_e + 2.0 * cb[1]
        sl_b = getattr(surf, "_basis_slope_left", None)
        sr_b = getattr(surf, "_basis_slope_right", None)
    elif getattr(surf, "_basis_cs", None) is not None:
        k_flat = k_clip.reshape(-1)
        b_mat = np.asarray(surf._basis_cs(k_flat), dtype=np.float64).reshape(lv_nt, lv_ny, n_u)
        bp_mat = np.asarray(surf._basis_cs(k_flat, 1), dtype=np.float64).reshape(lv_nt, lv_ny, n_u)
        bpp_mat = np.asarray(surf._basis_cs(k_flat, 2), dtype=np.float64).reshape(lv_nt, lv_ny, n_u)
        sl_b = getattr(surf, "_basis_slope_left", None)
        sr_b = getattr(surf, "_basis_slope_right", None)
    else:
        raise RuntimeError("RegularizedVolSurface missing basis spline representation.")

    if sl_b is not None and np.any(left_mask):
        e0 = np.zeros(n_u, dtype=np.float64)
        e0[0] = 1.0
        sl_b_bcast = sl_b[None, None, :] if sl_b.ndim == 1 else sl_b
        b_left = e0[None, None, :] + (s_mat - k_min)[:, :, None] * sl_b_bcast
        b_mat = np.where(left_mask[:, :, None], b_left, b_mat)
        bp_mat = np.where(left_mask[:, :, None], sl_b_bcast, bp_mat)
        bpp_mat = np.where(left_mask[:, :, None], 0.0, bpp_mat)
    if sr_b is not None and np.any(right_mask):
        e1 = np.zeros(n_u, dtype=np.float64)
        e1[-1] = 1.0
        sr_b_bcast = sr_b[None, None, :] if sr_b.ndim == 1 else sr_b
        b_right = e1[None, None, :] + (s_mat - k_max)[:, :, None] * sr_b_bcast
        b_mat = np.where(right_mask[:, :, None], b_right, b_mat)
        bp_mat = np.where(right_mask[:, :, None], sr_b_bcast, bp_mat)
        bpp_mat = np.where(right_mask[:, :, None], 0.0, bpp_mat)
    return b_mat, bp_mat, bpp_mat


def _build_lv_table_and_forward_cache(
    surf: Any,
    spot_0: float,
    tau_obs_np: npt.NDArray[np.float64],
    r_native_obs_np: npt.NDArray[np.float64],
    lv_t_grid: npt.NDArray[np.float64],
    lv_y_grid: npt.NDArray[np.float64],
    div_pv_obs_np: npt.NDArray[np.float64],
    *,
    grid_dig: Any = None,
    surf_dig: Optional[str] = None,
    rn_div_dig: Any = None,
    need_vega_jac: bool = True,
) -> Dict[str, Any]:
    """Builds 2D Dupire LV table and stores compact Stage-A state for factored Vega pullback."""
    from deepvol.models.autocall.greeks_engine import _escrowed_spot_softplus_np

    ensure_fn = getattr(surf, "ensure_basis_spline_coeffs", None) or surf._ensure_basis_spline
    ensure_fn()

    eff_surf_dig = surf_dig if surf_dig is not None else _surface_content_digest(surf)
    base_grid_dig = grid_dig if grid_dig is not None else _content_digest(tau_obs_np, lv_t_grid, lv_y_grid)
    eff_rd_dig = rn_div_dig if rn_div_dig is not None else _content_digest(r_native_obs_np, div_pv_obs_np)
    cache_key = (
        "lv_factored" if need_vega_jac else "lv_sde",
        eff_surf_dig,
        float(spot_0),
        base_grid_dig,
        eff_rd_dig,
    )
    cached = _CRN_VEGA_JAC_CACHE.get(cache_key)
    if cached is not None:
        return cast(Dict[str, Any], cached)

    t_eff = np.maximum(np.asarray(lv_t_grid, dtype=np.float64), 1e-4)
    t_surf, dt_surf_ratio = _map_t_eff_to_surf_np(surf, t_eff)
    tau_knots = np.concatenate([[0.0], np.asarray(tau_obs_np, dtype=np.float64)])
    r_nat_knots = np.concatenate([[0.0], np.asarray(r_native_obs_np, dtype=np.float64)])
    m_len = len(tau_obs_np)
    k_idx = np.clip(np.searchsorted(tau_knots, t_eff, side="right") - 1, 0, max(m_len - 1, 0))
    tau0 = tau_knots[k_idx]
    tau1 = tau_knots[k_idx + 1]
    dt_k = np.maximum(tau1 - tau0, 1e-8)
    alpha_k = np.clip((t_eff - tau0) / dt_k, 0.0, 1.0)
    r_nat_grid = r_nat_knots[k_idx] + alpha_k * (r_nat_knots[k_idx + 1] - r_nat_knots[k_idx])
    dr_nat_dt = (r_nat_knots[k_idx + 1] - r_nat_knots[k_idx]) / dt_k

    sp0_safe = max(float(spot_0), 1e-8)
    if len(div_pv_obs_np) == m_len:
        dpv_k_arr = np.concatenate([[0.0], np.asarray(div_pv_obs_np, dtype=np.float64)])
        esc_k_arr = _escrowed_spot_softplus_np(
            np.full(m_len + 1, sp0_safe, dtype=np.float64), dpv_k_arr
        )
        dpv_knots: Optional[npt.NDArray[np.float64]] = dpv_k_arr
        esc_knots: Optional[npt.NDArray[np.float64]] = esc_k_arr
        esc0 = esc_k_arr[k_idx]
        esc1 = esc_k_arr[k_idx + 1]
        q_div_grid = -np.log(np.maximum(esc1 / np.maximum(esc0, 1e-12), 1e-12)) / dt_k
        esc_grid = esc0 * np.exp(-q_div_grid * np.maximum(t_eff - tau0, 0.0))
    else:
        dpv_knots = None
        esc_knots = None
        q_div_grid = np.zeros_like(t_eff)
        esc_grid = np.full_like(t_eff, sp0_safe)

    f_t = np.maximum(esc_grid * np.exp(r_nat_grid), 1e-4)
    mu_eff_grid = dr_nat_dt - q_div_grid
    s_mat = np.maximum(f_t[:, None] * np.exp(lv_y_grid[None, :]), 1e-4)
    lv_nt, _ = s_mat.shape

    k_min = float(surf.strikes[0])
    k_max = float(surf.strikes[-1])
    k_clip = np.clip(s_mat, k_min, k_max)
    seg_idx = np.clip(np.searchsorted(surf.strikes, k_clip, side="right") - 1, 0, surf.n_unique_strikes - 2)
    dx = k_clip - surf.strikes[seg_idx]
    left_mask = s_mat < k_min
    right_mask = s_mat > k_max

    def _eval_slice(s_idx_1d: npt.NDArray[np.int64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        if getattr(surf, "_fast_slice_coeffs", None) is not None:
            c = surf._fast_slice_coeffs[:, seg_idx, s_idx_1d[:, None]]
            c0, c1, c2, c3 = c[0], c[1], c[2], c[3]
            val = ((c0 * dx + c1) * dx + c2) * dx + c3
            dval = (3.0 * c0 * dx + 2.0 * c1) * dx + c2
            d2val = 6.0 * c0 * dx + 2.0 * c1
            d3val = np.broadcast_to(6.0 * c0, val.shape).copy()
            if np.any(left_mask):
                sl = surf._slopes_left[s_idx_1d][:, None]
                q0 = surf.quotes[s_idx_1d, 0][:, None]
                val = np.where(left_mask, q0 + sl * (s_mat - k_min), val)
                dval = np.where(left_mask, sl, dval)
                d2val = np.where(left_mask, 0.0, d2val)
                d3val = np.where(left_mask, 0.0, d3val)
            if np.any(right_mask):
                sr = surf._slopes_right[s_idx_1d][:, None]
                q_end = surf.quotes[s_idx_1d, -1][:, None]
                val = np.where(right_mask, q_end + sr * (s_mat - k_max), val)
                dval = np.where(right_mask, sr, dval)
                d2val = np.where(right_mask, 0.0, d2val)
                d3val = np.where(right_mask, 0.0, d3val)
        else:
            bm, bpm, bppm = _eval_basis_matrices_for_slice(
                surf, s_mat, k_clip, seg_idx, dx, left_mask, right_mask, s_idx_1d
            )
            q_s = surf.quotes[s_idx_1d, :]
            q_s_3 = q_s[:, None, :]
            val = (bm * q_s_3).sum(axis=2)
            dval = (bpm * q_s_3).sum(axis=2)
            d2val = (bppm * q_s_3).sum(axis=2)
            d3val = np.zeros_like(val)
        min_v = float(surf.min_vol)
        clamped_low = val < min_v
        if np.any(clamped_low):
            val = np.maximum(val, min_v)
            dval = np.where(clamped_low, 0.0, dval)
            d2val = np.where(clamped_low, 0.0, d2val)
            d3val = np.where(clamped_low, 0.0, d3val)
        return val, dval, d2val, d3val, clamped_low

    t_first = float(surf.tenors[0])
    t_last = float(surf.tenors[-1])
    dw_dk: npt.NDArray[np.float64]
    d2w_dk2: npt.NDArray[np.float64]
    dw0_dk: npt.NDArray[np.float64]
    dw1_dk: npt.NDArray[np.float64]
    if surf.n_exp == 1:
        idx0 = np.zeros(lv_nt, dtype=np.int64)
        idx1 = idx0
        is_bnd = np.ones(lv_nt, dtype=np.bool_)
        v0, dv0, d2v0, d3v0, cl0 = _eval_slice(idx0)
        v1, dv1, d2v1, d3v1, cl1 = v0, dv0, d2v0, d3v0, cl0
        t0 = t_surf[:, None]
        t1 = t_surf[:, None]
        dt_inv = np.zeros((lv_nt, 1), dtype=np.float64)
        alpha = np.zeros((lv_nt, 1), dtype=np.float64)
        w = (v0 * v0) * t0
        dw_dt = v0 * v0
        dw0_dk = 2.0 * t0 * v0 * dv0
        dw1_dk = dw0_dk
        dw_dk = dw0_dk
        d2w_dk2 = 2.0 * t0 * (dv0 * dv0 + v0 * d2v0)
    else:
        is_left_t = t_surf <= t_first
        is_right_t = t_surf >= t_last
        is_bnd = is_left_t | is_right_t
        idx0 = np.clip(np.searchsorted(surf.tenors, t_surf, side="right") - 1, 0, surf.n_exp - 2)
        idx1 = idx0 + 1
        idx0 = np.where(is_left_t, 0, np.where(is_right_t, surf.n_exp - 1, idx0))
        idx1 = np.where(is_bnd, idx0, idx1)
        v0, dv0, d2v0, d3v0, cl0 = _eval_slice(idx0)
        v1, dv1, d2v1, d3v1, cl1 = _eval_slice(idx1)
        t0 = np.where(is_bnd, t_surf, surf.tenors[idx0])[:, None]
        t1 = np.where(is_bnd, t_surf, surf.tenors[idx1])[:, None]
        dt_inv = np.where(is_bnd[:, None], 0.0, 1.0 / np.maximum(t1 - t0, 1e-12))
        alpha = np.where(is_bnd[:, None], 0.0, (t_surf[:, None] - t0) * dt_inv)
        w0 = (v0 * v0) * t0
        w1 = (v1 * v1) * t1
        w = (1.0 - alpha) * w0 + alpha * w1
        dw_dt = np.where(is_bnd[:, None], v0 * v0, (w1 - w0) * dt_inv)
        dw0_dk = 2.0 * t0 * v0 * dv0
        dw1_dk = 2.0 * t1 * v1 * dv1
        dw_dk = np.asarray((1.0 - alpha) * dw0_dk + alpha * dw1_dk, dtype=np.float64)
        d2w0_dk2 = 2.0 * t0 * (dv0 * dv0 + v0 * d2v0)
        d2w1_dk2 = 2.0 * t1 * (dv1 * dv1 + v1 * d2v1)
        d2w_dk2 = np.asarray((1.0 - alpha) * d2w0_dk2 + alpha * d2w1_dk2, dtype=np.float64)

    y = np.log(s_mat / f_t[:, None])
    w_safe = np.maximum(w, 1e-8)
    dw_dy = s_mat * dw_dk
    dw_dt_y = dw_dt + mu_eff_grid[:, None] * dw_dy
    d2w_dy2 = (s_mat * s_mat) * d2w_dk2 + dw_dy

    denom = (
        1.0
        - (y / w_safe) * dw_dy
        + 0.25 * (-0.25 - 1.0 / w_safe + (y * y) / (w_safe * w_safe)) * (dw_dy * dw_dy)
        + 0.5 * d2w_dy2
    )
    denom_safe = np.maximum(denom, 0.05)
    min_v = float(surf.min_vol)
    max_lv = float(surf.max_local_vol)
    raw_var = dw_dt_y / denom_safe
    loc_var = np.maximum(raw_var, min_v * min_v)
    lv_dcc = np.sqrt(loc_var)
    sqrt_dt_ratio = np.sqrt(dt_surf_ratio)[:, None]
    lv_raw_365 = lv_dcc * sqrt_dt_ratio
    lv_out_c = np.ascontiguousarray(np.clip(lv_raw_365, min_v, max_lv), dtype=np.float64)
    active_mask = (raw_var > min_v * min_v) & (lv_raw_365 > min_v) & (lv_raw_365 < max_lv)

    entry: Dict[str, Any] = {
        "lv_out": lv_out_c,
        "lv_dcc": lv_dcc,
        "sqrt_dt_ratio": sqrt_dt_ratio,
        "s_mat": s_mat,
        "k_clip": k_clip,
        "seg_idx": seg_idx,
        "dx": dx,
        "left_mask": left_mask,
        "right_mask": right_mask,
        "y": y,
        "w": w,
        "w_safe": w_safe,
        "dw_dy": dw_dy,
        "dw_dt_y": dw_dt_y,
        "denom": denom,
        "denom_safe": denom_safe,
        "active_lv": active_mask,
        "mu_eff_grid": mu_eff_grid,
        "idx0": idx0,
        "idx1": idx1,
        "is_bnd": is_bnd,
        "t0": t0,
        "t1": t1,
        "dt_inv": dt_inv,
        "alpha": alpha,
        "v0": v0,
        "dv0": dv0,
        "d2v0": d2v0,
        "cl0": cl0,
        "v1": v1,
        "dv1": dv1,
        "d2v1": d2v1,
        "d3v0": d3v0,
        "d3v1": d3v1,
        "cl1": cl1,
        "dw_dk": dw_dk,
        "dw0_dk": dw0_dk,
        "dw1_dk": dw1_dk,
        "d2w_dk2": d2w_dk2,
        "m_len": m_len,
        "t_eff": t_eff,
        "tau0": tau0,
        "dt_k": dt_k,
        "k_idx": k_idx,
        "alpha_k": alpha_k,
        "sp0_safe": sp0_safe,
        "f_t": f_t,
        "dpv_knots": dpv_knots,
        "esc_knots": esc_knots,
    }
    _CRN_VEGA_JAC_CACHE.put(cache_key, entry)
    return entry


def _lv_chain_weights_np(lv_cache: Dict[str, Any]) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Computes and caches stage-A local-volatility chain-rule weights (c_w, c_wt, c_wy, c_wyy, c_den)."""
    cw = lv_cache.get("_chain_weights")
    if cw is not None:
        return cast(
            Tuple[
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
            ],
            cw,
        )
    lv_dcc = lv_cache["lv_dcc"]
    c_loc_var = np.where(lv_cache["active_lv"], lv_cache["sqrt_dt_ratio"], 0.0) / (
        2.0 * np.maximum(lv_dcc, 1e-12)
    )
    denom_raw = np.asarray(lv_cache["denom"], dtype=np.float64)
    denom_safe = lv_cache["denom_safe"]
    denom_jac = np.where(denom_raw >= 0.25, denom_safe, np.maximum(denom_safe, 0.25))
    c_wt = c_loc_var / denom_jac
    c_den = np.where(
        denom_raw > 0.05,
        -c_loc_var * lv_cache["dw_dt_y"] / (denom_jac * denom_jac),
        0.0,
    )
    y = lv_cache["y"]
    w_safe = lv_cache["w_safe"]
    wy = lv_cache["dw_dy"]
    c_wyy = 0.5 * c_den
    c_wy = c_den * (-(y / w_safe) + 0.5 * wy * (-0.25 - 1.0 / w_safe + (y * y) / (w_safe * w_safe)))
    d_denom_d_w = (y * wy) / (w_safe * w_safe) + 0.25 * (wy * wy) * (
        1.0 / (w_safe * w_safe) - 2.0 * (y * y) / (w_safe * w_safe * w_safe)
    )
    c_w = np.where(lv_cache["w"] > 1e-8, c_den * d_denom_d_w, 0.0)
    out = (c_w, c_wt, c_wy, c_wyy, c_den)
    lv_cache["_chain_weights"] = out
    return out


def _pullback_lv_to_r_nat_and_div(
    lv_cache: Dict[str, Any],
    bar_lv: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Propagates bar_lv (lv_nt, lv_ny) through buehler_eff_drift & escrowed LV grid to (r_native_obs, div_pv_obs)."""
    m_len = int(lv_cache["m_len"])
    bar_r_obs = np.zeros(m_len, dtype=np.float64)
    bar_div_obs = np.zeros(m_len, dtype=np.float64)
    if m_len <= 0 or float(np.max(np.abs(bar_lv))) <= 0.0:
        return bar_r_obs, bar_div_obs

    pb_rn = lv_cache.get("_rn_div_pullback")
    if pb_rn is None:
        c_w, c_wt, c_wy, c_wyy, _ = _lv_chain_weights_np(lv_cache)
        s_mat = lv_cache["s_mat"]
        f_t = lv_cache["f_t"]
        is_bnd = lv_cache["is_bnd"][:, None]
        alpha = lv_cache["alpha"]
        dt_inv = lv_cache["dt_inv"]
        t0, t1 = lv_cache["t0"], lv_cache["t1"]
        v0, dv0, d2v0, d3v0, cl0 = lv_cache["v0"], lv_cache["dv0"], lv_cache["d2v0"], lv_cache["d3v0"], lv_cache["cl0"]
        v1, dv1, d2v1, d3v1, cl1 = lv_cache["v1"], lv_cache["dv1"], lv_cache["d2v1"], lv_cache["d3v1"], lv_cache["cl1"]
        dw_dk = lv_cache["dw_dk"]
        dw0_dk = lv_cache["dw0_dk"]
        dw1_dk = lv_cache["dw1_dk"]
        d2w_dk2 = lv_cache["d2w_dk2"]

        d3w0_dk3 = np.where(cl0, 0.0, 2.0 * t0 * (3.0 * dv0 * d2v0 + v0 * d3v0))
        d3w1_dk3 = np.where(cl1, 0.0, 2.0 * t1 * (3.0 * dv1 * d2v1 + v1 * d3v1))
        d3w_dk3 = np.where(is_bnd, d3w0_dk3, (1.0 - alpha) * d3w0_dk3 + alpha * d3w1_dk3)

        s_act = (s_mat > 1e-4) & (f_t[:, None] > 1e-4)
        s_m = np.where(s_act, s_mat, 0.0)
        dw_dlnf = lv_cache["dw_dy"]
        dwy_dlnf = s_m * (dw_dk + s_m * d2w_dk2)
        dwyy_dlnf = (s_m * s_m * s_m) * d3w_dk3 + 3.0 * (s_m * s_m) * d2w_dk2 + s_m * dw_dk
        dwt_bnd = np.where(cl0, 0.0, 2.0 * v0 * (s_m * dv0))
        dwt_int = s_m * (dw1_dk - dw0_dk) * dt_inv
        dwt_dlnf = np.where(is_bnd, dwt_bnd, dwt_int)
        dwty_dlnf = dwt_dlnf + lv_cache["mu_eff_grid"][:, None] * dwy_dlnf

        w_lnf = np.ascontiguousarray(c_w * dw_dlnf + c_wy * dwy_dlnf + c_wyy * dwyy_dlnf + c_wt * dwty_dlnf, dtype=np.float64)
        w_mu = np.ascontiguousarray(c_wt * lv_cache["dw_dy"], dtype=np.float64)

        k_idx = np.asarray(lv_cache["k_idx"], dtype=np.int64)
        alpha_k = np.asarray(lv_cache["alpha_k"], dtype=np.float64)
        dt_k = np.asarray(lv_cache["dt_k"], dtype=np.float64)
        lv_nt_l = int(w_lnf.shape[0])
        t_idx = np.arange(lv_nt_l, dtype=np.int64)
        r_knot_lnf = np.zeros((m_len + 1, lv_nt_l), dtype=np.float64)
        r_knot_mu = np.zeros((m_len + 1, lv_nt_l), dtype=np.float64)
        r_knot_lnf[k_idx, t_idx] += 1.0 - alpha_k
        r_knot_lnf[k_idx + 1, t_idx] += alpha_k
        inv_dt_k = 1.0 / dt_k
        r_knot_mu[k_idx, t_idx] -= inv_dt_k
        r_knot_mu[k_idx + 1, t_idx] += inv_dt_k
        s_r_lnf = np.ascontiguousarray(r_knot_lnf[1:], dtype=np.float64)
        s_r_mu = np.ascontiguousarray(r_knot_mu[1:], dtype=np.float64)

        s_div_lnf: Optional[npt.NDArray[np.float64]] = None
        s_div_mu: Optional[npt.NDArray[np.float64]] = None
        esc_knots = lv_cache.get("esc_knots")
        dpv_knots = lv_cache.get("dpv_knots")
        if esc_knots is not None and dpv_knots is not None:
            w_esc = np.maximum(lv_cache["t_eff"] - lv_cache["tau0"], 0.0) / dt_k
            esc_lnf = np.zeros((m_len + 1, lv_nt_l), dtype=np.float64)
            esc_mu = np.zeros((m_len + 1, lv_nt_l), dtype=np.float64)
            esc_lnf[k_idx, t_idx] += 1.0 - w_esc
            esc_lnf[k_idx + 1, t_idx] += w_esc
            esc_mu[k_idx, t_idx] -= inv_dt_k
            esc_mu[k_idx + 1, t_idx] += inv_dt_k
            sp0_safe = float(lv_cache["sp0_safe"])
            floor_sp = 0.01 * sp0_safe
            scale_sp = max(floor_sp, 1e-12)
            z_esc = (sp0_safe - dpv_knots[1:] - floor_sp) / scale_sp
            sig_z = np.where(
                z_esc >= 0.0,
                1.0 / (1.0 + np.exp(-np.minimum(z_esc, 40.0))),
                np.exp(np.maximum(z_esc, -40.0)) / (1.0 + np.exp(np.maximum(z_esc, -40.0))),
            )
            div_scale = (-sig_z / np.maximum(esc_knots[1:], 1e-12))[:, None]
            s_div_lnf = np.ascontiguousarray(esc_lnf[1:] * div_scale, dtype=np.float64)
            s_div_mu = np.ascontiguousarray(esc_mu[1:] * div_scale, dtype=np.float64)

        pb_rn = (w_lnf, w_mu, s_r_lnf, s_r_mu, s_div_lnf, s_div_mu)
        lv_cache["_rn_div_pullback"] = pb_rn

    w_lnf, w_mu, s_r_lnf, s_r_mu, s_div_lnf, s_div_mu = pb_rn
    bar_lv_arr = np.asarray(bar_lv, dtype=np.float64)
    bar_ln_f = (bar_lv_arr * w_lnf).sum(axis=1)
    bar_mu_eff = (bar_lv_arr * w_mu).sum(axis=1)
    bar_r_obs = s_r_lnf @ bar_ln_f + s_r_mu @ bar_mu_eff
    if s_div_lnf is not None and s_div_mu is not None:
        bar_div_obs = s_div_lnf @ bar_ln_f + s_div_mu @ bar_mu_eff
    return bar_r_obs, bar_div_obs


def _pullback_lv_to_raw_quotes(
    surf: Any,
    lv_cache: Dict[str, Any],
    bar_lv: npt.NDArray[np.float64],
    *,
    smooth_t_gaps: bool = False,
) -> npt.NDArray[np.float64]:
    """Propagates bar_lv (lv_nt, lv_ny) via dense cached pullback matrix to raw surface quotes."""
    del smooth_t_gaps
    pb_cache = lv_cache.get("_dense_pullback")
    if pb_cache is None:
        c_w, c_wt, c_wy, c_wyy, _ = _lv_chain_weights_np(lv_cache)
        s_mat = lv_cache["s_mat"]
        c_dk = s_mat * (c_wy + lv_cache["mu_eff_grid"][:, None] * c_wt + c_wyy)
        c_dk2 = (s_mat * s_mat) * c_wyy
        is_bnd = lv_cache["is_bnd"][:, None]
        alpha = lv_cache["alpha"]
        dt_inv = lv_cache["dt_inv"]
        t0, t1 = lv_cache["t0"], lv_cache["t1"]
        v0, dv0, d2v0, cl0 = lv_cache["v0"], lv_cache["dv0"], lv_cache["d2v0"], lv_cache["cl0"]
        v1, dv1, d2v1, cl1 = lv_cache["v1"], lv_cache["dv1"], lv_cache["d2v1"], lv_cache["cl1"]
        c_w0 = (1.0 - alpha) * c_w - dt_inv * c_wt
        c_w1 = alpha * c_w + dt_inv * c_wt
        c_v0 = np.where(
            cl0,
            0.0,
            np.where(
                is_bnd,
                c_w * (2.0 * t0 * v0) + c_wt * (2.0 * v0) + c_dk * (2.0 * t0 * dv0) + c_dk2 * (2.0 * t0 * d2v0),
                c_w0 * (2.0 * t0 * v0)
                + (1.0 - alpha) * c_dk * (2.0 * t0 * dv0)
                + (1.0 - alpha) * c_dk2 * (2.0 * t0 * d2v0),
            ),
        )
        c_dv0 = np.where(
            cl0,
            0.0,
            np.where(
                is_bnd,
                c_dk * (2.0 * t0 * v0) + c_dk2 * (4.0 * t0 * dv0),
                (1.0 - alpha) * c_dk * (2.0 * t0 * v0) + (1.0 - alpha) * c_dk2 * (4.0 * t0 * dv0),
            ),
        )
        c_d2v0 = np.where(
            cl0,
            0.0,
            np.where(is_bnd, c_dk2 * (2.0 * t0 * v0), (1.0 - alpha) * c_dk2 * (2.0 * t0 * v0)),
        )
        c_v1 = np.where(
            cl1 | is_bnd,
            0.0,
            c_w1 * (2.0 * t1 * v1) + alpha * c_dk * (2.0 * t1 * dv1) + alpha * c_dk2 * (2.0 * t1 * d2v1),
        )
        c_dv1 = np.where(
            cl1 | is_bnd,
            0.0,
            alpha * c_dk * (2.0 * t1 * v1) + alpha * c_dk2 * (4.0 * t1 * dv1),
        )
        c_d2v1 = np.where(cl1 | is_bnd, 0.0, alpha * c_dk2 * (2.0 * t1 * v1))

        idx0, idx1 = lv_cache["idx0"], lv_cache["idx1"]
        per_slice = getattr(surf, "_fast_basis_coeffs_per_slice", None)
        if per_slice is not None:
            b0, bp0, bpp0 = _eval_basis_matrices_for_slice(
                surf, s_mat, lv_cache["k_clip"], lv_cache["seg_idx"], lv_cache["dx"],
                lv_cache["left_mask"], lv_cache["right_mask"], idx0,
            )
            b1, bp1, bpp1 = _eval_basis_matrices_for_slice(
                surf, s_mat, lv_cache["k_clip"], lv_cache["seg_idx"], lv_cache["dx"],
                lv_cache["left_mask"], lv_cache["right_mask"], idx1,
            )
        else:
            b0, bp0, bpp0 = _eval_basis_matrices_for_slice(
                surf, s_mat, lv_cache["k_clip"], lv_cache["seg_idx"], lv_cache["dx"],
                lv_cache["left_mask"], lv_cache["right_mask"], None,
            )
            b1, bp1, bpp1 = b0, bp0, bpp0
        m0 = np.asarray(c_v0[:, :, None] * b0 + c_dv0[:, :, None] * bp0 + c_d2v0[:, :, None] * bpp0, dtype=np.float64)
        m1 = np.asarray(c_v1[:, :, None] * b1 + c_dv1[:, :, None] * bp1 + c_d2v1[:, :, None] * bpp1, dtype=np.float64)
        n_exp_s = int(surf.n_exp)
        n_raw_s = int(surf.n_raw_strikes)
        lv_nt_s, lv_ny_s, n_u_s = m0.shape
        t_arange = np.arange(lv_nt_s)
        p_full = np.zeros((lv_nt_s, lv_ny_s, n_exp_s, n_u_s), dtype=np.float64)
        p_full[t_arange, :, idx0, :] += m0
        p_full[t_arange, :, idx1, :] += m1
        c_jac = np.asarray(surf.calendar_proj_jacobian, dtype=np.float64)
        eye_e = np.eye(n_exp_s, dtype=np.float64)[:, :, None]
        is_cal_eye = bool(c_jac.shape[:2] == (n_exp_s, n_exp_s) and np.array_equal(c_jac, np.broadcast_to(eye_e, c_jac.shape)))
        if not is_cal_eye:
            p_full = np.einsum("tyeu,eou->tyou", p_full, c_jac)
        d_mat_arr = np.ascontiguousarray(surf.dedup_matrix, dtype=np.float64)
        is_dedup_eye = bool(
            d_mat_arr.shape == (int(surf.n_unique_strikes), n_raw_s)
            and d_mat_arr.shape[0] == d_mat_arr.shape[1]
            and np.array_equal(d_mat_arr, np.eye(d_mat_arr.shape[0], dtype=np.float64))
        )
        if not is_dedup_eye:
            p_full = p_full @ d_mat_arr
        pullback_mat = np.ascontiguousarray(p_full.reshape(lv_nt_s * lv_ny_s, n_exp_s * n_raw_s), dtype=np.float64)
        pb_cache = (pullback_mat, n_exp_s, n_raw_s)
        lv_cache["_dense_pullback"] = pb_cache

    pullback_mat, n_exp_s, n_raw_s = pb_cache
    return np.asarray((np.ascontiguousarray(bar_lv, dtype=np.float64).reshape(-1) @ pullback_mat).reshape(n_exp_s, n_raw_s), dtype=np.float64)


def _cpu_lv_bilinear_np(
    lv_table_d: npt.NDArray[np.float64],
    t_val: float,
    y_val: npt.NDArray[np.float64],
    inv_dt_grid: float,
    y_min_d: float,
    inv_dy_d: float,
    max_u_t: float,
    max_u_y: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int, int, float, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    lv_nt, lv_ny = lv_table_d.shape
    u_t = min(max(t_val * inv_dt_grid, 0.0), max_u_t)
    it0 = int(u_t)
    it1 = min(it0 + 1, lv_nt - 1)
    wt = u_t - float(it0)

    u_y_raw = (y_val - y_min_d) * inv_dy_d
    in_y = (u_y_raw > 0.0) & (u_y_raw < max_u_y)
    u_y = np.clip(u_y_raw, 0.0, max_u_y)
    iy0 = u_y.astype(np.int64)
    iy1 = np.minimum(iy0 + 1, lv_ny - 1)
    wy = u_y - iy0

    v0 = (1.0 - wt) * lv_table_d[it0, iy0] + wt * lv_table_d[it1, iy0]
    v1 = (1.0 - wt) * lv_table_d[it0, iy1] + wt * lv_table_d[it1, iy1]
    sig_out = np.asarray((1.0 - wy) * v0 + wy * v1, dtype=np.float64)
    dsig_dy = np.asarray(np.clip(np.where(in_y, (v1 - v0) * inv_dy_d, 0.0), -4.0, 4.0), dtype=np.float64)
    return sig_out, dsig_dy, it0, it1, wt, iy0, iy1, np.asarray(wy, dtype=np.float64)


def _cpu_step_and_ckpt_sde(
    lv_tables: npt.NDArray[np.float64],
    z_indep: npt.NDArray[np.float64],
    chol_base: npt.NDArray[np.float64],
    tau_obs_np: npt.NDArray[np.float64],
    r_native_obs_np: npt.NDArray[np.float64],
    r_tot_obs_np: npt.NDArray[np.float64],
    lv_t_grid: npt.NDArray[np.float64],
    lv_y_min: float | npt.NDArray[np.float64],
    lv_y_max: float | npt.NDArray[np.float64],
    n_sub: int,
    eff_r: int,
    n_rep: int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """CPU Platen Weak-Order-2.0 forward checkpointing with period-boundary states."""
    d_assets, lv_nt, lv_ny = lv_tables.shape
    m_rem = len(tau_obs_np)
    n_total = eff_r * n_rep

    t_max = float(lv_t_grid[-1])
    inv_dt_grid = float(lv_nt - 1) / max(t_max, 1e-12)
    y_min_vec = np.broadcast_to(np.asarray(lv_y_min, dtype=np.float64), (d_assets,))
    y_max_vec = np.broadcast_to(np.asarray(lv_y_max, dtype=np.float64), (d_assets,))
    inv_dy_vec = float(lv_ny - 1) / np.maximum(y_max_vec - y_min_vec, 1e-12)
    max_u_t = float(lv_nt - 1) - 1e-6
    max_u_y = float(lv_ny - 1) - 1e-6

    z_corr = np.matmul(z_indep, chol_base.T)
    lx_period = np.empty((m_rem, d_assets, n_total), dtype=np.float64)
    norm_x_out = np.empty((m_rem, d_assets, n_total), dtype=np.float64)

    lx = np.zeros((d_assets, eff_r, n_rep), dtype=np.float64)
    t_prev = 0.0
    rt_prev = np.zeros(d_assets, dtype=np.float64)
    rn_prev = np.zeros(d_assets, dtype=np.float64)

    for m in range(m_rem):
        t_cur = float(tau_obs_np[m])
        rt_cur = r_tot_obs_np[:, m]
        rn_cur = r_native_obs_np[:, m]
        d_tau = max(t_cur - t_prev, 1e-10)
        dt = d_tau / float(n_sub)
        sqrt_dt = math.sqrt(dt)
        drt = rt_cur - rt_prev
        drn = rn_cur - rn_prev

        for s in range(n_sub):
            alpha_s = (float(s) + 0.5) / float(n_sub)
            t_mid = t_prev + alpha_s * d_tau
            t_start = max(t_mid - 0.5 * dt, 0.0)
            t_end = t_start + dt
            q_shift_mid = (rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn)

            for d in range(d_assets):
                sig0 = _cpu_lv_bilinear_np(
                    lv_tables[d], t_start, lx[d] + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )[0]
                zd = z_corr[m, s, :, :, d]
                dw = sqrt_dt * zd
                a0 = -0.5 * sig0 * sig0
                lx_hat = lx[d] + a0 * dt + sig0 * dw
                lx_p = lx[d] + a0 * dt + sig0 * sqrt_dt
                lx_m = lx[d] + a0 * dt - sig0 * sqrt_dt
                sigh = _cpu_lv_bilinear_np(
                    lv_tables[d], t_end, lx_hat + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )[0]
                sigp = _cpu_lv_bilinear_np(
                    lv_tables[d], t_end, lx_p + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )[0]
                sigm = _cpu_lv_bilinear_np(
                    lv_tables[d], t_end, lx_m + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )[0]
                ah = -0.5 * sigh * sigh
                lx[d] += (
                    0.5 * (a0 + ah) * dt
                    + 0.25 * (sigp + sigm + 2.0 * sig0) * dw
                    + 0.25 * (sigp - sigm) * sqrt_dt * (zd * zd - 1.0)
                )

        max_lx = np.max(lx, axis=2, keepdims=True)
        ex_s = np.exp(lx - max_lx)
        mean_rep = np.maximum(np.mean(ex_s, axis=2, keepdims=True), 1e-30)
        lx -= max_lx + np.log(mean_rep)
        lx_period[m] = lx.reshape(d_assets, n_total)
        norm_x_out[m] = (ex_s / mean_rep).reshape(d_assets, n_total)
        t_prev = t_cur
        rt_prev = rt_cur
        rn_prev = rn_cur

    return lx_period, lx_period, norm_x_out


def _cpu_bwd_sde_aad(
    lx_ckpt: npt.NDArray[np.float64],
    lx_norm: npt.NDArray[np.float64],
    bar_lx_post: npt.NDArray[np.float64],
    lv_tables: npt.NDArray[np.float64],
    z_indep: npt.NDArray[np.float64],
    chol_base: npt.NDArray[np.float64],
    tau_obs_np: npt.NDArray[np.float64],
    r_native_obs_np: npt.NDArray[np.float64],
    r_tot_obs_np: npt.NDArray[np.float64],
    lv_t_grid: npt.NDArray[np.float64],
    lv_y_min: float | npt.NDArray[np.float64],
    lv_y_max: float | npt.NDArray[np.float64],
    n_sub: int,
    eff_r: int,
    n_rep: int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """CPU exact reverse-mode Platen Weak-Order-2.0 SDE adjoint with period-boundary rematerialization."""
    _ = lx_ckpt
    d_assets, lv_nt, lv_ny = lv_tables.shape
    m_rem = len(tau_obs_np)
    n_steps = m_rem * n_sub

    t_max = float(lv_t_grid[-1])
    inv_dt_grid = float(lv_nt - 1) / max(t_max, 1e-12)
    y_min_vec = np.broadcast_to(np.asarray(lv_y_min, dtype=np.float64), (d_assets,))
    y_max_vec = np.broadcast_to(np.asarray(lv_y_max, dtype=np.float64), (d_assets,))
    inv_dy_vec = float(lv_ny - 1) / np.maximum(y_max_vec - y_min_vec, 1e-12)
    max_u_t = float(lv_nt - 1) - 1e-6
    max_u_y = float(lv_ny - 1) - 1e-6

    z_corr = np.matmul(z_indep, chol_base.T)
    bar_lv = np.zeros_like(lv_tables)
    bar_chol = np.zeros_like(chol_base)
    bar_q_shift = np.zeros((d_assets, n_steps), dtype=np.float64)
    bar_x = np.zeros((d_assets, eff_r, n_rep), dtype=np.float64)

    for m in range(m_rem - 1, -1, -1):
        t_prev = 0.0 if m == 0 else float(tau_obs_np[m - 1])
        t_cur = float(tau_obs_np[m])
        rt_prev = np.zeros(d_assets) if m == 0 else r_tot_obs_np[:, m - 1]
        rt_cur = r_tot_obs_np[:, m]
        rn_prev = np.zeros(d_assets) if m == 0 else r_native_obs_np[:, m - 1]
        rn_cur = r_native_obs_np[:, m]
        d_tau = max(t_cur - t_prev, 1e-10)
        dt = d_tau / float(n_sub)
        sqrt_dt = math.sqrt(dt)
        drt = rt_cur - rt_prev
        drn = rn_cur - rn_prev

        bar_tot = bar_lx_post[m].reshape(d_assets, eff_r, n_rep) + bar_x
        omega = np.exp(lx_norm[m].reshape(d_assets, eff_r, n_rep)) / float(n_rep)
        bar_x = bar_tot - omega * np.sum(bar_tot, axis=2, keepdims=True)

        lx_start = (
            np.zeros((d_assets, eff_r, n_rep), dtype=np.float64)
            if m == 0
            else lx_norm[m - 1].reshape(d_assets, eff_r, n_rep).copy()
        )
        lx_sub_pre = np.empty((n_sub, d_assets, eff_r, n_rep), dtype=np.float64)
        lx_run = lx_start.copy()
        for s in range(n_sub):
            lx_sub_pre[s] = lx_run
            alpha_s = (float(s) + 0.5) / float(n_sub)
            t_start = max(t_prev + alpha_s * d_tau - 0.5 * dt, 0.0)
            t_end = t_start + dt
            q_shift_mid = (rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn)
            for d in range(d_assets):
                sig0 = _cpu_lv_bilinear_np(
                    lv_tables[d], t_start, lx_run[d] + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )[0]
                zd = z_corr[m, s, :, :, d]
                dw = sqrt_dt * zd
                a0 = -0.5 * sig0 * sig0
                sigh = _cpu_lv_bilinear_np(
                    lv_tables[d], t_end, lx_run[d] + a0 * dt + sig0 * dw + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )[0]
                sigp = _cpu_lv_bilinear_np(
                    lv_tables[d], t_end, lx_run[d] + a0 * dt + sig0 * sqrt_dt + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )[0]
                sigm = _cpu_lv_bilinear_np(
                    lv_tables[d], t_end, lx_run[d] + a0 * dt - sig0 * sqrt_dt + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )[0]
                lx_run[d] += (
                    0.5 * (a0 - 0.5 * sigh * sigh) * dt
                    + 0.25 * (sigp + sigm + 2.0 * sig0) * dw
                    + 0.25 * (sigp - sigm) * sqrt_dt * (zd * zd - 1.0)
                )

        for s in range(n_sub - 1, -1, -1):
            step_idx = m * n_sub + s
            lx_pre = lx_sub_pre[s]
            alpha_s = (float(s) + 0.5) / float(n_sub)
            t_start = max(t_prev + alpha_s * d_tau - 0.5 * dt, 0.0)
            t_end = t_start + dt
            q_shift_mid = (rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn)

            for d in range(d_assets):
                sig0, d0, it0_0, it1_0, wt_0, iy0_0, iy1_0, wy_0 = _cpu_lv_bilinear_np(
                    lv_tables[d], t_start, lx_pre[d] + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )
                zd = z_corr[m, s, :, :, d]
                dw = sqrt_dt * zd
                a0 = -0.5 * sig0 * sig0
                sigh, dh, it0_h, it1_h, wt_h, iy0_h, iy1_h, wy_h = _cpu_lv_bilinear_np(
                    lv_tables[d], t_end, lx_pre[d] + a0 * dt + sig0 * dw + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )
                sigp, dp, it0_p, it1_p, wt_p, iy0_p, iy1_p, wy_p = _cpu_lv_bilinear_np(
                    lv_tables[d], t_end, lx_pre[d] + a0 * dt + sig0 * sqrt_dt + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )
                sigm, dm, it0_m, it1_m, wt_m, iy0_m, iy1_m, wy_m = _cpu_lv_bilinear_np(
                    lv_tables[d], t_end, lx_pre[d] + a0 * dt - sig0 * sqrt_dt + q_shift_mid[d], inv_dt_grid, y_min_vec[d], inv_dy_vec[d], max_u_t, max_u_y
                )

                bx = bar_x[d]
                zc2m1 = zd * zd - 1.0
                bar_lvh = bx * (-0.5 * dt * sigh)
                bar_lvp = bx * (0.25 * (dw + sqrt_dt * zc2m1))
                bar_lvm = bx * (0.25 * (dw - sqrt_dt * zc2m1))
                bar_lv0_dir = bx * (-0.5 * dt * sig0 + 0.5 * dw)

                bar_lx_hat = bar_lvh * dh
                bar_lx_p = bar_lvp * dp
                bar_lx_m = bar_lvm * dm

                bar_zc = (
                    bx * (0.25 * (sigp + sigm + 2.0 * sig0) * sqrt_dt + 0.5 * (sigp - sigm) * sqrt_dt * zd)
                    + bar_lx_hat * (sig0 * sqrt_dt)
                )
                for j in range(d_assets):
                    bar_chol[d, j] += float(np.sum(bar_zc * z_indep[m, s, :, :, j]))

                bar_a0 = dt * (bar_lx_hat + bar_lx_p + bar_lx_m)
                bar_lv0 = bar_lv0_dir + bar_a0 * (-sig0) + bar_lx_hat * dw + (bar_lx_p - bar_lx_m) * sqrt_dt

                for bval, it0_v, it1_v, wt_v, iy0_v, iy1_v, wy_v in (
                    (bar_lv0, it0_0, it1_0, wt_0, iy0_0, iy1_0, wy_0),
                    (bar_lvh, it0_h, it1_h, wt_h, iy0_h, iy1_h, wy_h),
                    (bar_lvp, it0_p, it1_p, wt_p, iy0_p, iy1_p, wy_p),
                    (bar_lvm, it0_m, it1_m, wt_m, iy0_m, iy1_m, wy_m),
                ):
                    np.add.at(bar_lv[d, it0_v], iy0_v, bval * (1.0 - wt_v) * (1.0 - wy_v))
                    np.add.at(bar_lv[d, it0_v], iy1_v, bval * (1.0 - wt_v) * wy_v)
                    np.add.at(bar_lv[d, it1_v], iy0_v, bval * wt_v * (1.0 - wy_v))
                    np.add.at(bar_lv[d, it1_v], iy1_v, bval * wt_v * wy_v)

                bar_y = bar_lv0 * d0 + bar_lx_hat + bar_lx_p + bar_lx_m
                bar_q_shift[d, step_idx] += float(np.sum(bar_y))
                bar_x[d] = bx + bar_y

    return bar_lv, bar_chol, bar_q_shift


def _extract_fixing_value(
    fixings_obj: Any,
    ticker: str,
    target_dt: Any,
) -> Optional[float]:
    """Looks up a historical spot fixing for ticker at target_dt if present."""
    if fixings_obj is None:
        return None
    target_d = target_dt.date() if hasattr(target_dt, "date") else target_dt
    target_s = target_d.isoformat() if hasattr(target_d, "isoformat") else str(target_d)

    def _norm_dt(val_dt: Any) -> Any:
        if hasattr(val_dt, "date"):
            return val_dt.date()
        if isinstance(val_dt, str):
            try:
                return dt.date.fromisoformat(val_dt[:10])
            except (ValueError, TypeError):
                return val_dt
        return val_dt

    candidates: Sequence[Any] = ()
    if isinstance(fixings_obj, Mapping):
        if ticker in fixings_obj:
            entry_t = fixings_obj[ticker]
            if isinstance(entry_t, Mapping):
                if target_dt in entry_t:
                    return float(entry_t[target_dt])
                if target_d in entry_t:
                    return float(entry_t[target_d])
                if target_s in entry_t:
                    return float(entry_t[target_s])
                for fk, fv in entry_t.items():
                    if _norm_dt(fk) == target_d and fv is not None:
                        return float(fv)
                return None
            if isinstance(entry_t, (list, tuple)):
                candidates = entry_t
        elif target_dt in fixings_obj:
            v = fixings_obj[target_dt]
            return float(v) if isinstance(v, (int, float)) else None
        elif target_d in fixings_obj:
            v = fixings_obj[target_d]
            return float(v) if isinstance(v, (int, float)) else None
        elif target_s in fixings_obj:
            v = fixings_obj[target_s]
            return float(v) if isinstance(v, (int, float)) else None
    elif isinstance(fixings_obj, (list, tuple)):
        candidates = fixings_obj
    else:
        for attr in ("spot_fixings", "fixings", "EquityFixings", "SpotFixings", "values"):
            sub = getattr(fixings_obj, attr, None)
            if sub is not None and sub is not fixings_obj and not callable(sub):
                found = _extract_fixing_value(sub, ticker, target_dt)
                if found is not None:
                    return found
        return None

    for item in candidates:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            if len(item) == 2:
                d_cmp = _norm_dt(item[0])
                if d_cmp == target_d and item[1] is not None:
                    return float(item[1])
            elif len(item) >= 3 and str(item[0]) == ticker:
                d_cmp = _norm_dt(item[1])
                if d_cmp == target_d and item[2] is not None:
                    return float(item[2])
            continue
        item_id = getattr(item, "Identifier", getattr(item, "AssetId", getattr(item, "ticker", None)))
        if item_id is not None and str(item_id) != ticker:
            continue
        item_dt = getattr(item, "Date", getattr(item, "FixingDate", getattr(item, "date", None)))
        if item_dt is None:
            continue
        if _norm_dt(item_dt) == target_d:
            val = getattr(item, "Value", getattr(item, "FixingValue", getattr(item, "value", None)))
            if val is not None:
                return float(val)
    return None


def _extract_request_mkt_attr(request: Any, *names: str) -> Any:
    """Extracts a MarketDataSet attribute from either a Pydantic request object or a Mapping payload."""
    if request is None:
        return None
    if isinstance(request, Mapping):
        mdl = request.get("Model")
        mkt = (
            (mdl.get("MarketDataSet") if isinstance(mdl, Mapping) else getattr(mdl, "MarketDataSet", None))
            if mdl is not None
            else None
        )
        if mkt is None:
            mkt = request.get("MarketDataSet") or request.get("MarketData") or request.get("market_data")
        if isinstance(mkt, Mapping):
            for nm in names:
                if nm in mkt and mkt[nm] is not None:
                    return mkt[nm]
            return None
        if mkt is not None:
            for nm in names:
                val = getattr(mkt, nm, None)
                if val is not None:
                    return val
        return None
    mkt_o = getattr(getattr(request, "Model", None), "MarketDataSet", None)
    if mkt_o is None:
        return None
    d_mkt = getattr(mkt_o, "__dict__", None) if hasattr(type(mkt_o), "model_fields") else None
    extra_mkt = getattr(mkt_o, "__pydantic_extra__", None) if d_mkt is not None else None
    for nm in names:
        if isinstance(mkt_o, Mapping):
            if nm in mkt_o and mkt_o[nm] is not None:
                return mkt_o[nm]
        elif d_mkt is not None:
            val = d_mkt.get(nm)
            if val is None and extra_mkt:
                val = extra_mkt.get(nm)
            if val is not None:
                return val
        else:
            val = getattr(mkt_o, nm, None)
            if val is not None:
                return val
    return None


def _collect_unreferenced_bundle_curves(
    request: Any,
    seen_cids: set[str],
) -> List[Tuple[str, int]]:
    """Returns `(curve_id, n_pillars)` for curves in MarketDataSet.RatesCurvesBundles absent from `seen_cids`."""
    bundles = _extract_request_mkt_attr(request, "RatesCurvesBundles", "rates_curves_bundles")
    if not isinstance(bundles, Mapping):
        return []
    extra: List[Tuple[str, int]] = []
    for b_key, b_curve in bundles.items():
        cid_b = str(getattr(b_curve, "Identifier", getattr(b_curve, "identifier", b_key)) or b_key)
        if cid_b in seen_cids:
            continue
        n_pil = 0
        if hasattr(b_curve, "tenors"):
            n_pil = len(b_curve.tenors)
        elif hasattr(b_curve, "RatesSchedule"):
            rs = b_curve.RatesSchedule
            rates_seq = getattr(rs, "Rates", None)
            dates_seq = getattr(rs, "Dates", None)
            n_pil = len(rates_seq) if rates_seq else (len(dates_seq) if dates_seq else 0)
        elif isinstance(b_curve, Mapping) and "RatesSchedule" in b_curve:
            rs_m = b_curve["RatesSchedule"]
            if isinstance(rs_m, Mapping):
                n_pil = len(rs_m.get("Rates") or rs_m.get("Dates") or ())
        extra.append((cid_b, int(n_pil)))
        seen_cids.add(cid_b)
    return extra


def _dedup_gpu_tensors(obj: Any) -> Tuple[torch.Tensor, ...]:
    """Collects unique CUDA base-storage tensors from a nested container for compact LRU tracking."""
    seen_storage: set[Tuple[int, int]] = set()
    out: List[torch.Tensor] = []

    def _visit(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                st = x.untyped_storage()
                key = (int(st.data_ptr()), int(st.nbytes()))
                if key not in seen_storage:
                    seen_storage.add(key)
                    out.append(x)
            return
        if isinstance(x, (tuple, list)):
            for item in x:
                _visit(item)
        elif isinstance(x, dict):
            for item in x.values():
                _visit(item)

    _visit(obj)
    return tuple(out)


def _is_usable_repo_curve(c_obj: Any) -> bool:
    """True when `c_obj` (or its `_inner_curve`) exposes non-empty `.tenors` and callable `.df`."""
    if c_obj is None:
        return False
    eff = getattr(c_obj, "_inner_curve", None) or c_obj
    return bool(
        hasattr(eff, "tenors")
        and callable(getattr(eff, "df", None))
        and len(getattr(eff, "tenors", ())) > 0
    )


def _merge_usable_repo_curves(
    primary_map: Optional[Mapping[str, Any]],
    fallback_map: Optional[Mapping[str, Any]],
    as_of_dt: Any,
    normalize_fn: Callable[..., Any],
) -> Dict[str, Any]:
    """Merges extracted repo curves, preventing un-normalized specs from shadowing valid curves."""
    out: Dict[str, Any] = {}
    for src_map in (primary_map, fallback_map):
        if not src_map:
            continue
        for k_rc, v_rc in src_map.items():
            if k_rc in out or v_rc is None:
                continue
            v_norm = normalize_fn(v_rc, as_of_date=as_of_dt)
            if _is_usable_repo_curve(v_norm):
                out[k_rc] = v_norm
            elif _is_usable_repo_curve(v_rc):
                out[k_rc] = v_rc
    return out


def _raw_repo_sigs(
    state: Any,
    request: Any,
    sorted_ids: Sequence[str],
    sig_memo: Optional[Dict[int, Any]] = None,
) -> Tuple[Any, ...]:
    """Extracts deterministic content signatures for all repo curves referenced by request or state."""
    req_repos: Any = None
    if request is not None:
        if isinstance(request, Mapping):
            mkt_m = (
                request.get("MarketData")
                or request.get("market_data")
                or (request.get("Model", {}) if isinstance(request.get("Model"), Mapping) else {}).get("MarketDataSet")
            )
            if isinstance(mkt_m, Mapping):
                req_repos = (
                    mkt_m.get("RepoCurves")
                    or mkt_m.get("repo_curves")
                    or mkt_m.get("RepoSpreadCurves")
                    or mkt_m.get("repo_spread_curves")
                    or mkt_m.get("Repos")
                )
        else:
            mkt_o = getattr(getattr(request, "Model", None), "MarketDataSet", None)
            req_repos = getattr(mkt_o, "RepoSpreadCurves", None)
    st_repos = getattr(state, "repo_curves_map", None)
    ds_repos = getattr(state.dividend_schedule, "repo_curves", None)
    sigs: List[Any] = []
    req_map = req_repos if isinstance(req_repos, Mapping) else None
    st_map = st_repos if isinstance(st_repos, Mapping) else None
    ds_map = ds_repos if isinstance(ds_repos, Mapping) else None
    for tk in sorted_ids:
        c_req = req_map.get(tk) if req_map is not None else None
        c_st = st_map.get(tk) if st_map is not None else None
        c_ds = ds_map.get(tk) if ds_map is not None else None
        s_req = _curve_fast_sig(c_req, sig_memo) if c_req is not None else None
        s_st = s_req if (c_st is c_req) else (_curve_fast_sig(c_st, sig_memo) if c_st is not None else None)
        s_ds = s_st if (c_ds is c_st) else (s_req if (c_ds is c_req) else (_curve_fast_sig(c_ds, sig_memo) if c_ds is not None else None))
        sigs.append((s_req, s_st, s_ds))
    return tuple(sigs)


def _compute_mf_dvar_gbm_np(
    state: Any,
    sorted_ids: Sequence[str],
    ref_spots_np: npt.NDArray[np.float64],
    fwd_perf_np: npt.NDArray[np.float64],
    tau_obs_np: npt.NDArray[np.float64],
    base_lv_tables: npt.NDArray[np.float64],
    lv_t_grid: npt.NDArray[np.float64],
    lv_y_grids: Sequence[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float32]:
    """Builds coarse-period incremental GBM variance `(D, M)` shared by GPU and CPU MF-CV."""
    d_assets, m_rem = fwd_perf_np.shape
    lv_ny = int(base_lv_tables.shape[2])
    tau_prev_mf = np.concatenate([[0.0], tau_obs_np[:-1]])
    dt_mf = np.maximum(tau_obs_np - tau_prev_mf, 1e-10)
    t_mid_mf = 0.5 * (tau_prev_mf + tau_obs_np)
    d_var_gbm_np = np.empty((d_assets, m_rem), dtype=np.float32)
    for d_i, tk_i in enumerate(sorted_ids):
        surf_i = state.vol_surfaces.get(tk_i)
        if surf_i is not None and hasattr(surf_i, "iv_batch"):
            fwd_px_i = np.maximum(fwd_perf_np[d_i] * ref_spots_np[d_i], 1e-6)
            t_surf_i, _ = _map_t_eff_to_surf_np(surf_i, tau_obs_np)
            iv_fwd_i = np.asarray(surf_i.iv_batch(t_surf_i, fwd_px_i), dtype=np.float64)
            var_cum_i = (iv_fwd_i * iv_fwd_i) * t_surf_i
            var_run_i = 0.0
            dv_i = np.empty(m_rem, dtype=np.float64)
            for m_i in range(m_rem):
                dv_m = max(float(var_cum_i[m_i]) - var_run_i, 1e-12)
                dv_i[m_i] = dv_m
                var_run_i += dv_m
            d_var_gbm_np[d_i] = dv_i.astype(np.float32)
        else:
            y_g_i = lv_y_grids[d_i]
            k_y0 = int(np.searchsorted(y_g_i, 0.0))
            if k_y0 <= 0:
                sig_y0_t = base_lv_tables[d_i, :, 0]
            elif k_y0 >= lv_ny:
                sig_y0_t = base_lv_tables[d_i, :, lv_ny - 1]
            else:
                wy_0 = (0.0 - y_g_i[k_y0 - 1]) / max(y_g_i[k_y0] - y_g_i[k_y0 - 1], 1e-12)
                sig_y0_t = (1.0 - wy_0) * base_lv_tables[d_i, :, k_y0 - 1] + wy_0 * base_lv_tables[d_i, :, k_y0]
            sig_mid_i = np.interp(t_mid_mf, lv_t_grid, sig_y0_t)
            d_var_gbm_np[d_i] = ((sig_mid_i * sig_mid_i) * dt_mf).astype(np.float32)
    return d_var_gbm_np


if _HAS_NUMBA:
    @numba.njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
    def _nb_eval_mf_gbm_batch(
        z_corr_rmnd: npt.NDArray[np.float64],
        sqrt_dv: npt.NDArray[np.float64],
        d_var_gbm: npt.NDArray[np.float64],
        fwd_perf_np: npt.NDArray[np.float64],
        df_pay_np: npt.NDArray[np.float64],
        w_fund_arr: npt.NDArray[np.float64],
        has_w_fund: bool,
        ko_arr: npt.NDArray[np.float64],
        is_call_arr: npt.NDArray[np.float64],
        cpn_arr: npt.NDArray[np.float64],
        is_guar_arr: npt.NDArray[np.float64],
        cpn_rates_arr: npt.NDArray[np.float64],
        s1_out: npt.NDArray[np.float64],
        cf_surv_out: npt.NDArray[np.float64],
        call_code: int,
        cpn_code: int,
        opt_code: int,
        ko_spread: float,
        eff_csp: float,
        ki_spread: float,
        k_os: float,
        one_star_eps: float,
        k_put: float,
        k_low: float,
        b_ki: float,
        option_gearing: float,
        ac_cpn: float,
        c_mem_0: float,
        i_ki_past: float,
        k_call: float,
        part_call_rate: float,
        os_cpn: float,
        has_ki: bool,
        ki_active: bool,
        is_memory: bool,
        is_long_basket: bool,
        no_notional: bool,
        force_cpn_ac: bool,
        dir_down: bool,
        type_out: bool,
        has_lower_strike: bool,
        has_part_call: bool,
        has_os: bool,
        antithetic: bool,
        compute_cf_surv: bool,
    ) -> None:
        r_b, m_rem, n_rep, d_assets = z_corr_rmnd.shape
        n_signs = 2 if antithetic else 1
        wt_sign = 0.5 if antithetic else 1.0
        inv_n = 1.0 / float(n_rep)
        inv_d = 1.0 / float(d_assets)
        hard_ko = abs(ko_spread) <= 1e-6
        bl_k = min(0.0, ko_spread)
        bh_k = max(0.0, ko_spread)
        hard_cpn = abs(eff_csp) <= 1e-6
        hard_ki = abs(ki_spread) <= 1e-6
        b_ki_eff = k_put if is_long_basket else b_ki
        blk_ki = min(b_ki_eff, b_ki_eff + ki_spread)
        bhk_ki = max(b_ki_eff, b_ki_eff + ki_spread)
        inv_w_ki = 1.0 / max(bhk_ki - blk_ki, 1e-12)
        put_act = (k_put > k_low and option_gearing > 0.0) if has_lower_strike else (option_gearing > 0.0)
        df_last = float(df_pay_np[m_rem - 1])

        for r_idx in numba.prange(r_b):
            for s_idx in range(n_signs):
                sgn = 1.0 if s_idx == 0 else -1.0
                lx = np.zeros((d_assets, n_rep), dtype=np.float64)
                p_cur = np.empty((d_assets, n_rep), dtype=np.float64)
                surv = np.ones(n_rep, dtype=np.float64)
                cmem = np.full(n_rep, c_mem_0, dtype=np.float64)
                pv = np.zeros(n_rep, dtype=np.float64)

                for m in range(m_rem):
                    for d in range(d_assets):
                        sdv = sgn * sqrt_dv[d, m]
                        half_dv = 0.5 * d_var_gbm[d, m]
                        max_lx = -1e300
                        for n in range(n_rep):
                            v = lx[d, n] - half_dv + sdv * z_corr_rmnd[r_idx, m, n, d]
                            lx[d, n] = v
                            if v > max_lx:
                                max_lx = v
                        sum_ex = 0.0
                        for n in range(n_rep):
                            ex = math.exp(lx[d, n] - max_lx)
                            p_cur[d, n] = ex
                            sum_ex += ex
                        mean_ex = sum_ex * inv_n
                        if mean_ex < 1e-30:
                            mean_ex = 1e-30
                        shift = max_lx + math.log(mean_ex)
                        scale_fwd = fwd_perf_np[d, m] / mean_ex
                        for n in range(n_rep):
                            lx[d, n] -= shift
                            p_cur[d, n] *= scale_fwd

                    callable_m = is_call_arr[m] > 0.5
                    guar_m = is_guar_arr[m] > 0.5
                    bk = float(ko_arr[m])
                    bc = float(cpn_arr[m])
                    blc = min(bc, bc + eff_csp)
                    bhc = max(bc, bc + eff_csp)
                    inv_wc = 1.0 / max(bhc - blc, 1e-12)
                    inv_wk = 1.0 / max(bk + bh_k - (bk + bl_k), 1e-12)
                    cr = float(cpn_rates_arr[m])
                    wf_m = float(w_fund_arr[m]) if has_w_fund else 0.0
                    df_m = float(df_pay_np[m])

                    sum_pm = 0.0
                    sum_qc = 0.0
                    sum_ccf = 0.0
                    sum_surv_pre = 0.0
                    is_last = m == (m_rem - 1)
                    sum_uput = 0.0
                    sum_ucall = 0.0
                    sum_los = 0.0
                    sum_pos = 0.0
                    sum_tcf = 0.0
                    sum_pv = 0.0

                    for n in range(n_rep):
                        mn_p = p_cur[0, n]
                        mx_p = mn_p
                        sm_p = mn_p
                        for d in range(1, d_assets):
                            pv_d = p_cur[d, n]
                            sm_p += pv_d
                            if pv_d < mn_p:
                                mn_p = pv_d
                            if pv_d > mx_p:
                                mx_p = pv_d
                        avg_p = sm_p * inv_d
                        wc_v = mn_p if call_code == 0 else (mx_p if call_code == 1 else avg_p)
                        wcp_v = mn_p if cpn_code == 0 else (mx_p if cpn_code == 1 else avg_p)

                        if callable_m:
                            if hard_ko:
                                gk = 1.0 if wc_v >= bk else 0.0
                            else:
                                uk = (wc_v - (bk + bl_k)) * inv_wk
                                gk = 0.0 if uk < 0.0 else (1.0 if uk > 1.0 else uk)
                        else:
                            gk = 0.0

                        if guar_m:
                            gc = 1.0
                        else:
                            if hard_cpn:
                                hd = (wcp_v <= bc) if dir_down else (wcp_v >= bc)
                                hd_f = 1.0 if hd else 0.0
                                gc = (1.0 - hd_f) if type_out else hd_f
                            else:
                                uc = (wcp_v - blc) * inv_wc
                                ru = 0.0 if uc < 0.0 else (1.0 if uc > 1.0 else uc)
                                hg = (1.0 - ru) if dir_down else ru
                                gc = (1.0 - hg) if type_out else hg

                        egc = gk if (force_cpn_ac and gk > gc) else gc
                        sv_n = surv[n]
                        pm = sv_n * gk
                        qc = sv_n * egc
                        if is_memory:
                            ec = cr + cmem[n]
                            ccf = qc * ec + pm * ac_cpn
                            cmem[n] = sv_n * ec * (1.0 - egc)
                        else:
                            ccf = qc * cr + pm * ac_cpn

                        cf_m = ccf + (0.0 if no_notional else pm)
                        pv_n = pv[n] + df_m * cf_m - wf_m * sv_n
                        sum_pm += pm
                        sum_qc += qc
                        sum_ccf += ccf
                        sum_surv_pre += sv_n
                        rem_s = sv_n - pm
                        if rem_s < 0.0:
                            rem_s = 0.0
                        surv[n] = rem_s

                        if not is_last:
                            pv[n] = pv_n
                        else:
                            wo_v = mn_p if opt_code == 0 else (mx_p if opt_code == 1 else avg_p)
                            if ki_active:
                                if hard_ki:
                                    g_ki = 1.0 if wo_v < b_ki_eff else 0.0
                                else:
                                    uki = (bhk_ki - wo_v) * inv_w_ki
                                    g_ki = 0.0 if uki < 0.0 else (1.0 if uki > 1.0 else uki)
                                ki_ind = i_ki_past if i_ki_past > g_ki else g_ki
                            else:
                                ki_ind = 1.0 if (not has_ki) else i_ki_past

                            if has_os:
                                if one_star_eps <= 1e-6:
                                    star_ind = 1.0 if mx_p >= k_os else 0.0
                                else:
                                    uos = (mx_p - (k_os - 0.5 * one_star_eps)) / one_star_eps
                                    star_ind = 0.0 if uos < 0.0 else (1.0 if uos > 1.0 else uos)
                            else:
                                star_ind = 0.0

                            if is_long_basket:
                                u_put = rem_s * ki_ind * (1.0 - star_ind) * (1.0 - avg_p)
                                put_term = -u_put
                            elif put_act:
                                dk_hi = k_put - wo_v
                                p_hi = dk_hi if dk_hi > 0.0 else 0.0
                                if has_lower_strike:
                                    dk_lo = k_low - wo_v
                                    p_lo = dk_lo if dk_lo > 0.0 else 0.0
                                else:
                                    p_lo = 0.0
                                u_put = rem_s * ki_ind * (1.0 - star_ind) * (p_hi - p_lo)
                                put_term = -option_gearing * u_put
                            else:
                                u_put = 0.0
                                put_term = 0.0

                            if has_part_call:
                                dc = wo_v - k_call
                                u_call = rem_s * (dc if dc > 0.0 else 0.0)
                            else:
                                u_call = 0.0
                            p_os = (rem_s * star_ind) if has_os else 0.0
                            l_os = (p_os * os_cpn) if (has_os and os_cpn != 0.0) else 0.0
                            opt_term = put_term + (part_call_rate * u_call if has_part_call else 0.0) + l_os
                            term_cf = opt_term if no_notional else (rem_s + opt_term)
                            pv_n += df_last * term_cf
                            sum_uput += u_put
                            sum_ucall += u_call
                            sum_los += l_os
                            sum_pos += p_os
                            sum_tcf += term_cf
                            sum_pv += pv_n

                    s1_out[r_idx, m] += wt_sign * (sum_pm * inv_n)
                    s1_out[r_idx, m_rem + m] += wt_sign * (sum_qc * inv_n)
                    s1_out[r_idx, 2 * m_rem + m] += wt_sign * (sum_ccf * inv_n)
                    if compute_cf_surv:
                        t_cf_m = sum_ccf + (0.0 if no_notional else sum_pm)
                        cf_surv_out[r_idx, 0, m] += wt_sign * (t_cf_m * inv_n)
                        cf_surv_out[r_idx, 1, m] += wt_sign * (sum_surv_pre * inv_n)
                    if is_last:
                        c0 = 3 * m_rem
                        s1_out[r_idx, c0 + 0] += wt_sign * (sum_uput * inv_n)
                        s1_out[r_idx, c0 + 1] += wt_sign * (sum_ucall * inv_n)
                        s1_out[r_idx, c0 + 2] += wt_sign * (sum_los * inv_n)
                        s1_out[r_idx, c0 + 3] += wt_sign * (sum_pos * inv_n)
                        s1_out[r_idx, c0 + 4] += wt_sign * (sum_pv * inv_n)
                        if compute_cf_surv:
                            cf_surv_out[r_idx, 0, m] += wt_sign * (sum_tcf * inv_n)


def _eval_mf_gbm_batch_cpu(
    z_corr_rmnd: npt.NDArray[np.float64],
    d_var_gbm: npt.NDArray[np.float64],
    fwd_perf_np: npt.NDArray[np.float64],
    df_pay_np: npt.NDArray[np.float64],
    w_fund_np: Optional[npt.NDArray[np.float64]],
    ko_arr: npt.NDArray[np.float64],
    is_call_arr: npt.NDArray[np.float64],
    cpn_arr: npt.NDArray[np.float64],
    is_guar_arr: npt.NDArray[np.float64],
    cpn_rates_arr: npt.NDArray[np.float64],
    inv_perm: Sequence[int],
    *,
    call_code: int,
    cpn_code: int,
    opt_code: int,
    ko_spread: float,
    cpn_spread: float,
    ki_spread: float,
    k_os: float,
    one_star_eps: float,
    k_put: float,
    k_low: float,
    b_ki: float,
    option_gearing: float,
    ac_cpn: float,
    c_mem_0: float,
    i_ki_past: float,
    k_call: float,
    part_call_rate: float,
    os_cpn: float,
    has_ki: bool,
    is_memory: bool,
    is_long_basket: bool,
    no_notional: bool,
    force_cpn_ac: bool,
    dir_down: bool,
    type_out: bool,
    has_lower_strike: bool,
    has_part_call: bool,
    antithetic: bool = False,
    compute_cf_surv: bool = False,
) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
    """Vectorized CPU Multifidelity-GBM payoff estimator returning Stage-1 replicate means matching GPU MF-CV."""
    r_b, m_rem, n_rep, d_assets = z_corr_rmnd.shape
    s1_out = np.zeros((r_b, 3 * m_rem + 5), dtype=np.float64)
    cf_surv_out = np.zeros((r_b, 2, m_rem), dtype=np.float64) if compute_cf_surv else None
    sqrt_dv = np.ascontiguousarray(np.sqrt(np.maximum(d_var_gbm, 1e-18)), dtype=np.float64)
    eff_csp = -cpn_spread if dir_down else cpn_spread
    has_os = bool(k_os > 0.0)
    ki_active = bool(has_ki and b_ki > 0.0 and i_ki_past < 0.5)
    if _HAS_NUMBA and call_code in (0, 1, 2) and cpn_code in (0, 1, 2) and opt_code in (0, 1, 2):
        cf_nb = cf_surv_out if cf_surv_out is not None else np.zeros((1, 2, 1), dtype=np.float64)
        wf_nb = (
            np.ascontiguousarray(w_fund_np, dtype=np.float64)
            if w_fund_np is not None
            else np.zeros(m_rem, dtype=np.float64)
        )
        _nb_eval_mf_gbm_batch(
            np.ascontiguousarray(z_corr_rmnd, dtype=np.float64),
            sqrt_dv,
            np.ascontiguousarray(d_var_gbm, dtype=np.float64),
            np.ascontiguousarray(fwd_perf_np, dtype=np.float64),
            np.ascontiguousarray(df_pay_np, dtype=np.float64),
            wf_nb,
            w_fund_np is not None,
            np.ascontiguousarray(ko_arr, dtype=np.float64),
            np.ascontiguousarray(is_call_arr, dtype=np.float64),
            np.ascontiguousarray(cpn_arr, dtype=np.float64),
            np.ascontiguousarray(is_guar_arr, dtype=np.float64),
            np.ascontiguousarray(cpn_rates_arr, dtype=np.float64),
            s1_out,
            cf_nb,
            int(call_code),
            int(cpn_code),
            int(opt_code),
            float(ko_spread),
            float(eff_csp),
            float(ki_spread),
            float(k_os),
            float(one_star_eps),
            float(k_put),
            float(k_low),
            float(b_ki),
            float(option_gearing),
            float(ac_cpn),
            float(c_mem_0),
            float(i_ki_past),
            float(k_call),
            float(part_call_rate),
            float(os_cpn),
            bool(has_ki),
            bool(ki_active),
            bool(is_memory),
            bool(is_long_basket),
            bool(no_notional),
            bool(force_cpn_ac),
            bool(dir_down),
            bool(type_out),
            bool(has_lower_strike),
            bool(has_part_call),
            bool(has_os),
            bool(antithetic),
            bool(compute_cf_surv),
        )
        return s1_out, cf_surv_out

    def _idx_from_perf(p_arr: npt.NDArray[np.float64], code: int) -> npt.NDArray[np.float64]:
        if code == 0:
            return np.asarray(np.min(p_arr, axis=2), dtype=np.float64)
        if code == 1:
            return np.asarray(np.max(p_arr, axis=2), dtype=np.float64)
        if code == 2:
            return np.asarray(np.mean(p_arr, axis=2), dtype=np.float64)
        if code == 3:
            a0, a1 = int(inv_perm[0]), int(inv_perm[min(1, d_assets - 1)])
            return np.asarray(0.5 * (p_arr[:, :, a0, :] + p_arr[:, :, a1, :]), dtype=np.float64)
        return np.asarray(np.min(p_arr, axis=2), dtype=np.float64)

    signs = (1.0, -1.0) if antithetic else (1.0,)
    wt_sign = 0.5 if antithetic else 1.0
    sqrt_dv = np.sqrt(np.maximum(d_var_gbm, 1e-18))  # (D, M)
    eff_csp = -cpn_spread if dir_down else cpn_spread
    has_os = bool(k_os > 0.0)
    ki_active = bool(has_ki and b_ki > 0.0 and i_ki_past < 0.5)

    for sgn in signs:
        lx = np.zeros((r_b, d_assets, n_rep), dtype=np.float64)
        perf = np.empty((r_b, m_rem, d_assets, n_rep), dtype=np.float64)
        for m in range(m_rem):
            sdv_m = sqrt_dv[:, m][None, :, None]
            dv_m = d_var_gbm[:, m][None, :, None]
            z_m = np.transpose(z_corr_rmnd[:, m, :, :], (0, 2, 1))  # (R, D, N)
            lx = lx - 0.5 * dv_m + (sgn * sdv_m) * z_m
            max_lx = np.max(lx, axis=2, keepdims=True)
            ex_s = np.exp(lx - max_lx)
            mean_ex = np.maximum(np.mean(ex_s, axis=2, keepdims=True), 1e-30)
            lx = lx - (max_lx + np.log(mean_ex))
            perf[:, m, :, :] = (ex_s / mean_ex) * fwd_perf_np[:, m][None, :, None]

        wc = _idx_from_perf(perf, call_code)
        wcp = wc if cpn_code == call_code else _idx_from_perf(perf, cpn_code)
        wo = wc if opt_code == call_code else (wcp if opt_code == cpn_code else _idx_from_perf(perf, opt_code))
        wavg_last = _idx_from_perf(perf[:, m_rem - 1 : m_rem], 2)[:, 0, :] if is_long_basket else None
        wb_last = _idx_from_perf(perf[:, m_rem - 1 : m_rem], 1)[:, 0, :] if has_os else None

        surv = np.ones((r_b, n_rep), dtype=np.float64)
        cmem = np.full((r_b, n_rep), c_mem_0, dtype=np.float64)
        pv = np.zeros((r_b, n_rep), dtype=np.float64)

        for m in range(m_rem):
            if is_call_arr[m] > 0.5:
                bk = float(ko_arr[m])
                if abs(ko_spread) <= 1e-6:
                    gk = (wc[:, m, :] >= bk).astype(np.float64)
                else:
                    bl, bh = min(bk, bk + ko_spread), max(bk, bk + ko_spread)
                    gk = np.clip((wc[:, m, :] - bl) / max(bh - bl, 1e-12), 0.0, 1.0)
            else:
                gk = np.zeros((r_b, n_rep), dtype=np.float64)

            if is_guar_arr[m] > 0.5:
                gc = np.ones((r_b, n_rep), dtype=np.float64)
            else:
                bc = float(cpn_arr[m])
                if abs(eff_csp) <= 1e-6:
                    hd = (wcp[:, m, :] <= bc) if dir_down else (wcp[:, m, :] >= bc)
                    gc = (1.0 - hd.astype(np.float64)) if type_out else hd.astype(np.float64)
                else:
                    blc, bhc = min(bc, bc + eff_csp), max(bc, bc + eff_csp)
                    ru = np.clip((wcp[:, m, :] - blc) / max(bhc - blc, 1e-12), 0.0, 1.0)
                    hg = (1.0 - ru) if dir_down else ru
                    gc = (1.0 - hg) if type_out else hg

            egc = np.maximum(gc, gk) if force_cpn_ac else gc
            pm = surv * gk
            qc = surv * egc
            cr = float(cpn_rates_arr[m])
            if is_memory:
                ec = cr + cmem
                ccf = qc * ec + pm * ac_cpn
                cmem = surv * ec * (1.0 - egc)
            else:
                ccf = qc * cr + pm * ac_cpn

            wf_m = float(w_fund_np[m]) if w_fund_np is not None else 0.0
            df_m = float(df_pay_np[m])
            pv = pv + df_m * (ccf + (0.0 if no_notional else pm)) - wf_m * surv
            if cf_surv_out is not None:
                cf_surv_out[:, 0, m] += wt_sign * np.mean(ccf + (0.0 if no_notional else pm), axis=1)
                cf_surv_out[:, 1, m] += wt_sign * np.mean(surv, axis=1)
            s1_out[:, m] += wt_sign * np.mean(pm, axis=1)
            s1_out[:, m_rem + m] += wt_sign * np.mean(qc, axis=1)
            s1_out[:, 2 * m_rem + m] += wt_sign * np.mean(ccf, axis=1)
            surv = np.maximum(surv - pm, 0.0)

        wo_last = wo[:, m_rem - 1, :]
        if ki_active:
            b_ki_eff = k_put if is_long_basket else b_ki
            if abs(ki_spread) <= 1e-6:
                g_ki = (wo_last < b_ki_eff).astype(np.float64)
            else:
                blk, bhk = min(b_ki_eff, b_ki_eff + ki_spread), max(b_ki_eff, b_ki_eff + ki_spread)
                g_ki = np.clip((bhk - wo_last) / max(bhk - blk, 1e-12), 0.0, 1.0)
            ki_ind = np.maximum(i_ki_past, g_ki)
        else:
            ki_ind = np.ones((r_b, n_rep), dtype=np.float64) if not has_ki else np.full((r_b, n_rep), i_ki_past)

        if wb_last is not None:
            if one_star_eps <= 1e-6:
                star_ind = (wb_last >= k_os).astype(np.float64)
            else:
                star_ind = np.clip((wb_last - (k_os - 0.5 * one_star_eps)) / one_star_eps, 0.0, 1.0)
        else:
            star_ind = np.zeros((r_b, n_rep), dtype=np.float64)

        if is_long_basket:
            assert wavg_last is not None
            u_put = surv * ki_ind * (1.0 - star_ind) * (1.0 - wavg_last)
            put_term = -u_put
        else:
            put_act = (k_put > k_low and option_gearing > 0.0) if has_lower_strike else (option_gearing > 0.0)
            if put_act:
                p_lo = np.maximum(k_low - wo_last, 0.0) if has_lower_strike else 0.0
                u_put = surv * ki_ind * (1.0 - star_ind) * (np.maximum(k_put - wo_last, 0.0) - p_lo)
                put_term = -option_gearing * u_put
            else:
                u_put = np.zeros((r_b, n_rep), dtype=np.float64)
                put_term = u_put

        u_call = surv * np.maximum(wo_last - k_call, 0.0) if has_part_call else np.zeros((r_b, n_rep), dtype=np.float64)
        p_os = surv * star_ind if has_os else np.zeros((r_b, n_rep), dtype=np.float64)
        l_os = p_os * os_cpn if (has_os and os_cpn != 0.0) else np.zeros((r_b, n_rep), dtype=np.float64)
        opt_term = put_term + (part_call_rate * u_call if has_part_call else 0.0) + l_os
        term_cf = opt_term if no_notional else (surv + opt_term)
        pv = pv + float(df_pay_np[m_rem - 1]) * term_cf
        if cf_surv_out is not None:
            cf_surv_out[:, 0, m_rem - 1] += wt_sign * np.mean(term_cf, axis=1)

        c0 = 3 * m_rem
        s1_out[:, c0 + 0] += wt_sign * np.mean(u_put, axis=1)
        s1_out[:, c0 + 1] += wt_sign * np.mean(u_call, axis=1)
        s1_out[:, c0 + 2] += wt_sign * np.mean(l_os, axis=1)
        s1_out[:, c0 + 3] += wt_sign * np.mean(p_os, axis=1)
        s1_out[:, c0 + 4] += wt_sign * np.mean(pv, axis=1)

    return s1_out, cf_surv_out


def compute_qmc_aad_crn_greeks(
    state: Any,
    request: Any = None,
    *,
    queries: Optional[Sequence[str]] = None,
    config: Optional[QmcAadGreeksConfig] = None,
    rho_unit_convention: Literal["quote", "decimal"] = "quote",
    theta_roll_days: int = 1,
    theta_convention: Literal["calendar_1d", "murex_bizday_forward_fixed"] = "murex_bizday_forward_fixed",
    device: Optional[torch.device | str] = None,
    return_cashflows: bool = True,
) -> Any:
    """Computes Tier-4 QMC CRN + Pathwise-AAD All-8 Autocall Greeks returning AutocallGreeksResult."""
    from deepvol.models.autocall.dupire_qmc_solver import _generate_sobol_normals
    from deepvol.models.autocall.greeks_engine import (
        AutocallGreeksResult,
        _collapse_crossed_periods_for_roll,
        _compute_funding_weights_per_period,
        _curve_df_cached_np,
        _curve_df_delta_jacobian,
        _escrowed_spot_softplus_np,
        _extract_repo_curves_map,
        _normalize_curve_obj,
        _resolve_quanto_fx_sign,
        _rolled_in_transit_pv,
    )
    from deepvol.models.autocall.seasoned_collapser import _year_fraction

    cfg = config or QmcAadGreeksConfig()
    out_device = state.spots_t0.device

    d_assets = int(state.d_assets)
    m_rem = int(state.m_rem)
    n_replicates_cfg = max(1, int(getattr(cfg, "n_replicates", 4)))
    n_paths_cfg = max(2 * n_replicates_cfg, int(getattr(cfg, "n_paths", 8192)))
    eff_r = n_replicates_cfg
    n_rep = _next_power_of_two_int(max(2, n_paths_cfg // eff_r))
    if n_rep < 2:
        n_rep = 2
    n_total = eff_r * n_rep
    if device is not None:
        qmc_device = torch.device(device)
    else:
        qmc_device = out_device
    if qmc_device.type == "cuda" and qmc_device.index is None and torch.cuda.is_available():
        qmc_device = torch.device("cuda", torch.cuda.current_device())
    n_sub = max(1, int(getattr(cfg, "n_substeps", 4)))
    lv_nt = min(max(8, int(getattr(cfg, "lv_grid_nt", 32))), 64)
    lv_ny = min(max(16, int(getattr(cfg, "lv_grid_ny", 64))), 128)
    h_spot_rel = float(getattr(cfg, "h_spot_rel", 0.01))
    h_rho = float(getattr(cfg, "h_rho", 1.0e-4))
    one_star_eps = float(getattr(cfg, "one_star_eps", 0.02))
    seed = int(getattr(cfg, "seed", 42))
    barrier_eps_min = float(getattr(cfg, "barrier_eps_min", 0.005))
    barrier_eps_max = float(getattr(cfg, "barrier_eps_max", 0.02))
    raw_ki_obs = getattr(state, "ki_obs_type", None)
    cfg_ki_mon = getattr(cfg, "ki_monitoring_type", None)
    if raw_ki_obs is not None and str(raw_ki_obs).strip():
        ki_obs_low = str(raw_ki_obs).strip().lower()
    elif cfg_ki_mon is not None:
        ki_obs_low = str(cfg_ki_mon).strip().lower()
    else:
        ki_obs_low = "european"
    if ki_obs_low in ("continuous", "cont"):
        ki_monitoring_type = "Continuous"
    elif ki_obs_low == "daily":
        ki_monitoring_type = "Daily"
    elif ki_obs_low == "discrete":
        ki_monitoring_type = "Discrete"
    else:
        ki_monitoring_type = "European"

    raw_queries = queries if queries is not None else state.queries
    q_set = {str(getattr(q, "value", q)) for q in raw_queries}

    want_delta = "Delta" in q_set
    want_gamma = "Gamma" in q_set
    want_cash_gamma = "MultivariateAggregatedCashGamma" in q_set
    want_vega = ("Vega" in q_set) or ("VegaGrid" in q_set)
    want_rho = "Rho" in q_set
    want_divs = "DividendsSensitivities" in q_set
    want_theta = "Theta" in q_set
    want_cega = "Cega" in q_set or "ConvexCega" in q_set

    orig_ids = list(state.original_identifiers)
    sorted_ids = list(state.sorted_identifiers)
    inv_perm = list(state.inv_lex_perm)

    use_gpu_triton = qmc_device.type == "cuda" and _HAS_TRITON
    engine_label = "qmc_aad_triton" if use_gpu_triton else "qmc_aad_cpp"

    pv0 = 0.0
    base_cfs: list[Any] = []
    base_se = 0.0
    base_postproc: Any = None

    as_of_orig = state.seasoned_state.as_of_date
    disc_curve = _normalize_curve_obj(state.discount_curve, as_of_date=as_of_orig)
    disc_id = str(getattr(disc_curve, "identifier", state.discount_curve_id or "DEFAULT"))

    delta_dict: dict[str, float] | None = None
    delta_tensor_orig: torch.Tensor | None = None
    gamma_dict: dict[str, dict[str, float]] | None = None
    gamma_tensor_orig: torch.Tensor | None = None
    cash_gamma_agg: float | None = None
    vega_dict: dict[str, list[list[float]]] | None = None
    vega_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    fx_vega_dict: dict[str, list[float]] | None = None
    rho_dict: dict[str, list[float]] | None = None
    rho_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    repo_rho_dict: dict[str, list[float]] | None = None
    divs_dict: dict[str, list[float]] | None = None
    divs_arrays: dict[str, npt.NDArray[np.float64]] | None = None
    cega_dict: dict[str, dict[str, float]] | None = None
    cega_tensor_orig: torch.Tensor | None = None
    fx_cega_dict: dict[str, dict[str, float]] | None = None
    theta_val: float | None = None

    def _roll_as_of_for_theta(base_dt: Any) -> Tuple[Any, float]:
        roll_n = max(1, int(theta_roll_days))
        if not isinstance(base_dt, dt.datetime):
            return None, float(roll_n) / 365.0
        if theta_convention == "murex_bizday_forward_fixed":
            cur = base_dt
            rem = roll_n
            while rem > 0:
                cur = cur + dt.timedelta(days=1)
                if cur.weekday() < 5:
                    rem -= 1
            return cur, max(float((cur - base_dt).days), 1.0) / 365.0
        cur = base_dt + dt.timedelta(days=roll_n)
        return cur, float(roll_n) / 365.0

    def _curve_quote_scale(c_obj: Any) -> float:
        return max(float(getattr(c_obj, "quote_scale", 0.01)), 1e-12)

    if m_rem == 0 or bool(state.already_autocalled):
        z_m = torch.zeros(m_rem, dtype=torch.float64, device=out_device)
        z_s = torch.tensor(0.0, dtype=torch.float64, device=out_device)
        base_postproc = assemble_autocall_cashflows_and_pv(
            state,
            p_stop=z_m,
            q_cpn=z_m,
            coupon_cf_undisc=np.zeros(m_rem, dtype=np.float64),
            l_put=z_s,
            l_call=z_s,
            l_onestar=0.0,
            p_onestar=0.0,
            option_se=0.0,
            total_se=0.0,
            build_cashflow_items=bool(return_cashflows),
        )
        pv0 = float(base_postproc.product_value.detach().cpu().item())
        base_cfs = list(base_postproc.cashflows)
        base_se = float(base_postproc.standard_error)

        delta_tensor_orig = torch.zeros(d_assets, dtype=torch.float64, device=out_device) if want_delta else None
        delta_dict = {orig_ids[i]: 0.0 for i in range(d_assets)} if want_delta else None
        gamma_tensor_orig = torch.zeros((d_assets, d_assets), dtype=torch.float64, device=out_device) if want_gamma else None
        gamma_dict = (
            {orig_ids[i]: {orig_ids[j]: 0.0 for j in range(d_assets)} for i in range(d_assets)}
            if want_gamma
            else None
        )
        cash_gamma_agg = 0.0 if want_cash_gamma else None
        if want_vega:
            vega_arrays = {
                tk: np.zeros(
                    (
                        state.vol_surfaces[tk].n_exp if tk in state.vol_surfaces else 0,
                        state.vol_surfaces[tk].n_raw_strikes if tk in state.vol_surfaces else 0,
                    ),
                    dtype=np.float64,
                )
                for tk in orig_ids
            }
            vega_dict = {tk: vega_arrays[tk].tolist() for tk in orig_ids}
        if want_rho:
            k_pil = len(disc_curve.tenors) if (disc_curve is not None and hasattr(disc_curve, "tenors")) else 0
            rho_disc_arr = np.zeros(k_pil, dtype=np.float64)
            if k_pil > 0 and disc_curve is not None and hasattr(disc_curve, "df"):
                in_tr_dts_0: List[Any] = []
                in_tr_amts_0: List[float] = []
                est_cfs_0 = [
                    cf
                    for cf in getattr(state.seasoned_state, "past_cashflows", ())
                    if getattr(cf, "StatusText", "") == "Estimate"
                ]
                if est_cfs_0:
                    for cf in est_cfs_0:
                        if abs(float(cf.Amount)) > 0.0:
                            in_tr_dts_0.append(cf.Date)
                            in_tr_amts_0.append(float(cf.Amount))
                elif getattr(state.seasoned_state, "in_transit_indices", None):
                    for rec in state.seasoned_state.historical_records:
                        if rec.status == "In-Transit" and (rec.coupon_paid != 0.0 or rec.notional_paid != 0.0):
                            in_tr_dts_0.append(rec.payment_date)
                            in_tr_amts_0.append(float(rec.discount_factor * (rec.coupon_paid + rec.notional_paid)))
                if in_tr_dts_0:
                    jac_it_0 = (
                        _curve_df_delta_jacobian(
                            disc_curve,
                            in_tr_dts_0,
                            k_pil,
                            torch.device("cpu"),
                            as_of_date=as_of_orig,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    rho_disc_arr = np.asarray(
                        jac_it_0.T @ np.asarray(in_tr_amts_0, dtype=np.float64),
                        dtype=np.float64,
                    )
                    if rho_unit_convention == "decimal":
                        rho_disc_arr = rho_disc_arr / _curve_quote_scale(disc_curve)
            rho_arrays = {disc_id: rho_disc_arr}
            rho_dict = {disc_id: [float(x) for x in rho_disc_arr]}
            seen_z: set[str] = {disc_id}
            for cid_b, n_pil_b in _collect_unreferenced_bundle_curves(request, seen_z):
                rho_arrays[cid_b] = np.zeros(n_pil_b, dtype=np.float64)
                rho_dict[cid_b] = [0.0] * n_pil_b
            repo_src = request if request is not None else state
            repo_curves_z = _merge_usable_repo_curves(
                _extract_repo_curves_map(repo_src, sorted_ids, as_of_orig),
                getattr(state, "repo_curves_map", None),
                as_of_orig,
                _normalize_curve_obj,
            )
            if repo_curves_z:
                repo_rho_dict = {}
                for tk in sorted_ids:
                    rc_z = repo_curves_z.get(tk)
                    if rc_z is not None:
                        rn_z = _normalize_curve_obj(rc_z, as_of_date=as_of_orig)
                        re_z = getattr(rn_z, "_inner_curve", None) or rn_z
                        n_pz = len(re_z.tenors) if (re_z is not None and hasattr(re_z, "tenors")) else 0
                        if n_pz > 0:
                            repo_rho_dict[tk] = [0.0] * n_pz
        if want_divs:
            divs_arrays = {
                tk: np.zeros(len(state.dividend_schedule.dividend_events.get(tk, [])), dtype=np.float64)
                for tk in orig_ids
            }
            divs_dict = {tk: divs_arrays[tk].tolist() for tk in orig_ids}
        cega_tensor_orig = torch.zeros((d_assets, d_assets), dtype=torch.float64, device=out_device) if want_cega else None
        cega_dict = (
            {orig_ids[i]: {orig_ids[j]: 0.0 for j in range(d_assets)} for i in range(d_assets)}
            if want_cega
            else None
        )
        if want_theta:
            as_of_roll_0, _ = _roll_as_of_for_theta(as_of_orig)
            if as_of_roll_0 is not None:
                in_tr_rolled_0 = float(
                    _rolled_in_transit_pv(state, disc_curve, as_of_roll_0, None, out_device).detach().cpu().item()
                )
                theta_val = float(in_tr_rolled_0 - float(state.pv_in_transit))
            else:
                theta_val = 0.0
        return AutocallGreeksResult(
            product_value=pv0,
            cash_flows=base_cfs,
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
            fx_vega=fx_vega_dict,
            fx_cega=fx_cega_dict,
            repo_rho=repo_rho_dict,
            reporting_currency=state.payment_currency,
            standard_error=base_se,
            cash_to_settle_today=float(state.cash_to_settle_today),
            postproc_output=base_postproc,
            greeks_engine_used=engine_label,
        )

    tau_obs_np = _np_f64(state.tau_obs)
    spots_np = _np_f64(state.spots_t0)
    ref_spots_np = _np_f64(state.ref_spots)
    div_pv_np = _np_f64(state.dividend_schedule.div_pv)
    fwd_perf_np = _np_f64(state.forward_perf)
    df_pay_np = _np_f64(state.df_pay)
    corr_np = _np_f64(state.corr_matrix)
    quanto_meta = getattr(state, "quanto_meta", None) or {}
    surf_dig_map: Dict[str, str] = {
        tk: _surface_content_digest(state.vol_surfaces[tk]) for tk in sorted_ids if tk in state.vol_surfaces
    }
    tau_obs_dig_base: Any = tau_obs_np.tobytes()

    r_tot_obs_np = _np_f64(state.dividend_schedule.r_tot_obs)
    r_native_attr = getattr(state.dividend_schedule, "r_native_obs", None)
    call_idx_str = str(state.callability_index)
    cpn_idx_str = str(state.coupon_index or call_idx_str)
    is_long_flag = bool(getattr(state, "is_long_basket", False))
    raw_opt_idx = state.option_index
    opt_idx_str = str(raw_opt_idx or call_idx_str)
    call_code = _resolve_internal_index_code(call_idx_str)
    cpn_code = _resolve_internal_index_code(cpn_idx_str)
    opt_code = _resolve_internal_index_code(opt_idx_str)
    ko_spread = float(state.ko_spread)
    cpn_spread = float(state.cpn_spread)
    ki_spread = float(state.ki_spread)
    has_ki = state.ki_barrier is not None and float(state.ki_barrier) > 0.0
    b_ki = float(state.ki_barrier) if has_ki else 0.0
    type_out = str(state.coupon_barrier_type).strip().lower() == "out"
    dir_down = str(state.coupon_barrier_direction).strip().lower() == "down"
    is_memory = bool(state.is_memory)
    force_cpn_ac = bool(state.force_coupons_on_autocall)
    no_notional = bool(state.no_notional_repayment)
    has_lower_strike_flag = state.lower_strike is not None
    k_put = float(state.strike)
    k_low = float(state.lower_strike) if has_lower_strike_flag else 0.0
    k_call = float(state.participation_call_strike) if state.participation_call_strike is not None else 0.0
    part_call_rate = float(state.participation_call_rate)
    has_part_call_flag = state.participation_call_strike is not None and k_call >= 0.0 and part_call_rate != 0.0
    raw_k_os = float(state.one_star_strike) if state.one_star_strike is not None else 0.0
    os_cpn = float(state.one_star_coupon)
    if raw_k_os > 0.0 and has_ki and ko_spread < -1e-6:
        one_star_eps = abs(float(ko_spread))
        k_os = raw_k_os + 0.5 * float(ko_spread)
    else:
        k_os = raw_k_os
    is_cont_ki_active = bool(
        state.ki_barrier is not None
        and float(state.ki_barrier) > 0.0
        and float(state.i_ki_past) < 0.5
        and ki_monitoring_type == "Continuous"
    )
    cpp_sde_v2_fn, cpp_sob_v2_fn, cpp_vib_v2_fn = _get_cpp_crn_v2_fns()
    cpp_step_sp_v2_fn = _get_cpp_step_spot_bumps_fn() if (not use_gpu_triton) else None
    use_vibrato_gamma = bool(
        (use_gpu_triton or (cpp_sde_v2_fn is not None and cpp_vib_v2_fn is not None))
        and (want_gamma or want_cash_gamma)
        and (ki_monitoring_type == "European" or not has_ki or float(state.i_ki_past) >= 0.5)
    )
    need_spot_bumps = bool(
        want_delta
        or (use_vibrato_gamma and m_rem > 1)
        or ((want_gamma or want_cash_gamma) and not use_vibrato_gamma)
    )
    eff_opt_code_s2 = opt_code
    use_triton_aad_s2 = bool(
        use_gpu_triton
        and not is_cont_ki_active
        and call_code in _TRITON_PAYOFF_AAD_CODES
        and cpn_code in _TRITON_PAYOFF_AAD_CODES
        and eff_opt_code_s2 in _TRITON_PAYOFF_AAD_CODES
    )
    want_rate_aad_smooth = bool(
        abs(ko_spread) > 1e-6
        and abs(cpn_spread) > 1e-6
        and ((not has_ki) or b_ki <= 0.0 or float(state.i_ki_past) >= 0.5 or abs(ki_spread) > 1e-6)
        and np.all(fwd_perf_np > 1e-12)
    )
    need_stage2_aad = bool(
        want_vega
        or (want_cega and d_assets > 1)
        or (
            (want_rho or want_divs or want_theta)
            and want_rate_aad_smooth
            and (use_triton_aad_s2 or (not use_gpu_triton and cpp_sde_v2_fn is not None))
        )
    )
    use_pathwise_rate_adj = bool(
        need_stage2_aad
        and (use_triton_aad_s2 or (not use_gpu_triton and cpp_sde_v2_fn is not None))
        and want_rate_aad_smooth
    )
    skip_step1_loo = bool(
        use_gpu_triton
        and need_stage2_aad
        and not need_spot_bumps
        and not want_rho
        and not want_divs
        and not want_theta
        and (use_pathwise_rate_adj or (not (want_vega and quanto_meta) and not (want_cega and quanto_meta)))
    )
    use_f32_step1 = bool(use_gpu_triton and n_rep <= 2048)
    emit_stage1_gpu = bool(
        need_stage2_aad or ("ProductValue" in q_set) or ("CashFlows" in q_set) or return_cashflows
    )
    cv_mode_cfg = str(getattr(cfg, "cv_mode", "multifidelity_gbm")).strip().lower()
    dev_norm_str = str(normalize_cuda_device(qmc_device)) if use_gpu_triton else "cpu"
    cfg_y_min = getattr(cfg, "lv_y_min", None)
    cfg_y_max = getattr(cfg, "lv_y_max", None)

    ko_src = _np_f64(state.ko_barriers)
    ic_src = _np_f64(state.is_callable)
    cp_src = _np_f64(state.cpn_barriers)
    ig_src = _np_f64(state.is_guaranteed_cpn)
    cr_src = _np_f64(state.coupon_amounts)
    payoff_s_np = np.array(
        [
            ko_spread,
            cpn_spread,
            ki_spread,
            k_os,
            one_star_eps,
            k_put,
            k_low,
            b_ki,
            float(state.option_gearing),
            float(state.autocall_coupon),
            float(state.c_mem_0),
            float(state.i_ki_past),
            k_call,
            part_call_rate,
            os_cpn,
        ],
        dtype=np.float64,
    )

    spots_roll_sig: Any = None
    if want_theta:
        as_of_rolled_sig, _ = _roll_as_of_for_theta(as_of_orig)
        if (
            as_of_rolled_sig is not None
            and state.seasoned_state.future_obs_dates
            and state.seasoned_state.future_obs_dates[0] <= as_of_rolled_sig
        ):
            fix_sig_obj = _extract_request_mkt_attr(request, "Fixings", "fixings")
            obs0_sig_dt = state.seasoned_state.future_obs_dates[0]
            spots_roll_sig = tuple(_extract_fixing_value(fix_sig_obj, tk, obs0_sig_dt) for tk in sorted_ids)

    curve_sig_memo: Dict[int, Any] = {}

    def _csig(c_obj_s: Any) -> Any:
        if c_obj_s is None:
            return None
        cid = id(c_obj_s)
        hit = curve_sig_memo.get(cid)
        if hit is not None:
            return hit
        val = _curve_fast_sig(c_obj_s, curve_sig_memo)
        curve_sig_memo[cid] = val
        return val

    def _funding_leg_sig_local() -> Any:
        if state.funding_leg is None:
            return None
        fl_s = state.funding_leg
        ff_raw = fl_s.get("funding_fixings") or ()
        ff_sig = (
            tuple(sorted((str(fk), float(fv)) for fk, fv in ff_raw.items()))
            if isinstance(ff_raw, Mapping)
            else tuple(ff_raw)
        )
        return (
            _csig(fl_s.get("funding_curve")),
            fl_s.get("funding_curve_id"),
            fl_s.get("rate_identifier"),
            float(fl_s.get("funding_spread") or 0.0),
            tuple(fl_s.get("payment_dates") or ()),
            ff_sig,
            state.seasoned_state.start_date,
            state.seasoned_state.day_count_convention,
            tuple(state.seasoned_state.future_pay_dates),
        )

    def _quanto_meta_sig_local() -> Any:
        if not quanto_meta:
            return ()
        return tuple(
            (
                tk,
                float(qm_s.get("rho_s_fx", 0.0)),
                str(qm_s.get("fx_id", "")),
                qm_s.get("fx_sign", None),
                qm_s.get("fx_quote_convention", None),
                tuple(float(xv) for xv in (qm_s.get("sig_fx_obs") or ())),
            )
            for tk in sorted_ids
            for qm_s in (quanto_meta.get(tk),)
            if qm_s
        )

    def _div_sched_sig_local() -> Tuple[Any, ...]:
        ds_obj = state.dividend_schedule
        ds_snap = getattr(ds_obj, "_crn_div_sched_sig_snap", None)
        t_ids = tuple(sorted_ids)
        if ds_snap is not None and ds_snap[0] == t_ids:
            _, s_tuple, ev_snap, arr_snaps = ds_snap
            ok_ds = True
            for idx_d, tk in enumerate(sorted_ids):
                evs_s = ds_obj.dividend_events.get(tk, ())
                if tuple((ev.ex_date, ev.payment_date) for ev in evs_s) != ev_snap[idx_d]:
                    ok_ds = False
                    break
                if evs_s:
                    a0, a1, a2 = arr_snaps[idx_d]
                    if not (
                        _arr_bytes_equal(_np_f64(ds_obj.dividend_amount_tensors[tk]), a0)
                        and _arr_bytes_equal(_np_f64(ds_obj.dividend_pv_weights[tk]), a1)
                        and _arr_bytes_equal(_np_f64(ds_obj.dividend_obs_incidence[tk]), a2)
                    ):
                        ok_ds = False
                        break
            if ok_ds:
                return cast(Tuple[Any, ...], s_tuple)
        d_parts: List[Any] = []
        ev_recs: List[Any] = []
        arr_recs: List[Any] = []
        for tk in sorted_ids:
            evs_s = ds_obj.dividend_events.get(tk, ())
            ev_t = tuple((ev.ex_date, ev.payment_date) for ev in evs_s)
            ev_recs.append(ev_t)
            if evs_s:
                a0 = _np_f64(ds_obj.dividend_amount_tensors[tk])
                a1 = _np_f64(ds_obj.dividend_pv_weights[tk])
                a2 = _np_f64(ds_obj.dividend_obs_incidence[tk])
                arr_recs.append((a0.copy(), a1.copy(), a2.copy()))
                d_parts.append((ev_t, a0.tobytes(), a1.tobytes(), a2.tobytes()))
            else:
                arr_recs.append(None)
                d_parts.append(None)
        res_ds = tuple(d_parts)
        try:
            setattr(ds_obj, "_crn_div_sched_sig_snap", (t_ids, res_ds, tuple(ev_recs), tuple(arr_recs)))
        except (AttributeError, TypeError):
            pass
        return res_ds

    def _past_itr_sig_local() -> Tuple[Any, Any]:
        past_cfs_s = getattr(state.seasoned_state, "past_cashflows", ())
        est_sig_l = tuple(
            (cf.Date, float(cf.Amount))
            for cf in past_cfs_s
            if getattr(cf, "StatusText", "") == "Estimate" and abs(float(cf.Amount)) > 0.0
        )
        itr_rec_sig_l = (
            tuple(
                (rec.payment_date, float(rec.discount_factor), float(rec.coupon_paid), float(rec.notional_paid))
                for rec in state.seasoned_state.historical_records
                if rec.status == "In-Transit" and (rec.coupon_paid != 0.0 or rec.notional_paid != 0.0)
            )
            if (not est_sig_l and state.seasoned_state.in_transit_indices)
            else ()
        )
        return est_sig_l, itr_rec_sig_l

    warm_bundle_key: Any = None
    _warm_cpu: Optional[_CrnCpuPrep] = None
    _warm_gpu: Optional[_CrnGpuViews] = None
    _warm_gpu_hit = False
    if use_gpu_triton and not isinstance(request, str):
        v_perf_sig_t = getattr(state, "v_perf", None)
        v_term_sig_b = (
            _np_f64(v_perf_sig_t[:, -1]).tobytes()
            if (cfg_y_min is None or cfg_y_max is None) and v_perf_sig_t is not None and v_perf_sig_t.numel() >= d_assets
            else b""
        )
        r_nat_sig_b = _np_f64(r_native_attr).tobytes() if r_native_attr is not None else b""
        f_leg_sig = _funding_leg_sig_local()
        disc_sig = _csig(state.discount_curve) if (want_rho or want_theta or state.funding_leg is not None) else None
        q_meta_sig = _quanto_meta_sig_local()
        rho_th_sig: Any = None
        if want_rho or want_theta:
            est_sig, itr_rec_sig = _past_itr_sig_local()
            unref_sig: Any = None
            if want_rho:
                b_map = _extract_request_mkt_attr(request, "RatesCurvesBundles", "rates_curves_bundles")
                if isinstance(b_map, Mapping):
                    unref_sig = tuple((str(bk), _csig(bc)) for bk, bc in b_map.items())
            rho_th_sig = (
                tuple(state.seasoned_state.future_obs_dates),
                tuple(state.seasoned_state.future_pay_dates),
                tuple(_csig((state.asset_financing_curves or {}).get(tk)) for tk in sorted_ids),
                _raw_repo_sigs(state, request, sorted_ids, curve_sig_memo),
                est_sig,
                itr_rec_sig,
                unref_sig,
                _np_f64(state.tau_pay).tobytes() if (want_theta and disc_curve is None) else b"",
                _np_f64(state.dividend_schedule.repo_spread_avg).tobytes() if want_theta else b"",
                spots_roll_sig,
            )
        div_sig: Any = _div_sched_sig_local() if (want_rho or want_divs or want_theta) else None
        raw_n_cheap_wb = getattr(cfg, "n_cheap_mf", None)
        warm_bundle_key = (
            "warm_bundle_v2",
            dev_norm_str,
            str(out_device),
            d_assets,
            m_rem,
            eff_r,
            n_rep,
            n_sub,
            lv_nt,
            lv_ny,
            h_spot_rel,
            h_rho,
            one_star_eps,
            seed,
            barrier_eps_min,
            barrier_eps_max,
            ki_monitoring_type,
            rho_unit_convention,
            int(theta_roll_days),
            theta_convention,
            cv_mode_cfg,
            int(raw_n_cheap_wb) if raw_n_cheap_wb is not None else 65536,
            cfg_y_min,
            cfg_y_max,
            use_f32_step1,
            os.environ.get("DEEPVOL_MF_EMA", "1"),
            os.environ.get("DEEPVOL_L2O_F32", "1"),
            os.environ.get("DEEPVOL_AAD_F32", "1"),
            os.environ.get("DEEPVOL_VIB_F32", "1"),
            want_delta,
            want_gamma,
            want_cash_gamma,
            want_vega,
            want_rho,
            want_divs,
            want_theta,
            want_cega,
            emit_stage1_gpu,
            tuple(orig_ids),
            tuple(sorted_ids),
            tuple(inv_perm),
            as_of_orig,
            state.payment_currency,
            disc_id,
            call_code,
            cpn_code,
            opt_code,
            has_ki,
            type_out,
            dir_down,
            is_memory,
            force_cpn_ac,
            no_notional,
            has_lower_strike_flag,
            has_part_call_flag,
            is_long_flag,
            is_cont_ki_active,
            float(state.pv_in_transit),
            tau_obs_dig_base,
            spots_np.tobytes(),
            ref_spots_np.tobytes(),
            div_pv_np.tobytes(),
            fwd_perf_np.tobytes(),
            df_pay_np.tobytes(),
            corr_np.tobytes(),
            r_tot_obs_np.tobytes(),
            r_nat_sig_b,
            ko_src.tobytes(),
            ic_src.tobytes(),
            cp_src.tobytes(),
            ig_src.tobytes(),
            cr_src.tobytes(),
            payoff_s_np.tobytes(),
            v_term_sig_b,
            tuple(surf_dig_map.get(tk, "") for tk in sorted_ids),
            f_leg_sig,
            disc_sig,
            q_meta_sig,
            rho_th_sig,
            div_sig,
        )
        warm_cached = _CRN_DEAL_PREP_CACHE.get(warm_bundle_key)
        if warm_cached is not None:
            _warm_cpu, _warm_gpu, _ = warm_cached
            _warm_gpu_hit = True

    vol_atm_obs_cache: Dict[int, npt.NDArray[np.float64]] = {}

    def _get_vol_atm_obs(i_idx: int) -> npt.NDArray[np.float64]:
        cached_v = vol_atm_obs_cache.get(i_idx)
        if cached_v is not None:
            return cached_v
        tk_s = sorted_ids[i_idx]
        s_dig = surf_dig_map.get(tk_s) or _surface_content_digest(state.vol_surfaces[tk_s])
        iv_key = ("iv_atm", s_dig, tau_obs_dig_base, float(spots_np[i_idx]))
        v_cached = _CRN_VEGA_JAC_CACHE.get(iv_key)
        if v_cached is not None:
            v_arr = cast(npt.NDArray[np.float64], v_cached)
        else:
            v_arr = np.asarray(
                state.vol_surfaces[tk_s].iv_batch(tau_obs_np, np.full(m_rem, spots_np[i_idx])),
                dtype=np.float64,
            )
            _CRN_VEGA_JAC_CACHE.put(iv_key, v_arr)
        vol_atm_obs_cache[i_idx] = v_arr
        return v_arr

    div_np_cache: Dict[str, Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}

    def _get_div_np(tk_s: str) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        cached_d = div_np_cache.get(tk_s)
        if cached_d is not None:
            return cached_d
        res_d = (
            _np_f64(state.dividend_schedule.dividend_amount_tensors[tk_s]),
            _np_f64(state.dividend_schedule.dividend_pv_weights[tk_s]),
            _np_f64(state.dividend_schedule.dividend_obs_incidence[tk_s]),
        )
        div_np_cache[tk_s] = res_d
        return res_d

    z_indep_full_f32: Optional[torch.Tensor] = None
    sv_sobol_gpu: Optional[torch.Tensor] = None
    shifts_sobol_gpu: Optional[torch.Tensor] = None
    u_rot_gpu: Optional[torch.Tensor] = None
    v_rot_gpu: Optional[torch.Tensor] = None
    z_indep_np: Optional[npt.NDArray[np.float64]] = None
    w_fund_base_np: Optional[npt.NDArray[np.float64]] = None
    esc_base: npt.NDArray[np.float64] = np.empty((0, 0), dtype=np.float64)
    growth_base: npt.NDArray[np.float64] = np.empty((0, 0), dtype=np.float64)
    base_lv_tables: npt.NDArray[np.float64] = np.empty((0, 0, 0), dtype=np.float64)
    sp_p_lv_list: List[npt.NDArray[np.float64]] = []
    sp_m_lv_list: List[npt.NDArray[np.float64]] = []
    lv_y_grids: List[npt.NDArray[np.float64]] = []
    h_spots_np = np.maximum(spots_np * h_spot_rel, 1e-4)

    dim_coarse = m_rem * d_assets
    dim_fine = m_rem * n_sub * d_assets
    n_dims_total = dim_coarse + dim_fine
    can_triton_sobol = bool(
        use_gpu_triton and n_dims_total <= 21201 and n_rep > 0 and (n_rep & (n_rep - 1)) == 0
    )
    sobol_key = (eff_r, n_rep, m_rem, n_sub, d_assets, seed, "gpu" if use_gpu_triton else "cpu")
    rot_dig: Any = tau_obs_dig_base
    rot_sobol_key = ("rot_z", sobol_key, rot_dig, dev_norm_str)

    if _warm_gpu_hit:
        assert _warm_cpu is not None and _warm_gpu is not None
        r_native_obs_np = _warm_cpu.r_native_obs_np
        w_fund_base_np = _warm_cpu.w_fund_base_np
        u_time_np = _warm_cpu.u_time_np
        chol_base = _warm_cpu.chol_base
        v_eig_np = _warm_cpu.v_eig_np
        sqrt_ev_np = _warm_cpu.sqrt_ev_np
        lv_y_min_vec = _warm_cpu.lv_y_min_vec
        lv_y_max_vec = _warm_cpu.lv_y_max_vec
        t_max = _warm_cpu.t_max
        lv_t_grid = _warm_cpu.lv_t_grid
        base_lv_tables = _warm_cpu.base_lv_tables
        base_lv_caches = _warm_cpu.base_lv_caches
        ko_arr_np = _warm_cpu.ko_arr_np
        cpn_arr_np = _warm_cpu.cpn_arr_np
        is_call_np = _warm_cpu.is_call_np
        is_guar_np = _warm_cpu.is_guar_np
        cpn_rates_np = _warm_cpu.cpn_rates_np
        u_rot_gpu = _warm_gpu.u_rot_gpu
        v_rot_gpu = _warm_gpu.v_rot_gpu
    else:
        if r_native_attr is not None:
            r_native_obs_np = _np_f64(r_native_attr)
        else:
            # qm["rho_s_fx"] already stores rho_quoted * fx_sign from escrowed_dividend; apply once.
            r_native_obs_np = r_tot_obs_np.copy()
            for i, ticker in enumerate(sorted_ids):
                qm = quanto_meta.get(ticker)
                if qm and ticker in state.vol_surfaces:
                    rho_live = float(qm.get("rho_s_fx", 0.0))
                    sig_fx_q = np.asarray(qm.get("sig_fx_obs", [0.0] * m_rem), dtype=np.float64)
                    sig_eq_q = _get_vol_atm_obs(i)
                    r_native_obs_np[i, :] = r_tot_obs_np[i, :] + (rho_live * sig_eq_q * sig_fx_q) * tau_obs_np

        esc_base = _escrowed_spot_softplus_np(spots_np[:, None], div_pv_np)
        growth_base = (fwd_perf_np * ref_spots_np[:, None]) / np.maximum(esc_base, 1e-12)
        if state.funding_leg is not None:
            w_fund_cpu = (
                _compute_funding_weights_per_period(
                    state,
                    period_start_idx=0,
                    device=torch.device("cpu") if use_gpu_triton else out_device,
                )
                .detach()
                .to(dtype=torch.float64)
            )
            w_fund_base_np = _np_f64(w_fund_cpu)
        u_time_np, chol_base, v_eig_np, sqrt_ev_np = _sym_sqrt_and_kl_pca_np(
            tau_obs_np,
            corr_np,
            corr_aad_mode=str(getattr(cfg, "corr_aad_mode", "cholesky")),
        )
        if use_gpu_triton:
            u_rot_gpu = torch.as_tensor(u_time_np, dtype=torch.float32, device=qmc_device)
            v_rot_gpu = torch.as_tensor(v_eig_np, dtype=torch.float32, device=qmc_device)

        if cfg_y_min is not None and cfg_y_max is not None:
            lv_y_min_vec = np.full(d_assets, float(cfg_y_min), dtype=np.float64)
            lv_y_max_vec = np.full(d_assets, float(cfg_y_max), dtype=np.float64)
        else:
            v_perf_t = getattr(state, "v_perf", None)
            if v_perf_t is not None and v_perf_t.numel() >= d_assets:
                v_term_np = _np_f64(v_perf_t[:, -1])
            else:
                v_term_np = np.full(d_assets, 0.25, dtype=np.float64)
            half_w_vec = np.clip(5.0 * np.abs(v_term_np), 0.50, 6.0)
            lv_y_min_vec = -half_w_vec
            lv_y_max_vec = half_w_vec
        t_max = max(float(tau_obs_np[-1]), 1e-4)
        lv_t_grid = np.linspace(1e-5, t_max, lv_nt, dtype=np.float64)
        lv_y_grids = [
            np.linspace(float(lv_y_min_vec[i]), float(lv_y_max_vec[i]), lv_ny, dtype=np.float64)
            for i in range(d_assets)
        ]

        base_lv_tables = np.empty((d_assets, lv_nt, lv_ny), dtype=np.float64)
        base_lv_caches = [None] * d_assets
        grid_t_dig_base: Any = (tau_obs_dig_base, float(t_max), lv_nt)
        for i, ticker in enumerate(sorted_ids):
            surf_i = state.vol_surfaces[ticker]
            surf_dig_i = surf_dig_map[ticker]
            grid_dig_i: Any = (grid_t_dig_base, float(lv_y_min_vec[i]), float(lv_y_max_vec[i]), lv_ny)
            rn_div_dig_i: Any = (r_native_obs_np[i].tobytes(), div_pv_np[i].tobytes())
            entry_i = _build_lv_table_and_forward_cache(
                surf_i,
                float(spots_np[i]),
                tau_obs_np,
                r_native_obs_np[i],
                lv_t_grid,
                lv_y_grids[i],
                div_pv_np[i],
                grid_dig=grid_dig_i,
                surf_dig=surf_dig_i,
                rn_div_dig=rn_div_dig_i,
                need_vega_jac=want_vega,
            )
            base_lv_caches[i] = entry_i
            base_lv_tables[i] = entry_i["lv_out"]
            if need_spot_bumps:
                sp_p_i = float(spots_np[i] + h_spots_np[i])
                sp_m_i = float(max(spots_np[i] - h_spots_np[i], 1e-6))
                sp_p_lv_list.append(
                    _build_lv_table_and_forward_cache(
                        surf_i, sp_p_i, tau_obs_np, r_native_obs_np[i], lv_t_grid, lv_y_grids[i], div_pv_np[i],
                        grid_dig=grid_dig_i, surf_dig=surf_dig_i, rn_div_dig=rn_div_dig_i, need_vega_jac=False,
                    )["lv_out"]
                )
                sp_m_lv_list.append(
                    _build_lv_table_and_forward_cache(
                        surf_i, sp_m_i, tau_obs_np, r_native_obs_np[i], lv_t_grid, lv_y_grids[i], div_pv_np[i],
                        grid_dig=grid_dig_i, surf_dig=surf_dig_i, rn_div_dig=rn_div_dig_i, need_vega_jac=False,
                    )["lv_out"]
                )

        if (not use_gpu_triton) or want_theta:
            ko_arr_np = np.ascontiguousarray(ko_src, dtype=np.float64)
            cpn_arr_np = np.ascontiguousarray(cp_src, dtype=np.float64)
            is_call_np = np.ascontiguousarray(ic_src, dtype=np.float64)
            is_guar_np = np.ascontiguousarray(ig_src, dtype=np.float64)
            cpn_rates_np = np.ascontiguousarray(cr_src, dtype=np.float64)
        else:
            empty_f64 = np.empty(0, dtype=np.float64)
            ko_arr_np = empty_f64
            cpn_arr_np = empty_f64
            is_call_np = empty_f64
            is_guar_np = empty_f64
            cpn_rates_np = empty_f64

    def _gen_sobol_cpu_slice(r_start_s: int, r_end_s: int) -> npt.NDArray[np.float64]:
        r_cnt_s = r_end_s - r_start_s
        n_dims_cpu = m_rem * (n_sub + 1) * d_assets
        if (
            cpp_sob_v2_fn is not None
            and n_dims_cpu <= 21201
            and n_rep > 0
            and (n_rep & (n_rep - 1)) == 0
        ):
            sv_cpu_np = np.ascontiguousarray(
                _get_sobol_direction_numbers(torch.device("cpu")).numpy(), dtype=np.uint32
            )
            rng_c = np.random.default_rng(seed)
            rng_f = np.random.default_rng(seed + 31337)
            shifts_c_full = rng_c.integers(0, 1 << 30, size=(eff_r, dim_coarse), dtype=np.uint32)
            shifts_f_full = rng_f.integers(0, 1 << 30, size=(eff_r, dim_fine), dtype=np.uint32)
            shifts_sub = np.ascontiguousarray(
                np.concatenate([shifts_c_full[r_start_s:r_end_s], shifts_f_full[r_start_s:r_end_s]], axis=1),
                dtype=np.uint32,
            )
            if callable(getattr(cpp_sob_v2_fn, "__call__", None)) and not hasattr(cpp_sob_v2_fn, "argtypes"):
                z_out_c, _ = cpp_sob_v2_fn(
                    d_assets=d_assets,
                    m_rem=m_rem,
                    n_sub=n_sub,
                    eff_r=r_cnt_s,
                    n_rep=n_rep,
                    dir_numbers=sv_cpu_np,
                    digital_shifts=shifts_sub,
                    n_bits=30,
                    use_gray_code=True,
                    apply_bb_time=False,
                    apply_corr=False,
                    antithetic=False,
                    return_indep=True,
                    return_corr=False,
                )
                assert z_out_c is not None
                return cast(npt.NDArray[np.float64], z_out_c)
            z_out_raw = np.empty((m_rem, n_sub, r_cnt_s, n_rep, d_assets), dtype=np.float64)
            sob_args = _SobolGenArgsC(
                d_assets=d_assets,
                m_rem=m_rem,
                n_sub=n_sub,
                eff_r=r_cnt_s,
                n_rep=n_rep,
                sobol_dim=int(sv_cpu_np.shape[0]),
                n_bits=30,
                dir_numbers=sv_cpu_np.ctypes.data,
                digital_shifts=shifts_sub.ctypes.data,
                bb_ortho_mat=0,
                chol_mat=0,
                z_indep_out=z_out_raw.ctypes.data,
                z_corr_out=0,
                use_gray_code=1,
                apply_bb_time=0,
                apply_corr=0,
                antithetic=0,
            )
            rc_sob = cpp_sob_v2_fn(ctypes.byref(sob_args))
            if rc_sob == 0:
                return z_out_raw
        z_reps_s = np.empty((r_cnt_s, n_rep, m_rem, n_sub, d_assets), dtype=np.float64)
        inv_sqrt_sub_s = 1.0 / math.sqrt(float(n_sub))
        for loc_r, r_idx in enumerate(range(r_start_s, r_end_s)):
            zc = _generate_sobol_normals(n_rep, dim_coarse, scramble=True, seed=seed + 7919 * r_idx).reshape(
                n_rep, m_rem, 1, d_assets
            )
            if n_sub == 1:
                z_reps_s[loc_r] = zc
            else:
                zf = _generate_sobol_normals(n_rep, dim_fine, scramble=True, seed=seed + 31337 + 7919 * r_idx).reshape(
                    n_rep, m_rem, n_sub, d_assets
                )
                z_reps_s[loc_r] = zf + (zc * inv_sqrt_sub_s - np.mean(zf, axis=2, keepdims=True))
        return np.ascontiguousarray(z_reps_s.transpose(2, 3, 0, 1, 4), dtype=np.float64)

    if use_gpu_triton:
        z_full_bytes = m_rem * n_sub * d_assets * n_total * 4
        if z_full_bytes <= 18 * 1024 * 1024:
            cached_z_rot = _CRN_SOBOL_CACHE.get(rot_sobol_key)
            if isinstance(cached_z_rot, torch.Tensor) and cached_z_rot.device == qmc_device:
                z_indep_full_f32 = cached_z_rot
        if can_triton_sobol:
            sv_sobol_gpu = _get_sobol_direction_numbers(qmc_device)
            shift_cache_key = ("sobol_shifts", eff_r, m_rem, n_sub, d_assets, seed, dev_norm_str)
            shifts_sobol_gpu = _CRN_SOBOL_CACHE.get(shift_cache_key)
            if shifts_sobol_gpu is None:
                rng_c = np.random.default_rng(seed)
                rng_f = np.random.default_rng(seed + 31337)
                shifts_c = rng_c.integers(0, 1 << 30, size=(eff_r, dim_coarse), dtype=np.int32)
                shifts_f = rng_f.integers(0, 1 << 30, size=(eff_r, dim_fine), dtype=np.int32)
                shifts_np = np.concatenate([shifts_c, shifts_f], axis=1)
                shifts_sobol_gpu = torch.from_numpy(shifts_np).to(device=qmc_device, dtype=torch.int32)
                _CRN_SOBOL_CACHE.put(shift_cache_key, shifts_sobol_gpu)

        def _get_z_indep_chunk_gpu(
            r_start_c: int,
            r_end_c: int,
            allow_full_cache: bool = True,
            *,
            shifts_ov: Optional[torch.Tensor] = None,
            u_rot_ov: Optional[torch.Tensor] = None,
            m_pad_ov: Optional[int] = None,
            d_pad_ov: Optional[int] = None,
            cg_flags_ov: Optional[torch.Tensor] = None,
            use_rt_flags: bool = False,
            use_cg_arena: bool = False,
        ) -> torch.Tensor:
            eff_m_z = int(m_pad_ov) if m_pad_ov is not None else m_rem
            eff_d_z = int(d_pad_ov) if d_pad_ov is not None else d_assets
            eff_sh_z = shifts_ov if shifts_ov is not None else shifts_sobol_gpu
            eff_urot_z = u_rot_ov if u_rot_ov is not None else u_rot_gpu
            eff_rt_z = bool(use_rt_flags and cg_flags_ov is not None and d_pad_ov is not None)
            if (not eff_rt_z) and allow_full_cache and z_indep_full_f32 is not None:
                if r_start_c == 0 and r_end_c == eff_r:
                    return z_indep_full_f32
                return z_indep_full_f32[:, :, r_start_c * n_rep : r_end_c * n_rep].contiguous()
            r_cnt_c = r_end_c - r_start_c
            n_tot_c = r_cnt_c * n_rep
            if can_triton_sobol and sv_sobol_gpu is not None and eff_sh_z is not None:
                z_raw_c = _alloc_cg_buf(
                    use_cg_arena,
                    qmc_device,
                    "z_raw_or_bzc",
                    (eff_m_z * n_sub, eff_d_z, n_tot_c),
                    torch.float32,
                    zero=eff_rt_z,
                )
                m_pow_r = int(round(math.log2(n_rep)))
                block_n_s = min(256, n_rep)
                n_tiles_r_s = (n_rep + block_n_s - 1) // block_n_s
                _fast_triton_launch(
                    _sobol_bridge_normals_kernel,
                    (n_tiles_r_s, eff_m_z * r_cnt_c),
                    sv_sobol_gpu,
                    eff_sh_z[r_start_c:r_end_c],
                    z_raw_c,
                    eff_m_z,
                    eff_d_z,
                    cg_flags_ptr=cg_flags_ov if eff_rt_z else None,
                    n_rep=n_rep,
                    n_sub=n_sub,
                    m_pow=m_pow_r,
                    BLOCK_N=block_n_s,
                    USE_RT_FLAGS=eff_rt_z,
                    D_PAD=eff_d_z if eff_rt_z else 0,
                    num_warps=4,
                )
            else:
                z_raw_cpu = _gen_sobol_cpu_slice(r_start_c, r_end_c)
                z_raw_c = torch.from_numpy(
                    np.ascontiguousarray(z_raw_cpu.transpose(0, 1, 4, 2, 3).reshape(eff_m_z * n_sub, eff_d_z, n_tot_c))
                ).to(device=qmc_device, dtype=torch.float32)
            assert eff_urot_z is not None
            z_4d_c = z_raw_c.view(eff_m_z, n_sub, eff_d_z, n_tot_c)
            if use_cg_arena:
                z_mean_c = _alloc_cg_buf(True, qmc_device, "perf_aux", (eff_m_z, eff_d_z, n_tot_c), torch.float32)
                torch.mean(z_4d_c, dim=1, out=z_mean_c)
                z_mean_rot_c = _alloc_cg_buf(True, qmc_device, "lx_period", (eff_m_z, eff_d_z, n_tot_c), torch.float32)
                torch.matmul(eff_urot_z, z_mean_c.view(eff_m_z, eff_d_z * n_tot_c), out=z_mean_rot_c.view(eff_m_z, eff_d_z * n_tot_c))
                torch.sub(z_mean_rot_c, z_mean_c, out=z_mean_rot_c)
                z_out_c = _alloc_cg_buf(True, qmc_device, "z_out", (eff_m_z * n_sub, eff_d_z, n_tot_c), torch.float32)
                torch.add(z_4d_c, z_mean_rot_c.unsqueeze(1), out=z_out_c.view(eff_m_z, n_sub, eff_d_z, n_tot_c))
            else:
                z_mean_c = z_4d_c.mean(dim=1)
                z_mean_rot_c = torch.matmul(
                    eff_urot_z,
                    z_mean_c.view(eff_m_z, eff_d_z * n_tot_c),
                ).view(eff_m_z, eff_d_z, n_tot_c)
                z_out_c = (z_4d_c + (z_mean_rot_c - z_mean_c).unsqueeze(1)).reshape(eff_m_z * n_sub, eff_d_z, n_tot_c).contiguous()
            del z_raw_c
            return z_out_c

        will_cg_eager_skip = bool(
            os.environ.get("DEEPVOL_QMC_CG", "1") != "0"
            and can_triton_sobol
            and (not is_cont_ki_active)
            and (need_stage2_aad or use_vibrato_gamma or need_spot_bumps)
        )
        if z_indep_full_f32 is None and z_full_bytes <= 18 * 1024 * 1024 and (not will_cg_eager_skip):
            z_indep_full_f32 = _get_z_indep_chunk_gpu(0, eff_r)
            _CRN_SOBOL_CACHE.put(rot_sobol_key, z_indep_full_f32)
    else:
        cpu_rot_key = rot_sobol_key
        cached_rot_cpu = _CRN_CPU_PREP_CACHE.get(cpu_rot_key) if n_total <= 16384 else None
        if cached_rot_cpu is not None:
            z_indep_np = cached_rot_cpu
        else:
            cpu_sob_key = (sobol_key, "v2" if cpp_sob_v2_fn is not None else "scipy")
            cached_z = _CRN_SOBOL_CACHE.get(cpu_sob_key)
            if cached_z is not None:
                z_raw_np = cached_z
            else:
                z_raw_np = _gen_sobol_cpu_slice(0, eff_r)
                if n_total <= 16384:
                    _CRN_SOBOL_CACHE.put(cpu_sob_key, z_raw_np)
            z_mean_np = np.mean(z_raw_np, axis=1)
            z_mean_rot_np = np.tensordot(u_time_np, z_mean_np, axes=(1, 0))
            z_indep_np = np.ascontiguousarray(
                z_raw_np + (z_mean_rot_np - z_mean_np)[:, None, :, :, :],
                dtype=np.float64,
            )
            if n_total <= 16384:
                _CRN_CPU_PREP_CACHE.put(cpu_rot_key, z_indep_np)

    cpp_step_fn, cpp_loo_fn = _get_cpp_crn_fns()

    # Step 1: Build base trajectories (GPU FP64 or C++23 AVX2) and evaluate CRN pv0_1d.
    def _gpu_step_1d_f64(
        lv_b_t: torch.Tensor,
        z_corr_1d: torch.Tensor,
        tau_t: torch.Tensor,
        rn_t: torch.Tensor,
        rt_t: torch.Tensor,
        z_mode: int,
        *,
        m_periods: Optional[int] = None,
        meta_lv_t: Optional[torch.Tensor] = None,
        meta_r_t: Optional[torch.Tensor] = None,
        meta_z_t: Optional[torch.Tensor] = None,
        y_min_vec_t: Optional[torch.Tensor] = None,
        inv_dy_vec_t: Optional[torch.Tensor] = None,
        inv_dt_vec_t: Optional[torch.Tensor] = None,
        meta_tset_t: Optional[torch.Tensor] = None,
        z_layout_mdn: bool = False,
        fwd_perf_scale_t: Optional[torch.Tensor] = None,
        out_mdn: bool = False,
        out_f32: bool = False,
        use_cg_arena: bool = False,
        cg_arena_slot: str = "perf_aux",
        cg_th_flags_g: Optional[torch.Tensor] = None,
        use_th_rt_flags: bool = False,
    ) -> torch.Tensor:
        eff_m = int(m_periods if m_periods is not None else m_rem)
        n_tot_in = int(z_corr_1d.shape[-1])
        eff_r_in = max(n_tot_in // n_rep, 1)
        if z_mode == 2:
            assert meta_lv_t is not None
            b_sz = int(meta_lv_t.shape[0])
        else:
            b_sz = int(z_corr_1d.shape[0]) if z_mode == 1 else int(lv_b_t.shape[0])
        y_min_v = y_min_vec_t if y_min_vec_t is not None else gpu_meta["y_min_vec"]
        inv_dy_v = inv_dy_vec_t if inv_dy_vec_t is not None else gpu_meta["inv_dy_vec"]
        inv_dt_v = inv_dt_vec_t if inv_dt_vec_t is not None else gpu_meta["inv_dt_base"]
        use_f32_k = bool(n_rep <= 2048 and lv_b_t.dtype == torch.float32)
        out_dt = torch.float32 if (use_f32_k and out_f32) else torch.float64
        out_shape = (eff_m, b_sz, n_tot_in) if (use_f32_k and out_mdn) else (b_sz, eff_m, n_tot_in)
        norm_x_res = _alloc_cg_buf(use_cg_arena, qmc_device, cg_arena_slot, out_shape, out_dt)
        if n_rep <= 2048:
            block_n = n_rep
            n_tiles = 1
            nw_step = 16 if block_n >= 2048 else (8 if block_n >= 512 else 4)
            if use_f32_k:
                d_z_st = int(z_corr_1d.shape[1]) if z_layout_mdn else d_assets
                _fast_triton_launch(
                    _crn_step_1d_f32_kernel,
                    (1, b_sz * eff_r_in),
                    lv_b_t,
                    z_corr_1d,
                    tau_t,
                    rn_t,
                    rt_t,
                    meta_lv_t,
                    meta_r_t,
                    meta_z_t,
                    meta_tset_t,
                    y_min_v,
                    inv_dy_v,
                    inv_dt_v,
                    fwd_perf_scale_t,
                    norm_x_res,
                    b_sz,
                    eff_m,
                    d_z_st,
                    cg_th_flags_ptr=cg_th_flags_g if use_th_rt_flags else None,
                    z_mode=z_mode,
                    n_rep=n_rep,
                    n_sub=n_sub,
                    N_t=lv_nt,
                    N_y=lv_ny,
                    BLOCK_N=block_n,
                    Z_LAYOUT_MDN=z_layout_mdn,
                    HAS_FWD_SCALE=bool(fwd_perf_scale_t is not None),
                    OUT_MDN=out_mdn,
                    OUT_F32=out_f32,
                    USE_TH_RT_FLAGS=use_th_rt_flags,
                    num_warps=nw_step,
                )
                return norm_x_res
            dummy_tile_ex = torch.empty((1,), dtype=torch.float64, device=qmc_device)
            _fast_triton_launch(
                _crn_step_1d_f64_kernel,
                (1, b_sz * eff_r_in),
                lv_b_t,
                z_corr_1d,
                tau_t,
                rn_t,
                rt_t,
                meta_lv_t,
                meta_r_t,
                meta_z_t,
                meta_tset_t,
                y_min_v,
                inv_dy_v,
                inv_dt_v,
                norm_x_res,
                dummy_tile_ex,
                dummy_tile_ex,
                b_sz,
                0,
                eff_m,
                n_tiles,
                z_mode=z_mode,
                n_rep=n_rep,
                n_sub=n_sub,
                N_t=lv_nt,
                N_y=lv_ny,
                BLOCK_N=block_n,
                FUSE_M=True,
                num_warps=nw_step,
            )
            return norm_x_res

        lx_out = torch.empty((b_sz, eff_m, n_tot_in), dtype=torch.float64, device=qmc_device)
        block_n = min(128, n_rep)
        n_tiles = (n_rep + block_n - 1) // block_n
        tile_max = torch.empty((b_sz * eff_r_in, n_tiles), dtype=torch.float64, device=qmc_device)
        tile_ex = torch.empty((b_sz * eff_r_in, n_tiles), dtype=torch.float64, device=qmc_device)
        for m in range(eff_m):
            _fast_triton_launch(
                _crn_step_1d_f64_kernel,
                (n_tiles, b_sz * eff_r_in),
                lv_b_t,
                z_corr_1d,
                tau_t,
                rn_t,
                rt_t,
                meta_lv_t,
                meta_r_t,
                meta_z_t,
                meta_tset_t,
                y_min_v,
                inv_dy_v,
                inv_dt_v,
                lx_out,
                tile_max,
                tile_ex,
                b_sz,
                m,
                eff_m,
                n_tiles,
                z_mode=z_mode,
                n_rep=n_rep,
                n_sub=n_sub,
                N_t=lv_nt,
                N_y=lv_ny,
                BLOCK_N=block_n,
                FUSE_M=False,
            )
            t_max_v = tile_max.view(b_sz, eff_r_in, n_tiles)
            g_max = t_max_v.max(dim=2, keepdim=True).values
            ex_sum = (tile_ex.view(b_sz, eff_r_in, n_tiles) * torch.exp(t_max_v - g_max)).sum(dim=2, keepdim=True)
            log_m = g_max + torch.log((ex_sum / float(n_rep)).clamp(min=1e-300))
            lx_m = lx_out[:, m, :].view(b_sz, eff_r_in, n_rep) - log_m
            lx_out[:, m, :] = lx_m.view(b_sz, n_tot_in)
            norm_x_res[:, m, :] = torch.exp(lx_m.view(b_sz, n_tot_in))
        return norm_x_res

    tg_c_base = np.ascontiguousarray(lv_t_grid, dtype=np.float64)

    def _cpp_step_1d_np(
        b_sz: int,
        m_steps: int,
        z_mode_v: int,
        lv_arr: npt.NDArray[np.float64],
        z_arr: npt.NDArray[np.float64],
        tau_arr: npt.NDArray[np.float64],
        rn_arr: npt.NDArray[np.float64],
        rt_arr: npt.NDArray[np.float64],
        out_arr: npt.NDArray[np.float64],
        *,
        y_min_v: Optional[float] = None,
        y_max_v: Optional[float] = None,
        t_grid_arr: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        assert cpp_step_fn is not None
        lv_c = np.ascontiguousarray(lv_arr, dtype=np.float64)
        z_c = np.ascontiguousarray(z_arr, dtype=np.float64)
        tau_c = np.ascontiguousarray(tau_arr, dtype=np.float64)
        rn_c = np.ascontiguousarray(rn_arr, dtype=np.float64)
        rt_c = np.ascontiguousarray(rt_arr, dtype=np.float64)
        tg_c = np.ascontiguousarray(t_grid_arr, dtype=np.float64) if t_grid_arr is not None else tg_c_base
        ym_lo = float(lv_y_min_vec[0]) if y_min_v is None else float(y_min_v)
        ym_hi = float(lv_y_max_vec[0]) if y_max_v is None else float(y_max_v)
        cpp_step_fn(
            int(b_sz),
            int(lv_nt),
            int(lv_ny),
            int(m_steps),
            int(n_sub),
            int(eff_r),
            int(n_rep),
            int(z_mode_v),
            ym_lo,
            ym_hi,
            lv_c.ctypes.data,
            z_c.ctypes.data,
            tau_c.ctypes.data,
            rn_c.ctypes.data,
            rt_c.ctypes.data,
            tg_c.ctypes.data,
            out_arr.ctypes.data,
        )

    def _run_loo_l2o_eval(
        perf_base_dev: Any,
        bump_a: Any,
        bump_b: Any,
        asset_a_np: npt.NDArray[np.int32],
        asset_b_np: npt.NDArray[np.int32],
        df_pay_b_np: Any,
        w_fund_b_np: Any,
        pv_in_tr_np: Any,
        *,
        alt_perf: Any = None,
        scale_mat: Any = None,
        a_slot_np: Optional[npt.NDArray[np.int32]] = None,
        b_slot_np: Optional[npt.NDArray[np.int32]] = None,
        s_slot_np: Optional[npt.NDArray[np.int32]] = None,
        df_row_idx_np: Optional[npt.NDArray[np.int32]] = None,
        wf_row_idx_np: Optional[npt.NDArray[np.int32]] = None,
        return_gpu: bool = False,
        pure_mc: bool = False,
        static_gpu_pack: Optional[
            Tuple[int, bool, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]
        ] = None,
        out_static_gpu_pack: Optional[List[Any]] = None,
        m_periods: Optional[int] = None,
        ko_ov: Optional[torch.Tensor] = None,
        is_call_ov: Optional[torch.Tensor] = None,
        cpn_ov: Optional[torch.Tensor] = None,
        is_guar_ov: Optional[torch.Tensor] = None,
        cpn_rates_ov: Optional[torch.Tensor] = None,
        payoff_f64_ov: Optional[torch.Tensor] = None,
        inv_perm_ov: Optional[torch.Tensor] = None,
        eps_vec_g: Optional[torch.Tensor] = None,
        frozen_term_g: Optional[torch.Tensor] = None,
        i_ki_ext_g: Optional[torch.Tensor] = None,
        use_c2_preterm: bool = False,
        use_frozen_term: bool = False,
        use_f32_eval: Optional[bool] = None,
        out_pv_g: Optional[torch.Tensor] = None,
        d_pad_rt: int = 0,
        cg_flags_g: Optional[torch.Tensor] = None,
        use_rt_flags: bool = False,
        cg_bumps_idx: int = -1,
        use_cg_arena: bool = False,
    ) -> Any:
        eff_m_eval = int(m_periods if m_periods is not None else m_rem)
        eff_d_eval = int(d_pad_rt) if (use_rt_flags and d_pad_rt > 0) else d_assets
        n_b = static_gpu_pack[0] if static_gpu_pack is not None else int(len(asset_a_np))
        if n_b == 0:
            return torch.empty((0,), dtype=torch.float64, device=qmc_device) if return_gpu else np.empty(0, dtype=np.float64)
        is_full_batch = (
            static_gpu_pack[1]
            if static_gpu_pack is not None
            else bool(n_b > 0 and int(asset_a_np[0]) == -2 and alt_perf is None and scale_mat is None and bump_a is not None)
        )
        if use_gpu_triton:
            def _keep_dev_fp(arr_in: Any) -> torch.Tensor:
                if (
                    isinstance(arr_in, torch.Tensor)
                    and arr_in.device == qmc_device
                    and arr_in.dtype in (torch.float32, torch.float64)
                    and arr_in.is_contiguous()
                ):
                    return arr_in
                return _to_gpu_tensor(arr_in, dtype=torch.float64, device=qmc_device)

            ba_gpu = _keep_dev_fp(bump_a) if bump_a is not None else None
            if perf_base_dev is not None:
                pb_gpu = _keep_dev_fp(perf_base_dev)
            elif ba_gpu is not None:
                pb_gpu = ba_gpu
            else:
                pb_gpu = torch.empty((1,), dtype=torch.float64, device=qmc_device)
            alt_gpu = _keep_dev_fp(alt_perf) if alt_perf is not None else pb_gpu
            bb_gpu = _keep_dev_fp(bump_b) if bump_b is not None else None
            n_tot_eval = int(alt_gpu.shape[-1]) if alt_perf is not None else int(pb_gpu.shape[-1])
            if static_gpu_pack is not None:
                _, _, sc_gpu, idx_gpu, df_gpu, wf_gpu, itr_gpu = static_gpu_pack
                aa_gpu, ab_gpu, a_sl_gpu, b_sl_gpu, s_sl_gpu, dfr_gpu, wfr_gpu = idx_gpu.unbind(0)
            else:
                sc_gpu = _to_gpu_tensor(scale_mat, dtype=torch.float64, device=qmc_device) if scale_mat is not None else None
                a_sl_arr = a_slot_np if a_slot_np is not None else np.arange(n_b, dtype=np.int32)
                b_sl_arr = b_slot_np if b_slot_np is not None else np.arange(n_b, dtype=np.int32)
                s_sl_arr = s_slot_np if s_slot_np is not None else np.full(n_b, -1, dtype=np.int32)
                dfr_arr = df_row_idx_np if df_row_idx_np is not None else np.arange(n_b, dtype=np.int32)
                wfr_arr = wf_row_idx_np if wf_row_idx_np is not None else np.arange(n_b, dtype=np.int32)
                idx_packed_np = np.stack(
                    [asset_a_np, asset_b_np, a_sl_arr, b_sl_arr, s_sl_arr, dfr_arr, wfr_arr], axis=0
                ).astype(np.int32, copy=False)
                idx_gpu = _to_gpu_tensor(idx_packed_np, dtype=torch.int32, device=qmc_device)
                aa_gpu, ab_gpu, a_sl_gpu, b_sl_gpu, s_sl_gpu, dfr_gpu, wfr_gpu = idx_gpu.unbind(0)
                if isinstance(df_pay_b_np, np.ndarray) and (w_fund_b_np is None or isinstance(w_fund_b_np, np.ndarray)):
                    df_flat = df_pay_b_np.reshape(-1)
                    n_df_r = int(df_pay_b_np.shape[0])
                    wf_flat = w_fund_b_np.reshape(-1) if w_fund_b_np is not None else np.empty(0, dtype=np.float64)
                    n_wf_r = int(w_fund_b_np.shape[0]) if w_fund_b_np is not None else 0
                    itr_flat = (
                        np.asarray(pv_in_tr_np, dtype=np.float64).reshape(-1)
                        if pv_in_tr_np is not None
                        else np.full(n_b, float(state.pv_in_transit), dtype=np.float64)
                    )
                    dwi_gpu = _to_gpu_tensor(
                        np.concatenate([df_flat, wf_flat, itr_flat], axis=0),
                        dtype=torch.float64,
                        device=qmc_device,
                    )
                    df_gpu = dwi_gpu[: len(df_flat)].view(n_df_r, eff_m_eval)
                    wf_gpu = dwi_gpu[len(df_flat) : len(df_flat) + len(wf_flat)].view(n_wf_r, eff_m_eval) if w_fund_b_np is not None else None
                    itr_gpu = dwi_gpu[len(df_flat) + len(wf_flat) :]
                else:
                    df_gpu = _to_gpu_tensor(df_pay_b_np, dtype=torch.float64, device=qmc_device)
                    wf_gpu = _to_gpu_tensor(w_fund_b_np, dtype=torch.float64, device=qmc_device) if w_fund_b_np is not None else None
                    if pv_in_tr_np is not None:
                        itr_gpu = _to_gpu_tensor(pv_in_tr_np, dtype=torch.float64, device=qmc_device)
                    else:
                        itr_gpu = torch.full((n_b,), float(state.pv_in_transit), dtype=torch.float64, device=qmc_device)
                if out_static_gpu_pack is not None:
                    out_static_gpu_pack.append((n_b, is_full_batch, sc_gpu, idx_gpu, df_gpu, wf_gpu, itr_gpu))

            if use_f32_eval is None:
                l2o_mode = os.environ.get("DEEPVOL_L2O_F32", "1")
                if l2o_mode == "0":
                    eff_f32_eval = False
                elif l2o_mode == "no_pre":
                    eff_f32_eval = bool(n_tot_eval <= 16384 and (not use_c2_preterm))
                else:
                    eff_f32_eval = bool(n_tot_eval <= 16384)
            else:
                eff_f32_eval = bool(use_f32_eval)
            l2o_bn_cap = int(os.environ.get("DEEPVOL_L2O_BN", "256"))
            l2o_nw = int(os.environ.get("DEEPVOL_L2O_NW", "4"))
            n_tile = min(l2o_bn_cap, max(64, _next_power_of_two_int(n_tot_eval)))
            block_n = n_tile
            n_t_tiles = (n_tot_eval + n_tile - 1) // n_tile
            eff_rt_loo = bool(use_rt_flags and cg_flags_g is not None and d_pad_rt > 0)
            pv_tiles_gpu = _alloc_cg_buf(
                use_cg_arena,
                qmc_device,
                "pv_tiles",
                (n_b, n_t_tiles),
                torch.float64,
                zero=eff_rt_loo,
            )
            _fast_triton_launch(
                _crn_loo_l2o_payoff_f64_kernel,
                (n_b, n_t_tiles),
                pb_gpu,
                alt_gpu,
                ba_gpu,
                bb_gpu,
                sc_gpu,
                aa_gpu,
                ab_gpu,
                a_sl_gpu,
                b_sl_gpu,
                s_sl_gpu,
                df_gpu,
                wf_gpu,
                dfr_gpu,
                wfr_gpu,
                pv_tiles_gpu,
                ko_ov if ko_ov is not None else gpu_meta["ko"],
                is_call_ov if is_call_ov is not None else gpu_meta["is_call"],
                cpn_ov if cpn_ov is not None else gpu_meta["cpn"],
                is_guar_ov if is_guar_ov is not None else gpu_meta["is_guar"],
                cpn_rates_ov if cpn_rates_ov is not None else gpu_meta["cpn_rates"],
                inv_perm_ov if inv_perm_ov is not None else gpu_meta["inv_perm"],
                eps_vec_g,
                frozen_term_g,
                i_ki_ext_g,
                payoff_f64_ov if payoff_f64_ov is not None else gpu_meta["payoff_f64"],
                n_b,
                eff_d_eval,
                eff_m_eval,
                n_tot_eval,
                0,
                cg_flags_ptr=cg_flags_g if eff_rt_loo else None,
                CALL_CODE=0 if eff_rt_loo else call_code,
                CPN_CODE=0 if eff_rt_loo else cpn_code,
                OPT_CODE=0 if eff_rt_loo else opt_code,
                HAS_KI=False if eff_rt_loo else has_ki,
                IS_MEM=False if eff_rt_loo else is_memory,
                IS_LONG=False if eff_rt_loo else is_long_flag,
                NO_NOTIONAL=False if eff_rt_loo else no_notional,
                FORCE_CPN_AC=False if eff_rt_loo else force_cpn_ac,
                CPN_DIR_DOWN=False if eff_rt_loo else dir_down,
                CPN_TYPE_OUT=False if eff_rt_loo else type_out,
                N_TILE=n_tile,
                BLOCK_N=block_n,
                IS_FULL_BATCH=is_full_batch,
                HAS_ONE_STAR=False if eff_rt_loo else bool(k_os > 0.0),
                HAS_LOWER_STRIKE=True if eff_rt_loo else has_lower_strike_flag,
                HAS_PART_CALL=False if eff_rt_loo else has_part_call_flag,
                USE_C2_PRETERM=use_c2_preterm,
                USE_FROZEN_TERM=use_frozen_term,
                HAS_EXT_KI=bool(i_ki_ext_g is not None),
                USE_F32_EVAL=eff_f32_eval,
                USE_RT_FLAGS=eff_rt_loo,
                D_PAD=d_pad_rt if eff_rt_loo else 0,
                CG_BUMPS_IDX=cg_bumps_idx if eff_rt_loo else -1,
                num_warps=l2o_nw,
            )
            if out_pv_g is not None and pure_mc:
                torch.sum(pv_tiles_gpu, dim=1, out=out_pv_g)
                if return_gpu:
                    return out_pv_g
                return np.asarray(out_pv_g.detach().cpu().numpy(), dtype=np.float64)
            mc_gpu = pv_tiles_gpu.sum(dim=1)
            pv_out_gpu = mc_gpu if pure_mc else (mc_gpu + itr_gpu)
            if return_gpu:
                return pv_out_gpu
            return np.asarray(pv_out_gpu.detach().cpu().numpy(), dtype=np.float64)

        in_tr_arr = (
            np.ascontiguousarray(pv_in_tr_np, dtype=np.float64)
            if pv_in_tr_np is not None
            else np.full(n_b, float(state.pv_in_transit), dtype=np.float64)
        )
        pb_c = np.ascontiguousarray(perf_base_dev, dtype=np.float64) if perf_base_dev is not None else None
        ba_c = np.ascontiguousarray(bump_a, dtype=np.float64) if bump_a is not None else None
        bb_c = np.ascontiguousarray(bump_b, dtype=np.float64) if bump_b is not None else None
        aa_c = np.ascontiguousarray(asset_a_np, dtype=np.int32)
        ab_c = np.ascontiguousarray(asset_b_np, dtype=np.int32)
        df_c = np.ascontiguousarray(df_pay_b_np, dtype=np.float64)
        wf_c = np.ascontiguousarray(w_fund_b_np, dtype=np.float64) if w_fund_b_np is not None else None
        pv_out_np = np.empty(n_b, dtype=np.float64)

        if (
            cpp_loo_fn is not None
            and ki_monitoring_type == "European"
            and (not is_long_flag or cpp_sde_v2_fn is not None)
            and (not has_lower_strike_flag or k_low >= 0.0)
            and (not has_part_call_flag or k_call > 0.0)
        ):
            desc = _CrnLooL2oBatchArgsC()
            desc.n_bumps = n_b
            desc.d_assets = d_assets
            desc.m_rem = m_rem
            desc.n_paths_total = n_total
            desc.perf_base = pb_c.ctypes.data if pb_c is not None else 0
            desc.bump_perf_a = ba_c.ctypes.data if ba_c is not None else 0
            desc.bump_perf_b = bb_c.ctypes.data if bb_c is not None else 0
            desc.bump_asset_a = aa_c.ctypes.data
            desc.bump_asset_b = ab_c.ctypes.data
            desc.df_pay_mat = df_c.ctypes.data
            desc.w_fund_mat = wf_c.ctypes.data if wf_c is not None else 0
            desc.pv_in_transit = in_tr_arr.ctypes.data
            desc.pv_out = pv_out_np.ctypes.data
            desc.ko_barriers = ko_arr_np.ctypes.data
            desc.is_callable = is_call_np.ctypes.data
            desc.cpn_barriers = cpn_arr_np.ctypes.data
            desc.is_guaranteed_cpn = is_guar_np.ctypes.data
            desc.coupon_rates = cpn_rates_np.ctypes.data
            desc.ko_spread = ko_spread
            desc.cpn_spread = cpn_spread
            desc.ki_spread = ki_spread
            desc.one_star_strike = k_os
            desc.one_star_eps = one_star_eps
            desc.strike = k_put
            desc.lower_strike = k_low
            desc.ki_barrier = b_ki
            desc.option_gearing = float(state.option_gearing)
            desc.autocall_coupon = float(state.autocall_coupon)
            desc.c_mem_0 = float(state.c_mem_0)
            desc.i_ki_past = float(state.i_ki_past)
            desc.part_call_strike = k_call
            desc.part_call_rate = part_call_rate
            desc.one_star_coupon = os_cpn
            desc.call_code = call_code
            desc.cpn_code = cpn_code
            desc.opt_code = opt_code
            desc.has_ki = 1 if has_ki else 0
            desc.is_memory = 1 if is_memory else 0
            desc.is_long_basket = 1 if is_long_flag else 0
            desc.no_notional = 1 if no_notional else 0
            desc.force_cpn_ac = 1 if force_cpn_ac else 0
            desc.cpn_dir_down = 1 if dir_down else 0
            desc.cpn_type_out = 1 if type_out else 0
            desc._reserved[0] = int(inv_perm[0]) + 1
            desc._reserved[1] = (int(inv_perm[1]) + 1) if len(inv_perm) > 1 else 1
            desc._reserved[2] = 0
            desc._reserved[3] = 0
            cpp_loo_fn(ctypes.byref(desc))
            return pv_out_np

        i_ki_base_cpu_cache: Optional[torch.Tensor] = None
        for b_i in range(n_b):
            is_unbumped_row = bool(aa_c[b_i] == -1 and ab_c[b_i] == -1)
            if aa_c[b_i] == -2 and ba_c is not None:
                a_row = int(a_slot_np[b_i]) if a_slot_np is not None else b_i
                pb_t = torch.from_numpy(ba_c[a_row]).clone()
            else:
                assert pb_c is not None
                pb_t = torch.from_numpy(pb_c).clone()
                if aa_c[b_i] >= 0 and ba_c is not None:
                    a_row = int(a_slot_np[b_i]) if a_slot_np is not None else b_i
                    pb_t[:, int(aa_c[b_i]), :] = torch.from_numpy(ba_c[a_row])
                if ab_c[b_i] >= 0 and bb_c is not None:
                    b_row = int(b_slot_np[b_i]) if b_slot_np is not None else b_i
                    pb_t[:, int(ab_c[b_i]), :] = torch.from_numpy(bb_c[b_row])
            wc = _eval_index_batch_torch(pb_t, call_code, inv_perm)
            wcp = wc if cpn_code == call_code else _eval_index_batch_torch(pb_t, cpn_code, inv_perm)
            wo = wc if opt_code == call_code else (wcp if opt_code == cpn_code else _eval_index_batch_torch(pb_t, opt_code, inv_perm))
            wavg_last_fb = _eval_index_batch_torch(pb_t[m_rem - 1 : m_rem], 2, inv_perm)[0] if is_long_flag else None
            wb = _eval_index_batch_torch(pb_t[m_rem - 1 : m_rem], 1, inv_perm)[0] if k_os > 0.0 else None
            surv_v = torch.ones(n_total, dtype=torch.float64)
            cmem_v = torch.full((n_total,), float(state.c_mem_0), dtype=torch.float64)
            pv_p = torch.zeros(n_total, dtype=torch.float64)
            eff_csp = -cpn_spread if dir_down else cpn_spread
            for m_i in range(m_rem):
                if is_call_np[m_i] > 0.5:
                    bk = ko_arr_np[m_i]
                    if abs(ko_spread) <= 1e-6:
                        gk = (wc[m_i] >= bk).to(torch.float64)
                    else:
                        bl, bh = min(bk, bk + ko_spread), max(bk, bk + ko_spread)
                        gk = ((wc[m_i] - bl) / max(bh - bl, 1e-12)).clamp(0.0, 1.0)
                else:
                    gk = torch.zeros(n_total, dtype=torch.float64)
                if is_guar_np[m_i] > 0.5:
                    gc = torch.ones(n_total, dtype=torch.float64)
                else:
                    bc = cpn_arr_np[m_i]
                    if abs(eff_csp) <= 1e-6:
                        hd = (wcp[m_i] <= bc) if dir_down else (wcp[m_i] >= bc)
                        gc = (1.0 - hd.to(torch.float64)) if type_out else hd.to(torch.float64)
                    else:
                        blc, bhc = min(bc, bc + eff_csp), max(bc, bc + eff_csp)
                        ru = ((wcp[m_i] - blc) / max(bhc - blc, 1e-12)).clamp(0.0, 1.0)
                        hg = (1.0 - ru) if dir_down else ru
                        gc = (1.0 - hg) if type_out else hg
                egc = torch.maximum(gc, gk) if force_cpn_ac else gc
                pm = surv_v * gk
                cr = float(cpn_rates_np[m_i])
                if is_memory:
                    ec = cr + cmem_v
                    ccf = surv_v * egc * ec + pm * float(state.autocall_coupon)
                    cmem_v = surv_v * ec * (1.0 - egc)
                else:
                    ccf = surv_v * egc * cr + pm * float(state.autocall_coupon)
                pv_p = pv_p + df_c[b_i, m_i] * ccf
                if not no_notional:
                    pv_p = pv_p + df_c[b_i, m_i] * pm
                if wf_c is not None:
                    pv_p = pv_p - wf_c[b_i, m_i] * surv_v
                surv_v = (surv_v - pm).clamp(min=0.0)
            if not no_notional:
                pv_p = pv_p + df_c[b_i, m_rem - 1] * surv_v
            kig: torch.Tensor | float
            if has_ki and float(state.i_ki_past) < 0.5 and ki_monitoring_type in ("Continuous", "Daily", "Discrete"):
                if is_unbumped_row and i_ki_base_cpu_cache is not None:
                    kig = i_ki_base_cpu_cache
                else:
                    iki_cpu_row: List[torch.Tensor] = []
                    _compute_payoff_aad_state_adjoints_torch(
                        pb_t,
                        ko_barriers=torch.from_numpy(ko_arr_np),
                        is_callable=torch.from_numpy(is_call_np),
                        cpn_barriers=torch.from_numpy(cpn_arr_np),
                        is_guar=torch.from_numpy(is_guar_np),
                        cpn_rates=torch.from_numpy(cpn_rates_np),
                        df_pay=torch.tensor(df_c[b_i], dtype=torch.float64),
                        w_fund=torch.tensor(wf_c[b_i], dtype=torch.float64) if wf_c is not None else None,
                        inv_perm=inv_perm,
                        call_code=call_code,
                        cpn_code=cpn_code,
                        opt_code=opt_code,
                        ko_spread=ko_spread,
                        cpn_spread=cpn_spread,
                        ki_spread=ki_spread,
                        cpn_dir_down=dir_down,
                        cpn_type_out=type_out,
                        is_memory=is_memory,
                        force_cpn_ac=force_cpn_ac,
                        no_notional=no_notional,
                        is_long_basket=is_long_flag,
                        has_ki=has_ki,
                        ki_barrier=b_ki,
                        i_ki_past=float(state.i_ki_past),
                        ki_monitoring_type=ki_monitoring_type,
                        tau_obs=tau_obs_np,
                        vol_atm_obs=vol_atm_obs,
                        strike=k_put,
                        lower_strike=state.lower_strike,
                        option_gearing=float(state.option_gearing),
                        ac_cpn=float(state.autocall_coupon),
                        c_mem_0=float(state.c_mem_0),
                        part_call_strike=state.participation_call_strike,
                        part_call_rate=part_call_rate,
                        one_star_strike=(k_os if k_os > 0.0 else state.one_star_strike),
                        one_star_coupon=os_cpn,
                        one_star_eps=one_star_eps,
                        barrier_eps_min=barrier_eps_min,
                        barrier_eps_max=barrier_eps_max,
                        lv_tables_t=torch.from_numpy(base_lv_tables),
                        corr_t=torch.from_numpy(corr_np),
                        fwd_perf_t=torch.from_numpy(fwd_perf_np),
                        y_min=lv_y_min_vec,
                        y_max=lv_y_max_vec,
                        n_sub=n_sub,
                        n_rep=n_rep,
                        emit_stage1=False,
                        compute_grads=False,
                        out_i_ki_ext=iki_cpu_row,
                    )
                    kig = iki_cpu_row[0] if iki_cpu_row else 1.0
                    if is_unbumped_row and isinstance(kig, torch.Tensor):
                        i_ki_base_cpu_cache = kig
            elif has_ki:
                b_ki_fb = k_put if is_long_flag else b_ki
                if abs(ki_spread) <= 1e-6:
                    gki = (wo[m_rem - 1] < b_ki_fb).to(torch.float64)
                else:
                    blk, bhk = min(b_ki_fb, b_ki_fb + ki_spread), max(b_ki_fb, b_ki_fb + ki_spread)
                    gki = ((bhk - wo[m_rem - 1]) / max(bhk - blk, 1e-12)).clamp(0.0, 1.0)
                kig = torch.maximum(torch.full_like(gki, float(state.i_ki_past)), gki)
            else:
                kig = 1.0
            sind: torch.Tensor | float
            if wb is not None:
                if one_star_eps <= 1e-6:
                    sind = (wb >= k_os).to(torch.float64)
                else:
                    sind = ((wb - (k_os - 0.5 * one_star_eps)) / one_star_eps).clamp(0.0, 1.0)
            else:
                sind = 0.0
            opay = torch.zeros(n_total, dtype=torch.float64)
            if is_long_flag:
                assert wavg_last_fb is not None
                opay = opay - kig * (1.0 - sind) * (1.0 - wavg_last_fb)
            else:
                put_act_fb = (k_put > k_low and float(state.option_gearing) > 0.0) if has_lower_strike_flag else (float(state.option_gearing) > 0.0)
                if put_act_fb:
                    p_lo_fb = (k_low - wo[m_rem - 1]).clamp(min=0.0) if has_lower_strike_flag else 0.0
                    p_in = (k_put - wo[m_rem - 1]).clamp(min=0.0) - p_lo_fb
                    opay = opay - float(state.option_gearing) * kig * (1.0 - sind) * p_in
            if has_part_call_flag:
                opay = opay + part_call_rate * (wo[m_rem - 1] - k_call).clamp(min=0.0)
            if k_os > 0.0 and os_cpn != 0.0:
                opay = opay + sind * os_cpn
            pv_p = pv_p + df_c[b_i, m_rem - 1] * surv_v * opay
            pv_out_np[b_i] = float(pv_p.mean().item()) + float(in_tr_arr[b_i])
        return pv_out_np

    repo_curves_map_shared: Optional[Dict[str, Any]] = None
    if (not _warm_gpu_hit) and (want_rho or want_theta):
        repo_src_shared = request if request is not None else state
        repo_curves_map_shared = _merge_usable_repo_curves(
            _extract_repo_curves_map(repo_src_shared, sorted_ids, as_of_orig),
            getattr(state, "repo_curves_map", None),
            as_of_orig,
            _normalize_curve_obj,
        )

    th_prep: Optional[Dict[str, Any]] = _warm_cpu.th_prep if (_warm_gpu_hit and _warm_cpu is not None) else None
    th_cache_key: Any = None
    if (not _warm_gpu_hit) and want_theta:
        th_est_sig, th_itr_rec_sig = _past_itr_sig_local()
        th_cache_key = (
            "th_prep_v2",
            as_of_orig,
            int(theta_roll_days),
            theta_convention,
            tau_obs_dig_base,
            tuple(state.seasoned_state.future_obs_dates),
            tuple(state.seasoned_state.future_pay_dates),
            _np_f64(state.tau_pay).tobytes(),
            spots_np.tobytes(),
            ref_spots_np.tobytes(),
            div_pv_np.tobytes(),
            df_pay_np.tobytes(),
            r_native_obs_np.tobytes(),
            r_tot_obs_np.tobytes(),
            _np_f64(state.dividend_schedule.repo_spread_avg).tobytes(),
            _div_sched_sig_local(),
            _quanto_meta_sig_local(),
            _funding_leg_sig_local(),
            float(state.pv_in_transit),
            th_est_sig,
            th_itr_rec_sig,
            ko_src.tobytes(),
            ic_src.tobytes(),
            cp_src.tobytes(),
            ig_src.tobytes(),
            cr_src.tobytes(),
            payoff_s_np.tobytes(),
            call_code,
            cpn_code,
            opt_code,
            has_ki,
            type_out,
            dir_down,
            is_memory,
            force_cpn_ac,
            no_notional,
            has_lower_strike_flag,
            has_part_call_flag,
            is_long_flag,
            bool(getattr(state, "already_autocalled", False)),
            bool(state.seasoned_state.already_autocalled),
            tuple(inv_perm),
            tuple(surf_dig_map.get(tk, "") for tk in sorted_ids),
            lv_y_min_vec.tobytes(),
            lv_y_max_vec.tobytes(),
            lv_nt,
            lv_ny,
            str(out_device),
            spots_roll_sig,
            _csig(state.discount_curve),
            tuple(_csig((state.asset_financing_curves or {}).get(tk)) for tk in sorted_ids),
            _raw_repo_sigs(state, request, sorted_ids, curve_sig_memo),
        )
        th_prep = _CRN_STATIC_CACHE.get(th_cache_key)
    if (not _warm_gpu_hit) and want_theta and th_prep is None:
        as_of_rolled, dt_shift = _roll_as_of_for_theta(as_of_orig)
        repo_map_th = repo_curves_map_shared or {}
        fixings_obj = _extract_request_mkt_attr(request, "Fixings", "fixings")
        spots_roll_np = spots_np.copy()
        fixed_mask_roll = np.zeros(d_assets, dtype=bool)
        if as_of_rolled is not None and state.seasoned_state.future_obs_dates and state.seasoned_state.future_obs_dates[0] <= as_of_rolled:
            obs0_dt = state.seasoned_state.future_obs_dates[0]
            for i_tk, tk in enumerate(sorted_ids):
                fx_v = _extract_fixing_value(fixings_obj, tk, obs0_dt)
                if fx_v is not None:
                    spots_roll_np[i_tk] = fx_v
                    fixed_mask_roll[i_tk] = True

        # Apply 3d_spotCarry S_i(t + dt) = S_i(t) * DF_repo(t, t+dt) / DF_fin(t, t+dt) when not fixing-overridden.
        if as_of_rolled is not None:
            for i_tk, tk in enumerate(sorted_ids):
                if fixed_mask_roll[i_tk]:
                    continue
                c_fin_r = _normalize_curve_obj((state.asset_financing_curves or {}).get(tk), as_of_date=as_of_orig) or disc_curve
                df_fin_r = (
                    max(float(_curve_df_cached_np(c_fin_r, [as_of_rolled])[0]), 1e-15)
                    if (c_fin_r is not None and hasattr(c_fin_r, "df"))
                    else 1.0
                )
                r_cur_r = _normalize_curve_obj(repo_map_th.get(tk), as_of_date=as_of_orig)
                if r_cur_r is not None and hasattr(r_cur_r, "df"):
                    df_rep_r = max(float(_curve_df_cached_np(r_cur_r, [as_of_rolled])[0]), 1e-15)
                else:
                    df_rep_r = math.exp(-float(state.dividend_schedule.repo_spread_avg[i_tk].item()) * dt_shift)
                spots_roll_np[i_tk] = max(float(spots_roll_np[i_tk]) * (df_rep_r / df_fin_r), 1e-6)

        has_ex_div_in_roll = False
        if as_of_rolled is not None:
            for tk in sorted_ids:
                for ev in state.dividend_schedule.dividend_events.get(tk, []):
                    if as_of_orig < ev.ex_date <= as_of_rolled:
                        has_ex_div_in_roll = True
                        break
                if has_ex_div_in_roll:
                    break

        k_cross, eff_autocalled_th, eff_iki_th, eff_cmem_th, crossed_pv = _collapse_crossed_periods_for_roll(
            state,
            disc_curve,
            as_of_rolled if as_of_rolled is not None else as_of_orig,
            spots_roll_np,
            ref_spots_np,
        )
        in_tr_th = (
            float(_rolled_in_transit_pv(state, disc_curve, as_of_rolled, None, out_device).detach().cpu().item())
            if as_of_rolled is not None
            else float(state.pv_in_transit)
        )
        th_prep = {
            "k_cross": int(k_cross),
            "eff_autocalled": bool(eff_autocalled_th),
            "eff_iki": float(eff_iki_th),
            "eff_cmem": float(eff_cmem_th),
            "crossed_pv": float(crossed_pv),
            "in_tr": float(in_tr_th),
            "dt_shift": float(dt_shift),
            "has_ex_div_in_roll": bool(has_ex_div_in_roll),
        }
        if not eff_autocalled_th and k_cross < m_rem:
            m_rem_th = m_rem - k_cross
            obs_dts_th = state.seasoned_state.future_obs_dates[k_cross:]
            pay_dts_th = state.seasoned_state.future_pay_dates[k_cross:]
            tau_obs_th_raw = tau_obs_np[k_cross:]
            tau_obs_th_np = np.maximum(tau_obs_th_raw - dt_shift, 1e-6)

            if as_of_rolled is not None and disc_curve is not None and hasattr(disc_curve, "df"):
                disc_anchor_th = max(float(_curve_df_cached_np(disc_curve, [as_of_rolled])[0]), 1e-15)
                df_pay_th_np = _curve_df_cached_np(disc_curve, pay_dts_th) / disc_anchor_th
                df_obs_disc_th = _curve_df_cached_np(disc_curve, obs_dts_th) / disc_anchor_th
            else:
                r_pay_imp = -np.log(np.maximum(df_pay_np[k_cross:], 1e-15)) / np.maximum(
                    state.tau_pay.detach().cpu().numpy().astype(np.float64)[k_cross:], 1e-6
                )
                tau_pay_th_np = np.maximum(
                    state.tau_pay.detach().cpu().numpy().astype(np.float64)[k_cross:] - dt_shift, 0.0
                )
                df_pay_th_np = np.exp(-r_pay_imp * tau_pay_th_np)
                df_obs_disc_th = df_pay_th_np

            fwd_perf_th_np = np.empty((d_assets, m_rem_th), dtype=np.float64)
            r_tot_th_np = np.empty((d_assets, m_rem_th), dtype=np.float64)
            r_nat_th_np = np.empty((d_assets, m_rem_th), dtype=np.float64)
            t_max_th = max(float(tau_obs_th_np[-1]), 1e-4)
            lv_t_grid_th = np.linspace(1e-5, t_max_th, lv_nt, dtype=np.float64)
            lv_th_tables = np.empty((d_assets, lv_nt, lv_ny), dtype=np.float64)

            grid_th_t_dig = _content_digest(tau_obs_th_np, lv_t_grid_th)
            tau_th_dig = _content_digest(tau_obs_th_np)
            df_fin_th_cache: Dict[Any, Tuple[float, npt.NDArray[np.float64]]] = {}
            obs_th_all_dts = [as_of_rolled] + list(obs_dts_th) if as_of_rolled is not None else []
            for i, ticker in enumerate(sorted_ids):
                c_fin_i = _normalize_curve_obj((state.asset_financing_curves or {}).get(ticker), as_of_date=as_of_orig) or disc_curve
                if as_of_rolled is not None and c_fin_i is not None and hasattr(c_fin_i, "df"):
                    inner_cf = getattr(c_fin_i, "_inner_curve", None) or c_fin_i
                    q_cf = getattr(inner_cf, "_raw_quotes", getattr(inner_cf, "quotes", getattr(inner_cf, "values", ())))
                    t_cf = getattr(inner_cf, "tenors", ())
                    c_fkey = (
                        type(c_fin_i).__name__,
                        str(getattr(c_fin_i, "identifier", "")),
                        str(getattr(c_fin_i, "dcc", "")),
                        np.ascontiguousarray(t_cf).tobytes() if isinstance(t_cf, np.ndarray) else repr(t_cf),
                        np.ascontiguousarray(q_cf, dtype=np.float64).tobytes() if q_cf is not None else b"",
                    )
                    cached_fin = df_fin_th_cache.get(c_fkey)
                    if cached_fin is None:
                        df_fin_batch = _curve_df_cached_np(c_fin_i, obs_th_all_dts)
                        fin_anc_th = max(float(df_fin_batch[0]), 1e-15)
                        df_fin_th_i = df_fin_batch[1:] / fin_anc_th
                        df_fin_th_cache[c_fkey] = (fin_anc_th, df_fin_th_i)
                    else:
                        fin_anc_th, df_fin_th_i = cached_fin
                else:
                    fin_anc_th = 1.0
                    df_fin_th_i = df_obs_disc_th

                r_curve_i = _normalize_curve_obj(repo_map_th.get(ticker), as_of_date=as_of_orig)
                r_anc_th = 1.0
                if as_of_rolled is not None and r_curve_i is not None and hasattr(r_curve_i, "df"):
                    df_r_batch = _curve_df_cached_np(r_curve_i, obs_th_all_dts)
                    r_anc_th = max(float(df_r_batch[0]), 1e-15)
                    df_repo_th_i = df_r_batch[1:] / r_anc_th
                else:
                    df_repo_th_i = np.exp(-float(state.dividend_schedule.repo_spread_avg[i].item()) * tau_obs_th_np)

                qm_i = quanto_meta.get(ticker)
                if qm_i:
                    rho_live_i = float(qm_i.get("rho_s_fx", 0.0))
                    sig_fx_q_i = np.asarray(qm_i.get("sig_fx_obs", [0.0] * m_rem), dtype=np.float64)[k_cross:]
                    s_dig_th = surf_dig_map.get(ticker) or _surface_content_digest(state.vol_surfaces[ticker])
                    spot_roll_i = float(spots_roll_np[i])
                    iv_th_key = ("iv_atm", s_dig_th, tau_th_dig, spot_roll_i)
                    iv_th_c = _CRN_VEGA_JAC_CACHE.get(iv_th_key)
                    if iv_th_c is not None:
                        sig_eq_q_i = cast(npt.NDArray[np.float64], iv_th_c)
                    else:
                        sig_eq_q_i = np.asarray(
                            state.vol_surfaces[ticker].iv_batch(tau_obs_th_np, np.full(m_rem_th, spot_roll_i)),
                            dtype=np.float64,
                        )
                        _CRN_VEGA_JAC_CACHE.put(iv_th_key, sig_eq_q_i)
                    q_carry_th_i = -rho_live_i * sig_eq_q_i * sig_fx_q_i * tau_obs_th_np
                    q_df_th_i = np.exp(q_carry_th_i)
                else:
                    q_carry_th_i = np.zeros(m_rem_th, dtype=np.float64)
                    q_df_th_i = np.ones(m_rem_th, dtype=np.float64)

                events_i = state.dividend_schedule.dividend_events.get(ticker, [])
                if len(events_i) > 0 and as_of_rolled is not None and c_fin_i is not None and hasattr(c_fin_i, "df"):
                    amt_i, _, inc_i_full = _get_div_np(ticker)
                    inc_i = inc_i_full[:, k_cross:]
                    pay_dts_i = [ev.payment_date for ev in events_i]
                    ex_dts_i = [ev.ex_date for ev in events_i]
                    df_div_pay = _curve_df_cached_np(c_fin_i, pay_dts_i) / fin_anc_th
                    if r_curve_i is not None and hasattr(r_curve_i, "df"):
                        df_div_ex = _curve_df_cached_np(r_curve_i, ex_dts_i) / r_anc_th
                    else:
                        df_div_ex = np.ones(len(events_i), dtype=np.float64)
                    active_th = np.array([1.0 if ev.ex_date > as_of_rolled else 0.0 for ev in events_i], dtype=np.float64)
                    w_th = active_th * df_div_pay / np.maximum(df_div_ex, 1e-12)
                    dpv_th_i = (amt_i * w_th) @ inc_i
                else:
                    dpv_th_i = div_pv_np[i, k_cross:]

                spot_roll_i = float(spots_roll_np[i])
                esc_th_i = _escrowed_spot_softplus_np(np.array([[spot_roll_i]]), dpv_th_i[None, :])[0]
                growth_th_i = (df_repo_th_i * q_df_th_i) / np.maximum(df_fin_th_i, 1e-12)
                fwd_perf_th_np[i] = (esc_th_i * growth_th_i) / ref_spots_np[i]
                r_nat_th_np[i] = np.log(np.maximum(df_repo_th_i / np.maximum(df_fin_th_i, 1e-15), 1e-15))
                r_tot_th_np[i] = r_nat_th_np[i] + q_carry_th_i

                grid_dig_th_i = _content_digest(grid_th_t_dig, lv_y_grids[i])
                lv_th_tables[i] = _build_lv_table_and_forward_cache(
                    state.vol_surfaces[ticker],
                    spot_roll_i,
                    tau_obs_th_np,
                    r_nat_th_np[i],
                    lv_t_grid_th,
                    lv_y_grids[i],
                    dpv_th_i,
                    grid_dig=grid_dig_th_i,
                    surf_dig=surf_dig_map.get(ticker),
                    rn_div_dig=_content_digest(r_nat_th_np[i], dpv_th_i),
                    need_vega_jac=False,
                )["lv_out"]

            w_fund_th_np: Optional[npt.NDArray[np.float64]] = None
            if state.funding_leg is not None and as_of_rolled is not None:
                df_fin_th_fn: Optional[Callable[[Any], torch.Tensor | float]] = None
                if disc_curve is not None and hasattr(disc_curve, "df"):
                    disc_anc_th = max(float(np.asarray(disc_curve.df(as_of_rolled), dtype=np.float64).reshape(-1)[0]), 1e-15)

                    def _df_th_disc(f_dts: Any) -> torch.Tensor | float:
                        val = np.asarray(disc_curve.df(f_dts), dtype=np.float64) / disc_anc_th
                        if not isinstance(f_dts, (list, tuple, np.ndarray)):
                            return float(val.reshape(-1)[0])
                        return torch.tensor(val, dtype=torch.float64, device=out_device)

                    df_fin_th_fn = _df_th_disc
                w_fund_th_np = (
                    _compute_funding_weights_per_period(
                        state,
                        as_of_override=as_of_rolled,
                        df_disc_fn=df_fin_th_fn,
                        df_fwd_fn=None,
                        period_start_idx=k_cross,
                        device=out_device,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            elif w_fund_base_np is not None:
                w_fund_th_np = w_fund_base_np[k_cross:]

            th_prep.update(
                {
                    "m_rem_th": m_rem_th,
                    "tau_obs": tau_obs_th_np,
                    "df_pay": df_pay_th_np,
                    "fwd_perf": fwd_perf_th_np,
                    "r_nat": r_nat_th_np,
                    "r_tot": r_tot_th_np,
                    "w_fund": w_fund_th_np,
                    "t_max": t_max_th,
                    "lv_t_grid": lv_t_grid_th,
                    "lv_tables": lv_th_tables,
                }
            )
        if th_cache_key is not None:
            _CRN_STATIC_CACHE.put(th_cache_key, th_prep)

    gpu_meta: Dict[str, Any] = _warm_gpu.gpu_meta if (_warm_gpu_hit and _warm_gpu is not None) else {}
    norm_x_base_dev: Any = None
    perf_base_dev: Any = None
    z_corr_base_np: Optional[npt.NDArray[np.float64]] = None

    # Pre-declare GPU & CPU buffers populated cleanly in the unified execution phase.
    stage1_rep_np: Optional[npt.NDArray[np.float64]] = None
    bar_chol_aad: Optional[npt.NDArray[np.float64]] = None
    bar_sigma_cont_np: Optional[npt.NDArray[np.float64]] = None
    bar_lv_np = np.zeros((d_assets, lv_nt, lv_ny), dtype=np.float64)
    bar_q_shift_np = np.zeros((d_assets, m_rem * n_sub), dtype=np.float64)
    bar_ln_fwd_np = np.zeros((d_assets, m_rem), dtype=np.float64)
    bar_cf_np: Optional[npt.NDArray[np.float64]] = None
    bar_surv_np: Optional[npt.NDArray[np.float64]] = None
    corr_grad_cont_list: List[torch.Tensor] = []

    adj_fxv_tasks: List[Tuple[str, int, float, npt.NDArray[np.float64]]] = []
    adj_rho_tasks: List[
        Tuple[
            str,
            Any,
            int,
            npt.NDArray[np.float64],
            Optional[npt.NDArray[np.float64]],
            npt.NDArray[np.float64],
            List[Tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]],
        ]
    ] = []
    adj_repo_tasks: List[
        Tuple[str, Any, int, int, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ] = []
    adj_div_task: Optional[
        Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64], npt.NDArray[np.float64], List[Tuple[str, int]]]
    ] = None
    adj_fxc_tasks: List[Tuple[str, str, int, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    fxv_tasks: List[Tuple[str, Tuple[int, int, int], float]] = []

    need_vol_atm_s2 = bool(need_stage2_aad or want_vega or want_cega or quanto_meta or use_vibrato_gamma)
    vol_atm_obs: Optional[npt.NDArray[np.float64]]
    if _warm_gpu_hit:
        assert _warm_cpu is not None and _warm_gpu is not None
        vol_atm_obs = _warm_cpu.vol_atm_obs
        chol_f32_g_cached = _warm_gpu.chol_f32_g
        lv_f32_g_cached = _warm_gpu.lv_f32_g
        lv_sp_pm_g = _warm_gpu.lv_sp_pm_g
        lv_th_g = _warm_gpu.lv_th_g
        chol_f64_g = _warm_gpu.chol_f64_g
        tau_step_g = _warm_gpu.tau_step_g
        rn_step_g = _warm_gpu.rn_step_g
        rt_step_g = _warm_gpu.rt_step_g
        fwd_perf_g = _warm_gpu.fwd_perf_g
        fwd_th_g = _warm_gpu.fwd_th_g
        y_min_base_g = _warm_gpu.y_min_base_g
        inv_dy_base_g = _warm_gpu.inv_dy_base_g
        inv_dt_base_g = _warm_gpu.inv_dt_base_g
        id_d_g = _warm_gpu.id_d_g
        sp_pm_meta_g = _warm_gpu.sp_pm_meta_g
        sched_f32_g = _warm_gpu.sched_f32_g
        eps_aad_g = _warm_gpu.eps_aad_g
    else:
        if need_vol_atm_s2:
            vol_atm_obs = np.empty((d_assets, m_rem), dtype=np.float64)
            for i in range(d_assets):
                vol_atm_obs[i] = _get_vol_atm_obs(i)
        else:
            vol_atm_obs = None
        chol_f32_g_cached = None
        lv_f32_g_cached = None
        lv_sp_pm_g = None
        lv_th_g = None
        chol_f64_g = None
        tau_step_g = None
        rn_step_g = None
        rt_step_g = None
        fwd_perf_g = None
        fwd_th_g = None
        y_min_base_g = None
        inv_dy_base_g = None
        inv_dt_base_g = None
        id_d_g = None
        sp_pm_meta_g = None
        sched_f32_g = None
        eps_aad_g = None

    if use_gpu_triton and not _warm_gpu_hit:
        inv_dy_np_vec = float(lv_ny - 1) / np.maximum(lv_y_max_vec - lv_y_min_vec, 1e-12)
        has_th_sde = bool(th_prep is not None and not th_prep["eff_autocalled"] and th_prep["k_cross"] < m_rem)
        inv_dt_base_val = float(lv_nt - 1) / max(t_max, 1e-12)
        inv_dt_arr = np.array(
            [inv_dt_base_val, float(lv_nt - 1) / max(float(th_prep["t_max"]), 1e-12)] if (has_th_sde and th_prep is not None) else [inv_dt_base_val],
            dtype=np.float64,
        )
        tau_step_np = np.concatenate([tau_obs_np, th_prep["tau_obs"]], axis=0) if (has_th_sde and th_prep is not None) else tau_obs_np
        rn_step_np = np.concatenate([r_native_obs_np.reshape(-1), th_prep["r_nat"].reshape(-1)], axis=0) if (has_th_sde and th_prep is not None) else r_native_obs_np.reshape(-1)
        rt_step_np = np.concatenate([r_tot_obs_np.reshape(-1), th_prep["r_tot"].reshape(-1)], axis=0) if (has_th_sde and th_prep is not None) else r_tot_obs_np.reshape(-1)
        fwd_th_flat = th_prep["fwd_perf"].reshape(-1) if (has_th_sde and th_prep is not None) else np.empty(0, dtype=np.float64)
        wf_base_flat = w_fund_base_np.reshape(-1) if w_fund_base_np is not None else np.empty(0, dtype=np.float64)

        tau_prev_np = np.concatenate([[0.0], tau_obs_np[:-1]])
        d_tau_vec = np.maximum(tau_obs_np - tau_prev_np, 1e-10)
        sig_est_s2 = (
            np.maximum(np.mean(vol_atm_obs, axis=0), 0.05)
            if vol_atm_obs is not None
            else np.full(m_rem, 0.25, dtype=np.float64)
        )
        eps_vec_s2 = np.clip(
            0.65 * sig_est_s2 * np.sqrt(d_tau_vec),
            max(float(barrier_eps_min), 0.035),
            max(float(barrier_eps_max), 0.08),
        )

        f64_chunks = [
            payoff_s_np,
            inv_dt_arr,
            lv_y_min_vec,
            inv_dy_np_vec,
            tau_step_np,
            df_pay_np,
            wf_base_flat,
            rn_step_np,
            rt_step_np,
            fwd_perf_np.reshape(-1),
            fwd_th_flat,
            ko_src,
            ic_src,
            cp_src,
            ig_src,
            cr_src,
            chol_base.reshape(-1),
            corr_np.reshape(-1),
            eps_vec_s2,
        ]
        pack_f64_g = torch.as_tensor(np.concatenate(f64_chunks, axis=0), dtype=torch.float64, device=qmc_device)
        p_off = 0

        def _take_f64(shape: Tuple[int, ...]) -> torch.Tensor:
            nonlocal p_off
            cnt = 1
            for s_dim in shape:
                cnt *= s_dim
            sl = pack_f64_g[p_off : p_off + cnt].view(*shape)
            p_off += cnt
            return sl

        payoff_f64_g = _take_f64((15,))
        inv_dt_base_g = _take_f64((len(inv_dt_arr),))
        y_min_base_g = _take_f64((d_assets,))
        inv_dy_base_g = _take_f64((d_assets,))
        tau_step_g = _take_f64((len(tau_step_np),))
        df_pay_g = _take_f64((m_rem,))
        w_fund_g_packed = _take_f64((m_rem,)) if w_fund_base_np is not None else None
        m_th_len = int(th_prep["m_rem_th"]) if (has_th_sde and th_prep is not None) else 0
        rn_step_g = _take_f64((d_assets * m_rem + d_assets * m_th_len,))
        rt_step_g = _take_f64((d_assets * m_rem + d_assets * m_th_len,))
        fwd_perf_g = _take_f64((d_assets, m_rem))
        fwd_th_g = _take_f64((d_assets, m_th_len)) if has_th_sde else None
        ko_g = _take_f64((m_rem,))
        is_call_g = _take_f64((m_rem,))
        cpn_g = _take_f64((m_rem,))
        is_guar_g = _take_f64((m_rem,))
        cpn_rates_g = _take_f64((m_rem,))
        chol_f64_g = _take_f64((d_assets, d_assets))
        corr_g = _take_f64((d_assets, d_assets))
        eps_aad_g = _take_f64((m_rem,))

        id_d_g = torch.arange(d_assets, dtype=torch.int32, device=qmc_device)
        if need_spot_bumps:
            id_2d = torch.arange(2 * d_assets, dtype=torch.int32, device=qmc_device)
            zr_2d = torch.cat([id_d_g, id_d_g], dim=0)
            sp_pm_meta_g = (id_2d, zr_2d, zr_2d)
            step1_dtype = torch.float32 if use_f32_step1 else torch.float64
            lv_sp_pm_g = torch.as_tensor(
                np.concatenate([np.stack(sp_p_lv_list, axis=0), np.stack(sp_m_lv_list, axis=0)], axis=0),
                dtype=step1_dtype,
                device=qmc_device,
            ).contiguous()
        else:
            sp_pm_meta_g = None
            lv_sp_pm_g = None
        lv_f32_g_cached = torch.as_tensor(base_lv_tables, dtype=torch.float32, device=qmc_device).contiguous()
        chol_f32_g_cached = chol_f64_g.to(dtype=torch.float32).contiguous()
        if has_th_sde and th_prep is not None:
            th_dtype_p = torch.float32 if use_f32_step1 else torch.float64
            lv_th_g = torch.as_tensor(th_prep["lv_tables"], dtype=th_dtype_p, device=qmc_device).contiguous()
        else:
            lv_th_g = None

        dt_v_vec = d_tau_vec / float(n_sub)
        a_s_vec = (np.arange(n_sub, dtype=np.float64) + 0.5) / float(n_sub)
        t_mid_mat = tau_prev_np[:, None] + d_tau_vec[:, None] * a_s_vec[None, :]
        dt_sub_mat = np.broadcast_to(dt_v_vec[:, None], (m_rem, n_sub))
        sqrt_dt_mat = np.sqrt(dt_sub_mat)
        q_diff_np = r_tot_obs_np - r_native_obs_np
        q_prev_np = np.concatenate([np.zeros((d_assets, 1), dtype=np.float64), q_diff_np[:, :-1]], axis=1)
        q_shift_mat = q_prev_np[:, :, None] + (q_diff_np - q_prev_np)[:, :, None] * a_s_vec[None, None, :]
        sched_f32_np = np.concatenate(
            [
                t_mid_mat.reshape(-1),
                dt_sub_mat.reshape(-1),
                sqrt_dt_mat.reshape(-1),
                q_shift_mat.reshape(-1),
                lv_y_min_vec,
                inv_dy_np_vec,
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        sched_f32_g = torch.as_tensor(sched_f32_np, dtype=torch.float32, device=qmc_device)

        gpu_meta = {
            "ko": ko_g,
            "is_call": is_call_g,
            "cpn": cpn_g,
            "is_guar": is_guar_g,
            "cpn_rates": cpn_rates_g,
            "inv_perm": torch.tensor(inv_perm, dtype=torch.int64, device=qmc_device).contiguous(),
            "tau": tau_step_g[:m_rem],
            "rn": rn_step_g[: d_assets * m_rem].view(d_assets, m_rem),
            "rt": rt_step_g[: d_assets * m_rem].view(d_assets, m_rem),
            "df_pay": df_pay_g,
            "w_fund": w_fund_g_packed,
            "fwd_perf": fwd_perf_g,
            "corr": corr_g,
            "y_min_vec": y_min_base_g,
            "inv_dy_vec": inv_dy_base_g,
            "inv_dt_base": inv_dt_base_g[:1],
            "payoff_f64": payoff_f64_g,
        }
    elif not use_gpu_triton:
        assert z_indep_np is not None
        z_corr_base_np = (
            np.ascontiguousarray(np.matmul(z_indep_np, chol_base.T), dtype=np.float64)
            if (need_spot_bumps or cpp_sde_v2_fn is None)
            else None
        )
        norm_x_base_np = np.empty((d_assets, m_rem, n_total), dtype=np.float64)
        lx_period_cpu: Optional[npt.NDArray[np.float64]] = None
        lx_norm_cpu: Optional[npt.NDArray[np.float64]] = None
        cpu_vib_lx_pre: Optional[npt.NDArray[np.float64]] = None
        cpu_vib_j_pre: Optional[npt.NDArray[np.float64]] = None
        cpu_vib_sig_pre: Optional[npt.NDArray[np.float64]] = None
        vib_tau_prev_c = np.concatenate([[0.0], tau_obs_np[:-1]])
        vib_d_tau_c = np.maximum(tau_obs_np - vib_tau_prev_c, 1e-10)
        vib_period_level_c = bool(m_rem == 1)
        vib_period_idx_c = m_rem - 1
        vib_sub_idx_c = n_sub - 1
        vib_step_start_c = vib_period_idx_c * n_sub if vib_period_level_c else (m_rem * n_sub - 1)
        vib_n_steps_c = n_sub if vib_period_level_c else 1
        h_vib_val_c = float(vib_d_tau_c[-1]) if vib_period_level_c else float(vib_d_tau_c[-1] / float(n_sub))
        df_last_val_c = float(df_pay_np[-1])
        if cpp_sde_v2_fn is not None:
            if use_vibrato_gamma:
                cpu_vib_lx_pre = np.empty((d_assets, n_total), dtype=np.float64)
                cpu_vib_j_pre = np.empty((d_assets, n_total), dtype=np.float64)
                cpu_vib_sig_pre = np.empty((d_assets, n_total), dtype=np.float64)
            if callable(getattr(cpp_sde_v2_fn, "__call__", None)) and not hasattr(cpp_sde_v2_fn, "argtypes"):
                lx_period_cpu, norm_x_m_v2, _, _, _ = cpp_sde_v2_fn(
                    lv_tables=base_lv_tables,
                    z_indep=z_indep_np,
                    chol_base=chol_base,
                    tau_obs=tau_obs_np,
                    r_native_obs=r_native_obs_np,
                    r_tot_obs=r_tot_obs_np,
                    lv_t_grid=lv_t_grid,
                    lv_y_min=lv_y_min_vec,
                    lv_y_max=lv_y_max_vec,
                    n_sub=n_sub,
                    eff_r=eff_r,
                    n_rep=n_rep,
                    run_forward=True,
                    run_backward=False,
                    out_lx_pre=cpu_vib_lx_pre,
                    out_j_pre=cpu_vib_j_pre,
                    out_sig_pre=cpu_vib_sig_pre,
                    vib_period_idx=vib_period_idx_c,
                    vib_sub_idx=vib_sub_idx_c,
                    save_vib_pre=(2 if vib_period_level_c else 1) if use_vibrato_gamma else 0,
                    vib_j_pre_max=6.0,
                    vib_dsig_dy_max=4.0,
                )
                assert norm_x_m_v2 is not None
                lx_norm_cpu = lx_period_cpu
                norm_x_base_np = np.transpose(norm_x_m_v2, (1, 0, 2))
            else:
                lx_p_arr = np.empty((m_rem, d_assets, n_total), dtype=np.float64)
                nx_p_arr = np.empty((m_rem, d_assets, n_total), dtype=np.float64)
                lv_c64 = np.ascontiguousarray(base_lv_tables, dtype=np.float64)
                z_c64 = np.ascontiguousarray(z_indep_np, dtype=np.float64)
                ch_c64 = np.ascontiguousarray(chol_base, dtype=np.float64)
                tau_c64 = np.ascontiguousarray(tau_obs_np, dtype=np.float64)
                rn_c64 = np.ascontiguousarray(r_native_obs_np, dtype=np.float64)
                rt_c64 = np.ascontiguousarray(r_tot_obs_np, dtype=np.float64)
                lvt_c64 = np.ascontiguousarray(lv_t_grid, dtype=np.float64)
                lvy0_c64 = np.ascontiguousarray(lv_y_min_vec, dtype=np.float64)
                lvy1_c64 = np.ascontiguousarray(lv_y_max_vec, dtype=np.float64)
                sde_args_fwd = _PlatenSdeAadArgsC(
                    d_assets=d_assets,
                    m_rem=m_rem,
                    n_sub=n_sub,
                    eff_r=eff_r,
                    n_rep=n_rep,
                    lv_nt=lv_nt,
                    lv_ny=lv_ny,
                    lv_tables=lv_c64.ctypes.data,
                    z_indep=z_c64.ctypes.data,
                    chol_base=ch_c64.ctypes.data,
                    tau_obs=tau_c64.ctypes.data,
                    r_native_obs=rn_c64.ctypes.data,
                    r_tot_obs=rt_c64.ctypes.data,
                    lv_t_grid=lvt_c64.ctypes.data,
                    lv_y_min=lvy0_c64.ctypes.data,
                    lv_y_max=lvy1_c64.ctypes.data,
                    bar_lx_post=0,
                    lx_norm_in=0,
                    q_shift_sub=0,
                    lx_period_out=lx_p_arr.ctypes.data,
                    norm_x_out=nx_p_arr.ctypes.data,
                    bar_lv_out=0,
                    bar_chol_out=0,
                    bar_q_shift_out=0,
                    run_forward=1,
                    run_backward=0,
                    compute_lv_aad=0,
                    compute_chol_aad=0,
                    compute_q_aad=0,
                    has_q_shift_sub=0,
                    out_lx_pre=cpu_vib_lx_pre.ctypes.data if cpu_vib_lx_pre is not None else 0,
                    out_j_pre=cpu_vib_j_pre.ctypes.data if cpu_vib_j_pre is not None else 0,
                    out_sig_pre=cpu_vib_sig_pre.ctypes.data if cpu_vib_sig_pre is not None else 0,
                    vib_period_idx=vib_period_idx_c,
                    vib_sub_idx=vib_sub_idx_c,
                    save_vib_pre=(2 if vib_period_level_c else 1) if use_vibrato_gamma else 0,
                    _pad_vib=0,
                    vib_j_pre_max=6.0,
                    vib_dsig_dy_max=4.0,
                )
                cpp_sde_v2_fn(ctypes.byref(sde_args_fwd))
                lx_period_cpu = lx_p_arr
                lx_norm_cpu = lx_p_arr
                norm_x_base_np = np.transpose(nx_p_arr, (1, 0, 2))
        else:
            lx_period_cpu, lx_norm_cpu, norm_x_m = _cpu_step_and_ckpt_sde(
                base_lv_tables,
                z_indep_np,
                chol_base,
                tau_obs_np,
                r_native_obs_np,
                r_tot_obs_np,
                lv_t_grid,
                lv_y_min_vec,
                lv_y_max_vec,
                n_sub,
                eff_r,
                n_rep,
            )
            norm_x_base_np = np.transpose(norm_x_m, (1, 0, 2))
        perf_base_m_d_n_np = np.ascontiguousarray(
            np.transpose(norm_x_base_np * fwd_perf_np[:, :, None], (1, 0, 2)), dtype=np.float64
        )
        norm_x_base_dev = norm_x_base_np
        perf_base_dev = perf_base_m_d_n_np

        use_cpu_np_aad = bool(
            (not is_cont_ki_active)
            and call_code in (0, 1, 2)
            and cpn_code in (0, 1, 2)
            and opt_code in (0, 1, 2)
            and str(ki_monitoring_type or "Discrete").strip().lower() in ("european", "discrete")
            and os.environ.get("DEEPVOL_CPU_NP_AAD", "1") != "0"
        )
        if need_stage2_aad or ("ProductValue" in q_set) or ("CashFlows" in q_set) or return_cashflows:
            s1_cpu_box: List[torch.Tensor] = []
            cf_wf_cpu_box: List[torch.Tensor] = []
            if need_stage2_aad or is_cont_ki_active or ("ProductValue" in q_set) or ("CashFlows" in q_set) or return_cashflows:
                if need_stage2_aad:
                    assert lx_period_cpu is not None and lx_norm_cpu is not None
                    if use_cpu_np_aad:
                        bar_lx_post_np, bar_ln_fwd_np, s1_np_out, cf_wf_np_out = _compute_payoff_aad_state_adjoints_cpu_np(
                            perf_base_m_d_n_np,
                            ko_barriers=ko_arr_np,
                            is_callable=is_call_np,
                            cpn_barriers=cpn_arr_np,
                            is_guar=is_guar_np,
                            cpn_rates=cpn_rates_np,
                            df_pay=df_pay_np,
                            w_fund=w_fund_base_np,
                            call_code=call_code,
                            cpn_code=cpn_code,
                            opt_code=opt_code,
                            ko_spread=ko_spread,
                            cpn_spread=cpn_spread,
                            ki_spread=ki_spread,
                            cpn_dir_down=dir_down,
                            cpn_type_out=type_out,
                            is_memory=is_memory,
                            force_cpn_ac=force_cpn_ac,
                            no_notional=no_notional,
                            is_long_basket=is_long_flag,
                            has_ki=has_ki,
                            ki_barrier=b_ki,
                            i_ki_past=float(state.i_ki_past),
                            ki_monitoring_type=ki_monitoring_type,
                            tau_obs=tau_obs_np,
                            vol_atm_obs=vol_atm_obs,
                            strike=k_put,
                            lower_strike=state.lower_strike,
                            option_gearing=float(state.option_gearing),
                            ac_cpn=float(state.autocall_coupon),
                            c_mem_0=float(state.c_mem_0),
                            part_call_strike=state.participation_call_strike,
                            part_call_rate=part_call_rate,
                            one_star_strike=(k_os if k_os > 0.0 else state.one_star_strike),
                            one_star_coupon=os_cpn,
                            one_star_eps=one_star_eps,
                            barrier_eps_min=barrier_eps_min,
                            barrier_eps_max=barrier_eps_max,
                            n_rep=n_rep,
                            emit_stage1=True,
                            compute_grads=True,
                            want_cf_wf=bool(use_pathwise_rate_adj or want_theta),
                        )
                        bar_lv_cont_cpu = None
                        if s1_np_out is not None:
                            stage1_rep_np = s1_np_out
                        if cf_wf_np_out is not None:
                            bar_cf_np = cf_wf_np_out[0]
                            bar_surv_np = cf_wf_np_out[1]
                    else:
                        perf_cpu_t = torch.from_numpy(perf_base_m_d_n_np)
                        wf_cpu = torch.from_numpy(w_fund_base_np) if w_fund_base_np is not None else None
                        perf_cpu_m_d_n = perf_cpu_t
                        bar_lx_post_cpu, bar_ln_fwd_t, bar_lv_cont_cpu = _compute_payoff_aad_state_adjoints_torch(
                            perf_cpu_m_d_n,
                            ko_barriers=torch.from_numpy(ko_arr_np),
                            is_callable=torch.from_numpy(is_call_np),
                            cpn_barriers=torch.from_numpy(cpn_arr_np),
                            is_guar=torch.from_numpy(is_guar_np),
                            cpn_rates=torch.from_numpy(cpn_rates_np),
                            df_pay=torch.from_numpy(df_pay_np),
                            w_fund=wf_cpu,
                            inv_perm=inv_perm,
                            call_code=call_code,
                            cpn_code=cpn_code,
                            opt_code=opt_code,
                            ko_spread=ko_spread,
                            cpn_spread=cpn_spread,
                            ki_spread=ki_spread,
                            cpn_dir_down=dir_down,
                            cpn_type_out=type_out,
                            is_memory=is_memory,
                            force_cpn_ac=force_cpn_ac,
                            no_notional=no_notional,
                            is_long_basket=is_long_flag,
                            has_ki=has_ki,
                            ki_barrier=b_ki,
                            i_ki_past=float(state.i_ki_past),
                            ki_monitoring_type=ki_monitoring_type,
                            tau_obs=tau_obs_np,
                            vol_atm_obs=vol_atm_obs,
                            strike=k_put,
                            lower_strike=state.lower_strike,
                            option_gearing=float(state.option_gearing),
                            ac_cpn=float(state.autocall_coupon),
                            c_mem_0=float(state.c_mem_0),
                            part_call_strike=state.participation_call_strike,
                            part_call_rate=part_call_rate,
                            one_star_strike=(k_os if k_os > 0.0 else state.one_star_strike),
                            one_star_coupon=os_cpn,
                            one_star_eps=one_star_eps,
                            barrier_eps_min=barrier_eps_min,
                            barrier_eps_max=barrier_eps_max,
                            lv_tables_t=torch.from_numpy(base_lv_tables),
                            corr_t=torch.from_numpy(corr_np),
                            fwd_perf_t=torch.from_numpy(fwd_perf_np),
                            y_min=lv_y_min_vec,
                            y_max=lv_y_max_vec,
                            n_sub=n_sub,
                            corr_grad_out=corr_grad_cont_list if want_cega else None,
                            n_rep=n_rep,
                            emit_stage1=True,
                            out_stage1_rep=s1_cpu_box,
                            out_cf_wf=cf_wf_cpu_box if (use_pathwise_rate_adj or want_theta) else None,
                        )
                        bar_lx_post_np = np.ascontiguousarray(bar_lx_post_cpu.to(dtype=torch.float64).numpy(), dtype=np.float64)
                        bar_ln_fwd_np = bar_ln_fwd_t.numpy()
                    if cpp_sde_v2_fn is not None:
                        need_lv_c = bool(want_vega or want_rho or want_divs or want_theta)
                        need_chol_c = bool(want_cega and d_assets > 1)
                        need_q_c = bool(quanto_meta and (want_vega or want_theta))
                        if callable(getattr(cpp_sde_v2_fn, "__call__", None)) and not hasattr(cpp_sde_v2_fn, "argtypes"):
                            _, _, b_lv_c, b_ch_c, b_q_c = cpp_sde_v2_fn(
                                lv_tables=base_lv_tables,
                                z_indep=z_indep_np,
                                chol_base=chol_base,
                                tau_obs=tau_obs_np,
                                r_native_obs=r_native_obs_np,
                                r_tot_obs=r_tot_obs_np,
                                lv_t_grid=lv_t_grid,
                                lv_y_min=lv_y_min_vec,
                                lv_y_max=lv_y_max_vec,
                                n_sub=n_sub,
                                eff_r=eff_r,
                                n_rep=n_rep,
                                bar_lx_post=bar_lx_post_np,
                                lx_norm_in=lx_norm_cpu,
                                run_forward=False,
                                run_backward=True,
                                compute_lv_aad=need_lv_c,
                                compute_chol_aad=need_chol_c,
                                compute_q_aad=need_q_c,
                                vib_j_pre_max=6.0,
                                vib_dsig_dy_max=4.0,
                            )
                            bar_lv_np = b_lv_c if b_lv_c is not None else np.zeros_like(base_lv_tables)
                            bar_chol_aad = b_ch_c if b_ch_c is not None else np.zeros_like(chol_base)
                            bar_q_shift_np = b_q_c if b_q_c is not None else np.zeros((d_assets, m_rem * n_sub), dtype=np.float64)
                        else:
                            b_lv_c = np.empty_like(base_lv_tables) if need_lv_c else np.zeros((1,), dtype=np.float64)
                            b_ch_c = np.empty_like(chol_base) if need_chol_c else np.zeros((1,), dtype=np.float64)
                            b_q_c = np.empty((d_assets, m_rem * n_sub), dtype=np.float64) if need_q_c else np.zeros((1,), dtype=np.float64)
                            lv_b64 = np.ascontiguousarray(base_lv_tables, dtype=np.float64)
                            z_b64 = np.ascontiguousarray(z_indep_np, dtype=np.float64)
                            ch_b64 = np.ascontiguousarray(chol_base, dtype=np.float64)
                            tau_b64 = np.ascontiguousarray(tau_obs_np, dtype=np.float64)
                            rn_b64 = np.ascontiguousarray(r_native_obs_np, dtype=np.float64)
                            rt_b64 = np.ascontiguousarray(r_tot_obs_np, dtype=np.float64)
                            lvt_b64 = np.ascontiguousarray(lv_t_grid, dtype=np.float64)
                            lvy0_b64 = np.ascontiguousarray(lv_y_min_vec, dtype=np.float64)
                            lvy1_b64 = np.ascontiguousarray(lv_y_max_vec, dtype=np.float64)
                            lxn_b64 = np.ascontiguousarray(lx_norm_cpu, dtype=np.float64)
                            sde_args_bwd = _PlatenSdeAadArgsC(
                                d_assets=d_assets,
                                m_rem=m_rem,
                                n_sub=n_sub,
                                eff_r=eff_r,
                                n_rep=n_rep,
                                lv_nt=lv_nt,
                                lv_ny=lv_ny,
                                lv_tables=lv_b64.ctypes.data,
                                z_indep=z_b64.ctypes.data,
                                chol_base=ch_b64.ctypes.data,
                                tau_obs=tau_b64.ctypes.data,
                                r_native_obs=rn_b64.ctypes.data,
                                r_tot_obs=rt_b64.ctypes.data,
                                lv_t_grid=lvt_b64.ctypes.data,
                                lv_y_min=lvy0_b64.ctypes.data,
                                lv_y_max=lvy1_b64.ctypes.data,
                                bar_lx_post=bar_lx_post_np.ctypes.data,
                                lx_norm_in=lxn_b64.ctypes.data,
                                q_shift_sub=0,
                                lx_period_out=0,
                                norm_x_out=0,
                                bar_lv_out=b_lv_c.ctypes.data if need_lv_c else 0,
                                bar_chol_out=b_ch_c.ctypes.data if need_chol_c else 0,
                                bar_q_shift_out=b_q_c.ctypes.data if need_q_c else 0,
                                run_forward=0,
                                run_backward=1,
                                compute_lv_aad=1 if need_lv_c else 0,
                                compute_chol_aad=1 if need_chol_c else 0,
                                compute_q_aad=1 if need_q_c else 0,
                                has_q_shift_sub=0,
                                out_lx_pre=0,
                                out_j_pre=0,
                                out_sig_pre=0,
                                vib_period_idx=0,
                                vib_sub_idx=0,
                                save_vib_pre=0,
                                _pad_vib=0,
                                vib_j_pre_max=6.0,
                                vib_dsig_dy_max=4.0,
                            )
                            cpp_sde_v2_fn(ctypes.byref(sde_args_bwd))
                            bar_lv_np = b_lv_c if need_lv_c else np.zeros_like(base_lv_tables)
                            bar_chol_aad = b_ch_c if need_chol_c else np.zeros_like(chol_base)
                            bar_q_shift_np = b_q_c if need_q_c else np.zeros((d_assets, m_rem * n_sub), dtype=np.float64)
                    else:
                        bar_lv_np, bar_chol_aad, bar_q_shift_np = _cpu_bwd_sde_aad(
                            lx_period_cpu,
                            lx_norm_cpu,
                            bar_lx_post_np,
                            base_lv_tables,
                            z_indep_np,
                            chol_base,
                            tau_obs_np,
                            r_native_obs_np,
                            r_tot_obs_np,
                            lv_t_grid,
                            lv_y_min_vec,
                            lv_y_max_vec,
                            n_sub,
                            eff_r,
                            n_rep,
                        )
                    if bar_lv_cont_cpu is not None:
                        bar_lv_np = bar_lv_np + bar_lv_cont_cpu.numpy()
                    if cf_wf_cpu_box:
                        cf_wf_cpu_np = np.asarray(cf_wf_cpu_box[0].detach().cpu().numpy(), dtype=np.float64)
                        bar_cf_np = cf_wf_cpu_np[0]
                        bar_surv_np = cf_wf_cpu_np[1]
                elif use_cpu_np_aad:
                    _, _, s1_np_out, _ = _compute_payoff_aad_state_adjoints_cpu_np(
                        perf_base_m_d_n_np,
                        ko_barriers=ko_arr_np,
                        is_callable=is_call_np,
                        cpn_barriers=cpn_arr_np,
                        is_guar=is_guar_np,
                        cpn_rates=cpn_rates_np,
                        df_pay=df_pay_np,
                        w_fund=w_fund_base_np,
                        call_code=call_code,
                        cpn_code=cpn_code,
                        opt_code=opt_code,
                        ko_spread=ko_spread,
                        cpn_spread=cpn_spread,
                        ki_spread=ki_spread,
                        cpn_dir_down=dir_down,
                        cpn_type_out=type_out,
                        is_memory=is_memory,
                        force_cpn_ac=force_cpn_ac,
                        no_notional=no_notional,
                        is_long_basket=is_long_flag,
                        has_ki=has_ki,
                        ki_barrier=b_ki,
                        i_ki_past=float(state.i_ki_past),
                        ki_monitoring_type=ki_monitoring_type,
                        tau_obs=tau_obs_np,
                        vol_atm_obs=vol_atm_obs,
                        strike=k_put,
                        lower_strike=state.lower_strike,
                        option_gearing=float(state.option_gearing),
                        ac_cpn=float(state.autocall_coupon),
                        c_mem_0=float(state.c_mem_0),
                        part_call_strike=state.participation_call_strike,
                        part_call_rate=part_call_rate,
                        one_star_strike=(k_os if k_os > 0.0 else state.one_star_strike),
                        one_star_coupon=os_cpn,
                        one_star_eps=one_star_eps,
                        barrier_eps_min=barrier_eps_min,
                        barrier_eps_max=barrier_eps_max,
                        n_rep=n_rep,
                        emit_stage1=True,
                        compute_grads=False,
                    )
                    if s1_np_out is not None:
                        stage1_rep_np = s1_np_out
                else:
                    wf_cpu = torch.from_numpy(w_fund_base_np) if w_fund_base_np is not None else None
                    _compute_payoff_aad_state_adjoints_torch(
                        torch.from_numpy(perf_base_m_d_n_np),
                        ko_barriers=torch.from_numpy(ko_arr_np),
                        is_callable=torch.from_numpy(is_call_np),
                        cpn_barriers=torch.from_numpy(cpn_arr_np),
                        is_guar=torch.from_numpy(is_guar_np),
                        cpn_rates=torch.from_numpy(cpn_rates_np),
                        df_pay=torch.from_numpy(df_pay_np),
                        w_fund=wf_cpu,
                        inv_perm=inv_perm,
                        call_code=call_code,
                        cpn_code=cpn_code,
                        opt_code=opt_code,
                        ko_spread=ko_spread,
                        cpn_spread=cpn_spread,
                        ki_spread=ki_spread,
                        cpn_dir_down=dir_down,
                        cpn_type_out=type_out,
                        is_memory=is_memory,
                        force_cpn_ac=force_cpn_ac,
                        no_notional=no_notional,
                        is_long_basket=is_long_flag,
                        has_ki=has_ki,
                        ki_barrier=b_ki,
                        i_ki_past=float(state.i_ki_past),
                        ki_monitoring_type=ki_monitoring_type,
                        tau_obs=tau_obs_np,
                        vol_atm_obs=vol_atm_obs,
                        strike=k_put,
                        lower_strike=state.lower_strike,
                        option_gearing=float(state.option_gearing),
                        ac_cpn=float(state.autocall_coupon),
                        c_mem_0=float(state.c_mem_0),
                        part_call_strike=state.participation_call_strike,
                        part_call_rate=part_call_rate,
                        one_star_strike=(k_os if k_os > 0.0 else state.one_star_strike),
                        one_star_coupon=os_cpn,
                        one_star_eps=one_star_eps,
                        barrier_eps_min=barrier_eps_min,
                        barrier_eps_max=barrier_eps_max,
                        lv_tables_t=torch.from_numpy(base_lv_tables),
                        corr_t=torch.from_numpy(corr_np),
                        fwd_perf_t=torch.from_numpy(fwd_perf_np),
                        y_min=lv_y_min_vec,
                        y_max=lv_y_max_vec,
                        n_sub=n_sub,
                        corr_grad_out=None,
                        n_rep=n_rep,
                        emit_stage1=True,
                        out_stage1_rep=s1_cpu_box,
                        compute_grads=False,
                    )
            if s1_cpu_box:
                stage1_rep_np = np.asarray(s1_cpu_box[0].detach().cpu().numpy(), dtype=np.float64)

        # CPU Multifidelity-GBM Control Variate parity with GPU when cv_mode == "multifidelity_gbm".
        cpu_mf_s1_crn: Optional[npt.NDArray[np.float64]] = None
        cpu_mf_s1_cheap: Optional[npt.NDArray[np.float64]] = None
        cpu_mf_se_cheap: Optional[float] = None
        cpu_use_mf_cv = bool(
            cv_mode_cfg == "multifidelity_gbm"
            and stage1_rep_np is not None
            and not is_cont_ki_active
            and ki_monitoring_type == "European"
            and d_assets <= 16
        )
        if cpu_use_mf_cv:
            d_var_gbm_cpu = _compute_mf_dvar_gbm_np(
                state, sorted_ids, ref_spots_np, fwd_perf_np, tau_obs_np, base_lv_tables, lv_t_grid, lv_y_grids
            )
            z_crn_c = np.matmul(np.mean(z_indep_np, axis=1) * math.sqrt(float(n_sub)), chol_base.T)
            z_crn_rmnd = np.ascontiguousarray(np.transpose(z_crn_c, (1, 0, 2, 3)), dtype=np.float64)
            cpu_need_cf_mf = bool(bar_cf_np is not None)
            cpu_mf_s1_crn, cf_surv_crn_cpu = _eval_mf_gbm_batch_cpu(
                z_crn_rmnd,
                d_var_gbm_cpu.astype(np.float64),
                fwd_perf_np,
                df_pay_np,
                w_fund_base_np,
                ko_arr_np,
                is_call_np,
                cpn_arr_np,
                is_guar_np,
                cpn_rates_np,
                inv_perm,
                call_code=call_code,
                cpn_code=cpn_code,
                opt_code=eff_opt_code_s2,
                ko_spread=ko_spread,
                cpn_spread=cpn_spread,
                ki_spread=ki_spread,
                k_os=k_os,
                one_star_eps=one_star_eps,
                k_put=k_put,
                k_low=k_low,
                b_ki=b_ki,
                option_gearing=float(state.option_gearing),
                ac_cpn=float(state.autocall_coupon),
                c_mem_0=float(state.c_mem_0),
                i_ki_past=float(state.i_ki_past),
                k_call=k_call,
                part_call_rate=part_call_rate,
                os_cpn=os_cpn,
                has_ki=has_ki,
                is_memory=is_memory,
                is_long_basket=is_long_flag,
                no_notional=no_notional,
                force_cpn_ac=force_cpn_ac,
                dir_down=dir_down,
                type_out=type_out,
                has_lower_strike=has_lower_strike_flag,
                has_part_call=has_part_call_flag,
                antithetic=False,
                compute_cf_surv=cpu_need_cf_mf,
            )
            raw_n_cheap_cpu = getattr(cfg, "n_cheap_mf", None)
            n_cheap_cpu_cfg = max(
                int(raw_n_cheap_cpu) if raw_n_cheap_cpu is not None else 32768,
                4 * n_total if n_total >= 32768 else 32768,
            )
            n_rep_mf_cpu = _next_power_of_two_int(min(max(n_rep, 1024), 4096))
            eff_r_mf_cpu = max(8, (n_cheap_cpu_cfg + (2 * n_rep_mf_cpu) - 1) // (2 * n_rep_mf_cpu))
            wf_mf_c_src: Any = w_fund_base_np
            mf_cpu_dig = _content_digest(
                "cpu_mf_v2",
                d_var_gbm_cpu,
                fwd_perf_np,
                df_pay_np,
                wf_mf_c_src,
                ko_arr_np,
                is_call_np,
                cpn_arr_np,
                is_guar_np,
                cpn_rates_np,
                corr_np,
                payoff_s_np,
                np.asarray(inv_perm, dtype=np.int64),
                u_time_np,
                chol_base,
                eff_r_mf_cpu,
                n_rep_mf_cpu,
                m_rem,
                d_assets,
                n_sub,
                seed,
                cpu_need_cf_mf,
                call_code,
                cpn_code,
                eff_opt_code_s2,
                has_ki,
                bool(has_ki and b_ki > 0.0 and float(state.i_ki_past) < 0.5),
                bool(k_os > 0.0),
                is_memory,
                is_long_flag,
                no_notional,
                force_cpn_ac,
                dir_down,
                type_out,
                has_lower_strike_flag,
                has_part_call_flag,
            )
            mf_cpu_hit = _CRN_MF_CHEAP_CACHE.get(mf_cpu_dig)
            if mf_cpu_hit is not None:
                cpu_mf_s1_cheap, cf_surv_cheap_cpu, cpu_mf_se_cheap = mf_cpu_hit
            else:
                if cpp_sob_v2_fn is not None and (m_rem * 2 * d_assets) <= 21201:
                    sv_cpu_mf = np.ascontiguousarray(
                        _get_sobol_direction_numbers(torch.device("cpu")).numpy(), dtype=np.uint32
                    )
                    rng_mf_c = np.random.default_rng(seed + 99991)
                    sh_mf_c = rng_mf_c.integers(0, 1 << 30, size=(eff_r_mf_cpu, m_rem * 2 * d_assets), dtype=np.uint32)
                    cv_chol_c = np.ascontiguousarray(chol_base, dtype=np.float64)
                    if callable(getattr(cpp_sob_v2_fn, "__call__", None)) and not hasattr(cpp_sob_v2_fn, "argtypes"):
                        _, zc_mf_out = cpp_sob_v2_fn(
                            d_assets=d_assets,
                            m_rem=m_rem,
                            n_sub=1,
                            eff_r=eff_r_mf_cpu,
                            n_rep=n_rep_mf_cpu,
                            dir_numbers=sv_cpu_mf,
                            digital_shifts=sh_mf_c,
                            bb_ortho_mat=np.ascontiguousarray(u_time_np, dtype=np.float64),
                            chol_mat=cv_chol_c,
                            n_bits=30,
                            use_gray_code=True,
                            apply_bb_time=True,
                            apply_corr=True,
                            antithetic=False,
                            return_indep=False,
                            return_corr=True,
                        )
                        assert zc_mf_out is not None
                        z_cheap_rmnd = np.ascontiguousarray(np.transpose(zc_mf_out[:, 0], (1, 0, 2, 3)), dtype=np.float64)
                    else:
                        zc_mf_raw = np.empty((m_rem, 1, eff_r_mf_cpu, n_rep_mf_cpu, d_assets), dtype=np.float64)
                        u_time_mf64 = np.ascontiguousarray(u_time_np, dtype=np.float64)
                        sob_mf_args = _SobolGenArgsC(
                            d_assets=d_assets,
                            m_rem=m_rem,
                            n_sub=1,
                            eff_r=eff_r_mf_cpu,
                            n_rep=n_rep_mf_cpu,
                            sobol_dim=int(sv_cpu_mf.shape[0]),
                            n_bits=30,
                            dir_numbers=sv_cpu_mf.ctypes.data,
                            digital_shifts=sh_mf_c.ctypes.data,
                            bb_ortho_mat=u_time_mf64.ctypes.data,
                            chol_mat=cv_chol_c.ctypes.data,
                            z_indep_out=0,
                            z_corr_out=zc_mf_raw.ctypes.data,
                            use_gray_code=1,
                            apply_bb_time=1,
                            apply_corr=1,
                            antithetic=0,
                        )
                        cpp_sob_v2_fn(ctypes.byref(sob_mf_args))
                        z_cheap_rmnd = np.ascontiguousarray(np.transpose(zc_mf_raw[:, 0], (1, 0, 2, 3)), dtype=np.float64)
                else:
                    z_raw_mf_cpu = np.empty((eff_r_mf_cpu, m_rem, n_rep_mf_cpu, d_assets), dtype=np.float64)
                    for r_mf_i in range(eff_r_mf_cpu):
                        z_raw_mf_cpu[r_mf_i] = _generate_sobol_normals(
                            n_rep_mf_cpu, m_rem * d_assets, scramble=True, seed=seed + 99991 + 7919 * r_mf_i
                        ).reshape(n_rep_mf_cpu, m_rem, d_assets).transpose(1, 0, 2)
                    cv_chol_c = np.ascontiguousarray(chol_base, dtype=np.float64)
                    z_cheap_rmnd = np.ascontiguousarray(
                        np.tensordot(u_time_np, np.matmul(z_raw_mf_cpu, cv_chol_c.T), axes=(1, 1)).transpose(1, 0, 2, 3),
                        dtype=np.float64,
                    )
                cpu_mf_s1_cheap, cf_surv_cheap_cpu = _eval_mf_gbm_batch_cpu(
                    z_cheap_rmnd,
                    d_var_gbm_cpu.astype(np.float64),
                    fwd_perf_np,
                    df_pay_np,
                    w_fund_base_np,
                    ko_arr_np,
                    is_call_np,
                    cpn_arr_np,
                    is_guar_np,
                    cpn_rates_np,
                    inv_perm,
                    call_code=call_code,
                    cpn_code=cpn_code,
                    opt_code=eff_opt_code_s2,
                    ko_spread=ko_spread,
                    cpn_spread=cpn_spread,
                    ki_spread=ki_spread,
                    k_os=k_os,
                    one_star_eps=one_star_eps,
                    k_put=k_put,
                    k_low=k_low,
                    b_ki=b_ki,
                    option_gearing=float(state.option_gearing),
                    ac_cpn=float(state.autocall_coupon),
                    c_mem_0=float(state.c_mem_0),
                    i_ki_past=float(state.i_ki_past),
                    k_call=k_call,
                    part_call_rate=part_call_rate,
                    os_cpn=os_cpn,
                    has_ki=has_ki,
                    is_memory=is_memory,
                    is_long_basket=is_long_flag,
                    no_notional=no_notional,
                    force_cpn_ac=force_cpn_ac,
                    dir_down=dir_down,
                    type_out=type_out,
                    has_lower_strike=has_lower_strike_flag,
                    has_part_call=has_part_call_flag,
                    antithetic=True,
                    compute_cf_surv=cpu_need_cf_mf,
                )
                pv_c_idx = 3 * m_rem + 4
                var_c_mf_cpu = (
                    float(np.var(cpu_mf_s1_cheap[:, pv_c_idx], ddof=1)) / float(eff_r_mf_cpu)
                    if eff_r_mf_cpu > 1
                    else 0.0
                )
                cpu_mf_se_cheap = math.sqrt(max(var_c_mf_cpu, 0.0))
                _CRN_MF_CHEAP_CACHE.put(mf_cpu_dig, (cpu_mf_s1_cheap, cf_surv_cheap_cpu, cpu_mf_se_cheap))
            if bar_cf_np is not None and cf_surv_crn_cpu is not None and cf_surv_cheap_cpu is not None:
                cf_crn_m = np.mean(cf_surv_crn_cpu, axis=0)
                cf_ch_m = np.mean(cf_surv_cheap_cpu, axis=0)
                bar_cf_np = bar_cf_np + (cf_ch_m[0] - cf_crn_m[0])
                if bar_surv_np is not None:
                    bar_surv_np = bar_surv_np + (cf_ch_m[1] - cf_crn_m[1])

    def _base_df_mat(n_b: int) -> Any:
        return np.broadcast_to(df_pay_np[None, :], (n_b, m_rem))

    def _base_wf_mat(n_b: int) -> Any:
        if w_fund_base_np is None:
            return None
        return np.broadcast_to(w_fund_base_np[None, :], (n_b, m_rem))

    pq_chunks: List[npt.NDArray[np.float64]] = []
    gpu_eval_specs: List[Dict[str, Any]] = []
    gpu_eval_rows_total = 0

    def _submit_payoff(
        aa_np: npt.NDArray[np.int32],
        ab_np: npt.NDArray[np.int32],
        *,
        bump_a: Any = None,
        bump_b: Any = None,
        alt_perf: Any = None,
        scale_mat: Any = None,
        df_rows: Any = None,
        wf_rows: Any = None,
        itr_arr: Optional[npt.NDArray[np.float64]] = None,
        spot_a_slots: Optional[npt.NDArray[np.int32]] = None,
        spot_b_slots: Optional[npt.NDArray[np.int32]] = None,
        fwd_alt_np: Optional[npt.NDArray[np.float64]] = None,
        fwd_bump_a_np: Optional[npt.NDArray[np.float64]] = None,
    ) -> Tuple[int, int, int]:
        nonlocal gpu_eval_rows_total
        n_b = int(len(aa_np))
        if n_b == 0:
            return (0, 0, 0)
        if not use_gpu_triton:
            if scale_mat is not None:
                sc_m_d = np.transpose(np.asarray(scale_mat, dtype=np.float64), (0, 2, 1))[:, :, :, None]
                alt_sc = np.ascontiguousarray(perf_base_dev[None, :, :, :] * sc_m_d, dtype=np.float64)
                pv_res = _run_loo_l2o_eval(
                    None,
                    alt_sc,
                    None,
                    np.full(n_b, -2, dtype=np.int32),
                    ab_np,
                    df_rows if df_rows is not None else _base_df_mat(n_b),
                    wf_rows if wf_rows is not None else _base_wf_mat(n_b),
                    itr_arr,
                )
            else:
                is_alt_only = bool(int(aa_np[0]) == -2 and alt_perf is not None)
                pv_res = _run_loo_l2o_eval(
                    None if is_alt_only else perf_base_dev,
                    alt_perf if is_alt_only else bump_a,
                    bump_b,
                    aa_np,
                    ab_np,
                    df_rows if df_rows is not None else _base_df_mat(n_b),
                    wf_rows if wf_rows is not None else _base_wf_mat(n_b),
                    itr_arr,
                )
            c_idx = len(pq_chunks)
            pq_chunks.append(pv_res)
            return (c_idx, 0, n_b)

        s_r = gpu_eval_rows_total
        gpu_eval_rows_total += n_b
        gpu_eval_specs.append(
            {
                "aa": aa_np,
                "ab": ab_np,
                "sc": np.ascontiguousarray(np.transpose(np.asarray(scale_mat, dtype=np.float64), (0, 2, 1)), dtype=np.float64)
                if scale_mat is not None
                else None,
                "spot_a": spot_a_slots,
                "spot_b": spot_b_slots,
                "fwd_alt": fwd_alt_np,
                "fwd_ba": fwd_bump_a_np,
                "df": np.asarray(df_rows, dtype=np.float64) if df_rows is not None else None,
                "wf": np.asarray(wf_rows, dtype=np.float64) if wf_rows is not None else None,
                "itr": np.asarray(itr_arr, dtype=np.float64)
                if itr_arr is not None
                else np.full(n_b, float(state.pv_in_transit), dtype=np.float64),
            }
        )
        return (0, s_r, s_r + n_b)

    def _get_pv(tok: Tuple[int, int, int]) -> npt.NDArray[np.float64]:
        c_idx, s_r, e_r = tok
        if s_r == e_r:
            return np.empty(0, dtype=np.float64)
        return pq_chunks[c_idx][s_r:e_r]

    need_pv0_eval = bool(
        (not skip_step1_loo)
        and (
            (not emit_stage1_gpu)
            or (want_rho and not adj_rho_tasks and not use_pathwise_rate_adj)
            or (want_divs and not use_pathwise_rate_adj)
            or (want_vega and quanto_meta and not use_pathwise_rate_adj)
            or (want_cega and quanto_meta and not use_pathwise_rate_adj)
            or want_theta
            or ((not use_gpu_triton) and need_spot_bumps)
            or ((not use_vibrato_gamma) and (want_gamma or want_cash_gamma))
        )
    )
    if _warm_gpu_hit:
        assert _warm_cpu is not None
        need_pv0_eval = _warm_cpu.need_pv0_eval
        tok_base = _warm_cpu.tok_base
        gpu_eval_rows_total = _warm_cpu.gpu_eval_rows_total
        adj_fxv_tasks = _warm_cpu.adj_fxv_tasks
        fxv_tasks = _warm_cpu.fxv_tasks
        sp_task_meta = _warm_cpu.sp_task_meta
        sp_p_vec = _warm_cpu.sp_p_vec
        sp_m_vec = _warm_cpu.sp_m_vec
        cdl_m_np = _warm_cpu.cdl_m_np
        cdl2_last_np = _warm_cpu.cdl2_last_np
        fwd_sp_p_np = _warm_cpu.fwd_sp_p_np
        fwd_sp_m_np = _warm_cpu.fwd_sp_m_np
        adj_rho_tasks = _warm_cpu.adj_rho_tasks
        rho_tasks = _warm_cpu.rho_tasks
        adj_repo_tasks = _warm_cpu.adj_repo_tasks
        repo_task_meta = _warm_cpu.repo_task_meta
        adj_div_task = _warm_cpu.adj_div_task
        div_task_meta = _warm_cpu.div_task_meta
        adj_fxc_tasks = _warm_cpu.adj_fxc_tasks
        fxc_tasks = _warm_cpu.fxc_tasks
        if _warm_cpu.rho_dict_tmpl is not None:
            rho_dict = {k_r: list(v_r) for k_r, v_r in _warm_cpu.rho_dict_tmpl.items()}
        if _warm_cpu.rho_arrays_tmpl is not None:
            rho_arrays = {k_r: v_r.copy() for k_r, v_r in _warm_cpu.rho_arrays_tmpl.items()}
        if _warm_cpu.repo_rho_has_map:
            repo_rho_dict = {}
        if _warm_cpu.divs_dict_tmpl is not None:
            divs_dict = {k_d: list(v_d) for k_d, v_d in _warm_cpu.divs_dict_tmpl.items()}
        if _warm_cpu.divs_arrays_tmpl is not None:
            divs_arrays = {k_d: v_d.copy() for k_d, v_d in _warm_cpu.divs_arrays_tmpl.items()}
        if _warm_cpu.fx_vega_has_map:
            fx_vega_dict = {}
        if _warm_cpu.fx_cega_has_map:
            fx_cega_dict = {}
    else:
        if (not use_gpu_triton) or need_pv0_eval:
            tok_base = _submit_payoff(
                np.array([-1], dtype=np.int32),
                np.array([-1], dtype=np.int32),
                itr_arr=np.array([float(state.pv_in_transit)], dtype=np.float64),
            )
        else:
            tok_base = (0, 0, 0)

    if (not _warm_gpu_hit) and want_vega and quanto_meta:
        fx_vega_dict = {}
        h_fxv = 1.0e-3
        for i, ticker in enumerate(sorted_ids):
            qm = quanto_meta.get(ticker)
            if not qm:
                continue
            rho_live_f = float(qm.get("rho_s_fx", 0.0))
            sig_eq_q_f = _get_vol_atm_obs(i)
            if use_pathwise_rate_adj:
                adj_fxv_tasks.append((ticker, i, rho_live_f, sig_eq_q_f))
                continue
            mult_fxv = np.exp(-rho_live_f * sig_eq_q_f * h_fxv * tau_obs_np)
            sc_fxv = np.ones((m_rem, d_assets, m_rem), dtype=np.float64)
            sc_fxv[np.arange(m_rem), i, np.arange(m_rem)] = mult_fxv
            tok_fxv = _submit_payoff(
                np.full(m_rem, -1, dtype=np.int32),
                np.full(m_rem, -1, dtype=np.int32),
                scale_mat=sc_fxv,
            )
            fxv_tasks.append((ticker, tok_fxv, h_fxv))

    # Step 3: Enqueue Delta / FD-Gamma / Vibrato spot derivatives.
    if not _warm_gpu_hit:
        sp_task_meta = None
        sp_p_vec = spots_np + h_spots_np
        sp_m_vec = np.maximum(spots_np - h_spots_np, 1e-6)
        cdl_m_np = None
        cdl2_last_np = None
        fwd_sp_p_np = None
        fwd_sp_m_np = None
    if (not _warm_gpu_hit) and (need_spot_bumps or use_vibrato_gamma):
        esc_p_mat = _escrowed_spot_softplus_np(sp_p_vec[:, None], div_pv_np)
        esc_m_mat = _escrowed_spot_softplus_np(sp_m_vec[:, None], div_pv_np)
        if quanto_meta:
            fwd_sp_p_np = np.empty((d_assets, m_rem), dtype=np.float64)
            fwd_sp_m_np = np.empty((d_assets, m_rem), dtype=np.float64)
            for i, ticker in enumerate(sorted_ids):
                qm = quanto_meta.get(ticker)
                if qm:
                    rho_live = float(qm.get("rho_s_fx", 0.0))
                    sig_fx_q = np.asarray(qm.get("sig_fx_obs", [0.0] * m_rem), dtype=np.float64)
                    surf_i = state.vol_surfaces[ticker]
                    sig_eq_0 = _get_vol_atm_obs(i)
                    sig_eq_p = np.asarray(surf_i.iv_batch(tau_obs_np, np.full(m_rem, float(sp_p_vec[i]))), dtype=np.float64)
                    sig_eq_m = np.asarray(surf_i.iv_batch(tau_obs_np, np.full(m_rem, float(sp_m_vec[i]))), dtype=np.float64)
                    q_mult_p = np.exp(-rho_live * (sig_eq_p - sig_eq_0) * sig_fx_q * tau_obs_np)
                    q_mult_m = np.exp(-rho_live * (sig_eq_m - sig_eq_0) * sig_fx_q * tau_obs_np)
                else:
                    q_mult_p = np.ones(m_rem, dtype=np.float64)
                    q_mult_m = np.ones(m_rem, dtype=np.float64)
                fwd_sp_p_np[i] = (esc_p_mat[i] * growth_base[i] * q_mult_p) / ref_spots_np[i]
                fwd_sp_m_np[i] = (esc_m_mat[i] * growth_base[i] * q_mult_m) / ref_spots_np[i]
        else:
            inv_ref = 1.0 / ref_spots_np[:, None]
            fwd_sp_p_np = (esc_p_mat * growth_base) * inv_ref
            fwd_sp_m_np = (esc_m_mat * growth_base) * inv_ref

        if use_vibrato_gamma:
            hp_v = sp_p_vec - spots_np
            hm_v = spots_np - sp_m_vec
            h_sum_v = np.maximum(hp_v + hm_v, 1e-12)
            ln_fp = np.log(np.maximum(fwd_sp_p_np, 1e-30))
            ln_fm = np.log(np.maximum(fwd_sp_m_np, 1e-30))
            ln_f0 = np.log(np.maximum(fwd_perf_np, 1e-30))
            cdl_m_np = ((spots_np / h_sum_v)[:, None] * (ln_fp - ln_fm)).T
            d2_last = 2.0 * (
                ln_fp[:, -1] / np.maximum(hp_v, 1e-12)
                + ln_fm[:, -1] / np.maximum(hm_v, 1e-12)
                - ln_f0[:, -1] * (h_sum_v / np.maximum(hp_v * hm_v, 1e-24))
            ) / h_sum_v
            cdl2_last_np = (spots_np * spots_np) * d2_last

    pv0_1d = 0.0
    pv_preterm_np = np.empty(0, dtype=np.float64)
    pv_th_fused_np = np.empty(0, dtype=np.float64)
    h_vib_np: Optional[npt.NDArray[np.float64]] = None
    cross_pairs_vib: List[Tuple[int, int]] = []
    if (not _warm_gpu_hit) and need_spot_bumps:
        assert fwd_sp_p_np is not None and fwd_sp_m_np is not None
        if not use_gpu_triton:
            assert z_corr_base_np is not None and z_indep_np is not None
            if cpp_step_sp_v2_fn is not None and lv_ny <= 128:
                lv_pm_batch_np = np.ascontiguousarray(
                    np.stack([np.stack([sp_p_lv_list[i], sp_m_lv_list[i]], axis=0) for i in range(d_assets)], axis=0),
                    dtype=np.float64,
                )
                out_nx_pm_cpu = cpp_step_sp_v2_fn(
                    lv_tables_pm=lv_pm_batch_np,
                    z_corr=z_corr_base_np,
                    tau_obs=tau_obs_np,
                    r_native_obs=r_native_obs_np,
                    r_tot_obs=r_tot_obs_np,
                    lv_t_grid=lv_t_grid,
                    lv_y_min=lv_y_min_vec,
                    lv_y_max=lv_y_max_vec,
                    n_sub=n_sub,
                    eff_r=eff_r,
                    n_rep=n_rep,
                )
                nx_sp_p_np = out_nx_pm_cpu[0]
                nx_sp_m_np = out_nx_pm_cpu[1]
            else:
                nx_sp_p_np = np.empty((d_assets, m_rem, n_total), dtype=np.float64)
                nx_sp_m_np = np.empty((d_assets, m_rem, n_total), dtype=np.float64)
                for i in range(d_assets):
                    lv_pm_np = np.stack([sp_p_lv_list[i], sp_m_lv_list[i]], axis=0)
                    if cpp_step_fn is not None and lv_ny <= 128:
                        out_pm = np.empty((2, m_rem, n_total), dtype=np.float64)
                        _cpp_step_1d_np(
                            2,
                            m_rem,
                            0,
                            lv_pm_np,
                            z_corr_base_np[..., i],
                            tau_obs_np,
                            r_native_obs_np[i],
                            r_tot_obs_np[i],
                            out_pm,
                            y_min_v=float(lv_y_min_vec[i]),
                            y_max_v=float(lv_y_max_vec[i]),
                        )
                        nx_sp_p_np[i], nx_sp_m_np[i] = out_pm[0], out_pm[1]
                    else:
                        _, _, out_p = _cpu_step_and_ckpt_sde(
                            sp_p_lv_list[i][None], z_indep_np[..., i : i + 1], np.ones((1, 1)),
                            tau_obs_np, r_native_obs_np[i : i + 1], r_tot_obs_np[i : i + 1],
                            lv_t_grid, lv_y_min_vec[i : i + 1], lv_y_max_vec[i : i + 1], n_sub, eff_r, n_rep,
                        )
                        _, _, out_m = _cpu_step_and_ckpt_sde(
                            sp_m_lv_list[i][None], z_indep_np[..., i : i + 1], np.ones((1, 1)),
                            tau_obs_np, r_native_obs_np[i : i + 1], r_tot_obs_np[i : i + 1],
                            lv_t_grid, lv_y_min_vec[i : i + 1], lv_y_max_vec[i : i + 1], n_sub, eff_r, n_rep,
                        )
                        nx_sp_p_np[i], nx_sp_m_np[i] = out_p[:, 0, :], out_m[:, 0, :]
            perf_sp_p_dev = nx_sp_p_np * fwd_sp_p_np[:, :, None]
            perf_sp_m_dev = nx_sp_m_np * fwd_sp_m_np[:, :, None]

        want_fd_dg = bool(want_delta or ((not use_vibrato_gamma) and want_gamma))
        want_fd_gamma_only = bool((not use_vibrato_gamma) and want_gamma)
        want_fd_cg_only = bool((not use_vibrato_gamma) and (not want_gamma) and want_cash_gamma and (not want_delta))
        if want_fd_dg:
            cross_pairs: List[Tuple[int, int]] = []
            if want_fd_gamma_only and d_assets > 1:
                for i in range(d_assets):
                    for j in range(i + 1, d_assets):
                        cross_pairs.append((i, j))
            n_cross = 2 * len(cross_pairs)
            n_sp_total = 2 * d_assets + n_cross
            aa_sp = np.empty(n_sp_total, dtype=np.int32)
            ab_sp = np.full(n_sp_total, -1, dtype=np.int32)
            a_sl_sp = np.zeros(n_sp_total, dtype=np.int32)
            b_sl_sp = np.full(n_sp_total, -1, dtype=np.int32)
            for i in range(d_assets):
                aa_sp[2 * i] = i
                aa_sp[2 * i + 1] = i
                a_sl_sp[2 * i] = i
                a_sl_sp[2 * i + 1] = d_assets + i
            for c_idx, (i, j) in enumerate(cross_pairs):
                b0 = 2 * d_assets + 2 * c_idx
                aa_sp[b0] = i
                ab_sp[b0] = j
                a_sl_sp[b0] = i
                b_sl_sp[b0] = j
                aa_sp[b0 + 1] = i
                ab_sp[b0 + 1] = j
                a_sl_sp[b0 + 1] = d_assets + i
                b_sl_sp[b0 + 1] = d_assets + j
            if use_gpu_triton:
                tok_sp = _submit_payoff(
                    aa_sp,
                    ab_sp,
                    spot_a_slots=a_sl_sp,
                    spot_b_slots=b_sl_sp if n_cross > 0 else None,
                )
            else:
                bump_a_sp = np.empty((n_sp_total, m_rem, n_total), dtype=np.float64)
                bump_b_sp: Optional[npt.NDArray[np.float64]] = (
                    np.zeros((n_sp_total, m_rem, n_total), dtype=np.float64) if n_cross > 0 else None
                )
                for i in range(d_assets):
                    bump_a_sp[2 * i] = perf_sp_p_dev[i]
                    bump_a_sp[2 * i + 1] = perf_sp_m_dev[i]
                for c_idx, (i, j) in enumerate(cross_pairs):
                    b0 = 2 * d_assets + 2 * c_idx
                    bump_a_sp[b0] = perf_sp_p_dev[i]
                    assert bump_b_sp is not None
                    bump_b_sp[b0] = perf_sp_p_dev[j]
                    bump_a_sp[b0 + 1] = perf_sp_m_dev[i]
                    bump_b_sp[b0 + 1] = perf_sp_m_dev[j]
                tok_sp = _submit_payoff(aa_sp, ab_sp, bump_a=bump_a_sp, bump_b=bump_b_sp)
            sp_task_meta = {"mode": "dg", "tok": tok_sp, "cross_pairs": cross_pairs}
        elif want_fd_cg_only:
            if use_gpu_triton:
                tok_epm = _submit_payoff(
                    np.array([-2, -2], dtype=np.int32),
                    np.array([-1, -1], dtype=np.int32),
                    spot_a_slots=np.array([-100, -101], dtype=np.int32),
                )
            else:
                perf_epm = np.ascontiguousarray(
                    np.stack([np.transpose(perf_sp_p_dev, (1, 0, 2)), np.transpose(perf_sp_m_dev, (1, 0, 2))], axis=0),
                    dtype=np.float64,
                )
                tok_epm = _submit_payoff(
                    np.array([-2, -2], dtype=np.int32),
                    np.array([-1, -1], dtype=np.int32),
                    alt_perf=perf_epm,
                )
            sp_task_meta = {"mode": "cg", "tok": tok_epm}

    if (
        not use_gpu_triton
        and use_vibrato_gamma
        and cpp_vib_v2_fn is not None
        and cpu_vib_lx_pre is not None
        and cpu_vib_j_pre is not None
        and cpu_vib_sig_pre is not None
    ):
        assert cdl_m_np is not None and cdl2_last_np is not None
        assert z_indep_np is not None
        sig_est_cpu_v = (
            np.maximum(np.mean(vol_atm_obs, axis=0), 0.05)
            if vol_atm_obs is not None
            else np.full(m_rem, 0.25, dtype=np.float64)
        )
        eps_cpu_vib = np.ascontiguousarray(
            np.clip(
                0.65 * sig_est_cpu_v * np.sqrt(vib_d_tau_c),
                max(float(barrier_eps_min), 0.035),
                max(float(barrier_eps_max), 0.08),
            ),
            dtype=np.float64,
        )
        l_inv_t_cpu = np.ascontiguousarray(np.linalg.pinv(chol_base, rcond=1e-12).T, dtype=np.float64)
        c_inv_cpu = np.ascontiguousarray(l_inv_t_cpu @ l_inv_t_cpu.T, dtype=np.float64)
        inv_perm_c64 = np.ascontiguousarray(inv_perm, dtype=np.int64)
        h_vib_out_c = np.zeros((d_assets, d_assets), dtype=np.float64)
        frozen_term_cpu = np.zeros((4, n_total), dtype=np.float64) if m_rem > 1 else np.zeros((1,), dtype=np.float64)
        z_vib_c64 = np.ascontiguousarray(z_indep_np, dtype=np.float64)
        ch_vib_c64 = np.ascontiguousarray(chol_base, dtype=np.float64)
        fwd_term_c64 = np.ascontiguousarray(fwd_perf_np[:, m_rem - 1], dtype=np.float64)
        cdl_m_c64 = np.ascontiguousarray(cdl_m_np, dtype=np.float64)
        cdl2_last_c64 = np.ascontiguousarray(cdl2_last_np, dtype=np.float64)
        vib_c_args = _VibratoGammaArgsC(
            d_assets=d_assets,
            m_rem=m_rem,
            n_paths_total=n_total,
            vib_step_start=int(vib_step_start_c),
            vib_n_steps=int(vib_n_steps_c),
            perf_mat=perf_base_m_d_n_np.ctypes.data,
            lx_pre=cpu_vib_lx_pre.ctypes.data,
            j_pre=cpu_vib_j_pre.ctypes.data,
            sig_pre=cpu_vib_sig_pre.ctypes.data,
            sv_vib=z_vib_c64.ctypes.data,
            chol_mat=ch_vib_c64.ctypes.data,
            l_inv_t=l_inv_t_cpu.ctypes.data,
            c_inv=c_inv_cpu.ctypes.data,
            fwd_term=fwd_term_c64.ctypes.data,
            cdl_m=cdl_m_c64.ctypes.data,
            cdl2_last=cdl2_last_c64.ctypes.data,
            ko_barriers=ko_arr_np.ctypes.data,
            is_callable=is_call_np.ctypes.data,
            cpn_barriers=cpn_arr_np.ctypes.data,
            is_guaranteed_cpn=is_guar_np.ctypes.data,
            coupon_rates=cpn_rates_np.ctypes.data,
            eps_vec=eps_cpu_vib.ctypes.data,
            inv_lex_perm=inv_perm_c64.ctypes.data,
            frozen_term_out=frozen_term_cpu.ctypes.data if m_rem > 1 else 0,
            h_vib_out=h_vib_out_c.ctypes.data,
            df_last=float(df_last_val_c),
            h_vib=float(h_vib_val_c),
            ko_spread=float(ko_spread),
            cpn_spread=float(cpn_spread),
            ki_spread=float(ki_spread),
            one_star_strike=float(k_os),
            one_star_eps=float(one_star_eps),
            strike=float(k_put),
            lower_strike=float(k_low),
            ki_barrier=float(b_ki),
            option_gearing=float(state.option_gearing),
            autocall_coupon=float(state.autocall_coupon),
            c_mem_0=float(state.c_mem_0),
            i_ki_past=float(state.i_ki_past),
            part_call_strike=float(k_call),
            part_call_rate=float(part_call_rate),
            one_star_coupon=float(os_cpn),
            call_code=int(call_code),
            cpn_code=int(cpn_code),
            opt_code=int(eff_opt_code_s2),
            has_ki=1 if has_ki else 0,
            ki_active=1 if (has_ki and b_ki > 0.0 and float(state.i_ki_past) < 0.5) else 0,
            has_one_star=1 if k_os > 0.0 else 0,
            is_memory=1 if is_memory else 0,
            is_long_basket=1 if is_long_flag else 0,
            no_notional=1 if no_notional else 0,
            force_cpn_ac=1 if force_cpn_ac else 0,
            cpn_dir_down=1 if dir_down else 0,
            cpn_type_out=1 if type_out else 0,
            has_lower_strike=1 if has_lower_strike_flag else 0,
            has_part_call=1 if has_part_call_flag else 0,
        )
        cpp_vib_v2_fn(ctypes.byref(vib_c_args))
        h_vib_np = h_vib_out_c
        if m_rem > 1:
            assert perf_sp_p_dev is not None and perf_sp_m_dev is not None
            if want_gamma:
                if d_assets > 1:
                    for i_v in range(d_assets):
                        for j_v in range(i_v + 1, d_assets):
                            cross_pairs_vib.append((i_v, j_v))
                n_pre_c = 2 * d_assets + 2 * len(cross_pairs_vib) + 1
                aa_pre_c = np.empty(n_pre_c, dtype=np.int32)
                ab_pre_c = np.full(n_pre_c, -1, dtype=np.int32)
                asl_pre_c = np.zeros(n_pre_c, dtype=np.int32)
                bsl_pre_c = np.full(n_pre_c, -1, dtype=np.int32)
                for i_v in range(d_assets):
                    aa_pre_c[2 * i_v] = i_v
                    aa_pre_c[2 * i_v + 1] = i_v
                    asl_pre_c[2 * i_v] = i_v
                    asl_pre_c[2 * i_v + 1] = d_assets + i_v
                for c_v, (i_v, j_v) in enumerate(cross_pairs_vib):
                    b0_v = 2 * d_assets + 2 * c_v
                    aa_pre_c[b0_v] = i_v
                    ab_pre_c[b0_v] = j_v
                    asl_pre_c[b0_v] = i_v
                    bsl_pre_c[b0_v] = j_v
                    aa_pre_c[b0_v + 1] = i_v
                    ab_pre_c[b0_v + 1] = j_v
                    asl_pre_c[b0_v + 1] = d_assets + i_v
                    bsl_pre_c[b0_v + 1] = d_assets + j_v
                aa_pre_c[-1] = -1
            else:
                aa_pre_c = np.array([-2, -2, -1], dtype=np.int32)
                ab_pre_c = np.array([-1, -1, -1], dtype=np.int32)
                asl_pre_c = np.array([0, 1, 0], dtype=np.int32)
                bsl_pre_c = np.full(3, -1, dtype=np.int32)
            pv_preterm_np = _eval_c2_preterm_cpu_np(
                perf_base_m_d_n_np,
                perf_sp_p_dev,
                perf_sp_m_dev,
                frozen_term_cpu,
                aa_pre=aa_pre_c,
                ab_pre=ab_pre_c,
                asl_pre=asl_pre_c,
                bsl_pre=bsl_pre_c,
                ko_barriers=ko_arr_np,
                is_callable=is_call_np,
                cpn_barriers=cpn_arr_np,
                is_guar=is_guar_np,
                cpn_rates=cpn_rates_np,
                df_pay=df_pay_np,
                w_fund=w_fund_base_np,
                eps_vec=eps_cpu_vib,
                call_code=call_code,
                cpn_code=cpn_code,
                ko_spread=ko_spread,
                cpn_spread=cpn_spread,
                cpn_dir_down=dir_down,
                cpn_type_out=type_out,
                is_memory=is_memory,
                force_cpn_ac=force_cpn_ac,
                no_notional=no_notional,
                ac_cpn=float(state.autocall_coupon),
                c_mem_0=float(state.c_mem_0),
                want_gamma=want_gamma,
            )

    # Step 4: Rho & RepoRho with vectorized curve/funding/in-transit pillar evaluation.
    if not _warm_gpu_hit:
        rho_tasks = []
        repo_task_meta = None
        div_task_meta = None
        fxc_tasks = []
    h_div = 1.0e-3
    if (not _warm_gpu_hit) and want_rho:
        rho_dict = {}
        rho_arrays = {}
        fut_obs_dts = state.seasoned_state.future_obs_dates
        fut_pay_dts = state.seasoned_state.future_pay_dates
        start_dt_state = state.seasoned_state.start_date
        f_dcc_state = state.seasoned_state.day_count_convention

        fut_obs_dts_t = tuple(fut_obs_dts)
        fut_pay_dts_t = tuple(fut_pay_dts)
        local_c_norm_meta: Dict[Any, Tuple[Any, Any, Any, Any]] = {}

        def _curve_sig_local(c_any: Any) -> Any:
            return _curve_fast_sig(c_any, curve_sig_memo)

        def _curve_mult_mat_np(c_any: Any, dts: Sequence[Any], n_pil: int) -> npt.NDArray[np.float64]:
            if len(dts) == 0 or n_pil <= 0:
                return np.ones((max(n_pil, 0), len(dts)), dtype=np.float64)
            c_key_loc = _curve_sig_local(c_any)
            meta = local_c_norm_meta.get(c_key_loc)
            if meta is None:
                c_n = _normalize_curve_obj(c_any, as_of_date=as_of_orig)
                q_arr = getattr(
                    c_n,
                    "_raw_quotes",
                    getattr(c_n, "quotes", getattr(c_n, "values", getattr(c_n, "rates", None))),
                )
                sp_c = getattr(c_n, "space_coords", "Rates")
                in_m = getattr(c_n, "_input_mode", None)
                q_sc = _curve_quote_scale(c_n)
                t_arr = getattr(c_n, "tenors", None)
                q_b = np.ascontiguousarray(q_arr, dtype=np.float64).tobytes() if q_arr is not None else b""
                t_b = (
                    np.ascontiguousarray(t_arr).tobytes()
                    if isinstance(t_arr, np.ndarray)
                    else repr(t_arr).encode("utf-8")
                )
                c_dig: Any = (t_b, q_b, str(getattr(c_n, "dcc", "")), str(sp_c), str(in_m or ""), float(q_sc))
                meta = (c_n, c_dig, sp_c, in_m)
                local_c_norm_meta[c_key_loc] = meta
                local_c_norm_meta[_curve_sig_local(c_n)] = meta
            else:
                c_n, c_dig, sp_c, in_m = meta
            dts_key = dts if isinstance(dts, tuple) else tuple(dts)
            cj_key = (c_dig, dts_key, int(n_pil), float(h_rho), as_of_orig)
            cached_mult = _CRN_CURVE_JAC_CACHE.get(cj_key)
            if cached_mult is not None:
                return cast(npt.NDArray[np.float64], cached_mult)
            # Evaluate on CPU so warm Rho jacobian lookups avoid CUDA sync stalls.
            jac_t = _curve_df_delta_jacobian(c_n, dts, n_pil, torch.device("cpu"), as_of_date=as_of_orig)
            jac_np = jac_t.detach().cpu().numpy().T  # (n_pil, len(dts))
            if in_m in ("rates", "log_df") or (in_m is None and sp_c in ("Rates", "LogarithmOfDiscountFactors")):
                mult_np = np.asarray(np.exp(h_rho * jac_np), dtype=np.float64)
            else:
                mult_np = np.asarray(np.maximum(1.0 + h_rho * jac_np, 1e-15), dtype=np.float64)
            _CRN_CURVE_JAC_CACHE.put(cj_key, mult_np)
            return mult_np

        f_curve_leg = (
            _normalize_curve_obj(state.funding_leg.get("funding_curve"), as_of_date=as_of_orig)
            if state.funding_leg is not None
            else None
        )
        f_cid_raw = (state.funding_leg or {}).get("funding_curve_id") if state.funding_leg is not None else None
        fin_curve_id = str(getattr(f_curve_leg, "identifier", f_cid_raw or disc_id)) if f_curve_leg is not None else None

        asset_fin_curves_norm: dict[str, Any] = {}
        asset_fin_cids: dict[str, str] = {}
        for ticker in sorted_ids:
            raw_c = (state.asset_financing_curves or {}).get(ticker)
            c_obj = _normalize_curve_obj(raw_c, as_of_date=as_of_orig) if raw_c is not None else disc_curve
            if c_obj is None:
                c_obj = disc_curve
            cid_t = str(getattr(c_obj, "identifier", getattr(c_obj, "Identifier", disc_id)) or disc_id)
            asset_fin_curves_norm[ticker] = c_obj
            asset_fin_cids[ticker] = cid_t

        ordered_curves: list[tuple[str, Any]] = []
        seen_cids: set[str] = set()
        if disc_curve is not None:
            ordered_curves.append((disc_id, disc_curve))
            seen_cids.add(disc_id)
        if f_curve_leg is not None and fin_curve_id is not None and fin_curve_id not in seen_cids:
            ordered_curves.append((fin_curve_id, f_curve_leg))
            seen_cids.add(fin_curve_id)
        for ticker in sorted_ids:
            cid_t = asset_fin_cids[ticker]
            c_obj = asset_fin_curves_norm[ticker]
            if c_obj is not None and cid_t not in seen_cids:
                ordered_curves.append((cid_t, c_obj))
                seen_cids.add(cid_t)
        for cid_b, n_pil_b in _collect_unreferenced_bundle_curves(request, seen_cids):
            rho_arrays[cid_b] = np.zeros(n_pil_b, dtype=np.float64)
            rho_dict[cid_b] = [0.0] * n_pil_b

        # Precompute vectorized funding schedule descriptors once across all curve pillars.
        fund_periods_meta: List[Tuple[int, float, float, float, Optional[float], Any, Any, float]] = []
        if w_fund_base_np is not None and state.funding_leg is not None:
            f_leg = state.funding_leg
            f_rid = f_leg.get("rate_identifier")
            f_spread = float(f_leg.get("funding_spread") or 0.0)
            f_dates = f_leg.get("payment_dates", ())
            f_fixings = f_leg.get("funding_fixings", ())
            prev_f_dt = start_dt_state
            same_f_disc = (
                f_curve_leg is None
                or f_curve_leg is state.discount_curve
                or (fin_curve_id is not None and fin_curve_id == disc_id)
            )
            eff_f_curve = disc_curve if same_f_disc else f_curve_leg
            for f_dt in f_dates:
                if f_dt <= as_of_orig:
                    prev_f_dt = f_dt
                    continue
                alpha_j = _year_fraction(prev_f_dt, f_dt, f_dcc_state)
                m_idx = m_rem - 1
                for m_p in range(m_rem):
                    if fut_pay_dts[m_p] >= f_dt:
                        m_idx = m_p
                        break
                fixed_r: Optional[float] = None
                if f_rid is None:
                    fixed_r = 0.0
                elif prev_f_dt < as_of_orig and f_fixings:
                    t_date = prev_f_dt.date()
                    for fx_d, fx_v in f_fixings:
                        if fx_d == prev_f_dt:
                            fixed_r = float(fx_v)
                            break
                        if fx_d.date() == t_date and fixed_r is None:
                            fixed_r = float(fx_v)
                alpha_rem = max(_year_fraction(as_of_orig, f_dt, f_dcc_state), 1e-8) if prev_f_dt < as_of_orig else max(alpha_j, 1e-8)
                fund_periods_meta.append((m_idx, alpha_j, f_spread, 1.0, fixed_r, prev_f_dt, f_dt, alpha_rem))
                _ = eff_f_curve
                prev_f_dt = f_dt

        # Precompute vectorized In-Transit cashflow schedule once.
        in_tr_dts: List[Any] = []
        in_tr_amts: List[float] = []
        past_cfs = getattr(state.seasoned_state, "past_cashflows", ())
        estimate_cfs = [cf for cf in past_cfs if getattr(cf, "StatusText", "") == "Estimate"]
        if estimate_cfs:
            for cf in estimate_cfs:
                if abs(float(cf.Amount)) > 0.0:
                    in_tr_dts.append(cf.Date)
                    in_tr_amts.append(float(cf.Amount))
        elif state.seasoned_state.in_transit_indices:
            for rec in state.seasoned_state.historical_records:
                if rec.status == "In-Transit" and (rec.coupon_paid != 0.0 or rec.notional_paid != 0.0):
                    in_tr_dts.append(rec.payment_date)
                    in_tr_amts.append(float(rec.discount_factor * (rec.coupon_paid + rec.notional_paid)))

        rho_tasks = []
        repo_task_meta = None
        for cid_cur, c_obj_cur in ordered_curves:
            n_pil_c = len(c_obj_cur.tenors) if (c_obj_cur is not None and hasattr(c_obj_cur, "tenors")) else 0
            if n_pil_c <= 0:
                rho_arrays[cid_cur] = np.zeros(0, dtype=np.float64)
                rho_dict[cid_cur] = []
                continue

            is_disc_c = cid_cur == disc_id
            is_fund_c = fin_curve_id is not None and cid_cur == fin_curve_id
            affected_asset_indices = [idx_a for idx_a, tk in enumerate(sorted_ids) if asset_fin_cids[tk] == cid_cur]

            m_obs_mat = _curve_mult_mat_np(c_obj_cur, fut_obs_dts_t, n_pil_c) if affected_asset_indices else np.ones((n_pil_c, m_rem), dtype=np.float64)
            m_pay_mat = _curve_mult_mat_np(c_obj_cur, fut_pay_dts_t, n_pil_c) if is_disc_c else np.ones((n_pil_c, m_rem), dtype=np.float64)

            if is_disc_c and in_tr_dts:
                m_it_mat = _curve_mult_mat_np(c_obj_cur, in_tr_dts, n_pil_c)
                in_tr_rho_b = np.asarray(m_it_mat @ np.asarray(in_tr_amts, dtype=np.float64), dtype=np.float64)
            else:
                in_tr_rho_b = np.full(n_pil_c, float(state.pv_in_transit), dtype=np.float64)

            if w_fund_base_np is not None and state.funding_leg is not None and (is_disc_c or is_fund_c) and fund_periods_meta:
                w_fund_rho_b = np.zeros((n_pil_c, m_rem), dtype=np.float64)
                f_dts_all = [x[6] for x in fund_periods_meta]
                p_dts_idx = [j for j, x in enumerate(fund_periods_meta) if x[5] >= as_of_orig and x[4] is None]
                df_d0_all = _curve_df_cached_np(disc_curve, f_dts_all) if (disc_curve is not None and hasattr(disc_curve, "df")) else np.ones(len(f_dts_all))
                m_d_all = _curve_mult_mat_np(disc_curve, f_dts_all, n_pil_c) if is_disc_c else np.ones((n_pil_c, len(f_dts_all)))
                df_d_bumped = df_d0_all[None, :] * m_d_all

                eff_fc = disc_curve if (fin_curve_id == disc_id or f_curve_leg is None) else f_curve_leg
                bump_fwd = (is_fund_c and f_curve_leg is not None) or (is_disc_c and (f_curve_leg is None or fin_curve_id == disc_id))
                df_fc0_all = _curve_df_cached_np(eff_fc, f_dts_all) if (eff_fc is not None and hasattr(eff_fc, "df")) else np.ones(len(f_dts_all))
                m_f_all = _curve_mult_mat_np(eff_fc, f_dts_all, n_pil_c) if bump_fwd else np.ones((n_pil_c, len(f_dts_all)))
                df_fc_bumped = df_fc0_all[None, :] * m_f_all

                if p_dts_idx:
                    prev_dts_sub = [fund_periods_meta[j][5] for j in p_dts_idx]
                    df_fp0_sub = _curve_df_cached_np(eff_fc, prev_dts_sub) if (eff_fc is not None and hasattr(eff_fc, "df")) else np.ones(len(prev_dts_sub))
                    m_fp_sub = _curve_mult_mat_np(eff_fc, prev_dts_sub, n_pil_c) if bump_fwd else np.ones((n_pil_c, len(prev_dts_sub)))
                    df_fp_map = {j: df_fp0_sub[k] * m_fp_sub[:, k] for k, j in enumerate(p_dts_idx)}
                else:
                    df_fp_map = {}

                for j, (m_idx, alpha_j, f_sp, _, fixed_r, prev_f_dt, _, alpha_rem) in enumerate(fund_periods_meta):
                    if fixed_r is not None:
                        r_fwd_v = np.full(n_pil_c, fixed_r, dtype=np.float64)
                    elif prev_f_dt < as_of_orig:
                        r_fwd_v = (1.0 / np.maximum(df_fc_bumped[:, j], 1e-12) - 1.0) / alpha_rem
                    else:
                        r_fwd_v = (df_fp_map[j] / np.maximum(df_fc_bumped[:, j], 1e-12) - 1.0) / alpha_rem
                    w_fund_rho_b[:, m_idx] += df_d_bumped[:, j] * (r_fwd_v + f_sp) * alpha_j
            elif w_fund_base_np is not None:
                w_fund_rho_b = np.broadcast_to(w_fund_base_np[None, :], (n_pil_c, m_rem)).copy()
            else:
                w_fund_rho_b = None

            df_pay_rho = df_pay_np[None, :] * m_pay_mat
            inv_m_obs = 1.0 / np.maximum(m_obs_mat, 1e-12)
            chunk_toks: List[Tuple[Tuple[int, int, int], int, int]] = []

            if use_pathwise_rate_adj:
                d_df_rho = df_pay_rho - df_pay_np[None, :]
                d_wf_rho = (
                    (w_fund_rho_b - w_fund_base_np[None, :])
                    if (w_fund_rho_b is not None and w_fund_base_np is not None)
                    else None
                )
                d_itr_rho = in_tr_rho_b - float(state.pv_in_transit)
                d_lf_list: List[Tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
                d_rn_a_base = np.log(inv_m_obs)
                for idx_a in affected_asset_indices:
                    tk_a = sorted_ids[idx_a]
                    events_a = state.dividend_schedule.dividend_events.get(tk_a, [])
                    if len(events_a) > 0:
                        amt_a, w_a, inc_a = _get_div_np(tk_a)
                        pay_dts_a = [ev.payment_date for ev in events_a]
                        active_ex_a = np.array([1.0 if ev.ex_date > as_of_orig else 0.0 for ev in events_a], dtype=np.float64)
                        m_div_pay_mat = _curve_mult_mat_np(c_obj_cur, pay_dts_a, n_pil_c)
                        w_bumped_mat = w_a[None, :] * np.where(active_ex_a[None, :] > 0.5, m_div_pay_mat, 1.0)
                        dpv_a_mat = (w_bumped_mat * amt_a[None, :]) @ inc_a
                        esc_a_mat = _escrowed_spot_softplus_np(np.full((n_pil_c, 1), spots_np[idx_a]), dpv_a_mat)
                        fwd_a_mat = esc_a_mat * (growth_base[idx_a][None, :] * inv_m_obs) / ref_spots_np[idx_a]
                        d_lf_a = np.log(np.maximum(fwd_a_mat / fwd_perf_np[idx_a][None, :], 1e-30))
                        d_dpv_a = dpv_a_mat - div_pv_np[idx_a][None, :]
                    else:
                        d_lf_a = d_rn_a_base
                        d_dpv_a = np.zeros_like(d_rn_a_base)
                    d_lf_list.append((idx_a, d_lf_a, d_rn_a_base, d_dpv_a))
                adj_rho_tasks.append((cid_cur, c_obj_cur, n_pil_c, d_df_rho, d_wf_rho, d_itr_rho, d_lf_list))
            elif len(affected_asset_indices) == 1:
                only_a = affected_asset_indices[0]
                tk_a = sorted_ids[only_a]
                events_a = state.dividend_schedule.dividend_events.get(tk_a, [])
                if len(events_a) > 0:
                    amt_a, w_a, inc_a = _get_div_np(tk_a)
                    pay_dts_a = [ev.payment_date for ev in events_a]
                    active_ex_a = np.array([1.0 if ev.ex_date > as_of_orig else 0.0 for ev in events_a], dtype=np.float64)
                    m_div_pay_mat = _curve_mult_mat_np(c_obj_cur, pay_dts_a, n_pil_c)
                    w_bumped_mat = w_a[None, :] * np.where(active_ex_a[None, :] > 0.5, m_div_pay_mat, 1.0)
                    dpv_a_mat = (w_bumped_mat * amt_a[None, :]) @ inc_a
                    esc_a_mat = _escrowed_spot_softplus_np(np.full((n_pil_c, 1), spots_np[only_a]), dpv_a_mat)
                    fwd_only_a = esc_a_mat * (growth_base[only_a][None, :] * inv_m_obs) / ref_spots_np[only_a]
                    can_sc_1 = bool(use_gpu_triton and np.all(fwd_perf_np[only_a] > 1e-12))
                    sc_only_a: Optional[npt.NDArray[np.float64]] = (
                        (fwd_only_a / fwd_perf_np[only_a][None, :]) if can_sc_1 else None
                    )
                else:
                    fwd_only_a = fwd_perf_np[only_a][None, :] * inv_m_obs
                    can_sc_1 = bool(use_gpu_triton)
                    sc_only_a = inv_m_obs if can_sc_1 else None
                if can_sc_1 and sc_only_a is not None:
                    sc_rho = np.ones((n_pil_c, d_assets, m_rem), dtype=np.float64)
                    sc_rho[:, only_a, :] = sc_only_a
                    tok_r = _submit_payoff(
                        np.full(n_pil_c, -1, dtype=np.int32),
                        np.full(n_pil_c, -1, dtype=np.int32),
                        scale_mat=sc_rho,
                        df_rows=df_pay_rho,
                        wf_rows=w_fund_rho_b,
                        itr_arr=in_tr_rho_b,
                    )
                else:
                    if use_gpu_triton:
                        tok_r = _submit_payoff(
                            np.full(n_pil_c, only_a, dtype=np.int32),
                            np.full(n_pil_c, -1, dtype=np.int32),
                            df_rows=df_pay_rho,
                            wf_rows=w_fund_rho_b,
                            itr_arr=in_tr_rho_b,
                            fwd_bump_a_np=fwd_only_a,
                        )
                    else:
                        bump_r_a = norm_x_base_dev[only_a][None, :, :] * fwd_only_a[:, :, None]
                        tok_r = _submit_payoff(
                            np.full(n_pil_c, only_a, dtype=np.int32),
                            np.full(n_pil_c, -1, dtype=np.int32),
                            bump_a=bump_r_a,
                            df_rows=df_pay_rho,
                            wf_rows=w_fund_rho_b,
                            itr_arr=in_tr_rho_b,
                        )
                chunk_toks.append((tok_r, 0, n_pil_c))
            elif len(affected_asset_indices) == 0:
                tok_r = _submit_payoff(
                    np.full(n_pil_c, -1, dtype=np.int32),
                    np.full(n_pil_c, -1, dtype=np.int32),
                    df_rows=df_pay_rho,
                    wf_rows=w_fund_rho_b,
                    itr_arr=in_tr_rho_b,
                )
                chunk_toks.append((tok_r, 0, n_pil_c))
            else:
                fwd_rho = np.broadcast_to(fwd_perf_np[None, :, :], (n_pil_c, d_assets, m_rem)).copy()
                sc_rho_multi: Optional[npt.NDArray[np.float64]] = (
                    np.ones((n_pil_c, d_assets, m_rem), dtype=np.float64) if use_gpu_triton else None
                )
                can_sc_m = bool(use_gpu_triton)
                for idx_a in affected_asset_indices:
                    tk_a = sorted_ids[idx_a]
                    events_a = state.dividend_schedule.dividend_events.get(tk_a, [])
                    if len(events_a) > 0:
                        amt_a, w_a, inc_a = _get_div_np(tk_a)
                        pay_dts_a = [ev.payment_date for ev in events_a]
                        active_ex_a = np.array([1.0 if ev.ex_date > as_of_orig else 0.0 for ev in events_a], dtype=np.float64)
                        m_div_pay_mat = _curve_mult_mat_np(c_obj_cur, pay_dts_a, n_pil_c)
                        w_bumped_mat = w_a[None, :] * np.where(active_ex_a[None, :] > 0.5, m_div_pay_mat, 1.0)
                        dpv_a_mat = (w_bumped_mat * amt_a[None, :]) @ inc_a
                        esc_a_mat = _escrowed_spot_softplus_np(np.full((n_pil_c, 1), spots_np[idx_a]), dpv_a_mat)
                        fwd_a_mat = esc_a_mat * (growth_base[idx_a][None, :] * inv_m_obs) / ref_spots_np[idx_a]
                        fwd_rho[:, idx_a, :] = fwd_a_mat
                        if sc_rho_multi is not None and np.all(fwd_perf_np[idx_a] > 1e-12):
                            sc_rho_multi[:, idx_a, :] = fwd_a_mat / fwd_perf_np[idx_a][None, :]
                        else:
                            can_sc_m = False
                    else:
                        fwd_rho[:, idx_a, :] = fwd_perf_np[idx_a][None, :] * inv_m_obs
                        if sc_rho_multi is not None:
                            sc_rho_multi[:, idx_a, :] = inv_m_obs

                if can_sc_m and sc_rho_multi is not None:
                    tok_r = _submit_payoff(
                        np.full(n_pil_c, -1, dtype=np.int32),
                        np.full(n_pil_c, -1, dtype=np.int32),
                        scale_mat=sc_rho_multi,
                        df_rows=df_pay_rho,
                        wf_rows=w_fund_rho_b,
                        itr_arr=in_tr_rho_b,
                    )
                    chunk_toks.append((tok_r, 0, n_pil_c))
                else:
                    max_chunk_p = max(1, (256 * 1024 * 1024) // max(m_rem * d_assets * n_total * 8, 1))
                    for p_start in range(0, n_pil_c, max_chunk_p):
                        p_end = min(n_pil_c, p_start + max_chunk_p)
                        nb_p = p_end - p_start
                        if use_gpu_triton:
                            tok_r = _submit_payoff(
                                np.full(nb_p, -2, dtype=np.int32),
                                np.full(nb_p, -1, dtype=np.int32),
                                df_rows=df_pay_rho[p_start:p_end],
                                wf_rows=w_fund_rho_b[p_start:p_end] if w_fund_rho_b is not None else None,
                                itr_arr=in_tr_rho_b[p_start:p_end],
                                fwd_alt_np=fwd_rho[p_start:p_end],
                            )
                        else:
                            p_r_batch = np.ascontiguousarray(
                                np.transpose(norm_x_base_dev[None, :, :, :] * fwd_rho[p_start:p_end, :, :, None], (0, 2, 1, 3)),
                                dtype=np.float64,
                            )
                            tok_r = _submit_payoff(
                                np.full(nb_p, -2, dtype=np.int32),
                                np.full(nb_p, -1, dtype=np.int32),
                                alt_perf=p_r_batch,
                                df_rows=df_pay_rho[p_start:p_end],
                                wf_rows=w_fund_rho_b[p_start:p_end] if w_fund_rho_b is not None else None,
                                itr_arr=in_tr_rho_b[p_start:p_end],
                            )
                        chunk_toks.append((tok_r, p_start, p_end))

            rho_tasks.append((cid_cur, c_obj_cur, n_pil_c, chunk_toks)) if not use_pathwise_rate_adj else None

        repo_curves_map = repo_curves_map_shared or {}
        if repo_curves_map:
            repo_rho_dict = {}
            repo_fwd_parts: List[npt.NDArray[np.float64]] = []
            repo_aa_parts: List[npt.NDArray[np.int32]] = []
            repo_meta: List[Tuple[str, Any, int]] = []
            for i, ticker in enumerate(sorted_ids):
                r_curve_i = repo_curves_map.get(ticker)
                if r_curve_i is None:
                    continue
                r_curve_norm = _normalize_curve_obj(r_curve_i, as_of_date=as_of_orig)
                r_curve_eff = getattr(r_curve_norm, "_inner_curve", None) or r_curve_norm
                n_pil_r = len(r_curve_eff.tenors) if (r_curve_eff is not None and hasattr(r_curve_eff, "tenors")) else 0
                if n_pil_r <= 0:
                    continue
                m_repo_obs_mat = _curve_mult_mat_np(r_curve_eff, fut_obs_dts, n_pil_r)
                events_i = state.dividend_schedule.dividend_events.get(ticker, [])
                if len(events_i) > 0:
                    amt_i, w_i, inc_i = _get_div_np(ticker)
                    ex_dts_i = [ev.ex_date for ev in events_i]
                    active_ex_i = np.array([1.0 if ev.ex_date > as_of_orig else 0.0 for ev in events_i], dtype=np.float64)
                    m_div_ex_mat = _curve_mult_mat_np(r_curve_eff, ex_dts_i, n_pil_r)
                    w_bumped_mat = w_i[None, :] / np.where(active_ex_i[None, :] > 0.5, np.maximum(m_div_ex_mat, 1e-12), 1.0)
                    dpv_i_mat = (w_bumped_mat * amt_i[None, :]) @ inc_i
                    esc_i_mat = _escrowed_spot_softplus_np(np.full((n_pil_r, 1), spots_np[i]), dpv_i_mat)
                    fwd_repo_i = esc_i_mat * (growth_base[i][None, :] * m_repo_obs_mat) / ref_spots_np[i]
                else:
                    fwd_repo_i = fwd_perf_np[i, :][None, :] * m_repo_obs_mat

                if use_pathwise_rate_adj:
                    d_rn_repo_i = np.log(np.maximum(m_repo_obs_mat, 1e-12))
                    if len(events_i) > 0:
                        d_lf_repo_i = np.log(np.maximum(fwd_repo_i / fwd_perf_np[i][None, :], 1e-30))
                        d_dpv_repo_i = dpv_i_mat - div_pv_np[i][None, :]
                    else:
                        d_lf_repo_i = d_rn_repo_i
                        d_dpv_repo_i = np.zeros_like(d_rn_repo_i)
                    adj_repo_tasks.append((ticker, r_curve_eff, n_pil_r, i, d_lf_repo_i, d_rn_repo_i, d_dpv_repo_i))
                else:
                    repo_fwd_parts.append(fwd_repo_i)
                    repo_aa_parts.append(np.full(n_pil_r, i, dtype=np.int32))
                    repo_meta.append((ticker, r_curve_eff, n_pil_r))

            if repo_fwd_parts:
                aa_repo_all = np.concatenate(repo_aa_parts, axis=0)
                fwd_repo_all_np = np.concatenate(repo_fwd_parts, axis=0)
                n_rep_tot = int(len(aa_repo_all))
                if use_gpu_triton and np.all(fwd_perf_np[aa_repo_all] > 1e-12):
                    sc_repo = np.ones((n_rep_tot, d_assets, m_rem), dtype=np.float64)
                    sc_repo[np.arange(n_rep_tot), aa_repo_all, :] = fwd_repo_all_np / fwd_perf_np[aa_repo_all]
                    tok_repo = _submit_payoff(
                        np.full(n_rep_tot, -1, dtype=np.int32),
                        np.full(n_rep_tot, -1, dtype=np.int32),
                        scale_mat=sc_repo,
                    )
                else:
                    if use_gpu_triton:
                        tok_repo = _submit_payoff(
                            aa_repo_all,
                            np.full(n_rep_tot, -1, dtype=np.int32),
                            fwd_bump_a_np=fwd_repo_all_np,
                        )
                    else:
                        repo_all = norm_x_base_dev[aa_repo_all] * fwd_repo_all_np[:, :, None]
                        tok_repo = _submit_payoff(
                            aa_repo_all,
                            np.full(n_rep_tot, -1, dtype=np.int32),
                            bump_a=repo_all,
                        )
                repo_task_meta = (tok_repo, repo_meta)

    # Step 5: Enqueue DividendsSensitivities via unified single-asset LOO CRN batch.
    if (not _warm_gpu_hit) and want_divs:
        divs_dict = {}
        divs_arrays = {}
        div_fwd_parts: List[npt.NDArray[np.float64]] = []
        div_dpv_parts: List[npt.NDArray[np.float64]] = []
        div_aa_parts: List[npt.NDArray[np.int32]] = []
        div_slices: List[Tuple[str, int]] = []
        for i_s, ticker in enumerate(sorted_ids):
            events_i = state.dividend_schedule.dividend_events.get(ticker, [])
            n_ev = len(events_i)
            if n_ev > 0:
                _, w_base, inc = _get_div_np(ticker)
                d_dpv_i = h_div * w_base[:, None] * inc
                dpv_mat = div_pv_np[i_s][None, :] + d_dpv_i
                esc_mat = _escrowed_spot_softplus_np(np.full((n_ev, 1), spots_np[i_s]), dpv_mat)
                fwd_b_mat = (esc_mat * growth_base[i_s][None, :]) / ref_spots_np[i_s]
                div_fwd_parts.append(fwd_b_mat)
                div_dpv_parts.append(d_dpv_i)
                div_aa_parts.append(np.full(n_ev, i_s, dtype=np.int32))
            div_slices.append((ticker, n_ev))

        if div_fwd_parts:
            aa_div_all = np.concatenate(div_aa_parts, axis=0)
            fwd_div_all_np = np.concatenate(div_fwd_parts, axis=0)
            d_dpv_div_all = np.concatenate(div_dpv_parts, axis=0)
            n_div_tot = int(len(aa_div_all))
            if use_pathwise_rate_adj:
                d_lf_div = np.log(np.maximum(fwd_div_all_np / fwd_perf_np[aa_div_all], 1e-30))
                adj_div_task = (aa_div_all, d_lf_div, d_dpv_div_all, div_slices)
            elif use_gpu_triton and np.all(fwd_perf_np[aa_div_all] > 1e-12):
                sc_div = np.ones((n_div_tot, d_assets, m_rem), dtype=np.float64)
                sc_div[np.arange(n_div_tot), aa_div_all, :] = fwd_div_all_np / fwd_perf_np[aa_div_all]
                tok_divs = _submit_payoff(
                    np.full(n_div_tot, -1, dtype=np.int32),
                    np.full(n_div_tot, -1, dtype=np.int32),
                    scale_mat=sc_div,
                )
                div_task_meta = (tok_divs, div_slices)
            else:
                if use_gpu_triton:
                    tok_divs = _submit_payoff(
                        aa_div_all,
                        np.full(n_div_tot, -1, dtype=np.int32),
                        fwd_bump_a_np=fwd_div_all_np,
                    )
                else:
                    div_all = norm_x_base_dev[aa_div_all] * fwd_div_all_np[:, :, None]
                    tok_divs = _submit_payoff(
                        aa_div_all,
                        np.full(n_div_tot, -1, dtype=np.int32),
                        bump_a=div_all,
                    )
                div_task_meta = (tok_divs, div_slices)
        else:
            for ticker, _ in div_slices:
                divs_arrays[ticker] = np.zeros(0, dtype=np.float64)
            for ticker in orig_ids:
                arr_d = divs_arrays.get(ticker, np.zeros(0, dtype=np.float64))
                divs_arrays[ticker] = arr_d
                divs_dict[ticker] = [float(x) for x in arr_d]

    # Step 6: Enqueue FXCega before chunked GPU payoff execution.
    if (not _warm_gpu_hit) and want_cega and quanto_meta:
        fx_cega_dict = {}
        h_fxc = 1.0e-2
        for i, ticker in enumerate(sorted_ids):
            qm = quanto_meta.get(ticker)
            if not qm:
                continue
            fx_id = str(qm.get("fx_id", "FX"))
            fx_sign_f = _resolve_quanto_fx_sign(qm, state.payment_currency)
            sig_fx_q = np.asarray(qm.get("sig_fx_obs", [0.0] * m_rem), dtype=np.float64)
            sig_eq_q = _get_vol_atm_obs(i)
            if use_pathwise_rate_adj:
                adj_fxc_tasks.append((ticker, fx_id, i, fx_sign_f, sig_eq_q, sig_fx_q))
                continue
            mult_fxc_p = np.exp(-fx_sign_f * h_fxc * sig_eq_q * sig_fx_q * tau_obs_np)
            mult_fxc_m = np.exp(fx_sign_f * h_fxc * sig_eq_q * sig_fx_q * tau_obs_np)
            sc_fxc = np.ones((2, d_assets, m_rem), dtype=np.float64)
            sc_fxc[0, i, :] = mult_fxc_p
            sc_fxc[1, i, :] = mult_fxc_m
            tok_fxc = _submit_payoff(
                np.full(2, -1, dtype=np.int32),
                np.full(2, -1, dtype=np.int32),
                scale_mat=sc_fxc,
            )
            fxc_tasks.append((ticker, fx_id, tok_fxc, h_fxc))

    # Execute unified chunked GPU Monte Carlo + AAD + Vibrato + PreTerm + Theta and sync via one D2H copy.
    if use_gpu_triton:
        th_chunk_in_step1 = bool(
            th_prep is not None and not th_prep["eff_autocalled"] and int(th_prep["k_cross"]) < m_rem
        )
        gpu_need_chunk_loop = bool(
            need_stage2_aad
            or use_vibrato_gamma
            or gpu_eval_rows_total > 0
            or emit_stage1_gpu
            or th_chunk_in_step1
        )
        if gpu_need_chunk_loop:
            assert qmc_device is not None
            assert sched_f32_g is not None and lv_f32_g_cached is not None and chol_f32_g_cached is not None
            assert chol_f64_g is not None and tau_step_g is not None and rn_step_g is not None and rt_step_g is not None
            assert y_min_base_g is not None and inv_dy_base_g is not None and inv_dt_base_g is not None and id_d_g is not None
            assert eps_aad_g is not None and fwd_perf_g is not None

            n_steps_tot_g = m_rem * n_sub
            t_mid_ptr_g = sched_f32_g[:n_steps_tot_g]
            dt_sub_ptr_g = sched_f32_g[n_steps_tot_g : 2 * n_steps_tot_g]
            sqrt_dt_ptr_g = sched_f32_g[2 * n_steps_tot_g : 3 * n_steps_tot_g]
            q_off_s = 3 * n_steps_tot_g
            q_len_s = d_assets * n_steps_tot_g
            q_shift_mid_ptr_g = sched_f32_g[q_off_s : q_off_s + q_len_s]
            y_min_f32_g = sched_f32_g[q_off_s + q_len_s : q_off_s + q_len_s + d_assets]
            inv_dy_f32_g = sched_f32_g[q_off_s + q_len_s + d_assets :]
            inv_dt_f32_val = _warm_gpu.inv_dt_f32_val if (_warm_gpu_hit and _warm_gpu is not None) else float(lv_nt - 1) / max(t_max, 1e-12)
            dummy_f32_1 = (
                _warm_gpu.dummy_f32_1
                if (_warm_gpu_hit and _warm_gpu is not None)
                else torch.empty((1,), dtype=torch.float32, device=qmc_device)
            )
            dummy_shifts_g = dummy_f32_1

            per_rep_bytes = max(
                m_rem * d_assets * (n_sub * 4 + 8 + (4 if need_stage2_aad else 0) + (8 if need_spot_bumps else 0)),
                1,
            ) * n_rep
            _r_env_cap = int(os.environ.get("DEEPVOL_R_CAP", "4"))
            r_chunk_cap = _r_env_cap if (d_assets * m_rem * n_sub >= 512) else eff_r
            r_chunk = min(eff_r, r_chunk_cap, max(1, (72 * 1024 * 1024) // per_rep_bytes))
            fuse_m_sde = bool(n_rep <= 1024 and m_rem <= 64)
            if n_rep <= 1024:
                block_n_sde = max(32, _next_power_of_two_int(n_rep))
                n_tiles_sde = 1
                nw_sde = 16 if block_n_sde >= 1024 else (8 if block_n_sde >= 512 else 4)
            else:
                block_n_sde = 256
                n_tiles_sde = (n_rep + block_n_sde - 1) // block_n_sde
                nw_sde = 8

            use_analytical_theta_plan = bool(
                want_theta
                and th_prep is not None
                and not th_prep["eff_autocalled"]
                and int(th_prep["k_cross"]) == 0
                and float(tau_obs_np[0]) > 2.0 * float(th_prep.get("dt_shift", 1.0 / 365.0))
                and not bool(th_prep.get("has_ex_div_in_roll", False))
                and not is_cont_ki_active
                and want_rate_aad_smooth
                and use_triton_aad_s2
            )
            if use_analytical_theta_plan:
                th_chunk_in_step1 = False
            need_fwd_ckpt = bool(use_f32_step1 or need_stage2_aad or use_vibrato_gamma)
            want_lv_aad_sde = bool(want_vega or ((want_rho or want_divs) and use_pathwise_rate_adj) or use_analytical_theta_plan)
            want_chol_aad_sde = bool(want_cega and d_assets > 1)
            want_q_aad_sde = bool((want_vega and quanto_meta) or (use_analytical_theta_plan and quanto_meta))
            want_cf_wf_aad = bool(need_stage2_aad and (use_pathwise_rate_adj or use_analytical_theta_plan))

            use_mf_cv = bool(
                cv_mode_cfg == "multifidelity_gbm"
                and can_triton_sobol
                and d_assets <= 16
                and m_rem * 2 * d_assets <= 21201
                and not is_cont_ki_active
                and call_code in _TRITON_PAYOFF_AAD_CODES
                and cpn_code in _TRITON_PAYOFF_AAD_CODES
                and eff_opt_code_s2 in _TRITON_PAYOFF_AAD_CODES
                and (emit_stage1_gpu or want_cf_wf_aad)
            )
            d_var_f32_g: Optional[torch.Tensor] = None
            fwd_perf_f32_g: Optional[torch.Tensor] = None
            df_pay_f32_g: Optional[torch.Tensor] = None
            w_fund_f32_g: Optional[torch.Tensor] = None
            ko_f32_g: Optional[torch.Tensor] = None
            is_call_f32_g: Optional[torch.Tensor] = None
            cpn_f32_g: Optional[torch.Tensor] = None
            is_guar_f32_g: Optional[torch.Tensor] = None
            cpn_rates_f32_g: Optional[torch.Tensor] = None
            eps_f32_g: Optional[torch.Tensor] = None
            payoff_f32_g: Optional[torch.Tensor] = None
            chol_scaled_f32_g: Optional[torch.Tensor] = None
            eff_r_mf = 0
            n_rep_mf = 0
            d_var_gbm_np: Optional[npt.NDArray[np.float32]] = None

            if _warm_gpu_hit:
                assert _warm_cpu is not None and _warm_gpu is not None
                mf_cheap_digest = _warm_cpu.mf_cheap_digest
                apply_ema_mf = _warm_cpu.apply_ema_mf
                ema_group_mf = _warm_cpu.ema_group_mf
                n_rep_mf = _warm_cpu.n_rep_mf
                n_rep_gen_mf = _warm_cpu.n_rep_gen_mf
                eff_r_mf = _warm_cpu.eff_r_mf
                d_var_f32_g = _warm_gpu.d_var_f32_g
                fwd_perf_f32_g = _warm_gpu.fwd_perf_f32_g
                df_pay_f32_g = _warm_gpu.df_pay_f32_g
                w_fund_f32_g = _warm_gpu.w_fund_f32_g
                ko_f32_g = _warm_gpu.ko_f32_g
                is_call_f32_g = _warm_gpu.is_call_f32_g
                cpn_f32_g = _warm_gpu.cpn_f32_g
                is_guar_f32_g = _warm_gpu.is_guar_f32_g
                cpn_rates_f32_g = _warm_gpu.cpn_rates_f32_g
                eps_f32_g = _warm_gpu.eps_f32_g
                payoff_f32_g = _warm_gpu.payoff_f32_g
                chol_scaled_f32_g = _warm_gpu.chol_scaled_f32_g
                if use_mf_cv:
                    mf_cheap_hit = _CRN_MF_CHEAP_CACHE.get(mf_cheap_digest)
                    if mf_cheap_hit is None and can_triton_sobol and sv_sobol_gpu is None:
                        sv_sobol_gpu = _get_sobol_direction_numbers(qmc_device)
                else:
                    mf_cheap_hit = None
            elif use_mf_cv:
                d_var_gbm_np = _compute_mf_dvar_gbm_np(
                    state,
                    sorted_ids,
                    ref_spots_np,
                    fwd_perf_np,
                    tau_obs_np,
                    base_lv_tables,
                    lv_t_grid,
                    lv_y_grids,
                )
                d_var_f32_g = torch.as_tensor(d_var_gbm_np, dtype=torch.float32, device=qmc_device).contiguous()
                fwd_perf_f32_g = fwd_perf_g.to(dtype=torch.float32).contiguous()
                df_pay_f32_g = gpu_meta["df_pay"].to(dtype=torch.float32).contiguous()
                w_fund_f32_g = gpu_meta["w_fund"].to(dtype=torch.float32).contiguous() if gpu_meta["w_fund"] is not None else None
                ko_f32_g = gpu_meta["ko"].to(dtype=torch.float32).contiguous()
                is_call_f32_g = gpu_meta["is_call"].to(dtype=torch.float32).contiguous()
                cpn_f32_g = gpu_meta["cpn"].to(dtype=torch.float32).contiguous()
                is_guar_f32_g = gpu_meta["is_guar"].to(dtype=torch.float32).contiguous()
                cpn_rates_f32_g = gpu_meta["cpn_rates"].to(dtype=torch.float32).contiguous()
                eps_f32_g = eps_aad_g.to(dtype=torch.float32).contiguous()
                payoff_f32_g = gpu_meta["payoff_f64"].to(dtype=torch.float32).contiguous()
                chol_scaled_f32_g = chol_f32_g_cached * math.sqrt(float(n_sub))
                apply_ema_mf = bool(n_rep <= 1024 and os.environ.get("DEEPVOL_MF_EMA", "1") == "1")
                raw_n_cheap_gpu = getattr(cfg, "n_cheap_mf", None)
                n_cheap_cfg = max(
                    int(raw_n_cheap_gpu) if raw_n_cheap_gpu is not None else 65536,
                    4 * n_total if n_total >= 32768 else 65536,
                )
                if apply_ema_mf:
                    n_rep_mf = n_rep
                    ema_group_mf = min(16, max(1, 16384 // max(n_rep_mf, 1)))
                    n_rep_gen_mf = n_rep_mf * ema_group_mf
                    n_cheap_target = max(n_cheap_cfg, 262144)
                    eff_r_mf = max(16, (n_cheap_target + n_rep_gen_mf - 1) // n_rep_gen_mf)
                else:
                    ema_group_mf = 1
                    n_rep_mf = _next_power_of_two_int(min(max(n_rep, n_cheap_cfg // 4), 16384))
                    n_rep_gen_mf = n_rep_mf
                    n_cheap_target = max(n_cheap_cfg, 262144)
                    eff_r_mf = max(16, (n_cheap_target + n_rep_mf - 1) // n_rep_mf)
                wf_mf_src: Any = w_fund_base_np if w_fund_base_np is not None else gpu_meta["w_fund"]
                mf_cheap_digest = _content_digest(
                    d_var_gbm_np,
                    fwd_perf_np,
                    df_pay_np,
                    wf_mf_src,
                    ko_src,
                    ic_src,
                    cp_src,
                    ig_src,
                    cr_src,
                    corr_np,
                    payoff_s_np,
                    np.asarray(inv_perm, dtype=np.int64),
                    u_time_np,
                    chol_base,
                    eff_r_mf,
                    ema_group_mf,
                    n_rep_mf,
                    m_rem,
                    d_assets,
                    n_sub,
                    seed,
                    apply_ema_mf,
                    want_cf_wf_aad,
                    call_code,
                    cpn_code,
                    eff_opt_code_s2,
                    has_ki,
                    bool(has_ki and b_ki > 0.0 and float(state.i_ki_past) < 0.5),
                    bool(k_os > 0.0),
                    is_memory,
                    no_notional,
                    force_cpn_ac,
                    dir_down,
                    type_out,
                    has_lower_strike_flag,
                    has_part_call_flag,
                    float(barrier_eps_min),
                    float(barrier_eps_max),
                )
                mf_cheap_hit = _CRN_MF_CHEAP_CACHE.get(mf_cheap_digest)
            else:
                mf_cheap_digest = ""
                mf_cheap_hit = None
                apply_ema_mf = False
                ema_group_mf = 1
                n_rep_gen_mf = 0

            def _run_mf_gbm_batch(
                z_coarse_corr_in: torch.Tensor,
                s1_out_g: torch.Tensor,
                cf_surv_out_g: Optional[torch.Tensor],
                *,
                antithetic: bool = False,
                layout_mdn: bool = False,
                r_b_ov: Optional[int] = None,
                n_rep_ov: Optional[int] = None,
                z_scale_val: float = 1.0,
                pre_zeroed: bool = False,
                d_pad_ov: Optional[int] = None,
                m_pad_ov: Optional[int] = None,
                d_var_ov: Optional[torch.Tensor] = None,
                fwd_ov: Optional[torch.Tensor] = None,
                df_ov: Optional[torch.Tensor] = None,
                wf_ov: Optional[torch.Tensor] = None,
                ko_ov: Optional[torch.Tensor] = None,
                ic_ov: Optional[torch.Tensor] = None,
                cp_ov: Optional[torch.Tensor] = None,
                ig_ov: Optional[torch.Tensor] = None,
                cr_ov: Optional[torch.Tensor] = None,
                eps_ov: Optional[torch.Tensor] = None,
                payoff_ov: Optional[torch.Tensor] = None,
                cg_flags_ov: Optional[torch.Tensor] = None,
                use_rt_flags: bool = False,
                inv_perm_ov: Optional[torch.Tensor] = None,
                use_cg_arena: bool = False,
            ) -> None:
                eff_dv_mf = d_var_ov if d_var_ov is not None else d_var_f32_g
                eff_fwd_mf = fwd_ov if fwd_ov is not None else fwd_perf_f32_g
                eff_df_mf = df_ov if df_ov is not None else df_pay_f32_g
                eff_wf_mf = wf_ov if wf_ov is not None else w_fund_f32_g
                eff_ko_mf = ko_ov if ko_ov is not None else ko_f32_g
                eff_ic_mf = ic_ov if ic_ov is not None else is_call_f32_g
                eff_cp_mf = cp_ov if cp_ov is not None else cpn_f32_g
                eff_ig_mf = ig_ov if ig_ov is not None else is_guar_f32_g
                eff_cr_mf = cr_ov if cr_ov is not None else cpn_rates_f32_g
                eff_eps_mf = eps_ov if eps_ov is not None else eps_f32_g
                eff_pf_mf = payoff_ov if payoff_ov is not None else payoff_f32_g
                assert (
                    eff_dv_mf is not None
                    and eff_fwd_mf is not None
                    and eff_df_mf is not None
                    and eff_ko_mf is not None
                    and eff_ic_mf is not None
                    and eff_cp_mf is not None
                    and eff_ig_mf is not None
                    and eff_cr_mf is not None
                    and eff_eps_mf is not None
                    and eff_pf_mf is not None
                )
                eff_d_mf = int(d_pad_ov) if d_pad_ov is not None else d_assets
                eff_m_mf = int(m_pad_ov) if m_pad_ov is not None else m_rem
                eff_rt_mf = bool(use_rt_flags and cg_flags_ov is not None)
                if layout_mdn:
                    assert r_b_ov is not None and n_rep_ov is not None
                    r_b = int(r_b_ov)
                    n_rep_in = int(n_rep_ov)
                else:
                    r_b = int(z_coarse_corr_in.shape[0])
                    n_rep_in = int(z_coarse_corr_in.shape[3])
                if not pre_zeroed:
                    s1_out_g.zero_()
                    if cf_surv_out_g is not None:
                        cf_surv_out_g.zero_()
                d_pad_mf = max(1, _next_power_of_two_int(eff_d_mf))
                mf_bn_cap = int(os.environ.get("DEEPVOL_MF_BN", "256"))
                block_mf = min(mf_bn_cap, max(128, _next_power_of_two_int(n_rep_in)))
                n_tiles_mf_k = (n_rep_in + block_mf - 1) // block_mf
                ki_active_mf = bool(has_ki and b_ki > 0.0 and float(state.i_ki_past) < 0.5)
                has_os_mf = bool(k_os > 0.0)
                has_cf_mf = bool(cf_surv_out_g is not None)
                do_ema_k = bool(apply_ema_mf and n_rep_in <= 1024)
                ema_shift_g: Optional[torch.Tensor] = None
                if do_ema_k:
                    n_sgn_e = 2 if antithetic else 1
                    ema_shift_g = _alloc_cg_buf(
                        use_cg_arena,
                        qmc_device,
                        "ema_shift",
                        (n_sgn_e, r_b, eff_m_mf, eff_d_mf),
                        torch.float32,
                    )
                    block_ema = max(32, _next_power_of_two_int(n_rep_in))
                    nw_ema = 8 if block_ema >= 1024 else (4 if block_ema >= 256 else 2)
                    _fast_triton_launch(
                        _mf_gbm_ema_shift_f32_kernel,
                        (r_b * n_sgn_e, eff_d_mf),
                        z_coarse_corr_in,
                        eff_dv_mf,
                        ema_shift_g,
                        r_b,
                        eff_d_mf,
                        eff_m_mf,
                        n_rep_in,
                        float(z_scale_val),
                        cg_flags_ptr=cg_flags_ov if eff_rt_mf else None,
                        ANTITHETIC=antithetic,
                        BLOCK_N=block_ema,
                        LAYOUT_MDN=layout_mdn,
                        USE_RT_FLAGS=eff_rt_mf,
                        num_warps=nw_ema,
                    )
                _fast_triton_launch(
                    _mf_gbm_fused_f32_kernel,
                    (r_b, n_tiles_mf_k),
                    z_coarse_corr_in,
                    eff_dv_mf,
                    eff_fwd_mf,
                    eff_df_mf,
                    eff_wf_mf,
                    eff_ko_mf,
                    eff_ic_mf,
                    eff_cp_mf,
                    eff_ig_mf,
                    eff_cr_mf,
                    eff_eps_mf,
                    inv_perm_ov if inv_perm_ov is not None else gpu_meta["inv_perm"],
                    eff_pf_mf,
                    s1_out_g,
                    cf_surv_out_g,
                    ema_shift_g,
                    eff_d_mf,
                    eff_m_mf,
                    n_rep_in,
                    float(z_scale_val),
                    cg_flags_ptr=cg_flags_ov if eff_rt_mf else None,
                    D_PAD=d_pad_mf,
                    CALL_CODE=0 if eff_rt_mf else call_code,
                    CPN_CODE=0 if eff_rt_mf else cpn_code,
                    EFF_OPT_CODE=0 if eff_rt_mf else eff_opt_code_s2,
                    HAS_KI=False if eff_rt_mf else has_ki,
                    KI_ACTIVE=False if eff_rt_mf else ki_active_mf,
                    HAS_ONE_STAR=False if eff_rt_mf else has_os_mf,
                    IS_MEM=False if eff_rt_mf else is_memory,
                    IS_LONG=False if eff_rt_mf else is_long_flag,
                    NO_NOTIONAL=False if eff_rt_mf else no_notional,
                    FORCE_CPN_AC=False if eff_rt_mf else force_cpn_ac,
                    CPN_DIR_DOWN=False if eff_rt_mf else dir_down,
                    CPN_TYPE_OUT=False if eff_rt_mf else type_out,
                    BLOCK_N=block_mf,
                    HAS_LOWER_STRIKE=True if eff_rt_mf else has_lower_strike_flag,
                    HAS_PART_CALL=False if eff_rt_mf else has_part_call_flag,
                    COMPUTE_CF_SURV=has_cf_mf,
                    Z_SIGN=1.0,
                    OUT_SCALE=1.0,
                    ANTITHETIC=antithetic,
                    APPLY_EMA=do_ema_k,
                    LAYOUT_MDN=layout_mdn,
                    USE_RT_FLAGS=eff_rt_mf,
                    num_warps=4,
                )

            stat_itr_np: Optional[npt.NDArray[np.float64]] = None
            if _warm_gpu_hit:
                assert _warm_cpu is not None and _warm_gpu is not None
                vib_period_level = _warm_cpu.vib_period_level
                vib_period_idx = _warm_cpu.vib_period_idx
                vib_sub_idx = _warm_cpu.vib_sub_idx
                vib_step_start = _warm_cpu.vib_step_start
                vib_n_steps = _warm_cpu.vib_n_steps
                h_vib_val = _warm_cpu.h_vib_val
                df_last_val = _warm_cpu.df_last_val
                l_inv_t_g = _warm_gpu.l_inv_t_g
                c_inv_g = _warm_gpu.c_inv_g
                cdl_m_g = _warm_gpu.cdl_m_g
                cdl2_last_g = _warm_gpu.cdl2_last_g
                fwd_sp_pm_g = _warm_gpu.fwd_sp_pm_g
                pre_pack = _warm_gpu.pre_pack
                n_pre_rows = _warm_cpu.n_pre_rows
                cross_pairs_vib = list(_warm_gpu.cross_pairs_vib)
                stat_pack = _warm_gpu.stat_pack
                stat_itr_g = _warm_gpu.stat_itr_g
                stat_itr_np = _warm_cpu.stat_itr_np
                stat_pos_g = _warm_gpu.stat_pos_g
                n_stat_rows = _warm_cpu.n_stat_rows
                dyn_entries = _warm_gpu.dyn_entries
                need_sp_cg_alt_g = _warm_gpu.need_sp_cg_alt_g
                th_pack = _warm_gpu.th_pack
                k_cross_g = _warm_cpu.k_cross_g
                m_rem_th_g = _warm_cpu.m_rem_th_g
                ko_th_g = _warm_gpu.ko_th_g
                ic_th_g = _warm_gpu.ic_th_g
                cp_th_g = _warm_gpu.cp_th_g
                ig_th_g = _warm_gpu.ig_th_g
                cr_th_g = _warm_gpu.cr_th_g
                payoff_th_g = _warm_gpu.payoff_th_g
                eps_th_g = _warm_gpu.eps_th_g
                df_th_g = _warm_gpu.df_th_g
                wf_th_g = _warm_gpu.wf_th_g
                fwd_perf_f32_step_g = _warm_gpu.fwd_perf_f32_step_g
                w_t_bwd_g = _warm_gpu.w_t_bwd_g
            else:
                # Precompute Vibrato metadata on GPU.
                vib_tau_prev = np.concatenate([[0.0], tau_obs_np[:-1]])
                vib_d_tau = np.maximum(tau_obs_np - vib_tau_prev, 1e-10)
                vib_period_level = bool(m_rem == 1)
                vib_period_idx = m_rem - 1
                vib_sub_idx = n_sub - 1
                vib_step_start = vib_period_idx * n_sub if vib_period_level else (m_rem * n_sub - 1)
                vib_n_steps = n_sub if vib_period_level else 1
                h_vib_val = float(vib_d_tau[-1]) if vib_period_level else float(vib_d_tau[-1] / float(n_sub))
                df_last_val = float(df_pay_np[-1])
                l_inv_t_g = None
                c_inv_g = None
                cdl_m_g = None
                cdl2_last_g = None
                if use_vibrato_gamma:
                    assert cdl_m_np is not None and cdl2_last_np is not None
                    l_inv_t_np = np.linalg.pinv(chol_base, rcond=1e-12).T
                    c_inv_np = l_inv_t_np @ l_inv_t_np.T
                    l_inv_t_g = torch.as_tensor(l_inv_t_np, dtype=torch.float64, device=qmc_device).contiguous()
                    c_inv_g = torch.as_tensor(c_inv_np, dtype=torch.float64, device=qmc_device).contiguous()
                    cdl_m_g = torch.as_tensor(cdl_m_np, dtype=torch.float64, device=qmc_device).contiguous()
                    cdl2_last_g = torch.as_tensor(cdl2_last_np, dtype=torch.float64, device=qmc_device).contiguous()

                fwd_sp_pm_g = None
                if need_spot_bumps:
                    assert fwd_sp_p_np is not None and fwd_sp_m_np is not None
                    fwd_sp_pm_g = torch.as_tensor(
                        np.concatenate([fwd_sp_p_np, fwd_sp_m_np], axis=0),
                        dtype=torch.float64,
                        device=qmc_device,
                    ).contiguous()

                # Build PreTerm C^2 FD descriptor once before chunk loop when Vibrato covers terminal step on M > 1.
                pre_pack = None
                pre_itr_g = None
                n_pre_rows = 0
                if use_vibrato_gamma and m_rem > 1:
                    if want_gamma:
                        if d_assets > 1:
                            for i_v in range(d_assets):
                                for j_v in range(i_v + 1, d_assets):
                                    cross_pairs_vib.append((i_v, j_v))
                        n_pre_cross = 2 * len(cross_pairs_vib)
                        n_pre_rows = 2 * d_assets + n_pre_cross + 1
                        aa_pre = np.empty(n_pre_rows, dtype=np.int32)
                        ab_pre = np.full(n_pre_rows, -1, dtype=np.int32)
                        asl_pre = np.zeros(n_pre_rows, dtype=np.int32)
                        bsl_pre = np.full(n_pre_rows, -1, dtype=np.int32)
                        for i_v in range(d_assets):
                            aa_pre[2 * i_v] = i_v
                            aa_pre[2 * i_v + 1] = i_v
                            asl_pre[2 * i_v] = i_v
                            asl_pre[2 * i_v + 1] = d_assets + i_v
                        for c_v, (i_v, j_v) in enumerate(cross_pairs_vib):
                            b0_v = 2 * d_assets + 2 * c_v
                            aa_pre[b0_v] = i_v
                            ab_pre[b0_v] = j_v
                            asl_pre[b0_v] = i_v
                            bsl_pre[b0_v] = j_v
                            aa_pre[b0_v + 1] = i_v
                            ab_pre[b0_v + 1] = j_v
                            asl_pre[b0_v + 1] = d_assets + i_v
                            bsl_pre[b0_v + 1] = d_assets + j_v
                        aa_pre[-1] = -1
                        ab_pre[-1] = -1
                        asl_pre[-1] = 0
                    else:
                        n_pre_rows = 3
                        aa_pre = np.array([-2, -2, -1], dtype=np.int32)
                        ab_pre = np.array([-1, -1, -1], dtype=np.int32)
                        asl_pre = np.array([0, 1, 0], dtype=np.int32)
                        bsl_pre = np.full(3, -1, dtype=np.int32)
                    idx_pre_g = torch.as_tensor(
                        np.stack(
                            [
                                aa_pre,
                                ab_pre,
                                asl_pre,
                                bsl_pre,
                                np.full(n_pre_rows, -1, dtype=np.int32),
                                np.zeros(n_pre_rows, dtype=np.int32),
                                np.zeros(n_pre_rows, dtype=np.int32),
                            ],
                            axis=0,
                        ),
                        dtype=torch.int32,
                        device=qmc_device,
                    )
                    pre_itr_g = torch.zeros((n_pre_rows,), dtype=torch.float64, device=qmc_device)
                    df_pre_g = gpu_meta["df_pay"].unsqueeze(0)
                    wf_pre_g = gpu_meta["w_fund"].unsqueeze(0) if gpu_meta["w_fund"] is not None else None
                    pre_pack = (n_pre_rows, False, None, idx_pre_g, df_pre_g, wf_pre_g, pre_itr_g)

                # Separate queued payoff descriptors into static pack (single kernel) vs rare zero-forward rows.
                stat_pos_list: List[int] = []
                stat_aa: List[npt.NDArray[np.int32]] = []
                stat_ab: List[npt.NDArray[np.int32]] = []
                stat_asl: List[npt.NDArray[np.int32]] = []
                stat_bsl: List[npt.NDArray[np.int32]] = []
                stat_ssl: List[npt.NDArray[np.int32]] = []
                stat_df_rows: List[npt.NDArray[np.float64]] = []
                stat_wf_rows: List[npt.NDArray[np.float64]] = []
                stat_itr_rows: List[npt.NDArray[np.float64]] = []
                stat_sc_list: List[npt.NDArray[np.float64]] = []
                has_stat_wf = gpu_meta["w_fund"] is not None or any(sp["wf"] is not None for sp in gpu_eval_specs)
                need_sp_cg_alt_g = False
                sc_count = 0
                dyn_entries = []
                spec_off = 0
                for sp_d in gpu_eval_specs:
                    nb_s = int(len(sp_d["aa"]))
                    if sp_d["fwd_alt"] is not None or sp_d["fwd_ba"] is not None:
                        df_d_g = (
                            torch.as_tensor(sp_d["df"], dtype=torch.float64, device=qmc_device)
                            if sp_d["df"] is not None
                            else gpu_meta["df_pay"].unsqueeze(0).expand(nb_s, m_rem).contiguous()
                        )
                        wf_d_g: Optional[torch.Tensor]
                        if sp_d["wf"] is not None:
                            wf_d_g = torch.as_tensor(sp_d["wf"], dtype=torch.float64, device=qmc_device)
                        elif gpu_meta["w_fund"] is not None:
                            wf_d_g = gpu_meta["w_fund"].unsqueeze(0).expand(nb_s, m_rem).contiguous()
                        else:
                            wf_d_g = None
                        dyn_entries.append((spec_off, spec_off + nb_s, sp_d, df_d_g, wf_d_g))
                    else:
                        stat_pos_list.extend(range(spec_off, spec_off + nb_s))
                        stat_aa.append(sp_d["aa"])
                        stat_ab.append(sp_d["ab"])
                        if sp_d["spot_a"] is not None:
                            s_a = np.asarray(sp_d["spot_a"], dtype=np.int32)
                            if np.any(s_a < -99):
                                need_sp_cg_alt_g = True
                                s_a = np.where(s_a == -100, 0, np.where(s_a == -101, 1, s_a)).astype(np.int32)
                            stat_asl.append(s_a)
                        else:
                            stat_asl.append(np.full(nb_s, -1, dtype=np.int32))
                        if sp_d["spot_b"] is not None:
                            stat_bsl.append(np.asarray(sp_d["spot_b"], dtype=np.int32))
                        else:
                            stat_bsl.append(np.full(nb_s, -1, dtype=np.int32))
                        if sp_d["sc"] is not None:
                            stat_sc_list.append(sp_d["sc"])
                            stat_ssl.append(np.arange(sc_count, sc_count + nb_s, dtype=np.int32))
                            sc_count += nb_s
                        else:
                            stat_ssl.append(np.full(nb_s, -1, dtype=np.int32))
                        stat_df_rows.append(
                            sp_d["df"] if sp_d["df"] is not None else np.broadcast_to(df_pay_np[None, :], (nb_s, m_rem))
                        )
                        if has_stat_wf:
                            if sp_d["wf"] is not None:
                                stat_wf_rows.append(sp_d["wf"])
                            elif w_fund_base_np is not None:
                                stat_wf_rows.append(np.broadcast_to(w_fund_base_np[None, :], (nb_s, m_rem)))
                            else:
                                stat_wf_rows.append(np.zeros((nb_s, m_rem), dtype=np.float64))
                        stat_itr_rows.append(sp_d["itr"])
                    spec_off += nb_s

                stat_pack = None
                stat_itr_g = None
                stat_pos_g = None
                n_stat_rows = len(stat_pos_list)
                if n_stat_rows > 0:
                    stat_sc_g = (
                        torch.as_tensor(np.concatenate(stat_sc_list, axis=0), dtype=torch.float64, device=qmc_device).contiguous()
                        if stat_sc_list
                        else None
                    )
                    stat_row_arange = np.arange(n_stat_rows, dtype=np.int32)
                    stat_idx_g = torch.as_tensor(
                        np.stack(
                            [
                                np.concatenate(stat_aa, axis=0),
                                np.concatenate(stat_ab, axis=0),
                                np.concatenate(stat_asl, axis=0),
                                np.concatenate(stat_bsl, axis=0),
                                np.concatenate(stat_ssl, axis=0),
                                stat_row_arange,
                                stat_row_arange,
                            ],
                            axis=0,
                        ),
                        dtype=torch.int32,
                        device=qmc_device,
                    )
                    stat_df_g = torch.as_tensor(np.concatenate(stat_df_rows, axis=0), dtype=torch.float64, device=qmc_device).contiguous()
                    stat_wf_g = (
                        torch.as_tensor(np.concatenate(stat_wf_rows, axis=0), dtype=torch.float64, device=qmc_device).contiguous()
                        if has_stat_wf
                        else None
                    )
                    stat_itr_np = np.ascontiguousarray(np.concatenate(stat_itr_rows, axis=0), dtype=np.float64)
                    stat_itr_g = torch.as_tensor(stat_itr_np, dtype=torch.float64, device=qmc_device).contiguous()
                    stat_pack = (n_stat_rows, False, stat_sc_g, stat_idx_g, stat_df_g, stat_wf_g, stat_itr_g)
                    stat_pos_g = torch.as_tensor(stat_pos_list, dtype=torch.long, device=qmc_device)

                # Build rolled Theta GPU descriptor once before chunk loop.
                th_pack = None
                k_cross_g = 0
                m_rem_th_g = m_rem
                ko_th_g = None
                ic_th_g = None
                cp_th_g = None
                ig_th_g = None
                cr_th_g = None
                payoff_th_g = None
                eps_th_g = None
                df_th_g = None
                wf_th_g = None
                if th_chunk_in_step1 and th_prep is not None:
                    assert fwd_th_g is not None and lv_th_g is not None
                    k_cross_g = int(th_prep["k_cross"])
                    m_rem_th_g = int(th_prep["m_rem_th"])
                    ko_th_g = gpu_meta["ko"][k_cross_g:]
                    ic_th_g = gpu_meta["is_call"][k_cross_g:]
                    cp_th_g = gpu_meta["cpn"][k_cross_g:]
                    ig_th_g = gpu_meta["is_guar"][k_cross_g:]
                    cr_th_g = gpu_meta["cpn_rates"][k_cross_g:]
                    payoff_th_g = gpu_meta["payoff_f64"].clone()
                    payoff_th_g[10] = float(th_prep["eff_cmem"])
                    payoff_th_g[11] = float(th_prep["eff_iki"])
                    eps_th_g = eps_aad_g[k_cross_g:]
                    df_th_g = torch.as_tensor(th_prep["df_pay"][None, :], dtype=torch.float64, device=qmc_device).contiguous()
                    wf_th_g = (
                        torch.as_tensor(th_prep["w_fund"][None, :], dtype=torch.float64, device=qmc_device).contiguous()
                        if th_prep["w_fund"] is not None
                        else None
                    )
                    th_idx_g = torch.as_tensor(
                        np.array([[-1], [-1], [-1], [-1], [-1], [0], [0]], dtype=np.int32),
                        dtype=torch.int32,
                        device=qmc_device,
                    )
                    th_itr_zero_g = torch.zeros((1,), dtype=torch.float64, device=qmc_device)
                    th_pack = (1, False, None, th_idx_g, df_th_g, wf_th_g, th_itr_zero_g)

                fwd_perf_f32_step_g = (
                    fwd_perf_f32_g
                    if fwd_perf_f32_g is not None
                    else fwd_perf_g.to(dtype=torch.float32).contiguous()
                )
                w_t_bwd_g = None

                if warm_bundle_key is not None:
                    cpu_p = _CrnCpuPrep()
                    cpu_p.lv_y_min_vec = lv_y_min_vec
                    cpu_p.lv_y_max_vec = lv_y_max_vec
                    cpu_p.t_max = t_max
                    cpu_p.lv_t_grid = lv_t_grid
                    cpu_p.u_time_np = u_time_np
                    cpu_p.chol_base = chol_base
                    cpu_p.v_eig_np = v_eig_np
                    cpu_p.sqrt_ev_np = sqrt_ev_np
                    cpu_p.r_native_obs_np = r_native_obs_np
                    cpu_p.w_fund_base_np = w_fund_base_np
                    cpu_p.base_lv_tables = base_lv_tables
                    cpu_p.base_lv_caches = base_lv_caches
                    cpu_p.vol_atm_obs = vol_atm_obs
                    cpu_p.th_prep = th_prep
                    cpu_p.th_chunk_in_step1 = th_chunk_in_step1
                    cpu_p.d_var_gbm_np = d_var_gbm_np
                    cpu_p.mf_cheap_digest = mf_cheap_digest
                    cpu_p.use_mf_cv = use_mf_cv
                    cpu_p.apply_ema_mf = apply_ema_mf
                    cpu_p.ema_group_mf = ema_group_mf
                    cpu_p.n_rep_mf = n_rep_mf
                    cpu_p.n_rep_gen_mf = n_rep_gen_mf
                    cpu_p.eff_r_mf = eff_r_mf
                    cpu_p.sp_p_vec = sp_p_vec
                    cpu_p.sp_m_vec = sp_m_vec
                    cpu_p.cdl_m_np = cdl_m_np
                    cpu_p.cdl2_last_np = cdl2_last_np
                    cpu_p.fwd_sp_p_np = fwd_sp_p_np
                    cpu_p.fwd_sp_m_np = fwd_sp_m_np
                    cpu_p.sp_task_meta = sp_task_meta
                    cpu_p.adj_rho_tasks = adj_rho_tasks
                    cpu_p.rho_tasks = rho_tasks
                    cpu_p.adj_repo_tasks = adj_repo_tasks
                    cpu_p.repo_task_meta = repo_task_meta
                    cpu_p.adj_div_task = adj_div_task
                    cpu_p.div_task_meta = div_task_meta
                    cpu_p.adj_fxc_tasks = adj_fxc_tasks
                    cpu_p.fxc_tasks = fxc_tasks
                    cpu_p.adj_fxv_tasks = adj_fxv_tasks
                    cpu_p.fxv_tasks = fxv_tasks
                    cpu_p.rho_dict_tmpl = {k_r: list(v_r) for k_r, v_r in rho_dict.items()} if rho_dict is not None else None
                    cpu_p.rho_arrays_tmpl = {k_r: v_r.copy() for k_r, v_r in rho_arrays.items()} if rho_arrays is not None else None
                    cpu_p.repo_rho_has_map = bool(repo_rho_dict is not None)
                    cpu_p.divs_dict_tmpl = {k_d: list(v_d) for k_d, v_d in divs_dict.items()} if divs_dict is not None else None
                    cpu_p.divs_arrays_tmpl = {k_d: v_d.copy() for k_d, v_d in divs_arrays.items()} if divs_arrays is not None else None
                    cpu_p.fx_vega_has_map = bool(fx_vega_dict is not None)
                    cpu_p.fx_cega_has_map = bool(fx_cega_dict is not None)
                    cpu_p.need_pv0_eval = need_pv0_eval
                    cpu_p.tok_base = tok_base
                    cpu_p.gpu_eval_rows_total = gpu_eval_rows_total
                    cpu_p.n_stat_rows = n_stat_rows
                    cpu_p.n_pre_rows = n_pre_rows
                    cpu_p.vib_period_level = vib_period_level
                    cpu_p.vib_period_idx = vib_period_idx
                    cpu_p.vib_sub_idx = vib_sub_idx
                    cpu_p.vib_step_start = vib_step_start
                    cpu_p.vib_n_steps = vib_n_steps
                    cpu_p.h_vib_val = h_vib_val
                    cpu_p.df_last_val = df_last_val
                    cpu_p.k_cross_g = k_cross_g
                    cpu_p.m_rem_th_g = m_rem_th_g
                    cpu_p.ko_arr_np = ko_arr_np
                    cpu_p.cpn_arr_np = cpn_arr_np
                    cpu_p.is_call_np = is_call_np
                    cpu_p.is_guar_np = is_guar_np
                    cpu_p.cpn_rates_np = cpn_rates_np
                    cpu_p.stat_itr_np = stat_itr_np

                    gpu_v = _CrnGpuViews()
                    gpu_v.gpu_meta = gpu_meta
                    gpu_v.lv_f32_g = lv_f32_g_cached
                    gpu_v.lv_sp_pm_g = lv_sp_pm_g
                    gpu_v.lv_th_g = lv_th_g
                    gpu_v.chol_f32_g = chol_f32_g_cached
                    gpu_v.chol_f64_g = chol_f64_g
                    gpu_v.tau_step_g = tau_step_g
                    gpu_v.rn_step_g = rn_step_g
                    gpu_v.rt_step_g = rt_step_g
                    gpu_v.fwd_perf_g = fwd_perf_g
                    gpu_v.fwd_th_g = fwd_th_g
                    gpu_v.y_min_base_g = y_min_base_g
                    gpu_v.inv_dy_base_g = inv_dy_base_g
                    gpu_v.inv_dt_base_g = inv_dt_base_g
                    gpu_v.id_d_g = id_d_g
                    gpu_v.sp_pm_meta_g = sp_pm_meta_g
                    gpu_v.sched_f32_g = sched_f32_g
                    gpu_v.eps_aad_g = eps_aad_g
                    gpu_v.u_rot_gpu = u_rot_gpu
                    gpu_v.v_rot_gpu = v_rot_gpu
                    gpu_v.d_var_f32_g = d_var_f32_g
                    gpu_v.fwd_perf_f32_g = fwd_perf_f32_g
                    gpu_v.df_pay_f32_g = df_pay_f32_g
                    gpu_v.w_fund_f32_g = w_fund_f32_g
                    gpu_v.ko_f32_g = ko_f32_g
                    gpu_v.is_call_f32_g = is_call_f32_g
                    gpu_v.cpn_f32_g = cpn_f32_g
                    gpu_v.is_guar_f32_g = is_guar_f32_g
                    gpu_v.cpn_rates_f32_g = cpn_rates_f32_g
                    gpu_v.eps_f32_g = eps_f32_g
                    gpu_v.payoff_f32_g = payoff_f32_g
                    gpu_v.chol_scaled_f32_g = chol_scaled_f32_g
                    gpu_v.l_inv_t_g = l_inv_t_g
                    gpu_v.c_inv_g = c_inv_g
                    gpu_v.cdl_m_g = cdl_m_g
                    gpu_v.cdl2_last_g = cdl2_last_g
                    gpu_v.fwd_sp_pm_g = fwd_sp_pm_g
                    gpu_v.pre_pack = pre_pack
                    gpu_v.stat_pack = stat_pack
                    gpu_v.stat_pos_g = stat_pos_g
                    gpu_v.stat_itr_g = stat_itr_g
                    gpu_v.dyn_entries = dyn_entries
                    gpu_v.need_sp_cg_alt_g = need_sp_cg_alt_g
                    gpu_v.cross_pairs_vib = tuple(cross_pairs_vib)
                    gpu_v.th_pack = th_pack
                    gpu_v.ko_th_g = ko_th_g
                    gpu_v.ic_th_g = ic_th_g
                    gpu_v.cp_th_g = cp_th_g
                    gpu_v.ig_th_g = ig_th_g
                    gpu_v.cr_th_g = cr_th_g
                    gpu_v.payoff_th_g = payoff_th_g
                    gpu_v.eps_th_g = eps_th_g
                    gpu_v.df_th_g = df_th_g
                    gpu_v.wf_th_g = wf_th_g
                    gpu_v.w_t_bwd_g = w_t_bwd_g
                    gpu_v.dummy_f32_1 = dummy_f32_1
                    gpu_v.fwd_perf_f32_step_g = fwd_perf_f32_step_g
                    gpu_v.inv_dt_f32_val = inv_dt_f32_val
                    gpu_v.cg_bucket_pack = None
                    gpu_v.cg_meta_cache = {}

                    backing_gpu = _dedup_gpu_tensors(
                        (
                            gpu_meta,
                            lv_f32_g_cached,
                            lv_sp_pm_g,
                            lv_th_g,
                            chol_f32_g_cached,
                            id_d_g,
                            sp_pm_meta_g,
                            sched_f32_g,
                            u_rot_gpu,
                            v_rot_gpu,
                            d_var_f32_g,
                            fwd_perf_f32_g,
                            df_pay_f32_g,
                            w_fund_f32_g,
                            ko_f32_g,
                            is_call_f32_g,
                            cpn_f32_g,
                            is_guar_f32_g,
                            cpn_rates_f32_g,
                            eps_f32_g,
                            payoff_f32_g,
                            chol_scaled_f32_g,
                            l_inv_t_g,
                            c_inv_g,
                            cdl_m_g,
                            cdl2_last_g,
                            fwd_sp_pm_g,
                            pre_pack,
                            stat_pack,
                            stat_pos_g,
                            dyn_entries,
                            th_pack,
                            payoff_th_g,
                            w_t_bwd_g,
                            dummy_f32_1,
                            fwd_perf_f32_step_g,
                        )
                    )
                    _CRN_DEAL_PREP_CACHE.put(warm_bundle_key, (cpu_p, gpu_v, backing_gpu))

            need_norm_x_kept = bool(dyn_entries)
            use_f32_perf_direct = bool(
                use_f32_step1
                and fuse_m_sde
                and n_total <= 16384
                and (not is_cont_ki_active)
                and (not need_norm_x_kept)
                and os.environ.get("DEEPVOL_L2O_F32", "1") != "0"
                and os.environ.get("DEEPVOL_AAD_F32", "1") == "1"
            )
            cg_bucket_dm = _resolve_qmc_cg_bucket(d_assets, m_rem, n_sub)
            use_cg_bucket = bool(
                cg_bucket_dm is not None
                and os.environ.get("DEEPVOL_QMC_CG", "1") != "0"
                and can_triton_sobol
                and use_f32_step1
                and fuse_m_sde
                and use_triton_aad_s2
                and use_f32_perf_direct
                and (not dyn_entries)
                and (not is_cont_ki_active)
                and (not need_sp_cg_alt_g)
                and (stat_pack is None or stat_pack[2] is None)
                and call_code in (0, 1, 2)
                and cpn_code in (0, 1, 2)
                and eff_opt_code_s2 in (0, 1, 2)
                and (need_stage2_aad or use_vibrato_gamma or need_spot_bumps)
                and ((not use_vibrato_gamma) or want_gamma or m_rem == 1)
                and (not torch.cuda.is_current_stream_capturing())  # type: ignore[no-untyped-call]
            )
            if use_cg_bucket:
                assert cg_bucket_dm is not None
                d_pad_b, m_pad_b = cg_bucket_dm
                if not (
                    (n_stat_rows == 0 or n_stat_rows <= 2 * d_pad_b + 2)
                    and n_pre_rows <= d_pad_b * (d_pad_b + 1) + 1
                ):
                    use_cg_bucket = False

            if use_cg_bucket and use_mf_cv and mf_cheap_hit is None and sv_sobol_gpu is not None:
                assert u_rot_gpu is not None and n_rep_gen_mf > 0
                mf_shift_key_pre = ("sobol_mf_shifts_v3", eff_r_mf, n_rep_gen_mf, m_rem, d_assets, seed + 99991, dev_norm_str)
                shifts_mf_pre = _CRN_SOBOL_CACHE.get(mf_shift_key_pre)
                if shifts_mf_pre is None:
                    rng_mf_pre = np.random.default_rng(seed + 99991)
                    shifts_mf_np_pre = rng_mf_pre.integers(0, 1 << 30, size=(eff_r_mf, m_rem * 2 * d_assets), dtype=np.int32)
                    shifts_mf_pre = torch.from_numpy(shifts_mf_np_pre).to(device=qmc_device, dtype=torch.int32)
                    _CRN_SOBOL_CACHE.put(mf_shift_key_pre, shifts_mf_pre)
                s1_cheap_tmp = torch.zeros((eff_r_mf * ema_group_mf, 3 * m_rem + 5), dtype=torch.float64, device=qmc_device)
                cf_cheap_tmp = (
                    torch.zeros((eff_r_mf * ema_group_mf, 2, m_rem), dtype=torch.float64, device=qmc_device)
                    if want_cf_wf_aad
                    else None
                )
                cv_eig_tmp = chol_f32_g_cached.contiguous()
                mf_per_rep_b = max(m_rem * d_assets * 8 + d_assets * 4, 1) * n_rep_gen_mf
                r_mf_c_lim = min(eff_r_mf, max(1, (12 * 1024 * 1024) // mf_per_rep_b))
                m_pow_pre = int(round(math.log2(n_rep_gen_mf)))
                blk_n_pre = min(256, n_rep_gen_mf)
                nt_mf_pre = (n_rep_gen_mf + blk_n_pre - 1) // blk_n_pre
                for r_mf_s in range(0, eff_r_mf, r_mf_c_lim):
                    r_mf_e = min(eff_r_mf, r_mf_s + r_mf_c_lim)
                    r_mf_c = r_mf_e - r_mf_s
                    n_mf_t = r_mf_c * n_rep_gen_mf
                    z_raw_pre = torch.empty((m_rem, d_assets, n_mf_t), dtype=torch.float32, device=qmc_device)
                    _fast_triton_launch(
                        _sobol_bridge_normals_kernel,
                        (nt_mf_pre, m_rem * r_mf_c),
                        sv_sobol_gpu,
                        shifts_mf_pre[r_mf_s:r_mf_e],
                        z_raw_pre,
                        m_rem,
                        d_assets,
                        n_rep=n_rep_gen_mf,
                        n_sub=1,
                        m_pow=m_pow_pre,
                        BLOCK_N=blk_n_pre,
                        num_warps=4,
                    )
                    tmp_pre = torch.matmul(cv_eig_tmp, z_raw_pre).view(m_rem, d_assets * n_mf_t)
                    del z_raw_pre
                    z_corr_pre = torch.matmul(u_rot_gpu, tmp_pre).view(m_rem, d_assets, n_mf_t)
                    del tmp_pre
                    sb_s = r_mf_s * ema_group_mf
                    sb_e = r_mf_e * ema_group_mf
                    _run_mf_gbm_batch(
                        z_corr_pre,
                        s1_cheap_tmp[sb_s:sb_e],
                        cf_cheap_tmp[sb_s:sb_e] if cf_cheap_tmp is not None else None,
                        antithetic=True,
                        layout_mdn=True,
                        r_b_ov=r_mf_c * ema_group_mf,
                        n_rep_ov=n_rep_mf,
                        z_scale_val=1.0,
                        pre_zeroed=True,
                    )
                    del z_corr_pre
                del cv_eig_tmp
                s1_ch_np = np.asarray(s1_cheap_tmp.detach().cpu().numpy(), dtype=np.float64)
                cf_ch_np = np.asarray(cf_cheap_tmp.detach().cpu().numpy(), dtype=np.float64) if cf_cheap_tmp is not None else None
                del s1_cheap_tmp, cf_cheap_tmp
                if ema_group_mf > 1:
                    s1_ch_np = np.ascontiguousarray(s1_ch_np.reshape(eff_r_mf, ema_group_mf, -1).mean(axis=1))
                    if cf_ch_np is not None:
                        cf_ch_np = np.ascontiguousarray(cf_ch_np.reshape(eff_r_mf, ema_group_mf, 2, m_rem).mean(axis=1))
                n_mf_rc = int(s1_ch_np.shape[0])
                var_c_pre = float(np.var(s1_ch_np[:, 3 * m_rem + 4], ddof=1)) / float(n_mf_rc) if n_mf_rc > 1 else 0.0
                se_pre = math.sqrt(max(var_c_pre, 0.0)) if math.isfinite(var_c_pre) else 0.0
                mf_cheap_hit = (s1_ch_np, cf_ch_np, se_pre)
                if mf_cheap_digest:
                    _CRN_MF_CHEAP_CACHE.put(mf_cheap_digest, mf_cheap_hit)

            bpk: Optional[Dict[str, Any]] = None
            if use_cg_bucket:
                assert cg_bucket_dm is not None
                d_pad_b, m_pad_b = cg_bucket_dm
                if _warm_gpu_hit and _warm_gpu is not None and _warm_gpu.cg_bucket_pack is not None:
                    bpk = _warm_gpu.cg_bucket_pack
                else:
                    n_steps_pad_b = m_pad_b * n_sub
                    n_stat_pad_b = (2 * d_pad_b + 2) if n_stat_rows > 0 else 0
                    n_pre_pad_b = (d_pad_b * (d_pad_b + 1) + 1) if use_vibrato_gamma else 0
                    n_pre_act_b = n_pre_rows if (use_vibrato_gamma and m_rem > 1) else 0
                    cg_flags_np = np.array(
                        [
                            d_assets,
                            m_rem,
                            call_code,
                            cpn_code,
                            eff_opt_code_s2,
                            int(has_ki),
                            int(bool(has_ki and b_ki > 0.0 and float(state.i_ki_past) < 0.5)),
                            int(bool(k_os > 0.0)),
                            int(is_memory),
                            int(is_long_flag),
                            int(no_notional),
                            int(force_cpn_ac),
                            int(dir_down),
                            int(type_out),
                            int(has_lower_strike_flag),
                            int(has_part_call_flag),
                            n_stat_rows,
                            n_pre_act_b,
                            m_rem - 1,
                            n_sub - 1,
                            ((m_rem - 1) * n_sub) if m_rem == 1 else (m_rem * n_sub - 1),
                            n_sub if m_rem == 1 else 1,
                            1 if m_rem == 1 else 0,
                            0,
                        ],
                        dtype=np.int32,
                    )
                    cg_flags_g_b = torch.as_tensor(cg_flags_np, dtype=torch.int32, device=qmc_device)
                    payoff_f64_18_g_b = torch.cat(
                        [
                            gpu_meta["payoff_f64"],
                            torch.tensor([df_last_val, h_vib_val, inv_dt_f32_val], dtype=torch.float64, device=qmc_device),
                        ],
                        dim=0,
                    ).contiguous()
                    payoff_f32_15_g_b = payoff_f64_18_g_b[:15].to(dtype=torch.float32).contiguous()

                    if d_pad_b == d_assets:
                        lv_pad_g_b = lv_f32_g_cached.clone()
                        lv_sp_pm_pad_g_b = lv_sp_pm_g.clone() if (need_spot_bumps and lv_sp_pm_g is not None) else None
                    else:
                        lv_pad_g_b = torch.zeros((d_pad_b, lv_nt, lv_ny), dtype=torch.float32, device=qmc_device)
                        lv_pad_g_b[:d_assets].copy_(lv_f32_g_cached)
                        lv_sp_pm_pad_g_b = None
                        if need_spot_bumps and lv_sp_pm_g is not None:
                            lv_sp_pm_pad_g_b = torch.zeros((2 * d_pad_b, lv_nt, lv_ny), dtype=torch.float32, device=qmc_device)
                            lv_sp_pm_pad_g_b[:d_assets].copy_(lv_sp_pm_g[:d_assets])
                            lv_sp_pm_pad_g_b[d_pad_b : d_pad_b + d_assets].copy_(lv_sp_pm_g[d_assets : 2 * d_assets])

                    chol_pad_f64_g_b = torch.eye(d_pad_b, dtype=torch.float64, device=qmc_device)
                    chol_pad_f64_g_b[:d_assets, :d_assets].copy_(chol_f64_g)
                    chol_pad_f32_g_b = chol_pad_f64_g_b.to(dtype=torch.float32).contiguous()
                    chol_scaled_pad_f32_g_b = (chol_pad_f32_g_b * math.sqrt(float(n_sub))).contiguous()

                    y_min_pad_f64_b = torch.zeros((d_pad_b,), dtype=torch.float64, device=qmc_device)
                    y_min_pad_f64_b[:d_assets].copy_(y_min_base_g)
                    inv_dy_pad_f64_b = torch.ones((d_pad_b,), dtype=torch.float64, device=qmc_device)
                    inv_dy_pad_f64_b[:d_assets].copy_(inv_dy_base_g)
                    inv_dt_pad_f64_b = inv_dt_base_g[:1].clone()
                    tau_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    tau_pad_f64_b[:m_rem].copy_(tau_step_g[:m_rem])
                    rn_pad_f64_b = torch.zeros((d_pad_b, m_pad_b), dtype=torch.float64, device=qmc_device)
                    rn_pad_f64_b[:d_assets, :m_rem].copy_(rn_step_g[: d_assets * m_rem].view(d_assets, m_rem))
                    rt_pad_f64_b = torch.zeros((d_pad_b, m_pad_b), dtype=torch.float64, device=qmc_device)
                    rt_pad_f64_b[:d_assets, :m_rem].copy_(rt_step_g[: d_assets * m_rem].view(d_assets, m_rem))

                    fwd_pad_f64_b = torch.ones((d_pad_b, m_pad_b), dtype=torch.float64, device=qmc_device)
                    fwd_pad_f64_b[:d_assets, :m_rem].copy_(fwd_perf_g)
                    fwd_pad_f32_b = fwd_pad_f64_b.to(dtype=torch.float32).contiguous()
                    fwd_sp_pm_pad_f64_b: Optional[torch.Tensor] = None
                    if need_spot_bumps and fwd_sp_pm_g is not None:
                        fwd_sp_pm_pad_f64_b = torch.ones((2 * d_pad_b, m_pad_b), dtype=torch.float64, device=qmc_device)
                        fwd_sp_pm_pad_f64_b[:d_assets, :m_rem].copy_(fwd_sp_pm_g[:d_assets])
                        fwd_sp_pm_pad_f64_b[d_pad_b : d_pad_b + d_assets, :m_rem].copy_(fwd_sp_pm_g[d_assets : 2 * d_assets])

                    id_d_pad_b = torch.arange(d_pad_b, dtype=torch.int32, device=qmc_device)
                    sp_pm_meta_pad_b: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
                    if need_spot_bumps:
                        id_2d_b = torch.arange(2 * d_pad_b, dtype=torch.int32, device=qmc_device)
                        zr_2d_b = torch.cat([id_d_pad_b, id_d_pad_b], dim=0)
                        sp_pm_meta_pad_b = (id_2d_b, zr_2d_b, zr_2d_b)

                    # Pad schedule buffer cleanly: [t_mid(M*S), dt(M*S), sqrt_dt(M*S), q_shift(D*M*S), y_min(D), inv_dy(D)].
                    n_steps_act = m_rem * n_sub
                    sched_pad_f32_b = torch.zeros((3 * n_steps_pad_b + d_pad_b * n_steps_pad_b + 2 * d_pad_b,), dtype=torch.float32, device=qmc_device)
                    sched_pad_f32_b[0:n_steps_act].copy_(sched_f32_g[0:n_steps_act])
                    sched_pad_f32_b[n_steps_pad_b : n_steps_pad_b + n_steps_act].copy_(sched_f32_g[n_steps_act : 2 * n_steps_act])
                    sched_pad_f32_b[2 * n_steps_pad_b : 2 * n_steps_pad_b + n_steps_act].copy_(sched_f32_g[2 * n_steps_act : 3 * n_steps_act])
                    q_act_src = sched_f32_g[3 * n_steps_act : 3 * n_steps_act + d_assets * n_steps_act].view(d_assets, n_steps_act)
                    q_pad_dst = sched_pad_f32_b[3 * n_steps_pad_b : 3 * n_steps_pad_b + d_pad_b * n_steps_pad_b].view(d_pad_b, n_steps_pad_b)
                    q_pad_dst[:d_assets, :n_steps_act].copy_(q_act_src)
                    y_src_off = 3 * n_steps_act + d_assets * n_steps_act
                    y_dst_off = 3 * n_steps_pad_b + d_pad_b * n_steps_pad_b
                    sched_pad_f32_b[y_dst_off : y_dst_off + d_assets].copy_(sched_f32_g[y_src_off : y_src_off + d_assets])
                    sched_pad_f32_b[y_dst_off + d_pad_b : y_dst_off + d_pad_b + d_assets].copy_(
                        sched_f32_g[y_src_off + d_assets : y_src_off + 2 * d_assets]
                    )
                    sched_pad_f32_b[y_dst_off + d_pad_b + d_assets : y_dst_off + 2 * d_pad_b].fill_(1.0)

                    u_rot_pad_b = torch.eye(m_pad_b, dtype=torch.float32, device=qmc_device)
                    assert u_rot_gpu is not None
                    u_rot_pad_b[:m_rem, :m_rem].copy_(u_rot_gpu)
                    assert shifts_sobol_gpu is not None
                    n_dims_act_b = int(shifts_sobol_gpu.shape[1])
                    n_dims_pad_b = m_pad_b * (n_sub + 1) * d_pad_b
                    shifts_pad_b = torch.zeros((eff_r, n_dims_pad_b), dtype=torch.int32, device=qmc_device)
                    shifts_pad_b[:, :n_dims_act_b].copy_(shifts_sobol_gpu)

                    df_pay_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    df_pay_pad_f64_b[:m_rem].copy_(gpu_meta["df_pay"])
                    w_fund_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    if gpu_meta["w_fund"] is not None:
                        w_fund_pad_f64_b[:m_rem].copy_(gpu_meta["w_fund"])
                    ko_pad_f64_b = torch.full((m_pad_b,), 1.0e9, dtype=torch.float64, device=qmc_device)
                    ko_pad_f64_b[:m_rem].copy_(gpu_meta["ko"])
                    ic_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    ic_pad_f64_b[:m_rem].copy_(gpu_meta["is_call"])
                    cp_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    cp_pad_f64_b[:m_rem].copy_(gpu_meta["cpn"])
                    ig_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    ig_pad_f64_b[:m_rem].copy_(gpu_meta["is_guar"])
                    cr_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    cr_pad_f64_b[:m_rem].copy_(gpu_meta["cpn_rates"])
                    eps_pad_f64_b = torch.full((m_pad_b,), 0.04, dtype=torch.float64, device=qmc_device)
                    eps_pad_f64_b[:m_rem].copy_(eps_aad_g)

                    d_var_pad_f32_b: Optional[torch.Tensor] = None
                    if use_mf_cv and d_var_f32_g is not None:
                        d_var_pad_f32_b = torch.zeros((d_pad_b, m_pad_b), dtype=torch.float32, device=qmc_device)
                        d_var_pad_f32_b[:d_assets, :m_rem].copy_(d_var_f32_g)
                    df_pay_pad_f32_b = df_pay_pad_f64_b.to(dtype=torch.float32).contiguous()
                    w_fund_pad_f32_b = w_fund_pad_f64_b.to(dtype=torch.float32).contiguous()
                    ko_pad_f32_b = ko_pad_f64_b.to(dtype=torch.float32).contiguous()
                    ic_pad_f32_b = ic_pad_f64_b.to(dtype=torch.float32).contiguous()
                    cp_pad_f32_b = cp_pad_f64_b.to(dtype=torch.float32).contiguous()
                    ig_pad_f32_b = ig_pad_f64_b.to(dtype=torch.float32).contiguous()
                    cr_pad_f32_b = cr_pad_f64_b.to(dtype=torch.float32).contiguous()
                    eps_pad_f32_b = eps_pad_f64_b.to(dtype=torch.float32).contiguous()

                    l_inv_t_pad_b: Optional[torch.Tensor] = None
                    c_inv_pad_b: Optional[torch.Tensor] = None
                    cdl_m_pad_b: Optional[torch.Tensor] = None
                    cdl2_last_pad_b: Optional[torch.Tensor] = None
                    if use_vibrato_gamma:
                        l_inv_t_pad_b = torch.eye(d_pad_b, dtype=torch.float64, device=qmc_device)
                        c_inv_pad_b = torch.eye(d_pad_b, dtype=torch.float64, device=qmc_device)
                        cdl_m_pad_b = torch.ones((m_pad_b, d_pad_b), dtype=torch.float64, device=qmc_device)
                        cdl2_last_pad_b = torch.zeros((d_pad_b,), dtype=torch.float64, device=qmc_device)
                        if l_inv_t_g is not None and c_inv_g is not None and cdl_m_g is not None and cdl2_last_g is not None:
                            l_inv_t_pad_b[:d_assets, :d_assets].copy_(l_inv_t_g)
                            c_inv_pad_b[:d_assets, :d_assets].copy_(c_inv_g)
                            cdl_m_pad_b[:m_rem, :d_assets].copy_(cdl_m_g)
                            cdl2_last_pad_b[:d_assets].copy_(cdl2_last_g)

                    def _remap_spot_slots(slots_t: torch.Tensor) -> torch.Tensor:
                        return torch.where(slots_t >= d_assets, d_pad_b + (slots_t - d_assets), slots_t)

                    stat_pack_pad_b: Optional[Tuple[Any, ...]] = None
                    if stat_pack is not None and n_stat_pad_b > 0:
                        _, _, _, st_idx_s, st_df_s, st_wf_s, st_itr_s = stat_pack
                        st_idx_pad = torch.full((7, n_stat_pad_b), -1, dtype=torch.int32, device=qmc_device)
                        st_idx_pad[:, :n_stat_rows].copy_(st_idx_s)
                        st_idx_pad[2, :n_stat_rows] = _remap_spot_slots(st_idx_pad[2, :n_stat_rows])
                        st_idx_pad[3, :n_stat_rows] = _remap_spot_slots(st_idx_pad[3, :n_stat_rows])
                        st_idx_pad[5, n_stat_rows:] = 0
                        st_idx_pad[6, n_stat_rows:] = 0
                        st_df_pad = torch.zeros((n_stat_pad_b, m_pad_b), dtype=torch.float64, device=qmc_device)
                        st_df_pad[:n_stat_rows, :m_rem].copy_(st_df_s)
                        st_wf_pad = torch.zeros((n_stat_pad_b, m_pad_b), dtype=torch.float64, device=qmc_device)
                        if st_wf_s is not None:
                            st_wf_pad[:n_stat_rows, :m_rem].copy_(st_wf_s)
                        st_itr_pad = torch.zeros((n_stat_pad_b,), dtype=torch.float64, device=qmc_device)
                        st_itr_pad[:n_stat_rows].copy_(st_itr_s)
                        stat_pack_pad_b = (n_stat_pad_b, False, None, st_idx_pad, st_df_pad, st_wf_pad, st_itr_pad)

                    pre_pack_pad_b: Optional[Tuple[Any, ...]] = None
                    if use_vibrato_gamma and n_pre_pad_b > 0:
                        pr_idx_pad = torch.full((7, n_pre_pad_b), -1, dtype=torch.int32, device=qmc_device)
                        pr_idx_pad[5].zero_()
                        pr_idx_pad[6].zero_()
                        if pre_pack is not None and n_pre_act_b > 0:
                            _, _, _, pr_idx_s, _, _, _ = pre_pack
                            pr_idx_pad[:, :n_pre_act_b].copy_(pr_idx_s)
                            pr_idx_pad[2, :n_pre_act_b] = _remap_spot_slots(pr_idx_pad[2, :n_pre_act_b])
                            pr_idx_pad[3, :n_pre_act_b] = _remap_spot_slots(pr_idx_pad[3, :n_pre_act_b])
                        pr_df_pad = df_pay_pad_f64_b.unsqueeze(0)
                        pr_wf_pad = w_fund_pad_f64_b.unsqueeze(0)
                        pr_itr_pad = torch.zeros((n_pre_pad_b,), dtype=torch.float64, device=qmc_device)
                        pre_pack_pad_b = (n_pre_pad_b, False, None, pr_idx_pad, pr_df_pad, pr_wf_pad, pr_itr_pad)

                    per_rep_b = max(
                        m_pad_b * d_pad_b * (n_sub * 4 + 8 + (4 if need_stage2_aad else 0) + (8 if need_spot_bumps else 0)),
                        1,
                    ) * n_rep
                    _r_env_cap_b = int(os.environ.get("DEEPVOL_R_CAP", "4"))
                    r_cap_b = 2 if (d_pad_b * m_pad_b >= 240) else (_r_env_cap_b if (d_pad_b * m_pad_b * n_sub >= 256) else eff_r)
                    r_chunk_b = min(eff_r, r_cap_b, max(1, (72 * 1024 * 1024) // per_rep_b))
                    inv_perm_pad_b = torch.arange(d_pad_b, dtype=torch.int64, device=qmc_device)
                    inv_perm_pad_b[:d_assets].copy_(gpu_meta["inv_perm"])

                    th_on_b = bool(th_chunk_in_step1 and th_pack is not None and th_prep is not None)
                    th_cg_flags_np = cg_flags_np.copy()
                    th_cg_flags_np[1] = int(m_rem_th_g) if th_on_b else 1
                    th_cg_flags_np[16] = 1 if th_on_b else 0
                    th_cg_flags_np[23] = int(k_cross_g) if th_on_b else 0
                    th_cg_flags_g_b = torch.as_tensor(th_cg_flags_np, dtype=torch.int32, device=qmc_device)
                    th_payoff_f64_18_g_b = payoff_f64_18_g_b.clone()
                    if th_on_b and d_pad_b == d_assets and lv_th_g is not None:
                        lv_th_pad_f32_b = lv_th_g.clone()
                    else:
                        lv_th_pad_f32_b = torch.zeros((d_pad_b, lv_nt, lv_ny), dtype=torch.float32, device=qmc_device)
                    fwd_th_pad_f64_b = torch.ones((d_pad_b, m_pad_b), dtype=torch.float64, device=qmc_device)
                    tau_th_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    rn_th_pad_f64_b = torch.zeros((d_pad_b, m_pad_b), dtype=torch.float64, device=qmc_device)
                    rt_th_pad_f64_b = torch.zeros((d_pad_b, m_pad_b), dtype=torch.float64, device=qmc_device)
                    inv_dt_th_pad_f64_b = (
                        inv_dt_base_g[1:2] if int(inv_dt_base_g.shape[0]) > 1 else inv_dt_base_g[:1]
                    ).clone()
                    ko_th_pad_f64_b = torch.full((m_pad_b,), 1.0e9, dtype=torch.float64, device=qmc_device)
                    ic_th_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    cp_th_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    ig_th_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    cr_th_pad_f64_b = torch.zeros((m_pad_b,), dtype=torch.float64, device=qmc_device)
                    eps_th_pad_f64_b = torch.full((m_pad_b,), 0.04, dtype=torch.float64, device=qmc_device)
                    df_th_pad_f64_b = torch.zeros((1, m_pad_b), dtype=torch.float64, device=qmc_device)
                    wf_th_pad_f64_b = torch.zeros((1, m_pad_b), dtype=torch.float64, device=qmc_device)
                    if th_on_b:
                        assert payoff_th_g is not None and lv_th_g is not None and fwd_th_g is not None
                        assert ko_th_g is not None and ic_th_g is not None and cp_th_g is not None and ig_th_g is not None
                        assert cr_th_g is not None and eps_th_g is not None and df_th_g is not None
                        th_payoff_f64_18_g_b[:15].copy_(payoff_th_g)
                        if d_pad_b != d_assets:
                            lv_th_pad_f32_b[:d_assets].copy_(lv_th_g)
                        fwd_th_pad_f64_b[:d_assets, :m_rem_th_g].copy_(fwd_th_g)
                        tau_th_pad_f64_b[:m_rem_th_g].copy_(tau_step_g[m_rem : m_rem + m_rem_th_g])
                        rn_th_pad_f64_b[:d_assets, :m_rem_th_g].copy_(
                            rn_step_g[d_assets * m_rem : d_assets * (m_rem + m_rem_th_g)].view(d_assets, m_rem_th_g)
                        )
                        rt_th_pad_f64_b[:d_assets, :m_rem_th_g].copy_(
                            rt_step_g[d_assets * m_rem : d_assets * (m_rem + m_rem_th_g)].view(d_assets, m_rem_th_g)
                        )
                        ko_th_pad_f64_b[:m_rem_th_g].copy_(ko_th_g)
                        ic_th_pad_f64_b[:m_rem_th_g].copy_(ic_th_g)
                        cp_th_pad_f64_b[:m_rem_th_g].copy_(cp_th_g)
                        ig_th_pad_f64_b[:m_rem_th_g].copy_(ig_th_g)
                        cr_th_pad_f64_b[:m_rem_th_g].copy_(cr_th_g)
                        eps_th_pad_f64_b[:m_rem_th_g].copy_(eps_th_g)
                        df_th_pad_f64_b[0, :m_rem_th_g].copy_(df_th_g[0])
                        if wf_th_g is not None:
                            wf_th_pad_f64_b[0, :m_rem_th_g].copy_(wf_th_g[0])
                    th_idx_pad_g = torch.as_tensor(
                        np.array([[-1], [-1], [-1], [-1], [-1], [0], [0]], dtype=np.int32),
                        dtype=torch.int32,
                        device=qmc_device,
                    )
                    th_itr_zero_pad_g = torch.zeros((1,), dtype=torch.float64, device=qmc_device)
                    th_pack_pad_b = (1, False, None, th_idx_pad_g, df_th_pad_f64_b, wf_th_pad_f64_b, th_itr_zero_pad_g)
                    bpk = {
                        "d_pad": d_pad_b,
                        "m_pad": m_pad_b,
                        "n_steps_pad": n_steps_pad_b,
                        "r_chunk": r_chunk_b,
                        "n_stat_pad": n_stat_pad_b,
                        "n_pre_pad": n_pre_pad_b,
                        "cg_flags": cg_flags_g_b,
                        "payoff_f64_18": payoff_f64_18_g_b,
                        "payoff_f32_15": payoff_f32_15_g_b,
                        "lv_f32": lv_pad_g_b,
                        "lv_sp_pm_f32": lv_sp_pm_pad_g_b,
                        "chol_f64": chol_pad_f64_g_b,
                        "chol_f32": chol_pad_f32_g_b,
                        "chol_scaled_f32": chol_scaled_pad_f32_g_b,
                        "y_min_f64": y_min_pad_f64_b,
                        "inv_dy_f64": inv_dy_pad_f64_b,
                        "inv_dt_f64": inv_dt_pad_f64_b,
                        "tau_f64": tau_pad_f64_b,
                        "rn_f64": rn_pad_f64_b,
                        "rt_f64": rt_pad_f64_b,
                        "fwd_pad_f64": fwd_pad_f64_b,
                        "fwd_pad_f32": fwd_pad_f32_b,
                        "fwd_sp_pm_f64": fwd_sp_pm_pad_f64_b,
                        "id_d": id_d_pad_b,
                        "sp_pm_meta": sp_pm_meta_pad_b,
                        "sched_f32": sched_pad_f32_b,
                        "t_mid_ptr": sched_pad_f32_b[0:n_steps_pad_b],
                        "dt_sub_ptr": sched_pad_f32_b[n_steps_pad_b : 2 * n_steps_pad_b],
                        "sqrt_dt_ptr": sched_pad_f32_b[2 * n_steps_pad_b : 3 * n_steps_pad_b],
                        "q_shift_ptr": sched_pad_f32_b[3 * n_steps_pad_b : y_dst_off],
                        "y_min_f32": sched_pad_f32_b[y_dst_off : y_dst_off + d_pad_b],
                        "inv_dy_f32": sched_pad_f32_b[y_dst_off + d_pad_b : y_dst_off + 2 * d_pad_b],
                        "u_rot": u_rot_pad_b,
                        "shifts": shifts_pad_b,
                        "df_pay_f64": df_pay_pad_f64_b,
                        "w_fund_f64": w_fund_pad_f64_b,
                        "ko_f64": ko_pad_f64_b,
                        "is_call_f64": ic_pad_f64_b,
                        "cpn_f64": cp_pad_f64_b,
                        "is_guar_f64": ig_pad_f64_b,
                        "cpn_rates_f64": cr_pad_f64_b,
                        "eps_f64": eps_pad_f64_b,
                        "d_var_f32": d_var_pad_f32_b,
                        "df_pay_f32": df_pay_pad_f32_b,
                        "w_fund_f32": w_fund_pad_f32_b,
                        "ko_f32": ko_pad_f32_b,
                        "is_call_f32": ic_pad_f32_b,
                        "cpn_f32": cp_pad_f32_b,
                        "is_guar_f32": ig_pad_f32_b,
                        "cpn_rates_f32": cr_pad_f32_b,
                        "eps_f32": eps_pad_f32_b,
                        "l_inv_t": l_inv_t_pad_b,
                        "c_inv": c_inv_pad_b,
                        "cdl_m": cdl_m_pad_b,
                        "cdl2_last": cdl2_last_pad_b,
                        "stat_pack": stat_pack_pad_b,
                        "pre_pack": pre_pack_pad_b,
                        "inv_perm": inv_perm_pad_b,
                        "th_cg_flags": th_cg_flags_g_b,
                        "th_payoff_f64_18": th_payoff_f64_18_g_b,
                        "lv_th_f32": lv_th_pad_f32_b,
                        "fwd_th_f64": fwd_th_pad_f64_b,
                        "tau_th_f64": tau_th_pad_f64_b,
                        "rn_th_f64": rn_th_pad_f64_b,
                        "rt_th_f64": rt_th_pad_f64_b,
                        "inv_dt_th_f64": inv_dt_th_pad_f64_b,
                        "ko_th_f64": ko_th_pad_f64_b,
                        "ic_th_f64": ic_th_pad_f64_b,
                        "cp_th_f64": cp_th_pad_f64_b,
                        "ig_th_f64": ig_th_pad_f64_b,
                        "cr_th_f64": cr_th_pad_f64_b,
                        "eps_th_f64": eps_th_pad_f64_b,
                        "th_pack": th_pack_pad_b,
                    }
                    bpk_src: List[torch.Tensor] = []
                    for bv_k in bpk.values():
                        if isinstance(bv_k, torch.Tensor):
                            bpk_src.append(bv_k)
                        elif isinstance(bv_k, tuple):
                            for sv_k in bv_k:
                                if isinstance(sv_k, torch.Tensor):
                                    bpk_src.append(sv_k)
                    bpk["src_tensors"] = _dedup_gpu_tensors(tuple(bpk_src))
                    if _warm_gpu is not None:
                        _warm_gpu.cg_bucket_pack = bpk
                    elif warm_bundle_key is not None:
                        warm_hit_check = _CRN_DEAL_PREP_CACHE.get(warm_bundle_key)
                        if warm_hit_check is not None:
                            warm_hit_check[1].cg_bucket_pack = bpk

            # Allocate a single contiguous float64 output buffer sliced into all GPU result views.
            q_mask = (
                bool(want_delta),
                bool(want_gamma),
                bool(want_cash_gamma),
                bool(want_vega),
                bool(want_rho),
                bool(want_divs),
                bool(want_theta),
                bool(want_cega),
            )
            cg_meta_key_w: Any = (
                q_mask,
                bool(return_cashflows),
                bool(use_cg_bucket),
                bool(use_mf_cv),
                mf_cheap_hit is not None,
                bool(want_q_aad_sde),
                bool(want_lv_aad_sde),
                bool(want_chol_aad_sde),
                bool(want_cf_wf_aad),
                bool(need_stage2_aad),
                bool(emit_stage1_gpu),
                bool(th_chunk_in_step1),
            )
            cg_meta_cached_w = (
                _warm_gpu.cg_meta_cache.get(cg_meta_key_w)
                if (_warm_gpu_hit and _warm_gpu is not None and not dyn_entries and not is_cont_ki_active)
                else None
            )
            cg_fast_d2h_np: Optional[npt.NDArray[np.float64]] = None
            cg_capturing_now = False
            cg_stream_entered = False
            cg_obj: Any = None
            cg_stream_ctx: Any = None
            cg_pool_lock: Any = None
            cur_sig_cg_cached: Optional[Tuple[int, ...]] = None
            out_layout: List[Tuple[str, Tuple[int, ...]]]
            pack_spans: Dict[str, Tuple[int, int, Tuple[int, ...]]]
            cg_shape_key: Any
            cg_src_tensors: List[torch.Tensor]
            if cg_meta_cached_w is not None:
                (
                    out_layout,
                    pack_spans,
                    tot_out_el,
                    can_cg_qmc,
                    cg_shape_key,
                    cg_src_tensors,
                    cur_sig_cg_cached,
                ) = cg_meta_cached_w
            else:
                out_layout = []
                if use_cg_bucket and bpk is not None:
                    d_out_b = int(bpk["d_pad"])
                    m_out_b = int(bpk["m_pad"])
                    ns_out_b = int(bpk["n_stat_pad"])
                    np_out_b = int(bpk["n_pre_pad"])
                    if emit_stage1_gpu:
                        out_layout.append(("stage1", (eff_r, 3 * m_out_b + 6)))
                    if ns_out_b > 0:
                        out_layout.append(("eval_pv", (ns_out_b,)))
                    if np_out_b > 0:
                        out_layout.append(("preterm", (np_out_b,)))
                    out_layout.append(("th_fused", (1,)))
                    if use_vibrato_gamma:
                        out_layout.append(("h_vib", (d_out_b, d_out_b)))
                    if want_cega or need_stage2_aad:
                        out_layout.append(("bar_chol", (d_out_b, d_out_b)))
                    if want_lv_aad_sde:
                        out_layout.append(("bar_lv", (d_out_b, lv_nt, lv_ny)))
                    if want_q_aad_sde or need_stage2_aad:
                        out_layout.append(("bar_q", (d_out_b, m_out_b * n_sub)))
                    if need_stage2_aad:
                        out_layout.append(("bar_lnf", (d_out_b, m_out_b)))
                    if want_cf_wf_aad:
                        out_layout.append(("bar_cf_wf", (2, m_out_b)))
                    if use_mf_cv:
                        out_layout.append(("s1_gbm_crn", (eff_r, 3 * m_out_b + 5)))
                        if want_cf_wf_aad:
                            out_layout.append(("cf_surv_gbm_crn", (eff_r, 2, m_out_b)))
                else:
                    if emit_stage1_gpu:
                        out_layout.append(("stage1", (eff_r, 3 * m_rem + 6)))
                    if gpu_eval_rows_total > 0:
                        out_layout.append(("eval_pv", (gpu_eval_rows_total,)))
                    if use_vibrato_gamma and m_rem > 1 and n_pre_rows > 0:
                        out_layout.append(("preterm", (n_pre_rows,)))
                    if (th_chunk_in_step1 and th_pack is not None) or (use_cg_bucket and bpk is not None):
                        out_layout.append(("th_fused", (1,)))
                    if use_vibrato_gamma:
                        out_layout.append(("h_vib", (d_assets, d_assets)))
                    if want_chol_aad_sde:
                        out_layout.append(("bar_chol", (d_assets, d_assets)))
                    if want_chol_aad_sde and is_cont_ki_active:
                        out_layout.append(("bar_sigma_cont", (d_assets, d_assets)))
                    if want_lv_aad_sde or (need_stage2_aad and is_cont_ki_active):
                        out_layout.append(("bar_lv", (d_assets, lv_nt, lv_ny)))
                    if want_q_aad_sde:
                        out_layout.append(("bar_q", (d_assets, n_steps_tot_g)))
                    if need_stage2_aad:
                        out_layout.append(("bar_lnf", (d_assets, m_rem)))
                    if want_cf_wf_aad:
                        out_layout.append(("bar_cf_wf", (2, m_rem)))
                    if use_mf_cv:
                        out_layout.append(("s1_gbm_crn", (eff_r, 3 * m_rem + 5)))
                        if mf_cheap_hit is None:
                            out_layout.append(("s1_gbm_cheap", (eff_r_mf * ema_group_mf, 3 * m_rem + 5)))
                        if want_cf_wf_aad:
                            out_layout.append(("cf_surv_gbm_crn", (eff_r, 2, m_rem)))
                            if mf_cheap_hit is None:
                                out_layout.append(("cf_surv_gbm_cheap", (eff_r_mf * ema_group_mf, 2, m_rem)))

                pack_spans = {}
                tot_out_el = 0
                for o_tag, o_shape in out_layout:
                    o_el = 1
                    for o_dim in o_shape:
                        o_el *= o_dim
                    if o_el > 0:
                        pack_spans[o_tag] = (tot_out_el, tot_out_el + o_el, o_shape)
                        tot_out_el += o_el

                can_cg_qmc = bool(
                    tot_out_el > 0
                    and (need_stage2_aad or use_vibrato_gamma or need_spot_bumps)
                    and can_triton_sobol
                    and (not dyn_entries)
                    and (not is_cont_ki_active)
                    and ((not use_mf_cv) or (mf_cheap_hit is not None))
                    and (not torch.cuda.is_current_stream_capturing())  # type: ignore[no-untyped-call]
                    and os.environ.get("DEEPVOL_QMC_CG", "1") != "0"
                )
                cg_shape_key = None
                cg_src_tensors = []

                if can_cg_qmc:
                    if use_cg_bucket and bpk is not None:
                        cg_shape_key = (
                            "cg_bucket_v1",
                            int(qmc_device.index or 0),
                            int(bpk["d_pad"]),
                            int(bpk["m_pad"]),
                            n_total,
                            n_sub,
                            eff_r,
                            n_rep,
                            lv_nt,
                            lv_ny,
                            int(bpk["r_chunk"]),
                            int(bpk["n_stat_pad"]),
                            int(bpk["n_pre_pad"]),
                            bool(use_mf_cv),
                            bool(emit_stage1_gpu),
                            bool(need_stage2_aad),
                            bool(want_cf_wf_aad),
                            bool(want_lv_aad_sde),
                            bool(want_cega or need_stage2_aad),
                            bool(want_q_aad_sde or need_stage2_aad),
                            bool(use_vibrato_gamma),
                            bool(need_spot_bumps),
                        )
                        cg_src_tensors = list(bpk["src_tensors"])
                    else:
                        cg_shape_key = (
                            int(qmc_device.index or 0),
                            d_assets,
                            m_rem,
                            n_total,
                            n_sub,
                            eff_r,
                            n_rep,
                            lv_nt,
                            lv_ny,
                            r_chunk,
                            q_mask,
                            tuple(out_layout),
                            n_stat_rows,
                            n_pre_rows,
                            bool(th_chunk_in_step1),
                            int(m_rem_th_g),
                            bool(use_f32_step1),
                            bool(fuse_m_sde),
                            bool(use_vibrato_gamma),
                            bool(vib_period_level),
                            bool(want_lv_aad_sde),
                            bool(want_chol_aad_sde),
                            bool(want_q_aad_sde),
                            bool(want_cf_wf_aad),
                            bool(use_mf_cv),
                            bool(apply_ema_mf),
                            int(eff_r_mf),
                            int(ema_group_mf),
                            int(n_rep_mf),
                            bool(need_spot_bumps),
                            bool(need_sp_cg_alt_g),
                            bool(use_f32_perf_direct),
                            call_code,
                            cpn_code,
                            eff_opt_code_s2,
                            bool(has_ki),
                            bool(k_os > 0.0),
                            bool(is_memory),
                            bool(is_long_flag),
                            bool(no_notional),
                            bool(force_cpn_ac),
                            bool(dir_down),
                            bool(type_out),
                            bool(has_lower_strike_flag),
                            bool(has_part_call_flag),
                            bool(gpu_meta["w_fund"] is not None),
                            int(stat_pack[2].shape[0]) if (stat_pack is not None and stat_pack[2] is not None) else -1,
                            bool(stat_pack is not None and stat_pack[5] is not None),
                        )
                        cg_src_candidates: List[Any] = [
                            lv_f32_g_cached,
                            chol_f32_g_cached,
                            chol_f64_g,
                            tau_step_g,
                            rn_step_g,
                            rt_step_g,
                            fwd_perf_g,
                            fwd_perf_f32_step_g,
                            sched_f32_g,
                            eps_aad_g,
                            u_rot_gpu,
                            v_rot_gpu,
                            shifts_sobol_gpu,
                            gpu_meta.get("ko"),
                            gpu_meta.get("is_call"),
                            gpu_meta.get("cpn"),
                            gpu_meta.get("is_guar"),
                            gpu_meta.get("cpn_rates"),
                            gpu_meta.get("inv_perm"),
                            gpu_meta.get("payoff_f64"),
                            gpu_meta.get("corr"),
                            d_var_f32_g,
                            fwd_perf_f32_g,
                            df_pay_f32_g,
                            w_fund_f32_g,
                            ko_f32_g,
                            is_call_f32_g,
                            cpn_f32_g,
                            is_guar_f32_g,
                            cpn_rates_f32_g,
                            eps_f32_g,
                            payoff_f32_g,
                            chol_scaled_f32_g,
                            l_inv_t_g,
                            c_inv_g,
                            cdl_m_g,
                            cdl2_last_g,
                            lv_sp_pm_g,
                            fwd_sp_pm_g,
                            lv_th_g,
                            fwd_th_g,
                            ko_th_g,
                            ic_th_g,
                            cp_th_g,
                            ig_th_g,
                            cr_th_g,
                            payoff_th_g,
                            eps_th_g,
                            df_th_g,
                            wf_th_g,
                        ]
                        for pk in (stat_pack, pre_pack, th_pack):
                            if pk is not None:
                                for item_p in pk:
                                    if isinstance(item_p, torch.Tensor):
                                        cg_src_candidates.append(item_p)
                        cg_src_tensors = [t_c for t_c in cg_src_candidates if isinstance(t_c, torch.Tensor)]
                cur_sig_cg_cached = tuple(s.data_ptr() for s in cg_src_tensors) if can_cg_qmc else None
                if _warm_gpu is not None and not dyn_entries and not is_cont_ki_active and ((not use_mf_cv) or (mf_cheap_hit is not None)):
                    _warm_gpu.cg_meta_cache[cg_meta_key_w] = (
                        out_layout,
                        pack_spans,
                        tot_out_el,
                        can_cg_qmc,
                        cg_shape_key,
                        cg_src_tensors,
                        cur_sig_cg_cached,
                    )

            if can_cg_qmc:
                cg_pool_lock = GLOBAL_GPU_CACHE_MANAGER.get_shared_graph_pool_lock(qmc_device)
                cg_pool_lock.acquire()
                try:
                    cg_hit = _CRN_QMC_CG_CACHE.get(cg_shape_key)
                    if cg_hit is not None:
                        cg_in_bufs = cg_hit["in_bufs"]
                        if cur_sig_cg_cached is not None and cg_hit.get("last_ptrs") == cur_sig_cg_cached:
                            cg_hit["cg"].replay()
                            out_pin = cg_hit.get("out_pin")
                            if out_pin is None or out_pin.shape != cg_hit["out_flat_g"].shape:
                                out_pin = torch.empty_like(cg_hit["out_flat_g"], device="cpu", pin_memory=True)
                                cg_hit["out_pin"] = out_pin
                            out_pin.copy_(cg_hit["out_flat_g"], non_blocking=False)
                            cg_fast_d2h_np = out_pin.numpy().copy()
                        elif len(cg_in_bufs) == len(cg_src_tensors) and all(
                            b.shape == s.shape and b.dtype == s.dtype for b, s in zip(cg_in_bufs, cg_src_tensors)
                        ):
                            cur_sig_cg = cur_sig_cg_cached if cur_sig_cg_cached is not None else tuple(s.data_ptr() for s in cg_src_tensors)
                            if cg_hit.get("last_ptrs") != cur_sig_cg:
                                for b_dst, s_src in zip(cg_in_bufs, cg_src_tensors):
                                    b_dst.copy_(s_src)
                                cg_hit["last_ptrs"] = cur_sig_cg
                            cg_hit["cg"].replay()
                            out_pin = cg_hit.get("out_pin")
                            if out_pin is None or out_pin.shape != cg_hit["out_flat_g"].shape:
                                out_pin = torch.empty_like(cg_hit["out_flat_g"], device="cpu", pin_memory=True)
                                cg_hit["out_pin"] = out_pin
                            out_pin.copy_(cg_hit["out_flat_g"], non_blocking=False)
                            cg_fast_d2h_np = out_pin.numpy().copy()
                except Exception:
                    if cg_pool_lock is not None:
                        cg_pool_lock.release()
                        cg_pool_lock = None
                    raise
                if cg_fast_d2h_np is not None and cg_pool_lock is not None:
                    cg_pool_lock.release()
                    cg_pool_lock = None

            try:
                out_flat_g: Optional[torch.Tensor] = (
                    torch.zeros((tot_out_el,), dtype=torch.float64, device=qmc_device)
                    if (tot_out_el > 0 and cg_fast_d2h_np is None)
                    else None
                )

                def _out_view(tag: str) -> Optional[torch.Tensor]:
                    if out_flat_g is None:
                        return None
                    sp_o = pack_spans.get(tag)
                    if sp_o is None:
                        return None
                    return out_flat_g[sp_o[0] : sp_o[1]].view(sp_o[2])

                stage1_rep_g = _out_view("stage1")
                gpu_eval_pv_g = _out_view("eval_pv")
                pv_preterm_g = _out_view("preterm")
                pv_th_fused_g = _out_view("th_fused")
                h_vib_tot_g = _out_view("h_vib")
                bar_chol_tot_g = _out_view("bar_chol")
                bar_sigma_cont_tot_g = _out_view("bar_sigma_cont")
                bar_lv_tot_g = _out_view("bar_lv")
                bar_q_tot_g = _out_view("bar_q")
                bar_ln_fwd_tot_g = _out_view("bar_lnf")
                bar_cf_wf_tot_g = _out_view("bar_cf_wf")
                s1_gbm_crn_g = _out_view("s1_gbm_crn")
                s1_gbm_cheap_g = _out_view("s1_gbm_cheap")
                cf_surv_gbm_crn_g = _out_view("cf_surv_gbm_crn")
                cf_surv_gbm_cheap_g = _out_view("cf_surv_gbm_cheap")
                stat_direct_alias = bool(
                    (not dyn_entries)
                    and n_stat_rows > 0
                    and gpu_eval_pv_g is not None
                    and (
                        (use_cg_bucket and bpk is not None and n_stat_rows == gpu_eval_rows_total)
                        or ((not use_cg_bucket) and n_stat_rows == gpu_eval_rows_total)
                    )
                )
                stat_acc_g: Optional[torch.Tensor] = (
                    gpu_eval_pv_g
                    if stat_direct_alias
                    else (
                        torch.zeros(
                            (int(bpk["n_stat_pad"]) if (use_cg_bucket and bpk is not None) else n_stat_rows,),
                            dtype=torch.float64,
                            device=qmc_device,
                        )
                        if (n_stat_rows > 0 and cg_fast_d2h_np is None)
                        else None
                    )
                )

                d_eff_k = int(bpk["d_pad"]) if (use_cg_bucket and bpk is not None) else d_assets
                m_eff_k = int(bpk["m_pad"]) if (use_cg_bucket and bpk is not None) else m_rem
                n_steps_eff_k = m_eff_k * n_sub
                r_chunk_eff = int(bpk["r_chunk"]) if (use_cg_bucket and bpk is not None) else r_chunk
                lv_f32_eff = bpk["lv_f32"] if (use_cg_bucket and bpk is not None) else lv_f32_g_cached
                chol_f32_eff = bpk["chol_f32"] if (use_cg_bucket and bpk is not None) else chol_f32_g_cached
                chol_f64_eff = bpk["chol_f64"] if (use_cg_bucket and bpk is not None) else chol_f64_g
                y_min_f32_eff = bpk["y_min_f32"] if (use_cg_bucket and bpk is not None) else y_min_f32_g
                inv_dy_f32_eff = bpk["inv_dy_f32"] if (use_cg_bucket and bpk is not None) else inv_dy_f32_g
                t_mid_eff = bpk["t_mid_ptr"] if (use_cg_bucket and bpk is not None) else t_mid_ptr_g
                dt_sub_eff = bpk["dt_sub_ptr"] if (use_cg_bucket and bpk is not None) else dt_sub_ptr_g
                sqrt_dt_eff = bpk["sqrt_dt_ptr"] if (use_cg_bucket and bpk is not None) else sqrt_dt_ptr_g
                q_shift_eff = bpk["q_shift_ptr"] if (use_cg_bucket and bpk is not None) else q_shift_mid_ptr_g
                fwd_f32_eff = bpk["fwd_pad_f32"] if (use_cg_bucket and bpk is not None) else fwd_perf_f32_step_g
                fwd_f64_eff = bpk["fwd_pad_f64"] if (use_cg_bucket and bpk is not None) else fwd_perf_g
                cg_flags_eff = bpk["cg_flags"] if (use_cg_bucket and bpk is not None) else None
                pf64_eff = bpk["payoff_f64_18"] if (use_cg_bucket and bpk is not None) else gpu_meta["payoff_f64"]
                ko_eff = bpk["ko_f64"] if (use_cg_bucket and bpk is not None) else gpu_meta["ko"]
                ic_eff = bpk["is_call_f64"] if (use_cg_bucket and bpk is not None) else gpu_meta["is_call"]
                cp_eff = bpk["cpn_f64"] if (use_cg_bucket and bpk is not None) else gpu_meta["cpn"]
                ig_eff = bpk["is_guar_f64"] if (use_cg_bucket and bpk is not None) else gpu_meta["is_guar"]
                cr_eff = bpk["cpn_rates_f64"] if (use_cg_bucket and bpk is not None) else gpu_meta["cpn_rates"]
                df_eff = bpk["df_pay_f64"] if (use_cg_bucket and bpk is not None) else gpu_meta["df_pay"]
                wf_eff = bpk["w_fund_f64"] if (use_cg_bucket and bpk is not None) else gpu_meta["w_fund"]
                eps_eff = bpk["eps_f64"] if (use_cg_bucket and bpk is not None) else eps_aad_g
                inv_perm_eff = bpk["inv_perm"] if (use_cg_bucket and bpk is not None) else gpu_meta["inv_perm"]
                l_inv_t_eff = bpk["l_inv_t"] if (use_cg_bucket and bpk is not None) else l_inv_t_g
                c_inv_eff = bpk["c_inv"] if (use_cg_bucket and bpk is not None) else c_inv_g
                cdl_m_eff = bpk["cdl_m"] if (use_cg_bucket and bpk is not None) else cdl_m_g
                cdl2_last_eff = bpk["cdl2_last"] if (use_cg_bucket and bpk is not None) else cdl2_last_g
                lv_sp_pm_eff = bpk["lv_sp_pm_f32"] if (use_cg_bucket and bpk is not None) else lv_sp_pm_g
                fwd_sp_pm_eff = bpk["fwd_sp_pm_f64"] if (use_cg_bucket and bpk is not None) else fwd_sp_pm_g
                sp_pm_meta_eff = bpk["sp_pm_meta"] if (use_cg_bucket and bpk is not None) else sp_pm_meta_g
                y_min_f64_eff = bpk["y_min_f64"] if (use_cg_bucket and bpk is not None) else y_min_base_g
                inv_dy_f64_eff = bpk["inv_dy_f64"] if (use_cg_bucket and bpk is not None) else inv_dy_base_g
                inv_dt_f64_eff = bpk["inv_dt_f64"] if (use_cg_bucket and bpk is not None) else inv_dt_base_g[0:1]
                tau_f64_eff = bpk["tau_f64"] if (use_cg_bucket and bpk is not None) else tau_step_g[:m_rem]
                rn_f64_eff = bpk["rn_f64"] if (use_cg_bucket and bpk is not None) else rn_step_g[: d_assets * m_rem].view(d_assets, m_rem)
                rt_f64_eff = bpk["rt_f64"] if (use_cg_bucket and bpk is not None) else rt_step_g[: d_assets * m_rem].view(d_assets, m_rem)
                stat_pack_eff = bpk["stat_pack"] if (use_cg_bucket and bpk is not None) else stat_pack
                pre_pack_eff = bpk["pre_pack"] if (use_cg_bucket and bpk is not None) else pre_pack

                if can_cg_qmc and cg_fast_d2h_np is None and out_flat_g is not None and cg_pool_lock is not None:
                    try:
                        cg_stream = _get_qmc_cg_stream(qmc_device)
                        _n_sh_ar = int(os.environ.get("DEEPVOL_LV_SHARDS", "128")) if want_lv_aad_sde else 1
                        _nb_ar = max(
                            64,
                            n_stat_rows,
                            n_pre_rows,
                            int(bpk["n_stat_pad"]) if (use_cg_bucket and bpk is not None) else 0,
                            int(bpk["n_pre_pad"]) if (use_cg_bucket and bpk is not None) else 0,
                        )
                        _nt_ev_ar = max(16, (r_chunk_eff * n_rep + 63) // 64)
                        _ensure_qmc_cg_scratch_arena(
                            qmc_device,
                            d_eff_k,
                            m_eff_k,
                            n_sub,
                            r_chunk_eff,
                            n_rep,
                            lv_nt,
                            lv_ny,
                            _n_sh_ar,
                            _nb_ar,
                            _nt_ev_ar,
                            max(m_eff_k, int(m_rem_th_g)),
                            max(d_eff_k, 16),
                        )
                        cg_stream.wait_stream(torch.cuda.current_stream(qmc_device))
                        with torch.cuda.stream(cg_stream):
                            if use_cg_bucket and bpk is not None:
                                _warm_z_cg = _get_z_indep_chunk_gpu(
                                    0,
                                    min(eff_r, r_chunk_eff),
                                    allow_full_cache=False,
                                    shifts_ov=bpk["shifts"],
                                    u_rot_ov=bpk["u_rot"],
                                    m_pad_ov=m_eff_k,
                                    d_pad_ov=d_eff_k,
                                    cg_flags_ov=cg_flags_eff,
                                    use_rt_flags=True,
                                    use_cg_arena=True,
                                )
                            else:
                                _warm_z_cg = _get_z_indep_chunk_gpu(
                                    0,
                                    min(eff_r, r_chunk_eff),
                                    allow_full_cache=False,
                                    use_cg_arena=True,
                                )
                            if use_f32_step1 and (need_fwd_ckpt or need_spot_bumps or th_chunk_in_step1 or use_mf_cv):
                                _warm_zc_cg = _alloc_cg_buf(
                                    True,
                                    qmc_device,
                                    "z_corr",
                                    (m_eff_k * n_sub, d_eff_k, min(eff_r, r_chunk_eff) * n_rep),
                                    torch.float32,
                                )
                                torch.matmul(chol_f32_eff, _warm_z_cg, out=_warm_zc_cg)
                                del _warm_zc_cg
                            del _warm_z_cg
                        cg_stream.synchronize()
                        torch.cuda.empty_cache()
                        cg_before_stats_qmc = torch.cuda.memory_stats(qmc_device)
                        cg_obj = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]
                        cg_stream_ctx = torch.cuda.stream(cg_stream)
                        cg_stream_ctx.__enter__()
                        cg_stream_entered = True
                        cg_obj.capture_begin(
                            pool=GLOBAL_GPU_CACHE_MANAGER.get_shared_graph_pool(qmc_device),
                            capture_error_mode="thread_local",
                        )
                        cg_capturing_now = True
                        out_flat_g.zero_()
                        if stat_acc_g is not None and (not stat_direct_alias):
                            stat_acc_g.zero_()
                    except Exception:
                        if cg_capturing_now and cg_obj is not None:
                            try:
                                cg_capturing_now = False
                                cg_obj.capture_end()
                            except Exception as end_exc:
                                logger.debug("QMC CUDAGraph capture_end cleanup failed: %s", end_exc)
                        if cg_stream_entered and cg_stream_ctx is not None:
                            try:
                                cg_stream_entered = False
                                cg_stream_ctx.__exit__(None, None, None)
                            except Exception as ctx_exc:
                                logger.debug("QMC CUDAGraph stream exit cleanup failed: %s", ctx_exc)
                        if cg_pool_lock is not None:
                            cg_pool_lock.release()
                            cg_pool_lock = None
                        cg_capturing_now = False
                        cg_stream_entered = False
                        cg_obj = None
                        cg_stream_ctx = None

                z_corr_full_f32: Optional[torch.Tensor] = None
                z_corr_coarse_f32: Optional[torch.Tensor] = None
                if (
                    (not use_cg_bucket)
                    and (not cg_capturing_now)
                    and cg_fast_d2h_np is None
                    and use_f32_step1
                    and z_indep_full_f32 is not None
                    and (need_fwd_ckpt or need_spot_bumps or th_chunk_in_step1 or use_mf_cv)
                ):
                    z_corr_bytes = (m_rem * n_sub + (m_rem if use_mf_cv else 0)) * d_assets * n_total * 4
                    if z_corr_bytes <= 18 * 1024 * 1024:
                        z_corr_key = ("zcorr_v1", rot_sobol_key, chol_base.tobytes(), bool(use_mf_cv))
                        zc_hit = _CRN_ZCORR_CACHE.get(z_corr_key)
                        if (
                            isinstance(zc_hit, tuple)
                            and len(zc_hit) == 2
                            and isinstance(zc_hit[0], torch.Tensor)
                            and zc_hit[0].device == qmc_device
                        ):
                            z_corr_full_f32, z_corr_coarse_f32 = zc_hit
                        else:
                            z_corr_full_f32 = torch.matmul(chol_f32_g_cached, z_indep_full_f32)
                            z_corr_coarse_f32 = (
                                z_corr_full_f32.view(m_rem, n_sub, d_assets, n_total).sum(dim=1)
                                if use_mf_cv
                                else None
                            )
                            _CRN_ZCORR_CACHE.put(z_corr_key, (z_corr_full_f32, z_corr_coarse_f32))

                single_chunk = bool(eff_r <= r_chunk_eff)
                defer_bwd_shard_sum = bool(
                    fuse_m_sde
                    and (not is_cont_ki_active)
                    and (eff_r > r_chunk_eff)
                    and (eff_r % r_chunk_eff == 0)
                )
                bar_q_acc_c: Optional[torch.Tensor] = None
                bar_lv_acc_f32_c: Optional[torch.Tensor] = None
                inv_sqrt_nsub_val = 1.0 / math.sqrt(float(n_sub))
                for r_start in ([] if cg_fast_d2h_np is not None else range(0, eff_r, r_chunk_eff)):
                    r_end = min(eff_r, r_start + r_chunk_eff)
                    r_cnt_c = r_end - r_start
                    n_chunk = r_cnt_c * n_rep
                    w_chunk = float(r_cnt_c) / float(eff_r)
                    if use_cg_bucket and bpk is not None:
                        z_indep_f32_c = _get_z_indep_chunk_gpu(
                            r_start,
                            r_end,
                            allow_full_cache=False,
                            shifts_ov=bpk["shifts"],
                            u_rot_ov=bpk["u_rot"],
                            m_pad_ov=m_eff_k,
                            d_pad_ov=d_eff_k,
                            cg_flags_ov=cg_flags_eff,
                            use_rt_flags=True,
                            use_cg_arena=cg_capturing_now,
                        )
                    else:
                        z_indep_f32_c = _get_z_indep_chunk_gpu(
                            r_start,
                            r_end,
                            allow_full_cache=(not cg_capturing_now),
                            use_cg_arena=cg_capturing_now,
                        )

                    lx_period_c: Optional[torch.Tensor] = None
                    vib_lx_pre_c: Optional[torch.Tensor] = None
                    vib_j_pre_c: Optional[torch.Tensor] = None
                    vib_sig_pre_c: Optional[torch.Tensor] = None
                    z_corr_c: Optional[torch.Tensor] = None
                    z_corr_f32_mdn_c: Optional[torch.Tensor] = None
                    if use_f32_step1 and (need_fwd_ckpt or need_spot_bumps or th_chunk_in_step1 or use_mf_cv):
                        if z_corr_full_f32 is not None:
                            z_corr_f32_mdn_c = (
                                z_corr_full_f32
                                if single_chunk
                                else z_corr_full_f32[:, :, r_start * n_rep : r_end * n_rep].contiguous()
                            )
                        else:
                            z_corr_f32_mdn_c = _alloc_cg_buf(
                                cg_capturing_now,
                                qmc_device,
                                "z_corr",
                                (m_eff_k * n_sub, d_eff_k, n_chunk),
                                torch.float32,
                            )
                            torch.matmul(chol_f32_eff, z_indep_f32_c, out=z_corr_f32_mdn_c)
                        if not (want_chol_aad_sde or use_vibrato_gamma or (need_stage2_aad and not fuse_m_sde)):
                            del z_indep_f32_c
                            z_indep_f32_c = dummy_f32_1

                    perf_base_c: Optional[torch.Tensor] = None
                    norm_x_c: Optional[torch.Tensor] = None
                    if need_fwd_ckpt:
                        lx_period_c = _alloc_cg_buf(
                            cg_capturing_now,
                            qmc_device,
                            "lx_period",
                            (m_eff_k, d_eff_k, n_chunk),
                            torch.float32,
                            zero=False,
                        )
                        if use_vibrato_gamma:
                            vib_lx_pre_c = _alloc_cg_buf(cg_capturing_now, qmc_device, "vib_lx", (d_eff_k, n_chunk), torch.float32, zero=False)
                            vib_j_pre_c = _alloc_cg_buf(cg_capturing_now, qmc_device, "vib_j", (d_eff_k, n_chunk), torch.float32, zero=False)
                            vib_sig_pre_c = _alloc_cg_buf(cg_capturing_now, qmc_device, "vib_sig", (d_eff_k, n_chunk), torch.float32, zero=False)
                            vib_j_state_c = (
                                torch.ones((d_eff_k, n_chunk), dtype=torch.float32, device=qmc_device)
                                if not fuse_m_sde
                                else dummy_f32_1
                            )
                        else:
                            vib_j_state_c = dummy_f32_1
                        use_zc_fwd = bool(z_corr_f32_mdn_c is not None)
                        z_fwd_in = z_corr_f32_mdn_c if z_corr_f32_mdn_c is not None else z_indep_f32_c
                        if fuse_m_sde:
                            if use_f32_perf_direct:
                                perf_base_c = _alloc_cg_buf(
                                    cg_capturing_now,
                                    qmc_device,
                                    "perf_base",
                                    (m_eff_k, d_eff_k, n_chunk),
                                    torch.float32,
                                    zero=False,
                                )
                            _fast_triton_launch(
                                _qmc_fwd_checkpoint_kernel,
                                (1, r_cnt_c * d_eff_k),
                                z_fwd_in,
                                dummy_shifts_g,
                                chol_f32_eff,
                                lv_f32_eff,
                                y_min_f32_eff,
                                inv_dy_f32_eff,
                                t_mid_eff,
                                dt_sub_eff,
                                sqrt_dt_eff,
                                q_shift_eff,
                                dummy_shifts_g,
                                dummy_f32_1,
                                dummy_f32_1,
                                lx_period_c,
                                vib_lx_pre_c if vib_lx_pre_c is not None else dummy_f32_1,
                                vib_j_pre_c if vib_j_pre_c is not None else dummy_f32_1,
                                vib_sig_pre_c if vib_sig_pre_c is not None else dummy_f32_1,
                                vib_j_state_c,
                                fwd_f32_eff,
                                perf_base_c if perf_base_c is not None else dummy_f32_1,
                                inv_dt_f32_val,
                                d_eff_k,
                                m_eff_k,
                                1,
                                vib_period_idx,
                                vib_sub_idx,
                                cg_flags_ptr=cg_flags_eff,
                                payoff_f64_ptr=pf64_eff if use_cg_bucket else None,
                                n_rep=n_rep,
                                n_sub=n_sub,
                                N_t=lv_nt,
                                N_y=lv_ny,
                                BLOCK_N=block_n_sde,
                                period_idx=0,
                                FUSE_M=True,
                                SAVE_VIBRATO=use_vibrato_gamma,
                                VIB_PERIOD_LEVEL=False if use_cg_bucket else vib_period_level,
                                USE_Z_CORR=use_zc_fwd,
                                EMIT_PERF=use_f32_perf_direct,
                                USE_RT_FLAGS=use_cg_bucket,
                                num_warps=nw_sde,
                            )
                        else:
                            tile_max_c = torch.empty((r_cnt_c, n_tiles_sde, d_assets), dtype=torch.float32, device=qmc_device)
                            tile_ex_sum_c = torch.empty((r_cnt_c, n_tiles_sde, d_assets), dtype=torch.float32, device=qmc_device)
                            for m_cur_s in range(m_rem):
                                _fast_triton_launch(
                                    _qmc_fwd_checkpoint_kernel,
                                    (n_tiles_sde, r_cnt_c * d_assets),
                                    z_fwd_in,
                                    dummy_shifts_g,
                                    chol_f32_g_cached,
                                    lv_f32_g_cached,
                                    y_min_f32_g,
                                    inv_dy_f32_g,
                                    t_mid_ptr_g,
                                    dt_sub_ptr_g,
                                    sqrt_dt_ptr_g,
                                    q_shift_mid_ptr_g,
                                    dummy_shifts_g,
                                    tile_max_c,
                                    tile_ex_sum_c,
                                    lx_period_c,
                                    vib_lx_pre_c if vib_lx_pre_c is not None else dummy_f32_1,
                                    vib_j_pre_c if vib_j_pre_c is not None else dummy_f32_1,
                                    vib_sig_pre_c if vib_sig_pre_c is not None else dummy_f32_1,
                                    vib_j_state_c,
                                    fwd_perf_f32_step_g,
                                    dummy_f32_1,
                                    inv_dt_f32_val,
                                    d_assets,
                                    m_rem,
                                    n_tiles_sde,
                                    vib_period_idx,
                                    vib_sub_idx,
                                    n_rep=n_rep,
                                    n_sub=n_sub,
                                    N_t=lv_nt,
                                    N_y=lv_ny,
                                    BLOCK_N=block_n_sde,
                                    period_idx=m_cur_s,
                                    FUSE_M=False,
                                    SAVE_VIBRATO=use_vibrato_gamma,
                                    VIB_PERIOD_LEVEL=vib_period_level,
                                    USE_Z_CORR=use_zc_fwd,
                                    EMIT_PERF=False,
                                    num_warps=nw_sde,
                                )
                                m_gmax = tile_max_c.max(dim=1, keepdim=True).values
                                ex_sum_rep = (tile_ex_sum_c * torch.exp(tile_max_c - m_gmax)).sum(dim=1, keepdim=True)
                                log_m_c = (m_gmax + torch.log((ex_sum_rep / float(n_rep)).clamp(min=1e-30))).squeeze(1).T.unsqueeze(-1)
                                lx_view_m = lx_period_c[m_cur_s].view(d_assets, r_cnt_c, n_rep)
                                lx_view_m.sub_(log_m_c)
                                if use_vibrato_gamma:
                                    omega_m = torch.exp(lx_view_m) / float(n_rep)
                                    j_dev_m = vib_j_state_c.view(d_assets, r_cnt_c, n_rep) - 1.0
                                    vib_j_state_c.copy_((1.0 + (j_dev_m - (omega_m * j_dev_m).sum(dim=2, keepdim=True))).reshape(d_assets, n_chunk))
                            del tile_max_c, tile_ex_sum_c
                        del vib_j_state_c
                        if perf_base_c is None:
                            if need_norm_x_kept:
                                norm_x_c = torch.exp(lx_period_c.to(dtype=torch.float64)).transpose(0, 1).contiguous()
                                perf_base_c = (norm_x_c * fwd_f64_eff.unsqueeze(-1)).permute(1, 0, 2).contiguous()
                            else:
                                perf_base_c = (torch.exp(lx_period_c.to(dtype=torch.float64)) * fwd_f64_eff.T.unsqueeze(-1)).contiguous()
                        if not need_stage2_aad:
                            del lx_period_c
                            lx_period_c = None
                    else:
                        z_corr_c = torch.matmul(
                            chol_f64_g,
                            z_indep_f32_c.to(dtype=torch.float64).permute(1, 0, 2).reshape(d_assets, -1),
                        ).view(d_assets, m_rem, n_sub, n_chunk).contiguous()
                        lv_step_in = torch.as_tensor(base_lv_tables, dtype=torch.float64, device=qmc_device).contiguous()
                        norm_x_c = _gpu_step_1d_f64(
                            lv_step_in,
                            z_corr_c,
                            tau_step_g[:m_rem],
                            rn_step_g[: d_assets * m_rem].view(d_assets, m_rem),
                            rt_step_g[: d_assets * m_rem].view(d_assets, m_rem),
                            2,
                            meta_lv_t=id_d_g,
                            meta_r_t=id_d_g,
                            meta_z_t=id_d_g,
                        )
                        del lv_step_in
                        perf_base_c = (norm_x_c * fwd_perf_g.unsqueeze(-1)).permute(1, 0, 2).contiguous()
                        if not need_norm_x_kept:
                            del norm_x_c
                            norm_x_c = None

                    assert perf_base_c is not None

                    # Evaluate Stage-1 & Stage-2 payoff adjoints + Continuous KI probabilities for this chunk.
                    i_ki_ext_c: Optional[torch.Tensor] = None
                    if is_cont_ki_active and (need_stage2_aad or emit_stage1_gpu or gpu_eval_rows_total > 0):
                        s1_c_box: List[torch.Tensor] = []
                        iki_c_box: List[torch.Tensor] = []
                        cg_c_box: List[torch.Tensor] = []
                        bar_lx_post_c, bar_ln_fwd_c, bar_lv_cont_c = _compute_payoff_aad_state_adjoints_torch(
                            perf_base_c,
                            ko_barriers=gpu_meta["ko"],
                            is_callable=gpu_meta["is_call"],
                            cpn_barriers=gpu_meta["cpn"],
                            is_guar=gpu_meta["is_guar"],
                            cpn_rates=gpu_meta["cpn_rates"],
                            df_pay=gpu_meta["df_pay"],
                            w_fund=gpu_meta["w_fund"],
                            inv_perm=inv_perm,
                            call_code=call_code,
                            cpn_code=cpn_code,
                            opt_code=opt_code,
                            ko_spread=ko_spread,
                            cpn_spread=cpn_spread,
                            ki_spread=ki_spread,
                            cpn_dir_down=dir_down,
                            cpn_type_out=type_out,
                            is_memory=is_memory,
                            force_cpn_ac=force_cpn_ac,
                            no_notional=no_notional,
                            is_long_basket=is_long_flag,
                            has_ki=has_ki,
                            ki_barrier=b_ki,
                            i_ki_past=float(state.i_ki_past),
                            ki_monitoring_type=ki_monitoring_type,
                            tau_obs=tau_obs_np,
                            vol_atm_obs=vol_atm_obs,
                            strike=k_put,
                            lower_strike=state.lower_strike,
                            option_gearing=float(state.option_gearing),
                            ac_cpn=float(state.autocall_coupon),
                            c_mem_0=float(state.c_mem_0),
                            part_call_strike=state.participation_call_strike,
                            part_call_rate=part_call_rate,
                            one_star_strike=(k_os if k_os > 0.0 else state.one_star_strike),
                            one_star_coupon=os_cpn,
                            one_star_eps=one_star_eps,
                            barrier_eps_min=barrier_eps_min,
                            barrier_eps_max=barrier_eps_max,
                            lv_tables_t=lv_f32_g_cached,
                            corr_t=gpu_meta["corr"],
                            fwd_perf_t=fwd_perf_g,
                            y_min=lv_y_min_vec,
                            y_max=lv_y_max_vec,
                            n_sub=n_sub,
                            corr_grad_out=cg_c_box if (want_cega and d_assets > 1) else None,
                            n_rep=n_rep,
                            emit_stage1=emit_stage1_gpu,
                            out_stage1_rep=s1_c_box,
                            compute_grads=need_stage2_aad,
                            out_i_ki_ext=iki_c_box,
                        )
                        i_ki_ext_c = iki_c_box[0] if iki_c_box else None
                        if stage1_rep_g is not None and s1_c_box:
                            stage1_rep_g[r_start:r_end] = s1_c_box[0]
                        if bar_ln_fwd_tot_g is not None:
                            bar_ln_fwd_tot_g += w_chunk * bar_ln_fwd_c
                        if bar_lv_tot_g is not None and bar_lv_cont_c is not None:
                            bar_lv_tot_g += w_chunk * bar_lv_cont_c
                        if bar_sigma_cont_tot_g is not None and cg_c_box:
                            bar_sigma_cont_tot_g += w_chunk * cg_c_box[0]
                    elif use_triton_aad_s2 and (need_stage2_aad or emit_stage1_gpu):
                        s1_chunk_g = (
                            stage1_rep_g[r_start:r_end]
                            if stage1_rep_g is not None
                            else None
                        )
                        bar_lx_post_c, bar_ln_fwd_c, bar_cf_wf_c = _compute_payoff_aad_state_adjoints_triton(
                            perf_base_c,
                            ko_barriers=ko_eff,
                            is_callable=ic_eff,
                            cpn_barriers=cp_eff,
                            is_guar=ig_eff,
                            cpn_rates=cr_eff,
                            df_pay=df_eff,
                            w_fund=wf_eff,
                            inv_perm_g=inv_perm_eff,
                            payoff_f64_g=pf64_eff,
                            call_code=call_code,
                            cpn_code=cpn_code,
                            eff_opt_code=eff_opt_code_s2,
                            cpn_dir_down=dir_down,
                            cpn_type_out=type_out,
                            is_memory=is_memory,
                            force_cpn_ac=force_cpn_ac,
                            no_notional=no_notional,
                            is_long_basket=is_long_flag,
                            has_ki=has_ki,
                            ki_barrier=b_ki,
                            i_ki_past=float(state.i_ki_past),
                            has_one_star=bool(k_os > 0.0),
                            tau_obs=tau_obs_np,
                            vol_atm_obs=vol_atm_obs,
                            barrier_eps_min=barrier_eps_min,
                            barrier_eps_max=barrier_eps_max,
                            compute_cf_wf=want_cf_wf_aad,
                            eps_g_cached=eps_eff,
                            n_rep=n_rep,
                            has_lower_strike=has_lower_strike_flag,
                            has_part_call=has_part_call_flag,
                            emit_stage1=emit_stage1_gpu,
                            out_stage1_rep_g=s1_chunk_g,
                            cg_flags_g=cg_flags_eff,
                            use_rt_flags=use_cg_bucket,
                            d_pad_k=d_eff_k,
                            use_cg_arena=cg_capturing_now,
                        )
                        if bar_ln_fwd_tot_g is not None:
                            if single_chunk:
                                bar_ln_fwd_tot_g.copy_(bar_ln_fwd_c)
                            else:
                                bar_ln_fwd_tot_g.add_(bar_ln_fwd_c, alpha=w_chunk)
                        if bar_cf_wf_tot_g is not None and bar_cf_wf_c is not None:
                            if single_chunk:
                                bar_cf_wf_tot_g.copy_(bar_cf_wf_c)
                            else:
                                bar_cf_wf_tot_g.add_(bar_cf_wf_c, alpha=w_chunk)
                        if not need_stage2_aad:
                            del bar_lx_post_c
                            bar_lx_post_c = None
                    else:
                        bar_lx_post_c = None

                    # Backpropagate bar_lx_post_c through rematerialized Platen substeps using parallel period shards.
                    if need_stage2_aad and bar_lx_post_c is not None and lx_period_c is not None:
                        want_q_launch = bool(want_q_aad_sde or (use_cg_bucket and bar_q_tot_g is not None))
                        want_chol_launch = bool(want_chol_aad_sde or (use_cg_bucket and bar_chol_tot_g is not None))
                        if defer_bwd_shard_sum:
                            if bar_q_acc_c is None and want_q_launch:
                                bar_q_acc_c = _alloc_cg_buf(
                                    cg_capturing_now,
                                    qmc_device,
                                    "bar_q",
                                    (d_eff_k, n_steps_eff_k),
                                    torch.float32,
                                    zero=True,
                                )
                            bar_q_c = bar_q_acc_c if (want_q_launch and bar_q_acc_c is not None) else dummy_f32_1
                        else:
                            bar_q_c = (
                                _alloc_cg_buf(cg_capturing_now, qmc_device, "bar_q", (d_eff_k, n_steps_eff_k), torch.float32, zero=True)
                                if want_q_launch
                                else dummy_f32_1
                            )
                        if fuse_m_sde:
                            n_lv_shards_bwd = int(os.environ.get("DEEPVOL_LV_SHARDS", "128")) if want_lv_aad_sde else 1
                            if defer_bwd_shard_sum:
                                if bar_lv_acc_f32_c is None and want_lv_aad_sde:
                                    bar_lv_acc_f32_c = _alloc_cg_buf(
                                        cg_capturing_now,
                                        qmc_device,
                                        "bar_lv",
                                        (n_lv_shards_bwd, d_eff_k, lv_nt, lv_ny),
                                        torch.float32,
                                        zero=True,
                                    )
                                bar_lv_f32_c = bar_lv_acc_f32_c if (want_lv_aad_sde and bar_lv_acc_f32_c is not None) else dummy_f32_1
                            else:
                                bar_lv_f32_c = (
                                    _alloc_cg_buf(
                                        cg_capturing_now,
                                        qmc_device,
                                        "bar_lv",
                                        (n_lv_shards_bwd, d_eff_k, lv_nt, lv_ny),
                                        torch.float32,
                                        zero=True,
                                    )
                                    if want_lv_aad_sde
                                    else dummy_f32_1
                                )
                            bar_zc_c = (
                                _alloc_cg_buf(
                                    cg_capturing_now,
                                    qmc_device,
                                    "z_raw_or_bzc",
                                    (n_steps_eff_k, d_eff_k, n_chunk),
                                    torch.float32,
                                    zero=use_cg_bucket,
                                )
                                if want_chol_launch
                                else dummy_f32_1
                            )
                            use_zc_bwd = bool(z_corr_f32_mdn_c is not None)
                            z_bwd_prop_in = z_corr_f32_mdn_c if z_corr_f32_mdn_c is not None else z_indep_f32_c
                            _fast_triton_launch(
                                _qmc_bwd_state_prop_kernel,
                                (1, r_cnt_c * d_eff_k),
                                z_bwd_prop_in,
                                chol_f32_eff,
                                lv_f32_eff,
                                y_min_f32_eff,
                                inv_dy_f32_eff,
                                t_mid_eff,
                                dt_sub_eff,
                                sqrt_dt_eff,
                                q_shift_eff,
                                bar_lx_post_c,
                                lx_period_c,
                                dummy_f32_1,
                                inv_dt_f32_val,
                                d_eff_k,
                                m_eff_k,
                                cg_flags_ptr=cg_flags_eff,
                                payoff_f64_ptr=pf64_eff if use_cg_bucket else None,
                                n_rep=n_rep,
                                n_sub=n_sub,
                                N_t=lv_nt,
                                N_y=lv_ny,
                                BLOCK_N=block_n_sde,
                                USE_Z_CORR=use_zc_bwd,
                                USE_RT_FLAGS=use_cg_bucket,
                                num_warps=nw_sde,
                            )
                            if want_lv_aad_sde or want_chol_launch or want_q_launch:
                                bwd_bn_cap = int(os.environ.get("DEEPVOL_BWD_BN", "128"))
                                bwd_nw = int(os.environ.get("DEEPVOL_BWD_NW", "4"))
                                block_p2_sde = min(bwd_bn_cap, max(32, _next_power_of_two_int(n_rep)))
                                n_tiles_p2_sde = (n_rep + block_p2_sde - 1) // block_p2_sde
                                _fast_triton_launch(
                                    _qmc_bwd_period_scatter_kernel,
                                    (n_tiles_p2_sde, m_eff_k * r_cnt_c * d_eff_k),
                                    z_indep_f32_c,
                                    z_bwd_prop_in,
                                    chol_f32_eff,
                                    lv_f32_eff,
                                    y_min_f32_eff,
                                    inv_dy_f32_eff,
                                    t_mid_eff,
                                    dt_sub_eff,
                                    sqrt_dt_eff,
                                    q_shift_eff,
                                    bar_lx_post_c,
                                    lx_period_c,
                                    bar_lv_f32_c,
                                    bar_zc_c,
                                    bar_q_c,
                                    inv_dt_f32_val,
                                    d_eff_k,
                                    m_eff_k,
                                    r_cnt_c,
                                    cg_flags_ptr=cg_flags_eff,
                                    payoff_f64_ptr=pf64_eff if use_cg_bucket else None,
                                    n_rep=n_rep,
                                    n_sub=n_sub,
                                    N_t=lv_nt,
                                    N_y=lv_ny,
                                    BLOCK_N=block_p2_sde,
                                    N_LV_SHARDS=n_lv_shards_bwd,
                                    COMPUTE_LV_AAD=want_lv_aad_sde,
                                    COMPUTE_CHOL_AAD=want_chol_launch,
                                    COMPUTE_Q_AAD=want_q_launch,
                                    USE_Z_CORR=use_zc_bwd,
                                    USE_RT_FLAGS=use_cg_bucket,
                                    num_warps=bwd_nw,
                                )
                            if want_lv_aad_sde and bar_lv_tot_g is not None:
                                if defer_bwd_shard_sum:
                                    if r_end == eff_r:
                                        if n_lv_shards_bwd > 1:
                                            torch.sum(bar_lv_f32_c, dim=0, dtype=torch.float64, out=bar_lv_tot_g)
                                        else:
                                            bar_lv_tot_g.copy_(bar_lv_f32_c[0])
                                        bar_lv_tot_g.mul_(w_chunk)
                                        bar_lv_acc_f32_c = None
                                elif n_lv_shards_bwd > 1:
                                    if single_chunk and (not is_cont_ki_active):
                                        torch.sum(bar_lv_f32_c, dim=0, dtype=torch.float64, out=bar_lv_tot_g)
                                    else:
                                        bar_lv_tot_g += w_chunk * bar_lv_f32_c.sum(dim=0, dtype=torch.float64)
                                elif single_chunk and (not is_cont_ki_active):
                                    bar_lv_tot_g.copy_(bar_lv_f32_c[0])
                                else:
                                    bar_lv_tot_g += w_chunk * bar_lv_f32_c[0].to(dtype=torch.float64)
                            if want_chol_launch and bar_chol_tot_g is not None:
                                if single_chunk:
                                    torch.sum(
                                        torch.bmm(bar_zc_c, z_indep_f32_c.transpose(1, 2)),
                                        dim=0,
                                        dtype=torch.float64,
                                        out=bar_chol_tot_g,
                                    )
                                else:
                                    bar_chol_tot_g.add_(
                                        torch.bmm(bar_zc_c, z_indep_f32_c.transpose(1, 2)).sum(dim=0, dtype=torch.float64),
                                        alpha=w_chunk,
                                    )
                            del bar_lv_f32_c, bar_zc_c
                        else:
                            bar_lv_f32_c = (
                                torch.zeros((d_assets, lv_nt, lv_ny), dtype=torch.float32, device=qmc_device)
                                if want_lv_aad_sde
                                else torch.empty((1,), dtype=torch.float32, device=qmc_device)
                            )
                            bar_chol_c = (
                                torch.zeros((d_assets, d_assets), dtype=torch.float32, device=qmc_device)
                                if want_chol_aad_sde
                                else torch.empty((1,), dtype=torch.float32, device=qmc_device)
                            )
                            bar_x_c = torch.zeros((d_assets, n_chunk), dtype=torch.float32, device=qmc_device)
                            for p_cur_s in range(m_rem - 1, -1, -1):
                                bar_tot_p = (bar_x_c + bar_lx_post_c[p_cur_s]).view(d_assets, r_cnt_c, n_rep)
                                omega_p = torch.exp(lx_period_c[p_cur_s].view(d_assets, r_cnt_c, n_rep)) / float(n_rep)
                                bar_x_c = (bar_tot_p - omega_p * bar_tot_p.sum(dim=2, keepdim=True)).reshape(d_assets, n_chunk).contiguous()
                                _fast_triton_launch(
                                    _qmc_bwd_sde_aad_kernel,
                                    (n_tiles_sde, r_cnt_c * d_assets),
                                    z_indep_f32_c,
                                    dummy_shifts_g,
                                    chol_f32_g_cached,
                                    lv_f32_g_cached,
                                    y_min_f32_g,
                                    inv_dy_f32_g,
                                    t_mid_ptr_g,
                                    dt_sub_ptr_g,
                                    sqrt_dt_ptr_g,
                                    q_shift_mid_ptr_g,
                                    bar_x_c,
                                    bar_lx_post_c,
                                    lx_period_c,
                                    bar_lv_f32_c,
                                    bar_chol_c,
                                    bar_q_c,
                                    inv_dt_f32_val,
                                    d_assets,
                                    m_rem,
                                    n_rep=n_rep,
                                    n_sub=n_sub,
                                    N_t=lv_nt,
                                    N_y=lv_ny,
                                    BLOCK_N=block_n_sde,
                                    period_idx=p_cur_s,
                                    FUSE_M=False,
                                    COMPUTE_LV_AAD=want_lv_aad_sde,
                                    COMPUTE_CHOL_AAD=want_chol_aad_sde,
                                    COMPUTE_Q_AAD=want_q_aad_sde,
                                    ATOM_SHARDS=1,
                                    num_warps=nw_sde,
                                )
                            del bar_x_c
                            if want_lv_aad_sde and bar_lv_tot_g is not None:
                                bar_lv_tot_g += w_chunk * bar_lv_f32_c.to(dtype=torch.float64)
                            if want_chol_aad_sde and bar_chol_tot_g is not None:
                                bar_chol_tot_g += w_chunk * bar_chol_c.to(dtype=torch.float64)
                            del bar_lv_f32_c, bar_chol_c
                        if want_q_launch and bar_q_tot_g is not None:
                            if defer_bwd_shard_sum:
                                if r_end == eff_r:
                                    bar_q_tot_g.copy_(bar_q_c)
                                    bar_q_tot_g.mul_(w_chunk)
                                    bar_q_acc_c = None
                            elif single_chunk:
                                bar_q_tot_g.copy_(bar_q_c)
                            else:
                                bar_q_tot_g += w_chunk * bar_q_c.to(dtype=torch.float64)
                        del bar_q_c
                    if bar_lx_post_c is not None:
                        del bar_lx_post_c
                    if lx_period_c is not None:
                        del lx_period_c

                    # Compute Antithetic Cholesky Vibrato Gamma + frozen terminal state in chunk memory.
                    frozen_term_c: Optional[torch.Tensor] = None
                    if use_vibrato_gamma:
                        assert vib_lx_pre_c is not None and vib_j_pre_c is not None and vib_sig_pre_c is not None
                        assert l_inv_t_eff is not None and c_inv_eff is not None and cdl_m_eff is not None and cdl2_last_eff is not None
                        assert h_vib_tot_g is not None
                        block_nv = min(128, max(32, _next_power_of_two_int(n_chunk)))
                        grid_nv = (n_chunk + block_nv - 1) // block_nv
                        use_f32_vib_pre = bool(os.environ.get("DEEPVOL_VIB_F32", "1") == "1")
                        d_ln_surv_tmp_c = _alloc_cg_buf(
                            cg_capturing_now,
                            qmc_device,
                            "d_ln_surv",
                            ((2 * d_eff_k if (use_cg_bucket or is_memory) else d_eff_k), n_chunk),
                            torch.float32 if use_f32_vib_pre else torch.float64,
                            zero=False,
                        )
                        h_vib_out_g = _alloc_cg_buf(
                            cg_capturing_now,
                            qmc_device,
                            "h_vib_out",
                            (grid_nv, d_eff_k, d_eff_k),
                            torch.float64,
                            zero=use_cg_bucket,
                        )
                        if use_cg_bucket or m_rem > 1:
                            frozen_term_c = _alloc_cg_buf(
                                cg_capturing_now,
                                qmc_device,
                                "frozen_term",
                                (4, n_chunk),
                                torch.float64,
                                zero=False,
                            )
                        _fast_triton_launch(
                            _vibrato_gamma_f64_kernel,
                            (grid_nv,),
                            perf_base_c,
                            vib_lx_pre_c,
                            vib_j_pre_c,
                            vib_sig_pre_c,
                            z_indep_f32_c,
                            chol_f64_eff,
                            l_inv_t_eff,
                            c_inv_eff,
                            fwd_f64_eff if use_cg_bucket else fwd_perf_g[:, m_rem - 1].contiguous(),
                            cdl_m_eff,
                            cdl2_last_eff,
                            ko_eff,
                            ic_eff,
                            cp_eff,
                            ig_eff,
                            cr_eff,
                            eps_eff,
                            inv_perm_eff,
                            pf64_eff,
                            d_ln_surv_tmp_c,
                            frozen_term_c,
                            h_vib_out_g,
                            df_last_val,
                            h_vib_val,
                            vib_step_start,
                            vib_n_steps,
                            d_eff_k,
                            m_eff_k,
                            n_chunk,
                            cg_flags_ptr=cg_flags_eff,
                            CALL_CODE=0 if use_cg_bucket else call_code,
                            CPN_CODE=0 if use_cg_bucket else cpn_code,
                            EFF_OPT_CODE=0 if use_cg_bucket else eff_opt_code_s2,
                            HAS_KI=False if use_cg_bucket else has_ki,
                            KI_ACTIVE=False if use_cg_bucket else bool(has_ki and b_ki > 0.0 and float(state.i_ki_past) < 0.5),
                            HAS_ONE_STAR=False if use_cg_bucket else bool(k_os > 0.0),
                            IS_MEM=False if use_cg_bucket else is_memory,
                            IS_LONG=False if use_cg_bucket else is_long_flag,
                            NO_NOTIONAL=False if use_cg_bucket else no_notional,
                            FORCE_CPN_AC=False if use_cg_bucket else force_cpn_ac,
                            CPN_DIR_DOWN=False if use_cg_bucket else dir_down,
                            CPN_TYPE_OUT=False if use_cg_bucket else type_out,
                            HAS_LOWER_STRIKE=True if use_cg_bucket else has_lower_strike_flag,
                            HAS_PART_CALL=False if use_cg_bucket else has_part_call_flag,
                            BLOCK_N=block_nv,
                            USE_F32_PRE=use_f32_vib_pre,
                            USE_RT_FLAGS=use_cg_bucket,
                            D_PAD=d_eff_k if use_cg_bucket else 0,
                            N_SUB_PAD=n_sub,
                            num_warps=4,
                        )
                        if single_chunk:
                            torch.sum(h_vib_out_g, dim=0, out=h_vib_tot_g)
                        else:
                            h_vib_tot_g.add_(h_vib_out_g.sum(dim=0), alpha=w_chunk)
                        del d_ln_surv_tmp_c, h_vib_out_g, vib_lx_pre_c, vib_j_pre_c, vib_sig_pre_c

                    if use_mf_cv and s1_gbm_crn_g is not None and chol_scaled_f32_g is not None:
                        if use_cg_bucket and bpk is not None and z_corr_f32_mdn_c is not None:
                            zc_crn_mdn = _alloc_cg_buf(
                                cg_capturing_now,
                                qmc_device,
                                "lx_period",
                                (m_eff_k, d_eff_k, n_chunk),
                                torch.float32,
                            )
                            torch.sum(z_corr_f32_mdn_c.view(m_eff_k, n_sub, d_eff_k, n_chunk), dim=1, out=zc_crn_mdn)
                            _run_mf_gbm_batch(
                                zc_crn_mdn,
                                s1_gbm_crn_g[r_start:r_end],
                                cf_surv_gbm_crn_g[r_start:r_end] if cf_surv_gbm_crn_g is not None else None,
                                layout_mdn=True,
                                r_b_ov=r_cnt_c,
                                n_rep_ov=n_rep,
                                z_scale_val=inv_sqrt_nsub_val,
                                pre_zeroed=True,
                                d_pad_ov=d_eff_k,
                                m_pad_ov=m_eff_k,
                                d_var_ov=bpk["d_var_f32"],
                                fwd_ov=bpk["fwd_pad_f32"],
                                df_ov=bpk["df_pay_f32"],
                                wf_ov=bpk["w_fund_f32"],
                                ko_ov=bpk["ko_f32"],
                                ic_ov=bpk["is_call_f32"],
                                cp_ov=bpk["cpn_f32"],
                                ig_ov=bpk["is_guar_f32"],
                                cr_ov=bpk["cpn_rates_f32"],
                                eps_ov=bpk["eps_f32"],
                                payoff_ov=bpk["payoff_f32_15"],
                                cg_flags_ov=cg_flags_eff,
                                use_rt_flags=True,
                                inv_perm_ov=bpk["inv_perm"],
                                use_cg_arena=cg_capturing_now,
                            )
                            del zc_crn_mdn
                        elif z_corr_coarse_f32 is not None:
                            zc_crn_mdn = (
                                z_corr_coarse_f32
                                if single_chunk
                                else z_corr_coarse_f32[:, :, r_start * n_rep : r_end * n_rep].contiguous()
                            )
                            _run_mf_gbm_batch(
                                zc_crn_mdn,
                                s1_gbm_crn_g[r_start:r_end],
                                cf_surv_gbm_crn_g[r_start:r_end] if cf_surv_gbm_crn_g is not None else None,
                                layout_mdn=True,
                                r_b_ov=r_cnt_c,
                                n_rep_ov=n_rep,
                                z_scale_val=inv_sqrt_nsub_val,
                                pre_zeroed=True,
                            )
                        elif z_corr_f32_mdn_c is not None:
                            zc_crn_mdn = z_corr_f32_mdn_c.view(m_rem, n_sub, d_assets, n_chunk).sum(dim=1)
                            _run_mf_gbm_batch(
                                zc_crn_mdn,
                                s1_gbm_crn_g[r_start:r_end],
                                cf_surv_gbm_crn_g[r_start:r_end] if cf_surv_gbm_crn_g is not None else None,
                                layout_mdn=True,
                                r_b_ov=r_cnt_c,
                                n_rep_ov=n_rep,
                                z_scale_val=inv_sqrt_nsub_val,
                                pre_zeroed=True,
                            )
                            del zc_crn_mdn
                        else:
                            zc_crn_mean = z_indep_f32_c.view(m_rem, n_sub, d_assets, n_chunk).mean(dim=1)
                            zc_crn_rmdn = zc_crn_mean.view(m_rem, d_assets, r_cnt_c, n_rep).permute(2, 0, 1, 3)
                            z_coarse_corr_crn = torch.matmul(chol_scaled_f32_g, zc_crn_rmdn).contiguous()
                            _run_mf_gbm_batch(
                                z_coarse_corr_crn,
                                s1_gbm_crn_g[r_start:r_end],
                                cf_surv_gbm_crn_g[r_start:r_end] if cf_surv_gbm_crn_g is not None else None,
                                pre_zeroed=True,
                            )
                            del zc_crn_mean, zc_crn_rmdn, z_coarse_corr_crn

                    # Convert chunk Z_indep -> Z_corr if spot bumps or Theta need 1D SDE stepping, then drop Z_indep.
                    if (need_spot_bumps or th_chunk_in_step1) and z_corr_c is None and z_corr_f32_mdn_c is None:
                        zc_dtype = torch.float32 if use_f32_step1 else torch.float64
                        chol_step = chol_f32_g_cached if use_f32_step1 else chol_f64_g
                        z_corr_c = torch.matmul(
                            chol_step,
                            z_indep_f32_c.to(dtype=zc_dtype).permute(1, 0, 2).reshape(d_assets, -1),
                        ).view(d_assets, m_rem, n_sub, n_chunk).contiguous()
                    del z_indep_f32_c

                    perf_sp_pm_c: Optional[torch.Tensor] = None
                    alt_sp_cg_c: Optional[torch.Tensor] = None
                    if need_spot_bumps:
                        assert sp_pm_meta_eff is not None and lv_sp_pm_eff is not None and fwd_sp_pm_eff is not None
                        if z_corr_f32_mdn_c is not None:
                            perf_sp_pm_c = _gpu_step_1d_f64(
                                lv_sp_pm_eff,
                                z_corr_f32_mdn_c,
                                tau_f64_eff,
                                rn_f64_eff,
                                rt_f64_eff,
                                2,
                                m_periods=m_eff_k,
                                meta_lv_t=sp_pm_meta_eff[0],
                                meta_r_t=sp_pm_meta_eff[1],
                                meta_z_t=sp_pm_meta_eff[2],
                                y_min_vec_t=y_min_f64_eff,
                                inv_dy_vec_t=inv_dy_f64_eff,
                                inv_dt_vec_t=inv_dt_f64_eff,
                                z_layout_mdn=True,
                                fwd_perf_scale_t=fwd_sp_pm_eff,
                                out_mdn=False,
                                out_f32=use_f32_perf_direct,
                                use_cg_arena=cg_capturing_now,
                                cg_arena_slot="perf_aux",
                            )
                        else:
                            assert z_corr_c is not None
                            nx_sp_c = _gpu_step_1d_f64(
                                lv_sp_pm_g,
                                z_corr_c,
                                tau_step_g[:m_rem],
                                rn_step_g[: d_assets * m_rem].view(d_assets, m_rem),
                                rt_step_g[: d_assets * m_rem].view(d_assets, m_rem),
                                2,
                                meta_lv_t=sp_pm_meta_g[0],
                                meta_r_t=sp_pm_meta_g[1],
                                meta_z_t=sp_pm_meta_g[2],
                            )
                            perf_sp_pm_c = (nx_sp_c * fwd_sp_pm_g.unsqueeze(-1)).contiguous()
                            del nx_sp_c
                        if (not use_cg_bucket) and (need_sp_cg_alt_g or (pre_pack is not None and not want_gamma)):
                            alt_sp_cg_c = torch.stack(
                                [
                                    perf_sp_pm_c[:d_assets].permute(1, 0, 2),
                                    perf_sp_pm_c[d_assets:].permute(1, 0, 2),
                                ],
                                dim=0,
                            ).contiguous()

                    if stat_pack_eff is not None and stat_acc_g is not None:
                        stat_mc_c = _run_loo_l2o_eval(
                            perf_base_c,
                            perf_sp_pm_c,
                            perf_sp_pm_c,
                            np.empty(0, dtype=np.int32),
                            np.empty(0, dtype=np.int32),
                            None,
                            None,
                            None,
                            alt_perf=alt_sp_cg_c if need_sp_cg_alt_g else None,
                            return_gpu=True,
                            pure_mc=True,
                            static_gpu_pack=stat_pack_eff,
                            m_periods=m_eff_k,
                            ko_ov=ko_eff,
                            is_call_ov=ic_eff,
                            cpn_ov=cp_eff,
                            is_guar_ov=ig_eff,
                            cpn_rates_ov=cr_eff,
                            payoff_f64_ov=pf64_eff,
                            inv_perm_ov=inv_perm_eff,
                            eps_vec_g=eps_eff,
                            i_ki_ext_g=i_ki_ext_c,
                            out_pv_g=stat_acc_g if single_chunk else None,
                            d_pad_rt=d_eff_k,
                            cg_flags_g=cg_flags_eff,
                            use_rt_flags=use_cg_bucket,
                            cg_bumps_idx=16 if use_cg_bucket else -1,
                            use_cg_arena=cg_capturing_now,
                        )
                        if not single_chunk:
                            stat_acc_g.add_(stat_mc_c, alpha=w_chunk)

                    if pre_pack_eff is not None and pv_preterm_g is not None and frozen_term_c is not None:
                        pre_mc_c = _run_loo_l2o_eval(
                            perf_base_c,
                            perf_sp_pm_c if want_gamma else None,
                            perf_sp_pm_c if want_gamma else None,
                            np.empty(0, dtype=np.int32),
                            np.empty(0, dtype=np.int32),
                            None,
                            None,
                            None,
                            alt_perf=alt_sp_cg_c if (not want_gamma) else None,
                            return_gpu=True,
                            pure_mc=True,
                            static_gpu_pack=pre_pack_eff,
                            m_periods=m_eff_k,
                            ko_ov=ko_eff,
                            is_call_ov=ic_eff,
                            cpn_ov=cp_eff,
                            is_guar_ov=ig_eff,
                            cpn_rates_ov=cr_eff,
                            payoff_f64_ov=pf64_eff,
                            inv_perm_ov=inv_perm_eff,
                            eps_vec_g=eps_eff,
                            frozen_term_g=frozen_term_c,
                            use_c2_preterm=True,
                            use_frozen_term=True,
                            out_pv_g=pv_preterm_g if single_chunk else None,
                            d_pad_rt=d_eff_k,
                            cg_flags_g=cg_flags_eff,
                            use_rt_flags=use_cg_bucket,
                            cg_bumps_idx=17 if use_cg_bucket else -1,
                            use_cg_arena=cg_capturing_now,
                        )
                        if not single_chunk:
                            pv_preterm_g.add_(pre_mc_c, alpha=w_chunk)

                    if dyn_entries and gpu_eval_pv_g is not None:
                        assert norm_x_c is not None
                        for s_r_d, e_r_d, sp_d, df_d_g, wf_d_g in dyn_entries:
                            nb_d = e_r_d - s_r_d
                            if sp_d["fwd_alt"] is not None:
                                fwd_a_g = torch.as_tensor(sp_d["fwd_alt"], dtype=torch.float64, device=qmc_device)
                                alt_d_c = (norm_x_c.unsqueeze(0) * fwd_a_g.unsqueeze(-1)).permute(0, 2, 1, 3).contiguous()
                                dyn_mc_c = _run_loo_l2o_eval(
                                    None,
                                    alt_d_c,
                                    None,
                                    np.full(nb_d, -2, dtype=np.int32),
                                    sp_d["ab"],
                                    df_d_g,
                                    wf_d_g,
                                    None,
                                    return_gpu=True,
                                    pure_mc=True,
                                    i_ki_ext_g=i_ki_ext_c,
                                )
                                del alt_d_c, fwd_a_g
                            else:
                                aa_d_g = torch.as_tensor(sp_d["aa"], dtype=torch.long, device=qmc_device)
                                fwd_ba_g = torch.as_tensor(sp_d["fwd_ba"], dtype=torch.float64, device=qmc_device)
                                ba_d_c = (norm_x_c.index_select(0, aa_d_g) * fwd_ba_g[:, :, None]).contiguous()
                                dyn_mc_c = _run_loo_l2o_eval(
                                    perf_base_c,
                                    ba_d_c,
                                    None,
                                    sp_d["aa"],
                                    sp_d["ab"],
                                    df_d_g,
                                    wf_d_g,
                                    None,
                                    return_gpu=True,
                                    pure_mc=True,
                                    i_ki_ext_g=i_ki_ext_c,
                                )
                                del aa_d_g, fwd_ba_g, ba_d_c
                            gpu_eval_pv_g[s_r_d:e_r_d] += w_chunk * dyn_mc_c
                    if norm_x_c is not None:
                        del norm_x_c
                    if frozen_term_c is not None:
                        del frozen_term_c
                    if alt_sp_cg_c is not None:
                        del alt_sp_cg_c
                    if perf_sp_pm_c is not None:
                        del perf_sp_pm_c
                    if i_ki_ext_c is not None:
                        del i_ki_ext_c
                    del perf_base_c

                    if use_cg_bucket and bpk is not None and pv_th_fused_g is not None and z_corr_f32_mdn_c is not None:
                        perf_th_c = _gpu_step_1d_f64(
                            bpk["lv_th_f32"],
                            z_corr_f32_mdn_c,
                            bpk["tau_th_f64"],
                            bpk["rn_th_f64"],
                            bpk["rt_th_f64"],
                            2,
                            m_periods=m_eff_k,
                            meta_lv_t=bpk["id_d"],
                            meta_r_t=bpk["id_d"],
                            meta_z_t=bpk["id_d"],
                            y_min_vec_t=y_min_f64_eff,
                            inv_dy_vec_t=inv_dy_f64_eff,
                            inv_dt_vec_t=bpk["inv_dt_th_f64"],
                            z_layout_mdn=True,
                            fwd_perf_scale_t=bpk["fwd_th_f64"],
                            out_mdn=True,
                            out_f32=use_f32_perf_direct,
                            use_cg_arena=cg_capturing_now,
                            cg_arena_slot="perf_aux",
                            cg_th_flags_g=bpk["th_cg_flags"],
                            use_th_rt_flags=True,
                        )
                        del z_corr_f32_mdn_c
                        z_corr_f32_mdn_c = None
                        th_mc_c = _run_loo_l2o_eval(
                            perf_th_c,
                            None,
                            None,
                            np.empty(0, dtype=np.int32),
                            np.empty(0, dtype=np.int32),
                            None,
                            None,
                            None,
                            return_gpu=True,
                            pure_mc=True,
                            static_gpu_pack=bpk["th_pack"],
                            m_periods=m_eff_k,
                            ko_ov=bpk["ko_th_f64"],
                            is_call_ov=bpk["ic_th_f64"],
                            cpn_ov=bpk["cp_th_f64"],
                            is_guar_ov=bpk["ig_th_f64"],
                            cpn_rates_ov=bpk["cr_th_f64"],
                            payoff_f64_ov=bpk["th_payoff_f64_18"],
                            inv_perm_ov=bpk["inv_perm"],
                            eps_vec_g=bpk["eps_th_f64"],
                            out_pv_g=pv_th_fused_g if single_chunk else None,
                            d_pad_rt=d_eff_k,
                            cg_flags_g=bpk["th_cg_flags"],
                            use_rt_flags=True,
                            cg_bumps_idx=16,
                            use_cg_arena=cg_capturing_now,
                        )
                        if not single_chunk:
                            pv_th_fused_g.add_(th_mc_c, alpha=w_chunk)
                        del perf_th_c
                    elif th_chunk_in_step1 and th_pack is not None and pv_th_fused_g is not None and th_prep is not None:
                        assert lv_th_g is not None and fwd_th_g is not None
                        if z_corr_f32_mdn_c is not None:
                            z_th_mdn = z_corr_f32_mdn_c if k_cross_g == 0 else z_corr_f32_mdn_c[k_cross_g * n_sub :]
                            del z_corr_f32_mdn_c
                            z_corr_f32_mdn_c = None
                            perf_th_c = _gpu_step_1d_f64(
                                lv_th_g,
                                z_th_mdn,
                                tau_step_g[m_rem:],
                                rn_step_g[d_assets * m_rem :].view(d_assets, m_rem_th_g),
                                rt_step_g[d_assets * m_rem :].view(d_assets, m_rem_th_g),
                                2,
                                m_periods=m_rem_th_g,
                                meta_lv_t=id_d_g,
                                meta_r_t=id_d_g,
                                meta_z_t=id_d_g,
                                inv_dt_vec_t=inv_dt_base_g[1:2],
                                z_layout_mdn=True,
                                fwd_perf_scale_t=fwd_th_g,
                                out_mdn=True,
                                out_f32=use_f32_perf_direct,
                                use_cg_arena=cg_capturing_now,
                                cg_arena_slot="perf_aux",
                            )
                            del z_th_mdn
                        else:
                            assert z_corr_c is not None
                            z_th_c = z_corr_c if k_cross_g == 0 else z_corr_c[:, k_cross_g:, :, :].contiguous()
                            del z_corr_c
                            z_corr_c = None
                            nx_th_c = _gpu_step_1d_f64(
                                lv_th_g,
                                z_th_c,
                                tau_step_g[m_rem:],
                                rn_step_g[d_assets * m_rem :].view(d_assets, m_rem_th_g),
                                rt_step_g[d_assets * m_rem :].view(d_assets, m_rem_th_g),
                                2,
                                m_periods=m_rem_th_g,
                                meta_lv_t=id_d_g,
                                meta_r_t=id_d_g,
                                meta_z_t=id_d_g,
                                inv_dt_vec_t=inv_dt_base_g[1:2],
                            )
                            del z_th_c
                            perf_th_c = (nx_th_c * fwd_th_g.unsqueeze(-1)).permute(1, 0, 2).contiguous()
                            del nx_th_c
                        i_ki_th_c: Optional[torch.Tensor] = None
                        if is_cont_ki_active:
                            assert df_th_g is not None and ko_th_g is not None and ic_th_g is not None
                            assert cp_th_g is not None and ig_th_g is not None and cr_th_g is not None
                            iki_th_box: List[torch.Tensor] = []
                            _compute_payoff_aad_state_adjoints_torch(
                                perf_th_c,
                                ko_barriers=ko_th_g,
                                is_callable=ic_th_g,
                                cpn_barriers=cp_th_g,
                                is_guar=ig_th_g,
                                cpn_rates=cr_th_g,
                                df_pay=df_th_g[0],
                                w_fund=wf_th_g[0] if wf_th_g is not None else None,
                                inv_perm=inv_perm,
                                call_code=call_code,
                                cpn_code=cpn_code,
                                opt_code=opt_code,
                                ko_spread=ko_spread,
                                cpn_spread=cpn_spread,
                                ki_spread=ki_spread,
                                cpn_dir_down=dir_down,
                                cpn_type_out=type_out,
                                is_memory=is_memory,
                                force_cpn_ac=force_cpn_ac,
                                no_notional=no_notional,
                                is_long_basket=is_long_flag,
                                has_ki=has_ki,
                                ki_barrier=b_ki,
                                i_ki_past=float(th_prep["eff_iki"]),
                                ki_monitoring_type=ki_monitoring_type,
                                tau_obs=th_prep["tau_obs"],
                                vol_atm_obs=vol_atm_obs[:, k_cross_g:] if vol_atm_obs is not None else None,
                                strike=k_put,
                                lower_strike=state.lower_strike,
                                option_gearing=float(state.option_gearing),
                                ac_cpn=float(state.autocall_coupon),
                                c_mem_0=float(th_prep["eff_cmem"]),
                                part_call_strike=state.participation_call_strike,
                                part_call_rate=part_call_rate,
                                one_star_strike=(k_os if k_os > 0.0 else state.one_star_strike),
                                one_star_coupon=os_cpn,
                                one_star_eps=one_star_eps,
                                barrier_eps_min=barrier_eps_min,
                                barrier_eps_max=barrier_eps_max,
                                lv_tables_t=lv_th_g,
                                corr_t=gpu_meta["corr"],
                                fwd_perf_t=fwd_th_g,
                                y_min=lv_y_min_vec,
                                y_max=lv_y_max_vec,
                                n_sub=n_sub,
                                n_rep=n_rep,
                                emit_stage1=False,
                                compute_grads=False,
                                out_i_ki_ext=iki_th_box,
                            )
                            i_ki_th_c = iki_th_box[0] if iki_th_box else None
                        th_mc_c = _run_loo_l2o_eval(
                            perf_th_c,
                            None,
                            None,
                            np.empty(0, dtype=np.int32),
                            np.empty(0, dtype=np.int32),
                            None,
                            None,
                            None,
                            return_gpu=True,
                            pure_mc=True,
                            static_gpu_pack=th_pack,
                            m_periods=m_rem_th_g,
                            ko_ov=ko_th_g,
                            is_call_ov=ic_th_g,
                            cpn_ov=cp_th_g,
                            is_guar_ov=ig_th_g,
                            cpn_rates_ov=cr_th_g,
                            payoff_f64_ov=payoff_th_g,
                            eps_vec_g=eps_th_g,
                            i_ki_ext_g=i_ki_th_c,
                            out_pv_g=pv_th_fused_g if single_chunk else None,
                            use_cg_arena=cg_capturing_now,
                        )
                        if not single_chunk:
                            pv_th_fused_g.add_(th_mc_c, alpha=w_chunk)
                        del perf_th_c
                        if i_ki_th_c is not None:
                            del i_ki_th_c
                    elif z_corr_c is not None:
                        del z_corr_c
                    if z_corr_f32_mdn_c is not None:
                        del z_corr_f32_mdn_c

                if use_mf_cv and mf_cheap_hit is None and s1_gbm_cheap_g is not None and sv_sobol_gpu is not None:
                    assert u_rot_gpu is not None and n_rep_gen_mf > 0
                    mf_shift_key = ("sobol_mf_shifts_v3", eff_r_mf, n_rep_gen_mf, m_rem, d_assets, seed + 99991, dev_norm_str)
                    shifts_mf_gpu = _CRN_SOBOL_CACHE.get(mf_shift_key)
                    if shifts_mf_gpu is None:
                        rng_mf = np.random.default_rng(seed + 99991)
                        shifts_mf_np = rng_mf.integers(0, 1 << 30, size=(eff_r_mf, m_rem * 2 * d_assets), dtype=np.int32)
                        shifts_mf_gpu = torch.from_numpy(shifts_mf_np).to(device=qmc_device, dtype=torch.int32)
                        _CRN_SOBOL_CACHE.put(mf_shift_key, shifts_mf_gpu)
                    cv_eig_f32_g = chol_f32_g_cached.contiguous()
                    mf_per_rep_bytes = max(m_rem * d_assets * 8 + d_assets * 4, 1) * n_rep_gen_mf
                    r_mf_chunk = min(eff_r_mf, max(1, (12 * 1024 * 1024) // mf_per_rep_bytes))
                    m_pow_mf = int(round(math.log2(n_rep_gen_mf)))
                    block_n_mf = min(256, n_rep_gen_mf)
                    n_tiles_mf = (n_rep_gen_mf + block_n_mf - 1) // block_n_mf
                    for r_mf_s in range(0, eff_r_mf, r_mf_chunk):
                        r_mf_e = min(eff_r_mf, r_mf_s + r_mf_chunk)
                        r_mf_c = r_mf_e - r_mf_s
                        n_mf_tot = r_mf_c * n_rep_gen_mf
                        z_raw_mf = torch.empty((m_rem, d_assets, n_mf_tot), dtype=torch.float32, device=qmc_device)
                        _fast_triton_launch(
                            _sobol_bridge_normals_kernel,
                            (n_tiles_mf, m_rem * r_mf_c),
                            sv_sobol_gpu,
                            shifts_mf_gpu[r_mf_s:r_mf_e],
                            z_raw_mf,
                            m_rem,
                            d_assets,
                            n_rep=n_rep_gen_mf,
                            n_sub=1,
                            m_pow=m_pow_mf,
                            BLOCK_N=block_n_mf,
                            num_warps=4,
                        )
                        tmp_mf = torch.matmul(cv_eig_f32_g, z_raw_mf).view(m_rem, d_assets * n_mf_tot)
                        del z_raw_mf
                        z_corr_mf = torch.matmul(u_rot_gpu, tmp_mf).view(m_rem, d_assets, n_mf_tot)
                        del tmp_mf
                        sb_s = r_mf_s * ema_group_mf
                        sb_e = r_mf_e * ema_group_mf
                        _run_mf_gbm_batch(
                            z_corr_mf,
                            s1_gbm_cheap_g[sb_s:sb_e],
                            cf_surv_gbm_cheap_g[sb_s:sb_e] if cf_surv_gbm_cheap_g is not None else None,
                            antithetic=True,
                            layout_mdn=True,
                            r_b_ov=r_mf_c * ema_group_mf,
                            n_rep_ov=n_rep_mf,
                            z_scale_val=1.0,
                            pre_zeroed=True,
                        )
                        del z_corr_mf
                    del cv_eig_f32_g

                # Finalize rare dynamic rows on GPU, then perform a single direct D2H transfer of out_flat_g.
                if cg_fast_d2h_np is None and (not stat_direct_alias) and gpu_eval_pv_g is not None:
                    if stat_acc_g is not None and stat_pos_g is not None and stat_itr_g is not None:
                        gpu_eval_pv_g[stat_pos_g] = stat_acc_g + stat_itr_g
                    for s_r_d, e_r_d, sp_d, _, _ in dyn_entries:
                        gpu_eval_pv_g[s_r_d:e_r_d] += torch.as_tensor(sp_d["itr"], dtype=torch.float64, device=qmc_device)

                if cg_capturing_now and cg_obj is not None and out_flat_g is not None and cg_pool_lock is not None:
                    cg_capturing_now = False
                    cg_obj.capture_end()
                    if cg_stream_entered and cg_stream_ctx is not None:
                        cg_stream_entered = False
                        cg_stream_ctx.__exit__(None, None, None)
                    GLOBAL_GPU_CACHE_MANAGER.account_pool_capture_bytes(qmc_device, cg_before_stats_qmc)
                    # Detach _CRN_DEAL_PREP_CACHE entry so subsequent deals copying into cg_hit["in_bufs"] cannot mutate cached deal tensors.
                    if use_cg_bucket:
                        if warm_bundle_key is not None:
                            warm_hit_c = _CRN_DEAL_PREP_CACHE.get(warm_bundle_key)
                            if warm_hit_c is not None:
                                warm_hit_c[1].cg_bucket_pack = None
                                warm_hit_c[1].cg_meta_cache.clear()
                    elif warm_bundle_key is not None:
                        _CRN_DEAL_PREP_CACHE.remove(warm_bundle_key)
                    if _warm_gpu is not None:
                        _warm_gpu.cg_bucket_pack = None
                        _warm_gpu.cg_meta_cache.clear()
                    _CRN_QMC_CG_CACHE.put(
                        cg_shape_key,
                        {
                            "cg": cg_obj,
                            "out_flat_g": out_flat_g,
                            "in_bufs": cg_src_tensors,
                            "last_ptrs": tuple(s.data_ptr() for s in cg_src_tensors),
                            "keepalive": (
                                out_flat_g,
                                stat_acc_g,
                                stat_pos_g,
                                sv_sobol_gpu,
                                shifts_sobol_gpu,
                                u_rot_gpu,
                                v_rot_gpu,
                                id_d_g,
                                dummy_f32_1,
                                *(sp_pm_meta_g or ()),
                                *cg_src_tensors,
                            ),
                        },
                    )
                    cg_obj.replay()
                    cg_fast_d2h_np = np.asarray(out_flat_g.clone().detach().cpu().numpy(), dtype=np.float64)
            finally:
                if cg_capturing_now and cg_obj is not None:
                    try:
                        cg_capturing_now = False
                        cg_obj.capture_end()
                    except Exception as end_exc:
                        logger.debug("QMC CUDAGraph capture_end cleanup failed: %s", end_exc)
                if cg_stream_entered and cg_stream_ctx is not None:
                    try:
                        cg_stream_entered = False
                        cg_stream_ctx.__exit__(None, None, None)
                    except Exception as ctx_exc:
                        logger.debug("QMC CUDAGraph stream exit cleanup failed: %s", ctx_exc)
                if cg_pool_lock is not None:
                    cg_pool_lock.release()
                    cg_pool_lock = None

            s1_gbm_crn_np: Optional[npt.NDArray[np.float64]] = None
            s1_gbm_cheap_np: Optional[npt.NDArray[np.float64]] = None
            if cg_fast_d2h_np is not None or out_flat_g is not None:
                flat_d2h_np = (
                    cg_fast_d2h_np
                    if cg_fast_d2h_np is not None
                    else np.asarray(out_flat_g.clone().detach().cpu().numpy(), dtype=np.float64)  # type: ignore[union-attr]
                )

                def _take_d2h(tag: str) -> Optional[npt.NDArray[np.float64]]:
                    sp_t = pack_spans.get(tag)
                    if sp_t is None:
                        return None
                    return flat_d2h_np[sp_t[0] : sp_t[1]].reshape(sp_t[2])

                def _compact_s1_arr(arr: npt.NDArray[np.float64], tail_k: int) -> npt.NDArray[np.float64]:
                    if (not use_cg_bucket) or m_eff_k == m_rem:
                        return arr
                    return np.ascontiguousarray(
                        np.concatenate(
                            [
                                arr[:, :m_rem],
                                arr[:, m_eff_k : m_eff_k + m_rem],
                                arr[:, 2 * m_eff_k : 2 * m_eff_k + m_rem],
                                arr[:, 3 * m_eff_k : 3 * m_eff_k + tail_k],
                            ],
                            axis=1,
                        )
                    )

                s1_arr_out = _take_d2h("stage1")
                if s1_arr_out is not None:
                    stage1_rep_np = _compact_s1_arr(s1_arr_out, 6)
                ev_arr_out = _take_d2h("eval_pv")
                if ev_arr_out is not None:
                    if use_cg_bucket:
                        ev_arr_out = ev_arr_out[:gpu_eval_rows_total]
                    if stat_direct_alias and stat_itr_np is not None:
                        ev_arr_out = ev_arr_out + stat_itr_np
                    pq_chunks.append(ev_arr_out)
                pt_arr_out = _take_d2h("preterm")
                if pt_arr_out is not None and use_vibrato_gamma and m_rem > 1 and n_pre_rows > 0:
                    pv_preterm_np = pt_arr_out[:n_pre_rows] if use_cg_bucket else pt_arr_out
                th_arr_out = _take_d2h("th_fused")
                if th_arr_out is not None:
                    if th_prep is not None:
                        th_arr_out = th_arr_out + (float(th_prep["in_tr"]) + float(th_prep["crossed_pv"]))
                    pv_th_fused_np = th_arr_out
                hv_arr_out = _take_d2h("h_vib")
                if hv_arr_out is not None:
                    h_vib_np = hv_arr_out[:d_assets, :d_assets] if use_cg_bucket else hv_arr_out
                bc_arr_out = _take_d2h("bar_chol")
                if bc_arr_out is not None and want_chol_aad_sde:
                    bar_chol_aad = bc_arr_out[:d_assets, :d_assets] if use_cg_bucket else bc_arr_out
                bsc_arr_out = _take_d2h("bar_sigma_cont")
                if bsc_arr_out is not None:
                    bar_sigma_cont_np = bsc_arr_out
                blv_arr_out = _take_d2h("bar_lv")
                if blv_arr_out is not None and (want_lv_aad_sde or (need_stage2_aad and is_cont_ki_active)):
                    bar_lv_np = blv_arr_out[:d_assets, :, :] if use_cg_bucket else blv_arr_out
                bq_arr_out = _take_d2h("bar_q")
                if bq_arr_out is not None and want_q_aad_sde:
                    bar_q_shift_np = bq_arr_out[:d_assets, :n_steps_tot_g] if use_cg_bucket else bq_arr_out
                blf_arr_out = _take_d2h("bar_lnf")
                if blf_arr_out is not None:
                    bar_ln_fwd_np = blf_arr_out[:d_assets, :m_rem] if use_cg_bucket else blf_arr_out
                bcf_arr_out = _take_d2h("bar_cf_wf")
                if bcf_arr_out is not None:
                    bcf_sl = bcf_arr_out[:, :m_rem] if use_cg_bucket else bcf_arr_out
                    bar_cf_np = bcf_sl[0]
                    bar_surv_np = bcf_sl[1]
                s1_gbm_crn_raw = _take_d2h("s1_gbm_crn")
                s1_gbm_crn_np = _compact_s1_arr(s1_gbm_crn_raw, 5) if s1_gbm_crn_raw is not None else None
                cf_surv_crn_raw = _take_d2h("cf_surv_gbm_crn")
                cf_surv_crn_out = cf_surv_crn_raw[:, :, :m_rem].copy() if (cf_surv_crn_raw is not None and use_cg_bucket) else cf_surv_crn_raw
                se_mu_mf_cached: Optional[float] = None
                if mf_cheap_hit is not None:
                    s1_gbm_cheap_np, cf_surv_cheap_out, se_mu_mf_cached = mf_cheap_hit
                else:
                    s1_gbm_cheap_np = _take_d2h("s1_gbm_cheap")
                    cf_surv_cheap_out = _take_d2h("cf_surv_gbm_cheap")
                    if ema_group_mf > 1:
                        if s1_gbm_cheap_np is not None:
                            s1_gbm_cheap_np = np.ascontiguousarray(
                                s1_gbm_cheap_np.reshape(eff_r_mf, ema_group_mf, -1).mean(axis=1)
                            )
                        if cf_surv_cheap_out is not None:
                            cf_surv_cheap_out = np.ascontiguousarray(
                                cf_surv_cheap_out.reshape(eff_r_mf, ema_group_mf, 2, m_rem).mean(axis=1)
                            )
                    if s1_gbm_cheap_np is not None:
                        n_mf_rows_c = int(s1_gbm_cheap_np.shape[0])
                        pv_col_mf = 3 * m_rem + 4
                        var_c_mf = (
                            float(np.var(s1_gbm_cheap_np[:, pv_col_mf], ddof=1)) / float(n_mf_rows_c)
                            if n_mf_rows_c > 1
                            else 0.0
                        )
                        se_mu_mf_cached = math.sqrt(max(var_c_mf, 0.0)) if math.isfinite(var_c_mf) else 0.0
                    else:
                        se_mu_mf_cached = 0.0
                    if use_mf_cv and mf_cheap_digest and s1_gbm_cheap_np is not None:
                        _CRN_MF_CHEAP_CACHE.put(
                            mf_cheap_digest,
                            (s1_gbm_cheap_np, cf_surv_cheap_out, se_mu_mf_cached),
                        )
                if cf_surv_crn_out is not None and cf_surv_cheap_out is not None and bar_cf_np is not None:
                    cf_surv_crn_m = np.mean(cf_surv_crn_out, axis=0)
                    cf_surv_cheap_m = np.mean(cf_surv_cheap_out, axis=0)
                    bar_cf_np = bar_cf_np + (cf_surv_cheap_m[0] - cf_surv_crn_m[0])
                    if bar_surv_np is not None:
                        bar_surv_np = bar_surv_np + (cf_surv_cheap_m[1] - cf_surv_crn_m[1])
        else:
            s1_gbm_crn_np = None
            s1_gbm_cheap_np = None
            se_mu_mf_cached = None
            eff_r_mf = 0
    else:
        s1_gbm_crn_np = cpu_mf_s1_crn
        s1_gbm_cheap_np = cpu_mf_s1_cheap
        se_mu_mf_cached = cpu_mf_se_cheap
        eff_r_mf = 0
        if corr_grad_cont_list:
            bar_sigma_cont_np = np.asarray(corr_grad_cont_list[0].detach().cpu().numpy(), dtype=np.float64)

    # Compute Stage-1 cashflows and PV from the single batched D2H transfer.
    if stage1_rep_np is not None:
        s1_mean = np.mean(stage1_rep_np, axis=0)
        c0_s1 = 3 * m_rem
        n_cv_cols = c0_s1 + 5
        if s1_gbm_crn_np is not None and s1_gbm_cheap_np is not None:
            s1_crn_mean = np.mean(s1_gbm_crn_np, axis=0)
            s1_cheap_mean = np.mean(s1_gbm_cheap_np, axis=0)
            beta_cv = 1.0
            if eff_r >= 4:
                x_cv = s1_gbm_crn_np[:, c0_s1 + 4]
                y_cv = stage1_rep_np[:, c0_s1 + 4]
                var_x_cv = float(np.var(x_cv, ddof=1))
                if var_x_cv > 1e-18:
                    xc = x_cv - float(np.mean(x_cv))
                    yc = y_cv - float(np.mean(y_cv))
                    cov_xy_cv = float(np.sum(xc * yc) / float(eff_r - 1))
                    # On eff_r >= 8 (e.g. 53391858 GPU), disable unit CV when variance actually increases (cov < 0.5 * var_x);
                    # on smaller eff_r=4 runs (noisy df=3 covariance), only zero when strictly anti-correlated.
                    if cov_xy_cv < 0.0 or (eff_r >= 8 and cov_xy_cv < 0.5 * var_x_cv):
                        beta_cv = 0.0
            s1_mean[:n_cv_cols] = s1_mean[:n_cv_cols] + beta_cv * (s1_cheap_mean - s1_crn_mean)
            if se_mu_mf_cached is not None and math.isfinite(se_mu_mf_cached):
                var_cheap = (beta_cv * beta_cv) * (se_mu_mf_cached * se_mu_mf_cached)
            else:
                n_mf_rows = int(s1_gbm_cheap_np.shape[0])
                var_cheap = (
                    (beta_cv * beta_cv) * float(np.var(s1_gbm_cheap_np[:, c0_s1 + 4], ddof=1)) / float(n_mf_rows)
                    if n_mf_rows > 1
                    else 0.0
                )
            if eff_r > 1:
                pv_reps_s1 = (stage1_rep_np[:, c0_s1 + 4] - beta_cv * s1_gbm_crn_np[:, c0_s1 + 4]) + beta_cv * float(
                    s1_cheap_mean[c0_s1 + 4]
                ) + float(state.pv_in_transit)
                var_crn = float(np.var(pv_reps_s1, ddof=1)) / float(eff_r) if pv_reps_s1.shape[0] > 1 else 0.0
                tot_se_s1 = math.sqrt(max(var_crn + var_cheap, 0.0))
            else:
                var_s1 = max(float(s1_mean[c0_s1 + 5]) - float(s1_mean[c0_s1 + 4]) ** 2, 0.0)
                tot_se_s1 = math.sqrt(max(var_s1 / float(max(n_total, 1)) + var_cheap, 0.0))
        else:
            pv_reps_s1 = stage1_rep_np[:, c0_s1 + 4] + float(state.pv_in_transit)
            if eff_r > 1 and pv_reps_s1.shape[0] > 1:
                tot_se_s1 = float(np.std(pv_reps_s1, ddof=1) / math.sqrt(float(eff_r)))
            else:
                var_s1 = max(float(s1_mean[c0_s1 + 5]) - float(s1_mean[c0_s1 + 4]) ** 2, 0.0)
                tot_se_s1 = math.sqrt(var_s1 / float(max(n_total, 1)))
        l_put_s1_val = float(s1_mean[c0_s1 + 0])
        if is_long_flag:
            eff_gearing_s1 = float(state.option_gearing)
            if abs(eff_gearing_s1) > 1e-12:
                l_put_s1_val /= eff_gearing_s1
        cpu_post_dev = torch.device("cpu")
        base_postproc = assemble_autocall_cashflows_and_pv(
            state,
            p_stop=torch.as_tensor(s1_mean[:m_rem], dtype=torch.float64, device=cpu_post_dev),
            q_cpn=torch.as_tensor(s1_mean[m_rem : 2 * m_rem], dtype=torch.float64, device=cpu_post_dev),
            l_put=torch.tensor(l_put_s1_val, dtype=torch.float64, device=cpu_post_dev),
            l_call=torch.tensor(float(s1_mean[c0_s1 + 1]), dtype=torch.float64, device=cpu_post_dev),
            l_onestar=float(s1_mean[c0_s1 + 2]),
            p_onestar=float(s1_mean[c0_s1 + 3]),
            coupon_cf_undisc=s1_mean[2 * m_rem : 3 * m_rem],
            option_se=0.0,
            total_se=tot_se_s1,
            build_cashflow_items=bool("CashFlows" in q_set or return_cashflows),
        )
        pv0 = float(base_postproc.product_value.detach().cpu().item()) if isinstance(base_postproc.product_value, torch.Tensor) else float(base_postproc.product_value)
        base_cfs = list(base_postproc.cashflows) if ("CashFlows" in q_set or return_cashflows) else []
        base_se = float(base_postproc.standard_error) if base_postproc.standard_error is not None else tot_se_s1

    if (not use_gpu_triton) or need_pv0_eval:
        pv0_1d = float(_get_pv(tok_base)[0])
        if stage1_rep_np is None:
            pv0 = pv0_1d + float(state.cash_to_settle_today)
    else:
        pv0_1d = float(pv0) - float(state.cash_to_settle_today)

    bar_r_nat_from_q = np.zeros((d_assets, m_rem), dtype=np.float64)
    if want_vega and quanto_meta and need_stage2_aad:
        a_vec_q = (np.arange(n_sub, dtype=np.float64) + 0.5) / float(n_sub)
        bq3 = bar_q_shift_np.reshape(d_assets, m_rem, n_sub)
        bar_r_nat_from_q = -np.einsum("dms,s->dm", bq3, a_vec_q)
        if m_rem > 1:
            bar_r_nat_from_q[:, :-1] -= np.einsum("dms,s->dm", bq3[:, 1:, :], 1.0 - a_vec_q)

    lv_rn_dpv_grads: Dict[int, Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}

    def _get_lv_rn_dpv_grad(idx_a: int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        hit = lv_rn_dpv_grads.get(idx_a)
        if hit is not None:
            return hit
        lv_c_a = base_lv_caches[idx_a]
        if lv_c_a is None or not need_stage2_aad:
            z_g = np.zeros(m_rem, dtype=np.float64)
            lv_rn_dpv_grads[idx_a] = (z_g, z_g)
            return z_g, z_g
        g_pair = _pullback_lv_to_r_nat_and_div(lv_c_a, bar_lv_np[idx_a])
        lv_rn_dpv_grads[idx_a] = g_pair
        return g_pair

    if want_vega:
        vega_dict = {}
        vega_arrays = {}

        for i, ticker in enumerate(sorted_ids):
            surf = state.vol_surfaces.get(ticker)
            if surf is None:
                vega_arrays[ticker] = np.zeros((0, 0), dtype=np.float64)
                continue
            lv_cache_i = base_lv_caches[i]
            assert lv_cache_i is not None
            arr_raw = _pullback_lv_to_raw_quotes(surf, lv_cache_i, bar_lv_np[i], smooth_t_gaps=False)

            qm_i = quanto_meta.get(ticker)
            if qm_i:
                rho_live_i = float(qm_i.get("rho_s_fx", 0.0))
                sig_fx_q_i = np.asarray(qm_i.get("sig_fx_obs", [0.0] * m_rem), dtype=np.float64)
                g_qm = (-bar_ln_fwd_np[i] + bar_r_nat_from_q[i]) * (rho_live_i * sig_fx_q_i * tau_obs_np)
                surf_dig = surf_dig_map.get(ticker) or _surface_content_digest(surf)
                jac_q_key = ("q_spot", surf_dig, tau_obs_dig_base, float(spots_np[i]))
                jac_q_i = _CRN_VEGA_JAC_CACHE.get(jac_q_key)
                if jac_q_i is None:
                    jac_q_i = np.stack(
                        [surf.iv_and_d_iv_d_quotes(float(tau_obs_np[m]), float(spots_np[i]))[1] for m in range(m_rem)],
                        axis=0,
                    )
                    _CRN_VEGA_JAC_CACHE.put(jac_q_key, jac_q_i)
                arr_raw = arr_raw + np.tensordot(g_qm, jac_q_i, axes=(0, 0))

            vega_arrays[ticker] = arr_raw

        for ticker in orig_ids:
            arr_raw = vega_arrays.get(ticker, np.zeros((0, 0), dtype=np.float64))
            vega_arrays[ticker] = arr_raw
            vega_dict[ticker] = arr_raw.tolist()

    if want_cega:
        cega_sorted_np = np.zeros((d_assets, d_assets), dtype=np.float64)
        if d_assets > 1:
            assert bar_chol_aad is not None
            corr_mode = str(getattr(cfg, "corr_aad_mode", "cholesky"))
            if corr_mode == "cholesky":
                s_chol = cholesky_corr_adjoint_np(chol_base, bar_chol_aad)
                cega_sorted_np = s_chol + s_chol.T - np.diag(np.diag(s_chol))
                if bar_sigma_cont_np is not None:
                    cega_sorted_np = cega_sorted_np + (bar_sigma_cont_np + bar_sigma_cont_np.T)
            else:
                bar_sigma_raw = sym_sqrt_corr_adjoint_np(v_eig_np, sqrt_ev_np, bar_chol_aad)
                if bar_sigma_cont_np is not None:
                    bar_sigma_raw = bar_sigma_raw + bar_sigma_cont_np
                cega_sorted_np = bar_sigma_raw + bar_sigma_raw.T
            np.fill_diagonal(cega_sorted_np, 0.0)

        cega_orig_np = cega_sorted_np[np.ix_(inv_perm, inv_perm)]
        cega_tensor_orig = torch.as_tensor(cega_orig_np, dtype=torch.float64, device=out_device)
        cega_rows_list = cega_orig_np.tolist()
        cega_dict = {
            orig_ids[i]: dict(zip(orig_ids, cega_rows_list[i]))
            for i in range(d_assets)
        }

    if want_vega and fx_vega_dict is not None:
        if adj_fxv_tasks:
            for ticker, i_a, rho_live_f, sig_eq_q_f in adj_fxv_tasks:
                vec_fxv = bar_ln_fwd_np[i_a] * (-rho_live_f * sig_eq_q_f * tau_obs_np)
                fx_vega_dict[ticker] = vec_fxv.tolist()
        elif fxv_tasks:
            for ticker, tok_fxv, h_fxv in fxv_tasks:
                pv_fxv = _get_pv(tok_fxv)
                fx_vega_dict[ticker] = ((pv_fxv - pv0_1d) / h_fxv).tolist()

    # Finalize Delta, Vibrato Gamma, PreTerm C^2 Gamma, and standalone MultivariateAggregatedCashGamma.
    if sp_task_meta is not None:
        if sp_task_meta["mode"] == "dg":
            pv_sp_all = _get_pv(sp_task_meta["tok"])
            cross_pairs_m = sp_task_meta["cross_pairs"]
            if want_delta:
                denom_d = np.maximum(sp_p_vec - sp_m_vec, 1e-12)
                d_sorted_np = (pv_sp_all[0 : 2 * d_assets : 2] - pv_sp_all[1 : 2 * d_assets : 2]) / denom_d
                d_orig_np = d_sorted_np[inv_perm]
                delta_tensor_orig = torch.as_tensor(d_orig_np, dtype=torch.float64, device=out_device)
                delta_dict = dict(zip(orig_ids, d_orig_np.tolist()))
            if want_gamma and not use_vibrato_gamma:
                hess_sorted_np = np.zeros((d_assets, d_assets), dtype=np.float64)
                hp_d = np.maximum(sp_p_vec - spots_np, 1e-12)
                hm_d = np.maximum(spots_np - sp_m_vec, 1e-12)
                diag_h = (
                    2.0
                    * (
                        pv_sp_all[0 : 2 * d_assets : 2] / hp_d
                        + pv_sp_all[1 : 2 * d_assets : 2] / hm_d
                        - pv0_1d * ((hp_d + hm_d) / (hp_d * hm_d))
                    )
                    / (hp_d + hm_d)
                )
                np.fill_diagonal(hess_sorted_np, diag_h)
                for c_idx, (i, j) in enumerate(cross_pairs_m):
                    b0 = 2 * d_assets + 2 * c_idx
                    hi = float(hp_d[i])
                    hj = float(hp_d[j])
                    d2 = float(pv_sp_all[b0]) - 2.0 * pv0_1d + float(pv_sp_all[b0 + 1])
                    gij = (d2 - hi * hi * float(diag_h[i]) - hj * hj * float(diag_h[j])) / (2.0 * hi * hj)
                    hess_sorted_np[i, j] = gij
                    hess_sorted_np[j, i] = gij
                gamma_orig_np = hess_sorted_np[np.ix_(inv_perm, inv_perm)]
                gamma_tensor_orig = torch.as_tensor(gamma_orig_np, dtype=torch.float64, device=out_device)
                gamma_rows_list = gamma_orig_np.tolist()
                gamma_dict = {
                    orig_ids[i]: dict(zip(orig_ids, gamma_rows_list[i]))
                    for i in range(d_assets)
                }
                if want_cash_gamma:
                    spots_orig_np = spots_np[inv_perm]
                    cash_gamma_agg = float(np.sum(np.outer(spots_orig_np, spots_orig_np) * gamma_orig_np))
        elif sp_task_meta["mode"] == "cg":
            pv_epm = _get_pv(sp_task_meta["tok"])
            cash_gamma_agg = float(pv_epm[0] - 2.0 * pv0_1d + pv_epm[1]) / (h_spot_rel * h_spot_rel)

    if use_vibrato_gamma and h_vib_np is not None:
        h_vib_spot = h_vib_np / np.maximum(np.outer(spots_np, spots_np), 1e-24)
        if want_gamma:
            hess_sorted_np = h_vib_spot.copy()
            if m_rem > 1 and pv_preterm_np.size >= 2 * d_assets + 1:
                pv0_pre = float(pv_preterm_np[-1])
                hp_v = np.maximum(sp_p_vec - spots_np, 1e-12)
                hm_v = np.maximum(spots_np - sp_m_vec, 1e-12)
                diag_pre = (
                    2.0
                    * (
                        pv_preterm_np[0 : 2 * d_assets : 2] / hp_v
                        + pv_preterm_np[1 : 2 * d_assets : 2] / hm_v
                        - pv0_pre * ((hp_v + hm_v) / (hp_v * hm_v))
                    )
                    / (hp_v + hm_v)
                )
                for d_i in range(d_assets):
                    hess_sorted_np[d_i, d_i] += float(diag_pre[d_i])
                for c_idx, (i, j) in enumerate(cross_pairs_vib):
                    b0 = 2 * d_assets + 2 * c_idx
                    hi = float(hp_v[i])
                    hj = float(hp_v[j])
                    d2 = float(pv_preterm_np[b0]) - 2.0 * pv0_pre + float(pv_preterm_np[b0 + 1])
                    gij_pre = (d2 - hi * hi * float(diag_pre[i]) - hj * hj * float(diag_pre[j])) / (2.0 * hi * hj)
                    hess_sorted_np[i, j] += gij_pre
                    hess_sorted_np[j, i] += gij_pre
            gamma_orig_np = hess_sorted_np[np.ix_(inv_perm, inv_perm)]
            gamma_tensor_orig = torch.as_tensor(gamma_orig_np, dtype=torch.float64, device=out_device)
            gamma_rows_list = gamma_orig_np.tolist()
            gamma_dict = {
                orig_ids[i]: dict(zip(orig_ids, gamma_rows_list[i]))
                for i in range(d_assets)
            }
            if want_cash_gamma:
                spots_orig_np = spots_np[inv_perm]
                cash_gamma_agg = float(np.sum(np.outer(spots_orig_np, spots_orig_np) * gamma_orig_np))
        elif want_cash_gamma:
            cg_vib = float(np.sum(h_vib_np))
            cg_pre = (
                float(pv_preterm_np[0] - 2.0 * pv_preterm_np[2] + pv_preterm_np[1]) / (h_spot_rel * h_spot_rel)
                if (m_rem > 1 and pv_preterm_np.size >= 3)
                else 0.0
            )
            cash_gamma_agg = float(cg_vib + cg_pre)

    if want_rho:
        assert rho_arrays is not None and rho_dict is not None
        if adj_rho_tasks:
            assert bar_cf_np is not None
            for cid_cur, c_obj_cur, _n_pil_c, d_df_rho, d_wf_rho, d_itr_rho, d_lf_list in adj_rho_tasks:
                dpv = d_itr_rho + d_df_rho @ bar_cf_np
                if d_wf_rho is not None and bar_surv_np is not None:
                    dpv = dpv - d_wf_rho @ bar_surv_np
                for idx_a, d_lf_a, d_rn_a, d_dpv_a in d_lf_list:
                    g_rn_lv, g_dpv_lv = _get_lv_rn_dpv_grad(idx_a)
                    dpv = dpv + d_lf_a @ bar_ln_fwd_np[idx_a] + d_rn_a @ g_rn_lv + d_dpv_a @ g_dpv_lv
                arr_r = dpv / h_rho
                if rho_unit_convention == "decimal":
                    arr_r = arr_r / _curve_quote_scale(c_obj_cur)
                rho_arrays[cid_cur] = arr_r
                rho_dict[cid_cur] = [float(x) for x in arr_r]
        else:
            for cid_cur, c_obj_cur, n_pil_c, chunk_toks in rho_tasks:
                pv_rho_all = np.empty(n_pil_c, dtype=np.float64)
                for tok_r, p_s, p_e in chunk_toks:
                    pv_rho_all[p_s:p_e] = _get_pv(tok_r)
                arr_r = (pv_rho_all - pv0_1d) / h_rho
                if rho_unit_convention == "decimal":
                    arr_r = arr_r / _curve_quote_scale(c_obj_cur)
                rho_arrays[cid_cur] = arr_r
                rho_dict[cid_cur] = [float(x) for x in arr_r]
        if adj_repo_tasks and repo_rho_dict is not None:
            for ticker, r_curve_eff, _n_pil_r, i_a, d_lf_repo_i, d_rn_repo_i, d_dpv_repo_i in adj_repo_tasks:
                g_rn_lv, g_dpv_lv = _get_lv_rn_dpv_grad(i_a)
                repo_arr_r = (d_lf_repo_i @ bar_ln_fwd_np[i_a] + d_rn_repo_i @ g_rn_lv + d_dpv_repo_i @ g_dpv_lv) / h_rho
                if rho_unit_convention == "decimal":
                    repo_arr_r = repo_arr_r / _curve_quote_scale(r_curve_eff)
                repo_rho_dict[ticker] = [float(x) for x in repo_arr_r]
        elif repo_task_meta is not None and repo_rho_dict is not None:
            tok_repo, repo_meta = repo_task_meta
            pv_repo_all = _get_pv(tok_repo)
            r_off = 0
            for ticker, r_curve_eff, n_pil_r in repo_meta:
                repo_arr_r = (pv_repo_all[r_off : r_off + n_pil_r] - pv0_1d) / h_rho
                if rho_unit_convention == "decimal":
                    repo_arr_r = repo_arr_r / _curve_quote_scale(r_curve_eff)
                repo_rho_dict[ticker] = [float(x) for x in repo_arr_r]
                r_off += n_pil_r

    if want_divs and adj_div_task is not None:
        assert divs_arrays is not None and divs_dict is not None
        aa_div_all, d_lf_div, d_dpv_div, div_slices = adj_div_task
        g_dpv_mat = np.stack([_get_lv_rn_dpv_grad(int(idx_d))[1] for idx_d in aa_div_all], axis=0)
        dpv_div_all = (np.sum(d_lf_div * bar_ln_fwd_np[aa_div_all], axis=1) + np.sum(d_dpv_div * g_dpv_mat, axis=1)) / h_div
        d_off = 0
        for ticker, n_ev in div_slices:
            if n_ev > 0:
                divs_arrays[ticker] = dpv_div_all[d_off : d_off + n_ev]
                d_off += n_ev
            else:
                divs_arrays[ticker] = np.zeros(0, dtype=np.float64)
        for ticker in orig_ids:
            arr_d = divs_arrays.get(ticker, np.zeros(0, dtype=np.float64))
            divs_arrays[ticker] = arr_d
            divs_dict[ticker] = [float(x) for x in arr_d]
    elif want_divs and div_task_meta is not None:
        assert divs_arrays is not None and divs_dict is not None
        tok_divs, div_slices = div_task_meta
        pv_div_all = _get_pv(tok_divs)
        d_off = 0
        for ticker, n_ev in div_slices:
            if n_ev > 0:
                divs_arrays[ticker] = (pv_div_all[d_off : d_off + n_ev] - pv0_1d) / h_div
                d_off += n_ev
            else:
                divs_arrays[ticker] = np.zeros(0, dtype=np.float64)
        for ticker in orig_ids:
            arr_d = divs_arrays.get(ticker, np.zeros(0, dtype=np.float64))
            divs_arrays[ticker] = arr_d
            divs_dict[ticker] = [float(x) for x in arr_d]

    if want_cega and fx_cega_dict is not None:
        if adj_fxc_tasks:
            for ticker, fx_id, i_a, fx_sign_f, sig_eq_q, sig_fx_q in adj_fxc_tasks:
                fx_cega_dict[ticker] = {
                    fx_id: float(np.sum(bar_ln_fwd_np[i_a] * (-fx_sign_f * sig_eq_q * sig_fx_q * tau_obs_np)))
                }
        elif fxc_tasks:
            for ticker, fx_id, tok_fxc, h_fxc in fxc_tasks:
                pv_fxc_pm = _get_pv(tok_fxc)
                fx_cega_dict[ticker] = {fx_id: float((pv_fxc_pm[0] - pv_fxc_pm[1]) / (2.0 * h_fxc))}

    # Step 7: Same-engine rolled Dupire QMC CRN Theta (analytical single-pass when k_cross == 0 & smooth).
    if want_theta and th_prep is not None:
        dt_shift_th = float(th_prep.get("dt_shift", float(max(1, theta_roll_days)) / 365.0))
        can_analytical_theta = bool(
            not th_prep["eff_autocalled"]
            and int(th_prep["k_cross"]) == 0
            and float(tau_obs_np[0]) > 2.0 * dt_shift_th
            and not bool(th_prep.get("has_ex_div_in_roll", False))
            and not is_cont_ki_active
            and want_rate_aad_smooth
            and bar_cf_np is not None
            and (use_triton_aad_s2 or (not use_gpu_triton and cpp_sde_v2_fn is not None))
        )
        if th_prep["eff_autocalled"] or int(th_prep["k_cross"]) >= m_rem:
            pv_th_qmc = float(th_prep["in_tr"]) + float(th_prep["crossed_pv"])
        elif can_analytical_theta and bar_cf_np is not None:
            anal_c = th_prep.get("_anal_cache")
            if anal_c is not None and (
                _arr_bytes_equal(anal_c[0], df_pay_np)
                and _arr_bytes_equal(anal_c[1], fwd_perf_np)
                and _arr_bytes_equal(anal_c[2], base_lv_tables)
                and ((anal_c[3] is None and w_fund_base_np is None) or (anal_c[3] is not None and w_fund_base_np is not None and _arr_bytes_equal(anal_c[3], w_fund_base_np)))
            ):
                d_df_th, d_wf_th, d_log_fwd_th, dq_shift_th, dlv_eff_th = anal_c[4]
            else:
                df_pay_th_np = th_prep["df_pay"]
                w_fund_th_np = th_prep["w_fund"]
                fwd_perf_th_np = th_prep["fwd_perf"]
                r_nat_th_np = th_prep["r_nat"]
                r_tot_th_np = th_prep["r_tot"]
                lv_t_grid_th = th_prep["lv_t_grid"]
                lv_th_tables = th_prep["lv_tables"]

                d_df_th = np.ascontiguousarray(df_pay_th_np - df_pay_np, dtype=np.float64)
                d_wf_th = (
                    np.ascontiguousarray(w_fund_th_np - w_fund_base_np, dtype=np.float64)
                    if (w_fund_th_np is not None and w_fund_base_np is not None)
                    else None
                )
                d_log_fwd_th = np.ascontiguousarray(
                    np.log(np.maximum(fwd_perf_th_np / np.maximum(fwd_perf_np, 1e-12), 1e-30)),
                    dtype=np.float64,
                )
                if quanto_meta:
                    q_diff_base = r_tot_obs_np - r_native_obs_np
                    q_diff_th = r_tot_th_np - r_nat_th_np
                    a_s_th = (np.arange(n_sub, dtype=np.float64) + 0.5) / float(n_sub)
                    q_prev_b = np.concatenate([np.zeros((d_assets, 1), dtype=np.float64), q_diff_base[:, :-1]], axis=1)
                    q_prev_t = np.concatenate([np.zeros((d_assets, 1), dtype=np.float64), q_diff_th[:, :-1]], axis=1)
                    dq_shift_th = np.ascontiguousarray(
                        (
                            (q_prev_t - q_prev_b)[:, :, None]
                            + ((q_diff_th - q_prev_t) - (q_diff_base - q_prev_b))[:, :, None] * a_s_th[None, None, :]
                        ).reshape(d_assets, m_rem * n_sub),
                        dtype=np.float64,
                    )
                else:
                    dq_shift_th = None

                tau0_b = max(float(tau_obs_np[0]), 1e-6)
                t_query_lv = np.where(
                    lv_t_grid <= tau0_b,
                    lv_t_grid * max((tau0_b - dt_shift_th) / tau0_b, 0.0),
                    lv_t_grid - dt_shift_th,
                )
                t_query_lv = np.clip(t_query_lv, float(lv_t_grid_th[0]), float(lv_t_grid_th[-1]))
                u_tq = np.interp(t_query_lv, lv_t_grid_th, np.arange(lv_nt, dtype=np.float64))
                it0_q = np.clip(np.floor(u_tq).astype(np.int64), 0, lv_nt - 1)
                it1_q = np.clip(it0_q + 1, 0, lv_nt - 1)
                wt_q = (u_tq - it0_q)[None, :, None]
                sig_scale_tau0 = math.sqrt(max(1.0 - dt_shift_th / tau0_b, 0.0)) - 1.0
                if lv_nt > 1:
                    dt_g_half = 0.5 * (t_max / float(lv_nt - 1))
                    w_p0_t = np.clip((tau0_b + dt_g_half - lv_t_grid) / max(2.0 * dt_g_half, 1e-12), 0.0, 1.0)[None, :, None]
                else:
                    w_p0_t = np.ones((1, 1, 1), dtype=np.float64)
                lv_th_interp_all = (1.0 - wt_q) * lv_th_tables[:, it0_q, :] + wt_q * lv_th_tables[:, it1_q, :]
                dlv_eff_th = np.ascontiguousarray(
                    (lv_th_interp_all - base_lv_tables) + (sig_scale_tau0 * w_p0_t) * base_lv_tables,
                    dtype=np.float64,
                )
                th_prep["_anal_cache"] = (
                    df_pay_np.copy(),
                    fwd_perf_np.copy(),
                    base_lv_tables.copy(),
                    w_fund_base_np.copy() if w_fund_base_np is not None else None,
                    (d_df_th, d_wf_th, d_log_fwd_th, dq_shift_th, dlv_eff_th),
                )

            dv_df_fund = float(np.dot(d_df_th, bar_cf_np))
            if d_wf_th is not None and bar_surv_np is not None:
                dv_df_fund -= float(np.dot(d_wf_th, bar_surv_np))
            dv_in_transit = float(th_prep["in_tr"]) - float(state.pv_in_transit) + float(th_prep["crossed_pv"])
            dv_quanto_fwd = float(np.sum(d_log_fwd_th * bar_ln_fwd_np))
            if dq_shift_th is not None:
                dv_quanto_fwd += float(np.sum(dq_shift_th * bar_q_shift_np))
            dv_lv_roll_and_tau = float(np.sum(bar_lv_np * dlv_eff_th))

            pv_th_qmc = pv0_1d + dv_df_fund + dv_lv_roll_and_tau + dv_in_transit + dv_quanto_fwd
        elif use_gpu_triton and pv_th_fused_np.size > 0:
            pv_th_qmc = float(pv_th_fused_np[0])
        else:
            k_cross = int(th_prep["k_cross"])
            m_rem_th = int(th_prep["m_rem_th"])
            tau_obs_th_np = th_prep["tau_obs"]
            df_pay_th_np = th_prep["df_pay"]
            fwd_perf_th_np = th_prep["fwd_perf"]
            r_nat_th_np = th_prep["r_nat"]
            r_tot_th_np = th_prep["r_tot"]
            w_fund_th_np = th_prep["w_fund"]
            lv_t_grid_th = th_prep["lv_t_grid"]
            lv_th_tables = th_prep["lv_tables"]
            in_tr_th = float(th_prep["in_tr"])
            eff_iki_th = float(th_prep["eff_iki"])
            eff_cmem_th = float(th_prep["eff_cmem"])
            crossed_pv = float(th_prep["crossed_pv"])

            assert z_indep_np is not None
            norm_x_th_np = np.empty((d_assets, m_rem_th, n_total), dtype=np.float64)
            if cpp_sde_v2_fn is not None:
                z_th_indep = np.ascontiguousarray(z_indep_np[k_cross:], dtype=np.float64)
                if callable(getattr(cpp_sde_v2_fn, "__call__", None)) and not hasattr(cpp_sde_v2_fn, "argtypes"):
                    _, out_m_th_v2, _, _, _ = cpp_sde_v2_fn(
                        lv_tables=lv_th_tables,
                        z_indep=z_th_indep,
                        chol_base=chol_base,
                        tau_obs=tau_obs_th_np,
                        r_native_obs=r_nat_th_np,
                        r_tot_obs=r_tot_th_np,
                        lv_t_grid=lv_t_grid_th,
                        lv_y_min=lv_y_min_vec,
                        lv_y_max=lv_y_max_vec,
                        n_sub=n_sub,
                        eff_r=eff_r,
                        n_rep=n_rep,
                        run_forward=True,
                        run_backward=False,
                    )
                    assert out_m_th_v2 is not None
                    norm_x_th_np = np.transpose(out_m_th_v2, (1, 0, 2))
                else:
                    lx_p_th = np.empty((m_rem_th, d_assets, n_total), dtype=np.float64)
                    nx_p_th = np.empty((m_rem_th, d_assets, n_total), dtype=np.float64)
                    lv_th64 = np.ascontiguousarray(lv_th_tables, dtype=np.float64)
                    ch_th64 = np.ascontiguousarray(chol_base, dtype=np.float64)
                    tau_th64 = np.ascontiguousarray(tau_obs_th_np, dtype=np.float64)
                    rn_th64 = np.ascontiguousarray(r_nat_th_np, dtype=np.float64)
                    rt_th64 = np.ascontiguousarray(r_tot_th_np, dtype=np.float64)
                    lvt_th64 = np.ascontiguousarray(lv_t_grid_th, dtype=np.float64)
                    lvy0_th64 = np.ascontiguousarray(lv_y_min_vec, dtype=np.float64)
                    lvy1_th64 = np.ascontiguousarray(lv_y_max_vec, dtype=np.float64)
                    sde_th_args = _PlatenSdeAadArgsC(
                        d_assets=d_assets,
                        m_rem=m_rem_th,
                        n_sub=n_sub,
                        eff_r=eff_r,
                        n_rep=n_rep,
                        lv_nt=lv_nt,
                        lv_ny=lv_ny,
                        lv_tables=lv_th64.ctypes.data,
                        z_indep=z_th_indep.ctypes.data,
                        chol_base=ch_th64.ctypes.data,
                        tau_obs=tau_th64.ctypes.data,
                        r_native_obs=rn_th64.ctypes.data,
                        r_tot_obs=rt_th64.ctypes.data,
                        lv_t_grid=lvt_th64.ctypes.data,
                        lv_y_min=lvy0_th64.ctypes.data,
                        lv_y_max=lvy1_th64.ctypes.data,
                        bar_lx_post=0,
                        lx_norm_in=0,
                        q_shift_sub=0,
                        lx_period_out=lx_p_th.ctypes.data,
                        norm_x_out=nx_p_th.ctypes.data,
                        bar_lv_out=0,
                        bar_chol_out=0,
                        bar_q_shift_out=0,
                        run_forward=1,
                        run_backward=0,
                        compute_lv_aad=0,
                        compute_chol_aad=0,
                        compute_q_aad=0,
                        has_q_shift_sub=0,
                    )
                    cpp_sde_v2_fn(ctypes.byref(sde_th_args))
                    norm_x_th_np = np.transpose(nx_p_th, (1, 0, 2))
            else:
                _, _, out_m_th = _cpu_step_and_ckpt_sde(
                    lv_th_tables,
                    z_indep_np[k_cross:],
                    chol_base,
                    tau_obs_th_np,
                    r_nat_th_np,
                    r_tot_th_np,
                    lv_t_grid_th,
                    lv_y_min_vec,
                    lv_y_max_vec,
                    n_sub,
                    eff_r,
                    n_rep,
                )
                norm_x_th_np = np.transpose(out_m_th, (1, 0, 2))
            perf_th_dev = np.ascontiguousarray(
                np.transpose(norm_x_th_np * fwd_perf_th_np[:, :, None], (1, 0, 2)),
                dtype=np.float64,
            )

            if k_cross == 0 and ki_monitoring_type == "European":
                pv_th_qmc = float(
                    _run_loo_l2o_eval(
                        perf_th_dev,
                        None,
                        None,
                        np.array([-1], dtype=np.int32),
                        np.array([-1], dtype=np.int32),
                        df_pay_th_np[None, :],
                        w_fund_th_np[None, :] if w_fund_th_np is not None else None,
                        np.array([in_tr_th + crossed_pv], dtype=np.float64),
                    )[0]
                )
            else:
                p_th_t = torch.from_numpy(perf_th_dev)
                wc_t = _eval_index_batch_torch(p_th_t, call_code, inv_perm)
                wcp_t = wc_t if cpn_code == call_code else _eval_index_batch_torch(p_th_t, cpn_code, inv_perm)
                wo_t = wc_t if opt_code == call_code else (wcp_t if opt_code == cpn_code else _eval_index_batch_torch(p_th_t, opt_code, inv_perm))
                wavg_last_th = _eval_index_batch_torch(p_th_t[m_rem_th - 1 : m_rem_th], 2, inv_perm)[0] if is_long_flag else None
                wb_t = _eval_index_batch_torch(p_th_t[m_rem_th - 1 : m_rem_th], 1, inv_perm)[0] if k_os > 0.0 else None

                surv_v = torch.ones(n_total, dtype=torch.float64)
                cmem_v = torch.full((n_total,), float(eff_cmem_th), dtype=torch.float64)
                pv_p = torch.zeros(n_total, dtype=torch.float64)
                eff_csp = -cpn_spread if dir_down else cpn_spread
                for m_loc in range(m_rem_th):
                    m_orig = k_cross + m_loc
                    if is_call_np[m_orig] > 0.5:
                        bk = ko_arr_np[m_orig]
                        if abs(ko_spread) <= 1e-6:
                            gk = (wc_t[m_loc] >= bk).to(torch.float64)
                        else:
                            bl, bh = min(bk, bk + ko_spread), max(bk, bk + ko_spread)
                            gk = ((wc_t[m_loc] - bl) / max(bh - bl, 1e-12)).clamp(0.0, 1.0)
                    else:
                        gk = torch.zeros(n_total, dtype=torch.float64)
                    if is_guar_np[m_orig] > 0.5:
                        gc = torch.ones(n_total, dtype=torch.float64)
                    else:
                        bc = cpn_arr_np[m_orig]
                        if abs(eff_csp) <= 1e-6:
                            hd = (wcp_t[m_loc] <= bc) if dir_down else (wcp_t[m_loc] >= bc)
                            gc = (1.0 - hd.to(torch.float64)) if type_out else hd.to(torch.float64)
                        else:
                            blc, bhc = min(bc, bc + eff_csp), max(bc, bc + eff_csp)
                            ru = ((wcp_t[m_loc] - blc) / max(bhc - blc, 1e-12)).clamp(0.0, 1.0)
                            hg = (1.0 - ru) if dir_down else ru
                            gc = (1.0 - hg) if type_out else hg
                    egc = torch.maximum(gc, gk) if force_cpn_ac else gc
                    pm = surv_v * gk
                    cr = float(cpn_rates_np[m_orig])
                    if is_memory:
                        ec = cr + cmem_v
                        ccf = surv_v * egc * ec + pm * float(state.autocall_coupon)
                        cmem_v = surv_v * ec * (1.0 - egc)
                    else:
                        ccf = surv_v * egc * cr + pm * float(state.autocall_coupon)
                    pv_p = pv_p + df_pay_th_np[m_loc] * ccf
                    if not no_notional:
                        pv_p = pv_p + df_pay_th_np[m_loc] * pm
                    if w_fund_th_np is not None:
                        pv_p = pv_p - w_fund_th_np[m_loc] * surv_v
                    surv_v = (surv_v - pm).clamp(min=0.0)
                if not no_notional:
                    pv_p = pv_p + df_pay_th_np[m_rem_th - 1] * surv_v
                kig_th: torch.Tensor | float
                if has_ki and float(eff_iki_th) < 0.5 and ki_monitoring_type in ("Continuous", "Daily", "Discrete"):
                    iki_th_cpu: List[torch.Tensor] = []
                    _compute_payoff_aad_state_adjoints_torch(
                        p_th_t,
                        ko_barriers=torch.from_numpy(ko_arr_np[k_cross:]),
                        is_callable=torch.from_numpy(is_call_np[k_cross:]),
                        cpn_barriers=torch.from_numpy(cpn_arr_np[k_cross:]),
                        is_guar=torch.from_numpy(is_guar_np[k_cross:]),
                        cpn_rates=torch.from_numpy(cpn_rates_np[k_cross:]),
                        df_pay=torch.from_numpy(df_pay_th_np),
                        w_fund=torch.from_numpy(w_fund_th_np) if w_fund_th_np is not None else None,
                        inv_perm=inv_perm,
                        call_code=call_code,
                        cpn_code=cpn_code,
                        opt_code=opt_code,
                        ko_spread=ko_spread,
                        cpn_spread=cpn_spread,
                        ki_spread=ki_spread,
                        cpn_dir_down=dir_down,
                        cpn_type_out=type_out,
                        is_memory=is_memory,
                        force_cpn_ac=force_cpn_ac,
                        no_notional=no_notional,
                        is_long_basket=is_long_flag,
                        has_ki=has_ki,
                        ki_barrier=b_ki,
                        i_ki_past=float(eff_iki_th),
                        ki_monitoring_type=ki_monitoring_type,
                        tau_obs=tau_obs_th_np,
                        vol_atm_obs=vol_atm_obs[:, k_cross:] if vol_atm_obs is not None else None,
                        strike=k_put,
                        lower_strike=state.lower_strike,
                        option_gearing=float(state.option_gearing),
                        ac_cpn=float(state.autocall_coupon),
                        c_mem_0=float(eff_cmem_th),
                        part_call_strike=state.participation_call_strike,
                        part_call_rate=part_call_rate,
                        one_star_strike=(k_os if k_os > 0.0 else state.one_star_strike),
                        one_star_coupon=os_cpn,
                        one_star_eps=one_star_eps,
                        barrier_eps_min=barrier_eps_min,
                        barrier_eps_max=barrier_eps_max,
                        lv_tables_t=torch.from_numpy(lv_th_tables),
                        corr_t=torch.from_numpy(corr_np),
                        fwd_perf_t=torch.from_numpy(fwd_perf_th_np),
                        y_min=lv_y_min_vec,
                        y_max=lv_y_max_vec,
                        n_sub=n_sub,
                        n_rep=n_rep,
                        emit_stage1=False,
                        compute_grads=False,
                        out_i_ki_ext=iki_th_cpu,
                    )
                    kig_th = iki_th_cpu[0] if iki_th_cpu else 1.0
                elif has_ki:
                    b_ki_th = k_put if is_long_flag else b_ki
                    if abs(ki_spread) <= 1e-6:
                        gki = (wo_t[m_rem_th - 1] < b_ki_th).to(torch.float64)
                    else:
                        blk, bhk = min(b_ki_th, b_ki_th + ki_spread), max(b_ki_th, b_ki_th + ki_spread)
                        gki = ((bhk - wo_t[m_rem_th - 1]) / max(bhk - blk, 1e-12)).clamp(0.0, 1.0)
                    kig_th = torch.maximum(torch.full_like(gki, float(eff_iki_th)), gki)
                else:
                    kig_th = 1.0
                sind_th: torch.Tensor | float
                if wb_t is not None:
                    if one_star_eps <= 1e-6:
                        sind_th = (wb_t >= k_os).to(torch.float64)
                    else:
                        sind_th = ((wb_t - (k_os - 0.5 * one_star_eps)) / one_star_eps).clamp(0.0, 1.0)
                else:
                    sind_th = 0.0
                opay = torch.zeros(n_total, dtype=torch.float64)
                if is_long_flag:
                    assert wavg_last_th is not None
                    opay = opay - kig_th * (1.0 - sind_th) * (1.0 - wavg_last_th)
                else:
                    has_low_k_th = state.lower_strike is not None
                    if (not has_low_k_th or k_put > k_low) and float(state.option_gearing) > 0.0:
                        p_lo_th = (k_low - wo_t[m_rem_th - 1]).clamp(min=0.0) if has_low_k_th else 0.0
                        p_in = (k_put - wo_t[m_rem_th - 1]).clamp(min=0.0) - p_lo_th
                        opay = opay - float(state.option_gearing) * kig_th * (1.0 - sind_th) * p_in
                if has_part_call_flag:
                    opay = opay + part_call_rate * (wo_t[m_rem_th - 1] - k_call).clamp(min=0.0)
                if k_os > 0.0 and os_cpn != 0.0:
                    opay = opay + sind_th * os_cpn
                pv_p = pv_p + df_pay_th_np[m_rem_th - 1] * surv_v * opay
                pv_th_qmc = float(pv_p.mean().item()) + in_tr_th + crossed_pv

        theta_val = float(pv_th_qmc - pv0_1d)

    return AutocallGreeksResult(
        product_value=pv0,
        cash_flows=base_cfs,
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
        fx_vega=fx_vega_dict,
        fx_cega=fx_cega_dict,
        repo_rho=repo_rho_dict,
        reporting_currency=state.payment_currency,
        standard_error=base_se,
        cash_to_settle_today=float(state.cash_to_settle_today),
        postproc_output=base_postproc,
        greeks_engine_used=engine_label,
    )
