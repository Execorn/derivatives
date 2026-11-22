"""dupire_qmc_gpu.py — GPU-Vectorized Owen-Scrambled RQMC Dupire Local-Vol Solver & PEMC.

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

from collections import OrderedDict
from dataclasses import dataclass
import math
import threading
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.stats import norm, qmc
import torch

from deepvol.models.autocall.dupire_qmc_solver import (
    DupireQMCResult,
    _assemble_qmc_postproc,
    _compute_funding_weights_np,
    _next_power_of_two,
    _resolve_r_native_obs_tensor,
    price_eq_autocall_dupire_qmc,
)
from deepvol.models.autocall.pre_post_processor import (
    AutocallPreprocState,
    assemble_autocall_cashflows_and_pv,
)

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False

_SOBOL_MAX_DIM: int = 21201
_SOBOL_BITS: int = 30
_SOBOL_INV_SCALE: float = 1.0 / float(1 << _SOBOL_BITS)

_SOBOL_HOST_SV: Optional[npt.NDArray[np.int32]] = None
_SOBOL_DEV_CACHE: Dict[str, torch.Tensor] = {}
_SOBOL_LOCK = threading.Lock()

_SURF_GPU_CACHE_LOCK = threading.Lock()
_SURF_GPU_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
_SURF_GPU_CACHE_MAXSIZE: int = 4096


@dataclass(frozen=True)
class DupireQMCConfig:
    """Configuration for GPU/CPU Owen-scrambled RQMC Dupire LV simulation."""

    n_paths: int = 32768
    n_replicates: int = 8
    n_substeps: int = 16
    mode: Literal["dupire_lv", "terminal_iv"] = "dupire_lv"
    precision: Literal["mixed_fp32_fp64", "fp64_audit"] = "mixed_fp32_fp64"
    cv_mode: Literal["bs1d_exact", "multifidelity_gbm", "none"] = "bs1d_exact"
    seed: int = 42
    lv_grid_nt: int = 64
    lv_grid_ny: int = 128


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
        """Multi-contract (D, M_rem)-bucketed GPU QMC teacher for <15-min M4 dataset generation.

        Returns:
            Tuple of (pv_mc, p_stop_mc, q_cpn_mc, l_put_mc, delta_mc_raw_or_none) in torch.float64.
        """
        ...


def _get_sobol_direction_numbers(device: torch.device) -> torch.Tensor:
    """Returns cached (21201, 30) Joe-Kuo Sobol direction numbers on `device` (< 2.6 MB)."""
    global _SOBOL_HOST_SV
    dev_key = str(device)
    with _SOBOL_LOCK:
        cached = _SOBOL_DEV_CACHE.get(dev_key)
        if cached is not None:
            return cached
        if _SOBOL_HOST_SV is None:
            sob = qmc.Sobol(d=_SOBOL_MAX_DIM, scramble=False)
            _SOBOL_HOST_SV = np.ascontiguousarray(sob._sv.astype(np.int32))
        t_gpu = torch.from_numpy(_SOBOL_HOST_SV).to(device=device, non_blocking=False)
        _SOBOL_DEV_CACHE[dev_key] = t_gpu
        return t_gpu


def _get_surface_gpu_tensors(surf: Any, device: torch.device) -> Dict[str, Any]:
    """External thread-safe LRU cache of GPU spline tensors for a `RegularizedVolSurface`."""
    key = (
        surf.strikes.shape,
        surf.strikes.tobytes(),
        surf.tenors.shape,
        surf.tenors.tobytes(),
        surf.quotes.shape,
        surf.quotes.tobytes(),
        str(getattr(surf, "calendar_proj_digest", "")),
        str(getattr(surf, "smile_extrap", "Flat")),
        float(surf.min_vol),
        float(surf.max_local_vol),
        str(device),
    )
    with _SURF_GPU_CACHE_LOCK:
        hit = _SURF_GPU_CACHE.get(key)
        if hit is not None:
            _SURF_GPU_CACHE.move_to_end(key)
            return hit

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
    with _SURF_GPU_CACHE_LOCK:
        _SURF_GPU_CACHE[key] = entry
        _SURF_GPU_CACHE.move_to_end(key)
        while len(_SURF_GPU_CACHE) > _SURF_GPU_CACHE_MAXSIZE:
            _SURF_GPU_CACHE.popitem(last=False)
    return entry


if _HAS_TRITON:

    @triton.jit  # type: ignore[untyped-decorator]
    def _sobol_normals_triton_kernel(  # type: ignore[no-untyped-def]
        sv_ptr,
        shifts_ptr,
        out_ptr,
        dim_offset,
        n_dims,
        n_rep: tl.constexpr,
        m_pow: tl.constexpr,
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
        for b in tl.static_range(m_pow):
            bit_on = (gray >> b) & 1
            dir_num = tl.load(sv_ptr + dim_idx * 30 + b).to(tl.uint32)
            acc = acc ^ tl.where(bit_on != 0, dir_num, tl.zeros((BLOCK_N,), dtype=tl.uint32))

        shift = tl.load(shifts_ptr + pid_r * n_dims + pid_d).to(tl.uint32)
        acc = acc ^ shift
        u = (acc.to(tl.float32) + 0.5) * (1.0 / 1073741824.0)
        u = tl.minimum(tl.maximum(u, 1e-6), 1.0 - 1e-6)
        z = tl.extra.cuda.libdevice.normcdfinv(u)

        out_off = (pid_r * n_dims + pid_d) * n_rep + offs_n
        tl.store(out_ptr + out_off, z, mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _dupire_substep_triton_kernel(  # type: ignore[no-untyped-def]
        log_x_ptr,      # (R, D, N_rep) float32
        z_corr_ptr,     # (R, D, n_sub, N_rep) float32
        lv_table_ptr,   # (D, N_t, N_y) float32
        y_min_ptr,      # (D,) float32
        inv_dy_ptr,     # (D,) float32
        t_start,        # float32
        dt_sub,         # float32
        sqrt_dt_sub,    # float32
        inv_dt_grid,    # float32 = (N_t - 1) / tau_max
        d_assets,       # int32
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
        lv_base = pid_d * (N_t * N_y)
        z_base = (pid_r * d_assets + pid_d) * (n_sub * n_rep)

        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4

        for s in tl.static_range(n_sub):
            t_step = t_start + (float(s) + 0.5) * dt_sub
            ut = tl.minimum(tl.maximum(t_step * inv_dt_grid, 0.0), max_t_f)
            it0 = ut.to(tl.int32)
            ft = ut - it0.to(tl.float32)
            it1 = tl.minimum(it0 + 1, N_t - 1)

            uy = tl.minimum(tl.maximum((lx - y_min) * inv_dy, 0.0), max_y_f)
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
            lv = v0 + ft * (v1 - v0)

            z = tl.load(z_corr_ptr + z_base + s * n_rep + offs_n, mask=mask_n, other=0.0)
            lx = lx + (-0.5 * lv * lv * dt_sub + lv * sqrt_dt_sub * z)

        tl.store(log_x_ptr + lx_off, lx, mask=mask_n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _dupire_batch_substep_triton_kernel(  # type: ignore[no-untyped-def]
        log_x_ptr,      # (B_c, R, D, N_rep) float32
        dlog_x_ptr,     # (B_c, R, D, N_rep) float32
        z_indep_ptr,    # (R, D, n_sub, N_rep) float32
        chol_ptr,       # (B_c, D, D) float32
        lv_table_ptr,   # (B_c, D, N_t, N_y) float32
        y_min_ptr,      # (B_c, D) float32
        inv_dy_ptr,     # (B_c, D) float32
        t_start_ptr,    # (B_c,) float32
        dt_sub_ptr,     # (B_c,) float32
        sqrt_dt_ptr,    # (B_c,) float32
        inv_dt_grid_ptr,# (B_c,) float32
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
        pid_bd = tl.program_id(1)  # b * D + d
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
        t_start = tl.load(t_start_ptr + pid_b)
        dt_sub = tl.load(dt_sub_ptr + pid_b)
        sqrt_dt = tl.load(sqrt_dt_ptr + pid_b)
        inv_dt_grid = tl.load(inv_dt_grid_ptr + pid_b)

        lv_base = (pid_b * d_assets + pid_d) * (N_t * N_y)
        chol_row_base = (pid_b * d_assets + pid_d) * d_assets

        max_t_f = float(N_t - 1) - 1e-4
        max_y_f = float(N_y - 1) - 1e-4

        for s in tl.static_range(n_sub):
            t_step = t_start + (float(s) + 0.5) * dt_sub
            ut = tl.minimum(tl.maximum(t_step * inv_dt_grid, 0.0), max_t_f)
            it0 = ut.to(tl.int32)
            ft = ut - it0.to(tl.float32)
            it1 = tl.minimum(it0 + 1, N_t - 1)

            uy = tl.minimum(tl.maximum((lx - y_min) * inv_dy, 0.0), max_y_f)
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
            lv = v0 + ft * (v1 - v0)

            # Inline Cholesky correlation across D inside registers
            z_c = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k in range(d_assets):
                l_dk = tl.load(chol_ptr + chol_row_base + k)
                z_k_off = ((pid_r * d_assets + k) * n_sub + s) * n_rep + offs_n
                z_k = tl.load(z_indep_ptr + z_k_off, mask=mask_n, other=0.0)
                z_c = z_c + l_dk * z_k

            lx = lx + (-0.5 * lv * lv * dt_sub + lv * sqrt_dt * z_c)
            if COMPUTE_DELTA:
                dlv_dy = ((1.0 - ft) * (v01 - v00) + ft * (v11 - v10)) * inv_dy
                dlx = dlx + (-lv * dlv_dy * dt_sub + dlv_dy * sqrt_dt * z_c) * (1.0 + dlx)

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
        unp_next = tl.where(is_mem > 0.5, eff_c * (1.0 - cpn_e), unp)
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
                d_unp_next = tl.where(is_mem > 0.5, d_eff_c * (1.0 - cpn_e) - eff_c * d_cpn, d_unp)
                d_tpv_next = d_tpv - d_al * wf_m + df_m * d_cpn_pay + not_mult * tl.where(is_last != 0, d_stop + d_al_next, d_stop)

                tl.store(d_alive_ptr + off_3d, d_al_next, mask=mask_n)
                tl.store(d_unpaid_ptr + off_3d, d_unp_next, mask=mask_n)
                tl.store(d_total_pv_ptr + off_3d, d_tpv_next, mask=mask_n)


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
        gen_s = torch.Generator(device=device)
        gen_s.manual_seed(int(seed) + dim_offset * 131 + 17)
        shifts = torch.randint(
            0, 1 << _SOBOL_BITS, (n_replicates, n_sobol), dtype=torch.int32, device=device, generator=gen_s
        )
        if use_triton and _HAS_TRITON and device.type == "cuda":
            z_sobol = torch.empty((n_replicates, n_sobol, n_rep), dtype=torch.float32, device=device)
            block_n = min(256, n_rep)
            grid = (triton.cdiv(n_rep, block_n), n_sobol, n_replicates)
            _sobol_normals_triton_kernel[grid](
                sv_gpu,
                shifts,
                z_sobol,
                dim_offset,
                n_sobol,
                n_rep=n_rep,
                m_pow=m_pow,
                BLOCK_N=block_n,
            )
        else:
            idx = torch.arange(n_rep, dtype=torch.int64, device=device)
            gray = idx ^ (idx >> 1)
            sv_slice = sv_gpu[dim_offset : dim_offset + n_sobol, :m_pow].to(torch.int64)  # (n_sobol, m_pow)
            bits = ((gray[None, :, None] >> torch.arange(m_pow, device=device)[None, None, :]) & 1)
            # Accumulate XOR over m_pow bits
            acc = torch.zeros((n_replicates, n_sobol, n_rep), dtype=torch.int64, device=device)
            for b in range(m_pow):
                mask_b = bits[0, :, b]  # (N_rep,)
                acc ^= mask_b[None, None, :] * sv_slice[None, :, b : b + 1]
            acc ^= shifts.to(torch.int64).unsqueeze(-1)
            u = (acc.to(torch.float32) + 0.5) * _SOBOL_INV_SCALE
            u = u.clamp(1e-6, 1.0 - 1e-6)
            z_sobol = torch.special.ndtri(u)
        parts.append(z_sobol)

    if n_philox > 0:
        gen_p = torch.Generator(device=device)
        gen_p.manual_seed(int(seed) + dim_offset * 977 + 99991)
        half = (n_rep + 1) // 2
        z_half = torch.randn(
            (n_replicates, n_philox, half), dtype=torch.float32, device=device, generator=gen_p
        )
        z_philox = torch.cat([z_half, -z_half], dim=-1)[:, :, :n_rep]
        parts.append(z_philox)

    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds batched (A, N_t, N_y) float32 Dupire LV tables via fused Triton kernel."""
    a_cnt = len(surfs)
    if a_cnt == 0:
        return (
            torch.empty((0, n_t, n_y), dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

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
        return (
            torch.stack(lv_l, dim=0),
            torch.tensor(ym_l, dtype=torch.float32, device=device),
            torch.tensor(idy_l, dtype=torch.float32, device=device),
        )

    tau_2d = tau_obs.unsqueeze(0).expand(a_cnt, -1) if tau_obs.ndim == 1 else tau_obs
    m_len = int(tau_2d.shape[1])
    tau_max = torch.clamp(tau_2d[:, -1], min=1e-4)
    steps = torch.linspace(0.0, 1.0, n_t, dtype=torch.float64, device=device)
    t_eff = torch.clamp(tau_max[:, None] * steps[None, :], min=1e-4)

    zeros_a = torch.zeros((a_cnt, 1), dtype=torch.float64, device=device)
    ones_a = torch.ones((a_cnt, 1), dtype=torch.float64, device=device)
    tau_knots = torch.cat([zeros_a, tau_2d], dim=1).contiguous()
    r_tot_knots = torch.cat([zeros_a, r_tot_obs], dim=1)
    log_df_knots = torch.log(torch.clamp(torch.cat([ones_a, df_fin_obs], dim=1), min=1e-15))

    k_idx = torch.clamp(torch.searchsorted(tau_knots, t_eff, right=True) - 1, 0, m_len - 1)
    tau0 = torch.gather(tau_knots, 1, k_idx)
    tau1 = torch.gather(tau_knots, 1, k_idx + 1)
    r0 = torch.gather(r_tot_knots, 1, k_idx)
    r1 = torch.gather(r_tot_knots, 1, k_idx + 1)
    ldf0 = torch.gather(log_df_knots, 1, k_idx)
    ldf1 = torch.gather(log_df_knots, 1, k_idx + 1)

    dt_k = torch.clamp(tau1 - tau0, min=1e-8)
    alpha_k = (t_eff - tau0) / dt_k
    r_tot_grid = r0 + alpha_k * (r1 - r0)
    r_zero_grid = -(ldf1 - ldf0) / dt_k
    q_repo_grid = r_zero_grid - (r1 - r0) / dt_k

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
        mask_ex = (ex_taus[:, None, :] > 0.0) & (ex_taus[:, None, :] <= t_eff[:, :, None] + 1e-12)
        d_pv_grid = (mask_ex.to(torch.float64) * ex_pvs[:, None, :]).sum(dim=2)
        mask_ex0 = (ex_taus[:, None, :] > 0.0) & (ex_taus[:, None, :] <= tau0[:, :, None] + 1e-12)
        d_pv0 = (mask_ex0.to(torch.float64) * ex_pvs[:, None, :]).sum(dim=2)
        esc0 = torch.clamp(spots_t0[:, None] - d_pv0, min=0.01 * spots_t0[:, None])
        mask_ex1 = (ex_taus[:, None, :] > 0.0) & (ex_taus[:, None, :] <= tau1[:, :, None] + 1e-12)
        d_pv1 = (mask_ex1.to(torch.float64) * ex_pvs[:, None, :]).sum(dim=2)
        esc1 = torch.clamp(spots_t0[:, None] - d_pv1, min=0.01 * spots_t0[:, None])
        q_div_grid = -torch.log(torch.clamp(esc1 / torch.clamp(esc0, min=1e-12), min=1e-12)) / dt_k
        q_repo_grid = q_repo_grid + q_div_grid
    else:
        d_pv_grid = torch.zeros((a_cnt, n_t), dtype=torch.float64, device=device)

    escrowed_grid = torch.clamp(spots_t0[:, None] - d_pv_grid, min=0.01 * spots_t0[:, None])
    f_grid = torch.clamp(escrowed_grid * torch.exp(r_tot_grid), min=1e-4)

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
    _build_lv_table_triton_kernel[(n_t, a_cnt)](
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
    return lv_out, y_min_64.to(torch.float32), inv_dy_32


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
    """Builds adaptive (N_t, N_y) float32 Dupire LV table on GPU in < 0.2 ms."""
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

    # Compute exact escrowed forward F(0, t_k) and r_zero(t_k), q_repo(t_k) across t_grid_365
    tau_knots = torch.cat(
        [torch.zeros(1, dtype=torch.float64, device=device), torch.as_tensor(tau_obs, dtype=torch.float64, device=device)]
    )
    r_tot_knots = torch.cat(
        [torch.zeros(1, dtype=torch.float64, device=device), torch.as_tensor(r_tot_obs_i, dtype=torch.float64, device=device)]
    )
    df_fin_knots = torch.cat(
        [torch.ones(1, dtype=torch.float64, device=device), torch.as_tensor(df_fin_obs_i, dtype=torch.float64, device=device)]
    )
    log_df_knots = torch.log(torch.clamp(df_fin_knots, min=1e-15))

    k_idx = torch.clamp(torch.searchsorted(tau_knots, t_eff_365, right=True) - 1, 0, len(tau_obs) - 1)
    dt_k = torch.clamp(tau_knots[k_idx + 1] - tau_knots[k_idx], min=1e-8)
    alpha_k = (t_eff_365 - tau_knots[k_idx]) / dt_k
    r_tot_grid = r_tot_knots[k_idx] + alpha_k * (r_tot_knots[k_idx + 1] - r_tot_knots[k_idx])
    r_zero_grid = -(log_df_knots[k_idx + 1] - log_df_knots[k_idx]) / dt_k
    drift_net_grid = (r_tot_knots[k_idx + 1] - r_tot_knots[k_idx]) / dt_k
    q_repo_grid = r_zero_grid - drift_net_grid

    if len(div_events_i) > 0:
        ex_taus = torch.tensor(
            [float(getattr(ev, "tau_ex_act365", 0.0)) for ev in div_events_i],
            dtype=torch.float64,
            device=device,
        )
        ex_pvs = torch.tensor(
            [float(getattr(ev, "amount", 0.0)) * float(getattr(ev, "pv_discount_weight", 0.0)) for ev in div_events_i],
            dtype=torch.float64,
            device=device,
        )
        mask_ex = (ex_taus[None, :] > 0.0) & (ex_taus[None, :] <= t_eff_365[:, None] + 1e-12)
        d_pv_grid = (mask_ex.to(torch.float64) * ex_pvs[None, :]).sum(dim=1)
        tau0_g = tau_knots[k_idx]
        tau1_g = tau_knots[k_idx + 1]
        mask_ex0 = (ex_taus[None, :] > 0.0) & (ex_taus[None, :] <= tau0_g[:, None] + 1e-12)
        d_pv0 = (mask_ex0.to(torch.float64) * ex_pvs[None, :]).sum(dim=1)
        esc0 = torch.clamp(spots_t0_i - d_pv0, min=0.01 * spots_t0_i)
        mask_ex1 = (ex_taus[None, :] > 0.0) & (ex_taus[None, :] <= tau1_g[:, None] + 1e-12)
        d_pv1 = (mask_ex1.to(torch.float64) * ex_pvs[None, :]).sum(dim=1)
        esc1 = torch.clamp(spots_t0_i - d_pv1, min=0.01 * spots_t0_i)
        q_div_grid = -torch.log(torch.clamp(esc1 / torch.clamp(esc0, min=1e-12), min=1e-12)) / dt_k
        q_repo_grid = q_repo_grid + q_div_grid
    else:
        d_pv_grid = torch.zeros(n_t, dtype=torch.float64, device=device)

    escrowed_grid = torch.clamp(spots_t0_i - d_pv_grid, min=0.01 * spots_t0_i)
    f_grid = torch.clamp(escrowed_grid * torch.exp(r_tot_grid), min=1e-4)  # (N_t,)

    half_w = min(max(5.0 * float(v_term_i), 0.50), 6.0)
    y_min_val = -half_w
    inv_dy_val = float(n_y - 1) / (2.0 * half_w)
    y_grid = torch.linspace(-half_w, half_w, n_y, dtype=torch.float64, device=device)
    s_grid = torch.clamp(f_grid[:, None] * torch.exp(y_grid[None, :]), min=1e-4)  # (N_t, N_y)

    # Map Act365 t_eff_365 to surface DCC coordinate and local variance-time ratio
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
        t_surf = tdcc_0[m_i] + slope_dcc * (t_eff_365 - t365_0[m_i])
        dt_surf_ratio = torch.clamp(slope_dcc, min=1e-12)

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

        # Handle t_surf <= tenors[0] and t_surf >= tenors[-1] cleanly
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


def _evaluate_monitoring_index_gpu(
    perf: torch.Tensor,
    index_name: str,
    inv_lex_perm_t: Optional[torch.Tensor] = None,
    *,
    asset_dim: int = -1,
) -> torch.Tensor:
    """Evaluates any of the 14 Murex monitoring functionals along `asset_dim` on GPU."""
    idx = str(index_name).strip()
    if idx in ("RatioAB", "RatioBA", "DiffAB", "DiffBA", "Outperformance", "Spread") and inv_lex_perm_t is not None:
        p = perf.index_select(asset_dim, inv_lex_perm_t)
    else:
        p = perf

    d_len = int(p.shape[asset_dim])
    if idx in ("WorstOf", "Lookback"):
        return p.min(dim=asset_dim).values
    if idx == "BestOf":
        return p.max(dim=asset_dim).values
    if idx in ("Average", "AverageOf", "MultiPerf"):
        return p.mean(dim=asset_dim)
    if idx == "Dispersion":
        mean_p = p.mean(dim=asset_dim, keepdim=True)
        return torch.sqrt(((p - mean_p) ** 2).mean(dim=asset_dim).clamp(min=0.0))
    if idx == "RatioBestWorst":
        return p.max(dim=asset_dim).values / p.min(dim=asset_dim).values.clamp(min=1e-12)
    if idx == "RatioBestAverage":
        return p.max(dim=asset_dim).values / p.mean(dim=asset_dim).clamp(min=1e-12)
    if idx == "RatioAverageWorst":
        return p.mean(dim=asset_dim) / p.min(dim=asset_dim).values.clamp(min=1e-12)
    if idx in ("DiffBestWorst", "Rainbow"):
        return p.max(dim=asset_dim).values - p.min(dim=asset_dim).values
    if idx == "DiffBestAverage":
        return p.max(dim=asset_dim).values - p.mean(dim=asset_dim)
    if idx == "DiffAverageWorst":
        return p.mean(dim=asset_dim) - p.min(dim=asset_dim).values
    if idx in ("RatioAB", "Outperformance"):
        p0 = p.select(asset_dim, 0)
        if d_len < 2:
            return p0
        return p0 / p.select(asset_dim, 1).clamp(min=1e-12)
    if idx == "RatioBA":
        p0 = p.select(asset_dim, 0)
        if d_len < 2:
            return p0
        return p.select(asset_dim, 1) / p0.clamp(min=1e-12)
    if idx in ("DiffAB", "Spread"):
        p0 = p.select(asset_dim, 0)
        if d_len < 2:
            return p0
        return p0 - p.select(asset_dim, 1)
    if idx == "DiffBA":
        p0 = p.select(asset_dim, 0)
        if d_len < 2:
            return p0
        return p.select(asset_dim, 1) - p0
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
    se = float(v_cv_r.std(unbiased=True).item() / math.sqrt(float(r_count)))
    return est, se


def _black76_put_exact_f64(fwd: float, strike: float, vol_sqrt_t: float) -> float:
    """Exact Black-76 undiscounted European put price E[(K - P)^+]."""
    if strike <= 0.0:
        return 0.0
    if vol_sqrt_t <= 1e-12:
        return max(strike - fwd, 0.0)
    d1 = (math.log(max(fwd, 1e-12) / strike) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))


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
        cfg = config or DupireQMCConfig()
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
        )

    def simulate_batch_for_training(
        self,
        states: Sequence[AutocallPreprocState],
        config: Optional[DupireQMCConfig] = None,
        compute_pathwise_delta: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cfg = config or DupireQMCConfig()
        b_size = len(states)
        if b_size == 0:
            z0 = torch.empty(0, dtype=torch.float64)
            return z0, torch.empty((0, 48), dtype=torch.float64), torch.empty((0, 48), dtype=torch.float64), z0, z0.unsqueeze(-1) if compute_pathwise_delta else None
        d_max = max(int(s.d_max) for s in states)
        m_max = max(int(s.m_max) for s in states)
        pv_out = torch.zeros(b_size, dtype=torch.float64)
        p_out = torch.zeros((b_size, m_max), dtype=torch.float64)
        q_out = torch.zeros((b_size, m_max), dtype=torch.float64)
        l_out = torch.zeros(b_size, dtype=torch.float64)
        delta_out = torch.zeros((b_size, d_max), dtype=torch.float64) if compute_pathwise_delta else None

        for b, st in enumerate(states):
            res = self.simulate_and_price(st, cfg)
            pv_out[b] = res.product_value
            p_out[b, : st.m_rem] = torch.as_tensor(res.p_stop, dtype=torch.float64)
            q_out[b, : st.m_rem] = torch.as_tensor(res.q_cpn, dtype=torch.float64)
            l_out[b] = res.l_put
            if delta_out is not None:
                for i in range(st.d_assets):
                    s0_i = float(st.spots_t0[i].item())
                    # Use baseline delta proxy scaled if full FD unavailable on raw state
                    delta_out[b, i] = float(st.baseline.proxy_delta_weights[i].item()) * s0_i
        return pv_out, p_out, q_out, l_out, delta_out


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
        cfg = config or DupireQMCConfig()
        dev = self.device
        d_assets = int(state.d_assets)
        m_rem = int(state.m_rem)

        if m_rem == 0 or state.seasoned_state.already_autocalled:
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

        spots_np = state.spots_t0.detach().cpu().numpy().astype(np.float64)
        ref_np = state.ref_spots.detach().cpu().numpy().astype(np.float64)
        if not (np.all(np.isfinite(spots_np)) and np.all(spots_np > 0.0)):
            raise ValueError("Spots must be strictly positive and finite.")
        if not (np.all(np.isfinite(ref_np)) and np.all(ref_np > 0.0)):
            raise ValueError("UnderlyingReferenceSpots must be strictly positive and finite.")

        eff_r = max(1, int(cfg.n_replicates))
        n_rep = _next_power_of_two(max(2, int(cfg.n_paths) // eff_r))
        n_total_paths = eff_r * n_rep
        n_sub = max(1, int(cfg.n_substeps))

        corr_np = state.corr_matrix.detach().cpu().numpy().astype(np.float64)
        corr_sym = 0.5 * (corr_np + corr_np.T)
        eigvals, eigvecs = np.linalg.eigh(corr_sym)
        if np.min(eigvals) < 1e-8:
            eigvals = np.maximum(eigvals, 1e-6)
            corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            d_inv = 1.0 / np.sqrt(np.diag(corr_psd))
            corr_sym = corr_psd * np.outer(d_inv, d_inv)
        chol_np = np.linalg.cholesky(corr_sym)
        chol_f32 = torch.as_tensor(chol_np, dtype=torch.float32, device=dev)

        tau_obs_np = state.tau_obs.detach().cpu().numpy().astype(np.float64)
        df_pay_np = state.df_pay.detach().cpu().numpy().astype(np.float64)
        ko_barriers_np = state.ko_barriers.detach().cpu().numpy().astype(np.float64)
        is_callable_np = state.is_callable.detach().cpu().numpy().astype(np.float64)
        cpn_barriers_np = state.cpn_barriers.detach().cpu().numpy().astype(np.float64)
        is_guar_np = state.is_guaranteed_cpn.detach().cpu().numpy().astype(np.float64)
        coupon_amounts_np = state.coupon_amounts.detach().cpu().numpy().astype(np.float64)
        fwd_prices_np = state.dividend_schedule.forward_price.detach().cpu().numpy().astype(np.float64)
        iv_at_fwd_np = state.iv_at_fwd.detach().cpu().numpy().astype(np.float64)
        v_perf_np = state.v_perf.detach().cpu().numpy().astype(np.float64)

        ref_spots_gpu = torch.as_tensor(ref_np, dtype=torch.float64, device=dev)
        fwd_prices_gpu = torch.as_tensor(fwd_prices_np, dtype=torch.float64, device=dev)
        w_fund_np = (
            _compute_funding_weights_np(state)
            if getattr(state, "funding_leg", None) is not None
            else np.zeros(m_rem, dtype=np.float64)
        )

        inv_perm_raw = getattr(state, "inv_lex_perm", None)
        inv_perm_gpu = (
            torch.as_tensor(inv_perm_raw, dtype=torch.long, device=dev)
            if inv_perm_raw is not None
            else None
        )

        # Build GPU 2D Dupire LV table (D, N_t, N_y) when mode == "dupire_lv"
        lv_table: Optional[torch.Tensor] = None
        y_min_gpu: Optional[torch.Tensor] = None
        inv_dy_gpu: Optional[torch.Tensor] = None
        tau_max = max(float(tau_obs_np[-1]), 1e-4)
        inv_dt_grid = float(cfg.lv_grid_nt - 1) / tau_max

        surfs_ordered = [state.vol_surfaces[t] for t in state.sorted_identifiers]
        div_events_map = getattr(state.dividend_schedule, "dividend_events", {})

        if cfg.mode == "dupire_lv":
            lv_table, y_min_gpu, inv_dy_gpu = _build_lv_tables_batch(
                surfs_ordered,
                state.spots_t0.to(device=dev, dtype=torch.float64),
                _resolve_r_native_obs_tensor(state).to(device=dev, dtype=torch.float64),
                state.dividend_schedule.df_fin_obs.to(device=dev, dtype=torch.float64),
                state.tau_obs.to(device=dev, dtype=torch.float64),
                [div_events_map.get(ticker, []) for ticker in state.sorted_identifiers],
                state.v_perf[:, -1].to(device=dev, dtype=torch.float64),
                cfg.lv_grid_nt,
                cfg.lv_grid_ny,
                dev,
                use_triton=self._use_triton,
            )

        # Dominant asset i* for bs1d_exact control variate
        try:
            dw = state.baseline.proxy_delta_weights.detach().cpu().numpy()
            dom_idx = int(np.argmax(np.abs(dw))) if np.max(np.abs(dw)) > 0 else 0
        except Exception:
            dom_idx = 0

        # State buffers on GPU (SoA layout)
        step_dtype = torch.float64 if cfg.precision == "fp64_audit" else torch.float32
        log_x = torch.zeros((eff_r, d_assets, n_rep), dtype=step_dtype, device=dev)
        alive = torch.ones((eff_r, n_rep), dtype=torch.float64, device=dev)
        unpaid_mem = torch.full((eff_r, n_rep), float(state.seasoned_state.c_mem_0), dtype=torch.float64, device=dev)
        total_pv_paths = torch.zeros((eff_r, n_rep), dtype=torch.float64, device=dev)

        # Control variate tracking buffers
        z_flat_cum = torch.zeros((eff_r, n_rep), dtype=torch.float64, device=dev)
        v_dom_term = float(v_perf_np[dom_idx, -1])
        tau_term = max(float(tau_obs_np[-1]), 1e-8)

        # Multifidelity GBM paired paths when cv_mode == "multifidelity_gbm"
        log_x_gbm: Optional[torch.Tensor] = None
        alive_gbm: Optional[torch.Tensor] = None
        unpaid_gbm: Optional[torch.Tensor] = None
        pv_gbm_fine: Optional[torch.Tensor] = None
        var_prev_gbm = np.zeros(d_assets, dtype=np.float64)
        if cfg.cv_mode == "multifidelity_gbm":
            log_x_gbm = torch.zeros((eff_r, d_assets, n_rep), dtype=torch.float64, device=dev)
            alive_gbm = torch.ones((eff_r, n_rep), dtype=torch.float64, device=dev)
            unpaid_gbm = torch.full((eff_r, n_rep), float(state.seasoned_state.c_mem_0), dtype=torch.float64, device=dev)
            pv_gbm_fine = torch.zeros((eff_r, n_rep), dtype=torch.float64, device=dev)

        p_stop_gpu = torch.zeros(m_rem, dtype=torch.float64, device=dev)
        cpn_alive_gpu = torch.zeros(m_rem, dtype=torch.float64, device=dev)
        surv_before_gpu = torch.zeros(m_rem, dtype=torch.float64, device=dev)
        cpn_cf_gpu = torch.zeros(m_rem, dtype=torch.float64, device=dev)

        eff_cpn_idx = state.coupon_index if state.coupon_index else state.callability_index
        eff_opt_idx = (
            "AverageOf"
            if bool(getattr(state, "is_long_basket", False))
            else (state.option_index if state.option_index else state.callability_index)
        )
        cpn_dir = str(getattr(state, "coupon_barrier_direction", "Up") or "Up").strip().lower()
        cpn_type = str(getattr(state, "coupon_barrier_type", "In") or "In").strip().lower()

        var_prev_iv = np.zeros(d_assets, dtype=np.float64)
        t_prev = 0.0
        coarse_dim_base = 0
        sub_dim_base = m_rem * d_assets

        for m in range(m_rem):
            t_curr = float(tau_obs_np[m])
            dt_period = max(t_curr - t_prev, 1e-8)

            # Generate coarse Sobol tile (R, D, N_rep) from lowest Sobol dimensions
            z_coarse_indep = _generate_gpu_sobol_tile(
                n_rep=n_rep,
                n_replicates=eff_r,
                n_dims=d_assets,
                dim_offset=coarse_dim_base + m * d_assets,
                seed=cfg.seed,
                device=dev,
                use_triton=self._use_triton,
            )  # (R, D, N_rep) float32
            z_coarse_corr = torch.einsum("ij,rjn->rin", chol_f32, z_coarse_indep)

            if cfg.cv_mode == "bs1d_exact":
                dvar_dom_m = (v_dom_term * v_dom_term) * (dt_period / tau_term)
                z_flat_cum = z_flat_cum + math.sqrt(max(dvar_dom_m, 0.0)) * z_coarse_corr[:, dom_idx, :].to(torch.float64)

            if cfg.mode == "terminal_iv":
                var_curr = (iv_at_fwd_np[:, m] ** 2) * tau_obs_np[m]
                d_var = np.maximum(var_curr - var_prev_iv, 1e-12)
                var_prev_iv = var_prev_iv + d_var
                dv_t = torch.as_tensor(d_var, dtype=torch.float64, device=dev).view(1, d_assets, 1)
                log_x_f64 = log_x.to(torch.float64) - 0.5 * dv_t + torch.sqrt(dv_t) * z_coarse_corr.to(torch.float64)
                log_x = log_x_f64.to(step_dtype)
            else:
                if n_sub == 1:
                    z_steps_corr = z_coarse_corr.unsqueeze(2)  # (R, D, 1, N_rep)
                else:
                    z_sub_raw = _generate_gpu_sobol_tile(
                        n_rep=n_rep,
                        n_replicates=eff_r,
                        n_dims=d_assets * n_sub,
                        dim_offset=sub_dim_base + m * (d_assets * n_sub),
                        seed=cfg.seed + 31337,
                        device=dev,
                        use_triton=self._use_triton,
                    ).view(eff_r, d_assets, n_sub, n_rep)
                    z_sub_centered = z_sub_raw - z_sub_raw.mean(dim=2, keepdim=True) + z_coarse_indep.unsqueeze(2) / math.sqrt(float(n_sub))
                    z_steps_corr = torch.einsum("ij,rjsn->risn", chol_f32, z_sub_centered).contiguous()

                dt_sub = dt_period / float(n_sub)
                sqrt_dt_sub = math.sqrt(dt_sub)
                assert lv_table is not None and y_min_gpu is not None and inv_dy_gpu is not None

                if self._use_triton and cfg.precision != "fp64_audit":
                    block_n = min(256, n_rep)
                    grid_sub = (triton.cdiv(n_rep, block_n), d_assets, eff_r)
                    _dupire_substep_triton_kernel[grid_sub](
                        log_x,
                        z_steps_corr,
                        lv_table,
                        y_min_gpu,
                        inv_dy_gpu,
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
                    lx_f = log_x.to(torch.float32)
                    d_idx = torch.arange(d_assets, device=dev).view(1, d_assets, 1)
                    ym_v = y_min_gpu.view(1, d_assets, 1)
                    idy_v = inv_dy_gpu.view(1, d_assets, 1)
                    for s in range(n_sub):
                        t_step = t_prev + (s + 0.5) * dt_sub
                        ut = min(max(t_step * inv_dt_grid, 0.0), max_t_f)
                        it0 = int(ut)
                        ft = ut - float(it0)
                        it1 = min(it0 + 1, cfg.lv_grid_nt - 1)
                        uy = torch.clamp((lx_f - ym_v) * idy_v, 0.0, max_y_f)
                        iy0 = uy.to(torch.long)
                        fy = uy - iy0.to(torch.float32)
                        iy1 = torch.clamp(iy0 + 1, max=cfg.lv_grid_ny - 1)
                        v00 = lv_table[d_idx, it0, iy0]
                        v01 = lv_table[d_idx, it0, iy1]
                        v10 = lv_table[d_idx, it1, iy0]
                        v11 = lv_table[d_idx, it1, iy1]
                        lv_s = (1.0 - ft) * (v00 + fy * (v01 - v00)) + ft * (v10 + fy * (v11 - v10))
                        z_s = z_steps_corr[:, :, s, :]
                        if cfg.precision == "fp64_audit":
                            log_x = log_x + (-0.5 * lv_s.to(torch.float64) ** 2 * dt_sub + lv_s.to(torch.float64) * sqrt_dt_sub * z_s.to(torch.float64))
                            lx_f = log_x.to(torch.float32)
                        else:
                            lx_f = lx_f + (-0.5 * lv_s * lv_s * dt_sub + lv_s * sqrt_dt_sub * z_s)
                    if cfg.precision != "fp64_audit":
                        log_x = lx_f

            # Strict float64 martingale renormalization per replicate
            log_x_f64 = log_x.to(torch.float64)
            mean_x = torch.exp(log_x_f64).mean(dim=-1, keepdim=True)
            log_shift = torch.log(torch.clamp(mean_x, min=1e-12))
            log_x_f64 = log_x_f64 - log_shift
            log_x = log_x_f64.to(step_dtype)

            s_m = fwd_prices_gpu[:, m].view(1, d_assets, 1) * torch.exp(log_x_f64)
            perf_m = (s_m / ref_spots_gpu.view(1, d_assets, 1)).permute(0, 2, 1)  # (R, N_rep, D)

            w_call = _evaluate_monitoring_index_gpu(perf_m, state.callability_index, inv_perm_gpu)
            w_cpn = _evaluate_monitoring_index_gpu(perf_m, eff_cpn_idx, inv_perm_gpu)

            if is_callable_np[m] > 0.5:
                ko_ind = _digital_ramp_torch(w_call, float(ko_barriers_np[m]), float(state.ko_spread))
            else:
                ko_ind = torch.zeros((eff_r, n_rep), dtype=torch.float64, device=dev)

            if is_guar_np[m] > 0.5:
                cpn_ind = torch.ones((eff_r, n_rep), dtype=torch.float64, device=dev)
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
                total_pv_paths = total_pv_paths - alive * float(w_fund_np[m])

            stop_m = alive * ko_ind
            p_stop_gpu[m] = stop_m.mean()
            surv_before_gpu[m] = alive.mean()
            cpn_alive_gpu[m] = (alive * cpn_ind).mean()

            c_m = float(coupon_amounts_np[m])
            eff_c_paths = (c_m + unpaid_mem) if state.is_memory else c_m
            cpn_undisc_m = alive * cpn_ind * eff_c_paths + stop_m * float(state.autocall_coupon)
            cpn_cf_gpu[m] = cpn_undisc_m.mean()
            total_pv_paths = total_pv_paths + float(df_pay_np[m]) * cpn_undisc_m

            if state.is_memory:
                unpaid_mem = (unpaid_mem + c_m) * (1.0 - cpn_ind)

            alive = alive * (1.0 - ko_ind)
            if not state.no_notional_repayment:
                if m < m_rem - 1:
                    total_pv_paths = total_pv_paths + float(df_pay_np[m]) * stop_m
                else:
                    total_pv_paths = total_pv_paths + float(df_pay_np[m]) * (stop_m + alive)

            # Paired fine flat-ATM GBM paths for multifidelity_gbm CV
            if cfg.cv_mode == "multifidelity_gbm" and log_x_gbm is not None and alive_gbm is not None and unpaid_gbm is not None and pv_gbm_fine is not None:
                var_c_g = (iv_at_fwd_np[:, m] ** 2) * tau_obs_np[m]
                dv_g = np.maximum(var_c_g - var_prev_gbm, 1e-12)
                var_prev_gbm = var_prev_gbm + dv_g
                dv_gt = torch.as_tensor(dv_g, dtype=torch.float64, device=dev).view(1, d_assets, 1)
                log_x_gbm = log_x_gbm - 0.5 * dv_gt + torch.sqrt(dv_gt) * z_coarse_corr.to(torch.float64)
                s_mg = fwd_prices_gpu[:, m].view(1, d_assets, 1) * torch.exp(log_x_gbm)
                perf_mg = (s_mg / ref_spots_gpu.view(1, d_assets, 1)).permute(0, 2, 1)
                wc_g = _evaluate_monitoring_index_gpu(perf_mg, state.callability_index, inv_perm_gpu)
                wq_g = _evaluate_monitoring_index_gpu(perf_mg, eff_cpn_idx, inv_perm_gpu)
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
                    unpaid_gbm = (unpaid_gbm + c_m) * (1.0 - cp_g)
                alive_gbm = alive_gbm * (1.0 - ko_g)
                if not state.no_notional_repayment:
                    pv_gbm_fine = pv_gbm_fine + float(df_pay_np[m]) * (st_g if m < m_rem - 1 else (st_g + alive_gbm))
                if m == m_rem - 1:
                    perf_tg = perf_mg

            if m == m_rem - 1:
                perf_term = perf_m

            t_prev = t_curr

        stats_cpu = torch.stack(
            [p_stop_gpu, cpn_alive_gpu, surv_before_gpu, cpn_cf_gpu], dim=0
        ).cpu().numpy()
        p_stop_arr = stats_cpu[0]
        surv_before_arr = stats_cpu[2]
        q_cpn_arr = np.where(
            surv_before_arr > 1e-12,
            stats_cpu[1] / np.maximum(surv_before_arr, 1e-12),
            0.0,
        )
        cpn_cf_expected = stats_cpu[3]

        # Terminal option payoff on GPU (strict float64)
        w_opt = _evaluate_monitoring_index_gpu(perf_term, eff_opt_idx, inv_perm_gpu)
        star_ind: Optional[torch.Tensor] = None
        if state.one_star_strike is not None and float(state.one_star_strike) > 0.0:
            w_best = _evaluate_monitoring_index_gpu(perf_term, "BestOf", inv_perm_gpu)
            star_ind = (w_best >= float(state.one_star_strike)).to(torch.float64)

        if state.strike > 1e-6 and state.option_gearing > 0.0:
            put_int = torch.clamp(float(state.strike) - w_opt, min=0.0)
            if state.lower_strike is not None and float(state.lower_strike) > 0.0:
                put_int = torch.clamp(put_int, max=max(float(state.strike) - float(state.lower_strike), 0.0))
            if state.ki_barrier is not None and float(state.ki_barrier) > 0.0 and state.seasoned_state.i_ki_past < 0.5:
                ki_ind = 1.0 - _digital_ramp_torch(w_opt, float(state.ki_barrier), float(state.ki_spread))
                put_int = put_int * ki_ind
            if star_ind is not None:
                put_int = put_int * (1.0 - star_ind)
            unit_put_paths = alive * put_int
        else:
            unit_put_paths = torch.zeros((eff_r, n_rep), dtype=torch.float64, device=dev)

        if state.participation_call_strike is not None and float(state.participation_call_rate) > 0.0:
            unit_call_paths = alive * torch.clamp(w_opt - float(state.participation_call_strike), min=0.0)
        else:
            unit_call_paths = torch.zeros((eff_r, n_rep), dtype=torch.float64, device=dev)

        l_put_val = float(unit_put_paths.mean().item())
        l_call_val = float(unit_call_paths.mean().item())
        l_onestar_val = 0.0
        p_onestar_val = float((alive * star_ind).mean().item()) if star_ind is not None else 0.0

        put_sign = -1.0
        opt_payoff_paths = float(df_pay_np[-1]) * (
            put_sign * float(state.option_gearing) * unit_put_paths
            + float(state.participation_call_rate) * unit_call_paths
        )
        if star_ind is not None and float(state.one_star_coupon) > 0.0:
            onestar_paths = alive * star_ind * float(state.one_star_coupon)
            opt_payoff_paths = opt_payoff_paths + float(df_pay_np[-1]) * onestar_paths
            l_onestar_val = float(onestar_paths.mean().item())

        total_pv_paths = total_pv_paths + opt_payoff_paths

        # Apply strictly unbiased Control Variate (CHM-001)
        raw_pv_mean = float(total_pv_paths.mean().item())
        if cfg.cv_mode == "bs1d_exact":
            f_dom = float(fwd_prices_np[dom_idx, -1]) / float(ref_np[dom_idx])
            k_cv = float(state.strike) if state.strike > 1e-6 else 1.0
            p_dom = f_dom * torch.exp(-0.5 * (v_dom_term * v_dom_term) + z_flat_cum)
            x_cv = float(df_pay_np[-1]) * torch.clamp(k_cv - p_dom, min=0.0)
            mu_cv = float(df_pay_np[-1]) * _black76_put_exact_f64(f_dom, k_cv, v_dom_term)
            if state.lower_strike is not None and float(state.lower_strike) > 0.0 and state.strike > 1e-6:
                k_low = float(state.lower_strike)
                x_cv = x_cv - float(df_pay_np[-1]) * torch.clamp(k_low - p_dom, min=0.0)
                mu_cv = mu_cv - float(df_pay_np[-1]) * _black76_put_exact_f64(f_dom, k_low, v_dom_term)
            cv_pv_est, total_se = _apply_loo_control_variate(total_pv_paths, x_cv, mu_cv)
        elif cfg.cv_mode == "multifidelity_gbm" and pv_gbm_fine is not None and alive_gbm is not None:
            # Terminal option on paired fine GBM paths
            w_og = _evaluate_monitoring_index_gpu(perf_tg, eff_opt_idx, inv_perm_gpu)
            st_g_term: Optional[torch.Tensor] = None
            if state.one_star_strike is not None and float(state.one_star_strike) > 0.0:
                st_g_term = (_evaluate_monitoring_index_gpu(perf_tg, "BestOf", inv_perm_gpu) >= float(state.one_star_strike)).to(torch.float64)
            if state.strike > 1e-6 and state.option_gearing > 0.0:
                pi_g = torch.clamp(float(state.strike) - w_og, min=0.0)
                if state.lower_strike is not None and float(state.lower_strike) > 0.0:
                    pi_g = torch.clamp(pi_g, max=max(float(state.strike) - float(state.lower_strike), 0.0))
                if state.ki_barrier is not None and float(state.ki_barrier) > 0.0 and state.seasoned_state.i_ki_past < 0.5:
                    pi_g = pi_g * (1.0 - _digital_ramp_torch(w_og, float(state.ki_barrier), float(state.ki_spread)))
                if st_g_term is not None:
                    pi_g = pi_g * (1.0 - st_g_term)
                up_g = alive_gbm * pi_g
            else:
                up_g = torch.zeros_like(alive_gbm)
            uc_g = alive_gbm * torch.clamp(w_og - float(state.participation_call_strike), min=0.0) if (state.participation_call_strike is not None and float(state.participation_call_rate) > 0.0) else torch.zeros_like(alive_gbm)
            pv_gbm_fine = pv_gbm_fine + float(df_pay_np[-1]) * (put_sign * float(state.option_gearing) * up_g + float(state.participation_call_rate) * uc_g)
            if st_g_term is not None and float(state.one_star_coupon) > 0.0:
                pv_gbm_fine = pv_gbm_fine + float(df_pay_np[-1]) * alive_gbm * st_g_term * float(state.one_star_coupon)

            # Independent M_cheap = 4 * N flat-GBM expectation using GPU Philox antithetic draws
            mu_mf = self._eval_cheap_gbm_expectation(
                state=state,
                chol_f32=chol_f32,
                tau_obs_np=tau_obs_np,
                df_pay_np=df_pay_np,
                ko_barriers_np=ko_barriers_np,
                is_callable_np=is_callable_np,
                cpn_barriers_np=cpn_barriers_np,
                is_guar_np=is_guar_np,
                coupon_amounts_np=coupon_amounts_np,
                fwd_prices_gpu=fwd_prices_gpu,
                iv_at_fwd_np=iv_at_fwd_np,
                ref_spots_gpu=ref_spots_gpu,
                w_fund_np=w_fund_np,
                eff_cpn_idx=eff_cpn_idx,
                eff_opt_idx=eff_opt_idx,
                cpn_dir=cpn_dir,
                cpn_type=cpn_type,
                inv_perm_gpu=inv_perm_gpu,
                n_cheap=4 * n_total_paths,
                seed=cfg.seed + 88888,
            )
            cv_pv_est, total_se = _apply_loo_control_variate(total_pv_paths, pv_gbm_fine, mu_mf)
        else:
            cv_pv_est = raw_pv_mean
            if eff_r > 1:
                rep_pv = total_pv_paths.mean(dim=1)
                total_se = float(rep_pv.std(unbiased=True).item() / math.sqrt(float(eff_r)))
            else:
                total_se = float(total_pv_paths.reshape(-1).std(unbiased=True).item() / math.sqrt(float(n_total_paths)))

        if eff_r > 1:
            rep_opt = opt_payoff_paths.mean(dim=1)
            option_se = float(rep_opt.std(unbiased=True).item() / math.sqrt(float(eff_r)))
        else:
            option_se = float(opt_payoff_paths.reshape(-1).std(unbiased=True).item() / math.sqrt(float(n_total_paths)))

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

        # Preserve exact CV PV delta inside option_pv + product_value while keeping CashFlows reconciled
        cv_shift = cv_pv_est - raw_pv_mean
        adj_prod = float(asm.product_value.item()) + cv_shift
        adj_opt = float(asm.option_pv.item()) + cv_shift
        if cv_shift != 0.0 and asm.cashflows:
            target_desc = (
                "IsLongBasket"
                if (
                    bool(getattr(state, "is_long_basket", False))
                    and any(
                        cf.Description == "IsLongBasket" and cf.StatusText == "Estimate"
                        for cf in asm.cashflows
                    )
                )
                else "Option"
            )
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
                    break

        return DupireQMCResult(
            product_value=adj_prod,
            standard_error=total_se,
            option_pv=adj_opt,
            option_se=option_se,
            notional_pv=float(asm.notional_pv.item()),
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

    def _eval_cheap_gbm_expectation(
        self,
        *,
        state: AutocallPreprocState,
        chol_f32: torch.Tensor,
        tau_obs_np: npt.NDArray[np.float64],
        df_pay_np: npt.NDArray[np.float64],
        ko_barriers_np: npt.NDArray[np.float64],
        is_callable_np: npt.NDArray[np.float64],
        cpn_barriers_np: npt.NDArray[np.float64],
        is_guar_np: npt.NDArray[np.float64],
        coupon_amounts_np: npt.NDArray[np.float64],
        fwd_prices_gpu: torch.Tensor,
        iv_at_fwd_np: npt.NDArray[np.float64],
        ref_spots_gpu: torch.Tensor,
        w_fund_np: npt.NDArray[np.float64],
        eff_cpn_idx: str,
        eff_opt_idx: str,
        cpn_dir: str,
        cpn_type: str,
        inv_perm_gpu: Optional[torch.Tensor],
        n_cheap: int,
        seed: int,
    ) -> float:
        """Evaluates independent cheap M_cheap=4*N scrambled-Sobol flat-ATM GBM expectation on GPU."""
        dev = self.device
        d_assets = int(state.d_assets)
        m_rem = int(state.m_rem)
        n_rep_mf = _next_power_of_two(min(max(n_cheap // 4, 2048), 16384))
        r_mf = max(1, (n_cheap + n_rep_mf - 1) // n_rep_mf)
        chunk_sz = r_mf * n_rep_mf

        var_cum = (iv_at_fwd_np ** 2) * tau_obs_np[None, :]  # (D, M_rem)
        var_prev_mat = np.concatenate(
            [np.zeros((d_assets, 1), dtype=np.float64), var_cum[:, :-1]], axis=1
        )
        dv_mat = np.maximum(var_cum - var_prev_mat, 1e-12)  # (D, M_rem)
        # Ensure strict telescoping increment matches sequential loop
        var_run = np.zeros(d_assets, dtype=np.float64)
        for m in range(m_rem):
            dv_col = np.maximum(var_cum[:, m] - var_run, 1e-12)
            var_run = var_run + dv_col
            dv_mat[:, m] = dv_col

        dv_all_f32 = torch.tensor(dv_mat.T.copy(), dtype=torch.float32, device=dev)  # (M_rem, D)
        sqrt_dv_f32 = torch.sqrt(dv_all_f32)
        fwd_rel_32 = (fwd_prices_gpu / ref_spots_gpu[:, None]).T.to(torch.float32).contiguous()

        r_chunk = max(1, min(r_mf, (4 * 1024 * 1024) // max(m_rem * d_assets * n_rep_mf, 1)))
        pv_sum = 0.0
        for r0 in range(0, r_mf, r_chunk):
            rc = min(r_chunk, r_mf - r0)
            z_all = _generate_gpu_sobol_tile(
                n_rep=n_rep_mf,
                n_replicates=rc,
                n_dims=m_rem * d_assets,
                dim_offset=0,
                seed=seed + r0 * 997,
                device=dev,
                use_triton=self._use_triton,
            ).view(rc, m_rem, d_assets, n_rep_mf)
            z_c = torch.einsum("ij,rmjn->rmni", chol_f32, z_all)
            del z_all
            dlog_all = -0.5 * dv_all_f32.view(1, m_rem, 1, d_assets) + sqrt_dv_f32.view(1, m_rem, 1, d_assets) * z_c
            del z_c
            log_x_all = torch.cumsum(dlog_all, dim=1)
            del dlog_all
            perf_all = fwd_rel_32.view(1, m_rem, 1, d_assets) * torch.exp(log_x_all)
            del log_x_all

            w_call_all = _evaluate_monitoring_index_gpu(perf_all, state.callability_index, inv_perm_gpu)
            w_cpn_all = (
                w_call_all
                if eff_cpn_idx == state.callability_index
                else _evaluate_monitoring_index_gpu(perf_all, eff_cpn_idx, inv_perm_gpu)
            )
            perf_term = perf_all[:, -1, :, :]
            del perf_all

            ko_bar_t = torch.tensor(ko_barriers_np, dtype=torch.float32, device=dev).view(1, m_rem, 1)
            is_call_t = torch.tensor(is_callable_np, dtype=torch.float32, device=dev).view(1, m_rem, 1)
            ko_spr_val = float(state.ko_spread)
            if abs(ko_spr_val) < 1e-9:
                ko_all_t = (w_call_all >= ko_bar_t).to(torch.float32) * is_call_t
            else:
                b_l = torch.minimum(ko_bar_t, ko_bar_t + ko_spr_val)
                b_h = torch.maximum(ko_bar_t, ko_bar_t + ko_spr_val)
                ko_all_t = torch.clamp((w_call_all - b_l) / torch.clamp(b_h - b_l, min=1e-9), 0.0, 1.0) * is_call_t
            del w_call_all

            cpn_bar_t = torch.tensor(cpn_barriers_np, dtype=torch.float32, device=dev).view(1, m_rem, 1)
            is_guar_t = torch.tensor(is_guar_np, dtype=torch.float32, device=dev).view(1, m_rem, 1) > 0.5
            s_cpn_eff = -float(state.cpn_spread) if cpn_dir == "down" else float(state.cpn_spread)
            if abs(s_cpn_eff) < 1e-9:
                bc_all = (w_cpn_all >= cpn_bar_t).to(torch.float32)
            else:
                bl_c = torch.minimum(cpn_bar_t, cpn_bar_t + s_cpn_eff)
                bh_c = torch.maximum(cpn_bar_t, cpn_bar_t + s_cpn_eff)
                bc_all = torch.clamp((w_cpn_all - bl_c) / torch.clamp(bh_c - bl_c, min=1e-9), 0.0, 1.0)
            del w_cpn_all
            if cpn_dir == "down":
                bc_all = 1.0 - bc_all
            cp_all_t = (1.0 - bc_all) if cpn_type == "out" else bc_all
            cp_all_t = torch.where(is_guar_t, torch.ones_like(cp_all_t), cp_all_t)
            if state.force_coupons_on_autocall:
                cp_all_t = torch.maximum(cp_all_t, ko_all_t)

            alive = torch.ones((rc, n_rep_mf), dtype=torch.float32, device=dev)
            unpaid = torch.full(
                (rc, n_rep_mf), float(state.seasoned_state.c_mem_0), dtype=torch.float32, device=dev
            )
            pv = torch.zeros((rc, n_rep_mf), dtype=torch.float32, device=dev)
            for m in range(m_rem):
                ko_i = ko_all_t[:, m, :]
                cp_i = cp_all_t[:, m, :]
                if w_fund_np[m] != 0.0:
                    pv = pv - alive * float(w_fund_np[m])
                st_m = alive * ko_i
                c_m = float(coupon_amounts_np[m])
                ec = (c_m + unpaid) if state.is_memory else c_m
                pv = pv + float(df_pay_np[m]) * (alive * cp_i * ec + st_m * float(state.autocall_coupon))
                if state.is_memory:
                    unpaid = (unpaid + c_m) * (1.0 - cp_i)
                alive = alive * (1.0 - ko_i)
                if not state.no_notional_repayment:
                    pv = pv + float(df_pay_np[m]) * (st_m if m < m_rem - 1 else (st_m + alive))
            del ko_all_t, cp_all_t
            opt_idx_eff = "AverageOf" if bool(getattr(state, "is_long_basket", False)) else eff_opt_idx
            w_opt = _evaluate_monitoring_index_gpu(perf_term, opt_idx_eff, inv_perm_gpu)
            st_term: Optional[torch.Tensor] = None
            if state.one_star_strike is not None and float(state.one_star_strike) > 0.0:
                st_term = (
                    _evaluate_monitoring_index_gpu(perf_term, "BestOf", inv_perm_gpu)
                    >= float(state.one_star_strike)
                ).to(torch.float32)
            if state.strike > 1e-6 and state.option_gearing > 0.0:
                pi = torch.clamp(float(state.strike) - w_opt, min=0.0)
                if state.lower_strike is not None and float(state.lower_strike) > 0.0:
                    pi = torch.clamp(pi, max=max(float(state.strike) - float(state.lower_strike), 0.0))
                if state.ki_barrier is not None and float(state.ki_barrier) > 0.0 and state.seasoned_state.i_ki_past < 0.5:
                    pi = pi * (1.0 - _digital_ramp_torch(w_opt, float(state.ki_barrier), float(state.ki_spread)))
                if st_term is not None:
                    pi = pi * (1.0 - st_term)
                up = alive * pi
            else:
                up = torch.zeros_like(alive)
            uc = (
                alive * torch.clamp(w_opt - float(state.participation_call_strike), min=0.0)
                if (state.participation_call_strike is not None and float(state.participation_call_rate) > 0.0)
                else torch.zeros_like(alive)
            )
            put_sign = -1.0
            pv = pv + float(df_pay_np[-1]) * (
                put_sign * float(state.option_gearing) * up + float(state.participation_call_rate) * uc
            )
            if st_term is not None and float(state.one_star_coupon) > 0.0:
                pv = pv + float(df_pay_np[-1]) * alive * st_term * float(state.one_star_coupon)
            pv_sum += float(pv.sum(dtype=torch.float64).item())

        return pv_sum / float(chunk_sz)

    def simulate_batch_for_training(
        self,
        states: Sequence[AutocallPreprocState],
        config: Optional[DupireQMCConfig] = None,
        compute_pathwise_delta: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Multi-contract (D, M_rem)-bucketed GPU QMC teacher with single-pass pathwise tangent Delta."""
        cfg = config or DupireQMCConfig(n_paths=8192, n_replicates=8, n_substeps=16, cv_mode="none")
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
                    pv_out[idx] = float(st.pv_in_transit)
                continue
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
        corr_stack_64 = torch.stack(
            [st.corr_matrix.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0
        ).to(dev)
        m2d_cpu = torch.stack(
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
        ).to(dev)  # (8, B_c, M)
        tau_t, df_pay_t, ko_t, is_call_t, cpn_t, is_guar_t, cpn_amt_t, w_fund_t = m2d_cpu.unbind(dim=0)

        m3d_cpu = torch.stack(
            [
                torch.stack([st.dividend_schedule.forward_price.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.cash_div_leverage.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([_resolve_r_native_obs_tensor(st).detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.dividend_schedule.df_fin_obs.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
            ],
            dim=0,
        ).to(dev)  # (4, B_c, D, M)
        fwd_t, cash_lev_t, r_tot_t, df_fin_t = m3d_cpu.unbind(dim=0)

        d1d_cpu = torch.stack(
            [
                torch.stack([st.ref_spots.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.spots_t0.detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
                torch.stack([st.v_perf[:, -1].detach().to(device="cpu", dtype=torch.float64) for st in st_chunk], dim=0),
            ],
            dim=0,
        ).to(dev)  # (3, B_c, D)
        ref_t, spots_t, v_term_t = d1d_cpu.unbind(dim=0)

        scal_np = np.empty((18, b_c), dtype=np.float64)
        surfs_flat: List[Any] = []
        div_events_flat: List[Sequence[Any]] = []
        for bi, st in enumerate(st_chunk):
            scal_np[0, bi] = float(st.seasoned_state.c_mem_0)
            scal_np[1, bi] = 1.0 if st.is_memory else 0.0
            scal_np[2, bi] = 1.0 if st.force_coupons_on_autocall else 0.0
            scal_np[3, bi] = 1.0 if st.no_notional_repayment else 0.0
            scal_np[4, bi] = float(st.autocall_coupon)
            scal_np[5, bi] = float(st.ko_spread)
            scal_np[6, bi] = float(st.cpn_spread)
            scal_np[7, bi] = float(st.pv_in_transit)
            scal_np[8, bi] = float(st.strike)
            scal_np[9, bi] = float(st.option_gearing)
            lk = st.lower_strike
            scal_np[10, bi] = float(lk) if (lk is not None and float(lk) > 0.0) else -1.0
            kb = st.ki_barrier
            scal_np[11, bi] = float(kb) if (kb is not None and float(kb) > 0.0 and st.seasoned_state.i_ki_past < 0.5) else -1.0
            scal_np[12, bi] = float(st.ki_spread)
            sk = st.one_star_strike
            scal_np[13, bi] = float(sk) if (sk is not None and float(sk) > 0.0) else -1.0
            scal_np[14, bi] = float(st.one_star_coupon)
            pk = st.participation_call_strike
            pr = st.participation_call_rate
            scal_np[15, bi] = float(pk) if (pk is not None and float(pr) > 0.0) else -1.0
            scal_np[16, bi] = float(pr)
            scal_np[17, bi] = -1.0
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
            e_opt = (
                "AverageOf"
                if bool(getattr(st, "is_long_basket", False))
                else str(st.option_index if st.option_index else st.callability_index)
            )
            opt_groups.setdefault((e_opt, p_key), []).append(bi)
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
                block_n = min(256, n_rep)
                grid_b = (triton.cdiv(n_rep, block_n), b_c * d_act, eff_r)
                _dupire_batch_substep_triton_kernel[grid_b](
                    log_x,
                    empty_f32_stub,
                    z_steps_indep,
                    chol_t,
                    lv_table_b,
                    ymin_t,
                    invdy_t,
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
                for s in range(n_sub):
                    t_step = t_start_t + (s + 0.5) * dt_sub_t
                    ut = torch.clamp(t_step * idg_v, 0.0, max_t_f)
                    it0 = ut.to(torch.long)
                    ft = ut - it0.to(torch.float32)
                    it1 = torch.clamp(it0 + 1, max=cfg.lv_grid_nt - 1)
                    uy = torch.clamp((log_x - ym_v) * idy_v, 0.0, max_y_f)
                    iy0 = uy.to(torch.long)
                    fy = uy - iy0.to(torch.float32)
                    iy1 = torch.clamp(iy0 + 1, max=cfg.lv_grid_ny - 1)
                    v00 = lv_table_b[b_idx, d_idx, it0, iy0]
                    v01 = lv_table_b[b_idx, d_idx, it0, iy1]
                    v10 = lv_table_b[b_idx, d_idx, it1, iy0]
                    v11 = lv_table_b[b_idx, d_idx, it1, iy1]
                    lv_s = (1.0 - ft) * (v00 + fy * (v01 - v00)) + ft * (v10 + fy * (v11 - v10))
                    z_s = z_steps_corr[:, :, :, s, :]
                    log_x = log_x + (-0.5 * lv_s * lv_s * dt_sub_t + lv_s * sqrt_dt_t * z_s)

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
                # Sticky-moneyness Dupire LV tangent avoids unbounded (1 + dlx) Euler amplification
                dperf_diag: Optional[torch.Tensor] = perf_m * cash_lev_t[:, :, m].unsqueeze(-1)
            else:
                dperf_diag = None

            # Vectorized grouped monitoring indices & derivatives in coalesced SoA (B_c, D, N_tot)
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
                block_step = min(256, n_tot)
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
                _training_period_step_triton_kernel[grid_step](
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
                    n_tot=n_tot,
                    d_assets=d_act,
                    is_last=1 if m == m_act - 1 else 0,
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
                        d_unpaid_next = d_eff_c * (1.0 - cpn_ind_eff).unsqueeze(1) - eff_c.unsqueeze(1) * d_cpn
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
                unpaid_mem = torch.where(is_mem_t > 0.5, eff_c * (1.0 - cpn_ind_eff), unpaid_mem)
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
        has_put_b = (strike_b > 1e-6) & (gearing_b > 0.0)
        low_k_b = scal_gpu[10].view(b_c, 1)
        has_low_b = low_k_b > 0.0
        ki_bar_b = scal_gpu[11].view(b_c, 1)
        ki_spr_b = scal_gpu[12].view(b_c, 1)
        has_ki_b = ki_bar_b > 0.0
        star_k_b = scal_gpu[13].view(b_c, 1)
        star_cpn_b = scal_gpu[14].view(b_c, 1)
        has_star_b = star_k_b > 0.0
        pcall_k_b = scal_gpu[15].view(b_c, 1)
        pcall_rate_b = scal_gpu[16].view(b_c, 1)
        has_pcall_b = pcall_k_b >= 0.0
        put_sign_b = scal_gpu[17].view(b_c, 1)

        has_any_low = bool(np.any(scal_np[10] > 0.0))
        has_any_ki = bool(np.any(scal_np[11] > 0.0))
        has_any_star = bool(np.any(scal_np[13] > 0.0))
        has_any_star_cpn = bool(np.any((scal_np[13] > 0.0) & (scal_np[14] > 0.0)))

        raw_put_b = torch.where(has_put_b, torch.clamp(strike_b - w_opt_b, min=0.0), torch.zeros_like(w_opt_b))
        d_put_dw_b = torch.where(has_put_b & (w_opt_b < strike_b), -torch.ones_like(w_opt_b), torch.zeros_like(w_opt_b))

        if has_any_low:
            cap_b = torch.clamp(strike_b - low_k_b, min=0.0)
            under_cap = raw_put_b < cap_b
            raw_put_b = torch.where(has_low_b, torch.minimum(raw_put_b, cap_b), raw_put_b)
            d_put_dw_b = torch.where(has_low_b & (~under_cap), torch.zeros_like(d_put_dw_b), d_put_dw_b)

        if has_any_ki:
            ki_up_b, dki_up_b = _digital_ramp_and_deriv_batch(
                w_opt_b, ki_bar_b, ki_spr_b, compute_deriv=compute_pathwise_delta
            )
            ki_ind_b = torch.where(has_ki_b, 1.0 - ki_up_b, torch.ones_like(ki_up_b))
            if compute_pathwise_delta and dki_up_b is not None:
                d_put_dw_b = torch.where(
                    has_ki_b, d_put_dw_b * ki_ind_b - raw_put_b * dki_up_b, d_put_dw_b
                )
            raw_put_b = raw_put_b * ki_ind_b

        star_ind_b = torch.zeros_like(w_opt_b)
        if has_any_star:
            w_best_b = _evaluate_monitoring_index_gpu(perf_term, "BestOf", None, asset_dim=1)
            star_ind_b = torch.where(has_star_b, (w_best_b >= star_k_b).to(torch.float64), star_ind_b)
            raw_put_b = raw_put_b * (1.0 - star_ind_b)
            d_put_dw_b = d_put_dw_b * (1.0 - star_ind_b)

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

        if compute_pathwise_delta and dw_opt_b is not None and d_alive is not None and d_total_pv is not None:
            d_unit_put_b = (
                d_alive * raw_put_b.unsqueeze(1)
                + (alive * d_put_dw_b).unsqueeze(1) * dw_opt_b
            )
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
                d_opt_b = d_opt_b + df_term_b.unsqueeze(1) * d_alive * (star_ind_b * star_cpn_b).unsqueeze(1)
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
    idx = str(index_name).strip()
    is_asym = idx in ("RatioAB", "RatioBA", "DiffAB", "DiffBA", "Outperformance", "Spread")
    p = perf.index_select(asset_dim, inv_lex_perm_t) if (is_asym and inv_lex_perm_t is not None) else perf
    d = int(p.shape[asset_dim])

    if idx in ("WorstOf", "Lookback"):
        w_val, _ = p.min(dim=asset_dim)
        if not compute_grad:
            return w_val, None
        ties = (p == w_val.unsqueeze(asset_dim)).to(p.dtype)
        grad = ties / torch.clamp(ties.sum(dim=asset_dim, keepdim=True), min=1.0)
        return w_val, grad
    if idx == "BestOf":
        w_val, _ = p.max(dim=asset_dim)
        if not compute_grad:
            return w_val, None
        ties = (p == w_val.unsqueeze(asset_dim)).to(p.dtype)
        grad = ties / torch.clamp(ties.sum(dim=asset_dim, keepdim=True), min=1.0)
        return w_val, grad
    if idx in ("Average", "AverageOf", "MultiPerf"):
        w_val = p.mean(dim=asset_dim)
        if not compute_grad:
            return w_val, None
        grad = torch.full_like(p, 1.0 / float(max(d, 1)))
        return w_val, grad
    if idx == "Dispersion":
        mean_p = p.mean(dim=asset_dim, keepdim=True)
        diff = p - mean_p
        w_val = torch.sqrt((diff * diff).mean(dim=asset_dim).clamp(min=1e-12))
        if not compute_grad:
            return w_val, None
        grad = diff / (float(max(d, 1)) * w_val.unsqueeze(asset_dim).clamp(min=1e-12))
        return w_val, grad

    w_val = _evaluate_monitoring_index_gpu(perf, idx, inv_lex_perm_t, asset_dim=asset_dim)
    if not compute_grad:
        return w_val, None

    grad = torch.zeros_like(p)
    if idx == "RatioBestWorst":
        w_max = p.max(dim=asset_dim, keepdim=True).values
        w_min = p.min(dim=asset_dim, keepdim=True).values.clamp(min=1e-12)
        t_max = (p == w_max).to(p.dtype)
        t_max = t_max / t_max.sum(dim=asset_dim, keepdim=True).clamp(min=1.0)
        t_min = (p == w_min).to(p.dtype)
        t_min = t_min / t_min.sum(dim=asset_dim, keepdim=True).clamp(min=1.0)
        grad = t_max / w_min - (w_max / (w_min * w_min)) * t_min
    elif idx == "RatioBestAverage":
        w_max = p.max(dim=asset_dim, keepdim=True).values
        w_avg = p.mean(dim=asset_dim, keepdim=True).clamp(min=1e-12)
        t_max = (p == w_max).to(p.dtype)
        t_max = t_max / t_max.sum(dim=asset_dim, keepdim=True).clamp(min=1.0)
        grad = t_max / w_avg - (w_max / (w_avg * w_avg)) / float(max(d, 1))
    elif idx == "RatioAverageWorst":
        w_avg = p.mean(dim=asset_dim, keepdim=True)
        w_min = p.min(dim=asset_dim, keepdim=True).values.clamp(min=1e-12)
        t_min = (p == w_min).to(p.dtype)
        t_min = t_min / t_min.sum(dim=asset_dim, keepdim=True).clamp(min=1.0)
        grad = (1.0 / float(max(d, 1))) / w_min - (w_avg / (w_min * w_min)) * t_min
    elif idx in ("DiffBestWorst", "Rainbow"):
        w_max = p.max(dim=asset_dim, keepdim=True).values
        w_min = p.min(dim=asset_dim, keepdim=True).values
        t_max = (p == w_max).to(p.dtype)
        t_max = t_max / t_max.sum(dim=asset_dim, keepdim=True).clamp(min=1.0)
        t_min = (p == w_min).to(p.dtype)
        t_min = t_min / t_min.sum(dim=asset_dim, keepdim=True).clamp(min=1.0)
        grad = t_max - t_min
    elif idx == "DiffBestAverage":
        w_max = p.max(dim=asset_dim, keepdim=True).values
        t_max = (p == w_max).to(p.dtype)
        t_max = t_max / t_max.sum(dim=asset_dim, keepdim=True).clamp(min=1.0)
        grad = t_max - 1.0 / float(max(d, 1))
    elif idx == "DiffAverageWorst":
        w_min = p.min(dim=asset_dim, keepdim=True).values
        t_min = (p == w_min).to(p.dtype)
        t_min = t_min / t_min.sum(dim=asset_dim, keepdim=True).clamp(min=1.0)
        grad = 1.0 / float(max(d, 1)) - t_min
    elif idx in ("RatioAB", "Outperformance"):
        if d >= 2:
            b_safe = p.select(asset_dim, 1).clamp(min=1e-12)
            grad.select(asset_dim, 0).copy_(1.0 / b_safe)
            grad.select(asset_dim, 1).copy_(-p.select(asset_dim, 0) / (b_safe * b_safe))
        else:
            grad.select(asset_dim, 0).fill_(1.0)
    elif idx == "RatioBA":
        if d >= 2:
            a_safe = p.select(asset_dim, 0).clamp(min=1e-12)
            grad.select(asset_dim, 1).copy_(1.0 / a_safe)
            grad.select(asset_dim, 0).copy_(-p.select(asset_dim, 1) / (a_safe * a_safe))
        else:
            grad.select(asset_dim, 0).fill_(1.0)
    elif idx in ("DiffAB", "Spread"):
        if d >= 2:
            grad.select(asset_dim, 0).fill_(1.0)
            grad.select(asset_dim, 1).fill_(-1.0)
        else:
            grad.select(asset_dim, 0).fill_(1.0)
    elif idx == "DiffBA":
        if d >= 2:
            grad.select(asset_dim, 1).fill_(1.0)
            grad.select(asset_dim, 0).fill_(-1.0)
        else:
            grad.select(asset_dim, 0).fill_(1.0)

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
    "DupireQMCConfig",
    "DupireQMCBackend",
    "GpuTritonDupireQMC",
    "GpuTorchDupireQMC",
    "CpuNumpyDupireQMC",
    "DupireQMCGpuSolver",
]
