"""Regression tests for the C++23 Tier-1 & CRN backend (`tier1_autocall_cpu.cpp`)."""

from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from deepvol.models.autocall import _tier1_backend as tb
from deepvol.models.autocall.dupire_qmc_solver import price_eq_autocall_dupire_qmc
from deepvol.models.autocall.greeks_engine import _escrowed_spot_softplus_np
from deepvol.models.autocall.pre_post_processor import (
    assemble_autocall_cashflows_and_pv,
    preprocess_eq_autocall,
)
from deepvol.models.autocall.qmc_aad_greeks import (
    _build_lv_table_and_forward_cache,
    _CrnLooL2oBatchArgsC,
    _get_cpp_crn_fns,
    _sym_sqrt_and_kl_pca_np,
)


def _s5_np(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    u = np.clip(t, 0.0, 1.0)
    return np.asarray(u * u * u * (10.0 + u * (-15.0 + 6.0 * u)), dtype=np.float64)


def _ds5_np(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    u = np.clip(t, 0.0, 1.0)
    return np.asarray(
        np.where((t > 0.0) & (t < 1.0), 30.0 * u * u * (1.0 - u) * (1.0 - u), 0.0),
        dtype=np.float64,
    )


def test_cpp23_is_long_basket_gearing_invariance_loo_l2o_and_vibrato() -> None:
    """Verify IsLongBasket with option_gearing=1.111111 in C++23 LOO/L2O and Vibrato Gamma."""
    backend = tb.get_tier1_backend()
    assert isinstance(backend, tb.Cpp23Tier1Backend)
    _, loo_fn = _get_cpp_crn_fns()
    assert loo_fn is not None

    rng = np.random.default_rng(59553619)
    d_assets, m_rem, n_paths = 5, 4, 64
    perf_base = np.exp(rng.normal(loc=-0.02, scale=0.22, size=(m_rem, d_assets, n_paths))).astype(
        np.float64
    )
    bump_a = np.exp(rng.normal(loc=0.0, scale=0.21, size=(4, m_rem, n_paths))).astype(np.float64)
    bump_b = np.exp(rng.normal(loc=-0.01, scale=0.23, size=(4, m_rem, n_paths))).astype(np.float64)
    full_perf = np.exp(rng.normal(loc=0.01, scale=0.20, size=(m_rem, d_assets, n_paths))).astype(
        np.float64
    )

    df_pay = np.broadcast_to(
        np.array([[0.97, 0.94, 0.91, 0.88]], dtype=np.float64), (4, m_rem)
    ).copy()
    ko_bar = np.array([1.00, 1.00, 1.00, 1.00], dtype=np.float64)
    is_call = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    cpn_bar = np.array([0.90, 0.90, 0.90, 0.90], dtype=np.float64)
    is_guar = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    cpn_rates = np.array([0.09, 0.09, 0.09, 0.09], dtype=np.float64)
    in_tr = np.zeros(4, dtype=np.float64)

    strike = 0.90
    ki_sp = -0.10
    ko_sp = -0.05
    cpn_sp = -0.10
    part_k = 1.05
    part_rate = 0.25

    def _ref_islb_pv(p_mat: npt.NDArray[np.float64], gearing: float) -> float:
        surv = np.ones(n_paths, dtype=np.float64)
        c_mem = np.zeros(n_paths, dtype=np.float64)
        pv = np.zeros(n_paths, dtype=np.float64)
        for m in range(m_rem):
            wc = np.min(p_mat[m], axis=0)
            blo_k, bhi_k = min(ko_bar[m], ko_bar[m] + ko_sp), max(ko_bar[m], ko_bar[m] + ko_sp)
            gk = np.clip((wc - blo_k) / (bhi_k - blo_k), 0.0, 1.0) if is_call[m] > 0.5 else 0.0
            blo_c, bhi_c = min(cpn_bar[m], cpn_bar[m] + cpn_sp), max(cpn_bar[m], cpn_bar[m] + cpn_sp)
            gc = np.clip((wc - blo_c) / (bhi_c - blo_c), 0.0, 1.0)
            pm = surv * gk
            ec = cpn_rates[m] + c_mem
            ccf = surv * gc * ec
            c_mem = surv * ec * (1.0 - gc)
            pv += df_pay[0, m] * (pm + ccf)
            surv = np.maximum(surv - pm, 0.0)
        w_wof = np.min(p_mat[-1], axis=0)
        w_avg = np.mean(p_mat[-1], axis=0)
        blo_ki, bhi_ki = min(strike, strike + ki_sp), max(strike, strike + ki_sp)
        ki_wof_k = np.clip((bhi_ki - w_wof) / (bhi_ki - blo_ki), 0.0, 1.0)
        eff_g = gearing if abs(gearing) > 1e-12 else 1.0
        unit_put = surv * ((1.0 - w_avg) * ki_wof_k) / eff_g
        unit_call = surv * np.maximum(w_wof - part_k, 0.0)
        pv += df_pay[0, -1] * (surv - gearing * unit_put + part_rate * unit_call)
        return float(np.mean(pv))

    aa = np.array([-1, 1, 0, -2], dtype=np.int32)
    ab = np.array([-1, -1, 3, -1], dtype=np.int32)
    ba_buf = np.ascontiguousarray(bump_a, dtype=np.float64)
    bb_buf = np.ascontiguousarray(bump_b, dtype=np.float64)

    # First batch: base (-1,-1), LOO (1,-1), L2O (0,3)
    pv_by_g: dict[float, npt.NDArray[np.float64]] = {}
    for gearing in (1.0, 1.111111, 1.5):
        pv_out = np.zeros(3, dtype=np.float64)
        desc = _CrnLooL2oBatchArgsC(
            n_bumps=3,
            d_assets=d_assets,
            m_rem=m_rem,
            n_paths_total=n_paths,
            perf_base=perf_base.ctypes.data,
            bump_perf_a=ba_buf.ctypes.data,
            bump_perf_b=bb_buf.ctypes.data,
            bump_asset_a=aa[:3].ctypes.data,
            bump_asset_b=ab[:3].ctypes.data,
            df_pay_mat=df_pay[:3].ctypes.data,
            w_fund_mat=0,
            pv_in_transit=in_tr[:3].ctypes.data,
            pv_out=pv_out.ctypes.data,
            ko_barriers=ko_bar.ctypes.data,
            is_callable=is_call.ctypes.data,
            cpn_barriers=cpn_bar.ctypes.data,
            is_guaranteed_cpn=is_guar.ctypes.data,
            coupon_rates=cpn_rates.ctypes.data,
            ko_spread=ko_sp,
            cpn_spread=cpn_sp,
            ki_spread=ki_sp,
            one_star_strike=0.0,
            one_star_eps=0.02,
            strike=strike,
            lower_strike=0.0,
            ki_barrier=1.0,
            option_gearing=gearing,
            autocall_coupon=0.0,
            c_mem_0=0.0,
            i_ki_past=0.0,
            part_call_strike=part_k,
            part_call_rate=part_rate,
            one_star_coupon=0.0,
            call_code=0,
            cpn_code=0,
            opt_code=0,
            has_ki=1,
            is_memory=1,
            is_long_basket=1,
            no_notional=0,
            force_cpn_ac=0,
            cpn_dir_down=0,
            cpn_type_out=0,
        )
        desc._reserved[0] = 1
        desc._reserved[1] = 2
        loo_fn(ctypes.byref(desc))
        pv_by_g[gearing] = pv_out
        for b_i in range(3):
            p_b = perf_base.copy()
            if aa[b_i] >= 0:
                p_b[:, int(aa[b_i]), :] = ba_buf[b_i]
            if ab[b_i] >= 0:
                p_b[:, int(ab[b_i]), :] = bb_buf[b_i]
            np.testing.assert_allclose(pv_out[b_i], _ref_islb_pv(p_b, gearing), atol=1e-12, rtol=1e-12)

    np.testing.assert_allclose(pv_by_g[1.111111], pv_by_g[1.0], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(pv_by_g[1.5], pv_by_g[1.0], atol=1e-14, rtol=1e-14)

    # Full-replacement row (aa == -2) must also match
    pv_full = np.zeros(1, dtype=np.float64)
    aa_full = np.array([-2], dtype=np.int32)
    ab_full = np.array([-1], dtype=np.int32)
    desc_full = _CrnLooL2oBatchArgsC(
        n_bumps=1,
        d_assets=d_assets,
        m_rem=m_rem,
        n_paths_total=n_paths,
        perf_base=0,
        bump_perf_a=full_perf.ctypes.data,
        bump_perf_b=0,
        bump_asset_a=aa_full.ctypes.data,
        bump_asset_b=ab_full.ctypes.data,
        df_pay_mat=df_pay[:1].ctypes.data,
        w_fund_mat=0,
        pv_in_transit=in_tr[:1].ctypes.data,
        pv_out=pv_full.ctypes.data,
        ko_barriers=ko_bar.ctypes.data,
        is_callable=is_call.ctypes.data,
        cpn_barriers=cpn_bar.ctypes.data,
        is_guaranteed_cpn=is_guar.ctypes.data,
        coupon_rates=cpn_rates.ctypes.data,
        ko_spread=ko_sp,
        cpn_spread=cpn_sp,
        ki_spread=ki_sp,
        one_star_strike=0.0,
        one_star_eps=0.02,
        strike=strike,
        lower_strike=0.0,
        ki_barrier=1.0,
        option_gearing=1.111111,
        autocall_coupon=0.0,
        c_mem_0=0.0,
        i_ki_past=0.0,
        part_call_strike=part_k,
        part_call_rate=part_rate,
        one_star_coupon=0.0,
        call_code=0,
        cpn_code=0,
        opt_code=0,
        has_ki=1,
        is_memory=1,
        is_long_basket=1,
        no_notional=0,
        force_cpn_ac=0,
        cpn_dir_down=0,
        cpn_type_out=0,
    )
    desc_full._reserved[0] = 1
    desc_full._reserved[1] = 2
    loo_fn(ctypes.byref(desc_full))
    np.testing.assert_allclose(pv_full[0], _ref_islb_pv(full_perf, 1.111111), atol=1e-12, rtol=1e-12)

    # Vibrato Gamma C++ check with IsLongBasket=True, option_gearing=1.111111 vs 1.0 & PartCall
    corr = np.full((d_assets, d_assets), 0.25, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    chol = np.ascontiguousarray(np.linalg.cholesky(corr), dtype=np.float64)
    l_inv_t = np.ascontiguousarray(np.linalg.inv(chol).T, dtype=np.float64)
    c_inv = np.ascontiguousarray(np.linalg.inv(corr), dtype=np.float64)
    vib_n_steps = 2
    sv_vib = np.ascontiguousarray(rng.standard_normal((vib_n_steps, d_assets, n_paths)), dtype=np.float64)
    lx_pre = np.ascontiguousarray(rng.normal(scale=0.14, size=(d_assets, n_paths)), dtype=np.float64)
    j_pre = np.ascontiguousarray(np.exp(lx_pre) * 1.01, dtype=np.float64)
    sig_pre = np.ascontiguousarray(rng.uniform(0.18, 0.30, size=(d_assets, n_paths)), dtype=np.float64)
    fwd_term = np.array([1.01, 0.99, 1.03, 0.98, 1.02], dtype=np.float64)
    cdl_m = np.ascontiguousarray(rng.uniform(0.96, 1.04, size=(m_rem, d_assets)), dtype=np.float64)
    cdl2_last = np.array([0.01, -0.02, 0.015, 0.0, -0.01], dtype=np.float64)
    eps_vec = np.full(m_rem, 0.03, dtype=np.float64)
    perm = np.arange(d_assets, dtype=np.int64)

    h_10, _ = tb.crn_eval_vibrato_gamma_cpu(
        perf_mat=perf_base,
        lx_pre=lx_pre,
        j_pre=j_pre,
        sig_pre=sig_pre,
        sv_vib=sv_vib,
        chol_mat=chol,
        l_inv_t=l_inv_t,
        c_inv=c_inv,
        fwd_term=fwd_term,
        cdl_m=cdl_m,
        cdl2_last=cdl2_last,
        ko_barriers=ko_bar,
        is_callable=is_call,
        cpn_barriers=cpn_bar,
        is_guaranteed_cpn=is_guar,
        coupon_rates=cpn_rates,
        eps_vec=eps_vec,
        inv_lex_perm=perm,
        vib_step_start=0,
        vib_n_steps=vib_n_steps,
        df_last=0.88,
        h_vib=0.08,
        ko_spread=ko_sp,
        cpn_spread=cpn_sp,
        ki_spread=ki_sp,
        one_star_strike=0.0,
        one_star_eps=0.02,
        strike=strike,
        lower_strike=0.0,
        ki_barrier=1.0,
        option_gearing=1.0,
        autocall_coupon=0.0,
        c_mem_0=0.0,
        i_ki_past=0.0,
        part_call_strike=part_k,
        part_call_rate=part_rate,
        one_star_coupon=0.0,
        call_code=0,
        cpn_code=0,
        opt_code=0,
        has_ki=True,
        ki_active=True,
        has_one_star=False,
        is_memory=True,
        is_long_basket=True,
        no_notional=False,
        force_cpn_ac=False,
        cpn_dir_down=False,
        cpn_type_out=False,
        has_lower_strike=False,
        has_part_call=True,
        emit_frozen_term=True,
    )
    h_gear, _ = tb.crn_eval_vibrato_gamma_cpu(
        perf_mat=perf_base,
        lx_pre=lx_pre,
        j_pre=j_pre,
        sig_pre=sig_pre,
        sv_vib=sv_vib,
        chol_mat=chol,
        l_inv_t=l_inv_t,
        c_inv=c_inv,
        fwd_term=fwd_term,
        cdl_m=cdl_m,
        cdl2_last=cdl2_last,
        ko_barriers=ko_bar,
        is_callable=is_call,
        cpn_barriers=cpn_bar,
        is_guaranteed_cpn=is_guar,
        coupon_rates=cpn_rates,
        eps_vec=eps_vec,
        inv_lex_perm=perm,
        vib_step_start=0,
        vib_n_steps=vib_n_steps,
        df_last=0.88,
        h_vib=0.08,
        ko_spread=ko_sp,
        cpn_spread=cpn_sp,
        ki_spread=ki_sp,
        one_star_strike=0.0,
        one_star_eps=0.02,
        strike=strike,
        lower_strike=0.0,
        ki_barrier=1.0,
        option_gearing=1.111111,
        autocall_coupon=0.0,
        c_mem_0=0.0,
        i_ki_past=0.0,
        part_call_strike=part_k,
        part_call_rate=part_rate,
        one_star_coupon=0.0,
        call_code=0,
        cpn_code=0,
        opt_code=0,
        has_ki=True,
        ki_active=True,
        has_one_star=False,
        is_memory=True,
        is_long_basket=True,
        no_notional=False,
        force_cpn_ac=False,
        cpn_dir_down=False,
        cpn_type_out=False,
        has_lower_strike=False,
        has_part_call=True,
        emit_frozen_term=True,
    )
    np.testing.assert_allclose(h_gear, h_10, atol=1e-14, rtol=1e-14)

    # Independent NumPy reference for Vibrato Gamma confirming WorstOf PartCall + IsLongBasket -1.0
    surv_pre = np.ones(n_paths, dtype=np.float64)
    c_mem_v = np.zeros(n_paths, dtype=np.float64)
    dlns = np.zeros((d_assets, n_paths), dtype=np.float64)
    dcm = np.zeros((d_assets, n_paths), dtype=np.float64)
    for m in range(m_rem - 1):
        wc = np.min(perf_base[m], axis=0)
        idx_min = np.argmin(perf_base[m], axis=0)
        inv_ko = 1.0 / abs(ko_sp)
        t_ko = (wc - (ko_bar[m] + ko_sp)) * inv_ko
        gk = _s5_np(t_ko) if is_call[m] > 0.5 else np.zeros(n_paths, dtype=np.float64)
        dgk = (_ds5_np(t_ko) * inv_ko) if is_call[m] > 0.5 else np.zeros(n_paths, dtype=np.float64)
        coef_s = -dgk / np.maximum(1.0 - gk, 1e-8)
        inv_cp = 1.0 / abs(cpn_sp)
        t_cp = (wc - (cpn_bar[m] + cpn_sp)) * inv_cp
        gc = _s5_np(t_cp)
        dgc = _ds5_np(t_cp) * inv_cp
        eff_c = cpn_rates[m] + c_mem_v
        for d in range(d_assets):
            mask_d = (idx_min == d).astype(np.float64)
            pd = perf_base[m, d]
            prev_dlns = dlns[d].copy()
            d_egc = mask_d * dgc * pd * cdl_m[m, d]
            dcm[d] = surv_pre * (prev_dlns * eff_c * (1.0 - gc) + dcm[d] * (1.0 - gc) - eff_c * d_egc)
            dlns[d] = prev_dlns + mask_d * coef_s * pd * cdl_m[m, d]
        c_mem_v = surv_pre * eff_c * (1.0 - gc)
        surv_pre = np.maximum(surv_pre - surv_pre * gk, 0.0)

    zk_mean = np.sum(sv_vib, axis=0) / math.sqrt(float(vib_n_steps))
    zc = chol @ zk_mean
    xi_ref = l_inv_t @ zk_mean
    sqrt_h = math.sqrt(0.08)
    eff_c_l = cpn_rates[-1] + c_mem_v
    vals: dict[int, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
    for sgn, mult in ((0, 0.0), (1, 1.0), (-1, -1.0)):
        xd = lx_pre - 0.5 * sig_pre * sig_pre * 0.08 + mult * sig_pre * sqrt_h * zc
        p_s = fwd_term[:, None] * np.exp(np.clip(xd, -30.0, 30.0))
        wc_s = np.min(p_s, axis=0)
        wa_s = np.mean(p_s, axis=0)
        gk_s = np.zeros(n_paths, dtype=np.float64)
        gc_s = np.clip((wc_s - (cpn_bar[-1] + cpn_sp)) / abs(cpn_sp), 0.0, 1.0)
        ki_s = np.clip((strike - wc_s) / abs(ki_sp), 0.0, 1.0)
        opt_s = -1.0 * ki_s * (1.0 - wa_s) + part_rate * np.maximum(wc_s - part_k, 0.0)
        vals[sgn] = (gk_s + gc_s * eff_c_l + (1.0 - gk_s) * (1.0 + opt_s), gc_s)

    df_diff = 0.5 * (vals[1][0] - vals[-1][0])
    df_hess = 0.5 * (vals[1][0] + vals[-1][0]) - vals[0][0]
    degc_diff = 0.5 * (vals[1][1] - vals[-1][1])
    inv_s = 1.0 / np.maximum(sig_pre * sqrt_h, 1e-6)
    j_last = j_pre * cdl_m[-1, :, None]
    h_ref = np.zeros((d_assets, d_assets), dtype=np.float64)
    for i in range(d_assets):
        w1_raw_i = df_diff * xi_ref[i] * inv_s[i] * j_pre[i]
        w1_i = df_diff * xi_ref[i] * inv_s[i] * j_last[i]
        w1_egc_i = degc_diff * xi_ref[i] * inv_s[i] * j_last[i]
        for j in range(d_assets):
            w1_j = df_diff * xi_ref[j] * inv_s[j] * j_last[j]
            w1_egc_j = degc_diff * xi_ref[j] * inv_s[j] * j_last[j]
            w2_ij = df_hess * (xi_ref[i] * xi_ref[j] - c_inv[i, j]) * inv_s[i] * inv_s[j] * j_last[i] * j_last[j]
            diag_corr = (w1_raw_i * cdl2_last[i]) if i == j else 0.0
            h_ij = 0.88 * surv_pre * (
                w2_ij + dlns[i] * w1_j + dlns[j] * w1_i + dcm[i] * w1_egc_j + dcm[j] * w1_egc_i + diag_corr
            )
            h_ref[i, j] = float(np.mean(h_ij))
    np.testing.assert_allclose(h_gear, h_ref, atol=1e-12, rtol=1e-12)


def test_cpp23_deal_59553619_platen_loo_vs_dupire_qmc_and_murex() -> None:
    """Verify deal 59553619 (is_long_basket=True, option_gearing=1.111111) C++23 vs dupire_qmc_solver & Murex."""
    req_path = Path("docs/eq_autocall_ps_batch_b10316f2/requests/trade_095_59553619.json")
    if not req_path.is_file():
        pytest.skip("Deal 59553619 fixture not present")

    req = json.loads(req_path.read_text(encoding="utf-8"))
    state = preprocess_eq_autocall(req)
    assert bool(state.is_long_basket) is True
    assert float(state.option_gearing) == pytest.approx(1.111111, abs=1e-9)

    murex_pv = 0.8874838262430271
    qmc_res = price_eq_autocall_dupire_qmc(
        state,
        n_paths=16384,
        n_replicates=8,
        n_substeps_per_period=16,
        device="cpu",
        use_gpu=False,
    )
    assert abs(qmc_res.product_value - murex_pv) * 1e4 < 15.0

    # Generate Sobol normals + Platen LV paths using the C++23 ABI v2 kernels
    backend = tb.get_tier1_backend()
    assert isinstance(backend, tb.Cpp23Tier1Backend)
    _, loo_fn = _get_cpp_crn_fns()
    assert loo_fn is not None

    d_assets = int(state.d_assets)
    m_rem = int(state.m_rem)
    n_sub = 16
    eff_r = 8
    n_rep = 2048
    n_total = eff_r * n_rep
    lv_nt, lv_ny = 32, 64

    sorted_ids = sorted(state.vol_surfaces.keys())
    spots_np = state.spots_t0.detach().cpu().numpy().astype(np.float64)
    ref_spots_np = state.ref_spots.detach().cpu().numpy().astype(np.float64)
    div_pv_np = state.dividend_schedule.div_pv.detach().cpu().numpy().astype(np.float64)
    fwd_perf_np = state.forward_perf.detach().cpu().numpy().astype(np.float64)
    tau_obs_np = state.tau_obs.detach().cpu().numpy().astype(np.float64)
    corr_np = state.corr_matrix.detach().cpu().numpy().astype(np.float64)
    chol_base = np.ascontiguousarray(np.linalg.cholesky(corr_np), dtype=np.float64)

    esc_base = _escrowed_spot_softplus_np(spots_np[:, None], div_pv_np)
    growth_base = (fwd_perf_np * ref_spots_np[:, None]) / np.maximum(esc_base, 1e-12)
    r_tot_obs = np.log(np.maximum(growth_base, 1e-15))
    r_nat_obs = (
        state.r_native_obs.detach().cpu().numpy().astype(np.float64)
        if state.r_native_obs is not None
        else r_tot_obs.copy()
    )

    lv_t_grid = np.linspace(1e-5, max(float(tau_obs_np[-1]), 1e-4), lv_nt, dtype=np.float64)
    lv_y_min = np.full(d_assets, -1.6, dtype=np.float64)
    lv_y_max = np.full(d_assets, 1.0, dtype=np.float64)
    lv_tables = np.empty((d_assets, lv_nt, lv_ny), dtype=np.float64)
    for i, tk in enumerate(sorted_ids):
        lv_y_grid = np.linspace(lv_y_min[i], lv_y_max[i], lv_ny, dtype=np.float64)
        surf = state.vol_surfaces[tk]
        entry_i = _build_lv_table_and_forward_cache(
            surf,
            float(spots_np[i]),
            tau_obs_np,
            r_nat_obs[i],
            lv_t_grid,
            lv_y_grid,
            div_pv_np[i],
            need_vega_jac=False,
        )
        lv_tables[i] = entry_i["lv_out"]

    from scipy.stats import qmc

    u_time_np, chol_base, _, _ = _sym_sqrt_and_kl_pca_np(tau_obs_np, corr_np, corr_aad_mode="cholesky")
    total_sobol_dims = m_rem * (1 + n_sub) * d_assets
    sob = qmc.Sobol(d=total_sobol_dims, scramble=False)
    sv = np.ascontiguousarray(sob._sv.astype(np.uint32))
    rng_s = np.random.default_rng(42)
    shifts = rng_s.integers(0, 1 << 30, size=(eff_r, total_sobol_dims), dtype=np.uint32)
    z_indep, _ = tb.crn_generate_sobol_normals_cpu(
        d_assets=d_assets,
        m_rem=m_rem,
        n_sub=n_sub,
        eff_r=eff_r,
        n_rep=n_rep,
        dir_numbers=sv,
        digital_shifts=shifts,
        bb_ortho_mat=np.ascontiguousarray(u_time_np, dtype=np.float64),
        chol_mat=chol_base,
        n_bits=30,
        use_gray_code=True,
        apply_bb_time=True,
        apply_corr=False,
        antithetic=True,
        return_indep=True,
        return_corr=False,
    )
    assert z_indep is not None
    _, norm_x_m, _, _, _ = backend.crn_platen_sde_aad_cpu(
        lv_tables=lv_tables,
        z_indep=z_indep,
        chol_base=chol_base,
        tau_obs=tau_obs_np,
        r_native_obs=r_nat_obs,
        r_tot_obs=r_tot_obs,
        lv_t_grid=lv_t_grid,
        lv_y_min=lv_y_min,
        lv_y_max=lv_y_max,
        n_sub=n_sub,
        eff_r=eff_r,
        n_rep=n_rep,
        run_forward=True,
        run_backward=False,
    )
    assert norm_x_m is not None
    perf_base = np.ascontiguousarray(
        norm_x_m * fwd_perf_np.T[:, :, None], dtype=np.float64
    )

    df_pay = state.df_pay.detach().cpu().numpy().astype(np.float64)[None, :]
    ko_bar = state.ko_barriers.detach().cpu().numpy().astype(np.float64)
    is_call = state.is_callable.detach().cpu().numpy().astype(np.float64)
    cpn_bar = state.cpn_barriers.detach().cpu().numpy().astype(np.float64)
    is_guar = state.is_guaranteed_cpn.detach().cpu().numpy().astype(np.float64)
    cpn_rates = state.coupon_amounts.detach().cpu().numpy().astype(np.float64)
    inv_perm = np.asarray(state.inv_lex_perm, dtype=np.int32)
    in_tr = np.array([float(state.pv_in_transit)], dtype=np.float64)
    pv_cpp = np.zeros(1, dtype=np.float64)
    aa = np.array([-1], dtype=np.int32)
    ab = np.array([-1], dtype=np.int32)

    desc = _CrnLooL2oBatchArgsC(
        n_bumps=1,
        d_assets=d_assets,
        m_rem=m_rem,
        n_paths_total=n_total,
        perf_base=perf_base.ctypes.data,
        bump_perf_a=0,
        bump_perf_b=0,
        bump_asset_a=aa.ctypes.data,
        bump_asset_b=ab.ctypes.data,
        df_pay_mat=df_pay.ctypes.data,
        w_fund_mat=0,
        pv_in_transit=in_tr.ctypes.data,
        pv_out=pv_cpp.ctypes.data,
        ko_barriers=ko_bar.ctypes.data,
        is_callable=is_call.ctypes.data,
        cpn_barriers=cpn_bar.ctypes.data,
        is_guaranteed_cpn=is_guar.ctypes.data,
        coupon_rates=cpn_rates.ctypes.data,
        ko_spread=float(state.ko_spread),
        cpn_spread=float(state.cpn_spread),
        ki_spread=float(state.ki_spread),
        one_star_strike=0.0,
        one_star_eps=0.02,
        strike=float(state.strike),
        lower_strike=0.0,
        ki_barrier=float(state.ki_barrier or 0.0),
        option_gearing=float(state.option_gearing),
        autocall_coupon=float(state.autocall_coupon),
        c_mem_0=float(state.c_mem_0),
        i_ki_past=float(state.i_ki_past),
        part_call_strike=0.0,
        part_call_rate=0.0,
        one_star_coupon=0.0,
        call_code=0,
        cpn_code=0,
        opt_code=0,
        has_ki=1,
        is_memory=1,
        is_long_basket=1,
        no_notional=0,
        force_cpn_ac=0,
        cpn_dir_down=0,
        cpn_type_out=0,
    )
    desc._reserved[0] = int(inv_perm[0]) + 1
    desc._reserved[1] = int(inv_perm[1]) + 1
    loo_fn(ctypes.byref(desc))

    # Compare PV & CashFlows against assemble_autocall_cashflows_and_pv with eff_gearing division
    import torch

    surv = np.ones(n_total, dtype=np.float64)
    c_mem = np.full(n_total, float(state.c_mem_0), dtype=np.float64)
    p_stop_mean = np.zeros(m_rem, dtype=np.float64)
    q_cpn_mean = np.zeros(m_rem, dtype=np.float64)
    cpn_undisc_mean = np.zeros(m_rem, dtype=np.float64)
    for m in range(m_rem):
        wc = np.min(perf_base[m], axis=0)
        blo_k, bhi_k = min(ko_bar[m], ko_bar[m] + float(state.ko_spread)), max(ko_bar[m], ko_bar[m] + float(state.ko_spread))
        gk = np.clip((wc - blo_k) / (bhi_k - blo_k), 0.0, 1.0) if is_call[m] > 0.5 else 0.0
        blo_c, bhi_c = min(cpn_bar[m], cpn_bar[m] + float(state.cpn_spread)), max(cpn_bar[m], cpn_bar[m] + float(state.cpn_spread))
        gc = np.clip((wc - blo_c) / (bhi_c - blo_c), 0.0, 1.0)
        pm = surv * gk
        qc = surv * gc
        ec = cpn_rates[m] + c_mem
        ccf = qc * ec
        c_mem = surv * ec * (1.0 - gc)
        p_stop_mean[m] = float(np.mean(pm))
        surv_prev_m = float(np.mean(surv))
        q_cpn_mean[m] = float(np.mean(qc)) / max(surv_prev_m, 1e-12)
        cpn_undisc_mean[m] = float(np.mean(ccf))
        surv = np.maximum(surv - pm, 0.0)

    w_wof = np.min(perf_base[-1], axis=0)
    w_avg = np.mean(perf_base[-1], axis=0)
    blo_ki = min(float(state.strike), float(state.strike) + float(state.ki_spread))
    bhi_ki = max(float(state.strike), float(state.strike) + float(state.ki_spread))
    ki_wof_k = np.clip((bhi_ki - w_wof) / (bhi_ki - blo_ki), 0.0, 1.0)
    l_put_div = float(np.mean(surv * (1.0 - w_avg) * ki_wof_k)) / float(state.option_gearing)

    asm = assemble_autocall_cashflows_and_pv(
        state,
        p_stop=torch.as_tensor(p_stop_mean, dtype=torch.float64),
        q_cpn=torch.as_tensor(q_cpn_mean, dtype=torch.float64),
        l_put=torch.tensor(l_put_div, dtype=torch.float64),
        coupon_cf_undisc=cpn_undisc_mean,
        build_cashflow_items=True,
    )
    pv_asm = float(asm.product_value.item())
    np.testing.assert_allclose(pv_cpp[0], pv_asm, atol=1e-12, rtol=1e-12)
    assert abs(pv_cpp[0] - murex_pv) * 1e4 < 15.0
    assert abs(pv_cpp[0] - qmc_res.product_value) * 1e4 < 15.0
    islb_cfs_cpp = [cf.Amount for cf in asm.cashflows if cf.Description == "IsLongBasket"]
    islb_cfs_qmc = [cf.Amount for cf in qmc_res.cashflows if cf.Description == "IsLongBasket"]
    assert len(islb_cfs_cpp) == 1 and len(islb_cfs_qmc) == 1
    assert abs(islb_cfs_cpp[0] - islb_cfs_qmc[0]) * 1e4 < 15.0


def test_cpp23_platen_sde_aad_vibrato_pre_and_clamping() -> None:
    """Verify PlatenSdeAadArgs Vibrato pre-state saving, clamping, and AVX2/scalar backward AAD."""
    backend = tb.get_tier1_backend()
    assert isinstance(backend, tb.Cpp23Tier1Backend)
    abi_v2 = backend.abi_info_v2()
    assert int(abi_v2.abi_version) == 20261125
    assert int(abi_v2.platen_sde_aad_args_size) == ctypes.sizeof(tb.PlatenSdeAadArgsC) == 272

    rng = np.random.default_rng(20261125)
    d_assets, m_rem, n_sub, eff_r, n_rep = 3, 3, 4, 2, 10  # n_rep=10 exercises AVX2 + scalar tail
    n_total = eff_r * n_rep
    lv_nt, lv_ny = 16, 24

    y_min = np.array([-1.4, -1.2, -1.5], dtype=np.float64)
    y_max = np.array([1.1, 1.0, 1.2], dtype=np.float64)
    lv_t_grid = np.linspace(1e-5, 1.5, lv_nt, dtype=np.float64)
    lv_tables = np.empty((d_assets, lv_nt, lv_ny), dtype=np.float64)
    for d in range(d_assets):
        yg = np.linspace(y_min[d], y_max[d], lv_ny)
        lv_tables[d] = 0.22 + 0.03 * lv_t_grid[:, None] + 0.08 * (yg[None, :] ** 2) - 0.02 * yg[None, :]

    corr = np.full((d_assets, d_assets), 0.3, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    chol = np.ascontiguousarray(np.linalg.cholesky(corr), dtype=np.float64)
    z_indep = rng.standard_normal((m_rem, n_sub, eff_r, n_rep, d_assets))
    tau_obs = np.array([0.4, 0.95, 1.5], dtype=np.float64)
    r_nat = np.array([[0.01, 0.025, 0.04], [0.015, 0.03, 0.045], [0.005, 0.015, 0.03]], dtype=np.float64)
    r_tot = r_nat + np.array([[0.002, 0.004, 0.006], [0.0, 0.001, 0.002], [-0.001, 0.0, 0.001]], dtype=np.float64)

    def _np_lv_bilinear(lv_d: npt.NDArray[np.float64], t_val: float, y_vec: npt.NDArray[np.float64], ym_lo: float, ym_hi: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        inv_dt = (lv_nt - 1) / max(float(lv_t_grid[-1]), 1e-12)
        inv_dy = (lv_ny - 1) / max(ym_hi - ym_lo, 1e-12)
        ut = float(np.clip(t_val * inv_dt, 0.0, (lv_nt - 1) - 1e-6))
        it0 = int(ut)
        it1 = min(it0 + 1, lv_nt - 1)
        wt = ut - it0
        uy_raw = (y_vec - ym_lo) * inv_dy
        in_y = (uy_raw > 0.0) & (uy_raw < (lv_ny - 1) - 1e-6)
        uy = np.clip(uy_raw, 0.0, (lv_ny - 1) - 1e-6)
        iy0 = uy.astype(np.int64)
        iy1 = np.minimum(iy0 + 1, lv_ny - 1)
        wy = uy - iy0
        v0 = (1.0 - wt) * lv_d[it0, iy0] + wt * lv_d[it1, iy0]
        v1 = (1.0 - wt) * lv_d[it0, iy1] + wt * lv_d[it1, iy1]
        val = (1.0 - wy) * v0 + wy * v1
        dy = np.where(in_y, (v1 - v0) * inv_dy, 0.0)
        return val, dy

    def _np_ref_vibrato(vib_p: int, vib_s: int, period_level: bool) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lx_per = np.zeros((m_rem, d_assets, n_total), dtype=np.float64)
        out_lx = np.zeros((d_assets, n_total), dtype=np.float64)
        out_j = np.ones((d_assets, n_total), dtype=np.float64)
        out_sig = np.zeros((d_assets, n_total), dtype=np.float64)
        z_corr = np.einsum("msrnd,kd->msrnk", z_indep, chol)
        for d in range(d_assets):
            for r in range(eff_r):
                lx = np.zeros(n_rep, dtype=np.float64)
                jlx = np.ones(n_rep, dtype=np.float64)
                t_p, rt_p, rn_p = 0.0, 0.0, 0.0
                for m in range(m_rem):
                    tc = float(tau_obs[m])
                    dtau = max(tc - t_p, 1e-10)
                    dt = dtau / n_sub
                    sdt = math.sqrt(dt)
                    rtc, rnc = float(r_tot[d, m]), float(r_nat[d, m])
                    if period_level and m == vib_p:
                        sl = slice(r * n_rep, (r + 1) * n_rep)
                        out_lx[d, sl] = lx
                        out_j[d, sl] = jlx
                        acc_sq = np.zeros(n_rep, dtype=np.float64)
                        for sk in range(n_sub):
                            ak = (sk + 0.5) / n_sub
                            tm = t_p + ak * dtau
                            qk = (rt_p + ak * (rtc - rt_p)) - (rn_p + ak * (rnc - rn_p))
                            sigk, _ = _np_lv_bilinear(lv_tables[d], tm, lx + qk, float(y_min[d]), float(y_max[d]))
                            acc_sq += sigk * sigk
                        out_sig[d, sl] = np.sqrt(acc_sq / n_sub)
                    for s in range(n_sub):
                        al = (s + 0.5) / n_sub
                        tm = t_p + al * dtau
                        ts = max(tm - 0.5 * dt, 0.0)
                        te = ts + dt
                        qm = (rt_p + al * (rtc - rt_p)) - (rn_p + al * (rnc - rn_p))
                        if (not period_level) and m == vib_p and s == vib_s:
                            sl = slice(r * n_rep, (r + 1) * n_rep)
                            out_lx[d, sl] = lx
                            out_j[d, sl] = jlx
                            sv, _ = _np_lv_bilinear(lv_tables[d], tm, lx + qm, float(y_min[d]), float(y_max[d]))
                            out_sig[d, sl] = sv
                        zd = z_corr[m, s, r, :, d]
                        dw = sdt * zd
                        s0, d0 = _np_lv_bilinear(lv_tables[d], ts, lx + qm, float(y_min[d]), float(y_max[d]))
                        a0 = -0.5 * s0 * s0
                        by = lx + a0 * dt + qm
                        sh, dh = _np_lv_bilinear(lv_tables[d], te, by + s0 * dw, float(y_min[d]), float(y_max[d]))
                        sp, dp = _np_lv_bilinear(lv_tables[d], te, by + s0 * sdt, float(y_min[d]), float(y_max[d]))
                        sm, dm = _np_lv_bilinear(lv_tables[d], te, by - s0 * sdt, float(y_min[d]), float(y_max[d]))
                        lx = lx + 0.5 * (a0 - 0.5 * sh * sh) * dt + 0.25 * (sp + sm + 2.0 * s0) * dw + 0.25 * (sp - sm) * sdt * (zd * zd - 1.0)
                        da0 = -s0 * d0
                        step_j = (
                            1.0
                            + 0.5 * (da0 - sh * dh * (1.0 + da0 * dt + d0 * dw)) * dt
                            + 0.25 * (dp * (1.0 + da0 * dt + d0 * sdt) + dm * (1.0 + da0 * dt - d0 * sdt) + 2.0 * d0) * dw
                            + 0.25 * (dp * (1.0 + da0 * dt + d0 * sdt) - dm * (1.0 + da0 * dt - d0 * sdt)) * sdt * (zd * zd - 1.0)
                        )
                        jlx = jlx * step_j
                    ml = float(np.max(lx))
                    ex = np.exp(lx - ml)
                    mr = max(float(np.mean(ex)), 1e-30)
                    om = ex / (n_rep * mr)
                    lx = lx - (ml + math.log(mr))
                    lx_per[m, d, r * n_rep : (r + 1) * n_rep] = lx
                    jlx = 1.0 + ((jlx - 1.0) - float(np.sum(om * (jlx - 1.0))))
                    t_p, rt_p, rn_p = tc, rtc, rnc
        return lx_per, out_lx, out_j, out_sig

    # 1. Substep-level Vibrato (save_vib_pre=1) + backwards compat with save_vib_pre=0
    lx0, nx0, _, _, _ = backend.crn_platen_sde_aad_cpu(
        lv_tables=lv_tables, z_indep=z_indep, chol_base=chol, tau_obs=tau_obs,
        r_native_obs=r_nat, r_tot_obs=r_tot, lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
        n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
    )
    out_lx = np.empty((d_assets, n_total), dtype=np.float64)
    out_j = np.empty((d_assets, n_total), dtype=np.float64)
    out_sig = np.empty((d_assets, n_total), dtype=np.float64)
    lx1, nx1, _, _, _ = backend.crn_platen_sde_aad_cpu(
        lv_tables=lv_tables, z_indep=z_indep, chol_base=chol, tau_obs=tau_obs,
        r_native_obs=r_nat, r_tot_obs=r_tot, lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
        n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
        out_lx_pre=out_lx, out_j_pre=out_j, out_sig_pre=out_sig,
        vib_period_idx=m_rem - 1, vib_sub_idx=n_sub - 1, save_vib_pre=1,
    )
    assert lx0 is not None and nx0 is not None and lx1 is not None and nx1 is not None
    np.testing.assert_allclose(lx1, lx0, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(nx1, nx0, atol=1e-14, rtol=1e-14)
    ref_lxp, ref_lx, ref_j, ref_sig = _np_ref_vibrato(m_rem - 1, n_sub - 1, period_level=False)
    np.testing.assert_allclose(lx1, ref_lxp, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out_lx, ref_lx, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out_j, ref_j, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out_sig, ref_sig, atol=1e-12, rtol=1e-12)

    # 2. Period-level Vibrato (save_vib_pre=2, matching M=1 GPU VIB_PERIOD_LEVEL=True)
    out_lx_p = np.empty((d_assets, n_total), dtype=np.float64)
    out_j_p = np.empty((d_assets, n_total), dtype=np.float64)
    out_sig_p = np.empty((d_assets, n_total), dtype=np.float64)
    backend.crn_platen_sde_aad_cpu(
        lv_tables=lv_tables, z_indep=z_indep[:1], chol_base=chol, tau_obs=tau_obs[:1],
        r_native_obs=r_nat[:, :1], r_tot_obs=r_tot[:, :1], lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
        n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
        out_lx_pre=out_lx_p, out_j_pre=out_j_p, out_sig_pre=out_sig_p,
        vib_period_idx=0, vib_sub_idx=n_sub - 1, save_vib_pre=2,
    )
    np.testing.assert_allclose(out_lx_p, 0.0, atol=1e-15)
    np.testing.assert_allclose(out_j_p, 1.0, atol=1e-15)
    # Verify RMS sigma against period-0 reference
    acc_sq0 = np.zeros((d_assets, n_total), dtype=np.float64)
    for d in range(d_assets):
        for sk in range(n_sub):
            ak = (sk + 0.5) / n_sub
            tm = ak * float(tau_obs[0])
            qk = ak * (float(r_tot[d, 0]) - float(r_nat[d, 0]))
            sigk, _ = _np_lv_bilinear(lv_tables[d], tm, np.full(n_total, qk), float(y_min[d]), float(y_max[d]))
            acc_sq0[d] += sigk * sigk
    np.testing.assert_allclose(out_sig_p, np.sqrt(acc_sq0 / n_sub), atol=1e-12, rtol=1e-12)

    # 3. Steep-cliff LV default clamping test (|dsig_dy| <= 4.0, step_j in [0.25, 2.5], j_lx in [-6.0, 6.0])
    lv_cliff = lv_tables.copy()
    lv_cliff[0, :, 12:] += 1.2
    out_lx_c = np.empty((d_assets, n_total), dtype=np.float64)
    out_j_c = np.empty((d_assets, n_total), dtype=np.float64)
    out_sig_c = np.empty((d_assets, n_total), dtype=np.float64)
    backend.crn_platen_sde_aad_cpu(
        lv_tables=lv_cliff, z_indep=z_indep, chol_base=chol, tau_obs=tau_obs,
        r_native_obs=r_nat, r_tot_obs=r_tot, lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
        n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
        out_lx_pre=out_lx_c, out_j_pre=out_j_c, out_sig_pre=out_sig_c,
        vib_period_idx=m_rem - 1, vib_sub_idx=n_sub - 1, save_vib_pre=1,
    )
    assert np.all(np.isfinite(out_j_c))
    assert np.all(np.abs(out_j_c) <= 6.0 + 1e-12)
    # Explicit override bounds
    backend.crn_platen_sde_aad_cpu(
        lv_tables=lv_cliff, z_indep=z_indep, chol_base=chol, tau_obs=tau_obs,
        r_native_obs=r_nat, r_tot_obs=r_tot, lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
        n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
        out_lx_pre=out_lx_c, out_j_pre=out_j_c, out_sig_pre=out_sig_c,
        vib_period_idx=m_rem - 1, vib_sub_idx=n_sub - 1, save_vib_pre=1,
        vib_j_pre_max=4.0, vib_dsig_dy_max=4.0,
    )
    assert np.all(np.isfinite(out_j_c))
    assert np.all(np.abs(out_j_c) <= 4.0 + 1e-12)

    # 4. AVX2 + scalar-tail backward AAD accuracy check vs central FD (LV, Cholesky, q_shift)
    bar_post = rng.standard_normal((m_rem, d_assets, n_total))
    q_shift_sub = rng.normal(scale=0.01, size=(d_assets, m_rem, n_sub))
    lx_b, _, _, _, _ = backend.crn_platen_sde_aad_cpu(
        lv_tables=lv_tables, z_indep=z_indep, chol_base=chol, tau_obs=tau_obs,
        r_native_obs=None, r_tot_obs=None, q_shift_sub=q_shift_sub,
        lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
        n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
    )
    _, _, bar_lv, bar_chol, bar_q = backend.crn_platen_sde_aad_cpu(
        lv_tables=lv_tables, z_indep=z_indep, chol_base=chol, tau_obs=tau_obs,
        r_native_obs=None, r_tot_obs=None, q_shift_sub=q_shift_sub,
        lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
        n_sub=n_sub, eff_r=eff_r, n_rep=n_rep,
        bar_lx_post=bar_post, lx_norm_in=lx_b,
        run_forward=False, run_backward=True, compute_lv_aad=True, compute_chol_aad=True, compute_q_aad=True,
    )
    assert bar_lv is not None and bar_chol is not None and bar_q is not None
    eps_fd = 1e-6
    # Check Cholesky AAD
    for i in range(d_assets):
        for j in range(i + 1):
            cp = chol.copy()
            cp[i, j] += eps_fd
            cm = chol.copy()
            cm[i, j] -= eps_fd
            lxp, _, _, _, _ = backend.crn_platen_sde_aad_cpu(
                lv_tables=lv_tables, z_indep=z_indep, chol_base=cp, tau_obs=tau_obs,
                r_native_obs=None, r_tot_obs=None, q_shift_sub=q_shift_sub,
                lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
                n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
            )
            lxm, _, _, _, _ = backend.crn_platen_sde_aad_cpu(
                lv_tables=lv_tables, z_indep=z_indep, chol_base=cm, tau_obs=tau_obs,
                r_native_obs=None, r_tot_obs=None, q_shift_sub=q_shift_sub,
                lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
                n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
            )
            assert lxp is not None and lxm is not None
            fd = float(np.sum(bar_post * (lxp - lxm))) / (2.0 * eps_fd)
            np.testing.assert_allclose(bar_chol[i, j], fd, atol=1e-7, rtol=1e-7)
    # Check q_shift AAD and selected interior LV knots
    q_fd = np.zeros_like(bar_q)
    for d in range(d_assets):
        for ms in range(m_rem * n_sub):
            qp = q_shift_sub.reshape(d_assets, m_rem * n_sub).copy()
            qp[d, ms] += eps_fd
            qm = q_shift_sub.reshape(d_assets, m_rem * n_sub).copy()
            qm[d, ms] -= eps_fd
            lxp, _, _, _, _ = backend.crn_platen_sde_aad_cpu(
                lv_tables=lv_tables, z_indep=z_indep, chol_base=chol, tau_obs=tau_obs,
                r_native_obs=None, r_tot_obs=None, q_shift_sub=qp,
                lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
                n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
            )
            lxm, _, _, _, _ = backend.crn_platen_sde_aad_cpu(
                lv_tables=lv_tables, z_indep=z_indep, chol_base=chol, tau_obs=tau_obs,
                r_native_obs=None, r_tot_obs=None, q_shift_sub=qm,
                lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
                n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
            )
            assert lxp is not None and lxm is not None
            q_fd[d, ms] = float(np.sum(bar_post * (lxp - lxm))) / (2.0 * eps_fd)
    np.testing.assert_allclose(bar_q, q_fd, atol=1e-7, rtol=1e-7)
    for d, it, iy in ((0, 5, 11), (1, 8, 12), (2, 11, 13)):
        lp = lv_tables.copy()
        lp[d, it, iy] += eps_fd
        lm = lv_tables.copy()
        lm[d, it, iy] -= eps_fd
        lxp, _, _, _, _ = backend.crn_platen_sde_aad_cpu(
            lv_tables=lp, z_indep=z_indep, chol_base=chol, tau_obs=tau_obs,
            r_native_obs=None, r_tot_obs=None, q_shift_sub=q_shift_sub,
            lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
            n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
        )
        lxm, _, _, _, _ = backend.crn_platen_sde_aad_cpu(
            lv_tables=lm, z_indep=z_indep, chol_base=chol, tau_obs=tau_obs,
            r_native_obs=None, r_tot_obs=None, q_shift_sub=q_shift_sub,
            lv_t_grid=lv_t_grid, lv_y_min=y_min, lv_y_max=y_max,
            n_sub=n_sub, eff_r=eff_r, n_rep=n_rep, run_forward=True, run_backward=False,
        )
        assert lxp is not None and lxm is not None
        fd_lv = float(np.sum(bar_post * (lxp - lxm))) / (2.0 * eps_fd)
        np.testing.assert_allclose(bar_lv[d, it, iy], fd_lv, atol=1e-7, rtol=1e-7)

    # 5. End-to-end zero-copy wiring of (out_lx, out_j, out_sig) into crn_eval_vibrato_gamma_cpu
    l_inv_t = np.ascontiguousarray(np.linalg.inv(chol).T, dtype=np.float64)
    c_inv = np.ascontiguousarray(np.linalg.inv(corr), dtype=np.float64)
    sv_vib = np.ascontiguousarray(np.transpose(z_indep.reshape(m_rem * n_sub, n_total, d_assets), (0, 2, 1)), dtype=np.float64)
    h_vib, frozen = tb.crn_eval_vibrato_gamma_cpu(
        perf_mat=nx1,
        lx_pre=out_lx,
        j_pre=out_j,
        sig_pre=out_sig,
        sv_vib=sv_vib,
        chol_mat=chol,
        l_inv_t=l_inv_t,
        c_inv=c_inv,
        fwd_term=np.ones(d_assets, dtype=np.float64),
        cdl_m=np.ones((m_rem, d_assets), dtype=np.float64),
        cdl2_last=np.zeros(d_assets, dtype=np.float64),
        ko_barriers=np.array([1.05, 1.05, 1.05], dtype=np.float64),
        is_callable=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        cpn_barriers=np.array([0.85, 0.85, 0.85], dtype=np.float64),
        is_guaranteed_cpn=np.zeros(m_rem, dtype=np.float64),
        coupon_rates=np.full(m_rem, 0.05, dtype=np.float64),
        eps_vec=np.full(m_rem, 0.04, dtype=np.float64),
        inv_lex_perm=np.arange(d_assets, dtype=np.int64),
        vib_step_start=m_rem * n_sub - 1,
        vib_n_steps=1,
        df_last=0.92,
        h_vib=float((tau_obs[-1] - tau_obs[-2]) / n_sub),
        ko_spread=-0.05,
        cpn_spread=-0.05,
        ki_spread=-0.05,
        one_star_strike=0.0,
        one_star_eps=0.02,
        strike=0.90,
        lower_strike=0.0,
        ki_barrier=0.80,
        option_gearing=1.0,
        autocall_coupon=0.0,
        c_mem_0=0.0,
        i_ki_past=0.0,
        part_call_strike=0.0,
        part_call_rate=0.0,
        one_star_coupon=0.0,
        call_code=0,
        cpn_code=0,
        opt_code=0,
        has_ki=True,
        ki_active=True,
        has_one_star=False,
        is_memory=True,
        is_long_basket=False,
        no_notional=False,
        force_cpn_ac=False,
        cpn_dir_down=False,
        cpn_type_out=False,
        has_lower_strike=False,
        has_part_call=False,
        emit_frozen_term=True,
    )
    assert h_vib.shape == (d_assets, d_assets) and np.all(np.isfinite(h_vib))
    assert frozen is not None and frozen.shape == (4, n_total) and np.all(np.isfinite(frozen))

    # Verify crn_eval_vibrato_gamma_cpp clamps j_pre to [-8.0, 8.0]
    j_huge = np.full_like(out_j, 50.0)
    j_clamped = np.full_like(out_j, 8.0)
    h_huge, _ = tb.crn_eval_vibrato_gamma_cpu(
        perf_mat=nx1, lx_pre=out_lx, j_pre=j_huge, sig_pre=out_sig, sv_vib=sv_vib,
        chol_mat=chol, l_inv_t=l_inv_t, c_inv=c_inv, fwd_term=np.ones(d_assets, dtype=np.float64),
        cdl_m=np.ones((m_rem, d_assets), dtype=np.float64), cdl2_last=np.zeros(d_assets, dtype=np.float64),
        ko_barriers=np.array([1.05, 1.05, 1.05], dtype=np.float64),
        is_callable=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        cpn_barriers=np.array([0.85, 0.85, 0.85], dtype=np.float64),
        is_guaranteed_cpn=np.zeros(m_rem, dtype=np.float64),
        coupon_rates=np.full(m_rem, 0.05, dtype=np.float64),
        eps_vec=np.full(m_rem, 0.04, dtype=np.float64),
        inv_lex_perm=np.arange(d_assets, dtype=np.int64),
        vib_step_start=m_rem * n_sub - 1, vib_n_steps=1, df_last=0.92,
        h_vib=float((tau_obs[-1] - tau_obs[-2]) / n_sub),
        ko_spread=-0.05, cpn_spread=-0.05, ki_spread=-0.05,
        one_star_strike=0.0, one_star_eps=0.02, strike=0.90, lower_strike=0.0,
        ki_barrier=0.80, option_gearing=1.0, autocall_coupon=0.0, c_mem_0=0.0, i_ki_past=0.0,
        part_call_strike=0.0, part_call_rate=0.0, one_star_coupon=0.0,
        call_code=0, cpn_code=0, opt_code=0,
        has_ki=True, ki_active=True, has_one_star=False, is_memory=True,
        is_long_basket=False, no_notional=False, force_cpn_ac=False,
        cpn_dir_down=False, cpn_type_out=False, has_lower_strike=False, has_part_call=False,
        emit_frozen_term=False,
    )
    h_clamp, _ = tb.crn_eval_vibrato_gamma_cpu(
        perf_mat=nx1, lx_pre=out_lx, j_pre=j_clamped, sig_pre=out_sig, sv_vib=sv_vib,
        chol_mat=chol, l_inv_t=l_inv_t, c_inv=c_inv, fwd_term=np.ones(d_assets, dtype=np.float64),
        cdl_m=np.ones((m_rem, d_assets), dtype=np.float64), cdl2_last=np.zeros(d_assets, dtype=np.float64),
        ko_barriers=np.array([1.05, 1.05, 1.05], dtype=np.float64),
        is_callable=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        cpn_barriers=np.array([0.85, 0.85, 0.85], dtype=np.float64),
        is_guaranteed_cpn=np.zeros(m_rem, dtype=np.float64),
        coupon_rates=np.full(m_rem, 0.05, dtype=np.float64),
        eps_vec=np.full(m_rem, 0.04, dtype=np.float64),
        inv_lex_perm=np.arange(d_assets, dtype=np.int64),
        vib_step_start=m_rem * n_sub - 1, vib_n_steps=1, df_last=0.92,
        h_vib=float((tau_obs[-1] - tau_obs[-2]) / n_sub),
        ko_spread=-0.05, cpn_spread=-0.05, ki_spread=-0.05,
        one_star_strike=0.0, one_star_eps=0.02, strike=0.90, lower_strike=0.0,
        ki_barrier=0.80, option_gearing=1.0, autocall_coupon=0.0, c_mem_0=0.0, i_ki_past=0.0,
        part_call_strike=0.0, part_call_rate=0.0, one_star_coupon=0.0,
        call_code=0, cpn_code=0, opt_code=0,
        has_ki=True, ki_active=True, has_one_star=False, is_memory=True,
        is_long_basket=False, no_notional=False, force_cpn_ac=False,
        cpn_dir_down=False, cpn_type_out=False, has_lower_strike=False, has_part_call=False,
        emit_frozen_term=False,
    )
    np.testing.assert_allclose(h_huge, h_clamp, atol=1e-14, rtol=1e-14)


def test_cpp23_step_spot_bumps_matches_step_1d_exact() -> None:
    """Verify batched crn_step_spot_bumps_cpu matches per-asset crn_step_1d_trajectories_cpp to < 1e-13."""
    backend = tb.get_tier1_backend()
    assert isinstance(backend, tb.Cpp23Tier1Backend)
    step_fn, _ = _get_cpp_crn_fns()
    assert step_fn is not None

    rng = np.random.default_rng(77123)
    d_assets, m_rem, n_sub, eff_r, n_rep = 4, 5, 3, 2, 13
    n_total = eff_r * n_rep
    lv_nt, lv_ny = 20, 32

    y_min = np.array([-1.6, -1.4, -1.5, -1.3], dtype=np.float64)
    y_max = np.array([1.0, 1.1, 0.9, 1.2], dtype=np.float64)
    lv_t_grid = np.linspace(1e-5, 2.5, lv_nt, dtype=np.float64)
    tau_obs = np.linspace(0.5, 2.5, m_rem, dtype=np.float64)
    r_nat = rng.uniform(0.0, 0.08, size=(d_assets, m_rem))
    r_tot = r_nat + rng.uniform(-0.01, 0.01, size=(d_assets, m_rem))
    z_corr = rng.standard_normal((m_rem, n_sub, eff_r, n_rep, d_assets))

    lv_pm = np.empty((d_assets, 2, lv_nt, lv_ny), dtype=np.float64)
    for d in range(d_assets):
        yg = np.linspace(y_min[d], y_max[d], lv_ny)
        base = 0.24 + 0.02 * lv_t_grid[:, None] + 0.05 * (yg[None, :] ** 2)
        lv_pm[d, 0] = base * 1.015
        lv_pm[d, 1] = base * 0.985

    out_batched = tb.crn_step_spot_bumps_cpu(
        lv_tables_pm=lv_pm,
        z_corr=z_corr,
        tau_obs=tau_obs,
        r_native_obs=r_nat,
        r_tot_obs=r_tot,
        lv_t_grid=lv_t_grid,
        lv_y_min=y_min,
        lv_y_max=y_max,
        n_sub=n_sub,
        eff_r=eff_r,
        n_rep=n_rep,
    )
    assert out_batched.shape == (2, d_assets, m_rem, n_total)

    ref_pm = np.empty((2, d_assets, m_rem, n_total), dtype=np.float64)
    for d in range(d_assets):
        z_d = np.ascontiguousarray(z_corr[..., d], dtype=np.float64)
        out_d = np.empty((2, m_rem, n_total), dtype=np.float64)
        step_fn(
            2,
            lv_nt,
            lv_ny,
            m_rem,
            n_sub,
            eff_r,
            n_rep,
            0,
            float(y_min[d]),
            float(y_max[d]),
            lv_pm[d].ctypes.data,
            z_d.ctypes.data,
            tau_obs.ctypes.data,
            r_nat[d].ctypes.data,
            r_tot[d].ctypes.data,
            lv_t_grid.ctypes.data,
            out_d.ctypes.data,
        )
        ref_pm[0, d] = out_d[0]
        ref_pm[1, d] = out_d[1]

    np.testing.assert_allclose(out_batched, ref_pm, atol=1e-13, rtol=1e-13)
