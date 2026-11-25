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
