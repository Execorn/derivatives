from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from deepvol.surrogates.eq_autocall_egno import (
    AutocallEdgeCol,
    AutocallGlobalCol,
    AutocallNodeCol,
    AutocallSchedCol,
    MaskedEQAutocallSurrogate,
    assemble_surrogate_product_value,
)


def _ndtr_tensor(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


def _build_synthetic_autocall_batch(
    batch_size: int,
    d_max: int = 8,
    m_max: int = 48,
    seed: int = 1234,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    node_feats = torch.zeros(batch_size, d_max, 14, dtype=torch.float64)
    edge_feats = torch.zeros(batch_size, d_max, d_max, 4, dtype=torch.float64)
    asset_mask = torch.zeros(batch_size, d_max, dtype=torch.float64)
    sched_feats = torch.zeros(batch_size, m_max, 12, dtype=torch.float64)
    sched_mask = torch.zeros(batch_size, m_max, dtype=torch.float64)
    global_feats = torch.zeros(batch_size, 10, dtype=torch.float64)
    target_pv = torch.zeros(batch_size, dtype=torch.float64)

    for b in range(batch_size):
        d_active = (b % d_max) + 1
        m_rem = min(max(4 + (b * 3) % 16, 4), m_max)

        asset_mask[b, :d_active] = 1.0
        sched_mask[b, :m_rem] = 1.0

        log_m0 = -0.35 + 0.55 * torch.rand(d_active, generator=gen, dtype=torch.float64)
        carry_1y = 0.03 + 0.08 * torch.rand(
            d_active, generator=gen, dtype=torch.float64
        )
        div_lev = 0.02 * torch.rand(d_active, generator=gen, dtype=torch.float64)
        repo_avg = 0.005 * torch.rand(d_active, generator=gen, dtype=torch.float64)
        iv_short = 0.22 + 0.25 * torch.rand(
            d_active, generator=gen, dtype=torch.float64
        )
        iv_slope = -0.04 + 0.06 * torch.rand(
            d_active, generator=gen, dtype=torch.float64
        )
        iv_term = (iv_short + iv_slope).clamp(min=0.10)
        ssvi_rho = -0.60 + 0.45 * torch.rand(
            d_active, generator=gen, dtype=torch.float64
        )
        ssvi_eta = 0.30 + 0.40 * torch.rand(
            d_active, generator=gen, dtype=torch.float64
        )
        pc1 = 0.02 * torch.randn(d_active, generator=gen, dtype=torch.float64)
        pc2 = 0.01 * torch.randn(d_active, generator=gen, dtype=torch.float64)

        k_put = 0.70 + 0.30 * float(
            torch.rand(1, generator=gen, dtype=torch.float64).item()
        )
        tau_term = 0.25 * m_rem
        v_term = iv_term * math.sqrt(tau_term)
        log_fp_term = log_m0 + carry_1y * tau_term
        dist_put = (log_fp_term - math.log(k_put)) / v_term
        soft_w = torch.softmax(-log_fp_term / 0.15, dim=0)

        node_feats[b, :d_active, AutocallNodeCol.LOG_MONEYNESS] = log_m0
        node_feats[b, :d_active, AutocallNodeCol.LOG_FWD_CARRY_1Y] = carry_1y
        node_feats[b, :d_active, AutocallNodeCol.CASH_DIV_LEVERAGE] = div_lev
        node_feats[b, :d_active, AutocallNodeCol.REPO_SPREAD_AVG] = repo_avg
        node_feats[b, :d_active, AutocallNodeCol.IV_SHORT_ATM] = iv_short
        node_feats[b, :d_active, AutocallNodeCol.IV_TERM_ATM] = iv_term
        node_feats[b, :d_active, AutocallNodeCol.IV_TERM_SLOPE] = iv_slope
        node_feats[b, :d_active, AutocallNodeCol.SSVI_SKEW_RHO] = ssvi_rho
        node_feats[b, :d_active, AutocallNodeCol.SSVI_CURV_ETA] = ssvi_eta
        node_feats[b, :d_active, AutocallNodeCol.SMILE_RESIDUAL_PC1] = pc1
        node_feats[b, :d_active, AutocallNodeCol.SMILE_RESIDUAL_PC2] = pc2
        node_feats[b, :d_active, AutocallNodeCol.DIST_TO_PUT_STRIKE] = dist_put
        node_feats[b, :d_active, AutocallNodeCol.PROXY_DELTA_WEIGHT] = soft_w
        node_feats[b, :d_active, AutocallNodeCol.ASSET_ACTIVE_MASK] = 1.0

        f_factor = torch.randn(d_active, 3, generator=gen, dtype=torch.float64)
        cov_raw = f_factor @ f_factor.T + 0.4 * torch.eye(
            d_active, dtype=torch.float64
        )
        std_raw = torch.sqrt(torch.diag(cov_raw)).unsqueeze(1)
        corr_mat = (cov_raw / (std_raw @ std_raw.T)).clamp(-0.99, 0.99)
        corr_mat.fill_diagonal_(1.0)

        for i in range(d_active):
            for j in range(d_active):
                if i != j:
                    edge_feats[b, i, j, AutocallEdgeCol.CORR_RHO] = corr_mat[i, j]
                    edge_feats[b, i, j, AutocallEdgeCol.COVARIANCE_TERM] = (
                        corr_mat[i, j] * v_term[i] * v_term[j]
                    )
                    edge_feats[b, i, j, AutocallEdgeCol.LOG_PERF_DIFF] = abs(
                        float(log_fp_term[i] - log_fp_term[j])
                    )
                    edge_feats[b, i, j, AutocallEdgeCol.OFF_DIAG_MASK] = 1.0

        min_log_m0 = float(log_m0.min().item())
        mean_iv = float(iv_term.mean().item())
        mean_skew = float(ssvi_rho.mean().item())
        mean_corr = (
            float(
                (corr_mat.sum() - d_active)
                / max(d_active * (d_active - 1), 1)
            )
            if d_active > 1
            else 1.0
        )

        surv_proxy = 1.0
        surv_true = 1.0
        p_true_list = []
        q_true_list = []
        df_list = []
        cpn_list = []

        for m in range(m_rem):
            tau_m = 0.25 * (m + 1)
            dtau = 0.25
            df_m = math.exp(-0.12 * tau_m)
            is_callable = 0.0 if (m == 0 and b % 3 == 0) else 1.0
            b_ko = 1.00 - 0.01 * m
            log_ko = math.log(min(b_ko, 3.0))
            is_guar = 1.0 if (m == 0 and b % 4 == 0) else 0.0
            b_cpn = 1e-5 if is_guar > 0.5 else 0.75
            log_cpn = math.log(max(b_cpn, 0.05))
            cpn_rate = 0.035

            stulz_log_fw = min_log_m0 + 0.04 * tau_m - 0.04 * math.log(d_active)
            stulz_vw = max(
                mean_iv * math.sqrt(tau_m) * (0.85 + 0.15 * mean_corr), 0.05
            )

            z_ko = (stulz_log_fw - log_ko - 0.5 * stulz_vw * stulz_vw) / stulz_vw
            h_proxy_m = float(
                _ndtr_tensor(torch.tensor(z_ko, dtype=torch.float64)).item()
            )
            h_proxy_m = min(max(h_proxy_m, 1e-4), 0.95) * is_callable
            p_proxy_m = surv_proxy * h_proxy_m
            surv_proxy = surv_proxy * (1.0 - h_proxy_m)

            z_cpn = (stulz_log_fw - log_cpn - 0.5 * stulz_vw * stulz_vw) / stulz_vw
            q_proxy_m = (
                1.0
                if is_guar > 0.5
                else float(
                    _ndtr_tensor(torch.tensor(z_cpn, dtype=torch.float64)).item()
                )
            )
            q_proxy_m = min(max(q_proxy_m, 1e-4), 1.0 - 1e-4)

            # Ground-truth skew/dispersion adjustment (~3-8 bps PV effect)
            skew_adj = 0.04 * mean_skew - 0.03 * (1.0 - mean_corr)
            logit_h_true = math.log(
                max(h_proxy_m, 1e-6) / max(1.0 - h_proxy_m, 1e-6)
            ) + skew_adj
            h_true_m = (
                is_callable / (1.0 + math.exp(-logit_h_true))
                if is_callable > 0.5
                else 0.0
            )
            p_true_m = surv_true * h_true_m
            surv_true = surv_true * (1.0 - h_true_m)

            logit_q_true = (
                math.log(q_proxy_m / (1.0 - q_proxy_m)) + 0.8 * skew_adj
            )
            q_true_m = (
                1.0
                if is_guar > 0.5
                else 1.0 / (1.0 + math.exp(-logit_q_true))
            )

            sched_feats[b, m, AutocallSchedCol.TAU_OBS] = tau_m
            sched_feats[b, m, AutocallSchedCol.DELTA_TAU] = dtau
            sched_feats[b, m, AutocallSchedCol.DF_PAY] = df_m
            sched_feats[b, m, AutocallSchedCol.LOG_KO_BARRIER] = log_ko
            sched_feats[b, m, AutocallSchedCol.IS_CALLABLE_FLAG] = is_callable
            sched_feats[b, m, AutocallSchedCol.LOG_CPN_BARRIER] = log_cpn
            sched_feats[b, m, AutocallSchedCol.IS_GUARANTEED_CPN] = is_guar
            sched_feats[b, m, AutocallSchedCol.COUPON_RATE] = cpn_rate
            sched_feats[b, m, AutocallSchedCol.STULZ_FWD_W] = stulz_log_fw
            sched_feats[b, m, AutocallSchedCol.STULZ_VOL_W] = stulz_vw
            sched_feats[b, m, AutocallSchedCol.PROXY_KO_PROB] = p_proxy_m
            sched_feats[b, m, AutocallSchedCol.PROXY_CPN_PROB] = q_proxy_m

            p_true_list.append(p_true_m)
            q_true_list.append(q_true_m)
            df_list.append(df_m)
            cpn_list.append(cpn_rate)

        fw_term = math.exp(
            float(sched_feats[b, m_rem - 1, AutocallSchedCol.STULZ_FWD_W].item())
        )
        vw_term = float(
            sched_feats[b, m_rem - 1, AutocallSchedCol.STULZ_VOL_W].item()
        )
        d1_p = (math.log(fw_term / k_put) + 0.5 * vw_term * vw_term) / vw_term
        d2_p = d1_p - vw_term
        put_bs = k_put * float(
            _ndtr_tensor(torch.tensor(-d2_p, dtype=torch.float64)).item()
        ) - fw_term * float(
            _ndtr_tensor(torch.tensor(-d1_p, dtype=torch.float64)).item()
        )
        l_put_proxy = min(max(surv_proxy * put_bs, 0.0), k_put * surv_proxy)
        l_put_true = min(
            max(
                l_put_proxy
                + surv_true * k_put * 0.002 * (-mean_skew + 0.2 * (1.0 - mean_corr)),
                0.0,
            ),
            k_put * surv_true,
        )

        global_feats[b, AutocallGlobalCol.K_PUT] = k_put
        global_feats[b, AutocallGlobalCol.B_KI] = 0.65
        global_feats[b, AutocallGlobalCol.HAS_KI_FLAG] = 1.0
        global_feats[b, AutocallGlobalCol.I_KI_PAST] = 0.0
        global_feats[b, AutocallGlobalCol.G_PUT] = 1.0
        global_feats[b, AutocallGlobalCol.C_MEM_0] = 0.02 if (b % 2 == 0) else 0.0
        global_feats[b, AutocallGlobalCol.IS_MEMORY_FLAG] = (
            1.0 if (b % 2 == 0) else 0.0
        )
        global_feats[b, AutocallGlobalCol.ABS_KO_SPREAD] = -0.02
        global_feats[b, AutocallGlobalCol.ABS_CPN_SPREAD] = -0.02
        global_feats[b, AutocallGlobalCol.L_PUT_PROXY] = l_put_proxy

        pv_notional = sum(df_list[m] * p_true_list[m] for m in range(m_rem)) + (
            df_list[-1] * surv_true
        )
        mem_buf = float(
            global_feats[b, AutocallGlobalCol.C_MEM_0].item()
            * global_feats[b, AutocallGlobalCol.IS_MEMORY_FLAG].item()
        )
        is_mem = float(global_feats[b, AutocallGlobalCol.IS_MEMORY_FLAG].item())
        s_prev = 1.0
        pv_cpn = 0.0
        for m in range(m_rem):
            c_eff = cpn_list[m] + mem_buf
            pv_cpn += df_list[m] * s_prev * q_true_list[m] * c_eff
            mem_buf = is_mem * (1.0 - q_true_list[m]) * c_eff
            s_prev -= p_true_list[m]
        pv_opt = -df_list[-1] * 1.0 * l_put_true
        target_pv[b] = pv_notional + pv_cpn + pv_opt

    return (
        node_feats.to(device=device, dtype=dtype),
        edge_feats.to(device=device, dtype=dtype),
        asset_mask.to(device=device, dtype=dtype),
        sched_feats.to(device=device, dtype=dtype),
        sched_mask.to(device=device, dtype=dtype),
        global_feats.to(device=device, dtype=dtype),
        target_pv.to(device=device, dtype=dtype),
    )


def test_G1_5_out_of_sample_surrogate_product_value_mae_across_d_1_to_8() -> None:
    torch.manual_seed(77)
    model = MaskedEQAutocallSurrogate(
        d_max=8,
        m_max=48,
        d_model=128,
        n_egno_layers=3,
        n_sched_layers=2,
        n_heads=4,
    )

    (
        node_tr,
        edge_tr,
        amask_tr,
        sched_tr,
        smask_tr,
        glob_tr,
        target_tr,
    ) = _build_synthetic_autocall_batch(batch_size=32, seed=1001, dtype=torch.float32)

    (
        node_te,
        edge_te,
        amask_te,
        sched_te,
        smask_te,
        glob_te,
        target_te,
    ) = _build_synthetic_autocall_batch(batch_size=32, seed=2002, dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(15):
        optimizer.zero_grad()
        out_tr = model(node_tr, edge_tr, amask_tr, sched_tr, smask_tr, glob_tr)
        pv_tr = model.assemble_product_value(out_tr, sched_tr, smask_tr, glob_tr)
        loss = torch.mean((pv_tr - target_tr) ** 2) + 1e-4 * (
            out_tr.delta_hazard_logits.pow(2).mean()
            + out_tr.delta_coupon_logits.pow(2).mean()
            + out_tr.delta_put_loss.pow(2).mean()
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out_te = model(node_te, edge_te, amask_te, sched_te, smask_te, glob_te)
        pv_te = model.assemble_product_value(out_te, sched_te, smask_te, glob_te)

    mae = (pv_te - target_te).abs().mean().item()
    assert mae <= 0.0015, f"Expected out-of-sample MAE <= 15 bps (0.0015), got {mae:.6f}"

    # Verify per-dimension MAE across all basket sizes D in [1, 8]
    active_counts = amask_te.sum(dim=1).to(torch.long)
    for d_val in range(1, 9):
        idx = active_counts == d_val
        assert idx.any()
        d_mae = (pv_te[idx] - target_te[idx]).abs().mean().item()
        assert d_mae <= 0.0015

    # Verify stopping simplex & put bounds
    assert (out_te.stopping_probs >= 0.0).all()
    assert (out_te.stopping_probs.sum(dim=1) <= 1.0 + 1e-6).all()
    assert (out_te.coupon_probs >= 0.0).all() and (out_te.coupon_probs <= 1.0).all()
    max_put = glob_te[:, AutocallGlobalCol.K_PUT] * out_te.survival_probs[:, -1]
    assert (out_te.put_loss >= 0.0).all()
    assert (out_te.put_loss <= max_put + 1e-6).all()


@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.float32, 1e-6), (torch.float64, 1e-13)],
)
def test_G1_6_sd_permutation_invariance_and_gradient_equivariance(
    dtype: torch.dtype,
    tol: float,
) -> None:
    torch.manual_seed(88)
    model = MaskedEQAutocallSurrogate(
        d_max=8,
        m_max=48,
        d_model=128,
        n_egno_layers=3,
        n_sched_layers=2,
        n_heads=4,
    ).to(dtype=dtype)
    model.eval()

    node_f, edge_f, amask, sched_f, smask, glob_f, _ = _build_synthetic_autocall_batch(
        batch_size=4, seed=3003, dtype=dtype
    )
    node_orig = node_f.detach().clone().requires_grad_(True)
    edge_orig = edge_f.detach().clone().requires_grad_(True)

    out_orig = model(node_orig, edge_orig, amask, sched_f, smask, glob_f)
    pv_orig = assemble_surrogate_product_value(out_orig, sched_f, smask, glob_f)
    grad_node_orig, grad_edge_orig = torch.autograd.grad(
        pv_orig.sum(), (node_orig, edge_orig), create_graph=True
    )
    hess_diag_node_orig = torch.autograd.grad(
        grad_node_orig.pow(2).sum(), node_orig
    )[0]

    perm = torch.tensor([2, 0, 4, 1, 3, 7, 5, 6], dtype=torch.long)
    inv_perm = torch.argsort(perm)

    node_perm = node_f.detach()[:, perm, :].clone().requires_grad_(True)
    edge_perm = edge_f.detach()[:, perm, :, :][:, :, perm, :].clone().requires_grad_(True)
    amask_perm = amask[:, perm]

    out_perm = model(node_perm, edge_perm, amask_perm, sched_f, smask, glob_f)
    pv_perm = assemble_surrogate_product_value(out_perm, sched_f, smask, glob_f)
    grad_node_perm, grad_edge_perm = torch.autograd.grad(
        pv_perm.sum(), (node_perm, edge_perm), create_graph=True
    )
    hess_diag_node_perm = torch.autograd.grad(
        grad_node_perm.pow(2).sum(), node_perm
    )[0]

    assert (pv_orig - pv_perm).abs().max().item() <= tol
    assert (out_orig.stopping_probs - out_perm.stopping_probs).abs().max().item() <= tol
    assert (out_orig.survival_probs - out_perm.survival_probs).abs().max().item() <= tol
    assert (out_orig.coupon_probs - out_perm.coupon_probs).abs().max().item() <= tol
    assert (out_orig.put_loss - out_perm.put_loss).abs().max().item() <= tol

    grad_node_unperm = grad_node_perm[:, inv_perm, :]
    grad_edge_unperm = grad_edge_perm[:, inv_perm, :, :][:, :, inv_perm, :]
    hess_diag_unperm = hess_diag_node_perm[:, inv_perm, :]

    assert (grad_node_orig - grad_node_unperm).abs().max().item() <= tol
    assert (grad_edge_orig - grad_edge_unperm).abs().max().item() <= tol
    assert (hess_diag_node_orig - hess_diag_unperm).abs().max().item() <= tol * 10.0


def test_G1_7_dummy_padded_asset_and_schedule_exact_zero_invariance() -> None:
    torch.manual_seed(99)
    model = MaskedEQAutocallSurrogate(
        d_max=8,
        m_max=48,
        d_model=128,
        n_egno_layers=3,
        n_sched_layers=2,
        n_heads=4,
    ).eval()

    node_f, edge_f, amask, sched_f, smask, glob_f, _ = _build_synthetic_autocall_batch(
        batch_size=3, seed=4004, dtype=torch.float32
    )
    amask[:, 4:] = 0.0
    smask[:, 12:] = 0.0

    # Set Period 0 as non-callable lock-out and guaranteed coupon
    sched_f[:, 0, AutocallSchedCol.IS_CALLABLE_FLAG] = 0.0
    sched_f[:, 0, AutocallSchedCol.IS_GUARANTEED_CPN] = 1.0

    node_base = node_f.detach().clone().requires_grad_(True)
    edge_base = edge_f.detach().clone().requires_grad_(True)
    sched_base = sched_f.detach().clone().requires_grad_(True)

    out_base = model(node_base, edge_base, amask, sched_base, smask, glob_f)
    pv_base = model.assemble_product_value(out_base, sched_base, smask, glob_f)
    gn_base, ge_base, gs_base = torch.autograd.grad(
        pv_base.sum(), (node_base, edge_base, sched_base)
    )

    assert (out_base.stopping_probs[:, 0].abs().max().item()) == 0.0
    assert (out_base.hazard_rates[:, 0].abs().max().item()) == 0.0
    assert torch.allclose(
        out_base.coupon_probs[:, 0], torch.ones_like(out_base.coupon_probs[:, 0])
    )
    assert (out_base.stopping_probs[:, 12:].abs().max().item()) == 0.0
    assert (out_base.coupon_probs[:, 12:].abs().max().item()) == 0.0

    # Perturb all inactive asset slots (i >= 4), diagonal edges, and inactive schedule slots (m >= 12) by +1e6
    node_pert = node_f.detach().clone()
    node_pert[:, 4:, :] += 1e6
    node_pert[0, 7, :] = float("nan")
    node_pert.requires_grad_(True)

    edge_pert = edge_f.detach().clone()
    edge_pert[:, 4:, :, :] += 1e6
    edge_pert[:, :, 4:, :] -= 1e6
    for i in range(8):
        edge_pert[:, i, i, :] += 1e6
    edge_pert.requires_grad_(True)

    sched_pert = sched_f.detach().clone()
    sched_pert[:, 12:, :] += 1e6
    sched_pert[0, 47, :] = float("nan")
    sched_pert.requires_grad_(True)

    out_pert = model(node_pert, edge_pert, amask, sched_pert, smask, glob_f)
    pv_pert = model.assemble_product_value(out_pert, sched_pert, smask, glob_f)
    gn_pert, ge_pert, gs_pert = torch.autograd.grad(
        pv_pert.sum(), (node_pert, edge_pert, sched_pert)
    )

    assert (pv_base - pv_pert).abs().max().item() == 0.0
    assert (out_base.stopping_probs - out_pert.stopping_probs).abs().max().item() == 0.0
    assert (out_base.survival_probs - out_pert.survival_probs).abs().max().item() == 0.0
    assert (out_base.hazard_rates - out_pert.hazard_rates).abs().max().item() == 0.0
    assert (out_base.coupon_probs - out_pert.coupon_probs).abs().max().item() == 0.0
    assert (out_base.put_loss - out_pert.put_loss).abs().max().item() == 0.0

    assert (gn_base[:, :4, :] - gn_pert[:, :4, :]).abs().max().item() == 0.0
    assert (ge_base[:, :4, :4, :] - ge_pert[:, :4, :4, :]).abs().max().item() == 0.0
    assert (gs_base[:, :12, :] - gs_pert[:, :12, :]).abs().max().item() == 0.0
    assert gn_pert[:, 4:, :].abs().max().item() == 0.0
    assert gs_pert[:, 12:, :].abs().max().item() == 0.0


def test_G4_5_cudnn_rnn_backward_freedom_and_eval_second_order_autograd() -> None:
    src_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "deepvol"
        / "surrogates"
        / "eq_autocall_egno.py"
    )
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    forbidden_rnn_names = {"GRU", "LSTM", "RNN", "GRUCell", "LSTMCell", "RNNCell"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_rnn_names:
            pytest.fail(f"Forbidden RNN construct {node.attr} found in eq_autocall_egno.py")
        if isinstance(node, ast.Name) and node.id in forbidden_rnn_names:
            pytest.fail(f"Forbidden RNN symbol {node.id} found in eq_autocall_egno.py")

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev_str in devices:
        dev = torch.device(dev_str)
        torch.manual_seed(505)
        model = (
            MaskedEQAutocallSurrogate(
                d_max=8,
                m_max=48,
                d_model=128,
                n_egno_layers=3,
                n_sched_layers=2,
                n_heads=4,
            )
            .to(dev)
            .eval()
        )

        for mod in model.modules():
            assert not isinstance(mod, (nn.RNNBase, nn.RNNCellBase))

        node_f, edge_f, amask, sched_f, smask, glob_f, _ = _build_synthetic_autocall_batch(
            batch_size=2, seed=5005, dtype=torch.float32, device=dev
        )
        # Set exact tie on active assets 0 and 1 in row 1 (where d_active >= 2)
        amask[1, :4] = 1.0
        node_f[1, 1, :] = node_f[1, 0, :].clone()
        edge_f[1, 1, 0, :] = edge_f[1, 0, 1, :].clone()
        edge_f[1, 1, 2:, :] = edge_f[1, 0, 2:, :].clone()
        edge_f[1, 2:, 1, :] = edge_f[1, 2:, 0, :].clone()

        node_in = node_f.detach().clone().requires_grad_(True)
        edge_in = edge_f.detach().clone().requires_grad_(True)
        sched_in = sched_f.detach().clone().requires_grad_(True)

        with torch.backends.cudnn.flags(enabled=True):
            out = model(node_in, edge_in, amask, sched_in, smask, glob_f)
            pv = model.assemble_product_value(out, sched_in, smask, glob_f)
            grad_node, grad_edge, grad_sched = torch.autograd.grad(
                pv.sum(),
                (node_in, edge_in, sched_in),
                create_graph=True,
            )
            direction = torch.ones_like(grad_node)
            hvp_node, hvp_edge, hvp_sched = torch.autograd.grad(
                (grad_node * direction).sum()
                + grad_edge.sum()
                + grad_sched.sum(),
                (node_in, edge_in, sched_in),
                retain_graph=False,
            )

        assert torch.isfinite(grad_node).all()
        assert torch.isfinite(grad_edge).all()
        assert torch.isfinite(grad_sched).all()
        assert torch.isfinite(hvp_node).all()
        assert torch.isfinite(hvp_edge).all()
        assert torch.isfinite(hvp_sched).all()
        assert hvp_node.abs().max().item() > 0.0
        assert torch.allclose(
            grad_node[1, 0, :], grad_node[1, 1, :], atol=1e-6, rtol=1e-5
        )
        assert torch.allclose(
            hvp_node[1, 0, :], hvp_node[1, 1, :], atol=1e-5, rtol=1e-4
        )
